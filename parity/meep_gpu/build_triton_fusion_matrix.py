#!/usr/bin/env python3
"""THE TRITON FUSION MATRIX: which corpus seam-instances a fused product serves.

The two 2026-08-19 per-product censuses answered "how many rows would THIS ONE
product admit". Neither answered the aggregate — what fraction of the corpus a
FUSED path can step on Triton — and neither enumerated the (curl arm,
constitutive arm) cells the corpus actually drives, so neither could rank the
missing products by demand. This does both.

WHAT A SEAM-INSTANCE IS, AND WHY THE DENOMINATOR IS NOT 759
-----------------------------------------------------------
The sub-step census counts SLOTS: rows x (step_B, step_D, update_H, update_E,
update_P and the two fold-only fills) — 759 on the 186-row basis those censuses
were cut over; the corpus is 194 rows since 2026-09-03. Fusion does not serve a slot; it
serves a SEAM — the gap between a curl and the constitutive that closes it, with
the boundary passes in between. The driver's timestep (driver.py:3280-3306) has
exactly three of them, and one is conditional:

  B->H   step_B(:3282) -> magnetic inject(:3284) -> fill_symmetry_bc_B(:3285)
         -> zero_metal_B(:3286) -> fill_folded_far_ghosts_B(:3287) -> update_H(:3289)
  D->E   step_D(:3293) -> electric inject(:3296/:3299) -> fill_symmetry_bc_D(:3300)
         -> zero_metal_D(:3301) -> fill_folded_far_ghosts_D(:3302) -> update_E(:3304)
  E->P   update_E(:3304) -> update_P(:3306)

B->H and D->E exist on every row: both curls and both constitutives are called
every timestep on every configuration, whatever any of them does. E->P exists
only where a susceptibility is registered — ``update_P`` is not called into on a
row with no polarization state — so it is counted only on those rows.

    DENOMINATOR = rows (B->H) + rows (D->E) + n_polarizations>0 (E->P)
                = 194 + 194 + 15 = 403 on the 2026-09-03 basis
                  (186 + 186 + 15 = 387 on every board before it)

The E->P count is measured from the record rather than assumed, and printed.

WHAT IS COUNTED AS "SERVED"
---------------------------
Three questions are asked per seam-instance and they are NOT the same question:

  (a) does a fused PRODUCT exist for this cell — i.e. for the (curl arm,
      constitutive arm) pair the composition actually selects on this row?
  (b) does that product's SHIPPED PREDICATE admit this row? (`folded_fused_
      magnetic_pair` exists, and refuses a magnetic source and a folded PERIODIC
      axis; "a product exists" and "this row is served" are different facts.)
  (c) can ``launch.plan_step`` COMPOSE it? Five of the fourteen products are
      reachable — ``fused_pair_B``/``fused_pair_D`` (built inside ``launch.py``),
      ``dispersive_fused_pair``, and the two FOLDED pairs, which the folded branch
      installs through ``_install_folded_fused_pairs`` (launch.py:3033). The other
      nine live in modules ``launch.py`` never imports, so (b) can be true where
      (c) is false. THIS COLUMN IS MEASURED, not declared: until 2026-08-29 it read
      a hardcoded flag that still said the two folded pairs were unreachable, two
      days after they were routed, and mis-scored 56 seam-instances.
      :func:`assert_the_wiring_column_is_measured` re-derives it from ``launch.py``'s
      parse tree on every cut and raises on a disagreement, in both directions.

The headline number is (b), which is what the brief asked for. (a) and (c) are
carried beside it because reporting only (b) would overstate what runs today and
reporting only (c) would understate what has been built.

WHERE THE NUMBERS COME FROM, AND THE FRESHNESS CHECK
----------------------------------------------------
NOT A RE-LIFT. Two standing records are read:

  * ``predicate_coverage_triton_2026-09-03_extended`` — the TRITON battery, and
    the DEFAULT (see :data:`CENSUS` for every move of the default since the 08-16
    record and the trap behind each; the first one built a whole board off the
    08-16 battery without saying so, and answered 119 served where the 08-28 record
    answered 170). 194 lifted rows since 2026-09-03 — the 186 every earlier cut held
    plus the eight example scripts the 2026-08-09 corpus campaign accepted in its
    _blocked/_gdsii/_sigma legs — every live Triton predicate called on the lifted engine
    object, plus ``plan_step``'s own per-slot admitted-arm record and the three
    fused SEAM predicates (``fused_pair_B``, ``fused_pair_D``,
    ``dispersive_fused_pair_D``) called with the real source list. Its ancestor
    ``predicate_coverage_2026-08-16_wired_convention`` is still selectable by name.
    That one is the ``_convention`` cut rather than the plain ``_wired`` one, for
    the reason the
    2026-08-19 complex census measured: ``_wired`` passed a probe artifact
    carrying no resolved subnormal policy, so clause 13 refused every complex
    family on all 186 rows. Verified here rather than inherited: this script
    diffs the two records and prints every predicate whose count differs — all
    of them complex-family, none of them anything else.
  * ``cuda_predicate_coverage_2026-08-20_final`` — the freshest complete corpus
    record (denominator 759 MATCH, "recorded rows NOT recovered: 0"). Its
    predicates are the CUDA battery's and are NOT read. Its ``configuration``
    blocks are, as a FRESHNESS CHECK: the rows the two records share and their
    configurations are backend-independent (the eight rows added on 2026-09-03 have
    no twin in the 2026-08-20 record — ``row_sets_identical`` reads False and says
    so, and they are compared on nothing), so if the census's configurations still agree with the
    2026-08-20 ones key-for-key, the four rounds of sub-step movement since did
    not move the corpus this derivation prices against. The comparison is printed.

THE STALENESS AUDIT IS MEASURED, NOT ASSERTED. Every predicate function whose
verdict this derivation reads is AST-hashed (docstrings stripped) at the census's
commit and in the worktree, and each is reported SAME or CHANGED. A CHANGED
function is not silently trusted: the one that matters — the complex expansion
licence, rewritten between the two — is RE-MEASURED here by handing today's
``complex_fields`` the exact probe artifact the census ran under.

**COVERAGE IS READ OFF ``covered_modulo_backend``.** The census lifted on NumPy,
so every Triton predicate carries "array module is 'numpy', not cupy" and the raw
``covered`` column is 0 on every row for every family. That is a statement
about the harness, not about any corpus row; the residual column is what the
census computed for exactly this purpose.

PROMOTED OUT OF ``results/`` ON 2026-08-28, and the promotion is the point
=========================================================================

This file is ``results/fusion_matrix_triton_2026-08-21_foldedcomplex/fusion_matrix.py``
moved to the canonical directory beside the gates. Twenty divergent copies of that
name live under ``results/`` -- one per board cut -- and picking the newest by
mtime lands on ``..._2026-08-21d/``, whose own ``REFUSED_TO_BUILD.md`` records that
its script and its JSON came from two different rounds. The Metal board was promoted
for exactly this reason (``build_fusion_matrix.py``:110-116) after a cut forked off a
pre-carry ancestor and silently re-applied a retired clause. There is now one Triton
board script, it takes its census and its output directory as arguments, and a cut is
a directory of RESULTS rather than another copy of the code.

Four things changed in the move and nothing else did: the roots are resolved BY NAME,
the census is selectable, the output directory is an argument, and the staleness audit
reads the census's own tree snapshot instead of a hard-coded commit (see
:func:`staleness_audit`). Every product entry, clause, predicate key, denominator and
floor is byte-for-byte the 08-21 cut's.

Usage (from the repository root)::

    PYTHONPATH=. python -u \
      parity/meep_gpu/build_triton_fusion_matrix.py \
      --census predicate_coverage_triton_2026-08-28 \
      --out parity/meep_gpu/results/fusion_matrix_triton_2026-08-28
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent


#: The package root, found BY NAME rather than by counting ``.parents[N]``. This
#: script has now moved once (out of a results tranche, up two levels), and a
#: depth-based root follows a move silently: the promoted census driver's own
#: comment records what that costs -- every child dies on import while the parent
#: still writes a full-length, right-shaped record of unmeasured rows and exits 0.
def _find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(
        f"cannot locate the repository root above {start}: no ancestor holds "
        "both parity/meep_gpu and meep_gpu. Refusing to guess.")


API = _find_api_root(HERE)
#: The results tree this cut reads its censuses from. Named, not derived from
#: ``HERE.parent``, because this script is CANONICAL and lives beside the gates.
RESULTS = HERE / "results"
#: The version-control root the ``git`` calls run in: the nearest ancestor of
#: ``API`` (``API`` included) that holds ``.git``, else ``API``.
REPO = next((candidate for candidate in (API, *API.parents)
             if (candidate / ".git").exists()), API)

#: The Triton battery's census. Its predicates and its ``plan_step`` record are what
#: every verdict below is read off. Selectable so a fresh cut does not need a fresh
#: copy of this file.
#:
#: THE DEFAULT MOVED 2026-08-29, and the reason is a trap that was paid for rather
#: than foreseen. It used to be ``predicate_coverage_2026-08-16_wired_convention`` —
#: the record the 08-21 cuts were taken over — so an argument-free run silently
#: BUILT A COMPLETE BOARD off a two-week-old battery: 119 served and 55 -> 17 served
#: in dispatch against the 08-28 battery's 170 and 55, on the same shipped code, with
#: nothing in the output saying which battery it was. A default that quietly answers
#: a different question is worse than one that refuses, and worse still now that the
#: board publishes a DISPATCH number a reader will act on. The default is now the
#: battery the current boards are cut over; the 08-16 record is still selectable by
#: name, which is what reproducing an old cut needs.
#: MOVED 2026-09-02 to the STENCIL-WELD census. The 08-28 record predates the
#: battery extension and carries no seam verdict for 31 of the predicates a current
#: cut reads, so the board refused over it rather than report those products at
#: ZERO without measuring them — the refusal working as designed, not a defect. The
#: chain is foldedbeta -> 09-02_residue (four complex/folded-beta columns) ->
#: 09-02_stencil (the two off-diagonal scratch-weld columns), each round adding the
#: columns its welds needed; _stencil is the head and the one the current boards are
#: cut over. Older records stay selectable by name for reproducing an old cut.
#:
#: HEAD MOVED TO ``_2026-09-02_unified`` ON 2026-09-02, and the reason is a MEASURED
#: reporting defect rather than a new round of columns. Every census up to ``_stencil``
#: was cut under ``complex_expansion_beta_extension_2026-09-01/probe_keep/probe.json``,
#: which classifies FIVE multiply patterns. The shipped arm "complex folded
#: off-diagonal" (``complex_offdiag_update_e.py``'s folded, active-PML tail — the file
#: ships TWO tails) binds a SIXTH, ``c8_mul_c8_parity_coefficient_left``, so on every
#: folded-complex off-diagonal row that arm refused with two clauses that are both
#: about the ARTIFACT and neither about the configuration.
#:
#: THE BOARD THEN REPORTED IT AS ABSENT RATHER THAN AS UNLICENSED, because
#: ``admitted()`` reads the census slot table verbatim and that table carries ONE
#: artifact verdict per arm with no alternative-artifact variant — so three
#: seam-instances read as "a slot has NO admitting arm", i.e. as a MISSING KERNEL, for
#: a kernel that ships, is registered at update_E, and whose device gate is released
#: (``fingerprints.json :: triton_complex_offdiag_device_gate``). Acting on that report
#: would have meant building a duplicate arm and taking the slot from one admitter to
#: two, which ``_select_slot`` empties — the phantom gap would have created a real one.
#:
#: ``unified_expansion_2026-08-27/keep/gate.json`` classifies SEVEN patterns and is a
#: STRICT SUPERSET: measured over 72 (case, slot) pairs across 15 configurations,
#: 0 admitters LOST and 11 GAINED, identical classifications on the five shared
#: patterns, same backend/device/CuPy version and the same 'keep' policy with
#: ftz_removed=1 (``probe_triton_folded_complex_offdiag_admission.py``, leg 4).
#: Cutting over it moved SERVED not at all (341 both ways) and moved one instance out
#: of "no admitting arm" into "STRUCTURALLY UNFUSABLE" — where the CUDA board had it
#: all along. The swap buys honesty, not coverage, which is why it is recorded here
#: rather than announced as a gain.
#:
#: HEAD MOVED TO ``_2026-09-03_extended`` ON 2026-09-03, and for the first time the
#: CORPUS moved it rather than a column or an artifact: the basis grew from 186 to 194
#: rows. Eight example scripts the 2026-08-09 corpus campaign accepted in its
#: _blocked/_gdsii/_sigma legs had gone unpriced on every board because the census
#: driver named only the stock leg's lift record; the driver now takes a script from
#: the first campaign record that accepts it and runs each row under the interpreter
#: its record names. On the new basis 403 PRICED / 384 FUSABLE / 359 ATTAINABLE (387 /
#: 368 / 344 before), and the board cut over it
#: (``fusion_matrix_triton_2026-09-03_complete``) reads 354/403 served against
#: ``_unified``'s 341/387, with 0 status changes on the 186 shared labels: 13 of the
#: 16 new instances served, the 3 unserved being coupler.py's D->E seam
#: (source_seam_unbracketable) and dipole_in_vacuum_cyl_off_axis.py's two seams
#: (missing_half: no arm admits complex64 cylindrical storage with m != 0). MOVES
#: TOGETHER WITH :data:`METAL_CENSUS`: ``build_matrix`` refuses unless the liveness
#: record holds the same rows, and the Metal record on this basis is
#: ``metal_coverage_2026-09-03_complete`` — not ``metal_coverage_2026-09-03_extended``,
#: whose five gdsii/sigma rows were evaluated where torch was absent and stamped
#: measured anyway; this board joined that record once
#: (``fusion_matrix_triton_2026-09-03_extended``, superseded the same day).
#:
#: HEAD MOVED TO ``_2026-09-04_cylm0`` ON 2026-09-06 — the census EVERY Triton board
#: from ``fusion_matrix_triton_2026-09-04_m0`` on was cut over while this default still
#: named ``_2026-09-03_extended``, so for two days an argument-free run answered a
#: different question from the standing board. What separates the two records is the
#: SUBJECT, measured by diffing them row by row: identical 194 rows, identical
#: configurations, identical ``plan_step.selected``, and exactly ONE row whose
#: predicate columns differ — ``examples:dipole_in_vacuum_cyl_off_axis.py``, where
#: the four ``cylindrical_complex_*`` predicates and the
#: ``cylindrical_fused_electric_pair_D`` seam column read False on ``_extended`` and
#: True on ``_cylm0``. That is the m = 0 complex cylindrical landing of 2026-09-04,
#: re-gated on the GPU host as ``results/triton_regate_2026-09-04_cylm0/``; ``_extended``
#: predates it and prices that row's two curl->constitutive instances as a missing
#: half where the shipped tree admits them. An argument-free cut over ``_cylm0``
#: reproduces the standing board with zero instances changed
#: (``fusion_matrix_triton_2026-09-06_consolidated`` against ``_2026-09-06_hd``).
#: MOVES TOGETHER WITH :data:`METAL_CENSUS`, as before.
#: MOVED 2026-09-11 to the realarms cut. The `_cylm0` default predated the battery
#: extension that scores `folded_offdiag_fused_ade_chain`, so an argument-free build
#: refused BY NAME rather than reporting that product at zero without measuring it --
#: while the board actually cut that morning had been given the newer census on the
#: command line. A default that no cut uses is a default that goes stale silently.
#: MOVED 2026-09-23 to ``predicate_coverage_triton_2026-09-23_sparse_licensed``: the
#: sparse-transport pass moved `stepping.py`, seven of the 66 subjects, so no earlier
#: census hashes to this tree. Cut LICENSED, which is two declarations: ``--run-policy
#: keep`` (the licence re-check below reads the policy the census declared and refuses
#: ``None`` by name) and ``--probe-artifact`` naming this round's own
#: ``triton_fleet_2026-09-23_sparse/unified_expansion/gate.json`` (the census default
#: is the artifact the 08-16 round rejected, under which the cylindrical-complex arms
#: read unlicensed on every row -- three cuts that day reproduced exactly that before
#: the flag was found). Over the complete corpus: 194 rows, 179 measured against the 170
#: the 09-03 census could reach before the sibling fixtures and `parameterized` were
#: restored.
#: THE DEFAULT MOVED AGAIN 2026-09-24, to ``predicate_coverage_triton_2026-09-24_final2``:
#: the NULL_SIDES citation fix moved `triton_kernels/no_pml_constitutive.py`, a subject, and
#: this builder does not check its census's subject manifest, so the census was re-cut
#: rather than relied on -- the same two licence flags (`--run-policy keep`, `--probe-artifact`
#: the 09-23 fleet's unified gate), the same profile (227 licensed rows, the 16 unmeasurable
#: rows None in both), 0 of 77 subjects moved against the final tree.
#: MOVED 2026-09-25 to ``predicate_coverage_triton_2026-09-25_roundb_licensed``, the census
#: ``fusion_matrix_triton_2026-09-25_roundb`` was cut on. That cut named it with ``--census``
#: (``round_2026-09-25_roundb_drivers/triton_all.sh``), so the default kept naming ``_final2``
#: -- the 09-11 trap again. 52 of the 77 subjects moved between the two cuts (51 under
#: ``triton_kernels`` plus ``stepping.py``); ``_final2`` hashes to this tree on 25 of 77 in
#: each leg, this census on 77 of 77 in each of its three legs (manifest ``fd675ed0``, which
#: the live tree recomputes). An argument-free cut on ``_final2`` reproduced the board with 0
#: of 403 seam instances differing outside ``halves_read_from``; what it got wrong was the
#: provenance -- its ``census`` and ``probe_artifact`` cited ``_final2`` and the 09-23 fleet
#: (``round_2026-09-26_close_audit``, S4). Cut LICENSED, the same two flags: ``--run-policy
#: keep`` and ``--probe-artifact`` the Round B fleet's
#: ``triton_fleet_2026-09-25_roundb/unified_expansion/gate.json``. This builder still reads no
#: ``subject_digests_*``, so the next subject edit leaves this default stale with no refusal.
CENSUS = RESULTS / (os.environ.get("MEEP_GPU_TRITON_CENSUS")
                    or "predicate_coverage_triton_2026-09-25_roundb_licensed")
#: The same run under an unlicensed probe. Read only to print the delta, so the
#: choice of record above is a measurement rather than a citation.
CENSUS_UNLICENSED = RESULTS / "predicate_coverage_2026-08-16_wired"
#: The freshest complete corpus record. Configurations only — see the docstring.
FRESH = RESULTS / "cuda_predicate_coverage_2026-08-20_final"

#: The commit the Triton census ran at. ``31c9e18`` (the round that added the five
#: fused products) is the only ``triton_kernels`` commit after it.
CENSUS_COMMIT = "47d6658"

#: ``symmetry.MIRROR_SOURCE_INDEX`` -- the stored index the near fill images.
#: READ, NOT SPELLED, and the two copies are required to agree rather than assumed
#: to. This was the literal ``2`` until 2026-08-28; a board that spells the number
#: keeps answering the old threshold after the engine moves and then ADMITS folds the
#: shipped predicate refuses -- over-counting, the one direction that reads as
#: success. The Metal board reads it the same way (build_fusion_matrix.py:144-166).
if str(API) not in sys.path:
    sys.path.insert(0, str(API))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# SERVED IN DISPATCH, measured rather than asserted in a caveat string. See
# ``dispatch_reachability`` for why a sentence was the wrong shape for that answer.
import dispatch_reachability as _reach  # noqa: E402

# THE RECONCILED BUCKET NAMES AND THE THREE TIERS, shared with the Metal and CUDA
# boards (release decision R1/R2, 2026-09-02). One name list in one file is what stops a
# fourth vocabulary growing back; the module also carries the floors that make the
# decomposition and the served total REFUSABLE rather than merely printed.
import dispatch_agreement  # FLOOR 8; see its docstring for why it is not in fusion_taxonomy
import fusion_taxonomy  # noqa: E402
# THE FOURTH SEAM, update_H -> step_D (2026-09-04): one pricing rule for the three
# boards, off the probe artifact and the driver's own text. See h_to_d_seam.py.
import h_to_d_seam  # noqa: E402

#: seam -> (curl slot, constitutive slot). Spelled once, so the missing-half bucket
#: can NAME which of the two bodies is absent rather than saying "a slot".
SEAM_SLOT_NAMES: Dict[str, Tuple[str, str]] = {
    "B->H": ("step_B", "update_H"),
    "D->E": ("step_D", "update_E"),
    "E->P": ("update_E", "update_P"),
}

try:
    from meep_gpu.stepping import MIRROR_SOURCE_INDEX as NEAR_SOURCE_INDEX
    from meep_gpu.triton_kernels.symmetry import (
        MIRROR_SOURCE_INDEX as _SYMMETRY_NEAR_SOURCE_INDEX)
except Exception:  # pragma: no cover - the row the near fill images is not guessable
    raise SystemExit("meep_gpu.stepping.MIRROR_SOURCE_INDEX is not importable; the "
                     "row the near fill images -- and so which folds the deposit "
                     "repair refuses -- cannot be read and this script refuses to "
                     "assume it")
if NEAR_SOURCE_INDEX != _SYMMETRY_NEAR_SOURCE_INDEX:
    raise SystemExit(
        f"stepping.MIRROR_SOURCE_INDEX is {NEAR_SOURCE_INDEX} and "
        f"triton_kernels.symmetry.MIRROR_SOURCE_INDEX is "
        f"{_SYMMETRY_NEAR_SOURCE_INDEX}; the array path's near fill and the kernels' "
        f"carry image different rows, so no single board clause is right for both")

#: The expansion probe artifact the census declared, re-read here for the EXTENDED
#: pattern set the folded complex weld binds. Named rather than inferred so the
#: record says which artifact answered.
#: THE CENSUS'S OWN ARTIFACT, re-cut 2026-09-01 as the BETA EXTENSION of the
#: 2026-08-16 standing artifact: the base four patterns (character-for-character
#: identical tables — the keep control) PLUS `c8_mul_c8_imaginary_coefficient_left`.
#: It still does NOT classify the PARITY pattern, so its answer for the folded
#: complex weld's extended set is recorded below rather than skipped.
CENSUS_PROBE_ARTIFACT = (RESULTS / "complex_expansion_beta_extension_2026-09-01"
                         / "probe_keep" / "probe.json")

#: THE ARTIFACT THE RELEASED DEVICE GATE CONSUMED, and the one the extended pattern
#: set is measured against. Same host, same 'keep' policy, and it carries
#: ``c8_mul_c8_parity_coefficient_left`` — which the census artifact does not,
#: because that pattern arrived with the tranche that added the parity multiply.
#:
#: WHY NOT THE CENSUS'S. Its refusal would be about the RECORD'S DATE, not about the
#: platform or the corpus: "pattern 'c8_mul_c8_parity_coefficient_left' is not
#: classified". Pricing a product at zero for that would be the same error as
#: pricing it at zero for a missing column. Both answers are recorded below so a
#: reader can see which artifact answered and what the other said.
PROBE_ARTIFACT = (RESULTS / "expansion_probe_2026-08-17"
                  / "expansion_probe_keep.json")


def log(message: str = "") -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rows(census: Path) -> List[dict]:
    """The measured rows (194 since 2026-09-03; 186 before), with the parameterised
    leg's matched rows substituted."""
    record = load(census / "examples.jsonl") + load(census / "tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load(census / "tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return [r for r in record if r.get("measured")]


def label(row: dict) -> str:
    return f"{row['leg']}:{row['row']}"


def covered(row: dict, key: str) -> bool:
    """A sub-step predicate's verdict with the NumPy-host clause discounted."""
    entry = row["predicates"].get(key, {})
    return bool(entry.get("covered_modulo_backend", entry.get("covered")))


def seam_covered(row: dict, key: str) -> bool:
    """A SEAM predicate's verdict — these are the shipped fused-pair predicates."""
    entry = (row.get("seam") or {}).get(key, {})
    return bool(entry.get("covered_modulo_backend", entry.get("covered")))


def admitted(row: dict, slot: str) -> Tuple[str, ...]:
    """The arms whose predicates admitted this slot, as ``_select_slot`` returned them."""
    block = (row.get("plan_step") or {}).get("slots", {}).get(slot)
    return tuple(block["admitted"]) if block else ()


def arm(row: dict, slot: str) -> Optional[str]:
    """The arm the composition would select: exactly one admitter, else none.

    NOT ``plan_step``'s ``selected`` map. The census host has no Triton, so every
    builder raised an import error and ``_select_slot`` (launch.py:1988-1993) left
    every slot unselected with a named reason — ``selected`` is empty on every row
    but the 14 whose no-absorber slots resolve to the null arm (measured on the
    2026-09-02 and 2026-09-03 censuses alike) and measures the harness. ``admitted``
    is the arm TABLE's verdict, which
    is builder-independent, and the one-admitter rule below is ``_select_slot``'s
    own (launch.py:1983-1996): two admitters leave the slot on the array path.
    """
    labels = admitted(row, slot)
    return labels[0] if len(labels) == 1 else None


# ---------------------------------------------------------------------------
# The products, and the cell each one occupies
# ---------------------------------------------------------------------------
#
# A cell is a (curl arm, constitutive arm) pair, spelled with ``launch.py``'s own
# arm labels. The three wired products' cells are read STRAIGHT OFF
# ``launch.FUSED_PAIR_ARMS`` (launch.py:2077-2081); the three unwired ones have no
# such entry, so their cells are derived from the predicates they conjoin, each
# cited.

PRODUCTS: Dict[str, Dict[str, Any]] = {
    "fused_pair_B": {
        "seam": "B->H",
        "cell": ("PML", "ordinary"),                  # launch.py:2078
        "module": "coverage.py:431 fused_pair_coverage / launch.py:1684 plan_fused_pair",
        "wired": True,
    },
    "fused_pair_D": {
        "seam": "D->E",
        "cell": ("PML", "ordinary"),                  # launch.py:2079
        "module": "coverage.py:431 fused_pair_coverage / launch.py:1684 plan_fused_pair",
        "wired": True,
    },
    "dispersive_fused_pair": {
        "seam": "D->E",
        "cell": ("PML", "dispersive"),                # launch.py:2080
        "module": "dispersive_fused_pair.py",
        "wired": True,
    },
    "folded_fused_magnetic_pair": {
        "seam": "B->H",
        # folded_fused_magnetic_pair.py:561 conjoins folded_composition_curl_coverage
        # (the "folded PML" arm, launch.py:2470) and folded_constitutive_coverage('H')
        # (the "folded" arm, launch.py:2643).
        "cell": ("folded PML", "folded"),
        "module": "folded_fused_magnetic_pair.py:549",
        # WIRED 2026-08-27, corrected here 2026-08-29. ``launch.py`` imports the
        # module (SUPPORT_MODULES) and ``plan_step``'s folded branch installs it
        # through ``_install_folded_fused_pairs`` (launch.py:3033). The flag is no
        # longer trusted: :func:`assert_the_wiring_column_is_measured` re-derives it
        # from ``launch.py``'s parse tree on every cut and RAISES on a disagreement,
        # which is what would have caught this one two days ago.
        "wired": True,
    },
    "complex_fused_magnetic_pair": {
        "seam": "B->H",
        # complex_fused_magnetic_pair.py:600 conjoins complex_pml_curl_coverage
        # (the "complex PML" arm) and complex_constitutive_coverage('H') (the
        # "complex" arm).
        "cell": ("complex PML", "complex"),
        "module": "complex_fused_magnetic_pair.py:581",
        "wired": True,
    },
    "folded_fused_pair": {
        "seam": "D->E",
        # folded_fused_pair.py conjoins folded_composition_curl_coverage('step_D')
        # (the "folded PML" arm, launch.py:2470) and folded_constitutive_coverage('E')
        # (the "folded" arm, launch.py:2643). RELEASED 2026-08-20 on the GPU host:
        # results/triton_folded_fused_pair_2026-08-20/gate.json.
        "cell": ("folded PML", "folded"),
        "module": "folded_fused_pair.py",
        # WIRED 2026-08-27, corrected here 2026-08-29 — see the twin above.
        "wired": True,
    },
    # ------------------------------------------------------------------ NEW
    # THE 2026-09-02 SCRATCH-OUTPUT OFF-DIAGONAL WELDS. Both occupy cells this
    # board scored, from 2026-08-20 until this round, as
    # "reachable; STRUCTURALLY UNFUSABLE — the constitutive arm is a STENCIL over
    # the curl arm's in-place output".
    #
    # THAT VERDICT WAS TRUE OF WHAT IT MEASURED and is superseded rather than
    # contradicted. `results/triton_fused_offdiag_electric_2026-08-20` measured a
    # ONE-ORDINARY-LAUNCH weld of the SHIPPED IN-PLACE halves: 42 of 60 subject
    # cases divergent at up to 6.58e-02, schedule-dependently. These two products
    # remove both of that weld's premises — the curl half writes LAUNCH-LOCAL
    # SCRATCH and never in place, the constitutive half RE-DERIVES every foreign
    # stencil tap from pre-launch state, and the plan rotates the D/fu references
    # after the launch — so nothing written is ever read and there is no
    # cross-program hazard for a block schedule to expose.
    # THE H->D SEAM'S FIRST PRODUCT, 2026-09-06, and the first entry on this board
    # whose seam is the fourth one. `fused_hd_pair.py:fused_hd_pair_coverage` conjoins
    # `coverage.constitutive_coverage(fields, pml, 'H')` (the "ordinary" arm,
    # launch.py's update_H slot) and `coverage.pml_curl_coverage(fields, pml,
    # 'step_D')` (the "PML" arm), in THAT order, because that is the driver's.
    #
    # `wired: False`, and the reason is a FILE BOUNDARY rather than the arithmetic or
    # the slot protocol. Routing it through `_install_certified_fused_products` costs
    # a composer label, and on this backend a composer label must also appear in
    # `fastpath.FUSED_ARM_CONSTITUENTS` and `PENDING_DEVICE_GATE_ARMS`; both are in
    # `meep_gpu/fastpath.py`, which the round that built this product does not own.
    # The product additionally declares `INSTALLABLE = False` for a measured
    # arbitration verdict of its own, so routing it would change no plan on any row.
    "fused_hd_pair": {
        "seam": "H->D",
        "cell": ("ordinary", "PML"),
        "module": "fused_hd_pair.py",
        "wired": False,
    },
    # THE CYLINDRICAL H->D SEAM, 2026-09-07 -- the two cells the 2026-09-05 board
    # filed `buildable_not_built` and the design pass before it had refused outright.
    # Both products are TWO launches and one array-module scan: launch 1 (the
    # `update_H` slot) is the certified constitutive at the program's own cell into
    # scratch plus the pre-scan increment of `cylindrical_rderiv_prefix` with `Hy`
    # RECOMPUTED at the backward radial neighbour; `xp.cumsum` runs untouched on the
    # array path's own pooled slot (a column-serial device scan is measured NOT to be
    # `cupy.cumsum`, so the oracle stays where it is); launch 2 is the certified
    # cylindrical curl, verbatim. `cylindrical_real_fused_hd_pair_coverage` conjoins
    # `cylindrical_triton.cylindrical_constitutive_coverage(fields, pml, 'H')` (the
    # "cylindrical" arm) and `cylindrical_curl_coverage` (the "cylindrical PML" arm);
    # `cylindrical_fused_hd_pair_coverage` conjoins the complex family's two
    # (`cylindrical_complex_constitutive_coverage('H')`, "cylindrical complex", and
    # `cylindrical_complex_curl_coverage('step_D')`, "cylindrical complex PML"), in
    # the driver's order. `wired: True`: routed through the composer, which refuses
    # each by its own `INSTALLABLE = False` on every row and records the reason.
    "cylindrical_real_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("cylindrical", "cylindrical PML"),
        "module": "cylindrical_real_fused_hd_pair.py",
        "wired": True,
    },
    "cylindrical_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("cylindrical complex", "cylindrical complex PML"),
        "module": "cylindrical_fused_hd_pair.py",
        "wired": True,
    },
    # THE 2026-09-07 H->D TAIL -- five products over the NINE cells that were this
    # backend's whole remaining H->D ceiling gap once the folded (75 instances) and
    # complex (17) cells were taken. Fifteen seam-instances, every one filed
    # `buildable_not_built` on `results/fusion_matrix_triton_2026-09-07_cyl`, every
    # one with `withdraw_in_seam: false`.
    #
    # THE SHAPE IS `fused_hd_pair`'s ON ALL OF THEM: the certified `update_H`
    # constitutive computed into launch-local WRITE-ONLY scratch at the thread's own
    # cell AND at every backward neighbour the curl taps, `D`/`fu_D` stepped in
    # place, and the H / f_w_H references rotated onto the scratch afterwards -- so
    # H, f_w_H and B are const for the whole dispatch and no thread observes another
    # thread's store.
    #
    # A CELL SERVED BY A SECOND EMITTED KERNEL OR A WIDER ADMISSION IS SCORED UNDER
    # ITS OWN NAME, the way `no_pml_fused_electric_pair_lossless` and
    # `folded_complex_fused_magnetic_pair_offdiag` already are -- so "which product
    # serves this cell" stays answerable from the board.
    #
    # `wired: False` on all of them, for `fused_hd_pair`'s reason exactly: routing
    # costs a composer label that must also live in `meep_gpu/fastpath.py`.
    "conductive_bfast_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("ordinary", "conductive PML"),
        "module": "conductive_bfast_fused_hd_pair.py",
        "wired": False,
    },
    "conductive_bfast_fused_hd_pair_bfast": {
        "seam": "H->D",
        "cell": ("BFAST run", "BFAST PML"),
        # The SECOND emitted kernel of the same module; `module` is the path the
        # board's RELEASE_BINDING resolves, so it is the file, not a description.
        "module": "conductive_bfast_fused_hd_pair.py",
        "wired": False,
    },
    "beta_real_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("real beta run", "real beta PML"),
        "module": "beta_real_fused_hd_pair.py",
        "wired": False,
    },
    "beta_real_fused_hd_pair_folded": {
        "seam": "H->D",
        "cell": ("folded beta run", "folded real beta PML"),
        # The second emitted kernel of the same module.
        "module": "beta_real_fused_hd_pair.py",
        "wired": False,
    },
    "folded_complex_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("folded complex", "folded complex PML"),
        "module": "folded_complex_fused_hd_pair.py",
        "wired": False,
    },
    "folded_complex_fused_hd_pair_offdiag": {
        "seam": "H->D",
        # THE SAME LAUNCH, not a second kernel: the shipped off-diagonal builders
        # return K1's own plan class around K1's own kernel and change only the
        # admission.
        "cell": ("folded complex off-diagonal", "folded complex off-diagonal PML"),
        # The SAME launch under a wider admission, not a second kernel.
        "module": "folded_complex_fused_hd_pair.py",
        "wired": False,
    },
    "beta_complex_fused_hd_pair": {
        "seam": "H->D",
        # THE update_H ARM IS `folded complex`, not "folded complex beta": the
        # shipped constitutive predicate carries no beta clause, and
        # CERTIFIED_FUSED_PAIR_ARMS already pairs it that way for this cell's B->H
        # and D->E siblings.
        "cell": ("folded complex", "folded complex beta PML"),
        "module": "beta_complex_fused_hd_pair.py",
        "wired": False,
    },
    "beta_complex_fused_hd_pair_unfolded": {
        "seam": "H->D",
        "cell": ("complex beta run", "complex beta PML"),
        # The second emitted kernel of the same module.
        "module": "beta_complex_fused_hd_pair.py",
        "wired": False,
    },
    # ADMISSION ONLY, NO NEW KERNEL. `nonlinear_fused_hd_pair` adds one predicate,
    # one plan builder and one checkable identity; the kernel a covered configuration
    # launches is `fused_hd_pair.fused_constitutive_curl_H_to_D`, character for
    # character. What its gate certifies is that the SHIPPED weld, driven on a
    # chi2/chi3 configuration, is bit-identical to the array path -- a measurement
    # the released `fused_hd_pair` artifact does not contain, because its own
    # predicate refuses every nonlinear run.
    "nonlinear_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("nonlinear run", "nonlinear run PML"),
        # ADMISSION ONLY: the kernel a covered run launches is fused_hd_pair's.
        "module": "nonlinear_fused_hd_pair.py",
        "wired": False,
    },
    # The Cartesian COMPLEX H->D cell, 17 seam-instances and the second largest
    # unbuilt one on this seam before it was built. `complex_fused_hd_pair_coverage`
    # conjoins `complex_fields.complex_constitutive_coverage(fields, pml, 'H')` (the
    # "complex" arm) and `complex_pml_curl_coverage(fields, pml, 'step_D')` (the
    # "complex PML" arm), in the driver's order. `wired: True`: routed through the
    # composer, which refuses it by its own `INSTALLABLE = False` on every row.
    "complex_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("complex", "complex PML"),
        "module": "complex_fused_hd_pair.py",
        "wired": True,
    },
    # THE FOLDED H->D CELL, 2026-09-07 -- the largest cell this board carried
    # unbuilt: 78 seam-instances, 75 `buildable_not_built` and 3 `withdraw_seam`.
    # ONE launch: the certified folded `update_H` (which on this backend IS
    # `kernels.constitutive_step` over the folded stored extent, so the weld imports
    # `fused_hd_pair._h_cell` rather than copying it) computed into write-only
    # scratch, the curl's own cell taken from registers, all six foreign taps
    # RECOMPUTED from pre-launch state, `D`/`fu_D` stepped in place, and the six
    # H/f_w_H references rotated afterwards. The curl half is
    # `symmetry.pml_curl_step_folded`'s own body with exactly the nine generated
    # redirects, so the fold's three deltas -- the inverted ghost branch, the cell-0
    # mask widened to `!= PERIODIC`, and the MIRROR_PERIODIC top-plane block -- are
    # the emitter's own text and not a transcription.
    # `folded_fused_hd_pair_coverage` conjoins
    # `symmetry.folded_constitutive_coverage(fields, pml, 'H')` (the "folded" arm) and
    # `symmetry.folded_composition_curl_coverage(fields, pml, 'step_D')` (the
    # "folded PML" arm), in the driver's order, plus the seam's withdraw clause.
    # `wired: False` for the plain product's reason: the label lives in
    # `meep_gpu/fastpath.py`, which the round that built this does not own.
    "folded_fused_hd_pair": {
        "seam": "H->D",
        "cell": ("folded", "folded PML"),
        "module": "folded_fused_hd_pair.py",
        "wired": False,
    },
    "offdiag_fused_electric_pair": {
        "seam": "D->E",
        # offdiag_fused_electric_pair.py conjoins coverage.pml_curl_coverage on
        # 'step_D' (the "PML" arm, launch.py:2078) and
        # offdiag_update_e.offdiag_constitutive_coverage (the "off-diagonal" arm).
        "cell": ("PML", "off-diagonal"),
        "module": "offdiag_fused_electric_pair.py",
        # WIRED 2026-09-17. The file boundary the `wired: False` note above
        # describes was crossed by this round: `launch.py` imports this module
        # (its import list, its cell map and its routing table all name it) and
        # its composer label sits in `fastpath.FUSED_ARM_CONSTITUENTS` with an
        # `ARM_CERTIFICATION` row beside it. The board cross-checks this column
        # against `launch.py`'s parse tree and refused the cut while the
        # declaration still said False -- correctly, since column (c) would have
        # reported a composition the planner does do as one it does not.
        "wired": True,
    },
    "folded_offdiag_fused_electric_pair": {
        "seam": "D->E",
        # folded_offdiag_fused_electric_pair.py conjoins
        # symmetry.folded_composition_curl_coverage('step_D') (the "folded PML"
        # arm, launch.py:2470) and
        # folded_offdiag_update_e.folded_offdiag_composition_coverage (the
        # "folded off-diagonal" arm). THE COMPOSITION VARIANTS, not the
        # standalone ones: those deliberately admit an unfolded grid so their own
        # gates can prove reduction, and using them here would make this product
        # and the unfolded weld both admit every unfolded off-diagonal run —
        # which `_select_slot` resolves by emptying the slot, costing BOTH cells.
        "cell": ("folded PML", "folded off-diagonal"),
        "module": "folded_offdiag_fused_electric_pair.py",
        # WIRED 2026-09-17. The file boundary the `wired: False` note above
        # describes was crossed by this round: `launch.py` imports this module
        # (its import list, its cell map and its routing table all name it) and
        # its composer label sits in `fastpath.FUSED_ARM_CONSTITUENTS` with an
        # `ARM_CERTIFICATION` row beside it. The board cross-checks this column
        # against `launch.py`'s parse tree and refused the cut while the
        # declaration still said False -- correctly, since column (c) would have
        # reported a composition the planner does do as one it does not.
        "wired": True,
    },
    "cylindrical_real_fused_magnetic_pair": {
        "seam": "B->H",
        # cylindrical_real_fused_magnetic_pair.py conjoins
        # cylindrical_triton.cylindrical_curl_coverage (the "cylindrical PML" arm) and
        # cylindrical_constitutive_coverage('H') (the "cylindrical" arm). RELEASED
        # 2026-08-20 on the GPU host:
        # results/triton_cylindrical_real_fused_magnetic_pair_2026-08-20/gate.json.
        "cell": ("cylindrical PML", "cylindrical"),
        "module": "cylindrical_real_fused_magnetic_pair.py",
        "wired": True,
    },
    # ------------------------------------------------------------------ NEW
    # The 2026-08-21 GAP-CELL round. Both are B->H welds on cells the 08-20 cut
    # ranked as "reachable; NO PRODUCT OCCUPIES THIS CELL — not built".
    "nonlinear_fused_magnetic_pair": {
        "seam": "B->H",
        # nonlinear_fused_magnetic_pair.py conjoins
        # nonlinear_update_e.nonlinear_run_pml_curl_coverage (the "nonlinear run
        # PML" arm, launch.py:2563) and nonlinear_run_constitutive_coverage('H')
        # (the "nonlinear run" arm, launch.py:2680), plus coverage.fused_pair_coverage
        # through a one-clause scope view. ADMISSION ONLY — the kernel it launches
        # is the SHIPPED kernels.fused_curl_constitutive_B through
        # launch.FusedPairPlan. RELEASED 2026-08-20 on the GPU host:
        # results/triton_nonlinear_fused_magnetic_pair_2026-08-21/gate.json.
        "cell": ("nonlinear run PML", "nonlinear run"),
        "module": "nonlinear_fused_magnetic_pair.py",
        "wired": True,
    },
    "bfast_fused_magnetic_pair": {
        "seam": "B->H",
        # bfast_fused_magnetic_pair.py conjoins bfast_curl.bfast_pml_curl_coverage
        # (the "BFAST PML" arm, launch.py:2526) and
        # bfast_run_constitutive_coverage('H') (the "BFAST run" arm,
        # launch.py:2650). A NEW kernel: kernels.fused_curl_constitutive_B plus
        # exactly the one `if HAS_BFAST:` statement bfast_curl.bfast_pml_curl_step
        # adds to pml_curl_step, asserted as two exact STATEMENT-list equalities by
        # the gate's transcription leg. RELEASED 2026-08-21 on the GPU host:
        # results/triton_bfast_fused_magnetic_pair_2026-08-21/keep/gate.json.
        "cell": ("BFAST PML", "BFAST run"),
        "module": "bfast_fused_magnetic_pair.py",
        "wired": True,
    },
    # THE ELECTRIC TWIN, 2026-08-31. The same shape of story as the real-beta pair:
    # the D->E cell was unreachable while an in-seam electric injection was a
    # structural bar, and `deposit_repair` (2026-08-28) is what carries it. The
    # 2026-08-31 board scores
    #
    #     ceiling  1  (rows 1)  D->E  BFAST PML -> BFAST run   NO PRODUCT
    #
    # — tests:TestReflectanceAngular.test_reflectance_angular_2_35_7, shape
    # (1, 1, 1800), all-periodic, `in_seam_source: true` and
    # `in_seam_source_blocks: false`. CARRIES_DEPOSIT_REPAIR is the product: at
    # False the source clause refuses the only row the cell has.
    "bfast_fused_electric_pair": {
        "seam": "D->E",
        # bfast_fused_electric_pair.py conjoins bfast_curl.bfast_pml_curl_coverage
        # on 'step_D' (the "BFAST PML" arm, launch.py:2714) and
        # bfast_run_constitutive_coverage('E') (the "BFAST run" arm,
        # launch.py:2935) — the same two arms as the magnetic twin, read on the
        # other seam. A NEW kernel: kernels.fused_curl_constitutive_D plus exactly
        # the ONE `if HAS_BFAST:` statement bfast_pml_curl_step adds to
        # pml_curl_step, asserted as two exact statement-list equalities by the
        # gate's transcription leg and re-measured on silicon by its reduction leg.
        "cell": ("BFAST PML", "BFAST run"),
        "module": "bfast_fused_electric_pair.py",
        "wired": True,
    },
    "beta_fused_magnetic_pair": {
        "seam": "B->H",
        # beta_fused_magnetic_pair.py conjoins special_kz.beta_pml_curl_coverage
        # (the "real beta PML" arm, launch.py:2516) and
        # beta_run_constitutive_coverage('H') (the "real beta run" arm,
        # launch.py:2640). A NEW kernel: kernels.fused_curl_constitutive_B plus
        # exactly the three lines special_kz.beta_pml_curl_step adds to
        # pml_curl_step, asserted as two exact statement-list equalities by the
        # gate's transcription leg. RELEASED 2026-08-21 on the GPU host:
        # results/triton_beta_fused_magnetic_pair_2026-08-21/gate.json.
        "cell": ("real beta PML", "real beta run"),
        "module": "beta_fused_magnetic_pair.py",
        "wired": True,
    },
    # THE ELECTRIC TWIN, 2026-08-31, and a cell the MAGNETIC twin's docstring
    # explicitly wrote off:
    #
    #     The D/E partner cell is worth ZERO on this corpus (that row injects
    #     electrically between step_D and update_E) and is not built.
    #
    # That was true when it was written and is not true now: `deposit_repair` went
    # live on 2026-08-28 and the injection it names is exactly what the repair
    # carries. The 2026-08-31 board scores the cell at
    #
    #     ceiling  1  (rows 1)  D->E  real beta PML -> real beta run   NO PRODUCT
    #
    # — examples:refl-angular-kz2d.py, `in_seam_source: true` and
    # `in_seam_source_blocks: false` — so CARRIES_DEPOSIT_REPAIR is not one clause
    # of this product, it IS the product: at False the source clause refuses the
    # only row the cell has. The magnetic twin keeps the flag at False for the
    # mirror-image reason: on ITS seam that same row is electric-only.
    "beta_fused_electric_pair": {
        "seam": "D->E",
        # beta_fused_electric_pair.py conjoins special_kz.beta_pml_curl_coverage on
        # 'step_D' (the "real beta PML" arm, launch.py:2704) and
        # beta_run_constitutive_coverage('E') (the "real beta run" arm,
        # launch.py:2925) — the same two arms as the magnetic twin, read on the
        # other seam. A NEW kernel: kernels.fused_curl_constitutive_D plus exactly
        # the three lines special_kz.beta_pml_curl_step adds to pml_curl_step,
        # asserted as two exact statement-list equalities by the gate's
        # transcription leg and re-measured on silicon by its reduction leg
        # (HAS_BETA = 0 must reproduce fused_curl_constitutive_D bit for bit).
        "cell": ("real beta PML", "real beta run"),
        "module": "beta_fused_electric_pair.py",
        "wired": True,
    },
    # THE CONDUCTIVE PML D->E WELD, 2026-08-31 — the LAST buildable non-folded cell
    # on this seam. The 2026-08-31 board scores it at
    #
    #     ceiling  1  (rows 1)  D->E  conductive PML -> ordinary   NO PRODUCT
    #
    # — tests:TestAdjointSolver.test_damping, shape (150, 150, 1), metallic in x and
    # y, `in_seam_source: true` (source_field_types ['D', 'B']) and
    # `in_seam_source_blocks: false`. CARRIES_DEPOSIT_REPAIR is again the product
    # rather than a clause of it: at False the source clause refuses the only row.
    #
    # ITS SIBLING IS `complex_conductive_fused_pair` AND THEY MAY NOT BE CONFLATED.
    # That one carries `_apply_conductive_update` — the NO-PML path, case D alone,
    # no f_cond and no fu — and refuses an active layer. This one carries the PML
    # SPLIT-FIELD recurrence: four cases selected by two EXACT float comparisons,
    # with f_cond written only where `dsig` and fu only where `dsigu`.
    #
    # AND IT SERVES ZERO OF THAT ONE ROW, which is why the cell stays a gap on this
    # board with a product beside it. MEASURED, on the GPU host GPU 6, 2026-08-31, one
    # complete driver step under the signed-zero value class, three routes:
    #
    #     is_integrated=True   identical over 3 steps, 3 deposits repaired
    #     is_integrated=False  Ez differs in 14 words and f_w_Ez in 74, every one of
    #                          them -0.0 against +0.0, while D itself AGREES
    #
    # The cause is not the kernel and not the repair. On a conductive run the driver
    # does not inject at the deposit points: it snapshots the target D component,
    # injects, and rescales BY DIFFERENCE over the WHOLE VOLUME —
    #
    #     array -= before[name]; array *= condinv; array += before[name]
    #                                                  (driver.py:3363-3370)
    #
    # — which is EXACT for every finite value and rewrites -0.0 to +0.0 at every cell
    # of that component. A fused pair computes update_E before the rewrite and
    # `deposit_repair` restores only the deposit points. `TestAdjointSolver.
    # test_damping`'s source is not integrated, so the predicate refuses it BY NAME
    # and the cell's count is zero until driver.py:3363-3370 scales at the deposit
    # indices the sources already publish (`deposit_repair._deposit_index` reads
    # them). That is a change in a file the Triton track does not own.
    #
    # The product ships anyway: it is complete, gated, and correct on every
    # conductive PML run it admits — magnetic-source runs, and INTEGRATED electric
    # ones, which driver.py:3355-3356 injects point-wise and which its carry legs
    # measure. The day that driver line changes, the clause comes out and the cell
    # closes with no other edit.
    # ------------------------------------------------------- 2026-08-31 (this cut)
    # THE NO-ABSORBER STORED-E D->E WELD, and the FIRST product on the SECOND deposit
    # repair. Every other fused product in this table inverts the split-field
    # constitutive recurrence when its seam carries a deposit; this configuration runs
    # `update_E`'s PLAIN overwrite instead (stepping.py:1019-1022), which has no `f_w`,
    # no coefficients and no previous value, so `deposit_repair.PLAIN_PATH` inverts it
    # and the module declares that path rather than the default.
    #
    # TWO CELLS, ONE MODULE, and the split is the census's arm LABELS rather than
    # anything about the kernel: `conductive_plain_curl_step`'s per-component `COND`
    # constexpr compiles to `plain_curl_step`'s own line at zero, so the same kernel is
    # both arms. `fused_pair_B`/`fused_pair_D` set that precedent.
    #
    # ITS CEILING IS 1 OF 3, NOT 3 OF 3, and the shortfall is NOT the deposit. The two
    # conductive rows declare a NON-INTEGRATED electric source, which sends the driver
    # through `_inject_electric_through_conductivity`'s whole-volume rescale
    # (driver.py:3363-3370) at cells no deposit closure can name; the product refuses
    # that by name and its gate MEASURES the divergence rather than asserting it.
    "no_pml_fused_electric_pair": {
        "seam": "D->E",
        "cell": ("conductive no-PML", "no-PML stored E"),
        "module": "no_pml_fused_electric_pair.py",
        "wired": True,
    },
    "no_pml_fused_electric_pair_lossless": {
        "seam": "D->E",
        "cell": ("no-PML", "no-PML stored E"),
        "module": "no_pml_fused_electric_pair.py",
        "wired": True,
    },
    # ------------------------------------------------------- 2026-08-31 (this cut)
    # THE TWO FOLDED-BETA WELDS — the last cells this board ranked "reachable; NO
    # PRODUCT OCCUPIES THIS CELL — not built", built from BUILDABLE_GAP_RECIPES'
    # own measured recipe. ONE corpus row drives both
    # (tests:TestSpecialKz.test_eigsrc_kz_1_real_imag: 2-D, beta = 0.2, Mirror(Y)
    # over a PERIODIC declaration, sources ['D', 'D', 'B', 'B']), so each product
    # is worth one seam-instance and CARRIES_DEPOSIT_REPAIR is the product on BOTH:
    # the row injects into each product's own seam, and at False the source clause
    # refuses the only row either cell has.
    "folded_beta_fused_electric_pair": {
        "seam": "D->E",
        # folded_beta_fused_electric_pair.py conjoins
        # folded_complex.folded_beta_pml_curl_coverage on 'step_D' (the "folded
        # real beta PML" arm, launch.py:2740) and
        # folded_beta_run_constitutive_coverage('E') (the "folded beta run" arm)
        # plus folded_fused_pair_coverage's re-spelled fold and seam clauses. The
        # kernel is folded_fused_pair.folded_fused_curl_constitutive_D plus exactly
        # the three lines folded_complex.folded_beta_pml_curl_step adds to
        # symmetry.pml_curl_step_folded, asserted as exact statement-list
        # equalities by the gate's transcription leg and re-measured on silicon by
        # its reduction leg (HAS_BETA = 0 must reproduce the plain folded kernel
        # bit for bit).
        "cell": ("folded real beta PML", "folded beta run"),
        "module": "folded_beta_fused_electric_pair.py",
        "wired": True,
    },
    "folded_beta_fused_magnetic_pair": {
        "seam": "B->H",
        # The MAGNETIC twin: folded_fused_magnetic_pair.folded_fused_curl_
        # constitutive_B plus the same three-line insert, same construction, same
        # single corpus row on the other seam (its TWO magnetic sources land in
        # THIS one, driver.py:3283-3284).
        "cell": ("folded real beta PML", "folded beta run"),
        "module": "folded_beta_fused_magnetic_pair.py",
        "wired": True,
    },
    "conductive_fused_electric_pair": {
        "seam": "D->E",
        # conductive_fused_electric_pair.py conjoins
        # conductivity.conductive_pml_curl_coverage on 'step_D' (the "conductive
        # PML" arm, launch.py:2654) and coverage.constitutive_coverage('E') (the
        # "ordinary" arm, launch.py:2903). A NEW kernel, and the only fused product
        # in this table that CALLS a shipped device function rather than copying it:
        # `_conductive_component_registers` is conductivity._conductive_component
        # with ONE statement moved out (the `f` store, which the caller performs
        # after the wall clear), asserted as an exact statement-list equality by the
        # gate's transcription leg and re-measured on silicon by its reduction leg
        # (COND=(0,0,0) must reproduce fused_curl_constitutive_D bit for bit).
        "cell": ("conductive PML", "ordinary"),
        "module": "conductive_fused_electric_pair.py",
        "wired": True,
    },
    # ------------------------------------------------------- 2026-08-21 (this cut)
    # THE FOLDED COMPLEX B->H WELD — the intersection of the two shipped B/H pairs,
    # and the one cell on this seam where neither can stand in for the other.
    # `folded_fused_magnetic_pair` refuses complex storage; `complex_fused_magnetic_
    # pair` refuses a fold BY NAME.
    #
    # TWO ENTRIES, ONE MODULE, and the split is the census's arm LABELS rather than
    # anything about the kernel. `folded_complex.py` registers a second family name
    # over the SAME kernel and the SAME plan class for a run carrying an
    # off-diagonal chi1inv row — `plan_folded_complex_offdiag_pml_curl` (:3101)
    # returns a `FoldedComplexPmlCurlPlan` and says so — so the shipped predicate
    # takes EITHER matched admission and this table prices both cells to the same
    # module. `fused_pair_B`/`fused_pair_D` already set that precedent.
    #
    # The Metal board records all four rows in ONE cell
    # (`fusion_matrix_metal_2026-08-21c`, folded_complex_fused_magnetic_pair, B->H
    # 4), which is the independent check that the union is right.
    "folded_complex_fused_magnetic_pair": {
        "seam": "B->H",
        "cell": ("folded complex PML", "folded complex"),
        "module": "folded_complex_fused_magnetic_pair.py",
        "wired": True,
    },
    "folded_complex_fused_magnetic_pair_offdiag": {
        "seam": "B->H",
        "cell": ("folded complex off-diagonal PML", "folded complex off-diagonal"),
        "module": "folded_complex_fused_magnetic_pair.py",
        "wired": True,
    },
    "cylindrical_fused_magnetic_pair": {
        "seam": "B->H",
        # cylindrical_fused_magnetic_pair.py:744-748 conjoins
        # cylindrical_complex.cylindrical_complex_curl_coverage (the "cylindrical
        # complex PML" arm) and cylindrical_complex_constitutive_coverage('H') (the
        # "cylindrical complex" arm). THE |m| >= 1 Dcyl CELL, a DIFFERENT cell from
        # the m = 0 real one. RELEASED 2026-08-20 on the GPU host:
        # results/triton_cylindrical_fused_magnetic_pair_2026-08-20/gate.json
        # (VERDICT PASS, 76 rows). Carried verbatim from
        # ../fusion_matrix_triton_2026-08-20c/.
        "cell": ("cylindrical complex PML", "cylindrical complex"),
        "module": "cylindrical_fused_magnetic_pair.py:729",
        "wired": True,
    },
    "fused_dispersive_chain": {
        "seam": "E->P",
        # fused_dispersive_chain.py:405 conjoins dispersive_fused_pair_coverage —
        # so it inherits that product's ("PML", "dispersive") cell — and
        # fused_ade_state_coverage for the single registered state.
        "cell": ("PML", "dispersive"),
        "module": "fused_dispersive_chain.py:389",
        "wired": False,
    },
    # --- THE TWO ADE CHAINS, PRICED 2026-08-30 --------------------------------
    # Both were BUILT and their gates RELEASED before this cut, and both were
    # reported at ZERO — `fused_ade_chain` under GATED_NOT_PRICED_HERE ("its
    # predicate reads live polarization objects the census block does not carry"),
    # `complex_fused_ade_chain` under LANDED_CONCURRENTLY_NOT_PRICED_HERE ("will
    # not price another round's family from its docstring"). Both abstentions were
    # correct about a TRANSCRIBED ladder and are answered by not transcribing one:
    # the census now records what each shipped predicate returned on the real
    # lifted objects, exactly as it already did for `fused_pair_B/D` and
    # `dispersive_fused_pair_D`, and this board reads that verdict.
    #
    # THE CELL COLUMN IS REPORTING, NOT ROUTING. A product is credited at this seam
    # because its OWN predicate admitted the row, never because a declared cell
    # matched — see the E->P branch below for why that distinction is the whole
    # point at a seam whose arm labels collapse two of one module's three arms.
    "fused_ade_chain": {
        "seam": "E->P",
        # ARMS = ("no_pml", "dispersive", "folded") at fused_ade_chain.py:208. The
        # first lands on the no-PML cell and the other two BOTH land on
        # ("dispersive", "ADE update_P"): the census's arm labels do not split
        # folded from unfolded on the E side. Declared as the set it occupies.
        "cell": ("dispersive", "ADE update_P"),
        "cells": (("dispersive", "ADE update_P"),
                  ("no-PML dispersive", "no-PML ADE update_P")),
        "module": "fused_ade_chain.py:826 fused_ade_chain_coverage",
        "wired": False,
    },
    "complex_fused_ade_chain": {
        "seam": "E->P",
        "cell": ("complex no-PML", "complex ADE update_P"),
        "cells": (("complex no-PML", "complex ADE update_P"),),
        "module": "complex_fused_ade_chain.py:658 complex_fused_ade_chain_coverage",
        "wired": False,
    },
    # --- THE THIRD ADE CHAIN, PRICED WITH THE ROUND THAT BUILT IT -------------
    # THE LAST BUILDABLE E->P CELL ON THIS BOARD. Before this product the seam
    # read 14 of 15 served, and the one instance left --
    # ``examples:absorbed_power_density.py`` -- was reported "reachable; NO
    # PRODUCT OCCUPIES THIS CELL -- not built": its E arm is the FOLDED
    # OFF-DIAGONAL DISPERSIVE constitutive, which `fused_ade_chain` refuses on
    # all three of its arms (the off-diagonal row on `folded`, the mirror plane
    # on `dispersive`, the active layer on `no_pml`) and
    # `complex_fused_ade_chain` refuses on storage.
    #
    # ONE LAUNCH, THREE COMPONENTS, ONE SCRATCH PER DRIVEN COMPONENT. The E half
    # is folded_offdiag_dispersive_update_e's certified body, contiguous and
    # verbatim, and the ADE half is kernels.ade_update_p appended per (component,
    # pole slot) under six declared renames. It is BOUND BY ITS GATE rather than
    # by a fingerprint: the kernel EXECUTES JIT helpers out of BOTH folded
    # off-diagonal modules, which seed_triton_welds' curated set does not pin, and
    # the gate's own `source_sha256` does.
    "folded_offdiag_fused_ade_chain": {
        "seam": "E->P",
        "cell": ("folded off-diagonal dispersive", "ADE update_P"),
        "cells": (("folded off-diagonal dispersive", "ADE update_P"),),
        "module": ("folded_offdiag_fused_ade_chain.py:710 "
                   "folded_offdiag_fused_ade_chain_coverage"),
        "wired": False,
    },
    # ------------------------------------------------- 2026-08-30 (this cut)
    # THE COMPLEX D->E WELD — the largest single BUILDABLE gap the 2026-08-30
    # board reported, tied with the cylindrical complex D->E cell:
    #
    #     ceiling  16  (rows 16)  D->E  complex PML -> complex   NO FUSED PRODUCT
    #
    # It is the electric twin of `complex_fused_magnetic_pair`, over the same two
    # certified halves on the other seam, and it did not exist for a measured
    # reason rather than an accidental one: 15 of the 16 complex rows declare an
    # ELECTRIC source, so under the source clause as it stood the arm admitted ONE
    # row against the magnetic twin's twelve. `deposit_repair` carrying a deposit
    # across the seam is what changed, and this product declares
    # CARRIES_DEPOSIT_REPAIR in the same change as the wiring that brackets its
    # launch — which is why the ceiling above is 16 and not 1.
    #
    # RELEASED 2026-08-30 on the GPU host GPU 7 (verified physically empty), Triton
    # 3.1.0, cc 8.6, under the `keep` policy:
    # results/triton_complex_fused_electric_pair_2026-08-30/
    # complex_fused_electric_pair/gate.json — 8 device cases (5 quiet + 3 carrying
    # an in-seam electric deposit through the SHIPPED repair bracket) x 10 complete
    # driver steps, word-for-word against BOTH the array path and the two separately
    # certified Triton products it replaces; 13 mutations, 12 caught as declared and
    # one null as declared.
    "complex_fused_electric_pair": {
        "seam": "D->E",
        # complex_fused_electric_pair.py conjoins complex_pml_curl_coverage on
        # 'step_D' (the "complex PML" arm) and complex_constitutive_coverage('E')
        # (the "complex" arm) — the same two arms as the magnetic twin, read on the
        # other seam.
        "cell": ("complex PML", "complex"),
        "module": "complex_fused_electric_pair.py",
        "wired": True,
    },
    # ------------------------------------------------- 2026-08-30 (this cut)
    # THE Dcyl D->E WELD — RANK 1 on the board after the Cartesian twin above
    # closed the tie, and the largest single cell left on ANY backend's board:
    #
    #     ceiling  16  (rows 16)  D->E  cylindrical complex PML
    #                             -> cylindrical complex        NO FUSED PRODUCT
    #
    # IT IS A CELL ONLY THIS BACKEND CAN TAKE. The Metal board scores the same
    # conjunction under "cannot be bound at any sharing" — its argument-table
    # ceiling is 31 bindings and this pair needs more — while Triton has no binding
    # ceiling at all.
    #
    # ALL SIXTEEN Dcyl ROWS DECLARE AN ELECTRIC SOURCE ('D' on fourteen, ('D','D')
    # on cylinder_cross_section.py and zone_plate.py), so under the source clause as
    # the MAGNETIC twin states it this arm would admit NOTHING AT ALL. The ceiling
    # of 16 is the count with the deposit repair carried, which the module declares
    # in the same change as the wiring that brackets its launch.
    "cylindrical_fused_electric_pair": {
        "seam": "D->E",
        # cylindrical_fused_electric_pair.py conjoins
        # cylindrical_complex_curl_coverage on 'step_D' (the "cylindrical complex
        # PML" arm) and cylindrical_complex_constitutive_coverage('E') (the
        # "cylindrical complex" arm) — the same two arms as the magnetic twin, read
        # on the other seam.
        "cell": ("cylindrical complex PML", "cylindrical complex"),
        "module": "cylindrical_fused_electric_pair.py",
        "wired": True,
    },
    # THE COMPLEX CONDUCTIVE no-PML D->E WELD — rank 2 among the buildable cells
    # after the two above:
    #
    #     ceiling   4  (rows 4)   D->E  complex conductive no-PML curl
    #                             -> complex no-PML stored E     NO FUSED PRODUCT
    #
    # Its four rows are TestLoadDump.test_load_dump_{structure,structure_sharded,
    # chunk_layout_file,chunk_layout_sim}_3d. UNLIKE the two pairs above it carries
    # NO deposit repair and needs none: every one of the four declares a MAGNETIC
    # source, which is injected in the B/H seam, so the electric clause costs this
    # cell zero rows. Its Metal sibling serves the same cell 4/4 and was released
    # 2026-08-29.
    "complex_conductive_fused_pair": {
        "seam": "D->E",
        # complex_conductive_fused_pair.py conjoins
        # complex_conductive_no_pml_curl_coverage on 'step_D' (the "complex
        # conductive no-PML curl" arm) and complex_stored_e_coverage (the "complex
        # no-PML stored E" arm).
        "cell": ("complex conductive no-PML curl", "complex no-PML stored E"),
        "module": "complex_conductive_fused_pair.py",
        "wired": True,
    },
    # ------------------------------------------------- 2026-08-31 (this cut)
    # THE FOLDED DISPERSIVE D->E WELD — the LARGEST cell the 2026-08-31 board
    # reported as "reachable; NO PRODUCT OCCUPIES THIS CELL — not built":
    #
    #     ceiling   4  (rows 4)   D->E  folded PML
    #                             -> folded dispersive          NO FUSED PRODUCT
    #
    # Its four rows are TestLoadDump.test_load_dump_{structure,structure_sharded,
    # chunk_layout_file,chunk_layout_sim}_2d. A GENUINE NEW WELD rather than a port:
    # both halves were separately certified (`folded_composition_curl` at step_D and
    # `folded_dispersive_constitutive` at update_E) and no sibling on any backend
    # performs this conjunction.
    #
    # ALL FOUR ROWS DECLARE AN ELECTRIC SOURCE, so under the source clause with
    # CARRIES_DEPOSIT_REPAIR at False this arm would admit NOTHING AT ALL. The
    # ceiling of 4 is the count with the deposit repair carried, which the module
    # declares in the same change as the wiring that brackets its launch.
    #
    # RELEASED 2026-08-31 on the GPU host GPU 6 (verified physically empty), Triton
    # 3.1.0 / CuPy 13.5.1, cc 8.6, under the `keep` policy through an ftz-stripped
    # cache: results/triton_folded_disp_2026-08-31/folded_dispersive_fused_pair/
    # gate.json — 18 device legs, 127 complete driver steps, 121 fused launches,
    # word-for-word against BOTH the array path and the three separately certified
    # Triton products it replaces; 3 CARRY legs each repairing 6 in-seam electric
    # deposits through the SHIPPED bracket, 3 NULL CONTROLS with the bracket removed
    # that all diverge on Ez as they must, 22 mutations (21 caught as declared, one
    # null confirmed) and 6/6 refusals.
    "folded_dispersive_fused_pair": {
        "seam": "D->E",
        # folded_dispersive_fused_pair.py conjoins
        # symmetry.folded_composition_curl_coverage on 'step_D' (the "folded PML"
        # arm, launch.py:2470) and
        # folded_dispersive_update_e.folded_dispersive_constitutive_coverage (the
        # "folded dispersive" arm) — the same curl arm `folded_fused_pair` takes,
        # welded to the OTHER constitutive. The two predicates are DISJOINT by
        # construction: that product refuses a registered susceptibility by name and
        # this one's E half requires one.
        "cell": ("folded PML", "folded dispersive"),
        "module": "folded_dispersive_fused_pair.py",
        "wired": True,
    },
    # THE Dcyl m = 0 D->E WELD — rank 2 among the buildable cells on the 2026-08-31
    # board, after the folded dispersive one above:
    #
    #     ceiling   3  (rows 3)   D->E  cylindrical PML
    #                             -> cylindrical                NO FUSED PRODUCT
    #
    # THE ELECTRIC TWIN of `cylindrical_real_fused_magnetic_pair`, over the SAME two
    # certified halves on the other seam, and the SAME three rows —
    # TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_{0_0,1_0} and
    # TestPMLCylindrical.test_pml_cyl_0_0_0. The gate's corpus leg measures that
    # equality rather than inheriting it.
    #
    # ALL THREE ROWS DECLARE AN ELECTRIC SOURCE, which is exactly why the MAGNETIC
    # twin's seam clause costs it nothing and why this one's would cost it
    # everything: with CARRIES_DEPOSIT_REPAIR at False this arm admits ZERO. The
    # module declares the flag in the same change as the wiring that brackets its
    # launch, and its twin keeps the flag at False for the mirror-image reason.
    #
    # RELEASED 2026-08-31 on the GPU host GPU 6 (verified physically empty), Triton
    # 3.1.0 / CuPy 13.5.1, cc 8.6, under the `keep` policy through an ftz-stripped
    # cache: results/triton_cyl_real_electric_2026-08-31c/
    # cylindrical_real_fused_electric_pair/gate.json — 10 device legs, 60 complete
    # driver steps, 54 fused launches, word-for-word against BOTH the array path and
    # the two separately certified Triton products it replaces; 2 CARRY legs
    # repairing 6 and 12 in-seam electric deposits through the SHIPPED bracket, 2
    # NULL CONTROLS with the bracket removed that both diverge on Ez, 13 mutations
    # (12 caught as declared, one null confirmed) and 4/4 refusals.
    #
    # THE CUPY VERSION IS PART OF THE CLAIM: everything downstream of the radial
    # prefix scan is bit-identical only for a given `cupy.cumsum` summation order, so
    # a CuPy bump is a correctness event for this family and a silent one.
    # THE FOUR ELECTRIC-PAIR ROWS ABOVE/BELOW WERE RE-POINTED 2026-08-31 to the
    # retry3 re-runs: the plain-branch round edited meep_gpu/deposit_repair.py and
    # triton_kernels/launch.py AFTER their first artifacts were cut, and each retry3
    # artifact was verified released=True with 0 drift against the live tree before
    # its repoint.
    "cylindrical_real_fused_electric_pair": {
        "seam": "D->E",
        # cylindrical_real_fused_electric_pair.py conjoins
        # cylindrical_triton.cylindrical_curl_coverage (the "cylindrical PML" arm)
        # and cylindrical_constitutive_coverage('E') (the "cylindrical" arm) — the
        # same two arms as the magnetic twin, read on the other seam.
        "cell": ("cylindrical PML", "cylindrical"),
        "module": "cylindrical_real_fused_electric_pair.py",
        "wired": True,
    },
    # THE FOLDED COMPLEX D->E WELD — the LARGEST remaining unbuilt cell on the
    # 2026-08-31 board, at two seam-instances:
    #
    #     ceiling   2  (rows 2)   D->E  folded complex PML
    #                             -> folded complex             NO FUSED PRODUCT
    #
    # — tests:TestEigCoeffs.test_binary_grating_special_kz_2_21_2 and
    # tests:TestModeDecomposition.test_triangular_lattice_oblique, the same two rows
    # its MAGNETIC twin already serves on the other seam, measured in
    # results/fusion_matrix_triton_2026-08-31_folded_disp.
    #
    # BOTH ROWS DECLARE AN ELECTRIC SOURCE, so with CARRIES_DEPOSIT_REPAIR at False
    # this arm admits ZERO. The module declares the flag in the same change as the
    # wiring that brackets its launch, and its magnetic twin keeps the flag at False
    # for the mirror-image reason: on the B seam those same two rows are
    # electric-only.
    #
    # ONE ADMISSION, unlike the magnetic twin. That family registers a SECOND arm
    # label over the same kernel for an off-diagonal chi1inv row, because on B->H
    # the two arms ARE the same kernel; on D->E the off-diagonal update_E is the
    # STENCIL this board refuses on every backend, so this product refuses the row by
    # name and occupies one cell rather than two.
    "folded_complex_fused_pair": {
        "seam": "D->E",
        # folded_complex_fused_pair.py conjoins
        # folded_complex.folded_complex_composition_curl_coverage on 'step_D' (the
        # "folded complex PML" arm, launch.py:2722),
        # folded_complex_constitutive_coverage('E') (the "folded complex" arm,
        # launch.py:2947) and the ghost fill's own predicate — the same three halves
        # the magnetic twin conjoins, read on the other seam.
        "cell": ("folded complex PML", "folded complex"),
        "module": "folded_complex_fused_pair.py",
        "wired": True,
    },
    # THE TWO COMPLEX-BETA WELDS, 2026-09-01 — the unfolded half of the
    # no-admitting-arm residue (audit §1.4). The 2026-09-01_foldedbeta board
    # scored BOTH seam-instances of tests:TestSpecialKz.test_special_kz as
    # "reachable; a slot has NO admitting arm (array path)" with
    # ``curl_admitters: []``: the K2 curl arm refused because the census probe
    # classified only the base four patterns and the beta tranche's fifth
    # (`c8_mul_c8_imaginary_coefficient_left`, special_kz.py:223) was unmeasured.
    # THE REFUSAL WAS DISCHARGED BY MEASUREMENT, not by edit:
    # results/complex_expansion_beta_extension_2026-09-01/probe_keep classifies
    # the fifth pattern on the device (AMBIGUOUS_BOTH on all 16 corpus
    # coefficients, arms measured 0 words apart per entry, FMA_V2 diagnostic
    # mismatching) and its base-four table is character-for-character identical
    # to the standing 2026-08-16 keep artifact. Under that probe both K2 arms
    # admit and these cells exist.
    "complex_beta_fused_magnetic_pair": {
        "seam": "B->H",
        # complex_beta_fused_magnetic_pair.py conjoins
        # special_kz.beta_bloch_pml_curl_coverage on 'step_B' (the "complex beta
        # PML" arm, launch.py:2718) and beta_run_complex_constitutive_coverage('H')
        # (the "complex beta run" arm, launch.py:2842).
        "cell": ("complex beta PML", "complex beta run"),
        "module": "complex_beta_fused_magnetic_pair.py",
        "wired": True,
    },
    "complex_beta_fused_electric_pair": {
        "seam": "D->E",
        # complex_beta_fused_electric_pair.py conjoins the same two arms on the
        # other seam ('step_D' / side 'E', launch.py:2718/2939). The row's one
        # electric source is injected in THIS seam, so CARRIES_DEPOSIT_REPAIR is
        # the product: at False the source clause refuses the cell's only row.
        "cell": ("complex beta PML", "complex beta run"),
        "module": "complex_beta_fused_electric_pair.py",
        "wired": True,
    },
    # THE TWO FOLDED COMPLEX-BETA WELDS, 2026-09-02 — the folded half of the
    # no-admitting-arm residue (audit §1.4). The foldedbeta board scored all SIX
    # seam-instances of the three folded beta rows (the two binary gratings and
    # test_eigsrc_kz_0_complex) as no-admitting-arm with ``curl_admitters: []``;
    # the same probe extension that unlocked the unfolded pair unlocks K3b
    # ("folded complex beta PML", launch.py:2744), and these two products occupy
    # the cells it opens. Their PRODUCT_PROBE_PATTERNS is the SIX-pattern union
    # (parity + beta on top of the base four), which the census probe does not
    # carry — the credit reads the ``@unified_expansion`` census verdicts, the
    # folded_complex twins' own mechanism, with the census-probe verdict carried
    # beside it per instance.
    "folded_beta_complex_fused_magnetic_pair": {
        "seam": "B->H",
        # folded_beta_complex_fused_magnetic_pair.py conjoins K3b
        # (folded_complex.folded_beta_bloch_pml_curl_coverage on 'step_B', the
        # "folded complex beta PML" arm, launch.py:2744) and the folded complex
        # constitutive (the "folded complex" arm) with both fills carried.
        "cell": ("folded complex beta PML", "folded complex"),
        "module": "folded_beta_complex_fused_magnetic_pair.py",
        "wired": True,
    },
    "folded_beta_complex_fused_pair": {
        "seam": "D->E",
        # The electric twin, same two arms on 'step_D' / side 'E'. All three of
        # the cell's rows inject electrically in THIS seam, so
        # CARRIES_DEPOSIT_REPAIR is the product: at False it serves zero.
        "cell": ("folded complex beta PML", "folded complex"),
        "module": "folded_beta_complex_fused_pair.py",
        "wired": True,
    },
}

#: The products whose `shipped` row reads an ``@unified_expansion`` census
#: verdict, mapped to the CENSUS-PROBE key for the same predicate — the pair of
#: columns every such instance carries (`credited_under_a_different_expansion_
#: licence` / `same_product_under_the_census_probe`). One predicate can own two
#: cells (the folded_complex magnetic pair and its `_offdiag` label), so this is
#: keyed by PRODUCT name, not by module.
UNIFIED_CREDITED_PRODUCTS: Dict[str, str] = {
    "folded_complex_fused_magnetic_pair": "folded_complex_fused_magnetic_pair_B",
    "folded_complex_fused_magnetic_pair_offdiag":
        "folded_complex_fused_magnetic_pair_B",
    "folded_complex_fused_pair": "folded_complex_fused_pair_D",
    "folded_beta_complex_fused_pair": "folded_beta_complex_fused_pair_D",
    "folded_beta_complex_fused_magnetic_pair":
        "folded_beta_complex_fused_magnetic_pair_B",
}

#: The E->P products, in the order the report consults them. ``fused_dispersive_chain``
#: is NOT here: it is scored by its own D->E cell (the pair it fuses) and credited into
#: this seam separately, because one launch of it performs ``update_P`` as well.
E_TO_P_PRODUCTS: Tuple[str, ...] = ("fused_ade_chain", "complex_fused_ade_chain",
                                    "folded_offdiag_fused_ade_chain")

#: Which recorded seam verdict credits each E->P product, and WHY that key.
#:
#: ``fused_ade_chain`` is asked once per arm and admitted if ANY arm admits. That is
#: not a widening: ``arm`` is an explicit argument to this module's predicate and to
#: its plan builder (fused_ade_chain.py:826, :990) rather than a table lookup, so the
#: caller names the arm and one launch serves the row whenever one arm says yes. The
#: cut asserts the arms are mutually exclusive on this corpus rather than assuming it
#: (:func:`assert_the_chain_arms_do_not_overlap`), so "any" and "the one" agree on
#: every measured row.
#:
#: ``complex_fused_ade_chain`` is credited from the UNIFIED-EXPANSION verdict, not
#: from the census-probe one, and the reason is measured rather than chosen. Its ADE
#: half needs ``c8_mul_python_float_field_left`` on top of the four base patterns and
#: NO per-family probe carries it: scored on 2026-08-30, this census's own probe
#: returns ``expansion=None`` ("pattern ... is not classified") and
#: results/unified_expansion_2026-08-27/keep/gate.json returns ``expansion=1`` with no
#: refusals. That is the same artifact the family's RELEASED gate requires, stated in
#: its own gate entry (drive_triton_weld_gates.py:255-265). BOTH verdicts are carried
#: in the census row and BOTH are reported below: crediting the family against the
#: licence its gate did not hold would be a wrong answer, and reporting zero because a
#: DIFFERENT family's probe was passed in would be a different wrong answer.
#: ``folded_offdiag_fused_ade_chain`` is asked ONCE, under ONE key, and both facts
#: are the product's own rather than a convention: its predicate takes no ``arm``
#: argument (there is one body and one admission) and it consumes NO expansion
#: record at all — the E half it composes is real float32 storage, so neither the
#: census probe nor the unified record enters its verdict and there is nothing for
#: a second key to distinguish.
E_TO_P_VERDICT_KEYS: Dict[str, Tuple[str, ...]] = {
    "fused_ade_chain": ("fused_ade_chain@no_pml", "fused_ade_chain@dispersive",
                        "fused_ade_chain@folded"),
    "complex_fused_ade_chain": ("complex_fused_ade_chain@unified_expansion",),
    "folded_offdiag_fused_ade_chain": ("folded_offdiag_fused_ade_chain",),
}

#: ``fused_ade_state`` is NOT in the table above and that is deliberate. It fuses
#: one susceptibility's per-component ``update_P`` launches into one launch; it
#: does not cross a sub-step boundary and occupies no (curl, constitutive) cell.
#: It is measured separately, below.


# ---------------------------------------------------------------------------
# The predicates, evaluated per row
# ---------------------------------------------------------------------------

def folded_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``folded_fused_magnetic_pair_coverage`` (folded_fused_magnetic_pair.py:549).

    Same derivation the 2026-08-19 census made, against the licensed record.
    """
    configuration = row["configuration"]
    shape = configuration["shape"]
    types = configuration.get("source_field_types") or []
    folded = [axis for axis in range(3) if configuration["mirrored"][axis]]
    return {
        # :561 folded curl half
        "curl_half": covered(row, "folded_composition_curl@step_B"),
        # :564 folded constitutive half
        "constitutive_half": covered(row, "folded_constitutive@update_H"),
        # :584-590 the magnetic source seam (driver.py:3284)
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # THE CLAUSE THAT WAS HERE IS GONE, and it is the whole of this cut.
        # `no_folded_periodic_axis` — `all(metallic[a] for a in folded)` — refused a
        # folded PERIODIC axis because `fill_folded_far_ghosts_B` (driver.py:3287)
        # runs inside the seam there and no product imaged the top stored slot. The
        # kernel carries it as of 2026-08-20, gated on the GPU host and released
        # (results/triton_folded_far_carry_2026-08-20/run_farcarry7/gate.json), so
        # the clause is retired rather than weakened.
        #
        # WHAT REPLACED IT IS NOT EVALUABLE FROM THE CENSUS BLOCK and is declared in
        # NOT_EVALUABLE below: the far carry needs `0 <= n_full - stored + 2 <
        # stored - 1`, not 0, and `stored - 1 != 2`. The block carries `shape` but
        # not `shape_full`. MEASURED INSTEAD over every folded PERIODIC extent Grid
        # will build (58 of them, cells 3..60): stored == ceil(n_full/2) + 2 and
        # reflect == floor(n_full/2), zero violations, minimum stored 4 — so all
        # three hold on every grid this engine can construct and the carry adds no
        # admission question.
        # :617-623 the near fill images stored cell 2 from that cell's own lane
        "fold_stores_the_source_row": all(int(shape[a]) > NEAR_SOURCE_INDEX
                                          for a in folded),
    }


def folded_complex_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``folded_complex_fused_magnetic_pair_coverage``.

    EITHER MATCHED ADMISSION. The two arm labels are DISJOINT by construction (each
    refuses the other's configuration by name), so ``plain or offdiag`` is a choice
    between two labels for one kernel rather than a widening; the shipped predicate
    takes them as a PAIR and so does this ladder.

    The FILL half's own predicate is not in the census — the battery carries no
    ``folded_mirror_ghost_fill_complex`` column — and every clause it adds beyond
    these two is either shared with them (the fold classification, the phase rules,
    the layout) or is the EXTENDED expansion pattern set, which is a property of the
    probe artifact rather than of a row. That artifact clause is measured once,
    globally, in :func:`parity_licence_holds`, and is therefore not a per-row term.
    """
    configuration = row["configuration"]
    metallic, mirrored = configuration["metallic"], configuration["mirrored"]
    shape = configuration["shape"]
    types = configuration.get("source_field_types") or []
    folded = [axis for axis in range(3) if mirrored[axis]]
    plain = (covered(row, "folded_complex_composition_curl@step_B")
             and covered(row, "folded_complex_constitutive@update_H"))
    offdiag = (covered(row, "folded_complex_offdiag_pml_curl@step_B")
               and covered(row, "folded_complex_offdiag_constitutive@update_H"))
    return {
        # the two halves, as a MATCHED pair from either admission
        "a_matched_admission": bool(plain or offdiag),
        # the MAGNETIC source seam (driver.py:3283-3284) — the clause that caps
        # this family, exactly as it caps every other B->H product
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # the near fill images stored cell 2 from that cell's OWN lane, so a folded
        # axis storing two cells or fewer is an out-of-range read
        "fold_stores_the_source_row": all(int(shape[a]) > NEAR_SOURCE_INDEX
                                          for a in folded),
        # THE WALL CLEAR AND THE FOLD MUST NOT MEET — and this is a DRIFT CHECK,
        # not a row filter. The shipped clause asks `coverage.zero_metal_axes`,
        # which is `has_metallic and is_metallic(axis) and NOT is_mirrored(axis)`
        # (coverage.py:423-428) — precisely `stepping._zero_metal`'s own question
        # (:2237-2239) — so on a FOLDED axis it is False by construction and the
        # clause cannot fire. Transcribed that way here rather than as
        # `metallic[a] and mirrored[a]`, which is the OPPOSITE reading and which
        # refused two of the four rows this product serves on the first cut of this
        # ladder: `solve-cw.py` and `test_array_metadata` declare mirrored=[T,T,F]
        # AND metallic=[T,T,F], where `zero_metal_axes` answers all-False.
        "wall_and_fold_are_disjoint": not any(
            bool(mirrored[a])
            and (bool(configuration.get("has_metallic")) and bool(metallic[a])
                 and not bool(mirrored[a]))
            for a in range(3)),
        # the EXTENDED expansion pattern set, measured once against the artifact the
        # census declared rather than re-asked per row
        "parity_licence": parity_licence_holds(),
    }


def complex_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``complex_fused_magnetic_pair_coverage`` (complex_fused_magnetic_pair.py:581)."""
    configuration = row["configuration"]
    types = configuration.get("source_field_types") or []
    return {
        # :600 complex curl half
        "curl_half": covered(row, "complex_pml_curl@step_B"),
        # :603 complex constitutive half
        "constitutive_half": covered(row, "complex_constitutive@update_H"),
        # :625-631 the magnetic source seam
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # :642-651 both symmetry passes must stay inert
        "no_symmetry_in_the_seam": (not bool(configuration.get("has_symmetry"))
                                    and not any(bool(v)
                                                for v in configuration["mirrored"])),
    }


_PARITY_LICENCE: Dict[str, Any] = {}


def parity_licence_holds() -> bool:
    """Does the census's own probe artifact license the EXTENDED pattern set?

    ``folded_complex_fused_magnetic_pair`` calls the fill's parity multiply, so it
    binds ``folded_complex.PARITY_PROBE_PATTERNS`` — the base four plus
    ``c8_mul_c8_parity_coefficient_left`` — where every other complex product here
    binds the base four. That is a property of the ARTIFACT, not of a corpus row, so
    it is measured ONCE against the record the census declared and cached.

    A record that licenses the base four and NOT the parity pattern would admit
    every other complex product and refuse this one; the answer is recorded in the
    artifact so the reader can see which it was.
    """
    if not _PARITY_LICENCE:
        import sys as _sys  # noqa: PLC0415
        if str(API) not in _sys.path:
            _sys.path.insert(0, str(API))
        from meep_gpu.triton_kernels import folded_complex as _fc  # noqa: PLC0415
        artifact = json.loads(PROBE_ARTIFACT.read_text())
        verdict = _fc.parity_expansion_license(artifact)
        census_verdict = _fc.parity_expansion_license(
            json.loads(CENSUS_PROBE_ARTIFACT.read_text()))
        _PARITY_LICENCE.update({
            "artifact": str(PROBE_ARTIFACT.relative_to(API)),
            "patterns": list(_fc.PARITY_PROBE_PATTERNS),
            "arm": verdict["arm"], "expansion": verdict["expansion"],
            "refusals": list(verdict["refusals"]),
            "licensed": verdict["expansion"] is not None,
            "census_artifact": str(CENSUS_PROBE_ARTIFACT.relative_to(API)),
            "census_artifact_licensed": census_verdict["expansion"] is not None,
            "census_artifact_refusals": list(census_verdict["refusals"]),
            "why_not_the_census_artifact":
                "the census artifact (the 2026-09-01 beta extension of the "
                "2026-08-16 standing cut) classifies the base four plus the beta "
                "tranche's imaginary-coefficient-left pattern, and NOT "
                "c8_mul_c8_parity_coefficient_left, which arrived with the tranche "
                "that added the parity multiply. Its refusal is about the record's "
                "pattern set, not the platform. The artifact used here is the one "
                "the RELEASED device gate consumed, on the same host under the same "
                "'keep' policy.",
        })
        log(f"  PARITY LICENCE (extended pattern set): "
            f"arm={_PARITY_LICENCE['arm']!r} "
            f"licensed={_PARITY_LICENCE['licensed']} "
            f"artifact={_PARITY_LICENCE['artifact']}")
        log(f"    the CENSUS's own artifact answers "
            f"licensed={_PARITY_LICENCE['census_artifact_licensed']}: "
            f"{_PARITY_LICENCE['census_artifact_refusals']}")
    return bool(_PARITY_LICENCE["licensed"])


def folded_electric_clauses(row: dict) -> Dict[str, bool]:
    """``folded_fused_pair_coverage`` (folded_fused_pair.py).

    The MIRROR IMAGE of :func:`folded_magnetic_clauses`, clause for clause, with the
    source seam's polarity flipped: an ELECTRIC source is injected between step_D and
    update_E (driver.py:3294-3299), a magnetic one is not. That flip is what takes
    this family from 45 rows on the B half to 2 on the D half.
    """
    configuration = row["configuration"]
    metallic, shape = configuration["metallic"], configuration["shape"]
    types = configuration.get("source_field_types") or []
    folded = [axis for axis in range(3) if configuration["mirrored"][axis]]
    return {
        # the folded curl half, on step_D
        "curl_half": covered(row, "folded_composition_curl@step_D"),
        # the folded constitutive half, on E
        "constitutive_half": covered(row, "folded_constitutive@update_E"),
        # the ELECTRIC source seam (driver.py:3294-3299)
        "no_electric_source": all(str(kind) == "B" for kind in types),
        # a folded PERIODIC axis puts fill_folded_far_ghosts_D (driver.py:3302)
        # inside the seam
        "no_folded_periodic_axis": all(bool(metallic[a]) for a in folded),
        # the near fill images stored cell 2 from that cell's own lane
        "fold_stores_the_source_row": all(int(shape[a]) > NEAR_SOURCE_INDEX
                                          for a in folded),
        # the kernel bakes the plain constitutive product, whose source is D and not
        # (D - sum P)
        "no_susceptibility": int(configuration.get("n_polarizations") or 0) == 0,
    }


def cylindrical_real_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``cylindrical_real_fused_magnetic_pair_coverage``.

    The cell with NO ATTRITION on this board: the two halves and the magnetic source
    seam all admit the same three rows. The fold clauses are the two fill passes'
    inertness, which the cylindrical curl predicate already requires — restated here
    because this product's REPLACES tuple names three driver passes and not five.
    """
    configuration = row["configuration"]
    types = configuration.get("source_field_types") or []
    return {
        # the cylindrical m = 0 curl half, on step_B
        "curl_half": covered(row, "cylindrical_curl@step_B"),
        # the cylindrical constitutive half (the ORDINARY kernel under a Dcyl-aware
        # predicate), on H
        "constitutive_half": covered(row, "cylindrical_constitutive@update_H"),
        # the MAGNETIC source seam (driver.py:3283-3284)
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # both fill passes must stay inert: they return unless grid.has_symmetry()
        "no_fold": (not bool(configuration.get("has_symmetry"))
                    and not any(bool(v) for v in configuration["mirrored"])),
    }


def nonlinear_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``nonlinear_fused_magnetic_pair_coverage``.

    ADMISSION ONLY — the kernel is the SHIPPED ``kernels.fused_curl_constitutive_B``.
    The predicate is the conjunction of the two shipped nonlinear SPINE arms plus
    ``coverage.fused_pair_coverage`` through a one-clause scope view, so the clause
    ladder here is: the two half predicates, the magnetic source seam, and the two
    symmetry passes' inertness. THE SCOPE VIEW ADDS NO CLAUSE OF ITS OWN — it
    inverts ``coverage._grid_reasons`` clause 10 (coverage.py:195-196) and forwards
    everything else, and the two spine predicates already restate every one of
    those clauses in force, which is why this ladder can be read off the census's
    own per-slot verdicts.

    ``fused_pair_coverage``'s wall-readability clause is NOT a rung: it asks
    whether the grid exposes ``has_metallic``/``is_metallic``/``is_mirrored``, which
    every engine-built ``Grid`` does and no census row can answer differently. It is
    listed in :data:`CLAUSES_NOT_EVALUABLE_FROM_THE_CENSUS_BLOCK`.
    """
    configuration = row["configuration"]
    types = configuration.get("source_field_types") or []
    return {
        # the CERTIFIED real PML curl under the nonlinear-scope admission, step_B
        "curl_half": covered(row, "nonlinear_run_pml_curl@step_B"),
        # the CERTIFIED constitutive kernel under the same admission, side H
        "constitutive_half": covered(row, "nonlinear_run_constitutive@update_H"),
        # the MAGNETIC source seam (driver.py:3283-3284)
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # both fill passes must stay inert: they return unless grid.has_symmetry()
        "no_fold": (not bool(configuration.get("has_symmetry"))
                    and not any(bool(v) for v in configuration["mirrored"])),
    }


def beta_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``beta_fused_magnetic_pair_coverage``.

    The conjunction of the two shipped REAL-beta arms plus the seam clauses. The
    beta family's clause 12 is INVERTED inside ``beta_pml_curl_coverage`` itself
    (``grid.beta`` must be nonzero), so it needs no rung here: a beta = 0 row fails
    ``curl_half`` and ``constitutive_half`` together.
    """
    configuration = row["configuration"]
    types = configuration.get("source_field_types") or []
    return {
        # the REAL special_kz curl half, on step_B
        "curl_half": covered(row, "beta_pml_curl@step_B"),
        # the CERTIFIED real constitutive kernel under the beta admission, side H
        "constitutive_half": covered(row, "beta_run_constitutive@update_H"),
        # the MAGNETIC source seam (driver.py:3283-3284)
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # both fill passes must stay inert
        "no_fold": (not bool(configuration.get("has_symmetry"))
                    and not any(bool(v) for v in configuration["mirrored"])),
    }


def bfast_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``bfast_fused_magnetic_pair_coverage``.

    The conjunction of the two shipped BFAST arms plus the seam clauses. The BFAST
    family's clause 11 is INVERTED inside ``bfast_pml_curl_coverage`` itself
    (``grid.bfast_active`` required), so it needs no rung here: a shear-free row
    fails ``curl_half`` and ``constitutive_half`` together. The ``is_invariant``
    readability clause the weld adds is not evaluable from the census block and is
    named in :func:`caveats`.
    """
    configuration = row["configuration"]
    types = configuration.get("source_field_types") or []
    return {
        # the BFAST curl half, on step_B
        "curl_half": covered(row, "bfast_pml_curl@step_B"),
        # the CERTIFIED real constitutive kernel under the BFAST admission, side H
        "constitutive_half": covered(row, "bfast_run_constitutive@update_H"),
        # the MAGNETIC source seam (driver.py:3283-3284)
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # both fill passes must stay inert
        "no_fold": (not bool(configuration.get("has_symmetry"))
                    and not any(bool(v) for v in configuration["mirrored"])),
    }


def chain_clauses(row: dict) -> Dict[str, bool]:
    """``fused_dispersive_chain_coverage`` (fused_dispersive_chain.py:389)."""
    configuration = row["configuration"]
    states = row.get("polarization") or []
    return {
        # :405 the pair half — the SHIPPED seam predicate, called with real sources
        "pair_half": seam_covered(row, "dispersive_fused_pair_D"),
        # :380-386 exactly one susceptibility
        "exactly_one_susceptibility": len(states) == 1,
        # :413 the ADE half, per the single state
        "ade_half": (len(states) == 1
                     and bool(states[0].get("fused", {}).get(
                         "covered_modulo_backend",
                         states[0].get("fused", {}).get("covered")))),
        # :443-447 drive_field returns f_w only under an active layer
        "pml_active": bool(configuration.get("pml_active")),
    }


def ade_state_clauses(row: dict) -> Dict[str, bool]:
    """``fused_ade_state_coverage``, rolled up the way ``plan_step`` rolls it up.

    launch.py:3080-3086 — ONE refused state keeps the whole ``update_P`` sub-step
    on the array path, because the driver advances every polarization in one pass.
    """
    states = row.get("polarization") or []
    return {
        "a_state_is_registered": bool(states),
        "every_state_admits": bool(states) and all(
            bool(s.get("fused", {}).get("covered_modulo_backend",
                                        s.get("fused", {}).get("covered")))
            for s in states),
    }


def cylindrical_complex_magnetic_clauses(row: dict) -> Dict[str, bool]:
    """``cylindrical_fused_magnetic_pair_coverage``
    (cylindrical_fused_magnetic_pair.py:729). Carried verbatim from
    ../fusion_matrix_triton_2026-08-20c/.

    THE |m| >= 1 Dcyl CELL. The predicate is a conjunction of the two halves' own
    verdicts plus two seam clauses; the Dcyl geometry clauses, the |m| >= 1 clause,
    the ``nr >= 2`` clause and the EXPANSION licence are all INHERITED from the
    halves (:738-741) and are therefore already inside ``curl_half`` here rather
    than restated as separate rungs.

    THE 08-20c CUT DECLARED THE WALL CLAUSES NOT EVALUABLE AND THAT WAS TOO
    CAUTIOUS, IN THE PRODUCT'S FAVOUR. ``zero_metal_axes`` is ``is_metallic(a) and
    not is_mirrored(a)`` (coverage.py:408-421) and the census configuration block
    records BOTH ``metallic`` and ``mirrored``, so the FORBIDDEN_WALL_AXES clause IS
    evaluable from the record and is evaluated below rather than deferred. The
    remaining not-evaluable clause is only the ACCESSOR-EXISTENCE one (that the grid
    object answers has_metallic / is_metallic / is_mirrored at all), which can only
    remove rows.
    """
    configuration = row["configuration"]
    types = configuration.get("source_field_types") or []
    metallic = [bool(v) for v in configuration["metallic"]]
    mirrored = [bool(v) for v in configuration["mirrored"]]
    return {
        # :744-746 the Dcyl complex curl half, on step_B
        "curl_half": covered(row, "cylindrical_complex_curl@step_B"),
        # :747-751 the constitutive half. It is the ORDINARY complex kernel under a
        # Dcyl-aware predicate: stepping.update_H (:907-925) carries NO cylindrical
        # branch, and the axis-zero passes are called from the CURL only.
        "constitutive_half": covered(row, "cylindrical_complex_constitutive@update_H"),
        # :753-772 the MAGNETIC source seam (driver.py:3283-3284). MEASURED COST ON
        # THIS CORPUS: ZERO — every cylindrical row declares electric sources only.
        "no_magnetic_source": all(str(kind) != "B" for kind in types),
        # :774-792 both symmetry passes must stay inert: fill_symmetry_bc_B
        # (driver.py:3285) and fill_folded_far_ghosts_B (:3287) return at their
        # first line unless grid.has_symmetry() (stepping.py:1481-1482, :1565-1566).
        "no_fold": (not bool(configuration.get("has_symmetry"))
                    and not any(bool(v) for v in configuration["mirrored"])),
        # :804-820 FORBIDDEN_WALL_AXES — zero_metal_axes must report no wall on r or
        # phi. zero_metal_axes(a) is `is_metallic(a) and not is_mirrored(a)`
        # (coverage.py:408-421), evaluated here on axes 0 and 1 from the census's own
        # metallic/mirrored triples. NOT a deferral: this is the clause itself.
        "no_wall_on_r_or_phi": not any(metallic[a] and not mirrored[a]
                                       for a in (0, 1)),
    }


def in_seam_source(row: dict, seam: str) -> bool:
    """Does the driver inject a source INSIDE this seam on this row?

    THE CEILING ON EVERY FUSED PRODUCT AT A SEAM, and it is a fact about the
    driver rather than about any product: the magnetic sources are injected at
    driver.py:3283-3284, between ``step_B`` and ``update_H``; the electric ones at
    :3294-3299, between ``step_D`` and ``update_E``. A launch spanning either
    seam would consume a pre-injection field. So no product for ANY (curl,
    constitutive) cell at that seam — built, unbuilt or imagined — can serve a row
    that carries one, and the three shipped predicates that ask this
    (coverage.py:474-487, folded_fused_magnetic_pair.py:584-590,
    complex_fused_magnetic_pair.py:625-631) are transcribing the driver, not
    choosing a scope.

    ``E->P`` CARRIES NO IN-SEAM SOURCE AT ALL, and the 2026-08-20 cuts of this
    script said otherwise. CORRECTED HERE, from the driver rather than from a
    product. ``driver.step`` runs::

        if fast is None or not fast.dispatch("update_E", self.fields):
            update_E(self.fields, self.pml)                    # driver.py:3303-3304
        if fast is None or not fast.dispatch("update_P", self.fields):
            update_P(self.fields, self.pml)                    # driver.py:3305-3306

    Nothing is between them — no injection, no symmetry fill, no wall clear, no
    far-ghost pass. The electric deposit at :3294-3299 lands BEFORE ``update_E``,
    so a launch spanning ``update_E -> update_P`` never straddles it.

    WHAT THE EARLIER CUTS WERE ACTUALLY MEASURING, and why the number was not
    nonsense: the only Triton product they knew at this seam is
    ``fused_dispersive_chain``, which spans THREE sub-steps — ``step_D ->
    update_E -> update_P`` (fused_dispersive_chain.py:405, conjoining
    ``dispersive_fused_pair_coverage``). The electric injection IS inside THAT
    PRODUCT'S span. But that is a clause of one product, not the ceiling of this
    seam, and putting it here made an E->P-only product unscoreable before it was
    written. The clause still binds ``fused_dispersive_chain`` — it reaches this
    matrix through that product's own ``chain_clauses``, which is where a product
    clause belongs.

    The Metal matrix has measured this seam at CEILING 15 since its first cut
    (``fusion_matrix_metal_2026-08-20/``: "E_to_P: 15 instances, 0 blocked by the
    source seam"). With this correction the two backends' TOTAL ceilings agree at
    172 over one corpus, which they did not before.
    """
    types = [str(kind) for kind in (row["configuration"].get("source_field_types") or [])]
    if seam == "B->H":
        return any(kind == "B" for kind in types)
    if seam == "E->P":
        return False
    return any(kind != "B" for kind in types)


def _folds(configuration: dict) -> List[int]:
    return [axis for axis in range(3) if configuration["mirrored"][axis]]


def _deposit_is_repairable(pair: str, configuration: dict) -> bool:
    """``deposit_repair.repairable``'s own refusals, read from the census block.

    PORTED VERBATIM IN SUBSTANCE FROM THE METAL BOARD (build_fusion_matrix.py:562-595)
    because ``meep_gpu/deposit_repair.py`` is SHARED -- one module, both backends -- so
    a second, differently worded transcription of the same clauses is how two boards
    end up disagreeing about one predicate.

    TWO REFUSALS ARE D-SIDE ONLY. An off-diagonal chi1inv row makes ``update_E`` a
    stencil over the PARTNER components' volumes at shifted indices
    (stepping.py:1228-1251), so no repair over the deposit points can reconstruct it;
    an instantaneous chi2/chi3 is not the linear accumulation the repair inverts.

    TWO ARE ABOUT THE FOLD and apply to either seam: the cylindrical r = 0 axis, whose
    below-axis ghost is the r_to_minus_r image rather than the near fill's
    cell 0 <- cell 2, and a folded axis storing no more than
    :data:`NEAR_SOURCE_INDEX` cells, where the row the near fill images does not exist.

    The remaining refusal -- ``f_w_<component>`` unallocated -- needs an engine object
    and is named in :func:`caveats`, so every count this clause admits is an UPPER
    BOUND, which is what this whole board already is.
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


def in_seam_source_blocks(row: dict, seam: str) -> bool:
    """Does this seam's injection put the instance BEYOND ANY fused product?

    THE CEILING CLAUSE, AND IT STOPPED BEING "is there a source" ON 2026-08-28.
    :func:`in_seam_source` is a fact about the driver and is unchanged. What changed
    is the consequence: ``deposit_repair`` lets a product save the two arrays at the
    sparse deposit points before the fused launch and recompute them after, so an
    in-seam injection is no longer a structural bar -- it is a bar only where
    ``repairable`` itself refuses. The allowlist that declares
    ``CARRIES_DEPOSIT_REPAIR`` held eight products on 2026-08-28, three of them
    Triton's (``fused_pair_B``, ``fused_pair_D``, ``dispersive_fused_pair``); on
    2026-08-30 it gained three more Triton families —
    ``folded_fused_pair``, ``folded_fused_magnetic_pair`` and
    ``complex_fused_magnetic_pair`` — once their device gates were re-run against
    the flipped bytes. The live set is pinned by
    ``test_fused_pair_deposit_wiring.WIRED_FOR_THE_REPAIR``, which is an equality in
    both directions, so this sentence is a summary of that list and not a second copy
    of it.

    THIS IS A CEILING, so it does NOT ask whether the product covering the cell
    carries the repair -- the ceiling is what SOME product could serve. Whether a
    particular product clears the seam is its shipped predicate's answer, which this
    board reads from the census rather than re-deriving.

    MEASURED, NOT ASSUMED, and the measurement is why this function exists: on the
    fresh 2026-08-28 census the shipped seam predicates ADMIT 51 instances that carry
    an in-seam source (22 fused_pair_B, 26 fused_pair_D, 3 dispersive_fused_pair).
    Under the old clause every one of those was bucketed "blocked by the source
    injection", so the board's own bucket table said 99 served while its aggregate
    said 150 -- one board, two answers.
    """
    if not in_seam_source(row, seam):
        return False
    pair = "B" if seam == "B->H" else "D"
    return not _deposit_is_repairable(pair, row["configuration"])


def specialized_family_owns_the_grid(row: dict) -> bool:
    """launch.py:2860-2866 — the guard that disables OPT-IN pair fusion by name."""
    configuration = row["configuration"]
    row_product_arms = {"off-diagonal", "folded off-diagonal",
                        "folded off-diagonal dispersive", "complex folded off-diagonal",
                        "complex no-PML off-diagonal"}          # launch.py:2184-2191
    return bool(
        configuration["force_complex_fields"]
        or float(configuration["beta"] or 0.0)
        or configuration["bfast_active"]
        or configuration["has_nonlinearity"]
        or configuration["has_offdiagonal_epsilon"]
        or any(a in row_product_arms for a in admitted(row, "update_E")))


#: The products ``plan_step`` reaches from its FOLDED branch. Read as a set rather
#: than as a per-product flag because the branch installs them together
#: (``launch._install_folded_fused_pairs``) and neither is reachable off a fold.
FOLDED_BRANCH_PRODUCTS = ("folded_fused_magnetic_pair", "folded_fused_pair")

#: The two products ``launch.py`` builds ITSELF (``plan_fused_pair``) rather than
#: importing from a sibling module. Named so the wiring measurement below can say
#: why it does not look for a module, instead of silently special-casing them.
IN_LAUNCH_PRODUCTS = ("fused_pair_B", "fused_pair_D")


def fusion_block_reached(row: dict, product: str) -> Tuple[bool, str]:
    """Which of ``plan_step``'s four ``fuse`` branches does this row take?

    THE FOLD IS NOT A BLANKET REFUSAL ANY MORE, and this function asserted that it
    was until 2026-08-29 — it answered ``has_fold: launch.py refuses every pair by
    name`` for EVERY product on all 86 folded rows, which is the pre-2026-08-27
    answer. ``launch.py:3033`` now routes the fold to
    ``_install_folded_fused_pairs`` and refuses only the ORDINARY pairs there,
    whose kernels carry no mirror fill. Measured rather than argued in two
    independent ways: ``launch.py``'s parse tree imports both folded modules
    (:func:`launch_imported_modules`), and the driver-route gate saw ``fused pair B
    (folded)`` selected at ``step_B`` and ``update_H`` on a real device
    (``results/dispatch_fused_route_2026-08-30_bind/``, folded_2d, and the
    superseded ``_veto`` and ``_seamgate`` runs before it) before rung (6) refused
    the plan for a different reason.

    THE REFUSALS THAT SURVIVE ARE NARROWER, not gone: on a fold the two folded
    products are reached and every other pair is refused BY NAME, which is what
    ``launch.py:3033-3043`` does.
    """
    configuration = row["configuration"]
    folded = any(bool(v) for v in configuration["mirrored"])
    if folded:
        if product in FOLDED_BRANCH_PRODUCTS:
            return True, ""
        return False, ("has_fold: launch.py:3033 routes the fold to the two folded "
                       "pairs and refuses every other pair by name — this product's "
                       "kernel carries no mirror fill")
    if product in FOLDED_BRANCH_PRODUCTS:
        return False, ("no fold: launch.py:3033 reaches the folded pairs only from "
                       "the folded branch")
    if configuration["cylindrical"]:
        return False, "has_cylindrical: launch.py:3045 refuses every pair by name"
    if specialized_family_owns_the_grid(row):
        return False, "a specialized family owns the grid: launch.py:3054"
    return True, ""


def launch_imported_modules() -> Tuple[str, ...]:
    """Every sibling ``triton_kernels`` module ``launch.py`` imports, from its AST.

    WHY MEASURED AND NOT DECLARED. This column used to read a hardcoded
    ``PRODUCTS[...]["wired"]`` flag whose refusal text was "not an arm: launch.py
    never imports the module". That sentence was TRUE when it was written and
    stopped being true on 2026-08-27, when the two folded pairs were routed — and a
    hardcoded flag cannot notice, so the board went on publishing
    ``planner_reachable: 0`` for 56 seam-instances the planner reaches. A board
    column that pins a pre-flip answer keeps printing it after the flip.

    AST, NOT A SUBSTRING SEARCH: every one of these module names appears in
    ``launch.py``'s prose as well, and a grep would score a sentence saying a module
    is NOT reached as a reach. Function-body imports count — the family modules are
    imported lazily on purpose (``launch.py`` FAMILY_MODULES/SUPPORT_MODULES), so a
    module-scope-only reading would call every one of them unreachable.
    """
    source = (API / "meep_gpu" / "triton_kernels" / "launch.py").read_text()
    found: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            found.extend(alias.name.split(".")[-1] for alias in node.names)
    return tuple(sorted(set(found)))


def assert_the_wiring_column_is_measured() -> Dict[str, Any]:
    """The declared ``wired`` flag must be what ``launch.py`` actually does.

    BOTH DIRECTIONS, because both failures are wrong numbers with opposite signs: a
    product declared unwired that ``launch.py`` DOES import is under-counted in
    column (c) (what happened to the two folded pairs for two days), and one
    declared wired that it does NOT import is over-counted, which would credit the
    planner with composing something it cannot reach. Raises rather than reporting:
    a column nothing checks is the defect this function exists to close.
    """
    imported = set(launch_imported_modules())
    disagreements: List[str] = []
    measured: Dict[str, bool] = {}
    for name, spec in PRODUCTS.items():
        # ``fused_pair_B``/``fused_pair_D`` are built by ``launch.py`` itself
        # (``plan_fused_pair``), so there is no sibling module to look for; their
        # module field names ``coverage.py``/``launch.py`` and they are wired by
        # construction. Every other product lives in its own module.
        module = spec["module"].split(".py")[0].split("/")[0].strip()
        own_module = name not in IN_LAUNCH_PRODUCTS
        reached = (module in imported) if own_module else True
        measured[name] = reached
        if reached != bool(spec["wired"]):
            evidence = (f"launch.py {'imports' if reached else 'does not import'} "
                        f"{module!r}" if own_module else
                        "launch.py BUILDS this product itself (plan_fused_pair), so "
                        "it is wired by construction and cannot be declared unwired")
            disagreements.append(
                f"{name}: declared wired={spec['wired']}, but {evidence}")
    if disagreements:
        raise SystemExit(
            "the planner-wiring column disagrees with launch.py's parse tree, so "
            "column (c) is reporting a composition the planner does not do (or "
            "missing one it does):" + "".join(f"\n    {d}" for d in disagreements))
    return {"launch_imports": sorted(imported & {m for m in measured}),
            "measured": measured,
            "read_from": "meep_gpu/triton_kernels/launch.py, parse tree",
            "products_wired": sorted(n for n, v in measured.items() if v)}


def assert_the_chain_arms_do_not_overlap(record: List[dict]) -> Dict[str, Any]:
    """``fused_ade_chain``'s three arms must admit DISJOINT rows.

    WHY THIS IS ASSERTED AND NOT ASSUMED. This board credits the family when ANY arm
    admits (:data:`E_TO_P_VERDICT_KEYS`). That is sound only because the caller names
    the arm — but if two arms admitted one row, "any" would be hiding a genuine
    question about WHICH launch serves it, and a reader would have no way to tell
    from the total. Two admitters is exactly the condition ``_select_slot``
    (launch.py:1983-1996) treats as a refusal for an arm table, so an overlap here
    would also be the thing that keeps a slot on the array path if this family were
    ever routed.

    Raises rather than reporting: a number whose derivation stopped being valid must
    not be published with a footnote.
    """
    overlaps = []
    for row in record:
        hits = [key for key in E_TO_P_VERDICT_KEYS["fused_ade_chain"]
                if seam_covered(row, key)]
        if len(hits) > 1:
            overlaps.append({"row": label(row), "arms": hits})
    if overlaps:
        raise SystemExit(
            "fused_ade_chain admits more than one arm on "
            f"{len(overlaps)} row(s), so 'any arm admits' is hiding a choice this "
            f"board does not make: {overlaps[:5]}")
    return {"rows_checked": len(record), "rows_with_two_admitting_arms": 0,
            "arms": list(E_TO_P_VERDICT_KEYS["fused_ade_chain"])}


def assert_at_most_one_e_to_p_product_admits(record: List[dict],
                                             admits_by_row: Dict[str, Dict[str, bool]]
                                             ) -> Dict[str, Any]:
    """The E->P branch picks ``e_to_p_admitters[0]``; that must not be a choice.

    The chains' clauses are disjoint by construction on TWO axes. STORAGE:
    ``fused_ade_chain`` refuses ``force_complex_fields=True`` on every arm and
    ``complex_fused_ade_chain`` REQUIRES it. BODY: ``folded_offdiag_fused_ade_chain``
    composes the folded off-diagonal dispersive E half, which requires a live mirror
    plane AND a surviving off-diagonal chi1inv row AND an active absorber —
    ``fused_ade_chain`` refuses that configuration on all three of its arms (the row
    on ``folded``, the mirror on ``dispersive``, the layer on ``no_pml``), and it is
    real storage, so the complex chain refuses it too.

    Re-earned per cut over all three, because "by construction" is how the folded
    periodic mirror stayed wrong for nine days.
    """
    clashes = [{"row": name, "products": [p for p in E_TO_P_PRODUCTS if verdicts[p]]}
               for name, verdicts in admits_by_row.items()
               if sum(bool(verdicts[p]) for p in E_TO_P_PRODUCTS) > 1]
    if clashes:
        raise SystemExit(
            f"{len(clashes)} row(s) are admitted by more than one E->P product, so "
            f"the branch's first-match pick is arbitrary: {clashes[:5]}")
    return {"rows_checked": len(admits_by_row),
            "rows_with_two_admitting_products": 0,
            "products": list(E_TO_P_PRODUCTS)}


def planner_reachable(row: dict, product: str) -> Tuple[bool, str]:
    """Could ``plan_step(fuse=True)`` actually put this product in the plan?"""
    if not PRODUCTS[product]["wired"]:
        return False, ("not an arm: launch.py never imports the module "
                       "(measured by assert_the_wiring_column_is_measured, which "
                       "reads launch.py's parse tree rather than a declaration)")
    if PRODUCTS[product]["seam"] == "E->P":
        # REFUSED BY NAME rather than answered wrongly. The two slot names below are
        # a B->H / D->E mapping, and an E->P product owns update_E -> update_P; a
        # `wired` E->P product would otherwise be measured against the wrong pair of
        # slots and reported reachable or unreachable for a reason about a different
        # seam. Unreachable today for the flag above, so this rung is a guard on the
        # next round rather than a live branch.
        return False, ("E->P reachability is not derivable from _pair_may_absorb: "
                       "that clause reads the curl/constitutive slot pair, and this "
                       "product owns update_E -> update_P")
    reached, why = fusion_block_reached(row, product)
    if not reached:
        return False, why
    curl_slot, update_slot = (("step_B", "update_H") if PRODUCTS[product]["seam"] == "B->H"
                              else ("step_D", "update_E"))
    want_curl, want_update = PRODUCTS[product]["cell"]
    # launch.py:2084 _pair_may_absorb — the pair may take a slot only when the arm
    # table gave that slot to the arm the fused kernel implements.
    if arm(row, curl_slot) != want_curl:
        return False, f"_pair_may_absorb: {curl_slot} arm is {arm(row, curl_slot)!r}"
    if arm(row, update_slot) != want_update:
        return False, f"_pair_may_absorb: {update_slot} arm is {arm(row, update_slot)!r}"
    if product == "dispersive_fused_pair" and seam_covered(row, "fused_pair_D"):
        # launch.py:2916-2921 — both admitting keeps the separate sub-step plans.
        return False, "both the ordinary and dispersive D/E predicates admitted"
    return True, ""


# ---------------------------------------------------------------------------
# The staleness audit — measured, not asserted
# ---------------------------------------------------------------------------

def function_digests(source: str) -> Dict[str, str]:
    """sha256 over each top-level+nested function's AST with its docstring stripped."""
    tree = ast.parse(source)
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        clone = ast.parse(ast.unparse(node)).body[0]
        body = getattr(clone, "body", [])
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            clone.body = body[1:]
        out[node.name] = hashlib.sha256(ast.dump(clone).encode()).hexdigest()[:16]
    return out


#: Every predicate whose verdict this derivation reads, and the module it lives in.
AUDITED: Tuple[Tuple[str, str], ...] = (
    ("coverage.py", "pml_curl_coverage"),
    ("coverage.py", "constitutive_coverage"),
    ("coverage.py", "fused_pair_coverage"),
    ("coverage.py", "ade_update_p_coverage"),
    ("dispersive_fused_pair.py", "dispersive_fused_pair_coverage"),
    ("dispersive_update_e.py", "dispersive_constitutive_coverage"),
    ("fused_ade_state.py", "fused_ade_state_coverage"),
    ("symmetry.py", "folded_composition_curl_coverage"),
    ("symmetry.py", "folded_constitutive_coverage"),
    ("complex_fields.py", "complex_pml_curl_coverage"),
    ("complex_fields.py", "complex_constitutive_coverage"),
    ("complex_fields.py", "_complex_grid_reasons"),
    ("complex_fields.py", "_complex_layout_reasons"),
    ("complex_fields.py", "_expansion_reasons"),
    # Added 2026-08-20c with the cylindrical-complex product. The 08-20b cut READ
    # the two cylindrical m = 0 predicates for its own new product without adding
    # them here, so its staleness audit did not in fact cover every predicate its
    # derivation read; both pairs are audited from that cut on.
    ("cylindrical_complex.py", "cylindrical_complex_curl_coverage"),
    ("cylindrical_complex.py", "cylindrical_complex_constitutive_coverage"),
    ("cylindrical_triton.py", "cylindrical_curl_coverage"),
    ("cylindrical_triton.py", "cylindrical_constitutive_coverage"),
    # Added in the CLOSED cut with the folded electric pair, for the same reason:
    # the 08-20b cut read folded_composition_curl_coverage at step_D and
    # folded_constitutive_coverage('E') without auditing either. Both names already
    # appear above (they are the same two functions the B half reads), so this is
    # noted rather than duplicated.
    #
    # ADDED 2026-08-30, and the reason is the rule this tuple exists to enforce:
    # this cut now reads the verdict of SEVEN more shipped predicates off the census
    # rather than transcribing them, and a predicate whose verdict is read but whose
    # bytes are unaudited is exactly the credit that rests on nothing.
    ("folded_fused_pair.py", "folded_fused_pair_coverage"),
    ("folded_fused_magnetic_pair.py", "folded_fused_magnetic_pair_coverage"),
    ("complex_fused_magnetic_pair.py", "complex_fused_magnetic_pair_coverage"),
    ("folded_complex_fused_magnetic_pair.py",
     "folded_complex_fused_magnetic_pair_coverage"),
    ("fused_ade_chain.py", "fused_ade_chain_coverage"),
    ("fused_ade_chain.py", "_ade_coverage"),
    # ADDED 2026-08-31 with the folded dispersive D->E weld: this cut reads that
    # product's verdict off the census, and its E half's — the composed admission
    # `folded_dispersive_update_e` owns. A predicate whose verdict is read but whose
    # bytes are unaudited is exactly the credit that rests on nothing.
    ("folded_dispersive_fused_pair.py", "folded_dispersive_fused_pair_coverage"),
    ("folded_dispersive_update_e.py", "folded_dispersive_constitutive_coverage"),
    # ...and the Dcyl m = 0 electric weld's, whose two halves are already audited
    # above (``cylindrical_triton``'s pair, which the magnetic twin reads).
    ("cylindrical_real_fused_electric_pair.py",
     "cylindrical_real_fused_electric_pair_coverage"),
    # ...and the folded complex electric weld's, whose three halves are already
    # audited above (``folded_complex``'s curl, constitutive and fill, which the
    # magnetic twin reads).
    ("folded_complex_fused_pair.py", "folded_complex_fused_pair_coverage"),
    # ...and the real-beta electric weld's, whose two halves are already audited
    # through the magnetic twin's ladder (``special_kz``'s pair).
    ("beta_fused_electric_pair.py", "beta_fused_electric_pair_coverage"),
    # ...and the BFAST electric weld's, whose two halves are already audited through
    # the magnetic twin's ladder (``bfast_curl``'s pair).
    ("bfast_fused_electric_pair.py", "bfast_fused_electric_pair_coverage"),
    # ...and the CONDUCTIVE PML electric weld's, whose curl half is audited through
    # ``conductivity``'s own ladder and whose constitutive half is the ordinary
    # ``coverage.constitutive_coverage`` every fused pair in this table already
    # routes through.
    ("conductive_fused_electric_pair.py",
     "conductive_fused_electric_pair_coverage"),
    # ...and the NO-ABSORBER STORED-E weld's — added 2026-08-31 in the folded-beta
    # round, closing a gap the interrupted plain-branch round left: its verdict has
    # been read off the census since the plainrepair cut while its bytes were
    # audited by nothing, which is exactly the credit-resting-on-nothing this tuple
    # exists to refuse.
    ("no_pml_fused_electric_pair.py", "no_pml_fused_electric_pair_coverage"),
    # ...and the TWO FOLDED-BETA welds' (2026-08-31, this cut), plus the two ARM
    # predicates they conjoin, which no earlier cut read for a shipped product:
    # K3a's curl verdict and the folded beta run constitutive verdict both live in
    # folded_complex.py.
    ("folded_beta_fused_electric_pair.py",
     "folded_beta_fused_electric_pair_coverage"),
    ("folded_beta_fused_magnetic_pair.py",
     "folded_beta_fused_magnetic_pair_coverage"),
    ("folded_complex.py", "folded_beta_pml_curl_coverage"),
    ("folded_complex.py", "folded_beta_run_constitutive_coverage"),
    ("complex_fused_ade_chain.py", "complex_fused_ade_chain_coverage"),
    # ...and the THIRD ADE chain's, with the round that built it: this cut reads
    # its verdict off the census, and the E half it conjoins is already audited
    # two rows above (``folded_offdiag_dispersive_update_e``'s composed
    # admission, added with the folded dispersive D->E weld). A predicate whose
    # verdict is read but whose bytes are unaudited is exactly the credit that
    # rests on nothing.
    ("folded_offdiag_fused_ade_chain.py", "folded_offdiag_fused_ade_chain_coverage"),
    # The seam clause all four pair predicates route through, and the repair whose
    # verdict it now consults. `seam_source_reasons` is the function
    # CARRIES_DEPOSIT_REPAIR is passed to, and `repairable` is what decides an
    # in-seam deposit once the flag lets it be asked — so 2026-08-30's flip moved
    # the answer for 77 seam-instances into these two bodies. They live in
    # `deposit_repair.py`, one level up from `triton_kernels/`, which the audit
    # resolves by name.
    ("../deposit_repair.py", "seam_source_reasons"),
    ("../deposit_repair.py", "repairable"),
    ("../deposit_repair.py", "repair_cells"),
    ("../deposit_repair.py", "_fill_image_rules"),
)


def _census_provenance() -> Dict[str, Any]:
    """What the census says about itself, or ``{}`` if it says nothing.

    A census cut before 2026-08-28 carries no such file, and the absence is the
    signal that its predicate verdicts must be dated by a COMMIT.
    """
    path = CENSUS / "census_provenance.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def staleness_audit() -> Dict[str, Any]:
    """Has any predicate this derivation reads MOVED since the census called it?

    TWO SOURCES FOR "WHAT THE CENSUS CALLED", and the census picks which.

    A census cut at a released commit is dated by that commit, and the comparison is
    ``git show <commit>:<module>`` against the worktree -- what every Triton board
    before 2026-08-28 did, with :data:`CENSUS_COMMIT` hard-coded to ``47d6658``.

    A census cut against the WORKING TREE cannot be dated that way: this track is not
    committed, so ``git show HEAD:...`` returns bytes the census never saw and the
    audit would report CHANGED for predicates that never moved -- reading, to anyone
    scanning the JSON, as "the census is stale" precisely when it is fresh. So a
    worktree-cut census SNAPSHOTS ``triton_kernels`` beside itself, and this compares
    that snapshot against the tree. That is a strictly stronger check than the commit
    one: it also catches an edit made BETWEEN the census and the board, which a
    commit-dated audit cannot see at all.

    Either way the question, the digest function and the verdict vocabulary are the
    same, and the source is reported rather than implied.
    """
    log("\n=== STALENESS AUDIT: every predicate this derivation reads ===")
    provenance = _census_provenance()
    snapshot = CENSUS / "tree_snapshot" / "triton_kernels"
    use_snapshot = provenance.get("cut_against") == "WORKTREE" and snapshot.is_dir()
    if use_snapshot:
        log(f"census cut against the WORKING TREE at git {provenance.get('git_head', '?')[:7]}"
            f"; comparing its own snapshot of triton_kernels against the tree now")
        if not provenance.get("snapshot_taken_before_any_kernel_edit"):
            raise SystemExit(
                f"{CENSUS}'s provenance does not assert that its tree snapshot was "
                f"taken before any kernel edit, so the snapshot is not known to be "
                f"the bytes the census called and this audit would prove nothing")
    else:
        log(f"census commit {CENSUS_COMMIT} vs the worktree "
            f"(the only triton_kernels commit since is 31c9e18)")
    verdicts: Dict[str, str] = {}
    for index, (module, function) in enumerate(AUDITED, start=1):
        # A ``module`` may sit one level ABOVE the package — ``deposit_repair.py``
        # does, and this cut reads two of its functions. The paths are normalised
        # here rather than special-cased at each use: both the snapshot and the tree
        # resolve ``..`` natively, and git does NOT, so the git spelling is the one
        # that has to be flattened.
        tree_path = (API / "meep_gpu" / "triton_kernels" / module).resolve()
        git_path = os.path.relpath(tree_path, REPO)
        if use_snapshot:
            source = (snapshot / module).resolve()
            if not source.exists():
                verdicts[f"{module}:{function}"] = "ABSENT FROM THE CENSUS SNAPSHOT"
                log(f"  [{index:2d}/{len(AUDITED)}] {module}:{function:38s} "
                    f"ABSENT FROM THE CENSUS SNAPSHOT")
                continue
            old = source.read_text()
        else:
            try:
                old = subprocess.run(
                    ["git", "show", f"{CENSUS_COMMIT}:{git_path}"],
                    cwd=REPO, capture_output=True, text=True, check=True).stdout
            except subprocess.CalledProcessError as exc:
                verdicts[f"{module}:{function}"] = f"UNREADABLE AT {CENSUS_COMMIT}: {exc}"
                log(f"  [{index:2d}/{len(AUDITED)}] {module}:{function:38s} "
                    f"UNREADABLE AT {CENSUS_COMMIT}")
                continue
        new = tree_path.read_text()
        before, after = function_digests(old).get(function), function_digests(new).get(function)
        verdict = ("SAME" if before == after and before is not None
                   else f"CHANGED ({before} -> {after})")
        verdicts[f"{module}:{function}"] = verdict
        log(f"  [{index:2d}/{len(AUDITED)}] {module}:{function:38s} {verdict}")
    verdicts["_dated_by"] = ("the census's own tree snapshot" if use_snapshot
                             else f"git {CENSUS_COMMIT}")
    return verdicts


def recheck_the_expansion_licence() -> Dict[str, Any]:
    """The one CHANGED clause that could move a number: re-measure it today.

    ``_expansion_reasons`` and the whole ``expansion_license`` ladder were
    rewritten between the census and now. The census's complex verdicts are only
    usable if TODAY's code still licenses the artifact the census ran under, so
    that is measured here rather than argued.
    """
    log("\n=== RE-MEASURING THE COMPLEX EXPANSION LICENCE ON TODAY'S CODE ===")
    probe_path = None
    for row in rows(CENSUS):
        if row.get("probe_artifact"):
            probe_path, declared = row["probe_artifact"], row.get("run_policy_declared")
            break
    if probe_path is None:
        log("  the census recorded no probe artifact path")
        return {"probe_artifact": None}
    log(f"  artifact  : {probe_path}")
    log(f"  policy the census declared: {declared!r}")
    from meep_gpu import expansion_refusal                        # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fields as cf      # noqa: PLC0415
    record = cf.load_expansion_probe(str(API / probe_path))
    licence = cf.expansion_license(record)
    with expansion_refusal.declaring_run_policy(str(declared)):
        reasons = cf._expansion_reasons(record)
    log(f"  expansion_license(record) refusals : {licence.get('refusals')}")
    log(f"  expansion arm                      : {licence.get('arm')!r}")
    log(f"  _expansion_reasons under {declared!r}      : "
        f"{reasons if reasons else 'NONE — still licensed'}")
    return {
        "probe_artifact": probe_path,
        "policy_declared_by_the_census": declared,
        "licence_refusals_today": list(licence.get("refusals") or ()),
        "arm_today": licence.get("arm"),
        "expansion_reasons_today_under_that_policy": [str(r) for r in reasons],
        "still_licensed": not reasons,
    }


# ---------------------------------------------------------------------------
# The freshness check
# ---------------------------------------------------------------------------

#: THE E->P SEAM IS SCORED FROM A DIFFERENT RECORD, AND HERE IS WHY.
#:
#: `CENSUS` (2026-08-16) records `plan_step.slots` for step_B / step_D / update_E /
#: update_H and FOR NOTHING ELSE. `update_P` is in STEP_ORDER (launch.py:1874-1877)
#: and plan_step resolves an ADE family for it (launch.py:2984-3048), but that
#: census leg never walked the slot, so the record carries no column for either half
#: of this seam on the no-PML side and no column at all for the P side. Scoring E->P
#: from it can only ever return "half the pair is on the array path", which is a
#: fact about the record.
#:
#: `FRESH` (2026-08-20, the CUDA predicate battery) DOES carry both halves:
#: `cuda_ade.update_P` and `cuda_no_pml_ade.update_P` on the P side,
#: `cuda_dispersive.update_E`, `cuda_no_pml_dispersive.update_E`,
#: `cuda_folded_offdiag.update_E` and `cuda_complex_no_pml.update_E` on the E side.
#: It does NOT carry `source_field_types` — which is why B->H and D->E must stay on
#: the 08-16 record — and at THIS seam that absence costs nothing, because the E->P
#: seam has no in-seam source to ask about (see `in_seam_source`).
#:
#: So: B->H and D->E from the census, E->P from 08-20, each seam read from the record
#: that measures it, the split stated rather than buried, and the row-level
#: agreement between the two records asserted below rather than assumed.
#:
#: AMENDED 2026-09-04 — HALF OF THAT PREMISE EXPIRED. The premise above was written
#: against the 2026-08-16 census, which carried NEITHER column. The censuses this
#: board reads now (2026-09-03 and 2026-09-04) carry a full 21-arm `update_E` block
#: on every row and STILL no `update_P` column, so only the P half is read from the
#: CUDA battery. The E half moved onto the live census, because asking a CUDA family
#: whether a TRITON arm admits mispriced one instance outright (see the comment at
#: the `e_arm = arm(row, "update_E")` line). `_E_SIDE_BLOCKS` below is retained ONLY
#: as the per-instance control column `curl_arm_in_the_2026_08_20_cuda_battery`.
_E_SIDE_BLOCKS: Tuple[Tuple[str, str], ...] = (
    ("cuda_dispersive", "dispersive"),
    ("cuda_no_pml_dispersive", "no-PML dispersive"),
    ("cuda_folded_offdiag", "folded off-diagonal dispersive"),
    ("cuda_complex_no_pml", "complex no-PML"),
)
_P_SIDE_BLOCKS: Tuple[Tuple[str, str], ...] = (
    ("cuda_ade", "ADE update_P"),
    ("cuda_no_pml_ade", "no-PML ADE update_P"),
    ("cuda_complex_no_pml", "complex ADE update_P"),
)


def _fresh_covered(row: dict, block: str, slot: str) -> bool:
    """One CUDA-battery cell, with the NumPy-host clause discounted.

    The same convention `covered` uses on the 08-16 record: the battery ran on a
    host with no CuPy, so every predicate's RAW verdict is False for that reason
    alone and `covered_modulo_backend` is the column that carries the numerical
    answer.
    """
    entry = (row.get(block) or {}).get(slot) or {}
    return bool(entry.get("covered_modulo_backend", entry.get("covered")))


def _fresh_arm(row: dict, blocks: Tuple[Tuple[str, str], ...], slot: str
               ) -> Optional[str]:
    """The arm that would win a slot: exactly one admitter, else none.

    `_select_slot`'s own rule (launch.py:1983-1996) — two admitters leave the slot
    on the array path — applied to the CUDA battery's columns rather than to the
    08-16 record's `admitted` lists, because those lists do not exist for this seam.
    """
    labels = [name for block, name in blocks if _fresh_covered(row, block, slot)]
    return labels[0] if len(labels) == 1 else None


def freshness_check() -> Dict[str, Any]:
    log("\n=== FRESHNESS: the 2026-08-16 configurations vs the 2026-08-20 record ===")
    census, fresh = rows(CENSUS), rows(FRESH)
    key = lambda r: (r["leg"], r["row"])                          # noqa: E731
    fresh_by_key = {key(r): r for r in fresh}
    same_rows = {key(r) for r in census} == set(fresh_by_key)
    shared = sorted(set(census[0]["configuration"]) & set(fresh[0]["configuration"]))
    mismatch: Dict[str, int] = collections.Counter()
    for row in census:
        other = fresh_by_key.get(key(row))
        if other is None:
            continue
        for name in shared:
            if row["configuration"][name] != other["configuration"][name]:
                mismatch[name] += 1
    log(f"  rows: census {len(census)}, fresh {len(fresh)}, identical set: {same_rows}")
    log(f"  configuration keys compared: {len(shared)}")
    log(f"  rows differing on any shared key: {dict(mismatch) or 'NONE'}")
    log("  (source_field_types is in the TRITON record only — the CUDA battery does")
    log("   not record it — so the source-seam clauses are evaluated from the 08-16")
    log("   record, whose every other configuration key the 08-20 record confirms.)")
    return {"row_sets_identical": same_rows,
            "configuration_keys_compared": shared,
            "rows_differing_per_key": dict(mismatch)}


def licensed_vs_unlicensed() -> Dict[str, int]:
    log("\n=== WHY THE _convention CUT: the predicate deltas, measured ===")
    unl = {(r["leg"], r["row"]): r for r in rows(CENSUS_UNLICENSED)}
    lic = {(r["leg"], r["row"]): r for r in rows(CENSUS)}
    deltas: Dict[str, List[int]] = {}
    for name in sorted(next(iter(lic.values()))["predicates"]):
        before = sum(covered(unl[k], name) for k in unl)
        after = sum(covered(lic[k], name) for k in lic)
        if before != after:
            deltas[name] = [before, after]
            log(f"  {name:46s} {before:4d} -> {after:4d}")
    log(f"  predicates that moved: {len(deltas)} — every one a complex family")
    for name in ("fused_pair_B", "fused_pair_D", "dispersive_fused_pair_D"):
        before = sum(seam_covered(unl[k], name) for k in unl)
        after = sum(seam_covered(lic[k], name) for k in lic)
        log(f"  SEAM {name:41s} {before:4d} -> {after:4d}")
    return deltas


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

#: The Metal battery. READ HERE FOR ONE COLUMN ONLY: ``plan_step.live``, the
#: engine's own list of which driver passes run on a row. That is a DRIVER fact —
#: ``driver.step`` calls the same passes whatever backend fills the slots — and this
#: census does not record it, while the Metal one does, on the SAME rows (the join
#: in ``build_matrix`` asserts the two row sets are identical).
#:
#: THE ALTERNATIVE WAS TO DERIVE IT AND THE DERIVATION IS WRONG. The two 2026-08-20
#: carry cuts derived liveness from the boundary triple: near fill live iff some axis
#: is mirrored. Measured against ``plan_step.live`` on these 186 rows that rule
#: disagrees on ``fill_B`` on 32 rows and on ``fill_D`` on 85 — a near fill is also
#: live on an UNFOLDED row that carries a source — and both cuts published the
#: derived number. It agrees exactly on the two wall clears and both far-ghost
#: passes. This cut reads the record and prints the comparison rather than repeating
#: the derivation.
#: Selectable for the same reason CENSUS is: this file is canonical now, and the
#: liveness record is a DRIVER fact that a later Metal census re-measures on the same
#: rows. THE DEFAULT MOVED 2026-09-03 from ``metal_coverage_tranche6_2026-08-19`` — the
#: tranche every Triton board up to ``fusion_matrix_triton_2026-09-02_unified`` joined
#: against — to ``metal_coverage_2026-09-03_complete``, the Metal census cut on the
#: 194-row basis and the only liveness record holding the same rows as the default
#: :data:`CENSUS`; the join below refuses any other pairing by name. The 08-19
#: tranche stays selectable for reproducing a 186-row cut. Not
#: ``metal_coverage_2026-09-03_extended``: superseded the same day (its five
#: gdsii/sigma rows were evaluated where torch was absent).
#:
#: MOVED 2026-09-06 to ``metal_coverage_2026-09-04_m0complex``, the standing Metal
#: census (``build_fusion_matrix.CENSUS``), so the two boards read one Metal record.
#: The liveness fact (``plan_step.live``) is a DRIVER fact and the two records agree
#: on it row for row: the only difference between them is one row's arm SELECTION
#: (``examples:dipole_in_vacuum_cyl_off_axis.py``, the m = 0 complex arm), which the
#: join below does not read. The join's own floors — same rows, 0 drift on
#: mirrored/metallic/source_field_types — hold on the new pairing.
METAL_CENSUS = RESULTS / (os.environ.get("MEEP_GPU_METAL_LIVENESS_CENSUS")
                          or "metal_coverage_2026-09-04_m0complex")

IN_SEAM_PASSES: Dict[str, Tuple[str, str, str]] = {
    "B->H": ("fill_B", "zero_metal_B", "fill_folded_far_ghosts_B"),
    "D->E": ("fill_D", "zero_metal_D", "fill_folded_far_ghosts_D"),
}


def _derived_in_seam_passes(configuration: dict, seam: str) -> Tuple[str, ...]:
    """The 2026-08-20 carry cuts' DERIVED rule. Kept ONLY to measure it against the
    engine's recorded answer; never used to score a cell."""
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


def in_seam_liveness() -> Dict[str, set]:
    """``plan_step.live`` per row, off the Metal census, with the join ASSERTED.

    A FLOOR, not a comment: if the two records did not hold the same rows, or if the
    two configuration keys the driver's pass selection turns on disagreed anywhere,
    joining them would attribute one row's passes to another. Both are checked and
    this RAISES rather than degrading to a derivation.
    """
    metal = {label(r): r for r in rows(METAL_CENSUS)}
    return metal


def _live_passes(metal_row: Optional[dict], seam: str) -> Tuple[str, ...]:
    names = IN_SEAM_PASSES.get(seam)
    if names is None or metal_row is None:
        return ()
    live = set((metal_row.get("plan_step") or {}).get("live") or ())
    return tuple(name for name in names if name in live)


def report_in_seam_liveness(record: List[dict], metal: Dict[str, dict]
                            ) -> Dict[str, Any]:
    log("\n=== IN-SEAM PASS LIVENESS: THE ENGINE'S RECORD vs THE DERIVED RULE ===")
    log(f"  read from ../{METAL_CENSUS.name}/ plan_step.live — a DRIVER")
    log("  fact, and the record this census does not carry. The 2026-08-20 carry cuts")
    log("  DERIVED it from the boundary triple; both numbers are printed.")
    out: Dict[str, Any] = {"read": {}, "derived": {}, "rows_the_derived_rule_gets_wrong": {}}
    for seam, names in IN_SEAM_PASSES.items():
        out["read"][seam], out["derived"][seam] = {}, {}
        out["rows_the_derived_rule_gets_wrong"][seam] = {}
        for pass_name in names:
            read = sum(1 for r in record
                       if pass_name in _live_passes(metal.get(label(r)), seam))
            guess = sum(1 for r in record
                        if pass_name in _derived_in_seam_passes(r["configuration"],
                                                                seam))
            wrong = sum(1 for r in record
                        if (pass_name in _live_passes(metal.get(label(r)), seam))
                        != (pass_name in _derived_in_seam_passes(r["configuration"],
                                                                 seam)))
            out["read"][seam][pass_name] = read
            out["derived"][seam][pass_name] = guess
            out["rows_the_derived_rule_gets_wrong"][seam][pass_name] = wrong
            mark = ("AGREES" if not wrong
                    else f"DISAGREES ON {wrong} ROWS — the derived rule is WRONG "
                         f"and the carry cuts published its number")
            log(f"    {seam:5s} {pass_name:26s} live on {read:4d} / {len(record)} "
                f"rows (derived rule said {guess:4d})  {mark}")
    return out


def build_matrix(record: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
    log("\n=== THE MATRIX: per row, per seam ===")
    # The 08-20 CUDA battery, indexed for the E->P seam (see _E_SIDE_BLOCKS).
    fresh_by_label = {label(r): r for r in rows(FRESH)}
    # A FLOOR, not a comment. If an E->P row were missing from the fresh record the
    # seam would be scored on a subset and the count would silently under-report;
    # this raises instead. `freshness_check` already reports whether the two records
    # hold the same rows (since 2026-09-03 they do not: the eight added example rows
    # have no twin in the 2026-08-20 record) and that the shared rows agree on all 22
    # shared configuration keys — this
    # asserts the part THIS seam depends on, per row, rather than inheriting a
    # summary line printed elsewhere in the run.
    missing = sorted(label(r) for r in record
                     if r["configuration"]["n_polarizations"] > 0
                     and label(r) not in fresh_by_label)
    if missing:
        raise SystemExit(
            f"{len(missing)} E->P rows are absent from {FRESH.name} and this seam "
            f"cannot be scored from a subset without under-reporting it: {missing}")
    # The Metal battery, for the in-seam liveness column ONLY. The join is ASSERTED:
    # same row set, and agreement on the two configuration keys the driver's pass
    # selection turns on. A silent mis-join would attribute one row's passes to
    # another, so this raises.
    metal = in_seam_liveness()
    if set(metal) != {label(r) for r in record}:
        only_here = sorted({label(r) for r in record} - set(metal))
        only_there = sorted(set(metal) - {label(r) for r in record})
        raise SystemExit(
            f"the Metal census and this one do not hold the same rows, so "
            f"plan_step.live cannot be joined onto this board: "
            f"{len(only_here)} only here {only_here[:3]}, "
            f"{len(only_there)} only there {only_there[:3]}")
    drift = [label(r) for r in record
             if (tuple(r["configuration"]["mirrored"]),
                 tuple(r["configuration"]["metallic"]),
                 tuple(r["configuration"].get("source_field_types") or ()))
             != (tuple(metal[label(r)]["configuration"]["mirrored"]),
                 tuple(metal[label(r)]["configuration"]["metallic"]),
                 tuple(metal[label(r)]["configuration"].get("source_field_types")
                       or ()))]
    if drift:
        raise SystemExit(
            f"{len(drift)} rows' mirrored/metallic/source_field_types differ between "
            f"the two censuses; the liveness join is not sound: {drift[:5]}")
    log(f"  LIVENESS JOIN: {len(metal)} rows matched against "
        f"{METAL_CENSUS.name}, 0 configuration drift")

    # A CENSUS THAT DOES NOT CARRY THESE KEYS SCORES SEVEN PRODUCTS AT ZERO, SILENTLY.
    #
    # `seam_covered` reads a missing key as False, which is the right default for a
    # verdict that was recorded as a refusal and the WRONG one for a verdict that was
    # never recorded at all. Before 2026-08-30 this board transcribed these products'
    # clauses, so an older census legitimately has no seam entry for them — and
    # pointed at one, this cut would publish "0 of 15 E->P" and "2 of 53 folded D->E"
    # with no indication that it had measured nothing. That is exactly the shape of
    # wrong answer the whole board exists to avoid, so it is a refusal instead.
    required = ({"folded_fused_pair_D", "folded_fused_magnetic_pair_B",
                 "complex_fused_magnetic_pair_B",
                 "complex_fused_electric_pair_D",
                 "cylindrical_fused_electric_pair_D",
                 "complex_conductive_fused_pair_D",
                 "folded_dispersive_fused_pair_D",
                 "cylindrical_real_fused_electric_pair_D",
                 "folded_complex_fused_magnetic_pair_B",
                 "folded_complex_fused_magnetic_pair_B@unified_expansion",
                 "folded_complex_fused_pair_D",
                 "folded_complex_fused_pair_D@unified_expansion",
                 "beta_fused_electric_pair_D",
                 "bfast_fused_electric_pair_D",
                 "conductive_fused_electric_pair_D",
                 # ONE key for TWO cells: `no_pml_fused_electric_pair` and its
                 # `_lossless` twin are the same predicate under two arm labels, so the
                 # census carries one column and the shipped map above reads it twice.
                 "no_pml_fused_electric_pair_D",
                 # THE TWO SCRATCH-OUTPUT OFF-DIAGONAL WELDS, 2026-09-02. Required
                 # here, not merely read below: a census cut before the battery
                 # gained these two columns would price both cells at zero and
                 # print a headline, which is the failure this uniformity check
                 # exists to stop.
                 "offdiag_fused_electric_pair_D",
                 "folded_offdiag_fused_electric_pair_D",
                 "folded_beta_fused_electric_pair_D",
                 "folded_beta_fused_magnetic_pair_B",
                 "complex_beta_fused_electric_pair_D",
                 "complex_beta_fused_magnetic_pair_B",
                 "folded_beta_complex_fused_pair_D",
                 "folded_beta_complex_fused_pair_D@unified_expansion",
                 "folded_beta_complex_fused_magnetic_pair_B",
                 "folded_beta_complex_fused_magnetic_pair_B@unified_expansion",
                 "complex_fused_ade_chain", "complex_fused_ade_chain@unified_expansion"}
                | {key for keys in E_TO_P_VERDICT_KEYS.values() for key in keys})
    missing_keys = sorted(required - set((record[0].get("seam") or {}).keys()))
    if missing_keys:
        raise SystemExit(
            f"{CENSUS.name} carries no seam verdict for {len(missing_keys)} of the "
            f"predicates this cut reads: {missing_keys}. That census predates the "
            f"battery extension, so scoring it here would report those products at "
            f"ZERO without measuring them. Re-cut the census with "
            f"triton_predicate_battery as it stands.")
    thin = [label(r) for r in record
            if not required <= set((r.get("seam") or {}).keys())]
    if thin:
        raise SystemExit(
            f"{len(thin)} rows carry only part of the required seam block, so the "
            f"census is not uniform and a per-cell count over it would mix measured "
            f"rows with unmeasured ones: {thin[:5]}")
    log(f"  SEAM BLOCK: all {len(required)} shipped-predicate verdicts present on "
        f"all {len(record)} rows")

    cell_owner: Dict[Tuple[str, Tuple[Optional[str], Optional[str]]], str] = {
        (spec["seam"], spec["cell"]): name for name, spec in PRODUCTS.items()
        if spec["seam"] != "E->P"}

    instances: List[dict] = []
    ladder_disagreements: List[dict] = []
    admits_by_row: Dict[str, Dict[str, bool]] = {}
    for index, row in enumerate(record, start=1):
        configuration = row["configuration"]
        name = label(row)
        folded = folded_magnetic_clauses(row)
        complexed = complex_magnetic_clauses(row)
        folded_complex_pair = folded_complex_magnetic_clauses(row)
        chain = chain_clauses(row)

        electric = folded_electric_clauses(row)
        cylindrical = cylindrical_real_magnetic_clauses(row)
        cylindrical_complex_pair = cylindrical_complex_magnetic_clauses(row)
        nonlinear_pair = nonlinear_magnetic_clauses(row)
        beta_pair = beta_magnetic_clauses(row)
        bfast_pair = bfast_magnetic_clauses(row)
        # FOUR PRODUCTS MOVED FROM A TRANSCRIBED LADDER TO THEIR OWN SHIPPED VERDICT
        # ON 2026-08-30, AND THE MOVE IS A BUG FIX, NOT A CONVENIENCE.
        #
        # A ladder here is a SECOND COPY of a predicate that lives in the product. It
        # cannot notice when the product changes, and one of them did not:
        # `folded_electric_clauses` still carried `no_folded_periodic_axis`, which
        # `folded_fused_pair_coverage` RETIRED on 2026-08-21 when the far carry landed
        # inline (folded_fused_pair.py:1046-1110) under a released gate
        # (results/triton_folded_far_carry_d_2026-08-21/run_farcarryD5/). The magnetic
        # twin's mirror was updated when its own carry landed; the electric one was
        # not, so this board went on refusing rows the product admits. The census now
        # records the shipped verdict, so the board holds no copy to go stale.
        #
        # THE LADDERS ARE KEPT AND ARE NOW A CROSS-CHECK, not a scorer. Every
        # disagreement between a ladder and the product it mirrors is collected per
        # row and reported (`ladder_vs_shipped` below). A ladder that agrees on all
        # census rows is evidence the transcription was right; one that disagrees names
        # the clause and the rows, which is how this defect should have surfaced.
        shipped = {
            "folded_fused_pair": seam_covered(row, "folded_fused_pair_D"),
            "folded_fused_magnetic_pair":
                seam_covered(row, "folded_fused_magnetic_pair_B"),
            "complex_fused_magnetic_pair":
                seam_covered(row, "complex_fused_magnetic_pair_B"),
            # THE ELECTRIC TWIN, 2026-08-30. Read from the shipped predicate like its
            # sibling and under the SAME census probe: its PRODUCT_PROBE_PATTERNS is
            # the base four, so it needs no unified-record swap. There is NO ladder
            # for it and there never will be — the clause that decides 15 of its 16
            # rows consults `deposit_repair.repairable` on the live run, which no
            # configuration block carries. A transcription could only have guessed.
            "complex_fused_electric_pair":
                seam_covered(row, "complex_fused_electric_pair_D"),
            # THE Dcyl ELECTRIC TWIN and the COMPLEX CONDUCTIVE no-PML PAIR, both
            # 2026-08-30, both read from their shipped predicates under the census
            # probe. NEITHER HAS A LADDER, and the reason is the same in both
            # directions: the Dcyl one's source clause consults
            # `deposit_repair.repairable` on the live run (no configuration block
            # carries that, and all sixteen of its rows declare an electric source,
            # so a transcription guessing False would price the board's largest cell
            # at zero), and the conductive one's E half reads live polarization
            # objects the same way `fused_ade_chain` does.
            "cylindrical_fused_electric_pair":
                seam_covered(row, "cylindrical_fused_electric_pair_D"),
            "complex_conductive_fused_pair":
                seam_covered(row, "complex_conductive_fused_pair_D"),
            # THE FOLDED DISPERSIVE D->E WELD, 2026-08-31, read from its shipped
            # predicate under the census probe like every product above it. NO
            # LADDER, and there never will be one, for BOTH of the reasons the two
            # entries above give: the clause that decides all four of its rows
            # consults `deposit_repair.repairable` on the live run (no configuration
            # block carries that, and all four rows declare an electric source, so a
            # transcription guessing False would price the largest remaining cell at
            # zero), and its E half reads live polarization objects — the pole
            # partition and each state's `driven()` — the way `fused_ade_chain` does.
            #
            # NO PROBE ARGUMENT: this family is REAL-STORAGE throughout, so it
            # launches no complex multiply and needs no expansion licence.
            "folded_dispersive_fused_pair":
                seam_covered(row, "folded_dispersive_fused_pair_D"),
            # THE Dcyl m = 0 D->E WELD, 2026-08-31. Read from its shipped predicate
            # under the census probe. NO LADDER: the clause that decides all three of
            # its rows consults `deposit_repair.repairable` on the live run, and all
            # three declare an electric source, so a transcription guessing False
            # would price the cell at zero. Its MAGNETIC twin above IS scored from a
            # ladder, and that asymmetry is the point: on the B seam those same three
            # rows are electric-only, so no repair is consulted at all.
            "cylindrical_real_fused_electric_pair":
                seam_covered(row, "cylindrical_real_fused_electric_pair_D"),
            # UNDER THE UNIFIED RECORD, and the swap is measured rather than
            # preferred. This product's PRODUCT_PROBE_PATTERNS add
            # `c8_mul_c8_parity_coefficient_left`, which no per-family probe
            # classifies: under this census's own probe the fill half refuses on
            # every row, and under results/unified_expansion_2026-08-27/keep/gate.json
            # it admits. That artifact is the one its RELEASED gate requires — run
            # with the complex family's probe the gate refuses before its first
            # device leg. The census-probe verdict is carried beside it and reported,
            # so the licence this credit rests on is visible per row.
            #
            # ITS SIBLING NEEDS NO SWAP: complex_fused_magnetic_pair admits under
            # BOTH artifacts, measured the same way, so it is read from the census
            # probe like every other product here.
            "folded_complex_fused_magnetic_pair": seam_covered(
                row, "folded_complex_fused_magnetic_pair_B@unified_expansion"),
            # THE ELECTRIC TWIN, 2026-08-31, UNDER THE SAME UNIFIED RECORD AND FOR
            # THE SAME MEASURED REASON: its PRODUCT_PROBE_PATTERNS is the same
            # extended set (`folded_complex.PARITY_PROBE_PATTERNS`), which no
            # per-family probe classifies. The census-probe verdict is carried beside
            # it in the record and reported, so the licence this credit rests on is
            # visible per row. NO LADDER: the clause that decides both of its rows
            # consults `deposit_repair.repairable` on the live run, and both declare
            # an electric source, so a transcription guessing False would price the
            # cell at zero.
            "folded_complex_fused_pair": seam_covered(
                row, "folded_complex_fused_pair_D@unified_expansion"),
            # THE REAL-BETA ELECTRIC WELD, 2026-08-31. Read from its shipped
            # predicate. NO LADDER, and its MAGNETIC twin above IS scored from one —
            # that asymmetry is the point: the clause that decides this cell's only
            # row consults `deposit_repair.repairable` on the live run, and the row
            # declares an electric source, so a transcription guessing False would
            # price the cell at zero. NO PROBE ARGUMENT: real storage throughout.
            "beta_fused_electric_pair":
                seam_covered(row, "beta_fused_electric_pair_D"),
            # THE BFAST ELECTRIC WELD, 2026-08-31. Read from its shipped predicate,
            # for the reason the real-beta one is: the clause that decides its one
            # row consults `deposit_repair.repairable` on the live run. NO PROBE:
            # real storage throughout.
            "bfast_fused_electric_pair":
                seam_covered(row, "bfast_fused_electric_pair_D"),
            # THE CONDUCTIVE PML ELECTRIC WELD, 2026-08-31. Read from its shipped
            # predicate for the same reason as the two above -- the clause that
            # decides its one row consults `deposit_repair.repairable` on the live
            # run -- and for one more: its curl half's `conductive_targets` reads
            # `Fields.condfac_for` PER COMPONENT, which is not a census fact.
            "conductive_fused_electric_pair":
                seam_covered(row, "conductive_fused_electric_pair_D"),
            # THE NO-ABSORBER STORED-E ELECTRIC WELD, 2026-08-31. ONE predicate, TWO
            # cells -- the same shape `folded_complex_fused_magnetic_pair` and its
            # `_offdiag` twin set, and for the same reason: the arm LABEL decides which
            # cell an instance lands in, and the verdict is the same object either way.
            # Here the two labels are `conductive no-PML` and `no-PML`, which the
            # kernel's per-component COND constexpr makes one body.
            #
            # READ FROM THE SHIPPED PREDICATE and not from a census ladder, for the
            # reason the three welds above are: the clause that decides all three of
            # its rows consults `deposit_repair.repairable` on the LIVE run -- and on
            # this family a second live clause does too, the one that reads each
            # source's `is_integrated` against `Fields.condinv_for`. Neither is a
            # census fact, and a transcription guessing either would misprice the cell
            # in both directions.
            "no_pml_fused_electric_pair":
                seam_covered(row, "no_pml_fused_electric_pair_D"),
            "no_pml_fused_electric_pair_lossless":
                seam_covered(row, "no_pml_fused_electric_pair_D"),
            # THE TWO FOLDED-BETA WELDS, 2026-08-31. Read from their shipped
            # predicates for the reason every 08-31 weld is: the clause that
            # decides their single row consults `deposit_repair.repairable` on the
            # LIVE run — and here the repair's FOLD clauses read the grid's own
            # fill map, which no configuration block carries. NO PROBE ARGUMENT:
            # real storage throughout.
            "folded_beta_fused_electric_pair":
                seam_covered(row, "folded_beta_fused_electric_pair_D"),
            "folded_beta_fused_magnetic_pair":
                seam_covered(row, "folded_beta_fused_magnetic_pair_B"),
            # THE TWO COMPLEX-BETA WELDS, 2026-09-01. Read from their shipped
            # predicates UNDER THE CENSUS PROBE — which works only because the
            # census probe now carries the beta tranche's fifth pattern (the
            # 2026-09-01 extension artifact; see the PRODUCTS entry). NO LADDER:
            # the electric twin's deciding clause consults
            # `deposit_repair.repairable` on the live run, and both halves'
            # expansion clauses read the live probe artifact.
            "complex_beta_fused_electric_pair":
                seam_covered(row, "complex_beta_fused_electric_pair_D"),
            "complex_beta_fused_magnetic_pair":
                seam_covered(row, "complex_beta_fused_magnetic_pair_B"),
            # THE TWO FOLDED COMPLEX-BETA WELDS, 2026-09-02, UNDER THE UNIFIED
            # RECORD like the beta-less folded complex twins and for their reason
            # extended by one pattern: PRODUCT_PROBE_PATTERNS is the SIX-pattern
            # union (parity + beta), the census probe classifies five (no
            # parity), and the unified record classifies all six with both
            # extended licences answering FMA_V1. The census-probe verdict is
            # carried beside each instance. NO LADDER: the deciding clauses
            # consult `deposit_repair.repairable` on the live run.
            "folded_beta_complex_fused_pair": seam_covered(
                row, "folded_beta_complex_fused_pair_D@unified_expansion"),
            "folded_beta_complex_fused_magnetic_pair": seam_covered(
                row, "folded_beta_complex_fused_magnetic_pair_B@unified_expansion"),
            # THE TWO SCRATCH-OUTPUT OFF-DIAGONAL WELDS, 2026-09-02. Read from
            # their shipped predicates and NOT from a census ladder, for two
            # reasons that are both live-run facts: the source clause consults
            # `deposit_repair.seam_source_reasons` on the LIVE source list (these
            # products declare CARRIES_DEPOSIT_REPAIR False, so every in-seam
            # electric source refuses BY NAME, and 8 of the unfolded cell's 16
            # rows and 10 of the folded cell's 19 turn on exactly that), and the
            # folded one's far-ghost clause asks `stepping._stored_past_owned` of
            # the live grid rather than a boundary code.
            "offdiag_fused_electric_pair":
                seam_covered(row, "offdiag_fused_electric_pair_D"),
            "folded_offdiag_fused_electric_pair":
                seam_covered(row, "folded_offdiag_fused_electric_pair_D"),
        }
        ladder = {
            "folded_fused_pair": all(electric.values()),
            "folded_fused_magnetic_pair": all(folded.values()),
            "complex_fused_magnetic_pair": all(complexed.values()),
            "folded_complex_fused_magnetic_pair": all(folded_complex_pair.values()),
        }
        for product_name, ladder_says in ladder.items():
            if ladder_says != shipped[product_name]:
                ladder_disagreements.append({
                    "row": name, "product": product_name,
                    "ladder_says": ladder_says,
                    "the_shipped_predicate_says": shipped[product_name],
                    "ladder_clauses_that_are_False": sorted(
                        clause for clause, ok in {
                            "folded_fused_pair": electric,
                            "folded_fused_magnetic_pair": folded,
                            "complex_fused_magnetic_pair": complexed,
                            "folded_complex_fused_magnetic_pair": folded_complex_pair,
                        }[product_name].items() if not ok),
                    "the_shipped_predicate_refused": [
                        r for r in ((row.get("seam") or {}).get({
                            "folded_fused_pair": "folded_fused_pair_D",
                            "folded_fused_magnetic_pair":
                                "folded_fused_magnetic_pair_B",
                            "complex_fused_magnetic_pair":
                                "complex_fused_magnetic_pair_B",
                            "folded_complex_fused_magnetic_pair":
                                "folded_complex_fused_magnetic_pair_B",
                        }[product_name], {}) or {}).get("residual_reasons") or ()][:3],
                })
        admits = {
            "bfast_fused_magnetic_pair": all(bfast_pair.values()),
            "nonlinear_fused_magnetic_pair": all(nonlinear_pair.values()),
            "beta_fused_magnetic_pair": all(beta_pair.values()),
            "fused_pair_B": seam_covered(row, "fused_pair_B"),
            "fused_pair_D": seam_covered(row, "fused_pair_D"),
            "dispersive_fused_pair": seam_covered(row, "dispersive_fused_pair_D"),
            "folded_fused_magnetic_pair": shipped["folded_fused_magnetic_pair"],
            "folded_fused_pair": shipped["folded_fused_pair"],
            "cylindrical_real_fused_magnetic_pair": all(cylindrical.values()),
            "cylindrical_fused_magnetic_pair": all(cylindrical_complex_pair.values()),
            "complex_fused_magnetic_pair": shipped["complex_fused_magnetic_pair"],
            "complex_fused_electric_pair": shipped["complex_fused_electric_pair"],
            "cylindrical_fused_electric_pair":
                shipped["cylindrical_fused_electric_pair"],
            "complex_conductive_fused_pair":
                shipped["complex_conductive_fused_pair"],
            "folded_dispersive_fused_pair":
                shipped["folded_dispersive_fused_pair"],
            "cylindrical_real_fused_electric_pair":
                shipped["cylindrical_real_fused_electric_pair"],
            # ONE predicate, TWO cells. The arm label decides which cell an instance
            # lands in; the verdict is the same object either way.
            "folded_complex_fused_magnetic_pair":
                shipped["folded_complex_fused_magnetic_pair"],
            "folded_complex_fused_magnetic_pair_offdiag":
                shipped["folded_complex_fused_magnetic_pair"],
            "folded_complex_fused_pair":
                shipped["folded_complex_fused_pair"],
            "beta_fused_electric_pair":
                shipped["beta_fused_electric_pair"],
            "bfast_fused_electric_pair":
                shipped["bfast_fused_electric_pair"],
            "conductive_fused_electric_pair":
                shipped["conductive_fused_electric_pair"],
            # ONE predicate, TWO cells, same shape as folded_complex above: the
            # 2026-08-31 plain-branch product answers for both the lossy and the
            # lossless no-PML D->E cell, and the arm label decides which cell an
            # instance lands in. Added when the KeyError at pricing showed the
            # product had every wire EXCEPT this dict (the round that built it was
            # killed by a spend limit between the census entry and this row).
            "no_pml_fused_electric_pair":
                shipped["no_pml_fused_electric_pair"],
            "no_pml_fused_electric_pair_lossless":
                shipped["no_pml_fused_electric_pair"],
            "folded_beta_fused_electric_pair":
                shipped["folded_beta_fused_electric_pair"],
            "folded_beta_fused_magnetic_pair":
                shipped["folded_beta_fused_magnetic_pair"],
            "complex_beta_fused_electric_pair":
                shipped["complex_beta_fused_electric_pair"],
            "complex_beta_fused_magnetic_pair":
                shipped["complex_beta_fused_magnetic_pair"],
            "folded_beta_complex_fused_pair":
                shipped["folded_beta_complex_fused_pair"],
            "folded_beta_complex_fused_magnetic_pair":
                shipped["folded_beta_complex_fused_magnetic_pair"],
            "offdiag_fused_electric_pair":
                shipped["offdiag_fused_electric_pair"],
            "folded_offdiag_fused_electric_pair":
                shipped["folded_offdiag_fused_electric_pair"],
            "fused_dispersive_chain": all(chain.values()),
            "fused_ade_chain": any(
                seam_covered(row, key)
                for key in E_TO_P_VERDICT_KEYS["fused_ade_chain"]),
            "complex_fused_ade_chain": any(
                seam_covered(row, key)
                for key in E_TO_P_VERDICT_KEYS["complex_fused_ade_chain"]),
            "folded_offdiag_fused_ade_chain": any(
                seam_covered(row, key)
                for key in E_TO_P_VERDICT_KEYS["folded_offdiag_fused_ade_chain"]),
        }
        admits_by_row[name] = dict(admits)

        for seam, curl_slot, update_slot in (("B->H", "step_B", "update_H"),
                                             ("D->E", "step_D", "update_E")):
            cell = (arm(row, curl_slot), arm(row, update_slot))
            product = cell_owner.get((seam, cell))
            entry = {
                "row": name, "seam": seam,
                "curl_arm": cell[0], "constitutive_arm": cell[1],
                "curl_admitters": list(admitted(row, curl_slot)),
                "constitutive_admitters": list(admitted(row, update_slot)),
                "product": product,
                "predicate_admits": bool(product and admits[product]),
                "in_seam_source": in_seam_source(row, seam),
                # The DRIVER fact above, and the CEILING consequence below, are two
                # columns now: the injection still happens on every one of these rows,
                # and since 2026-08-28 it only BLOCKS where the repair refuses.
                "in_seam_source_blocks": in_seam_source_blocks(row, seam),
                # READ off the Metal census's plan_step.live, not derived.
                "live_in_seam_passes": list(_live_passes(metal.get(name), seam)),
                # THE LICENCE THIS INSTANCE'S CREDIT RESTS ON, where it is not the
                # census's own probe. Carried on every instance so the swap is a
                # COLUMN rather than a remark in a docstring; None where no swap
                # applies. Widened 2026-09-02 from the single magnetic special
                # case to every product whose `shipped` row reads an
                # `@unified_expansion` key — the electric folded_complex twin
                # and the two folded beta products were credited under the swap
                # while the column printed None for them.
                "credited_under_a_different_expansion_licence": (
                    "unified_expansion_2026-08-27/keep/gate.json"
                    if product in UNIFIED_CREDITED_PRODUCTS else None),
                "same_product_under_the_census_probe": (
                    seam_covered(row, UNIFIED_CREDITED_PRODUCTS[product])
                    if product in UNIFIED_CREDITED_PRODUCTS else None),
                "planner_reachable": False, "planner_refusal": None,
            }
            if product and admits[product]:
                ok, why = planner_reachable(row, product)
                entry["planner_reachable"], entry["planner_refusal"] = ok, (why or None)
            instances.append(entry)

        if configuration["n_polarizations"] > 0:
            # THE E->P CELL IS (update_E arm, update_P arm), CORRECTED HERE.
            #
            # The 2026-08-20 cuts keyed this instance on ``(step_D arm, update_E
            # arm)`` — the D->E cell — because the only product they knew here,
            # ``fused_dispersive_chain``, spans step_D -> update_E -> update_P and
            # so genuinely occupies the D->E cell. But the SEAM is update_E(:3304)
            # -> update_P(:3306), its halves are those two slots, and a matrix that
            # names it by a different pair cannot express an E->P-ONLY product at
            # all — which is what the Metal backend shipped on 2026-08-20
            # (metal_kernels/fused_ade_chain.py, three arms, 10/15 admitted).
            #
            # ``update_P`` IS a slot: launch.py:1874-1877 lists it in STEP_ORDER and
            # :2984-3048 resolves an ADE family for it. THE CENSUS THIS SCRIPT READS
            # DOES NOT RECORD IT — ``plan_step.slots`` carries only step_B, step_D,
            # update_E, update_H (and the two fill slots where a fold has them), never
            # update_P, on every row — so ``arm(row, "update_P")`` is
            # None on every row and every E->P cell reads as half-on-the-array-path.
            # That is a STALE RECORD, reported as one below, not a property of the
            # engine; clearing it needs a census leg that walks the update_P slot.
            fresh_row = fresh_by_label[name]
            # THE E HALF IS READ FROM THE LIVE TRITON CENSUS, THE P HALF FROM THE
            # CUDA BATTERY (2026-09-04). The note at :2117 gave ONE reason for BOTH
            # halves — the 2026-08-16 census carried neither column. Only the
            # update_P half of that is still true: `plan_step.slots` on the
            # 2026-09-03 and 2026-09-04 censuses carries step_B / step_D / update_E /
            # update_H (+ the two fill slots) and NO update_P, while its update_E
            # block lists 21 arms on every row. Reading the E half from the CUDA
            # battery asked a CUDA family whether a TRITON arm admits:
            # `cuda_folded_offdiag` is the NON-dispersive off-diagonal kernel and
            # refuses examples:absorbed_power_density.py on "dispersion: update_E's
            # source is (D - sum P), not D", which is true of it and false of
            # meep_gpu/triton_kernels/folded_offdiag_dispersive_update_e.py
            # (registered at launch.py:3584). That filed the row `missing_half`
            # naming a kernel that SHIPS. `arm()` here is the SAME call the row's
            # D->E instance already makes for this same slot.
            e_arm = arm(row, "update_E")
            p_arm = _fresh_arm(fresh_row, _P_SIDE_BLOCKS, "update_P")
            # ``fused_dispersive_chain`` is scored by its own cell (D->E) because
            # that is the pair it fuses; it is credited at this seam too when it
            # admits, since one launch of it performs update_P as well.
            spanning = "fused_dispersive_chain"
            spans_into_e_to_p = ((arm(row, "step_D"), arm(row, "update_E"))
                                 == PRODUCTS[spanning]["cell"])
            product = spanning if spans_into_e_to_p else None

            # THE TWO E->P-ONLY PRODUCTS, CONSULTED BY THEIR OWN VERDICT AND NOT BY
            # A CELL MATCH. Until 2026-08-30 this branch knew exactly one product,
            # the SPANNING one, and asked whether the row's D->E cell was that
            # product's; a matrix keyed that way cannot express an E->P-only product
            # at all, which is why the two ADE chains read zero for a week after
            # they were built and gated.
            #
            # A CELL MATCH WOULD STILL BE WRONG HERE, so it is not what is used.
            # `fused_ade_chain` has THREE arms and the census's E-side arm labels
            # collapse two of them onto one cell ("dispersive", "ADE update_P"), so
            # "the cell owner admits" and "this product admits this row" are
            # different questions on 7 of the 15 instances. The verdict recorded
            # from the shipped predicate answers the second one, which is the one
            # the board is asking. The cell is still reported, as description.
            e_to_p_admitters = [candidate for candidate in E_TO_P_PRODUCTS
                                if admits[candidate]]
            # A PRODUCT THAT ADMITS BEATS ONE THAT SPANS AND REFUSES, and getting
            # that backwards cost three instances on the first cut of this branch.
            #
            # `fused_dispersive_chain` OCCUPIES this cell wherever the row's D->E
            # cell is its own — it performs update_P as part of one launch — so it
            # is named here first. On the three `stochastic_emitter` rows it then
            # REFUSES, on `exactly_one_susceptibility` (fused_dispersive_chain.py:380):
            # they carry six susceptibilities. `fused_ade_chain` admits those same
            # rows on its `dispersive` arm, because six poles per component is
            # exactly its CHAIN_MAX_POLES and the cap does not fire.
            #
            # SERVED asks whether SOME shipped product can perform this seam in one
            # launch, so a spanning product's refusal must not hide another's
            # admission. The instance still records both — `product` names who serves
            # it, `e_to_p_products_admitting` names every candidate that would.
            if product is not None and not admits[product] and e_to_p_admitters:
                product = e_to_p_admitters[0]
            elif product is None and e_to_p_admitters:
                # Deterministic, and asserted not to be a coin-toss: the two
                # products' storage clauses are disjoint (one refuses
                # force_complex_fields, the other requires it), so at most one can
                # admit a row. `assert_at_most_one_e_to_p_product_admits` re-earns
                # that on every cut rather than trusting this sentence.
                product = e_to_p_admitters[0]
            if product is not None:
                e_to_p_reach, e_to_p_why = planner_reachable(row, product)
            else:
                e_to_p_reach, e_to_p_why = (False, "no product spans this cell")
            entry = {
                "row": name, "seam": "E->P",
                "curl_arm": e_arm, "constitutive_arm": p_arm,
                "curl_admitters": list(admitted(row, "update_E")),
                # THE OLD SOURCING, CARRIED AS A CONTROL rather than deleted. Over
                # the 15 E->P instances the CUDA battery agrees with the live census
                # on 3 (the stochastic_emitter rows), disagrees on the LABEL only on
                # 11, and disagreed on ADMISSION on 1 — absorbed_power_density.py —
                # which is what moved the E half onto the census. Kept so the swap is
                # visible on every instance rather than only in the comment above.
                "curl_arm_in_the_2026_08_20_cuda_battery":
                    _fresh_arm(fresh_row, _E_SIDE_BLOCKS, "update_E"),
                "constitutive_admitters": [n for b, n in _P_SIDE_BLOCKS
                                           if _fresh_covered(fresh_row, b, "update_P")],
                "product": product,
                "predicate_admits": bool(product and admits[product]),
                "in_seam_source": in_seam_source(row, "E->P"),
                # E->P has nothing between its two sub-steps, so neither column can
                # ever be True here; both are carried so every instance has the same
                # shape whatever seam it is on.
                "in_seam_source_blocks": in_seam_source_blocks(row, "E->P"),
                # DERIVED from the same measurement every other instance uses, not
                # spelled: the E->P product's own reachability went stale the same
                # way the two folded pairs' did, and a second hand-written copy of
                # the answer is a second thing to forget.
                "planner_reachable": e_to_p_reach,
                "planner_refusal": e_to_p_why or None,
                # Recorded per instance so the record split is a COLUMN and not a
                # remark: the reader can count the rows each record contributes.
                "update_P_slot_absent_from_the_census": not admitted(row, "update_P"),
                # ONE KEY PER HALF SINCE 2026-09-04: the two halves are read from two
                # records and a single record name here would misattribute one of
                # them.
                "halves_read_from": {"update_E": CENSUS.name, "update_P": FRESH.name},
                "poles": int(configuration["n_polarizations"]),
                # WHICH RECORD CREDITED THIS INSTANCE. The spanning product is
                # scored from a transcribed ladder over the CUDA battery's E/P
                # blocks; the two chains are scored from their own shipped verdict
                # recorded in this census. A reader must be able to tell those apart
                # without reading this file.
                "e_to_p_products_admitting": list(e_to_p_admitters),
                "scored_from": ("the shipped predicate, recorded in the census seam "
                                "block" if product in E_TO_P_PRODUCTS else
                                "a transcribed clause ladder (chain_clauses)"
                                if product else None),
                "chain_verdicts": {
                    key: seam_covered(row, key)
                    for keys in E_TO_P_VERDICT_KEYS.values() for key in keys},
                # The census-probe verdict for the complex chain, carried BESIDE the
                # unified one it is credited from, so the licence swap is visible on
                # every instance rather than only in this file's prose.
                "complex_chain_under_the_census_probe":
                    seam_covered(row, "complex_fused_ade_chain"),
            }
            instances.append(entry)

        if index % 20 == 0 or index == len(record):
            served = sum(1 for e in instances if e["predicate_admits"])
            log(f"  row {index:3d}/{len(record)}  seam-instances {len(instances):3d}  "
                f"served {served:3d}  [{name}]")

    # THE TWO PROMISES THE E->P BRANCH MAKES, RE-EARNED ON THIS RECORD. Both raise.
    chain_arms = assert_the_chain_arms_do_not_overlap(record)
    e_to_p_pick = assert_at_most_one_e_to_p_product_admits(record, admits_by_row)
    log(f"  E->P: chain arms disjoint on {chain_arms['rows_checked']} rows; "
        f"at most one E->P product admits on {e_to_p_pick['rows_checked']} rows")

    # THE CROSS-CHECK, REPORTED WHETHER IT AGREES OR NOT. A silent cross-check is
    # not one: the count goes in the artifact on every cut, so "0 disagreements"
    # is a measurement someone made rather than an absence someone hopes for.
    log(f"  LADDER vs SHIPPED PREDICATE: {len(ladder_disagreements)} disagreement(s) "
        f"across {len(record)} rows x 4 transcribed products")
    for entry in ladder_disagreements[:6]:
        log(f"    {entry['row']}  {entry['product']}: ladder "
            f"{entry['ladder_says']} vs shipped {entry['the_shipped_predicate_says']}"
            f"  (ladder False on {entry['ladder_clauses_that_are_False']})")
    if len(ladder_disagreements) > 6:
        log(f"    ... and {len(ladder_disagreements) - 6} more")

    return instances, {
        "cell_owner": {f"{s} {c}": p for (s, c), p in cell_owner.items()},
        "ladder_vs_shipped": {
            "what_this_is": (
                "Four products are scored from the verdict their SHIPPED predicate "
                "returned, recorded in the census seam block. The hand-transcribed "
                "clause ladders that used to score them are kept and evaluated "
                "beside them; every row where the two disagree is listed here. A "
                "disagreement means the transcription is stale, the product moved, "
                "or the ladder was always wrong — never that the board may pick."),
            "n_disagreements": len(ladder_disagreements),
            "rows_checked": len(record),
            "products_cross_checked": ["folded_fused_pair",
                                       "folded_fused_magnetic_pair",
                                       "complex_fused_magnetic_pair",
                                       "folded_complex_fused_magnetic_pair"],
            "disagreements": ladder_disagreements,
        },
        "e_to_p_chain_arms_are_disjoint": chain_arms,
        "e_to_p_at_most_one_product_admits": e_to_p_pick,
    }


def report(instances: List[dict], record: List[dict]) -> Dict[str, Any]:
    per_seam = collections.Counter(e["seam"] for e in instances)
    ade_rows = sum(1 for r in record if r["configuration"]["n_polarizations"] > 0)
    denominator = len(instances)

    log("\n=== THE DENOMINATOR ===")
    log(f"  B->H seam-instances (one per row)                : {per_seam['B->H']}")
    log(f"  D->E seam-instances (one per row)                : {per_seam['D->E']}")
    log(f"  E->P seam-instances (rows with n_polarizations>0): {per_seam['E->P']} "
        f"(measured: {ade_rows} rows)")
    log(f"  H->D seam-instances (one per row, since 2026-09-04): {len(record)}")
    log(f"  DENOMINATOR (three-seam ledger)                  : {denominator}   "
        f"PRICED, four seams: {denominator + len(record)}")

    a = sum(1 for e in instances if e["product"])
    b = sum(1 for e in instances if e["predicate_admits"])
    c = sum(1 for e in instances if e["planner_reachable"])
    # SERVED is (b) MINUS the instances whose product's release binding went stale
    # and was never rebound. The two are reported separately and never merged: (b)
    # is what the shipped PREDICATE says, which is a fact about the predicate and
    # does not change when a gate goes stale; SERVED is what this board is willing to
    # CREDIT, and a credit whose certified bytes no longer ship rests on nothing.
    # See CREDIT_WITHHELD_STALE_BINDING for the four products and their measured drift.
    withheld_instances = sum(1 for e in instances if e["predicate_admits"]
                             and e["product"] in CREDIT_WITHHELD_STALE_BINDING)
    # NAMED served_total, not `served`: this function already binds a local `served`
    # twice inside later loops (the per-cell and per-gap counts), and the first cut of
    # this block used the bare name -- so the LOG printed 150 and the JSON key carried
    # whatever the last loop iteration had left behind, which was 0. The floor two
    # blocks down compares the log's number, so it did not catch it; the JSON did.
    served_total = b - withheld_instances
    # SERVED IN DISPATCH — the other number, and until 2026-08-29 this board
    # asserted it was zero in a caveat string instead of asking. It is now the
    # shipped ladder's own answer, per served instance: the arm the product writes
    # must be one ``fastpath.RELEASED_FUSED_ARMS`` names AND the row must be inside
    # ``FUSED_RELEASE_ENVELOPE``. Passing the CREDITED instances, not the admitting
    # ones, so a cell whose release binding went stale cannot dispatch on this
    # board's arithmetic either.
    configurations = {label(r): dict(r["configuration"], facts=r.get("facts") or {})
                      for r in record}  # facts ride along; see configuration_of below
    # The lift facts ride with the configuration (2026-09-13): dispatch_reachability
    # derives `dimensions` by MEEP's rule from them, not from the extent count.
    configuration_of = {label(r): dict(r["configuration"], facts=r.get("facts") or {})
                        for r in record}
    # The null constitutive arm, named the way the cell table below names it
    # (line ~1297: `cell[1] == "no-PML null"`). Spelled once here rather than read
    # off `gaps`, which this block runs before.
    NULL_CONSTITUTIVE_ARM = "no-PML null"
    # ------------------------------------------- THE FOURTH SEAM, update_H -> step_D
    # One instance per row (2026-09-04). The two halves are arms this board already
    # read off the census — update_H is B->H's constitutive half, step_D is D->E's
    # curl half, both by the one-admitter rule — and the null verdict is the same
    # NULL_CONSTITUTIVE_ARM test B->H applies, asked of the FIRST half here. Priced
    # by h_to_d_seam.price below from the probe artifact (the withdraw is measured
    # per lifted row, not read off a name). `denominator` stays the three-seam
    # ledger's own; the board's PRICED denominator is `priced`.
    _by_row_seam = {(e["row"], e["seam"]): e for e in instances}
    # The seam probe's own per-row facts, joined once for the served verdict below.
    _h_to_d_probe = h_to_d_seam.join([label(r) for r in record],
                                     h_to_d_seam.probe_rows())
    h_to_d_entries = []
    for r in record:
        name = label(r)
        h_arm = _by_row_seam[(name, "B->H")]["constitutive_arm"]
        d_arm = _by_row_seam[(name, "D->E")]["curl_arm"]
        # THE PER-ROW `served_by` VERDICT, and it is this board's to give: the shared
        # rule cannot evaluate a per-backend predicate, and a row left unanswered
        # while a spanning product exists is an ABSENCE that `h_to_d_seam.price`
        # refuses by name.
        #
        # DERIVED THE WAY THE PREDICATE IS BUILT, not restated: `fused_hd_pair`'s
        # coverage is the conjunction of the two arms' own predicates on the two
        # slots, and those arms are what this board already read off the census by
        # the one-admitter rule -- so the cell membership IS the predicate's first
        # two clauses. Its remaining clause with a per-row answer is the seam's:
        # `HOISTS_THE_WITHDRAW` is False, so a row whose electric withdraw does work
        # is refused BY NAME, and that flag is measured per row by the seam probe
        # rather than read off an arm.
        spanning = [name_ for name_, spec in PRODUCTS.items()
                    if fusion_taxonomy.canonical_seam(spec["seam"])
                    == h_to_d_seam.SEAM]
        served_by = None
        for candidate in spanning:
            if tuple(PRODUCTS[candidate]["cell"]) != (h_arm, d_arm):
                continue
            if _h_to_d_probe[name]["withdraw_in_seam"]:
                continue  # HOISTS_THE_WITHDRAW is False on every product here today
            served_by = candidate
            break
        # THE THIRD ANSWER (2026-09-10). `served_by` stays the PREDICATE's verdict;
        # what this board will not do is CREDIT a product whose release binding is
        # stale, and the reason is the table's own sentence. The membership test is
        # the same one every other site in this function applies to the table
        # (`in CREDIT_WITHHELD_STALE_BINDING`), so the fourth seam withholds on
        # exactly the products the three seams withhold on — and a table entry with
        # an empty sentence reaches `h_to_d_seam.price`'s flag refusal rather than
        # being read as credited. None on every row while the table is empty.
        credit_withheld = (CREDIT_WITHHELD_STALE_BINDING[served_by]
                           if served_by and served_by in CREDIT_WITHHELD_STALE_BINDING
                           else None)
        h_to_d_entries.append({"row": name, "update_H": h_arm, "step_D": d_arm,
                               "update_H_is_null": h_arm == NULL_CONSTITUTIVE_ARM,
                               "served_by": served_by,
                               "credit_withheld": credit_withheld})
    priced = denominator + len(h_to_d_entries)

    # THE PRODUCT, NOT THE LABEL. ``served_in_dispatch`` does the product -> arm
    # label join itself since 2026-09-02, so that a certified product no composer
    # installs can be counted apart from a product the join has never heard of.
    # AND EVERY SERVED INSTANCE, ON ALL FOUR SEAMS, since 2026-09-06: the H->D rows
    # this board's own `served_by` verdict admits are served instances too, so they
    # are handed over with the three-seam ones and the measurement's `served_counted`
    # is the served headline rather than a three-seam subtotal beside it. The H->D
    # product is in dispatch_reachability.CERTIFIED_BUT_NOT_INSTALLED, so its rows
    # land in that bucket by name; `served_in_dispatch` itself does not move.
    # THE CREDITED H->D ROWS ONLY, as for the three seams (2026-09-10): a withheld
    # H->D row is not served, `served_counted` is a SERVED_HEADLINE_KEY, and
    # headline_agreement would refuse the cut if it disagreed with the ledger.
    # The seam rides along; see build_fusion_matrix.py's note at the same call.
    dispatch_reach = _reach.served_in_dispatch("triton", [
        (e["row"], e["product"], configurations[e["row"]], e["seam"])
        for e in instances
        if e["predicate_admits"] and e["product"]
        and e["product"] not in CREDIT_WITHHELD_STALE_BINDING
    ] + [
        (e["row"], e["served_by"], configurations[e["row"]], h_to_d_seam.SEAM)
        for e in h_to_d_entries
        if e["served_by"] and not e.get("credit_withheld")])
    log("\n=== THE AGGREGATE ===")
    log(f"  (a) a fused PRODUCT exists for the cell           : {a} / {denominator} "
        f"({100.0 * a / denominator:.1f}%)")
    log(f"  (b) its shipped PREDICATE admits the row          : {b} / {denominator} "
        f"({100.0 * b / denominator:.1f}%)")
    log(f"      of which the CREDIT IS WITHHELD (stale binding): {withheld_instances}")
    log(f"  SERVED - predicate admits AND the credit is bound : {served_total} / "
        f"{denominator} ({100.0 * served_total / denominator:.1f}%)   <-- THE NUMBER")
    log(f"  (c) plan_step can COMPOSE it                      : {c} / {denominator} "
        f"({100.0 * c / denominator:.1f}%)")
    log(f"  SERVED IN DISPATCH - the ladder would launch it   : "
        f"{dispatch_reach['served_in_dispatch']} / {denominator} "
        f"({100.0 * dispatch_reach['served_in_dispatch'] / denominator:.1f}%)"
        f"   across {dispatch_reach['rows']} rows")
    for arm_label, count in dispatch_reach["arms"].items():
        log(f"      {arm_label:<26s} {count:4d}")
    log(f"      released by {dispatch_reach['driver_route_gate']}; needs "
        f"{dispatch_reach['dispatch_by_default'] and 'nothing' or 'MEEP_GPU_DISPATCH=1'}")
    for axis, count in dispatch_reach["refused_by_envelope"].items():
        log(f"      refused, outside the envelope: {count:4d}  {axis}")

    log("\n  SERVED BY SEAM - where every credited instance actually is:")
    for seam in ("B->H", "D->E", "E->P"):
        here = [e for e in instances if e["seam"] == seam]
        credited = sum(1 for e in here if e["predicate_admits"]
                       and e["product"] not in CREDIT_WITHHELD_STALE_BINDING)
        log(f"    {seam:5s}  served {credited:4d} / {len(here):4d}"
            f"   (predicate admits "
            f"{sum(1 for e in here if e['predicate_admits']):4d})")
    _hd_credited = sum(1 for e in h_to_d_entries
                       if e["served_by"] and not e.get("credit_withheld"))
    _hd_withheld = sum(1 for e in h_to_d_entries if e.get("credit_withheld"))
    log(f"    H->D   served {_hd_credited:4d} / {len(h_to_d_entries):4d}   "
        f"(predicate admits {_hd_credited + _hd_withheld:4d}; credit withheld "
        f"{_hd_withheld}; this board's own served_by verdict, priced by h_to_d_seam "
        f"below)")
    kinds = collections.Counter(
        tuple(sorted(set(r["configuration"].get("source_field_types") or [])))
        for r in record)
    log(f"  source field types across the corpus: {dict(kinds)}")

    log("\n  (b) BUT NOT (c) — why a predicate-admitted instance is not composable:")
    unreachable = collections.Counter(
        e["planner_refusal"] for e in instances
        if e["predicate_admits"] and not e["planner_reachable"])
    for reason, count in unreachable.most_common():
        log(f"    {count:4d}  {reason}")

    log("\n  per product (predicate admits / cell reached):")
    by_product = collections.Counter(e["product"] for e in instances if e["product"])
    admits_by = collections.Counter(e["product"] for e in instances if e["predicate_admits"])
    reach_by = collections.Counter(e["product"] for e in instances if e["planner_reachable"])
    for name in PRODUCTS:
        log(f"    {name:30s} cell reached {by_product[name]:4d}   "
            f"predicate admits {admits_by[name]:4d}   planner {reach_by[name]:4d}")

    log("\n=== THE SOURCE SEAM: the ceiling on every product at a seam ===")
    log("  driver.py:3283-3284 (magnetic) and :3294-3299 (electric) put real work")
    log("  between the curl and the constitutive. Until 2026-08-28 that alone put the")
    log("  instance beyond every fused product; deposit_repair now carries the deposit")
    log("  across the seam, so an injection blocks ONLY where `repairable` refuses.")
    log("  A second bound sits under it: a cell where one slot has NO admitting arm")
    log("  has no pair to fuse at all, so its instances are unreachable until the")
    log("  UNFUSED arm covers them (launch.py:1983-2010, :2084 _pair_may_absorb).")
    seam_ceiling: Dict[str, Dict[str, int]] = {}
    for seam in ("B->H", "D->E", "E->P"):
        here = [e for e in instances if e["seam"] == seam]
        blocked = sum(1 for e in here if e["in_seam_source_blocks"])
        carried = sum(1 for e in here
                      if e["in_seam_source"] and not e["in_seam_source_blocks"])
        paired = sum(1 for e in here
                     if not e["in_seam_source_blocks"]
                     and e["curl_arm"] and e["constitutive_arm"])
        seam_ceiling[seam] = {"seam_instances": len(here),
                              "carry_an_in_seam_source_that_blocks": blocked,
                              "carry_an_in_seam_source_a_repair_can_carry": carried,
                              "ceiling": len(here) - blocked,
                              "reachable_ceiling_both_arms_admit": paired}
        log(f"    {seam:5s}  {len(here):4d} instances   "
            f"{blocked:4d} blocked by an unrepairable in-seam source "
            f"({carried:4d} carried)   "
            f"CEILING {len(here) - blocked:4d}   "
            f"of which both arms admit: {paired:4d}")
    # ------------------------------------------------- THE BUCKET DECOMPOSITION
    # Every one of the PRICED instances (403 on the 2026-09-03 basis, 387 before it)
    # lands in exactly ONE bucket and the buckets must total that count. The floor at
    # the end is not decoration: when the structural
    # verdict below was added to the OTHER board its ledger silently fell to 153 of
    # 172 because the new verdict head matched no filter, and the run still printed
    # a headline. A decomposition that does not have to add up does not measure.
    #
    # THE STENCIL BUCKET. `stepping._offdiagonal_terms` (stepping.py:1235-1253) reads
    # each partner component's D volume at four indices, and `step_D` writes those
    # volumes IN PLACE in the first half of the same launch; there is no grid-wide
    # barrier inside one launch on Triton. Its only two callers are inside `update_E`
    # (stepping.py:1006 and, via `_nonlinear_constitutive`, :1099), so this is a D->E
    # fact and NOT a B->H one — the B->H off-diagonal cell below is NOT in this
    # bucket, because `update_H` never calls the stencil. Measured on an RTX A6000 by
    # the 2026-08-20 off-diagonal round, both float32 subnormal cuts, on a kernel
    # spliced verbatim out of the two shipped bodies.
    #
    # DERIVED FROM THE CENSUS, NOT FROM AN ARM NAME: a D->E instance is in the bucket
    # when its row declares has_offdiagonal_epsilon.
    buckets: Dict[str, int] = collections.Counter()
    for entry in instances:
        cell = (entry["seam"], entry["curl_arm"], entry["constitutive_arm"])
        if entry["in_seam_source_blocks"]:
            buckets["blocked by the source injection in the seam"] += 1
        elif (entry["predicate_admits"]
              and entry["product"] in CREDIT_WITHHELD_STALE_BINDING):
            # NOT "served". The predicate admits, and this board will not credit it:
            # the gate that certified this product ran against bytes the tree no
            # longer ships (CREDIT_WITHHELD_STALE_BINDING). Its own bucket rather
            # than "not built", because the product IS built -- a bucket that said
            # otherwise would send a reader to write a kernel that already exists.
            buckets["reachable; a product exists and its predicate ADMITS, but its "
                    "RELEASE BINDING is stale - credit withheld"] += 1
        elif entry["predicate_admits"]:
            buckets["SERVED by a shipped product"] += 1
        elif entry["curl_arm"] is None or entry["constitutive_arm"] is None:
            buckets["reachable; a slot has NO admitting arm (array path)"] += 1
        elif entry["constitutive_arm"] == NULL_CONSTITUTIVE_ARM:
            buckets["NOT A FUSION CANDIDATE — the constitutive sub-step launches nothing, so there is no second kernel to weld"] += 1
        elif entry["product"] is not None:
            buckets["reachable; a product exists and REFUSES on another clause"] += 1
        elif (entry["seam"] == "D->E"
              and bool(configuration_of[entry["row"]]["has_offdiagonal_epsilon"])):
            buckets["reachable; STRUCTURALLY UNFUSABLE — the constitutive arm is a "
                    "STENCIL over the curl arm's in-place output"] += 1
        else:
            buckets["reachable; NO PRODUCT OCCUPIES THIS CELL — not built"] += 1
    if not any(e["constitutive_arm"] == NULL_CONSTITUTIVE_ARM for e in instances):
        raise SystemExit(
            f"no instance names the constitutive arm {NULL_CONSTITUTIVE_ARM!r}, so "
            f"the null-constitutive bucket filters nothing and is silently disabled. "
            f"Constitutive arms present: "
            f"{sorted({str(e['constitutive_arm']) for e in instances})}")
    # A PREDICATE MAY ONLY CLEAR AN IN-SEAM SOURCE IF ITS MODULE SAYS IT CARRIES THE
    # REPAIR. The 51 instances this board now credits across a live injection are a
    # census verdict; this asks the SHIPPED FLAG the same question and requires the
    # two to agree. A census read against a different tree, or a product that learned
    # to admit a deposit without the bracket that repairs it, both look exactly like
    # a coverage gain until this fails. Import the module, never a name table:
    # `launch._install_fused_pair` reads the same attribute to decide whether to put
    # a TrailingRepairPlan in the seam's second slot.
    import importlib  # noqa: PLC0415
    import re as _re  # noqa: PLC0415
    carrying = {e["product"] for e in instances
                if e["in_seam_source"] and e["predicate_admits"] and e["product"]}
    for family in sorted(carrying):
        match = _re.match(r"([a-z0-9_]+)\.py", PRODUCTS[family]["module"])
        if not match:
            raise SystemExit(f"{family}: 'module' does not name a triton_kernels "
                             f"file: {PRODUCTS[family]['module']!r}")
        module = importlib.import_module(
            f"meep_gpu.triton_kernels.{match.group(1)}")
        if not getattr(module, "CARRIES_DEPOSIT_REPAIR", False):
            raise SystemExit(
                f"{family}'s predicate admits a seam-instance carrying an in-seam "
                f"source, but meep_gpu/triton_kernels/{match.group(1)}.py declares "
                f"CARRIES_DEPOSIT_REPAIR False. Either the census was read against a "
                f"different tree, or a product admits a deposit nothing repairs.")
    log(f"\n  deposit repair, cross-checked against the shipped flag: "
        f"{sorted(carrying)} admit an in-seam source and every one declares "
        f"CARRIES_DEPOSIT_REPAIR")

    # THE TWO SERVED NUMBERS MUST BE ONE NUMBER. The aggregate counts `served`
    # directly off the instances; the bucket table reaches "SERVED by a shipped
    # product" only after the structural branches above have had their turn. They
    # agreed on every census through 2026-08-21 and DISAGREED on the first fresh one
    # -- 150 against 99 -- because the source-seam branch still treated every in-seam
    # injection as fatal after three products had learned to carry it. The run
    # printed both, in the same log, and neither number was flagged. It raises now.
    if buckets["SERVED by a shipped product"] != served_total:
        raise SystemExit(
            f"the aggregate says {served_total} seam-instances are SERVED and the "
            f"bucket "
            f"table says {buckets['SERVED by a shipped product']}. One board may not "
            f"publish two answers to its own headline: a structural branch above the "
            f"SERVED branch is claiming instances whose shipped predicate admits "
            f"them, which means the branch's premise has stopped being true.")
    log("\n=== THE BUCKETS — every seam-instance, exactly once ===")
    for name, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
        log(f"    {count:4d}  {name}")
    log(f"    {sum(buckets.values()):4d}  TOTAL")
    if sum(buckets.values()) != denominator:
        raise SystemExit(
            f"the buckets total {sum(buckets.values())} against a denominator of "
            f"{denominator}; some instance falls into none or into two, and the "
            f"decomposition does not measure. Buckets: {dict(buckets)}")

    # ------------------------------------------------- THE RECONCILED TAXONOMY
    # THE SAME EIGHT NAMES THE METAL AND CUDA BOARDS EMIT (release decision R1; the
    # eighth, withdraw_seam, arrived with the H->D seam on 2026-09-04),
    # filed over the SAME instances the decomposition above walks. That
    # decomposition is kept unchanged and is not replaced -- it is this board's own
    # vocabulary, and its sentences are what the sub-reasons below carry. What the
    # taxonomy adds is a decomposition a reader can lay beside the sibling boards
    # without translating three vocabularies, and the three tiers (R2) the headline
    # now reads against.
    #
    # THE PRECEDENCE DIFFERS FROM THE BLOCK ABOVE IN ONE PLACE AND IT IS R4's:
    # `not_fusion_surface` is asked FIRST, ahead of the source seam, because an
    # instance whose constitutive half launches nothing has no weld for a deposit to
    # obstruct. The move it makes on this board is counted below rather than
    # assumed to be zero.
    taxonomy = fusion_taxonomy.Taxonomy(
        backend="triton",
        priced=priced,
        rows=len(record),
        platform_note=(
            "TRITON HAS NO ANALOGUE OF METAL'S ARGUMENT-TABLE CEILING, and this "
            "zero is measured rather than skipped: a Triton kernel takes its "
            "pointers as ordinary kernel arguments and the JIT imposes no "
            "MAX_BUFFER_BINDINGS-style cap, so no cell on this board is ever "
            "refused for the number of volumes its fused signature would bind. The "
            "REAL per-backend constraints here are named in their own buckets "
            "instead -- the composer's specialized_family_owns_the_grid guard and "
            "_pair_may_absorb are DISPATCH clauses, counted under "
            "planner_can_compose_it and never mixed into a fusion verdict"),
        seam_instances={**{seam: per_seam[seam] for seam in ("B->H", "D->E", "E->P")},
                        "H->D": len(h_to_d_entries)})
    for entry in instances:
        seam, row_label = entry["seam"], entry["row"]
        configuration = configuration_of[row_label]
        if entry["constitutive_arm"] == NULL_CONSTITUTIVE_ARM:
            taxonomy.add(
                "not_fusion_surface", seam, row_label,
                "NOT A FUSION CANDIDATE — the constitutive sub-step launches "
                "nothing, so there is no second kernel to weld",
                {"constitutive_arm": entry["constitutive_arm"]})
        elif (entry["predicate_admits"]
              and entry["product"] not in CREDIT_WITHHELD_STALE_BINDING):
            taxonomy.add("served", seam, row_label,
                         f"SERVED by a shipped product ({entry['product']})",
                         {"product": entry["product"]})
        elif entry["in_seam_source_blocks"]:
            taxonomy.add(
                "source_seam_unbracketable", seam, row_label,
                "blocked by the source injection in the seam — the driver deposits "
                "between the halves and deposit_repair.repairable REFUSES to "
                "reconstruct this row's constitutive half BY NAME",
                {"refused_by": sorted(
                    k for k, v in (
                        ("off-diagonal chi1inv",
                         bool(configuration["has_offdiagonal_epsilon"])),
                        ("instantaneous chi2/chi3",
                         bool(configuration["has_nonlinearity"])),
                        ("an unreadable fold map",
                         bool(configuration["has_symmetry"])))
                    if v)})
        elif entry["predicate_admits"]:
            # THE STALE-BINDING REFUSAL, CARRIED THROUGH THE RENAME. The product IS
            # built and its predicate DOES admit; what this board will not do is
            # CREDIT it, because the gate that certified it ran against bytes the
            # tree no longer ships. None of the eight names says "built but
            # uncredited", so it lands in `buildable_not_built` with its own
            # sentence intact -- and the sentence is the part that stops a reader
            # going off to write a kernel that already exists.
            #
            # THE SENTENCE'S ONE HOME IS fusion_taxonomy (2026-09-10), so the fourth
            # seam (`h_to_d_seam.classify`, handed `credit_withheld` per row below)
            # files the IDENTICAL key under `buildable_not_built` and the bucket's
            # sub_reasons carry one withheld spelling across all four seams. NOTE
            # THE RUNG: this walk files withheld BELOW `source_seam_unbracketable`
            # (a withheld product on a source-blocked row lands in that bucket
            # above), while the fourth seam decides withheld AT THE SERVED RUNG,
            # above `withdraw_seam`, because a withdraw row's attainability was
            # measured by the product's own licensed gate. No board can reach the
            # divergent case today; "withholding moves nothing else" is a statement
            # about the H->D seam.
            taxonomy.add(
                "buildable_not_built", seam, row_label,
                fusion_taxonomy.credit_withheld_sub_reason(entry["product"]),
                {"product": entry["product"], "credit_withheld": True,
                 "credit_withheld_reason":
                     CREDIT_WITHHELD_STALE_BINDING[entry["product"]]})
        elif entry["curl_arm"] is None or entry["constitutive_arm"] is None:
            curl_slot, const_slot = SEAM_SLOT_NAMES[seam]
            missing_slot = const_slot if entry["constitutive_arm"] is None \
                else curl_slot
            partner = (entry["curl_arm"] if entry["constitutive_arm"] is None
                       else entry["constitutive_arm"])
            taxonomy.add(
                "missing_half", seam, row_label,
                f"A KERNEL COVERAGE GAP, NOT A FUSION REFUSAL: no certified arm "
                f"admits {missing_slot} on this row, so the pair has no second body "
                f"to weld and the array path steps the slot "
                f"(launch.py:1983-2010, :2084 _pair_may_absorb)",
                {"missing_slot": missing_slot,
                 "missing_kernel":
                     f"a triton_kernels arm on {missing_slot} admitting this row — "
                     f"the seam's other half is already covered by {partner!r}, so "
                     f"what is absent is the {missing_slot} body for that cell. "
                     f"Closing it is a SUB-STEP kernel's job, not a fusion one",
                 "partner_arm": partner,
                 "arms_that_admit_this_slot_today": (
                     entry["constitutive_admitters"]
                     if entry["constitutive_arm"] is None
                     else entry["curl_admitters"]),
                 "census_note": (
                     "the census carries no update_P column, so THAT half is read "
                     "from the 2026-08-20 CUDA battery; the update_E half is read "
                     "from this census (since 2026-09-04)"
                     if entry.get("update_P_slot_absent_from_the_census")
                     else entry.get("planner_refusal"))})
        elif (seam == "D->E"
              and bool(configuration["has_offdiagonal_epsilon"])):
            taxonomy.add(
                "structurally_unweldable", seam, row_label,
                "reachable; STRUCTURALLY UNFUSABLE — the constitutive arm is a "
                "STENCIL over the curl arm's in-place output: "
                "stepping._offdiagonal_terms (stepping.py:1235-1253) reads each "
                "partner component's D volume at four indices and step_D writes "
                "those volumes IN PLACE in the first half of the same launch, with "
                "no grid-wide barrier inside one Triton launch. MEASURED on an RTX "
                "A6000 by the 2026-08-20 off-diagonal round, both float32 subnormal "
                "cuts, on a kernel spliced verbatim out of the two shipped bodies",
                {"kind": "stencil"})
        elif entry["product"] is not None:
            taxonomy.add(
                "buildable_not_built", seam, row_label,
                f"a product exists on this cell ({entry['product']}) and REFUSES on "
                f"a clause other than the source seam — what is missing is coverage "
                f"inside a built kernel, not a kernel",
                {"product": entry["product"]})
        else:
            taxonomy.add(
                "buildable_not_built", seam, row_label,
                "reachable; NO PRODUCT OCCUPIES THIS CELL — not built",
                {"cell": [entry["curl_arm"], entry["constitutive_arm"]]})
    # The fourth seam, filed by the shared rule. What this board hands it is its own
    # PRODUCTS table: which entries span update_H+step_D (none today), and — the day
    # one does — a per-row `served_by` verdict, which is the only thing that can carry
    # a per-backend predicate into a rule three boards share. h_to_d_seam.price NAMES
    # the rows a board leaves unanswered rather than pricing them as unbuilt.
    h_to_d_block = h_to_d_seam.price(
        taxonomy, "triton", "triton_kernels", h_to_d_entries,
        products_spanning_the_seam=[
            name for name, spec in PRODUCTS.items()
            if fusion_taxonomy.canonical_seam(spec["seam"]) == h_to_d_seam.SEAM])
    # `h_to_d_block["served"]` already excludes the withheld rows, so the served
    # floor below needs no arithmetic of its own for them.
    log(f"      of which the CREDIT IS WITHHELD on H->D (stale binding): "
        f"{len(h_to_d_block['credit_withheld_rows'])}")
    taxonomy_block = taxonomy.finish(
        served_expected=served_total + h_to_d_block["served"],
        served_expected_source=(
            f"`served_total` — predicate_admits minus the instances whose product's "
            f"release binding is stale, plus the H->D seam's served "
            f"({h_to_d_block['served']} of {h_to_d_block['instances_priced']}; "
            f"{h_to_d_block['served_note']})"))

    # THE TWO DECOMPOSITIONS MUST RECONCILE EXACTLY, and R4's precedence is the only
    # licensed difference: an instance both null-constitutive and source-blocked is
    # filed under the source seam above and under `not_fusion_surface` here.
    _ledger_blocked = buckets["blocked by the source injection in the seam"]
    _ledger_null = buckets[
        "NOT A FUSION CANDIDATE — the constitutive sub-step launches nothing, so "
        "there is no second kernel to weld"]
    _moved = _ledger_blocked - taxonomy_block["buckets"]["source_seam_unbracketable"]
    # The H->D seam's null rows are the same null update_H the bucket table counts
    # at B->H, filed once more at the fourth seam; the table prices three seams and
    # the taxonomy four, so the difference is exactly that count.
    _h_to_d_null = h_to_d_block["buckets"].get("not_fusion_surface", 0)
    if taxonomy_block["buckets"]["not_fusion_surface"] != (
            _ledger_null + _moved + _h_to_d_null):
        raise SystemExit(
            f"the bucket table and the reconciled taxonomy do not reconcile: "
            f"{_ledger_null} null-constitutive and {_ledger_blocked} source-blocked "
            f"against {taxonomy_block['buckets']['not_fusion_surface']} and "
            f"{taxonomy_block['buckets']['source_seam_unbracketable']}, with R4's "
            f"precedence accounting for only {_moved} and the H->D seam's null "
            f"update_H for {_h_to_d_null}. One walk is reading a different corpus.")
    taxonomy_block["r4_precedence_move"] = {
        "instances_moved_from_source_seam_to_not_fusion_surface": _moved,
        "what_it_is": (
            "instances this board's own bucket table filed as blocked by the source "
            "injection and whose constitutive sub-step launches NOTHING. Under R4 "
            "an instance is counted in its most fundamental bucket: where there is "
            "no second kernel to weld, the deposit obstructs nothing"),
        "the_bucket_table_still_says": {"source_blocked": _ledger_blocked,
                                        "null_constitutive": _ledger_null},
    }
    taxonomy.report(log)
    log(f"\n  R4 PRECEDENCE: {_moved} instance(s) move from "
        f"source_seam_unbracketable to not_fusion_surface on this board "
        f"(the bucket table's own order files them the other way)")

    total_ceiling = sum(v["ceiling"] for v in seam_ceiling.values())
    reachable = sum(v["reachable_ceiling_both_arms_admit"] for v in seam_ceiling.values())
    log(f"    TOTAL CEILING on any fused Triton path over this corpus: "
        f"{total_ceiling} / {denominator} ({100.0 * total_ceiling / denominator:.1f}%)")
    log(f"    REACHABLE CEILING (both arms admit today)              : "
        f"{reachable} / {denominator} ({100.0 * reachable / denominator:.1f}%)")
    log(f"    ACHIEVED TODAY: {b} / {reachable} of the reachable ceiling "
        f"({100.0 * b / reachable:.1f}%)")

    # ---------------------------------------------------------------- E->P
    # Two facts this cut asks that no earlier one could, printed together because
    # they are the two things that stand between the Triton E->P seam and the ten
    # instances the Metal backend already serves there.
    e_to_p = [e for e in instances if e["seam"] == "E->P"]
    stale = [e for e in e_to_p if e["update_P_slot_absent_from_the_census"]]
    log("\n=== E->P: WHAT BLOCKS THIS SEAM ON TRITON, MEASURED ===")
    log("  (1) THE 2026-08-16 CENSUS CANNOT SEE THE update_P SLOT — SO THIS SEAM IS")
    log("      READ FROM THE 2026-08-20 CUDA BATTERY INSTEAD.")
    log(f"      {len(stale)} of {len(e_to_p)} E->P instances record NO admitting arm at")
    log("      update_P in the 08-16 record, whose plan_step.slots block carries only")
    log("      step_B / step_D / update_E / update_H on every row. update_P is in")
    log("      STEP_ORDER (launch.py:1874-1877) and plan_step resolves an ADE family")
    log("      for it (launch.py:2984-3048), so that is a stale RECORD, not an absent")
    log("      slot. The 08-20 CUDA battery DOES carry both halves, and every one of")
    log(f"      the {len(e_to_p)} E->P rows is present in it (asserted in build_matrix,")
    log("      not assumed), so both arms below are read from that record. B->H and")
    log("      D->E stay on the 08-16 record because only it carries")
    log("      source_field_types, which those two seams need and this one does not.")
    both = sum(1 for e in e_to_p if e["curl_arm"] and e["constitutive_arm"])
    log(f"      WITH BOTH HALVES READABLE: {both} of {len(e_to_p)} E->P instances have an")
    log("      admitting arm on EACH side — a pair exists to fuse. NONE is served,")
    log("      because no Triton product occupies an (update_E, update_P) cell:")
    log("      fused_dispersive_chain spans step_D -> update_E -> update_P and is")
    log("      scored at its own D->E cell. Those instances are REACHABLE, NOT BUILT.")
    # THE GAP RANKING BELOW WILL UNDER-REPORT THIS SEAM, so the number is printed
    # here instead of being left to be discovered. That ranking treats a cell as
    # owned when ANY instance in it names a product, and at E->P a cell can be
    # PARTLY owned: `fused_dispersive_chain` is keyed on its D->E cell, so within
    # the one (dispersive -> ADE update_P) cell it claims the unfolded rows and not
    # the folded ones. Those unclaimed instances are gaps and would otherwise be
    # invisible. (This cannot happen at B->H or D->E, where the product is looked up
    # by the very cell the instance is keyed on.)
    unowned = [e for e in e_to_p
               if e["curl_arm"] and e["constitutive_arm"] and not e["product"]]
    log(f"      OF THOSE {both}, {len(unowned)} SIT IN NO PRODUCT'S CELL AT ALL:")
    for e in sorted(unowned, key=lambda x: x["row"]):
        log(f"        {e['row'][:56]:56s} K={e['poles']} "
            f"{e['curl_arm']} -> {e['constitutive_arm']}")
    log("      FOR SCALE, THE OTHER BACKEND: metal_kernels/fused_ade_chain.py serves")
    log("      10 of these 15 today (../fusion_matrix_metal_2026-08-20_closed/, E_to_P")
    log(f"      10/15). It claims 10 rather than the {both} above because it declines the")
    log("      four complex-storage rows and the one off-diagonal row BY NAME. A")
    log("      Triton twin of that product would carry the same 10; of the remaining")
    log(f"      {both - 10}, four are the complex64 chain (complex_fused_ade_chain,")
    log("      built and gated 2026-08-30) and the last is the OFF-DIAGONAL row, which")
    log("      folded_offdiag_fused_ade_chain claims on this backend and which no Metal")
    log("      product does. That is the difference the two boards now report.")
    log("  (2) THE ONE TRITON PRODUCT AT THIS SEAM IS BOUNDED AT ONE POLE PER")
    log("      COMPONENT, AND THAT BOUND IS WORTH ZERO ROWS HERE.")
    poles = collections.Counter(e["poles"] for e in e_to_p)
    log(f"      susceptibilities per E->P row, measured: {dict(sorted(poles.items()))}")
    single = sum(count for k, count in poles.items() if k == 1)
    log(f"      rows at ONE susceptibility: {single} of {len(e_to_p)}")
    log("      fused_dispersive_chain declares CHAIN_MAX_POLES = 1 and refuses K > 1")
    log("      BY NAME (fused_dispersive_chain.py, the SCOPE block: the sub-step-share")
    log("      probe left update_P at 51.9% at K=6 with a 9.5-28.3pp within-run spread,")
    log("      so the precondition for multi-pole ADE fusion is bracketed, not cleared).")
    # WHY THAT BOUND IS WORTH ZERO, MEASURED PER ROW rather than asserted. Every
    # K = 1 row must ALSO be shown to fail on a second, independent clause;
    # otherwise the sentence "its admitted count of 0 is structural" is a claim
    # with nothing behind it. The two clauses checked are the ones the K = 1 rows
    # trip and are read from the configuration block, not from a product.
    by_row = {label(r): r for r in record}
    survivors: List[str] = []
    for e in e_to_p:
        if e["poles"] != 1:
            continue
        configuration = by_row[e["row"]]["configuration"]
        blockers = []
        if configuration["has_offdiagonal_epsilon"]:
            blockers.append("off-diagonal chi1inv")
        if configuration["force_complex_fields"]:
            blockers.append("complex storage")
        log(f"        K=1 {e['row'][:52]:52s} also refused by: "
            f"{', '.join(blockers) or 'NOTHING ELSE'}")
        if not blockers:
            survivors.append(e["row"])
    if survivors:
        log(f"      *** {len(survivors)} K=1 row(s) are refused by NOTHING ELSE: "
            f"{survivors} — the K bound alone is what costs them, and this cut's "
            f"claim that the zero is structural DOES NOT HOLD.")
    else:
        log(f"      All {single} K=1 rows are refused by a SECOND, independent clause,")
        log("      so the admitted count of 0 is structural rather than incidental.")
    log("      A Triton product that serves this corpus at E->P must therefore carry")
    log("      K up to 6, which is a different kernel, not a widened predicate.")

    log("\n=== THE CELLS THE CORPUS DRIVES, and whether a product covers them ===")
    log("  'ceiling' = instances at that cell whose in-seam source, if any, a"
        " deposit repair can carry; the most any")
    log("  product for that cell could ever serve.")
    ranked: List[dict] = []
    for seam in ("B->H", "D->E", "E->P"):
        counts = collections.Counter(
            (e["curl_arm"], e["constitutive_arm"]) for e in instances if e["seam"] == seam)
        log(f"\n  --- {seam} ---")
        for cell, count in counts.most_common():
            here = [e for e in instances
                    if e["seam"] == seam and (e["curl_arm"], e["constitutive_arm"]) == cell]
            # ONE CELL CAN CARRY MORE THAN ONE PRODUCT ANSWER, and taking the first
            # instance's would report an ambiguity as a decision. It happens at
            # E->P: the CUDA battery's `cuda_dispersive` column does not separate
            # FOLDED dispersive from unfolded, so both land in one cell, while
            # `fused_dispersive_chain` — scored at its own D->E cell — claims only
            # the unfolded rows. Distinct owners are named, not collapsed.
            owners = sorted({e["product"] for e in here if e["product"]})
            product = owners[0] if len(owners) == 1 else (owners[0] if owners else None)
            served = sum(1 for e in here if e["predicate_admits"])
            ceiling = sum(1 for e in here if not e["in_seam_source_blocks"])
            if cell[0] is None and cell[1] is None:
                status = "NO ARM EITHER SIDE — nothing to fuse"
            elif cell[0] is None or cell[1] is None:
                status = "HALF THE PAIR IS ON THE ARRAY PATH — nothing to fuse"
            elif owners:
                claimed = sum(1 for e in here if e["product"])
                status = (f"product {', '.join(owners)}: admits {served}/{count}"
                          + (f" (claims only {claimed} of the {count})"
                             if claimed != count else ""))
            else:
                status = "NO FUSED PRODUCT"
            log(f"    {count:4d}  ceiling {ceiling:4d}   {str(cell[0]):28s} -> "
                f"{str(cell[1]):26s}  {status}")
            ranked.append({
                "seam": seam, "curl_arm": cell[0], "constitutive_arm": cell[1],
                "rows_driving_it": count, "ceiling_no_in_seam_source": ceiling,
                "product": product, "seam_instances_the_product_admits": served,
                "fusable": bool(cell[0] and cell[1]),
                # THE NULL ARM LAUNCHES NOTHING. The array path's update_H returns
                # at stepping.py:944-945 under an inactive absorber and update_E at
                # :983-984, which is what no_pml_constitutive.py:610-613 transcribes
                # and what launch.py:2331-2338 says the arm is. A pair fusing a curl
                # into it eliminates ZERO launches, so its ceiling is real coverage
                # and zero throughput.
                "constitutive_launches_nothing": cell[1] == "no-PML null",
                # THE STENCIL. True on a D->E cell every one of whose REACHABLE
                # instances declares has_offdiagonal_epsilon: stepping.
                # _offdiagonal_terms (stepping.py:1235-1253) reads each partner's D
                # volume at four indices, so the constitutive arm there is a STENCIL
                # and not a pointwise map. Recorded per cell so the ranking below
                # cannot advertise such a cell as an ordinary buildable gap, which
                # is what the Metal ranking did for one of these until 2026-08-20.
                #
                # WHAT THIS FLAG DOES NOT SAY, SINCE 2026-09-02. It used to read
                # "is refused structurally on any backend", on the strength of
                # `results/triton_fused_offdiag_electric_2026-08-20` — and that
                # artifact measured a real, schedule-dependent race, in a weld of
                # the SHIPPED IN-PLACE halves. Two products now occupy two of these
                # cells and are bit-identical on device at every block size
                # (`results/triton_offdiag_stencil_welds_2026-09-02`), because the
                # curl half writes launch-local SCRATCH and the constitutive half
                # re-derives every foreign tap from pre-launch state — so the
                # in-place premise the refusal rested on is gone. THE FLAG IS A
                # FACT ABOUT THE ROWS, and the ranking below only ever prints its
                # refusal note for a cell with NO product, which is where the
                # premise still holds.
                "structurally_unfusable_stencil": bool(
                    seam == "D->E"
                    and [e for e in here if not e["in_seam_source_blocks"]]
                    and all(configuration_of[e["row"]]["has_offdiagonal_epsilon"]
                            for e in here if not e["in_seam_source_blocks"])),
            })

    log("\n=== THE GAPS, RANKED BY WHAT THEY COULD ACTUALLY SERVE ===")
    log("  A gap is a (curl, constitutive) cell the corpus drives with NO fused")
    log("  product. It is ranked by CEILING, not by rows driving it: a cell whose")
    log("  rows all carry an UNREPAIRABLE in-seam source is worth ZERO however many")
    log("  rows reach it.")
    log("  A cell with one half on the array path is not a gap — there is no pair.")
    gaps = sorted((g for g in ranked if g["fusable"] and not g["product"]),
                  key=lambda g: (-g["ceiling_no_in_seam_source"], -g["rows_driving_it"]))
    for entry in gaps:
        note = ""
        if not entry["ceiling_no_in_seam_source"]:
            note = ("   <-- WORTH ZERO (every row carries an in-seam source no "
                    "deposit repair can carry)")
        elif entry.get("structurally_unfusable_stencil"):
            note = ("   <-- STENCIL: the constitutive arm reads each partner's D "
                    "at four indices (stepping.py:1235-1253). An IN-PLACE weld of "
                    "the shipped halves races here, measured 2026-08-20; a "
                    "SCRATCH-OUTPUT weld does not, measured 2026-09-02 on the two "
                    "cells now served. Buildable, but not as an ordinary pair")
        elif entry["constitutive_launches_nothing"]:
            note = "   <-- ZERO THROUGHPUT (the null arm launches nothing)"
        log(f"    ceiling {entry['ceiling_no_in_seam_source']:4d}  "
            f"(rows {entry['rows_driving_it']:4d})  {entry['seam']:5s}  "
            f"{entry['curl_arm']:31s} -> {entry['constitutive_arm']}{note}")
    worthless = [g for g in gaps if not g["ceiling_no_in_seam_source"]]
    null_arm = [g for g in gaps
                if g["ceiling_no_in_seam_source"] and g["constitutive_launches_nothing"]]
    stencil = [g for g in gaps if g.get("structurally_unfusable_stencil")
               and g["ceiling_no_in_seam_source"]]
    log(f"  gap cells: {len(gaps)}, of which {len(worthless)} are worth ZERO, "
        f"{len(null_arm)} more fuse into an arm that launches nothing, and "
        f"{len(stencil)} are STRUCTURALLY UNFUSABLE "
        f"({sum(g['ceiling_no_in_seam_source'] for g in stencil)} seam-instances)")
    log(f"  BUILDABLE gap ceiling, with those three classes removed: "
        f"{sum(g['ceiling_no_in_seam_source'] for g in gaps if not g['constitutive_launches_nothing'] and not g.get('structurally_unfusable_stencil'))}")
    log(f"  seam-instances every gap together could serve: "
        f"{sum(g['ceiling_no_in_seam_source'] for g in gaps)} "
        f"(rows driving them: {sum(g['rows_driving_it'] for g in gaps)})")

    log("\n  BUILDABLE GAPS WITH A MEASURED RECIPE (the hand-off):")
    printed = 0
    for cell, recipe in BUILDABLE_GAP_RECIPES.items():
        matching = [g for g in gaps
                    if (g["seam"], g["curl_arm"], g["constitutive_arm"]) == cell]
        ceiling = sum(g["ceiling_no_in_seam_source"] for g in matching)
        log(f"    {cell[0]:5s}  {cell[1]:24s} -> {cell[2]:20s} ceiling {ceiling}")
        for line in textwrap.wrap(recipe, 96):
            log(f"        {line}")
        printed += 1
    if not printed:
        log("    (none)")

    log("\n  THE SHORTFALL WITHIN EXISTING PRODUCTS (cell reached, predicate refuses):")
    for name in PRODUCTS:
        here = [e for e in instances if e["product"] == name]
        shortfall = [e for e in here if not e["predicate_admits"]]
        if not shortfall:
            continue
        sourced = sum(1 for e in shortfall if e["in_seam_source_blocks"])
        log(f"    {name:30s} {len(shortfall):4d} refused, of which {sourced:4d} "
            f"for an UNREPAIRABLE in-seam source and {len(shortfall) - sourced:4d} "
            f"for something else")

    log("\n=== CROSS-CHECK against the two 2026-08-19 per-product censuses ===")
    checks: Dict[str, Any] = {}
    for name, path, key in (
            ("folded_fused_magnetic_pair",
             "triton_folded_fused_magnetic_pair_census_2026-08-19",
             "folded_fused_magnetic_pair_B_to_H"),
            ("fused_pair_B",
             "triton_folded_fused_magnetic_pair_census_2026-08-19",
             "plain_fused_pair_B_already_wired"),
            ("complex_fused_magnetic_pair",
             "triton_complex_fused_magnetic_pair_census_2026-08-19T2355",
             "complex_fused_magnetic_pair_B_to_H")):
        artifact = RESULTS / path / "corpus_admission.json"
        if not artifact.exists():
            checks[name] = "the prior artifact is absent"
            continue
        prior = json.loads(artifact.read_text())[key]
        mine = {e["row"] for e in instances
                if e["product"] == name and e["predicate_admits"]}
        agree = set(prior["rows"]) == mine and prior["admitted"] == len(mine)
        checks[name] = {"prior": prior["admitted"], "here": len(mine),
                        "row_sets_identical": set(prior["rows"]) == mine}
        log(f"    {name:30s} prior {prior['admitted']:3d}  here {len(mine):3d}  "
            f"row sets identical: {set(prior['rows']) == mine}  "
            f"{'AGREE' if agree else '*** DISAGREE ***'}")

    # THE KEY MUST CARRY THE NUMBER THE LOG PRINTED. It did not, once: the headline
    # was bound to a bare `served`, a later loop in this same function rebound that
    # name, and the JSON shipped 0 while the log said 150 and the bucket floor above
    # passed. Recomputed here straight from the instances, so the key cannot drift
    # from its own definition however this function grows.
    _served_check = sum(1 for e in instances if e["predicate_admits"]
                        and e["product"] not in CREDIT_WITHHELD_STALE_BINDING)
    if _served_check != served_total:
        raise SystemExit(
            f"the SERVED key is about to be written as {served_total} but the "
            f"instances say {_served_check}; a name in this function has been "
            f"rebound between the headline and the return.")
    return {
        "cross_check_against_2026_08_19": checks,
        "why_not_planner_composable": dict(unreachable),
        "seam_ceiling": seam_ceiling,
        "total_ceiling": total_ceiling,
        "reachable_ceiling": reachable,
        # THE RECONCILED DECOMPOSITION, in full and under the key the Metal and CUDA
        # boards use. The bucket table above is unchanged and stays: this is the
        # translation, not a replacement, and every reconciled bucket carries this
        # board's own prose beneath it as the sub-reason.
        "taxonomy": taxonomy_block,
        "taxonomy_buckets": taxonomy_block["buckets"],
        "tiers": {name: block["value"]
                  for name, block in taxonomy_block["tiers"].items()},
        "denominator": priced,
        "denominator_three_seam_ledger": denominator,
        "denominator_justification": taxonomy.priced_justification,
        "denominator_parts": {"B->H": per_seam["B->H"], "D->E": per_seam["D->E"],
                              "E->P": per_seam["E->P"],
                              "H->D": h_to_d_block["instances_priced"]},
        # THE FOURTH SEAM'S OWN BLOCK: what sits between the halves (located in
        # driver.py at cut time), the probe it was priced from, its buckets and its
        # instances. The bucket table and the ledger keys above stay three-seam.
        "h_to_d_seam": h_to_d_block,
        "a_product_exists_for_the_cell": a,
        # THREE KEYS NOW. `predicate_admits` is what `served_by_a_fused_product`
        # counted in every cut through 2026-08-21, when every credited product's gate
        # still described the tree and the two numbers were equal; it is the
        # three-seam walk's admitted count. `served_three_seam_ledger` is that walk's
        # CREDITED count (admitted AND credit-bound), which is what the headline key
        # carried through the 2026-09-06 `_hd` cut — 356 beside a four-seam taxonomy
        # bucket of 403 in one artifact, because the key had never been taught the
        # fourth seam. `served_by_a_fused_product` — the name every downstream reader
        # treats as the headline — is now DERIVED from the instance ledger the
        # taxonomy sums, and `fusion_taxonomy.headline_agreement` refuses the cut if
        # any served key in the artifact disagrees with it.
        "predicate_admits": b,
        "served_by_a_fused_product": taxonomy_block["buckets"]["served"],
        "served_three_seam_ledger": served_total,
        "credit_withheld_stale_binding": withheld_instances,
        # The FOURTH seam's withheld count, kept apart from the three-seam key above
        # (whose denominator is the three-seam ledger); the per-row list is at
        # aggregate.h_to_d_seam.credit_withheld_rows.
        "credit_withheld_h_to_d_seam": len(h_to_d_block["credit_withheld_rows"]),
        "credit_withheld_products": sorted(
            set(CREDIT_WITHHELD_STALE_BINDING) & set(PRODUCTS)),
        "planner_can_compose_it": c,
        "served_in_dispatch": dispatch_reach["served_in_dispatch"],
        "served_in_dispatch_measurement": dispatch_reach,
        "per_product": {name: {"cell_reached": by_product[name],
                               "predicate_admits": admits_by[name],
                               "planner_reachable": reach_by[name]}
                        for name in PRODUCTS},
        "cells": ranked,
        "gaps_ranked": gaps,
        "buckets": dict(buckets),
        "buckets_total_the_denominator": sum(buckets.values()) == denominator,
    }


def funnels(record: List[dict]) -> Dict[str, Any]:
    """Which clause is the binding one, per derived product. Printed, not inferred."""
    log("\n=== THE FUNNELS: which clause costs what ===")
    out: Dict[str, Any] = {}
    for name, clauses, order in (
            # `no_folded_periodic_axis` IS GONE from this ladder, which is the whole
            # of this cut: the kernel carries fill_folded_far_ghosts_B now. The
            # ladder below is the shipped predicate's, so the funnel prints the
            # measured admission and not a historical one. The D/E twin keeps the
            # clause — that carry is NOT built.
            ("folded_fused_magnetic_pair", folded_magnetic_clauses,
             ["curl_half", "constitutive_half", "no_magnetic_source",
              "fold_stores_the_source_row"]),
            ("complex_fused_magnetic_pair", complex_magnetic_clauses,
             ["curl_half", "constitutive_half", "no_magnetic_source",
              "no_symmetry_in_the_seam"]),
            ("fused_dispersive_chain", chain_clauses,
             ["pair_half", "exactly_one_susceptibility", "ade_half", "pml_active"]),
            # ---- the four products the 08-20b/c cuts added, funnelled here too --
            ("folded_fused_pair", folded_electric_clauses,
             ["curl_half", "constitutive_half", "no_electric_source",
              "no_folded_periodic_axis", "fold_stores_the_source_row",
              "no_susceptibility"]),
            ("cylindrical_real_fused_magnetic_pair", cylindrical_real_magnetic_clauses,
             ["curl_half", "constitutive_half", "no_magnetic_source", "no_fold"]),
            ("cylindrical_fused_magnetic_pair", cylindrical_complex_magnetic_clauses,
             ["curl_half", "constitutive_half", "no_magnetic_source", "no_fold",
              "no_wall_on_r_or_phi"])):
        log(f"\n  --- {name} ---")
        surviving = [(label(r), clauses(r)) for r in record]
        cumulative: Dict[str, int] = {}
        for clause in order:
            surviving = [e for e in surviving if e[1][clause]]
            cumulative[clause] = len(surviving)
            log(f"    after {clause:<30s}: {len(surviving):3d}")
        out[name] = {"admitted": len(surviving), "cumulative": cumulative,
                     "rows": [n for n, _ in surviving]}
        # A funnel that empties on its FIRST clause says nothing about the rest.
        # The counterfactual is what makes the zero informative: how many rows the
        # product's OWN clauses would take if the inherited half admitted.
        own = [clause for clause in order if clause not in ("curl_half",
                                                            "constitutive_half",
                                                            "pair_half")]
        standalone = [label(r) for r in record
                      if all(clauses(r)[clause] for clause in own)]
        out[name]["rows_clearing_only_this_products_own_clauses"] = len(standalone)
        log(f"    [counterfactual] rows clearing this product's OWN clauses, with "
            f"the inherited half set aside: {len(standalone)}")
    ade = [label(r) for r in record if all(ade_state_clauses(r).values())]
    registered = [label(r) for r in record if r["configuration"]["n_polarizations"] > 0]
    log(f"\n  --- fused_ade_state (update_P, intra-slot; NOT a (curl, constitutive) cell) ---")
    log(f"    rows with a susceptibility registered : {len(registered)}")
    log(f"    rows where EVERY state admits          : {len(ade)}")
    out["fused_ade_state"] = {"rows_with_a_state": len(registered),
                              "rows_every_state_admits": len(ade), "rows": ade}
    for name in ("fused_pair_B", "fused_pair_D", "dispersive_fused_pair_D"):
        served = [label(r) for r in record if seam_covered(r, name)]
        log(f"  SHIPPED SEAM PREDICATE {name:24s}: {len(served)} / {len(record)}")
        out[name] = {"admitted": len(served), "rows": served}
    return out


# ---------------------------------------------------------------------------
# What cannot be fused at all
# ---------------------------------------------------------------------------

UNFUSABLE: Tuple[Dict[str, str], ...] = (
    # THE FIRST TWO ROWS USED TO SAY "CANNOT BE BUILT", AND THE CODE THEY CITE SAYS
    # THE OPPOSITE. coverage.py's clause 3 (:462-487) reads "a source from this
    # pair's field family is CARRIED, not refused": the product declares
    # CARRIES_DEPOSIT_REPAIR, _install_fused_pair brackets the launch with
    # deposit_repair.LeadingRepairPlan / TrailingRepairPlan, and the deposit points
    # are recomputed from the injected field afterwards. The released composition
    # runs 38 such pairs (27 on the D seam, 11 on B), so a row calling the shape
    # unbuildable was describing a seam that had already shipped. What is still
    # refused is only what the repair cannot reconstruct, and that is what these
    # rows now say.
    {"pair": "step_B -> update_H on a row whose MAGNETIC source deposit the repair "
             "cannot reconstruct",
     "reason": "the driver injects every magnetic source BETWEEN the two "
               "(source.inject), which is real work inside the seam. A product "
               "declaring CARRIES_DEPOSIT_REPAIR carries it: the deposit is saved "
               "before the launch and recomputed after the driver's inject, "
               "symmetry fill and wall clear. What remains refused is what that "
               "repair cannot invert exactly — a source that does not publish the "
               "index it writes, or an absorber whose recurrence the repair does "
               "not undo. A product WITHOUT the declaration still refuses every "
               "in-seam deposit outright",
     "where": "driver.py:3292-3293",
     "refused_by": "deposit_repair.seam_source_reasons, per run, from "
                   "coverage.py:462-487 (fused_pair) / "
                   "folded_fused_magnetic_pair.py:584-590 / "
                   "complex_fused_magnetic_pair.py:625-631",
     "kind": "BUILT AND DISPATCHED with the deposit repair; refused only where the "
             "repair cannot reconstruct the deposit"},
    {"pair": "step_D -> update_E on a row whose ELECTRIC source deposit the repair "
             "cannot reconstruct",
     "reason": "the same seam on the electric side, carried the same way. On a "
               "conductive row the injection goes through "
               "_inject_electric_through_conductivity, a second scaling pass "
               "inside the seam, which the trailing repair also runs after — it "
               "reads the field once the driver has finished with it",
     "where": "driver.py:3303-3308",
     "refused_by": "deposit_repair.seam_source_reasons, per run, from "
                   "coverage.py:462-487 (fused_pair) / "
                   "dispersive_fused_pair.py (inherited)",
     "kind": "BUILT AND DISPATCHED with the deposit repair; refused only where the "
             "repair cannot reconstruct the deposit"},
    {"pair": "update_H -> step_D (crossing from the magnetic to the electric half)",
     "reason": "step_D's curl reads H at NEIGHBOUR offsets, so every cell's "
               "update_H must have landed everywhere before any step_D lane "
               "reads it — a grid-wide barrier no single launch provides. The "
               "electric integrated-source withdraw also sits between them",
     "where": "driver.py:3297-3301",
     "refused_by": "structural; no predicate exists because no product does",
     "kind": "CANNOT BE BUILT as one launch"},
    {"pair": "step_B -> step_D (the two curls)",
     "reason": "same barrier, in the other direction: step_D reads the H that "
               "update_H has not yet produced",
     "where": "driver.py:3290-3301",
     "refused_by": "structural",
     "kind": "CANNOT BE BUILT as one launch"},
    {"pair": "folded PML -> folded, on a folded PERIODIC axis",
     "reason": "fill_folded_far_ghosts_B runs inside the seam and images the top "
               "stored slot from a RUNTIME reflect row — a value the launch would "
               "have to read after writing it elsewhere in the same grid",
     "where": "driver.py:3296",
     "refused_by": "folded_fused_magnetic_pair.py:602-608",
     "kind": "REFUSED BY NAME — a runtime cross-lane read, not merely unbuilt"},
    {"pair": "complex PML -> complex, on a row with a symmetry",
     "reason": "fill_symmetry_bc_B and fill_folded_far_ghosts_B stop being no-ops "
               "the moment grid.has_symmetry() is true, and this weld carries "
               "neither pass",
     "where": "driver.py:3294, :3296",
     "refused_by": "complex_fused_magnetic_pair.py:642-651",
     "kind": "NOT BUILT — the folded product is the one that carries a fold"},
    {"pair": "any (curl, constitutive) cell where one half is on the array path",
     "reason": "there is no pair to fuse: fusion cannot exceed the UNFUSED arm "
               "coverage, so a slot no arm admits caps the seam at zero",
     "where": "launch.py:1983-2010 (_select_slot: >1 admitter or 0 leaves the "
              "slot unselected, which is the array path)",
     "refused_by": "launch.py:2084 _pair_may_absorb refuses a slot the arm table "
                   "did not give to the fused product's own arm",
     "kind": "CANNOT BE SERVED until the underlying arm covers it"},
    {"pair": "update_E -> update_P beyond ONE susceptibility with one pole",
     "reason": "multi-pole ADE fusion has no measured argument: the sub-step-share "
               "probe left update_P at 51.9% at K=6 with a 9.5-28.3pp within-run "
               "spread, so the precondition is bracketed, not cleared",
     "where": "fused_dispersive_chain.py:84 (CHAIN_MAX_POLES = 1), :380-386",
     "refused_by": "fused_dispersive_chain.py:380-386, :418-429",
     "kind": "REFUSED BY NAME — scope, not a barrier"},
    {"pair": "one kernel per timestep, on any backend",
     "reason": "the grid-wide barrier between the magnetic and electric halves, "
               "not the source seams: step_D reads H at neighbour offsets, so no "
               "launch can span both curls. Stated against the barrier because the "
               "source seams are no longer the argument — the deposit repair "
               "carries them, and a repair CANNOT carry this one: it restores a "
               "deposit at known points, where this needs every lane's write "
               "visible to every other lane mid-launch",
     "where": "driver.py:3297-3301",
     "refused_by": "structural",
     "kind": "CANNOT BE BUILT"},
)


def caveats(record: List[dict]) -> Tuple[str, ...]:
    """The board's caveats, with every corpus count read off the record it prices.

    A tuple of literals (``CAVEATS``) until 2026-09-03; the counts it spelled — the
    row total, the rows on an inactive absorber, the rows whose ``plan_step.selected``
    is empty — were the 186-row census's, and would have been emitted unchanged into
    the board cut over a 194-row one.
    """
    n = len(record)
    examples = sum(1 for r in record if r.get("leg") == "examples")
    tests = n - examples
    inactive = sum(1 for r in record if not r["configuration"].get("pml_active"))
    selected = [((r.get("plan_step") or {}).get("selected") or {}) for r in record]
    unselected = sum(1 for s in selected if not s)
    selected_arms = sorted({str(v) for s in selected for v in s.values()})
    return (
        "IS AN UPPER BOUND. Every clause below can only REMOVE seam-instances.",
        # ADDED 2026-08-28 with `_deposit_is_repairable`, and added because the ceiling
        # now rests on that clause: an in-seam source stopped being a structural bar and
        # became a question the shipped repair answers. Two of its refusals need an
        # engine object, so every instance this board credits ACROSS a live injection is
        # an upper bound in exactly the sense the line above states.
        "THE CARRIED DEPOSIT: each in-seam source publishes the index it writes — "
        "`deposit_repair._deposit_index` reads `_point_ix/_point_iy/_point_iz` and "
        "`seam_source_reasons` refuses a source that returns None. All four source "
        "classes DECLARE those fields but each sets them to None when the row's indices "
        "are empty on this chunk, which needs a built source to observe; the census "
        "records `n_sources` and `source_field_types` and stops there.",
        "THE CARRIED DEPOSIT: that `f_w_<component>` is allocated for all three of the "
        "seam's constitutive targets — the third refusal in `deposit_repair.repairable` "
        "— needs the engine's field arrays and is not in the census block.",
        "DERIVED, NOT RE-LIFTED. fused_pair_B/D and dispersive_fused_pair_D are the "
        "SHIPPED seam predicates called on the lifted object with the real source "
        "list, so those three are measured.",
        # 2026-08-30. FOUR MORE PRODUCTS MOVED FROM A TRANSCRIBED LADDER TO THEIR OWN
        # SHIPPED VERDICT, and with them go the three per-product caveats that used to
        # stand here (folded_fused_magnetic_pair's mirror_phase readability and
        # wall/fold disjointness; complex_fused_magnetic_pair's accessor readability and
        # int32 word bound; folded_fused_pair's whole ladder). None of them was ever a
        # property of the products — each was a limit of asking a CONFIGURATION BLOCK a
        # question only the predicate can answer. The census now records the answer.
        "MEASURED, NOT TRANSCRIBED (2026-08-30). folded_fused_pair, "
        "folded_fused_magnetic_pair, complex_fused_magnetic_pair and "
        "folded_complex_fused_magnetic_pair are scored from the verdict their SHIPPED "
        "predicate returned on the lifted object with the real source list, recorded in "
        "the census seam block — not from a clause ladder rebuilt here. Every clause "
        "those predicates carry is therefore asked, including the readability and "
        "word-bound clauses earlier cuts had to declare unevaluable. The hand-written "
        "ladders are RETAINED and evaluated beside them as a cross-check; "
        "`ladder_vs_shipped` reports every row where the two disagree.",
        "WHY THAT WAS A CORRECTION AND NOT A CONVENIENCE. `folded_electric_clauses` "
        "carried `no_folded_periodic_axis` — `all(metallic[a] for a in folded)` — which "
        "`folded_fused_pair_coverage` RETIRED on 2026-08-21 when the far carry landed "
        "inline (folded_fused_pair.py:1046-1110) under released gate "
        "results/triton_folded_far_carry_d_2026-08-21/run_farcarryD5/. The magnetic "
        "twin's mirror was updated when its own carry landed; the electric one was not, "
        "so this board refused rows the product admits for nine days. A board holding a "
        "second copy of a predicate is a board that can disagree with it, and this is "
        "what that costs.",
        "fused_dispersive_chain: the pole-identity clauses (:418-438 — that the "
        "subtracted pole volume IS the state's own P, and that drives==(count==1) "
        "per component) read live arrays and are not in the block. Its PML clause "
        "reads fields._pml_active (fields.py:538, set at :721) and is evaluated here "
        "from configuration.pml_active (pml is not None and pml.is_active), which is "
        "correlated but not the same attribute.",
        "folded_complex_fused_magnetic_pair: the FAR CARRY's reflect-row clauses "
        "(the row must lie in [0, stored-1), must not be 0, and stored-1 must not be "
        "the near fill's source plane) read stepping._far_reflect_rows on a live Grid "
        "and are not evaluable from the census block, which carries `shape` and not "
        "`shape_full`. The same measurement the D-side carry round made applies: over "
        "every folded PERIODIC extent Grid will build, stored == ceil(n_full/2) + 2 and "
        "reflect == floor(n_full/2), zero violations. The PARITY WORD readability "
        "clause (that mirror_parity_coefficients returns a nonzero pair on every folded "
        "axis) likewise reads the grid; folded_axis_kinds already refuses a folded axis "
        "whose phase is not +-1, which is the same question one rung up.",
        "bfast_fused_magnetic_pair: the grid.is_invariant READABILITY clause (the six "
        "k1/k2 words are gated on it, bfast_curl_coefficients / grid.py:1165-1183) is "
        "not evaluable from the census block, which records shapes rather than the "
        "grid's declared dimensionality accessors.",
        "nonlinear_fused_magnetic_pair, beta_fused_magnetic_pair and "
        "bfast_fused_magnetic_pair: the "
        "wall-readability clause each inherits from coverage.fused_pair_coverage "
        "(does the grid EXPOSE has_metallic / is_metallic / is_mirrored) is not "
        "evaluable — the configuration records the ANSWERS, not whether the "
        "accessors exist. Every engine-built Grid exposes all three.",
        "layout clauses (float32, C-contiguity, grid.shape) beyond what each half's "
        "own predicate already checks are not evaluable from the block.",
        "THE ARM TABLE HAS GROWN SINCE THE CENSUS. plan_step today consults 18 curl "
        "arms and 21 update_E arms; the record has 15 and 16. The eight added are "
        "conductive no-PML, complex conductive no-PML curl, complex no-PML curl, "
        "complex folded off-diagonal, folded off-diagonal dispersive, complex no-PML "
        "off-diagonal, complex no-PML stored E, no-PML stored E. Six are gated on an "
        f"INACTIVE absorber ({inactive} of {n} rows) and two on a fold plus an off-diagonal "
        "row. A new arm can move a cell in two directions — filling a slot nothing "
        "admitted, or creating an ambiguity that EMPTIES one (launch.py:1983-1985) — "
        "so the per-cell counts on those rows are the 2026-08-16 arm table's, not "
        "today's. No fused product's cell is among the added arms, so the served "
        "count is unaffected in the first direction; it could only fall in the second.",
        "BUILDERS WERE NOT MEASURABLE. The census host has no Triton, so every "
        f"builder raised and plan_step.selected is empty on {unselected} of {n} rows "
        f"(the remaining {n - unselected} carry only {', '.join(selected_arms)!r}, the "
        "null arm). The arm "
        "reported per slot is the arm TABLE's single admitter (launch.py:1980-1986), "
        "which is what _pair_may_absorb would see if the builder succeeded. A builder "
        "that refuses on the device would lower (c), never raise it.",
        "NUMPY HOST. Every verdict is read off covered_modulo_backend, which "
        "discounts the one 'array module is numpy, not cupy' clause. On a CuPy host "
        "that clause passes; on this record it is a fact about the harness.",
        f"THE CORPUS IS THE ENGINE-ACCEPTED {n} ({examples} example scripts + {tests} "
        "python-test cases; 186 = 52 + 134 until 2026-09-03), not all of MEEP. A row "
        "the lift "
        "refuses never reaches a predicate and is outside every number here.",
        "A SERVED SEAM-INSTANCE LICENSES A PREDICATE VERDICT, NOT A RUN. The two are "
        "now reported side by side rather than one being asserted: SERVED IN DISPATCH "
        "is measured per instance against fastpath.RELEASED_FUSED_ARMS and "
        "FUSED_RELEASE_ENVELOPE (aggregate.served_in_dispatch_measurement), and it is "
        "itself an UPPER BOUND - the composer's specialized-family guard, "
        "_pair_may_absorb and the warm pass all sit below it and none was measurable "
        "on a census host with no Triton. The enable condition every instance it "
        f"counts is under: {_reach.dispatch_condition()} (read off "
        "fastpath.DISPATCH_BY_DEFAULT on this cut).",
    )


#: Modules under ``triton_kernels/`` whose name carries ``fused`` or ``chain`` and
#: which deliberately occupy NO (curl, constitutive) seam cell. Each needs a reason,
#: because the floor below treats every other such module as a product this matrix
#: must ask about.
NOT_A_SEAM_PRODUCT: Dict[str, str] = {
    "fused_ade_state.py":
        "fuses one susceptibility's per-component update_P launches into one launch; "
        "it does not cross a sub-step boundary and occupies no (curl, constitutive) "
        "cell. Measured separately, in funnels().",
}


#: FUSED MODULES THE TREE SHIPS AND THIS CUT DOES NOT PRICE, named so the
#: completeness floor still bites for anything else. All three landed on this tree
#: on 2026-08-20 while this cut was being built, against the 2026-08-16 census this
#: walk scores; pricing another family's clause ladder from its module without its
#: own admission census is how a wrong number gets published with a floor's
#: blessing. What it costs the headline is stated in the run, not buried: whatever
#: they serve is counted here as a GAP, so the served total is a LOWER BOUND. It
#: does not touch this cut's DELTA, which is measured against
#: ``../fusion_matrix_triton_2026-08-20_closed/``.
#: EMPTY IN THIS CUT, AND THE EMPTINESS IS THE POINT. The 2026-08-20 farcarry cut
#: carried three entries here (beta / bfast / nonlinear), landed while it was being
#: built and scored as GAPS, which made its served total a LOWER BOUND. The
#: 2026-08-21 gapcells cut priced all three. This cut merges both lineages, so the
#: hatch has nothing left in it; `assert_every_built_product_is_asked_about` now
#: FAILS on a non-empty dict rather than printing a lower-bound warning.
NOT_PRICED_HERE: Dict[str, str] = {}


#: CERTIFIED BUT NOT PRICED HERE — AND THE BOUND IS STATED RATHER THAN GUESSED.
#: This is NOT `NOT_A_SEAM_PRODUCT` (occupies no cell) and NOT the old
#: `NOT_PRICED_HERE` (a stale bookkeeping lag over products this cut does price).
#: The module below is a real E->P product, on a real cell, with a real released
#: device gate. What this cut does not have is an ADMISSION CENSUS for it.
#:
#: THE HISTORY, BECAUSE IT IS THE EVIDENCE. `triton_kernels/fused_ade_chain.py`
#: landed UNTRACKED at 2026-08-21T01:19:38, three minutes before this cut first
#: ran, from a CONCURRENT round. At that moment it had no fingerprints entry and no
#: results directory, so it was exempted as UNGATED and its credit withheld. At
#: 01:55:50 — twelve minutes after the cut — that round finished: this script's own
#: self-retiring guard then RAISED, naming
#: `fingerprints keys ['triton_fused_ade_chain_device_gate']` and
#: `results directories ['triton_fused_ade_chain_2026-08-21']`. The exemption could
#: not outlive its reason, which is what it was written to guarantee.
#:
#: THE GATE IS REAL, VERIFIED HERE RATHER THAN ASSUMED: NVIDIA RTX A6000, Triton
#: 3.1.0, 60 steps; `gate_keep.json` and `gate_flush.json` both PASS with
#: `canonical_verdict.released = true`; `gate_planted.json` FLIPS to FAIL against
#: `ade_association_flattened_everywhere`; 31 mutations, 9 product legs, 4 separate
#: controls; the `triton_fused_ade_chain_device_gate` fingerprint records 9 sources
#: with ZERO drift against this tree.
#:
#: WHY IT IS STILL NOT PRICED. Its predicate reads LIVE OBJECTS —
#: `fields.polarizations`, `state.driven()`, `state.P`, `poles_per_component`
#: (fused_ade_chain.py:838-897). None of those is in the census configuration block
#: this walk scores, exactly like the clauses already listed for
#: `fused_dispersive_chain` in CLAUSES_NOT_EVALUABLE_FROM_THE_CENSUS_BLOCK. Pricing
#: a family's clause ladder from its module WITHOUT its own admission census is how
#: a wrong number gets published with a floor's blessing.
#:
#: THE BOUND, DERIVED FROM THE CELL STRUCTURE, WHICH *IS* IN THE BLOCK.
#: `ARMS = ('no_pml', 'dispersive', 'folded')` and complex64 is refused BY NAME.
#: Triton's four E->P cells are: (dispersive x ADE) 7, (no-PML dispersive x no-PML
#: ADE) 3, (complex no-PML x complex ADE) 4 — refused on storage — and one instance
#: with no E arm at all. So this product can serve AT MOST 7 + 3 = 10 of the 15,
#: and the module's own docstring says ten. Every unevaluable clause can only
#: REMOVE instances (this matrix is an upper bound by construction), so the true
#: figure lies in [0, 10] and is not measured here.
#:
#: CONSEQUENCE, STATED NOT BURIED: Triton E->P is reported 0 of 15 and the Triton
#: total of 115 is a FLOOR, short by at most 10. Closing it needs one admission
#: census on a Triton host — not another matrix cut on this laptop.
#: FUSED MODULES THAT LANDED ON THIS TREE FROM A SEPARATE CHANGE WHILE THIS CUT
#: WAS BEING MADE, and which this cut does not price. NOT ``GATED_NOT_PRICED_HERE``:
#: that category ASSERTS a released gate and re-earns the assertion on every run.
#: This one asserts nothing about the module's certification, because this cut has
#: not looked and may not speak for another round's work.
#:
#: THE FLOOR STILL BITES. The entry must name a module the tree really ships (an
#: exception that outlives its exception silences the floor), its instances are
#: counted as GAPS, and the served total is therefore a LOWER BOUND by whatever it
#: serves. What this cut refuses to do is publish a number for a family whose
#: admission ladder it has not transcribed.
#: BOTH ENTRIES WERE DISCHARGED ON 2026-08-30 AND THE DICTS ARE EMPTY. Neither was
#: discharged by a decision: `complex_fused_ade_chain.py` abstained because this cut
#: "has not transcribed its admission ladder", and `fused_ade_chain.py` because "its
#: predicate reads live polarization objects the census block does not carry". Both
#: sentences were true, and both are answered by the same change — the census now
#: RECORDS what each shipped predicate returned on the real lifted objects, so
#: neither ladder has to be transcribed and no configuration block has to carry a
#: live polarization object. See E_TO_P_VERDICT_KEYS.
#:
#: THE STRUCTURES STAY, EMPTY, ON PURPOSE. Their floors are what make the abstention
#: cost something: an entry names a module the tree really ships and its instances
#: are counted as GAPS. Deleting the mechanism with the last entry would leave the
#: next round's abstention free.
LANDED_CONCURRENTLY_NOT_PRICED_HERE: Dict[str, str] = {}


GATED_NOT_PRICED_HERE: Dict[str, str] = {}


#: WHERE EACH PRODUCT'S RELEASE LIVES. This matrix credits 102 seam-instances on
#: the strength of device gates that ran against particular bytes; if a module has
#: moved since, the credit rests on nothing. Eight of the nine products' gates
#: recorded their digests in ``triton_kernels/fingerprints.json`` under the
#: current-generation ``triton_*_device_gate`` keys. (Three OLDER keys —
#: ``dispersive_fused_pair_gate``, ``fused_ade_state_gate``, ``fused_electric_gate``
#: — are superseded records under a different, mixed path convention and are not
#: read here.)
#:
#: THE NINTH IS DIFFERENT AND THE DIFFERENCE IS DELIBERATE.
#: ``cylindrical_fused_magnetic_pair.py`` carries NO fingerprints.json entry, and
#: says why in its own docstring: "this module's licensed bytes are a function of a
#: PROBE-BOUND arm, so a checked-in hash would record a choice rather than a
#: measurement". Its gate DID record a module digest and the tree no longer matches
#: it. MEASURED RATHER THAN ASSUMED: the gated bytes were fetched from the staging
#: copy the gate ran out of (the GPU host,
#: <staging-root>/cylfuse/source/, sha 4ee180e4) and diffed
#: against the tree (sha 9265e122). The whole difference is ONE hunk inside the
#: module docstring, lines 190-215 — the DEVICE STATUS block the release round wrote
#: — and the docstring-stripped ASTs are IDENTICAL. So the 16 seam-instances
#: credited to it rest on bytes whose executable content is provably the certified
#: ones. That equivalence is re-checked below on every run, against a copy of the
#: gated bytes kept beside this script, so the claim does not decay into a remark.
FINGERPRINT_BOUND: Dict[str, str] = {
    "fused_pair_B": "triton_fused_electric_device_gate",
    "fused_pair_D": "triton_fused_electric_device_gate",
    "dispersive_fused_pair": "triton_dispersive_fused_pair_device_gate",
    "folded_fused_magnetic_pair": "triton_folded_fused_magnetic_pair_device_gate",
    "complex_fused_magnetic_pair": "triton_complex_fused_magnetic_pair_device_gate",
    "fused_dispersive_chain": "triton_fused_dispersive_chain_device_gate",
    "folded_fused_pair": "triton_folded_fused_pair_device_gate",
    "cylindrical_real_fused_magnetic_pair":
        "triton_cylindrical_real_fused_magnetic_pair_device_gate",
    # THE TWO E->P CHAINS, priced for the first time on 2026-08-30. Both were BUILT
    # and RELEASED before this cut and reported at zero for want of an admission
    # census; the census now records what their shipped predicates returned, so the
    # credit is real and owes a binding like every other. Neither gate is new — this
    # row is the board finally reading the record that already existed.
    "fused_ade_chain": "triton_fused_ade_chain_device_gate",
    "complex_fused_ade_chain": "triton_complex_fused_ade_chain_device_gate",
    # THE COMPLEX D->E PAIR, welded 2026-08-30 from its own first device run. The
    # entry was written from the proposal packet propose_triton_welds.py produced,
    # whose digests are the RUN's own imported_source_sha256 — so this row binds the
    # credit to bytes a device executed rather than to bytes a board hashed.
    "complex_fused_electric_pair":
        "triton_complex_fused_electric_pair_device_gate",
}

#: The one product bound by AST equivalence instead, with the gated copy it is
#: compared against and the gate whose verdict it carries.
AST_BOUND: Dict[str, Tuple[str, str]] = {
    # BOTH paths are relative to parity/meep_gpu now. The copy used to be relative
    # to the board's own directory, which is precisely the coupling the promotion
    # removes: the gated bytes belong to the tranche that fetched them, not to
    # whichever directory a later cut happens to write into.
    #
    # RE-POINTED 2026-08-28 TO A FRESH DEVICE RUN, and the equivalence it re-checks
    # is now an IDENTITY rather than a tolerance. The 08-20 pairing bound the tree
    # to bytes that differed from it inside the module docstring; that AST
    # equivalence held on 08-21 and had STOPPED holding by 08-28, which is what put
    # this family in ``CREDIT_WITHHELD_STALE_BINDING`` and its 16 seam-instances in
    # the withheld bucket. Re-running the gate is the whole repair: the probe was
    # run against THIS tree on the GPU host GPU 3 (physically empty, 6 MiB, no compute
    # process) on 2026-08-28, verdict PASS over 76 rows, and it recorded
    # ``cylindrical_fused_magnetic_pair.py`` at 831b2467 -- the sha the tree ships.
    # The copy below is that file, so the digest branch and the docstring-stripped
    # AST branch both compare the module against its own certified bytes. Nothing
    # was edited to make them agree; the gate was re-run so that they do.
    # EMPTY SINCE 2026-09-04, AND THE MODE IS KEPT DELIBERATELY. Its one member,
    # ``cylindrical_fused_magnetic_pair``, moved to ``GATE_BOUND``: the m = 0 re-gate
    # recorded the module at the digest the tree ships, so its credit now rests on
    # byte IDENTITY and the AST equivalence would be the weaker of two available
    # claims. The mode itself is not retired, because the condition that created it
    # recurs by construction — this module carries no fingerprints.json entry by its
    # own design, and a release round that stamps a DEVICE STATUS block into its
    # docstring after the gate has run puts the tree one docstring away from its
    # gated bytes again. A family may only be entered here with a copy of the gated
    # bytes to compare against.
}

#: THE 2026-08-21 GAP-CELL PRODUCTS, bound to their own gate ARTIFACT rather than
#: to a ``fingerprints.json`` entry. Nothing was written to ``fingerprints.json``
#: this round — that file is a release ledger for the shipped kernel set, and a
#: matrix cut may not add to it — so the binding reads the gate's own
#: ``source_sha256`` map, which records every file the run bound itself to, and
#: re-hashes each against the tree.
#:
#: A FALSE ``released`` IS THE TRAP THIS BRANCH IS WRITTEN AGAINST. A stress or
#: host leg that writes ``released`` gets it copied into a canonical verdict, and
#: the package's weld tooling then reads a laptop run as a passed device gate. So
#: the branch below requires ALL THREE of ``device_status == 'RUN'``,
#: ``release.released is True`` and ``passed is True``, and a
#: ``--no-device`` artifact fails every one of them by construction.
#: BUILDABLE GAP CELLS AND WHAT EACH ONE NEEDS -- the hand-off, so a cell nobody
#: built is distinguishable from a cell somebody measured and declined.
#:
#: A cell appears here only when BOTH its arms admit on a live object and the board
#: scores the instance REACHABLE. The text is the measured recipe, not a wish: every
#: line number in it was read, and every claim about a delta was taken as an exact
#: statement-list difference between two shipped bodies rather than by eye.
#: EMPTY AS OF 2026-08-31 (this cut): the two folded-beta recipes it held were
#: BUILT, to their own measured letter — ``folded_beta_fused_electric_pair.py``
#: and ``folded_beta_fused_magnetic_pair.py``, each the named base kernel plus the
#: exact three-line insert, each with the fold and seam clauses RE-SPELLED and an
#: equivalence leg in its gate asserting the weld admits only where both shipped
#: arms admit. The entries left this table in the same change that added the two
#: products to ``PRODUCTS``, so a recipe can never describe a cell some product
#: already owns.
BUILDABLE_GAP_RECIPES: Dict[Tuple[str, str, str], str] = {}


#: RE-POINTED AGAIN 2026-09-02 (SEVENTH ROUND) TO
#: ``results/triton_regate_2026-09-02_foldedrelease``, for the NINETEEN rows below
#: (seventeen distinct artifacts) that the sixth round had pointed at
#: ``results/triton_regate_2026-09-02_residue``.
#:
#: WHY, AND WHY IT IS A RE-POINT RATHER THAN A RE-RUN. The dispatch Phase 1 change
#: (``fill_B``/``fill_D`` became driver consults) moved ``meep_gpu/driver.py``,
#: ``meep_gpu/fastpath.py``, ``meep_gpu/triton_kernels/launch.py`` and
#: ``meep_gpu/triton_kernels/__init__.py`` AFTER the residue campaign ran, so every
#: residue artifact credited bytes the tree no longer ships and
#: ``assert_every_credit_is_bound_to_released_bytes`` refused the cut on 21 products.
#: The drift is REAL and not the comment-only kind: measured with
#: ``device_identity.weld_survives_edit``'s own rule, ``driver.py``'s code digest
#: moved 7daf71630b -> 6ec5f47dbe and ``launch.py``'s 89a4956cd2 -> 13c9549f92.
#:
#: The re-run those 19 rows would have needed had ALREADY HAPPENED: the 37-gate
#: ``drive_triton_weld_gates.py`` campaign into ``_foldedrelease`` gated every one of
#: these products against the FINAL bytes. Each row below was re-pointed only after
#: its new artifact was checked to declare the SAME ``product`` module as the residue
#: artifact it replaces and to read ``released`` — a name match alone would be the
#: "binds a record to a run that measured something else" mistake this file names
#: elsewhere. 55 of the campaign's 59 gate artifacts record their imported source
#: hashes and 0 of them are stale against this checkout.
#:
#: THE TWO ROWS NOT RE-POINTED are the off-diagonal scratch-output welds
#: (``offdiag_fused_electric_pair``, ``folded_offdiag_fused_electric_pair``). The
#: weld driver genuinely does not cut them — they have no gate in the campaign — so
#: they were RE-RUN on device instead, and their entries are updated separately.
#:
#: The sixth round's record, which the above supersedes for those 19 rows:
#: RE-POINTED WHOLESALE 2026-09-02 TO ``results/triton_regate_2026-09-02_residue``,
#: and this is the sixth round of the same repair rather than a new kind of one.
#: ``meep_gpu/driver.py`` — a file under EVERY Triton weld's ``source_sha256`` map —
#: moved twice: first when ``_inject_electric_through_conductivity`` replaced its
#: three whole-volume difference passes with a per-cell replay at the deposit
#: indices the sources publish, and then when a docstring recorded the CuPy
#: subnormal measurement that change also produced. MEASURED before this round
#: began: nineteen of the table's rows below (fifteen distinct artifacts) recorded
#: a ``driver.py`` the tree no longer ships, so ``assert_every_credit_is_bound_to_
#: released_bytes`` refused the cut — correctly, and on the FIRST product it
#: reached rather than on all of them.
#:
#: THE SECOND EDIT WAS COMMENT-ONLY AND IT STILL COUNTS. That is the position this
#: file has held since 2026-08-16 (``rebind_triton_welds.py``'s own docstring names
#: the round where a comment-only edit kept twenty-one welds green while they
#: described bytes that no longer shipped) and re-running is cheap: every gate below
#: took between 40 s and 5 min on one device.
#:
#: THE CAMPAIGN. the GPU host, GPU 1, one process per gate, sequential, ``keep`` policy
#: through a private ``ftz_stripped`` CuPy cache per gate. GPU 1 rather than the
#: round's nominal GPU 3 because 3 was NOT free — another process held memory and
#: compute on it through the whole window, and 0/1 were the physically
#: empty devices (4 and 5 are forbidden). These artifacts carry no timing claim, so a
#: shared device could not have changed a verdict; the placement is recorded because
#: the lane's precondition was "verify free" and it did not hold.
#:
#: FOUR ROWS ARE NEW PRODUCTS rather than re-points — the two complex-beta welds and
#: the two folded complex-beta welds, whose FIRST device runs are in this campaign.
#: The complex-beta pair had an earlier passing run (2026-09-01, GPU 3) that this
#: table deliberately does NOT cite: it bound the pre-docstring ``driver.py`` and the
#: binding check would have refused it.
#:
#: REPOINTED 2026-09-15 FROM ``triton_fleet_2026-09-13_batch_{A,B,C}`` TO
#: ``triton_fleet_2026-09-14_target_{A,B,C}``. The 09-13 artifacts bound
#: ``triton_kernels/launch.py`` at ``de606ebe`` and the tree ships ``b7701503``; the
#: 09-14 fleet re-ran every one of those gates against ``b7701503`` (63 of its 64
#: gate records carry that digest). The first cut of
#: ``fusion_matrix_triton_2026-09-15_offdiag`` refused on exactly that: 37 products
#: drifted, 18 of them on ``launch.py`` alone (or with their own test file). The
#: hd_pair and stencil-weld rows below have no 09-14 run and keep their 09-13 paths
#: until they are re-gated.
#:
#: RE-GATED AND REPOINTED THE SAME DAY: ``triton_regate_offdiag.sh`` ran the ten
#: ``gate_triton_*_hd_pair.py`` scripts and both stencil-weld families against the live
#: bytes, so every row that still named ``_2026-09-13_batch`` now names
#: ``_2026-09-15_offdiag``; all twelve artifacts carry ``release.released`` true. TWO OF
#: THEM NAME ``keep_probe/`` RATHER THAN ``keep/``: the folded-complex and beta-complex
#: hd gates refused in three seconds on their first run -- "no complex-multiply expansion
#: probe artifact is available for this backend" -- because the run was launched without
#: ``MEEP_GPU_COMPLEX_EXPANSION_PROBE``, which their predicates read and which their
#: 2026-09-13 artifacts do not record. The follow-up run supplied the unified probe and
#: wrote a FRESH directory rather than reusing ``keep/``: these gates APPEND to
#: ``gate.rows.jsonl``, so the refused run's rows are still in ``keep/`` beside it and
#: would have been mixed into the released run's record.
GATE_BOUND: Dict[str, str] = {
    # THE E->P CHAIN, released on its first device run (2026-09-11, the GPU host GPU 7,
    # cc 8.6): PASS, release.released true, no reasons. Bound to the gate ARTIFACT
    # rather than to its fingerprints.json entry because the gate's curated set is
    # the wider one -- it pins both E-half modules, the ADE kernel and the gate
    # script, and the board re-hashes every one of them on each cut.
    "folded_offdiag_fused_ade_chain":
        "results/triton_fleet_2026-09-25_roundb/folded_offdiag_fused_ade_chain/gate.json",
    # THE THREE 08-21 GAP-CELL FAMILIES, RE-POINTED 2026-08-28 TO A FRESH DEVICE
    # CAMPAIGN. Their 08-21 artifacts each drifted on FOUR of eleven files -- the
    # three the 08-27 routing round rewrote under every Triton weld (``driver.py``,
    # ``triton_kernels/coverage.py``, ``triton_kernels/launch.py``) plus, for two of
    # them, their OWN kernel -- which is what withheld their credit on the 08-28
    # board and made SERVED a floor short by 20.
    #
    # THE CREDIT IS RE-EARNED BY RE-RUNNING, NOT BY EDITING A DIGEST. All three
    # probes were run against THIS tree on the GPU host GPU 3 (verified physically
    # empty: 6 MiB, no compute process), one process each, sequential, under the
    # ``keep`` policy through a private ``ftz_stripped`` CuPy cache -- so every
    # artifact stamps ``subnormal_policy ieee_keep_ftz_stripped`` rather than the
    # half-applied ``not_installed`` a bare default would have earned. Each is
    # ``device_status RUN``, ``passed``, ``release.released``, no planted defect,
    # and 11 of 11 recorded files match the tree with zero drift.
    #
    # A FIRST PASS OF THIS CAMPAIGN FAILED ALL THREE AND IS KEPT, NOT HIDDEN
    # (``results/triton_gapcell_regate_2026-08-28/``). Every device leg, mutation
    # and refusal passed there too; what failed was the NO-DEVICE
    # ``corpus_admission`` leg, on one finding -- "the fusion matrix is absent:
    # .../results/fusion_matrix_triton_2026-08-20_closed/fusion_matrix.json" -- the
    # cross-check that requires the cell membership this leg derives from the census
    # to equal what the 08-20 board scored at the same cell. The staged tree carried
    # the census but not that board. Copying the board across and re-running is the
    # repair; the leg was not touched, and it is the reason the first campaign
    # directory is superseded rather than deleted.
    #
    # RE-POINTED 2026-08-30 TO A FRESH CAMPAIGN, for the same reason the 08-28 one
    # was cut: a shared file under every Triton weld moved. This round it is
    # `triton_kernels/launch.py`, whose two absorb-table comment blocks stated that
    # the fold boundary rested on `CARRIES_DEPOSIT_REPAIR` being False on both
    # folded families — a premise the deposit carry retired. None of these three
    # families changed at all; each is re-gated because its gate PINS that file, and
    # a weld that pins a file it no longer matches credits bytes nothing measured.
    # All three released with zero drift against the tree.
    # THE TWO SCRATCH-OUTPUT OFF-DIAGONAL WELDS, 2026-09-02, and this pair of
    # entries is what moves the STENCIL bucket for the first time since it was
    # opened on 2026-08-20.
    #
    # WHAT THE GATE MEASURED. the GPU host, RTX A6000, one family per verified-empty
    # GPU (0 and 1), `keep` policy through a private ftz_stripped CuPy cache and a
    # private Triton cache. Per COMPLETE driver step as uint32 over every stored
    # volume, against BOTH the CuPy array path AND the separately certified Triton
    # kernels the weld replaces; every case carries a non-vacuity floor (the words
    # the reference itself moved) and two independent launch counters.
    #
    # WHY IT SUPERSEDES RATHER THAN CONTRADICTS
    # `results/triton_fused_offdiag_electric_2026-08-20`. That artifact measured a
    # ONE-ORDINARY-LAUNCH weld of the SHIPPED IN-PLACE halves and found 42 of 60
    # subject cases divergent, schedule-dependently. Its own PROVENANCE names what
    # it measured — "the off-diagonal constitutive arm is a STENCIL over the curl
    # arm's own IN-PLACE output" — and records its halo-recompute closure as "a
    # DERIVATION from a parsed fact rather than a device measurement". These
    # products remove the in-place premise the derivation rests on: the curl half
    # writes launch-local scratch, the constitutive half re-derives every foreign
    # tap from pre-launch state, and the plan rotates the D/fu references after the
    # launch. Nothing written is ever read, so the S2 leg sweeps the same block
    # sizes on the same 4096-cell grid and the count stays at zero.
    #
    # RE-POINTED 2026-09-02 FROM `results/triton_offdiag_stencil_welds_2026-09-02`,
    # and re-pointed for a PROVENANCE reason rather than a numerical one. That
    # campaign's two artifacts carry no provenance block at all: the gate's save()
    # wrote its payload without calling `gate_provenance.stamp`, so nothing recorded
    # which bytes the releasing process actually imported — the one fact the stamp
    # exists to fix, and the gap `test_gate_provenance` refuses at a budget of zero.
    # Adding the stamp MOVES THE GATE SCRIPT, which that gate pins four times across
    # its own two artifacts (`source_sha256` and `imported_source_sha256` in each),
    # so the pins had to be re-cut rather than tolerated: a weld that pins a file it
    # no longer matches credits bytes nothing measured. The re-run is the SAME gate
    # on the SAME two families, `keep` policy, one family per verified-empty GPU.
    # THE NUMBERS IT HAD TO REPRODUCE, and did: S1 60/60 IDENTICAL against both the
    # array path and the certified singles, and S2 ZERO differing words at every
    # block size (64/128/256/512/1024) on both families — the same two numbers the
    # superseded 08-20 artifact's own instrument reports, unmoved by the edit, which
    # is the point: a provenance stamp must not be able to change a verdict.
    # Nothing was edited to make anything match; the pointer is written BEFORE the
    # run, so no artifact here certifies bytes that no longer exist.
    "offdiag_fused_electric_pair":
        "results/triton_offdiag_stencil_welds_2026-09-25_roundb/offdiag_fused_electric_pair/gate.json",
    "folded_offdiag_fused_electric_pair":
        "results/triton_offdiag_stencil_welds_2026-09-25_roundb/folded_offdiag_fused_electric_pair/gate.json",
    "bfast_fused_magnetic_pair":
        
        "results/triton_fleet_2026-09-25_roundb/bfast_fused_magnetic_pair/gate.json",
    "nonlinear_fused_magnetic_pair":
        
        "results/triton_fleet_2026-09-25_roundb_env_probe/nonlinear_fused_magnetic_pair/gate.json",
    "beta_fused_magnetic_pair":
        
        "results/triton_fleet_2026-09-25_roundb/beta_fused_magnetic_pair/gate.json",
    # ONE GATE, TWO PRICED CELLS. The arm label decides which cell an instance lands
    # in; the module, the kernel and the gate are the same object, so both entries
    # bind to the same artifact and both are checked against the tree.
    #
    # RE-POINTED 2026-08-28 FROM THE 08-21 ARTIFACT, and re-pointed because the
    # 08-21 one no longer describes the tree. The 2026-08-27 routing round rewrote
    # launch.py, coverage.py and driver.py under every Triton weld; measured here,
    # the 08-21 artifact drifts on six of its fourteen files. THE FIX IS NOT TO
    # TOLERATE THAT DRIFT -- it is that the family was RE-GATED afterwards and the
    # fresh artifact is released and matches the tree exactly (14 of 14 files,
    # device_status RUN, passed, release.released). This entry names the run that
    # certified the bytes that ship. Nothing was edited to make it match.
    #
    # RE-POINTED AGAIN 2026-08-30, same rule, this round's cause: the deposit carry
    # flipped CARRIES_DEPOSIT_REPAIR on the three sibling Triton fused families and
    # moved both complex_fused_magnetic_pair.py and launch.py, which this gate pins.
    # The family itself changed only its DEVICE_STATUS pointer, and that pointer was
    # edited BEFORE the run rather than after it -- a run whose artifact pointer is
    # written afterwards certifies bytes that no longer exist. Re-run on the GPU host:
    # 50 device legs, 456 complete steps, 450 fused launches, released.
    #
    # RE-POINTED 2026-09-01, and this round the FAMILY ITSELF changed:
    # CARRIES_DEPOSIT_REPAIR flipped False -> True on THIS module (2026-08-31),
    # discharging the plainrepair7 board's one clause-refusal on it
    # (tests:TestHoleyWvgBands.test_fields_at_kx, an in-seam magnetic
    # VolumeSource the repair can carry). The gate was EXTENDED in the same
    # change -- the magnetic-source refusal leg became two rows (no-index
    # refused, indexed ADMITTED), and it gained 6 CARRY legs (the folded
    # PERIODIC base and the OFFDIAGONAL arm label the discharged row carries,
    # x3 value classes) repairing a real in-seam magnetic deposit through the
    # SHIPPED LeadingRepairPlan/TrailingRepairPlan bracket, plus 2 NULL
    # CONTROLS with the bracket removed that both diverge -- and RE-RUN on
    # the GPU host GPU 3 against the flipped bytes: 58 device legs, released,
    # 14 of 14 recorded files matching this tree.
    # REPOINTED 2026-09-11 TO THE RE-GATE, and this row is a BOARD edit rather than a
    # device run because the run already happened. Editing the module's DEVICE_STATUS
    # pointer put the very gate that pins that module into drift -- the ordering trap
    # this family's own test states twice ("a run whose artifact pointer is edited
    # AFTER it finishes certifies bytes that no longer exist") -- so the probe was
    # re-cut into results/triton_regate_2026-09-11_dispatch/ on the GPU host under
    # ieee_keep_ftz_stripped, released, 49655 PTX instructions audited with 0 carrying
    # .ftz, and it re-hashes clean against this tree. That artifact is ALREADY what
    # both test_triton_folded_complex_fused_magnetic_pair.py's GATE_ARTIFACT and
    # triton_kernels/folded_complex_fused_magnetic_pair.py's DEVICE_STATUS name; the
    # board was the one place left pointing at the superseded realarms_c leg.
    # REPOINTED 2026-09-13 to the batch fleet (shard C), the run that re-measured the
    # probe after it learned to record the device identity (phase A).
    "folded_complex_fused_magnetic_pair":
        "results/triton_fleet_2026-09-25_roundb_env_noprobe/probe_triton_folded_complex_fused_magnetic_pair/gate.json",
    "folded_complex_fused_magnetic_pair_offdiag":
        # REPOINTED 2026-09-17 to this round's env-probe lane, and to the SAME
        # artifact its base row cites: the derivative shares the base family's
        # evidence, which is what the 2026-09-14 pair of values already did. The
        # base moved to the env-probe lane rather than the base fleet lane, because
        # this family's fleet run reads `released=False` without the licence, so a
        # stamp swap alone would have landed on the refused artifact.
        # REPOINTED 2026-09-21 to this round's env-probe lane, by hand rather than by
        # the repoint tool: that tool anchors on `"<family>": "<path>"` together, and
        # this row's comment sits between its key and its value, so the anchor does not
        # match and the row was left alone while its base row moved. The two must cite
        # ONE artifact -- the derivative has no evidence of its own -- and the 09-17
        # copy records a `deposit_repair.py` the tree no longer ships.
        # REPOINTED 2026-09-23 to the sparse round's env-probe lane, by hand for the
        # same reason (the literal is split across two lines, which the repoint tool's
        # single-string pattern does not see), to the SAME artifact its base row cites.
        # REPOINTED 2026-09-25 by hand, same reason, to the SAME artifact its base row
        # cites: this round ran the gate in the licence-exported no-`--probe` lane
        # (`_env_noprobe`), released; its argv and measured expansion (FMA_V1, five
        # patterns, keep) equal the 09-23 env-probe run's, and a second run in the
        # env-probe environment (`triton_fleet_2026-09-25_roundb_env_probe_r3`, released)
        # reads the same -- kept as corroboration, not bound.
        "results/triton_fleet_2026-09-25_roundb_env_noprobe/"
        "probe_triton_folded_complex_fused_magnetic_pair/gate.json",
    # THE TWO 2026-08-30 D->E CELLS, bound to their OWN first device runs rather
    # than to a ``fingerprints.json`` entry. That file is a release ledger for the
    # shipped kernel set and a matrix cut may not add to it, so the binding reads
    # each gate's ``source_sha256`` map — every file the run bound itself to — and
    # re-hashes each against the tree. Both artifacts were produced by running the
    # gates against THIS tree, after every edit in the round: a run whose pointer is
    # written afterwards certifies bytes that no longer exist.
    # The subdirectory is the DRIVER's gate name (drive_triton_weld_gates.py:539,
    # ``out_root / gate.name``), not the probe's file name — the campaign was run
    # through that driver so the residency guard, the ftz-stripped cache and the
    # policy install are the fleet's rather than this round's.
    # RE-POINTED 2026-09-04 TO THE m = 0 RE-GATE, and the re-gate is the rule rather
    # than a repair: the m = 0 complex cylindrical landing edited
    # cylindrical_complex.py and this module, so the 08-31 artifact's recorded
    # digests stopped describing the tree and this binding correctly refused the
    # board. The gates were RE-RUN on the GPU host GPU 7 (campaign.json, finished
    # 2026-09-04T07:48:48Z) rather than the digests re-pointed by hand: released,
    # device_status RUN, passed, 34 of 34 recorded files matching this tree with
    # zero drift, verified before this repoint.
    # REPOINTED 2026-09-13 to the batch fleet (shard A; the probe now records the
    # device identity and the policy keys the ledger reads).
    "cylindrical_fused_electric_pair":
        "results/triton_fleet_2026-09-25_roundb/cylindrical_fused_electric_pair/gate.json",
    # THE Dcyl MAGNETIC TWIN, MOVED HERE FROM ``AST_BOUND`` ON 2026-09-04 — a
    # STRICTLY STRONGER BINDING, not a relaxed one. Through 2026-09-03 this module
    # was bound by docstring-stripped AST equivalence against a copy of its gated
    # bytes, because its 08-28 gate recorded bytes that differed from the tree inside
    # the module docstring. The 2026-09-04 re-gate records this module at the digest
    # the tree SHIPS (34 of 34 recorded files match, zero drift), so the weaker
    # equivalence has nothing left to do: identity replaces it. Its artifact carries
    # the older verdict-prose shape, read by the second condition set in
    # ``assert_every_credit_is_bound_to_released_bytes``.
    "cylindrical_fused_magnetic_pair":
        "results/triton_fleet_2026-09-25_roundb/cylindrical_fused_magnetic_pair/gate.json",
    # REPOINTED 2026-09-17. This round's fleet artifact carries NO `release`
    # block -- its verdict is in `canonical_verdict` ({"released": true,
    # "read_from": "verdict(prose)"}), which is the shape composition and shape
    # gates write and the reason five fleet gates read `released=None` to a reader
    # that checks only `release.released`. The gate passed; only the key differs.
    # REPOINTED 2026-09-17: route A released this product's arm, and that batch edited
    # `probe_triton_complex_conductive_fused_pair.py` to stamp the device identity and the
    # policy block `seed_triton_welds.py` refuses to invent. The 2026-09-07 artifact
    # records the probe's OLD bytes, so the binding check refused it by name. This is the
    # run the ledger entry was seeded from. (Comment moved above the key 2026-09-19 so the
    # row is one anchorable key/value pair for repoint_triton_gate_bound.py.)
    "complex_conductive_fused_pair":
        "results/triton_fleet_2026-09-25_roundb/complex_conductive_fused_pair/gate.json",
    # THE 2026-08-31 FOLDED DISPERSIVE CELL, bound to its OWN first device run for
    # the reason the two above are: `fingerprints.json` is a release ledger for the
    # shipped kernel set and a matrix cut may not add to it, so the binding reads the
    # gate's `source_sha256` map — every file the run bound itself to — and re-hashes
    # each against the tree. The artifact was produced by running the gate against
    # THIS tree, after every edit in the round: a run whose pointer is written
    # afterwards certifies bytes that no longer exist. 17 of 17 files match with zero
    # drift.
    # RE-POINTED to the 2026-08-31 retry3 RE-RUN, same rule as the electric cell
    # below: the plain-branch round edited meep_gpu/deposit_repair.py AFTER the
    # original artifact was cut, so its recorded digests stopped describing the tree
    # and this binding correctly refused the board. The re-run released against the
    # current bytes -- 33 recorded files, 0 drift, verified before this repoint.
    "folded_dispersive_fused_pair":
        
        "results/triton_fleet_2026-09-25_roundb/folded_dispersive_fused_pair/gate.json",
    # THE 2026-08-31 Dcyl m = 0 ELECTRIC CELL, bound the same way and for the same
    # reason. 13 of 13 recorded files match the tree with zero drift.
    # RE-POINTED to the RE-RUN, and the re-run is the rule rather than a repair: the
    # 08-31 artifact pins this family's HOST SUITE, and a mechanical whole-body diff
    # against the released magnetic twin was added to that suite after the first
    # release. Editing a pinned file after the gate means the artifact certifies bytes
    # that no longer exist, so the gate was RE-RUN rather than the digest re-pointed by
    # hand. 13 of 13 recorded files match the tree with zero drift.
    "cylindrical_real_fused_electric_pair":
        
        "results/triton_fleet_2026-09-25_roundb/cylindrical_real_fused_electric_pair/gate.json",
    # THE 2026-08-31 REAL-BETA D->E CELL — a cell the MAGNETIC twin's own docstring
    # wrote off as worth zero before the deposit repair existed, and the first
    # Triton weld whose HAS_BETA reduction arm was measured DISCRIMINATING on all
    # three value classes rather than merely run. 25 device legs, 611 complete
    # driver steps, 593 fused launches, 26 volumes compared word-for-word against
    # both oracles; 6 carry legs repairing 3 deposits each through the SHIPPED
    # bracket, 2 null controls that both diverge, 14/14 kernel mutations and 3/3
    # host mutations caught, 5/5 refusals.
    "beta_fused_electric_pair":
        
        "results/triton_fleet_2026-09-25_roundb/beta_fused_electric_pair/gate.json",
    # THE 2026-08-31 FOLDED COMPLEX D->E CELL — the largest cell left unbuilt on the
    # 08-31 board and the deepest carry any Triton weld performs: 63 device legs
    # (15 configurations x 3 value classes plus the carry, null-control and armed
    # families), 619 complete driver steps, 613 fused launches, 26 volumes compared
    # word-for-word against three separately certified products; 12 carry legs
    # repairing 6 deposits each through the SHIPPED bracket, 4 null controls that
    # all diverge, 18/22 mutations caught with 2 confirmed nulls and 2 recorded
    # UNREACHED against a word-layer measurement, 5/5 refusals.
    # RE-POINTED 2026-08-31 to the re-run: the plain-branch round edited
    # meep_gpu/deposit_repair.py after the 31f artifact was cut. The re-run needed
    # MEEP_GPU_COMPLEX_EXPANSION_PROBE (the folded_complex probe) and released with
    # 18 pins, 0 drift against the live tree, verified before this repoint.
    # RE-POINTED AGAIN 2026-09-01, same rule: this gate pins its MAGNETIC twin's
    # module (the twin-difference leg reads it), and that module moved when
    # CARRIES_DEPOSIT_REPAIR flipped on it. The family itself did not change;
    # RE-RUN on the GPU host GPU 3 under the same folded_complex expansion probe,
    # released, 18 pins, 0 drift against this tree, verified before this repoint.
    "folded_complex_fused_pair":
        
        "results/triton_fleet_2026-09-25_roundb_env_probe/folded_complex_fused_pair/gate.json",
    # THE 2026-08-31 BFAST D->E CELL — the third of the 08-31 electric cells, and the
    # one whose ARM is a single `if HAS_BFAST:` statement appended to the shipped
    # `kernels.fused_curl_constitutive_D`, so its transcription is checked at
    # STATEMENT grain rather than by text subtraction (the tail's `if BCY ==
    # METALLIC:` lines also occur in the curl mask, and a line-grain diff leaves
    # orphans). 25 device legs, 445 complete driver steps, 413 fused launches, 32
    # volumes compared word-for-word against both oracles; 6 carry legs repairing 6
    # deposits each through the SHIPPED bracket, 2 null controls that both diverge,
    # 19/19 kernel mutations and 4/4 host mutations caught, 5/5 refusals.
    #
    # ITS REDUCTION ARM IS DISCRIMINATING ON ALL THREE VALUE CLASSES: at HAS_BFAST=0
    # the kernel is bit-identical to the shipped `fused_curl_constitutive_D` while
    # BOTH differ from the array path. That leg's floor is TWO-SIDED (the BFAST state
    # inert here, moving on the array path) because the absolute "every array moved"
    # floor it replaced asserted the negation of the leg's own claim and failed a
    # correct kernel on run c; the run c artifact is kept beside this one.
    "bfast_fused_electric_pair":
        
        "results/triton_fleet_2026-09-25_roundb/bfast_fused_electric_pair/gate.json",
    # THE 2026-08-31 CONDUCTIVE PML D->E CELL — released, and serving NOTHING. The
    # binding is recorded anyway because the product is real and the artifact is what
    # says why the cell stays a gap: its one corpus row's electric source is not
    # integrated, and driver.py:3363-3370 rescales such an injection over the WHOLE
    # volume rather than at the deposit. 6 no-device legs and 35 device legs, 897
    # complete driver steps, 858 fused launches, 29 volumes compared word-for-word
    # against both oracles; 9 carry legs repairing 3 deposits each through the SHIPPED
    # bracket, 3 null controls that all diverge, 21/23 kernel mutations caught with 2
    # confirmed nulls re-measured every run, 5/5 host mutations, 7/7 refusals.
    "conductive_fused_electric_pair":
        
        "results/triton_fleet_2026-09-25_roundb/conductive_fused_electric_pair/gate.json",
    # THE 2026-08-31 NO-ABSORBER STORED-E D->E CELLS — released, and the FIRST
    # product on `deposit_repair.PLAIN_PATH`, the second repair. ONE artifact binds
    # BOTH cells because one kernel is both curl arms: `conductive_plain_curl_step`'s
    # per-component COND constexpr compiles to `plain_curl_step`'s own line at zero.
    #
    # 3 no-device legs and 38 device legs: 21 quiet (7 cases x 3 value classes) and
    # 12 CARRY legs repairing 3 deposit cells each through the SHIPPED
    # LeadingRepairPlan/TrailingRepairPlan bracket, each bit-identical per COMPLETE
    # driver step against BOTH the array path and the two separately certified
    # products it replaces; 4 null controls with the bracket REMOVED that all
    # diverge; 1 armed harness leg that must fail on the launch counter alone.
    # 8/10 kernel mutations caught with 2 declared NULL and each carrying the
    # measurement that established it; 4/4 host mutations; 5/5 refusals.
    #
    # AND ITS PRICED REFUSAL IS IN THE ARTIFACT. Two of this cell's three corpus rows
    # declare a NON-INTEGRATED electric source on a conductive run, which the product
    # refuses BY NAME; the gate runs that configuration anyway, with the bracket on,
    # and records it diverging at STEP 1 (Ez, 22 words, 0x80000000 against 0x00000000
    # -- the -0.0 the driver's whole-volume rescale canonicalises) while the SAME case
    # compared only at the last step reads identical.
    "no_pml_fused_electric_pair":
        
        "results/triton_fleet_2026-09-25_roundb/no_pml_fused_electric_pair/gate.json",
    "no_pml_fused_electric_pair_lossless":
        
        "results/triton_fleet_2026-09-25_roundb/no_pml_fused_electric_pair/gate.json",
    # THE TWO 2026-08-31 FOLDED-BETA CELLS — the last two "not built" instances on
    # the plainrepair7 board, bound to their OWN first device runs for the reason
    # every 08-31 weld is: `fingerprints.json` is a release ledger for the shipped
    # kernel set and a matrix cut may not add to it, so the binding reads each
    # gate's `source_sha256` map — every file the run bound itself to — and
    # re-hashes each against the tree. Both artifacts were produced by running the
    # gates against THIS tree, after every edit in the round.
    "folded_beta_fused_electric_pair":
        
        "results/triton_fleet_2026-09-25_roundb/folded_beta_fused_electric_pair/gate.json",
    "folded_beta_fused_magnetic_pair":
        
        "results/triton_fleet_2026-09-25_roundb/folded_beta_fused_magnetic_pair/gate.json",
    # THE TWO 2026-09-01 COMPLEX-BETA CELLS, bound to their OWN first device runs
    # for the reason every recent weld is: `fingerprints.json` is a release
    # ledger for the shipped kernel set and a matrix cut may not add to it, so
    # the binding reads each gate's `source_sha256` map — every file the run
    # bound itself to — and re-hashes each against the tree. Both gates ran on
    # the GPU host GPU 3 under the `keep` policy through an ftz-stripped cache, with
    # MEEP_GPU_COMPLEX_EXPANSION_PROBE naming the 2026-09-01 extension artifact
    # (the probe whose fifth pattern is the whole reason these cells exist).
    "complex_beta_fused_magnetic_pair":
        
        "results/triton_fleet_2026-09-25_roundb_env_probe/complex_beta_fused_magnetic_pair/gate.json",
    "complex_beta_fused_electric_pair":
        
        "results/triton_fleet_2026-09-25_roundb_env_probe/complex_beta_fused_electric_pair/gate.json",
    # THE TWO 2026-09-02 FOLDED COMPLEX-BETA CELLS, bound the same way and run
    # on the GPU host GPU 3 under the `keep` policy through an ftz-stripped cache,
    # with MEEP_GPU_COMPLEX_EXPANSION_PROBE naming the SIX-pattern unified
    # record their predicates and gates require.
    "folded_beta_complex_fused_pair":
        
        "results/triton_fleet_2026-09-25_roundb_env_probe/folded_beta_complex_fused_pair/gate.json",
    "folded_beta_complex_fused_magnetic_pair":
        
        "results/triton_fleet_2026-09-25_roundb_env_probe/folded_beta_complex_fused_magnetic_pair/gate.json",
    # THE H->D WELD, 2026-09-06, bound to its OWN first device run for the reason
    # every recent weld is: `fingerprints.json` is a release ledger for the shipped
    # kernel set and a matrix cut may not add to it, so the binding reads the gate's
    # `source_sha256` map -- every file the run bound itself to, the DRIVER and the
    # COMPOSER included -- and re-hashes each against the tree.
    #
    # WHAT THE GATE MEASURED. the GPU host, RTX A6000, one verified-empty GPU per policy,
    # BOTH canonical float32 subnormal policies (`keep` through a private
    # ftz_stripped CuPy cache, `flush` with MEEP imported for the host half). Per
    # COMPLETE driver step as uint32 over every stored volume, against FOUR
    # references -- the CuPy array path, the two certified singles, the composition
    # the composer installs on these rows today, and the same slots dispatched
    # unfused -- plus `weld_seam_only`, the arrangement that isolates this seam from
    # its neighbours' own agreement with the array path.
    #
    # THE KEEP ARTIFACT IS THE ONE BOUND. Both released; `keep` is bound because it
    # is the policy under which subnormals are preserved on both sides, which is the
    # harder comparison.
    #
    # RE-GATED 2026-09-07 UTC (`_p9b`); THE 2026-09-06 RUN IS SUPERSEDED. Its lift leg
    # re-lifted each corpus row in a child interpreter that never installed the
    # subnormal policy -- native CuPy (flush) against native Triton (keep) under the
    # parent's stamp -- so on the `flush` leg the two certified singles disagreed with
    # the array path on 42 of 43 driven rows and each row was carried by the seam
    # claim alone with its budget cut at the denormal band.
    # `results/triton_hd_lift_policy_2026-09-06` is the autopsy: the two executors part
    # at `dtdx * stencil` where its exact result is subnormal, the IEEE model
    # reproducing the Triton word and the flush model the array path's. The repaired
    # gate installs the policy in the child before its first device compile, refuses a
    # driven row whose child stamp is not the requested policy, requires every
    # arrangement to equal the array path over the full 60-step budget on lifted rows,
    # and runs the ancillary legs at 60 steps rather than 12. Both policies released
    # from EMPTY CuPy and Triton caches: 43 / 43 driven rows bit-identical, 0
    # disagreements with the array path, 43 / 43 children stamped and attained (`keep`
    # children: 32 NVRTC compiles, 32 `-ftz=true` strips, 0 of 12,062 audited PTX
    # instructions carrying `.ftz`; `flush` children: 12,062 of 12,062).
    "fused_hd_pair":
        
        "results/triton_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    # THE TWO CYLINDRICAL H->D PRODUCTS, 2026-09-07, bound the same way to their own
    # first device runs (the GPU host GPU 1, RTX A6000, both policies from EMPTY CuPy and
    # Triton caches, the lift child installing the policy before its first compile).
    # The `keep` artifact is bound for the reason the template gives. The complex
    # product's `flush` record is a DIFFERENT CLAIM -- its arms are certified under
    # `keep` only, so under `flush` the composer selects nothing on the seam and the
    # record measures the forced-arm arithmetic against the array path and the two
    # certified singles built the same way; `composer_reachable_under_this_policy`
    # in the artifact says which reading applies.
    "cylindrical_real_fused_hd_pair":
        
        "results/triton_cylindrical_real_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    "cylindrical_fused_hd_pair":
        
        "results/triton_cylindrical_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    # THE 2026-09-07 H->D TAIL, five products bound to their own first device runs
    # (the GPU host GPU 3 and GPU 6, RTX A6000, EMPTY policy-separated CuPy and Triton
    # caches, the lift child installing the policy before its first device compile).
    #
    # A RE-GATE IS OWED BEFORE ANY OF THESE BINDINGS IS REAL. Every one of these
    # gates PINS `meep_gpu/triton_kernels/launch.py` and `meep_gpu/fastpath.py` in
    # its own `source_sha256` -- MEASURED, not assumed: the artifacts list both --
    # and the sibling H->D rounds landing beside this one edit exactly those files.
    # The artifacts say so themselves, in `what_a_release_here_does_not_move`.
    # Binding them without re-running is the drift these two tables exist to catch.
    #
    # THE TWO COMPLEX FAMILIES ARE BOUND TO THEIR `keep` ARTIFACTS AND TO NOTHING
    # ELSE. Every complex arm in this package was certified under `keep` only, so
    # under `flush` the predicates these products conjoin refuse BY THAT CLAUSE and
    # the `flush` record is a NAMED POLICY REFUSAL rather than a release -- kept
    # beside the keep record, released false, and not a binding.
    "conductive_bfast_fused_hd_pair":
        "results/triton_conductive_bfast_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    "conductive_bfast_fused_hd_pair_bfast":
        "results/triton_conductive_bfast_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    "beta_real_fused_hd_pair":
        "results/triton_beta_real_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    "beta_real_fused_hd_pair_folded":
        "results/triton_beta_real_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    "folded_complex_fused_hd_pair":
        "results/triton_folded_complex_fused_hd_pair_2026-09-25_roundb/keep_probe/gate.json",
    "folded_complex_fused_hd_pair_offdiag":
        "results/triton_folded_complex_fused_hd_pair_2026-09-25_roundb/keep_probe/gate.json",
    "beta_complex_fused_hd_pair":
        "results/triton_beta_complex_fused_hd_pair_2026-09-25_roundb/keep_probe/gate.json",
    "beta_complex_fused_hd_pair_unfolded":
        "results/triton_beta_complex_fused_hd_pair_2026-09-25_roundb/keep_probe/gate.json",
    "nonlinear_fused_hd_pair":
        "results/triton_nonlinear_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    # THE KEEP ARTIFACT IS THE ONE BOUND, for the reason the two cylindrical rows
    # give: the complex arms are certified under `keep` only, so under `flush` the
    # composer selects nothing complex on this seam and the record measures the
    # forced-arm arithmetic against the array path and the two certified singles
    # built the same way. `composer_reachable_under_this_policy` in the artifact
    # says which reading applies.
    "complex_fused_hd_pair":
        "results/triton_complex_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
    # THE FOLDED H->D PRODUCT, bound to its own first device run (the GPU host GPU 1,
    # RTX A6000, both subnormal policies from EMPTY CuPy and Triton caches, the lift
    # child installing the policy before its first device compile). The `keep`
    # artifact is bound for the reason the template above gives.
    "folded_fused_hd_pair":
        "results/triton_folded_fused_hd_pair_2026-09-25_roundb/keep/gate.json",
}


#: PRODUCTS WHOSE RELEASE BINDING WENT STALE AND WAS NEVER REBOUND -- credit
#: WITHHELD, cells reported as gaps, and the headline stated as a floor.
#:
#: EMPTY AS OF 2026-08-28, AND EMPTIED BY A DEVICE CAMPAIGN RATHER THAN BY A
#: JUDGEMENT. What it held, and what closed it, is kept here because the mechanism
#: is worth more than the four entries were: this table is the only reason the
#: defect was ever visible.
#:
#: WHAT IT HELD. The 2026-08-27 routing round rewrote ``triton_kernels/launch.py``,
#: ``triton_kernels/coverage.py`` and ``driver.py``, which every Triton weld pins,
#: and it also rewrote three of the four modules themselves. Thirty ledger entries
#: were re-gated and rebound over 08-27/08-28 (``rebind_triton_welds.py``,
#: ``results/triton_fused_regate_2026-08-28/``); THESE FOUR WERE IN NONE OF THOSE
#: CAMPAIGNS. They carry no ``fingerprints.json`` key at all -- they were bound
#: directly to their 2026-08-20/21 gate artifacts -- so nothing in the weld tooling
#: was ever going to notice. Their drift was in each product's OWN kernel as well as
#: in the shared spine, which is what foreclosed the "only its neighbours moved"
#: reading::
#:
#:     beta_fused_magnetic_pair.py       726a52f4 -> f7a8c561
#:     bfast_fused_magnetic_pair.py      2b9b0887 -> 46ba5670
#:     nonlinear_fused_magnetic_pair.py  77cfbbec -> 9cec1394
#:     cylindrical_fused_magnetic_pair   docstring-stripped AST DIFFERED from its
#:                                       gated copy (it MATCHED on 2026-08-21)
#:
#: WHAT CLOSED IT: all four probes re-run on the GPU host GPU 3 on 2026-08-28 against
#: the tree that ships (``results/triton_gapcell_regate_2026-08-28b/``), each one
#: released and each one matching every file it records. ``GATE_BOUND`` and
#: ``AST_BOUND`` above now name those runs. NOTHING WAS DECLARED AWAY: no digest was
#: edited, no clause relaxed, and the four families' 20 seam-instances move out of
#: the withheld bucket because a device measured the bytes the tree ships, not
#: because this table stopped asking.
#:
#: THE MECHANISM STAYS ARMED, AND EMPTY IS ITS STRICTEST STATE. With no entry here
#: every drift is UNNAMED drift, and the unnamed path RAISES rather than reporting a
#: gap -- so the next weld that goes stale stops the cut instead of quietly costing
#: it instances. Re-add an entry only to publish a board across a drift that a
#: re-gate has not yet repaired, and delete it again the moment one lands.
#:
#: THIS IS STRICTLY STRICTER THAN ``EXTERNAL_DRIFT_AT_CUT_TIME``, which records a
#: concurrent edit and still counts the credit. Nothing here was ever counted.
#:
#: THE EXEMPTION IS RE-EARNED ON EVERY RUN. A family listed here whose bytes DO
#: match its gate raises: the entry would then be withholding credit for a reason
#: that has stopped being true, and a stale exemption understates the board exactly
#: as silently as a stale credit overstates it. Delete the entry when a re-gate
#: lands; do not edit a digest.
#:
#: EMPTIED AGAIN 2026-09-09, BY THE CLOSE ROUND, AND AGAIN BY DEVICE MEASUREMENT
#: RATHER THAN BY A JUDGEMENT. All 36 families it held on 2026-09-08 were re-gated
#: on the committed bytes (15d15c1) in four shards on the GPU host GPUs 0-3 -- the
#: fleet roots ``results/triton_fleet_2026-09-09_close_keep`` and ``..._close0_keep``
#: (``unified_expansion`` first, then the six gates whose probes consume that
#: record), the own H->D gates under ``results/triton_<family>_2026-09-09_close{,2,3}``
#: and the stencil families under ``results/triton_offdiag_stencil_welds_2026-09-09_close4``.
#: Every artifact released, every one re-hashes 0 moved / 0 missing against the
#: tree, ``GATE_BOUND`` names those runs for all 36 rows, and every entry below
#: was deleted the moment its re-gate landed, as the clause requires. The board
#: that reads the empty table is ``results/fusion_matrix_triton_2026-09-09_closed``:
#: 529 of 597, ``buildable_not_built`` 1, withheld 0 -- +70 over the 2026-09-08 cut's
#: 459, all of it the withheld credit returning. The 2026-09-08 account is kept
#: beneath because the mechanism is worth more than the entries were.
#:
#: RE-ARMED 2026-09-08, WITH 36 ENTRIES, AND THE CLAUSE THIS TABLE WROTE FOR
#: ITSELF IS THE ONE BEING USED: "Re-add an entry only to publish a board across
#: a drift that a re-gate has not yet repaired, and delete it again the moment
#: one lands." The wiring merge moved `meep_gpu/triton_kernels/launch.py`
#: (7c4f465a -> de606ebe) and `meep_gpu/fastpath.py` (67b754de -> 4efc4598),
#: plus the five H->D family modules, and the 2026-09-08 Triton campaign ran 30
#: gates -- the ARM gates and the fingerprint-bound fused-pair probes -- so all
#: 11 `FINGERPRINT_BOUND` rows re-hash clean while 36 of the 38 `GATE_BOUND`
#: rows do not. Those 36 were in no campaign this round: the queue it was cut
#: from lists six owed and 49 drifted METAL families and no Triton one.
#:
#: NOTHING HERE IS DECLARED AWAY AND NOTHING HERE IS COUNTED. Every one of these
#: families' seam-instances leaves `served` for the withheld bucket, which is
#: strictly the conservative direction: the board this publishes is a FLOOR
#: under the drift, not a re-statement of `fusion_matrix_triton_2026-09-07_cyl`
#: (422 of 597), and the two figures are not comparable. The names and the moved
#: files the table then held were read off this builder's OWN refusal on the run
#: before it was filled -- 36 rows, one per family -- rather than compiled by hand.
#:
#: THE TWO GATE_BOUND ROWS THAT WERE NEVER HERE are `complex_conductive_fused_pair`
#: and `cylindrical_fused_electric_pair`: their artifacts re-hashed clean against
#: that tree, and an exemption for a family whose bytes DO match raises, exactly
#: as this table's own preamble requires.
#:
#: A DEFECT THIS TABLE'S FIRST NON-EMPTY RUN EXPOSED, CLOSED 2026-09-10.
#: "Nothing here was ever counted" held on the THREE-SEAM walk, which files a
#: withheld instance into `buildable_not_built` under its own named sub-reason —
#: 70 instances on the 2026-09-08 cut. It did NOT hold on the FOURTH seam:
#: `h_to_d_seam.price` was handed a per-row `served_by` this board computes from
#: PRODUCTS' predicates alone, and that module's contract then had no value for
#: "admits but its credit is withheld" — `served_by: None` means the predicate
#: REFUSED, which would be a different and false claim. So the H->D seam credited
#: withheld products, and on that cut it credited nothing else: all 173 of its
#: served instances named one of the 14 spanning products, and 14 of 14 were in
#: this table. The published headline was therefore 286 (three-seam, withheld-
#: aware) + 173 (fourth seam, not withheld-aware) = 459, where applying this
#: table consistently gives 286. Both are in that artifact — `credit_withheld_
#: products` and `h_to_d_seam.served_by_product` — and neither is inferred.
#: CLOSED 2026-09-10, WHILE THE TABLE WAS EMPTY AND IT COST NOTHING:
#: `h_to_d_seam.price` now takes a third per-row answer, `credit_withheld` (this
#: table's own sentence for the product, beside `served_by`, which stays the
#: predicate's verdict), and files such a row buildable_not_built under the SAME
#: withheld sentence this walk files (`fusion_taxonomy.credit_withheld_sub_reason`),
#: out of `served`, `served_by_product` and the dispatch input. The rule was changed
#: on the ``_closed`` cut and moved no number on any board — the replay of every
#: standing board's own H->D instances through the new rule reproduces 173 / 167 /
#: 173 — which is the only time a counting rule may be edited. The next non-empty
#: run of this table withholds on all four seams.
CREDIT_WITHHELD_STALE_BINDING: Dict[str, str] = {
}


#: FILES THAT MOVED UNDER THIS CUT'S FEET, named rather than tolerated.
#:
#: The tree can change while a cut runs. While this cut was being
#: assembled (2026-08-20T22:06 and 22:10 local, measured from the files' mtimes) a
#: CONCURRENT change edited ``folded_fused_magnetic_pair.py`` and its gate, so the
#: digests ``fingerprints.json``'s ``triton_folded_fused_magnetic_pair_device_gate``
#: recorded no longer describe the tree. NOTHING IN THIS ROUND TOUCHED EITHER FILE.
#:
#: THE FLOOR IS NOT LOOSENED. An unnamed drift still raises. What this table does is
#: convert an unexplained refusal into a DOCUMENTED one: the affected product's
#: credit is marked UNVERIFIABLE ON THIS RUN and its seam-instances are reported
#: SEPARATELY, so the headline can be read with and without them. The bytes may well
#: be fine — a re-gate of those files is what would say so — but this
#: cut cannot assert it, and asserting it anyway is exactly the failure the binding
#: check exists to prevent.
EXTERNAL_DRIFT_AT_CUT_TIME: Dict[str, str] = {
    "folded_fused_magnetic_pair":
        "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py and "
        "parity/meep_gpu/probe_triton_folded_fused_magnetic_pair.py were modified by "
        "a CONCURRENT change while this cut ran (mtimes 2026-08-20T22:06/22:10); "
        "this round edited neither. Its credit is reported separately below.",
}
#: THE FILES EACH EXEMPTION WAS WRITTEN FOR, and only those. Until 2026-09-15 the loop
#: below exempted ANY drift on a family named above, so when
#: ``triton_kernels/launch.py`` moved under ``folded_fused_magnetic_pair`` -- the same
#: file 21 refused products drifted on -- this cut credited its 74 dispatching
#: instances silently. A drift on any file not listed here is an ordinary refusal.
EXTERNAL_DRIFT_FILES: Dict[str, Tuple[str, ...]] = {
    "folded_fused_magnetic_pair": (
        "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py",
        "parity/meep_gpu/probe_triton_folded_fused_magnetic_pair.py",
    ),
}


def _docstring_stripped_ast(path: Path) -> str:
    """The module's executable content, with EVERY docstring removed and nothing
    else. A prose edit moves the file's sha and leaves this string alone; any edit
    to a statement, an expression, a name or a non-docstring constant moves it."""
    import ast  # noqa: PLC0415
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())
    ast.fix_missing_locations(tree)
    return ast.dump(tree)


def assert_every_credit_is_bound_to_released_bytes() -> Dict[str, Any]:
    """EVERY product's credited bytes must still BE the bytes its gate certified.

    RAISES. An unverifiable credit is worse than an absent one.
    """
    import hashlib  # noqa: PLC0415
    unbound = sorted(set(PRODUCTS) - set(FINGERPRINT_BOUND) - set(AST_BOUND)
                     - set(GATE_BOUND))
    if unbound:
        raise SystemExit(
            f"{unbound} are credited by this matrix but bound to no gate record, so "
            f"their seam-instances rest on bytes nothing here can check")
    package = API / "meep_gpu" / "triton_kernels"
    fingerprints = json.loads((package / "fingerprints.json").read_text())
    out: Dict[str, Any] = {}
    drift: List[str] = []
    external: List[str] = []
    withheld: List[str] = []
    # The families the withholding ACTUALLY fired on, kept as a set rather than
    # re-parsed out of the message strings above (two of which have different shapes).
    withheld_families: set = set()
    log("\n=== RELEASE BINDING: are the credited bytes the certified bytes? ===")
    for family in sorted(PRODUCTS):
        if family in FINGERPRINT_BOUND:
            key = FINGERPRINT_BOUND[family]
            entry = fingerprints.get(key)
            if not entry:
                raise SystemExit(f"triton fingerprints.json carries no {key!r} entry")
            recorded = entry.get("source_sha256") or {}
            if not recorded:
                raise SystemExit(f"{key} records no source digests")
            moved = []
            for relative, want in sorted(recorded.items()):
                target = API / relative
                got = (hashlib.sha256(target.read_bytes()).hexdigest()
                       if target.exists() else "MISSING")
                if got != want:
                    moved.append(relative)
            exempt = set(EXTERNAL_DRIFT_FILES.get(family, ()))
            unexempted = [relative for relative in moved if relative not in exempt]
            if unexempted:
                drift.append(f"{family} / {key}: {unexempted}")
            elif moved:
                external.append(f"{family} / {key}: {moved}")
            out[family] = {"bound_by": "fingerprint", "key": key,
                           "credit_unverifiable_on_this_run": bool(
                               moved and not unexempted),
                           "status": entry.get("status"),
                           "files_checked": len(recorded), "files_moved": moved}
            log(f"    {family:42s} {str(entry.get('status')):5s} "
                f"{len(recorded):2d} files  "
                f"{'DRIFT ' + str(moved) if moved else 'bytes match'}")
        elif family in GATE_BOUND:
            gate_relative = GATE_BOUND[family]
            gate_path = RESULTS.parent / gate_relative
            if not gate_path.exists():
                raise SystemExit(
                    f"{family} is credited by this matrix but its gate artifact "
                    f"{gate_relative} is absent, so the credit rests on nothing")
            gate = json.loads(gate_path.read_text())
            release = gate.get("release") or {}
            # THE THREE CONDITIONS, all of them. See GATE_BOUND's note: a run that
            # measured no bytes on any device must not be readable as a release.
            #
            # TWO ARTIFACT SHAPES SINCE 2026-09-04, AND THE SECOND IS NOT A WEAKER
            # RULE. The probes in this package write their verdict in two shapes.
            # The one this branch was written for carries `device_status`, `release`
            # and `passed` as keys. The OLDER shape — the one
            # probe_triton_cylindrical_fused_magnetic_pair.py writes — carries none
            # of the three: its verdict is the prose `verdict` key, which is what
            # `canonical_verdict.read_from` names, and it is the same shape
            # rebind_triton_welds._verdict reads when it binds a weld. Refusing that
            # shape outright would not be strictness, it would be reading the
            # artifact's FORMAT as its verdict.
            #
            # So a second condition set is applied to it, and it is chosen to be at
            # least as hard to satisfy off a device as the first. `device_status ==
            # "RUN"` exists to stop a `--no-device` or host-only leg being read as a
            # release; the substitute for it here is the run's OWN subnormal-policy
            # install stamps, which record, per canonical policy, which executors
            # were ATTAINED. A host-only run attains `host` and nothing else; the
            # artifact this admits records `cupy` AND `triton` attained under BOTH
            # canonical policies (measured on
            # results/triton_regate_2026-09-04_cylm0/cylindrical_fused_magnetic_pair:
            # keep and flush, {'cupy': True, 'host': True, 'triton': True} on each),
            # which no run without a working CUDA device can write. `passed` is
            # replaced by the prose verdict PASS plus an EMPTY failure list and every
            # recorded leg passing — 89 of 89 on that artifact — which is a stronger
            # statement than the single boolean.
            reasons = []
            if any(key in gate for key in ("device_status", "release", "passed")):
                shape = "device_status/release/passed"
                if gate.get("device_status") != "RUN":
                    reasons.append(f"device_status={gate.get('device_status')!r}")
                if release.get("released") is not True:
                    reasons.append(f"release.released={release.get('released')!r}")
                if gate.get("passed") is not True:
                    reasons.append(f"passed={gate.get('passed')!r}")
            else:
                shape = "verdict(prose) + attained-executor stamps"
                canonical = gate.get("canonical_verdict") or {}
                stamps = ((gate.get("environment") or {})
                          .get("subnormal_policy_stamps") or {})
                attained = {
                    policy: {name: bool(block.get("attained"))
                             for name, block in
                             (((stamp.get("install") or {}).get("executors")) or {}
                              ).items()}
                    for policy, stamp in stamps.items()}
                on_device = sorted(
                    policy for policy, executors in attained.items()
                    if executors.get("cupy") and executors.get("triton"))
                legs = gate.get("rows") or []
                if canonical.get("released") is not True:
                    reasons.append(
                        f"canonical_verdict.released={canonical.get('released')!r}")
                if str(gate.get("verdict")).upper() != "PASS":
                    reasons.append(f"verdict={gate.get('verdict')!r}")
                if sorted(on_device) != ["flush", "keep"]:
                    reasons.append(
                        f"cupy+triton executors attained under {on_device} rather "
                        f"than under both canonical policies ({attained})")
                if gate.get("failures") or gate.get("failed_rows"):
                    reasons.append(f"failures={gate.get('failures')!r} "
                                   f"failed_rows={gate.get('failed_rows')!r}")
                if not legs or any(leg.get("passed") is not True for leg in legs):
                    reasons.append(
                        f"{sum(1 for leg in legs if leg.get('passed') is True)} of "
                        f"{len(legs)} recorded legs passed")
            if gate.get("planted_defect"):
                reasons.append(f"planted_defect={gate.get('planted_defect')!r}")
            if reasons:
                raise SystemExit(
                    f"{family}'s gate artifact {gate_relative} is not a passed "
                    f"device release (read under the {shape} shape): {reasons}")
            # THE DIGEST MAP, AND THE ONE NON-PATH KEY IN IT. `source_sha256` is a
            # map of checkout-relative path -> digest on every artifact this branch
            # reads, with ONE exception: the verdict-prose shape also carries the key
            # `gate`, which is the digest of the PROBE SCRIPT rather than a path. It
            # is not dropped — a key nobody checks is a pin nobody has — it is
            # RESOLVED: the same run's `imported_source_sha256` records the probe by
            # its real path, and the two digests must agree before the map is walked.
            # The prose-shaped artifact's `source_sha256` carries only the module and
            # that key, so the import map (34 paths on the 2026-09-04 magnetic gate,
            # every file the run actually imported) is what the re-hash walks.
            recorded = dict(gate.get("source_sha256") or {})
            imported = dict(gate.get("imported_source_sha256") or {})
            gate_digest = recorded.pop("gate", None)
            if gate_digest is not None:
                probe = sorted(path for path, digest in imported.items()
                               if digest == gate_digest
                               and Path(path).name.startswith("probe_"))
                if not probe:
                    raise SystemExit(
                        f"{family}'s gate artifact {gate_relative} records a `gate` "
                        f"digest {gate_digest[:12]} that matches no probe script in "
                        f"its own import map, so the script that produced the "
                        f"verdict cannot be identified or re-hashed")
                recorded.update({path: imported[path] for path in probe})
            if shape.startswith("verdict(prose)"):
                # ONLY THIS SHAPE WIDENS TO THE IMPORT MAP, and only because its
                # `source_sha256` is two keys wide. The other shape records its own
                # full file list there, and widening it here would silently change
                # what every other family on this board is checked against.
                recorded.update(imported)
            module_relative = ("meep_gpu/triton_kernels/"
                               + PRODUCTS[family]["module"].split(":")[0])
            if module_relative not in recorded:
                raise SystemExit(
                    f"{family}'s gate records no digest for {module_relative}, so "
                    f"the credited bytes are not the ones it ran against")
            moved = []
            for relative, want in sorted(recorded.items()):
                target = API / relative
                got = (hashlib.sha256(target.read_bytes()).hexdigest()
                       if target.exists() else "MISSING")
                if got != want:
                    moved.append(relative)
            if moved:
                if family in CREDIT_WITHHELD_STALE_BINDING:
                    withheld.append(f"{family} / {gate_relative}: {moved}")
                    withheld_families.add(family)
                else:
                    drift.append(f"{family} / {gate_relative}: {moved}")
            out[family] = {"bound_by": "gate artifact source_sha256",
                           "gate": gate_relative,
                           "credit_withheld": bool(
                               moved and family in CREDIT_WITHHELD_STALE_BINDING),
                           "status": "RELEASED" if release.get("released") else "NO",
                           "subnormal_policy": (gate.get("subnormal_policy")
                                                or {}).get("policy"),
                           "files_checked": len(recorded), "files_moved": moved}
            log(f"    {family:42s} RELEASED  {len(recorded):2d} files  "
                f"{'DRIFT ' + str(moved) if moved else 'bytes match'}  "
                f"({gate_relative})")
        elif family in AST_BOUND:
            copy_relative, gate_relative = AST_BOUND[family]
            gated = RESULTS.parent / copy_relative
            module = package / (PRODUCTS[family]["module"].split(":")[0])
            gate = json.loads((RESULTS.parent / gate_relative).read_text())
            if not gated.exists():
                raise SystemExit(
                    f"{gated} is absent, so the AST equivalence that binds "
                    f"{family}'s credit cannot be re-checked and the credit is "
                    f"unverifiable on this run")
            # The gated copy must BE the bytes the gate recorded, or the comparison
            # is against a file of unknown provenance.
            want = (gate.get("source_sha256") or {}).get(
                f"meep_gpu/triton_kernels/{module.name}")
            got = hashlib.sha256(gated.read_bytes()).hexdigest()
            if want and got != want:
                raise SystemExit(
                    f"the local copy of {family}'s gated bytes ({gated}) has sha "
                    f"{got[:12]}, but the gate recorded {want[:12]}; the copy is not "
                    f"the certified file and proves nothing")
            same = _docstring_stripped_ast(gated) == _docstring_stripped_ast(module)
            if not same:
                message = (f"{family}: executable content differs from the gated "
                           f"bytes, not only the docstring")
                if family in CREDIT_WITHHELD_STALE_BINDING:
                    withheld.append(message)
                    withheld_families.add(family)
                else:
                    drift.append(message)
            out[family] = {"bound_by": "docstring-stripped AST",
                           "credit_withheld": bool(
                               not same and family in CREDIT_WITHHELD_STALE_BINDING),
                           "gate": gate_relative, "status": gate.get("verdict"),
                           "gated_sha256": got,
                           "tree_sha256": hashlib.sha256(
                               module.read_bytes()).hexdigest(),
                           "executable_content_identical": same}
            log(f"    {family:42s} {str(gate.get('verdict')):5s} "
                f"no fingerprints.json entry (deliberate); tree sha "
                f"{out[family]['tree_sha256'][:8]} != gated {got[:8]}, "
                f"docstring-stripped AST "
                f"{'IDENTICAL' if same else '*** DIFFERS ***'}")
        else:
            # UNREACHABLE while the `unbound` guard above stands, and written out
            # rather than left as an implicit fall-through: the AST table is empty
            # since 2026-09-04, and a family that reached here would otherwise be
            # credited with no binding read at all.
            raise SystemExit(
                f"{family} is credited by this matrix and named by no binding table; "
                f"the `unbound` guard should have refused before this loop")
    # THE EXEMPTION IS RE-EARNED, NEVER INHERITED. A family named in
    # CREDIT_WITHHELD_STALE_BINDING that did NOT land in `withheld` is a family whose
    # bytes now match its gate -- so the reason the entry states is false, and
    # withholding its instances would understate the board on a fiction.
    unearned = sorted(set(CREDIT_WITHHELD_STALE_BINDING) & set(PRODUCTS)
                      - withheld_families)
    if unearned:
        raise SystemExit(
            f"CREDIT_WITHHELD_STALE_BINDING withholds credit from {unearned}, but "
            f"their credited bytes MATCH their gates on this tree. The stated reason "
            f"has stopped being true: delete the entry (a re-gate has landed) rather "
            f"than leaving a board that under-counts a released product.")
    if drift:
        raise SystemExit("the bytes this cut credits are NOT the bytes the gates "
                         "certified:" + "".join(f"\n    {d}" for d in drift))
    if withheld:
        log("\n  *** CREDIT WITHHELD: STALE RELEASE BINDING, NEVER REBOUND ***")
        for entry in withheld:
            log(f"    {entry}")
        for family in sorted(withheld_families):
            log(f"      {family}: {CREDIT_WITHHELD_STALE_BINDING[family]}")
        log("    These products' seam-instances are counted as GAPS in their own "
            "bucket. The served total below is a FLOOR, short by whatever they serve.")
    if external:
        log("\n  *** EXTERNAL DRIFT, NAMED AND NOT TOLERATED SILENTLY ***")
        for entry in external:
            family = entry.split(" /")[0]
            log(f"    {entry}")
            log(f"      {EXTERNAL_DRIFT_AT_CUT_TIME[family]}")
        log("    Those products' seam-instances are reported SEPARATELY below; the "
            "headline is given with and without them.")
    out["_credit_withheld"] = {
        "families": sorted(withheld_families),
        "entries": withheld,
        "why": dict(CREDIT_WITHHELD_STALE_BINDING),
    }
    out["_external_drift"] = {
        "families": sorted({entry.split(" /")[0] for entry in external}),
        "entries": external,
        "why": {k: v for k, v in EXTERNAL_DRIFT_AT_CUT_TIME.items()
                if any(entry.startswith(k + " /") for entry in external)},
    }
    return out


def assert_every_built_product_is_asked_about() -> Dict[str, Any]:
    """A MATRIX THAT DOES NOT ASK ABOUT A LANDED PRODUCT CANNOT REPORT IT.

    That is not hypothetical here. Both 2026-08-20b matrices declined in writing to
    ask about ``cylindrical_fused_magnetic_pair.py`` while the kernel was already
    welded and gated, and the same happened to the Metal E->P chain: a released
    product whose worth had never been counted because no census had a column for
    it. This floor makes that failure loud instead of silent — every fused module in
    the package must appear in :data:`PRODUCTS` or be exempted BY NAME above.
    """
    package = API / "meep_gpu" / "triton_kernels"
    built = sorted(path.name for path in package.glob("*.py")
                   if ("fused" in path.name or "chain" in path.name)
                   and not path.name.startswith("test_"))
    asked = {spec["module"].split(":")[0].split()[0] for spec in PRODUCTS.values()}
    unasked = [name for name in built
               if name not in asked and name not in NOT_A_SEAM_PRODUCT
               and name not in NOT_PRICED_HERE
               and name not in GATED_NOT_PRICED_HERE
               and name not in LANDED_CONCURRENTLY_NOT_PRICED_HERE]
    if unasked:
        raise SystemExit(
            f"triton_kernels ships fused modules this matrix does not ask about: "
            f"{unasked}. A landed product missing from PRODUCTS is invisible — its "
            f"cell reads 'no product' and its seam-instances are reported as a gap. "
            f"Wire it in, or name it in NOT_A_SEAM_PRODUCT with the reason it "
            f"occupies no cell.")

    # ------------------------------------------------------------------
    # THE EXEMPTION IS RE-EARNED ON EVERY RUN, NEVER INHERITED, AND THE CHECK IS
    # THE OPPOSITE OF THE ONE IT REPLACED.
    #
    # The first version of this block withheld credit because NOTHING HAD CERTIFIED
    # the module, and RAISED the moment a gate appeared. It raised, twelve minutes
    # later, exactly as designed — the concurrent round finished and landed both a
    # fingerprint key and a results directory. So the reason changed, and with it
    # the check: the product is now CERTIFIED, and what it still lacks is an
    # ADMISSION CENSUS on this corpus.
    #
    # This therefore asserts the OPPOSITE condition — the gate must be REAL,
    # measured from the tree, not asserted in a comment. A gate that is missing,
    # unreleased, or drifted from the tree means the entry's stated reason is
    # false, and a false reason on an exemption is how a gap stops being visible.
    # ------------------------------------------------------------------
    for name in sorted(LANDED_CONCURRENTLY_NOT_PRICED_HERE):
        if not (package / name).exists():
            raise SystemExit(
                f"LANDED_CONCURRENTLY_NOT_PRICED_HERE names {name}, which "
                f"triton_kernels does not ship. An exception list that outlives its "
                f"exception silences the floor it was carved out of; delete it.")
    if LANDED_CONCURRENTLY_NOT_PRICED_HERE:
        log("\n*** LANDED CONCURRENTLY, NOT PRICED HERE "
            f"({len(LANDED_CONCURRENTLY_NOT_PRICED_HERE)} module(s)) — this cut's "
            "Triton total is a LOWER BOUND by whatever they serve:")
        for name, why in sorted(LANDED_CONCURRENTLY_NOT_PRICED_HERE.items()):
            log(f"***   {name:26s} {why}")

    ungated_audit: Dict[str, Any] = {}
    fingerprints_path = API / "meep_gpu" / "triton_kernels" / "fingerprints.json"
    fingerprints = json.loads(fingerprints_path.read_text())
    for name in sorted(GATED_NOT_PRICED_HERE):
        stem = name[:-3]
        if not (package / name).exists():
            raise SystemExit(
                f"GATED_NOT_PRICED_HERE names {name}, which triton_kernels does "
                f"not ship. An exception list that outlives its exception silences "
                f"the floor it was carved out of; delete the entry.")
        keys = sorted(k for k in fingerprints if stem in k)
        directories = sorted(p.name for p in RESULTS.glob(f"triton_{stem}*")
                             if p.is_dir())
        if not keys and not directories:
            raise SystemExit(
                f"{name} is exempted as GATED-BUT-NOT-PRICED, but the tree carries "
                f"NO evidence of a gate: no fingerprints key and no results "
                f"directory. An UNCERTIFIED product must be refused credit for a "
                f"different reason than a certified one, and the entry's stated "
                f"reason is now false.")
        drifted = []
        for key in keys:
            entry = fingerprints.get(key) or {}
            if entry.get("status") != "PASS":
                raise SystemExit(
                    f"{name}: fingerprint {key} records status "
                    f"{entry.get('status')!r}, not PASS — the exemption claims a "
                    f"released gate it does not have")
            for rel, want in (entry.get("source_sha256") or {}).items():
                target = API / rel
                got = (hashlib.sha256(target.read_bytes()).hexdigest()
                       if target.exists() else "MISSING")
                if got != want:
                    drifted.append(f"{rel} {got[:12]} != {want[:12]}")
        if drifted:
            raise SystemExit(
                f"{name}: the gate this exemption cites has DRIFTED from the tree: "
                f"{drifted}. Re-gate, or the stated reason is false.")
        ungated_audit[name] = {
            "why": GATED_NOT_PRICED_HERE[name],
            "fingerprint_keys_found": keys,
            "gate_result_directories_found": directories,
            "fingerprint_status": [fingerprints[k].get("status") for k in keys],
            "fingerprint_sources_checked": sum(
                len(fingerprints[k].get("source_sha256") or {}) for k in keys),
            "fingerprint_sources_drifted": 0,
            "credit": "NOT COUNTED — 0 seam-instances; upper bound 10 of 15 E->P",
        }
    if GATED_NOT_PRICED_HERE:
        log("\n*** CERTIFIED BUT NOT PRICED HERE "
            f"({len(GATED_NOT_PRICED_HERE)} module(s)) — this cut's Triton total "
            f"is a FLOOR:")
        for name, record in sorted(ungated_audit.items()):
            log(f"***   {name:22s} {record['why']}")
            log(f"***   {'':22s} gate: {record['fingerprint_keys_found']} "
                f"{record['fingerprint_status']}, "
                f"{record['fingerprint_sources_checked']} sources, 0 drifted; "
                f"dirs {record['gate_result_directories_found']}")
        log("*** Their instances are counted as GAPS below. Closing the gap needs "
            "an ADMISSION")
        log("*** CENSUS on a Triton host — not another cut on this laptop.")
    # ------------------------------------------------------------------
    # THE RELATION, ASSERTED IN BOTH DIRECTIONS, AND BOTH COUNTS REPORTED.
    #
    # Forward: every fused module the package ships is priced by PRODUCTS (the
    # block above). Reverse: every priced product names a module that exists.
    # A verifier found the METAL board registering 14 fused families while its
    # matrix priced 7 — the board under-reported ITSELF, and a one-directional
    # floor cannot see that. The failure is symmetric, so the floor is too.
    #
    # The relation is many-to-one, not one-to-one: one module may host several
    # products (Metal's `fused_ade_chain.py` hosts three). So the check is not
    # "the counts are equal" — it is "neither SET exceeds the other under the
    # module map", and BOTH counts are printed so a reader can see the fan-out
    # rather than infer it.
    # ------------------------------------------------------------------
    priced_to_module = {name: spec["module"].split(":")[0].split()[0]
                        for name, spec in PRODUCTS.items()}
    absent = sorted(f"{name} -> {module}" for name, module in priced_to_module.items()
                    if not (package / module).exists())
    if absent:
        raise SystemExit(
            f"PRODUCTS prices families whose module is not in the package: "
            f"{absent}. A credit against a file that does not exist is a credit "
            f"against nothing.")

    # NOT_PRICED_HERE is an escape hatch that makes the served total a LOWER
    # BOUND. This cut exists to retire it: every family that had landed
    # un-priced is priced here. A non-empty hatch is therefore a FAILURE, not a
    # warning — a warning is how a lower bound gets read as a measurement.
    stale = sorted(name for name in NOT_PRICED_HERE if name not in built)
    if stale:
        raise SystemExit(
            f"NOT_PRICED_HERE names {stale}, which the tree does not ship. An "
            f"exception list that outlives its exception silences the floor it "
            f"was carved out of; delete the entry.")
    if NOT_PRICED_HERE:
        raise SystemExit(
            f"NOT_PRICED_HERE is non-empty: {sorted(NOT_PRICED_HERE)}. This cut "
            f"reports a MEASUREMENT, not a lower bound. Price the family or "
            f"withdraw the headline.")

    covered_modules = sorted(set(priced_to_module.values()))
    log(f"\n=== COMPLETENESS: the product set, asserted in both directions ===")
    log(f"  fused modules the package ships     : {len(built)}")
    log(f"  ... exempt (occupy no seam cell)    : "
        f"{len([n for n in built if n in NOT_A_SEAM_PRODUCT])}")
    log(f"  ... priced by PRODUCTS              : {len(covered_modules)}")
    log(f"  products priced (may fan out per module): {len(PRODUCTS)}")
    log(f"  priced products with no module      : {len(absent)}")
    log(f"  ... CERTIFIED but not priced here (0 counted): "
        f"{len(GATED_NOT_PRICED_HERE)}")
    for name in built:
        why = NOT_A_SEAM_PRODUCT.get(name)
        if name in GATED_NOT_PRICED_HERE:
            state = "GATED, NOT PRICED HERE - 0 counted, cells are GAPS"
        elif name in LANDED_CONCURRENTLY_NOT_PRICED_HERE:
            state = "LANDED CONCURRENTLY, NOT PRICED HERE - 0 counted, cells are GAPS"
        elif why:
            state = "EXEMPT - " + why[:60]
        else:
            state = "in PRODUCTS"
        hosted = sorted(n for n, m in priced_to_module.items() if m == name)
        if hosted and len(hosted) > 1:
            state += f"  [hosts {len(hosted)}: {', '.join(hosted)}]"
        log(f"  {name:44s} {state}")
    unpriced_and_unexempt = [n for n in built
                             if n not in covered_modules
                             and n not in NOT_A_SEAM_PRODUCT
                             and n not in GATED_NOT_PRICED_HERE
                             and n not in LANDED_CONCURRENTLY_NOT_PRICED_HERE]
    if unpriced_and_unexempt:
        raise SystemExit(
            f"forward direction FAILS: {unpriced_and_unexempt}")
    return {"modules_found": built,
            "exempt": dict(NOT_A_SEAM_PRODUCT),
            "not_priced_here": dict(NOT_PRICED_HERE),
            "gated_but_not_priced_here": ungated_audit,
            "products_asked": sorted(PRODUCTS),
            "n_modules_built": len(built),
            "n_modules_priced": len(covered_modules),
            "n_modules_gated_but_not_priced_here": len(GATED_NOT_PRICED_HERE),
            "n_products_priced": len(PRODUCTS),
            "every_priced_product_has_a_module": priced_to_module,
            "relation_asserted_in_both_directions": True}


def report_the_carry(instances: List[dict]) -> Dict[str, Any]:
    """Which in-seam passes a product actually CARRIES, and what that gates.

    MEASURED ON THE CORPUS, NOT READ OFF ``REPLACES``. A product's ``REPLACES``
    lists the driver call sites one launch takes over INCLUDING INERT ONES, so
    reading it as a carry credits a pass to a product that never executes it. A pass
    counts as CARRIED only where some product ADMITS an instance on which that pass
    is LIVE.
    """
    carried: Dict[str, set] = {seam: set() for seam in IN_SEAM_PASSES}
    for entry in instances:
        if entry["seam"] in carried and entry["predicate_admits"]:
            carried[entry["seam"]].update(entry.get("live_in_seam_passes", ()))
    evidence = {seam: any(e["predicate_admits"] for e in instances
                          if e["seam"] == seam)
                for seam in IN_SEAM_PASSES}
    log("\n=== WHICH IN-SEAM PASSES A PRODUCT ACTUALLY CARRIES ===")
    for seam, names in IN_SEAM_PASSES.items():
        for pass_name in names:
            if not evidence[seam]:
                mark = "UNDETERMINED — nothing is served at this seam"
            elif pass_name in carried[seam]:
                mark = "CARRIED"
            else:
                mark = "NOT CARRIED BY ANY PRODUCT"
            log(f"    {seam:5s} {pass_name:26s} {mark}")

    gap: Dict[str, List[dict]] = collections.defaultdict(list)
    for entry in instances:
        seam = entry["seam"]
        if seam not in IN_SEAM_PASSES or entry["predicate_admits"]:
            continue
        if entry["in_seam_source_blocks"] or not evidence[seam]:
            continue
        for pass_name in entry.get("live_in_seam_passes", ()):
            if pass_name in carried[seam]:
                continue
            gap[pass_name].append({
                "row": entry["row"], "seam": seam,
                "cell": [entry["curl_arm"], entry["constitutive_arm"]],
                "has_a_product": bool(entry["product"])})
    log("\n=== THE CARRY GAP — reachable, unserved, waiting on ONE uncarried pass ===")
    if not gap:
        log("    none")
    for pass_name, entries in sorted(gap.items(), key=lambda kv: -len(kv[1])):
        inside = sum(1 for e in entries if e["has_a_product"])
        log(f"    {pass_name:26s} {len(entries):4d} seam-instances "
            f"({inside} inside a cell a SHIPPED product covers and REFUSES)")
        for cell, count in collections.Counter(
                (e["seam"], str(e["cell"][0]), str(e["cell"][1]))
                for e in entries).most_common():
            log(f"          {count:4d}  {cell[0]:5s}  ({cell[1]}, {cell[2]})")
    return {"carried_by_some_product": {s: sorted(v) for s, v in carried.items()},
            "seam_has_evidence": evidence,
            "carry_gap_totals": {k: len(v) for k, v in gap.items()},
            "carry_gap": {k: v for k, v in gap.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="the Triton fusion board")
    parser.add_argument("--out", type=Path, required=True,
                        help="directory to write fusion_matrix.json into")
    parser.add_argument("--census", default=None,
                        help="census directory name under parity/meep_gpu/results; "
                             "overrides MEEP_GPU_TRITON_CENSUS")
    args = parser.parse_args()
    global CENSUS
    if args.census:
        CENSUS = RESULTS / args.census
    if not (CENSUS / "examples.jsonl").exists():
        raise SystemExit(f"{CENSUS} carries no examples.jsonl; that is not a census")
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"TRITON FUSION MATRIX — {len(rows(CENSUS))} corpus rows, four seams")
    log(f"census : {CENSUS}")
    log(f"fresh  : {FRESH}")
    log(f"out    : {out_dir}")

    result: Dict[str, Any] = {
        "backend": "triton",
        "census": str(CENSUS),
        "census_provenance": _census_provenance(),
        "census_unlicensed_control": str(CENSUS_UNLICENSED),
        "fresh_corpus_record": str(FRESH),
        "coverage_column": "covered_modulo_backend",
    }
    result["freshness"] = freshness_check()
    result["licensed_vs_unlicensed_predicate_deltas"] = licensed_vs_unlicensed()
    result["staleness_audit"] = staleness_audit()
    result["expansion_licence_recheck"] = recheck_the_expansion_licence()

    result["product_table_is_complete"] = assert_every_built_product_is_asked_about()
    result["release_binding_per_product"] = assert_every_credit_is_bound_to_released_bytes()
    result["planner_wiring_is_measured"] = assert_the_wiring_column_is_measured()

    record = rows(CENSUS)
    instances, meta = build_matrix(record)
    result.update(meta)
    result["in_seam_pass_liveness"] = report_in_seam_liveness(
        record, in_seam_liveness())
    result["carry"] = report_the_carry(instances)
    result["aggregate"] = report(instances, record)
    result["funnels"] = funnels(record)
    result["seam_instances"] = instances

    log("\n=== THE UNFUSABLE ===")
    for entry in UNFUSABLE:
        log(f"  {entry['kind']}")
        log(f"    {entry['pair']}")
        log(f"    why  : {entry['reason']}")
        log(f"    where: {entry['where']}")
        log(f"    refused by: {entry['refused_by']}")
    result["unfusable"] = [dict(entry) for entry in UNFUSABLE]

    log("\n=== CAVEATS ===")
    lines = caveats(record)
    for line in lines:
        log(f"  * {line}")
    result["is_an_upper_bound"] = True
    result["clauses_not_evaluable_from_the_census_block"] = list(lines)

    # FLOOR 7: every served total this artifact publishes, at any depth, is the
    # instance ledger's count — refused, not recorded, where one is not.
    result["headline_agreement"] = fusion_taxonomy.headline_agreement(
        result, result["aggregate"]["taxonomy"], "triton")
    # AND THE SAME FLOOR ON THE DISPATCH AXIS. `served` and `served_in_dispatch` are
    # different numbers, and until 2026-09-11 only the first had a floor -- which is
    # how the Metal board came to publish a hardcoded dispatch 0 beside its own
    # measured 176 in one artifact. `dispatch_agreement` reads the measurement as the
    # authority and refuses any other number for the same question, listing (never
    # comparing) the totals that sit under a declared different CONDITION.
    # READ OFF THE ARTIFACT, not off a local: this board's ``dispatch_reach`` is
    # computed in the aggregate builder, not in ``main``, and the block it embedded is
    # the same object -- reading it here keeps one authority instead of two.
    result["dispatch_agreement"] = dispatch_agreement.dispatch_agreement(
        result, result["aggregate"]["served_in_dispatch_measurement"], "triton")
    log(f"DISPATCH AGREEMENT: "
        f"{result['dispatch_agreement']['served_in_dispatch']} in dispatch at "
        f"{len(result['dispatch_agreement']['paths_checked'])} published paths, all "
        f"read from the measurement")
    log(f"\nHEADLINE AGREEMENT: {result['headline_agreement']['served']} served at "
        f"{len(result['headline_agreement']['paths_checked'])} published paths, all "
        f"derived from the instance ledger")

    out = out_dir / "fusion_matrix.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
