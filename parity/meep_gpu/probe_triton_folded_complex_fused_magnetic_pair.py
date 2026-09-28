"""Byte gate for the FOLDED COMPLEX fused magnetic pair — folded Bloch ``step_B``
welded into complex ``update_H``, with both mirror fills and the wall clear carried.

DEVICE STATUS: **RELEASED 2026-08-21**, the GPU host RTX A6000 (physical GPU 1, pinned
    by UUID and verified empty before and after), Triton 3.1.0, CuPy 13.5.1, under
    the ``keep`` float32 subnormal policy. 46 byte legs (16 configurations x 3
    operand value classes), 447 COMPLETE driver steps, 447 fused launches, 26
    volumes per leg, ALL BIT-IDENTICAL against both oracles; 4/4 armed harness
    mutations refused as designed; 15 kernel mutations 12 CAUGHT / 1 null / 2
    UNREACHED; 3/3 refusals; 9/9 no-device legs. Artifact:
    ``results/triton_folded_complex_fused_magnetic_pair_2026-08-21/keep/gate.json``,
    ``release.released = true``, 0 source drift against the tree.

    SEVEN RUNS, AND THE SIX THAT FAILED ARE WHY THE SEVENTH MEANS ANYTHING. Each
    refusal was a defect in THIS HARNESS rather than in the kernel, and each is
    fixed in place with the measurement that found it written beside it:

    * an ABSOLUTE vacuity floor ("every array moved") failed a correct kernel on
      every 2-D ``zero_lattice`` leg, where the TE polarization is exactly zero
      forever, and on every single-step leg, where ``fu_B`` does not move. The floor
      is now RELATIVE to the array path's own census plus a positive seam-output
      requirement — strictly stronger where it matters;
    * ``armed:unlicensed_expansion_arm`` reported the two arms byte-identical
      because it was scored on an UNPHASED case, where the only body that can tell
      them apart — the per-axis Bloch block calling ``_rotate_field_left`` — is
      compiled out. Moved to a Bloch case under ``subnormal_band``; it now diverges;
    * ``m6`` and ``m10`` replaced an ``if`` header with ``pass`` at the header's own
      indent and failed to IMPORT, which scores as UNARMED rather than as a catch;
    * ``m8`` (the ownership AND, the defect the real board's 2026-08-20 device run
      found) was first aimed at a composite mask where ``own1`` is redundant, then
      at the right block but over three steps. It is a RACE: at 30 steps it is
      caught, at 3 it is not;
    * ``m1``/``m2`` were declared CAUGHT, measured UNCAUGHT, and are now UNREACHED
      with the word-layer sweep that proves the defect real;
    * the separate ORACLE route asked only the plain admission and took the whole
      off-diagonal leg down with it.

    STILL NOT WIRED. A weld licenses a claim, not a dispatch.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT, once it runs: for every configuration
:func:`~meep_gpu.triton_kernels.folded_complex_fused_magnetic_pair.folded_complex_fused_magnetic_pair_coverage`
admits, ONE launch of ``folded_complex_fused_curl_constitutive_B`` leaves the engine
in a state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_B`` / ``fill_symmetry_bc_B`` /
    ``zero_metal_B`` / ``fill_folded_far_ghosts_B`` / ``update_H``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the folded complex curl
    (``folded_complex.plan_folded_complex_pml_curl`` on ``step_B``), the folded
    complex ghost fill (``folded_complex.plan_folded_mirror_ghost_fill_complex`` on
    family ``B``, run as its TWO passes) and the folded complex constitutive
    (``folded_complex.plan_folded_complex_constitutive`` on side ``H``), with
    ``zero_metal_B`` on the array path in both routes because no Triton product
    owns the wall clear,

over every allocated volume: the primaries, the split-field PML auxiliaries and the
constitutive ``f_w`` history. Comparison is on the uint32 view — TWO words per
complex64 cell. ``allclose`` appears nowhere.

===========================================================================
THE ONE THING THIS GATE MEASURES THAT NO SIBLING DOES
===========================================================================

**THE PARITY CHAIN ORDER.** :mod:`.folded_fused_magnetic_pair` composes a
destination's parities into one compile-time ``±1`` and states the result is
independent of the order the array path applies its axes in — true under real
storage, where every fill is an exact sign flip. Under complex storage the parity is
a FULL complex multiply by ``(±1.0, +0.0)`` and complex float multiplication is not
associative, so the order and the grouping both move bytes.

This gate does not take that on the Metal board's word. Two device legs measure it
here, on this kernel:

* ``armed:parity_chain_reordered`` applies the far axes DESCENDING instead of
  ascending. If the bytes still agree, the ordering claim is decorative on this
  platform and the artifact must say so — the leg passes only when it DIVERGES;
* ``armed:parity_chain_folded`` multiplies the two coefficient words together on
  the host and applies the product once. Same rule.

Both are scored on a case with a doubly-unowned corner at MIXED phases, because
with equal phases the composition commutes bit-exactly and the question does not
exist (the Triton fill tranche measured 0/128 diverging words at ``(+1,+1)`` and
``(-1,-1)``, 8/128 at ``(+1,-1)``). ``parity_chain_reachability`` — a no-device leg
— asserts that reachability rather than leaving it to the case name.

===========================================================================
WHAT THE GATE REFUSES TO INFER
===========================================================================

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is counted
  through a proxy that owns the kernel object, every substitution is counted per
  route in the installer, and one ARMED HARNESS MUTATION removes the substitution so
  the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** Every compared array
  must MOVE during the leg; a leg whose magnetic trio is frozen in all three routes
  is armed and must be caught by the moved-state census.
* **A branch no case compiles is unmeasured.** ``branch_reachability`` executes every
  live constexpr guard in the shipped kernel against the constexprs each case really
  compiles and FAILS on any guard no case enters. The sibling folded gate shipped a
  three-axis carry for a day with no leg reaching it; the count is what caught it.
* **Composition DEPTH is declared per leg.** ``ghost_destinations`` is asserted
  against the plan's own ``near``/``far``, so a leg cannot report a depth its
  constexprs forbid.
* **An operand class that cannot contain what it claims to test measures nothing.**
  Every device leg records an operand census — subnormals, negative zeros, zeros —
  and the ``subnormal_band`` and ``zero_lattice`` classes must actually carry them.

===========================================================================
THE SEAM
===========================================================================

``driver.step`` runs five passes (driver.py:3281-3289)::

    step_B -> MAGNETIC SOURCES -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

The magnetic sources are refused by the predicate; the other four are carried
inline. From the 2026-08-21c Triton fusion matrix this cell is FOUR reachable B->H
seam-instances, two of which need the far carry.

POLICY, stamped and unchanged: ``num_warps=1`` (the settled cross-sub-step policy),
``BLOCK=complex_fields.DEFAULT_BLOCK``,
``enable_fp_fusion=kernels.ENABLE_FP_FUSION`` (False). This gate does not tune and
does not time.

THE SUBNORMAL POLICY IS AN ARGUMENT, NOT A DEFAULT DECISION. ``--subnormal-policy``
drives every executor before the first device compile and the artifact records the
stamp. The EXPANSION licence is policy-conditional: a keep-cut probe read by a flush
run licenses nothing, and this gate refuses before its first leg rather than
measuring against a broken comparison. Run it ONCE PER POLICY, each with the probe
artifact cut under that policy, and keep both artifacts.

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_complex_fused_magnetic_pair.py \\
        --no-device --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device
    CUDA_VISIBLE_DEVICES=<UUID of a verified-empty device> \\
    MEEP_GPU_COMPLEX_EXPANSION_PROBE=<probe cut under the same policy> \\
    python -u parity/meep_gpu/probe_triton_folded_complex_fused_magnetic_pair.py \\
        --subnormal-policy keep --out parity/meep_gpu/results/<fresh-dir>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

#: THE PER-CASE SEED IS A SHA-256 DIGEST OF THE CASE NAME, never ``hash()``.
#: ``hash()`` of a str is salted per process (PYTHONHASHSEED), so a record naming a
#: seeded state would not identify the state it measured — two runs of the same
#: case would seed differently and the artifact could not say which one produced a
#: divergence.
SEED_ROOT = "triton_folded_complex_fused_magnetic_pair/2026-08-21"


def case_seed(name: str, value_class: str = "uniform") -> int:
    """A stable 63-bit seed for one (case, value class)."""
    digest = hashlib.sha256(f"{SEED_ROOT}/{name}/{value_class}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


#: The value classes every device case is run under. ``uniform`` provably draws no
#: subnormal and no signed zero, so a gate carrying only it says nothing about the
#: float32 subnormal policy at all; ``subnormal_band`` puts the operands, their
#: products with the absorber coefficients and the whole recurrence in and around
#: the band; ``zero_lattice`` is the +-0 product — every cell one of the four
#: (+-0, +-0) complex pairs — which is where the parity multiply's zero cross terms
#: and the ``* -1.0`` negation spelling decide bytes.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "zero_lattice")

#: The values a policy decides the fate of and that ``uniform(-1, 1)`` provably
#: never draws. Transcribed from ``probe_fused_kernel_bit_identity.SUBNORMAL_NEEDLES``
#: (:589-591) and pinned equal to it by ``test_triton_folded_complex_fused_magnetic_pair``.
SUBNORMAL_NEEDLES = np.array(
    [0.0, -0.0, 1.1754944e-38, -1.1754944e-38, 1.1754942e-38, 1e-45, -1e-45],
    dtype=np.float32)

#: ``(name, dimensions, cell, boundaries, mirrors, k_point, pml, steps)``. A 2-D case
#: is spelled with ``z = 0.0``.
#:
#: THE Z AXIS IS PERIODIC IN EVERY 2-D CASE, NOT METALLIC. These cells are 2-D, so z
#: is translationally invariant: MEEP does not loop over it and it carries no
#: boundary condition. A metallic declaration there is a polarization filter, not a
#: wall, and ``Grid`` refuses it by name (grid.py:842-877).
#:
#: EVERY ROW BELOW IS HERE BECAUSE ``branch_reachability`` DEMANDED IT. The first cut
#: of this table folded Y in almost every case and carried no Bloch phase and no wall
#: off the x axis, and that leg counted SIX shipped guards no case compiled — an
#: unfolded PERIODIC Y, all three phase blocks, and the y and z wall clears. They are
#: not decoration and the leg fails if any of them stops being entered.
CASES: Tuple[Tuple[str, int, Tuple[float, float, float], Any,
                   Tuple[Tuple[str, int], ...], Tuple[float, float, float],
                   Dict[str, Any], int], ...] = (
    # ONE folded axis, MIRROR_METALLIC: the near fill only, no far ghost, no
    # top-plane mask. The simplest admitted shape, and it carries a live x wall.
    ("y_fold_metallic", 2, (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # ONE folded axis, MIRROR_PERIODIC at an EVEN full count: the far carry, the
    # reflect row and the top-plane mask all live. This is the corpus class
    # (``special_kz_2_21_2`` / ``triangular_lattice_oblique``).
    #
    # MEASURED: stored y = 20 and ``_far_reflect_rows`` answers 18 = stored - 2.
    ("y_fold_periodic", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # An ODD full count on the folded axis, at the SAME stored extent. MEASURED:
    # stored y = 20 again and the reflect row is 17 = stored - 3. That pair is what
    # makes ``m5_reflect_row_baked`` a real discrimination rather than a shape
    # change: ``ny - 2`` is 18, exactly right on the row above and a whole cell
    # wrong here.
    ("y_fold_periodic_odd", 2, (3.2, 2.9, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # ODD MIRROR PHASE: the parity words are (-1, +0) near and (+1, +0) far.
    ("y_fold_periodic_odd_phase", 2, (3.2, 3.0, 0.0), "periodic", (("Y", -1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # A FOLD ON X WITH AN UNFOLDED PERIODIC Y — the ghost rule's wrap branch on an
    # axis nothing folds, which every Y-folding case leaves uncompiled.
    ("x_fold_periodic_open_y", 2, (3.0, 3.2, 0.0), "periodic", (("X", 1),),
     (0.0, 0.0, 0.0), {"x": {"high": 5}, "y": 5}, 10),
    # TWO folded axes at MIXED phases — the doubly-unowned corner, and the ONLY
    # shape on which the chain-order question exists at all.
    ("xy_mixed_phase_periodic", 2, (3.0, 3.0, 0.0), "periodic",
     (("X", 1), ("Y", -1)), (0.0, 0.0, 0.0),
     {"x": {"high": 5}, "y": {"high": 5}}, 10),
    # THREE folded PERIODIC axes, mixed phases: SEVEN ghost cells owned by one
    # source lane — the deepest composition this family can emit, and the only shape
    # that compiles the triple-composite blocks.
    ("xyz_mixed_phase_periodic_3d", 3, (2.0, 2.0, 2.0), "periodic",
     (("X", 1), ("Y", -1), ("Z", 1)), (0.0, 0.0, 0.0),
     {"x": {"high": 3}, "y": {"high": 3}, "z": {"high": 3}}, 6),
    # WALLS, one per axis, each beside a fold on a DIFFERENT axis: `_zero_metal`
    # skips a folded axis, so a wall and a fold never share one.
    ("y_fold_periodic_wall_x", 2, (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "periodic", "z": "periodic"}, (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    ("x_fold_periodic_wall_y", 2, (3.0, 3.2, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("X", 1),),
     (0.0, 0.0, 0.0), {"x": {"high": 5}, "y": 5}, 10),
    ("y_fold_periodic_wall_z_3d", 3, (2.0, 2.0, 2.0),
     {"x": "periodic", "y": "periodic", "z": "metallic"}, (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 3, "y": {"high": 3}, "z": 3}, 6),
    # BLOCH PHASES, one per axis, each on an UNFOLDED periodic axis. A folded axis
    # may carry none (`_complex_phase_reasons` refuses it and
    # `driver._require_bloch_is_representable` refuses the configuration outright),
    # so the phase blocks are reachable only like this.
    ("y_fold_periodic_bloch_x", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.19, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    ("x_fold_periodic_bloch_y", 2, (3.0, 3.2, 0.0), "periodic", (("X", 1),),
     (0.0, 0.23, 0.0), {"x": {"high": 5}, "y": 5}, 10),
    ("y_fold_periodic_bloch_z_3d", 3, (2.0, 2.0, 2.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.17), {"x": 3, "y": {"high": 3}, "z": 3}, 6),
    # AN OFF-DIAGONAL ``chi1inv`` ROW. Two of the four corpus rows this product
    # serves carry one (``examples:solve-cw.py`` and
    # ``tests:TestArrayMetadata.test_array_metadata``), and they are admitted through
    # the SECOND arm label — ``folded_complex_offdiag_*`` — over the same kernel and
    # the same plan class. ``update_H`` reads no inverse epsilon, so the row must not
    # change a byte; if that were wrong, this is the case it would show on.
    ("y_fold_periodic_offdiag", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # LONG HORIZON: ``y_fold_periodic``'s shape for SIXTY complete steps. One launch
    # and ten launches are not the same evidence as sixty — a one-ULP drift in the
    # ghost plane compounds, and this is where it shows.
    ("y_fold_periodic_long_horizon", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 60),
    # ONE LAUNCH. The other end of the same axis: exactly one complete step, so a
    # divergence here is attributable to a single launch rather than to an
    # accumulation.
    ("y_fold_periodic_single_launch", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 1),
)

#: One row of :data:`CASES` by name — mutations name the SHAPE they need rather than
#: an index, so inserting a case cannot silently re-score every mutation.
CASES_BY_NAME: Dict[str, Tuple[Any, ...]] = {case[0]: case for case in CASES}

#: Cases that install an OFF-DIAGONAL ``chi1inv`` row, by name. Named here rather
#: than inferred from the case name so a rename cannot silently drop the material.
OFFDIAGONAL_CASES: Tuple[str, ...] = ("y_fold_periodic_offdiag",)

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

#: Which case each mutation is scored on — BY NAME, and every entry is justified by
#: the branch the rewrite touches. TRAP: a mutation scored on a grid that never
#: ENTERS the branch it rewrites reports UNCAUGHT while measuring nothing.
#: ``test_triton_folded_complex_fused_magnetic_pair`` checks each entry's guard
#: against its scored case's constexprs.
MUTATION_CASE: Dict[str, str] = {
    # the ordering pair: two folded PERIODIC axes at DIFFERING phases, or the
    # composition commutes bit-exactly and both are nulls by construction
    "m1_parity_chain_reordered": "xy_mixed_phase_periodic",
    "m2_parity_chain_folded": "xy_mixed_phase_periodic",
    # the triple composite exists only under three folded PERIODIC axes
    "m3_triple_composite_dropped": "xyz_mixed_phase_periodic_3d",
    # a far ghost at all
    "m4_far_parity_is_the_near_one": "y_fold_periodic",
    # an ODD full count, where `ny - 2` is a whole cell wrong
    "m5_reflect_row_baked": "y_fold_periodic_odd",
    # the top-plane mask is emitted only under MIRROR_PERIODIC
    "m6_top_plane_mask_dropped": "y_fold_periodic",
    # any fold; the near fill runs on either mirror code
    "m7_near_source_index_three": "y_fold_metallic",
    # a lane that is the SOURCE of one fill and the DESTINATION of the other
    "m8_ownership_mask_not_ANDed": "xy_mixed_phase_periodic",
    "m9_destination_coefficient_reused": "y_fold_periodic",
    # a live wall
    "m10_wall_clear_dropped": "y_fold_periodic_wall_x",
    "m11_curl_parens_flattened": "y_fold_periodic",
    "m12_prev_read_after_the_store": "y_fold_periodic",
    "m13_far_fill_dropped": "y_fold_periodic",
    "m14_near_fill_dropped": "y_fold_metallic",
    "m15_ghost_B_store_dropped": "y_fold_periodic",
}
DEFAULT_MUTATION_CASE = "y_fold_periodic"

#: Which VALUE CLASS each mutation is scored under. Default ``uniform``.
#:
#: THE TWO ORDERING MUTATIONS ARE SCORED UNDER ``zero_lattice``, and the class was
#: chosen BY MEASUREMENT rather than by preference — twice.
#:
#: :func:`parity_chain_associativity_leg` sweeps the shipped multiply's two arms over
#: signed zeros, the subnormal needles and ordinary values and counts **32 of 2048
#: words moved by a RE-ORDER and 34 by a FOLD**, identically on both arms. Reading
#: WHICH words move settles the class: with ``c_im`` bitwise ``+0.0`` the product
#: ``(c_im * z_im) * -1.0`` is a SIGNED ZERO whose sign follows ``z_im``, and
#: ``fma(c_re, z_re, that)`` canonicalizes ``-0 + +0`` to ``+0``. On every NORMAL
#: operand the multiply is an exact sign flip and the composition commutes to the
#: bit. So the defect lives at signed zeros and nowhere else.
#:
#: ``uniform(-0.25, 0.25)`` provably draws no signed zero. ``subnormal_band`` draws
#: them only through its needles, one cell in sixteen, and the 2026-08-21 run scored
#: both mutations there and measured both UNCAUGHT over three complete steps. That
#: is a statement about the seeding, not about the kernel. ``zero_lattice`` is every
#: cell a ``(±0, ±0)`` pair — the class the defect is reachable from.
MUTATION_VALUE_CLASS: Dict[str, str] = {
    "m1_parity_chain_reordered": "zero_lattice",
    "m2_parity_chain_folded": "zero_lattice",
}
DEFAULT_MUTATION_VALUE_CLASS = "uniform"

#: Per-mutation step budget, where :data:`MUTATION_STEPS` is not enough. Three
#: complete steps is the default because a defect that needs more than that is
#: usually a reachability problem rather than a horizon one — but two classes of
#: defect here really are horizon questions and are given room rather than declared
#: unreachable at the first budget that missed them:
#:
#: * the parity CHAIN ORDER, whose word-layer sweep says the defect is real (32 of
#:   2048 words) but which needs a signed zero to arrive at the ghost SOURCE lane;
#: * the ownership AND, whose violation is a DATA RACE — non-manifestation over a
#:   short horizon is not absence, and a longer one is the only thing a byte gate
#:   can offer against it.
MUTATION_STEPS_FOR: Dict[str, int] = {
    "m1_parity_chain_reordered": 30,
    "m2_parity_chain_folded": 30,
    "m8_ownership_mask_not_ANDed": 30,
}

#: What establishes that a mutation declared ``unreached`` is a REAL defect.
#:
#: ``unreached`` is a third expectation beside ``caught`` and ``null``, and it exists
#: because collapsing the two is how a gate lies in both directions. A ``null`` says
#: THE REWRITE CHANGES NOTHING — the defect is not a defect. ``unreached`` says the
#: defect IS real, names what proves it, and records that THIS gate's device legs do
#: not reach it. Calling the second a null would retire a hazard without carrying
#: it; calling it a catch would be false.
#:
#: THE VERDICT RULE IS NOT SOFTER FOR IT. An ``unreached`` mutation must come back
#: UNCAUGHT — a catch means the declaration is stale and the entry must move to
#: ``caught`` — and it must name its evidence here, or the gate fails.
MUTATION_EVIDENCE: Dict[str, str] = {
    "m1_parity_chain_reordered":
        "parity_chain_associativity_leg measures the defect REAL at the word layer: "
        "re-ordering a two-parity chain moves 32 of 2048 uint32 words, identically "
        "on BOTH expansion arms, and every moved word is at a signed zero. What this "
        "gate could not do is get such a word to a ghost SOURCE lane: 30 complete "
        "steps on xy_mixed_phase_periodic under the +-0 lattice (937 negative-zero "
        "words seeded) left every compared volume byte-identical. The ORDER is held "
        "by construction — parity_chain_leg compares the shipped sequence against "
        "folded_complex_fused_magnetic_pair.parity_chain block by block — and NOT by "
        "a whole-step device leg. A synthetic bare-array leg with planted words at "
        "the source lane is what would close it.",
    "m2_parity_chain_folded":
        "the same sweep measures 34 of 2048 words moved by FOLDING the chain into "
        "one coefficient, on both arms, again all at signed zeros; and the same 30 "
        "complete steps did not reach one. Held by construction in parity_chain_leg "
        "(which asserts no two parity coefficients are ever multiplied together in "
        "the shipped body) rather than by measurement here.",
}

#: THE METAL BOARD REACHED THE SAME POSITION INDEPENDENTLY, and that is worth having
#: beside the two entries above rather than only in a report.
#: ``gate_metal_folded_complex_fused_magnetic_pair.py``'s ``NULL_EDITS`` carries
#: ``parity_chain_order_reversed`` and ``parity_chain_folded_into_one_word`` with the
#: reason "two exact +-1 multiplies commute bitwise on normal operands. Leg
#: parity_chain_order measures the residual at 8 of 128 words on the engineered
#: table; this needle reports 0 differing words over 12 launches on the device
#: fixture" (:1078-1089). Re-run on this Mac's MPS 2026-08-21: VERDICT PASS, 72/72,
#: released, and both needles report diff=0 — same finding, different backend,
#: different ``c_mul`` spelling and a different negation lowering.
#:
#: The two boards' word-layer numbers differ (Metal 8 of 128 on its table, Triton 32
#: of 2048 on this one) because the tables differ; what agrees is the SHAPE of the
#: result — real at the word layer, unreached at the whole-step layer.
CROSS_BOARD_CORROBORATION: str = (
    "gate_metal_folded_complex_fused_magnetic_pair.py:1078-1089 records the same "
    "two rewrites as NULL_EDITS with 0 differing words over 12 device launches; "
    "artifact results/metal_folded_complex_replaces_recheck_2026-08-21/gate.json, "
    "VERDICT PASS, released, re-run 2026-08-21 on Apple MPS.")

#: The driver call sites this product spans, in driver order (driver.py:3281-3289),
#: under the names ``driver.py`` really binds — this is what :func:`install`
#: monkeypatches, so a name that is not a driver attribute is a harness that
#: replaces nothing.
SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

#: THERE ARE TWO VOCABULARIES FOR THIS SEAM AND THEY ARE NOT THE SAME LIST. The
#: driver binds ``fill_symmetry_bc_B``; the RESIDENCY model
#: (``metal_kernels.coverage.RESIDENCY_ORDER``:264-268) spells that slot ``fill_B``,
#: and ``REPLACES`` is declared in the residency spelling because that is what a
#: composer reads. Mapping them here rather than asserting them equal is the honest
#: form: :func:`seam_binding_leg` checks that the map is a bijection and that each
#: driver name is a real ``driver.py`` call, so a rename on either side fails rather
#: than passing by coincidence.
RESIDENCY_NAME: Dict[str, str] = {
    "step_B": "step_B",
    "fill_symmetry_bc_B": "fill_B",
    "zero_metal_B": "zero_metal_B",
    "fill_folded_far_ghosts_B": "fill_folded_far_ghosts_B",
    "update_H": "update_H",
}

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this tuple
#: is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: The volumes THIS SEAM writes. A leg in which none of them moves measured nothing
#: about this launch, whatever else moved.
SEAM_OUTPUTS: Tuple[str, ...] = (
    "Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
    "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: The read-only material volumes, under the names ``inventory`` really finds. A
#: name that matches nothing is not a weaker check, it is two absent ones — the
#: vacuity floor would demand a read-only input move, and ``material_changed`` could
#: never fire. :func:`_assert_material_names_are_real` is what stops a rename from
#: doing that.
MATERIAL = ("eps", "inv_eps")

#: Constexprs no admitted configuration can enter, skipped WHOLE by the branch
#: reachability leg. ``BACKWARD`` is 0 in every builder and its ``else:`` arms are
#: the live half.
DELIBERATELY_UNREACHABLE: Tuple[str, ...] = ("BACKWARD",)

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and material_changed would never fire. Fix the names, do not loosen "
            f"the floor.")


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    # THE POLICY STAMP IS RE-READ, NOT CARRIED. Taken once at install time it
    # records every counter at zero, because nothing had compiled yet — a record
    # that cannot tell an installed policy apart from an inert one.
    if "subnormal_policy" in payload:
        try:
            from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

            payload["subnormal_policy"] = _policy.policy_stamp()
        except Exception as exc:  # noqa: BLE001
            payload["subnormal_policy_reread_error"] = repr(exc)
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        _stamp_provenance(payload)
    except Exception as exc:  # noqa: BLE001 - a missing stamper must not lose the run
        payload["provenance_stamp_error"] = repr(exc)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    """The exact bytes this gate binds itself to."""
    names = (
        "meep_gpu/triton_kernels/folded_complex_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/folded_complex.py",
        "meep_gpu/triton_kernels/complex_fields.py",
        "meep_gpu/triton_kernels/complex_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_folded_complex_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Reading the shipped source — from the FILE, so the merge bar can run this
# ---------------------------------------------------------------------------

_SOURCE_OF = {
    "folded_bloch_pml_curl_step": "folded_complex.py",
    "folded_mirror_ghost_fill_complex": "folded_complex.py",
    "complex_fused_curl_constitutive_B": "complex_fused_magnetic_pair.py",
    "folded_complex_fused_curl_constitutive_B":
        "folded_complex_fused_magnetic_pair.py",
    "_carry_ghost_complex": "folded_complex_fused_magnetic_pair.py",
    "_mul_imag_coefficient_left": "special_kz.py",
}


def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, docstring removed.

    Read from the FILE rather than imported: these legs run on a host with no
    Triton, so the transcription check bites at the merge bar rather than only on a
    device run. The text is EXACT — never ``ast.unparse``d — because what is being
    checked is the PARENTHESISATION, and unparsing re-derives minimal parentheses.
    The docstring IS stripped: every one of these bodies documents the spellings it
    does NOT carry, so a text search over the raw source finds them in prose.
    """
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels", _SOURCE_OF[name])
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    if node is None:
        raise AssertionError(f"{name} is not defined in {path}")
    segment = ast.get_source_segment(text, node)
    if segment is None:  # pragma: no cover - only on a source-less module
        raise AssertionError(f"{name}'s source segment is unavailable")
    lines = segment.splitlines()
    body = node.body[0]
    if (isinstance(body, ast.Expr) and isinstance(body.value, ast.Constant)
            and isinstance(body.value.value, str)):
        start = body.lineno - node.lineno
        end = (body.end_lineno or body.lineno) - node.lineno
        lines = lines[:start] + lines[end + 1:]
    # Normalised to what ``textwrap.dedent(inspect.getsource(fn))`` produces, which
    # is what :func:`shipped_source` hands the mutation table on a CUDA host. Two
    # spellings of the same text would let a rewrite match here and miss there —
    # the exact way a mutation silently disarms.
    margin = node.col_offset
    return "\n".join([lines[0].lstrip()] + [
        line[margin:] if line[:margin].strip() == "" else line.lstrip()
        for line in lines[1:]])


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace one consecutive run of statements, matched on STRIPPED text.

    Matching on stripped lines rather than on an exact substring keeps a rewrite
    armed across a reindentation or a trailing comment. A mutation that silently
    stops matching reports a real defect as uncaught, which this tree has already
    paid for once.
    """
    lines = source.splitlines(keepends=True)
    stripped = [line.split("#", 1)[0].strip() for line in lines]
    target = [item.strip() for item in before]
    for start in range(len(lines) - len(target) + 1):
        if stripped[start:start + len(target)] != target:
            continue
        raw = lines[start]
        indent = raw[:len(raw) - len(raw.lstrip())]
        block = "".join(indent + item.strip() + "\n" for item in after)
        return ("".join(lines[:start]) + block
                + "".join(lines[start + len(target):]), 1)
    return source, 0


def _replace_all(source: str, before: str, after: str) -> Tuple[str, int]:
    """Replace every occurrence of one stripped line."""
    lines = source.splitlines(keepends=True)
    hits = 0
    out: List[str] = []
    for raw in lines:
        if raw.split("#", 1)[0].strip() == before.strip():
            indent = raw[:len(raw) - len(raw.lstrip())]
            out.append(indent + after.strip() + "\n")
            hits += 1
        else:
            out.append(raw)
    return "".join(out), hits


def _statements(text: str) -> List[str]:
    """Executable lines, comments and blanks removed, indentation normalised."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, read off the shipped source
# ---------------------------------------------------------------------------

#: Every arithmetic line the fused body must reproduce VERBATIM from the certified
#: folded complex curl. Not a paraphrase: each of these is asserted present in BOTH
#: texts, and the certified text is read from the file rather than quoted here.
CURL_LINES: Tuple[str, ...] = (
    "t0_re = ((c_y_re - c_re) + (b_re - b_z_re))",
    "t0_im = ((c_y_im - c_im) + (b_im - b_z_im))",
    "t1_re = ((a_z_re - a_re) + (c_re - c_x_re))",
    "t1_im = ((a_z_im - a_im) + (c_im - c_x_im))",
    "t2_re = ((b_x_re - b_re) + (a_re - a_y_re))",
    "t2_im = ((b_x_im - b_im) + (a_im - a_y_im))",
    "curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)",
    "curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)",
    "curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)",
    "q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)",
    "n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)",
    "r_re = (r_re + n0_re) - p0_re",
    "r_im = (r_im + n0_im) - p0_im",
    "v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)",
    "v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)",
    "v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)",
    "ox = si * nyz + j * nz + k",
    "oy = i * nyz + sj * nz + k",
    "oz = i * nyz + j * nz + sk",
    "last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1",
    "at_x, at_y, at_z = i == 0, j == 0, k == 0",
)

#: Every arithmetic line the fused body must reproduce VERBATIM from the certified
#: UNFOLDED complex pair, which is itself ``bloch_constitutive_step``'s ``SCALE = 0``
#: arm with ``src`` bound to a register.
CONSTITUTIVE_LINES: Tuple[str, ...] = (
    "t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)",
)

#: Spellings the kernel must NOT contain. Each is a measured platform fact rather
#: than a style rule.
FORBIDDEN_SPELLINGS: Tuple[Tuple[str, str], ...] = (
    ("PHASE *", "the parity is a full complex multiply under complex storage, not "
                "a scalar sign flip (folded_mirror_ghost_fill_complex's own "
                "measurement: 8/128 engineered words at BOTH parities)"),
    ("= -(", "Triton lowers unary `-x` as `0.0 - x` and canonicalizes signed "
             "zeros; the licensed spelling is `* -1.0` (special_kz.py:305-313)"),
    ("np.complex64", "the parity words are host-rounded ONCE by "
                     "folded_complex.mirror_parity_coefficients and PASSED; a "
                     "kernel that builds them builds them under a different policy"),
)


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line traced back to the shipped source it came from.

    A FUSED PRODUCT IS NOT A RE-DERIVATION. Each region of this kernel must diff
    clean against the certified module it names, and this leg reads BOTH texts off
    the filesystem rather than quoting either.
    """
    ours = _shipped_text("folded_complex_fused_curl_constitutive_B")
    curl = _shipped_text("folded_bloch_pml_curl_step")
    constitutive = _shipped_text("complex_fused_curl_constitutive_B")
    helper = _shipped_text("_carry_ghost_complex")
    findings: List[str] = []

    for line in CURL_LINES:
        if line not in curl:
            findings.append(
                f"the certified folded complex curl no longer carries {line!r}; "
                f"this product transcribes that line and cannot vouch for a copy "
                f"of something that moved")
        if line not in ours:
            findings.append(f"the fused body does not carry the certified curl "
                            f"line {line!r}")
    for line in CONSTITUTIVE_LINES:
        if line not in constitutive:
            findings.append(
                f"the certified complex pair no longer carries {line!r}")
        if line not in ours:
            findings.append(f"the fused body does not carry the certified "
                            f"constitutive line {line!r}")

    # THE ONE EDIT TO THE CURL, in both directions.
    for component in range(3):
        certified_load = (f"e_re = tl.load(f{component} + 2 * idx, mask=live, "
                          f"other=0.0)")
        ours_load = (f"e_re = tl.load(f{component} + 2 * idx, mask=own{component}, "
                     f"other=0.0)")
        if certified_load not in curl:
            findings.append(f"the certified curl's B load moved: {certified_load!r}")
        if ours_load not in ours:
            findings.append(f"the fused body does not mask its B load by ownership: "
                            f"{ours_load!r}")
        if certified_load in ours:
            findings.append(
                f"the fused body still loads B under `live` on component "
                f"{component}: a destination lane would READ a word another lane "
                f"writes, which is the race the ownership move exists to remove")

    # THE GHOST CONSTITUTIVE IS THE OWNED CELL'S SEQUENCE, MOVED.
    order = ("prev_re = tl.load(w + 2 * dst",
             "tl.store(w + 2 * dst, ghost_re",
             "acc_re = tl.load(h + 2 * dst",
             "_mul_coefficient_left(kp_d, ghost_re, ghost_im, EXPANSION)",
             "_mul_coefficient_left(km_d, prev_re, prev_im, EXPANSION)",
             "tl.store(h + 2 * dst, acc_re",
             "tl.store(f + 2 * dst, ghost_re")
    positions = []
    for fragment in order:
        if fragment not in helper:
            findings.append(f"_carry_ghost_complex does not carry {fragment!r}")
            positions.append(-1)
        else:
            positions.append(helper.index(fragment))
    if positions != sorted(positions):
        findings.append(
            "_carry_ghost_complex's statements are out of the array path's order; "
            "`prev` must be read BEFORE the workspace store and the flux store "
            "must come last")

    for spelling, why in FORBIDDEN_SPELLINGS:
        if spelling in ours:
            findings.append(f"the fused body contains {spelling!r}: {why}")

    # ONLY the licensed multiply helpers.
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as module  # noqa: PLC0415

    tree = ast.parse("def _f():\n" + textwrap.indent(ours, "    "))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    multiplies = sorted(name for name in called
                        if name.startswith("_mul") or name.startswith("_rotate"))
    unlicensed = [name for name in multiplies
                  if name not in module.LICENSED_MULTIPLY_HELPERS]
    if unlicensed:
        findings.append(
            f"the fused body calls {unlicensed}, which are not in "
            f"LICENSED_MULTIPLY_HELPERS: a new operand orientation needs its own "
            f"probe pattern before it can be licensed")

    return {
        "leg": "transcription",
        "device": False,
        "curl_lines": len(CURL_LINES),
        "constitutive_lines": len(CONSTITUTIVE_LINES),
        "multiply_helpers_called": multiplies,
        "licensed_multiply_helpers": list(module.LICENSED_MULTIPLY_HELPERS),
        "fused_statements": len(_statements(ours)),
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the parity chain, read off the shipped kernel
# ---------------------------------------------------------------------------

#: Which kernel argument carries which ``(axis, pass)`` coefficient — the SIGNATURE's
#: own naming, asserted against it below.
COEFFICIENT_ARGUMENTS: Dict[str, Tuple[int, str]] = {
    "n0r": (0, "near"), "n1r": (1, "near"), "n2r": (2, "near"),
    "d0r": (0, "far"), "d1r": (1, "far"), "d2r": (2, "far"),
}


def _kernel_node():
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        "folded_complex_fused_magnetic_pair.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "folded_complex_fused_curl_constitutive_B"):
            return node
    raise AssertionError("the shipped kernel is not defined at module scope")


def carry_blocks() -> List[Dict[str, Any]]:
    """Every emitted carry block, PARSED off the shipped body.

    ``chain`` is the ORDERED ``(axis, pass)`` list taken from the
    ``_mul_imag_coefficient_left`` calls' first argument — the kernel's own
    sequence. Nothing here computes a parity or a ghost value: a leg that
    re-implemented the kernel would mirror its defects instead of executing them.
    """
    out: List[Dict[str, Any]] = []
    for statement in _kernel_node().body:
        if not isinstance(statement, ast.If):
            continue
        calls = [node for node in ast.walk(statement)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", "") == "_carry_ghost_complex"]
        if not calls:
            continue
        if len(calls) != 1:
            raise AssertionError(
                f"the block `if {ast.unparse(statement.test)}:` emits "
                f"{len(calls)} carries; this parser assumes one per block")
        chain: List[Tuple[int, str]] = []
        for inner in statement.body:
            if not isinstance(inner, ast.Assign):
                continue
            value = inner.value
            if (isinstance(value, ast.Call)
                    and getattr(value.func, "id", "") == "_mul_imag_coefficient_left"):
                chain.append(COEFFICIENT_ARGUMENTS[value.args[0].id])
        call = calls[0]
        out.append({
            "guard": ast.unparse(statement.test),
            "target": call.args[0].id,
            "destination": ast.unparse(call.args[3]),
            "coefficient": ast.unparse(call.args[6]),
            "mask": ast.unparse(call.args[8]),
            "chain": tuple(chain),
        })
    return out


def parity_chain_leg() -> Dict[str, Any]:
    """The chain each block applies, against the driver order the product declares.

    THE ORDER IS THE DRIVER'S. ``fill_symmetry_bc_B`` (:3285) runs to completion
    before ``fill_folded_far_ghosts_B`` (:3287), and the far pass applies its axes
    ASCENDING (``stepping._fill_folded_far_ghosts``:1518, :1528). This leg compares
    the SHIPPED sequence against
    :func:`~meep_gpu.triton_kernels.folded_complex_fused_magnetic_pair.parity_chain`,
    which is a statement about which coefficient goes where and in what order — not
    a second copy of the arithmetic.
    """
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as module  # noqa: PLC0415

    findings: List[str] = []
    parameters = [argument.arg for argument in _kernel_node().args.args]
    for name in COEFFICIENT_ARGUMENTS:
        if name not in parameters:
            findings.append(
                f"the kernel takes no argument {name!r}; this leg's coefficient "
                f"table names the signature and would otherwise be vacuous")

    blocks = carry_blocks()
    rows: List[Dict[str, Any]] = []
    matched: set = set()
    for component in range(3):
        near = (component,)
        far = tuple(axis for axis in range(3) if axis != component)
        target = f"f{component}"
        for subset, carries_near in module.carried_destinations(near, far):
            expected = module.parity_chain(near, subset, carries_near)
            hits = [block for block in blocks
                    if block["target"] == target and block["chain"] == expected]
            rows.append({
                "component": component, "far_axes_at_top": list(subset),
                "carries_near": carries_near,
                "expected_chain": [list(entry) for entry in expected],
                "emitted_blocks": len(hits),
                "guard": hits[0]["guard"] if hits else None,
                "coefficient": hits[0]["coefficient"] if hits else None,
                "mask": hits[0]["mask"] if hits else None,
            })
            if len(hits) != 1:
                findings.append(
                    f"component {component}, far {subset}, near {carries_near}: "
                    f"{len(hits)} emitted blocks carry the driver-ordered chain "
                    f"{expected}")
                continue
            matched.add(id(hits[0]))
            block = hits[0]
            # THE COEFFICIENT INDEX MOVES ONLY FOR THE NEAR HALF.
            wanted = f"kp_d{component}" if carries_near else f"kp_{component}"
            if block["coefficient"] != wanted:
                findings.append(
                    f"component {component}, far {subset}, near {carries_near}: "
                    f"the destination takes {block['coefficient']!r}, expected "
                    f"{wanted!r} — update_H indexes component m on axis m and the "
                    f"NEAR fill images along that same axis")
            # EVERY CARRY MASK IS ANDED WITH THE OWNERSHIP MASK.
            if not block["mask"].startswith(f"own{component} &"):
                findings.append(
                    f"component {component}: the carry mask {block['mask']!r} is "
                    f"not ANDed with own{component}; a lane that is the source of "
                    f"one fill and the destination of the other would write a "
                    f"ghost from a `v` its own mask zeroed")
    for block in blocks:
        if id(block) not in matched:
            findings.append(
                f"the kernel emits a carry the enumeration does not name: "
                f"`if {block['guard']}:` -> {block['destination']}")

    # THE FOLDED SPELLING IS ABSENT.
    ours = _shipped_text("folded_complex_fused_curl_constitutive_B")
    for left in COEFFICIENT_ARGUMENTS:
        for right in COEFFICIENT_ARGUMENTS:
            if f"{left} * {right}" in ours:
                findings.append(
                    f"the kernel multiplies two parity coefficients together "
                    f"({left} * {right}); folding a chain into one word moves "
                    f"bytes under complex storage")
    return {
        "leg": "parity_chain",
        "device": False,
        "emitted_blocks": len(blocks),
        "destinations_enumerated": len(rows),
        "rows": rows,
        "findings": findings,
        "passed": not findings,
    }


def mutation_vocabulary_leg() -> Dict[str, Any]:
    """Every declared expectation is one of the three, and ``unreached`` is EARNED.

    The third value exists so a real defect this gate cannot reach is recorded as
    such rather than as a null — but only if the record says what proves the defect
    real. An ``unreached`` entry with no evidence is a null with a longer word, and
    this leg refuses it at the merge bar rather than on a device.
    """
    allowed = {"caught", "null", "unreached"}
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    for name, _why, expectation, _rewrite in mutation_table():
        rows.append({"mutation": name, "expectation": expectation,
                     "evidence": MUTATION_EVIDENCE.get(name)})
        if expectation not in allowed:
            findings.append(f"{name}: expectation {expectation!r} is not one of "
                            f"{sorted(allowed)}")
        if expectation == "unreached" and not MUTATION_EVIDENCE.get(name):
            findings.append(
                f"{name} is declared 'unreached' and MUTATION_EVIDENCE names "
                f"nothing that establishes the defect is real; an unreached "
                f"mutation with no evidence is a null with a longer word")
        if expectation != "unreached" and MUTATION_EVIDENCE.get(name):
            findings.append(
                f"{name} carries MUTATION_EVIDENCE but is not declared 'unreached'; "
                f"the evidence would never be read")
    return {"leg": "mutation_vocabulary", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


def parity_chain_reachability_leg() -> Dict[str, Any]:
    """The chain-ORDER question must EXIST on the case it is scored on.

    With equal mirror phases on two folded axes the composition commutes
    bit-exactly (the Triton fill tranche measured 0/128 diverging words at
    ``(+1, +1)`` and at ``(-1, -1)``, 8/128 at ``(+1, -1)``), so a two-axis case
    carrying ONE declared phase makes both ordering mutations a null BY
    CONSTRUCTION rather than by measurement. This leg asserts that the cases the
    ordering mutations are scored on carry a DOUBLY-UNOWNED corner at DIFFERING
    phases — the reachability, not the result.
    """
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    for name in ("m1_parity_chain_reordered", "m2_parity_chain_folded"):
        case = CASES_BY_NAME[MUTATION_CASE[name]]
        phases = sorted({phase for _axis, phase in case[4]})
        folded = len(case[4])
        periodic = case[3] == "periodic" or (
            isinstance(case[3], dict)
            and all(case[3].get(axis.lower()) in (None, "periodic")
                    for axis, _phase in case[4]))
        row = {"mutation": name, "case": case[0], "folded_axes": folded,
               "distinct_phases": phases, "folded_axes_are_periodic": periodic}
        rows.append(row)
        if folded < 2:
            findings.append(
                f"{name} is scored on {case[0]}, which folds {folded} axis: a "
                f"chain shorter than two entries has no order")
        if len(phases) < 2:
            findings.append(
                f"{name} is scored on {case[0]}, whose folded axes all declare "
                f"phase {phases}: the composition commutes bit-exactly at equal "
                f"phases and the mutation would be a null by construction")
        if not periodic:
            findings.append(
                f"{name} is scored on {case[0]}, whose folded axes are not all "
                f"PERIODIC: without a far ghost there is no second parity to order")
    return {"leg": "parity_chain_reachability", "device": False,
            "rows": rows, "findings": findings, "passed": not findings}



# ---------------------------------------------------------------------------
# NO-DEVICE LEG — is the parity chain's ORDER observable AT ALL on this backend?
# ---------------------------------------------------------------------------

#: The word table the associativity measurement sweeps. Signed zeros, the subnormal
#: needles, the smallest normals and a few ordinary values — the classes a policy or
#: a rounding decides the fate of, and the ones ``uniform(-1, 1)`` provably misses.
PARITY_WORDS: Tuple[float, ...] = (
    0.0, -0.0, 1e-45, -1e-45, 1.1754942e-38, -1.1754942e-38,
    1.1754944e-38, -1.1754944e-38, 5.9604645e-08, -5.9604645e-08,
    1.0, -1.0, 0.1, -0.1, 3.4028235e38, -3.4028235e38,
)


def _fma32(a: float, b: float, c: float) -> np.float32:
    """``fma(a, b, c)`` on float32 inputs, computed exactly.

    A float32 product is EXACT in float64 (24 + 24 = 48 mantissa bits against 53),
    and the addend is exact there too, so one float64 add followed by one cast to
    float32 rounds exactly once — which is fused-multiply-add's defining property.
    Not an approximation of the hardware instruction: the same value, by
    construction.
    """
    return np.float32(np.float64(np.float32(a)) * np.float64(np.float32(b))
                      + np.float64(np.float32(c)))


def _parity_multiply(c_re, c_im, z_re, z_im, arm: int):
    """``special_kz._mul_imag_coefficient_left``'s two arms, transcribed.

    FMA_V1 (arm 1)::

        out_re = tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)
        out_im = tl.math.fma(c_re, z_im, c_im * z_re)

    NAIVE (arm 0)::

        out_re = (c_re * z_re) - (c_im * z_im)
        out_im = (c_re * z_im) + (c_im * z_re)

    The lines above are the shipped ones (special_kz.py:328-334) and the leg asserts
    they are still there before it trusts this transcription.
    """
    c_re = np.float32(c_re); c_im = np.float32(c_im)
    z_re = np.float32(z_re); z_im = np.float32(z_im)
    if arm == 1:
        out_re = _fma32(c_re, z_re, np.float32(np.float32(c_im * z_im) * np.float32(-1.0)))
        out_im = _fma32(c_re, z_im, np.float32(c_im * z_re))
    else:
        out_re = np.float32(np.float32(c_re * z_re) - np.float32(c_im * z_im))
        out_im = np.float32(np.float32(c_re * z_im) + np.float32(c_im * z_re))
    return out_re, out_im


def _words(pair) -> Tuple[int, int]:
    return (int(np.float32(pair[0]).view(np.uint32)),
            int(np.float32(pair[1]).view(np.uint32)))


def parity_chain_associativity_leg() -> Dict[str, Any]:
    """IS THE ORDERING OBSERVABLE? Measured, on the shipped multiply's own arms.

    THE MEASUREMENT THIS LEG EXISTS FOR. Two armed device mutations re-order and
    fold this kernel's parity chain, and on 2026-08-21 BOTH came back UNCAUGHT —
    byte-identical over three complete steps on a mixed-phase two-fold grid seeded
    in the subnormal band. A null is only honest if the question behind it is
    settled rather than hidden, so it is settled here, at the word layer, on the
    arithmetic itself rather than on the whole-step state.

    THE ARITHMETIC. The parity coefficients are
    :func:`.folded_complex.mirror_parity_coefficients`' output: ``c_re`` exactly
    ``±1.0`` and ``c_im`` BITWISE ``+0.0`` at both mirror phases and for the near and
    far coefficient alike. Under both arms the multiply then reduces to a sign flip
    plus a signed-zero addend, which is exact on every finite operand — so composing
    two of them commutes and folds to the bit. This leg sweeps
    :data:`PARITY_WORDS` x the four ``(±1, ±1)`` coefficient pairs and COUNTS.

    WHAT IT DOES NOT LICENSE. A zero count here says the ORDER is unobservable for
    THIS coefficient class on THIS backend — it does not say the kernel may apply
    the passes in any order, and the kernel does not: it transcribes the driver's,
    and :func:`parity_chain_leg` pins that by construction. Nor does it transfer to
    Metal, whose own re-ordering measurement moved 8 of 128 words with a different
    ``c_mul`` spelling and a different negation lowering.

    THE LEG FAILS IF THE COUNT IS NONZERO WHILE THE TABLE DECLARES THOSE MUTATIONS
    NULL — a declared null must agree with the arithmetic, or one of the two is
    wrong.
    """
    shipped = _shipped_text("_mul_imag_coefficient_left")
    findings: List[str] = []
    for line in ("out_re = tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)",
                 "out_im = tl.math.fma(c_re, z_im, c_im * z_re)",
                 "out_re = (c_re * z_re) - (c_im * z_im)",
                 "out_im = (c_re * z_im) + (c_im * z_re)"):
        if line not in shipped:
            findings.append(
                f"special_kz._mul_imag_coefficient_left no longer carries {line!r}; "
                f"this leg transcribes that body and cannot vouch for a copy of "
                f"something that moved")

    from meep_gpu.triton_kernels.folded_complex import (  # noqa: PLC0415
        mirror_parity_coefficients,
    )

    rows: List[Dict[str, Any]] = []
    for arm in (0, 1):
        reorder = fold = total = 0
        for phase_a in (1, -1):
            for phase_b in (1, -1):
                # The two FAR coefficients of a two-folded-axis grid at those phases.
                c_a = mirror_parity_coefficients(phase_a)[1]
                c_b = mirror_parity_coefficients(phase_b)[1]
                folded = _parity_multiply(c_a[0], c_a[1], c_b[0], c_b[1], arm)
                for z_re in PARITY_WORDS:
                    for z_im in PARITY_WORDS:
                        total += 2
                        ascending = _parity_multiply(
                            c_b[0], c_b[1],
                            *_parity_multiply(c_a[0], c_a[1], z_re, z_im, arm),
                            arm)
                        descending = _parity_multiply(
                            c_a[0], c_a[1],
                            *_parity_multiply(c_b[0], c_b[1], z_re, z_im, arm),
                            arm)
                        one_shot = _parity_multiply(
                            folded[0], folded[1], z_re, z_im, arm)
                        reorder += sum(
                            1 for x, y in zip(_words(ascending), _words(descending))
                            if x != y)
                        fold += sum(
                            1 for x, y in zip(_words(ascending), _words(one_shot))
                            if x != y)
        rows.append({"arm": arm, "words_compared": total,
                     "reorder_differing_words": reorder,
                     "fold_differing_words": fold})

    table = {name: expectation for name, _why, expectation, _r in mutation_table()}
    for name, key in (("m1_parity_chain_reordered", "reorder_differing_words"),
                      ("m2_parity_chain_folded", "fold_differing_words")):
        observed = max(row[key] for row in rows)
        declared = table.get(name)
        if declared in ("null",) and observed:
            findings.append(
                f"{name} is declared NULL but the arithmetic moves {observed} of "
                f"{rows[0]['words_compared']} words: the declaration and the "
                f"measurement disagree")
        if declared == "caught" and not observed:
            findings.append(
                f"{name} is declared CAUGHT but the arithmetic moves NO word on "
                f"either arm: it cannot be caught at the whole-step layer either, "
                f"and a mutation that cannot be seen is not a passing mutation")
    return {
        "leg": "parity_chain_associativity",
        "device": False,
        "coefficient_class": "c_re is exactly +-1.0 and c_im is bitwise +0.0 "
                             "(folded_complex.mirror_parity_coefficients)",
        "value_words": len(PARITY_WORDS),
        "rows": rows,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG — branch reachability
# ---------------------------------------------------------------------------

def _live_constexpr_guards(text: str) -> List[Tuple[int, str]]:
    """Every ``if <constexpr>:`` in the kernel whose body some case could reach.

    The ``if BACKWARD:`` suites are skipped WHOLE — header and body — because
    ``BACKWARD`` is 0 in every admitted configuration and their nested guards are
    dead with them. Their ``else:`` arms are not skipped: they are the live half.
    """
    out: List[Tuple[int, str]] = []
    skip_below: Optional[int] = None
    for index, raw in enumerate(text.splitlines()):
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if skip_below is not None:
            if stripped and indent <= skip_below:
                skip_below = None
            else:
                continue
        if any(stripped.startswith(f"if {name}")
               for name in DELIBERATELY_UNREACHABLE):
            skip_below = indent
            continue
        if stripped.startswith("if ") and stripped.endswith(":"):
            out.append((index + 1, stripped[3:-1].strip()))
    return out


def ghost_destinations(near: Sequence[Any], far: Sequence[Any]) -> Tuple[int, int, int]:
    """Ghost cells ONE source lane owns, per B component, from the plan's booleans.

    Component ``m`` is a NEAR destination on axis ``m`` alone and a FAR destination
    on the two axes that are NOT ``m``, and the two fills COMPOSE — every nonempty
    subset of the destination planes is itself a destination. So the count is
    ``(1 + near_m) * 2 ** (far axes other than m) - 1``: 1 under a single metallic
    fold, 3 under one periodic fold or two folds, and 7 under three folded periodic
    axes, which is the largest this family can produce.

    Counted, not re-implemented: it decides how many ``_carry_ghost_complex`` calls a
    lane makes, not what any of them computes.
    """
    out: List[int] = []
    for component in range(3):
        others = sum(1 for axis in range(3)
                     if axis != component and bool(far[axis]))
        out.append((1 + int(bool(near[component]))) * (2 ** others) - 1)
    return (out[0], out[1], out[2])


def build_grid(case, prefer_gpu: bool = True):
    """The grid and absorber one CASES row names, with no field seeding.

    Split out of :func:`build_driver` so the LAPTOP can build the same geometry the
    device leg compiles against — the branch reachability leg reads its constexprs
    from here, and a second spelling of the geometry would be a mirrored evaluator
    rather than a check.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    (_name, dimensions, cell, boundaries, mirrors, k_point, pml_spec,
     _steps) = case
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35,  # non-power-of-two, deliberate
        boundaries=boundaries, k_point=k_point,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    driver.setup_pml(dict(pml_spec))
    return driver


def case_constexprs(case) -> Dict[str, Any]:
    """The constexprs one CASES row compiles the kernel with, read on NumPy.

    Every value comes from the function the PLAN calls — ``folded_axis_kinds``,
    ``zero_metal_axes``, ``bloch_phase_table`` — and ``near``/``far`` are derived
    from ``bc`` exactly as the plan derives them, so this cannot disagree with what
    a device leg compiles.
    """
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as module  # noqa: PLC0415
    from meep_gpu.triton_kernels.complex_fields import (  # noqa: PLC0415
        _phase_arguments, bloch_phase_table,
    )
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels.folded_complex import (  # noqa: PLC0415
        CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
        folded_axis_kinds,
    )

    driver = build_grid(case, prefer_gpu=False)
    try:
        grid = driver.fields.grid
        codes, reasons = folded_axis_kinds(grid, driver.pml)
        if codes is None:
            raise AssertionError(f"{case[0]}: folded_axis_kinds refused: {reasons}")
        codes = tuple(int(value) for value in codes)
        walls = tuple(bool(value) for value in zero_metal_axes(grid))
        phased, _values = _phase_arguments(
            bloch_phase_table(grid, _boundary_kinds(grid, driver.pml)),
            backward=bool(module.BACKWARD))
        near = tuple(code in module.MIRROR_CODES for code in codes)
        far = tuple(code == CODE_MIRROR_PERIODIC for code in codes)
        return {
            "PERIODIC": int(CODE_PERIODIC),
            "MIRROR_METALLIC": int(CODE_MIRROR_METALLIC),
            "MIRROR_PERIODIC": int(CODE_MIRROR_PERIODIC),
            "BACKWARD": int(module.BACKWARD),
            "BCX": codes[0], "BCY": codes[1], "BCZ": codes[2],
            "NEAR_X": near[0], "NEAR_Y": near[1], "NEAR_Z": near[2],
            "FAR_X": far[0], "FAR_Y": far[1], "FAR_Z": far[2],
            "ZM_X": walls[0], "ZM_Y": walls[1], "ZM_Z": walls[2],
            "PHX": int(phased[0]), "PHY": int(phased[1]), "PHZ": int(phased[2]),
            "shape": tuple(int(value) for value in grid.shape),
            "parity": module.parity_coefficient_words(grid),
            "ghost_destinations": ghost_destinations(near, far),
        }
    finally:
        driver.close()


def branch_reachability_leg() -> Dict[str, Any]:
    """SHIPPED CODE NO LEG EXECUTES, counted rather than argued.

    THE FINDING THIS LEG EXISTS FOR is the sibling folded gate's: on 2026-08-20 that
    kernel shipped three-axis blocks and NO DEVICE LEG EVER EXECUTED THEM, because
    every ``mirrors`` value in the artifact carried at most two folded axes. Byte
    identity over ten steps on ten cases says nothing about a branch none of the ten
    compiled. A case list is not a coverage argument until something counts it.

    THE EVALUATION IS THE ENGINE'S, NOT A MODEL OF IT: the constexprs come from
    :func:`case_constexprs`, which reads them off a real grid, and the guard text
    comes from the shipped source. Neither side is a re-implementation.
    """
    text = _shipped_text("folded_complex_fused_curl_constitutive_B")
    guards = _live_constexpr_guards(text)
    environments = [(case[0], case_constexprs(case)) for case in CASES]
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for line, condition in guards:
        entered = sorted(
            name for name, environment in environments
            if eval(condition, {"__builtins__": {}}, dict(environment)))  # noqa: S307
        rows.append({"line": line, "condition": condition,
                     "entered_by": entered, "cases": len(entered)})
        if not entered:
            findings.append(
                f"line {line}: `if {condition}:` is shipped and NO case in this "
                f"table compiles it — the leg would launch a kernel that does not "
                f"contain those lines, and every byte row would be silent about them")
    deepest = max(max(environment["ghost_destinations"])
                  for _name, environment in environments)
    if deepest != 7:
        findings.append(
            f"the deepest composition any case reaches is {deepest}, not 7: the "
            f"triple-composite blocks are shipped and unexecuted")
    return {
        "leg": "branch_reachability",
        "device": False,
        "guards": len(guards),
        "deliberately_unreachable": list(DELIBERATELY_UNREACHABLE),
        "cases": [{"case": name,
                   "shape": list(environment["shape"]),
                   "near": [environment[f"NEAR_{a}"] for a in "XYZ"],
                   "far": [environment[f"FAR_{a}"] for a in "XYZ"],
                   "walls": [environment[f"ZM_{a}"] for a in "XYZ"],
                   "parity": [list(pair) for pair in environment["parity"]],
                   "ghost_destinations": list(environment["ghost_destinations"])}
                  for name, environment in environments],
        "deepest_ghost_destinations": deepest,
        "rows": rows,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the predicate battery
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every seam clause, in both directions, on NumPy grids."""
    import numpy  # noqa: PLC0415
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as module  # noqa: PLC0415

    def probe_record(patterns=None):
        from meep_gpu.test_triton_complex_fields import stamp_probe_record  # noqa: PLC0415

        names = module.PRODUCT_PROBE_PATTERNS if patterns is None else patterns
        return stamp_probe_record(
            {"backend": "cupy", "patterns": {name: "FMA_V1" for name in names}})

    def build(boundaries="periodic", mirrors=(("Y", 1),), complex_storage=True,
              thickness=None, extent=2.0, offdiagonal=False):
        size = [1.6, 1.6, 0.0]
        for axis, _phase in mirrors:
            size["XYZ".index(axis)] = extent
        declared = boundaries
        if isinstance(boundaries, str) and boundaries != "periodic":
            declared = {axis.lower(): boundaries for axis, _p in mirrors}
        grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=2,
                    courant=0.35, boundaries=declared,
                    symmetry=tuple(Mirror(a, p) for a, p in mirrors), xp=numpy)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_pml_storage()
        if offdiagonal:
            epsilon = numpy.full(grid.shape, 2.0, dtype=numpy.float32)
            inverse = numpy.full(grid.shape, 0.5, dtype=numpy.float32)
            partner = numpy.full(grid.shape, 0.04, dtype=numpy.float32)
            fields.set_epsilon_volumes(
                {name: epsilon for name in ("Ex", "Ey", "Ez")},
                {name: inverse for name in ("Ex", "Ey", "Ez")},
                chi1inv_offdiagonal={"Ex": {"Ey": partner},
                                     "Ey": {"Ex": partner}})
        if thickness is None:
            thickness = []
            folded = {"XYZ".index(a) for a, _p in mirrors}
            for index in range(3):
                if grid.shape[index] < 6:
                    thickness.append((0, 0))
                elif index in folded:
                    thickness.append((0, 2))
                else:
                    thickness.append((2, 2))
            thickness = tuple(thickness)
        return fields, PML(grid=grid, thickness=thickness)

    class _Magnetic:
        field_type = "B"

    class _MagneticIndexed:
        """A magnetic source that publishes the index the injection writes —
        what ``deposit_repair.save`` needs, and what the corpus row's own
        VolumeSource provides."""

        field_type = "B"

        def __init__(self, index=(2, 2, 0)) -> None:
            self._point_ix, self._point_iy, self._point_iz = index

    class _Electric:
        field_type = "D"

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []

    def ask(name, expect_covered, needle=None, **kwargs):
        # THE RUN POLICY IS DECLARED, NOT LEFT IMPLICIT. Every record built here is
        # stamped 'keep', and the complex families' policy clause FAILS CLOSED when
        # the policy in force can be neither read nor declared — which is what a
        # laptop with no CuPy is. Declaring it states the premise this leg's
        # questions are about; it does not soften any clause.
        from meep_gpu.expansion_refusal import declaring_run_policy  # noqa: PLC0415

        sources = kwargs.pop("sources", ())
        probe = kwargs.pop("probe", None)
        fields, pml = build(**kwargs)
        with declaring_run_policy("keep"):
            verdict = module.folded_complex_fused_magnetic_pair_coverage(
                fields, pml, sources,
                probe=probe if probe is not None else probe_record())
        residual = [r for r in verdict.reasons if "array module" not in r]
        covered = not residual
        rows.append({"case": name, "expected_covered": expect_covered,
                     "covered_modulo_backend": covered,
                     "reasons": residual[:4]})
        if covered != expect_covered:
            findings.append(f"{name}: covered_modulo_backend={covered}, expected "
                            f"{expect_covered} ({residual[:2]})")
        if needle is not None and not any(needle in r for r in verdict.reasons):
            findings.append(f"{name}: no reason contains {needle!r}: {residual[:3]}")

    ask("folded_periodic_admitted", True)
    # THE SECOND ARM LABEL. Two of the four corpus rows this product serves carry an
    # off-diagonal chi1inv row, and `folded_complex_composition_curl_coverage`
    # refuses one BY NAME. They are admitted through `folded_complex_offdiag_*`,
    # over the same kernel and the same plan class, because update_H reads no
    # inverse epsilon.
    ask("offdiagonal_row_admitted", True, offdiagonal=True)
    ask("folded_metallic_admitted", True, boundaries="metallic")
    ask("two_folds_admitted", True, mirrors=(("X", 1), ("Y", -1)))
    ask("electric_source_admitted", True, sources=(_Electric(),))
    # THE 2026-08-31 SPLIT, made in the same edit as the CARRIES_DEPOSIT_REPAIR
    # flip on this module: a magnetic deposit the repair can carry is ADMITTED
    # (it is what discharged the board's one clause-refusal on this family,
    # tests:TestHoleyWvgBands.test_fields_at_kx), and one that cannot publish
    # the index it writes is refused BY NAME — the same two rows every carrying
    # sibling's predicate leg holds.
    ask("magnetic_source_with_an_index_admitted", True,
        sources=(_MagneticIndexed(),))
    ask("magnetic_source_without_an_index_refused", False,
        needle="does not publish the index", sources=(_Magnetic(),))
    ask("undeclared_sources_refused", False, needle="was not declared",
        sources=None)
    ask("real_storage_refused", False, needle="storage is real float32",
        complex_storage=False)
    ask("unfolded_refused", False, needle="no mirror plane is active", mirrors=())
    ask("no_pml_refused", False, needle="no active PML", thickness=0.0)
    ask("base_probe_only_refused", False,
        needle=folded_complex.PARITY_PROBE_PATTERN,
        probe=probe_record(folded_complex.PROBE_PATTERNS))
    ask("no_probe_refused", False, needle="expansion probe artifact", probe={})

    return {"leg": "predicate", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — the seam binding
# ---------------------------------------------------------------------------

def seam_binding_leg() -> Dict[str, Any]:
    """``REPLACES`` against the driver's own call sites, and against the body.

    TRAP, paid for on this track: a module carried ``fill_folded_far_ghosts_B``
    while its ``REPLACES`` tuple said it did not, and a passing test pinned the
    stale tuple. So the tuple is checked against BOTH the driver source and the
    kernel body here, not against a copy of itself.
    """
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as module  # noqa: PLC0415

    findings: List[str] = []
    driver_text = open(os.path.join(API_ROOT, "meep_gpu", "driver.py"),
                       encoding="utf-8").read()
    for name in SEAM_PASSES:
        if f"{name}(" not in driver_text:
            findings.append(f"driver.py does not call {name}; this gate's seam "
                            f"model names a pass the driver does not run")
    expected = tuple(RESIDENCY_NAME[name] for name in SEAM_PASSES)
    if tuple(module.REPLACES) != expected:
        findings.append(
            f"REPLACES {tuple(module.REPLACES)} is not this gate's seam in the "
            f"RESIDENCY spelling {expected}")
    if len(set(RESIDENCY_NAME.values())) != len(SEAM_PASSES):
        findings.append(
            f"the driver-name to residency-name map is not a bijection: "
            f"{RESIDENCY_NAME}")
    try:
        from meep_gpu.metal_kernels.coverage import RESIDENCY_ORDER  # noqa: PLC0415

        unknown = [name for name in RESIDENCY_NAME.values()
                   if name not in RESIDENCY_ORDER]
        if unknown:
            findings.append(
                f"{unknown} are not in metal_kernels.coverage.RESIDENCY_ORDER; the "
                f"residency spelling this gate maps to is not the package's")
    except Exception as exc:  # noqa: BLE001 - an unreadable residency model is a finding
        findings.append(f"RESIDENCY_ORDER is unreadable: {exc!r}")

    body = _shipped_text("folded_complex_fused_curl_constitutive_B")
    # WHAT THE BODY ACTUALLY CARRIES, keyed in the residency spelling ``REPLACES``
    # uses. TRAP, paid for on this track: a module carried the far fill while its
    # REPLACES tuple said it did not, and a passing test pinned the stale tuple.
    carried = {
        "step_B": "curl0_re, curl0_im = _mul_coefficient_left(dtdx" in body,
        "fill_B": all(f"if NEAR_{axis}:" in body for axis in "XYZ"),
        "zero_metal_B": all(f"if ZM_{axis}:" in body for axis in "XYZ"),
        "fill_folded_far_ghosts_B": all(f"if FAR_{axis}:" in body for axis in "XYZ"),
        "update_H": "_carry_ghost_complex" in body
                    and "_mul_coefficient_left(kp_0, src_re" in body,
    }
    for name, present in carried.items():
        if name in module.REPLACES and not present:
            findings.append(
                f"REPLACES names {name} but the kernel body carries no evidence "
                f"of it")
        if name not in module.REPLACES and present:
            findings.append(
                f"the kernel body carries {name} but REPLACES does not name it; "
                f"a declaration that disagrees with the kernel is the failure this "
                f"check exists for")
    return {"leg": "seam_binding", "device": False,
            "replaces": list(module.REPLACES), "carried": carried,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 6 — the expansion licence
# ---------------------------------------------------------------------------

def expansion_licence_leg() -> Dict[str, Any]:
    """The EXTENDED pattern set is required, and each way a record can fail it."""
    from meep_gpu.test_triton_complex_fields import stamp_probe_record  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as module  # noqa: PLC0415

    findings: List[str] = []
    rows: List[Dict[str, Any]] = []

    def record(patterns, value="FMA_V1"):
        return stamp_probe_record(
            {"backend": "cupy", "patterns": {name: value for name in patterns}})

    full = record(module.PRODUCT_PROBE_PATTERNS)
    base = record(folded_complex.PROBE_PATTERNS)
    for name, artifact, expect in (
        ("extended_record_licenses", full, True),
        ("base_only_record_refuses", base, False),
        ("neither_arm_refuses",
         record(module.PRODUCT_PROBE_PATTERNS, "NEITHER"), False),
        ("wrong_backend_refuses",
         stamp_probe_record({"backend": "numpy", "patterns": {
             name: "FMA_V1" for name in module.PRODUCT_PROBE_PATTERNS}}), False),
    ):
        expansion = folded_complex.parity_expansion_from_probe(artifact)
        rows.append({"record": name, "expansion": expansion,
                     "licensed": expansion is not None})
        if (expansion is not None) != expect:
            findings.append(f"{name}: expansion={expansion}, expected "
                            f"licensed={expect}")
    if set(module.PRODUCT_PROBE_PATTERNS) != set(folded_complex.PARITY_PROBE_PATTERNS):
        findings.append(
            "the product's pattern set is not folded_complex.PARITY_PROBE_PATTERNS; "
            "the carry calls the fill's own multiply and must bind the fill's set")
    return {"leg": "expansion_licence", "device": False,
            "pattern_set": list(module.PRODUCT_PROBE_PATTERNS),
            "rows": rows, "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device state
# ---------------------------------------------------------------------------

def value_class_hosts(names: Sequence[str], shape, rng,
                      value_class: str) -> Dict[str, np.ndarray]:
    """Host complex64 arrays for one value class.

    ``uniform`` provably draws no subnormal and no signed zero, so a gate carrying
    only it is silent about the float32 subnormal policy. ``subnormal_band`` spans
    1e-45 to 1e-30 with one cell in sixteen forced to a needle. ``zero_lattice`` is
    the +-0 product: every cell one of the four ``(+-0, +-0)`` complex pairs, which
    is where the parity multiply's zero cross terms and the ``* -1.0`` negation
    spelling decide bytes.

    THE AUXILIARIES START NONZERO in the first two classes: a zero ``fu``/``f_w``
    makes ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds,
    which would hide a mis-indexed coefficient until step two.
    """
    host: Dict[str, np.ndarray] = {}
    if value_class == "uniform":
        for name in names:
            host[name] = (rng.uniform(-0.25, 0.25, size=shape)
                          + 1j * rng.uniform(-0.25, 0.25, size=shape)
                          ).astype(np.complex64)
        return host
    if value_class == "subnormal_band":
        for name in names:
            parts = []
            for _half in range(2):
                exponents = rng.uniform(-45.0, -30.0, size=shape)
                signs = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
                plane = (signs * np.power(10.0, exponents)).astype(np.float32)
                picks = rng.integers(0, 16, size=shape) == 0
                choice = SUBNORMAL_NEEDLES[
                    rng.integers(0, SUBNORMAL_NEEDLES.size, size=shape)]
                parts.append(np.where(picks, choice, plane).astype(np.float32))
            host[name] = (parts[0] + 1j * parts[1]).astype(np.complex64)
        return host
    if value_class == "zero_lattice":
        zeros = np.array([0.0, -0.0], dtype=np.float32)
        for name in names:
            real = zeros[rng.integers(0, 2, size=shape)]
            imaginary = zeros[rng.integers(0, 2, size=shape)]
            host[name] = (real + 1j * imaginary).astype(np.complex64)
        return host
    raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")


def operand_census(arrays: Dict[str, np.ndarray]) -> Dict[str, int]:
    """How many subnormals, signed zeros and zeros a leg's operands ACTUALLY hold.

    A LEG WHOSE OPERANDS CANNOT CONTAIN THE CLASS IT CLAIMS TO TEST MEASURES
    NOTHING, and reports a pass while doing it. Classified off the BITS, not by
    comparison: ``x < FLT_MIN`` is true for zero too, and ``x == -0.0`` is true for
    ``+0.0``.
    """
    flat = np.concatenate([np.asarray(a, dtype=np.complex64).ravel().view(np.float32)
                           for a in arrays.values()])
    raw = flat.view(np.uint32)
    exponent = (raw >> 23) & 0xFF
    mantissa = raw & 0x7FFFFF
    return {
        "words": int(flat.size),
        "subnormals": int(((exponent == 0) & (mantissa != 0)).sum()),
        "negative_zeros": int((raw == 0x80000000).sum()),
        "zeros": int((raw == 0).sum()),
    }


#: The floor each value class must clear, per leg, or the leg measured nothing of
#: what its name claims.
CLASS_FLOOR: Dict[str, Dict[str, int]] = {
    "uniform": {},
    "subnormal_band": {"subnormals": 1},
    "zero_lattice": {"negative_zeros": 1, "zeros": 1},
}

SEED_FIELDS = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
SEED_AUXILIARIES = ("f_w_Hx", "f_w_Hy", "f_w_Hz")


def build_driver(cp, case, value_class: str, electric: bool,
                 magnetic: bool = False):
    """One folded complex driver, seeded identically for every route.

    ``magnetic=True`` (2026-08-31, with the ``CARRIES_DEPOSIT_REPAIR`` flip)
    adds a REAL magnetic source IN THIS SEAM — the shape the corpus row
    ``tests:TestHoleyWvgBands.test_fields_at_kx`` declares — off the mirror
    plane on every folded axis, exactly like the electric one.
    """
    driver = build_grid(case)
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    if case[0] in OFFDIAGONAL_CASES:
        # MEEP's ``chi1inv[c][d]`` for ``d`` not the row component's own direction.
        # The values are inverse-tensor entries, free to be negative; they are read
        # by ``stepping.update_E`` alone and must not reach this launch.
        partner = np.ascontiguousarray(
            (0.04 * np.cos(index * np.float32(0.019))).astype(np.float32))
        # `set_epsilon_components` takes the per-component EFFECTIVE permittivity and
        # derives each diagonal chi1inv entry itself (driver.py:1184-1251); the
        # off-diagonal row is passed beside it as MEEP's chi1inv[c][d].
        driver.set_epsilon_components(
            {name: cp.asarray(epsilon) for name in ("Ex", "Ey", "Ez")},
            chi1inv_offdiagonal={"Ex": {"Ey": cp.asarray(partner)},
                                 "Ey": {"Ex": cp.asarray(partner)}})
    else:
        driver.set_epsilon(cp.asarray(epsilon))
    if electric:
        # An ELECTRIC source is admitted: the driver injects it in the D/E seam,
        # not this one. Carrying one is what stops the source clause from being
        # tested only in its refusing direction.
        #
        # OFF THE MIRROR PLANE ON EVERY FOLDED AXIS. A point Ez at the origin sits
        # ON the plane, and on an ODD fold that is refused with a reason rather than
        # stepped: Ez has parity -1 there and no extent along the folded axis, so
        # the parity condition constrains the profile against itself and admits only
        # zero (from_meep.py:2686-2698).
        center = [0.0, 0.0, 0.0]
        for axis_name, _phase in case[4]:
            axis = "XYZ".index(axis_name.upper())
            center[axis] = 0.25 * (case[2][axis] / 2.0)
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    if magnetic:
        # THE DEPOSIT THIS PRODUCT'S FLAG EXISTS FOR, injected BETWEEN step_B
        # and update_H (driver.py:3283-3284), off the mirror plane on every
        # folded axis.
        center = [0.0, 0.0, 0.0]
        for axis_name, _phase in case[4]:
            axis = "XYZ".index(axis_name.upper())
            center[axis] = 0.25 * (case[2][axis] / 2.0)
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    rng = np.random.default_rng(case_seed(case[0], value_class))
    hosts = value_class_hosts(SEED_FIELDS, shape, rng, value_class)
    for name, values in hosts.items():
        driver.set_field(name, cp.asarray(np.ascontiguousarray(values)))
    auxiliary = value_class_hosts(SEED_AUXILIARIES, shape, rng, value_class)
    for name, values in auxiliary.items():
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(np.ascontiguousarray(values))
            hosts[name] = values
    return driver, hosts


def inventory(driver) -> Dict[str, Any]:
    """Every device volume the step can touch, found by scanning rather than listing."""
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        if name.startswith("__"):
            continue
        if (getattr(value, "shape", None) == shape
                and getattr(value, "dtype", None) is not None):
            found[name] = value
    _assert_material_names_are_real(found)
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not complete")
    return found


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def snapshot(cp, driver) -> Dict[str, np.ndarray]:
    return {name: words(cp, array) for name, array in inventory(driver).items()}


def first_divergence(left: Dict[str, np.ndarray],
                     right: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            return {"array": name, "reason": "shape",
                    "left": list(a.shape), "right": list(b.shape)}
        if not np.array_equal(a, b):
            where = int(np.flatnonzero(a != b)[0])
            return {"array": name, "index": where,
                    "left_word": int(a[where]), "right_word": int(b[where]),
                    "differing_words": int(np.count_nonzero(a != b))}
    only = sorted(set(left) ^ set(right))
    return {"array": "<inventory>", "reason": "asymmetric", "names": only} if only else None


def moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> List[str]:
    return [name for name in sorted(set(before) & set(after))
            if not np.array_equal(before[name], after[name])]


class CountingKernel:
    """Owns the JIT kernel and counts every launch through the plan's ``run``."""

    __slots__ = ("jit", "calls", "grids")

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.calls = 0
        self.grids: List[Any] = []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*args, **kwargs):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            return launcher(*args, **kwargs)

        return run


def kernel_ptx(jit: Any) -> List[str]:
    out: List[str] = []
    for per_device in (getattr(jit, "cache", None) or {}).values():
        for compiled in per_device.values():
            asm = getattr(compiled, "asm", None)
            if asm and "ptx" in asm:
                out.append(asm["ptx"])
    return out


class Route:
    """The plan bundle installed for one driver, keyed by driver call site."""

    def __init__(self, plans: Dict[str, Any]) -> None:
        self.plans = plans
        #: Which of ``folded_complex``'s two arm labels the oracle's plans came
        #: from, recorded so a leg's row says which admission it measured.
        self.admission: Optional[str] = None


class _Absorbed:
    """The sentinel left where a driver pass was absorbed by the fused launch."""

    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self) -> None:
        return None


class _TwoPassFill:
    """The certified complex ghost fill, run as the driver runs it.

    ONE LAUNCH PER FOLDED AXIS PER PASS, in X, Y, Z order, and the two PASSES
    separately: ``fill_symmetry_bc_B`` is every folded axis's NEAR plane and
    ``fill_folded_far_ghosts_B`` every folded axis's FAR one, with ``zero_metal_B``
    between them. Fusing them per axis is a DIFFERENT ORDER and a measured different
    answer under complex storage (``folded_mirror_ghost_fill_complex``'s docstring:
    5 words of ``By`` on the gate's engineered rows), which is exactly why the
    oracle route must not take the shortcut the fused kernel is being measured
    against.
    """

    __slots__ = ("plan", "phase")

    def __init__(self, plan: Any, phase: str) -> None:
        if phase not in ("near", "far"):
            raise ValueError(f"phase must be 'near' or 'far', got {phase!r}")
        self.plan = plan
        self.phase = phase

    def run(self) -> None:
        # `run_near` / `run_far`, NEVER `run` — the plan's own `run` is near-then-far
        # in one call, and the driver puts `zero_metal_B` BETWEEN them
        # (driver.py:3285-3287). Calling `run` here would fuse across the wall clear
        # and give the oracle a different order than the array path it is supposed
        # to reproduce.
        if self.phase == "near":
            self.plan.run_near()
        else:
            self.plan.run_far()


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the five seam passes for the fields objects named in ``routes``.

    Counting is PER ROUTE. A single global counter would mix the reference driver's
    legitimate array-path calls with a fallback in the fused route, which is
    precisely the event this instrument exists to see.
    """
    originals = {name: getattr(driver_module, name) for name in SEAM_PASSES}

    def bump(role: str, kind: str, name: str) -> None:
        key = f"{role}/{kind}:{name}"
        counter[key] = counter.get(key, 0) + 1

    def replacement(name):
        def wrapper(fields, *args):
            role = roles.get(id(fields), "unknown")
            plan = next((candidate for owner, candidate in routes
                         if fields is owner), None)
            if plan is None or name not in plan.plans:
                bump(role, "array_path", name)
                return originals[name](fields, *args)
            bump(role, "substituted", name)
            plan.plans[name].run()
            return None
        return wrapper

    for name in SEAM_PASSES:
        setattr(driver_module, name, replacement(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def separate_route(driver, probe):
    """The three separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    # EITHER MATCHED ADMISSION, and the oracle takes the same one the product does.
    # `folded_complex.py` registers a second family name over the SAME kernel and the
    # SAME plan class for a run carrying an off-diagonal chi1inv row; a route that
    # only asked the plain builders would refuse the two corpus rows this product
    # exists to serve and would take the whole leg down with it — which is exactly
    # what it did on 2026-08-21 before this branch was written.
    curl = folded_complex.plan_folded_complex_pml_curl(
        driver.fields, driver.pml, "step_B", probe=probe)
    constitutive = folded_complex.plan_folded_complex_constitutive(
        driver.fields, driver.pml, "H", probe=probe)
    admission = "plain"
    if curl is None or constitutive is None:
        offdiag_curl = folded_complex.plan_folded_complex_offdiag_pml_curl(
            driver.fields, driver.pml, "step_B", probe=probe)
        offdiag_constitutive = folded_complex.plan_folded_complex_offdiag_constitutive(
            driver.fields, driver.pml, "H", probe=probe)
        if offdiag_curl is not None and offdiag_constitutive is not None:
            curl, constitutive = offdiag_curl, offdiag_constitutive
            admission = "offdiag"
    fill = folded_complex.plan_folded_mirror_ghost_fill_complex(
        driver.fields, "B", probe=probe)
    missing = [name for name, plan in
               (("folded complex curl", curl), ("folded complex ghost fill", fill),
                ("folded complex constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    # `zero_metal_B` stays on the ARRAY PATH here: no Triton product owns the wall
    # clear. It is counted, so a difference in whether it ran is a counter event
    # rather than an invisible correction.
    route = Route({"step_B": curl,
                   "fill_symmetry_bc_B": _TwoPassFill(fill, "near"),
                   "fill_folded_far_ghosts_B": _TwoPassFill(fill, "far"),
                   "update_H": constitutive})
    route.admission = admission          # recorded per leg, not inferred
    return route


def fused_route(plan, driver=None, sources=(), bracket: bool = True):
    """The fused plan in its slots, BRACKETED when the seam carries a deposit.

    ADDED 2026-08-31 with the ``CARRIES_DEPOSIT_REPAIR`` flip on this module:
    the bracket is the SHIPPED ``deposit_repair.LeadingRepairPlan`` /
    ``TrailingRepairPlan`` pair — the exact objects ``launch._install_fused_pair``
    puts in these two slots — whose fold rules (``repair_cells``) save the
    deposit's IMAGES through the same fill map this kernel's forward carry
    implements. ``bracket=False`` is the NULL CONTROL and MUST diverge: it is
    the same launch with the repair removed, which is the configuration the
    flag exists to forbid. Returns ``(route, leading)``; ``leading`` is ``None``
    when the seam carries no deposit.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "B")
    if not seam or not bracket:
        return Route({name: (plan if name == "step_B" else _Absorbed(name, plan))
                      for name in SEAM_PASSES}), None
    leading = deposit_repair.LeadingRepairPlan(plan, driver.fields, driver.pml,
                                               seam, "B")
    trailing = deposit_repair.TrailingRepairPlan("update_H", leading,
                                                 driver.fields, driver.pml)
    plans = {name: _Absorbed(name, plan) for name in SEAM_PASSES}
    plans["step_B"] = leading
    plans["update_H"] = trailing
    return Route(plans), leading


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, case, steps: int, product, probe,
            value_class: str = "uniform", mutant: Any = None,
            install_fused: bool = True, freeze_magnetic: bool = False,
            electric: bool = True, expansion: Optional[int] = None,
            parity_override: Optional[Sequence[Tuple[float, float]]] = None,
            magnetic: bool = False, bracket: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence.

    ``magnetic=True`` puts a REAL magnetic deposit in this seam and the fused
    route runs it through the SHIPPED repair bracket (``bracket=False`` is the
    null control, which must diverge).
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference, seeded = build_driver(cp, case, value_class, electric,
                                     magnetic=magnetic)
    separate, _ = build_driver(cp, case, value_class, electric,
                               magnetic=magnetic)
    fused, _ = build_driver(cp, case, value_class, electric, magnetic=magnetic)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_complex_fused_curl_constitutive_B_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "case": case[0], "steps_budget": steps,
        "value_class": value_class, "seed": case_seed(case[0], value_class),
        "shape": list(reference.shape), "boundaries": str(case[3]),
        "mirrors": [list(entry) for entry in case[4]],
        "k_point": list(case[5]), "pml": str(case[6]),
        "electric_source": bool(electric),
        "magnetic_source": bool(magnetic), "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "magnetic_frozen": bool(freeze_magnetic),
        "expansion_override": expansion,
        "parity_override": (None if parity_override is None
                            else [list(pair) for pair in parity_override]),
        "operand_census": operand_census(seeded),
        "offdiagonal_chi1inv": bool(
            getattr(reference.fields, "has_offdiagonal_epsilon", False)),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_folded_complex_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        if plan is None:
            verdict = product.folded_complex_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources), probe=probe)
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if expansion is not None:
            # THE ARM, OVERRIDDEN. Not a source rewrite: the constexpr IS the
            # licence, so flipping it here measures whether the licence is
            # load-bearing rather than decorative.
            plan.expansion = int(expansion)
        if parity_override is not None:
            plan.parity = tuple((float(a), float(b)) for a, b in parity_override)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["licensed_expansion"] = plan.expansion
        row["near"] = [bool(value) for value in plan.near]
        row["far"] = [bool(value) for value in plan.far]
        row["reflect_rows"] = list(plan.reflect)
        row["parity"] = [list(pair) for pair in plan.parity]
        row["ghost_destinations_per_source_lane"] = list(
            ghost_destinations(plan.near, plan.far))
        row["triple_composite_executes"] = bool(
            max(ghost_destinations(plan.near, plan.far)) == 7)
        if install_fused:
            route, leading = fused_route(plan, fused, tuple(fused._sources),
                                         bracket=bracket)
        else:
            route = Route({})
        separate_plans = separate_route(separate, probe)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}
        row["oracle_admission"] = separate_plans.admission

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_magnetic:
            # ARMED: every route's magnetic seam is inert. All three then agree
            # trivially and only the moved-state census can refuse it.
            frozen = {key: _Absorbed(key, None) for key in SEAM_PASSES}
            route.plans.update(frozen)
            separate_plans.plans.update(frozen)
            routes.append((reference.fields, Route(dict(frozen))))
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, fused)
        reference_opening = snapshot(cp, reference)
        row["per_step"] = []
        started = time.time()
        for step in range(1, steps + 1):
            before = snapshot(cp, fused)
            reference.step()
            separate.step()
            fused.step()
            cp.cuda.runtime.deviceSynchronize()
            after_reference = snapshot(cp, reference)
            after_separate = snapshot(cp, separate)
            after_fused = snapshot(cp, fused)
            versus_array = first_divergence(after_fused, after_reference)
            versus_separate = first_divergence(after_fused, after_separate)
            control = first_divergence(after_separate, after_reference)
            step_moved = moved(before, after_fused)
            row["per_step"].append({
                "step": step,
                "fused_vs_array": versus_array,
                "fused_vs_separate_certified_products": versus_separate,
                "oracle_control_vs_array": control,
                "arrays_moved": len(step_moved),
                "fused_launches": kernel.calls,
                "deposit_repairs": (leading.repairs if leading is not None
                                    else None),
            })
            divergence = versus_array or versus_separate
            log(f"  {name} step {step}/{steps} identical={divergence is None} "
                f"control={control is None} moved={len(step_moved)} "
                f"launches={kernel.calls} "
                f"repairs={leading.repairs if leading is not None else '-'} "
                f"({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, fused)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        # THE FLOOR IS RELATIVE TO THE ARRAY PATH, and that is a 2026-08-21
        # correction rather than a softening. An absolute "every array moved" floor
        # fails a CORRECT kernel for properties of the configuration: on a 2-D grid
        # seeded to the +-0 lattice and driven by an Ez source the whole TE
        # polarization (Ex, Ey, Hz) is exactly zero forever, and over ONE complete
        # step `fu_B` does not move. Neither is evidence about this launch. What IS
        # evidence is an array the ARRAY PATH moves and the fused route does not —
        # that is a pass this weld swallowed — so the reference driver's own census
        # is taken and the two are compared. The positive floor (REQUIRED_TO_MOVE)
        # keeps the trivial-agreement case refused.
        reference_final = snapshot(cp, reference)
        reference_moved = moved(reference_opening, reference_final)
        row["reference_never_moved"] = sorted(
            set(reference_final) - set(reference_moved) - set(material))
        row["inert_here_but_moving_on_the_array_path"] = sorted(
            (set(final) - set(ever_moved) - set(material))
            & set(reference_moved))
        row["seam_outputs_moved"] = sorted(set(ever_moved) & set(SEAM_OUTPUTS))
        row["reference_seam_outputs_moved"] = sorted(
            set(reference_moved) & set(SEAM_OUTPUTS))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(reference_final))
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["deposit_repairs"] = leading.repairs if leading is not None else None
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        return row
    finally:
        undo()
        for target in (reference, separate, fused):
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_launches_at_least: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False,
               require_ghost_destinations: Optional[int] = None
               ) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    floor = CLASS_FLOOR.get(str(row.get("value_class")), {})
    census = row.get("operand_census") or {}
    for key, minimum in floor.items():
        if int(census.get(key, 0)) < minimum:
            failures.append(
                f"VACUOUS OPERAND CLASS: value_class={row.get('value_class')!r} "
                f"seeded {census.get(key, 0)} {key}, below the floor {minimum}; "
                f"this leg cannot have measured what its class name claims")
    if require_ghost_destinations is not None:
        observed = row.get("ghost_destinations_per_source_lane") or []
        if max(observed or [0]) != int(require_ghost_destinations):
            failures.append(
                f"this leg was declared to reach composition depth "
                f"{require_ghost_destinations} — the ghost cells one source lane "
                f"owns for its most-composed component — and reached "
                f"{max(observed or [0])} ({observed})")
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if not require_identical and row.get("first_divergence") is None:
        failures.append(
            "this leg REQUIRES divergence and found none: the control it exists "
            "to be is inert, and the thing it was meant to prove load-bearing "
            "is not")
    if require_repairs and not row.get("deposit_repairs"):
        failures.append(
            "the carry leg repaired NO deposit cell: the bracket ran but the "
            "seam carried nothing, so this leg measured the quiet case under a "
            "carry name")
    if (require_launches_at_least is not None
            and (row.get("fused_kernel_launches") or 0) < require_launches_at_least):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} "
            f"times, fewer than the {require_launches_at_least} this leg needs")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg "
            f"could not have measured the fused launch: {row['control_divergence']}")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"expected {require_launches}: the fused path is not what executed")
    if require_moved:
        swallowed = row.get("inert_here_but_moving_on_the_array_path") or []
        if swallowed:
            failures.append(
                f"the fused route left {swallowed} INERT while the array path "
                f"moves them: a pass this weld was supposed to carry did not run")
        reference_movers = set(row.get("reference_seam_outputs_moved") or ())
        if not reference_movers:
            failures.append(
                "VACUOUS: the ARRAY PATH itself moves nothing this seam writes on "
                "this configuration, so agreement with it is agreement about "
                "nothing")
        movers = set(row.get("seam_outputs_moved") or ())
        if movers != reference_movers:
            failures.append(
                f"the fused route's seam-output movement {sorted(movers)} is not "
                f"the array path's {sorted(reference_movers)}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    fell_back = {key: value for key, value in (row.get("launches") or {}).items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: {fell_back}")
    return (not failures), failures


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def shipped_source(product) -> str:
    """The kernel AND the device function it calls, as one mutable text.

    ``_carry_ghost_complex`` is included for two reasons, both failures a mutation
    harness cannot see from its own result. A name the mutant module does not define
    makes it fail to IMPORT, which is recorded as UNARMED rather than as a caught
    defect; and a device function left OUT of the mutated text is a piece of the
    product no mutation can reach at all.
    """
    return (textwrap.dedent(inspect.getsource(product._carry_ghost_complex.fn))
            + "\n\n"
            + textwrap.dedent(inspect.getsource(
                product.folded_complex_fused_curl_constitutive_B.fn)))


def normalise_kernel_text(text: str) -> List[str]:
    """One normal form for the two readers of the shipped kernel.

    ``inspect.getsource`` includes the ``@triton.jit`` decorator line and the
    docstring; :func:`_shipped_text` drops both (the decorator because
    ``ast.get_source_segment`` starts at ``def``, the docstring deliberately —
    every one of these bodies documents the spellings it does NOT carry). Neither
    difference is arithmetic, so the comparison is made on the statements that
    remain: decorators removed, docstrings removed, comments and blanks removed.
    """
    lines = [line for line in text.splitlines()
             if not line.strip().startswith("@")]
    tree = ast.parse(textwrap.dedent("\n".join(lines)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return _statements(ast.unparse(tree))


def shipped_source_from_file() -> str:
    """The same text :func:`shipped_source` builds, read from the FILE.

    THE MERGE BAR HAS NO TRITON, so ``inspect.getsource(product._carry_ghost_complex
    .fn)`` cannot run there — the whole ``if triton is not None:`` block is skipped
    and the names do not exist. Every check that the mutation table is ARMED, and
    that each mutation is scored on a case which ENTERS the branch it rewrites,
    would then be device-only: exactly the two failures this track has already paid
    for. This route lets both run on a laptop.

    :func:`main` asserts the two texts EQUAL on a device before it arms anything, so
    this is a second reader of one source rather than a second source.
    """
    return (_shipped_text("_carry_ghost_complex") + "\n\n"
            + _shipped_text("folded_complex_fused_curl_constitutive_B"))


def enclosing_constexpr_guards(source: str, line_index: int) -> List[str]:
    """The ``if <constexpr>:`` guards a line of the shipped text sits inside.

    Read by indentation off the same text the rewrites match against, so a mutation
    can be checked against the constexprs its scored case really compiles. TRAP 1:
    a rewrite can hit real lines under a guard the scored case does not enter and
    report UNCAUGHT while measuring nothing.
    """
    lines = source.splitlines()
    guards: List[str] = []
    indent = len(lines[line_index]) - len(lines[line_index].lstrip())
    for index in range(line_index - 1, -1, -1):
        raw = lines[index]
        if not raw.strip():
            continue
        here = len(raw) - len(raw.lstrip())
        if here >= indent:
            continue
        indent = here
        stripped = raw.strip()
        if stripped.startswith("if ") and stripped.endswith(":"):
            guards.append(stripped[3:-1].strip())
        elif stripped.startswith("else:") or stripped.startswith("elif "):
            guards.append("<else>")
    return list(reversed(guards))


def rewritten_lines(source: str, mutated: str) -> List[int]:
    """The indices of ``source``'s lines a rewrite removed or changed."""
    before = source.splitlines()
    after = mutated.splitlines()
    import difflib  # noqa: PLC0415

    out: List[int] = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
            None, before, after).get_opcodes():
        if tag in ("replace", "delete"):
            out.extend(range(i1, i2))
    return out


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest.

    THE CONSTEXPR CODES COME FROM ``folded_complex``'S OWN CONSTANTS, not from
    literals: a renumbering there would give the mutant a different boundary
    vocabulary than the kernel it stands for, with no error at all. The four
    multiply helpers are IMPORTED rather than copied, for the reason the shipped
    module imports them — a mutant carrying its own copy would no longer be testing
    the shipped multiply.
    """
    from meep_gpu.triton_kernels.folded_complex import (  # noqa: PLC0415
        CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
    )

    header = ("import triton\n"
              "import triton.language as tl\n"
              "from meep_gpu.triton_kernels.complex_fields import (\n"
              "    _rotate_field_left, _mul_field_left, _mul_coefficient_left)\n"
              "from meep_gpu.triton_kernels.special_kz import (\n"
              "    _mul_imag_coefficient_left)\n"
              f"PERIODIC = tl.constexpr({CODE_PERIODIC})\n"
              f"MIRROR_METALLIC = tl.constexpr({CODE_MIRROR_METALLIC})\n"
              f"MIRROR_PERIODIC = tl.constexpr({CODE_MIRROR_PERIODIC})\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_folded_complex_pair.py", delete=False,
        encoding="utf-8")
    handle.write(header + source.replace(
        "folded_complex_fused_curl_constitutive_B", kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_complex_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was
    MEASURED, never what was hoped."""

    def m1_parity_chain_reordered(source: str) -> Tuple[str, int]:
        """THE ORDER, reversed on one two-parity chain.

        The far axes ascend in the driver (``_fill_folded_far_ghosts``:1518); this
        applies them descending. THE BLOCK IS COMPONENT 2's ``FAR_X and FAR_Y``, and
        the choice is not free: this mutation is scored on a fold-X + fold-Y grid,
        so a rewrite landing in the ``FAR_X and FAR_Z`` block would be DEAD CODE
        there and would report UNCAUGHT while measuring nothing. (That is exactly
        what the first cut of this table did, and
        ``test_every_mutation_is_scored_on_a_case_that_ENTERS_the_branch_it_rewrites``
        is what caught it.)
        """
        return _rewrite_block(
            source,
            ["gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, v2_re, v2_im,",
             "EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, gc_re, gc_im,",
             "EXPANSION)"],
            ["gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, v2_re, v2_im, EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, gc_re, gc_im, EXPANSION)"])

    def m2_parity_chain_folded(source: str) -> Tuple[str, int]:
        """THE GROUPING, folded: two complex multiplies collapsed into one.

        The real-storage twin's spelling — one compile-time sign — applied here.
        Under complex storage the product of the coefficient words is not the
        composition of the two multiplies.
        """
        return _rewrite_block(
            source,
            ["gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v0_re, v0_im,",
             "EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, gc_re, gc_im,",
             "EXPANSION)"],
            ["gc_re, gc_im = _mul_imag_coefficient_left(",
             "    n0r * d1r - n0i * d1i, n0r * d1i + n0i * d1r,",
             "    v0_re, v0_im, EXPANSION)"])

    def m3_triple_composite_dropped(source: str) -> Tuple[str, int]:
        """The deepest destination never written. Needs three folded PERIODIC axes."""
        return _rewrite_block(
            source,
            ["_carry_ghost_complex(f0, w0, h0, idx + dn_x + df_y + df_z,",
             "gc_re, gc_im, kp_d0, km_d0,",
             "own0 & near_i & far_j & far_k, EXPANSION)"],
            ["pass"])

    def m4_far_parity_is_the_near_one(source: str) -> Tuple[str, int]:
        """``+phase`` where the far fill takes ``-phase``.

        ``mirror_parity(c, axis, phase) == phase * (1 - 2*iyee)`` (fields.py:117):
        the far fill touches shift-1 components alone and takes the NEGATED phase.
        """
        return _replace_all(
            source,
            "gy_re, gy_im = _mul_imag_coefficient_left(d1r, d1i, v0_re, v0_im,",
            "gy_re, gy_im = _mul_imag_coefficient_left(n1r, n1i, v0_re, v0_im,")

    def m5_reflect_row_baked(source: str) -> Tuple[str, int]:
        """``n - 2`` instead of the runtime row.

        ``_far_reflect_rows`` is ``n_full - stored + 2``, which is ``stored - 2`` at
        an EVEN full count and ``stored - 3`` at an ODD one. Baking reflects about
        the window top instead of about the second mirror. Scored on the ODD case;
        a null on the even one is arithmetic, not a gap.
        """
        return _rewrite_block(
            source, ["far_j = live & (j == ry)"], ["far_j = live & (j == ny - 2)"])

    def m6_top_plane_mask_dropped(source: str) -> Tuple[str, int]:
        """The folded curl's DELTA 3, removed on one axis.

        The mask arrived WITH the far carry: a product that took the fill and left
        the block out would leave that plane unmasked.
        """
        # BODY ONLY. Replacing the `if` header too writes `pass` at the header's
        # indent, which is an IndentationError and an UNARMED mutation — measured
        # 2026-08-21, when this rewrite and m10 both failed to import.
        return _rewrite_block(
            source,
            ["curl0_re = tl.where(last_y, 0.0, curl0_re)",
             "curl0_im = tl.where(last_y, 0.0, curl0_im)"],
            ["pass"])

    def m7_near_source_index_three(source: str) -> Tuple[str, int]:
        """``MIRROR_SOURCE_INDEX`` off by one: the near fill images stored cell 2."""
        return _rewrite_block(
            source, ["near_j = live & (j == 2)"], ["near_j = live & (j == 3)"])

    def m8_ownership_mask_not_ANDed(source: str) -> Tuple[str, int]:
        """The 2026-08-20 device defect, re-planted.

        With two fills a lane can be the SOURCE of one and the DESTINATION of the
        other; dropping ``own?`` from the carry mask lets it write a composite cell
        from a ``v`` its own mask zeroed.
        """
        # THE BLOCK IS COMPONENT 1's FAR-ONLY DESTINATION ON X, and the choice is
        # measured rather than free. `own1` is `live & (j != 0) & (i != nx - 1)`; on
        # the composite destination (`near_j & far_i`, i.e. j == 2) the `j != 0`
        # clause is TRUE anyway, so dropping `own1` there changes no lane and the
        # mutation is a value-level NO-OP — measured UNCAUGHT on 2026-08-21. The
        # lane the real defect was about is the one at `(i == rx, j == 0)`: the far
        # source for By on x, and itself the NEAR fill's destination on y. That lane
        # is excluded by `own1` and by nothing else in this mask.
        return _replace_all(
            source,
            "kp_1, km_1, own1 & far_i, EXPANSION)",
            "kp_1, km_1, far_i, EXPANSION)")

    def m9_destination_coefficient_reused(source: str) -> Tuple[str, int]:
        """The near destination takes the SOURCE lane's coefficient pair.

        ``update_H`` indexes component m on axis m and the near fill images along
        that same axis, so the destination's pair is at index 0 and is NOT the
        source's. On a folded axis whose absorber sits on the HIGH face alone the
        two entries are the same word to the bit, so this is predicted a NULL —
        recorded WITH that reason rather than as a catch.
        """
        return _replace_all(
            source, "kp_d1, km_d1, own1 & near_j, EXPANSION)",
            "kp_1, km_1, own1 & near_j, EXPANSION)")

    def m10_wall_clear_dropped(source: str) -> Tuple[str, int]:
        """``zero_metal_B`` not carried: MEEP's ``step_boundaries(B_stuff)``, undone."""
        # BODY ONLY — see m6.
        return _rewrite_block(
            source,
            ["v0_re = tl.where(at_x, 0.0, v0_re)",
             "v0_im = tl.where(at_x, 0.0, v0_im)"],
            ["pass"])

    def m11_curl_parens_flattened(source: str) -> Tuple[str, int]:
        """``stepping._curl_from_operands``' grouping, re-associated."""
        return _rewrite_block(
            source, ["t0_re = ((c_y_re - c_re) + (b_re - b_z_re))"],
            ["t0_re = (c_y_re - c_re + b_re) - b_z_re"])

    def m12_prev_read_after_the_store(source: str) -> Tuple[str, int]:
        """The one ordering the constitutive cannot survive being wrong about."""
        return _rewrite_block(
            source,
            ["prev_re = tl.load(w + 2 * dst, mask=mask, other=0.0)",
             "prev_im = tl.load(w + 2 * dst + 1, mask=mask, other=0.0)",
             "tl.store(w + 2 * dst, ghost_re, mask=mask)",
             "tl.store(w + 2 * dst + 1, ghost_im, mask=mask)"],
            ["tl.store(w + 2 * dst, ghost_re, mask=mask)",
             "tl.store(w + 2 * dst + 1, ghost_im, mask=mask)",
             "prev_re = tl.load(w + 2 * dst, mask=mask, other=0.0)",
             "prev_im = tl.load(w + 2 * dst + 1, mask=mask, other=0.0)"])

    def m13_far_fill_dropped(source: str) -> Tuple[str, int]:
        """``fill_folded_far_ghosts_B`` not carried on one component."""
        return _rewrite_block(
            source,
            ["gy_re, gy_im = _mul_imag_coefficient_left(d1r, d1i, v0_re, v0_im,",
             "EXPANSION)",
             "_carry_ghost_complex(f0, w0, h0, idx + df_y, gy_re, gy_im,",
             "kp_0, km_0, own0 & far_j, EXPANSION)"],
            ["pass"])

    def m14_near_fill_dropped(source: str) -> Tuple[str, int]:
        """``fill_symmetry_bc_B`` not carried on one component."""
        return _rewrite_block(
            source,
            ["gn_re, gn_im = _mul_imag_coefficient_left(n1r, n1i, v1_re, v1_im,",
             "EXPANSION)",
             "_carry_ghost_complex(f1, w1, h1, idx + dn_y, gn_re, gn_im,",
             "kp_d1, km_d1, own1 & near_j, EXPANSION)"],
            ["pass"])

    def m15_ghost_B_store_dropped(source: str) -> Tuple[str, int]:
        """The ghost's own ``B`` store removed: ``H`` is written and ``B`` is not.

        The fills write the FIELD; ``update_H`` then reads it. A carry that ran the
        constitutive and skipped the field store leaves the next step's curl reading
        a stale plane.
        """
        return _rewrite_block(
            source,
            ["tl.store(f + 2 * dst, ghost_re, mask=mask)",
             "tl.store(f + 2 * dst + 1, ghost_im, mask=mask)"],
            ["pass"])

    return (
        # THE ORDERING IS ARITHMETICALLY REAL, MEASURED, and the operand class
        # these two are scored under is what that measurement decided.
        # `parity_chain_associativity_leg` sweeps `_mul_imag_coefficient_left`'s two
        # arms over signed zeros, the subnormal needles and ordinary values and
        # counts 32 of 2048 words moved by a RE-ORDER and 34 by a FOLD, identically
        # on both arms. It is the SIGNED ZEROS that do it: with `c_im` bitwise +0.0,
        # `(c_im * z_im) * -1.0` is a signed zero whose sign depends on `z_im`, and
        # `fma(c_re, z_re, that)` canonicalizes `-0 + +0` to `+0` — so a flip
        # composed with a flip is NOT the composed flip at a zero word.
        # The 2026-08-21 run scored both under `subnormal_band` and both came back
        # UNCAUGHT, which is a statement about that seeding and not about the
        # kernel: a band-seeded state carries subnormals, and this defect needs
        # SIGNED ZEROS at the ghost source lane. `zero_lattice` is the class that
        # puts them there.
        # UNREACHED, not null. The word-layer sweep says both defects are REAL; 30
        # complete steps under the +-0 lattice say this gate's whole-step legs do not
        # reach them. Both halves of that are recorded, in MUTATION_EVIDENCE.
        ("m1_parity_chain_reordered",
         "the far axes applied DESCENDING; the driver applies them ascending. The "
         "word-layer sweep moves 32 of 2048 words on this rewrite, all of them at "
         "signed zeros; 30 complete steps under the +-0 lattice did not put one at "
         "a ghost source lane",
         "unreached", m1_parity_chain_reordered),
        ("m2_parity_chain_folded",
         "two complex multiplies collapsed into one coefficient product; 34 of 2048 "
         "words at the word layer, same class, same 30-step non-reach",
         "unreached", m2_parity_chain_folded),
        ("m3_triple_composite_dropped",
         "the deepest destination never written",
         "caught", m3_triple_composite_dropped),
        ("m4_far_parity_is_the_near_one",
         "+phase where fields.mirror_parity gives -phase on a shift-1 component",
         "caught", m4_far_parity_is_the_near_one),
        ("m5_reflect_row_baked",
         "n - 2 instead of stepping._far_reflect_rows' answer, at an ODD count",
         "caught", m5_reflect_row_baked),
        ("m6_top_plane_mask_dropped",
         "the folded curl's MIRROR_PERIODIC top-plane mask, removed on one axis",
         "caught", m6_top_plane_mask_dropped),
        ("m7_near_source_index_three",
         "the near fill images stored cell 3 rather than MIRROR_SOURCE_INDEX",
         "caught", m7_near_source_index_three),
        ("m8_ownership_mask_not_ANDed",
         "the ownership defect the 2026-08-20 device run found, re-planted",
         "caught", m8_ownership_mask_not_ANDed),
        ("m9_destination_coefficient_reused",
         "the near destination takes the source lane's kps/kms pair; PREDICTED "
         "NULL because a folded axis absorbs on its HIGH face alone, so the two "
         "entries are the same word to the bit",
         "null", m9_destination_coefficient_reused),
        ("m10_wall_clear_dropped",
         "zero_metal_B not carried, on a case with a live wall",
         "caught", m10_wall_clear_dropped),
        ("m11_curl_parens_flattened",
         "the curl's float32 grouping re-associated",
         "caught", m11_curl_parens_flattened),
        ("m12_prev_read_after_the_store",
         "the ghost constitutive reads its workspace AFTER writing it",
         "caught", m12_prev_read_after_the_store),
        ("m13_far_fill_dropped",
         "fill_folded_far_ghosts_B not carried on component 0",
         "caught", m13_far_fill_dropped),
        ("m14_near_fill_dropped",
         "fill_symmetry_bc_B not carried on component 1",
         "caught", m14_near_fill_dropped),
        ("m15_ghost_B_store_dropped",
         "the ghost's field store removed while its H accumulation stays",
         "caught", m15_ghost_B_store_dropped),
    )


def mutation_case_for(name: str) -> Tuple[str, Tuple[Any, ...]]:
    """The (case name, case) a mutation is scored on."""
    case_name = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    return case_name, CASES_BY_NAME[case_name]


def run_mutations(cp, product, probe, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source(product)
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "device": True}
        if hits == 0 or mutated == source:
            # A rewrite that matched nothing is a HARNESS defect, not a null: it
            # would report every mutation as uncaught while launching the shipped
            # kernel.
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_folded_complex_fused_B"
        try:
            mutant = compile_mutant(mutated, kernel_name)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"the mutant failed to import: {exc!r}"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED ({exc!r})")
            continue
        case_name, case = mutation_case_for(name)
        row["case"] = case[0]
        assert case_name == case[0]
        value_class = MUTATION_VALUE_CLASS.get(name, DEFAULT_MUTATION_VALUE_CLASS)
        steps = MUTATION_STEPS_FOR.get(name, MUTATION_STEPS)
        row["value_class"] = value_class
        row["steps"] = steps
        leg = run_leg(cp, f"mutation:{name}", case, steps, product, probe,
                      value_class=value_class, mutant=mutant)
        row["caught"] = leg.get("first_divergence") is not None
        row["first_divergence"] = leg.get("first_divergence")
        row["launches"] = leg.get("fused_kernel_launches")
        row["ghost_destinations"] = leg.get("ghost_destinations_per_source_lane")
        row["operand_census"] = leg.get("operand_census")
        row["ptx_moved"] = sorted(kernel_ptx(mutant)) != sorted(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} on {row['case']} "
            f"launches={row['launches']} expectation={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product, probe) -> List[Dict[str, Any]]:
    """Configurations the product must refuse — and one it must now ADMIT.

    THE 2026-08-31 SPLIT, in the same edit as the ``CARRIES_DEPOSIT_REPAIR``
    flip: the old single ``magnetic_source`` refusal row became two, because
    "a magnetic source in this seam" stopped being a refusal class. One that
    publishes its deposit index is ADMITTED (the repair carries it — the carry
    legs measure that in bytes); one that cannot is refused by name.
    """
    from meep_gpu.sources import FIELD_TYPE_B  # noqa: PLC0415

    class _Magnetic:
        field_type = FIELD_TYPE_B

    class _MagneticIndexed:
        field_type = FIELD_TYPE_B

        def __init__(self, index=(2, 2, 0)) -> None:
            self._point_ix, self._point_iy, self._point_iz = index

    base = CASES_BY_NAME["y_fold_periodic"]
    rows: List[Dict[str, Any]] = []
    for name, case, needle, sources, use_probe, admit in (
        ("magnetic_source_with_an_index_admitted", base, None,
         (_MagneticIndexed(),), probe, True),
        ("magnetic_source_without_an_index", base, "does not publish the index",
         (_Magnetic(),), probe, False),
        ("undeclared_sources", base, "was not declared", None, probe, False),
        ("no_probe_artifact", base, "expansion probe artifact", (), {}, False),
    ):
        driver, _ = build_driver(cp, case, "uniform", electric=False)
        try:
            verdict = product.folded_complex_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources, probe=use_probe)
            plan = product.plan_folded_complex_fused_magnetic_pair(
                driver.fields, driver.pml, sources, probe=use_probe)
            if admit:
                passed = bool(verdict.covered) and plan is not None
            else:
                passed = ((not verdict.covered) and plan is None
                          and any(needle in reason
                                  for reason in verdict.reasons))
            rows.append({
                "case": name, "device": True, "covered": bool(verdict.covered),
                "reasons": list(verdict.reasons), "plan_is_none": plan is None,
                "admits": admit, "passed": passed,
            })
        finally:
            driver.close()
            cp.get_default_memory_pool().free_all_blocks()
        log(f"  refusal {name}: covered={rows[-1]['covered']} "
            f"passed={rows[-1]['passed']}")
    return rows


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def environment(cp: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "expansion_probe_environment_variable": os.environ.get(
            "MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
    }
    try:
        import triton  # noqa: PLC0415

        payload["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        payload["triton"] = f"absent: {exc!r}"
    if cp is not None:
        payload["cupy"] = cp.__version__
        payload["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        payload["device_uuid"] = str(
            cp.cuda.runtime.getDeviceProperties(0).get("uuid"))
    return payload


NO_DEVICE_LEGS = (transcription_leg, parity_chain_leg, parity_chain_reachability_leg,
                  parity_chain_associativity_leg, mutation_vocabulary_leg,
                  seam_binding_leg, expansion_licence_leg, predicate_leg,
                  branch_reachability_leg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_folded_complex_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. A run that installs NOTHING is a MIXED "
             "configuration (CuPy appends -ftz=true unconditionally; Triton "
             "natively keeps) attributable to no policy at all, and a keep-cut "
             "licence read by such a run is a broken comparison. Run this gate "
             "ONCE PER POLICY with the probe artifact cut under that policy.")
    parser.add_argument(
        "--laptop-branch-reachability", action="store_true",
        help="include the branch-reachability and predicate legs on a host with no "
             "CuPy; they build NumPy grids and need no device")
    args = parser.parse_args(argv)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_folded_complex_fused_magnetic_pair",
        "product": "meep_gpu.triton_kernels.folded_complex_fused_magnetic_pair",
        "kernel": "folded_complex_fused_curl_constitutive_B",
        "replaces": list(SEAM_PASSES),
        "seed_root": SEED_ROOT,
        "source_sha256": source_hashes(),
        "environment": triton_device_identity.record(environment()),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "complex_fields.DEFAULT_BLOCK",
                   "subnormal_policy": args.subnormal_policy},
        "value_classes": list(VALUE_CLASSES),
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in NO_DEVICE_LEGS:
        started = time.time()
        try:
            row = leg()
        except Exception as exc:  # noqa: BLE001 - a leg that cannot run is a failure
            row = {"leg": getattr(leg, "__name__", str(leg)), "device": False,
                   "error": repr(exc), "passed": False, "findings": [repr(exc)]}
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} "
            f"findings={row.get('findings')} ({row['seconds']} s)")
        save(payload, args.out)

    if args.no_device:
        payload["device_status"] = (
            "UNRUN — invoked with --no-device; no CUDA leg, mutation or refusal in "
            "this artifact")
        payload["passed"] = all(row["passed"] for row in payload["no_device_legs"])
        # ``release`` IS THE KEY ``gate_provenance.read_verdict`` CONSULTS FIRST, and
        # it is set explicitly here because ``passed`` alone would stamp
        # ``released: True`` on an artifact that measured no bytes on any device.
        payload["release"] = {
            "released": False,
            "reasons": ["the no-device legs passed, but no device leg, mutation or "
                        "refusal has run: this artifact releases nothing"],
        }
        save(payload, args.out)
        log(f"\nno-device verdict: {payload['passed']}  ->  {args.out}")
        return 0 if payload["passed"] else 1

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep and
    # quietly did not is a process whose bytes mean nothing — and on this family it
    # is worse than that, because the EXPANSION licence it consumes is
    # policy-conditional and a keep-cut record read under flush licenses nothing.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import complex_fields, folded_complex  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as product  # noqa: PLC0415

    # THE PROBE THE DEVICE LEGS CONSUME, resolved once and RECORDED. The device bytes
    # are a function of the arm, so an artifact that does not name the record that
    # licensed it cannot be read.
    probe = complex_fields.load_expansion_probe()
    licence = folded_complex.parity_expansion_license(probe)
    payload["expansion"] = {
        "probe_path": os.environ.get("MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
        "pattern_set": list(product.PRODUCT_PROBE_PATTERNS),
        "arm": licence["arm"], "basis": licence["basis"],
        "expansion": licence["expansion"], "refusals": list(licence["refusals"]),
        "policy_resolved": licence.get("policy_resolved"),
        "candidate_policy": licence.get("candidate_policy"),
    }
    if licence["expansion"] is None:
        payload["device_status"] = "REFUSED BEFORE THE FIRST LEG"
        payload["passed"] = False
        payload["release"] = {"released": False, "reasons": [
            "no EXPANSION arm is licensed for this run, so no device leg could "
            "measure anything: " + "; ".join(licence["refusals"])]}
        save(payload, args.out)
        log(f"\nREFUSED: {licence['refusals']}")
        return 1
    log(f"expansion arm licensed: {licence['arm']} (basis {licence['basis']})")

    backends.guard_kernel_compilation(cp)
    payload["environment"] = triton_device_identity.record(environment(cp))
    # Stamped BEFORE the first leg so the partial artifact an aborted run leaves
    # behind — which is the artifact a failure is read from — says what it is.
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "byte_cases": {case[0]: case[7] for case in CASES},
        "value_classes": list(VALUE_CLASSES),
        "steps_per_mutation": MUTATION_STEPS,
    }
    save(payload, args.out)

    log("\n=== device legs ===")
    for case in CASES:
        for value_class in VALUE_CLASSES:
            # The long-horizon case runs the uniform class only: sixty steps under
            # three classes buys repetition rather than reach, and the classes are
            # what the ten-step cases sweep.
            if case[0].endswith("long_horizon") and value_class != "uniform":
                continue
            steps = case[7]
            row = run_leg(cp, f"{case[0]}/{value_class}", case, steps, product,
                          probe, value_class=value_class)
            passed, failures = verdict_of(row, require_launches=steps)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, args.out)

    log("\n=== device legs: the CARRY family (2026-08-31, with the flag flip) ===")
    # A REAL magnetic deposit in this seam, carried through the SHIPPED
    # LeadingRepairPlan/TrailingRepairPlan bracket. TWO cases: the base folded
    # PERIODIC fold (the repair must image the deposit through the far fill
    # too), and the OFFDIAGONAL row — the arm label of the corpus row this flip
    # discharged (tests:TestHoleyWvgBands.test_fields_at_kx).
    CARRY_CASES = ("y_fold_periodic", "y_fold_periodic_offdiag")
    CARRY_STEPS = 8
    for case_name in CARRY_CASES:
        carry_case = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{case_name}/{value_class}", carry_case,
                          CARRY_STEPS, product, probe, value_class=value_class,
                          magnetic=True)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, args.out)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        carry_case = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{case_name}", carry_case, CARRY_STEPS,
                      product, probe, magnetic=True, bracket=False)
        passed, failures = verdict_of(row, require_identical=False,
                                      require_launches_at_least=1,
                                      require_moved=False)
        row["armed"] = True
        row["passed"], row["failures"] = passed, failures
        row["why"] = ("the fused launch WITHOUT the shipped deposit repair; it "
                      "consumes a pre-injection B and MUST diverge")
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== armed harness mutations ===")
    base = CASES_BY_NAME["y_fold_periodic"]
    # The substitution removed: the bytes still agree (the array path is what ran)
    # and ONLY the launch counter can refuse it.
    row = run_leg(cp, "armed:no_substitution", base, 3, product, probe,
                  install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_magnetic_seam", base, 3, product, probe,
                  freeze_magnetic=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and not (row.get("seam_outputs_moved") or ())
    row["why"] = ("every route's magnetic seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    # THE LICENCE IS LOAD-BEARING, measured.
    other = 1 - int(licence["expansion"])
    # UNDER ``subnormal_band``, and that is a reachability decision measured on
    # 2026-08-21: the same leg under ``uniform`` reported the two arms BYTE-IDENTICAL
    # over three complete steps. That is not a finding about the licence, it is the
    # known property that FMA_V1 and NAIVE agree on normal operands — the probe's own
    # record says the arms are 74 words apart under keep only once an operand class
    # able to separate them is present, and uniform(-0.25, 0.25) provably has none.
    # ON A BLOCH CASE, and that is the second half of the same correction. The arms
    # differ by ONE rounding only where a full complex-by-complex multiply runs:
    # `_mul_field_left` and `_mul_coefficient_left` carry a ZERO operand, so
    # `fma(a, b, +-0)` and `(a*b) - +-0` agree on every finite input, and the only
    # body that calls `_rotate_field_left` is the per-axis Bloch block — which is
    # compiled out entirely on an unphased case. Scored on `y_fold_periodic` the leg
    # was measuring a kernel with no discriminating multiply in it at all.
    row = run_leg(cp, "armed:unlicensed_expansion_arm",
                  CASES_BY_NAME["y_fold_periodic_bloch_x"], 3, product, probe,
                  value_class="subnormal_band", expansion=other)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = (f"the EXPANSION constexpr is forced to arm {other} while the "
                  f"platform measured {licence['expansion']}; the bytes must "
                  f"diverge, or the licence constrains nothing here")
    payload["device_legs"].append(row)
    save(payload, args.out)

    # THE PARITY WORDS ARE READ, NOT SYNTHESISED. The kernel is handed a NONZERO
    # imaginary word; if the bytes do not move, the kernel is not reading the word
    # it is passed.
    mixed = CASES_BY_NAME["xy_mixed_phase_periodic"]
    poisoned = list(case_constexprs(mixed)["parity"])
    poisoned[1] = (poisoned[1][0], 0.5)
    row = run_leg(cp, "armed:parity_imaginary_word_is_read", mixed, 3, product,
                  probe, parity_override=poisoned)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = ("a NONZERO imaginary parity word is passed; the bytes must move "
                  "or the kernel synthesises the word rather than reading it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    # THE TWO READERS OF ONE SOURCE MUST AGREE. The merge bar checks the mutation
    # table against `shipped_source_from_file`; this run arms it from `inspect`. If
    # they ever differ, a rewrite can match on the laptop and miss here — the exact
    # way a mutation silently disarms.
    if (normalise_kernel_text(shipped_source(product))
            != normalise_kernel_text(shipped_source_from_file())):
        payload["device_status"] = "REFUSED BEFORE THE FIRST MUTATION"
        payload["passed"] = False
        payload["release"] = {"released": False, "reasons": [
            "shipped_source (inspect) and shipped_source_from_file (the file) "
            "disagree; the mutation table cannot be checked off-device against the "
            "text it is armed against"]}
        save(payload, args.out)
        log("\nREFUSED: the two readers of the shipped source disagree")
        return 1
    pristine = kernel_ptx(product.folded_complex_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, probe, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product, probe)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    # A DECLARED NULL MUST BE CONFIRMED, not merely permitted: under a rule that
    # accepted any expectation other than "caught", moving a stubbornly uncaught
    # mutation to a null spelling would silence it rather than explain it. Every
    # mutation must also have been ARMED — a rewrite that hit nothing measured
    # nothing, whatever it then reported.
    def _mutation_ok(row: Dict[str, Any]) -> bool:
        if row.get("error") is not None:
            return False
        if not row.get("rewrite_hits"):
            return False
        if row["expectation"] == "unreached":
            # A real defect this gate's device legs do not reach. It must come back
            # UNCAUGHT — a catch means the declaration is stale — and it must name
            # the evidence that the defect is real, or this is a null wearing a
            # longer word.
            row["evidence"] = MUTATION_EVIDENCE.get(row["mutation"])
            return (not row.get("caught")) and bool(row["evidence"])
        return bool(row.get("caught")) is (row["expectation"] == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and mutation_ok
        and all(row["passed"] for row in payload["no_device_legs"]))
    payload["device_status"] = "RUN"
    payload["release"] = {
        "released": payload["passed"],
        "reasons": ([] if payload["passed"] else
                    [f"device legs ok: {device_ok}",
                     f"refusals ok: {refusal_ok}",
                     f"mutations ok: {mutation_ok}"]),
    }
    save(payload, args.out)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"\nverdict: {payload['passed']}  ->  {args.out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
