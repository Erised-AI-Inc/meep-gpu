"""The hand-CUDA kernel table's half of the dispatch seam: rows, release, merge.

WHY THIS IS A SIBLING OF ``fastpath.py`` AND NOT A SECTION OF IT, which is the
whole cost argument for the file's existence. ``fastpath.py`` is pinned by the
Triton ``driver_dispatch`` record (it binds exactly ``driver.py``, ``fastpath.py``
and ``fields.py``), by the CUDA ``driver_dispatch`` record once it is cut, and — the
expensive one — by every Metal gate artifact, because ``gate_provenance.provenance``
digests whatever is in ``sys.modules`` and a NumPy/Metal gate process has imported
``fastpath``. So an edit there costs a full Metal fleet re-cut.

Nothing in a NumPy step imports THIS module: it is reached from ``fastpath._decide``
only after the backend rung has established a CuPy engine, so it appears in no Metal
manifest and a change to the newest, least-measured logic in the seam — the merge
rule, the release rows, the tier split — is priced at the two route re-gates and the
two record re-cuts, with no Metal cost at all. That is the reason the merge rule
lives here rather than beside the ladder that calls it.

WHAT IS HERE AND WHAT IS NOT. Here: the CUDA table's LABELS, its certifications, its
release envelope, its pending reasons, the candidate check, and
:func:`merge_tables`. Not here: the ladder (``fastpath._decide``), the precedence
constants (``fastpath.BACKEND_PRECEDENCE`` — precedence is a rung, and rungs live
with the ladder), the tail (``fastpath._finish``), and any composer. This module
imports neither ``cupy`` nor ``meep_gpu.cuda_kernels`` at module level, so
``test_package_boundary`` holds and importing it costs nothing but this file.

THE LABELS ARE NAMESPACED AT THE MERGE BOUNDARY. ``fastpath.ARM_CERTIFICATION`` and
its siblings are keyed by BARE label, and the CUDA composer's sub-step labels
(``PML``, ``ordinary``, ``dispersive``, ``complex``, ``cylindrical``, ``nonlinear``,
``off-diagonal``, ``folded complex``, ``cylindrical complex``) are also Triton
labels with different certifications. A parenthetical alias per label would be a
second vocabulary to keep in sync with the composer's own literals; a mechanical
``"cuda:"`` prefix on the composer's OWN literal cannot drift, and it leaves every
existing Triton record, weld, gate needle and board join untouched.

THE ORDERING TRAP APPLIES TO THIS FILE TOO. :data:`CUDA_DRIVER_ROUTE_FUSED_GATE`
and :data:`CUDA_RELEASED_FUSED_ARMS` are written BEFORE the campaign that measures
them, exactly as ``fastpath.DRIVER_ROUTE_FUSED_GATE`` is: the gate is named first so
the run it names is the run the shipping tree asked for, and the contract tests are
RED on the named record until the campaign cuts it. A row here is a promise, and the
red test is what keeps the promise from being mistaken for evidence.
"""

from __future__ import annotations

import json
import os
from typing import FrozenSet, Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

#: The prefix every CUDA label carries once it crosses into the shared seam. Kept
#: here rather than in ``fastpath`` beside its readers because THIS module is the
#: one that writes it: ``fastpath`` only ever splits a label it was handed.
CUDA_LABEL_PREFIX = "cuda:"

#: The table's name in ``record["tables"]``, ``record["arbitration"]`` and
#: ``FastPathPlan.backends``.
CUDA_TABLE = "cuda"


def namespaced(label: str) -> str:
    """``"fused magnetic pair"`` -> ``"cuda:fused magnetic pair"``. Idempotent."""
    text = str(label)
    return text if text.startswith(CUDA_LABEL_PREFIX) else CUDA_LABEL_PREFIX + text


def bare(label: str) -> str:
    """The composer's own literal, with the namespace taken back off."""
    text = str(label)
    return text[len(CUDA_LABEL_PREFIX):] if text.startswith(CUDA_LABEL_PREFIX) else text


def step_order() -> Tuple[str, ...]:
    """The driver's slot order, read from the one definition of it.

    ``cuda_kernels/arms.py`` imports this same tuple out of
    ``triton_kernels.launch``; so does ``metal_kernels.launch``. There is one
    definition of the driver's slot order and every composer is ordered against it,
    which is what lets :func:`merge_tables` put two tables' plans into one mapping
    without either one deciding the order.
    """
    from .triton_kernels.launch import STEP_ORDER  # noqa: PLC0415

    return tuple(STEP_ORDER)


# ---------------------------------------------------------------------------
# The labels the composer writes
# ---------------------------------------------------------------------------

#: EVERY LITERAL ``cuda_kernels/fused_pairs.py`` PASSES AS ``label=``, typed here so
#: this module needs no ``cuda_kernels`` import to answer "is that a fused label",
#: and PINNED to that file's AST by
#: ``test_cuda_certified_fused_products.test_every_typed_label_is_the_literal_its_builder_passes``.
#: Typed-plus-pinned rather than derived, for the reason the package boundary states:
#: a NumPy step must not import the CUDA composer, and ``fastpath.arm_is_fused`` is
#: reachable from one.
#:
#: NONE OF THE 38 MATCHES A TRITON LABEL OR PREFIX — measured, not assumed: Triton's
#: fused vocabulary is the ``"fused pair"`` prefix plus ``"dispersive fused pair"``
#: and ``"fused ADE state"``, and no literal below begins with ``"fused pair"``.
#: That is why the namespace is a prefix on the composer's literal rather than a
#: translation table: the two vocabularies are already disjoint, and the prefix is
#: what keeps them SEEN to be.
CUDA_FUSED_LABELS: Tuple[str, ...] = (
    "BFAST fused electric pair",
    "BFAST fused magnetic pair",
    "complex beta fused H/D pair",
    "complex beta fused electric pair",
    "complex beta fused magnetic pair",
    "complex fused H/D pair",
    "complex fused electric pair",
    "complex fused magnetic pair",
    "complex no-PML fused polarization pair",
    "complex no-absorber fused electric pair",
    "complex no-absorber off-diagonal stencil weld",
    "complex no-absorber three-slot weld",
    "conductive fused electric pair",
    "conductive/BFAST fused H/D pair",
    "cylindrical complex fused H/D pair",
    "cylindrical fused electric pair",
    "cylindrical fused magnetic pair",
    "cylindrical real fused H/D pair",
    "cylindrical real fused electric pair",
    "cylindrical real fused magnetic pair",
    "dispersive fused electric pair",
    "dispersive off-diagonal polarization pair",
    "folded complex fused electric pair",
    "folded complex fused magnetic pair",
    "folded complex off-diagonal stencil weld",
    "folded off-diagonal stencil weld",
    "fused H/D pair",
    "fused electric pair",
    "fused magnetic pair",
    "fused polarization pair",
    "no-PML dispersive fused electric pair",
    "no-absorber fused polarization pair",
    "no-absorber three-slot dispersive weld",
    "off-diagonal stencil weld",
    "special_kz fused H/D pair",
    "special_kz fused electric pair",
    "special_kz fused magnetic pair",
    "three-slot dispersive weld",
)

#: The CUDA composer's null constitutive arm. The same words the Triton composer
#: uses (``fastpath.NULL_ARM_LABEL``), which is why rung 7's drop compares the BARE
#: label: a ``"cuda:no-PML null"`` that reached the dispatch set would launch nothing
#: and the record would call the step fused.
NULL_ARM_LABEL = "no-PML null"


def is_fused(label: str) -> bool:
    """Is this label (bare or namespaced) one of the composer's fused products?"""
    return bare(label) in set(CUDA_FUSED_LABELS)


# ---------------------------------------------------------------------------
# What certified each product
# ---------------------------------------------------------------------------

#: ``"cuda:<label>" -> (product family, cuda_kernels/fingerprints.json key)``.
#:
#: THE FAMILY IS THE ``FUSED_PRODUCTS`` KEY, not a registry family: on this track a
#: fused product IS the unit a device gate certified, and the ledger's
#: ``kernel_module`` names that product's own file.
#:
#: FIVE ROWS BELOW NAME A LEDGER KEY THAT DOES NOT EXIST YET, and that is the
#: ordering rule rather than an oversight — see this module's docstring.
#: ``test_every_cuda_certification_points_at_a_readable_record`` is RED on exactly
#: those five until the campaign's writers and ``rebind_cuda_welds.py --seed`` cut
#: them, and the red list is the campaign's own to-do:
#:
#:   * ``cuda_fused_electric_pair_2026-08-31_fillcarry_r3`` — the released artifact
#:     exists and its curated pins (product module, ``compile_cache.py``, gate
#:     script, probe) all match the checkout, measured 2026-09-10, so a block writer
#:     plus a seed binds it with no device time;
#:   * ``cuda_three_slot_dispersive_weld_2026-09-11_regate`` — the 2026-09-03 block
#:     exists but both product modules have drifted and its four-leg layout
#:     (``keep_pml``/``flush_pml``/``keep_no_pml``/``flush_no_pml``) is not one
#:     ``rebind_cuda_welds`` can read, so the campaign re-gates into canonical
#:     ``keep``/``flush``;
#:   * ``cuda_conductive_fused_electric_pair_2026-09-02b`` — block present, curated
#:     pins clean, seed only;
#:   * ``cuda_complex_fused_magnetic_pair_2026-09-11_regate`` and
#:     ``cuda_complex_fused_electric_pair_2026-09-11_regate`` — no block at all, and
#:     the 2026-08-31 artifact's gate script has drifted, so the campaign re-runs
#:     ``gate_cuda_fused_complex_pairs.py`` and ``record_fused_complex_pairs.py``
#:     derives the blocks from what it wrote.
CUDA_ARM_CERTIFICATION: Mapping[str, Tuple[str, str]] = {
    # --- resolving today (17) ---
    "cuda:fused magnetic pair": (
        "cuda_fused_magnetic_pair", "cuda_fused_magnetic_pair_2026-08-27"),
    "cuda:folded complex fused magnetic pair": (
        "cuda_complex_folded_fused_magnetic_pair",
        "cuda_complex_folded_fused_magnetic_pair_2026-09-02"),
    "cuda:folded complex fused electric pair": (
        "cuda_complex_folded_fused_electric_pair",
        "cuda_complex_folded_fused_electric_pair_2026-09-02"),
    "cuda:complex beta fused magnetic pair": (
        "cuda_complex_beta_fused_magnetic_pair",
        "cuda_complex_beta_fused_magnetic_pair_2026-09-02"),
    "cuda:complex beta fused electric pair": (
        "cuda_complex_beta_fused_electric_pair",
        "cuda_complex_beta_fused_electric_pair_2026-09-02"),
    "cuda:cylindrical real fused magnetic pair": (
        "cuda_cylindrical_real_fused_magnetic_pair",
        "cuda_cylindrical_real_fused_magnetic_pair_2026-09-02"),
    "cuda:cylindrical real fused electric pair": (
        "cuda_cylindrical_real_fused_electric_pair",
        "cuda_cylindrical_real_fused_electric_pair_2026-09-02"),
    "cuda:special_kz fused magnetic pair": (
        "cuda_special_kz_fused_magnetic_pair",
        "cuda_special_kz_fused_magnetic_pair_2026-09-02"),
    "cuda:special_kz fused electric pair": (
        "cuda_special_kz_fused_electric_pair",
        "cuda_special_kz_fused_electric_pair_2026-09-02"),
    "cuda:BFAST fused magnetic pair": (
        "cuda_bfast_fused_magnetic_pair", "cuda_bfast_fused_magnetic_pair_2026-09-02"),
    "cuda:BFAST fused electric pair": (
        "cuda_bfast_fused_electric_pair", "cuda_bfast_fused_electric_pair_2026-09-02"),
    "cuda:off-diagonal stencil weld": (
        "cuda_offdiag_fused_electric_pair", "cuda_offdiag_stencil_welds_2026-09-02"),
    "cuda:folded off-diagonal stencil weld": (
        "cuda_folded_offdiag_fused_electric_pair",
        "cuda_offdiag_stencil_welds_2026-09-02"),
    # THE TWO COMPLEX STENCIL WELDS, RELEASED 2026-09-17 against the block their
    # pending rows named. That block, cuda_complex_offdiag_stencil_welds_2026-09-03,
    # stood released in certification.json with nothing in cuda_kernels/
    # fingerprints.json binding it -- 132 of 132 S1 cases bit-identical under both
    # canonical policies and no ledger entry -- and six of its 53 imported pins had
    # since moved, so it could only have been seeded DRIFTED. The gate was RE-RUN on
    # the live tree instead (keep 116.56 s, flush 120.74 s, both released, the same
    # 132/66/11 counts), recorded as cuda_complex_offdiag_stencil_welds_2026-09-17
    # and seeded clean at 0 source drift.
    "cuda:folded complex off-diagonal stencil weld": (
        "cuda_folded_complex_offdiag_fused_electric_pair",
        "cuda_complex_offdiag_stencil_welds_2026-09-17"),
    "cuda:complex no-absorber off-diagonal stencil weld": (
        "cuda_complex_no_pml_offdiag_fused_electric_pair",
        "cuda_complex_offdiag_stencil_welds_2026-09-17"),
    "cuda:dispersive off-diagonal polarization pair": (
        "cuda_dispersive_offdiag_fused_polarization_pair",
        "cuda_dispersive_offdiag_polarization_pair_2026-09-02"),
    "cuda:fused polarization pair": (
        "cuda_fused_polarization_pair", "cuda_fused_polarization_pair_2026-09-01"),
    "cuda:no-absorber fused polarization pair": (
        "cuda_no_pml_fused_polarization_pair",
        "cuda_no_pml_fused_polarization_pair_2026-09-01"),
    "cuda:complex no-PML fused polarization pair": (
        "cuda_complex_no_pml_fused_polarization_pair",
        "cuda_complex_no_pml_fused_polarization_pair_2026-09-02b"),
    # --- promised by the campaign, RED until it seeds them (5) ---
    "cuda:fused electric pair": (
        "cuda_fused_electric_pair",
        "cuda_fused_electric_pair_2026-08-31_fillcarry_r3"),
    "cuda:three-slot dispersive weld": (
        "cuda_three_slot_dispersive_weld",
        "cuda_three_slot_dispersive_weld_2026-09-11_regate"),
    "cuda:conductive fused electric pair": (
        "cuda_conductive_fused_electric_pair",
        "cuda_conductive_fused_electric_pair_2026-09-02b"),
    "cuda:complex fused magnetic pair": (
        "cuda_complex_fused_magnetic_pair",
        "cuda_complex_fused_magnetic_pair_2026-09-11_regate"),
    "cuda:complex fused electric pair": (
        "cuda_complex_fused_electric_pair",
        "cuda_complex_fused_electric_pair_2026-09-11_regate"),
    # --- Tier 2, 2026-09-13: the cylindrical complex pairs' 2026-09-04 blocks bound
    # module bytes that drifted on 2026-08-31, so the families were re-gated on the
    # shipping bytes (results/cuda_regate_2026-09-13_cyl) and the welds seeded by
    # rebind_cuda_welds --seed from that campaign. ---
    "cuda:cylindrical fused magnetic pair": (
        "cuda_cylindrical_fused_magnetic_pair",
        "cuda_cylindrical_fused_magnetic_pair_2026-09-13_cyl"),
    "cuda:cylindrical fused electric pair": (
        "cuda_cylindrical_fused_electric_pair",
        "cuda_cylindrical_fused_electric_pair_2026-09-13_cyl"),
    # --- Tier 3, 2026-09-13: the no-absorber three-slot weld needs NO device round
    # of its own. It shares the dispersive weld's re-gate, whose ``kernel_module``
    # names BOTH products' files (``no_pml_three_slot_dispersive_weld.py`` beside
    # ``three_slot_dispersive_weld.py``) -- measured here against the checkout, and
    # the key resolves in cuda_kernels/fingerprints.json AND certification.json
    # today, unlike the five rows above it when they were typed. What this arm was
    # waiting for was a ROUTE case and the ``pml_active`` envelope move, not a weld.
    "cuda:no-absorber three-slot dispersive weld": (
        "cuda_three_slot_no_pml_dispersive_weld",
        "cuda_three_slot_dispersive_weld_2026-09-11_regate"),
    # --- 2026-09-14: the COMPLEX no-absorber span's own weld. Unlike the row above
    # it, this one did need a device round: the 2026-09-04 block had stood in
    # certification.json since it was cut with NO fingerprints.json entry behind it,
    # so the label named a narrative and not a measurement. Both canonical float32
    # policy legs were re-run on device (results/cuda_regate_2026-09-14_n2/, keep
    # and flush, both released), record_cuda_regate.py transcribed the campaign and
    # rebind_cuda_welds.py --seed cut the ledger entry FROM that run rather than by
    # hand. The weld pins no release table, so the row below cannot drift it.
    "cuda:complex no-absorber three-slot weld": (
        "cuda_three_slot_complex_no_pml_dispersive_weld",
        "cuda_complex_no_pml_three_slot_dispersive_weld_2026-09-04"),
    # --- 2026-09-17: the two SINGLE arms the merge may adopt as a seam
    # (SINGLE_ARM_SEAMS). The family here is the REGISTRY family, not a
    # FUSED_PRODUCTS key -- a single has no product module -- and the ledger key is
    # the byte gate that certified the two entry points the single launches
    # (kernels step_{B,D}_pml_real and update_{E,H}_pml_real respectively). They are
    # here so the veto leg of a real-PML row can be served by this table's own
    # certified kernels, and a fusion number measured against the kernels the
    # product replaces rather than against another table's.
    "cuda:PML": ("cuda_curl", "bit_identity_gate"),
    "cuda:ordinary": ("cuda_constitutive", "constitutive_2026-08-15"),
    # --- 2026-09-27: three more single arms the merge may adopt on the D/E seam
    # (partnered with the step_D 'PML' single above), and the fill arm the merge
    # adopts as a TWIN pair (FILL_ARM_TWINS). The family is the REGISTRY family and
    # the ledger key the byte gate that certified the entry point(s) the plan
    # launches; every block below was re-gated on the Round B bytes
    # (certification.json ``cuda_regate_2026-09-25_roundb``) and each fingerprints
    # entry is re-bound to it. No kernel is new and no gate is new: what is new is a
    # launch-argument resolution and a launcher per family (``cuda_kernels/arms.py``)
    # that call the derivations the family's own wrapper calls when handed the layer.
    "cuda:off-diagonal": ("cuda_offdiag", "offdiag_2026-08-16"),
    "cuda:folded off-diagonal": (
        "cuda_folded_offdiag", "cuda_folded_offdiag_2026-08-21"),
    "cuda:dispersive off-diagonal": (
        "cuda_dispersive_offdiag", "cuda_dispersive_offdiag_2026-08-20"),
    # The six real-storage in-seam kernels (fill_symmetry_{B,D},
    # fill_folded_far_{B,D}, zero_metal_{B,D}); the fill arm launches the first four,
    # and zero_metal stays on the array path between its two passes.
    "cuda:mirror fill": ("cuda_mirror_fill", "in_seam_2026-08-21"),
}

#: Which sub-step arms each fused product substitutes, namespaced on both sides.
#: The twin of ``fastpath.FUSED_ARM_CONSTITUENTS`` and it exists for the same
#: reason: a fused product writes ONE label into every slot it absorbs, so the arms
#: it implements disappear from ``selected`` and the pending rung would be judging a
#: label rather than the arithmetic underneath it.
#:
#: TYPED, AND PINNED EQUAL TO ``fused_pairs.FUSED_PAIR_ARMS`` PLUS
#: ``FUSED_PAIR_EXTRA_ARMS`` by test — same rule as the labels above, same reason.
#: Where a product declares extra absorb rows (the nonlinear cell, the no-PML curl
#: alternates) the union is taken: a constituent is an arm the product MAY
#: substitute on some admitted row, and the rung asks whether any of them is pending.
CUDA_FUSED_ARM_CONSTITUENTS: Mapping[str, Tuple[str, ...]] = {
    "cuda:fused magnetic pair": ("cuda:PML", "cuda:nonlinear", "cuda:ordinary"),
    "cuda:complex fused magnetic pair": ("cuda:complex",),
    "cuda:cylindrical fused magnetic pair": ("cuda:complex", "cuda:cylindrical complex"),
    "cuda:folded complex fused magnetic pair": ("cuda:complex", "cuda:folded complex"),
    "cuda:complex beta fused magnetic pair": ("cuda:complex beta",),
    "cuda:cylindrical real fused magnetic pair": ("cuda:cylindrical", "cuda:ordinary"),
    "cuda:special_kz fused magnetic pair": ("cuda:real beta",),
    "cuda:BFAST fused magnetic pair": ("cuda:BFAST",),
    "cuda:fused electric pair": ("cuda:PML", "cuda:ordinary"),
    "cuda:dispersive fused electric pair": ("cuda:PML", "cuda:dispersive"),
    "cuda:no-PML dispersive fused electric pair": (
        "cuda:conductive", "cuda:no-PML curl", "cuda:no-PML dispersive store"),
    "cuda:conductive fused electric pair": ("cuda:conductive", "cuda:ordinary"),
    "cuda:complex fused electric pair": ("cuda:complex",),
    "cuda:cylindrical fused electric pair": ("cuda:complex", "cuda:cylindrical complex"),
    "cuda:complex no-absorber fused electric pair": (
        "cuda:complex no-PML curl", "cuda:complex no-PML stored E"),
    "cuda:folded complex fused electric pair": ("cuda:complex", "cuda:folded complex"),
    "cuda:complex beta fused electric pair": ("cuda:complex beta",),
    "cuda:cylindrical real fused electric pair": ("cuda:cylindrical", "cuda:ordinary"),
    "cuda:special_kz fused electric pair": ("cuda:real beta",),
    "cuda:BFAST fused electric pair": ("cuda:BFAST",),
    "cuda:off-diagonal stencil weld": ("cuda:PML", "cuda:off-diagonal"),
    "cuda:folded off-diagonal stencil weld": ("cuda:PML", "cuda:folded off-diagonal"),
    "cuda:folded complex off-diagonal stencil weld": (
        "cuda:complex off-diagonal PML", "cuda:folded complex"),
    "cuda:complex no-absorber off-diagonal stencil weld": (
        "cuda:complex no-PML curl", "cuda:complex off-diagonal no-PML"),
    "cuda:fused polarization pair": ("cuda:ADE", "cuda:dispersive"),
    "cuda:no-absorber fused polarization pair": (
        "cuda:no-PML ADE", "cuda:no-PML dispersive store"),
    "cuda:complex no-PML fused polarization pair": (
        "cuda:complex no-PML ADE", "cuda:complex no-PML stored E"),
    "cuda:dispersive off-diagonal polarization pair": (
        "cuda:ADE", "cuda:dispersive off-diagonal"),
    "cuda:three-slot dispersive weld": ("cuda:ADE", "cuda:PML", "cuda:dispersive"),
    "cuda:no-absorber three-slot dispersive weld": (
        "cuda:conductive", "cuda:no-PML ADE", "cuda:no-PML curl",
        "cuda:no-PML dispersive store"),
    "cuda:complex no-absorber three-slot weld": (
        "cuda:complex no-PML ADE", "cuda:complex no-PML curl",
        "cuda:complex no-PML stored E"),
    "cuda:fused H/D pair": ("cuda:PML", "cuda:nonlinear", "cuda:ordinary"),
    "cuda:cylindrical real fused H/D pair": ("cuda:cylindrical", "cuda:ordinary"),
    "cuda:cylindrical complex fused H/D pair": (
        "cuda:complex", "cuda:cylindrical complex"),
    "cuda:complex fused H/D pair": ("cuda:complex",),
    "cuda:special_kz fused H/D pair": ("cuda:real beta",),
    "cuda:complex beta fused H/D pair": ("cuda:complex beta",),
    "cuda:conductive/BFAST fused H/D pair": (
        "cuda:BFAST", "cuda:conductive", "cuda:ordinary"),
}


# ---------------------------------------------------------------------------
# The release
# ---------------------------------------------------------------------------

#: THE CUDA DRIVER-ROUTE GATE, NAMED BEFORE IT RUNS. ``gate_dispatch_fused_route.py
#: --backend cuda`` writes ``results/<this name>/`` with five legs, and
#: ``recut_driver_dispatch_record.py --backend cuda`` refuses to cut the record
#: unless every ``(arm, case, leg)`` row of :data:`CUDA_RELEASED_FUSED_ARMS` appears
#: in that leg's ``arms_driven``. Naming it here first is what makes the run the
#: shipping tree asked for the run that is measured.
#: REPOINTED 2026-09-15 to a FRESH stamp rather than re-run under the old one: the
#: ``2026-09-14_weldgrid`` run is what the 346-of-594 board cites, and overwriting it
#: would strand that board's provenance. The re-run exists because the replacement-case
#: edits moved ``fastpath.py`` and ``gate_dispatch_fused_route.py``, which the record
#: binds, so the board no longer describes the tree.
# ``_2026-09-15_offdiag``: the same cases and legs again, because the Triton
# off-diagonal fix moved fastpath.py and the route file, both bound by this record.
# ``dispatch_fused_route_cuda_2026-09-15_weldgrid`` released all five legs and stays.
# ``_2026-09-15_gapclose``: the same cases and legs again, because one Triton batch
# moved fastpath.py and the route file, both bound by this record: the release of
# ``fused pair D (complex conductive no-PML)``, which restored the Triton
# ``complex_no_pml_3d`` row, and the lift of ``complex folded off-diagonal`` with the
# widening of ``fused pair B (folded complex)``, which turned the two off-diagonal
# folds back into Triton dispatch rows. On ``complex_no_pml_3d`` the Triton incumbent
# under default precedence is now a fused label. DRIVE_CUDA's case set is unchanged:
# the two off-diagonal folds stay out of it (see the note in the route file).
# ``_2026-09-15_offdiag`` released five legs and stays.
# ``_2026-09-17_allpaths``: the round the two withdrawn notes above both point at.
# DRIVE_CUDA's case set CHANGES here for the first time since the target round: the
# two off-diagonal folds return (``folded_complex_offdiag_2d`` and
# ``folded_complex_nobloch_offdiag_2d``), with the
# :data:`CUDA_OFF_DIAGONAL_CORNERS` entry that bounds them. What that buys is three
# census instances on B->H, and what it risks is named where the row is written: the
# arbitration-default-precedence control is a NEW measurement on those grids, because
# the 2026-09-15 batch moved the Triton table's answer there. ``_2026-09-15_gapclose``
# released its legs and stays as the record of the table with the folds held out.
# ``_2026-09-17_singles2`` released on all five legs and cut the 352 / 597 board; it stays
# as that record's provenance. ``_2026-09-19_overhead`` is the re-run after the two
# fused-path overhead fixes: ``cuda_kernels/fused_pairs.py`` now holds each product's
# emitted source (and, for the non-rotating products, its resolved arguments) per plan
# instead of rebuilding them on every launch, and the route gate records the bytes of
# ``deposit_repair.py`` and ``fused_pairs.py`` it ran. CORRECTED 2026-09-20: only
# ``fused_pairs.py`` is BOUND by this record; ``deposit_repair.py`` is recorded in each
# leg's provenance and bound by no CUDA record (the route gate's own note reads
# "RECORDED, NOT BOUND"), which the sentence above overstated.
# ``_2026-09-20_restrict`` is the re-run after the deposit repair was restricted to the
# component the source writes: the bracket a fused pair carries on the source's seam now
# repairs one target per single-component source instead of three, bit-identical, and
# every route row that dispatches a bracketed seam executes different bytes. Measured in
# process on this table before it landed: 1.50-1.66x on the two-pair step at <= 1.0M
# cells (``results/repair_phase0_2026-09-19_phase0/``).
# ``_2026-09-24_hoist`` is the re-run after the fused magnetic pair's own-cell loads
# were hoisted to the top of the kernel (``cuda_kernels/fused_magnetic_pair.py``, module
# docstring item 6): same arithmetic, same stores, the twelve loads issued before the first
# store instead of one at a time after each; 1.34-1.37x on the pair at 2.1-7.1M cells.
# ``_2026-09-25_roundb`` (named 2026-09-25, before its campaign) is the re-run after the
# same own-cell hoist reached four more kernels: the two certified real-PML constitutive
# singles (``update_H_pml_real`` / ``update_E_pml_real`` in
# ``cuda_kernels/constitutive_kernels.py``, register view: the certified helper steps
# registers, every own-cell word is loaded before the first store and written back in
# the certified store order, six stores each, same arithmetic) and the Dcyl complex
# fused magnetic and electric pairs (``cuda_kernels/cylindrical_fused_{magnetic,
# electric}_pair.py``). The products that lift the singles read the certified statement
# form back through ``cuda_kernels/own_cell_hoist.py`` and emit byte-identical text, so
# ``fused_magnetic_pair.py`` executes different module bytes but the same device text.
# The run is also the first on the batched ``stepping.py`` wall clears and the citation
# re-point. Measured kernel-only on one RTX A6000 at each arm's corpus-maximum grid
# (``results/cuda_a1_screen_2026-09-25/``): update_H 1.92x, update_E 1.75x, the
# cylindrical pairs 1.34x / 1.33x; no whole-step number of record yet.
# ``_2026-09-25_roundb`` released all five legs and is the route of record the
# 352 / 597 board cites; it stays as that board's provenance.
# ``_2026-09-27_flip`` (named 2026-09-27, before its campaign) is the re-run after two
# edits to this file and to ``cuda_kernels/arms.py`` / ``cuda_kernels/registry.py``,
# none of which adds or changes a kernel. (1) The six certified real-storage in-seam
# fill kernels (``in_seam_2026-08-21``, re-gated in Round B) are registered as a CUDA
# fill arm on ``fill_B``/``fill_D`` (``cuda:mirror fill``), so rung 6b stops refusing a
# real mirror-folded row whole when this table composes without a second table. (2)
# The three off-diagonal ``update_E`` singles (``cuda:off-diagonal``,
# ``cuda:folded off-diagonal``, ``cuda:dispersive off-diagonal``) carry a
# launch-argument resolver and a launcher, so the merge adopts them as the D/E seam
# beside the already-launchable ``step_D`` 'PML'. The campaign also gains a sixth leg,
# ``cuda_alone``: the preference unset and Triton made unimportable in the gate
# process, which is the composition a host without a validated Triton gets and which
# no earlier leg drove.
# ``_2026-09-30_091`` (named 2026-09-30, before its campaign) is release 0.9.1's route:
# the off-diagonal ``update_E`` single now launches its own-cell hoisted text, so the
# legs that drive it execute different device bytes. ``_2026-09-27_flip`` stays the
# record of the 0.9.0 route.
CUDA_DRIVER_ROUTE_FUSED_GATE = "dispatch_fused_route_cuda_2026-09-30_091"

#: THE ARMS THIS TABLE MAY DISPATCH, and the gate cases that drove each. TIER 1
#: (2026-09-11), TIER 2 (2026-09-13) AND TIER 3 (2026-09-13, typed ahead of its
#: campaign — read the tier paragraph below before reading a row as evidence).
#:
#: THE TIER SPLIT IS A RELEASE DISCIPLINE, NOT A SCOPE CUT.
#: ``recut_driver_dispatch_record.py`` refuses the whole record unless EVERY
#: ``(arm, case)`` in this table appears in a dispatching leg's ``arms_driven``, so a
#: single case that does not lift on the campaign host makes the record un-cuttable
#: and costs a second device round. Tiers 1 and 2 have their route case MEASURED ON
#: DEVICE before it was typed: Tier 1 off-device against the fourteen cases
#: (``cuda_kernels.arms.plan_step(fuse=True)`` over a NumPy lift, 2026-09-10) and on
#: the GPU host in the tier1 campaign; Tier 2 through the route gate's own ``run_case``
#: on the GPU host (2026-09-13, one RTX A6000, 96-step ladders, every case PASS-FUSED
#: with substitution EXACT and the second consult site PASS) with the rows below
#: patched in-process before they were written down.
#:
#: TIER 3 IS THE OTHER ORDER AND SAYS SO IN EVERY ROW IT TOUCHES. The twelve case
#: names added on 2026-09-13 for the widened axes — ``dispersive5_2d``,
#: ``dispersive6_2d``, ``complex_1d``, ``complex_3d_thinline``, ``complex_3d``,
#: ``folded_complex_3d``, ``folded_complex_offdiag_2d``,
#: ``folded_complex_nobloch_offdiag_2d``, ``folded_complex_beta_bloch_2d``,
#: ``folded_special_kz_2d``, ``absorber_1d`` and ``material_dispersion_0d`` — were
#: LIFTED off-device on stock MEEP 1.33.0 (``lift_simulation(prefer_gpu=False)``,
#: the run shape read straight off ``fastpath._run_shape``) and have NOT been driven
#: through a seam on any device. What each row's reason string states is the LIFT and
#: nothing more, and the arm this buys is admitted only once
#: :data:`CUDA_DRIVER_ROUTE_FUSED_GATE`'s campaign drives the case: that is this
#: module's own ordering rule (the gate is named before it runs, and the contract
#: tests are RED on the named record until the campaign cuts it), applied to rows
#: rather than to a gate name. A reader pricing coverage off this table before that
#: campaign is reading a promise.
#:
#: TIER 2 (2026-09-13) — the rows measured and typed, per label:
#:
#:   * cylindrical fused magnetic pair / cylindrical fused electric pair — FIVE Dcyl
#:     complex cases, m in {-1, 0, 1, 2, 3} (``cylindrical_m0_complex`` forces complex
#:     storage at m = 0), probe leg; the ``m`` row is the frozenset of those five;
#:   * off-diagonal stencil weld / folded off-diagonal stencil weld —
#:     ``offdiag_magnetic_2d`` (a cylinder with an Hz source, so the in-plane E the
#:     weld owns is live) and its fold ``folded_offdiag_magnetic_2d`` (Mirror(Y,
#:     phase=-1), the fills served by the Triton table beside it), shipped leg;
#:   * complex beta fused magnetic/electric pair — ``complex_beta_2d`` (the
#:     special_kz cell stored complex, ``kz_2d="complex"``), probe leg;
#:   * BFAST fused magnetic/electric pair — ``bfast_1d`` (refl-angular's shape: a
#:     z-only cell declared 3-D, ``bfast_scaled_k`` set, real storage), shipped leg;
#:   * the ordinary magnetic pair's OFF-DIAGONAL row is gone, replaced by a CORNER
#:     (:data:`CUDA_OFF_DIAGONAL_CORNERS`) driven on ``offdiag_2d`` (Ez), the two Hz
#:     cases above and ``pml_3d`` (the sphere, which dispatches this arm now rather
#:     than standing as the envelope control);
#:   * the complex pairs' ``bloch`` row is gone: ``complex_nobloch_2d``
#:     (``force_complex_fields`` with no k_point) drives False beside ``bloch_2d``'s
#:     True, every other axis single-valued across both.
#:
#: NOT TAKEN, AND WHY, AFTER TIER 3. ``nonlinear_1d`` was measured PASS-FUSED for
#: the magnetic pair (its nonlinear cell is welded) but the ``nonlinearity`` row is
#: KEPT as the campaign's envelope witness — once the off-diagonal row went per-arm
#: it is the one axis on which this arm's predicate admits and the release refuses
#: (``gate_dispatch_fused_route.ENVELOPE_ROWS["cuda"]``); it costs 2 instances, and
#: Tier 3 deliberately left it standing rather than buy them, because a release that
#: has only ever said yes is not known to be able to say no. The ``pml_active``
#: envelope move Tier 3 DID make (see :data:`CUDA_FUSED_RELEASE_ENVELOPE`) is what
#: released the no-absorber three-slot dispersive weld on ``absorber_1d`` and
#: ``material_dispersion_0d``; the two COMPLEX no-absorber products
#: (``complex_no_pml_3d``, ``complex_no_pml_offdiag``) stay pending for the ledger
#: rather than for the envelope. The dispersive off-diagonal polarization pair and
#: the widening of the magnetic pair's off-diagonal corner to one pole were BOTH
#: dropped from Tier 3 on a measurement: their case ``folded_offdiag_dispersive_2d``
#: does not lift on a stock MEEP (see that arm's pending reason). The folded complex
#: off-diagonal stencil weld (1 instance) is not released here either — the case it
#: has always been waiting for is now named by this table for the folded complex
#: magnetic pair, but the weld's own ledger entry is missing, which is a seed rather
#: than a row.
#:
#: A MEASURED HAZARD THE CASE ORDER RESPECTS. A Dcyl COMPLEX case stepped earlier in
#: the same process makes every later REAL-storage Cartesian case diverge from the
#: array path at subnormal magnitudes (<= 1e-35, ~half the differing words
#: subnormal, one side exactly zero) from step 24-96 on -- on the fused AND the
#: unfused dispatch leg alike, with the CUDA cylindrical pairs installed or not,
#: while the host flush flag never moves (measured 2026-09-13: cylindrical_m1 then
#: pml_2d diverges; cylindrical (m = 0, real), complex_nobloch_2d or complex_beta_2d
#: then pml_2d does not; every case alone is byte-identical). So the five
#: cylindrical cases are the LAST rows of ``DRIVE_CUDA`` and lift only on the probe
#: leg; nothing real-storage steps after them. The cause is open (expansion plan
#: §10) and this table makes no claim about a process that interleaves the two.
#:
#: THE LEG MATTERS AND IS PART OF THE ROW. Real-storage arms are driven on
#: ``cuda_shipped`` (and re-driven on ``cuda_harness_keep``); complex-storage arms
#: dispatch ONLY on ``cuda_shipped_expansion_probe``, because their expansion
#: licence comes from the unified probe record and nothing else offers it. The
#: recut reads :data:`CUDA_RELEASED_ARM_LEGS` beside this table.
CUDA_RELEASED_FUSED_ARMS: Mapping[str, Tuple[str, ...]] = {
    # ``pml_3d`` (the sphere) writes off-diagonal chi1inv rows through subpixel
    # averaging -- measured on the GPU host 2026-09-11 -- and until 2026-09-13 this arm's
    # own axes pinned that False, so the tier1 row listed ``pml_3d_diagonal`` and the
    # sphere stood as the envelope control. The off-diagonal admission is now a
    # CORNER (:data:`CUDA_OFF_DIAGONAL_CORNERS`) driven at 2-D by ``offdiag_2d`` (Ez)
    # and the two Hz cases, and at 3-D by the sphere itself.
    "cuda:fused magnetic pair": (
        "pml_2d", "magnetic_seam_2d", "conductive_2d", "dispersive_2d",
        "folded_dispersive_2d", "pml_1d", "folded_2d",
        "pml_3d_diagonal", "pml_3d", "offdiag_2d", "offdiag_magnetic_2d",
        "folded_offdiag_magnetic_2d",
        "dispersive5_2d", "dispersive6_2d"),
    "cuda:fused electric pair": (
        "pml_2d", "magnetic_seam_2d", "pml_1d", "folded_2d", "pml_3d_diagonal"),
    # THE TWO MULTI-POLE CASES DRIVE THIS WELD TOO, since the 2026-09-14 target round:
    # on dispersive5_2d and dispersive6_2d the composer selects this weld on
    # step_D/update_E/update_P beside the magnetic pair, and the shipped leg of
    # dispatch_fused_route_cuda_2026-09-14_target measured both PASS-FUSED with the
    # substitution EXACT (drops of 14 and 17 launches a step -- the weld collapses the
    # per-pole polarization launches the Triton baseline issues one by one). That
    # first cut did NOT release (material_dispersion_0d withheld the leg), so the
    # re-run is the record that banks these two; they are listed because the route
    # table names this arm on those rows and a release list that disagreed with the
    # gate's own DRIVE table would be the drift test_dispatch_reachability refuses.
    "cuda:three-slot dispersive weld": ("dispersive_2d", "folded_dispersive_2d",
                                        "dispersive5_2d", "dispersive6_2d"),
    "cuda:conductive fused electric pair": ("conductive_2d",),
    "cuda:complex fused magnetic pair": (
        "bloch_2d", "complex_nobloch_2d",
        "complex_1d", "complex_3d_thinline", "complex_3d"),
    "cuda:complex fused electric pair": (
        "bloch_2d", "complex_nobloch_2d",
        "complex_1d", "complex_3d_thinline", "complex_3d"),
    "cuda:folded complex fused magnetic pair": (
        # THE TWO OFF-DIAGONAL CASES WERE WITHDRAWN 2026-09-14. Both lift, and on the
        # CUDA-preferred leg both drove this pair PASS-FUSED with the substitution
        # EXACT. But under DEFAULT precedence the step is REFUSED, so neither can be
        # shown dispatching on the shipped precedence: on a folded complex
        # off-diagonal grid a D-side slot falls to the Triton single arm
        # `complex folded off-diagonal`, and that arm is PENDING -- its family byte
        # gate passed on the device, but no planner/driver recertification stands behind
        # it and the nine families the 2026-08-14 recert covered do not include its
        # own. The route gate's arbitration-default-precedence control reads REFUSED,
        # which is the honest answer rather than a gate defect. Re-add both cases, and
        # the CUDA_OFF_DIAGONAL_CORNERS entry that bounded them, in the round that
        # certifies that arm.
        #
        # RE-ADDED 2026-09-17, in that round. `complex folded off-diagonal` left
        # PENDING_DEVICE_GATE_ARMS on the 2026-09-15 ruling, so the reason the two
        # cases were withdrawn is gone. WHAT THIS ROUND MEASURES IS NOT WHAT THE
        # 2026-09-14 ROUND MEASURED, and the row is written knowing that: the same
        # 2026-09-15 batch changed the TRITON table's answer on these grids (its
        # folded complex magnetic pair now holds step_B/update_H under default
        # precedence), so the arbitration-default-precedence control is a NEW
        # measurement rather than a restored one and may legitimately read REFUSED
        # again. If it does, both cases come back out with this note rewritten to say
        # so -- the row is not evidence until the leg that drives it releases.
        "folded_complex_2d", "folded_complex_3d",
        "folded_complex_offdiag_2d", "folded_complex_offdiag_hz_2d",
        "folded_complex_nobloch_offdiag_2d"),
    "cuda:folded complex fused electric pair": (
        "folded_complex_2d", "folded_complex_3d"),
    "cuda:special_kz fused magnetic pair": ("special_kz_2d", "folded_special_kz_2d"),
    "cuda:special_kz fused electric pair": ("special_kz_2d", "folded_special_kz_2d"),
    "cuda:cylindrical real fused magnetic pair": ("cylindrical",),
    "cuda:cylindrical real fused electric pair": ("cylindrical",),
    # --- Tier 2, 2026-09-13 ---
    "cuda:cylindrical fused magnetic pair": (
        "cylindrical_m0_complex", "cylindrical_mneg1", "cylindrical_m1",
        "cylindrical_m2", "cylindrical_m3"),
    "cuda:cylindrical fused electric pair": (
        "cylindrical_m0_complex", "cylindrical_mneg1", "cylindrical_m1",
        "cylindrical_m2", "cylindrical_m3"),
    "cuda:off-diagonal stencil weld": ("offdiag_magnetic_2d",),
    "cuda:folded off-diagonal stencil weld": ("folded_offdiag_magnetic_2d",),
    # --- 2026-09-17, THE TWO COMPLEX STENCIL WELDS. One route case each, which is the
    # narrowness their axes rows are written to: every axis below is pinned from a
    # single lift, and a case that does not lift on the campaign host makes
    # recut_driver_dispatch_record refuse the WHOLE record rather than this row.
    # THE FOLDED COMPLEX WELD'S CASE IS THE Hz SIBLING, not the Ez builder the Triton
    # row and the magnetic pair share: the weld declares CARRIES_DEPOSIT_REPAIR False,
    # an electric point source is injected on its seam, and its own predicate refuses
    # there (2026-09-17_allpaths read VACUOUS-PASS on the Ez row for that reason).
    "cuda:folded complex off-diagonal stencil weld": ("folded_complex_offdiag_hz_2d",),
    # ``cuda:complex no-absorber off-diagonal stencil weld`` WAS HERE on
    # ``complex_no_pml_offdiag`` for one campaign (2026-09-17_singles) and came out
    # the same day: the arm dispatches by preference (EXACT-ABSORBED-ARRAY-SLOT,
    # 19/19 arrays identical), but its arbitration-default-precedence control reads
    # REFUSED, because under the shipped order the Triton table selects its PENDING
    # ``complex no-PML off-diagonal`` at update_E and rung 7a sends the WHOLE step to
    # the array path. Same property of the incumbent that withdrew complex_no_pml_3d
    # on 2026-09-14; it sits in CUDA_PENDING_DEVICE_GATE_ARMS with that reason and
    # returns the day fastpath.PENDING_DEVICE_GATE_ARMS releases the Triton arm.
    "cuda:complex beta fused magnetic pair": (
        "complex_beta_2d", "folded_complex_beta_bloch_2d"),
    "cuda:complex beta fused electric pair": (
        "complex_beta_2d", "folded_complex_beta_bloch_2d"),
    "cuda:BFAST fused magnetic pair": ("bfast_1d",),
    "cuda:BFAST fused electric pair": ("bfast_1d",),
    # --- Tier 3, 2026-09-13: the one arm this tier RELEASES rather than widens.
    # ``absorber_1d`` carries an ``mp.Absorber``, so it lifts ``pml_active`` False --
    # the value the shared envelope refused until this tier moved that axis onto the
    # arms.
    #
    # ``material_dispersion_0d`` WAS THE SECOND CASE AND IS NOT ANY MORE (2026-09-14).
    # Its zero-extent (1, 1, 1) cell makes every non-vacuity control of the driver
    # route gate vacuous -- no monitor to compare a flux on, no curl for the magnetic
    # half-step to act on, an inactive layer whose constitutive is a pure overwrite,
    # and a mutated wall plane that IS the deposit cell the repair recomputes (cells
    # mutated but not repaired: ZERO of 1, against 399 of 400 on ``absorber_1d``). A
    # case that can only pass is not evidence, so the two axes it alone drove are
    # pinned below to what ``absorber_1d`` drove rather than left open on its
    # authority. ``parity/meep_gpu/gate_dispatch_fused_route.DRIVE_CUDA`` carries the
    # measurement.
    "cuda:no-absorber three-slot dispersive weld": ("absorber_1d",),
    # --- 2026-09-14. The complex twin of the row above. Released when the N2
    # re-gate seeded its weld, withdrawn the same evening when the route leg refused
    # on the incumbent table's pending arms, and RESTORED when the release decision
    # released those arms. ``complex_no_pml_3d`` is the ONLY route case this arm
    # has, which is why every axis on its row below is pinned rather than left open,
    # and it dispatches on the expansion-probe leg alone: the arm stores complex
    # fields, so its licence has to come from the probe record.
    "cuda:complex no-absorber three-slot weld": ("complex_no_pml_3d",),
}

#: Which LEG of the campaign drives each released arm. Read by
#: ``recut_driver_dispatch_record.py --backend cuda``, which walks
#: ``(arm, case, leg)`` rather than ``(arm, case)``: a complex arm that appeared in
#: the shipped leg's ``arms_driven`` would mean the expansion licence had been
#: consumed without the probe, which is the one thing that leg exists to keep apart.
CUDA_RELEASED_ARM_LEGS: Mapping[str, Tuple[str, ...]] = {
    arm: (("cuda_shipped_expansion_probe",)
          if arm in ("cuda:complex fused magnetic pair",
                     "cuda:complex fused electric pair",
                     "cuda:folded complex fused magnetic pair",
                     "cuda:folded complex fused electric pair",
                     "cuda:cylindrical fused magnetic pair",
                     "cuda:cylindrical fused electric pair",
                     "cuda:complex beta fused magnetic pair",
                     "cuda:complex beta fused electric pair",
                     # 2026-09-14. Complex storage, so its expansion licence comes
                     # from the probe record and nowhere else; listed here for the
                     # reason stated above, that a complex arm turning up in the
                     # shipped leg's ``arms_driven`` would mean the licence had been
                     # consumed without the probe.
                     "cuda:complex no-absorber three-slot weld",
                     # 2026-09-17. The two complex off-diagonal stencil welds, for
                     # the same reason: both store complex fields, so both take
                     # their expansion licence from the probe record, and both are
                     # driven on the probe leg alone.
                     "cuda:folded complex off-diagonal stencil weld")
          else ("cuda_shipped", "cuda_harness_keep"))
    for arm in CUDA_RELEASED_FUSED_ARMS
}

#: THE AXES EVERY RELEASED CUDA ARM WAS DRIVEN ON WITH THE SAME VALUE — AND SINCE
#: TIER 3 (2026-09-13) THERE ARE NONE. The table is deliberately EMPTY, which is a
#: statement rather than an absence: this release now spans real and complex storage,
#: Cartesian and Dcyl grids, lossless and conductive media, one and two and three
#: dimensions, a special-kz beta AND — the axis that emptied it — absorbing and
#: NON-absorbing cells. Not one configuration fact is true of every released arm, so
#: every axis is a PER-ARM row below, because which VALUE was driven depends on which
#: arm was driven on it: the same rule ``fastpath.FUSED_RELEASE_ENVELOPE`` states for
#: the Triton table, taken to its end.
#:
#: WHAT AN EMPTY SHARED TABLE DOES AND DOES NOT MEAN. :func:`fused_release_reasons`
#: keeps exactly one clause — an UNREADABLE run shape refuses everything — and
#: otherwise returns no reason, so the whole admission rests on
#: :data:`CUDA_FUSED_RELEASE_ARM_AXES` plus the rule that an arm with no row there
#: is refused by name. That is why ``pml_active`` did not simply disappear when it
#: moved: it is appended to EVERY arm row below with the value that arm's own cases
#: carried, so an arm which has only ever run inside an absorber still refuses a
#: no-absorber grid, and the only arm that answers False is the one whose two cases
#: lift False. An axis deleted from here without being re-stated per arm would have
#: been a widening of every released arm at once, which is the one move this pair of
#: tables exists to make impossible.
CUDA_FUSED_RELEASE_ENVELOPE: Tuple[Tuple[str, Any, str], ...] = ()

#: THE AXES ONE ARM'S OWN CASES DROVE. An axis is pinned for an arm when that arm's
#: case list drove exactly one of its values; an axis its cases drove both values of
#: is deliberately absent (that is what ``cuda:fused magnetic pair``'s missing
#: ``conductivity`` and ``dimensions`` rows mean — its fourteen cases span 1-D, 2-D
#: and 3-D, lossless and conductive, dispersive and not). ``susceptibilities`` is
#: the counter-example and the reason the rule is stated as a COUNT of driven values
#: rather than "the axis varies": that arm's cases drive four pole counts and the row
#: is the frozenset of exactly those four.
#:
#: AN ARM WITH NO ROW FAILS CLOSED, exactly as on the Triton table — and since Tier 3
#: emptied the shared envelope above, that rule carries the whole release: a label
#: with no row here would be admitted on every configuration the run shape can read,
#: which is the over-admission this pair of tables exists to prevent. Every arm's row
#: therefore ends with a ``pml_active`` statement of its own.
#:
#: ``nonlinearity`` IS PINNED FALSE ON ``cuda:fused magnetic pair`` even though the
#: family carries a welded nonlinear cell (2026-09-01_nonlinear) and declares a
#: second absorb row for it. The WELD is about the kernel; this table is about the
#: DRIVER ROUTE, and no route case has driven a nonlinear grid through the seam. The
#: row costs the board 2 of the family's 130 instances and is what makes
#: ``nonlinear_1d`` a Tier 2 item rather than an assumption.
CUDA_FUSED_RELEASE_ARM_AXES: Mapping[str, Tuple[Tuple[str, Any, str], ...]] = {
    "cuda:fused magnetic pair": (
        ("complex_storage", False,
         "all fourteen cases carry real storage; the complex magnetic pair is a "
         "separate product released on its own case"),
        ("cylindrical", False,
         "no Dcyl case drove this arm; the cylindrical grid has its own pair"),
        ("bfast", False, "no BFAST case drove this arm"),
        ("beta", 0, "no special-kz case drove this arm"),
        # AN INTEGER AXIS IS NOT BINARY, and the 2026-09-13 CUDA verifier found this
        # arm crediting the five-pole TestLoadDump rows and the six-pole
        # stochastic_emitter rows -- all DIAGONAL -- on evidence of 0 and 1. The row
        # is an ENUMERATION of the counts cases actually carried, never an open axis:
        # Tier 2's cases drove 0 poles (the PML and off-diagonal cases) and 1
        # (dispersive_2d, folded_dispersive_2d), and Tier 3 adds 5 and 6 because it
        # adds a case at each -- dispersive5_2d and dispersive6_2d, one Lorentzian
        # list apart from each other and from nothing else. A diagonal grid at 2, 3
        # or 4 poles is still refused rather than credited, and unpinning the axis
        # "since the seam does not read chi" is exactly the move the verifier caught.
        # That the B seam does not read chi is why four counts can bound the axis at
        # all -- it is a coverage bound matching Metal's fused magnetic B/H pair, not
        # a claim the kernel varies with poles; the OFF-diagonal side is stricter (its
        # cases drove 0 poles only) and is CUDA_OFF_DIAGONAL_CORNERS's
        # susceptibilities row, which Tier 3 did NOT widen: the one-pole off-diagonal
        # case it would have rested on does not lift (see the dispersive off-diagonal
        # polarization pair's pending reason).
        ("susceptibilities", frozenset({0, 1, 5, 6}),
         "the diagonal cases drove 0 and 1 poles on device, and dispersive5_2d and "
         "dispersive6_2d lifted off-device 2026-09-13 to susceptibilities 5 and 6 "
         "on a 2-D real diagonal PML grid (Tier 3: typed before the campaign drove "
         "them); no case drove this arm at 2, 3 or 4, and the off-diagonal side is "
         "bounded to 0 by the corner"),
        ("nonlinearity", False,
         "nonlinear_1d drove this arm byte-identically on 2026-09-13 (its nonlinear "
         "cell is welded), but the row is KEPT as the campaign's envelope witness: "
         "with the off-diagonal row gone it is the one axis on which this arm's "
         "predicate admits and the release refuses (ENVELOPE_ROWS['cuda'])"),
        ("pml_active", True,
         "every case that drove this arm carries a PML; the no-absorber curl is a "
         "separate product with its own weld"),
        # ``off_diagonal_epsilon`` is deliberately ABSENT: the arm drives both values,
        # and the True side is bounded by CUDA_OFF_DIAGONAL_CORNERS rather than a row.
    ),
    "cuda:fused electric pair": (
        ("complex_storage", False, "all five cases carry real storage"),
        ("cylindrical", False, "no Dcyl case drove this arm"),
        ("bfast", False, "no BFAST case drove this arm"),
        ("beta", 0, "no special-kz case drove this arm"),
        ("conductivity", False,
         "conductive_2d gives its D seam to the conductive product, so no case "
         "drove THIS arm on a conductivity"),
        ("susceptibilities", 0,
         "the dispersive cases give their D seam to the three-slot weld"),
        ("nonlinearity", False, "no nonlinear case drove this arm"),
        ("off_diagonal_epsilon", False, "no off-diagonal case drove this arm"),
        ("pml_active", True, "all five cases carry a PML"),
    ),
    "cuda:three-slot dispersive weld": (
        # ``susceptibilities`` IS LEFT UNPINNED, the declared exemption Triton
        # (fastpath.py) and Metal (metal_dispatch.py) also grant: this arm's E half
        # REQUIRES a susceptibility, its two route cases drove exactly one Lorentz
        # pole, and the family gate cuda_three_slot_dispersive_weld_2026-09-11_regate
        # drove passed corpus cases at arity [6, 6, 6] byte-identical -- the ADE
        # recurrence is certified per-pole to six. Pinning the axis at 1 would refuse
        # the five- and six-pole rows this weld is the composer's selection on
        # (TestLoadDump 2d, stochastic_emitter) rather than narrow anything; the
        # high-pole coverage is the family gate's, not a route case's.
        ("dimensions", 2, "dispersive_2d and folded_dispersive_2d are 2-D grids"),
        ("complex_storage", False, "both cases carry real storage"),
        ("cylindrical", False, "no Dcyl case drove this arm"),
        ("conductivity", False, "both cases are lossless"),
        ("off_diagonal_epsilon", False,
         "the off-diagonal dispersive seam has its own polarization product"),
        ("bfast", False, "no BFAST case drove this arm"),
        ("beta", 0, "no special-kz case drove this arm"),
        ("nonlinearity", False, "dispersive_2d and folded_dispersive_2d are linear"),
        ("pml_active", True,
         "both cases carry a PML; the no-absorber dispersive span is the separate "
         "weld Tier 3 released below"),
    ),
    "cuda:conductive fused electric pair": (
        ("dimensions", 2, "conductive_2d is a 2-D grid"),
        ("complex_storage", False, "conductive_2d carries real storage"),
        ("cylindrical", False, "conductive_2d is a Cartesian grid"),
        ("conductivity", True,
         "this arm IS the conductive product; on a lossless grid the ordinary "
         "fused electric pair is what the composer selects"),
        ("susceptibilities", 0, "conductive_2d carries no susceptibility"),
        ("folded", None, "conductive_2d carries no mirror"),
        ("off_diagonal_epsilon", False, "conductive_2d has a diagonal epsilon"),
        ("nonlinearity", False, "conductive_2d is linear"),
        ("pml_active", True, "conductive_2d carries a PML"),
    ),
    # THE ``dimensions`` ROW ON BOTH COMPLEX PAIRS IS AN ENUMERATION OF THREE CASES,
    # not a widening to "any Cartesian grid". Tier 3 adds complex_1d (a cell declared
    # 1-D, lifted to grid (1, 1, 480)), complex_3d_thinline (a (0, 0, L) cell DECLARED
    # 3-D, lifted to (1, 1, 275) -- the degenerate shape the four 1-D-looking corpus
    # rows actually carry) and complex_3d (a genuine 4x4x4 block, lifted to
    # (40, 40, 40)). The third buys no instance and is in the list for the reason the
    # corner tables exist: without it the 3-D point of this row would rest entirely on
    # grids one cell wide in x and y, which is evidence for a dimension COUNT and not
    # for a 3-D stencil.
    "cuda:complex fused magnetic pair": (
        ("dimensions", frozenset({1, 2, 3}),
         "bloch_2d is 2-D on device; complex_1d, complex_3d_thinline and complex_3d "
         "lifted off-device 2026-09-13 to dimensions 1, 3 and 3 (Tier 3: typed "
         "before the campaign drove them)"),
        ("complex_storage", True,
         "this arm REQUIRES complex storage; on a real grid the ordinary magnetic "
         "pair is what the composer selects"),
        # ``bloch`` is deliberately ABSENT: bloch_2d drives True and complex_nobloch_2d
        # drives False, every other axis of this row single-valued across both.
        ("cylindrical", False, "bloch_2d is a Cartesian grid"),
        ("folded", None, "the folded complex grid has its own pair"),
        ("beta", 0, "bloch_2d carries no special-kz beta"),
        ("off_diagonal_epsilon", False, "bloch_2d has a diagonal epsilon"),
        ("nonlinearity", False, "bloch_2d and complex_nobloch_2d are linear"),
        ("pml_active", True, "all five cases carry a PML"),
    ),
    "cuda:complex fused electric pair": (
        ("dimensions", frozenset({1, 2, 3}),
         "bloch_2d is 2-D on device; complex_1d, complex_3d_thinline and complex_3d "
         "lifted off-device 2026-09-13 to dimensions 1, 3 and 3 (Tier 3: typed "
         "before the campaign drove them)"),
        ("complex_storage", True, "this arm REQUIRES complex storage"),
        # ``bloch`` is deliberately ABSENT: bloch_2d drives True and complex_nobloch_2d
        # drives False, every other axis of this row single-valued across both.
        ("cylindrical", False, "bloch_2d is a Cartesian grid"),
        ("folded", None, "the folded complex grid has its own pair"),
        ("beta", 0, "bloch_2d carries no special-kz beta"),
        ("conductivity", False, "bloch_2d is lossless"),
        ("susceptibilities", 0, "bloch_2d carries no susceptibility"),
        ("off_diagonal_epsilon", False, "bloch_2d has a diagonal epsilon"),
        ("nonlinearity", False, "bloch_2d and complex_nobloch_2d are linear"),
        ("pml_active", True, "all five cases carry a PML"),
    ),
    # THE MAGNETIC HALF LOST ITS ``off_diagonal_epsilon`` ROW IN TIER 3 AND GAINED A
    # CORNER, which is the same move Tier 2 made on the ordinary magnetic pair and is
    # sound for the same reason: the B seam takes no inverse-epsilon operand, so an
    # off-diagonal chi1inv cannot reach its arithmetic. Dropping the row WITHOUT
    # registering a corner would have admitted nothing -- fused_release_arm_reasons
    # refuses an off-diagonal grid on an arm with no corner BY NAME -- and registering
    # one is what bounds the True side to the two cases that actually drove it. The
    # ELECTRIC half keeps its row: no off-diagonal case drove it, and on an
    # off-diagonal fold its D seam belongs to the folded complex off-diagonal stencil
    # weld, which is still pending its ledger entry.
    "cuda:folded complex fused magnetic pair": (
        ("dimensions", frozenset({2, 3}),
         "folded_complex_2d is 2-D on device; folded_complex_3d lifted off-device "
         "2026-09-13 to dimensions 3 on the triangular-lattice oblique cell, grid "
         "(8, 21, 126) (Tier 3: typed before the campaign drove it)"),
        ("complex_storage", True, "this arm REQUIRES complex storage"),
        ("folded", "required",
         "this arm IS the folded complex product; on an unfolded complex grid the "
         "plain complex pair is what the composer selects"),
        ("cylindrical", False, "folded_complex_2d is a Cartesian grid"),
        ("beta", 0, "folded_complex_2d carries no special-kz beta"),
        ("nonlinearity", False, "folded_complex_2d is linear"),
        ("pml_active", True, "every case that drove this arm carries a PML"),
        # ``off_diagonal_epsilon`` is deliberately ABSENT since Tier 3: the arm drives
        # both values, and the True side is bounded by CUDA_OFF_DIAGONAL_CORNERS.
        #
        # RESTORED 2026-09-17, and the row that stood here until now was a
        # CONTRADICTION of the two lines above it. The 2026-09-14 withdrawal pinned
        # the axis to False and left this comment in place; this round re-added the
        # arm's corner entry and its two release cases but not the pin's removal, so
        # the arm carried a corner bounding an off-diagonal grid AND a row refusing
        # every off-diagonal grid. The row wins, and the corner was dead text.
        #
        # THE CAMPAIGN SAID SO IN ONE LINE rather than being reasoned out: the CUDA
        # route's probe leg drove `folded_complex_offdiag_2d` and
        # `folded_complex_nobloch_offdiag_2d` to PASS-FUSED with
        # ``tables_dispatched: ['triton']`` -- the Triton incumbent held every slot
        # because the CUDA arm was never admitted -- and
        # ``fused_release_arm_reasons`` answered why by quoting this row's own text
        # back. The record recut then refused the whole campaign, which is the
        # ledger doing its job: a release claiming a case no leg's artifact shows
        # driven is exactly what it exists to catch.
        #
        # The same defect was made and caught on the Metal side this round, where
        # ``test_the_metal_off_diagonal_false_clause_is_gone_and_the_axis_is_per_arm``
        # names the rule and refused eleven arms typed this way. The CUDA table has
        # no such test; that is why this one survived to a device campaign.
    ),
    "cuda:folded complex fused electric pair": (
        ("dimensions", frozenset({2, 3}),
         "folded_complex_2d is 2-D on device; folded_complex_3d lifted off-device "
         "2026-09-13 to dimensions 3 (Tier 3: typed before the campaign drove it)"),
        ("complex_storage", True, "this arm REQUIRES complex storage"),
        ("folded", "required", "this arm IS the folded complex product"),
        ("cylindrical", False, "folded_complex_2d is a Cartesian grid"),
        ("beta", 0, "folded_complex_2d carries no special-kz beta"),
        ("conductivity", False, "folded_complex_2d is lossless"),
        ("susceptibilities", 0, "folded_complex_2d carries no susceptibility"),
        ("off_diagonal_epsilon", False,
         "the folded complex off-diagonal seam has its own stencil weld"),
        ("nonlinearity", False, "folded_complex_2d is linear"),
        ("pml_active", True, "both cases carry a PML"),
    ),
    # ``folded`` LEFT BOTH special_kz ROWS IN TIER 3. The axis has three spellings --
    # absent, "required", or equality on the fold's own free text -- so an arm driven
    # on a folded AND an unfolded case can only say so by carrying no row, which is
    # the documented meaning of a missing axis here. special_kz_2d is the unfolded
    # case and folded_special_kz_2d the folded one, and there is no folded special_kz
    # PRODUCT for the fold to belong to instead: this family serves both.
    #
    # A SIDE EFFECT WORTH STATING RATHER THAN DISCOVERING IN AN ARTIFACT. With
    # ``folded`` gone and no ``beta`` row on the magnetic half, that arm is released
    # on ordinary FOLDED 2-D real grids as well -- measured 2026-09-13: it appears in
    # the released set on dispersive5_2d and dispersive6_2d beside the ordinary
    # magnetic pair. The same was already true of unfolded grids (the arm has never
    # pinned beta, because it IS the beta product and its predicate requires one), so
    # what the lever changes is the fold, not the class; which arm the composer
    # SELECTS on such a grid is a different question from which the release admits.
    "cuda:special_kz fused magnetic pair": (
        ("dimensions", 2,
         "special_kz_2d and folded_special_kz_2d are both 2-D grids"),
        ("complex_storage", False,
         "special_kz is the REAL-storage beta product; a beta grid stored complex "
         "goes to the complex beta pair"),
        ("cylindrical", False, "special_kz_2d is a Cartesian grid"),
        ("bfast", False, "special_kz_2d is not a BFAST grid"),
        ("off_diagonal_epsilon", False, "special_kz_2d has a diagonal epsilon"),
        ("nonlinearity", False, "special_kz_2d is linear"),
        ("pml_active", True, "both cases carry a PML"),
        # ``folded`` is deliberately ABSENT since Tier 3: folded_special_kz_2d lifted
        # off-device 2026-09-13 to folded 'mirror plane on Y', beta 0.4, real storage,
        # grid (120, 62, 1), beside special_kz_2d's unfolded cell.
    ),
    "cuda:special_kz fused electric pair": (
        ("dimensions", 2,
         "special_kz_2d and folded_special_kz_2d are both 2-D grids"),
        ("complex_storage", False, "special_kz is the REAL-storage beta product"),
        ("cylindrical", False, "special_kz_2d is a Cartesian grid"),
        ("bfast", False, "special_kz_2d is not a BFAST grid"),
        ("conductivity", False, "special_kz_2d is lossless"),
        ("susceptibilities", 0, "special_kz_2d carries no susceptibility"),
        ("off_diagonal_epsilon", False, "special_kz_2d has a diagonal epsilon"),
        ("nonlinearity", False, "special_kz_2d is linear"),
        ("pml_active", True, "both cases carry a PML"),
        # ``folded`` is deliberately ABSENT since Tier 3, as on the magnetic half.
    ),
    "cuda:cylindrical real fused magnetic pair": (
        ("cylindrical", True,
         "this arm IS the Dcyl real product; on a Cartesian grid the ordinary pair "
         "is what the composer selects"),
        ("complex_storage", False,
         "the m = 0 Dcyl shape stores real fields; a complex Dcyl grid goes to the "
         "cylindrical complex pair"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "the cylindrical case is lossless"),
        ("susceptibilities", 0, "the cylindrical case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "the cylindrical case has a diagonal epsilon"),
        ("nonlinearity", False, "the cylindrical case is linear"),
        ("pml_active", True, "the cylindrical case carries a PML"),
    ),
    "cuda:cylindrical real fused electric pair": (
        ("cylindrical", True, "this arm IS the Dcyl real product"),
        ("complex_storage", False, "the m = 0 Dcyl shape stores real fields"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "the cylindrical case is lossless"),
        ("susceptibilities", 0, "the cylindrical case carries no susceptibility"),
        ("off_diagonal_epsilon", False, "the cylindrical case has a diagonal epsilon"),
        ("nonlinearity", False, "the cylindrical case is linear"),
        ("pml_active", True, "the cylindrical case carries a PML"),
    ),
    # --- Tier 2, 2026-09-13. Every row below was READ off the lifted run shape of
    # the case that drove it, on the GPU host, before it was typed. ---
    "cuda:cylindrical fused magnetic pair": (
        ("cylindrical", True,
         "this arm IS the Dcyl complex product; on a Cartesian grid the complex "
         "pair is what the composer selects"),
        ("complex_storage", True,
         "this arm steps complex64 word pairs only; the real m = 0 Dcyl shape has "
         "its own pair"),
        # ``m`` IS DELIBERATELY NOT PINNED, and soundness rests on KERNEL-CLASS
        # coverage rather than on every order being driven. The shape
        # dispatch_reachability derives from the census carries no ``m`` key, so a
        # pinned ``m`` would read "m did not read" and refuse every board row.
        # cyl_step_B_pml_complex has THREE m classes (m == 0; |m| == 1; |m| >= 2, the
        # near-axis zeroing); the five route cases drove m = -1, 0 (complex storage
        # forced), 1, 2 and 3 -- exercising all three -- and the family re-gate drove
        # |m| >= 2 at m in {-2, 2, 3} byte-identical. So an integer order the route did
        # not drive (perturbation_theory.py runs m = 5; dipole_in_vacuum_cyl_off_axis,
        # point_dipole_cyl and disc_extraction_efficiency sweep m up a while-loop) runs
        # the SAME certified |m| >= 2 code path with m a runtime scalar the array path
        # multiplies identically -- NOT "every order the corpus carries", which is
        # false. A non-integer m (the m = 1.2 adjoint test) is not in the census and is
        # not claimed here. The arm is bounded on complex storage and the cylindrical
        # grid, as its real m = 0 twin is bounded on real storage.
        ("bloch", False, "no cylindrical case carries a Bloch phase"),
        ("beta", 0, "no cylindrical case carries a special-kz beta"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "the cylindrical cases are lossless"),
        ("susceptibilities", 0, "the cylindrical cases carry no susceptibility"),
        ("nonlinearity", False, "the cylindrical cases are linear"),
        ("off_diagonal_epsilon", False, "the cylindrical cases have a diagonal epsilon"),
        ("bfast", False, "no cylindrical case is a BFAST grid"),
        ("pml_active", True, "all five cylindrical cases carry a PML"),
    ),
    "cuda:cylindrical fused electric pair": (
        ("cylindrical", True, "this arm IS the Dcyl complex product"),
        ("complex_storage", True, "this arm steps complex64 word pairs only"),
        # ``m`` IS DELIBERATELY NOT PINNED, and soundness rests on KERNEL-CLASS
        # coverage rather than on every order being driven. The shape
        # dispatch_reachability derives from the census carries no ``m`` key, so a
        # pinned ``m`` would read "m did not read" and refuse every board row.
        # cyl_step_B_pml_complex has THREE m classes (m == 0; |m| == 1; |m| >= 2, the
        # near-axis zeroing); the five route cases drove m = -1, 0 (complex storage
        # forced), 1, 2 and 3 -- exercising all three -- and the family re-gate drove
        # |m| >= 2 at m in {-2, 2, 3} byte-identical. So an integer order the route did
        # not drive (perturbation_theory.py runs m = 5; dipole_in_vacuum_cyl_off_axis,
        # point_dipole_cyl and disc_extraction_efficiency sweep m up a while-loop) runs
        # the SAME certified |m| >= 2 code path with m a runtime scalar the array path
        # multiplies identically -- NOT "every order the corpus carries", which is
        # false. A non-integer m (the m = 1.2 adjoint test) is not in the census and is
        # not claimed here. The arm is bounded on complex storage and the cylindrical
        # grid, as its real m = 0 twin is bounded on real storage.
        ("bloch", False, "no cylindrical case carries a Bloch phase"),
        ("beta", 0, "no cylindrical case carries a special-kz beta"),
        ("folded", None, "no folded Dcyl case has been driven"),
        ("conductivity", False, "the cylindrical cases are lossless"),
        ("susceptibilities", 0, "the cylindrical cases carry no susceptibility"),
        ("nonlinearity", False, "the cylindrical cases are linear"),
        ("off_diagonal_epsilon", False, "the cylindrical cases have a diagonal epsilon"),
        ("bfast", False, "no cylindrical case is a BFAST grid"),
        ("pml_active", True, "all five cylindrical cases carry a PML"),
    ),
    "cuda:off-diagonal stencil weld": (
        ("dimensions", 2, "offdiag_magnetic_2d is a 2-D grid"),
        ("complex_storage", False, "offdiag_magnetic_2d carries real storage"),
        ("cylindrical", False, "offdiag_magnetic_2d is a Cartesian grid"),
        ("folded", None,
         "this is the UNFOLDED weld; a mirror plane goes to its folded twin"),
        ("off_diagonal_epsilon", True,
         "this arm IS the off-diagonal product; on a diagonal grid the ordinary "
         "electric pair is what the composer selects"),
        ("conductivity", False, "offdiag_magnetic_2d is lossless"),
        ("susceptibilities", 0, "offdiag_magnetic_2d carries no susceptibility"),
        ("nonlinearity", False, "offdiag_magnetic_2d is linear"),
        ("bloch", False, "offdiag_magnetic_2d carries no Bloch phase"),
        ("beta", 0, "offdiag_magnetic_2d carries no special-kz beta"),
        ("bfast", False, "offdiag_magnetic_2d is not a BFAST grid"),
        ("pml_active", True, "offdiag_magnetic_2d carries a PML"),
    ),
    "cuda:folded off-diagonal stencil weld": (
        ("dimensions", 2, "folded_offdiag_magnetic_2d is a 2-D grid"),
        ("complex_storage", False, "folded_offdiag_magnetic_2d carries real storage"),
        ("cylindrical", False, "folded_offdiag_magnetic_2d is a Cartesian grid"),
        ("folded", "required",
         "this arm IS the folded weld; an unfolded off-diagonal grid goes to the "
         "plain stencil weld"),
        ("off_diagonal_epsilon", True, "this arm IS the folded off-diagonal product"),
        ("conductivity", False, "folded_offdiag_magnetic_2d is lossless"),
        ("susceptibilities", 0, "folded_offdiag_magnetic_2d carries no susceptibility"),
        ("nonlinearity", False, "folded_offdiag_magnetic_2d is linear"),
        ("bloch", False, "folded_offdiag_magnetic_2d carries no Bloch phase"),
        ("beta", 0, "folded_offdiag_magnetic_2d carries no special-kz beta"),
        ("bfast", False, "folded_offdiag_magnetic_2d is not a BFAST grid"),
        ("pml_active", True, "folded_offdiag_magnetic_2d carries a PML"),
    ),
    # 2026-09-17. Both rows are read off `fastpath._run_shape` on that case's own
    # lift (prefer_gpu=False, stock MEEP 1.33.0), not transcribed from the plain
    # welds above: these two differ from them on complex_storage AND on bloch, and
    # the no-absorber one on pml_active as well.
    "cuda:folded complex off-diagonal stencil weld": (
        ("dimensions", 2, "folded_complex_offdiag_hz_2d is a 2-D grid, (80, 62, 1)"),
        ("complex_storage", True,
         "this arm IS the complex weld; a real-storage fold goes to the plain "
         "folded stencil weld above"),
        ("cylindrical", False, "folded_complex_offdiag_hz_2d is a Cartesian grid"),
        ("folded", "required",
         "'mirror plane on Y'; an unfolded complex off-diagonal grid goes to the "
         "no-absorber weld below, which is the unfolded one"),
        ("off_diagonal_epsilon", True, "this arm IS the off-diagonal product"),
        ("bloch", True, "folded_complex_offdiag_hz_2d carries k_point (0.3, 0, 0)"),
        ("conductivity", False, "folded_complex_offdiag_hz_2d is lossless"),
        ("susceptibilities", 0,
         "folded_complex_offdiag_hz_2d carries no susceptibility"),
        ("nonlinearity", False, "folded_complex_offdiag_hz_2d is linear"),
        ("beta", 0, "folded_complex_offdiag_hz_2d carries no special-kz beta"),
        ("bfast", False, "folded_complex_offdiag_hz_2d is not a BFAST grid"),
        ("pml_active", True, "folded_complex_offdiag_hz_2d carries a PML"),
    ),
    "cuda:complex no-absorber off-diagonal stencil weld": (
        ("dimensions", 2, "complex_no_pml_offdiag is a 2-D grid, (25, 25, 1)"),
        ("complex_storage", True, "this arm IS the complex weld"),
        ("cylindrical", False, "complex_no_pml_offdiag is a Cartesian grid"),
        ("folded", None,
         "this is the UNFOLDED weld; a mirror plane goes to the folded complex "
         "stencil weld above"),
        ("off_diagonal_epsilon", True, "this arm IS the off-diagonal product"),
        ("bloch", True,
         "complex_no_pml_offdiag carries k_point (0.3892, 0.1597, 0)"),
        ("conductivity", False, "complex_no_pml_offdiag is lossless"),
        ("susceptibilities", 0, "complex_no_pml_offdiag carries no susceptibility"),
        ("nonlinearity", False, "complex_no_pml_offdiag is linear"),
        ("beta", 0, "complex_no_pml_offdiag carries no special-kz beta"),
        ("bfast", False, "complex_no_pml_offdiag is not a BFAST grid"),
        ("pml_active", False,
         "this arm IS the NO-ABSORBER weld: complex_no_pml_offdiag carries no "
         "boundary layer, which is the axis that separates it from the folded "
         "complex weld above"),
    ),
    # TIER 3 DROPPED BOTH THE ``folded`` AND THE ``bloch`` ROWS HERE, and the two go
    # together because ONE case carries both values. folded_complex_beta_bloch_2d --
    # the special_kz cell stored complex with an IN-PLANE k_x beside the z beta and a
    # mirror on Y -- lifted off-device 2026-09-13 to folded 'mirror plane on Y',
    # bloch True, beta 0.4, grid (120, 62, 1), while the released complex_beta_2d is
    # unfolded and lifts bloch False at the same beta. So each axis now has both of
    # its values driven, which is what a missing row means on this table, and the
    # refusal text the bloch row used to carry ("a z-only k_point lifts as beta, not
    # as a Bloch phase") described the CASE rather than the corpus rows, which carry
    # a z beta AND an in-plane phase at once. There is no folded complex-beta PRODUCT
    # for the fold to belong to instead: this family serves both foldings.
    "cuda:complex beta fused magnetic pair": (
        ("dimensions", 2,
         "complex_beta_2d and folded_complex_beta_bloch_2d are both 2-D grids"),
        ("complex_storage", True,
         "this arm IS the complex beta product; the real-storage beta grid goes to "
         "the special_kz pair"),
        # ``beta`` is deliberately ABSENT, as on the special_kz pairs: the arm IS
        # the beta product and its predicate requires beta != 0; both cases drove 0.4.
        ("cylindrical", False, "complex_beta_2d is a Cartesian grid"),
        ("conductivity", False, "complex_beta_2d is lossless"),
        ("susceptibilities", 0, "complex_beta_2d carries no susceptibility"),
        ("nonlinearity", False, "complex_beta_2d is linear"),
        ("off_diagonal_epsilon", False, "complex_beta_2d has a diagonal epsilon"),
        ("bfast", False, "complex_beta_2d is not a BFAST grid"),
        ("pml_active", True, "both cases carry a PML"),
    ),
    "cuda:complex beta fused electric pair": (
        ("dimensions", 2,
         "complex_beta_2d and folded_complex_beta_bloch_2d are both 2-D grids"),
        ("complex_storage", True, "this arm IS the complex beta product"),
        ("cylindrical", False, "complex_beta_2d is a Cartesian grid"),
        ("conductivity", False, "complex_beta_2d is lossless"),
        ("susceptibilities", 0, "complex_beta_2d carries no susceptibility"),
        ("nonlinearity", False, "complex_beta_2d is linear"),
        ("off_diagonal_epsilon", False, "complex_beta_2d has a diagonal epsilon"),
        ("bfast", False, "complex_beta_2d is not a BFAST grid"),
        ("pml_active", True, "both cases carry a PML"),
    ),
    "cuda:BFAST fused magnetic pair": (
        ("dimensions", 3,
         "bfast_1d is refl-angular's shape: a z-only cell DECLARED 3-D, which MEEP "
         "builds in 3-D with one cell in x and y (the lift reads dimensions 3)"),
        ("complex_storage", False,
         "BFAST is the REAL-storage broadband-angle product; a BFAST grid stored "
         "complex is refused by the family's own predicate"),
        ("bfast", True,
         "this arm IS the BFAST product; on an ordinary grid the ordinary pair is "
         "what the composer selects"),
        ("cylindrical", False, "bfast_1d is a Cartesian grid"),
        ("folded", None, "no folded BFAST case has been driven"),
        ("bloch", False, "bfast_1d's k_point is zero; BFAST carries the angle itself"),
        ("beta", 0, "bfast_1d carries no special-kz beta"),
        ("conductivity", False, "bfast_1d is lossless"),
        ("susceptibilities", 0, "bfast_1d carries no susceptibility"),
        ("nonlinearity", False, "bfast_1d is linear"),
        ("off_diagonal_epsilon", False, "bfast_1d has a diagonal epsilon"),
        ("pml_active", True, "bfast_1d carries a PML"),
    ),
    "cuda:BFAST fused electric pair": (
        ("dimensions", 3, "bfast_1d is a z-only cell declared 3-D (the lift reads 3)"),
        ("complex_storage", False, "BFAST is the REAL-storage product"),
        ("bfast", True, "this arm IS the BFAST product"),
        ("cylindrical", False, "bfast_1d is a Cartesian grid"),
        ("folded", None, "no folded BFAST case has been driven"),
        ("bloch", False, "bfast_1d's k_point is zero"),
        ("beta", 0, "bfast_1d carries no special-kz beta"),
        ("conductivity", False, "bfast_1d is lossless"),
        ("susceptibilities", 0, "bfast_1d carries no susceptibility"),
        ("nonlinearity", False, "bfast_1d is linear"),
        ("off_diagonal_epsilon", False, "bfast_1d has a diagonal epsilon"),
        ("pml_active", True, "bfast_1d carries a PML"),
    ),
    # --- Tier 3, 2026-09-13. Both rows below were read off an OFF-DEVICE lift, and
    # the arm is released ahead of the campaign that will drive it. ---
    "cuda:no-absorber three-slot dispersive weld": (
        ("pml_active", False,
         "this arm IS the no-absorber span; absorber_1d carries an mp.Absorber (a "
         "scalar conductivity, not a split-field PML), so it lifted pml_active False "
         "off-device 2026-09-13 -- on an absorbing grid the three-slot dispersive "
         "weld is what the composer selects"),
        # ``dimensions`` AND ``conductivity`` NARROWED 2026-09-14, and the narrowing
        # is the honest half of dropping ``material_dispersion_0d`` as a route case.
        # That case was the whole of this row's evidence for ``dimensions`` 2 and for
        # ``conductivity`` False, and its zero-extent cell makes every non-vacuity
        # control of the driver route gate vacuous (see CUDA_RELEASED_FUSED_ARMS
        # above). An axis whose second value was driven only by a case that could
        # not fail is an axis with one value driven, so both are pinned to what
        # ``absorber_1d`` drove. Re-widening either one wants a NON-DEGENERATE 2-D
        # no-boundary-layer case whose controls can arm.
        ("dimensions", 1,
         "absorber_1d lifted off-device 2026-09-13 to dimensions 1, grid "
         "(1, 1, 400); it is the only route case this arm has"),
        ("conductivity", True,
         "absorber_1d's mp.Absorber lifts conductivity True and no case drives the "
         "False side of this row any more"),
        ("complex_storage", False,
         "the case stores real fields; the complex no-absorber span has its own "
         "three-slot weld, which is still pending"),
        ("cylindrical", False, "the case is not a Dcyl grid"),
        ("folded", None, "the case carries no mirror"),
        ("bloch", False,
         "the case carries no k_point, so it lifted bloch False; an axis is "
         "pinned here because ONE value was driven, not because it looks incidental"),
        ("beta", 0, "the case carries no special-kz beta"),
        ("bfast", False, "the case is not a BFAST grid"),
        ("nonlinearity", False, "the case is linear"),
        ("off_diagonal_epsilon", False,
         "the case carries a diagonal epsilon; the off-diagonal no-absorber seam is "
         "a separate product"),
        #
        # ``susceptibilities`` IS LEFT UNPINNED under the SAME DECLARED EXEMPTION
        # THE ABSORBING THREE-SLOT WELD CARRIES, cited here by name rather than
        # inherited silently -- and it is the ONE axis on this row that the
        # 2026-09-14 narrowing does not touch, because its authority was never the
        # route cases. The E half of this span REQUIRES a susceptibility, its route
        # case drives 5 poles (absorber_1d), and the family gate
        # cuda_three_slot_dispersive_weld_2026-09-11_regate certified the ADE
        # recurrence per-pole to six on THIS ARM'S OWN LEGS rather than on its
        # absorbing twin's -- keep_no_pml and flush_no_pml, 14 product cases each
        # bit-identical over arities [1,1,1], [2,0,3], [2,2,2], [3,3,3], [5,5,5] and
        # [6,6,6], on fixtures that include corpus_absorber_wall_z and
        # corpus_material_dispersion. Pinning the axis at the two counts the route
        # drove would refuse the rows this weld is the composer's selection on
        # rather than narrow anything. The same legs are where ``complex_storage``
        # and ``off_diagonal_epsilon`` above are more than route-case coverage: the
        # product's own refusal leg requires it to DECLINE both by name.
    ),
    # --- 2026-09-14. The complex no-absorber three-slot weld. Every value below was
    # read off the SINGLE route case this arm has (``complex_no_pml_3d``, lifted
    # off-device 2026-09-13), so every axis is PINNED: one case drives one value,
    # and an axis is left open here only where both values were driven, which is
    # true of no axis on this row. The row was removed with the case when the route
    # leg refused on the incumbent table's pending arms and restored with it when
    # the release decision released them; the values never changed, because the lift
    # they were read off never did.
    "cuda:complex no-absorber three-slot weld": (
        ("pml_active", False,
         "this arm IS the complex no-absorber span; complex_no_pml_3d carries no "
         "boundary layer at all, so it lifted pml_active False. Restated per arm "
         "because the shared envelope above is empty -- an arm whose only case ran "
         "without a PML must keep refusing an absorbing grid"),
        ("dimensions", 3,
         "complex_no_pml_3d lifted to dimensions 3, grid (23, 21, 27); it is the "
         "only route case this arm has"),
        ("complex_storage", True,
         "the case forces complex storage, which is the whole of what separates "
         "this weld from the real-storage no-absorber three-slot weld above"),
        ("conductivity", True,
         "the case carries a conductivity and no case drives the False side"),
        ("susceptibilities", 1,
         "the case carries ONE Lorentz pole. PINNED rather than left open under the "
         "absorbing weld's declared exemption above: that exemption rests on a "
         "family gate which certified the ADE recurrence per-pole across six "
         "arities, and this arm's gate certifies two kernels "
         "(three_slot_no_pml_complex and three_slot_no_pml_complex_conductive) "
         "rather than an arity sweep, so there is no per-pole evidence here to "
         "inherit. Widening it wants a multi-pole complex no-absorber case"),
        ("bloch", True,
         "the case carries a k_point, so it lifted bloch True; pinned because ONE "
         "value was driven, not because it looks incidental"),
        ("cylindrical", False, "the case is not a Dcyl grid"),
        ("folded", None, "the case carries no mirror"),
        ("beta", 0, "the case carries no special-kz beta"),
        ("bfast", False, "the case is not a BFAST grid"),
        ("nonlinearity", False, "the case is linear"),
        ("off_diagonal_epsilon", False,
         "the case carries a diagonal epsilon; complex_no_pml_offdiag is the "
         "off-diagonal case of this span and its arm is still pending"),
    ),
}

#: THE OFF-DIAGONAL CORNER OF EVERY ARM WHOSE ROW LEAVES THAT AXIS OPEN. The rule is
#: the Metal table's (``metal_dispatch.METAL_OFF_DIAGONAL_CORNERS``): an arm that
#: drives BOTH values of ``off_diagonal_epsilon`` carries no row for it, but "both
#: values driven" is sound only where the True side was driven ACROSS the arm's other
#: open axes -- and it was not. The ordinary magnetic pair's fourteen cases span 1-D,
#: 2-D and 3-D, lossless and conductive, 0, 1, 5 and 6 poles, folded and not; its
#: OFF-DIAGONAL cases (offdiag_2d, offdiag_magnetic_2d, folded_offdiag_magnetic_2d,
#: pml_3d) sit at 2-D and 3-D, lossless, 0 poles, no Bloch phase, folded and not. So
#: the corner lists, per arm, every axis its row leaves open with the values its
#: off-diagonal cases actually carried, and ``fused_release_arm_reasons`` refuses an
#: off-diagonal grid outside it BY THE AXIS -- an off-diagonal grid over a
#: susceptibility (absorbed_power_density's shape on the Metal census) or with a
#: conductivity is refused, not admitted on evidence it never had. ``folded`` is not
#: listed: it is a presence axis, both sides were driven off-diagonal, and its value
#: is free text a corner cannot enumerate. An arm whose row leaves the axis open and
#: has NO corner here is refused on every off-diagonal grid, which is the fail-closed
#: direction. ``test_cuda_certified_fused_products`` pins the rule that a corner
#: lists exactly the open axes.
CUDA_OFF_DIAGONAL_CORNERS: Mapping[str, Mapping[str, FrozenSet[Any]]] = {
    "cuda:fused magnetic pair": {
        "dimensions": frozenset({2, 3}),
        "conductivity": frozenset({False}),
        # NOT WIDENED IN TIER 3, though the row above it was. The one-pole
        # off-diagonal case this corner would have rested on,
        # folded_offdiag_dispersive_2d, does not lift on a stock MEEP 1.33.0 -- see
        # the dispersive off-diagonal polarization pair's pending reason for the
        # measurement -- so an off-diagonal grid over a susceptibility stays refused
        # by this axis, on evidence rather than by omission.
        "susceptibilities": frozenset({0}),
        "bloch": frozenset({False}),
    },
    # TIER 3, 2026-09-13. The folded complex magnetic pair's row dropped its
    # ``off_diagonal_epsilon`` pin and this is the bound that replaces it. Both of
    # its off-diagonal cases were LIFTED OFF-DEVICE: folded_complex_offdiag_2d, a
    # cylinder on the fold plane of the released folded_complex_2d cell (dimensions
    # 2, bloch True, lossless, 0 poles, off_diagonal True, grid (80, 62, 1)) and
    # folded_complex_nobloch_offdiag_2d (the same shape unphased, two mirror planes,
    # bloch False, grid (81, 81, 1)). ``bloch`` is listed with BOTH values because
    # both were driven -- a corner that omitted it would admit the same set while
    # breaking the rule the table states -- and ``bfast`` because the arm's row
    # leaves it open while neither off-diagonal case is a BFAST grid.
    #
    # THE ENTRY THIS COMMENT DESCRIBES WAS WITHDRAWN WITH ITS CASES ON 2026-09-14 and
    # is RESTORED 2026-09-17 with them, unchanged: the bound was never what failed --
    # the arbitration under default precedence was, on an update_E arm that has since
    # been lifted. The comment above outlived the entry for three days, which is the
    # failure mode a corner table is most exposed to, so it is worth naming: a bound
    # that is described but not declared admits NOTHING, and reads to a later reader
    # as though it admits what the prose says.
    "cuda:folded complex fused magnetic pair": {
        "dimensions": frozenset({2}),
        "conductivity": frozenset({False}),
        "susceptibilities": frozenset({0}),
        "bloch": frozenset({True, False}),
        "bfast": frozenset({False}),
    },
}

#: THE REASON EACH UNRELEASED LABEL REFUSES THE WHOLE PLAN. Seventeen rows
#: (twenty-six until Tier 2 released eight on 2026-09-13 and Tier 3 one more), three
#: kinds, and the kind is what a reader needs first:
#:
#:   * SUPERSEDED (6) — the composer itself refuses these on every corpus row,
#:     because a strictly containing span admits there. Releasing one would dispatch
#:     a composition the board serves zero instances of.
#:   * NOT INSTALLABLE (7) — the H->D products declare ``INSTALLABLE = False`` in
#:     their own modules; the arbitration that produced that verdict is quoted in
#:     each module and re-read by ``fused_pairs._declared_uninstallable``.
#:   * NO ROUTE CASE (4) — buildable, installable, welded or seedable, and waiting
#:     on a driver-route case; the case each needs is named, and where that case is
#:     known NOT to be buildable the measurement says so rather than leaving the row
#:     reading as a scheduling matter. Tier 2 (2026-09-13) took eight rows out of
#:     this kind — the cylindrical complex pairs, the two off-diagonal stencil welds,
#:     the complex beta pairs and the BFAST pairs — and Tier 3 took the no-absorber
#:     three-slot dispersive weld, whose two cases lifted once ``pml_active`` moved
#:     off the shared envelope.
CUDA_PENDING_DEVICE_GATE_ARMS: Mapping[str, str] = {
    # --- reachable by preference only, 2026-09-17 ---
    "cuda:complex no-absorber off-diagonal stencil weld":
        "cuda:complex no-absorber off-diagonal stencil weld dispatches by preference "
        "on complex_no_pml_offdiag (2026-09-17_singles probe leg: PASS-FUSED, "
        "substitution EXACT-ABSORBED-ARRAY-SLOT, 19/19 arrays identical) but its "
        "arbitration-default-precedence control reads REFUSED: under the shipped "
        "order the Triton table selects its PENDING `complex no-PML off-diagonal` "
        "at update_E and rung 7a sends the whole step to the array path, so no "
        "by-default number can be conditioned on it. Returns with its route row "
        "when fastpath.PENDING_DEVICE_GATE_ARMS releases that Triton arm (its family "
        "byte gate passed on the device; the planner recertification and fingerprint "
        "entry are what is pending)",
    # --- superseded on every corpus row ---
    "cuda:dispersive fused electric pair":
        "cuda:dispersive fused electric pair is installable but SUPERSEDED on every "
        "corpus row: cuda_three_slot_dispersive_weld admits wherever this product "
        "does and spans step_D/update_E/update_P, which strictly contains this "
        "product's step_D/update_E, so the composer refuses the shorter span by name "
        "(fused_pairs._superseded_by_a_longer_span). The board serves 0 instances "
        "through it; releasing it would dispatch a composition nothing credits",
    "cuda:no-PML dispersive fused electric pair":
        "cuda:no-PML dispersive fused electric pair is SUPERSEDED on every corpus "
        "row by cuda_three_slot_no_pml_dispersive_weld, whose span strictly contains "
        "it; the board serves 0 instances through it",
    "cuda:complex no-absorber fused electric pair":
        "cuda:complex no-absorber fused electric pair is SUPERSEDED on every corpus "
        "row by cuda_three_slot_complex_no_pml_dispersive_weld, whose span strictly "
        "contains it; the board serves 0 instances through it",
    "cuda:fused polarization pair":
        "cuda:fused polarization pair claims update_E/update_P, and on every "
        "admitting row the three-slot dispersive weld already holds update_E; the "
        "composer refuses it as a neighbouring-seam claimant and the board serves 0 "
        "instances through it",
    "cuda:no-absorber fused polarization pair":
        "cuda:no-absorber fused polarization pair claims update_E/update_P and the "
        "no-absorber three-slot weld already holds update_E on every admitting row; "
        "the board serves 0 instances through it",
    "cuda:complex no-PML fused polarization pair":
        "cuda:complex no-PML fused polarization pair claims update_E/update_P and "
        "the complex no-absorber three-slot weld already holds update_E on every "
        "admitting row; the board serves 0 instances through it",
    # --- declared uninstallable in their own modules ---
    "cuda:fused H/D pair":
        "cuda:fused H/D pair declares INSTALLABLE = False in its own module: over "
        "the driver's step_B - update_H - step_D - update_E path launches are "
        "4 - (installed pairs), and an H->D span takes one slot from each released "
        "neighbour, so it can only TIE or LOSE. The kernel is welded; what is "
        "refused is the COMPOSITION",
    "cuda:cylindrical real fused H/D pair":
        "cuda:cylindrical real fused H/D pair declares INSTALLABLE = False: the "
        "cylindrical arithmetic re-prices the span in its favour, but no composition "
        "gate has counted device launches per step on a lifted Dcyl row with this "
        "product installed against both neighbours, and the composer's own rule "
        "counts launches as 4 - pairs",
    "cuda:cylindrical complex fused H/D pair":
        "cuda:cylindrical complex fused H/D pair declares INSTALLABLE = False, for "
        "the reason its real sibling does: the Dcyl re-pricing is unmeasured",
    "cuda:complex fused H/D pair":
        "cuda:complex fused H/D pair declares INSTALLABLE = False: on this cell the "
        "Cartesian algebra applies unchanged and both neighbours are released "
        "products",
    "cuda:special_kz fused H/D pair":
        "cuda:special_kz fused H/D pair declares INSTALLABLE = False: on this cell "
        "the Cartesian algebra applies unchanged and both neighbours are released "
        "products",
    "cuda:complex beta fused H/D pair":
        "cuda:complex beta fused H/D pair declares INSTALLABLE = False: on this cell "
        "the Cartesian algebra applies unchanged and both neighbours are released "
        "products",
    "cuda:conductive/BFAST fused H/D pair":
        "cuda:conductive/BFAST fused H/D pair declares INSTALLABLE = False: on both "
        "of its cells the span takes one slot from each released neighbour and can "
        "only tie or lose",
    # --- buildable and welded, waiting on a driver-route case ---
    "cuda:dispersive off-diagonal polarization pair":
        "cuda:dispersive off-diagonal polarization pair has no CUDA driver-route "
        "case, and Tier 3 measured that the case CANNOT BE BUILT on a stock MEEP "
        "rather than that nobody has built it. folded_offdiag_dispersive_2d — "
        "folded_offdiag_magnetic_2d with its cylinder made Lorentzian, the Hz "
        "source kept because this product spans update_E/update_P on the two "
        "components the off-diagonal rows couple — refuses to lift on MEEP 1.33.0: "
        "the chi1inv tensor is non-diagonal at the curved interface and this MEEP "
        "carries no fields.get_susceptibility_sigma, so the sigma cannot be read "
        "back. Both escapes the refusal offers were tried on 2026-09-13 and neither "
        "produces this shape: eps_averaging=False lifts and reads "
        "off_diagonal_epsilon False (subpixel averaging IS what writes the "
        "off-diagonal rows, so turning it off deletes the axis the case exists to "
        "drive), and moving the dispersion onto an axis-aligned block beside a "
        "NON-dispersive cylinder still refuses, because the check is over the cell "
        "rather than per region. The case needs a MEEP built with MEEP_SIGMA_PATCH=1 "
        "(parity/meep_gpu/build_meep_133_macos.sh, macOS-only today); until one is "
        "on the campaign host neither this arm nor the one-pole widening of "
        "CUDA_OFF_DIAGONAL_CORNERS['cuda:fused magnetic pair'] has evidence",
}


def fused_release_reasons(shape: Mapping[str, Any]) -> Tuple[str, ...]:
    """Why this configuration is outside what the CUDA route gate measured.

    Fails closed on an unread shape for the reason its Triton twin does: the
    decision it feeds is an ADMISSION, and admitting on an unread fact asserts one.

    SINCE TIER 3 THE UNREADABLE CLAUSE IS THE ONLY ONE THIS FUNCTION HAS, because
    :data:`CUDA_FUSED_RELEASE_ENVELOPE` is empty: no configuration fact is true of
    every released CUDA arm any more. An empty return here therefore means "nothing
    SHARED refuses this shape" and never "this shape is admitted" —
    :func:`released_fused_arms` still asks every arm's own row, and an arm without
    one is refused by name.
    """
    from . import fastpath  # noqa: PLC0415

    if "unreadable" in shape:
        return (f"the run shape did not read ({shape['unreadable']}); admitting a "
                "released CUDA arm on a configuration this record cannot describe "
                "would be asserting the configuration",)
    return fastpath._axis_reasons(shape, CUDA_FUSED_RELEASE_ENVELOPE)  # noqa: SLF001


def fused_release_arm_reasons(shape: Mapping[str, Any], arm: str) -> Tuple[str, ...]:
    """Why THIS CUDA arm is outside what its own cases drove. Fails closed with no row."""
    from . import fastpath  # noqa: PLC0415

    label = namespaced(arm)
    axes = CUDA_FUSED_RELEASE_ARM_AXES.get(label)
    if axes is None:
        return (f"{label} has no per-arm axis row "
                "(CUDA_FUSED_RELEASE_ARM_AXES), so the configurations its own gate "
                "cases drove have never been written down; the shared envelope "
                "alone cannot admit it",)
    reasons = list(fastpath._axis_reasons(shape, axes))  # noqa: SLF001
    # THE OFF-DIAGONAL CORNER, for an arm whose row leaves that axis open: see
    # CUDA_OFF_DIAGONAL_CORNERS. A row that pins the axis has already answered above.
    if shape.get("off_diagonal_epsilon") and not any(
            axis == "off_diagonal_epsilon" for axis, _value, _why in axes):
        corner = CUDA_OFF_DIAGONAL_CORNERS.get(label)
        if corner is None:
            reasons.append(
                f"off_diagonal_epsilon={shape.get('off_diagonal_epsilon')!r}; no case "
                f"drove {label} on a grid whose chi1inv carries off-diagonal rows")
        else:
            for axis, driven in corner.items():
                if axis not in shape:
                    reasons.append(
                        f"off_diagonal_epsilon=True and {axis} did not read, so "
                        f"{label}'s off-diagonal corner cannot be checked")
                elif shape[axis] not in driven:
                    reasons.append(
                        f"off_diagonal_epsilon=True at {axis}={shape[axis]!r}, and "
                        f"{label} was driven off-diagonal only at {axis} in "
                        f"{sorted(driven, key=repr)!r}")
    return tuple(reasons)


def released_fused_arms(shape: Mapping[str, Any]) -> Tuple[str, ...]:
    """The NAMESPACED CUDA labels admitted here: shared envelope AND per-arm axes."""
    if fused_release_reasons(shape):
        return ()
    return tuple(sorted(arm for arm in CUDA_RELEASED_FUSED_ARMS
                        if not fused_release_arm_reasons(shape, arm)))


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

_LEDGER: Dict[str, Mapping[str, Any]] = {}


def _read_json(name: str) -> Mapping[str, Any]:
    """A ``cuda_kernels`` record, read once per process. Never raises.

    Serves ``certification.json``. The weld ledger beside it is NOT read here: it
    goes through :func:`fastpath._fingerprints`, the one reader admission uses (see
    :func:`fingerprints`).
    """
    if name not in _LEDGER:
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cuda_kernels", name)
            with open(path, "r", encoding="utf-8") as handle:
                _LEDGER[name] = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - an unreadable record is not fatal
            _LEDGER[name] = {"_unreadable": repr(exc)}
    return _LEDGER[name]


def fingerprints() -> Mapping[str, Any]:
    """``cuda_kernels/fingerprints.json`` — the WELD ledger, one entry per campaign.

    Read through :func:`fastpath._fingerprints`, the reader device admission uses,
    and NOT through this module's own cache. Two caches of one file can disagree as
    soon as either is replaced, and admission derives the architectures this table
    may run on from this ledger while :func:`certification_for` quotes the run for
    the architecture it is running on: from two caches, a plan admitted on an
    architecture could report that no run certified it there.
    """
    from . import fastpath  # noqa: PLC0415

    return fastpath._fingerprints(fastpath.CUDA_TABLE)


def certification() -> Mapping[str, Any]:
    """``cuda_kernels/certification.json`` — the NARRATIVE record the welds cite."""
    return _read_json("certification.json")


def certification_for(arm: str, *, capability: Optional[str] = None) -> Dict[str, Any]:
    """What certified this CUDA arm, in the shape ``fastpath._certification_for`` returns.

    THE ENTRY SHAPE IS THE SHARED ONE ON PURPOSE — family, gate, the gate's own
    recorded facts, and a MISS that names itself. A reader comparing a Triton arm and
    a CUDA arm in the same artifact is comparing two rows of one table, and a second
    shape would make "this arm cites no record" and "this arm's record has different
    fields" indistinguishable.

    NO RE-CERT SWEEP IS QUOTED. ``RECERT_SWEEP_KEY`` is the TRITON ledger's key;
    this ledger carries no such block, and quoting that sweep's verdict beside a
    CUDA gate would be crediting one campaign with another's re-run.
    """
    label = namespaced(arm)
    family, gate = CUDA_ARM_CERTIFICATION.get(label, ("unmapped", "unmapped"))
    entry: Dict[str, Any] = {
        "family": family,
        "gate": gate,
        "certification_policy": SUBNORMAL_POLICY,
        "step_budget": None,
    }
    record = fingerprints().get(gate)
    if not isinstance(record, Mapping):
        entry["gate_record"] = (
            f"no record under {gate!r} in cuda_kernels/fingerprints.json; this arm "
            "names no certification this module can quote. A label in "
            "CUDA_PENDING_DEVICE_GATE_ARMS is the expected case; a RELEASED label "
            "here means the campaign has not seeded its weld yet, and "
            "CUDA_ARM_CERTIFICATION names which writer cuts it")
    else:
        # THE RUN THAT CERTIFIED THIS DEVICE'S ARCHITECTURE, for the reason
        # ``fastpath._certification_for`` gives: this ledger keeps one run per compute
        # capability, and quoting another one's host beside this plan would misdescribe
        # the evidence.
        from . import fastpath  # noqa: PLC0415

        live = fastpath.live_capabilities(record)
        entry["capabilities_live"] = list(live)
        run: Mapping[str, Any] = {}
        if capability is None:
            entry["run_record"] = (
                "this host's compute capability could not be read, so no run of this "
                "gate is quoted; the capabilities it has live runs for are above")
        elif capability in live:
            run = record[fastpath.RUNS][capability]
            entry["capability"] = capability
        else:
            entry["run_record"] = (
                f"this gate has no live run on compute capability {capability}: its "
                f"live capabilities are {list(live)}. A plan that dispatched here was "
                f"admitted by the opt-in, not by a record")
        for key in ("recorded_utc", "host", "purpose", "records", "step_budget",
                    "status", "subnormal_policy", "legs", "kernels",
                    "kernel_module"):
            if key in run:
                entry[key] = run[key]
            elif key in record:
                entry[key] = record[key]
    if entry.get("step_budget") is None:
        entry["step_budget"] = (
            f"not stated per family by {gate}; see that record's legs and the "
            "campaign-wide certification_budget at the top of this artifact")
    return entry


#: The policy every CUDA weld's DISPATCHING leg was cut under. The ledger records
#: two legs per entry — ``meep_x86_flush`` and ``ieee_keep_ftz_stripped`` — and the
#: keep leg is the certified one, which is the same value the Triton table requires.
#: Kept here as a name so ``fastpath.TABLE_SUBNORMAL_POLICY`` and this module can be
#: pinned equal by test rather than agreeing by coincidence.
SUBNORMAL_POLICY = "keep"

#: The executors that must all attain that policy before a CUDA kernel may run. The
#: same row as the Triton table's: this table composes over a CuPy engine, the
#: kernels compile through NVRTC with ``-ftz`` stripped, and Triton is governed too
#: whenever it is importable because BOTH tables' plans can be in one step.
GOVERNED_EXECUTORS: Tuple[str, ...] = ("host", "cupy", "triton")


def validated_compute_capabilities() -> Tuple[str, ...]:
    """The GPU architectures this table's cited gates ran on. DERIVED, never typed.

    THE INTERSECTION OF THE CITED WELDS' LIVE RUNS, through the same reader the Triton
    table uses (``fastpath.capability_report``): an architecture this table claims is
    one that EVERY family it dispatches has a live run on, because an arm can be served
    from any of them. It used to be the UNION of the capabilities named by the
    ``certification.json`` blocks the welds point at, plus the route record's own list,
    which had two defects: a re-cut of one narrative block admitted every arm of the
    table on that architecture, and the route record's list was read from a field no
    gate writes, so it was always empty.

    ``None`` from the report means the ledger could not be read, and rung 4b treats
    that as unknown. ``()`` is a readable ledger whose cited welds share no live
    capability, and it refuses every device.
    """
    from . import fastpath  # noqa: PLC0415

    # READ THROUGH ``fastpath``'s LEDGER CACHE, not this module's. Both read the same
    # file, so in a run they agree; but rung 4b asks ``fastpath.capability_admission``,
    # and two readers of one file mean a test (or a tool) that swaps one of them gets
    # a record whose declared list and whose verdict disagree. One reader, one answer.
    keys = {gate for _family, gate in CUDA_ARM_CERTIFICATION.values()}
    admitted = fastpath.table_capabilities(
        fastpath._fingerprints(fastpath.CUDA_TABLE), sorted(keys))  # noqa: SLF001
    return () if admitted is None else admitted


# ---------------------------------------------------------------------------
# Candidacy and licences
# ---------------------------------------------------------------------------

def cuda_candidate(record: Mapping[str, Any], xp: Any) -> Dict[str, Any]:
    """Is the hand-CUDA table a candidate on this host? A NAMED refusal either way.

    Asked only after the backend rung has established a CuPy engine, so the question
    left is whether this package can be REACHED: ``cuda_kernels`` is pure Python
    plus NVRTC sources and its arm table imports without ``cupy``, but a truncated
    checkout or a partially-installed package is a refusal rather than a crash.

    THE DEVICE IS NOT ASKED HERE. Rung 4b asks it per table, against
    :func:`validated_compute_capabilities`, and its three-valued rule (an unreadable
    device is unknown, not refused) belongs with the other device reads rather than
    duplicated in this function.
    """
    block: Dict[str, Any] = {"candidate": False, "refused_because": None}
    backend = getattr(xp, "__name__", None)
    if backend != "cupy":
        block["refused_because"] = (
            f"the engine's array module is {backend!r}; the hand-CUDA table "
            "launches through cupy.RawKernel and has nothing to launch on")
        return block
    try:
        from .cuda_kernels import arms as _arms  # noqa: PLC0415, F401
    except Exception as exc:  # noqa: BLE001 - an unreachable package is a refusal
        block["refused_because"] = (
            f"meep_gpu.cuda_kernels is not importable on this host ({exc!r})")
        return block
    block["candidate"] = True
    return block


#: Which expansion licences the unified complex-expansion probe can license, by the
#: registry key the CUDA composer reads them under.
COMPLEX_LICENCE_KEYS: Tuple[str, ...] = (
    "LICENSE_COMPLEX", "LICENSE_COMPLEX_NO_PML", "LICENSE_COMPLEX_OFFDIAG")


def licences_from_probe(probe: Any) -> Optional[Dict[str, Any]]:
    """The CUDA composer's ``licenses`` mapping, built from ONE probe record.

    THE SAME ARTIFACT LICENSES BOTH TABLES, and that is the point of reading it
    here: ``fastpath`` rung 4c has already judged the record against the policy this
    dispatch installs and has DROPPED it if it disagrees, so a refused artifact
    arrives here as a refusal value rather than as ``None``. What this function does
    is ask each licence function whether it licenses that record, exactly as
    ``cuda_predicate_battery._composer_licenses`` does off-device, and OMIT any key
    that cannot be licensed — an absent key makes the family refuse by its own name,
    which is a named refusal, whereas a key mapped to nothing would be a licence
    claim nobody made.

    Returns ``None`` when no key licenses, so the composer sees "no licence offered"
    rather than an empty mapping it might read as "every licence refused".
    """
    if probe is None:
        return None
    try:
        from .cuda_kernels import registry as _registry  # noqa: PLC0415
        from .triton_kernels import complex_fields as _complex  # noqa: PLC0415
        from .triton_kernels import folded_complex as _folded  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - an unreadable rule licenses nothing
        return None
    licences: Dict[str, Any] = {}
    for name, reader in (
            ("LICENSE_COMPLEX", getattr(_complex, "expansion_license", None)),
            ("LICENSE_COMPLEX_NO_PML", getattr(_complex, "expansion_license", None)),
            ("LICENSE_COMPLEX_OFFDIAG",
             getattr(_folded, "parity_expansion_license", None))):
        key = getattr(_registry, name, None)
        if key is None or reader is None:
            continue
        try:
            verdict = reader(probe)
        except Exception:  # noqa: BLE001 - a refusal is a refusal, not a crash
            verdict = None
        if verdict is None:
            continue
        # THE VERDICT ITSELF IS THE LICENCE, not a wrapper around it, and the
        # difference was silent. ``StepContext.license_for`` hands whatever is stored
        # here straight to the predicate, and that predicate reads ``arm``,
        # ``refusals``, ``expansion``, ``basis`` and ``policy_resolved`` OFF THIS
        # OBJECT (``cuda_kernels/coverage.py``). Storing ``{"arm": verdict, ...}``
        # put the whole verdict dict where the arm NAME belongs, so every complex
        # family refused with "the expansion licence names arm {…}, which is not one
        # of ['NAIVE', 'FMA_V1']" -- a correctly licensed run reading as an
        # unlicensed one. Measured 2026-09-11 on bloch_2d: four slots served by the
        # incumbent's complex singles and not one CUDA fused pair installed, on the
        # ONE leg those arms can dispatch on.
        #
        # THE OFF-DEVICE BATTERY HAD IT RIGHT ALL ALONG
        # (``cuda_predicate_battery._composer_licenses`` stores ``verdict``), which
        # is why the census credited these families and the device could not run
        # them: the composer and the instrument that measures its coverage disagreed
        # about the shape of one argument.
        #
        # AND THE SAME DISAGREEMENT HAD A SECOND HALF, measured 2026-09-13 and fixed
        # here. ``expansion_license`` never returns a ``patterns`` key -- it returns
        # ``probe_patterns``, the set it REQUIRED the record to classify, not the
        # record's own classification table -- while
        # ``complex_no_pml_kernels._ade_probe_refusal`` asks a question about the
        # RECORD: whether it classifies the orientation ``update_P`` launches (a
        # complex64 array times a Python float), which is outside the four patterns
        # the base licence rule reads. Looking for it in ``verdict["patterns"]`` and
        # then ``verdict["record"]["patterns"]``, it found neither and refused EVERY
        # record, seven-pattern ones included, with "the expansion licence carries no
        # readable pattern table". So the complex no-absorber families could not
        # install whatever artifact the run was given. The off-device battery again
        # had it right and again by one line
        # (``_complex_no_pml_license``'s ``verdict.setdefault("patterns",
        # record.get("patterns"))``); this is that line, on the composer's side, and
        # it is why ``MEEP_GPU_COMPLEX_EXPANSION_PROBE`` must point at a record that
        # classifies the fifth orientation -- a SEVEN-pattern unified expansion
        # record. Measured on one: with the table attached ``_ade_probe_refusal``
        # returns None and the licence binds FMA_V1 with no refusals; without it,
        # the same record refuses. A four-pattern record still refuses, and by a
        # DIFFERENT reason ("does not classify"), which is the half of the mismatch
        # that stays a refusal and should.
        #
        # The verdict is COPIED rather than mutated: it is the licence object the
        # predicate reads, and a probe record read once per process may be handed to
        # more than one reader.
        patterns = None
        try:
            patterns = probe.get("patterns")
        except Exception:  # noqa: BLE001 - a refusal value is not a record
            patterns = None
        if isinstance(verdict, dict) and patterns is not None and (
                verdict.get("patterns") is None):
            verdict = dict(verdict, patterns=patterns)
        licences[key] = verdict
    return licences or None


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------

#: MAY A WHOLE SECONDARY UNIT TAKE SLOTS A PRIMARY ARM HOLDS WHEN THAT ARM IS
#: PENDING? The plan's §5.2 phrase "hand-CUDA fills its refusals" has two readings
#: and this constant is which one ships.
#:
#: FALSE TODAY, WITH A MEASURED REASON. A pending primary arm refuses the WHOLE plan
#: at rung 7a — that is the standing rule, and it is what keeps an unmeasured
#: composition off the device. Yielding those slots to a secondary unit is a
#: DIFFERENT composition from either table's gates, and no leg has driven one: the
#: campaign's ``cuda_default_precedence`` leg is what would measure the first, on a
#: case where Triton selects a pending arm and CUDA has a launchable unit over the
#: same slots (``complex_no_pml_3d`` and the 3-D no-absorber corpus rows are the
#: candidates). This flips to True only when that leg reports PASS-FUSED with byte
#: identity, and flipping it is an edit to THIS file: two route re-gates and two
#: record re-cuts, and no Metal cost.
#:
#: WHAT FALSE COSTS, STATED SO THE NUMBER IS NOT MISREAD: under the shipped default
#: precedence no released CUDA product installs on any of the fourteen measured
#: route cases, because Triton's arms hold every slot. Every CUDA
#: ``served_in_dispatch`` number is therefore CONDITIONED on
#: ``MEEP_GPU_BACKEND_PREFERENCE=cuda`` or on a host with no validated Triton, and
#: the board publishes both numbers with that condition in the same sentence.
YIELD_PENDING_PRIMARY_SLOTS = False


class MergedStepPlan:
    """Two composers' plans in one mapping, with the table each slot came from.

    THE SHAPE IS A COMPOSER'S, not a new one: ``plans``/``selected``/``reasons`` are
    the three attributes ``fastpath`` reads off a ``TritonStepPlan`` or a
    ``CudaStepPlan``, so every rung below the merge — the null drop, completeness,
    the fold rung, the warm pass, ``_split_pairs``, ``_record_slots`` — works on the
    merged plan without knowing a merge happened. ``backends`` is the one field that
    is new, and nothing decides anything from it: it is the record's.

    ``reasons`` IS A REAL DICT because the warm pass mutates it
    (``step_plan.reasons.update(...)`` when a slot's kernel will not compile).
    """

    __slots__ = ("plans", "reasons", "selected", "backends")

    def __init__(self, plans: Mapping[str, Any], reasons: Mapping[str, Any],
                 selected: Mapping[str, str],
                 backends: Mapping[str, str]) -> None:
        self.plans: Dict[str, Any] = dict(plans)
        self.reasons: Dict[str, Any] = dict(reasons)
        self.selected: Dict[str, str] = dict(selected)
        self.backends: Dict[str, str] = dict(backends)

    @property
    def replaces(self) -> Tuple[str, ...]:
        order = step_order()
        return tuple(name for name in order if name in self.plans)

    def describe(self) -> str:
        covered = ", ".join(self.replaces) if self.replaces else "nothing"
        return f"MergedStepPlan(replaces=[{covered}])"

    def __repr__(self) -> str:
        return self.describe()


def _units(plans: Mapping[str, Any]) -> List[Tuple[int, List[str]]]:
    """Group a composer's plans into UNITS by transitive ``absorbed_by`` identity.

    A unit is the set of slots ONE plan object (or one fused product) occupies. The
    walk has to be transitive: a three-slot weld puts a ``_TripleHalfPlan`` in
    ``update_P`` that names the triple's LEADING half, which names the triple, so a
    single hop would split one product across two identities and the merge would
    adopt two thirds of it. ``fastpath._pair_identity`` is that walk, bounded, and it
    is shared rather than re-implemented so the merge and ``_split_pairs`` cannot
    disagree about what a product is.
    """
    from . import fastpath  # noqa: PLC0415

    order = step_order()
    position = {name: index for index, name in enumerate(order)}
    grouped: Dict[int, List[str]] = {}
    for name in sorted(plans, key=lambda n: position.get(n, len(position))):
        grouped.setdefault(
            fastpath._pair_identity(plans[name]), []).append(name)  # noqa: SLF001
    return [(identity, members) for identity, members in grouped.items()]


def _launchable(plan: Any) -> bool:
    """Can the object that does the work actually be RUN? Three-valued underneath.

    Read THROUGH ``absorbed_by`` exactly as ``CudaStepPlan.launchable`` reads it: the
    wrappers a fused install leaves in the absorbed slots declare nothing themselves,
    and reading them bare would report a launchable product as unlaunchable the
    moment it started carrying a deposit repair.

    AN ABSENT DECLARATION IS NOT A ``False``, and conflating the two is a real
    defect rather than a hypothetical: the hand-CUDA composer DECLARES ``launchable``
    on every plan it installs (a single arm with no established launch-argument
    resolution declares ``False``, which is what keeps this phase to fused products),
    while the Triton and Metal composers declare nothing at all and every plan they
    install runs. Reading the absence as ``False`` would refuse a released Triton
    pair the moment it was composed SECOND — under
    ``MEEP_GPU_BACKEND_PREFERENCE=cuda`` — and send a seam that dispatches today to
    the array path. Measured on the join probe before this clause existed.
    """
    if isinstance(plan, (list, tuple)):
        # ``update_P`` HOLDS A LIST ON THE TRITON TABLE, and it is the one slot that
        # does. Reading a list for ``run`` answers None and would drop the whole ADE
        # slot out of every merged plan — measured, on eight tests, the first time
        # the primary table was walked through this filter.
        return bool(plan) and all(_launchable(entry) for entry in plan)
    target = getattr(plan, "absorbed_by", None) or plan
    declared = getattr(target, "launchable", None)
    if declared is None:
        return callable(getattr(target, "run", None))
    return bool(declared)


#: The two seams a CUDA single may be adopted on, partner by partner. WHEN THIS TABLE
#: COMPOSES FIRST (MEEP_GPU_BACKEND_PREFERENCE=cuda) a single is never adopted alone:
#: until 2026-09-27, on pml_3d / offdiag_2d / offdiag_magnetic_2d /
#: folded_offdiag_magnetic_2d ``step_D`` 'PML' was launchable while ``update_E``
#: 'off-diagonal' was not -- a lone launchable step_D beside it would have blocked the
#: incumbent's whole D/E unit at the whole-displacement clause below and pushed
#: update_E to the array path, a changed composition on a released row rather than a
#: baseline for one. The three off-diagonal update_E families carry a launch since
#: 2026-09-27, so on those rows the D/E seam is now adopted WHOLE from this table; the
#: rule is unchanged and still decides every seam whose partner is not launchable
#: (the no-absorber, conductive, nonlinear and dispersive update_E families, and every
#: complex one).
#: CORRECTED 2026-09-25: the rule is closed over THIS table's own plans only.
#: :func:`_adoptable_single_slots` keeps a single when its partner is among the CUDA
#: table's own launchable, certified singles; it does not look at what an earlier table
#: already holds. So when this table composes SECOND and the primary holds ONE slot of a
#: seam, the slot clause refuses that half and the CUDA other half IS adopted alone
#: (reproduced 4 of 4 with real ``CudaSlotPlan``s,
#: ``results/cuda_hoist_next_2026-09-24/c1_single_arms_verify/probe_real_plans.json``).
#: It was not observed on ``dispatch_fused_route_cuda_2026-09-24_hoist``, the route this
#: count was taken on (the route of record since is ``_2026-09-25_roundb``, which ran
#: the same cases and legs on the Round B bytes and was not re-tallied for this): 19
#: paired seams and 0 lone singles on each of its three single-dispatching legs and no
#: CUDA single on its default-precedence leg (``route_tally.json`` beside the probe).
#: That leg's 6 half-filled seam legs (4 B/H and 2 D/E, of 40 plan-carrying legs,
#: ``half_held_scan.json``) are all no-PML cases (``no_pml_2d``, ``absorber_1d``), where
#: :func:`.cuda_kernels.coverage.covers_real_pml_curl` refuses and no certified seam is
#: offered. Whether to enforce the seam rule when this table is secondary is an open
#: behaviour decision FOR THE PML + ORDINARY SEAMS, and this note changes nothing
#: about them. The three off-diagonal ``update_E`` singles made launchable on
#: 2026-09-27 widened the population the question applies to -- a Triton ``step_D``
#: beside an unfilled ``update_E`` on an off-diagonal grid would have taken the CUDA
#: ``update_E`` alone, a two-table D/E seam no gate drove -- so for THEIR seam the
#: rule is closed against the merged plan (:data:`CROSS_TABLE_SEAM_REFUSED_SINGLES`),
#: as the FILL twins below are: both are new with the arm rather than a change to the
#: seam rule's standing behaviour.
SINGLE_ARM_SEAMS: Mapping[str, str] = {"step_B": "update_H", "update_H": "step_B",
                                       "step_D": "update_E", "update_E": "step_D"}

#: THE SINGLES WHOSE SEAM IS CLOSED AGAINST THE MERGED PLAN, not only against this
#: table's own plans. A CUDA single SEAM unit is refused by name when its partner slot
#: (:data:`SINGLE_ARM_SEAMS`) is already held by ANOTHER table and either the unit's
#: own label or the partner's CUDA-selected label is in this set -- so neither the
#: CUDA off-diagonal ``update_E`` beside a Triton ``step_D``, nor the CUDA ``step_D``
#: 'PML' beside a Triton ``update_E`` on that grid, can reach the device. The three
#: are the off-diagonal ``update_E`` families that gained a launch on 2026-09-27. No
#: gate drove a two-table D/E seam on them; the route campaign that certifies them
#: drives their seam only where this table holds both halves of it.
CROSS_TABLE_SEAM_REFUSED_SINGLES: FrozenSet[str] = frozenset({
    "cuda:off-diagonal", "cuda:folded off-diagonal", "cuda:dispersive off-diagonal"})

#: THE FILL SLOTS ARE ADOPTED AS A PAIR, one per family, and never one without the
#: other. Rung 6b of the seam refuses a folded run unless BOTH fill slots carry a
#: kernel, so a lone CUDA fill could only ever reach the device beside ANOTHER
#: table's fill on the other family -- a two-table fold no gate drove. Unlike
#: :data:`SINGLE_ARM_SEAMS` (see the note above it), this closure is ALSO taken
#: against the merged plan: a CUDA fill whose twin slot another table already holds
#: is refused by name even when its own slot is free. That rule is new with the arm
#: (2026-09-27) rather than a change to the seam rule's standing behaviour.
FILL_ARM_TWINS: Mapping[str, str] = {"fill_B": "fill_D", "fill_D": "fill_B"}


def _adoptable_single_slots(plans: Mapping[str, Any],
                            selected: Mapping[str, str]) -> Set[str]:
    """Slots holding a CUDA single the merge may adopt, closed under the seam partner.

    Three facts per slot, all read off the composer's own objects: ``launchable``
    (a declared resolver AND launcher, ``arms.CudaSlotPlan.launchable``), a callable
    ``run`` (the consult calls ``plan.run()`` at ``fastpath.FastPathPlan.dispatch``
    and an exception there propagates mid-step), and a :data:`CUDA_ARM_CERTIFICATION`
    row for the label, so the dispatch record can name the byte gate the single
    rides on. A slot whose partner fails any of the three is dropped with it.
    """
    single: Set[str] = set()
    for slot, plan in plans.items():
        label = selected.get(slot)
        if (slot in SINGLE_ARM_SEAMS and label is not None
                and not _is_fused_label(CUDA_TABLE, label)
                and namespaced(label) in CUDA_ARM_CERTIFICATION
                and _launchable(plan) and callable(getattr(plan, "run", None))):
            single.add(slot)
    return {slot for slot in single if SINGLE_ARM_SEAMS[slot] in single}


def _adoptable_fill_slots(plans: Mapping[str, Any],
                          selected: Mapping[str, str]) -> Set[str]:
    """Fill slots holding a CUDA fill the merge may adopt, closed under the twin.

    The single-arm facts, with the one the dispatch seam actually calls in place of
    ``run``: a fill slot's consult calls ``run_near`` and the far-pass consult
    ``run_far`` (``fastpath.FastPathPlan._dispatch_far_fill``), so a fill plan
    must expose BOTH as callables -- a plan with ``run`` alone would be driven as a
    fused near-and-far launch, which moves the far pass in front of the wall wipe
    and is not what the in-seam gate certified. A slot whose twin fails any fact is
    dropped with it.
    """
    fills: Set[str] = set()
    for slot, plan in plans.items():
        label = selected.get(slot)
        if (slot in FILL_ARM_TWINS and label is not None
                and not _is_fused_label(CUDA_TABLE, label)
                and namespaced(label) in CUDA_ARM_CERTIFICATION
                and _launchable(plan)
                and callable(getattr(plan, "run_near", None))
                and callable(getattr(plan, "run_far", None))):
            fills.add(slot)
    return {slot for slot in fills if FILL_ARM_TWINS[slot] in fills}


def merge_tables(tables_in_order: Sequence[Tuple[str, Any]],
                 pending_of: Mapping[str, Mapping[str, str]],
                 record: Dict[str, Any]) -> MergedStepPlan:
    """Compose two tables into one plan by WHOLE UNITS, recording every refusal.

    THE PRIMARY GOES IN VERBATIM. Its plans, its labels, its reasons — nothing about
    the incumbent table changes when a second one is present, which is what makes
    "add a table" a change that cannot move a released composition.

    A SECONDARY UNIT IS ADOPTED ONLY IF ALL FOUR HOLD:

    * it is a FUSED PRODUCT of that table, or a certified launchable single SEAM of
      it (:data:`SINGLE_ARM_SEAMS`, :func:`_adoptable_single_slots`; on the
      off-diagonal D/E seam its partner must not be held by another table,
      :data:`CROSS_TABLE_SEAM_REFUSED_SINGLES`), or a certified launchable FILL TWIN
      of it (:data:`FILL_ARM_TWINS`, :func:`_adoptable_fill_slots`, whose twin must
      not be held by another table); any other single arm plan is left where it is,
      and the reason is recorded;
    * every slot of it is LAUNCHABLE, read through ``absorbed_by``;
    * every slot it wants is either unfilled in the merged plan, or held by a
      primary arm that table lists as PENDING and
      :data:`YIELD_PENDING_PRIMARY_SLOTS` is True;
    * every primary UNIT it would displace is displaced WHOLE. Half a primary pair
      surviving is the one composition the seam cannot guard (``_split_pairs``
      refuses it at plan time), and a merge that produced it would be manufacturing
      the shape rather than catching it.

    A PAIR IS NEVER SPLIT IN EITHER DIRECTION, and the refusal is by name: the
    record carries which slots were held and by which table's label, so a reader of
    a default-precedence artifact can see the arbitration rather than infer it from
    an absence.
    """
    if not tables_in_order:
        return MergedStepPlan({}, {}, {}, {})
    primary_name, primary = tables_in_order[0]
    plans: Dict[str, Any] = {}
    selected: Dict[str, str] = {}
    reasons: Dict[str, Any] = {}
    backends: Dict[str, str] = {}

    arbitration = record.setdefault("arbitration", {})
    refused: Dict[str, Dict[str, str]] = arbitration.setdefault("refused", {})
    secondary_reasons: Dict[str, Dict[str, Any]] = arbitration.setdefault(
        "secondary_reasons", {})

    # THE PRIMARY GOES THROUGH THE SAME LOOP, and the two clauses that make it
    # different are stated rather than implied. The SLOT clause is vacuous for it
    # (nothing is filled yet, so no unit is ever blocked); the FUSED-OR-SEAM clause
    # is keyed on the TABLE and not on the position, because it is a restriction on
    # the hand-CUDA table — 42 of whose 51 single arms carry no established
    # launch-argument resolution (2026-09-27, was "45 of 49": the nine that do are the
    # curl's step_B / step_D 'PML', the constitutive's update_H / update_E 'ordinary',
    # the three off-diagonal update_E families and the mirror fill's fill_B / fill_D,
    # the six ``registry._RESOLVERS`` / ``_LAUNCHERS`` tokens) — rather than
    # a rule about being second. Running
    # the primary through the loop is what keeps an unlaunchable plan out of the
    # dispatch set on the leg where that table composes FIRST, which a verbatim copy
    # of its plans would have dispatched.
    for table_name, plan in tables_in_order:
        table_refusals = refused.setdefault(table_name, {})
        table_plans = dict(getattr(plan, "plans", {}) or {})
        table_selected = dict(getattr(plan, "selected", {}) or {})
        table_reasons = dict(getattr(plan, "reasons", {}) or {})
        pending = dict(pending_of.get(table_name) or {})
        # A SEAM'S OWN REFUSAL IS NOT A SLOT'S. ``fused_pair_*`` keys name a
        # PRODUCT rather than a ``STEP_ORDER`` slot, and they are the only place the
        # label offer's refusal is visible at all, so they merge into the shared
        # mapping. Every other key is a statement about a slot this table did not
        # win, and it goes where a reader will not mistake it for the reason the
        # slot is on the array path.
        leftovers: Dict[str, Any] = {}
        for key, value in table_reasons.items():
            if key.startswith("fused_pair_"):
                reasons.setdefault(key, value)
            elif table_name == primary_name:
                reasons.setdefault(key, value)
            else:
                leftovers[key] = value
        if leftovers:
            secondary_reasons[table_name] = leftovers

        adoptable_singles = ((_adoptable_single_slots(table_plans, table_selected)
                              | _adoptable_fill_slots(table_plans, table_selected))
                             if table_name == CUDA_TABLE else set())
        for _identity, members in _units(table_plans):
            labels = {table_selected.get(slot) for slot in members}
            label = next(iter(labels)) if len(labels) == 1 else None
            display = (namespaced(label) if table_name == CUDA_TABLE and label
                       else str(label))
            if table_name == CUDA_TABLE and (
                    label is None
                    or not (_is_fused_label(table_name, label)
                            or set(members) <= adoptable_singles)):
                for slot in members:
                    table_refusals.setdefault(
                        slot,
                        (f"{display} is a fill plan on the {table_name} table that "
                         "is not an adoptable twin: a CUDA fill dispatches only with "
                         "its twin on the other family (FILL_ARM_TWINS), and only "
                         "when both carry an established launch-argument "
                         "resolution, a launcher with run_near and run_far, and a "
                         "CUDA_ARM_CERTIFICATION row")
                        if slot in FILL_ARM_TWINS else
                        (f"{display} is a single-arm plan on the {table_name} table "
                         "that is not an adoptable seam: a CUDA single dispatches "
                         "only with its seam partner (SINGLE_ARM_SEAMS), and only "
                         "when both carry an established launch-argument "
                         "resolution, a launcher and a CUDA_ARM_CERTIFICATION row"))
                continue
            # THE LAUNCHABILITY FILTER IS NOT APPLIED TO A PRIMARY THAT DOES NOT
            # DECLARE IT, and that is a real distinction rather than a carve-out. The
            # hand-CUDA composer DECLARES ``launchable`` on every plan it installs,
            # so filtering on it there is reading the composer's own answer. The
            # Triton and Metal composers declare nothing, so the filter would fall
            # back to "does this object have a run", and a stub that does not — the
            # shape twenty-eight contract tests use to drive clause 8 — would be
            # DROPPED here instead of refused BY NAME four rungs down. Measured: the
            # named refusals of every certified Triton label went to "no slot is left
            # carrying a kernel" the first time this filter was applied to them.
            if (table_name == CUDA_TABLE or table_name != primary_name) and not all(
                    _launchable(table_plans[slot]) for slot in members):
                table_refusals[display] = (
                    f"not launchable: {', '.join(members)} — the product carries no "
                    "established launch-argument resolution, and a guessed one is a "
                    "half-cell error away from converged, smooth and wrong")
                continue
            held = [slot for slot in members if slot in plans]
            blocked: List[str] = []
            for slot in held:
                incumbent = selected.get(slot, "unknown")
                if not (YIELD_PENDING_PRIMARY_SLOTS and incumbent in pending_of.get(
                        primary_name, {})):
                    # THE TABLE IS NAMED ONCE. A namespaced incumbent already says
                    # which table wrote it; prefixing it again produced
                    # "cuda:cuda:fused magnetic pair" on the preference leg, which
                    # reads as a different label rather than as a repeat.
                    who = (incumbent if incumbent.startswith(CUDA_LABEL_PREFIX)
                           else f"{primary_name}:{incumbent}")
                    blocked.append(f"{slot} held by {who}")
            if blocked:
                table_refusals[display] = "slots " + "; ".join(blocked)
                continue
            # THE FILL TWIN IS CLOSED AGAINST THE MERGED PLAN, not only against this
            # table's own plans (FILL_ARM_TWINS). Every unit of an earlier table is
            # already in ``plans`` by now, so a twin slot held there is held by
            # ANOTHER table -- and adopting this half would put the fold's two fill
            # passes on two tables. A twin this table itself adopted a moment ago is
            # the pair being assembled, and passes.
            if table_name == CUDA_TABLE and len(members) == 1 \
                    and members[0] in FILL_ARM_TWINS:
                twin = FILL_ARM_TWINS[members[0]]
                if twin in plans and backends.get(twin) != CUDA_TABLE:
                    incumbent = selected.get(twin, "unknown")
                    who = (incumbent if incumbent.startswith(CUDA_LABEL_PREFIX)
                           else f"{backends.get(twin, primary_name)}:{incumbent}")
                    table_refusals[f"{display} ({members[0]})"] = (
                        f"its twin {twin} is held by {who}; a fold whose two fill "
                        "passes run on two tables is a composition no gate drove")
                    continue
            # THE OFF-DIAGONAL D/E SEAM IS CLOSED AGAINST THE MERGED PLAN TOO
            # (CROSS_TABLE_SEAM_REFUSED_SINGLES). ``_units`` walks the driver's slot
            # order and every earlier table's units are already in ``plans``, so a
            # partner held there is held by ANOTHER table whichever half comes first;
            # a partner this table adopted a moment ago is the seam being assembled.
            if table_name == CUDA_TABLE and len(members) == 1 \
                    and members[0] in SINGLE_ARM_SEAMS:
                partner = SINGLE_ARM_SEAMS[members[0]]
                seam_labels = {namespaced(label) for label in (
                    table_selected.get(members[0]), table_selected.get(partner))
                    if label}
                if (seam_labels & CROSS_TABLE_SEAM_REFUSED_SINGLES
                        and partner in plans and backends.get(partner) != CUDA_TABLE):
                    incumbent = selected.get(partner, "unknown")
                    who = (incumbent if incumbent.startswith(CUDA_LABEL_PREFIX)
                           else f"{backends.get(partner, primary_name)}:{incumbent}")
                    table_refusals[f"{display} ({members[0]})"] = (
                        f"its seam partner {partner} is held by {who}; an "
                        "off-diagonal D/E seam whose two halves run on two tables "
                        "is a composition no gate drove")
                    continue
            # THE DISPLACEMENT IS WHOLE OR IT IS REFUSED. See the docstring's
            # fourth clause: a primary unit only partly covered by this one would
            # leave half a fused pair standing.
            wanted = set(members)
            split_primary = [
                ", ".join(unit) for _pid, unit in _units(plans)
                if set(unit) & wanted and not set(unit) <= wanted]
            if split_primary:
                table_refusals[display] = (
                    f"adopting it would split the {primary_name} product occupying "
                    f"{'; '.join(split_primary)}, and a pair that runs half of "
                    "itself either skips a sub-step or double-applies one")
                continue
            # THE PENDING CHECK IS THE SECONDARY TABLE'S ONLY. A PENDING label on the
            # PRIMARY table refuses the WHOLE plan at rung 7a, by name, with the
            # reason that map carries — dropping its unit here instead would send the
            # step to the array path with "no slot is left carrying a kernel", which
            # is true and tells the reader nothing. A secondary table's pending unit
            # is a different question: refusing it leaves the slot with the
            # incumbent, which is a composition that IS certified.
            if table_name != primary_name and display in pending:
                table_refusals[display] = pending[display]
                continue
            for slot in members:
                plans[slot] = table_plans[slot]
                selected[slot] = display
                backends[slot] = table_name
                reasons.pop(slot, None)

    arbitration["adopted"] = {slot: table for slot, table in sorted(backends.items())
                              if table != primary_name}
    # THE REFUSALS A SLOT ENDED UP CARRYING ANYWAY ARE NOT REFUSALS. A unit blocked
    # on one pass can be adopted on a later one only in the yield case, and a stale
    # entry there would read as "this slot went to the array path" beside a slot
    # that dispatched.
    for table_name, entries in list(refused.items()):
        for key in [k for k in entries if k in plans and backends.get(k) == table_name]:
            entries.pop(key, None)
    return MergedStepPlan(plans, reasons, selected, backends)


def _is_fused_label(table_name: str, label: str) -> bool:
    """Is ``label`` a FUSED product of ``table_name``? Asked at the merge boundary."""
    if table_name == CUDA_TABLE:
        return is_fused(label)
    from . import fastpath  # noqa: PLC0415

    return bool(fastpath.arm_is_fused(label))


def pending_labels() -> Set[str]:
    """The namespaced labels rung 7a refuses. One reader, one spelling."""
    return set(CUDA_PENDING_DEVICE_GATE_ARMS)
