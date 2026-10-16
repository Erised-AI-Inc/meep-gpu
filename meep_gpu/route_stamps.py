"""The route-campaign stamps of the three kernel tables, in one module.

WHAT A STAMP NAMES. Each kernel table's release (``fastpath.RELEASED_FUSED_ARMS``,
``fastpath_cuda.CUDA_RELEASED_FUSED_ARMS``, ``metal_dispatch.METAL_RELEASED_FUSED_ARMS``)
is cut from one driver-route campaign per compute capability or GPU architecture, all
under one stamp:

    results/dispatch_fused_route_<stamp>_cc86/              gate_dispatch_fused_route.py
    results/dispatch_fused_route_cuda_<stamp>_cc86/         ... --backend cuda
    results/dispatch_metal_route_<stamp>_applegpu_g13s/     run_metal_dispatch_campaign.sh

and the dispatch-by-default licence of the primary NVIDIA table from
``gate_dispatch_end_to_end.py`` under ``results/dispatch_end_to_end_<stamp>/``. The stamp
is named BEFORE the campaign runs: ``recut_driver_dispatch_record.py`` refuses a run
directory other than the one :func:`route_gate` spells for the device its legs stamped
(``fastpath.route_campaign``, ``metal_runs.route_campaign``), and a licence directory
other than :func:`licence_campaign`, so a record names the run the tree asked for.

WHY THE STAMPS LIVE HERE AND NOT IN THE DISPATCH MODULES. Each stamp used to be a
constant of its table's dispatch module (``fastpath.py``, ``fastpath_cuda.py``,
``metal_dispatch.py``). A gate artifact's import record is every repository module its
process held, and every gate that plans a driver holds those modules, so a re-route,
a one-line stamp edit, voided the evidence of every gate that imports the planner
(``fastpath.py`` is in the run closure of 98 of the 250 gate and probe scripts). The
dispatch path read the stamp only as a label: the run record's
``fusion.driver_route_gate`` and the "released by ..." clause of a refusal; no decision
keyed on it. Those labels now name the ledger slot that records the campaign
(``fastpath.ROUTE_RECORD``, ``fastpath_cuda.CUDA_ROUTE_RECORD``,
``metal_dispatch.METAL_ROUTE_RECORD``), and the stamps are here.

WHO READS THIS MODULE. Only code that names a campaign: the record re-cut
(``parity/meep_gpu/recut_driver_dispatch_record.py``), the round preflights
(``parity/meep_gpu/rounds/1.0/legs.py`` and ``metal_round.py``), the dispatch
reachability the boards cite (``parity/meep_gpu/dispatch_reachability.py``), and the
tests of those records. No package module imports it and no gate or probe holds it
(``python parity/meep_gpu/import_closure.py bill meep_gpu/route_stamps.py``;
``parity/meep_gpu/test_route_stamps.py`` holds both). A re-route therefore edits this
file and voids no gate artifact by digest. The route records are voided BY NAME: each record's
``runs[<device>].released_fused_arms.gate`` must be the campaign this module spells for
that device (``meep_gpu/test_dispatch_contract.py``), so a table re-routed alone
(:data:`REROUTED`) changes only that table's name.

THE ROUND'S STAMP IS SET ONCE, AT THE FREEZE. :data:`ROUND` reads :data:`PLACEHOLDER`
until the freeze of the round that runs the campaigns is cut. The freeze sets it, one
line in this file and no other byte of the package::

    ROUND = "2026-10-15_10"        # <date the freeze is cut>_<release label>

and moves the stamps of the round before it into :data:`SPENT` (their campaigns ran).
While it reads the placeholder, every preflight and the re-cut refuse by name
(``route_stamp_is_unset``), and the route-record contract tests are red by contract
(``tools/ci/pending_certification.txt``). A campaign name that has run is refused
(``route_campaign_is_spent``): campaign names are one-shot, because a gate appends rows
to a directory it finds. ``python -m meep_gpu.route_stamps`` prints the campaign names
the stamps spell and exits non-zero while any is refused.

Standard library only, and no import of the package: whatever imports this module
gains this file in its import record and nothing else.
"""

from __future__ import annotations

import sys
from typing import Mapping, Optional, Sequence, Tuple

#: The kernel tables, as the re-cut's ``--backend`` spells them.
TABLES: Tuple[str, ...] = ("triton", "cuda", "metal")

#: The results-directory prefix of each table's route campaign. ``dispatch_fused_route_``
#: is a prefix of the CUDA one too, so a campaign name is BUILT from a stamp
#: (:func:`route_gate`), never parsed back into one.
ROUTE_PREFIXES: Mapping[str, str] = {
    "triton": "dispatch_fused_route_",
    "cuda": "dispatch_fused_route_cuda_",
    "metal": "dispatch_metal_route_",
}

#: The results-directory prefix of the dispatch-by-default licence
#: (``gate_dispatch_end_to_end.py``'s ship leg and null control), filed on the primary
#: NVIDIA table of a compute capability (``fastpath.primary_table``).
LICENCE_PREFIX = "dispatch_end_to_end_"

#: The tables a licence can be filed on. The Metal table's default is licensed by its
#: route campaign's shipped leg, not by ``gate_dispatch_end_to_end``.
LICENCE_TABLES: Tuple[str, ...] = ("triton", "cuda")

#: What :data:`ROUND` reads until the freeze sets it. Not a date, so no campaign can be
#: mistaken for one run under it.
PLACEHOLDER = "UNSET"

#: THE ROUND'S ONE STAMP, shared by the three tables and by every compute capability and
#: GPU architecture that drives them (``<stamp>_cc86``, ``<stamp>_applegpu_g13s``).
#: Set at the freeze; see the module docstring.
ROUND = "2026-10-15_10"

#: A TABLE RE-ROUTED ALONE after the freeze: its route gate fixed and re-run while the
#: other tables' campaigns stand. It takes its own stamp here, for example
#: ``{"metal": "2026-10-13_20"}``, and only its record changes name. Emptied when the
#: next round sets :data:`ROUND`.
REROUTED: Mapping[str, str] = {}

#: CAMPAIGN NAMES THAT MAY NOT BE LAUNCHED OR RECORDED AGAIN, per table: results
#: directories a campaign has run in. Keyed by the whole directory name, because one
#: stamp serves every device (``<stamp>_cc86`` running does not spend ``<stamp>_cc90``);
#: a stamp whose campaign predates the per-device names spends the stamp itself, which
#: :func:`campaign_refusal` reads as such. ``test_route_stamps`` requires every route run
#: the three ledgers record to be listed here.
SPENT: Mapping[str, Tuple[str, ...]] = {
    "triton": (
        "dispatch_fused_route_2026-09-23_sparse",
        "dispatch_fused_route_2026-09-27_flip",
        "dispatch_fused_route_2026-09-30_091_cc86",
        "dispatch_fused_route_2026-10-05_092_cc86",
        "dispatch_fused_route_2026-10-11_10_cc86",
        "dispatch_fused_route_2026-10-13_10_cc86",
        "dispatch_fused_route_2026-10-14_10_cc86",
    ),
    "cuda": (
        "dispatch_fused_route_cuda_2026-09-24_hoist",
        "dispatch_fused_route_cuda_2026-09-25_roundb",
        "dispatch_fused_route_cuda_2026-09-27_flip",
        "dispatch_fused_route_cuda_2026-09-30_091_cc86",
        "dispatch_fused_route_cuda_2026-10-05_092_cc86",
        "dispatch_fused_route_cuda_2026-10-11_10_cc86",
        "dispatch_fused_route_cuda_2026-10-13_10_cc86",
        "dispatch_fused_route_cuda_2026-10-14_10_cc86",
    ),
    "metal": (
        "dispatch_metal_route_2026-09-25_night",
        "dispatch_metal_route_2026-10-02_arch3",
        "dispatch_metal_route_2026-10-05_perarch_applegpu_g13s",
        "dispatch_metal_route_2026-10-11_10_applegpu_g13s",
        "dispatch_metal_route_2026-10-12_10_applegpu_g13s",
        "dispatch_metal_route_2026-10-13_10_applegpu_g13s",
        "dispatch_metal_route_2026-10-13_20_applegpu_g13s",
        "dispatch_metal_route_2026-10-14_10_applegpu_g13s",
    ),
}


def _table(table: str) -> str:
    if table not in TABLES:
        raise ValueError(f"{table!r} is not a kernel table; the tables are {list(TABLES)}")
    return table


def stamp(table: str) -> str:
    """The stamp ``table``'s route runs under: its :data:`REROUTED` entry, else ROUND."""
    return REROUTED.get(_table(table), ROUND)


def route_gate(table: str) -> str:
    """The stem of ``table``'s route campaigns: ``dispatch_fused_route_<stamp>`` and kin.

    A device's campaign directory is this stem followed by the device, as
    ``fastpath.route_campaign`` (``_cc86``) and ``metal_runs.route_campaign``
    (``_applegpu_g13s``) spell it.
    """
    return ROUTE_PREFIXES[_table(table)] + stamp(table)


def licence_campaign(table: str) -> str:
    """The licence directory of ``table``'s round: ``dispatch_end_to_end_<stamp>``."""
    if _table(table) not in LICENCE_TABLES:
        raise ValueError(f"the {table} table files no dispatch-by-default licence; its "
                         "default is licensed by its route campaign's shipped leg")
    return LICENCE_PREFIX + stamp(table)


def is_unset(table: str) -> bool:
    """True while ``table``'s stamp is the placeholder the freeze replaces."""
    return stamp(table) == PLACEHOLDER


def _unset_reason(table: str) -> str:
    return (f"route_stamp_is_unset: meep_gpu/route_stamps.py names no stamp for the "
            f"{table} table (ROUND = {PLACEHOLDER!r}); set ROUND at the freeze, before "
            f"the round's campaigns run")


def campaign_refusal(table: str, campaign: str) -> Optional[str]:
    """Why ``campaign`` may not be launched or recorded as ``table``'s route now, or None.

    ``campaign`` is a results-directory name. Refused, each by a name the reason opens
    with: the stamp is still the placeholder (``route_stamp_is_unset``); the name is not
    a campaign under the current stamp (``route_campaign_is_not_this_stamp``); it, or a
    campaign named by the stamp alone, has already run (``route_campaign_is_spent``).
    """
    if is_unset(table):
        return _unset_reason(table)
    gate = route_gate(table)
    if not campaign.startswith(gate + "_"):
        return (f"route_campaign_is_not_this_stamp: {campaign!r} is not a {table} route "
                f"campaign under the current stamp, whose campaigns are {gate}_<device> "
                f"(meep_gpu/route_stamps.py)")
    spent = SPENT[table]
    if campaign in spent or gate in spent:
        return (f"route_campaign_is_spent: {campaign if campaign in spent else gate!r} "
                f"has already run (meep_gpu/route_stamps.SPENT); campaign names are "
                f"one-shot, so a re-run takes a new stamp")
    return None


def licence_refusal(table: str, campaign: str, device: str) -> Optional[str]:
    """Why ``campaign`` is not ``table``'s licence directory for ``device``, or None.

    ``campaign`` is the licence's results directory (the one holding ``ship/`` and
    ``null/``); ``device`` is the suffix a route campaign carries for the device the
    licence ran on (``cc86``). The licence of the round is :func:`licence_campaign`,
    alone or followed by ``_<device>``; a licence of another round, or of another
    device, is refused (``licence_is_for_another_campaign``), as is any licence while
    the stamp is the placeholder.
    """
    if is_unset(table):
        return _unset_reason(table)
    wanted = licence_campaign(table)
    if campaign in (wanted, f"{wanted}_{device}"):
        return None
    return (f"licence_is_for_another_campaign: the licence directory is {campaign!r}; "
            f"the {table} table's round licenses dispatch by default only from "
            f"{wanted!r} (or {wanted}_{device}), the stamp meep_gpu/route_stamps.py "
            f"names; re-run gate_dispatch_end_to_end there")


def report() -> Tuple[Sequence[str], Sequence[str]]:
    """``(lines, refusals)``: each table's stamp and route stem, and what is refused."""
    lines, refusals = [], []
    for table in TABLES:
        gate = route_gate(table)
        licence = (f"  licence {licence_campaign(table)}" if table in LICENCE_TABLES
                   else "")
        lines.append(f"{table}: stamp {stamp(table)}  route {gate}_<device>{licence}")
        if is_unset(table):
            refusals.append(_unset_reason(table))
        elif gate in SPENT[table]:
            refusals.append(f"route_campaign_is_spent: {gate!r} has already run")
    return lines, refusals


def main(argv: Optional[Sequence[str]] = None) -> int:
    lines, refusals = report()
    for line in lines:
        print(line, flush=True)
    for line in refusals:
        print(f"REFUSED {line}", flush=True)
    return 1 if refusals else 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
#
# What each table's stamp named, and why it moved, in the words written beside the
# constant at each move. Moved here verbatim with the constants (2026-10-13, owner
# ruling R22); "this file" and "this line" in them mean the dispatch module the
# constant then lived in. New moves are recorded below the last paragraph of the
# table's section.
#
# THE TRITON TABLE: fastpath.DRIVER_ROUTE_FUSED_GATE until 2026-10-13.
#
# THE DRIVER-ROUTE GATE FOR THE FUSED ROUTE, and the artifact that released it.
# ``parity/meep_gpu/gate_dispatch_fused_route.py``, run 2026-08-30 on the GPU host
# GPU 6 (RTX A6000, sm_86, Triton 3.1.0, CuPy 13.5.1, MEEP 1.33.0), four legs,
# all four ``release.released == true``.
#
# WHY IT IS NAMED IN PRODUCT CODE. :data:`RELEASED_FUSED_ARMS` below is the first
# constant in this file that ADMITS rather than refuses, so a reader has to be
# able to get from the admission to the measurement without being told where to
# look, and a dispatching run's own artifact has to cite it. The same rule
# :data:`ARM_CERTIFICATION` follows for a sub-step family, one level up.
#
# WHY IT NAMES ``_bind`` AND NOT THE TWO RUNS BEFORE IT. All three released and
# all three measured a real route; what separates them is WHICH BYTES they ran.
# ``_seamgate`` reached every arm through ``MEEP_GPU_FUSE_ARMS``, because at the
# time it ran no arm was released and the switch was the only way in — evidence
# about a path no user takes. ``_veto`` drove the three in-envelope cases with the
# switch UNSET, with this constant's own table doing the admitting, and added two
# measurements ``_seamgate`` could not: the ENVELOPE declining a configuration on
# a device (``pml_3d_diagonal``, both directions) and a substitution proof against
# a baseline that is genuinely unfused (see :data:`FUSE_ARMS_VETO`, without which
# the baseline fused too and the drop was 0.0). But ``_veto`` ran this file at
# 29ac1216, and what shipped afterwards was 87f740ee and then d11454ba — so the
# citation named a run that had never executed the code it was cited from, and
# three different digests for one file were in play at once
# (``fingerprints.json`` recorded a fourth, 06cdbf37). ``_bind`` is that run
# repeated against the bytes that ship, and it adds the case named below.
#
# THE ARTIFACT NO LONGER PREDATES THE LINE THAT NAMES IT, and closing that gap is
# what this round is. The name below, the case list in :data:`RELEASED_FUSED_ARMS`
# and every other edit landed BEFORE the campaign started, so the four legs each
# recorded this file at the digest it ships at, and
# ``fingerprints.json``'s ``driver_dispatch.source_sha256`` is cut from that
# recording rather than typed. ``test_dispatch_contract`` checks all three against
# each other — record, artifact and tree — so a self-naming citation that goes
# stale again fails there instead of being argued about in prose.
#
# WHY IT MOVED OFF ``_bind``, 2026-09-02, and it is the same rule applied to a
# different file. ``driver.py``'s ``_inject_electric_through_conductivity`` now
# rescales SPARSELY at the deposit indices the sources publish, where it used to
# run three whole-volume passes; the passes canonicalised ``-0.0`` to ``+0.0``
# across the whole component and, on CuPy, flushed every non-deposit subnormal
# word, which is what forced the fused electric pairs to refuse conductive rows.
# ``_bind``'s four legs ran the OLD ``driver.py`` (cd437accdd37), so the citation
# named a run that had not executed the bytes that ship — the exact defect the
# paragraph above was written to close, one file over. ``_bind`` is NOT deleted:
# it stays as the record of what the old driver did. This name is the same
# four-leg protocol repeated against the new bytes, with every source edit again
# landed before the campaign started. What the re-run settles is whether the
# sparse rescale moved anything the gate certifies; the artifact it names is the
# answer, and ``test_dispatch_contract`` is what holds this line to it.
#
# WHY IT MOVED AGAIN, 2026-09-02, and this time the run is LARGER rather than
# repeated. ``_sparserescale`` drove six cases and released three arms; this one
# drives nine and releases five. What is new in it, and what each addition is for:
#
#   * ``folded_2d`` and ``folded_dispersive_2d`` DISPATCH rather than refuse. The
#     fill consults (:data:`DRIVER_SLOTS`) retired the completeness rung for the
#     fills, so a folded composition reaches the seam for the first time — six of
#     seven slots, both fill slots carrying the mirror-fill kernel, and the two
#     folded pairs absorbing four of them. Every earlier campaign recorded the
#     folded cases as refusals, and a folded case that does not run a fill
#     in-launch measures nothing about this composition.
#   * ``pml_1d`` is the 1-D shape the previous envelope refused on four corpus
#     seam-instances. The composer selects the same two ordinary pairs on it that
#     it selects in 2-D.
#   * ``conductive_2d`` had REGRESSED between the two runs: the 2026-09-02 product
#     wave wired ``conductive_fused_electric_pair``, whose label cannot be
#     admitted, and clause (8) then rejected the whole plan rather than one seam.
#     ``plan_step``'s ``fuse_labels`` offer is what restores it, and this case is
#     what measures that it is restored.
#   * ``pml_3d_diagonal`` stays the ENVELOPE control and is deliberately NOT
#     released. After ``dimensions`` widened to admit 1-D it is the only reachable
#     shape left on which a RELEASED arm is selected and the release declines it,
#     so releasing 3-D would have cost the one device measurement that the
#     envelope can say no at all. One corpus seam-instance, weighed against that,
#     and named here so the trade is legible rather than implied.
# WHY IT MOVED AGAIN, 2026-09-04, and this time nothing about the release
# changed — the FILES the release is cited from did. Three of them moved in one
# round: ``driver.py`` and this file gained the by-name magnetic-half-step
# consult (:data:`SYNC_UPDATE_H_PASS`), and ``triton_kernels/launch.py`` gained
# the H_to_D seam row and the neighbouring-seam arbitration rule. All four legs
# of ``_foldedrelease`` ran the pre-edit bytes, so the citation named a run that
# had not executed the code citing it — the same defect this line has been moved
# for twice before, a third file over. This name is the same four-leg protocol
# repeated against the bytes that ship, with every source edit of the round
# landed BEFORE the campaign started; ``fingerprints.json``'s
# ``driver_dispatch.source_sha256`` is re-cut from that recording by
# ``parity/meep_gpu/recut_driver_dispatch_record.py``, which refuses unless leg,
# leg and tree all agree. ``_foldedrelease`` is NOT deleted: it stays as the
# record of what the pre-channel driver did, and the two runs' verdicts are
# compared case by case rather than assumed equal.
# RE-RUN 2026-09-07 as `dispatch_fused_route_2026-09-07_cylfinal`, and repointed
# here, for the reason the paragraph above gives: the cylindrical H_to_D wiring
# round routed both CuPy-side products through their composers, which edited this
# very file (fastpath.py a5934d2bba7a -> b74a4c1ac90a) and triton_kernels/launch.py.
# The 09-04 record bound the bytes its own flush leg executed, so leaving the
# citation here would have the record binding one fastpath.py and the CITED gate
# having run another -- which is exactly what test_the_cited_driver_route_gate_ran_
# the_bytes_the_record_binds refuses. All four legs of the new run released
# (shipped, harness_keep, flush, shipped_expansion_probe) and its substitution
# counts reproduce the 09-04 run's case by case. `_syncchannel` is NOT deleted: it
# stays as the record of what the pre-routing driver did, and neither is
# `_cylwire`, the first run of this round -- it measured the bytes as they were
# BEFORE this constant was repointed, which is the ordering trap the paragraph
# above names: every source edit of a round must land BEFORE its campaign starts,
# and editing this line is itself such an edit.
#
# WHY IT MOVED, 2026-09-11, and the first thing to say is what this line was NOT
# naming. ``fingerprints.json``'s ``driver_dispatch`` record binds
# ``dispatch_fused_route_2026-09-08_wired`` — the run cut after the H_to_D wiring
# round — while this constant still read ``_cylfinal``, so two record welds have
# been red since 09-08 (``test_the_recorded_dispatch_shape_matches_the_wiring_it_
# describes``, which compares the two names, and ``test_the_cited_driver_route_
# gate_ran_the_bytes_the_record_binds``, whose cited directory holds a different
# ``fastpath.py`` digest from the one the record binds). The ledger's authority
# has been the 09-08 run throughout; this line was the stale half.
#
# WHAT ``_realarms`` ADDS, and it is a release rather than a repeat. Four more
# fused arms are driven through the seam on THREE cases: ``conductive_2d`` and
# ``folded_dispersive_2d``, which have dispatched in every campaign since 09-02
# and grow from one pair to two now that the D-seam product beside them is
# admitted, and ``cylindrical_m0`` — a NEW route-gate case, the end-to-end gate's
# Dcyl m = 0 shape with a flux monitor — which is the first cylindrical
# configuration any release gate has driven and the reason ``cylindrical`` leaves
# the shared envelope for :data:`FUSED_RELEASE_ARM_AXES` below.
#
# THE ORDERING RULE IS UNCHANGED AND IS WHAT THIS PARAGRAPH IS FOR. Every source
# edit of this round — this name, the four :data:`RELEASED_FUSED_ARMS` rows, the
# envelope move, the three :data:`ARM_CERTIFICATION` rows, the three
# :data:`PENDING_DEVICE_GATE_ARMS` deletions, the route gate's new case and its
# DRIVE rows — lands BEFORE the campaign starts, so all four legs record this
# file at the digest it ships at, and ``fingerprints.json``'s
# ``driver_dispatch.source_sha256`` is re-cut FROM that recording by
# ``parity/meep_gpu/recut_driver_dispatch_record.py``, which refuses unless leg,
# leg and tree agree. Between the edit and the recut the two welds above stay
# red, joined by the three ARM_CERTIFICATION lookups whose ledger entries do not
# exist yet — that window is named in the campaign log rather than papered over,
# because the alternative is writing a ledger entry by hand, which is the record
# forgery this campaign forbids. ``_cylfinal`` and ``_wired`` are NOT deleted:
# they stay as the record of what the pre-release route did, and their verdicts
# are compared case by case against this run rather than assumed equal.
# REPOINTED 2026-09-15 to a FRESH stamp. The ``2026-09-14_weldgrid`` run drove only
# 9 of the 36 DRIVE cases (its driver passed a hardcoded ``--cases`` subset), so its
# recut refused all 44 release rows whose case it never ran -- and not one of the 44
# was a composer finding. That directory is kept rather than overwritten: it is still
# the on-device record that ``no_pml_dispersive_2d`` passes fused on three legs.
# REPOINTED AGAIN to ``_2026-09-15_offdiag``. The ``_2026-09-15_weldgrid`` run drove
# all 36 cases and released three legs; ``shipped_expansion_probe`` refused on
# ``folded_complex_offdiag_2d`` and ``folded_complex_nobloch_offdiag_2d``, which a
# premature widening of ``fused pair B (folded complex)`` had marked ``dispatch``
# while the ``update_E`` arm they need is pending. The widening is reverted, the two
# cases are ``no-fused-arm`` controls, and that run is kept as the measurement.
# REPOINTED 2026-09-15 to ``_2026-09-15_gapclose``, before that campaign starts, for
# ONE batch carrying two changes to this table. (1) It releases ``fused pair D
# (complex conductive no-PML)`` on ``complex_no_pml_3d`` (:data:`RELEASED_FUSED_ARMS`,
# :data:`FUSED_RELEASE_ARM_AXES`, :data:`ARM_CERTIFICATION`, the
# :data:`PENDING_DEVICE_GATE_ARMS` deletion) and restores that case's Triton DRIVE
# row. (2) With the 2026-09-15 ruling that lifted ``complex folded off-diagonal`` out
# of :data:`PENDING_DEVICE_GATE_ARMS`, it widens ``fused pair B (folded complex)``
# back onto the two off-diagonal folds, which are DRIVE rows again. Every source edit
# of the batch lands before the new campaign starts, so every leg records this file
# at the digest it ships at, and the ``triton_complex_offdiag_device_gate`` re-gate
# runs on those bytes too. ``_2026-09-15_offdiag`` is NOT deleted: all four of its
# legs released, and it stays the record of the table before both changes, including
# both off-diagonal folds as refusal controls (PASS-NO-FUSED-ARM, refusing_rung
# ``pending-device-gate``) on the bytes before the lift.
#
# REPOINTED 2026-09-17 to ``_2026-09-17_allpaths``, again before the campaign runs,
# for the batch that routes the two SCRATCH-OUTPUT OFF-DIAGONAL WELDS. That batch
# adds ``fused pair D (off-diagonal)`` and ``fused pair D (folded off-diagonal)`` to
# :data:`ARM_CERTIFICATION` and to the certified-product table in
# ``triton_kernels/launch.py``, and gives :class:`ScratchWeldPairPlan` a ``warm()``
# that launches once at grid ``(0,)`` without rotating. ``_2026-09-15_gapclose`` is
# NOT deleted: it stays the record of the table with both welds present, released and
# DELIBERATELY UNWIRED, which is the control this batch overturns. The stencil gate
# is re-run on the bytes this edit ships before the route campaign starts, so the
# ledger entries these two labels name are cut from the same tree the route records.
#
# REPOINTED 2026-09-27 to ``_2026-09-27_flip``, before that campaign starts, for the
# edit that turns :data:`DISPATCH_BY_DEFAULT` on and makes the enable refuse every
# value but ``0`` and ``1`` by name. Neither changes which arm a plan admits, but
# both change this file's bytes, and the record's recut refuses a route run whose
# legs executed other bytes than the tree ships. ``_2026-09-23_sparse`` is NOT
# deleted: it stays the record of the route on the opt-in tree.
#
# REPOINTED 2026-09-30 to ``_2026-09-30_091`` for release 0.9.1, whose certification
# round re-runs every table's route on the released files; ``_2026-09-27_flip`` stays
# the record of the 0.9.0 route.
#
# REPOINTED 2026-10-05 to ``_2026-10-05_092`` for release 0.9.2, before its
# campaigns run: this file moved (a supported NVIDIA identity that is not certified
# runs by default, and a table certified here outranks one that is only supported),
# so the route is re-run on it. One stamp for the release, shared by every compute
# capability that drives it (``<stamp>_cc86``, ``<stamp>_cc90``).
# ``_2026-09-30_091`` stays the record of the 0.9.1 route.
#
# REPOINTED 2026-10-11 to ``_2026-10-11_10`` for release 1.0, at its freeze and before
# its campaigns run: the release moves this file (the flush dispatch licence,
# :data:`TABLE_DISPATCH_POLICIES`, and the deleted complex-multiply expansion licence)
# and the array path every route case is compared against, so the route is re-run on
# the frozen bytes. One stamp for the release, shared by every compute capability that
# drives it (``<stamp>_cc86``, ``<stamp>_cc90``). ``_2026-10-05_092`` stays the record
# of the 0.9.2 route.
#
# REPOINTED 2026-10-12 to ``_2026-10-13_10``, at the 1.0 freeze that follows
# per-kernel certification and before its campaigns run: the package root now binds
# its names on first use, so every gate's import record changes shape and the whole
# round is re-run once, and this file, the lift and the three composers moved since
# the route last ran. ``_2026-10-11_10`` is spent (its campaigns released on the
# frozen commit before this one) and stays the record of that pass.
#
# MOVED 2026-10-13 to this module (owner ruling R22, the route-stamp registry): ROUND is
# the placeholder until the R22 round's freeze sets it. ``_2026-10-13_10`` is spent (its
# campaign released on the frozen commit of the records the series carries).
#
# THE HAND-CUDA TABLE: fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE until 2026-10-13.
#
# THE CUDA DRIVER-ROUTE GATE, NAMED BEFORE IT RUNS. ``gate_dispatch_fused_route.py
# --backend cuda`` writes ``results/<this name>/`` with five legs, and
# ``recut_driver_dispatch_record.py --backend cuda`` refuses to cut the record
# unless every ``(arm, case, leg)`` row of :data:`CUDA_RELEASED_FUSED_ARMS` appears
# in that leg's ``arms_driven``. Naming it here first is what makes the run the
# shipping tree asked for the run that is measured.
# REPOINTED 2026-09-15 to a FRESH stamp rather than re-run under the old one: the
# ``2026-09-14_weldgrid`` run is what the 346-of-594 board cites, and overwriting it
# would strand that board's provenance. The re-run exists because the replacement-case
# edits moved ``fastpath.py`` and ``gate_dispatch_fused_route.py``, which the record
# binds, so the board no longer describes the tree.
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
# ``_2026-10-05_092`` (named 2026-10-05, before its campaign) is release 0.9.2's route:
# this file and ``fastpath.py`` moved (a supported device that is not certified runs
# by default, and a table certified here outranks one that is only supported), and so
# did ``gate_dispatch_fused_route.py``, which this record binds. One stamp for the
# release, shared by every compute capability that drives it. ``_2026-09-30_091``
# stays the record of the 0.9.1 route.
# ``_2026-10-11_10`` (named 2026-10-11, at the freeze, before its campaign) is release
# 1.0's route: this file, ``fastpath.py`` (the flush dispatch licence) and
# ``gate_dispatch_fused_route.py``, which this record binds, moved, and so did the
# kernel table and the array path every route case is compared against. One stamp for
# the release, shared by every compute capability that drives it. ``_2026-10-05_092``
# stays the record of the 0.9.2 route.
# ``_2026-10-13_10`` (named 2026-10-12, at the 1.0 freeze that follows per-kernel
# certification, before its campaign) is release 1.0's route on the new frozen
# bytes: the package root binds its names on first use, so every gate's import
# record changes shape and the whole round is re-run once, and ``fastpath.py``, the
# lift and the kernel table's composer moved since the route last ran.
# ``_2026-10-11_10`` is spent (its campaign released on the frozen commit before this
# one) and stays the record of that pass.
#
# MOVED 2026-10-13 to this module (owner ruling R22, the route-stamp registry): ROUND is
# the placeholder until the R22 round's freeze sets it. ``_2026-10-13_10`` is spent (its
# campaign released on the frozen commit of the records the series carries).
#
# THE METAL TABLE: metal_dispatch.METAL_DRIVER_ROUTE_GATE until 2026-10-13.
#
# THE DRIVER-ROUTE GATE THIS TABLE'S RELEASE IS CUT FROM, named BEFORE the
# campaign runs and not after it.
#
# THAT ORDER IS THE TRAP, and the Triton constant records paying for it: every
# source edit of a round must land BEFORE the round's campaign starts, and editing
# this line is itself such an edit — so a run that discovers the constant is wrong
# has to be repeated, not repointed. The campaign driver
# (``parity/meep_gpu/run_metal_dispatch_campaign.sh``) writes its legs under
# ``results/<this name>_<architecture>/`` (see the 2026-10-05 entry below), and
# ``recut_driver_dispatch_record.py --backend metal`` refuses to cut the record
# unless the directory it reads is the one this constant spells for the
# architecture its legs recorded.
# REPOINTED 2026-09-15, BEFORE the relaunch, which is the order the paragraph above
# demands. The ``2026-09-14_weldgrid`` campaign died with its host at 44 of 62 fleet
# gates, before its dispatch stage, and it had been launched while this line still
# named ``_target`` -- so it could not have been cut even had it finished.
# ``_2026-09-15_offdiag``: the ``_2026-09-15_weldgrid`` campaign was stopped part-way
# through its fleet, because the Triton off-diagonal fix moved fastpath.py, which
# every Metal gate records.
# ``_2026-09-15_gapclose``: one Triton batch moved fastpath.py again -- the release of
# ``fused pair D (complex conductive no-PML)`` and the lift of ``complex folded
# off-diagonal`` with the widening of ``fused pair B (folded complex)`` -- which every
# Metal gate and this table's record bind, so the fleet and the dispatch stage re-run
# under a fresh name. The ``_2026-09-15_offdiag`` campaign is kept as the record of the
# tree before it.
# REPOINTED 2026-09-17 to ``_2026-09-17_envelope``, BEFORE the campaign runs, for the
# batch that moved ``pml_active`` and ``bfast`` off the shared envelope and onto 29
# per-arm pins (16 added, values read off each arm's own release cases). The
# constant had read ``_2026-09-15_gapclose`` through the whole 2026-09-17_allpaths
# campaign, so that campaign's record could never have bound; this one can.
# ``_2026-09-17_envelope`` ran and refused on three of four legs, on two defects of
# the ROUTE GATE's instrument rather than of any product: its launch witness never
# wrapped the real-cylindrical plans' function tables (``cylindrical``,
# ``cylindrical_m0`` VACUOUS-PASS on kernels that ran), and its withheld-consult
# control had no NOT-APPLICABLE rung for an inactive layer (``absorber_1d``,
# ``no_pml_dispersive_2d`` NULL on an array-against-array comparison). It stays as
# the record of that. REPOINTED 2026-09-19 to ``_2026-09-19_witness`` for the
# re-run with both repaired.
# ``_2026-09-19_witness`` refused ONE case on its probe leg, and on the route gate's
# EXPECTATION rather than on any product: ``complex_no_pml_3d`` (first fused on this
# round) read DROPPED-BUT-NOT-EXACT on a drop of 3.0 against an expected 1, because
# its unfused ``complex no-PML stored E`` single launches once per component (1800
# over 600 steps) and the pair folds all three into its one launch -- 8.0 -> 5.0 a
# step, compiled calls agreeing, byte-identical to the array path. The gate's
# "one launch fewer per pair" was the N = 1 case of ``pairs + sum(N - 1)``; that one
# drive row now DECLARES ``collapsed_single_launches_per_step = {"update_E": 3}``,
# cross-checked against the unfused leg's own counter (``DECLARATION-MISMATCH``
# refuses). Replayed over every stored row of that campaign's four case legs, it is
# the only verdict that moves. REPOINTED 2026-09-19 to ``_2026-09-19_components``,
# before that campaign ran; it released every leg (7.65 h) and cut the 327 / 597 board.
# REPOINTED 2026-09-20 to ``_2026-09-20_restrict``, before that campaign runs: the
# deposit repair this table brackets its source-seam pairs with (host NumPy here) now
# repairs only the component the source writes, bit-identical, so every bracketed row
# executes different bytes and the fleet and the route are re-run on them. The campaign
# runs its three long legs as concurrent lanes for the first time; the gate makes no
# timing claim, and each leg remains one process with its own policy and environment.
# REPOINTED 2026-09-27 to ``_2026-09-27_flip``, before that campaign runs: dispatch
# is on by default and held residency is this table's default, with the release at
# every freeze, the ``reset`` layer and the admission rung, so every leg executes
# different bytes and the fleet and the route are re-run on them.
# REPOINTED 2026-09-30 to ``_2026-09-30_091`` for release 0.9.1, whose certification
# round re-runs the fleet and the route on the released files.
# REPOINTED 2026-10-02 to ``_2026-10-02_arch3``, before that campaign runs: the
# 0.9.1 Metal round never ran, and this one is the first whose welds record the GPU
# architecture each ran on, which the route record requires every row to read
# certified. (``_2026-10-02_arch`` was abandoned during its fleet: its lift legs
# found no MEEP corpus. ``_2026-10-02_arch2`` was killed 40 gates in, every one
# released, when the shell that started it exited.)
# REPOINTED 2026-10-05 to ``_2026-10-05_perarch``, before that campaign runs: the
# admission reads the per-architecture run records and the route record keeps one
# run per GPU architecture, so this file moved and the route is re-run on it. THE
# CONSTANT IS NOW THE STAMP, not the directory: each architecture's campaign writes
# its legs under ``results/<metal_runs.route_campaign(this, architecture)>/``
# (``dispatch_metal_route_2026-10-05_perarch_applegpu_g13s``), the way the NVIDIA
# constants name ``<stamp>_cc86``, so a second Mac drives the same release on the
# same bytes without editing this line.
# REPOINTED 2026-10-11 to ``_2026-10-11_10`` for release 1.0, at its freeze and before
# its campaign runs: the release moves this file, the Metal kernels and the array path
# every route case is compared against, so the fleet and the held route are re-run on
# the frozen bytes, each architecture under ``<stamp>_<architecture>``
# (``dispatch_metal_route_2026-10-11_10_applegpu_g13s`` on an Apple M1 Max).
# ``_2026-10-05_perarch`` stays the record of the 0.9.2 route.
# REPOINTED 2026-10-11 to ``_2026-10-12_10``, before that campaign runs: release 1.0 is
# re-frozen with ``fastpath.py`` moved (rung 8b's per-arm policy evidence now answers a
# bare label in a CUDA-led plan from the Triton ledger), and the route record pins that
# file, so the held route is re-run on the re-frozen bytes
# (``dispatch_metal_route_2026-10-12_10_applegpu_g13s`` on an Apple M1 Max). The
# ``_2026-10-11_10`` campaign released every leg on the first frozen bytes and stays
# their record; route campaign names are one-shot, so it cannot be re-entered.
# REPOINTED 2026-10-12 to ``_2026-10-13_10``, at the 1.0 freeze that follows
# per-kernel certification and before that campaign runs: the package root binds its
# names on first use, so every gate's import record changes shape and the fleet and
# the held route are re-run once, and ``fastpath.py``, the lift and the Metal
# composer moved since the route last ran
# (``dispatch_metal_route_2026-10-13_10_applegpu_g13s`` on an Apple M1 Max). The
# ``_2026-10-12_10`` campaign released every leg on the frozen commit before this one
# and stays its record.
# REPOINTED 2026-10-13 to ``_2026-10-13_20``, before that campaign runs: the route
# gate's must-diverge control ``residency-inner-bracket-dropped`` armed on the start of
# H (``launch.StartOfHPlan``) after the run's first ``update_H``, where it is a
# pass-through, and read NULL-DID-NOT-DIVERGE on ``special_kz_2d`` and ``bfast_1d``;
# the gate now looks through a spent start to its occupant. The route gate's bytes
# moved, so the held route is re-run on them
# (``dispatch_metal_route_2026-10-13_20_applegpu_g13s`` on an Apple M1 Max). The
# ``_2026-10-13_10`` campaign ran on the frozen commit before this change, did not
# release (two legs of four) and is spent. Only the route takes the new stamp: the
# gate roots of the round stay ``2026-10-13_10``.
#
# MOVED 2026-10-13 to this module (owner ruling R22, the route-stamp registry): ROUND is
# the placeholder until the R22 round's freeze sets it. ``_2026-10-13_10`` and
# ``_2026-10-13_20`` are spent (the second released on the frozen commit of the records
# the series carries).
