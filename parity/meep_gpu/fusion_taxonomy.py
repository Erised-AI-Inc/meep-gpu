#!/usr/bin/env python3
"""ONE TAXONOMY FOR THE THREE FUSION BOARDS — the shared bucket names, the three
tiers, and the floors that make both refusable.

WHY THIS FILE EXISTS. Metal, Triton and CUDA price the SAME seam-instances (597 since
the H->D seam was priced on 2026-09-04; 403 from the 2026-09-03 basis extension to
then, 387 before it) and until 2026-09-02 priced them under
three different bucket vocabularies, three different ceiling definitions and two
different counting rules. The (row, seam) key
sets are identical on all three boards — verified by reconstructing each board's
census per row and joining — so every difference between their headlines was a
difference in what the boards CALLED things, and the three numbers could not be
compared or summed honestly. The release decision was to reconcile them to one
taxonomy and to introduce a denominator that counts the seams actually DESIRABLE to
fuse rather than seams writ large.

THIS MODULE IS THE ONE PLACE THE VOCABULARY LIVES. A board classifies its own
instances — it is the only thing that can, since the evidence is per-backend — and
hands each one to :meth:`Taxonomy.add` under one of the eight names below, WITH the
board's own prose kept as the sub-reason. Three boards importing one name list is
what stops a fourth vocabulary from growing back: a bucket name not in
:data:`BUCKETS` raises here rather than appearing in a headline.

THE SUB-REASON IS NOT DECORATION. The "why" text on these boards is the most
valuable thing on them — it is where the measurement lives ("MEASURED by
perturbation on the array path", "the constitutive arm is a STENCIL over the curl
arm's in-place output", "the composer selects no arm at update_E"). A
reconciliation that replaced those sentences with eight nouns would be a loss
dressed as a gain. So every instance carries its board's own sentence into the
report, counted, under the reconciled bucket.

--------------------------------------------------------------------------------
THE FOUR SEAMS
--------------------------------------------------------------------------------

The driver's timestep (``driver.py``, ``step()``) has four boundaries a fused launch
could cross, and every one is priced on every row that has it:

``B->H``  step_B -> update_H, the magnetic curl into its constitutive; the magnetic
          injection, the B fill, the B wall clear and the folded far fill sit inside.
``D->E``  step_D -> update_E, the electric twin; the electric injection and the D
          passes sit inside.
``E->P``  update_E -> update_P, the polarization chain; nothing sits inside, and the
          seam exists only on a row carrying a susceptibility.
``H->D``  update_H -> step_D, the magnetic constitutive into the ELECTRIC CURL — the
          boundary that closes the loop and that no board priced until 2026-09-04
          (the fusion-residue audit's §2 "never priced" finding). Exactly ONE pass
          sits inside it: ``for source in electric: getattr(source, "withdraw",
          _no_withdraw)(self.fields)`` — the integrated electric sources returning
          their standing dipole offset to D before the curl ladder reads it. No
          injection, no fill, no wall clear. It is read off ``driver.py`` at cut time
          by ``h_to_d_seam.driver_seam_fact`` and refused if anything else appears
          there. Its per-row presence is MEASURED on the lifted row by
          ``probe_h_to_d_seam.py`` (an integrated electric source owning at least one
          deposit point), never read off a source's name.

The first three were the three seams of every board through 2026-09-03; the fourth
is priced by the same three boards from ``h_to_d_seam.py``, one rule for three
backends, off the same census rows.

--------------------------------------------------------------------------------
THE EIGHT BUCKETS (release decision R1), in PRECEDENCE order
--------------------------------------------------------------------------------

``not_fusion_surface``
    The constitutive sub-step launches NOTHING, so one launch already performs the
    entire seam and there is no second kernel to weld. This is not a refusal — there
    is no fusion to refuse.

``served``
    A shipped product's own predicate admits this instance.

``source_seam_unbracketable``
    The driver injects a source between the halves and no repair can reconstruct
    it (``deposit_repair.repairable`` refuses BY NAME: an off-diagonal chi1inv, an
    instantaneous chi2/chi3, an unreadable fold map).

``withdraw_seam``
    The H->D analogue of the source seam, and filed only there: the row carries an
    integrated ELECTRIC source owning at least one deposit point, so the driver's
    withdraw pass runs between ``update_H`` and ``step_D`` and returns the standing
    dipole to D before the curl reads it. The withdraw touches D only and reads
    nothing ``update_H`` writes (``sources.VolumeSource.withdraw``), so a fused
    launch running AFTER it, or carrying it, is plausibly order-free — but "plausibly"
    is not a measurement, and this package claims byte identity only from a device
    gate. Until the device probe the audit asked for exists (a fused
    update_H+step_D launch byte-compared against the array path on a row with an
    integrated electric source), these instances are priced as not attainable, named
    for what sits in them. Every instance MUST carry the count of integrated electric
    sources that put it here; a board filing one without it raises in
    :meth:`Taxonomy.finish`.

``missing_half``
    A KERNEL COVERAGE GAP, not a fusion refusal: no single certified arm admits one
    of the seam's two slots, so the pair cannot be built because a HALF is absent.
    Every instance in this bucket MUST NAME THE MISSING KERNEL — a board that files
    an instance here without one raises in :meth:`Taxonomy.finish`.

``structurally_unweldable``
    A weld is a meaningful object here and cannot be built: the constitutive half is
    a stencil over the curl's in-place output, or the per-component interleave
    reorders a dependency the driver's order guarantees. Carry the sub-reason — the
    two are different measurements.

``platform_specific``
    Genuinely per-backend and NEVER silently merged with the above. Metal's
    31-binding argument-table ceiling is the real case; the other two backends
    report zero here WITH the reason they have no analogue, so a zero means
    "measured, none" rather than "the board forgot to ask".

``buildable_not_built``
    Every precondition met, nobody built it.

PRECEDENCE (release decision R4). An instance is counted ONCE, in its most fundamental
bucket, and the ladder above is that order. ``withdraw_seam`` sits where the source
seam sits, ahead of ``missing_half``: like the source seam it is a fact about the
SEAM-INSTANCE (the same on every backend), and a coverage gap beneath it would be
reported as buildable work on a seam no launch has been shown to cross.
``not_fusion_surface`` outranks
``source_seam_unbracketable``: if there is nothing to weld, the deposit in the seam
is irrelevant, and a board that priced such an instance as source-blocked was
reporting an obstruction that obstructs nothing. ``served`` sits second and not
first for one reason — an instance whose constitutive half launches nothing cannot
be served, and if some board ever says otherwise that is a contradiction this
module must surface rather than absorb, which is exactly what the served floor in
:meth:`Taxonomy.finish` does. On the H->D seam the same floor cannot see it — a
board's H->D served total is derived from the very walk that files these instances —
so ``h_to_d_seam.price`` raises there instead, on the same argument.

AND ``served`` OUTRANKS ``withdraw_seam``, WHICH IS NOT FREE. A product whose
predicate admits one of the withdraw rows takes that instance out of the bucket whose
whole content is "a pass the driver runs between the halves that no launch has been
measured across" AND out of the subtraction ``ATTAINABLE`` makes for it — so the
seam's own refusal would be discharged by a product rather than by a measurement. The
precedence stands (a served instance is served), and the price of it is paid where the
verdict is filed: ``h_to_d_seam.price`` refuses a served verdict on a withdraw row
that does not NAME the device measurement licensing it, and the instance carries its
integrated-source count into the served bucket rather than losing it.

--------------------------------------------------------------------------------
THE THREE TIERS (release decision R2) — each printed WITH ITS DENOMINATOR NAMED
--------------------------------------------------------------------------------

``PRICED``      the corpus fact: rows x 2 curl->constitutive seams + rows x 1
                constitutive->curl seam (H->D) + susceptibility rows x 1 E->P seam.
                194 x 3 + 15 = 597 seam-instances since the H->D seam was priced on
                2026-09-04 (194 x 2 + 15 = 403 from the 2026-09-03 basis extension
                to then; 186 x 2 + 15 = 387 before it). DERIVED from the census by
                :func:`denominator_justification`, never transcribed, and floored in
                :class:`Taxonomy` against the priced count the board hands in — a
                board that hands in seam counts without the H->D seam is refused,
                so the old three-seam denominator cannot grow back.
``FUSABLE``     PRICED - not_fusion_surface. The seams where a weld is a MEANINGFUL
                OBJECT — the new denominator, and the one the headline now reads
                against. Counting a seam whose second half launches nothing as an
                unclosed fusion gap was the old boards' quiet overstatement of the
                work remaining.
``ATTAINABLE``  FUSABLE - source_seam_unbracketable - withdraw_seam. What is still
                gettable with work, and it is a property of the SEAM-INSTANCE, so it
                reads the same on every backend (359 on the three-seam 194-row basis
                since 2026-09-03, 344 on the 186-row basis before it; with the H->D
                seam priced it gains 194 minus the H->D null-constitutive rows minus
                the withdraw rows, both measured, both identical across backends).
                The withdraw subtraction is the same kind of fact as the source
                seam's — a pass the driver runs between the halves that no launch has
                been measured across — and it comes back the day the device probe
                releases it.

                IT DOES NOT SUBTRACT ``structurally_unweldable``, and that was a
                real defect in this tier's first definition, caught by the
                2026-09-02 adversarial pass. "Structurally unweldable" is a fact
                about a BACKEND AND A WELD SHAPE, not about the seam: every
                instance filed under it is SERVED TODAY by a shipped, gate-passing
                product on a sibling backend, and the boards' own sub-reasons say
                so. Subtracting it made Metal's headline read TO_DO 0 — "nothing
                left to do" — while Metal's own prose named the buildable shape and
                credited CUDA with building it. The only refusal that is genuinely
                instance-intrinsic is the source seam, which R3 proved identical
                across all three backends (24 instances, the same 24, on the 186-row
                basis; 25 since the 2026-09-03 basis extension added coupler.py's D->E
                seam, again the same 25 on every backend). So
                ATTAINABLE is backend-independent by construction, and ``served``
                and ``TO_DO`` are left as the only backend-dependent numbers —
                which is what a reconciliation should produce.

Headline coverage is ``served / FUSABLE``. The to-do list is ``ATTAINABLE - served``.

--------------------------------------------------------------------------------
THE FLOORS — this module REFUSES, it does not warn
--------------------------------------------------------------------------------

1. The eight buckets must total PRICED exactly, and must total each seam's own
   instance count exactly. An instance in no bucket or in two makes the
   decomposition stop measuring, and a board that printed a headline over one would
   be publishing a number nothing checked.
2. The ``served`` bucket must equal the board's own independently computed served
   total. This is the reconciliation's own safety catch: a taxonomy that changes
   what is SERVED is a bug, not a result, and this is where such a bug stops.
3. Every ``missing_half`` instance must name the kernel that is missing.
4. Every bucket name must be one of the eight.
5. ``platform_specific`` must carry a stated reason on every board, including the
   boards that report zero there.
6. Every ``withdraw_seam`` instance must sit on the H->D seam and name the
   integrated electric source(s) that put it there; the board must price the H->D
   seam at all (one instance per row), or the denominator is the old three-seam one.
7. Every served total the ARTIFACT publishes — under any of the names in
   :data:`SERVED_HEADLINE_KEYS`, at any depth — equals the served bucket, whose
   instances are listed beside it (:func:`headline_agreement`). Added 2026-09-06,
   when the first boards with an H->D product carried a four-seam served bucket
   beside three headline keys the three-seam walk still totalled: Metal 405 / 358,
   Triton 403 / 356, CUDA 483 / 359, each in one artifact. The headline is DERIVED
   from the instance ledger now, and a cut where any published total disagrees is
   refused rather than written.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["BUCKETS", "PER_ROW_SEAMS", "SERVED_HEADLINE_KEYS", "Taxonomy",
           "canonical_seam", "credit_withheld_sub_reason", "headline_agreement"]

#: The eight reconciled bucket names, in PRECEDENCE order (release decision R1 + R4;
#: ``withdraw_seam`` added 2026-09-04 with the H->D seam). A board files each
#: instance under exactly one of these. The order here IS the precedence: a
#: classifier walks it top to bottom and stops at the first match.
BUCKETS: Sequence[str] = (
    "not_fusion_surface",
    "served",
    "source_seam_unbracketable",
    "withdraw_seam",
    "missing_half",
    "structurally_unweldable",
    "platform_specific",
    "buildable_not_built",
)

#: THE SEAM NAMES, CANONICALISED. The Metal and CUDA boards spell the seams
#: ``B_to_H``/``D_to_E``/``E_to_P`` and the Triton board spells them
#: ``B->H``/``D->E``/``E->P``. Two spellings is enough to stop a join dead, and the
#: whole value of this taxonomy is that the three boards can be joined at instance
#: granularity — so the spelling is normalised here rather than left to whoever
#: writes the next comparison script.
_CANONICAL_SEAM = {"B->H": "B_to_H", "D->E": "D_to_E", "E->P": "E_to_P",
                   "H->D": "H_to_D"}

#: The seams every row has exactly one instance of: the two curl->constitutive seams
#: and the constitutive->curl seam that closes the loop. E->P is the one seam whose
#: instance count is not the row count.
PER_ROW_SEAMS: Sequence[str] = ("B_to_H", "D_to_E", "H_to_D")


def canonical_seam(seam: str) -> str:
    return _CANONICAL_SEAM.get(seam, seam)


def credit_withheld_sub_reason(product: str) -> str:
    """THE ONE WITHHELD SENTENCE.

    Filed under ``buildable_not_built`` by the Triton three-seam walk (a product whose
    predicate admits the row but whose release binding is stale — the board will not
    CREDIT it) and by :func:`h_to_d_seam.classify` on the fourth seam, where it is the
    third per-row answer beside ``served_by`` = name / None. The bucket stays ONE
    vocabulary because both call this and neither carries a copy (pinned by
    ``test_fusion_taxonomy``, which scans both files for the literal): a reader
    summing ``sub_reasons["buildable_not_built"]`` across the four seams sees one
    withheld key, not a spelling per seam. The withdraw facts on an H->D row ride in
    the instance's detail, never in this sentence, for the same reason.

    This is the string ``results/fusion_matrix_triton_2026-09-08_regate`` filed 70
    times, byte for byte (em dash, no trailing period).
    """
    return (f"a product exists on this cell ({product}) and its predicate ADMITS, but "
            f"its RELEASE BINDING is stale — credit withheld. The KERNEL is not what "
            f"is missing; a re-gate against the shipped bytes is")


#: THE KEYS UNDER WHICH A BOARD PUBLISHES ITS SERVED TOTAL, and the reason this
#: tuple exists is the 2026-09-06 H->D landing: every board summed its served bucket
#: over four seams while its headline keys were still totalled by the three-seam
#: walk, so ``fusion_matrix_metal_2026-09-06_hd`` read ``served_by_predicate: 358``
#: beside ``taxonomy.buckets.served: 405`` (Triton 356 / 403, CUDA
#: ``served_by_a_fused_product_today`` 359 / 483). The headline keys had never been
#: taught the fourth seam. Each board now writes these keys FROM
#: ``taxonomy_block["buckets"]["served"]`` and :func:`headline_agreement` walks the
#: finished artifact for every occurrence of each name, at any depth, and refuses the
#: cut where one disagrees with the instance ledger. A board's THREE-seam total is a
#: different fact and lives under a key that is not in this tuple
#: (``served_three_seam_ledger``).
#:
#: A NAME ADDED TO A BOARD'S HEADLINE FOR ITS SERVED TOTAL GOES HERE IN THE SAME
#: CHANGE. The walk cannot see a key it was not told about; that is how the split
#: this floor closes came to exist.
SERVED_HEADLINE_KEYS: Sequence[str] = (
    "served_by_predicate",            # the Metal board's headline
    "served_by_a_fused_product",      # Metal top level; the Triton aggregate
    "served_by_a_fused_product_today",  # the CUDA board's headline
    "seam_credited_total",            # the Metal disjointness cross-check's credit side
    "served_counted",                 # dispatch_reachability: what the board handed it
)


def _group_served_by_product(rows: Sequence[Dict[str, Any]]) -> Dict[Any, List[Any]]:
    grouped: Dict[Any, List[Any]] = {}
    for r in rows:
        if r["bucket"] != "served":
            continue
        grouped.setdefault(r["detail"].get("product"), []).append((r["seam"], r["row"]))
    return grouped


def _walk(node: Any, path: str, found: List[Any]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in SERVED_HEADLINE_KEYS:
                found.append((here, value))
            _walk(value, here, found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", found)


def headline_agreement(result: Dict[str, Any], block: Dict[str, Any],
                       backend: str) -> Dict[str, Any]:
    """FLOOR 7 — every served total the artifact publishes equals the instance ledger.

    Walks ``result`` -- the whole artifact a board is about to write -- for every
    occurrence of a :data:`SERVED_HEADLINE_KEYS` name at any depth and compares it to
    ``block["buckets"]["served"]``, the count of instances the taxonomy filed served
    (and whose ``instances["served"]`` list is in the same block). Refuses the cut on
    the first disagreement, NAMING the path, and refuses an artifact that publishes
    no served key at all: a headline the walk cannot find is a headline it cannot
    check. Returns the block the board writes as ``headline_agreement`` -- the number,
    the paths checked and the two ledgers it is derived from -- so a reader can see
    the check ran rather than trust that it did.

    ``block`` must be a finished taxonomy block; the served list and the count are
    cross-checked here too, so the number the headline derives from is one a list of
    instances stands behind.
    """
    served = int(block["buckets"]["served"])
    listed = block.get("instances", {}).get("served")
    if listed is None or len(listed) != served:
        raise SystemExit(
            f"{backend}: the taxonomy block counts {served} served instances and "
            f"lists {None if listed is None else len(listed)}; the headline cannot "
            f"derive from a ledger that does not list what it counts.")
    found: List[Any] = []
    _walk(result, "", found)
    if not found:
        raise SystemExit(
            f"{backend}: the artifact publishes none of {list(SERVED_HEADLINE_KEYS)}. "
            f"A served total under a name this floor was not told about is a headline "
            f"nothing checks; add the key to fusion_taxonomy.SERVED_HEADLINE_KEYS.")
    disagreeing = [(path, value) for path, value in found if value != served]
    if disagreeing:
        shown = ", ".join(f"{path}={value!r}" for path, value in disagreeing[:6])
        raise SystemExit(
            f"{backend}: the taxonomy files {served} seam-instances as SERVED over "
            f"{len(block['per_seam'])} seams and the artifact publishes a different "
            f"served total at {len(disagreeing)} path(s): {shown}. A headline is "
            f"DERIVED from the instance ledger, never totalled by a second walk -- the "
            f"2026-09-06 boards carried the three-seam walk's total under these keys "
            f"beside a four-seam bucket, and that is the split this floor refuses.")
    return {
        "served": served,
        "derived_from": "taxonomy.buckets.served == len(taxonomy.instances.served)",
        "per_seam": {seam: int(counts.get("served", 0))
                     for seam, counts in block["per_seam"].items()},
        "paths_checked": [path for path, _value in found],
        "agree": True,
    }


def denominator_justification(rows: int, susceptibility_rows: int) -> str:
    """The PRICED sentence, derived from the census's own counts.

    ONE HOME, DERIVED. Until 2026-09-03 this sentence was transcribed into each of the
    three boards with ``186`` and ``387`` spelled by hand, so a census with a different
    row count would have computed one denominator and written a justification naming
    another into the same artifact. Since 2026-09-04 it derives the fourth seam the
    same way it derives the first three.
    """
    priced = len(PER_ROW_SEAMS) * rows + susceptibility_rows
    slots = 4 * rows + susceptibility_rows
    return (
        f"{rows} rows x 2 curl->constitutive seams (B->H, D->E) + {rows} rows x 1 "
        f"constitutive->curl seam (H->D: update_H -> step_D, priced since 2026-09-04) + "
        f"{susceptibility_rows} rows carrying a susceptibility x 1 E->P chain seam = "
        f"{priced} seam-instances. A fused product serves a ROW at a SEAM, not a slot; "
        f"the driver's timestep (driver.py, step()) has exactly these four boundaries "
        f"between consecutive sub-steps. B->H and D->E are bounded by the two source "
        f"injections that cannot be crossed inside one launch; H->D holds only the "
        f"integrated electric sources' withdraw (present on a row only where such a "
        f"source owns a deposit point, MEASURED by probe_h_to_d_seam.py); E->P holds "
        f"nothing and is counted only where an update_P pass exists, the same way the "
        f"{slots}-slot denominator adds {susceptibility_rows} rather than {rows} for "
        f"that sub-step. {priced} is NOT a partition of {slots} and is not meant to be.")


#: What each bucket means, emitted into every board's JSON so a reader of one board
#: never has to find another to learn the vocabulary.
BUCKET_MEANING: Dict[str, str] = {
    "not_fusion_surface": (
        "the constitutive sub-step launches NOTHING, so one launch already performs "
        "the entire seam and there is no second kernel to weld — not a refusal, an "
        "absence of the object"),
    "served": "a shipped product's own predicate admits this seam-instance",
    "source_seam_unbracketable": (
        "the driver injects a source between the halves and deposit_repair.repairable "
        "refuses to reconstruct it BY NAME (off-diagonal chi1inv / instantaneous "
        "chi2-chi3 / unreadable fold map)"),
    "withdraw_seam": (
        "H->D only: the row carries an integrated ELECTRIC source owning a deposit "
        "point, so the driver's withdraw pass runs between update_H and step_D "
        "(returning the standing dipole to D before the curl reads it). A fused "
        "update_H+step_D launch has not been byte-measured across or after that "
        "pass on any backend — the device probe the audit asked for — so the "
        "instance is priced as not attainable, named for what sits in it"),
    "missing_half": (
        "A KERNEL COVERAGE GAP, NOT A FUSION REFUSAL: no certified arm admits one of "
        "the seam's two slots, so the pair cannot be built because a HALF is absent. "
        "Every instance names the missing kernel"),
    "structurally_unweldable": (
        "a weld is a meaningful object here and cannot be built — the constitutive "
        "half is a stencil over the curl's in-place output, or the per-component "
        "interleave reorders a dependency the driver's order guarantees"),
    "platform_specific": (
        "a refusal that belongs to THIS BACKEND and to no other, reported under its "
        "own name and never merged with a structural one"),
    "buildable_not_built": "every precondition met, nobody built it",
}


class Taxonomy:
    """One board's instances, filed under the reconciled names.

    Usage: construct with the backend name, the priced denominator, the census row
    count and the per-seam instance counts; call :meth:`add` once per seam-instance,
    then :meth:`finish` — which raises on any floor breach and otherwise returns the
    JSON block, and :meth:`report` to print it. ``finish`` must be called before
    ``report``.
    """

    def __init__(self, backend: str, priced: int, rows: int, platform_note: str,
                 seam_instances: Dict[str, int]) -> None:
        self.backend = backend
        self.priced = int(priced)
        self.rows = int(rows)
        #: What this backend's ``platform_specific`` bucket asks. Required even when
        #: the count is zero: a zero with no stated question is indistinguishable
        #: from a board that never asked.
        self.platform_note = platform_note
        #: seam -> the number of instances the board prices at that seam, used by
        #: the per-seam floor and to derive the PRICED sentence.
        self.seam_instances = {canonical_seam(k): v for k, v in seam_instances.items()}
        # THE PRICED FLOOR: the denominator the board hands in must be the one its
        # own rows imply. A board whose universe drifted from its census would
        # otherwise print a correct-looking sentence beside a number it does not
        # explain.
        susceptibility_rows = int(self.seam_instances.get("E_to_P", 0))
        # THE FOURTH SEAM IS NOT OPTIONAL. A board that hands in B->H and D->E alone
        # is pricing the three-seam denominator every board carried through
        # 2026-09-03 — the one the fusion-residue audit's §2 found left the
        # update_H -> step_D boundary unpriced — and it is refused here by name
        # rather than accepted with a smaller floor.
        absent = [seam for seam in PER_ROW_SEAMS if seam not in self.seam_instances]
        if absent:
            raise SystemExit(
                f"{backend}: the board prices no instances at {absent}. Every row has "
                f"exactly one instance of each of {list(PER_ROW_SEAMS)}; a board "
                f"without the H->D seam is publishing the pre-2026-09-04 three-seam "
                f"denominator, and the taxonomy will not carry it.")
        short = {seam: int(self.seam_instances[seam]) for seam in PER_ROW_SEAMS
                 if int(self.seam_instances[seam]) != self.rows}
        if short:
            raise SystemExit(
                f"{backend}: {short} instances priced at per-row seams against "
                f"{self.rows} rows; each of {list(PER_ROW_SEAMS)} has exactly one "
                f"instance per row, so one of the walks is missing rows.")
        implied = len(PER_ROW_SEAMS) * self.rows + susceptibility_rows
        if implied != self.priced:
            raise SystemExit(
                f"{backend}: the board prices {self.priced} seam-instances but its "
                f"{self.rows} rows x {len(PER_ROW_SEAMS)} + {susceptibility_rows} "
                f"E->P rows imply {implied}. The denominator and the rows it is "
                f"explained by must come from the same census.")
        self.priced_justification = denominator_justification(
            self.rows, susceptibility_rows)
        self._rows: List[Dict[str, Any]] = []
        self._finished: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- filing
    def add(self, bucket: str, seam: str, row: str, sub_reason: str,
            detail: Optional[Dict[str, Any]] = None) -> None:
        """File one seam-instance.

        ``sub_reason`` is THE BOARD'S OWN SENTENCE, carried verbatim. It is not a
        label this module knows about and never a paraphrase: the reconciliation
        renames the bucket and keeps the measurement.
        """
        if bucket not in BUCKETS:
            raise SystemExit(
                f"{self.backend}: {bucket!r} is not one of the reconciled buckets "
                f"{list(BUCKETS)}. A board may not introduce an eighth name — that "
                f"is the divergence this taxonomy exists to close.")
        if not sub_reason:
            raise SystemExit(
                f"{self.backend}: an instance was filed under {bucket!r} with no "
                f"sub-reason. The prose is the measurement; a bucket without one is "
                f"a rename that lost the evidence.")
        self._rows.append({"bucket": bucket, "seam": canonical_seam(seam),
                           "row": row, "sub_reason": sub_reason,
                           "detail": detail or {}})

    # ---------------------------------------------------------------- floors
    def finish(self, served_expected: int,
               served_expected_source: str) -> Dict[str, Any]:
        """Run every floor and return the JSON block.

        ``served_expected`` is the board's OWN served total, computed by the board's
        own walk, and ``served_expected_source`` names where it came from. The two
        must agree exactly: this is the catch that stops a reconciliation from
        moving the one number it may not move.
        """
        counts = collections.Counter(r["bucket"] for r in self._rows)
        total = sum(counts.values())

        # FLOOR 1 — THE DECOMPOSITION MUST ADD UP.
        if total != self.priced:
            raise SystemExit(
                f"{self.backend}: the reconciled buckets total {total} against "
                f"{self.priced} PRICED seam-instances. "
                f"{abs(self.priced - total)} instance(s) fall into no bucket or into "
                f"two, and the decomposition has stopped measuring. Buckets: "
                f"{dict(counts)}")
        per_seam = collections.Counter(
            (r["seam"], r["bucket"]) for r in self._rows)
        seam_totals = collections.Counter(r["seam"] for r in self._rows)
        for seam, expected in self.seam_instances.items():
            if seam_totals[seam] != expected:
                raise SystemExit(
                    f"{self.backend}: the reconciled buckets file "
                    f"{seam_totals[seam]} instances at {seam} against the "
                    f"{expected} the board prices there. Per-seam: "
                    f"{dict(seam_totals)}")
        unknown = sorted(set(seam_totals) - set(self.seam_instances))
        if self.seam_instances and unknown:
            raise SystemExit(
                f"{self.backend}: instances filed at seam(s) {unknown} the board "
                f"does not price. The taxonomy and the census disagree about the "
                f"corpus.")

        # FLOOR 2 — SERVED MAY NOT MOVE BY RELABELLING.
        if counts["served"] != served_expected:
            raise SystemExit(
                f"{self.backend}: the reconciled taxonomy files "
                f"{counts['served']} seam-instances as SERVED and the board's own "
                f"walk ({served_expected_source}) says {served_expected}. A "
                f"reconciliation that changes what is SERVED is a bug, not a "
                f"result: a bucket above SERVED in the precedence ladder is "
                f"claiming an instance a shipped predicate admits, which means that "
                f"bucket's premise has stopped being true.")

        # FLOOR 3 — missing_half MUST NAME THE MISSING KERNEL.
        unnamed = [r for r in self._rows if r["bucket"] == "missing_half"
                   and not r["detail"].get("missing_kernel")]
        if unnamed:
            raise SystemExit(
                f"{self.backend}: {len(unnamed)} missing_half instance(s) name no "
                f"missing kernel (first: {unnamed[0]['row']} @ "
                f"{unnamed[0]['seam']}). missing_half is a COVERAGE gap and its "
                f"whole content is WHICH kernel is absent; without that it is an "
                f"unactionable synonym for 'not built'.")

        # FLOOR 6 — withdraw_seam sits on H->D only and names what put it there.
        # It is the fourth seam's own refusal: a board filing it elsewhere is
        # confusing it with the source seam, and one filing it without the count of
        # integrated electric sources has lost the measurement the bucket exists to
        # carry (probe_h_to_d_seam.py measures it per lifted row).
        misfiled = [r for r in self._rows if r["bucket"] == "withdraw_seam"
                    and r["seam"] != "H_to_D"]
        if misfiled:
            raise SystemExit(
                f"{self.backend}: {len(misfiled)} withdraw_seam instance(s) filed at "
                f"{sorted({r['seam'] for r in misfiled})} (first: "
                f"{misfiled[0]['row']}). The withdraw sits between update_H and "
                f"step_D and nowhere else; an injection seam is "
                f"source_seam_unbracketable.")
        unnamed_withdraw = [r for r in self._rows if r["bucket"] == "withdraw_seam"
                            and not r["detail"].get("integrated_electric_sources")]
        if unnamed_withdraw:
            raise SystemExit(
                f"{self.backend}: {len(unnamed_withdraw)} withdraw_seam instance(s) "
                f"name no integrated electric source (first: "
                f"{unnamed_withdraw[0]['row']}). The bucket's whole content is the "
                f"MEASURED presence of the withdraw on the lifted row; without the "
                f"count it is a label read off a name.")

        # FLOOR 4 — platform_specific must carry a stated question, even at zero.
        if not self.platform_note:
            raise SystemExit(
                f"{self.backend}: platform_specific carries no stated reason. A "
                f"zero with no question asked is indistinguishable from a board "
                f"that never asked, and merging a platform refusal into a "
                f"structural one is the failure this bucket exists to prevent.")

        fusable = self.priced - counts["not_fusion_surface"]
        # NOT minus structurally_unweldable — see this module's docstring. That
        # bucket is a (backend, weld-shape) fact and every instance in it is served
        # by a sibling backend today, so subtracting it hid real work. MINUS
        # withdraw_seam, since 2026-09-04: like the source seam it is a pass the
        # driver runs between two halves that no launch has been measured across,
        # and it is a fact about the row, identical on every backend.
        attainable = (fusable - counts["source_seam_unbracketable"]
                      - counts["withdraw_seam"])
        if fusable <= 0 or attainable < 0:
            raise SystemExit(
                f"{self.backend}: FUSABLE {fusable} / ATTAINABLE {attainable} is "
                f"not a usable denominator; the tiers are degenerate.")

        sub_reasons: Dict[str, Dict[str, int]] = {}
        for bucket in BUCKETS:
            here = collections.Counter(r["sub_reason"] for r in self._rows
                                       if r["bucket"] == bucket)
            sub_reasons[bucket] = dict(here.most_common())

        self._finished = {
            "backend": self.backend,
            "bucket_names": list(BUCKETS),
            "bucket_meaning": BUCKET_MEANING,
            "precedence": (
                "an instance is counted ONCE, in its most fundamental bucket; the "
                "order of bucket_names IS the precedence. not_fusion_surface "
                "outranks source_seam_unbracketable: where there is nothing to weld "
                "the deposit in the seam is irrelevant (release decision R4)"),
            "buckets": {bucket: counts[bucket] for bucket in BUCKETS},
            "sub_reasons": sub_reasons,
            "per_seam": {
                seam: {bucket: per_seam[(seam, bucket)] for bucket in BUCKETS
                       if per_seam[(seam, bucket)]}
                for seam in sorted(seam_totals)},
            "platform_specific_question": self.platform_note,
            # THE ROWS THEMSELVES, for EVERY bucket. Counts alone made the three
            # boards look reconciled without anyone being able to check it: "19
            # not_fusion_surface" on three boards is consistent with three DIFFERENT
            # sets of 19. These lists are what turn the reconciliation into
            # something joinable at instance granularity, which is how the
            # 2026-09-02 audit established the key sets were identical in the first
            # place. SERVED was omitted through 2026-09-06 on the argument that the
            # other seven lists plus the denominator determine it -- they do not:
            # they determine it only together with the UNIVERSE of instances, which
            # the block did not carry, so an instance-by-instance diff of two boards
            # had to rebuild the universe from each board's own per-row tables (three
            # different shapes). The served list is the instance ledger the headline
            # now derives from (:func:`headline_agreement`), so it is written.
            "instances": {
                bucket: sorted((r["seam"], r["row"]) for r in self._rows
                               if r["bucket"] == bucket)
                for bucket in BUCKETS},
            # The served ledger BY PRODUCT, off each served instance's own detail --
            # the same map the boards' ledgers print, derived here so the two cannot
            # disagree. An instance filed served with no product in its detail is
            # listed under None rather than dropped.
            "served_by_product": {
                str(product): sorted(items) for product, items in
                _group_served_by_product(self._rows).items()},
            "missing_half_detail": [
                {"row": r["row"], "seam": r["seam"], **r["detail"]}
                for r in self._rows if r["bucket"] == "missing_half"],
            "structurally_unweldable_detail": [
                {"row": r["row"], "seam": r["seam"], "why": r["sub_reason"],
                 **r["detail"]}
                for r in self._rows if r["bucket"] == "structurally_unweldable"],
            "platform_specific_detail": [
                {"row": r["row"], "seam": r["seam"], "why": r["sub_reason"],
                 **r["detail"]}
                for r in self._rows if r["bucket"] == "platform_specific"],
            "tiers": {
                "PRICED": {
                    "value": self.priced,
                    "denominator": self.priced,
                    "denominator_is": "itself — the corpus's own seam count",
                    "definition": self.priced_justification,
                },
                "FUSABLE": {
                    "value": fusable,
                    "denominator": self.priced,
                    "denominator_is": f"{self.priced} PRICED seam-instances",
                    "definition": (
                        f"PRICED - not_fusion_surface "
                        f"({self.priced} - {counts['not_fusion_surface']}) = "
                        f"{fusable}. The seams where a weld is a MEANINGFUL OBJECT"),
                },
                "ATTAINABLE": {
                    "value": attainable,
                    "denominator": fusable,
                    "denominator_is": f"{fusable} FUSABLE seam-instances",
                    "definition": (
                        f"FUSABLE - source_seam_unbracketable - withdraw_seam "
                        f"({fusable} - {counts['source_seam_unbracketable']} - "
                        f"{counts['withdraw_seam']}) = {attainable}. "
                        f"What is still gettable with work. Backend-independent by "
                        f"construction: the source seam and the H->D withdraw seam "
                        f"are the only refusals that are a property of the "
                        f"seam-instance rather than of a backend and a weld shape, "
                        f"so this reads the same on all three boards and only "
                        f"served/TO_DO differ. "
                        f"structurally_unweldable ("
                        f"{counts['structurally_unweldable']}) is deliberately NOT "
                        f"subtracted — every instance in it is served by a sibling "
                        f"backend today, so it is REMAINING WORK, not a refusal"),
                },
                "COVERAGE": {
                    "value": counts["served"],
                    "denominator": fusable,
                    "denominator_is": f"{fusable} FUSABLE seam-instances",
                    "fraction": round(counts["served"] / fusable, 4),
                    "definition": (
                        "served / FUSABLE — the headline. NOT served / PRICED, which "
                        "counted seams with no second kernel as unclosed gaps"),
                },
                "TO_DO": {
                    "value": attainable - counts["served"],
                    "denominator": attainable,
                    "denominator_is": f"{attainable} ATTAINABLE seam-instances",
                    "definition": "ATTAINABLE - served — the work that remains",
                },
            },
            "served_floor": {
                "taxonomy_says": counts["served"],
                "the_board_s_own_walk_says": served_expected,
                "read_from": served_expected_source,
                "agree": True,
            },
        }
        return self._finished

    # ---------------------------------------------------------------- report
    def report(self, emit) -> None:
        """Print the taxonomy and the three tiers. ``emit`` is the board's own
        flushing print/log, so this text lands in the board's log rather than in a
        second stream."""
        if self._finished is None:
            raise SystemExit(f"{self.backend}: report() before finish(); the floors "
                             f"have not run and no number here may be printed.")
        block = self._finished
        counts = block["buckets"]
        emit("")
        emit("=" * 78)
        emit(f"THE RECONCILED TAXONOMY (R1) — {self.backend}   "
             f"denominator {self.priced} PRICED seam-instances")
        emit("=" * 78)
        emit("  one name list on all three boards; each board's own prose kept "
             "beneath it as the sub-reason")
        for bucket in BUCKETS:
            emit(f"\n  {counts[bucket]:4d} / {self.priced}   {bucket.upper()}")
            emit(f"        {BUCKET_MEANING[bucket]}")
            for reason, n in block["sub_reasons"][bucket].items():
                emit(f"        {n:4d}  {reason}")
            if bucket == "platform_specific" and not counts[bucket]:
                emit(f"        (zero, MEASURED not assumed) "
                     f"{self.platform_specific_line()}")
        emit(f"\n  BUCKET FLOOR: the {len(BUCKETS)} buckets total "
             f"{sum(counts.values())} == the {self.priced} PRICED seam-instances")
        emit(f"  SERVED FLOOR: {counts['served']} == the "
             f"{block['served_floor']['the_board_s_own_walk_says']} the board's own "
             f"walk reports ({block['served_floor']['read_from']})")

        emit("")
        emit("=" * 78)
        emit(f"THE THREE TIERS (R2) — {self.backend}   every tier with its "
             f"denominator NAMED")
        emit("=" * 78)
        for name in ("PRICED", "FUSABLE", "ATTAINABLE", "COVERAGE", "TO_DO"):
            tier = block["tiers"][name]
            pct = (f"  ({100.0 * tier['value'] / tier['denominator']:.1f}%)"
                   if tier["denominator"] else "")
            emit(f"  {name:11s} {tier['value']:4d} / {tier['denominator']:4d}   "
                 f"[denominator: {tier['denominator_is']}]{pct}")
            emit(f"              {tier['definition']}")

    def platform_specific_line(self) -> str:
        return self.platform_note
