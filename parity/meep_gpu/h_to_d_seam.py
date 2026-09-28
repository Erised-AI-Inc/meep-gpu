#!/usr/bin/env python3
"""THE FOURTH SEAM — update_H -> step_D — ONE PRICING RULE FOR THREE BOARDS.

WHY THIS FILE EXISTS. The fusion-residue audit (§2) found the boundary between the
magnetic constitutive and the electric curl priced by no board: the three boards
walked B->H, D->E and E->P and stopped, so up to 194 seam-instances per backend sat
outside every denominator. Pricing it in three builders separately would have grown
three rules; this module is the one rule, and each builder hands it the two arms it
selects at ``update_H`` and ``step_D`` on each row plus its own null-constitutive
verdict, and receives the bucket, the sentence and the detail to file.

WHAT SITS IN THE SEAM is not transcribed: :func:`driver_seam_fact` locates the two
consults inside ``FdtdDriver.step`` by ast and RAISES unless the only statements
between them are the electric integrated-source withdraw loop. A driver that grew a
pass there would stop every board rather than be priced against a seam it no longer
runs.

WHETHER THE WITHDRAW RUNS ON A ROW is measured, not read off a name:
``probe_h_to_d_seam.py`` lifts every row of the basis under the census machinery and
records each source's ``is_integrated`` flag, its ``withdraw`` attribute and its
deposit-point count; :func:`probe_rows` reads that artifact and :func:`join` refuses
BY NAME any board row the probe did not measure. A row the probe could not lift is an
absence, never a gap (the 2026-09-03 phantom-halves lesson).

THE BUCKETS, in the taxonomy's precedence:

``not_fusion_surface``   ``update_H`` launches nothing on this row — the board's own
                          null-constitutive verdict, the same rule it applies to B->H
                          — so ``step_D`` alone performs the whole seam.
``served``               a product spanning update_H + step_D admits this row, BY THE
                          BOARD'S OWN PREDICATE WALK (``entries[i]["served_by"]``).
                          This module cannot evaluate a predicate — only the board
                          can, since the evidence is per-backend — so it carries the
                          verdict and files it second, where the taxonomy's precedence
                          puts it. No product spans the seam today, and that is
                          ASSERTED rather than assumed: with no spanning product a
                          served verdict is refused, and with one the board must file
                          a verdict on EVERY row or :func:`price` names the rows it
                          owes (see :func:`price`) — and, since 2026-09-10, a THIRD
                          answer beside ``served_by`` = name / None:
                          ``credit_withheld``, the board's reason sentence, for a
                          row the predicate ADMITS whose product's release binding
                          is stale. It is decided at this rung and filed under
                          ``buildable_not_built`` under the one withheld sentence
                          (``fusion_taxonomy.credit_withheld_sub_reason``) the
                          Triton three-seam walk files, so withholding moves an
                          instance served -> buildable_not_built and nothing else.
``withdraw_seam``        the row carries an integrated electric source owning a
                          deposit point, so the withdraw runs between the halves.
                          Not attainable until a device probe byte-compares a fused
                          launch across or after it — which this module does not run.
``missing_half``         the composer selects no arm at one of the two slots.
``buildable_not_built``  both halves admitted, nothing between them, nobody built it.
                          The weld shape is NOT an in-place splice — ``step_D``'s curl
                          reads H at neighbouring cells ``update_H`` writes in the
                          same launch — so the buildable shape is the halo recompute
                          of the pointwise H that the audit named. Or a product
                          admits it and its credit is withheld
                          (``detail["credit_withheld"]`` True, the product still
                          named).

WHY ``served`` SITS SECOND AND WHAT IT SWALLOWS. Precedence is the taxonomy's
(``fusion_taxonomy.BUCKETS``), and it is matched here exactly so one instance lands in
one bucket: ``not_fusion_surface`` outranks ``served`` — a seam whose first half
launches nothing cannot be served, and a board claiming otherwise is a contradiction
:func:`price` raises on rather than absorbs — and ``served`` outranks
``withdraw_seam``. That second ordering has teeth: a product admitting one of the
withdraw rows takes the instance OUT of the bucket whose whole content is "a pass the
driver runs between the halves that no launch has been measured across", and out of
the subtraction ``ATTAINABLE`` makes for it. So a served verdict on a withdraw row
must name the measurement that licenses it (``served_over_the_withdraw``), or
:func:`price` refuses it by name.

WHERE A WITHHELD CREDIT SITS, AND WHY IT IS NOT WHERE THE THREE-SEAM WALK PUTS IT.
The fourth seam decides ``credit_withheld`` AT THE SERVED RUNG — above
``withdraw_seam``, so a withheld product on a withdraw row is still filed
``buildable_not_built`` and ``ATTAINABLE`` is unchanged by withholding. That is
deliberate here: the row's attainability across the withdraw was measured by the
product's own licensed gate, and the binding is what is stale, not the measurement.
So such a row still owes the ``served_over_the_withdraw`` licence (it becomes served
the moment the table entry is deleted; better it fails now than then). The Triton
THREE-seam walk files withheld BELOW ``source_seam_unbracketable`` — a withheld
product on a source-blocked row lands in that bucket there, not in
``buildable_not_built`` — and no board can reach the divergent case today (the
withheld table is empty; the three seams have no withdraw bucket). "Withholding
moves nothing else" is therefore a statement about THIS seam. ``not_fusion_surface``
outranks a withheld verdict as it outranks a credited one: ``served_by`` on a
null-update_H row is the same contradiction whichever way the board answers.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Sibling module: every consumer already has parity/meep_gpu on sys.path to import
# this file (the builders insert HERE; test_h_to_d_seam inserts its own directory).
import fusion_taxonomy

_HERE = Path(__file__).resolve().parent
_API = _HERE.parents[1]
RESULTS = _HERE / "results"
DRIVER = _API / "meep_gpu" / "driver.py"

SEAM = "H_to_D"
HALVES: Tuple[str, str] = ("update_H", "step_D")

#: The probe campaign the boards read. Overridable so a re-cut on a later tree can be
#: priced without editing this file; dated so the standing one is never overwritten.
PROBE_DIR = RESULTS / (os.environ.get("MEEP_GPU_H_TO_D_PROBE") or "h_to_d_seam_2026-09-04")

#: The statements the driver runs between the two consults, as they must read once
#: comments are stripped. Anything else between them is a pass this pricing does not
#: know about, and :func:`driver_seam_fact` refuses rather than prices around it.
EXPECTED_BETWEEN: Tuple[str, ...] = (
    "update_H(self.fields, self.pml)",
    "for source in electric:",
    'getattr(source, "withdraw", _no_withdraw)(self.fields)',
)

__all__ = ["SEAM", "HALVES", "PROBE_DIR", "driver_seam_fact", "probe_rows", "join",
           "classify", "price"]


# --- the driver fact -----------------------------------------------------------------


def driver_seam_fact(driver_path: Path = DRIVER) -> Dict[str, Any]:
    """Locate the update_H and step_D consults inside ``step`` and return what sits
    between them; RAISE unless it is exactly the electric withdraw loop.

    Located by ast inside the ONE method named ``step`` (the driver carries a second
    dispatch-guarded update_H in ``synchronize_magnetic_fields``, whose seam has no
    step_D at all), by the two ``fast.dispatch("...")`` consults, so a renamed or
    moved pass fails here by name rather than being priced as absent.
    """
    source = driver_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    spans = [(node.lineno, node.end_lineno or node.lineno)
             for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == "step"]
    if len(spans) != 1:
        raise SystemExit(f"{driver_path.name} declares {len(spans)} functions named "
                         f"`step`; the H->D seam is located inside exactly one")
    start, end = spans[0]

    def consult(slot: str) -> int:
        hits = [i + 1 for i in range(start - 1, end)
                if f'fast.dispatch("{slot}"' in lines[i]]
        if len(hits) != 1:
            raise SystemExit(f"{driver_path.name} `step` (lines {start}-{end}) has "
                             f"{len(hits)} dispatch consults for {slot}; expected one")
        return hits[0]

    h_at, d_at = consult("update_H"), consult("step_D")
    if not h_at < d_at:
        raise SystemExit(f"update_H consult at :{h_at} does not precede the step_D "
                         f"consult at :{d_at}; the timestep's shape has changed")
    between = []
    for number in range(h_at + 1, d_at):
        text = lines[number - 1].split("#", 1)[0].strip()
        if text:
            between.append((number, text))
    statements = tuple(text for _n, text in between)
    if statements != EXPECTED_BETWEEN:
        raise SystemExit(
            f"between the update_H consult (:{h_at}) and the step_D consult (:{d_at}) "
            f"{driver_path.name} runs {list(statements)}, not the electric withdraw "
            f"loop alone {list(EXPECTED_BETWEEN)}. The H->D seam holds a pass this "
            f"pricing does not know about; it cannot be priced until the rule is "
            f"re-derived.")
    return {
        "driver": str(driver_path.relative_to(_API)),
        "step_span": [start, end],
        "update_H_consult": f"driver.py:{h_at}",
        "step_D_consult": f"driver.py:{d_at}",
        "between": [{"line": n, "statement": t} for n, t in between],
        "between_summary": ("the array-path update_H call under its consult, then "
                            "`for source in electric: getattr(source, 'withdraw', "
                            "_no_withdraw)(self.fields)` — the integrated electric "
                            "sources' withdraw and nothing else"),
    }


# --- the probe artifact --------------------------------------------------------------


def _load(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def probe_rows(probe_dir: Path = PROBE_DIR) -> Dict[str, dict]:
    """label -> the ``h_to_d_seam`` block, measured rows only, the parameterised leg's
    matched rows substituted — the same rows() rule the boards read censuses by."""
    if not (probe_dir / "examples.jsonl").is_file():
        raise SystemExit(
            f"{probe_dir} carries no examples.jsonl: the H->D seam probe has not been "
            f"cut there. Run parity/meep_gpu/probe_h_to_d_seam.py (or point "
            f"MEEP_GPU_H_TO_D_PROBE at a cut); the seam is priced from a measurement "
            f"or not at all.")
    record = _load(probe_dir / "examples.jsonl") + _load(probe_dir / "tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in _load(probe_dir / "tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    out: Dict[str, dict] = {}
    for row in record:
        if not row.get("measured") or "h_to_d_seam" not in row:
            continue
        out[f"{row['leg']}:{row['row']}"] = row["h_to_d_seam"]
    return out


def probe_provenance(probe_dir: Path = PROBE_DIR) -> Dict[str, Any]:
    digests = {}
    for leg in ("examples", "tests", "tests_param"):
        path = probe_dir / f"subject_digests_{leg}.json"
        if path.is_file():
            digests[leg] = json.loads(path.read_text(encoding="utf-8")).get("manifest_sha256")
    return {"probe": str(probe_dir), "subject_manifest_sha256_per_leg": digests,
            "measured_rows": len(probe_rows(probe_dir))}


def join(labels: Iterable[str], probe: Dict[str, dict]) -> Dict[str, dict]:
    """The probe block for every board row, or a refusal naming the rows it lacks."""
    wanted = list(labels)
    missing = sorted(set(wanted) - set(probe))
    if missing:
        raise SystemExit(
            f"the H->D probe measured no row named {missing[:6]}"
            f"{' ...' if len(missing) > 6 else ''} ({len(missing)} of {len(wanted)} "
            f"board rows). A row the probe did not lift is an ABSENCE and may not be "
            f"priced as a gap or as clean; re-cut the probe on this census's basis.")
    return {name: probe[name] for name in wanted}


# --- the rule ------------------------------------------------------------------------


def classify(row: str, update_h_arm: Optional[str], step_d_arm: Optional[str],
             update_h_is_null: bool, block: dict, backend: str,
             kernel_package: str, served_by: Optional[str] = None,
             served_over_the_withdraw: Optional[str] = None,
             credit_withheld: Optional[str] = None,
             ) -> Tuple[str, str, Dict[str, Any]]:
    """One seam-instance -> (bucket, the sentence, the detail), in taxonomy precedence.

    ``served_by`` is THE BOARD'S OWN VERDICT — the name of the product spanning
    update_H+step_D whose predicate admits this row, or None. It is not re-derived
    here: a predicate is per-backend and this is the one rule three boards share.
    ``served_over_the_withdraw`` names the measurement licensing a served verdict on a
    row whose seam holds the withdraw; :func:`price` is what requires it.
    ``credit_withheld`` is the board's REASON (a sentence, e.g.
    ``CREDIT_WITHHELD_STALE_BINDING[served_by]``) for withholding credit from a row
    ``served_by`` admits; None means credited. With it set the instance files
    ``buildable_not_built`` under ``fusion_taxonomy.credit_withheld_sub_reason``,
    decided here at the served rung.
    """
    withdraws = [s for s in block.get("sources", ())
                 if s.get("electric") and s.get("withdraw_does_work")]
    if update_h_is_null:
        return ("not_fusion_surface",
                "NOT A FUSION CANDIDATE — update_H launches nothing on this row (the "
                "null magnetic constitutive: stepping.update_H returns before its "
                "first statement without an active PML), so step_D alone performs the "
                "whole H->D seam and there is no second kernel to weld",
                {"constitutive_slot": "update_H", "constitutive_arm": update_h_arm,
                 "array_path_returns_early":
                     block["update_H_array_path"]["returns_before_its_first_statement"],
                 "poison_measurement_changed_an_H_array":
                     block["update_H_array_path"]["poison_measurement"].get("changed")})
    if served_by and credit_withheld:
        # DECIDED HERE, AT THE SERVED RUNG; FILED EIGHTH. The predicate admits
        # (served_by is the product's own verdict and is not rewritten to None), the
        # board will not credit it, and the sentence is the three-seam walk's own so
        # the bucket stays one vocabulary. The withdraw facts ride in the detail,
        # never in the sentence.
        return ("buildable_not_built",
                fusion_taxonomy.credit_withheld_sub_reason(served_by),
                {"product": served_by,
                 "cell": [update_h_arm, step_d_arm],
                 "withdraw_in_seam": bool(block["withdraw_in_seam"]),
                 "integrated_electric_sources": len(withdraws),
                 "served_over_the_withdraw": served_over_the_withdraw,
                 "credit_withheld": True,
                 "credit_withheld_reason": credit_withheld,
                 "predicate_admits": True})
    if served_by:
        # SECOND, and only second. Above it sits the one bucket a served instance may
        # not overtake (there is no seam to serve where update_H launches nothing);
        # below it sits everything a product's own predicate is entitled to answer for.
        return ("served",
                f"SERVED by {served_by} — a product spanning update_H+step_D whose own "
                f"predicate admits this row ({update_h_arm!r} at update_H, "
                f"{step_d_arm!r} at step_D), so ONE launch performs the seam. The "
                f"verdict is the BOARD'S: this shared rule carries a per-backend "
                f"predicate walk, it does not re-run it. It is a PREDICATE verdict and "
                f"not a claim that the composer installs the product on this row — "
                f"what a composer does with it is the board's own arbitration column"
                + (f". The seam here also holds the withdraw ({len(withdraws)} "
                   f"integrated electric source(s) owning deposit points), and what "
                   f"licenses serving it across that pass is: "
                   f"{served_over_the_withdraw}"
                   if block["withdraw_in_seam"] else ""),
                {"product": served_by,
                 "cell": [update_h_arm, step_d_arm],
                 # CARRIED EVEN WHEN SERVED. `served` outranks `withdraw_seam`, so
                 # without these two keys a product admitting a withdraw row would
                 # delete the measurement that bucket exists to hold.
                 "withdraw_in_seam": bool(block["withdraw_in_seam"]),
                 "integrated_electric_sources": len(withdraws),
                 "served_over_the_withdraw": served_over_the_withdraw,
                 "served_is_a_predicate_verdict_not_an_installation": True})
    if block["withdraw_in_seam"]:
        return ("withdraw_seam",
                "the electric integrated-source withdraw sits between the halves — "
                "driver.py step(): `for source in electric: getattr(source, "
                "'withdraw', _no_withdraw)(self.fields)` — MEASURED on the lifted row: "
                f"{len(withdraws)} integrated electric source(s) owning deposit points "
                "return their standing dipole to D before step_D reads it. A fused "
                "update_H+step_D launch would have to run after that withdraw or "
                "carry it; whether either is byte-identical is the device probe this "
                "cut does not run, so the instance is priced as not attainable",
                {"integrated_electric_sources": len(withdraws),
                 "sources": [{"type": s["type"], "component": s["component"],
                              "envelope": s["envelope"],
                              "n_source_points": s["n_source_points"]}
                             for s in withdraws],
                 "cell": [update_h_arm, step_d_arm],
                 "device_probe_owed": (
                     "a fused update_H+step_D launch byte-compared against the array "
                     "path, on both float32 subnormal policies, on rows carrying an "
                     "integrated electric source, with the unbracketed launch as the "
                     "null control")})
    if update_h_arm is None or step_d_arm is None:
        missing_slot = "update_H" if update_h_arm is None else "step_D"
        partner = step_d_arm if update_h_arm is None else update_h_arm
        return ("missing_half",
                f"A KERNEL COVERAGE GAP, NOT A FUSION REFUSAL: the composer selects no "
                f"arm at {missing_slot} on this row, so the H->D pair has no second "
                f"body to weld and the array path steps the slot",
                {"missing_slot": missing_slot,
                 "missing_kernel":
                     f"a {kernel_package} arm on {missing_slot} admitting this row — "
                     + (f"the seam's other half is already covered by {partner!r}, so "
                        f"what is absent is the {missing_slot} body for that cell"
                        if partner else
                        "the seam's other half is unselected too, so BOTH bodies are "
                        "absent")
                     + ". Closing it is a SUB-STEP kernel's job, not a fusion one",
                 "partner_arm": partner})
    return ("buildable_not_built",
            f"reachable and NOT BUILT — both halves are admitted ({update_h_arm!r} at "
            f"update_H, {step_d_arm!r} at step_D), nothing runs between them on this "
            f"row (no integrated electric source: the withdraw is the no-op on every "
            f"electric source here), and no product on any backend spans "
            f"update_H+step_D. The weld is not an in-place splice: step_D's curl reads "
            f"H at neighbouring cells that update_H writes in the same launch, so the "
            f"buildable shape is the halo recompute of the pointwise H the "
            f"fusion-residue audit named (§2)",
            {"cell": [update_h_arm, step_d_arm],
             "weld_shape": "halo recompute of the pointwise magnetic constitutive; an "
                           "in-place splice would be a stencil over the first half's "
                           "output",
             "electric_sources": block["n_electric_sources"],
             "integrated_electric_sources": block["n_integrated_electric_sources"]})


def price(taxonomy, backend: str, kernel_package: str, entries: Iterable[dict],
          products_spanning_the_seam: Iterable[str] = ()) -> Dict[str, Any]:
    """File one H->D instance per entry into ``taxonomy`` and return the board block.

    ``entries``: ``{"row": label, "update_H": arm-or-None, "step_D": arm-or-None,
    "update_H_is_null": bool}`` — the board's own selection and its own null verdict —
    plus, ONCE A PRODUCT SPANS THE SEAM, ``"served_by"``: the spanning product whose
    predicate admits that row, or None where none does (and
    ``"served_over_the_withdraw"`` on a served row whose seam holds the withdraw) and,
    optionally, ``"credit_withheld"``: the board's reason sentence where ``served_by``
    admits the row but the product's release binding is stale (Triton:
    ``CREDIT_WITHHELD_STALE_BINDING[served_by]``); None or absent means credited.
    Such a row is filed buildable_not_built under the withheld sub-reason, is NOT in
    ``served`` / ``served_rows`` / ``served_by_product``, and is listed under
    ``credit_withheld_rows`` / ``credit_withheld_by_product``. It still carries
    ``served_by`` and still owes every refusal a served row owes (listed product, the
    withdraw licence, no null update_H).
    ``products_spanning_the_seam``: the board's product-table entries whose seam is
    H->D.

    WHAT THIS REFUSES, AND WHY IT IS NOT THE OLD BLANKET REFUSAL. Until 2026-09-04 the
    first spanning product stopped this walk outright, which was right while no
    ``served`` branch existed and wrong the moment one did. What stands in its place is
    narrower and says what is owed: this module cannot evaluate a per-backend
    predicate, so a board that names a spanning product must answer FOR EVERY ROW, and
    a row it leaves unanswered is an ABSENCE — priced neither as served nor as
    buildable — and is NAMED in the refusal. The mirror refusal guards the other
    direction: a served verdict with no product spanning the seam is a credit with
    nothing behind it.

    A board that answers None on every row is not refused. That is a filed measurement
    — a predicate that admits nothing on this census — and refusing it would refuse the
    honest answer; what it may not do is skip the question.
    """
    spanning = sorted(products_spanning_the_seam)
    fact = driver_seam_fact()
    entries = list(entries)
    if spanning:
        owed = [e["row"] for e in entries if "served_by" not in e]
        if owed or not entries:
            shown = owed[:8]
            raise SystemExit(
                f"{backend}: {spanning} span update_H+step_D and the board files a "
                f"served verdict on {len(entries) - len(owed)} of {len(entries)} H->D "
                f"seam-instance(s); {len(owed)} row(s) are OWED one"
                + (f": {shown}{' ...' if len(owed) > len(shown) else ''}" if owed
                   else " (the board filed no H->D instances at all)")
                + ". h_to_d_seam cannot evaluate a product's predicate — it is "
                  "per-backend and this is the one rule three boards share — so "
                  "every entry must carry `served_by`: the spanning product whose own "
                  "predicate admits that row, or None where none does. A row left "
                  "unanswered while a product that might serve it exists is an "
                  "absence, and pricing it as buildable_not_built would under-report "
                  "the seam by exactly the rows nobody asked about.")
    else:
        claimed = [e for e in entries if e.get("served_by")]
        if claimed:
            raise SystemExit(
                f"{backend}: {len(claimed)} H->D seam-instance(s) are filed served "
                f"(first: {claimed[0]['row']}, by {claimed[0]['served_by']!r}) while "
                f"the board's own product table names NO product spanning "
                f"update_H+step_D. "
                f"A served instance is a shipped product's predicate admitting it; "
                f"without the product it is a credit with nothing behind it.")
    unlisted = sorted({e.get("served_by") for e in entries
                       if e.get("served_by") and e["served_by"] not in spanning})
    if unlisted:
        raise SystemExit(
            f"{backend}: rows are filed served by {unlisted}, which the board's own "
            f"product table does not list as spanning update_H+step_D ({spanning}). "
            f"The two walks disagree about which products exist.")
    # THE THIRD ANSWER IS HELD TO ITS SHAPE, after the served_by checks above so a row
    # missing `served_by` altogether is named OWED (the more fundamental absence)
    # rather than "withheld from nobody".
    orphaned = [e["row"] for e in entries
                if e.get("credit_withheld") and not e.get("served_by")]
    if orphaned:
        raise SystemExit(
            f"{backend}: {len(orphaned)} row(s) carry `credit_withheld` with no "
            f"`served_by` (first: {orphaned[0]}). Credit is withheld FROM a product "
            f"whose predicate admits the row; a withheld verdict naming no product is "
            f"a served_by=None row wearing a second label, and 'the predicate refused' "
            f"and 'the predicate admits but the binding is stale' are different "
            f"measurements.")
    # Keyed on PRESENCE, not truthiness: only None / absent means credited. False, 0
    # and the empty sentence are refused by name — the empty sentence is exactly what
    # a table entry with no text behind it would hand over, and reading it as
    # credited would be the silent direction.
    flagged = [e for e in entries
               if "credit_withheld" in e and e["credit_withheld"] is not None
               and not (isinstance(e["credit_withheld"], str)
                        and e["credit_withheld"].strip())]
    if flagged:
        raise SystemExit(
            f"{backend}: {len(flagged)} row(s) carry `credit_withheld` as a flag "
            f"(first: {flagged[0]['row']}: {flagged[0]['credit_withheld']!r}). It is "
            f"the board's REASON — the measured drift its table records for the "
            f"product — not a boolean; a withholding with no sentence behind it "
            f"cannot be re-earned or refuted.")
    probe = join([e["row"] for e in entries], probe_rows())
    # A SERVED VERDICT ON A WITHDRAW ROW MUST NAME ITS MEASUREMENT. `served` outranks
    # `withdraw_seam`, so such a row leaves the bucket whose content is "no launch has
    # been measured across this pass" AND leaves ATTAINABLE's subtraction for it. The
    # licence is a sentence the board supplies, not a flag: the probe campaign
    # (h_to_d_withdraw_order_*) measured the hoist on the ARRAY PATH only.
    unlicensed = [e["row"] for e in entries if e.get("served_by")
                  and probe[e["row"]]["withdraw_in_seam"]
                  and not e.get("served_over_the_withdraw")]
    if unlicensed:
        raise SystemExit(
            f"{backend}: {len(unlicensed)} row(s) are filed served on a seam that "
            f"holds the withdraw (first: {unlicensed[0]}) with no "
            f"`served_over_the_withdraw` licence. Serving one of these takes it out of "
            f"the withdraw_seam bucket and out of ATTAINABLE's subtraction for it, so "
            f"the board must name what byte-measured a fused update_H+step_D launch "
            f"across or after `for source in electric: withdraw(...)` on a device. "
            f"The array-path hoist campaign does not license it.")
    instances: List[dict] = []
    counts: Dict[str, int] = {}
    for entry in entries:
        block = probe[entry["row"]]
        bucket, sentence, detail = classify(
            entry["row"], entry["update_H"], entry["step_D"],
            bool(entry["update_H_is_null"]), block, backend, kernel_package,
            served_by=entry.get("served_by"),
            served_over_the_withdraw=entry.get("served_over_the_withdraw"),
            credit_withheld=entry.get("credit_withheld"))
        expected = None
        if entry.get("served_by"):
            expected = "buildable_not_built" if entry.get("credit_withheld") else "served"
        if expected and (bucket != expected
                         or (expected == "buildable_not_built"
                             and detail.get("credit_withheld") is not True)):
            # THE ONE CONTRADICTION THE PRECEDENCE CAN PRODUCE, SURFACED NOT ABSORBED.
            # Only `not_fusion_surface` outranks `served`, and it means update_H
            # launches nothing on this row — so a product's predicate admitting it is
            # a claim about a seam with one half. fusion_taxonomy's docstring asks for
            # exactly this to be surfaced; the board's served floor cannot see it,
            # because the board's own H->D served total is derived from this walk.
            # A withheld row is held to the same contradiction, and the detail flag is
            # checked so a served_by row cannot reach buildable_not_built by falling
            # through the ladder rather than by the withheld branch.
            raise SystemExit(
                f"{backend}: {entry['row']} is filed "
                f"{'withheld from' if entry.get('credit_withheld') else 'served by'} "
                f"{entry['served_by']!r} and the precedence files it {bucket!r} "
                f"instead. update_H launches nothing on this row (the null magnetic "
                f"constitutive), so there is no second kernel for a product to weld "
                f"to and no seam for it to serve or to withhold. One of the two walks "
                f"is wrong about this row and this rule will not choose between them.")
        taxonomy.add(bucket, SEAM, entry["row"], sentence, detail)
        counts[bucket] = counts.get(bucket, 0) + 1
        instances.append({"row": entry["row"], "update_H": entry["update_H"],
                          "step_D": entry["step_D"], "bucket": bucket,
                          "served_by": entry.get("served_by"),
                          "credit_withheld": entry.get("credit_withheld") or None,
                          "withdraw_in_seam": block["withdraw_in_seam"],
                          "integrated_electric_sources":
                              block["n_integrated_electric_sources"],
                          "update_H_is_null": bool(entry["update_H_is_null"])})
    served_rows = sorted(e["row"] for e in instances if e["bucket"] == "served")
    withheld_rows = sorted(e["row"] for e in instances if e["credit_withheld"])
    withheld_products = sorted({e["served_by"] for e in instances
                                if e["credit_withheld"]})
    return {
        "seam": SEAM, "halves": list(HALVES), "backend": backend,
        "instances_priced": len(instances),
        "one_per_row": True,
        "buckets": counts,
        "served": counts.get("served", 0),
        "products_spanning_the_seam": spanning,
        "served_rows": served_rows,
        # KEYED ON THE BUCKET, as fusion_taxonomy._group_served_by_product is. Until
        # 2026-09-10 this keyed on the raw `served_by`, which coincided only because
        # the contradiction refusal above forbade a served_by row landing anywhere
        # else; a withheld row carries served_by AND is not served, so it would have
        # been counted here.
        "served_by_product": {
            product: sorted(e["row"] for e in instances
                            if e["bucket"] == "served" and e["served_by"] == product)
            for product in spanning},
        "credit_withheld_rows": withheld_rows,
        "credit_withheld_by_product": {
            product: sorted(e["row"] for e in instances
                            if e["credit_withheld"] and e["served_by"] == product)
            for product in spanning},
        # NAMED FOR THIS SEAM so it cannot be read beside the Triton aggregate's
        # three-seam `predicate_admits` (denominator 403) as the same number; its
        # denominator is `instances_priced`.
        "predicate_admits_h_to_d": len(served_rows) + len(withheld_rows),
        "served_rows_carrying_the_withdraw": sorted(
            e["row"] for e in instances
            if e["bucket"] == "served" and e["withdraw_in_seam"]),
        "served_note": (
            f"no product on this backend spans update_H+step_D (asserted against the "
            f"product table at cut time), so served is zero by measurement, not by "
            f"omission" if not spanning else
            f"{spanning} span update_H+step_D; every one of the "
            f"{len(instances)} seam-instances carries the board's own predicate "
            f"verdict, and {len(served_rows)} of them are admitted. A served instance "
            f"here is a PREDICATE verdict — whether the composer installs the product "
            f"on the row is a separate column"
            # Appended ONLY when something is withheld, so the note is byte-identical
            # to the pre-2026-09-10 one on every board while the table is empty.
            + (f"; {len(withheld_rows)} further admitted row(s) have their CREDIT "
               f"WITHHELD ({withheld_products}) and are filed buildable_not_built "
               f"under the withheld sub-reason, not served — predicate admits "
               f"{len(served_rows) + len(withheld_rows)} of {len(instances)}, "
               f"credited {len(served_rows)}"
               if withheld_rows else "")),
        "what_sits_between_the_halves": fact,
        "probe": probe_provenance(),
        "withdraw_rows": sorted(e["row"] for e in instances if e["withdraw_in_seam"]),
        "device_probe_owed": (
            "the withdraw_seam instances come back to ATTAINABLE only when a fused "
            "update_H+step_D launch is byte-compared against the array path on rows "
            "carrying an integrated electric source (both float32 subnormal policies, "
            "unbracketed launch as the null control); nothing in this cut ran on a "
            "device"),
        "instances": instances,
    }
