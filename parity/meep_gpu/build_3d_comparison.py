"""One table, two hosts: this package's routes beside stock MEEP, in Mcell-steps/s.

WHY ONE SCRIPT. The comparison is run on two machines -- an NVIDIA host (one RTX
A6000 beside up to 96 Xeon cores) and an Apple host (an M1 Max, Metal beside 8
performance cores) -- and it is only comparable if every column means the same
thing on both. So both hosts' rows go through this one reader, which takes the
same inputs from each: ``bench_fused_products`` rows for our routes and the stock-MEEP
bench's ``rows_r<N>.jsonl`` for MEEP at N ranks. Nothing is transcribed by hand.

A RATIO IS FORMED ONLY BETWEEN TWO TIMINGS OF THE SAME SIMULATION. Until 2026-10-02 this
reader paired a GPU row with a MEEP row by cell count alone, and the 2026-09-22 MEEP
rows it paired timed a different problem (a 1 um epsilon-12 block, PML 1.0, no monitor)
from the GPU rows' ``pml_3d`` at the same cell counts. Now a GPU row and a MEEP size
pair only when

* their case labels are equal (the GPU row's ``case``, or its ``timing_case.name``;
  the MEEP rows' ``case``), and
* the GPU row's geometry digest -- read from the harness-lift digest files under
  ``--digests`` (``digest_harness_lift.py`` output: case, resolution, cells, digest) --
  equals the ONE digest every MEEP pass at that size carries, and that lift's driver
  agreed with the MEEP object it lifted;
* both sides carried the same monitors (the GPU row timed with them attached);
* for a case of ``timing_cases.py``, the GPU row, the lift and the MEEP rows were built
  by the same builder source;
* MEEP's best rank count at that size passes the 5 % gate (every pass's within-window
  spread, and the spread of its pass medians).

Otherwise the pair is REFUSED BY NAME, both values still shown and its ratio cell
reading ``refused: <reason>``:

=====================================  ===============================================
``case-label-mismatch``                the two sides name different cases at that size
``meep-digest-missing``                a MEEP row at that size carries no digest (rows
                                       written before the identical-case bench)
``gpu-digest-missing``                 no harness-lift digest for the GPU row's case
                                       and cell count under ``--digests``
``lift-driver-disagrees``              the harness lift's driver disagreed with the MEEP
                                       object it lifted (grid, dt, PML, sources,
                                       monitors), so its digest names no GPU simulation
``digest-mismatch``                    the digests differ (both are printed)
``meep-digests-disagree-across-passes``  MEEP passes at that size carry different digests
``meep-configurations-mixed``          rows at one (case, cells, ranks) ran different
                                       MEEP configurations (build, threads per rank,
                                       chunk splitting, stepper, binding); a median
                                       that blends them is not taken
``gpu-monitors-detached``              the GPU row timed without its monitors, or with a
                                       monitor count other than the MEEP rows' DFT
                                       objects
``builder-mismatch``                   the GPU row's builder source differs from the
                                       lift's or the MEEP rows' (cases of
                                       ``timing_cases.py``; a built-in case's rows carry
                                       no builder hash and are tied by the ladder's code
                                       digest instead)
``meep-best-fails-gate``               MEEP's fastest rank count at that size fails the
                                       5 % gate; it is never replaced by the next one
=====================================  ===============================================

MEEP ROWS THAT FAILED A FLOOR are dropped before any median: a row the bench struck
(its monitor not stepped) or whose monitor shows no sign of having been stepped.

The summary line reads "paired k of D; refused m of D" and the exit status is non-zero
when m > 0.

WHAT EACH COLUMN IS.
* ``MEEP@N``  stock MEEP 1.33.0 at N MPI ranks, single precision, its own step loop:
              the MEDIAN over the passes (fresh processes) that (case, cells, ranks)
              carries.
* ``array``   our engine's host array path (NumPy on the Apple host, CuPy on NVIDIA).
* ``<route>`` our fused dispatch route on the named kernel table, monitors ATTACHED --
              the number a user experiences, which is the conservative one. ONE GPU PASS
              PER ROUTE DIRECTORY: the last row of each case directory is read, so a
              route directory is one pass; ``build_identical_case_table.py`` reads every
              pass of a ladder.
* ``vs best`` the route against MEEP's fastest rank count AT THAT SIZE, never against
              one rank. A ratio against a configuration nobody runs is not a claim.
* ``verdict`` the bench's own: only ``TIMED`` rows are citable. Rows that failed a
              floor are shown, struck through in the notes, and never enter a ratio.

Usage:
  build_3d_comparison.py --host <label> --routes <name>=<dir> [...] --meep <dir>
      --digests <dir> --out <md> [--json <json>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Every reason a pair can be refused for, in the order they are checked.
REFUSAL_REASONS = ("case-label-mismatch", "meep-configurations-mixed",
                   "meep-digest-missing", "meep-digests-disagree-across-passes",
                   "gpu-digest-missing", "lift-driver-disagrees", "digest-mismatch",
                   "gpu-monitors-detached", "builder-mismatch", "meep-best-fails-gate")

#: The gate every MEEP leg (one rank count) must pass: each pass's within-window spread
#: and the spread of its pass medians, both (max - min) / median.
GATE = 0.05

#: The MEEP configuration a row that records none of its own ran (the 2026-09-28 bench).
REFERENCE_CONFIGURATION = {"build": "reference", "threads_per_rank": 1,
                           "split_chunks_evenly": "default", "stepper": "fields_step",
                           "binding": None}


def cells_of(row: Dict[str, Any]) -> int:
    count = 1
    for axis in row.get("grid_shape") or []:
        count *= int(axis)
    return count


def case_of(row: Dict[str, Any]) -> Optional[str]:
    return row.get("case") or (row.get("timing_case") or {}).get("name")


def read_route(directory: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """(case, cells) -> {fused, array, unfused (ms/step), verdict, copies, tables}.

    One pass: the LAST row of each ``<directory>/*/rows.jsonl``.
    """
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for rows in sorted(glob.glob(os.path.join(directory, "*", "rows.jsonl"))):
        with open(rows, "r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        if not lines:
            continue
        row = lines[-1]
        cells = cells_of(row)
        case = case_of(row)
        if not cells or not case:
            continue
        legs = row.get("per_leg") or {}

        def ms(leg: str) -> Optional[float]:
            value = (legs.get(leg) or {}).get("median_seconds_per_step")
            return None if value is None else value * 1e3

        attached = (row.get("monitors_attached") or {}).get("fused")
        out[(case, cells)] = {
            "case": case, "cells": cells,
            "fused_ms": ms("fused"), "array_ms": ms("array"), "unfused_ms": ms("unfused"),
            "verdict": row.get("verdict"),
            "floors_failed": [k for k, v in (row.get("floors") or {}).items() if v is False],
            "copies": (legs.get("fused") or {}).get("residency_copies_per_step"),
            "tables": (row.get("substitution") or {}).get("tables_dispatched"),
            "monitors": row.get("monitors_mode"),
            "monitor_count": (sum(int(v) for v in attached.values()
                                  if isinstance(v, (int, float)))
                              if isinstance(attached, dict) else None),
            "builder_source_sha256": (row.get("timing_case") or {}).get(
                "builder_source_sha256"),
            "bit_identical": (row.get("bit_identity") or {}).get("fused_vs_array_identical"),
        }
    return out


def read_digests(directory: Optional[str]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """(case, cells) -> the harness lift's digest, from ``harness_lift_digest*.json``.

    Both file names ``digest_harness_lift.py`` has written are read
    (``harness_lift_digest_res<r>.json`` and ``harness_lift_digest_<case>_res<r>.json``);
    the case, resolution and cells are read from the file, never from its name. Two
    files that disagree for one (case, cells) are refused.
    """
    found: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not directory:
        return found
    for path in sorted(glob.glob(os.path.join(directory, "**", "harness_lift_digest*.json"),
                                 recursive=True)):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        geometry = record.get("geometry") or {}
        cells = (geometry.get("facts") or {}).get("cells")
        digest = geometry.get("digest")
        case = record.get("case")
        if not (case and cells and digest):
            continue
        key = (str(case), int(cells))
        if key in found and found[key]["digest"] != digest:
            raise SystemExit(f"REFUSING: two harness-lift digest files disagree for {key}: "
                             f"{found[key]['file']} and {path}")
        verdict = record.get("verdict") or {}
        found[key] = {"digest": digest, "resolution": record.get("resolution"),
                      "file": os.path.abspath(path),
                      # A LIFT WHOSE DRIVER DISAGREED WITH THE MEEP OBJECT is kept, so the
                      # pair can be refused by name, and never pairs.
                      "driver_agrees": verdict.get("driver_agrees") is True,
                      "builder_source_sha256": (record.get("builder") or {}).get(
                          "builder_source_sha256")}
    return found


def configuration_of(row: Dict[str, Any]) -> str:
    """The configuration a MEEP measurement ran, as a stable string."""
    found = dict(REFERENCE_CONFIGURATION)
    for key, value in (row.get("configuration") or {}).items():
        if key in found and value is not None:
            found[key] = value
    if row.get("stepper"):
        found["stepper"] = row["stepper"]
    if found["binding"] is None:
        found["binding"] = (row.get("environment") or {}).get("binding_label")
    return json.dumps(found, sort_keys=True)


def spread(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    median = statistics.median(values)
    return (max(values) - min(values)) / median if median else float("inf")


def failed_floor(row: Dict[str, Any]) -> Optional[str]:
    """Why a MEEP measurement row enters no median, or ``None``. Rows written before
    the identical-case bench carry no ``monitor`` record and are judged by their
    digest instead (they are refused as ``meep-digest-missing``)."""
    if row.get("struck"):
        return f"struck: {row['struck']}"
    monitor = row.get("monitor")
    if monitor is not None and monitor.get("stepped") is not True:
        return "the monitor shows no sign of having been stepped"
    return None


def read_meep(directory: str) -> Dict[Tuple[str, int], Dict[int, Dict[str, Any]]]:
    """(case, cells) -> {ranks -> {median, passes, digests, configurations, mixed, gate}}.

    A (case, cells, ranks) carries one measurement row per pass (fresh process) and the
    value of record is their MEDIAN, never the last one written. A row that failed a
    floor (:func:`failed_floor`) is DROPPED first and counted. Rows at one (case,
    cells, ranks) that ran different configurations are MARKED ``mixed`` and their
    median is not taken (``meep-configurations-mixed``). A row's digest is
    ``geometry_digest`` (``None`` on rows written before the identical-case bench).
    ``gate`` is the 5 % gate on the leg: every pass's within-window spread and the
    spread of its pass medians.
    """
    groups: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    dropped: Dict[Tuple[str, int, int], List[str]] = {}
    for path in sorted(glob.glob(os.path.join(directory, "rows_r*.jsonl"))):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("row") != "measurement" or row.get("dimensions") != 3:
                    continue
                key = (str(row.get("case")), int(row["cells"]), int(row["ranks"]))
                why = failed_floor(row)
                if why:
                    dropped.setdefault(key, []).append(f"{row.get('row_id')}: {why}")
                    continue
                groups.setdefault(key, []).append(row)
    out: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = {}
    for (case, cells, ranks), rows in groups.items():
        configurations = sorted({configuration_of(r) for r in rows})
        mixed = len(configurations) > 1
        values = [float(r["mcell_steps_per_s"]) for r in rows]
        within = [r.get("spread") for r in rows]
        strikes = [f"within-window spread {'unread' if w is None else format(w, '.1%')} "
                   f"in pass {i}" for i, w in enumerate(within, 1)
                   if w is None or float(w) > GATE]
        between = spread(values) if len(values) > 1 else None
        if between is not None and between > GATE:
            strikes.append(f"between-pass spread {between:.1%}")
        counts = sorted({int((r.get("monitors_attached") or {}).get("dft_objects"))
                         for r in rows
                         if (r.get("monitors_attached") or {}).get("dft_objects")
                         is not None})
        builders = sorted({str(((r.get("environment") or {}).get("builder") or {}).get(
            "builder_source_sha256")) for r in rows
            if ((r.get("environment") or {}).get("builder") or {}).get(
                "builder_source_sha256")})
        out.setdefault((case, cells), {})[ranks] = {
            "median": None if mixed else statistics.median(values),
            "passes": len(values), "values": values,
            "digests": sorted({str(r.get("geometry_digest")) for r in rows
                               if r.get("geometry_digest")}),
            "digest_missing": any(not r.get("geometry_digest") for r in rows),
            "configurations": [json.loads(c) for c in configurations], "mixed": mixed,
            "within_window_spread": within, "between_pass_spread": between,
            "passes_gate": not strikes, "gate_strikes": strikes,
            "dropped": dropped.get((case, cells, ranks), []),
            "dft_objects": counts, "builder_source_sha256": builders,
        }
    return out


def rate(cells: int, ms: Optional[float]) -> Optional[float]:
    return None if not ms else cells / (ms * 1e-3) / 1e6


def pair_reasons(gpu_case: str, cells: int,
                 meep: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]],
                 digests: Dict[Tuple[str, int], Dict[str, Any]],
                 gpu: Optional[Dict[str, Any]] = None
                 ) -> Tuple[List[str], Dict[str, Any]]:
    """Every reason the GPU row (case, cells) may not be divided by MEEP at ``cells``."""
    reasons: List[str] = []
    facts: Dict[str, Any] = {}
    at_size = {case: by_rank for (case, c), by_rank in meep.items() if c == cells}
    if gpu_case not in at_size:
        reasons.append("case-label-mismatch")
        facts["meep_cases_at_size"] = sorted(at_size)
    groups = list(at_size.get(gpu_case, {}).values()) or [
        g for by_rank in at_size.values() for g in by_rank.values()]
    if any(g["mixed"] for g in groups):
        reasons.append("meep-configurations-mixed")
    if any(g["digest_missing"] for g in groups):
        reasons.append("meep-digest-missing")
    meep_digests = sorted({d for g in groups for d in g["digests"]})
    facts["meep_digests"] = meep_digests
    if len(meep_digests) > 1:
        reasons.append("meep-digests-disagree-across-passes")
    lift = digests.get((gpu_case, cells))
    facts["gpu_digest"] = lift["digest"] if lift else None
    if lift is None:
        reasons.append("gpu-digest-missing")
    else:
        if not lift.get("driver_agrees", True):
            reasons.append("lift-driver-disagrees")
        if len(meep_digests) == 1 and meep_digests[0] != lift["digest"]:
            reasons.append("digest-mismatch")
            facts["digest_mismatch"] = (f"GPU {lift['digest'][:16]} / MEEP "
                                        f"{meep_digests[0][:16]}")
    if gpu is not None:
        counts = sorted({c for g in groups for c in g.get("dft_objects") or []})
        facts["monitors"] = {"gpu_mode": gpu.get("monitors"),
                             "gpu_count": gpu.get("monitor_count"), "meep_dft_objects": counts}
        if gpu.get("monitors") != "attached" or (
                counts and gpu.get("monitor_count") is not None
                and counts != [int(gpu["monitor_count"])]):
            reasons.append("gpu-monitors-detached")
        ours = gpu.get("builder_source_sha256")
        theirs = {lift.get("builder_source_sha256")} if lift else set()
        theirs |= {b for g in groups for b in g.get("builder_source_sha256") or []}
        theirs.discard(None)
        if ours and theirs and theirs != {ours}:
            reasons.append("builder-mismatch")
            facts["builders"] = {"gpu": ours, "others": sorted(theirs)}
    measured = {n: g for n, g in at_size.get(gpu_case, {}).items()
                if g["median"] is not None}
    if measured:
        best_ranks = max(measured, key=lambda n: measured[n]["median"])
        facts["meep_best_ranks"] = best_ranks
        if not measured[best_ranks]["passes_gate"]:
            reasons.append("meep-best-fails-gate")
            facts["meep_best_gate"] = measured[best_ranks]["gate_strikes"]
    return reasons, facts


def build(host: str, routes: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]],
          meep: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]],
          digests: Dict[Tuple[str, int], Dict[str, Any]]
          ) -> Tuple[List[str], Dict[str, Any]]:
    ranks = sorted({n for by in meep.values() for n in by})
    cases = sorted({case for case, _c in meep} | {case for r in routes.values()
                                                    for case, _c in r})
    lines: List[str] = [f"## {host}", ""]
    record: Dict[str, Any] = {"host": host, "ranks": ranks, "sections": {},
                              "refused_pairs": []}
    notes: List[str] = []
    paired = refused = considered = 0
    meep_sizes = {c for _case, c in meep}
    for case in cases:
        sizes = sorted({c for k, c in meep if k == case}
                       | {c for r in routes.values() for k, c in r if k == case})
        header = (["cells"] + [f"MEEP@{n}" for n in ranks] + ["array"]
                  + list(routes) + [f"{r} vs best" for r in routes])
        lines += [f"### `{case}`", "",
                  "| " + " | ".join(header) + " |",
                  "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
        entries: List[Dict[str, Any]] = []
        for cells in sizes:
            by_rank = meep.get((case, cells), {})
            medians = {n: g["median"] for n, g in by_rank.items() if g["median"] is not None}
            best = max(medians.values()) if medians else None
            best_rank = next((n for n, v in medians.items() if v == best), None)
            cols = [f"{cells:,}"] + [
                (f"{by_rank[n]['median']:.1f}" if n in by_rank and by_rank[n]["median"]
                 is not None else ("mixed" if n in by_rank else "—")) for n in ranks]
            array_rates = [rate(cells, r[(case, cells)]["array_ms"]) for r in routes.values()
                           if (case, cells) in r and r[(case, cells)]["array_ms"]]
            array = sum(array_rates) / len(array_rates) if array_rates else None
            cols.append(f"{array:.1f}" if array else "—")
            entry: Dict[str, Any] = {"case": case, "cells": cells,
                                     "meep": {n: g["median"] for n, g in by_rank.items()},
                                     "meep_passes": {n: g["passes"] for n, g in by_rank.items()},
                                     "meep_best_ranks": best_rank, "array_mcell_s": array,
                                     "routes": {}}
            ratios = []
            for name, table in routes.items():
                row = table.get((case, cells))
                if not row or not row["fused_ms"]:
                    cols.append("—")
                    ratios.append("—")
                    continue
                fused = rate(cells, row["fused_ms"])
                citable = row["verdict"] == "TIMED"
                cols.append(f"{fused:.1f}" if citable else f"~~{fused:.1f}~~")
                route_entry: Dict[str, Any] = {
                    "fused_mcell_s": fused, "verdict": row["verdict"],
                    "floors_failed": row["floors_failed"],
                    "copies_per_step": row["copies"], "tables": row["tables"],
                    "monitors": row["monitors"], "bit_identical": row["bit_identical"],
                    "vs_best_meep": None, "refused": None}
                if cells in meep_sizes:
                    considered += 1
                    reasons, facts = pair_reasons(case, cells, meep, digests, row)
                    route_entry["pairing"] = facts
                    if reasons or best is None:
                        reasons = reasons or ["meep-configurations-mixed"]
                        refused += 1
                        route_entry["refused"] = reasons
                        record["refused_pairs"].append({
                            "case": case, "cells": cells, "route": name,
                            "reasons": reasons, **facts})
                        shown = ", ".join(reasons)
                        if "digest-mismatch" in reasons:
                            shown += f" ({facts['digest_mismatch']})"
                        ratios.append(f"refused: {shown}")
                    else:
                        paired += 1
                        route_entry["vs_best_meep"] = fused / best
                        ratios.append(f"**{fused / best:.2f}×**" if citable
                                      else f"~~{fused / best:.2f}×~~")
                else:
                    ratios.append("—")
                if not citable:
                    notes.append(f"{case} {cells:,} cells, {name}: {row['verdict']} "
                                 f"(floors failed: {', '.join(row['floors_failed']) or 'none'})"
                                 " — struck through, enters no ratio")
                entry["routes"][name] = route_entry
            cols += ratios
            lines.append("| " + " | ".join(cols) + " |")
            entries.append(entry)
        record["sections"][case] = entries
        lines.append("")
    lines.append("MEEP@N is stock MEEP 1.33.0 single precision at N MPI ranks, the median over "
                 "the passes each (case, cells, ranks) carried. `array` is our host array "
                 "path. Route columns are the fused dispatch route with monitors attached, "
                 "ONE GPU PASS PER ROUTE DIRECTORY (the last row of each case directory); "
                 "`vs best` is against MEEP's fastest rank count at that size, formed only "
                 "when the case labels and the geometry digests of the two sides agree.")
    multi = sorted((case, c, n, g["passes"]) for (case, c), by in meep.items()
                   for n, g in by.items() if g["passes"] > 1)
    if multi:
        lines.append("")
        lines.append("MEEP passes per (case, cells, ranks): "
                     + ", ".join(f"{case} {c:,}@{n} x{k}" for case, c, n, k in multi) + ".")
    if notes:
        lines += ["", "**Rows that are not citable:**"] + [f"- {n}" for n in notes]
    summary = f"paired {paired} of {considered}; refused {refused} of {considered}"
    lines += ["", f"**Pairs**: {summary}."]
    record.update({"paired": paired, "refused": refused, "denominator": considered,
                   "summary": summary})
    return lines, record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--routes", nargs="+", required=True,
                        help="name=dir pairs, e.g. metal=results/.../metal_held")
    parser.add_argument("--meep", required=True, help="dir holding rows_r<N>.jsonl")
    parser.add_argument("--digests", default=None,
                        help="dir holding the harness-lift digest files "
                             "(harness_lift_digest*.json); without it every pair is "
                             "refused as gpu-digest-missing")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    routes = {}
    for pair in args.routes:
        name, directory = pair.split("=", 1)
        routes[name] = read_route(directory)
    lines, record = build(args.host, routes, read_meep(args.meep),
                          read_digests(args.digests))
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=str)
    print("\n".join(lines))
    print(record["summary"])
    return 1 if record["refused"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
