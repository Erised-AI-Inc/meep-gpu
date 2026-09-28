"""One table, two hosts: this package's routes beside stock MEEP, in Mcell-steps/s.

WHY ONE SCRIPT. The comparison is run on two machines -- an NVIDIA host (one RTX
A6000 beside up to 96 Xeon cores) and an Apple host (an M1 Max, Metal beside 8
performance cores) -- and it is only comparable if every column means the same
thing on both. So both hosts' rows go through this one reader, which takes the
same inputs from each: ``bench_fused_products`` rows for our routes and the stock-MEEP
bench's ``rows_r<N>.jsonl`` for MEEP at N ranks. Nothing is transcribed by hand.

WHAT EACH COLUMN IS.
* ``MEEP@N``  stock MEEP 1.33.0 at N MPI ranks, single precision, its own step loop.
* ``array``   our engine's host array path (NumPy on the Apple host, CuPy on NVIDIA).
* ``<route>`` our fused dispatch route on the named kernel table, monitors ATTACHED --
              the number a user experiences, which is the conservative one.
* ``vs best`` the route against MEEP's fastest rank count AT THAT SIZE, never against
              one rank. A ratio against a configuration nobody runs is not a claim.
* ``verdict`` the bench's own: only ``TIMED`` rows are citable. Rows that failed a
              floor are shown, struck through in the notes, and never enter a ratio.

Usage:
  build_3d_comparison.py --host <label> --routes <name>=<dir> [...] --meep <dir> --out <md>
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import os
from typing import Tuple, Any, Dict, List, Optional


def cells_of(row: Dict[str, Any]) -> int:
    count = 1
    for axis in row.get("grid_shape") or []:
        count *= int(axis)
    return count


def read_route(directory: str) -> Dict[int, Dict[str, Any]]:
    """cells -> {fused, array, unfused (ms/step), verdict, copies, tables}."""
    out: Dict[int, Dict[str, Any]] = {}
    for rows in sorted(glob.glob(os.path.join(directory, "*", "rows.jsonl"))):
        lines = [json.loads(line) for line in open(rows) if line.strip()]
        if not lines:
            continue
        row = lines[-1]
        cells = cells_of(row)
        if not cells:
            continue
        legs = row.get("per_leg") or {}

        def ms(leg: str) -> Optional[float]:
            value = (legs.get(leg) or {}).get("median_seconds_per_step")
            return None if value is None else value * 1e3

        out[cells] = {
            "fused_ms": ms("fused"), "array_ms": ms("array"), "unfused_ms": ms("unfused"),
            "verdict": row.get("verdict"),
            "floors_failed": [k for k, v in (row.get("floors") or {}).items() if v is False],
            "copies": (legs.get("fused") or {}).get("residency_copies_per_step"),
            "tables": (row.get("substitution") or {}).get("tables_dispatched"),
            "monitors": row.get("monitors_mode"),
            "bit_identical": (row.get("bit_identity") or {}).get("fused_vs_array_identical"),
        }
    return out


def read_meep(directory: str) -> Dict[int, Dict[int, float]]:
    """cells -> {ranks -> Mcell-steps/s}, from rows_r<N>.jsonl files.

    A (cells, ranks) pair can carry SEVERAL measurement rows -- the bench writes one
    per pass, and the GPU-host campaign ran two -- and the value of record is their
    MEDIAN, not the last one written. Until 2026-09-24 this reader kept whichever row
    came last, which on the 2,097,152-cell MEEP@32 pair was the higher of two passes
    by 8% (573.3 against a median of 552.1): an independent re-derivation from the
    rows caught it. ``MEEP_PASSES`` records how many passes each pair had so a table
    can say so.
    """
    passes: Dict[int, Dict[int, List[float]]] = {}
    for path in sorted(glob.glob(os.path.join(directory, "rows_r*.jsonl"))):
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row") != "measurement" or row.get("dimensions") != 3:
                continue
            passes.setdefault(int(row["cells"]), {}).setdefault(
                int(row["ranks"]), []).append(float(row["mcell_steps_per_s"]))
    MEEP_PASSES.clear()
    out: Dict[int, Dict[int, float]] = {}
    for cells, by_rank in passes.items():
        for ranks, values in by_rank.items():
            out.setdefault(cells, {})[ranks] = statistics.median(values)
            MEEP_PASSES[(cells, ranks)] = len(values)
    return out


#: (cells, ranks) -> number of measurement passes the median above was taken over.
MEEP_PASSES: Dict[Tuple[int, int], int] = {}


def rate(cells: int, ms: Optional[float]) -> Optional[float]:
    return None if not ms else cells / (ms * 1e-3) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--routes", nargs="+", required=True,
                        help="name=dir pairs, e.g. metal=results/.../metal_held")
    parser.add_argument("--meep", required=True, help="dir holding rows_r<N>.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    routes = {}
    for pair in args.routes:
        name, directory = pair.split("=", 1)
        routes[name] = read_route(directory)
    meep = read_meep(args.meep)
    ranks = sorted({n for by in meep.values() for n in by})
    sizes = sorted(set(meep) | {c for r in routes.values() for c in r})

    header = (["cells"] + [f"MEEP@{n}" for n in ranks] + ["array"]
              + list(routes) + [f"{r} vs best" for r in routes])
    lines = [f"## {args.host}", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    record: List[Dict[str, Any]] = []
    notes: List[str] = []
    for cells in sizes:
        by_rank = meep.get(cells, {})
        best = max(by_rank.values()) if by_rank else None
        best_rank = next((n for n, v in by_rank.items() if v == best), None)
        cols = [f"{cells:,}"] + [f"{by_rank[n]:.1f}" if n in by_rank else "—" for n in ranks]
        array_rates = [rate(cells, r[cells]["array_ms"]) for r in routes.values()
                       if cells in r and r[cells]["array_ms"]]
        array = sum(array_rates) / len(array_rates) if array_rates else None
        cols.append(f"{array:.1f}" if array else "—")
        entry: Dict[str, Any] = {"cells": cells, "meep": by_rank, "meep_best_ranks": best_rank,
                                 "array_mcell_s": array, "routes": {}}
        ratios = []
        for name, table in routes.items():
            row = table.get(cells)
            if not row or not row["fused_ms"]:
                cols.append("—"); ratios.append("—"); continue
            fused = rate(cells, row["fused_ms"])
            citable = row["verdict"] == "TIMED"
            cols.append(f"{fused:.1f}" if citable else f"~~{fused:.1f}~~")
            if best and citable:
                ratios.append(f"**{fused / best:.2f}×**")
            elif best:
                ratios.append(f"~~{fused / best:.2f}×~~")
            else:
                ratios.append("—")
            if not citable:
                notes.append(f"{cells:,} cells, {name}: {row['verdict']} "
                             f"(floors failed: {', '.join(row['floors_failed']) or 'none'}) — "
                             f"struck through, enters no ratio")
            entry["routes"][name] = {"fused_mcell_s": fused, "verdict": row["verdict"],
                                     "floors_failed": row["floors_failed"],
                                     "copies_per_step": row["copies"], "tables": row["tables"],
                                     "monitors": row["monitors"],
                                     "bit_identical": row["bit_identical"],
                                     "vs_best_meep": (fused / best) if best else None}
        cols += ratios
        lines.append("| " + " | ".join(cols) + " |")
        record.append(entry)
    lines.append("")
    lines.append("MEEP@N is stock MEEP 1.33.0 single precision at N MPI ranks. `array` is our "
                 "host array path. Route columns are the fused dispatch route with monitors "
                 "attached; `vs best` is against MEEP's fastest rank count at that size.")
    multi = sorted((c, n, k) for (c, n), k in MEEP_PASSES.items() if k > 1)
    if multi:
        lines.append("")
        lines.append("MEEP values are the MEDIAN over the passes a (cells, ranks) pair carried: "
                     + ", ".join(f"{c:,}@{n} x{k}" for c, n, k in multi) + ".")
    if notes:
        lines += ["", "**Rows that are not citable:**"] + [f"- {n}" for n in notes]
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"host": args.host, "ranks": ranks, "rows": record}, handle, indent=2)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
