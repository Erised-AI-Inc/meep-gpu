#!/usr/bin/env python
"""Timing-validity audit of a ladder pull, GPU legs and MEEP legs. Read-only.

GPU legs: every pass's rate and window spread recomputed from the harness's own windows
(``bench_fused_products`` ``rows.jsonl``), the first window against the rest, kernels
first seen inside a timed window, launches per step, the OTHER devices' utilisation (the
box string's ``others=`` lists every device; the pinned one is left out) and the host's
load while the row ran (joined from ``box_trace.log``).

MEEP legs: every median and spread recomputed from the window rows, the number of
distinct processes (one per pass), the socket of each rank's first CPU, threads per
rank, the busy logical CPUs while the row ran, the warm-up taken and the drift.

What the 2026-09-28 audits typed, this one reads: the run directory from ``--pull``,
the pinned GPU from the ledger, the socket of each CPU from the recorded topology
(``environment.json`` ``topology.package_of_cpu``, or ``lscpu``'s NUMA lists for a
ladder that recorded none), and both box-trace spellings (``busy_cores=`` before
2026-10-02, ``busy_cpus=`` after).

Usage: audit_timing_validity.py --pull <ladder dir> [--tier priority] [--side gpu|meep|both]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import re
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Sequence

BOX = re.compile(r"(\S+) load=([\d.]+) ([\d.]+) ([\d.]+) busy_(?:cores|cpus)=([\d.]+|unreadable)"
                 r"(?:.*?gpu(\d+)=(\d+) MiB, (\d+) %)?(?:.*?others=([\d,]*))?"
                 r".*?account=(\w+) row=\[(\w*) ?(\d*)")


def ts(text: str) -> dt.datetime:
    return dt.datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")


def read_trace(path: str) -> List[Dict[str, Any]]:
    trace = []
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError:
        return trace
    with handle:
        for line in handle:
            match = BOX.match(line)
            if not match:
                continue
            others = [int(v) for v in (match.group(9) or "").strip(",").split(",") if v]
            trace.append({"t": ts(match.group(1)), "load": float(match.group(2)),
                          "busy": None if match.group(5) == "unreadable"
                          else float(match.group(5)),
                          "gpu_mib": int(match.group(7)) if match.group(7) else None,
                          "gpu_util": int(match.group(8)) if match.group(8) else None,
                          "others": others, "row_kind": match.group(11),
                          "pinned": int(match.group(6)) if match.group(6) else None})
    return trace


def socket_map(environment: Dict[str, Any]) -> Dict[int, int]:
    """CPU -> socket, from the recorded topology or lscpu's NUMA lists."""
    packages = (environment.get("topology") or {}).get("package_of_cpu")
    if packages:
        return {int(cpu): int(package) for cpu, package in packages.items()}
    found: Dict[int, int] = {}
    for line in (environment.get("lscpu") or "").splitlines():
        match = re.match(r"NUMA node(\d+) CPU\(s\):\s*(\S+)", line.strip())
        if not match:
            continue
        for part in match.group(2).split(","):
            low, _, high = part.partition("-")
            for cpu in range(int(low), int(high or low) + 1):
                found[cpu] = int(match.group(1))
    return found


def during(trace: List[Dict[str, Any]], start: dt.datetime,
           end: dt.datetime) -> List[Dict[str, Any]]:
    return [x for x in trace if start <= x["t"] <= end]


def audit_gpu(pull: str, ledger: List[Dict[str, Any]], trace: List[Dict[str, Any]],
              tier: str) -> List[str]:
    lines: List[str] = []
    groups: Dict[tuple, Dict[int, Dict[str, Any]]] = collections.defaultdict(dict)
    for entry in ledger:
        if entry.get("ledger") != "gpu" or entry.get("tier") != tier:
            continue
        rows_file = entry.get("rows_file") or ""
        run_dir = entry.get("run_dir")
        if run_dir and rows_file.startswith(run_dir):
            rows_file = pull + rows_file[len(run_dir):]
        else:
            marker = rows_file.find("/gpu_")
            rows_file = pull + rows_file[marker:] if marker >= 0 else rows_file
        if not os.path.exists(rows_file):
            lines.append(f"MISSING {rows_file}")
            continue
        with open(rows_file, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        row = rows[-1]
        cells = 1
        for axis in row.get("grid_shape") or []:
            cells *= int(axis)
        windows: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for window in row.get("windows") or []:
            windows[window["leg"]].append(window)
        legs = {}
        for leg, items in windows.items():
            items = sorted(items, key=lambda w: w["window"])
            per_step = [w["seconds_per_step"] for w in items]
            median = st.median(per_step)
            legs[leg] = {
                "rate": cells / median / 1e6, "spread": (max(per_step) - min(per_step)) / median,
                "n": len(items),
                "w0_over_rest": per_step[0] / st.median(per_step[1:]) if len(per_step) > 1
                else None,
                "new_kernels": sum(w.get("new_kernels_inside_timed_region", 0) for w in items),
                "launches_per_step": sorted({w.get("launches_per_step") for w in items},
                                            key=lambda v: (v is None, v)),
            }
        start = ts(row["utc"])
        end = start + dt.timedelta(seconds=float(row.get("elapsed_s") or 0))
        seen = during(trace, start, end)
        key = (entry.get("route"), entry.get("case") or row.get("case"), cells)
        # A GPU pinned by UUID (``--gpu auto``) is read back as the index the box
        # string keys it by (``gpu<N>=``), so it is still left out of "others".
        pinned = int(entry["gpu"]) if str(entry.get("gpu", "")).isdigit() else next(
            (x["pinned"] for x in seen if x.get("pinned") is not None), None)
        other_utils = [u for x in seen for i, u in enumerate(x["others"]) if i != pinned]
        groups[key][int(entry["pass"])] = {
            "legs": legs, "verdict": row.get("verdict"), "utc": row.get("utc"),
            "other_util_max": max(other_utils, default=None),
            "load_max": max((x["load"] for x in seen), default=None),
            "busy_max": max((x["busy"] for x in seen if x["busy"] is not None), default=None),
            "trace_points": len(seen), "gpu": entry.get("gpu"),
            "tables": (row.get("substitution") or {}).get("tables_dispatched")}
    for key in sorted(groups, key=str):
        passes = groups[key]
        for leg in ("fused", "unfused", "array"):
            rates = [passes[p]["legs"][leg]["rate"] for p in sorted(passes)
                     if leg in passes[p]["legs"]]
            if not rates:
                continue
            median = st.median(rates)
            lines.append(
                f"GPU {key[0]:8} {key[1]} {key[2]:>10,} {leg:7} passes={sorted(passes)} "
                f"rates={[round(r, 1) for r in rates]} med={median:.1f} "
                f"between={(max(rates) - min(rates)) / median * 100:.1f}% within="
                f"{[round(passes[p]['legs'][leg]['spread'] * 100, 1) for p in sorted(passes)]}"
                f" w0/rest={[passes[p]['legs'][leg]['w0_over_rest'] and round(passes[p]['legs'][leg]['w0_over_rest'], 3) for p in sorted(passes)]}"
                f" new_kernels={[passes[p]['legs'][leg]['new_kernels'] for p in sorted(passes)]}"
                f" launches/step={[passes[p]['legs'][leg]['launches_per_step'] for p in sorted(passes)]}")
        for p in sorted(passes):
            x = passes[p]
            lines.append(f"     p{p} {x['utc']} gpu={x['gpu']} verdict={x['verdict']} "
                         f"tables={x['tables']} other_gpu_util_max={x['other_util_max']} "
                         f"load_max={x['load_max']} busy_cpus_max={x['busy_max']} "
                         f"trace_points={x['trace_points']}")
    return lines


def audit_meep(pull: str, environment: Dict[str, Any], trace: List[Dict[str, Any]],
               tier: str) -> List[str]:
    sockets = socket_map(environment)
    lines: List[str] = []
    out: Dict[tuple, Dict[int, Dict[str, Any]]] = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(pull, "meep*", "rows_r*.jsonl"))):
        environments: Dict[str, Any] = {}
        measurements: Dict[str, Any] = {}
        windows: Dict[str, List[Any]] = collections.defaultdict(list)
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("tag") != tier:
                    continue
                if row["row"] == "environment":
                    environments[row["row_id"]] = row
                elif row["row"] == "measurement":
                    measurements[row["row_id"]] = row
                elif row["row"] == "window":
                    windows[row["row_id"]].append(row)
        for row_id, measurement in measurements.items():
            environment_row = environments.get(row_id) or measurement.get("environment") or {}
            items = sorted(windows[row_id], key=lambda w: w["window"])
            per_step = [w["seconds_per_step"] for w in items]
            median = measurement["cells"] / st.median(per_step) / 1e6
            bindings = environment_row.get("rank_bindings") or []
            first_socket = collections.Counter(
                sockets.get(b["affinity"][0], "?") for b in bindings if b.get("affinity"))
            start = ts(environment_row.get("utc") or measurement["utc"])
            seen = during(trace, start, ts(measurement["utc"]))
            configuration = json.dumps(measurement.get("configuration") or {},
                                       sort_keys=True)
            key = (measurement.get("case"), measurement["cells"], measurement["ranks"],
                   os.path.basename(os.path.dirname(path)), configuration)
            observed = measurement.get("threads_observed") or {}
            out[key][measurement["pass"]] = {
                "median": median, "recorded": measurement["mcell_steps_per_s"],
                "spread": (max(per_step) - min(per_step)) / st.median(per_step),
                "windows": len(items),
                "pid": bindings[0]["pid"] if bindings else None,
                "sockets": dict(first_socket),
                "cpus_per_rank": dict(collections.Counter(
                    len(b["affinity"]) for b in bindings if b.get("affinity"))),
                "busy": [x["busy"] for x in seen], "warm": (measurement.get("warm_up") or {})
                .get("steps"), "drift": measurement.get("drift_last_over_first"),
                "digest": str(measurement.get("geometry_digest"))[:12],
                "threads": observed.get("threads_min"),
                "cpu_over_wall": observed.get("cpu_seconds_over_wall_min"),
                "omp": ((environment_row.get("threads") or {}).get("OMP_NUM_THREADS")),
                "nice": environment_row.get("nice")}
    for key in sorted(out, key=str):
        passes = out[key]
        medians = [passes[p]["median"] for p in sorted(passes)]
        median = st.median(medians)
        lines.append(
            f"MEEP {key[0]} {key[1]:>10,} @{key[2]:<3} {key[3]} passes={sorted(passes)} "
            f"processes={len({passes[p]['pid'] for p in passes})} med={median:.1f} "
            f"between={(max(medians) - min(medians)) / median * 100:.1f}% within_max="
            f"{max(passes[p]['spread'] for p in passes) * 100:.1f}% "
            f"recorded_mismatch={max(abs(passes[p]['median'] - passes[p]['recorded']) for p in passes):.2g} "
            f"sockets={[passes[p]['sockets'] for p in sorted(passes)][0]} "
            f"cpus/rank={[passes[p]['cpus_per_rank'] for p in sorted(passes)][0]} "
            f"busy_cpus_max={[max([b for b in passes[p]['busy'] if b is not None] or [None]) if passes[p]['busy'] else None for p in sorted(passes)]} "
            f"warm={[passes[p]['warm'] for p in sorted(passes)]} "
            f"digest={sorted({passes[p]['digest'] for p in passes})} "
            f"omp={sorted({str(passes[p]['omp']) for p in passes})} "
            f"threads={[passes[p]['threads'] for p in sorted(passes)]} "
            f"cpu/wall={[passes[p]['cpu_over_wall'] for p in sorted(passes)]} "
            f"nice={sorted({str(passes[p]['nice']) for p in passes})} config={key[4]}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pull", required=True)
    parser.add_argument("--tier", default="priority")
    parser.add_argument("--side", default="both", choices=("gpu", "meep", "both"))
    args = parser.parse_args(argv)
    pull = os.path.abspath(args.pull)
    trace = read_trace(os.path.join(pull, "box_trace.log"))
    environment: Dict[str, Any] = {}
    try:
        with open(os.path.join(pull, "environment.json"), "r", encoding="utf-8") as handle:
            environment = json.load(handle)
    except (OSError, ValueError):
        pass
    ledger = []
    try:
        with open(os.path.join(pull, "ladder_rows.jsonl"), "r", encoding="utf-8") as handle:
            ledger = [json.loads(line) for line in handle if line.strip()]
    except OSError:
        print(f"no ladder_rows.jsonl under {pull}", file=sys.stderr)
    print(f"box trace: {len(trace)} readings; sockets known for "
          f"{len(socket_map(environment))} CPUs")
    if args.side in ("gpu", "both"):
        for line in audit_gpu(pull, ledger, trace, args.tier):
            print(line)
    if args.side in ("meep", "both"):
        for line in audit_meep(pull, environment, trace, args.tier):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
