#!/usr/bin/env python
"""The identical-case table: the GPU routes against stock MEEP on the same simulation.

WHAT IT READS. Only a timing ladder's own ledger (``ladder_rows.jsonl``) and the files
it names: ``bench_fused_products`` ``rows.jsonl`` for the GPU rows,
``bench_meep_identical_case`` ``rows_r<N>.jsonl`` for MEEP, ``environment.json``,
``binding_decision.json`` and the digests of record under ``digests/``. Every fact the
table states is read from those files: the routes and the tables they asked for, the
cases and sizes, the passes, the rank counts and MEEP configurations, the GPU's name and
compute capability, the host's cores, the binding rule's decision. Nothing is typed.

WHERE THE LEDGER'S PATHS POINT. A ledger records the GPU host's absolute paths. Each
entry the released ladder writes carries ``run_dir``, the run directory those paths are
under, and is mapped onto the pull by it. A ledger written before that field existed
(2026-09-28) is mapped with ``--remote-root <the run directory on the host>``.

THE GATE, applied to EVERY leg, MEEP included. A leg is one column at one (case, size):
a GPU route's fused leg, the array-path control, or MEEP at one rank count in one
configuration. It passes when
  (a) every pass's within-window spread is at most 5 %, and
  (b) the spread of its pass medians is at most 5 %,
both spreads (max - min) / median. A GPU leg must also carry the harness's own verdict
``TIMED`` on every pass, launch what its route expects, be bit-identical to the array
path, carry the deposit-repair route its case declares, and not have run with
``MEEP_GPU_ALLOW_UNCERTIFIED=1`` (such a leg is shown and struck as uncertified); a
MEEP leg must carry its size's geometry digest of record, a stepped flux monitor and
a single-precision build.
Every strike names its reason.

TWO RULES, both reported, because which one is of record is an owner ruling this table
does not make. Per-row: a size's ratios enter only if EVERY leg at that size passes.
Per-ratio: a ratio enters if its two legs pass. MEEP's best is taken over every rank
count and configuration tried, blind to the gate; a best that fails the gate strikes
its ratios and is never replaced by the next one. The reference build's best is
reported beside it.

THE DIGEST OF RECORD per (case, size) is the harness lift's when the ladder's
``digests`` tier took one (``digests/<case>_res<r>.source`` reads ``harness-lift``) or
``--lift-digests`` holds one; otherwise the first MEEP row's, and the table says
"harness-lift digest pending; ratios provisional" for that size. Where a lift digest
exists, every MEEP leg is compared against IT, not only against the digest held.

A SIZE IS STRUCK WHOLE -- every ratio at it, under both rules, each strike named --
when the harness lift's digest differs from the digest of record (``DIFFERENT``) or
its driver disagreed with the MEEP object it lifted, when its GPU, MEEP and lift rows
ran different code (the ledger's ``code_digest``) or were
built by different builder sources, or when the GPU rows' monitor count is not the MEEP
rows' DFT-object count.

Usage:
  build_identical_case_table.py --pull <ladder dir> --out-md TABLE.md --out-json table.json
      [--remote-root <run dir on the host>] [--lift-digests DIR] [--manifest <sha256 list>]
      [--standing label=comparison.json ...] [--tiers "priority rest largest"]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

GATE = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
COMPARISON_BUILDER = os.path.join(HERE, "build_3d_comparison.py")
RESULT_TIERS = ("priority", "rest", "largest")


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def spread(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    median = statistics.median(values)
    return (max(values) - min(values)) / median if median else float("inf")


def jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                raise SystemExit(f"{path}:{number} does not parse: {error}")
    return rows


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


class Pull:
    def __init__(self, root: str, remote_root: Optional[str] = None) -> None:
        self.root = os.path.abspath(root)
        self.remote_root = remote_root.rstrip("/") if remote_root else None

    def local(self, remote: Optional[str], run_dir: Optional[str] = None) -> Optional[str]:
        if not remote:
            return None
        for prefix in (run_dir, self.remote_root):
            if prefix and remote.startswith(prefix.rstrip("/") + "/"):
                return self.root + remote[len(prefix.rstrip("/")):]
        if not os.path.isabs(remote):
            return os.path.join(self.root, remote)
        return remote

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)


def verify_pull(pull: Pull, manifest: Optional[str]) -> Dict[str, Any]:
    if not manifest:
        return {"checked": False}
    bad: List[str] = []
    count = 0
    with open(manifest, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            digest, name = line.rstrip("\n").split("  ", 1)
            count += 1
            path = pull.path(name[2:] if name.startswith("./") else name)
            try:
                with open(path, "rb") as blob:
                    local = hashlib.sha256(blob.read()).hexdigest()
            except OSError:
                local = None
            if local != digest:
                bad.append(name)
    return {"checked": True, "manifest": os.path.abspath(manifest), "files": count,
            "matching": count - len(bad), "mismatched": bad}


def pinned_foreign(box: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """Compute apps on the PINNED device other than the row's own, joined by UUID."""
    if not box:
        return None
    index = str(box.get("CUDA_VISIBLE_DEVICES", "")).split(",")[0].strip()
    uuid = index if index.startswith("GPU-") else None
    for line in str(box.get("gpus") or "").splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 2 and cells[0] == index:
            uuid = cells[1]
    if uuid is None:
        return None
    own = str(box.get("pid"))
    return [line.strip() for line in str(box.get("compute_apps") or "").splitlines()
            if [c.strip() for c in line.split(",")][:1] == [uuid]
            and [c.strip() for c in line.split(",")][1:2] != [own]]


def comparison_builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_3d_comparison", COMPARISON_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------

def gate_leg(values: List[float], within: List[Optional[float]],
             extra: Sequence[str] = ()) -> Dict[str, Any]:
    reasons = list(extra)
    within_ok = True
    for index, value in enumerate(within, 1):
        if value is None or value > GATE:
            within_ok = False
            reasons.append(f"within-window spread "
                           f"{'unread' if value is None else format(value, '.1%')} in pass "
                           f"{index}")
    between = spread(values)
    between_ok = True
    note = None
    if len(values) < 2:
        note = "not measurable (one pass)"
    elif between is None or between > GATE:
        between_ok = False
        reasons.append(f"between-pass spread {between:.1%}")
    return {"passes": values, "median": statistics.median(values) if values else None,
            "between_pass_spread": between if len(values) > 1 else None,
            "between_pass_note": note, "within_window_spread": within,
            "within_window_spread_max": max((w for w in within if w is not None),
                                            default=None),
            "within_ok": within_ok, "between_ok": between_ok, "other_ok": not extra,
            "passes_gate": not reasons and bool(values), "strikes": reasons or (
                [] if values else ["no pass measured"])}


def precedent_pass(leg: Dict[str, Any]) -> bool:
    """The asymmetric rule: GPU legs gated on their windows, MEEP on its passes."""
    if not leg["other_ok"]:
        return False
    return leg["between_ok"] if leg["kind"] == "meep" else leg["within_ok"]


def configuration_label(entry: Dict[str, Any]) -> str:
    """``reference`` for the default MEEP configuration, else the control's name."""
    if entry.get("control"):
        return str(entry["control"])
    configuration = entry.get("configuration") or {}
    parts = []
    if configuration.get("build") not in (None, "reference"):
        parts.append(str(configuration["build"]))
    if int(configuration.get("threads_per_rank") or 1) > 1:
        parts.append(f"t{configuration['threads_per_rank']}")
    if configuration.get("split_chunks_evenly") not in (None, "default"):
        parts.append(f"split={configuration['split_chunks_evenly']}")
    return "-".join(parts) or "reference"


def case_of(entry: Dict[str, Any], default_case: str) -> str:
    return str(entry.get("case") or (entry.get("measured") or {}).get("case")
               or default_case)


def gpu_rows(ledger: List[Dict[str, Any]], pull: Pull, tiers: Sequence[str],
             default_case: str) -> Dict[Tuple, Dict]:
    out: Dict[Tuple, Dict] = {}
    for entry in ledger:
        if entry.get("ledger") != "gpu" or entry.get("tier") not in tiers:
            continue
        measured = entry.get("measured") or {}
        rows_file = pull.local(entry.get("rows_file"), entry.get("run_dir"))
        row = jsonl(rows_file)[-1] if rows_file and os.path.exists(rows_file) else {}
        legs = measured.get("legs") or {}
        cells = measured.get("cells") or entry.get("cells_planned")
        per_step = ((row.get("per_leg") or {}).get("fused") or {}).get(
            "median_seconds_per_step")
        from_row = cells / per_step / 1e6 if cells and per_step else None
        ledger_rate = (legs.get("fused") or {}).get("mcell_steps_per_s")
        plans = (row.get("plans") or {}).get("fused") or {}
        key = (entry["route"], case_of(entry, default_case), int(cells or 0),
               int(entry["pass"]))
        attached = measured.get("monitors_attached")
        host = [row.get("host_before"), row.get("host_after")]
        darwin = entry.get("platform") == "darwin" or entry.get("route") == "metal"
        out[key] = {
            "monitors_mode": measured.get("monitors_mode"),
            "monitor_count": (sum(int(v) for v in attached.values()
                                  if isinstance(v, (int, float)))
                              if isinstance(attached, dict) else None),
            "builder_source_sha256": ((measured.get("timing_case") or row.get("timing_case")
                                       or {}).get("builder_source_sha256")),
            "darwin": darwin,
            "foreign_metal": ([p for h in host if h for p in h.get("foreign_processes") or []]
                              if darwin and any(h is not None for h in host) else None),
            "index": entry.get("index"), "of": entry.get("of"), "cells": cells,
            "res": entry.get("res"), "table_asked": entry.get("table_asked"),
            "verdict": measured.get("verdict"),
            "what_launched": measured.get("what_launched"),
            "expected_launched": entry.get("expected_launched"),
            "bit_identical": measured.get("bit_identical_fused_vs_array"),
            "deposit_repair": entry.get("deposit_repair"),
            "allow_uncertified": bool(entry.get("allow_uncertified")),
            "struck": entry.get("struck"),
            "launch_counters": {slot: (str(v.get("arm")) if str(v.get("arm")).startswith(
                f"{v.get('backend')}:") else f"{v.get('backend')}:{v.get('arm')}")
                for slot, v in (plans.get("launch_counters") or {}).items()},
            "legs": {label: {k: leg.get(k) for k in
                             ("mcell_steps_per_s", "ms_per_step", "spread", "windows",
                              "steps_per_window", "launches_per_step")}
                     for label, leg in legs.items()},
            "lift_seconds": measured.get("lift_seconds"),
            "row_elapsed_s": measured.get("row_elapsed_s"),
            "gate_waited_s": entry.get("gate_waited_s"),
            "code_digest": entry.get("code_digest"),
            "foreign_on_device_before": pinned_foreign(row.get("box_before")),
            "foreign_on_device_after": pinned_foreign(row.get("box_after")),
            "gpu_state_before": entry.get("gpu_state_before"),
            "row_file_agrees": (from_row is not None and ledger_rate is not None
                                and abs(from_row - ledger_rate) <= 1e-9 * ledger_rate),
        }
    return out


def meep_rows(ledger: List[Dict[str, Any]], pull: Pull, tiers: Sequence[str],
              default_case: str) -> Dict[Tuple, Dict]:
    out: Dict[Tuple, Dict] = {}
    for entry in ledger:
        if entry.get("ledger") != "meep" or entry.get("tier") not in tiers:
            continue
        facts = entry.get("measured") or {}
        rows_file = pull.local(entry.get("rows_file"), entry.get("run_dir"))
        rows = ([r for r in jsonl(rows_file) if r.get("row_id") == entry.get("row_id")
                 and r.get("row") == "measurement"]
                if rows_file and os.path.exists(rows_file) else [])
        row = rows[-1] if rows else {}
        warm = row.get("warm_up") or {}
        cells = facts.get("cells") or entry.get("cells_planned")
        key = (case_of(entry, default_case), int(cells or 0), configuration_label(entry),
               int(entry["ranks"]), int(entry["pass"]), entry.get("binding"))
        out[key] = {
            "dft_objects": (facts.get("monitors_attached") or {}).get("dft_objects"),
            "builder_source_sha256": (facts.get("builder") or {}).get(
                "builder_source_sha256"),
            "index": entry.get("index"), "cells": cells, "res": entry.get("res"),
            "tier": entry.get("tier"),
            "mcell_steps_per_s": facts.get("mcell_steps_per_s"),
            "spread": facts.get("spread"), "windows": facts.get("windows"),
            "steps_per_window": facts.get("steps_per_window"),
            "binding_label": facts.get("binding_label") or entry.get("binding"),
            "bound": (facts.get("binding") or {}).get("bound"),
            "cpus_per_rank": (facts.get("binding") or {}).get("cpus_per_rank"),
            "init_seconds": facts.get("init_seconds"),
            "geometry_digest": facts.get("geometry_digest"),
            "digest_of_record": entry.get("geometry_digest_of_record"),
            "monitor_stepped": (facts.get("monitor") or {}).get("stepped"),
            "single_precision": facts.get("single_precision"),
            "meep": facts.get("meep"), "struck": entry.get("struck"),
            "code_digest": entry.get("code_digest"),
            "configuration": facts.get("configuration") or entry.get("configuration"),
            "window_rates": row.get("mcell_steps_per_s_windows"),
            "drift_last_over_first": row.get("drift_last_over_first"),
            "warm_up_steps": warm.get("steps"), "warm_up_steps_asked": warm.get("steps_asked"),
            "warm_up_stopped_by": warm.get("stopped_by"),
            "warm_up_seconds": warm.get("seconds"),
            "row_file_agrees": (row.get("mcell_steps_per_s") == facts.get("mcell_steps_per_s")),
        }
    return out


def device_reading(g: Dict[str, Any]) -> Optional[List[str]]:
    """Foreign processes on the GPU around a row, or ``None`` when nothing was read:
    the pinned NVIDIA device's compute apps (joined by UUID), or on a Mac the bench's
    ``host_*.foreign_processes``."""
    if g.get("darwin"):
        return g.get("foreign_metal")
    before, after = g.get("foreign_on_device_before"), g.get("foreign_on_device_after")
    if before is None and after is None:
        return None
    return list(before or []) + list(after or [])


def digests_of_record(pull: Pull, lift_dir: Optional[str],
                      builder: Any) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """(case, res) -> {digest, source} from ``digests/``; lift files keyed (case, cells)."""
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in sorted(glob.glob(pull.path("digests", "*_res*.sha256"))):
        match = re.match(r"(.+)_res(\d+)\.sha256$", os.path.basename(path))
        if not match:
            continue
        with open(path, "r", encoding="utf-8") as handle:
            digest = handle.read().strip()
        source_path = path[:-len(".sha256")] + ".source"
        source = "first-meep-row (no .source file)"
        if os.path.exists(source_path):
            with open(source_path, "r", encoding="utf-8") as handle:
                source = handle.read().strip()
        out[(match.group(1), int(match.group(2)))] = {"digest": digest, "source": source}
    return out


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def fmt(value: Optional[float], spec: str = ".1f") -> str:
    return "—" if value is None else format(value, spec)


def pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{100 * value:.1f} %"


def environment_facts(environment: Dict[str, Any]) -> Dict[str, Any]:
    lscpu = {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip()
             for line in (environment.get("lscpu") or "").splitlines() if ":" in line}
    topology = environment.get("topology") or {}
    sysctl = environment.get("sysctl") or {}
    gpus = [line for line in (environment.get("nvidia_smi_gpus") or "").splitlines()
            if line.strip()]
    gpu_index = str(environment.get("gpu"))
    # The pinned GPU by index or by UUID (``--gpu auto`` pins the allocation's GPU by
    # its UUID, the second column of the record's nvidia-smi listing).
    gpu_line = next((line for line in gpus
                     if gpu_index in [c.strip() for c in line.split(",")[:2]]),
                    gpus[0] if gpus else None)
    gpu_cells = [c.strip() for c in gpu_line.split(",")] if gpu_line else []
    packages = environment.get("packages") or {}
    builds = environment.get("meep_builds") or {}
    reference = builds.get("reference") or {}
    sockets = topology.get("sockets") or (int(lscpu["Socket(s)"]) if lscpu.get("Socket(s)")
                                          else None)
    physical = topology.get("physical_cores")
    if physical is None and lscpu.get("Socket(s)") and lscpu.get("Core(s) per socket"):
        physical = int(lscpu["Socket(s)"]) * int(lscpu["Core(s) per socket"])
    return {
        "cpu": lscpu.get("Model name") or sysctl.get("machdep.cpu.brand_string")
        or topology.get("cpu"),
        "logical_cpus": topology.get("logical_cpus") or lscpu.get("CPU(s)")
        or sysctl.get("hw.logicalcpu"),
        "physical_cores": physical, "sockets": sockets,
        "threads_per_core": topology.get("threads_per_core") or lscpu.get("Thread(s) per core"),
        "gpu_index": gpu_index, "gpu_name": gpu_cells[2] if len(gpu_cells) > 2 else None,
        "compute_capability": gpu_cells[7] if len(gpu_cells) > 7 else None,
        "packages": packages,
        "meep_configure_line": (reference.get("configure_line")
                                or environment.get("meep_configure_line")),
        "meep_cxxflags": ((reference.get("makefile_flags") or {}).get("CXXFLAGS")
                          or (environment.get("meep_build_flags") or {}).get("CXXFLAGS")),
        "mpirun": ((environment.get("mpirun_version") or reference.get("launcher_version")
                    or "").splitlines() or [None])[0],
        "builds": {name: {"configure_line": b.get("configure_line"),
                          "cxxflags": (b.get("makefile_flags") or {}).get("CXXFLAGS"),
                          "openmp": b.get("openmp"), "unrecorded": b.get("unrecorded")}
                   for name, b in builds.items()},
        "platform": environment.get("platform"),
        "apple_gpu": ((environment.get("apple_gpu") or {}).get("state")
                      if environment.get("platform") == "darwin" else None),
        "performance_cores": topology.get("performance_cores"),
        "efficiency_cores": topology.get("efficiency_cores"),
        "tree": environment.get("tree"),
        "kmp_duplicate_lib_ok_inherited": environment.get("kmp_duplicate_lib_ok_inherited"),
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    pull = Pull(args.pull, args.remote_root)
    ledger = jsonl(pull.path("ladder_rows.jsonl"))
    tiers = tuple(args.tiers.split()) if args.tiers else RESULT_TIERS
    environment = read_json(pull.path("environment.json")) or {}
    decision = read_json(pull.path("binding_decision.json"))
    builder = comparison_builder()
    record: Dict[str, Any] = {"pull": pull.root, "gate": GATE, "tiers": list(tiers),
                              "pull_verification": verify_pull(pull, args.manifest)}
    planned_cases = [c for c in (((environment.get("params") or {}).get("CASES") or {})
                                 .get("value") or "").replace(",", " ").split() if c]
    default_case = planned_cases[0] if planned_cases else "pml_3d"
    gpus = gpu_rows(ledger, pull, tiers, default_case)
    meeps = meep_rows(ledger, pull, tiers, default_case)
    checks = meep_rows(ledger, pull, ("bindcheck",), default_case)
    probes = meep_rows(ledger, pull, ("bindprobe",), default_case)
    controls: Dict[Tuple, Dict] = {}
    for entry in ledger:
        if entry.get("ledger") == "array_control" and entry.get("tier") in tiers:
            controls[(case_of(entry, default_case), int(entry["res"]),
                      int(entry["pass"]))] = entry
    of_record = digests_of_record(pull, args.lift_digests, builder)
    lift = builder.read_digests(args.lift_digests) if args.lift_digests else {}
    lift.update(builder.read_digests(pull.path("digests_lift"))
                if os.path.isdir(pull.path("digests_lift")) else {})
    lift_code: Dict[Tuple[str, int], str] = {}
    for entry in ledger:
        if entry.get("ledger") == "lift" and entry.get("code_digest"):
            lift_code[(case_of(entry, default_case), int(entry.get("res") or 0))] = \
                entry["code_digest"]
    preflight_lift = {}
    for path in glob.glob(pull.path("preflight", "digest", "harness_lift_digest*.json")):
        found = read_json(path) or {}
        preflight_lift[str(found.get("case"))] = found

    routes: Dict[str, str] = {}
    for (route, _case, _cells, _p), row in sorted(gpus.items(),
                                                  key=lambda kv: kv[1].get("index") or 0):
        routes.setdefault(route, str(row.get("table_asked") or route))
    route_label = {route: f"GPU {route} route ({table})" for route, table in routes.items()}
    passes = sorted({p for (_r, _c, _cells, p) in gpus} | {k[4] for k in meeps})
    sizes_keys = sorted({(case, cells) for (_r, case, cells, _p) in gpus}
                        | {(k[0], k[1]) for k in meeps})
    ratio_ranks = [int(v) for v in (args.ratio_ranks or "").replace(",", " ").split() if v]

    # ---- denominators ----------------------------------------------------------------
    by_kind: Dict[str, int] = {}
    for entry in ledger:
        key = f"{entry.get('tier')}/{entry.get('ledger')}"
        by_kind[key] = by_kind.get(key, 0) + 1
    code_digests = sorted({e.get("code_digest") for e in ledger if e.get("code_digest")})
    planned = max((int(e.get("of") or 0) for e in ledger), default=0)
    record["denominators"] = {
        "ledger_entries": len(ledger), "rows_planned": planned,
        "ledger_by_tier_and_kind": by_kind,
        "gpu_rows": len(gpus),
        "gpu_timed": sum(g["verdict"] == "TIMED" for g in gpus.values()),
        "gpu_launched_as_expected": sum(g["what_launched"] == g["expected_launched"]
                                        for g in gpus.values()),
        "gpu_bit_identical": sum(g["bit_identical"] is True for g in gpus.values()),
        "gpu_repair_as_declared": sum((g["deposit_repair"] or {}).get("ok") is True
                                      for g in gpus.values()),
        "gpu_repair_recorded": sum(g["deposit_repair"] is not None for g in gpus.values()),
        "gpu_struck_by_ledger": sum(bool(g["struck"]) for g in gpus.values()),
        "gpu_foreign_on_device_before_or_after": sum(
            bool(device_reading(g)) for g in gpus.values()),
        "gpu_device_read": sum(device_reading(g) is not None for g in gpus.values()),
        "gpu_row_file_agrees_with_ledger": sum(g["row_file_agrees"] for g in gpus.values()),
        "meep_rows": len(meeps),
        "meep_struck_by_ledger": sum(bool(m["struck"]) for m in meeps.values()),
        "meep_monitor_stepped": sum(m["monitor_stepped"] is True for m in meeps.values()),
        "meep_single_precision": sum(m["single_precision"] is True for m in meeps.values()),
        "meep_row_file_agrees_with_ledger": sum(m["row_file_agrees"] for m in meeps.values()),
        "meep_warm_up_stopped_by_seconds_cap": sum(
            m["warm_up_stopped_by"] == "seconds_cap" for m in meeps.values()),
        "array_controls": len(controls),
        "array_controls_struck": sum(bool(c.get("struck")) for c in controls.values()),
        "bindcheck_rows": len(checks), "bindprobe_rows": len(probes),
        "code_digests_on_ledger_rows": code_digests,
        "ledger_rows_carrying_code_digest": sum(bool(e.get("code_digest")) for e in ledger),
        "ledger_gpu_and_meep_rows": sum(e.get("ledger") in ("gpu", "meep") for e in ledger),
    }
    record["environment"] = environment_facts(environment)
    record["binding_decision"] = decision
    record["routes"] = {route: {"table_asked": table, "label": route_label[route]}
                        for route, table in routes.items()}
    record["passes"] = passes

    # ---- per (case, size) -------------------------------------------------------------
    sizes: List[Dict[str, Any]] = []
    for case, cells in sizes_keys:
        legs: Dict[str, Dict[str, Any]] = {}
        info: Dict[str, Any] = {}
        res = next((g["res"] for (r, c, n, p), g in gpus.items() if c == case and n == cells),
                   None) or next((m["res"] for k, m in meeps.items()
                                  if k[0] == case and k[1] == cells), None)
        res = int(res) if res is not None else None
        held = of_record.get((case, res)) if res is not None else None
        lift_digest = lift.get((case, cells))
        digest_status: Dict[str, Any] = {"digest_of_record": (held or {}).get("digest"),
                                         "source": (held or {}).get("source"),
                                         "harness_lift": (lift_digest or {}).get("digest")}
        if held and str(held.get("source", "")).startswith("harness-lift"):
            digest_status["status"] = "harness-lift digest of record"
        elif held and lift_digest:
            digest_status["status"] = ("equal" if lift_digest["digest"] == held["digest"]
                                       else "DIFFERENT")
        else:
            digest_status["status"] = "harness-lift digest pending; ratios provisional"
        # ---- strikes on the whole size -------------------------------------------------
        size_strikes: List[str] = []
        if lift_digest and lift_digest.get("driver_agrees") is not True:
            # A lift whose driver disagreed with the MEEP object it lifted wrote its
            # JSON before it exited non-zero: its digest names no GPU simulation.
            size_strikes.append("the harness lift's driver disagreed with the MEEP object "
                                f"it lifted ({lift_digest.get('file')}); its digest "
                                "settles nothing")
            digest_status["harness_lift_driver_agrees"] = False
        if digest_status["status"] == "DIFFERENT":
            size_strikes.append(f"the harness lift's digest {lift_digest['digest'][:16]} is "
                                f"not the digest of record {held['digest'][:16]} "
                                f"({held.get('source')}): the GPU rows and the MEEP rows "
                                "did not time one simulation")
        at_size_gpu = [g for (r, c, n, p), g in gpus.items() if c == case and n == cells]
        at_size_meep = [m for k, m in meeps.items() if k[0] == case and k[1] == cells]
        codes = {g.get("code_digest") for g in at_size_gpu} | {
            m.get("code_digest") for m in at_size_meep}
        if res is not None and (case, res) in lift_code:
            codes.add(lift_code[(case, res)])
        codes.discard(None)
        if len(codes) > 1:
            size_strikes.append(f"the GPU, MEEP and lift rows at this size ran "
                                f"{len(codes)} different code digests "
                                f"({', '.join(sorted(c[:12] for c in codes))})")
        builders = {g.get("builder_source_sha256") for g in at_size_gpu} | {
            m.get("builder_source_sha256") for m in at_size_meep}
        if lift_digest:
            builders.add(lift_digest.get("builder_source_sha256"))
        builders.discard(None)
        if len(builders) > 1:
            size_strikes.append(f"the rows at this size were built by {len(builders)} "
                                "different builder sources")
        gpu_monitors = {g.get("monitor_count") for g in at_size_gpu} - {None}
        meep_monitors = {m.get("dft_objects") for m in at_size_meep} - {None}
        if gpu_monitors and meep_monitors and gpu_monitors != meep_monitors:
            size_strikes.append(f"the GPU rows carried {sorted(gpu_monitors)} monitor(s) "
                                f"and the MEEP rows {sorted(meep_monitors)} DFT object(s)")
        detached = sorted({str(g.get("monitors_mode")) for g in at_size_gpu
                           if g.get("monitors_mode") not in (None, "attached")})
        if detached:
            size_strikes.append(f"GPU rows timed with monitors {', '.join(detached)}")
        for route, table in routes.items():
            rows = [gpus[k] for k in sorted(gpus) if k[0] == route and k[1] == case
                    and k[2] == cells]
            if not rows:
                continue
            extra = []
            for p, row in enumerate(rows, 1):
                if row["verdict"] != "TIMED":
                    extra.append(f"pass {p} verdict {row['verdict']}")
                if row["expected_launched"] and row["what_launched"] != row["expected_launched"]:
                    extra.append(f"pass {p} launched {row['what_launched']}")
                if row["bit_identical"] is not True:
                    extra.append(f"pass {p} not bit-identical")
                if row["deposit_repair"] is not None and not row["deposit_repair"].get("ok"):
                    extra.append(f"pass {p} deposit-repair route not as declared")
                if row["allow_uncertified"]:
                    extra.append(f"pass {p} ran with MEEP_GPU_ALLOW_UNCERTIFIED=1: "
                                 "uncertified, enters no certified figure")
                if row["struck"]:
                    extra.append(f"pass {p} struck by the ledger: {row['struck']}")
            fused = gate_leg([r["legs"].get("fused", {}).get("mcell_steps_per_s") for r in rows
                              if r["legs"].get("fused", {}).get("mcell_steps_per_s")],
                             [r["legs"].get("fused", {}).get("spread") for r in rows], extra)
            fused.update({"label": route_label[route], "kind": "gpu", "route": route,
                          "launched": sorted({str(r["what_launched"]) for r in rows}),
                          "arms": rows[0]["launch_counters"],
                          "lift_seconds": [r["lift_seconds"] for r in rows],
                          "row_elapsed_s": [r["row_elapsed_s"] for r in rows],
                          "gate_waited_s": [r["gate_waited_s"] for r in rows],
                          "launches_per_step": [r["legs"].get("fused", {}).get(
                              "launches_per_step") for r in rows]})
            legs[table] = fused
            info[f"{table}_singles"] = gate_leg(
                [r["legs"].get("unfused", {}).get("mcell_steps_per_s") for r in rows
                 if r["legs"].get("unfused", {}).get("mcell_steps_per_s")],
                [r["legs"].get("unfused", {}).get("spread") for r in rows])
        per_pass, within, sources = [], [], []
        for p in passes:
            control = controls.get((case, res, p)) if res is not None else None
            if control is None:
                continue
            found = [s for s in control.get("sources") or [] if s.get("mcell_steps_per_s")]
            per_pass.append(control.get("mcell_steps_per_s_mean"))
            within.append(max((s.get("spread") or 0.0) for s in found) if found else None)
            sources.append({"rates": [s["mcell_steps_per_s"] for s in found],
                            "spreads": [s.get("spread") for s in found],
                            "kernels_vetoed": [s.get("kernels_vetoed") for s in found],
                            "struck": control.get("struck")})
        if sources:
            extra = [f"pass {p} struck: {s['struck']}" for p, s in enumerate(sources, 1)
                     if s["struck"]]
            extra += [f"pass {p}: an array leg launched a kernel"
                      for p, s in enumerate(sources, 1) if not all(s["kernels_vetoed"])]
            array = gate_leg([v for v in per_pass if v is not None], within, extra)
            pairs = [s["rates"] for s in sources if len(s["rates"]) > 1]
            array.update({"label": "array-path control", "kind": "array", "sources": sources,
                          "two_rows_disagree_max": max(
                              (max(r) / min(r) - 1 for r in pairs), default=None)})
            legs["array"] = array
        configurations = sorted({k[2] for k in meeps if k[0] == case and k[1] == cells},
                                key=lambda c: (c != "reference", c))
        meep_legs: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for configuration in configurations:
            for ranks in sorted({k[3] for k in meeps if k[0] == case and k[1] == cells
                                 and k[2] == configuration}):
                rows = sorted((m for k, m in meeps.items() if k[0] == case and k[1] == cells
                               and k[2] == configuration and k[3] == ranks),
                              key=lambda m: m["index"] or 0)
                extra = []
                for p, m in enumerate(rows, 1):
                    if m["struck"]:
                        extra.append(f"pass {p} struck by the ledger: {m['struck']}")
                    if held and m["geometry_digest"] != held["digest"]:
                        extra.append(f"pass {p} geometry digest is not the digest of record")
                    if lift_digest and m["geometry_digest"] != lift_digest["digest"]:
                        extra.append(f"pass {p} geometry digest is not the harness lift's")
                leg = gate_leg([m["mcell_steps_per_s"] for m in rows
                                if m["mcell_steps_per_s"] is not None],
                               [m["spread"] for m in rows], extra)
                name = (f"MEEP@{ranks}" if configuration == "reference"
                        else f"MEEP@{ranks} [{configuration}]")
                leg.update({"label": name, "kind": "meep", "ranks": ranks,
                            "configuration": configuration,
                            "binding": sorted({str(m["binding_label"]) for m in rows}),
                            "bound": sorted({str(m["bound"]) for m in rows}),
                            "windows": [m["windows"] for m in rows],
                            "drift_last_over_first": [m["drift_last_over_first"] for m in rows],
                            "warm_up_steps": [m["warm_up_steps"] for m in rows],
                            "init_seconds": [m["init_seconds"] for m in rows],
                            "max_window_rate": max((max(m["window_rates"] or [0])
                                                    for m in rows), default=None)})
                meep_legs[(configuration, ranks)] = leg
                legs[("meep@" + str(ranks)) if configuration == "reference"
                     else f"meep@{ranks}[{configuration}]"] = leg
        measured = {k: v for k, v in meep_legs.items() if v["median"] is not None}
        if not measured:
            sizes.append({"case": case, "res": res, "cells": cells, "legs": legs,
                          "info": info, "digest": digest_status, "meep_best": None,
                          "size_strikes": size_strikes,
                          "ratios": [], "row_passes_every_leg": False,
                          "row_strikes": ["no MEEP leg measured"],
                          "row_passes_precedent_rule": False,
                          "legs_passing_precedent_rule": {}})
            continue
        best_key = max(measured, key=lambda k: measured[k]["median"])
        reference = {k: v for k, v in measured.items() if k[0] == "reference"}
        reference_key = max(reference, key=lambda k: reference[k]["median"]) \
            if reference else None
        ordered = sorted(measured, key=lambda k: -measured[k]["median"])
        runner = ordered[1] if len(ordered) > 1 else None
        best = measured[best_key]
        best_window = max((l["max_window_rate"] or 0) for l in measured.values())
        row_passes = all(l["passes_gate"] for l in legs.values())
        row_passes_precedent = all(precedent_pass(l) for l in legs.values())
        failing = [f"{l['label']}: {'; '.join(l['strikes'])}" for l in legs.values()
                   if not l["passes_gate"]]
        denominators: List[Tuple[Any, Tuple[str, int]]] = [("best", best_key)]
        if reference_key is not None and reference_key != best_key:
            denominators.append(("reference_best", reference_key))
        for n in ratio_ranks:
            if ("reference", n) in measured:
                denominators.append((n, ("reference", n)))
        ratios = []
        gpu_names = [t for t in routes.values() if t in legs] + (
            ["array"] if "array" in legs else [])
        for name in gpu_names:
            gpu = legs[name]
            if gpu["median"] is None:
                continue
            for label, key in denominators:
                meep = measured[key]
                value = gpu["median"] / meep["median"]
                paired = [g / m for g, m in zip(gpu["passes"], meep["passes"])]
                why = [f"{gpu['label']}: {'; '.join(gpu['strikes'])}"] \
                    if not gpu["passes_gate"] else []
                why += [f"{meep['label']}: {'; '.join(meep['strikes'])}"] \
                    if not meep["passes_gate"] else []
                whole = [f"size: {strike}" for strike in size_strikes]
                ratios.append({
                    "gpu": name, "gpu_label": gpu["label"], "denominator": label,
                    "meep_ranks": key[1], "meep_configuration": key[0], "value": value,
                    "paired_pass_range": [min(paired), max(paired)] if paired else None,
                    "enters_per_row": row_passes and not size_strikes,
                    "enters_per_ratio": not why and not size_strikes,
                    "per_ratio_strikes": why + whole,
                    "enters_per_row_precedent_rule": (row_passes_precedent
                                                      and not size_strikes),
                    "enters_per_ratio_precedent_rule": (precedent_pass(gpu)
                                                        and precedent_pass(meep)
                                                        and not size_strikes),
                    "provisional": digest_status["status"].startswith("harness-lift digest "
                                                                      "pending"),
                    "vs_best_window_bound": (gpu["median"] / best_window
                                             if label == "best" and best_window else None)})
        sizes.append({"case": case, "res": res, "cells": cells, "legs": legs, "info": info,
                      "digest": digest_status, "size_strikes": size_strikes,
                      "meep_best": {"ranks": best_key[1], "configuration": best_key[0],
                                    "median": best["median"],
                                    "reference_ranks": reference_key[1] if reference_key
                                    else None,
                                    "reference_median": (reference[reference_key]["median"]
                                                         if reference_key else None),
                                    "runner_up": list(runner) if runner else None,
                                    "runner_up_median": (measured[runner]["median"]
                                                         if runner else None),
                                    "margin_over_runner_up": (
                                        best["median"] / measured[runner]["median"] - 1
                                        if runner else None),
                                    "tried": [list(k) for k in sorted(measured)],
                                    "fastest_single_window_any_rank": best_window},
                      "row_passes_every_leg": row_passes and not size_strikes,
                      "row_strikes": failing + [f"size: {x}" for x in size_strikes],
                      "row_passes_precedent_rule": row_passes_precedent,
                      "legs_passing_precedent_rule": {n: precedent_pass(l)
                                                      for n, l in legs.items()},
                      "ratios": ratios})
    record["sizes"] = sizes
    record["what_launched"] = {
        route: {"tables": sorted({str(g["what_launched"]) for k, g in gpus.items()
                                  if k[0] == route}),
                "arms": sorted({json.dumps(g["launch_counters"], sort_keys=True)
                                for k, g in gpus.items() if k[0] == route}),
                "fused_launches_per_step": sorted({g["legs"].get("fused", {}).get(
                    "launches_per_step") for k, g in gpus.items() if k[0] == route},
                    key=lambda v: (v is None, v)),
                "rows": sum(1 for k in gpus if k[0] == route)}
        for route in routes}
    record["preflight_lift_digest"] = {
        case: {"compared": [{"cells": c.get("cells"), "digest_equal": c.get("digest_equal"),
                             "epsilon_exact_equal": c.get("epsilon_exact_equal")}
                            for c in found.get("compared_with") or []],
               "agreement": found.get("agreement"),
               "cells": ((found.get("geometry") or {}).get("facts") or {}).get("cells")}
        for case, found in preflight_lift.items()}

    # ---- bindcheck ----------------------------------------------------------------
    bindcheck = []
    for key, m in sorted(checks.items(), key=lambda kv: (kv[0][1], kv[0][3])):
        case, cells, _configuration, ranks = key[0], key[1], key[2], key[3]
        size = next((s for s in sizes if s["case"] == case and s["cells"] == cells), None)
        bound = (size or {}).get("legs", {}).get(f"meep@{ranks}") if size else None
        value = m["mcell_steps_per_s"]
        bindcheck.append({
            "case": case, "cells": cells, "ranks": ranks, "binding": m["binding_label"],
            "bound_flag": m["bound"], "mcell_steps_per_s": value, "windows": m["windows"],
            "within_window_spread": m["spread"],
            "passes_within_window_gate": m["spread"] is not None and m["spread"] <= GATE,
            "chosen_median": (bound or {}).get("median"),
            "chosen_passes": (bound or {}).get("passes"),
            "other_over_chosen_median": (value / bound["median"] - 1
                                         if bound and bound.get("median") else None),
            "above_every_chosen_pass": (value > max(bound["passes"])
                                        if bound and bound.get("passes") else None),
            "meep_best": ((size or {}).get("meep_best") or {}).get("median"),
            "would_change_meep_best": (value > size["meep_best"]["median"]
                                       if size and size.get("meep_best") else None),
            "struck": m["struck"]})
    record["bindcheck"] = bindcheck
    outside = []
    for entry in ledger:
        if entry.get("tier") not in ("preflight", "bindprobe", "digests"):
            continue
        measured_entry = entry.get("measured") or {}
        item: Dict[str, Any] = {"index": entry.get("index"), "tier": entry.get("tier"),
                                "kind": entry.get("ledger"), "res": entry.get("res"),
                                "case": case_of(entry, default_case),
                                "pass": entry.get("pass"), "struck": entry.get("struck"),
                                "cells": measured_entry.get("cells")
                                or entry.get("cells_planned")}
        if entry.get("ledger") == "gpu":
            fused = (measured_entry.get("legs") or {}).get("fused") or {}
            item.update({"route": entry.get("route"), "verdict": measured_entry.get("verdict"),
                         "launched": measured_entry.get("what_launched"),
                         "bit_identical": measured_entry.get("bit_identical_fused_vs_array"),
                         "rate": fused.get("mcell_steps_per_s"),
                         "within_window_spread": fused.get("spread")})
        elif entry.get("ledger") == "meep":
            item.update({"ranks": entry.get("ranks"), "binding": entry.get("binding"),
                         "rate": measured_entry.get("mcell_steps_per_s"),
                         "within_window_spread": measured_entry.get("spread"),
                         "monitor_stepped": (measured_entry.get("monitor") or {}).get(
                             "stepped")})
        outside.append(item)
    record["outside_the_table"] = outside

    # ---- the comparison reader's own view, as a cross-check ----------------------
    cross: Dict[str, Any] = {"builder": COMPARISON_BUILDER}
    agree = disagree = 0
    meep_dir = pull.path("meep")
    if os.path.isdir(meep_dir):
        theirs_meep = builder.read_meep(meep_dir)
        for size in sizes:
            for name, leg in size["legs"].items():
                if leg["kind"] != "meep" or leg["configuration"] != "reference":
                    continue
                theirs = (theirs_meep.get((size["case"], size["cells"]), {})
                          .get(leg["ranks"]) or {}).get("median")
                if theirs is not None and leg["median"] is not None and \
                        abs(theirs - leg["median"]) <= 1e-9 * leg["median"]:
                    agree += 1
                else:
                    disagree += 1
    cross["meep_medians_equal"] = agree
    cross["meep_medians_differ"] = disagree
    agree = disagree = 0
    for route, table in routes.items():
        for p in passes:
            directory = pull.path(f"gpu_{route}", f"pass{p}")
            if not os.path.isdir(directory):
                continue
            read = builder.read_route(directory)
            for size in sizes:
                theirs = builder.rate(size["cells"], (read.get((size["case"], size["cells"]))
                                                      or {}).get("fused_ms"))
                ours = (gpus.get((route, size["case"], size["cells"], p)) or {}).get(
                    "legs", {}).get("fused", {}).get("mcell_steps_per_s")
                if ours is None:
                    continue
                if theirs is not None and abs(theirs - ours) <= 1e-9 * ours:
                    agree += 1
                else:
                    disagree += 1
    cross["gpu_pass_rates_equal"] = agree
    cross["gpu_pass_rates_differ"] = disagree
    record["comparison_reader_cross_check"] = cross

    # ---- standing tables -------------------------------------------------------------
    standing = []
    for pair in args.standing or []:
        label, path = pair.split("=", 1)
        with open(path, "r", encoding="utf-8") as handle:
            old = json.load(handle)
        old_rows = old.get("rows") or [r for rows in (old.get("sections") or {}).values()
                                       for r in rows]
        for size in sizes:
            if not size.get("meep_best"):
                continue
            row = next((r for r in old_rows if r.get("cells") == size["cells"]), None)
            if row is None:
                continue
            old_meep = {int(k): v for k, v in (row.get("meep") or {}).items() if v}
            old_best_ranks = row.get("meep_best_ranks")
            for ratio in size["ratios"]:
                if ratio["denominator"] != "best":
                    continue
                name = ratio["gpu"]
                if name == "array":
                    old_gpu = row.get("array_mcell_s")
                else:
                    old_gpu = ((row.get("routes") or {}).get(name) or {}).get("fused_mcell_s")
                old_d = old_meep.get(int(old_best_ranks)) if old_best_ranks else None
                entry = {"standing": label, "case": size["case"], "cells": size["cells"],
                         "gpu": name, "old_gpu": old_gpu,
                         "new_gpu": size["legs"][name]["median"], "old_meep": old_d,
                         "new_meep": size["meep_best"]["median"],
                         "old_meep_ranks": old_best_ranks,
                         "new_meep_ranks": size["meep_best"]["ranks"]}
                if old_gpu and old_d:
                    entry["old_ratio"] = old_gpu / old_d
                    entry["new_ratio"] = ratio["value"]
                    entry["ratio_change"] = entry["new_ratio"] / entry["old_ratio"] - 1
                    entry["gpu_factor"] = entry["new_gpu"] / old_gpu
                    entry["meep_factor"] = old_d / entry["new_meep"]
                standing.append(entry)
    record["standing_comparison"] = standing
    return record


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def render(record: Dict[str, Any]) -> str:
    d = record["denominators"]
    env = record["environment"]
    sizes = record["sizes"]
    out: List[str] = []
    add = out.append
    cases = sorted({s["case"] for s in sizes})
    passes = record["passes"]
    add("# Identical-case 3-D table: GPU routes against stock MEEP on the same simulation")
    add("")
    add(f"Built by `build_identical_case_table.py` from the ladder at `{record['pull']}`, "
        f"reading its ledger `ladder_rows.jsonl` and the row files it names. Cases: "
        + ", ".join(f"`{c}`" for c in cases) + f". Mcell-steps/s throughout; every value is "
        f"the MEDIAN over {len(passes)} fresh-process pass(es) ({', '.join(map(str, passes))}); "
        "spreads are (max − min) / median. Tiers read: " + ", ".join(record["tiers"]) + ".")
    add("")
    summary = []
    for s in sizes:
        best = [r for r in s["ratios"] if r["denominator"] == "best" and r["gpu"] != "array"]
        if not best:
            continue
        summary.append(f"`{s['case']}` {s['cells']:,} cells: "
                       + ", ".join(f"{r['gpu_label']} {r['value']:.2f}×" for r in best)
                       + f" against MEEP@{s['meep_best']['ranks']}"
                       + (f" [{s['meep_best']['configuration']}]"
                          if s["meep_best"]["configuration"] != "reference" else "")
                       + ("" if not best[0]["provisional"] else " (provisional: "
                          "harness-lift digest pending)"))
    entered = sum(r["enters_per_ratio"] for s in sizes for r in s["ratios"])
    total = sum(len(s["ratios"]) for s in sizes)
    add("**Summary.** Against MEEP's best configuration and rank count on the identical "
        "case: " + "; ".join(summary) + ". Under the per-row rule "
        + (", ".join(f"`{s['case']}` {s['cells']:,}" for s in sizes
                     if s["row_passes_every_leg"]) or "no size")
        + f" enters; under the per-ratio rule {entered} of {total} ratios enter.")
    add("")
    add("## Provenance and denominators")
    add("")
    pv = record["pull_verification"]
    if pv.get("checked"):
        add(f"* **Pull**: {pv['matching']} of {pv['files']} files match the GPU host's "
            "sha256 listing.")
    else:
        add("* **Pull**: not verified against a manifest (pass `--manifest`).")
    if env.get("apple_gpu"):
        apple = env["apple_gpu"]
        add(f"* **Host**: {apple.get('model') or 'an Apple GPU'} "
            f"({apple.get('gpu_core_count') or '?'} GPU cores) beside "
            f"{env.get('logical_cpus')} logical CPUs, {env.get('cpu')} "
            f"({env.get('performance_cores')} performance and "
            f"{env.get('efficiency_cores')} efficiency cores; the rank basis is the "
            f"performance cores). {env.get('mpirun')}.")
    else:
        add(f"* **Host**: GPU {env.get('gpu_index')} {env.get('gpu_name') or 'unrecorded'} "
            f"(compute capability {env.get('compute_capability') or 'not recorded by the ladder'}) "
            f"beside {env.get('logical_cpus')} logical CPUs, {env.get('cpu')} "
            f"({env.get('sockets')} socket(s), {env.get('physical_cores')} physical cores, "
            f"{env.get('threads_per_core')} thread(s) per core). {env.get('mpirun')}.")
    pk = env.get("packages") or {}
    add(f"* **MEEP**: {pk.get('meep')}, `is_single_precision() == "
        f"{pk.get('meep_single_precision')}`, configured `{env.get('meep_configure_line')}`, "
        f"CXXFLAGS `{env.get('meep_cxxflags')}`."
        + "".join(f" Build `{name}`: CXXFLAGS `{b.get('cxxflags')}`, OpenMP "
                  f"{b.get('openmp')}" + (" (build flags unrecorded)" if b.get("unrecorded")
                                          else "") + "."
                  for name, b in (env.get("builds") or {}).items() if name != "reference")
        + f" GPU side: CuPy {pk.get('cupy')}, Triton {pk.get('triton')}, PyTorch "
          f"{pk.get('torch')}, float32. `KMP_DUPLICATE_LIB_OK` inherited: "
          f"{env.get('kmp_duplicate_lib_ok_inherited')!r}.")
    add(f"* **Code**: {d['ledger_rows_carrying_code_digest']} of "
        f"{d['ledger_gpu_and_meep_rows']} GPU and MEEP ledger rows carry a code digest, "
        f"{len(d['code_digests_on_ledger_rows'])} distinct: "
        f"`{', '.join(c[:16] for c in d['code_digests_on_ledger_rows'])}`.")
    add(f"* **Ledger**: {d['ledger_entries']} entries for {d['rows_planned']} planned rows "
        f"(a binding decision writes `binding_decision.json`, not a ledger line). GPU rows "
        f"{d['gpu_timed']} of {d['gpu_rows']} `TIMED`, {d['gpu_launched_as_expected']} of "
        f"{d['gpu_rows']} launched what their route expects, {d['gpu_bit_identical']} of "
        f"{d['gpu_rows']} bit-identical to the array path, {d['gpu_repair_as_declared']} of "
        f"{d['gpu_repair_recorded']} carrying the deposit-repair route their case declares, "
        f"{d['gpu_struck_by_ledger']} of {d['gpu_rows']} struck by the ledger; MEEP rows "
        f"{d['meep_rows'] - d['meep_struck_by_ledger']} of {d['meep_rows']} measured "
        f"unstruck, {d['meep_monitor_stepped']} of {d['meep_rows']} with the flux monitor "
        f"stepped, {d['meep_single_precision']} of {d['meep_rows']} single precision; "
        f"array controls {d['array_controls'] - d['array_controls_struck']} of "
        f"{d['array_controls']} unstruck.")
    cross = record["comparison_reader_cross_check"]
    add(f"* **Row files agree with the ledger**: GPU {d['gpu_row_file_agrees_with_ledger']} "
        f"of {d['gpu_rows']}, MEEP {d['meep_row_file_agrees_with_ledger']} of "
        f"{d['meep_rows']}. **Comparison reader cross-check** "
        f"(`build_3d_comparison.read_meep` / `read_route`): MEEP medians equal on "
        f"{cross['meep_medians_equal']} of {cross['meep_medians_equal'] + cross['meep_medians_differ']} "
        f"reference (case, cells, ranks) legs; per-pass GPU fused rates equal on "
        f"{cross['gpu_pass_rates_equal']} of "
        f"{cross['gpu_pass_rates_equal'] + cross['gpu_pass_rates_differ']}.")
    for case, found in record["preflight_lift_digest"].items():
        compared = found["compared"]
        add(f"* **Preflight digest, `{case}`**: the harness lift's geometry digest equals the "
            f"MEEP bench's on {sum(bool(c['digest_equal']) for c in compared)} of "
            f"{len(compared)} preflight MEEP rows at {found.get('cells') or '?'} cells; "
            f"driver agreement {found.get('agreement')}.")
    for s in sizes:
        add(f"* **Digest of record, `{s['case']}` {s['cells']:,} cells**: "
            f"{s['digest']['status']} (source {s['digest']['source']})."
            + ("" if not s.get("size_strikes") else
               " **Every ratio at this size is struck**: "
               + "; ".join(s["size_strikes"]) + "."))
    not_read = d["gpu_rows"] - d.get("gpu_device_read", d["gpu_rows"])
    add(f"* **Device**: {d['gpu_foreign_on_device_before_or_after']} of "
        f"{d.get('gpu_device_read', d['gpu_rows'])} GPU rows read had a foreign process on "
        "the GPU before or after (NVIDIA: the pinned device's compute apps, joined by "
        "UUID; macOS: other processes driving the Apple GPU)"
        + (f"; {not_read} of {d['gpu_rows']} rows carry no device reading (not read)"
           if not_read else "") + ".")
    for route, wl in record["what_launched"].items():
        arms = "; ".join(", ".join(f"{slot} `{arm}`" for slot, arm in
                                   sorted(json.loads(a).items())) for a in wl["arms"])
        add(f"* **What launched, {record['routes'][route]['label']}** ({wl['rows']} rows): "
            f"{', '.join(f'`{t}`' for t in wl['tables'])}; {len(wl['arms'])} distinct slot "
            f"map(s): {arms}; fused launches/step "
            f"{', '.join(fmt(x, '.2f') for x in wl['fused_launches_per_step'])}.")
    add("")
    for case in cases:
        case_sizes = [s for s in sizes if s["case"] == case]
        add(f"## `{case}`: the table (medians over passes)")
        add("")
        add("A struck leg is shown ~~struck~~ and marked †; the reason is under **The gate**.")
        add("")
        meep_names = sorted({n for s in case_sizes for n, l in s["legs"].items()
                             if l["kind"] == "meep"},
                            key=lambda n: (("[" in n), int(re.search(r"@(\d+)", n).group(1)), n))
        gpu_names = [n for n in ([t for t in (r["table_asked"] for r in
                                              record["routes"].values())] + ["array"])
                     if any(n in s["legs"] for s in case_sizes)]
        head = ["cells"] + [n.replace("meep@", "MEEP@") for n in meep_names] + [
            "MEEP best"] + [("array" if n == "array" else
                             next(r["label"] for r in record["routes"].values()
                                  if r["table_asked"] == n)) for n in gpu_names]
        add("| " + " | ".join(head) + " |")
        add("|" + "|".join("---:" for _ in head) + "|")
        for s in case_sizes:
            def cell(leg: Optional[Dict[str, Any]]) -> str:
                if not leg:
                    return "—"
                return fmt(leg["median"]) if leg["passes_gate"] else \
                    f"~~{fmt(leg['median'])}~~ †"
            best = s["meep_best"]
            cols = [f"{s['cells']:,}"] + [cell(s["legs"].get(n)) for n in meep_names]
            cols.append(f"{fmt(best['median'])} (@{best['ranks']}"
                        + (f" {best['configuration']}" if best["configuration"] != "reference"
                           else "") + ")" if best else "—")
            cols += [cell(s["legs"].get(n)) for n in gpu_names]
            add("| " + " | ".join(cols) + " |")
        add("")
        add("## `" + case + "`: ratios")
        add("")
        add("GPU median over MEEP median. `pass range` pairs pass p of the GPU leg with pass "
            "p of the MEEP leg.")
        add("")
        add("| cells | GPU leg | denominator | ratio | pass range | per-row | per-ratio |")
        add("|---:|---|---|---:|---:|---|---|")
        for s in case_sizes:
            for r in s["ratios"]:
                denominator = (f"MEEP best @{r['meep_ranks']}" if r["denominator"] == "best"
                               else f"reference best @{r['meep_ranks']}"
                               if r["denominator"] == "reference_best"
                               else f"MEEP@{r['meep_ranks']}")
                if r["meep_configuration"] != "reference":
                    denominator += f" [{r['meep_configuration']}]"
                spread_text = (f"{r['paired_pass_range'][0]:.2f}–{r['paired_pass_range'][1]:.2f}"
                               if r["paired_pass_range"] else "—")
                add(f"| {s['cells']:,} | {r['gpu_label']} | {denominator} | {r['value']:.2f}×"
                    + (" (provisional)" if r["provisional"] else "") + f" | {spread_text} | "
                    f"{'enters' if r['enters_per_row'] else 'struck'} | "
                    f"{'enters' if r['enters_per_ratio'] else 'struck'} |")
        add("")
        add("## `" + case + "`: the gate, applied to every leg")
        add("")
        add("| cells | leg | binding | passes | median | between passes | within windows | gate |")
        add("|---:|---|---|---|---:|---:|---|---|")
        for s in case_sizes:
            for name, leg in s["legs"].items():
                binding = ", ".join(leg.get("binding") or []) if leg["kind"] == "meep" else (
                    "+".join(leg.get("launched") or []) if leg["kind"] == "gpu"
                    else "array path, 0 launches/step")
                add(f"| {s['cells']:,} | {leg['label']} | {binding} | "
                    f"{' / '.join(fmt(v) for v in leg['passes'])} | {fmt(leg['median'])} | "
                    f"{pct(leg['between_pass_spread'])} | "
                    f"{' / '.join(pct(v) for v in leg['within_window_spread'])} | "
                    + ("pass" if leg["passes_gate"] else
                       "**STRUCK**: " + "; ".join(leg["strikes"])) + " |")
        add("")
    struck = [(s["case"], s["cells"], l) for s in sizes for l in s["legs"].values()
              if not l["passes_gate"]]
    add(f"**{len(struck)} of {sum(len(s['legs']) for s in sizes)} legs are struck**"
        + (": " + "; ".join(f"`{c}` {n:,} {l['label']} ({'; '.join(l['strikes'])})"
                            for c, n, l in struck) if struck else "") + ".")
    add("")
    struck_precedent = sum(not ok for s in sizes
                           for ok in s["legs_passing_precedent_rule"].values())
    add(f"**The same data under the asymmetric rule** (GPU legs gated on their windows, "
        f"MEEP on its passes), for comparison only: {struck_precedent} legs struck; the "
        "per-ratio rule would then admit "
        f"{sum(r['enters_per_ratio_precedent_rule'] for s in sizes for r in s['ratios'])} of "
        f"{total} ratios.")
    add("")
    for rule, key, what in (("Per-row rule", "enters_per_row",
                             "a size's ratios enter only if every leg at that size passes"),
                            ("Per-ratio rule", "enters_per_ratio",
                             "a ratio enters if its two legs pass")):
        add(f"## {rule}: {what}")
        add("")
        for s in sizes:
            bad = [r for r in s["ratios"] if not r[key]]
            add(f"* `{s['case']}` {s['cells']:,}: {len(s['ratios']) - len(bad)} of "
                f"{len(s['ratios'])} enter"
                + ("" if not bad else "; struck: " + "; ".join(
                    f"{r['gpu_label']} / MEEP@{r['meep_ranks']}" for r in bad)) + ".")
        add(f"* {sum(r[key] for s in sizes for r in s['ratios'])} of {total} ratios enter "
            "under this rule.")
        add("")
    if record["standing_comparison"]:
        add("## Against standing tables")
        add("")
        add("| standing | case | cells | GPU leg | old ratio (ranks) | new ratio (ranks) | change | GPU factor | MEEP factor |")
        add("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for e in record["standing_comparison"]:
            if "old_ratio" not in e:
                continue
            add(f"| {e['standing']} | `{e['case']}` | {e['cells']:,} | {e['gpu']} | "
                f"{e['old_ratio']:.2f}× (@{e['old_meep_ranks']}) | {e['new_ratio']:.2f}× "
                f"(@{e['new_meep_ranks']}) | {100 * e['ratio_change']:+.1f} % | "
                f"{e['gpu_factor']:.3f} | {e['meep_factor']:.3f} |")
        add("")
        add("Each move is split into `new / old = GPU factor × MEEP factor`. A standing table "
            "whose MEEP rows timed a different case moves by more than either side's speed.")
        add("")
    if record["binding_decision"] or record["bindcheck"]:
        add("## Binding: the probe, and the binding not chosen")
        add("")
        bd = record["binding_decision"] or {}
        add(f"The probe chose `{bd.get('chosen')}` ({bd.get('reason')}): best-rank medians "
            + ", ".join(f"{label} {fmt(v.get('best'))}"
                        for label, v in (bd.get("bindings") or {}).items()) + ".")
        add("")
        if record["bindcheck"]:
            add("| case | cells | ranks | binding | rate | within windows | chosen median | other vs chosen | would change MEEP best? |")
            add("|---|---:|---:|---|---:|---:|---:|---:|---|")
            for b in record["bindcheck"]:
                add(f"| `{b['case']}` | {b['cells']:,} | {b['ranks']} | {b['binding']} | "
                    f"{fmt(b['mcell_steps_per_s'])} | {pct(b['within_window_spread'])} | "
                    f"{fmt(b['chosen_median'])} | {pct(b['other_over_chosen_median'])} | "
                    f"{'YES' if b['would_change_meep_best'] else 'no'} |")
            changed = sum(bool(b["would_change_meep_best"]) for b in record["bindcheck"])
            add("")
            add(f"**{changed} of {len(record['bindcheck'])} bindcheck rows would change "
                "MEEP's best.**")
            add("")
    add("## Rows outside the table of record")
    add("")
    add("| ledger row | tier | what | case | cells | pass | Mcell-steps/s | within windows | status |")
    add("|---:|---|---|---|---:|---:|---:|---:|---|")
    for item in record["outside_the_table"]:
        if item["kind"] == "gpu":
            what = f"GPU {item.get('route')} route, launched `{item.get('launched')}`"
        elif item["kind"] == "meep":
            what = f"MEEP@{item.get('ranks')} ({item.get('binding')})"
        else:
            what = str(item["kind"])
        status = f"**struck**: {item['struck']}" if item["struck"] and item["kind"] in (
            "gpu", "meep") else (str(item["struck"]) if item["struck"] else "measured")
        add(f"| {item['index']} | {item['tier']} | {what} | `{item['case']}` | "
            f"{fmt(item.get('cells'), ',') if item.get('cells') else '—'} | {item['pass']} | "
            f"{fmt(item.get('rate'))} | {pct(item.get('within_window_spread'))} | {status} |")
    add("")
    add("## One-off lift seconds per GPU row")
    add("")
    add("| case | cells | route | fused lift | singles lift | array lift | row wall (harness) |")
    add("|---|---:|---|---|---|---|---|")
    for s in sizes:
        for name, leg in s["legs"].items():
            if leg["kind"] != "gpu":
                continue
            lifts = [l or {} for l in leg["lift_seconds"]]
            add(f"| `{s['case']}` | {s['cells']:,} | {leg['label']} | "
                + " | ".join(" / ".join(fmt(l.get(k), '.1f') for l in lifts)
                             for k in ("fused", "unfused", "array"))
                + " | " + " / ".join(fmt(v, '.0f') for v in leg["row_elapsed_s"]) + " |")
    add("")
    add("## MEEP warm-up and drift")
    add("")
    add(f"{d['meep_warm_up_stopped_by_seconds_cap']} of {d['meep_rows']} MEEP rows stopped "
        "their warm-up at the wall cap, short of the steps asked. Where the windows still "
        "rise, the median understates MEEP's settled rate, which flatters the GPU; the "
        "fastest single window bounds it:")
    add("")
    for s in sizes:
        if not s.get("meep_best"):
            continue
        rising = [(l["label"], p + 1, dr) for l in s["legs"].values() if l["kind"] == "meep"
                  for p, dr in enumerate(l["drift_last_over_first"]) if dr and dr > 1.03]
        bound = [r for r in s["ratios"] if r["denominator"] == "best" and r["gpu"] != "array"]
        add(f"* `{s['case']}` {s['cells']:,} cells: windows rising by more than 3 %: "
            + (", ".join(f"{lab} pass {p} (×{dr:.3f})" for lab, p, dr in rising) or "none")
            + f". Fastest single MEEP window {fmt(s['meep_best']['fastest_single_window_any_rank'])}"
            + ("; against it " + ", ".join(f"{r['gpu_label']} would read "
                                           f"{r['vs_best_window_bound']:.2f}× instead of "
                                           f"{r['value']:.2f}×" for r in bound)
               if bound else "") + ".")
    add("")
    add("## What this licenses, and what it does not")
    add("")
    gpu_name = env.get("gpu_name") or "the GPU the environment record names"
    add(f"* **Licensed**: on one {gpu_name} against stock MEEP {pk.get('meep')} "
        f"(single precision: {pk.get('meep_single_precision')}) on the same host's "
        f"{env.get('physical_cores')} physical cores, on the case(s) "
        + ", ".join(f"`{c}`" for c in cases) + " with the monitors each case carries on "
        "both sides, at "
        + ", ".join(f"`{s['case']}` {s['cells']:,}" for s in sizes)
        + " cells: the ratios that enter under the rule stated beside them, and only those "
          "whose digest of record is the harness lift's.")
    add("* **Not licensed**: another case, problem size, GPU, CPU or MEEP build than the "
        "ones recorded above; 2-D, cylindrical or dispersive rows; monitor-heavy runs; "
        "per-watt or per-dollar claims; any claim against MEEP@1.")
    add("* **Open**: which gate rule is of record (per-row or per-ratio) is an owner ruling "
        "this table does not make.")
    add("")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pull", required=True)
    parser.add_argument("--remote-root", default=None, dest="remote_root",
                        help="the run directory on the host, for a ledger whose entries "
                             "carry no run_dir")
    parser.add_argument("--lift-digests", default=None, dest="lift_digests",
                        help="harness-lift digest files taken outside the ladder")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--standing", nargs="*", default=[])
    parser.add_argument("--tiers", default=None,
                        help="tiers whose rows form the table; priority rest largest")
    parser.add_argument("--ratio-ranks", default="8,1", dest="ratio_ranks",
                        help="reference rank counts every GPU leg is also divided by")
    parser.add_argument("--out-md", required=True, dest="out_md")
    parser.add_argument("--out-json", required=True, dest="out_json")
    args = parser.parse_args(argv)
    record = build(args)
    text = render(record)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True, default=str)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
