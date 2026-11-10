"""Which cases' selected plans a new set of timing records would change. No timing.

The dispatch preference reads the timing records at every configuration freeze, so a
re-cut record can move which certified kernels hold a case's slots. A route record
is evidence about one plan; this tool says, before anything is re-run, which route-gate
cases a candidate record set moves to another plan and which it leaves alone.

For each route-gate case of a campaign (the DRIVE rows of ``gate_dispatch_fused_route``
-- ``triton``, ``cuda`` with the hand-CUDA table first, or ``cuda_alone``, the CUDA
campaign's leg on a host without Triton -- or of ``gate_dispatch_metal_route``, plus the
size sweep of the NVIDIA timing catalog), it
lifts the case the way the route gate lifts it, with the case's own route and the
dispatch preference left at the package default, takes ONE step (the plan is frozen at
the first step and the dispatch record is complete), and reduces the record to its
canonical plan (``selected_plan.py``). Each case runs in a process of its own.

Commands::

    plan_changes.py report  --campaign C --record FILE [--record FILE ...] --out DIR
    plan_changes.py report  --campaign C --records DIR --out DIR
    plan_changes.py freeze  --campaign C [--records-root DIR] --out FILE.jsonl
    plan_changes.py compare BEFORE.jsonl AFTER.jsonl --out DIR

``report`` freezes every case twice, under the records the tree ships and under the
candidate records laid over them, and writes ``DIR/rows.jsonl`` (one
``meep_gpu.plan_change/1`` row per case) and ``DIR/summary.json``. A candidate is placed
as its architecture's representative, the record a run on that architecture reads: it
replaces the shipped record of its table and architecture, or is added beside them, its
destination read off the record itself. Two candidates of one table and architecture
are refused (``two_candidates_for_one_architecture``). ``freeze`` writes one side only,
so two trees can be compared with ``compare`` (one ``freeze`` per tree). ``--records DIR``
takes a directory laid out as the package lays its records out
(``<table>_kernels/timing_<architecture>.json``, a card record
``timing_<architecture>-<card>.json``).

Runs on a host with the campaign's GPU and MEEP. One flushed line per case on stdout;
rows are appended as each case lands, and a re-run (``freeze`` or ``report`` on the same
output) skips only the cases its output already holds frozen, without an error, under
the very records the run reads now; any other row is frozen again and the new row,
appended after it, is the one read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import selected_plan as sp  # noqa: E402

CAMPAIGNS = ("triton", "cuda", "cuda_alone", "metal")

#: The kernel table each campaign's DRIVE rows belong to.
TABLE_OF_CAMPAIGN = {"triton": "triton", "cuda": "cuda", "cuda_alone": "cuda",
                     "metal": "metal"}
ROW_SCHEMA = "meep_gpu.plan_change/1"
SUMMARY_SCHEMA = "meep_gpu.plan_change_summary/1"
CHILD_TAG = "PLAN_ROW "
NVIDIA_CATALOG = os.path.join(HERE, "rounds", "1.0", "cert8_catalog_nvidia.json")


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] plan_changes: {message}", flush=True)


class OverlayRefused(ValueError):
    """The candidate records cannot be laid over the shipped ones, refused by name."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# The cases of a campaign
# ---------------------------------------------------------------------------

def gate_module(campaign: str) -> Any:
    """The route gate that owns the campaign's cases, set to the campaign's table."""
    if campaign == "metal":
        import gate_dispatch_metal_route as metal  # noqa: PLC0415
        return metal
    import gate_dispatch_fused_route as route  # noqa: PLC0415
    route.BACKEND = TABLE_OF_CAMPAIGN[campaign]
    return route


def drive_rows(campaign: str) -> Dict[str, Dict[str, Any]]:
    module = gate_module(campaign)
    return dict(module.DRIVE_CUDA if TABLE_OF_CAMPAIGN[campaign] == "cuda"
                else module.DRIVE)


def sweep_rows(campaign: str, catalog: str = NVIDIA_CATALOG) -> List[Tuple[str, int]]:
    """The size sweep the NVIDIA timing catalog times for this table: (case, res)."""
    if campaign == "metal" or not os.path.exists(catalog):
        return []
    with open(catalog) as handle:
        tables = ((json.load(handle).get("fused") or {}).get("tables") or {})
    sweep = (tables.get(TABLE_OF_CAMPAIGN[campaign]) or {}).get("sweep") or {}
    return [(str(case), int(res)) for case in sweep.get("cases") or ()
            for res in sweep.get("res") or ()]


def case_rows(campaign: str, cases: Optional[Sequence[str]] = None,
              sweep: bool = True) -> List[Tuple[str, Optional[int]]]:
    """Every (case, res) the campaign freezes: each DRIVE row at its own size, then the sweep."""
    drive = drive_rows(campaign)
    rows: List[Tuple[str, Optional[int]]] = [(case, None) for case in drive]
    if sweep:
        rows += [(case, res) for case, res in sweep_rows(campaign) if case in drive]
    if cases:
        wanted = set(cases)
        rows = [row for row in rows if row[0] in wanted]
    return rows


# ---------------------------------------------------------------------------
# The records a side reads
# ---------------------------------------------------------------------------

def package_root() -> str:
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415
    return os.path.dirname(os.path.abspath(dp.__file__))


def records_at(root: str) -> Dict[str, Dict[str, Any]]:
    """Every timing record under ``root``, keyed as the package keys them, with digests."""
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    out: Dict[str, Dict[str, Any]] = {}
    for path in dp.record_files(root):
        name = os.path.relpath(path, root).replace(os.sep, "/")
        with open(path, "rb") as handle:
            entry: Dict[str, Any] = {"file_sha256": hashlib.sha256(handle.read()).hexdigest(),
                                     "record_sha256": None}
        try:
            entry["record_sha256"] = dp.load_record(path).get(dp.DIGEST_FIELD)
        except dp.RecordRefused as exc:
            entry["refused_because"] = str(exc)
        out[name] = entry
    return out


def _cards(record: Mapping[str, Any]) -> str:
    """The card a record's rows ran on, as its ``host[]`` rows name it."""
    return ", ".join(sorted({str((host or {}).get("device")) for host in record["host"]}))


def overlay(candidates: Sequence[str], directories: Sequence[str], into: str) -> str:
    """A records root: the shipped records with every candidate laid over them.

    A candidate's place is read off the record (its table and the architecture its rows
    ran on), through ``dispatch_preference.record_path``: it is placed as its
    architecture's REPRESENTATIVE, so it replaces exactly the shipped record a run on
    that architecture reads, or lands beside the others. That models the candidate made
    the representative, which is what landing it does. It is not the cutter's
    destination rule, which would place another card's record beside the shipped one,
    where that card alone reads it and every other card of the architecture still reads
    the shipped one. Two candidates of one table and architecture (a staging
    directory holding a bare and a card-named cut, say) are refused before anything is
    laid (``two_candidates_for_one_architecture``): which one represents the
    architecture is not the overlay's to choose.
    """
    from meep_gpu import dispatch_preference as dp  # noqa: PLC0415

    files = list(candidates)
    for directory in directories:
        files += dp.record_files(directory)
    laid: List[Tuple[str, Dict[str, Any], str]] = []
    for path in files:
        record = dp.load_record(path)
        architectures = dp.record_capabilities(record)
        laid.append((path, record,
                     dp.record_path(record["table"], architectures[0], root=into)))
    for target in sorted({target for _path, _record, target in laid}):
        same = [(path, record) for path, record, there in laid if there == target]
        if len(same) > 1:
            first = same[0][1]
            raise OverlayRefused(
                "two_candidates_for_one_architecture",
                f"{len(same)} candidates are {first['table']} records of architecture "
                f"{dp.record_capabilities(first)[0]}: " + "; ".join(
                    f"{path} (card {_cards(record)})" for path, record in same)
                + f". Each would be laid at {os.path.relpath(target, into)} as the "
                "architecture's representative, and a run reads one: pass one")
    if os.path.exists(into):
        shutil.rmtree(into)
    os.makedirs(into)
    shipped = package_root()
    for path in dp.record_files(shipped):
        target = os.path.join(into, os.path.relpath(path, shipped))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(path, target)
    for path, record, target in laid:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(path, target)
        say(f"candidate {path} -> {os.path.relpath(target, into)} "
            f"({record[dp.DIGEST_FIELD][:12]}, {record.get('schema')}, card "
            f"{_cards(record)}), as its architecture's representative")
    return into


# ---------------------------------------------------------------------------
# One case, in a process of its own
# ---------------------------------------------------------------------------

def preference_summary(block: Mapping[str, Any]) -> Dict[str, Any]:
    """The parts of the preference block a reader of a plan change needs."""
    return {
        "mode": block.get("mode"),
        "records": {table: {name: entry.get(name) for name in
                            ("status", "path", "record_sha256", "cost_model")
                            if entry.get(name) is not None}
                    for table, entry in sorted((block.get("records") or {}).items())},
        "rule_by_table": block.get("rule_by_table"),
        "withheld_by_table": block.get("withheld_by_table"),
        "compositions": block.get("compositions"),
        "candidates": [{"label": item.get("label"), "rule": item.get("rule"),
                        "applied": item.get("applied"),
                        "reason": (item.get("verdict") or {}).get("reason"),
                        "why_not_applied": item.get("why_not_applied")}
                       for item in block.get("candidates") or ()],
        "note": block.get("note"),
    }


def freeze_one(campaign: str, case: str, res: Optional[int],
               records_root: Optional[str]) -> Dict[str, Any]:
    """Lift ``case`` as its route gate does, under the package default, and read the plan.

    ``cuda_alone`` withholds Triton the way the route gate's Triton-absent leg does
    (``gate_dispatch_fused_route.withhold_triton``), before anything imports it.
    """
    module = gate_module(campaign)
    if campaign == "cuda_alone":
        module.TRITON_WITHHELD = module.withhold_triton()
    from meep_gpu import fastpath  # noqa: PLC0415

    if records_root is not None:
        fastpath.TIMING_RECORD_ROOT = os.path.abspath(records_root)
    spec = drive_rows(campaign)[case]
    reached_by = spec.get("reached_by", _default_reached_by(module))
    env = module._fuse_env(spec["arms"], reached_by, shape_rule=False)  # noqa: SLF001
    if campaign == "metal":
        report, path = module._one_step_report(case, env, res)  # noqa: SLF001
    else:
        report, path = module._one_step_report(case, env, 0, res)  # noqa: SLF001
    block = report.get("preference") or {}
    grid = (report.get("run_shape") or {}).get("grid_shape") or ()
    cells = 1
    for extent in grid:
        cells *= int(extent)
    plan = sp.selected_plan(report)
    return {
        "campaign": campaign, "case": case, "res": res, "reached_by": reached_by,
        "cells": cells if grid else None, "grid_shape": list(grid),
        "env": env, "step_path": path, "decision": report.get("decision"),
        "refused_because": report.get("refused_because"),
        "plan": plan, "plan_sha256": sp.plan_sha256(plan),
        "withheld": sorted(label for labels in (block.get("withheld_by_table") or {}
                                                ).values() for label in labels),
        "preference": preference_summary(block),
        "preference_block_sha256": hashlib.sha256(json.dumps(
            block, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest(),
        "records_root": "shipped" if records_root is None else "candidate",
        "records": records_at(records_root or package_root()),
    }


def _default_reached_by(module: Any) -> str:
    import inspect  # noqa: PLC0415
    return inspect.signature(module._fuse_env).parameters["reached_by"].default  # noqa: SLF001


def run_child(campaign: str, case: str, res: Optional[int],
              records_root: Optional[str], log: Optional[str]) -> Dict[str, Any]:
    """One case in a fresh interpreter; its row, or the error it ended with."""
    command = [sys.executable, "-u", os.path.abspath(__file__), "one", "--campaign",
               campaign, "--case", case]
    if res is not None:
        command += ["--res", str(res)]
    if records_root is not None:
        command += ["--records-root", records_root]
    started = time.time()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    if log:
        with open(log, "a") as handle:
            handle.write(f"### {campaign} {case} res={res} root={records_root} "
                         f"rc={result.returncode}\n{result.stdout[-20000:]}\n")
    rows = [line[len(CHILD_TAG):] for line in result.stdout.splitlines()
            if line.startswith(CHILD_TAG)]
    if result.returncode == 0 and rows:
        row = json.loads(rows[-1])
    else:
        tail = result.stdout.strip().splitlines()[-3:]
        row = {"campaign": campaign, "case": case, "res": res,
               "error": f"rc {result.returncode}: {' | '.join(tail)[-1500:]}",
               "plan": None, "plan_sha256": None}
    row["elapsed_s"] = round(time.time() - started, 1)
    return row


# ---------------------------------------------------------------------------
# Sides, comparison, report
# ---------------------------------------------------------------------------

def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, Optional[int]]:
    return (str(row.get("campaign")), str(row.get("case")), row.get("res"))


def _read_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def freeze(campaign: str, rows: Sequence[Tuple[str, Optional[int]]],
           records_root: Optional[str], out: str, log: Optional[str]) -> List[Dict[str, Any]]:
    """One side: every row frozen once, appended to ``out`` as it lands; resumable.

    A stored row is reused only when it froze without an error and under the records
    this side reads now (each file's digest, :func:`records_at`), so a second report
    into the same directory with other candidate records, or a re-run after a
    transient failure, freezes those cases again. The row appended last for a case is
    the one read.
    """
    current = records_at(records_root or package_root())
    done = {_row_key(row): row for row in _read_rows(out)
            if not row.get("error") and row.get("records") == current}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for index, (case, res) in enumerate(rows, 1):
        key = (campaign, case, res)
        if key in done:
            continue
        row = run_child(campaign, case, res, records_root, log)
        done[key] = row
        with open(out, "a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        say(f"{campaign} {index}/{len(rows)} {case} res={res} "
            f"{'ERROR ' + row['error'][:200] if row.get('error') else row['plan_sha256'][:12]}"
            f" withheld={row.get('withheld')} ({row['elapsed_s']} s)")
    return [done[(campaign, case, res)] for case, res in rows]


def _side(row: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    row = row or {}
    return {"records": row.get("records"), "plan": row.get("plan"),
            "plan_sha256": row.get("plan_sha256"), "withheld": row.get("withheld"),
            "preference_block_sha256": row.get("preference_block_sha256"),
            "error": row.get("error")}


def compare(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
            ) -> List[Dict[str, Any]]:
    """One ``plan_change/1`` row per (campaign, case, res) either side holds."""
    first = {_row_key(row): row for row in before}
    second = {_row_key(row): row for row in after}
    out = []
    for key in sorted(set(first) | set(second), key=repr):
        a, b = first.get(key), second.get(key)
        errored = (a is None or b is None or a.get("error") or b.get("error"))
        known = b or a or {}
        out.append({"schema": ROW_SCHEMA, "campaign": key[0], "case": key[1],
                    "res": key[2], "cells": known.get("cells"),
                    "reached_by": known.get("reached_by"),
                    "before": _side(a), "after": _side(b),
                    "changed": bool(not errored
                                    and a.get("plan_sha256") != b.get("plan_sha256")),
                    "preference_block_changed": bool(
                        not errored and a.get("preference_block_sha256")
                        != b.get("preference_block_sha256"))})
    return out


def summarize(rows: Sequence[Mapping[str, Any]], campaign: Optional[str]) -> Dict[str, Any]:
    errors = [row for row in rows if row["before"].get("error") or row["after"].get("error")
              or row["before"].get("plan") is None or row["after"].get("plan") is None]
    changed = [row for row in rows if row["changed"]]
    records_before = next((row["before"]["records"] for row in rows
                           if row["before"].get("records")), None)
    records_after = next((row["after"]["records"] for row in rows
                          if row["after"].get("records")), None)
    return {"schema": SUMMARY_SCHEMA, "campaign": campaign, "cases": len(rows),
            "changed": len(changed), "unchanged": len(rows) - len(changed) - len(errors),
            "errors": len(errors),
            "changed_cases": [[row["case"], row["res"]] for row in changed],
            "error_cases": [[row["case"], row["res"]] for row in errors],
            "preference_block_changed": sum(1 for row in rows
                                            if row.get("preference_block_changed")),
            "records_before": records_before, "records_after": records_after}


def write_report(rows: Sequence[Mapping[str, Any]], out_dir: str,
                 campaign: Optional[str]) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(rows, campaign)
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
        handle.write("\n")
    say(f"{campaign}: {summary['cases']} cases, {summary['changed']} changed, "
        f"{summary['unchanged']} unchanged, {summary['errors']} errors -> {out_dir}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("one", help="(internal) freeze one case and print its row")
    one.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    one.add_argument("--case", required=True)
    one.add_argument("--res", type=int, default=None)
    one.add_argument("--records-root", default=None)
    for name in ("freeze", "report"):
        command = sub.add_parser(name)
        command.add_argument("--campaign", choices=CAMPAIGNS, required=True)
        command.add_argument("--cases", default=None, help="comma-separated subset")
        command.add_argument("--no-sweep", action="store_true",
                             help="the DRIVE rows only, not the catalog's size sweep")
        command.add_argument("--out", required=True)
        command.add_argument("--log", default=None, help="every child's output, appended")
        if name == "freeze":
            command.add_argument("--records-root", default=None,
                                 help="read the timing records from here (default: "
                                      "the records the tree ships)")
        else:
            command.add_argument("--record", action="append", default=[],
                                 help="a candidate timing record; repeatable")
            command.add_argument("--records", action="append", default=[],
                                 help="a directory of candidate records, laid out as "
                                      "<table>_kernels/timing_<architecture>.json (a card "
                                      "record, timing_<architecture>-<card>.json, is a "
                                      "candidate too, laid as its architecture's "
                                      "representative)")
    both = sub.add_parser("compare")
    both.add_argument("before")
    both.add_argument("after")
    both.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.command == "one":
        row = freeze_one(args.campaign, args.case, args.res, args.records_root)
        print(CHILD_TAG + json.dumps(row, sort_keys=True, default=str), flush=True)
        return 0
    if args.command == "compare":
        rows = compare(_read_rows(args.before), _read_rows(args.after))
        campaigns = sorted({row["campaign"] for row in rows})
        write_report(rows, args.out, ",".join(campaigns) or None)
        return 0
    rows = case_rows(args.campaign, args.cases.split(",") if args.cases else None,
                     sweep=not args.no_sweep)
    say(f"{args.command} {args.campaign}: {len(rows)} case(s)")
    if args.command == "freeze":
        freeze(args.campaign, rows, args.records_root, args.out, args.log)
        return 0
    if not (args.record or args.records):
        parser.error("report needs at least one --record or --records")
    os.makedirs(args.out, exist_ok=True)
    root = overlay(args.record, args.records, os.path.join(args.out, "candidate_root"))
    before = freeze(args.campaign, rows, None, os.path.join(args.out, "before.jsonl"),
                    args.log)
    after = freeze(args.campaign, rows, root, os.path.join(args.out, "after.jsonl"),
                   args.log)
    summary = write_report(compare(before, after), args.out, args.campaign)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
