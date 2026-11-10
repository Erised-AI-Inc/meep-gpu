#!/usr/bin/env python3
"""Carry route legs (CERT-4) and L3 rows (CERT-5) to a new tree, by plan and closure.

Why. A route leg and an L3 row are each evidence about the plans they dispatched,
executed by the bytes their import record names. When the tree moves, only the runs whose
closure moved, or whose plan the new tree's planner data changes, owe a re-run; every
other run is the same evidence and is carried, with the proof by digest. In particular a
re-cut timing record, whose only effect is on which certified product holds a slot,
re-certifies only the cases and rows whose plan it changes. The rules and states are
``parity/meep_gpu/plan_carry.py``'s; the canonical plan form is ``plan_key.py``'s. This
tool runs on any host: it reads files and steps nothing. The plan passes it compares
against run on the device (``--plan-only`` on the route gates, ``l3_corpus.py
--plan-only``).

    python carry.py route --table triton|cuda|metal --source CAMPAIGN [--source ...]
                          --repo R --device cc86 [--plans DIR] --out CARRY.json
    python carry.py l3    --table triton|cuda|metal --baseline keep|flush
                          [--shipped-default] [--sets A,B] [--primary A] [--image-set A]
                          --source LEG [--source LEG ...] --repo R --device cc86
                          [--plans PLAN_LEG] [--expect-labels FILE] --out CARRY.json
    python carry.py place --carry CARRY.json --into DIR
    python carry.py verify --carry CARRY.json --into DIR --repo R

``route``. Every leg the round's route stage runs for the table (``legs.route_legs`` on
NVIDIA, ``metal_round.ROUTE_EXPECT`` on Metal; preflights are not legs of the record) is
looked for in every ``--source`` campaign directory (``<leg>/gate.json``) and judged on
``--repo``; ``--plans DIR`` holds one plan pass per leg (``DIR/<leg>/plans.json``, the
gate run with the leg's own arguments and ``--plan-only``). A leg in no source is
MISSING and one in two is a DUPLICATE (neither is taken). The Metal campaign is one item
whose band witness is an input of every other leg, so it carries whole or re-runs whole.
``--device`` is required: the device is read from every run and never assumed.
The record names, per leg, its state, its proof or why not, its source and the sha256 of
every file of its directory; then the counts (``carried`` of ``legs``), and the commands
that re-run what is owed, under fresh one-shot names: a plan pass into
``<plans>/<leg>``, a leg into ``<into>/<leg>``.

``l3``. The leg being built is named as ``l3_corpus.py`` names it (``--table``,
``--baseline``, ``--shipped-default``, ``--sets``, ``--primary``, ``--image-set``, with its
defaults); every row of every ``--source`` leg (``rows.jsonl``, the last line of a label
counts, as ``l3_corpus.py`` resumes) is judged on ``--repo``, and a row of other leg facts
(table, mode, shipped default, sets, primary, image set) is not this leg's: its label is
MISSING. A row the leg declared run-to-run variable (its line carries ``attempt`` or the
attribution stamp ``l3_corpus.ATTRIBUTION_KEY``) carries only when its stamp's outcome
holds (``l3_corpus.HOLDING_OUTCOMES``): a finding, an undecided row, or a run with no
stamp, re-runs (:func:`declared_row_problem`, checked again by ``place`` for every row it
writes and by ``verify``). Its one last line is placed, and its stamp carries the row's
attribution, each run's table digest among it; ``l3_corpus.py`` never runs a placed line
again. A row from a leg that declared nothing carries as any row does. ``--plans`` is a plan-pass leg on the new tree (``l3_corpus.py --plan-only``
with the same leg arguments). ``--expect-labels`` (one label per line) names the rows the
new leg plans; a label no source holds is MISSING.

``place``. Writes the carried part into a fresh directory, refusing one that exists: on
``route``, a copy of every carried leg (byte-identical, checked) and ``CARRY.json``, the
record with its plan-pass paths made absolute; the directory must be the campaign name
the route-stamp registry spells now (``meep_gpu/route_stamps.py``), and ``round.py launch
route --resume`` under that stamp then skips every placed leg (released) and runs only the
owed ones into it. On ``l3``, ``rows.jsonl`` holding
each carried row's line with ``carried_from`` (its source, the sha256 of the line and of
its full row file, the state and the proof), the full row files under ``rows/`` and
``CARRY.json``; ``l3_corpus.py --out DIR`` then runs only the rows not carried (it
resumes by label) and counts the carried ones under ``carried``.

``verify``. Judges every carried leg or row again in ``--into`` against ``--repo``: the
same files, the same bytes (on ``l3``, each placed row file by its sha256 and each placed
line, ``carried_from`` aside, by the sha256 of its canonical JSON), the same leg, and
CARRIED again (``recut_driver_dispatch_record.py
--carry`` runs the same check before it cuts). Exit 0 when every carried item holds.

Exit status: 0 when every leg or row is carried (``route``, ``l3``) or every check holds
(``place``, ``verify``); 1 when something re-runs, is owed or does not hold; 2 on usage;
3 when a placement is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
REPO_OF_TOOL = HERE.parents[3]
for path in (str(HERE), str(HARNESS), str(REPO_OF_TOOL)):
    if path not in sys.path:
        sys.path.insert(0, path)

import ledger_scope  # noqa: E402
import plan_carry  # noqa: E402
import plan_key  # noqa: E402

EXIT_CARRIED, EXIT_NOT_ALL, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3
GATES = {"triton": "parity/meep_gpu/gate_dispatch_fused_route.py",
         "cuda": "parity/meep_gpu/gate_dispatch_fused_route.py",
         "metal": "parity/meep_gpu/gate_dispatch_metal_route.py"}
#: Each Metal route leg's gate arguments, as the campaign driver hands them
#: (``parity/meep_gpu/run_metal_dispatch_campaign.sh``, ``leg_arguments``; a test holds the
#: two equal): a plan pass stands for a leg only under the leg's own arguments.
METAL_LEG_ARGUMENTS = {"band_witness": ["--install-policy", "keep", "--band-witness"],
                       "harness_keep": ["--install-policy", "keep"],
                       "harness_flush": ["--install-policy", "flush"],
                       "shipped": []}


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------------------

def route_leg_specs(table: str) -> Dict[str, Optional[List[str]]]:
    """``{leg: gate arguments}``: the legs the route record needs for ``table``."""
    if table == "metal":
        import metal_round  # noqa: PLC0415
        return {leg: list(METAL_LEG_ARGUMENTS[leg]) for leg in metal_round.ROUTE_EXPECT}
    import legs  # noqa: PLC0415
    out: Dict[str, Optional[List[str]]] = {}
    for leg in legs.route_legs(table):
        args = list(leg["args"])
        if "--out" in args and str(args[args.index("--out") + 1]).startswith("@OUT@/"):
            out[leg["id"]] = args
    return out


def rerun_command(table: str, leg: str, args: List[str], out: str,
                  plan_only: bool = False) -> str:
    if table == "metal":
        if plan_only:
            return " ".join([f"python -u {GATES[table]} --out {out}", *args, "--plan-only"])
        return ("metal_round.py route into a fresh campaign directory (the Metal campaign "
                "runs as one item)")
    rest = list(args)
    at = rest.index("--out")
    rest[at + 1] = out
    return f"python -u {GATES[table]} {' '.join(rest)}" + (" --plan-only" if plan_only else "")


def classify_route(table: str, sources: Sequence[Path], repo: Path, device: Optional[str],
                   plans: Optional[Path]) -> Dict[str, Any]:
    specs = route_leg_specs(table)
    live = plan_key.plan_inputs(str(repo))
    cache: Dict[str, Any] = {}
    found: Dict[str, List[Path]] = {}
    for source in sources:
        for leg in specs:
            if (Path(source) / leg / "gate.json").is_file():
                found.setdefault(leg, []).append(Path(source) / leg)
    out: Dict[str, Any] = {}
    for leg in specs:
        where = found.get(leg) or []
        if not where:
            out[leg] = {"state": plan_carry.MISSING, "why": "no source holds this leg"}
            continue
        if len(where) > 1:
            out[leg] = {"state": plan_carry.DUPLICATE, "sources": [str(p) for p in where],
                        "why": "two sources hold this leg; neither is taken"}
            continue
        directory = where[0]
        artifact = json.loads((directory / "gate.json").read_text(encoding="utf-8"))
        plan_pass = None
        pass_dir = (Path(plans) / leg) if plans is not None else None
        if pass_dir is not None:
            plan_pass = plan_carry.read_plan_pass(pass_dir)
            if plan_pass is not None:
                plan_pass["_sha256"] = plan_carry.sha256_file(pass_dir / "plans.json")
        verdict = plan_carry.judge_route_leg(artifact, repo, live, device, plan_pass, cache)
        verdict["source"] = str(directory)
        verdict["files"] = plan_carry.tree_files(directory)
        if verdict["state"] == plan_carry.CARRY_PLAN:
            verdict["plan_pass"] = str(pass_dir.resolve())
        out[leg] = verdict
    if table == "metal" and any(v["state"] not in plan_carry.CARRIED for v in out.values()):
        for leg, verdict in out.items():
            if verdict["state"] in plan_carry.CARRIED:
                out[leg] = {"state": plan_carry.RERUN_WITH_CAMPAIGN,
                            "own_state": verdict["state"], "source": verdict.get("source"),
                            "why": ("the Metal campaign runs as one item and its band "
                                    "witness is an input of every leg's cliff controls, so "
                                    "it carries whole or re-runs whole")}
    carried = sorted(leg for leg, v in out.items() if v["state"] in plan_carry.CARRIED)
    rerun = sorted(leg for leg in out if leg not in carried)
    commands: List[str] = []
    for leg in rerun:
        state = out[leg]["state"]
        lines = [rerun_command(table, leg, specs[leg], f"<into>/{leg}")]
        if state == plan_carry.OWED_PLAN:
            lines.insert(0, rerun_command(table, leg, specs[leg], f"<plans>/{leg}", True))
        commands.extend(line for line in lines if line not in commands)
    return {"schema": plan_carry.SCHEMA, "kind": "route", "table": table, "device": device,
            "repo": str(repo), "sources": [str(s) for s in sources],
            "plans": str(plans) if plans else None, "live_plan_inputs": live,
            "legs": out, "carried": carried, "rerun": rerun, "commands": commands,
            "counts": {"carried": len(carried), "legs": len(out),
                       "by_state": _count(out)}}


def _count(entries: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in entries.values():
        counts[entry["state"]] = counts.get(entry["state"], 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------------------
# L3
# ---------------------------------------------------------------------------------------

def leg_rows(directory: Path) -> Dict[str, Dict[str, Any]]:
    """``{label: {"line": text, "row": parsed}}``, the last line of a label counting."""
    out: Dict[str, Dict[str, Any]] = {}
    path = Path(directory) / "rows.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row.get("label"))] = {"line": line, "row": row}
    return out


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    """sha256 of a row as canonical JSON (sorted keys, ``l3_corpus.Leg.land``'s form)."""
    text = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def declared_row_problem(row: Mapping[str, Any]) -> Optional[str]:
    """Why a row of a leg that declared it run-to-run variable is not evidence of a holding
    row, or None (a row not declared, or one whose stamped attribution holds)."""
    import l3_corpus  # noqa: PLC0415
    if "attempt" not in row and l3_corpus.ATTRIBUTION_KEY not in row:
        return None
    found = row.get(l3_corpus.ATTRIBUTION_KEY)
    if not isinstance(found, Mapping):
        return ("a run of a row declared run-to-run variable that carries no attribution "
                "stamp, so whether it holds cannot be read off its line")
    if found.get("outcome") not in l3_corpus.HOLDING_OUTCOMES:
        return (f"a row declared run-to-run variable whose attribution is "
                f"{found.get('outcome')!r}: {found.get('why')}")
    return None


def l3_target(table: str, baseline: str, shipped_default: bool = False,
              sets: Optional[str] = None, primary: Optional[str] = None,
              image_set: Optional[str] = None) -> Dict[str, Any]:
    """The leg facts of an L3 leg, resolved as ``l3_corpus.py`` resolves its arguments
    (its own defaults for the sets, the primary and the image set)."""
    import l3_corpus  # noqa: PLC0415
    argv = ["--table", table, "--baseline", baseline, "--record", "unused.json",
            "--out", "unused"]
    if shipped_default:
        argv.append("--shipped-default")
    for flag, value in (("--sets", sets), ("--primary", primary), ("--image-set", image_set)):
        if value:
            argv += [flag, value]
    args = l3_corpus.resolve(l3_corpus.parser().parse_args(argv))
    return {"table": args.table, "mode": args.baseline,
            "shipped_default": bool(args.shipped_default), "sets": list(args.set_list),
            "primary": args.primary, "image_set": args.image_set}


def l3_target_flags(target: Mapping[str, Any]) -> str:
    """The ``l3_corpus.py`` arguments that name ``target``."""
    return (f"--table {target.get('table')} --baseline {target.get('mode')}"
            + (" --shipped-default" if target.get("shipped_default") else "")
            + f" --sets {','.join(target.get('sets') or ())} --primary {target.get('primary')}"
            f" --image-set {target.get('image_set')}")


def classify_l3(sources: Sequence[Path], repo: Path, device: Optional[str],
                plans: Optional[Path], expect: Optional[List[str]],
                target: Mapping[str, Any]) -> Dict[str, Any]:
    live = plan_key.plan_inputs(str(repo))
    cache: Dict[str, Any] = {}
    held: Dict[str, List[tuple]] = {}
    for source in sources:
        for label, entry in leg_rows(Path(source)).items():
            held.setdefault(label, []).append((Path(source), entry))
    plan_rows = {label: entry["row"] for label, entry in leg_rows(plans).items()} \
        if plans is not None else {}
    labels = sorted(set(held) | set(expect or ()))
    out: Dict[str, Any] = {}
    for label in labels:
        where = held.get(label) or []
        if not where:
            out[label] = {"state": plan_carry.MISSING, "why": "no source leg holds this row"}
            continue
        if len(where) > 1:
            out[label] = {"state": plan_carry.DUPLICATE,
                          "sources": [str(source) for source, _ in where],
                          "why": "two source legs hold this row; neither is taken"}
            continue
        source, entry = where[0]
        verdict = plan_carry.judge_l3_row(entry["row"], repo, live, device,
                                          plan_rows.get(label), cache)
        declared = declared_row_problem(entry["row"])
        if declared:
            verdict = {"state": plan_carry.RERUN_UNRELEASED, "own_state": verdict["state"],
                       "why": declared}
        differ = plan_carry.l3_leg_differences(entry["row"], target)
        if differ:
            verdict = {"state": plan_carry.MISSING, "own_state": verdict["state"],
                       "other_leg": differ,
                       "why": (f"the only source row is of another leg (row, this leg): "
                               f"{differ}; this leg runs it")}
        verdict["source"] = str(source)
        verdict["line_sha256"] = hashlib.sha256(
            entry["line"].encode("utf-8")).hexdigest()
        verdict["row_sha256"] = canonical_row_sha256(entry["row"])
        full = source / "rows" / f"{label}.json"
        verdict["row_file_sha256"] = plan_carry.sha256_file(full) if full.is_file() else None
        out[label] = verdict
    if expect is not None:
        for label in sorted(set(out) - set(expect)):
            out[label]["outside_the_new_plan"] = True
    carried = sorted(label for label, v in out.items() if v["state"] in plan_carry.CARRIED
                     and not v.get("outside_the_new_plan"))
    rerun = sorted(label for label, v in out.items() if label not in carried
                   and not v.get("outside_the_new_plan"))
    owed = ",".join(l for l in rerun if out[l]["state"] == plan_carry.OWED_PLAN)
    return {"schema": plan_carry.SCHEMA, "kind": "l3", "device": device, "repo": str(repo),
            "target": dict(target), "table": target.get("table"), "mode": target.get("mode"),
            "sources": [str(s) for s in sources], "plans": str(plans) if plans else None,
            "live_plan_inputs": live, "rows": out, "carried": carried, "rerun": rerun,
            "commands": ([f"l3_corpus.py {l3_target_flags(target)} --plan-only --out <plans> "
                          f"--rows {owed} --corpus-manifest <corpus manifest>"] if owed else [])
            + ([f"l3_corpus.py {l3_target_flags(target)} --out <into> --corpus-manifest <corpus "
                f"manifest> (resumes past the {len(carried)} carried rows; runs {len(rerun)})"]
               if rerun else []),
            "counts": {"carried": len(carried), "rows": len(carried) + len(rerun),
                       "by_state": _count({l: v for l, v in out.items()
                                           if not v.get("outside_the_new_plan")})}}


# ---------------------------------------------------------------------------------------
# Place and verify
# ---------------------------------------------------------------------------------------

def campaign_name_refusal(carry: Mapping[str, Any], name: str) -> Optional[str]:
    """Why a route placement may not take ``name``, or None: a directory named like a
    route campaign must be the one the route-stamp registry spells now and not spent."""
    if carry.get("kind") != "route":
        return None
    from meep_gpu import route_stamps  # noqa: PLC0415 - standard library only
    if not name.startswith(tuple(route_stamps.ROUTE_PREFIXES.values())):
        return None
    return route_stamps.campaign_refusal(str(carry.get("table")), name)


def place_route(carry: Mapping[str, Any], into: Path) -> List[str]:
    problems: List[str] = []
    for leg in carry.get("carried") or ():
        entry = carry["legs"][leg]
        source = Path(entry["source"])
        if plan_carry.tree_files(source) != entry["files"]:
            problems.append(f"{leg}: the source {source} no longer holds the judged bytes")
            continue
        shutil.copytree(source, into / leg)
        if plan_carry.tree_files(into / leg) != entry["files"]:
            problems.append(f"{leg}: the copy in {into / leg} differs from its source")
    placed = json.loads(json.dumps(carry))
    placed["placed_into"] = str(into)
    write_json(into / "CARRY.json", placed)
    return problems


def place_l3(carry: Mapping[str, Any], into: Path) -> List[str]:
    problems: List[str] = []
    (into / "rows").mkdir(parents=True)
    lines: List[str] = []
    for label in carry.get("carried") or ():
        entry = carry["rows"][label]
        source = Path(entry["source"])
        rows = leg_rows(source)
        if label not in rows:
            problems.append(f"{label}: the source {source} no longer holds the row")
            continue
        line = rows[label]["line"]
        if hashlib.sha256(line.encode("utf-8")).hexdigest() != entry["line_sha256"]:
            problems.append(f"{label}: the source line moved since it was judged")
            continue
        full = source / "rows" / f"{label}.json"
        if full.is_file():
            if plan_carry.sha256_file(full) != entry.get("row_file_sha256"):
                problems.append(f"{label}: the source row file moved since it was judged")
                continue
            shutil.copyfile(full, into / "rows" / f"{label}.json")
        row = json.loads(line)
        declared = declared_row_problem(row)
        if declared:  # a record that names it carried was not written by classify_l3
            problems.append(f"{label}: not placed: {declared}")
            continue
        row["carried_from"] = {"source": str(source), "line_sha256": entry["line_sha256"],
                               "row_file_sha256": entry.get("row_file_sha256"),
                               "state": entry["state"], "proof": entry.get("proof")}
        lines.append(json.dumps(row, sort_keys=True, default=str))
    (into / "rows.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    placed = json.loads(json.dumps(carry))
    placed["placed_into"] = str(into)
    write_json(into / "CARRY.json", placed)
    return problems


def verify_l3(carry: Mapping[str, Any], into: Path, repo: Path) -> List[str]:
    """Every carried row placed in ``into`` with the bytes it was judged on, of the leg
    the record names, and CARRIED again on ``repo``; [] when every one holds."""
    if not carry.get("device"):
        return ["carry_record_names_no_device: the device is read and never assumed"]
    if not isinstance(carry.get("target"), Mapping):
        return ["carry_record_names_no_leg: an L3 carry record names the leg it builds"]
    problems: List[str] = []
    live = plan_key.plan_inputs(str(repo))
    rows = leg_rows(into)
    cache: Dict[str, Any] = {}
    plan_rows = ({label: entry["row"] for label, entry in leg_rows(Path(carry["plans"])).items()}
                 if carry.get("plans") else {})
    for label in carry.get("carried") or ():
        entry = carry["rows"][label]
        held = rows.get(label)
        if held is None or not held["row"].get("carried_from"):
            problems.append(f"{label}: carried by the record and not placed in {into}")
            continue
        row = dict(held["row"])
        origin = row.pop("carried_from")
        if origin.get("line_sha256") != entry["line_sha256"]:
            problems.append(f"{label}: placed from another line than the one judged")
            continue
        if canonical_row_sha256(row) != entry.get("row_sha256"):
            problems.append(f"{label}: the placed line is not the row that was judged "
                            "(its canonical sha256 differs)")
            continue
        placed = into / "rows" / f"{label}.json"
        want = entry.get("row_file_sha256")
        have = plan_carry.sha256_file(placed) if placed.is_file() else None
        if have != want:
            problems.append(f"{label}: the placed row file {placed} is sha256 {have}, the "
                            f"judged one {want}")
            continue
        differ = plan_carry.l3_leg_differences(row, carry["target"])
        if differ:
            problems.append(f"{label}: the placed row is of another leg: {differ}")
            continue
        again = plan_carry.judge_l3_row(row, repo, live, carry.get("device"),
                                        plan_rows.get(label), cache)
        declared = declared_row_problem(row)
        if declared:
            problems.append(f"{label}: no longer holds: {declared}")
            continue
        if again["state"] not in plan_carry.CARRIED:
            problems.append(f"{label}: no longer holds: {again['state']}: {again.get('why')}")
    return problems


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    route = sub.add_parser("route")
    route.add_argument("--table", required=True, choices=sorted(GATES))
    route.add_argument("--source", action="append", required=True)
    route.add_argument("--repo", required=True)
    route.add_argument("--device", required=True,
                       help="the device class certified, as runs stamp it (cc86, "
                            "applegpu_g13s, ...)")
    route.add_argument("--plans", default=None)
    route.add_argument("--out", required=True)
    l3 = sub.add_parser("l3")
    l3.add_argument("--table", required=True, choices=sorted(GATES))
    l3.add_argument("--baseline", required=True, choices=("keep", "flush"))
    l3.add_argument("--shipped-default", action="store_true")
    l3.add_argument("--sets", default=None)
    l3.add_argument("--primary", default=None)
    l3.add_argument("--image-set", default=None)
    l3.add_argument("--source", action="append", required=True)
    l3.add_argument("--repo", required=True)
    l3.add_argument("--device", required=True)
    l3.add_argument("--plans", default=None)
    l3.add_argument("--expect-labels", default=None)
    l3.add_argument("--out", required=True)
    for name in ("place", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--carry", required=True)
        p.add_argument("--into", required=True)
        if name == "verify":
            p.add_argument("--repo", required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.command in ("route", "l3"):
        repo = Path(args.repo).resolve()
        if not (repo / "meep_gpu").is_dir():
            print(f"--repo {repo} holds no meep_gpu/", file=sys.stderr)
            return EXIT_USAGE
        if args.command == "route":
            record = classify_route(args.table, [Path(s) for s in args.source], repo,
                                    args.device, Path(args.plans) if args.plans else None)
            items = record["legs"]
            denominator = record["counts"]["legs"]
        else:
            expect = None
            if args.expect_labels:
                expect = [line.strip() for line in
                          Path(args.expect_labels).read_text(encoding="utf-8").splitlines()
                          if line.strip() and not line.startswith("#")]
            try:
                target = l3_target(args.table, args.baseline, args.shipped_default,
                                   args.sets, args.primary, args.image_set)
            except Exception as exc:  # noqa: BLE001 - l3_corpus's own refusal, by name
                print(f"the leg named is not one l3_corpus.py runs: {exc}", file=sys.stderr)
                return EXIT_USAGE
            record = classify_l3([Path(s) for s in args.source], repo, args.device,
                                 Path(args.plans) if args.plans else None, expect, target)
            items = {k: v for k, v in record["rows"].items()
                     if not v.get("outside_the_new_plan")}
            denominator = record["counts"]["rows"]
        write_json(Path(args.out), record)
        for name, entry in sorted(items.items()):
            say(f"{name}: {entry['state']}"
                + (f" -- {str(entry.get('why'))[:240]}" if entry.get("why") else ""))
        say(f"CARRIED {record['counts']['carried']} / {denominator} "
            f"({args.command}); by state {record['counts']['by_state']}; record {args.out}")
        for line in record["commands"]:
            say(f"OWED: {line}")
        missing, moved = ledger_scope.states_not_archived(
            ((entry.get("proof") or {}).get("set_aside") or {}, entry.get("compared_whole") or {})
            for entry in items.values())
        say(f"ledger states not archived: {len(missing)} of {moved} moved ledger digests"
            + (f" ({', '.join(missing)}): add one with ledger_scope.py archive"
               if missing else ""))
        return EXIT_CARRIED if record["counts"]["carried"] == denominator else EXIT_NOT_ALL
    carry_path = Path(args.carry)
    carry = json.loads(carry_path.read_text(encoding="utf-8"))
    into = Path(args.into)
    if args.command == "place":
        if into.exists():
            print(f"REFUSING: {into} exists; a placement is one-shot, name a fresh "
                  "directory", file=sys.stderr)
            return EXIT_REFUSED
        refused = campaign_name_refusal(carry, into.name)
        if refused:
            print(f"REFUSING: {refused}", file=sys.stderr)
            return EXIT_REFUSED
        into.mkdir(parents=True)
        problems = (place_route(carry, into) if carry.get("kind") == "route"
                    else place_l3(carry, into))
        verb = "placed"
    else:
        if carry.get("kind") == "route":
            _carried, problems = plan_carry.verify_route_carry(carry, into,
                                                               Path(args.repo).resolve(),
                                                               base=carry_path.parent)
        else:
            problems = verify_l3(carry, into, Path(args.repo).resolve())
        verb = "verified"
    for line in problems:
        say(f"PROBLEM: {line}")
    say(f"{verb} {len(carry.get('carried') or ()) - len(problems)} / "
        f"{len(carry.get('carried') or ())} carried into {into}")
    return EXIT_CARRIED if not problems else EXIT_NOT_ALL


if __name__ == "__main__":
    sys.exit(main())
