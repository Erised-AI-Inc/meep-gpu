#!/usr/bin/env python3
"""The 1.0 NVIDIA certification round on the GPU host: plan, launch and follow its lanes.

Two stages, each a detached process that outlives the shell that started it:

``gates`` (CERT-3)
    1. ``legs.py check`` on the checkout (and, with ``--expect``, the frozen commit and
       a clean tree), with every results root of the stamp required absent;
    2. the gate lanes, at most ``--gpus`` at once, each taking its own GPU lock: the
       CUDA legs split into ``--cuda-split`` campaigns by whole subdirectory, and the
       Triton work under BOTH policies (R13) packed around them;
    3. ``GATES_DONE`` with every lane's N of D, and the bind commands for the host that
       holds the evidence archive (``legs.py bind``), which this stage never runs.

``route`` (CERT-4)
    1. ``legs.py check --route`` (the rebound ledgers are the only files allowed to
       differ from the frozen commit) and ``legs.py flush-coverage`` (N of D arms
       admitted under flush per table, stated; at least one per table, or that
       table's flush leg could not dispatch a case);
    2. the CUDA route lane (blocking preflight, the non-blocking cuda_alone preflight,
       six legs) and the Triton route lane (blocking preflight, four legs, then the
       end-to-end licence: plan-only preflight, ship, null, verdict) side by side on
       two GPUs, every route and licence leg under the route watchdog
       (``ROUTE_TIMEOUT_FACTOR``, ``ROUTE_MIN_TIMEOUT``: a killed leg has used its
       one-shot name);
    3. ``ROUTE_DONE``.

Every lane is ``run_lane.py``: its own lock (released the moment its last step
returns; no lock keeper), its own ``progress.log`` and ``steps.jsonl``, resume by
re-running the same command, and ``<lane>/STOP`` to stop it between steps. This
script waits on child PIDs only.

It writes no ledger, fingerprint or record.

CHUNKED (``--chunk-index I --chunk c<k>``, the gate stage only). The chunk's own stamp
(``chunks.py``: ``<round>_<device>_c<k>``) replaces ``--stamp``, so every root the stage writes
is the chunk's; only the chunk's units run (``chunks.nvidia_plan``: its CUDA campaign
``cuda_regate_<stamp>_g0`` and its Triton steps); its lanes live under ``<home>/<stamp>``, and
``<home>/<stamp>/LAUNCH.lock`` keeps a second launcher of the same chunk out while the stage
runs. ``--gpu-ids`` pins lane k to the k-th card named (each lane still takes that card's lock
under ``--lock-dir``, shared by every chunk so the account cap counts them all); without it a
lane takes any idle card. The stage refuses an index whose chunks could collide
(``chunks.collisions``), a checkout whose route-stamp registry names another round, and an
index whose device is not ``--capability``'s or whose mode disagrees with
``--new-capability``. Any chunk of the index runs whenever its closure is final, side by side
with others; the round's bind is ``chunks.py verify --selection`` then ``legs.py bind --chunks
I --selection F``, which bind each unit from its own newest run that holds.

A NEW CAPABILITY (``--new-capability``, the gate stage of a chunk whose index was partitioned
with ``--new-device``): ``legs.check`` admits a capability the ledgers hold no live record for
only with it, and refuses it on a certified capability and on the route stage, which runs
once the bind has added the new capability's records. The licence of every capability lands
in ``<licence campaign>_cc<capability>`` (``legs.licence_directory``).

THE MEEP CORPUS. The gate stage names the corpus once (``legs.resolve_corpus``:
``--corpus-root``, else the root the reference build record of ``--python``'s environment
names, verified against its list) and passes it to every lane, which exports it to every
child as ``MEEP_GPU_CORPUS_ROOT``.

usage:
  round.py launch gates --repo R --stamp S --capability 8.6 --home H --lock-dir L [...]
  round.py launch gates --repo R --chunk-index I --chunk c1 --capability 8.6 --home H \
      --lock-dir L [--gpu-ids 3] [--new-capability] [...]
  round.py launch route --repo R --stamp S --capability 8.6 --home H --lock-dir L [...]
  round.py status --home H
  round.py run <stage> ...      (what ``launch`` runs, in the foreground)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chunks as C  # noqa: E402
import legs as L  # noqa: E402
import round_status  # noqa: E402


#: THE ROUTE STAGE'S WATCHDOG. A route or licence leg killed by its watchdog leaves its
#: directory unreleased, and route campaign names are one-shot: the leg could then only
#: rerun under a new route stamp, which is a new freeze. The route stage gives every route
#: and licence step max(ROUTE_MIN_TIMEOUT, ROUTE_TIMEOUT_FACTOR x expect_s), the floor and
#: factor the 1.0 pre-certification pass ran these legs with, and expect_s is the leg's
#: MEASURED wall: legs.py prices every route and licence leg at the wall that pass measured
#: on a shared RTX A6000 (ROUTE_CUDA, ROUTE_TRITON, E2E_EXPECT), so every watchdog is at
#: least 12 x the wall it guards. run_lane.py's default, max(1,800 s, 4 x expect_s), sat
#: below Triton shipped (3,180 s) and flush (3,065 s), and 12 x the 0.9.2 prices those legs
#: carried (450 s) was 1.7 x their walls. A longer watchdog only costs wall time on a hung
#: leg, which burns the stamp either way.
ROUTE_TIMEOUT_FACTOR = 12
ROUTE_MIN_TIMEOUT = 3600


def route_timeout(step: Dict[str, Any], factor: int, floor: int) -> int:
    """The watchdog of one route or licence step, in seconds."""
    return max(int(floor), int(factor) * int(step.get("expect_s") or 0))


def say(home: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [round] {message}"
    print(line, flush=True)
    with (home / "round.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def subs_for(args: argparse.Namespace, results: Path, facts: Dict[str, Any]) -> Dict[str, str]:
    """Every placeholder the leg tables spell, for this stamp (a chunk's, when chunked):
    the Triton roots and the H->D hand legs' ``results/triton_<gate>_<stamp>``."""
    return C.stamp_subs(results, args.stamp)


def lane_argv(args: argparse.Namespace, plan: Path, lane_dir: Path, results: Path,
              subs: Dict[str, str], gpu: Optional[int] = None) -> List[str]:
    argv = [sys.executable, "-u", str(HERE / "run_lane.py"), "--plan", str(plan),
            "--lane-dir", str(lane_dir), "--repo", str(args.repo), "--results", str(results),
            "--python", args.python, "--subs", json.dumps(subs), "--lock-dir", args.lock_dir,
            "--max-locks", str(args.max_locks), "--label", f"1.0 round {args.stamp}"]
    if gpu is not None:
        argv += ["--gpu", str(gpu)]       # this card, after taking its lock
    if args.avoid:
        argv += ["--avoid", args.avoid]
    for spec in getattr(args, "share_idle_foreign", None) or ():
        argv += ["--share-idle-foreign", spec]
    if getattr(args, "share_idle_foreign", None):
        argv += ["--share-min-free-mib", str(args.share_min_free_mib)]
    if args.libcuda_stub:
        argv += ["--libcuda-stub", args.libcuda_stub]
    if getattr(args, "corpus_root", ""):
        argv += ["--corpus-root", args.corpus_root]
    for item in args.timeout_floor or ():
        argv += ["--timeout-floor", item]
    return argv


def toolchain_step(args: argparse.Namespace) -> Dict[str, Any]:
    return {"id": "toolchain", "kind": "toolchain", "capability": args.capability,
            "versions": {"triton": args.triton_version, "cupy": args.cupy_version,
                         "meep": args.meep_version},
            "expect_s": 30, "blocking": True}


def filter_steps(steps: List[Dict[str, Any]], only: set, triton_gates: set) -> List[Dict[str, Any]]:
    """The dry-run subset: CUDA legs by id, Triton gates by name (fleet gates and the H->D
    hand legs), other hand legs by id."""
    if not only and not triton_gates:
        return steps
    kept = []
    for step in steps:
        if step["kind"] == "fleet":
            gates = [g for g in step["gates"] if g in triton_gates]
            if gates:
                kept.append(dict(step, gates=gates,
                                 expect_s=sum(L.TRITON_EXPECT.get(g, 300) for g in gates)))
        elif step["id"] in only or (step["kind"] == "hd" and step["gate"] in triton_gates):
            kept.append(step)
    return kept


def run_lanes(home: Path, args: argparse.Namespace, lanes: List[List[Dict[str, Any]]],
              results: Path, subs: Dict[str, str], prefix: str) -> Dict[str, int]:
    """Start one run_lane per non-empty lane, wait on each PID, return exit codes."""
    procs: Dict[str, subprocess.Popen] = {}
    gpu_ids = [int(x) for x in (getattr(args, "gpu_ids", "") or "").split(",") if x.strip()]
    for k, steps in enumerate(lanes):
        if not steps:
            continue
        name = f"{prefix}_{k}"
        lane_dir = home / name
        lane_dir.mkdir(parents=True, exist_ok=True)
        plan = lane_dir / "plan.json"
        plan.write_text(json.dumps(steps, indent=1) + "\n")
        out = (lane_dir / "lane.out").open("a", encoding="utf-8")
        gpu = gpu_ids[k] if k < len(gpu_ids) else None
        procs[name] = subprocess.Popen(lane_argv(args, plan, lane_dir, results, subs, gpu),
                                       stdout=out, stderr=subprocess.STDOUT, cwd=str(HERE))
        say(home, f"lane {name}: pid {procs[name].pid}, {len(steps)} step(s), "
                  f"{sum(int(s.get('expect_s', 0)) for s in steps)} s expected")
        time.sleep(args.stagger)      # lanes take their first idle readings apart
    codes = {}
    for name, proc in procs.items():
        codes[name] = proc.wait()
        say(home, f"lane {name} exit {codes[name]}: "
                  f"{(home / name / 'DONE').read_text().strip() if (home / name / 'DONE').is_file() else 'no DONE marker'}")
    return codes


def stage_gates(args: argparse.Namespace, home: Path) -> int:
    repo = Path(args.repo).resolve()
    results = Path(args.results).resolve() if args.results else repo / "parity" / "meep_gpu" / "results"
    # THE CORPUS IS NAMED ONCE, here, and every lane is handed it: --corpus-root, else the
    # root the reference environment's build record names, verified against its list.
    found = L.resolve_corpus(args.python, args.corpus_root, dict(os.environ))
    root = found["root"] or ""
    args.corpus_root = root
    say(home, f"MEEP corpus: {root or 'none'} (record {found['record']}; "
              f"{'; '.join(found['notes'] + found['problems'])})")
    problems, facts = L.check(repo, args.capability, False, corpus=root, python=args.python,
                              new_capability=args.new_capability)
    chunk = getattr(args, "chunk_manifest", None)
    if chunk is not None:
        fresh = [results / name for name in C.chunk_roots(chunk, "nvidia")]
    else:
        fresh = [results / name for name in L.round_roots(args.stamp, args.cuda_split,
                                                          args.capability, facts)
                 if not name.startswith(("dispatch_",))]
    problems, facts = L.check(repo, args.capability, False, git_expect=args.expect,
                              fresh=[] if args.resume else fresh, corpus=root,
                              python=args.python, new_capability=args.new_capability)
    if chunk is not None:
        problems += chunk_problems(args, repo, results)
    elif args.new_capability:
        problems.append(L.NEW_CAPABILITY_UNCHUNKED)
    (home / "check_gates.json").write_text(json.dumps({"problems": problems, "facts": facts},
                                                      indent=1) + "\n")
    for line in problems:
        say(home, f"CHECK FAIL {line}")
    if problems and not args.dry_run_allow:
        say(home, f"STOP: legs.py check refused ({len(problems)} problem(s)); nothing launched")
        return 3
    if problems:
        say(home, f"DRY RUN: going on past {len(problems)} check problem(s) (--dry-run-allow)")
    if chunk is not None:
        why = C.write_launch_record(results, args.chunk_index_data, chunk, repo)
        if why:
            say(home, f"STOP: {why}; nothing launched")
            return 3
        say(home, f"launch record {C.launch_root(chunk['stamp'])}/LAUNCH.json "
                  f"(chunk {chunk['id']}, {len(chunk['units'])} units)")
    subs = subs_for(args, results, facts)
    only = {s for s in args.only.split(",") if s}
    triton_gates = {s for s in args.triton_gates.split(",") if s}
    # the gate lanes
    if chunk is not None:
        lanes = len([x for x in args.gpu_ids.split(",") if x.strip()]) or \
            int(chunk.get("gpus", 1))
        plan, summary = C.nvidia_plan(repo, results, chunk, lanes)
    else:
        plan, summary = L.plan_lanes(repo, results, args.stamp, gpus=args.gpus,
                                     tables=tuple(t for t in args.tables.split(",") if t),
                                     cuda_split=args.cuda_split)
    plan = [filter_steps(steps, only, triton_gates) for steps in plan]
    plan = [([toolchain_step(args)] + steps) if steps else [] for steps in plan]
    (home / "lanes_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    say(home, f"gates: {sum(1 for p in plan if p)} lane(s) on at most {args.gpus} GPU(s), "
              f"expected wall {summary['wall_s']} s (legs.py's measured walls; the full plan)")
    codes = run_lanes(home, args, plan, results, subs, "gates")
    rc = 0 if not any(codes.values()) else 1
    if chunk is not None:
        bind = [f"# a chunk binds with the round: from parity/meep_gpu on the evidence host, "
                f"with every launched chunk's roots pulled (chunk_<stamp> included), each unit "
                f"from its own newest run that holds",
                f"python rounds/1.0/chunks.py verify --index {args.chunk_index} --repo ../.. "
                f"--results results --selection <selection>",
                f"python rounds/1.0/legs.py bind --chunks {args.chunk_index} "
                f"--selection <selection>"]
    else:
        bind = L.bind_plan(results, args.stamp, args.cuda_split, write=False)
    (home / "BIND_COMMANDS.txt").write_text("\n".join(bind) + "\n")
    (home / "GATES_DONE").write_text(f"{time.strftime('%FT%TZ', time.gmtime())} "
                                     f"exit codes {codes}\n")
    say(home, f"GATES DONE {codes}; bind commands (report mode) in {home / 'BIND_COMMANDS.txt'}")
    return rc


def chunk_problems(args: argparse.Namespace, repo: Path, results: Path) -> List[str]:
    """Why this chunk may not launch: an index whose chunks could collide, a checkout whose
    route-stamp registry names another round, or an index of another device or mode."""
    problems = [f"chunk index: {line}" for line in
                C.collisions(args.chunk_index_data, results=results, repo=repo)]
    problems += L.index_device_problems(args.chunk_index_data, args.capability,
                                        args.new_capability)
    return problems + C.round_problems(repo, args.chunk_index_data)


class LaunchLock:
    """``<home>/LAUNCH.lock``: one launcher per chunk while its stage runs (pid inside; a
    lock whose pid is gone is taken over and said so)."""

    def __init__(self, home: Path, label: str):
        self.dir, self.label, self.held = home / "LAUNCH.lock", label, False

    def take(self) -> Optional[str]:
        try:
            self.dir.mkdir()
        except FileExistsError:
            pid = (self.dir / "pid").read_text().strip() if (self.dir / "pid").is_file() else ""
            if pid.isdigit() and C._alive(int(pid)):
                return f"{self.dir} is held by pid {pid}: this chunk is running"
        (self.dir / "pid").write_text(f"{os.getpid()}\n")
        (self.dir / "label").write_text(f"{self.label}\n")
        self.held = True
        return None

    def release(self) -> None:
        if not self.held:
            return
        for name in ("pid", "label"):
            try:
                (self.dir / name).unlink()
            except OSError:
                pass
        try:
            self.dir.rmdir()
        except OSError:
            pass
        self.held = False


def stage_route(args: argparse.Namespace, home: Path) -> int:
    repo = Path(args.repo).resolve()
    results = Path(args.results).resolve() if args.results else repo / "parity" / "meep_gpu" / "results"
    problems, facts = L.check(repo, args.capability, True, git_expect=args.expect,
                              new_capability=args.new_capability)
    route_roots = []
    for key in ("triton_route_campaign", "cuda_route_campaign"):
        if facts.get(key):
            route_roots += [results / facts[key], results / f"{facts[key]}_preflight"]
    # THE LICENCE DIRECTORY IS THE REGISTRY'S (route_stamps.licence_campaign) FOLLOWED BY
    # THE DEVICE, one per capability (legs.licence_directory), a name the re-cut accepts a
    # licence from (route_stamps.licence_refusal); legs.check reads it into the facts.
    licence_root = results / (facts.get("licence_directory") or L.licence_directory(
        facts.get("licence_campaign") or f"dispatch_end_to_end_{args.stamp}",
        args.capability))
    route_roots.append(licence_root)
    if not args.resume:
        problems += [f"{p} exists; route campaign names are one-shot"
                     for p in route_roots if p.exists()]
    coverage, lacking = L.flush_coverage(repo, args.capability)
    (home / "check_route.json").write_text(json.dumps(
        {"problems": problems, "facts": facts, "flush_coverage": coverage}, indent=1) + "\n")
    for table in ("triton", "cuda"):
        row = coverage[table]
        say(home, f"flush licence on {args.capability}: {table} {row['arms_covered']} of "
                  f"{row['arms']} arms admitted")
    # PER-ARM ADMISSION MAKES A FLUSH LEG MIXED: a case dispatches under flush where
    # every arm it plans has flush evidence, and is refused by name otherwise. So the
    # go/no-go is not completeness (a gate whose artifact states no attained policy
    # cannot carry evidence; RUNBOOK.md section 5) but that each table admits SOME arm,
    # or its flush leg could not dispatch one case and could not release.
    for table in ("triton", "cuda"):
        if not coverage[table]["arms_covered"]:
            problems.append(f"no {table} arm carries flush evidence on "
                            f"{args.capability}: bind the gates' flush runs first")
    if lacking:
        say(home, f"flush evidence absent for {len(lacking)} gate(s); their arms are "
                  f"refused by name on the flush legs: {lacking[:8]}")
    for line in problems:
        say(home, f"CHECK FAIL {line}")
    if problems and not args.dry_run_allow:
        say(home, f"STOP: the route stage's go/no-go refused ({len(problems)}); nothing launched")
        return 3
    subs = subs_for(args, results, facts)
    lanes = []
    for table, key in (("cuda", "cuda_route_campaign"), ("triton", "triton_route_campaign")):
        campaign = results / facts[key]
        steps = [toolchain_step(args)]
        for leg in L.route_legs(table):
            leg["args"] = [a.replace("@OUT@", str(campaign))
                           .replace("@PF@", f"{campaign}_preflight") for a in leg["args"]]
            steps.append(leg)
        if table == "triton":
            e2e = licence_root
            needs = list(L.required_route_legs("triton"))
            steps += [
                {"id": "e2e_preflight", "kind": "e2e", "leg": "preflight", "out": str(e2e / "preflight"),
                 "cases": "pml_2d", "args": ["--enable-from-default", "--plan-only"],
                 "plan_only": True, "expect_s": L.E2E_EXPECT["preflight"], "needs": needs},
                {"id": "e2e_ship", "kind": "e2e", "leg": "ship", "out": str(e2e / "ship"),
                 "cases": "all", "args": ["--enable-from-default"],
                 "expect_s": L.E2E_EXPECT["ship"], "needs": needs + ["e2e_preflight"]},
                {"id": "e2e_null", "kind": "e2e", "leg": "null", "out": str(e2e / "null"),
                 "cases": "all", "args": ["--enable-from-default", "--null-control"],
                 "expect_s": L.E2E_EXPECT["null"], "needs": needs + ["e2e_preflight"]},
                {"id": "e2e_verdict", "kind": "e2e_verdict", "out": str(e2e),
                 "capability": args.capability, "expect_s": 5,
                 "needs": ["e2e_ship", "e2e_null"]}]
        only = {s for s in args.only.split(",") if s}
        if only:
            steps = [s for s in steps if s["id"] in only or s["kind"] == "toolchain"]
        for step in steps:
            if step["kind"] in ("route", "e2e", "e2e_verdict"):
                step["timeout_s"] = route_timeout(step, args.route_timeout_factor,
                                                  args.route_min_timeout)
        lanes.append(steps)
    codes = run_lanes(home, args, lanes, results, subs, "route")
    (home / "ROUTE_DONE").write_text(f"{time.strftime('%FT%TZ', time.gmtime())} "
                                     f"exit codes {codes}\n")
    say(home, f"ROUTE DONE {codes}; recut with recut_driver_dispatch_record.py on the bind "
              f"host (RUNBOOK.md, CERT-4)")
    return 0 if not any(codes.values()) else 1


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("stage", choices=("gates", "route"))
    p.add_argument("--repo", required=True)
    p.add_argument("--results", default="", help="default: <repo>/parity/meep_gpu/results")
    p.add_argument("--stamp", default=None,
                   help="the round's stamp; with --chunk-index, the chunk's stamp is used")
    p.add_argument("--chunk-index", default="",
                   help="gate stage: a chunks.py index (nvidia); run only --chunk, under its "
                        "own stamp, lanes home <home>/<stamp> and launch lock")
    p.add_argument("--chunk", default="", help="the chunk id of --chunk-index (c<k>)")
    p.add_argument("--gpu-ids", default="",
                   help="the cards the lanes run on, lane k on the k-th (each lane still takes "
                        "that card's lock); a chunk plans one lane per card named")
    p.add_argument("--capability", required=True)
    p.add_argument("--new-capability", action="store_true",
                   help="gate stage of a chunk: the capability is one the ledgers hold no "
                        "live record for (an H200 at 9.0), as the index's device_mode says")
    p.add_argument("--home", required=True, help="the lanes' home (logs, markers)")
    p.add_argument("--lock-dir", default=os.environ.get("ROUND_LOCK_DIR", ""))
    p.add_argument("--max-locks", type=int, default=4)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--cuda-split", type=int, default=2)
    p.add_argument("--tables", default="cuda,triton")
    p.add_argument("--avoid", default="")
    p.add_argument("--share-idle-foreign", action="append", default=[],
                   help="gate stage: <uuid>:<pid>[,<pid>...], handed to every lane: a card whose "
                        "only compute apps are these idle PIDs may be taken (run_lane.py, A "
                        "SHARED CARD; owner ruling R28)")
    p.add_argument("--share-min-free-mib", type=int, default=0,
                   help="with --share-idle-foreign: the free memory a shared card must have, MiB")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--libcuda-stub", default=os.environ.get("TRITON_LIBCUDA_PATH", ""))
    p.add_argument("--expect", default=None, help="the frozen commit (full hash)")
    p.add_argument("--triton-version", default="3.1.0")
    p.add_argument("--cupy-version", default="13.5.1")
    p.add_argument("--meep-version", default="1.33.0")
    p.add_argument("--timeout-floor", action="append", default=[])
    p.add_argument("--route-timeout-factor", type=int, default=ROUTE_TIMEOUT_FACTOR,
                   help="route stage: a route or licence leg's watchdog is this many "
                        "times its measured wall (legs.py), and at least "
                        "--route-min-timeout")
    p.add_argument("--route-min-timeout", type=int, default=ROUTE_MIN_TIMEOUT,
                   help="route stage: the shortest watchdog of a route or licence leg (s)")
    p.add_argument("--corpus-root", default="",
                   help="the MEEP corpus the H->D lifts walk, passed to every lane (default: "
                        "the root the reference build record of --python's environment "
                        "names); the gate stage refuses one that does not verify against the "
                        "record's list or holds no examples/tests")
    p.add_argument("--stagger", type=int, default=20)
    p.add_argument("--resume", action="store_true",
                   help="re-run into existing roots: each lane skips what already passed")
    p.add_argument("--only", default="", help="dry run: step ids to keep")
    p.add_argument("--triton-gates", default="",
                   help="dry run: Triton gates to keep (fleet gates and H->D gates)")
    p.add_argument("--dry-run-allow", action="store_true",
                   help="dry run: go on past legs.py check problems (never in a round)")


def apply_chunk(args: argparse.Namespace) -> Optional[str]:
    """Resolve ``--chunk-index``/``--chunk`` onto ``args``: the chunk's manifest and its
    stamp in place of ``--stamp``. Returns why the arguments are refused, or None: a
    capability not spelled major.minor among them, by name, before any stage reads it."""
    args.chunk_manifest = None
    try:
        L.capability_device(args.capability)
    except ValueError as refused:
        return f"--capability is refused: {refused}"
    if not (args.chunk_index or args.chunk):
        return None if args.stamp else "--stamp is required (or --chunk-index with --chunk)"
    if args.stage != "gates" or not (args.chunk_index and args.chunk):
        return "--chunk-index and --chunk go together, on the gate stage only"
    if not Path(args.chunk_index).is_absolute():
        return ("--chunk-index must be an absolute path: the launched stage runs from the "
                "driver's own directory")
    args.chunk_index_data = C.read_index(Path(args.chunk_index))
    if args.chunk_index_data["machine"] != "nvidia":
        return f"{args.chunk_index} is a {args.chunk_index_data['machine']} index"
    args.chunk_manifest = C.find_chunk(args.chunk_index_data, args.chunk)
    if args.stamp and args.stamp != args.chunk_manifest["stamp"]:
        return (f"--stamp {args.stamp} is not chunk {args.chunk}'s stamp "
                f"{args.chunk_manifest['stamp']}")
    args.stamp = args.chunk_manifest["stamp"]
    return None


def share_problem(args: argparse.Namespace) -> Optional[str]:
    """Why ``--share-idle-foreign`` is refused, or None: the gate stage only, each value
    well formed (``run_lane.share_specs``) and ``--share-min-free-mib`` given with it."""
    specs = getattr(args, "share_idle_foreign", None) or []
    if not specs:
        return None
    if args.stage != "gates":
        return "--share-idle-foreign is for the gate stage only"
    import run_lane  # noqa: PLC0415 - beside this file; the one reader of the value
    try:
        run_lane.share_specs(specs)
    except ValueError as refused:
        return str(refused)
    if int(args.share_min_free_mib or 0) <= 0:
        return "--share-idle-foreign needs --share-min-free-mib (the unit's peak plus margin)"
    return None


def lanes_home(args: argparse.Namespace) -> Path:
    """The lanes' home: ``--home``, or ``<home>/<chunk stamp>`` for a chunk, so two chunks
    launched with one ``--home`` never share a lane directory."""
    home = Path(args.home).resolve()
    return home / args.stamp if getattr(args, "chunk_manifest", None) else home


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_common(sub.add_parser("run"))
    add_common(sub.add_parser("launch"))
    st = sub.add_parser("status")
    st.add_argument("--home", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "status":
        return round_status.main(["lanes", args.home])
    why = apply_chunk(args) or share_problem(args)
    if why:
        ap.error(why)
    home = lanes_home(args)
    home.mkdir(parents=True, exist_ok=True)
    if not args.lock_dir:
        ap.error("--lock-dir (or $ROUND_LOCK_DIR) is required")
    if args.cmd == "launch":
        out = home / f"{args.stage}.out"
        child = [sys.executable, "-u", str(Path(__file__).resolve()), "run",
                 *sys.argv[sys.argv.index("launch") + 1:]]
        with out.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(child, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True,
                                    cwd=str(HERE))
        (home / f"{args.stage}.pid").write_text(f"{proc.pid}\n")
        time.sleep(3)
        if proc.poll() is not None:
            print(f"the {args.stage} stage exited at once (rc {proc.returncode}); see {out}")
            return 1
        print(f"{args.stage}: pid {proc.pid} (own session); follow {home / 'round.log'} "
              f"and {home}/*/progress.log")
        return 0
    say(home, f"START stage {args.stage} stamp {args.stamp} cc {args.capability} repo "
              f"{args.repo} gpus<= {args.gpus} max_locks {args.max_locks}"
              + (f" chunk {args.chunk} ({len(args.chunk_manifest['units'])} units)"
                 if args.chunk_manifest else ""))
    if args.chunk_manifest is None:
        return stage_gates(args, home) if args.stage == "gates" else stage_route(args, home)
    lock = LaunchLock(home, f"1.0 round chunk {args.stamp} pid {os.getpid()}")
    why = lock.take()
    if why:
        say(home, f"STOP: {why}")
        return 3
    try:
        return stage_gates(args, home)
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
