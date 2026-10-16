#!/usr/bin/env python3
"""CERT-8, the release timing round, one ROW at a time.

WHY. Timing follows the certification's modular pattern: the unit is a ROW
(``parity/meep_gpu/timing_rows.py``), any subset of rows can be (re-)timed at any time,
each row's record is self-contained, and every table, ladder and dispatch-preference
record is CUT by tool from the newest valid row of each key. This driver is the
procedure the 1.0 round ran from lane scripts, in the release, so a new user re-times a
row with the release alone. ``docs/development/timing.md`` is the manual.

WHAT IT RUNS. Nothing of its own that is timed: every unit is a fresh process of the
release's own instruments -- ``timing_ladder.py`` for the identical-case rows (its
interleaving of GPU routes, MEEP blocks and the array control IS the row, so R5 holds by
construction) and ``bench_fused_products.py`` for the fused-product rows. This file
orders them, holds the session's host gates, watches the box while each unit runs, and
writes one record per row into the STORE.

    cert8.py plan    --catalog C --store S [--release-root R] [--rows K ...|--owed] [--json]
                     [--environment-record F]
    cert8.py run     --catalog C --store S --python PY --run-root DIR --session ID
                     [--rows K ...|--owed] [--environment-record F] [--gpu N]
                     [--ladder-params SITE_FILE] [--bench-env K=V ...]
                     [--tree DIR --role ab] [--no-variance WHY]
    cert8.py collect ladder --store S --session ID --dir LADDER_DIR --catalog C [...]
    cert8.py collect fused  --store S --session ID --dir OUT_DIR --key K [...]
    cert8.py selftest

THE CATALOG (JSON) names the rows of an architecture: ``architecture`` (``8.6``,
``applegpu_g13s``), ``ladder`` (``cases``, sizes per tier, ``passes``, ``routes``,
``variance``: the precondition's size, route and MEEP rank count, and ``ladder_args``
passed to every ladder unit: the build record, the binding, the site's paths) and
``fused`` (per table its DRIVE cases and its sweep: cases, resolutions, passes).
``cert8_catalog_nvidia.json`` and ``cert8_catalog_apple.json`` beside this file are the
1.0 round's two.

THE SESSION, in order (rule 7: one flushed line per step in ``<run>/progress.log``,
one marker per finished unit; the same command resumes it at its first unfinished
unit, and a unit is never resumed inside: an earlier output of it is set aside and the
unit runs whole, because a row is timed in ONE process under one watcher, R5):

1. The QUIET WINDOW, before EVERY unit, read twice ``HOLD_S`` apart and recorded in the
   unit's rows: no foreign process on the device, no other work of this account (a
   harness, test or MPI process outside this session, a busy Python process, a busy
   build, compiler or bulk tool, and on Linux any busy process of the account: CPU
   shares measured between readings, :class:`CpuShares`), no
   neighbour turning busy (R18: an idle resident process of another account on another
   GPU is recorded, never waited on); on Linux the load average below 10 % of the
   logical CPUs and available memory above 25 % of the total, the ladder's own GPU-row
   thresholds (on macOS, memory and the background daemons; the ladder gates each of
   its rows on the calibrated CPU threshold itself).
2. The VARIANCE PRECONDITION, once per session that times a ladder row: one point
   repeated five times on the GPU route and at MEEP's reference rank count. GO when the
   GPU route's repeat spread is within 5 %; MEEP's repeat spread is RECORDED AND NEVER
   GATED (owner ruling). Its verdict is in every row of the session, and a ratio of a
   session whose precondition did not read GO (``--no-variance``) never enters a table.
3. The UNITS: ladder rows grouped by (case, tier) into one ladder each (its passes
   interleave its sizes, as the 1.0 ladder did; one row alone is one size), then the
   fused DRIVE units, the sweep, and the dispatch checks. Every ladder unit is passed
   ``--require-ac-power 0``: no stage is gated on power (owner ruling); power is
   recorded beside every row.
4. The WATCHER, every ``WATCH_S`` while a unit runs: two consecutive readings that see a
   foreign process on the device, other work of this account, or a neighbour turning
   busy INTRUDE (R18). The unit is stopped, its output is set aside (renamed with the
   time, never deleted, never collected), and it re-runs once the window is quiet
   again; a record that carries an intrusion is refused by every cut.
5. COLLECT: each unit's rows become records in the store, with the session, the gates,
   the unit's command, environment and exit status, the reference environment's build
   record (R27) and the in-harness import records. A ladder unit that exits non-zero is
   recorded and refused by every cut (its passes are not the plan's).

WHICH ROWS. ``--rows`` names keys; ``--owed`` takes every catalog key whose newest row is
not valid on the release tree (``timing_rows.select``): a re-time for a stale or absent
key, a dispatch check for a key owed one. With ``--environment-record`` (which ``run``
requires) a row that ran in another reference environment is stale too (R27): an
environment is no file of the import record, so a rebuilt environment owes every row. ``timing_rows.py bill FILE`` says, before a
change lands, which keys it will re-time.

THE TRITON LAUNCHER'S TOOLCHAIN (Linux). ``run`` gives every unit ``CC`` and
``TRITON_LIBCUDA_PATH`` (``legs.launcher_toolchain``): the caller's, else the compiler and
the libcuda link of the reference environment ``--python`` runs in; a site's
``LIBCUDA_STUB`` (``--ladder-params``) and ``--bench-env`` still name another directory in
their units. A reference environment without its compiler, or without the link its build
recorded, is refused by name before the session starts: Triton would compile the timed
launchers with the host's ``gcc``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
for _path in (str(HARNESS), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import timing_rows as TR  # noqa: E402

CATALOG_SCHEMA = TR.CATALOG_SCHEMA
#: Seconds between the two readings of the quiet window, and between watcher readings.
HOLD_S = 120.0
WATCH_S = 15.0
#: How many times a unit an intrusion stopped is run again before the session stops.
MAX_ATTEMPTS = 3
#: The dispatch check's bench windows: short, timed for nothing, kept for the counters.
CHECK_BENCH_FLAGS = ["--repeats", "2", "--target-seconds", "0.5"]
#: The variance precondition's repeats (``timing_records.VARIANCE_MIN_REPEATS``).
VARIANCE_PASSES = 5
EXIT_REFUSED, EXIT_VARIANCE, EXIT_UNIT = 2, 5, 6
#: What an NVIDIA fused unit's CuPy cache reads until the session mints the keep cache.
KEEP_CACHE_PENDING = "<the session's keep cache>"
#: The keep-cache mint, the ladder's own (``timing_ladder.Ladder.mint_keep_cache``).
KEEP_CACHE_CODE = ("from meep_gpu import subnormal_policy as sp\n"
                   "r = sp.point_cupy_cache_at_keep_policy(); p = r.get('provenance') or {}\n"
                   "print(r['after'] if (not p.get('reasons')) and p.get('marked') else "
                   "'REFUSED ' + repr(p.get('reasons')))")


def export_launcher_toolchain(python: str) -> Dict[str, Any]:
    """Export the Triton launcher's toolchain into this process's environment, which every
    unit inherits (``legs.launcher_toolchain`` for ``python``'s reference environment, the
    caller's ``CC`` and ``TRITON_LIBCUDA_PATH`` first), and return it; refused by name when
    that environment lacks its compiler or its recorded libcuda link."""
    if str(HERE) not in sys.path:
        sys.path.append(str(HERE))
    import legs  # noqa: PLC0415 - beside this file; the toolchain rule lives there
    prefix = (legs.interpreter_prefix(python)
              if sys.platform.startswith(legs.LAUNCHER_PLATFORMS) else None)
    toolchain = legs.launcher_toolchain(prefix, dict(os.environ))
    if toolchain["problems"]:
        raise Refused("; ".join(toolchain["problems"]))
    toolchain["line"] = legs.launcher_line(toolchain)
    legs.launcher_environment(os.environ, toolchain)
    return toolchain


def mint_keep_cache(args: Any) -> str:
    """The session's CuPy cache, pointed at the keep policy, as the ladder mints its own.
    Run from the tree's root: ``python -c`` puts the working directory first on the
    path, so the tree that runs is the one that answers."""
    session_dir = os.path.join(args.run_root, args.session)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=args.tree,
               CUPY_CACHE_DIR=os.path.join(session_dir, "cache", "cupy"),
               PYTHONDONTWRITEBYTECODE="1")
    text = subprocess.run([args.python, "-c", KEEP_CACHE_CODE], cwd=args.tree, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900).stdout
    lines = [line for line in text.splitlines() if line.strip()]
    cache = lines[-1].strip() if lines else ""
    if "ftz_stripped" not in cache or not os.path.abspath(cache).startswith(
            os.path.abspath(session_dir) + os.sep):
        raise Refused(f"the keep cache was not minted inside the session: "
                      f"{cache or text[-400:]}", EXIT_UNIT)
    return cache


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Refused(SystemExit):
    def __init__(self, message: str, code: int = EXIT_REFUSED):
        super().__init__(code)
        self.message = message


# ---------------------------------------------------------------------------
# The catalog and its units
# ---------------------------------------------------------------------------

def load_catalog(path: str) -> Dict[str, Any]:
    try:
        return TR.load_catalog(path)
    except TR.RowRefused as refusal:
        raise Refused(str(refusal))


#: How the release's DRIVE cases of a table are read: the bench's own case selection
#: (``--estimate`` lifts nothing and runs on any host), one process per table.
ESTIMATE_LINE = re.compile(r"ESTIMATE   (\S+): ")


def release_drive_cases(python: str, tree: str, table: str, out: str) -> List[str]:
    """The DRIVE cases the release dispatches on ``table``, read from the bench."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([tree, os.path.join(tree, "parity", "meep_gpu")]))
    text = subprocess.run([python, "-u", "parity/meep_gpu/bench_fused_products.py",
                           "--drive-table", table, "--out", out, "--estimate"], cwd=tree,
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900).stdout
    cases = ESTIMATE_LINE.findall(text)
    if not cases:
        raise Refused(f"no {table} DRIVE case list from the bench: {text[-400:]}")
    return cases


def catalog_keys(catalog: Mapping[str, Any],
                 drive: Optional[Callable[[str], Sequence[str]]] = None) -> List[str]:
    """Every row key the catalog names, ladder rows first (``timing_rows.catalog_keys``,
    the spelling the cuts account against). A table whose ``drive`` is ``"release"``
    takes the release's own DRIVE cases from ``drive(table)``."""
    try:
        return TR.catalog_keys(catalog, drive)
    except TR.RowRefused as refusal:
        raise Refused(str(refusal))


def owed_keys(catalog: Mapping[str, Any], store: str, release_root: str,
              allow_unrecorded_environment: bool = False,
              drive: Optional[Callable[[str], Sequence[str]]] = None,
              environment_record: Optional[str] = None
              ) -> Tuple[List[str], List[str]]:
    """``(re-time, dispatch-check)``: the catalog keys whose newest row is not valid.
    With ``environment_record`` (the reference environment the session runs in, R27) a
    row of any other environment is not valid: a rebuilt environment owes every row."""
    selection = TR.select(TR.load_store(store), release_root, allow_unrecorded_environment,
                          environment=TR.file_sha256(environment_record))
    retime, check = [], []
    for key in catalog_keys(catalog, drive):
        chosen = selection.get(key)
        if chosen is None or chosen["owed"] == "a re-time":
            retime.append(key)
        elif chosen["owed"] == "a dispatch check":
            check.append(TR.check_key(key))
    return retime, check


class Unit:
    """One process the session runs, and the row keys it records."""

    def __init__(self, name: str, kind: str, keys: Sequence[str], argv: List[str],
                 out: str, env: Mapping[str, Optional[str]], cwd: str):
        self.name, self.kind, self.keys = name, kind, list(keys)
        self.argv, self.out, self.env, self.cwd = argv, out, dict(env), cwd

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "keys": self.keys,
                "argv": self.argv, "out": self.out, "cwd": self.cwd,
                "env": {k: v for k, v in sorted(self.env.items())}}


def _ladder_argv(args: Any, catalog: Mapping[str, Any], stamp: str, case: str,
                 tiers: List[str], sizes: Mapping[str, List[int]], passes: int,
                 extra: Sequence[str] = ()) -> List[str]:
    ladder = catalog.get("ladder") or {}
    argv = [args.python, "-u", "parity/meep_gpu/timing_ladder.py", "--stamp", stamp,
            "--python", args.python,
            "--run-root", os.path.join(args.run_root, args.session),
            "--tree-src", args.tree, "--cases", case,
            "--tiers", " ".join(tiers), "--passes", str(passes),
            # NEVER GATED ON POWER (owner ruling): power is recorded beside every row.
            "--require-ac-power", "0"]
    if ladder.get("routes"):
        argv += ["--routes", ",".join(ladder["routes"])]
    for tier, cells in sorted(sizes.items()):
        argv += [f"--{tier}-sizes" if tier != "preflight" else "--preflight-size",
                 ",".join(str(int(c)) for c in cells)]
    if args.gpu is not None:
        argv += ["--gpu", str(args.gpu)]
    if getattr(args, "ladder_params", None):
        # THE SITE'S OWN VALUES (the MEEP build record, a library stub, the corpus, the
        # account), in a file outside the tree: a catalog names no machine.
        argv += ["--params", args.ladder_params]
    return argv + list(ladder.get("ladder_args") or ()) + list(extra)


def _bench_env(args: Any, table: str) -> Dict[str, Optional[str]]:
    """What a fused unit's process is given: the ladder's GPU-row rules, by name."""
    env: Dict[str, Optional[str]] = {
        "PYTHONPATH": os.pathsep.join([args.tree, os.path.join(args.tree,
                                                               "parity", "meep_gpu")]),
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1", "MEEP_GPU_DISPATCH": None, "MEEP_GPU_FUSED": None,
        "MEEP_GPU_FUSE_ARMS": None, "MEEP_GPU_BACKEND_PREFERENCE": None,
        "MEEP_GPU_KERNEL_TABLE": None,
    }
    if table == "metal":
        env.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTORCH_ENABLE_MPS_FALLBACK": None,
                    "MEEP_GPU_HARDWARE": None})
    else:
        # THE LADDER'S NVIDIA GPU-ROW RULES (``timing_ladder.gpu_env``): a certified
        # device or nothing, the Triton and CuPy caches of this session, the CuPy cache
        # minted under the keep policy the rows time (filled in by the session).
        session_dir = os.path.join(args.run_root, args.session)
        env.update({"MEEP_GPU_HARDWARE": "certified", "CUPY_ACCELERATORS": "",
                    "CUDA_VISIBLE_DEVICES": None if args.gpu is None else str(args.gpu),
                    "TRITON_CACHE_DIR": os.path.join(session_dir, "cache", "triton"),
                    "CUPY_CACHE_DIR": KEEP_CACHE_PENDING,
                    "MEEP_GPU_SUBNORMAL_POLICY": "keep"})
        if table == "cuda":
            env["MEEP_GPU_BACKEND_PREFERENCE"] = "cuda"
    for item in args.bench_env or ():
        name, _, value = item.partition("=")
        env[name] = value
    return env


def _fused_argv(args: Any, spec: Mapping[str, Any], parsed: Mapping[str, Any],
                out: str) -> List[str]:
    """A fused unit's bench command; a dispatch check appends its short windows, which
    argparse lets override the unit's own."""
    argv = [args.python, "-u", "parity/meep_gpu/bench_fused_products.py",
            "--drive-table", parsed["table"], "--out", out, "--gpu", "0",
            "--case", parsed["case"]]
    if parsed["size"] != "drive":
        argv += ["--res", parsed["size"][3:]] + list(spec.get("sweep_args") or ())
    return argv + list(spec.get("bench_args") or ())


def _fused_order(items: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """DRIVE units first; then the sweep pass by pass, ascending sizes on odd passes and
    descending on even ones, so a drift lands on every size alike (the 1.0 order)."""
    drive = sorted((i for i in items if i["size"] == "drive"),
                   key=lambda i: (i["table"], i["case"]))
    sweep = [i for i in items if i["size"] != "drive"]
    ordered = list(drive)
    for index in sorted({i["pass"] for i in sweep}):
        held = [i for i in sweep if i["pass"] == index]
        held.sort(key=lambda i: (int(i["size"][3:]), i["table"], i["case"]),
                  reverse=index % 2 == 0)
        ordered += held
    return ordered


def build_units(args: Any, catalog: Mapping[str, Any], keys: Sequence[str]
                ) -> List[Unit]:
    """The session's units for ``keys``, in the session's order."""
    architecture = str(catalog["architecture"])
    ladder = catalog.get("ladder") or {}
    passes = int(ladder.get("passes") or 3)
    session_dir = os.path.join(args.run_root, args.session)
    units: List[Unit] = []
    groups: Dict[Tuple[str, str], List[int]] = {}
    fused: List[Dict[str, Any]] = []
    checks: List[str] = []
    for key in keys:
        parsed = TR.parse_key(key)
        if parsed["kind"] == "check":
            checks.append(parsed["target"])
            continue
        if parsed.get("architecture") != architecture:
            raise Refused(f"{key}: the catalog is for architecture {architecture}")
        if parsed["kind"] == "ladder":
            groups.setdefault((parsed["case"], parsed["tier"]), []).append(parsed["cells"])
        else:
            fused.append(dict(parsed, key=key))
    index = 0
    for (case, tier), cells in sorted(groups.items()):
        index += 1
        stamp = f"{args.session}_u{index:02d}"
        rows = [TR.ladder_key(architecture, case, c, tier) for c in sorted(set(cells))]
        argv = _ladder_argv(args, catalog, stamp, case, ["preflight", "digests", tier],
                            {tier: sorted(set(cells))}, passes)
        units.append(Unit(f"u{index:02d}_ladder_{case}_{tier}", "ladder", rows, argv,
                          os.path.join(session_dir, f"ladder_{stamp}"), {}, args.tree))
    spec = (catalog.get("fused") or {})
    for item in _fused_order(fused):
        index += 1
        out = os.path.join(session_dir, f"u{index:02d}_fused")
        argv = _fused_argv(args, spec, item, out)
        units.append(Unit(f"u{index:02d}_fused_{item['table']}_{item['case']}_"
                          f"{item['size']}_p{item['pass']}", "fused", [item["key"]], argv,
                          out, _bench_env(args, item["table"]), args.tree))
    for target in checks:
        index += 1
        parsed = TR.parse_key(target)
        if parsed["kind"] == "ladder":
            stamp = f"{args.session}_u{index:02d}"
            argv = _ladder_argv(args, catalog, stamp, parsed["case"], ["preflight"],
                                {"preflight": [parsed["cells"]]}, 1)
            out = os.path.join(session_dir, f"ladder_{stamp}")
            env: Dict[str, Optional[str]] = {}
        else:
            out = os.path.join(session_dir, f"u{index:02d}_check")
            argv = _fused_argv(args, spec, parsed, out) + CHECK_BENCH_FLAGS
            env = _bench_env(args, parsed["table"])
        units.append(Unit(f"u{index:02d}_check", "check", [TR.check_key(target)], argv,
                          out, env, args.tree))
    return units


def variance_unit(args: Any, catalog: Mapping[str, Any]) -> Optional[Unit]:
    """The session's variance precondition, or None when it times no ladder row."""
    spec = (catalog.get("ladder") or {}).get("variance")
    if not spec:
        return None
    stamp = f"{args.session}_variance"
    argv = _ladder_argv(args, catalog, stamp, spec["case"], ["preflight", "priority"],
                        {"priority": [int(spec["cells"])]}, VARIANCE_PASSES,
                        ["--routes", spec["route"], "--ranks", f"{spec['meep_ranks']}||",
                         "--controls", "none"])
    return Unit("variance", "variance", [], argv,
                os.path.join(args.run_root, args.session, f"ladder_{stamp}"), {},
                args.tree)


# ---------------------------------------------------------------------------
# Gates: the quiet window, the variance precondition, the watcher
# ---------------------------------------------------------------------------

def cpu_seconds(text: str) -> Optional[float]:
    """``ps``'s accumulated CPU time in seconds: ``[DD-]HH:MM:SS`` (Linux) or
    ``MM:SS.ss`` (macOS)."""
    days, _, clock = str(text).strip().rpartition("-")
    try:
        total = 0.0
        for part in clock.split(":"):
            total = total * 60.0 + float(part)
        return total + 86400.0 * (int(days) if days else 0)
    except ValueError:
        return None


class CpuShares:
    """Each process's share of one core BETWEEN TWO READINGS of the box, in percent.

    ``ps`` %cpu is not it: on Linux it is the process's lifetime average, so a process
    of the account that sat idle for an hour and then turned busy reads low on it. The
    1.0 GPU host's watcher measured the share from the CPU time used between readings,
    and this does the same, for the quiet window and the watcher alike."""

    def __init__(self) -> None:
        self._last: Optional[Tuple[float, Dict[int, Tuple[str, float]]]] = None
        self._lock = threading.Lock()

    def update(self, used: Mapping[int, Tuple[str, float]], now: float
               ) -> Dict[int, float]:
        with self._lock:
            previous, self._last = self._last, (now, dict(used))
        if previous is None or now <= previous[0]:
            return {}
        then, before = previous
        out = {}
        for pid, (command, seconds) in used.items():
            held = before.get(pid)
            if held is not None and held[0] == command and seconds >= held[1]:
                out[pid] = 100.0 * (seconds - held[1]) / (now - then)
        return out


#: One reader of the box per driver process: what each process used since the last one.
SHARES = CpuShares()


def _process_table(shares: Optional[CpuShares] = None
                   ) -> Dict[int, Tuple[int, int, float, str]]:
    """``{pid: (ppid, uid, %cpu, command)}`` from ``ps``, on Linux and macOS alike. With
    ``shares`` the %cpu is the share measured since the previous reading where there is
    one, else ``ps``'s own figure."""
    try:
        text = subprocess.run(["ps", "-wwaxo", "pid=,ppid=,uid=,pcpu=,time=,command="],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    table = {}
    used: Dict[int, Tuple[str, float]] = {}
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                share = float(parts[3])
            except ValueError:
                share = 0.0
            pid = int(parts[0])
            table[pid] = (int(parts[1]), int(parts[2]) if parts[2].isdigit() else -1,
                          share, parts[5])
            seconds = cpu_seconds(parts[4])
            if seconds is not None:
                used[pid] = (parts[5], seconds)
    if shares is not None:
        measured = shares.update(used, time.time())
        table = {pid: (row[0], row[1], measured.get(pid, row[2]), row[3])
                 for pid, row in table.items()}
    return table


def own_pids(table: Mapping[int, Tuple[Any, ...]], root: int,
             markers: Sequence[str] = ()) -> set:
    """This session's processes: the driver's descendants, and every process whose
    command names the session's run directory (an MPI rank a launcher daemon started
    is not a descendant), with theirs."""
    mine = {root} | {pid for pid, row in table.items()
                     if any(marker and marker in row[-1] for marker in markers)}
    grew = True
    while grew:
        grew = False
        for pid, row in table.items():
            if pid not in mine and row[0] in mine:
                mine.add(pid)
                grew = True
    return mine


#: Other work of this account that disturbs a timing (R18, "no other work of ours"): a
#: harness script, a test run, a round driver, a MEEP launch. Matched on the command.
OTHER_WORK = re.compile(
    r"pytest|bench_\w+\.py|gate_\w+\.py|probe_\w+\.py|timing_ladder\.py|round\.py|"
    r"run_lane\.py|l3_row|l3_corpus|metal_round\.py|recut_\w+|rebind_\w+|record_\w+\.py|"
    r"build_\w+\.py|cut_timing_record\.py|census|fleet_launch|check_failures|legs\.py")
#: The MPI launchers and their daemons: a MEEP run of this account's.
MPI_PROGRAMS = ("mpirun", "mpiexec", "prterun", "prted", "orted", "srun", "hydra_pmi_proxy")
#: A Python process of this account above this share of one core is busy.
BUSY_PYTHON_PERCENT = 25.0
#: Builds, compilers and bulk tools of this account (an environment build, a kernel
#: compile, a copy, an archive, a search): heavy only when they burn CPU, at this share
#: of one core (the 1.0 Apple watcher's tool rule, which the build tools join).
BUSY_TOOLS = frozenset((
    "make", "gmake", "ninja", "cmake", "cc", "c++", "cc1", "cc1plus", "gcc", "g++",
    "clang", "clang++", "ld", "nvcc", "ptxas", "cicc", "fatbinary", "gfortran", "f951",
    "conda", "mamba", "micromamba", "pip", "tar", "rsync", "find", "du", "git", "cp",
    "ditto", "zip", "unzip", "gzip", "xz", "zstd", "shasum", "sha256sum", "ugrep", "rg",
    "grep", "mdfind"))
BUSY_TOOL_PERCENT = 20.0
#: On Linux, the 1.0 GPU host's rule: ANY process of this account outside the session
#: above this share of one core intrudes (its account runs jobs, not a desktop). On macOS
#: the account also runs the desktop, so only the kinds above are read there.
BUSY_ANY_PERCENT = 25.0


def other_work(table: Mapping[int, Tuple[int, int, float, str]], mine: set, uid: int,
               self_pid: Optional[int] = None, platform: Optional[str] = None
               ) -> List[str]:
    """This account's processes outside the session that disturb a timing, each named.
    The %cpu of ``table`` is the share measured between readings (:class:`CpuShares`)."""
    platform = sys.platform if platform is None else platform
    out = []
    for pid, (_ppid, owner, share, command) in sorted(table.items()):
        if pid in mine or owner != uid or pid == self_pid:
            continue
        argv = command.split()
        program = os.path.basename(argv[0]) if argv else ""
        script = argv[1] if len(argv) > 1 and argv[1] != "-c" else ""
        if program in MPI_PROGRAMS:
            why = "an MPI launch"
        elif ((program.startswith("python") or program == "pytest")
              and OTHER_WORK.search(command)) or (
                  program in ("bash", "sh", "zsh") and OTHER_WORK.search(script)):
            why = "a harness, test or round process"
        elif program.startswith("python") and share >= BUSY_PYTHON_PERCENT:
            why = f"a busy Python process ({share:.0f} % of a core)"
        elif program in BUSY_TOOLS and share >= BUSY_TOOL_PERCENT:
            why = f"a busy build or tool process ({share:.0f} % of a core)"
        elif not platform.startswith("darwin") and share >= BUSY_ANY_PERCENT:
            why = f"a busy process ({share:.0f} % of a core)"
        else:
            continue
        out.append(f"other work of this account ({why}): {pid} {command[:160]}")
    return out


def gate_reasons(platform: str, load1: Optional[float], memory: Mapping[str, Any],
                 logical_cpus: Optional[int], daemons: Optional[List[str]] = None
                 ) -> List[str]:
    """What keeps a unit from STARTING beyond what the watcher reads: the ladder's own
    GPU-row thresholds (``timing_host.default_cpu_max``, ``default_mem_min_gib``). On
    macOS the load average is not the ladder's metric (it calibrates busy CPUs per
    ladder), so memory and the background daemons are read there and load is recorded."""
    import timing_host  # noqa: PLC0415

    reasons = []
    available = (memory or {}).get("available_gib")
    floor = timing_host.default_mem_min_gib((memory or {}).get("total_gib"))
    if available is None:
        reasons.append("available memory unreadable")
    elif not float(available) > float(floor):
        reasons.append(f"available memory {available} GiB <= {floor} (25 % of the total)")
    if platform == "darwin":
        if daemons is None:
            reasons.append("background daemons unreadable (ps)")
        elif daemons:
            reasons.append(f"background daemons busy: {daemons}")
        return reasons
    ceiling = timing_host.default_cpu_max(logical_cpus)
    if load1 is None:
        reasons.append("load average unreadable")
    elif not float(load1) < float(ceiling):
        reasons.append(f"load1 {load1} >= {ceiling} (10 % of {logical_cpus} logical CPUs)")
    return reasons


def host_reading(gpu: Optional[str], root_pid: int, gate: bool = False,
                 markers: Sequence[str] = ()) -> Dict[str, Any]:
    """One reading of the box: what would intrude, and what is only recorded. With
    ``gate`` (the quiet window before a unit) the load and memory thresholds join."""
    import timing_host  # noqa: PLC0415 - the ladder's own readers

    table = _process_table(SHARES)
    mine = own_pids(table, root_pid, markers)
    reading: Dict[str, Any] = {"utc": utc(), "intrusions": [], "recorded": {}}
    uid = os.getuid()
    if not table:
        reading["intrusions"].append("unreadable: ps")
    reading["intrusions"] += other_work(table, mine, uid, os.getpid())
    memory = timing_host.memory()
    reading["recorded"]["load"] = timing_host.loadavg()
    reading["recorded"]["memory"] = memory
    if sys.platform == "darwin":
        found = timing_host.metal_neighbours()
        if found is None:
            reading["intrusions"].append("unreadable: ps")
        else:
            foreign = [line for line in found if int(line.split()[0]) not in mine]
            reading["intrusions"] += [f"another process driving the Apple GPU: {line}"
                                      for line in foreign]
        daemons = timing_host.darwin_daemons()
        reading["recorded"]["daemons"] = daemons
        reading["recorded"]["power"] = timing_host.cpu_state_darwin()
        reading["recorded"]["apple_gpu"] = timing_host.apple_gpu_state()
        if gate:
            reading["intrusions"] += gate_reasons("darwin", timing_host.load1(), memory,
                                                  os.cpu_count(), daemons)
        return reading
    if gate:
        reading["intrusions"] += gate_reasons(sys.platform, timing_host.load1(), memory,
                                              os.cpu_count())
    apps = timing_host.compute_apps(None)
    uuids = timing_host.gpu_uuids() or {}
    if apps is None:
        reading["intrusions"].append("unreadable: nvidia-smi compute apps")
        return reading
    pinned = timing_host.as_uuid(gpu, uuids) if gpu is not None else None
    util = {}
    text = timing_host.run_text(["nvidia-smi", "--query-gpu=uuid,utilization.gpu",
                                 "--format=csv,noheader,nounits"]) or ""
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            util[parts[0]] = int(parts[1])
    neighbours = []
    for app in apps:
        pid = int(app.get("pid") or 0)
        owner = timing_host.uid_of_pid(pid)
        device = app.get("gpu_uuid")
        if pid in mine:
            continue
        if device == pinned:
            reading["intrusions"].append(f"a foreign compute process on the device: {app}")
        elif owner == uid:
            reading["intrusions"].append(f"another GPU process of this account: {app}")
        elif util.get(device):
            reading["intrusions"].append(f"a neighbour turned busy ({util[device]} %): "
                                         f"{app}")
        else:
            neighbours.append(dict(app, utilization=util.get(device)))
    reading["recorded"]["idle_neighbours"] = neighbours
    return reading


class Watcher:
    """Readings every ``every`` seconds while a unit runs; two in a row that intrude
    stop it (``on_intrusion``). The reader is injectable so the rule is tested without
    a device."""

    def __init__(self, reader: Callable[[], Dict[str, Any]], every: float = WATCH_S,
                 log: Optional[Callable[[str], None]] = None,
                 on_intrusion: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.reader, self.every, self.log = reader, every, log or (lambda _m: None)
        self.on_intrusion = on_intrusion
        self.readings: List[Dict[str, Any]] = []
        self.intrusions: List[Dict[str, Any]] = []
        self._episode = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def observe(self, reading: Dict[str, Any]) -> bool:
        """Take one reading; True when it and the one before it both intrude."""
        self.readings.append(reading)
        self.log("watch " + json.dumps(reading, sort_keys=True, default=str))
        previous = self.readings[-2] if len(self.readings) > 1 else None
        if not reading.get("intrusions"):
            self._episode = False
            return False
        if previous and previous.get("intrusions"):
            # ONE EVENT PER EPISODE: a neighbour that stays busy for ten readings is one
            # intrusion, first seen on the reading before the second.
            if not self._episode:
                self._episode = True
                event = {"utc": reading.get("utc"), "first_seen": previous.get("utc"),
                         "what": sorted(set(reading["intrusions"]))}
                self.intrusions.append(event)
                if self.on_intrusion is not None:
                    self.on_intrusion(event)
            return True
        return False

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.observe(self.reader())
                except Exception as error:  # noqa: BLE001 - a reading that fails is noted
                    self.observe({"utc": utc(), "intrusions": [f"unreadable: {error!r}"]})
                self._stop.wait(self.every)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.every + 5)
        # THE FIRST AND THE LAST READING, KEPT WHOLE (R18): the neighbours, load, memory
        # and power at the start and the end of every timed unit.
        return {"every_s": self.every, "readings": len(self.readings),
                "first": self.readings[0] if self.readings else None,
                "last": self.readings[-1] if self.readings else None,
                "intrusions": list(self.intrusions)}


def quiet_window(reader: Callable[[], Dict[str, Any]], hold: float = HOLD_S,
                 give_up: float = 43200.0, log: Callable[[str], None] = print,
                 sleep: Callable[[float], None] = time.sleep) -> List[Dict[str, Any]]:
    """Two readings ``hold`` apart, neither intruding: the window the session opens on."""
    started = time.time()
    while True:
        first = reader()
        if not first.get("intrusions"):
            sleep(hold)
            second = reader()
            if not second.get("intrusions"):
                log(f"quiet window: clear on two readings {int(hold)} s apart")
                return [first, second]
            log(f"quiet window: second reading not clear: {second['intrusions']}")
        else:
            log(f"quiet window: not clear: {first['intrusions']}")
        if time.time() - started > give_up:
            raise Refused(f"no quiet window in {int(give_up)} s", EXIT_UNIT)
        sleep(60.0)


def variance_gate(ledger: str, meep_ranks: int, route: str) -> Dict[str, Any]:
    """The precondition's verdict: GO on the GPU route's repeat spread alone.

    ``timing_records.variance_verdict`` computes every side; here the GPU side decides,
    and MEEP's repeat spread is RECORDED, never gated (owner ruling), as its pass spread
    is everywhere else in the round."""
    import timing_records  # noqa: PLC0415

    verdict = timing_records.variance_verdict(ledger, meep_ranks, route,
                                              check_ratio=False)
    gpu = verdict["sides"]["gpu"]
    why = []
    if gpu["readable"] < verdict["min_repeats"]:
        why.append(f"gpu: {gpu['readable']} readable repeats < {verdict['min_repeats']}")
    elif gpu["spread"] is None or gpu["spread"] > verdict["max_spread"]:
        why.append(f"gpu: repeat spread {gpu['spread']} > {verdict['max_spread']}")
    verdict.update({"ok": not why, "verdict": "GO" if not why else "NO-GO", "why": why,
                    "meep_rule": TR.MEEP_SPREAD_RULE,
                    "gated_sides": ["gpu"]})
    return verdict


# ---------------------------------------------------------------------------
# Environment and records
# ---------------------------------------------------------------------------

def environment_block(path: Optional[str], run_dir: Optional[str] = None
                      ) -> Dict[str, Any]:
    """The reference environment's build record (R27) and the ladder's MEEP build
    records, each by name and digest: never a path, which names a machine."""
    block: Dict[str, Any] = {"reference": None, "meep_builds": {}}
    if path:
        with open(path, "rb") as handle:
            payload = handle.read()
        try:
            parsed = json.loads(payload)
        except ValueError:
            parsed = None
        block["reference"] = {"name": os.path.basename(path),
                              "sha256": hashlib.sha256(payload).hexdigest(),
                              "record": parsed}
    builds = os.path.join(run_dir, "meep_builds") if run_dir else None
    if builds and os.path.isdir(builds):
        for name in sorted(os.listdir(builds)):
            with open(os.path.join(builds, name), "rb") as handle:
                block["meep_builds"][name] = hashlib.sha256(handle.read()).hexdigest()
    return block


def collect(unit: Unit, *, store: str, session: Mapping[str, Any], catalog: Mapping[str, Any],
            role: str, environment: Optional[str], gates: Mapping[str, Any],
            rc: Optional[int] = None) -> List[str]:
    """Write the unit's rows into the store; the paths written. The unit block carries
    its command, the environment this driver gave it (``None``: unset) and its exit
    status, which the cuts read (a ladder that stopped early is refused)."""
    architecture = str(catalog["architecture"])
    unit_block = {"name": unit.name, "argv": list(unit.argv),
                  "env": {k: v for k, v in sorted(unit.env.items())}, "rc": rc}
    written = []
    if unit.kind in ("ladder",) or (unit.kind == "check" and
                                    TR.parse_key(unit.keys[0])["target"].startswith(
                                        "ladder|")):
        records = TR.ladder_records(
            unit.out, session=session, architecture=architecture,
            environment=environment_block(environment, unit.out), gates=gates,
            unit=unit_block, role=role,
            tiers=TR.TIMED_TIERS if unit.kind == "ladder" else ("preflight",))
        for record in records:
            if unit.kind == "check":
                parsed = TR.parse_key(record["key"])
                target = TR.parse_key(unit.keys[0])["target"]
                record = dict(record, kind="check", key=unit.keys[0])
                if TR.parse_key(target)["cells"] != parsed["cells"]:
                    continue
            elif record["key"] not in unit.keys:
                continue
            written.append(TR.write_record(store, record))
        return written
    key = unit.keys[0]
    record, line = TR.fused_record(
        os.path.join(unit.out, "rows.jsonl"), key=key, session=session,
        environment=environment_block(environment), gates=gates, unit=unit_block,
        role=role, kind="check" if unit.kind == "check" else "fused")
    written.append(TR.write_record(store, record, bench_line=line))
    return written


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, args: Any, catalog: Mapping[str, Any],
                 reader: Optional[Callable[[], Dict[str, Any]]] = None,
                 runner: Optional[Callable[[Unit, Callable[[], bool]], int]] = None,
                 minter: Optional[Callable[[Any], str]] = None,
                 gate_reader: Optional[Callable[[], Dict[str, Any]]] = None):
        self.args, self.catalog = args, catalog
        self.minter = minter or mint_keep_cache
        self.dir = os.path.join(args.run_root, args.session)
        os.makedirs(os.path.join(self.dir, "markers"), exist_ok=True)
        self.log_path = os.path.join(self.dir, "progress.log")
        # THE WATCHER'S READING, AND THE QUIET WINDOW'S: the window adds the load and
        # memory thresholds a running unit itself would breach. A process whose command
        # names this session's directory is this session's.
        markers = (os.path.abspath(self.dir) + os.sep,)
        self.reader = reader or (lambda: host_reading(args.gpu, os.getpid(),
                                                      markers=markers))
        self.gate_reader = gate_reader or reader or (
            lambda: host_reading(args.gpu, os.getpid(), gate=True, markers=markers))
        self.runner = runner or self.run_process
        self.child: Optional[subprocess.Popen] = None
        self.started = utc()

    def say(self, message: str) -> None:
        line = f"{utc()} {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def marker(self, unit: Unit) -> str:
        return os.path.join(self.dir, "markers", f"{unit.name}.done")

    def run_process(self, unit: Unit, stopped: Callable[[], bool]) -> int:
        env = dict(os.environ)
        if unit.env:
            # A BENCH UNIT INHERITS NO PACKAGE SWITCH: every MEEP_GPU_* the unit does not
            # set is unset, as the ladder unsets them for its rows.
            for name in [n for n in env if n.startswith("MEEP_GPU_") and n not in unit.env]:
                env.pop(name)
        for name, value in unit.env.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        os.makedirs(os.path.dirname(unit.out) or ".", exist_ok=True)
        log = os.path.join(self.dir, f"{unit.name}.log")
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(f"# {utc()} {shlex.join(unit.argv)}\n")
            handle.flush()
            self.child = subprocess.Popen(unit.argv, cwd=unit.cwd, env=env, stdout=handle,
                                          stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL,
                                          start_new_session=True)
            while self.child.poll() is None:
                if stopped():
                    self.stop_child()
                time.sleep(1.0)
            rc = self.child.returncode
        self.child = None
        return rc

    def stop_child(self) -> None:
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(60):
            if child.poll() is not None:
                return
            time.sleep(1.0)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:
            pass

    def set_aside(self, unit: Unit, why: str) -> None:
        if os.path.exists(unit.out):
            aside = f"{unit.out}.{why}_{utc().replace(':', '')}"
            os.rename(unit.out, aside)
            self.say(f"{unit.name}: an earlier output set aside as "
                     f"{os.path.basename(aside)}")

    def run_unit(self, unit: Unit) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
        """Run one unit, opened on its own quiet window and under the watcher; re-run it
        after an intrusion. Returns its exit status, the watcher's block and the window.

        A UNIT IS NEVER RESUMED INSIDE (R5): an output already there (an attempt of an
        earlier process, which this watcher did not see) is set aside, never deleted, and
        the unit runs whole, so every leg of a row is timed in this process's window."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            quiet = quiet_window(self.gate_reader, hold=getattr(self.args, "hold_s", HOLD_S),
                                 log=self.say)
            self.set_aside(unit, "set_aside")
            hit = threading.Event()
            watcher = Watcher(self.reader, getattr(self.args, "watch_every_s", WATCH_S),
                              log=lambda m: None,
                              on_intrusion=lambda event: (hit.set(), self.say(
                                  f"INTRUSION during {unit.name}: {event['what']}")))
            self.say(f"{unit.name}: START attempt {attempt}: keys {unit.keys}")
            started = time.time()
            watcher.start()
            try:
                rc = self.runner(unit, hit.is_set)
            finally:
                watch = watcher.stop()
            self.say(f"{unit.name}: END rc {rc} after {int(time.time() - started)} s; "
                     f"watcher {watch['readings']} readings, "
                     f"{len(watch['intrusions'])} intrusion(s)")
            if not watch["intrusions"]:
                return rc, watch, quiet
            self.set_aside(unit, "intruded")
            self.say(f"{unit.name}: re-running once the window is quiet")
        raise Refused(f"{unit.name}: intruded on {MAX_ATTEMPTS} times", EXIT_UNIT)

    def run(self, units: Sequence[Unit]) -> int:
        args = self.args
        self.say(f"SESSION {args.session} START: {len(units)} unit(s), role {args.role}, "
                 f"architecture {self.catalog['architecture']}")
        gates: Dict[str, Any] = {"variance": None}
        if any(unit.kind == "ladder" for unit in units):
            if args.no_variance:
                gates["variance"] = {"ok": None, "not_run": args.no_variance}
                self.say(f"variance precondition NOT RUN: {args.no_variance}")
            else:
                unit = variance_unit(args, self.catalog)
                if unit is None:
                    raise Refused("the catalog names no variance precondition; pass "
                                  "--no-variance WHY to time ladder rows without one")
                rc, _watch, _quiet = self.run_unit(unit)
                spec = self.catalog["ladder"]["variance"]
                verdict = variance_gate(os.path.join(unit.out, "ladder_rows.jsonl"),
                                        int(spec["meep_ranks"]), spec["route"])
                with open(os.path.join(self.dir, "variance.json"), "w") as handle:
                    json.dump(verdict, handle, indent=1, sort_keys=True)
                gates["variance"] = verdict
                self.say(f"variance precondition {verdict['verdict']} (rc {rc}): "
                         f"{verdict['why'] or 'GPU repeat spread within the gate'}")
                if not verdict["ok"]:
                    raise Refused("the variance precondition refused: the box's spread "
                                  "swamps the differences", EXIT_VARIANCE)
        if any(unit.env.get("CUPY_CACHE_DIR") == KEEP_CACHE_PENDING for unit in units):
            cache = self.minter(args)
            self.say(f"keep cache: {cache}")
            for unit in units:
                if unit.env.get("CUPY_CACHE_DIR") == KEEP_CACHE_PENDING:
                    unit.env["CUPY_CACHE_DIR"] = cache
        session = {"id": args.session, "started_utc": self.started, "ended_utc": None}
        failed = 0
        for index, unit in enumerate(units, 1):
            if os.path.exists(self.marker(unit)):
                self.say(f"unit {index}/{len(units)} {unit.name}: done earlier, skipped")
                continue
            rc, watch, quiet = self.run_unit(unit)
            # THE UNIT'S OWN GATES: the window it opened on, the watcher while it ran,
            # power recorded and never gated (owner ruling).
            unit_gates = dict(gates, quiet=quiet, watcher=watch,
                              power={"gated": False,
                                     "recorded": quiet[-1].get("recorded", {}).get(
                                         "power")})
            session["ended_utc"] = utc()
            try:
                written = collect(unit, store=args.store, session=session,
                                  catalog=self.catalog, role=args.role,
                                  environment=args.environment_record, gates=unit_gates,
                                  rc=rc)
            except (TR.RowRefused, OSError, ValueError) as error:
                failed += 1
                self.say(f"unit {index}/{len(units)} {unit.name}: NOT COLLECTED: {error}")
                continue
            if rc != 0 or not written:
                # NOT MARKED: the next run of the same command runs it again, whole (its
                # output is set aside), and what was written stays in the store.
                failed += 1
                self.say(f"unit {index}/{len(units)} {unit.name}: rc {rc}, "
                         f"{len(written)} record(s) written; NOT MARKED DONE")
                continue
            with open(self.marker(unit), "w") as handle:
                handle.write(f"{utc()} rc {rc} records {len(written)}\n")
            self.say(f"unit {index}/{len(units)} {unit.name}: rc {rc}, "
                     f"{len(written)} record(s) written")
        self.say(f"SESSION {args.session} DONE: {len(units) - failed} of {len(units)} "
                 "unit(s) collected")
        return 0 if not failed else EXIT_UNIT


# ---------------------------------------------------------------------------
# Self-test: the intrusion path with stand-ins, no device
# ---------------------------------------------------------------------------

def selftest() -> int:
    """A stand-in unit (``sleep``) under the watcher, with a reader that turns intrusive
    on its third reading: the unit must be stopped on the fourth, and one-off readings
    must not stop it."""
    readings = iter([{"intrusions": []}, {"intrusions": ["x"]}, {"intrusions": []},
                     {"intrusions": ["y"]}, {"intrusions": ["y"]}] + [{"intrusions": []}]
                    * 100)
    hit = threading.Event()
    watcher = Watcher(lambda: dict(next(readings), utc=utc()), every=0.2,
                      on_intrusion=lambda _event: hit.set())
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    started = time.time()
    watcher.start()
    while child.poll() is None and time.time() - started < 20:
        if hit.is_set():
            os.killpg(child.pid, signal.SIGTERM)
        time.sleep(0.1)
    watch = watcher.stop()
    ok = (child.poll() is not None and len(watch["intrusions"]) == 1
          and watch["intrusions"][0]["what"] == ["y"] and time.time() - started < 20)
    print(json.dumps({"ok": ok, "stopped_after_s": round(time.time() - started, 1),
                      "intrusions": watch["intrusions"]}, indent=1, sort_keys=True))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _drive_reader(args: Any) -> Callable[[str], Sequence[str]]:
    held: Dict[str, Sequence[str]] = {}

    def read(table: str) -> Sequence[str]:
        if table not in held:
            held[table] = release_drive_cases(
                args.python, args.tree, table,
                os.path.join(args.run_root, args.session, f"estimate_{table}"))
        return held[table]
    return read


def _keys_asked(args: Any, catalog: Mapping[str, Any]) -> List[str]:
    drive = _drive_reader(args)
    if args.rows:
        if all(TR.parse_key(key).get("kind", "check") == "ladder"
               or TR.parse_key(TR.parse_key(key).get("target", key))["kind"] == "ladder"
               for key in args.rows):
            drive = None   # a ladder-only selection reads no DRIVE list
        known = set(catalog_keys(dict(catalog, fused={}) if drive is None else catalog,
                                 drive))
        for key in args.rows:
            target = TR.parse_key(key).get("target", key)
            if target not in known:
                raise Refused(f"{key}: not a row of the catalog")
        return list(args.rows)
    if args.owed:
        retime, check = owed_keys(catalog, args.store, args.release_root,
                                  args.allow_unrecorded_environment, drive,
                                  args.environment_record)
        return retime + check
    return catalog_keys(catalog, drive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--catalog", required=True)
        p.add_argument("--store", required=True,
                       help="the row store, kept under parity/meep_gpu/results/<name>/ "
                            "so a record cut from it names its rows as results/...")
        p.add_argument("--release-root", default=str(REPO_ROOT))
        group = p.add_mutually_exclusive_group()
        group.add_argument("--rows", nargs="+", help="the keys to time (check|<key> for "
                                                     "a dispatch check)")
        group.add_argument("--owed", action="store_true",
                           help="every key whose newest row is not valid on the release")
        p.add_argument("--allow-unrecorded-environment", action="store_true")
        p.add_argument("--environment-record",
                       help="the reference environment's build record (R27): every row "
                            "records it, and --owed owes a re-time to every row of "
                            "another environment")
        p.add_argument("--python", default=sys.executable)
        staging = os.environ.get("MGPU_SITE_STAGING_ROOT")
        p.add_argument("--run-root", required=staging is None,
                       default=os.path.join(staging, "cert8") if staging else None,
                       help="where the units write: outside every tree the ladder "
                            "reads (default $MGPU_SITE_STAGING_ROOT/cert8)")
        p.add_argument("--session", default=time.strftime("%Y-%m-%dT%H%M%SZ",
                                                          time.gmtime()))
        p.add_argument("--tree", default=str(REPO_ROOT),
                       help="the tree that runs: the release checkout, or with --role ab "
                            "the A/B base (the previous release, staged by git archive)")
        p.add_argument("--role", choices=TR.ROLES, default="release")
        p.add_argument("--gpu", default=None)
        p.add_argument("--ladder-params", metavar="FILE",
                       help="a timing_ladder.py --params file of the site's values "
                            "(kept outside the tree), passed to every ladder unit")
        p.add_argument("--bench-env", action="append", default=[],
                       metavar="KEY=VALUE", help="a site variable of the fused units "
                                                 "(cache directories, a library stub)")

    plan = sub.add_parser("plan", help="the units a run would start; nothing runs")
    common(plan)
    plan.add_argument("--json", action="store_true")
    run = sub.add_parser("run", help="time the rows asked for, one session")
    common(run)
    run.add_argument("--no-variance", metavar="WHY",
                     help="time ladder rows without the precondition, and say why")
    run.add_argument("--hold-s", type=float, default=HOLD_S)
    run.add_argument("--watch-every-s", type=float, default=WATCH_S)
    gather = sub.add_parser("collect", help="record a unit run by hand: kept in the store "
                                            "for status and bill, never cut (no quiet "
                                            "window or watcher was recorded around it)")
    gather.add_argument("what", choices=("ladder", "fused"))
    gather.add_argument("--catalog", required=True)
    gather.add_argument("--store", required=True)
    gather.add_argument("--session", required=True)
    gather.add_argument("--dir", required=True)
    gather.add_argument("--key", action="append", default=[])
    gather.add_argument("--role", choices=TR.ROLES, default="release")
    gather.add_argument("--environment-record")
    gather.add_argument("--started-utc", default=None)
    gather.add_argument("--ended-utc", default=None)
    sub.add_parser("selftest", help="the watcher's intrusion rule on stand-ins")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "selftest":
            return selftest()
        catalog = load_catalog(args.catalog)
        if args.command == "collect":
            session = {"id": args.session, "started_utc": args.started_utc or utc(),
                       "ended_utc": args.ended_utc or utc()}
            gates = {"quiet": None, "variance": None, "watcher": None,
                     "power": {"gated": False, "recorded": None},
                     "collected_by_hand": True}
            if args.what == "ladder":
                keys = args.key or [k for k in catalog_keys(catalog)
                                    if k.startswith("ladder|")]
                unit = Unit("collected", "ladder", keys, ["collect"], args.dir, {}, ".")
            else:
                if len(args.key) != 1:
                    raise Refused("collect fused takes exactly one --key")
                unit = Unit("collected", "fused", args.key, ["collect"], args.dir, {}, ".")
            written = collect(unit, store=args.store, session=session, catalog=catalog,
                              role=args.role, environment=args.environment_record,
                              gates=gates)
            print(f"collected {len(written)} record(s)", flush=True)
            return 0 if written else EXIT_UNIT
        keys = _keys_asked(args, catalog)
        units = build_units(args, catalog, keys)
        if args.command == "plan":
            described = {"session": args.session, "keys": keys,
                         "units": [unit.describe() for unit in units],
                         "variance": (variance_unit(args, catalog).describe()
                                      if variance_unit(args, catalog) and any(
                                          u.kind == "ladder" for u in units) else None)}
            if args.json:
                print(json.dumps(described, indent=1, sort_keys=True))
            else:
                print(f"PLAN {len(keys)} key(s), {len(units)} unit(s); nothing runs")
                for unit in units:
                    print(f"  {unit.name}: {len(unit.keys)} key(s): "
                          f"{shlex.join(unit.argv)}")
            return 0
        if not args.environment_record and not args.allow_unrecorded_environment:
            raise Refused("--environment-record is required: every row names the "
                          "reference environment it ran in (R27)")
        print(export_launcher_toolchain(args.python)["line"], flush=True)
        return Session(args, catalog).run(units)
    except Refused as refusal:
        print(f"REFUSED {refusal.message}", flush=True)
        return int(refusal.code)


if __name__ == "__main__":
    raise SystemExit(main())
