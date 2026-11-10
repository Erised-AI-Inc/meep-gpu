#!/usr/bin/env python3
"""CERT-8, the release timing round, one ROW at a time.

Why. Timing follows the certification's modular pattern: the unit is a ROW
(``parity/meep_gpu/timing_rows.py``), any subset of rows can be (re-)timed at any time,
each row's record is self-contained, and every table, ladder and dispatch-preference
record is CUT by tool from the newest valid row of each key. This driver is the
procedure the 1.0 certification round ran from scripts outside the release, now in the
release, so a new user re-times a row with the release alone.
``docs/development/timing.md`` is the manual.

What it runs. Nothing of its own that is timed: every unit is a fresh process of the
release's own instruments -- ``timing_ladder.py`` for the identical-case rows (its
interleaving of GPU routes, MEEP blocks and the array control IS the row, so MEEP is
re-timed in the same session as the GPU rows by construction) and
``bench_fused_products.py`` for the fused-product rows. This file
orders them, holds the session's host gates, watches the box while each unit runs, and
writes one record per row into the STORE.

    cert8.py plan    --catalog C --store S [--release-root R] [--rows K ...|--owed] [--json]
                     [--environment-record F]
    cert8.py run     --catalog C --store S --python PY --run-root DIR --session ID
                     [--rows K ...|--owed] [--environment-record F] [--gpu N]
                     [--ladder-params SITE_FILE] [--bench-env K=V ...]
                     [--tree DIR --role ab] [--no-variance WHY | --variance]
                     [--gpu-clock-lock G,M] [--windows-cpu-max CPUS]
                     [--vm-affinity MASK [--split-ladder [--meep-vm-affinity MASK]]]
    cert8.py collect ladder --store S --session ID --dir LADDER_DIR --catalog C [...]
    cert8.py collect fused  --store S --session ID --dir OUT_DIR --key K [...]
    cert8.py selftest

The catalog (JSON) names the rows of an architecture: ``architecture`` (``8.6``,
``applegpu_g13s``), ``ladder`` (``cases``, sizes per tier, ``passes``, ``routes``,
``variance``: the precondition's size, route and MEEP rank count, and ``ladder_args``
passed to every ladder unit: the build record, the binding, the site's paths) and
``fused`` (per table its DRIVE cases and its sweep: cases, resolutions, passes).
``cert8_catalog_nvidia.json`` and ``cert8_catalog_apple.json`` beside this file are the
1.0 certification round's two.

The session, in order (rule 7: one flushed line per step in ``<run>/progress.log``,
one marker per finished unit; the same command resumes it at its first unfinished
unit, and a unit is never resumed inside: an earlier output of it is set aside and the
unit runs whole, because a row is timed in ONE process under one watcher):

1. The QUIET WINDOW, before EVERY unit, read twice ``HOLD_S`` apart and recorded in the
   unit's rows: no foreign process on the device, no other work of this account (a
   harness, test or MPI process outside this session, a busy Python process, a busy
   build, compiler or bulk tool, and on Linux any busy process of the account: CPU
   shares measured between readings, :class:`CpuShares`), no
   neighbour turning busy (an idle resident process of another account on another
   GPU is recorded, never waited on); on Linux the load average below 10 % of the
   logical CPUs and available memory above 25 % of the total, the ladder's own GPU-row
   thresholds (on macOS, memory and the background daemons; the ladder gates each of
   its rows on the calibrated CPU threshold itself). On a WSL2 host, where those readers
   see the VM alone, the Windows side joins (:class:`WindowsSide`): the card at the
   declared clock lock (``--gpu-clock-lock``; with none declared, its clocks unchanged
   across the two readings), no clock-event reason but ``gpu_idle``, and Windows' busy
   CPUs outside the VM below ``--windows-cpu-max``. With ``--vm-affinity`` the session
   sets that CPU affinity mask on the WSL VM's process before its first unit, refuses
   unless it reads back, requires it at every reading and restores the earlier mask at
   its end; no MEEP is timed under it (a session with ladder units is refused, unless it
   splits them: ``--split-ladder``, step 3). A Windows side that cannot be read refuses
   the session by name before any unit; on any other host the rule is skipped by name in
   the session's gates.
2. The VARIANCE PRECONDITION, once per session that times a ladder row, and once per
   session run with ``--variance`` (a fused-only session that qualifies its host the same
   way; it refuses on NO-GO before its first unit): one point
   repeated five times on the GPU route and at MEEP's reference rank count. GO when the
   GPU route's repeat spread is within 5 %; MEEP's repeat spread is recorded and never
   gated. Its verdict is in every row of the session, and a ratio of a
   session whose precondition did not read GO (``--no-variance``) never enters a table.
   In a session that declares a VM affinity mask (step 1), the split ladder's (step 3)
   or one that qualifies a host for fused rows, it times the GPU route alone, as a GPU
   half under the mask: it gates nothing on MEEP, and MEEP is never timed under the mask.
3. The UNITS: ladder rows grouped by (case, tier) into one ladder each (its passes
   interleave its sizes, as the 1.0 ladder did; one row alone is one size), then the
   fused DRIVE units, the sweep, and the dispatch checks. Every ladder unit is passed
   ``--require-ac-power 0``: no stage is gated on power; power is
   recorded beside every row.
   THE SPLIT LADDER (``--split-ladder``, for a host whose GPU rows need a VM affinity
   mask that MEEP must not run under): each ladder unit is timed as TWO child runs of
   the ladder in this session, its GPU half (``--ranks none``: the GPU routes and the
   array control) under the declared mask and its MEEP half (``--routes none``: every
   MEEP leg, the preflight's and the digests' included) with the mask cleared (every one
   of Windows' logical CPUs, or ``--meep-vm-affinity``), at no more MEEP ranks than the
   rank blocks the session's site file declares (``RANKS`` in ``--ladder-params``, handed
   to the half as ``--meep-rank-ceiling``; a split session whose site file declares none
   is refused by name). Each half has its mask set and
   read back before its own quiet window, its own watcher, and the rule of every unit: a
   half during which its mask changed is struck, set aside and run again. The halves run
   one after the other, so a drift between them enters that unit's ratios in full: the
   split does not cancel it the way the interleave inside one ladder does. The first half
   alternates across the session's ladder units (GPU first on the first unit, MEEP first
   on the second, ...), so such a drift does not push every row's ratios the same way; a
   session of one ladder unit runs its GPU half first. A half that exits non-zero ends
   the unit (the other half is not run), which runs again, whole, on the same command.
   The two halves are composed into one row per key (``timing_rows.split_ladder_records``)
   that records the order, each half's unit, times, exit status and masks as set and as
   read back; every ratio cut from it is labelled split. A ladder row's dispatch check
   and the variance precondition run as GPU halves alone. In every other session nothing
   changes: the ladder unit is one process, as above.
   On macOS the session holds the Mac awake for its life (``caffeinate -i -s -w`` on its
   own pid, the ladder's method; :func:`keep_awake`), and says so; without
   ``caffeinate`` it says that and runs on. Power is recorded, never gated.
4. The WATCHER, every ``WATCH_S`` while a unit runs: two consecutive readings that see a
   foreign process on the device, other work of this account, or a neighbour turning
   busy INTRUDE, and so does ONE reading on which a WSL2 host's Windows side fails its
   rule (the card off its lock or under a clock-event reason, a busy Windows, a value
   that cannot be read). The unit is stopped, its output is set aside (renamed with the
   time, never deleted, never collected), and it re-runs once the window is quiet
   again; a record that carries an intrusion is refused by every cut.
5. COLLECT: each unit's rows become records in the store, with the session, the gates,
   the unit's command, environment and exit status, the reference environment's build
   record (its digest over the file as written, its facts with every path as
   ``neutral_ids`` spells it) and the in-harness import records. A ladder unit that
   exits non-zero is recorded and refused by every cut (its passes are not the plan's).
   A unit is marked
   done when it exits 0 and wrote its records; a bench unit exits 0 when its case was
   timed, reportable or struck by its floors, and its row must say so
   (:data:`TIMED_VERDICTS`). Any other unit stays unmarked, and the same command runs
   it again.

Which rows. ``--rows`` names keys; ``--owed`` takes every catalog key whose newest row is
not valid on the release tree (``timing_rows.select``): a re-time for a stale or absent
key, a dispatch check for a key owed one. With ``--environment-record`` (which ``run``
requires) a row that ran in another reference environment is stale too: an
environment is no file of the import record, so a rebuilt environment owes every row. ``timing_rows.py bill FILE`` says, before a
change lands, which keys it will re-time.

The Triton launcher's toolchain (Linux). ``run`` gives every unit ``CC`` and
``TRITON_LIBCUDA_PATH`` (``legs.launcher_toolchain``): the caller's, else the compiler and
the libcuda link of the reference environment ``--python`` runs in; a site's
``LIBCUDA_STUB`` (``--ladder-params``) and ``--bench-env`` still name another directory in
their units. A reference environment without its compiler, or without the link its build
recorded, is refused by name before the session starts: Triton would compile the timed
launchers with the host's ``gcc``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
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
#: The verdicts of a bench row whose case was timed to the end: reportable, or struck by
#: its floors with the measurement kept (the record says which floor). The bench exits 0
#: exactly when every case it was given ends in one of these; any other verdict (an
#: error, a case too large to time, a harness failure) exits 1.
TIMED_VERDICTS = ("TIMED", "TIMED-NOT-REPORTABLE")
#: What an NVIDIA fused unit's CuPy cache reads until the session mints the keep cache.
KEEP_CACHE_PENDING = "<the session's keep cache>"
#: The keep-cache mint, the ladder's own (``timing_ladder.Ladder.mint_keep_cache``).
KEEP_CACHE_CODE = ("from meep_gpu import subnormal_policy as sp\n"
                   "r = sp.point_cupy_cache_at_keep_policy(); p = r.get('provenance') or {}\n"
                   "print(r['after'] if (not p.get('reasons')) and p.get('marked') else "
                   "'REFUSED ' + repr(p.get('reasons')))")
#: The split ladder's two halves and what each child ladder is given beside the unit's own
#: command: the GPU half times no MEEP row, the MEEP half no GPU row (``timing_ladder.py``);
#: the MEEP half is also handed the host's MEEP rank ceiling (:func:`split_rank_ceiling`).
SPLIT_HALVES = {"gpu": ["--ranks", "none"], "meep": ["--routes", "none"]}
#: The order of the halves, by the split unit's place among the session's split units:
#: GPU first on the first, MEEP first on the second, and so on.
SPLIT_ORDERS = (("gpu", "meep"), ("meep", "gpu"))


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

def load_catalog(path: str, architecture: Optional[str] = None) -> Dict[str, Any]:
    try:
        return TR.load_catalog(path, architecture)
    except TR.RowRefused as refusal:
        raise Refused(str(refusal))


def variance_ranks(spec: Mapping[str, Any], physical: Optional[int] = None) -> int:
    """The precondition's MEEP rank count: a number, or ``P``, ``P/2``, ``2P/3`` of the
    host's physical cores (a portable catalog names no host's core count)."""
    value = spec["meep_ranks"]
    if isinstance(value, int) or str(value).isdigit():
        return int(value)
    match = re.fullmatch(r"(\d*)P(?:/(\d+))?", str(value).strip())
    if not match:
        raise Refused(f"variance meep_ranks {value!r}: a number, or P, P/N, KP/N")
    if physical is None:
        import timing_host  # noqa: PLC0415 - the host's own topology reader
        physical = int(timing_host.topology().get("physical_cores") or 0)
        if not physical:
            raise Refused("variance meep_ranks names the physical cores and they could "
                          "not be read on this host")
    numerator = int(match.group(1) or 1)
    denominator = int(match.group(2) or 1)
    return max(1, int(round(numerator * physical / float(denominator))))


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
    With ``environment_record`` (the reference environment the session runs in) a
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
    """One process the session runs, and the row keys it records.

    A ladder unit of the split ladder runs no process of its own: its ``children`` are its
    two halves, in the order they run, each a ladder process whose ``side`` (``gpu`` or
    ``meep``) says which half it is. A ladder row's dispatch check in a split session is a
    GPU half alone."""

    def __init__(self, name: str, kind: str, keys: Sequence[str], argv: List[str],
                 out: str, env: Mapping[str, Optional[str]], cwd: str,
                 side: Optional[str] = None, children: Sequence["Unit"] = ()):
        self.name, self.kind, self.keys = name, kind, list(keys)
        self.argv, self.out, self.env, self.cwd = argv, out, dict(env), cwd
        self.side, self.children = side, list(children)

    def describe(self) -> Dict[str, Any]:
        out = {"name": self.name, "kind": self.kind, "keys": self.keys,
               "argv": self.argv, "out": self.out, "cwd": self.cwd,
               "env": {k: v for k, v in sorted(self.env.items())}}
        if self.side is not None:
            out["side"] = self.side
        if self.children:
            out["split"] = {"order": [child.side for child in self.children],
                            "halves": [child.describe() for child in self.children]}
        return out


def _ladder_argv(args: Any, catalog: Mapping[str, Any], stamp: str, case: str,
                 tiers: List[str], sizes: Mapping[str, List[int]], passes: int,
                 extra: Sequence[str] = ()) -> List[str]:
    ladder = catalog.get("ladder") or {}
    argv = [args.python, "-u", "parity/meep_gpu/timing_ladder.py", "--stamp", stamp,
            "--python", args.python,
            "--run-root", os.path.join(args.run_root, args.session),
            "--tree-src", args.tree, "--cases", case,
            "--tiers", " ".join(tiers), "--passes", str(passes),
            # Never gated on power: power is recorded beside every row.
            "--require-ac-power", "0"]
    routes = list(ladder.get("routes") or ())
    if getattr(args, "preference_legs", False):
        # The preference on / off legs (timing_ladder.py <route>_prefon / _prefoff):
        # the catalog names them beside its routes; they run only when asked, because
        # on a package that reads no preference switch they time the same dispatch.
        routes += [r for r in ladder.get("preference_routes") or () if r not in routes]
    if routes:
        argv += ["--routes", ",".join(routes)]
    for tier, cells in sorted(sizes.items()):
        argv += [f"--{tier}-sizes" if tier != "preflight" else "--preflight-size",
                 ",".join(str(int(c)) for c in cells)]
    if args.gpu is not None:
        argv += ["--gpu", str(args.gpu)]
    if getattr(args, "ladder_params", None):
        # The site's own values (the MEEP build record, a library stub, the corpus, the
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
        # Certified hardware or nothing, as on an NVIDIA table: a Metal device, torch or
        # frontend the bound ledger does not certify is refused by name and the fused leg
        # takes the array path, so no fused row is timed on an uncertified identity. The
        # ladder's own Metal rows keep the Metal default (``timing_ladder.py``).
        env.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTORCH_ENABLE_MPS_FALLBACK": None,
                    "MEEP_GPU_HARDWARE": "certified"})
    else:
        # The ladder's NVIDIA GPU-row rules (``timing_ladder.gpu_env``): a certified
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


def split_rank_ceiling(args: Any) -> int:
    """The most MEEP ranks the MEEP half of a split ladder unit may take: the largest rank
    count of the rank blocks the session's site file declares (``RANKS`` in
    ``--ladder-params``: ``8|1|`` on an 8-performance-core desktop timed at 1 to 8 MEEP
    ranks). A split session runs on a host whose GPU rows need a mask MEEP must not run
    under, and such a host's MEEP limit is its own: the half's ladder is handed it as
    ``--meep-rank-ceiling``, so no control a catalog asks for by flag takes MEEP above it,
    and no rank block, probe or preflight row runs above it. A site file that declares no
    rank blocks (a ladder would derive them from the cores it detects, 12 under WSL2 on
    that desktop) is refused by name, and so is a split session given none."""
    import timing_ladder  # noqa: PLC0415 - the site file is the ladder's own params file

    path = getattr(args, "ladder_params", None)
    if not path:
        raise Refused("--split-ladder: the MEEP halves are timed at no more MEEP ranks than "
                      "the rank blocks the site declares (RANKS in --ladder-params, 8|1| on "
                      "an 8-performance-core desktop), and no --ladder-params was given")
    try:
        declared = (timing_ladder.read_params_file(path).get("RANKS") or "").strip()
        blocks = (timing_ladder.parse_rank_blocks(declared)
                  if declared and declared.lower() != "none" else ([], [], []))
    except OSError as error:
        raise Refused(f"--ladder-params {os.path.basename(path)}: {error.strerror}")
    except SystemExit as error:
        raise Refused(f"--ladder-params {os.path.basename(path)}: "
                      f"{getattr(error, 'refusal', error)}")
    ranks = blocks[0] + blocks[1] + blocks[2]
    if not ranks:
        raise Refused(f"--split-ladder: the MEEP halves are timed at no more MEEP ranks than "
                      f"the rank blocks the site declares, and its site file "
                      f"{os.path.basename(path)} declares none (RANKS=8|1| on an "
                      "8-performance-core desktop): a ladder would derive them from the cores "
                      "it detects")
    return max(ranks)


def build_units(args: Any, catalog: Mapping[str, Any], keys: Sequence[str]
                ) -> List[Unit]:
    """The session's units for ``keys``, in the session's order. With ``--split-ladder``
    a ladder unit holds its two halves (:class:`Unit`), its MEEP half handed the host's
    MEEP rank ceiling (:func:`split_rank_ceiling`), and a ladder row's dispatch check is a
    GPU half."""
    architecture = str(catalog["architecture"])
    ladder = catalog.get("ladder") or {}
    passes = int(ladder.get("passes") or 3)
    session_dir = os.path.join(args.run_root, args.session)
    split = bool(getattr(args, "split_ladder", False))
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
    halves_extra = dict(SPLIT_HALVES)
    if split and groups:
        halves_extra["meep"] = (["--meep-rank-ceiling", str(split_rank_ceiling(args))]
                                + SPLIT_HALVES["meep"])
    for (case, tier), cells in sorted(groups.items()):
        index += 1
        stamp = f"{args.session}_u{index:02d}"
        rows = [TR.ladder_key(architecture, case, c, tier) for c in sorted(set(cells))]
        name = f"u{index:02d}_ladder_{case}_{tier}"
        if split:
            halves = {side: Unit(f"{name}_{side}", "ladder", rows, _ladder_argv(
                args, catalog, f"{stamp}_{side}", case, ["preflight", "digests", tier],
                {tier: sorted(set(cells))}, passes, extra), os.path.join(
                    session_dir, f"ladder_{stamp}_{side}"), {}, args.tree, side=side)
                for side, extra in halves_extra.items()}
            order = SPLIT_ORDERS[(index - 1) % len(SPLIT_ORDERS)]
            units.append(Unit(name, "ladder", rows, [], os.path.join(
                session_dir, f"ladder_{stamp}"), {}, args.tree,
                children=[halves[side] for side in order]))
            continue
        argv = _ladder_argv(args, catalog, stamp, case, ["preflight", "digests", tier],
                            {tier: sorted(set(cells))}, passes)
        units.append(Unit(name, "ladder", rows, argv,
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
        side = None
        if parsed["kind"] == "ladder":
            stamp = f"{args.session}_u{index:02d}"
            # A check reads the GPU legs' launch counters alone: in a split session it is
            # a GPU half, which times no MEEP row under the mask.
            side = "gpu" if split else None
            argv = _ladder_argv(args, catalog, stamp, parsed["case"], ["preflight"],
                                {"preflight": [parsed["cells"]]}, 1,
                                SPLIT_HALVES["gpu"] if split else ())
            out = os.path.join(session_dir, f"ladder_{stamp}")
            env: Dict[str, Optional[str]] = {}
        else:
            out = os.path.join(session_dir, f"u{index:02d}_check")
            argv = _fused_argv(args, spec, parsed, out) + CHECK_BENCH_FLAGS
            env = _bench_env(args, parsed["table"])
        units.append(Unit(f"u{index:02d}_check", "check", [TR.check_key(target)], argv,
                          out, env, args.tree, side=side))
    return units


def variance_unit(args: Any, catalog: Mapping[str, Any], masked: bool = False
                  ) -> Optional[Unit]:
    """The session's variance precondition, or None when it times no ladder row.

    In a session that declares a VM affinity mask (``masked``, the session's Windows side
    applying it) or splits its ladder units (``--split-ladder``, which needs one) it is a
    GPU half (side ``gpu``): the GPU route alone (``--ranks none``), under the declared
    mask. The precondition gates the GPU route's repeat spread alone
    (:func:`variance_gate`), and MEEP is never timed under the mask, so such a session
    times no MEEP side for it. ``cert8.py plan`` takes no mask, so it plans a masked
    session's precondition only when it splits."""
    spec = (catalog.get("ladder") or {}).get("variance")
    if not spec:
        return None
    stamp = f"{args.session}_variance"
    split = bool(getattr(args, "split_ladder", False)) or masked
    ranks = SPLIT_HALVES["gpu"] if split else ["--ranks", f"{variance_ranks(spec)}||"]
    argv = _ladder_argv(args, catalog, stamp, spec["case"], ["preflight", "priority"],
                        {"priority": [int(spec["cells"])]}, VARIANCE_PASSES,
                        ["--routes", spec["route"]] + ranks + ["--controls", "none"])
    return Unit("variance", "variance", [], argv,
                os.path.join(args.run_root, args.session, f"ladder_{stamp}"), {},
                args.tree, side="gpu" if split else None)


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
    """Each process's share of one core between two readings of the box, in percent.

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


#: Other work of this account that disturbs a timing (the quiet window's "no other work of
#: this account" condition, step 1 of the module docstring): a harness script, a test run,
#: a round driver, a MEEP launch. Matched on the command.
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
    The %cpu of ``table`` is the share measured between readings (:class:`CpuShares`).

    The pids and the uid are compared here, in process; a reading is recorded, so each
    intrusion names the process by its program and the harness name that matched, never by
    its pid or its arguments (``neutral_ids.process_label``)."""
    import neutral_ids  # noqa: PLC0415
    platform = sys.platform if platform is None else platform
    out = []
    for pid, (_ppid, owner, share, command) in sorted(table.items()):
        if pid in mine or owner != uid or pid == self_pid:
            continue
        argv = command.split()
        program = os.path.basename(argv[0]) if argv else ""
        script = argv[1] if len(argv) > 1 and argv[1] != "-c" else ""
        matched = OTHER_WORK.search(script if program in ("bash", "sh", "zsh") else command)
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
        out.append(f"other work of this account ({why}): "
                   f"{neutral_ids.process_label(command, matched and matched.group())}")
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


def _recorded_neighbour(line: str, uid: int) -> str:
    """One ``<pid> <ppid> <uid> <command>`` line (``timing_host.metal_neighbours``) as a
    reading records it: ``own`` or ``other`` account, the program and the base name of the
    script it runs (the harness script ``timing_host.DEVICE_NEIGHBOURS`` matched to list
    it); no pid, uid or argument (``neutral_ids.process_label``)."""
    import neutral_ids  # noqa: PLC0415
    parts = str(line).split(None, 3)
    if len(parts) < 4:
        return neutral_ids.process_label(str(line))
    owner = "own" if parts[2] == str(uid) else "other"
    script = next((os.path.basename(word) for word in parts[3].split()[1:]
                   if word.endswith((".py", ".sh"))), None)
    return f"{owner} {neutral_ids.process_label(parts[3], script)}"


def _recorded_app(app: Mapping[str, Any], **extra: Any) -> Dict[str, Any]:
    """One compute app as a reading records it: its card id and memory, no pid or UUID."""
    import neutral_ids  # noqa: PLC0415
    return dict({"gpu": neutral_ids.card_id(app.get("gpu_uuid")),
                 "used_memory": app.get("used_memory")}, **extra)


def _app_text(app: Mapping[str, Any]) -> str:
    recorded = _recorded_app(app)
    return f"{recorded['gpu']}, {recorded['used_memory']}"


class WindowsSide:
    """The Windows side of a WSL2 host, read with every reading of the box: the quiet
    window's and the watcher's (``timing_host.windows_reading``, judged by
    ``timing_host.windows_reasons``). On any other host it is skipped, and the session's
    gates say why.

    Inside WSL2 the box's other readers see the VM alone: its load, and an ``nvidia-smi``
    that lists no process. So on WSL2 a unit also needs, at every reading, the card at
    the declared clock lock (with no lock declared: the clocks unchanged across the
    quiet window's two readings), no clock-event reason but ``gpu_idle``, and Windows'
    busy CPUs outside the VM below the threshold. While a unit runs, ONE reading that
    fails this strikes it (:class:`Watcher`): a card that left its lock, or a busy
    Windows, changed the timing whatever the next reading says.

    With a VM affinity mask declared (``--vm-affinity``), the session sets the mask on the
    VM's process before its first unit and reads it back (:meth:`set_mask`), every reading
    then requires it, and the session gives the process its earlier mask back at its end
    (:meth:`restore_mask`). MEEP is never timed under it (:meth:`Session.refuse_meep`).

    The split ladder holds a mask per unit (:meth:`hold`): the declared mask for every
    unit but a MEEP half, which runs under the MEEP mask. Every reading is then judged
    against the mask held (:meth:`judging_rule`), and the earlier mask given back at the
    end is still the one read before the session's first set."""

    def __init__(self, host: Mapping[str, Any], rule: Any = None,
                 reader: Optional[Callable[[], Dict[str, Any]]] = None,
                 gpu_uuid: Optional[str] = None, skipped: Optional[str] = None,
                 setter: Optional[Callable[[int], Dict[str, Any]]] = None):
        import timing_host  # noqa: PLC0415
        self.host = dict(host)
        self.rule = rule if rule is not None else timing_host.WindowsRule()
        self.reader = reader or (lambda: timing_host.windows_reading(gpu_uuid))
        self.setter = setter or timing_host.set_vm_affinity
        self.mask_set: Optional[Dict[str, Any]] = None
        self.mask_restored: Optional[Dict[str, Any]] = None
        #: The mask held for the unit now running (the split ladder); None: the declared.
        self.required: Optional[int] = None
        self._judge = timing_host.windows_reasons
        self._skipped = skipped
        self._last_window: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    @property
    def applied(self) -> bool:
        return bool(self.host.get("wsl2")) and self._skipped is None

    @property
    def masked(self) -> bool:
        """A VM affinity mask is declared and this host applies the rule."""
        return self.applied and self.rule.vm_affinity is not None

    def set_mask(self) -> Dict[str, Any]:
        """Set the declared mask on the WSL VM's process and read it back (once)."""
        if self.mask_set is None:
            self.mask_set = dict(self.setter(self.rule.vm_affinity))
        return self.mask_set

    def hold(self, mask: int) -> Dict[str, Any]:
        """Set ``mask`` on the WSL VM's process and read it back, for the unit about to run
        (the split ladder sets a mask before every unit); every reading from here on is
        judged against it. The first set (:meth:`set_mask`) stays the one restored from."""
        done = dict(self.setter(mask))
        self.required = mask
        return done

    def judging_rule(self) -> Any:
        """The rule a reading is judged by: the declared one, with the mask held by
        :meth:`hold` in place of the declared mask when one is held."""
        if self.required is None:
            return self.rule
        return dataclasses.replace(self.rule, vm_affinity=self.required)

    def restore_mask(self) -> Optional[Dict[str, Any]]:
        """Give the VM's process the mask it held before the set took, read back; None when
        nothing was set or the earlier mask was not read."""
        self.required = None
        done, self.mask_set = self.mask_set, None
        before = (done or {}).get("before") or ""
        if (done or {}).get("set") != "ok" or not before.startswith("0x") or not int(before, 16):
            return None
        self.mask_restored = dict(self.setter(int(before, 16)))
        return self.mask_restored

    def describe(self) -> Dict[str, Any]:
        """The rule as the session's gates record it, applied or skipped by name."""
        if not self.applied:
            out: Dict[str, Any] = {"applied": False, "skipped": self._skipped or (
                f"not a WSL2 host: {self.host.get('why')}")}
            if self.rule.locked:
                out["declared_lock_not_checked"] = self.rule.describe()["lock"]
            if self.rule.vm_affinity is not None:
                out["declared_vm_affinity_not_checked"] = f"{self.rule.vm_affinity:#x}"
            return out
        out = dict(self.rule.describe(), applied=True, host="WSL2")
        if self.mask_set is not None:
            out["vm_affinity_set"] = dict(self.mask_set)
        return out

    def unit_record(self, quiet: Sequence[Mapping[str, Any]],
                    watch: Mapping[str, Any]) -> Dict[str, Any]:
        """The rule, and on a WSL2 host what the Windows side read around one unit: the
        quiet window's two readings and the watcher's summary (the variance record)."""
        out = self.describe()
        if out["applied"]:
            out["quiet"] = [((reading.get("recorded") or {}).get("windows"))
                            for reading in quiet]
            out["watcher"] = watch.get("windows")
        return out

    def read(self, gate: bool) -> Tuple[Dict[str, Any], List[str]]:
        """One Windows reading and why it is not quiet. A quiet-window reading is compared
        with the one before it (the clocks of a card with no declared lock)."""
        observed = self.reader()
        with self._lock:
            previous = self._last_window if gate else None
            reasons = self._judge(self.judging_rule(), observed, previous)
            if gate:
                self._last_window = observed
        if self.required is not None and self.required != self.rule.vm_affinity:
            # The mask held for a MEEP half is not the declared one: the reason names it so.
            reasons = [reason.replace(f"not the declared {self.required:#x}",
                                      f"not the {self.required:#x} held for MEEP")
                       for reason in reasons]
        return observed, reasons


def windows_side(args: Any) -> WindowsSide:
    """The session's Windows side from ``--gpu-clock-lock``, ``--windows-cpu-max`` and
    ``--vm-affinity`` (or ``MGPU_TIMING_GPU_CLOCK_LOCK``, ``MGPU_TIMING_WINDOWS_CPU_MAX``
    and ``MGPU_TIMING_VM_AFFINITY``)."""
    import timing_host  # noqa: PLC0415
    lock = getattr(args, "gpu_clock_lock", None)
    cpu_max = getattr(args, "windows_cpu_max", None)
    mask = getattr(args, "vm_affinity", None)
    try:
        graphics, memory_clock = timing_host.parse_clock_lock(lock) if lock else (None, None)
        cpu_max = None if cpu_max in (None, "") else float(cpu_max)
        mask = timing_host.parse_affinity_mask(mask) if mask not in (None, "") else None
    except ValueError as error:
        raise Refused(f"--gpu-clock-lock / --windows-cpu-max / --vm-affinity: {error}")
    if cpu_max is not None and not cpu_max > 0:
        raise Refused(f"--windows-cpu-max {cpu_max}: a threshold of busy CPUs above 0")
    rule = timing_host.WindowsRule(graphics, memory_clock, cpu_max, mask)
    host = timing_host.wsl_host()
    uuid = None
    if host.get("wsl2") and getattr(args, "gpu", None) is not None:
        uuid = timing_host.as_uuid(str(args.gpu), timing_host.gpu_uuids() or {})
    return WindowsSide(host, rule, gpu_uuid=uuid)


def host_reading(gpu: Optional[str], root_pid: int, gate: bool = False,
                 markers: Sequence[str] = (),
                 windows: Optional[WindowsSide] = None) -> Dict[str, Any]:
    """One reading of the box: what would intrude, and what is only recorded. With
    ``gate`` (the quiet window before a unit) the load and memory thresholds join. On a
    WSL2 host (``windows`` applied) the Windows side joins too: recorded under
    ``recorded.windows``, and what it fails is also listed under ``decisive``.

    Every comparison (own pids, the account's uid, the pinned card's UUID) is made on
    raw values here; the reading is recorded, so it names a process by its command or,
    on a GPU, by its card id and memory, never by pid, uid or UUID."""
    import timing_host  # noqa: PLC0415 - the ladder's own readers

    table = _process_table(SHARES)
    mine = own_pids(table, root_pid, markers)
    reading: Dict[str, Any] = {"utc": utc(), "intrusions": [], "recorded": {}}
    uid = os.getuid()
    if not table:
        reading["intrusions"].append("unreadable: ps")
    reading["intrusions"] += other_work(table, mine, uid, os.getpid())
    if windows is not None and windows.applied:
        observed, failed = windows.read(gate)
        reading["recorded"]["windows"] = observed
        reading["intrusions"] += failed
        if failed:
            reading["decisive"] = failed
    memory = timing_host.memory()
    reading["recorded"]["load"] = timing_host.loadavg()
    reading["recorded"]["memory"] = memory
    if sys.platform == "darwin":
        found = timing_host.metal_neighbours()
        if found is None:
            reading["intrusions"].append("unreadable: ps")
        else:
            foreign = [line for line in found if int(line.split()[0]) not in mine]
            reading["intrusions"] += [f"another process driving the Apple GPU: "
                                      f"{_recorded_neighbour(line, uid)}"
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
            reading["intrusions"].append(f"a foreign compute process on the device: "
                                         f"{_app_text(app)}")
        elif owner == uid:
            reading["intrusions"].append(f"another GPU process of this account: "
                                         f"{_app_text(app)}")
        elif util.get(device):
            reading["intrusions"].append(f"a neighbour turned busy ({util[device]} %): "
                                         f"{_app_text(app)}")
        else:
            neighbours.append(_recorded_app(app, utilization=util.get(device)))
    reading["recorded"]["idle_neighbours"] = neighbours
    return reading


class Watcher:
    """Readings every ``every`` seconds while a unit runs; two in a row that intrude
    stop it (``on_intrusion``), and so does one reading whose intrusions it lists under
    ``decisive`` (the Windows side of a WSL2 host). The reader is injectable so the rule
    is tested without a device."""

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
        twice = bool(previous and previous.get("intrusions"))
        if twice or reading.get("decisive"):
            # One event per episode: a neighbour that stays busy for ten readings is one
            # intrusion, first seen on the reading before the second (on the reading
            # itself when one reading decides).
            if not self._episode:
                self._episode = True
                event = {"utc": reading.get("utc"),
                         "first_seen": (previous if twice else reading).get("utc"),
                         "what": sorted(set(reading["intrusions"]))}
                if reading.get("decisive"):
                    event["decisive"] = sorted(set(reading["decisive"]))
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
        # The first and the last reading, kept whole: the neighbours, load, memory
        # and power at the start and the end of every timed unit.
        block = {"every_s": self.every, "readings": len(self.readings),
                 "first": self.readings[0] if self.readings else None,
                 "last": self.readings[-1] if self.readings else None,
                 "intrusions": list(self.intrusions)}
        # On a WSL2 host, what the Windows side read through the whole unit as well.
        windows = [reading["recorded"]["windows"] for reading in self.readings
                   if isinstance(reading.get("recorded"), Mapping)
                   and reading["recorded"].get("windows")]
        if windows:
            import timing_host  # noqa: PLC0415
            block["windows"] = timing_host.windows_summary(windows)
        return block


def merge_windows_summaries(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Several runs' Windows-side summaries (``timing_host.windows_summary``) as one: the
    counts added, the values met joined, the largest of each maximum kept. The two halves
    of a split ladder unit are two runs whose row holds one watcher block."""
    out: Dict[str, Any] = {}
    for summary in summaries:
        for key, value in summary.items():
            if isinstance(value, list):
                out[key] = sorted(set(out.get(key) or ()) | set(value))
            elif key.endswith("_max"):
                known = [v for v in (out.get(key), value) if v is not None]
                out[key] = max(known) if known else None
            elif isinstance(value, int):
                out[key] = int(out.get(key) or 0) + value
            else:
                out.setdefault(key, value)
    return out


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


def variance_gate(ledger: str, meep_ranks: Optional[int], route: str) -> Dict[str, Any]:
    """The precondition's verdict: GO on the GPU route's repeat spread alone.

    ``timing_records.variance_verdict`` computes every side; here the GPU side decides,
    and MEEP's repeat spread is recorded, never gated, as its pass spread
    is everywhere else in the CERT-8 timing. ``meep_ranks`` None: no MEEP side was timed
    (the split ladder's precondition), and none is recorded."""
    import timing_records  # noqa: PLC0415

    verdict = timing_records.variance_verdict(
        ledger, meep_ranks if meep_ranks is not None else 0, route, check_ratio=False)
    if meep_ranks is None:
        verdict["sides"] = {"gpu": verdict["sides"]["gpu"]}
        verdict["meep_ranks"] = None
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

#: A path a record would still name after ``neutral_ids`` (under no root it labels: a
#: project or scratch filesystem, a mount), an account label inside it included; never the
#: rest of a labelled one (``<home>//env``: a slash right after a slash continues it).
_UNLABELLED_PATH = re.compile(r"(?:(?<![A-Za-z0-9._~<>/-])|(?<=(?<![A-Za-z0-9_])-[A-Za-z]))"
                              r"/(?:[^\s'\"<>,;:()\[\]{}=]|<account>)+")
#: What such a path is written as: the label, then its last component.
UNLABELLED_PATH_LABEL = "<path>"


def _keys_in(obj: Any) -> int:
    """How many dictionary keys ``obj`` holds, at every depth."""
    if isinstance(obj, Mapping):
        return len(obj) + sum(_keys_in(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_keys_in(v) for v in obj)
    return 0


def neutral_build_record(parsed: Any) -> Any:
    """A build record as a row record holds it: every string, a dictionary key included,
    through ``neutral_ids.value`` (a path under its root's label: ``<home>``,
    ``<staging-root>``, ``<tmp>``; the account as ``<account>``), and then any path still
    outside the system directories (a project or scratch filesystem, a mount: under no
    root ``neutral_ids`` knows) as :data:`UNLABELLED_PATH_LABEL` and its last component,
    since every such path names a site. A URL and a system directory (``/usr``,
    ``/proc``, ...) stay. Refused by name when two keys become one (two images under two
    roots with one name would keep one digest) or when a path is still left
    (``timing_rows.local_paths``)."""
    import neutral_ids  # noqa: PLC0415

    def path(match: "re.Match[str]") -> str:
        token = match.group()
        if token.startswith(("//",) + neutral_ids._SYSTEM_ROOTS):
            return token
        return f"{UNLABELLED_PATH_LABEL}/{token.rstrip('/').rsplit('/', 1)[-1]}"

    def write(node: Any) -> Any:
        if isinstance(node, str):
            return _UNLABELLED_PATH.sub(path, node)
        if isinstance(node, Mapping):
            return {write(k) if isinstance(k, str) else k: write(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [write(v) for v in node]
        return node

    written = write(neutral_ids.value(parsed))
    if _keys_in(written) != _keys_in(parsed):
        raise Refused(f"the reference environment's build record holds {_keys_in(parsed)} "
                      f"keys and {_keys_in(written)} once its paths are written under their "
                      "labels: two keys (two paths under two roots with one name) would "
                      "become one and a digest would be lost")
    left = TR.local_paths(written)
    if left:
        raise Refused(f"the reference environment's build record still names "
                      f"{len(left)} local path(s) once its paths are written under their "
                      "labels; a record leaves the machine")
    return written


def environment_block(path: Optional[str], run_dir: Optional[str] = None
                      ) -> Dict[str, Any]:
    """The reference environment's build record and the ladder's MEEP build
    records, each by name and digest: never a path, which names a machine.

    The build record's facts are kept (its flags, versions, OpenMP and FMA census, the
    digests of its images), with every path in them, a dictionary key included, under a
    label (:func:`neutral_build_record`): under its root's label (``<home>``,
    ``<staging-root>``, ``<tmp>``), the account as ``<account>``, and a path under no such
    root as ``<path>/`` and its last component. A record is read on other machines, and
    the build record names the reference environment's prefix in a dozen places. The
    digest is over the file as written, so it compares the same however the facts are
    spelled (``timing_rows.file_sha256``). ``main`` reads it once before a session starts,
    so a build record that cannot be so written is refused before anything runs."""
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
                              "record": neutral_build_record(parsed)}
    builds = os.path.join(run_dir, "meep_builds") if run_dir else None
    if builds and os.path.isdir(builds):
        for name in sorted(os.listdir(builds)):
            with open(os.path.join(builds, name), "rb") as handle:
                block["meep_builds"][name] = hashlib.sha256(handle.read()).hexdigest()
    return block


def catalog_words(catalog: Any) -> List[str]:
    """Every name a catalog holds (its keys and its identifier-like values: tables,
    cases, sizes, routes): the words its records carry beside the shared vocabulary."""
    out: List[str] = []
    if isinstance(catalog, Mapping):
        for key, value in catalog.items():
            out.append(str(key))
            out += catalog_words(value)
    elif isinstance(catalog, (list, tuple)):
        for value in catalog:
            out += catalog_words(value)
    elif isinstance(catalog, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", catalog):
        out.append(catalog)
    return out


def is_ladder_unit(unit: Unit) -> bool:
    """A ladder unit, or the dispatch check of a ladder row; every other unit is one
    run of the fused-product bench."""
    return unit.kind == "ladder" or (unit.kind == "check" and TR.parse_key(
        unit.keys[0])["target"].startswith("ladder|"))


def bench_verdict(unit: Unit) -> Optional[str]:
    """The verdict of the last row a bench unit wrote: the row its record keeps."""
    try:
        with open(os.path.join(unit.out, "rows.jsonl"), "rb") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
        return str(json.loads(lines[-1]).get("verdict")) if lines else None
    except (OSError, ValueError, AttributeError):
        return None


def unit_done(unit: Unit, rc: int, written: Sequence[str]) -> Tuple[bool, str]:
    """Whether a collected unit is finished, and why not when it is not.

    A unit is done when it exited 0 and wrote its records. A bench unit must also have
    timed its case: a row struck by its floors is a finished measurement (its record
    is kept and its gate status says why it enters no table), while an error is not,
    and the same session command runs that unit again."""
    if rc != 0 or not written:
        return False, f"rc {rc}, {len(written)} record(s) written"
    if not is_ladder_unit(unit):
        verdict = bench_verdict(unit)
        if verdict not in TIMED_VERDICTS:
            return False, (f"rc {rc}, {len(written)} record(s) written, but the bench row "
                           f"reads {verdict!r}, not a timed case")
    return True, ""


def collect(unit: Unit, *, store: str, session: Mapping[str, Any], catalog: Mapping[str, Any],
            role: str, environment: Optional[str], gates: Mapping[str, Any],
            rc: Optional[int] = None, split: Optional[Mapping[str, Any]] = None
            ) -> List[str]:
    """Write the unit's rows into the store; the paths written. The unit block carries
    its command, the environment this driver gave it (``None``: unset) and its exit
    status, which the cuts read (a ladder that stopped early is refused). A split ladder
    unit's block carries each half's (``halves``), and its rows are composed from the two
    halves' ladders with ``split`` (:meth:`Session.run_split`)."""
    import neutral_ids  # noqa: PLC0415
    architecture = str(catalog["architecture"])

    def block(of: Unit, status: Optional[int]) -> Dict[str, Any]:
        # A record leaves the machine: the command and the environment the unit was given
        # are written as ``neutral_ids`` spells them (paths under their root's label, a
        # GPU UUID as its card id), and so is every reading of the box the gates hold.
        out = {"name": of.name, "argv": neutral_ids.argv(of.argv),
               "env": neutral_ids.environment({k: v for k, v in sorted(of.env.items())}),
               "rc": status}
        if of.side is not None:
            # A half of a split ladder unit, or a split session's dispatch check (a GPU half).
            out["side"] = of.side
        return out

    unit_block = block(unit, rc)
    gates = neutral_ids.value(gates)
    written = []
    if is_ladder_unit(unit) and unit.children:
        status = {half.get("side"): half.get("rc") for half in (split or {}).get("halves") or ()}
        unit_block["halves"] = [block(child, status.get(child.side)) for child in unit.children]
        halves = {child.side: child for child in unit.children}
        records = TR.split_ladder_records(
            halves["gpu"].out, halves["meep"].out, session=session,
            architecture=architecture,
            environment=environment_block(environment, halves["meep"].out), gates=gates,
            unit=unit_block, role=role, split=neutral_ids.value(dict(split or {})))
        for record in records:
            if record["key"] in unit.keys:
                written.append(TR.write_record(store, record))
        return written
    if is_ladder_unit(unit):
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

#: How the session holds a Mac awake: the ladder's own method (``timing_ladder.py``,
#: ``keep_awake``), no idle sleep and no system sleep on AC power, for the life of the
#: process whose pid follows ``-w``.
KEEP_AWAKE = ("caffeinate", "-i", "-s", "-w")


def keep_awake(pid: int, platform: Optional[str] = None,
               which: Callable[[str], Optional[str]] = shutil.which,
               start: Callable[..., Any] = subprocess.Popen) -> Tuple[Any, str]:
    """``(process, line)``: on macOS, hold the Mac awake for the life of process ``pid``
    (:data:`KEEP_AWAKE`), and the progress line that says so. Off macOS it starts
    nothing and says nothing; on a Mac without ``caffeinate`` it starts nothing and says
    so, and the session runs on. Power is never a gate: nothing here waits or refuses.
    The line names no pid."""
    platform = sys.platform if platform is None else platform
    if not platform.startswith("darwin"):
        return None, ""
    if which(KEEP_AWAKE[0]) is None:
        return None, ("keep-awake: caffeinate is absent, so the Mac is not held awake; "
                      "the session runs on")
    process = start([*KEEP_AWAKE, str(pid)], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process, f"keep-awake: {' '.join(KEEP_AWAKE)} <this session>"


class Session:
    def __init__(self, args: Any, catalog: Mapping[str, Any],
                 reader: Optional[Callable[[], Dict[str, Any]]] = None,
                 runner: Optional[Callable[[Unit, Callable[[], bool]], int]] = None,
                 minter: Optional[Callable[[Any], str]] = None,
                 gate_reader: Optional[Callable[[], Dict[str, Any]]] = None,
                 windows: Optional[WindowsSide] = None,
                 awake: Optional[Callable[[int], Tuple[Any, str]]] = None):
        self.args, self.catalog = args, catalog
        self.minter = minter or mint_keep_cache
        # Holding a Mac awake (:func:`keep_awake`) is this host's; a session whose
        # readings are given to it is not run on the host it reads, so it holds nothing.
        if awake is None:
            awake = (keep_awake if reader is None and gate_reader is None
                     else lambda pid: (None, ""))
        self.awake = awake
        self.keepawake: Any = None
        self.dir = os.path.join(args.run_root, args.session)
        os.makedirs(os.path.join(self.dir, "markers"), exist_ok=True)
        self.log_path = os.path.join(self.dir, "progress.log")
        # The Windows side of a WSL2 host, read with both readings below; skipped by
        # name on any other host, and when the session's readings are given to it.
        if windows is None:
            windows = (windows_side(args) if reader is None and gate_reader is None
                       else WindowsSide({"wsl2": False}, skipped="the session's readings "
                                        "are given to it, not read from this host"))
        self.windows = windows
        self._windows_opened: Optional[Dict[str, Any]] = None
        # The split ladder (``--split-ladder``): the mask its MEEP halves run under (set
        # when the Windows side is opened) and the last clean attempt of a unit run.
        self.split = bool(getattr(args, "split_ladder", False))
        self.meep_mask: Optional[int] = None
        self.split_masks: Optional[Dict[str, Any]] = None
        self.attempt: Dict[str, Any] = {}
        # The watcher's reading, and the quiet window's: the window adds the load and
        # memory thresholds a running unit itself would breach. A process whose command
        # names this session's directory is this session's.
        markers = (os.path.abspath(self.dir) + os.sep,)
        self.reader = reader or (lambda: host_reading(args.gpu, os.getpid(),
                                                      markers=markers, windows=windows))
        self.gate_reader = gate_reader or reader or (
            lambda: host_reading(args.gpu, os.getpid(), gate=True, markers=markers,
                                 windows=windows))
        self.runner = runner or self.run_process
        self.child: Optional[subprocess.Popen] = None
        self.started = utc()

    def say(self, message: str) -> None:
        """One progress line, through ``neutral_ids.text``: the session's log leaves the
        machine, so a path is under its root's label and a host name is the machine id."""
        import neutral_ids  # noqa: PLC0415
        line = neutral_ids.text(f"{utc()} {message}")
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def marker(self, unit: Unit) -> str:
        return os.path.join(self.dir, "markers", f"{unit.name}.done")

    def run_process(self, unit: Unit, stopped: Callable[[], bool]) -> int:
        import neutral_ids  # noqa: PLC0415
        env = dict(os.environ)
        if unit.env:
            # A bench unit inherits no package switch: every MEEP_GPU_* the unit does not
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
            handle.write(f"# {utc()} {shlex.join(neutral_ids.argv(unit.argv))}\n")
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
        # The unit's output went straight to its log while it ran (it is timed);
        # written neutrally once it has exited.
        neutral_ids.rewrite_file(log)
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

    def open_windows(self) -> Dict[str, Any]:
        """The Windows side of a WSL2 host, checked once per session before its first
        unit: refused by name when it cannot be read, since the gate would then see the VM
        alone; skipped by name on any other host. Returns the rule as the gates record it."""
        if self._windows_opened is not None:
            return self._windows_opened
        windows = self.windows.describe()
        if windows["applied"]:
            first = self.windows.reader()
            if first.get("unreadable"):
                raise Refused("a WSL2 host whose Windows side cannot be read: "
                              f"{'; '.join(first['unreadable'])}. The gate would see the "
                              "VM alone (not Windows' load, nor the card's clocks and "
                              "clock-event reasons), so no unit runs")
            if self.windows.masked and self.split and self.meep_mask is None:
                self.meep_mask = self.split_meep_mask(first)
            if self.windows.masked:
                done = self.windows.set_mask()
                self.say(f"VM affinity: declared {done.get('declared')}, before "
                         f"{done.get('before')}, set {done.get('set')}, read back "
                         f"{done.get('read_back')}")
                if not done.get("ok"):
                    self.close_windows()
                    raise Refused(f"the WSL VM's CPU affinity could not be held at the "
                                  f"declared {done.get('declared')}: {done.get('why')} (read "
                                  f"back {done.get('read_back')}), so no unit runs")
                windows = self.windows.describe()
            lock = windows["lock"]
            self.say("WSL2 host: the Windows side joins every reading ("
                     + (f"clock lock {lock['graphics_mhz']} / {lock['memory_mhz']} MHz"
                        if lock else "no clock lock declared")
                     + f", Windows CPU max {windows['cpu_max_outside_vm']} outside the VM"
                     + (f", VM affinity {windows['vm_affinity']}"
                        if windows.get("vm_affinity") else "") + ")")
            if self.split_masks is not None:
                self.say(f"split ladder: GPU halves and every other unit under "
                         f"{self.split_masks['gpu']}, MEEP halves under "
                         f"{self.split_masks['meep']} ({self.split_masks['meep_source']}); "
                         "each set and read back before the unit's quiet window")
        else:
            self.say(f"Windows-side rule skipped: {windows['skipped']}")
        self._windows_opened = windows
        return windows

    def check_split(self) -> None:
        """``--split-ladder`` needs a VM affinity mask to split a ladder unit around, on a
        host that applies it: refused by name otherwise, before anything is set. Where one
        setting serves both sides, run without ``--split-ladder``: a ladder unit then runs
        whole, its GPU and MEEP rows interleaved in one process, which is the better
        instrument. A malformed MEEP mask is refused here too."""
        if not self.split:
            return
        if not self.windows.masked:
            why = self.windows.describe().get("skipped") or (
                "no VM affinity mask is declared (--vm-affinity)")
            raise Refused(f"--split-ladder times each ladder unit's GPU half and MEEP half "
                          f"under two VM affinity masks, and this session holds none: {why}. "
                          "Run without --split-ladder: a ladder unit then runs whole, its GPU "
                          "and MEEP rows interleaved in one process")
        declared = getattr(self.args, "meep_vm_affinity", None)
        if declared not in (None, ""):
            import timing_host  # noqa: PLC0415
            try:
                timing_host.parse_affinity_mask(declared)
            except ValueError as error:
                raise Refused(f"--meep-vm-affinity: {error}")

    def split_meep_mask(self, first: Mapping[str, Any]) -> int:
        """The mask the split ladder's MEEP halves run under: ``--meep-vm-affinity``, else
        every one of Windows' logical CPUs as the session's first Windows reading counts
        them (the mask cleared). Refused by name when that count cannot be read, or when it
        is the declared mask, under which MEEP is never timed."""
        import timing_host  # noqa: PLC0415
        declared = getattr(self.args, "meep_vm_affinity", None)
        if declared not in (None, ""):
            mask = timing_host.parse_affinity_mask(declared)
            source = "declared with --meep-vm-affinity"
        else:
            cpus = (first.get("cpu") or {}).get("logical_cpus")
            if not isinstance(cpus, int) or cpus <= 0:
                raise Refused("--split-ladder: the MEEP halves run with the VM affinity mask "
                              "cleared, every one of Windows' logical CPUs, and the first "
                              "Windows reading did not count them; declare the mask with "
                              "--meep-vm-affinity")
            mask = (1 << cpus) - 1
            source = f"cleared: every one of Windows' {cpus} logical CPUs"
        if mask == self.windows.rule.vm_affinity:
            raise Refused(f"--split-ladder: the MEEP halves' mask {mask:#x} is the declared "
                          f"VM affinity mask, and MEEP is never timed under it")
        self.split_masks = {"gpu": f"{self.windows.rule.vm_affinity:#x}",
                            "meep": f"{mask:#x}", "meep_source": source}
        return mask

    def hold_for(self, unit: Unit) -> Optional[Dict[str, Any]]:
        """The split ladder holds the mask a unit runs under before its quiet window: the
        MEEP mask for a MEEP half, the declared mask for every other unit (a GPU half, a
        fused unit, a check, the variance precondition), each set and read back, refused by
        name unless it reads back. Nothing is set in any other session (the declared mask
        is set once, by :meth:`open_windows`)."""
        if not (self.split and self.windows.masked):
            return None
        if unit.side == "meep" and self.meep_mask is None:
            raise Refused(f"{unit.name}: a MEEP half with no MEEP mask determined")
        mask = (self.meep_mask if unit.side == "meep"
                else int(self.windows.rule.vm_affinity))
        done = self.windows.hold(mask)
        self.say(f"VM affinity for {unit.name}: declared {done.get('declared')}, before "
                 f"{done.get('before')}, set {done.get('set')}, read back "
                 f"{done.get('read_back')}")
        if not done.get("ok"):
            raise Refused(f"the WSL VM's CPU affinity could not be held at "
                          f"{done.get('declared')} for {unit.name}: {done.get('why')} (read "
                          f"back {done.get('read_back')}), so no unit runs")
        return done

    def close_windows(self) -> None:
        """Give the WSL VM's process back the mask it held before this session set one,
        so no later MEEP work runs under it. A mask also dies with the VM."""
        restored = self.windows.restore_mask()
        if restored is not None:
            self.say(f"VM affinity restored: set {restored.get('set')}, read back "
                     f"{restored.get('read_back')} (declared {restored.get('declared')})")
        # A unit run after this opens the Windows side again, and sets the mask again.
        self._windows_opened = None

    def refuse_meep(self, units: Sequence[Unit]) -> None:
        """With a VM affinity mask declared, no unit that times MEEP runs under it: a whole
        ladder unit and a whole ladder row's dispatch check time MEEP beside the GPU legs in
        one ladder process, where no mask can be switched between them, and MEEP under the
        mask is slow and unsteady. Such a session is refused by name before anything is
        set, unless it splits them (``--split-ladder``): a split unit's MEEP half runs with
        the mask cleared, and its GPU half and a split session's dispatch check time no
        MEEP. The variance precondition qualifies the host under the mask: it gates the GPU
        route's repeat spread alone, so under a mask it is a GPU half
        (:func:`variance_unit`), and one that would time MEEP under the mask is refused by
        name."""
        if not self.windows.masked:
            return

        def admitted(unit: Unit) -> bool:
            if unit.kind == "variance":
                return unit.side == "gpu"
            if not is_ladder_unit(unit) or unit.side == "gpu":
                return True
            return self.split and (bool(unit.children) or unit.side == "meep")

        variance = [unit.name for unit in units if unit.kind == "variance"
                    and not admitted(unit)]
        if variance:
            raise Refused(f"the variance precondition ({variance[0]}) would time MEEP under "
                          f"the declared VM affinity mask {self.windows.rule.vm_affinity:#x}, "
                          "under which MEEP is never timed: under a declared mask it times the "
                          "GPU route alone (--ranks none), a GPU half")
        meep = [unit.name for unit in units if not admitted(unit)]
        if meep:
            raise Refused(f"a VM affinity mask is declared ({self.windows.rule.vm_affinity:#x}) "
                          f"and MEEP is never timed under it, but {len(meep)} unit(s) time "
                          f"MEEP legs beside their GPU legs in one ladder process: "
                          f"{', '.join(meep[:6])}{' ...' if len(meep) > 6 else ''}. Time them "
                          "in a session without --vm-affinity, or split each ladder unit "
                          "into a GPU half under the mask and a MEEP half without it "
                          "(--split-ladder)")

    def run_unit(self, unit: Unit) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
        """Run one unit, opened on its own quiet window and under the watcher; re-run it
        after an intrusion. Returns its exit status, the watcher's block and the window.

        A unit is never resumed inside: an output already there (an attempt of an
        earlier process, which this watcher did not see) is set aside, never deleted, and
        the unit runs whole, so every leg of a row is timed in this process's window. A
        caller that runs a unit alone (a variance precondition run on its own) has the
        session's Windows side checked first, as :meth:`run` does. In a split session the
        unit's mask is set and read back before each attempt's quiet window
        (:meth:`hold_for`), and the clean attempt is kept in ``attempt``."""
        if unit.children:
            raise Refused(f"{unit.name} is a split ladder unit: its halves run one by one "
                          "(Session.run_split), never as one process")
        self.refuse_meep([unit])
        self.open_windows()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            held = self.hold_for(unit)
            quiet = quiet_window(self.gate_reader, hold=getattr(self.args, "hold_s", HOLD_S),
                                 log=self.say)
            self.set_aside(unit, "set_aside")
            hit = threading.Event()
            watcher = Watcher(self.reader, getattr(self.args, "watch_every_s", WATCH_S),
                              log=lambda m: None,
                              on_intrusion=lambda event: (hit.set(), self.say(
                                  f"INTRUSION during {unit.name}: {event['what']}")))
            self.say(f"{unit.name}: START attempt {attempt}: keys {unit.keys}")
            started, started_utc = time.time(), utc()
            watcher.start()
            try:
                rc = self.runner(unit, hit.is_set)
            finally:
                watch = watcher.stop()
            self.say(f"{unit.name}: END rc {rc} after {int(time.time() - started)} s; "
                     f"watcher {watch['readings']} readings, "
                     f"{len(watch['intrusions'])} intrusion(s)")
            if not watch["intrusions"]:
                self.attempt = {"attempt": attempt, "started_utc": started_utc,
                                "ended_utc": utc(),
                                "elapsed_s": round(time.time() - started, 1), "held": held}
                return rc, watch, quiet
            self.set_aside(unit, "intruded")
            self.say(f"{unit.name}: re-running once the window is quiet")
        raise Refused(f"{unit.name}: intruded on {MAX_ATTEMPTS} times", EXIT_UNIT)

    def run_split(self, unit: Unit) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]],
                                             Dict[str, Any]]:
        """A split ladder unit: its two halves in the unit's order, each a run of its own
        (:meth:`run_unit`: its mask held and read back, its quiet window, its watcher, and a
        half that intrudes set aside and run again). A half that exits non-zero ends the
        unit: the other half is not run, and the unit's rows are refused and run again
        whole. Returns the unit's exit status (0 only when both halves exited 0), the
        watcher's block over the halves run (their Windows summaries merged:
        :func:`merge_windows_summaries`), their quiet windows in run order, and the
        ``split`` block every row of the unit records: the order, and per half run its unit,
        ladder, attempt, times, exit status and VM affinity as declared, as set, as read
        back and as every reading of the half saw it."""
        halves: List[Dict[str, Any]] = []
        blocks: List[Dict[str, Any]] = []
        quiet: List[Dict[str, Any]] = []
        status, not_run = 0, None
        for child in unit.children:
            if status != 0:
                not_run = {"side": child.side, "why": f"the {halves[-1]['side']} half exited "
                                                      f"rc {status}"}
                self.say(f"{unit.name}: {child.side} half not run: {not_run['why']}")
                # An earlier output of the half not run is set aside, never collected.
                self.set_aside(child, "set_aside")
                break
            rc, watch, window = self.run_unit(child)
            attempt = dict(self.attempt)
            held = attempt.pop("held", None) or {}
            # Every mask the half's readings saw: its quiet window's and its watcher's.
            seen = {((((reading.get("recorded") or {}).get("windows") or {}).get(
                "vm_affinity") or {}).get("mask")) for reading in window}
            seen |= set((watch.get("windows") or {}).get("vm_affinity") or ())
            seen.discard(None)
            halves.append(dict(
                attempt, side=child.side, unit=child.name,
                ladder=os.path.basename(child.out), rc=rc,
                vm_affinity={"declared": held.get("declared"), "before": held.get("before"),
                             "set": held.get("set"), "read_back": held.get("read_back"),
                             "read": sorted(seen)},
                quiet_utc=[reading.get("utc") for reading in window],
                watcher={"readings": watch.get("readings"),
                         "intrusions": len(watch.get("intrusions") or ()),
                         "windows": watch.get("windows")}))
            blocks.append(watch)
            quiet += window
            status = rc
            self.say(f"{unit.name}: {child.side} half rc {rc} (attempt "
                     f"{attempt.get('attempt')}, {attempt.get('elapsed_s')} s, VM affinity "
                     f"{held.get('read_back')})")
        watch = {"every_s": blocks[0].get("every_s"),
                 "readings": sum(int(block.get("readings") or 0) for block in blocks),
                 "first": blocks[0].get("first"), "last": blocks[-1].get("last"),
                 "intrusions": [event for block in blocks
                                for event in block.get("intrusions") or ()]}
        windows = [block["windows"] for block in blocks if block.get("windows")]
        if windows:
            watch["windows"] = merge_windows_summaries(windows)
        split: Dict[str, Any] = {
            "rule": "the GPU legs and the MEEP legs of the unit timed as two child runs of one "
                    "session, the GPU half under the declared VM affinity mask and the MEEP "
                    "half with the mask cleared, the first half alternating unit by unit",
            "order": [child.side for child in unit.children], "halves": halves}
        if not_run is not None:
            split["not_run"] = not_run
        return status, watch, quiet, split

    def variance_precondition(self) -> Dict[str, Any]:
        """The catalog's variance unit, opened on its quiet window and under the watcher,
        judged by :func:`variance_gate` (GO on the GPU route's repeat spread alone), with
        the Windows side of a WSL2 host (the VM affinity mask as set and as read back);
        written to ``<run>/variance.json`` (as ``neutral_ids`` spells it: the file leaves
        the machine with the session's records) and returned. A fused-only session runs
        it only with ``--variance``; a caller that qualifies a host with it alone calls
        this, then :meth:`close_windows`. Under a declared VM affinity mask it is a GPU half and records
        no MEEP side (:func:`variance_unit`)."""
        import neutral_ids  # noqa: PLC0415
        args = self.args
        unit = variance_unit(args, self.catalog, masked=self.windows.masked)
        if unit is None:
            raise Refused("the catalog names no variance precondition; pass "
                          "--no-variance WHY to time ladder rows without one")
        rc, watch, quiet = self.run_unit(unit)
        spec = self.catalog["ladder"]["variance"]
        gpu_alone = unit.side == "gpu"
        verdict = variance_gate(os.path.join(unit.out, "ladder_rows.jsonl"),
                                None if gpu_alone else variance_ranks(spec), spec["route"])
        verdict["windows"] = self.windows.unit_record(quiet, watch)
        if gpu_alone:
            verdict["meep_side"] = ("not run: under a declared VM affinity mask the "
                                    "precondition times the GPU route alone, under the mask; "
                                    "it gates the GPU route's repeat spread alone, and MEEP "
                                    "is never timed under the mask")
        with open(os.path.join(self.dir, "variance.json"), "w") as handle:
            json.dump(neutral_ids.value(verdict), handle, indent=1, sort_keys=True)
        self.say(f"variance precondition {verdict['verdict']} (rc {rc}): "
                 f"{verdict['why'] or 'GPU repeat spread within the gate'}")
        return verdict

    def run(self, units: Sequence[Unit]) -> int:
        args = self.args
        self.say(f"SESSION {args.session} START: {len(units)} unit(s), role {args.role}, "
                 f"architecture {self.catalog['architecture']}")
        if getattr(args, "variance", False) and args.no_variance:
            raise Refused("--variance runs the variance precondition and --no-variance "
                          "skips it: give one")
        self.check_split()
        self.refuse_meep(units)
        self.keepawake, line = self.awake(os.getpid())
        if line:
            self.say(line)
        try:
            return self._run(units)
        finally:
            self.close_windows()
            if self.keepawake is not None:
                self.keepawake.terminate()

    def _run(self, units: Sequence[Unit]) -> int:
        args = self.args
        gates: Dict[str, Any] = {"variance": None, "windows": self.open_windows()}
        if self.split_masks is not None:
            gates["split_ladder"] = dict(self.split_masks)
        # A session that times a ladder row runs the precondition; so does one asked to
        # with --variance, which qualifies the host for a fused-only session the same way.
        if any(unit.kind == "ladder" for unit in units) or getattr(args, "variance", False):
            if args.no_variance:
                gates["variance"] = {"ok": None, "not_run": args.no_variance}
                self.say(f"variance precondition NOT RUN: {args.no_variance}")
            else:
                verdict = self.variance_precondition()
                gates["variance"] = verdict
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
            split = None
            if unit.children:
                rc, watch, quiet, split = self.run_split(unit)
            else:
                rc, watch, quiet = self.run_unit(unit)
            # The unit's own gates: the window it opened on, the watcher while it ran,
            # power recorded and never gated.
            unit_gates = dict(gates, quiet=quiet, watcher=watch,
                              power={"gated": False,
                                     "recorded": quiet[-1].get("recorded", {}).get(
                                         "power")})
            session["ended_utc"] = utc()
            try:
                written = collect(unit, store=args.store, session=session,
                                  catalog=self.catalog, role=args.role,
                                  environment=args.environment_record, gates=unit_gates,
                                  rc=rc, split=split)
            except (TR.RowRefused, OSError, ValueError) as error:
                failed += 1
                self.say(f"unit {index}/{len(units)} {unit.name}: NOT COLLECTED: {error}")
                continue
            done, why = unit_done(unit, rc, written)
            if not done:
                # Not marked: the next run of the same command runs it again, whole (its
                # output is set aside), and what was written stays in the store.
                failed += 1
                self.say(f"unit {index}/{len(units)} {unit.name}: {why}; not marked done")
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
                       help="the reference environment's build record: every row "
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
        p.add_argument("--architecture", default=None,
                       help="with a portable catalog: the architecture this host's GPU "
                            "stamps (a newly certified GPU runs the same rows under its "
                            "own keys); the catalog's own by default")
        p.add_argument("--preference-legs", action="store_true",
                       help="add the catalog's preference_routes to every ladder unit "
                            "(the dispatch preference timed on and off)")
        p.add_argument("--split-ladder", action="store_true",
                       default=os.environ.get("MGPU_TIMING_SPLIT_LADDER") == "1",
                       help="WSL2 host with --vm-affinity: time each ladder unit as two "
                            "child runs, its GPU half under the mask and its MEEP half with "
                            "the mask cleared, the first half alternating unit by unit, "
                            "composed into one row per key whose ratios are labelled split; "
                            "a ladder row's dispatch check and the variance precondition run "
                            "as GPU halves alone. Refused where no mask applies (default: "
                            "MGPU_TIMING_SPLIT_LADDER=1)")

    plan = sub.add_parser("plan", help="the units a run would start; nothing runs")
    common(plan)
    plan.add_argument("--json", action="store_true")
    run = sub.add_parser("run", help="time the rows asked for, one session")
    common(run)
    run.add_argument("--no-variance", metavar="WHY",
                     help="time ladder rows without the precondition, and say why")
    run.add_argument("--variance", action="store_true",
                     help="run the catalog's variance precondition before the first unit "
                          "of a session that times no ladder row (a fused-only session), "
                          "and refuse on NO-GO; a session with ladder rows runs it anyway")
    run.add_argument("--hold-s", type=float, default=HOLD_S)
    run.add_argument("--watch-every-s", type=float, default=WATCH_S)
    run.add_argument("--gpu-clock-lock", metavar="GRAPHICS,MEMORY",
                     default=os.environ.get("MGPU_TIMING_GPU_CLOCK_LOCK"),
                     help="WSL2 host: the clocks in MHz the card is locked at on Windows "
                          "(nvidia-smi -lgc / -lmc), required at every reading (default "
                          "$MGPU_TIMING_GPU_CLOCK_LOCK; unset: the clocks must not move "
                          "across the quiet window's readings)")
    run.add_argument("--vm-affinity", metavar="MASK",
                     default=os.environ.get("MGPU_TIMING_VM_AFFINITY"),
                     help="WSL2 host: the CPU affinity mask (0x999988 or decimal) the WSL "
                          "VM's process must hold; the session sets it through interop "
                          "before its first unit (an elevated context is needed), reads "
                          "it back, requires it at every reading and restores the earlier "
                          "mask at its end. MEEP is never timed under it: a session with "
                          "ladder units is refused unless it splits them (--split-ladder) "
                          "(default $MGPU_TIMING_VM_AFFINITY)")
    run.add_argument("--meep-vm-affinity", metavar="MASK",
                     default=os.environ.get("MGPU_TIMING_MEEP_VM_AFFINITY"),
                     help="with --split-ladder: the mask the MEEP halves run under; every one "
                          "of Windows' logical CPUs by default (the mask cleared), never the "
                          "--vm-affinity mask (default $MGPU_TIMING_MEEP_VM_AFFINITY)")
    run.add_argument("--windows-cpu-max", metavar="CPUS",
                     default=os.environ.get("MGPU_TIMING_WINDOWS_CPU_MAX"),
                     help="WSL2 host: Windows' busy logical CPUs outside the WSL VM must "
                          "stay below this at every reading (default "
                          "$MGPU_TIMING_WINDOWS_CPU_MAX, else 10 %% of Windows' logical "
                          "CPUs)")
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
    gather.add_argument("--architecture", default=None)
    gather.add_argument("--started-utc", default=None)
    gather.add_argument("--ended-utc", default=None)
    sub.add_parser("selftest", help="the watcher's intrusion rule on stand-ins")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "selftest":
            return selftest()
        catalog = load_catalog(args.catalog, getattr(args, "architecture", None))
        import timing_host  # noqa: PLC0415 - the host-name collision every writer refuses
        if args.command != "plan" and timing_host.vocabulary_collisions(
                catalog_words(catalog)):
            raise Refused(timing_host.VOCABULARY_REFUSAL)
        if args.command == "collect":
            session = {"id": args.session, "started_utc": args.started_utc or utc(),
                       "ended_utc": args.ended_utc or utc()}
            gates = {"quiet": None, "variance": None, "watcher": None,
                     "power": {"gated": False, "recorded": None},
                     "collected_by_hand": True}
            if args.what == "ladder":
                # A ladder's keys need no DRIVE list: the fused tables are left out, as
                # for a ladder-only selection (:func:`_keys_asked`).
                keys = args.key or catalog_keys(dict(catalog, fused={}))
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
                    if unit.children:
                        print(f"  {unit.name}: {len(unit.keys)} key(s), split: "
                              f"{' then '.join(child.side or '' for child in unit.children)}")
                        for child in unit.children:
                            print(f"    {child.side} half: {shlex.join(child.argv)}")
                        continue
                    print(f"  {unit.name}: {len(unit.keys)} key(s): "
                          f"{shlex.join(unit.argv)}")
            return 0
        if not args.environment_record and not args.allow_unrecorded_environment:
            raise Refused("--environment-record is required: every row names the "
                          "reference environment it ran in")
        # Every row writes the reference build record; one it cannot write without naming
        # a machine is refused here, before anything is set or run.
        environment_block(args.environment_record)
        print(export_launcher_toolchain(args.python)["line"], flush=True)
        return Session(args, catalog).run(units)
    except Refused as refusal:
        print(f"REFUSED {refusal.message}", flush=True)
        return int(refusal.code)


if __name__ == "__main__":
    raise SystemExit(main())
