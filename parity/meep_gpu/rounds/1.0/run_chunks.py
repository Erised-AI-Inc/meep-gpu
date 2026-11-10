#!/usr/bin/env python3
"""Run chunks of an NVIDIA chunk index one after another, each in the foreground, on the cards named.

    python parity/meep_gpu/rounds/1.0/run_chunks.py --index "$INDEX" --lanes "$LANES" \\
        --lock-dir "$LOCKS" --expect "$EXPECT" [--gpu-ids 0] [--max-locks 1] \\
        [--continue-on-fail] c8 c0 c1 ...

Run it with the reference environment's interpreter (it is the interpreter every lane
gets, ``--python``), from anywhere; it runs the round in the clone that holds this file.
For each chunk, in the order given:

* a chunk whose lanes home (``<lanes>/<chunk stamp>``) holds ``GATES_DONE`` and whose
  ``round.py status`` reads every step passing is skipped (already done);
* a chunk with a launch record (``results/chunk_<stamp>/LAUNCH.json``) runs again with
  ``--resume``: every step that already passed is skipped;
* otherwise ``legs.py check`` must end ``CHECK OK``, then the chunk runs.

The chunk runs through ``round.py run gates`` in the foreground, so the next chunk starts
only when it has finished and a batch job that runs this script holds the cards for as long
as the gates run. The capability and whether it is new come from the index (``device``,
``device_mode``): an index cut with ``chunks.py partition --new-device`` is run with
``--new-capability``, as ``legs.py check`` and ``round.py`` require for a capability the
ledgers hold no record for. ``--gpu-ids`` names the cards (as ``nvidia-smi`` indexes them;
a chunk plans one lane per card named) and ``--max-locks`` caps how many cards the lanes
under ``--lock-dir`` hold at once, across every runner sharing it. With ``--spread`` the
chunks are spread over the cards instead: each card named runs one chunk at a time, on that
card alone, and the next chunk given goes to the next card that is free, so a machine with
four cards runs four chunks side by side from one command (``--max-locks`` then defaults to
the number of cards).

By default the run stops at the first chunk that does not finish with every step passing
and says what to read; with ``--continue-on-fail`` it notes that chunk, goes on with the
next, and exits 1 at the end naming every chunk that did not pass (each unit's evidence is
its own, so one unit's failure never spoils another's run). One flushed line per event goes
to stdout; the lanes write their own logs under ``<lanes>/<chunk stamp>``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ROUND_DIR = Path(__file__).resolve().parent
#: The checkout this file belongs to (``parity/meep_gpu/rounds/1.0/run_chunks.py``).
REPO = ROUND_DIR.parents[3]

#: ``run(argv, cwd, log) -> exit status``: how a child is run (``log`` None: inherit stdout).
Runner = Callable[[Sequence[str], Path, Optional[Path]], int]


def say(text: str) -> None:
    """One line of this tool's log, written neutrally (no host name, account or local path):
    the log goes back with the results."""
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {_neutral().text(text)}",
          flush=True)


def _neutral():
    """The record spellings of ``parity/meep_gpu/neutral_ids.py`` (``round_status`` loads it)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import round_status  # noqa: PLC0415 - beside this file

    return round_status.neutral_ids()


def run_child(argv: Sequence[str], cwd: Path, log: Optional[Path]) -> int:
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    if log is None:
        return subprocess.run(list(argv), cwd=cwd, env=env).returncode
    done = subprocess.run(list(argv), cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    Path(log).write_text(_neutral().text(done.stdout.decode("utf-8", errors="replace")),
                         encoding="utf-8")
    return done.returncode


def capability_of(device: str) -> str:
    """``cc90`` -> ``9.0``, ``cc86`` -> ``8.6``: the index's device as a capability."""
    match = re.fullmatch(r"cc(\d+)(\d)", device or "")
    if not match:
        raise ValueError(f"device {device!r} is not an NVIDIA device (cc<major><minor>)")
    return f"{int(match.group(1))}.{match.group(2)}"


def read_index(path: Path) -> Tuple[Dict[str, str], str, bool]:
    """``({chunk id: stamp}, capability, new capability)`` of an NVIDIA chunk index."""
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("machine") != "nvidia":
        raise ValueError(f"{path} is a {index.get('machine')!r} index; this runner runs NVIDIA "
                         f"chunks (an Apple GPU's run through metal_round.py)")
    if index.get("device_mode") not in ("new", "certified"):
        raise ValueError(f"{path} names no device_mode (cut it with chunks.py partition)")
    stamps = {chunk["id"]: chunk["stamp"] for chunk in index.get("chunks", [])}
    return stamps, capability_of(index.get("device", "")), index["device_mode"] == "new"


class Chunks:
    def __init__(self, args: argparse.Namespace, run: Runner = run_child) -> None:
        self.args = args
        self.run = run
        self.repo = Path(args.repo).resolve()
        self.results = self.repo / "parity" / "meep_gpu" / "results"
        self.lanes = Path(args.lanes).resolve()
        self.locks = Path(args.lock_dir).resolve()
        self.python = args.python
        self.stamps, self.capability, self.new = read_index(Path(args.index))
        self.new_flag = ["--new-capability"] if self.new else []

    def round_py(self, *rest: str) -> List[str]:
        return [self.python, "-s", "-u", str(ROUND_DIR / "round.py"), *rest]

    def status(self, home: Path, out: Path) -> int:
        return self.run(self.round_py("status", "--home", str(home)), self.repo, out)

    def check(self, chunk: str) -> Tuple[bool, str]:
        out = self.lanes / f"check_{chunk}.txt"
        self.run([self.python, "-s", str(ROUND_DIR / "legs.py"), "check", "--repo", ".",
                  "--capability", self.capability, *self.new_flag, "--git-expect",
                  self.args.expect, "--chunk-index", str(Path(self.args.index).resolve()),
                  "--chunk", chunk, "--python", self.python], self.repo, out)
        try:
            lines = [x for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
        except OSError:
            lines = []
        last = lines[-1] if lines else "no output"
        return last.startswith("CHECK OK"), last

    def one(self, chunk: str, cards: str) -> Optional[bool]:
        """True: every step passing; False: not passed; None: refused before it ran."""
        stamp = self.stamps.get(chunk)
        if not stamp:
            say(f"STOP: {chunk} is not a chunk of {self.args.index}")
            return None
        home = self.lanes / stamp
        if (home / "GATES_DONE").exists():
            if self.status(home, home / "status_before.txt") == 0:
                say(f"SKIP {chunk} ({stamp}): GATES_DONE and every step passing")
                return True
        resume = (self.results / f"chunk_{stamp}" / "LAUNCH.json").exists()
        if resume:
            say(f"RESUME {chunk} ({stamp}): its launch record exists; steps that passed are "
                f"skipped")
        else:
            ok, last = self.check(chunk)
            if not ok:
                say(f"STOP: legs.py check for {chunk} reads: {last} (full text: "
                    f"{self.lanes / f'check_{chunk}.txt'})")
                return None
            say(f"CHECK OK {chunk} ({stamp})")
        say(f"START {chunk} ({stamp}) on GPU {cards}; follow {home}/round.log and "
            f"{home}/*/progress.log")
        rc = self.run(self.round_py(
            "run", "gates", "--repo", str(self.repo), "--chunk-index",
            str(Path(self.args.index).resolve()), "--chunk", chunk, "--gpu-ids", cards,
            "--capability", self.capability, *self.new_flag, "--home", str(self.lanes),
            "--lock-dir", str(self.locks), "--max-locks", str(self.args.max_locks),
            "--expect", self.args.expect, "--python", self.python,
            *(["--resume"] if resume else [])), self.repo, None)
        home.mkdir(parents=True, exist_ok=True)
        src = self.status(home, home / "status.txt")
        if rc == 0 and src == 0 and (home / "GATES_DONE").exists():
            say(f"DONE {chunk} ({stamp}): every step passing")
            return True
        say(f"NOT PASSED {chunk} ({stamp}): round.py rc {rc}, status rc {src}. Read "
            f"{home / 'status.txt'} and the failing step's row in {home}/*/steps.jsonl")
        return False

    def all(self, chunks: Sequence[str]) -> int:
        unknown = [chunk for chunk in chunks if chunk not in self.stamps]
        if unknown:
            say(f"STOP: {', '.join(unknown)} is not a chunk of {self.args.index}")
            return 2
        failed: List[str] = []
        refused: List[str] = []
        if self.args.spread:
            self.spread(chunks, failed, refused)
        else:
            for chunk in chunks:
                outcome = self.one(chunk, self.args.gpu_ids)
                if outcome is None:
                    refused.append(chunk)
                    break
                if outcome is False:
                    failed.append(chunk)
                    if not self.args.continue_on_fail:
                        say(f"STOP after {chunk}")
                        return 1
        if refused:
            return 3
        if failed and not self.args.continue_on_fail:
            return 1
        if failed:
            say(f"FINISHED WITH CHUNKS NOT PASSED: {' '.join(failed)}")
            return 1
        say(f"ALL GIVEN CHUNKS DONE: {' '.join(chunks)}")
        return 0

    def spread(self, chunks: Sequence[str], failed: List[str], refused: List[str]) -> None:
        """One worker per card, each taking the next chunk not yet started. A chunk refused
        before it ran, or one that did not pass without ``--continue-on-fail``, stops every
        card from taking another; the chunks running finish."""
        pending = list(chunks)
        lock = threading.Lock()
        halt = threading.Event()

        def work(card: str) -> None:
            while not halt.is_set():
                with lock:
                    if not pending:
                        return
                    chunk = pending.pop(0)
                outcome = self.one(chunk, card)
                with lock:
                    if outcome is None:
                        refused.append(chunk)
                        halt.set()
                    elif outcome is False:
                        failed.append(chunk)
                        if not self.args.continue_on_fail:
                            say(f"STOP after {chunk}: no card takes another chunk")
                            halt.set()

        workers = [threading.Thread(target=work, args=(card,), name=f"card {card}")
                   for card in self.args.gpu_ids.split(",")]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        failed.sort(key=list(chunks).index)


def main(argv: Optional[Sequence[str]] = None, run: Runner = run_child) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunks", nargs="+", help="chunk ids of the index (c0 c1 ...)")
    ap.add_argument("--index", required=True, help="the chunk index chunks.py partition wrote")
    ap.add_argument("--lanes", required=True, help="the lanes' home, outside the clone")
    ap.add_argument("--lock-dir", required=True, help="the directory the lanes take card locks in")
    ap.add_argument("--expect", required=True, help="the commit to run, 40 hexadecimal characters")
    ap.add_argument("--gpu-ids", default="0", help="the cards, as nvidia-smi indexes them")
    ap.add_argument("--max-locks", type=int, default=0,
                    help="how many cards the lanes sharing --lock-dir hold at once (default: "
                         "the number of cards named)")
    ap.add_argument("--spread", action="store_true",
                    help="spread the chunks over the cards: one chunk per card at a time")
    ap.add_argument("--continue-on-fail", action="store_true")
    ap.add_argument("--python", default=sys.executable,
                    help="the reference environment's interpreter (default: this one)")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expect):
        say(f"STOP: --expect is '{args.expect}', not a 40-character commit hash")
        return 2
    if not re.fullmatch(r"\d+(,\d+)*", args.gpu_ids):
        say(f"STOP: --gpu-ids is '{args.gpu_ids}', not card indices (0 or 0,1)")
        return 2
    if args.max_locks <= 0:
        args.max_locks = len(args.gpu_ids.split(","))
    try:
        runner = Chunks(args, run)
    except (OSError, ValueError, KeyError) as refused:
        say(f"STOP: {refused}")
        return 2
    runner.lanes.mkdir(parents=True, exist_ok=True)
    runner.locks.mkdir(parents=True, exist_ok=True)
    say(f"capability {runner.capability} ({'new' if runner.new else 'certified'} device), "
        f"cards {args.gpu_ids}" + (", one chunk per card at a time" if args.spread else "")
        + f", at most {args.max_locks} lock(s) under {runner.locks}")
    return runner.all(args.chunks)


if __name__ == "__main__":
    sys.exit(main())
