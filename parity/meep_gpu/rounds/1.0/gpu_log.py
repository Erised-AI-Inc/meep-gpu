#!/usr/bin/env python3
"""A log of the cards' memory and utilisation every 30 s, for the length of a round.

    python parity/meep_gpu/rounds/1.0/gpu_log.py start --dir "$ROUND_DIR" [--gpu-ids 0,1]
    python parity/meep_gpu/rounds/1.0/gpu_log.py peak --dir "$ROUND_DIR"
    python parity/meep_gpu/rounds/1.0/gpu_log.py stop --dir "$ROUND_DIR"

It records how much device memory the round's units reach on a card no earlier round has
measured; the log, ``<dir>/gpu_log.csv``, goes back with the results. Reading a card this
way (``nvidia-smi --query-gpu ... -l 30``) starts no compute process and allocates no
device memory, so it does not disturb the round's idle checks. ``start`` runs the reader in
a session of its own and writes its pid to ``<dir>/gpu_log.pid``; inside a batch job the
scheduler ends it with the job, so start it in the job that runs the gates. ``peak`` prints,
per card, the largest ``memory.used`` so far, the lowest reading beside it (what the card
holds with nothing of the round's on it: a desktop, on a card that drives a display) and
their difference.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

FIELDS = ("timestamp", "index", "memory.used", "memory.total", "utilization.gpu",
          "temperature.gpu", "power.draw")
HEADER = ("timestamp, index, memory.used [MiB], memory.total [MiB], utilization.gpu [%], "
          "temperature.gpu, power.draw [W]")
INTERVAL_S = 30


def reader_argv(gpu_ids: str, interval: int = INTERVAL_S) -> List[str]:
    argv = ["nvidia-smi"]
    if gpu_ids:
        argv.append(f"--id={gpu_ids}")
    return argv + [f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits",
                   "-l", str(interval)]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def peaks(text: str) -> Dict[str, Dict[str, object]]:
    """Per card: the peak ``memory.used`` (MiB), when it was read, and the lowest reading."""
    out: Dict[str, Dict[str, object]] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[2].isdigit():
            continue
        stamp, card, used = parts[0], parts[1], int(parts[2])
        row = out.setdefault(card, {"peak": used, "at": stamp, "lowest": used, "readings": 0})
        row["readings"] = int(row["readings"]) + 1
        if used > int(row["peak"]):
            row["peak"], row["at"] = used, stamp
        if used < int(row["lowest"]):
            row["lowest"] = used
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("start", "stop", "peak"))
    ap.add_argument("--dir", default=os.environ.get("ROUND_DIR", ""),
                    help="where gpu_log.csv and gpu_log.pid live (default $ROUND_DIR)")
    ap.add_argument("--gpu-ids", default="", help="the cards to read (default: every card listed)")
    args = ap.parse_args(argv)
    if not args.dir:
        ap.error("--dir (or $ROUND_DIR) is required")
    where = Path(args.dir)
    where.mkdir(parents=True, exist_ok=True)
    log, pidfile = where / "gpu_log.csv", where / "gpu_log.pid"
    pid = None
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        pass
    if args.action == "start":
        if pid is not None and alive(pid):
            print(f"already running (pid {pid}), logging to {log}")
            return 0
        if not log.exists() or log.stat().st_size == 0:
            log.write_text(HEADER + "\n", encoding="utf-8")
        with open(log, "a", encoding="utf-8") as handle:
            reader = subprocess.Popen(reader_argv(args.gpu_ids), stdout=handle,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      start_new_session=True)
        pidfile.write_text(f"{reader.pid}\n")
        print(f"logging every {INTERVAL_S} s to {log} (pid {reader.pid})")
        return 0
    if args.action == "stop":
        if pid is not None and alive(pid):
            os.kill(pid, signal.SIGTERM)
        try:
            pidfile.unlink()
        except FileNotFoundError:
            pass
        text = log.read_text(encoding="utf-8") if log.exists() else ""
        print(f"stopped; {max(len(text.splitlines()) - 1, 0)} readings in {log}")
        return 0
    text = log.read_text(encoding="utf-8") if log.exists() else ""
    found = peaks(text)
    if not found:
        print(f"no readings in {log}")
        return 1
    for card, row in sorted(found.items()):
        print(f"GPU {card}: peak memory.used {row['peak']} MiB at {row['at']}; lowest reading "
              f"{row['lowest']} MiB; difference {int(row['peak']) - int(row['lowest'])} MiB "
              f"({row['readings']} readings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
