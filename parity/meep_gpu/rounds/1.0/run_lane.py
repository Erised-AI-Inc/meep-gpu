#!/usr/bin/env python3
"""Run one lane of the 1.0 NVIDIA certification round on one GPU: take it, run its steps, give it back.

A lane is a JSON list of steps written by ``legs.py`` (``plan lanes``/``plan route``).
Each step is one of:

* ``gate``    a CUDA gate leg (the 0.9.x CUDA-leg script's argv, caches and layout);
* ``fleet``   one run of the checkout's Triton fleet driver into a fresh root
              (``fleet_launch.py``, which only ever RAISES a timeout);
* ``stencil`` / ``bitid``  the two hand-run Triton legs, with the 0.9.x round driver's argv;
* ``hd``      one Triton H->D gate, every leg, lifts on, into ``<root>/<policy>/gate.json``
              (``legs.triton_hd_steps``), in the fleet driver's per-gate environment;
* ``route``   one leg of a route campaign (``gate_dispatch_fused_route.py``); once the gate
              has exited, the records it wrote without the record spellings (its rows,
              controls, provenance and progress log) are written through them
              (:func:`finish_route_leg`);
* ``e2e``     one leg of the end-to-end licence, in an ``env -i`` environment (a
              ``plan_only`` leg must show the default enable dispatching on the NVIDIA
              tables, as the 0.9.2 licence preflight required);
* ``e2e_verdict``  the licence's GO / NO-GO from its ship and null legs, written to
              ``VERDICT.txt`` beside them (the 0.9.2 verdict rule, unchanged);
* ``toolchain``  the card and toolchain check every lane runs first: compute
              capability, CuPy, Triton, MEEP single precision with MPI, and the
              package imported from the checkout. A refusal stops the lane.

The GPU. Without ``--gpu`` the lane takes a lock the way every lane on this host does:
it waits while ``--max-locks`` (default 4) ``gpu_lock_*`` directories exist under
``--lock-dir``, then takes a card that read idle twice ``--poll`` seconds apart (at
most ``--idle-mib`` MiB, no compute process, not in ``--avoid``), with ``mkdir`` as
the atomic claim, and writes its label and pid inside. A card that drives a display
reads idle by the display rule instead (``display_card.py``): no compute process listed,
at most :data:`display_card.UTIL_MAX_PCT` % utilisation on both readings, and on the
second at most the first reading's memory (the display resident) plus
:data:`display_card.SLACK_MIB` MiB. The lane then names the card, the display state and
the resident in its log (``DISPLAY CARD``), its lock's label and every ``steps.jsonl``
row. It writes how it took the card, by which rule, to ``<lane>/placement.json``
(``display_card.admission_record``) and hands that record to its fleet steps (the fleet
driver's ``--lane-placement``), which run on the card the lane took and ask the same rule
before every gate; the lane asks it again before every step it starts (:meth:`Lane.place_step`,
``--step-patience`` seconds of waiting, then the step reads ``refused`` and runs again on a
later launch), so every gate is placed by the one rule, not only the first. It is released by a
``finally`` and by the signal handlers, the moment the last step returns: there is no
lock keeper and no bare wait: a keeper waiting for a marker its own exit trap writes
never returns, and held a card idle for 83 minutes in a 1.0 pre-certification pass.
With ``--gpu N --no-lock`` it runs on N as told
(a caller that already holds the lock).

A shared card, by opt-in only (certification units may run beside another account's idle
process and yield the card when that user needs it; timing still needs an otherwise idle
host). ``--share-idle-foreign <uuid>:<pid>[,<pid>]``
(repeatable) and ``--share-min-free-mib M`` admit the card ``<uuid>`` although compute apps run
on it, when on BOTH readings every compute app on that card is one of the PIDs named, each
named PID reads 0 % SM utilisation (``nvidia-smi pmon``), the card reads 0 % utilisation, and
its free memory is at least ``M`` MiB (the unit's peak plus the margin the caller adds). Any
other app on the card, a named PID computing, or memory short refuses it, as an unnamed card
with an app is refused without the flag. The lane records the shared card, the named PIDs and
the free memory in its log and in its lock's label. Without the flag nothing changes. Sharing
admits a card; yielding it to the other user later is the scheduler's (a SIGTERM to the lane
stops its child process group and releases the lock, as for any signal).

Resume (AGENTS.md rule 7, and the round's runbook). Every finished step appends one
row to ``<lane>/steps.jsonl`` (fsynced) and one flushed line to
``<lane>/progress.log``. A rerun of the lane skips a step whose last row passed and
whose artifact still passes; a gate leg whose directory holds an UNRELEASED payload is
refused by name (a gate re-run into an old directory mixes two runs: move it out of
the campaign by hand, keep it, and rerun); a fleet step whose root exists runs only
its gates without a passing artifact, into the next fresh continuation root
``<root>_r2``, ``_r3`` (the Triton rebind reads several roots); a route or e2e leg
whose directory exists is never rerun (the campaign names are one-shot), and a
blocking preflight that did not pass stops the lane. ``<lane>/STOP`` stops the lane
before its next step. Markers: ``DONE`` or ``STOPPED`` with the reason.

The MEEP corpus. The CUDA H->D legs (``--lift-steps``) and the Triton H->D legs (``hd``)
re-run MEEP corpus rows. The lane resolves the root once (``--corpus-root``, else
``legs.corpus_root``'s chain), stops BEFORE taking a GPU when its plan lifts and the root
holds no ``examples``/``tests``, and exports the root as ``MEEP_GPU_CORPUS_ROOT`` to every
child.

The child environments (``RUNBOOK.md``, section 3, tabulates them). Every child but the
licence legs starts from the calling shell with every ``MEEP_GPU_*`` and ``TRIDENT_FDTD_*``
variable removed BY PREFIX (:data:`SCRUB_PREFIXES`), whatever its name, and the round's own
set on top (:meth:`Lane.base_env`): ``CUPY_ACCELERATORS`` empty, ``PYTHONNOUSERSITE=1``,
threads 1, the corpus root, and per step the policy, dispatch off and the caches. The
licence legs start from nothing (``env -i``, :meth:`Lane.licence_env`), as a user's process
does: no package switch and no ``CUPY_ACCELERATORS``, which the licence verdict refuses by
name. The user site directory is off in both: a distribution there sits on the import path
ahead of the environment's own site-packages (``environments/locks/README.md``). Before it
takes a GPU the lane starts its interpreter under each of the two environments and stops
when either still enables the user site directory (:meth:`Lane.user_site_problems`); the
card and toolchain step refuses it again on the card.

The Triton launcher's toolchain. Every child, the licence legs included, gets ``CC`` and
``TRITON_LIBCUDA_PATH`` (``legs.launcher_toolchain``, :meth:`Lane.toolchain`): the caller's
``CC``, else the reference environment's own compiler (``x86_64-conda-linux-gnu-cc`` in the
``bin`` of ``--python``'s prefix); ``--libcuda-stub`` (default: the caller's
``TRITON_LIBCUDA_PATH``), else the directory the environment's build recorded
(``share/meep-gpu-build/libcuda.json``), also first on ``LD_LIBRARY_PATH``. Without ``CC``
Triton compiles every launcher with the host's ``gcc``, outside the environment. Before it
takes a GPU the lane stops, naming it, when a reference environment lacks its compiler or
its recorded libcuda link.

It writes no ledger, fingerprint or record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import display_card  # noqa: E402
import round_status  # noqa: E402

#: The record spellings (``parity/meep_gpu/neutral_ids.py``): every log line and row this
#: lane writes goes back with the round, so it names no machine, account, card or path.
N = round_status.neutral_ids()

#: Removed from every child's environment BY PREFIX, then set per leg: the package's
#: switches, whatever their names (a fixed list missed ``MEEP_GPU_SUBNORMAL_INSTALL``), the
#: retired ``MEEP_GPU_ALLOW_UNCERTIFIED`` the package refuses by name, and the retired
#: ``TRIDENT_FDTD_*`` spellings. The round names the one it passes on
#: (``MEEP_GPU_CORPUS_ROOT``, :meth:`Lane.base_env`).
SCRUB_PREFIXES = ("MEEP_GPU_", "TRIDENT_FDTD_")
#: Removed by name, then set per leg: the two compile caches.
SCRUB_NAMES = ("CUPY_CACHE_DIR", "TRITON_CACHE_DIR")
#: The steps that do not use the GPU, and so are not placed on it (:meth:`Lane.place_step`).
GPU_FREE_KINDS = ("e2e_verdict",)
#: A route gate's own payloads, written through the record spellings by the gate itself
#: (``gate_provenance.stamp``): :func:`finish_route_leg` leaves them exactly as written.
ROUTE_STAMPED = ("summary.json", "gate.json", "plans.json")
#: The text records of a route leg :func:`finish_route_leg` reads (every other file is left).
ROUTE_RECORD_SUFFIXES = (".jsonl", ".json", ".log", ".txt")
#: What a child interpreter prints about its user site directory: ``False`` when it is off.
USER_SITE_PROBE = "import site; print(site.ENABLE_USER_SITE)"


def scrubbed(environ: Dict[str, str]) -> Dict[str, str]:
    """``environ`` without a package switch or a compile cache (:data:`SCRUB_PREFIXES`,
    :data:`SCRUB_NAMES`)."""
    return {k: v for k, v in environ.items()
            if not k.startswith(SCRUB_PREFIXES) and k not in SCRUB_NAMES}


def share_specs(values: Optional[Sequence[str]]) -> Dict[str, Set[int]]:
    """``--share-idle-foreign`` values (``<uuid>:<pid>[,<pid>...]``, repeatable) as
    ``{uuid: {pid, ...}}``; raises ValueError naming a malformed value."""
    out: Dict[str, Set[int]] = {}
    for value in values or ():
        uuid, sep, pids = str(value).rpartition(":")
        named = [p.strip() for p in pids.split(",") if p.strip()]
        if not sep or not uuid.startswith("GPU-") or not named or \
                not all(p.isdigit() for p in named):
            raise ValueError(f"--share-idle-foreign {value!r}: expected <uuid>:<pid>[,<pid>...] "
                             f"with the card's GPU-... uuid")
        out.setdefault(uuid, set()).update(int(p) for p in named)
    return out


def _finished_text(text: str, indent: Optional[int]) -> str:
    """One JSON document (a row of a ``.jsonl`` file, or a whole ``.json`` file) as the record
    spellings write it: through ``N.value`` when serializing it again reproduces ``text``
    exactly (so nothing but the strings that name a machine, account, card or local path
    changes), else ``text`` through ``N.text`` (a row cut short by a stopped child, or a
    document written in another layout)."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return N.text(text)
    if json.dumps(parsed, indent=indent, default=str) != text:
        return N.text(text)
    return json.dumps(N.value(parsed), indent=indent, default=str)


def finish_route_leg(out: Path) -> Dict[str, Any]:
    """Write the records a route gate left in ``out`` through the record spellings, once the
    gate has exited: ``{"rewritten": {file: lines changed}, "unfinished": {file: why}}``,
    each file named relative to ``out``.

    Why. ``gate_dispatch_fused_route.py`` writes ``cases.jsonl`` and ``controls.jsonl`` a row
    at a time as each lands (the progress rule), and ``plans.jsonl``, ``provenance.json`` and
    ``progress.log`` as it goes, without the record spellings; only its own payloads
    (:data:`ROUTE_STAMPED`) go through them (``gate_provenance.stamp``). A case row copies
    the plan's subnormal gate, whose ``cupy_cache_dir`` names the leg's keep cache as the
    process used it (``before``, ``after``, ``provenance.marker``): a directory under this
    lane's ``caches/``, so under the account's home directory on a host whose round runs
    there. The gate and the package it reads are in the import records of the timed fused
    rows, so the lane, which owns the leg, finishes the records it wrote, as the timed
    sessions finish the logs their children wrote directly (``neutral_ids.rewrite_file``).

    How. Every ``.jsonl`` row and every ``.json`` file is read as JSON and written back
    through ``N.value``, the spelling ``gate_provenance.stamp`` gives the same rows in
    ``summary.json`` (:func:`_finished_text`); a log line goes through ``N.text``. A file that
    comes back unchanged is not written; a changed one is replaced whole (a sibling written
    first, then ``os.replace``). The stamped payloads, symbolic links and any other file are
    left as they are, and a file that cannot be read or written is named under
    ``unfinished`` and left as it is."""
    report: Dict[str, Any] = {"rewritten": {}, "unfinished": {}}
    if not out.is_dir() or out.is_symlink():
        return report
    for top, directories, names in os.walk(out, followlinks=False):
        directories[:] = sorted(d for d in directories if d != "caches")
        for name in sorted(names):
            path = Path(top) / name
            if (name in ROUTE_STAMPED or path.suffix not in ROUTE_RECORD_SUFFIXES
                    or path.is_symlink() or not path.is_file()):
                continue
            named = path.relative_to(out).as_posix()
            try:
                before = path.read_bytes().decode("utf-8", errors="surrogateescape")
                if path.suffix == ".jsonl":
                    lines = before.split("\n")
                    after_lines = [_finished_text(line, None) if line.strip() else line
                                   for line in lines]
                elif path.suffix == ".json":
                    body = before[:-1] if before.endswith("\n") else before
                    lines = [body]
                    after_lines = [_finished_text(body, 1)]
                else:
                    lines = before.split("\n")
                    after_lines = [N.text(line) for line in lines]
                changed = sum(1 for a, b in zip(lines, after_lines) if a != b)
                if not changed:
                    continue
                after = "\n".join(after_lines)
                if path.suffix == ".json" and before.endswith("\n"):
                    after += "\n"
                staged = path.with_name(f".{name}.finishing")
                staged.write_bytes(after.encode("utf-8", errors="surrogateescape"))
                os.chmod(staged, path.stat().st_mode & 0o7777)
                os.replace(staged, path)
            except (OSError, ValueError) as exc:
                report["unfinished"][named] = f"{type(exc).__name__}: {N.text(str(exc))}"
                continue
            report["rewritten"][named] = changed
    return report


class Lane:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dir = Path(args.lane_dir).resolve()
        self.repo = Path(args.repo).resolve()
        self.harness = self.repo / "parity" / "meep_gpu"
        self.results = Path(args.results).resolve()
        self.py = args.python
        self.gpu: Optional[int] = args.gpu
        self.lock: Optional[Path] = None
        self.child: Optional[subprocess.Popen] = None
        self.t0 = time.time()
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.dir / "caches").mkdir(parents=True, exist_ok=True)
        self.progress = self.dir / "progress.log"
        self.steps_path = self.dir / "steps.jsonl"
        self.subs = json.loads(args.subs) if args.subs else {}
        import legs  # noqa: PLC0415 - beside this file; the corpus rule lives there
        self.legs = legs
        self.corpus = legs.corpus_root(getattr(args, "corpus_root", "") or "")
        self._toolchain: Optional[Dict[str, Any]] = None
        #: The latest admission reading of each shared card (``--share-idle-foreign``).
        self.share_seen: Dict[int, Dict[str, Any]] = {}
        #: Each card's latest reading (uuid, memory in use, utilisation, a listed compute
        #: process), and what the display rule saw of each card the idle rule refused there.
        self.readings: Dict[int, Dict[str, Any]] = {}
        self.display_seen: Dict[int, Dict[str, Any]] = {}
        #: The rule's latest decision for each card of the latest reading (display_card.decide).
        self.decisions: Dict[int, Dict[str, Any]] = {}
        #: The record of how this lane took its card (display_card.admission_record), written
        #: to ``<lane>/placement.json`` and handed to its fleet steps; None for a card the
        #: caller gave (``--gpu N --no-lock``) and for a card taken under the share opt-in.
        self.placement_record: Optional[Dict[str, Any]] = None
        #: The rule that holds the card before every step (display_card.Placement), set when
        #: the card is taken, or at the first step for a card the caller gave.
        self.card_rule: Optional[display_card.Placement] = None
        #: True when the card was taken under ``--share-idle-foreign``.
        self.shared_card = False
        #: The display rule's admission of the card this lane holds (None: the idle rule
        #: took it, or the card was given with ``--gpu N --no-lock``).
        self.display: Optional[Dict[str, Any]] = None
        self.wsl2 = display_card.wsl2_host()

    # ---------------------------------------------------------------- reporting ----
    def say(self, message: str) -> None:
        line = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [{self.dir.name}] "
                f"{N.text(message)}")
        print(line, flush=True)
        with self.progress.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def row(self, row: Dict[str, Any]) -> None:
        row = dict(row, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gpu=self.gpu,
                   lane=self.dir.name)
        if self.display is not None:
            row["display"] = self.display
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(N.value(row), sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def last_rows(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if self.steps_path.is_file():
            for line in self.steps_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                out[row.get("id", "")] = row
        return out

    def stop(self, why: str) -> int:
        self.say(f"STOP: {why}")
        (self.dir / "STOPPED").write_text(f"{time.strftime('%FT%TZ', time.gmtime())} {why}\n")
        return 3

    # --------------------------------------------------------------------- GPU ----
    def smi(self, *query: str) -> str:
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        try:
            return subprocess.run(["nvidia-smi", *query], capture_output=True, text=True,
                                  timeout=60, env=env).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"nvidia-smi unavailable: {exc}"

    def avoided(self) -> Set[int]:
        """The cards the caller set aside (``--avoid``)."""
        return {int(x) for x in (getattr(self.args, "avoid", "") or "").split(",")
                if x.strip().isdigit()}

    def idle_cards(self, resident: Optional[Dict[int, int]] = None) -> List[int]:
        """The cards that read idle now. ``resident`` maps a card to its memory at the
        previous reading: the display resident the display rule holds this reading to."""
        reading = self.smi("--query-gpu=index,uuid,memory.used,utilization.gpu",
                           "--format=csv,noheader,nounits")
        apps = self.smi("--query-compute-apps=gpu_uuid", "--format=csv,noheader")
        avoid = self.avoided()
        shares = share_specs(getattr(self.args, "share_idle_foreign", None))
        idle = []
        self.readings, self.display_seen, self.decisions = {}, {}, {}
        for line in reading.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            index, uuid, mib, util = int(parts[0]), parts[1], parts[2], parts[3]
            if not mib.isdigit() or not util.isdigit():
                continue
            self.readings[index] = {"uuid": uuid, "used_mib": int(mib), "util_pct": int(util),
                                    "listed": uuid in apps}
            if index not in avoid and uuid in shares and uuid in apps:
                if self.shared_card_admitted(index, uuid, shares[uuid], int(util)):
                    idle.append(index)
                continue
            # The round's one rule (display_card.decide): the idle rule first; where it
            # refuses, a display card is judged by the display rule, asked only then.
            decision = display_card.decide(
                index, self.readings[index], idle_mib=self.args.idle_mib,
                state=lambda i=index: display_card.display_state(self.smi, i, wsl2=self.wsl2),
                resident_mib=None if resident is None else resident.get(index),
                set_aside=avoid)
            self.decisions[index] = decision
            if decision["display"]:
                self.display_seen[index] = decision["display"]
            if decision["taken"]:
                idle.append(index)
        return idle

    def shared_card_admitted(self, index: int, uuid: str, named: Set[int], util: int) -> bool:
        """One reading of a card ``--share-idle-foreign`` names (module docstring, A SHARED
        CARD): True when every compute app on it is a named PID, each named PID and the card
        read 0 % utilisation, and its free memory is at least ``--share-min-free-mib``. The
        reading, and why it was refused, are kept in :attr:`share_seen`."""
        pids: Dict[int, str] = {}
        for line in self.smi("--query-compute-apps=gpu_uuid,pid,used_memory",
                             "--format=csv,noheader,nounits").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] == uuid and parts[1].isdigit():
                pids[int(parts[1])] = parts[2] if len(parts) > 2 else "?"
        sm: Dict[int, int] = {}
        for line in self.smi("pmon", "-i", str(index), "-c", "1", "-s", "u").splitlines():
            cols = line.split()
            if line.startswith("#") or len(cols) < 4 or not cols[1].isdigit():
                continue
            sm[int(cols[1])] = int(cols[3]) if cols[3].isdigit() else 0
        free = None
        for line in self.smi("--query-gpu=index,memory.free",
                             "--format=csv,noheader,nounits").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(index) and parts[1].isdigit():
                free = int(parts[1])
        reasons = []
        unnamed = sorted(set(pids) - named)
        # Counted, never listed: the reasons reach the lane's log.
        if unnamed:
            reasons.append(f"{len(unnamed)} compute app(s) not named by --share-idle-foreign")
        busy = sorted(pid for pid in pids if pid in named and sm.get(pid, 0) > 0)
        if busy:
            reasons.append(f"{len(busy)} named compute app(s) computing (at most "
                           f"{max(sm[pid] for pid in busy)} % SM)")
        if util != 0:
            reasons.append(f"card utilisation {util} %")
        need = int(getattr(self.args, "share_min_free_mib", 0) or 0)
        if free is None or free < need:
            reasons.append(f"free memory {free} MiB below {need} MiB")
        self.share_seen[index] = {"uuid": uuid, "pids": {pid: pids[pid] for pid in sorted(pids)},
                                  "sm": {pid: sm.get(pid, 0) for pid in sorted(pids)},
                                  "free_mib": free, "need_mib": need, "refused": reasons}
        return not reasons

    def display_admission(self, index: int, first_readings: Dict[int, Dict[str, Any]],
                          first_seen: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The record of a card taken under the display rule on either of its two readings,
        or None when the idle rule took it on both."""
        seen = [s for s in (first_seen.get(index), self.display_seen.get(index)) if s]
        if not any(not s["refused"] for s in seen):
            return None
        readings = [first_readings[index], self.readings[index]]
        return {"rule": "display", "card": N.card_id(readings[-1]["uuid"]),
                "state": seen[-1]["state"],
                "resident_mib": readings[0]["used_mib"],
                "readings": [{k: r[k] for k in ("used_mib", "util_pct")} for r in readings],
                "slack_mib": display_card.SLACK_MIB, "util_max_pct": display_card.UTIL_MAX_PCT}

    def not_idle(self) -> str:
        """Why each card of the latest reading was not taken, briefly."""
        out = []
        for index, r in sorted(self.readings.items()):
            decision = self.decisions.get(index)
            shared = self.share_seen.get(index)
            if decision is not None:
                if decision["taken"]:
                    continue
                why = "; ".join(decision["refused"])
            elif shared is not None:
                if not shared.get("refused"):
                    continue
                why = f"card {index}: " + "; ".join(shared["refused"])
            else:
                continue
            out.append(why)
        return " | ".join(out) or "every card read idle on one reading but not on both"

    def nlocks(self) -> int:
        return len(list(Path(self.args.lock_dir).glob("gpu_lock_*")))

    def take_gpu(self) -> bool:
        lock_dir = Path(self.args.lock_dir)
        lock_dir.mkdir(parents=True, exist_ok=True)
        waited, wanted = 0, self.args.want_gpu
        while True:
            if (self.dir / "STOP").exists():
                return False
            if self.nlocks() < self.args.max_locks:
                first = set(self.idle_cards())
                first_readings, first_seen = dict(self.readings), dict(self.display_seen)
                time.sleep(self.args.poll)
                both = sorted(first & set(self.idle_cards(
                    resident={i: first_readings[i]["used_mib"] for i in first})))
                if wanted is not None:
                    both = [i for i in both if i == wanted]
                if self.nlocks() < self.args.max_locks:
                    for index in both:
                        path = lock_dir / f"gpu_lock_{index}"
                        try:
                            path.mkdir()
                        except FileExistsError:
                            continue
                        shared = self.share_seen.get(index)
                        note = (f" shared={shared['uuid']}:"
                                f"{','.join(str(p) for p in shared['pids'])}"
                                if shared else "")
                        display = self.display_admission(index, first_readings, first_seen)
                        if display:
                            note += f" display_resident={display['resident_mib']}MiB"
                        # The lock is LOCAL: its label, pid and the card's UUID stay here
                        # and are never copied into a record (the round's logs and rows
                        # name the card by its id, N.card_id).
                        (path / "label").write_text(f"{self.args.label} lane={self.dir.name} "
                                                    f"host={os.uname().nodename}{note}\n")
                        (path / "pid").write_text(f"{os.getpid()}\n")
                        uuid = (self.readings.get(index) or {}).get("uuid", "")
                        (path / "card").write_text(f"{N.card_id(uuid)} {uuid}\n")
                        self.lock, self.gpu = path, index
                        self.say(f"took gpu_lock_{index} (idle on two readings "
                                 f"{self.args.poll} s apart; {self.nlocks()} lock(s) now held "
                                 f"on the account)")
                        if shared:
                            # Counted, never listed: a foreign process is another account's.
                            mib = sum(int(v) for v in shared["pids"].values()
                                      if str(v).isdigit())
                            busy = sum(1 for v in shared["sm"].values() if v)
                            self.say(f"SHARED CARD {index} ({N.card_id(shared['uuid'])}): "
                                     f"{len(shared['pids'])} foreign compute app(s), {mib} MiB, "
                                     f"named by --share-idle-foreign, {busy} computing on "
                                     f"both readings; free {shared['free_mib']} MiB >= "
                                     f"{shared['need_mib']} MiB")
                        self.record_placement(index, first_readings, shared, display)
                        if display:
                            self.display = display
                            used = [r["used_mib"] for r in display["readings"]]
                            util = [r["util_pct"] for r in display["readings"]]
                            self.say(f"DISPLAY CARD {index} ({display['card']}): "
                                     f"{display_card.describe(display['state'])}; taken under "
                                     f"the display rule on two readings: {used} MiB in use "
                                     f"(display resident {display['resident_mib']} MiB + "
                                     f"{display['slack_mib']} MiB), {util} % utilisation (at "
                                     f"most {display['util_max_pct']} %), no compute process "
                                     f"listed")
                        return True
            if waited % 1800 == 0:
                self.say(f"waiting for a GPU: {self.nlocks()} lock(s) held "
                         f"(cap {self.args.max_locks}), waited {waited} s"
                         + (f"; {self.not_idle()}" if self.readings else ""))
            time.sleep(self.args.poll)
            waited += 2 * self.args.poll
            if waited > self.args.lock_patience:
                return False

    def release(self) -> None:
        if self.lock is not None and self.lock.exists():
            for name in ("label", "pid", "card"):
                try:
                    (self.lock / name).unlink()
                except OSError:
                    pass
            try:
                self.lock.rmdir()
            except OSError:
                pass
            self.say(f"released {self.lock.name} (held {int(time.time() - self.t0)} s)")
        self.lock = None

    # ------------------------------------------------------------------ placement ----
    def record_placement(self, index: int, first_readings: Dict[int, Dict[str, Any]],
                         shared: Optional[Dict[str, Any]],
                         display: Optional[Dict[str, Any]]) -> None:
        """Write how the card was taken (``<lane>/placement.json``, display_card.admission_record)
        and hold it to the rule that took it for every later step (:meth:`place_step`)."""
        rule = "share" if shared else "display" if display else "idle"
        self.shared_card = bool(shared)
        record = display_card.admission_record(
            index, rule, idle_mib=self.args.idle_mib,
            readings=[first_readings[index], self.readings[index]],
            display=None if not display else {"state": display["state"],
                                              "resident_mib": display["resident_mib"]},
            lane=self.dir.name)
        (self.dir / "placement.json").write_text(
            json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self.placement_record = record
        self.card_rule = display_card.Placement(
            self.smi, idle_mib=self.args.idle_mib, set_aside=self.avoided(), wsl2=self.wsl2,
            say=self.say)
        self.card_rule.adopt(record, index)

    def card_still_taken(self) -> Tuple[bool, str]:
        """Does the round's rule still take the lane's card, on one reading now? A card taken
        under the share opt-in is asked the share admission again; a card the caller gave
        (``--gpu N --no-lock``) is taken, at the first step, by this lane's own first reading."""
        if self.shared_card:
            taken = self.gpu in self.idle_cards()
            return taken, "" if taken else self.not_idle()
        if self.card_rule is None:
            self.card_rule = display_card.Placement(
                self.smi, idle_mib=self.args.idle_mib, set_aside=self.avoided(),
                wsl2=self.wsl2, say=self.say)
        return self.card_rule.check(self.gpu)

    def step_wait(self) -> Tuple[int, int]:
        """``(patience, poll)``, seconds: how long a step waits for its card and how often it
        reads it meanwhile (``--step-patience``; ``display_card.RECHECK_S``, or ``--poll`` when
        that is shorter). The fleet driver waits for each gate by the same two numbers."""
        return (int(getattr(self.args, "step_patience", 300)),
                max(1, min(display_card.RECHECK_S, int(self.args.poll))))

    def place_step(self, step: Dict[str, Any]) -> Optional[str]:
        """Place one step on the lane's card: None when the rule takes the card, else why not,
        after waiting up to ``--step-patience`` seconds for it. Every step that uses the GPU is
        placed, not only the first: a process that arrives on the card between two steps stops
        the next one rather than sharing the card with it. A card the rule refuses is waited
        for, never shared; the step then reads ``refused`` and runs again on a later launch."""
        if step["kind"] in GPU_FREE_KINDS or self.gpu is None:
            return None
        patience, step_poll = self.step_wait()
        waited = 0
        while True:
            taken, why = self.card_still_taken()
            if taken:
                return None
            if waited >= patience or (self.dir / "STOP").exists():
                return why
            if waited == 0:
                self.say(f"{step['id']}: the card is not free: {why}; waiting up to "
                         f"{patience} s")
            time.sleep(step_poll)
            waited += step_poll

    # --------------------------------------------------------------- environment ----
    def toolchain(self) -> Dict[str, Any]:
        """The Triton launcher's toolchain every child gets (``legs.launcher_toolchain``),
        resolved once: the caller's ``CC`` and ``--libcuda-stub`` first, else the reference
        environment of ``--python``'s."""
        if self._toolchain is None:
            prefix = (self.legs.interpreter_prefix(self.py)
                      if sys.platform.startswith(self.legs.LAUNCHER_PLATFORMS) else None)
            self._toolchain = self.legs.launcher_toolchain(prefix, dict(os.environ),
                                                           self.args.libcuda_stub or "")
        return self._toolchain

    def base_env(self) -> Dict[str, str]:
        """Every child but the licence legs: the calling shell, scrubbed by prefix
        (:func:`scrubbed`), with the round's own variables on top."""
        env = scrubbed(dict(os.environ))
        env.update({"CUDA_VISIBLE_DEVICES": str(self.gpu), "CUPY_ACCELERATORS": "",
                    "PYTHONPATH": f"{self.repo}:{self.harness}", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE",
                    "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
        self.legs.launcher_environment(env, self.toolchain())
        # The corpus is named, not inherited by accident: the lifting gates read this
        # variable first, so every child reads the root this lane checked.
        if all((Path(self.corpus) / name).is_dir() for name in self.legs.CORPUS_DIRECTORIES):
            env[self.legs.CORPUS_VARIABLE] = self.corpus
        return env

    def licence_env(self, leg: str, out: Path) -> Dict[str, str]:
        """A licence leg's environment, from nothing (``env -i``): what a user's process
        holds, plus the leg's caches and dispatch log. No ``MEEP_GPU_*`` switch but the
        log, and no ``CUPY_ACCELERATORS``: the licence verdict refuses both by name
        (``recut_driver_dispatch_record.LICENCE_ENVIRONMENT_REFUSED_*``)."""
        caches = self.dir / "caches"
        env = {"HOME": os.environ.get("HOME", ""),
               "PATH": f"{Path(self.py).parent}:/usr/local/bin:/usr/bin:/bin",
               "USER": os.environ.get("USER", ""), "LOGNAME": os.environ.get("LOGNAME", ""),
               "LANG": os.environ.get("LANG", "C.UTF-8"), "CUDA_VISIBLE_DEVICES": str(self.gpu),
               "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "NUMEXPR_NUM_THREADS": "1", "PYTHONPATH": str(self.repo),
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
               "CUPY_CACHE_DIR": str(caches / f"cupy_cache_e2e_{leg}"),
               "TRITON_CACHE_DIR": str(caches / f"triton_cache_e2e_{leg}"),
               "MEEP_GPU_DISPATCH_LOG": str(out.parent / f"{leg}_dispatch_log.jsonl")}
        for name in ("CUDA_PATH", "CUDA_HOME"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        return self.legs.launcher_environment(env, self.toolchain())

    def user_site_problems(self) -> List[str]:
        """Why a child would import from the user site directory: the lane's interpreter,
        started under the gate environment and under the licence environment, must report
        ``site.ENABLE_USER_SITE`` False. Run before a GPU is taken."""
        problems = []
        for name, env in (("gate", self.base_env()),
                          ("licence", self.licence_env("user_site_probe",
                                                       self.dir / "user_site_probe"))):
            try:
                done = subprocess.run([self.py, "-c", USER_SITE_PROBE], cwd=str(self.dir),
                                      env=env, capture_output=True, text=True, timeout=300)
                answer = (done.stdout.strip().splitlines() or [""])[-1]
                why = answer or (done.stderr.strip().splitlines() or [f"rc {done.returncode}"])[-1]
            except (OSError, subprocess.SubprocessError) as exc:
                answer, why = "", f"{type(exc).__name__}: {exc}"
            if answer != "False":
                problems.append(f"the {name} children's interpreter {self.py} enables the user "
                                f"site directory with PYTHONNOUSERSITE=1 (site.ENABLE_USER_SITE: "
                                f"{why[:200]}); a distribution there would sit ahead of the "
                                f"environment's own")
        return problems

    def sub(self, value: Any) -> Any:
        if isinstance(value, str):
            for key, rep in self.subs.items():
                value = value.replace(key, rep)
            return value
        if isinstance(value, list):
            return [self.sub(v) for v in value]
        if isinstance(value, dict):
            return {k: self.sub(v) for k, v in value.items()}
        return value

    # ---------------------------------------------------------------- one child ----
    def run_child(self, argv: List[str], env: Dict[str, str], log: Path, timeout: int,
                  label: str) -> Dict[str, Any]:
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open("w", encoding="utf-8") as handle:
            handle.write(N.text(f"# {time.strftime('%FT%TZ', time.gmtime())} {label}\n# argv: "
                                f"{' '.join(shlex.quote(a) for a in argv)}\n"))
            handle.flush()
            # The child's output reaches the step log through N.copy_lines, written
            # neutrally line by line: a gate prints its paths, and the log goes back.
            self.child = subprocess.Popen(argv, cwd=str(self.repo), env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            copier = N.copy_lines(self.child.stdout, handle)
            timed_out, beat = False, started + self.args.heartbeat
            while True:
                try:
                    self.child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.time()
                if now - started > timeout:
                    timed_out = True
                    self._kill_child()
                    break
                if now >= beat:
                    beat = now + self.args.heartbeat
                    self.say(f"[beat] {label} {int(now - started)} s | {_last_line(log)[:160]}")
            copier.join(timeout=120)
        rc = None if timed_out else self.child.returncode
        self.child = None
        return {"rc": rc, "timed_out": timed_out, "elapsed_s": round(time.time() - started, 1)}

    def _kill_child(self) -> None:
        if self.child is None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
            self.child.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except OSError:
                pass
            self.child.wait()

    # ------------------------------------------------------------------- steps ----
    def timeout_for(self, step: Dict[str, Any]) -> int:
        return int(step.get("timeout_s") or max(self.args.min_timeout,
                                                self.args.timeout_factor * step["expect_s"]))

    def step_gate(self, step: Dict[str, Any]) -> Dict[str, Any]:
        artifact = Path(step["artifact"])
        leg_dir = Path(step["leg_dir"])
        before = round_status.classify(artifact)
        if round_status.is_pass(before["state"]):
            return {"state": before["state"], "skipped": "artifact already passes", **before}
        if leg_dir.is_dir() and any(leg_dir.glob("*.json")):
            return {"state": "refused", "reasons": [
                f"{leg_dir} already holds a payload ({before['state']}); a gate re-run into an "
                f"old directory mixes two runs. Move it OUT of the campaign by hand (keep it) "
                f"and rerun the lane"]}
        leg_dir.mkdir(parents=True, exist_ok=True)
        if step.get("subjects_file") and not Path(step["subjects_file"]).is_file():
            # The ADE re-cut's subject manifest, written before its first leg compiles:
            # record_cuda_ade_recut.py re-hashes these files and refuses undeclared drift.
            lines = []
            for rel in step["subjects"]:
                lines.append(f"{hashlib.sha256((self.repo / rel).read_bytes()).hexdigest()}  {rel}")
            Path(step["subjects_file"]).write_text("\n".join(lines) + "\n")
            self.say(f"ADE subject manifest: {len(lines)} files -> {step['subjects_file']}")
        tag = step["id"].replace("/", "__")
        token = "cupy_ftz_stripped" if step["policy"] == "keep" else "cupy_plain"
        env = self.base_env()
        env.update({"MEEP_GPU_DISPATCH": "0", "MEEP_GPU_BACKEND_PREFERENCE": "cuda",
                    "CUPY_CACHE_DIR": str(self.dir / "caches" / f"{token}_{tag}"),
                    "TRITON_CACHE_DIR": str(self.dir / "caches" / f"triton_{tag}")})
        if step.get("wrap"):
            # Run from the LANE copy, outside the checkout, so gate_provenance never
            # records the wrapper among the gate's imports.
            wrapper = self.dir / "meep_host_import.py"
            if not wrapper.is_file():
                wrapper.write_text((HERE / "meep_host_import.py").read_text())
            argv = [self.py, "-u", str(wrapper), str(self.repo), step["script"], *step["args"]]
        else:
            argv = [self.py, "-u", step["script"], *step["args"]]
        log = self.dir / "logs" / f"{tag}.log"
        result = self.run_child(argv, env, log, self.timeout_for(step), step["id"])
        try:
            (leg_dir / "run.log").write_text(log.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
        return {**result, **round_status.classify(artifact), "log": str(log)}

    def fleet_roots(self, root: Path) -> List[Path]:
        roots = [root] + sorted(root.parent.glob(f"{root.name}_r[0-9]*"))
        return [r for r in roots if r.is_dir()]

    def step_fleet(self, step: Dict[str, Any]) -> Dict[str, Any]:
        root = Path(step["root"])
        existing = self.fleet_roots(root)
        gates = list(step["gates"])
        if existing:
            gates = round_status.remaining_gates(existing, gates)
            if not gates:
                return {"state": "released", "skipped": "every gate passes in "
                        + ", ".join(r.name for r in existing)}
            root = root.parent / f"{root.name}_r{len(existing) + 1}"
            self.say(f"{step['id']}: {len(step['gates']) - len(gates)} of {len(step['gates'])} "
                     f"gates already pass; the {len(gates)} left run into {root.name}")
        if root.exists():
            return {"state": "refused", "reasons": [f"{root} exists"]}
        env = self.base_env()
        env.update({k: v for k, v in step.get("env", {}).items() if v})
        if step.get("manifest"):
            root.mkdir(parents=True)
            manifest = subprocess.run(
                "find meep_gpu parity/meep_gpu -name '*.py' -not -path 'parity/meep_gpu/results/*' "
                "-not -path '*/__pycache__/*' -not -name '._*' | LC_ALL=C sort | xargs sha256sum",
                shell=True, cwd=str(self.repo), capture_output=True, text=True)
            (root / "staged_source_sha256.txt").write_text(manifest.stdout)
        argv = [self.py, "-u", str(HERE / "fleet_launch.py"), "--repo", str(self.repo)]
        for item in self.args.timeout_floor:
            argv += ["--timeout-floor", item]
        if step["policy"] == "flush":
            # The flush host half needs MEEP imported first; the wrapper is the lane's
            # copy, outside the checkout (fleet_launch.py).
            wrapper = self.dir / "meep_host_import.py"
            if not wrapper.is_file():
                wrapper.write_text((HERE / "meep_host_import.py").read_text())
            argv += ["--host-import-wrapper", str(wrapper)]
        patience, poll = self.step_wait()
        argv += ["--", "--gpu", str(self.gpu), "--forbidden-gpus",
                 ",".join(str(i) for i in sorted(self.avoided())), "--policy",
                 step["policy"], "--patience", str(patience), "--poll", str(poll),
                 "--out-root", str(root), "--gates", ",".join(gates)]
        if self.placement_record is not None:
            # The fleet driver runs on the card this lane took and holds every gate to the
            # rule that took it (drive_triton_weld_gates --lane-placement); it records the
            # lane's decision beside each gate's own check.
            argv += ["--lane-placement", str(self.dir / "placement.json")]
        # The fleet driver sets CUDA_VISIBLE_DEVICES per gate from --gpu (a PHYSICAL
        # index) and reads nvidia-smi itself, so the mask is not exported to it.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        log = self.dir / "logs" / f"{step['id'].replace('/', '__')}_{root.name}.log"
        timeout = sum(self.args.timeout_factor * int(e) for e in
                      [step["expect_s"]]) + 3600 * len(gates)
        result = self.run_child(argv, env, log, timeout, step["id"])
        rows = []
        for r in self.fleet_roots(Path(step["root"])):
            rows += round_status.fleet_root(r)
        wanted = set(step["gates"])
        rows = [r for r in rows if r["gate"] in wanted]
        passing = {r["gate"] for r in rows if round_status.is_pass(r["state"])}
        failing = sorted(wanted - passing)
        return {**result, "state": "released" if not failing else "not_released",
                "passing": f"{len(passing)} of {len(wanted)}", "root": str(root),
                "reasons": [f"not passing: {failing}"] if failing else [], "log": str(log)}

    def step_hand(self, step: Dict[str, Any]) -> Dict[str, Any]:
        root = Path(step["root"])
        policy = step["policy"]
        caches = self.dir / "caches"
        token = "ftz_stripped" if policy == "keep" else "plain"
        env = self.base_env()
        env["MEEP_GPU_DISPATCH"] = "0"
        mint = None
        if step["kind"] == "hd":
            # A Triton H->D gate, every leg, lifts on, into <root>/<policy>/gate.json. Its
            # environment is the fleet driver's for every Triton gate
            # (drive_triton_weld_gates.gate_environment): the policy named, dispatch off,
            # the certified hardware scope, a private CuPy cache whose name carries the
            # policy token (a keep cache minted by the package first) and a fresh Triton
            # cache.
            gate = step["gate"]
            artifact = root / policy / "gate.json"
            argv = [self.py, "-u", str(self.harness / f"gate_triton_{gate}.py"),
                    "--subnormal-policy", policy, "--lift-timeout",
                    str(int(step.get("lift_timeout") or 3600)), "--out", str(artifact)]
            if policy == "flush":
                argv.append("--import-meep-for-host-policy")
            env.update({"MEEP_GPU_SUBNORMAL_POLICY": policy, "MEEP_GPU_HARDWARE": "certified",
                        "CUPY_CACHE_DIR": str(caches / f"cupy_cache_{token}_hd_{gate}"),
                        "TRITON_CACHE_DIR": str(caches / f"triton_cache_hd_{gate}")})
            if policy == "keep":
                mint = Path(env["CUPY_CACHE_DIR"])
        elif step["kind"] == "bitid":
            artifact = root / "gate.json"
            argv = [self.py, "-u", str(self.harness / "probe_fused_kernel_bit_identity.py"),
                    "--module", str(self.harness / "track_triton_pml.py"), "--track", "triton",
                    "--experiments", step.get("experiments") or
                    "pml,multistep,constitutive,constitutive_multistep,fused_pair,"
                    "fused_pair_multistep,wholestep", "--subnormal-policy", policy,
                    "--out", str(artifact)]
            if policy == "flush":
                argv.append("--import-meep-for-host-policy")
            env.update({"CUPY_CACHE_DIR": str(caches / f"cupy_cache_{token}_bit_identity"),
                        "TRITON_CACHE_DIR": str(caches / "triton_cache_bit_identity")})
        else:
            fam = step["family"]
            artifact = root / fam / "gate.json"
            gate = [str(self.harness / "gate_triton_offdiag_stencil_welds.py"), "--family", fam,
                    "--subnormal-policy", policy, "--out-root", str(root)]
            if policy == "flush":
                # The stencil gate takes no --import-meep-for-host-policy; the flush host
                # half is reached by importing MEEP first, from the lane's copy of the
                # wrapper (outside the checkout, so no weld records it).
                wrapper = self.dir / "meep_host_import.py"
                if not wrapper.is_file():
                    wrapper.write_text((HERE / "meep_host_import.py").read_text())
                argv = [self.py, "-u", str(wrapper), str(self.repo), *gate]
            else:
                argv = [self.py, "-u", *gate]
            env.update({"CUPY_CACHE_DIR": str(caches / f"cupy_cache_{token}_stencil_{fam}"),
                        "TRITON_CACHE_DIR": str(caches / f"triton_cache_stencil_{fam}")})
        before = round_status.classify(artifact)
        if round_status.is_pass(before["state"]):
            return {"state": before["state"], "skipped": "artifact already passes", **before}
        if artifact.exists():
            return {"state": "refused", "reasons": [f"{artifact} exists unreleased; move it out "
                                                    f"by hand (keep it) and rerun"]}
        if step["kind"] == "hd" and artifact.parent.is_dir() and any(artifact.parent.iterdir()):
            # A gate re-run into a directory an earlier run wrote mixes two runs.
            return {"state": "refused", "reasons": [f"{artifact.parent} holds an earlier run's "
                                                    f"files; move it out by hand (keep it) and "
                                                    f"rerun"]}
        root.mkdir(parents=True, exist_ok=True)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if mint is not None:
            why = self.mint_keep_cache(mint)
            if why:
                return {"state": "refused", "reasons": [f"the keep CuPy cache {mint} could not "
                                                        f"be minted: {why}"]}
        log = self.dir / "logs" / f"{step['id'].replace('/', '__')}.log"
        result = self.run_child(argv, env, log, self.timeout_for(step), step["id"])
        if step["kind"] == "hd":
            try:
                (artifact.parent / "run.log").write_text(
                    log.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
            return {**result, **round_status.classify(artifact), "log": str(log)}
        if not (root / "device.json").exists():
            subprocess.run([self.py, str(self.harness / "triton_device_identity.py"), "--write",
                            str(root)], cwd=str(self.repo), env=self.base_env(),
                           capture_output=True, check=False)
        return {**result, **round_status.classify(artifact), "log": str(log)}

    def mint_keep_cache(self, cache: Path) -> Optional[str]:
        cache.mkdir(parents=True, exist_ok=True)
        env = self.base_env()
        env["CUPY_CACHE_DIR"] = str(cache)
        done = subprocess.run([self.py, "-c", "from meep_gpu import subnormal_policy as sp; "
                               "print(sp.point_cupy_cache_at_keep_policy()['provenance']"
                               "['marked'])"], cwd=str(self.repo), env=env,
                              capture_output=True, text=True)
        return None if done.returncode == 0 else (done.stderr or done.stdout)[-400:]

    def step_route(self, step: Dict[str, Any]) -> Dict[str, Any]:
        args = list(step["args"])
        out = Path(args[args.index("--out") + 1])
        if out.exists():
            state = round_status.classify(out)
            if round_status.is_pass(state["state"]):
                return {"skipped": "leg already released", **state}
            return {"state": "refused", "reasons": [
                f"{out} exists and did not release; route campaign names are one-shot, so this "
                f"leg needs a new route stamp (RUNBOOK.md, CERT-4 resume)"]}
        out.parent.mkdir(parents=True, exist_ok=True)
        env = self.base_env()
        env.update({k: v for k, v in step.get("env", {}).items() if v})
        name = step["id"]
        cache = self.dir / "caches" / (f"cache_ftz_stripped_{name}" if step["cache"] == "keep"
                                       else f"cache_plain_{name}")
        env.update({"CUPY_CACHE_DIR": str(cache),
                    "TRITON_CACHE_DIR": str(self.dir / "caches" / f"tcache_{name}")})
        if step["cache"] == "keep":
            problem = self.mint_keep_cache(cache)
            if problem:
                return {"state": "refused", "reasons": [f"keep cache mint failed: {problem}"]}
        else:
            cache.mkdir(parents=True, exist_ok=True)
        log = self.dir / "logs" / f"leg_{name}.log"
        try:
            result = self.run_child([self.py, "-u", str(self.harness / "gate_dispatch_fused_route.py"),
                                     *args], env, log, self.timeout_for(step), name)
        finally:
            # The gate has exited, or was stopped with the lane (the signal handler stops
            # the child before it exits the lane): its records are finished before the leg
            # is judged, released or not. Only this lane's own child is finished: a leg
            # that existed before this step is left as it was (released legs are carried
            # by the sha256 of every file, ``carry.py``).
            finished = finish_route_leg(out)
            self.say(f"{name}: records finished: rewritten {finished['rewritten'] or 'none'}"
                     + (f"; UNFINISHED {finished['unfinished']}" if finished["unfinished"]
                        else ""))
        verdict = round_status.classify(out)
        if step.get("cases_required"):
            fused = sum(1 for line in _lines(log) if "=== case " in line and "-> PASS-FUSED" in line)
            verdict["cases_pass_fused"] = f"{fused} of {step['cases_required']}"
            if fused != step["cases_required"] and verdict["state"] == "released":
                verdict["state"] = "not_released"
                verdict["reasons"] = [f"blocking preflight: {fused} of "
                                      f"{step['cases_required']} cases PASS-FUSED"]
        return {**result, **verdict, "log": str(log), "finished": finished}

    def step_e2e(self, step: Dict[str, Any]) -> Dict[str, Any]:
        out = Path(step["out"])
        if out.exists():
            state = round_status.classify(out / "summary.json")
            if round_status.is_pass(state["state"]):
                return {"skipped": "leg already released", **state}
            return {"state": "refused", "reasons": [f"{out} exists; the licence reruns under a "
                                                    f"fresh E2E name (RUNBOOK.md, CERT-4)"]}
        leg = step["leg"]
        env = self.licence_env(leg, out)
        log = self.dir / "logs" / f"e2e_{leg}.log"
        argv = [self.py, "-u", str(self.harness / "gate_dispatch_end_to_end.py"), "--out", str(out),
                "--cases", step.get("cases", "all"), "--gpu-id", "0", *step.get("args", [])]
        result = self.run_child(argv, env, log, self.timeout_for(step), f"e2e {leg}")
        if step.get("plan_only"):
            return {**result, **plan_only_verdict(out / "summary.json"), "log": str(log)}
        return {**result, **round_status.classify(out / "summary.json"), "log": str(log)}

    def step_toolchain(self, step: Dict[str, Any]) -> Dict[str, Any]:
        script = self.dir / "toolchain_check.py"
        script.write_text(TOOLCHAIN_CHECK)
        log = self.dir / "logs" / "toolchain.log"
        result = self.run_child([self.py, "-u", str(script), str(self.repo),
                                 step["capability"], json.dumps(step.get("versions") or {})],
                                self.base_env(), log, 600, "toolchain")
        lines = [l for l in _lines(log) if l.startswith("TOOLCHAIN")]
        for line in lines:
            if not line.startswith(STACK_LINE):
                self.say(line)
        ok = result["rc"] == 0 and any(l.startswith("TOOLCHAIN OK") for l in lines)
        if ok and step.get("stack_dir"):
            why = self.record_stack(Path(step["stack_dir"]), lines)
            if why:
                return {**result, "state": "refused", "reasons": [why], "log": str(log)}
        return {**result, "state": "released" if ok else "refused",
                "reasons": [] if ok else (lines[-1:] or ["no TOOLCHAIN line"]),
                "log": str(log)}

    def record_stack(self, directory: Path, lines: List[str]) -> Optional[str]:
        """Write ``<directory>/<lane>.json``, the stack this lane's gates run beside, as the
        toolchain step read it on the lane's card; or say why the lane must not go on.

        The bind records a capability's certified stack from these files
        (``record_nvidia_stack.py``). A lane that resumes finds its own record: on the same
        stack it is kept as written, and on another one the lane stops, because its gates
        would then have run beside two stacks under one stamp.
        """
        found = [l[len(STACK_LINE):].strip() for l in lines if l.startswith(STACK_LINE)]
        try:
            read = json.loads(found[-1])
        except (IndexError, ValueError) as exc:
            return f"the toolchain step printed no readable {STACK_LINE.strip()} line ({exc!r})"
        record = {"schema": STACK_SCHEMA, "lane": self.dir.name,
                  "capability": read.get("capability"), "stack": read.get("stack")}
        path = directory / f"{self.dir.name}.json"
        # Named relative to the results tree in what the lane says, never by its absolute
        # path: the log and the step rows travel with the round's evidence.
        try:
            named = path.relative_to(self.results).as_posix()
        except ValueError:
            named = f"{directory.name}/{path.name}"
        if path.is_file():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return (f"{named} is unreadable ({type(exc).__name__}); it is this lane's "
                        f"stack record")
            moved = sorted(k for k in ("capability", "stack") if old.get(k) != record[k])
            if moved:
                return (f"{named} records another {' and '.join(moved)} than this lane reads "
                        f"now ({old.get('stack')} against {record['stack']}): the lane's "
                        f"gates would run beside two stacks under one stamp; run its units "
                        f"again in a new chunk")
            self.say(f"stack record {named} kept (the same stack)")
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(record, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime())),
                                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self.say(f"stack record {named}: " + ", ".join(
            f"{axis} {value}" for axis, value in sorted((record["stack"] or {}).items())))
        return None

    def step_e2e_verdict(self, step: Dict[str, Any]) -> Dict[str, Any]:
        out = Path(step["out"])
        if (out / "VERDICT.txt").is_file():
            text = (out / "VERDICT.txt").read_text()
            return {"state": "released" if text.startswith("VERDICT GO") else "not_released",
                    "skipped": "VERDICT.txt exists", "reasons": [text.strip()[:300]]}
        script = self.dir / "e2e_verdict.py"
        script.write_text(E2E_VERDICT)
        env = self.base_env()
        fastpath_sha = hashlib.sha256((self.repo / "meep_gpu" / "fastpath.py").read_bytes()).hexdigest()
        log = self.dir / "logs" / "e2e_verdict.log"
        result = self.run_child([self.py, str(script), str(out), fastpath_sha,
                                 step["capability"]], env, log, 300, "e2e verdict")
        text = "\n".join(l for l in _lines(log) if l.startswith("VERDICT"))
        (out / "VERDICT.txt").write_text(N.text(text) + "\n")
        go = text.startswith("VERDICT GO")
        return {**result, "state": "released" if go else "not_released",
                "reasons": [text[:300]], "log": str(log)}

    def run(self, plan: List[Dict[str, Any]]) -> int:
        last = self.last_rows()
        steps = [self.sub(step) for step in plan]
        n = len(steps)
        self.say(f"START {n} step(s), {sum(int(s.get('expect_s', 0)) for s in steps)} s expected "
                 f"(legs.py's measured A6000 walls), repo {self.repo}, results {self.results}")
        problems = self.user_site_problems()
        if problems:
            return self.stop("; ".join(problems))
        self.say(f"user site directory off in the gate and licence children ({self.py}, "
                 f"PYTHONNOUSERSITE=1)")
        toolchain = self.toolchain()
        if toolchain["problems"]:
            return self.stop("; ".join(toolchain["problems"]))
        self.say(self.legs.launcher_line(toolchain))
        lifting = self.legs.lifting_legs(steps)
        if lifting:
            problems = self.legs.corpus_problems(steps, self.corpus)
            if problems:
                return self.stop("; ".join(problems))
            self.say(f"MEEP corpus {self.corpus} ({len(lifting)} lifting leg(s)), exported as "
                     f"{self.legs.CORPUS_VARIABLE}")
        if self.gpu is None and not self.take_gpu():
            return self.stop("no GPU lock acquired" if self.args.want_gpu is None
                             else f"gpu_lock_{self.args.want_gpu} not acquired")
        self.say(f"GPU {self.gpu}: {self.smi('--id=' + str(self.gpu), '--query-gpu=name,compute_cap,memory.used', '--format=csv,noheader').strip()}")
        good = bad = 0
        for i, step in enumerate(steps, 1):
            if (self.dir / "STOP").exists():
                return self.stop(f"STOP file present before step {i}/{n}")
            prior = last.get(step["id"])
            needs = [x for x in step.get("needs", []) if not round_status.is_pass(
                (self.last_rows().get(x) or {}).get("state", ""))]
            if needs:
                result = {"state": "skipped", "reasons": [f"needs {needs}, which did not pass"]}
            elif prior and round_status.is_pass(prior.get("state", "")) and step["kind"] in ("route", "e2e"):
                result = {"state": prior["state"], "skipped": "passed in an earlier run of this lane"}
            elif (unplaced := self.place_step(step)):
                result = {"state": "refused", "reasons": [f"not placed: {unplaced}"]}
            else:
                self.say(f"[{i}/{n}] {step['id']} start (expect ~{step.get('expect_s')} s)")
                kind = step["kind"]
                result = {"gate": self.step_gate, "fleet": self.step_fleet,
                          "stencil": self.step_hand, "bitid": self.step_hand,
                          "hd": self.step_hand,
                          "route": self.step_route, "e2e": self.step_e2e,
                          "e2e_verdict": self.step_e2e_verdict,
                          "toolchain": self.step_toolchain}[kind](step)
            state = result.get("state", "unknown")
            row = {"id": step["id"], "kind": step["kind"], "expect_s": step.get("expect_s"),
                   **{k: v for k, v in result.items() if k not in ("id", "kind")}}
            self.row(row)
            ok = round_status.is_pass(state)
            good += ok
            bad += not ok
            self.say(f"[{i}/{n}] {step['id']} {state} rc={result.get('rc')} "
                     f"{result.get('elapsed_s', 0)} s {result.get('passing', '')} "
                     f"{'; '.join(str(x) for x in (result.get('reasons') or [])[:2])[:200]} | "
                     f"lane {int((time.time() - self.t0) / 60)} min, {good} of {i} passing")
            if step.get("blocking") and not ok:
                return self.stop(f"blocking step {step['id']} did not pass; the rest of the lane "
                                 f"is not run")
        (self.dir / "DONE").write_text(f"{time.strftime('%FT%TZ', time.gmtime())} "
                                       f"{good} of {n} steps passing\n")
        self.say(f"DONE {good} of {n} steps passing")
        return 0 if bad == 0 else 1


def plan_only_verdict(summary: Path) -> Dict[str, Any]:
    """The licence preflight: the default enable dispatched on the NVIDIA tables."""
    try:
        rows = json.loads(summary.read_text(encoding="utf-8")).get("rows") or [{}]
    except (OSError, ValueError) as exc:
        return {"state": "missing", "reasons": [f"{summary}: {exc}"]}
    plan = (rows[0] or {}).get("plan") or {}
    enable = plan.get("enable") or {}
    tables = set((plan.get("composition") or {}).get("tables_dispatched") or [])
    ok = (plan.get("decision") == "dispatched" and enable.get("value") is None
          and enable.get("effective") is True and bool(tables) and tables <= {"triton", "cuda"})
    return {"state": "released" if ok else "not_released",
            "reasons": [] if ok else [f"decision={plan.get('decision')} enable={enable} "
                                      f"tables={sorted(tables)}"]}


#: The schema of a lane's stack record (``record_nvidia_stack.py`` reads it), and the line the
#: toolchain check prints it on.
STACK_SCHEMA = "meep_gpu.nvidia_stack/1"
STACK_LINE = "TOOLCHAIN STACK "

#: The card and toolchain check (the 0.9.2 lanes' rl_write_toolchain_check), run as a
#: step on the lane's GPU so it opens its CUDA context where the gates will. It also prints
#: the NVIDIA stack the lane's gates run beside, read as dispatch reads it
#: (``fastpath._nvidia_stack_identity``), which a gate stage records for the bind.
TOOLCHAIN_CHECK = r'''
import json, os, site, sys
repo, want_cc, versions = os.path.realpath(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
import cupy, triton, meep
import meep_gpu
from meep_gpu import fastpath
f = os.path.realpath(meep_gpu.__file__)
p = cupy.cuda.runtime.getDeviceProperties(0)
name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
cc = f"{p['major']}.{p['minor']}"
print(f"TOOLCHAIN card {name} | cc {cc} | cupy {cupy.__version__} | triton {triton.__version__} | "
      f"meep {meep.__version__} {'single' if meep.is_single_precision() else 'double'} "
      f"{'MPI' if meep.with_mpi() else 'serial'} | meep_gpu from {f} | "
      f"user site {'on' if site.ENABLE_USER_SITE else 'off'}", flush=True)
print("TOOLCHAIN STACK " + json.dumps({"capability": cc,
                                       "stack": fastpath._nvidia_stack_identity(cupy)},
                                      sort_keys=True), flush=True)
bad = []
if site.ENABLE_USER_SITE: bad.append("the user site directory is enabled (PYTHONNOUSERSITE=1 not honoured)")
if not f.startswith(repo + os.sep): bad.append(f"meep_gpu imports from {f}, not from {repo}")
if cc != want_cc: bad.append(f"compute capability {cc}, expected {want_cc}")
for mod, want in (("triton", versions.get("triton")), ("cupy", versions.get("cupy")),
                  ("meep", versions.get("meep"))):
    have = {"triton": triton, "cupy": cupy, "meep": meep}[mod].__version__
    if want and have != want: bad.append(f"{mod} {have}, expected {want}")
if not (meep.is_single_precision() and meep.get_realnum_size() == 4): bad.append("MEEP is not single precision")
if not meep.with_mpi(): bad.append("MEEP is not an MPI build")
print(("TOOLCHAIN REFUSED: " + "; ".join(bad)) if bad else "TOOLCHAIN OK", flush=True)
sys.exit(1 if bad else 0)
'''

#: The licence's GO / NO-GO (the 0.9.2 certification round's verdict rule, unchanged).
E2E_VERDICT = r'''
import json, os, sys
out, tree_fastpath, want_cc = sys.argv[1:4]
import gate_dispatch_end_to_end as e2e
from recut_driver_dispatch_record import LICENCE_MIN_DISPATCHED
ship = json.load(open(os.path.join(out, "ship", "summary.json")))
null = json.load(open(os.path.join(out, "null", "summary.json")))
prov = ship.get("provenance") or {}; verdicts = ship.get("verdicts") or {}
reasons = []
if sorted(verdicts) != sorted(e2e.CASES): reasons.append(f"covers {sorted(verdicts)}")
bad = {c: v for c, v in verdicts.items() if v not in ("PASS-DISPATCHED", "PASS-FELL-BACK")}
if bad: reasons.append(f"not released: {bad}")
dispatched = sorted(c for c, v in verdicts.items() if v == "PASS-DISPATCHED")
if len(dispatched) < LICENCE_MIN_DISPATCHED: reasons.append(f"{len(dispatched)} dispatched < {LICENCE_MIN_DISPATCHED}")
div = {c: v for c, v in (ship.get("first_divergent_checkpoint") or {}).items() if v is not None}
if div: reasons.append(f"divergent: {div}")
env = prov.get("environment_at_start") or {}
shaping = sorted(n for n in env if (n.startswith(("MEEP_GPU_", "TRIDENT_FDTD_")) and n != "MEEP_GPU_DISPATCH_LOG") or n == "CUPY_ACCELERATORS")
if shaping: reasons.append(f"shaping variables {shaping}")
if prov.get("install_policy_requested") is not None: reasons.append("harness installed a policy")
if not ship.get("enable_from_default"): reasons.append("enable was set")
if (prov.get("source_sha256") or {}).get("fastpath.py") != tree_fastpath: reasons.append("fastpath.py ran != tree")
cc = (prov.get("device") or {}).get("compute_capability")
if cc != want_cc: reasons.append(f"device compute capability {cc}")
nv = null.get("verdicts") or {}
if not (null.get("null_control") and sorted(nv) == sorted(e2e.CASES) and set(nv.values()) == {"PASS-FELL-BACK"}
        and all(v is None for v in (null.get("first_divergent_checkpoint") or {}).values())):
    reasons.append(f"null control not clean: {nv}")
print("VERDICT", "GO" if not reasons else "NO-GO", f"dispatched {len(dispatched)} of {len(verdicts)} {dispatched}", reasons)
'''


def _lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _last_line(path: Path) -> str:
    lines = [l for l in _lines(path) if l.strip()]
    return lines[-1] if lines else ""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--lane-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--subs", default="", help="JSON map of placeholder -> value")
    ap.add_argument("--gpu", type=int, default=None, help="run on this card (with --no-lock, "
                    "as told; otherwise take its lock)")
    ap.add_argument("--no-lock", dest="lock", action="store_false")
    ap.add_argument("--lock-dir", default=os.environ.get("ROUND_LOCK_DIR", ""),
                    help="where the account's gpu_lock_* directories live (default "
                         "$ROUND_LOCK_DIR); required unless --gpu N --no-lock")
    ap.add_argument("--max-locks", type=int, default=4)
    ap.add_argument("--avoid", default="",
                    help="comma-separated card indices the lane never takes, whatever they read; "
                         "also handed to the fleet driver as its set-aside cards")
    ap.add_argument("--idle-mib", type=int, default=display_card.IDLE_MIB,
                    help="the idle rule's memory bound, MiB (display_card.IDLE_MIB)")
    ap.add_argument("--step-patience", type=int, default=300,
                    help="seconds a step waits for the round's rule to take the card again "
                         "before it reads refused (every step is placed, not only the first); "
                         "the fleet driver waits as long before each gate")
    ap.add_argument("--share-idle-foreign", action="append", default=[],
                    help="<uuid>:<pid>[,<pid>...]: admit card <uuid> although these compute "
                         "apps run on it, when they and the card read 0 %% utilisation and "
                         "--share-min-free-mib is free, on both readings (repeatable)")
    ap.add_argument("--share-min-free-mib", type=int, default=0,
                    help="with --share-idle-foreign: the free memory a shared card must have "
                         "(the unit's peak plus the margin), MiB")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--lock-patience", type=int, default=12 * 3600)
    ap.add_argument("--label", default="1.0 certification round")
    ap.add_argument("--libcuda-stub", default=os.environ.get("TRITON_LIBCUDA_PATH", ""),
                    help="the directory Triton links -lcuda from (default: "
                         "$TRITON_LIBCUDA_PATH, else the one the reference environment's "
                         "build recorded in share/meep-gpu-build/libcuda.json)")
    ap.add_argument("--heartbeat", type=int, default=120)
    ap.add_argument("--timeout-factor", type=int, default=4)
    ap.add_argument("--min-timeout", type=int, default=1800)
    ap.add_argument("--timeout-floor", action="append", default=[])
    ap.add_argument("--corpus-root", default="",
                    help="the MEEP corpus the H->D lifts walk (default: "
                         "$MEEP_GPU_CORPUS_ROOT, $MGPU_SITE_MEEP_SOURCE/python, ~/meep/python)")
    args = ap.parse_args(argv)
    try:
        share_specs(args.share_idle_foreign)
    except ValueError as refused:
        ap.error(str(refused))
    if args.share_idle_foreign and args.share_min_free_mib <= 0:
        ap.error("--share-idle-foreign needs --share-min-free-mib (the unit's peak plus margin)")
    args.want_gpu = args.gpu
    if args.gpu is not None and args.lock:
        args.gpu = None           # take the named card's lock first
    if args.gpu is None and not args.lock_dir:
        ap.error("a lane that takes its own GPU lock needs --lock-dir or $ROUND_LOCK_DIR")
    # The plan holds paths under their labels (round.py writes it with N.path_value); they
    # are resolved back on this machine before anything runs.
    plan = N.resolve_value(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    lane = Lane(args)

    def on_signal(signum, _frame):
        lane.say(f"terminated by signal {signum}; stopping the running child")
        lane._kill_child()
        lane.release()
        (lane.dir / "STOPPED").write_text(f"signal {signum}\n")
        sys.exit(143)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    try:
        return lane.run(plan)
    finally:
        lane.release()


if __name__ == "__main__":
    sys.exit(main())
