#!/usr/bin/env python3
"""The round's one rule for taking a GPU, and what it records: idle, or a card that also drives a display.

Every gate of the certification round runs on a card this rule took, and no other file decides
whether a card is free. ``run_lane.py`` applies it when it takes a card (two readings
``--poll`` seconds apart, :func:`decide`) and again before every step it starts; the Triton
fleet driver (``drive_triton_weld_gates.py``) holds no placement logic of its own: inside a
lane it runs on the card the lane gives it and the lane's recorded decision
(``--lane-placement``), and before every gate it asks the same rule with :class:`Placement`;
run on its own, with no lane, it asks that same rule for the card it is told to use and
refuses the card when the rule does.

THE IDLE RULE. A card reads idle when no compute process of any user is listed on it, its
utilisation is 0 % and its memory in use is at most :data:`IDLE_MIB`. Memory that no listed
process accounts for counts as another user's: a container hides other accounts' processes
from ``nvidia-smi``. A card that is set aside (``run_lane.py --avoid``, the fleet driver's
``--forbidden-gpus``) is never available, whatever it reads.

THE DISPLAY RULE. A card that drives a display never reads idle by that rule. The desktop
keeps memory on it that no compute process accounts for (692 MiB on an RTX 3090 driving a
Windows desktop); under WSL2 no process of the Windows side is ever listed. A card that the
idle rule refuses, that carries no listed compute process and that is a display card is judged
by what idle means there:

* A display card is one whose NVML display state says a display is attached
  (``display_attached`` Yes; on a driver that predates that field, ``display_mode`` Enabled,
  which meant the same there) or active (``display_active`` Enabled), or any card of a WSL2
  host, where Windows owns the display and every Windows process is outside the Linux view.
* It reads idle when no compute process of any user is listed on it, its memory in use is at
  most the resident measured when the round first read it (the DISPLAY RESIDENT) plus
  :data:`SLACK_MIB`, and its utilisation is at most :data:`UTIL_MAX_PCT`.

A card that is not a display card is judged by the idle rule alone: unattributable memory
there is another user's resident. A display state that cannot be read is no display. The
display rule only widens admission where the idle rule refuses: a card the idle rule takes is
taken as before, a card carrying a listed compute process is never taken by it (sharing such a
card stays ``run_lane.py --share-idle-foreign``'s opt-in, and the fleet driver never shares),
and a card that is set aside stays set aside.

WHAT A TAKEN CARD KEEPS. The rule that took a card governs every later reading of it: a card
taken by the idle rule is held to the idle rule (the display rule does not take over when
something else arrives on it), and a card taken under the display rule is held to the
resident it was taken at, so memory a program takes on the card while the round holds it is
read as another resident, and the caller waits for it to leave before refusing the gate.

THE THRESHOLDS, measured on an RTX 3090 (24,576 MiB, driver 616.56) driving a Windows desktop,
read from WSL2 with ``nvidia-smi`` as the round reads it:

* :data:`SLACK_MIB` (128 MiB). At rest the desktop's memory moved by 5 MiB (692 to 697 MiB
  over 709 readings in 15 minutes); one bare CUDA context (CuPy, one element) took 258 MiB
  more. The slack sits far above the first and below half the second, so a compute context
  that arrives after the first reading exceeds it even where its process is not listed (a
  Windows process under WSL2, another account's in a container).
* :data:`UTIL_MAX_PCT` (10 %). At rest the desktop read 0 % in 706 of those 709 readings and
  6 % at the most; a running gate reads up to 99 %.
* :data:`IDLE_MIB` (200 MiB), the one idle bound. Until the fleet driver shared this rule it
  kept its own, 64 MiB, and read no utilisation; a card holding 65 to 200 MiB with no listed
  process and 0 % utilisation is now taken by the lane and the driver alike, and a card that
  drives no display and reads any utilisation is refused by both.
* :data:`RECHECK_S` (10 s), how often a caller that holds a card reads it again while it waits
  for it (the lane before a step, the fleet driver before a gate). Just after a gate exited the
  card read 16 % and 28 % utilisation, and 0 % ten seconds later.

Under WSL2 ``nvidia-smi`` lists the compute processes of the Linux side (the CuPy context above
was listed on each of its 26 readings) and none of the Windows side; the display state reads
there as on Windows (``display_attached`` Yes, ``display_active`` Enabled, ``display_mode``
deprecated, on driver 616.56). The ``display_mode`` reading is the documented meaning on
drivers that predate ``display_attached``; it was not measured here.

THE RECORD of a decision (:func:`admission_record`, written by the lane to
``<lane>/placement.json`` and handed to the fleet driver with ``--lane-placement``): the rule
that took the card (``idle``, ``display`` or ``share``), the GPU index, the idle bound, the
readings that took it, and for a display card the display state, the resident and both
thresholds. The lane also writes ``DISPLAY CARD`` to its ``progress.log``, ``display_resident=
<MiB>`` to its lock's label and a ``display`` block to every ``steps.jsonl`` row; the fleet
driver writes the decision of each gate's check (``placement``, ``placement_check``) into the
gate's row of ``gates.jsonl`` and ``campaign.json``, and the lane's record (or its own first
reading, when it ran alone) into ``campaign.json`` (``placement``). A card taken by the idle
rule records nothing in ``steps.jsonl`` that it did not before.

usage: display_card.py [--gpu N] [--poll S] [--idle-mib M]
    reads card N twice, S seconds apart (default 60, as a lane does), and prints what the
    lane and the fleet driver would decide; exit 0 when both take it, 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

#: Memory a display card may gain over its display resident and still read idle, MiB.
SLACK_MIB = 128
#: The utilisation a display card may read and still read idle, percent.
UTIL_MAX_PCT = 10
#: The idle rule's memory bound, MiB: the most a card may hold with no listed process and 0 %
#: utilisation and still read idle (``run_lane.py --idle-mib``'s default).
IDLE_MIB = 200
#: Seconds between two readings while a caller waits for a card it holds.
RECHECK_S = 10
#: What says a host is WSL2: a Microsoft kernel, and the paravirtual GPU device.
OSRELEASE = Path("/proc/sys/kernel/osrelease")
WSL_GPU_DEVICE = Path("/dev/dxg")
#: The schema of the lane's record of a taken card (``--lane-placement``).
PLACEMENT_SCHEMA = "round_card_placement/1"
#: The rules that can take a card; ``share`` is ``run_lane.py --share-idle-foreign``'s.
RULES = ("idle", "display", "share")

#: ``smi(*arguments) -> stdout``: how a caller asks ``nvidia-smi``.
Smi = Callable[..., str]


def wsl2_host(osrelease: Path = OSRELEASE, device: Path = WSL_GPU_DEVICE) -> bool:
    """True on a WSL2 host: its kernel names Microsoft and the GPU is reached through
    ``/dev/dxg``. A host that cannot answer is not one."""
    try:
        release = osrelease.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "microsoft" in release.lower() and device.exists()


def run_smi(*arguments: str) -> str:
    """``nvidia-smi``'s stdout, with ``CUDA_VISIBLE_DEVICES`` removed (a mask would rename
    the physical index asked about); an unavailable tool answers with a line no reader parses."""
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    try:
        return subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True,
                              timeout=60, env=env).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"


def _flag(value: str) -> Optional[bool]:
    word = value.strip().lower()
    if word in ("yes", "enabled"):
        return True
    if word in ("no", "disabled"):
        return False
    return None


def _first_line(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return lines[0] if lines else ""


def display_state(smi: Smi, index: int, *, wsl2: bool) -> Dict[str, Any]:
    """The card's NVML display state, field by field, and whether it is a display card.

    ``display_attached`` is asked on its own: a driver that predates it refuses the whole
    query that names it. A value other than Yes/No, Enabled/Disabled (a deprecation notice,
    an error) reads as unknown, and unknown is no display."""
    pair = _first_line(smi(f"--id={index}", "--query-gpu=display_active,display_mode",
                           "--format=csv,noheader"))
    parts = [p.strip() for p in pair.split(",")] if pair.count(",") == 1 else ["", ""]
    attached = _first_line(smi(f"--id={index}", "--query-gpu=display_attached",
                               "--format=csv,noheader"))
    if "," in attached:
        attached = ""
    fields = {"display_attached": attached, "display_active": parts[0],
              "display_mode": parts[1]}
    said = [f"{name} {value}" for name, value in fields.items() if _flag(value)]
    if wsl2:
        said.append("a WSL2 host")
    return {"display": bool(said), "said": said, "fields": fields, "wsl2": bool(wsl2)}


def describe(state: Dict[str, Any]) -> str:
    """``display card (display_attached Yes, ...)`` or ``no display (...)``."""
    if state["display"]:
        return f"display card ({', '.join(state['said'])})"
    shown = ", ".join(f"{k} {v or '?'}" for k, v in state["fields"].items())
    return f"no display ({shown}; not a WSL2 host)"


def judge(state: Dict[str, Any], *, used_mib: int, util_pct: int, apps: int,
          resident_mib: Optional[int]) -> List[str]:
    """Why the display rule refuses one reading of a card, each by name ([] admits it).

    ``resident_mib`` is the display resident; None on the reading that measures it."""
    refused = []
    if not state["display"]:
        refused.append(f"not a display card: {describe(state)}")
    if apps:
        refused.append(f"{apps} compute process(es) listed on the card")
    if util_pct > UTIL_MAX_PCT:
        refused.append(f"utilisation {util_pct} % above {UTIL_MAX_PCT} %")
    if resident_mib is not None and used_mib > resident_mib + SLACK_MIB:
        refused.append(f"{used_mib} MiB used, above the display resident {resident_mib} MiB "
                       f"+ {SLACK_MIB} MiB")
    return refused


def lane_rule(used_mib: int, util_pct: int, *, idle_mib: int,
              state: Callable[[], Dict[str, Any]], resident_mib: Optional[int],
              display_allowed: bool = True) -> Tuple[Optional[str], Dict[str, Any]]:
    """Which rule takes one reading of a card with no listed compute process:
    ``"idle"``, ``"display"`` or None, and what the display rule saw (``{}`` when the idle
    rule took it, or when ``display_allowed`` is False and the idle rule refused). ``state``
    is asked only when the idle rule refuses and the display rule may be tried."""
    if used_mib <= idle_mib and util_pct == 0:
        return "idle", {}
    if not display_allowed:
        return None, {}
    seen = state()
    refused = judge(seen, used_mib=used_mib, util_pct=util_pct, apps=0,
                    resident_mib=resident_mib)
    return (None if refused else "display"), {
        "state": seen, "used_mib": used_mib, "util_pct": util_pct,
        "resident_mib": used_mib if resident_mib is None else resident_mib,
        "refused": refused}


def decide(index: int, reading: Mapping[str, Any], *, idle_mib: int = IDLE_MIB,
           state: Callable[[], Dict[str, Any]], resident_mib: Optional[int] = None,
           set_aside: Collection[int] = (), display_allowed: bool = True) -> Dict[str, Any]:
    """THE rule, for one reading of one card: whether it is taken, by which rule, and why not.

    ``reading`` holds ``used_mib``, ``util_pct`` and ``listed`` (True when any compute process
    is listed on the card; None when that is unreadable). ``resident_mib`` is the display
    resident (None on the reading that measures it); ``display_allowed`` False holds a card
    to the idle rule alone (a card the idle rule took). The answer: ``taken``, ``rule``
    (``idle``, ``display`` or None), ``refused`` (each reason by name), ``text`` (one line
    saying what was decided), ``display`` (what the display rule saw, when it was asked) and
    the reading."""
    used, util, listed = reading.get("used_mib"), reading.get("util_pct"), reading.get("listed")
    out: Dict[str, Any] = {"gpu": index, "taken": False, "rule": None, "refused": [],
                           "display": None, "used_mib": used, "util_pct": util}
    if index in set_aside:
        out["refused"] = [f"GPU {index} is set aside on this host (named with --avoid or "
                          f"--forbidden-gpus) and is never available to the round"]
    elif used is None or util is None or listed is None:
        out["refused"] = [f"GPU {index} memory, utilisation or process list is unreadable"]
    elif listed:
        out["refused"] = [f"GPU {index} carries a compute process (a listed process of any "
                          f"user refuses a card)"]
    else:
        rule, seen = lane_rule(used, util, idle_mib=idle_mib, state=state,
                               resident_mib=resident_mib, display_allowed=display_allowed)
        out["display"] = seen or None
        if rule == "idle":
            out["taken"], out["rule"] = True, "idle"
            out["text"] = f"GPU {index} idle: {used} MiB, {util} % utilisation, no compute " \
                          f"process listed"
            return out
        base = (f"GPU {index} reports {used} MiB used and {util} % utilisation with no listed "
                f"process (idle: at most {idle_mib} MiB and 0 %): an unattributable resident "
                f"is still a resident")
        if not display_allowed:
            out["refused"] = [f"{base}; the card was taken by the idle rule, which holds it"]
        elif rule == "display":
            out["taken"], out["rule"] = True, "display"
            out["text"] = (f"GPU {index} taken under the display rule: "
                           f"{describe(seen['state'])}; {used} MiB in use against the display "
                           f"resident {seen['resident_mib']} MiB + {SLACK_MIB} MiB, "
                           f"utilisation {util} % (at most {UTIL_MAX_PCT} %), no compute "
                           f"process listed")
            return out
        elif seen["state"]["display"]:
            out["refused"] = [f"{base}; the display rule refuses it too: "
                              f"{'; '.join(seen['refused'])}"]
        else:
            out["refused"] = [f"{base}; {seen['refused'][0]}"]
    out["text"] = "; ".join(out["refused"])
    return out


def read_card(smi: Smi, index: int) -> Dict[str, Any]:
    """One reading of a card: memory in use, utilisation and the listed compute PIDs.
    ``None`` for a value ``nvidia-smi`` did not answer."""
    line = _first_line(smi(f"--id={index}", "--query-gpu=memory.used,utilization.gpu",
                           "--format=csv,noheader,nounits"))
    parts = [p.strip() for p in line.split(",")]
    used = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else None
    util = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else None
    pids: Optional[List[int]] = []
    for row in smi(f"--id={index}", "--query-compute-apps=pid",
                   "--format=csv,noheader").splitlines():
        row = row.strip()
        if not row or row.lower().startswith("no running"):
            continue
        if row.isdigit():
            pids.append(int(row))
        else:
            pids = None
            break
    return {"used_mib": used, "util_pct": util, "pids": pids}


def reading_of(smi: Smi, index: int) -> Dict[str, Any]:
    """:func:`read_card` as :func:`decide` takes it: ``listed`` True when any compute process
    is listed, None when the list is unreadable."""
    card = read_card(smi, index)
    return {"used_mib": card["used_mib"], "util_pct": card["util_pct"],
            "listed": None if card["pids"] is None else bool(card["pids"])}


def admission_record(index: int, rule: str, *, idle_mib: int,
                     readings: Sequence[Mapping[str, Any]],
                     display: Optional[Mapping[str, Any]] = None, lane: str = "",
                     source: str = "") -> Dict[str, Any]:
    """The record of a card taken (module docstring, THE RECORD). ``display`` is, for the
    display rule, ``{"state": ..., "resident_mib": ...}``."""
    record: Dict[str, Any] = {
        "schema": PLACEMENT_SCHEMA, "gpu": int(index), "rule": rule, "idle_mib": int(idle_mib),
        "readings": [{"used_mib": r["used_mib"], "util_pct": r["util_pct"]} for r in readings],
        "taken_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if display:
        record["display"] = {"state": display["state"], "resident_mib": display["resident_mib"],
                             "slack_mib": SLACK_MIB, "util_max_pct": UTIL_MAX_PCT}
    if lane:
        record["lane"] = lane
    if source:
        record["source"] = source
    return record


def record_problems(record: Any, index: int) -> List[str]:
    """Why ``record`` (a lane's record of a taken card) cannot be adopted for card ``index``,
    each by name; empty when it can."""
    if not isinstance(record, dict) or record.get("schema") != PLACEMENT_SCHEMA:
        return [f"not a card placement record ({PLACEMENT_SCHEMA})"]
    problems = []
    if record.get("gpu") != index:
        problems.append(f"the record is of GPU {record.get('gpu')!r} and this run is on "
                        f"GPU {index}")
    if record.get("rule") not in RULES:
        problems.append(f"rule {record.get('rule')!r} is not one of {list(RULES)}")
    if not isinstance(record.get("idle_mib"), int):
        problems.append("idle_mib is not a number")
    if record.get("rule") == "display" and not isinstance(
            (record.get("display") or {}).get("resident_mib"), int):
        problems.append("a display card's record carries no display resident")
    return problems


class Placement:
    """The round's rule applied to one card, reading by reading, for a caller that holds no
    lane's lock: the Triton fleet driver before every gate (and the lane before every step).

    The card's rule is set once: from the lane's record (:meth:`adopt`), from this caller's
    own first reading (:meth:`start`, for a run with no lane), or by the first reading that
    takes the card; every later reading is judged by :func:`decide` under it (WHAT A TAKEN
    CARD KEEPS). :attr:`last` holds the latest decision, for the gate's row."""

    def __init__(self, smi: Smi, *, idle_mib: int = IDLE_MIB,
                 set_aside: Collection[int] = (), wsl2: Optional[bool] = None,
                 say: Callable[[str], None] = print):
        self.smi, self.idle_mib, self.set_aside, self.say = smi, idle_mib, frozenset(set_aside), say
        self.wsl2 = wsl2_host() if wsl2 is None else wsl2
        self.index: Optional[int] = None
        #: The rule the card was taken under (None until it is), its display resident and
        #: where that came from, and the record handed in or made at the start.
        self.rule: Optional[str] = None
        self.resident: Optional[int] = None
        self.resident_from: Optional[str] = None
        self.record: Optional[Dict[str, Any]] = None
        self.source: Optional[str] = None
        self.last: Optional[Dict[str, Any]] = None

    def _state(self, index: int) -> Dict[str, Any]:
        return display_state(self.smi, index, wsl2=self.wsl2)

    def adopt(self, record: Dict[str, Any], index: int, source: str = "the lane") -> None:
        """Hold card ``index`` to the lane's record of how it took it. A card the lane took
        by the share opt-in is held to no rule here (a driver never shares: the rule refuses
        it while another process is listed)."""
        problems = record_problems(record, index)
        if problems:
            raise ValueError("; ".join(problems))
        self.index, self.record, self.source = index, dict(record), source
        self.idle_mib = int(record["idle_mib"])
        self.rule = record["rule"] if record["rule"] in ("idle", "display") else None
        if self.rule == "display":
            self.resident, self.resident_from = record["display"]["resident_mib"], source
        self.say(f"placement: GPU {index} taken by {source} under the {record['rule']} rule "
                 + (f"(display resident {self.resident} MiB; a gate starts at no listed compute "
                    f"process, at most {self.resident} + {SLACK_MIB} MiB in use and at most "
                    f"{UTIL_MAX_PCT} % utilisation)" if self.rule == "display" else
                    f"(a gate starts at no listed compute process, at most {self.idle_mib} MiB "
                    f"in use and 0 % utilisation)" if self.rule == "idle" else
                    "(the share opt-in; this caller never shares)"))

    def start(self, index: int) -> Dict[str, Any]:
        """A run with no lane: the first reading of card ``index`` sets the rule, as the
        lane's first reading does (no memory test on it for a display card: its memory is the
        resident). Returns the decision; a card the rule refuses stays refused, and each gate's
        check asks again."""
        self.index, self.source = index, "this run's first reading"
        reading = reading_of(self.smi, index)
        decision = decide(index, reading, idle_mib=self.idle_mib,
                          state=lambda: self._state(index), resident_mib=None,
                          set_aside=self.set_aside)
        self._set_rule(decision, "this run's first reading")
        self.record = admission_record(
            index, decision["rule"] or "refused", idle_mib=self.idle_mib, readings=[reading],
            display=decision["display"], source=self.source)
        if decision["rule"] is None:
            self.record.pop("display", None)
        self.say(f"placement: {decision['text']}" + (
            f"; display resident {self.resident} MiB (from {self.resident_from}): a gate starts "
            f"at no listed compute process, at most {self.resident} + {SLACK_MIB} MiB in use "
            f"and at most {UTIL_MAX_PCT} % utilisation" if decision["rule"] == "display"
            else ""))
        return decision

    def _set_rule(self, decision: Mapping[str, Any], source: str) -> None:
        if decision["taken"] and self.rule is None:
            self.rule = decision["rule"]
            if self.rule == "display":
                self.resident = decision["display"]["resident_mib"]
                self.resident_from = source

    def check(self, index: int) -> Tuple[bool, str]:
        """One reading of card ``index`` under the rule it was taken under: ``(taken, why)``.
        A card nothing has taken yet is taken by the first reading that passes the rule."""
        reading = reading_of(self.smi, index)
        decision = decide(
            index, reading, idle_mib=self.idle_mib, state=lambda: self._state(index),
            resident_mib=self.resident if self.rule == "display" else None,
            set_aside=self.set_aside, display_allowed=self.rule in (None, "display"))
        self._set_rule(decision, "the first reading that took the card")
        self.last = {k: decision[k] for k in ("taken", "rule", "used_mib", "util_pct", "refused")}
        if decision["display"]:
            self.last["display"] = {
                "state": decision["display"]["state"],
                "resident_mib": self.resident if self.rule == "display"
                else decision["display"]["resident_mib"],
                "resident_from": self.resident_from, "slack_mib": SLACK_MIB,
                "util_max_pct": UTIL_MAX_PCT}
        return decision["taken"], decision["text"]

    def summary(self) -> Optional[Dict[str, Any]]:
        """The caller's account of how it placed the card, for the campaign record: where the
        decision came from, the record, and the rule the card is held to now."""
        if self.record is None:
            return None
        return {"source": self.source, "held_to": self.rule, "record": self.record}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, default=0, help="the PHYSICAL card index (default 0)")
    ap.add_argument("--poll", type=int, default=60,
                    help="seconds between the two readings (run_lane.py's default)")
    ap.add_argument("--idle-mib", type=int, default=IDLE_MIB,
                    help="the idle rule's memory bound (run_lane.py --idle-mib)")
    args = ap.parse_args(argv)
    wsl2 = wsl2_host()
    name = _first_line(run_smi(f"--id={args.gpu}", "--query-gpu=name,compute_cap,memory.total",
                               "--format=csv,noheader"))
    print(f"CARD {args.gpu}: {name or 'unreadable'}", flush=True)
    readings, lane = [], []
    for k in range(2):
        if k:
            time.sleep(args.poll)
        card = read_card(run_smi, args.gpu)
        readings.append(card)
        # Counted, never listed: a compute process on the card may be another account's.
        print(f"READING {k + 1}: {card['used_mib']} MiB in use, {card['util_pct']} % "
              f"utilisation, compute processes listed: "
              f"{len(card['pids']) if card['pids'] is not None else 'unreadable'}", flush=True)
        lane.append(decide(
            args.gpu, {"used_mib": card["used_mib"], "util_pct": card["util_pct"],
                       "listed": None if card["pids"] is None else bool(card["pids"])},
            idle_mib=args.idle_mib,
            state=lambda: display_state(run_smi, args.gpu, wsl2=wsl2),
            resident_mib=None if k == 0 else readings[0]["used_mib"]))
    state = display_state(run_smi, args.gpu, wsl2=wsl2)
    print(f"DISPLAY: {describe(state)}", flush=True)
    rules = [d["rule"] for d in lane]
    if None in rules:
        print("LANE: refused (" + "; ".join(r for d in lane for r in d["refused"]) + ")")
    elif "display" in rules:
        print(f"LANE: taken under the display rule (display resident "
              f"{readings[0]['used_mib']} MiB + {SLACK_MIB} MiB, utilisation at most "
              f"{UTIL_MAX_PCT} %, no compute process listed)")
    else:
        print(f"LANE: taken by the idle rule (at most {args.idle_mib} MiB and 0 % on both "
              f"readings, no compute process listed)")
    # The fleet driver's check is the same rule, asked of a card the lane's record describes.
    placement = Placement(run_smi, idle_mib=args.idle_mib, wsl2=wsl2, say=lambda _m: None)
    if None not in rules:
        placement.adopt(admission_record(
            args.gpu, "display" if "display" in rules else "idle", idle_mib=args.idle_mib,
            readings=readings, display={"state": state, "resident_mib": readings[0]["used_mib"]}
            if "display" in rules else None), args.gpu)
    fleet_taken, fleet_why = placement.check(args.gpu)
    print(f"FLEET: {'taken' if fleet_taken else 'refused'}: {fleet_why}")
    taken = None not in rules and fleet_taken
    print(f"DECISION: {'TAKEN' if taken else 'REFUSED'}: card {args.gpu} "
          + ((f"is taken under the {'display' if 'display' in rules else 'idle'} rule") if taken
             else "is not taken; the lane would wait for it and the fleet driver refuse it"))
    return 0 if taken else 1


if __name__ == "__main__":
    sys.exit(main())
