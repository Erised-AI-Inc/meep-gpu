"""The round's one rule for taking a GPU (``display_card.py``), in the lane and the fleet driver.

Host-only: ``nvidia-smi`` is a fake that answers each query from a table of cards, and the
WSL2 probe is given, never read off the host. A card reads idle, or, when it drives a display
and the idle rule refuses it, by the display rule: no compute process listed, utilisation at
most ``UTIL_MAX_PCT`` and memory within ``SLACK_MIB`` of the resident measured first. The lane
applies the rule when it takes a card and before every step; the fleet driver holds no rule of
its own and asks the same one before every gate, from the lane's record or, run alone, from
its own first reading. The shared-machine protections stand: a card that drives no display
and holds memory no listed process accounts for is refused, a listed compute process of any
user refuses a card, the share opt-in admits as before, and a device set aside stays set aside.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

ROUND = Path(__file__).resolve().parent
HARNESS = ROUND.parent.parent
for _path in (HARNESS, ROUND):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import display_card  # noqa: E402
import drive_triton_weld_gates as fleet  # noqa: E402
import run_lane  # noqa: E402

DEPRECATED = "[Requested functionality has been deprecated]"
INVALID = 'Field "display_attached" is not a valid field to query.'
PMON_HEAD = "# gpu         pid  type    sm   mem   enc   dec   command\n" \
            "# Idx           #   C/G     %     %     %     %   name\n"


class Card:
    def __init__(self, index: int, used: int = 692, util: int = 0, apps=None,
                 attached: Optional[str] = "Yes", active: str = "Enabled",
                 mode: str = DEPRECATED, free: int = 23000):
        self.index, self.uuid = index, f"GPU-{index}{index}{index}{index}-card"
        self.used, self.util, self.apps = used, util, dict(apps or {})
        self.attached, self.active, self.mode, self.free = attached, active, mode, free


def headless(index: int, **kw) -> Card:
    """A card that drives no display, on a driver that predates ``display_attached``."""
    return Card(index, attached=None, active="Disabled", mode="Disabled", **kw)


class FakeSmi:
    """``nvidia-smi`` for a set of cards, in the one spelling every caller of the rule uses;
    a query nothing in the round asks is an error. ``display_queries`` counts the
    display-state questions asked."""

    def __init__(self, *cards: Card):
        self.cards: Dict[int, Card] = {c.index: c for c in cards}
        self.display_queries = 0

    def __call__(self, *query: str) -> str:
        query = list(query)
        card = None
        if query[0].startswith("--id="):
            card = self.cards[int(query[0].split("=", 1)[1])]
            query = query[1:]
        q = query[0]
        if card is None:
            if q == "--query-gpu=index,uuid,memory.used,utilization.gpu":
                return "".join(f"{c.index}, {c.uuid}, {c.used}, {c.util}\n"
                               for c in self.cards.values())
            if q == "--query-compute-apps=gpu_uuid":
                return "".join(f"{c.uuid}\n" for c in self.cards.values() for _ in c.apps)
            if q == "--query-compute-apps=gpu_uuid,pid,used_memory":
                return "".join(f"{c.uuid}, {pid}, {mib}\n" for c in self.cards.values()
                               for pid, mib in c.apps.items())
            if q == "--query-gpu=index,memory.free":
                return "".join(f"{c.index}, {c.free}\n" for c in self.cards.values())
            if q == "--query-gpu=index,name,memory.used,utilization.gpu":
                return "".join(f"{c.index}, NVIDIA GeForce RTX 3090, {c.used} MiB, {c.util} %\n"
                               for c in self.cards.values())
            if q == "pmon":
                c = self.cards[int(query[2])]
                return PMON_HEAD + "".join(f"    {c.index} {pid:>11}     C     0     -     -     "
                                           f"-   python\n" for pid in c.apps)
        else:
            if q == "--query-gpu=display_active,display_mode":
                self.display_queries += 1
                return f"{card.active}, {card.mode}\n"
            if q == "--query-gpu=display_attached":
                self.display_queries += 1
                return f"{card.attached}\n" if card.attached is not None else f"{INVALID}\n"
            if q == "--query-gpu=memory.used,utilization.gpu":
                return f"{card.used}, {card.util}\n"
            if q == "--query-gpu=name,compute_cap,memory.used":
                return f"NVIDIA GeForce RTX 3090, 8.6, {card.used} MiB\n"
            if q == "--query-compute-apps=pid":
                return "".join(f"{pid}\n" for pid in card.apps)
        raise AssertionError(f"unexpected query {query}")


def _lane(tmp_path: Path, monkeypatch, smi: FakeSmi, *, wsl2: bool = False, share=(),
          need: int = 0, want: Optional[int] = None, between=None,
          avoid: str = "") -> run_lane.Lane:
    args = argparse.Namespace(
        lane_dir=str(tmp_path / "lane"), repo=str(tmp_path), results=str(tmp_path / "results"),
        python=sys.executable, gpu=None, subs="", libcuda_stub="", corpus_root="",
        min_timeout=1800, timeout_factor=4, heartbeat=120, avoid=avoid, idle_mib=200,
        share_idle_foreign=list(share), share_min_free_mib=need, lock_dir=str(tmp_path / "locks"),
        max_locks=4, poll=1, lock_patience=1, label="1.0 round test", want_gpu=want,
        timeout_floor=[])
    lane = run_lane.Lane(args)
    lane.wsl2 = wsl2
    monkeypatch.setattr(lane, "smi", smi)
    calls = {"n": 0}

    def sleep(_seconds):
        calls["n"] += 1
        if between is not None and calls["n"] == 1:
            between()          # what changes on the card between the two readings
    monkeypatch.setattr(run_lane.time, "sleep", sleep)
    return lane


# ------------------------------------------------------------------- the display state ----
def test_the_display_state_reads_each_field_and_unknown_is_no_display():
    on = display_card.display_state(FakeSmi(Card(0)), 0, wsl2=False)
    assert on["display"] and on["said"] == ["display_attached Yes", "display_active Enabled"]
    old = display_card.display_state(FakeSmi(Card(0, attached=None, active="Disabled",
                                                   mode="Enabled")), 0, wsl2=False)
    assert old["display"] and old["said"] == ["display_mode Enabled"]
    assert old["fields"]["display_attached"] == INVALID
    off = display_card.display_state(FakeSmi(headless(0)), 0, wsl2=False)
    assert not off["display"] and off["said"] == []
    assert "no display" in display_card.describe(off)
    garbled = display_card.display_state(lambda *q: "nvidia-smi unavailable: x", 0, wsl2=False)
    assert not garbled["display"]
    wsl = display_card.display_state(FakeSmi(headless(0)), 0, wsl2=True)
    assert wsl["display"] and wsl["said"] == ["a WSL2 host"]


def test_a_wsl2_host_is_a_microsoft_kernel_with_the_paravirtual_gpu(tmp_path):
    release, device = tmp_path / "osrelease", tmp_path / "dxg"
    release.write_text("6.18.54.1-microsoft-standard-WSL2\n")
    assert not display_card.wsl2_host(release, device)
    device.write_text("")
    assert display_card.wsl2_host(release, device)
    release.write_text("6.8.0-45-generic\n")
    assert not display_card.wsl2_host(release, device)
    assert not display_card.wsl2_host(tmp_path / "absent", device)


@pytest.mark.parametrize("used, util, apps, resident, needle", [
    (692, 0, 0, 692, None),
    (692 + display_card.SLACK_MIB, display_card.UTIL_MAX_PCT, 0, 692, None),
    (692 + display_card.SLACK_MIB + 1, 0, 0, 692, "above the display resident 692 MiB"),
    (692, display_card.UTIL_MAX_PCT + 1, 0, 692, "utilisation 11 % above 10 %"),
    (692, 0, 1, 692, "compute process(es) listed"),
], ids=["idle", "at_the_bounds", "memory_grew", "utilisation", "listed_process"])
def test_the_display_rule_judges_memory_utilisation_and_listed_processes(used, util, apps,
                                                                        resident, needle):
    state = display_card.display_state(FakeSmi(Card(0)), 0, wsl2=False)
    refused = display_card.judge(state, used_mib=used, util_pct=util, apps=apps,
                                 resident_mib=resident)
    assert (refused == []) if needle is None else any(needle in r for r in refused)


def test_the_thresholds_are_the_ones_measured():
    assert (display_card.SLACK_MIB, display_card.UTIL_MAX_PCT, display_card.IDLE_MIB) == \
        (128, 10, 200)
    assert run_lane.display_card is display_card


# ------------------------------------------------------------------------- the one rule ----
def _reading(used=692, util=0, listed=False):
    return {"used_mib": used, "util_pct": util, "listed": listed}


def _decide(card, reading, **kw):
    smi = FakeSmi(card)
    kw.setdefault("resident_mib", None)
    return display_card.decide(
        card.index, reading, idle_mib=200,
        state=lambda: display_card.display_state(smi, card.index, wsl2=False), **kw)


@pytest.mark.parametrize("card, reading, kw, taken, rule, needle", [
    (Card(0, used=3), _reading(3), {}, True, "idle", "idle: 3 MiB"),
    (Card(0), _reading(692), {}, True, "display", "taken under the display rule"),
    (Card(0), _reading(692 + display_card.SLACK_MIB), {"resident_mib": 692}, True, "display",
     "display resident 692 MiB + 128 MiB"),
    (Card(0), _reading(692 + display_card.SLACK_MIB + 1), {"resident_mib": 692}, False, None,
     "the display rule refuses it too: 821 MiB used, above the display resident 692 MiB"),
    (Card(0), _reading(692, util=11), {}, False, None, "utilisation 11 % above 10 %"),
    (headless(0), _reading(692), {}, False, None,
     "an unattributable resident is still a resident; not a display card: no display"),
    (Card(0), _reading(692, listed=True), {}, False, None, "carries a compute process"),
    (Card(0), _reading(692, listed=None), {}, False, None, "is unreadable"),
    (Card(0), _reading(692), {"set_aside": {0}}, False, None,
     "GPU 0 is set aside on this host"),
    (Card(0), _reading(3), {"set_aside": {0}}, False, None, "never available to the round"),
    (Card(0), _reading(692), {"display_allowed": False}, False, None,
     "the card was taken by the idle rule, which holds it"),
    (Card(0, used=3), _reading(3), {"display_allowed": False}, True, "idle", "idle: 3 MiB"),
    (headless(0, used=150), _reading(150, util=1), {}, False, None,
     "150 MiB used and 1 % utilisation with no listed process (idle: at most 200 MiB and 0 %)"),
    (headless(0, used=150), _reading(150), {}, True, "idle", "idle: 150 MiB, 0 % utilisation"),
], ids=["quiet", "display_first_reading", "display_at_slack", "display_past_slack",
        "display_busy", "no_display", "listed_process", "unreadable", "set_aside",
        "set_aside_even_if_quiet", "idle_holds_the_card", "idle_holds_a_quiet_card",
        "no_display_busy_within_the_bound", "no_display_quiet_within_the_bound"])
def test_the_rule_decides_one_reading_and_says_why(card, reading, kw, taken, rule, needle):
    decision = _decide(card, reading, **kw)
    assert (decision["taken"], decision["rule"]) == (taken, rule), decision
    assert needle in decision["text"], decision["text"]
    assert decision["refused"] == ([] if taken else [decision["text"]])


def test_the_display_state_is_asked_only_where_the_idle_rule_refuses():
    smi = FakeSmi(Card(0, used=3))
    display_card.decide(0, _reading(3), idle_mib=200, resident_mib=None,
                        state=lambda: display_card.display_state(smi, 0, wsl2=False))
    assert smi.display_queries == 0
    asked = FakeSmi(Card(0))
    display_card.decide(0, _reading(692, listed=True), idle_mib=200, resident_mib=None,
                        state=lambda: display_card.display_state(asked, 0, wsl2=False))
    assert asked.display_queries == 0       # a listed process refuses before the display rule


def test_the_record_of_a_taken_card_names_the_rule_and_is_checked_before_it_is_adopted():
    state = display_card.display_state(FakeSmi(Card(0)), 0, wsl2=False)
    record = display_card.admission_record(
        0, "display", idle_mib=200, readings=[_reading(692), _reading(694)],
        display={"state": state, "resident_mib": 692}, lane="gates_0")
    assert record["schema"] == display_card.PLACEMENT_SCHEMA and record["gpu"] == 0
    assert record["display"]["resident_mib"] == 692 and record["display"]["slack_mib"] == 128
    assert display_card.record_problems(record, 0) == []
    assert any("GPU 3" in p for p in display_card.record_problems(record, 3))
    assert display_card.record_problems({"schema": "x"}, 0)
    broken = dict(record, rule="other")
    assert any("rule 'other'" in p for p in display_card.record_problems(broken, 0))
    no_resident = dict(record, display={})
    assert any("no display resident" in p for p in display_card.record_problems(no_resident, 0))
    idle = display_card.admission_record(0, "idle", idle_mib=200, readings=[_reading(3)])
    assert "display" not in idle and display_card.record_problems(idle, 0) == []


# ----------------------------------------------------------------------------- the lane ----
def test_a_display_card_is_taken_under_the_display_rule_and_recorded(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi(Card(0)))
    assert lane.take_gpu() is True and lane.gpu == 0
    assert lane.display["resident_mib"] == 692 and lane.display["readings"] == [
        {"used_mib": 692, "util_pct": 0}, {"used_mib": 692, "util_pct": 0}]
    assert lane.display["state"]["said"] == ["display_attached Yes", "display_active Enabled"]
    label = (tmp_path / "locks" / "gpu_lock_0" / "label").read_text()
    assert "display_resident=692MiB" in label
    log = (tmp_path / "lane" / "progress.log").read_text()
    card = run_lane.N.card_id("GPU-0000-card")
    assert f"DISPLAY CARD 0 ({card}): display card (display_attached Yes" in log
    lane.row({"id": "x", "state": "released"})
    row = (tmp_path / "lane" / "steps.jsonl").read_text()
    assert '"display": {' in row and '"resident_mib": 692' in row
    # The card by its id, in the log and the row; its UUID only in the local lock.
    assert f'"card": "{card}"' in row and "GPU-0000-card" not in row + log
    assert (tmp_path / "locks" / "gpu_lock_0" / "card").read_text().split() == [
        card, "GPU-0000-card"]
    # the lane's record of how it took the card
    record = json.loads((tmp_path / "lane" / "placement.json").read_text())
    assert (record["rule"], record["gpu"], record["lane"]) == ("display", 0, "lane")
    assert record["display"]["resident_mib"] == 692 and record["idle_mib"] == 200
    assert "GPU-0000-card" not in json.dumps(record)       # a record names no card identity
    assert display_card.record_problems(record, 0) == []
    seen = {}

    def run_child(argv, env, log, timeout, label):
        seen["argv"] = argv
        return {"rc": 0, "timed_out": False, "elapsed_s": 0.0}
    monkeypatch.setattr(lane, "run_child", run_child)
    monkeypatch.setattr(lane, "base_env", lambda: {})
    lane.step_fleet({"id": "triton/fleet", "root": str(tmp_path / "fleet"), "gates": ["complex"],
                     "policy": "keep", "expect_s": 1})
    argv = seen["argv"]
    at = argv.index("--lane-placement")
    assert argv[at + 1] == str(tmp_path / "lane" / "placement.json") and at > argv.index("--")
    assert "--display-baseline-mib" not in argv
    assert argv[argv.index("--forbidden-gpus") + 1] == ""
    lane.release()


def test_a_card_the_caller_set_aside_is_never_taken(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi(Card(4, used=3)), avoid="4")
    assert lane.take_gpu() is False and not (tmp_path / "locks" / "gpu_lock_4").exists()
    assert "GPU 4 is set aside on this host" in (tmp_path / "lane" / "progress.log").read_text()
    free = _lane(tmp_path / "other", monkeypatch, FakeSmi(Card(4, used=3), Card(5, used=3)),
                 avoid="4")
    assert free.take_gpu() is True and free.gpu == 5
    free.release()


def test_the_lane_hands_its_set_aside_cards_to_the_fleet_driver(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi(Card(0, used=3), Card(4, used=3), Card(5)),
                 avoid="4,5")
    assert lane.take_gpu() is True and lane.gpu == 0
    seen = {}
    monkeypatch.setattr(lane, "run_child", lambda argv, *a: seen.update(argv=argv) or {
        "rc": 0, "timed_out": False, "elapsed_s": 0.0})
    monkeypatch.setattr(lane, "base_env", lambda: {})
    lane.step_fleet({"id": "triton/fleet", "root": str(tmp_path / "fleet"), "gates": ["complex"],
                     "policy": "keep", "expect_s": 1})
    assert seen["argv"][seen["argv"].index("--forbidden-gpus") + 1] == "4,5"
    lane.release()


def test_a_display_card_on_an_older_driver_or_a_wsl2_host_is_taken(tmp_path, monkeypatch):
    old = _lane(tmp_path / "old", monkeypatch,
                FakeSmi(Card(0, attached=None, active="Disabled", mode="Enabled")))
    assert old.take_gpu() is True and old.display["state"]["said"] == ["display_mode Enabled"]
    wsl = _lane(tmp_path / "wsl", monkeypatch, FakeSmi(headless(0)), wsl2=True)
    assert wsl.take_gpu() is True and wsl.display["state"]["said"] == ["a WSL2 host"]


def test_a_card_that_drives_no_display_keeps_the_idle_rule(tmp_path, monkeypatch):
    """Unattributable memory on a card with no display is another user's resident."""
    lane = _lane(tmp_path, monkeypatch, FakeSmi(headless(0)))
    assert lane.take_gpu() is False and lane.display is None
    assert not (tmp_path / "locks" / "gpu_lock_0").exists()
    assert not (tmp_path / "lane" / "placement.json").exists()
    log = (tmp_path / "lane" / "progress.log").read_text()
    assert "an unattributable resident is still a resident; not a display card: no display" in log


def test_the_idle_rule_takes_a_quiet_card_without_asking_the_display_state(tmp_path,
                                                                        monkeypatch):
    smi = FakeSmi(Card(0, used=3), headless(1, used=5))
    lane = _lane(tmp_path, monkeypatch, smi)
    assert lane.take_gpu() is True and lane.gpu == 0 and lane.display is None
    assert smi.display_queries == 0
    assert "display_resident" not in (tmp_path / "locks" / "gpu_lock_0" / "label").read_text()
    lane.row({"id": "x", "state": "released"})
    assert '"display"' not in (tmp_path / "lane" / "steps.jsonl").read_text()
    record = json.loads((tmp_path / "lane" / "placement.json").read_text())
    assert record["rule"] == "idle" and "display" not in record


@pytest.mark.parametrize("change, taken", [
    (lambda c: setattr(c, "used", 692 + display_card.SLACK_MIB), True),
    (lambda c: setattr(c, "used", 692 + display_card.SLACK_MIB + 1), False),
    (lambda c: setattr(c, "util", display_card.UTIL_MAX_PCT), True),
    (lambda c: setattr(c, "util", display_card.UTIL_MAX_PCT + 1), False),
    (lambda c: c.apps.update({4242: 258}), False),
], ids=["memory_at_slack", "memory_past_slack", "util_at_bound", "util_past_bound",
        "process_arrives"])
def test_the_second_reading_is_held_to_the_first(tmp_path, monkeypatch, change, taken):
    card = Card(0)
    lane = _lane(tmp_path, monkeypatch, FakeSmi(card), between=lambda: change(card))
    assert lane.take_gpu() is taken
    assert (tmp_path / "locks" / "gpu_lock_0").exists() is taken


def test_a_busy_first_reading_refuses_a_display_card(tmp_path, monkeypatch):
    card = Card(0, util=display_card.UTIL_MAX_PCT + 1)
    lane = _lane(tmp_path, monkeypatch, FakeSmi(card), between=lambda: setattr(card, "util", 0))
    assert lane.take_gpu() is False


def test_another_user_s_compute_process_refuses_a_display_card(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi(Card(0, apps={4030427: 4970})), wsl2=True)
    assert lane.take_gpu() is False and lane.display is None
    assert "GPU 0 carries a compute process" in (tmp_path / "lane" / "progress.log").read_text()


def test_the_share_opt_in_admits_a_display_card_as_before(tmp_path, monkeypatch):
    card = Card(0, used=692 + 4970, apps={4030427: 4970})
    lane = _lane(tmp_path, monkeypatch, FakeSmi(card), share=[f"{card.uuid}:4030427"],
                 need=8192)
    assert lane.take_gpu() is True and lane.display is None
    assert "shared=GPU-0000-card:4030427" in (tmp_path / "locks" / "gpu_lock_0" /
                                              "label").read_text()
    record = json.loads((tmp_path / "lane" / "placement.json").read_text())
    assert record["rule"] == "share"
    other = _lane(tmp_path / "x", monkeypatch, FakeSmi(Card(0, apps={4030427: 4970, 99: 1})),
                  share=["GPU-0000-card:4030427"], need=8192)
    assert other.take_gpu() is False


# ------------------------------------------------- every step is placed by the same rule ----
def _step(kind="gate", id_="s"):
    return {"id": id_, "kind": kind}


def _taken(tmp_path, monkeypatch, *cards, **kw):
    smi = FakeSmi(*cards)
    lane = _lane(tmp_path, monkeypatch, smi, **kw)
    assert lane.take_gpu() is True
    lane.args.step_patience = 3
    return lane, smi


def test_a_step_runs_when_the_rule_still_takes_the_card(tmp_path, monkeypatch):
    lane, _smi = _taken(tmp_path, monkeypatch, Card(0))
    assert lane.place_step(_step()) is None
    lane.release()


def test_a_process_that_arrives_between_two_steps_stops_the_next_one(tmp_path, monkeypatch):
    card = Card(0, used=3)
    lane, _smi = _taken(tmp_path, monkeypatch, card)
    assert lane.place_step(_step()) is None
    card.apps[4242] = 258                       # another user's process, listed
    why = lane.place_step(_step())
    assert why and "GPU 0 carries a compute process" in why
    said = (tmp_path / "lane" / "progress.log").read_text()
    assert "the card is not free" in said and "waiting up to 3 s" in said
    del card.apps[4242]                         # it leaves: the next step is placed
    assert lane.place_step(_step()) is None
    lane.release()


def test_a_step_waits_for_the_card_and_runs_when_it_frees(tmp_path, monkeypatch):
    card = Card(0)
    lane, _smi = _taken(tmp_path, monkeypatch, card)
    card.used = 692 + display_card.SLACK_MIB + 50                  # a program took memory
    slept = []

    def free_after_two(_seconds):
        slept.append(1)
        if len(slept) == 2:
            card.used = 700
    monkeypatch.setattr(run_lane.time, "sleep", free_after_two)
    assert lane.place_step(_step()) is None and len(slept) == 2
    lane.release()


def test_a_display_card_is_held_to_the_resident_it_was_taken_at(tmp_path, monkeypatch):
    card = Card(0)
    lane, _smi = _taken(tmp_path, monkeypatch, card)
    card.used = 692 + display_card.SLACK_MIB + 1
    why = lane.place_step(_step())
    assert why and "above the display resident 692 MiB" in why
    card.util = display_card.UTIL_MAX_PCT + 1
    card.used = 692
    assert "utilisation 11 % above 10 %" in lane.place_step(_step())
    lane.release()


def test_a_card_taken_as_idle_is_held_to_the_idle_rule(tmp_path, monkeypatch):
    """The display rule does not take over later: memory that arrives on a card the lane took
    as idle is another resident, whatever the card's display state."""
    card = Card(0, used=3)
    lane, smi = _taken(tmp_path, monkeypatch, card)
    asked = smi.display_queries
    card.used = 692                                  # unlisted memory, as a desktop holds
    why = lane.place_step(_step())
    assert why and "the card was taken by the idle rule, which holds it" in why
    assert smi.display_queries == asked
    lane.release()


def test_a_step_that_does_not_use_the_gpu_is_not_placed(tmp_path, monkeypatch):
    card = Card(0, used=3)
    lane, _smi = _taken(tmp_path, monkeypatch, card)
    card.apps[4242] = 258
    assert lane.place_step(_step("e2e_verdict")) is None
    assert lane.place_step(_step("route"))
    lane.release()


def test_a_shared_card_is_held_to_the_share_admission(tmp_path, monkeypatch):
    card = Card(0, used=692 + 4970, apps={4030427: 4970})
    lane, _smi = _taken(tmp_path, monkeypatch, card, share=[f"{card.uuid}:4030427"], need=8192)
    assert lane.shared_card and lane.place_step(_step()) is None
    card.apps[99] = 1                                # an unnamed process joins
    why = lane.place_step(_step())
    assert why and "not named by --share-idle-foreign" in why
    lane.release()


def test_a_card_the_caller_gave_is_placed_at_the_first_step(tmp_path, monkeypatch):
    card = Card(0)
    lane = _lane(tmp_path, monkeypatch, FakeSmi(card))
    lane.gpu, lane.args.step_patience = 0, 2
    assert lane.place_step(_step()) is None          # the first reading sets the rule
    assert lane.card_rule.rule == "display" and lane.card_rule.resident == 692
    card.used = 692 + display_card.SLACK_MIB + 1
    assert lane.place_step(_step())
    busy = _lane(tmp_path / "busy", monkeypatch, FakeSmi(headless(0)))
    busy.gpu, busy.args.step_patience = 0, 2
    assert "not a display card" in busy.place_step(_step())


def test_the_lane_marks_a_refused_step_and_places_the_next(tmp_path, monkeypatch):
    card = Card(0, used=3)
    lane, _smi = _taken(tmp_path, monkeypatch, card)
    lane.args.step_patience = 1
    ran = []
    monkeypatch.setattr(lane, "step_toolchain", lambda step: ran.append(step["id"]) or {
        "state": "released"})
    monkeypatch.setattr(lane, "user_site_problems", lambda: [])
    monkeypatch.setattr(lane, "toolchain", lambda: {"problems": [], "line": ""})
    monkeypatch.setattr(lane.legs, "launcher_line", lambda toolchain: "toolchain")
    monkeypatch.setattr(lane, "take_gpu", lambda: True)
    plan = [{"id": "a", "kind": "toolchain"}, {"id": "b", "kind": "toolchain"}]
    stages = {"n": 0}
    real_place = lane.place_step

    def arrives_before_b(step):
        if step["id"] == "b" and stages["n"] == 0:
            stages["n"] = 1
            card.apps[4242] = 258
        return real_place(step)
    monkeypatch.setattr(lane, "place_step", arrives_before_b)
    lane.run(plan)
    rows = {json.loads(l)["id"]: json.loads(l) for l in
            (tmp_path / "lane" / "steps.jsonl").read_text().splitlines()}
    assert ran == ["a"] and rows["a"]["state"] == "released"
    assert rows["b"]["state"] == "refused" and "not placed: GPU 0 carries a compute process" \
        in rows["b"]["reasons"][0]
    lane.release()


# ----------------------------------------------------------------------- the fleet driver ----
@pytest.fixture
def driver(monkeypatch):
    """The fleet driver on a fake ``nvidia-smi`` (no WSL2 host), its placement made by ``main``."""
    def use(smi, *, wsl2=False):
        monkeypatch.setattr(fleet.display_card, "run_smi", smi)
        monkeypatch.setattr(fleet.display_card, "wsl2_host", lambda *a: wsl2)
        monkeypatch.setattr(fleet, "log", lambda message: None)
        monkeypatch.setattr(fleet, "PLACEMENT", None)
        return smi
    return use


def _placement(smi, **kw):
    kw.setdefault("wsl2", False)
    return display_card.Placement(smi, say=lambda message: None, **kw)


def test_the_driver_shares_the_round_s_one_copy_of_the_rule():
    assert fleet.display_card is display_card
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import drive_triton_weld_gates as d; "
             "print(d.display_card.__file__); print(any(p.endswith('1.0') for p in sys.path))")
    done = subprocess.run([sys.executable, "-c", probe, str(HARNESS)], capture_output=True,
                          text=True, check=True)
    loaded, on_path = done.stdout.split()
    assert Path(loaded) == ROUND / "display_card.py" and on_path == "False"


def test_the_driver_keeps_no_placement_logic_of_its_own():
    for name in ("idle_rule", "IDLE_MIB_CEILING", "device_is_empty", "_nvidia_smi"):
        assert not hasattr(fleet, name), name
    source = (HARNESS / "drive_triton_weld_gates.py").read_text(encoding="utf-8")
    for needle in ("query-compute-apps", "memory.used,utilization", "display_attached",
                   "unattributable"):
        assert needle not in source.replace("--query-gpu=index,name,memory.used,utilization.gpu",
                                            ""), needle
    for name in ("FleetPlacement", "fleet_idle_check", "FLEET_IDLE_MIB", "LANE_IDLE_MIB"):
        assert not hasattr(display_card, name), name


def test_the_lane_and_the_driver_say_the_same_thing_about_a_card_they_refuse(tmp_path,
                                                                           monkeypatch):
    cases = [headless(0), Card(0, apps={4030427: 300}), Card(0, util=50)]
    for k, card in enumerate(cases):
        smi = FakeSmi(card)
        lane = _lane(tmp_path / str(k), monkeypatch, smi)
        assert lane.take_gpu() is False
        placement = _placement(smi)
        placement.start(0)
        taken, why = placement.check(0)
        assert taken is False
        assert lane.decisions[0]["text"] == why, (lane.decisions[0]["text"], why)
        assert why in (tmp_path / str(k) / "lane" / "progress.log").read_text()


def test_the_driver_takes_a_display_card_under_the_lane_s_record(driver):
    card = Card(0)
    smi = FakeSmi(card)
    state = display_card.display_state(smi, 0, wsl2=False)
    record = display_card.admission_record(
        0, "display", idle_mib=200, readings=[_reading(692)] * 2,
        display={"state": state, "resident_mib": 692}, lane="gates_0")
    placement = _placement(smi)
    placement.adopt(record, 0)
    assert placement.rule == "display" and placement.resident == 692
    taken, why = placement.check(0)
    assert taken is True and why.startswith("GPU 0 taken under the display rule: display card")
    assert placement.last["rule"] == "display" and placement.last["display"]["resident_mib"] == 692
    card.used = 692 + display_card.SLACK_MIB + 1
    taken, why = placement.check(0)
    assert taken is False and "the display rule refuses it too" in why
    assert placement.last["taken"] is False
    card.used, card.util = 692, display_card.UTIL_MAX_PCT + 1
    assert placement.check(0)[0] is False


def test_the_driver_still_refuses_a_listed_process_and_a_device_set_aside(driver):
    smi = FakeSmi(Card(0, apps={4030427: 300}), Card(4))
    placement = _placement(smi, wsl2=True, set_aside={4})
    placement.start(0)
    taken, why = placement.check(0)
    assert taken is False and why.startswith("GPU 0 carries a compute process")
    asked = smi.display_queries
    taken, why = placement.check(4)
    assert taken is False and "set aside" in why and smi.display_queries == asked


def test_a_run_alone_takes_the_rule_from_its_first_reading(driver):
    card = Card(0)
    placement = _placement(FakeSmi(card))
    decision = placement.start(0)
    assert decision["rule"] == "display" and placement.resident == 692
    assert placement.resident_from == "this run's first reading"
    card.used = 692 + display_card.SLACK_MIB + 1
    assert placement.check(0)[0] is False
    card.used = 700
    assert placement.check(0)[0] is True
    quiet = _placement(FakeSmi(Card(0, used=3)))
    assert quiet.start(0)["rule"] == "idle" and quiet.rule == "idle" and quiet.resident is None
    # a card nothing has taken yet is taken by the first reading that passes
    waiting = _placement(FakeSmi(Card(0, apps={1: 1})))
    waiting.start(0)
    assert waiting.rule is None and waiting.check(0)[0] is False


def test_the_driver_holds_a_card_to_the_idle_bound_the_lane_recorded(driver):
    card = headless(0, used=100)
    record = display_card.admission_record(0, "idle", idle_mib=64, readings=[_reading(3)] * 2)
    placement = _placement(FakeSmi(card))
    placement.adopt(record, 0)
    assert placement.rule == "idle" and placement.idle_mib == 64
    taken, why = placement.check(0)
    assert taken is False and "(idle: at most 64 MiB and 0 %)" in why
    card.used = 64
    assert placement.check(0)[0] is True


def test_a_lane_record_of_a_share_takes_no_rule_for_a_driver_that_never_shares(driver):
    card = Card(0, used=692 + 4970, apps={4030427: 4970})
    record = display_card.admission_record(0, "share", idle_mib=200, readings=[_reading()])
    placement = _placement(FakeSmi(card))
    placement.adopt(record, 0)
    assert placement.rule is None
    taken, why = placement.check(0)
    assert taken is False and "GPU 0 carries a compute process" in why


class _ExitedGate:
    """A gate process that exited 0 at once and wrote nothing."""

    def __init__(self, *args, **kwargs):
        self.returncode = 0
        self.stdout = io.BytesIO(b"")       # a driver that pipes a gate's output reads this

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _GatesExit:
    """The driver's ``subprocess`` with every gate process replaced by :class:`_ExitedGate`."""

    Popen = _ExitedGate

    def __getattr__(self, name):
        return getattr(subprocess, name)


def _campaign(tmp_path, monkeypatch, driver, smi, *extra, forbidden=""):
    driver(smi)
    monkeypatch.setattr(fleet, "record_device", lambda python, out_root, env: (
        {"compute_capability": "8.6", "device": "NVIDIA GeForce RTX 3090"}, None))
    monkeypatch.setattr(fleet, "subprocess", _GatesExit())
    out = tmp_path / "campaign"
    rc = fleet.main(["--gpu", "0", "--forbidden-gpus", forbidden, "--out-root", str(out),
                     "--gates", "fused_electric", "--patience", "0", *extra])
    return rc, out


def _record_file(tmp_path, smi, rule="display", resident=692, gpu=0):
    state = display_card.display_state(smi, 0, wsl2=False)
    record = display_card.admission_record(
        gpu, rule, idle_mib=200, readings=[_reading(resident)] * 2, lane="gates_0",
        display={"state": state, "resident_mib": resident} if rule == "display" else None)
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(record))
    return path


def test_a_standalone_driver_run_takes_a_display_card_and_records_it(tmp_path, monkeypatch,
                                                                     driver):
    rc, out = _campaign(tmp_path / "alone", monkeypatch, driver, FakeSmi(Card(0)))
    assert rc == 0
    summary = json.loads((out / "campaign.json").read_text())
    row = summary["rows"][0]
    assert row["placement"].startswith("GPU 0 taken under the display rule")
    assert "refused_by_driver" not in row and row["placement_check"]["taken"] is True
    assert row["placement_check"]["display"]["resident_mib"] == 692
    assert summary["placement"]["source"] == "this run's first reading"
    assert summary["placement"]["record"]["rule"] == "display"
    assert summary["placement"]["record"]["display"]["resident_mib"] == 692


def test_a_driver_in_a_lane_runs_on_the_lane_s_record_and_records_it(tmp_path, monkeypatch,
                                                                     driver):
    smi = FakeSmi(Card(0, used=700))
    rc, out = _campaign(tmp_path / "lane", monkeypatch, driver, smi, "--lane-placement",
                        str(_record_file(tmp_path, smi)))
    assert rc == 0
    summary = json.loads((out / "campaign.json").read_text())
    row = summary["rows"][0]
    assert summary["placement"]["source"] == "the lane" and summary["placement"]["held_to"] == \
        "display"
    assert summary["placement"]["record"]["lane"] == "gates_0"
    assert row["placement_check"]["display"]["resident_from"] == "the lane"
    assert row["placement_check"]["used_mib"] == 700


def test_a_driver_in_a_lane_refuses_a_gate_when_the_card_moves_off_the_lane_s_rule(
        tmp_path, monkeypatch, driver):
    smi = FakeSmi(Card(0, used=692 + display_card.SLACK_MIB + 1))
    rc, out = _campaign(tmp_path / "lane", monkeypatch, driver, smi, "--lane-placement",
                        str(_record_file(tmp_path, smi)))
    row = json.loads((out / "campaign.json").read_text())["rows"][0]
    assert "the display rule refuses it too" in row["refused_by_driver"]
    assert row["placement_check"]["taken"] is False


def test_a_lane_record_that_does_not_fit_the_run_aborts_it(tmp_path, monkeypatch, driver):
    smi = FakeSmi(Card(0), Card(1))
    wrong_card = _record_file(tmp_path, smi, gpu=1)
    rc, out = _campaign(tmp_path / "a", monkeypatch, driver, smi, "--lane-placement",
                        str(wrong_card))
    assert rc == 2 and not (out / "campaign.json").exists()
    unreadable = tmp_path / "bad.json"
    unreadable.write_text("not json")
    assert _campaign(tmp_path / "b", monkeypatch, driver, smi, "--lane-placement",
                     str(unreadable))[0] == 2
    assert _campaign(tmp_path / "c", monkeypatch, driver, smi, "--lane-placement",
                     str(tmp_path / "absent.json"))[0] == 2


@pytest.mark.parametrize("card, needle", [
    (headless(0), "an unattributable resident is still a resident"),
    (Card(0, apps={4030427: 4970}), "carries a compute process"),
], ids=["no_display", "another_users_process"])
def test_a_standalone_driver_run_still_refuses_the_shared_machine_cases(tmp_path, monkeypatch,
                                                                        driver, card, needle):
    rc, out = _campaign(tmp_path, monkeypatch, driver, FakeSmi(card))
    row = json.loads((out / "campaign.json").read_text())["rows"][0]
    assert needle in row["refused_by_driver"] and row["placement_check"]["taken"] is False
    # Another account's process is counted, never named by its process id.
    written = (out / "campaign.json").read_text() + (out / "gates.jsonl").read_text()
    assert "4030427" not in written


def test_the_lane_and_a_driver_run_alone_take_a_card_by_the_one_idle_bound(tmp_path,
                                                                           monkeypatch, driver):
    """A card that drives no display, at 100 MiB and 0 %: inside the one idle bound
    (display_card.IDLE_MIB, 200 MiB), above the 64 MiB the driver kept before it shared it."""
    card = headless(0, used=100)
    lane = _lane(tmp_path / "lane", monkeypatch, FakeSmi(card))
    assert lane.take_gpu() is True and lane.gpu == 0 and lane.display is None
    lane.release()
    rc, out = _campaign(tmp_path / "alone", monkeypatch, driver, FakeSmi(card))
    assert rc == 0 and fleet.PLACEMENT.idle_mib == display_card.IDLE_MIB
    summary = json.loads((out / "campaign.json").read_text())
    assert summary["placement"]["record"]["rule"] == "idle"
    assert summary["placement"]["record"]["idle_mib"] == display_card.IDLE_MIB
    assert summary["rows"][0]["placement_check"]["taken"] is True


def test_a_standalone_driver_run_refuses_a_card_set_aside_at_once(tmp_path, monkeypatch,
                                                                  driver):
    rc, out = _campaign(tmp_path, monkeypatch, driver, FakeSmi(Card(0)), forbidden="0")
    assert rc == fleet.EXIT_CANNOT_CERTIFY and not (out / "campaign.json").exists()


def test_the_driver_hands_the_cards_it_was_told_to_set_aside_to_the_rule(tmp_path, monkeypatch,
                                                                         driver):
    rc, _out = _campaign(tmp_path, monkeypatch, driver, FakeSmi(Card(0), Card(4)),
                         forbidden="4")
    assert rc == 0
    taken, why = fleet.PLACEMENT.check(4)           # the placement main made for this run
    assert taken is False and "GPU 4 is set aside on this host" in why


class _Clock:
    """``time`` for :func:`fleet.wait_for_device`: ``sleep`` advances ``time`` and runs a hook."""

    def __init__(self, on_sleep=None):
        self.now, self.slept, self.on_sleep = 1000.0, [], on_sleep

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.slept.append(seconds)
        if self.on_sleep:
            self.on_sleep(len(self.slept))


def test_the_driver_waits_for_a_busy_card_and_then_refuses_it(monkeypatch, driver):
    card = Card(0, apps={4030427: 300})
    driver(FakeSmi(card))
    fleet.PLACEMENT = _placement(FakeSmi(card))
    monkeypatch.setattr(fleet, "time", _Clock())
    taken, why = fleet.wait_for_device(0, 150, forbidden=frozenset(), poll_s=60)
    assert taken is False and "GPU 0 carries a compute process" in why
    assert fleet.time.slept == [60, 60, 60]                    # 150 s of patience, then refused
    card.apps[4030427] = 300
    clock = _Clock(on_sleep=lambda n: card.apps.clear())         # the other process leaves
    monkeypatch.setattr(fleet, "time", clock)
    taken, _why = fleet.wait_for_device(0, 150, forbidden=frozenset(), poll_s=60)
    assert taken is True and clock.slept == [60]
    # a card that is set aside is never waited for
    fleet.PLACEMENT = _placement(FakeSmi(Card(4)), set_aside={4})
    monkeypatch.setattr(fleet, "time", _Clock())
    assert fleet.wait_for_device(4, 150, forbidden=frozenset({4}))[0] is False
    assert fleet.time.slept == []


def test_the_report_names_the_decision(monkeypatch, capsys):
    smi = FakeSmi(Card(0))
    monkeypatch.setattr(display_card, "run_smi", smi)
    monkeypatch.setattr(display_card, "wsl2_host", lambda *a: False)
    monkeypatch.setattr(display_card.time, "sleep", lambda s: None)
    smi.cards[0].attached = "Yes"

    def name_query(*query):
        if query[1:2] == ("--query-gpu=name,compute_cap,memory.total",):
            return "NVIDIA GeForce RTX 3090, 8.6, 24576 MiB\n"
        return smi(*query)
    monkeypatch.setattr(display_card, "run_smi", name_query)
    assert display_card.main(["--gpu", "0", "--poll", "1"]) == 0
    out = capsys.readouterr().out
    assert "LANE: taken under the display rule" in out and "FLEET: taken" in out
    assert out.strip().endswith("is taken under the display rule")
    smi.cards[0] = headless(0)
    assert display_card.main(["--gpu", "0", "--poll", "1"]) == 1
    out = capsys.readouterr().out
    assert "LANE: refused" in out and "unattributable resident" in out
    # a process on the card is counted, never named
    smi.cards[0] = Card(0, apps={4030427: 300})
    assert display_card.main(["--gpu", "0", "--poll", "1"]) == 1
    out = capsys.readouterr().out
    assert "compute processes listed: 1" in out and "4030427" not in out
    assert "GPU 0 carries a compute process" in out


class _FleetClock(_Clock):
    """:class:`_Clock` for the whole driver: what it does not fake is the real ``time``."""

    def __getattr__(self, name):
        import time
        return getattr(time, name)


def test_the_fleet_driver_waits_for_each_gate_by_the_lane_s_step_wait(tmp_path, monkeypatch,
                                                                     driver):
    card = Card(0, used=3)
    lane = _lane(tmp_path, monkeypatch, FakeSmi(card))
    assert lane.take_gpu() is True and lane.gpu == 0
    lane.args.step_patience, lane.args.poll = 45, 3
    assert lane.step_wait() == (45, 3)
    seen = {}
    monkeypatch.setattr(lane, "run_child", lambda argv, *a: seen.update(argv=argv) or {
        "rc": 0, "timed_out": False, "elapsed_s": 0.0})
    monkeypatch.setattr(lane, "base_env", lambda: {})
    lane.step_fleet({"id": "triton/fleet", "root": str(tmp_path / "fleet"),
                     "gates": ["fused_electric"], "policy": "keep", "expect_s": 1})
    lane.release()
    argv = seen["argv"]
    tail = argv[argv.index("--") + 1:]
    assert tail[tail.index("--patience") + 1] == "45" and tail[tail.index("--poll") + 1] == "3"
    # the driver, run with exactly that tail, waits for a busy card by those two numbers
    card.apps[4030427] = 300
    driver(FakeSmi(card))
    monkeypatch.setattr(fleet, "record_device", lambda python, out_root, env: (
        {"compute_capability": "8.6", "device": "NVIDIA GeForce RTX 3090"}, None))
    monkeypatch.setattr(fleet, "subprocess", _GatesExit())
    clock = _FleetClock()
    monkeypatch.setattr(fleet, "time", clock)
    assert fleet.main(tail) == 0
    row = json.loads((tmp_path / "fleet" / "campaign.json").read_text())["rows"][0]
    assert "GPU 0 carries a compute process" in row["refused_by_driver"]
    assert clock.slept == [3] * 15                       # 45 s of patience, a reading every 3 s
