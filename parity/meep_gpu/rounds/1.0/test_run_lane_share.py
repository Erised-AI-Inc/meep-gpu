"""A shared card, by opt-in only (``run_lane.py --share-idle-foreign``; owner ruling R28).

Host-only: ``nvidia-smi`` is a fake that answers each query from a table. Without the flag a
card holding any compute app is never taken, as before. With it, the card the flag names is
taken only when every app on it is a named PID, the named PIDs and the card read 0 %
utilisation and its free memory reaches ``--share-min-free-mib``, on both readings; the lane
records the shared card in its log and its lock's label. ``round.py`` hands the flag to every
lane of the gate stage and refuses it elsewhere or without the memory bound.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

ROUND = Path(__file__).resolve().parent
HARNESS = ROUND.parent.parent
for _path in (HARNESS, ROUND):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import round as round_driver  # noqa: E402
import run_lane  # noqa: E402

FREE_CARD, SHARED_CARD = 6, 7
SHARED_UUID, OTHER_UUID = "GPU-7fe9-shared", "GPU-0000-other"
IDLE_PID = 4030427
PMON_HEAD = "# gpu         pid  type    sm   mem   enc   dec   command\n" \
            "# Idx           #   C/G     %     %     %     %   name\n"


class FakeSmi:
    """``nvidia-smi`` for two cards: card 6 empty, card 7 holding ``apps``."""

    def __init__(self, apps=None, sm=None, free=43000, util=0):
        self.apps = {IDLE_PID: 4970} if apps is None else apps
        self.sm = sm or {}
        self.free, self.util = free, util

    def __call__(self, *query: str) -> str:
        if query[0] == "--query-gpu=index,uuid,memory.used,utilization.gpu":
            used = sum(self.apps.values())
            return f"6, GPU-6666-free, 3, 0\n7, {SHARED_UUID}, {used}, {self.util}\n"
        if query[0] == "--query-compute-apps=gpu_uuid":
            return "".join(f"{SHARED_UUID}\n" for _ in self.apps)
        if query[0] == "--query-compute-apps=gpu_uuid,pid,used_memory":
            return "".join(f"{SHARED_UUID}, {pid}, {mib}\n" for pid, mib in self.apps.items())
        if query[0] == "pmon":
            assert query[1:3] == ("-i", str(SHARED_CARD))
            return PMON_HEAD + "".join(
                f"    7 {pid:>11}     C {self.sm.get(pid, '-'):>5}     -     -     -   python\n"
                for pid in self.apps)
        if query[0] == "--query-gpu=index,memory.free":
            return f"6, 48500\n7, {self.free}\n"
        raise AssertionError(f"unexpected query {query}")


def _lane(tmp_path: Path, monkeypatch, smi, share=(), need=0, want=None) -> run_lane.Lane:
    args = argparse.Namespace(
        lane_dir=str(tmp_path / "lane"), repo=str(tmp_path), results=str(tmp_path / "results"),
        python=sys.executable, gpu=None, subs="", libcuda_stub="", corpus_root="",
        min_timeout=1800, timeout_factor=4, heartbeat=120, avoid="", idle_mib=200,
        share_idle_foreign=list(share), share_min_free_mib=need, lock_dir=str(tmp_path / "locks"),
        max_locks=4, poll=1, lock_patience=1, label="1.0 round test", want_gpu=want)
    lane = run_lane.Lane(args)
    monkeypatch.setattr(lane, "smi", smi)
    monkeypatch.setattr(run_lane.time, "sleep", lambda _s: None)
    return lane


def test_without_the_flag_a_card_holding_any_compute_app_is_never_taken(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi())
    assert lane.idle_cards() == [FREE_CARD]
    assert lane.share_seen == {}


def test_the_named_idle_pid_admits_its_card_with_the_memory_there(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi(), share=[f"{SHARED_UUID}:{IDLE_PID}"],
                 need=8192 + 300)
    assert lane.idle_cards() == [FREE_CARD, SHARED_CARD]
    seen = lane.share_seen[SHARED_CARD]
    assert seen["refused"] == [] and seen["pids"] == {IDLE_PID: "4970"}
    assert seen["free_mib"] == 43000 and seen["need_mib"] == 8492


@pytest.mark.parametrize("smi, needle", [
    (FakeSmi(sm={IDLE_PID: 3}), "computing"),
    (FakeSmi(apps={IDLE_PID: 4970, 999: 800}), "[999] not named"),
    (FakeSmi(free=8000), "free memory 8000 MiB below 8492 MiB"),
    (FakeSmi(util=4), "card utilisation 4 %"),
], ids=["named_pid_computing", "unnamed_pid", "memory_short", "card_busy"])
def test_a_shared_card_is_refused_by_name_when_any_condition_fails(tmp_path, monkeypatch, smi,
                                                                   needle):
    lane = _lane(tmp_path, monkeypatch, smi, share=[f"{SHARED_UUID}:{IDLE_PID}"], need=8492)
    assert lane.idle_cards() == [FREE_CARD]
    assert any(needle in why for why in lane.share_seen[SHARED_CARD]["refused"])


def test_the_flag_admits_only_the_card_it_names(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi(), share=[f"{OTHER_UUID}:{IDLE_PID}"],
                 need=8492)
    assert lane.idle_cards() == [FREE_CARD]


def test_a_shared_card_is_taken_on_two_readings_and_recorded(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, FakeSmi(), share=[f"{SHARED_UUID}:{IDLE_PID}"],
                 need=8492, want=SHARED_CARD)
    assert lane.take_gpu() is True
    assert lane.gpu == SHARED_CARD
    label = (tmp_path / "locks" / "gpu_lock_7" / "label").read_text()
    assert f"shared={SHARED_UUID}:{IDLE_PID}" in label
    log = (tmp_path / "lane" / "progress.log").read_text()
    assert f"SHARED CARD {SHARED_CARD} ({SHARED_UUID})" in log and str(IDLE_PID) in log
    lane.release()
    assert not (tmp_path / "locks" / "gpu_lock_7").exists()


def test_a_shared_card_that_fails_on_the_second_reading_is_not_taken(tmp_path, monkeypatch):
    smi = FakeSmi()
    calls = {"pmon": 0}

    def flipping(*query):
        if query[0] == "pmon":
            calls["pmon"] += 1
            smi.sm = {IDLE_PID: 0 if calls["pmon"] == 1 else 12}
        return smi(*query)

    lane = _lane(tmp_path, monkeypatch, flipping, share=[f"{SHARED_UUID}:{IDLE_PID}"],
                 need=8492, want=SHARED_CARD)
    assert lane.take_gpu() is False
    assert not (tmp_path / "locks" / "gpu_lock_7").exists()


@pytest.mark.parametrize("value", ["4030427", "GPU-x:", "GPU-x:12a", "card7:4030427"])
def test_a_malformed_share_value_is_refused_by_name(value):
    with pytest.raises(ValueError, match="--share-idle-foreign"):
        run_lane.share_specs([value])


def test_the_lane_refuses_the_flag_without_the_memory_bound(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("[]\n")
    with pytest.raises(SystemExit):
        run_lane.main(["--plan", str(plan), "--lane-dir", str(tmp_path / "lane"), "--repo",
                       str(tmp_path), "--results", str(tmp_path), "--lock-dir", str(tmp_path),
                       "--share-idle-foreign", f"{SHARED_UUID}:{IDLE_PID}"])


def _round_args(**extra):
    base = dict(repo="/r", python="/p", lock_dir="/l", max_locks=4, stamp="2026-10-15_10_cc86_c9",
                avoid="", libcuda_stub="", corpus_root="", timeout_floor=[], stage="gates",
                share_idle_foreign=[], share_min_free_mib=0)
    base.update(extra)
    return argparse.Namespace(**base)


def test_round_hands_the_flag_to_every_lane_and_nothing_without_it(tmp_path):
    plain = round_driver.lane_argv(_round_args(), tmp_path / "plan.json", tmp_path, tmp_path, {}, 7)
    assert "--share-idle-foreign" not in plain and "--share-min-free-mib" not in plain
    shared = round_driver.lane_argv(_round_args(share_idle_foreign=[f"{SHARED_UUID}:{IDLE_PID}"],
                                                share_min_free_mib=8492),
                                    tmp_path / "plan.json", tmp_path, tmp_path, {}, 7)
    k = shared.index("--share-idle-foreign")
    assert shared[k + 1] == f"{SHARED_UUID}:{IDLE_PID}"
    assert shared[shared.index("--share-min-free-mib") + 1] == "8492"


@pytest.mark.parametrize("extra, needle", [
    (dict(stage="route", share_min_free_mib=8492), "gate stage only"),
    (dict(share_min_free_mib=0), "needs --share-min-free-mib"),
    (dict(share_idle_foreign=["7:4030427"], share_min_free_mib=8492), "expected <uuid>"),
], ids=["route_stage", "no_memory_bound", "malformed"])
def test_round_refuses_the_flag_by_name(extra, needle):
    args = _round_args(**{"share_idle_foreign": [f"{SHARED_UUID}:{IDLE_PID}"], **extra})
    assert needle in round_driver.share_problem(args)
    assert round_driver.share_problem(_round_args()) is None
