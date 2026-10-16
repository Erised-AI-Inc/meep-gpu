"""The 1.0 round's route stages give their route and licence legs a watchdog the measured walls fit.

Route campaign names are one-shot: a route leg killed by its watchdog cannot rerun under the
same stamp, so a stage's watchdog must not fire below walls the round has measured, and these
tests hold it at three times or more. The NVIDIA route stage writes ``timeout_s`` into each
route and licence step, which ``run_lane.py`` honours before its own default; the Metal route
stage hands its one campaign item a watchdog derived from its legs' measured walls
(host-only).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROUND = Path(__file__).resolve().parent / "rounds" / "1.0"
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

import legs as L  # noqa: E402
import metal_round  # noqa: E402
import round as round_driver  # noqa: E402
import run_lane  # noqa: E402

#: Walls the 1.0 final pre-certification pass measured for every route and licence leg
#: (2026-10-10, tree b676c5f5, one RTX A6000 shared by every route lane), in seconds: the
#: record the leg tables are priced from, kept here as the oracle they are checked against.
MEASURED_ROUTE_WALLS = {
    "c_preflight": 469.1, "c_preflight_alone": 344.9, "cuda_shipped": 2526.1,
    "cuda_harness_keep": 1958.6, "cuda_flush": 2791.9, "cuda_default_precedence": 2659.0,
    "cuda_alone": 1980.7, "t_preflight": 671.6, "shipped": 3180.3, "harness_keep": 1572.6,
    "flush": 3065.3}
MEASURED_LICENCE_WALLS = {"preflight": 60.2, "ship": 376.8, "null": 304.9}
#: The same pass's held Metal route campaign on one Apple M1 Max (tree 15da8cd4, 3 lanes,
#: a loaded Mac): the four legs and the campaign.
MEASURED_METAL_WALLS = {"band_witness": 883, "harness_keep": 6479, "shipped": 5283,
                        "harness_flush": 5275}
MEASURED_METAL_CAMPAIGN = 12647.6
#: The floor these tests hold every route watchdog to, as a multiple of its measured wall.
MARGIN = 3


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    round_driver.add_common(ap)
    return ap


def test_the_route_watchdog_defaults_are_the_pre_certification_values():
    args = _parser().parse_args(["route", "--repo", ".", "--stamp", "s", "--capability", "8.6",
                                 "--home", "h"])
    assert args.route_timeout_factor == round_driver.ROUTE_TIMEOUT_FACTOR == 12
    assert args.route_min_timeout == round_driver.ROUTE_MIN_TIMEOUT == 3600


def test_every_route_leg_is_priced_at_or_above_its_measured_wall():
    legs = {step["id"]: step["expect_s"] for table in ("cuda", "triton")
            for step in L.route_legs(table)}
    assert set(legs) == set(MEASURED_ROUTE_WALLS)
    for leg, wall in MEASURED_ROUTE_WALLS.items():
        assert wall <= legs[leg] < wall + 1, (leg, legs[leg], wall)
    assert set(L.E2E_EXPECT) == set(MEASURED_LICENCE_WALLS)
    for leg, wall in MEASURED_LICENCE_WALLS.items():
        assert wall <= L.E2E_EXPECT[leg] < wall + 1, (leg, L.E2E_EXPECT[leg], wall)


def test_every_route_and_licence_leg_gets_a_watchdog_three_times_its_measured_wall():
    factor, floor = round_driver.ROUTE_TIMEOUT_FACTOR, round_driver.ROUTE_MIN_TIMEOUT
    for table in ("cuda", "triton"):
        for step in L.route_legs(table):
            timeout = round_driver.route_timeout(step, factor, floor)
            assert timeout >= floor
            assert timeout >= MARGIN * MEASURED_ROUTE_WALLS[step["id"]], (step["id"], timeout)
    for leg, expect in L.E2E_EXPECT.items():
        timeout = round_driver.route_timeout({"expect_s": expect}, factor, floor)
        assert timeout >= MARGIN * MEASURED_LICENCE_WALLS[leg], (leg, timeout)


def test_the_route_stage_hands_every_route_and_licence_step_its_watchdog(tmp_path, monkeypatch):
    lanes = []
    monkeypatch.setattr(round_driver.L, "check", lambda repo, capability, route, **kw: (
        [], {"triton_route_campaign": "t_campaign", "cuda_route_campaign": "c_campaign"}))
    monkeypatch.setattr(round_driver.L, "flush_coverage", lambda repo, cap: (
        {"triton": {"arms_covered": 1, "arms": 1}, "cuda": {"arms_covered": 1, "arms": 1}}, []))
    monkeypatch.setattr(round_driver, "run_lanes",
                        lambda home, args, plan, results, subs, prefix: lanes.extend(plan) or {})
    args = _parser().parse_args(["route", "--repo", str(tmp_path), "--results",
                                 str(tmp_path / "res"), "--stamp", "s", "--capability", "8.6",
                                 "--home", str(tmp_path), "--lock-dir", str(tmp_path),
                                 "--route-timeout-factor", "20", "--route-min-timeout", "7200"])
    assert round_driver.stage_route(args, tmp_path) == 0
    lane = run_lane.Lane.__new__(run_lane.Lane)
    lane.args = argparse.Namespace(min_timeout=1800, timeout_factor=4)
    steps = [step for steps in lanes for step in steps if step["kind"] != "toolchain"]
    assert {s["id"] for s in steps} >= set(MEASURED_ROUTE_WALLS) | {
        f"e2e_{leg}" for leg in MEASURED_LICENCE_WALLS}
    for step in steps:
        assert lane.timeout_for(step) == step["timeout_s"] == max(7200, 20 * step["expect_s"])


def test_the_lane_honours_a_step_watchdog_before_its_default():
    args = argparse.Namespace(min_timeout=1800, timeout_factor=4)
    lane = run_lane.Lane.__new__(run_lane.Lane)
    lane.args = args
    assert lane.timeout_for({"expect_s": 450}) == 1800
    assert lane.timeout_for({"expect_s": 450, "timeout_s": 5400}) == 5400


def test_the_metal_route_table_is_the_campaign_driver_legs_at_their_measured_walls():
    text = (ROUND.parent.parent / "run_metal_dispatch_campaign.sh").read_text()
    legs = []
    for name in ("SERIAL_LEGS", "LANE_LEGS"):
        legs += re.search(rf"^{name}=\(([^)]*)\)", text, re.M).group(1).split()
    assert sorted(legs) == sorted(metal_round.ROUTE_EXPECT)
    assert metal_round.ROUTE_EXPECT == MEASURED_METAL_WALLS


def test_the_metal_route_watchdog_is_three_times_its_measured_walls():
    watchdog = metal_round.route_watchdog()
    assert watchdog >= metal_round.ROUTE_MIN_TIMEOUT
    assert watchdog >= MARGIN * sum(MEASURED_METAL_WALLS.values())
    assert watchdog >= MARGIN * MEASURED_METAL_CAMPAIGN
    assert metal_round.route_watchdog(5, 1) == 5 * sum(MEASURED_METAL_WALLS.values())
    assert metal_round.route_watchdog(1, 99999) == 99999


def test_the_metal_route_stage_hands_the_campaign_that_watchdog_without_the_user_site(
        tmp_path, monkeypatch):
    seen = {}

    class Named:
        stdout = ('{"arch": "applegpu_g13s", "stamp": "dispatch_metal_route_x", '
                  '"campaign": "dispatch_metal_route_x_applegpu_g13s", "refused": null}\n')
        stderr = ""
        returncode = 0

    monkeypatch.setattr(metal_round.subprocess, "run", lambda *a, **k: Named())
    monkeypatch.setattr(metal_round.Round, "tree_is_the_commit", lambda self, stage: None)

    def run_item(self, stage, label, argv, env, artifact, timeout):
        seen.update(label=label, timeout=timeout, nousersite=env.get("PYTHONNOUSERSITE"))
        return 0

    monkeypatch.setattr(metal_round.Round, "run_item", run_item)
    common = ["route", "--repo", str(tmp_path), "--stamp", "s", "--results", str(tmp_path),
              "--logs", str(tmp_path / "logs")]
    assert metal_round.main(common) == 0
    assert seen == {"label": "route_campaign", "timeout": metal_round.route_watchdog(),
                    "nousersite": "1"}
    assert metal_round.main([*common, "--logs", str(tmp_path / "logs2"),
                             "--route-timeout-factor", "6", "--route-min-timeout", "60"]) == 0
    assert seen["timeout"] == 6 * sum(MEASURED_METAL_WALLS.values())
