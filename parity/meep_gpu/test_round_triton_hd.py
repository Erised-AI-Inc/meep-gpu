"""The 1.0 round runs the ten Triton H->D gates, and binds the composition records it re-cuts.

The Triton board binds its H->D products through ``build_triton_fusion_matrix.GATE_BOUND``,
which re-hashes each gate's artifact against the tree, so a board cut on a frozen tree needs
those gates run on it; until the round's leg table held them they ran outside it. These
tests pin the legs (one hand leg per gate under keep, into ``triton_<gate>_<stamp>/keep/``,
the layout GATE_BOUND names), their prices against the walls they were measured at, the
step a lane runs for one, the one-shot root names, the corpus check, and the bind's
composition line (host-only).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROUND = HERE / "rounds" / "1.0"
for path in (str(ROUND), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import legs as L  # noqa: E402
import round as round_driver  # noqa: E402
import run_lane  # noqa: E402

#: Walls (seconds) the ten gates were measured at, under keep, every leg, lifts on, each
#: run released. PRE-CERTIFICATION: the 1.0 pre-certification pass (2026-10-10, two streams
#: of five on one RTX A6000 beside seven other streams). FROZEN: seven of them re-run on
#: the frozen tree of the 1.0 round (2026-10-12, two streams on one card). The oracle the
#: prices are checked against.
MEASURED_PRECERT_WALLS = {
    "fused_hd_pair": 3052, "folded_fused_hd_pair": 3523, "complex_fused_hd_pair": 1893,
    "cylindrical_fused_hd_pair": 1111, "folded_complex_fused_hd_pair": 929,
    "beta_complex_fused_hd_pair": 639, "conductive_bfast_fused_hd_pair": 602,
    "cylindrical_real_fused_hd_pair": 388, "beta_real_fused_hd_pair": 352,
    "nonlinear_fused_hd_pair": 257}
MEASURED_FROZEN_WALLS = {
    "fused_hd_pair": 1315, "folded_fused_hd_pair": 1363, "complex_fused_hd_pair": 482,
    "folded_complex_fused_hd_pair": 206, "beta_complex_fused_hd_pair": 173,
    "conductive_bfast_fused_hd_pair": 127, "beta_real_fused_hd_pair": 100}
#: The roots and the GATE_BOUND rows the generator re-points (NREC's 14 H->D entries).
GATE_BOUND_HD_ROOTS = {
    "fused_hd_pair", "folded_fused_hd_pair", "complex_fused_hd_pair",
    "cylindrical_fused_hd_pair", "cylindrical_real_fused_hd_pair",
    "folded_complex_fused_hd_pair", "beta_complex_fused_hd_pair",
    "conductive_bfast_fused_hd_pair", "beta_real_fused_hd_pair", "nonlinear_fused_hd_pair"}


def test_the_ten_gates_are_the_ones_the_board_binds_and_each_is_in_the_checkout():
    assert set(L.TRITON_HD) == GATE_BOUND_HD_ROOTS and len(L.TRITON_HD) == 10
    for gate in L.TRITON_HD:
        assert (HERE / f"gate_triton_{gate}.py").is_file(), gate
    import build_triton_fusion_matrix as board  # noqa: PLC0415
    named = {re.match(r"triton_(\w+?hd_pair)_", relative.split("/")[1]).group(1)
             for family, relative in board.GATE_BOUND.items() if "hd_pair" in family}
    assert named == set(L.TRITON_HD)


def test_the_h_to_d_gates_are_hand_legs_not_fleet_gates():
    import drive_triton_weld_gates as fleet  # noqa: PLC0415
    assert not set(L.TRITON_HD) & set(fleet.GATES)
    assert L.TRITON_HD_POLICIES == ("keep",)


def test_every_gate_is_priced_at_the_larger_wall_and_its_watchdog_holds_three_times_it():
    assert set(L.TRITON_HD_EXPECT) == set(L.TRITON_HD)
    for gate in L.TRITON_HD:
        larger = max(MEASURED_PRECERT_WALLS[gate], MEASURED_FROZEN_WALLS.get(gate, 0))
        assert L.TRITON_HD_EXPECT[gate] >= larger, gate
        watchdog = max(1800, 4 * L.TRITON_HD_EXPECT[gate])  # run_lane.py's default
        assert watchdog >= 3 * larger, gate


def test_the_plan_holds_each_gate_once_under_keep_in_its_own_root():
    plan, _summary = L.plan_lanes(Path("/repo"), Path("/repo/results"), "s", gpus=4,
                                  tables=("cuda", "triton"))
    steps = [step for lane in plan for step in lane if step["kind"] == "hd"]
    assert sorted(step["gate"] for step in steps) == sorted(L.TRITON_HD)
    for step in steps:
        assert step["policy"] == "keep"
        assert step["id"] == f"triton/keep/hd/{step['gate']}"
        assert step["root"] == f"@RESULTS@/triton_{step['gate']}_@STAMP@"
        assert step["expect_s"] == L.TRITON_HD_EXPECT[step["gate"]]
        assert step["lift_timeout"] == L.TRITON_HD_LIFT_TIMEOUT == 3600
    units = [u["name"] for u in L.triton_units() if "_hd_" in u["name"]]
    assert sorted(units) == sorted(f"triton_keep_hd_{gate}" for gate in L.TRITON_HD)


def test_the_round_names_each_gate_root_one_shot_and_fills_the_placeholders(tmp_path):
    names = L.round_roots("2000-01-01_t", 2)
    for gate in L.TRITON_HD:
        assert names.count(f"triton_{gate}_2000-01-01_t") == 1
    subs = round_driver.subs_for(argparse.Namespace(stamp="2000-01-01_t"), tmp_path, {})
    lane = run_lane.Lane(_lane_args(tmp_path, subs))
    step = lane.sub(L.triton_hd_steps()[0])
    assert step["root"] == str(tmp_path / f"triton_{L.TRITON_HD[0]}_2000-01-01_t")


def test_the_h_to_d_legs_lift_and_refuse_a_missing_corpus_by_name(tmp_path):
    steps = L.triton_hd_steps()
    assert L.lifting_legs(steps) == [step["id"] for step in steps]
    problems = L.corpus_problems(steps, str(tmp_path / "absent"))
    assert len(problems) == 1 and "10 H->D leg(s)" in problems[0]


def _lane_args(tmp_path: Path, subs=None) -> argparse.Namespace:
    return argparse.Namespace(lane_dir=str(tmp_path / "lane"), repo=str(tmp_path / "repo"),
                              results=str(tmp_path / "results"), python="python",
                              gpu=0, subs=json.dumps(subs or {}), libcuda_stub="",
                              corpus_root="", timeout_factor=4, min_timeout=1800,
                              heartbeat=120)


def _run_hd(tmp_path, monkeypatch, policy="keep", gate="fused_hd_pair"):
    lane = run_lane.Lane(_lane_args(tmp_path))
    seen = {}

    def run_child(argv, env, log, timeout, label):
        seen.update(argv=argv, env=env, timeout=timeout)
        artifact = Path(argv[argv.index("--out") + 1])
        artifact.write_text(json.dumps({"canonical_verdict": {"released": True}}))
        log.write_text("ran\n")
        return {"rc": 0, "timed_out": False, "elapsed_s": 1.0}

    def mint(cache):
        seen["minted"] = cache
        return None

    monkeypatch.setattr(lane, "run_child", run_child)
    monkeypatch.setattr(lane, "mint_keep_cache", mint)
    root = tmp_path / "results" / f"triton_{gate}_s"
    step = {"id": f"triton/{policy}/hd/{gate}", "kind": "hd", "gate": gate,
            "root": str(root), "policy": policy, "lift_timeout": 3600,
            "expect_s": L.TRITON_HD_EXPECT[gate]}
    return lane, step, root, seen


def test_a_lane_runs_one_gate_into_its_keep_root_in_the_fleet_environment(tmp_path,
                                                                          monkeypatch):
    lane, step, root, seen = _run_hd(tmp_path, monkeypatch)
    result = lane.step_hand(step)
    artifact = root / "keep" / "gate.json"
    assert result["state"] == "released"
    assert seen["argv"][1:] == ["-u", str(lane.harness / "gate_triton_fused_hd_pair.py"),
                                "--subnormal-policy", "keep", "--lift-timeout", "3600",
                                "--out", str(artifact)]
    env = seen["env"]
    assert env["MEEP_GPU_DISPATCH"] == "0" and env["MEEP_GPU_HARDWARE"] == "certified"
    assert env["MEEP_GPU_SUBNORMAL_POLICY"] == "keep"
    assert "ftz_stripped" in Path(env["CUPY_CACHE_DIR"]).name
    assert seen["minted"] == Path(env["CUPY_CACHE_DIR"])
    assert seen["timeout"] == max(1800, 4 * L.TRITON_HD_EXPECT["fused_hd_pair"])
    assert (root / "keep" / "run.log").read_text() == "ran\n"
    assert not (root / "device.json").exists()
    # A rerun finds the released artifact and runs nothing.
    seen.clear()
    assert lane.step_hand(step)["skipped"] == "artifact already passes" and not seen


def test_a_flush_leg_imports_meep_first_and_mints_no_keep_cache(tmp_path, monkeypatch):
    lane, step, _root, seen = _run_hd(tmp_path, monkeypatch, policy="flush")
    lane.step_hand(step)
    assert seen["argv"][-1] == "--import-meep-for-host-policy"
    assert "ftz_stripped" not in seen["env"]["CUPY_CACHE_DIR"] and "minted" not in seen


@pytest.mark.parametrize("leftover", ["gate.json", "lift_progress.jsonl"])
def test_a_directory_an_earlier_run_wrote_is_refused(tmp_path, monkeypatch, leftover):
    lane, step, root, seen = _run_hd(tmp_path, monkeypatch)
    (root / "keep").mkdir(parents=True)
    (root / "keep" / leftover).write_text("{}")
    result = lane.step_hand(step)
    assert result["state"] == "refused" and not seen


def test_the_dry_run_filter_keeps_an_h_to_d_leg_by_gate_name():
    steps = L.triton_hd_steps()
    kept = round_driver.filter_steps(steps, set(), {"nonlinear_fused_hd_pair"})
    assert [s["gate"] for s in kept] == ["nonlinear_fused_hd_pair"]


def test_the_bind_re_cuts_the_composition_records_by_their_keys(tmp_path):
    import recut_composition_records as recut  # noqa: PLC0415
    derived = tuple(key for key, (legs, _probe) in recut.LEGS.items()
                    if set(legs) & set(L.TRITON_D))
    assert derived == L.TRITON_COMPOSITION_KEYS
    lines = L.bind_plan(tmp_path, "2000-01-01_t", 2, write=True)
    composition = [line for line in lines if "--run results/triton_composition_" in line]
    assert composition == [
        "python recut_composition_records.py --run results/triton_composition_2000-01-01_t "
        "--flush-run results/triton_composition_2000-01-01_t_flush --only "
        "cylindrical_composition_gate,symmetry_composition_gate,conductivity_composition_gate,"
        "dispersive_composition_gate,no_pml_composition_gate --write"]
