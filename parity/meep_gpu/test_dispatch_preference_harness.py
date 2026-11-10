"""The harness half of the dispatch preference: who times, cuts and certifies which rule.

The package's default arbitrates the shape rule's fused products by measured cost
(``fastpath.PREFERENCE_SWITCH``, default ``measured``; ``span`` is the shape rule
alone). Four harness tools have to agree on which of the two each of their legs runs
under, or the loop that feeds the timing records erases itself:

* ``bench_fused_products`` times the fused leg under the shape rule (a product a record
  vetoes would otherwise never be timed again), honours a value the caller exports,
  records it in the row's ``process_environment`` and copies the plan's preference
  block into the row;
* ``cut_timing_record`` admits a fused row only when its ``process_environment`` says
  the shape rule, stamps that rule in the record, and recomputes a record cut before
  the rule without it;
* the two route gates run every composition-asserting leg under the shape rule, and
  one default-mode leg per released case records which timing records gave the
  default;
* ``recut_driver_dispatch_record`` carries that statement into the record as pins the
  ledger walker checks, and refuses a campaign whose records moved.

No device, no MEEP run: plans and rows are built in the test, and the live tree's
timing records are read where the tools read them.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_fused_products as bench  # noqa: E402
import cut_timing_record as cut  # noqa: E402
import gate_dispatch_fused_route as route_gate  # noqa: E402
import gate_dispatch_metal_route as metal_route  # noqa: E402
import recut_driver_dispatch_record as recut  # noqa: E402

from meep_gpu import dispatch_preference as dp  # noqa: E402
from meep_gpu import fastpath  # noqa: E402
from meep_gpu import weld_record_walk as walk  # noqa: E402

import test_cut_timing_record as rows_fixture  # noqa: E402

SWITCH = fastpath.PREFERENCE_SWITCH


@pytest.fixture(autouse=True)
def _no_inherited_preference(monkeypatch):
    """Each test starts with the switch unset and the bench's campaign value cleared."""
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setattr(bench, "PREFERENCE", None)


# ---------------------------------------------------------------------------
# The bench
# ---------------------------------------------------------------------------

def test_the_bench_times_the_shape_rule_when_the_switch_is_unset():
    resolved = bench.resolve_preference({})
    assert resolved["variable"] == SWITCH
    assert resolved["value"] == "span"
    assert "default" in resolved["source"]


@pytest.mark.parametrize("value", ["measured", "span"])
def test_a_value_the_caller_exports_is_honoured_and_named_as_theirs(value):
    resolved = bench.resolve_preference({SWITCH: value})
    assert resolved["value"] == value
    assert resolved["source"] == "the process environment"


def test_a_value_the_package_refuses_is_refused_before_anything_is_lifted():
    with pytest.raises(SystemExit) as refused:
        bench.resolve_preference({SWITCH: "fastest"})
    assert SWITCH in str(refused.value) and "fastest" in str(refused.value)


def test_the_campaign_exports_its_preference_and_the_row_records_it(monkeypatch):
    exported = bench.export_preference()
    assert exported["value"] == "span"
    assert os.environ[SWITCH] == "span"
    assert bench.process_environment()[SWITCH] == "span"
    assert bench.campaign_preference() is exported


def test_the_campaign_keeps_an_exported_default_and_the_row_records_that(monkeypatch):
    monkeypatch.setenv(SWITCH, "measured")
    bench.export_preference()
    assert bench.process_environment()[SWITCH] == "measured"


def test_the_fused_leg_runs_the_campaign_value_whatever_the_gate_leg_names(monkeypatch):
    """The gate's ``_fuse_env`` names the shape rule; a campaign that asked for the
    default still runs its fused leg under the default, and the other legs are left
    alone."""
    module = types.SimpleNamespace(
        _fuse_env=lambda arms, reached_by="release": {"MEEP_GPU_FUSE_ARMS": None,
                                                       SWITCH: "span"},
        unfused_env=lambda: {"MEEP_GPU_FUSE_ARMS": "0"},
        ARRAY_ENV={"MEEP_GPU_FUSED": "0"})
    monkeypatch.setenv(SWITCH, "measured")
    bench.export_preference()
    spec = {"arms": ["fused pair D"], "reached_by": "release"}
    assert bench.leg_env(module, spec, "fused")[SWITCH] == "measured"
    assert SWITCH not in bench.leg_env(module, spec, "unfused")
    assert SWITCH not in bench.leg_env(module, spec, "array")
    monkeypatch.delenv(SWITCH)
    bench.export_preference()
    assert bench.leg_env(module, spec, "fused")[SWITCH] == "span"


def test_the_row_plan_carries_the_preference_block_beside_the_summary():
    block = {"variable": SWITCH, "value": "span", "mode": "span", "records": {},
             "candidates": [], "compositions": 1}
    report = {"decision": "dispatched", "slots": {}, "preference": block}
    leg = {"driver": types.SimpleNamespace(fast_path_report=lambda: report)}
    plan = bench.plan_of(leg)
    assert plan["preference"] == block
    assert plan["decision"] == "dispatched"
    assert plan["preference"] is not block  # a copy, never the live record


# ---------------------------------------------------------------------------
# The cutter
# ---------------------------------------------------------------------------

def _row(case, preference="span", **options):
    row = rows_fixture.bench_row(case=case, **options)
    if preference is None:
        row.pop("process_environment", None)
    else:
        row["process_environment"] = {SWITCH: preference}
    return row


def _excluded(record, case):
    return [item["reason"] for item in record["excluded"] if item["case"] == case]


def test_a_fused_row_timed_under_the_default_is_left_out_by_name(tmp_path):
    rows_fixture.write(tmp_path, "tree/rows.jsonl", [
        _row("shape_rule"), _row("shape_rule", utc="2026-09-19T14:00:00Z"),
        _row("default", preference="measured"), _row("unrecorded", preference=None)])
    record = rows_fixture.cut_tree(tmp_path)
    assert _excluded(record, "shape_rule") == []
    (default,) = _excluded(record, "default")
    assert default.startswith(cut.NOT_THE_SHAPE_RULE)
    assert f"{SWITCH}='measured'" in default
    (unrecorded,) = _excluded(record, "unrecorded")
    assert unrecorded.startswith(cut.NOT_THE_SHAPE_RULE)
    assert cut.ENVIRONMENT_KEY in unrecorded


def test_a_cut_stamps_the_rule_and_states_it(tmp_path):
    rows_fixture.write(tmp_path, "tree/rows.jsonl", [
        _row("pml_2d"), _row("pml_2d", utc="2026-09-19T14:00:00Z")])
    record = rows_fixture.cut_tree(tmp_path)
    assert record["cut"]["fused_leg_preference"] == "span"
    assert "span" in record["rules"]["fused_leg_preference"]
    older = rows_fixture.cut_tree(tmp_path, preference=None)
    assert "fused_leg_preference" not in older["cut"]
    assert "fused_leg_preference" not in older["rules"]


def test_check_recomputes_each_record_by_the_rule_it_was_cut_by(tmp_path, capsys):
    """A record cut before the rule, from rows that never recorded the switch,
    recomputes byte for byte; one cut under it recomputes too; and the stamp is what
    decides, so a stamped record whose rows lost the switch DIFFERS."""
    rows_fixture.write(tmp_path, "old/rows.jsonl", [
        _row("pml_2d", preference=None),
        _row("pml_2d", preference=None, utc="2026-09-19T14:00:00Z")])
    rows_fixture.write(tmp_path, "new/rows.jsonl", [
        _row("pml_2d"), _row("pml_2d", utc="2026-09-19T14:00:00Z")])
    old = rows_fixture.cut_tree(tmp_path, roots=("old",), preference=None)
    new = rows_fixture.cut_tree(tmp_path, roots=("new",))
    assert old["keys"] and new["keys"]
    for name, record in (("old.json", old), ("new.json", new)):
        (tmp_path / name).write_text(dp.dumps(record))
        assert cut.check(str(tmp_path / name), str(tmp_path)) == cut.EXIT_OK, name
    rows_fixture.write(tmp_path, "new/rows.jsonl", [
        _row("pml_2d", preference=None),
        _row("pml_2d", preference=None, utc="2026-09-19T14:00:00Z")])
    assert cut.check(str(tmp_path / "new.json"), str(tmp_path)) == cut.EXIT_DIFFERS


def test_a_timing_harness_point_without_the_switch_is_left_out(tmp_path):
    point = rows_fixture.phase0_rows()
    for row in point:
        row.pop("process_environment", None)
    rows_fixture.write(tmp_path, "tree/rows.jsonl", point)
    lines, _files = cut.read_files(cut.row_files([str(tmp_path / "tree")]), str(tmp_path))
    measurements, excluded, _counts = cut.gather(lines, "triton", False)
    assert measurements == []
    timed = [item["reason"] for item in excluded
             if not item["reason"].startswith("verdict ")]
    assert timed and all(reason.startswith(cut.NOT_THE_SHAPE_RULE) for reason in timed)
    with pytest.raises(cut.CutRefused, match="nothing_admitted"):
        rows_fixture.cut_tree(tmp_path)


def test_a_case_timed_only_under_the_default_is_owed_by_the_release(tmp_path):
    rows_fixture.write(tmp_path, "tree/rows.jsonl", [
        _row("pml_2d"), _row("pml_2d", utc="2026-09-19T14:00:00Z"),
        _row("magnetic_seam_2d", preference="measured")])
    record = rows_fixture.cut_tree(
        tmp_path, released={"fused pair D": ["pml_2d", "magnetic_seam_2d"]})
    owed = {(item["product"], item["case"]): item["reason"]
            for item in record.get("missing") or ()}
    assert ("fused pair D", "magnetic_seam_2d") in owed
    assert cut.NOT_THE_SHAPE_RULE in owed[("fused pair D", "magnetic_seam_2d")]


# ---------------------------------------------------------------------------
# The route gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reached_by", ["release", "opt-in"])
def test_every_fused_route_leg_runs_the_shape_rule_unless_it_asks_for_the_default(
        reached_by):
    assert route_gate._fuse_env(["fused pair D"], reached_by)[SWITCH] == "span"
    default = route_gate._fuse_env(["fused pair D"], reached_by, shape_rule=False)
    assert SWITCH in default and default[SWITCH] is None


@pytest.mark.parametrize("reached_by", ["release", "opt-in"])
def test_every_metal_route_leg_runs_the_shape_rule_unless_it_asks_for_the_default(
        reached_by):
    assert metal_route._fuse_env(["fused pair D"], reached_by)[SWITCH] == "span"
    default = metal_route._fuse_env(["fused pair D"], reached_by, shape_rule=False)
    assert SWITCH in default and default[SWITCH] is None


def test_the_engine_smoke_names_the_shape_rule_whichever_leg_ran_before_it():
    for residency in metal_route.ENGINE_SMOKE_RESIDENCIES:
        assert metal_route.engine_smoke_env(residency)[SWITCH] == "span"


def test_a_package_that_no_longer_reads_the_shape_rule_is_refused(monkeypatch):
    monkeypatch.setattr(fastpath, "PREFERENCE_MODES", ("measured",))
    with pytest.raises(RuntimeError, match="span"):
        route_gate.dispatch_preference_env(True)


def _default_report(**overrides):
    block = {
        "variable": SWITCH, "value": None, "mode": fastpath.PREFERENCE_DEFAULT,
        "architecture": "8.6",
        "records": {"triton": {"path": "triton_kernels/timing_cc86.json",
                               "status": "read", "record_sha256": "ab" * 32},
                    "cuda": {"path": "cuda_kernels/timing_cc86.json",
                             "status": "absent"}},
        "prior": {"derived_from": [{"path": "cuda_kernels/timing_cc86.json",
                                    "table": "cuda", "architecture": ["8.6"],
                                    "record_sha256": "cd" * 32, "veto_bands": []}]},
        "candidates": [{"label": "fused pair D", "table": "triton", "rule": "record",
                        "slots": ["step_D", "update_E"], "applied": True,
                        "took_the_slots": ["PML", "ordinary"]},
                       {"label": "fused pair B", "table": "triton", "applied": False}],
        "note": "meep_gpu: NOTE dispatch preference: ..."}
    block.update(overrides.pop("block", {}))
    report = {"decision": "dispatched", "fusion": {"driven": {"step_B": "fused pair B"}},
              "preference": block}
    report.update(overrides)
    return report


def test_a_default_mode_plan_with_its_records_named_is_recorded():
    row = route_gate.default_preference_row(_default_report(), "fused")
    assert row["verdict"] == route_gate.DEFAULT_PREFERENCE_RECORDED
    assert row["records_read"]["triton"]["record_sha256"] == "ab" * 32
    assert [item["label"] for item in row["vetoes_applied"]] == ["fused pair D"]
    assert row["prior_derived_from"][0]["record_sha256"] == "cd" * 32
    assert route_gate.DEFAULT_PREFERENCE_RECORDED in route_gate.CONTROL_PASS
    assert route_gate.DEFAULT_PREFERENCE_RECORDED in metal_route.CONTROL_PASS


@pytest.mark.parametrize("change, needle", [
    ({"decision": "refused", "refused_because": "rung 8b"}, "did not dispatch"),
    ({"block": {"mode": "span"}}, "not the default"),
    ({"block": {"records": {"triton": {"path": "t", "status": "refused",
                                       "refused_because": "edited after it was cut"}}}},
     "edited after it was cut"),
    ({"block": {"records": {"triton": {"path": "t", "status": "read"}}}},
     "no record_sha256"),
    ({"block": {"records": {"triton": {"path": "t",
                                       "status": "architecture_unread"}}}},
     "architecture_unread"),
])
def test_a_default_that_cannot_name_its_records_is_not_recorded(change, needle):
    row = route_gate.default_preference_row(_default_report(**change), "fused")
    assert row["verdict"] == route_gate.DEFAULT_PREFERENCE_NOT_RECORDED
    assert any(needle in reason for reason in row["why"]), row["why"]


def test_a_plan_whose_preference_was_never_reached_is_not_recorded():
    report = _default_report()
    report["preference"] = {"variable": SWITCH, "value": None,
                            "not_reached": "refused before the composer rung"}
    row = route_gate.default_preference_row(report, "array")
    assert row["verdict"] == route_gate.DEFAULT_PREFERENCE_NOT_RECORDED
    assert any("not reached" in reason for reason in row["why"])


def test_the_live_tree_ships_the_records_the_default_reads():
    shipped = route_gate.shipped_timing_records()
    assert shipped, "the tree ships no timing record"
    for name, entry in shipped.items():
        path = pathlib.Path(dp.__file__).resolve().parent / name
        assert entry["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["record_sha256"] == dp.load_record(str(path))[dp.DIGEST_FIELD]


def test_the_run_summary_holds_what_the_default_read_to_what_the_tree_ships():
    shipped = {"triton_kernels/timing_cc86.json": {"file_sha256": "11" * 32,
                                                   "record_sha256": "ab" * 32},
               "cuda_kernels/timing_cc86.json": {"file_sha256": "22" * 32,
                                                 "record_sha256": "cd" * 32}}
    row = route_gate.default_preference_row(_default_report(), "fused")
    row["case"] = "pml_2d"
    skipped = {"control": "default-preference", "case": "opt", "verdict": "NOT-APPLICABLE"}
    summary = route_gate.dispatch_preference_summary([row, skipped], shipped)
    assert summary["default_legs"] == 1 and summary["default_legs_recorded"] == 1
    assert summary["read_matches_shipped"] is True
    assert sorted(summary["read_by_the_default"]) == sorted(shipped)
    assert summary["cases"]["pml_2d"]["vetoes_applied"] == ["fused pair D"]
    assert summary["shape_rule_legs"] == "span"
    moved = dict(shipped, **{"triton_kernels/timing_cc86.json": {
        "file_sha256": "33" * 32, "record_sha256": "ef" * 32}})
    stale = route_gate.dispatch_preference_summary([row], moved)
    assert stale["read_matches_shipped"] is False
    assert stale["read_but_not_as_shipped"] == ["triton_kernels/timing_cc86.json"]


def test_a_summary_that_cannot_read_the_records_is_written_as_an_error(monkeypatch):
    def unreadable():
        raise OSError("no such file")
    monkeypatch.setattr(route_gate, "shipped_timing_records", unreadable)
    assert "no such file" in route_gate._dispatch_preference_or_error([])["error"]


# ---------------------------------------------------------------------------
# The recut
# ---------------------------------------------------------------------------

def _stated(shipped, recorded=1, matches=True, read=None):
    return {"variable": SWITCH, "shape_rule_legs": "span", "default_legs": recorded,
            "default_legs_recorded": recorded,
            "shipped_records": {name: dict(entry) for name, entry in shipped.items()},
            "read_by_the_default": {name: [entry["record_sha256"]]
                                    for name, entry in (read or shipped).items()},
            "read_matches_shipped": matches, "read_but_not_as_shipped": [],
            "cases": {"pml_2d": {"vetoes_applied": ["fused pair D"]}}}


def test_the_record_carries_the_default_as_pins_the_ledger_walker_checks():
    live = route_gate.shipped_timing_records()
    artifacts = {"shipped": {"dispatch_preference": _stated(live)},
                 "harness_keep": {"dispatch_preference": _stated(live, recorded=0)}}
    block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run")
    assert problems == []
    assert set(block["shipped_records"]) == {f"meep_gpu/{name}" for name in live}
    assert block["vetoes_applied"] == {"pml_2d": ["fused pair D"]}
    # Where the cut files it: in the route run's own record.
    found = walk.census({"driver_dispatch": {walk.RUNS: {"8.6": {recut.PREFERENCE_KEY: block}}}})
    assert found and all(f.tier == walk.TIER_RAW for f in found), [f.rule for f in found]
    repo = pathlib.Path(recut._API)  # noqa: SLF001
    for f in found:
        assert hashlib.sha256((repo / f.subject).read_bytes()).hexdigest() == f.value


@pytest.mark.parametrize("damage, code", [
    (lambda artifacts, live: artifacts["shipped"].pop("dispatch_preference"),
     "route_leg_states_no_dispatch_preference"),
    (lambda artifacts, live: artifacts["shipped"]["dispatch_preference"].update(
        error="OSError: no such file"), "route_leg_states_no_dispatch_preference"),
    (lambda artifacts, live: next(iter(
        artifacts["harness_keep"]["dispatch_preference"]["shipped_records"].values()
    )).update(file_sha256="00" * 32), "timing_records_moved_since_the_route_ran"),
    (lambda artifacts, live: artifacts["shipped"]["dispatch_preference"].update(
        default_legs_recorded=0), "route_leg_recorded_no_default"),
    (lambda artifacts, live: artifacts["shipped"]["dispatch_preference"].update(
        read_matches_shipped=False), "default_read_records_the_tree_does_not_ship"),
    (lambda artifacts, live: next(iter(live.values())).update(
        refused_because="edited after it was cut"), "shipped_timing_record_is_refused"),
])
def test_a_campaign_that_cannot_certify_the_shipped_default_is_refused(damage, code):
    live = {name: dict(entry) for name, entry in route_gate.shipped_timing_records().items()}
    artifacts = {"shipped": {"dispatch_preference": _stated(live)},
                 "harness_keep": {"dispatch_preference": _stated(live, recorded=0)}}
    damage(artifacts, live)
    _block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run")
    assert any(problem.startswith(code) for problem in problems), problems


def test_a_refusal_leg_need_not_have_recorded_the_default():
    live = route_gate.shipped_timing_records()
    artifacts = {"shipped": {"dispatch_preference": _stated(live)},
                 "harness_keep": {"dispatch_preference": _stated(live, recorded=0)}}
    _block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run")
    assert problems == []
