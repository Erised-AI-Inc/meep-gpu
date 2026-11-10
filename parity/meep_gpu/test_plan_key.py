"""The canonical plan form (``plan_key.py``): what it keeps, what it leaves out, its inputs.

Host-only; no device, no MEEP. Every test names the rule it pins.
"""
from __future__ import annotations

import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import plan_key  # noqa: E402


def full_report(steps: int = 2400, arm_b: str = "fused pair B", backend: str = "triton",
                applied=(), mode: str = "measured") -> dict:
    """A dispatch record of the shape ``driver.fast_path_report()`` returns."""
    return {
        "decision": "dispatched", "step_path": "fused", "refused_because": None,
        "slots": {
            "step_B": {"state": "dispatched", "arm": arm_b, "reason": None},
            "update_H": {"state": "dispatched", "arm": arm_b, "reason": None},
            "step_D": {"state": "dispatched", "arm": "fused pair D", "reason": None},
            "update_E": {"state": "dispatched", "arm": "fused pair D", "reason": None},
            "fill_B": {"state": "array", "arm": None, "reason": "no consulted product"},
            "update_P": {"state": "array", "arm": None, "reason": "no consulted product"},
        },
        "families": {arm_b: {"family": "fused_electric", "gate": "g", "recorded_utc": "t0",
                             "host": "h0"},
                     "fused pair D": {"family": "fused_electric", "gate": "g",
                                      "recorded_utc": "t0", "host": "h0"}},
        "launch_counters": {
            slot: {"arm": arm, "backend": backend, "dispatches": steps,
                   "programs_per_dispatch": 94 if slot.startswith("step") else None,
                   "plan_launches": None}
            for slot, arm in (("step_B", arm_b), ("update_H", arm_b),
                              ("step_D", "fused pair D"), ("update_E", "fused pair D"))},
        "table": backend,
        "composition": {"tables_dispatched": [backend], "mixed": True, "table": backend},
        "run_shape": {"grid_shape": [200, 120, 1], "dimensions": 2},
        "environment": {"host": "h0", "cuda_visible_devices": "2"},
        "enable": {"variable": "MEEP_GPU_DISPATCH", "value": "1"},
        "subnormal": {"installed_by": "dispatch, at the configuration freeze"},
        "preference": {"mode": mode, "records": {},
                       "candidates": [{"label": label, "applied": True} for label in applied]},
    }


def summary_of(report: dict) -> dict:
    """The route gates' summary of a record (``e2e._plan_summary``), reproduced."""
    slots = report["slots"]
    dispatched = [name for name, entry in slots.items() if entry["state"] == "dispatched"]
    return {"decision": report["decision"], "step_path": report["step_path"],
            "refused_because": report["refused_because"],
            "dispatched_slots": dispatched,
            "array_slots": [name for name in slots if name not in dispatched],
            "arms": {name: slots[name]["arm"] for name in dispatched},
            "families": report["families"], "composition": report["composition"],
            "run_shape": report["run_shape"], "launch_counters": report["launch_counters"],
            "subnormal": report["subnormal"], "environment": report["environment"]}


def test_one_step_and_a_whole_run_of_one_plan_digest_equal():
    """Launches are kept per step, so a plan pass (1 step) and a leg (2400) compare."""
    assert (plan_key.key(full_report(1), 1)["digest"]
            == plan_key.key(full_report(2400), 2400)["digest"])


def test_a_record_and_its_summary_describe_the_same_plan():
    report = full_report()
    assert plan_key.describe(report, 2400) == plan_key.describe(summary_of(report), 2400)
    assert (plan_key.selected_plan_sha256(report)
            == plan_key.selected_plan_sha256(summary_of(report)))


#: The canonical selected plan's published test vectors (schema meep_gpu.selected_plan/1).
VECTORS = [
    ({"decision": "dispatched", "table": "triton", "slots": {
        "step_B": {"state": "dispatched", "arm": "fused pair B"},
        "fill_B": {"state": "array", "arm": None},
        "update_H": {"state": "dispatched", "arm": "fused pair B"},
        "step_D": {"state": "dispatched", "arm": "PML"},
        "fill_D": {"state": "array", "arm": None},
        "update_E": {"state": "dispatched", "arm": "ordinary"},
        "update_P": {"state": "array", "arm": None}}},
     "ba89aa0ace0ce81b8828e534650873f21c9aa0fb0bb1ac640b0428942fc330cd"),
    ({"decision": "dispatched", "composition": {"table": "triton"},
      "arms": {"update_E": "ordinary", "step_B": "fused pair B", "step_D": "PML",
               "update_H": "fused pair B"}},
     "ba89aa0ace0ce81b8828e534650873f21c9aa0fb0bb1ac640b0428942fc330cd"),
    ({"decision": "dispatched", "table": "cuda", "slots": {
        "step_B": {"state": "dispatched", "arm": "cuda:fused magnetic pair"},
        "update_H": {"state": "dispatched", "arm": "cuda:fused magnetic pair"},
        "step_D": {"state": "dispatched", "arm": "cuda:PML"},
        "update_E": {"state": "dispatched", "arm": "cuda:ordinary"}}},
     "104f24552ae01b9b4c8bae0a13019cd5a997eeb71ba7e5bf3405f10540323372"),
    ({"decision": "refused", "slots": {}},
     "735c085bcf5d58a03036fe4a06b26ffe9dc3faadd68864e754fe08ebfeb20d4f"),
]


@pytest.mark.parametrize("report, expected", VECTORS)
def test_the_selected_plan_digest_reproduces_the_published_vectors(report, expected):
    assert plan_key.selected_plan_sha256(report) == expected


def test_the_key_reads_the_one_canonical_selected_plan():
    import selected_plan
    assert plan_key.selected_plan is selected_plan.selected_plan
    assert plan_key.selected_plan_sha256 is selected_plan.selected_plan_sha256
    assert plan_key.SELECTED_PLAN_SCHEMA == selected_plan.SCHEMA


@pytest.mark.parametrize("path,value", [
    (("environment", "host"), "another-host"),
    (("enable", "value"), None),
    (("subnormal", "installed_by"), "an earlier configuration freeze in this process"),
    (("families", "fused pair D", "recorded_utc"), "t1"),
    (("families", "fused pair D", "host"), "h1"),
    (("slots", "fill_B", "reason"), "another reason"),
    (("run_shape", "grid_shape"), [400, 120, 1]),
    (("preference", "mode"), "span"),
    (("launch_counters", "step_D", "programs_per_dispatch"), 188),
])
def test_process_and_evidence_facts_do_not_move_the_digest(path, value):
    report = full_report()
    moved = copy.deepcopy(report)
    target = moved
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    assert plan_key.key(moved, 2400)["digest"] == plan_key.key(report, 2400)["digest"]


@pytest.mark.parametrize("change", ["arm", "launches", "plan_launches", "array_slot",
                                    "platform", "decision", "cuda_label"])
def test_every_selection_fact_moves_the_digest(change):
    report = full_report()
    moved = copy.deepcopy(report)
    if change == "arm":
        moved["slots"]["step_B"]["arm"] = "PML"
    elif change == "launches":
        moved["launch_counters"]["step_D"]["dispatches"] = 4800
    elif change == "plan_launches":
        moved["launch_counters"]["step_D"]["plan_launches"] = 2400
    elif change == "array_slot":
        moved["slots"]["update_P"]["state"] = "dispatched"
        moved["slots"]["update_P"]["arm"] = "ADE"
    elif change == "platform":
        moved["table"] = "metal"
    elif change == "decision":
        moved["decision"] = "refused"
    elif change == "cuda_label":
        moved["slots"]["step_D"]["arm"] = "cuda:fused electric pair"
    assert plan_key.key(moved, 2400)["digest"] != plan_key.key(report, 2400)["digest"]


def test_per_step_is_exact_and_names_a_remainder():
    assert plan_key.per_step(4800, 2400) == 2
    assert plan_key.per_step(2401, 2400) == "2401/2400"
    assert plan_key.per_step(None, 5) is None
    assert plan_key.per_step(6, 4) == "3/2"


def test_a_refused_plan_is_its_decision_and_no_slots():
    report = {"decision": "refused", "step_path": "array",
              "refused_because": "kill switch MEEP_GPU_FUSED=0", "slots": {}}
    plan = plan_key.describe(report, 1)
    assert plan["selected_plan"]["dispatched"] == [] and plan["launches_per_step"] == {}
    other = dict(report, refused_because="another wording")
    assert plan_key.key(other, 1)["digest"] == plan_key.key(report, 1)["digest"]


def test_key_digest_refuses_a_tampered_block_and_another_form():
    block = plan_key.key(full_report(), 2400)
    assert plan_key.key_digest(block) == block["digest"]
    tampered = copy.deepcopy(block)
    tampered["plan"]["selected_plan"]["dispatched"][0][1] = "PML"
    assert plan_key.key_digest(tampered) is None
    other = dict(block, form="meep_gpu.plan/0")
    assert plan_key.key_digest(other) is None
    assert plan_key.key_digest(None) is None


def test_route_leg_keys_fold_rows_and_the_default_control():
    report = full_report()
    row = {"case": "pml_2d", "steps_taken": {"fused": 2400, "unfused": 2400, "array": 2400},
           "legs": {"fused": {"plan": summary_of(report)},
                    "unfused": {"plan": summary_of(full_report(arm_b="PML"))},
                    "array": {"plan": {"decision": "refused", "step_path": "array",
                                       "refused_because": "kill switch"}}}}
    control = {"control": "default-preference", "case": "pml_2d",
               plan_key.KEY_FIELD: plan_key.key(report, 1)}
    other = {"control": "release-route", "case": "pml_2d"}
    keys = plan_key.route_leg_keys([row], [control, other])
    assert sorted(keys["pml_2d"]) == ["array", "default-preference", "fused", "unfused"]
    assert keys["pml_2d"]["default-preference"] == control[plan_key.KEY_FIELD]["digest"]
    # A row without its step count is not keyed (a digest over an unknown count).
    row["steps_taken"].pop("unfused")
    assert "unfused" not in plan_key.route_leg_keys([row], [])["pml_2d"]


def _tree(tmp_path):
    root = tmp_path / "repo"
    for table in plan_key.TABLES:
        (root / "meep_gpu" / f"{table}_kernels").mkdir(parents=True)
    (root / "parity" / "meep_gpu").mkdir(parents=True)
    ledger = root / "meep_gpu" / "triton_kernels" / "fingerprints.json"
    ledger.write_text(json.dumps({"a": {"x": 1}, "driver_dispatch": {"runs": {}}}))
    record = root / "meep_gpu" / "triton_kernels" / "timing_cc86.json"
    record.write_text('{"record_sha256": "r1"}')
    (root / "meep_gpu" / "triton_kernels" / "timing_cc86.report.txt").write_text("report")
    return root, ledger, record


def test_plan_inputs_name_every_record_and_each_ledger_view(tmp_path):
    root, ledger, record = _tree(tmp_path)
    inputs = plan_key.plan_inputs(str(root))
    assert list(inputs["timing_records"]) == ["meep_gpu/triton_kernels/timing_cc86.json"]
    assert list(inputs["ledgers"]) == ["meep_gpu/triton_kernels/fingerprints.json"]


def test_the_ledger_view_leaves_out_only_driver_dispatch(tmp_path):
    root, ledger, record = _tree(tmp_path)
    before = plan_key.plan_inputs(str(root))
    ledger.write_text(json.dumps({"a": {"x": 1}, "driver_dispatch": {"runs": {"8.6": 1}}},
                                 indent=4))
    assert plan_key.inputs_moved(before, plan_key.plan_inputs(str(root))) == {}
    ledger.write_text(json.dumps({"a": {"x": 2}, "driver_dispatch": {}}))
    assert plan_key.inputs_moved(before, plan_key.plan_inputs(str(root))) == {
        "ledgers": ["meep_gpu/triton_kernels/fingerprints.json"]}


def test_a_new_or_moved_record_moves_the_inputs(tmp_path):
    root, ledger, record = _tree(tmp_path)
    before = plan_key.plan_inputs(str(root))
    record.write_text('{"record_sha256": "r2"}')
    assert plan_key.inputs_moved(before, plan_key.plan_inputs(str(root))) == {
        "timing_records": ["meep_gpu/triton_kernels/timing_cc86.json"]}
    record.write_text('{"record_sha256": "r1"}')
    (root / "meep_gpu" / "cuda_kernels" / "timing_cc86.json").write_text("{}")
    assert plan_key.inputs_moved(before, plan_key.plan_inputs(str(root))) == {
        "timing_records": ["meep_gpu/cuda_kernels/timing_cc86.json"]}
    assert plan_key.inputs_moved(None, before)["timing_records"]
