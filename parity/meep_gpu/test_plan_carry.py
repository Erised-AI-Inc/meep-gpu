"""The carry rules (``plan_carry.py``) and the route re-cut's use of them.

Host-only: synthetic legs over a small synthetic tree, one test per rule, and the re-cut's
``carried_preference`` over the live records. Every carry rule has a test that carries and
one that refuses.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import plan_carry as pc  # noqa: E402
import plan_key  # noqa: E402

CODE = ("meep_gpu/fastpath.py", "parity/meep_gpu/gate_dispatch_fused_route.py")


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(arm_d="fused pair D", applied=()):
    return {"decision": "dispatched", "step_path": "fused",
            "slots": {"step_D": {"state": "dispatched", "arm": arm_d},
                      "update_E": {"state": "dispatched", "arm": arm_d}},
            "families": {arm_d: {"family": "fused_electric"}},
            "launch_counters": {s: {"backend": "triton", "dispatches": 1}
                                for s in ("step_D", "update_E")},
            "composition": {"tables_dispatched": ["triton"]},
            "run_shape": {"grid_shape": [200, 120, 1]},
            "preference": {"mode": "measured",
                           "candidates": [{"label": l, "applied": True} for l in applied]}}


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "repo"
    for name in CODE:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"# {name}\n")
    kernels = root / "meep_gpu" / "triton_kernels"
    kernels.mkdir(parents=True)
    (kernels / "fingerprints.json").write_text(json.dumps({"arm": {"x": 1},
                                                           "driver_dispatch": {}}))
    (kernels / "timing_cc86.json").write_text('{"record_sha256": "a"}')
    return root


def leg(root, defaults=("pml_2d",), released=True, device="8.6", env=None,
        arm_d="fused pair D"):
    keys = {case: {"default-preference": plan_key.key(report(arm_d), 1)["digest"],
                   "fused": "f" * 64} for case in defaults}
    keys.setdefault("magnetic_seam_2d", {})["fused"] = "e" * 64
    return {"release": {"released": released, "reasons": []},
            "imported_source_sha256": {name: sha(root / name) for name in CODE},
            plan_key.INPUTS_FIELD: plan_key.plan_inputs(str(root)),
            "plan_keys": keys,
            "stamped_device": {"compute_capability": device},
            "provenance": dict(env or {"python": "3.10.20", "meep": "1.35.0",
                                       "backend": "triton", "install_policy_requested": None,
                                       "harness_installs_nothing": True})}


def plan_pass(root, cases=("pml_2d",), arm_d="fused pair D", **changes):
    summary = leg(root, cases, arm_d=arm_d)
    summary.pop("release")
    summary["mode"] = "plan-only"
    summary["plan_keys"] = {case: {"default-preference":
                                   plan_key.key(report(arm_d), 1)["digest"]}
                            for case in cases}
    summary["failures"] = []
    summary["rows"] = [{"control": "default-preference", "case": case,
                        "verdict": pc.DEFAULT_RECORDED} for case in cases]
    summary.update(changes)
    return summary


def judge(root, artifact, passed=None, device="cc86"):
    return pc.judge_route_leg(artifact, root, plan_key.plan_inputs(str(root)), device,
                              passed)


def move_record(root):
    (root / "meep_gpu" / "triton_kernels" / "timing_cc86.json").write_text(
        '{"record_sha256": "b"}')


# --------------------------------------------------------------------- route legs ----

def test_unchanged_inputs_carry_by_digest(tree):
    assert judge(tree, leg(tree))["state"] == pc.CARRY_DIGEST


def test_a_moved_closure_file_re_runs_the_leg_by_name(tree):
    artifact = leg(tree)
    (tree / CODE[0]).write_text("# edited\n")
    verdict = judge(tree, artifact)
    assert verdict["state"] == pc.RERUN_CLOSURE and verdict["moved"] == [CODE[0]]


def test_a_moved_record_with_no_default_plan_carries_by_the_shape_rule(tree):
    artifact = leg(tree, defaults=())
    move_record(tree)
    assert judge(tree, artifact)["state"] == pc.CARRY_SHAPE_RULE


def test_a_moved_record_without_a_plan_pass_is_owed(tree):
    artifact = leg(tree)
    move_record(tree)
    verdict = judge(tree, artifact)
    assert verdict["state"] == pc.OWED_PLAN and verdict["cases"] == ["pml_2d"]


def test_a_moved_record_with_the_same_plans_carries_by_plan(tree):
    artifact = leg(tree)
    move_record(tree)
    verdict = judge(tree, artifact, plan_pass(tree))
    assert verdict["state"] == pc.CARRY_PLAN
    assert verdict["proof"]["plans"] == {"pml_2d": artifact["plan_keys"]["pml_2d"][
        "default-preference"]}


def test_a_changed_plan_re_runs_and_names_the_case(tree):
    artifact = leg(tree, defaults=("pml_2d", "pml_3d"))
    move_record(tree)
    passed = plan_pass(tree, cases=("pml_2d", "pml_3d"))
    passed["plan_keys"]["pml_3d"]["default-preference"] = plan_key.key(
        report("PML"), 1)["digest"]
    verdict = judge(tree, artifact, passed)
    assert verdict["state"] == pc.RERUN_PLAN and verdict["cases"] == ["pml_3d"]


def test_a_plan_pass_missing_a_recorded_case_is_a_changed_plan(tree):
    artifact = leg(tree, defaults=("pml_2d", "pml_3d"))
    move_record(tree)
    verdict = judge(tree, artifact, plan_pass(tree, cases=("pml_2d",)))
    assert verdict["state"] == pc.RERUN_PLAN and verdict["cases"] == ["pml_3d"]


@pytest.mark.parametrize("damage, needle", [
    ("not_a_pass", "not a plan pass"),
    ("other_bytes", "ran on other bytes"),
    ("other_inputs", "other planner data"),
    ("other_device", "stamped device"),
    ("other_environment", "another environment"),
    ("other_leg", "another leg"),
    ("triton_withheld", "another leg"),
    ("other_resolution", "another leg"),
    ("unrecorded_failure", "did not record the default"),
    ("unrecorded_row", "did not record the default"),
])
def test_a_plan_pass_that_cannot_stand_for_the_tree_is_refused(tree, damage, needle):
    artifact = leg(tree)
    move_record(tree)
    passed = plan_pass(tree)
    if damage == "not_a_pass":
        passed["mode"] = "gate"
    elif damage == "other_bytes":
        passed["imported_source_sha256"] = dict(passed["imported_source_sha256"],
                                                **{CODE[0]: "0" * 64})
    elif damage == "other_inputs":
        passed[plan_key.INPUTS_FIELD] = artifact[plan_key.INPUTS_FIELD]
    elif damage == "other_device":
        passed["stamped_device"] = {"compute_capability": "9.0"}
    elif damage == "other_environment":
        passed["provenance"] = {"python": "3.10.20", "meep": "1.33.0"}
    elif damage == "other_leg":
        passed["provenance"] = dict(passed["provenance"], install_policy_requested="keep",
                                    harness_installs_nothing=False)
    elif damage == "triton_withheld":
        passed["provenance"] = dict(passed["provenance"], triton_withheld={"mechanism": "x"})
    elif damage == "other_resolution":
        passed["provenance"] = dict(passed["provenance"], resolution=40)
    elif damage == "unrecorded_failure":
        passed["failures"] = ["pml_2d"]
    elif damage == "unrecorded_row":
        passed["rows"][0]["verdict"] = "DEFAULT-PREFERENCE-NOT-RECORDED"
    verdict = judge(tree, artifact, passed)
    assert verdict["state"] == pc.OWED_PLAN and needle in verdict["why"], verdict
    assert verdict["cases"] == ["pml_2d"]


def test_the_rules_spell_the_gate_s_recorded_verdict():
    _recut_mod, route_gate = _recut()
    assert pc.DEFAULT_RECORDED == route_gate.DEFAULT_PREFERENCE_RECORDED


def _default_report(mode=None):
    from meep_gpu import fastpath  # noqa: PLC0415
    plan = report()
    plan["preference"] = {"mode": mode or fastpath.PREFERENCE_DEFAULT, "value": None,
                          "records": {"triton": {"path": "triton_kernels/timing_cc86.json",
                                                 "status": "read",
                                                 "record_sha256": "ab" * 32}},
                          "candidates": []}
    return plan


def test_only_a_row_that_recorded_the_default_offers_its_plan_as_a_key():
    _recut_mod, route_gate = _recut()
    recorded = route_gate.default_preference_row(_default_report(), "fused")
    assert recorded["verdict"] == route_gate.DEFAULT_PREFERENCE_RECORDED
    assert plan_key.key_digest(recorded[plan_key.KEY_FIELD]) is not None
    other = route_gate.default_preference_row(_default_report(mode="span"), "fused")
    assert other["verdict"] == route_gate.DEFAULT_PREFERENCE_NOT_RECORDED
    assert plan_key.KEY_FIELD not in other
    assert plan_key.key_digest(other[plan_key.UNRECORDED_KEY_FIELD]) is not None
    rows = [dict(recorded, case="pml_2d"), dict(other, case="pml_3d")]
    assert plan_key.route_leg_keys([], rows) == {
        "pml_2d": {"default-preference": recorded[plan_key.KEY_FIELD]["digest"]}}


def test_a_moved_ledger_view_re_runs_and_its_own_entry_does_not(tree):
    artifact = leg(tree)
    ledger = tree / "meep_gpu" / "triton_kernels" / "fingerprints.json"
    ledger.write_text(json.dumps({"arm": {"x": 1}, "driver_dispatch": {"runs": {"8.6": {}}}}))
    assert judge(tree, artifact)["state"] == pc.CARRY_DIGEST
    ledger.write_text(json.dumps({"arm": {"x": 2}, "driver_dispatch": {}}))
    assert judge(tree, artifact)["state"] == pc.RERUN_INPUTS


@pytest.mark.parametrize("change, state", [
    (lambda a: a.update(release={"released": False}), pc.RERUN_UNRELEASED),
    (lambda a: a.pop("plan_keys"), pc.RERUN_UNKEYED),
    (lambda a: a.pop(plan_key.INPUTS_FIELD), pc.RERUN_UNKEYED),
    (lambda a: a.pop("imported_source_sha256"), pc.RERUN_UNKEYED),
    (lambda a: a.update(stamped_device={"compute_capability": "9.0"}), pc.RERUN_DEVICE),
])
def test_a_leg_that_is_not_evidence_here_re_runs(tree, change, state):
    artifact = leg(tree)
    change(artifact)
    assert judge(tree, artifact)["state"] == state


def test_a_cut_that_names_no_device_carries_nothing(tree):
    for stamped in ("8.6", "9.0"):
        verdict = judge(tree, leg(tree, device=stamped), device=None)
        assert verdict["state"] == pc.RERUN_DEVICE and "never assumed" in verdict["why"]
    assert judge(tree, leg(tree, device="9.0"))["state"] == pc.RERUN_DEVICE
    artifact = leg(tree)
    move_record(tree)
    problems = pc.plan_pass_problems(plan_pass(tree), tree, plan_key.plan_inputs(str(tree)),
                                     None, artifact, {}, ["pml_2d"])
    assert any("stamped device" in line for line in problems)
    assert judge_l3(tree, l3_row(tree), device=None)["state"] == pc.RERUN_DEVICE
    passed = l3_row(tree, plan_only=True)
    assert any("stamped device" in line for line in pc.l3_plan_row_problems(
        l3_row(tree), passed, tree, plan_key.plan_inputs(str(tree)), None, {}))


def test_the_device_is_read_never_assumed():
    assert pc.device_of({"stamped_device": {"compute_capability": "8.6"}}) == "cc86"
    assert pc.device_of({"provenance": {"apple_gpu": {"architecture": "applegpu_g13s"}}}) \
        == "applegpu_g13s"
    assert pc.device_of({"stamped_device": {"unreadable": "no cupy"}}) is None


# ------------------------------------------------------------------------- L3 rows ----

TOOLCHAIN = {"python": "3.10.20", "numpy": "2.2.6", "cupy": "13.6.0", "triton": "3.3.1",
             "torch": None}


def l3_env(cache_parent="/caches/triton_keep"):
    return {"MEEP_GPU_HARDWARE": "certified", "CUPY_ACCELERATORS": "",
            "MEEP_GPU_BACKEND_PREFERENCE": None, "OMP_NUM_THREADS": "1",
            "CUPY_CACHE_DIR": f"{cache_parent}/cupy_ftz_stripped",
            "TRITON_CACHE_DIR": f"{cache_parent}/triton",
            "PYTHONPATH": f"/site:{cache_parent}/repo", "CUDA_VISIBLE_DEVICES": "3"}


def l3_row(root, arm_d="fused pair D", **extra):
    row = {"label": "ex.a", "table": "triton", "mode": "keep", "shipped_default": False,
           "sets": ["linux_keep"], "primary": "linux_keep", "image_set": "linux_keep",
           "l3_verdict": "holds_dispatched",
           "meep": {"images": {"_meep.so": "m"}, "version": "1.35.0",
                    "single_precision": True},
           "env": l3_env(), plan_key.TOOLCHAIN_FIELD: dict(TOOLCHAIN),
           plan_key.KEY_FIELD: plan_key.key(report(arm_d), 1),
           plan_key.INPUTS_FIELD: plan_key.plan_inputs(str(root)),
           "imported_source_sha256": {name: sha(root / name) for name in CODE},
           "stamped_device": {"compute_capability": "8.6"}}
    row.update(extra)
    return row


def judge_l3(root, row, plan_row=None, device="cc86"):
    return pc.judge_l3_row(row, root, plan_key.plan_inputs(str(root)), device, plan_row)


def test_an_l3_row_carries_by_digest_and_by_plan(tree):
    row = l3_row(tree)
    assert judge_l3(tree, row)["state"] == pc.CARRY_DIGEST
    move_record(tree)
    assert judge_l3(tree, row)["state"] == pc.OWED_PLAN
    assert judge_l3(tree, row, l3_row(tree, plan_only=True))["state"] == pc.CARRY_PLAN


def test_an_l3_row_whose_plan_changed_re_runs(tree):
    row = l3_row(tree)
    move_record(tree)
    verdict = judge_l3(tree, row, l3_row(tree, arm_d="PML", plan_only=True))
    assert verdict["state"] == pc.RERUN_PLAN


@pytest.mark.parametrize("field, value, needle", [
    ("table", "cuda", "another leg"), ("mode", "flush", "another leg"),
    ("shipped_default", True, "another leg"), ("sets", ["linux_keep", "x"], "another leg"),
    ("primary", "x", "another leg"), ("image_set", "x", "another leg"),
    ("meep", {"images": {"_meep.so": "other"}, "version": "1.35.0",
              "single_precision": True}, "another MEEP"),
    ("meep", {"images": {"_meep.so": "m"}, "version": "1.33.0",
              "single_precision": True}, "another MEEP"),
    ("env", dict(l3_env(), MEEP_GPU_HARDWARE=None), "another environment"),
    ("env", dict(l3_env(), CUPY_CACHE_DIR="/caches/triton_keep/cupy_plain"),
     "another environment"),
    (plan_key.TOOLCHAIN_FIELD, dict(TOOLCHAIN, triton="3.4.0"), "another toolchain"),
])
def test_an_l3_plan_row_of_another_leg_cannot_stand(tree, field, value, needle):
    row = l3_row(tree)
    move_record(tree)
    verdict = judge_l3(tree, row, l3_row(tree, plan_only=True, **{field: value}))
    assert verdict["state"] == pc.OWED_PLAN and needle in verdict["why"], verdict


def test_an_l3_plan_row_is_not_compared_on_paths_or_card_index(tree):
    row = l3_row(tree)
    move_record(tree)
    elsewhere = dict(l3_env("/elsewhere/other_leg"), CUDA_VISIBLE_DEVICES="0")
    passed = l3_row(tree, plan_only=True, env=elsewhere)
    assert judge_l3(tree, row, passed)["state"] == pc.CARRY_PLAN


def test_an_l3_row_without_a_toolchain_cannot_be_compared(tree):
    row = l3_row(tree)
    row.pop(plan_key.TOOLCHAIN_FIELD)
    move_record(tree)
    verdict = judge_l3(tree, row, l3_row(tree, plan_only=True))
    assert verdict["state"] == pc.OWED_PLAN and "toolchain" in verdict["why"]


@pytest.mark.parametrize("verdict, state", [("holds_dispatched", pc.CARRY_DIGEST),
                                            ("holds_array", pc.CARRY_DIGEST),
                                            ("differs", pc.RERUN_UNRELEASED),
                                            ("error", pc.RERUN_UNRELEASED)])
def test_an_l3_row_carries_only_when_it_held(tree, verdict, state):
    assert judge_l3(tree, l3_row(tree, l3_verdict=verdict))["state"] == state


def test_the_leg_facts_of_an_l3_row():
    row = {"table": "triton", "mode": "keep", "shipped_default": False,
           "sets": ["b", "a"], "primary": "a", "image_set": "a"}
    assert pc.l3_leg_differences(row, dict(row, sets=["a", "b"])) == {}
    assert set(pc.l3_leg_differences(row, dict(row, table="cuda", primary="b"))) == {
        "table", "primary"}


def test_an_l3_row_with_an_error_or_no_key_re_runs(tree):
    assert judge_l3(tree, l3_row(tree, error="boom"))["state"] == pc.RERUN_UNRELEASED
    unkeyed = l3_row(tree)
    unkeyed.pop(plan_key.KEY_FIELD)
    assert judge_l3(tree, unkeyed)["state"] == pc.RERUN_UNKEYED
    tampered = l3_row(tree)
    tampered[plan_key.KEY_FIELD]["plan"]["step_path"] = "array"
    assert judge_l3(tree, tampered)["state"] == pc.RERUN_UNKEYED


# ------------------------------------------------------------------- the re-cut ----

def _recut():
    import recut_driver_dispatch_record as recut  # noqa: PLC0415
    import gate_dispatch_fused_route as route_gate  # noqa: PLC0415
    return recut, route_gate


def _stated(shipped):
    return {"variable": "MEEP_GPU_PREFER", "shape_rule_legs": "span", "default_legs": 1,
            "default_legs_recorded": 1,
            "shipped_records": {name: dict(entry) for name, entry in shipped.items()},
            "read_by_the_default": {name: [entry.get("record_sha256")]
                                    for name, entry in shipped.items()},
            "read_matches_shipped": True, "read_but_not_as_shipped": [],
            "cases": {"pml_2d": {"vetoes_applied": []}}}


def _pass_rows(live, vetoes=()):
    """A plan pass's default row on pml_2d, reading every live record as shipped."""
    read = {name.split("_kernels/")[0]: {"path": name, "status": "read",
                                         "record_sha256": entry.get("record_sha256")}
            for name, entry in sorted(live.items()) if entry.get("record_sha256")}
    return [{"control": "default-preference", "case": "pml_2d",
             "verdict": pc.DEFAULT_RECORDED, "records_read": read,
             "prior_derived_from": [],
             "vetoes_applied": [{"label": label} for label in vetoes]}]


def _set_aside(names):
    return {f"meep_gpu/{name}": {"why": "other_vendor"} for name in names}


@pytest.mark.parametrize("state, refused", [(pc.CARRY_PLAN, False),
                                            (pc.CARRY_SHAPE_RULE, False),
                                            (pc.CARRY_SLICE, False),
                                            (pc.CARRY_DIGEST, True), (None, True)])
def test_the_recut_takes_moved_records_only_on_a_plan_carry(state, refused):
    recut, route_gate = _recut()
    live = {name: dict(entry) for name, entry in route_gate.shipped_timing_records().items()}
    ran = copy.deepcopy(live)
    for entry in ran.values():
        entry["file_sha256"] = "0" * 64
    artifacts = {"shipped": {"dispatch_preference": _stated(ran)}}
    carried = ({"shipped": {"state": state, "source": "/elsewhere/shipped",
                            "proof": {"plans": {"pml_2d": "c" * 64},
                                      "set_aside": _set_aside(live)},
                            "plan_pass_rows": _pass_rows(live)}} if state else {})
    block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run", carried)
    moved = [p for p in problems if p.startswith("timing_records_moved_since_the_route_ran")]
    assert bool(moved) is refused, problems
    if not refused:
        assert problems == [], problems
        logged = block[recut.CARRY_LOG]["shipped"]
        assert logged["state"] == state and logged["plans"] == {"pml_2d": "c" * 64}
        assert set(block["shipped_records"]) == {f"meep_gpu/{n}" for n in live}


@pytest.mark.parametrize("named, refused", [("all", False), ("one_short", True),
                                            ("only_a_ledger", True), ("none", True)])
def test_a_slice_carry_takes_only_the_records_it_set_aside(named, refused):
    """A leg carried past another device's ledger bind proves nothing about a record the
    leg ran at other bytes: the statements disagreeing there still refuse."""
    recut, route_gate = _recut()
    live = {name: dict(entry) for name, entry in route_gate.shipped_timing_records().items()}
    ran = {name: dict(entry, file_sha256="0" * 64) for name, entry in live.items()}
    names = sorted(live)
    excused = {"all": names, "one_short": names[1:], "only_a_ledger": [], "none": []}[named]
    proof = {"plans": {"pml_2d": "c" * 64}, "set_aside": _set_aside(excused)}
    if named == "only_a_ledger":
        proof["set_aside"]["meep_gpu/metal_kernels/fingerprints.json"] = {"why": "slice_equal"}
    if named == "none":
        del proof["set_aside"]
    carried = {"shipped": {"state": pc.CARRY_SLICE, "proof": proof}}
    artifacts = {"shipped": {"dispatch_preference": _stated(ran)}}
    _block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run", carried)
    moved = [p for p in problems if p.startswith("timing_records_moved_since_the_route_ran")]
    assert bool(moved) is refused, problems


def test_a_slice_carry_takes_a_new_record_it_set_aside_and_no_other():
    recut, route_gate = _recut()
    live = {name: dict(entry) for name, entry in route_gate.shipped_timing_records().items()}
    new = sorted(live)[0]
    ran = {name: dict(entry) for name, entry in live.items() if name != new}
    artifacts = {"shipped": {"dispatch_preference": _stated(ran)}}
    for excused, refused in (([new], False), ([], True)):
        carried = {"shipped": {"state": pc.CARRY_SLICE,
                               "proof": {"set_aside": _set_aside(excused)}}}
        _block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run", carried)
        moved = [p for p in problems
                 if p.startswith("timing_records_moved_since_the_route_ran")]
        assert bool(moved) is refused, problems


def test_the_carry_log_names_only_the_records_that_moved_or_were_added():
    """The carry log is history, which the weld walk reads as stale by construction, so it
    holds what moved and nothing else: the digests a carried leg ran on that the tree no
    longer ships (``timing_records_from``) and the records the tree ships that the leg did
    not run on (``timing_records_added``). A digest equal to the live one is not logged; a
    record that never moved would otherwise sit in the route record as a stale digest."""
    recut, route_gate = _recut()
    live = {name: dict(entry) for name, entry in route_gate.shipped_timing_records().items()}
    names = sorted(live)
    assert len(names) >= 3, names
    added, moved, kept = names[0], names[1], names[2]
    ran = {name: dict(entry) for name, entry in live.items() if name != added}
    ran[moved]["file_sha256"] = "0" * 64
    carried = {"shipped": {"state": pc.CARRY_SLICE,
                           "proof": {"set_aside": _set_aside([added, moved])}}}
    artifacts = {"shipped": {"dispatch_preference": _stated(ran)}}
    block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run", carried)
    assert problems == [], problems
    logged = block[recut.CARRY_LOG]["shipped"]
    assert logged["timing_records_from"] == {moved: "0" * 64}, (kept, logged)
    assert logged["timing_records_added"] == [added], logged
    assert logged["timing_records_to"] == "shipped_records", logged


def test_the_recut_states_a_plan_carried_leg_s_default_as_its_pass_made_it():
    recut, route_gate = _recut()
    live = {name: dict(entry) for name, entry in route_gate.shipped_timing_records().items()}
    ran = {name: dict(entry, file_sha256="0" * 64) for name, entry in live.items()}
    stated = _stated(ran)
    stated["cases"] = {"pml_2d": {"vetoes_applied": ["the leg's veto"]}}
    stated["read_by_the_default"] = {"triton_kernels/timing_old.json": ["0" * 64]}
    artifacts = {"shipped": {"dispatch_preference": stated}}
    rows = _pass_rows(live, ["the pass's veto"])
    rows.append(dict(rows[0], case="pml_3d", vetoes_applied=[{"label": "a case the leg did "
                                                                       "not run"}]))
    carried = {"shipped": {"state": pc.CARRY_PLAN, "proof": {"plans": {"pml_2d": "c" * 64}},
                           "plan_pass_rows": rows}}
    block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run", carried)
    assert problems == [], problems
    assert block["vetoes_applied"] == {"pml_2d": ["the pass's veto"]}
    assert "meep_gpu/triton_kernels/timing_old.json" not in block["read_by_the_default"]
    assert block["by_leg"]["shipped"]["read_by_the_default"] == block["read_by_the_default"]
    as_ran = block[recut.CARRY_LOG]["shipped"]["as_ran"]
    assert as_ran["vetoes_applied"] == {"pml_2d": ["the leg's veto"]}
    assert as_ran["read_by_the_default"] == ["meep_gpu/triton_kernels/timing_old.json"]
    carried["shipped"]["plan_pass_rows"] = []
    _block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run", carried)
    assert any(line.startswith("carried_leg_plan_pass_states_no_default") for line in problems)


def test_the_carry_log_is_history_to_the_weld_walker():
    recut, route_gate = _recut()
    from meep_gpu import weld_record_walk as walk  # noqa: PLC0415
    live = route_gate.shipped_timing_records()
    ran = {name: dict(entry, file_sha256="0" * 64) for name, entry in live.items()}
    artifacts = {"shipped": {"dispatch_preference": _stated(ran)}}
    carried = {"shipped": {"state": pc.CARRY_PLAN, "proof": {"plans": {"pml_2d": "c" * 64}},
                           "plan_pass_rows": _pass_rows(live, ["fused pair D"])}}
    block, problems = recut.carried_preference(artifacts, {"shipped"}, live, "run", carried)
    assert problems == [] and "as_ran" in block[recut.CARRY_LOG]["shipped"]
    # Where the cut files it: in the route run's own record.
    found = walk.census({"driver_dispatch": {walk.RUNS: {"8.6": {recut.PREFERENCE_KEY: block}}}})
    logged = [f for f in found if recut.CARRY_LOG in f.trail]
    assert logged and all(f.rule == "historical_recut_log" for f in logged)
    assert all(f.rule != walk.UNCLASSIFIED for f in found)
