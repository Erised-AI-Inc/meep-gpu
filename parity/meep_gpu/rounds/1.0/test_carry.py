"""``carry.py``: route and L3 carry records, placement and verification, on the host.

Synthetic campaigns over a synthetic tree; the leg lists are the round's own
(``legs.route_legs``, ``metal_round.ROUTE_EXPECT``). The dry run over unchanged inputs
must carry every leg and every row.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
for _path in (str(HERE), str(HARNESS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import ast  # noqa: E402

import carry  # noqa: E402
import plan_carry as pc  # noqa: E402
import plan_key  # noqa: E402

CODE = ("meep_gpu/fastpath.py", "parity/meep_gpu/gate_dispatch_fused_route.py")
TRITON_LEGS = ("flush", "harness_keep", "shipped")
#: The L3 leg the tests build: Triton, keep, its default sets (linux_keep).
LEG = ("--table", "triton", "--baseline", "keep")


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def report(arm_d="fused pair D"):
    return {"decision": "dispatched", "step_path": "fused",
            "slots": {"step_D": {"state": "dispatched", "arm": arm_d}},
            "families": {arm_d: {"family": "fused_electric"}},
            "launch_counters": {"step_D": {"backend": "triton", "dispatches": 1}},
            "composition": {"tables_dispatched": ["triton"]},
            "run_shape": {"grid_shape": [200, 120, 1]},
            "preference": {"mode": "measured", "candidates": []}}


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "repo"
    for name in CODE:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"# {name}\n")
    kernels = root / "meep_gpu" / "triton_kernels"
    kernels.mkdir(parents=True)
    (kernels / "fingerprints.json").write_text('{"arm": {"x": 1}}')
    (kernels / "timing_cc86.json").write_text('{"record_sha256": "a"}')
    return root


def artifact(root, defaults=("pml_2d",), **extra):
    payload = {"release": {"released": True, "reasons": []},
               "imported_source_sha256": {name: sha(root / name) for name in CODE},
               plan_key.INPUTS_FIELD: plan_key.plan_inputs(str(root)),
               "plan_keys": {case: {"default-preference": plan_key.key(report(), 1)["digest"]}
                             for case in defaults},
               "stamped_device": {"compute_capability": "8.6"},
               "provenance": {"python": "3.10.20", "meep": "1.35.0"}}
    payload.update(extra)
    return payload


def campaign(root, where, legs=TRITON_LEGS, no_default=("flush",)):
    for leg in legs:
        (where / leg).mkdir(parents=True)
        payload = artifact(root, defaults=() if leg in no_default else ("pml_2d",))
        (where / leg / "gate.json").write_text(json.dumps(payload))
        (where / leg / "cases.jsonl").write_text('{"case": "pml_2d"}\n')
    return where


def plans(root, where, legs=TRITON_LEGS, arm_d="fused pair D"):
    for leg in legs:
        (where / leg).mkdir(parents=True)
        payload = artifact(root)
        payload.pop("release")
        payload["mode"] = "plan-only"
        payload["plan_keys"] = {"pml_2d": {"default-preference":
                                           plan_key.key(report(arm_d), 1)["digest"]}}
        payload["failures"] = []
        payload["rows"] = [{"control": "default-preference", "case": "pml_2d",
                            "verdict": pc.DEFAULT_RECORDED, "vetoes_applied": []}]
        (where / leg / "plans.json").write_text(json.dumps(payload))
    return where


def run(*argv):
    return carry.main([str(a) for a in argv])


def record(path):
    return json.loads(pathlib.Path(path).read_text())


def move_record(root):
    (root / "meep_gpu" / "triton_kernels" / "timing_cc86.json").write_text(
        '{"record_sha256": "b"}')


def test_the_dry_run_over_unchanged_inputs_carries_every_leg(tree, tmp_path):
    source = campaign(tree, tmp_path / "dispatch_fused_route_x_cc86")
    out = tmp_path / "carry.json"
    assert run("route", "--table", "triton", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_CARRIED
    carried = record(out)
    assert carried["counts"]["carried"] == carried["counts"]["legs"] == 3
    assert set(carried["counts"]["by_state"]) == {pc.CARRY_DIGEST}
    assert carried["commands"] == []


def test_a_missing_leg_is_owed_by_name_with_a_fresh_output(tree, tmp_path):
    source = campaign(tree, tmp_path / "c", legs=("shipped", "harness_keep"))
    out = tmp_path / "carry.json"
    assert run("route", "--table", "triton", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_NOT_ALL
    carried = record(out)
    assert carried["legs"]["flush"]["state"] == pc.MISSING
    assert carried["rerun"] == ["flush"]
    assert any("--out <into>/flush" in line for line in carried["commands"])


def test_a_leg_in_two_sources_is_a_duplicate_and_neither_is_taken(tree, tmp_path):
    first = campaign(tree, tmp_path / "a")
    second = campaign(tree, tmp_path / "b", legs=("shipped",))
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", first, "--source", second, "--repo", tree,
        "--device", "cc86", "--out", out)
    carried = record(out)
    assert carried["legs"]["shipped"]["state"] == pc.DUPLICATE
    assert "shipped" not in carried["carried"] and len(carried["carried"]) == 2


def test_a_re_cut_record_is_carried_by_plan_where_the_plans_hold(tree, tmp_path):
    source = campaign(tree, tmp_path / "c")
    move_record(tree)
    out = tmp_path / "carry.json"
    assert run("route", "--table", "triton", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_NOT_ALL
    owed = record(out)
    assert owed["legs"]["shipped"]["state"] == pc.OWED_PLAN
    assert owed["legs"]["flush"]["state"] == pc.CARRY_SHAPE_RULE
    assert any("--plan-only" in line and "<plans>/shipped" in line for line in owed["commands"])
    passes = plans(tree, tmp_path / "plans")
    assert run("route", "--table", "triton", "--source", source, "--repo", tree,
               "--device", "cc86", "--plans", passes, "--out", out) == carry.EXIT_CARRIED
    assert record(out)["legs"]["shipped"]["state"] == pc.CARRY_PLAN
    changed = plans(tree, tmp_path / "plans2", arm_d="PML")
    assert run("route", "--table", "triton", "--source", source, "--repo", tree,
               "--device", "cc86", "--plans", changed, "--out", out) == carry.EXIT_NOT_ALL
    assert record(out)["legs"]["shipped"]["state"] == pc.RERUN_PLAN


def test_place_copies_the_carried_legs_once_and_verify_rechecks_them(tree, tmp_path):
    source = campaign(tree, tmp_path / "c")
    move_record(tree)
    passes = plans(tree, tmp_path / "plans")
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", source, "--repo", tree, "--device", "cc86",
        "--plans", passes, "--out", out)
    into = tmp_path / "placed_campaign"
    assert run("place", "--carry", out, "--into", into) == carry.EXIT_CARRIED
    assert pc.tree_files(into / "shipped") == pc.tree_files(source / "shipped")
    assert run("place", "--carry", out, "--into", into) == carry.EXIT_REFUSED
    assert run("verify", "--carry", into / "CARRY.json", "--into", into,
               "--repo", tree) == carry.EXIT_CARRIED
    carried, _problems = pc.verify_route_carry(record(into / "CARRY.json"), into, tree)
    assert carried["shipped"]["state"] == pc.CARRY_PLAN
    assert [row["case"] for row in carried["shipped"]["plan_pass_rows"]] == ["pml_2d"]
    assert "plan_pass_rows" not in carried["flush"]
    (into / "harness_keep" / "cases.jsonl").write_text("edited\n")
    assert run("verify", "--carry", into / "CARRY.json", "--into", into,
               "--repo", tree) == carry.EXIT_NOT_ALL


def test_place_refuses_a_source_leg_that_moved_after_it_was_judged(tree, tmp_path):
    source = campaign(tree, tmp_path / "c")
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", source, "--repo", tree, "--device", "cc86",
        "--out", out)
    (source / "shipped" / "cases.jsonl").write_text('{"case": "edited"}\n')
    into = tmp_path / "placed"
    assert run("place", "--carry", out, "--into", into) == carry.EXIT_NOT_ALL
    assert not (into / "shipped").exists()
    assert (into / "harness_keep" / "gate.json").is_file()


def test_a_cut_must_name_its_device(tree, tmp_path):
    source = campaign(tree, tmp_path / "c")
    with pytest.raises(SystemExit):
        run("route", "--table", "triton", "--source", source, "--repo", tree,
            "--out", tmp_path / "carry.json")
    with pytest.raises(SystemExit):
        run("l3", "--table", "triton", "--baseline", "keep", "--source", source,
            "--repo", tree, "--out", tmp_path / "carry.json")
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", source, "--repo", tree, "--device", "cc86",
        "--out", out)
    nameless = dict(record(out), device=None)
    carried, problems = pc.verify_route_carry(nameless, source, tree)
    assert carried == {} and problems[0].startswith("carry_record_names_no_device")


def test_a_placement_takes_only_the_campaign_name_the_registry_spells_now(tree, tmp_path):
    from meep_gpu import route_stamps  # noqa: PLC0415
    source = campaign(tree, tmp_path / "c")
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", source, "--repo", tree, "--device", "cc86",
        "--out", out)
    spent = route_stamps.SPENT["triton"][-1]
    assert run("place", "--carry", out, "--into", tmp_path / spent) == carry.EXIT_REFUSED
    other = route_stamps.route_gate("cuda") + "_cc86"
    assert run("place", "--carry", out, "--into", tmp_path / other) == carry.EXIT_REFUSED
    current = route_stamps.route_gate("triton") + "_cc86"
    if route_stamps.campaign_refusal("triton", current) is None:
        assert run("place", "--carry", out, "--into", tmp_path / current) == carry.EXIT_CARRIED


def test_verify_refuses_a_carry_whose_closure_moved_since(tree, tmp_path):
    source = campaign(tree, tmp_path / "c")
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", source, "--repo", tree, "--device", "cc86",
        "--out", out)
    (tree / CODE[0]).write_text("# moved after the carry\n")
    carried, problems = pc.verify_route_carry(record(out), source, tree)
    assert carried == {} and len(problems) == 3
    assert all("carried_leg_no_longer_holds" in line and pc.RERUN_CLOSURE in line
               for line in problems)


def test_verify_refuses_a_plan_pass_that_moved_since(tree, tmp_path):
    source = campaign(tree, tmp_path / "c")
    move_record(tree)
    passes = plans(tree, tmp_path / "plans")
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", source, "--repo", tree, "--device", "cc86",
        "--plans", passes, "--out", out)
    (passes / "shipped" / "plans.json").write_text(
        (passes / "shipped" / "plans.json").read_text() + " ")
    _carried, problems = pc.verify_route_carry(record(out), source, tree)
    assert any(line.startswith("carried_leg_plan_pass_moved: shipped") for line in problems)


def band_witness_summary_keys():
    """``{key: value node}`` of the summary the Metal route gate's band-witness leg writes
    (the dict literal assigned to ``summary`` under ``if arguments.band_witness:``)."""
    source = (HARNESS / "gate_dispatch_metal_route.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                and node.test.attr == "band_witness"):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Dict)
                        and any(isinstance(t, ast.Name) and t.id == "summary"
                                for t in inner.targets)):
                    return {k.value: v for k, v in zip(inner.value.keys, inner.value.values)
                            if isinstance(k, ast.Constant)}
    raise AssertionError("the Metal route gate holds no band-witness summary")


def band_witness_artifact(root):
    """A band-witness gate.json as the gate writes it: its keys and its literal values
    are the gate's own, the closure, planner data and device the synthetic tree's."""
    keys = band_witness_summary_keys()
    payload = {"release": {"released": True, "reasons": []},
               "imported_source_sha256": {name: sha(root / name) for name in CODE},
               "stamped_device": {"compute_capability": "8.6"},
               "provenance": {"python": "3.10.20", "meep": "1.35.0",
                              "band_witness_leg": True}}
    for name, node in keys.items():
        if name == plan_key.INPUTS_FIELD:
            payload[name] = plan_key.plan_inputs(str(root))
        elif name not in payload:
            try:
                payload[name] = ast.literal_eval(node)
            except ValueError:
                payload[name] = None  # computed in the gate; not read by the carry
    return payload


def test_the_band_witness_leg_states_an_empty_plan_map():
    node = band_witness_summary_keys().get("plan_keys")
    assert isinstance(node, ast.Dict) and not node.keys, (
        "the band-witness leg must write plan_keys: {} (it dispatches nothing); without "
        "it every Metal campaign reads RERUN-UNKEYED")
    assert plan_key.INPUTS_FIELD in band_witness_summary_keys()


def test_an_unchanged_metal_campaign_carries_every_leg(tree, tmp_path):
    import metal_round  # noqa: PLC0415
    legs = tuple(metal_round.ROUTE_EXPECT)
    assert "band_witness" in legs
    source = campaign(tree, tmp_path / "m", legs=[l for l in legs if l != "band_witness"],
                      no_default=legs)
    (source / "band_witness").mkdir()
    (source / "band_witness" / "gate.json").write_text(json.dumps(band_witness_artifact(tree)))
    out = tmp_path / "carry.json"
    assert run("route", "--table", "metal", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_CARRIED
    carried = record(out)
    assert carried["counts"]["carried"] == carried["counts"]["legs"] == 4
    assert carried["counts"]["by_state"] == {pc.CARRY_DIGEST: 4}
    move_record(tree)
    assert run("route", "--table", "metal", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_CARRIED
    assert record(out)["counts"]["by_state"] == {pc.CARRY_SHAPE_RULE: 4}


def test_the_metal_leg_arguments_are_the_campaign_driver_s():
    import re  # noqa: PLC0415
    import metal_round  # noqa: PLC0415
    script = (HARNESS / "run_metal_dispatch_campaign.sh").read_text(encoding="utf-8")
    body = script[script.index("leg_arguments() {"):]
    body = body[:body.index("\n}")]
    found = {leg: args.split() for leg, args in
             re.findall(r"^\s*(\w+)\)\s+reply=\(([^)]*)\)", body, flags=re.M)}
    assert found == carry.METAL_LEG_ARGUMENTS
    assert set(carry.METAL_LEG_ARGUMENTS) == set(metal_round.ROUTE_EXPECT)


def test_an_owed_metal_leg_s_plan_pass_runs_under_its_own_arguments(tree, tmp_path):
    import metal_round  # noqa: PLC0415
    legs = [leg for leg in metal_round.ROUTE_EXPECT if leg != "band_witness"]
    source = campaign(tree, tmp_path / "m", legs=legs, no_default=())
    (source / "band_witness").mkdir()
    (source / "band_witness" / "gate.json").write_text(json.dumps(band_witness_artifact(tree)))
    move_record(tree)
    out = tmp_path / "carry.json"
    run("route", "--table", "metal", "--source", source, "--repo", tree, "--device", "cc86",
        "--out", out)
    commands = record(out)["commands"]
    passes = {leg: [line for line in commands if f"<plans>/{leg} " in line] for leg in legs}
    assert passes["harness_flush"] and "--install-policy flush --plan-only" in \
        passes["harness_flush"][0]
    assert passes["harness_keep"] and "--install-policy keep --plan-only" in \
        passes["harness_keep"][0]
    assert passes["shipped"] and "--install-policy" not in passes["shipped"][0]
    assert sum("metal_round.py route" in line for line in commands) == 1


def test_the_metal_campaign_carries_whole_or_re_runs_whole(tree, tmp_path):
    import metal_round  # noqa: PLC0415
    legs = tuple(metal_round.ROUTE_EXPECT)
    source = campaign(tree, tmp_path / "m", legs=legs[1:], no_default=legs)
    out = tmp_path / "carry.json"
    assert run("route", "--table", "metal", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_NOT_ALL
    carried = record(out)
    assert carried["legs"][legs[0]]["state"] == pc.MISSING
    assert {carried["legs"][leg]["state"] for leg in legs[1:]} == {pc.RERUN_WITH_CAMPAIGN}
    assert carried["counts"]["carried"] == 0


# ---------------------------------------------------------------------------- L3 ----

def l3_leg(root, where, labels=("ex.a", "ex.b"), plan_only=False, arm_d="fused pair D",
           table="triton", verdict="holds_dispatched", extra=None):
    (where / "rows").mkdir(parents=True)
    lines = []
    for label in labels:
        row = {"label": label, "table": table, "mode": "keep", "shipped_default": False,
               "sets": ["linux_keep"], "primary": "linux_keep", "image_set": "linux_keep",
               "l3_verdict": "plan_only" if plan_only else verdict,
               "l3_verdicts": {"linux_keep": "plan_only" if plan_only else "holds"},
               "per_set": {"linux_keep": {"stored_equal": 3, "digests": 3}},
               "meep": {"images": {"_meep.so": "m"}, "version": "1.35.0",
                        "single_precision": True},
               "env": {"MEEP_GPU_HARDWARE": "certified", "CUPY_ACCELERATORS": ""},
               plan_key.TOOLCHAIN_FIELD: {"python": "3.10.20", "triton": "3.3.1"},
               plan_key.KEY_FIELD: plan_key.key(report(arm_d), 1),
               plan_key.INPUTS_FIELD: plan_key.plan_inputs(str(root)),
               "imported_source_sha256": {name: sha(root / name) for name in CODE},
               "stamped_device": {"compute_capability": "8.6"}}
        if plan_only:
            row["plan_only"] = True
        row.update((extra or {}).get(label) or {})
        lines.append(json.dumps(row, sort_keys=True))
        (where / "rows" / f"{label}.json").write_text(json.dumps(row) + "\n")
    (where / "rows.jsonl").write_text("".join(line + "\n" for line in lines))
    return where


def test_the_l3_dry_run_over_unchanged_inputs_carries_every_row(tree, tmp_path):
    source = l3_leg(tree, tmp_path / "leg")
    out = tmp_path / "carry.json"
    assert run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86",
               "--out", out) == carry.EXIT_CARRIED
    assert record(out)["counts"] == {"carried": 2, "rows": 2,
                                     "by_state": {pc.CARRY_DIGEST: 2}}


def test_l3_rows_carry_by_plan_and_re_run_where_the_plan_moved(tree, tmp_path):
    source = l3_leg(tree, tmp_path / "leg")
    move_record(tree)
    passes = l3_leg(tree, tmp_path / "plans", labels=("ex.a",), plan_only=True)
    out = tmp_path / "carry.json"
    assert run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86", "--plans",
               passes, "--out", out) == carry.EXIT_NOT_ALL
    carried = record(out)
    assert carried["rows"]["ex.a"]["state"] == pc.CARRY_PLAN
    assert carried["rows"]["ex.b"]["state"] == pc.OWED_PLAN
    moved = l3_leg(tree, tmp_path / "plans2", labels=("ex.a", "ex.b"), plan_only=True,
                   arm_d="PML")
    run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86", "--plans", moved,
        "--out", out)
    assert {v["state"] for v in record(out)["rows"].values()} == {pc.RERUN_PLAN}


def test_l3_missing_duplicate_and_unplanned_rows(tree, tmp_path):
    first = l3_leg(tree, tmp_path / "a", labels=("ex.a", "ex.b"))
    second = l3_leg(tree, tmp_path / "b", labels=("ex.b",))
    expect = tmp_path / "labels.txt"
    expect.write_text("ex.a\nex.b\nex.c\n")
    out = tmp_path / "carry.json"
    run("l3", *LEG, "--source", first, "--source", second, "--repo", tree, "--device", "cc86",
        "--expect-labels", expect, "--out", out)
    carried = record(out)
    assert carried["rows"]["ex.b"]["state"] == pc.DUPLICATE
    assert carried["rows"]["ex.c"]["state"] == pc.MISSING
    assert carried["carried"] == ["ex.a"] and carried["rerun"] == ["ex.b", "ex.c"]
    expect.write_text("ex.b\n")
    run("l3", *LEG, "--source", first, "--repo", tree, "--device", "cc86", "--expect-labels",
        expect, "--out", out)
    carried = record(out)
    assert carried["rows"]["ex.a"].get("outside_the_new_plan") is True
    assert carried["counts"]["rows"] == 1 and carried["carried"] == ["ex.b"]


def test_l3_place_seeds_the_new_leg_and_verify_rechecks_it(tree, tmp_path):
    source = l3_leg(tree, tmp_path / "leg")
    out = tmp_path / "carry.json"
    run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86", "--out", out)
    into = tmp_path / "new_leg"
    assert run("place", "--carry", out, "--into", into) == carry.EXIT_CARRIED
    lines = [json.loads(line) for line in (into / "rows.jsonl").read_text().splitlines()]
    assert [row["label"] for row in lines] == ["ex.a", "ex.b"]
    assert all(row["carried_from"]["state"] == pc.CARRY_DIGEST for row in lines)
    assert (into / "rows" / "ex.a.json").read_bytes() == (source / "rows" / "ex.a.json").read_bytes()
    assert run("verify", "--carry", into / "CARRY.json", "--into", into,
               "--repo", tree) == carry.EXIT_CARRIED
    assert record(into / "CARRY.json")["target"]["sets"] == ["linux_keep"]
    (tree / CODE[1]).write_text("# moved\n")
    assert run("verify", "--carry", into / "CARRY.json", "--into", into,
               "--repo", tree) == carry.EXIT_NOT_ALL
    assert run("place", "--carry", out, "--into", into) == carry.EXIT_REFUSED


def _placed(tree, tmp_path):
    source = l3_leg(tree, tmp_path / "leg")
    out = tmp_path / "carry.json"
    run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86", "--out", out)
    into = tmp_path / "new_leg"
    assert run("place", "--carry", out, "--into", into) == carry.EXIT_CARRIED
    return into


def _verify(into, tree):
    return carry.verify_l3(record(into / "CARRY.json"), into, tree)


@pytest.mark.parametrize("field, value", [("l3_verdict", "differs"),
                                          ("l3_verdicts", {"linux_keep": "differs"}),
                                          ("per_set", {"linux_keep": {"stored_equal": 999}})])
def test_l3_verify_refuses_a_placed_row_whose_fields_moved(tree, tmp_path, field, value):
    into = _placed(tree, tmp_path)
    assert _verify(into, tree) == []
    lines = (into / "rows.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row[field] = value
    lines[0] = json.dumps(row, sort_keys=True)
    (into / "rows.jsonl").write_text("".join(line + "\n" for line in lines))
    problems = _verify(into, tree)
    assert len(problems) == 1 and "not the row that was judged" in problems[0], problems


def test_l3_verify_refuses_a_placed_row_file_that_moved(tree, tmp_path):
    into = _placed(tree, tmp_path)
    (into / "rows" / "ex.a.json").write_text("{}\n")
    problems = _verify(into, tree)
    assert len(problems) == 1 and "placed row file" in problems[0], problems
    (into / "rows" / "ex.a.json").unlink()
    assert "placed row file" in _verify(into, tree)[0]


def test_l3_verify_refuses_rows_placed_for_another_leg(tree, tmp_path):
    into = _placed(tree, tmp_path)
    carried = record(into / "CARRY.json")
    carried["target"] = dict(carried["target"], table="cuda")
    problems = carry.verify_l3(carried, into, tree)
    assert len(problems) == 2 and all("of another leg" in line for line in problems), problems


def test_l3_verify_refuses_a_record_without_a_device_or_a_leg(tree, tmp_path):
    into = _placed(tree, tmp_path)
    for change in ({"device": None}, {"target": None}):
        carried = dict(record(into / "CARRY.json"), **change)
        problems = carry.verify_l3(carried, into, tree)
        assert problems and problems[0].startswith(("carry_record_names_no_device",
                                                    "carry_record_names_no_leg"))


def test_l3_rows_of_another_leg_are_not_carried_into_this_one(tree, tmp_path):
    source = l3_leg(tree, tmp_path / "cuda_leg", table="cuda")
    out = tmp_path / "carry.json"
    assert run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86",
               "--out", out) == carry.EXIT_NOT_ALL
    rows = record(out)["rows"]
    assert {v["state"] for v in rows.values()} == {pc.MISSING}
    assert all(v["other_leg"]["table"] == ["cuda", "triton"] for v in rows.values())
    assert all(v["own_state"] == pc.CARRY_DIGEST for v in rows.values())
    assert run("l3", "--table", "cuda", "--baseline", "keep", "--source", source,
               "--repo", tree, "--device", "cc86", "--out", out) == carry.EXIT_CARRIED


#: A row of a leg that declared it run-to-run variable (l3_corpus.py): its line carries
#: ``attempt`` and the attribution stamp.
def _declared(outcome, why="", stamped=True):
    line = {"attempt": 2}
    if stamped:
        line["l3_attribution"] = {"outcome": outcome, "why": why, "tables": ["aa", "bb"]}
    return line


@pytest.mark.parametrize("extra, carried", [
    (_declared("holds_variable"), True),
    (_declared("differs", "the same source tables gave other words"), False),
    (_declared("undecided", "no run of 1 held so far"), False),
    (_declared(None, stamped=False), False),
])
def test_l3_a_declared_row_carries_only_on_a_holding_stamp(tree, tmp_path, extra, carried):
    # Every line reads l3_verdict holds_dispatched: the stamp alone decides.
    source = l3_leg(tree, tmp_path / "leg", extra={"ex.a": extra})
    out = tmp_path / "carry.json"
    run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86", "--out", out)
    rows = record(out)["rows"]
    assert rows["ex.b"]["state"] == pc.CARRY_DIGEST
    if carried:
        assert rows["ex.a"]["state"] == pc.CARRY_DIGEST
    else:
        assert rows["ex.a"]["state"] == pc.RERUN_UNRELEASED
        assert rows["ex.a"]["own_state"] == pc.CARRY_DIGEST
        assert ("run-to-run variable" in rows["ex.a"]["why"])


def test_l3_place_and_verify_refuse_a_declared_row_whose_stamp_does_not_hold(tree, tmp_path):
    # A record that names a finding carried was not written by classify_l3: place does not
    # write it, and verify reports a placed one.
    source = l3_leg(tree, tmp_path / "leg", extra={"ex.a": _declared("holds_variable")})
    out = tmp_path / "carry.json"
    run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86", "--out", out)
    good = record(out)
    assert good["carried"] == ["ex.a", "ex.b"]
    placed = tmp_path / "placed_ok"
    assert run("place", "--carry", out, "--into", placed) == carry.EXIT_CARRIED
    # The source line turned into a finding, and the record forged to match it.
    lines = (source / "rows.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    rows[0]["l3_attribution"] = {"outcome": "differs", "why": "same tables"}
    text = [json.dumps(row, sort_keys=True) for row in rows]
    (source / "rows.jsonl").write_text("".join(line + "\n" for line in text))
    forged = json.loads(json.dumps(good))
    forged["rows"]["ex.a"]["line_sha256"] = hashlib.sha256(text[0].encode("utf-8")).hexdigest()
    forged["rows"]["ex.a"]["row_sha256"] = carry.canonical_row_sha256(rows[0])
    (tmp_path / "forged.json").write_text(json.dumps(forged))
    into = tmp_path / "placed"
    assert run("place", "--carry", tmp_path / "forged.json", "--into", into) == carry.EXIT_NOT_ALL
    assert [json.loads(line)["label"] for line in (into / "rows.jsonl").read_text().splitlines()] \
        == ["ex.b"]
    # A placed line edited into a finding, with its record forged to match, fails verify.
    placed_rows = [json.loads(line) for line in (placed / "rows.jsonl").read_text().splitlines()]
    placed_rows[0]["l3_attribution"] = {"outcome": "differs", "why": "same tables"}
    (placed / "rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                               for row in placed_rows))
    on_disk = record(placed / "CARRY.json")
    body = {k: v for k, v in placed_rows[0].items() if k != "carried_from"}
    on_disk["rows"]["ex.a"]["row_sha256"] = carry.canonical_row_sha256(body)
    problems = carry.verify_l3(on_disk, placed, tree)
    assert any(line.startswith("ex.a: no longer holds: a row declared run-to-run variable")
               for line in problems), problems


def test_l3_rows_that_did_not_hold_re_run(tree, tmp_path):
    source = l3_leg(tree, tmp_path / "leg", verdict="differs")
    out = tmp_path / "carry.json"
    assert run("l3", *LEG, "--source", source, "--repo", tree, "--device", "cc86",
               "--out", out) == carry.EXIT_NOT_ALL
    carried = record(out)
    assert carried["counts"] == {"carried": 0, "rows": 2,
                                 "by_state": {pc.RERUN_UNRELEASED: 2}}
    assert any("--table triton --baseline keep" in line for line in carried["commands"])


def test_an_l3_leg_l3_corpus_cannot_run_is_refused(tree, tmp_path):
    source = l3_leg(tree, tmp_path / "leg")
    assert run("l3", "--table", "metal", "--baseline", "keep", "--source", source,
               "--repo", tree, "--device", "cc86",
               "--out", tmp_path / "carry.json") == carry.EXIT_USAGE


# ------------------------------------------------- ledgers of the other platform ----

def other_platform_ledger(root, name):
    """Another table's ledger in the tree, moved by the caller."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"arm": {"x": 1}}')
    return path


METAL_LEDGER = "meep_gpu/metal_kernels/fingerprints.json"
TRITON_LEDGER = "meep_gpu/triton_kernels/fingerprints.json"
#: A Metal L3 row as it is recorded: the table's one baseline, stamped with an Apple GPU.
METAL_L3_FACTS = {"mode": "flush", "sets": ["linux_flush", "mac_flush", "linux_keep"],
                  "primary": "linux_flush", "image_set": "mac_flush",
                  "stamped_device": {"unreadable": "no CUDA device was open"},
                  "apple_gpu": {"architecture": "applegpu_g13s"}}


def test_route_a_moved_metal_ledger_carries_an_nvidia_campaign_and_its_own_does_not(
        tree, tmp_path):
    ledger = other_platform_ledger(tree, METAL_LEDGER)
    source = campaign(tree, tmp_path / "c", no_default=TRITON_LEGS)
    out = tmp_path / "carry.json"
    ledger.write_text('{"arm": {"x": 2}}')
    assert run("route", "--table", "triton", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_CARRIED
    assert record(out)["counts"]["by_state"] == {pc.CARRY_DIGEST: len(TRITON_LEGS)}
    (tree / TRITON_LEDGER).write_text('{"arm": {"x": 2}}')
    assert run("route", "--table", "triton", "--source", source, "--repo", tree,
               "--device", "cc86", "--out", out) == carry.EXIT_NOT_ALL
    assert record(out)["counts"]["by_state"] == {pc.RERUN_INPUTS: len(TRITON_LEGS)}


def test_l3_a_moved_nvidia_ledger_carries_a_metal_leg_and_its_own_does_not(tree, tmp_path):
    ledger = other_platform_ledger(tree, METAL_LEDGER)
    source = l3_leg(tree, tmp_path / "leg", table="metal",
                    extra={"ex.a": METAL_L3_FACTS, "ex.b": METAL_L3_FACTS})
    target = ["--table", "metal", "--baseline", "flush",
              "--sets", "linux_flush,mac_flush,linux_keep", "--primary", "linux_flush",
              "--image-set", "mac_flush"]
    out = tmp_path / "carry.json"
    (tree / TRITON_LEDGER).write_text('{"arm": {"x": 2}}')
    assert run("l3", *target, "--source", source, "--repo", tree,
               "--device", "applegpu_g13s", "--out", out) == carry.EXIT_CARRIED
    assert record(out)["counts"] == {"carried": 2, "rows": 2,
                                     "by_state": {pc.CARRY_DIGEST: 2}}
    ledger.write_text('{"arm": {"x": 2}}')
    assert run("l3", *target, "--source", source, "--repo", tree,
               "--device", "applegpu_g13s", "--out", out) == carry.EXIT_NOT_ALL
    assert record(out)["counts"]["by_state"] == {pc.RERUN_INPUTS: 2}


def test_the_tool_says_how_many_moved_ledger_digests_no_state_answers_for(tree, tmp_path, capsys):
    """A moved ledger on a tree with no archive compares whole, by name, and stays visible."""
    source = campaign(tree, tmp_path / "c")
    name = "meep_gpu/triton_kernels/fingerprints.json"
    (tree / name).write_text(json.dumps({"moved": 1}))
    out = tmp_path / "carry.json"
    run("route", "--table", "triton", "--source", source, "--repo", tree, "--device", "cc86",
        "--out", out)
    printed = capsys.readouterr().out
    assert "RERUN-INPUTS" in printed
    assert "ledger states not archived: 0 of 1 moved ledger digests" in printed
