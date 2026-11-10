"""The harness half of the size-aware dispatch preference, on the host (no GPU, no MEEP).

* the record cutter's ``/3`` records (``cut_timing_record.cut(costs=True)``, ``--costs``):
  the same keys as ``/2`` plus the cost points ``dispatch_preference.add_costs`` derives
  from them, stamped so ``--check`` recomputes either kind byte for byte;
* the canonical selected plan (``selected_plan.py``): its slot order is the driver's,
  a full dispatch record and its stored summary give one digest, and the published test
  vectors hold;
* the plan-change tool (``plan_changes.py``): which cases it freezes, where a candidate
  record lands (as its architecture's representative, and never two for one
  architecture), how two sides compare and summarize, and that a side resumes only from
  rows frozen without an error under the records it reads now.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cut_timing_record as cut  # noqa: E402
import plan_changes as pc  # noqa: E402
import selected_plan as sp  # noqa: E402
from meep_gpu import dispatch_preference as dp  # noqa: E402
from test_cut_timing_record import (ask_cells, bench_row, cut_tree,  # noqa: E402
                                    entries, out_path, run, standard_tree, sweep_tree,
                                    write)


# ---------------------------------------------------------------------------
# The cutter's /3 records
# ---------------------------------------------------------------------------

def test_a_v3_cut_is_the_v2_cut_with_its_cost_points(tmp_path):
    sweep_tree(tmp_path, [(24000, 0.35), (2400000, 0.73), (5400000, 1.08)])
    plain = cut_tree(tmp_path)
    costed = cut_tree(tmp_path, costs=True)
    assert plain["schema"] == dp.SCHEMA and costed["schema"] == dp.SCHEMA_COSTS
    assert costed["keys"] == plain["keys"]
    assert costed["cut"]["costs"] is True and "costs" not in plain["cut"]
    assert "costs" in costed["rules"] and "costs" not in plain["rules"]
    assert costed["costs"] == dp.build_costs(plain["keys"], plain["cut"]["min_rows"])
    curve = costed["costs"]["fused pair D|real|2|0|bracketed"]
    assert [p["cells"] for p in curve["points"]] == [24000, 2400000, 5400000]


def test_a_v2_cut_restamped_is_byte_for_byte_the_v3_cut(tmp_path, capsys):
    """A landing can take a /2 record to /3 without the rows: the package's re-stamp is
    the cutter's own, and --check recomputes the re-stamped record from its rows."""
    sweep_tree(tmp_path, [(24000, 0.35), (2400000, 0.73), (5400000, 1.08)])
    restamped = dp.add_costs(cut_tree(tmp_path))
    assert dp.dumps(restamped) == dp.dumps(cut_tree(tmp_path, costs=True))
    out = out_path(tmp_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dp.dumps(restamped))
    capsys.readouterr()
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_OK


def test_the_cut_curve_prices_the_gap_its_bands_leave(tmp_path):
    sweep_tree(tmp_path, [(24000, 0.35), (2400000, 0.73), (5400000, 1.08)])
    costed = cut_tree(tmp_path, costs=True)

    def ask(cells):
        return dp.consult_cost(
            costed, table="triton", capability=dp.record_capabilities(costed)[0],
            candidate="fused pair D", displaces=("PML", "ordinary"),
            run_shape={"dimensions": 2, "grid_shape": [cells, 1, 1]}, bracketed=True,
            mode="measured", repair_route=costed["route"],
            subject_sha256={source: digests[0] for source, digests in
                            costed["subject_barrier"]["digests"].items()})

    assert ask(3000000).vetoed and ask(24000).vetoed
    assert ask(5400000).outcome == "not_vetoed"
    assert ask(23999).reason == "outside_the_cost_points"


def _ask_cost(record, cells):
    return dp.consult_cost(
        record, table="triton", capability=dp.record_capabilities(record)[0],
        candidate="fused pair D", displaces=("PML", "ordinary"),
        run_shape={"dimensions": 2, "grid_shape": [cells, 1, 1]}, bracketed=True,
        mode="measured", repair_route=record["route"],
        subject_sha256=dp.compared_emitters(record))


@pytest.mark.parametrize("legs, outcome", [
    # one row slower beyond its spreads (0.90x) and one not (0.98x) at each size: the
    # band keeps, and the medians (0.94x against 0.03) would read slower
    (((0.90, 0.02, 0.01), (0.98, 0.02, 0.01)), "not_vetoed"),
    # both rows slower beyond their own spreads at each size: the band vetoes, and the
    # medians (0.80x) against the largest spreads (0.20 + 0.04) would read not slower
    (((0.70, 0.20, 0.01), (0.90, 0.01, 0.04)), "vetoed")])
def test_a_cut_band_and_its_curve_give_one_verdict_at_and_between_its_sizes(
        tmp_path, legs, outcome):
    rows = []
    for index, cells in enumerate((24000, 2400000)):
        for repeat, (base_ms, fused_spread, base_spread) in enumerate(legs):
            rows.append(bench_row(case=f"two_{cells}_{repeat}", grid=(cells, 1, 1),
                                  fused_ms=1.0, base_ms=base_ms,
                                  fused_spread=fused_spread, base_spread=base_spread,
                                  utc=f"2026-09-19T13:{index:02d}:{repeat:02d}Z"))
    write(tmp_path, "tree/rows.jsonl", rows)
    costed = cut_tree(tmp_path, costs=True)
    entry, = entries(costed, "fused pair D")
    assert entry["band"] == {"lo_cells": 24000, "hi_cells": 2400000}
    for cells in (24000, 24001, 1212000, 2399999, 2400000):
        assert ask_cells(costed, cells) == _ask_cost(costed, cells).outcome == outcome, cells


def test_the_command_line_cuts_a_v3_record_and_checks_both_kinds(tmp_path, capsys):
    standard_tree(tmp_path)
    out = out_path(tmp_path)
    assert run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"),
               "--out", str(out), "--costs") == 0
    assert dp.load_record(str(out))["schema"] == dp.SCHEMA_COSTS
    capsys.readouterr()
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_OK
    assert "recomputes byte for byte" in capsys.readouterr().out
    assert run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"),
               "--out", str(out)) == 0
    assert dp.load_record(str(out))["schema"] == dp.SCHEMA
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_OK


def test_a_v3_record_whose_curve_was_edited_refuses_on_check(tmp_path, capsys):
    standard_tree(tmp_path)
    out = out_path(tmp_path)
    assert run(tmp_path, "--table", "triton", "--rows", str(tmp_path / "tree"),
               "--out", str(out), "--costs") == 0
    record = json.loads(out.read_text())
    name = sorted(record["costs"])[0]
    record["costs"][name]["points"][0]["rows"] += 1
    out.write_text(dp.dumps(dp.seal(record)))
    capsys.readouterr()
    assert run(tmp_path, "--check", str(out)) == cut.EXIT_REFUSED
    assert "do not give" in capsys.readouterr().out


def test_the_timing_run_record_step_cuts_v3_records_when_asked(tmp_path):
    import timing_rows as TR
    from test_timing_rows import ARCH, RELEASED, fused_row, release_tree

    root = release_tree(tmp_path)
    store = str(tmp_path / "store")
    for index in (1, 2):
        fused_row(tmp_path, store, case="pml_2d", architecture=ARCH, session=f"s{index}",
                  ended=f"2026-10-2{index}T10:00:00Z", pass_index=index)
    selection = TR.select(TR.load_store(store), root)
    for costs, schema in ((False, dp.SCHEMA), (True, dp.SCHEMA_COSTS)):
        out = str(tmp_path / f"package_{costs}")
        results = TR.cut_records(selection, out, base=str(tmp_path),
                                 released={"triton": RELEASED}, costs=costs)
        assert [r.get("refused") for r in results] == [None]
        record = dp.load_record(dp.record_path("triton", ARCH, root=out), "triton")
        assert record["schema"] == schema
        assert ("costs" in record) is costs


# ---------------------------------------------------------------------------
# The canonical selected plan
# ---------------------------------------------------------------------------

FULL = {"decision": "dispatched", "table": "triton", "slots": {
    "step_B": {"state": "dispatched", "arm": "fused pair B"},
    "fill_B": {"state": "array", "arm": None},
    "update_H": {"state": "dispatched", "arm": "fused pair B"},
    "step_D": {"state": "dispatched", "arm": "PML"},
    "fill_D": {"state": "array", "arm": None},
    "update_E": {"state": "dispatched", "arm": "ordinary"},
    "update_P": {"state": "array", "arm": None}}}
SUMMARY = {"decision": "dispatched", "composition": {"table": "triton"},
           "arms": {"update_E": "ordinary", "step_B": "fused pair B", "step_D": "PML",
                    "update_H": "fused pair B"}}


def test_the_slot_order_is_the_drivers():
    from meep_gpu import fastpath
    from meep_gpu.triton_kernels import launch

    assert sp.SLOT_ORDER == tuple(fastpath.DRIVER_SLOTS) == tuple(launch.STEP_ORDER)


def test_the_published_test_vectors_hold():
    plan = sp.selected_plan(FULL)
    assert sp.canonical_bytes(plan) == (
        b'{"decision":"dispatched","dispatched":[["step_B","fused pair B"],'
        b'["update_H","fused pair B"],["step_D","PML"],["update_E","ordinary"]],'
        b'"platform":"nvidia","schema":"meep_gpu.selected_plan/1"}')
    assert sp.plan_sha256(plan) == (
        "ba89aa0ace0ce81b8828e534650873f21c9aa0fb0bb1ac640b0428942fc330cd")
    assert sp.selected_plan_sha256({"decision": "refused", "refused_because": "x"}) == (
        "735c085bcf5d58a03036fe4a06b26ffe9dc3faadd68864e754fe08ebfeb20d4f")
    cuda = {"decision": "dispatched", "table": "cuda", "slots": {
        "step_B": {"state": "dispatched", "arm": "cuda:fused magnetic pair"},
        "update_H": {"state": "dispatched", "arm": "cuda:fused magnetic pair"},
        "step_D": {"state": "dispatched", "arm": "cuda:PML", "backend": "cuda"},
        "update_E": {"state": "dispatched", "arm": "cuda:ordinary"}}}
    assert sp.selected_plan_sha256(cuda) == (
        "104f24552ae01b9b4c8bae0a13019cd5a997eeb71ba7e5bf3405f10540323372")


def test_a_record_and_its_stored_summary_are_one_plan():
    assert sp.selected_plan(FULL) == sp.selected_plan(SUMMARY)


def test_the_plan_moves_with_an_arm_and_with_nothing_else():
    other = json.loads(json.dumps(FULL))
    other["slots"]["update_P"]["reason"] = "another host's wording"
    other["refused_because"] = None
    other["environment"] = {"device": {"name": "ANOTHER CARD"}}
    assert sp.selected_plan_sha256(other) == sp.selected_plan_sha256(FULL)
    other["slots"]["step_D"]["arm"] = "fused pair D"
    assert sp.selected_plan_sha256(other) != sp.selected_plan_sha256(FULL)
    metal = dict(FULL, table="metal")
    assert sp.selected_plan(metal)["platform"] == "metal"
    assert sp.selected_plan_sha256(metal) != sp.selected_plan_sha256(FULL)


def test_a_slot_the_driver_does_not_name_sorts_after_its_own():
    record = json.loads(json.dumps(FULL))
    record["slots"]["aux_Z"] = {"state": "dispatched", "arm": "x"}
    record["slots"]["aux_A"] = {"state": "dispatched", "arm": "y"}
    slots = [slot for slot, _arm in sp.selected_plan(record)["dispatched"]]
    assert slots == ["step_B", "update_H", "step_D", "update_E", "aux_A", "aux_Z"]


# ---------------------------------------------------------------------------
# The plan-change tool
# ---------------------------------------------------------------------------

def test_a_campaign_freezes_its_drive_rows_and_the_catalogs_sweep():
    import gate_dispatch_fused_route as route

    rows = pc.case_rows("triton")
    drive = [row for row in rows if row[1] is None]
    assert [case for case, _res in drive] == list(route.DRIVE)
    sweep = [row for row in rows if row[1] is not None]
    assert ("pml_2d", 300) in sweep and ("offdiag_2d", 20) in sweep
    cuda = pc.case_rows("cuda")
    assert [case for case, res in cuda if res is None] == list(route.DRIVE_CUDA)
    assert ("offdiag_2d", 20) not in cuda
    assert pc.case_rows("cuda_alone") == cuda
    metal = pc.case_rows("metal")
    assert all(res is None for _case, res in metal) and metal
    assert pc.case_rows("triton", cases=["pml_2d"], sweep=False) == [("pml_2d", None)]


def _record_file(path, table, capability):
    from meep_gpu.test_dispatch_preference import HOST, _entry, _record

    entry = _entry("fused pair D", bracketed=True, lo=1000, hi=100000, fused=1.4,
                   baseline=0.5, displaces=("PML", "ordinary"))
    record = _record([entry], table=table,
                     host=[dict(HOST, compute_capability=capability)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dp.dumps(record))
    return record


def test_a_candidate_lands_where_its_own_table_and_architecture_say(tmp_path):
    candidate = _record_file(tmp_path / "in" / "anything.json", "cuda", "8.9")
    root = pc.overlay([str(tmp_path / "in" / "anything.json")], [],
                      str(tmp_path / "root"))
    placed = pathlib.Path(dp.record_path("cuda", "8.9", root=root))
    assert json.loads(placed.read_text()) == candidate
    shipped = dp.record_files(pc.package_root())
    names = {os.path.relpath(path, root) for path in dp.record_files(root)}
    assert {os.path.relpath(path, pc.package_root()) for path in shipped} <= names
    assert os.path.relpath(str(placed), root) in names


def test_a_candidate_directory_replaces_the_shipped_record_it_names(tmp_path):
    directory = tmp_path / "records"
    target = pathlib.Path(dp.record_path("triton", "8.6", root=str(directory)))
    candidate = _record_file(target, "triton", "8.6")
    root = pc.overlay([], [str(directory)], str(tmp_path / "root"))
    replaced = pathlib.Path(dp.record_path("triton", "8.6", root=root))
    assert json.loads(replaced.read_text()) == candidate
    assert pc.records_at(root)["triton_kernels/timing_cc86.json"]["record_sha256"] == (
        candidate[dp.DIGEST_FIELD])


def test_two_candidates_for_one_architecture_are_refused_and_lay_nothing(tmp_path,
                                                                         capsys):
    """A staging directory holding a bare and a card-named cut of one table and
    architecture: each would be laid as the representative, so the overlay refuses to
    choose, by name, before it lays anything."""
    directory = tmp_path / "records"
    bare = pathlib.Path(dp.record_path("triton", "8.6", root=str(directory)))
    _record_file(bare, "triton", "8.6")
    _record_file(bare.with_name("timing_cc86-example_gpu.json"), "triton", "8.6")
    into = tmp_path / "root"
    with pytest.raises(pc.OverlayRefused) as refused:
        pc.overlay([], [str(directory)], str(into))
    assert refused.value.code == "two_candidates_for_one_architecture"
    assert "timing_cc86-example_gpu.json" in str(refused.value)
    assert not into.exists()
    # a file named and the same record in a directory are two candidates too
    with pytest.raises(pc.OverlayRefused):
        pc.overlay([str(bare)], [str(bare.parent.parent)], str(into))
    assert not into.exists()
    # one candidate is laid, and its line names the card its rows ran on
    capsys.readouterr()
    pc.overlay([str(bare)], [], str(into))
    said = capsys.readouterr().out
    assert "triton_kernels/timing_cc86.json" in said and "card EXAMPLE GPU" in said


def _side_row(case, plan_arm, res=None, error=None, block="b0"):
    record = json.loads(json.dumps(FULL))
    record["slots"]["step_D"]["arm"] = plan_arm
    plan = sp.selected_plan(record)
    row = {"campaign": "triton", "case": case, "res": res, "cells": 24000,
           "reached_by": "release", "plan": plan, "plan_sha256": sp.plan_sha256(plan),
           "withheld": [], "preference_block_sha256": block,
           "records": {"triton_kernels/timing_cc86.json": {"record_sha256": "r"}}}
    if error:
        row.update(error=error, plan=None, plan_sha256=None)
    return row


def test_two_sides_compare_case_by_case():
    before = [_side_row("a", "PML"), _side_row("b", "PML"), _side_row("c", "PML"),
              _side_row("a", "PML", res=300)]
    after = [_side_row("a", "fused pair D"), _side_row("b", "PML", block="b1"),
             _side_row("c", "PML", error="rc 1: lift failed"),
             _side_row("a", "PML", res=300)]
    rows = pc.compare(before, after)
    by = {(row["case"], row["res"]): row for row in rows}
    assert by[("a", None)]["changed"] is True
    assert by[("b", None)]["changed"] is False
    assert by[("b", None)]["preference_block_changed"] is True
    assert by[("c", None)]["changed"] is False
    assert by[("a", 300)]["changed"] is False
    assert all(row["schema"] == pc.ROW_SCHEMA for row in rows)
    summary = pc.summarize(rows, "triton")
    assert summary["schema"] == pc.SUMMARY_SCHEMA
    assert (summary["cases"], summary["changed"], summary["unchanged"],
            summary["errors"]) == (4, 1, 2, 1)
    assert summary["changed_cases"] == [["a", None]]
    assert summary["error_cases"] == [["c", None]]


def _children(monkeypatch, digest, fail=()):
    """A fake child per case: its row names the records ``digest["now"]`` says the root
    holds (as ``records_at`` digests the files there); a case in ``fail`` errors once."""
    calls = []
    failed = set()

    def records_at(root):
        return {"triton_kernels/timing_cc86.json": {"record_sha256": digest["now"]}}

    def child(campaign, case, res, records_root, log):
        calls.append((case, res))
        if case in fail and case not in failed:
            failed.add(case)
            return dict(_side_row(case, "PML", res=res, error="rc 1: transient"),
                        records=records_at(records_root), elapsed_s=0.0)
        return dict(_side_row(case, "PML", res=res), records=records_at(records_root),
                    elapsed_s=0.0)

    monkeypatch.setattr(pc, "records_at", records_at)
    monkeypatch.setattr(pc, "run_child", child)
    return calls


def test_a_side_resumes_from_the_rows_it_already_holds(tmp_path, monkeypatch):
    calls = _children(monkeypatch, {"now": "r"})
    out = str(tmp_path / "side.jsonl")
    pc.freeze("triton", [("a", None)], None, out, None)
    rows = pc.freeze("triton", [("a", None), ("b", None)], None, out, None)
    assert calls == [("a", None), ("b", None)]
    assert [row["case"] for row in rows] == ["a", "b"]
    assert len(pathlib.Path(out).read_text().splitlines()) == 2


def test_a_second_report_into_the_same_directory_freezes_under_its_own_records(
        tmp_path, monkeypatch):
    """``report --record B --out DIR`` after ``--record A --out DIR``: the candidate root is
    rebuilt with B's file, so every row frozen under A is frozen again, and the row
    appended last is the one read."""
    digest = {"now": "A"}
    calls = _children(monkeypatch, digest)
    out = str(tmp_path / "after.jsonl")
    root = str(tmp_path / "candidate_root")
    pc.freeze("triton", [("a", None)], root, out, None)
    digest["now"] = "B"
    rows = pc.freeze("triton", [("a", None)], root, out, None)
    assert calls == [("a", None), ("a", None)]
    assert rows[0]["records"]["triton_kernels/timing_cc86.json"]["record_sha256"] == "B"
    assert pc._read_rows(out)[-1]["records"] == rows[0]["records"]  # noqa: SLF001
    assert pc.freeze("triton", [("a", None)], root, out, None) == rows
    assert calls == [("a", None), ("a", None)]


def test_a_row_that_failed_is_frozen_again(tmp_path, monkeypatch):
    calls = _children(monkeypatch, {"now": "r"}, fail=("a",))
    out = str(tmp_path / "side.jsonl")
    first, = pc.freeze("triton", [("a", None)], None, out, None)
    assert first["error"] == "rc 1: transient"
    again, = pc.freeze("triton", [("a", None)], None, out, None)
    assert "error" not in again and again["plan_sha256"]
    assert calls == [("a", None), ("a", None)]


def test_a_child_that_fails_is_an_error_row_not_a_plan(tmp_path):
    row = pc.run_child("triton", "no_such_case", None, None, str(tmp_path / "log.txt"))
    assert row["plan"] is None and row["plan_sha256"] is None
    assert row["error"].startswith("rc ")
    assert "no_such_case" in (tmp_path / "log.txt").read_text()
