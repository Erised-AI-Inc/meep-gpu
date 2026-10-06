"""``build_identical_case_table.py`` on a synthetic ladder unlike 2026-09-28's.

Two cases at one size, two passes, MEEP at 3 and 6 ranks plus a chunk-splitting control
at 6, one GPU route on the Metal table: every number, rank count, route and pass count
differs from the 2026-09-28 ladder, so a literal left in the builder would show here.
The ledger is written by the ladder's own recorders, with ``run_dir`` pointing at a
directory that does not exist, so the builder must map the paths onto the pull.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_identical_case_table as table  # noqa: E402
import timing_records as tr  # noqa: E402

REMOTE = "/remote/host/ladder_x"
DIGEST = {"pml_3d": "a" * 64, "thin_pml_3d": "b" * 64}
CELLS = 64_000  # pml_3d at resolution 10, thin_pml_3d at 5
RES = {"pml_3d": 10, "thin_pml_3d": 5}


def gpu_row(case, rate, spread=0.01):
    ms = CELLS / (rate * 1e6) * 1e3
    return {"case": case, "grid_shape": [40, 40, 40], "verdict": "TIMED", "floors": {},
            "substitution": {"tables_dispatched": ["metal"], "proved": True},
            "bit_identity": {"fused_vs_array_identical": True},
            "plans": {"fused": {"launch_counters": {"step_B": {"backend": "metal",
                                                              "arm": "fused pair B"}}}},
            "per_leg": {"fused": {"median_seconds_per_step": ms * 1e-3, "spread": spread,
                                  "launches_per_step": 3.0,
                                  "deposit_repair_route": {"bracketed_slots": []}},
                        "unfused": {"median_seconds_per_step": ms * 1.1e-3, "spread": 0.01},
                        "array": {"median_seconds_per_step": ms * 5e-3, "spread": 0.01,
                                  "launches_per_step": 0.0}},
            "box_before": {}, "box_after": {}}


def meep_measurement(case, ranks, rate, pass_index, configuration=None, spread=0.01):
    row = {"row": "measurement", "row_id": f"{case}_r{ranks}_p{pass_index}"
           + ("_split" if configuration else ""), "case": case, "cells": CELLS,
           "dimensions": 3, "ranks": ranks, "pass": pass_index, "mcell_steps_per_s": rate,
           "ms_per_step": CELLS / rate / 1e3, "spread": spread, "windows": 6,
           "steps_per_window": 100, "stepper": "fields_step", "init_seconds": 1.0,
           "geometry_digest": DIGEST[case], "monitor": {"stepped": True},
           "mcell_steps_per_s_windows": [rate] * 6, "drift_last_over_first": 1.0,
           "warm_up": {"steps": 400, "steps_asked": 400, "stopped_by": "steps"},
           "environment": {"meep": "1.33.0", "single_precision": True,
                           "binding_label": "unbound", "binding": {"bound": False}}}
    if configuration:
        row["configuration"] = configuration
    return row


@pytest.fixture
def pull(tmp_path):
    root = tmp_path / "pull"
    ledger = str(root / "ladder_rows.jsonl")
    (root / "digests").mkdir(parents=True)
    index = 0
    meep_rates = {("pml_3d", 3): (100.0, 101.0), ("pml_3d", 6): (180.0, 179.0),
                  ("thin_pml_3d", 3): (90.0, 91.0), ("thin_pml_3d", 6): (150.0, 140.0)}
    for pass_index in (1, 2):
        for case in ("pml_3d", "thin_pml_3d"):
            index += 1
            row_dir = root / "gpu_metal" / f"pass{pass_index}" / f"{case}_res{RES[case]}"
            row_dir.mkdir(parents=True)
            (row_dir / "rows.jsonl").write_text(json.dumps(gpu_row(
                case, 400.0 + pass_index)) + "\n")
            tr.record_gpu(ledger, str(row_dir).replace(str(root), REMOTE), None, "metal",
                          None, False, {"index": index, "of": 24, "tier": "priority",
                                        "route": "metal", "table_asked": "metal",
                                        "res": RES[case], "pass": pass_index, "case": case,
                                        "run_dir": REMOTE})
            # the recorder read the REMOTE path, which does not exist here: rewrite it
            # with the local file so the measured block is filled, as on the host
            entries = [json.loads(l) for l in open(ledger)]
            entries[-1]["measured"] = tr.gpu_facts(gpu_row(case, 400.0 + pass_index), None)
            entries[-1].pop("struck", None)
            with open(ledger, "w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry) + "\n")
            for ranks in (3, 6):
                index += 1
                rows_file = root / "meep" / f"rows_r{ranks}.jsonl"
                rows_file.parent.mkdir(parents=True, exist_ok=True)
                measurement = meep_measurement(case, ranks,
                                               meep_rates[(case, ranks)][pass_index - 1],
                                               pass_index)
                with open(rows_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(measurement) + "\n")
                local, _ = tr.record_meep(
                    str(root / "scratch.jsonl"), str(rows_file), measurement["row_id"], None,
                    str(root / "digests" / f"{case}_res{RES[case]}.sha256"), None, {})
                local.update({"index": index, "of": 24, "tier": "priority", "ranks": ranks,
                              "res": RES[case], "pass": pass_index, "case": case,
                              "binding": "unbound", "run_dir": REMOTE,
                              "rows_file": str(rows_file).replace(str(root), REMOTE)})
                tr.append(ledger, local)
            index += 1
            split_file = root / "meep_splitcost" / "rows_r6.jsonl"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            split = meep_measurement(case, 6, 200.0 if case == "pml_3d" else 100.0,
                                     pass_index, configuration={
                                         "split_chunks_evenly": False})
            with open(split_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(split) + "\n")
            local, _ = tr.record_meep(str(root / "scratch.jsonl"), str(split_file),
                                      split["row_id"], None, None, DIGEST[case], {})
            local.update({"index": index, "of": 24, "tier": "priority", "ranks": 6,
                          "res": RES[case], "pass": pass_index, "case": case,
                          "binding": "unbound", "control": "split", "run_dir": REMOTE,
                          "rows_file": str(split_file).replace(str(root), REMOTE)})
            tr.append(ledger, local)
            index += 1
            control = {"ledger": "array_control", "tier": "priority", "res": RES[case],
                       "pass": pass_index, "case": case, "index": index,
                       "mcell_steps_per_s_mean": 80.0 + pass_index,
                       "sources": [{"mcell_steps_per_s": 80.0 + pass_index, "spread": 0.01,
                                    "kernels_vetoed": True}]}
            tr.append(ledger, control)
    (root / "environment.json").write_text(json.dumps({
        "params": {"CASES": {"value": "pml_3d,thin_pml_3d"}}, "gpu": None,
        "platform": "darwin", "topology": {"physical_cores": 6, "logical_cpus": 8},
        "packages": {"meep": "1.33.0", "meep_single_precision": True}}))
    return root


def build(pull_root, tmp_path, *extra):
    md, js = tmp_path / "t.md", tmp_path / "t.json"
    assert table.main(["--pull", str(pull_root), "--out-md", str(md), "--out-json", str(js),
                       "--ratio-ranks", "3"] + list(extra)) == 0
    return json.loads(js.read_text()), md.read_text()


def test_a_synthetic_ladder_builds_with_nothing_typed(pull, tmp_path):
    record, text = build(pull, tmp_path)
    assert record["passes"] == [1, 2]
    assert record["routes"] == {"metal": {"table_asked": "metal",
                                          "label": "GPU metal route (metal)"}}
    sizes = {(s["case"], s["cells"]): s for s in record["sizes"]}
    assert set(sizes) == {("pml_3d", CELLS), ("thin_pml_3d", CELLS)}
    pml = sizes[("pml_3d", CELLS)]
    assert set(pml["legs"]) == {"metal", "array", "meep@3", "meep@6", "meep@6[split]"}
    assert pml["legs"]["metal"]["median"] == pytest.approx(401.5)
    assert pml["legs"]["meep@6"]["median"] == pytest.approx(179.5)
    # MEEP's best over every configuration, with the reference build's best beside it
    assert (pml["meep_best"]["configuration"], pml["meep_best"]["ranks"]) == ("split", 6)
    assert pml["meep_best"]["reference_ranks"] == 6
    denominators = {(r["gpu"], r["denominator"]) for r in pml["ratios"]}
    assert ("metal", "best") in denominators and ("metal", "reference_best") in denominators
    assert ("metal", 3) in denominators
    best = next(r for r in pml["ratios"] if r["gpu"] == "metal" and r["denominator"] == "best")
    assert best["value"] == pytest.approx(401.5 / 200.0)
    assert best["provisional"] is True
    assert "harness-lift digest pending; ratios provisional" in text
    assert "`pml_3d`" in text and "`thin_pml_3d`" in text


def test_both_gate_rules(pull, tmp_path):
    record, _text = build(pull, tmp_path)
    thin = next(s for s in record["sizes"] if s["case"] == "thin_pml_3d")
    # MEEP@6 on thin_pml_3d: 150 then 140, a 6.9 % between-pass spread
    assert thin["legs"]["meep@6"]["passes_gate"] is False
    assert thin["row_passes_every_leg"] is False
    entering = {(r["gpu"], r["denominator"]): r["enters_per_ratio"] for r in thin["ratios"]}
    assert entering[("metal", 3)] is True
    assert all(r["enters_per_row"] is False for r in thin["ratios"])
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert pml["row_passes_every_leg"] is True
    assert all(r["enters_per_ratio"] for r in pml["ratios"])


def test_an_uncertified_gpu_leg_is_struck(pull, tmp_path):
    ledger = pull / "ladder_rows.jsonl"
    entries = [json.loads(line) for line in open(ledger)]
    for entry in entries:
        if entry.get("ledger") == "gpu" and entry.get("case") == "pml_3d":
            entry["allow_uncertified"] = 1
    ledger.write_text("".join(json.dumps(e) + "\n" for e in entries))
    record, _text = build(pull, tmp_path)
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert pml["legs"]["metal"]["passes_gate"] is False
    assert any("MEEP_GPU_ALLOW_UNCERTIFIED=1" in strike
               for strike in pml["legs"]["metal"]["strikes"])
    assert not any(r["enters_per_ratio"] for r in pml["ratios"] if r["gpu"] == "metal")


def test_a_lift_digest_settles_the_digest_of_record(pull, tmp_path):
    lifts = tmp_path / "lifts"
    lifts.mkdir()
    for case, digest in DIGEST.items():
        (lifts / f"harness_lift_digest_{case}_res{RES[case]}.json").write_text(json.dumps(
            {"case": case, "resolution": RES[case], "verdict": {"driver_agrees": True},
             "geometry": {"digest": digest, "facts": {"cells": CELLS}}}))
    record, text = build(pull, tmp_path, "--lift-digests", str(lifts))
    assert {s["digest"]["status"] for s in record["sizes"]} == {"equal"}
    assert all(not r["provisional"] for s in record["sizes"] for r in s["ratios"])
    assert all(not s["size_strikes"] for s in record["sizes"])
    assert "ratios provisional" not in text


def test_a_lift_whose_driver_disagreed_settles_nothing(pull, tmp_path):
    lifts = tmp_path / "lifts"
    lifts.mkdir()
    (lifts / "harness_lift_digest_pml_3d_res10.json").write_text(json.dumps(
        {"case": "pml_3d", "resolution": 10, "verdict": {"driver_agrees": False},
         "geometry": {"digest": DIGEST["pml_3d"], "facts": {"cells": CELLS}}}))
    record, _text = build(pull, tmp_path, "--lift-digests", str(lifts))
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert any("driver disagreed" in strike for strike in pml["size_strikes"])
    assert not any(r["enters_per_ratio"] or r["enters_per_row"] for r in pml["ratios"])


def _rewrite(pull, change):
    ledger = pull / "ladder_rows.jsonl"
    entries = [json.loads(line) for line in open(ledger)]
    for entry in entries:
        change(entry)
    ledger.write_text("".join(json.dumps(e) + "\n" for e in entries))


def test_a_lift_digest_that_differs_strikes_every_ratio_at_its_size(pull, tmp_path):
    lifts = tmp_path / "lifts"
    lifts.mkdir()
    (lifts / "harness_lift_digest_pml_3d_res10.json").write_text(json.dumps(
        {"case": "pml_3d", "resolution": 10, "verdict": {"driver_agrees": True},
         "geometry": {"digest": "f" * 64, "facts": {"cells": CELLS}}}))
    record, text = build(pull, tmp_path, "--lift-digests", str(lifts))
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert pml["digest"]["status"] == "DIFFERENT"
    assert pml["size_strikes"] and "did not time one simulation" in pml["size_strikes"][0]
    assert not any(r["enters_per_ratio"] or r["enters_per_row"]
                   or r["enters_per_ratio_precedent_rule"] for r in pml["ratios"])
    assert any("not the harness lift's" in strike
               for strike in pml["legs"]["meep@3"]["strikes"])
    assert "Every ratio at this size is struck" in text
    thin = next(s for s in record["sizes"] if s["case"] == "thin_pml_3d")
    assert not thin["size_strikes"]


def test_rows_from_different_code_or_with_other_monitors_strike_the_size(pull, tmp_path):
    def code(entry):
        if entry.get("ledger") in ("gpu", "meep"):
            entry["code_digest"] = "1" * 64 if entry.get("ledger") == "gpu" else "2" * 64
    _rewrite(pull, code)
    record, _text = build(pull, tmp_path)
    assert all(any("different code digests" in s for s in size["size_strikes"])
               for size in record["sizes"])
    assert not any(r["enters_per_ratio"] for s in record["sizes"] for r in s["ratios"])


def test_a_monitor_count_that_differs_strikes_the_size(pull, tmp_path):
    def monitors(entry):
        if entry.get("ledger") == "gpu":
            entry["measured"]["monitors_mode"] = "attached"
            entry["measured"]["monitors_attached"] = {"_dft_monitors": 0,
                                                      "_flux_monitors": 2}
        if entry.get("ledger") == "meep":
            entry["measured"]["monitors_attached"] = {"dft_objects": 1}
    _rewrite(pull, monitors)
    record, _text = build(pull, tmp_path)
    assert all(any("monitor(s)" in s for s in size["size_strikes"])
               for size in record["sizes"])


def test_metal_rows_without_a_device_reading_are_not_counted_as_clean(pull, tmp_path):
    record, text = build(pull, tmp_path)
    d = record["denominators"]
    assert d["gpu_device_read"] == 0 and d["gpu_foreign_on_device_before_or_after"] == 0
    assert f"{d['gpu_rows']} of {d['gpu_rows']} rows carry no device reading" in text
