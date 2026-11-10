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
    assert any("ran with the ladder's ALLOW_UNCERTIFIED=1" in strike
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


def test_the_code_digest_count_reads_one_set_of_rows_on_a_resumed_ladder(pull, tmp_path):
    """The 1.0 timing run read "86 of 79 GPU and MEEP ledger rows carry a code digest" on
    the Mac ladder. The numerator counted the lift and digest entries too; a resume
    records a preflight digest again. Numerator and denominator now count the same rows,
    once."""
    def code(entry):
        if entry.get("ledger") in ("gpu", "meep"):
            entry["code_digest"] = "c" * 64
    _rewrite(pull, code)
    ledger = pull / "ladder_rows.jsonl"
    entries = [json.loads(line) for line in open(ledger)]
    rows = sum(e.get("ledger") in ("gpu", "meep") for e in entries)
    extra = [{"ledger": "lift", "tier": "digests", "index": 90 + i, "case": "pml_3d",
              "res": RES["pml_3d"], "code_digest": "c" * 64} for i in range(2)]
    digest = {"ledger": "digest", "tier": "preflight", "index": 89, "code_digest": "c" * 64}
    # the resumed run records the preflight digest and the last MEEP row a second time
    again = dict(next(e for e in reversed(entries) if e.get("ledger") == "meep"))
    ledger.write_text("".join(json.dumps(e) + "\n"
                              for e in entries + extra + [digest, digest, again]))
    record, text = build(pull, tmp_path)
    d = record["denominators"]
    assert d["ledger_gpu_and_meep_rows"] == rows
    assert d["ledger_rows_carrying_code_digest"] == rows
    assert d["ledger_other_entries_carrying_code_digest"] == 4
    assert f"{rows} of {rows} GPU and MEEP ledger rows carry a code digest" in text


def test_a_build_named_by_its_record_alone_reports_the_records_flags(pull, tmp_path):
    """In the 1.0 timing run the Mac build was named by its record and no config.log, so
    the probe's configure line read None and its flags {} beside a record that carried
    both."""
    environment = json.loads((pull / "environment.json").read_text())
    given = {"configure_line": "./configure --enable-single --enable-portable-binary",
             "makefile_flags": {"CXXFLAGS": "-g -O2"}, "openmp": False}
    environment["meep_builds"] = {
        "reference": {"configure_line": None, "makefile_flags": {}, "config_log": None,
                      "openmp": False, "build_record_file": given},
        "native": {"configure_line": None, "makefile_flags": {},
                   "build_record_file": {"makefile_flags": {"CXXFLAGS": "-O3 -march=native"}}}}
    (pull / "environment.json").write_text(json.dumps(environment))
    record, text = build(pull, tmp_path)
    env = record["environment"]
    assert env["meep_configure_line"] == given["configure_line"]
    assert env["meep_cxxflags"] == "-g -O2"
    assert env["builds"]["native"]["cxxflags"] == "-O3 -march=native"
    assert f"configured `{given['configure_line']}`, CXXFLAGS `-g -O2`" in text
    # a probed value still wins over the record it was given
    environment["meep_builds"]["reference"]["makefile_flags"] = {"CXXFLAGS": "-O2"}
    (pull / "environment.json").write_text(json.dumps(environment))
    record, _text = build(pull, tmp_path)
    assert record["environment"]["meep_cxxflags"] == "-O2"


def test_gpu_only_rows_are_identical_only_by_their_identity_row(pull, tmp_path):
    """A GPU-only row has no array leg: the table reads its identity from the identity
    row the ledger cited, and strikes the pass when there was none."""
    def solo(entry):
        if entry.get("ledger") == "gpu":
            entry["measured"]["legs_mode"] = "gpu-only"
            entry["measured"]["bit_identical_fused_vs_array"] = None
            entry["bit_identity_from"] = {"ok": True, "key": "identity_gpu_metal_res10_p0"}
    _rewrite(pull, solo)
    record, text = build(pull, tmp_path)
    d = record["denominators"]
    assert d["gpu_rows_gpu_only"] == d["gpu_rows"] == d["gpu_identity_from_identity_row"]
    assert d["gpu_bit_identical"] == d["gpu_rows"]
    assert f"({d['gpu_rows']} of the {d['gpu_rows']} GPU-only rows by their identity row)" in text
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert not any("not bit-identical" in s for s in pml["legs"]["metal"]["strikes"])

    def uncited(entry):
        if entry.get("ledger") == "gpu":
            entry.pop("bit_identity_from", None)
    _rewrite(pull, uncited)
    record, _text = build(pull, tmp_path)
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert any("not bit-identical" in s for s in pml["legs"]["metal"]["strikes"])
    assert record["denominators"]["gpu_bit_identical"] == 0


def test_the_pinned_device_join_reads_a_row_whose_tables_left_the_machine():
    """The box tables are written with card ids and an ``own``/``other`` owner since they
    leave the machine (``timing_host.recorded_compute_apps``); the join reads the same
    neighbour off them as off an older row's raw UUIDs and pid."""
    import timing_host  # noqa: PLC0415

    uuid_a = "GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37"
    uuid_b = "GPU-9b2e7c41-0d6a-4f83-b5e9-1c7a3d8f6e02"
    gpus = f"0, {uuid_a}, 3 MiB, 0 %\n1, {uuid_b}, 4970 MiB, 0 %"
    apps = f"{uuid_a}, 4242, 1200 MiB\n{uuid_a}, 271828, 800 MiB\n{uuid_b}, 900, 4970 MiB"
    raw = {"CUDA_VISIBLE_DEVICES": "0", "pid": 4242, "gpus": gpus, "compute_apps": apps}
    new = {"CUDA_VISIBLE_DEVICES": "0", "gpus": timing_host.recorded_gpus(gpus),
           "compute_apps": timing_host.recorded_compute_apps(apps, 4242)}
    assert len(table.pinned_foreign(raw)) == len(table.pinned_foreign(new)) == 1
    assert "271828" not in json.dumps(new) and uuid_a not in json.dumps(new)
    by_card = dict(new, CUDA_VISIBLE_DEVICES=timing_host.recorded_device(uuid_a))
    assert len(table.pinned_foreign(by_card)) == 1


def test_the_pinned_gpu_is_found_in_an_environment_record_written_by_card_id():
    """``--gpu auto`` pins by UUID; the environment record writes the pin and the
    listing's UUID column as the same card id, so the pinned card's model is found."""
    import timing_host  # noqa: PLC0415

    uuid_a = "GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37"
    uuid_b = "GPU-9b2e7c41-0d6a-4f83-b5e9-1c7a3d8f6e02"
    listing = (f"0, {uuid_a}, NVIDIA RTX A6000, 535.129.03, 49140 MiB, 3 MiB, 0 %, 8.6\n"
               f"1, {uuid_b}, NVIDIA H100 PCIe, 535.129.03, 81559 MiB, 3 MiB, 0 %, 9.0")
    environment = {"gpu": timing_host.recorded_device(uuid_b),
                   "nvidia_smi_gpus": timing_host.recorded_gpus(listing)}
    facts = table.environment_facts(environment)
    assert facts["gpu_name"] == "NVIDIA H100 PCIe" and facts["compute_capability"] == "9.0"
    assert uuid_b not in json.dumps(facts)


# ---------------------------------------------------------------------------
# What a table names, and a split ladder unit read from its two halves
# ---------------------------------------------------------------------------

def test_the_table_names_no_local_path(pull, tmp_path):
    """The table leaves the machine: the pull, a manifest, the comparison builder and a lift
    file are named by their own names, never by an absolute path."""
    import timing_rows as TR  # noqa: PLC0415

    lifts = tmp_path / "lifts"
    lifts.mkdir()
    (lifts / "harness_lift_digest_pml_3d_res10.json").write_text(json.dumps(
        {"case": "pml_3d", "resolution": 10, "verdict": {"driver_agrees": False},
         "geometry": {"digest": DIGEST["pml_3d"], "facts": {"cells": CELLS}}}))
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text("")
    record, text = build(pull, tmp_path, "--lift-digests", str(lifts), "--manifest",
                         str(manifest))
    assert TR.local_paths(record) == [] and TR.local_paths(text) == []
    assert record["pull"] == "pull" and record["pull_verification"]["manifest"] == "SHA256SUMS"
    assert record["comparison_reader_cross_check"]["builder"] == \
        "parity/meep_gpu/build_3d_comparison.py"
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert any("(harness_lift_digest_pml_3d_res10.json)" in s for s in pml["size_strikes"])


def split_pulls(pull, tmp_path, names=("ladder_s1_u01_gpu", "ladder_s1_u01_meep")):
    """The ladder split into the two halves of a split unit: the GPU half keeps its routes
    and the array control, the MEEP half its MEEP rows and digests."""
    import shutil  # noqa: PLC0415

    halves = {}
    # What the other half's ladder alone writes is not in this half's directory.
    elsewhere = {"gpu": ("meep", "meep_splitcost", "digests"), "meep": ("gpu_metal",)}
    for side, name in zip(("gpu", "meep"), names):
        root = tmp_path / name
        shutil.copytree(pull, root)
        for directory in elsewhere[side]:
            shutil.rmtree(root / directory)
        kinds = table.SPLIT_KINDS[side]
        entries = [json.loads(line) for line in open(root / "ladder_rows.jsonl")]
        (root / "ladder_rows.jsonl").write_text("".join(
            json.dumps(e) + "\n" for e in entries if e.get("ledger") in kinds))
        halves[side] = root
    return halves


def split_row(tmp_path, halves=None):
    import test_timing_rows as rows  # noqa: PLC0415
    import timing_rows as TR  # noqa: PLC0415

    record, = rows.split_record(tmp_path / "rec", None if halves is None
                                else rows._halves_at(*halves))
    return TR.write_record(str(tmp_path / "store"), record)


def test_a_split_unit_reads_as_the_whole_ladder_with_every_ratio_labelled_split(pull, tmp_path):
    """The GPU half's legs and the MEEP half's, with the unit's row record: the ratios a
    whole ladder gives, each labelled split, and the masks and order beside them."""
    (tmp_path / "whole").mkdir()
    whole, _text = build(pull, tmp_path / "whole")
    halves = split_pulls(pull, tmp_path)
    record, text = build(halves["gpu"], tmp_path, "--meep-pull", str(halves["meep"]),
                         "--split-row", split_row(tmp_path))
    assert record["pull"] == "ladder_s1_u01_gpu" and record["meep_pull"] == "ladder_s1_u01_meep"
    assert record["split"]["masks"] == {"gpu": "0x999988", "meep": "0xffffff"}
    assert record["split"]["order"] == ["gpu", "meep"]

    def ratios(of):
        return {(s["case"], r["gpu"], r["denominator"]): r["value"]
                for s in of["sizes"] for r in s["ratios"]}
    assert ratios(record) == pytest.approx(ratios(whole)) and ratios(record)
    assert all(r["split"] is True for s in record["sizes"] for r in s["ratios"])
    assert not any("split" in r for s in whole["sizes"] for r in s["ratios"])
    assert [s["digest"] for s in record["sizes"]] == [s["digest"] for s in whole["sizes"]]
    for name in ("meep_rows", "gpu_rows", "array_controls", "ledger_entries",
                 "ledger_by_tier_and_kind"):
        assert record["denominators"][name] == whole["denominators"][name], name
    assert record["denominators"]["rows_planned"] == 48, "each half's own plan, both counted"
    assert record["comparison_reader_cross_check"] == whole["comparison_reader_cross_check"]
    assert "× (provisional) (split) |" in text and "× (split) against MEEP@" in text
    assert record["split"]["label"] in text
    assert ("from the two halves of one split ladder unit, `ladder_s1_u01_gpu` (its GPU legs) "
            "and `ladder_s1_u01_meep` (its MEEP legs)") in text


@pytest.mark.parametrize("change, named", [
    ("masks", "split_row_refused"),
    ("name", "--pull not under its half's ladder"),
    ("rows", "the gpu half's ladder ladder_s1_u01_gpu holds meep rows"),
    ("row", "a split ladder unit is read from its two halves and its row record"),
])
def test_a_split_unit_is_refused_by_name_unless_its_halves_hold_the_sessions_masks(
        pull, tmp_path, change, named):
    """Halves that did not run under the session's masks, a pull that is not the half the
    record names, a half that holds the other half's rows, and a MEEP half given without
    the unit's record: each refused by name, and no table written."""
    names = ("ladder_s2_u01_gpu", "ladder_s1_u01_meep") if change == "name" else (
        "ladder_s1_u01_gpu", "ladder_s1_u01_meep")
    halves = split_pulls(pull, tmp_path, names)
    if change == "rows":
        import shutil  # noqa: PLC0415
        shutil.copy(pull / "ladder_rows.jsonl", halves["gpu"] / "ladder_rows.jsonl")
    extra = ["--meep-pull", str(halves["meep"])]
    if change != "row":
        extra += ["--split-row", split_row(tmp_path, ("0xffffff", "0x999988")
                                           if change == "masks" else None)]
    with pytest.raises(SystemExit) as refused:
        build(halves["gpu"], tmp_path, *extra)
    assert named in str(refused.value) and str(refused.value).startswith("REFUSING")
    assert not (tmp_path / "t.json").exists()


def test_a_split_units_meep_side_is_read_from_its_meep_half(pull, tmp_path):
    """The digests of record, the harness lift, the preflight's lift digest, the binding
    decision, the lift's code digest and the MEEP builds are the MEEP half's; the host and
    the GPU are the GPU half's."""
    halves = split_pulls(pull, tmp_path)
    meep = halves["meep"]
    for case, res in RES.items():
        (meep / "digests_lift").mkdir(exist_ok=True)
        (meep / "digests_lift" / f"harness_lift_digest_{case}_res{res}.json").write_text(
            json.dumps({"case": case, "resolution": res, "verdict": {"driver_agrees": True},
                        "geometry": {"digest": DIGEST[case], "facts": {"cells": CELLS}}}))
    (meep / "preflight" / "digest").mkdir(parents=True)
    (meep / "preflight" / "digest" / "harness_lift_digest_pml_3d_res10.json").write_text(
        json.dumps({"case": "pml_3d", "agreement": "agrees", "compared_with": [
            {"cells": CELLS, "digest_equal": True, "epsilon_exact_equal": True}]}))
    (meep / "binding_decision.json").write_text(json.dumps({"chosen": "bound"}))
    with open(meep / "ladder_rows.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ledger": "lift", "tier": "digests", "case": "pml_3d",
                                 "res": RES["pml_3d"], "code_digest": "f" * 64}) + "\n")
    environment = json.loads((meep / "environment.json").read_text())
    environment["meep_builds"] = {"reference": {"configure_line": "./configure --enable-single",
                                                "makefile_flags": {"CXXFLAGS": "-O2"}}}
    (meep / "environment.json").write_text(json.dumps(environment))
    record, _text = build(halves["gpu"], tmp_path, "--meep-pull", str(meep),
                          "--split-row", split_row(tmp_path))
    assert {s["digest"]["status"] for s in record["sizes"]} == {"equal"}
    assert record["preflight_lift_digest"]["pml_3d"]["agreement"] == "agrees"
    assert record["binding_decision"] == {"chosen": "bound"}
    assert record["environment"]["meep_configure_line"] == "./configure --enable-single"
    assert record["environment"]["meep_cxxflags"] == "-O2"
    assert record["environment"]["platform"] == "darwin"
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert not any("code digests" in strike for strike in pml["size_strikes"])
    # The lift's code digest is read from the MEEP half: a GPU half that ran other code
    # strikes the size.
    _rewrite(halves["gpu"], lambda entry: entry.update(code_digest="e" * 64))
    record, _text = build(halves["gpu"], tmp_path, "--meep-pull", str(meep),
                          "--split-row", split_row(tmp_path / "again"))
    pml = next(s for s in record["sizes"] if s["case"] == "pml_3d")
    assert any("2 different code digests" in strike for strike in pml["size_strikes"])


def test_a_pull_written_before_its_ladder_wrote_neutral_records_names_no_path(pull, tmp_path):
    """A pull whose ladder wrote its environment record as the host spelled it (before the
    ladder wrote neutral records): every string the table copies from it, the configure
    line, the packages, the tree and a struck row's reason, is written under its root's
    label, in the JSON and in the markdown alike."""
    import timing_rows as TR  # noqa: PLC0415

    home = "/" + "ho" + "me" + "/gpuhostaccount"
    environment = json.loads((pull / "environment.json").read_text())
    environment.setdefault("meep_builds", {})["reference"] = {
        "configure_line": f"./configure --prefix={home}/miniforge3/envs/ref --enable-single",
        "makefile_flags": {"CXXFLAGS": f"-O2 -I{home}/miniforge3/envs/ref/include"}}
    environment["tree"] = dict(environment.get("tree") or {}, root=f"{home}/meep_gpu_target")
    (pull / "environment.json").write_text(json.dumps(environment))
    record, text = build(pull, tmp_path)
    assert TR.local_paths(record) == [] and TR.local_paths(text) == []
    assert "gpuhostaccount" not in json.dumps(record) and "gpuhostaccount" not in text
    assert record["environment"]["meep_configure_line"] == (
        "./configure --prefix=<home>/miniforge3/envs/ref --enable-single")


def test_each_half_of_a_split_unit_is_checked_against_its_own_listing(pull, tmp_path):
    """A split unit's table checks the GPU half against ``--manifest`` and the MEEP half
    against ``--meep-manifest``, and names each half's count; a half not checked says so."""
    import hashlib  # noqa: PLC0415

    halves = split_pulls(pull, tmp_path)

    def listing(root, name):
        lines = [f"{hashlib.sha256((root / p).read_bytes()).hexdigest()}  ./{p}"
                 for p in ("ladder_rows.jsonl", "environment.json")]
        (tmp_path / name).write_text("\n".join(lines) + "\n")
        return str(tmp_path / name)
    record, text = build(halves["gpu"], tmp_path, "--meep-pull", str(halves["meep"]),
                         "--split-row", split_row(tmp_path),
                         "--manifest", listing(halves["gpu"], "gpu.sha256"),
                         "--meep-manifest", listing(halves["meep"], "meep.sha256"))
    assert record["pull_verification"]["matching"] == 2
    assert record["meep_pull_verification"] == {
        "checked": True, "manifest": "meep.sha256", "files": 2, "matching": 2,
        "mismatched": []}
    assert ("* **Pull** of the GPU half `ladder_s1_u01_gpu`: 2 of 2 files match") in text
    assert ("* **Pull** of the MEEP half `ladder_s1_u01_meep`: 2 of 2 files match") in text
    (tmp_path / "again").mkdir()
    record, text = build(halves["gpu"], tmp_path / "again", "--meep-pull",
                         str(halves["meep"]), "--split-row", split_row(tmp_path / "again"))
    assert record["meep_pull_verification"] == {"checked": False}
    assert ("* **Pull** of the MEEP half `ladder_s1_u01_meep`: not verified against a "
            "manifest (pass `--meep-manifest`).") in text


def test_the_account_a_pull_records_is_never_written_under_a_home_of_two_levels(pull,
                                                                               tmp_path):
    """A home of two levels (``/home/<group>/<account>``) written on another machine keeps the
    account after ``<home>``; the account the pull's own ladder records (its ACCOUNT
    parameter) is written as ``<account>`` wherever it stands as a word. An account spelled
    like one of the record's words names nothing and rewrites nothing."""
    import timing_rows as TR  # noqa: PLC0415

    home = "/" + "ho" + "me" + "/grpdir/gpuacct"
    environment = json.loads((pull / "environment.json").read_text())
    environment.setdefault("params", {})["ACCOUNT"] = {"value": "gpuacct", "source": "flag"}
    environment.setdefault("meep_builds", {})["reference"] = {
        "configure_line": f"./configure --prefix={home}//miniforge3/envs/ref --enable-single"}
    environment["meep_builds"]["gpuacct"] = {"configure_line": "./configure"}  # a key
    (pull / "environment.json").write_text(json.dumps(environment))
    record, text = build(pull, tmp_path)
    assert "<account>" in record["environment"]["builds"]
    assert "gpuacct" not in json.dumps(record) and "gpuacct" not in text
    assert "grpdir" not in json.dumps(record) and "grpdir" not in text
    assert record["environment"]["meep_configure_line"] == (
        "./configure --prefix=<home>/<account>//miniforge3/envs/ref --enable-single")
    assert TR.local_paths(record) == [] and TR.local_paths(text) == []
    environment["params"]["ACCOUNT"] = {"value": "metal", "source": "flag"}
    (pull / "environment.json").write_text(json.dumps(environment))
    (tmp_path / "words").mkdir()
    record, _text = build(pull, tmp_path / "words")
    assert record["routes"]["metal"]["label"] == "GPU metal route (metal)"
