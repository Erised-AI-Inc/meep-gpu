"""``build_3d_comparison.py`` forms a ratio only between timings of one simulation.

Every refusal reason is produced here by name, on rows written in the shapes the two
benches write: identical-case MEEP rows (``case``, ``geometry_digest``,
``configuration``), and rows shaped as the 2026-09-22 MEEP rows were (``case``
``3d_pml``, ``cells``, ``dimensions``, ``mcell_steps_per_s``, ``ms_per_step``,
``ranks``, ``resolution``, ``row``, ``seconds``, ``steps_*``, and no digest).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_3d_comparison as cmp  # noqa: E402

DIGEST = "1" * 64
OTHER = "2" * 64


def write_route(root, case="pml_3d", res=20, cells_axis=80, ms=0.5, verdict="TIMED",
                monitors="attached", flux=1, timing_case=None):
    directory = root / f"{case}_res{res}"
    directory.mkdir(parents=True, exist_ok=True)
    row = {"case": case, "grid_shape": [cells_axis] * 3, "verdict": verdict,
           "per_leg": {"fused": {"median_seconds_per_step": ms * 1e-3},
                       "array": {"median_seconds_per_step": 4e-3}},
           "floors": {}, "substitution": {"tables_dispatched": ["triton"]},
           "monitors_mode": monitors,
           "monitors_attached": {"fused": ({"_dft_monitors": 0, "_flux_monitors": flux}
                                           if monitors == "attached" else {})},
           "bit_identity": {"fused_vs_array_identical": True}}
    if timing_case:
        row["timing_case"] = timing_case
    (directory / "rows.jsonl").write_text(json.dumps(row) + "\n")


def write_meep(root, ranks, rows):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"rows_r{ranks}.jsonl", "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def meep_row(rate, ranks=32, digest=DIGEST, case="pml_3d", cells=512_000, **extra):
    row = {"row": "measurement", "case": case, "cells": cells, "dimensions": 3,
           "ranks": ranks, "mcell_steps_per_s": rate, "geometry_digest": digest,
           "stepper": "fields_step", "spread": 0.01, "monitor": {"stepped": True},
           "monitors_attached": {"dft_objects": 1}}
    row.update(extra)
    return row


def shaped_2026_09_22(rate, ranks=32, cells=512_000):
    return {"row": "measurement", "case": "3d_pml", "dimensions": 3, "resolution": 80,
            "cells": cells, "steps_requested": 20, "steps_measured": 20,
            "seconds": 0.05, "ms_per_step": 2.5, "mcell_steps_per_s": rate, "ranks": ranks}


def write_digest(root, case="pml_3d", res=20, cells=512_000, digest=DIGEST, agrees=True,
                 builder=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"harness_lift_digest_{case}_res{res}.json").write_text(json.dumps(
        {"case": case, "resolution": res, "verdict": {"driver_agrees": agrees},
         "builder": {"builder_source_sha256": builder},
         "geometry": {"digest": digest, "facts": {"cells": cells}}}))


def run(tmp_path, meep_dir, digests_dir):
    out = tmp_path / "table.md"
    record_path = tmp_path / "table.json"
    status = cmp.main(["--host", "test", "--routes", f"triton={tmp_path / 'gpu'}",
                       "--meep", str(meep_dir), "--digests", str(digests_dir),
                       "--out", str(out), "--json", str(record_path)])
    return status, json.loads(record_path.read_text()), out.read_text()


def test_a_matched_pair_forms_a_ratio(tmp_path):
    write_route(tmp_path / "gpu")
    write_meep(tmp_path / "meep", 32, [meep_row(500.0), meep_row(520.0), meep_row(510.0)])
    write_meep(tmp_path / "meep", 48, [meep_row(400.0, ranks=48)])
    write_digest(tmp_path / "digests")
    status, record, text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status == 0
    entry = record["sections"]["pml_3d"][0]
    assert entry["meep_best_ranks"] == 32 and entry["meep"]["32"] == 510.0
    assert entry["routes"]["triton"]["vs_best_meep"] == pytest.approx(1024.0 / 510.0)
    assert record["summary"] == "paired 1 of 1; refused 0 of 1"
    assert "**2.01×**" in text and "ONE GPU PASS PER ROUTE DIRECTORY" in text


@pytest.mark.parametrize("name, build, reason", [
    ("label", lambda m: write_meep(m, 32, [meep_row(500.0, case="3d_pml")]),
     "case-label-mismatch"),
    ("missing", lambda m: write_meep(m, 32, [meep_row(500.0, digest=None)]),
     "meep-digest-missing"),
    ("disagree", lambda m: (write_meep(m, 32, [meep_row(500.0)]),
                            write_meep(m, 48, [meep_row(400.0, ranks=48, digest=OTHER)])),
     "meep-digests-disagree-across-passes"),
    ("mismatch", lambda m: write_meep(m, 32, [meep_row(500.0, digest=OTHER)]),
     "digest-mismatch"),
    ("mixed", lambda m: write_meep(m, 32, [
        meep_row(500.0), meep_row(560.0, configuration={"build": "native"})]),
     "meep-configurations-mixed"),
])
def test_each_refusal_is_produced_by_name(tmp_path, name, build, reason):
    write_route(tmp_path / "gpu")
    build(tmp_path / "meep")
    write_digest(tmp_path / "digests")
    status, record, text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status != 0
    assert record["refused"] == 1 and record["paired"] == 0
    assert reason in record["refused_pairs"][0]["reasons"], record["refused_pairs"]
    assert f"refused: " in text and reason in text
    if reason == "digest-mismatch":
        assert f"GPU {DIGEST[:16]} / MEEP {OTHER[:16]}" in text


def test_a_lift_whose_driver_disagreed_is_refused(tmp_path):
    write_route(tmp_path / "gpu")
    write_meep(tmp_path / "meep", 32, [meep_row(500.0)])
    write_digest(tmp_path / "digests", agrees=False)
    status, record, _text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status != 0 and record["refused_pairs"][0]["reasons"] == ["lift-driver-disagrees"]


@pytest.mark.parametrize("route", [dict(monitors="detached"), dict(flux=2)])
def test_a_gpu_row_with_other_monitors_is_refused(tmp_path, route):
    write_route(tmp_path / "gpu", **route)
    write_meep(tmp_path / "meep", 32, [meep_row(500.0)])
    write_digest(tmp_path / "digests")
    status, record, _text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status != 0 and record["refused_pairs"][0]["reasons"] == ["gpu-monitors-detached"]


def test_a_new_case_built_by_another_builder_is_refused(tmp_path):
    write_route(tmp_path / "gpu", case="thin_pml_3d", res=10,
                timing_case={"name": "thin_pml_3d", "builder_source_sha256": "a" * 64})
    write_meep(tmp_path / "meep", 32, [meep_row(500.0, case="thin_pml_3d")])
    write_digest(tmp_path / "digests", case="thin_pml_3d", res=10, builder="b" * 64)
    status, record, _text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status != 0 and record["refused_pairs"][0]["reasons"] == ["builder-mismatch"]
    write_digest(tmp_path / "digests", case="thin_pml_3d", res=10, builder="a" * 64)
    status, record, _text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status == 0, record["refused_pairs"]


def test_failed_meep_rows_are_dropped_and_a_failing_best_refuses(tmp_path):
    write_route(tmp_path / "gpu")
    write_meep(tmp_path / "meep", 32, [meep_row(500.0), meep_row(505.0),
                                       meep_row(900.0, struck="monitor not stepped"),
                                       meep_row(950.0, monitor={"stepped": False})])
    write_meep(tmp_path / "meep", 48, [meep_row(400.0, ranks=48, spread=0.07)])
    write_digest(tmp_path / "digests")
    status, record, _text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status == 0
    entry = record["sections"]["pml_3d"][0]
    assert entry["meep"]["32"] == pytest.approx(502.5) and entry["meep_passes"]["32"] == 2
    noisy = tmp_path / "noisy"
    write_meep(noisy, 32, [meep_row(500.0, spread=0.08)])
    write_meep(noisy, 48, [meep_row(400.0, ranks=48)])
    status, record, text = run(tmp_path, noisy, tmp_path / "digests")
    assert status != 0 and record["refused_pairs"][0]["reasons"] == ["meep-best-fails-gate"]
    assert record["refused_pairs"][0]["meep_best_gate"] == ["within-window spread 8.0% in "
                                                           "pass 1"]


def test_a_gpu_row_without_a_lift_digest_is_refused(tmp_path):
    write_route(tmp_path / "gpu")
    write_meep(tmp_path / "meep", 32, [meep_row(500.0)])
    (tmp_path / "digests").mkdir()
    status, record, _text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status != 0 and record["refused_pairs"][0]["reasons"] == ["gpu-digest-missing"]


def test_rows_shaped_as_the_2026_09_22_rows_are_refused_on_both_counts(tmp_path):
    write_route(tmp_path / "gpu")
    write_meep(tmp_path / "meep", 32, [shaped_2026_09_22(560.0), shaped_2026_09_22(570.0)])
    write_meep(tmp_path / "meep", 16, [shaped_2026_09_22(450.0, ranks=16)])
    write_digest(tmp_path / "digests")
    status, record, text = run(tmp_path, tmp_path / "meep", tmp_path / "digests")
    assert status != 0
    reasons = record["refused_pairs"][0]["reasons"]
    assert "case-label-mismatch" in reasons and "meep-digest-missing" in reasons
    assert "paired 0 of 1; refused 1 of 1" in text
    # Both values are still shown: the MEEP rows in their own section, the GPU row in its.
    assert set(record["sections"]) == {"3d_pml", "pml_3d"}
    assert record["sections"]["3d_pml"][0]["meep"]["32"] == 565.0


def test_two_conflicting_lift_files_are_refused(tmp_path):
    write_digest(tmp_path / "d" / "a")
    write_digest(tmp_path / "d" / "b", digest=OTHER)
    with pytest.raises(SystemExit, match="disagree"):
        cmp.read_digests(str(tmp_path / "d"))


def test_the_2026_09_28_file_name_is_read_by_its_content(tmp_path):
    (tmp_path / "harness_lift_digest_res20.json").write_text(json.dumps(
        {"case": "pml_3d", "resolution": 20,
         "geometry": {"digest": DIGEST, "facts": {"cells": 512_000}}}))
    assert cmp.read_digests(str(tmp_path))[("pml_3d", 512_000)]["digest"] == DIGEST
