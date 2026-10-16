# SPDX-License-Identifier: GPL-2.0-or-later
"""Table 2 as MEEP identity: the Tier B archive tool, the table builder, the context L2.

Host-only: no MEEP, no capsule replay. The records are synthetic, in the runner's own
schema (``meep_identity_tier_c.py``'s ``result.json`` and ``digests.json``), so each
judgement, count and rendering rule is pinned without a reference MEEP.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pytest

from parity.meep_gpu import build_meep_identity_table as table
from parity.meep_gpu import meep_identity_context_l2 as context
from parity.meep_gpu import meep_identity_smoke as smoke
from parity.meep_gpu import meep_identity_tier_a as tier_a
from parity.meep_gpu import record_meep_identity_archive as archive

COMPONENTS = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz")


def _result(mode="keep", words=100, differing=0, reader_differing=0, steps=(0, 1, 2), **extra):
    result = {
        "case": "row", "mode": mode, "n_steps": steps[-1], "steps": list(steps),
        "stored": {"words": words, "word_differing": differing, "value_differing": differing,
                   "sign_of_zero_only": 0},
        "reader": {"words": 2 * words, "word_differing": reader_differing, "value_differing": 0},
        "ghost": {"words": 4, "word_differing": 1, "sign_of_zero_only": 1},
        "reads": {key: 0 for key in ("inexact", "imaginary", "point_read_differing",
                                     "not_stored_compared", "held_copies_differing",
                                     "held_aliased_unexplained", "held_aliased_complex",
                                     "held_unconnected_unexplained")},
        "subnormals": {"mode": mode}, "fpu_checks": len(steps),
        "planted_flip": {"caught": True, "words_differing": 1, "digest_caught": True,
                         "planes_differing": [3], "plane": 3},
        "smoke_record": {"equal": 0, "of": 0, "lattice_equal": None},
        "first_difference": ({"step": 2, "component": "Ez"} if differing else None),
        "meep": {"images": {"libmeep": "aa"}}, "chunk_division": {"chunks": 1},
    }
    result.update(extra)
    return result


def test_a_row_that_meets_every_tier_c_claim_holds():
    assert archive.judge_row("keep", _result()) == ("holds", [])


def test_differing_words_make_a_row_differ_and_name_the_counts():
    status, problems = archive.judge_row("keep", _result(differing=5))
    assert status == "differs"
    assert any("5 value-differing and 5 word-differing" in p for p in problems)
    assert archive.judge_row("keep", _result(reader_differing=2))[0] == "differs"


@pytest.mark.parametrize("change", [
    {"fpu_checks": 1},
    {"subnormals": {"mode": "flush"}},
    {"planted_flip": {"caught": False}},
    {"reads": {"inexact": 1}},
])
def test_a_failed_claim_other_than_the_words_is_an_error(change):
    status, problems = archive.judge_row("keep", _result(**change))
    assert status == "error" and problems


def test_a_refused_case_keeps_its_reason():
    assert archive.judge_row("keep", {"refused": "MeepSimulationNotLiftable: no"}) == (
        "refused", ["MeepSimulationNotLiftable: no"])


def test_exclusions_need_a_reason():
    assert archive.parse_exclusions(["ex.a=random source"]) == {"ex.a": "random source"}
    with pytest.raises(ValueError):
        archive.parse_exclusions(["ex.a"])


def _write_row(out: Path, mode: str, label: str, result, digests=None, capsule=True):
    row = out / mode / label
    row.mkdir(parents=True)
    (row / "result.json").write_text(json.dumps(dict(result, case=label)))
    (row / "digests.json").write_text(json.dumps(digests or _digests(result["steps"])))
    (row / "planes.npy").write_bytes(b"planes")
    if capsule:
        (row / "capsule.capsule").write_bytes(b"capsule")
        (row / "currents.json").write_text("{}")
    (out / "meep_identity.json").write_text("{}")


def _digests(steps, change_at=None):
    stored = {str(s): {c: f"s{s}{c}" for c in COMPONENTS} for s in steps}
    reader = {str(s): {c: f"r{s}{c}" for c in COMPONENTS} for s in steps}
    if change_at is not None:
        stored[str(change_at)]["Ez"] = "other"
    return {"steps": list(steps), "components": list(COMPONENTS), "stored": stored, "reader": reader}


def _source(tmp_path: Path, name: str, mode: str, rows) -> Path:
    root = tmp_path / name
    for label, result, digests in rows:
        out = root / "rows" / label
        if result is None:
            (out / mode).mkdir(parents=True)
            (out / mode / f"{label}.error.json").write_text(json.dumps({"error": "gdspy is absent"}))
        else:
            _write_row(out, mode, label, result, digests)
    return root / "rows"


def test_the_archive_tool_judges_copies_counts_and_reads_back(tmp_path):
    source = _source(tmp_path, "mac", "keep", [
        ("ex.a", _result(), None), ("ex.b", _result(differing=3), None),
        ("ex.c", None, None), ("ex.d", _result(differing=7), None)])
    out = tmp_path / "tier_b"
    assert archive.main(["--archive", str(out), "--set", "mac_keep", "--mode", "keep",
                         "--source", str(source), "--exclude", "ex.d=random source"]) == 0
    record = json.loads((out / archive.ARCHIVE_FILE).read_text())
    leaf = record["sets"]["mac_keep"]
    assert leaf["counts"] == {"holds": 1, "differs": 1, "error": 0, "refused": 1, "excluded": 1,
                              "rows": 4, "stored_words_held": 100, "reader_words_held": 200}
    assert leaf["rows"]["ex.d"]["excluded"] == "random source"
    assert leaf["rows"]["ex.c"]["status"] == "refused"
    assert set(leaf["rows"]["ex.a"]["files"]) == {"result", "digests", "planes", "capsule", "currents", "ledger"}
    assert archive.check_set(out, "mac_keep") == 6 + 6 + 1 + 6
    (out / leaf["rows"]["ex.a"]["files"]["planes"]["file"]).write_bytes(b"moved")
    with pytest.raises(ValueError, match="does not read back"):
        archive.check_set(out, "mac_keep")


def _archive_two_hosts(tmp_path):
    steps = (0, 1, 2)
    mac = _source(tmp_path, "mac", "keep", [("ex.a", _result(), _digests(steps)),
                                            ("t.b", _result(), _digests(steps)),
                                            ("t.c", _result(differing=2), _digests(steps))])
    lin = _source(tmp_path, "lin", "keep", [("ex.a", _result(), _digests(steps)),
                                            ("t.b", _result(), _digests(steps, change_at=1)),
                                            ("t.c", None, None)])
    out = tmp_path / "tier_b"
    archive.main(["--archive", str(out), "--set", "mac_keep", "--mode", "keep", "--source", str(mac)])
    archive.main(["--archive", str(out), "--set", "linux_keep", "--mode", "keep", "--source", str(lin)])
    manifest = {"rows": [
        {"label": "ex.a", "kind": "example", "facts": {}, "table2": {"grid_cells": 10, "driver_steps": 2, "rel_l2": 1e-6}},
        {"label": "t.b", "kind": "test", "facts": {"source_kinds": ["EigenModeSource"]},
         "table2": {"grid_cells": 20, "driver_steps": 2, "rel_l2": 2e-6}},
        {"label": "t.c", "kind": "test", "facts": {}, "table2": {"grid_cells": 30, "driver_steps": 2, "rel_l2": None}}]}
    return out, manifest


def test_the_table_counts_rows_per_reference_and_classes_l0(tmp_path):
    out, manifest = _archive_two_hosts(tmp_path)
    built = table.build(out, manifest, ["mac_keep", "linux_keep"], {"ex.a": {"parity_rel_l2": 3.0e-7}})
    mac, lin = built["summary"]["mac_keep"], built["summary"]["linux_keep"]
    assert (mac["example"], mac["test"]) == ({"holds": 1, "of": 1}, {"holds": 1, "of": 2})
    assert [label for label, _ in mac["differs"]] == ["t.c"]
    assert [label for label, _ in lin["refused"]] == ["t.c"]
    l0 = built["l0"]["mac_keep|linux_keep"]
    assert l0["equal"] == 1 and l0["not comparable"] == 1
    assert l0["differ"] == [("t.b", {"step": 1, "component": "Ez", "kind": "stored"},
                             "eigenmode source (each MEEP's own MPB solve)")]
    text = table.render(built, "Table 2", "provenance")
    assert "| mac_keep | 1 of 1 | 1 of 2 | 1 | 0 | 0 | 0 | 200 | 400 |" in text
    assert "0 of 100 / 0 of 200" in text and "3.000e-07" in text and "1.000e-06" in text
    assert "- **t.c** (linux_keep): gdspy is absent" in text and "not run (reason above)" in text


def test_a_set_the_archive_lacks_is_refused(tmp_path):
    out, manifest = _archive_two_hosts(tmp_path)
    with pytest.raises(ValueError, match="no set"):
        table.build(out, manifest, ["mac_flush"], {})


def test_cells_shorten_only_when_w_and_v_match_the_first_reference():
    first = archive.row_record("keep", "x", _result(), None)
    other = archive.row_record("keep", "x", _result(differing=1), None)
    assert table.cell_text(first) == "0 of 100 / 0 of 200"
    assert table.cell_text(other, first) == "1 / 0 @2:Ez"
    bigger = archive.row_record("keep", "x", _result(words=101), None)
    assert table.cell_text(bigger, first) == "0 of 101 / 0 of 202"


def test_the_context_reads_count_words_by_bit_pattern():
    a = numpy.array([0.0, 1.0], dtype=numpy.float32)
    b = numpy.array([-0.0, 1.0], dtype=numpy.float32)
    c = numpy.array([1 + 1j], dtype=numpy.complex64)
    assert context.summarize_reads({"Ex": (a, b), "Ey": (c, c)}) == {"words": 4, "word_differing": 1}
    assert context.summarize_reads({"Ex": (a, a[:1])}) == {"words": 1, "word_differing": 1}


def test_tier_a_cases_are_the_probe_cases_then_the_family_rows():
    assert tier_a.CASES[:4] == tuple(smoke.CASES)
    assert tier_a.CASES[4:] == tuple(tier_a.FAMILY_ROWS)
    assert len(set(tier_a.CASES)) == len(tier_a.CASES)


def test_link_hard_links_the_row_files_and_the_record_still_reads_back(tmp_path):
    source = _source(tmp_path, "mac", "keep", [("ex.a", _result(), None)])
    out = tmp_path / "tier_b"
    assert archive.main(["--archive", str(out), "--set", "mac_keep", "--mode", "keep",
                         "--source", str(source), "--link"]) == 0
    spec = json.loads((out / archive.ARCHIVE_FILE).read_text())["sets"]["mac_keep"]["rows"]["ex.a"]["files"]
    original = source / "ex.a" / "keep" / "ex.a" / "digests.json"
    assert (out / spec["digests"]["file"]).stat().st_ino == original.stat().st_ino
    assert archive.check_set(out, "mac_keep") == 6


def test_the_flip_column_judges_only_the_references_where_a_row_holds(tmp_path):
    out, manifest = _archive_two_hosts(tmp_path)
    text = table.render(table.build(out, manifest, ["mac_keep", "linux_keep"], {}), "Table 2", "")
    row_c = next(line for line in text.splitlines() if line.startswith("| ") and " t.c " in line)
    assert row_c.split(" | ")[-3] == "-"
