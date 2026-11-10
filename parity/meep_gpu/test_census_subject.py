"""The Metal and Triton fusion boards refuse a census cut against another tree.

``census_subject.subject_pin`` re-computes the census's subject manifest from the tree and
compares it with the stamp every measured row carries, as the CUDA board does. These tests
pin three things: the manifest is spelled as the census driver spells it (so a good census is
never refused), each way a stamp can be wrong is refused by name, and both boards call the
check before they read anything else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_fusion_matrix as metal_board  # noqa: E402
import build_triton_fusion_matrix as triton_board  # noqa: E402
import census_subject  # noqa: E402
import measure_predicate_coverage as census_driver  # noqa: E402


def _tree(root: Path, package: str, modules=("a", "b", "zeta")) -> Path:
    """A minimal tree: ``meep_gpu/<package>/<module>.py`` and the shared engine files."""
    (root / "meep_gpu" / package).mkdir(parents=True)
    for name in modules:
        (root / "meep_gpu" / package / f"{name}.py").write_text(f"X = '{name}'\n")
    for relative in census_subject.SUBJECT_EXTRA:
        (root / relative).write_text(f"# {relative}\n")
    battery = root / "battery.py"
    battery.write_text(f"SUBJECT_PACKAGE = '{package}'\n")
    return battery


def _rows(stamp, n=3):
    return [{"leg": "tests", "row": f"r{k}", "measured": True, "subject_manifest_sha256": stamp}
            for k in range(n)]


@pytest.mark.parametrize("package", ["triton_kernels", "metal_kernels"])
def test_the_manifest_is_the_one_the_census_driver_stamps(tmp_path, package):
    battery = _tree(tmp_path, package)
    stamped = census_driver.subject_digest(tmp_path, package, battery)
    digest, count = census_subject.live_manifest(tmp_path, package)
    assert (digest, count) == (stamped["manifest_sha256"], stamped["file_count"])
    assert count == 3 + len(census_subject.SUBJECT_EXTRA)
    assert tuple(stamped["scope"][1:]) == census_subject.SUBJECT_EXTRA


def test_the_engine_files_are_the_census_drivers():
    assert census_subject.SUBJECT_EXTRA == census_driver.SUBJECT_EXTRA


def test_a_census_cut_against_this_tree_is_accepted(tmp_path):
    _tree(tmp_path, "triton_kernels")
    digest, count = census_subject.live_manifest(tmp_path, "triton_kernels")
    pin = census_subject.subject_pin(_rows(digest), api_root=tmp_path,
                                     package="triton_kernels", census="census_x")
    assert pin["manifest_sha256"] == digest and pin["subject_files"] == count
    assert pin["rows_checked"] == 3 and pin["agrees_with_census"] is True


@pytest.mark.parametrize("change", ["edit", "add", "remove", "engine"])
def test_a_subject_that_moved_since_the_cut_is_refused_by_name(tmp_path, change):
    _tree(tmp_path, "triton_kernels")
    digest, _count = census_subject.live_manifest(tmp_path, "triton_kernels")
    package = tmp_path / "meep_gpu" / "triton_kernels"
    if change == "edit":
        (package / "a.py").write_text("X = 'edited'\n")
    elif change == "add":
        (package / "c.py").write_text("X = 'c'\n")
    elif change == "remove":
        (package / "b.py").unlink()
    else:
        (tmp_path / "meep_gpu" / "pml.py").write_text("# moved\n")
    with pytest.raises(SystemExit) as refused:
        census_subject.subject_pin(_rows(digest), api_root=tmp_path, package="triton_kernels",
                                   census="census_x")
    text = str(refused.value)
    assert "the census subject and the working tree disagree" in text and "census_x" in text
    assert digest in text and "Re-cut the census" in text


def test_rows_without_a_stamp_are_refused_not_read_as_a_match(tmp_path):
    _tree(tmp_path, "metal_kernels")
    digest, _ = census_subject.live_manifest(tmp_path, "metal_kernels")
    rows = _rows(digest) + _rows(None, 2)
    with pytest.raises(SystemExit) as refused:
        census_subject.subject_pin(rows, api_root=tmp_path, package="metal_kernels",
                                   census="census_x")
    assert "2 of 5 measured rows carry no subject_manifest_sha256" in str(refused.value)
    with pytest.raises(SystemExit, match="5 of 5 measured rows carry no"):
        census_subject.subject_pin(_rows(None, 5), api_root=tmp_path, package="metal_kernels",
                                   census="census_x")


def test_rows_stamped_with_two_manifests_are_not_one_census(tmp_path):
    _tree(tmp_path, "metal_kernels")
    digest, _ = census_subject.live_manifest(tmp_path, "metal_kernels")
    with pytest.raises(SystemExit, match="the census subject and the working tree disagree"):
        census_subject.subject_pin(_rows(digest) + _rows("0" * 64, 1), api_root=tmp_path,
                                   package="metal_kernels", census="census_x")


def test_a_census_with_no_measured_rows_has_nothing_to_check(tmp_path):
    _tree(tmp_path, "metal_kernels")
    with pytest.raises(SystemExit, match="has no measured rows"):
        census_subject.subject_pin([], api_root=tmp_path, package="metal_kernels",
                                   census="census_x")


def _census(root: Path, stamp) -> Path:
    """A census directory in the shape the boards read, one measured row, one unmeasured."""
    root.mkdir()
    measured = {"leg": "examples", "row": "a.py", "measured": True}
    unmeasured = {"leg": "examples", "row": "b.py", "measured": False, "note": "child died"}
    if stamp is not None:
        measured["subject_manifest_sha256"] = stamp
    (root / "examples.jsonl").write_text(json.dumps(measured) + "\n" + json.dumps(unmeasured)
                                         + "\n")
    (root / "tests.jsonl").write_text("")
    (root / "tests_param_matched.jsonl").write_text("")
    return root


def test_the_unmeasured_rows_of_a_census_are_not_asked_for_a_stamp(tmp_path):
    digest, _ = census_subject.live_manifest(triton_board.API, "triton_kernels")
    census = _census(tmp_path / "good", digest)
    pin = census_subject.subject_pin(triton_board.rows(census), api_root=triton_board.API,
                                     package="triton_kernels", census=census)
    assert pin["rows_checked"] == 1


def test_the_triton_board_refuses_a_census_cut_against_another_tree(tmp_path, monkeypatch,
                                                                    capsys):
    census = _census(tmp_path / "old", "1" * 64)
    out = tmp_path / "board"
    monkeypatch.setattr(sys, "argv", ["build_triton_fusion_matrix.py", "--out", str(out),
                                      "--census", str(census)])
    monkeypatch.setattr(triton_board, "CENSUS", triton_board.CENSUS)   # restored after main
    with pytest.raises(SystemExit) as refused:
        triton_board.main()
    assert "the census subject and the working tree disagree" in str(refused.value)
    assert "triton_kernels" in str(refused.value) and not out.exists()
    assert capsys.readouterr().out == ""            # nothing was read or priced first


def test_the_triton_board_refuses_a_census_with_no_stamp(tmp_path, monkeypatch):
    census = _census(tmp_path / "unstamped", None)
    monkeypatch.setattr(sys, "argv", ["build_triton_fusion_matrix.py", "--out",
                                      str(tmp_path / "board"), "--census", str(census)])
    monkeypatch.setattr(triton_board, "CENSUS", triton_board.CENSUS)
    with pytest.raises(SystemExit, match="1 of 1 measured rows carry no subject_manifest_sha256"):
        triton_board.main()


def test_the_metal_board_refuses_a_census_cut_against_another_tree(tmp_path, monkeypatch,
                                                                   capsys):
    census = _census(tmp_path / "old", "2" * 64)
    out = tmp_path / "board"
    monkeypatch.setattr(sys, "argv", ["build_fusion_matrix.py", "--out", str(out)])
    monkeypatch.setattr(metal_board, "CENSUS", census)
    with pytest.raises(SystemExit) as refused:
        metal_board.main()
    assert "the census subject and the working tree disagree" in str(refused.value)
    assert "metal_kernels" in str(refused.value)
    # the pin comes before the release-binding walk and every measurement
    assert "RELEASE BINDING" not in capsys.readouterr().out


def test_the_metal_pin_is_over_the_census_rows_not_the_reclose_replacement(tmp_path,
                                                                           monkeypatch):
    """``rows()`` replaces one census row by a different run's record, which carries its own
    (older) stamp or none; the census's stamp describes the rows the census recorded."""
    census = _census(tmp_path / "c", "3" * 64)
    monkeypatch.setattr(metal_board, "CENSUS", census)
    assert [r["row"] for r in metal_board.census_rows()] == ["a.py"]
    assert all(r["subject_manifest_sha256"] == "3" * 64 for r in metal_board.census_rows())
