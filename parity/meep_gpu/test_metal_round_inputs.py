"""The tool that packs, verifies and unpacks the inputs bundle of a Metal certification round.

Host-only, no device and no evidence archive: ``build_metal_round_inputs.py`` is driven on a
synthetic results tree and a synthetic round directory built in a temporary directory.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_metal_round_inputs as tool  # noqa: E402

REPO = HERE.parents[1]


# ---------------------------------------------------------------------------
# The tool, on a synthetic tree
# ---------------------------------------------------------------------------

FILES = {
    "census_x/examples.jsonl": b'{"row": "a"}\n' * 50,
    "census_x/tests.jsonl": b'{"row": "b"}\n' * 20,
    "seam_y/seam.jsonl": b'{"label": "c"}\n',
    "baseline_z/sub/provenance.json": b'{"kernel_sources": {}}\n',
}


def _round(tmp_path: Path, results: Path) -> Path:
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    names = sorted(FILES)
    (round_dir / tool.FILES).write_text("".join(n + "\n" for n in names))
    (round_dir / tool.DIRS).write_text("baseline_z\ncensus_x\nseam_y\n")
    (round_dir / tool.SUMS).write_text("".join(
        f"{hashlib.sha256(FILES[n]).hexdigest()}  {n}\n" for n in names))
    for name, data in FILES.items():
        (results / name).parent.mkdir(parents=True, exist_ok=True)
        (results / name).write_bytes(data)
    return round_dir


@pytest.fixture
def tree(tmp_path):
    results = tmp_path / "archive"
    results.mkdir()
    round_dir = _round(tmp_path, results)
    bundle = tmp_path / "out" / tool.BUNDLE
    digest = tool.pack(round_dir, results, bundle)
    (round_dir / tool.BUNDLE_SUM).write_text(f"{digest}  {tool.BUNDLE}\n")
    return round_dir, results, bundle, digest


def test_pack_is_deterministic_and_carries_exactly_the_manifest(tree, tmp_path):
    round_dir, results, bundle, digest = tree
    again = tool.pack(round_dir, results, tmp_path / "again" / tool.BUNDLE)
    assert again == digest
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
    assert [m.name for m in members] == [tool.SUMS] + sorted(FILES)
    assert {(m.mtime, m.uid, m.gid, m.uname, m.gname) for m in members} == {(0, 0, 0, "", "")}
    assert tool.verify_bundle(round_dir, bundle) == []
    assert (bundle.parent / (tool.BUNDLE + ".sha256")).read_text().startswith(digest)


def test_pack_refuses_a_source_that_differs_from_the_manifest(tree, tmp_path):
    round_dir, results, _bundle, _digest = tree
    (results / "seam_y/seam.jsonl").write_bytes(b"edited\n")
    with pytest.raises(tool.InputsRefused, match="differs: seam_y/seam.jsonl"):
        tool.pack(round_dir, results, tmp_path / "x" / tool.BUNDLE)


def test_pack_refuses_to_write_inside_the_repository(tree):
    round_dir, results, _bundle, _digest = tree
    with pytest.raises(tool.InputsRefused, match="inside the repository"):
        tool.pack(round_dir, results, REPO / "parity" / "meep_gpu" / "rounds" / "x.tgz")
    assert not (REPO / "parity" / "meep_gpu" / "rounds" / "x.tgz").exists()


def test_unpack_extracts_what_is_absent_and_is_idempotent(tree, tmp_path):
    round_dir, _results, bundle, _digest = tree
    target = tmp_path / "second_mac_results"
    target.mkdir()
    assert tool.unpack(round_dir, bundle, target) == (len(FILES), 0)
    assert tool.check_present(round_dir, target) == []
    assert tool.unpack(round_dir, bundle, target) == (0, len(FILES))
    assert not list(target.rglob("*.tmp")) and not (target / tool.SUMS).exists()


def test_unpack_never_writes_through_a_symbolic_link(tree, tmp_path):
    """The archive host's results/ entries link into the archive."""
    round_dir, results, bundle, _digest = tree
    target = tmp_path / "linked_results"
    target.mkdir()
    (target / "seam_y").symlink_to(results / "seam_y")
    before = sorted(p.name for p in target.iterdir())
    with pytest.raises(tool.InputsRefused, match="symbolic link"):
        tool.unpack(round_dir, bundle, target)
    assert sorted(p.name for p in target.iterdir()) == before, (
        "a refused unpack wrote some files before refusing")


def test_unpack_refuses_a_present_file_that_differs(tree, tmp_path):
    round_dir, _results, bundle, _digest = tree
    target = tmp_path / "t"
    (target / "census_x").mkdir(parents=True)
    (target / "census_x/tests.jsonl").write_bytes(b"another run's census\n")
    with pytest.raises(tool.InputsRefused, match="not the manifest's file"):
        tool.unpack(round_dir, bundle, target)
    assert (target / "census_x/tests.jsonl").read_bytes() == b"another run's census\n"
    assert not (target / "census_x/examples.jsonl").exists()


def test_unpack_refuses_a_bundle_other_than_the_one_the_manifest_names(tree, tmp_path):
    round_dir, results, _bundle, _digest = tree
    other = tmp_path / "other.tgz"
    with tarfile.open(other, "w:gz") as archive:
        data = b"x"
        info = tarfile.TarInfo("census_x/tests.jsonl")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    target = tmp_path / "t2"
    target.mkdir()
    with pytest.raises(tool.InputsRefused, match="not the bundle cut for this manifest"):
        tool.unpack(round_dir, other, target)
    assert not any(target.iterdir())
    problems = tool.verify_bundle(round_dir, other)
    assert problems and "members differ" in problems[0]


def test_verify_refuses_a_bundle_whose_digest_is_not_the_committed_one(tree, tmp_path):
    """Members that verify are not enough: ``unpack`` checks the digest, so does ``verify``."""
    round_dir, _results, bundle, digest = tree
    assert tool.main(["verify", "--round", str(round_dir), "--bundle", str(bundle)]) == 0
    other = "0" * 64 if digest != "0" * 64 else "1" * 64
    (round_dir / tool.BUNDLE_SUM).write_text(f"{other}  {tool.BUNDLE}\n")
    assert tool.main(["verify", "--round", str(round_dir), "--bundle", str(bundle)]) == 1


@pytest.mark.parametrize("line", ["/abs/path", "census_x/../../escape", "./x"])
def test_a_manifest_path_that_leaves_results_is_refused(tmp_path, line):
    results = tmp_path / "archive"
    results.mkdir()
    round_dir = _round(tmp_path, results)
    (round_dir / tool.FILES).write_text(line + "\n")
    with pytest.raises(tool.ManifestError):
        tool.read_manifest(round_dir)


def test_the_command_line_exit_status(tree, tmp_path, capsys):
    round_dir, results, bundle, _digest = tree
    assert tool.main(["check", "--round", str(round_dir), "--results", str(results)]) == 0
    os.remove(results / "seam_y/seam.jsonl")
    assert tool.main(["check", "--round", str(round_dir), "--results", str(results)]) == 1
    assert "missing: seam_y/seam.jsonl" in capsys.readouterr().out
    assert tool.main(["verify", "--round", str(round_dir), "--bundle", str(bundle)]) == 0
    (round_dir / tool.SUMS).write_text("not a digest line\n")
    assert tool.main(["check", "--round", str(round_dir), "--results", str(results)]) == 2
