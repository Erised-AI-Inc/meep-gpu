"""The Metal round's inputs bundle: the committed manifest, and the tool that packs it.

Host-only, no device and no evidence archive: the committed manifest is checked for
consistency and for host facts, and ``build_metal_round_inputs.py`` is driven on a
synthetic results tree built in a temporary directory.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_metal_round_inputs as tool  # noqa: E402

ROUND = HERE / "rounds" / "metal_2026-10-04"
REPO = HERE.parents[1]


# ---------------------------------------------------------------------------
# The committed round directory
# ---------------------------------------------------------------------------

def test_the_committed_manifest_is_consistent():
    files, dirs, sums = tool.read_manifest(ROUND)
    assert len(files) == 7 and len(dirs) == 4
    assert files == sorted(files), "INPUT_FILES.txt is kept in sorted order"
    assert set(sums) == set(files)
    assert len(tool.committed_bundle_digest(ROUND)) == 64


def test_the_four_dated_expansion_probes_are_not_in_the_bundle():
    """They are measured per Mac; the round cuts its own (README.md)."""
    files, dirs, _sums = tool.read_manifest(ROUND)
    dated = re.compile(r"metal_(complex_audit|complex|special_kz|folded_complex|"
                       r"cylindrical_complex)_2026-08-1\d")
    assert not [name for name in files + dirs if dated.match(name)]


def test_the_spine_baseline_ships_because_the_gate_reads_it_at_that_path():
    files, _dirs, _sums = tool.read_manifest(ROUND)
    gate = (HERE / "gate_metal_spine_byte_neutrality.py").read_text(encoding="utf-8")
    assert '"metal_pml_2026-08-14", "provenance.json"' in gate
    assert "metal_pml_2026-08-14/provenance.json" in files


@pytest.mark.parametrize("name", sorted(p.name for p in ROUND.iterdir() if p.is_file()))
def test_no_committed_round_file_carries_a_host_fact(name):
    text = (ROUND / name).read_text(encoding="utf-8")
    # Spelled in parts so this file does not itself match a scrub for home paths.
    homes = ("/" + "Users/", "/" + "home/")
    for pattern in homes + (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}",):
        assert not re.search(pattern, text), f"{name} matches {pattern}"


def test_the_round_driver_verifies_the_inputs_before_the_fleet():
    text = (ROUND / "run_round.sh").read_text(encoding="utf-8")
    order = re.search(r"^ORDER=\(([^)]*)\)", text, re.M).group(1).split()
    assert order == ["preflight", "inputs", "fleet", "status", "pack"]
    assert "build_metal_round_inputs.py\" unpack" in text
    assert "build_metal_round_inputs.py\" check" in text
    assert "need inputs" in text.split("stage_fleet() {", 1)[1].split("}", 1)[0]


def _body(text: str, function: str) -> str:
    """A stage function's body: from its opening line to the first closing brace at
    column 0. The stages checked here hold no heredoc, so that brace is theirs."""
    return text.split(f"{function}() {{\n", 1)[1].split("\n}\n", 1)[0]


def test_every_stage_refuses_to_start_before_the_one_it_follows():
    text = (ROUND / "run_round.sh").read_text(encoding="utf-8")
    # The first line of each body: a check placed after other work would run that
    # work first. stage_status holds a heredoc, so it is read by its first line only.
    for function, before in (("stage_inputs", "preflight"), ("stage_fleet", "inputs"),
                             ("stage_status", "fleet"), ("stage_pack", "status")):
        first = text.split(f"{function}() {{\n", 1)[1].splitlines()[0].strip()
        assert first == f"need {before}", f"{function} starts with {first!r}"


def test_the_stages_that_run_or_pack_gates_re_check_the_commit():
    """``all`` skips a finished preflight, so a resumed run re-checks here."""
    text = (ROUND / "run_round.sh").read_text(encoding="utf-8")
    check = _body(text, "tree_is_the_commit")
    assert '"$head" == "$EXPECT"' in check and "status --porcelain" in check
    for function in ("stage_preflight", "stage_fleet", "stage_pack"):
        assert "tree_is_the_commit" in _body(text, function), function
    assert "gpu_idle" in _body(text, "stage_fleet")


def test_the_driver_reads_the_manifest_from_the_checkout_and_checks_links_first():
    text = (ROUND / "run_round.sh").read_text(encoding="utf-8")
    assert "ROUND=$G/rounds/metal_2026-10-04" in text
    assert "${0:A:h}" not in text and "$HERE" not in text
    link_check = text.index('[[ ! -L "$R" ]]')
    assert link_check < text.index('mkdir -p "$L"'), (
        "the log directory is created under results/ before the link check")


needs_zsh_here = pytest.mark.skipif(shutil.which("zsh") is None,
                                    reason="the round driver needs zsh")


@pytest.mark.requires_resource("zsh")
@needs_zsh_here
@pytest.mark.parametrize("layout", ["results_is_a_link", "no_manifest"])
def test_the_driver_stops_before_writing_anything(tmp_path, layout):
    """A results/ that is a link, or a REPO without this round's manifest, stops the
    driver before its first write: nothing lands in the link's target or in results/."""
    repo = tmp_path / "repo"
    harness = repo / "parity" / "meep_gpu"
    harness.mkdir(parents=True)
    target = tmp_path / "archive"
    target.mkdir()
    if layout == "results_is_a_link":
        (harness / "results").symlink_to(target)
    else:
        (harness / "results").mkdir()
    environment = {k: v for k, v in os.environ.items() if not k.startswith("MEEP_GPU_")}
    environment.update(REPO=str(repo), EXPECT="0" * 40, STAMP="2000-01-01_test")
    completed = subprocess.run(["zsh", str(ROUND / "run_round.sh"), "preflight"],
                               env=environment, capture_output=True, text=True,
                               timeout=60, check=False)
    assert completed.returncode == 3, completed.stdout + completed.stderr
    expected = "symbolic link" if layout == "results_is_a_link" else "holds no manifest"
    assert expected in completed.stdout
    assert not any(target.iterdir())
    assert not any((harness / "results").iterdir())


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
