"""The 1.0 round's inputs manifest and the tool that derives, packs and stages it.

Host-only, no device and no evidence archive. The committed manifest is held to the code
both ways: every file it lists outside the Tier B record is one a reader of the round
opens (``round_inputs.derive``), every such file is listed, and every module-level name
that binds a path under ``results/`` in a module the round runs is a reader or a named
non-input (``round_inputs.audit``). The tool itself is driven on a synthetic results tree
built in a temporary directory.
"""
from __future__ import annotations

import ast
import hashlib
import io
import os
import re
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROUND = HERE
REPO = HERE.parents[3]
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

import round_inputs as tool  # noqa: E402


# ---------------------------------------------------------------------------
# The committed manifest
# ---------------------------------------------------------------------------

def test_the_committed_manifest_is_consistent():
    files, dirs, sums = tool.read_manifest(ROUND)
    assert files == sorted(files) and dirs == tool.top_dirs(files)
    assert set(sums) == set(files)
    assert len(tool.committed_bundle_digest(ROUND)) == 64


def test_the_code_readers_name_exactly_the_manifest_outside_tier_b():
    """A reader added, moved or dropped in a gate fails here, before the round."""
    files, _dirs, _sums = tool.read_manifest(ROUND)
    derived = tool.derive(None, code_only=True)
    listed = [name for name in files if not name.startswith(tool.TIER_B + "/")]
    assert sorted(derived) == listed, (
        f"only derived: {sorted(set(derived) - set(listed))[:8]}; "
        f"only listed: {sorted(set(listed) - set(derived))[:8]}")


def test_every_results_binding_of_the_round_is_a_reader_or_a_named_non_input():
    problems, facts = tool.audit(REPO)
    assert problems == [], problems
    assert facts["bindings"] >= len(tool.READERS)


def test_the_tier_b_part_is_the_record_and_the_row_files_l3_verifies():
    files, _dirs, _sums = tool.read_manifest(ROUND)
    tier_b = [name for name in files if name.startswith(tool.TIER_B + "/")]
    assert tool.TIER_B_RECORD in tier_b
    rows = [name for name in tier_b if name != tool.TIER_B_RECORD]
    shape = re.compile(r"meep_identity_tier_b/(linux_keep|linux_flush|mac_flush)/"
                       r"([^/]+)/(keep|flush)/\2/(digests\.json|planes\.npy|result\.json)")
    assert rows and all(shape.fullmatch(name) for name in rows), [
        name for name in rows if not shape.fullmatch(name)][:5]
    # Every planned row ships its three files in each set it ships in.
    by_row = {}
    for name in rows:
        set_name, label = name.split("/")[1:3]
        by_row.setdefault((set_name, label), set()).add(name.rsplit("/", 1)[1])
    assert all(found == {"digests.json", "planes.npy", "result.json"}
               for found in by_row.values())
    # No leg judges mac_keep, and the Metal leg's three sets hold the same rows.
    sets = {}
    for set_name, label in by_row:
        sets.setdefault(set_name, set()).add(label)
    assert set(sets) == {"linux_keep", "linux_flush", "mac_flush"}
    assert sets["mac_flush"] <= sets["linux_keep"] and sets["mac_flush"] == sets["linux_flush"]


def test_l3_verifies_the_roles_this_tool_ships():
    """``L3_ROLES`` is read off ``l3_corpus.verify_row``'s own loop, not assumed."""
    source = (ROUND / "l3_corpus.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "verify_row")
    loops = [node.iter for node in ast.walk(function) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "role"]
    assert len(loops) == 1 and isinstance(loops[0], ast.Tuple)
    assert tuple(element.value for element in loops[0].elts) == tool.L3_ROLES


def test_the_corpus_manifest_is_derived_from_the_shipped_legs_not_shipped():
    files, _dirs, _sums = tool.read_manifest(ROUND)
    assert all(leg in files for leg in tool.corpus_legs())
    assert len(tool.corpus_legs()) == 7
    assert not any(name.endswith("corpus_manifest.json") for name in files)


def test_records_no_stage_of_the_round_reads_are_not_shipped():
    """The 0.9.x inputs the 1.0 code no longer opens: the complex expansion licence
    (deleted in 1.0, ``legs.py``'s module docstring) and the signed-zero probe."""
    files, dirs, _sums = tool.read_manifest(ROUND)
    for retired in ("complex_signed_zero_2026-08-12", "complex_expansion_convention_2026-08-16",
                    "complex_expansion_beta_extension_2026-09-01", "expansion_probe_2026-08-17",
                    "unified_expansion_2026-08-27", "triton_welds_2026-08-19",
                    "cuda_complex_regate_2026-08-26"):
        assert retired not in dirs, retired
    # A census directory ships the files its readers open, never its per-row records.
    assert not [name for name in files if "/per_row" in name]


@pytest.mark.parametrize("name", [tool.FILES, tool.DIRS, tool.SUMS, tool.BUNDLE_SUM])
def test_no_committed_manifest_file_carries_a_host_fact(name):
    text = (ROUND / name).read_text(encoding="utf-8")
    # Spelled in parts so this file does not itself match a scrub for home paths.
    homes = ("/" + "Users/", "/" + "home/")
    for pattern in homes + (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}",):
        assert not re.search(pattern, text), f"{name} matches {pattern}"


# ---------------------------------------------------------------------------
# Reading a reader's binding from its source
# ---------------------------------------------------------------------------

def _module(harness: Path, name: str, text: str) -> None:
    (harness / f"{name}.py").write_text(text, encoding="utf-8")


def test_bound_path_reads_every_spelling_the_readers_use(tmp_path):
    _module(tmp_path, "kit", 'CENSUS = "census_2026-01-01"\n')
    _module(tmp_path, "reader", "\n".join([
        "import os", "import kit",
        'A = "seam_2026-01-02"',
        'B = os.path.join(HERE, "results", "board_2026-01-03", "matrix.json")',
        'C = os.path.join(API_ROOT, "parity", "meep_gpu", "results", "c_2026-01-04")',
        'D = kit.CENSUS',
        'E: str = RESULTS / "e_2026-01-05"',
        'RESULTS = _HERE / "results"',
        'F = A']) + "\n")
    expect = {"A": "seam_2026-01-02", "B": "board_2026-01-03/matrix.json",
              "C": "c_2026-01-04", "D": "census_2026-01-01", "E": "e_2026-01-05",
              "F": "seam_2026-01-02"}
    for name, path in expect.items():
        assert tool.bound_path("reader", name, tmp_path) == path, name
    # ``D`` and ``F`` spell no path of their own (an alias, a name); ``RESULTS`` is the
    # root itself. The audit finds a reader by the binding that spells the path.
    assert tool.results_bindings("reader", tmp_path) == ["A", "B", "C", "E"]


@pytest.mark.parametrize("text", [
    'X = os.path.join(HERE, "outside_2026-01-01")\n',
    'X = some_function("results")\n',
    'X = os.path.join(HERE, "results")\n',
])
def test_bound_path_refuses_what_is_not_a_record_under_results(tmp_path, text):
    _module(tmp_path, "reader", "import os\n" + text)
    with pytest.raises(tool.ManifestError):
        tool.bound_path("reader", "X", tmp_path)


# ---------------------------------------------------------------------------
# The tool, on a synthetic tree
# ---------------------------------------------------------------------------

FILES = {
    "census_x/examples.jsonl": b'{"row": "a"}\n' * 50,
    "census_x/tests.jsonl": b'{"row": "b"}\n' * 20,
    "seam_y/h_to_d_seam.jsonl": b'{"label": "c"}\n',
    "baseline_z/sub/provenance.json": b'{"kernel_sources": {}}\n',
    "meep_identity_tier_b/linux_flush/r/flush/r/planes.npy": bytes(range(256)) * 8,
}


def _round(tmp_path: Path, results: Path) -> Path:
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    names = sorted(FILES)
    (round_dir / tool.FILES).write_text("".join(n + "\n" for n in names))
    (round_dir / tool.DIRS).write_text("".join(d + "\n" for d in tool.top_dirs(names)))
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
    assert {(m.mtime, m.uid, m.gid, m.uname, m.gname, m.mode)
            for m in members} == {(0, 0, 0, "", "", 0o644)}
    assert tool.verify_bundle(round_dir, bundle) == []
    assert (bundle.parent / (tool.BUNDLE + ".sha256")).read_text().startswith(digest)


def test_pack_reads_through_a_view_of_links_into_the_archive(tree, tmp_path):
    """The archive host packs from a directory of links (the Tier B record's home is a
    link to wherever the archive keeps it); the bundle is the same bytes."""
    round_dir, results, _bundle, digest = tree
    view = tmp_path / "view"
    view.mkdir()
    for top in tool.top_dirs(FILES):
        (view / top).symlink_to(results / top)
    assert tool.pack(round_dir, view, tmp_path / "from_view" / tool.BUNDLE) == digest


def test_pack_refuses_a_source_that_differs_from_the_manifest(tree, tmp_path):
    round_dir, results, _bundle, _digest = tree
    (results / "seam_y/h_to_d_seam.jsonl").write_bytes(b"edited\n")
    with pytest.raises(tool.InputsRefused, match="differs: seam_y/h_to_d_seam.jsonl"):
        tool.pack(round_dir, results, tmp_path / "x" / tool.BUNDLE)


def test_pack_refuses_to_write_inside_the_repository(tree):
    round_dir, results, _bundle, _digest = tree
    target = REPO / "parity" / "meep_gpu" / "rounds" / "x.tgz"
    with pytest.raises(tool.InputsRefused, match="inside the repository"):
        tool.pack(round_dir, results, target)
    assert not target.exists()


def test_unpack_extracts_what_is_absent_and_is_idempotent(tree, tmp_path):
    round_dir, _results, bundle, _digest = tree
    target = tmp_path / "second_host_results"
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


def test_unpack_refuses_a_results_directory_that_is_a_link(tree, tmp_path):
    round_dir, results, bundle, _digest = tree
    target = tmp_path / "results_link"
    target.symlink_to(results)
    with pytest.raises(tool.InputsRefused, match="must be a real directory"):
        tool.unpack(round_dir, bundle, target)


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
    round_dir, _results, _bundle, _digest = tree
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


def test_verify_refuses_a_bundle_whose_digest_is_not_the_committed_one(tree):
    """Members that verify are not enough: ``unpack`` checks the digest, so does ``verify``."""
    round_dir, _results, bundle, digest = tree
    argv = ["verify", "--round", str(round_dir), "--bundle", str(bundle)]
    assert tool.main(argv) == 0
    other = "0" * 64 if digest != "0" * 64 else "1" * 64
    (round_dir / tool.BUNDLE_SUM).write_text(f"{other}  {tool.BUNDLE}\n")
    assert tool.main(argv) == 1


@pytest.mark.parametrize("line", ["/abs/path", "census_x/../../escape", "./x", "a//b"])
def test_a_manifest_path_that_leaves_results_is_refused(tmp_path, line):
    results = tmp_path / "archive"
    results.mkdir()
    round_dir = _round(tmp_path, results)
    (round_dir / tool.FILES).write_text(line + "\n")
    with pytest.raises(tool.ManifestError):
        tool.read_manifest(round_dir)


def test_a_dirs_list_that_is_not_the_files_top_directories_is_refused(tmp_path):
    results = tmp_path / "archive"
    results.mkdir()
    round_dir = _round(tmp_path, results)
    (round_dir / tool.DIRS).write_text("census_x\n")
    with pytest.raises(tool.ManifestError, match="top-level directories"):
        tool.read_manifest(round_dir)


def test_write_manifest_round_trips_through_read_manifest(tmp_path):
    results = tmp_path / "archive"
    results.mkdir()
    round_dir = _round(tmp_path, results)
    for name in (tool.FILES, tool.DIRS, tool.SUMS):
        os.remove(round_dir / name)
    sums = tool.write_manifest(round_dir, results, list(FILES))
    files, dirs, read = tool.read_manifest(round_dir)
    assert files == sorted(FILES) and read == sums and dirs == tool.top_dirs(FILES)
    with pytest.raises(tool.InputsRefused, match="absent"):
        tool.write_manifest(round_dir, results, list(FILES) + ["census_x/absent.jsonl"])


def test_the_command_line_exit_status(tree, capsys):
    round_dir, results, bundle, _digest = tree
    check = ["check", "--round", str(round_dir), "--results", str(results)]
    assert tool.main(check) == 0
    os.remove(results / "seam_y/h_to_d_seam.jsonl")
    assert tool.main(check) == 1
    assert "missing: seam_y/h_to_d_seam.jsonl" in capsys.readouterr().out
    assert tool.main(["verify", "--round", str(round_dir), "--bundle", str(bundle)]) == 0
    (round_dir / tool.SUMS).write_text("not a digest line\n")
    assert tool.main(check) == 2
