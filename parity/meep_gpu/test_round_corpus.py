"""The 1.0 round names the MEEP corpus the CUDA H->D lifts walk (host-only).

The four CUDA H->D gates (``--lift-steps``) re-run MEEP corpus rows from
``measure_predicate_coverage.EXAMPLES_DIR``/``TESTS_DIR``. A round that names no corpus
on a host where none of the defaults exists refuses every lift, measured on the GPU host
in the 1.0 pre-certification pass. These tests pin the resolution chain, the refusal
before a GPU is taken, and the export to every child.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pytest

ROUND = Path(__file__).resolve().parent / "rounds" / "1.0"
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

import legs as L  # noqa: E402
import round as round_driver  # noqa: E402
import run_lane  # noqa: E402

LIFT = {"id": "residue/cuda_fused_hd_pair_2026-09-06/keep", "kind": "gate",
        "script": "gate_cuda_fused_hd_pair.py",
        "args": ["--subnormal-policy", "keep", "--steps", "60", "--lift-steps", "60"]}
PLAIN = {"id": "cited/x/keep/gate.json", "kind": "gate", "script": "gate_cuda_x.py",
         "args": ["--subnormal-policy", "keep"]}


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "meep" / "python"
    for name in L.CORPUS_DIRECTORIES:
        (root / name).mkdir(parents=True)
    return root


def test_the_resolution_chain_is_the_lifting_gates_own():
    env = {"MEEP_GPU_CORPUS_ROOT": "/a", "MGPU_SITE_MEEP_SOURCE": "/b"}
    assert L.corpus_root("/explicit", env) == "/explicit"
    assert L.corpus_root("", env) == "/a"
    assert L.corpus_root("", {"MGPU_SITE_MEEP_SOURCE": "/b"}) == "/b/python"
    assert L.corpus_root("", {}) == str(Path("~/meep/python").expanduser())


def test_the_eight_lifting_legs_are_the_four_hd_families_under_both_policies(tmp_path):
    lifting = L.lifting_legs(L.cuda_legs(tmp_path, tmp_path / "results", "s"))
    assert len(lifting) == 8
    assert {leg.split("/")[1] for leg in lifting} == {
        "cuda_fused_hd_pair_2026-09-06", "cuda_complex_fused_hd_pair_2026-09-07",
        "cuda_cylindrical_fused_hd_pair_2026-09-07",
        "cuda_cylindrical_real_fused_hd_pair_2026-09-07"}


def test_a_plan_without_lifts_needs_no_corpus(tmp_path):
    assert L.corpus_problems([PLAIN], str(tmp_path / "absent")) == []


def test_a_lifting_plan_with_no_corpus_is_refused_by_name(tmp_path):
    problems = L.corpus_problems([PLAIN, LIFT], str(tmp_path / "absent"))
    assert len(problems) == 1
    assert "MEEP_GPU_CORPUS_ROOT" in problems[0] and "examples/tests" in problems[0]
    half = tmp_path / "half"
    (half / "examples").mkdir(parents=True)
    assert "tests" in L.corpus_problems([LIFT], str(half))[0]
    assert L.corpus_problems([LIFT], str(_corpus(tmp_path))) == []


def _lane_args(tmp_path: Path, corpus: str) -> argparse.Namespace:
    return argparse.Namespace(lane_dir=str(tmp_path / "lane"), repo=str(tmp_path),
                              results=str(tmp_path / "results"), python=sys.executable,
                              gpu=0, subs="", libcuda_stub="", corpus_root=corpus)


def test_the_lane_exports_the_checked_root_to_every_child(tmp_path, monkeypatch):
    monkeypatch.delenv("MEEP_GPU_CORPUS_ROOT", raising=False)
    root = _corpus(tmp_path)
    lane = run_lane.Lane(_lane_args(tmp_path, str(root)))
    assert lane.base_env()["MEEP_GPU_CORPUS_ROOT"] == str(root)
    lane = run_lane.Lane(_lane_args(tmp_path, str(tmp_path / "absent")))
    assert "MEEP_GPU_CORPUS_ROOT" not in lane.base_env()


def test_a_lifting_lane_with_no_corpus_stops_before_its_first_step(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([LIFT]))
    lane_dir = tmp_path / "lane"
    rc = run_lane.main(["--plan", str(plan), "--lane-dir", str(lane_dir), "--repo",
                        str(tmp_path), "--results", str(tmp_path / "results"), "--gpu", "0",
                        "--no-lock", "--corpus-root", str(tmp_path / "absent")])
    assert rc == 3
    assert "MEEP corpus" in (lane_dir / "STOPPED").read_text() or \
        "corpus root" in (lane_dir / "STOPPED").read_text()
    assert not (lane_dir / "steps.jsonl").exists()


def test_the_round_passes_the_corpus_to_every_lane(tmp_path):
    args = argparse.Namespace(repo=str(tmp_path), python=sys.executable, lock_dir="/l",
                              max_locks=4, stamp="s", avoid="", libcuda_stub="",
                              timeout_floor=[], corpus_root="/meep/python")
    argv = round_driver.lane_argv(args, tmp_path / "plan.json", tmp_path / "lane",
                                  tmp_path / "results", {})
    assert argv[argv.index("--corpus-root") + 1] == "/meep/python"


@pytest.mark.parametrize("flag", ["--corpus-root"])
def test_the_check_takes_the_corpus_on_its_command_line(flag):
    text = (ROUND / "legs.py").read_text(encoding="utf-8")
    assert f'c.add_argument("{flag}"' in text
    assert "corpus=args.corpus_root" in text


# ------------------------------------------------- the corpus siblings, one rule ------
def _record_writer():
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location(
        "build_meep_133_record_for_the_round", Path(__file__).resolve().parent
        / "build_meep_133_record.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _built_prefix(tmp_path: Path):
    """A prefix whose corpus and record the build's own writer made (keep-corpus, then the
    manifest), with an interpreter that reports the prefix."""
    writer = _record_writer()
    top = tmp_path / "tarball" / "meep-1.33.0"
    for rel, data in {"python/examples/a.py": b"a = 1\n", "python/tests/test_b.py": b"b\n",
                      **{rel: b"h5 " + rel.encode() for rel in writer.CORPUS_SIBLINGS}}.items():
        (top / rel).parent.mkdir(parents=True, exist_ok=True)
        (top / rel).write_bytes(data)
    prefix = tmp_path / "env"
    record_dir = prefix.joinpath(*L.BUILD_RECORD_DIR)
    record_dir.mkdir(parents=True)
    root = record_dir / "meep-1.33.0" / "python"
    listing, siblings = record_dir / "corpus_sha256.txt", record_dir / "corpus_siblings_sha256.txt"
    writer.keep_corpus(str(top / "python"), str(root), str(listing), str(siblings))
    writer.write_json(str(record_dir / L.BUILD_RECORD), {
        "meep": {"tarball_sha256": "0" * 64},
        "corpus": writer.corpus_entry(str(record_dir), "share/meep-gpu-build/meep-1.33.0/python",
                                      str(listing), str(siblings))})
    (prefix / "bin").mkdir()
    python = prefix / "bin" / "python"
    python.write_text(f"#!/bin/sh\necho {prefix}\n")
    python.chmod(0o755)
    return writer, python, record_dir, root


def _three_readers(python: Path, record_dir: Path):
    """``(legs, metal_round, l3_corpus)``: each reader's problems with the corpus the record
    in ``record_dir`` names (``[]`` when it verifies)."""
    import l3_corpus  # noqa: PLC0415
    import metal_round  # noqa: PLC0415
    found = []
    for module in (L, metal_round):
        found.append(module.resolve_corpus(str(python), "", {})["problems"])
    try:
        l3_corpus.resolve_corpus(argparse.Namespace(
            build_record=str(record_dir / L.BUILD_RECORD), python=str(python), corpus_root=None))
        found.append([])
    except l3_corpus.Refusal as exc:
        found.append([str(exc)])
    return found


@pytest.mark.parametrize("change, words", [
    (None, None),
    ("absent", "the corpus sibling tests/cyl-ellipsoid-eps-ref.h5 is absent from"),
    ("changed", "the corpus sibling tests/cyl-ellipsoid-eps-ref.h5 under .* is changed"),
    ("unrecorded", "names no corpus siblings"),
    ("climbs", "is not a file path relative to the corpus root's parent"),
    ("inside", "is not a file path relative to the corpus root's parent"),
])
def test_the_build_record_and_every_reader_judge_the_siblings_by_one_rule(tmp_path, change,
                                                                         words):
    """The record the build writes, read by the gate stage (legs), the Metal round and L3:
    a sibling absent or changed, a record that names no siblings, or a sibling path that
    leaves the source tree or lies in python/, refuses all three, naming it."""
    writer, python, record_dir, root = _built_prefix(tmp_path)
    (sibling,) = writer.CORPUS_SIBLINGS
    path = record_dir / L.BUILD_RECORD
    document = json.loads(path.read_text())
    if change == "absent":
        (root.parent / sibling).unlink()
    elif change == "changed":
        (root.parent / sibling).write_bytes(b"other")
    elif change == "unrecorded":
        del document["corpus"]["siblings"]
    elif change == "climbs":
        document["corpus"]["siblings"][0]["path"] = "../" + sibling
    elif change == "inside":
        document["corpus"]["siblings"][0]["path"] = "python/examples/a.py"
    writer.write_json(str(path), document)
    found = _three_readers(python, record_dir)
    if change is None:
        assert found == [[], [], []]
        notes = L.resolve_corpus(str(python), "", {})["notes"]
        assert f"1 of 1 recorded sibling(s) match beside it ({sibling})" in notes[-1]
        return
    for problems in found:
        assert len(problems) == 1 and re.search(words, problems[0]), found
    assert found[0] == found[1]
