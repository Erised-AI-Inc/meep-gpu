"""A new compute capability certifies through the same round, by a flag that says so (host-only).

``certification.md`` says a family gate runs on any card, because none goes through dispatch
admission: only the route and licence legs need the ledgers to admit the card. These tests pin
how the 1.0 round tooling makes that true without ever admitting a card silently:

* ``legs.check`` refuses a gate stage on a capability a table does not admit unless
  ``--new-capability`` says the capability is new, refuses the flag on a capability both
  tables admit and on the route stage, and records the mode it ran in;
* a new capability runs chunked (``round.py`` refuses its unchunked gate stage), its chunk
  index names the card, and a launch on another card or in the other mode is refused;
* every capability's licence lands in a directory of its own, which the re-cut accepts;
* the round names the MEEP corpus once, from ``--corpus-root`` or the reference
  environment's build record, checks the tree against that record's list, and hands it
  to every lane.
"""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROUND = HERE / "rounds" / "1.0"
for path in (str(ROUND), str(HERE), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

import legs as L  # noqa: E402
import round as round_driver  # noqa: E402

from meep_gpu import fastpath, route_stamps  # noqa: E402


# ------------------------------------------------------------------------ the device -----
def test_a_capability_names_its_device_one_way():
    assert L.capability_device("8.6") == "cc86" and L.capability_device("9.0") == "cc90"
    assert L.capability_device("12.0") == "cc120"
    for bad in ("86", "sm_90", "9", "9.10", "09.0", "", None):
        with pytest.raises(ValueError):
            L.capability_device(bad)
    # The suffix a route campaign carries for the same card.
    assert fastpath.route_campaign("x", "9.0") == "x_" + L.capability_device("9.0")


def test_each_capabilitys_licence_has_a_directory_the_recut_accepts(monkeypatch):
    monkeypatch.setattr(route_stamps, "ROUND", "2099-03-04_10")
    monkeypatch.setattr(route_stamps, "REROUTED", {})
    licence = route_stamps.licence_campaign("triton")
    for capability in ("8.6", "9.0"):
        directory = L.licence_directory(licence, capability)
        assert directory == f"{licence}_{L.capability_device(capability)}"
        assert route_stamps.licence_refusal("triton", directory,
                                            L.capability_device(capability)) is None
    a, b = L.licence_directory(licence, "8.6"), L.licence_directory(licence, "9.0")
    assert not a.startswith(b + "_") and not b.startswith(a + "_") and a != b
    names = L.round_roots("2099-03-04_10", 2, "9.0", {"licence_campaign": licence})
    assert f"{licence}_cc90" in names and licence in names


# ------------------------------------------------------------------------- the check -----
@pytest.fixture()
def admitted(monkeypatch, tmp_path):
    """The check on this checkout, with both tables admitting 8.6 only and the corpus and
    hygiene legs answered, so only the capability decides."""
    monkeypatch.setattr(fastpath, "capability_admission",
                        lambda table="triton": {"admitted": ("8.6",), "by_key": {}})
    monkeypatch.setattr(L, "hygiene_problems", lambda repo, legs: [])
    # The corpus leg answered too: in a reference environment legs.check verifies the corpus
    # against the build record of this interpreter, which this fixture's tree is not.
    monkeypatch.setattr(L, "resolve_corpus", lambda python, explicit, env: {
        "root": explicit, "record": None, "problems": [], "notes": []})
    corpus = tmp_path / "python"
    for name in L.CORPUS_DIRECTORIES:
        (corpus / name).mkdir(parents=True)

    def run(capability, route=False, new=False):
        return L.check(REPO, capability, route, corpus=str(corpus), new_capability=new)
    return run


def _capability_problems(problems):
    return [p for p in problems if "capability" in p]


def test_an_unadmitted_capability_is_refused_without_the_flag(admitted):
    problems, facts = admitted("9.0")
    found = _capability_problems(problems)
    assert len(found) == 2 and all("does not admit 9.0" in p and "--new-capability" in p
                                   for p in found), found
    assert facts["capability_mode"] == "certified"


def test_the_flag_admits_a_new_capability_at_the_gate_stage(admitted):
    problems, facts = admitted("9.0", new=True)
    assert _capability_problems(problems) == [], problems
    assert facts["capability_mode"] == "new"
    assert facts["licence_directory"].endswith("_cc90")
    assert facts["triton_route_campaign"].endswith("_cc90")


def test_the_flag_is_refused_on_a_certified_capability_and_on_the_route_stage(admitted):
    problems, _facts = admitted("8.6", new=True)
    assert any("already admit 8.6" in p for p in problems), problems
    problems, _facts = admitted("9.0", route=True, new=True)
    assert any("gate-stage mode" in p for p in problems), problems
    assert sum("does not admit 9.0" in p for p in problems) == 2
    problems, _facts = admitted(None, new=True)
    assert any("names no capability" in p for p in problems), problems


@pytest.mark.parametrize("typo", ["9", "8.60", "sm_90", "9,0", "09.0"])
def test_a_misspelled_capability_is_refused_by_name_with_or_without_the_flag(admitted, typo):
    """A typo is never a new capability: it is refused by name before any table, registry or
    licence name reads it."""
    for new in (False, True):
        problems, facts = admitted(typo, new=new)
        found = [p for p in problems if p.startswith("--capability is refused")]
        assert len(found) == 1 and "major.minor" in found[0], problems
        assert not [p for p in problems if "could not import the checkout" in p], problems
        assert "licence_directory" not in facts


def test_a_certified_capability_runs_as_before(admitted):
    problems, facts = admitted("8.6")
    assert _capability_problems(problems) == [], problems
    assert facts["capability_mode"] == "certified"
    assert facts["licence_directory"].endswith("_cc86")


# --------------------------------------------------------------------- the round driver ---
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    round_driver.add_common(ap)
    return ap


def test_a_new_capabilitys_unchunked_gate_stage_is_refused(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(round_driver.L, "check", lambda repo, capability, route, **kw: (
        [], {"licence_campaign": "dispatch_end_to_end_s"}))
    monkeypatch.setattr(round_driver, "run_lanes", lambda *a, **k: launched.append(a) or {})
    args = _parser().parse_args(["gates", "--repo", str(tmp_path), "--stamp", "s",
                                 "--capability", "9.0", "--new-capability", "--home",
                                 str(tmp_path), "--lock-dir", str(tmp_path), "--corpus-root",
                                 str(tmp_path)])
    assert round_driver.stage_gates(args, tmp_path) == 3 and launched == []
    log = (tmp_path / "round.log").read_text()
    assert "--new-capability runs chunked" in log


@pytest.mark.parametrize("capability, device, mode", [("8.6", "cc86", "certified"),
                                                     ("9.0", "cc90", "new")])
def test_a_chunks_gate_stage_goes_on_from_its_launch_record_to_its_lanes(
        tmp_path, monkeypatch, capability, device, mode):
    """A chunk's gate stage writes its launch record and goes on to plan and run the
    chunk's lanes and to write its bind commands: nothing on the way reads a field a chunk
    of the index's schema does not carry."""
    C = round_driver.C
    units = C.units_for(REPO, "nvidia")
    index = C.make_index("nvidia", "2099-01-01_10", C.partition(units, 4), {"rule": "test"},
                         device=device, device_mode=mode)
    path = tmp_path / "index.json"
    C.write_index(path, index)
    monkeypatch.setattr(round_driver.L, "check", lambda repo, capability, route, **kw: (
        [], {}))
    monkeypatch.setattr(round_driver, "chunk_problems", lambda args, repo, results: [])
    recorded, ran = [], []
    monkeypatch.setattr(round_driver.C, "write_launch_record",
                        lambda results, index, chunk, repo: recorded.append(chunk["id"]))
    monkeypatch.setattr(round_driver, "run_lanes",
                        lambda home, args, plan, results, subs, prefix: ran.extend(plan) or {})
    python, _corpus = _environment(tmp_path)
    argv = ["gates", "--repo", str(REPO), "--results", str(tmp_path / "results"),
            "--capability", capability, "--home", str(tmp_path), "--lock-dir", str(tmp_path),
            "--python", python, "--chunk-index", str(path), "--chunk", "c1"]
    args = _parser().parse_args(argv + (["--new-capability"] if mode == "new" else []))
    assert round_driver.apply_chunk(args) is None
    assert round_driver.stage_gates(args, tmp_path) == 0
    assert recorded == ["c1"] and [step for steps in ran for step in steps]
    assert f"launch record chunk_{args.stamp}/LAUNCH.json (chunk c1, " in (
        tmp_path / "round.log").read_text()
    assert "chunks.py verify" in (tmp_path / "BIND_COMMANDS.txt").read_text()


def test_the_route_stage_files_the_licence_under_its_capability(tmp_path, monkeypatch):
    lanes = []
    monkeypatch.setattr(round_driver.L, "check", lambda repo, capability, route, **kw: (
        [], {"triton_route_campaign": "t_cc90", "cuda_route_campaign": "c_cc90",
             "licence_campaign": "dispatch_end_to_end_s"}))
    monkeypatch.setattr(round_driver.L, "flush_coverage", lambda repo, cap: (
        {"triton": {"arms_covered": 1, "arms": 1}, "cuda": {"arms_covered": 1, "arms": 1}}, []))
    monkeypatch.setattr(round_driver, "run_lanes",
                        lambda home, args, plan, results, subs, prefix: lanes.extend(plan) or {})
    args = _parser().parse_args(["route", "--repo", str(tmp_path), "--results",
                                 str(tmp_path / "res"), "--stamp", "s", "--capability", "9.0",
                                 "--home", str(tmp_path), "--lock-dir", str(tmp_path)])
    assert round_driver.stage_route(args, tmp_path) == 0
    outs = {step["id"]: step.get("out") for steps in lanes for step in steps
            if step["kind"] in ("e2e", "e2e_verdict")}
    assert outs and all(str(tmp_path / "res" / "dispatch_end_to_end_s_cc90") in out
                        for out in outs.values()), outs


# ------------------------------------------------------------------------- the corpus -----
def _environment(tmp_path: Path, record: bool = True, root_files=None) -> tuple:
    """A reference environment whose interpreter reports its prefix, holding a kept corpus
    tree and, unless told otherwise, the corpus record of it."""
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.write_text(f"#!/bin/sh\necho {prefix}\n")
    python.chmod(0o755)
    corpus = prefix / "share" / "meep-gpu-build" / "meep-1.33.0" / "python"
    files = root_files or {"examples/a.py": "A = 1\n", "tests/test_b.py": "B = 1\n"}
    for rel, text in files.items():
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_text(text)
    sibling = corpus.parent / SIBLING
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_bytes(b"eps")
    if record:
        directory = prefix.joinpath(*L.BUILD_RECORD_DIR)
        listing = directory / "corpus_sha256.txt"
        listing.write_text("".join(f"{L.sha256_of(corpus / rel)}  {rel}\n" for rel in files))
        (directory / L.BUILD_RECORD).write_text(json.dumps({
            "corpus": {"root": "share/meep-gpu-build/meep-1.33.0/python",
                       "list": listing.name, "list_sha256": L.sha256_of(listing),
                       "files": len(files),
                       "siblings": [{"path": SIBLING, "sha256": L.sha256_of(sibling)}]},
            "meep": {"tarball_sha256": "0" * 64}}))
    return str(python), corpus


#: The corpus sibling the reference build keeps beside python/ (build_meep_133_record.py).
SIBLING = "tests/cyl-ellipsoid-eps-ref.h5"


def test_the_corpus_is_the_root_the_reference_environments_record_names(tmp_path):
    python, corpus = _environment(tmp_path)
    found = L.resolve_corpus(python, "", {})
    assert found["root"] == str(corpus.resolve()) and found["problems"] == []
    assert found["record"].endswith("share/meep-gpu-build/reference_build.json")
    assert "2 of 2 recorded files match" in found["notes"][1]
    # --corpus-root names the tree; the record still judges it, and the sibling beside it.
    copy = tmp_path / "copy" / "python"
    for rel in ("examples/a.py", "tests/test_b.py", f"../{SIBLING}"):
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        (copy / rel).write_bytes((corpus / rel).read_bytes())
    assert L.resolve_corpus(python, str(copy), {})["problems"] == []
    (copy / "examples" / "a.py").write_text("A = 2\n")
    (copy / "tests" / "planted.py").write_text("")
    problems = L.resolve_corpus(python, str(copy), {})["problems"]
    assert any("changed (sha256): examples/a.py" in p for p in problems), problems
    assert any("unlisted and importable: tests/planted.py" in p for p in problems), problems



def test_an_unlisted_file_python_could_import_is_refused_and_a_cache_is_not(tmp_path):
    """Source, byte code outside ``__pycache__`` (imported without a source) and an extension
    module could each shadow an import of a corpus row; a byte-code cache of a listed source
    and unlisted data could not."""
    python, corpus = _environment(tmp_path)
    (corpus / "examples" / "__pycache__").mkdir()
    (corpus / "examples" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")
    (corpus / "examples" / "out.png").write_bytes(b"x")
    found = L.resolve_corpus(python, "", {})
    assert found["problems"] == [] and "examples/out.png" in found["notes"][-1], found
    extension = f"tests/_shadow{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    (corpus / "tests" / "meep.py").write_text("")
    (corpus / "tests" / "gdspy.pyc").write_bytes(b"x")
    (corpus / extension).write_bytes(b"x")
    found = L.resolve_corpus(python, "", {})
    importable = ", ".join(sorted([extension, "tests/gdspy.pyc", "tests/meep.py"]))
    assert found["problems"] == [
        f"3 file(s) under {corpus.resolve()} unlisted and importable: {importable}"], found

def test_a_corpus_list_the_record_does_not_pin_is_refused(tmp_path):
    python, corpus = _environment(tmp_path)
    listing = Path(python).parents[1].joinpath(*L.BUILD_RECORD_DIR, "corpus_sha256.txt")
    listing.write_text(listing.read_text() + f"{'0' * 64}  examples/planted.py\n")
    found = L.resolve_corpus(python, "", {})
    assert "is not the one" in found["problems"][0] and found["root"] is None


def test_a_reference_environment_with_no_build_record_is_refused_by_name(tmp_path):
    python, _corpus = _environment(tmp_path, record=False)
    found = L.resolve_corpus(python, "", {})
    assert found["root"] is None and "no reference build record at" in found["problems"][0]
    lift = {"id": "residue/x/keep", "kind": "gate", "args": ["--lift-steps", "60"]}
    (problem,) = L.corpus_problems([lift], found["root"], found["problems"])
    assert "does not verify: no reference build record at" in problem
    plain = {"id": "cited/x/keep", "kind": "gate", "args": []}
    assert L.corpus_problems([plain], found["root"], found["problems"]) == []


def test_the_gate_stage_hands_every_lane_the_corpus_it_verified(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(round_driver.L, "check", lambda repo, capability, route, **kw: (
        seen.update(kw) or ["stop here"], {}))
    python, corpus = _environment(tmp_path)
    args = _parser().parse_args(["gates", "--repo", str(tmp_path), "--stamp", "s",
                                 "--capability", "8.6", "--home", str(tmp_path), "--lock-dir",
                                 str(tmp_path), "--python", python])
    assert round_driver.stage_gates(args, tmp_path) == 3
    assert seen["corpus"] == str(corpus.resolve()) == args.corpus_root
    argv = round_driver.lane_argv(args, tmp_path / "plan.json", tmp_path / "lane",
                                  tmp_path / "results", {})
    assert argv[argv.index("--corpus-root") + 1] == str(corpus.resolve())
    assert f"MEEP corpus: {corpus.resolve()}" in (tmp_path / "round.log").read_text()
