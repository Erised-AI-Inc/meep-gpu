"""The Metal round's side of the lift legs: the per-cell preflight, the MEEP corpus, and
the exit code of ``all``.

``rounds/1.0/metal_round.py`` is in no gate's closure, so everything here moves no gate.
Host-only checks (no MEEP, no GPU):

* which fleet gates launch lift children, read off the source: exactly nine on this tree,
  and the one that reads their four inputs in process (the complex gate);
* the per-cell preflight child, run for real against synthetic gates and stand-in
  ``meep``/``gdspy`` modules: rows per gate, an absent input named, a gate whose basis
  cannot be computed named, a row the host cannot lift reported and not stopped for,
  an in-process reader's inputs checked;
* ``metal_round.main`` driven through ``preflight`` with its probes answered here: it
  stops naming an absent lift input or a corpus file that does not match its record,
  and goes on past rows that are only unlifted on this host;
* the corpus: named by the reference build record and its sha256 list, verified file by
  file, exported to the fleet;
* ``all`` exits with the status stage's code.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROUND = HERE / "rounds" / "1.0"
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

import metal_round  # noqa: E402

GDSII = "corpus_final_numpy_2026-08-09_monitorfix_gdsii"
STOCK = "corpus_final_numpy_2026-08-09_monitorfix"
#: The gates of this tree that launch lift children (six with their own ``lift_basis``,
#: three on ``hd_tail_gate_common``); the complex gates name a census but lift in process.
LIFTING = {
    "gate_metal_bfast_fused_hd_pair": "direct",
    "gate_metal_conductive_fused_hd_pair": "direct",
    "gate_metal_cylindrical_real_fused_hd_pair": "direct",
    "gate_metal_folded_fused_hd_pair": "direct",
    "gate_metal_fused_hd_pair": "direct",
    "gate_metal_fused_hd_pair_nonlinear": "direct",
    "gate_metal_beta_complex_fused_hd_pair": "tail",
    "gate_metal_beta_real_fused_hd_pair": "tail",
    "gate_metal_folded_complex_fused_hd_pair": "tail",
}


def test_nine_gates_launch_lift_children_and_the_complex_gate_is_not_one():
    kinds = {p.stem: metal_round.lifting_kind(p.read_text(encoding="utf-8"))
             for p in sorted(HERE.glob("gate_metal_*.py"))}
    assert {stem: kind for stem, kind in kinds.items() if kind} == LIFTING
    assert kinds["gate_metal_complex_fused_hd_pair"] is None
    # The complex gate reads the same four inputs in process: the preflight checks them.
    readers = sorted(p.stem for p in HERE.glob("gate_metal_*.py")
                     if metal_round.census_reader(p.read_text(encoding="utf-8")))
    assert readers == ["gate_metal_complex_fused_hd_pair"]
    # Each names its inputs at module level, so the preflight can check them unimported.
    for stem in LIFTING:
        assert metal_round.census_names((HERE / f"{stem}.py").read_text(encoding="utf-8")) \
            == ("metal_coverage_2026-09-04_m0complex", "h_to_d_seam_2026-09-04"), stem


# ------------------------------------------------------------ a synthetic checkout ----
_DIRECT = '''
CENSUS = "census_x"
SEAM_RECORD = "seam_x"


def lift_basis(results):
    import json
    import lift_host
    missing = lift_host.missing_inputs(results, CENSUS, SEAM_RECORD)
    if missing:
        raise SystemExit(lift_host.missing_reason(results, missing))
    seam = {row["label"]: row for row in map(json.loads, (results / SEAM_RECORD
            / "h_to_d_seam.jsonl").read_text().splitlines())}
    rows = []
    for line in (results / CENSUS / "examples.jsonl").read_text().splitlines():
        record = json.loads(line)
        label = "examples:" + record["row"]
        rows.append({"label": label, **lift_host.row_fields(record, seam.get(label, {}))})
    return rows, {}
'''
_TAIL_KIT = '''
class Product:
    def __init__(self, census, seam_record):
        self.census, self.seam_record = census, seam_record


def lift_basis(product, results):
    return [{"label": "tests:T.t", "lift_needs": []}], {}
'''
_TAIL = '''
import hd_tail_gate_common as common

PRODUCT = common.Product("census_x", "seam_x")
'''


def _checkout(tmp_path: Path, libgdsii: bool = False) -> SimpleNamespace:
    """A checkout with a direct gate (two rows, one from the gdsii lift record), a gate on
    a stand-in tail kit, a gate that names a census and lifts nothing, a gate whose
    import needs torch, a lifting gate that names no census, the real ``lift_host``,
    stand-in ``meep`` and ``gdspy`` at the checkout root, and the records the gates
    read."""
    repo = tmp_path / "repo"
    harness = repo / "parity" / "meep_gpu"
    harness.mkdir(parents=True)
    shutil.copy(HERE / "lift_host.py", harness / "lift_host.py")
    (harness / "gate_metal_direct.py").write_text(_DIRECT, encoding="utf-8")
    (harness / "hd_tail_gate_common.py").write_text(_TAIL_KIT, encoding="utf-8")
    (harness / "gate_metal_tail.py").write_text(_TAIL, encoding="utf-8")
    (harness / "gate_metal_names_a_census.py").write_text(
        'CENSUS = "census_x"\nSEAM_RECORD = "seam_x"\n', encoding="utf-8")
    (harness / "gate_metal_torchy.py").write_text(
        'CENSUS = "census_x"\nSEAM_RECORD = "seam_x"\nimport torch\n\n\n'
        "def lift_basis(results):\n    return [], {}\n", encoding="utf-8")
    (harness / "gate_metal_unnamed.py").write_text(
        "def lift_basis(results):\n    return [], {}\n", encoding="utf-8")
    (repo / "meep.py").write_text(f"def with_libGDSII():\n    return {libgdsii!r}\n",
                                  encoding="utf-8")
    (repo / "gdspy.py").write_text("", encoding="utf-8")
    results = tmp_path / "results"
    for rel, rows in (("census_x/examples.jsonl",
                       [{"row": "plain.py", "lift_record": STOCK, "interpreter": "/x/py"},
                        {"row": "coupler.py", "lift_record": GDSII, "interpreter": "/y/py"}]),
                      ("census_x/tests.jsonl", []), ("census_x/tests_param_matched.jsonl", []),
                      ("seam_x/h_to_d_seam.jsonl", [])):
        (results / rel).parent.mkdir(parents=True, exist_ok=True)
        (results / rel).write_text("".join(json.dumps(r) + "\n" for r in rows),
                                   encoding="utf-8")
    return SimpleNamespace(repo=repo, harness=harness, results=results)


def _round(tmp_path: Path, checkout: SimpleNamespace, **extra) -> "metal_round.Round":
    args = SimpleNamespace(repo=str(checkout.repo), stamp="t", results=str(checkout.results),
                           logs=str(tmp_path / "logs"), python=sys.executable, expect=None,
                           meep_source="", only="", corpus_root="", **extra)
    return metal_round.Round(args)


def test_the_cell_preflight_runs_each_lifting_gates_own_basis(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path)
    monkeypatch.setattr(metal_round, "resolve_corpus", lambda *a, **k: {
        "root": None, "record": None, "problems": [], "notes": []})
    rnd = _round(tmp_path, checkout)
    problems, notes = rnd.lift_preflight()
    # A lifting gate whose inputs cannot be named stops the preflight; one whose basis
    # the child cannot compute (its import needs torch, unimportable there) while its
    # four inputs are present is a warning, which its own lift leg reports.
    assert len(problems) == 1, problems
    assert problems[0].startswith("gate_metal_unnamed: its lift inputs could not be named")
    [warning] = [n for n in notes if n.startswith("WARNING gate_metal_torchy:")]
    assert "its four inputs are present" in warning and "ImportError" in warning, warning
    assert "gate_metal_direct: 2 lift rows, 1 of them unlifted on this host" in notes
    assert "gate_metal_tail: 1 lift rows, 0 of them unlifted on this host" in notes
    [unlifted] = [n for n in notes if "UNLIFTED ON THIS HOST" in n]
    assert unlifted.startswith("gate_metal_direct: examples:coupler.py: UNLIFTED ON THIS HOST")
    assert "meep.with_libGDSII() is False" in unlifted
    # A gate that names the census and lifts in process has its inputs checked, and no
    # basis computed.
    assert ("gate_metal_names_a_census: its four lift inputs are present (it reads them "
            "in process and launches no lift child)") in notes
    # The four lifting gates are counted; the reader is not one of them.
    assert notes[-1].startswith("3 lift rows over 4 gates;"), notes[-1]
    # Respecting --only: the gate set is the fleet's.
    rnd.args.only = "gate_metal_direct"
    problems, notes = rnd.lift_preflight()
    assert problems == [] and notes[-1].startswith("2 lift rows over 1 gates; 1 of 2")
    # An absent input is named, by gate and file.
    (checkout.results / "census_x" / "tests_param_matched.jsonl").unlink()
    problems, _ = rnd.lift_preflight()
    assert problems == [f"gate_metal_direct: lift input(s) absent under {checkout.results}: "
                        f"census_x/tests_param_matched.jsonl"]
    # So is one of the in-process reader's, alone in its fleet.
    rnd.args.only = "gate_metal_names_a_census"
    problems, _ = rnd.lift_preflight()
    assert problems == [f"gate_metal_names_a_census: lift input(s) absent under "
                        f"{checkout.results}: census_x/tests_param_matched.jsonl"]


def test_a_host_with_every_need_lifts_every_row(tmp_path, monkeypatch):
    checkout = _checkout(tmp_path, libgdsii=True)
    monkeypatch.setattr(metal_round, "resolve_corpus", lambda *a, **k: {
        "root": None, "record": None, "problems": [], "notes": []})
    rnd = _round(tmp_path, checkout)
    rnd.args.only = "gate_metal_direct,gate_metal_tail"
    problems, notes = rnd.lift_preflight()
    assert problems == [] and not [n for n in notes if "UNLIFTED" in n], notes
    assert notes[-1].startswith("3 lift rows over 2 gates; 0 of 3 unlifted")


# ------------------------------------------------------------------- the corpus ----
def _corpus(tmp_path: Path, prefix: Path) -> Path:
    """A kept corpus under ``prefix``, its sha256 list and the reference build record that
    names both, in the build scripts' layout; returns a stand-in interpreter that reports
    ``prefix`` as its ``sys.prefix``."""
    record_dir = prefix / "share" / "meep-gpu-build"
    root = record_dir / "meep-1.33.0" / "python"
    files = {"examples/a.py": b"print(1)\n", "tests/test_b.py": b"# b\n",
             "examples/data.h5": b"\x00\x01"}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    listing = "".join(f"{hashlib.sha256(data).hexdigest()}  {rel}\n"
                      for rel, data in sorted(files.items()))
    (record_dir / "corpus_sha256.txt").write_text(listing, encoding="utf-8")
    sibling = root.parent / "tests" / "cyl-ellipsoid-eps-ref.h5"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"eps")
    (record_dir / "reference_build.json").write_text(json.dumps({
        "schema": "meep-gpu/reference-build/1",
        "meep": {"version": "1.33.0", "tarball_sha256": "0" * 64},
        "corpus": {"root": "share/meep-gpu-build/meep-1.33.0/python",
                   "list": "corpus_sha256.txt",
                   "list_sha256": hashlib.sha256(listing.encode()).hexdigest(),
                   "files": len(files),
                   "siblings": [{"path": "tests/cyl-ellipsoid-eps-ref.h5",
                                 "sha256": hashlib.sha256(b"eps").hexdigest()}]}}),
        encoding="utf-8")
    python = tmp_path / "python"
    python.write_text(f"#!/bin/sh\necho {prefix}\n", encoding="utf-8")
    python.chmod(0o755)
    return python


def test_the_corpus_comes_from_the_build_record_and_is_verified(tmp_path):
    prefix = tmp_path / "prefix"
    python = _corpus(tmp_path, prefix)
    root = prefix / "share" / "meep-gpu-build" / "meep-1.33.0" / "python"
    found = metal_round.resolve_corpus(str(python), "", os.environ)
    assert found["problems"] == [] and found["root"] == str(root.resolve()), found
    assert found["notes"][-1] == (f"3 of 3 recorded files match under {root.resolve()}; 1 of 1 "
                                  f"recorded sibling(s) match beside it "
                                  f"(tests/cyl-ellipsoid-eps-ref.h5)")
    # Byte-code caches and unlisted data are not refused; an unlisted file Python could
    # import is: source, byte code outside __pycache__ (imported without a source), and an
    # extension module.
    (root / "examples" / "__pycache__").mkdir()
    (root / "examples" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")
    (root / "examples" / "out.png").write_bytes(b"x")
    found = metal_round.resolve_corpus(str(python), "", os.environ)
    assert found["problems"] == [] and "examples/out.png" in found["notes"][-1]
    extension = f"tests/_shadow{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    (root / "tests" / "meep.py").write_text("", encoding="utf-8")
    (root / "tests" / "gdspy.pyc").write_bytes(b"x")
    (root / extension).write_bytes(b"x")
    (root / "examples" / "a.py").write_text("print(2)\n", encoding="utf-8")
    (root / "tests" / "test_b.py").unlink()
    found = metal_round.resolve_corpus(str(python), "", os.environ)
    importable = ", ".join(sorted([extension, "tests/gdspy.pyc", "tests/meep.py"]))
    assert found["problems"] == [
        f"1 file(s) under {root.resolve()} absent: tests/test_b.py",
        f"1 file(s) under {root.resolve()} changed (sha256): examples/a.py",
        f"3 file(s) under {root.resolve()} unlisted and importable: {importable}"]
    # --corpus-root names another copy, verified against the same record, siblings beside it
    # included.
    shutil.copytree(root.parent, tmp_path / "copy")
    copy = tmp_path / "copy" / "python"
    found = metal_round.resolve_corpus(str(python), str(copy), os.environ)
    assert found["root"] == str(copy.resolve()) and len(found["problems"]) == 3
    (copy.parent / "tests" / "cyl-ellipsoid-eps-ref.h5").unlink()
    found = metal_round.resolve_corpus(str(python), str(copy), os.environ)
    assert found["problems"][-1] == (f"the corpus sibling tests/cyl-ellipsoid-eps-ref.h5 is "
                                     f"absent from {copy.parent.resolve()}")


def test_a_missing_or_altered_record_is_named(tmp_path):
    prefix = tmp_path / "prefix"
    python = _corpus(tmp_path, prefix)
    record_dir = prefix / "share" / "meep-gpu-build"
    record, listing = record_dir / "reference_build.json", record_dir / "corpus_sha256.txt"
    # A list that is not the one the record pinned is refused before any file is read.
    listing.write_text(listing.read_text(encoding="utf-8") + f"{'0' * 64}  examples/x.py\n",
                       encoding="utf-8")
    found = metal_round.resolve_corpus(str(python), "", os.environ)
    assert found["root"] is None and found["problems"] == [
        f"the corpus list {listing} is not the one {record} recorded (sha256 differs)"]
    document = json.loads(record.read_text(encoding="utf-8"))
    del document["corpus"]
    record.write_text(json.dumps(document), encoding="utf-8")
    found = metal_round.resolve_corpus(str(python), "", os.environ)
    assert found["root"] is None and "names no corpus" in found["problems"][0]
    record.unlink()
    found = metal_round.resolve_corpus(str(python), str(tmp_path), os.environ)
    assert found["root"] == str(tmp_path.resolve()), "an explicit root is still exported"
    assert found["problems"] == [f"no reference build record at {record}: the reference "
                                 f"environment's build writes it "
                                 f"(environments/locks/README.md)"]


def test_the_fleet_exports_the_rounds_corpus_never_the_shells(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    python = _corpus(tmp_path, prefix)
    monkeypatch.setenv("MEEP_GPU_CORPUS_ROOT", "/inherited/corpus")
    args = SimpleNamespace(repo=str(tmp_path), stamp="t", results=str(tmp_path / "results"),
                           logs=str(tmp_path / "logs"), python=str(python), expect=None,
                           meep_source="", corpus_root="")
    env = metal_round.Round(args).fleet_env()
    assert env["MEEP_GPU_CORPUS_ROOT"] == str(
        (prefix / "share" / "meep-gpu-build" / "meep-1.33.0" / "python").resolve())
    assert "MEEP_GPU_CORPUS_ROOT" in metal_round.SCRUBBED


# ------------------------------------------------------- the stages, through main ----
def _drive(tmp_path, monkeypatch, cells, python):
    """``metal_round.main`` on a checkout with one lifting gate, every probe answered here;
    ``cells`` is the per-cell preflight child's answer."""
    repo = tmp_path / "repo"
    harness = repo / "parity" / "meep_gpu"
    harness.mkdir(parents=True, exist_ok=True)
    (harness / "gate_metal_lifting.py").write_text(_DIRECT, encoding="utf-8")
    real_run = subprocess.run

    def answered(command, *args, **kwargs):
        text = " ".join(map(str, command))
        if "print(sys.prefix)" in text:
            return real_run(command, *args, **kwargs)     # the stand-in's prefix
        out = (f"False\n{repo.resolve()}/meep_gpu/__init__.py\n" if "ENABLE_USER_SITE" in text else
               '{"arch": "applegpu_test", "admitted": ["applegpu_test"]}\n'
               if metal_round.ADMISSION_PROGRAM in command else
               "applegpu_test 2.0 Apple\n" if "metal_environment" in text else
               metal_round.LIFT_CELLS_PREFIX + json.dumps(cells) + "\n"
               if metal_round.LIFT_CELLS_PROGRAM in command else "")
        return subprocess.CompletedProcess(command, 0, out, "")

    monkeypatch.setattr(metal_round.subprocess, "run", answered)
    monkeypatch.setattr(metal_round.signal, "signal", lambda *args: None)
    for name in [k for k in os.environ if k in metal_round.SCRUBBED]:
        monkeypatch.delenv(name)
    monkeypatch.delenv("PYTORCH_MPS_FAST_MATH", raising=False)

    def preflight(name):
        logs = tmp_path / name
        rc = metal_round.main(["preflight", "--repo", str(repo), "--stamp", "t",
                               "--results", str(tmp_path / "results"), "--logs", str(logs),
                               "--python", str(python), "--disk-floor-kb", "0"])
        return rc, (logs / "round.log").read_text(encoding="utf-8")

    return preflight


def _cells(**entry):
    return {"gates": {"gate_metal_lifting": {"kind": "direct", **entry}},
            "host": {"interpreter": "/env/bin/python", "unreadable": None,
                     "needs": {"gdspy": None, "libGDSII": "meep.with_libGDSII() is False"}}}


def test_the_preflight_goes_on_past_unlifted_rows_and_stops_for_an_absent_input(
        tmp_path, monkeypatch):
    python = _corpus(tmp_path, tmp_path / "prefix")
    unlifted = _cells(rows=49, needs={"examples:coupler.py": ["gdspy", "libGDSII"]},
                      unlifted={"examples:coupler.py": "needs gdspy, libGDSII"},
                      missing=[])
    rc, log = _drive(tmp_path, monkeypatch, unlifted, python)("unlifted")
    assert rc == 0, log
    assert "lift: gate_metal_lifting: 49 lift rows, 1 of them unlifted on this host" in log
    assert ("lift: gate_metal_lifting: examples:coupler.py: UNLIFTED ON THIS HOST, "
            "needs gdspy, libGDSII") in log
    assert "MEEP corpus: " in log and "corpus: 3 of 3 recorded files match" in log
    absent = _cells(missing=["census_x/tests_param_matched.jsonl"])
    rc, log = _drive(tmp_path, monkeypatch, absent, python)("absent")
    assert rc == 3, log
    assert (f"FAIL lift: gate_metal_lifting: lift input(s) absent under "
            f"{tmp_path / 'results'}: census_x/tests_param_matched.jsonl") in log


def test_the_preflight_stops_for_a_corpus_that_does_not_match_its_record(
        tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    python = _corpus(tmp_path, prefix)
    (prefix / "share" / "meep-gpu-build" / "meep-1.33.0" / "python" / "examples"
     / "a.py").write_text(
        "print(3)\n", encoding="utf-8")
    rc, log = _drive(tmp_path, monkeypatch, _cells(rows=1, missing=[]), python)("corpus")
    assert rc == 3 and "FAIL corpus: 1 file(s) under " in log and "examples/a.py" in log, log


@pytest.mark.parametrize("status_rc", [0, 1])
def test_all_exits_with_the_status_stages_code(tmp_path, monkeypatch, status_rc):
    calls = []
    for stage, rc in (("preflight", 0), ("fleet", 0), ("status", status_rc)):
        monkeypatch.setattr(metal_round.Round, f"stage_{stage}",
                            lambda self, stage=stage, rc=rc: calls.append(stage) or rc)
    monkeypatch.setattr(metal_round.signal, "signal", lambda *args: None)
    rc = metal_round.main(["all", "--repo", str(tmp_path), "--stamp", "t",
                           "--results", str(tmp_path / "results"),
                           "--logs", str(tmp_path / f"logs{status_rc}")])
    assert calls == ["preflight", "fleet", "status"] and rc == status_rc
