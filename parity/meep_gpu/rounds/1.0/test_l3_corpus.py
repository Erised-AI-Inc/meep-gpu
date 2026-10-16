"""Host-only tests of the L3 corpus check (``l3_corpus.py``, ``l3_row.py``): no GPU, no MEEP.

* the arguments: each table's reference sets, primary, image set and environment, and the
  combinations refused by name before any row runs;
* the per-row judge, on synthetic words and on a Tier A fixture stepped by the NumPy array
  path, with a planted one-ulp flip that must be caught at its step and component;
* the leg's output schema, its verdict and exit status, and its resume, with a stub row
  program standing in for the lift.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l3_corpus  # noqa: E402
import l3_row  # noqa: E402
import legs  # noqa: E402

from meep_gpu import meep_identity as identity  # noqa: E402


# ---------------------------------------------------------------------------------------
# Arguments and environment
# ---------------------------------------------------------------------------------------

def _resolved(*argv: str):
    return l3_corpus.resolve(l3_corpus.parser().parse_args(
        ["--record", "/nonexistent/record.json", "--out", "/nonexistent/out", *argv]))


def test_each_table_judges_its_own_reference_sets():
    triton = _resolved("--table", "triton", "--baseline", "keep")
    assert (triton.set_list, triton.primary, triton.image_set) == (["linux_keep"], "linux_keep",
                                                                   "linux_keep")
    assert triton.cross_mode == set() and triton.leg == "triton_keep"
    cuda = _resolved("--table", "cuda", "--baseline", "flush")
    assert (cuda.set_list, cuda.primary, cuda.leg) == (["linux_flush"], "linux_flush", "cuda_flush")
    metal = _resolved("--table", "metal", "--baseline", "flush")
    assert metal.set_list == ["linux_flush", "mac_flush", "linux_keep"]
    assert (metal.primary, metal.image_set, metal.cross_mode) == ("linux_flush", "mac_flush",
                                                                  {"linux_keep"})
    shipped = _resolved("--table", "triton", "--baseline", "keep", "--shipped-default")
    assert shipped.leg == "triton_keep_shipped"


@pytest.mark.parametrize("argv, words", [
    (("--table", "metal", "--baseline", "keep"), "flush only"),
    (("--table", "triton", "--baseline", "keep", "--sets", "linux_keep", "--primary", "mac_keep"),
     "is not among the sets"),
    (("--table", "metal", "--baseline", "flush", "--primary", "linux_keep"),
     "cannot be a cross-mode set"),
    (("--table", "triton", "--baseline", "keep", "--expect-differ", "x"), "go together"),
])
def test_impossible_combinations_are_refused_by_name(argv, words):
    with pytest.raises(l3_corpus.Refusal, match=words):
        _resolved(*argv)


def test_the_tables_are_the_three_the_row_program_knows():
    assert set(l3_corpus.TABLES) == {"triton", "cuda", "metal"}
    with pytest.raises(SystemExit):
        l3_corpus.parser().parse_args(["--table", "rocm", "--baseline", "keep",
                                       "--record", "r", "--out", "o"])


def test_the_leg_environment_scrubs_the_shell_and_sets_the_table(tmp_path):
    shell = {"HOME": "/h/u", "USER": "u", "PATH": "/p", "MEEP_GPU_DISPATCH": "0",
             "MEEP_GPU_HARDWARE": "any", "MEEP_GPU_ALLOW_UNCERTIFIED": "1",
             "TRIDENT_FDTD_X": "1", "PYTHONPATH": "/elsewhere", "CUPY_CACHE_DIR": "/c",
             "CUPY_ACCELERATORS": "cub", "OMP_NUM_THREADS": "8", "CC": "cc"}
    out = tmp_path / "leg"
    args = _resolved("--table", "cuda", "--baseline", "keep", "--out", str(out), "--gpu", "3",
                     "--meep-site", "/site", "--repo", str(tmp_path), "--libcuda-stub", "/stub")
    env = l3_corpus.leg_environment(args, shell)
    assert {k for k in env if k.startswith("MEEP_GPU_")} == {"MEEP_GPU_BACKEND_PREFERENCE"}
    assert env["MEEP_GPU_BACKEND_PREFERENCE"] == "cuda"
    assert "TRIDENT_FDTD_X" not in env and env["CC"] == "cc" and env["PATH"] == "/p"
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["CUPY_ACCELERATORS"] == ""
    assert env["PYTHONPATH"] == f"/site:{tmp_path.resolve()}"
    assert all(env[name] == "1" for name in l3_corpus.THREAD_VARIABLES)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "ftz_stripped" in env["CUPY_CACHE_DIR"]
    assert env["CUPY_CACHE_DIR"].startswith(str(out.resolve()))
    assert env["TRITON_LIBCUDA_PATH"] == "/stub" and env["LD_LIBRARY_PATH"].startswith("/stub")
    triton = l3_corpus.leg_environment(_resolved("--table", "triton", "--baseline", "flush",
                                                 "--out", str(out)), shell)
    assert not any(k.startswith("MEEP_GPU_") for k in triton)
    assert "ftz_stripped" not in triton["CUPY_CACHE_DIR"]
    shipped = l3_corpus.leg_environment(_resolved("--table", "triton", "--baseline", "keep",
                                                  "--out", str(out), "--shipped-default"), shell)
    assert "CUPY_ACCELERATORS" not in shipped and "ftz_stripped" not in shipped["CUPY_CACHE_DIR"]
    metal = l3_corpus.leg_environment(_resolved("--table", "metal", "--baseline", "flush",
                                                "--out", str(out), "--clean-env",
                                                "--python", "/env/bin/python"), shell)
    assert metal["PATH"] == "/env/bin:/usr/bin:/bin" and metal["LANG"] == "C"
    assert "CC" not in metal and "CUPY_CACHE_DIR" not in metal
    assert not any(k.startswith("MEEP_GPU_") for k in metal)


# ---------------------------------------------------------------------------------------
# The judge, on synthetic words
# ---------------------------------------------------------------------------------------

SHAPE = (3, 2, 1)


def _fields(step: int):
    rng = numpy.random.default_rng(1000 + step)
    return {name: rng.standard_normal(SHAPE).astype(numpy.float32) for name in identity.COMPONENTS}


def _found(step: int, fields, masks):
    words = {name: identity.compared_words(fields[name], masks[name]) for name in identity.COMPONENTS}
    stored = {name: identity.digest_words(words[name]) for name in identity.COMPONENTS}
    reader = {name: identity.digest_words(identity.reader_words(fields[name]))
              for name in identity.COMPONENTS}
    planes = numpy.stack([identity.plane_digests(fields[name], masks[name])
                          for name in identity.COMPONENTS])
    return identity.StepDigests(step=step, stored=stored, reader=reader, planes=planes, words=words)


def _synthetic_record(steps, masks, flip_word=4, flip_step=1, flip_component="Ez"):
    found = {step: _found(step, _fields(step), masks) for step in steps}
    plane = identity.word_cell(masks[flip_component], 1, flip_word)[0][0]
    document = {"stored": {str(s): found[s].stored for s in steps},
                "reader": {str(s): found[s].reader for s in steps},
                "planes": {"steps": list(steps)},
                "planted_flip": {"step": flip_step, "component": flip_component,
                                 "word": flip_word, "plane": plane}}
    planes = numpy.stack([found[s].planes for s in steps])
    return document, planes


def _judge_synthetic(document, planes, steps, masks, flip_at=None):
    tally, flip = l3_row.new_tally(), l3_row.new_flip(document)
    for step in steps:
        fields = _fields(step)
        found = _found(step, fields, masks)
        if flip_at is not None and step == flip_at[0]:
            name, index = flip_at[1], flip_at[2]
            planted = dict(fields)
            view = planted[name].copy().view(numpy.uint32)
            cell, _part = identity.word_cell(masks[name], 1, index)
            view[cell] ^= numpy.uint32(1)
            planted[name] = view.view(numpy.float32)
            words = dict(found.words)
            words[name] = identity.compared_words(planted[name], masks[name])
            stored = dict(found.stored)
            stored[name] = identity.digest_words(words[name])
            plane_rows = found.planes.copy()
            plane_rows[identity.COMPONENTS.index(name)] = identity.plane_digests(planted[name],
                                                                                 masks[name])
            found = identity.StepDigests(step=step, stored=stored, reader=found.reader,
                                         planes=plane_rows, words=words)
        l3_row.compare_step(document, planes, list(steps), found, step, tally, flip,
                            lambda comp, word, _f=fields: identity.flipped_planes(_f[comp],
                                                                                  masks[comp], word))
    return {"per_set": {"ref": dict(tally, planted_flip=flip)}, "fpu_checks": len(steps),
            "compared_steps": len(steps)}


def test_the_judge_holds_equal_words_and_catches_a_one_ulp_flip():
    masks = {name: numpy.ones(SHAPE, dtype=bool) for name in identity.COMPONENTS}
    steps = [0, 1, 2]
    document, planes = _synthetic_record(steps, masks)
    clean = _judge_synthetic(document, planes, steps, masks)
    tally = clean["per_set"]["ref"]
    assert (tally["digests"], tally["stored_equal"], tally["reader_equal"]) == (36, 36, 36)
    assert tally["planted_flip"]["caught"] and tally["planted_flip"]["placed"]
    assert tally["first"] is None and l3_row.set_verdict(clean, "ref") == "holds"

    # One stored word of Hy, one ulp, at step 2: the stored digest of that one component
    # at that one step differs, the reader words were not touched, and the plane is named.
    planted = _judge_synthetic(document, planes, steps, masks, flip_at=(2, "Hy", 3))
    tally = planted["per_set"]["ref"]
    assert (tally["digests"], tally["stored_equal"], tally["reader_equal"]) == (36, 35, 36)
    plane = identity.word_cell(masks["Hy"], 1, 3)[0][0]
    assert tally["first"] == {"step": 2, "components": {
        "Hy": {"stored_equal": False, "reader_equal": True, "planes_differing": [plane]}}}
    assert tally["differing_count"] == 1
    assert l3_row.set_verdict(planted, "ref") == "differs"


def test_a_comparison_not_shown_able_to_fail_is_not_a_hold():
    masks = {name: numpy.ones(SHAPE, dtype=bool) for name in identity.COMPONENTS}
    steps = [0, 1, 2]
    document, planes = _synthetic_record(steps, masks)
    document["planted_flip"]["plane"] = (document["planted_flip"]["plane"] + 1) % SHAPE[0]
    judged = _judge_synthetic(document, planes, steps, masks)
    flip = judged["per_set"]["ref"]["planted_flip"]
    assert flip["caught"] and not flip["placed"]
    assert l3_row.set_verdict(judged, "ref") == "differs"
    # A compared step whose FPU was not measured is not a hold either.
    document, planes = _synthetic_record(steps, masks)
    judged = _judge_synthetic(document, planes, steps, masks)
    judged["fpu_checks"] = 2
    assert l3_row.set_verdict(judged, "ref") == "differs"


# ---------------------------------------------------------------------------------------
# The judge, on a Tier A fixture stepped by the NumPy array path
# ---------------------------------------------------------------------------------------

FIXTURE_CASE = "ex.material-dispersion"


class _FlipOneStoredWord:
    """A driver whose stored words of one component carry a one-ulp flip at one step."""

    def __init__(self, driver, component, step, index, mask):
        self._driver, self._component, self._step = driver, component, step
        self._index, self._mask = index, mask

    def __getattr__(self, name):
        return getattr(self._driver, name)

    def get_field(self, name, cell_centered=False):
        array = self._driver.get_field(name, cell_centered=cell_centered)
        if name != self._component or self._driver.step_count != self._step:
            return array
        copy = numpy.array(array, copy=True)
        parts = 2 if copy.dtype == numpy.complex64 else 1
        words = (copy.view(numpy.float32).reshape(copy.shape + (2,)) if parts == 2
                 else copy.reshape(copy.shape + (1,))).view(numpy.uint32)
        cell, part = identity.word_cell(self._mask, parts, self._index)
        words[cell + (part,)] ^= numpy.uint32(1)
        return copy


def _fixture_judge(planted=None):
    from meep_gpu import capsule

    root = identity.fixtures_root()
    files = identity.read_ledger(root)["sets"]["keep"]["cases"][FIXTURE_CASE]["files"]
    document, planes = identity.read_digests(root / files["digests"]["file"],
                                             root / files["planes"]["file"])
    stored = capsule.read_capsule(root / files["capsule"]["file"])
    currents = capsule.CurrentsTable.read(root / files["currents"]["file"])
    steps = [int(step) for step in document["steps"]]
    with identity.host_subnormal_mode("keep"):
        driver = stored.build_driver(prefer_gpu=False, currents=currents)
        try:
            masks = {name: identity.cell_classes(driver.grid, name)["compared"]
                     for name in identity.COMPONENTS}
            stepped = driver
            if planted is not None:
                component, step = planted
                stepped = _FlipOneStoredWord(driver, component, step, 0, masks[component])
            judged = l3_row.judge(stepped, {"ref": document}, {"ref": planes}, masks, steps,
                                  mode="keep", enforce_fpu=True,
                                  fpu_state=identity.subnormal_state)
        finally:
            driver.close()
    judged["compared_steps"] = len(steps)
    return judged, steps, document


def test_the_judge_holds_on_a_fixture_and_catches_a_flip_planted_in_the_driver():
    judged, steps, document = _fixture_judge()
    tally = judged["per_set"]["ref"]
    digests = len(steps) * len(identity.COMPONENTS)
    assert (tally["digests"], tally["stored_equal"], tally["reader_equal"]) == (digests,) * 3
    assert judged["fpu_checks"] == len(steps) == judged["fpu_steps_in_mode"]
    assert l3_row.set_verdict(judged, "ref") == "holds"

    flip_step = int(document["planted_flip"]["step"])
    target = next(step for step in reversed(steps) if step != flip_step and step > 0)
    judged, _steps, _document = _fixture_judge(planted=("Ez", target))
    tally = judged["per_set"]["ref"]
    assert (tally["stored_equal"], tally["reader_equal"]) == (digests - 1, digests)
    assert tally["first"]["step"] == target and set(tally["first"]["components"]) == {"Ez"}
    assert tally["planted_flip"]["caught"] and tally["planted_flip"]["placed"]
    assert l3_row.set_verdict(judged, "ref") == "differs"


# ---------------------------------------------------------------------------------------
# Dispatch state
# ---------------------------------------------------------------------------------------

def _dispatch_row(step_path="fused", slots=("step_B",), dispatches=10, plan=(4, None)):
    counters = {"step_B": {"arm": "PML", "backend": "triton", "dispatches": dispatches,
                           "plan_launches": plan[0]},
                "update_E": {"arm": "PML", "backend": "triton", "dispatches": 0,
                             "plan_launches": plan[1]}}
    return {"dispatch_at_end": {"step_path": step_path,
                                "dispatch": {"dispatched_slots": list(slots)}},
            "launch_sums": l3_row.counter_sums(counters), "launch_counters": counters}


def test_dispatch_state_needs_a_fused_step_dispatched_slots_and_launches():
    assert l3_row.dispatch_state(_dispatch_row()) == "dispatched"
    assert l3_row.dispatch_state(_dispatch_row(plan=(None, None))) == "dispatched"
    assert l3_row.dispatch_state(_dispatch_row(step_path="array")) == "array"
    assert l3_row.dispatch_state(_dispatch_row(slots=())) == "array"
    assert l3_row.dispatch_state(_dispatch_row(dispatches=0)) == "array"
    assert l3_row.dispatch_state(_dispatch_row(plan=(0, 0))) == "array"
    assert l3_row.dispatch_state({"error": "x"}) == "error"
    counters = _dispatch_row()["launch_counters"]
    sums = l3_row.counter_sums(counters)
    assert (sums["dispatches"], sums["plan_launches"]) == (10, 4)
    assert sums["plan_launches_unknown_slots"] == ["update_E"]
    assert sums["launched_slot_backends"] == {"step_B": "triton"}
    assert l3_row.counter_leaf_total(counters) == 14


def test_the_row_verdict_names_the_array_path():
    held = {"per_set": {"s": {"digests": 2, "stored_equal": 2, "reader_equal": 2,
                              "planted_flip": {"caught": True, "placed": True}}},
            "fpu_checks": 1, "compared_steps": 1}
    assert l3_row.row_verdict({**held, **_dispatch_row()}, "s") == "holds_dispatched"
    assert l3_row.row_verdict({**held, **_dispatch_row(step_path="array")}, "s") == "holds_array"
    assert l3_row.row_verdict({**held, "error": "boom"}, "s") == "error"


# ---------------------------------------------------------------------------------------
# The leg, with a stub row program
# ---------------------------------------------------------------------------------------

STUB = textwrap.dedent('''
    import json, os, sys
    spec = json.load(open(sys.argv[1]))
    with open(os.environ["STUB_CALLS"], "a") as handle:
        handle.write(spec["label"] + "\\n")
    rows = {}
    for name in spec["refs"]:
        differs = spec["label"] == "r_diff" and (name == spec["primary"]
                                                 or bool(os.environ.get("STUB_DIFFER_EVERYWHERE")))
        equal = 23 if differs else 24
        rows[name] = {"digests": 24, "stored_equal": equal, "reader_equal": 24,
                      "planted_flip": {"caught": True, "placed": True}, "first": None}
    counters = {"step_B": {"arm": "PML", "backend": "triton", "dispatches": 5,
                           "plan_launches": 5}}
    result = {"label": spec["label"], "mode": spec["mode"], "table": spec["table"],
              "per_set": rows, "fpu_checks": 2, "compared_steps": 2, "n_steps": 1,
              "dispatch_at_end": {"step_path": "fused", "dispatch": {
                  "table": "triton", "dispatched_slots": ["step_B"], "slots": 7}},
              "launch_counters": counters,
              "launch_sums": {"dispatches": 5, "plan_launches": 5,
                              "plan_launches_unknown_slots": []},
              "fast_path_report": {"big": True},
              "env": {k: v for k, v in os.environ.items()
                      if k == "CC" or k.startswith(("MEEP_GPU_", "MGPU_SITE_", "CUPY_", "OMP_",
                                                    "PYTHONNOUSERSITE", "TRITON_LIBCUDA_PATH",
                                                    "LD_LIBRARY_PATH"))}}
    json.dump(result, open(spec["out_json"], "w"))
    print("L3-JSON " + json.dumps(result), flush=True)
''')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_record(tmp_path: Path, sets=("linux_flush",), bad_planes=()):
    ref = tmp_path / "ref"
    record = {"sets": {}}
    for name in sets:
        rows = {}
        for label in ("r_hold", "r_diff", "r_bad"):
            files = {}
            for role, file in (("digests", "digests.json"), ("planes", "planes.npy"),
                               ("result", "result.json")):
                path = ref / name / label / file
                path.parent.mkdir(parents=True, exist_ok=True)
                # The two Apple and Linux flush references disagree on r_diff only.
                words = "mac" if (label == "r_diff" and name == "mac_flush") else "same"
                path.write_text(json.dumps({"stored": {"0": {"Ez": words}},
                                            "reader": {"0": {"Ez": "same"}}, "role": role}))
                files[role] = {"file": f"{name}/{label}/{file}", "sha256": _sha(path)}
            if label in bad_planes:
                files["planes"]["sha256"] = "0" * 64
            rows[label] = {"status": "holds", "files": files}
        rows["r_refused"] = {"status": "refused", "reason": "a named refusal"}
        mode = "keep" if name == "linux_keep" else "flush"
        record["sets"][name] = {"mode": mode, "rows": rows,
                                "meep_images": {"_meep.so": ["a" * 64]}}
    path = ref / "meep_identity_tier_b.json"
    path.write_text(json.dumps(record))
    repo = tmp_path / "repo"
    (repo / "parity" / "meep_gpu").mkdir(parents=True)
    (repo / "parity" / "meep_gpu" / "meep_identity_corpus.py").write_text("")
    stub = tmp_path / "stub_row.py"
    stub.write_text(STUB)
    return path, repo, stub


#: The corpus sibling the reference build keeps beside python/ (build_meep_133_record.py).
SIBLING = "tests/cyl-ellipsoid-eps-ref.h5"


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "meep-1.33.0" / "python"
    for name, text in (("examples/a.py", "a = 1\n"), ("tests/test_b.py", "b = 2\n"),
                       (f"../{SIBLING}", "eps\n")):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _build_record(tmp_path: Path, root: Path, relative: bool = False) -> Path:
    """A reference environment's build record naming ``root`` and its file list, as the
    reference build scripts write them (``reference_build.json``, ``corpus_sha256.txt``)."""
    prefix = tmp_path / "env"
    record = prefix / "share" / "meep-gpu-build" / "reference_build.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{_sha(path)}  {path.relative_to(root).as_posix()}\n"
             for path in sorted(root.rglob("*")) if path.is_file()]
    listing = record.parent / "corpus_sha256.txt"
    listing.write_text("".join(lines))
    named = str(Path("..") / root.relative_to(tmp_path)) if relative else str(root)
    record.write_text(json.dumps({"schema": "meep-gpu/reference-build/1", "corpus": {
        "root": named, "list": listing.name, "list_sha256": _sha(listing),
        "files": len(lines),
        "siblings": [{"path": SIBLING, "sha256": _sha(root.parent / SIBLING)}]}}))
    return record


def _leg(tmp_path, record, repo, stub, *extra):
    out = tmp_path / "out"
    named = any(arg in extra for arg in ("--corpus-root", "--build-record"))
    corpus = [] if named else ["--corpus-root", str(_corpus(tmp_path)),
                               "--build-record", str(tmp_path / "absent.json")]
    rc = l3_corpus.main(["--record", str(record), "--repo", str(repo), "--out", str(out),
                         "--child", str(stub), "--python", sys.executable, "--heartbeat", "0",
                         "--parallel", "2", *corpus, *extra])
    return rc, out


def _calls(tmp_path):
    path = tmp_path / "calls.txt"
    return path.read_text().split() if path.exists() else []


def test_the_leg_writes_its_schema_as_rows_land_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "0")
    record, repo, stub = _fake_record(tmp_path, bad_planes=("r_bad",))
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush")
    assert rc == l3_corpus.EXIT_NOT_HOLDS
    assert sorted(_calls(tmp_path)) == ["r_diff", "r_hold"]

    lines = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert sorted(row["label"] for row in lines) == ["r_diff", "r_hold"]
    for row in lines:
        assert {"label", "l3_verdicts", "l3_dispatch", "l3_verdict", "child_rc",
                "wall_seconds", "per_set", "launch_sums"} <= set(row)
        assert "fast_path_report" not in row
        assert json.loads((out / "rows" / f"{row['label']}.json").read_text())["fast_path_report"]
        assert {k for k in row["env"] if k.startswith("MEEP_GPU_")} == {
            "MEEP_GPU_DISPATCH_LOG", "MEEP_GPU_CORPUS_ROOT"}
        corpus = (tmp_path / "meep-1.33.0" / "python").resolve()
        assert row["env"]["MEEP_GPU_CORPUS_ROOT"] == str(corpus)
        assert row["env"]["MGPU_SITE_MEEP_SOURCE"] == str(corpus.parent)
        assert row["env"]["CUPY_ACCELERATORS"] == "" and row["env"]["OMP_NUM_THREADS"] == "1"
    verdicts = {row["label"]: row["l3_verdict"] for row in lines}
    assert verdicts == {"r_hold": "holds_dispatched", "r_diff": "differs"}

    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS"
    assert (summary["rows_planned"], summary["rows_done"]) == (3, 2)
    assert summary["not_run_record"] == {"r_refused": "refused"}
    assert set(summary["refused_unverified_files"]) == {"r_bad"}
    assert summary["by_set"]["linux_flush"]["verdicts"] == {"holds": 1, "differs": 1}
    assert summary["by_set"]["linux_flush"]["stored_digests_equal"] == 47
    assert summary["by_set"]["linux_flush"]["digests_compared"] == 48
    assert summary["rows_not_holding_against_primary"] == ["r_diff"]
    assert summary["dispatched_by_table"] == {"triton": 2}
    assert "repo_at_end" in summary
    leg = json.loads((out / "leg.json").read_text())
    assert leg["args"]["table"] == "triton" and "environment" in leg
    assert leg["corpus"]["source"] == "--corpus-root"
    assert leg["corpus"]["verified"].startswith("not verified")
    progress = (out / "progress.log").read_text()
    assert "START leg triton_flush" in progress and "VERDICT DIFFERS" in progress
    assert progress.count("triton_flush 1/3") + progress.count("triton_flush 2/3") == 2
    assert json.loads((out / "LEG_DONE").read_text())["verdict"] == "DIFFERS"

    # Resume: nothing that landed runs again, nothing is appended twice.
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush")
    assert rc == l3_corpus.EXIT_NOT_HOLDS and len(_calls(tmp_path)) == 2
    assert len((out / "rows.jsonl").read_text().splitlines()) == 2


def _reference_python(tmp_path: Path, compiler: bool = True) -> Path:
    """An interpreter in a reference environment as its build leaves it (the build record,
    the compiler Triton's launchers need, the libcuda link and its record): this
    interpreter, linked into ``<prefix>/bin``."""
    prefix = tmp_path / "refenv"
    records = prefix / "share" / "meep-gpu-build"
    records.mkdir(parents=True)
    (records / "reference_build.json").write_text("{}")
    (prefix / "bin").mkdir()
    (prefix / "bin" / "python").symlink_to(sys.executable)
    if compiler:
        cc = prefix.joinpath(*legs.LAUNCHER_COMPILER)
        cc.write_text("#!/bin/sh\nexit 0\n")
        cc.chmod(0o755)
    (prefix / "lib" / "meep-gpu-libcuda").mkdir(parents=True)
    (prefix / "lib" / "meep-gpu-libcuda" / "libcuda.so").write_text("")
    (records / legs.LIBCUDA_RECORD).write_text(json.dumps({"directory": "lib/meep-gpu-libcuda"}))
    return prefix / "bin" / "python"


def _as_on_linux(monkeypatch):
    monkeypatch.setattr(legs, "LAUNCHER_PLATFORMS", (sys.platform,))
    for name in ("CC", "TRITON_LIBCUDA_PATH", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)


def test_an_nvidia_leg_builds_launchers_with_the_environment_s_compiler(tmp_path, monkeypatch):
    _as_on_linux(monkeypatch)
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    python = _reference_python(tmp_path)
    prefix = python.parent.parent
    record, repo, stub = _fake_record(tmp_path)
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                   "--python", str(python))
    assert rc == l3_corpus.EXIT_NOT_HOLDS
    assert sorted(_calls(tmp_path)) == ["r_bad", "r_diff", "r_hold"]
    for line in (out / "rows.jsonl").read_text().splitlines():
        env = json.loads(line)["env"]
        assert env["CC"] == str(prefix / "bin" / "x86_64-conda-linux-gnu-cc")
        assert env["TRITON_LIBCUDA_PATH"] == str(prefix / "lib" / "meep-gpu-libcuda")
        assert env["LD_LIBRARY_PATH"].split(":")[0] == env["TRITON_LIBCUDA_PATH"]
    leg = json.loads((out / "leg.json").read_text())
    assert leg["launcher_toolchain"]["cc_source"] == "reference environment"
    assert leg["environment"]["CC"] == str(prefix / "bin" / "x86_64-conda-linux-gnu-cc")
    # A --clean-env leg gets the toolchain too; the caller's own CC and stub win, with or
    # without --clean-env; a Metal leg gets none.
    def clean(*extra):
        args = l3_corpus.resolve(l3_corpus.parser().parse_args(
            ["--record", str(record), "--out", str(out), "--table", "cuda", "--baseline",
             "keep", "--python", str(python), "--clean-env", *extra]))
        return args
    args = clean()
    args.toolchain = l3_corpus.leg_toolchain(args, {"HOME": "/h"})
    env = l3_corpus.leg_environment(args, {"HOME": "/h"})
    assert env["CC"] == str(prefix / "bin" / "x86_64-conda-linux-gnu-cc")
    assert env["PATH"] == f"{prefix}/bin:/usr/bin:/bin"
    args = clean("--libcuda-stub", "/stub")
    args.toolchain = l3_corpus.leg_toolchain(args, {"CC": "/opt/cc", "HOME": "/h"})
    env = l3_corpus.leg_environment(args, {"CC": "/opt/cc", "HOME": "/h"})
    assert (env["CC"], env["TRITON_LIBCUDA_PATH"]) == ("/opt/cc", "/stub")
    metal = l3_corpus.leg_environment(_resolved("--table", "metal", "--baseline", "flush",
                                                "--out", str(out), "--clean-env"), {"HOME": "/h"})
    assert "CC" not in metal and "TRITON_LIBCUDA_PATH" not in metal


def test_an_nvidia_leg_is_refused_when_the_environment_lacks_its_compiler(tmp_path,
                                                                         monkeypatch):
    _as_on_linux(monkeypatch)
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    python = _reference_python(tmp_path, compiler=False)
    record, repo, stub = _fake_record(tmp_path)
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                   "--python", str(python))
    assert rc == l3_corpus.EXIT_REFUSED and _calls(tmp_path) == []
    progress = (out / "progress.log").read_text()
    assert "REFUSING" in progress and "has no C compiler at" in progress
    assert not (out / "leg.json").exists()


def test_a_metal_leg_judges_every_set_and_honours_pre_registered_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    record, repo, stub = _fake_record(tmp_path, sets=("linux_flush", "mac_flush", "linux_keep"))
    expected = tmp_path / "l0.txt"
    expected.write_text("# pre-registered\nr_diff\n")
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "metal", "--baseline", "flush",
                   "--expect-differ", str(expected), "--expect-differ-holds", "mac_flush")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["by_set"]["linux_flush"]["verdicts"] == {"holds": 2, "differs": 1}
    assert summary["by_set"]["mac_flush"]["verdicts"] == {"holds": 3}
    assert summary["expected_differ"]["as_expected"] == ["r_diff"]
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert all({k for k in row["env"] if k.startswith("MEEP_GPU_")}
               == {"MEEP_GPU_DISPATCH_LOG", "MEEP_GPU_CORPUS_ROOT"} for row in rows)


def test_the_rows_expected_to_differ_can_be_derived_from_the_record(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    record, repo, stub = _fake_record(tmp_path, sets=("linux_flush", "mac_flush", "linux_keep"))
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "metal", "--baseline", "flush",
                   "--expect-differ", "references", "--expect-differ-holds", "mac_flush")
    listed = read_lines(out / "expected_differ.txt")
    assert listed == ["r_diff"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["expected_differ"]["rows"] == ["r_diff"]
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS


@pytest.mark.parametrize("everywhere, listed, failed", [
    ("1", "references", "r_diff"),
    ("", "r_hold", "r_hold"),
])
def test_a_row_expected_to_differ_holds_only_as_expected(tmp_path, monkeypatch, everywhere,
                                                         listed, failed):
    # Expected to differ against the primary set and hold against the named one: a row that
    # differs against both, or holds against the primary set instead, is not a hold.
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_DIFFER_EVERYWHERE", everywhere)
    record, repo, stub = _fake_record(tmp_path, sets=("linux_flush", "mac_flush", "linux_keep"))
    if listed != "references":
        (tmp_path / "l0.txt").write_text(listed + "\n")
        listed = str(tmp_path / "l0.txt")
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "metal", "--baseline", "flush",
                   "--expect-differ", listed, "--expect-differ-holds", "mac_flush")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["expected_differ"]["not_as_expected"] == [failed]
    assert summary["expected_differ"]["as_expected"] == []
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS


def read_lines(path: Path):
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


@pytest.mark.parametrize("argv, words", [
    (("--table", "triton", "--baseline", "keep", "--sets", "linux_flush"), "is mode flush"),
    (("--table", "triton", "--baseline", "flush", "--rows", "r_refused"), "not holding"),
    (("--table", "triton", "--baseline", "flush", "--sets", "mac_keep"), "not in the record"),
    (("--table", "triton", "--baseline", "flush", "--expect-record-sha256", "0" * 64),
     "is not sha256"),
])
def test_a_leg_the_record_cannot_support_runs_no_row(tmp_path, monkeypatch, argv, words):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    record, repo, stub = _fake_record(tmp_path)
    rc, out = _leg(tmp_path, record, repo, stub, *argv)
    assert rc == l3_corpus.EXIT_REFUSED and _calls(tmp_path) == []
    assert words in (out / "progress.log").read_text()
    assert not (out / "rows.jsonl").exists()


# ---------------------------------------------------------------------------------------
# The corpus: named, never inherited
# ---------------------------------------------------------------------------------------

def _corpus_args(tmp_path: Path, *argv: str):
    return _resolved("--table", "triton", "--baseline", "keep", "--out", str(tmp_path / "leg"),
                     *argv)


def test_the_corpus_comes_from_the_build_record_and_is_verified_file_by_file(tmp_path):
    root = _corpus(tmp_path)
    record = _build_record(tmp_path, root, relative=True)
    (root / "tests" / "__pycache__").mkdir()
    (root / "tests" / "__pycache__" / "test_b.cpython-312.pyc").write_bytes(b"\0")
    corpus = l3_corpus.resolve_corpus(_corpus_args(tmp_path, "--build-record", str(record)))
    assert corpus["root"] == str(root.resolve()) and corpus["source"] == "build record"
    assert corpus["verified"] == (f"2 of 2 recorded files match under {root.resolve()}; 1 of 1 "
                                  f"recorded sibling(s) match beside it ({SIBLING})")
    assert corpus["siblings"] == {SIBLING: _sha(root.parent / SIBLING)}
    assert corpus["build_record_sha256"] == _sha(record) and corpus["unlisted"] == []
    (root / "examples" / "data.h5").write_bytes(b"\0")
    corpus = l3_corpus.resolve_corpus(_corpus_args(tmp_path, "--build-record", str(record)))
    assert corpus["unlisted"] == [f"1 unlisted non-code file(s) under {root.resolve()}, "
                                  f"not refused: examples/data.h5"]
    assert corpus["verified"].endswith("not refused: examples/data.h5")


@pytest.mark.parametrize("name, refused", [
    ("examples/._a.py", True),             # an AppleDouble file with a source suffix
    ("tests/stray.pyc", True),             # byte code outside __pycache__
    ("tests/stray.cpython-312-darwin.so", True),
    ("examples/.DS_Store", False),
    ("tests/__pycache__/test_b.cpython-312.pyc", False),
])
def test_l3_and_the_gate_stage_judge_a_corpus_tree_by_one_rule(tmp_path, name, refused):
    root = _corpus(tmp_path)
    record = _build_record(tmp_path, root)
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(b"\0")
    recorded = l3_corpus.recorded_corpus(record)
    problems, _ = legs.verify_corpus(root, recorded["files"], recorded["siblings"])
    assert bool(problems) is refused
    argv = ("--build-record", str(record))
    if refused:
        with pytest.raises(l3_corpus.Refusal, match=re.escape(name)):
            l3_corpus.resolve_corpus(_corpus_args(tmp_path, *argv))
    else:
        assert l3_corpus.resolve_corpus(_corpus_args(tmp_path, *argv))["root"] == str(root.resolve())


@pytest.mark.parametrize("change, words", [
    ("edit", r"changed \(sha256\): examples/a.py"),
    ("remove", "absent: tests/test_b.py"),
    ("add", "unlisted and importable: tests/stray.py"),
    ("list", "is not the one the build record"),
    ("count", "holds 2 files; the build record names 3"),
    ("uncounted", "without a root, a file list, its sha256 and its file count"),
    ("sibling", "the corpus sibling tests/cyl-ellipsoid-eps-ref.h5 under .* is changed"),
    ("no sibling", "the corpus sibling tests/cyl-ellipsoid-eps-ref.h5 is absent from"),
    ("unrecorded", "names no corpus siblings"),
])
def test_a_corpus_that_is_not_the_recorded_one_is_refused_naming_the_file(tmp_path, change,
                                                                          words):
    root = _corpus(tmp_path)
    record = _build_record(tmp_path, root)
    if change == "edit":
        (root / "examples" / "a.py").write_text("a = 2\n")
    elif change == "remove":
        (root / "tests" / "test_b.py").unlink()
    elif change == "add":
        (root / "tests" / "stray.py").write_text("")
    elif change == "count":
        document = json.loads(record.read_text())
        document["corpus"]["files"] += 1
        record.write_text(json.dumps(document))
    elif change == "uncounted":
        document = json.loads(record.read_text())
        del document["corpus"]["files"]
        record.write_text(json.dumps(document))
    elif change == "sibling":
        (root.parent / SIBLING).write_text("other\n")
    elif change == "no sibling":
        (root.parent / SIBLING).unlink()
    elif change == "unrecorded":
        document = json.loads(record.read_text())
        del document["corpus"]["siblings"]
        record.write_text(json.dumps(document))
    else:
        listing = record.parent / "corpus_sha256.txt"
        listing.write_text(listing.read_text() + listing.read_text().splitlines()[0] + "\n")
    for argv in (("--build-record", str(record)),
                 ("--build-record", str(record), "--corpus-root", str(root))):
        with pytest.raises(l3_corpus.Refusal, match=words):
            l3_corpus.resolve_corpus(_corpus_args(tmp_path, *argv))


def test_a_corpus_neither_given_nor_recorded_is_refused_by_name(tmp_path):
    absent = tmp_path / "absent" / "reference_build.json"
    with pytest.raises(l3_corpus.Refusal, match="no MEEP corpus.*does not exist"):
        l3_corpus.resolve_corpus(_corpus_args(tmp_path, "--build-record", str(absent)))
    silent = tmp_path / "silent.json"
    silent.write_text(json.dumps({"schema": "meep-gpu/reference-build/1"}))
    with pytest.raises(l3_corpus.Refusal, match="names none .'corpus'."):
        l3_corpus.resolve_corpus(_corpus_args(tmp_path, "--build-record", str(silent)))
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"corpus": {"root": str(tmp_path)}}))
    with pytest.raises(l3_corpus.Refusal,
                       match="without a root, a file list, its sha256 and its file count"):
        l3_corpus.resolve_corpus(_corpus_args(tmp_path, "--build-record", str(broken)))


def test_a_given_corpus_without_a_record_runs_unverified_and_says_so(tmp_path):
    root = _corpus(tmp_path)
    corpus = l3_corpus.resolve_corpus(_corpus_args(
        tmp_path, "--corpus-root", str(root), "--build-record", str(tmp_path / "none.json")))
    assert corpus["source"] == "--corpus-root" and corpus["build_record"] is None
    assert corpus["verified"].startswith("not verified")


@pytest.mark.parametrize("layout, words", [
    ("meep-1.33.0/examples", "is not a python/ tree"),
    ("half/python", "holds no tests/"),
])
def test_a_corpus_root_the_rows_cannot_read_is_refused(tmp_path, layout, words):
    root = tmp_path / layout
    (root / "examples").mkdir(parents=True)
    with pytest.raises(l3_corpus.Refusal, match=words):
        l3_corpus.resolve_corpus(_corpus_args(
            tmp_path, "--corpus-root", str(root), "--build-record", str(tmp_path / "none.json")))


def test_every_row_reads_the_named_corpus_never_the_shell_s(tmp_path):
    root = _corpus(tmp_path)
    args = _corpus_args(tmp_path, "--corpus-root", str(root), "--build-record",
                        str(tmp_path / "none.json"))
    args.corpus = l3_corpus.resolve_corpus(args)
    shell = {"MEEP_GPU_CORPUS_ROOT": "/elsewhere/python", "MGPU_SITE_MEEP_SOURCE": "/elsewhere"}
    for clean in (False, True):
        args.clean_env = clean
        env = l3_corpus.leg_environment(args, shell)
        assert env["MEEP_GPU_CORPUS_ROOT"] == str(root.resolve())
        assert env["MGPU_SITE_MEEP_SOURCE"] == str(root.resolve().parent)


def test_a_leg_with_no_corpus_runs_no_row(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    record, repo, stub = _fake_record(tmp_path)
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                   "--build-record", str(tmp_path / "absent.json"))
    assert rc == l3_corpus.EXIT_REFUSED and _calls(tmp_path) == []
    assert "no MEEP corpus" in (out / "progress.log").read_text()


def test_a_leg_takes_the_recorded_corpus_and_records_it(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    record, repo, stub = _fake_record(tmp_path)
    root = _corpus(tmp_path)
    build = _build_record(tmp_path, root)
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                   "--build-record", str(build))
    leg = json.loads((out / "leg.json").read_text())
    assert leg["corpus"]["source"] == "build record"
    assert leg["corpus"]["root"] == str(root.resolve())
    assert leg["environment"]["MEEP_GPU_CORPUS_ROOT"] == str(root.resolve())
    assert "MEEP corpus " + str(root.resolve()) in (out / "progress.log").read_text()


# ---------------------------------------------------------------------------------------
# What ships
# ---------------------------------------------------------------------------------------

#: A home or scratch directory, or an e-mail address (built from parts, so this file
#: carries none itself).
PRIVATE = re.compile("|".join([re.escape("/" + part + "/") for part in ("Users", "home")]
                              + [re.escape("/private/" + part + "/") for part in ("var", "tmp")]
                              + [re.escape("/var/" + "folders/"), r"[A-Za-z]:\\Users",
                                 r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"]))


@pytest.mark.parametrize("name", ["l3_corpus.py", "l3_row.py", "test_l3_corpus.py", "RUNBOOK.md"])
def test_the_tool_and_its_runbook_name_no_local_path_or_address(name):
    text = (HERE / name).read_text(encoding="utf-8")
    assert not [match.group(0) for match in PRIVATE.finditer(text)]
