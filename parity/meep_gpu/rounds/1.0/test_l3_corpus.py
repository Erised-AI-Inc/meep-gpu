"""Host-only tests of the L3 corpus check (``l3_corpus.py``, ``l3_row.py``): no GPU, no MEEP.

* the arguments: each table's reference sets, primary, image set and environment, and the
  combinations refused by name before any row runs;
* the per-row judge, on synthetic words and on a Tier A fixture stepped by the NumPy array
  path, with a planted one-ulp flip that must be caught at its step and component;
* the leg's output schema, its verdict and exit status, and its resume, with a stub row
  program standing in for the lift;
* the rows declared run-to-run variable: the class read off the corpus manifest, the
  re-runs in fresh processes, their attribution by the source tables each run deposited,
  and a declared row's recorders (its tables' digest and its own digests).
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


def _fixture_judge(planted=None, keep_digests=False):
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
                                  fpu_state=identity.subnormal_state, keep_digests=keep_digests)
        finally:
            driver.close()
    judged["compared_steps"] = len(steps)
    return judged, steps, document


def test_the_judge_holds_on_a_fixture_and_catches_a_flip_planted_in_the_driver():
    judged, steps, document = _fixture_judge()
    assert l3_row.RUN_DIGESTS_KEY not in judged  # kept for a declared row only
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
    if os.environ.get("STUB_SPECS"):
        with open(os.environ["STUB_SPECS"], "a") as handle:
            handle.write(json.dumps({"label": spec["label"],
                                     "declared": spec.get("declared_variable"),
                                     "attempt": spec.get("attempt")}) + "\\n")
    # A declared row's run n: STUB_DIFFER_RUNS names the runs that differ against the
    # primary set, STUB_DIFFER_EVERYWHERE_RUNS those that differ against every set,
    # STUB_ERROR_RUNS those that error, STUB_TABLE the source-table digest
    # each run deposits (comma lists; a run beyond STUB_TABLE records no digest).
    declared = bool(spec.get("declared_variable"))
    attempt = str(spec.get("attempt"))
    def runs(name):
        return [item for item in os.environ.get(name, "").split(",") if item]
    if declared and attempt in runs("STUB_ERROR_RUNS"):
        result = {"label": spec["label"], "mode": spec["mode"], "table": spec["table"],
                  "error": "RuntimeError: a stub error on run " + attempt}
        json.dump(result, open(spec["out_json"], "w"))
        print("L3-JSON " + json.dumps(result), flush=True)
        sys.exit(0)
    rows = {}
    for name in spec["refs"]:
        differs = ((spec["label"] == "r_diff" and (name == spec["primary"]
                                                   or bool(os.environ.get("STUB_DIFFER_EVERYWHERE"))))
                   or spec["label"] == os.environ.get("STUB_ALSO_DIFFERS")
                   or (declared and name == spec["primary"] and attempt in runs("STUB_DIFFER_RUNS"))
                   or (declared and attempt in runs("STUB_DIFFER_EVERYWHERE_RUNS")))
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
    if os.environ.get("STUB_IMPORTS"):
        digest = ("a" if spec["label"] == "r_hold" else "b") * 64
        result["imported_source_sha256"] = {"meep_gpu/driver.py": digest,
                                            "meep_gpu/fastpath.py": "c" * 64}
    if declared:
        tables = runs("STUB_TABLE")
        index = int(attempt) - 1
        result["attempt"] = int(attempt)
        result["declared_variable"] = (
            {"tables": {"sha256": tables[index], "per_source": []}, "solves": []}
            if index < len(tables) else {"error": "no table digest", "solves": []})
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


#: The corpus sibling the reference build keeps beside python/ (build_meep_reference_record.py).
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
        # The rows leave the machine: paths under their labels (neutral_ids.py).
        corpus = (tmp_path / "meep-1.33.0" / "python").resolve()
        assert row["env"]["MEEP_GPU_CORPUS_ROOT"] == l3_corpus.N.text(str(corpus))
        assert row["env"]["MGPU_SITE_MEEP_SOURCE"] == l3_corpus.N.text(str(corpus.parent))
        assert l3_corpus.N.residue(row) == []
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
    assert summary["import_records"] == {"rows_with_one": 0, "rows": 2, "files": 0,
                                         "files_with_two_digests": []}
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


def test_the_summary_counts_the_rows_import_records(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_IMPORTS", "1")
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "0")
    record, repo, stub = _fake_record(tmp_path)
    _rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush")
    lines = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    assert all(row[l3_row.IMPORTED_KEY]["meep_gpu/fastpath.py"] == "c" * 64 for row in lines)
    summary = json.loads((out / "summary.json").read_text())
    # Three rows ran (r_hold, r_diff, r_bad); r_hold's driver.py digest differs.
    assert summary["import_records"] == {"rows_with_one": 3, "rows": 3, "files": 2,
                                         "files_with_two_digests": ["meep_gpu/driver.py"]}


def _row_spec(tmp_path: Path) -> Path:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"label": "r_hold", "table": "triton", "mode": "flush",
                                "out_json": str(tmp_path / "result.json")}))
    return spec


@pytest.mark.parametrize("raises", [False, True])
def test_the_row_program_writes_its_own_import_record(tmp_path, monkeypatch, capsys, raises):
    """Every repository module the row's process held, by sha256, keyed as every gate
    artifact keys it: the row program itself and the reader among them, on the error
    path too."""
    def run(spec):
        if raises:
            raise RuntimeError("the lift failed")
        return {"label": spec["label"]}

    monkeypatch.setattr(l3_row, "run", run)
    assert l3_row.main([str(_row_spec(tmp_path))]) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1][len(l3_row.MARK):])
    assert printed == result
    assert ("error" in result) is raises
    record = result[l3_row.IMPORTED_KEY]
    assert record["parity/meep_gpu/rounds/1.0/l3_row.py"] == hashlib.sha256(
        (HERE / "l3_row.py").read_bytes()).hexdigest()
    assert "parity/meep_gpu/gate_provenance.py" in record
    assert all(path.startswith(("meep_gpu/", "parity/meep_gpu/")) for path in record)


def test_a_row_whose_reader_cannot_be_imported_says_so(tmp_path, monkeypatch):
    import parity.meep_gpu as package  # noqa: PLC0415

    monkeypatch.setattr(l3_row, "run", lambda spec: {"label": spec["label"]})
    monkeypatch.delattr(package, "gate_provenance", raising=False)
    monkeypatch.setitem(sys.modules, "parity.meep_gpu.gate_provenance", None)
    assert l3_row.main([str(_row_spec(tmp_path))]) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert l3_row.IMPORTED_KEY not in result
    assert "gate_provenance" in result["import_record_error"]


def test_the_row_import_record_uses_the_gate_artifacts_key():
    from parity.meep_gpu import gate_provenance  # noqa: PLC0415

    assert l3_row.IMPORTED_KEY == gate_provenance.IMPORTED_KEY


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
        assert env["CC"] == l3_corpus.N.text(str(prefix / "bin" / "x86_64-conda-linux-gnu-cc"))
        assert env["TRITON_LIBCUDA_PATH"] == l3_corpus.N.text(str(prefix / "lib" / "meep-gpu-libcuda"))
        assert env["LD_LIBRARY_PATH"].split(":")[0] == env["TRITON_LIBCUDA_PATH"]
    leg = json.loads((out / "leg.json").read_text())
    assert leg["launcher_toolchain"]["cc_source"] == "reference environment"
    assert leg["environment"]["CC"] == l3_corpus.N.text(str(prefix / "bin" / "x86_64-conda-linux-gnu-cc"))
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


@pytest.mark.parametrize("table, sets, argv", [
    ("metal", ("linux_flush", "mac_flush", "linux_keep"),
     ("--expect-differ", "references", "--expect-differ-holds", "mac_flush")),
    ("cuda", None, ("--second-reference", "linux_avx2_flush")),
])
def test_the_rows_expected_to_differ_name_the_record_as_the_rows_do(tmp_path, monkeypatch, table,
                                                                    sets, argv):
    # expected_differ.txt goes back with the round: its header names no local path.
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    if sets:
        record, repo, stub = _fake_record(tmp_path, sets=sets)
    else:
        record, repo, stub = _second_reference_record(tmp_path, mode="flush")
    _rc, out = _leg(tmp_path, record, repo, stub, "--table", table, "--baseline", "flush", *argv)
    text = (out / "expected_differ.txt").read_text()
    assert l3_corpus.N.text(text) == text and l3_corpus.N.text(str(record)) in text
    assert read_lines(out / "expected_differ.txt") == ["r_diff"]


def read_lines(path: Path):
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


def _excluded_record(primary_status="holds"):
    """Three sets; ``r_excl`` is an exclusion on the two Linux sets, measured ``primary_status``
    under flush and differing under keep, and holds on the Apple set."""
    sets = {}
    for name, mode in (("linux_flush", "flush"), ("mac_flush", "flush"), ("linux_keep", "keep")):
        rows = {"r_hold": {"status": "holds"}, "r_refused": {"status": "refused"}}
        if name == "mac_flush":
            rows["r_excl"] = {"status": "holds"}
        else:
            status = primary_status if name == "linux_flush" else "differs"
            rows["r_excl"] = {"status": status, "excluded": "a named exclusion"}
        sets[name] = {"mode": mode, "rows": rows, "meep_images": {}}
    return {"sets": sets}


@pytest.mark.parametrize("primary_status", ["holds", "differs"])
@pytest.mark.parametrize("table, baseline", [("triton", "flush"), ("cuda", "flush"),
                                             ("metal", "flush")])
def test_an_excluded_row_is_never_planned_whatever_it_measured(table, baseline, primary_status):
    # An exclusion is planned out by name even where its own process happened to hold, so a
    # leg never judges it and a set where it differs does not refuse the leg.
    plan = l3_corpus.plan_rows(_resolved("--table", table, "--baseline", baseline),
                               _excluded_record(primary_status))
    assert plan["labels"] == ["r_hold"]
    assert plan["not_run"] == {"r_refused": "refused", "r_excl": f"excluded ({primary_status})"}


def test_an_excluded_row_is_refused_when_named_to_run():
    with pytest.raises(l3_corpus.Refusal, match="not holding"):
        l3_corpus.plan_rows(_resolved("--table", "triton", "--baseline", "flush", "--rows", "r_excl"),
                            _excluded_record())


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
# A second reference build (--second-reference)
# ---------------------------------------------------------------------------------------

SECOND = "linux_avx2_keep"


def _words(mode: str = "keep"):
    """The MEEP words of each row in each set of a ``mode`` fixture: ``r_hold`` is equal
    everywhere; ``r_diff`` is host-dependent (the Apple and Linux references differ) and the
    second Linux build differs on it too; ``r_hd`` is host-dependent and the two Linux
    builds agree on it."""
    return {f"linux_{mode}": {"r_hold": "w", "r_diff": "linux", "r_hd": "linux"},
            f"mac_{mode}": {"r_hold": "w", "r_diff": "apple", "r_hd": "apple"},
            f"linux_avx2_{mode}": {"r_hold": "w", "r_diff": "second", "r_hd": "linux"}}


def _image(name: str) -> str:
    """A set's stand-in MEEP image digest, distinct per set."""
    return hashlib.sha256(name.encode()).hexdigest()


def _second_reference_record(tmp_path: Path, words=None, sets=None, mode="keep", readers=None,
                             modes=None):
    """A ``mode`` record holding ``linux_<mode>``, ``mac_<mode>`` and the second reference,
    with the three row files of every row at their sha256 (the digests carry ``words`` as
    the stored digests and ``readers``, default equal everywhere, as the reader digests);
    ``modes`` names a set recorded in another mode."""
    words = words or _words(mode)
    readers = readers or {}
    ref = tmp_path / "ref"
    record = {"sets": {}}
    for name in sets or words:
        rows = {}
        for label, word in words[name].items():
            files = {}
            for role, file in (("digests", "digests.json"), ("planes", "planes.npy"),
                               ("result", "result.json")):
                path = ref / name / label / file
                path.parent.mkdir(parents=True, exist_ok=True)
                reader = (readers.get(name) or {}).get(label, "same")
                path.write_text(json.dumps({"stored": {"0": {"Ez": word}},
                                            "reader": {"0": {"Ez": reader}}, "role": role}))
                files[role] = {"file": f"{name}/{label}/{file}", "sha256": _sha(path)}
            rows[label] = {"status": "holds", "files": files}
        rows["r_refused"] = {"status": "refused", "reason": "a named refusal"}
        record["sets"][name] = {"mode": (modes or {}).get(name, mode), "rows": rows,
                                "meep_images": {"_meep.so": [_image(name)]}}
    path = ref / "meep_identity_tier_b.json"
    path.write_text(json.dumps(record))
    repo = tmp_path / "repo"
    (repo / "parity" / "meep_gpu").mkdir(parents=True, exist_ok=True)
    (repo / "parity" / "meep_gpu" / "meep_identity_corpus.py").write_text("")
    stub = tmp_path / "stub_row.py"
    stub.write_text(STUB)
    return path, repo, stub


@pytest.mark.parametrize("table", ["triton", "cuda"])
def test_a_second_reference_leg_judges_both_linux_sets_and_loads_the_second_build(table):
    args = _resolved("--table", table, "--baseline", "keep", "--second-reference", SECOND)
    assert (args.set_list, args.primary, args.image_set) == (["linux_keep", SECOND],
                                                             "linux_keep", SECOND)
    assert args.expect_differ_holds == SECOND and args.cross_mode == set()


@pytest.mark.parametrize("argv, words", [
    (("--table", "metal", "--baseline", "flush", "--second-reference", "linux_avx2_flush"),
     "an NVIDIA leg loads"),
    (("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
      "--sets", f"linux_keep,{SECOND}"), "is not given with --sets"),
    (("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
      "--expect-differ", "references", "--expect-differ-holds", SECOND),
     "is not given with --expect-differ, --expect-differ-holds"),
    (("--table", "triton", "--baseline", "keep", "--second-reference", "linux_keep"),
     "is the leg's primary set"),
    # The new build's images with only the old set judged: the lift would load a MEEP the
    # leg never judges against.
    (("--table", "triton", "--baseline", "keep", "--image-set", SECOND),
     "the image set linux_avx2_keep is not among the sets"),
    (("--table", "metal", "--baseline", "flush", "--sets", "linux_flush,linux_keep"),
     "the image set mac_flush is not among the sets"),
])
def test_a_second_reference_named_wrongly_is_refused_by_name(argv, words):
    with pytest.raises(l3_corpus.Refusal, match=re.escape(words)):
        _resolved(*argv)


def test_the_second_build_s_images_are_admitted_only_through_second_reference(tmp_path):
    # The same sets and image set spelled by hand skip the host-dependence check: refused.
    record, _repo, _stub = _second_reference_record(tmp_path)
    args = _resolved("--table", "triton", "--baseline", "keep", "--sets", f"linux_keep,{SECOND}",
                     "--image-set", SECOND, "--expect-differ", "references",
                     "--expect-differ-holds", SECOND, "--record", str(record))
    with pytest.raises(l3_corpus.Refusal, match="admitted only by --second-reference"):
        l3_corpus.plan_rows(args, json.loads(record.read_text()))


def test_a_second_reference_expects_only_its_host_dependent_differences(tmp_path):
    record, _repo, _stub = _second_reference_record(tmp_path)
    args = _resolved("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
                     "--record", str(record))
    plan = l3_corpus.plan_rows(args, json.loads(record.read_text()))
    assert plan["labels"] == ["r_hold", "r_diff", "r_hd"]
    # r_hd is host-dependent, but the two Linux builds agree on it: it must hold against both.
    assert plan["expected_differ"] == ["r_diff"]
    assert plan["meep_images"] == {"_meep.so": [_image(SECOND)]}
    assert plan["second_reference"] == {
        "set": SECOND, "of": "linux_keep", "host_dependence": "mac_keep",
        "rows_expected_to_differ": ["r_diff"],
        "apple_digests_read": ["mac_keep/r_diff/digests.json"]}


@pytest.mark.parametrize("change, words", [
    # The second build differs on a row the Apple and Linux references agree on.
    ({"mac_keep": {"r_hold": "w", "r_diff": "linux", "r_hd": "apple"}},
     "where the record's Apple and Linux references agree"),
    ({SECOND: {"r_hold": "other", "r_diff": "second", "r_hd": "linux"}},
     "['r_hold']; a difference there is a finding"),
])
def test_a_second_reference_differing_where_the_hosts_agree_is_refused(tmp_path, change, words):
    record, _repo, _stub = _second_reference_record(tmp_path, {**_words(), **change})
    args = _resolved("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
                     "--record", str(record))
    with pytest.raises(l3_corpus.Refusal, match=re.escape(words)):
        l3_corpus.plan_rows(args, json.loads(record.read_text()))


def test_a_second_reference_differing_only_in_its_reader_words_is_judged_the_same(tmp_path):
    # The get_array words count as the stored words do: a reader-only difference on a
    # host-dependent row is expected, and on a host-agreeing row refuses the leg.
    readers = {SECOND: {"r_hd": "second"}, "mac_keep": {"r_hd": "apple"}}
    record, _repo, _stub = _second_reference_record(tmp_path, readers=readers)
    args = _resolved("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
                     "--record", str(record))
    plan = l3_corpus.plan_rows(args, json.loads(record.read_text()))
    assert plan["expected_differ"] == ["r_diff", "r_hd"]
    readers = {SECOND: {"r_hold": "second"}}
    record, _repo, _stub = _second_reference_record(tmp_path / "agree", readers=readers)
    args = _resolved("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
                     "--record", str(record))
    with pytest.raises(l3_corpus.Refusal, match=re.escape("['r_hold']; a difference there")):
        l3_corpus.plan_rows(args, json.loads(record.read_text()))


def test_a_second_reference_of_another_mode_is_refused_even_named_cross_mode(tmp_path):
    record, _repo, _stub = _second_reference_record(tmp_path, modes={SECOND: "flush"})
    args = _resolved("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
                     "--cross-mode-sets", SECOND, "--record", str(record))
    with pytest.raises(l3_corpus.Refusal, match="the second reference linux_avx2_keep is mode flush"):
        l3_corpus.plan_rows(args, json.loads(record.read_text()))


@pytest.mark.parametrize("damage, words", [
    ("no_apple_set", "the record holds no mac_keep"),
    ("apple_digests_edited", "mac_keep r_diff: the digests file's sha256 differs"),
    ("second_digests_absent", "linux_avx2_keep r_hold: the record names no digests file"),
])
def test_a_second_reference_the_record_cannot_show_host_dependent_is_refused(tmp_path, damage,
                                                                             words):
    sets = [name for name in _words() if not (damage == "no_apple_set" and name == "mac_keep")]
    record, _repo, _stub = _second_reference_record(tmp_path, sets=sets)
    if damage == "apple_digests_edited":
        (record.parent / "mac_keep" / "r_diff" / "digests.json").write_text("{}")
    if damage == "second_digests_absent":
        (record.parent / SECOND / "r_hold" / "digests.json").unlink()
    args = _resolved("--table", "triton", "--baseline", "keep", "--second-reference", SECOND,
                     "--record", str(record))
    with pytest.raises(l3_corpus.Refusal, match=re.escape(words)):
        l3_corpus.plan_rows(args, json.loads(record.read_text()))


def test_a_host_dependent_row_holds_against_the_second_build_as_expected(tmp_path, monkeypatch):
    # r_diff differs against linux_keep and holds against the second build: expected, a
    # hold. r_hd and r_hold hold against both.
    # (A flush leg: a keep leg mints its CuPy cache through the package first.)
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    second = "linux_avx2_flush"
    record, repo, stub = _second_reference_record(tmp_path, mode="flush")
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                   "--second-reference", second)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
    assert (summary["primary"], summary["image_set"]) == ("linux_flush", second)
    assert summary["by_set"]["linux_flush"]["verdicts"] == {"holds": 2, "differs": 1}
    assert summary["by_set"][second]["verdicts"] == {"holds": 3}
    assert summary["expected_differ"] == {"rows": ["r_diff"], "holds_against": second,
                                          "as_expected": ["r_diff"], "not_as_expected": []}
    assert summary["second_reference"] == {
        "set": second, "of": "linux_flush", "host_dependence": "mac_flush",
        "apple_digests_read": ["mac_flush/r_diff/digests.json"]}
    assert sorted(_calls(tmp_path)) == ["r_diff", "r_hd", "r_hold"]
    assert read_lines(out / "expected_differ.txt") == ["r_diff"]
    assert "--second-reference" in (out / "expected_differ.txt").read_text()
    assert "second reference linux_avx2_flush" in (out / "progress.log").read_text()


@pytest.mark.parametrize("everywhere, also, failing, not_as_expected", [
    # A host-agreeing row that differs against linux_keep: a finding, whatever the second
    # build says.
    ("", "r_hold", ["r_hold"], []),
    # A host-dependent row the Linux builds agree on is not expected to differ.
    ("", "r_hd", ["r_hd"], []),
    # The expected row differs against the second build too.
    ("1", "", [], ["r_diff"]),
])
def test_a_second_reference_leg_admits_no_other_difference(tmp_path, monkeypatch, everywhere,
                                                           also, failing, not_as_expected):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_DIFFER_EVERYWHERE", everywhere)
    monkeypatch.setenv("STUB_ALSO_DIFFERS", also)
    record, repo, stub = _second_reference_record(tmp_path, mode="flush")
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "cuda", "--baseline", "flush",
                   "--second-reference", "linux_avx2_flush")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert summary["rows_not_holding_against_primary"] == failing
    assert summary["expected_differ"]["not_as_expected"] == not_as_expected


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
    assert leg["corpus"]["root"] == l3_corpus.N.text(str(root.resolve()))
    assert leg["environment"]["MEEP_GPU_CORPUS_ROOT"] == l3_corpus.N.text(str(root.resolve()))
    assert l3_corpus.N.residue(leg) == []
    assert "MEEP corpus " + l3_corpus.N.text(str(root.resolve())) in (out / "progress.log").read_text()


# ---------------------------------------------------------------------------------------
# Rows declared run-to-run variable (an EigenModeSource among the row's sources)
# ---------------------------------------------------------------------------------------

def _manifest(tmp_path: Path, eigenmode=("r_hold",), blind=()) -> Path:
    """A corpus manifest naming the fake record's rows, every one with ``mp.Source`` but
    those in ``eigenmode``, which carry an ``EigenModeSource``; ``blind`` rows carry no
    ``source_kinds``."""
    rows = []
    for label in ("r_hold", "r_diff", "r_bad", "r_refused"):
        facts = {"src_time_kinds": ["GaussianSource"]}
        if label not in blind:
            facts["source_kinds"] = ["EigenModeSource" if label in eigenmode else "Source"]
        rows.append({"label": label, "facts": facts})
    path = tmp_path / "corpus_manifest.json"
    path.write_text(json.dumps({"schema": "meep_gpu.meep_identity_corpus.manifest",
                                "version": 1, "rows": rows}))
    return path


def _lines(out: Path, label: str):
    return [row for row in (json.loads(line) for line in
                            (out / "rows.jsonl").read_text().splitlines())
            if row["label"] == label]


def _variable_leg(tmp_path, monkeypatch, *extra, differ="", tables="", errors="",
                  manifest=True):
    """A triton flush leg of r_hold and r_bad, r_hold declared run-to-run variable."""
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_SPECS", str(tmp_path / "specs.jsonl"))
    monkeypatch.setenv("STUB_DIFFER_RUNS", differ)
    monkeypatch.setenv("STUB_TABLE", tables)
    monkeypatch.setenv("STUB_ERROR_RUNS", errors)
    monkeypatch.delenv("MEEP_IDENTITY_CORPUS_MANIFEST", raising=False)
    if not (tmp_path / "ref").exists():
        _fake_record(tmp_path)
    record, repo, stub = (tmp_path / "ref" / "meep_identity_tier_b.json", tmp_path / "repo",
                          tmp_path / "stub_row.py")
    named = ["--corpus-manifest", str(_manifest(tmp_path))] if manifest else []
    return _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                "--rows", "r_hold,r_bad", *named, *extra)


def test_the_variable_class_is_the_table_generator_s_eigenmode_class():
    # Named by the classifier's one home, never transcribed.
    from parity.meep_gpu.build_meep_identity_table import source_class  # noqa: PLC0415

    assert l3_corpus.VARIABLE_CLASS == source_class({"source_kinds": ["EigenModeSource"]})
    assert source_class({"source_kinds": ["EigenmodeSource", "Source"]}) == l3_corpus.VARIABLE_CLASS
    assert source_class({"source_kinds": ["Source"]}) != l3_corpus.VARIABLE_CLASS
    assert l3_corpus.VARIABLE_ATTEMPTS == 4


def test_the_declared_class_is_read_off_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_SPECS", str(tmp_path / "specs.jsonl"))
    monkeypatch.setenv("STUB_TABLE", "aa")
    record, repo, stub = _fake_record(tmp_path)
    manifest = _manifest(tmp_path)
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                   "--corpus-manifest", str(manifest))
    assert read_lines(out / "declared_variable.txt") == ["r_hold"]
    header = (out / "declared_variable.txt").read_text().splitlines()[0]
    assert header.startswith("# 1 of 3 planned rows declared run-to-run variable")
    assert _sha(manifest) in header and l3_corpus.N.text(header) == header
    specs = [json.loads(line) for line in (tmp_path / "specs.jsonl").read_text().splitlines()]
    assert {spec["label"]: spec["declared"] for spec in specs} == {
        "r_hold": True, "r_diff": False, "r_bad": False}
    assert [spec["attempt"] for spec in specs if spec["label"] == "r_hold"] == [1]
    leg = json.loads((out / "leg.json").read_text())
    assert leg["declared_variable"]["rows"] == ["r_hold"]
    assert leg["declared_variable"]["manifest_sha256"] == _sha(manifest)
    assert l3_corpus.N.residue(leg) == []
    summary = json.loads((out / "summary.json").read_text())
    variable = summary["run_to_run_variable"]
    assert (variable["read"], variable["rows"], variable["held_first"]) == (True, ["r_hold"],
                                                                           ["r_hold"])
    assert variable["class"] == l3_corpus.VARIABLE_CLASS and variable["findings"] == {}
    line = _lines(out, "r_hold")[0]
    assert line["attempt"] == 1 and l3_corpus.table_digest(line) == "aa"
    assert (out / "attempts" / "r_hold.run1.json").is_file()
    assert (out / "attempts" / "r_hold.run1.log").is_file()
    # The undeclared row that differs is a finding, read and never run again (a control).
    assert rc == l3_corpus.EXIT_NOT_HOLDS and summary["rows_not_holding_against_primary"] == ["r_diff"]
    assert len(_lines(out, "r_diff")) == 1 and "attempt" not in _lines(out, "r_diff")[0]
    assert "1 declared run-to-run variable" in (out / "progress.log").read_text()


def test_a_leg_without_a_manifest_declares_nothing_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_ALSO_DIFFERS", "r_hold")
    rc, out = _variable_leg(tmp_path, monkeypatch, tables="aa,bb", manifest=False)
    assert rc == l3_corpus.EXIT_NOT_HOLDS
    text = (out / "declared_variable.txt").read_text()
    assert read_lines(out / "declared_variable.txt") == []
    assert text.startswith("# 0 rows declared run-to-run variable: no corpus manifest named")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["run_to_run_variable"]["read"] is False
    assert summary["run_to_run_variable"]["rows"] == []
    assert summary["rows_not_holding_against_primary"] == ["r_hold"]
    # Judged by the strict rule: one run, and its spec declares nothing.
    assert len(_lines(out, "r_hold")) == 1 and _calls(tmp_path).count("r_hold") == 1
    assert "no row declared run-to-run variable" in (out / "progress.log").read_text()


def test_a_manifest_without_source_kinds_refuses_the_leg_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    record, repo, stub = _fake_record(tmp_path)
    manifest = _manifest(tmp_path, blind=("r_bad",))
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "triton", "--baseline", "flush",
                   "--corpus-manifest", str(manifest))
    assert rc == l3_corpus.EXIT_REFUSED and _calls(tmp_path) == []
    assert "names no source_kinds for ['r_bad']" in (out / "progress.log").read_text()
    assert not (out / "rows.jsonl").exists()


def test_a_declared_row_holding_on_a_rerun_on_other_tables_holds(tmp_path, monkeypatch):
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,bb")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
    lines = _lines(out, "r_hold")
    assert [line["attempt"] for line in lines] == [1, 2]
    assert [line["l3_verdict"] for line in lines] == ["differs", "holds_dispatched"]
    variable = summary["run_to_run_variable"]
    assert variable["held_on_rerun"] == {"r_hold": {"runs": 2, "tables": ["aa", "bb"],
                                                    "first_differences": [None]}}
    assert variable["held_first"] == [] and variable["findings"] == {}
    assert summary["by_set"]["linux_flush"]["verdicts"] == {"holds": 2}
    assert summary["rows_not_holding_against_primary"] == []
    for name in ("r_hold.run1.json", "r_hold.run2.json", "r_hold.run1.log", "r_hold.run2.log"):
        assert (out / "attempts" / name).is_file()
    assert json.loads((out / "rows" / "r_hold.json").read_text())["attempt"] == 2
    progress = (out / "progress.log").read_text()
    assert "r_hold: declared run-to-run variable and not as required on run 1 of 4" in progress
    assert "held on a re-run 1 ['r_hold']" in progress


def test_a_declared_row_rerun_on_the_same_tables_is_a_finding(tmp_path, monkeypatch):
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,aa")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    lines = _lines(out, "r_hold")
    assert len(lines) == 2
    assert summary["rows_not_holding_against_primary"] == ["r_hold"]
    assert "the same source tables gave other words" in \
        summary["run_to_run_variable"]["findings"]["r_hold"]
    # The finding is on the row's own last line: its words held, its verdict reads differs.
    last = lines[-1]
    assert last["l3_verdicts"]["linux_flush"] == "holds" and last["l3_verdict"] == "differs"
    assert last[l3_corpus.ATTRIBUTION_KEY]["outcome"] == "differs"
    assert last[l3_corpus.ATTRIBUTION_KEY]["tables"] == ["aa", "aa"]
    assert lines[0][l3_corpus.ATTRIBUTION_KEY]["outcome"] == "undecided"
    assert summary["verdicts_against_primary"] == {"differs": 1, "holds_dispatched": 1}
    done = json.loads((out / "LEG_DONE").read_text())
    assert done["verdict"] == "DIFFERS" and set(done["run_to_run_findings"]) == {"r_hold"}


def _placed(tmp_path: Path, out: Path, name: str) -> Path:
    """A new leg directory holding the last line of every label of ``out``, as
    ``carry.py place`` writes one (``carried_from`` on each line)."""
    into = tmp_path / name
    (into / "rows").mkdir(parents=True)
    rows = {}
    for line in (out / "rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[row["label"]] = row
    for row in rows.values():
        row["carried_from"] = {"source": str(out), "state": "CARRY-DIGEST"}
    (into / "rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                             for row in rows.values()))
    return into


def test_a_carried_finding_row_does_not_carry_and_stays_a_finding(tmp_path, monkeypatch):
    import carry  # noqa: PLC0415
    import plan_carry  # noqa: PLC0415

    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,aa")
    finding = _lines(out, "r_hold")[-1]
    # The carry judge refuses it twice over: its verdict reads differs, and its stamp is not
    # a hold. A holding row the leg did not declare carries as before.
    judged = plan_carry.judge_l3_row(finding, tmp_path, {}, None)
    assert judged["state"] == plan_carry.RERUN_UNRELEASED
    assert "attribution is 'differs'" in carry.declared_row_problem(finding)
    held = dict(finding, l3_verdict="holds_dispatched")
    assert "attribution is 'differs'" in carry.declared_row_problem(held)
    unstamped = {key: value for key, value in held.items() if key != l3_corpus.ATTRIBUTION_KEY}
    assert "carries no attribution stamp" in carry.declared_row_problem(unstamped)
    assert carry.declared_row_problem(_lines(out, "r_bad")[-1]) is None
    # Placed anyway, as its one last line (a hand-made record: place refuses it), it is
    # never run again and stays the finding: the leg reads the stamp.
    into = _placed(tmp_path, out, "placed")
    rc, _out = _leg(tmp_path, tmp_path / "ref" / "meep_identity_tier_b.json", tmp_path / "repo",
                    tmp_path / "stub_row.py", "--table", "triton", "--baseline", "flush",
                    "--rows", "r_hold,r_bad", "--corpus-manifest", str(_manifest(tmp_path)),
                    "--out", str(into))
    summary = json.loads((into / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert "the same source tables" in summary["run_to_run_variable"]["findings"]["r_hold"]
    assert summary["carried"] == ["r_bad", "r_hold"] and _calls(tmp_path).count("r_hold") == 2


def test_a_carried_row_that_held_on_a_rerun_keeps_its_runs(tmp_path, monkeypatch):
    import carry  # noqa: PLC0415

    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,bb")
    last = _lines(out, "r_hold")[-1]
    assert carry.declared_row_problem(last) is None
    into = _placed(tmp_path, out, "placed")
    rc, _out = _leg(tmp_path, tmp_path / "ref" / "meep_identity_tier_b.json", tmp_path / "repo",
                    tmp_path / "stub_row.py", "--table", "triton", "--baseline", "flush",
                    "--rows", "r_hold,r_bad", "--corpus-manifest", str(_manifest(tmp_path)),
                    "--out", str(into))
    summary = json.loads((into / "summary.json").read_text())
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
    assert summary["run_to_run_variable"]["held_on_rerun"]["r_hold"]["tables"] == ["aa", "bb"]
    assert summary["run_to_run_variable"]["held_first"] == []


@pytest.mark.parametrize("manifest", ["none", "undeclared"])
def test_a_finding_row_resumed_without_its_declaration_is_refused(tmp_path, monkeypatch,
                                                                  manifest):
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,aa")
    assert rc == l3_corpus.EXIT_NOT_HOLDS
    leg = (out / "leg.json").read_text()
    first = (out / l3_corpus.FIRST_START).read_text()
    assert json.loads(first)["rows"] == ["r_hold"]
    calls = len(_calls(tmp_path))
    if manifest == "none":
        rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,aa",
                                manifest=False)
    else:
        rc, out = _leg(tmp_path, tmp_path / "ref" / "meep_identity_tier_b.json",
                       tmp_path / "repo", tmp_path / "stub_row.py", "--table", "triton",
                       "--baseline", "flush", "--rows", "r_hold,r_bad", "--corpus-manifest",
                       str(_manifest(tmp_path, eigenmode=())))
    assert rc == l3_corpus.EXIT_REFUSED and len(_calls(tmp_path)) == calls
    progress = (out / "progress.log").read_text()
    assert "declared then and not now ['r_hold']" in progress
    assert "start it with the corpus manifest it started with" in progress
    assert (out / "leg.json").read_text() == leg  # a refused start writes nothing
    assert (out / l3_corpus.FIRST_START).read_text() == first
    assert json.loads((out / "summary.json").read_text())["verdict"] == "DIFFERS"
    # With the first-start record gone, the rows' own lines still name the declared row.
    (out / l3_corpus.FIRST_START).unlink()
    rc, out = _variable_leg(tmp_path, monkeypatch, manifest=False)
    assert rc == l3_corpus.EXIT_REFUSED
    assert "declared then and not now ['r_hold']" in (out / "progress.log").read_text()
    # With its own manifest the resume reads the stamp: still the finding, nothing re-run.
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,aa")
    assert rc == l3_corpus.EXIT_NOT_HOLDS and len(_calls(tmp_path)) == calls
    summary = json.loads((out / "summary.json").read_text())
    assert summary["rows_not_holding_against_primary"] == ["r_hold"]


def test_a_declared_run_with_no_table_digest_is_not_attributed(tmp_path, monkeypatch):
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert "no source-table digest" in summary["run_to_run_variable"]["findings"]["r_hold"]


def test_errored_runs_are_not_among_the_four(tmp_path, monkeypatch):
    # Run 2 errors and stops the runs; after --retry-errors runs 3 to 5 differ: four runs
    # with words, five lines, and no holding run in 4.
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1,3,4,5", errors="2",
                            tables="aa,bb,cc,dd,ee")
    assert rc == l3_corpus.EXIT_NOT_HOLDS and len(_lines(out, "r_hold")) == 2
    variable = json.loads((out / "summary.json").read_text())["run_to_run_variable"]
    assert variable["findings"] == {} and "no run of 1 held so far" in variable["undecided"]["r_hold"]
    rc, out = _variable_leg(tmp_path, monkeypatch, "--retry-errors", differ="1,3,4,5",
                            errors="2", tables="aa,bb,cc,dd,ee")
    summary = json.loads((out / "summary.json").read_text())
    assert [line.get("attempt") for line in _lines(out, "r_hold")] == [1, 2, 3, 4, 5]
    assert _calls(tmp_path).count("r_hold") == 5
    variable = summary["run_to_run_variable"]
    assert variable["findings"] == {"r_hold": "no run of 4 held"}
    assert variable["runs"]["r_hold"] == {"with_words": 4, "errored": 1}
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS


def test_a_declared_row_s_driver_error_line_names_its_run(tmp_path, monkeypatch):
    original = l3_corpus.Leg.run_row

    def run_row(self, label, refs):
        if label == "r_hold":
            raise RuntimeError("the driver failed")
        return original(self, label, refs)

    monkeypatch.setattr(l3_corpus.Leg, "run_row", run_row)
    rc, out = _variable_leg(tmp_path, monkeypatch, tables="aa")
    line = _lines(out, "r_hold")[-1]
    assert line["error"].startswith("driver: RuntimeError") and line["attempt"] == 1
    assert line[l3_corpus.ATTRIBUTION_KEY]["outcome"] == "error"
    assert rc == l3_corpus.EXIT_NOT_HOLDS


def test_a_plan_pass_attributes_nothing(tmp_path, monkeypatch):
    rc, out = _variable_leg(tmp_path, monkeypatch, "--plan-only", tables="aa")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "PLAN-ONLY" and rc == l3_corpus.EXIT_HOLDS
    variable = summary["run_to_run_variable"]
    assert variable["judged"] is False and variable["findings"] == {}
    assert l3_corpus.ATTRIBUTION_KEY not in _lines(out, "r_hold")[-1]


def _manifest_both(tmp_path):
    return str(_manifest(tmp_path, eigenmode=("r_hold", "r_bad")))


def _rows_leg(tmp_path, rows, *extra):
    return _leg(tmp_path, tmp_path / "ref" / "meep_identity_tier_b.json", tmp_path / "repo",
                tmp_path / "stub_row.py", "--table", "triton", "--baseline", "flush",
                "--rows", rows, *extra)


def test_a_resume_that_widens_or_narrows_the_planned_rows_is_accepted(tmp_path, monkeypatch):
    # Both rows declared. Start with r_hold alone, widen to both, narrow to r_bad, then the
    # full command again: every start is accepted, and the first start's record stands.
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_TABLE", "aa")
    monkeypatch.delenv("MEEP_IDENTITY_CORPUS_MANIFEST", raising=False)
    _fake_record(tmp_path)
    out = tmp_path / "out"

    def start(rows):
        return _rows_leg(tmp_path, rows, "--corpus-manifest", _manifest_both(tmp_path))[0]

    assert start("r_hold") == l3_corpus.EXIT_HOLDS
    first = (out / l3_corpus.FIRST_START).read_text()
    assert json.loads(first)["rows"] == ["r_hold"] and json.loads(first)["planned"] == ["r_hold"]
    assert start("r_hold,r_bad") == l3_corpus.EXIT_HOLDS
    assert start("r_bad") == l3_corpus.EXIT_HOLDS
    assert start("r_hold,r_bad") == l3_corpus.EXIT_HOLDS
    assert (out / l3_corpus.FIRST_START).read_text() == first
    assert "REFUSING" not in (out / "progress.log").read_text()


def test_a_start_after_a_narrowed_one_keeps_the_first_declaration(tmp_path, monkeypatch):
    # The first start plans both rows and runs none (STOP); a start narrowed to r_bad runs it;
    # the first command, with the same manifest, is then accepted and runs r_hold declared.
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_TABLE", "aa")
    monkeypatch.delenv("MEEP_IDENTITY_CORPUS_MANIFEST", raising=False)
    _fake_record(tmp_path)
    manifest = str(_manifest(tmp_path))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "STOP").write_text("")
    rc, out = _rows_leg(tmp_path, "r_hold,r_bad", "--corpus-manifest", manifest)
    assert _calls(tmp_path) == [] and json.loads((out / "summary.json").read_text())["verdict"] \
        == "INCOMPLETE"
    (out / "STOP").unlink()
    rc, out = _rows_leg(tmp_path, "r_bad", "--corpus-manifest", manifest)
    assert rc == l3_corpus.EXIT_HOLDS and _calls(tmp_path) == ["r_bad"]
    assert json.loads((out / l3_corpus.FIRST_START).read_text())["rows"] == ["r_hold"]
    rc, out = _rows_leg(tmp_path, "r_hold,r_bad", "--corpus-manifest", manifest)
    assert rc == l3_corpus.EXIT_HOLDS and _calls(tmp_path) == ["r_bad", "r_hold"]
    assert _lines(out, "r_hold")[-1]["attempt"] == 1
    # Without the manifest that same command is refused by name, before any row runs.
    rc, out = _rows_leg(tmp_path, "r_hold,r_bad")
    assert rc == l3_corpus.EXIT_REFUSED and "declared then and not now ['r_hold']" in \
        (out / "progress.log").read_text()


def test_a_carry_whose_plan_drops_a_declared_row_is_accepted(tmp_path, monkeypatch):
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,bb")
    into = _placed(tmp_path, out, "placed")
    rc, _out = _rows_leg(tmp_path, "r_bad", "--corpus-manifest", str(_manifest(tmp_path)),
                         "--out", str(into))
    summary = json.loads((into / "summary.json").read_text())
    assert rc == l3_corpus.EXIT_HOLDS and summary["verdict"] == "HOLDS"
    assert "REFUSING" not in (into / "progress.log").read_text()
    # A later start that plans the declared row must declare it: its carried line names it
    # (the first start did not plan it, so only the rows' own lines can say so).
    rc, _out = _rows_leg(tmp_path, "r_hold,r_bad", "--out", str(into))
    assert rc == l3_corpus.EXIT_REFUSED
    assert "plans and does not declare: ['r_hold']" in (into / "progress.log").read_text()
    rc, _out = _rows_leg(tmp_path, "r_hold,r_bad", "--corpus-manifest", str(_manifest(tmp_path)),
                         "--out", str(into))
    assert rc == l3_corpus.EXIT_HOLDS


def test_a_placed_finding_of_no_holding_run_is_never_run_again(tmp_path, monkeypatch):
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1,2,3,4", tables="aa,bb,cc,dd")
    assert json.loads((out / "summary.json").read_text())["run_to_run_variable"]["findings"] \
        == {"r_hold": "no run of 4 held"}
    calls = _calls(tmp_path).count("r_hold")
    into = _placed(tmp_path, out, "placed")
    monkeypatch.setenv("STUB_DIFFER_RUNS", "")  # a fresh run would hold
    rc, _out = _rows_leg(tmp_path, "r_hold,r_bad", "--corpus-manifest",
                         str(_manifest(tmp_path)), "--out", str(into))
    summary = json.loads((into / "summary.json").read_text())
    assert _calls(tmp_path).count("r_hold") == calls  # not run again
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert summary["run_to_run_variable"]["findings"] == {"r_hold": "no run of 4 held"}
    assert summary["run_to_run_variable"]["runs"]["r_hold"] == {"with_words": 4, "errored": 0}


def test_a_stop_with_runs_due_leaves_the_leg_incomplete_and_the_row_undecided(tmp_path,
                                                                             monkeypatch):
    original = l3_corpus.Leg.run_row

    def run_row(self, label, refs):
        row = original(self, label, refs)
        if label == "r_hold":
            (self.out / "STOP").write_text("")
        return row

    monkeypatch.setattr(l3_corpus.Leg, "run_row", run_row)
    rc, out = _variable_leg(tmp_path, monkeypatch, "--parallel", "1", differ="1",
                            tables="aa,bb")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "INCOMPLETE" and rc == l3_corpus.EXIT_NOT_HOLDS
    variable = summary["run_to_run_variable"]
    assert variable["runs_due"] == ["r_hold"] and variable["findings"] == {}
    assert "r_hold" in variable["undecided"]
    assert summary["rows_not_holding_against_primary"] == []
    assert json.loads((out / "LEG_DONE").read_text())["verdict"] == "INCOMPLETE"
    monkeypatch.setattr(l3_corpus.Leg, "run_row", original)
    (out / "STOP").unlink()
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1", tables="aa,bb")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
    assert summary["run_to_run_variable"]["held_on_rerun"]["r_hold"]["tables"] == ["aa", "bb"]


def test_an_equal_tables_finding_after_two_earlier_runs(tmp_path, monkeypatch):
    # Run 1 on aa and run 2 on bb differ, run 3 holds on aa: run 1 used the holding run's
    # own tables, so the row is a finding (each earlier run is compared, not one of them).
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1,2", tables="aa,bb,aa")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert "the same source tables" in summary["run_to_run_variable"]["findings"]["r_hold"]
    last = _lines(out, "r_hold")[-1]
    assert last["attempt"] == 3 and last["l3_verdict"] == "differs"
    assert last[l3_corpus.ATTRIBUTION_KEY]["tables"] == ["aa", "bb", "aa"]


def test_a_finding_on_a_declared_row_expected_to_differ(tmp_path, monkeypatch):
    # r_diff is expected to differ against linux_flush and hold against mac_flush. Run 1
    # differs against every set, run 2 is as expected on the same tables: a finding, and the
    # leg reads DIFFERS though the last run's own verdicts are the expected ones.
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_TABLE", "aa,aa")
    monkeypatch.setenv("STUB_DIFFER_EVERYWHERE_RUNS", "1")
    monkeypatch.delenv("MEEP_IDENTITY_CORPUS_MANIFEST", raising=False)
    record, repo, stub = _fake_record(tmp_path, sets=("linux_flush", "mac_flush", "linux_keep"))
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "metal", "--baseline", "flush",
                   "--expect-differ", "references", "--expect-differ-holds", "mac_flush",
                   "--corpus-manifest", str(_manifest(tmp_path, eigenmode=("r_diff",))))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert summary["rows_not_holding_against_primary"] == ["r_diff"]
    assert summary["expected_differ"]["as_expected"] == []
    assert "the same source tables" in summary["run_to_run_variable"]["findings"]["r_diff"]
    last = _lines(out, "r_diff")[-1]
    assert last["l3_verdicts"]["linux_flush"] == "differs" and last["l3_verdicts"]["mac_flush"] \
        == "holds"


def test_a_declared_row_that_never_holds_is_a_finding(tmp_path, monkeypatch):
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1,2,3,4", tables="aa,bb,cc,dd")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert [line["attempt"] for line in _lines(out, "r_hold")] == [1, 2, 3, 4]
    assert _calls(tmp_path).count("r_hold") == 4
    assert summary["run_to_run_variable"]["findings"] == {"r_hold": "no run of 4 held"}
    assert summary["rows_not_holding_against_primary"] == ["r_hold"]


def test_an_undeclared_row_that_differs_is_never_rerun(tmp_path, monkeypatch):
    # r_hold is declared but r_bad is not: r_bad differing is read, never run again.
    monkeypatch.setenv("STUB_ALSO_DIFFERS", "r_bad")
    rc, out = _variable_leg(tmp_path, monkeypatch, tables="aa")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert len(_lines(out, "r_bad")) == 1 and _calls(tmp_path).count("r_bad") == 1
    assert summary["rows_not_holding_against_primary"] == ["r_bad"]
    assert summary["run_to_run_variable"]["held_first"] == ["r_hold"]


def test_an_errored_run_decides_nothing_and_a_retry_that_holds_holds(tmp_path, monkeypatch):
    # [error, holds]: the error stops the row's runs as any error does, and is DIFFERS until
    # --retry-errors runs it again; the error carries no tables, so the run that then holds
    # is the row's first draw, and holds.
    rc, out = _variable_leg(tmp_path, monkeypatch, errors="1", tables="aa,bb")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
    assert len(_lines(out, "r_hold")) == 1 and "r_hold" in summary["errors"]
    rc, out = _variable_leg(tmp_path, monkeypatch, "--retry-errors", errors="1", tables="aa,bb")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
    lines = _lines(out, "r_hold")
    assert [line.get("attempt") for line in lines] == [1, 2] and "error" in lines[0]
    assert summary["run_to_run_variable"]["held_first"] == ["r_hold"]
    assert summary["run_to_run_variable"]["findings"] == {}


def test_a_resumed_leg_continues_a_declared_row_s_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(l3_corpus, "VARIABLE_ATTEMPTS", 2)
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1,2", tables="aa,bb,cc")
    assert rc == l3_corpus.EXIT_NOT_HOLDS and len(_lines(out, "r_hold")) == 2
    monkeypatch.setattr(l3_corpus, "VARIABLE_ATTEMPTS", 4)
    rc, out = _variable_leg(tmp_path, monkeypatch, differ="1,2", tables="aa,bb,cc")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
    assert [line["attempt"] for line in _lines(out, "r_hold")] == [1, 2, 3]
    assert summary["run_to_run_variable"]["held_on_rerun"]["r_hold"]["tables"] == ["aa", "bb", "cc"]
    # r_bad held on its first run and is not run again.
    assert _calls(tmp_path).count("r_bad") == 1
    assert "1 declared to run again" in (out / "progress.log").read_text()


@pytest.mark.parametrize("everywhere", ["", "1"])
def test_a_declared_row_expected_to_differ_is_held_to_the_expected_rule(tmp_path, monkeypatch,
                                                                       everywhere):
    # r_diff is expected to differ against linux_flush and hold against mac_flush. Differing
    # as expected is as required on its first run; differing against both sets is not, so it
    # runs again, and after 4 runs it is not as expected.
    monkeypatch.setenv("STUB_CALLS", str(tmp_path / "calls.txt"))
    monkeypatch.setenv("STUB_DIFFER_EVERYWHERE", everywhere)
    monkeypatch.setenv("STUB_TABLE", "aa,bb,cc,dd")
    monkeypatch.delenv("MEEP_IDENTITY_CORPUS_MANIFEST", raising=False)
    record, repo, stub = _fake_record(tmp_path, sets=("linux_flush", "mac_flush", "linux_keep"))
    rc, out = _leg(tmp_path, record, repo, stub, "--table", "metal", "--baseline", "flush",
                   "--expect-differ", "references", "--expect-differ-holds", "mac_flush",
                   "--corpus-manifest", str(_manifest(tmp_path, eigenmode=("r_diff",))))
    summary = json.loads((out / "summary.json").read_text())
    if not everywhere:
        assert summary["expected_differ"]["as_expected"] == ["r_diff"]
        assert summary["run_to_run_variable"]["held_first"] == ["r_diff"]
        assert summary["verdict"] == "HOLDS" and rc == l3_corpus.EXIT_HOLDS
        assert len(_lines(out, "r_diff")) == 1
    else:
        assert summary["expected_differ"]["not_as_expected"] == ["r_diff"]
        assert summary["run_to_run_variable"]["findings"] == {"r_diff": "no run of 4 held"}
        assert summary["rows_not_holding_against_primary"] == []
        assert summary["verdict"] == "DIFFERS" and rc == l3_corpus.EXIT_NOT_HOLDS
        assert len(_lines(out, "r_diff")) == 4


def _line(kind, table="aa"):
    line = {"label": "r", "l3_verdicts": {"p": kind}}
    if kind == "error":
        line = {"label": "r", "error": "x"}
    if table:
        line["declared_variable"] = {"tables": {"sha256": table}}
    return line


@pytest.mark.parametrize("lines, outcome", [
    ([_line("holds")], "holds"),
    ([_line("error", None), _line("holds", "bb")], "holds"),
    ([_line("differs", "aa"), _line("holds", "bb")], "holds_variable"),
    ([_line("differs", "aa"), _line("error", None), _line("holds", "bb")], "holds_variable"),
    ([_line("differs", "aa"), _line("holds", "aa")], "differs"),
    ([_line("differs", None), _line("holds", "bb")], "differs"),
    ([_line("differs", "aa"), _line("holds", None)], "differs"),
    ([_line("differs", "aa"), _line("differs", "bb"), _line("holds", "aa")], "differs"),
    ([_line("differs", "bb"), _line("differs", "aa"), _line("holds", "aa")], "differs"),
    ([_line("differs", "aa"), _line("differs", "bb"), _line("holds", "cc")], "holds_variable"),
    ([_line("differs", "aa")], "undecided"),
    ([_line("differs", "aa"), _line("differs", "bb")], "undecided"),
    ([_line("differs", t) for t in ("aa", "bb", "cc", "dd")], "differs"),
    ([_line("error", None)], "error"),
])
def test_the_attribution_rule(lines, outcome):
    held = lambda line: (line.get("l3_verdicts") or {}).get("p") == "holds"  # noqa: E731
    assert l3_corpus.attribute(lines, held)[0] == outcome


class _Source:
    def __init__(self, entries):
        self._src_vol = entries


class _OnDevice:
    """An array as CuPy holds it: ``.get()`` copies it to the host."""

    def __init__(self, array):
        self._array = array

    def get(self):
        return self._array


class _Tensor:
    """An array as torch holds it: ``.detach().cpu().numpy()`` copies it to the host."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def _tables(amps=(1 + 2j, 3 - 4j), on_device=None):
    from meep_gpu.sources import SrcVolTable  # noqa: PLC0415

    ix, iy, iz = (numpy.array(v, dtype=numpy.int64) for v in ([1, 2], [0, 0], [3, 4]))
    amp = numpy.array(amps, dtype=numpy.complex128)
    if on_device:
        wrap = _OnDevice if on_device == "cupy" else _Tensor
        ix, amp = wrap(ix), wrap(amp)
    driver = type("Driver", (), {})()
    driver._sources = [_Source((SrcVolTable(ix, iy, iz, amp, 2),)), _Source(())]
    return l3_row.source_tables(driver)


def test_the_row_records_its_source_tables_and_its_own_digests():
    # The deposit tables' digest: stable, moved by one amplitude, read through .get().
    first = _tables()
    assert first == _tables() == _tables(on_device="cupy") == _tables(on_device="torch")
    assert [s["entries"] for s in first["per_source"]] == [1, 0]
    moved = _tables(amps=(1 + 2j, complex(3, -numpy.nextafter(4.0, 5.0))))
    assert moved["sha256"] != first["sha256"]
    assert moved["per_source"][1] == first["per_source"][1]
    unbound = type("Driver", (), {})()
    unbound._sources = [_Source(None)]
    with pytest.raises(RuntimeError, match="not bound"):
        l3_row.source_tables(unbound)

    # The run's own digests, on the Tier A fixture: every compared step, every component,
    # stored and reader, equal to what the judge compared (the fixture holds).
    judged, steps, document = _fixture_judge(keep_digests=True)
    assert l3_row.set_verdict(judged, "ref") == "holds"
    own = judged[l3_row.RUN_DIGESTS_KEY]
    assert sorted(own, key=int) == [str(step) for step in steps]
    for step in steps:
        assert set(own[str(step)]) == set(identity.COMPONENTS)
        for component, (stored_digest, reader_digest) in own[str(step)].items():
            assert stored_digest == document["stored"][str(step)][component]
            assert reader_digest == document["reader"][str(step)][component]


def test_the_lift_s_solves_are_watched_and_passed_through():
    class Edata:
        freq, group_velocity = 0.15, 0.2846

    class Sim:
        def get_eigenmode(self, frequency, *args, **kwargs):
            return Edata()

    sim, sink = Sim(), []
    restore = l3_row.watch_solves(sim, sink)
    got = sim.get_eigenmode(0.15, "x", band=1)
    assert isinstance(got, Edata)
    assert sink == [{"freq": (0.15).hex(), "group_velocity": (0.2846).hex()}]
    restore()
    assert "get_eigenmode" not in vars(sim) and len(sink) == 1
    sim.get_eigenmode(0.15)
    assert len(sink) == 1

    # A value that does not read as a float is recorded as such, and the solve still returns.
    Edata.group_velocity = "not a number"
    restore = l3_row.watch_solves(sim, sink)
    assert isinstance(sim.get_eigenmode(0.15), Edata)
    restore()
    assert sink[-1]["error"].startswith("ValueError")


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
