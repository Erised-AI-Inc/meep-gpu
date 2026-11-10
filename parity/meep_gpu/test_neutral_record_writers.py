"""What the corpus, census, identity and stress harnesses WRITE names no machine or account,
nor do the gates and probes that write their own files.

Every record a run writes leaves the machine that ran it, so each writer below spells
its interpreter, its paths, a child's stderr and its log lines through ``neutral_ids``
(labels such as ``<home>``, this machine's names as its machine id). Each test runs a
writer's own record-building code on a machine whose login, host name, home and
temporary root are known tokens, and checks that nothing it wrote still names them
(``neutral_ids.residue`` empty, no token in the serialized record). The readers that
compare or execute a recorded interpreter are held beside them: they compare recorded
spellings with recorded spellings and execute the path the labels resolve to here.

No MEEP and no device are needed, except where a test says so: MEEP's terminal calls
are stubbed, and every child process is replaced by a fake that answers as a child would.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _entry in (str(HERE), str(API_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import neutral_ids  # noqa: E402

LOGIN = "someone"
HOST = "labbox7"
SECRET = bytes.fromhex("00112233445566778899aabbccddeeff")


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A machine whose login, host name, home, temporary root and interpreter are tokens."""
    home = tmp_path / "home" / LOGIN
    tmp = tmp_path / "scratchtmp"
    home.mkdir(parents=True)
    tmp.mkdir()
    python = home / "envs" / "ref" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr(neutral_ids.tempfile, "gettempdir", lambda: str(tmp))
    monkeypatch.setattr(neutral_ids.os.path, "expanduser",
                        lambda p: p.replace("~", str(home), 1) if p.startswith("~") else p)
    monkeypatch.setattr(neutral_ids, "_machine_secret", lambda: (SECRET, "/etc/machine-id"))
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: f"{HOST}.example.org")
    monkeypatch.setattr(neutral_ids, "_node", lambda: HOST)
    monkeypatch.setattr(neutral_ids, "login", lambda: LOGIN)
    monkeypatch.setattr(sys, "executable", str(python))
    neutral_ids.reset()
    yield {"home": str(home), "tmp": str(tmp), "python": str(python), "root": tmp_path}
    neutral_ids.reset()


def assert_neutral(obj, machine) -> None:
    """Nothing in ``obj`` names the fixture machine: no residue, and no token in its text."""
    found = neutral_ids.residue(obj)
    assert found == [], found
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    for token in (LOGIN, HOST, machine["home"]):
        assert token not in text, (token, text[:3000])


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _leak(machine) -> str:
    """A child's stderr that names a home path and this host, as a traceback does."""
    return (f'Traceback (most recent call last):\n  File "{machine["home"]}/envs/ref/lib/'
            f'site.py", line 1, in <module>\nRuntimeError: no data on {HOST} under '
            f'{machine["home"]}/meep/data\n')


def _completed(command, returncode=1, stderr="", stdout=b""):
    return subprocess.CompletedProcess(command, returncode, stdout,
                                       stderr.encode("utf-8") if isinstance(stderr, str)
                                       else stderr)


# --- a stand-in MEEP for the corpus captures ------------------------------------------

def _stub_meep():
    mp = types.ModuleType("meep")
    mp.__version__ = "0.0-stand-in"

    class Vector3:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Simulation:
        def __init__(self, cell_size=None, resolution=10, **_kwargs):
            self.cell_size = cell_size or Vector3(1.0, 1.0, 0.0)
            self.resolution = resolution
            self.Courant = 0.5
            self.k_point = False

        def run(self, *args, **kwargs):
            raise AssertionError("the capture did not stub run")

        def init_sim(self):
            raise AssertionError("the capture did not stub init_sim")

        def set_boundary(self, side, direction, condition):
            raise AssertionError("the capture did not stub set_boundary")

    mp.Vector3, mp.Simulation = Vector3, Simulation
    mp.is_single_precision = lambda: True
    return mp


CAPTURED_SCRIPT = "import meep as mp\nsim = mp.Simulation()\nsim.run(until=1)\n"


def _failing_script(machine) -> str:
    return (f"raise RuntimeError('cannot open {machine['home']}/meep/data/x.h5 on "
            f"{HOST}')\n")


# ======================================================================================
# survey_meep_examples: the capture every corpus record is built on, and the parent
# ======================================================================================

def test_the_example_capture_spells_the_interpreter_and_an_error_neutrally(machine, tmp_path,
                                                                           monkeypatch):
    from parity.meep_gpu import survey_meep_examples as survey

    monkeypatch.setitem(sys.modules, "meep", _stub_meep())
    examples = Path(machine["home"]) / "meep" / "examples"
    examples.mkdir(parents=True)
    (examples / "captured.py").write_text(CAPTURED_SCRIPT, encoding="utf-8")
    (examples / "failing.py").write_text(_failing_script(machine), encoding="utf-8")
    namespace: dict = {}
    exec(compile(survey.CHILD_PREAMBLE, "<survey-child>", "exec"), namespace)  # noqa: S102
    record, sim, restore = namespace["_capture"](str(examples / "captured.py"))
    restore()
    assert sim is not None and record["outcome"] == "captured", record
    assert record["python"] == "<home>/envs/ref/bin/python", record
    assert_neutral(record, machine)
    record, sim, restore = namespace["_capture"](str(examples / "failing.py"))
    restore()
    assert record["outcome"] == "error" and "<home>/meep/data/x.h5" in record["error"], record
    assert_neutral(record, machine)


def test_the_example_survey_writes_a_dead_childs_tail_neutrally(machine, monkeypatch):
    from parity.meep_gpu import survey_meep_examples as survey

    examples = Path(machine["home"]) / "meep" / "python" / "examples"
    examples.mkdir(parents=True)
    (examples / "dies.py").write_text("pass\n", encoding="utf-8")
    (examples / "writes.py").write_text("pass\n", encoding="utf-8")
    out = Path(machine["home"]) / "survey"

    def fake_run(command, **_kwargs):
        if Path(command[command.index("--one") + 1]).name == "dies.py":
            return _completed(command, 1, _leak(machine))
        # A child (of an older capture) whose record names the interpreter as it ran.
        Path(command[command.index("--out-json") + 1]).write_text(json.dumps(
            {"script": "writes.py", "python": machine["python"], "has_simulation": False,
             "outcome": "error", "error": f"open {machine['home']}/x on {HOST}"}))
        return _completed(command, 0, "")

    monkeypatch.setattr(survey.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["survey", "--examples", str(examples), "--out", str(out)])
    assert survey.main() == 0
    rows = _jsonl(out / "survey.jsonl")
    assert rows[0]["outcome"] == "child_died" and rows[0]["stderr_tail"], rows
    assert rows[1]["python"] == "<home>/envs/ref/bin/python", rows
    assert_neutral(rows, machine)
    assert_neutral((out / "per_example" / "dies.stderr.txt").read_text(), machine)
    assert_neutral(json.loads((out / "summary.json").read_text()), machine)


# ======================================================================================
# sweep_corpus_lift_parity: a child's row, a dead child's row, the log
# ======================================================================================

def test_a_sweep_childs_row_is_written_neutrally(machine, tmp_path):
    from parity.meep_gpu import sweep_corpus_lift_parity as sweep

    out = tmp_path / "row.json"
    sweep.write_record(str(out), {"script": "a.py", "python": machine["python"],
                                  "lift_error": f"open {machine['home']}/x.h5 on {HOST}"})
    record = json.loads(out.read_text())
    assert record["python"] == "<home>/envs/ref/bin/python"
    assert_neutral(record, machine)


def test_the_sweep_progress_log_is_written_neutrally(machine, tmp_path):
    from parity.meep_gpu import sweep_corpus_lift_parity as sweep

    log = tmp_path / "lift.progress.log"
    sweep.ProgressLog(str(log), "a.py")(f"lift failed: {machine['home']}/x.h5 on {HOST}")
    assert "<home>/x.h5" in log.read_text()
    assert_neutral(log.read_text(), machine)


def test_the_sweep_writes_a_dead_childs_row_and_its_log_neutrally(machine, monkeypatch):
    from parity.meep_gpu import sweep_corpus_lift_parity as sweep

    home = Path(machine["home"])
    examples = home / "meep" / "python" / "examples"
    examples.mkdir(parents=True)
    (examples / "dies.py").write_text("pass\n", encoding="utf-8")
    survey = home / "survey.jsonl"
    survey.write_text("".join(json.dumps({"script": name, "has_simulation": True,
                                          "supported": True}) + "\n"
                              for name in ("dies.py", "writes.py")), encoding="utf-8")
    out = home / "sweep"

    def fake_run(command, **_kwargs):
        if Path(command[command.index("--one") + 1]).name == "dies.py":
            return _completed(command, 1, _leak(machine))
        Path(command[command.index("--out-json") + 1]).write_text(json.dumps(
            {"script": "writes.py", "python": machine["python"], "lifts": False,
             "lift_error": f"open {machine['home']}/x on {HOST}"}))
        return _completed(command, 0, "")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sweep", "--examples", str(examples), "--survey",
                                      str(survey), "--out", str(out), "--retries", "0"])
    assert sweep.main() == 0
    rows = _jsonl(out / "lift.jsonl")
    assert rows[0]["outcome"] == "child_died" and rows[0]["stderr_tail"], rows
    assert rows[1]["python"] == "<home>/envs/ref/bin/python", rows
    assert_neutral(rows, machine)
    assert_neutral((out / "per_script_lift" / "dies.stderr.txt").read_text(), machine)
    log = (out / "lift.log").read_text()
    assert "<home>/sweep/lift.jsonl" in log
    assert_neutral(log, machine)


# ======================================================================================
# survey_meep_tests: the case capture, the amp_func_file notes, the batch and the parent
# ======================================================================================

def _test_module(machine, directory: Path) -> Path:
    module = directory / "test_stand_in.py"
    module.write_text(textwrap.dedent(f'''
        import unittest
        import meep as mp


        class TestStandIn(unittest.TestCase):
            def test_runs(self):
                sim = mp.Simulation()
                sim.run(until=1)

            def test_fails(self):
                raise RuntimeError("cannot open {machine['home']}/meep/data/x.h5 on {HOST}")
    '''), encoding="utf-8")
    return module


def test_the_test_case_capture_spells_the_interpreter_and_an_error_neutrally(machine,
                                                                             monkeypatch):
    from parity.meep_gpu import survey_meep_tests as harness

    monkeypatch.setitem(sys.modules, "meep", _stub_meep())
    tests = Path(machine["home"]) / "meep" / "python" / "tests"
    tests.mkdir(parents=True)
    module_path = _test_module(machine, tests)
    namespace = harness.build_child_namespace()
    module = namespace["_import_module"](str(module_path))
    try:
        record, sim, restore, _case = namespace["_run_case"](
            module, str(module_path), "TestStandIn", "test_runs", 30.0)
        restore()
        assert sim is not None and record["outcome"] == "captured", record
        assert record["python"] == "<home>/envs/ref/bin/python", record
        assert_neutral(record, machine)
        record, sim, restore, _case = namespace["_run_case"](
            module, str(module_path), "TestStandIn", "test_fails", 30.0)
        restore()
        assert record["outcome"] == "error" and "<home>/meep/data/x.h5" in record["error"]
        assert_neutral(record, machine)
    finally:
        sys.modules.pop("test_stand_in", None)


def test_an_amp_func_file_note_names_the_file_not_the_directory(machine):
    from parity.meep_gpu import survey_meep_tests as harness

    template = ("sources[0]: amp_func_file {!r} read here (h5py) and handed to the lift as "
                "amp_data(10,)")
    spellings = [f"{machine['home']}/meep/python/tests/data/amp_func_file:amp_data",
                 "/elsewhere/meep/python/tests/data/amp_func_file:amp_data",
                 "amp_func_file:amp_data"]
    notes = [harness.neutral_amp_notes([template.format(spec)]) for spec in spellings]
    assert notes[0] == notes[1] == notes[2], notes
    assert "'amp_func_file:amp_data'" in notes[0][0], notes
    assert_neutral(notes, machine)
    # A dataset path keeps its own slashes: only the file is reduced to its name.
    assert "'f.h5:group/data'" in harness.neutral_amp_notes(
        [template.format("/x/y/f.h5:group/data")])[0]
    assert harness.neutral_amp_notes(None) is None


def test_the_test_survey_childs_batch_rows_are_written_neutrally(machine, tmp_path):
    from parity.meep_gpu import survey_meep_tests as harness

    module = Path(machine["home"]) / "test_unimportable.py"
    module.write_text(f"raise ImportError('needs {machine['home']}/lib on {HOST}')\n",
                      encoding="utf-8")
    out = tmp_path / "batch.jsonl"
    namespace = harness.build_child_namespace()
    args = types.SimpleNamespace(case_timeout=30.0)
    harness.run_child_module(namespace, str(module), "lift", out, args, lambda message: None)
    rows = _jsonl(out)
    assert rows and rows[0]["outcome"] == "module_import_failed", rows
    assert_neutral(rows, machine)


def test_the_test_survey_writes_a_dead_modules_row_and_its_log_neutrally(machine,
                                                                        monkeypatch):
    from parity.meep_gpu import survey_meep_tests as harness

    home = Path(machine["home"])
    tests = home / "meep" / "python" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_dies.py").write_text("pass\n", encoding="utf-8")
    out = home / "tests_survey"
    out.mkdir()
    (out / "inventory.jsonl").write_text(json.dumps(
        {"module": "test_dies.py", "constructs_simulation": True,
         "test_methods": ["TestDies.test_one"]}) + "\n", encoding="utf-8")
    def fake_run(command, **_kwargs):
        # A child (of an older capture) that wrote one raw row, then died.
        Path(command[command.index("--child-out") + 1]).write_text(json.dumps(
            {"module": "test_dies.py", "case": "TestDies.test_one", "stage": "lift",
             "python": machine["python"], "lifts": False,
             "lift_error": f"open {machine['home']}/x on {HOST}"}) + "\n")
        return _completed(command, 1, _leak(machine))

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["survey", "--stage", "lift", "--tests", str(tests),
                                      "--out", str(out)])
    assert harness.main() == 0
    rows = _jsonl(out / "lift.jsonl")
    assert rows[0]["python"] == "<home>/envs/ref/bin/python", rows
    assert rows[-1]["case"] == "<module tail>" and rows[-1]["stderr_tail"], rows
    assert_neutral(rows, machine)
    assert_neutral((out / "batches" / "test_dies.py.lift.stderr.txt").read_text(), machine)
    assert_neutral((out / "lift.log").read_text(), machine)


# ======================================================================================
# measure_predicate_coverage: which interpreter a row runs under, and what it writes
# ======================================================================================

def _census():
    import measure_predicate_coverage as census  # noqa: PLC0415

    return census


def test_a_row_runs_under_the_interpreter_its_recorded_spelling_resolves_to(machine):
    census = _census()
    home = machine["home"]
    stock = "<home>/envs/ref/bin/python"
    remap = {"<home>/envs/moved/bin/python": "/opt/moved/python",
             "<home>/envs/old/bin/python": "/opt/old/python"}
    # The stock leg's rows run under this process's own interpreter.
    assert census.interpreter_for(stock, stock, remap, "/own/python") == "/own/python"
    # A sibling's recorded spelling runs as its labels resolve on this machine.
    assert census.interpreter_for("<home>/envs/side/bin/python", stock, remap,
                                  "/own/python") == f"{home}/envs/side/bin/python"
    # The remap is keyed by the RECORDED spelling, never by the path it resolves to.
    assert census.interpreter_for("<home>/envs/moved/bin/python", stock, remap,
                                  "/own/python") == "/opt/moved/python"
    assert census.interpreter_for("<home>/envs/moved/bin/python", stock,
                                  {f"{home}/envs/moved/bin/python": "/opt/moved/python"},
                                  "/own/python") == f"{home}/envs/moved/bin/python"
    # A record cut before the labels: compared as recorded, run as recorded, and remapped
    # by its own spelling or by its labelled one (the one the refusal note names).
    old = f"{home}/envs/old/bin/python"
    assert census.interpreter_for(old, old, {}, "/own/python") == "/own/python"
    assert census.interpreter_for(old, stock, {}, "/own/python") == old
    assert census.interpreter_for(old, stock, remap, "/own/python") == "/opt/old/python"
    assert census.interpreter_for(old, stock, {old: "/opt/raw/python"},
                                  "/own/python") == "/opt/raw/python"


def test_the_examples_leg_chooses_each_scripts_interpreter_from_the_first_record(machine):
    census = _census()
    lifts = [("a/stock/lift.jsonl", [{"script": "a.py", "python": "<home>/envs/ref/bin/python"},
                                     {"script": "b.py", "python": "<home>/envs/ref/bin/python"}]),
             ("a/side/lift.jsonl", [{"script": "b.py", "python": "<home>/envs/side/bin/python"},
                                    {"script": "c.py", "python": "<home>/envs/side/bin/python"}])]
    chosen = census.choose_examples_interpreters(lifts, {}, own="/own/python")
    assert chosen == {
        "a.py": ("/own/python", "stock", "<home>/envs/ref/bin/python"),
        "b.py": ("/own/python", "stock", "<home>/envs/ref/bin/python"),
        "c.py": (f"{machine['home']}/envs/side/bin/python", "side",
                 "<home>/envs/side/bin/python")}
    with pytest.raises(SystemExit, match="exactly one"):
        census.choose_examples_interpreters(
            [("a/stock/lift.jsonl", [{"script": "a.py", "python": "x"},
                                     {"script": "b.py", "python": "y"}])], {})


def _write_lifts(home: Path, rows_by_leg):
    paths = []
    for leg, rows in rows_by_leg:
        path = home / "lifts" / leg / "lift.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps({**row, "accepted": True}) + "\n" for row in rows),
                        encoding="utf-8")
        paths.append(str(path))
    return tuple(paths)


def test_the_census_examples_leg_runs_and_writes_each_row_neutrally(machine, monkeypatch):
    census = _census()
    home = Path(machine["home"])
    for env in ("side", "remap_target"):
        target = home / "envs" / env / "bin" / "python"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    old_missing = f"{home}/envs/gone/bin/python"  # a record cut before the labels
    lifts = _write_lifts(home, [
        ("stock", [{"script": "a.py", "python": "<home>/envs/ref/bin/python"},
                   {"script": "dies.py", "python": "<home>/envs/ref/bin/python"}]),
        ("side", [{"script": "b.py", "python": "<home>/envs/side/bin/python"}]),
        ("old", [{"script": "c.py", "python": old_missing}]),
        ("moved", [{"script": "d.py", "python": "<home>/envs/moved/bin/python"}]),
        ("lost", [{"script": "e.py", "python": "<home>/envs/lost/bin/python"}])])
    examples = home / "meep" / "python" / "examples"
    examples.mkdir(parents=True)
    out = home / "census"
    launched = {}

    def fake_run(command, **_kwargs):
        command = [str(part) for part in command]
        if "--runtime-preflight" in command:
            return _completed(command, 0, "", (census.RUNTIME_PREFLIGHT_PREFIX + "[]\n").encode())
        script = Path(command[command.index("--child-script") + 1]).name
        launched[script] = command[0]
        if script == "dies.py":
            return _completed(command, 1, _leak(machine))
        record_path = Path(command[command.index("--out-json") + 1])
        record_path.write_text(json.dumps(
            {"row": script, "measured": True, "python": "<home>/envs/ref/bin/python",
             "note": f"lifted from {examples}/{script} on {HOST}"}), encoding="utf-8")
        return _completed(command, 0, "")

    monkeypatch.setattr(census, "EXAMPLES_LIFTS", lifts)
    monkeypatch.setattr(census, "EXAMPLES_DIR", str(examples))
    monkeypatch.setattr(census.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "census", "--leg", "examples", "--battery", "cuda_predicate_battery", "--out", str(out),
        "--interpreter", f"<home>/envs/moved/bin/python={home}/envs/remap_target/bin/python",
        "--interpreter", "<home>/envs/lost/bin/python=/nonexistent/remapped/python"])
    assert census.main() == 0
    # The executed interpreters: this process's, the sibling's resolved spelling, the remap.
    assert launched == {"a.py": machine["python"], "dies.py": machine["python"],
                        "b.py": f"{home}/envs/side/bin/python",
                        "d.py": f"{home}/envs/remap_target/bin/python"}, launched
    rows = {row["row"]: row for row in _jsonl(out / "examples.jsonl")}
    assert rows["a.py"]["interpreter"] == "<home>/envs/ref/bin/python"
    assert rows["b.py"]["interpreter"] == "<home>/envs/side/bin/python"
    assert rows["d.py"]["interpreter"] == "<home>/envs/remap_target/bin/python"
    assert rows["dies.py"]["stderr_tail"] and rows["dies.py"]["measured"] is False
    # The refusal names the interpreter by its labelled spelling, the key the remap takes.
    assert "--interpreter <home>/envs/gone/bin/python=PATH" in rows["c.py"]["note"], rows
    # A remap whose PATH is missing too: the note still names the RECORDED key.
    assert "--interpreter <home>/envs/lost/bin/python=PATH" in rows["e.py"]["note"], rows
    assert "interpreter missing: /nonexistent/remapped/python" in rows["e.py"]["note"]
    assert_neutral(list(rows.values()), machine)
    assert_neutral((out / "examples.log").read_text(), machine)
    digests = json.loads((out / "subject_digests_examples.json").read_text())
    assert digests["battery"].startswith("<staging-root>/"), digests["battery"]
    assert_neutral(digests, machine)


def test_a_census_childs_per_row_file_is_written_neutrally(machine, tmp_path):
    census = _census()
    out = tmp_path / "row.json"
    census._write_child(str(out), [{"row": "T.t", "python": machine["python"],
                                    "lift_error": f"{machine['home']}/x on {HOST}"}])
    rows = json.loads(out.read_text())
    assert rows[0]["python"] == "<home>/envs/ref/bin/python"
    assert_neutral(rows, machine)


def test_a_runtime_refusal_names_the_interpreter_and_its_reasons_neutrally(machine):
    census = _census()
    record = census.runtime_refusal(
        "mie_scattering.py", machine["python"], "monitorfix_sigma",
        [f"runtime preflight under {machine['python']} exited 1: no torch on {HOST}"])
    assert "<home>/envs/ref/bin/python" in record["note"], record
    assert_neutral(record, machine)


def test_a_childs_stderr_is_neutral_before_it_is_cut(machine, tmp_path):
    census = _census()
    # The cut lands five characters before ``/home/<login>``: cut first, the account
    # would survive in a fragment no label recognises any more.
    keep = 5 + len(f"/home/{LOGIN}/")
    payload = f"{machine['home']}/" + "b" * (1200 - keep)
    assert LOGIN in payload[-1200:]
    tail = census._neutral_tail(payload, 1200)
    assert_neutral(tail, machine)
    fake = tmp_path / "python"
    fake.write_text(f"#!/bin/sh\nprintf '%s' '{machine['home']}/{'c' * (600 - keep)}' >&2\n"
                    "exit 3\n", encoding="utf-8")
    fake.chmod(0o755)
    reasons = census.preflight_runtime(str(fake), "predicate_battery", dict(os.environ))
    assert len(reasons) == 1 and "exited 3" in reasons[0], reasons
    assert LOGIN not in reasons[0], reasons


# ======================================================================================
# probe_h_to_d_seam and probe_h_to_d_withdraw_order
# ======================================================================================

def test_the_seam_join_writes_a_census_rows_interpreter_under_its_label(machine):
    import probe_h_to_d_seam as probe  # noqa: PLC0415

    for recorded in (machine["python"], "<home>/envs/ref/bin/python"):
        entry = probe.join_entry("examples:a.py", {"leg": "examples", "row": "a.py",
                                                   "interpreter": recorded}, {}, {})
        assert entry["interpreter"] == "<home>/envs/ref/bin/python", entry
        assert_neutral(entry, machine)


def test_a_recovered_seam_case_is_appended_and_logged_neutrally(machine, tmp_path,
                                                               monkeypatch):
    import probe_h_to_d_seam as probe  # noqa: PLC0415

    out = tmp_path / "seam"
    (out / "per_row_tests").mkdir(parents=True)
    (out / "subject_digests_tests.json").write_text(json.dumps(
        {"manifest_sha256": "m" * 64, "battery_sha256": "b" * 64}), encoding="utf-8")
    lines = []

    def fake_run(command, **_kwargs):
        solo = Path(command[command.index("--out-json") + 1])
        solo.write_text(json.dumps([{"row": "T.t", "measured": False,
                                     "lift_error": f"{machine['home']}/x on {HOST}"}]))
        return _completed(command, 0, "")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    assert probe.recover_case(out, "test_x.py:T.t", dict(os.environ), lines.append) == 0
    assert_neutral(_jsonl(out / "tests.jsonl"), machine)
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda command, **_kw: _completed(command, 1, _leak(machine)))
    assert probe.recover_case(out, "test_x.py:T.u", dict(os.environ), lines.append) == 4
    # The probe's own log() neutralizes; here the tail it is handed must already be.
    assert_neutral(lines[-1], machine)


def test_the_seam_probe_log_is_neutral(machine, monkeypatch):
    import probe_h_to_d_seam as probe  # noqa: PLC0415

    def fake_run(command, **_kw):
        failed = any(str(part).endswith("match_param_rows.py") for part in command)
        return type("Done", (), {"returncode": 3 if failed else 0})()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    out = Path(machine["home"]) / "seam_probe"
    assert probe.main(["--out", str(out), "--leg", "examples"]) == 3
    log = (out / "probe.log").read_text()
    assert "out <home>/seam_probe" in log, log
    assert_neutral(log, machine)


def _withdraw_seam(tmp_path: Path) -> Path:
    seam = tmp_path / "seam"
    seam.mkdir()
    (seam / "summary.json").write_text(json.dumps(
        {"sources": {"withdraw_rows": ["examples:a.py", "tests:T.t"]}}), encoding="utf-8")
    (seam / "examples.jsonl").write_text(json.dumps(
        {"row": "a.py", "interpreter": "<home>/envs/side/bin/python",
         "lift_record": "side"}) + "\n", encoding="utf-8")
    (seam / "tests.jsonl").write_text(json.dumps(
        {"row": "T.t", "module": "test_x.py"}) + "\n", encoding="utf-8")
    return seam


def test_the_withdraw_basis_keeps_the_recorded_spelling(machine, tmp_path):
    import probe_h_to_d_withdraw_order as order  # noqa: PLC0415

    rows = {row["label"]: row for row in order.basis(_withdraw_seam(tmp_path))}
    assert rows["examples:a.py"]["interpreter"] == "<home>/envs/side/bin/python"
    # A row whose record names none runs under this interpreter, recorded neutrally.
    assert rows["tests:T.t"]["interpreter"] == "<home>/envs/ref/bin/python"
    assert_neutral(list(rows.values()), machine)


def test_the_withdraw_probe_executes_the_resolved_path_and_records_the_spelling(
        machine, tmp_path, monkeypatch):
    import measure_predicate_coverage as census  # noqa: PLC0415
    import probe_h_to_d_withdraw_order as order  # noqa: PLC0415

    home = Path(machine["home"])
    examples = home / "meep" / "python" / "examples"
    examples.mkdir(parents=True)
    commands = []

    def fake_run(command, **_kwargs):
        command = [str(part) for part in command]
        commands.append(command)
        if "--child-cases" in command:  # the tests row's child dies, its stderr a path
            return _completed(command, 1, _leak(machine))
        record = Path(command[command.index("--out-json") + 1])
        record.write_text(json.dumps({"row": "a.py", "measured": False,
                                      "lift_error": f"{home}/x.h5 missing on {HOST}"}))
        return _completed(command, 0, "")

    monkeypatch.setattr(census, "EXAMPLES_DIR", str(examples))
    monkeypatch.setattr(census, "TESTS_DIR", str(home / "meep" / "python" / "tests"))
    monkeypatch.setattr(order.subprocess, "run", fake_run)
    out = home / "withdraw"
    monkeypatch.setattr(sys, "argv", ["order", "--out", str(out),
                                      "--seam", str(_withdraw_seam(tmp_path))])
    order.main()
    assert sorted(command[0] for command in commands) == sorted(
        [f"{home}/envs/side/bin/python", machine["python"]]), commands
    rows = {row["label"]: row for row in _jsonl(out / "rows.jsonl")}
    assert rows["examples:a.py"]["interpreter"] == "<home>/envs/side/bin/python"
    assert rows["tests:T.t"]["interpreter"] == "<home>/envs/ref/bin/python"
    assert rows["tests:T.t"]["stderr_tail"], rows
    assert_neutral(list(rows.values()), machine)
    # The record the PARENT writes for a child that died (a live child writes its own
    # through measure_predicate_coverage._write_child).
    assert_neutral(json.loads((out / "per_row" / "tests__T_t.json").read_text()), machine)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["probe"].startswith("<staging-root>/"), summary["probe"]
    assert_neutral(summary, machine)
    assert_neutral((out / "probe.log").read_text(), machine)


# ======================================================================================
# lift_host: the rows a gate records and the capability probe
# ======================================================================================

def test_a_lift_row_records_both_interpreters_under_their_labels(machine):
    import lift_host  # noqa: PLC0415

    fields = lift_host.row_fields({"interpreter": f"{machine['home']}/envs/old/bin/python"}, {})
    assert fields == {"interpreter": "<home>/envs/ref/bin/python",
                      "recorded_interpreter": "<home>/envs/old/bin/python",
                      "lift_record": None, "lift_needs": []}
    assert_neutral(fields, machine)


def test_an_unreadable_capability_probe_is_named_neutrally(machine):
    import lift_host  # noqa: PLC0415

    silent = Path(machine["home"]) / "envs" / "quiet" / "bin" / "python"
    silent.parent.mkdir(parents=True)
    silent.write_text(f"#!/bin/sh\necho 'no answer from {machine['home']} on {HOST}' >&2\n"
                      "exit 1\n", encoding="utf-8")
    silent.chmod(0o755)
    probed = lift_host.probe(["gdstk"], executable=str(silent))
    assert probed["interpreter"] == "<home>/envs/quiet/bin/python"
    assert "printed no answer" in probed["unreadable"], probed
    assert_neutral(probed, machine)
    absent = lift_host.probe(["gdstk"], executable=f"{machine['home']}/envs/absent/python")
    assert "did not finish" in absent["unreadable"], absent
    assert_neutral(absent, machine)


# ======================================================================================
# meep_identity_corpus: the manifest reads any spelling and writes one
# ======================================================================================

def _row(**extra):
    record = {"parity": "MEASURED", "parity_rel_l2": 1e-6, "grid_cells": 40,
              "grid_shape": [40, 1, 1], "driver_steps": 200, "cpu_meep_steps": 200,
              "time_matched": True,
              "capped": {"resolution_original": 20.0, "resolution_used": 10.0,
                         "resolution_capped": True, "steps_planned": 200}, "facts": {}}
    record.update(extra)
    return record


def test_the_corpus_manifest_is_neutral_whatever_spelling_its_legs_recorded(machine, tmp_path):
    from parity.meep_gpu import meep_identity_corpus as corpus

    note = ("sources[0]: amp_func_file {!r} read here (h5py) and handed to the lift as "
            "amp_data(10,)")
    old = _row(module="test_x.py", case="TestX.test_amp", python=machine["python"],
               amp_func_file_substituted=[note.format(
                   f"{machine['home']}/meep/python/tests/data/amp_func_file:amp_data")])
    new = _row(module="test_x.py", case="TestX.test_amp",
               python="<home>/envs/ref/bin/python",
               amp_func_file_substituted=[note.format("amp_func_file:amp_data")])
    rows_by_leg = {"tests": [old], corpus.RESURVEYS[0]: [new]}
    lifts = {corpus.RESURVEYS[0]: [{"module": "test_x.py", "case": "TestX.test_gone",
                                    "has_simulation": False, "outcome": "error",
                                    "error": f"{machine['home']}/data.h5 on {HOST}"}]}
    for leg, directory in corpus.LEGS.items():
        path = tmp_path / directory
        path.mkdir(parents=True)
        (path / "parity.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows_by_leg.get(leg, [])), encoding="utf-8")
        if leg in corpus.RESURVEYS:
            (path / "lift.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in lifts.get(leg, [])), encoding="utf-8")
    # The re-survey measured the same run: one spelling of one file, so no refusal.
    document = corpus.manifest_rows(tmp_path)
    row = document["rows"][0]
    assert row["python"] == "<home>/envs/ref/bin/python"
    assert row["table2"]["amp_func_file_substituted"] == new["amp_func_file_substituted"]
    assert document["remeasured"][0]["label"] == "t.test_x.TestX.test_amp"
    assert_neutral(document, machine)


# ======================================================================================
# meep_identity_tier_c, _smoke, _context_l2: no path to MEEP; a child's report neutral
# ======================================================================================

def test_a_tier_c_childs_report_is_neutral(machine, monkeypatch):
    from parity.meep_gpu import meep_identity_tier_c as tier_c

    class FakeChild:
        returncode = 0

        def __init__(self, command, stdout=None, stderr=None, text=None, cwd=None):
            stderr.write(_leak(machine))
            stderr.flush()
            report = {"case": "x", "error": f"open {machine['home']}/data.h5 on {HOST}",
                      "meep": {"version": "0.0"},
                      "libm": {"path": f"{machine['home']}/envs/ref/lib/libm.so.6"}}
            self.stdout = iter([tier_c.MARK + json.dumps(report) + "\n"])

        def wait(self):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(tier_c.subprocess, "Popen", FakeChild)
    result = tier_c.run_child({"kind": "compare", "case": "x", "mode": "keep"})
    assert result["error"] and result["stderr_tail"], result
    assert_neutral(result, machine)

    class SilentChild(FakeChild):
        returncode = 3

        def __init__(self, command, stdout=None, stderr=None, text=None, cwd=None):
            stderr.write(_leak(machine))
            stderr.flush()
            self.stdout = iter([])

    monkeypatch.setattr(tier_c.subprocess, "Popen", SilentChild)
    result = tier_c.run_child({"kind": "compare", "case": "x", "mode": "keep"})
    assert "exited 3" in result["error"], result
    assert_neutral(result, machine)


@pytest.mark.requires_resource("meep")
def test_the_identity_records_name_meep_by_version_not_by_path(tmp_path):
    """Run in a child: each identity writer's MEEP block, with MEEP stopped at its first
    refusal (``is_single_precision`` answering False), carries no path."""
    pytest.importorskip("meep")
    code = textwrap.dedent(f'''
        import json, sys
        sys.path[:0] = [{str(HERE)!r}, {str(API_ROOT)!r}]
        import meep
        meep.is_single_precision = lambda: False
        from parity.meep_gpu import meep_identity_tier_c as tier_c
        from parity.meep_gpu import meep_identity_smoke as smoke
        from parity.meep_gpu import meep_identity_context_l2 as l2
        out = {{}}
        out["tier_c"] = tier_c.child_compare({{"case": "x", "mode": "keep",
                                               "out": {str(tmp_path / "c")!r}}})["meep"]
        out["smoke"] = smoke.child("x", "keep", {{}}, [], None)["meep"]
        l2.child("x", {str(tmp_path / "absent_cases.py")!r}, {str(tmp_path / "l2.json")!r})
        record = json.loads(open({str(tmp_path / "l2.json")!r}).read())
        out["l2"] = record["meep"]
        errors = [record.get("error")]
        print("BLOCKS=" + json.dumps(out))
        print("ERRORS=" + json.dumps(errors))
    ''')
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=600, cwd=str(tmp_path),
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                               "OMP_NUM_THREADS": "1"})
    line = [part for part in done.stdout.splitlines() if part.startswith("BLOCKS=")]
    assert line, done.stdout[-2000:] + done.stderr[-3000:]
    blocks = json.loads(line[-1][len("BLOCKS="):])
    for name, block in blocks.items():
        assert set(block) == {"version", "single_precision"}, (name, block)
    assert neutral_ids.residue(blocks) == []
    errors = json.loads([part for part in done.stdout.splitlines()
                         if part.startswith("ERRORS=")][-1][len("ERRORS="):])
    # The l2 row's error names the cases file it could not read, under its label.
    assert errors[0] and "absent_cases.py" in errors[0], errors
    assert neutral_ids.residue(errors) == [], errors


# ======================================================================================
# record_meep_identity_archive: sources, host, exclusion reasons, rows
# ======================================================================================

def test_the_tier_b_archive_records_its_sources_host_and_reasons_neutrally(machine,
                                                                          tmp_path):
    from parity.meep_gpu import record_meep_identity_archive as archive

    source = Path(machine["home"]) / "tier_c_out"
    (source / "keep").mkdir(parents=True)
    (source / "keep" / "row_a.error.json").write_text(json.dumps(
        {"case": "row_a", "error": f"FileNotFoundError: {machine['home']}/x.h5 on {HOST}"}))
    target = tmp_path / "tier_b"
    assert archive.main(["--archive", str(target), "--set", "mac_keep", "--mode", "keep",
                         "--source", str(source), "--host", HOST,
                         "--exclude", f"row_a=see {machine['home']}/notes.txt"]) == 0
    document = json.loads((target / archive.ARCHIVE_FILE).read_text())
    leaf = document["sets"]["mac_keep"]
    assert leaf["minted"]["sources"] == ["<home>/tier_c_out"], leaf["minted"]
    assert leaf["host"] == neutral_ids.machine_id(), leaf["host"]
    assert leaf["rows"]["row_a"]["excluded"] == "see <home>/notes.txt", leaf["rows"]
    assert_neutral(document, machine)
    with pytest.raises(ValueError):
        archive.parse_exclusions(["row_b="])
    assert archive.parse_exclusions([f"row_b=on {HOST} in {machine['home']}/x"]) == {
        "row_b": f"on {neutral_ids.machine_id()} in <home>/x"}
    assert archive.recorded_sources([str(source)]) == ["<home>/tier_c_out"]


# ======================================================================================
# capsule_identity: the manifest, the environment, the progress log
# ======================================================================================

def test_a_capsule_legs_manifest_and_log_are_neutral(machine, tmp_path, monkeypatch):
    import capsule_identity as capsule  # noqa: PLC0415

    monkeypatch.setenv("PYTHONPATH", f"{machine['home']}/src:{machine['home']}/lib")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", f"{machine['home']}/pycache")
    out = tmp_path / "leg"
    progress = capsule._start(str(out), "lift/x")
    environment = capsule._environment()
    assert environment["python"] == "<home>/envs/ref/bin/python"
    assert environment["environ"]["PYTHONPATH"] == "<home>/src:<home>/lib", environment
    assert_neutral(environment, machine)
    manifest = {"cases_file": f"{machine['home']}/cases.py",
                "meep": {"file": f"{machine['home']}/envs/ref/lib/meep/__init__.py"},
                "error": f"RuntimeError: {machine['home']}/x on {HOST}", **environment}
    progress.say(f"loaded {machine['home']}/capsule.capsule on {HOST}")
    capsule._finish(str(out), manifest, 1)
    assert_neutral(json.loads((out / "manifest.json").read_text()), machine)
    assert_neutral((out / "progress.log").read_text(), machine)


# ======================================================================================
# recut_metal_corpus: every record it writes, and the readers that open a recorded path
# ======================================================================================

def test_the_metal_corpus_records_are_neutral(machine, tmp_path):
    import recut_metal_corpus as corpus  # noqa: PLC0415

    home = Path(machine["home"])
    record = {"meep_root": str(home / "meep"), "python": machine["python"],
              "input_sha256": {"simulation_source": {"path": str(home / "meep" / "a.py"),
                                                     "sha256": "0" * 64}},
              "source_sha256": {str(home / "src" / "a.py"): "1" * 64},
              "reasons": [f"example child a.py exited 1: on {HOST}"]}
    corpus._atomic_json(tmp_path / "report.json", record)
    corpus._append_jsonl(tmp_path / "raw.jsonl", record)
    (tmp_path / "rows.jsonl").write_text(corpus._rows_text([record]))
    corpus._write_manifest(tmp_path / "source_sha256.txt", record["source_sha256"])
    corpus._progress(tmp_path / "progress.log", "a.py", f"read {home}/a.py on {HOST}", 0.0)
    assert_neutral(json.loads((tmp_path / "report.json").read_text()), machine)
    assert_neutral(_jsonl(tmp_path / "raw.jsonl"), machine)
    assert_neutral(_jsonl(tmp_path / "rows.jsonl"), machine)
    for name in ("source_sha256.txt", "progress.log"):
        assert_neutral((tmp_path / name).read_text(), machine)


def test_the_metal_corpus_reopens_a_recorded_path_on_this_machine(machine):
    import recut_metal_corpus as corpus  # noqa: PLC0415

    home = Path(machine["home"])
    source = home / "api" / "planner.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n", encoding="utf-8")
    digest = corpus.sha256(source)
    recorded = neutral_ids.path(str(source))
    assert recorded == "<home>/api/planner.py"
    assert corpus.verify_runtime_sources({recorded: digest}, root=home / "api") == []
    row = {"measured": True, "leg": "examples", "row": "a.py",
           "input_sha256": {"simulation_source": {"path": recorded, "sha256": digest}}}
    assert corpus.verify_row_inputs([row]) == []
    # An absolute path recorded before the labels is opened as it was.
    row["input_sha256"]["simulation_source"]["path"] = str(source)
    assert corpus.verify_row_inputs([row]) == []


def test_a_metal_corpus_childs_stderr_is_neutral_before_it_is_cut(machine, tmp_path):
    import recut_metal_corpus as corpus  # noqa: PLC0415

    keep = 5 + len(f"/home/{LOGIN}/")  # the cut lands before /home/<login>
    payload = f"{machine['home']}/" + "b" * (1600 - keep)
    assert LOGIN in payload[-1600:]
    code, tail = corpus._run_child(["/bin/sh", "-c", f"printf '%s' '{payload}' >&2; exit 2"],
                                   cwd=tmp_path, environment=dict(os.environ), timeout=60)
    assert code == 2 and tail.endswith("b" * 100), tail[-200:]
    assert_neutral(tail, machine)


# ======================================================================================
# stress_cuda_end_to_end and snapshot_census_tree
# ======================================================================================

def test_the_stress_artifact_rows_and_progress_are_neutral(machine, tmp_path, monkeypatch):
    import stress_cuda_end_to_end as stress  # noqa: PLC0415

    monkeypatch.setattr(stress, "_PROGRESS_PATH", str(tmp_path / "progress.log"))
    payload = {"rows": [{"label": "a", "capture": {"python": machine["python"]},
                         "error": f"RuntimeError: {machine['home']}/x on {HOST}",
                         "stderr_tail": _leak(machine)}]}
    stress.save(payload, str(tmp_path / "stress.json"))
    stress.append_jsonl(str(tmp_path / "rows.jsonl"),
                        {"label": "b", "capture": {"python": machine["python"]},
                         "error": f"RuntimeError: {machine['home']}/y on {HOST}"})
    stress.say(f"[a] ERROR {machine['home']}/x on {HOST}")
    assert_neutral(json.loads((tmp_path / "stress.json").read_text()), machine)
    assert_neutral(_jsonl(tmp_path / "rows.jsonl"), machine)
    assert_neutral((tmp_path / "progress.log").read_text(), machine)


def test_the_census_provenance_stamp_is_neutral(machine, tmp_path):
    import snapshot_census_tree as snapshot  # noqa: PLC0415

    census = tmp_path / "census"
    census.mkdir()
    (census / "census_provenance.json").write_text(json.dumps(
        {"tree_snapshot_sha256": {}, "cut_against": "WORKTREE"}))
    snapshot.verify(census, {"corpus_root": f"{machine['home']}/meep/python",
                             "driver": f"{machine['home']}/trident/driver.py"})
    provenance = json.loads((census / "census_provenance.json").read_text())
    assert provenance["corpus_root"] == "<home>/meep/python", provenance
    assert_neutral(provenance, machine)


# ======================================================================================
# The CUDA and Triton gates: the interpreter on a lift row and in provenance
# ======================================================================================

GATE_CHILD = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    from types import SimpleNamespace
    here, root, results = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    sys.path[:0] = [here, root]
    sys.executable = os.path.join(os.environ["HOME"], "envs", "ref", "bin", "python")
    import neutral_ids
    neutral_ids.reset()
    out = {}

    def jsonl(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\\n" for row in rows))

    import beta_hd_gate_common as beta
    jsonl(results / beta.BOARD / "fusion_matrix_cuda.json", [])
    (results / beta.BOARD / "fusion_matrix_cuda.json").write_text(json.dumps(
        {"h_to_d_seam": {"instances": [{"row": "examples:a.py", "update_H": "h",
                                         "step_D": "d"}]}}))
    out["beta"] = [r["interpreter"] for r in
                   beta.lift_basis(SimpleNamespace(cell_arms=("h", "d")), results)[0]]

    import gate_cuda_complex_fused_hd_pair as complex_gate
    cell = complex_gate.CELLS["plain"]
    (results / complex_gate.BOARD).mkdir(parents=True, exist_ok=True)
    (results / complex_gate.BOARD / "fusion_matrix_cuda.json").write_text(json.dumps(
        {"h_to_d_seam": {"instances": [{"row": "examples:a.py", "update_H": cell[0],
                                         "step_D": cell[1]}]}}))
    out["complex"] = [r["interpreter"] for r in complex_gate.lift_basis(results)[0]]

    import gate_cuda_fused_hd_pair as fused
    import gate_cuda_conductive_bfast_fused_hd_pair as conductive
    for name, gate, arms in (("fused", fused, fused.CELL_ARMS),
                             ("conductive", conductive, conductive.CELL_ARMS["conductive"])):
        seam = results / gate.SEAM_RECORD
        jsonl(seam / "h_to_d_seam.jsonl", [{"label": "examples:a.py", "leg": "examples",
                                            "row": "a.py", "h_to_d_seam": {},
                                            "arms": {"cuda": {"update_H": arms[0],
                                                              "step_D": arms[1]}}}])
        for leg in ("tests", "tests_param_matched", "examples"):
            jsonl(results / gate.CENSUS / f"{leg}.jsonl", [])
        out[name] = [r["interpreter"] for r in gate.lift_basis(results)[0]]

    import numpy as np
    import gate_triton_no_pml_constitutive as triton_gate
    import probe_triton_no_pml_constitutive_composition as triton_probe
    out["triton_gate"] = [triton_gate.provenance(np, "numpy")["python"]]
    out["triton_probe"] = [triton_probe.provenance("numpy")["python"]]
    print("INTERPRETERS=" + json.dumps(out))
''')


def test_the_gates_record_their_interpreter_under_its_label(tmp_path):
    home = tmp_path / "home" / LOGIN
    home.mkdir(parents=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("MEEP_GPU_")}
    environment.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1", MEEP_GPU_DISPATCH="0")
    done = subprocess.run([sys.executable, "-c", GATE_CHILD, str(HERE), str(API_ROOT),
                           str(tmp_path / "results")], capture_output=True, text=True,
                          timeout=600, cwd=str(tmp_path), env=environment)
    line = [part for part in done.stdout.splitlines() if part.startswith("INTERPRETERS=")]
    assert line, done.stdout[-2000:] + done.stderr[-4000:]
    found = json.loads(line[-1][len("INTERPRETERS="):])
    assert set(found) == {"beta", "complex", "fused", "conductive", "triton_gate",
                          "triton_probe"}, found
    for name, interpreters in found.items():
        assert interpreters == ["<home>/envs/ref/bin/python"], (name, interpreters)


# ======================================================================================
# The composition probes and a Metal gate's row file: writers that write their own files
# ======================================================================================

#: The composition probes that write their artifact themselves (no ``gate_provenance``
#: stamp): the fleet runs the first as ``cylindrical_complex_composition``.
COMPOSITION_PROBES = ("probe_triton_cylindrical_complex_composition",
                      "probe_triton_folded_offdiag_composition")


@pytest.mark.parametrize("name", COMPOSITION_PROBES)
def test_a_composition_probes_artifact_names_no_home(machine, monkeypatch, name):
    """The artifact a round sends back: its environment block (the cache directory the
    round sets under the checkout), its provenance file and an error that quotes a path
    are written under their labels, and the cases list stays the payload's own."""
    import importlib  # noqa: PLC0415

    probe = importlib.import_module(name)
    home = Path(machine["home"])
    results = home / "meep-gpu-cert" / "parity" / "meep_gpu" / "results"
    monkeypatch.setenv("CUPY_CACHE_DIR", str(results / "cupy_cache_ftz_stripped_keep_c6"))
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(__version__="0.0-stand-in"))
    runtime = types.SimpleNamespace(getDeviceProperties=lambda index: {"name": b"Stand-in GPU"})
    monkeypatch.setattr(probe, "cp", types.SimpleNamespace(
        __version__="0.0-stand-in", cuda=types.SimpleNamespace(runtime=runtime)))
    out = results / "triton_fleet_r_cc89_c6" / "composition" / "gate.json"
    cases: list = []
    payload = {"environment": probe.environment(), "cases": cases}
    probe.save(payload, str(out))
    payload["provenance"] = probe.write_provenance(str(out.parent))
    cases.append({"case": "m3_z_metallic_cw", "ok": False,
                  "error": f"OSError: cannot open {home}/meep-gpu-cert/state.npy on {HOST}"})
    probe.save(payload, str(out))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert_neutral(written, machine)
    assert written["environment"]["CUPY_CACHE_DIR"] == (
        "<home>/meep-gpu-cert/parity/meep_gpu/results/cupy_cache_ftz_stripped_keep_c6")
    assert written["provenance"].startswith("<home>/meep-gpu-cert/"), written["provenance"]
    assert written["provenance"].endswith("/composition/composition_provenance.json")
    assert payload["cases"] is cases and len(written["cases"]) == 1
    assert_neutral(json.loads((out.parent / "composition_provenance.json").read_text()),
                   machine)


METAL_ROW_CHILD = textwrap.dedent('''
    import json, os, sys
    here, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path[:0] = [here, root]
    import gate_metal_cylindrical_real_fused_hd_pair as gate
    corpus = os.path.join(os.environ["HOME"], "envs", "ref", "share", "meep-gpu-build",
                          "meep-1.35.0", "python")
    with open(out, "w", encoding="utf-8") as handle:
        gate.emit(handle, {"index": 1, "total": 1, "leg": "lift", "label": "lift",
                           "corpus_root": corpus, "rows": [{"script": corpus + "/x.py"}]})
''')


def test_the_metal_cylindrical_real_hd_pair_row_file_names_no_home(tmp_path):
    """``gate.jsonl`` leaves the machine beside the gate's artifact: a row is written under
    its labels (the lift leg records the MEEP corpus it read, under the home directory)."""
    home = tmp_path / "home" / LOGIN
    home.mkdir(parents=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("MEEP_GPU_")}
    environment.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1", MEEP_GPU_DISPATCH="0")
    out = tmp_path / "gate.jsonl"
    done = subprocess.run([sys.executable, "-c", METAL_ROW_CHILD, str(HERE), str(API_ROOT),
                           str(out)], capture_output=True, text=True, timeout=600,
                          cwd=str(tmp_path), env=environment)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    text = out.read_text(encoding="utf-8")
    row = json.loads(text)
    assert row["corpus_root"] == "<home>/envs/ref/share/meep-gpu-build/meep-1.35.0/python", row
    assert row["rows"] == [{"script": row["corpus_root"] + "/x.py"}], row
    assert str(home) not in text and LOGIN not in text, text
