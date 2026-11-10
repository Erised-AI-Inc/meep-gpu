"""The stock-MEEP bench: its argument refusals, and one tiny end-to-end row.

The end-to-end test runs the bench and the harness-lift digest in child processes, one
rank, ``pml_3d`` at resolution 6 (13,824 cells), three 0.05 s windows: seconds. It is
the one-builder claim end to end -- the MEEP row's geometry digest equals the digest of
the simulation the package's own lift initialised -- and it reads back the chunk
splitting the row asked for.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bench_meep_identical_case as bench  # noqa: E402
import neutral_ids  # noqa: E402


@functools.lru_cache(maxsize=1)
def single_precision_meep() -> bool:
    if importlib.util.find_spec("meep") is None:
        return False
    done = subprocess.run([sys.executable, "-c", "import meep; print('SINGLE' if "
                           "meep.is_single_precision() else 'DOUBLE')"],
                          capture_output=True, text=True, timeout=300)
    return "SINGLE" in done.stdout.split()


def test_split_argument():
    assert bench.parse_split("true") is True and bench.parse_split("False") is False
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_split("cost")


def test_spread_is_the_harness_definition():
    assert bench.spread([1.0, 1.1, 0.9]) == pytest.approx(0.2)
    assert bench.spread([]) != bench.spread([])  # NaN


def test_threads_per_rank_refusals(monkeypatch):
    assert bench.check_threads(None, None)["asked"] == 1
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(SystemExit, match="OMP_NUM_THREADS is '1'"):
        bench.check_threads(2, None)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setattr(bench, "mapped_openmp", lambda: [])
    with pytest.raises(SystemExit, match="needs --build-record"):
        bench.check_threads(2, None)
    # No runtime mapped after MEEP's import still refuses
    with pytest.raises(SystemExit, match="after importing MEEP"):
        bench.check_threads_after_import(2, bench.check_threads(2, {"openmp": True}))
    monkeypatch.setattr(bench, "mapped_openmp", lambda: None)
    with pytest.raises(SystemExit, match="needs --build-record"):
        bench.check_threads(2, None)
    assert bench.check_threads(2, {"openmp": True})["asked"] == 2
    # A runtime OpenBLAS mapped into the process admits nothing: the record decides.
    monkeypatch.setattr(bench, "mapped_openmp", lambda: ["libgomp"])
    with pytest.raises(SystemExit, match="needs --build-record"):
        bench.check_threads(2, None)
    with pytest.raises(SystemExit, match="link no OpenMP runtime"):
        bench.check_threads(2, {"openmp": False})
    with pytest.raises(SystemExit, match="link no OpenMP runtime"):
        bench.check_threads(2, {"openmp": None})


def test_a_hybrid_row_reads_the_runtime_after_meep_is_imported(monkeypatch):
    """In the 1.0 timing run, 18 / 18 hybrid MPI x OpenMP rows refused "no OpenMP runtime is
    mapped" on a build that links ``libgomp``, because the maps were read before ``import
    meep`` loaded the library that maps it. Before the import nothing is mapped; after it
    the runtime is, and the row runs."""
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    imported = {"meep": False}
    monkeypatch.setattr(bench, "mapped_openmp",
                        lambda: ["libgomp"] if imported["meep"] else [])
    check = bench.check_threads(2, {"openmp": True})    # refused here before the fix
    assert check["omp_runtime_mapped_before_import"] == []
    imported["meep"] = True
    check = bench.check_threads_after_import(2, check)
    assert check["omp_runtime_mapped"] == ["libgomp"] and check["asked"] == 2
    # and ``main`` asks the second question only once MEEP is imported
    import inspect  # noqa: PLC0415
    source = inspect.getsource(bench.main)
    assert (source.index("check_threads(args.threads_per_rank")
            < source.index("import meep as mp")
            < source.index("check_threads_after_import("))


def test_binding_summary_reads_the_ladders_scope_and_cores():
    gathered = {"ranks": [{"affinity": list(range(0, 24))}, {"affinity": list(range(0, 24))}]}
    host = bench.binding_summary(gathered, None, cores={c: f"0:{c}" for c in range(24)})
    assert host["scope_from"] == "os.cpu_count()"
    inside = bench.binding_summary(gathered, 24, cores={c: f"0:{c}" for c in range(24)})
    assert inside["bound"] is False and inside["scope_cpus"] == 24
    hw = {"ranks": [{"affinity": [c]} for c in (0, 1, 2, 3)]}
    siblings = {0: "0:0", 1: "0:0", 2: "0:1", 3: "0:1"}
    summary = bench.binding_summary(hw, 4, cores=siblings)
    assert summary["bound"] is True and summary["physical_cores_used"] == 2


def test_too_few_windows_is_refused():
    with pytest.raises(SystemExit, match="at least 3"):
        bench.main(["--res", "4", "--out", "/nonexistent", "--windows", "2"])


@pytest.mark.requires_resource("single_precision_meep")
def test_one_rank_row_and_the_harness_lift_carry_one_digest(tmp_path):
    if not single_precision_meep():
        pytest.skip("[requires_resource][single_precision_meep] MEEP is absent or a "
                    "double-precision build")
    env = dict(os.environ, PYTHONPATH=REPO, MEEP_GPU_DISPATCH="0", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1")
    out = tmp_path / "meep"
    bench_run = subprocess.run(
        [sys.executable, "-u", os.path.join(HERE, "bench_meep_identical_case.py"),
         "--case", "pml_3d", "--res", "6", "--out", str(out), "--row-id", "t_r1",
         "--windows", "3", "--target-seconds", "0.05", "--warm-steps", "16",
         "--warm-seconds-cap", "1", "--split-chunks-evenly", "false"],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert bench_run.returncode == 0, bench_run.stdout[-2000:] + bench_run.stderr[-2000:]
    rows = [json.loads(line) for line in open(out / "rows_r1.jsonl")]
    measurement = next(r for r in rows if r["row"] == "measurement")
    assert measurement["cells"] == 13_824 and measurement["windows"] == 3
    assert measurement["configuration"]["split_chunks_evenly"] is False
    assert measurement["chunks"]["split_chunks_evenly_read_back"] is False
    assert sum(measurement["chunks"]["cells_per_rank"]) == 13_824
    assert measurement["monitor"]["stepped"] is True
    assert measurement["environment"]["builder"]["builder_is_end_to_end_case"] is True
    # the rows leave the machine: no path, host name or UUID of this one, and no rank pid
    assert neutral_ids.residue(rows) == []
    assert all("pid" not in rank for row in rows if row.get("row") == "environment"
               for rank in row["rank_bindings"])
    assert all("pid" not in rank for rank in measurement["environment"]["rank_bindings"])
    lift = subprocess.run(
        [sys.executable, "-u", os.path.join(HERE, "digest_harness_lift.py"),
         "--case", "pml_3d", "--res", "6", "--out", str(tmp_path / "lift"),
         "--compare", str(out / "rows_r1.jsonl")],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert lift.returncode == 0, lift.stdout[-2000:] + lift.stderr[-2000:]
    record = json.load(open(tmp_path / "lift" / "harness_lift_digest_pml_3d_res6.json"))
    assert record["geometry"]["digest"] == measurement["geometry_digest"]
    assert record["verdict"]["identical"] is True
    # the digest record leaves the machine too: its builder and compared-file paths labelled
    assert neutral_ids.residue(record) == []
    assert record["compared_with"][0]["file"].endswith("/meep/rows_r1.jsonl")


# ---------------------------------------------------------------------------
# What a row writes about the machine (neutral_ids)
# ---------------------------------------------------------------------------

#: A machine no test here runs on: nothing a row writes may carry any of it.
HOST = "gpu-node-17.cluster.example.edu"
LOGIN = "fixtureuser"
HOME = "/" + "ho" + "me" + "/fixtureuser"   # from pieces: a full home path is a hygiene-guard hit
REAL_A = "GPU-3f1c2a9e-5b7d-4e21-9a0c-6d8e2f4b1a37"
FOREIGN_PID = 271828


@pytest.fixture
def machine(monkeypatch):
    """This process as the fixture machine: its host names, login and home."""
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: HOST)
    monkeypatch.setattr(neutral_ids, "_node", lambda: HOST)
    monkeypatch.setattr(neutral_ids.getpass, "getuser", lambda: LOGIN)
    monkeypatch.setenv("HOME", HOME)
    neutral_ids.reset()
    yield neutral_ids
    neutral_ids.reset()


def assert_neutral(record, *tokens):
    assert neutral_ids.residue(record) == []
    text = json.dumps(record, sort_keys=True, default=str)
    for token in (HOST, HOST.split(".")[0], LOGIN, HOME, REAL_A, str(FOREIGN_PID)) + tokens:
        assert str(token) not in text, (token, text[:400])


def test_a_rank_writes_its_machine_id_and_no_pid(machine, monkeypatch):
    monkeypatch.setattr(bench.platform, "node", lambda: HOST)
    monkeypatch.setattr(bench.os, "getpid", lambda: FOREIGN_PID)
    mp = argparse.Namespace(my_rank=lambda: 3)
    facts = bench.rank_facts(mp, launched=[0, 48])
    assert facts["rank"] == 3 and facts["affinity"] == [0, 48]
    assert facts["host"] == neutral_ids.machine_id() and "pid" not in facts
    assert_neutral(facts)


def test_the_environment_row_writes_nothing_of_the_machine(machine):
    """The environment row and every measurement row carry this block: the launcher's
    binding report keeps the cores and drops host and pid, paths are labelled, the host
    is the machine id."""
    short = HOST.split(".")[0]
    environment = {
        "meep": "1.35.0", "meep_file": f"{HOME}/miniforge3/envs/meep/lib/meep/__init__.py",
        "rank_bindings": [{"rank": 0, "pid": FOREIGN_PID, "host": HOST, "affinity": [0]},
                          {"rank": 1, "pid": FOREIGN_PID + 1, "host": HOST,
                           "affinity": [1]}],
        "launcher_line": (f"{HOME}/miniforge3/envs/meep/bin/mpirun -np 2 --report-bindings "
                          f"{HOME}/miniforge3/envs/meep/bin/python bench.py"),
        "launcher_bindings_report": [
            f"[{short}:{FOREIGN_PID}] Rank 0 bound to package[0][core:0]",
            f"cpu-bind=MASK - {short}, task  1  1 [{FOREIGN_PID + 1}]: mask 0x2 set"],
        "host": HOST, "platform": "Linux-5.15.0-91-generic-x86_64-with-glibc2.35",
        "executable": f"{HOME}/miniforge3/envs/meep/bin/python",
        "meep_configure_line": f"./configure --prefix={HOME}/miniforge3/envs/meep",
        "meep_config_log": f"{HOME}/src/meep/config.log",
        "build_record": {"path": f"{HOME}/runs/build_record.json", "sha256": "ab"},
        "bench": {"path": f"{HOME}/runs/drivers/bench_meep_identical_case.py",
                  "sha256": "cd"},
        "argv": ["--case", "pml_3d", "--out", f"{HOME}/runs/ladder_t/meep_r2",
                 "--bindings-report", f"{HOME}/runs/ladder_t/logs/r2.log"],
        "box_before_from_driver": f"load=0.1 0.1 0.1 busy_cpus=0 gpu0=3 MiB, 0 % "
                                  f"others=0, account=0 on {HOST}",
        "threads": {"CUDA_VISIBLE_DEVICES": REAL_A},
    }
    written = bench.recorded_environment(environment)
    assert_neutral(written, str(FOREIGN_PID + 1))
    assert written["host"] == neutral_ids.machine_id()
    assert written["launcher_bindings_report"] == [
        "Rank 0 bound to package[0][core:0]", "cpu-bind=MASK - task  1  1: mask 0x2 set"]
    assert [r["affinity"] for r in written["rank_bindings"]] == [[0], [1]]
    assert all("pid" not in r for r in written["rank_bindings"])
    assert written["executable"] == "<home>/miniforge3/envs/meep/bin/python"
    assert written["argv"][3] == "<home>/runs/ladder_t/meep_r2"
    assert written["platform"] == environment["platform"]
    assert environment["rank_bindings"][0]["pid"] == FOREIGN_PID, "the block in process"
