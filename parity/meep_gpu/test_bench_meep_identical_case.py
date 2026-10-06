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
    with pytest.raises(SystemExit, match="no OpenMP runtime"):
        bench.check_threads(2, None)
    monkeypatch.setattr(bench, "mapped_openmp", lambda: None)
    with pytest.raises(SystemExit, match="needs --build-record"):
        bench.check_threads(2, None)
    assert bench.check_threads(2, {"openmp": True})["asked"] == 2
    # A RUNTIME OPENBLAS MAPPED INTO THE PROCESS ADMITS NOTHING: the record decides.
    monkeypatch.setattr(bench, "mapped_openmp", lambda: ["libgomp"])
    with pytest.raises(SystemExit, match="needs --build-record"):
        bench.check_threads(2, None)
    with pytest.raises(SystemExit, match="link no OpenMP runtime"):
        bench.check_threads(2, {"openmp": False})
    with pytest.raises(SystemExit, match="link no OpenMP runtime"):
        bench.check_threads(2, {"openmp": None})


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
    lift = subprocess.run(
        [sys.executable, "-u", os.path.join(HERE, "digest_harness_lift.py"),
         "--case", "pml_3d", "--res", "6", "--out", str(tmp_path / "lift"),
         "--compare", str(out / "rows_r1.jsonl")],
        capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert lift.returncode == 0, lift.stdout[-2000:] + lift.stderr[-2000:]
    record = json.load(open(tmp_path / "lift" / "harness_lift_digest_pml_3d_res6.json"))
    assert record["geometry"]["digest"] == measurement["geometry_digest"]
    assert record["verdict"]["identical"] is True
