"""``bench_timing_case.py``: a timing case injected into the frozen bench, and taken out.

No case is lifted: ``run_case`` is replaced, as ``test_bench_fused_products_provenance``
replaces it, and the launch witnesses are the same no-op bundle. What is checked is the
wrapper's own contract: the injected case is SELECTED by the bench's unmodified
``selected_cases``, every row it writes carries ``timing_case``, the bench's provenance
keys are unchanged, and the gate tables are restored after a normal return and after an
exception.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bench_fused_products as bench  # noqa: E402
import bench_timing_case as wrapper  # noqa: E402
import timing_cases  # noqa: E402


class _Counters:
    def __init__(self):
        self.triton = SimpleNamespace(install=lambda: None)
        self.cuda = SimpleNamespace(install=lambda: None, wrap_memo_store=lambda: None)

    @staticmethod
    def snapshot():
        return {"triton": {"total": 0, "by_kernel": {}}, "cuda": {"total": 0, "by_kernel": {}}}

    @staticmethod
    def delta(_before, after):
        return after


@pytest.fixture
def stubbed(monkeypatch):
    seen = []

    def fake_run_case(case, spec, module, table, *_args, **_kwargs):
        seen.append({"case": case, "spec": spec, "builder": module.CASES[case],
                     "table": table})
        return {"case": case, "verdict": "SKIP-NOT-THIS-LEG"}

    monkeypatch.setattr(bench, "run_case", fake_run_case)
    monkeypatch.setattr(bench.e2e, "LaunchCounters", _Counters)
    monkeypatch.setattr(bench.e2e, "PREFER_GPU", True)
    monkeypatch.setattr(bench.route, "BACKEND", bench.route.BACKEND)
    monkeypatch.setattr(bench, "WITNESSES", [])
    monkeypatch.setattr(bench, "ONLY_ARMS", [])
    return seen


def tables_snapshot():
    return (set(bench.route.CASES), set(bench.route.DRIVE), set(bench.route.DRIVE_CUDA),
            bench.append_jsonl)


def test_the_injected_case_is_selected_and_stamped(stubbed, tmp_path):
    before = tables_snapshot()
    stamp = bench.provenance("triton")
    status = wrapper.main(["--harness-root", HERE, "--drive-table", "cuda", "--out",
                           str(tmp_path), "--case", "thin_pml_3d_diagonal", "--smoke"])
    assert status == 1  # no row is reportable: the stub times nothing
    assert [s["case"] for s in stubbed] == ["thin_pml_3d_diagonal"]
    assert stubbed[0]["builder"] is timing_cases.case_thin_pml_3d_diagonal
    assert stubbed[0]["spec"]["arms"] == bench.route.DRIVE_CUDA["pml_3d_diagonal"]["arms"]
    assert stubbed[0]["spec"]["unfused_baseline"] == "cuda"
    rows = [json.loads(line) for line in open(tmp_path / "rows.jsonl")]
    assert len(rows) == 1
    identity = rows[0]["timing_case"]
    assert identity["name"] == "thin_pml_3d_diagonal"
    assert identity["template"] == "pml_3d_diagonal" and identity["table"] == "cuda"
    assert len(identity["builder_source_sha256"]) == 64
    assert len(identity["module_sha256"]) == 64
    assert sorted(rows[0]["provenance"]["sha256"]) == sorted(stamp["sha256"])
    assert "timing_case" not in rows[0]["provenance"]["sha256"]
    assert tables_snapshot() == before


def test_the_tables_are_restored_after_an_exception(stubbed, monkeypatch, tmp_path):
    before = tables_snapshot()

    def explode(*_args, **_kwargs):
        assert "thin_pml_3d" in bench.route.CASES  # injected while the bench runs
        raise RuntimeError("the campaign died")

    monkeypatch.setattr(bench, "_campaign", explode)
    with pytest.raises(RuntimeError, match="the campaign died"):
        wrapper.main(["--harness-root", HERE, "--out", str(tmp_path), "--case",
                      "thin_pml_3d", "--smoke"])
    assert tables_snapshot() == before


def test_a_builtin_or_colliding_name_is_refused(stubbed, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="is a built-in case"):
        wrapper.main(["--harness-root", HERE, "--out", str(tmp_path), "--case", "pml_3d",
                      "--smoke"])
    monkeypatch.setitem(bench.route.DRIVE, "thin_pml_3d_hz", {"arms": []})
    with pytest.raises(SystemExit, match="already exists"):
        wrapper.main(["--harness-root", HERE, "--out", str(tmp_path), "--case",
                      "thin_pml_3d_hz", "--smoke"])
    assert "thin_pml_3d_hz" not in bench.route.CASES


def test_the_wrapper_option_is_not_passed_to_the_bench():
    assert wrapper._passthrough(["--harness-root", "/x", "--out", "o", "--case", "c"]) == [
        "--out", "o", "--case", "c"]
    assert wrapper._passthrough(["--harness-root=/x", "--res", "6"]) == ["--res", "6"]
