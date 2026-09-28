"""The census driver's runtime preflight: no row is measured where its battery cannot evaluate.

Pins the 2026-09-03 defect class at both ends. On the battery side, the Metal battery's
``runtime_reasons`` IS ``coverage._metal_backend_reasons`` over a host grid under the
battery's declared policy -- one definition, so the hook can never disagree with the
predicates about the same interpreter. On the driver side, ``preflight_runtime`` asks an
interpreter once and fails closed, and ``runtime_refusal`` is the by-name record a row
gets instead of a full-length row of unanimous refusals.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy
import pytest

from meep_gpu.metal_kernels import coverage as metal_coverage
from meep_gpu.metal_kernels import subnormal
from parity.meep_gpu import measure_predicate_coverage as census

_HERE = Path(__file__).resolve().parent
_API = _HERE.parents[1]


class _HostGrid:
    xp = numpy


@pytest.fixture(autouse=True)
def _batteries_importable_by_bare_name(monkeypatch):
    # The driver imports a battery by its bare module name from beside itself, exactly
    # as it does when run as a script (its own directory is then sys.path[0]).
    monkeypatch.syspath_prepend(str(_HERE))


def _child_environment(*extra_path: str) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([*extra_path, str(_API)])
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    return environment


def test_without_torch_the_hook_is_the_one_import_reason_and_nothing_more(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # ``import torch`` raises ImportError
    reasons = census.runtime_reasons_of("predicate_battery")
    assert len(reasons) == 1, reasons
    assert reasons[0].startswith("torch is not importable"), reasons
    assert reasons == metal_coverage._metal_backend_reasons(_HostGrid())


def test_with_torch_under_the_declared_policy_the_hook_is_the_backend_clause(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)   # the battery's own policy
    reasons = census.runtime_reasons_of("predicate_battery")
    assert reasons == metal_coverage._metal_backend_reasons(_HostGrid())
    # A host grid adds no array-module clause and flush adds no policy clause, so on a
    # host that can launch Metal the hook is empty -- and only then.
    assert reasons == [r for r in reasons if not r.startswith("array module is")]


def test_a_policy_the_executor_cannot_honour_is_a_process_reason(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.KEEP)
    reasons = census.runtime_reasons_of("predicate_battery")
    assert any("cannot honour" in r for r in reasons), reasons


def test_the_pure_python_batteries_declare_no_hook():
    assert census.runtime_reasons_of("triton_predicate_battery") == []
    assert census.runtime_reasons_of("cuda_predicate_battery") == []


def test_in_process_and_child_preflight_agree_for_the_same_interpreter(tmp_path):
    own = census.preflight_runtime(sys.executable, "predicate_battery", _child_environment())
    link = tmp_path / "python"           # a distinct path to the same interpreter forces
    link.symlink_to(sys.executable)      # the child route
    assert census.preflight_runtime(str(link), "predicate_battery", _child_environment()) == own


def test_a_child_whose_torch_import_fails_reports_that_reason(tmp_path):
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "torch.py").write_text("raise ImportError('shadowed for the test')\n",
                                     encoding="utf-8")
    link = tmp_path / "python"
    link.symlink_to(sys.executable)
    reasons = census.preflight_runtime(str(link), "predicate_battery",
                                       _child_environment(str(shadow)))
    assert len(reasons) == 1 and reasons[0].startswith("torch is not importable"), reasons
    assert "shadowed for the test" in reasons[0]


def test_an_interpreter_that_cannot_answer_is_itself_a_reason(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    fake.chmod(0o755)
    reasons = census.preflight_runtime(str(fake), "predicate_battery", _child_environment())
    assert len(reasons) == 1, reasons
    assert "exited 3" in reasons[0] and str(fake) in reasons[0], reasons


def test_an_interpreter_that_prints_no_list_is_itself_a_reason(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\necho hello\n", encoding="utf-8")
    fake.chmod(0o755)
    reasons = census.preflight_runtime(str(fake), "predicate_battery", _child_environment())
    assert reasons == [f"runtime preflight under {fake} printed no reason list"]


def test_the_refusal_record_names_the_interpreter_the_record_and_every_reason():
    record = census.runtime_refusal("mie_scattering.py", "/envs/sigma/bin/python",
                                    "corpus_final_numpy_2026-08-09_monitorfix_sigma",
                                    ["torch is not importable (x)", "no MPS device"])
    assert record["measured"] is False and record["row"] == "mie_scattering.py"
    for token in ("/envs/sigma/bin/python", "monitorfix_sigma",
                  "torch is not importable (x)", "no MPS device"):
        assert token in record["note"], (token, record["note"])
    assert record["runtime_reasons"] == ["torch is not importable (x)", "no MPS device"]


# ---------------------------------------------------------------------------
# The CUDA battery's refusal classification, for the arms registered 2026-09-27
# ---------------------------------------------------------------------------

def _battery():
    from parity.meep_gpu import cuda_predicate_battery  # noqa: PLC0415

    return cuda_predicate_battery


def test_the_in_seam_spelling_of_the_backend_clause_is_the_backend_clause():
    """The mirror-fill arm refuses a NumPy grid in the in-seam predicates' words, and
    an unshimmed winner refused for that alone must read ``backend``, not ``other``."""
    battery = _battery()
    assert battery.is_backend_clause_alone(f"PML: {battery.CUDA_BACKEND_CLAUSE}")
    assert battery.is_backend_clause_alone(
        "mirror fill: backend is 'numpy', not cupy; these are CUDA kernels")
    # ACCUMULATED with another clause it is not the backend clause alone.
    assert not battery.is_backend_clause_alone(
        "mirror fill: backend is 'numpy', not cupy; these are CUDA kernels; "
        "Hx is not C-contiguous")
    assert not battery.is_backend_clause_alone("mirror fill: complex64 storage")


def test_the_in_seam_clause_the_battery_matches_is_the_shipped_one():
    """Read off the predicate itself, so a reworded clause fails here, not in a census."""
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415

    class _Named:
        __name__ = "numpy"

    grid = type("G", (), {"xp": _Named(), "is_axis": staticmethod(lambda axis: False)})()
    reasons = in_seam_coverage._shared_reasons(None, grid, "B")  # noqa: SLF001
    assert any(_battery().IN_SEAM_BACKEND_CLAUSE.fullmatch(reason)
               for reason in reasons), reasons


def test_an_unfolded_rows_fill_slot_is_not_folded_and_a_refused_fold_is_a_gap():
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    battery = _battery()
    unfolded = (f"mirror fill: {arms.MIRROR_FILL_UNFOLDED_REFUSAL}: fill_symmetry_bc_B "
                "and fill_folded_far_ghosts_B touch no array here",)
    assert battery._refusal_kind("fill_B", unfolded)["kind"] == "not_folded"  # noqa: SLF001
    refused_fold = ("mirror fill: Hx is complex64; the in-seam kernels index float32",)
    assert battery._refusal_kind("fill_B", refused_fold)["kind"] == "no_admitter"  # noqa: SLF001
