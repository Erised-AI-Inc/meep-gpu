"""The 1.0 round's check refuses a gate stage whose Triton H->D lift inputs are absent.

WHY THIS FILE EXISTS. The ten Triton H->D hand legs join their corpus rows to a census
and a seam record under ``parity/meep_gpu/results`` (each gate's ``CENSUS`` and
``SEAM_RECORD``), which a checkout does not carry. On the 1.0 round the census was not
staged on the GPU host's certification clone and ``legs.py check`` said CHECK OK: the
first such leg ran every device leg, then stopped at its lift with "the lift leg needs
...", 113 s into a leg priced at 3,052 s, and the other nine would have died the same
way. ``check`` now reads both names from the gates' own source and refuses, by name,
a gate stage on which either directory is absent or lacks a file the lift reads.
Host-only: the gates are read, never imported.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROUND = HERE / "rounds" / "1.0"
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

import legs as L  # noqa: E402

CENSUS = "predicate_coverage_triton_2026-09-04_cylm0"
SEAM = "h_to_d_seam_2026-09-04"


def _stage(results: Path, *, census=True, seam=True, drop=()) -> Path:
    """Stage the inputs the lifts read, one line per file, minus ``drop``."""
    for staged, root, kind in ((census, CENSUS, "CENSUS"), (seam, SEAM, "SEAM_RECORD")):
        if not staged:
            continue
        (results / root).mkdir(parents=True, exist_ok=True)
        for name in L.TRITON_HD_INPUT_FILES[kind]:
            if name not in drop:
                (results / root / name).write_text("{}\n")
    return results


def test_every_gate_names_its_inputs_in_its_source_or_its_kits():
    """Read through ``kit.CENSUS`` and the tail kit's ``base.CENSUS`` alike."""
    inputs = L.triton_hd_inputs(HERE, L.triton_hd_steps())
    assert sorted(inputs) == sorted(L.TRITON_HD)
    assert all(names == {"CENSUS": CENSUS, "SEAM_RECORD": SEAM}
               for names in inputs.values()), inputs


def test_absent_inputs_are_refused_by_name_and_staged_ones_pass(tmp_path):
    steps = L.triton_hd_steps()
    problems, _inputs = L.triton_hd_input_problems(HERE, steps, tmp_path / "results")
    assert len(problems) == 2, problems
    assert any(CENSUS in p and "does not exist" in p for p in problems), problems
    assert any(SEAM in p and "does not exist" in p for p in problems), problems
    assert all("10 Triton H->D leg(s)" in p for p in problems), problems
    results = _stage(tmp_path / "results")
    assert L.triton_hd_input_problems(HERE, steps, results)[0] == []


@pytest.mark.parametrize("missing", ["tests_param_matched.jsonl", "h_to_d_seam.jsonl"])
def test_an_input_lacking_a_file_the_lift_reads_is_refused(tmp_path, missing):
    """A census leg file the lift skips silently empties every row it would have named."""
    results = _stage(tmp_path / "results", drop=(missing,))
    problems, _ = L.triton_hd_input_problems(HERE, L.triton_hd_steps(), results)
    assert len(problems) == 1 and f"lacks {missing}" in problems[0], problems


def test_a_gate_whose_inputs_cannot_be_read_is_refused(tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "kit_x.py").write_text('CENSUS = "c"\nSEAM_RECORD = "s"\n')
    (harness / "gate_triton_a_hd.py").write_text("import kit_x as kit\n"
                                                 "CENSUS = kit.CENSUS\n"
                                                 "SEAM_RECORD = kit.SEAM_RECORD\n")
    (harness / "gate_triton_b_hd.py").write_text("CENSUS = name()\nSEAM_RECORD = 's'\n")
    steps = [{"kind": "hd", "gate": "a_hd"}, {"kind": "hd", "gate": "b_hd"}]
    inputs = L.triton_hd_inputs(harness, steps)
    assert inputs == {"a_hd": {"CENSUS": "c", "SEAM_RECORD": "s"},
                      "b_hd": {"CENSUS": None, "SEAM_RECORD": "s"}}
    problems, _ = L.triton_hd_input_problems(harness, steps, _stage(tmp_path / "r"))
    assert any("['b_hd']" in p and "cannot be checked" in p for p in problems), problems


@pytest.fixture
def checked(tmp_path, monkeypatch):
    """``legs.check`` on this checkout's gate stage, its H->D results root at ``tmp``.

    The gates read ``HERE / "results"``; the fixture points the one function that
    reads it at a scratch root, so the check runs on the real harness either way.
    """
    corpus = tmp_path / "meep" / "python"
    for name in L.CORPUS_DIRECTORIES:
        (corpus / name).mkdir(parents=True)
    real = L.triton_hd_input_problems
    results = tmp_path / "results"
    monkeypatch.setattr(L, "triton_hd_input_problems",
                        lambda harness, steps, _results=None: real(harness, steps, results))
    monkeypatch.setattr(sys, "path", list(sys.path))

    def run():
        return L.check(REPO, "8.6", False, corpus=str(corpus))

    return run, results


def test_the_gate_stage_check_refuses_absent_hd_inputs(checked):
    """MUST CATCH: on the round's bytes this check said CHECK OK with both absent."""
    run, _results = checked
    problems, facts = run()
    hd = [p for p in problems if "Triton H->D leg(s)" in p]
    assert len(hd) == 2 and any(CENSUS in p for p in hd), problems
    assert sorted(facts["triton_hd_inputs"]) == sorted(L.TRITON_HD)


def test_the_gate_stage_check_passes_staged_hd_inputs(checked):
    run, results = checked
    _stage(results)
    problems, _facts = run()
    assert not [p for p in problems if "Triton H->D leg(s)" in p], problems


def test_the_route_stage_does_not_ask_for_them(checked):
    _run, _results = checked
    problems, facts = L.check(REPO, "8.6", True)
    assert not [p for p in problems if "Triton H->D leg(s)" in p], problems
    assert "triton_hd_inputs" not in facts


def test_this_checkouts_gate_stage_check_answers_for_its_own_results(tmp_path):
    """MUST CATCH, on the gates' own results root: refused exactly when it lacks them.

    A checkout carries no ``results/`` (the CI view), so there the check must refuse;
    a certification clone with the round's inputs staged must pass it.
    """
    corpus = tmp_path / "meep" / "python"
    for name in L.CORPUS_DIRECTORIES:
        (corpus / name).mkdir(parents=True)
    results = HERE / "results"
    staged = all((results / root / name).is_file()
                 for root, kind in ((CENSUS, "CENSUS"), (SEAM, "SEAM_RECORD"))
                 for name in L.TRITON_HD_INPUT_FILES[kind])
    saved = list(sys.path)
    try:
        problems, _facts = L.check(REPO, "8.6", False, corpus=str(corpus))
    finally:
        sys.path[:] = saved
    hd = [p for p in problems if "Triton H->D leg(s)" in p or CENSUS in p or SEAM in p]
    assert bool(hd) is not staged, (staged, problems)
