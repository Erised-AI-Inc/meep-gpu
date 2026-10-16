"""The H->D seam's shared measurements (``hd_seam_restore``), each beside the defect it
must catch.

What each test pins, and what would break it:

- ``driver_seam_facts`` passes on the tree's driver and FAILS, by name, on a copy with a
  pass planted between the two consults -- the CUDA and Triton ``driver_order`` legs read
  it, so a seam that grew a pass would stop them.
- ``withdraw_refusal_checks`` asks a predicate twice. A predicate that admits the engine
  integrated source and refuses the foreign withdraw by name passes; one that admits the
  foreign withdraw, refuses it without naming the withdraw, or refuses the engine source
  fails the check that names it.
- ``lift_withdraw_accounting`` derives the refusals a row owes from its OWN sources. The
  seam record's ``withdraw_in_seam`` rows are admitted; a row refused without a working
  withdraw, a row with one admitted, a refusal that does not name the withdraw and a row
  with no source list each fail it. A magnetic source's withdraw is never owed here.
- ``lifter_refusal`` names a row the engine's lifter refused by name and nothing else.
- ``plans_bind_the_driver_chunk_layout`` hands a lifted row's Triton plans the driver's
  chunk division and restores the engine's function on exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import h_to_d_seam  # noqa: E402
import hd_seam_restore as seam  # noqa: E402


# --- the driver fact -----------------------------------------------------------------


def test_the_seam_holds_only_the_magnetic_restore_on_this_tree():
    facts = seam.driver_seam_facts()
    assert facts["passed"], facts
    assert tuple(facts["statements_between_the_two_consults"]) == h_to_d_seam.EXPECTED_BETWEEN
    assert not facts["a_withdraw_stands_between_the_two_consults"]


def test_a_withdraw_planted_in_the_seam_fails_the_driver_order_facts(tmp_path):
    """The null control: a withdraw loop back between the consults must fail by name."""
    source = h_to_d_seam.DRIVER.read_text(encoding="utf-8")
    needle = ('            f_minus_p_B = None\n'
              '        if fast is None or not fast.dispatch("step_D", self.fields):')
    assert source.count(needle) == 1, "the seam's text moved; re-anchor this control"
    planted = source.replace(
        needle,
        '            f_minus_p_B = None\n'
        '        for source in electric:\n'
        '            getattr(source, "withdraw", _no_withdraw)(self.fields)\n'
        '        if fast is None or not fast.dispatch("step_D", self.fields):')
    copy = tmp_path / "driver.py"
    copy.write_text(planted, encoding="utf-8")
    facts = seam.driver_seam_facts(copy)
    assert facts["passed"] is False
    assert "withdraw" in facts["error"]


# --- the refusal pair -------------------------------------------------------------------


class _EngineIntegrated:
    """Shaped as an engine integrated source reads to the predicates: integrated, with
    points, and NO withdraw (``f_minus_p`` carries its dipole)."""

    field_type = "D"
    component = "Ez"
    is_integrated = True
    _n_source_points = 3


def _predicate(*, admit_foreign: bool = False, name_it: bool = True,
               refuse_engine: bool = False):
    def ask(sources):
        from meep_gpu import withdraw_hoist  # noqa: PLC0415

        standing = withdraw_hoist.standing_withdraws(sources)
        if standing and not admit_foreign:
            return False, ("source 0 has a standing integrated electric withdraw"
                           if name_it else "source 0 is not served here")
        if refuse_engine and not standing:
            return False, "integrated sources are not served here"
        return True, "covered"
    return ask


def test_the_refusal_pair_passes_on_a_correct_predicate():
    checks = seam.withdraw_refusal_checks(_predicate(), _EngineIntegrated())
    assert all(entry["passed"] for entry in checks.values()), checks


def test_a_predicate_admitting_a_foreign_withdraw_fails_the_refusal_check():
    checks = seam.withdraw_refusal_checks(_predicate(admit_foreign=True),
                                          _EngineIntegrated())
    assert not checks["refuses_a_foreign_standing_integrated_electric_withdraw"]["passed"]


def test_a_refusal_that_does_not_name_the_withdraw_fails():
    checks = seam.withdraw_refusal_checks(_predicate(name_it=False), _EngineIntegrated())
    assert not checks["refuses_a_foreign_standing_integrated_electric_withdraw"]["passed"]


def test_a_predicate_refusing_the_engine_integrated_source_fails_the_admission_check():
    checks = seam.withdraw_refusal_checks(_predicate(refuse_engine=True),
                                          _EngineIntegrated())
    assert not checks["admits_an_engine_integrated_electric_source"]["passed"]
    assert checks["refuses_a_foreign_standing_integrated_electric_withdraw"]["passed"]


# --- the lift's refusal accounting ------------------------------------------------------


def _block(admits: bool, *sources, reason: str = "covered"):
    return {"predicate_admits": admits, "predicate_reason": reason,
            "sources": [dict(source) for source in sources]}


ENGINE = {"field_type": "D", "is_integrated": True, "withdraw_does_work": False}
FOREIGN = {"field_type": "D", "is_integrated": True, "withdraw_does_work": True}
MAGNETIC_FOREIGN = {"field_type": "B", "is_integrated": True, "withdraw_does_work": True}
REFUSED_BY_NAME = "source 0 has a standing integrated electric withdraw"


def test_the_seam_records_withdraw_rows_admitted_and_nothing_owed_pass():
    blocks = {"a": _block(True, ENGINE), "b": _block(True), "c": _block(True, MAGNETIC_FOREIGN)}
    result = seam.lift_withdraw_accounting(blocks, seam_withdraw_rows=["a"])
    assert result["passed"], result
    assert result["expected_refused_from_the_row_sources"] == []


def test_a_seam_record_row_still_refused_fails():
    blocks = {"a": _block(False, ENGINE, reason=REFUSED_BY_NAME), "b": _block(True)}
    result = seam.lift_withdraw_accounting(blocks, seam_withdraw_rows=["a"])
    assert not result["passed"]
    assert result["of_those_refused_without_a_working_withdraw"] == ["a"]


def test_a_row_with_a_working_withdraw_must_be_refused_by_name():
    admitted = {"a": _block(True, FOREIGN)}
    assert not seam.lift_withdraw_accounting(admitted)["passed"]
    unnamed = {"a": _block(False, FOREIGN, reason="not served")}
    assert not seam.lift_withdraw_accounting(unnamed)["passed"]
    named = {"a": _block(False, FOREIGN, reason=REFUSED_BY_NAME)}
    assert seam.lift_withdraw_accounting(named)["passed"]


def test_a_row_without_a_source_list_is_ignorance_and_fails():
    blocks = {"a": {"predicate_admits": True, "predicate_reason": "covered"}}
    result = seam.lift_withdraw_accounting(blocks)
    assert not result["passed"]
    assert result["rows_without_a_source_list"] == ["a"]


def test_a_refusal_the_caller_owns_is_reported_not_scored():
    blocks = {"a": _block(False, ENGINE, reason="the policy refuses the licence"),
              "b": _block(True)}
    result = seam.lift_withdraw_accounting(
        blocks, tolerated=lambda label, block: label == "a")
    assert result["passed"], result
    assert result["refused_by_a_refusal_the_caller_owns"] == ["a"]


# --- the lifter's own refusal -----------------------------------------------------------


def test_only_a_lifter_refusal_by_name_is_counted_out():
    refusal = {"has_simulation": True,
               "lift_error": "MeepSimulationNotLiftable: This mp.Simulation cannot be lifted"}
    assert seam.lifter_refusal(refusal)
    assert seam.lifter_refusal({"has_simulation": True,
                                "child_error": refusal["lift_error"]})
    assert seam.lifter_refusal({**refusal, "has_simulation": False}) is None
    assert seam.lifter_refusal({"has_simulation": True,
                                "lift_error": "RuntimeError: out of memory"}) is None


# --- the lifted row's chunk division -----------------------------------------------------


def test_a_lifted_rows_plans_bind_the_driver_chunk_layout(monkeypatch):
    """Inside the context a ``None`` division is the driver's; an explicit one passes
    through; on exit the engine's own function is back."""
    from types import SimpleNamespace

    from meep_gpu.triton_kernels import launch

    def recorder(fields, pml, chunk_layout):  # noqa: ARG001
        return chunk_layout

    import sys
    import types

    monkeypatch.setattr(launch, "chunk_bounds_for", recorder)
    # A kernel module that bound the function by name at import, as several do.
    bound = types.ModuleType("meep_gpu.triton_kernels._fxh_bound_by_name")
    bound.chunk_bounds_for = recorder
    monkeypatch.setitem(sys.modules, bound.__name__, bound)
    driver = SimpleNamespace(chunk_layout="the driver's division")
    with seam.plans_bind_the_driver_chunk_layout(driver):
        assert launch.chunk_bounds_for(None, None, None) == "the driver's division"
        assert launch.chunk_bounds_for(None, None, "explicit") == "explicit"
        assert bound.chunk_bounds_for(None, None, None) == "the driver's division"
    assert launch.chunk_bounds_for is recorder
    assert bound.chunk_bounds_for is recorder
    assert launch.chunk_bounds_for(None, None, None) is None
