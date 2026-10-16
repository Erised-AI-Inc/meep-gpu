"""The array-path oracle: what it measures, what it refuses, and that it can fail.

The ledger comparison itself lives in the three weld contracts
(``test_<table>_weld_contract.py::test_the_<table>_ledger_records_the_live_array_path_oracle``),
which are certification files. This file holds the oracle to its own claims on synthetic
inputs and on the live fixture: the fingerprint is deterministic, every declared form
class is reached, an arithmetic change to the array path moves it and is reported by
name, a missing class or a libm call or a flushing FPU is a named refusal, and the form
counts use the one-rank chunk rule the PML itself uses.

NumPy only; the whole fixture set steps in well under a second.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib

import numpy
import pytest

from . import array_path_oracle as oracle
from . import fastpath
from . import weld_record_walk as walk

PACKAGE = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE.parent


def _synthetic_ledger(result):
    """A two-entry ledger that records ``result`` the way the minting tool writes it."""
    return {
        "a_weld": {"source_sha256": {"meep_gpu/stepping.py": "0" * 64},
                   oracle.LEAF_KEY: oracle.entry_leaf(result)},
        "_a_note": "prose binds no bytes",
        "driver_dispatch": {"source_sha256": {"meep_gpu/driver.py": "1" * 64},
                            oracle.LEAF_KEY: oracle.dispatch_leaf(result, REPO)},
    }


def test_the_fixture_reaches_every_declared_form_class():
    result = oracle.live()
    assert oracle.coverage_problems(result.form_classes) == []
    assert len(oracle.DECLARED_FORM_CLASSES) == len(set(oracle.DECLARED_FORM_CLASSES))
    assert len(result.case_sha256) == len(oracle.CASES)


def test_the_fingerprint_is_a_function_of_the_tree_alone():
    """Two runs in one process step to the same words, case by case."""
    first, second = oracle.run(), oracle.run()
    assert first.case_sha256 == second.case_sha256
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert first.form_classes == second.form_classes


def test_a_missing_form_class_is_named_and_refuses():
    """Dropping a case's absorber leaves its declared recurrence classes at zero cells."""
    without_layer = tuple(dataclasses.replace(spec, pml=None) if spec.name == "pml_3d"
                          else spec for spec in oracle.CASES)
    result = oracle.run(cases=without_layer)
    problems = oracle.coverage_problems(result.form_classes)
    assert len(problems) == 1, problems
    assert "pml_3d/curl_B/Bx/recurrence" in problems[0]
    assert "pml_3d/constitutive_E/Ez/aux" in problems[0]


def test_an_arithmetic_change_to_the_array_path_moves_the_fingerprint(monkeypatch):
    """The pairwise curl grouping in place of MEEP's: every curl-carrying case moves.

    ``(g1s - g1) + (g2 - g2s)`` against the shipped left-to-right
    ``((g1s - g1) + g2) - g2s`` (step_generic.cpp:104), on every term with both partners
    present: one regrouping of one function, which is the class of edit the tripwire
    exists for.
    """
    from . import stepping

    shipped = oracle.live()
    original = stepping._curl_from_operands

    def regrouped(operands, dtdx, presence, scratch=None):
        if presence != (True, True):
            return original(operands, dtdx, presence, scratch=scratch)
        return dtdx * ((operands.shifted_first - operands.first)
                       + (operands.second - operands.shifted_second))

    monkeypatch.setattr(stepping, "_curl_from_operands", regrouped)
    moved = oracle.run()
    assert moved.fingerprint_sha256 != shipped.fingerprint_sha256
    changed = sorted(name for name in shipped.case_sha256
                     if shipped.case_sha256[name] != moved.case_sha256[name])
    assert changed == sorted(shipped.case_sha256), changed
    problems = oracle.ledger_problems(_synthetic_ledger(shipped), moved, "planted")
    assert any("moved since" in problem and "pml_3d" in problem for problem in problems), \
        problems
    assert any("byte-binding entries record another array path" in problem
               for problem in problems), problems


def test_the_ledger_rule_names_each_way_a_ledger_can_fail():
    result = oracle.live()
    ledger = _synthetic_ledger(result)
    assert oracle.ledger_problems(ledger, result, "t") == []

    missing = _synthetic_ledger(result)
    del missing["a_weld"][oracle.LEAF_KEY]
    assert any("carry no array_path_oracle leaf" in p
               for p in oracle.ledger_problems(missing, result, "t"))

    stray = _synthetic_ledger(result)
    stray["unbound"] = {"purpose": "x", oracle.LEAF_KEY: oracle.entry_leaf(result)}
    assert any("bind no bytes" in p for p in oracle.ledger_problems(stray, result, "t"))

    one_case = _synthetic_ledger(result)
    one_case["driver_dispatch"][oracle.LEAF_KEY]["case_sha256"]["no_pml_3d"] = "2" * 64
    problems = oracle.ledger_problems(one_case, result, "t")
    assert any(f"1 of {len(oracle.CASES)} fixture cases" in p and "no_pml_3d" in p
               for p in problems), problems

    no_dispatch_leaf = _synthetic_ledger(result)
    del no_dispatch_leaf["driver_dispatch"][oracle.LEAF_KEY]
    assert any("driver_dispatch carries no" in p
               for p in oracle.ledger_problems(no_dispatch_leaf, result, "t"))

    counts = _synthetic_ledger(result)
    counts["driver_dispatch"][oracle.LEAF_KEY]["form_classes"]["pml_3d/curl_B/Bx/plain"] += 1
    assert any("form_classes differ" in p for p in oracle.ledger_problems(counts, result, "t"))


def test_the_provenance_is_reported_and_never_judged():
    """A provenance digest that names other code is not a problem; the fingerprint is."""
    result = oracle.live()
    ledger = _synthetic_ledger(result)
    leaf = ledger["driver_dispatch"][oracle.LEAF_KEY]
    assert leaf["provenance_absent"] == []
    assert "meep_gpu/chunk_map.py" in leaf["provenance_code_sha256"]
    assert "meep_gpu/stepping.py" in leaf["provenance_code_sha256"]
    leaf["provenance_code_sha256"]["meep_gpu/stepping.py"] = "3" * 64
    assert oracle.ledger_problems(ledger, result, "t") == []
    found = walk.census(ledger)
    rules = {walk.dotted(f.trail): f.rule for f in found}
    assert rules["driver_dispatch.array_path_oracle.provenance_code_sha256."
                 "meep_gpu/stepping.py"] == "array_path_oracle_provenance"
    assert rules["a_weld.array_path_oracle.fingerprint_sha256"] == \
        "array_path_oracle_fingerprint"
    assert all(f.tier is None for f in found if f.rule.startswith("array_path_oracle"))


def test_the_membership_line_is_the_one_rank_chunk_rule():
    """The class counts use the one-rank chunk rule, written apart from the package's own.

    Checked against ``chunk_map.ChunkLayout.one_rank`` on every case, axis and lattice,
    so the two cannot disagree while both exist.
    """
    from .chunk_map import ChunkLayout  # noqa: PLC0415

    compared = 0
    for spec in oracle.CASES:
        if spec.pml is None:
            continue
        driver, _sources, _currents = oracle._build(spec)  # noqa: SLF001
        try:
            pml, grid = driver.pml, driver.grid
            layout = ChunkLayout.one_rank(grid, pml)
            for axis, name in enumerate("xyz"):
                for half in (False, True):
                    line = oracle.membership_line(int(grid.shape[axis]),
                                                  pml.thickness_by_face[axis], half,
                                                  grid.axis_wraps(axis))
                    assert numpy.array_equal(line, layout.membership(axis, half)), \
                        (spec.name, name, half)
                    compared += 1
        finally:
            driver.close()
    assert compared >= 60


def test_the_chunk_presence_is_the_one_rank_rule_and_the_fixture_reaches_both_kinds():
    """The per-chunk conductivity classes use the oracle's own rule, written apart from
    ``chunk_map``'s; checked against ``RegionPlan.conductive_mask`` on every conductive
    component of every case with a conductivity under a layer, owned cells only. The
    case built for it reaches chunks of both kinds; the others keep the conductivity in
    every chunk."""
    from .chunk_map import ChunkLayout, RegionPlan  # noqa: PLC0415
    from .fields import IYEE_SHIFTS  # noqa: PLC0415

    compared = 0
    kinds = {}
    for spec in oracle.CASES:
        if spec.pml is None or not (spec.d_conductivity or spec.b_conductivity):
            continue
        driver, _sources, _currents = oracle._build(spec)  # noqa: SLF001
        try:
            pml, grid, fields = driver.pml, driver.grid, driver.fields
            layout = ChunkLayout.one_rank(grid, pml)
            plan = RegionPlan(layout, grid, pml)
            shape = tuple(int(n) for n in grid.shape)
            for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
                sigma = fields.conductivity_for(component)
                if sigma is None:
                    continue
                shifts = IYEE_SHIFTS[component]
                ours = oracle.chunk_keeps(numpy.asarray(sigma), [
                    (shape[a], pml.thickness_by_face[a] if not grid.is_invariant(a) else (0, 0),
                     bool(shifts[a]), bool(grid.axis_wraps(a)) and not grid.is_invariant(a))
                    for a in range(3)])
                theirs = numpy.asarray(plan.conductive_mask(component, sigma))
                owned = numpy.ones(shape, dtype=bool)
                for a in range(3):
                    owned &= layout.owned(a, bool(shifts[a])).reshape(
                        [-1 if b == a else 1 for b in range(3)])
                assert numpy.array_equal(ours[owned], theirs[owned]), (spec.name, component)
                kinds.setdefault(spec.name, set()).update(bool(v) for v in numpy.unique(ours[owned]))
                compared += 1
        finally:
            driver.close()
    assert compared >= 12, compared
    assert kinds["conductive_chunks_pml_2d"] == {False, True}, kinds
    assert kinds["conductive_pml_3d"] == {True}, kinds


def test_a_libm_call_while_stepping_is_a_named_refusal():
    with pytest.raises(oracle.OracleRefusal, match=r"numpy\.exp"):
        with oracle._no_libm_calls("planted"):  # noqa: SLF001
            numpy.exp(numpy.float32(1.0))
    with pytest.raises(oracle.OracleRefusal, match=r"math\.sin"):
        with oracle._no_libm_calls("planted"):  # noqa: SLF001
            math.sin(0.5)
    assert float(numpy.exp(numpy.float64(0.0))) == 1.0
    assert math.sin(0.0) == 0.0


def test_the_pml_grading_holds_no_libm_value():
    """Inside the fixture's build, ``PML.prefac`` is the stored constant, and only there."""
    from .pml import PML

    shipped = PML.prefac
    with oracle._libm_free_pml_grading():  # noqa: SLF001
        assert PML.prefac is oracle._libm_free_prefac  # noqa: SLF001
    assert PML.prefac is shipped


def test_a_flushing_fpu_that_cannot_be_driven_to_keep_refuses(monkeypatch):
    from . import backends, subnormal_policy

    monkeypatch.setattr(backends, "subnormals_flushed", lambda: True)
    monkeypatch.setattr(subnormal_policy, "_drive_host_fpu",
                        lambda want_flush: (None, "no lever"))
    with pytest.raises(oracle.OracleRefusal, match="could not be driven to keep"):
        oracle.run(cases=oracle.CASES[:1])


def test_a_stored_current_outside_the_table_refuses():
    table = oracle._StoredCurrents(7, 2.0 ** -4)  # noqa: SLF001
    assert table(0.5 * 2.0 ** -4) == table.table[0]
    assert all(value.real != 0 and value.imag != 0 for value in table.table)
    with pytest.raises(oracle.OracleRefusal, match="outside the"):
        table(100.0)


def test_a_complex_source_amplitude_refuses():
    class _Source:
        _point_amps = numpy.array([1.0 + 0.5j], dtype=numpy.complex64)

    with pytest.raises(oracle.OracleRefusal, match="imaginary"):
        oracle._require_real_amplitudes("planted", _Source())  # noqa: SLF001


def test_the_bound_fields_and_the_walker_key_agree_with_their_owners():
    assert oracle.BOUND_FIELDS == fastpath.BOUND_FIELDS
    assert walk.ARRAY_PATH_ORACLE == oracle.LEAF_KEY


def test_the_seeded_words_are_integer_arithmetic_and_plant_both_zeros():
    words = oracle._state_words(1234, (64, 64))  # noqa: SLF001
    bits = words.view(numpy.uint32)
    assert numpy.count_nonzero(bits == 0) > 0
    assert numpy.count_nonzero(bits == 0x80000000) > 0
    subnormal = (bits & 0x7F800000) == 0
    assert numpy.count_nonzero(subnormal & ((bits & 0x7FFFFF) != 0)) > 0
    again = oracle._state_words(1234, (64, 64))  # noqa: SLF001
    assert numpy.array_equal(bits, again.view(numpy.uint32))
