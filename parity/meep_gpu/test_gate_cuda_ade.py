"""The ADE CUDA gate's clauses that need no device -- which is most of them.

THE GATE ITSELF NEEDS A GPU FOR ONE THING ONLY: comparing NVRTC's output against
the array path. Everything that decides WHAT THAT COMPARISON MEANS runs here, and
every one of them fails as a silent wrong ANSWER rather than as a crash:

* whether each planted defect still matches the text it was written against. A
  needle that has drifted apart from ``ade_kernels.py`` exercises nothing and
  reports a pass while doing it, and this family's device code is TWO strings, so
  a needle can rot on one and still match the other;
* whether the four vacuity floors refuse what they exist to refuse. Zero-init is a
  fixed point of this recurrence and a kernel that never launched leaves the twin
  fixture exactly as the oracle found it -- so "the bytes agreed" is worth nothing
  until the record says the array path MOVED, the drive term was live, the history
  term was live, and the launch counter came out at the number owed;
* whether the multi-step expectation really requires BOTH halves. ``h3`` is the
  leg that makes the 60-launch budget mean something, and an adjudicator that
  scored it on the caught half alone would pass a gate whose single-launch control
  had silently started diverging too;
* whether a run on the off-device backend can be mistaken for a certification.

The device legs are exercised here against fixtures that are DELIBERATELY WRONG,
so "the gate refuses when it should" is a measurement rather than a property
nobody tried to break.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
REPO_API = HERE.parents[1]
if str(REPO_API) not in sys.path:
    sys.path.insert(0, str(REPO_API))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

GATE_PATH = HERE / "gate_cuda_ade.py"
KERNEL_MODULE = REPO_API / "meep_gpu" / "cuda_kernels" / "ade_kernels.py"


def load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = load(GATE_PATH, "gate_cuda_ade_under_test")


@pytest.fixture(scope="module")
def device_text():
    """The two SHIPPED device strings, read off the syntax tree (no CuPy needed)."""
    return gate.device_source_texts()


@pytest.fixture(scope="module")
def backend(device_text):
    return gate.EvaluatorBackend(device_text)


@pytest.fixture
def xp():
    return gate._NumpyWearingCupysName()


def base_spec(**overrides):
    spec = {"shape": (8, 8, 8), "label": "8x8x8",
            "boundaries": gate.BOUNDARY_TRIPLES[0],
            "kind": gate.LORENTZIAN, "sigma_form": "volume",
            "layout": "one_state_two_components", "states": 1,
            "driven": ("Ex", "Ez"), "courant": 0.5, "value_class": "uniform"}
    spec.update(overrides)
    return spec


# --------------------------------------------------------------------------
# THE NEEDLES STILL MATCH THE SHIPPED TEXT
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(gate.SOURCE_MUTATIONS))
def test_every_planted_defect_still_matches_both_device_strings(name, device_text):
    """A mutation that matches nothing exercises nothing and reports a pass.

    BOTH STRINGS, counted separately. The volume and uniform variants differ in one
    parameter declaration and one load; a needle anchored on the volume variant's
    ``float s = sigma[idx];`` would match nothing in the uniform one, and a leg
    armed on both would then be reporting a verdict for a kernel it never touched.
    """
    transform = gate.SOURCE_MUTATIONS[name]
    for volume, text in device_text.items():
        mutated, count = transform(text)
        assert count > 0, (
            f"{name} matched nothing in the "
            f"{'volume' if volume else 'uniform'} device string; it has drifted "
            f"apart from ade_kernels.py")
        assert mutated != text


def test_the_reference_expression_is_the_one_the_needles_are_cut_against(device_text):
    """The gate's own copy of the expression, pinned against the shipped source.

    ``_EXPRESSION`` is what three needles are anchored on. If ``ade_kernels.py``
    rewrites the line, those three stop matching -- and the test above would catch
    that -- but this says WHICH string moved, which is the difference between a
    diagnosable failure and a puzzling one.
    """
    for text in device_text.values():
        assert f"p_out[idx] = {gate._EXPRESSION};" in text


def test_the_two_nulls_and_their_discriminators_edit_the_same_expressions(device_text):
    """A battery of only-must-be-caught legs scores identically whether the
    comparator works or has degenerated into failing everything. Each null is
    therefore paired with a leg editing the SAME sub-expression that must be
    caught, and the pairing is asserted rather than left to the prose."""
    text = device_text[True]
    commuted, _ = gate.n_commute_the_history_product(text)
    rebound, _ = gate.m_swap_c_now_and_c_prev(text)
    assert commuted != text and rebound != text and commuted != rebound
    regrouped, _ = gate.m_associate_c_drive_with_sigma(text)
    reordered, _ = gate.n_commute_the_drive_product(text)
    assert regrouped != text and reordered != text and regrouped != reordered


# --------------------------------------------------------------------------
# THE EVALUATOR REALLY EXECUTES THE TEXT
# --------------------------------------------------------------------------

def test_the_evaluator_reads_the_source_rather_than_a_transcription(device_text):
    """Mutate the text and the laptop backend must compute something else.

    THE HAZARD: an evaluator that had the expression hard-coded would score every
    source mutation a null and every leg would pass. The whole value of the off-
    device backend is that the same bytes drive it.
    """
    shape = (4, 4, 4)
    rng = np.random.default_rng(7)
    arrays = {name: rng.uniform(-1, 1, shape).astype(np.float32)
              for name in ("p_out", "p_now", "p_prev", "drive", "sigma")}
    scalars = {"c_now": np.float32(1.9), "c_prev": np.float32(-0.98),
               "c_drive": np.float32(0.11)}
    pristine = {k: v.copy() for k, v in arrays.items()}
    gate.evaluate_device_source(device_text[True], arrays, scalars)
    shipped = arrays["p_out"].copy()

    arrays = {k: v.copy() for k, v in pristine.items()}
    mutated, _ = gate.m_drop_sigma(device_text[True])
    gate.evaluate_device_source(mutated, arrays, scalars)
    assert not np.array_equal(shipped, arrays["p_out"])

    # And the null must really be a null THROUGH THE EVALUATOR, not merely in the
    # text: IEEE multiply commutes bitwise and this is where that is measured.
    arrays = {k: v.copy() for k, v in pristine.items()}
    commuted, _ = gate.n_commute_the_drive_product(device_text[True])
    gate.evaluate_device_source(commuted, arrays, scalars)
    assert np.array_equal(shipped.view(np.uint32), arrays["p_out"].view(np.uint32))


def test_the_evaluator_refuses_a_statement_it_has_no_grammar_for(device_text):
    """Silently skipping a line is how a leg comes to measure the parser.

    The device backend has no such hazard -- the compiler executes every line --
    which is exactly why the two backends' catch tables differ and both are
    reported.
    """
    broken = device_text[True].replace("float p = p_now[idx];",
                                       "float p = fmaf(p_now[idx], 1.0f, 0.0f);")
    assert gate.evaluator_blind_lines(broken), (
        "a statement outside the grammar must be visible to the classifier")
    with pytest.raises(ValueError):
        gate.evaluate_device_source(broken, {"p_out": np.zeros((2, 2, 2), np.float32)},
                                    {})


# --------------------------------------------------------------------------
# THE FOUR FLOORS REFUSE WHAT THEY EXIST TO REFUSE
# --------------------------------------------------------------------------

def test_the_shipped_fixture_clears_every_floor(backend, xp):
    record = gate.one_case(backend, xp, base_spec(), "fmad_false",
                           ("--fmad=false",), 1)
    valid, why = gate.case_is_valid(record)
    assert valid, why
    assert record["bit_identical"]
    assert record["fixtures_started_identical"]
    assert record["oracle_moved_words"] > 0
    assert record["drive_term_max_abs"] > 0.0
    assert record["history_term_max_abs"] > 0.0
    assert record["launches"] == record["expected_launches"] == 2
    assert record["predicate_covers"], record["predicate_reason"]


def test_a_fixture_that_is_a_fixed_point_is_refused_not_passed():
    """Zero-init IS a fixed point of ``P = c_now*P + c_prev*P_prev + c_drive*s*W``.

    A gate that scored such a case would report bit-identity between an oracle that
    did nothing and a kernel that did nothing, forever.
    """
    record = {"fixtures_started_identical": True, "oracle_moved": False,
              "oracle_all_finite": True, "drive_term_is_live": True,
              "history_term_is_live": True, "launch_count_as_expected": True,
              "launch_error": None}
    valid, why = gate.case_is_valid(record)
    assert not valid and "fixed point" in why


@pytest.mark.parametrize("floor,fragment", [
    ("drive_term_is_live", "cannot distinguish the drive binding"),
    ("history_term_is_live", "first order"),
    ("oracle_all_finite", "overflowed"),
    ("fixtures_started_identical", "two different problems"),
])
def test_each_floor_refuses_by_name(floor, fragment):
    record = {"fixtures_started_identical": True, "oracle_moved": True,
              "oracle_all_finite": True, "drive_term_is_live": True,
              "history_term_is_live": True, "launch_count_as_expected": True,
              "launch_error": None}
    record[floor] = False
    valid, why = gate.case_is_valid(record)
    assert not valid and fragment in why


def test_a_launch_that_never_happened_is_refused_even_when_the_bytes_agree():
    """BYTES ALONE CANNOT PROVE THE KERNEL RAN. Two fixtures seeded identically and
    a launcher that did nothing differ only in that the oracle moved -- so the
    launch counter is a clause of validity, not a diagnostic."""
    record = {"fixtures_started_identical": True, "oracle_moved": True,
              "oracle_all_finite": True, "drive_term_is_live": True,
              "history_term_is_live": True, "launch_count_as_expected": False,
              "launches": 0, "expected_launches": 6, "launch_error": None}
    valid, why = gate.case_is_valid(record)
    assert not valid and "cannot prove the kernel ran" in why


def test_a_dead_drive_field_is_measured_as_a_dead_drive_term(backend, xp):
    """The floor is taken on the FIXTURE, not asserted about it.

    Zero the drive and ``c_drive * (sigma * w)`` is identically zero: the case can
    no longer distinguish ``drop_sigma`` or the wrong-drive control, and the gate
    has to say so rather than count the pass.
    """
    spec = base_spec()
    fields, layer, grid, _ = gate.build(xp, spec, 11)
    for component in gate.COMPONENTS:
        getattr(fields, "f_w_" + component)[...] = 0.0
    floors = gate.term_magnitudes(fields)
    assert floors["drive_term_max_abs"] == 0.0
    assert floors["history_term_max_abs"] > 0.0


# --------------------------------------------------------------------------
# THE PLANTED DEFECTS, THROUGH THE WHOLE HARNESS
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["reassociate_three_terms",
                                  "associate_c_drive_with_sigma",
                                  "drop_previous_history", "swap_p_and_q",
                                  "swap_c_now_and_c_prev", "drop_sigma"])
def test_each_defect_diverges_through_the_gate(backend, xp, name, device_text):
    mutated, sites = gate.mutate_sources(device_text, gate.SOURCE_MUTATIONS[name])
    assert sites > 0
    backend.set_sources(mutated)
    try:
        record = gate.one_case(backend, xp, base_spec(), "fmad_false",
                               ("--fmad=false",), 1, source_mutation=name)
    finally:
        backend.disarm()
    assert gate.case_is_valid(record)[0]
    assert not record["bit_identical"], f"{name} was not caught"
    assert record["differing_floats"] > 0


@pytest.mark.parametrize("name", ["commute_the_history_product",
                                  "commute_the_drive_product",
                                  "reorder_the_two_loads"])
def test_each_null_really_is_a_null_through_the_gate(backend, xp, name, device_text):
    mutated, sites = gate.mutate_sources(device_text, gate.SOURCE_MUTATIONS[name])
    assert sites > 0
    backend.set_sources(mutated)
    try:
        record = gate.one_case(backend, xp, base_spec(), "fmad_false",
                               ("--fmad=false",), 1, source_mutation=name)
    finally:
        backend.disarm()
    assert record["bit_identical"], (
        f"{name} was predicted inert and diverged: {record['differing_floats']} "
        f"words")


def test_the_arming_is_undone_and_the_check_can_fail(backend, device_text):
    """A gate whose arming leaked would report catches forever."""
    mutated, _ = gate.mutate_sources(device_text, gate.m_drop_sigma)
    backend.set_sources(mutated)
    assert not backend.is_pristine()
    assert backend.disarm()
    assert backend.is_pristine()


# --------------------------------------------------------------------------
# THE HOST DEFECTS -- the ones no edit to the device text can reach
# --------------------------------------------------------------------------

def test_the_wrong_drive_is_caught_and_is_a_null_where_the_two_agree(backend, xp):
    """THE CONTROL THIS FAMILY MOST OWES, and its discriminator.

    ``drive_field`` returns ``f_w`` under an active layer and the stored E without
    one, and the two agree EXACTLY outside the absorber -- so binding the wrong one
    is invisible in every no-PML test. Catching it is only a statement about the
    POINTER if the same planted binding is a null when the two arrays agree.
    """
    caught = gate.one_case(backend, xp, base_spec(), "fmad_false",
                           ("--fmad=false",), 1, wrong_drive=True)
    assert gate.case_is_valid(caught)[0]
    assert not caught["bit_identical"]
    assert caught["differing_floats"] > 0

    null = gate.one_case(backend, xp, base_spec(drive_equals_stored_E=True),
                         "fmad_false", ("--fmad=false",), 1, wrong_drive=True)
    assert gate.case_is_valid(null)[0]
    assert null["bit_identical"], (
        "with the stored E seeded equal to f_w the wrong binding must be inert; "
        "otherwise the catch above is a statement about some other difference")


def test_stale_pointers_are_invisible_at_one_launch_and_caught_at_sixty(backend, xp):
    """THE ROTATION LEG, and the whole reason the multi-step budget is here.

    One driven component and one launch: the stale views ARE the fresh ones and the
    defect cannot be seen. Sixty launches: a rotation has happened in between and it
    is wrong from the second onward. Both halves are asserted, because either alone
    would be satisfied by a harness that had stopped working.
    """
    spec = base_spec(layout="one_state_one_component", driven=("Ex",))
    single = gate.one_case(backend, xp, spec, "fmad_false", ("--fmad=false",), 1,
                           host_plan="stale_pointers")
    multi = gate.one_case(backend, xp, spec, "fmad_false", ("--fmad=false",), 12,
                          host_plan="stale_pointers")
    assert gate.case_is_valid(single)[0] and gate.case_is_valid(multi)[0]
    assert single["bit_identical"], (
        "one component, one launch: the stale plan IS the shipped plan here, and a "
        "divergence would mean the two differ for some other reason")
    assert not multi["bit_identical"], (
        "twelve launches with the pointers frozen must diverge, or the multi-step "
        "budget is buying nothing")


def test_the_rotation_itself_is_load_bearing(backend, xp):
    """Launch correctly and never move the three names: this step's result is
    written into the scratch and then discarded."""
    record = gate.one_case(backend, xp, base_spec(), "fmad_false",
                           ("--fmad=false",), 1, host_plan="no_rotation")
    assert not record["bit_identical"]


def test_the_scalar_specialization_bound_to_a_graded_sigma_is_caught(backend, xp):
    """``coverage.ade_sigma_is_volume`` is ONE function so the compile-time choice
    and the predicate cannot disagree. This is what their disagreeing costs."""
    record = gate.one_case(backend, xp, base_spec(sigma_form="volume"),
                           "fmad_false", ("--fmad=false",), 1,
                           host_plan="scalar_sigma_for_a_volume")
    assert gate.case_is_valid(record)[0]
    assert not record["bit_identical"]


def test_a_run_with_no_launch_at_all_diverges(backend, xp):
    """The floor under every other leg. If this passed, the gate would be comparing
    two copies of the same untouched fixture."""
    record = gate.one_case(backend, xp, base_spec(), "fmad_false",
                           ("--fmad=false",), 1, host_plan="no_launch")
    assert record["launches"] == 0
    assert gate.case_is_valid(record)[0], "the leg asked for zero launches"
    assert not record["bit_identical"]


def test_allclose_lets_a_real_defect_through(backend, xp, device_text):
    """The magnitude comparison a byte gate exists to replace, MEASURED.

    The reassociation is algebraically equal and differs by about one ulp relative
    -- far below ``rtol=1e-5`` -- so a tolerant comparator reports a clean pass on a
    kernel that is not the array path.
    """
    mutated, _ = gate.mutate_sources(device_text, gate.m_reassociate_three_terms)
    backend.set_sources(mutated)
    try:
        relaxed = gate.one_case(backend, xp, base_spec(), "fmad_false",
                                ("--fmad=false",), 1,
                                comparator=gate.allclose_compare)
        exact = gate.one_case(backend, xp, base_spec(), "fmad_false",
                              ("--fmad=false",), 1)
    finally:
        backend.disarm()
    assert relaxed["bit_identical"], "np.allclose was expected to be blind here"
    assert not exact["bit_identical"], "the byte comparator must not be"


# --------------------------------------------------------------------------
# THE ADJUDICATOR
# --------------------------------------------------------------------------

def case(identical: bool, steps: int = 1):
    return {"case_is_valid": True, "bit_identical": identical, "steps": steps}


@pytest.mark.parametrize("expect,cases,required", [
    ("pass", [case(True), case(True)], True),
    ("pass", [case(True), case(False)], False),
    ("caught", [case(False), case(False)], True),
    ("caught", [case(False), case(True)], False),
    ("uncaught", [case(True), case(True)], True),
    ("uncaught", [case(True), case(False)], False),
    # No case at all is NEVER as required: an empty leg scores like a perfect one
    # under any rule that only counts disagreements.
    ("pass", [], False),
    ("caught", [], False),
    ("uncaught", [], False),
])
def test_the_adjudicator_scores_each_expectation(expect, cases, required):
    assert gate.adjudicate({"expect": expect}, cases)["as_required"] is required


@pytest.mark.parametrize("single,multi,required", [
    (True, False, True),      # the measurement: inert at one launch, caught at N
    (True, True, False),      # never caught: the budget bought nothing
    (False, False, False),    # caught at one launch too: not a rotation claim
    (False, True, False),
])
def test_the_multi_step_expectation_requires_both_halves(single, multi, required):
    """``h3`` is what makes the 60-launch budget a statement about the ROTATION.

    An adjudicator that scored the caught half alone would pass a gate whose
    single-launch control had started diverging for an unrelated reason -- which is
    the state in which "caught only at depth" stops being true.
    """
    cases = [case(single, steps=1), case(multi, steps=60)]
    verdict = gate.adjudicate({"expect": "caught_only_multi_step"}, cases)
    assert verdict["as_required"] is required


def test_the_multi_step_expectation_refuses_a_leg_missing_a_half():
    for cases in ([case(True, 1)], [case(False, 60)]):
        assert gate.adjudicate(
            {"expect": "caught_only_multi_step"}, cases)["as_required"] is False


# --------------------------------------------------------------------------
# NO RUN OFF DEVICE CAN BE MISTAKEN FOR A CERTIFICATION
# --------------------------------------------------------------------------

def test_the_evaluator_backend_certifies_nothing(backend):
    assert backend.certifies is False


def test_a_full_numpy_run_reports_passed_false_and_says_why(tmp_path):
    out = tmp_path / "gate.json"
    code = gate.main(["--backend", "numpy", "--product", "reduced",
                      "--legs", "bytes", "--out", str(out)])
    import json
    payload = json.loads(out.read_text())
    assert code == 0, "the harness itself is expected to be green off device"
    assert payload["certifies"] is False
    assert payload["verdict"]["passed"] is False
    assert "compiles nothing" in payload["verdict"]["why_not_certified"]
    assert payload["bytes"]["summary"]["per_guard"]["fmad_false"]["ran"] > 0


def test_the_canonical_verdict_reads_passed_and_not_certifies(tmp_path):
    """``gate_provenance.read_verdict`` falls back to ``certifies`` -- a key that
    says only that the backend COULD certify -- so the gate writes ``passed`` at
    top level, and writes it FALSE before any leg runs.

    Measured hazard: without the early write, an artifact from a run that died
    mid-leg would read as RELEASED, because ``certifies`` is true from the first
    save on a CuPy host.
    """
    from gate_provenance import read_verdict
    assert read_verdict({"certifies": True, "passed": False})["released"] is False
    assert read_verdict({"certifies": True, "passed": True})["released"] is True
    # And the bare shape a partially written artifact would have.
    assert read_verdict({"certifies": True})["released"] is True, (
        "this is the fallback the early passed=False write exists to pre-empt")
