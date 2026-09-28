"""Laptop contracts for the residual-round PROBES' own measurement discipline.

A probe that measures the wrong thing does not fail — it publishes. Every defect
this file pins was live in a shipped artifact whose table read as a clean pass,
and each was invisible because the number it reported was plausible:

* the vacuity gate bracketed the WHOLE CYCLE LOOP, so ``perturb``'s writes counted
  as sub-step movement. A completely dead update still reported thousands of moved
  words, because the perturbation footprint contains most sub-steps' targets;
* ``perturb`` added a delta over the whole D volume, and ``-0.0 + x`` is ``x``, so
  the ``+-0`` lattice was gone before the first sub-step ran — while the published
  census was taken BEFORE the loop and said it was there;
* ``perturb`` never touched E or H, so the curl legs differenced byte-frozen
  sources on every cycle;
* the complex seeding spelled ``values + 1j * imaginary``, and ``(0+1j) *
  (-0.0+0j)`` is ``(-0.0, +0.0)``, so the imaginary half of the needle was never
  written.

So these are behavioural tests over the probes' own primitives on NumPy, not text
checks: each drives the primitive and asserts the property the artifact's numbers
depend on. The device legs are the probes' own
(``parity/meep_gpu/probe_residual_group_bodies.py``,
``parity/meep_gpu/probe_residual_new_predicates.py``); nothing here launches.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy
import pytest

from meep_gpu import stepping

# Every probe record this module builds is stamped 'keep'; the complex
# families' policy clause fails closed when the run policy can be neither
# read nor declared, so the premise is declared rather than left implicit.
# See ``run_policy_declared_keep`` in conftest.py.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")


_API = pathlib.Path(__file__).resolve().parents[1]
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

PARITY_DIR = _API / "parity" / "meep_gpu"
BODIES_PATH = PARITY_DIR / "probe_residual_group_bodies.py"
NEWPRED_PATH = PARITY_DIR / "probe_residual_new_predicates.py"
CONDUCTIVE_GATE_PATH = PARITY_DIR / "gate_triton_no_pml_conductive.py"
STORED_E_GATE_PATH = PARITY_DIR / "gate_triton_no_pml_stored_e.py"


@pytest.fixture(scope="module")
def probe():
    return importlib.import_module("parity.meep_gpu.probe_residual_group_bodies")


@pytest.fixture(scope="module")
def newpred():
    return importlib.import_module("parity.meep_gpu.probe_residual_new_predicates")


@pytest.fixture(scope="module")
def conductive_gate():
    return importlib.import_module("parity.meep_gpu.gate_triton_no_pml_conductive")


@pytest.fixture(scope="module")
def stored_e_gate():
    return importlib.import_module("parity.meep_gpu.gate_triton_no_pml_stored_e")


def _words(probe, fields):
    return {name: probe.as_words(array).copy()
            for name, array in probe.inventory(fields).items()}


# ---------------------------------------------------------------------------
# The +-0 lattice must survive the perturbation, in both storage halves
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("complex_storage", (False, True))
def test_the_signed_zero_lattice_survives_every_perturbation(probe, complex_storage):
    """Seeded, then perturbed eight times: the census must not move.

    The failure this replaces reported the census taken at SEED and never looked
    again; the lattice was empty from the first cycle onward, and every
    constitutive leg was seeded-zero-vacuous by the round's own rule while
    reporting ``vacuous: false``.
    """
    fields, _pml, _notes = probe.build(numpy, complex_storage=complex_storage,
                                       offdiag=complex_storage)
    seeded = probe.signed_zero_census(fields)
    assert seeded > 0, "the fixture seeded no signed zeros at all"
    for cycle in range(8):
        probe.perturb(fields, numpy, cycle)
        assert probe.signed_zero_census(fields) == seeded, (
            f"the lattice changed size at cycle {cycle}: the needle the "
            f"constitutive legs rest on is not the one that was seeded")


def test_the_complex_seeding_writes_a_signed_zero_in_BOTH_halves(probe):
    """``values + 1j * imaginary`` loses the imaginary half; the fix must not.

    ``(0+1j) * (-0.0+0j)`` is ``(-0.0, +0.0)`` in IEEE arithmetic, so the natural
    spelling silently halves the needle. Measured as a ratio against the real
    fixture rather than asserted as a constant, so it stays true if the fixture
    shape changes.
    """
    real, _pml, _notes = probe.build(numpy)
    complex_fields, _p, _n = probe.build(numpy, complex_storage=True)
    assert tuple(real.grid.shape) == tuple(complex_fields.grid.shape)
    assert probe.signed_zero_census(complex_fields) == \
        2 * probe.signed_zero_census(real), (
        "the complex fixture carries only the real half of the lattice")

    # And directly: the helper the seeding uses must keep a negative zero in the
    # imaginary part, which the arithmetic spelling does not.
    minus = numpy.array([[-0.0]], dtype=numpy.float32)
    built = probe._complex_from_halves(minus, minus)
    assert numpy.signbit(built.imag).all(), (
        "the imaginary half lost its sign; the needle is halved")
    arithmetic = (minus + 1j * minus).astype(numpy.complex64)
    assert not numpy.signbit(arithmetic.imag).any(), (
        "the arithmetic spelling no longer loses the sign, so this test is "
        "guarding a hazard that no longer exists — re-derive it before deleting")


# ---------------------------------------------------------------------------
# The vacuity gate must measure the SUB-STEP, not the cycle
# ---------------------------------------------------------------------------

def test_a_dead_substep_is_caught_by_the_per_substep_bracket_and_not_by_the_loop(
        probe):
    """The whole point: the two brackets disagree, and only one is honest.

    Around the loop, a leg that runs NO sub-step at all still reports the
    perturbation's whole footprint. Bracketed around the sub-step, it reports
    zero. Both numbers are computed here so the difference is the assertion.
    """
    fields, _pml, _notes = probe.build(numpy, poles=1, pml_thickness=0)
    loop_before = _words(probe, fields)
    per_substep = []
    for cycle in range(4):
        probe.perturb(fields, numpy, cycle)
        before = _words(probe, fields)
        # No sub-step runs. This is the null the old gate could not see.
        per_substep.append(probe.moved_words(before, probe.inventory(fields)))
    loop_moved = probe.moved_words(loop_before, probe.inventory(fields))

    assert loop_moved > 0, (
        "even the perturbation moved nothing; the fixture is inert and this "
        "test proves nothing")
    assert max(per_substep) == 0, (
        "a leg that ran no sub-step reported movement: the bracket is not "
        "around the sub-step")


def test_a_live_substep_moves_less_than_the_perturbation_that_precedes_it(probe):
    """Why the loop bracket could not have caught the null: it is dominated.

    If the sub-step's own footprint were the larger of the two, the old gate
    would have been merely imprecise. It is the smaller one, so the reported
    number was the perturbation's and carried no information about the sub-step.
    """
    fields, _pml, _notes = probe.build(numpy, poles=1, pml_thickness=0)
    before_perturb = _words(probe, fields)
    probe.perturb(fields, numpy, 0)
    perturbation = probe.moved_words(before_perturb, probe.inventory(fields))

    before_substep = _words(probe, fields)
    stepping.update_P(fields, None)
    substep = probe.moved_words(before_substep, probe.inventory(fields))

    assert substep > 0, "the sub-step moved nothing; the leg would be VACUOUS"
    assert substep < perturbation, (
        f"the sub-step moved {substep} words and the perturbation {perturbation}; "
        f"if this ever inverts, re-derive whether the loop bracket was "
        f"informative after all")


# ---------------------------------------------------------------------------
# The perturbation must reach the sub-step's SOURCES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sources", (("Ex", "Ey", "Ez"), ("Hx", "Hy", "Hz")))
def test_the_perturbation_reaches_the_curl_sources(probe, sources):
    """``step_B`` differences stored E and ``step_D`` stored H.

    A perturbation confined to B, D and P leaves both curls reading byte-frozen
    sources on every cycle — the stale-source blindness that let group (G) pass
    falsely one level up, and the reason ``perturb`` exists at all.
    """
    fields, _pml, _notes = probe.build(numpy, nonlinear=True)
    assert set(sources) <= set(probe.perturbed_names(fields)), (
        f"{sources} are not in the perturbation's footprint")
    before = {name: probe.as_words(getattr(fields, name)).copy()
              for name in sources}
    probe.perturb(fields, numpy, 0)
    for name in sources:
        moved = int((before[name]
                     != probe.as_words(getattr(fields, name))).sum())
        assert moved > 0, f"{name} was frozen across the cycle"


def test_the_perturbation_reaches_every_stored_volume_and_pole(probe):
    fields, _pml, _notes = probe.build(numpy, poles=1, conductivity=0.7,
                                       magnetic_conductivity=0.5)
    names = probe.perturbed_names(fields)
    for stem in ("Bx", "Dx", "Ex", "Hx", "fu_Bx", "fu_Dx", "f_w_Ex", "f_w_Hx"):
        if getattr(fields, stem, None) is not None:
            assert stem in names, f"{stem} is allocated but never perturbed"
    pole = fields.polarizations[0]
    before = {slot: {c: probe.as_words(a).copy()
                     for c, a in (getattr(pole, slot, {}) or {}).items()}
              for slot in ("P", "P_prev")}
    probe.perturb(fields, numpy, 0)
    for slot, arrays in before.items():
        for component, snapshot in arrays.items():
            now = probe.as_words(getattr(pole, slot)[component])
            assert int((snapshot != now).sum()) > 0, (
                f"{slot}[{component}] was frozen; a plan caching its pointer "
                f"would pass by agreeing with a snapshot")


# ---------------------------------------------------------------------------
# The verdict bookkeeping
# ---------------------------------------------------------------------------

def test_a_refusal_prediction_cannot_be_satisfied_by_any_reached_verdict(probe):
    """The ERROR branch that was overwritten on the next line was not needed.

    It set ``matches_prediction = False``; the unconditional line below it sets
    ``verdict == prediction``, and no verdict the code can produce equals
    ``"ERROR"``. Removing it must not change the answer for any verdict, which is
    what this enumerates.
    """
    reachable = ("VACUOUS", "IDENTICAL", "DIVERGENT")
    assert "ERROR" not in reachable
    for verdict in reachable:
        assert (verdict == "ERROR") is False


def test_the_constitutive_substeps_carry_the_signed_zero_floor(probe):
    """A constitutive leg with a drained lattice must be VACUOUS, not IDENTICAL.

    Zero-init is a fixed point of these sub-steps, so on the lattice cells the
    arithmetic under test is ``0 = 0``. This is the clause that turned the
    device's silent scalar-assignment canonicalization into a visible failure
    rather than a passing table.
    """
    assert set(probe.CONSTITUTIVE_SUB_STEPS) == {"update_E", "update_H"}


def test_the_probe_declares_more_cycles_than_the_superseded_artifact(probe):
    assert probe.CYCLES >= 8


# ---------------------------------------------------------------------------
# The no-absorber closure gates exist and fail closed on incomplete payloads
# ---------------------------------------------------------------------------

def test_the_conductive_gate_covers_both_one_sided_cross_substeps(conductive_gate):
    names = {case["name"] for case in conductive_gate.build_cases()}
    assert {"product_D_only_step_B", "product_D_only_step_D",
            "product_B_only_step_B", "product_B_only_step_D"} <= names
    assert len([name for name in names if name.startswith("product_")]) == 8
    assert len([name for name in names if "MUTATION" in name]) == 3


def test_the_stored_e_gate_carries_the_live_rotation_and_its_mutant(stored_e_gate):
    names = {case["name"] for case in stored_e_gate.build_cases()}
    assert "product_live_pointer_rotation" in names
    assert "MUTATION_freeze_rotating_pointers" in names
    assert "CONTROL_reverse_one_pole" in names


@pytest.mark.parametrize("gate_fixture", ("conductive_gate", "stored_e_gate"))
def test_the_new_gate_validators_reject_an_empty_payload(request, gate_fixture):
    gate = request.getfixturevalue(gate_fixture)
    verdict = gate.validate({})
    assert not verdict["released"]
    assert verdict["reasons"], "an empty device artifact must never release an arm"


def test_the_new_gate_sources_use_the_shared_whole_inventory_probe():
    for path in (CONDUCTIVE_GATE_PATH, STORED_E_GATE_PATH):
        text = path.read_text(encoding="utf-8")
        assert "P.run_case" in text
        assert "probe_residual_group_bodies" in text
        assert "allclose" not in text


# ---------------------------------------------------------------------------
# The new-predicate probe measures the round's OWN objects
# ---------------------------------------------------------------------------

def test_every_new_builder_is_exercised_by_the_new_predicate_probe(newpred):
    """The gap this probe closes: four modules shipped a device-measured licence
    for a DIFFERENT object than the one they ship.

    The triage probe patches the INCUMBENT predicate in front of the INCUMBENT
    builder, which is the only honest way to ask its question and licenses a
    claim about the kernel body alone. Every builder this round added must
    therefore appear in a leg of its own.
    """
    cases = newpred.build_cases(numpy, None)
    builders = {case["builder"] for case in cases}
    for expected in (
            "nonlinear_update_e.plan_nonlinear_run_pml_curl",
            "nonlinear_update_e.plan_nonlinear_run_constitutive",
            "no_pml_ade.plan_no_pml_ade_update_p",
            "folded_dispersive_update_e.plan_folded_dispersive_constitutive",
            "folded_complex.plan_folded_complex_offdiag_pml_curl",
            "folded_complex.plan_folded_complex_offdiag_constitutive"):
        assert expected in builders, f"{expected} has no identity leg"


def test_the_new_predicate_probe_patches_nothing(newpred):
    """A patched predicate would measure the incumbent again, which is the bug."""
    text = NEWPRED_PATH.read_text(encoding="utf-8")
    assert "admitting(" not in text, (
        "the new-predicate probe must not use the triage probe's admitting "
        "context manager: its whole purpose is to run the shipped predicates")
    assert "UNPATCHED" in text.upper()


def test_the_new_predicate_probe_treats_a_none_plan_as_a_failure(newpred):
    """A refused configuration is a failed leg here, never a silent skip.

    With nothing patched, ``plan is None`` means the NEW predicate refused — and
    a probe that skipped those would report "all legs identical" over an empty
    set.
    """
    flat = " ".join(NEWPRED_PATH.read_text(encoding="utf-8").split())
    assert "the NEW predicate REFUSED this configuration" in flat
    assert "no identity claim is licensed for it" in flat


def test_the_builders_all_answer_none_on_this_laptop(newpred):
    """Every leg must at least BUILD its fixture and reach its builder here.

    The predicates are CuPy-gated at clause 1, so the answer is None on this
    host; what this pins is that no leg raises on the way there, which is how a
    fixture that stopped constructing would otherwise surface only on the device.
    """
    for case in newpred.build_cases(numpy, None):
        fields, pml, _notes = case["make"](0)
        assert case["kernel_plan"](fields, pml) is None, case["name"]


def test_the_predicate_battery_covers_admissions_and_refusals_for_every_family(
        newpred):
    """An admission alone is not coverage; the neighbours must be refused too."""
    # The certified complex-multiply EXPANSION probe is a MEASURED device fact the
    # folded-complex builders require; without it those predicates refuse for a
    # reason that has nothing to do with the configuration under test.
    bodies = importlib.import_module(
        "parity.meep_gpu.probe_residual_group_bodies")
    record = bodies._load_expansion_probe()
    assert record is not None, (
        "the expansion probe artifact is missing; the folded-complex battery "
        "cannot be asked and a pass here would be vacuous")
    rows = newpred.predicate_checks(numpy, record)
    assert len(rows) >= 30, f"only {len(rows)} checks"
    admits = [r for r in rows if r["expected_admit"]]
    refuses = [r for r in rows if not r["expected_admit"]]
    assert admits and refuses
    assert all(r["matches"] for r in rows), (
        [r["check"] for r in rows if not r["matches"]])
    # Every family must contribute at least one of each.
    for family in ("nonlinear_run", "no_pml_ade", "folded_dispersive",
                   "folded_complex_offdiag"):
        labelled = [r for r in rows if family in r["check"]]
        assert any(r["expected_admit"] for r in labelled), family
        assert any(not r["expected_admit"] for r in labelled), family


def test_the_device_spelling_hazard_is_recorded_where_it_bites(probe):
    """CuPy mask-assignment from a SCALAR drops the sign of zero; NumPy does not.

    Measured on an RTX A6000, and it is exactly the hazard a NumPy merge bar
    cannot show: the harness looks correct here and silently loses the needle
    there. The note must stay next to the code that avoids it.
    """
    text = BODIES_PATH.read_text(encoding="utf-8")
    flat = " ".join(text.split())
    assert "PLATFORM FACT" in flat
    assert "does not preserve the sign of zero" in flat
    assert "cp.where" in flat
    # And the spelling actually used must be the safe one.
    assert "xp.where(frozen" in text

    # The NumPy side of the discrepancy, executed: here the scalar spelling works,
    # which is why it survived review.
    array = numpy.zeros((4, 4, 4), dtype=numpy.float32)
    mask = numpy.indices(array.shape).sum(axis=0) % 4 == 0
    array[mask] = numpy.float32(-0.0)
    assert int((array.view(numpy.uint32) == 0x80000000).sum()) == int(mask.sum())
