"""Laptop-safe contracts for ``launch.plan_step``'s selection across fifteen arms.

NINE certified families are now CONSULTED by the central composer: complex/Bloch,
special_kz (real and complex), chi2/chi3 ``update_E``, off-diagonal ``update_E``,
BFAST, and — added in the round this file's later sections pin — the folded
complex cluster (K1/K2/K3a/K3b plus its constitutive arms), folded off-diagonal
``update_E``, complex Dcyl, and the NULL constitutive family, whose product
launches nothing at all. Each was certified on its own device gate against a
hand-composed set of plans; what no prior artifact measures is the composer
itself: that the planner, given a configuration, admits exactly one product per
sub-step, refuses by name where none applies, and fails CLOSED where two apply.

That is what this module pins, and it pins it with no GPU and no Triton. Every
arm here is stubbed by replacing a module-level name in ``launch``, exactly as
``test_triton_symmetry_composition`` and ``test_triton_cylindrical_composition``
do, so the assertions are about the SELECTION RULE rather than about arithmetic
that lives in another file and has its own gate.

The device half of the evidence — that the planner's chosen products then step
byte-identically to the array path — is a separate probe on a CUDA host, and
this module deliberately does not stand in for it.

THIS MODULE DOES NOT DISPATCH. It asks the composer directly and never steps a
driver through ``meep_gpu.fastpath.plan_fast_path``, which is where dispatch is
decided — on by default since 2026-09-27, and pinned at the one test below that
reaches it (``test_dispatch_is_wired_and_on_by_default_with_the_veto_above_it``).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest

from meep_gpu.device_identity import weld_survives_edit
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import launch as launch_module

# Every probe record this module builds is stamped 'keep'; the complex
# families' policy clause fails closed when the run policy can be neither
# read nor declared, so the premise is declared rather than left implicit.
# See ``run_policy_declared_keep`` in conftest.py.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")



PACKAGE_DIR = pathlib.Path(launch_module.__file__).parent

ADMITTED = coverage_module.Coverage(True, ())
REFUSED = coverage_module.Coverage(False, ("outside this routing test",))

#: Every module-level predicate ``plan_step`` may consult, with the arm label it
#: serves. Spelled once so a new arm cannot be added without appearing here.
CURL_PREDICATES = {
    "PML": "pml_curl_coverage",
    "conductive PML": "conductive_pml_curl_coverage",
    "no-PML": "plain_curl_coverage",
    "conductive no-PML": "conductive_plain_curl_coverage",
    "folded PML": "folded_composition_curl_coverage",
    "cylindrical PML": "cylindrical_curl_coverage",
    "complex PML": "complex_pml_curl_coverage",
    "complex conductive no-PML curl":
        "complex_conductive_no_pml_curl_coverage",
    "complex no-PML curl": "complex_no_pml_curl_coverage",
    "real beta PML": "beta_pml_curl_coverage",
    "complex beta PML": "beta_bloch_pml_curl_coverage",
    "BFAST PML": "bfast_pml_curl_coverage",
    "folded complex PML": "folded_complex_composition_curl_coverage",
    "folded real beta PML": "folded_beta_pml_curl_coverage",
    "folded complex beta PML": "folded_beta_bloch_pml_curl_coverage",
    "cylindrical complex PML": "cylindrical_complex_curl_coverage",
    # The residual-group ADMISSIONS: a CERTIFIED curl body on a configuration a
    # shared clause refused it for. No new kernel in either.
    "nonlinear run PML": "nonlinear_run_pml_curl_coverage",
    "folded complex off-diagonal PML": "folded_complex_offdiag_pml_curl_coverage",
}

#: The two ghost-fill arms. ``fill_B``/``fill_D`` went through the arm table in
#: the same round that added the complex-storage fill: before that the block
#: consulted ONE predicate and was the last place here that could have picked a
#: second admitting product by position.
FILL_PREDICATES = {
    "mirror fill": "mirror_ghost_fill_coverage",
    "folded complex fill": "folded_mirror_ghost_fill_complex_coverage",
}

#: The constitutive predicates, by arm label. ``ordinary`` serves both sides and
#: ``dispersive``/``nonlinear``/``off-diagonal``/``folded off-diagonal`` are
#: E-only. ``no-PML null`` is the one whose product launches nothing.
CONSTITUTIVE_PREDICATES = {
    "ordinary": "constitutive_coverage",
    "folded": "folded_constitutive_coverage",
    "cylindrical": "cylindrical_constitutive_coverage",
    "complex": "complex_constitutive_coverage",
    "real beta run": "beta_run_constitutive_coverage",
    "complex beta run": "beta_run_complex_constitutive_coverage",
    "BFAST run": "bfast_run_constitutive_coverage",
    "dispersive": "dispersive_constitutive_coverage",
    "nonlinear": "nonlinear_constitutive_coverage",
    "off-diagonal": "offdiag_constitutive_coverage",
    "folded complex": "folded_complex_constitutive_coverage",
    "folded beta run": "folded_beta_run_constitutive_coverage",
    "cylindrical complex": "cylindrical_complex_constitutive_coverage",
    "folded off-diagonal": "folded_offdiag_composition_coverage",
    "complex folded off-diagonal":
        "complex_folded_offdiag_update_e_coverage",
    "folded off-diagonal dispersive":
        "folded_offdiag_dispersive_constitutive_coverage",
    "complex no-PML off-diagonal":
        "complex_no_pml_offdiag_update_e_coverage",
    "complex no-PML stored E": "complex_stored_e_coverage",
    # The residual-group ADMISSIONS. ``nonlinear run`` and ``folded complex
    # off-diagonal`` are update_H-ONLY here: both family predicates take a
    # ``side`` and both refuse ``side='E'`` BY NAME, so neither appears in the
    # update_E table. ``folded dispersive`` is E-only in the other direction —
    # its predicate takes no side at all.
    "nonlinear run": "nonlinear_run_constitutive_coverage",
    "folded complex off-diagonal": "folded_complex_offdiag_constitutive_coverage",
    "folded dispersive": "folded_dispersive_constitutive_coverage",
    "no-PML stored E": "stored_e_constitutive_coverage",
    "no-PML null": "null_constitutive_coverage",
}

BUILDERS = {
    "pml_curl_coverage": "plan_pml_curl",
    "conductive_pml_curl_coverage": "plan_conductive_pml_curl",
    "plain_curl_coverage": "plan_plain_curl",
    "conductive_plain_curl_coverage": "plan_conductive_plain_curl",
    "folded_composition_curl_coverage": "plan_folded_pml_curl",
    "cylindrical_curl_coverage": "plan_cylindrical_curl",
    "complex_pml_curl_coverage": "plan_complex_pml_curl",
    "complex_conductive_no_pml_curl_coverage":
        "plan_complex_conductive_no_pml_curl",
    "complex_no_pml_curl_coverage": "plan_complex_no_pml_curl",
    "beta_pml_curl_coverage": "plan_beta_pml_curl",
    "beta_bloch_pml_curl_coverage": "plan_beta_bloch_pml_curl",
    "bfast_pml_curl_coverage": "plan_bfast_pml_curl",
    "folded_complex_composition_curl_coverage": "plan_folded_complex_pml_curl",
    "folded_beta_pml_curl_coverage": "plan_folded_beta_pml_curl",
    "folded_beta_bloch_pml_curl_coverage": "plan_folded_beta_bloch_pml_curl",
    "cylindrical_complex_curl_coverage": "plan_cylindrical_complex_curl",
    "mirror_ghost_fill_coverage": "plan_mirror_ghost_fill",
    "folded_mirror_ghost_fill_complex_coverage":
        "plan_folded_mirror_ghost_fill_complex",
    "constitutive_coverage": "plan_constitutive",
    "folded_constitutive_coverage": "plan_folded_constitutive",
    "cylindrical_constitutive_coverage": "plan_cylindrical_constitutive",
    "complex_constitutive_coverage": "plan_complex_constitutive",
    "beta_run_constitutive_coverage": "plan_beta_run_constitutive",
    "beta_run_complex_constitutive_coverage": "plan_beta_run_complex_constitutive",
    "bfast_run_constitutive_coverage": "plan_bfast_run_constitutive",
    "dispersive_constitutive_coverage": "plan_dispersive_constitutive",
    "nonlinear_constitutive_coverage": "plan_nonlinear_constitutive",
    "offdiag_constitutive_coverage": "plan_offdiagonal_constitutive",
    "folded_complex_constitutive_coverage": "plan_folded_complex_constitutive",
    "folded_beta_run_constitutive_coverage": "plan_folded_beta_run_constitutive",
    "cylindrical_complex_constitutive_coverage":
        "plan_cylindrical_complex_constitutive",
    "folded_offdiag_composition_coverage": "plan_folded_offdiagonal_constitutive",
    "complex_folded_offdiag_update_e_coverage":
        "plan_complex_folded_offdiag_update_e",
    "folded_offdiag_dispersive_constitutive_coverage":
        "plan_folded_offdiag_dispersive_constitutive",
    "complex_no_pml_offdiag_update_e_coverage":
        "plan_complex_no_pml_offdiag_update_e",
    "complex_stored_e_coverage": "plan_complex_stored_e",
    "null_constitutive_coverage": "plan_null_constitutive",
    "nonlinear_run_pml_curl_coverage": "plan_nonlinear_run_pml_curl",
    "nonlinear_run_constitutive_coverage": "plan_nonlinear_run_constitutive",
    "folded_dispersive_constitutive_coverage": "plan_folded_dispersive_constitutive",
    "stored_e_constitutive_coverage": "plan_stored_e_constitutive",
    "folded_complex_offdiag_pml_curl_coverage":
        "plan_folded_complex_offdiag_pml_curl",
    "folded_complex_offdiag_constitutive_coverage":
        "plan_folded_complex_offdiag_constitutive",
}

#: Every specialized family module the planner may import, spelled the way
#: ``launch.FAMILY_MODULES`` spells it and checked against it below. This
#: replaced ``RESERVED_FAMILIES`` as the lazy-import seam's anchor when that
#: tuple went empty: a test that iterated an empty tuple would have gone VACUOUS
#: exactly at the change it exists to catch.
FAMILY_MODULES = tuple(
    f"meep_gpu.triton_kernels.{name}" for name in (
        "complex_fields",
        "special_kz",
        "bfast_curl",
        "nonlinear_update_e",
        "offdiag_update_e",
        "folded_complex",
        "folded_offdiag_update_e",
        "cylindrical_complex",
        "no_pml_constitutive",
        "folded_dispersive_update_e",
        "folded_offdiag_dispersive_update_e",
        "complex_offdiag_update_e",
        "complex_ade",
        "no_pml_ade",
        "complex_no_pml_curl",
        "complex_no_pml_conductive",
        "no_pml_conductive",
        "no_pml_stored_e",
        "complex_no_pml_stored_e",
    ))


# ---------------------------------------------------------------------------
# Doubles — the smallest object each gate reads
# ---------------------------------------------------------------------------

def make_fields(**grid_attributes):
    """A fields double whose grid answers every gate's attribute."""
    field_attributes = {
        name: grid_attributes.pop(name)
        for name in ("force_complex_fields", "has_nonlinearity",
                     "has_offdiagonal_epsilon", "chi1inv_offdiagonal_for")
        if name in grid_attributes
    }
    grid = SimpleNamespace(
        has_bloch=grid_attributes.pop("has_bloch", False),
        beta=grid_attributes.pop("beta", 0.0),
        bfast_active=grid_attributes.pop("bfast_active", False),
        cylindrical=grid_attributes.pop("cylindrical", False),
        has_symmetry=grid_attributes.pop("has_symmetry", lambda: False),
        is_mirrored=grid_attributes.pop("is_mirrored", lambda axis: False),
        **grid_attributes,
    )
    return SimpleNamespace(grid=grid, polarizations=(), **field_attributes)


ORDINARY = dict()
COMPLEX_RUN = dict(has_bloch=True)
REAL_BETA_RUN = dict(beta=0.7)
COMPLEX_BETA_RUN = dict(beta=0.7, force_complex_fields=True)
BFAST_RUN = dict(bfast_active=True)
NONLINEAR_RUN = dict(has_nonlinearity=True)
OFFDIAG_RUN = dict(has_offdiagonal_epsilon=True)
CYLINDRICAL_COMPLEX_RUN = dict(cylindrical=True, force_complex_fields=True)


class InactiveLayer:
    """A layer that ANSWERS ``is_active`` with False — the null family's territory.

    ``object()`` is NOT this: it cannot answer at all, and the null family refuses
    an unreadable layer by name while ``absorber_inactive`` still consults the arm.
    The distinction is load-bearing, so it gets a class rather than a lambda.
    """

    is_active = False


def folded_fields(**attributes):
    """A fields double whose grid reports a real fold on axis 1."""
    return make_fields(has_symmetry=lambda: True,
                       is_mirrored=lambda axis: axis == 1, **attributes)


def install_stubs(monkeypatch, admit=(), *, probe_record=object()):
    """Refuse every arm, then admit the labels named. Builders return sentinels.

    The builder sentinel is ``"<predicate name>:<slot or side>"`` so a test can
    tell which BUILDER ran, and ``plan.selected`` says which ARM chose it. Both
    are needed: three different arms build a ``launch.ConstitutivePlan``.
    """
    admit = set(admit)
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: False)
    monkeypatch.setattr(launch_module, "load_expansion_probe",
                        lambda path=None: probe_record)

    for label, predicate in {**CURL_PREDICATES, **FILL_PREDICATES}.items():
        verdict = ADMITTED if label in admit else REFUSED
        monkeypatch.setattr(launch_module, predicate,
                            (lambda verdict: lambda *a, **k: verdict)(verdict),
                            raising=False)
    for label, predicate in CONSTITUTIVE_PREDICATES.items():
        verdict = ADMITTED if label in admit else REFUSED
        monkeypatch.setattr(launch_module, predicate,
                            (lambda verdict: lambda *a, **k: verdict)(verdict),
                            raising=False)
    for predicate, builder in BUILDERS.items():
        monkeypatch.setattr(
            launch_module, builder,
            (lambda predicate: lambda fields, pml, *rest, **k:
                f"{predicate}:{rest[0] if rest and isinstance(rest[0], str) else 'x'}"
             )(predicate),
            raising=False)

    monkeypatch.setattr(launch_module, "fused_pair_coverage",
                        lambda *a, **k: REFUSED)
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage",
                        lambda *a, **k: REFUSED)


# ---------------------------------------------------------------------------
# C1 — SELECTION, per family
# ---------------------------------------------------------------------------
#
# ``plan.selected`` is what makes these tests meaningful rather than decorative.
# "ordinary", "real beta run" and "BFAST run" all build a
# ``launch.ConstitutivePlan``, so a test that asserted only the plan object
# would pass for all three and would not notice a mis-selection at all.

@pytest.mark.parametrize("label,configuration,curl_arm,constitutive_arm", [
    ("complex", COMPLEX_RUN, "complex PML", "complex"),
    ("real beta", REAL_BETA_RUN, "real beta PML", "real beta run"),
    ("complex beta", COMPLEX_BETA_RUN, "complex beta PML", "complex beta run"),
    ("bfast", BFAST_RUN, "BFAST PML", "BFAST run"),
])
def test_the_planner_selects_one_family_for_all_four_substeps(
        monkeypatch, label, configuration, curl_arm, constitutive_arm):
    """A whole-step family: both curls and both constitutive sides are its own."""
    install_stubs(monkeypatch, admit=(curl_arm, constitutive_arm))

    plan = launch_module.plan_step(make_fields(**configuration), object())

    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.selected == {
        "step_B": curl_arm, "step_D": curl_arm,
        "update_H": constitutive_arm, "update_E": constitutive_arm,
    }, label


def test_the_planner_selects_the_nonlinear_family_for_update_E_alone(monkeypatch):
    """chi2/chi3 is an ``update_E``-only product: the rest stays on the array path.

    The certified curls and update_H refuse a chi2/chi3 grid by name (the Pade
    factor they do not carry), so a nonlinear run is PARTIAL coverage — which is
    legal and expected, and the plan says so rather than pretending otherwise.
    """
    install_stubs(monkeypatch, admit=("nonlinear",))

    plan = launch_module.plan_step(make_fields(**NONLINEAR_RUN), object())

    assert plan.replaces == ("update_E",)
    assert plan.selected == {"update_E": "nonlinear"}
    for slot in ("step_B", "step_D", "update_H"):
        assert slot not in plan.plans
        assert plan.reasons[slot]


def test_the_planner_selects_offdiagonal_update_E_beside_the_shipped_curls(
        monkeypatch):
    """The shipped curl and H products admit an off-diagonal grid DELIBERATELY.

    The row product's whole effect is inside ``update_E``; the curls read no
    chi1inv at all. So the composition here is genuinely mixed — certified
    shipped products on three slots, the specialized family on the fourth.
    """
    install_stubs(monkeypatch, admit=("PML", "off-diagonal"))
    # The shipped constitutive predicate admits the H side and refuses the E
    # side by name on the installed row — that asymmetry is what makes the
    # composition mixed rather than ambiguous.
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: ADMITTED if side == "H" else REFUSED)

    plan = launch_module.plan_step(make_fields(**OFFDIAG_RUN), object())

    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.selected == {
        "step_B": "PML", "step_D": "PML",
        "update_H": "ordinary", "update_E": "off-diagonal",
    }


@pytest.mark.parametrize("label,folded,configuration,curl_arm,constitutive_arm", [
    ("folded complex", True, dict(force_complex_fields=True),
     "folded complex PML", "folded complex"),
    ("folded complex beta", True, dict(force_complex_fields=True, beta=0.7),
     "folded complex beta PML", "folded complex"),
    ("cylindrical complex", False, CYLINDRICAL_COMPLEX_RUN,
     "cylindrical complex PML", "cylindrical complex"),
])
def test_the_four_new_families_each_select_one_product_per_substep(
        monkeypatch, label, folded, configuration, curl_arm, constitutive_arm):
    """The wiring's own C1: one product per slot, named by ARM not by class.

    ``folded complex`` and ``cylindrical complex`` both build a
    ``ComplexConstitutivePlan`` and ``folded beta run`` builds a plain
    ``launch.ConstitutivePlan`` exactly as ``ordinary``, ``real beta run`` and
    ``BFAST run`` do — so a test that asserted the plan CLASS would pass for a
    mis-selection between any of them. ``plan.selected`` is the only assertion
    that distinguishes them.

    The ``folded complex beta`` row is the grating shape: K3b takes the curls
    while the CONSTITUTIVE slots go to ``folded complex``, because that arm
    deliberately carries no beta clause (the constitutive kernel is
    storage-dependent, not beta-dependent) and ``folded beta run`` refuses
    complex storage.
    """
    install_stubs(monkeypatch, admit=(curl_arm, constitutive_arm))
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: folded)

    plan = launch_module.plan_step(make_fields(**configuration), object())

    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.selected == {
        "step_B": curl_arm, "step_D": curl_arm,
        "update_H": constitutive_arm, "update_E": constitutive_arm,
    }, label


def test_a_folded_real_beta_run_selects_K3a_and_the_real_constitutive_arm(
        monkeypatch):
    """The one new family whose constitutive arm is REAL, so it needs its own row."""
    install_stubs(monkeypatch, admit=("folded real beta PML", "folded beta run"))
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: True)

    plan = launch_module.plan_step(make_fields(beta=0.7), object())

    assert plan.selected == {
        "step_B": "folded real beta PML", "step_D": "folded real beta PML",
        "update_H": "folded beta run", "update_E": "folded beta run",
    }


def test_a_folded_offdiagonal_grid_mixes_three_shipped_arms_with_the_new_one(
        monkeypatch):
    """The fold's own curls and H product, and the row product on ``update_E``.

    The same mixed shape the unfolded off-diagonal row has, one level down: the
    curls read no chi1inv at all, so the folded curl and the folded constitutive
    product keep three slots and the specialized row product takes the fourth.
    """
    install_stubs(monkeypatch, admit=("folded PML", "folded", "folded off-diagonal"))
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: True)
    monkeypatch.setattr(launch_module, "folded_constitutive_coverage",
                        lambda fields, pml, side: ADMITTED if side == "H" else REFUSED,
                        raising=False)

    fields = make_fields(has_offdiagonal_epsilon=True)
    plan = launch_module.plan_step(fields, object())

    assert plan.selected["update_E"] == "folded off-diagonal"
    assert plan.selected["update_H"] == "folded"
    assert plan.selected["step_B"] == plan.selected["step_D"] == "folded PML"


def test_an_inactive_layer_selects_the_null_product_on_both_constitutive_sides(
        monkeypatch):
    """The family that ships NO KERNEL, selected exactly like one that does.

    Under an inactive absorber ``stepping.update_H`` returns at :916-917 and
    ``update_E`` at :954-955 before reading any array, so the covered product is
    a ``NullConstitutivePlan``. It goes through the same arm table, the same
    gate rule and the same ``_select_slot``; nothing about it is special-cased.
    """
    install_stubs(monkeypatch, admit=("no-PML", "no-PML null"))

    plan = launch_module.plan_step(make_fields(**ORDINARY), InactiveLayer())

    assert plan.selected == {
        "step_B": "no-PML", "step_D": "no-PML",
        "update_H": "no-PML null", "update_E": "no-PML null",
    }


def test_the_previously_certified_selections_still_hold(monkeypatch):
    """The regression surface: ordinary grids must select exactly what they did.

    Adding nine arms is only safe if every one of them is gated OFF where it
    does not apply. On an ordinary grid the shipped PML curls and the ordinary
    constitutive product must still win, unopposed and unambiguous.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))

    plan = launch_module.plan_step(make_fields(**ORDINARY), object())

    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.selected == {
        "step_B": "PML", "step_D": "PML",
        "update_H": "ordinary", "update_E": "ordinary",
    }
    assert plan.reasons == {}


#: The five families the earlier round wired, with the configuration each owns
#: and the arms it must still win. THE REGRESSION SURFACE OF THIS ROUND: adding
#: six curl arms, two fill arms and eight constitutive arms is only safe if none
#: of them changes any of these.
PREVIOUSLY_CERTIFIED = (
    ("ordinary", ORDINARY, "PML", "ordinary", "ordinary"),
    ("complex", COMPLEX_RUN, "complex PML", "complex", "complex"),
    ("real beta", REAL_BETA_RUN, "real beta PML", "real beta run", "real beta run"),
    ("complex beta", COMPLEX_BETA_RUN, "complex beta PML",
     "complex beta run", "complex beta run"),
    ("bfast", BFAST_RUN, "BFAST PML", "BFAST run", "BFAST run"),
    ("nonlinear", NONLINEAR_RUN, None, None, "nonlinear"),
    ("off-diagonal", OFFDIAG_RUN, "PML", "ordinary", "off-diagonal"),
)


@pytest.mark.parametrize("label,configuration,curl,update_h,update_e",
                         PREVIOUSLY_CERTIFIED,
                         ids=[row[0] for row in PREVIOUSLY_CERTIFIED])
def test_the_five_already_wired_families_select_exactly_what_they_did(
        monkeypatch, label, configuration, curl, update_h, update_e):
    """Every previously certified selection, re-asserted against fifteen arms.

    Each row admits ONLY the arms that family owned before this round. If any new
    arm were consulted where it does not apply, it would either win the slot
    (wrong product) or co-admit and leave it unselected (lost coverage) — both
    show up here as a ``selected`` dict that is not the one the family's own gate
    certified.
    """
    admit = {name for name in (curl, update_h, update_e) if name}
    configuration = dict(configuration)
    if update_e == "off-diagonal":
        configuration["chi1inv_offdiagonal_for"] = lambda row: {}
    install_stubs(monkeypatch, admit=admit)
    if update_h != update_e and update_h is not None:
        # ``ordinary`` serves both sides; the off-diagonal row is E-only, so the
        # shipped predicate must admit H and refuse E for the split to be real.
        monkeypatch.setattr(
            launch_module, "constitutive_coverage",
            lambda fields, pml, side: ADMITTED if side == "H" else REFUSED)

    plan = launch_module.plan_step(make_fields(**configuration), object())

    expected = {}
    if curl:
        expected["step_B"] = expected["step_D"] = curl
    if update_h:
        expected["update_H"] = update_h
    if update_e:
        expected["update_E"] = update_e
    assert plan.selected == expected, label


# ---------------------------------------------------------------------------
# C2 — AMBIGUITY, per slot and per new pair
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot,configuration,first,second,quantifier", [
    ("step_B", COMPLEX_RUN, "PML", "complex PML", None),
    ("step_D", COMPLEX_BETA_RUN, "real beta PML", "complex beta PML", None),
    ("update_H", BFAST_RUN, "ordinary", "BFAST run", None),
    ("update_E", NONLINEAR_RUN, "nonlinear", "off-diagonal", "both"),
    ("update_E", OFFDIAG_RUN, "ordinary", "off-diagonal", "both"),
])
def test_two_admitting_predicates_leave_the_slot_on_the_array_path(
        monkeypatch, slot, configuration, first, second, quantifier):
    """An overlap is a PREDICATE DEFECT, never an ordering choice.

    An over-covering dispatch is a silent wrong answer; a slot on the array path
    is a slower right answer. Every ambiguity has to resolve to the second, and
    the refusal has to NAME both claimants so the defect is diagnosable.
    """
    configuration = dict(configuration)
    if second == "off-diagonal":
        configuration["chi1inv_offdiagonal_for"] = lambda row: {"y": object()}
    install_stubs(monkeypatch, admit=(first, second))

    plan = launch_module.plan_step(make_fields(**configuration), object())

    assert slot not in plan.plans
    assert slot not in plan.selected
    reason, = plan.reasons[slot]
    assert "ambiguous" in reason, reason
    assert first in reason and second in reason, reason
    if quantifier is not None:
        assert quantifier in reason and "update_E" in reason, reason


@pytest.mark.parametrize("slot,folded,configuration,first,second", [
    ("step_B", True, dict(force_complex_fields=True),
     "folded PML", "folded complex PML"),
    ("step_D", True, dict(force_complex_fields=True, beta=0.7),
     "folded real beta PML", "folded complex beta PML"),
    ("update_E", True, dict(has_offdiagonal_epsilon=True),
     "folded", "folded off-diagonal"),
    ("update_H", False, CYLINDRICAL_COMPLEX_RUN,
     "complex", "cylindrical complex"),
    ("update_E", True, dict(force_complex_fields=True),
     "folded complex", "cylindrical complex"),
])
def test_each_new_arm_pair_also_fails_closed_when_both_admit(
        monkeypatch, slot, folded, configuration, first, second):
    """Same rule, new pairs: an overlap is a predicate defect, never an order.

    The pairs here are the ones the four new families make newly POSSIBLE. None
    of them occurs on any real object in the disjointness sweep — that is the
    measurement, and this is what happens if it ever stops being true.
    """
    configuration = dict(configuration)
    if second == "folded off-diagonal":
        configuration["chi1inv_offdiagonal_for"] = lambda row: {"y": object()}
    install_stubs(monkeypatch, admit=(first, second))
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: folded)
    monkeypatch.setattr(launch_module, "cylindrical_grid_active",
                        lambda fields: True, raising=False)

    plan = launch_module.plan_step(make_fields(**configuration), object())

    assert slot not in plan.plans
    assert slot not in plan.selected
    reason, = plan.reasons[slot]
    assert "ambiguous" in reason, reason
    assert first in reason and second in reason, reason


@pytest.mark.parametrize("slot,side", [("update_H", "H"), ("update_E", "E")])
def test_a_null_plan_never_displaces_a_kernel_and_a_kernel_never_displaces_a_null(
        monkeypatch, slot, side):
    """THE RULE SECTION C OF THE DESIGN EXISTS FOR, pinned so it cannot be relaxed.

    ``no-PML null`` is the only arm in the table whose product LAUNCHES NOTHING,
    and it is the one arm a reader is most tempted to give a preference rule.
    Both directions are wrong, and neither is worse:

    * null preferred over a kernel = a real sub-step silently replaced by a
      no-op. The whole constitutive accumulation vanishes with no exception and a
      plausible smooth field;
    * kernel preferred over the null = launching the ``dsigw`` accumulation on a
      step the array path skips. That kernel WRITES, and on a no-PML run
      ``f_w_*`` is not even allocated.

    So the overlap fails closed exactly as every other overlap does, and failing
    closed costs nothing at runtime here: the array path's ``update_H`` returns
    immediately, so the "slower right answer" is not even slower.

    A mutant that added ``if 'no-PML null' in admitted: pick the other`` — or the
    reverse — fails this test in one direction or the other.
    """
    del side
    install_stubs(monkeypatch, admit=("ordinary", "no-PML null"))

    plan = launch_module.plan_step(make_fields(**ORDINARY), InactiveLayer())

    assert slot not in plan.plans, "a null and a kernel resolved to a product"
    assert slot not in plan.selected
    reason, = plan.reasons[slot]
    assert "ambiguous" in reason, reason
    assert "ordinary" in reason and "no-PML null" in reason, reason


def test_the_two_fill_arms_fail_closed_together_like_every_other_slot(monkeypatch):
    """``fill_B``/``fill_D`` used to consult ONE predicate and pick by position.

    The two arms split on STORAGE and were measured never to co-admit, so this
    configuration is synthetic — which is the point: the conversion to the arm
    table is what makes the synthetic case fail closed instead of silently
    selecting whichever the code reached first.
    """
    install_stubs(monkeypatch, admit=("mirror fill", "folded complex fill"))
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: True)

    plan = launch_module.plan_step(
        make_fields(force_complex_fields=True), object())

    for slot in ("fill_B", "fill_D"):
        assert slot not in plan.plans
        reason, = plan.reasons[slot]
        assert "ambiguous" in reason and "mirror-fill" in reason, reason
        assert "mirror fill" in reason and "folded complex fill" in reason, reason


def test_the_complex_fill_plans_adapter_contract_is_written_down_not_inferred():
    """The two fill arms are NOT interchangeable at ``run()``, and it must be stated.

    ``symmetry.MirrorGhostFillPlan`` fuses the near and far passes and is entitled
    to: the parity factors are +-1 and float32 multiplication by them is exact.
    ``FoldedMirrorGhostFillComplexPlan`` may NOT be run that way — the driver puts
    ``zero_metal_*`` between the two passes and complex float multiplication is
    not associative (measured: 5 words of By, 5 of Dx differ on a two-axis fold).

    A driver adapter that inferred the contract from the plan CLASS would get this
    exactly wrong, and ``selected[slot]`` is the seam it should branch on instead.
    Dispatch is disabled, so today this is a contract rather than a live hazard —
    which is precisely why it has to be written where the adapter author will read
    it rather than left to be rediscovered.
    """
    doc = launch_module.plan_folded_mirror_ghost_fill_complex.__doc__ or ""
    assert "zero_metal" in doc, doc
    assert "not associative" in doc.lower(), doc
    assert "selected[slot]" in doc.replace("``", ""), doc


def test_a_folded_real_run_still_selects_the_mirror_fill_it_always_did(monkeypatch):
    """The fill conversion's regression control: one arm, same label, same plan."""
    install_stubs(monkeypatch, admit=("mirror fill",))
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: True)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object())

    assert plan.selected["fill_B"] == "mirror fill"
    assert plan.selected["fill_D"] == "mirror fill"
    assert "fill_B" not in plan.reasons and "fill_D" not in plan.reasons


# ---------------------------------------------------------------------------
# C3 — the ONE reachable double-admit, and the gate that must not hide it
# ---------------------------------------------------------------------------

def test_a_row_planted_past_the_installer_is_consulted_not_suppressed(monkeypatch):
    """The cheap gate spelling would turn this configuration into a wrong answer.

    Where the ``has_offdiagonal_epsilon`` FLAG and the live ROW SLOTS disagree,
    the SHIPPED E predicate admits (it reads the flag) and the specialized one
    admits (it counts slots). Gating the specialized arm on the flag alone would
    suppress it exactly there, the ordinary product would win unopposed, and the
    row would be silently dropped.

    So the gate is the DISJUNCTION, the ambiguity survives, and the slot goes to
    the array path. A mutant gate that reads only the flag fails here.

    WHO CAN PRODUCE THE DISAGREEMENT is pinned separately, against the engine's
    own class, by
    ``test_the_engine_cannot_produce_a_flag_false_live_row`` — not the engine,
    but every harness that builds a ``Fields`` double, which is what this module
    and the parity probes do. That is why the disjunction is kept AND why the
    slot-level veto exists beside it: this test covers the case where both arms
    survive to be counted, and the veto covers the case where the other clauses
    have already removed the specialized one.
    """
    fields = make_fields(has_offdiagonal_epsilon=False,
                         chi1inv_offdiagonal_for=lambda row: {"y": object()})

    assert bool(getattr(fields, "has_offdiagonal_epsilon", False)) is False
    assert launch_module.offdiag_rows_possible(fields) is True

    install_stubs(monkeypatch, admit=("ordinary", "off-diagonal"))
    plan = launch_module.plan_step(fields, object())

    assert "update_E" not in plan.plans
    reason, = plan.reasons["update_E"]
    assert "ambiguous" in reason and "off-diagonal" in reason, reason


def test_an_offdiagonal_flag_with_no_live_row_is_refused_by_every_arm(monkeypatch):
    """The mirror case: the flag set, every slot dead. Nothing may claim it.

    The ordinary product refuses on the flag and the specialized one refuses on
    the slot count, so ``update_E`` is unselected with BOTH refusals reported —
    fail-closed by construction, and no ambiguity to resolve.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: (
                            ADMITTED if side == "H"
                            else coverage_module.Coverage(
                                False, ("an off-diagonal chi1inv row is installed",))))
    monkeypatch.setattr(
        launch_module, "offdiag_constitutive_coverage",
        lambda fields, pml: coverage_module.Coverage(
            False, ("no off-diagonal chi1inv row survived installation",)),
        raising=False)

    plan = launch_module.plan_step(make_fields(**OFFDIAG_RUN), object())

    assert "update_E" not in plan.plans
    assert any("ordinary: an off-diagonal chi1inv row is installed" == reason
               for reason in plan.reasons["update_E"])
    assert any(reason.startswith("off-diagonal: no off-diagonal chi1inv row")
               for reason in plan.reasons["update_E"])


# ---------------------------------------------------------------------------
# C4 — GATE NECESSITY: the machine-checkable form of the gate rule
# ---------------------------------------------------------------------------
#
# A gate is only allowed to remove REFUSALS. If a gate could be False where the
# family predicate would have ADMITTED, then gating would decide coverage and
# could silently hand a slot to a competing arm. These tests call the REAL family
# predicates — no stubs — on a double where the gate is False, and check the
# implication holds rather than assuming it.

@pytest.mark.parametrize("gate_name,module_name,predicate_name,arguments,clause", [
    ("complex_storage_active", "complex_fields", "complex_pml_curl_coverage",
     ("step_B",), "storage is real float32"),
    ("beta_active", "special_kz", "beta_pml_curl_coverage",
     ("step_B",), "grid.beta is zero"),
    ("bfast_grid_active", "bfast_curl", "bfast_pml_curl_coverage",
     ("step_B",), "bfast_active is False"),
    ("nonlinear_active", "nonlinear_update_e", "nonlinear_constitutive_coverage",
     (), "no chi2/chi3 is installed"),
    ("offdiag_rows_possible", "offdiag_update_e", "offdiag_constitutive_coverage",
     (), "no off-diagonal chi1inv row survived"),
    # The four families this round wired. Three reuse an existing gate, so the
    # implication that needs checking is the one against THEIR clause, not
    # against the clause the gate was written for.
    ("complex_storage_active", "folded_complex",
     "folded_complex_composition_curl_coverage", ("step_B",),
     "storage is real float32"),
    ("beta_active", "folded_complex", "folded_beta_pml_curl_coverage",
     ("step_B",), "grid.beta is zero"),
    ("complex_storage_active", "cylindrical_complex",
     "cylindrical_complex_curl_coverage", ("step_B",),
     "force_complex_fields"),
    ("offdiag_rows_possible", "folded_offdiag_update_e",
     "folded_offdiag_composition_coverage", (), "row"),
    # The residual-group ADMISSIONS. Three reuse a gate written for another
    # family's clause, so what needs checking is the implication against THEIR
    # inverted clause; ``dispersion_active`` is the one new gate.
    ("nonlinear_active", "nonlinear_update_e", "nonlinear_run_pml_curl_coverage",
     ("step_B",), "no chi2/chi3 is installed"),
    ("nonlinear_active", "nonlinear_update_e",
     "nonlinear_run_constitutive_coverage", ("H",), "no chi2/chi3 is installed"),
    ("dispersion_active", "folded_dispersive_update_e",
     "folded_dispersive_constitutive_coverage", (),
     "no susceptibility is registered"),
    ("complex_storage_active", "folded_complex",
     "folded_complex_offdiag_pml_curl_coverage", ("step_B",),
     "storage is real float32"),
    ("offdiag_rows_possible", "folded_complex",
     "folded_complex_offdiag_constitutive_coverage", ("H",),
     "no off-diagonal chi1inv row is installed"),
])
def test_a_closed_gate_implies_the_family_predicate_would_have_refused(
        gate_name, module_name, predicate_name, arguments, clause):
    fields = make_fields(**ORDINARY)
    gate = getattr(launch_module, gate_name)
    assert gate(fields) is False, gate_name

    module = importlib.import_module(f"meep_gpu.triton_kernels.{module_name}")
    verdict = getattr(module, predicate_name)(fields, None, *arguments)
    assert verdict.covered is False
    assert any(clause in reason for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("side", ["H", "E"])
def test_a_closed_absorber_gate_implies_the_null_family_would_have_refused(side):
    """The one NEW gate, checked the same way: it reads what the family reads.

    ``absorber_inactive`` is False exactly when the layer reports
    ``is_active`` True, and that is the attribute
    ``no_pml_constitutive._inactive_layer_reasons`` refuses on — the same
    ``pml.is_active`` ``stepping._pml_is_active`` branches on at stepping.py:944
    and :982. So gating the arm off can only ever remove a refusal.

    The UNREADABLE layer is the other half of the rule and is asserted here too:
    the gate consults the arm and the arm refuses BY NAME, which is this file's
    convention everywhere and is what keeps an unanswerable object from being
    treated as an inert one.
    """
    from meep_gpu.triton_kernels import no_pml_constitutive

    fields = make_fields(**ORDINARY)
    active = SimpleNamespace(is_active=True)
    assert launch_module.absorber_inactive(active) is False
    verdict = no_pml_constitutive.null_constitutive_coverage(fields, active, side)
    assert verdict.covered is False
    assert any("active PML layer is installed" in reason
               for reason in verdict.reasons), verdict.reasons

    assert launch_module.absorber_inactive(object()) is True
    unreadable = no_pml_constitutive.null_constitutive_coverage(
        fields, object(), side)
    assert unreadable.covered is False
    assert any("does not report is_active" in reason
               for reason in unreadable.reasons), unreadable.reasons

    assert launch_module.absorber_inactive(None) is True
    assert launch_module.absorber_inactive(InactiveLayer()) is True


def test_every_gate_reads_the_attribute_its_family_inverts():
    """A gate computed from a DIFFERENT attribute is the failure mode above.

    Flipping only the attribute the family's inverted clause reads must flip the
    gate; nothing else may.
    """
    fields = make_fields(**ORDINARY)
    assert launch_module.complex_storage_active(fields) is False
    assert launch_module.complex_storage_active(make_fields(has_bloch=True)) is True
    assert launch_module.complex_storage_active(
        make_fields(force_complex_fields=True)) is True
    assert launch_module.beta_active(make_fields(beta=0.7)) is True
    assert launch_module.bfast_grid_active(make_fields(bfast_active=True)) is True
    assert launch_module.nonlinear_active(make_fields(has_nonlinearity=True)) is True
    assert launch_module.offdiag_rows_possible(
        make_fields(has_offdiagonal_epsilon=True)) is True
    # ``dispersion_active`` reads the polarization LIST, which is what
    # ``folded_dispersive_constitutive_coverage`` inverts. ``make_fields`` builds
    # an empty one, so the flip is visible in both directions from one attribute.
    assert launch_module.dispersion_active(make_fields(**ORDINARY)) is False
    with_pole = make_fields(**ORDINARY)
    with_pole.polarizations = (object(),)
    assert launch_module.dispersion_active(with_pole) is True


def test_an_unreadable_beta_consults_the_arm_rather_than_skipping_it():
    """Refusing to read is not evidence the family does not apply."""
    class Hostile:
        def __float__(self):
            raise ValueError("beta is not a number")

    assert launch_module.beta_active(make_fields(beta=Hostile())) is True


# ---------------------------------------------------------------------------
# C5 — REASON HYGIENE: the frozen refusal shapes
# ---------------------------------------------------------------------------

def test_the_update_E_refusal_still_reports_the_two_shipped_products_first(
        monkeypatch):
    """The frozen prefix, duplicated here so a regression is diagnosed twice.

    Sixteen arms now feed this slot; on a plain object with no grid, thirteen of
    them must be gated off and contribute NOTHING — not an empty string, not a
    prefix, nothing.

    THE FOURTEENTH AND FIFTEENTH ARE CONSULTED AND ARE SUPPOSED TO BE. ``pml=object()`` cannot
    answer ``is_active``, and an unreadable layer CONSULTS the null arm
    (``absorber_inactive``) rather than being assumed active — so the null arm
    refuses it by name and appends one entry. That is the whole change to this
    frozen record. The stored-E arm is immediately before the null arm, and both
    refuse the gridless double by name; the assertion pins the ordered suffix.
    """
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: False)
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda *a: coverage_module.Coverage(False, ("ordinary reason",)))
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda *a: coverage_module.Coverage(False, ("dispersive reason",)))
    monkeypatch.setattr(launch_module, "pml_curl_coverage", lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage",
                        lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "plain_curl_coverage", lambda *a: REFUSED)

    plan = launch_module.plan_step(SimpleNamespace(polarizations=()), object())

    assert plan.reasons["update_E"] == (
        "ordinary: ordinary reason",
        "dispersive: dispersive reason",
        "no-PML stored E: fields carries no grid",
        "no-PML null: fields carries no grid",
    )
    # The H side's ``ordinary`` arm stays UNPREFIXED — that is what it reported
    # before this round and retrofitting a prefix is a behaviour change with no
    # gate behind it — so the null arm's entry is the only labelled one here.
    assert plan.reasons["update_H"] == (
        "ordinary reason",
        "no-PML null: fields carries no grid",
    )


def test_no_new_arm_appears_in_the_folded_or_cylindrical_refusals(monkeypatch):
    """The two certified composition doubles, re-read against fifteen arms.

    WHAT CHANGED AND WHY, stated rather than papered over. Both doubles pass
    ``pml=object()``, which cannot answer ``is_active``, so the NULL arm is
    consulted on both (an unreadable layer consults its arm) and refuses by name.
    That is one new labelled entry per constitutive slot and it is correct.

    Everything else must be unchanged: the other eight new arms are gated OFF on
    these doubles — no complex storage, no beta, no BFAST, no chi2/chi3, no
    off-diagonal accessor — and none of their labels or refusal text may appear.
    The banned list is matched against the reason TEXT, not just the arm label,
    which is why ``has_nonlinearity`` in a reason string would trip it: a refusal
    that merely mentions a family a reader would then go looking for is the
    failure this test is about.
    """
    monkeypatch.setattr(launch_module, "pml_curl_coverage", lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage",
                        lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "plain_curl_coverage", lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "constitutive_coverage", lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda *a: REFUSED)
    monkeypatch.setattr(launch_module, "folded_composition_curl_coverage",
                        lambda *a: REFUSED, raising=False)
    monkeypatch.setattr(launch_module, "folded_constitutive_coverage",
                        lambda *a: REFUSED, raising=False)
    monkeypatch.setattr(launch_module, "mirror_ghost_fill_coverage",
                        lambda *a: REFUSED, raising=False)
    monkeypatch.setattr(launch_module, "cylindrical_curl_coverage",
                        lambda *a: REFUSED, raising=False)
    monkeypatch.setattr(launch_module, "cylindrical_constitutive_coverage",
                        lambda *a: REFUSED, raising=False)

    folded = SimpleNamespace(
        grid=SimpleNamespace(has_symmetry=lambda: True,
                             is_mirrored=lambda axis: axis == 1),
        polarizations=())
    cylindrical = SimpleNamespace(
        grid=SimpleNamespace(cylindrical=True, has_symmetry=lambda: False),
        polarizations=())

    new_labels = ("complex", "beta", "BFAST", "nonlinear", "off-diagonal",
                  "folded complex", "folded beta run", "cylindrical complex",
                  "folded off-diagonal")
    for fields in (folded, cylindrical):
        plan = launch_module.plan_step(fields, object())
        for slot, reasons in plan.reasons.items():
            null_entries = [r for r in reasons if r.startswith("no-PML null: ")]
            if slot in ("update_H", "update_E"):
                # Consulted, and refusing BY NAME on every clause it could not
                # answer — the E side reports four, one per unreadable flag. Its
                # own reason text is exempt from the banned list precisely
                # because it IS the one declared change; that the exemption
                # reaches no other slot is what the else-branch pins.
                assert null_entries, (slot, reasons)
            else:
                assert null_entries == [], (slot, reasons)
            for reason in reasons:
                if reason.startswith("no-PML null: "):
                    continue
                for label in new_labels:
                    assert label not in reason, (slot, reason)


# ---------------------------------------------------------------------------
# C6 — a raising predicate is a REFUSAL; a raising builder refuses its slot
# ---------------------------------------------------------------------------

def test_a_raising_predicate_refuses_instead_of_crashing_the_composition(
        monkeypatch):
    """``plan_step`` never raises: the array path is always a correct answer.

    A predicate that raises into a caller which would otherwise have stepped
    correctly is not. And an exception must never count as an ADMISSION, or one
    broken clause would hand a slot to a product that never agreed to take it.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    # The shipped H product refuses a BFAST grid by name, so the BFAST arm is
    # the only candidate for update_H — and it is the one that breaks.
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: REFUSED if side == "H" else ADMITTED)

    def explode(*args, **kwargs):
        raise RuntimeError("clause 11 read a missing attribute")

    monkeypatch.setattr(launch_module, "bfast_run_constitutive_coverage",
                        explode, raising=False)

    plan = launch_module.plan_step(make_fields(**BFAST_RUN), object())

    assert "update_H" not in plan.plans
    assert any("BFAST run predicate raised" in reason
               and "clause 11 read a missing attribute" in reason
               for reason in plan.reasons["update_H"]), plan.reasons["update_H"]
    # The rest of the composition is unaffected: one broken arm is one refusal.
    assert plan.selected["update_E"] == "ordinary"


def test_a_raising_builder_leaves_its_slot_unselected_with_a_named_reason(
        monkeypatch):
    """The off-diagonal builder RAISES on a row that aliases one of its outputs.

    The predicate is supposed to refuse first, and does. Wrapping the builder is
    what makes the None-means-refused contract hold even if that ordering is
    ever broken — the slot falls to the array path instead of the exception
    reaching a caller that could have stepped.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary", "off-diagonal"))
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: ADMITTED if side == "H" else REFUSED)

    def explode(*args, **kwargs):
        raise ValueError("a row volume aliases Ex")

    monkeypatch.setattr(launch_module, "plan_offdiagonal_constitutive", explode,
                        raising=False)

    plan = launch_module.plan_step(make_fields(**OFFDIAG_RUN), object())

    assert "update_E" not in plan.plans
    reason, = plan.reasons["update_E"]
    assert "off-diagonal constitutive coverage admitted update_E" in reason
    assert "a row volume aliases Ex" in reason


def test_a_builder_returning_none_leaves_its_slot_unselected(monkeypatch):
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "plan_pml_curl",
                        lambda *a, **k: None, raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object())

    for slot in ("step_B", "step_D"):
        assert slot not in plan.plans
        reason, = plan.reasons[slot]
        assert reason == (f"PML curl coverage admitted {slot} but its "
                          "builder refused")


# ---------------------------------------------------------------------------
# C7 — the fusion guard: an unmeasured composition is refused by name
# ---------------------------------------------------------------------------

def test_opt_in_fusion_cannot_eat_the_seam_a_specialized_family_owns(monkeypatch):
    """The E-SIDE seam only, since the 2026-09-13 split — and BOTH halves pinned.

    THE GUARD USED TO BE BLANKET and is now keyed on the constitutive side.
    ``has_offdiagonal_epsilon`` sat in the ``specialized_family_owns_the_grid``
    disjunction, so an off-diagonal grid refused EVERY opt-in pair one branch
    early. It now sits in ``off_diagonal_owns_update_e``, which ``plan_step``
    applies only where ``spec["constitutive"] == "E"``.

    THE NARROWING IS READ OFF THE KERNELS, not argued: ``FUSED_PAIRS["B"]`` is
    constitutive "H"; ``plan_fused_pair`` passes the inverse-epsilon volumes only
    when the constitutive side is "E"; and ``kernels.fused_curl_constitutive_B``
    carries no inverse-epsilon pointer in its signature at all — so an off-diagonal
    chi1inv cannot change one value the B seam computes.

    BOTH HALVES ARE PINNED BECAUSE EITHER COULD REGRESS: the D pair and the
    dispersive D/E pair must still be refused BY NAME (their reason text is
    unchanged on purpose — it is the string the frozen gate artifacts under
    ``parity/meep_gpu/results`` recorded), and the magnetic pair must still be
    COMPOSED. Without this test the split could be reverted, or widened to the E
    seam, and nothing else would fail.

    ``plan_fused_pair`` IS STUBBED, and that is load-bearing rather than tidiness:
    unstubbed, B falls through the narrowed guard to the REAL builder and raises
    ``ModuleNotFoundError: No module named 'triton'`` on a host without Triton, so
    the composition rule this test exists for is never reached and the failure
    reads like a refusal that is not one.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary", "off-diagonal"))
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: ADMITTED if side == "H" else REFUSED)
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage",
                        lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *a, **k: object(), raising=False)

    plan = launch_module.plan_step(make_fields(**OFFDIAG_RUN), object(),
                                   fuse=True, sources=())

    assert plan.selected["update_E"] == "off-diagonal"
    # THE SEAM THE FAMILY OWNS: still refused by name, on both D-side pairs.
    for key in ("fused_pair_D", "fused_pair_dispersive_D"):
        assert any("specialized" in reason for reason in plan.reasons[key]), key
    # THE MAGNETIC SEAM: composed now, and carrying NO refusal at all.
    assert plan.selected["step_B"] == "fused pair B"
    assert plan.selected["update_H"] == "fused pair B"
    assert isinstance(plan.plans["update_H"], launch_module.NoopPlan)
    assert not (plan.reasons.get("fused_pair_B") or ())


@pytest.mark.parametrize("configuration", [
    COMPLEX_RUN, REAL_BETA_RUN, COMPLEX_BETA_RUN, BFAST_RUN, NONLINEAR_RUN,
])
def test_every_specialized_family_disables_opt_in_pair_fusion(monkeypatch,
                                                              configuration):
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *a, **k: ADMITTED)

    plan = launch_module.plan_step(make_fields(**configuration), object(),
                                   fuse=True, sources=())

    assert not isinstance(plan.plans.get("step_B"), launch_module.FusedPairPlan)
    for key in ("fused_pair_B", "fused_pair_D", "fused_pair_dispersive_D"):
        assert any("specialized" in reason for reason in plan.reasons[key]), key


def test_an_ordinary_grid_still_fuses_exactly_as_it_did(monkeypatch):
    """The regression pin behind the guard's deliberately narrow off-diagonal term.

    ``offdiag_rows_possible`` is true for every real ``Fields`` (it holds the
    accessor), so gating the guard on it would silently disable fusion for every
    engine-route run and change the control the fused gates were measured
    against. The guard therefore reads the installed FLAG, and an ordinary grid
    must keep fusing.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *a, **k: "fused", raising=False)

    fields = make_fields(chi1inv_offdiagonal_for=lambda row: {})
    assert launch_module.offdiag_rows_possible(fields) is True

    plan = launch_module.plan_step(fields, object(), fuse=True, sources=())

    assert plan.plans["step_B"] == "fused"
    assert isinstance(plan.plans["update_H"], launch_module.NoopPlan)
    assert "fused_pair_B" not in plan.reasons


# ---------------------------------------------------------------------------
# C8 — STEP_ORDER is frozen; BFAST adds no slot
# ---------------------------------------------------------------------------

def test_step_order_is_unchanged_by_this_rounds_five_families():
    assert launch_module.STEP_ORDER == (
        "step_B", "fill_B", "update_H",
        "step_D", "fill_D", "update_E", "update_P",
    )


def test_a_bfast_run_produces_no_slot_outside_the_drivers_step_order(monkeypatch):
    """BFAST's second additive term is INSIDE the curl, not a pass of its own.

    One launch performs the curl, adds the second term and writes the
    ``f_bfast_*`` IIR state in place. A separate ``STEP_ORDER`` slot would
    describe a pass that does not exist.
    """
    install_stubs(monkeypatch, admit=("BFAST PML", "BFAST run"))

    plan = launch_module.plan_step(make_fields(**BFAST_RUN), object())

    assert set(plan.plans) <= set(launch_module.STEP_ORDER)
    assert set(plan.selected) <= set(launch_module.STEP_ORDER)
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")


# ---------------------------------------------------------------------------
# C9 — the expansion probe is resolved ONCE and passed explicitly
# ---------------------------------------------------------------------------

def test_the_probe_record_is_read_once_and_reaches_every_complex_arm(monkeypatch):
    """Reading it per predicate call would open a file up to eight times a plan.

    Worse than the cost: two reads inside one composition could see two
    different records, and the plan would then be composed from two different
    platform facts.
    """
    calls = []
    record = {"backend": "cupy"}
    seen = []

    install_stubs(monkeypatch, admit=("complex PML", "complex"))
    monkeypatch.setattr(launch_module, "load_expansion_probe",
                        lambda path=None: (calls.append(path), record)[1])
    for name in ("complex_pml_curl_coverage", "complex_constitutive_coverage"):
        monkeypatch.setattr(
            launch_module, name,
            (lambda: lambda *args, **kwargs: (seen.append(args[-1]), ADMITTED)[1])(),
            raising=False)

    plan = launch_module.plan_step(make_fields(**COMPLEX_RUN), object())

    assert len(calls) == 1, calls
    assert len(seen) == 4, seen           # both curls and both constitutive sides
    assert all(record is entry for entry in seen)
    assert plan.selected["update_E"] == "complex"


def test_an_ordinary_grid_never_opens_the_probe_artifact(monkeypatch):
    """Every complex arm is gated on complex storage, so an ordinary run skips it."""
    calls = []
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "load_expansion_probe",
                        lambda path=None: (calls.append(path), None)[1])

    launch_module.plan_step(make_fields(**ORDINARY), object())

    assert calls == []


def test_an_explicit_probe_argument_overrides_the_environment(monkeypatch):
    calls = []
    explicit = {"backend": "cupy", "explicit": True}
    seen = []

    install_stubs(monkeypatch, admit=("complex PML", "complex"))
    monkeypatch.setattr(launch_module, "load_expansion_probe",
                        lambda path=None: (calls.append(path), {"env": True})[1])
    monkeypatch.setattr(
        launch_module, "complex_pml_curl_coverage",
        lambda *args, **kwargs: (seen.append(args[-1]), ADMITTED)[1],
        raising=False)

    launch_module.plan_step(make_fields(**COMPLEX_RUN), object(), probe=explicit)

    assert calls == []
    assert seen and all(entry is explicit for entry in seen)


def test_without_a_probe_artifact_the_complex_arms_refuse_by_name(monkeypatch):
    """A None record is a REFUSAL, not a default expansion.

    The EXPANSION constexpr is a measured platform fact. On a host with no probe
    artifact the correct answer is the array path with a reason that says so.
    """
    from meep_gpu.triton_kernels import complex_fields

    monkeypatch.delenv(complex_fields.PROBE_PATH_ENVIRONMENT, raising=False)
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: False)

    plan = launch_module.plan_step(make_fields(**COMPLEX_RUN), None)

    assert "step_B" not in plan.plans and "update_E" not in plan.plans
    assert any("may not be guessed" in reason
               for reason in plan.reasons["step_B"]), plan.reasons["step_B"]
    assert any("may not be guessed" in reason
               for reason in plan.reasons["update_E"]), plan.reasons["update_E"]


# ---------------------------------------------------------------------------
# C10 — the family seam: every family consulted, none imported eagerly
# ---------------------------------------------------------------------------
#
# ``RESERVED_FAMILIES`` used to carry the two families the planner did not
# consult, and two tests iterated it. It is EMPTY now, so those two tests would
# have gone VACUOUS (an empty loop body, and an `assert launch.RESERVED_FAMILIES`
# that simply fails) at exactly the change they existed to catch. What replaced
# them checked only the DECLARED names and was vacuous in the other direction —
# it could not see a module nobody declared — so the check below is an EXHAUSTIVE
# account of the package derived from the shipped AST instead.

#: Modules that exist in the package, are NOT ``plan_step`` arms, and are not
#: named in ``launch.py`` either. Each one's own test file asserts that
#: ``launch.py``'s text never mentions it — that assertion is the seam keeping a
#: deferred product deferred, so the name lives HERE too, from the other direction.
#:
#: 29 -> 6 ON 2026-09-02, and the shrinkage is the whole of the installer wave.
#: Twenty-three of the entries this tuple used to carry were FUSED PAIRS, and the
#: reason given for most of them was the same one: each "admits exactly the
#: configurations a WIRED spine arm already admits", so ``_select_slot`` would see
#: two admitters and leave the slot UNSELECTED — taking the certified curl off the
#: device along with the fusion that displaced it.
#:
#: THAT REASON WAS ABOUT A ROUTE NONE OF THEM WAS EVER GOING TO TAKE, which the two
#: folded pairs had already demonstrated when they left this tuple on 2026-08-27. A
#: fused pair is NOT AN ARM: it never registers on a slot, ``_select_slot`` never
#: sees it, and it is reached only from the opt-in ``fuse`` block, where
#: ``_pair_may_absorb`` lets it take two slots exactly when the arm table has
#: already given both to the arms its kernel implements. The two-admitter hazard
#: does not apply to that route at all. What was actually missing was a composition
#: rule, and ``launch.CERTIFIED_FUSED_PRODUCTS`` plus
#: ``launch._install_certified_fused_products`` are it — so they are declared in
#: ``launch.SUPPORT_MODULES`` beside ``dispersive_fused_pair`` and the two folded
#: pairs, and ``test_triton_certified_fused_products`` pins the route in both
#: directions (installed, and refused at dispatch because unreleased).
#:
#: ``no_pml_ade`` and ``folded_dispersive_update_e`` were here while their
#: admissions shipped unwired — the round that built them did not own ``launch.py``
#: and could add no arm — and they left when they became arms. (The same round's
#: other two closures live inside family modules that were already arms, so they
#: never appeared here.)
#:
#: Every entry below is a claim about ``launch.py`` and is read off it by
#: :func:`test_the_deferred_modules_are_still_not_arms`.
NOT_AN_ARM = (
    # THE FOUR E->P CHAIN PRODUCTS. Each spans ``update_E`` -> ``update_P``, and
    # that seam differs from the pair seam in three ways at once: its second slot
    # holds a LIST of per-susceptibility plans rather than a plan, its first slot
    # is a CONSTITUTIVE sub-step rather than a curl, and their builders take the
    # winning ARM (``fused_ade_chain.ARMS``) rather than a source list. Any one of
    # those makes ``_install_certified_fused_products``' contract false about them,
    # so the installer wave left the seam out of
    # ``launch.CERTIFIED_FUSED_PAIR_SEAMS`` rather than widening a loop it fits
    # none of. The disjointness argument the earlier text asked for DOES now exist
    # — the Triton board measures it on every corpus row (194 since 2026-09-03)
    # (``e_to_p_at_most_one_product_admits``: 0 rows with two admitters;
    # ``e_to_p_chain_arms_are_disjoint``: 0 rows with two admitting arms) — so what
    # is left is the slot protocol, and that is a composition somebody builds
    # deliberately. 15 seam-instances (10 + 4 + 0 + 1) are priced here.
    #
    # ``folded_offdiag_fused_ade_chain`` joined them in the round that built it
    # (2026-09-10) and narrows the account by one clause: its builder takes NO
    # arm — one body, one admission — so of the three differences above only the
    # SLOT PROTOCOL applies to it, which is the difference that was never about
    # the arm argument. It is the fold + tensor-epsilon + dispersion body's chain
    # and the last buildable E->P cell on the Triton board.
    "fused_ade_chain", "complex_fused_ade_chain", "fused_dispersive_chain",
    "folded_offdiag_fused_ade_chain",
    # THE SCRATCH-OUTPUT WELDS' SHARED BASE. ``offdiag_scratch_weld`` is NOT A
    # PRODUCT AT ALL: it is the plan base (the post-launch rotation and, since
    # 2026-09-15, the ``warm`` that never rotates) and the machine-checked lift
    # helpers the two off-diagonal D->E welds and their probe import. It ships no
    # kernel, no predicate and no arm, and ``launch.py`` reaches it only through
    # those two product modules, so there is nothing for the planner to consult.
    #
    # THE TWO WELDS THEMSELVES LEFT THIS TUPLE ON 2026-09-15. They were the only
    # PAIR-shaped products the 2026-09-02 installer wave did not route, and the
    # reason was THE ROTATION rather than the slot arithmetic or the certification:
    # with no ``warm`` on the base, ``fastpath.warm_plan`` fell to
    # ``_warm_with_empty_grid``, which calls ``run`` with an emptied ``_grid``, and
    # ``run`` rotates the engine's ``Dx``/``fu_Dx`` unconditionally. The base's
    # ``warm`` removes that path; both modules are in ``launch.SUPPORT_MODULES`` now,
    # and ``test_triton_certified_fused_products.test_the_two_scratch_output_welds_warm_without_rotating``
    # reads the mechanism off the tree in its new direction.
    "offdiag_scratch_weld",
    # THE H->D WELD, 2026-09-06, AND THE REASON IS A FILE BOUNDARY RATHER THAN THE
    # ARITHMETIC OR THE SLOT PROTOCOL. ``fused_hd_pair`` is pair-shaped: it spans two
    # adjacent slots, it takes a source list, ``launch.CERTIFIED_FUSED_PAIR_SEAMS``
    # already carries the ``update_H -> step_D`` row it would fill, and
    # ``_install_fused_pair`` already routes that row to ``withdraw_hoist``. What
    # routing it costs is a label, and on THIS backend a composer label is not a
    # composer-only fact: ``fastpath`` is the dispatcher, so every label
    # ``plan_step`` can write must also appear in ``fastpath.FUSED_ARM_CONSTITUENTS``
    # (the see-through the pending-gate rung reads, asserted by
    # ``test_triton_certified_fused_products``) and in ``PENDING_DEVICE_GATE_ARMS``
    # until a ledger entry exists. Both live in ``meep_gpu/fastpath.py``, which this
    # round does not own, so the product ships with its predicate, its plan, its
    # tests and its device gate, and the composer is not offered it.
    #
    # NOTHING ABOUT THE COMPOSITION TURNS ON THAT. The product declares
    # ``INSTALLABLE = False`` for a measured verdict of its own
    # (``fused_hd_pair.INSTALLABLE_REASON``): over ``step_B - update_H - step_D -
    # update_E`` launches are ``4 - (installed pairs)`` and this span takes one slot
    # from each neighbour, so on every row of its cell the released B->H pair holds
    # ``update_H`` first and ``_pair_may_absorb`` refuses it. That refusal is DRIVEN
    # rather than asserted, by
    # ``test_triton_fused_hd_pair.test_the_composer_refuses_this_product_on_both_brakes``
    # and by the gate's arbitration leg, both of which inject the row in process
    # instead of editing a file this round does not own.
    "fused_hd_pair",
    # THE 2026-09-07 H->D TAIL. Five more families on the same seam, for the same
    # reason and no other: routing any of them costs a composer LABEL, and on this
    # backend a label `plan_step` can write must also appear in
    # `fastpath.FUSED_ARM_CONSTITUENTS` and in `PENDING_DEVICE_GATE_ARMS` until a
    # ledger entry exists -- both in `meep_gpu/fastpath.py`, which the round that
    # built them does not own. Each declares `INSTALLABLE = False` besides.
    #
    # `nonlinear_fused_hd_pair` is listed here for a WEAKER reason than the other
    # four and the difference is worth a line: it ships no kernel and no plan class
    # at all. It builds `fused_hd_pair.FusedHdPairPlan` through a one-clause scope
    # view, so the flag `launch._declared_uninstallable` reads is that family's, and
    # it is in this list because the account below is over MODULES rather than over
    # products.
    "conductive_bfast_fused_hd_pair", "beta_real_fused_hd_pair",
    "folded_complex_fused_hd_pair", "beta_complex_fused_hd_pair",
    "nonlinear_fused_hd_pair",
    # THE FOLDED H->D WELD, 2026-09-07, here for the SAME file boundary and no other.
    # ``folded_fused_hd_pair`` is pair-shaped exactly as ``fused_hd_pair`` is -- the
    # same two adjacent slots, the same ``CERTIFIED_FUSED_PAIR_SEAMS['update_H']`` row,
    # the same ``withdraw_hoist`` routing -- so what routing it costs is the same LABEL
    # in ``meep_gpu/fastpath.py``, which the round that built it does not own. It
    # covers the largest cell this board carried unbuilt, ``(update_H folded ->
    # step_D folded PML)``: 78 seam-instances of a 597 denominator on
    # ``results/fusion_matrix_triton_2026-09-07_cyl``, 75 ``buildable_not_built``.
    #
    # ITS ARBITRATION IS MEASURED, AND TWO DIFFERENT NUMBERS ARE IN CIRCULATION FOR IT.
    # The gate's own lift leg prices the DRIVEN rows at loss 55 / tie 19 / gain 0
    # (``results/triton_folded_fused_hd_pair_2026-09-07/keep/gate.json``,
    # ``arbitration_counts``, over 74 driven rows) and that is the figure to quote. A
    # predicate-level reading of the same cell has been written down as 67/11; it
    # cannot be a subset of the measured one (19 > 11) because it prices by predicate
    # ADMISSION rather than by INSTALLATION, and 9 of its rows are served at D->E only
    # by ``folded_offdiag_fused_electric_pair``, which this board itself marks
    # ``planner_reachable=false``. Both readings agree on the only load-bearing half:
    # GAIN is ZERO on every row of the cell, so installing this span never lowers the
    # per-step launch count. ``INSTALLABLE = False`` records that.
    "folded_fused_hd_pair")


def _launch_imports():
    """``(module-scope, in-body)`` sibling module names, read off launch.py's AST."""
    tree = ast.parse((PACKAGE_DIR / "launch.py").read_text(encoding="utf-8"))
    in_body = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.ImportFrom) and inner.level == 1
                    and inner.module):
                in_body.add(inner.module)
    module_scope = {node.module for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.level == 1
                    and node.module}
    return module_scope, in_body


def test_every_module_in_the_package_is_accounted_for_by_one_declaration():
    """THE ACCOUNT IS EXHAUSTIVE, in both directions, or it measures nothing.

    The version this replaces asserted that every DECLARED family exists on disk
    and that ``RESERVED_FAMILIES`` is empty, then claimed in its own docstring
    that "a tenth family added without being listed fails here". It did not:
    nothing scanned the package, so a module the planner imports without being
    declared — which is what six of them were — left all three assertions true.

    So the direction is reversed. Every ``*.py`` beside ``launch.py`` must fall in
    exactly one bucket, and the buckets are read off the shipped AST rather than
    re-listed here: what ``launch.py`` imports at module scope, what it imports
    from inside a function body, and what it does not import at all. A new module
    in the package fails here until it is declared; a declared name that stops
    being imported fails here too.
    """
    module_scope, in_body = _launch_imports()

    families = set(launch_module.FAMILY_MODULES)
    support = set(launch_module.SUPPORT_MODULES)
    assert families & support == set(), families & support
    # The lazy seam, EXACT: the two declared tuples are precisely what the
    # shipped file imports from inside a body — no more (a name that stopped
    # being consulted) and no less (a module wired in without a declaration).
    assert families | support == in_body, sorted(
        (families | support) ^ in_body)
    assert (families | support) & module_scope == set()

    on_disk = {path.stem for path in PACKAGE_DIR.glob("*.py")}
    accounted = (families | support | module_scope | set(NOT_AN_ARM)
                 | set(launch_module.RESERVED_FAMILIES)
                 | {"launch", "__init__"})
    assert on_disk == accounted, sorted(on_disk ^ accounted)

    # ...and the nine gated families are still the nine this file drives.
    assert families == {name.rsplit(".", 1)[1] for name in FAMILY_MODULES}
    assert launch_module.RESERVED_FAMILIES == ()


#: What a family module may NOT claim about the planner once it is an arm. Each
#: needle is a present-tense structural assertion about ``launch.py`` — a file
#: the family module does not own — and every one of them was shipped FALSE by
#: the round that wired these four families, in the same commit that made them
#: false.
UNWIRED_CLAIMS = (
    "is not consulted by ``launch.plan_step``",
    "``launch.plan_step`` does not know about this module",
    "composition is NOT touched",
    "composition is DEFERRED",
    "not performed this round",
    "WIRING — none, deliberately",
    "nothing below has been done",
    # The residual-closure round's spelling of the same claim: a section header
    # naming work a LATER round owes ``launch.py``. Once the arm exists the header
    # is a false statement about a file the module does not own.
    "AWAITING WIRING",
    "added no arm to ``plan_step``. What the coordinated wiring round has to do",
)


@pytest.mark.parametrize("family", [
    "folded_complex", "folded_offdiag_update_e", "cylindrical_complex",
    "no_pml_constitutive",
    # The two the residual-group wiring moved out of NOT_AN_ARM. Each shipped
    # with an "AWAITING WIRING" block that told a reader, in present tense, what
    # a later round would have to do to ``launch.py`` — true when written, false
    # the moment the arm landed. This is the seam that reads the claim from the
    # planner's side.
    "folded_dispersive_update_e", "no_pml_ade",
    # A family since the chi2/chi3 tranche, but its update_H and curl admissions
    # were unwired until the same round, and its integration block described them
    # in the imperative.
    "nonlinear_update_e",
    "complex_no_pml_curl", "complex_no_pml_conductive",
    "no_pml_conductive", "no_pml_stored_e", "complex_no_pml_stored_e",
])
def test_a_wired_family_no_longer_claims_the_planner_ignores_it(family):
    """Prose about ANOTHER FILE is a claim, and this is the check on it.

    Every one of these four modules told a reader, in present tense, that
    ``launch.plan_step`` does not consult it — while ``launch.FAMILY_MODULES``
    named it, an arm carried its predicate and ``__init__`` exported it. One went
    further and enumerated the wiring steps under "nothing below has been done"
    with two of them done.

    Nothing detected it because a module's own tests measure its arithmetic, and
    the planner's tests read the planner. This reads the module's TEXT from the
    planner's side, which is the seam where the claim is false.

    Not a prose-style rule: the needles are structural claims about a file the
    module does not own, and a module that stops being an arm is expected to say
    so again (and to leave ``FAMILY_MODULES``, which fails the account above).
    """
    assert family in launch_module.FAMILY_MODULES, family
    text = (PACKAGE_DIR / f"{family}.py").read_text(encoding="utf-8")
    for needle in UNWIRED_CLAIMS:
        assert needle not in text, (family, needle)
    # ...and the claim that IS true stays true: dispatch is not wired.
    assert "fastpath" in text or "fast-path" in text, family


def test_the_deferred_modules_are_still_not_arms():
    """``NOT_AN_ARM`` is a claim about ``launch.py``, so it is read off it.

    Naming a module in ``launch.py`` — even in a comment — trips its own family's
    deferral test, which is why the account above holds the names here. This is
    the other half: each module really is absent from the planner, not merely
    absent from the declarations.
    """
    source = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    for name in NOT_AN_ARM:
        # AS A TOKEN, NOT A SUBSTRING. ``fused_hd_pair`` is deferred while the two
        # cylindrical H->D products (``cylindrical_fused_hd_pair``,
        # ``cylindrical_real_fused_hd_pair``) are routed since 2026-09-07, and the
        # deferred name is a suffix of both routed ones; a substring check would
        # have refused a correct tree.
        assert not re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                             source), name
        assert (PACKAGE_DIR / f"{name}.py").exists(), name


def test_every_family_module_is_imported_only_from_inside_a_function_body():
    """The lazy-import seam, read off the AST rather than off intent.

    Two things break the moment one of these moves to module scope, and only one
    of them is obvious. (1) The package would import nine family modules on a
    bare ``import meep_gpu.triton_kernels``, which every family's own test forbids
    on a machine with no Triton. (2) ``folded_complex`` and ``cylindrical_complex``
    import FROM ``launch`` at THEIR module scope, so a module-scope import here
    closes an import CYCLE and the failure lands on ``import ...launch`` rather
    than on anything a reader of the arm table is looking at.
    """
    tree = ast.parse((PACKAGE_DIR / "launch.py").read_text(encoding="utf-8"))
    nested = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.module:
                nested.add(inner.module)

    module_scope = {node.module for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module}
    for family in launch_module.FAMILY_MODULES:
        assert family in nested, f"{family} is not consulted by the planner"
        assert family not in module_scope, f"{family} is imported at module scope"


#: Which family modules each family module pulls in AT ITS OWN MODULE SCOPE.
#: Three of the four newly wired families restate or import a certified base:
#: importing them is importing those too, and no lazy forwarder can undo that.
#: Declared here so the import-closure test stays exact rather than being
#: loosened to a subset check — and pinned against the shipped files by
#: :func:`test_the_declared_family_import_closure_matches_the_shipped_modules`.
FAMILY_DEPENDENCIES = {
    "special_kz": ("complex_fields",),
    "complex_ade": ("complex_fields",),
    "complex_no_pml_curl": ("complex_fields",),
    "complex_no_pml_conductive": (
        "complex_fields", "complex_no_pml_curl", "no_pml_conductive"),
    "complex_no_pml_stored_e": ("complex_fields",),
    "complex_offdiag_update_e": (
        "complex_fields", "folded_complex", "folded_offdiag_update_e",
        "offdiag_update_e"),
    "folded_complex": ("complex_fields", "special_kz"),
    "cylindrical_complex": ("complex_fields",),
    "folded_offdiag_update_e": ("offdiag_update_e",),
    "folded_offdiag_dispersive_update_e": (
        "folded_offdiag_update_e", "offdiag_update_e"),
}


def _import_closure(names):
    """``names`` plus everything they import at module scope, transitively."""
    pending = list(names)
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(FAMILY_DEPENDENCIES.get(name, ()))
    return seen


def test_the_declared_family_import_closure_matches_the_shipped_modules():
    """The closure above is a claim about other files; here it is read off them."""
    families = set(launch_module.FAMILY_MODULES)
    for family in families:
        tree = ast.parse(
            (PACKAGE_DIR / f"{family}.py").read_text(encoding="utf-8"))
        module_scope = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_scope.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_scope.add(node.module)
                else:
                    # ``from . import offdiag_update_e as _offdiag`` — the
                    # sibling module is an ALIAS here, not the module field, and
                    # a scan that only read ``node.module`` would miss it.
                    module_scope.update(alias.name for alias in node.names)
        assert set(FAMILY_DEPENDENCIES.get(family, ())) == (
            module_scope & families), family


#: One configuration per newly wired family, with the family modules whose gate
#: that configuration turns ON. Every other family module must stay out of
#: ``sys.modules``: the gate block runs before any predicate, so a gate that is
#: False is the only thing standing between an ordinary run and nine imports.
GATED_IMPORT_CASES = (
    ("ordinary", dict(), (), object()),
    ("folded complex", dict(force_complex_fields=True, has_symmetry=lambda: True,
                            is_mirrored=lambda axis: axis == 1),
     ("complex_fields", "folded_complex"), object()),
    ("folded beta", dict(beta=0.7, has_symmetry=lambda: True,
                         is_mirrored=lambda axis: axis == 1),
     ("special_kz", "folded_complex"), object()),
    ("cylindrical complex", dict(cylindrical=True, force_complex_fields=True),
     ("complex_fields", "cylindrical_complex"), object()),
    ("folded off-diagonal", dict(has_offdiagonal_epsilon=True,
                                 has_symmetry=lambda: True,
                                 is_mirrored=lambda axis: axis == 1),
     ("offdiag_update_e", "folded_offdiag_update_e"), object()),
    ("no-PML", dict(), ("no_pml_constitutive",), None),
)


@pytest.mark.parametrize("label,configuration,expected,pml", GATED_IMPORT_CASES,
                         ids=[case[0] for case in GATED_IMPORT_CASES])
def test_a_family_module_is_imported_only_when_its_gate_is_on(
        label, configuration, expected, pml):
    """A closed gate must cost an IMPORT, not just a predicate call.

    The lazy forwarders are what make that true, and nothing else does: the
    import is inside the body, so an arm that is never consulted never opens its
    module. This is measured per configuration rather than asserted once,
    because the four new gates are conjunctions and a gate written with ``or``
    where it needed ``and`` would still pass a single-configuration check.
    """
    # Saved and RESTORED rather than re-imported, for the reason spelled out on
    # ``test_the_package_still_imports_none_of_the_nine_families_eagerly``.
    #
    # THE PACKAGE ATTRIBUTE IS RESTORED TOO, and leaving it out was a real leak
    # rather than tidiness. ``plan_step``'s re-import inside the window rebinds
    # ``meep_gpu.triton_kernels.folded_complex`` (the ATTRIBUTE) to a throwaway
    # module object, while ``sys.modules`` gets the original back — so afterwards
    # ``from meep_gpu.triton_kernels import folded_complex`` and the lazy
    # forwarder's ``from .folded_complex import ...`` returned DIFFERENT module
    # objects. Measured: a later test monkeypatching a module-level helper through
    # the package attribute patched the throwaway, the predicate ran from the
    # original, and the mutation silently did nothing — a test that passes alone
    # and fails in a suite, which is the worst shape this can take.
    saved = {name: sys.modules.get(name) for name in FAMILY_MODULES}
    package = importlib.import_module("meep_gpu.triton_kernels")
    attributes = {name.rsplit(".", 1)[1]:
                  getattr(package, name.rsplit(".", 1)[1], None)
                  for name in FAMILY_MODULES}
    try:
        for name in FAMILY_MODULES:
            sys.modules.pop(name, None)
        launch_module.plan_step(make_fields(**configuration), pml)
        opened = {name.rsplit(".", 1)[1] for name in FAMILY_MODULES
                  if name in sys.modules}
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
        for short, module in attributes.items():
            if module is not None:
                setattr(package, short, module)

    # ``no_pml_constitutive``'s gate reads the LAYER, not the fields, and every
    # configuration here except the active-layer ones leaves it open.
    allowed = set(expected)
    if launch_module.absorber_inactive(pml):
        allowed.update(("no_pml_constitutive", "no_pml_conductive",
                        "no_pml_stored_e"))
        if launch_module.complex_storage_active(make_fields(**configuration)):
            allowed.update(("complex_no_pml_curl",
                            "complex_no_pml_conductive"))
    # EXACT, in both directions. ``<=`` would pass a wiring that consulted
    # nothing at all, which is the other way this seam can rot.
    assert opened == _import_closure(allowed), (
        label, sorted(opened ^ _import_closure(allowed)))


# ---------------------------------------------------------------------------
# C11 / C12 — the import graph and the export surface
# ---------------------------------------------------------------------------

def test_the_gated_import_window_leaves_one_module_object_per_family():
    """The restore above, measured — and the hazard it closes, named.

    A family is reachable two ways: ``from meep_gpu.triton_kernels import
    folded_complex`` (the package ATTRIBUTE) and ``from .folded_complex import
    ...`` inside a lazy forwarder (which resolves through ``sys.modules``). The
    test above pops every family out of ``sys.modules`` and lets ``plan_step``
    re-import inside the window, so without restoring the ATTRIBUTE too the two
    routes came back naming DIFFERENT module objects — and a monkeypatch applied
    through the attribute then patched a throwaway while the predicate ran from
    the original. That is a test which passes alone and fails in a suite, which
    is the worst shape this can take; it was found exactly that way.

    Scoped to THIS file's window on purpose. The same asymmetry is reachable
    anywhere a test does ``monkeypatch.delitem(sys.modules, ...)`` and then
    re-imports — monkeypatch restores the dict entry and nothing restores the
    attribute — and several optional-dependency tests in the family suites do.
    Asserting it globally here would report those rather than this, so the
    durable defence is the one the residual-group suite uses: resolve a family
    module the way the SHIPPED forwarders resolve it, through ``sys.modules``.
    """
    package = importlib.import_module("meep_gpu.triton_kernels")
    saved = {name: sys.modules.get(name) for name in FAMILY_MODULES}
    attributes = {name.rsplit(".", 1)[1]:
                  getattr(package, name.rsplit(".", 1)[1], None)
                  for name in FAMILY_MODULES}
    # Only the families that AGREE going in can say anything about this window's
    # restore. A family some earlier optional-dependency test already split is a
    # fact about that test; asserting it here would report the session's history
    # instead of this block's behaviour, and would make this test's verdict depend
    # on collection order.
    agreed = {name for name in FAMILY_MODULES
              if name in sys.modules
              and getattr(package, name.rsplit(".", 1)[1], None) is sys.modules[name]}
    try:
        for name in FAMILY_MODULES:
            sys.modules.pop(name, None)
        launch_module.plan_step(
            make_fields(force_complex_fields=True, has_symmetry=lambda: True,
                        is_mirrored=lambda axis: axis == 1),
            object())
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
        for short, module in attributes.items():
            if module is not None:
                setattr(package, short, module)

    assert agreed, "no family agreed going in; this window measured nothing"
    for name in sorted(agreed):
        short = name.rsplit(".", 1)[1]
        assert getattr(package, short) is sys.modules[name], short


def test_the_package_still_imports_none_of_the_nine_families_eagerly():
    """Duplicated from the nine family tests, asserted at the PLANNER boundary.

    Wiring the planner is exactly the change that would break this, because it
    is the change that gives the package a reason to name those modules.

    THE ORIGINAL MODULE OBJECTS ARE PUT BACK, not re-imported. Re-importing makes
    a NEW module object with new classes, and any test elsewhere in the suite
    holding a reference to the old one then fails an ``isinstance`` for reasons
    that have nothing to do with what it measures. Restoring the objects keeps
    this test's blast radius inside this test.
    """
    saved = {name: sys.modules.get(name)
             for name in (*FAMILY_MODULES, "meep_gpu.triton_kernels")}
    try:
        for name in saved:
            sys.modules.pop(name, None)
        importlib.import_module("meep_gpu.triton_kernels")
        eager = [name for name in FAMILY_MODULES if name in sys.modules]
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module
    assert eager == []


NEW_EXPORTS = (
    "load_expansion_probe",
    "complex_pml_curl_coverage", "plan_complex_pml_curl",
    "complex_constitutive_coverage", "plan_complex_constitutive",
    "beta_pml_curl_coverage", "plan_beta_pml_curl",
    "beta_bloch_pml_curl_coverage", "plan_beta_bloch_pml_curl",
    "beta_run_constitutive_coverage", "plan_beta_run_constitutive",
    "beta_run_complex_constitutive_coverage", "plan_beta_run_complex_constitutive",
    "bfast_pml_curl_coverage", "plan_bfast_pml_curl",
    "bfast_run_constitutive_coverage", "plan_bfast_run_constitutive",
    "nonlinear_constitutive_coverage", "plan_nonlinear_constitutive",
    "offdiag_constitutive_coverage", "plan_offdiagonal_constitutive",
    "folded_complex_composition_curl_coverage", "plan_folded_complex_pml_curl",
    "folded_beta_pml_curl_coverage", "plan_folded_beta_pml_curl",
    "folded_beta_bloch_pml_curl_coverage", "plan_folded_beta_bloch_pml_curl",
    "folded_mirror_ghost_fill_complex_coverage",
    "plan_folded_mirror_ghost_fill_complex",
    "folded_complex_constitutive_coverage", "plan_folded_complex_constitutive",
    "folded_beta_run_constitutive_coverage", "plan_folded_beta_run_constitutive",
    "folded_offdiag_composition_coverage", "plan_folded_offdiagonal_constitutive",
    "cylindrical_complex_curl_coverage", "plan_cylindrical_complex_curl",
    "cylindrical_complex_constitutive_coverage",
    "plan_cylindrical_complex_constitutive",
    "null_constitutive_coverage", "plan_null_constitutive",
)


def test_the_package_exports_every_wired_family_entry_point():
    package = importlib.import_module("meep_gpu.triton_kernels")
    for name in NEW_EXPORTS:
        assert name in package.__all__, name
        assert callable(getattr(package, name)), name
        assert callable(getattr(launch_module, name)), name


@pytest.mark.parametrize("name", NEW_EXPORTS)
def test_every_wired_entry_point_refuses_a_gridless_double_without_raising(name):
    """A predicate that raises is a crash where a refusal was the right answer."""
    package = importlib.import_module("meep_gpu.triton_kernels")
    entry = getattr(package, name)
    fields = SimpleNamespace(polarizations=())
    parameters = inspect.signature(entry).parameters

    if name == "load_expansion_probe":
        assert entry("/nonexistent/probe.json") is None
        return
    if "family" in parameters:
        # The two ghost-fill entry points take a FIELD FAMILY where the others
        # take a layer: their second parameter is "B"/"D", not the PML.
        arguments = [fields, "B"]
    else:
        arguments = [fields, None]
        if "sub_step" in parameters:
            arguments.append("step_B")
        elif "side" in parameters:
            arguments.append("E")

    result = entry(*arguments)
    if result is None or isinstance(result, coverage_module.Coverage):
        assert result is None or result.covered is False
    else:  # pragma: no cover - a plan on a gridless double is the defect
        pytest.fail(f"{name} built {result!r} for a fields double with no grid")


# ---------------------------------------------------------------------------
# C13 — the REAL predicates, driven through plan_step
# ---------------------------------------------------------------------------
#
# Everything above this line replaces all nineteen predicates with constants, so
# every assertion about which arm wins is an assertion about ``_select_slot``'s
# arithmetic on verdicts the test itself supplied. That is the right shape for a
# selection-rule test and it is BLIND to the whole feature-interaction surface:
# a family predicate that admitted a grid it does not implement would leave every
# test above green.
#
# These call plan_step with the SHIPPED predicates and replace only the BUILDERS,
# at the module seam this file's header describes. The doubles are real ``Grid``/
# ``Fields``/``PML`` objects with one thing faked — the array module's NAME, which
# is the shared clause that would otherwise refuse a NumPy host at clause 1 and
# make every one of these cases vacuous. Nothing here launches: the builders are
# sentinels and the machine has no Triton.

class NamedAsCupy:
    """NumPy, answering to the name the shared backend clause looks for.

    ``_grid_reasons`` clause 1 refuses anything whose ``grid.xp.__name__`` is not
    ``"cupy"``, so on the merge-bar host every real predicate would refuse at the
    first clause and every case below would be vacuous. THE NAME IS THE ONLY
    FAKE: every attribute is NumPy's, the arrays are real, and the clauses that
    follow — layout, dtype, contiguity, boundary kinds, coefficient tables, row
    slots — are evaluated against them.
    """

    __name__ = "cupy"

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)


def real_objects(cell_size=(0.8, 0.8, 0.8), fields_class=None, **grid_kwargs):
    """A real Grid/Fields/PML triple whose array module ANSWERS TO 'cupy'."""
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=cell_size, **grid_kwargs)
    grid.xp = NamedAsCupy(numpy)
    fields = (fields_class or Fields)(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=2)


def no_absorber_objects():
    """A real Grid/Fields pair with NO absorber and NO PML storage switched on.

    Not ``real_objects``: that one calls ``enable_pml_storage``, and the null
    family refuses a run whose storage was switched on behind an inert layer —
    conservatively, and by name (``_storage_switch_reasons``). Getting this wrong
    would make every null test below vacuous in the "refused" direction.
    """
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35)
    grid.xp = NamedAsCupy(numpy)
    fields = Fields(grid=grid, force_complex_fields=False)
    return fields, PML(grid=grid, thickness=0)


def install_sentinel_builders(monkeypatch):
    """Replace every BUILDER with a sentinel. EVERY PREDICATE STAYS REAL.

    The builders are what need Triton and a device; the predicates are what this
    file is about. Splitting them at the module seam is what lets the shipped
    coverage rules run on a laptop.
    """
    for builder in sorted(set(BUILDERS.values())) + [
            "plan_mirror_ghost_fill", "plan_ade_update_p",
            "plan_fused_ade_state", "plan_fused_pair",
            "plan_dispersive_fused_pair",
            # The update_P sibling. Not in ``BUILDERS`` because that dict is keyed
            # by ARM-TABLE predicate and update_P has no arm table — its family is
            # chosen once for the whole sub-step inside the polarization block.
            "plan_no_pml_ade_update_p", "plan_complex_ade_update_p"]:
        if builder == "plan_null_constitutive":
            # THE ONE BUILDER LEFT REAL. Every other one launches a kernel and
            # needs Triton and a device; this one allocates nothing, imports
            # nothing and returns a NullConstitutivePlan, so replacing it would
            # only hide the one plan object this file can legitimately inspect.
            continue
        monkeypatch.setattr(
            launch_module, builder,
            (lambda builder: lambda *a, **k: f"<{builder}>")(builder),
            raising=False)


def plant_offdiagonal_row(fields):
    """Install one live ROW SLOT — ``ROW_SLOTS[0]``, ('Ex', 'Ey')."""
    import numpy

    from meep_gpu.triton_kernels.offdiag_update_e import ROW_SLOTS

    row, partner = ROW_SLOTS[0]
    volume = numpy.ascontiguousarray(
        numpy.full(fields.grid.shape, 0.25, dtype=numpy.float32))
    fields._chi1inv_offdiagonal = {row: {partner: volume}}
    return volume


def flag_blind_fields_class():
    """A ``Fields`` subclass whose off-diagonal FLAG is forced False.

    A subclass rather than an attribute assignment because the engine's flag is a
    read-only property over the same dict the rows live in — see
    ``test_the_engine_cannot_produce_a_flag_false_live_row``. Writing the test
    this way keeps the harness honest about what it is doing: MANUFACTURING a
    disagreement the engine cannot produce, because the gates, the probes and
    this suite all build doubles and any of them could.
    """
    from meep_gpu.fields import Fields

    class FlagBlindFields(Fields):
        @property
        def has_offdiagonal_epsilon(self) -> bool:
            return False

    return FlagBlindFields


def test_an_ordinary_grid_selects_the_shipped_products_with_every_real_predicate(
        monkeypatch):
    """THE NON-VACUITY CONTROL for everything in this section.

    Real predicates, real arrays, one faked backend name: the shipped PML curls
    and the ordinary constitutive product must win all four slots, unopposed and
    with no reasons at all. If this goes empty the rest of the section is
    measuring nothing.
    """
    install_sentinel_builders(monkeypatch)
    fields, pml = real_objects()

    plan = launch_module.plan_step(fields, pml)

    assert plan.selected == {
        "step_B": "PML", "step_D": "PML",
        "update_H": "ordinary", "update_E": "ordinary",
    }, plan.reasons
    assert plan.reasons == {}


def test_the_array_path_forms_the_row_product_only_under_the_FLAG():
    """GROUND TRUTH for the veto, measured on ``stepping.update_E`` itself.

    The veto's whole justification is "the selected product would drop a coupling
    the array path computes", so what the ARRAY PATH does with a live row is the
    only thing that can settle when it should fire. ``stepping.update_E`` reads
    ``offdiagonal = fields.has_offdiagonal_epsilon`` (stepping.py:992) — the FLAG,
    not the slots — so on a flag-blind object it forms no row term at all, and a
    Triton product that also forms none is byte-exact rather than wrong.

    Two identical-input copies of one object, one with ``ROW_SLOTS[0]`` planted
    and one without, stepped through the real ``update_E`` and compared word for
    word. This is what makes the flag half of the veto's trigger a measurement
    rather than a preference.
    """
    import numpy

    from meep_gpu import stepping

    def moved(fields_class, thickness):
        pair = []
        for plant in (False, True):
            fields, layer = real_objects(fields_class=fields_class)
            if thickness == 0:
                from meep_gpu.pml import PML
                layer = PML(grid=fields.grid, thickness=0)
            rng = numpy.random.default_rng(7)
            for name in ("Dx", "Dy", "Dz"):
                getattr(fields, name)[...] = rng.standard_normal(
                    fields.grid.shape).astype(numpy.float32)
            if plant:
                plant_offdiagonal_row(fields)
            stepping.update_E(fields, layer)
            pair.append(numpy.concatenate([
                numpy.ascontiguousarray(getattr(fields, name)).ravel().view(
                    numpy.uint32) for name in ("Ex", "Ey", "Ez")]))
        return int((pair[0] != pair[1]).sum()), pair[0].size

    blind = flag_blind_fields_class()
    assert moved(blind, 2) == (0, 1536)      # flag False, active PML
    assert moved(blind, 0) == (0, 1536)      # flag False, inert layer
    differing, total = moved(None, 2)        # the real class: the flag is True
    assert differing == 512 and total == 1536, (differing, total)


def test_a_flag_blind_live_row_is_NOT_vetoed_because_nothing_is_dropped(
        monkeypatch):
    """The half of the trigger the wiring round shipped without, pinned.

    This configuration — BFAST on, a live ``ROW_SLOTS[0]`` volume, the
    off-diagonal FLAG blinded False — used to be the veto's headline case: the
    off-diagonal arm drops out through the SHARED grid clauses, ``BFAST run``
    survives alone, and its element-wise kernel has no row-product term. But the
    array path has none either on this object (the test above measures 0 of 1536
    words), so vetoing removed a CORRECT selection and installed a reason that
    was false about the product it named.

    A veto that can only fire where nothing is dropped is not fail-closed, it is
    lost coverage with a misleading record.
    """
    install_sentinel_builders(monkeypatch)
    fields, pml = real_objects(fields_class=flag_blind_fields_class(),
                               bfast_scaled_k=(0.3, 0.0, 0.0))
    plant_offdiagonal_row(fields)

    assert fields.has_offdiagonal_epsilon is False
    assert launch_module.live_offdiagonal_rows(fields) is True

    plan = launch_module.plan_step(fields, pml)

    assert plan.selected == {"step_B": "BFAST PML", "step_D": "BFAST PML",
                             "update_H": "BFAST run", "update_E": "BFAST run"}
    assert "update_E" not in plan.reasons


def test_a_flag_true_live_row_still_vetoes_a_product_that_would_drop_it(
        monkeypatch):
    """THE OVER-COVER THIS SECTION EXISTS FOR, on the half that is real.

    Same shape as the case above with the FLAG left alone: the array path forms
    the row product here (512 of 1536 words, measured above), so a flag-reading
    element-wise product really would drop it and the slot must go to the array
    path with a named reason.

    Reaching it takes a mutant, and that is the point of keeping the veto. Under
    the SHIPPED predicates all thirteen non-exempt ``update_E`` arms carry their
    own ``has_offdiagonal_epsilon`` refusal, so a flag-True BFAST grid leaves the
    slot to nobody — the assertion below records that as the shipped verdict.
    Delete that clause from any one of the thirteen and the veto is what catches
    it, at the composition, rather than the next device gate for that family.

    The other three slots are untouched either way. A veto empties one slot; it is
    not a selection rule and cannot hand the slot to anyone else.
    """
    install_sentinel_builders(monkeypatch)
    fields, pml = real_objects(bfast_scaled_k=(0.3, 0.0, 0.0))
    plant_offdiagonal_row(fields)

    assert fields.has_offdiagonal_epsilon is True
    assert launch_module.live_offdiagonal_rows(fields) is True

    shipped = launch_module.plan_step(fields, pml)
    assert "update_E" not in shipped.selected
    assert not any("live off-diagonal" in reason
                   for reason in shipped.reasons["update_E"]), shipped.reasons

    # The mutant: one arm's flag clause deleted, so it survives alone.
    monkeypatch.setattr(launch_module, "bfast_run_constitutive_coverage",
                        lambda fields, pml, side: coverage_module.Coverage(True, ()))
    plan = launch_module.plan_step(fields, pml)

    assert "update_E" not in plan.plans
    assert "update_E" not in plan.selected
    reason, = plan.reasons["update_E"]
    assert "live off-diagonal chi1inv row slot" in reason, reason
    assert "'BFAST run'" in reason, reason
    assert plan.selected == {"step_B": "BFAST PML", "step_D": "BFAST PML",
                             "update_H": "BFAST run"}


def test_the_veto_does_not_fire_where_no_row_is_live(monkeypatch):
    """A mutant veto that fired on the ACCESSOR rather than on a live ROW would
    disable ``update_E`` for every engine-route run — every real ``Fields`` holds
    the accessor. This is the case that catches it."""
    install_sentinel_builders(monkeypatch)
    fields, pml = real_objects(bfast_scaled_k=(0.3, 0.0, 0.0))

    assert callable(getattr(fields, "chi1inv_offdiagonal_for", None))
    assert launch_module.live_offdiagonal_rows(fields) is False

    plan = launch_module.plan_step(fields, pml)

    assert plan.selected["update_E"] == "BFAST run"
    assert "update_E" not in plan.reasons


def live_row_accessor():
    """A ``chi1inv_offdiagonal_for`` that fills ``ROW_SLOTS[0]`` and nothing else."""
    from meep_gpu.triton_kernels.offdiag_update_e import ROW_SLOTS

    row, partner = ROW_SLOTS[0]
    volume = object()
    return lambda name: ({partner: volume} if name == row else {})


def test_a_vetoed_update_E_takes_its_fused_pair_curl_with_it():
    """A fused pair is ONE kernel; half of it cannot be kept.

    The veto has to run after the fusion block — a fused D/E pair inlines the
    same flag-reading constitutive product, so a veto placed before it would be
    overwritten by the very product it refused. And when the vetoed ``update_E``
    was absorbed, the curl slot that carried it goes too.

    DRIVEN AT THE VETO rather than through ``plan_step``, because since the
    trigger gained its flag half the two cannot meet in one composition — see
    the test below, which pins that instead of leaving it implied. The absorbed
    shape is still the one the fusion block builds: a ``NoopPlan`` whose
    ``absorbed_by`` is the pair object sitting in the curl slot.
    """
    pair = object()
    plans = {"step_D": pair, "update_E": launch_module.NoopPlan("update_E", pair)}
    reasons = {}
    selected = {"step_D": "fused pair D", "update_E": "fused pair D"}
    fields = make_fields(has_offdiagonal_epsilon=True,
                         chi1inv_offdiagonal_for=live_row_accessor())
    assert launch_module.live_offdiagonal_rows(fields) is True

    launch_module._veto_dropped_offdiagonal_coupling(
        fields, plans, reasons, selected)

    # The D pair is gone, both halves, each with its own reason.
    assert "update_E" not in plans and "step_D" not in plans
    assert "update_E" not in selected and "step_D" not in selected
    assert any("live off-diagonal chi1inv row slot" in reason
               for reason in reasons["update_E"]), reasons["update_E"]
    assert any("one kernel cannot be half-kept" in reason
               for reason in reasons["step_D"]), reasons["step_D"]


def test_opt_in_fusion_and_the_veto_can_no_longer_meet_in_one_composition(
        monkeypatch):
    """A structural consequence of the flag half, pinned rather than assumed.

    THE CLAIM SURVIVED THE 2026-09-13 SPLIT BUT ITS MECHANISM CHANGED, and this
    docstring is rewritten rather than the assertions relaxed. It used to read: the
    veto needs ``has_offdiagonal_epsilon`` True, and that same attribute refuses
    EVERY opt-in pair by name, so the two can never meet. The second half is no
    longer true — ``off_diagonal_owns_update_e`` refuses only the E-side pair, so
    the magnetic pair composes here.

    WHAT KEEPS THEM APART NOW IS THE VETO'S OWN FIRST LINE:
    ``if selected.get("update_E") in ROW_PRODUCT_ARMS: return``. That tuple
    contains ``None``, and on this configuration nothing wins ``update_E`` at all
    (only PML and ordinary are admitted), so the veto EARLY-RETURNS and has
    nothing to take. The interaction the test above exercises directly is still
    unreachable through ``plan_step`` — and a change to either half (the veto
    ceasing to early-return, or the E-side refusal widening back over B) fails
    here.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *a, **k: object(), raising=False)

    fields = make_fields(has_offdiagonal_epsilon=True,
                         chi1inv_offdiagonal_for=live_row_accessor())
    assert launch_module.live_offdiagonal_rows(fields) is True

    plan = launch_module.plan_step(fields, object(), fuse=True, sources=())

    # THE VETO EARLY-RETURNED, which is what keeps it away from the fused pair
    # below: nothing wins ``update_E`` on this configuration (only PML and ordinary
    # are admitted), and ``None`` is a MEMBER of ``ROW_PRODUCT_ARMS`` — the tuple
    # the veto's own first line tests.
    assert plan.selected.get("update_E") in launch_module.ROW_PRODUCT_ARMS
    # THE E-SIDE SEAM is still refused by name...
    assert any("specialized family owns this grid" in reason
               for reason in plan.reasons["fused_pair_D"])
    # ...while the magnetic pair COMPOSES, carrying no refusal at all. Before the
    # 2026-09-13 split this line read ``"fused pair" not in str(plan.selected)``,
    # which the narrowing inverted for B and only for B.
    assert plan.selected["step_B"] == "fused pair B"
    assert plan.selected["update_H"] == "fused pair B"
    assert not (plan.reasons.get("fused_pair_B") or ())
    # ...and the separately built plans survive: a refused fusion is not a
    # refused step, and the veto had nothing absorbed to take with it.
    assert plan.selected["step_D"] == "PML"


def test_the_engine_cannot_produce_a_flag_false_live_row():
    """The reachability the gate's docstring rests on, MEASURED on the real class.

    ``has_offdiagonal_epsilon`` is a read-only property returning
    ``bool(self._chi1inv_offdiagonal)`` and ``chi1inv_offdiagonal_for`` reads the
    same dict, so the two move together and the flag cannot be pushed False under
    a live row from outside. The OPPOSITE direction — flag True, every slot dead —
    is the reachable one, and there every arm refuses.

    This is why the veto is not redundant with the disjunctive gate: on the
    engine route the gate's ambiguity is unreachable, while the veto's trigger
    (a live row the winning product does not implement) is reachable the moment a
    second feature removes the specialized arm.
    """
    import inspect as _inspect

    from meep_gpu.fields import Fields

    descriptor = _inspect.getattr_static(Fields, "has_offdiagonal_epsilon")
    assert isinstance(descriptor, property)
    assert descriptor.fset is None

    fields, _ = real_objects()
    assert fields.has_offdiagonal_epsilon is False
    with pytest.raises(AttributeError):
        fields.has_offdiagonal_epsilon = False

    plant_offdiagonal_row(fields)
    assert fields.has_offdiagonal_epsilon is True
    assert launch_module.live_offdiagonal_rows(fields) is True


# ---------------------------------------------------------------------------
# C13b — the veto's exemption list, and the null plan's two structural facts
# ---------------------------------------------------------------------------

def test_the_folded_row_product_survives_the_veto_that_exists_to_protect_it():
    """THE ONE PRE-EXISTING BUG THIS WIRING WOULD HAVE TRIPPED, pinned.

    ``_veto_dropped_offdiagonal_coupling`` refuses an ``update_E`` product that
    reads the off-diagonal FLAG instead of counting live ROW SLOTS. Its exemption
    list held two entries — ``None`` and ``off-diagonal`` — and the arm this
    round adds counts slots exactly as ``off-diagonal`` does. Without the third
    entry the veto fires on the very product that implements the row term, empties
    the slot, and reports "has no row-product term" about a kernel whose whole
    body is that term.

    It would have been silent: the slot falls to the array path, which is the
    correct answer, so only the reasons record would have been wrong — and wrong
    in the direction that sends a reader to the wrong file.
    """
    assert "folded off-diagonal" in launch_module.ROW_PRODUCT_ARMS

    plans = {"update_E": "<the folded row product>"}
    reasons = {}
    selected = {"update_E": "folded off-diagonal"}
    fields = make_fields(has_offdiagonal_epsilon=True,
                         chi1inv_offdiagonal_for=live_row_accessor())
    assert launch_module.live_offdiagonal_rows(fields) is True

    launch_module._veto_dropped_offdiagonal_coupling(
        fields, plans, reasons, selected)

    assert selected["update_E"] == "folded off-diagonal"
    assert plans["update_E"] == "<the folded row product>"
    assert "update_E" not in reasons


def test_the_null_arm_cannot_be_the_one_the_veto_empties():
    """Why ``no-PML null`` is NOT in ``ROW_PRODUCT_ARMS``, measured not argued.

    The veto below still removes it from a hand-built dict, and that is the one
    entry in the list whose reason string would be false if ``plan_step`` could
    ever produce it: the null product is selected only where the layer is inert
    and ``stores_E`` is False, and there ``stepping.update_E`` returns at :954
    without executing a statement — a sub-step that writes nothing cannot drop a
    coupling.

    It cannot produce it. The veto now needs the off-diagonal FLAG, and the null
    family's own E-side predicate refuses an installed row BY NAME on exactly
    that attribute. So the pair below is the reachability argument, and adding
    ``no-PML null`` to the exemption tuple would fix nothing while making one
    tuple mean two things — "implements the row product" and "has no product at
    all".
    """
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML
    from meep_gpu.triton_kernels.no_pml_constitutive import (
        null_constitutive_coverage)

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
                xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=False)
    plant_offdiagonal_row(fields)
    layer = PML(grid=grid, thickness=0)

    assert fields.has_offdiagonal_epsilon is True
    verdict = null_constitutive_coverage(fields, layer, "E")
    assert verdict.covered is False
    assert any("off-diagonal chi1inv row is installed" in reason
               for reason in verdict.reasons), verdict.reasons
    assert "no-PML null" not in launch_module.ROW_PRODUCT_ARMS


def test_the_veto_still_fires_on_every_arm_that_does_not_count_row_slots():
    """The exemption grew by one entry and by exactly one entry.

    A mutant that exempted the whole fold — or that dropped the check — would
    pass the test above and fail here: the folded ORDINARY constitutive product
    reads the flag like the other twelve and must still be vetoed off a grid
    whose row slot is live.

    ``no-PML null`` is in the list because the exemption tuple must not grow to
    cover it, not because ``plan_step`` can reach it with that arm selected — see
    the test above for why it cannot.
    """
    fields = make_fields(has_offdiagonal_epsilon=True,
                         chi1inv_offdiagonal_for=live_row_accessor())
    for arm in ("folded", "ordinary", "complex", "BFAST run", "folded complex",
                "cylindrical complex", "no-PML null"):
        plans = {"update_E": object()}
        reasons = {}
        selected = {"update_E": arm}
        launch_module._veto_dropped_offdiagonal_coupling(
            fields, plans, reasons, selected)
        assert "update_E" not in selected, arm
        assert "update_E" not in plans, arm
        assert any("live off-diagonal chi1inv row slot" in reason
                   for reason in reasons["update_E"]), arm


def test_a_null_slot_is_a_null_plan_and_is_not_the_fused_pairs_noop(monkeypatch):
    """``NullConstitutivePlan`` is NOT ``launch.NoopPlan``, and the difference is a FACT.

    ``NoopPlan`` means "the work happened, over there" and carries ``absorbed_by``;
    a null slot means "there was no work". A driver adapter that read the first
    for the second would go looking for a fused kernel that does not exist — and
    one that read the second for the first would run the array path's ``update_H``
    on top of a plan that already replaced it.

    ``replaces`` reporting ``update_H`` is correct and load-bearing: the sub-step
    IS replaced, so the adapter must NOT also call the array path's.
    """
    from meep_gpu.triton_kernels.no_pml_constitutive import NullConstitutivePlan

    install_sentinel_builders(monkeypatch)
    fields, layer = no_absorber_objects()

    plan = launch_module.plan_step(fields, layer)

    assert plan.selected["update_H"] == "no-PML null"
    installed = plan.plans["update_H"]
    assert isinstance(installed, NullConstitutivePlan)
    assert not isinstance(installed, launch_module.NoopPlan)
    assert not hasattr(installed, "absorbed_by")
    assert "update_H" in plan.replaces


@pytest.mark.parametrize("attribute,slot", [
    ("stores_E", "update_E"),
    ("polarizations", "update_H"),
])
def test_the_null_predicate_raising_on_an_ndarray_flag_is_a_NAMED_refusal(
        attribute, slot):
    """A REAL defect in the family module, contained by the composer — measured.

    ``no_pml_constitutive._flag`` catches exceptions from ``getattr`` but computes
    ``bool(value)`` OUTSIDE the try, and ``_magnetic_susceptibility_reasons``
    calls ``tuple(...)`` on whatever ``polarizations`` holds. An ndarray in either
    place raises ``ValueError`` — measured, not supposed — and that family's own
    docstring names the rule it breaks: "a predicate that raises where its sibling
    refuses becomes a crash the moment ``launch.plan_step`` consults it".

    It does not become a crash, because ``_arm_verdict`` treats a raising
    predicate as a REFUSAL. This test is what keeps that containment honest: the
    slot is unselected, the reason NAMES the arm and says it raised, and the plan
    comes back. The fix still belongs in the family module; until it lands, this
    is the record that the composition is safe without it.
    """
    import numpy

    from meep_gpu.triton_kernels import no_pml_constitutive

    fields = make_fields(**ORDINARY)
    setattr(fields, attribute, numpy.zeros(3))

    with pytest.raises(ValueError):
        no_pml_constitutive.null_constitutive_coverage(fields, None, slot[-1])

    plan = launch_module.plan_step(fields, None)      # must not raise

    assert slot not in plan.plans
    assert any("no-PML null" in reason and "raised" in reason
               for reason in plan.reasons[slot]), plan.reasons[slot]


def test_opt_in_fusion_cannot_absorb_a_null_slot(monkeypatch):
    """TWO independent barriers, and neither one is a special case for the null.

    The pair predicate refuses at its curl half ("no active PML layer"), and
    ``_pair_may_absorb`` requires ``selected['update_H'] == 'ordinary'`` while a
    null slot reports ``'no-PML null'``. So ``specialized_family_owns_the_grid``
    needs no new term for this family: the general rule already covers it.
    """
    install_sentinel_builders(monkeypatch)
    fields, layer = no_absorber_objects()

    plan = launch_module.plan_step(fields, layer, fuse=True, sources=())

    from meep_gpu.triton_kernels.no_pml_constitutive import NullConstitutivePlan
    assert isinstance(plan.plans["update_H"], NullConstitutivePlan)
    assert plan.reasons["fused_pair_B"], plan.reasons
    assert "step_B" not in plan.selected or plan.selected["step_B"] != "fused pair B"


# ---------------------------------------------------------------------------
# C14 — "never raises" is a contract for the WHOLE function
# ---------------------------------------------------------------------------
#
# ``_select_slot`` honoured it for four of the seven STEP_ORDER slots. The folded
# ghost fills, the opt-in fusion block and the polarization loop did not, and
# neither did the gate block — so on the stated merge-bar host (no Triton) a
# covered dispersive run and every fuse=True run raised out of the composer.

def test_a_covered_dispersive_run_refuses_rather_than_raising_without_triton(
        monkeypatch):
    """The merge-bar host. A builder's missing optional dependency is a REFUSAL.

    No builder is stubbed here: this is the real ``plan_ade_update_p`` reaching
    the real ``from .kernels import DEFAULT_BLOCK`` on a machine with no Triton.
    """
    import numpy

    from meep_gpu.dispersion import PolarizationState, Susceptibility

    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: False)
    fields, pml = real_objects()
    fields.enable_field_storage()
    volume = numpy.zeros(fields.grid.shape, dtype=numpy.float32)
    volume[1:-1, 1:-1, 1:-1] = 0.3
    state = PolarizationState(
        Susceptibility(frequency=1.0, gamma=0.1, kind="lorentzian"),
        {"Ex": 0.0, "Ey": 0.0, "Ez": volume}, fields.grid, numpy.float32)
    fields.polarizations.append(state)

    plan = launch_module.plan_step(fields, pml)      # must not raise

    assert "update_P" not in plan.plans
    assert any("triton" in reason.lower() for reason in plan.reasons["update_P"]), (
        plan.reasons["update_P"])


def test_opt_in_fusion_refuses_rather_than_raising_without_triton():
    """``fuse=True`` on a covered ordinary grid, on a host with no Triton.

    Nothing is stubbed. Every arm's builder refuses on the missing optional
    dependency, so all four slots fall to the array path with that reason named —
    and the fused pair, which may only absorb slots the arm table filled, is
    refused for that reason rather than raising a second copy of the import
    error out of the composer.
    """
    fields, pml = real_objects()

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=())

    assert plan.replaces == ()
    assert not isinstance(plan.plans.get("step_B"), launch_module.NoopPlan)
    assert any("triton" in reason.lower()
               for reason in plan.reasons["step_B"]), plan.reasons["step_B"]
    for key in ("fused_pair_B", "fused_pair_D", "fused_pair_dispersive_D"):
        assert plan.reasons[key], key


def test_a_malformed_source_list_refuses_the_fusion_rather_than_raising(
        monkeypatch):
    """``sources`` comes from the driver, not from this composer.

    ``fused_pair_coverage`` does ``tuple(sources)``; a non-sequence raised out of
    plan_step from a block with no guard.
    """
    install_sentinel_builders(monkeypatch)
    fields, pml = real_objects()

    plan = launch_module.plan_step(fields, pml, fuse=True, sources=object())

    assert plan.selected["step_B"] == "PML"          # the composition survived
    assert any("raised" in reason for reason in plan.reasons["fused_pair_B"]), (
        plan.reasons["fused_pair_B"])


@pytest.mark.parametrize("attribute", [
    "cylindrical", "has_bloch", "bfast_active",
])
def test_an_unreadable_grid_gate_consults_its_arm_instead_of_aborting_the_plan(
        monkeypatch, attribute):
    """A gate reads an attribute; the gate block runs BEFORE any predicate.

    An attribute whose truth value raises — a numpy array is the everyday one —
    used to take down all seven slots, including the ones that arm has nothing to
    do with. The file's own rule is the opposite: an unreadable gate CONSULTS its
    arm and lets the arm's predicate refuse by name.
    """
    import numpy

    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    fields = make_fields(**{attribute: numpy.zeros(3)})

    plan = launch_module.plan_step(fields, object())      # must not raise

    assert plan.selected["step_B"] == "PML"


@pytest.mark.parametrize("attribute", [
    "has_nonlinearity", "has_offdiagonal_epsilon", "force_complex_fields",
])
def test_an_unreadable_fields_gate_consults_its_arm_instead_of_aborting(
        monkeypatch, attribute):
    import numpy

    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    fields = make_fields(**{attribute: numpy.zeros(3)})

    plan = launch_module.plan_step(fields, object())      # must not raise

    assert plan.selected["step_B"] == "PML"


class RaisingAttributeProxy:
    """A fields proxy whose ONE named attribute raises when it is READ.

    Not the same hostility as ``make_fields(x=numpy.zeros(3))``: that one returns
    an object whose truth value raises, and every gate here already survived it.
    This one raises out of the attribute ACCESS, which is what a property with a
    side effect, a ``__getattr__`` over a closed handle, or a lazily materialised
    volume does — and which the three gates that hoisted their ``fields.grid``
    read out of their own ``try`` did not survive.
    """

    def __init__(self, inner, name):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, name):
        if name == object.__getattribute__(self, "_name"):
            raise RuntimeError(f"{name} unreadable")
        return getattr(object.__getattribute__(self, "_inner"), name)


@pytest.mark.parametrize("attribute", [
    "grid", "chi1inv_offdiagonal_for", "has_offdiagonal_epsilon",
    "force_complex_fields", "has_nonlinearity", "polarizations", "stores_E",
])
def test_an_attribute_that_raises_on_ACCESS_never_escapes_plan_step(
        monkeypatch, attribute):
    """"Never raises" covers the gate block and the veto, not only the arm table.

    Two shipped paths broke this contract on a real ``Grid``/``Fields``/``PML``
    triple, and both were reads hoisted out of a ``try`` that was already there
    for exactly this:

    * ``fields.grid`` — ``cylindrical_grid_active``, ``complex_storage_active``
      and ``beta_active`` each read it before their guard, so an unreadable grid
      took down the composition from the GATE BLOCK, before any predicate ran;
    * ``fields.chi1inv_offdiagonal_for`` — ``live_offdiagonal_rows`` read the
      accessor before its guard, and the veto is the one block in ``plan_step``
      with no ``_guarded_*`` wrapper around it, so it escaped from there.

    Neither is reachable from the engine's own ``Fields`` (``grid`` is a plain
    dataclass field), which is exactly why this is asserted rather than assumed:
    the gates, the probes and this suite all build doubles.
    """
    install_sentinel_builders(monkeypatch)
    fields, pml = real_objects()
    plant_offdiagonal_row(fields)

    plan = launch_module.plan_step(RaisingAttributeProxy(fields, attribute), pml)

    # The contract is the return, not the value: a plan, never an exception, and
    # never a slot claimed on evidence the composer could not read.
    assert isinstance(plan, launch_module.TritonStepPlan)
    for slot, arm in plan.selected.items():
        assert slot in plan.plans, (slot, arm)


def test_the_null_family_claims_its_slots_on_a_host_with_no_cupy(monkeypatch):
    """A MEANING CHANGE for ``replaces``, pinned where a reader will find it.

    Every kernel arm refuses a host whose array module is not CuPy at
    ``coverage._grid_reasons`` clause 1. The null family carries no backend
    clause, deliberately — a sub-step that returns before its first statement
    launches nothing, so it is correct on NumPy — so on a plain NumPy host with
    an inert layer ``plan.replaces`` is no longer empty.

    That is the true answer: those two sub-steps really are replaced, by a plan
    that does what the array path does. But ``bool(plan.replaces)`` stopped
    meaning "a Triton KERNEL covers something here", and ``plan.selected`` is the
    seam that still answers that question. Asserted here so the change is a
    recorded fact rather than something a consumer discovers.
    """
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    install_sentinel_builders(monkeypatch)
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
                xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=False)

    plan = launch_module.plan_step(fields, PML(grid=grid, thickness=0))

    assert grid.xp.__name__ == "numpy"       # no faked backend name here
    assert plan.replaces == ("update_H", "update_E")
    assert plan.selected == {"update_H": "no-PML null", "update_E": "no-PML null"}
    # ...and every KERNEL arm still refused, the backend among its reasons.
    for slot in ("step_B", "step_D"):
        assert slot not in plan.selected
        assert any("not cupy" in reason for reason in plan.reasons[slot]), (
            slot, plan.reasons[slot])


def test_an_unreadable_polarization_list_refuses_update_P_rather_than_raising(
        monkeypatch):
    import numpy

    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    fields = make_fields(**ORDINARY)
    fields.polarizations = numpy.zeros(3)

    plan = launch_module.plan_step(fields, object())      # must not raise

    assert "update_P" not in plan.plans
    assert plan.reasons["update_P"]


def test_a_state_whose_driven_raises_refuses_update_P_rather_than_raising(
        monkeypatch):
    """``driven()`` is engine code called from the composer's explanation path."""
    install_stubs(monkeypatch, admit=("PML", "ordinary"))

    class Hostile:
        def driven(self):
            raise RuntimeError("the susceptibility cannot say what it drives")

    fields = make_fields(**ORDINARY)
    fields.polarizations = (Hostile(),)
    monkeypatch.setattr(launch_module, "plan_ade_update_p",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr(launch_module, "ade_update_p_coverage",
                        lambda *a, **k: REFUSED, raising=False)

    plan = launch_module.plan_step(fields, object())      # must not raise

    assert "update_P" not in plan.plans
    assert plan.reasons["update_P"]


class RaisingAttribute:
    """A descriptor whose READ raises — the shape a hostile engine object takes."""

    def __init__(self, name):
        self._name = name

    def __get__(self, obj, owner=None):
        raise RuntimeError(f"{self._name} cannot be read")


class RaisingBfastGrid:
    bfast_active = RaisingAttribute("bfast_active")

    def __init__(self):
        self.has_symmetry = lambda: False
        self.is_mirrored = lambda axis: False
        self.cylindrical = False
        self.has_bloch = False
        self.beta = 0.0
        self.shape = (4, 4, 4)


class RaisingDriven:
    def driven(self):
        raise RuntimeError("the susceptibility cannot say what it drives")


def unreadable():
    """An attribute whose truth value RAISES — the everyday shape is an ndarray."""
    import numpy

    return numpy.zeros(3)


def degenerate_cases():
    """Every shape the fail-open audit made ``plan_step`` raise on, plus siblings."""
    array = unreadable
    return [
        ("fold grid with shape=None",
         lambda: SimpleNamespace(
             grid=SimpleNamespace(shape=None, has_symmetry=lambda: True,
                                  is_mirrored=lambda axis: axis == 1,
                                  cylindrical=False, has_bloch=False, beta=0.0,
                                  bfast_active=False),
             polarizations=()), {}),
        ("non-sequence sources under fuse",
         lambda: make_fields(**ORDINARY), dict(fuse=True, sources=object())),
        ("a polarization whose driven() raises",
         lambda: _with_polarizations(RaisingDriven()), {}),
        ("fuse_ade with a polarization whose driven() raises",
         lambda: _with_polarizations(RaisingDriven()), dict(fuse_ade=True)),
        ("grid.bfast_active raises on read",
         lambda: SimpleNamespace(grid=RaisingBfastGrid(), polarizations=()), {}),
        ("ndarray grid.cylindrical", lambda: make_fields(cylindrical=array()), {}),
        ("ndarray grid.has_bloch", lambda: make_fields(has_bloch=array()), {}),
        ("ndarray grid.bfast_active",
         lambda: make_fields(bfast_active=array()), {}),
        ("ndarray fields.has_nonlinearity",
         lambda: make_fields(has_nonlinearity=array()), {}),
        ("ndarray fields.has_offdiagonal_epsilon",
         lambda: make_fields(has_offdiagonal_epsilon=array()), {}),
        ("ndarray fields.force_complex_fields",
         lambda: make_fields(force_complex_fields=array()), {}),
        ("ndarray fields.polarizations",
         lambda: _with_polarizations(array(), raw=True), {}),
        ("a bare object with nothing on it", SimpleNamespace, {}),
        ("fuse=True on a plain double",
         lambda: make_fields(**ORDINARY), dict(fuse=True, sources=())),
    ]


def _with_polarizations(value, raw=False):
    fields = make_fields(**ORDINARY)
    fields.polarizations = value if raw else (value,)
    return fields


DEGENERATE_CASES = degenerate_cases()


@pytest.mark.parametrize("label,build,kwargs", DEGENERATE_CASES,
                         ids=[case[0] for case in DEGENERATE_CASES])
def test_plan_step_never_raises_on_a_degenerate_object(label, build, kwargs):
    """The contract, fuzzed. NO STUBS: real predicates, real builders, no Triton.

    Nine of these made the composer raise before this round's guards — from the
    gate block, the folded ghost-fill block, the fusion block and the
    polarization loop, none of which was inside a try/except. A crash where the
    array path was the right answer is strictly worse than a refusal.

    ``replaces == ()`` WAS PART OF THIS CONTRACT AND IS NOT ANY MORE, for a
    reason that is a verdict rather than a concession. Every case here passes
    ``pml=None``, and ``pml=None`` IS an inactive layer — ``stepping.update_H``
    really does return at :916-917 on it — so the null family covers ``update_H``
    on most of these doubles and is RIGHT to. Asserting an empty ``replaces``
    would now be asserting that the composer must fail to notice a sub-step that
    does nothing. What is still asserted, and is the whole of the original point:
    it does not raise, every reported reason is non-empty, and any slot it DID
    fill holds a null plan rather than a kernel — nothing on a degenerate object
    may reach a launch.
    """
    from meep_gpu.triton_kernels.no_pml_constitutive import NullConstitutivePlan

    plan = launch_module.plan_step(build(), None, **kwargs)

    assert plan.reasons, label
    for slot, reasons in plan.reasons.items():
        assert reasons, (label, slot)
    for slot in plan.replaces:
        assert slot in ("update_H", "update_E"), (label, slot)
        assert isinstance(plan.plans[slot], NullConstitutivePlan), (label, slot)


def test_a_folded_grid_whose_shape_is_unreadable_refuses_the_fill_slots(
        monkeypatch):
    """The folded ghost-fill block reads ``grid.shape`` through symmetry.py."""
    install_stubs(monkeypatch, admit=())
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: True)
    monkeypatch.setattr(launch_module, "mirror_ghost_fill_coverage",
                        lambda *a, **k: (_ for _ in ()).throw(
                            TypeError("grid.shape is None")), raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object())

    for slot in ("fill_B", "fill_D"):
        assert slot not in plan.plans
        assert any("raised" in reason for reason in plan.reasons[slot]), slot


# ---------------------------------------------------------------------------
# C15 — the fusion block may not claim a slot the arm table refused
# ---------------------------------------------------------------------------
#
# The fusion block is the ONLY place in plan_step that fills a STEP_ORDER slot
# without going through the arm table, and it used to write both slots
# unconditionally. That put a hole straight through the fail-closed contract at
# its own point of decision.

def test_an_ambiguous_curl_slot_is_not_claimed_by_the_fused_pair(monkeypatch):
    """Clause (b) has to survive fusion.

    Two admitting curl predicates leave ``step_B`` unselected. The fused pair
    then claimed it anyway — with the ORDINARY product, which is exactly the
    product the ambiguity refused to pick — and ``reasons`` and ``plans`` came
    back contradicting each other for the same slot.
    """
    install_stubs(monkeypatch, admit=("PML", "conductive PML", "ordinary"))
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *a, **k: "fused", raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object(),
                                   fuse=True, sources=())

    assert "step_B" not in plan.plans
    assert "step_B" not in plan.selected
    assert any("ambiguous" in reason for reason in plan.reasons["step_B"])
    assert not isinstance(plan.plans.get("update_H"), launch_module.NoopPlan)
    assert any("not selected by any arm" in reason
               for reason in plan.reasons["fused_pair_B"]), plan.reasons


def test_a_slot_whose_builder_refused_is_not_claimed_by_the_fused_pair(
        monkeypatch):
    """Clause (c) has to survive fusion too — the REACHABLE variant.

    ``plan_pml_curl`` re-runs its own predicate so it cannot return None after
    coverage admitted, but it CAN raise: ``_flat`` refuses a non-contiguous or
    non-float32 coefficient column, and a missing pml table is an AttributeError.
    ``_select_slot`` catches those and leaves the slot unselected — and the fused
    pair then claimed it.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "plan_pml_curl",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *a, **k: "fused", raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object(),
                                   fuse=True, sources=())

    for slot in ("step_B", "step_D"):
        assert slot not in plan.plans
        assert "builder refused" in plan.reasons[slot][0]
    for key in ("fused_pair_B", "fused_pair_D"):
        assert any("not selected by any arm" in reason
                   for reason in plan.reasons[key]), key


def test_a_fused_pair_may_not_substitute_a_product_another_arm_won(monkeypatch):
    """The pair implements the ORDINARY curl plus the ordinary constitutive term.

    A slot won by a different arm is a different numerical product, and swapping
    it for the fused one is over-covering dispatch by another route.
    """
    install_stubs(monkeypatch, admit=("conductive PML", "ordinary"))
    monkeypatch.setattr(launch_module, "fused_pair_coverage", lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *a, **k: "fused", raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object(),
                                   fuse=True, sources=())

    assert plan.selected["step_B"] == "conductive PML"
    assert plan.plans["step_B"] != "fused"
    assert any("conductive PML" in reason and "implements" in reason
               for reason in plan.reasons["fused_pair_B"]), plan.reasons


def test_the_dispersive_pair_still_fuses_the_configuration_its_gate_measured(
        monkeypatch):
    """THE POSITIVE CONTROL for the absorb rule: it must not narrow a gated case.

    ``probe_triton_dispersive_fused_pair`` runs ordinary PML grids carrying poles,
    where ``step_D`` is won by the ``PML`` arm and ``update_E`` by the
    ``dispersive`` one. That is exactly the pair the rule permits, and this test
    is what stops the rule from quietly disabling the product the gate certified.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary", "dispersive"))
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: ADMITTED if side == "H" else REFUSED)
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage",
                        lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_dispersive_fused_pair",
                        lambda *a, **k: "dispersive fused", raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object(),
                                   fuse=True, sources=())

    assert plan.plans["step_D"] == "dispersive fused"
    assert isinstance(plan.plans["update_E"], launch_module.NoopPlan)
    assert plan.selected["step_D"] == "dispersive fused pair"
    assert "fused_pair_dispersive_D" not in plan.reasons


def test_the_dispersive_pair_is_held_to_the_same_rule(monkeypatch):
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage",
                        lambda *a, **k: ADMITTED)
    monkeypatch.setattr(launch_module, "plan_dispersive_fused_pair",
                        lambda *a, **k: "dispersive fused", raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object(),
                                   fuse=True, sources=())

    # update_E was won by the ORDINARY arm, not the dispersive one.
    assert plan.selected["update_E"] == "ordinary"
    assert plan.plans["step_D"] != "dispersive fused"
    assert any("implements" in reason
               for reason in plan.reasons["fused_pair_dispersive_D"]), plan.reasons


# ---------------------------------------------------------------------------
# C16 — verdict hygiene: what a predicate RETURNS is checked, not assumed
# ---------------------------------------------------------------------------

def test_a_predicate_returning_a_non_verdict_refuses_instead_of_crashing(
        monkeypatch):
    """``verdict.covered`` used to be dereferenced outside the try/except.

    A predicate that RETURNS something wrong is the same class of event as one
    that raises, and gets the same treatment: a named refusal.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "pml_curl_coverage",
                        lambda *a, **k: True, raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object())

    assert "step_B" not in plan.plans
    assert any("not a coverage verdict" in reason
               for reason in plan.reasons["step_B"]), plan.reasons["step_B"]


def test_a_predicate_returning_none_is_a_named_refusal_not_a_silent_skip(
        monkeypatch):
    """``None`` used to be indistinguishable from "the gate was closed".

    The arm vanished from ``plan.reasons`` entirely — a consulted product whose
    refusal left no trace, which is the one thing the refusal record exists to
    prevent.
    """
    install_stubs(monkeypatch, admit=("ordinary",))
    monkeypatch.setattr(launch_module, "pml_curl_coverage",
                        lambda *a, **k: None, raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object())

    assert any("PML predicate returned None" in reason
               for reason in plan.reasons["step_B"]), plan.reasons["step_B"]


def test_an_unselected_slot_never_reports_a_falsy_reason_tuple(monkeypatch):
    """A caller testing ``if plan.reasons.get(slot)`` must not read () as fine."""
    install_stubs(monkeypatch, admit=())
    for predicate in ("pml_curl_coverage", "conductive_pml_curl_coverage",
                      "plain_curl_coverage", "constitutive_coverage",
                      "dispersive_constitutive_coverage"):
        monkeypatch.setattr(launch_module, predicate,
                            lambda *a, **k: coverage_module.Coverage(False, ()),
                            raising=False)

    plan = launch_module.plan_step(make_fields(**ORDINARY), object())

    for slot in ("step_B", "step_D", "update_H", "update_E"):
        assert plan.reasons[slot], slot


# ---------------------------------------------------------------------------
# C17 — the update_P slot: the interface, and the whole refusal record
# ---------------------------------------------------------------------------

def test_the_update_P_slot_is_the_one_list_and_every_other_slot_answers_run(
        monkeypatch):
    """The slot interface, stated as the two halves it actually has.

    ``NoopPlan``'s docstring used to say "the substitution wrapper calls
    ``run()`` on whatever is in the slot" without qualification, while
    ``update_P`` — reported by ``replaces`` beside slots that DO answer it — held
    a bare ``list``. An adapter that applied the rule uniformly would raise
    ``AttributeError`` on that one slot.

    The list is the CONTRACT, not the defect: the driver advances the whole
    polarization list in one pass, each entry needs ``fields.drive_field`` passed
    to it, and every installer in the tree already branches on ``update_P`` to do
    exactly that. Two device gates pin the slot's type NAME as ``"list"``, so
    changing the value would break them on a CUDA host. What was wrong was the
    documentation, and this test is what stops either half from drifting again.
    """
    ran = []
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(
        launch_module, "plan_ade_update_p",
        lambda fields, state, block=None: SimpleNamespace(
            run=lambda *a, **k: ran.append(state)),
        raising=False)

    fields = make_fields(**ORDINARY)
    fields.polarizations = (SimpleNamespace(driven=lambda: ("Ez",)),
                            SimpleNamespace(driven=lambda: ("Ex",)))

    plan = launch_module.plan_step(fields, object())

    assert "update_P" in plan.replaces
    slot = plan.plans["update_P"]
    assert type(slot) is list, type(slot)            # the pinned device-gate type
    assert slot == list(plan.polarization_plans)
    for entry in slot:                               # each entry answers run(drive)
        entry.run(object())
    assert len(ran) == 2
    # The other half: every product class a STEP_ORDER slot can hold answers run().
    for plan_class in (launch_module.PmlCurlPlan, launch_module.ConstitutivePlan,
                       launch_module.FusedPairPlan, launch_module.NoopPlan,
                       launch_module.AdeUpdatePPlan):
        assert callable(getattr(plan_class, "run", None)), plan_class.__name__

    source = inspect.getsource(launch_module.NoopPlan)
    assert "update_P" in source, (
        "NoopPlan documents the slot interface; the one exception must be named "
        "there or the next adapter will apply the rule uniformly again")


def test_every_refusing_polarization_is_explained_not_only_the_first(monkeypatch):
    """The whole sub-step falls to the array path, so the whole list is explained.

    The loop used to ``break`` on the first refusal, so ``reasons['update_P']``
    documented state 0 and nothing else — sending a reader looking in the wrong
    place for states 1 and 2.
    """
    install_stubs(monkeypatch, admit=("PML", "ordinary"))
    monkeypatch.setattr(launch_module, "plan_ade_update_p",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        launch_module, "ade_update_p_coverage",
        lambda fields, state, component: coverage_module.Coverage(
            False, (f"{state.name} refused {component}",)),
        raising=False)

    fields = make_fields(**ORDINARY)
    fields.polarizations = tuple(
        SimpleNamespace(name=f"state{index}", driven=lambda: ("Ez",))
        for index in range(3))

    plan = launch_module.plan_step(fields, object())

    assert "update_P" not in plan.plans
    joined = " | ".join(plan.reasons["update_P"])
    for index in range(3):
        assert f"polarization {index} Ez: state{index} refused Ez" in joined, joined


# ---------------------------------------------------------------------------
# C18 — the package wrapper and launch.py must build the SAME configuration
# ---------------------------------------------------------------------------

def test_the_package_entry_points_inherit_the_measured_warp_default():
    """``num_warps=None`` is not the same value as the measured default.

    ``None`` means "let Triton choose" — 4 warps for a 1-D BLOCK=256 program,
    which launch.py records as a 16% throughput loss on the pair — while
    ``launch``'s own default is 1. The package wrappers defaulted to ``None``, so
    the same call through ``meep_gpu.triton_kernels`` and through ``.launch``
    built two different launch configurations from identical inputs, and every
    gate and probe imports these names from the package.
    """
    package = importlib.import_module("meep_gpu.triton_kernels")

    for builder in (package.plan_step, package.plan_fused_pair,
                    package.plan_dispersive_fused_pair):
        default = inspect.signature(builder).parameters["num_warps"].default
        assert default is not None, builder.__name__
        assert default is package._INHERIT_LAUNCH_DEFAULT, builder.__name__
    assert package._num_warps(package._INHERIT_LAUNCH_DEFAULT) == (
        launch_module.FUSED_DEFAULT_NUM_WARPS) == 1
    # An explicitly passed None must still mean Triton's own default.
    assert package._num_warps(None) is None
    assert package._num_warps(8) == 8


def test_both_routes_reach_the_pair_builder_with_the_same_warp_count(monkeypatch):
    """The divergence, measured where it mattered: at the builder call itself.

    Signature defaults are the mechanism; what the fused pair is BUILT with is
    the fact. Before this fix the same call reached ``plan_fused_pair`` as
    ``num_warps=1`` through ``.launch`` and ``num_warps=None`` through the
    package — and every gate and probe imports ``plan_step`` from the package.
    """
    package = importlib.import_module("meep_gpu.triton_kernels")
    install_sentinel_builders(monkeypatch)
    fields, pml = real_objects()

    seen = []
    monkeypatch.setattr(
        launch_module, "plan_fused_pair",
        lambda f, p, pair="B", sources=None, block=None, num_warps=None:
            (seen.append((pair, num_warps)), "<pair>")[1],
        raising=False)

    launch_module.plan_step(fields, pml, fuse=True, sources=())
    through_launch = list(seen)
    seen.clear()
    package.plan_step(fields, pml, fuse=True, sources=())

    assert through_launch == [("B", 1), ("D", 1)], through_launch
    assert seen == through_launch, seen


# ---------------------------------------------------------------------------
# C19 — the host drift is DECLARED, not silently carried
# ---------------------------------------------------------------------------

def test_every_drifted_host_file_is_named_in_the_pending_recut_record():
    """A red weld must be EXPLAINED by the record it contradicts.

    Every digest under ``host_sha256`` names bytes that ran on a GPU, so none of
    them may be rewritten from a laptop: a digest updated without a run turns a
    measurement into an assertion. The two weld tests therefore stay red on
    purpose — and this is what stops "expected red" from decaying into "red, and
    nobody remembers why".

    The check is exact in both directions: a file that drifts without being
    declared fails here, and a declaration that outlives its drift — or whose
    recorded live digest goes stale against the file — fails here too.
    """
    import hashlib
    import json

    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    certified = record["host_sha256"]
    declared = record["pending_host_recut"]["files"]

    drifted = {}
    for name, digest in certified.items():
        path = PACKAGE_DIR / name
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if live == digest:
            continue
        # ONE HOME FOR THE RULE (device_identity.py:209), and the SAME call the
        # weld test in test_triton_kernels makes. That is the point: this test
        # exists to explain that test's red, so the two must mean the same thing
        # by "drifted". An edit the shared rule clears is not a red weld and owes
        # no declaration here; anything it cannot establish is drift, unchanged.
        if weld_survives_edit(path, record, name):
            continue
        drifted[name] = live

    assert set(declared) == set(drifted), (
        f"declared={sorted(declared)} drifted={sorted(drifted)}")
    for name, live in drifted.items():
        assert declared[name]["live"] == live, name
        assert declared[name]["certified"] == certified[name], name
    for name in declared:
        assert record["pending_host_recut"]["what_clears_it"], name


def test_the_pending_recut_record_says_what_bounds_dispatch_now():
    """The claim that bounds the drift's blast radius, pinned to the shipped file.

    It used to be "dispatch is not wired at all". The driver-seam round wired it,
    so the bound moved to the ENABLE, and the record said "WIRED BUT OFF BY
    DEFAULT". That stopped being true when ``DISPATCH_BY_DEFAULT`` turned on, so
    the claim is now bound to the constant it restates: the ledger and the code
    cannot disagree about the default without this failing.
    """
    import json

    from meep_gpu import fastpath

    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    claim = record["pending_host_recut"]["dispatch"]
    expected = ("WIRED AND ON BY DEFAULT" if fastpath.DISPATCH_BY_DEFAULT
                else "WIRED BUT OFF BY DEFAULT")
    assert claim.startswith(expected), (
        f"fingerprints.json['pending_host_recut']['dispatch'] reads {claim[:48]!r} "
        f"but fastpath.DISPATCH_BY_DEFAULT is {fastpath.DISPATCH_BY_DEFAULT}; "
        f"rewrite the claim to start {expected!r} when the ledger is next cut")


# ---------------------------------------------------------------------------
# Dispatch is wired and on by default — and what bounds it is the certified set
# ---------------------------------------------------------------------------

def test_dispatch_is_wired_and_on_by_default_with_the_veto_above_it(monkeypatch):
    """``plan_fast_path`` returns plans, and an unset enable no longer holds it shut.

    The claim this replaced — "returns None on every branch" — was the whole of
    the old safety argument, and it was retired when dispatch was wired; its
    successor, "a run that asks for nothing gets nothing", was retired on
    2026-09-27, when ``DISPATCH_BY_DEFAULT`` turned on under the end-to-end licence
    ``test_dispatch_contract`` binds it to. What bounds a default run now is the
    certified set, which the rungs BELOW the enable check, so an unset enable must
    pass rung 2 and be answered lower down. The kill switch keeps its own
    precedence above all of it.

    Asserted on BEHAVIOUR against a stub CuPy backend rather than on the source
    text, because "the source contains no return" stopped being the property that
    matters the moment a return appeared. WHICH lower rung refuses the stub varies
    by host (no Triton here, an uncertified one elsewhere), so this pins only that
    the enable did not.
    """
    import types

    from meep_gpu import fastpath

    assert fastpath.DISPATCH_BY_DEFAULT is True, (
        "the default is bound to fingerprints.json['driver_dispatch']"
        "['dispatch_by_default_licence']; see test_dispatch_contract")

    xp = types.SimpleNamespace()
    xp.__name__ = "cupy"
    grid = types.SimpleNamespace(xp=xp)
    # Through monkeypatch, not os.environ directly: a developer who exported the
    # enable in their shell must get their value back, or this test would quietly
    # change what every test collected after it measures.
    monkeypatch.delenv(fastpath.DISPATCH_ENABLE, raising=False)
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    record = fastpath.last_dispatch_report()
    assert record["enable"]["value"] is None, record["enable"]
    assert record["enable"]["effective"] is True, record["enable"]
    reason = record["refused_because"] or ""
    assert fastpath.DISPATCH_ENABLE not in reason, reason
    assert "dispatch is opt-in" not in reason, reason
    assert record["environment"].get("not_read") not in (
        "refused at the enable", "refused at an unrecognised enable value",
        "refused at a retired switch name"), record["environment"]

    # And the veto still outranks the enable, from a cold read of the environment.
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    assert "kill switch" in fastpath.last_dispatch_report()["refused_because"]


# ---------------------------------------------------------------------------
# C20 — THE DISJOINTNESS SWEEP: at most one admitter per slot, on real objects
# ---------------------------------------------------------------------------
#
# The claim the whole fifteen-arm table rests on is that no two arms admit the
# same slot on any configuration a real engine can build. That is a MEASUREMENT,
# and a measurement recorded only in a document decays. So it lives here, as 69
# real ``Grid``/``Fields``/``PML`` triples driven through the SHIPPED composer
# with every builder replaced by a sentinel — the same seam the C13 block uses.
# It runs in about a fifth of a second on the merge-bar host with no GPU and no
# Triton.
#
# WHY THE ADMITTER COUNT IS OBSERVABLE THROUGH ``plan_step``: a slot with two or
# more admitters is left UNSELECTED with an "ambiguous" reason naming every one
# of them (clause (b)). So "exactly one admitter" is "selected", "none" is
# "unselected with named refusals", and "two or more" is "unselected AND
# ambiguous". Asserting the full ``selected`` dict per configuration therefore
# pins the disjointness AND the routing at once, and it cannot drift from
# launch.py's arm table the way a second copy of that table in this file would.
#
# THE BACKEND CLAUSE IS NEUTRALISED, not ignored: ``grid.xp`` answers to the name
# "cupy" (``NamedAsCupy``) so every predicate's clause 1 passes and the clauses
# that follow are evaluated against real arrays. On a device every run is CuPy
# and that clause distinguishes nothing, so passing it here is what makes the
# laptop result the same result — and a strictly stronger one than filtering the
# clause out of the reason lists would be.

def _sweep_probe():
    """The complex-multiply expansion record, covering every family's patterns."""
    from meep_gpu.triton_kernels import (
        complex_ade,
        complex_fields,
        folded_complex,
        special_kz,
    )

    from meep_gpu.test_triton_complex_fields import stamp_probe_record

    return stamp_probe_record({
        "backend": "cupy",
        "patterns": {name: "FMA_V1" for name in sorted(
            set(complex_fields.PROBE_PATTERNS)
            | set(complex_ade.COMPLEX_ADE_PROBE_PATTERNS)
            | set(folded_complex.PARITY_PROBE_PATTERNS)
            | set(folded_complex.FOLDED_BETA_PROBE_PATTERNS)
            | set(special_kz.BETA_PROBE_PATTERNS))},
    })


def _sweep_fill(fields, seed=5):
    import numpy

    rng = numpy.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.4, 0.4, array.shape).astype(array.dtype)
    return fields


def _sweep_eps(fields, rows=None, seed=7):
    import numpy

    shape = tuple(fields.grid.shape)
    rng = numpy.random.default_rng(seed)
    names = ("Ex", "Ey", "Ez")
    epsilon = {n: numpy.full(shape, v, numpy.float32)
               for n, v in zip(names, (2.0, 2.5, 3.0))}
    inverse = {n: numpy.full(shape, 1.0 / v, numpy.float32)
               for n, v in zip(names, (2.0, 2.5, 3.0))}
    if rows is None:
        fields.set_epsilon_volumes(epsilon, inverse)
    else:
        built = {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(numpy.float32)
                       for partner in partners}
                 for row, partners in rows.items()}
        fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=built)
    return fields


def _sweep_pol(components=("Ez",), kind="lorentzian"):
    names = tuple(components)
    return SimpleNamespace(driven=lambda: names,
                           drives=lambda name: name in names,
                           susceptibility=SimpleNamespace(kind=kind))


def cart(cell=(2.0, 2.1, 1.2), pml=2, complex_storage=False, storage=True,
         eps=True, rows=None, **grid_kwargs):
    """An ordinary Cartesian triple."""
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=cell, courant=0.35, xp=numpy,
                **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if eps:
        _sweep_eps(fields, rows)
    if storage:
        fields.enable_pml_storage()
    return _sweep_fill(fields), PML(grid=grid, thickness=pml)


def complex_no_pml():
    """A stored-complex, inert-layer triple with no PML storage.

    This is group J's curl configuration in the central real-object sweep.
    ``enable_field_storage`` is intentional: the complex no-PML binding needs
    stored E, while PML storage under an inert layer is a separate refusal.
    """
    fields, pml = cart(pml=0, complex_storage=True, storage=False)
    fields.enable_field_storage()
    return fields, pml


def complex_no_pml_offdiag():
    """Group J's complex stored-E row product with an inert absorber."""
    fields, pml = cart(
        pml=0, complex_storage=True, storage=False,
        rows={"Ex": ("Ey",), "Ey": ("Ez",), "Ez": ("Ex",)})
    fields.enable_field_storage()
    return fields, pml


def fold(axis="Y", phase=1, extent=2.0, other=1.6, dimensions=2, boundaries=None,
         complex_storage=True, beta=0.0, k_point=(0.0, 0.0, 0.0), rows=None,
         eps=True, pml_zero=False, **grid_kwargs):
    """A folded triple: one mirror axis, PML only on the un-mirrored face."""
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML

    size = [other, other, 0.0] if dimensions == 2 else [other, other, other]
    size["XYZ".index(axis)] = extent
    declared = boundaries
    if dimensions == 2 and boundaries not in (None, "periodic"):
        declared = {axis.lower(): boundaries}
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=dimensions,
                courant=0.35, boundaries=declared,
                symmetry=(Mirror(axis, phase),), k_point=k_point, beta=beta,
                xp=numpy, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if eps:
        _sweep_eps(fields, rows)
    fields.enable_pml_storage()
    if pml_zero:
        return _sweep_fill(fields), PML(grid=grid, thickness=0)
    thickness = []
    for index in range(3):
        if grid.shape[index] < 6:
            thickness.append((0, 0))
        elif index == "XYZ".index(axis):
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    return _sweep_fill(fields), PML(grid=grid, thickness=tuple(thickness))


def fold_two(complex_storage=True, phases=(1, -1)):
    """Two mirror axes — the shape whose K2 fill passes may not be fused."""
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid, Mirror
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
                symmetry=(Mirror("X", phases[0]), Mirror("Y", phases[1])),
                xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    _sweep_eps(fields)
    fields.enable_pml_storage()
    return _sweep_fill(fields), PML(grid=grid, thickness=((0, 2), (0, 2), (2, 2)))


def dcyl(shape=(16, 1, 20), m=-1, accurate=False, complex_storage=True,
         z_metallic=True, pml_cells=None, eps=True):
    """A cylindrical triple at the given azimuthal order."""
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=int(m),
                boundaries=({"z": "metallic"} if z_metallic else None),
                accurate_fields_near_cylorigin=bool(accurate),
                courant=0.37, xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if eps:
        _sweep_eps(fields)
    fields.enable_pml_storage()
    cells = {"x": (0, 4), "z": 4} if pml_cells is None else pml_cells
    return _sweep_fill(fields), PML(grid=grid, thickness=cells)


def conductive(pair, value=0.3, magnetic=False):
    import numpy

    fields, pml = pair
    shape = tuple(fields.grid.shape)
    volumes = {name: numpy.full(shape, numpy.float32(value))
               for name in (("Bx", "By", "Bz") if magnetic else ("Dx", "Dy", "Dz"))}
    if magnetic:
        fields.set_b_conductivity(volumes)
    else:
        fields.set_d_conductivity(volumes)
    return fields, pml


def with_pol(pair, components=("Ez",)):
    fields, pml = pair
    fields.polarizations.append(_sweep_pol(components))
    return fields, pml


def with_real_pole(pair, sigmas=(0.0, 0.0, 0.3), kind="lorentzian", seed=23):
    """Append a REAL ``PolarizationState`` — buffers, coefficients and sigma.

    ``_sweep_pol`` is a three-attribute double: enough for every clause that asks
    what a state DRIVES, and not enough for any predicate that goes on to read
    ``P``/``P_prev``/``_scratch``/``sigma``. Both ADE predicates and the folded
    dispersive one do, so on that double they refuse for "P[Ez] is not allocated"
    — a refusal that says nothing about the configuration under test and would
    make every row below vacuous in the admitting direction.
    """
    import numpy

    from meep_gpu.dispersion import PolarizationState, Susceptibility

    fields, pml = pair
    grid = fields.grid
    term = Susceptibility(frequency=1.0, gamma=0.1, kind=kind)
    state = PolarizationState(term, dict(zip(("Ex", "Ey", "Ez"), sigmas)),
                              grid, numpy.float32)
    fields.polarizations.append(state)
    rng = numpy.random.default_rng(seed)
    for component in state.driven():
        for slot in ("P", "P_prev"):
            getattr(state, slot)[component][...] = rng.uniform(
                -0.2, 0.2, size=grid.shape).astype(numpy.float32)
    return fields, pml


def absorber_free_dispersive(conductive=False):
    """The ``mp.Absorber`` class: stored E, NO split-field layer, no ``f_w``.

    Not ``cart(pml=0)``: that one either switches PML STORAGE on (which the
    inert-ADE predicate refuses by name — ``drive_field`` would hand back a
    frozen ``f_w``) or leaves E unstored (which it also refuses, because
    ``update_E`` then returns at stepping.py:984 and the drive is never written).
    The admitted class is the third one, ``enable_field_storage``, and it has to
    be built deliberately.
    """
    import numpy

    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), courant=0.35, xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_field_storage()
    epsilon = numpy.full(grid.shape, 2.25, dtype=numpy.float32)
    names = ("Ex", "Ey", "Ez")
    fields.set_epsilon_volumes({n: epsilon for n in names},
                               {n: (1.0 / epsilon).astype(numpy.float32)
                                for n in names})
    if conductive:
        # What mp.Absorber actually installs. Measured as making no difference to
        # this sub-step (0 of 7680 words over the closure round's cycles), so it
        # is here to keep that admission non-vacuous rather than to change it.
        fields.set_d_conductivity(numpy.full(grid.shape, 0.2, numpy.float32))
        fields.set_b_conductivity(numpy.full(grid.shape, 0.2, numpy.float32))
    return with_real_pole((_sweep_fill(fields), PML(grid=grid, thickness=0)))


def absorber_free_complex_dispersive():
    """Group I's complex stored-E Lorentz pass with conductive curls."""
    import numpy

    from meep_gpu.dispersion import PolarizationState, Susceptibility

    fields, pml = complex_no_pml()
    grid = fields.grid
    sigma = {"Ex": 0.2, "Ey": 0.0, "Ez": 0.3}
    state = PolarizationState(
        Susceptibility(frequency=1.0, gamma=0.1, kind="lorentzian"),
        sigma, grid, numpy.complex64)
    fields.polarizations.append(state)
    rng = numpy.random.default_rng(37)
    for component in state.driven():
        for slot in (state.P, state.P_prev):
            values = (rng.uniform(-0.2, 0.2, grid.shape)
                      + 1j * rng.uniform(-0.2, 0.2, grid.shape))
            slot[component][...] = values.astype(numpy.complex64)
    conductivity = numpy.full(grid.shape, 0.2, numpy.float32)
    fields.set_d_conductivity(conductivity)
    fields.set_b_conductivity(conductivity)
    return fields, pml


def nonlinear(pair):
    import numpy

    fields, pml = pair
    shape = tuple(fields.grid.shape)
    fields.set_nonlinear_volumes({"Ez": numpy.full(shape, numpy.float32(0.1))},
                                 {"Ez": numpy.full(shape, numpy.float32(0.2))})
    return fields, pml


def plant_rows(pair):
    """A live row SLOT with ``has_offdiagonal_epsilon`` still reporting False.

    ``launch.offdiag_rows_possible``'s docstring records this direction as
    UNREACHABLE against the engine's own class and reachable only on a double —
    which this is. It is planted because it is the one class that must still
    DOUBLE-ADMIT and fail closed, and the folded copy of it is what proves the
    new folded arm reproduces that shape rather than adding a new one.
    """
    import numpy

    fields, pml = pair
    row = numpy.full(tuple(fields.grid.shape), numpy.float32(0.1))

    class FlagFalse:
        def __init__(self, wrapped):
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, name):
            if name == "has_offdiagonal_epsilon":
                return False
            if name == "chi1inv_offdiagonal_for":
                return lambda target: {"Ey": row} if target == "Ex" else {}
            return getattr(object.__getattribute__(self, "_wrapped"), name)

    return FlagFalse(fields), pml


def flag_only(pair):
    """``has_offdiagonal_epsilon`` True with every ROW SLOT dead — the reachable one."""
    fields, pml = pair

    class FlagTrue:
        def __init__(self, wrapped):
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, name):
            if name == "has_offdiagonal_epsilon":
                return True
            if name == "chi1inv_offdiagonal_for":
                return lambda target: {}
            return getattr(object.__getattribute__(self, "_wrapped"), name)

    return FlagTrue(fields), pml


#: Configurations with the EXACT ``selected`` dict the shipped composer
#: produces and the slots it leaves ambiguous. Recorded from a run of this same
#: sweep, and checked against the hand-evaluated design measurement. The test
#: derives the current counts rather than duplicating them here.
#:
#: An empty dict is a legitimate and expected answer: it means every arm refused
#: and the whole step is on the array path, which is always correct.
SWEEP = (
    ("cart_pml_real", lambda: cart(),
     dict(step_B="PML", step_D="PML", update_H="ordinary", update_E="ordinary"), ()),
    ("cart_pml_real_metallic", lambda: cart(boundaries="metallic"),
     dict(step_B="PML", step_D="PML", update_H="ordinary", update_E="ordinary"), ()),
    ("cart_pml_complex_forced", lambda: cart(complex_storage=True),
     dict(step_B="complex PML", step_D="complex PML",
          update_H="complex", update_E="complex"), ()),
    ("cart_pml_bloch", lambda: cart(k_point=(0.3, 0.0, 0.0), complex_storage=False),
     {}, ()),
    ("cart_pml_bloch_complex",
     lambda: cart(k_point=(0.3, 0.0, 0.0), complex_storage=True),
     dict(step_B="complex PML", step_D="complex PML",
          update_H="complex", update_E="complex"), ()),

    ("nopml_real_nostore", lambda: cart(pml=0, storage=False, eps=False),
     dict(step_B="no-PML", step_D="no-PML",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_real_stored", lambda: cart(pml=0, storage=True, eps=False), {}, ()),
    ("nopml_real_eps_nostore", lambda: cart(pml=0, storage=False),
     dict(step_B="no-PML", step_D="no-PML",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_none_layer",
     lambda: (cart(pml=0, storage=False, eps=False)[0], None),
     dict(step_B="no-PML", step_D="no-PML",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_complex_nostore",
     lambda: cart(pml=0, storage=False, eps=False, complex_storage=True),
     dict(update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_complex_stored", complex_no_pml,
     dict(step_B="complex no-PML curl", step_D="complex no-PML curl",
          update_H="no-PML null"), ()),
    ("nopml_complex_offdiag", complex_no_pml_offdiag,
     dict(step_B="complex no-PML curl", step_D="complex no-PML curl",
          update_H="no-PML null",
          update_E="complex no-PML off-diagonal"), ()),
    ("nopml_fold_complex", lambda: fold(pml_zero=True, complex_storage=True),
     dict(fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    ("nopml_fold_real", lambda: fold(pml_zero=True, complex_storage=False),
     dict(fill_B="mirror fill", fill_D="mirror fill"), ()),
    ("nopml_bfast", lambda: cart(pml=0, storage=False, eps=False,
                                 bfast_scaled_k=(0.2, 0.0, 0.0)),
     dict(update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_magnetic_conductive",
     lambda: conductive(cart(pml=0, storage=False, eps=False), magnetic=True),
     dict(step_B="conductive no-PML", step_D="conductive no-PML",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_cyl", lambda: dcyl(pml_cells=0, eps=False), {}, ()),
    ("nopml_conductive",
     lambda: conductive(cart(pml=0, storage=False, eps=False)),
     dict(step_B="conductive no-PML", step_D="conductive no-PML",
          update_H="no-PML null", update_E="no-PML null"), ()),
    ("nopml_offdiag",
     lambda: cart(pml=0, storage=False, rows={"Ex": ("Ey",)}),
     dict(step_B="no-PML", step_D="no-PML", update_H="no-PML null"), ()),
    ("nopml_pol_driven",
     lambda: with_pol(cart(pml=0, storage=False, eps=False)),
     dict(step_B="no-PML", step_D="no-PML", update_H="no-PML null",
          update_P="ADE update_P"), ()),
    ("nopml_pol_magnetic",
     lambda: with_pol(cart(pml=0, storage=False, eps=False), components=("Bx",)),
     dict(update_E="no-PML null", update_P="ADE update_P"), ()),
    ("nopml_nonlinear_flagged",
     lambda: nonlinear(cart(pml=0, storage=False, eps=False)),
     dict(update_H="no-PML null"), ()),

    ("fold_real_periodic", lambda: fold(complex_storage=False),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded", update_E="folded"), ()),
    ("fold_real_metallic",
     lambda: fold(complex_storage=False, boundaries="metallic"),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded", update_E="folded"), ()),
    ("fold_real_3d", lambda: fold(complex_storage=False, dimensions=3),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded", update_E="folded"), ()),
    ("fold_real_odd_phase", lambda: fold(complex_storage=False, phase=-1),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded", update_E="folded"), ()),
    ("fold_real_dispersive", lambda: with_pol(fold(complex_storage=False)),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded", update_P="ADE update_P"), ()),
    # Conductivity changes the CURL recurrence only, so the constitutive slots
    # stay covered while both curls are refused. Pre-existing and deliberate
    # (symmetry._folded_constitutive_grid_reasons omits the clause).
    ("fold_real_conductive", lambda: conductive(fold(complex_storage=False)),
     dict(fill_B="mirror fill", fill_D="mirror fill",
          update_H="folded", update_E="folded"), ()),
    ("fold_real_nonlinear", lambda: nonlinear(fold(complex_storage=False)),
     dict(fill_B="mirror fill", fill_D="mirror fill"), ()),

    ("fold_complex_forced", lambda: fold(complex_storage=True),
     dict(step_B="folded complex PML", step_D="folded complex PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex", update_E="folded complex"), ()),
    ("fold_complex_metallic",
     lambda: fold(complex_storage=True, boundaries="metallic"),
     dict(step_B="folded complex PML", step_D="folded complex PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex", update_E="folded complex"), ()),
    ("fold_complex_3d", lambda: fold(complex_storage=True, dimensions=3),
     dict(step_B="folded complex PML", step_D="folded complex PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex", update_E="folded complex"), ()),
    ("fold_complex_bloch_unfolded_axis",
     lambda: fold(axis="Y", complex_storage=True, k_point=(0.3, 0.0, 0.0)),
     dict(step_B="folded complex PML", step_D="folded complex PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex", update_E="folded complex"), ()),

    ("fold_real_beta", lambda: fold(complex_storage=False, beta=0.2),
     dict(step_B="folded real beta PML", step_D="folded real beta PML",
          fill_B="mirror fill", fill_D="mirror fill",
          update_H="folded beta run", update_E="folded beta run"), ()),
    # K3b takes the curls; the CONSTITUTIVE slots go to `folded complex`, which
    # carries no beta clause on purpose.
    ("fold_complex_beta", lambda: fold(complex_storage=True, beta=0.2),
     dict(step_B="folded complex beta PML", step_D="folded complex beta PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex", update_E="folded complex"), ()),
    ("fold_complex_beta_bloch",
     lambda: fold(complex_storage=True, beta=-0.685, k_point=(0.3, 0.0, 0.0)),
     dict(step_B="folded complex beta PML", step_D="folded complex beta PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex", update_E="folded complex"), ()),

    # THE TWO ROWS THE VETO EXEMPTION EXISTS FOR: `folded off-diagonal` wins
    # update_E and must SURVIVE `_veto_dropped_offdiagonal_coupling`.
    ("fold_real_offdiag_full",
     lambda: fold(complex_storage=False, dimensions=3,
                  rows={"Ex": ("Ey", "Ez"), "Ey": ("Ex", "Ez"),
                        "Ez": ("Ex", "Ey")}),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded",
          update_E="folded off-diagonal"), ()),
    ("fold_real_offdiag_one_slot",
     lambda: fold(complex_storage=False, dimensions=3, rows={"Ey": ("Ez",)}),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded",
          update_E="folded off-diagonal"), ()),
    # THE SCOPE CHANGE, MEASURED. This row used to read "UNCOVERED BY DESIGN:
    # folded_complex refuses off-diagonal ... so nothing admits either curl or
    # either constitutive slot", and three of those four refusals were over-broad:
    # the row product lives in update_E alone (stepping._offdiagonal_terms,
    # S:1190-1225, reached from S:972-979) and the certified bodies were measured
    # BYTE-IDENTICAL on the other three over eight complete cycles, on this shape
    # and on two row shapes. The Metal round measured the same nine slots
    # independently. So the three sub-steps are covered by the sibling arms and
    # update_E was the remaining 25041/110592-word divergence in that round; the
    # complex row-product family below is the new product that now owns it.
    ("fold_complex_offdiag",
     lambda: fold(complex_storage=True, dimensions=3, rows={"Ex": ("Ey",)}),
     dict(step_B="folded complex off-diagonal PML",
          step_D="folded complex off-diagonal PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex off-diagonal",
          update_E="complex folded off-diagonal"), ()),
    ("fold_real_offdiag_beta",
     lambda: fold(complex_storage=False, beta=0.2, rows={"Ex": ("Ey",)}),
     dict(fill_B="mirror fill", fill_D="mirror fill"), ()),

    ("beta_real_2d", lambda: cart(cell=(1.6, 1.6, 0.0), dimensions=2, beta=0.2,
                                  pml=((2, 2), (2, 2), (0, 0))),
     dict(step_B="real beta PML", step_D="real beta PML",
          update_H="real beta run", update_E="real beta run"), ()),
    ("beta_complex_2d", lambda: cart(cell=(1.6, 1.6, 0.0), dimensions=2, beta=0.2,
                                     complex_storage=True,
                                     pml=((2, 2), (2, 2), (0, 0))),
     dict(step_B="complex beta PML", step_D="complex beta PML",
          update_H="complex beta run", update_E="complex beta run"), ()),
    ("beta_complex_2d_bloch",
     lambda: cart(cell=(1.6, 1.6, 0.0), dimensions=2, beta=0.2,
                  complex_storage=True, k_point=(0.3, 0.0, 0.0),
                  pml=((2, 2), (2, 2), (0, 0))),
     dict(step_B="complex beta PML", step_D="complex beta PML",
          update_H="complex beta run", update_E="complex beta run"), ()),
    ("bfast_real", lambda: cart(bfast_scaled_k=(0.2, 0.0, 0.0)),
     dict(step_B="BFAST PML", step_D="BFAST PML",
          update_H="BFAST run", update_E="BFAST run"), ()),
    # A WHOLE STEP, and the only residual group that takes real-physics rows there
    # (3rd-harm-1d.py and Test3rdHarm1d.test_3rd_harm_1d, no TestLoadDump). The
    # Pade factor lives in update_E and the shipped clause refused it on all four
    # sub-steps; the curls difference the stored E and H and update_H is B/mu with
    # the integer coefficient pair, none of which reads chi2 or chi3.
    ("nonlinear_real", lambda: nonlinear(cart()),
     dict(step_B="nonlinear run PML", step_D="nonlinear run PML",
          update_H="nonlinear run", update_E="nonlinear"), ()),
    ("offdiag_real",
     lambda: cart(rows={"Ex": ("Ey", "Ez"), "Ey": ("Ex", "Ez"),
                        "Ez": ("Ex", "Ey")}),
     dict(step_B="PML", step_D="PML", update_H="ordinary",
          update_E="off-diagonal"), ()),
    ("offdiag_one_slot", lambda: cart(rows={"Ey": ("Ez",)}),
     dict(step_B="PML", step_D="PML", update_H="ordinary",
          update_E="off-diagonal"), ()),
    ("dispersive_real", lambda: with_pol(cart()),
     dict(step_B="PML", step_D="PML", update_H="ordinary",
          update_P="ADE update_P"), ()),
    ("dispersive_complex", lambda: with_pol(cart(complex_storage=True)),
     dict(update_P="ADE update_P"), ()),
    ("conductive_real", lambda: conductive(cart()),
     dict(step_B="PML", step_D="conductive PML", update_H="ordinary",
          update_E="ordinary"), ()),
    ("conductive_magnetic_real", lambda: conductive(cart(), magnetic=True),
     dict(step_B="conductive PML", step_D="PML", update_H="ordinary",
          update_E="ordinary"), ()),
    ("conductive_complex", lambda: conductive(cart(complex_storage=True)), {}, ()),
    ("offdiag_complex",
     lambda: cart(complex_storage=True, rows={"Ex": ("Ey",)}),
     dict(step_B="complex PML", step_D="complex PML", update_H="complex"), ()),
    ("nonlinear_complex", lambda: nonlinear(cart(complex_storage=True)), {}, ()),
    ("bfast_complex", lambda: cart(bfast_scaled_k=(0.2, 0.0, 0.0),
                                   complex_storage=True), {}, ()),
    ("fold_complex_dispersive", lambda: with_pol(fold(complex_storage=True)),
     dict(fill_B="folded complex fill", fill_D="folded complex fill",
          update_P="ADE update_P"), ()),
    ("fold_complex_conductive", lambda: conductive(fold(complex_storage=True)),
     dict(fill_B="folded complex fill", fill_D="folded complex fill"), ()),
    ("fold_two_axes_real", lambda: fold_two(complex_storage=False),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded", update_E="folded"), ()),
    ("fold_two_axes_complex", lambda: fold_two(complex_storage=True),
     dict(step_B="folded complex PML", step_D="folded complex PML",
          fill_B="folded complex fill", fill_D="folded complex fill",
          update_H="folded complex", update_E="folded complex"), ()),

    ("dcyl_m0_real", lambda: dcyl(m=0, complex_storage=False),
     dict(step_B="cylindrical PML", step_D="cylindrical PML",
          update_H="cylindrical", update_E="cylindrical"), ()),
    ("dcyl_m0_real_periodic_z",
     lambda: dcyl(m=0, complex_storage=False, z_metallic=False), {}, ()),
    ("dcyl_m1_complex", lambda: dcyl(m=1),
     dict(step_B="cylindrical complex PML", step_D="cylindrical complex PML",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m-1_complex", lambda: dcyl(m=-1),
     dict(step_B="cylindrical complex PML", step_D="cylindrical complex PML",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m2_complex_accurate", lambda: dcyl(m=2, accurate=True),
     dict(step_B="cylindrical complex PML", step_D="cylindrical complex PML",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m3_complex", lambda: dcyl(m=3),
     dict(step_B="cylindrical complex PML", step_D="cylindrical complex PML",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    # m = 0 UNDER COMPLEX STORAGE resolves to the cylindrical complex arms since
    # 2026-09-04 (cylindrical_complex.M_ZERO); the real-storage m = 0 triple above
    # still resolves to cylindrical_triton's, and the split is by storage.
    ("dcyl_m0_complex", lambda: dcyl(m=0, complex_storage=True),
     dict(step_B="cylindrical complex PML", step_D="cylindrical complex PML",
          update_H="cylindrical complex", update_E="cylindrical complex"), ()),
    ("dcyl_m1_nr1",
     lambda: dcyl(shape=(1, 1, 20), m=1, pml_cells={"x": (0, 0), "z": 4}),
     {}, ()),

    # THE TWO PLANTED DOUBLE-ADMITS. Both fail closed with the ambiguity naming
    # both claimants, and the folded one reproduces the unfolded shape exactly
    # rather than introducing a new one.
    ("offdiag_flag_false_slot_live", lambda: plant_rows(cart()),
     dict(step_B="PML", step_D="PML", update_H="ordinary"), ("update_E",)),
    ("fold_offdiag_flag_false_slot_live",
     lambda: plant_rows(fold(complex_storage=False, dimensions=3)),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded"), ("update_E",)),
    # The mirror case, which IS reachable on the engine: flag True, every slot
    # dead. Every arm refuses; no ambiguity to resolve.
    ("offdiag_flag_true_slots_dead", lambda: flag_only(cart()),
     dict(step_B="PML", step_D="PML", update_H="ordinary"), ()),
    ("fold_offdiag_flag_true_slots_dead",
     lambda: flag_only(fold(complex_storage=False, dimensions=3)),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded"), ()),

    # THE OTHER TWO RESIDUAL-GROUP ADMISSIONS, on doubles that carry REAL poles.
    # ``fold_real_dispersive`` above uses ``_sweep_pol``, whose state has no
    # buffers, so it cannot reach either family's real clauses — these rows are
    # what make those two arms non-vacuous in this sweep.
    ("fold_real_dispersive_poles",
     lambda: with_real_pole(fold(complex_storage=False, dimensions=3,
                                 extent=2.0, other=1.6)),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded",
          update_E="folded dispersive", update_P="ADE update_P"), ()),
    ("fold_real_offdiag_dispersive_poles",
     lambda: with_real_pole(fold(
         complex_storage=False, dimensions=3, extent=2.0, other=1.6,
         rows={"Ex": ("Ey", "Ez"), "Ey": ("Ex", "Ez"),
               "Ez": ("Ex", "Ey")})),
     dict(step_B="folded PML", step_D="folded PML", fill_B="mirror fill",
          fill_D="mirror fill", update_H="folded",
          update_E="folded off-diagonal dispersive",
          update_P="ADE update_P"), ()),
    # The absorber-free ADE now composes with Arm S's stored-E update.
    ("nopml_ade_inert_absorber", lambda: absorber_free_dispersive(),
     dict(step_B="no-PML", step_D="no-PML", update_H="no-PML null",
          update_E="no-PML stored E", update_P="no-PML ADE update_P"), ()),
    ("nopml_ade_inert_conductive",
     lambda: absorber_free_dispersive(conductive=True),
     dict(step_B="conductive no-PML", step_D="conductive no-PML",
          update_H="no-PML null", update_E="no-PML stored E",
          update_P="no-PML ADE update_P"), ()),
    ("nopml_complex_ade_inert_conductive",
     absorber_free_complex_dispersive,
     dict(step_B="complex conductive no-PML curl",
          step_D="complex conductive no-PML curl",
          update_H="no-PML null", update_E="complex no-PML stored E",
          update_P="complex ADE update_P"), ()),
)


@pytest.mark.parametrize("label,build,expected,ambiguous", SWEEP,
                         ids=[row[0] for row in SWEEP])
def test_every_slot_resolves_to_at_most_one_product_on_real_objects(
        monkeypatch, label, build, expected, ambiguous):
    """The disjointness measurement, as a test that cannot go stale.

    ``selected`` says a slot had EXACTLY ONE admitter and names it. A slot that
    is absent from ``selected`` and carries an "ambiguous" reason had TWO OR
    MORE. Any slot in ``ambiguous`` that is not declared here — or any declared
    one that stops being ambiguous — is a predicate defect and fails here.
    """
    import numpy

    install_sentinel_builders(monkeypatch)
    fields, pml = build()
    fields.grid.xp = NamedAsCupy(numpy)

    plan = launch_module.plan_step(fields, pml, probe=_sweep_probe())

    assert plan.selected == expected, (label, plan.reasons)
    overlapping = tuple(sorted(
        slot for slot, reasons in plan.reasons.items()
        if any("ambiguous" in reason for reason in reasons)))
    assert overlapping == tuple(ambiguous), (label, plan.reasons)
    for slot in ambiguous:
        reason, = plan.reasons[slot]
        assert reason.count(",") >= 1, reason


def test_the_planted_double_admits_name_both_claimants():
    """The two overlaps are PLANTED, and each names the pair it fails closed on.

    Neither is engine-reachable: ``has_offdiagonal_epsilon`` is a read-only
    property over the same dict ``chi1inv_offdiagonal_for`` reads, so the two
    move together. They are built here because the gates, the probes and this
    suite all construct ``Fields`` doubles, and any of them could produce the
    disagreement the gate deliberately keeps VISIBLE.
    """
    import numpy

    for build, pair in (
            (lambda: plant_rows(cart()), ("ordinary", "off-diagonal")),
            (lambda: plant_rows(fold(complex_storage=False, dimensions=3)),
             ("folded", "folded off-diagonal"))):
        fields, pml = build()
        fields.grid.xp = NamedAsCupy(numpy)
        plan = launch_module.plan_step(fields, pml, probe=_sweep_probe())
        reason, = plan.reasons["update_E"]
        for label in pair:
            assert label in reason, (pair, reason)
        assert "update_E" not in plan.plans


def test_every_arm_in_the_table_wins_at_least_one_configuration():
    """NON-VACUITY. A sweep where an arm never wins measures nothing about it.

    Read off the recorded expectations rather than re-run, so this stays cheap
    and still fails the moment an arm is added to ``launch.py`` without a
    configuration that exercises it.
    """
    winners = set()
    for _label, _build, expected, _ambiguous in SWEEP:
        winners.update(expected.values())

    for arm in (*CURL_PREDICATES, *FILL_PREDICATES, *CONSTITUTIVE_PREDICATES):
        if arm in ("dispersive", "conductive PML"):
            continue        # E-only pole product / per-axis conductivity, below
        assert arm in winners, arm
    assert "conductive PML" in winners
    # `dispersive` is the one arm no configuration here wins: on every
    # dispersive double the pole product's own predicate refuses on the laptop
    # host and update_E falls to the array path, which its OWN gate covers.
    assert "dispersive" not in winners
