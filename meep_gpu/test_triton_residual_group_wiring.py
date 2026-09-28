"""What the residual-group wiring closed, measured through the SHIPPED composer.

Four predicate families and six entry points were built and device-certified by
the residual-group closure round (2026-08-15) and then reached by nothing: its own
artifact says "plan_step gains no arm from this round, and the new predicates are
consulted by nothing". This file is the contract on the round that made them arms.

TWENTY-TWO SLOTS, and the accounting is a test rather than a sentence:

    group A   chi2/chi3 at step_B, step_D, update_H       2 rows x 3 =  6
    group C   fold + dispersion at update_E               4 rows x 1 =  4
    group D   complex + fold + off-diagonal at the three
              non-update_E sub-steps                      3 rows x 3 =  9
    group B/H ADE update_P with the absorber INERT        3 rows x 1 =  3
                                                          TOTAL       22

NO NEW KERNEL IS INVOLVED. Every one of these is a CERTIFIED body admitted onto a
configuration a shared clause refused it for, and the identity that licenses each
was measured on an RTX A6000 over eight complete cycles, non-vacuous on both sides
with a live signed-zero census, through these builders with these predicates
unpatched (results/residual_closure_2026-08-15/device/newpred/).

THE RISK THIS FILE IS AIMED AT IS DISJOINTNESS, NOT ARITHMETIC. Every family's
grid-reason list answers the same numbered questions and each carried family
inverts exactly one, so widening a clause can make TWO arms admit one slot — and
``_select_slot`` fails closed on that, leaving the slot UNSELECTED and on the
array path. That is a silent coverage LOSS, not an error, and it is the failure
mode the scope change here could plausibly have produced. So the scope change is
made by ADDING an arm, the incumbent clause is left exactly as it is, and the cost
of the alternative is MEASURED below rather than argued.

Everything runs on the merge-bar laptop with no GPU and no Triton: the predicates
are real, only the builders are sentinels, and ``grid.xp`` answers to the name the
shared backend clause looks for.
"""

from __future__ import annotations

import pytest

from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.test_triton_planner_composition import (
    NamedAsCupy,
    _sweep_probe,
    absorber_free_dispersive,
    cart,
    fold,
    install_sentinel_builders,
    nonlinear,
    with_real_pole,
)

# Every probe record built here is stamped 'keep'; the complex families' policy
# clause fails closed when the run policy can be neither read nor declared.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")


def family(name):
    """A family module resolved the way the SHIPPED lazy forwarders resolve it.

    ``launch``'s forwarders do ``from .folded_complex import ...``, which reads
    ``sys.modules``; ``from meep_gpu.triton_kernels import folded_complex`` reads
    the package ATTRIBUTE. They are the same object until some test pops a family
    out of ``sys.modules`` and lets an import inside that window rebind the
    attribute — ``monkeypatch.delitem`` restores the dict entry and nothing
    restores the attribute — after which a patch applied through the attribute is
    invisible to the predicate. ``importlib.import_module`` returns the
    ``sys.modules`` entry, so this resolves what the code under test will use.

    Found the hard way: ``test_narrowing_the_incumbent_clause_instead_would_have_
    cost_all_nine_slots`` passed alone and failed after the planner suite, because
    it patched a throwaway ``folded_complex``.
    """
    import importlib

    return importlib.import_module(f"meep_gpu.triton_kernels.{name}")


def folded_complex_offdiag():
    """Group (D)'s shape: a mirror fold, complex64 storage, a live row slot."""
    return fold(complex_storage=True, dimensions=3, rows={"Ex": ("Ey",)})


def folded_dispersive():
    """Group (C)'s shape: a real mirror fold with a REGISTERED pole."""
    return with_real_pole(fold(complex_storage=False, dimensions=3))


#: One entry per residual group: the arm labels the composer must select, and the
#: number of CORPUS ROWS that configuration stands for. The row counts are the
#: closure artifact's, quoted; the slot arithmetic is computed from them below.
CLOSED = (
    ("A_chi2_chi3", 2, lambda: nonlinear(cart()),
     {"step_B": "nonlinear run PML",
      "step_D": "nonlinear run PML",
      "update_H": "nonlinear run"}),
    ("C_fold_dispersion", 4, folded_dispersive,
     {"update_E": "folded dispersive"}),
    ("D_complex_fold_offdiag", 3, folded_complex_offdiag,
     {"step_B": "folded complex off-diagonal PML",
      "step_D": "folded complex off-diagonal PML",
      "update_H": "folded complex off-diagonal"}),
    ("BH_no_pml_ade", 3, absorber_free_dispersive,
     {"update_P": "no-PML ADE update_P"}),
)


def plan_for(build, monkeypatch, **kwargs):
    """``plan_step`` on real objects: real predicates, sentinel builders."""
    import numpy

    install_sentinel_builders(monkeypatch)
    fields, pml = build()
    fields.grid.xp = NamedAsCupy(numpy)
    return launch_module.plan_step(fields, pml, probe=_sweep_probe(), **kwargs)


def ambiguous_slots(plan):
    return tuple(sorted(slot for slot, reasons in plan.reasons.items()
                        if any("ambiguous" in reason for reason in reasons)))


# ---------------------------------------------------------------------------
# The admissions are REACHED, and each by exactly one arm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,_rows,build,expected",
                         CLOSED, ids=[row[0] for row in CLOSED])
def test_each_newly_wired_admission_is_selected_by_the_planner(
        monkeypatch, label, _rows, build, expected):
    """A certified predicate consulted by nothing closes no slot at all.

    ``plan.selected`` is the seam that says a slot had EXACTLY ONE admitter AND
    names which one — the distinction the plan CLASS cannot make, because three
    different arms build a ``launch.ConstitutivePlan`` and two build a
    ``PmlCurlPlan``. Asserting the arm label is what makes this a routing test
    rather than a decoration.
    """
    plan = plan_for(build, monkeypatch)
    for slot, arm in expected.items():
        assert plan.selected.get(slot) == arm, (
            label, slot, plan.selected.get(slot), plan.reasons.get(slot))
    assert ambiguous_slots(plan) == (), (label, plan.reasons)


def test_the_closed_slot_count_is_the_twenty_two_the_licence_paid_for():
    """The accounting, computed from the table rather than restated beside it.

    22 = 2x3 + 4x1 + 3x3 + 3x1. A group whose arm set shrinks — or a row count
    edited without the artifact behind it — changes this number, which is the
    point of computing it here instead of writing it in a comment.
    """
    per_group = {label: rows * len(expected)
                 for label, rows, _build, expected in CLOSED}
    assert per_group == {"A_chi2_chi3": 6, "C_fold_dispersion": 4,
                         "D_complex_fold_offdiag": 9, "BH_no_pml_ade": 3}
    assert sum(per_group.values()) == 22


@pytest.mark.parametrize("name", [
    "nonlinear_run_pml_curl_coverage", "plan_nonlinear_run_pml_curl",
    "nonlinear_run_constitutive_coverage", "plan_nonlinear_run_constitutive",
    "folded_dispersive_constitutive_coverage",
    "plan_folded_dispersive_constitutive",
    "folded_complex_offdiag_pml_curl_coverage",
    "plan_folded_complex_offdiag_pml_curl",
    "folded_complex_offdiag_constitutive_coverage",
    "plan_folded_complex_offdiag_constitutive",
    "no_pml_ade_update_p_coverage", "plan_no_pml_ade_update_p",
])
def test_every_entry_point_is_reachable_through_the_planner_and_the_package(name):
    """Both seams, because gates and probes import from the package, not ``launch``.

    A forwarder that exists on one and not the other is how the same call built
    two different configurations once before (``num_warps``), and it is also how
    a family reads as wired while every caller outside this package still cannot
    see it.
    """
    import importlib

    package = importlib.import_module("meep_gpu.triton_kernels")
    assert callable(getattr(launch_module, name)), name
    assert callable(getattr(package, name)), name
    assert name in package.__all__, name


def test_the_two_new_family_modules_are_declared_and_no_longer_deferred():
    """``FAMILY_MODULES`` is what the lazy-import seam is checked against."""
    assert "folded_dispersive_update_e" in launch_module.FAMILY_MODULES
    assert "no_pml_ade" in launch_module.FAMILY_MODULES
    assert launch_module.RESERVED_FAMILIES == ()


# ---------------------------------------------------------------------------
# THE SCOPE CHANGE: the off-diagonal refusal is scoped to update_E, and stays
# ---------------------------------------------------------------------------
#
# Two backends licensed exactly this. The Triton closure round measured the three
# certified bodies against ``stepping.py`` on a folded + complex64 + live
# off-diagonal row grid over eight complete cycles — 0 of 110592 words differing
# at step_B, step_D and update_H, on the synthetic shape and on two row shapes —
# and 25041 of 110592 differing at update_E, the largest divergence of the round.
# The Metal round measured the same nine slots on its own backend as byte-identical
# over 12 complete driver steps and refuses the row at update_E alone.

def test_the_offdiagonal_family_admits_three_substeps_and_refuses_update_E():
    """The clause is scoped to where the row product IS, and to nowhere else.

    ``stepping._offdiagonal_terms`` (S:1190-1225) is reached from ``update_E``
    alone (S:972-979). The curls difference the STORED E and H and read no
    ``chi1inv``; ``update_H`` is B/mu with the PML accumulation and reads none
    either. THIS TEST FAILS IF THE CLAUSE DRIFTS BACK — either by re-broadening
    the sibling predicates to refuse the curls, or by narrowing them far enough to
    claim ``update_E``.
    """
    import numpy

    folded_complex = family("folded_complex")

    fields, pml = folded_complex_offdiag()
    fields.grid.xp = NamedAsCupy(numpy)
    probe = _sweep_probe()

    for sub_step in ("step_B", "step_D"):
        verdict = folded_complex.folded_complex_offdiag_pml_curl_coverage(
            fields, pml, sub_step, probe=probe)
        assert verdict.covered, (sub_step, verdict.reasons)
    admitted = folded_complex.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "H", probe=probe)
    assert admitted.covered, admitted.reasons

    refused = folded_complex.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "E", probe=probe)
    assert not refused.covered
    assert any("25041/110592" in reason for reason in refused.reasons), (
        "the update_E refusal must carry the measurement that licenses it, not "
        "an analogy")
    assert folded_complex.plan_folded_complex_offdiag_constitutive(
        fields, pml, "E", probe=probe) is None


def test_update_E_on_that_grid_selects_the_later_complex_row_product(monkeypatch):
    """The later numerical closure owns the slot this admission left refused.

    The 2026-08-15 residual-group licence deliberately covers only the three
    non-row-product sub-steps above.  The distinct complex folded off-diagonal
    ``update_E`` body was subsequently built and byte-gated, so the composer
    must select that new arm without widening the older family.
    """
    plan = plan_for(folded_complex_offdiag, monkeypatch)
    assert plan.selected["update_E"] == "complex folded off-diagonal"
    assert "update_E" in plan.plans
    assert ambiguous_slots(plan) == (), plan.reasons


def test_narrowing_the_incumbent_clause_would_now_cost_all_twelve_slots(
        monkeypatch):
    """The alternative repair, MEASURED — this is why the clause is untouched.

    The obvious-looking change is to narrow ``folded_complex._media_reasons``'
    off-diagonal clause in place so the incumbent curl and constitutive
    predicates stop refusing the row on three of four sub-steps. Do that and the
    incumbent arms admit exactly what the sibling arms admit, ``_select_slot``
    sees TWO admitters and fails closed.  The three older slots and the later
    complex row-product ``update_E`` slot all fall to the array path: four slots
    on each of the three corpus rows, or twelve lost slots with no error anywhere.

    Simulated here by dropping that one clause from the shared helper, which is
    what an in-place narrowing amounts to at every sub-step the row does not
    reach.
    """
    folded_complex = family("folded_complex")

    original = folded_complex._media_reasons

    def narrowed(fields, grid, targets):
        return [reason for reason in original(fields, grid, targets)
                if "off-diagonal chi1inv row is installed" not in reason]

    monkeypatch.setattr(folded_complex, "_media_reasons", narrowed)
    plan = plan_for(folded_complex_offdiag, monkeypatch)

    assert ambiguous_slots(plan) == (
        "step_B", "step_D", "update_E", "update_H"), plan.reasons
    for slot in ("step_B", "step_D", "update_E", "update_H"):
        assert slot not in plan.selected
        reason, = plan.reasons[slot]
        assert "folded complex" in reason and "off-diagonal" in reason, reason


def test_the_incumbent_clause_is_still_in_force_on_all_four_substeps():
    """The other half of the disjointness: the incumbent has NOT been narrowed.

    Everything above rests on the incumbent folded-complex predicates refusing an
    off-diagonal row everywhere, so the sibling arms are the only admitters. This
    reads that off the shipped predicates rather than trusting the arrangement.
    """
    import numpy

    folded_complex = family("folded_complex")

    fields, pml = folded_complex_offdiag()
    fields.grid.xp = NamedAsCupy(numpy)
    probe = _sweep_probe()

    verdicts = [folded_complex.folded_complex_composition_curl_coverage(
                    fields, pml, name, probe=probe) for name in
                ("step_B", "step_D")]
    verdicts += [folded_complex.folded_complex_constitutive_coverage(
                     fields, pml, side, probe=probe) for side in ("H", "E")]
    for verdict in verdicts:
        assert not verdict.covered
        assert any("off-diagonal chi1inv row is installed" in reason
                   for reason in verdict.reasons), verdict.reasons


# ---------------------------------------------------------------------------
# The other three families: disjointness in both directions
# ---------------------------------------------------------------------------

def test_the_nonlinear_run_admission_does_not_reach_update_E():
    """``side='E'`` is where the Pade factor lives, and it belongs to another arm.

    The CONTROL leg measured the certified constitutive body at 1536 of 12288
    words differing there. ``nonlinear_constitutive_coverage`` — a separate arm,
    wired a round earlier — is what covers that slot.
    """
    import numpy

    nonlinear_update_e = family("nonlinear_update_e")

    fields, pml = nonlinear(cart())
    fields.grid.xp = NamedAsCupy(numpy)

    admitted = nonlinear_update_e.nonlinear_run_constitutive_coverage(
        fields, pml, "H")
    assert admitted.covered, admitted.reasons
    refused = nonlinear_update_e.nonlinear_run_constitutive_coverage(
        fields, pml, "E")
    assert not refused.covered
    assert any("1536/12288" in reason for reason in refused.reasons), refused.reasons


def test_a_nonlinear_grid_still_refuses_opt_in_pair_fusion(monkeypatch):
    """The D-pair COMPOSES update_E, so it may not inherit the curl admission.

    Named in the family's integration note as the warning for this round: a fused
    D-pair on a nonlinear run is exactly the configuration the CONTROL leg
    measured diverging. ``specialized_family_owns_the_grid`` keeps it out
    structurally, and that has to survive the curls becoming coverable.
    """
    plan = plan_for(lambda: nonlinear(cart()), monkeypatch, fuse=True, sources=())
    for name in ("fused_pair_B", "fused_pair_D", "fused_pair_dispersive_D"):
        assert plan.reasons.get(name), name
        assert any("specialized family owns this grid" in reason
                   or "specialized-family" in reason
                   for reason in plan.reasons[name]), (name, plan.reasons[name])
    assert plan.selected["step_D"] == "nonlinear run PML"
    assert plan.selected["update_E"] == "nonlinear"


def test_the_folded_dispersive_arm_is_disjoint_from_both_incumbents():
    """Two inverted clauses, so two refusals have to hold for this arm to be safe."""
    import numpy

    coverage_module = family("coverage")
    folded_dispersive_update_e = family("folded_dispersive_update_e")
    symmetry = family("symmetry")

    fields, pml = folded_dispersive()
    fields.grid.xp = NamedAsCupy(numpy)

    assert folded_dispersive_update_e.folded_dispersive_constitutive_coverage(
        fields, pml).covered

    folded = symmetry.folded_constitutive_coverage(fields, pml, "E")
    assert not folded.covered
    assert any("susceptibility" in reason for reason in folded.reasons), folded.reasons

    unfolded = coverage_module.dispersive_constitutive_coverage(fields, pml)
    assert not unfolded.covered
    assert any("mirror" in reason or "folded" in reason
               for reason in unfolded.reasons), unfolded.reasons


def test_the_two_ade_families_are_disjoint_on_the_configurations_they_serve():
    """The inverted clause is the LAYER, so each family refuses the other's run.

    Non-vacuity in both directions: an admission whose neighbour also admits is
    an ambiguity the composer fails closed on, and one whose neighbour was never
    asked is not measured at all.
    """
    import numpy

    coverage_module = family("coverage")
    no_pml_ade = family("no_pml_ade")

    fields, pml = absorber_free_dispersive()
    fields.grid.xp = NamedAsCupy(numpy)
    state, = fields.polarizations
    for component in state.driven():
        assert no_pml_ade.no_pml_ade_update_p_coverage(
            fields, pml, state, component).covered
        incumbent = coverage_module.ade_update_p_coverage(fields, state, component)
        assert not incumbent.covered
        assert any("f_w_" in reason and "not allocated" in reason
                   for reason in incumbent.reasons), incumbent.reasons

    # And the mirror direction, on a run that DOES have a split-field layer.
    from meep_gpu.test_triton_no_pml_ade import build as ade_build

    pml_fields, pml_layer, states = ade_build(storage="pml")
    pml_fields.grid.xp = NamedAsCupy(numpy)
    for component in states[0].driven():
        refused = no_pml_ade.no_pml_ade_update_p_coverage(
            pml_fields, pml_layer, states[0], component)
        assert not refused.covered
        assert any("active PML layer is installed" in reason
                   for reason in refused.reasons), refused.reasons


def test_a_double_admission_on_update_P_leaves_the_slot_on_the_array_path(
        monkeypatch):
    """update_P has no arm table, so its fail-closed rule is pinned separately.

    The block is BUILDER-FIRST — each builder asks its own predicate — so a plain
    fallback would have preferred whichever family was tried first the moment both
    admitted. Planted here because the two are disjoint by construction and the
    ordering hazard would otherwise be untestable.
    """
    Coverage = family("coverage").Coverage

    monkeypatch.setattr(launch_module, "ade_update_p_coverage",
                        lambda *a, **k: Coverage(True, ()), raising=False)
    plan = plan_for(absorber_free_dispersive, monkeypatch)

    assert "update_P" not in plan.selected
    assert "update_P" not in plan.plans
    assert not plan.polarization_plans
    reason, = plan.reasons["update_P"]
    assert "ambiguous" in reason
    assert "ADE update_P" in reason and "no-PML ADE update_P" in reason, reason


def test_a_state_that_drives_nothing_is_not_a_unanimous_admission(monkeypatch):
    """``all(())`` is True, and that is how a family wins a slot over no evidence.

    The unit of admission for this sub-step is the whole polarization pass, so a
    state with an empty driven list has to count as a refusal rather than as a
    vacuous pass.
    """
    from types import SimpleNamespace

    Coverage = family("coverage").Coverage

    silent = SimpleNamespace(driven=lambda: ())
    assert launch_module._ade_family_admits(
        (silent,), lambda state, component: Coverage(True, ())) is False
    assert launch_module._ade_family_admits(
        (), lambda state, component: Coverage(True, ())) is False


def test_a_raising_update_P_predicate_is_a_refusal_not_a_crash():
    """The arm table's rule, applied to the block that has no arm table."""
    from types import SimpleNamespace

    def hostile(state, component):
        raise RuntimeError("this predicate cannot answer")

    state = SimpleNamespace(driven=lambda: ("Ez",))
    assert launch_module._ade_family_admits((state,), hostile) is False


def test_the_absorber_free_run_reports_both_families_when_neither_admits(
        monkeypatch):
    """A refusal naming only the incumbent sends a reader to the wrong clause.

    On an absorber-free run the incumbent refuses for "f_w is not allocated",
    which is the least informative thing that can be said about that
    configuration. Both consulted families explain themselves.
    """
    import numpy

    fields, pml = absorber_free_dispersive()
    # Break the sibling's admission in one place only: the drive must BE the
    # stored E, and switching PML storage MODE on is what makes it not (the
    # layer stays inert, so the gate stays open and the arm stays consulted).
    fields._pml_active = True

    install_sentinel_builders(monkeypatch)
    # Both builders answer None, which is what the REAL ones do when their own
    # predicate refuses — the sentinels above answer a string unconditionally and
    # would fill the slot from a family that had refused.
    for builder in ("plan_ade_update_p", "plan_no_pml_ade_update_p"):
        monkeypatch.setattr(launch_module, builder, lambda *a, **k: None,
                            raising=False)
    fields.grid.xp = NamedAsCupy(numpy)
    plan = launch_module.plan_step(fields, pml, probe=_sweep_probe())

    assert "update_P" not in plan.selected
    joined = " | ".join(plan.reasons["update_P"])
    assert "f_w_" in joined, joined
    assert "no-PML ADE: " in joined, joined
