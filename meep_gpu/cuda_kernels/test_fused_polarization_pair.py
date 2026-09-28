"""The E->P fused pair: the lift, the predicates, the wiring and the rotation protocol.

WHAT THIS FILE CAN AND CANNOT SAY. The weld's ARITHMETIC is a device claim --
``gate_cuda_fused_polarization_pair.py`` carries it, RELEASED 2026-09-01 under
both float32 subnormal policies (the two ``*_2026-09-01`` blocks in
``certification.json``). What a laptop can pin is everything whose failure would
be silent BEFORE a device is reached:

1. **The lift.** Every emitted line is either the certified emitters' own text
   (asserted against ``dispersive_kernels``' emission and ``ade_kernels``' syntax
   tree) or one of :data:`~.fused_polarization_pair.LIFT_EDITS`' declared
   departures -- and a certified string that drifts RAISES rather than emitting a
   kernel that is quietly not the certified arithmetic.
2. **The predicates.** The two products partition on the absorber; the seam
   clauses (drive identity, extent identity) are load-bearing and named; the
   source list is inert in BOTH directions, which is the driver fact that lets
   this seam skip the deposit protocol.
3. **The wiring.** The composer installs the pair only where the arm table gave
   ``update_E`` to ``dispersive`` (or its no-PML twin) and ``update_P`` to
   ``ADE``; the seam row's ``None`` deposit-list name takes the ``NoopPlan``
   branch even when electric sources exist, because they are injected OUTSIDE
   this seam.
4. **The rotation protocol.** Three launches per run, host rotation between,
   with the final buffer assignment of every state EXACTLY the certified
   per-(state, component) sequence's -- "advances one buffer twice and freezes
   another" is the failure that still computes, so it is pinned here with a
   mocked kernel and pointer identities rather than left to the gate.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from ..triton_kernels.launch import NoopPlan
from . import arms, fused_pairs, registry
from . import dispersive_kernels
from . import fused_polarization_pair as family
from .test_dispersive import build


# ---------------------------------------------------------------------------
# 1. THE LIFT
# ---------------------------------------------------------------------------

def test_the_emitted_chain_is_the_certified_emitters_own_lines():
    """Every chain line of every digest spec appears verbatim in the certified
    dispersive emission at the same arity -- already enforced inside the emitter;
    exercised here across the sweep so the enforcement itself is reached."""
    sources = family.device_sources()
    assert len(sources) >= 60, "the digest sweep collapsed"
    for key, source in sources.items():
        assert source.isascii(), key
        assert source.count('extern "C" __global__') == 1, key


@pytest.mark.parametrize("arm", ("pml", "no_pml"))
def test_a_drifted_certified_emission_fails_the_splice(arm, monkeypatch):
    """The anchors are load-bearing: certified text that moves under this family
    must fail the splice, never survive it. Driven by mutating the full emission
    the pieces are verified against -- a lifted chain line that stops appearing
    there is the drift the anchor exists to catch."""
    real = dispersive_kernels.dispersive_source

    def drifted(inner_arm, counts):
        return real(inner_arm, counts).replace(
            "s_x - P_Ex_0[idx]", "s_x - P_Ex_0[idx + 1]")

    monkeypatch.setattr(dispersive_kernels, "dispersive_source", drifted)
    with pytest.raises(AssertionError, match="not in the certified emission"):
        family.fused_polarization_pair_source(arm, 0, 1, (True,))

    # ... and a vanished tail is the other side of the same anchor.
    def tailless(inner_arm, counts):
        text = real(inner_arm, counts)
        return text.replace(family._pml_tail_line(0) + "\n", "") if \
            inner_arm == "pml" else text.replace(
            family._no_pml_tail_line(0) + "\n", "")

    monkeypatch.setattr(dispersive_kernels, "dispersive_source", tailless)
    with pytest.raises(AssertionError, match="no longer carries"):
        family.fused_polarization_pair_source(arm, 0, 1, (True,))


def test_a_drifted_certified_recurrence_raises_instead_of_emitting(monkeypatch):
    strings = family._certified_ade_strings()
    broken = dict(strings)
    broken["_update_P_pml_real_kernel_code"] = strings[
        "_update_P_pml_real_kernel_code"].replace(
        "(c_drive * (s * w))", "((c_drive * s) * w)")
    monkeypatch.setattr(family, "_certified_ade_strings", lambda: broken)
    with pytest.raises(AssertionError):
        family.fused_polarization_pair_source("pml", 0, 1, (True,))


def test_the_recurrence_keeps_the_certified_grouping_per_pole():
    """The left-associated ((p*c_now) + (c_prev*q)) + (c_drive*(s*w)) grouping,
    per pole, under the per-pole names -- float32 addition is not associative and
    the grouping is the array path's two ``+=`` statements."""
    source = family.fused_polarization_pair_source("pml", 0, 2, (True, False))
    for i in range(2):
        assert (f"p_out_{i}[idx] = ((p_{i} * c_now_{i}) + (c_prev_{i} * q_{i})) "
                f"+ (c_drive_{i} * (s_{i} * w));") in source


@pytest.mark.parametrize("arm", ("pml", "no_pml"))
def test_the_pole_pointer_is_bound_once_and_p_now_is_not_a_parameter(arm):
    """THE ALIASING ANSWER. The E half's chain and the P half's recurrence read
    the same allocation, so the signature binds it once and the certified
    ``p_now`` parameter does not exist here."""
    source = family.fused_polarization_pair_source(arm, 0, 2, (True, True))
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("//"))
    assert "p_now" not in code
    for i in range(2):
        assert code.count(f"const float* __restrict__ P_Ex_{i}") == 1
        # ... and BOTH halves read through it: once in the chain, once as p_i.
        assert code.count(f"P_Ex_{i}[idx]") == 2


@pytest.mark.parametrize("arm", ("pml", "no_pml"))
def test_the_seam_register_replaces_the_drive_and_no_drive_pointer_exists(arm):
    source = family.fused_polarization_pair_source(arm, 1, 1, (True,))
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("//"))
    assert code.count("float w = src_y;") == 1
    assert "drive[" not in code
    assert "drive" not in [token for line in code.splitlines()
                           for token in line.replace("(", " ").split()
                           if token == "drive"]


def test_the_zero_arity_body_is_the_e_half_alone():
    """A component nothing drives still launches (the driver's update_E covers
    all three), and its kernel is the certified per-component E text with no
    seam and no recurrence."""
    source = family.fused_polarization_pair_source("pml", 2, 0, ())
    assert "float w" not in source
    assert "p_out" not in source
    assert "constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);" in source


def test_an_arity_past_the_measured_cap_is_refused_by_name():
    with pytest.raises(ValueError, match="POLE_COUNT_CAP"):
        family.fused_polarization_pair_source(
            "pml", 0, dispersive_kernels.POLE_COUNT_CAP + 1,
            (True,) * (dispersive_kernels.POLE_COUNT_CAP + 1))
    with pytest.raises(ValueError, match="sigma_kinds"):
        family.fused_polarization_pair_source("pml", 0, 2, (True,))


def test_the_contraction_guard_is_spelled_here_and_matches_both_halves():
    """``--fmad=false`` is CORRECTNESS on both halves; the fused module spells
    its own tuple so a by-path load cannot pick up a different one."""
    source = pathlib.Path(family.__file__).read_text(encoding="utf-8")
    assert "_COMPILE_OPTIONS: Tuple[str, ...] = ('--fmad=false',)" in source
    assert family._COMPILE_OPTIONS == dispersive_kernels._COMPILE_OPTIONS


# ---------------------------------------------------------------------------
# 2. THE PREDICATES
# ---------------------------------------------------------------------------

def test_the_pml_product_admits_its_cells_configuration():
    fields, layer, grid = build(poles=2, pml_on=True)
    covered, reason = family.covers_fused_polarization_pair(fields, layer, grid)
    assert covered, reason


def test_the_no_pml_product_admits_its_cells_configuration():
    fields, layer, grid = build(poles=2, pml_on=False)
    covered, reason = family.covers_no_pml_fused_polarization_pair(
        fields, layer, grid)
    assert covered, reason


def test_the_two_products_partition_on_the_absorber_alone():
    """No configuration may admit both -- two admitters leave the seam UNFUSED
    naming both, so the disjointness is what buys the slots."""
    for pml_on in (True, False):
        fields, layer, grid = build(poles=2, pml_on=pml_on)
        pml_covered, pml_reason = family.covers_fused_polarization_pair(
            fields, layer, grid)
        bare_covered, bare_reason = family.covers_no_pml_fused_polarization_pair(
            fields, layer, grid)
        assert pml_covered == pml_on, pml_reason
        assert bare_covered == (not pml_on), bare_reason
        refused = bare_reason if pml_on else pml_reason
        assert "update_E half:" in refused or "update_P half:" in refused


def test_a_halves_refusal_arrives_prefixed_with_the_side_that_said_it():
    fields, layer, grid = build(poles=2, pml_on=True)
    fields.polarizations.clear()
    covered, reason = family.covers_fused_polarization_pair(fields, layer, grid)
    assert not covered
    assert reason.startswith("update_E half:")
    assert "no susceptibility is registered" in reason


def test_the_drive_identity_clause_is_load_bearing_and_shadowed_not_absent():
    """The register hand-off IS the certified drive only while drive_field names
    the array the E half writes.

    TWO LAYERS ANSWER, AND BOTH ARE MEASURED. The ADE half's own predicate
    refuses a wrong drive first (this predicate is a conjunction that asks it
    first), so the conjunction's refusal is the half's -- and the seam's OWN
    clause is then defence in depth, pinned directly so a later reordering of
    the conjunction cannot quietly turn the shadowing into a gap: the fused
    kernel binds NO drive pointer at all, so this product's soundness rests on
    the identity even where the half's clause were licensed away.
    """
    fields, layer, grid = build(poles=2, pml_on=True)
    fields.drive_field = lambda component: getattr(
        fields, "D" + component[1])  # the wrong-drive control's own binding
    covered, reason = family.covers_fused_polarization_pair(fields, layer, grid)
    assert not covered
    assert "drive_field" in reason

    problem = family._drive_identity_problem(fields, "pml", "Ex")
    assert problem is not None
    assert "register hand-off" in problem and "different number" in problem
    # ... and the launcher re-asks it, so a run reaching the launch with a
    # drive the predicate never saw is refused there too.
    with pytest.raises(ValueError, match="register hand-off"):
        family.launch_fused_polarization_component(
            fields, "no_pml", "Ex", tuple(fields.polarizations))


def test_the_extent_clause_behind_the_dropped_guard_is_load_bearing():
    """The half's own shape clause fires first on a mis-shaped buffer (the
    conjunction asks it first); the seam's own extent clause is then defence in
    depth behind the DROPPED update_P guard, pinned directly."""
    fields, layer, grid = build(poles=2, pml_on=True)
    fields.polarizations[0]._scratch = fields.polarizations[0]._scratch[:-1]
    covered, reason = family.covers_fused_polarization_pair(fields, layer, grid)
    assert not covered
    assert "_scratch" in reason

    problem = family._seam_clauses(fields, "pml")
    assert problem is not None
    assert "one number" in problem and "drops the certified update_P guard" in problem


def test_the_source_list_is_inert_in_both_directions():
    """THE DRIVER FACT. Nothing is injected between driver.py:3313 and :3315, so
    no source list -- undeclared, empty, electric or magnetic -- may move the
    verdict. This is what test_fused_pairs.py's deposit-wiring suite would call
    an entry TRUE of the product: the B and D products must refuse an undeclared
    list because their seams have an injection slot; this seam has none to be
    ignorant about."""

    class _Electric:
        field_type = "D"

    class _Magnetic:
        field_type = "B"

    for pml_on, predicate in (
            (True, family.covers_fused_polarization_pair),
            (False, family.covers_no_pml_fused_polarization_pair)):
        fields, layer, grid = build(poles=2, pml_on=pml_on)
        verdicts = {label: predicate(fields, layer, grid, sources)
                    for label, sources in (
                        ("undeclared", None), ("empty", ()),
                        ("electric", [_Electric()]), ("magnetic", [_Magnetic()]))}
        assert len({verdict for verdict in verdicts.values()}) == 1, verdicts
        assert all(covered for covered, _reason in verdicts.values()), verdicts


def test_carries_deposit_repair_is_false_and_no_repair_wiring_exists():
    """The flag and the wiring change together or not at all -- here BOTH are
    absent, because the seam has no deposit to carry: the module never touches
    deposit_repair and never names the two repair plans."""
    assert family.CARRIES_DEPOSIT_REPAIR is False
    source = pathlib.Path(family.__file__).read_text(encoding="utf-8")
    assert "LeadingRepairPlan" not in source
    assert "TrailingRepairPlan" not in source
    assert "seam_source_reasons" not in source


# ---------------------------------------------------------------------------
# 3. THE WIRING
# ---------------------------------------------------------------------------

def test_fusion_is_opt_in_and_the_default_composition_is_unchanged():
    fields, layer, grid = build(poles=2, pml_on=True)
    plan = arms.plan_step(fields, layer, grid, sources=())
    assert plan.selected["update_E"] == "dispersive"
    assert plan.selected["update_P"] == "ADE"
    assert not [key for key in plan.reasons if key.startswith("fused_pair_")]


#: WHERE THIS PAIR STILL INSTALLS, 2026-09-02, and it is a DRIVER FACT rather than a
#: contrived fixture. ``cuda_(no_pml_)three_slot_dispersive_weld`` spans ``step_D`` ->
#: ``update_E`` -> ``update_P`` and, where it admits, takes all three slots -- its
#: predicate strictly contains this pair's, so ``_superseded_by_a_longer_span`` gives
#: it the span and ``_pair_may_absorb`` then refuses this pair BY NAME.
#:
#: AN UNDECLARED SOURCE LIST IS THE ASYMMETRY, and it is exactly this module's own
#: subject: the D seam has an INJECTION in it, so every product spanning it refuses
#: ``sources=None`` ("the source set was not declared") and cannot infer an empty
#: seam from ``Fields``. The E->P seam has none -- the driver puts nothing between
#: driver.py:3332 and :3334 -- so this pair answers with the list undeclared, and is
#: the only product on the run that can. The two seam tests below therefore walk BOTH
#: readings: the weld takes the span when the sources are declared, and this pair
#: fuses exactly as it always did when they are not.
_UNDECLARED = None


def test_the_pml_seam_fuses_where_the_table_gave_it_the_right_arms():
    fields, layer, grid = build(poles=2, pml_on=True)
    plan = arms.plan_step(fields, layer, grid, sources=_UNDECLARED, fuse=True)
    assert plan.selected["update_E"] == plan.selected["update_P"] == \
        "fused polarization pair"
    assert isinstance(plan.plans["update_E"], fused_pairs.CudaFusedPairPlan)
    assert isinstance(plan.plans["update_P"], NoopPlan)
    assert plan.plans["update_E"].replaces == ("update_E", "update_P")
    assert fused_pairs.declaring_plan(plan.plans["update_P"]) is \
        plan.plans["update_E"]
    assert "update_P" in fused_pairs.replaced_sub_steps(plan.plans)


def test_the_no_pml_seam_fuses_into_its_own_product():
    fields, layer, grid = build(poles=2, pml_on=False)
    plan = arms.plan_step(fields, layer, grid, sources=_UNDECLARED, fuse=True)
    assert plan.selected["update_E"] == plan.selected["update_P"] == \
        "no-absorber fused polarization pair"
    assert isinstance(plan.plans["update_E"], fused_pairs.CudaFusedPairPlan)
    assert plan.plans["update_E"].family == family.FAMILY_NO_PML


@pytest.mark.parametrize("pml_on,label", (
    (True, "three-slot dispersive weld"),
    (False, "no-absorber three-slot dispersive weld")))
def test_a_declared_source_list_gives_the_span_to_the_three_slot_weld(pml_on, label):
    """THE OTHER READING, and it is a GAIN for this seam rather than a loss.

    Before the three-slot weld existed, a declared source list on these runs left
    ``update_E`` with this pair and ``step_D`` on the array path -- the D->E product
    that admitted the same rows was withheld by ``_later_seam_claimant`` because the
    trade was measured net zero. Now ONE product owns all three slots, so both seams
    are served and this pair is refused BY NAME, with the reason naming the slot
    rather than a trade.

    PINNED HERE, in this module's own file, because "who has update_E" is exactly
    what this product is about, and a reader arriving from its docstring needs the
    answer to be one grep away.
    """
    fields, layer, grid = build(poles=2, pml_on=pml_on)
    source = _electric_source(grid)
    plan = arms.plan_step(fields, layer, grid, sources=(source,), fuse=True)
    for slot in ("step_D", "update_E", "update_P"):
        assert plan.selected[slot] == label, slot
    predicate = (family.covers_fused_polarization_pair if pml_on
                 else family.covers_no_pml_fused_polarization_pair)
    covered, _why = predicate(fields, layer, grid, (source,))
    assert covered, ("the PREDICATE must still admit it -- the refusal is the "
                     "installer's arbitration, not a verdict about this weld")
    key = ("fused_pair_cuda_fused_polarization_pair" if pml_on
           else "fused_pair_cuda_no_pml_fused_polarization_pair")
    assert "update_E was selected by" in plan.reasons[key][0]


def _electric_source(grid):
    from ..sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def test_an_electric_source_does_not_move_the_seam_to_the_repair_plans():
    """The installer's ``None`` deposit-list branch, driven end to end: an
    electric source IS injected by the driver -- between step_D and update_E,
    one seam earlier -- and this seam must still take the NoopPlan branch."""

    class _Electric:
        field_type = "D"
        _point_ix = _point_iy = _point_iz = 0

    fields, layer, grid = build(poles=2, pml_on=False)
    plans, selected = {}, {}
    marker = object()
    fused_pairs._install_fused_pair(
        plans, selected, fields, layer, [_Electric()], None,
        "update_E", "update_P", "fused polarization pair", marker)
    assert plans["update_E"] is marker
    assert isinstance(plans["update_P"], NoopPlan)
    assert selected["update_E"] == selected["update_P"] == \
        "fused polarization pair"


def test_a_pair_may_not_claim_slots_the_table_gave_to_other_arms():
    fields, layer, grid = build(poles=2, pml_on=True)
    plans, reasons = {}, {}
    selected = {"update_E": "dispersive", "update_P": "complex no-PML ADE"}
    context = arms.StepContext(fields, layer, grid, sources=())
    fused_pairs.install_fused_pairs(plans, reasons, selected, context)
    assert not plans
    reason = reasons["fused_pair_cuda_fused_polarization_pair"][0]
    assert "was selected by the 'complex no-PML ADE' arm" in reason


def test_the_absorb_declaration_matches_the_arms_the_predicates_conjoin():
    """``FUSED_PAIR_ARMS`` is READ FROM THE PREDICATES, not assigned to them --
    the same pin ``test_fused_pairs.py`` holds on the magnetic product."""
    source = pathlib.Path(family.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    def called_by(name):
        node = next(item for item in tree.body
                    if isinstance(item, ast.FunctionDef) and item.name == name)
        return {called.func.attr if isinstance(called.func, ast.Attribute)
                else called.func.id
                for called in ast.walk(node) if isinstance(called, ast.Call)
                and isinstance(called.func, (ast.Attribute, ast.Name))}

    assert {"covers_real_pml_dispersive_constitutive",
            "covers_real_pml_ade_update_p"} <= called_by(
        "covers_fused_polarization_pair")
    assert {"covers_no_pml_dispersive_constitutive",
            "covers_no_pml_ade_update_p"} <= called_by(
        "covers_no_pml_fused_polarization_pair")

    by_predicate = {row["name"]: row["label"] for row in registry._TABLE}
    assert fused_pairs.FUSED_PAIR_ARMS["cuda_fused_polarization_pair"] == (
        by_predicate["covers_real_pml_dispersive_constitutive"],
        by_predicate["covers_real_pml_ade_update_p"])
    assert fused_pairs.FUSED_PAIR_ARMS["cuda_no_pml_fused_polarization_pair"] == (
        by_predicate["covers_no_pml_dispersive_constitutive"],
        by_predicate["covers_no_pml_ade_update_p"])


def test_the_product_rows_and_module_constants_agree():
    for name, constant in (("cuda_fused_polarization_pair", family.FAMILY_PML),
                           ("cuda_no_pml_fused_polarization_pair",
                            family.FAMILY_NO_PML)):
        assert name == constant
        assert fused_pairs.FUSED_PRODUCTS[name]["curl_slot"] == family.SLOT
        assert fused_pairs.FUSED_PRODUCTS[name]["module"] == \
            "fused_polarization_pair"
        assert name in fused_pairs.FUSED_PAIR_ARMS
    assert family.REPLACES == ("update_E", "update_P")


# ---------------------------------------------------------------------------
# 4. THE ROTATION PROTOCOL, WITH A MOCKED KERNEL
# ---------------------------------------------------------------------------

class _Recorder:
    """Stands in for the compiled kernel: records the launch, computes nothing.

    The ARITHMETIC is the device gate's question; what a laptop can pin is the
    protocol around the launches -- how many, in what order, against which
    pointers -- which is exactly where "stale in a way that still computes"
    lives.
    """

    def __init__(self, log):
        self._log = log

    def __call__(self, blocks, threads, arguments):
        self._log.append((blocks, threads, arguments))


def test_three_launches_replace_one_plus_three_s_and_the_rotation_is_certified(
        monkeypatch):
    """THE COUNT AND THE ROTATION. 3 launches whatever the state count, and the
    final buffer assignment of every state is EXACTLY the certified
    per-(state, component) sequence's:

        after x: P[x] = scratch0,        P_prev[x] = old P[x]
        after y: P[y] = old P_prev[x],   P_prev[y] = old P[y]
        after z: P[z] = old P_prev[y],   P_prev[z] = old P[z]
        finally: _scratch = old P_prev[z]

    (dispersion.py:689-691 applied per component, the retired history becoming
    the next component's scratch). A launcher that advanced one buffer twice and
    froze another -- the failure that still computes -- cannot produce this
    pointer assignment.
    """
    fields, layer, grid = build(poles=2, pml_on=False)
    log = []
    monkeypatch.setattr(family, "_get_kernel",
                        lambda *args, **kwargs: _Recorder(log))
    before = [{"P": dict(state.P), "P_prev": dict(state.P_prev),
               "scratch": state._scratch} for state in fields.polarizations]

    result = family.run_fused_polarization_pair(fields, None, "no_pml")

    assert result["launches"] == 3
    assert result["recurrences"] == 2 * 3
    assert len(log) == 3
    for state, old in zip(fields.polarizations, before):
        assert state.P["Ex"] is old["scratch"]
        assert state.P_prev["Ex"] is old["P"]["Ex"]
        assert state.P["Ey"] is old["P_prev"]["Ex"]
        assert state.P_prev["Ey"] is old["P"]["Ey"]
        assert state.P["Ez"] is old["P_prev"]["Ey"]
        assert state.P_prev["Ez"] is old["P"]["Ez"]
        assert state._scratch is old["P_prev"]["Ez"]


def test_the_pole_pointers_are_resolved_after_the_previous_rotation(monkeypatch):
    """The second launch must bind the buffers as they stand AFTER the first
    component's rotation -- a launcher that resolved once before the loop would
    hand launch y a p_out that launch x's rotation already moved into P[x]."""
    fields, layer, grid = build(poles=1, pml_on=False)
    state = fields.polarizations[0]
    scratch0 = state._scratch
    prev_x = state.P_prev["Ex"]
    log = []
    monkeypatch.setattr(family, "_get_kernel",
                        lambda *args, **kwargs: _Recorder(log))
    family.run_fused_polarization_pair(fields, None, "no_pml")
    # launch x's p_out (argument 4: E, D, inv_eps, P, p_out, ...) is scratch0;
    # launch y's is the buffer x's rotation retired, never scratch0 again.
    x_args, y_args = log[0][2], log[1][2]
    assert x_args[4] is scratch0
    assert y_args[4] is prev_x
    assert y_args[4] is not scratch0


def test_a_wrong_size_scratch_is_refused_at_the_launch_too(monkeypatch):
    """Belt and braces with the predicate: the launcher re-checks the extent the
    dropped guard rested on, per launch, after the rotation that decided it."""
    fields, layer, grid = build(poles=1, pml_on=False)
    monkeypatch.setattr(family, "_get_kernel",
                        lambda *args, **kwargs: _Recorder([]))
    fields.polarizations[0]._scratch = fields.polarizations[0]._scratch.reshape(-1)[:-7]
    with pytest.raises(ValueError, match="one number"):
        family.run_fused_polarization_pair(fields, None, "no_pml")


def test_the_pml_arm_requires_the_certified_tables_and_names_the_resolver():
    fields, layer, grid = build(poles=1, pml_on=True)
    with pytest.raises(ValueError, match="dispersive_tables"):
        family.launch_fused_polarization_component(
            fields, "pml", "Ex", tuple(fields.polarizations), tables=None)


def test_a_launch_is_refused_by_name_where_cupy_is_absent():
    if family.cp is not None:
        pytest.skip("CuPy is importable on this host")
    with pytest.raises(RuntimeError, match="CuPy is not importable"):
        family._get_kernel("pml", 0, 1, (True,))


def test_run_refuses_an_unknown_arm_rather_than_defaulting():
    fields, layer, grid = build(poles=1, pml_on=False)
    with pytest.raises(ValueError, match="arm must be one of"):
        family.run_fused_polarization_pair(fields, None, "metal")
    with pytest.raises(ValueError, match="arm must be one of"):
        family.kernel_name("metal")


# ---------------------------------------------------------------------------
# 5. THE PARTITION AND THE RECORD BEHIND IT
# ---------------------------------------------------------------------------

def test_both_kernels_are_certified_and_the_partition_names_them_exactly():
    """RELEASED 2026-09-01; the record blocks are what license this partition,
    and ``test_kernel_partition.py``/``test_certification_metadata.py`` weld the
    two: every certified name must be claimed by a block, the blocks' artifact
    digests must recompute, and a name in neither set fails the walk."""
    assert set(family.CERTIFIED_KERNELS) == set(family.KERNEL_KEYS.values())
    assert family.UNCERTIFIED_KERNELS == {}


def test_the_lift_edits_are_five_and_the_seam_is_among_them():
    """An extra departure from the certified text may not arrive quietly."""
    assert len(family.LIFT_EDITS) == 5
    lines = {edit["line"] for edit in family.LIFT_EDITS}
    assert "    float w = drive[idx];" in lines
    assert any("p_now" in line for line in lines)
    seam = next(edit for edit in family.LIFT_EDITS
                if edit["line"] == "    float w = drive[idx];")
    assert "identity on the bits" in seam["why"]
