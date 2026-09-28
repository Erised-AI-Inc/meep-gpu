"""The THREE-SLOT weld: the lift, the predicates, the supersession and the bracket order.

WHAT THIS FILE CAN AND CANNOT SAY. The weld's ARITHMETIC is a device claim --
``parity/meep_gpu/gate_cuda_three_slot_weld.py`` carries it. What a laptop can pin is
everything whose failure would be SILENT before a device is reached, and on this
product that set is larger than on any pair, because the product's own host work sits
between its own two device groups:

1. **The lift.** The new kernel is ``ade_kernels``' certified recurrence under
   ``fused_polarization_pair``'s own per-pole renames, IMPORTED rather than copied, so
   a certified string that drifts raises here instead of emitting a body that is
   quietly not the certified arithmetic. Two things the E->P weld had to edit are
   RESTORED and that is asserted rather than described: ``float w = drive[idx];`` and
   the ADE guard.
2. **The predicates.** Each is a conjunction of two shipped ones, so its admission set
   is a SUBSET of each half's -- which is what makes the supersession rule non-lossy,
   and is measured here in both directions rather than argued.
3. **The wiring, which is the product.** ``_install_fused_triple`` puts the repair
   BETWEEN the two device groups: ``LeadingRepairPlan`` in ``step_D``,
   ``TrailingRepairPlan`` in ``update_E``, the polarization half in ``update_P``. Any
   other order computes and is wrong only at the deposit cells
   (``probe_cuda_three_slot_weld.py``: ``p_inside_the_launch`` and ``repair_after_p``
   each diverge on 6 of 7 configurations), so the ORDER is pinned here with recording
   stand-ins rather than left to the gate.
4. **The arbitration.** The three-slot weld supersedes its own D->E half by name, the
   E->P pair is failed closed out by the EXISTING ``_pair_may_absorb`` clause, and
   neither refusal fires on a run the weld does not admit.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ..triton_kernels.launch import NoopPlan
from .. import deposit_repair
from . import arms, fused_pairs, registry
from . import fused_polarization_pair as ep
from . import no_pml_three_slot_dispersive_weld as no_pml_family
from . import three_slot_dispersive_weld as family
from .test_dispersive import build


# ---------------------------------------------------------------------------
# 1. THE LIFT
# ---------------------------------------------------------------------------

def test_every_digest_source_is_ascii_and_declares_one_entry_point():
    sources = family.device_sources()
    assert len(sources) >= 40, "the digest sweep collapsed"
    for key, source in sources.items():
        assert source.isascii(), key
        assert source.count('extern "C" __global__') == 1, key
        assert source.count(f"void {family.KERNEL_NAME}(") == 1, key


def test_the_certified_drive_load_is_restored_unedited():
    """THE ROW THAT DISAPPEARED. ``fused_polarization_pair`` had to replace
    ``float w = drive[idx];`` with a register hand-off and argue that a float32 word
    stored and reloaded is the identity on the bits. This weld's kernel has no
    ``update_E`` half above it, so the certified line is spliced whole -- ONCE per
    component, never once per pole -- and there IS a ``drive`` parameter again."""
    source = family.three_slot_polarization_source(1, 3, (True, True, False))
    assert source.count("    float w = drive[idx];\n") == 1
    assert source.count("const float* __restrict__ drive,") == 1
    assert "src_" not in source, "no seam register survives in this family"
    # ...and the line is the certified body's own, not a retyped twin.
    volume = ep._ADE_VOLUME_BODY
    assert "    float w = drive[idx];\n" in volume


def test_the_certified_guard_is_restored_and_n_elem_is_a_parameter():
    """The other row that disappeared. The E->P weld dropped the ADE guard because an
    ``update_E`` body above had already declared ``idx``; nothing here has."""
    source = family.three_slot_polarization_source(0, 1, (False,))
    assert ep._ADE_GUARD in source
    assert source.count("    int n_elem\n") == 1


def test_a_drifted_certified_recurrence_raises_instead_of_emitting(monkeypatch):
    """A certified string that moves must fail HERE, on a laptop, not compile a body
    that is quietly not the certified arithmetic."""
    monkeypatch.setattr(ep, "_ADE_VOLUME_BODY",
                        ep._ADE_VOLUME_BODY.replace("c_prev * q", "q * c_prev"))
    with pytest.raises(AssertionError, match="no longer carries the recurrence"):
        family.three_slot_polarization_source(0, 1, (True,))


def test_the_recurrence_keeps_the_certified_grouping_per_pole():
    """float32 addition is not associative: the grouping IS the arithmetic."""
    source = family.three_slot_polarization_source(2, 2, (True, False))
    for index in range(2):
        assert (f"p_out_{index}[idx] = ((p_{index} * c_now_{index}) + "
                f"(c_prev_{index} * q_{index})) + "
                f"(c_drive_{index} * (s_{index} * w));") in source


def test_the_pole_pointer_is_bound_once_per_pole_and_stays_restrict():
    """Each ``P_<c>_<i>`` appears exactly once in the signature. It keeps the
    certified ``__restrict__`` -- which the E->P weld could not, because its
    ``update_E`` half read the same allocation through another parameter."""
    source = family.three_slot_polarization_source(0, 3, (True, True, True))
    signature = source.split(") {")[0].split('extern "C"')[1]
    body = source.split(") {")[1]
    for index in range(3):
        assert signature.count(f"P_Ex_{index}") == 1
        assert f"const float* __restrict__ P_Ex_{index}," in signature
    # ``p_now`` survives only in the certified ADE prologue's prose, which is lifted
    # whole; no parameter and no load carries the name.
    assert "p_now" not in signature and "p_now" not in body


def test_the_device_text_does_not_depend_on_the_arm():
    """The ``update_E`` half is gone, so what is left is the recurrence, and the
    absorber reaches the polarization through the DRIVE alone (stepping.py:1424).
    The no-absorber module therefore emits the PML module's bytes, not a twin."""
    assert no_pml_family.device_sources() == family.device_sources()
    assert no_pml_family.KERNEL_NAME == family.KERNEL_NAME


def test_a_component_with_no_driving_state_is_refused_rather_than_emitted_empty():
    with pytest.raises(ValueError, match="no recurrence to emit"):
        family.three_slot_polarization_source(0, 0, ())


def test_an_arity_past_the_measured_cap_is_refused_by_name():
    from . import dispersive_kernels

    cap = dispersive_kernels.POLE_COUNT_CAP
    with pytest.raises(ValueError):
        family.three_slot_polarization_source(0, cap + 1, (True,) * (cap + 1))


def test_the_lift_edits_are_three_and_the_two_risky_rows_are_gone():
    """DATA, not prose: the shrink from five rows to three is the design claim, so it
    is asserted. The two that left are the drive register and the dropped guard."""
    assert len(family.LIFT_EDITS) == 3
    assert len(ep.LIFT_EDITS) == 5
    text = " ".join(row["line"] + row["became"] for row in family.LIFT_EDITS)
    assert "src_<axis>" not in text
    assert "dropped" not in text
    assert no_pml_family.LIFT_EDITS is family.LIFT_EDITS


# ---------------------------------------------------------------------------
# 2. THE PREDICATES
# ---------------------------------------------------------------------------

def _pml_case(**kwargs):
    return build(poles=2, pml_on=True, **kwargs)


def _no_pml_case(**kwargs):
    return build(poles=2, pml_on=False, **kwargs)


def test_the_pml_weld_admits_its_cells_configuration():
    fields, layer, grid = _pml_case()
    covered, reason = family.covers_three_slot_dispersive_weld(
        fields, layer, grid, ())
    assert covered, reason


def test_the_no_pml_weld_admits_its_cells_configuration():
    fields, layer, grid = _no_pml_case()
    covered, reason = no_pml_family.covers_no_pml_three_slot_dispersive_weld(
        fields, layer, grid, ())
    assert covered, reason


def test_the_two_welds_partition_on_the_absorber_alone():
    for pml_on in (True, False):
        fields, layer, grid = build(poles=2, pml_on=pml_on)
        pml_covered, _r = family.covers_three_slot_dispersive_weld(
            fields, layer, grid, ())
        no_pml_covered, _r = (
            no_pml_family.covers_no_pml_three_slot_dispersive_weld(
                fields, layer, grid, ()))
        assert pml_covered is pml_on
        assert no_pml_covered is (not pml_on)


@pytest.mark.parametrize("pml_on", (True, False))
def test_the_admission_set_is_a_subset_of_both_halves(pml_on):
    """THE PROPERTY THE SUPERSESSION RULE RESTS ON, measured rather than argued: this
    weld's predicate is a conjunction, so wherever it admits, both two-slot products
    it displaces admit too. If that ever stopped holding, refusing the shorter span
    would start costing rows."""
    from . import dispersive_fused_electric_pair as de
    from . import no_pml_dispersive_fused_electric_pair as no_pml_de

    fields, layer, grid = build(poles=2, pml_on=pml_on)
    if pml_on:
        weld, _r = family.covers_three_slot_dispersive_weld(fields, layer, grid, ())
        half, half_reason = de.covers_dispersive_fused_electric_pair(
            fields, layer, grid, ())
        tail, tail_reason = ep.covers_fused_polarization_pair(fields, layer, grid, ())
    else:
        weld, _r = no_pml_family.covers_no_pml_three_slot_dispersive_weld(
            fields, layer, grid, ())
        half, half_reason = (
            no_pml_de.covers_no_pml_dispersive_fused_electric_pair(
                fields, layer, grid, ()))
        tail, tail_reason = ep.covers_no_pml_fused_polarization_pair(
            fields, layer, grid, ())
    assert weld
    assert half, half_reason
    assert tail, tail_reason


def test_a_halves_refusal_arrives_prefixed_with_the_side_that_said_it():
    """A conjunction that swallowed which half refused would send a reader to the
    wrong predicate."""
    fields, layer, grid = _pml_case()
    fields.polarizations.clear()
    covered, reason = family.covers_three_slot_dispersive_weld(
        fields, layer, grid, ())
    assert not covered
    assert reason.startswith("step_D->update_E half: ")


def test_the_source_list_is_refused_when_undeclared_and_carried_when_declared():
    """THE D SEAM HAS AN INJECTION IN IT, unlike the E->P seam, so ``sources`` is NOT
    inert here: ``None`` is a refusal and a real electric deposit is CARRIED, through
    the D->E half's own ``deposit_repair.seam_source_reasons`` routing."""
    from ..sources import GaussianEnvelope, VolumeSource

    fields, layer, grid = _pml_case()
    covered, reason = family.covers_three_slot_dispersive_weld(
        fields, layer, grid, None)
    assert not covered and "not declared" in reason

    source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    covered, reason = family.covers_three_slot_dispersive_weld(
        fields, layer, grid, (source,))
    assert covered, reason


def test_the_conjunction_adds_no_clause_of_its_own_and_needs_none():
    """WRITTEN AS A MEASUREMENT, because this weld shipped with a coefficient-triple
    clause until the halves turned out to ask it already. Two directions:

    * a malformed coefficient triple IS refused, so the fact is covered -- and the
      refusal is prefixed with the E->P half, which is where it comes from;
    * no clause of this module's own appears between the two conjuncts, so nothing
      here claims to be load-bearing while being redundant.
    """
    import inspect

    fields, layer, grid = _pml_case()
    fields.polarizations[0]._coefficients = (1.0, 2.0)
    covered, reason = family.covers_three_slot_dispersive_weld(
        fields, layer, grid, ())
    assert not covered
    assert reason.startswith("update_E->update_P half: ")
    assert "triple has 2 entries" in reason

    body = inspect.getsource(family.covers_three_slot_dispersive_weld)
    body = body.split('"""')[-1]
    assert body.count("if not covered:") == 2
    assert "_triple_seam_clauses" not in body


def test_both_modules_declare_the_repair_and_the_paths_their_halves_declare():
    from . import dispersive_fused_electric_pair as de
    from . import no_pml_dispersive_fused_electric_pair as no_pml_de

    assert family.CARRIES_DEPOSIT_REPAIR is True
    assert no_pml_family.CARRIES_DEPOSIT_REPAIR is True
    assert family.REPAIR_PATHS == (deposit_repair.SPLIT_FIELD_PATH,)
    assert no_pml_family.REPAIR_PATHS == no_pml_de.REPAIR_PATHS
    assert no_pml_family.REPAIR_PATHS == (deposit_repair.PLAIN_PATH,)
    # The PML half ships no REPAIR_PATHS at all, which IS the split-field default;
    # spelled here so the inheritance is a measurement rather than a comment.
    assert getattr(de, "REPAIR_PATHS", family.REPAIR_PATHS) == family.REPAIR_PATHS


# ---------------------------------------------------------------------------
# 3. THE WIRING -- three slots, and the bracket BETWEEN the two device groups
# ---------------------------------------------------------------------------

def test_the_product_rows_and_module_constants_agree():
    for module in (family, no_pml_family):
        row = fused_pairs.FUSED_PRODUCTS[module.FAMILY]
        assert tuple(row["slots"]) == module.SLOTS
        assert row["curl_slot"] == module.SLOT == module.SLOTS[0]
        assert fused_pairs.span_of(module.FAMILY, row) == module.SLOTS
        assert len(fused_pairs.FUSED_PAIR_ARMS[module.FAMILY]) == 3
        assert module.REPLACES[-1] == "update_P"
        assert module.SLOTS == ("step_D", "update_E", "update_P")


def test_the_absorb_declaration_matches_the_arms_the_predicates_conjoin():
    """Read off the predicates, as every row of that table is: the three labels are
    the arms the composer must already have selected."""
    assert fused_pairs.FUSED_PAIR_ARMS["cuda_three_slot_dispersive_weld"] == (
        "PML", "dispersive", "ADE")
    assert fused_pairs.FUSED_PAIR_ARMS["cuda_three_slot_no_pml_dispersive_weld"] == (
        "conductive", "no-PML dispersive store", "no-PML ADE")
    assert fused_pairs.FUSED_PAIR_EXTRA_ARMS[
        "cuda_three_slot_no_pml_dispersive_weld"] == (
        ("no-PML curl", "no-PML dispersive store", "no-PML ADE"),)


def test_neither_weld_is_registered_as_a_slot_arm_and_both_say_why():
    """It bids at no slot: all three it spans are already served, so registering it
    would fire the composer's ambiguity refusal on every row it covers."""
    for name in ("three_slot_dispersive_weld.covers_three_slot_dispersive_weld",
                 "no_pml_three_slot_dispersive_weld."
                 "covers_no_pml_three_slot_dispersive_weld"):
        assert name in registry.NOT_REGISTERED
        assert "THREE" in registry.NOT_REGISTERED[name] or \
            "NO-ABSORBER twin" in registry.NOT_REGISTERED[name]


class _RecordingHalf:
    """A stand-in for one device group: records that it ran, computes nothing."""

    def __init__(self, log, name):
        self._log = log
        self._name = name
        self.absorbed_by = None

    @property
    def replaces_sub_steps(self):
        return ("step_D", "update_E", "update_P")

    @property
    def launchable(self):
        return True

    def run(self, *_a, **_k):
        self._log.append(self._name)


class _RecordingTriple:
    def __init__(self, log):
        self.leading = _RecordingHalf(log, "group1")
        self.trailing = _RecordingHalf(log, "group2")
        self.label = "recording triple"


def test_the_installer_puts_the_repair_between_the_two_device_groups(monkeypatch):
    """THE ORDER IS THE PRODUCT. The driver consults step_D, then update_E, then
    update_P, so the objects the installer writes into those slots decide whether the
    repair reads P^n or P^(n+1). Recorded rather than reasoned about, because the
    wrong order computes."""
    fields, layer, grid = _pml_case()
    from ..sources import GaussianEnvelope, VolumeSource

    source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    log = []
    triple = _RecordingTriple(log)
    plans, selected = {}, {}
    saved = []
    applied = []
    monkeypatch.setattr(deposit_repair, "save",
                        lambda *a, **k: (saved.append(len(log)) or {"x": 1}))
    monkeypatch.setattr(deposit_repair, "apply",
                        lambda *a, **k: (applied.append(len(log)) or 7))

    fused_pairs._install_fused_triple(
        plans, selected, fields, layer, (source,), "D",
        ("step_D", "update_E", "update_P"), "three-slot", triple,
        (deposit_repair.SPLIT_FIELD_PATH,))

    assert type(plans["step_D"]).__name__ == "LeadingRepairPlan"
    assert type(plans["update_E"]).__name__ == "TrailingRepairPlan"
    assert plans["update_P"] is triple.trailing
    assert set(selected.values()) == {"three-slot"}

    for slot in ("step_D", "update_E", "update_P"):
        plans[slot].run()
    # THE SAVE IS BEFORE GROUP 1, THE APPLY IS BETWEEN THE GROUPS, AND GROUP 2 RUNS
    # LAST. Any other interleaving is one of the probe's diverging orderings.
    assert log == ["group1", "group2"]
    assert saved == [0], "the deposit was not captured before the first launch"
    assert applied == [1], "the repair did not run between the two device groups"


def test_a_sourceless_seam_still_puts_the_polarization_half_in_update_P(monkeypatch):
    fields, layer, grid = _pml_case()
    log = []
    triple = _RecordingTriple(log)
    plans, selected = {}, {}
    fused_pairs._install_fused_triple(
        plans, selected, fields, layer, (), "D",
        ("step_D", "update_E", "update_P"), "three-slot", triple,
        (deposit_repair.SPLIT_FIELD_PATH,))
    assert plans["step_D"] is triple.leading
    assert isinstance(plans["update_E"], NoopPlan)
    assert plans["update_P"] is triple.trailing


def test_every_slot_reports_the_whole_span_through_declaring_plan():
    """A wrapper read bare would fall back to "replaces only my own slot" and the
    composition would report five driver passes as running on the array path."""
    fields, layer, grid = _pml_case()
    log = []
    triple = _RecordingTriple(log)
    plans, selected = {}, {}
    fused_pairs._install_fused_triple(
        plans, selected, fields, layer, (), "D",
        ("step_D", "update_E", "update_P"), "three-slot", triple,
        (deposit_repair.SPLIT_FIELD_PATH,))
    for slot in ("step_D", "update_E", "update_P"):
        declaring = fused_pairs.declaring_plan(plans[slot])
        assert declaring.replaces_sub_steps == ("step_D", "update_E", "update_P")


def test_a_triple_plan_needs_both_launchers_to_be_launchable():
    plan = fused_pairs.CudaFusedTriplePlan(
        family="x", label="x", kernel_label="k",
        replaces=("step_D",), slots=("step_D", "update_E", "update_P"),
        context=None, resolve_launch_args=lambda _c: {},
        launch_leading=lambda _f, _a: {}, launch_trailing=None)
    assert not plan.launchable
    with pytest.raises(RuntimeError, match="no trailing launcher"):
        plan.run_half("trailing")
    with pytest.raises(ValueError, match="'leading' and a 'trailing' group"):
        plan.run_half("middle")


def test_a_triple_plan_refuses_a_span_that_is_not_three_slots():
    with pytest.raises(ValueError, match="owns three slots"):
        fused_pairs.CudaFusedTriplePlan(
            family="x", label="x", kernel_label="k", replaces=(),
            slots=("step_D", "update_E"), context=None)


# ---------------------------------------------------------------------------
# 4. THE ARBITRATION
# ---------------------------------------------------------------------------

def _plan(fields, layer, grid, sources):
    return arms.plan_step(fields, layer, grid, sources=sources, fuse=True)


@pytest.mark.parametrize("pml_on", (True, False))
def test_the_weld_takes_all_three_slots_and_names_what_it_superseded(pml_on):
    from ..sources import GaussianEnvelope, VolumeSource

    fields, layer, grid = build(poles=2, pml_on=pml_on)
    source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    plan = _plan(fields, layer, grid, (source,))
    label = ("three-slot dispersive weld" if pml_on
             else "no-absorber three-slot dispersive weld")
    for slot in ("step_D", "update_E", "update_P"):
        assert plan.selected.get(slot) == label, (slot, plan.reasons.get(slot))
    assert type(plan.plans["step_D"]).__name__ == "LeadingRepairPlan"
    assert type(plan.plans["update_E"]).__name__ == "TrailingRepairPlan"

    displaced = ("cuda_dispersive_fused_electric_pair" if pml_on
                 else "cuda_no_pml_dispersive_fused_electric_pair")
    reason = plan.reasons[f"fused_pair_{displaced}"][0]
    assert "strictly contains" in reason and "slot arbitration" in reason
    tail = ("cuda_fused_polarization_pair" if pml_on
            else "cuda_no_pml_fused_polarization_pair")
    tail_reason = plan.reasons[f"fused_pair_{tail}"][0]
    assert "update_E was selected by" in tail_reason


def test_the_supersession_does_not_fire_where_the_weld_does_not_admit():
    """The arity-zero run belongs to the REAL electric pair, and nothing about the
    three-slot welds may touch it."""
    fields, layer, grid = build(poles=0, pml_on=True)
    plan = _plan(fields, layer, grid, ())
    assert plan.selected.get("step_D") == "fused electric pair"
    assert plan.selected.get("update_P") is None
    for name in ("cuda_three_slot_dispersive_weld",
                 "cuda_three_slot_no_pml_dispersive_weld"):
        assert "strictly contains" not in " ".join(
            plan.reasons.get(f"fused_pair_{name}", ()))


def test_supersession_needs_strict_containment_not_merely_a_longer_span():
    """Two spans that only OVERLAP are still an ambiguity: neither serves what the
    other does, so neither may take the seam."""
    candidates = [("short", {"curl_slot": "step_D",
                             "slots": ("step_D", "update_E")}, ("a", "b")),
                  ("other", {"curl_slot": "step_D",
                             "slots": ("update_E", "update_P")}, ("b", "c"))]
    assert fused_pairs._superseded_by_a_longer_span(candidates) == {}
    candidates[1][1]["slots"] = ("step_D", "update_E", "update_P")
    superseded = fused_pairs._superseded_by_a_longer_span(candidates)
    assert set(superseded) == {"short"}


def test_the_later_seam_claimant_still_withholds_from_a_two_slot_product():
    """THE RULE THAT KEPT THE D->E PAIR OUT IS NOT WEAKENED, only made specific to a
    product that trades. Asked with a two-slot span it answers exactly as before."""
    fields, layer, grid = build(poles=2, pml_on=True)
    context = arms.StepContext(fields, layer, grid, None, None, ())
    claimant = fused_pairs._later_seam_claimant(
        {}, {}, {}, context, "update_E", ("step_D", "update_E"))
    assert claimant == "cuda_fused_polarization_pair"
    # ...and returns None for a product that serves the later seam itself.
    assert fused_pairs._later_seam_claimant(
        {}, {}, {}, context, "update_E",
        ("step_D", "update_E", "update_P")) is None


def test_a_three_slot_span_is_asked_the_absorb_question_once_per_seam():
    """``_spans_may_absorb`` is ``_pair_may_absorb`` called per seam, never a second
    copy of its clause -- and it names the slot that actually went elsewhere."""
    selected = {"step_D": "PML", "update_E": "dispersive", "update_P": "ADE"}
    span = ("step_D", "update_E", "update_P")
    assert fused_pairs._spans_may_absorb(
        selected, span, ("PML", "dispersive", "ADE")) is None
    refusal = fused_pairs._spans_may_absorb(
        selected, span, ("PML", "dispersive", "no-PML ADE"))
    assert refusal is not None and "update_P" in refusal


def test_fusion_is_opt_in_and_the_default_composition_is_unchanged():
    fields, layer, grid = build(poles=2, pml_on=True)
    plan = arms.plan_step(fields, layer, grid, sources=())
    assert plan.selected.get("update_P") == "ADE"
    assert "update_P" in plan.replaces


# ---------------------------------------------------------------------------
# 5. THE PARTITION AND THE RECORD
# ---------------------------------------------------------------------------

def test_both_modules_name_the_module_that_owns_the_emitted_text():
    """The compile memo and ``SOURCE_TRANSFORM`` live in ONE module, and both arms
    say which. A gate plants a defect by assigning ``SOURCE_TRANSFORM``; assigned to
    the no-absorber module it would set an attribute no compile path reads -- the
    launch still goes through the twin's ``_get_kernel``, the unmutated body is
    emitted, and every device mutation on that arm scores applied-and-null. This
    pins the declaration a gate reads instead of hardcoding which arm is the twin."""
    assert family.KERNEL_OWNER_MODULE == "three_slot_dispersive_weld"
    assert no_pml_family.KERNEL_OWNER_MODULE == family.KERNEL_OWNER_MODULE
    # The owner is the module that actually carries the doors; the twin carries none.
    for door in ("SOURCE_TRANSFORM", "_get_kernel", "_clear_kernel_cache",
                 "_COMPILE_OPTIONS", "_FUSED_THREADS"):
        assert hasattr(family, door), f"the owner must carry {door}"
    assert not hasattr(no_pml_family, "_get_kernel"), (
        "the twin must not carry a second compile path; one kernel name is owned by "
        "exactly one family and a second memo is a second place for it to drift")


def test_the_kernel_is_owned_by_one_module_and_is_certified():
    """The partition after the 2026-09-03 gate: the owner certifies the ONE name,
    nothing is left on the dead side, and the twin still declares no kernel of its
    own. ``test_kernel_partition.py`` holds the other half of this contract -- that
    a ``certification.json`` block names the kernel -- so the two files cannot say
    different things about whether a verdict exists."""
    assert family.CERTIFIED_KERNELS == (family.KERNEL_NAME,)
    assert family.UNCERTIFIED_KERNELS == {}
    assert no_pml_family.CERTIFIED_KERNELS == ()
    assert no_pml_family.UNCERTIFIED_KERNELS == {}


def test_the_gate_this_product_owes_exists_on_disk():
    gate = (pathlib.Path(__file__).resolve().parents[2] / "parity" / "meep_gpu"
            / "gate_cuda_three_slot_weld.py")
    assert gate.is_file(), f"{gate} is named as the owed gate and does not exist"
