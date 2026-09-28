"""The COMPLEX no-absorber THREE-SLOT weld: the lift, the predicate, the arbitration.

WHAT THIS FILE CAN AND CANNOT SAY. The weld's ARITHMETIC is a device claim --
``parity/meep_gpu/gate_cuda_complex_three_slot_weld.py`` carries it. What a laptop
can pin is everything whose failure would be SILENT before a device is reached:

1. **The lift.** The kernel is three certified bodies spliced -- one component block
   of the D->E weld's certified curl body, that weld's constitutive store
   value-named, and the E->P weld's per-pole recurrence under its own seam -- each
   read off the module that owns it, so a drift raises here instead of emitting a
   body that is quietly not the certified arithmetic.
2. **The predicate.** A conjunction of the two shipped ones, so its admission set is
   a SUBSET of each half's; measured in both directions.
3. **The wiring.** ``_install_fused_triple`` puts the leading half in ``step_D``, a
   ``NoopPlan`` in ``update_E`` (the seam is empty by predicate) and the trailing
   half in ``update_P``; the trailing half answers the consult and launches nothing.
4. **The arbitration.** The weld supersedes the D->E pair by strict containment and
   the E->P pair is failed closed out by ``_pair_may_absorb``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ..triton_kernels.launch import NoopPlan
from . import arms, fused_pairs, registry
from .registry import LICENSE_COMPLEX, LICENSE_COMPLEX_NO_PML
from . import complex_no_pml_fused_polarization_pair as ep
from . import complex_no_pml_three_slot_dispersive_weld as family
from . import no_pml_complex_fused_electric_pair as de
from .test_no_pml_complex_fused_electric_pair import (LICENCE, POLICY,  # noqa: F401
                                                       build, xp)


# ---------------------------------------------------------------------------
# 1. THE LIFT
# ---------------------------------------------------------------------------

def test_every_digest_source_is_ascii_and_declares_one_entry_point():
    sources = family.device_sources()
    assert len(sources) == 60, "the digest sweep collapsed"
    for key, source in sources.items():
        assert source.isascii(), key
        assert source.count('extern "C" __global__') == 1, key
        symbol = key.split("|")[0]
        assert source.count(f"void {symbol}(") == 1, key


@pytest.mark.parametrize("conductive", [False, True])
@pytest.mark.parametrize("component", [0, 1, 2])
def test_exactly_one_component_block_is_lifted_and_it_is_the_certified_one(
        conductive, component):
    """The D->E weld's curl body has three braced component blocks; this launch
    carries ONE, character for character, and never another's capture."""
    source = family.three_slot_no_pml_complex_source("FMA_V1", conductive, component,
                                                     1, (True,))
    header, block = family.certified_curl_block("FMA_V1", conductive, component)
    assert block in source
    assert source.count("\n    // Target ") == 1
    whole = de.certified_curl_body("FMA_V1", conductive)
    assert block in whole and header in whole
    tail = "conductive_apply_reg" if conductive else "no_pml_apply_reg"
    assert source.count(f"d{component} = {tail}(f{component}, idx, curl") == 1
    body = source.split(") {", 1)[1]
    for other in range(3):
        if other != component:
            # The header still DECLARES all three registers (it is lifted whole);
            # what may not survive is another block's CAPTURE.
            assert f"        d{other} = " not in body


def test_the_constitutive_store_is_the_d_to_e_welds_line_value_named():
    """Same expression tree, same store; the value is NAMED so the recurrence can
    read it from a register -- the E->P weld's one edit, on the D->E weld's line."""
    source = family.three_slot_no_pml_complex_source("NAIVE", True, 1, 2, (True, False))
    assert ("    cf ev = mul_field_left(\n"
            "        minus_poles_reg(d1, b0, b1, b2, b3, b4, b5, b6, b7, idx, np1),"
            "   // THE SEAM\n"
            "        inv_eps_1[idx]);\n"
            "    cf_store(h1, idx, ev);\n") in source
    # ...and the certified body really does carry the unnamed form.
    body = de.certified_constitutive_body("NAIVE")
    assert "    cf_store(h1, idx, mul_field_left(\n" in body


def test_the_recurrence_is_the_e_to_p_welds_seam_per_pole():
    """The drive load is the register ``ev`` and ``p_now`` is the bank slot the
    constitutive already reads through -- the E->P weld's own substitutions."""
    source = family.three_slot_no_pml_complex_source("FMA_V1", False, 2, 2,
                                                     (True, False))
    assert ("    cf_store(p_out_0, idx, ade_step(\n"
            "        cf_load(c0, idx), cf_load(p_prev_0, idx), ev,\n"
            "        sigma_0[idx], c_now_0, c_prev_0, c_drive_0));") in source
    assert ("    cf_store(p_out_1, idx, ade_step(\n"
            "        cf_load(c1, idx), cf_load(p_prev_1, idx), ev,\n"
            "        sigma_1, c_now_1, c_prev_1, c_drive_1));") in source
    assert "cf_load(drive, idx)" not in source and "p_now" not in source.split(
        ") {", 1)[1]
    twin = ep.fused_polarization_pair_no_pml_complex_source("FMA_V1", 2, 2,
                                                           (True, False))
    assert source.split("// pole 0")[1] == twin.split("// pole 0")[1]


def test_a_zero_pole_component_emits_the_curl_and_the_constitutive_only():
    source = family.three_slot_no_pml_complex_source("FMA_V1", True, 0, 0, ())
    assert "// pole 0" not in source and "p_out_" not in source
    assert "cf_store(h0, idx, ev);" in source
    assert "int np0\n" in source


def test_a_drifted_certified_body_raises_instead_of_emitting(monkeypatch):
    """A certified string that moves must fail HERE, on a laptop."""
    original = de.certified_constitutive_body

    def drifted(arm):
        return original(arm).replace("   // THE SEAM", "")
    monkeypatch.setattr(de, "certified_constitutive_body", drifted)
    with pytest.raises(AssertionError, match="LIFTS that line"):
        family.three_slot_no_pml_complex_source("FMA_V1", True, 0, 1, (True,))


def test_a_curl_body_with_the_wrong_block_count_is_refused(monkeypatch):
    original = de.certified_curl_body

    def extra(arm, conductive):
        body = original(arm, conductive)
        return body + "\n    // Target 3: nothing\n    {\n    }\n"
    monkeypatch.setattr(de, "certified_curl_body", extra)
    with pytest.raises(AssertionError, match="component blocks"):
        family.certified_curl_block("FMA_V1", True, 0)


def test_the_kernel_symbols_and_the_partition_agree():
    assert set(family.KERNEL_KEYS.values()) == set(family.CERTIFIED_KERNELS)
    assert family.UNCERTIFIED_KERNELS == {}
    assert family.KERNEL_OWNER_MODULE == "complex_no_pml_three_slot_dispersive_weld"
    with pytest.raises(ValueError, match="arm must be one of"):
        family.kernel_name("split_field")


def test_the_lift_edits_are_three_and_name_the_two_seams():
    assert len(family.LIFT_EDITS) == 3
    text = " ".join(row["why"] for row in family.LIFT_EDITS)
    assert "register" in text and "bound ONCE" in text


# ---------------------------------------------------------------------------
# 2. THE PREDICATE
# ---------------------------------------------------------------------------

def _plan(fields, layer, grid, *, fuse):
    """``arms.plan_step`` with the two licences the complex arms bind through."""
    return arms.plan_step(fields, layer, grid,
                          licenses={LICENSE_COMPLEX_NO_PML: LICENCE,
                                    LICENSE_COMPLEX: LICENCE},
                          subnormal_policy=POLICY, sources=(), fuse=fuse)


def _fixture(xp, **kwargs):
    """The D->E pair's own fixture: complex64, Bloch-phased, periodic, no absorber,
    anisotropic epsilon, a graded D and B conductivity, ``poles`` states driving all
    three components -- the TestLoadDump 3-D shape."""
    return build(xp, **kwargs)


def test_the_weld_admits_its_cells_configuration(xp):
    """The TestLoadDump 3-D configuration: complex64, Bloch, periodic, no absorber, a
    D and a B conductivity, one pole on every component, an EMPTY electric seam."""
    fields, layer, grid = _fixture(xp, conductive=True)
    covered, why = family.covers_three_slot_complex_no_pml_dispersive_weld(
        fields, layer, grid, (), LICENCE, POLICY)
    assert covered, why


@pytest.mark.parametrize("conductive", [False, True])
def test_the_admission_set_is_a_subset_of_both_halves(xp, conductive):
    fields, layer, grid = _fixture(xp, conductive=conductive)
    ours = family.covers_three_slot_complex_no_pml_dispersive_weld(
        fields, layer, grid, (), LICENCE, POLICY)[0]
    d_to_e = de.covers_no_pml_complex_fused_electric_pair(
        fields, layer, grid, (), LICENCE, POLICY)[0]
    e_to_p = ep.covers_complex_no_pml_fused_polarization_pair(
        fields, layer, grid, (), LICENCE, POLICY)[0]
    assert ours is True and d_to_e is True and e_to_p is True


def test_a_halves_refusal_arrives_prefixed_with_the_side_that_said_it(xp):
    fields, layer, grid = _fixture(xp, conductive=True)
    covered, why = family.covers_three_slot_complex_no_pml_dispersive_weld(
        fields, layer, grid, None, LICENCE, POLICY)
    assert not covered and why.startswith("step_D->update_E half: ")
    assert "not declared" in why


def test_an_electric_in_seam_source_is_refused_by_the_d_to_e_half(xp):
    """THE CLAUSE THAT LICENSES THE SINGLE LAUNCH: no deposit may sit inside the span."""
    fields, layer, grid = _fixture(xp, conductive=True)

    class _Electric:
        field_type = "D"
    covered, why = family.covers_three_slot_complex_no_pml_dispersive_weld(
        fields, layer, grid, (_Electric(),), LICENCE, POLICY)
    assert not covered and "electric" in why and "step_D->update_E half" in why

    class _Magnetic:
        field_type = "B"
    covered, why = family.covers_three_slot_complex_no_pml_dispersive_weld(
        fields, layer, grid, (_Magnetic(),), LICENCE, POLICY)
    assert covered, why


def test_a_run_with_no_polarization_is_refused_by_the_e_to_p_half(xp):
    fields, layer, grid = _fixture(xp, conductive=True, poles=0)
    covered, why = family.covers_three_slot_complex_no_pml_dispersive_weld(
        fields, layer, grid, (), LICENCE, POLICY)
    assert not covered and why.startswith("update_E->update_P half: ")


def test_both_real_three_slot_welds_refuse_this_configuration(xp):
    from . import no_pml_three_slot_dispersive_weld as real_no_pml  # noqa: PLC0415
    from . import three_slot_dispersive_weld as real_pml  # noqa: PLC0415

    fields, layer, grid = _fixture(xp, conductive=True)
    assert not real_no_pml.covers_no_pml_three_slot_dispersive_weld(
        fields, layer, grid, ())[0]
    assert not real_pml.covers_three_slot_dispersive_weld(fields, layer, grid, ())[0]


def test_the_module_declares_no_repair_and_the_wiring_agrees():
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.REPLACES == ("step_D", "update_E", "update_P")
    assert family.SLOTS == family.REPLACES and family.SLOT == "step_D"
    assert not hasattr(family, "REPAIR_PATHS")


# ---------------------------------------------------------------------------
# 3. THE COMPOSER
# ---------------------------------------------------------------------------

def test_the_product_row_and_module_constants_agree():
    row = fused_pairs.FUSED_PRODUCTS[family.FAMILY]
    assert row["curl_slot"] == "step_D"
    assert tuple(row["slots"]) == family.SLOTS
    assert row["module"] == "complex_no_pml_three_slot_dispersive_weld"
    assert fused_pairs.span_of(family.FAMILY, row) == family.SLOTS


def test_the_absorb_declaration_matches_the_arms_the_predicate_conjoins():
    by_predicate = {row["name"]: row["label"] for row in registry._TABLE}
    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY] == (
        by_predicate["covers_complex_no_pml_curl"],
        by_predicate["covers_complex_no_pml_stored_e"],
        by_predicate["covers_complex_no_pml_ade_update_p"])
    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY][:2] == (
        fused_pairs.FUSED_PAIR_ARMS["cuda_no_pml_complex_fused_electric_pair"])
    assert fused_pairs.FUSED_PAIR_ARMS[family.FAMILY][1:] == (
        fused_pairs.FUSED_PAIR_ARMS["cuda_complex_no_pml_fused_polarization_pair"])


def test_the_weld_takes_all_three_slots_and_names_what_it_superseded(xp):
    fields, layer, grid = _fixture(xp, conductive=True)
    plan = _plan(fields, layer, grid, fuse=True)
    label = "complex no-absorber three-slot weld"
    assert {plan.selected.get(s) for s in family.SLOTS} == {label}
    assert isinstance(plan.plans["update_E"], NoopPlan)
    assert type(plan.plans["step_D"]).__name__ == "_TripleHalfPlan"
    assert type(plan.plans["update_P"]).__name__ == "_TripleHalfPlan"
    d_reason = " ".join(plan.reasons.get(
        "fused_pair_cuda_no_pml_complex_fused_electric_pair", ()))
    assert "strictly contains" in d_reason
    e_reason = " ".join(plan.reasons.get(
        "fused_pair_cuda_complex_no_pml_fused_polarization_pair", ()))
    assert "was selected by" in e_reason or "selected" in e_reason
    assert fused_pairs.replaced_sub_steps(plan.plans)[-3:] == family.REPLACES[-3:]


def test_the_supersession_does_not_fire_where_the_weld_does_not_admit(xp):
    fields, layer, grid = _fixture(xp, conductive=True, poles=0)
    plan = _plan(fields, layer, grid, fuse=True)
    assert plan.selected.get("step_D") == "complex no-absorber fused electric pair"
    assert "fused_pair_" + family.FAMILY in plan.reasons


def test_the_trailing_half_answers_the_consult_without_launching():
    plan = fused_pairs._complex_no_pml_three_slot_weld_plan(_Context())
    assert plan.launchable
    report = plan.launch_trailing(None, {})
    assert report["launched"] is False and report["launches"] == 0
    assert report["advanced_in"] == "leading"


class _Context:
    fields = pml = grid = sources = None
    subnormal_policy = POLICY

    @staticmethod
    def license_for(_name):
        return LICENCE


def test_fusion_is_opt_in_and_the_default_composition_is_unchanged(xp):
    fields, layer, grid = _fixture(xp, conductive=True)
    plan = _plan(fields, layer, grid, fuse=False)
    assert plan.selected.get("step_D") == "complex no-PML curl"
    assert plan.selected.get("update_P") == "complex no-PML ADE"


def test_the_gate_this_product_owes_exists_on_disk():
    gate = (pathlib.Path(__file__).resolve().parents[2] / "parity" / "meep_gpu"
            / "gate_cuda_complex_three_slot_weld.py")
    assert gate.is_file()
    assert "complex_no_pml_three_slot_dispersive_weld" in gate.read_text()
