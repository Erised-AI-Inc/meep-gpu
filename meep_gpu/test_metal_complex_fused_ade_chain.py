"""The COMPLEX fused E->P chain, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_complex_fused_ade_chain.py`` steps two
engines side by side for sixty complete seam steps and compares uint32 words — and
no assertion here duplicates that. What lives here is everything true about the
family WITHOUT a device: what the emitted source says, which configurations the
predicate refuses and by what name, that the arm is registered UNWIRED, and that
the signature respects the platform ceiling by ARITHMETIC.

THE TRANSCRIPTION TESTS PARSE, THEY DO NOT MIRROR. Each one CALLS the certified
emitter and searches its output; none re-implements a line. A test that re-derived
the arithmetic would mirror a defect instead of executing it, which is a failure
mode this project has measured (three planted assembly defects, 86 of 87 tests
still green).

THE ADE HALF NAMES NO COMPLEX ARM. Its complex body is MEEP's two real passes,
each product a real word times one part, with no complex helper; so whichever helper
body the E half is emitted with cannot disagree with it in the one fused scope. This
suite pins that absence rather than an arm clause.
"""

from __future__ import annotations

import pathlib
import re

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_complex_fused_ade_chain.py"

from meep_gpu.metal_kernels import (  # noqa: E402
    ade_update_p, arms, complex_fused_ade_chain as family,
    complex_no_pml_stored_e, fused_ade_chain, registry, templates,
)
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS  # noqa: E402

# ---------------------------------------------------------------------------
# 1. The source is a transcription of two certified bodies, and it PARSES as one
# ---------------------------------------------------------------------------

def test_the_complex_ade_recurrence_is_the_certified_bodys_own_lines():
    """The five lines that carry the complex polarization arithmetic, verbatim.

    ``ade_update_p.ade_source`` is CALLED; each line is searched for in its output
    and in this family's, with only the documented pointer renames applied.
    """
    certified = ade_update_p.ade_source("complex64", True)
    fused = family.complex_fused_ade_chain_source(2, (True, True))
    for line in ("float2 a = two_minus_od * p;",
                 "float2 b = gamma1 * q;",
                 "float2 sw = s * w;",
                 "float2 d = omega0dtsqr * sw;"):
        assert f"    {line}" in certified, f"the certified body's {line!r} moved"
        assert f"        {line}" in fused, (
            f"{line!r} is not the certified line; it is a re-derivation")
    store = "p_out[idx] = gamma1inv * ((a - b) + d);"
    assert f"    {store}" in certified
    for index in (0, 1):
        assert f"        {store.replace('p_out', f'o{index}')}" in fused, (
            f"pole {index}'s store is not the certified line with the documented "
            f"rename")


def test_the_shared_pole_pointer_is_the_fusion_and_it_is_bound_once():
    """``p_now`` is not bound a second time: pole k's ``P[c]`` IS the E chain's.

    That shared binding is the whole footprint of this fusion, so it is asserted
    from the SIGNATURE rather than described in prose: the emitted parameter list
    must declare ``p{k}`` once and never a ``p_now``.
    """
    fused = family.complex_fused_ade_chain_source(3, (False, False, False))
    signature = fused.split("uint idx [[thread_position_in_grid]]")[0]
    assert "p_now" not in signature
    for index in range(3):
        assert signature.count(f"p{index} ") == 1, (
            f"p{index} is declared {signature.count(f'p{index} ')} times")
        assert f"o{index}" in signature and f"q{index}" in signature
    # and the ADE half reads that same pointer for its `p`
    assert "float2 p = p0[idx];" in fused
    assert "float2 q = q0[idx];" in fused


def test_the_pole_chain_and_the_constitutive_are_the_certified_e_bodys_own():
    certified = complex_no_pml_stored_e.complex_stored_e_source(2)
    fused = family.complex_fused_ade_chain_source(2, (True, True))
    for line in ("    float2 source = d_in[idx];",
                 "    source = source - p0[idx];",
                 "    source = source - p1[idx];"):
        assert line in certified and line in fused


def test_the_one_rewritten_line_is_exactly_the_split_and_nothing_else():
    """The certified body stores the product unnamed; the fusion names it.

    The rewrite is asserted to BE the split — the certified text with
    ``e_out[idx] = `` replaced by ``float2 e = `` — so a future edit that changed
    the arithmetic while keeping the shape fails here.
    """
    certified = complex_no_pml_stored_e.complex_stored_e_source(1)
    stored = "    e_out[idx] = c_mul_field_left(source, inv_e[idx]);"
    assert stored in certified
    fused = family.complex_fused_ade_chain_source(1, (False,))
    named = stored.replace("e_out[idx] = ", "float2 e = ")
    assert named in fused, "the constitutive product is not the certified line split"
    assert "    e_out[idx] = e;" in fused
    assert stored not in fused, "the unsplit line is still present as well"


def test_the_seam_reads_a_register_where_the_certified_ade_read_a_volume():
    certified = ade_update_p.ade_source("complex64", False)
    assert "    float2 w = drive[idx];" in certified
    fused = family.complex_fused_ade_chain_source(1, (False,))
    assert "        float2 w = e;" in fused
    assert "drive[idx]" not in fused, "the drive volume is still being read"


def test_the_helper_block_is_the_certified_emitters_own_text():
    fused = family.complex_fused_ade_chain_source(1, (True,))
    assert templates.complex_helpers() in fused


# ---------------------------------------------------------------------------
# 2. The body is straight line, and only two helpers are actually CALLED
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("poles", (0, 1, 3, 6))
def test_the_body_is_straight_line_so_no_mutation_can_be_dead(poles):
    """Exactly one branch — the ``n_elem`` guard — and no ternary, no loop.

    THE DEAD-BRANCH MUTATION is a defect class this project has paid for: an edit
    can rewrite real lines the scored grid never reaches and report UNCAUGHT while
    measuring nothing. This family cannot have one, and that is asserted from the
    emitted TEXT rather than argued from the source layout.
    """
    flags = tuple((index % 2) == 0 for index in range(poles))
    source = family.complex_fused_ade_chain_source(poles, flags)
    assert len(re.findall(r"\bif\s*\(", source)) == 1
    assert "?" not in source
    assert not re.findall(r"\b(for|while)\s*\(", source)


def test_only_the_field_left_helper_is_called_which_is_why_no_needle_is_armed_elsewhere():
    """The other half of the dead-code guard, and it is a real one.

    The helper block is emitted VERBATIM from :func:`.templates.complex_helpers` —
    the certified emitters' own text — but only the E half's ``D * inv_eps`` calls a
    helper (``c_mul_field_left``): the ADE half's products are written inline, one
    per part, and an E->P chain has no complex-by-complex multiply. A mutation
    planted in an uncalled helper body would report UNCAUGHT while measuring nothing,
    which is why the gate arms none there.
    """
    source = family.complex_fused_ade_chain_source(2, (True, False))
    body = source.split("kernel void", 1)[1]
    assert "c_mul(" not in body, "c_mul is called after all; the gate must arm it"
    assert "c_mul_field_left(" in body
    assert "c_mul_coefficient_left(" not in body
    # and both ARE defined, so the absence above is about reachability not spelling
    assert "static inline float2 c_mul(float2 z, float2 p)" in source
    assert "static inline float2 c_mul_coefficient_left(float c, float2 z)" in source


# ---------------------------------------------------------------------------
# 3. The binding arithmetic, against the EMITTED signature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("poles", range(0, 7))
def test_binding_count_equals_the_emitted_attribute_count(poles):
    for volumes in (0, poles):
        flags = tuple(index < volumes for index in range(poles))
        expected = family.binding_count(poles, volumes)
        if expected > MAX_BUFFER_BINDINGS:
            continue
        source = family.complex_fused_ade_chain_source(poles, flags)
        indices = [int(m) for m in re.findall(r"\[\[buffer\((\d+)\)\]\]", source)]
        assert sorted(indices) == list(range(expected)), (
            f"{poles} poles / {volumes} volume sigmas: the emitted signature "
            f"occupies {sorted(indices)}, not 0..{expected - 1}")


def test_an_over_ceiling_signature_is_refused_by_name_not_left_to_the_compiler():
    poles = 7
    flags = (True,) * poles
    assert family.binding_count(poles, poles) > MAX_BUFFER_BINDINGS
    with pytest.raises(ValueError) as raised:
        family.complex_fused_ade_chain_source(poles, flags)
    assert "device.py:72" in str(raised.value)


def test_the_pole_slots_are_not_padded_to_eight():
    """The pad is what would break the ceiling; its absence is the design."""
    source = family.complex_fused_ade_chain_source(2, (False, False))
    assert "p2" not in source.split("uint idx")[0]


def test_the_packer_is_the_float32_familys_own_and_not_a_second_layout():
    """One measured layout, one implementation. A second is one more thing to drift."""
    assert family.pack_params is fused_ade_chain.pack_params
    assert family.params_words is fused_ade_chain.params_words


def test_the_ade_half_calls_no_complex_helper():
    """MEEP's DOCMP2 per-part products: no helper, no complex-by-complex product."""
    for volume in (False, True):
        certified = ade_update_p.ade_source("complex64", volume)
        assert "c_mul" not in certified
        assert templates.complex_helpers() not in certified
    fused = family.complex_fused_ade_chain_source(2, (True, False))
    ade_half = fused[fused.index("update_P for every pole"):]
    assert "c_mul" not in ade_half
    assert not hasattr(family, "ade_expansion_arm")


# ---------------------------------------------------------------------------
# 4. The ADE half names no complex arm
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. The predicate: what it refuses, and by what name
# ---------------------------------------------------------------------------

class _Fields:
    """A degenerate engine object. The predicate must REFUSE it, never raise."""

    grid = None
    polarizations = ()
    has_offdiagonal_epsilon = False
    has_nonlinearity = False


def _reasons(fields, pml=None):
    return family.complex_fused_ade_chain_coverage(fields, pml).reasons


def test_the_predicate_never_raises_on_a_degenerate_object():
    verdict = family.complex_fused_ade_chain_coverage(object(), None)
    assert verdict.covered is False and verdict.reasons


def test_both_halves_predicates_are_conjoined_and_labelled():
    reasons = _reasons(_Fields())
    assert any(reason.startswith("E half: ") for reason in reasons)
    assert any(reason.startswith("ADE half: ") for reason in reasons)


@pytest.mark.parametrize("flag", ("has_offdiagonal_epsilon", "has_nonlinearity"))
def test_the_interleave_clause_refuses_anything_that_couples_components(flag):
    """The seam clause is about the INTERLEAVE, not about the arm's arithmetic.

    Both features are already refused by the E half, and both are named again
    because the fact asserted here is different: the per-component interleave
    equals the driver's order only while ``update_E(c)`` reads component ``c``
    alone. That dependency is MEASURED in
    ``parity/meep_gpu/probe_metal_eop_interleave_dependency.py``.
    """
    fields = _Fields()
    setattr(fields, flag, True)
    assert any("per-component interleave is not the driver's order" in reason
               for reason in _reasons(fields))


def test_there_is_no_source_clause_and_the_absence_is_deliberate():
    """This is the ONE seam the driver injects nothing into (driver.py:3304-3306).

    Every fused product at the two curl seams must be told the source inventory.
    Requiring one here would refuse configurations the driver cannot break, so the
    predicate takes no ``sources`` argument at all — asserted from the SIGNATURE.
    """
    import inspect  # noqa: PLC0415
    parameters = inspect.signature(
        family.complex_fused_ade_chain_coverage).parameters
    assert "sources" not in parameters
    assert "THERE IS NO SOURCE CLAUSE" in (
        family.complex_fused_ade_chain_coverage.__doc__ or "")


# ---------------------------------------------------------------------------
# 6. Registration: the arm exists, is UNWIRED, and cannot be composed
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_the_first_slot_it_spans():
    registry_specs = [spec for spec in arms.registered()
                      if spec.family == family.FAMILY]
    assert len(registry_specs) == 1
    spec = registry_specs[0]
    assert spec.slot == family.SLOT == "update_E"
    assert spec.wired is False
    assert family.REPLACES == ("update_E", "update_P")


def test_an_unwired_arm_is_invisible_to_the_composer():
    """``arms_for`` filters on ``spec.wired``, so ``plan_step`` cannot select it."""
    class _Context:
        fields = _Fields()
        pml = None
        residency = None
        contract_variants = ()
        extra: dict = {}
        sources = ()

    # `_Arm` carries the LABEL, not the family, so the label is what is checked
    # here — and the registered spec's label is read from the table rather than
    # spelled a second time, so a rename cannot make this test pass vacuously.
    spec = next(s for s in arms.registered() if s.family == family.FAMILY)
    offered = {arm.label for arm in arms.arms_for("update_E", _Context())}
    assert spec.label not in offered
    # the floor: SOME wired arm is offered, or `arms_for` is simply empty here
    assert offered, "no arm at all is offered; this assertion would be vacuous"


def test_the_module_is_named_in_the_registry_so_it_is_not_invisible():
    """A family in the tree but not in ``FAMILY_MODULES`` is a silent coverage loss."""
    assert "complex_fused_ade_chain" in registry.FAMILY_MODULES


def test_the_device_gate_exists_and_names_this_family():
    assert GATE.exists()
    text = GATE.read_text()
    assert "complex_fused_ade_chain as family" in text
    assert "gate_provenance" in text, "the gate must stamp its imported bytes"
