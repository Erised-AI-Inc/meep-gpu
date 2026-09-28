"""The fused electric Metal kernel, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's -- ``parity/meep_gpu/gate_metal_fused_electric_pair.py`` steps two engines
side by side for twelve complete steps and compares uint32 words, and no assertion
here duplicates that. What lives here is everything true about the family WITHOUT a
device: what the emitted source says, which configurations the predicate refuses and
by what name, that the arm is registered UNWIRED, and that the plan binds the lattice
each half is supposed to take.

THE THREE CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. That the product is invisible to ``plan_step``'s DEFAULT composition -- a property
   of the TABLE, and the whole reason this family may land beside the wired tree
   without touching it.
2. That the wall-clear table is THE OFF-DIAGONAL. The gate measures that the shipped
   kernel is right; only a table-level assertion says that the B-side family's table
   is the WRONG one here, which is the single most likely slip in porting this
   product from its sibling.
3. That the SIGNATURE IS EXACTLY THE CEILING. This family binds 31 of the 31
   attributes the platform allows, so "it fits" and "one more pointer does not" are
   the same fact and the second half is the one a future edit will meet first.
   ``test_the_shipped_packed_signature_is_the_ceiling_itself`` states it as an
   equality rather than an inequality for that reason.
"""

from __future__ import annotations

import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_fused_electric_pair.py"

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, fused_dispersive_pair as d_side, fused_electric_pair as family,
    fused_magnetic_pair as sibling, registry, shaders, templates,
)


# ---------------------------------------------------------------------------
# 1. The source is a LIFT, not a transcription — so the lift is checkable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes", ((0, 0, 0), (1, 0, 1), (1, 1, 1)))
def test_the_curl_half_is_the_certified_curl_bodys_own_bytes(codes):
    """The certified ``step_D`` body appears VERBATIM, on both sides of the splice.

    This family does not retype either half: it lifts them from
    ``shaders.curl_source`` and ``shaders.constitutive_source``. That makes the
    transcription rule a substring check rather than a promise.
    """
    source = family.fused_electric_pair_source(codes, (False, False, False))
    body = family.certified_curl_body(codes)
    head, tail = body.split(family._CURL_STORE, 1)
    assert head in source
    assert (family._CURL_STORE + tail) in source


def test_the_curl_half_is_the_backward_direction():
    """``step_D`` differences DOWN. ``step_B``'s forward strides are another product."""
    source = family.fused_electric_pair_source((1, 1, 1), (True, True, True))
    assert "int si = i - 1, sj = j - 1, sk = k - 1;" in source
    assert "int si = i + 1, sj = j + 1, sk = k + 1;" not in source


def test_the_constitutive_half_differs_only_in_the_three_seam_lines():
    """The lift edits the SOURCE READ and nothing else.

    The certified E body opens each component with ``float srcN = gN[ii] * ieN[ii];``
    -- a reload of the displacement the curl just stored, scaled by the inverse
    permittivity. Fused, the reload is the register and the FACTOR DOES NOT MOVE.
    Any OTHER differing line would mean a rename reached the arithmetic.
    """
    certified = family.certified_constitutive_body().splitlines()
    source = family.fused_electric_pair_source((1, 1, 1), (True, True, True))
    marker = "    // --- update_E (stepping.update_E"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:]
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(spliced) == len(certified)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [(f"    float src{t} = g{t}[ii] * ie{t}[ii];",
                        f"    float src{t} = v{t} * ie{t}[ii];") for t in range(3)]


def test_the_displacement_stays_on_the_left_of_the_permittivity():
    """``stepping.py:1011-1013`` writes ``source * inverse_epsilon_for(component)``.

    Float multiplication is bitwise commutative and a transcription is not a place to
    rely on that, so the operand ORDER is pinned rather than the product.
    """
    source = family.fused_electric_pair_source((0, 0, 0), (False,) * 3)
    for target in range(3):
        assert f"float src{target} = v{target} * ie{target}[ii];" in source
        assert f"float src{target} = ie{target}[ii] * v{target};" not in source


def test_the_e_side_carries_its_inverse_epsilon_where_the_b_side_drops_it():
    """THE THREE POINTERS THAT MAKE THIS 30 AND THE MAGNETIC TWIN 27.

    ``update_H`` is ``H = B`` with mu = 1 baked into the array path, so
    :mod:`.fused_magnetic_pair` DROPS the three volumes the certified constitutive
    keeps in its signature. ``update_E`` cannot. A reader who assumed the sibling's
    layout would bind three buffers short.
    """
    source = family.fused_electric_pair_source((1, 1, 1), (True,) * 3)
    for target in range(3):
        assert f"device const float* ie{target}" in source
    magnetic = sibling.fused_magnetic_pair_source((1, 1, 1), (True,) * 3)
    assert "ie0" not in magnetic
    assert family.PACKED_BINDINGS - sibling.PACKED_BINDINGS == 3


def test_the_two_crossing_renames_both_landed_exactly_once():
    """``f`` -> ``e`` and ``e`` -> ``ie``, and neither may have run over the other.

    In the certified E body ``e0`` is the INVERSE EPSILON and ``f0`` is the TARGET;
    fused, the target must become ``e0`` and the inverse epsilon ``ie0``. Done in
    sequence rather than through a placeholder, the first rename's output would be
    the second rename's input and the kernel would multiply the target by itself --
    which still COMPILES, which is why this is asserted rather than left to the
    compiler.
    """
    body = family.certified_constitutive_body()
    assert "ie0[ii]" in body and "ie1[ii]" in body and "ie2[ii]" in body
    assert "float a0 = e0[ii];" in body and "e0[ii] = a0;" in body
    # The lift must leave NO f-target behind and no bare placeholder.
    assert "f0[ii]" not in body and "__INVERSE_EPSILON" not in body


def test_no_magnetic_pointer_survives_in_the_electric_half():
    """``g`` is H in the fused signature and was D in the certified E body.

    One surviving ``gN[`` read in the spliced half would take the magnetic field as a
    displacement -- a smooth, plausible, entirely wrong answer rather than a crash.
    """
    source = family.fused_electric_pair_source((1, 1, 1), (True,) * 3)
    marker = "    // --- update_E (stepping.update_E"
    electric = source.split(marker, 1)[1]
    assert not any(f"g{t}[" in electric for t in range(3))
    # ...and the CURL half still reads all three, or the seam moved the wrong line.
    curl = source.split(marker, 1)[0]
    assert all(f"g{t}[ii]" in curl for t in range(3))


def test_a_boundary_triple_that_is_not_a_triple_is_refused():
    with pytest.raises(ValueError):
        family.fused_electric_pair_source((1, 1), (True, True, True))
    with pytest.raises(ValueError):
        family.fused_electric_pair_source((1, 1, 1), (True, True))


# ---------------------------------------------------------------------------
# 2. zero_metal_D, carried inline — and it is THE OFF-DIAGONAL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("walled,expected", (
    ((True, False, False), (("v1", "at_x"), ("v2", "at_x"))),
    ((False, True, False), (("v0", "at_y"), ("v2", "at_y"))),
    ((False, False, True), (("v0", "at_z"), ("v1", "at_z"))),
))
def test_the_inline_wall_clear_is_the_d_yee_shift_table(walled, expected):
    """``_zero_metal`` clears a component only where its Yee shift is 0.

    ``IYEE_SHIFTS`` (fields.py:214-219) gives Dx (1,0,0), Dy (0,1,0), Dz (0,0,1), so
    one wall clears exactly TWO D components and they are the two that do NOT share
    the wall's axis. Getting this backwards clears the normal component instead of
    the tangential pair, which is a smooth, plausible, completely wrong boundary --
    never a crash.
    """
    emitted = family.zero_metal_mask(walled)
    lines = [line for line in emitted.splitlines() if "?" in line]
    assert len(lines) == 2, emitted
    for line, (register, flag) in zip(lines, expected):
        assert line.strip().startswith(register)
        assert flag in line


def test_the_d_table_is_the_complement_of_the_b_table():
    """THE PORT HAZARD, pinned. The B side is the DIAGONAL, D is the OFF-diagonal.

    Reusing :data:`.fused_magnetic_pair._ZERO_METAL_ROWS` here would clear the wrong
    component on every walled run. The two tables are asserted DISJOINT rather than
    merely different, because "different" would still pass if one row were shared.
    """
    electric = {(target, axis) for target, axis, _ in family._ZERO_METAL_ROWS}
    magnetic = {(target, axis) for target, axis, _ in sibling._ZERO_METAL_ROWS}
    assert electric == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}
    assert not (electric & magnetic), (electric, magnetic)
    assert electric | magnetic == {(t, a) for t in range(3) for a in range(3)}


def test_the_table_is_imported_and_not_re_derived():
    """One table, one home. The D-side rows are ``fused_dispersive_pair``'s own.

    A second copy would be a second place to get the complement wrong, and the two
    could drift without either file changing visibly.
    """
    assert family._ZERO_METAL_ROWS is d_side._ZERO_METAL_ROWS


def test_the_wall_clear_uses_the_selects_measured_zero_literal():
    """The same ``v ? 0.0f : v`` form the certified ownership mask emits.

    shaders.py rule 3: the ternary delivers the exact ``0.0`` a clamped load would
    not, and Metal's ``min``/``max`` builtins disagree with numpy on signed zeros.
    """
    emitted = family.zero_metal_mask((True, False, False))
    assert emitted.splitlines() == ["    v1 = at_x ? 0.0f : v1;",
                                    "    v2 = at_x ? 0.0f : v2;"]
    assert "? 0.0f :" in templates.ownership_mask((1, 1, 1), True)


def test_a_run_with_no_wall_emits_no_clear_at_all():
    """A periodic run must be byte-identical to one that never carried the pass.

    ``_zero_metal`` returns before touching a plane when nothing is metallic
    (stepping.py:2280-2286), so emitting a select that is always false would be
    correct arithmetic and a different kernel.
    """
    source = family.fused_electric_pair_source((0, 0, 0), (False, False, False))
    assert "? 0.0f : v0" not in source
    assert "no walled axis clears a D component" in source


def test_the_auxiliary_is_never_masked():
    """``zero_metal_D`` passes ``D_COMPONENTS`` (stepping.py:2250); ``fu_D`` is not
    in it, so masking the split-field auxiliary would be an over-carry."""
    source = family.fused_electric_pair_source((1, 1, 1), (True, True, True))
    assert "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;" in source
    assert "u0[ii] = at_y" not in source


# ---------------------------------------------------------------------------
# 3. The seam predicate
# ---------------------------------------------------------------------------

class _Source:
    """The smallest thing the predicate reads: a declared ``field_type``."""

    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def test_an_undeclared_source_set_is_refused_by_name():
    """IGNORANCE IS NOT AN EMPTY SET. ``Fields`` does not hold the source list, so
    inferring "no sources" from not being told is the over-covering refusal this
    clause exists to prevent."""
    coverage = family.metal_fused_electric_pair_coverage(object(), None, None)
    assert not coverage.covered
    assert any("was not declared" in reason for reason in coverage.reasons)


def test_an_electric_source_is_carried_and_a_magnetic_one_is_not_this_seam():
    """THE POLARITY, and on this seam the carry is the product rather than a widening.

    An electric current lands BETWEEN ``step_D`` and ``update_E``
    (driver.py:3294-3299); a magnetic one lands in the other half entirely.
    ``CARRIES_DEPOSIT_REPAIR`` is True, so the seam clause returns only what the
    REPAIR cannot reconstruct. On the degenerate ``object()`` fields used here that
    is the missing ``f_w_E`` storage and the absent deposit index -- both refusals
    ABOUT THE REPAIR, which is what distinguishes this from a blanket clause.
    """
    electric = family.metal_fused_electric_pair_coverage(
        object(), None, (_Source("D"),))
    assert not any("is electric" in reason and "driver.py:3294-3299" in reason
                   for reason in electric.reasons), electric.reasons
    assert any("the deposit repair cannot carry this seam" in reason
               for reason in electric.reasons), electric.reasons
    magnetic = family.metal_fused_electric_pair_coverage(
        object(), None, (_Source("B"),))
    assert not any("source 0" in reason for reason in magnetic.reasons)


def test_the_polarity_is_the_opposite_of_the_b_side_pairs():
    """The two plain fused pairs must not both refuse the same source, or one of them
    has copied the other's clause rather than read the driver."""
    b_side = sibling.metal_fused_magnetic_pair_coverage(
        object(), None, (_Source("D"),))
    assert not any("source 0" in reason for reason in b_side.reasons)


def test_the_flag_and_the_installer_move_together():
    """THE CLAIM THE FLAG MAKES, checked against the composer that must honour it.

    ``deposit_repair`` is explicit that True without the wrappers is the exact defect
    it exists to prevent. On this track the wrappers are built by
    ``launch._install_fused_pairs``, and the families that loop can reach are bounded
    by ``launch.FUSED_PAIR_ARMS`` -- so the flag is only honest while this family
    holds a row there, naming the two arms its own predicate is built out of.
    """
    from meep_gpu.metal_kernels import launch as metal_launch

    assert family.CARRIES_DEPOSIT_REPAIR is True
    assert metal_launch.FUSED_PAIR_ARMS[family.FAMILY] == ("PML", "ordinary")
    assert metal_launch.FUSED_PAIR_SEAMS[family.SLOT] == ("update_E", "D")


def test_the_predicate_never_raises_on_a_degenerate_object():
    """``plan_step``'s whole contract is that a refusal is a verdict, not an
    exception: a raising predicate would escape the composer's arm loop."""
    coverage = family.metal_fused_electric_pair_coverage(object(), None, ())
    assert not coverage.covered and coverage.reasons


def test_both_halves_predicates_are_conjoined_and_labelled():
    """A configuration either half refuses is refused here with THAT half's reasons,
    prefixed, so a reader can tell which side spoke."""
    coverage = family.metal_fused_electric_pair_coverage(object(), None, ())
    prefixes = {reason.split(":")[0] for reason in coverage.reasons}
    assert "curl half" in prefixes
    assert "constitutive half" in prefixes


# ---------------------------------------------------------------------------
# 4. The table: registered, enumerable, and INVISIBLE to plan_step
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_exactly_one_slot():
    rows = [spec for spec in arms.registered() if spec.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT == "step_D"
    assert rows[0].wired is False, (
        "this product spans step_D, zero_metal_D and update_E; wiring it would "
        "hand plan_step a slot assignment no composition rule has measured")


def test_plan_step_cannot_select_it():
    """``arms_for`` is the gate between the table and the composer.

    An unwired row stays ENUMERABLE -- which is what lets a disjointness sweep see it
    -- while never reaching ``_select_slot``.
    """
    context = arms.StepContext(object(), None, None, (), sources=())
    labels = [arm.label for arm in arms.arms_for(family.SLOT, context)]
    assert "fused electric D/E pair" not in labels
    assert "fused electric D/E pair" in [
        spec.label for spec in arms.registered(family.SLOT)]


def test_the_family_is_in_family_modules():
    """A family in the tree but not in ``registry`` is invisible to ``plan_step``,
    which is a silent coverage loss rather than an error."""
    assert family.FAMILY in registry.FAMILY_MODULES


def test_the_plan_declares_every_pass_it_performs():
    """``replaces_sub_steps`` is READ by the composer and by the whole-step walk.

    A plan that under-declared would have the seam passes it actually performs
    bracketed with a host sync, which is a stale mirror rather than a wrong number.
    The two FILLS are absent on purpose: both return before touching a cell unless
    ``grid.has_symmetry()``, and both halves refuse every folded grid.
    """
    assert family.REPLACES == ("step_D", "zero_metal_D", "update_E")
    assert family.MetalFusedElectricPairPlan.replaces_sub_steps == family.REPLACES
    assert family.MetalFusedElectricPairPlan.launches_per_run == 1
    assert family.MetalFusedElectricPairPlan.performs_device_work is True
    assert "fill_D" not in family.REPLACES
    assert "fill_folded_far_ghosts_D" not in family.REPLACES


# ---------------------------------------------------------------------------
# 5. The platform ceiling, which for this family IS the signature
# ---------------------------------------------------------------------------

def test_the_refuted_separate_scalar_signature_really_does_break_the_ceiling():
    """The 35-binding count is spelled once and the emitter agrees with it.

    Whether it FAILS TO COMPILE is a device measurement and belongs to the gate's
    ``binding_ceiling`` leg; what belongs here is that the signature this suite and
    that leg talk about is the same one, and that it is over the documented limit.
    """
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.refuted_separate_scalar_source()
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.SEPARATE_SCALAR_BINDINGS == 35
    assert family.SEPARATE_SCALAR_BINDINGS > MAX_BUFFER_BINDINGS


def test_the_shipped_packed_signature_is_the_ceiling_itself():
    """30 pointers plus one packed struct is 31 -- EQUAL to the ceiling, not under it.

    Stated as an equality on purpose. "It fits" would still be true with headroom and
    would go on being true after an edit that spent the last slot; this assertion
    fails the moment the shape moves in either direction, which is the only way a
    family with zero headroom can be pinned.
    """
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.fused_electric_pair_source((1, 1, 1), (True, True, True))
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.PACKED_BINDINGS == 31
    assert family.PACKED_BINDINGS == MAX_BUFFER_BINDINGS


@pytest.mark.parametrize("codes", ((0, 0, 0), (1, 0, 1), (1, 1, 1)))
@pytest.mark.parametrize("walls", ((False, False, False), (True, True, True)))
def test_every_specialisation_binds_exactly_thirty_pointers(codes, walls):
    """The count must not depend on the boundary triple or the wall set.

    A specialisation that emitted one more pointer on some grid would compile
    everywhere else and fail on that one, which is the failure a family sitting
    exactly on the ceiling is most exposed to.
    """
    import re

    source = family.fused_electric_pair_source(codes, walls)
    pointers = set(re.findall(
        r"device\s+(?:const\s+)?\w+\s*\*\s*\w+\s*\[\[buffer\((\d+)\)\]\]", source))
    scalars = set(re.findall(
        r"constant\s+(?:const\s+)?[\w:]+\s*&\s*\w+\s*\[\[buffer\((\d+)\)\]\]", source))
    assert len(pointers) == 30
    assert len(scalars) == 1


def test_the_contraction_guard_is_emitted_in_both_modes():
    """The guard is a property of the SOURCE on this backend, so each mode is a
    different compiled kernel and a plan built without one must raise rather than
    launch the pinned one."""
    off = family.fused_electric_pair_source((1, 1, 1), (True,) * 3,
                                            shaders.CONTRACT_OFF)
    fast = family.fused_electric_pair_source((1, 1, 1), (True,) * 3,
                                             shaders.CONTRACT_FAST)
    assert "contract(off)" in off
    assert off != fast


# ---------------------------------------------------------------------------
# 6. The gate exists, and the bytes it certified are these bytes
# ---------------------------------------------------------------------------

def test_the_device_gate_is_present_and_routes_through_the_runner():
    assert GATE.is_file()
    text = GATE.read_text(encoding="utf-8")
    assert "run_current_measurement" in text, (
        "a Metal gate must run through metal_gate_runner so its artifact records "
        "the sources the PROCESS imported")
    assert "gate_provenance" in text
