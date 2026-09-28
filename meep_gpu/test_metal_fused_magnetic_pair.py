"""The fused magnetic Metal kernel, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_fused_magnetic_pair.py`` steps two engines
side by side for twelve complete steps and compares uint32 words, and no assertion
here duplicates that. What lives here is everything true about the family WITHOUT
a device: what the emitted source says, which configurations the predicate refuses
and by what name, that the arm is registered UNWIRED, and that the plan binds the
lattice each half is supposed to take.

THE TWO CLAIMS THIS SUITE MAKES THAT THE GATE CANNOT.

1. That the product is invisible to ``plan_step``'s DEFAULT composition — a
   property of the TABLE, and the whole reason this family may land beside the
   wired tree without touching it. ``plan_step(fuse=True)`` is the one path that
   can reach it, opt-in and off by default; what it composes, and the two-slot
   protocol it composes it with, are pinned in
   ``test_metal_fused_pair_deposit_wiring.py``.
2. That the wall-clear table is THE DIAGONAL. The gate measures that the shipped
   kernel is right; only a table-level assertion says that the D-side family's
   table is the WRONG one here, which is the single most likely slip in porting
   this product from its sibling.
"""

from __future__ import annotations

import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_fused_magnetic_pair.py"

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, fused_dispersive_pair as sibling, fused_magnetic_pair as family,
    registry, shaders, templates,
)


# ---------------------------------------------------------------------------
# 1. The source is a LIFT, not a transcription — so the lift is checkable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("codes", ((0, 0, 0), (1, 0, 1), (1, 1, 1)))
def test_the_curl_half_is_the_certified_curl_bodys_own_bytes(codes):
    """The certified ``step_B`` body appears VERBATIM, on both sides of the splice.

    This family does not retype either half: it lifts them from
    ``shaders.curl_source`` and ``shaders.constitutive_source``. That makes the
    transcription rule a substring check rather than a promise.
    """
    source = family.fused_magnetic_pair_source(codes, (False, False, False))
    body = family.certified_curl_body(codes)
    head, tail = body.split(family._CURL_STORE, 1)
    assert head in source
    assert (family._CURL_STORE + tail) in source


def test_the_curl_half_is_the_forward_direction():
    """``step_B`` differences UP. ``step_D``'s negated strides are another product."""
    source = family.fused_magnetic_pair_source((1, 1, 1), (True, True, True))
    assert "int si = i + 1, sj = j + 1, sk = k + 1;" in source
    assert "int si = i - 1, sj = j - 1, sk = k - 1;" not in source


def test_the_constitutive_half_differs_only_in_the_three_seam_lines():
    """The lift edits the SOURCE READ and nothing else.

    The certified H body opens each component with ``float srcN = gN[ii];`` — a
    reload of the flux density the curl just stored. Fused, that is the register.
    Any OTHER differing line would mean a rename reached the arithmetic.
    """
    certified = family.certified_constitutive_body().splitlines()
    source = family.fused_magnetic_pair_source((1, 1, 1), (True, True, True))
    marker = "    // --- update_H (stepping.update_H"
    spliced = [line for line in source.split(marker, 1)[1].splitlines()
               if "// THE SEAM:" not in line]
    spliced = spliced[1:]
    while spliced and spliced[-1].strip() in ("", "}"):
        spliced.pop()
    assert len(spliced) == len(certified)
    changed = [(a, b) for a, b in zip(certified, spliced) if a != b]
    assert changed == [(f"    float src{t} = g{t}[ii];", f"    float src{t} = v{t};")
                       for t in range(3)]


def test_the_h_side_carries_no_inverse_mu():
    """mu = 1 is baked into the array path too (``update_H`` passes ``fields.Bx``).

    The certified constitutive kernel keeps three inverse-epsilon pointers on BOTH
    sides so its buffer layout is one layout; this family serves the H side only
    and DROPS them, which is the whole reason the signature is 27 pointers and not
    30. A reader who assumed the sibling's layout would bind three buffers the
    kernel does not declare.
    """
    source = family.fused_magnetic_pair_source((1, 1, 1), (True, True, True))
    assert "inv_e" not in source and "ie0" not in source
    assert "device const float* e0" not in source


def test_a_boundary_triple_that_is_not_a_triple_is_refused():
    with pytest.raises(ValueError):
        family.fused_magnetic_pair_source((1, 1), (True, True, True))
    with pytest.raises(ValueError):
        family.fused_magnetic_pair_source((1, 1, 1), (True, True))


# ---------------------------------------------------------------------------
# 2. zero_metal_B, carried inline — and it is THE DIAGONAL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("walled,expected", (
    ((True, False, False), ("v0", "at_x")),
    ((False, True, False), ("v1", "at_y")),
    ((False, False, True), ("v2", "at_z")),
))
def test_the_inline_wall_clear_is_the_b_yee_shift_table(walled, expected):
    """``_zero_metal`` clears a component only where its Yee shift is 0.

    ``IYEE_SHIFTS`` (fields.py:216-217) gives Bx (0,1,1), By (1,0,1), Bz (1,1,0),
    so one wall clears exactly ONE B component and it is the one that shares the
    wall's axis. Getting this backwards clears the tangential pair instead of the
    normal component, which is a smooth, plausible, completely wrong boundary —
    never a crash.
    """
    emitted = family.zero_metal_mask(walled)
    lines = [line for line in emitted.splitlines() if "?" in line]
    assert len(lines) == 1, emitted
    assert lines[0].strip().startswith(expected[0])
    assert expected[1] in lines[0]


def test_the_b_table_is_the_complement_of_the_d_table():
    """THE PORT HAZARD, pinned. The D side is the OFF-diagonal, B is the DIAGONAL.

    Reusing :data:`.fused_dispersive_pair._ZERO_METAL_ROWS` here would clear the
    wrong two components on every walled run. The two tables are asserted DISJOINT
    rather than merely different, because "different" would still pass if one row
    were shared.
    """
    magnetic = {(target, axis) for target, axis, _ in family._ZERO_METAL_ROWS}
    electric = {(target, axis) for target, axis, _ in sibling._ZERO_METAL_ROWS}
    assert magnetic == {(0, 0), (1, 1), (2, 2)}
    assert not (magnetic & electric), (magnetic, electric)
    assert magnetic | electric == {(t, a) for t in range(3) for a in range(3)}


def test_the_wall_clear_uses_the_selects_measured_zero_literal():
    """The same ``v ? 0.0f : v`` form the certified ownership mask emits.

    shaders.py rule 3: the ternary delivers the exact ``0.0`` a clamped load would
    not, and Metal's ``min``/``max`` builtins disagree with numpy on signed zeros.
    """
    emitted = family.zero_metal_mask((True, False, True))
    assert emitted.splitlines() == ["    v0 = at_x ? 0.0f : v0;",
                                    "    v2 = at_z ? 0.0f : v2;"]
    assert "? 0.0f :" in templates.ownership_mask((1, 1, 1), False)


def test_a_run_with_no_wall_emits_no_clear_at_all():
    """A periodic run must be byte-identical to one that never carried the pass.

    ``_zero_metal`` returns before touching a plane when nothing is metallic
    (stepping.py:2280-2286), so emitting a select that is always false would be
    correct arithmetic and a different kernel.
    """
    source = family.fused_magnetic_pair_source((0, 0, 0), (False, False, False))
    assert "? 0.0f : v0" not in source
    assert "no walled axis clears a B component" in source


def test_the_auxiliary_is_never_masked():
    """``zero_metal_B`` passes ``B_COMPONENTS`` (stepping.py:2250); ``fu_B`` is not
    in it, so masking the split-field auxiliary would be an over-carry."""
    source = family.fused_magnetic_pair_source((1, 1, 1), (True, True, True))
    assert "    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;" in source
    assert "u0[ii] = at_x" not in source


# ---------------------------------------------------------------------------
# 3. The seam predicate — the binding clause of this family
# ---------------------------------------------------------------------------

class _Source:
    """The smallest thing the predicate reads: a declared ``field_type``."""

    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def test_an_undeclared_source_set_is_refused_by_name():
    """IGNORANCE IS NOT AN EMPTY SET. ``Fields`` does not hold the source list, so
    inferring "no sources" from not being told is the over-covering refusal this
    clause exists to prevent."""
    coverage = family.metal_fused_magnetic_pair_coverage(object(), None, None)
    assert not coverage.covered
    assert any("was not declared" in reason for reason in coverage.reasons)


def test_a_magnetic_source_is_carried_and_an_electric_one_is_not_this_seam():
    """THE POLARITY, and it is the whole reason this seam is worth more than the
    D-side one. A magnetic current lands BETWEEN ``step_B`` and ``update_H``
    (driver.py:3283-3284); an electric one lands in the other half entirely.

    THE MAGNETIC SIDE IS NOW CARRIED, NOT REFUSED. ``CARRIES_DEPOSIT_REPAIR`` is
    True, so the seam clause stops returning the flat "is magnetic" refusal and
    returns instead only what the REPAIR cannot reconstruct. On the degenerate
    ``object()`` fields used here that is the missing ``f_w_H`` storage and the
    absent deposit index -- both of which are refusals ABOUT THE REPAIR, which is
    what distinguishes this from the old blanket clause. An electric source still
    contributes nothing either way.
    """
    magnetic = family.metal_fused_magnetic_pair_coverage(
        object(), None, (_Source("B"),))
    assert not any("is magnetic" in reason and "driver.py:3283-3284" in reason
                   for reason in magnetic.reasons), magnetic.reasons
    assert any("the deposit repair cannot carry this seam" in reason
               for reason in magnetic.reasons), magnetic.reasons
    assert any("does not publish the index it writes" in reason
               for reason in magnetic.reasons), magnetic.reasons
    electric = family.metal_fused_magnetic_pair_coverage(
        object(), None, (_Source("D"),))
    assert not any("source 0" in reason for reason in electric.reasons)


def test_the_flag_and_the_installer_move_together():
    """THE CLAIM THE FLAG MAKES, checked against the composer that must honour it.

    ``deposit_repair`` is explicit that True without the wrappers is the exact defect
    it exists to prevent. On this track the wrappers are built by
    ``launch._install_fused_pairs``, and the families that loop can reach are bounded
    by ``launch.FUSED_PAIR_ARMS`` -- so the flag is only honest while this family
    holds a row there.
    """
    from meep_gpu.metal_kernels import launch as metal_launch

    assert family.CARRIES_DEPOSIT_REPAIR is True
    assert family.FAMILY in metal_launch.FUSED_PAIR_ARMS
    assert metal_launch.FUSED_PAIR_SEAMS[family.SLOT] == ("update_H", "B")


def test_the_polarity_is_the_opposite_of_the_d_side_pairs():
    """The two fused pairs must not both refuse the same source, or one of them
    has copied the other's clause rather than read the driver."""
    d_side = sibling.metal_fused_dispersive_pair_coverage(
        object(), None, (_Source("B"),))
    assert not any("source 0" in reason for reason in d_side.reasons)


def test_the_predicate_never_raises_on_a_degenerate_object():
    """``plan_step``'s whole contract is that a refusal is a verdict, not an
    exception: a raising predicate would escape the composer's arm loop."""
    coverage = family.metal_fused_magnetic_pair_coverage(object(), None, ())
    assert not coverage.covered and coverage.reasons


def test_both_halves_predicates_are_conjoined_and_labelled():
    """A configuration either half refuses is refused here with THAT half's
    reasons, prefixed, so a reader can tell which side spoke."""
    coverage = family.metal_fused_magnetic_pair_coverage(object(), None, ())
    prefixes = {reason.split(":")[0] for reason in coverage.reasons}
    assert "curl half" in prefixes
    assert "constitutive half" in prefixes


# ---------------------------------------------------------------------------
# 4. The table: registered, enumerable, and INVISIBLE to plan_step
# ---------------------------------------------------------------------------

def test_the_arm_is_registered_unwired_on_exactly_one_slot():
    rows = [spec for spec in arms.registered() if spec.family == family.FAMILY]
    assert len(rows) == 1, rows
    assert rows[0].slot == family.SLOT == "step_B"
    assert rows[0].wired is False, (
        "this product spans step_B, zero_metal_B and update_H; wiring it would "
        "hand plan_step a slot assignment no composition rule has measured")


def test_plan_step_cannot_select_it():
    """``arms_for`` is the gate between the table and the composer.

    An unwired row stays ENUMERABLE — which is what lets a disjointness sweep see
    it — while never reaching ``_select_slot``.
    """
    context = arms.StepContext(object(), None, None, (), sources=())
    labels = [arm.label for arm in arms.arms_for(family.SLOT, context)]
    assert "fused magnetic B/H pair" not in labels
    assert "fused magnetic B/H pair" in [
        spec.label for spec in arms.registered(family.SLOT)]


def test_the_family_is_in_family_modules():
    """A family in the tree but not in ``registry`` is invisible to ``plan_step``,
    which is a silent coverage loss rather than an error."""
    assert family.FAMILY in registry.FAMILY_MODULES


def test_the_plan_declares_every_pass_it_performs():
    """``replaces_sub_steps`` is READ by the composer and by the whole-step walk.

    A plan that under-declared would have the seam passes it actually performs
    bracketed with a host sync, which is a stale mirror rather than a wrong number.
    """
    assert family.REPLACES == ("step_B", "zero_metal_B", "update_H")
    assert family.MetalFusedMagneticPairPlan.replaces_sub_steps == family.REPLACES
    assert family.MetalFusedMagneticPairPlan.launches_per_run == 1
    assert family.MetalFusedMagneticPairPlan.performs_device_work is True


# ---------------------------------------------------------------------------
# 5. The platform ceiling that decides the product's SHAPE
# ---------------------------------------------------------------------------

def test_the_refuted_separate_scalar_signature_really_does_break_the_ceiling():
    """The 32-binding count is spelled once and the emitter agrees with it.

    Whether it FAILS TO COMPILE is a device measurement and belongs to the gate's
    ``binding_ceiling`` leg; what belongs here is that the signature this suite and
    that leg talk about is the same one, and that it is over the documented limit.
    """
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.refuted_separate_scalar_source()
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.SEPARATE_SCALAR_BINDINGS == 32
    assert family.SEPARATE_SCALAR_BINDINGS > MAX_BUFFER_BINDINGS


def test_the_shipped_packed_signature_fits_with_room():
    """27 pointers plus one packed struct is 28 — under the ceiling, and one
    dispatch for all three components rather than the sibling's three."""
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.fused_magnetic_pair_source((1, 1, 1), (True, True, True))
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.PACKED_BINDINGS == 28
    assert family.PACKED_BINDINGS <= MAX_BUFFER_BINDINGS


def test_the_contraction_guard_is_emitted_in_both_modes():
    """The guard is a property of the SOURCE on this backend, so each mode is a
    different compiled kernel and a plan built without one must raise rather than
    launch the pinned one."""
    off = family.fused_magnetic_pair_source((1, 1, 1), (True,) * 3,
                                            shaders.CONTRACT_OFF)
    fast = family.fused_magnetic_pair_source((1, 1, 1), (True,) * 3,
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
