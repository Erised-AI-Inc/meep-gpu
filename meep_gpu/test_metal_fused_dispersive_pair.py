"""The FIRST fused Metal kernel, as claims a laptop can check without a GPU.

WHAT THIS SUITE OWNS AND WHAT IT DELIBERATELY DOES NOT. The BYTES are the device
gate's — ``parity/meep_gpu/gate_metal_fused_dispersive_pair.py`` steps two engines
side by side for twelve complete steps and compares uint32 words, and no assertion
here duplicates that. What lives here is everything that is true about the family
WITHOUT a device: what the emitted source says, which configurations the predicate
refuses and by what name, that the arm is registered UNWIRED, and that the plan
binds the lattice each half is supposed to take.

THE ONE CLAIM THIS SUITE MAKES THAT THE GATE CANNOT. A gate measures the
configurations it runs. It cannot say the product is invisible to ``plan_step`` —
that is a property of the TABLE, and it is the whole reason this family may land
beside twenty-nine wired ones without touching any of them.
"""

from __future__ import annotations

import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent
GATE = REPO / "parity" / "meep_gpu" / "gate_metal_fused_dispersive_pair.py"

from meep_gpu.metal_kernels import (  # noqa: E402
    arms, fused_dispersive_pair as family, registry, shaders, templates,
)


# ---------------------------------------------------------------------------
# 1. The source is a transcription, and the diff against its two parents is small
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", (0, 1, 2))
def test_the_curl_half_is_the_certified_curl_bodys_own_lines(axis):
    """Every arithmetic line of the curl half appears in ``shaders._CURL_TEMPLATE``.

    This is the transcription rule made checkable. The fused body may DROP lines
    the target does not need — a per-component kernel differences two of the three
    H volumes — but it may not INVENT one, and an invented float expression is
    exactly the class the validate-against-the-reference rule forbids.
    """
    certified = set(shaders._CURL_TEMPLATE.splitlines())
    source = family.fused_dispersive_pair_source((0, 0, 0), axis, 0,
                                                 (False, False, False))
    # The E half has its own parent and its own test below; split on the marker
    # the emitter writes rather than guessing which half a line belongs to.
    head, marker, tail = source.partition("    // --- pole-aware update_E")
    assert marker and tail, "the emitter's E-half marker moved"
    carried = [line for line in head.splitlines()
               if line.strip().startswith(("float curl", "float p", "float n",
                                           "float v", "float a", "float b",
                                           "float c", "float km_"))]
    assert carried, "the curl half emitted no arithmetic at all"
    missing = [line for line in carried if line not in certified]
    assert not missing, (
        f"axis {axis}: these lines are NOT in the certified curl template and are "
        f"therefore a re-derivation rather than a transcription: {missing}")


@pytest.mark.parametrize("axis,coordinate", ((0, "i"), (1, "j"), (2, "k")))
def test_the_electric_half_is_the_pointwise_bodys_own_lines_but_for_the_seam(
        axis, coordinate):
    """The E half differs from the certified pointwise body in EXACTLY the seam.

    ``float source = d_in[idx];`` becomes ``float source = v{axis};`` and the pole
    buffers are respelled ``q`` (the certified curl already owns ``p0``..``p2`` in
    this scope). Nothing else moves — and if something does, this fails naming it.
    """
    from meep_gpu.metal_kernels import dispersive_update_e as pointwise

    pinned = pointwise.dispersive_e_source(2, axis)
    fused = family.fused_dispersive_pair_source((0, 0, 0), axis, 2,
                                                (False, False, False))
    for line in ("    float prev = fw[idx];",
                 "    float src = source * inv_e[idx];",
                 "    fw[idx] = src;",
                 "    float value = e_out[idx];",
                 f"    value = value + kps[{coordinate}] * src;",
                 f"    value = value - kms[{coordinate}] * prev;",
                 "    e_out[idx] = value;"):
        assert line in pinned, f"the pinned line moved in the pointwise family: {line}"
        assert line in fused, f"the fused body dropped {line!r}"
    assert "    float source = d_in[idx];" in pinned
    assert "    float source = d_in[idx];" not in fused, (
        "the fused body reloads the displacement it just wrote; the seam is not closed")
    assert f"    float source = v{axis};" in fused
    assert fused.count("source = source - q") == 2
    assert "p0[idx]" not in fused, (
        "a pole buffer is still spelled p0, which collides with the curl's own "
        "split-field previous value in this scope")


def test_the_pole_chain_is_ordered_and_length_is_the_specialisation():
    """``D - P0 - P1 - ...`` left to right, one line per pole, no more."""
    for count in range(0, family.MAX_POLES + 1):
        source = family.fused_dispersive_pair_source((0, 0, 0), 0, count,
                                                     (False, False, False))
        chain = [line.strip() for line in source.splitlines()
                 if line.strip().startswith("source = source - ")]
        assert chain == [f"source = source - q{index}[idx];"
                         for index in range(count)], (count, chain)


@pytest.mark.parametrize("count", (-1, family.MAX_POLES + 1))
def test_a_pole_count_outside_the_compiled_range_is_refused(count):
    with pytest.raises(ValueError):
        family.fused_dispersive_pair_source((0, 0, 0), 0, count, (False,) * 3)


@pytest.mark.parametrize("bad", ((0, 0), (0, 0, 0, 0)))
def test_a_boundary_triple_that_is_not_a_triple_is_refused(bad):
    with pytest.raises(ValueError):
        family.fused_dispersive_pair_source(bad, 0, 1, (False,) * 3)


def test_an_axis_outside_the_three_targets_is_refused():
    with pytest.raises(ValueError):
        family.fused_dispersive_pair_source((0, 0, 0), 3, 1, (False,) * 3)


# ---------------------------------------------------------------------------
# 2. zero_metal_D, carried inline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,walled,expected", (
    (0, (True, False, False), ()),          # Dx shift on x is 1: not on the wall
    (0, (False, True, False), ("at_y",)),
    (0, (False, False, True), ("at_z",)),
    (1, (True, False, False), ("at_x",)),
    (1, (False, True, False), ()),          # Dy shift on y is 1
    (2, (False, False, True), ()),          # Dz shift on z is 1
    (2, (True, True, False), ("at_x", "at_y")),
))
def test_the_inline_wall_clear_is_the_yee_shift_table(target, walled, expected):
    """``_zero_metal`` clears a component only where its Yee shift is 0.

    ``IYEE_SHIFTS`` (fields.py:216) gives Dx (1,0,0), Dy (0,1,0), Dz (0,0,1), so
    the rows are the complement of the diagonal. Getting this backwards zeroes the
    NORMAL component instead of the tangential pair, which is a smooth, plausible,
    completely wrong boundary — never a crash.
    """
    emitted = family.zero_metal_mask(target, walled)
    flags = tuple(line.split("=")[1].split("?")[0].strip()
                  for line in emitted.splitlines() if "?" in line)
    assert flags == expected, emitted
    if not expected:
        assert "no walled axis" in emitted


def test_the_wall_clear_uses_the_selects_measured_zero_literal():
    """The same ``v ? 0.0f : v`` form the certified ownership mask emits.

    shaders.py rule 3: the ternary delivers the exact ``0.0`` a clamped load would
    not, and Metal's ``min``/``max`` builtins disagree with numpy on signed zeros.
    """
    emitted = family.zero_metal_mask(0, (False, True, True))
    assert emitted.splitlines() == ["    v0 = at_y ? 0.0f : v0;",
                                    "    v0 = at_z ? 0.0f : v0;"]
    mask = templates.ownership_mask((0, 1, 1), True)
    assert "? 0.0f :" in mask, "the certified emitter's literal changed"


def test_a_run_with_no_wall_emits_no_clear_at_all():
    """A periodic run must be byte-identical to one that never carried the pass.

    ``_zero_metal`` returns before touching a plane when nothing is metallic
    (stepping.py:2280-2286), so emitting a select that is always false would be
    correct arithmetic and a different kernel.
    """
    source = family.fused_dispersive_pair_source((0, 0, 0), 0, 1, (False,) * 3)
    assert "? 0.0f : v0" not in source


# ---------------------------------------------------------------------------
# 3. The seam predicate
# ---------------------------------------------------------------------------

class _Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def _reasons(fields, pml, sources, residency=None):
    return family.metal_fused_dispersive_pair_coverage(
        fields, pml, sources, residency).reasons


def test_an_undeclared_source_set_is_refused_by_name():
    """IGNORANCE IS NOT AN EMPTY SET, and this is the clause that says so.

    ``Fields`` does not hold the source list — the driver does — so a predicate
    that inferred "no electric source" from not being told would licence a fused
    pair over an injection it never saw.
    """
    reasons = _reasons(object(), None, None)
    assert any("source set was not declared" in reason for reason in reasons)


def test_an_electric_source_is_carried_by_name_and_a_magnetic_one_is_not_this_seam():
    """The injection sits INSIDE this seam; the magnetic one sits in the B/H half.

    ``CARRIES_DEPOSIT_REPAIR`` is True since 2026-08-28, so an electric source is no
    longer refused for BEING electric -- it is CARRIED, and what refuses on this stub is
    what the repair cannot establish: no ``f_w_Ex`` to save, and a source publishing no
    deposit index (deposit_repair.py:108-111, :244-247). Both clauses are reached only
    for an in-seam source, so the polarity is still what this measures: the magnetic
    source produces neither.
    """
    electric = _reasons(object(), None, (_Source("E"),))
    assert any("f_w_Ex is not allocated" in reason for reason in electric), electric
    assert any("does not publish the index it writes" in reason
               for reason in electric), electric
    magnetic = _reasons(object(), None, (_Source("B"),))
    assert not any("f_w_Ex is not allocated" in reason for reason in magnetic), magnetic
    assert not any("does not publish the index" in reason for reason in magnetic)


def test_the_pre_flip_refusal_is_still_what_the_flag_returns_when_held_false(monkeypatch):
    """The other direction, which is what makes the flip a change rather than a
    deletion. Held at False by monkeypatch; this is the prose this product's gate pins
    and the one every unrouted Metal family still returns."""
    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    electric = _reasons(object(), None, (_Source("E"),))
    assert any("is electric" in reason and "BETWEEN step_D and update_E" in reason
               for reason in electric), electric
    magnetic = _reasons(object(), None, (_Source("B"),))
    assert not any("is electric" in reason for reason in magnetic)


def test_the_predicate_never_raises_on_a_degenerate_object():
    """A refusal, never an exception: the composer's whole contract is that a
    configuration nothing covers falls to the array path."""
    verdict = family.metal_fused_dispersive_pair_coverage(object(), None, ())
    assert verdict.covered is False and verdict.reasons


def test_both_halves_predicates_are_conjoined_and_labelled():
    """A refusal from either half must arrive NAMED, with the half that said it."""
    reasons = _reasons(object(), None, ())
    assert any(reason.startswith("curl half: ") for reason in reasons)
    assert any(reason.startswith("dispersive constitutive half: ")
               for reason in reasons)


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

    An unwired row stays ENUMERABLE — which is what lets a disjointness sweep see
    it — while never reaching ``_select_slot``. That distinction is the entire
    safety argument for landing this family beside the wired tree.
    """
    context = arms.StepContext(object(), None, None, (), sources=())
    labels = [arm.label for arm in arms.arms_for(family.SLOT, context)]
    assert "fused dispersive D/E pair" not in labels
    assert "fused dispersive D/E pair" in [
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
    assert family.REPLACES == ("step_D", "zero_metal_D", "update_E")
    assert family.MetalFusedDispersivePairPlan.replaces_sub_steps == family.REPLACES
    assert family.MetalFusedDispersivePairPlan.launches_per_run == 3
    assert family.MetalFusedDispersivePairPlan.performs_device_work is True


# ---------------------------------------------------------------------------
# 5. The platform ceiling that decides the product's SHAPE
# ---------------------------------------------------------------------------

def test_the_refuted_all_component_signature_really_does_break_the_ceiling():
    """The 35-binding count is spelled once and the emitter agrees with it.

    Whether it FAILS TO COMPILE is a device measurement and belongs to the gate's
    ``binding_ceiling`` leg; what belongs here is that the signature this suite and
    that leg talk about is the same one, and that it is over the documented limit.
    """
    from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS

    source = family.refuted_all_component_source()
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest + 1 == family.ALL_COMPONENT_BINDINGS == 35
    assert family.ALL_COMPONENT_BINDINGS > MAX_BUFFER_BINDINGS


def test_the_shipped_per_component_signature_fits_with_room():
    source = family.fused_dispersive_pair_source((1, 1, 1), 0, family.MAX_POLES,
                                                 (True, True, True))
    highest = max(int(chunk.split(")")[0])
                  for chunk in source.split("[[buffer(")[1:])
    assert highest == 28, highest


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
