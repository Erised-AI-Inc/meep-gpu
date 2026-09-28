"""The folded fused ELECTRIC Metal pair — the D-side twin, and the file it never had.

THIS SUITE WAS MISSING. ``metal_kernels/folded_fused_pair.py`` is RELEASED, is
credited 2 seam-instances on the 2026-08-20 closed Metal board, and had no laptop
test file of its own — its only appearance anywhere in ``meep_gpu/test_*.py`` was
one incidental line in the magnetic twin's suite, checking that the two families
refuse OPPOSITE source polarities. Everything else about it was measured only by
its device gate, which is the one thing a laptop cannot re-run.

What this file owns is what a laptop can decide, and it is deliberately the
family's REFUSALS rather than its arithmetic: the bytes belong to
``parity/meep_gpu/gate_metal_folded_fused_pair.py`` and nothing here duplicates
them. Specifically —

1. **The clause that cost this family its remaining rows was REQUIRED, not
   inherited — so only a CARRY could retire it.** A composition product's clause may
   be retired by a gate when something ABOVE it already refuses everything it would;
   both halves are asked here, on the same object, and both ADMIT a folded PERIODIC
   axis, so nothing above was ever going to discharge it. As of 2026-08-21 the pass
   is CARRIED INLINE (``folded_fused_pair`` REPLACES it) and the clause is retired
   by execution rather than by deletion.
2. **Retiring that clause discharges TWO obligations, not one.** The far fill is the
   obvious one. The second is the folded curl's TOP-PLANE MASK, which the refusal
   used to make unreachable — so a carry that took ``fill_folded_far_ghosts_D`` and
   stopped there would have taken up half the clause and left a plane of unmasked
   cells behind. Both are asserted on the emitter, in both directions: the periodic
   source must emit the mask AND the metallic source must not.
3. **The polarity, from the other side, and the deposit it no longer refuses.** An
   ELECTRIC source lands inside this seam and a MAGNETIC one does not. Until
   2026-08-28 that cost this family every row carrying one; since
   ``deposit_repair.repair_cells`` closed the fill images, a real electric deposit is
   CARRIED, a source that publishes no deposit index is still refused BY NAME, and
   holding the shipped flag False brings the old refusal straight back. All three
   directions are measured, because the admission alone could equally mean the clause
   was dropped.
4. **The arm is registered UNWIRED**, so ``plan_step`` cannot select it.

Measured 2026-08-20 (parity/meep_gpu/results/covering_refusals_2026-08-20/); the
far-carry legs re-measured 2026-08-21
(parity/meep_gpu/results/folded_far_carry_d_2026-08-21/).
"""

from __future__ import annotations

import pathlib

import pytest

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
GATE = (PACKAGE_DIR.parent / "parity" / "meep_gpu"
        / "gate_metal_folded_fused_pair.py")

from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import arms  # noqa: E402
from meep_gpu.metal_kernels import folded_fused_pair as family  # noqa: E402
#: The top stored index per axis and the flag the curl head declares for it — the
#: two spellings the far carry builds its destination from. Imported rather than
#: written out: a literal here would be a second transcription of the extents.
from meep_gpu.metal_kernels.folded_fused_pair import _LAST, _LAST_FLAG  # noqa: E402
from meep_gpu.metal_kernels.symmetry import (  # noqa: E402
    CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC,
    folded_axis_kinds, folded_top_plane_mask,
)
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402

#: ``step_D``'s direction flag. The one slip that would turn this suite into a
#: second copy of the magnetic twin's, so it is asserted against the module's own
#: sub-step table rather than left as a literal.
BACKWARD = True
#: A folded axis's declared parity, and no wall — the smallest configuration this
#: family's emitter accepts.
PHASES = (0, 1, 0)
WALLS = (False, False, False)


def codes(boundaries: str):
    """The per-axis quadruple for one fixture, RESOLVED and never spelled.

    A hand-built triple is this family's single point of failure — its emitter says
    so — and one written into a test would be a second transcription of a split
    only ``stepping._stored_past_owned`` owns.
    ``test_the_fixture_really_builds_both_terminations`` is what pins that the two
    fixtures below really are the two terminations.
    """
    fields, pml = build(boundaries=boundaries)
    resolved, reasons = folded_axis_kinds(fields.grid, pml)
    assert not reasons, (boundaries, reasons)
    return tuple(resolved)


class Source:
    """The smallest thing the predicate reads: a declared ``field_type``."""

    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def build(boundaries: str = "metallic", cell=(1.6, 3.0, 1.0),
          mirrors=(("Y", 1),), thickness: float = 0.2):
    """A real Grid/Fields/PML triple on NumPy, with the PML storage allocated.

    ``boundaries='periodic'`` is what makes the fold MIRROR_PERIODIC — the outer
    declaration is what ``_stored_past_owned`` reads, so the termination is chosen
    by the boundary and never by a hand-built code triple.
    """
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness)


def residual(verdict) -> list:
    """The refusals that are not this host's own backend or executor clauses.

    A laptop with no MPS refuses every Metal family on the resolved subnormal
    policy and on an undeclared residency, on EVERY configuration. Those say
    nothing about a corpus row and would drown the clause each test is about.
    """
    return [reason for reason in verdict.reasons
            if "subnormal" not in reason and "residency" not in reason]


def masked_targets(text: str) -> set:
    """Which curl targets a body TOP-PLANE-MASKS, read off the text.

    Applied to ``symmetry.folded_top_plane_mask``'s own output and to the fused
    body's, so the two can be compared without either being spelled: a test that
    wrote the mask out would agree with a kernel that wrote it out wrongly.
    """
    return {line.strip()[4] for line in text.splitlines()
            if line.strip().startswith("curl") and " = last_" in line}


# ---------------------------------------------------------------------------
# 1. The fold clause: it cost the family its rows, and a CARRY is what retired it
# ---------------------------------------------------------------------------

def test_the_fixture_really_builds_both_terminations():
    """Non-vacuity, first: every leg below is a comparison between these two.

    The Y axis must be folded in both and its TERMINATION must differ, or every
    comparison in this file is between one configuration and itself.
    """
    metallic, periodic = codes("metallic"), codes("periodic")
    assert metallic[1] == CODE_MIRROR_METALLIC, metallic
    assert periodic[1] == CODE_MIRROR_PERIODIC, periodic
    assert CODE_MIRROR_PERIODIC not in metallic, metallic
    assert CODE_MIRROR_METALLIC not in periodic, periodic

    # AND THE DIRECTION. Every top-plane leg below passes BACKWARD, and passing the
    # magnetic twin's would mask the twin's planes and agree with itself.
    from meep_gpu.metal_kernels.symmetry import SUB_STEPS

    assert BACKWARD is bool(SUB_STEPS["step_D"]["backward"])
    assert BACKWARD is not bool(SUB_STEPS["step_B"]["backward"])


def test_a_folded_periodic_axis_is_ADMITTED_and_its_D_side_far_fill_CARRIED():
    """The post-carry contract, and the three things that have to move TOGETHER.

    This test used to pin the opposite — ``fill_folded_far_ghosts_D`` refused BY
    NAME on a folded PERIODIC axis. The pass is carried inline as of 2026-08-21, so
    what a laptop can still decide is that the retirement is COHERENT: the predicate
    admits, the declaration claims the pass, and the emitter actually writes it. A
    round that moved any one of the three alone leaves either a plan that dies late
    or a covered_passes claim the bytes do not honour.

    THE PASS NAME IS STILL THE POINT. The magnetic twin carries ``_B``; a family
    that declared the other one has copied a neighbour rather than read the driver.
    """
    fields, pml = build(boundaries="periodic")
    assert residual(family.metal_folded_fused_pair_coverage(fields, pml, ())) == []

    # 1. THE DECLARATION. `REPLACES` is what the whole-step gate reads as
    # `covered_passes`; the driver's order is the spelling (driver.py:3292-3304).
    assert family.REPLACES == ("step_D", "fill_D", "zero_metal_D",
                               "fill_folded_far_ghosts_D", "update_E")
    assert "fill_folded_far_ghosts_B" not in family.REPLACES

    # 2. THE CARRY FIRES, and only on the termination that has a far ghost. The
    # destination set is read off the module rather than spelled: a literal here
    # would be a second transcription of the Yee table this family is keyed on.
    periodic, metallic = codes("periodic"), codes("metallic")
    far_periodic = tuple(family.far_fill_axes(periodic, target) for target in range(3))
    far_metallic = tuple(family.far_fill_axes(metallic, target) for target in range(3))
    assert any(far_periodic), far_periodic
    assert not any(far_metallic), far_metallic
    for target, far in enumerate(far_periodic):
        assert len(far) <= 1, (target, far)  # at most ONE far plane per D component

    # 3. THE BYTES. The far destination is built from the top stored index and the
    # runtime reflect row; both spellings come from the module's own tables, so this
    # fails if the emitter stops writing the plane rather than agreeing with itself.
    text = family.folded_fused_pair_source(periodic, PHASES, WALLS)
    carrying = [target for target, far in enumerate(far_periodic) if far]
    assert carrying, far_periodic
    for target in carrying:
        axis = far_periodic[target][0]
        assert _LAST[axis] in text, (target, axis, _LAST[axis])
        assert _LAST_FLAG[axis] in text, (target, axis, _LAST_FLAG[axis])
    # The row is a RUNTIME word, not a specialisation constant, so the destination
    # is GUARDED against it rather than compared with a literal plane.
    assert "== reflect_" in text, text
    assert "_fill_folded_far_ghosts" in text, "the block cites the pass it carries"

    # AND THE METALLIC SOURCE MUST NOT CARRY IT. A far-fill block emitted where
    # `_stored_past_owned` is false images a plane the array path never touches.
    # The reflect rows are UNPACKED unconditionally (they ride in `Params` on every
    # configuration), so the discriminator is the guarded block, not the binding.
    metallic_text = family.folded_fused_pair_source(metallic, PHASES, WALLS)
    assert "== reflect_" not in metallic_text, metallic_text
    assert "_fill_folded_far_ghosts" not in metallic_text, metallic_text


def test_only_a_CARRY_could_have_retired_the_clause_no_half_ever_refused_it():
    """Why deletion was never available here — decided by execution, not reading.

    "Above" is a RUN-TIME relation: a clause behind an early return is unreachable
    however live it looks, and one whose configurations a half already refuses can
    be deleted with no verdict changing. Both halves are asked the same question on
    the same object here. Both ADMIT a folded PERIODIC axis, so the refusal lived
    only in the product — which is exactly why the 2026-08-21 round had to CARRY
    ``fill_folded_far_ghosts_D`` to retire it, and why a round that had merely
    deleted the clause would have shipped a seam that skips the pass.

    THE OTHER TERMINATION IS THE NON-VACUITY LEG and it is kept: the product must
    admit metallic too, or the periodic admission above is a comparison against a
    fixture that admits everything.
    """
    fields, pml = build(boundaries="periodic")
    for half in (family.folded_composition_curl_coverage(fields, pml, "step_D", None),
                 family.folded_constitutive_coverage(fields, pml, "E", None)):
        assert residual(half) == [], residual(half)

    metallic, metallic_pml = build(boundaries="metallic")
    assert residual(family.metal_folded_fused_pair_coverage(
        metallic, metallic_pml, ())) == []


def test_the_source_builder_admits_the_same_configuration_the_predicate_does():
    """A predicate that admits what the emitter raises on is a plan that dies late.

    The two used to AGREE BY REFUSING and now agree by ADMITTING; what is pinned is
    the agreement, which is the property that survives either side of the carry. A
    round that had retired the predicate's clause and left the emitter raising would
    pass a coverage sweep and die at compile time on a real corpus row.
    """
    for boundaries in ("periodic", "metallic"):
        fields, pml = build(boundaries=boundaries)
        admitted = residual(
            family.metal_folded_fused_pair_coverage(fields, pml, ())) == []
        try:
            built = bool(family.folded_fused_pair_source(
                codes(boundaries), PHASES, WALLS))
        except ValueError:
            built = False
        assert admitted is built is True, (boundaries, admitted, built)

    # THE EMITTER STILL REFUSES WHAT IT CANNOT BUILD, so this is not agreement by
    # having no clause left: a folded axis with an undeclared parity is still a
    # compile-time specialisation this family cannot default.
    with pytest.raises(ValueError):
        family.folded_fused_pair_source(codes("periodic"), (0, 7, 0), WALLS)


# ---------------------------------------------------------------------------
# 2. The SECOND obligation, and that the carry discharged it too
# ---------------------------------------------------------------------------

def test_the_carry_takes_the_top_plane_mask_WITH_the_far_fill_not_just_the_fill():
    """The far fill AND the folded curl's top-plane mask.

    THIS IS THE OBLIGATION A CARRY OF ``fill_folded_far_ghosts_D`` DOES NOT
    DISCHARGE ON ITS OWN, and it is the leg that decides whether the 2026-08-21
    retirement was a carry or a deletion. ``symmetry.folded_top_plane_mask``
    (symmetry.py:528-552)
    is ``_mask_non_owned_cells``' top-plane arm: on a folded PERIODIC axis the last
    plane of every target whose Yee shift is 1 there sits past MEEP's owned window
    and the FILL, not the curl, writes it. On a folded METALLIC axis the stored
    array stops at ``big_corner``, that plane is owned, and masking it would delete
    a real cell — which is why the emitter answers differently on the two.

    Measured by CALLING the emitter on both terminations rather than by spelling
    the mask: a test that spelled it would agree with a kernel that spelled it
    wrongly. Both directions are asserted, so this fails if either side moves.
    """
    metallic = folded_top_plane_mask(codes("metallic"), BACKWARD)
    periodic = folded_top_plane_mask(codes("periodic"), BACKWARD)

    def live(text: str) -> list:
        return [line.strip() for line in text.splitlines()
                if line.strip() and not line.strip().startswith("//")]

    assert live(metallic) == [], live(metallic)
    assert live(periodic), (
        "the top-plane mask emits nothing on a folded PERIODIC axis, so refusing "
        "that termination would buy only the far fill and this family's docstring "
        "would be wrong about what its clause discharges")

    # AND THE FUSED BODY MUST CARRY IT WHERE IT IS REACHABLE AND NOWHERE ELSE.
    # BOTH directions are asserted because each catches the opposite half-done
    # round: a periodic body with no mask is a carry that took the fill and left a
    # plane of unmasked cells behind, and a metallic body WITH one masks a plane
    # that is owned and stepped, deleting a real cell.
    metallic_text = family.folded_fused_pair_source(codes("metallic"), PHASES, WALLS)
    periodic_text = family.folded_fused_pair_source(codes("periodic"), PHASES, WALLS)
    assert masked_targets(metallic_text) == masked_targets(metallic) == set(), (
        metallic_text)
    assert masked_targets(periodic_text) == masked_targets(periodic) != set(), (
        "the emitted body must mask exactly the targets "
        "`symmetry.folded_top_plane_mask` names for this termination; a carry of "
        "fill_folded_far_ghosts_D that skipped the mask leaves that plane wrong")


def test_the_two_seams_mask_DIFFERENT_planes():
    """Why a sweep on the B sub-step proves nothing about this one.

    ``TARGET_IYEE`` puts the B family's shifts at 1 on the two axes that are NOT
    its own and the D family's on its OWN axis alone, so the near ghost lands on a
    cell-0-masked plane and the far ghost on a top-plane-masked one — and the two
    sub-steps mask different targets. Derived from the emitter, not restated.
    """
    magnetic = folded_top_plane_mask(codes("periodic"), False)
    electric = folded_top_plane_mask(codes("periodic"), True)
    assert magnetic != electric, (magnetic, electric)

    def targets(text: str) -> set:
        return {line.strip()[4] for line in text.splitlines()
                if line.strip().startswith("curl")}

    assert targets(magnetic) == {"0", "2"}, magnetic
    assert targets(electric) == {"1"}, electric


# ---------------------------------------------------------------------------
# 3. The polarity, and the deferral
# ---------------------------------------------------------------------------

def test_an_electric_deposit_is_CARRIED_since_the_fill_images_are_repaired():
    """WHAT ``CARRIES_DEPOSIT_REPAIR = True`` BUYS THIS FAMILY, on a real deposit.

    The driver injects an electric source between ``step_D`` and ``update_E``
    (driver.py:3301-3307) — inside THIS seam — and a magnetic one inside the twin's.
    Until 2026-08-28 that asymmetry took this family from 54 seam-instances on the B
    half to 4 on this one. It no longer refuses the deposit: ``deposit_repair`` saves
    and restores the deposit index TOGETHER WITH the cells this seam's two fills image
    it into, so the source clause clears and a magnetic source stays irrelevant.
    """
    fields, pml = build()
    source = GaussianPulsedSource(grid=fields.grid, component="Ez",
                                  center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                  frequency=1.0, fwidth=0.2, amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    electric = residual(family.metal_folded_fused_pair_coverage(
        fields, pml, (source,)))
    assert not any("source 0" in reason for reason in electric), electric
    assert not any("deposit repair" in reason for reason in electric), electric
    magnetic = residual(family.metal_folded_fused_pair_coverage(
        fields, pml, (Source("B"),)))
    assert not any("source 0" in reason for reason in magnetic), magnetic


def test_an_electric_source_that_publishes_no_deposit_index_is_still_refused():
    """THE OTHER DIRECTION, and it is what keeps the admission above honest.

    The repair carries a deposit by saving and restoring the cells it writes, so a
    source that never says which cells those are cannot be carried at any closure.
    ``Source`` here is exactly that: an in-seam electric source with no
    ``_point_ix``/``_point_iy``/``_point_iz``, which the clause refuses BY NAME
    rather than admitting on the flag alone.
    """
    fields, pml = build()
    refused = residual(family.metal_folded_fused_pair_coverage(
        fields, pml, (Source("D"),)))
    assert any("does not publish the index it writes" in reason
               for reason in refused), refused


def test_the_flag_is_what_admits_the_deposit_and_holding_it_false_refuses_by_name(
        monkeypatch):
    """The same real deposit, with the SHIPPED flag held False: the pre-2026-08-28
    refusal comes straight back, naming the seam the driver injects into. Without
    this, the admission above would be consistent with the source clause having been
    dropped rather than earned."""
    fields, pml = build()
    source = GaussianPulsedSource(grid=fields.grid, component="Ez",
                                  center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                  frequency=1.0, fwidth=0.2, amplitude=1.0)
    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    refused = residual(family.metal_folded_fused_pair_coverage(
        fields, pml, (source,)))
    assert any("is electric" in reason and "BETWEEN step_D and update_E" in reason
               for reason in refused), refused


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    """IGNORANCE IS NOT AN EMPTY SET, and inferring one is the over-covering
    refusal this clause exists to prevent: ``Fields`` does not hold the source
    list, so not being told is not being told nothing is there."""
    fields, pml = build()
    refused = residual(family.metal_folded_fused_pair_coverage(fields, pml, None))
    assert any("was not declared" in reason for reason in refused), refused


def test_the_arm_is_registered_UNWIRED_so_plan_step_cannot_select_it():
    """The deferral, read off the live table rather than the module's docstring."""
    registered = {spec.family: spec for spec in arms.registered()}
    assert family.FAMILY in registered, sorted(registered)
    assert registered[family.FAMILY].wired is False

    # ENUMERABLE BUT NOT SELECTABLE, and both halves matter: `arms.families_on`
    # must still see it (the disjointness sweep walks that list) while `arms_for`,
    # which is what `plan_step` consults, filters on `spec.wired` and cannot.
    assert family.FAMILY in arms.families_on(family.SLOT)
    source = pathlib.Path(arms.__file__).read_text(encoding="utf-8")
    assert "if spec.wired" in source


def test_the_device_gate_exists_and_names_this_product():
    assert GATE.exists(), GATE
    text = GATE.read_text(encoding="utf-8")
    assert "folded_fused_pair" in text
