"""The Metal composer's two-slot protocol, and the deposit repair riding it.

``test_fused_pair_deposit_wiring.py`` proves the protocol on the Triton composer and
``test_deposit_repair.py`` proves the arithmetic against the driver's own order. THIS
FILE PROVES THE METAL TRACK IS WIRED TO THE SAME MODULE, and it proves it through
``metal_kernels.launch.plan_step`` rather than by calling the installer directly --
because the claim that matters is not "the function exists" but "a Metal composition
can reach it".

WHAT WAS MISSING AND WHAT LANDED. Every Metal fused product registers ``wired=False``
on the FIRST slot of the seam it spans and declares the rest through
``replaces_sub_steps``; nothing on this track had ever written the SECOND slot on a
pair's behalf, so ``metal_kernels/launch.py`` carried no reference to
``deposit_repair`` at all. It now carries ``_install_fused_pair``, the same shape as
the Triton composer's, reached from the opt-in ``fuse`` block.

THE FLAG IS NOW TRUE ON ALL FIVE FAMILIES THE SEAM LOOP CAN REACH AND FALSE ON THE
OTHER SEVEN WELDS, and this suite reads the shipped values rather than patching them
into place. The repair cases run against the SHIPPED ``True``; the monkeypatch that remains runs the other way, holding
``CARRIES_DEPOSIT_REPAIR`` at ``False`` for one case so the refusal prose every other
Metal family still returns stays measured. Those two directions together are the whole
claim: the wiring is in place, the flag turns it on, and the False branch is still the
flat refusal it always was.

WHICH FAMILIES MAY CARRY IT. ``launch.FUSED_PAIR_ARMS`` bounds the seam loop's reach:
a Metal weld with no row there is refused at the absorb table before its predicate is
asked, so a family flipping the flag without gaining a row would be declaring a
bracket the loop refuses to build. That is why the sweep at the bottom of this file is
derived from that table rather than typed. The table holds five rows as of 2026-08-28
— ``fused_magnetic_pair``, the two folded pairs, ``complex_fused_magnetic_pair`` and
``fused_dispersive_pair`` — and ROUTING IS STILL NOT REPAIRING: the two decisions stay
separate even now that all five rows have made both, which is what
:data:`ROUTED_WITHOUT_THE_REPAIR` keeps a place for.

THE FOUR FAMILIES ADDED ON 2026-08-28. Two are unfolded and each brings a constitutive
half ``deposit_repair`` had never reconstructed — complex64 storage on the B seam, and a
live pole on the D seam. The other two are the FOLDED pair, and what they brought is a
seam that runs ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*`` between the halves:
a deposit landing on a row one of those fills reads from is imaged into cells a point
repair never visits, which is what ``deposit_repair.repair_cells`` closed. The arithmetic
for all four is measured in ``test_deposit_repair.py`` (``COMPLEX_CASES``,
``DISPERSIVE_CASES`` and the folded cases: byte-identical to the driver's order, every
null control diverging, and the folded ones diverging again when the closure is cut back
to the deposit index); what is measured HERE is the other half of the same claim, that a
Metal composition actually reaches the bracket on the configurations those families
serve.

WHY A MAGNETIC SOURCE. ``fused_magnetic_pair`` is the one Metal fused product whose
absorb declaration is READ rather than guessed -- its predicate is literally the
``PML`` curl arm's and the ``ordinary`` constitutive arm's, and the ``cart_pml_real``
matrix row records those two labels winning ``step_B`` and ``update_H``. The driver
injects a magnetic source between exactly those two consults (driver.py:3283-3284),
so this is the seam the repair exists for.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                       "parity", "meep_gpu")
_PARITY = os.path.abspath(_PARITY)
if _PARITY not in sys.path:
    sys.path.insert(0, _PARITY)

import metal_composition_matrix as matrix  # noqa: E402

from meep_gpu import deposit_repair  # noqa: E402
from meep_gpu.metal_kernels import (  # noqa: E402
    arms, device, fused_hd_pair as hd, fused_magnetic_pair as family, launch, shaders,
)
from meep_gpu.sources import (  # noqa: E402
    GaussianEnvelope, GaussianPulsedSource, VolumeSource,
)
from meep_gpu.triton_kernels.coverage import Coverage  # noqa: E402
from meep_gpu.triton_kernels.launch import NoopPlan  # noqa: E402

ENVIRONMENT = matrix.prepare_environment()

CURL_SLOT = "step_B"
UPDATE_SLOT = "update_H"
LABEL = "fused magnetic B/H pair"


def _probes():
    from meep_gpu.metal_kernels import (
        complex_fields, cylindrical_complex, folded_complex, special_kz,
    )

    return {"complex_probe": complex_fields.load_expansion_probe(),
            "beta_probe": special_kz.load_expansion_probe(),
            "folded_complex_probe": folded_complex.load_expansion_probe(),
            "cylindrical_complex_probe":
                cylindrical_complex.load_expansion_probe()}


def _magnetic_source(grid):
    """A point magnetic source, so the seam carries a deposit the repair can save.

    A source with no deposit point would let every assertion below pass while
    measuring nothing, so the caller asserts ``_n_source_points`` before using it.
    """
    return VolumeSource(grid=grid, component="Hz", center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def _plan(sources, fuse):
    fields, pml = matrix.cart()
    return fields, launch.plan_step(fields, pml, residency=device.Residency(),
                                    sources=sources, fuse=fuse, **_probes())


def _fused_reasons(plan):
    return plan.reasons.get(f"fused_pair_{family.FAMILY}", ())


# ---------------------------------------------------------------------------
# 0. The precondition, so a vacuous pass is impossible
# ---------------------------------------------------------------------------

def test_the_two_arms_this_pair_absorbs_really_do_win_their_slots():
    """If they did not, every fusion below would refuse for the RIGHT reason and
    the suite would measure nothing. ``_pair_may_absorb`` is a real gate here."""
    _fields, plan = _plan((), fuse=False)
    assert plan.selected.get(CURL_SLOT) == "PML"
    assert plan.selected.get(UPDATE_SLOT) == "ordinary"
    assert launch.FUSED_PAIR_ARMS[family.FAMILY] == ("PML", "ordinary")


def test_the_flag_is_true_as_shipped_and_this_family_can_actually_be_bracketed():
    """THE FLAG AND THE WIRING, ASSERTED TOGETHER, which is the rule
    ``deposit_repair`` states: True without the wrappers is the exact defect it
    exists to prevent. On this track the wrappers are built by the seam loop in
    ``_install_fused_pairs``, and the families that loop can reach are bounded by
    ``FUSED_PAIR_ARMS`` -- so the flag is honest only while this family holds a row
    there, on the seam ``deposit_repair`` knows as ``B``."""
    assert family.CARRIES_DEPOSIT_REPAIR is True
    assert launch.FUSED_PAIR_ARMS[family.FAMILY] == ("PML", "ordinary")
    assert launch.FUSED_PAIR_SEAMS[CURL_SLOT] == (UPDATE_SLOT, "B")


#: Families the seam loop CAN install but which must NOT carry the deposit repair,
#: with the reason each is held there. Routing and repairing are two decisions and
#: this is the list where they come apart.
#:
#: THE FOLDED PAIR WAS THE ONLY ENTRY AND IT IS GONE, 2026-08-28. Both folded families
#: were given an absorb row on 2026-08-27 so a CLEAN folded seam could fuse, and neither
#: could carry the repair: a folded seam also runs ``fill_symmetry_bc_*`` and
#: ``fill_folded_far_ghosts_*`` AFTER the injection (driver.py:3293-3295, :3308-3311),
#: the near fill writes cell 0 from cell 2 (stepping.py:1450-1451), and
#: ``deposit_repair.apply`` wrote only the deposit INDEX. ``repair_cells`` closed that by
#: extending the saved and restored set to the closure of the images, both directions
#: measured in ``test_deposit_repair.py``, and both flags flipped in the same edit.
#:
#: EMPTY IS A MEASUREMENT HERE, NOT AN OMISSION: the assertion below is an equality
#: against ``FUSED_PAIR_ARMS`` minus the flipped set, so a Metal family that gains an
#: absorb row without flipping — or that flips without one — fails by name whether this
#: tuple is empty or not. A family arriving here again is a decision somebody has to
#: write down beside it.
#:
#: THE 2026-09-01 RESIDUE ROUND'S MAGNETIC BETA-COMPLEX PAIR IS THAT FAMILY, and the
#: decision is a census measurement rather than a caution: its one reachable corpus
#: row (``tests:TestSpecialKz.test_special_kz``) declares ``source_field_types ==
#: ["D"]`` — electric deposits only — so its ``step_B``/``update_H`` seam contains
#: NO deposit for a repair to bracket, and ``CARRIES_DEPOSIT_REPAIR = False`` is the
#: measured value (``test_metal_residue_fused_pairs.py::
#: test_the_carries_flag_is_the_census_measurement`` asserts exactly this against
#: the census row). The row's D seam deposit is carried by the ELECTRIC twin
#: (``beta_complex_fused_electric_pair``, flag True, same round). The unrouted
#: magnetic pairs with False flags (``bfast_fused_magnetic_pair``,
#: ``beta_fused_magnetic_pair``) never arrive here because they hold no
#: ``FUSED_PAIR_ARMS`` row at all.
#: THE FOUR STENCIL WELDS OF 2026-09-01 ARE THE OTHER FOUR, and their decision is
#: FORCED rather than measured per row: an OFF-DIAGONAL chi1inv constitutive is one
#: of the two shapes ``deposit_repair.repairable`` refuses BY NAME
#: (deposit_repair.py), because a point repair recomputes E at the deposit cell from
#: THAT CELL's displacement and an off-diagonal constitutive reads its NEIGHBOURS'.
#: So these four cannot carry the bracket however their corpus rows are sourced, and
#: each one's predicate refuses an in-seam electric source outright — which is
#: exactly the distance between their cells' 16/19/2/3 rows and the 8/9/1/1 a fused
#: product can ever reach there. Measured, with its admitting control, in
#: ``test_metal_offdiag_stencil_welds.py::
#: test_the_carries_flag_is_false_and_the_clause_that_forces_it_is_measured`` and
#: again on device by that round's gate (leg ``seam_refusal``).
ROUTED_WITHOUT_THE_REPAIR: tuple = (
    "beta_complex_fused_magnetic_pair",
    "offdiag_fused_electric_pair",
    "folded_offdiag_fused_electric_pair",
    "complex_no_pml_offdiag_fused_electric_pair",
    "folded_complex_offdiag_fused_electric_pair",
    # 2026-09-05: the H->D weld. Its seam carries no injection at all -- the
    # electric withdraw sits between the two consults and `withdraw_hoist` owns it
    # -- so the flag is a fact about the driver, and the absorb row is what lets
    # the seam loop ask the product (it then refuses it on INSTALLABLE = False).
    "fused_hd_pair",
    # 2026-09-06: the Dcyl m = 0 H->D pair. The same seam, the same fact about the
    # driver -- nothing is injected between the two consults -- so the flag is False
    # and the absorb row is what lets the seam loop ask it (it then refuses it on
    # INSTALLABLE = False, and with the flag out of the way on the released D->E
    # cylindrical pair holding step_D first).
    "cylindrical_real_fused_hd_pair",
    # 2026-09-06: the complex Dcyl H->D product. Same seam, same fact about the
    # driver -- nothing is injected between the two consults -- and the same
    # INSTALLABLE = False refusal in the seam loop.
    "cylindrical_complex_fused_hd_pair",
    # 2026-09-07: the FOLDED H->D pair. The same seam and the same fact about the
    # driver, and on a fold that fact is STRONGER rather than weaker: all three B-side
    # fill/clear/far passes close BEFORE the update_H consult (driver.py:3305-3310) and
    # all three D-side ones open AFTER the step_D consult (:3325-3330), so nothing is
    # injected between the two halves and nothing is imaged across the launch either.
    # The absorb row lets the seam loop ask the product; it then refuses it on
    # INSTALLABLE = False, and with the flag out of the way on the released
    # folded_fused_magnetic_pair holding update_H first (78 of 78 corpus rows).
    "folded_fused_hd_pair",
    # 2026-09-07: the Cartesian complex/Bloch H->D product. Same seam, same fact
    # about the driver -- nothing is injected between the two consults -- and the
    # same INSTALLABLE = False refusal in the seam loop, on a measured
    # arbitration (a LOSS on all 17 rows of its cell).
    "complex_fused_hd_pair",
    # 2026-09-07, THE THREE H->D TAILS. The same seam and the same fact about the
    # driver on all three -- nothing is injected between the two consults -- and on
    # the folded variants that fact is STRONGER rather than weaker: all three B-side
    # fill/clear/far passes close BEFORE the update_H consult and all three D-side
    # ones open AFTER the step_D consult, so nothing is injected between the halves
    # and nothing is imaged across the launch either. Each absorb row lets the seam
    # loop ASK the product; each then refuses it on INSTALLABLE = False, on an
    # arbitration MEASURED through the shipped composer on that product's own eight
    # gate fixtures (a LOSS on 8 of 8, a TIE on 0, a GAIN on 0).
    "folded_complex_fused_hd_pair",
    "beta_complex_fused_hd_pair",
    "beta_real_fused_hd_pair",
    # 2026-09-08, THE SEAM'S LAST TWO METAL CELLS. The same seam and the same fact
    # about the driver a seventh and eighth time -- nothing is injected between the
    # update_H and step_D consults, so CARRIES_DEPOSIT_REPAIR = False is a statement
    # about driver.py rather than about either family
    # (conductive_fused_hd_pair.py:198, bfast_fused_hd_pair.py:177) -- and each
    # absorb row is what lets the seam loop ASK the product before it refuses it on
    # INSTALLABLE = False. THE TWO ARBITRATIONS DIFFER and both were measured through
    # the shipped composer: on an electric-conductivity run BOTH neighbours serve
    # (the released conductive_fused_electric_pair on step_D/update_E and
    # fused_magnetic_pair on step_B/update_H, because the magnetic curl there is the
    # ORDINARY one), so that span is a LOSS; on a BFAST run only ONE neighbour
    # installs -- bfast_fused_magnetic_pair carries no absorb row at all -- so that
    # span is a TIE, which the released precedent resolves for the incumbent.
    "conductive_fused_hd_pair",
    "bfast_fused_hd_pair",
    # 2026-09-17, THE THREE FAMILIES THE ALL-PATHS ROUND ROUTED. Each was CERTIFIED
    # and carried a released weld while holding no row in `FUSED_PAIR_ARMS`, so the
    # seam loop had no arm pair to absorb and could never ask the product at all --
    # `dispatch_reachability.metal_certified_but_not_installed` reported all three in
    # those words, and the board measured the cost at 25 census instances.
    #
    # THE FLAG IS FALSE ON ALL THREE, AND EACH MODULE ALREADY CARRIED THE MEASUREMENT
    # BEFORE THE ROW DID. The two Dcyl magnetic pairs sit on the B->H seam, where the
    # driver injects a MAGNETIC source between the two consults (driver.py:3283-3284);
    # `cylindrical_fused_magnetic_pair.py:705` records the corpus cost as ZERO because
    # all sixteen cylindrical rows declare electric sources only, and
    # `cylindrical_real_fused_magnetic_pair.py:522` records the same for its three.
    # `complex_conductive_fused_pair` sits on D->E, where the injection is ELECTRIC
    # (driver.py:3294-3299, and on a conductive row through
    # `_inject_electric_through_conductivity`); its four corpus rows are
    # magnetic-sourced, which its own predicate calls "exactly the polarity that makes
    # this cell's four corpus rows reachable". In all three the source clause refuses
    # an in-seam deposit BY NAME and asks `Fields` for the declaration rather than
    # inferring an empty set from silence, so the row widens what the composer may ASK
    # and nothing about what it may carry unbracketed.
    "cylindrical_complex_fused_magnetic_pair",
    "cylindrical_real_fused_magnetic_pair",
    "complex_conductive_fused_pair",
)


def test_no_other_metal_family_declares_the_repair():
    """The absorb table BOUNDS what may be flipped, so the two are asserted against
    each other rather than each against a typed list.

    THE DIRECTION THAT IS A DEFECT is a family declaring the repair that the seam
    loop cannot install: it would run unbracketed. That direction stays exact. The
    other direction — a routable family that deliberately does NOT carry the repair
    — became real on 2026-08-27 and is enumerated in
    :data:`ROUTED_WITHOUT_THE_REPAIR` with its reason, rather than being allowed to
    open a silent gap in the equality.
    """
    import ast
    import pathlib

    # KEYED BY THE ARM FAMILY NAME, NOT BY THE MODULE STEM, and the difference is
    # real rather than pedantic: `FUSED_PAIR_ARMS` is keyed by `FAMILY`, and the two
    # DIVERGE on the Dcyl complex products — module
    # `cylindrical_fused_electric_pair.py`, family
    # `cylindrical_complex_fused_electric_pair`. Keyed by the stem, such a module
    # reads as "declares the repair and has no absorb row", which is the exact defect
    # this assertion exists to catch, reported about a family that does have one.
    # (`build_fusion_matrix._wiring()` carries the same correction, for the same
    # module, and records that keying by the wrong name once disabled a check
    # silently.)
    here = pathlib.Path(__file__).parent / "metal_kernels"
    flipped = set()
    for path in sorted(here.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "CARRIES_DEPOSIT_REPAIR = True" not in text:
            continue
        declared = [node.value.value for node in ast.walk(ast.parse(text))
                    if isinstance(node, ast.Assign)
                    and any(getattr(target, "id", None) == "FAMILY"
                            for target in node.targets)
                    and isinstance(node.value, ast.Constant)]
        assert len(declared) == 1, (path.name, declared)
        flipped.add(declared[0])
    assert flipped <= set(launch.FUSED_PAIR_ARMS), (
        f"Metal families declaring the repair: {sorted(flipped)}; families the seam "
        f"loop can install: {sorted(launch.FUSED_PAIR_ARMS)}. A family in the first "
        f"set and not the second would run unbracketed")
    # ...and every routable family that does NOT declare it is accounted for by
    # name, so the subset above cannot quietly grow a family nobody decided about.
    assert set(launch.FUSED_PAIR_ARMS) - flipped == set(ROUTED_WITHOUT_THE_REPAIR), (
        sorted(set(launch.FUSED_PAIR_ARMS) - flipped))
    for family_name in ROUTED_WITHOUT_THE_REPAIR:
        modules = [path for path in sorted(here.glob("*.py"))
                   if f'FAMILY = "{family_name}"' in path.read_text(encoding="utf-8")]
        assert len(modules) == 1, (family_name, modules)
        assert "CARRIES_DEPOSIT_REPAIR = False" in modules[0].read_text(
            encoding="utf-8"), family_name


# ---------------------------------------------------------------------------
# 1. Opt-in: off is exactly what it was
# ---------------------------------------------------------------------------

def test_fuse_defaults_off_and_leaves_both_slots_on_their_separate_products():
    """``plan_step`` with no ``fuse`` argument composes what it composed before the
    block existed: the weld row is invisible because ``arms_for`` skips it."""
    _fields, plan = _plan((), fuse=False)
    assert plan.selected[CURL_SLOT] == "PML"
    assert plan.selected[UPDATE_SLOT] == "ordinary"
    assert not _fused_reasons(plan)
    assert not isinstance(plan.plans[UPDATE_SLOT], NoopPlan)


# ---------------------------------------------------------------------------
# 2. Opt-in, clean seam: the second slot is filled, with the cheap sentinel
# ---------------------------------------------------------------------------

def test_a_sourceless_seam_fills_both_slots_and_reports_both_in_replaces():
    """THE FAIL-CLOSED HALF OF THE CONTRACT. A pair that owned only ``step_B``
    would tell a reader -- and the residency verdict -- that ``update_H`` ran on
    the array path, when the kernel performed it inside the same launch."""
    _fields, plan = _plan((), fuse=True)
    assert plan.selected[CURL_SLOT] == plan.selected[UPDATE_SLOT] == LABEL
    assert isinstance(plan.plans[CURL_SLOT], family.MetalFusedMagneticPairPlan)
    assert isinstance(plan.plans[UPDATE_SLOT], NoopPlan)
    assert plan.plans[UPDATE_SLOT].absorbed_by is plan.plans[CURL_SLOT]
    assert CURL_SLOT in plan.replaces and UPDATE_SLOT in plan.replaces


def test_the_absorbed_slot_declares_the_pairs_passes_not_its_own_name():
    """``declaring_plan`` is what keeps the ``replaces_sub_steps`` default from
    turning into an UNDER-report: read bare, the sentinel says "update_H only" and
    ``zero_metal_B`` -- a pass the kernel performs -- reads as array-path."""
    _fields, plan = _plan((), fuse=True)
    for slot in (CURL_SLOT, UPDATE_SLOT):
        declared = launch.declaring_plan(plan.plans[slot])
        assert declared.replaces_sub_steps == family.REPLACES
        assert "zero_metal_B" in declared.replaces_sub_steps


def test_the_composer_credits_the_pairs_wall_clear_to_the_kernel_that_performs_it():
    """THE ACCOUNTING, ASSERTED ON THE COMPOSER'S OUTPUT rather than on the helper.

    A METALLIC configuration is what makes this measurable: ``zero_metal_B`` is then
    LIVE and the pair carries it (``REPLACES``). The residency verdict names every
    live pass that runs on the ARRAY PATH while a mirror is held, so if the leading
    slot's declaration were read off the repair WRAPPER -- which declares nothing --
    ``zero_metal_B`` would appear in that list, telling a reader the wall clear ran
    on the host when the kernel performed it inside the launch. Measured: with the
    wrapper read bare, ``zero_metal_B`` joins the two reasons below.

    ``fill_B`` is in the list and MUST be: the driver's source injection really does
    run on the array path inside this seam, and no Metal product carries it.
    """
    fields, pml = matrix.cart(boundaries="metallic")
    source = _magnetic_source(fields.grid)
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=[source], fuse=True, **_probes())
    assert isinstance(plan.plans[CURL_SLOT], deposit_repair.LeadingRepairPlan)
    assert "zero_metal_B" in plan.live, "the case cannot discriminate without a wall"
    stale = tuple(reason for reason in plan.residency.reasons
                  if "runs on the array path" in reason)
    assert not any(reason.startswith("zero_metal_B") for reason in stale), stale
    assert any(reason.startswith("fill_B") for reason in stale), stale


# ---------------------------------------------------------------------------
# 3. The deposit seam: refused as shipped, repaired once the flag says so
# ---------------------------------------------------------------------------

def test_an_in_seam_deposit_is_refused_by_name_while_the_flag_is_false(monkeypatch):
    """THE FALSE BRANCH, which is still what the other eleven Metal families return.

    Held at False by monkeypatch now that this family ships True -- the same code
    path in the opposite direction, and the reason is the product's own prose, which
    is what every one of those families' gates pins.
    """
    monkeypatch.setattr(family, "CARRIES_DEPOSIT_REPAIR", False)
    fields, pml = matrix.cart()
    source = _magnetic_source(fields.grid)
    assert source._n_source_points, "the case deposits nothing and cannot discriminate"
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=[source], fuse=True, **_probes())
    reasons = plan.reasons.get(f"fused_pair_{family.FAMILY}", ())
    assert any("is magnetic" in reason and "BETWEEN step_B and update_H" in reason
               for reason in reasons), reasons
    # And the seam stays on the separate products, which is the array-path answer.
    assert plan.selected[CURL_SLOT] == "PML"
    assert plan.selected[UPDATE_SLOT] == "ordinary"


def test_an_in_seam_deposit_lands_the_two_repair_plans_in_the_two_slots():
    """THE ONE THIS FILE EXISTS FOR: the Metal composer reaches ``deposit_repair``.

    Run against the SHIPPED flag. It used to patch ``CARRIES_DEPOSIT_REPAIR`` on the
    module object to reach this branch; the flip made the patch a no-op, and reading
    the shipped value is what makes this a statement about what the package does.
    """
    fields, pml = matrix.cart()
    source = _magnetic_source(fields.grid)
    assert source._n_source_points, "the case deposits nothing and cannot discriminate"
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=[source], fuse=True, **_probes())
    assert not _fused_reasons(plan), _fused_reasons(plan)

    leading = plan.plans[CURL_SLOT]
    trailing = plan.plans[UPDATE_SLOT]
    assert isinstance(leading, deposit_repair.LeadingRepairPlan)
    assert isinstance(trailing, deposit_repair.TrailingRepairPlan)
    # The two wrappers name the SAME launch, and it is the fused product's plan.
    assert isinstance(leading.inner, family.MetalFusedMagneticPairPlan)
    assert trailing.absorbed_by is leading.inner
    assert leading.pair == "B" and trailing.slot == UPDATE_SLOT
    # The seam it captured is this source, not the whole declared list.
    assert leading._sources == (source,)
    # Both slots are reported, and both report the pair's passes.
    assert plan.selected[CURL_SLOT] == plan.selected[UPDATE_SLOT] == LABEL
    assert CURL_SLOT in plan.replaces and UPDATE_SLOT in plan.replaces
    for slot in (CURL_SLOT, UPDATE_SLOT):
        assert (launch.declaring_plan(plan.plans[slot]).replaces_sub_steps
                == family.REPLACES)


def test_an_electric_source_does_not_make_the_magnetic_seam_carry_a_repair():
    """A source on the far side of the step is not this seam's deposit; the second
    slot must stay the sentinel rather than acquiring a repair with nothing to do."""
    fields, pml = matrix.cart()
    from meep_gpu.sources import GaussianPulsedSource

    electric = GaussianPulsedSource(grid=fields.grid, component="Ez",
                                    center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                    frequency=1.0, fwidth=0.2, amplitude=1.0)
    assert str(electric.field_type) != deposit_repair.MAGNETIC_FIELD_TYPE
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=[electric], fuse=True, **_probes())
    assert isinstance(plan.plans[UPDATE_SLOT], NoopPlan)
    assert not isinstance(plan.plans[CURL_SLOT], deposit_repair.LeadingRepairPlan)


# ---------------------------------------------------------------------------
# 3b. The two families routed on 2026-08-28, on the seams they actually serve
# ---------------------------------------------------------------------------
#
# Each is driven on the configuration its arms win, with a deposit of its OWN seam's
# field type. The claim is not "the flag is True" -- that is asserted from source in
# `test_no_other_metal_family_declares_the_repair` -- but that a real Metal
# composition puts the two repair plans in the two slots for these families too.

#: family module, curl slot, update slot, seam, the configuration builder, and a
#: factory for a deposit in that seam. The first two rows are UNFOLDED -- their
#: predicates refuse a mirrored axis by name -- and the last two are the FOLDED pair,
#: driven on a mirrored grid where both post-injection fills are live in the seam.
def _electric_source(grid):
    from meep_gpu.sources import GaussianPulsedSource

    return GaussianPulsedSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                                size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2,
                                amplitude=1.0)


def _folded_magnetic_source(grid):
    """``Hy``, not ``Hz``: an odd component centred on an even mirror plane is refused
    by ``sources._validate_symmetry_parity`` before it can deposit anything, and a
    source with no points would let every assertion below pass while measuring
    nothing."""
    return VolumeSource(grid=grid, component="Hy", center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


ROUTED_2026_08_28 = (
    ("complex_fused_magnetic_pair", "step_B", "update_H", "B",
     lambda: matrix.cart(complex_storage=True), _magnetic_source,
     ("complex/Bloch", "complex/Bloch")),
    ("fused_dispersive_pair", "step_D", "update_E", "D",
     lambda: matrix.dispersive(matrix.cart()), _electric_source,
     ("PML", "dispersive PML E")),
    # THE TWO FOLDED FAMILIES, same day, once `deposit_repair.repair_cells` closed the
    # fill images. Their configuration is the one thing new here: on `matrix.folded()`
    # the seam carries `fill_symmetry_bc_*` and `fill_folded_far_ghosts_*` between the
    # halves, so a bracket that landed but repaired only the deposit index would be
    # wrong in a way this composition test cannot see and `test_deposit_repair.py`'s
    # folded cases can.
    ("folded_fused_magnetic_pair", "step_B", "update_H", "B",
     matrix.folded, _folded_magnetic_source, ("folded", "folded")),
    ("folded_fused_pair", "step_D", "update_E", "D",
     matrix.folded, _electric_source, ("folded", "folded")),
)


@pytest.mark.parametrize(
    "family_name,curl_slot,update_slot,seam,build,source_factory,expected_arms",
    ROUTED_2026_08_28, ids=[row[0] for row in ROUTED_2026_08_28])
def test_the_2026_08_28_families_are_bracketed_on_the_seam_they_serve(
        family_name, curl_slot, update_slot, seam, build, source_factory,
        expected_arms):
    import importlib

    module = importlib.import_module(f"meep_gpu.metal_kernels.{family_name}")
    assert module.CARRIES_DEPOSIT_REPAIR is True
    assert launch.FUSED_PAIR_ARMS[family_name] == expected_arms
    assert launch.FUSED_PAIR_SEAMS[curl_slot] == (update_slot, seam)

    # THE PRECONDITION: the arms this pair absorbs really do win their slots, or the
    # fusion below would refuse for the right reason and measure nothing.
    fields, pml = build()
    unfused = launch.plan_step(fields, pml, residency=device.Residency(), sources=(),
                               fuse=False, **_probes())
    assert (unfused.selected.get(curl_slot),
            unfused.selected.get(update_slot)) == expected_arms

    fields, pml = build()
    source = source_factory(fields.grid)
    assert source._n_source_points, "the case deposits nothing and cannot discriminate"
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=[source], fuse=True, **_probes())
    reasons = plan.reasons.get(f"fused_pair_{family_name}", ())
    assert not reasons, reasons
    leading = plan.plans[curl_slot]
    trailing = plan.plans[update_slot]
    assert isinstance(leading, deposit_repair.LeadingRepairPlan)
    assert isinstance(trailing, deposit_repair.TrailingRepairPlan)
    assert trailing.absorbed_by is leading.inner
    assert leading.pair == seam and trailing.slot == update_slot
    assert leading._sources == (source,)
    # Both slots report the pair's passes, so the absorbed slot cannot read as
    # array-path work.
    for slot in (curl_slot, update_slot):
        assert (launch.declaring_plan(plan.plans[slot]).replaces_sub_steps
                == module.REPLACES)


@pytest.mark.parametrize(
    "family_name,curl_slot,update_slot,seam,build,source_factory,expected_arms",
    ROUTED_2026_08_28, ids=[row[0] for row in ROUTED_2026_08_28])
def test_the_2026_08_28_families_still_refuse_a_deposit_with_the_flag_held_false(
        family_name, curl_slot, update_slot, seam, build, source_factory,
        expected_arms, monkeypatch):
    """The other direction, which is what says the flag is what admits the deposit
    rather than something else having changed underneath it."""
    import importlib

    module = importlib.import_module(f"meep_gpu.metal_kernels.{family_name}")
    monkeypatch.setattr(module, "CARRIES_DEPOSIT_REPAIR", False)
    fields, pml = build()
    source = source_factory(fields.grid)
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=[source], fuse=True, **_probes())
    reasons = plan.reasons.get(f"fused_pair_{family_name}", ())
    assert any("driver injects it BETWEEN" in reason for reason in reasons), reasons
    # ...and the seam stays on the separate arms, which is the array-path answer.
    assert (plan.selected[curl_slot], plan.selected[update_slot]) == expected_arms


def test_a_clean_seam_on_the_new_families_takes_the_sentinel_not_a_repair():
    """No deposit means nothing to carry: the second slot must be the cheap sentinel,
    so the repair cannot be read as unconditional overhead of routing a family."""
    for family_name, curl_slot, update_slot, _seam, build, _factory, _arms in (
            ROUTED_2026_08_28):
        fields, pml = build()
        plan = launch.plan_step(fields, pml, residency=device.Residency(),
                                sources=(), fuse=True, **_probes())
        assert not plan.reasons.get(f"fused_pair_{family_name}", ()), family_name
        assert isinstance(plan.plans[update_slot], NoopPlan), family_name
        assert plan.plans[update_slot].absorbed_by is plan.plans[curl_slot], family_name


# ---------------------------------------------------------------------------
# 4. The absorb clause, which is the only thing standing between the block and
#    a slot no arm admitted
# ---------------------------------------------------------------------------

def test_a_specialized_family_owning_the_curl_keeps_the_pair_out_by_predicate():
    """A conductive MAGNETIC run hands ``step_B`` to the conductive family, and the
    fused pair's own CONJUNCTION is what refuses -- its curl half is the ``PML``
    arm's predicate, which the conductivity refuses by name.

    Recorded as a predicate refusal rather than as an absorb refusal ON PURPOSE:
    containment is by construction here (``arms.ArmSpec.is_weld``), so a
    configuration another arm wins is one the weld's conjunction already refuses.
    That is why the absorb clause is exercised below through a PLANTED ambiguity --
    the one way a slot goes unselected while the pair still admits.
    """
    fields, pml = matrix.conductive(matrix.cart(), magnetic=True)
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=(), fuse=True, **_probes())
    assert plan.selected[CURL_SLOT] == "conductive PML curl"
    assert plan.selected[UPDATE_SLOT] == "ordinary"
    assert not isinstance(plan.plans[UPDATE_SLOT], NoopPlan)
    assert any("this curl belongs to the conductive PML product" in reason
               for reason in _fused_reasons(plan)), _fused_reasons(plan)


def test_a_pair_may_not_absorb_a_slot_the_arm_table_left_unselected():
    """THE ABSORB CLAUSE, on the case it exists for: an AMBIGUOUS constitutive slot.

    A second admitter is planted on ``update_H``, so ``_select_slot`` leaves that
    slot empty by clause (b) -- the array path, which is always correct. The fused
    pair's own predicate still admits, and without ``_pair_may_absorb`` it would
    claim BOTH slots and run the very constitutive product the ambiguity refused to
    pick. The refusal must name the slot and leave the curl slot alone.
    """
    from meep_gpu.metal_kernels.coverage import Coverage

    spec = arms.register(
        family="_test_double_constitutive", slot=UPDATE_SLOT, label="planted",
        coverage=lambda ctx, slot: Coverage(True, ()),
        plan=lambda ctx, slot: object(),
        prefix="planted: ", noun="planted constitutive", wired=True)
    try:
        fields, pml = matrix.cart()
        plan = launch.plan_step(fields, pml, residency=device.Residency(),
                                sources=(), fuse=True, **_probes())
        assert UPDATE_SLOT not in plan.plans, "the planted ambiguity did not fire"
        assert any(UPDATE_SLOT in reason and "not selected by any arm" in reason
                   for reason in _fused_reasons(plan)), _fused_reasons(plan)
        # The curl slot keeps the product the table gave it; nothing was absorbed.
        assert plan.selected[CURL_SLOT] == "PML"
        assert not isinstance(plan.plans[CURL_SLOT],
                              family.MetalFusedMagneticPairPlan)
    finally:
        arms._REGISTRY[UPDATE_SLOT].remove(spec)


#: Every registered Metal weld that spans a curl slot and its constitutive slot and
#: has NO absorb declaration -- derived from the table rather than typed, so a family
#: added later is covered without anyone remembering to list it.
UNDECLARED_WELDS = tuple(sorted({
    spec.family for curl, (update, _pair) in launch.FUSED_PAIR_SEAMS.items()
    for spec in arms.registered(curl)
    if spec.is_weld and update in spec.replaces
    and spec.family not in launch.FUSED_PAIR_ARMS}))


def test_the_undeclared_weld_sweep_is_not_empty():
    """A parametrisation over an empty list passes without running, so the list is
    asserted non-empty here rather than trusted below.

    THE FLOOR IS ON THE POPULATION, NOT ON THE SWEEP, since 2026-08-27. It used to
    read ``>= 12``, which is the number that was undeclared on the day it was
    written; routing two families moved two of them into ``FUSED_PAIR_ARMS`` and
    would have failed a bound that cannot tell "a weld disappeared" from "a weld was
    routed". The sum is what does not move when a family is routed, so it is what
    carries the floor — and the sweep is still asserted non-empty in its own right.
    """
    assert UNDECLARED_WELDS
    assert len(UNDECLARED_WELDS) + len(launch.FUSED_PAIR_ARMS) >= 13, (
        UNDECLARED_WELDS, sorted(launch.FUSED_PAIR_ARMS))


@pytest.mark.parametrize("family_name", UNDECLARED_WELDS)
def test_every_undeclared_weld_is_refused_by_name_rather_than_absorbing(family_name):
    """Twelve other Metal products weld a curl slot to its constitutive slot and none
    has had the arm its kernel implements established. Each must REFUSE, not fall
    through to a default."""
    assert family_name not in launch.FUSED_PAIR_ARMS
    fields, pml = matrix.cart()
    plan = launch.plan_step(fields, pml, residency=device.Residency(),
                            sources=(), fuse=True, **_probes())
    reasons = plan.reasons.get(f"fused_pair_{family_name}", ())
    assert any("no absorb declaration" in reason for reason in reasons), (
        family_name, reasons)


# --------------------------------------------------------------------------
# THE H->D SEAM ROW, 2026-09-04 -- landed ahead of its product
# --------------------------------------------------------------------------
#
# What the row buys before any Metal H->D kernel exists is the ARBITRATION and the
# KEY ORDER. Both are pinned here, in the file that already pins this table.


def test_the_h_to_d_seam_row_is_last_because_placed_first_it_would_displace_a_released_product():
    """The H->D row is APPENDED, and the key order is a correctness property.

    ``_install_fused_pairs`` iterates ``FUSED_PAIR_SEAMS.items()`` in dict order and
    reads the LIVE ``selected`` as it goes. The H->D seam's FIRST slot is
    ``update_H``, which is the B->H seam's SECOND slot, and its SECOND slot is
    ``step_D``, which is the D->E seam's FIRST. Offered first, an H->D product would
    find both of those slots still carrying ARM labels -- neither incumbent having
    run -- ``_pair_may_absorb`` would let it take them, and a released, gate-passing
    product would lose its seam on every row that reached it. Offered last it finds a
    fused label there and is refused by name.

    Pinned as an ORDER rather than argued in a comment because dict order is
    invisible at the call site: a row moved while tidying the table would be a silent
    behaviour change on every corpus row with no test to notice.
    """
    order = list(launch.FUSED_PAIR_SEAMS)
    assert order.index("update_H") > order.index("step_B"), order
    assert order.index("update_H") > order.index("step_D"), order
    assert order[-1] == "update_H", order


def test_the_h_to_d_row_names_the_withdraw_seam_and_not_a_deposit_list():
    """Three kinds of seam name, and this row is the third.

    ``'B'``/``'D'`` name a ``deposit_repair`` INJECTION list. This seam has none --
    nothing is injected between ``update_H`` and ``step_D`` -- and a letter here
    would be worse than useless: ``deposit_repair._in_seam_indexed`` reads any
    ``pair != "B"`` as the ELECTRIC list, so the seam would be bracketed with a
    repair for an injection that happens in the NEXT seam. What the driver does put
    between the two consults is the electric ``withdraw``, and the name says so.
    """
    from meep_gpu import withdraw_hoist  # noqa: PLC0415

    assert launch.FUSED_PAIR_SEAMS["update_H"] == ("step_D", withdraw_hoist.SEAM)
    assert withdraw_hoist.SEAM not in deposit_repair.SEAMS
    assert withdraw_hoist.SEAM_SPAN == ("update_H", "step_D")


def test_the_h_to_d_seam_is_installed_through_the_withdraw_hoist_not_the_deposit_bracket():
    """``_install_fused_pair`` routes the seam NAME, so both tracks read one protocol.

    THE BRANCH IS IN THE INSTALLER AND NOT IN A FAMILY -- the same argument that
    keeps the deposit bracket out of the families. A family free to decide which
    mechanism its seam carries is a family that can decide wrong, and the wrong
    answer here (the deposit bracket, on a seam with no injection) computes and
    converges.
    """
    from meep_gpu import withdraw_hoist  # noqa: PLC0415

    class Pair:
        label = "h to d weld"

    plans, selected = {}, {}
    launch._install_fused_pair(plans, selected, None, None, (), withdraw_hoist.SEAM,
                               "update_H", "step_D", "h to d weld", Pair())
    assert isinstance(plans["update_H"], withdraw_hoist.LeadingWithdrawPlan)
    assert plans["update_H"].span == ("update_H", "step_D")
    assert plans["update_H"].placement == withdraw_hoist.BEFORE_UPDATE_H
    # NO TRAILING PLAN: nothing is injected here, so the second slot holds the no-op
    # sentinel and never ``TrailingRepairPlan``.
    assert isinstance(plans["step_D"], NoopPlan)
    assert not isinstance(plans["step_D"], deposit_repair.TrailingRepairPlan)
    assert selected == {"update_H": "h to d weld", "step_D": "h to d weld"}


def test_the_only_h_to_d_bidder_is_registered_and_installs_on_nothing(monkeypatch):
    """WHAT CHANGED 2026-09-05: the seam has a bidder WITH an absorb row, and it still
    fuses nothing.

    This test used to pin the row's EMPTINESS, then the bidder's missing absorb row.
    Both were the right claim in their day and both are the shape that goes red when
    the gap closes -- ``metal_kernels/fused_hd_pair`` landed, then its
    ``FUSED_PAIR_ARMS`` row did. So the claim is now the strongest one: exactly ONE
    product bids, the seam loop ASKS it, and the composition is unchanged, held out by
    TWO independent brakes that are each driven here.

    THE TWO BRAKES. The product declares ``INSTALLABLE = False`` with the measured
    verdict as its reason -- a fact about the PRODUCT, reported on every
    configuration. And with that flag out of the way (flipped here, on the live
    module) it is STILL refused, by ``_pair_may_absorb`` naming the released B->H
    pair that installed first: the end-edge guard in ``_neighbouring_seam_claimant``
    never asks the B->H pair whether this product claims ``update_H``, so the pair
    installs, writes its label, and the arm table gives this product nothing. That
    second brake is the one the tie measurement found the tree did not have on
    2026-09-04, and this test is red without it.
    """
    bidders = [spec for spec in arms.registered("update_H")
               if spec.is_weld and "step_D" in spec.replaces]
    # TEN BIDDERS SINCE THE 2026-09-08 WIRING ROUND: three from 2026-09-06 (the
    # Cartesian product, the Dcyl m = 0 one and the Dcyl complex one), the two wired
    # earlier that day (the FOLDED Cartesian product and the COMPLEX/Bloch Cartesian
    # one), the three tails of 2026-09-07 -- the FOLDED COMPLEX one and the two BETA
    # ones -- and the seam's last two cells this round closed, the CONDUCTIVE one and
    # the BFAST one. They partition the seam by storage, fold, beta, conductivity,
    # ``grid.bfast_active`` and coordinate system, each through its own half's
    # predicate, so no configuration admits two of them; the Cartesian real unfolded
    # one is what this test drives, and its own cell is what ``matrix.cart()`` builds.
    #
    # THE TWO BETA BIDDERS EACH COVER TWO CELLS through ONE arm row: their predicate
    # resolves the variant from the grid FIRST and asks only that variant's two
    # parent predicates, so the partition is exclusive inside the family as well as
    # between families. Their second cell is a ``FUSED_PAIR_EXTRA_ARMS`` row.
    assert sorted(spec.family for spec in bidders) == [
        "beta_complex_fused_hd_pair", "beta_real_fused_hd_pair",
        "bfast_fused_hd_pair", "complex_fused_hd_pair",
        "conductive_fused_hd_pair", "cylindrical_complex_fused_hd_pair",
        "cylindrical_real_fused_hd_pair", "folded_complex_fused_hd_pair",
        "folded_fused_hd_pair", "fused_hd_pair"], bidders
    # THE PARTITION, ASSERTED RATHER THAN DESCRIBED, and over the CELLS rather than
    # the families: every (constitutive arm, curl arm) pair any bidder may absorb --
    # its primary ``FUSED_PAIR_ARMS`` row plus each ``FUSED_PAIR_EXTRA_ARMS`` row --
    # is distinct across the whole seam, so an eleventh bidder colliding with one of
    # these fails here rather than reaching install time with two admitted products.
    cells = []
    for spec in bidders:
        cells.append(launch.FUSED_PAIR_ARMS[spec.family])
        cells.extend(launch.FUSED_PAIR_EXTRA_ARMS.get(spec.family, ()))
    assert len(cells) == len(set(cells)), sorted(
        cell for cell in set(cells) if cells.count(cell) > 1)
    # AND THE COMPOSITION CLAIM THE TEST'S NAME MAKES, over all ten rather than only
    # the one it then drives: none of them is wired, and every one is refused on its
    # own ``INSTALLABLE = False`` before any configuration is asked.
    assert not [spec.family for spec in bidders if spec.wired]
    assert not [spec.family for spec in bidders
                if launch._declared_uninstallable(spec) is None]
    bidder = next(spec for spec in bidders if spec.family == "fused_hd_pair")
    assert bidder.wired is False
    assert launch.FUSED_PAIR_ARMS[bidder.family] == ("ordinary", "PML")
    assert launch._declared_uninstallable(bidder) is not None

    fields, pml = matrix.cart()
    shipped = launch.plan_step(fields, pml, residency=device.Residency(),
                               sources=(), fuse=True, **_probes())
    # The seam stays unfused: whatever holds ``update_H`` also holds ``step_B`` (the
    # released B->H pair) and whatever holds ``step_D`` also holds ``update_E``, so no
    # label spans this seam.
    assert shipped.selected["update_H"] == shipped.selected["step_B"] == LABEL
    assert shipped.selected["step_D"] == shipped.selected["update_E"] != LABEL
    assert bidder.label not in set(shipped.selected.values())
    reason = shipped.reasons.get("fused_pair_fused_hd_pair")
    assert reason and reason[0].startswith(
        "fused_hd_pair declares INSTALLABLE = False: "), reason

    # THE SECOND BRAKE, ARMED: drop the flag and the composition does not move.
    monkeypatch.setattr(hd, "INSTALLABLE", True)
    assert launch._declared_uninstallable(bidder) is None
    fields, pml = matrix.cart()
    unflagged = launch.plan_step(fields, pml, residency=device.Residency(),
                                 sources=(), fuse=True, **_probes())
    assert unflagged.selected == shipped.selected
    reason = unflagged.reasons.get("fused_pair_fused_hd_pair")
    assert reason and reason[0].startswith(
        f"update_H was selected by the {LABEL!r} arm"), reason


def test_the_neighbouring_seam_rule_refuses_a_product_that_would_take_a_released_slot(
        monkeypatch):
    """The rule, armed against a stub H->D product.

    ``_pair_may_absorb`` already refuses an H->D product on a row where the B->H pair
    installed FIRST, but it refuses with an arm-label reason and only because of the
    table's key order. The rule below is what says the measured thing: over
    ``step_B..update_E`` launches are ``4 - #pairs``, so a product taking a slot from
    a neighbouring seam can only tie or lose. The claimant is the D->E pair, named
    through the seam this span's LAST slot opens: an earlier-neighbour half naming
    the B->H pair was added and retired on 2026-09-04, because the only product class
    it could actually fire on was E->P, where it reversed a released ruling. The
    control is the SAME question asked on a seam with no neighbouring bidder.
    """
    fields, pml = matrix.cart()
    context = arms.StepContext(fields, pml, residency=device.Residency(),
                               contract_variants=(shaders.CONTRACT_OFF,), sources=(),
                               extra=_probes())
    found = launch._neighbouring_seam_claimant(
        context, "h_to_d_stub", "update_H", "step_D", ("update_H", "step_D"))
    assert found is not None, "fused_electric_pair admits a plain Cartesian PML grid"
    claimant, refusal = found
    assert claimant == "fused_electric_pair"
    assert "step_D/update_E seam" in refusal
    assert "4 - (installed pairs)" in refusal
    assert "_neighbouring_seam_claimant" in refusal and "slot arbitration" in refusal
    # THE CONTROL: the released B->H pair itself is NOT withheld -- and since
    # 2026-09-05 that no longer rests on the bidder's `INSTALLABLE = False`. The B->H
    # span is an END EDGE of `FUSED_PAIR_SEAMS` (`step_B` sits in one row), so the
    # rule never asks the question of it at all; the flag is belt and braces. Both
    # states of the flag are driven here because the tie measurement of 2026-09-04
    # found this exact control passing ONLY on the flag, with a registered admitting
    # stand-in costing the released pair its seam on every row.
    for flag in (hd.INSTALLABLE, not hd.INSTALLABLE):
        monkeypatch.setattr(hd, "INSTALLABLE", flag)
        assert launch._neighbouring_seam_claimant(
            context, "fused_magnetic_pair", "step_B", "update_H",
            ("step_B", "update_H")) is None, flag


# ---------------------------------------------------------------------------
# THE END-EDGE GUARD, 2026-09-05 -- both directions, driven through the composer
# ---------------------------------------------------------------------------
#
# WHAT WAS MISSING, AND WHY EVERY TEST BELOW ASSERTS BOTH SIDES. Until this round the
# tests of the neighbouring-seam rule asserted who was WITHHELD and never who
# INSTALLS. Measured on 2026-09-04 through this composer: with the ``update_H`` seam
# row in the table, registering an H->D stand-in whose predicate admits cost the
# RELEASED B->H pair its seam on every row it reached -- even a stand-in with no
# absorb declaration, which can never install (arrangement F: 3 launches / 1 seam
# against the shipped 2 / 2 on the 155-row ``cart_pml_real`` cell) -- and the only
# thing standing in the way was ``fused_hd_pair``'s own ``INSTALLABLE = False``. The
# guard is R1: a span is asked the question only when both of its END slots have
# degree >= 2 in ``FUSED_PAIR_SEAMS``; the tie on the 22 rows where only the B->H
# pair serves is then resolved by ``_pair_may_absorb`` reading the live ``selected``
# after that pair installs first. Each test below asserts the installed side in the
# same breath as the withheld one.

SLOTS = ("step_B", "update_H", "step_D", "update_E")
SEAM_PAIRS = (("step_B", "update_H"), ("update_H", "step_D"), ("step_D", "update_E"))
STAND_IN = "h_to_d_stand_in"
STAND_IN_LABEL = "fused H/D pair (stand-in)"


class _StandInPlan:
    """A planning stand-in for an H->D product: never run, counted once."""

    launches_per_run = 1
    performs_device_work = True
    replaces_sub_steps = ("update_H", "step_D")
    repair_paths = ()

    def run(self, *args, **kwargs):  # pragma: no cover - the composer never runs it
        raise AssertionError("the stand-in is a planning stand-in and is never run")


@contextlib.contextmanager
def _stand_in(admit=True, absorb=None, installable=None):
    """Register an H->D stand-in on ``update_H`` for the duration of a block.

    ``absorb`` is its ``FUSED_PAIR_ARMS`` row (``None`` leaves it without one, which
    is the tie measurement's arrangement F); ``installable`` is its module-level
    ``INSTALLABLE`` (``None`` declares nothing, which the composer reads as
    installable). The predicate's ``__module__`` is pointed at a real ``sys.modules``
    entry because that is where ``_declared_uninstallable`` reads the flag from.
    """
    name = f"meep_gpu_test_{STAND_IN}"
    module = types.ModuleType(name)
    if installable is not None:
        module.INSTALLABLE = installable
        module.INSTALLABLE_REASON = "a stand-in declaring itself uninstallable"

    def coverage(context, slot):
        return Coverage(admit, () if admit
                        else ("the stand-in refuses this configuration by name",))

    def plan(context, slot):
        return _StandInPlan()

    coverage.__module__ = plan.__module__ = name
    module.coverage, module.plan = coverage, plan
    sys.modules[name] = module
    spec = arms.register(STAND_IN, "update_H", STAND_IN_LABEL, coverage=coverage,
                         plan=plan, wired=False, replaces=("update_H", "step_D"))
    if absorb is not None:
        launch.FUSED_PAIR_ARMS[STAND_IN] = tuple(absorb)
    try:
        yield spec
    finally:
        arms._REGISTRY["update_H"].remove(spec)
        launch.FUSED_PAIR_ARMS.pop(STAND_IN, None)
        sys.modules.pop(name, None)


def _composition(plan):
    """Launches over the four slots, which seams are served, and who holds each slot.

    Launches are summed over the DISTINCT declaring owners of the four slots, so a
    pair counts once -- the tie measurement's accounting, which the composer's own
    ``launches = 4 - (installed pairs)`` algebra was checked against.
    """
    owners = []
    for slot in SLOTS:
        occupant = plan.plans.get(slot)
        if occupant is None:
            continue
        owner = launch.declaring_plan(occupant)
        if not any(owner is seen for seen in owners):
            owners.append(owner)
    seams = {}
    for first, second in SEAM_PAIRS:
        a, b = plan.plans.get(first), plan.plans.get(second)
        seams[f"{first}->{second}"] = bool(
            a is not None and b is not None
            and launch.declaring_plan(a) is launch.declaring_plan(b))
    return {"selected": {slot: plan.selected.get(slot) for slot in SLOTS},
            "launches": sum(int(getattr(owner, "launches_per_run", 1))
                            for owner in owners),
            "seams": seams}


def _one_electric_source(fields):
    """ONE real electric source at the origin, from the engine's own class.

    This is the corpus condition on the 24 rows where only the B->H pair serves: an
    in-seam electric deposit that an off-diagonal constitutive cannot repair, so the
    D->E product refuses the seam. With ``sources=()`` the same fixture admits its
    D->E product and is the wrong cell.
    """
    return (GaussianPulsedSource(grid=fields.grid, frequency=0.15, fwidth=0.1,
                                 component="Ez", center=(0.0, 0.0, 0.0),
                                 size=(0.0, 0.0, 0.0)),)


def _plan_on(build, sources=None):
    fields, pml = build()
    src = _one_electric_source(fields) if sources == "one_electric" else ()
    return launch.plan_step(fields, pml, residency=device.Residency(), sources=src,
                            fuse=True, **_probes())


#: The three row classes of the tie measurement, each as the fixture its cell was
#: driven on: (name, builder, sources, corpus rows in the cell, the stand-in's absorb
#: row for that cell's arms).
ROW_CLASSES = {
    "both_incumbents_install": (lambda: matrix.cart(), None, 155,
                                ("ordinary", "PML")),
    "only_b_to_h_installs": (lambda: matrix.cart(rows={"Ex": ("Ey",)}),
                             "one_electric", 9, ("ordinary", "PML")),
    "neither_installs": (lambda: matrix.nonlinear(matrix.cart()), None, 2,
                         ("nonlinear PML magnetic", "nonlinear PML curl")),
}


def test_slot_degrees_are_read_off_the_table_and_name_the_end_edges():
    """THE GUARD IS DERIVED, NOT SPELLED. This is what fails the day a fifth seam row
    is priced without thinking about position."""
    degrees = launch._slot_degrees(launch.FUSED_PAIR_SEAMS)
    assert degrees == {"step_B": 1, "update_H": 2, "step_D": 2, "update_E": 1}, degrees
    spans = {curl: (curl, update) for curl, (update, _p) in launch.FUSED_PAIR_SEAMS.items()}
    end_edges = {curl for curl, span in spans.items()
                 if launch._is_end_edge_span(span, launch.FUSED_PAIR_SEAMS)}
    assert end_edges == {"step_B", "step_D"}, end_edges
    assert not launch._is_end_edge_span(spans["update_H"], launch.FUSED_PAIR_SEAMS)
    # And a span that REACHES an end of the path is an end edge however long it is:
    # the four-slot weld is the one shape the arbitration was always meant to let
    # through.
    assert launch._is_end_edge_span(SLOTS, launch.FUSED_PAIR_SEAMS)


def test_an_end_edge_span_is_never_asked_even_when_an_interior_claimant_admits():
    """Both released pairs sit on end edges; neither yields to an admitting H->D
    stand-in that IS installable. The armed control is the stand-in's own span, which
    is interior and DOES yield -- to the D->E pair, through the later half."""
    fields, pml = matrix.cart()
    context = arms.StepContext(fields, pml, residency=device.Residency(),
                               contract_variants=(shaders.CONTRACT_OFF,), sources=(),
                               extra=_probes())
    with _stand_in(admit=True, absorb=("ordinary", "PML")):
        for family_name, curl, update in (("fused_magnetic_pair", "step_B", "update_H"),
                                          ("fused_electric_pair", "step_D", "update_E")):
            assert launch._neighbouring_seam_claimant(
                context, family_name, curl, update, (curl, update)) is None, family_name
        asked = launch._neighbouring_seam_claimant(
            context, STAND_IN, "update_H", "step_D", ("update_H", "step_D"))
        assert asked is not None and asked[0] == "fused_electric_pair", asked


def test_registering_a_claimant_that_cannot_install_changes_nothing():
    """ARRANGEMENT F, THE INVARIANT STATED AS A PROPERTY: a registered stand-in whose
    predicate ADMITS and which cannot install (no absorb row) leaves the number of
    installed pairs -- and every slot's owner -- exactly as shipped, on a both-install
    fixture and on an only-B->H one. It was red on 2026-09-04: 3 / 1 against 2 / 2,
    and 4 / 0 against 3 / 1. Paired with arrangement G (the predicate REFUSES), which
    must also be unchanged, so the test cannot pass by the stand-in being invisible.
    """
    for name in ("both_incumbents_install", "only_b_to_h_installs"):
        build, sources, _rows, _absorb = ROW_CLASSES[name]
        shipped = _composition(_plan_on(build, sources))
        expected_launches = 2 if name == "both_incumbents_install" else 3
        assert shipped["launches"] == expected_launches, (name, shipped)
        assert shipped["seams"]["step_B->update_H"], (name, shipped)
        with _stand_in(admit=True, absorb=None):
            f_plan = _plan_on(build, sources)
        assert _composition(f_plan) == shipped, (name, _composition(f_plan))
        assert "has no absorb declaration" in f_plan.reasons[f"fused_pair_{STAND_IN}"][0]
        with _stand_in(admit=False, absorb=("ordinary", "PML")):
            g_plan = _plan_on(build, sources)
        assert _composition(g_plan) == shipped, (name, _composition(g_plan))
        assert "refuses this configuration" in g_plan.reasons[f"fused_pair_{STAND_IN}"][0]
        # And the tree afterwards is the tree before: registration left no residue.
        assert _composition(_plan_on(build, sources)) == shipped


@pytest.mark.parametrize("row_class", sorted(ROW_CLASSES))
def test_the_h_to_d_span_yields_or_installs_by_row_class(row_class):
    """An INSTALLABLE stand-in (predicate admits, absorb row matches the cell's arms),
    on the three row classes of the tie measurement, asserting who holds every slot.

    * both install -- the B->H pair holds ``step_B``/``update_H``, the D->E pair
      ``step_D``/``update_E``; the stand-in is refused by ``_pair_may_absorb`` by
      name; 2 launches.
    * only B->H installs -- the B->H pair still holds both of its slots; the stand-in
      is refused by name; 3 launches; the one seam served is ``step_B->update_H``.
      This is the TIE, resolved for the released incumbent.
    * neither installs -- the stand-in INSTALLS: 3 launches, the seam served is
      ``update_H->step_D``. Without this leg the rule could be a blanket veto and
      nothing would notice; it is the 2-row nonlinear cell, the one measured gain.
    """
    build, sources, _rows, absorb = ROW_CLASSES[row_class]
    shipped = _composition(_plan_on(build, sources))
    with _stand_in(admit=True, absorb=absorb):
        plan = _plan_on(build, sources)
    got = _composition(plan)
    reason = plan.reasons.get(f"fused_pair_{STAND_IN}", ("",))[0]
    if row_class == "both_incumbents_install":
        assert shipped["launches"] == 2 and got == shipped, got
        assert got["selected"]["step_B"] == got["selected"]["update_H"] == LABEL
        assert got["selected"]["step_D"] == got["selected"]["update_E"] != LABEL
        assert reason.startswith(f"update_H was selected by the {LABEL!r} arm"), reason
    elif row_class == "only_b_to_h_installs":
        assert shipped["launches"] == 3 and got == shipped, got
        assert got["selected"]["step_B"] == got["selected"]["update_H"] == LABEL
        assert got["seams"] == {"step_B->update_H": True, "update_H->step_D": False,
                                "step_D->update_E": False}
        assert reason.startswith(f"update_H was selected by the {LABEL!r} arm"), reason
    else:
        assert shipped["launches"] == 4 and not any(shipped["seams"].values()), shipped
        assert got["launches"] == 3, got
        assert got["seams"] == {"step_B->update_H": False, "update_H->step_D": True,
                                "step_D->update_E": False}
        assert got["selected"]["update_H"] == got["selected"]["step_D"] == STAND_IN_LABEL
        assert f"fused_pair_{STAND_IN}" not in plan.reasons


def test_the_h_to_d_seam_row_order_is_what_resolves_the_tie_for_the_released_incumbent():
    """THE CONSEQUENCE THE ORDER TEST ABOVE PINS BY POSITION ALONE, MEASURED.

    Under the end-edge guard the key order no longer decides anything on a row where
    BOTH incumbents install (the B->H pair is never asked; the stand-in is refused by
    the later half either way) -- asserted below as 2 / 2 in both orders. Where it
    still decides is the TIE: on a row where only the B->H pair serves, the row
    offered LAST leaves the seam with the released incumbent, and the row offered
    FIRST hands it to the ungated stand-in at the same 3 launches / 1 seam. Pinning
    the order without pinning what it buys was the same shape of gap as a control
    asserted against an empty table.
    """
    def row_first():
        original = dict(launch.FUSED_PAIR_SEAMS)
        permuted = {"update_H": original["update_H"]}
        permuted.update((k, v) for k, v in original.items() if k != "update_H")
        launch.FUSED_PAIR_SEAMS.clear()
        launch.FUSED_PAIR_SEAMS.update(permuted)
        return original

    for row_class, served_when_first in (("both_incumbents_install", "step_B->update_H"),
                                         ("only_b_to_h_installs", "update_H->step_D")):
        build, sources, _rows, absorb = ROW_CLASSES[row_class]
        with _stand_in(admit=True, absorb=absorb):
            last = _composition(_plan_on(build, sources))
            original = row_first()
            try:
                first = _composition(_plan_on(build, sources))
            finally:
                launch.FUSED_PAIR_SEAMS.clear()
                launch.FUSED_PAIR_SEAMS.update(original)
        assert last["seams"]["step_B->update_H"], (row_class, last)
        assert first["launches"] == last["launches"], (row_class, first, last)
        assert first["seams"][served_when_first], (row_class, first)
        if row_class == "both_incumbents_install":
            assert first == last, (first, last)
        else:
            assert not first["seams"]["step_B->update_H"], first
            assert first["selected"]["update_H"] == STAND_IN_LABEL, first
    assert list(launch.FUSED_PAIR_SEAMS)[-1] == "update_H"
