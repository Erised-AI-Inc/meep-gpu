"""The withdraw hoist: what it admits, what it refuses, and the licence it may not widen.

``parity/meep_gpu/results/h_to_d_withdraw_order_2026-09-04/`` measured the ordering on
seven lifted corpus rows (420 complete driver steps, 7,364,789,280 uint32 words, 0
differing) with two null controls that had to diverge. This file is the package-level
test of the module that measurement became, and it is weighted the way the risk is: the
failure mode here is a SILENT one -- a product that hoists what it may not, or that
hoists part of the list, computes plausible fields nobody notices -- so every claim is
paired with the thing that must fail.

Three habits, all from this package's own conventions:

* THE IDEMPOTENCE IS DRIVEN, NOT ASSERTED. The hoist rests on ``withdraw`` returning
  early on a second call (``sources.py:2116-2121``); :func:`test_withdraw_is_idempotent`
  injects, withdraws twice and byte-compares, with the first withdraw required to have
  CHANGED D so the second's no-op is measured rather than trivially satisfied.
* EVERY REFUSAL IS EARNED. The two placements the module refuses are the campaign's two
  null controls, and the tests do not merely check that the refusal fires: they perform
  the refused thing on a real grid and require it to DIVERGE from the driver's order.
* THE LICENCE IS PINNED. :func:`test_the_licence_is_exactly_what_was_measured` fails if
  ``LICENSED_PLACEMENTS`` grows, and :func:`test_the_recorded_measurement_matches_the_artifact`
  compares every number in ``MEASUREMENT`` against the record's own ``summary.json``.
"""

from __future__ import annotations

import json
import pathlib

import numpy
import pytest

from . import stepping, withdraw_hoist
from .fields import Fields
from .grid import Grid
from .pml import PML
from .sources import ContinuousEnvelope, VolumeSource

RECORD = (pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu" /
          "results" / "h_to_d_withdraw_order_2026-09-04")

SEEDED = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz",
          "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
          "fu_Dx", "fu_Dy", "fu_Dz", "fu_Bx", "fu_By", "fu_Bz")


# ---------------------------------------------------------------------------
# The fixture: a small absorbing grid with one integrated electric point source
# ---------------------------------------------------------------------------


def _build(cell=(1.2, 1.2, 0.0), resolution=12.0, thickness=2, seed=20260904,
           integrated=True, component="Ez"):
    """Grid, fields, layer and ONE source whose withdraw does work.

    The layer is ACTIVE on purpose: with an inactive one ``stepping.update_H`` returns
    before its first statement (``stepping.py:944-945``) and the seam this module serves
    has no first half at all -- the boards' ``not_fusion_surface`` bucket, not this
    module's subject.

    The arrays are SEEDED WITH NOISE rather than zeroed, the ``test_deposit_repair``
    habit: zero is a fixed point of the constitutive recurrence, so a zeroed comparison
    passes whether or not the pass under test did anything.
    """
    dimensions = sum(1 for extent in cell if extent > 0.0)
    grid = Grid(resolution=resolution, cell_size=cell, dimensions=dimensions)
    fields = Fields(grid=grid)
    pml = PML(grid=grid, thickness=tuple(thickness if e > 0.0 else 0 for e in cell))
    fields.enable_pml_storage()
    generator = numpy.random.default_rng(seed)
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = generator.standard_normal(array.shape).astype(array.dtype)
    envelope = ContinuousEnvelope(frequency=1.0, is_integrated=integrated)
    source = VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0), envelope=envelope)
    return grid, fields, pml, source


def _words(fields):
    """Every stored volume as bytes. Words, not ``allclose``: ``-0.0 == 0.0`` lies."""
    out = {}
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        out[name] = numpy.ascontiguousarray(array).ravel().view(numpy.uint8).copy()
    return out


def _differing(a, b):
    return {name: int((a[name] != b[name]).sum())
            for name in a if int((a[name] != b[name]).sum())}


class _Recorder:
    """A stand-in for a fused product's launch, which records when it ran."""

    def __init__(self, log):
        self.log = log
        self.runs = 0

    def run(self, *_args, **_kwargs):
        self.runs += 1
        self.log.append("launch")


# ---------------------------------------------------------------------------
# The mechanism the licence rests on, DRIVEN
# ---------------------------------------------------------------------------


def test_withdraw_is_idempotent():
    """A second ``withdraw`` in the same step writes nothing -- driven, not asserted.

    This is what lets a product hoist the pass and leave ``driver.py:3313-3314`` in
    place: the driver's own loop runs after the hoisted call and must be free. The first
    withdraw is REQUIRED to change D, or the second one's no-op would be measured against
    nothing.
    """
    _grid, fields, _pml, source = _build()
    source.inject(fields, 0.0)
    assert source._applied_dipole != 0, "the injection did not arm a standing dipole"

    before = _words(fields)
    source.withdraw(fields)
    after_first = _words(fields)
    assert _differing(before, after_first), (
        "the first withdraw wrote nothing, so this case cannot tell an idempotent "
        "second call from a missing one")

    source.withdraw(fields)
    after_second = _words(fields)
    assert not _differing(after_first, after_second), _differing(after_first, after_second)
    assert source._applied_dipole == 0


def test_update_H_and_the_withdraw_touch_disjoint_arrays():
    """The mechanism the module's docstring claims, measured on this fixture.

    ``stepping.update_H`` writes ``H*``/``f_w_H*``; the electric withdraw writes the
    component's own D array at its deposit points. Neither reads what the other wrote,
    which is why the hoist has a chance of being free -- and the next test is what makes
    that a measurement rather than an argument.
    """
    _grid, fields, pml, source = _build()
    source.inject(fields, 0.0)

    before = _words(fields)
    stepping.update_H(fields, pml)
    moved_by_update_H = set(_differing(before, _words(fields)))
    assert moved_by_update_H, "update_H wrote nothing; the layer is not active"

    between = _words(fields)
    source.withdraw(fields)
    moved_by_withdraw = set(_differing(between, _words(fields)))
    assert moved_by_withdraw == {"Dz"}, moved_by_withdraw
    assert not (moved_by_update_H & moved_by_withdraw), (
        moved_by_update_H & moved_by_withdraw)


# ---------------------------------------------------------------------------
# The orderings, on a real grid: the subject and both null controls
# ---------------------------------------------------------------------------


#: The four orderings, as the campaign spells them. ``in_seam`` is whether the driver's
#: own loop (``driver.py:3313-3314``) runs; ``before`` is what the product's leading slot
#: withdraws ahead of ``update_H``; ``after`` is the refused placement.
#:
#: ``hoisted`` keeps the in-seam loop, which the campaign's `hoisted` mode replaced by a
#: suppression -- and that is DELIBERATELY STRONGER here: the shipped product leaves the
#: driver untouched, so the loop really does run a second time, and this mode is only
#: byte-identical if it is free. The two null controls SUPPRESS the loop, because a
#: dropped or displaced withdraw that the driver then performs anyway is no control at
#: all.
_ORDERINGS = {
    "driver":             {"in_seam": True,  "before": "none", "after": "none"},
    "hoisted":            {"in_seam": True,  "before": "all",  "after": "none"},
    "hoisted_drop_first": {"in_seam": False, "before": "rest", "after": "none"},
    "after_step_D":       {"in_seam": False, "before": "none", "after": "all"},
}


def _run(mode, steps=6, **build):
    """Complete steps under one ordering, returning the final bytes.

    Mirrors ``FdtdDriver.step`` (``driver.py:3289-3332``) for a run with no
    susceptibility, so ``update_P`` is the no-op it would be there.
    """
    ordering = _ORDERINGS[mode]
    _grid, fields, pml, source = _build(**build)
    sources = [source]
    standing = [s for _index, s in withdraw_hoist.standing_withdraws(sources)]
    before = {"none": [], "all": standing, "rest": standing[1:]}[ordering["before"]]
    after = {"none": [], "all": standing}[ordering["after"]]
    dt = fields.grid.dt
    time = 0.0
    for _index in range(steps):
        stepping.step_B(fields, pml)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        # BY HAND rather than through `hoist`, because `hoist` refuses a partial list by
        # name -- which is the other half of this file's claim
        # (test_a_dropped_withdraw_is_refused_by_name). The subject mode is driven
        # through the module itself in test_the_leading_plan_hoists_before_the_launch.
        for candidate in before:
            candidate.withdraw(fields)
        stepping.update_H(fields, pml)
        if ordering["in_seam"]:  # driver.py:3313-3314
            for candidate in sources:
                candidate.withdraw(fields)
        stepping.step_D(fields, pml)
        for candidate in after:
            candidate.withdraw(fields)
        for candidate in sources:
            candidate.inject(fields, time + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        stepping.update_E(fields, pml)
        time += dt
    return _words(fields)


def test_the_hoisted_order_is_byte_identical_and_both_null_controls_diverge():
    """The campaign's claim, reproduced at package scale on one small absorbing grid.

    ``hoisted`` must be byte-identical to the driver's order and BOTH refused orderings
    must diverge. A case where the controls agreed would be a case that cannot tell a
    correct hoist from a wrong one.
    """
    reference = _run("driver")
    assert _differing(reference, _run("driver")) == {}, "the fixture is not deterministic"

    hoisted = _differing(reference, _run("hoisted"))
    assert hoisted == {}, hoisted

    after = _differing(reference, _run("after_step_D"))
    assert after, ("moving the withdraw after step_D did not change a single word; the "
                   "comparison is not sensitive to where in the seam it runs")

    dropped = _differing(reference, _run("hoisted_drop_first"))
    assert dropped, ("dropping the only working withdraw did not change a single word; "
                     "the words it writes are outside the compared set")


def test_a_run_with_no_standing_withdraw_is_unaffected_by_the_hoist():
    """A non-integrated source has nothing standing, and the hoist is then a no-op.

    Establishes that this module's subject is the INTEGRATED path: with
    ``is_integrated=False`` the driver's loop is the no-op on every electric source, and
    a product spanning the seam needs no hoist at all.
    """
    _grid, fields, _pml, source = _build(integrated=False)
    assert withdraw_hoist.standing_withdraws([source]) == ()
    assert _differing(_run("driver", integrated=False),
                      _run("hoisted", integrated=False)) == {}


# ---------------------------------------------------------------------------
# Which withdraws stand: the configuration test, and what it must not become
# ---------------------------------------------------------------------------


def test_standing_withdraws_reports_the_callers_own_index():
    """"source 1" must name the second source the RUN declared, not the second match."""
    _grid, _fields, _pml, electric = _build()
    magnetic = VolumeSource(grid=electric.grid, component="Hy", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0,
                                                        is_integrated=True))
    standing = withdraw_hoist.standing_withdraws([magnetic, electric])
    assert [index for index, _s in standing] == [1], standing
    assert standing[0][1] is electric


def test_a_magnetic_source_is_not_this_seams_business():
    """The magnetic withdraw is hoisted over ``step_B``, one seam earlier.

    It must be neither hoisted here nor refused here: a product spanning
    ``update_H``/``step_D`` never swallows it, so its presence bars nothing.
    """
    _grid, fields, _pml, electric = _build()
    magnetic = VolumeSource(grid=electric.grid, component="Hy", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0,
                                                        is_integrated=True))
    assert withdraw_hoist.standing_withdraws([magnetic]) == ()
    ok, reasons = withdraw_hoist.hoistable(fields, [magnetic, electric])
    assert ok, reasons


def test_the_configuration_test_is_not_the_per_step_gate():
    """``_applied_dipole`` is zero before the first injection, and must not be consulted.

    A predicate that read it at plan-build time would answer "nothing stands here" for a
    row whose withdraw does work on every step from the second on -- and would admit that
    row into a product carrying no hoist.
    """
    _grid, fields, _pml, source = _build()
    assert source._applied_dipole == 0
    assert len(withdraw_hoist.standing_withdraws([source])) == 1
    source.inject(fields, 0.0)
    assert source._applied_dipole != 0
    assert len(withdraw_hoist.standing_withdraws([source])) == 1


def test_a_source_owning_no_deposit_point_has_nothing_to_withdraw():
    """``withdraw`` returns at ``sources.py:2114-2115`` with no points, so none stands."""
    _grid, _fields, _pml, source = _build()
    source._n_source_points = 0
    assert withdraw_hoist.standing_withdraws([source]) == ()


# ---------------------------------------------------------------------------
# The refusals, each paired with the thing it refuses
# ---------------------------------------------------------------------------


def test_the_after_step_D_placement_is_refused_by_name():
    """The campaign's ``after_step_D`` null control, refused AND shown to diverge."""
    _grid, fields, _pml, source = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [source],
                                           placement=withdraw_hoist.AFTER_STEP_D)
    assert not ok
    assert any("REFUSED PLACEMENT after_step_D" in reason for reason in reasons), reasons
    assert any("diverged at step 2 on all 7 rows" in reason for reason in reasons), reasons
    # The refusal is EARNED: performing it really does change the answer.
    assert _differing(_run("driver"), _run("after_step_D"))


def test_the_in_launch_placement_is_refused_as_unmeasured():
    """Carrying the withdraw inside the launch is refused for want of a measurement."""
    ok, reasons = withdraw_hoist.hoistable(None, [],
                                           placement=withdraw_hoist.IN_LAUNCH)
    assert not ok
    assert any("REFUSED PLACEMENT in_launch" in reason for reason in reasons), reasons
    assert any("never been measured" in reason for reason in reasons), reasons


def test_an_unknown_placement_is_a_declaration_error():
    """A name outside :data:`PLACEMENTS` is refused, never resolved to the licensed one."""
    ok, reasons = withdraw_hoist.hoistable(None, [], placement="whenever")
    assert not ok
    assert any("UNKNOWN PLACEMENT 'whenever'" in reason for reason in reasons), reasons


def test_a_dropped_withdraw_is_refused_by_name():
    """The campaign's ``hoisted_drop_first`` null control, refused AND shown to diverge.

    A hoist list missing a standing withdraw is the shape that diverged at step 2 on all
    7 rows, so it is refused whatever else is in order.
    """
    _grid, fields, _pml, source = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [source], hoisted=[])
    assert not ok
    assert any("DROPPED WITHDRAW" in reason for reason in reasons), reasons
    assert any("hoisted_drop_first" in reason for reason in reasons), reasons
    # Earned the same way: the dropped pass really does change the answer.
    assert _differing(_run("driver"), _run("hoisted_drop_first"))


def test_a_magnetic_source_in_the_hoist_list_is_refused_by_name():
    """Declaring a magnetic source hoisted is a claim about a seam this never measured."""
    _grid, fields, _pml, electric = _build()
    magnetic = VolumeSource(grid=electric.grid, component="Hy", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0,
                                                        is_integrated=True))
    ok, reasons = withdraw_hoist.hoistable(fields, [magnetic, electric],
                                           hoisted=[magnetic, electric])
    assert not ok
    assert any("NON-ELECTRIC SOURCE" in reason for reason in reasons), reasons
    assert any("driver.py:3289-3290" in reason for reason in reasons), reasons


def test_a_source_the_run_does_not_declare_is_refused():
    """Hoisting a source the driver's own loop will never re-run is an unmeasured order."""
    _grid, fields, _pml, source = _build()
    other = VolumeSource(grid=source.grid, component="Ex", center=(0.0, 0.0, 0.0),
                         size=(0.0, 0.0, 0.0),
                         envelope=ContinuousEnvelope(frequency=1.0,
                                                     is_integrated=True))
    ok, reasons = withdraw_hoist.hoistable(fields, [source], hoisted=[source, other])
    assert not ok
    assert any("UNDECLARED SOURCE" in reason for reason in reasons), reasons


def test_a_foreign_withdraw_body_is_refused_by_name():
    """The idempotence belongs to ``VolumeSource.withdraw``, not to the attribute name."""

    class _OwnWithdraw:
        field_type = "D"
        is_integrated = True
        component = "Ez"
        _n_source_points = 1
        _applied_dipole = 0j

        def withdraw(self, _fields):  # a body nothing has ever driven twice
            return None

    _grid, fields, _pml, _source = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [_OwnWithdraw()])
    assert not ok
    assert any("FOREIGN WITHDRAW" in reason for reason in reasons), reasons
    assert any("sources.py:2116-2121" in reason for reason in reasons), reasons


def test_a_source_publishing_no_standing_offset_is_refused_by_name():
    """Without ``_applied_dipole`` the second call's early return cannot be established."""

    class _NoOffset:
        field_type = "D"
        is_integrated = True
        component = "Ez"
        _n_source_points = 1

        def withdraw(self, _fields):
            return None

    _grid, fields, _pml, _source = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [_NoOffset()])
    assert not ok
    assert any("NO STANDING OFFSET" in reason for reason in reasons), reasons


def test_an_unresolvable_component_is_refused_before_the_launch():
    """``withdraw`` resolves one array; a component these fields cannot resolve refuses."""
    _grid, fields, _pml, source = _build()
    source.component = "Eq"
    ok, reasons = withdraw_hoist.hoistable(fields, [source])
    assert not ok
    assert any("UNRESOLVED ARRAY" in reason for reason in reasons), reasons


def test_no_source_list_is_a_refusal_not_an_empty_answer():
    """Fail closed: a run that declares nothing has established nothing."""
    ok, reasons = withdraw_hoist.hoistable(None, None)
    assert not ok
    assert any("NO SOURCE LIST" in reason for reason in reasons), reasons


# ---------------------------------------------------------------------------
# The span: what a product must own for the hoist to be the measured one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("span,marker", [
    (("step_B", "update_H", "step_D", "update_E"), "SPAN WIDER THAN THE SEAM"),
    (("update_H",), "SPAN HOLDS ONE HALF"),
    (("step_D",), "SPAN HOLDS ONE HALF"),
    (("step_B", "update_H"), "SPAN HOLDS ONE HALF"),
    (("step_B", "update_E"), "SPAN OUTSIDE THE SEAM"),
])
def test_a_span_that_is_not_the_seam_is_refused_by_name(span, marker):
    """Both directions fail closed, and for different reasons -- see ``_span_reasons``."""
    _grid, fields, _pml, source = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [source], span=span)
    assert not ok
    assert any(marker in reason for reason in reasons), reasons


def test_the_seam_span_itself_is_admitted():
    _grid, fields, _pml, source = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [source],
                                           span=withdraw_hoist.SEAM_SPAN)
    assert ok, reasons


# ---------------------------------------------------------------------------
# The two-consult contract's shape: the flag, and the leading plan
# ---------------------------------------------------------------------------


def _refusal(index, source):
    return f"in-seam withdraw from source {index} ({type(source).__name__})"


def test_a_product_that_does_not_hoist_keeps_refusing_the_row():
    """The default is the answer the three boards file today under ``withdraw_seam``."""
    _grid, fields, _pml, source = _build()
    reasons = withdraw_hoist.seam_withdraw_reasons(
        fields, [source], undeclared="no source list", refusal=_refusal)
    assert reasons == ("in-seam withdraw from source 0 (VolumeSource)",)


def test_a_product_that_hoists_admits_the_row():
    _grid, fields, _pml, source = _build()
    assert withdraw_hoist.seam_withdraw_reasons(
        fields, [source], undeclared="no source list", refusal=_refusal,
        hoists_the_withdraw=True) == ()


def test_the_declared_hoist_does_not_survive_a_refused_placement():
    """The flag declares wiring; it does not license a placement the campaign refused."""
    _grid, fields, _pml, source = _build()
    reasons = withdraw_hoist.seam_withdraw_reasons(
        fields, [source], undeclared="no source list", refusal=_refusal,
        hoists_the_withdraw=True, placement=withdraw_hoist.AFTER_STEP_D)
    assert reasons and all(r.startswith("the withdraw hoist cannot carry this seam:")
                           for r in reasons), reasons


def test_an_undeclared_source_list_uses_the_sites_own_text():
    """The decision lives in this module; the prose stays where its gate pinned it."""
    assert withdraw_hoist.seam_withdraw_reasons(
        None, None, undeclared="THE SITE'S OWN TEXT", refusal=_refusal) == (
        "THE SITE'S OWN TEXT",)


def test_the_leading_plan_hoists_before_the_launch():
    _grid, fields, _pml, source = _build()
    source.inject(fields, 0.0)
    log = []
    plan = withdraw_hoist.LeadingWithdrawPlan(_Recorder(log), fields, [source])
    plan.run()
    assert plan.withdrawn == 1
    assert log == ["launch"]
    assert source._applied_dipole == 0, (
        "the plan ran the launch without withdrawing first")
    assert plan.inner.runs == 1


def test_the_leading_plan_refuses_before_it_writes():
    """A refused configuration must not reach the launch, and must not half-hoist."""
    _grid, fields, _pml, source = _build()
    source.inject(fields, 0.0)
    standing = source._applied_dipole
    log = []
    plan = withdraw_hoist.LeadingWithdrawPlan(_Recorder(log), fields, [source],
                                              placement=withdraw_hoist.AFTER_STEP_D)
    with pytest.raises(withdraw_hoist.WithdrawNotHoistable):
        plan.run()
    assert log == []
    assert source._applied_dipole == standing


def test_hoist_returns_the_count_and_refuses_a_bad_span():
    _grid, fields, _pml, source = _build()
    source.inject(fields, 0.0)
    with pytest.raises(withdraw_hoist.WithdrawNotHoistable):
        withdraw_hoist.hoist(fields, [source], span=("update_H",))
    assert withdraw_hoist.hoist(fields, [source]) == 1
    assert withdraw_hoist.hoist(fields, [source]) == 1  # idempotent, and still counted


# ---------------------------------------------------------------------------
# The licence itself
# ---------------------------------------------------------------------------


def test_the_licence_is_exactly_what_was_measured():
    """FAILS IF THE LICENCE IS WIDENED.

    One placement was measured and one is licensed. Adding a second without a campaign
    behind it fails here, which is the point: this constant is the whole difference
    between a product that may hoist and one that may not.
    """
    assert withdraw_hoist.LICENSED_PLACEMENTS == (withdraw_hoist.BEFORE_UPDATE_H,)
    assert set(withdraw_hoist.PLACEMENT_REFUSALS) == (
        set(withdraw_hoist.PLACEMENTS) - set(withdraw_hoist.LICENSED_PLACEMENTS))
    assert withdraw_hoist.MEASUREMENT["placement"] == withdraw_hoist.BEFORE_UPDATE_H
    for phrase in ("array path", "device kernel", "magnetic"):
        assert phrase in withdraw_hoist.WHAT_THIS_DOES_NOT_LICENSE


def test_the_recorded_measurement_matches_the_artifact():
    """Every number in ``MEASUREMENT`` against the record that produced it.

    Skipped rather than passed where the record is absent -- the package ships without
    ``parity/`` -- because a silent skip and a pass are different claims.
    """
    summary = RECORD / "summary.json"
    if not summary.is_file():
        pytest.skip(f"{RECORD} is not in this checkout")
    record = json.loads(summary.read_text(encoding="utf-8"))
    claim = withdraw_hoist.MEASUREMENT
    assert record["campaign"] == claim["campaign"]
    assert record["verdict"] == claim["verdict"]
    assert record["rows"] == claim["rows"]
    assert record["rows_order_equivalent"] == claim["rows_order_equivalent"]
    assert record["steps_per_leg"] == claim["steps_per_leg"]
    assert tuple(record["must_be_identical"]) == claim["must_be_identical"]
    assert tuple(record["must_diverge"]) == claim["must_diverge"]
    per_row = record["per_row"]
    assert len(per_row) == claim["rows"]
    assert sum(row["words_compared"] for row in per_row.values()) == claim["words_compared"]
    assert sum(row["steps_compared"]
               for row in per_row.values()) == claim["complete_driver_steps"]
    assert sum(row["hoisted_differing_words"]
               for row in per_row.values()) == claim["differing_words"]
    assert all(row["restore_control_differing_words"] == 0 for row in per_row.values())
    for row in per_row.values():
        for control in claim["must_diverge"]:
            assert row["null_controls"][control]["first_divergence"] == (
                claim["null_controls_first_divergence_step"])
