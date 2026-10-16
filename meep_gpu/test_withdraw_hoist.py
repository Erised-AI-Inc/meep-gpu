"""The ``update_H -> step_D`` seam's source pass: no withdraw stands in it, and the one
restore that does commutes with ``step_D``.

An integrated source enters at the constitutive update through ``f_minus_p``
(``sources.f_minus_p_begin`` / ``f_minus_p_end``), so no source carries a ``withdraw`` and
nothing is returned to D between the seam's two consults. What does stand there, in a run
whose H is stored, is the magnetic ``f_minus_p`` restore, which writes B alone. This file
drives the two facts a fused ``update_H + step_D`` launch rests on, the failure mode being
SILENT (a launch reading the wrong words computes a plausible field nobody notices), so
each claim is paired with the thing that must fail:

* THE RESTORE WRITES ONLY B, measured on a grid rather than asserted;
* THE FUSED ORDER IS THE DRIVER'S, byte for byte over 30 complete steps, and the two
  orders the commutation does not license -- the restore before ``update_H``, and no
  ``f_minus_p`` at all -- diverge.

The module's predicates still fail closed on a source class that carries a ``withdraw``
(none in this package): the body the 2026-09-04 licence measured no longer exists, so no
such withdraw may be hoisted. Those refusals are kept below with stand-in sources, and the
recorded 2026-09-04 measurement is still compared against its artifact.
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
from .sources import (ContinuousEnvelope, ContinuousSource, ExtendedSource, FIELD_TYPE_B,
                      FIELD_TYPE_D, GaussianEnvelope, GaussianPulsedSource, VolumeSource,
                      f_minus_p_begin, f_minus_p_end)

RECORD = (pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu" /
          "results" / "h_to_d_withdraw_order_2026-09-04")

SEEDED = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz",
          "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
          "fu_Dx", "fu_Dy", "fu_Dz", "fu_Bx", "fu_By", "fu_Bz")


# ---------------------------------------------------------------------------
# The fixture: a small absorbing grid with an integrated electric and an integrated
# magnetic point source
# ---------------------------------------------------------------------------


def _build(cell=(1.2, 1.2, 0.0), resolution=12.0, thickness=2, seed=20261007):
    """Grid, fields, an active layer and two integrated sources (Ez, Hz).

    The layer is ACTIVE on purpose: H is then stored, which is the case where the magnetic
    restore stands between the seam's two consults. The arrays are SEEDED WITH NOISE (the
    ``test_deposit_repair`` habit: zero is a fixed point of the constitutive recurrence),
    and the state is mid-run (``h_separated``), so no first-step start is in play.
    """
    dimensions = sum(1 for extent in cell if extent > 0.0)
    grid = Grid(resolution=resolution, cell_size=cell, dimensions=dimensions)
    fields = Fields(grid=grid)
    pml = PML(grid=grid, thickness=tuple(thickness if e > 0.0 else 0 for e in cell))
    fields.enable_pml_storage()
    fields.h_separated = True
    generator = numpy.random.default_rng(seed)
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = generator.standard_normal(array.shape).astype(array.dtype)
    electric = VolumeSource(grid=grid, component="Ez", center=(0.05, -0.08, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0, is_integrated=True))
    magnetic = VolumeSource(grid=grid, component="Hz", center=(-0.11, 0.07, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=GaussianEnvelope(frequency=1.1, fwidth=0.8, start_time=-2.0,
                                                      is_integrated=True))
    return grid, fields, pml, [electric, magnetic]


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


class _StandingWithdraw:
    """A source class carrying a ``withdraw`` of its own -- the shape the predicates were
    built for, and which no engine source has any more."""

    field_type = "D"
    is_integrated = True
    component = "Ez"
    _n_source_points = 1
    _applied_dipole = 0j

    def withdraw(self, _fields):
        return None


class _MagneticStandingWithdraw(_StandingWithdraw):
    field_type = "B"
    component = "Hz"


def _refusal(index, source):
    return f"in-seam withdraw from source {index} ({type(source).__name__})"


# ---------------------------------------------------------------------------
# No withdraw stands in the seam
# ---------------------------------------------------------------------------


def test_no_engine_source_carries_a_withdraw():
    """The integrated path enters through ``f_minus_p``; nothing stands in D or B to withdraw."""
    for kind in (VolumeSource, ContinuousSource, ExtendedSource, GaussianPulsedSource):
        assert not hasattr(kind, "withdraw"), kind
        assert not hasattr(kind, "_applied_dipole"), kind
    _grid, fields, _pml, sources = _build()
    assert all(source.is_integrated for source in sources)
    assert withdraw_hoist.standing_withdraws(sources) == ()
    assert withdraw_hoist.seam_withdraw_reasons(
        fields, sources, undeclared="no source list", refusal=_refusal) == ()
    ok, reasons = withdraw_hoist.hoistable(fields, sources)
    assert ok, reasons
    assert withdraw_hoist.hoist(fields, sources) == 0


def test_an_integrated_inject_deposits_nothing():
    """``step_source`` skips an integrated source (step.cpp:300): its ``inject`` writes no word."""
    _grid, fields, _pml, sources = _build()
    before = _words(fields)
    for source in sources:
        source.inject(fields, 0.37)
    assert _differing(before, _words(fields)) == {}


def test_the_magnetic_restore_writes_B_alone():
    """The one pass between the seam's consults, measured: it moves B and nothing else.

    The replace is REQUIRED to change B first, so the restore's footprint is measured
    against a change rather than against nothing.
    """
    _grid, fields, pml, sources = _build()
    # The driver's own boundary passes run before the bracket (FdtdDriver.step).
    stepping.fill_symmetry_bc_B(fields)
    stepping.zero_metal_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    before = _words(fields)
    held = f_minus_p_begin(fields, sources, FIELD_TYPE_B, 0.5 * fields.grid.dt)
    assert held is not None
    replaced = set(_differing(before, _words(fields)))
    assert replaced == {"Bz"}, replaced
    stepping.update_H(fields, pml)
    after_update = _words(fields)
    f_minus_p_end(fields, held)
    restored = set(_differing(after_update, _words(fields)))
    assert restored == {"Bz"}, restored
    assert set(withdraw_hoist.SEAM_PASS["writes"]) >= restored
    assert withdraw_hoist.SEAM_PASS["step_D_reads_none_of_it"] is True


# ---------------------------------------------------------------------------
# The orderings, on a real grid: the subject and both null controls
# ---------------------------------------------------------------------------


#: Where the magnetic restore runs, relative to the seam's two halves.
_ORDERINGS = ("driver", "fused", "restore_before_update_H", "no_magnetic_f_minus_p")


def _run(mode, steps=30, **build):
    """Complete steps under one ordering, returning the final bytes.

    Mirrors ``FdtdDriver.step`` for a run with no susceptibility and only integrated
    sources (whose ``inject`` deposits nothing), so ``update_P`` is the no-op it would be
    there. ``fused`` is what a launch spanning ``update_H`` and ``step_D`` leaves: the
    restore after both halves.
    """
    assert mode in _ORDERINGS
    _grid, fields, pml, sources = _build(**build)
    dt = fields.grid.dt
    time = 0.0
    for _index in range(steps):
        stepping.step_B(fields, pml)
        for source in sources:
            source.inject(fields, time)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        held = None
        if mode != "no_magnetic_f_minus_p":
            held = f_minus_p_begin(fields, sources, FIELD_TYPE_B, time + 0.5 * dt)
        if mode == "restore_before_update_H":
            f_minus_p_end(fields, held)
            held = None
        stepping.update_H(fields, pml)
        if mode == "driver":
            f_minus_p_end(fields, held)
            held = None
        stepping.step_D(fields, pml)
        f_minus_p_end(fields, held)
        for source in sources:
            source.inject(fields, time + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        held_d = f_minus_p_begin(fields, sources, FIELD_TYPE_D, time + dt)
        stepping.update_E(fields, pml)
        f_minus_p_end(fields, held_d)
        time += dt
    return _words(fields)


def test_the_fused_order_is_byte_identical_and_both_null_controls_diverge():
    """A launch spanning the seam may take the restore past ``step_D``; nothing else may move.

    ``fused`` must be byte-identical to the driver's order and BOTH refused orderings must
    diverge. A case where the controls agreed would be a case that cannot tell a correct
    fusion from a wrong one.
    """
    reference = _run("driver")
    assert _differing(reference, _run("driver")) == {}, "the fixture is not deterministic"

    fused = _differing(reference, _run("fused"))
    assert fused == {}, fused

    early = _differing(reference, _run("restore_before_update_H"))
    assert early, ("restoring B before update_H changed no word; the comparison cannot see "
                   "which B the constitutive update read")

    none = _differing(reference, _run("no_magnetic_f_minus_p"))
    assert none, ("dropping the magnetic f_minus_p changed no word; the words it writes "
                  "are outside the compared set")


# ---------------------------------------------------------------------------
# The refusals the predicates keep, each on a source that carries a withdraw
# ---------------------------------------------------------------------------


def test_the_after_step_D_placement_is_refused_by_name():
    _grid, fields, _pml, sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, sources, placement=withdraw_hoist.AFTER_STEP_D)
    assert not ok
    assert any("REFUSED PLACEMENT after_step_D" in reason for reason in reasons), reasons


def test_the_in_launch_placement_is_refused_as_unmeasured():
    _grid, fields, _pml, sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, sources, placement=withdraw_hoist.IN_LAUNCH)
    assert not ok
    assert any("never been measured" in reason for reason in reasons), reasons


def test_an_unknown_placement_is_a_declaration_error():
    _grid, fields, _pml, sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, sources, placement="somewhere")
    assert not ok
    assert any("UNKNOWN PLACEMENT" in reason for reason in reasons), reasons


def test_a_withdraw_body_of_its_own_is_refused_by_name():
    """The measured body is gone, so any ``withdraw`` a source class brings is foreign."""
    _grid, fields, _pml, _sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [_StandingWithdraw()])
    assert not ok
    assert any("FOREIGN WITHDRAW" in reason for reason in reasons), reasons
    assert any("no longer exists" in reason for reason in reasons), reasons


def test_a_source_publishing_no_standing_offset_is_refused_by_name():
    class _NoOffset:
        field_type = "D"
        is_integrated = True
        component = "Ez"
        _n_source_points = 1

        def withdraw(self, _fields):
            return None

    _grid, fields, _pml, _sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [_NoOffset()])
    assert not ok
    assert any("NO STANDING OFFSET" in reason for reason in reasons), reasons


def test_a_dropped_withdraw_is_refused_by_name():
    _grid, fields, _pml, _sources = _build()
    standing = _StandingWithdraw()
    ok, reasons = withdraw_hoist.hoistable(fields, [standing], hoisted=[])
    assert not ok
    assert any("DROPPED WITHDRAW" in reason for reason in reasons), reasons


def test_a_magnetic_source_in_the_hoist_list_is_refused_by_name():
    _grid, fields, _pml, _sources = _build()
    magnetic = _MagneticStandingWithdraw()
    assert withdraw_hoist.standing_withdraws([magnetic]) == ()
    ok, reasons = withdraw_hoist.hoistable(fields, [magnetic], hoisted=[magnetic])
    assert not ok
    assert any("NON-ELECTRIC SOURCE" in reason for reason in reasons), reasons


def test_a_source_the_run_does_not_declare_is_refused():
    _grid, fields, _pml, _sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [], hoisted=[_StandingWithdraw()])
    assert not ok
    assert any("UNDECLARED SOURCE" in reason for reason in reasons), reasons


def test_an_unresolvable_component_is_refused_before_the_launch():
    class _Unresolvable(_StandingWithdraw):
        component = "Eq"

    _grid, fields, _pml, _sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, [_Unresolvable()])
    assert not ok
    assert any("UNRESOLVED ARRAY" in reason for reason in reasons), reasons


def test_no_source_list_is_a_refusal_not_an_empty_answer():
    """Fail closed: a run that declares nothing has established nothing."""
    ok, reasons = withdraw_hoist.hoistable(None, None)
    assert not ok
    assert any("NO SOURCE LIST" in reason for reason in reasons), reasons


@pytest.mark.parametrize("span,marker", [
    (("step_B", "update_H", "step_D", "update_E"), "SPAN WIDER THAN THE SEAM"),
    (("update_H",), "SPAN HOLDS ONE HALF"),
    (("step_D",), "SPAN HOLDS ONE HALF"),
    (("step_B", "update_H"), "SPAN HOLDS ONE HALF"),
    (("step_B", "update_E"), "SPAN OUTSIDE THE SEAM"),
])
def test_a_span_that_is_not_the_seam_is_refused_by_name(span, marker):
    """Both directions fail closed, and for different reasons -- see ``_span_reasons``."""
    _grid, fields, _pml, sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, sources, span=span)
    assert not ok
    assert any(marker in reason for reason in reasons), reasons


def test_the_seam_span_itself_is_admitted():
    _grid, fields, _pml, sources = _build()
    ok, reasons = withdraw_hoist.hoistable(fields, sources, span=withdraw_hoist.SEAM_SPAN)
    assert ok, reasons


# ---------------------------------------------------------------------------
# The two-consult contract's shape: the flag, and the leading plan
# ---------------------------------------------------------------------------


def test_a_product_that_does_not_hoist_keeps_refusing_a_standing_withdraw():
    _grid, fields, _pml, _sources = _build()
    reasons = withdraw_hoist.seam_withdraw_reasons(
        fields, [_StandingWithdraw()], undeclared="no source list", refusal=_refusal)
    assert reasons == ("in-seam withdraw from source 0 (_StandingWithdraw)",)


def test_a_product_that_hoists_cannot_carry_a_withdraw_body_of_its_own():
    """The declared hoist was licensed for the measured body; there is none to hoist now."""
    _grid, fields, _pml, _sources = _build()
    reasons = withdraw_hoist.seam_withdraw_reasons(
        fields, [_StandingWithdraw()], undeclared="no source list", refusal=_refusal,
        hoists_the_withdraw=True)
    assert reasons and all(r.startswith("the withdraw hoist cannot carry this seam:")
                           for r in reasons), reasons


def test_an_undeclared_source_list_uses_the_sites_own_text():
    """The decision lives in this module; the prose stays where its gate pinned it."""
    assert withdraw_hoist.seam_withdraw_reasons(
        None, None, undeclared="THE SITE'S OWN TEXT", refusal=_refusal) == (
        "THE SITE'S OWN TEXT",)


def test_the_leading_plan_runs_the_launch_and_withdraws_nothing():
    _grid, fields, _pml, sources = _build()
    before = _words(fields)
    log = []
    plan = withdraw_hoist.LeadingWithdrawPlan(_Recorder(log), fields, sources)
    plan.run()
    assert plan.withdrawn == 0
    assert log == ["launch"]
    assert plan.inner.runs == 1
    assert _differing(before, _words(fields)) == {}


def test_the_leading_plan_refuses_before_it_launches():
    """A refused configuration must not reach the launch."""
    _grid, fields, _pml, sources = _build()
    log = []
    plan = withdraw_hoist.LeadingWithdrawPlan(_Recorder(log), fields, sources,
                                              placement=withdraw_hoist.AFTER_STEP_D)
    with pytest.raises(withdraw_hoist.WithdrawNotHoistable):
        plan.run()
    assert log == []


def test_hoist_returns_the_count_and_refuses_a_bad_span():
    _grid, fields, _pml, sources = _build()
    with pytest.raises(withdraw_hoist.WithdrawNotHoistable):
        withdraw_hoist.hoist(fields, sources, span=("update_H",))
    assert withdraw_hoist.hoist(fields, sources) == 0


# ---------------------------------------------------------------------------
# The 2026-09-04 licence, kept as its record
# ---------------------------------------------------------------------------


def test_the_licence_is_exactly_what_was_measured_and_is_superseded():
    """FAILS IF THE LICENCE IS WIDENED, and it names what superseded it."""
    assert withdraw_hoist.LICENSED_PLACEMENTS == (withdraw_hoist.BEFORE_UPDATE_H,)
    assert set(withdraw_hoist.PLACEMENT_REFUSALS) == (
        set(withdraw_hoist.PLACEMENTS) - set(withdraw_hoist.LICENSED_PLACEMENTS))
    assert withdraw_hoist.MEASUREMENT["placement"] == withdraw_hoist.BEFORE_UPDATE_H
    assert "f_minus_p" in withdraw_hoist.MEASUREMENT["superseded_by"]
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
