"""What the driver-route gate's SECOND consult site may and may not blame itself for.

``second_consult_site`` drives ``synchronize_magnetic_fields`` on three legs and
compares what comes out. Every comparison it makes is between states taken AFTER a
leg's own half-step, so until the ``state_entering_the_site`` measurement landed
(2026-09-11) a divergence the legs carried INTO the site was indistinguishable, in
the row, from one the site produced — and the row's own name points at the
half-step.

THE ROW WAS DIAGNOSABLE, but only by arithmetic a reader had to do. With every
``restore_returns_the_state`` True, each leg is byte-equal to its own pre-site
state, so ``restored_fused_vs_array`` is exactly ``before[fused] == before[array]``:
a row reporting all-True restores beside a False restored comparison is reporting a
divergence that PREDATES the half-step, whatever ``synchronized_state`` says. A
2026-09-11 report of a kernel-vs-array divergence at this site on
``folded_dispersive_2d`` carried that exact combination, and the measured seam was
clean — every sub-step of the half-step byte-identical between the dispatched and
array legs on two resolutions and both folded cases.

So these tests pin the classification rather than the physics: the site reports
what it produced, and says so when it was handed something it did not.

THE OTHER HALF OF THIS FILE IS THE SAME RULING FROM THE OTHER END (2026-09-14): a
case whose builder returns NO MONITOR has neither a flux comparison nor a second
consult site, and both must be recorded NOT-APPLICABLE rather than scored as
passes. ``e2e.compare_spectra`` over two empty mappings answers ``identical True``,
and the site can be driven by hand on any driver, so both clauses were producing
green words for measurements the configuration never takes. The Metal route gate
has kept exactly this record for ``cylindrical`` since 2026-09-12.

IT IS DERIVED FROM THE BUILDER, not declared per case: ``run_leg`` carries the
monitor list the case RETURNED, so a builder that starts returning one is scored
again with no edit to the gate, and there is no key a future round can set to quiet
an inconvenient clause. The tests below drive both directions — the decline, and a
case that declares a monitor still FAILING on the same legs.

MEASURED CONSEQUENCE on the 2026-09-14 CUDA shipped leg: ``material_dispersion_0d``
(cell ``(1, 1, 1)``) read SECOND-CONSULT-HARNESS-FAILURE because every curl on a
one-cell grid is zero, so the magnetic half-step genuinely changes nothing and the
site's own non-vacuity clause fired on a true property of the cell.

Host-only. The stub legs below carry no device and no plan; what is under test is
the gate helper's bookkeeping, which is the half that was wrong.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict

import numpy
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gate_dispatch_end_to_end as e2e  # noqa: E402
import gate_dispatch_fused_route as route_gate  # noqa: E402


class _Fields:
    """The smallest object ``gate_dispatch_end_to_end.collect_state`` will walk."""

    def __init__(self, seed: float) -> None:
        self.Bx = numpy.full((4, 3), seed, dtype=numpy.float32)
        self.Hx = numpy.full((4, 3), 2.0 * seed, dtype=numpy.float32)


class _Driver:
    """A driver whose half-step is a known, reversible edit of its own arrays.

    ``synchronize_magnetic_fields`` here does what the real one does in the only
    respect these tests are about: it changes the state, and ``restore`` puts back
    exactly what was there. Nothing is dispatched, so the site's launch counters
    read zero on every leg and the clauses under test are the state ones.
    """

    def __init__(self, seed: float, bump: float = 1.0) -> None:
        self.fields = _Fields(seed)
        self.pml = None
        self._bump = bump
        self._backup: Dict[str, Any] = {}
        self.syncs = 0

    def synchronize_magnetic_fields(self) -> None:
        self.syncs += 1
        self._backup = {name: getattr(self.fields, name).copy()
                        for name in ("Bx", "Hx")}
        for name in ("Bx", "Hx"):
            getattr(self.fields, name)[...] += numpy.float32(self._bump)

    def restore_magnetic_fields(self) -> None:
        for name, saved in self._backup.items():
            getattr(self.fields, name)[...] = saved
        self._backup = {}


class _Counter:
    """The launch counter's two methods, answering "nothing launched"."""

    @staticmethod
    def snapshot() -> Dict[str, Any]:
        return {"total": 0, "hook_calls": 0, "zero_grid_launches": 0,
                "by_kernel": {}}


def _legs(seeds: Dict[str, float], bump: float = 1.0) -> Dict[str, Dict[str, Any]]:
    return {label: {"label": label, "env": {}, "driver": _Driver(seed, bump)}
            for label, seed in seeds.items()}


def _run(legs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return route_gate.second_consult_site("stub_case", legs, _Counter())


def test_a_snapshot_survives_a_later_write_to_the_array_it_was_taken_from():
    """``collect_state`` must COPY, and on a host-backed driver it did not.

    ``backends.to_numpy`` copies a device array and passes a NumPy array through, and
    ``ascontiguousarray`` returns a contiguous array unchanged — so every snapshot a
    NumPy lift took was a live view of the fields. Measured consequence in the
    2026-09-11 ``--smoke`` rows: ``sync_changed_the_state`` read False on all three
    legs of all three cases, because ``before`` and ``after`` were the same object.

    Device legs were never affected, which is why it survived: the bug spares exactly
    the legs that carry the claim.
    """
    driver = _Driver(1.0)
    before = e2e.collect_state(driver)
    driver.fields.Bx[...] += numpy.float32(5.0)
    after = e2e.collect_state(driver)
    assert not e2e.compare_state(before, after)["identical"], (
        "the snapshot moved with the array it was taken from, so every before/after "
        "comparison in this harness is between a state and itself")


def test_three_identical_legs_pass_and_record_that_they_entered_identical():
    """The arming control: the clauses can say PASS, and say why they could."""
    out = _run(_legs({"fused": 1.0, "unfused": 1.0, "array": 1.0}))
    assert out["the_legs_entered_the_site_identical"] is True
    assert out["state_entering_the_site"] == {"fused_vs_array": True,
                                              "unfused_vs_array": True}
    assert out["synchronized_state"]["fused_vs_array"] is True
    assert out["synchronized_state"]["unfused_vs_array"] is True
    # Non-vacuity: a half-step that changed nothing would pass every clause above.
    assert all(out["synchronized_state"]["sync_changed_the_state"].values())
    assert all(out["restore_returns_the_state"].values())
    assert out["restored_fused_vs_array"] is True
    assert out["verdict"] == "PASS"


def test_a_divergence_the_legs_carried_IN_is_a_harness_failure_not_a_site_failure():
    """The row this file exists for, and the combination that gives it away.

    The array leg is lifted to a different state than the other two — everything
    else about the protocol is unchanged. The half-step then does exactly what it
    does on the passing legs, and the restore is exact on every one of them, so the
    three fields that used to carry the whole story read: synchronized states
    differ, restores all True, restored comparison False. That is the fingerprint of
    an inherited divergence, and the verdict must name it rather than blame
    ``step_B``, the fills or the constitutive.
    """
    out = _run(_legs({"fused": 1.0, "unfused": 1.0, "array": 2.0}))
    assert out["the_legs_entered_the_site_identical"] is False
    assert out["state_entering_the_site"]["fused_vs_array"] is False
    assert out["state_entering_the_site"]["unfused_vs_array"] is False
    # The fingerprint, measured rather than asserted from the docstring.
    assert out["synchronized_state"]["fused_vs_array"] is False
    assert out["synchronized_state"]["unfused_vs_array"] is False
    assert all(out["restore_returns_the_state"].values())
    assert out["restored_fused_vs_array"] is False
    assert out["verdict"] == "HARNESS-FAILURE"
    assert "entering the half-step" in out["why"]


def test_a_divergence_the_HALF_STEP_produced_is_still_a_site_failure():
    """The discrimination, in the other direction: the guard must not launder.

    Same entry state on all three legs and a half-step that treats one of them
    differently — which is what a kernel disagreeing with the array path at one of
    the half-step's sub-steps looks like from here. The entry clause passes, so the
    verdict falls through to the state clauses and FAILS, which is the answer the
    site is for.
    """
    legs = _legs({"fused": 1.0, "unfused": 1.0, "array": 1.0})
    legs["fused"]["driver"] = _Driver(1.0, bump=3.0)
    out = _run(legs)
    assert out["the_legs_entered_the_site_identical"] is True
    assert out["synchronized_state"]["fused_vs_array"] is False
    # The restore is exact on every leg, so the states come back equal: the site
    # says FAIL on what the half-step DID, not on what it left behind.
    assert all(out["restore_returns_the_state"].values())
    assert out["restored_fused_vs_array"] is True
    assert out["verdict"] == "FAIL"


def _legs_declaring(monitors_per_leg: Dict[str, Any],
                    seeds: Dict[str, float] | None = None,
                    bump: float = 1.0) -> Dict[str, Dict[str, Any]]:
    """Stub legs that DECLARE a monitor list, the way ``run_leg`` does."""
    legs = _legs(seeds or {"fused": 1.0, "unfused": 1.0, "array": 1.0}, bump)
    for label, monitors in monitors_per_leg.items():
        legs[label]["monitors"] = monitors
    return legs


def test_a_case_whose_builder_returns_NO_MONITOR_declines_the_site():
    """The ruling: no observable, no measurement, and no green word either.

    The clauses below the decline are not evaluated at all — the row says the site
    was declined and why — so a reader cannot mistake it for a site that ran.
    """
    out = _run(_legs_declaring({"fused": [], "unfused": [], "array": []}))
    assert out["verdict"] == "NOT-APPLICABLE"
    assert out["monitors_declared"] == {"fused": 0, "unfused": 0, "array": 0}
    assert "NO MONITORS" in out["why"]
    # Declined, not run: none of the site's own measurements are in the row.
    assert "synchronized_state" not in out
    assert "restore_returns_the_state" not in out


def test_a_case_that_DECLARES_a_monitor_is_still_scored_and_can_still_fail():
    """THE MUTATION, in the direction that matters: the decline is not a silencer.

    The same legs that fail :func:`test_a_divergence_the_HALF_STEP_produced...` —
    one leg whose half-step treats it differently — but now every leg declares a
    monitor. The site runs and FAILS, which is what it is for. If the applicability
    were read off anything but the case's own monitors this is where it would go
    quiet.
    """
    legs = _legs_declaring({"fused": ["flux"], "unfused": ["flux"],
                            "array": ["flux"]})
    legs["fused"]["driver"] = _Driver(1.0, bump=3.0)
    out = _run(legs)
    assert out["verdict"] == "FAIL"
    assert out["monitors_declared"] == {"fused": 1, "unfused": 1, "array": 1}


def test_the_decline_FAILS_CLOSED_when_a_leg_declares_nothing_at_all():
    """A leg dict without the key is UNKNOWN, and unknown does not decline.

    One leg declaring no list — a harness that forgot to carry it, or an older
    caller — leaves the site armed rather than silently skipping it, which is the
    direction a wrong answer here has to fall.
    """
    legs = _legs_declaring({"fused": [], "unfused": []})  # "array" declares nothing
    legs["fused"]["driver"] = _Driver(1.0, bump=3.0)
    out = _run(legs)
    assert out["verdict"] == "FAIL"
    assert out["monitors_declared"]["array"] is None


def test_the_flux_clause_is_dropped_where_there_is_no_flux_and_kept_where_there_is():
    """``compare_spectra`` over two empty mappings is True, and it must not count.

    The enriched block says which: ``applicable`` is derived from the builder's
    return, and :func:`flux_agrees` contributes nothing on a case that declares no
    monitor while still carrying the comparison on a case that does.
    """
    empty = route_gate.stamp_flux_applicability({"monitors": 0, "identical": True}, 0)
    assert empty["applicable"] is False
    assert empty["migration_agrees"] is True
    assert route_gate.flux_agrees(empty) is True  # contributes no clause

    # THE MUTATION: a case that DOES declare a monitor, whose spectra differ, must
    # still fail the conjunct — the drop is about absence, not about disagreement.
    differing = route_gate.stamp_flux_applicability(
        {"monitors": 1, "identical": False}, 1)
    assert differing["applicable"] is True
    assert route_gate.flux_agrees(differing) is False
    agreeing = route_gate.stamp_flux_applicability(
        {"monitors": 2, "identical": True}, 2)
    assert route_gate.flux_agrees(agreeing) is True


def test_a_monitor_built_and_never_migrated_cannot_reach_the_no_monitor_branch():
    """The guard on the derivation, and the reason it is not just a length.

    ``applicable`` is read from what the BUILDER returned; ``monitors`` is what
    reached the driver. A monitor dropped in the lift would otherwise arrive here
    indistinguishable from a case that declares none — and would silence a flux
    comparison that should have run. The mismatch is its own case verdict.
    """
    dropped = route_gate.stamp_flux_applicability({"monitors": 0, "identical": True}, 1)
    assert dropped["applicable"] is True
    assert dropped["migration_agrees"] is False


@pytest.mark.parametrize("site_verdict,expected", [
    ("PASS", None),
    ("NOT-APPLICABLE", None),
    ("FAIL", "SECOND-CONSULT-FAILURE"),
    ("HARNESS-FAILURE", "SECOND-CONSULT-HARNESS-FAILURE"),
])
def test_the_case_verdict_carries_the_distinction_up(site_verdict, expected):
    """A reader of ``summary.json`` must see the two apart without opening the block.

    The case-level verdict used to collapse both to ``SECOND-CONSULT-FAILURE``, so a
    run that handed the site unequal legs and a run whose kernels disagreed at the
    half-step were the same word in the summary — and only the second is about the
    seam. ``None`` is the pass-through: a PASS at the site decides nothing here and
    the case's other clauses answer.
    """
    site = {"verdict": site_verdict, "why": "stub"}
    assert route_gate.case_verdict_for_second_consult(site) == expected
