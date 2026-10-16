"""The route gate's placement control looks through a SPENT start of H, and only that.

WHY THIS FILE EXISTS. ``residency-inner-bracket-dropped`` (``_drop_the_inner_bracket``)
moves a wrapper's residency bracket from its inner launch to the whole slot and must
diverge wherever the wrapper does host work before that launch. Series entry 134 put
``launch.StartOfHPlan`` around whatever serves a run's first ``update_H``; its inner is
a bracketed launch, so the control armed on it. But the start does host work only while
``fields.h_separated`` is False, and the control arms after one step, when the flag is
True: moving the bracket across a pass-through cannot change a byte, and the held route
campaign of 2026-10-13 read NULL-DID-NOT-DIVERGE on ``special_kz_2d`` and ``bfast_1d``
on both dispatch legs, which failed them. The control now judges such a slot by the
start's occupant, and a start that would still do host work at an armed step is armed
on as before.

Host-only: the plans are composed from the real ``launch`` and wrapper classes around
stand-in launches, and the gate's own ``residency_controls`` is driven with the lift,
the stepping and the byte comparator replaced, so the verdict and its text are the
gate's.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_dispatch_metal_route as metal_route  # noqa: E402

from meep_gpu import deposit_repair, withdraw_hoist  # noqa: E402
from meep_gpu.metal_kernels import launch  # noqa: E402

CONTROL = "residency-inner-bracket-dropped"


class _Launch:
    """A device plan that only records that it ran."""

    def __init__(self, name):
        self.name = name

    def run(self, *args, **kwargs):  # pragma: no cover - never stepped here
        raise AssertionError(f"{self.name} was launched by a host-only test")


class _Residency:
    held = frozenset()

    def sync_in(self, names=None):  # pragma: no cover - never stepped here
        raise AssertionError("sync_in reached in a host-only test")

    def sync_out(self, names=None):  # pragma: no cover - never stepped here
        raise AssertionError("sync_out reached in a host-only test")


def _composition(fields, *, h_occupant="single"):
    """The shape ``special_kz_2d`` and ``bfast_1d`` carried on the 2026-10-13 route.

    ``step_D`` holds a leading deposit repair around its bracketed launch and
    ``update_H`` a start of H around its occupant: a bare bracketed single
    (``"single"``) or a leading withdraw around one (``"withdraw"``), each bracketed
    by ``launch.synced`` exactly as the composer brackets it.
    """
    residency = _Residency()
    repair = deposit_repair.LeadingRepairPlan(_Launch("electric pair"), fields, None, (),
                                              "D")
    if h_occupant == "single":
        occupant = _Launch("update_H single")
    else:
        occupant = withdraw_hoist.LeadingWithdrawPlan(_Launch("H->D weld"), fields, ())
    start = launch.StartOfHPlan(occupant, fields, None, after_launch=False)
    plans = {"step_D": launch.synced(repair, residency),
             "update_H": launch.synced(start, residency)}
    driver = types.SimpleNamespace(
        fields=fields, closed=False,
        _fast_path=types.SimpleNamespace(step_plan=types.SimpleNamespace(plans=plans)))
    driver.close = lambda: setattr(driver, "closed", True)
    return driver, plans


def _snapshot(plans):
    """Every object identity the control may rewire, slot by slot."""
    out = {}
    for slot, entry in plans.items():
        chain, node = [], entry
        while node is not None and len(chain) < 5:
            chain.append(id(node))
            node = node.inner if type(node).__name__ != "_Launch" else None
        out[slot] = tuple(chain)
    return out


# ------------------------------------------------------------- the arm itself ----

def test_a_spent_start_is_looked_through_to_its_occupant():
    """After the first update_H the start is a pass-through: nothing is armed."""
    fields = types.SimpleNamespace(h_separated=True)
    driver, plans = _composition(fields)
    before = _snapshot(plans)
    restore, state = metal_route._drop_the_inner_bracket(driver)  # noqa: SLF001
    try:
        assert state["armed"] is False, state
        assert state["slots"] == [], state
        assert state["read_only_slots"] == ["step_D"], state
        assert state["wrapper_kinds"] == {"step_D": "LeadingRepairPlan"}, state
        assert state["looked_through"] == {"update_H": {
            "wrapper": "StartOfHPlan", "occupant": "SyncedPlan", "starts_at_arm": 0}}
        assert _snapshot(plans) == before
    finally:
        restore()
    assert _snapshot(plans) == before
    assert "started_while_looked_through" not in state


def test_a_start_still_owed_is_not_looked_through_and_the_control_arms_on_it():
    """Before the first update_H the start WRITES first: it is armed on, as before."""
    fields = types.SimpleNamespace(h_separated=False)
    driver, plans = _composition(fields)
    before = _snapshot(plans)
    start = plans["update_H"]
    bracketed = start.inner
    restore, state = metal_route._drop_the_inner_bracket(driver)  # noqa: SLF001
    try:
        assert state["armed"] is True, state
        assert state["slots"] == ["update_H"], state
        assert state["wrapper_kinds"] == {"step_D": "LeadingRepairPlan",
                                          "update_H": "StartOfHPlan"}, state
        assert state.get("looked_through", {}) == {}, state
        # The bracket now sits around the whole slot, and the start's own launch is bare.
        moved = plans["update_H"]
        assert isinstance(moved, launch.SyncedPlan) and moved.inner is start
        assert start.inner is bracketed.inner
    finally:
        restore()
    assert plans["update_H"] is start and start.inner is bracketed
    assert _snapshot(plans) == before


def test_a_writing_occupant_of_a_spent_start_is_armed_inside_it():
    """A spent start does not hide a wrapper whose own pre-launch work writes."""
    fields = types.SimpleNamespace(h_separated=True)
    driver, plans = _composition(fields, h_occupant="withdraw")
    before = _snapshot(plans)
    start = plans["update_H"]
    withdraw = start.inner
    bracketed = withdraw.inner
    assert isinstance(bracketed, launch.SyncedPlan)
    restore, state = metal_route._drop_the_inner_bracket(driver)  # noqa: SLF001
    try:
        assert state["armed"] is True, state
        assert state["slots"] == ["update_H"], state
        assert state["wrapper_kinds"]["update_H"] == "LeadingWithdrawPlan", state
        assert state["looked_through"]["update_H"]["occupant"] == "LeadingWithdrawPlan"
        # The start stays in its slot; inside it the bracket moved off the launch.
        assert plans["update_H"] is start
        assert isinstance(start.inner, launch.SyncedPlan)
        assert start.inner.inner is withdraw
        assert withdraw.inner is bracketed.inner
    finally:
        restore()
    assert plans["update_H"] is start and start.inner is withdraw
    assert withdraw.inner is bracketed
    assert _snapshot(plans) == before


def test_a_start_that_ran_while_looked_through_is_named_at_restore():
    fields = types.SimpleNamespace(h_separated=True)
    driver, plans = _composition(fields)
    restore, state = metal_route._drop_the_inner_bracket(driver)  # noqa: SLF001
    plans["update_H"].starts += 1    # a start performed during the armed steps
    restore()
    assert state["started_while_looked_through"] == ["update_H"], state


# ------------------------------------------------ the gate's verdict on the row ----

@pytest.fixture
def drive_controls(monkeypatch):
    """Run the gate's ``residency_controls`` on the composition above, off-device.

    The two legs are built before the first step, as the lift leaves them
    (``h_separated`` False). ``first_step_separates_h`` says whether the stand-in
    stepping marks H separated after a step, as the driver does after every step;
    the byte comparator reports the two legs identical, which is what a
    pass-through bracket move produces. The sibling controls are replaced by stubs
    that arm nothing, so only the row under test reads the composition.
    """

    def run(first_step_separates_h=True, h_occupant="single"):
        legs = []

        def pair(case, spec, res, counter, label):
            made = []
            for _ in range(2):
                driver, _plans = _composition(types.SimpleNamespace(h_separated=False),
                                              h_occupant=h_occupant)
                made.append({"driver": driver})
            legs.extend(made)
            return made[0], made[1]

        def step(case, leg, counter, num_steps):
            if first_step_separates_h:
                leg["driver"].fields.h_separated = True

        identical = {"identical": True, "arrays_differing": 0, "differences": []}
        monkeypatch.setattr(metal_route, "_pair_of_lifted", pair)
        monkeypatch.setattr(metal_route, "step_leg", step)
        monkeypatch.setattr(metal_route.e2e, "collect_state", lambda driver: None)
        monkeypatch.setattr(metal_route.e2e, "compare_state", lambda a, b: dict(identical))
        monkeypatch.setattr(metal_route, "_drop_one_sync_in",
                            lambda driver: ((lambda: None), {"armed": True,
                                                             "steps_with_stale_mirrors": 1}))
        monkeypatch.setattr(metal_route, "_verify_after_every_sync_out",
                            lambda driver: ((lambda: None), {"reads": 1, "mirrors": 1}))
        rows = metal_route.residency_controls("special_kz_2d", {}, None, None)
        assert all(leg["driver"].closed for leg in legs)
        return {row["control"]: row for row in rows}

    return run


def test_the_control_reads_not_applicable_after_the_first_update_h(drive_controls):
    """MUST CATCH: on the frozen bytes this row read NULL-DID-NOT-DIVERGE."""
    row = drive_controls()[CONTROL]
    assert row["verdict"] == "NOT-APPLICABLE", row
    assert row["state"]["armed"] is False
    assert row["state"]["looked_through"]["update_H"]["wrapper"] == "StartOfHPlan"
    assert "looked through to its occupant" in row["why"], row["why"]
    assert "only READS the fields" in row["why"], row["why"]


def test_a_writing_occupant_still_fails_the_leg_when_nothing_diverges(drive_controls):
    """Looking through the start keeps the control armed on what writes inside it."""
    row = drive_controls(h_occupant="withdraw")[CONTROL]
    assert row["verdict"] == "NULL-DID-NOT-DIVERGE", row
    assert row["verdict"] not in metal_route.CONTROL_PASS
    assert row["state"]["slots"] == ["update_H"]


def test_the_null_names_this_controls_own_slots_and_wrappers(drive_controls):
    """Where the start is still owed it is armed on, and its NULL says what moved."""
    row = drive_controls(first_step_separates_h=False)[CONTROL]
    assert row["verdict"] == "NULL-DID-NOT-DIVERGE", row
    assert row["state"]["slots"] == ["update_H"]
    why = row["why"]
    assert "None" not in why, why
    assert "['update_H']" in why and "StartOfHPlan" in why, why
    assert f"for {metal_route.CONTROL_STEPS} steps" in why, why
    assert row["state"].get("looked_through", {}) == {}


def test_a_start_that_ran_while_looked_through_is_a_harness_failure(monkeypatch,
                                                                    drive_controls):
    real = metal_route._drop_the_inner_bracket  # noqa: SLF001

    def arm(driver):
        restore, state = real(driver)

        def restore_after_a_start():
            driver._fast_path.step_plan.plans["update_H"].starts += 1  # noqa: SLF001
            restore()

        return restore_after_a_start, state

    monkeypatch.setattr(metal_route, "_drop_the_inner_bracket", arm)
    row = drive_controls()[CONTROL]
    assert row["verdict"] == "HARNESS-FAILURE", row
    assert row["verdict"] not in metal_route.CONTROL_PASS
