"""The held form of the route gate's sync_in control says only what it measured.

WHY THIS FILE EXISTS. ``residency-sync_in-dropped`` (``_drop_one_sync_in``) has two
forms. Per-launch, it skips one upload and reads ``Residency.verify()`` at that instant,
so "the suppressed upload was a no-op" is a measurement. HELD, it patches
``Residency.acquire_write`` to leave the first claimed mirror clean, and reads no
``verify()`` at all. Until 2026-10-13 the held state kept the per-launch fields at the
zero they were initialised to, so the gate read that zero as "residency.verify() read
EMPTY at the instant of every skipped sync_in": on the round's held route campaign the
row read NOT-APPLICABLE with that text on 42 / 42 dispatched cases on both dispatch
legs, with ``suppressed`` 0 on every row. The same branch would also have read a held
null that DID suppress a write and did not diverge as NOT-APPLICABLE.

The held form now records what it reaches (entries into ``acquire_write``, those on a
registered mirror, and the sparse doors' counts over the armed steps) and is graded by
:func:`held_sync_in_verdict` on those fields alone. Host-only: stand-in residencies and
launches around the real ``launch.SyncedPlan``; the gate's own ``residency_controls``
is driven with the lift, the stepping and the byte comparator replaced.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_dispatch_metal_route as metal_route  # noqa: E402

from meep_gpu.metal_kernels import device as device_module  # noqa: E402
from meep_gpu.metal_kernels import launch  # noqa: E402

CONTROL = "residency-sync_in-dropped"
#: The text the frozen bytes of the round printed on every held row, which no code
#: on the held path measures.
UNMEASURED_CLAIM = "residency.verify() read EMPTY"


class _Mirror:
    def __init__(self, name):
        self.name = name
        self.state = device_module.HOST_OWNED


class _Launch:
    def __init__(self, name):
        self.name = name

    def run(self, *args, **kwargs):  # pragma: no cover - never stepped here
        raise AssertionError(f"{self.name} was launched by a host-only test")


def _held_residency_class():
    """A FRESH class per test: the control patches ``acquire_write`` on the class."""

    class _HeldResidency:
        held = frozenset({"Ez"})

        def __init__(self):
            self.mirrors = {"Ez": _Mirror("Ez")}
            self.sparse_scatters = 0
            self.sparse_zeros = 0
            self.double_subtracts = 0

        def _targets(self, target):
            if target not in self.mirrors:
                raise KeyError(target)
            return [self.mirrors[target]]

        def acquire_write(self, target):
            for mirror in self._targets(target):
                mirror.state = device_module.HOST_OWNED
            return target

        def sync_in(self, names=None):  # pragma: no cover - never stepped here
            raise AssertionError("sync_in reached in a host-only test")

        def sync_out(self, names=None):  # pragma: no cover - never stepped here
            raise AssertionError("sync_out reached in a host-only test")

    return _HeldResidency


def _held_driver(residency):
    plans = {"step_D": launch.SyncedPlan(_Launch("electric pair"), residency)}
    driver = types.SimpleNamespace(
        closed=False,
        _fast_path=types.SimpleNamespace(step_plan=types.SimpleNamespace(plans=plans)))
    driver.close = lambda: setattr(driver, "closed", True)
    return driver


# ------------------------------------------------------------- the arm itself ----

def test_the_held_arm_carries_no_field_it_does_not_measure():
    residency = _held_residency_class()()
    restore, state = metal_route._drop_one_sync_in(_held_driver(residency))  # noqa: SLF001
    try:
        assert state["mode"] == metal_route.HELD_SYNC_IN_MODE, state
        assert state["armed"] is True
        assert "steps_with_stale_mirrors" not in state, state
        assert "stale_mirrors" not in state, state
    finally:
        restore()


def test_the_held_arm_counts_its_reach_and_the_doors_over_the_armed_steps():
    cls = _held_residency_class()
    residency = cls()
    real = cls.acquire_write
    residency.sparse_scatters = 5      # before the arm: not this control's steps
    restore, state = metal_route._drop_one_sync_in(_held_driver(residency))  # noqa: SLF001
    try:
        residency.sparse_scatters += 3    # the doors write during the armed steps
        residency.double_subtracts += 2
    finally:
        restore()
    assert cls.acquire_write is real
    assert state["suppressed"] == 0 and state["acquire_calls"] == 0, state
    assert state["acquires_on_a_mirror"] == 0
    assert state["door_counts"] == {"sparse_scatters": 3, "sparse_zeros": 0,
                                    "double_subtracts": 2}, state


def test_a_held_acquire_on_a_mirror_is_left_unclaimed_once_and_counted():
    cls = _held_residency_class()
    residency = cls()
    restore, state = metal_route._drop_one_sync_in(_held_driver(residency))  # noqa: SLF001
    try:
        residency.acquire_write("Ez")
        assert residency.mirrors["Ez"].state == device_module.CLEAN_BOTH
        residency.acquire_write("Ez")     # the second claim is not suppressed
        assert residency.mirrors["Ez"].state == device_module.HOST_OWNED
        with pytest.raises(KeyError):
            residency.acquire_write("scratch")   # an unmirrored target still refuses
    finally:
        restore()
    assert state["suppressed"] == 1 and state["slot"] == "Ez", state
    assert state["acquire_calls"] == 3 and state["acquires_on_a_mirror"] == 2, state


# --------------------------------------------------------------- the verdict ----

def _state(**fields):
    base = {"mode": metal_route.HELD_SYNC_IN_MODE, "armed": True, "suppressed": 0,
            "slot": None, "acquire_calls": 0, "acquires_on_a_mirror": 0,
            "door_counts": {"sparse_scatters": 24, "sparse_zeros": 0,
                            "double_subtracts": 12}}
    base.update(fields)
    return base


def test_nothing_suppressed_and_identical_is_not_applicable_by_its_own_fields():
    verdict, why = metal_route.held_sync_in_verdict(_state(), identical=True)
    assert verdict == "NOT-APPLICABLE"
    assert UNMEASURED_CLAIM not in why, why
    assert "acquire_write was entered 0 time(s), 0 of them on a registered mirror" in why
    assert "'sparse_scatters': 24" in why and "verify() is not read" in why, why


def test_nothing_suppressed_and_divergent_is_a_harness_failure():
    verdict, _why = metal_route.held_sync_in_verdict(_state(), identical=False)
    assert verdict == "HARNESS-FAILURE"
    assert verdict not in metal_route.CONTROL_PASS


def test_a_suppressed_write_that_changes_nothing_is_a_null():
    state = _state(suppressed=1, slot="Ez", acquire_calls=4, acquires_on_a_mirror=4)
    verdict, why = metal_route.held_sync_in_verdict(state, identical=True)
    assert verdict == "NULL-DID-NOT-DIVERGE" and "'Ez'" in why, why
    assert verdict not in metal_route.CONTROL_PASS
    assert metal_route.held_sync_in_verdict(state, identical=False)[0] == \
        "DIVERGED-AS-REQUIRED"


# ------------------------------------------------ the gate's verdict on the row ----

@pytest.fixture
def drive_controls(monkeypatch):
    """Run the gate's ``residency_controls`` on a held composition, off-device.

    ``host_writes`` is what the stand-in stepping does on each armed step: ``"doors"``
    counts a sparse scatter and a device-side deposit, as the held route's host writes
    do; ``"acquire"`` claims the mirror through ``acquire_write``. The comparator
    reports the legs ``identical``; the sibling controls arm nothing.
    """

    def run(host_writes="doors", identical=True):
        legs = []
        cls = _held_residency_class()

        def pair(case, spec, res, counter, label):
            made = []
            for _ in range(2):
                residency = cls()
                made.append({"driver": _held_driver(residency), "residency": residency})
            legs.extend(made)
            return made[0], made[1]

        def step(case, leg, counter, num_steps):
            for _ in range(num_steps):
                if host_writes == "doors":
                    leg["residency"].sparse_scatters += 1
                    leg["residency"].double_subtracts += 1
                else:
                    leg["residency"].acquire_write("Ez")

        calls = []

        def compare(a, b):
            # Per control: the precondition (identical, as a fresh pair is), then the
            # verdict this run is set up to read.
            calls.append(None)
            same = identical or len(calls) % 2 == 1
            return {"identical": same, "arrays_differing": 0 if same else 1,
                    "differences": []}

        monkeypatch.setattr(metal_route, "_pair_of_lifted", pair)
        monkeypatch.setattr(metal_route, "step_leg", step)
        monkeypatch.setattr(metal_route.e2e, "collect_state", lambda driver: None)
        monkeypatch.setattr(metal_route.e2e, "compare_state", compare)
        monkeypatch.setattr(metal_route, "_drop_the_inner_bracket",
                            lambda driver: ((lambda: None), {"armed": False}))
        monkeypatch.setattr(metal_route, "_verify_after_every_sync_out",
                            lambda driver: ((lambda: None), {"reads": 1, "mirrors": 1}))
        rows = metal_route.residency_controls("pml_2d", {}, None, None)
        assert all(leg["driver"].closed for leg in legs)
        assert cls.acquire_write.__name__ == "acquire_write"   # restored
        return {row["control"]: row for row in rows}

    return run


def test_the_held_row_no_longer_claims_a_verify_reading(drive_controls):
    """MUST CATCH: on the frozen bytes every held row printed UNMEASURED_CLAIM."""
    row = drive_controls()[CONTROL]
    assert row["verdict"] == "NOT-APPLICABLE", row
    assert UNMEASURED_CLAIM not in row["why"], row["why"]
    steps = metal_route.CONTROL_STEPS
    assert f"over {steps} armed steps" in row["why"], row["why"]
    assert row["state"]["door_counts"]["sparse_scatters"] == steps, row["state"]
    assert row["state"]["acquire_calls"] == 0


def test_a_held_null_that_bit_and_did_not_diverge_fails_the_leg(drive_controls):
    """MUST CATCH: on the frozen bytes this row read NOT-APPLICABLE ("a no-op")."""
    row = drive_controls(host_writes="acquire")[CONTROL]
    assert row["verdict"] == "NULL-DID-NOT-DIVERGE", row
    assert row["verdict"] not in metal_route.CONTROL_PASS
    assert row["state"]["suppressed"] == 1, row["state"]
    assert row["state"]["acquires_on_a_mirror"] == metal_route.CONTROL_STEPS


def test_a_held_null_that_bit_and_diverged_passes(drive_controls):
    row = drive_controls(host_writes="acquire", identical=False)[CONTROL]
    assert row["verdict"] == "DIVERGED-AS-REQUIRED", row
