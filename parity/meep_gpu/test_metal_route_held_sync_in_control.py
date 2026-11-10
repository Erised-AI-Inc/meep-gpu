"""The held form of the route gate's sync_in control drops one deposit write, and must diverge.

Why this file exists. ``residency-sync_in-dropped`` (``_drop_one_sync_in``) has two
forms. Per-launch, it skips one upload and reads ``Residency.verify()`` at that instant,
so "the suppressed upload was a no-op" is a measurement. Under held residency an upload
is not the transport: between launches the field mirrors are device-owned, and the host
passes beside the kernels write them through the residency's sparse doors, which write
the current side and move no ownership. A null that withheld an ownership move through
``acquire_write`` never reached those writes, so on a held route it read NOT-APPLICABLE
on every dispatched case and the held seam had no must-diverge control of its own.

The held form now drops the device write of ONE source deposit
(``Residency.subtract_double_index``), the door every held composition drives on every
step, on a mirror only the device holds; it measures the drop (a deposit that changes no
stored word is passed through and counted) and is graded by
:func:`held_sync_in_verdict` on its own fields alone. These tests drive the real arm on a
real ``Residency`` whose mirrors live on the CPU (the doors take their CPU forms there),
and the gate's own ``residency_controls`` with the lift, the stepping and the byte
comparator replaced. The verdict tests need no PyTorch.
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_dispatch_metal_route as metal_route  # noqa: E402

from meep_gpu.metal_kernels import launch  # noqa: E402

CONTROL = "residency-sync_in-dropped"
CELLS = 16
POINTS = np.array([3, 4, 9], dtype=np.int64)


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - platform gate
        from conftest import requires_resource_skip  # noqa: PLC0415
        requires_resource_skip("torch", "the residency's mirrors are torch tensors")
    return torch


class _Launch:
    def run(self, *args, **kwargs):  # pragma: no cover - never stepped here
        raise AssertionError("a host-only test launched a kernel")


def _driver(residency):
    plans = {"step_D": launch.SyncedPlan(_Launch(), residency)}
    driver = types.SimpleNamespace(
        closed=False,
        _fast_path=types.SimpleNamespace(step_plan=types.SimpleNamespace(plans=plans)))
    driver.close = lambda: setattr(driver, "closed", True)
    return driver


def _held(dtype=np.float32, *, device_owned=True):
    """A held residency on the CPU with one field mirror, device-owned as between launches."""
    _torch()
    from meep_gpu.metal_kernels.device import DEVICE_OWNED, Residency  # noqa: PLC0415

    residency = Residency("cpu")
    host = (np.arange(CELLS) + 1).astype(dtype)
    residency.mirror("Ez", host, dtype=dtype)
    residency.arm_hold(["Ez"])
    if device_owned:
        residency.sync_out()            # what a held launch's bracket leaves behind
        assert residency.state_of("Ez") == DEVICE_OWNED
    return residency, host


def _device(residency):
    return residency._by_host_identity(residency.host("Ez")).tensor.detach().cpu().numpy().copy()  # noqa: SLF001


# ------------------------------------------------------------- the arm itself ----

def test_the_held_arm_carries_no_field_it_does_not_measure():
    class _Residency:
        held = frozenset({"Ez"})
        sparse_scatters = sparse_zeros = double_subtracts = 0

    restore, state = metal_route._drop_one_sync_in(_driver(_Residency()))  # noqa: SLF001
    try:
        assert state["mode"] == metal_route.HELD_SYNC_IN_MODE, state
        assert state["armed"] is True and state["suppressed"] == 0
        assert "steps_with_stale_mirrors" not in state, state
        assert "stale_mirrors" not in state, state
    finally:
        restore()
    assert state["deposits"] == 0 and state["door_counts"]["double_subtracts"] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_the_first_deposit_that_changes_a_word_on_a_device_owned_mirror_is_dropped(dtype):
    residency, host = _held(dtype)
    imaginary = 0.25 if dtype == np.complex64 else None
    door = type(residency).subtract_double_index
    restore, state = metal_route._drop_one_sync_in(_driver(residency))  # noqa: SLF001
    try:
        before = _device(residency)
        residency.subtract_double_index(host, POINTS, 0.0,
                                        0.0 if imaginary is not None else None)
        assert state["suppressed"] == 0 and state["deposits_unchanged"] == 1, state
        residency.subtract_double_index(host, POINTS, 0.5, imaginary)
        assert state["suppressed"] == 1 and state["slot"] == "Ez", state
        words = len(POINTS) * (2 if imaginary is not None else 1)
        assert state["words_dropped"] == words, state
        assert np.array_equal(_device(residency).view(np.uint32), before.view(np.uint32))
        residency.subtract_double_index(host, POINTS, 0.5, imaginary)   # runs shipped
        after = _device(residency)
        assert not np.array_equal(after.view(np.uint32), before.view(np.uint32))
        assert np.array_equal(np.delete(after, POINTS), np.delete(before, POINTS))
    finally:
        restore()
    assert type(residency).subtract_double_index is door
    assert state["deposits"] == 3, state
    assert state["door_counts"]["double_subtracts"] == 3, state


def test_a_deposit_on_a_mirror_the_device_does_not_own_is_passed_through():
    residency, host = _held(device_owned=False)
    restore, state = metal_route._drop_one_sync_in(_driver(residency))  # noqa: SLF001
    try:
        residency.subtract_double_index(host, POINTS, 0.5)
    finally:
        restore()
    assert state["suppressed"] == 0 and state["deposits_not_device_owned"] == 1, state
    assert np.array_equal(_device(residency), host)      # written on both sides


def test_another_residency_and_an_unmirrored_array_run_shipped():
    residency, host = _held()
    other, other_host = _held()
    loose = np.ones(CELLS, dtype=np.float32)
    restore, state = metal_route._drop_one_sync_in(_driver(residency))  # noqa: SLF001
    try:
        before = _device(other)
        other.subtract_double_index(other_host, POINTS, 0.5)
        assert not np.array_equal(_device(other), before)
        residency.subtract_double_index(loose, POINTS, 0.5)
        assert np.array_equal(loose[POINTS], np.full(len(POINTS), 0.5, dtype=np.float32))
    finally:
        restore()
    assert state["deposits"] == 1 and state["deposits_unmirrored"] == 1, state
    assert state["suppressed"] == 0


def test_acquire_write_is_counted_and_not_changed():
    from meep_gpu.metal_kernels.device import HOST_OWNED  # noqa: PLC0415

    residency, host = _held()
    restore, state = metal_route._drop_one_sync_in(_driver(residency))  # noqa: SLF001
    try:
        residency.acquire_write(host)
        assert residency.state_of("Ez") == HOST_OWNED
        with pytest.raises(KeyError):
            residency.acquire_write("not-a-mirror")
    finally:
        restore()
    assert state["acquire_calls"] == 2 and state["acquires_on_a_mirror"] == 1, state


# --------------------------------------------------------------- the verdict ----

def _state(**fields):
    base = {"mode": metal_route.HELD_SYNC_IN_MODE, "armed": True, "suppressed": 0,
            "slot": None, "deposits": 12, "deposits_unchanged": 12,
            "deposits_not_device_owned": 0, "deposits_unmirrored": 0,
            "words_dropped": 0, "acquire_calls": 0, "acquires_on_a_mirror": 0,
            "door_counts": {"sparse_scatters": 12, "sparse_zeros": 0,
                            "double_subtracts": 12}}
    base.update(fields)
    return base


def test_no_deposit_that_changes_a_word_is_not_applicable_by_its_own_fields():
    verdict, why = metal_route.held_sync_in_verdict(_state(), identical=True)
    assert verdict == "NOT-APPLICABLE"
    assert "entered 12 time(s)" in why and "12 changed no stored word" in why, why
    assert "verify()" not in why, why


def test_a_held_host_write_through_acquire_write_is_not_armed():
    state = _state(acquire_calls=3, acquires_on_a_mirror=3)
    verdict, why = metal_route.held_sync_in_verdict(state, identical=True)
    assert verdict == "NOT-ARMED" and "3 time(s) on a registered mirror" in why, why
    assert verdict not in metal_route.CONTROL_PASS
    assert metal_route.predict_control(CONTROL, state) == "NOT-ARMED"


def test_nothing_dropped_and_divergent_is_a_harness_failure():
    verdict, _why = metal_route.held_sync_in_verdict(_state(), identical=False)
    assert verdict == "HARNESS-FAILURE"
    assert verdict not in metal_route.CONTROL_PASS


def test_a_dropped_write_that_changes_nothing_downstream_is_a_null():
    state = _state(suppressed=1, slot="Ez", words_dropped=3)
    verdict, why = metal_route.held_sync_in_verdict(state, identical=True)
    assert verdict == "NULL-DID-NOT-DIVERGE" and "'Ez'" in why and "3 stored" in why, why
    assert verdict not in metal_route.CONTROL_PASS
    assert metal_route.held_sync_in_verdict(state, identical=False)[0] == \
        "DIVERGED-AS-REQUIRED"
    assert metal_route.predict_control(CONTROL, state) == "ARMS"


# ------------------------------------------------ the gate's verdict on the row ----

@pytest.fixture
def drive_controls(monkeypatch):
    """Run the gate's ``residency_controls`` on a held composition, off-device.

    ``host_writes`` is what the stand-in stepping does on each armed step:
    ``"deposit"`` subtracts a source amplitude through the deposit door, as a held
    route's source does; ``"silent"`` deposits a zero amplitude; ``"acquire"``
    claims the mirror through ``acquire_write``. The comparator reports the legs
    ``identical`` unless the device words of the two legs differ (``compare="words"``),
    which is the byte comparison the gate makes; the sibling controls arm nothing.
    """

    def run(host_writes="deposit", compare="words", controls=None):
        legs = []

        def pair(case, spec, res, counter, label):
            made = []
            for _ in range(2):
                residency, host = _held()
                made.append({"driver": _driver(residency), "residency": residency,
                             "host": host})
            legs.extend(made)
            return made[0], made[1]

        def step(case, leg, counter, num_steps):
            residency, host = leg["residency"], leg["host"]
            for _ in range(num_steps):
                if host_writes == "acquire":
                    residency.acquire_write(host)
                    residency.sync_out()
                else:
                    residency.subtract_double_index(
                        host, POINTS, 0.5 if host_writes == "deposit" else 0.0)

        def collect(driver):
            plan = driver._fast_path.step_plan.plans["step_D"]  # noqa: SLF001
            return _device(plan.residency)

        def compare_state(a, b):
            same = (np.array_equal(a.view(np.uint32), b.view(np.uint32))
                    if compare == "words" else compare)
            return {"identical": same, "arrays_differing": 0 if same else 1,
                    "differences": []}

        monkeypatch.setattr(metal_route, "_pair_of_lifted", pair)
        monkeypatch.setattr(metal_route, "step_leg", step)
        monkeypatch.setattr(metal_route.e2e, "collect_state", collect)
        monkeypatch.setattr(metal_route.e2e, "compare_state", compare_state)
        monkeypatch.setattr(metal_route, "_drop_the_inner_bracket",
                            lambda driver: ((lambda: None), {"armed": False}))
        monkeypatch.setattr(metal_route, "_verify_after_every_sync_out",
                            lambda driver: ((lambda: None), {"reads": 1, "mirrors": 1}))
        rows = metal_route.residency_controls("pml_2d", {}, None, None, controls)
        assert all(leg["driver"].closed for leg in legs)
        assert all(type(leg["residency"]).subtract_double_index.__name__
                   == "subtract_double_index" for leg in legs)        # restored
        return {row["control"]: row for row in rows}

    return run


def test_a_held_route_whose_source_deposits_reads_diverged(drive_controls):
    """MUST CATCH: the held null reached no door and read NOT-APPLICABLE here."""
    row = drive_controls()[CONTROL]
    assert row["verdict"] == "DIVERGED-AS-REQUIRED", row
    assert row["state"]["suppressed"] == 1 and row["state"]["slot"] == "Ez", row["state"]
    steps = metal_route.CONTROL_STEPS
    assert row["state"]["deposits"] == steps, row["state"]
    assert row["state"]["door_counts"]["double_subtracts"] == steps, row["state"]


def test_a_dropped_deposit_the_comparison_cannot_see_fails_the_leg(drive_controls):
    row = drive_controls(compare=True)[CONTROL]
    assert row["verdict"] == "NULL-DID-NOT-DIVERGE", row
    assert row["verdict"] not in metal_route.CONTROL_PASS


def test_a_zero_amplitude_source_is_not_applicable_and_says_why(drive_controls):
    row = drive_controls(host_writes="silent")[CONTROL]
    steps = metal_route.CONTROL_STEPS
    assert row["verdict"] == "NOT-APPLICABLE", row
    assert f"{steps} changed no stored word" in row["why"], row["why"]


def test_host_writes_through_acquire_write_fail_the_leg(drive_controls):
    row = drive_controls(host_writes="acquire")[CONTROL]
    assert row["verdict"] == "NOT-ARMED", row
    assert row["state"]["acquires_on_a_mirror"] == metal_route.CONTROL_STEPS


def test_a_subset_of_the_controls_runs_only_those(drive_controls):
    rows = drive_controls(controls=(CONTROL,))
    assert list(rows) == [CONTROL]
