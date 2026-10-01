"""Held Metal residency as the default, and the fallback that keeps a supported run whole.

``MEEP_GPU_METAL_RESIDENCY`` unset now means ``held``: the device keeps the engine's
volumes between launches and the host acquires them. Three mechanisms in
``metal_dispatch`` make that safe for the documented driver surface, and each test
below pins one of them against the ARRAY PATH, over every array ``collect_state``
walks on the ``Fields`` and the PML -- not the four primaries the first probes read:

* **the freeze releases the prior hold** (``_release_prior_hold``). ``set_field``,
  ``set_epsilon``, ``add_source`` and ``invalidate_fast_path`` drop the plan and
  release nothing, so the next freeze met arrays the dead plan still sealed and the
  next launch raised ``assignment destination is read-only``;
* **``reset`` releases before it zeroes** (the ``reset`` layer over the barrier
  class). ``Fields.reset`` fills in place and runs before the driver invalidates;
* **the release is full**: the polarization states' dict and ``_scratch`` barriers
  go too. Stripping only the ``Fields`` class swap was measured silently wrong on
  ``dispersive_2d`` (1239 of 4800 words on each of four ``P``/``P_prev`` arrays).

And the admission rung refuses a hold it cannot serve BY NAME, to the array path --
never to shipped, which is slower than the array path at every measured size.

EVERY TEST SETS BOTH SWITCHES ITSELF. The package conftest pins
``MEEP_GPU_DISPATCH=0`` for every test, so a test that forgot the enable would run
the array path against the array path and pass vacuously; each dispatching leg here
asserts the plan it froze actually HELD (mode ``held``, held count > 0, nothing
refused). The held leg always freezes FIRST: rung 8bM installs the flush policy
process-wide, and the array reference must step under it to be comparable.

Progress reporting: each parametrised case prints one flushed line with what it measured.
"""

from __future__ import annotations

import gc
import os
import pathlib
import sys
import weakref

import numpy
import pytest

_PARITY = pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"
if str(_PARITY) not in sys.path:
    sys.path.insert(0, str(_PARITY))

from conftest import requires_resource_skip  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - platform gate
    requires_resource_skip("torch", "held residency mirrors onto MPS through torch",
                           allow_module_level=True)
if not torch.backends.mps.is_available():  # pragma: no cover - platform gate
    requires_resource_skip("mps_device", "held residency mirrors onto an MPS device",
                           allow_module_level=True)
try:
    import meep as mp  # noqa: E402
except ImportError:  # pragma: no cover - platform gate
    requires_resource_skip("meep", "the cases are lifted from mp.Simulation declarations",
                           allow_module_level=True)

import gate_dispatch_end_to_end as e2e  # noqa: E402
import gate_dispatch_metal_route as route  # noqa: E402

from meep_gpu import fastpath, host_writes, metal_dispatch  # noqa: E402
from meep_gpu import lift_simulation  # noqa: E402
from meep_gpu.absorber import AbsorberLayer  # noqa: E402
from meep_gpu.dispersion import Susceptibility  # noqa: E402
from meep_gpu.driver import FdtdDriver  # noqa: E402
from meep_gpu.metal_kernels import barrier, device  # noqa: E402

RESIDENCY_ENV = metal_dispatch.RESIDENCY_ENV
RES = 10

#: Steps before and after the operation, per case. ``dispersive_2d`` needs enough
#: that P is non-zero at the operation and sizeable at the end (measured: 0 at step
#: 10, 6e-5 at 100); at 13 steps an earlier probe read "identical" over all-zero P.
STEPS = {"pml_2d": 15, "offdiag_2d": 30, "dispersive_2d": 100}

#: The fewest arrays ``collect_state`` must return per case, so a comparison over a
#: shrunken walk cannot pass. Measured on the array path at resolution 10.
MINIMUM_ARRAYS = {"pml_2d": 30, "offdiag_2d": 30, "dispersive_2d": 38}


def say(line: str) -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Process state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _leave_the_process_as_this_file_found_it():
    """Put the FPU and the policy bookkeeping back when this file is done.

    Every held freeze here passes rung 8bM, which installs ``flush`` and drives the
    host FPU there; an attained install deliberately stands. ``test_metal_dispatch``
    carries the same guard for the same measured leak.
    """
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    flushing_at_entry = backends.subnormals_flushed()
    try:
        yield
    finally:
        subnormal_policy._reset_for_tests()  # noqa: SLF001
        if backends.subnormals_flushed() != flushing_at_entry:
            subnormal_policy._drive_host_fpu(flushing_at_entry)  # noqa: SLF001


@pytest.fixture(autouse=True)
def _no_hold_outlives_a_test():
    """Release every residency still registered, so ``holding()`` never leaks."""
    yield
    for residency in host_writes._live():  # noqa: SLF001
        metal_dispatch.release_residency_hold(residency, None)


def _switches(monkeypatch, *, dispatch: bool, mode=None) -> None:
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1" if dispatch else "0")
    if mode is None:
        monkeypatch.delenv(RESIDENCY_ENV, raising=False)
    else:
        monkeypatch.setenv(RESIDENCY_ENV, mode)


def _lift(case: str) -> FdtdDriver:
    mp.verbosity(0)
    if case == "pml_2d":
        sim, _monitors, _until = e2e.case_pml_2d(mp, RES)
    else:
        sim, _monitors, _until = route.CASES[case](mp, RES)
    driver = lift_simulation(sim, prefer_gpu=True)
    assert driver.gpu == "metal", driver.gpu
    return driver


def _block(driver) -> dict:
    return dict((driver.fast_path_report() or {}).get("residency") or {})


def _assert_held(driver) -> dict:
    """The plan this driver froze last is a HELD Metal plan, not the array path."""
    report = driver.fast_path_report() or {}
    block = dict(report.get("residency") or {})
    assert driver.active_step_path == "fused", report.get("refused_because")
    assert block.get("mode") == metal_dispatch.RESIDENCY_HELD, block.get("mode")
    assert len(block.get("held") or ()) > 0, "a hold that held nothing is not a hold"
    assert list(block.get("hold_refused") or ()) == [], block.get("hold_refused")
    return block


def _identical(held, ref, case: str) -> dict:
    verdict = e2e.compare_state(e2e.collect_state(held), e2e.collect_state(ref))
    assert verdict["identical"], (verdict["differences"][:4],
                                  verdict["missing_on_one_side"][:4])
    assert verdict["arrays"] >= MINIMUM_ARRAYS[case], verdict["arrays"]
    return verdict


def _max_abs_p(state_arrays: dict) -> float:
    values = [float(numpy.abs(v).max()) for k, v in state_arrays.items()
              if ".P[" in k or ".P_prev[" in k]
    return max(values) if values else 0.0


def _engine_read_only(driver) -> list:
    return [label for label, array in metal_dispatch._engine_arrays_raw(driver.fields)  # noqa: SLF001
            if not array.flags.writeable]


def _held_then_reference(monkeypatch, case: str, steps: int, mode="held"):
    """Freeze the held leg FIRST (it installs the policy), then step the reference."""
    held, ref = _lift(case), _lift(case)
    _switches(monkeypatch, dispatch=True, mode=mode)
    held.run(num_steps=steps)
    _switches(monkeypatch, dispatch=False)
    ref.run(num_steps=steps)
    return held, ref


def _mid(driver):
    nx, ny, _nz = (int(v) for v in driver.shape)
    return (nx // 2, ny // 2, 0)


def _set_field(driver) -> None:
    seed = driver.get_field("Dz", cell_centered=False)
    seed[_mid(driver)] += numpy.float32(1e-3)
    driver.set_field("Dz", seed)


def _set_epsilon(driver) -> None:
    driver.set_epsilon(numpy.asarray(driver.get_epsilon(), dtype=numpy.float32))


def _add_source(driver) -> None:
    driver.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0),
                       "size": (0.0, 0.0, 0.0), "frequency": 0.25,
                       "source_type": "continuous"})


REFREEZE_OPERATIONS = {
    "set_field": _set_field,
    "set_epsilon": _set_epsilon,
    "add_source": _add_source,
    "invalidate_fast_path": lambda d: d.invalidate_fast_path(),
    "reset": lambda d: d.reset(),
}


# ---------------------------------------------------------------------------
# 1. The default, and the explicit values
# ---------------------------------------------------------------------------

def test_the_default_with_the_variable_unset_is_held(monkeypatch):
    monkeypatch.delenv(RESIDENCY_ENV, raising=False)
    assert metal_dispatch._residency_mode() == metal_dispatch.RESIDENCY_HELD  # noqa: SLF001
    driver = _lift("pml_2d")
    try:
        _switches(monkeypatch, dispatch=True, mode=None)
        driver.run(num_steps=3)
        block = _assert_held(driver)
        assert block["admission"]["refused"] is False, block["admission"]
        assert block["released_prior_hold"] is None, "a first freeze has nothing to release"
    finally:
        driver.close()


def test_explicit_shipped_still_ships(monkeypatch):
    driver = _lift("pml_2d")
    try:
        _switches(monkeypatch, dispatch=True, mode=metal_dispatch.RESIDENCY_SHIPPED)
        driver.run(num_steps=3)
        block = _block(driver)
        assert driver.active_step_path == "fused"
        assert block["mode"] == metal_dispatch.RESIDENCY_SHIPPED
        assert block["held"] == [] and block["hoisted"] == []
        assert block["invariant"] == "host-authoritative between launches"
        assert block["admission"] is None, "admission is the hold's rung only"
        assert _engine_read_only(driver) == [], "shipped seals nothing"
        assert not host_writes.holding()
    finally:
        driver.close()


def test_a_misspelled_mode_refuses_by_name_before_the_policy_install(monkeypatch):
    """Read at the admission rung, the typo is a named refusal on an untouched process.

    Read inside the arming, after rung 8bM, it escaped the ladder as ``internal
    error`` with the subnormal policy already installed.
    """
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    subnormal_policy._reset_for_tests()  # noqa: SLF001
    assert not subnormal_policy.policy_is_installed()
    driver = _lift("pml_2d")
    try:
        _switches(monkeypatch, dispatch=True, mode="hled")
        driver.run(num_steps=3)
        report = driver.fast_path_report() or {}
        assert driver.active_step_path == "array"
        assert report["decision"] == "refused"
        reason = report["refused_because"]
        assert RESIDENCY_ENV in reason and "'hled'" in reason, reason
        assert "internal error" not in reason, reason
        assert not subnormal_policy.policy_is_installed(), (
            "a configuration refused by name must not have its arithmetic moved")
        assert _engine_read_only(driver) == []
        assert not host_writes.holding()
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# 2. Every documented re-freeze, and reset, mid-run: identical, then held again
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operation", sorted(REFREEZE_OPERATIONS))
@pytest.mark.parametrize("case", ["pml_2d", "offdiag_2d", "dispersive_2d"])
def test_a_mid_run_mutation_completes_identically_and_holds_again(monkeypatch, case,
                                                                   operation):
    steps = STEPS[case]
    held, ref = _held_then_reference(monkeypatch, case, steps)
    try:
        first = _assert_held(held)
        REFREEZE_OPERATIONS[operation](held)
        REFREEZE_OPERATIONS[operation](ref)
        _switches(monkeypatch, dispatch=True, mode="held")
        held.run(num_steps=steps)
        _switches(monkeypatch, dispatch=False)
        ref.run(num_steps=steps)

        block = _assert_held(held)
        released = block["released_prior_hold"]
        if operation == "reset":
            # The layer released at the mutator, so the freeze found nothing held.
            assert released is None, released
        else:
            assert released is not None, "the re-freeze must release the old hold"
            assert released["names_flushed"], released
            assert released["sealed_without_a_live_holder"] == [], released
            assert "barrier_key" in released["released"][0]["found_by"], released
        plan_residency = held._fast_path.residency  # noqa: SLF001
        for state in held.fields.polarizations:
            for attribute in ("P", "P_prev"):
                mapping = getattr(state, attribute)
                assert isinstance(mapping, barrier.BarrierDict), attribute
                assert mapping._residency is plan_residency, (  # noqa: SLF001
                    f"{attribute} is bound to a residency other than the live one")
        verdict = _identical(held, ref, case)
        reference_state = e2e.collect_state(ref)
        p_max = _max_abs_p(reference_state)
        if case == "dispersive_2d":
            assert any(".P[" in name for name in reference_state), "P is not compared"
            assert p_max > 0.0, "an identical verdict over all-zero P is vacuous"
        say(f"{case} {operation}: identical over {verdict['arrays']} arrays "
            f"({verdict['words_compared']} words); held {len(first['held'])} -> "
            f"{len(block['held'])}, flushed "
            f"{len((released or {}).get('names_flushed') or ())}, max|P| {p_max:.3g}")
    finally:
        held.close()
        ref.close()


# ---------------------------------------------------------------------------
# 3. The release itself
# ---------------------------------------------------------------------------

def test_the_full_release_strips_every_guard_and_is_safe_twice_and_after_close(
        monkeypatch):
    held, ref = _held_then_reference(monkeypatch, "dispersive_2d", 40)
    try:
        _assert_held(held)
        fields = held.fields
        residency = held._fast_path.residency  # noqa: SLF001
        original = type(fields).__bases__[0].__bases__[0]
        assert "reset" in vars(type(fields)), "the reset layer is installed"

        moved = metal_dispatch.release_residency_hold(residency, fields)
        assert moved, "a hold 40 steps in owns words the release must flush"
        assert type(fields) is original
        assert barrier._ORIGINAL_CLASS not in vars(fields)  # noqa: SLF001
        assert barrier._BARRIER_RESIDENCY not in vars(fields)  # noqa: SLF001
        for state in fields.polarizations:
            assert type(state.P) is dict and type(state.P_prev) is dict
            assert barrier._BARRIER_RESIDENCY not in vars(state)  # noqa: SLF001
        assert _engine_read_only(held) == []
        assert residency not in host_writes._live()  # noqa: SLF001
        assert residency.held == frozenset() and residency.hoisted == frozenset()

        assert metal_dispatch.release_residency_hold(residency, fields) == ()
        assert type(fields) is original
        held.close()
        assert metal_dispatch.release_residency_hold(residency, None) == ()
        assert metal_dispatch._release_prior_hold(None, "a closed driver") is None  # noqa: SLF001
    finally:
        held.close()
        ref.close()


def test_close_with_a_live_hold_leaves_a_release_that_tolerates_no_fields(monkeypatch):
    held, ref = _held_then_reference(monkeypatch, "pml_2d", 5)
    try:
        _assert_held(held)
        residency = held._fast_path.residency  # noqa: SLF001
        held.close()
        assert held.fields is None
        moved = metal_dispatch.release_residency_hold(residency, None)
        assert isinstance(moved, tuple)
        assert residency not in host_writes._live()  # noqa: SLF001
    finally:
        held.close()
        ref.close()


def test_the_reset_layer_keeps_barriered_names_truthful_and_leaves_with_the_barrier(
        monkeypatch):
    driver = _lift("pml_2d")
    other = _lift("pml_2d")
    try:
        _switches(monkeypatch, dispatch=True, mode="held")
        driver.run(num_steps=3)
        other.run(num_steps=3)
        _assert_held(driver)
        layer = type(driver.fields)
        barriered_class = layer.__bases__[0]
        original = barriered_class.__bases__[0]
        assert set(vars(layer)) & {"reset"} == {"reset"}
        properties = sorted(name for name, value in vars(barriered_class).items()
                            if isinstance(value, property))
        assert properties, "the barrier installed no names"
        assert list(barrier.barriered_names(driver.fields)) == properties
        assert type(other.fields) is layer, "one layer per barrier class, not per freeze"

        driver.reset()
        assert type(driver.fields) is original
        assert barrier.barriered_names(driver.fields) == ()
        assert _engine_read_only(driver) == []
        driver.run(num_steps=2)
        _assert_held(driver)
        assert type(driver.fields) is layer
    finally:
        driver.close()
        other.close()


# ---------------------------------------------------------------------------
# 4. Admission: each rule refuses BY NAME, to the array path
# ---------------------------------------------------------------------------

def test_rule_a_a_synced_pass_outside_the_door_set_refuses_to_the_array_path(
        monkeypatch):
    """``pml_2d``'s one synced pass is ``fill_D``; take it out of the set."""
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    monkeypatch.setattr(metal_dispatch, "HELD_DOOR_WIRED_PASSES",
                        metal_dispatch.HELD_DOOR_WIRED_PASSES - {"fill_D"})
    subnormal_policy._reset_for_tests()  # noqa: SLF001
    held, ref = _held_then_reference(monkeypatch, "pml_2d", STEPS["pml_2d"])
    try:
        report = held.fast_path_report() or {}
        assert held.active_step_path == "array"
        assert report["decision"] == "refused"
        reason = report["refused_because"]
        assert "fill_D" in reason and "performance guard" in reason, reason
        assert "slower than the array path" in reason, reason
        assert report["residency"]["admission"]["synced_outside_door_wired_set"] == [
            "fill_D"]
        assert not subnormal_policy.policy_is_installed(), (
            "the refusal came before rung 8bM moved the arithmetic")
        assert _engine_read_only(held) == [] and not host_writes.holding()
        _identical(held, ref, "pml_2d")
    finally:
        held.close()
        ref.close()


def test_rule_b_an_engine_owned_mirror_that_cannot_be_sealed_refuses(monkeypatch):
    real = device.Mirror.sealable
    monkeypatch.setattr(device.Mirror, "sealable",
                        lambda self: False if self.name == "Dz" else real(self))
    held, ref = _held_then_reference(monkeypatch, "pml_2d", STEPS["pml_2d"])
    try:
        report = held.fast_path_report() or {}
        assert held.active_step_path == "array"
        assert "cannot seal the engine-owned mirror(s) Dz" in report["refused_because"]
        assert report["residency"]["admission"]["unsealable_engine_owned"] == ["Dz"]
        _identical(held, ref, "pml_2d")
    finally:
        held.close()
        ref.close()


def test_rule_b_names_seals_a_dead_hold_left_and_the_array_path_raises_on_them(
        monkeypatch):
    """The one way a seal outlives its residency: the key removed without a release.

    Unsealing such arrays was measured to complete WITHOUT raising and differ from
    the array path on 12 of 32 arrays, because the device words died with the
    residency. So the freeze reports them and refuses by name, and the array path's
    first write to one raises instead of computing on stale words.
    """
    warm = _lift("pml_2d")                  # the first plan in a process is pinned
    _switches(monkeypatch, dispatch=True, mode="held")
    warm.run(num_steps=1)
    warm.close()
    driver = _lift("pml_2d")
    try:
        driver.run(num_steps=10)
        _assert_held(driver)
        dead = weakref.ref(driver._fast_path.residency)  # noqa: SLF001
        driver.invalidate_fast_path()
        barrier.remove_read_barrier(driver.fields)   # the key goes, nothing released
        gc.collect()
        assert dead() is None, "the residency must be gone for the seals to be orphaned"
        with pytest.raises(ValueError, match="read-only"):
            driver.run(num_steps=1)
        report = driver.fast_path_report() or fastpath.last_dispatch_report() or {}
        assert report["decision"] == "refused" and report["step_path"] == "array"
        orphaned = report["residency"]["released_prior_hold"][
            "sealed_without_a_live_holder"]
        assert "Dz" in orphaned, orphaned
        assert "no live residency holds" in report["refused_because"]
    finally:
        driver.close()


def test_rule_c_a_partial_release_is_refused_by_name_rather_than_read_stale(
        monkeypatch):
    """The backstop: a regressed release leaves the P mappings bound elsewhere.

    The release below is the pre-2026-09-27 one -- the ``Fields`` class swap only.
    Re-armed on top of it, the hold installed no dict barrier of its own and read
    stale P (1239 of 4800 words on four arrays at 200 + 200 steps). Now the freeze
    refuses the hold by name and the run finishes on the array path, identical.
    """
    steps = 60
    held, ref = _held_then_reference(monkeypatch, "dispersive_2d", steps)
    try:
        _assert_held(held)
        residency = held._fast_path.residency  # noqa: SLF001
        residency.flush()
        residency.release_hold()
        host_writes.unregister(residency)
        barrier.remove_read_barrier(held.fields)
        del residency
        held.invalidate_fast_path()
        ref.invalidate_fast_path()
        _switches(monkeypatch, dispatch=True, mode="held")
        held.run(num_steps=steps)
        _switches(monkeypatch, dispatch=False)
        ref.run(num_steps=steps)
        report = held.fast_path_report() or {}
        assert held.active_step_path == "array"
        reason = report["refused_because"]
        assert "polarizations[0].P" in reason and "released only in part" in reason, reason
        assert report["residency"]["admission"]["foreign_polarization_barriers"]
        verdict = _identical(held, ref, "dispersive_2d")
        assert _max_abs_p(e2e.collect_state(ref)) > 0.0
        say(f"rule (c): refused by name, identical over {verdict['arrays']} arrays")
    finally:
        held.close()
        ref.close()


@pytest.mark.parametrize("tail", ["split_pair", "no_slot_left"])
def test_a_tail_refusal_after_the_hold_was_armed_hands_the_host_its_arrays_back(
        monkeypatch, tail):
    """``fastpath._finish`` can refuse AFTER this ladder armed the hold.

    Both tail refusals come after the mirrors were sealed and the barriers
    installed. Before the release in ``decide`` the run continued on the array path
    with 3 of 30 engine arrays read-only and 27 of 30 still owned by the dead plan's
    residency (``pml_2d``, forced split-pair refusal) until the next Metal freeze.
    """
    if tail == "split_pair":
        monkeypatch.setattr(fastpath, "_split_pairs",
                            lambda plans, slots, order: ["forced split-pair refusal"])
    else:
        real_finish = fastpath._finish  # noqa: SLF001

        def emptied(**kwargs):
            kwargs["dispatchable"] = set()
            return real_finish(**kwargs)

        monkeypatch.setattr(fastpath, "_finish", emptied)
    held, ref = _held_then_reference(monkeypatch, "pml_2d", STEPS["pml_2d"])
    try:
        report = held.fast_path_report() or fastpath.last_dispatch_report() or {}
        assert held.active_step_path == "array", report
        block = report["residency"]
        assert block["mode"] == metal_dispatch.RESIDENCY_HELD
        assert len(block["held"]) > 0, "the hold must have been armed for this to test"
        assert _engine_read_only(held) == []
        arrays = metal_dispatch._engine_arrays_raw(held.fields)  # noqa: SLF001
        assert [label for label, array in arrays
                if host_writes._owner(array) is not None] == []  # noqa: SLF001
        assert barrier._BARRIER_RESIDENCY not in vars(held.fields)  # noqa: SLF001
        assert "released_after_refusal" in block, sorted(block)
        verdict = _identical(held, ref, "pml_2d")
        say(f"tail refusal {tail}: released {len(block['released_after_refusal'])} "
            f"names, 0 read-only, identical over {verdict['arrays']} arrays")
    finally:
        held.close()
        ref.close()


# ---------------------------------------------------------------------------
# 5. Every public driver mutator: identical under a hold, or refused identically
# ---------------------------------------------------------------------------

def _shape(driver):
    return tuple(int(v) for v in driver.shape)


#: One call per public in-place mutator of the driver. The enumeration test fails
#: on any mutator missing here, so a new one is a failing row, not a silent gap.
DRIVER_MUTATORS = {
    "set_epsilon": _set_epsilon,
    "set_epsilon_components": lambda d: d.set_epsilon_components(
        {c: numpy.full(_shape(d), 1.5, dtype=numpy.float32) for c in ("Ex", "Ey", "Ez")}),
    "set_epsilon_smoothed": lambda d: d.set_epsilon_smoothed(
        lambda x, y, z: 2.0 + 0.0 * x, component="Ez"),
    "add_susceptibility": lambda d: d.add_susceptibility(
        Susceptibility(frequency=0.4, gamma=0.02),
        numpy.full(_shape(d), 0.1, dtype=numpy.float32)),
    "set_conductivity": lambda d: d.set_conductivity(
        numpy.full(_shape(d), 0.01, dtype=numpy.float32)),
    "set_b_conductivity": lambda d: d.set_b_conductivity(
        numpy.full(_shape(d), 0.01, dtype=numpy.float32)),
    "set_absorber": lambda d: d.set_absorber([AbsorberLayer(thickness=1.0, axis=0,
                                                            side="low")]),
    "set_chi2": lambda d: d.set_chi2(1e-4),
    "set_chi3": lambda d: d.set_chi3(1e-4),
    "setup_pml": lambda d: d.setup_pml(10),
    "add_source": _add_source,
    "add_dft_monitor": lambda d: d.add_dft_monitor(
        frequency=0.25, components=("Ez",), center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.0)),
    "add_yee_region_dft": lambda d: d.add_yee_region_dft(
        "Ez", ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.0)), frequencies=0.25),
    "add_energy_monitor": lambda d: d.add_energy_monitor(
        frequency=0.25, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.0)),
    "add_force_monitor": lambda d: d.add_force_monitor(
        frequency=0.25, center=(0.0, 0.0, 0.0), size=(0.0, 1.0, 0.0), force_direction=0,
        normal=0),
    "add_flux_monitor": lambda d: d.add_flux_monitor(
        frequency=0.25, center=(0.0, 0.0, 0.0), size=(0.0, 1.0, 0.0)),
    "set_field": _set_field,
    "reset": lambda d: d.reset(),
    "invalidate_fast_path": lambda d: d.invalidate_fast_path(),
}


def _public_driver_mutators():
    return sorted(
        name for name, value in vars(FdtdDriver).items()
        if callable(value) and not name.startswith("_")
        and (name.startswith(("set_", "add_", "setup_"))
             or name in ("reset", "invalidate_fast_path")))


def test_the_mutator_table_covers_every_public_driver_mutator():
    assert sorted(DRIVER_MUTATORS) == _public_driver_mutators()


def _attempt(call, returned=None):
    try:
        value = call()
        if returned is not None:
            returned.append(value)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - the outcome is the measurement
        return f"{type(exc).__name__}: {exc}"


def _accumulators(monitor) -> dict:
    """Every array a mid-run monitor holds, as host copies keyed by path.

    Walked over the instance ``__dict__`` rather than the dataclass fields, because
    the force and Yee-region monitors keep their accumulators in attributes set
    after construction (``_accumulators``, ``_dft``) that a field walk never sees.
    """
    from meep_gpu import to_numpy  # noqa: PLC0415

    found: dict = {}

    def walk(value, path, depth):
        if depth > 5 or value is None:
            return
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            found.setdefault(path, value)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth + 1)
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                walk(value[key], f"{path}[{key!r}]", depth + 1)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            for name in sorted(vars(value)):
                if name != "grid":
                    walk(vars(value)[name], f"{path}.{name}", depth + 1)

    walk(monitor, "monitor", 0)
    return {name: numpy.ascontiguousarray(to_numpy(array)).copy()
            for name, array in found.items()}


@pytest.mark.parametrize("name", sorted(DRIVER_MUTATORS))
def test_every_public_driver_mutator_works_under_a_hold_or_is_refused_identically(
        monkeypatch, name):
    steps = STEPS["pml_2d"]
    held, ref = _held_then_reference(monkeypatch, "pml_2d", steps)
    try:
        _assert_held(held)
        held_returned: list = []
        ref_returned: list = []
        held_at = _attempt(lambda: DRIVER_MUTATORS[name](held), held_returned)
        ref_at = _attempt(lambda: DRIVER_MUTATORS[name](ref), ref_returned)
        if ref_at != "ok":
            assert held_at == ref_at, (held_at, ref_at)
            say(f"{name}: refused by the API on both paths ({ref_at.split(':')[0]})")
            return
        assert held_at == "ok", held_at
        _switches(monkeypatch, dispatch=True, mode="held")
        held_next = _attempt(lambda: held.run(num_steps=steps))
        _switches(monkeypatch, dispatch=False)
        ref.run(num_steps=steps)
        assert held_next == "ok", held_next
        block = _assert_held(held)
        verdict = _identical(held, ref, "pml_2d")
        monitor_arrays = 0
        if name.endswith(("_monitor", "_dft")):
            accumulated = e2e.compare_state(_accumulators(held_returned[0]),
                                            _accumulators(ref_returned[0]))
            assert accumulated["identical"], accumulated["differences"][:4]
            assert accumulated["arrays"] > 0, "the monitor accumulated nothing to compare"
            monitor_arrays = accumulated["arrays"]
        say(f"{name}: identical over {verdict['arrays']} arrays"
            f"{f' and {monitor_arrays} monitor arrays' if monitor_arrays else ''}, held "
            f"{len(block['held'])} after")
    finally:
        held.close()
        ref.close()
