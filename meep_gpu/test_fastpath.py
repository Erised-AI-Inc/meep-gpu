"""The fast-path seam ON THIS HOST: the plan cache, the kill switch, invalidation.

Everything here runs on the NumPy laptop, which is one of the dispatch contract's
own cases and not an approximation of another: the array path IS the reference
for a NumPy engine with no kernel table behind it, and ``plan_fast_path`` refuses
at the backend rung of its ladder without importing ``cupy`` at all. So every
driver here steps the array path, and these tests pin that together with the
machinery every certification leans on — that each material mutator invalidates
the cached plan, and that the driver re-plans exactly once per configuration
freeze.

"NO KERNEL TABLE BEHIND IT" IS A CONDITION NOW, NOT A PROPERTY OF NUMPY. Since
the seam grew a second table, a NumPy engine on a host with an MPS device is a
candidate for the Metal table, so the tests about the backend rung answer the
hardware probe themselves through :func:`no_metal_device` rather than inheriting
whichever machine ran them. The ones about the plan cache pin
``MEEP_GPU_DISPATCH=0`` themselves (:func:`opted_out`), which refuses a rung
earlier than either table is consulted. An unset enable no longer does: dispatch
is on by default, so on a Mac an unpinned NumPy step would plan through the Metal
table.

The DECISIONS above that rung — the fail-closed ladder, completeness, the null
drop, the warm pass, the artifact and the seven driver consults — are
``test_dispatch_contract.py``. The numerical certifications the seam rides on are
the device gates recorded in ``triton_kernels/fingerprints.json``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import meep_gpu.driver as driver_module
import meep_gpu.fastpath as fastpath_module
from meep_gpu.dispersion import Susceptibility
from meep_gpu.driver import FdtdDriver
from meep_gpu.fastpath import (
    FUSED_KILL_SWITCH,
    FastPathPlan,
    fused_dispatch_enabled,
    plan_fast_path,
)


def no_metal_device(monkeypatch) -> None:
    """Answer the Metal hardware probe "no", so the BACKEND rung is what refuses.

    THIS HOST HAS AN MPS DEVICE, and since the dispatch seam grew a second table
    a NumPy engine here is a *candidate* for the Metal table rather than an
    automatic refusal. The tests below are about the backend rung — "the array
    path IS the reference on a plain NumPy engine" — so they hold the hardware
    answer still instead of inheriting whichever laptop ran them. Without it a
    Mac would reach the Metal table (the enable is on by default) and plan kernels
    where the test means the backend rung's refusal, and the same green would mean
    two different things on two machines.

    THE ENABLE IS LEFT AT ITS DEFAULT, not merely inherited, so the backend rung is
    the one that answers: a shell or fixture that exported ``MEEP_GPU_DISPATCH=0``
    would otherwise have these tests pinning the enable rung while reading as if
    they had pinned the backend one.

    ``raising=False`` because the rung-3 branch that defines
    ``fastpath.metal_hardware_present`` lands in its own batch: before it, there
    is nothing to hold still and the refusal is unconditional anyway.
    """
    monkeypatch.setattr(fastpath_module, "metal_hardware_present",
                        lambda: False, raising=False)
    monkeypatch.delenv(fastpath_module.DISPATCH_ENABLE, raising=False)


def opted_out(monkeypatch) -> None:
    """Pin ``MEEP_GPU_DISPATCH=0``, so the ENABLE rung refuses on every host.

    For the tests about the plan cache and invalidation, which count plan builds
    and read ``active_step_path`` and do not care which rung refused. Before
    dispatch turned on by default an unset enable did this; now an unset enable on
    a Mac plans through the Metal table, compiles shaders and installs a subnormal
    policy for the whole process.
    """
    monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, "0")


def build_driver(gpu=None, **kwargs) -> FdtdDriver:
    """A minimal 4-cell-per-axis driver on the NumPy backend, epsilon installed.

    ``gpu="metal"`` makes it an Apple GPU driver's engine without the hardware, so
    ``step()`` consults the planner (a reference driver never does).
    """
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8, **kwargs)
    driver.gpu = gpu
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    return driver


class _StubBackend:
    """Stands in for a non-NumPy xp module without importing anything real."""

    __name__ = "not_numpy"


class _StubGrid:
    def __init__(self, xp):
        self.xp = xp


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def test_plan_is_none_on_the_numpy_backend(monkeypatch):
    no_metal_device(monkeypatch)
    driver = build_driver()
    assert plan_fast_path(driver.fields, driver.pml, driver.grid) is None
    reason = fastpath_module.last_dispatch_report()["refused_because"]
    assert "backend is not CuPy" in reason, reason


def test_plan_is_none_for_every_configuration_on_this_host(monkeypatch):
    """No configuration dispatches on a plain NumPy backend, covered or not.

    Not a coverage statement: these two configurations are inside several
    certified families. The refusal is the backend rung of the ladder, and it is
    the correct answer — the array path is the reference implementation for a
    NumPy engine with no kernel table behind it.
    """
    no_metal_device(monkeypatch)
    core = build_driver(force_complex_fields=True)
    assert plan_fast_path(core.fields, core.pml, core.grid) is None
    with_pml = build_driver()
    with_pml.setup_pml(1)
    assert plan_fast_path(with_pml.fields, with_pml.pml, with_pml.grid) is None


def test_plan_is_none_off_the_numpy_backend_without_importing_cupy(monkeypatch):
    """Fail-closed even for an unknown backend, and decided without cupy.

    ``plan_fast_path`` may only touch cupy AFTER deciding the backend is CuPy;
    a stub backend proves the Phase 0 predicate never gets that far.

    THE ENABLE AND THE KILL SWITCH ARE CLEARED, so the BACKEND rung is the one that
    answers. Under the suite's ``MEEP_GPU_DISPATCH=0`` pin rung 2 refused first and
    this test passed without the backend rung ever running — it would have held
    with that rung deleted.
    """
    monkeypatch.delenv(fastpath_module.DISPATCH_ENABLE, raising=False)
    monkeypatch.delenv(FUSED_KILL_SWITCH, raising=False)
    grid = _StubGrid(_StubBackend())
    before = {name for name in sys.modules if name.partition(".")[0] == "cupy"}
    assert plan_fast_path(None, None, grid) is None
    after = {name for name in sys.modules if name.partition(".")[0] == "cupy"}
    assert after == before, f"plan_fast_path imported cupy modules: {sorted(after - before)}"
    reason = fastpath_module.last_dispatch_report()["refused_because"]
    assert "backend is not CuPy" in reason, reason


def test_kill_switch_forces_the_array_path(monkeypatch):
    monkeypatch.setenv(FUSED_KILL_SWITCH, "0")
    assert not fused_dispatch_enabled()
    # Even a backend the predicate would otherwise inspect is refused first.
    assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
    record = fastpath_module.last_dispatch_report()
    assert record["refused_because"] == f"kill switch {FUSED_KILL_SWITCH}=0"
    assert record["environment"] == {"not_read": "refused at the kill switch"}


@pytest.mark.parametrize("value", [None, "1"])
def test_the_kill_switch_passes_on_one_or_unset(monkeypatch, value):
    """``1`` and unset are "not vetoed": dispatch is left to the enable."""
    from meep_gpu.fastpath import FUSED_KILL_SWITCH_VALUES, _kill_switch_value_refusal

    assert dict(FUSED_KILL_SWITCH_VALUES) == {"1": True, "0": False}
    if value is None:
        monkeypatch.delenv(FUSED_KILL_SWITCH, raising=False)
    else:
        monkeypatch.setenv(FUSED_KILL_SWITCH, value)
    assert fused_dispatch_enabled() is True
    assert _kill_switch_value_refusal(value) is None


#: Values a permissive reader would have guessed at. Before the strict reader every
#: one of these but the two padded zeros LEFT DISPATCH ON (the reader stripped, and
#: only ``0`` vetoed), including the ones a user types to mean "off". The padded
#: zeros vetoed then and are refused now, which lands on the array path either way.
REJECTED_KILL_SWITCH_VALUES = ["", " ", "yes", "no", "true", "false", "off", "on",
                               " 0", "0 ", "00", "2"]


@pytest.mark.parametrize("value", REJECTED_KILL_SWITCH_VALUES)
def test_an_unrecognised_kill_switch_value_is_refused_by_name_at_rung_one(
        monkeypatch, capsys, value):
    """Refused BY NAME at rung 1, on the array path, and announced.

    The non-raising reader reads it as vetoed (the run takes the array path), and
    rung 1 answers the unrecognised value BEFORE the ``0`` test, so the reason names
    the value the run carried rather than calling it ``MEEP_GPU_FUSED=0``. The
    enable is left at its default so nothing but the kill switch could refuse.
    """
    from meep_gpu.fastpath import _kill_switch_value_refusal

    monkeypatch.delenv(fastpath_module.DISPATCH_ENABLE, raising=False)
    monkeypatch.setenv(FUSED_KILL_SWITCH, value)
    assert fused_dispatch_enabled() is False
    message = _kill_switch_value_refusal(value)
    assert f"{FUSED_KILL_SWITCH}={value!r} is not an accepted value" in message, message
    fastpath_module.reset_dispatch_announcements()
    try:
        assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
        record = fastpath_module.last_dispatch_report()
        assert record["decision"] == "refused" and record["step_path"] == "array"
        assert record["refused_because"] == message, record["refused_because"]
        assert record["environment"] == {
            "not_read": "refused at an unrecognised kill-switch value"}
        assert record["kill_switch"]["value"] == value
        assert record["kill_switch"]["vetoes"] is True
        err = capsys.readouterr().err
        assert "dispatch refused" in err and message in err, err
    finally:
        fastpath_module.reset_dispatch_announcements()


def test_the_kill_switch_names_a_bad_enable_value_it_answered_before(
        monkeypatch, capsys):
    """``MEEP_GPU_FUSED=0`` answers at rung 1, and a bad enable value is still NAMED.

    Rung 1 never reads the enable, so without this the value would ride unseen
    until the kill switch was lifted and then refuse the run. A ``0`` kill switch
    beside an accepted enable value stays quiet, as an opted-out run does.
    """
    monkeypatch.setenv(FUSED_KILL_SWITCH, "0")
    fastpath_module.reset_dispatch_announcements()
    try:
        monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, "1")
        assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
        assert capsys.readouterr().err == ""

        monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, "true")
        assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
        record = fastpath_module.last_dispatch_report()
        assert record["refused_because"] == f"kill switch {FUSED_KILL_SWITCH}=0"
        err = capsys.readouterr().err
        assert f"{FUSED_KILL_SWITCH}=0" in err, err
        assert f"{fastpath_module.DISPATCH_ENABLE}='true' is not an accepted value" in err, err
    finally:
        fastpath_module.reset_dispatch_announcements()


def test_a_host_with_no_kernel_table_says_so_once_per_process(monkeypatch, capsys):
    """A GPU driver's backend-rung refusal is ANNOUNCED, once per process.

    With dispatch on by default, a GPU driver whose NumPy engine finds no MPS device
    at rung 3 is refused there, and the line says the run is on the array path ON THE
    HOST CPU and why. (The drivers here are made GPU drivers' engines by hand,
    ``gpu="metal"``, with the device stubbed absent: a real ``prefer_gpu=True`` on
    such a host raises at construction, and a ``prefer_gpu=False`` driver is the
    NumPy reference, which never plans and prints nothing on a CPU-only host.) It is
    the same line for every driver in the process, so it prints once; a process that
    steps a hundred drivers must not print a hundred.
    """
    no_metal_device(monkeypatch)
    monkeypatch.delenv(FUSED_KILL_SWITCH, raising=False)
    fastpath_module.reset_dispatch_announcements()
    try:
        for _ in range(3):
            driver = build_driver(gpu="metal")  # a GPU driver: the reference never plans
            driver.run(num_steps=1)
            assert driver.active_step_path == "array"
            driver.close()
        err = capsys.readouterr().err
        lines = [line for line in err.splitlines()
                 if line.startswith("meep_gpu: step path array on the host CPU; "
                                    "dispatch refused:")]
        assert len(lines) == 1, err
        assert "backend is not CuPy" in lines[0], lines[0]
        assert "no MPS device is available on this host" in lines[0], lines[0]

        # And an opted-out GPU driver on the same host says nothing at all: rung 2
        # refuses it, and a refusal the run chose is quiet.
        fastpath_module.reset_dispatch_announcements()
        monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, "0")
        driver = build_driver(gpu="metal")
        driver.run(num_steps=1)
        assert driver.fast_path_report()["refused_because"] == (
            f"dispatch disabled by {fastpath_module.DISPATCH_ENABLE}=0")
        driver.close()
        assert "meep_gpu:" not in capsys.readouterr().err
    finally:
        fastpath_module.reset_dispatch_announcements()


def test_plan_describe_names_what_it_dispatches_and_which_arm_won():
    """The arm label is the only seam that distinguishes the products.

    Three ``update_H`` arms — ordinary, real beta run, BFAST run — all build a
    ``ConstitutivePlan``, so a description naming the class would be reporting an
    ambiguity as a decision.
    """
    empty = FastPathPlan(fields=None, step_plan=None, slots=(), arms={},
                         dropped_null={}, unwarmed={}, record={})
    assert "nothing" in empty.describe()
    assert empty.dispatches_a_kernel is False
    plan = FastPathPlan(fields=None, step_plan=None, slots=("step_B", "step_D"),
                        arms={"step_B": "PML", "step_D": "conductive PML"},
                        dropped_null={}, unwarmed={}, record={})
    assert "step_B, step_D" in plan.describe()
    assert "PML, conductive PML" in plan.describe() or "conductive PML, PML" in plan.describe()
    assert plan.dispatches_a_kernel is True


# ---------------------------------------------------------------------------
# The driver seam: active_step_path
# ---------------------------------------------------------------------------


def test_the_enable_defaults_on_and_accepts_only_zero_and_one(monkeypatch):
    """Unset is "on": the default is a claim about evidence, so it is a constant.

    ``DISPATCH_BY_DEFAULT`` flipped once ``gate_dispatch_end_to_end``'s ship leg —
    two drivers byte-compared across complete ``step()`` calls, the harness
    installing no policy and the enable left unset — released on the shipping
    bytes; ``test_dispatch_contract`` binds the constant to that licence. ``0`` is
    the opt-out and ``1`` the explicit opt-in; every other value is refused, which
    :func:`test_an_unrecognised_enable_value_is_refused_by_name` pins.
    """
    from meep_gpu.fastpath import (
        DISPATCH_BY_DEFAULT,
        DISPATCH_ENABLE,
        DISPATCH_ENABLE_VALUES,
        dispatch_enabled,
    )

    assert DISPATCH_BY_DEFAULT is True
    assert dict(DISPATCH_ENABLE_VALUES) == {"1": True, "0": False}
    monkeypatch.delenv(DISPATCH_ENABLE, raising=False)
    assert dispatch_enabled() is DISPATCH_BY_DEFAULT
    monkeypatch.setenv(DISPATCH_ENABLE, "1")
    assert dispatch_enabled() is True
    monkeypatch.setenv(DISPATCH_ENABLE, "0")
    assert dispatch_enabled() is False


#: Values a permissive reader would have guessed at. Before the strict parser every
#: one of these but the empty and whitespace-only values ENABLED dispatch, and those
#: two took the default.
REJECTED_ENABLE_VALUES = ["", " ", "\t", "true", "false", "True", "off", "on",
                          "no", "yes", " 1", "1 ", " 0", "0 ", "01", "2", "-1"]


@pytest.mark.parametrize("value", REJECTED_ENABLE_VALUES)
def test_an_unrecognised_enable_value_is_refused_by_name(monkeypatch, value):
    """The raising reader names the variable, the value and the accepted values.

    Raised as ``DispatchEnableValueError``, a ``RuntimeError``, which is the
    retired-name convention: rung 2 catches it and refuses by name, and every
    other caller of ``dispatch_enabled`` hears it loudly. The non-raising reader
    the record uses answers ``False``, which is what the run gets.
    """
    from meep_gpu.fastpath import (
        DISPATCH_ENABLE,
        DispatchEnableValueError,
        _enable_as_set,
        dispatch_enabled,
    )

    monkeypatch.setenv(DISPATCH_ENABLE, value)
    with pytest.raises(DispatchEnableValueError) as raised:
        dispatch_enabled()
    assert isinstance(raised.value, RuntimeError)
    message = str(raised.value)
    assert f"{DISPATCH_ENABLE}={value!r} is not an accepted value" in message, message
    assert f"{DISPATCH_ENABLE}=1" in message and f"{DISPATCH_ENABLE}=0" in message, message
    assert "unset" in message, message
    assert _enable_as_set() is False


@pytest.mark.parametrize("value", REJECTED_ENABLE_VALUES)
def test_the_ladder_refuses_an_unrecognised_enable_value_at_rung_two(monkeypatch, value):
    """Refused by name, on the array path, before any table is consulted.

    A stub backend is enough: rung 2 answers before rung 3 reads the engine, and
    the record says which rule answered rather than calling it a retired name or
    an internal error.
    """
    monkeypatch.delenv(FUSED_KILL_SWITCH, raising=False)
    monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, value)
    assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
    record = fastpath_module.last_dispatch_report()
    assert record["decision"] == "refused" and record["step_path"] == "array"
    reason = record["refused_because"]
    assert (f"{fastpath_module.DISPATCH_ENABLE}={value!r} is not an accepted value"
            in reason), reason
    assert "internal error" not in reason, reason
    assert record["environment"] == {
        "not_read": "refused at an unrecognised enable value"}, record["environment"]
    assert record["enable"]["value"] == value
    assert record["enable"]["effective"] is False


def test_an_unrecognised_enable_value_is_announced_not_swallowed(monkeypatch, capsys):
    """The stderr line stays quiet for an opt-out and speaks for a refused value.

    The announcement is skipped when the enable reads ineffective, because an
    opted-out run does not need telling. An unrecognised value also reads
    ineffective, and a run whose ``MEEP_GPU_DISPATCH=true`` was refused DOES need
    telling — it asked for something and got the array path.
    """
    monkeypatch.delenv(FUSED_KILL_SWITCH, raising=False)
    fastpath_module.reset_dispatch_announcements()
    try:
        monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, "0")
        assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
        assert "dispatch refused" not in capsys.readouterr().err

        monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, "true")
        assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
        err = capsys.readouterr().err
        assert "dispatch refused" in err, err
        assert "MEEP_GPU_DISPATCH='true' is not an accepted value" in err, err
    finally:
        fastpath_module.reset_dispatch_announcements()


def test_zero_disables_and_unset_enables_at_the_ladder(monkeypatch):
    """The two accepted readings, measured at rung 2 rather than at the reader.

    ``0`` refuses at the enable with the one refusal that names it; unset passes
    the enable, so the stub backend is refused one rung LOWER, at the backend rung.
    """
    monkeypatch.delenv(FUSED_KILL_SWITCH, raising=False)
    monkeypatch.setenv(fastpath_module.DISPATCH_ENABLE, "0")
    assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
    record = fastpath_module.last_dispatch_report()
    assert record["refused_because"] == (
        f"dispatch disabled by {fastpath_module.DISPATCH_ENABLE}=0")
    assert record["environment"] == {"not_read": "refused at the enable"}

    monkeypatch.delenv(fastpath_module.DISPATCH_ENABLE, raising=False)
    assert plan_fast_path(None, None, _StubGrid(_StubBackend())) is None
    record = fastpath_module.last_dispatch_report()
    assert record["enable"]["value"] is None
    assert record["enable"]["effective"] is True
    assert fastpath_module.DISPATCH_ENABLE not in record["refused_because"], (
        record["refused_because"])
    assert record["environment"].get("not_read") not in (
        "refused at the enable", "refused at an unrecognised enable value"), (
        record["environment"])


def test_active_step_path_is_array_without_a_kernel_table(monkeypatch):
    """A GPU driver whose host offers no table is planned to the array path.

    ``gpu="metal"`` with the device stubbed absent, so ``step()`` reaches the planner
    and rung 3 refuses. A reference driver reads "array" too, but by construction and
    without planning; ``test_prefer_gpu.py`` holds that contract.
    """
    no_metal_device(monkeypatch)
    driver = build_driver(gpu="metal")
    assert driver.active_step_path == "array"  # Before the first step: no plan built.
    driver.step()
    assert driver.active_step_path == "array"  # After the freeze: planned to None.
    record = driver.fast_path_report()
    assert record["reference_driver"] is False
    assert "backend is not CuPy" in record["refused_because"], record["refused_because"]
    driver.run(num_steps=3)
    assert driver.active_step_path == "array"


def test_active_step_path_is_array_after_each_material_mutator(monkeypatch):
    """Every mutator leaves the run on the array path — exercised through a step.

    On GPU drivers' engines (``gpu="metal"``, the device stubbed absent), so each
    freeze and each re-freeze after a mutator goes through the planner and is refused
    at rung 3, which is what this test has always exercised.
    """
    no_metal_device(monkeypatch)

    def refused_at_the_backend_rung(driver) -> bool:
        record = driver.fast_path_report()
        return (record["reference_driver"] is False
                and "backend is not CuPy" in record["refused_because"])

    mid_run_mutations = {
        "set_epsilon": lambda d: d.set_epsilon(np.full(d.shape, 2.0, dtype=np.float32)),
        "set_epsilon_components": lambda d: d.set_epsilon_components(
            {c: np.full(d.shape, 1.5, dtype=np.float32) for c in ("Ex", "Ey", "Ez")}
        ),
        "set_epsilon_smoothed": lambda d: d.set_epsilon_smoothed(
            lambda x, y, z: 2.0 + 0.0 * x, component="Ez"
        ),
        "set_field": lambda d: d.set_field(
            "Dz", np.zeros(d.shape, dtype=np.complex64)
        ),
        "reset": lambda d: d.reset(),
    }
    for name, mutate in mid_run_mutations.items():
        driver = build_driver(gpu="metal")
        driver.step()
        mutate(driver)
        driver.step()
        assert driver.active_step_path == "array", f"{name} left path {driver.active_step_path}"
        assert refused_at_the_backend_rung(driver), (name, driver.fast_path_report())

    before_first_step_mutations = {
        "setup_pml": lambda d: d.setup_pml(1),
        "set_conductivity": lambda d: d.set_conductivity(0.25),
        "set_b_conductivity": lambda d: d.set_b_conductivity(0.25),
        "set_chi2": lambda d: d.set_chi2(1e-4),
        "set_chi3": lambda d: d.set_chi3(1e-4),
        "add_susceptibility": lambda d: d.add_susceptibility(
            Susceptibility(frequency=1.1, gamma=1e-5), 0.5
        ),
    }
    for name, mutate in before_first_step_mutations.items():
        driver = build_driver(gpu="metal")
        mutate(driver)
        driver.step()
        assert driver.active_step_path == "array", f"{name} left path {driver.active_step_path}"
        assert refused_at_the_backend_rung(driver), (name, driver.fast_path_report())


# ---------------------------------------------------------------------------
# The invalidation seam
# ---------------------------------------------------------------------------


def _count_plans(monkeypatch):
    """Route the driver's plan builds through a counter, preserving behavior."""
    calls = {"count": 0}

    def counting_plan(fields, pml, grid, sources=None):
        calls["count"] += 1
        # Forwarded rather than dropped: the driver hands its source list to the
        # freeze, and a stub that swallowed it would plan a different configuration
        # from the one under test.
        return plan_fast_path(fields, pml, grid, sources)

    monkeypatch.setattr(driver_module, "plan_fast_path", counting_plan)
    return calls


def test_the_plan_is_built_once_per_configuration_freeze(monkeypatch):
    opted_out(monkeypatch)
    calls = _count_plans(monkeypatch)
    driver = build_driver(gpu="metal")
    assert calls["count"] == 0  # Construction plans nothing; the freeze is the first step.
    driver.run(num_steps=4)
    assert calls["count"] == 1, "one freeze must plan exactly once, not per step"


def test_mid_run_epsilon_change_invalidates_and_replans(monkeypatch):
    """The covering seam test the directive names: mutate epsilon MID-RUN.

    Plain ``set_epsilon`` is legal after stepping has begun and REPLACES the
    material arrays, so a plan cached at the first freeze would keep stale
    pointers. The fused-vs-array numerical comparison across the change is a
    Phase 1+ CUDA test (no kernel exists to compare); what must hold from
    Phase 0 on is the mechanism: mutation -> invalidation -> exactly one re-plan.
    """
    opted_out(monkeypatch)
    calls = _count_plans(monkeypatch)
    driver = build_driver(gpu="metal")
    driver.run(num_steps=2)
    assert calls["count"] == 1
    driver.set_epsilon(np.full(driver.shape, 4.0, dtype=np.float32))
    assert driver._fast_path_stale, "set_epsilon must invalidate the cached plan"
    driver.run(num_steps=2)
    assert calls["count"] == 2, "the invalidated plan must be rebuilt at the next step"
    driver.run(num_steps=2)
    assert calls["count"] == 2, "no further mutation, no further re-plan"


def test_every_material_mutator_invalidates_the_cached_plan():
    """Each mutator flips the stale flag — the greppable contract of the one funnel.

    Mutators that are refused after the first step are exercised against a fresh
    driver whose flag is manually cleared, which isolates the invalidation call
    itself from the step-count guards around it.
    """
    mutations = {
        "set_epsilon": lambda d: d.set_epsilon(np.full(d.shape, 2.0, dtype=np.float32)),
        "set_epsilon_components": lambda d: d.set_epsilon_components(
            {c: np.full(d.shape, 1.5, dtype=np.float32) for c in ("Ex", "Ey", "Ez")}
        ),
        "set_epsilon_smoothed": lambda d: d.set_epsilon_smoothed(
            lambda x, y, z: 2.0 + 0.0 * x, component="Ez"
        ),
        "add_susceptibility": lambda d: d.add_susceptibility(
            Susceptibility(frequency=1.1, gamma=1e-5), 0.5
        ),
        "set_conductivity": lambda d: d.set_conductivity(0.25),
        "set_conductivity_to_none": lambda d: d.set_conductivity(None),
        "set_b_conductivity": lambda d: d.set_b_conductivity(0.25),
        "set_b_conductivity_to_none": lambda d: d.set_b_conductivity(None),
        "set_chi2": lambda d: d.set_chi2(1e-4),
        "set_chi3": lambda d: d.set_chi3(1e-4),
        "setup_pml": lambda d: d.setup_pml(1),
        "set_field": lambda d: d.set_field("Dz", np.zeros(d.shape, dtype=np.complex64)),
        "reset": lambda d: d.reset(),
    }
    missed = []
    for name, mutate in mutations.items():
        driver = build_driver()
        driver._fast_path_stale = False  # As after a step(): the plan cache is warm.
        mutate(driver)
        if not driver._fast_path_stale:
            missed.append(name)
        driver.close()
    assert not missed, f"material mutators that did NOT invalidate the fast path: {missed}"


def test_close_drops_the_plan_state(monkeypatch):
    opted_out(monkeypatch)
    driver = build_driver(gpu="metal")  # a GPU driver: its freeze goes through the planner
    driver.step()
    driver.close()
    assert driver._fast_path is None
