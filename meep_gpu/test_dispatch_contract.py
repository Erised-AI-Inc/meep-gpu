"""The dispatch contract: what reaches Triton, what reaches the array path, and why.

The array path is the universal fallback and the oracle; Triton is an accelerated
SUBSET. So what these tests pin is not coverage — it is the three properties that
make a partial accelerator safe to wire into a driver:

* every degradation reaches the array path with a NAMED reason and none of them
  raises (the fail-closed ladder, C6);
* a step never runs partly on kernels and partly on the array path in a shape no
  composition probe certified (completeness, C3, and the null drop, C4);
* which of the two happened is legible from the run artifact alone (T2).

NOTHING HERE LAUNCHES A KERNEL. These run on the NumPy laptop, where the real
composer refuses every kernel arm at ``coverage._grid_reasons`` clause 1 — which
is itself one of the cases below. The dispatch DECISIONS are exercised against
plan objects that count their launches, so the seam is measured without a device;
the numerical certifications the seam rides on are the device gates recorded in
``triton_kernels/fingerprints.json``, and the composition contracts are
``test_triton_planner_composition``'s.

WHAT IS DELIBERATELY NOT HERE: any fused-vs-array numerical comparison through
the real driver seam. That is the driver-route gate (two drivers from one
builder, byte-compared as uint32 after every complete ``step()``), it needs a
GPU, and its end-to-end ship leg is the licence ``fastpath.DISPATCH_BY_DEFAULT``
needs on the shipped bytes — to be recorded as ``driver_dispatch.dispatch_by_default_licence``
(not yet run) and bound to the constant by
``test_the_recorded_dispatch_shape_matches_the_wiring_it_describes``.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import types

import numpy as np
import pytest

from conftest import requires_resource_skip

import meep_gpu.driver as driver_module
import meep_gpu.fastpath as fastpath
from meep_gpu.driver import FdtdDriver
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels.coverage import Coverage
from meep_gpu.triton_kernels.launch import TritonStepPlan
from . import weld_record_walk as walk
from .code_identity import code_digest
from .device_identity import weld_survives_edit


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _quiet_announcements():
    """Each test starts with the one-line stderr announcements unspoken."""
    fastpath.reset_dispatch_announcements()
    yield
    fastpath.reset_dispatch_announcements()


@pytest.fixture(autouse=True)
def _cold_policy_state(monkeypatch, tmp_path):
    """A cold subnormal policy around every test in this file.

    The gate at rung 8b INSTALLS one, and the install wraps seams on Triton's
    backend class and CuPy's compiler module — process state, which is the point
    of it. Leaving it behind would have one test's install decide the next test's
    verdict. The private cache directory is per test for the same reason, and so
    that nothing here touches the developer's own ``~/.cupy`` tree.
    """
    from meep_gpu import subnormal_policy

    subnormal_policy._reset_for_tests()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None
    monkeypatch.delenv(fastpath.SUBNORMAL_INSTALL_SWITCH, raising=False)
    monkeypatch.delenv(subnormal_policy.POLICY_ENV, raising=False)
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    yield
    subnormal_policy._reset_for_tests()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None


@pytest.fixture(autouse=True)
def _opted_in(monkeypatch):
    """Default the ladder's first two rungs to "not vetoed, opted in".

    Each rung has its own test; every OTHER test wants to reach the rungs below
    them, and leaving the enable to the environment would make this file's
    verdicts depend on the shell it ran in.
    """
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.delenv(fastpath.DISPATCH_LOG, raising=False)
    monkeypatch.delenv(fastpath.WARM_SWITCH, raising=False)


#: THE ARCHITECTURE THE SHIPPED TRITON LEDGER HAS LIVE RUNS FOR, and the one every
#: test below that asks for a RUN fact asks on. Each weld keeps one record per compute
#: capability under ``fastpath.RUNS``, so a lookup that names no capability quotes no
#: host, no ``recorded_utc`` and no per-family budget at all — deliberately, because
#: quoting one card's run beside a plan on another is the mismatch the container closes.
#: Spelled as the literal the ledger is keyed by rather than derived from
#: :func:`fastpath.capability_admission`, so a round that certifies a SECOND
#: architecture does not silently move which run these assertions read.
CERTIFIED_CAPABILITY = "8.6"


class StubBackend(types.SimpleNamespace):
    """Stands in for the ``cupy`` module without importing or needing it."""


def cupy_like_grid():
    xp = StubBackend()
    xp.__name__ = "cupy"
    xp.__version__ = "13.5.1"
    return types.SimpleNamespace(xp=xp)


def numpy_backed_cupy_alias():
    """A module NAMED cupy whose every function is NumPy's.

    The ladder's backend rung reads ``grid.xp.__name__``, so a driver that must
    reach the rungs BELOW it needs a backend that answers "cupy" and still steps
    the array path correctly. A ``SimpleNamespace`` cannot: the real sub-step calls
    would fail on it. This is the same module-typed alias the audit drove the
    driver seam with.
    """
    module = types.ModuleType("cupy")
    module.__dict__.update(np.__dict__)
    module.__name__ = "cupy"
    module.__version__ = "13.5.1"
    return module


def stub_triton(monkeypatch, version="3.1.0"):
    """A Triton the policy gate can actually be driven onto.

    IT USED TO BE A BARE ``types.ModuleType``, and that was enough only because
    the gate decided governance with ``find_spec`` and read its ``ValueError`` on
    a spec-less module as "this executor is absent" — so every test in this file
    ran with ``governed_executors == ['host']`` and a Triton that was present,
    certified at rung 4, and quietly ungoverned. The gate now governs what rungs 3
    and 4 established, so a stub that impersonates a certified Triton has to be
    drivable or the freeze correctly refuses. ``governable_triton`` is the double
    the policy-uniformity suite defines; nothing else about these tests changed.
    """
    from meep_gpu.test_dispatch_policy_uniformity import governable_triton

    governable_triton(monkeypatch, version)
    return sys.modules["triton"]


def block_triton(monkeypatch):
    """Make ``import triton`` fail the way a host without it does."""
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def refusing_import(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("no module named 'triton'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", refusing_import)


class CountingPlan:
    """A plan that counts its launches instead of performing one."""

    def __init__(self, name, warm_raises=None):
        self.name = name
        self.runs = 0
        self.warms = 0
        self.drives = []
        self._warm_raises = warm_raises

    def run(self, drive=None, guard=None):
        self.runs += 1
        if drive is not None:
            self.drives.append(drive)

    def warm(self):
        self.warms += 1
        if self._warm_raises is not None:
            raise self._warm_raises


def step_plan(plans=None, selected=None, reasons=None):
    """A ``TritonStepPlan`` with exactly the fills, arms and refusals asked for."""
    plans = dict(plans or {})
    selected = dict(selected or {})
    reasons = dict(reasons or {})
    for slot in launch_module.STEP_ORDER:
        if slot not in plans and slot not in reasons:
            reasons[slot] = (f"no consulted product admitted {slot}",)
    return TritonStepPlan(plans, [], reasons, selected)


def install_composer(monkeypatch, plan):
    """Route ``plan_fast_path``'s one composer call at a prepared plan."""
    import meep_gpu.triton_kernels as package

    calls = []

    def fake_plan_step(fields, pml, **kwargs):
        calls.append(kwargs)
        if isinstance(plan, Exception):
            raise plan
        return plan

    monkeypatch.setattr(package, "plan_step", fake_plan_step)
    return calls


def plan_on_a_cupy_host(monkeypatch, plan, fields=None, version="3.1.0",
                        grid=None, pml=None):
    stub_triton(monkeypatch, version)
    install_composer(monkeypatch, plan)
    return fastpath.plan_fast_path(fields or object(), pml or object(),
                                   grid or cupy_like_grid())


# ---------------------------------------------------------------------------
# The configuration the driver-route gate actually drove
# ---------------------------------------------------------------------------
#
# ``cupy_like_grid`` answers the backend rung and nothing else: it carries no
# ``dimensions``, no extent and no ``is_mirrored``, so ``_run_shape`` reads almost
# nothing off it and ``fused_release_reasons`` correctly refuses to apply the
# release to a configuration it cannot describe. That is the right default for
# every other test in this file and it is USELESS for measuring the release, which
# is why this triple exists: it reproduces, field by field, the shape the
# dispatching cases of ``dispatch_fused_route_2026-08-30_bind`` reported when
# they were lifted from real ``mp.Simulation`` objects — ``dimensions 2``,
# ``grid_shape (200, 120, 1)``, unfolded, real storage, no Bloch phase, PML live.


def certified_envelope_grid():
    grid = cupy_like_grid()
    grid.dimensions = 2
    grid.shape = (200, 120, 1)
    grid.cylindrical = False
    grid.is_mirrored = lambda axis: False
    grid.has_symmetry = False
    grid.has_bloch = False
    grid.k_point = (0.0, 0.0, 0.0)
    grid.beta = 0.0
    grid.bfast_active = False
    return grid


def certified_envelope_fields():
    return types.SimpleNamespace(
        force_complex_fields=False, has_conductivity=False,
        has_nonlinearity=False, has_offdiagonal_epsilon=False, polarizations=())


def certified_envelope_pml():
    return types.SimpleNamespace(is_active=True)


def inactive_envelope_pml():
    """The absorber-free triple, for the arms whose release row pins
    ``pml_active`` False.

    It exists because ``pml_active`` left the SHARED envelope in the 2026-09-14
    target round and became a per-arm row: before that move no released arm could
    be measured on an absorber-free grid, so every shape here could carry the one
    active PML. ``fused pair D (no-PML stored E)`` is the first arm released on the
    other value of that axis, and a denominator that could not express it would
    silently drop it from the parametrised release test.
    """
    return types.SimpleNamespace(is_active=False)


def folded_envelope_grid():
    """The gate's ``folded_2d`` shape: the same 2-D PML grid with a Y mirror.

    THE FOLDED ARMS ARE RELEASED ON A DIFFERENT SHAPE FROM THE ORDINARY ONES, which
    is the whole content of ``FUSED_RELEASE_ARM_AXES``, so measuring them needs a
    second triple. ``_fold_description`` reads the fold off ``is_mirrored``, exactly
    as rung (6b) does.
    """
    grid = certified_envelope_grid()
    grid.is_mirrored = lambda axis: axis == 1
    grid.has_symmetry = True
    return grid


def cylindrical_envelope_grid():
    """The gate's ``cylindrical_m0`` shape: the e2e Dcyl m = 0 grid, 2026-09-11.

    A Dcyl LIFT REPORTS ``dimensions 2``, and that is the whole reason
    ``cylindrical`` had to become a per-arm axis rather than being implied by
    dimensionality: on every axis the 2026-09-02 table carried, this shape and
    ``certified_envelope_grid`` agree. ``m`` is set because ``_run_shape`` records
    it on a cylindrical grid and a reader comparing an artifact against the gate
    needs it; no arm PINS it, because the shape ``dispatch_reachability`` derives
    from the census carries no ``m`` key and the ``cylindrical_real`` coverage
    predicates refuse ``m != 0`` themselves.
    """
    grid = certified_envelope_grid()
    grid.cylindrical = True
    grid.m = 0
    grid.shape = (80, 1, 80)
    return grid


def conductive_envelope_fields():
    """The gate's ``conductive_2d`` fields: a ``D_conductivity`` block, no poles.

    The conductive product is released on ``conductivity True`` and the ORDINARY
    ``fused pair D`` is pinned at False, so this is the read that separates them:
    on the certified grid with these fields the composer selects ``fused pair B``
    for the magnetic seam and the conductive product for the D seam.
    """
    fields = certified_envelope_fields()
    fields.has_conductivity = True
    return fields


def folded_dispersive_envelope_fields():
    """The gate's ``folded_dispersive_2d`` fields: one Lorentz pole, lossless.

    Paired with :func:`folded_envelope_grid`. The pole is what separates the
    folded dispersive product from ``fused pair D (folded)``, which pins
    ``susceptibilities`` at 0 precisely because this case gives its D seam away.

    ONE POLE AND NOT FIVE, and the count is load-bearing in both directions since
    the re-attribution round: ``fused pair D (folded dispersive)`` is released on
    this shape with its own row pinned at 1, while the route gate's Triton envelope
    witness is the same fold at FIVE poles, which every folded row refuses.
    """
    fields = certified_envelope_fields()
    fields.polarizations = (object(),)
    return fields


def offdiag_envelope_fields():
    """The gate's ``offdiag_magnetic_2d`` fields: an off-diagonal chi1inv, lossless.

    Paired with :func:`certified_envelope_grid` for ``fused pair D (off-diagonal)``
    and with :func:`folded_envelope_grid` for ``fused pair D (folded off-diagonal)``
    (``folded_offdiag_magnetic_2d``), the two scratch-output welds released
    2026-09-15. The flag is what separates them from ``fused pair D`` and ``fused
    pair D (folded)``, which pin ``off_diagonal_epsilon`` at False.
    """
    fields = certified_envelope_fields()
    fields.has_offdiagonal_epsilon = True
    return fields


def dict_grid(grid, **values):
    for name, value in values.items():
        setattr(grid, name, value)
    return grid


def dict_fields(fields, **values):
    for name, value in values.items():
        setattr(fields, name, value)
    return fields


def beta_envelope_grid():
    """The gate's ``special_kz_2d`` shape: the certified grid with a real special-kz beta."""
    return dict_grid(certified_envelope_grid(), beta=0.4)


def folded_beta_envelope_grid():
    """``folded_special_kz_2d``: the fold with a beta."""
    return dict_grid(folded_envelope_grid(), beta=0.4)


def bloch_envelope_grid():
    """``bloch_2d``: a Bloch phase along x (complex storage lives on the fields)."""
    return dict_grid(certified_envelope_grid(), has_bloch=True, k_point=(0.3, 0.0, 0.0))


def folded_bloch_envelope_grid():
    """``folded_complex_2d``: the fold under a Bloch phase."""
    return dict_grid(folded_envelope_grid(), has_bloch=True, k_point=(0.3, 0.0, 0.0))


def bfast_envelope_grid():
    """``bfast_1d``: refl-angular's z-only cell DECLARED 3-D (the lift reads 3), BFAST on."""
    return dict_grid(certified_envelope_grid(), dimensions=3, shape=(1, 1, 250),
                     bfast_active=True)


def nonlinear_envelope_grid():
    """``nonlinear_1d``: 3rd-harm-1d's 1-D grid."""
    return dict_grid(certified_envelope_grid(), dimensions=1, shape=(1, 1, 400))


def complex_envelope_fields():
    """Complex64 storage, everything else as the certified fields."""
    return dict_fields(certified_envelope_fields(), force_complex_fields=True)


def complex_no_pml_3d_envelope_grid():
    """``complex_no_pml_3d``: the (23, 21, 27) absorber cell under an oblique k_point.

    Paired with :func:`complex_conductive_envelope_fields` and
    :func:`inactive_envelope_pml`. The route builder's lift (2026-09-13) reads
    dimensions 3, bloch True, k_point (0.4, -1.3, 0.7) and beta 0.0, and those are
    the grid values set here.
    """
    return dict_grid(certified_envelope_grid(), dimensions=3, shape=(23, 21, 27),
                     has_bloch=True, k_point=(0.4, -1.3, 0.7))


def complex_conductive_envelope_fields():
    """``complex_no_pml_3d``'s fields: complex storage, a conductivity, ONE pole.

    ALL THREE ARE LOAD-BEARING. ``fused pair D (complex conductive no-PML)`` pins
    ``complex_storage`` True, ``conductivity`` True and ``susceptibilities`` 1, so
    dropping any one of them measures a refusal where the parametrised release test
    is asking about the release. The pole count is the case's one Lorentzian.
    """
    return dict_fields(complex_envelope_fields(), has_conductivity=True,
                       polarizations=(object(),))


def nonlinear_envelope_fields():
    return dict_fields(certified_envelope_fields(), has_nonlinearity=True)


def arms_released_on(fields=None, grid=None, pml=None):
    """What the shipped predicate admits on one shape — asked, never assumed."""
    return list(fastpath.released_fused_arms(fastpath._run_shape(
        fields or certified_envelope_fields(), pml or certified_envelope_pml(),
        grid or certified_envelope_grid())))


def plan_inside_the_released_envelope(monkeypatch, plan, folded=False,
                                      grid=None, fields=None, pml=None):
    """``plan_on_a_cupy_host`` on the configuration the gate drove.

    ``folded`` is the shorthand for the fold pair's grid and stays; ``grid`` and
    ``fields`` are what the 2026-09-11 arms need, because all three of the arms
    that round released are released on a shape that differs from the certified
    triple in something other than the fold (a Dcyl grid, a conductivity). The
    fourth shape that round drove, a folded DISPERSIVE grid, needs BOTH — the fold
    on the grid and the single Lorentz pole on the fields — and it is measured for
    a DISPATCH again: the arm it releases spent one round in
    ``PENDING_DEVICE_GATE_ARMS`` on a route-gate divergence at
    ``synchronize_magnetic_fields`` that was re-attributed to the subnormal-policy
    install ordering, and the shape now admits the folded magnetic pair beside the
    folded dispersive one.
    """
    return plan_on_a_cupy_host(
        monkeypatch, plan, fields=fields or certified_envelope_fields(),
        grid=grid or (folded_envelope_grid() if folded
                      else certified_envelope_grid()),
        pml=pml or certified_envelope_pml())


def numpy_driver(**kwargs) -> FdtdDriver:
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8, **kwargs)
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    return driver


# ---------------------------------------------------------------------------
# The negative matrix: every rung of the fail-closed ladder
# ---------------------------------------------------------------------------


def test_the_kill_switch_refuses_from_a_cold_process_and_names_itself(monkeypatch):
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                      {"step_B": "PML"}))
    assert plan is None
    record = fastpath.last_dispatch_report()
    assert record["step_path"] == "array"
    assert record["decision"] == "refused"
    assert "kill switch" in record["refused_because"]
    assert fastpath.FUSED_KILL_SWITCH in record["refused_because"]


def test_the_kill_switch_beats_the_enable(monkeypatch):
    """A veto that can never enable — which is why it is a separate variable."""
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    assert not fastpath.fused_dispatch_enabled()
    assert fastpath.dispatch_enabled()  # The enable says yes...
    assert fastpath.plan_fast_path(None, None, cupy_like_grid()) is None  # ...and loses.


@pytest.mark.parametrize("retired", sorted(fastpath.LEGACY_SWITCH_NAMES))
def test_the_kill_switch_outranks_a_retired_switch_name(monkeypatch, retired):
    """RUNG 1 IS FIRST, INCLUDING BEFORE THE RETIRED-NAME RULE.

    ``_base_record`` reports the enable's effective value on every record, and it is
    built before the ladder is entered. Asking the RAISING reader for it put the
    retired-name check ahead of rung 1 and outside the wrapper at once, so this exact
    pair of variables came out of ``plan_fast_path`` as a ``RuntimeError`` — breaking
    "the kill switch, highest precedence, read before anything else" and "never raises"
    with one call. Parametrised over the whole retired set so a name added later is
    covered without being remembered.
    """
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    monkeypatch.setenv(retired, "1")
    assert fastpath.plan_fast_path(None, None, cupy_like_grid()) is None
    assert "kill switch" in fastpath.last_dispatch_report()["refused_because"]


@pytest.mark.parametrize("retired", sorted(fastpath.LEGACY_SWITCH_NAMES))
def test_a_retired_switch_name_refuses_by_its_own_name_not_as_an_internal_error(
        monkeypatch, retired):
    """Every rung answers with a NAMED reason and none of them raise — this one included."""
    monkeypatch.setenv(retired, "1")
    assert fastpath.plan_fast_path(None, None, cupy_like_grid()) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "retired environment switch set" in reason, reason
    assert retired in reason and fastpath.LEGACY_SWITCH_NAMES[retired] in reason, reason
    assert "internal error" not in reason, (
        "the refusal arrived through the catch-all wrapper, which names no rung")


@pytest.mark.parametrize("retired", sorted(fastpath.LEGACY_SWITCH_NAMES))
def test_the_retired_name_rule_itself_still_raises(monkeypatch, retired):
    """THE CONTROL for the two above. Catching the rule at rung 2 must not soften it:
    ``dispatch_enabled`` still refuses a stale name loudly to every other caller."""
    monkeypatch.setenv(retired, "1")
    with pytest.raises(RuntimeError, match="retired environment switch set"):
        fastpath.dispatch_enabled()


def test_the_composer_is_never_reached_when_the_kill_switch_is_set(monkeypatch):
    calls = install_composer(monkeypatch, step_plan())
    stub_triton(monkeypatch)
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid()) is None
    assert calls == [], "the kill switch must be answered before anything is planned"


def test_dispatch_is_on_when_the_enable_is_unset(monkeypatch):
    """Unset takes the default, and the default dispatches.

    The same configuration the opt-in version of this test refused. The constant
    itself is bound to its licence by
    ``test_the_recorded_dispatch_shape_matches_the_wiring_it_describes``; this pins
    what the ladder does with it, and what the record says it did.
    """
    monkeypatch.delenv(fastpath.DISPATCH_ENABLE, raising=False)
    assert fastpath.DISPATCH_BY_DEFAULT is True, (
        "the default is bound to fingerprints.json['driver_dispatch']"
        "['dispatch_by_default_licence'], not to a code change")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                      {"step_B": "PML"}))
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    enable = plan.report()["enable"]
    assert enable["variable"] == fastpath.DISPATCH_ENABLE
    assert enable["value"] is None
    assert enable["default"] is True
    assert enable["effective"] is True


@pytest.mark.parametrize("value", ["", " ", "true", "false", "off", " 1", "0 "])
def test_an_unrecognised_enable_value_refuses_a_plan_that_would_dispatch(
        monkeypatch, value):
    """Refused BY NAME at rung 2, before the composer is asked, on the array path.

    The control first: the identical composition dispatches under ``1``, so the
    refusal below is the enable's and nothing else's. Before the strict parser
    every one of these values but the empty and whitespace-only ones enabled
    dispatch, and those two took the default.
    """
    composed = step_plan({"step_B": CountingPlan("b")}, {"step_B": "PML"})
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    assert plan_on_a_cupy_host(monkeypatch, composed) is not None, (
        fastpath.last_dispatch_report()["refused_because"])

    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, value)
    calls = install_composer(monkeypatch, composed)
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid()) is None
    assert calls == [], "the enable must be answered before anything is planned"
    record = fastpath.last_dispatch_report()
    assert record["step_path"] == "array" and record["decision"] == "refused"
    reason = record["refused_because"]
    assert f"{fastpath.DISPATCH_ENABLE}={value!r} is not an accepted value" in reason, (
        reason)
    assert "internal error" not in reason and "retired" not in reason, reason
    assert record["environment"] == {
        "not_read": "refused at an unrecognised enable value"}
    assert record["enable"]["value"] == value
    assert record["enable"]["effective"] is False


def test_a_retired_name_is_refused_before_an_unrecognised_enable_value(monkeypatch):
    """Two rules raise at rung 2; the retired name is checked first and says so."""
    monkeypatch.setenv("TRIDENT_FDTD_TRITON", "1")
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "true")
    assert fastpath.plan_fast_path(None, None, cupy_like_grid()) is None
    record = fastpath.last_dispatch_report()
    assert "retired environment switch set" in record["refused_because"]
    assert record["environment"] == {"not_read": "refused at a retired switch name"}


def test_the_enable_set_to_zero_names_itself(monkeypatch):
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "0")
    assert plan_on_a_cupy_host(monkeypatch, step_plan()) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert reason == f"dispatch disabled by {fastpath.DISPATCH_ENABLE}=0"


def _no_metal_hardware(monkeypatch) -> None:
    """Answer the candidate-table rung as a machine with no MPS device would.

    THE TWO REFUSALS BELOW ARE ABOUT THE BACKEND AND NOT ABOUT THE HARDWARE, and on
    a Mac those are different facts. Rung 3 picks the CANDIDATE SET from the
    hardware — a NumPy engine with an MPS device takes the Metal table — so on this
    laptop a NumPy backend is admitted rather than refused, and a test asserting the
    refusal would pass only on hosts without a GPU. Pinning the hardware read is
    what makes these two tests about the ladder's own clause on every machine.

    A no-op while ``metal_hardware_present`` has not landed, so the two tests keep
    measuring exactly what they measured before the dispatch batch.
    """
    if hasattr(fastpath, "metal_hardware_present"):
        monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: False)


def test_a_numpy_backend_is_refused_without_importing_cupy(monkeypatch):
    """Clause (f) of the ladder, and the package boundary at the same time."""
    _no_metal_hardware(monkeypatch)
    before = {name for name in sys.modules if name.partition(".")[0] == "cupy"}
    grid = types.SimpleNamespace(xp=np)
    assert fastpath.plan_fast_path(None, None, grid) is None
    after = {name for name in sys.modules if name.partition(".")[0] == "cupy"}
    assert after == before
    assert "not CuPy" in fastpath.last_dispatch_report()["refused_because"]


def test_an_unknown_backend_is_refused_by_name(monkeypatch):
    _no_metal_hardware(monkeypatch)
    xp = StubBackend()
    xp.__name__ = "not_numpy"
    assert fastpath.plan_fast_path(None, None, types.SimpleNamespace(xp=xp)) is None
    assert "not_numpy" in fastpath.last_dispatch_report()["refused_because"]


def block_cuda(monkeypatch):
    """Answer the CUDA table's candidacy rung as a host without that package would.

    The hand-CUDA table's candidacy is a package reachability question
    (``fastpath_cuda.cuda_candidate``), so a checkout that HAS the package is always
    a candidate for it — which is every machine this suite runs on. Pinning the
    answer is what lets the two tests below measure "the candidate set emptied"
    rather than measuring which packages happen to be installed here.
    """
    from meep_gpu import fastpath_cuda

    monkeypatch.setattr(
        fastpath_cuda, "cuda_candidate",
        lambda record, xp: {"candidate": False,
                            "refused_because": "the hand-CUDA table is blocked "
                                               "for this test"})


def test_triton_absent_drops_that_table_and_keeps_the_other(monkeypatch):
    """Clause (e) stopped being a WHOLE-PLAN refusal when the second NVIDIA table landed.

    Every Triton kernel builder imports Triton, so a host without it has no
    Triton-bearing slot — but that is a fact about ONE TABLE, and refusing the plan
    on it would take the hand-CUDA table down on exactly the host where it is the
    only one left. The refusal is now per table and the plan is refused only when
    the candidate set empties.

    THE OTHER DIRECTION IS THE TEST BELOW, and it is a separate one rather than a
    second half: a ``monkeypatch`` set for the first half is still in force for the
    second, and undoing it here would also undo the autouse fixtures that set the
    enable — so the "both tables blocked" leg would be measured twice and read as a
    pass.
    """
    block_triton(monkeypatch)
    block_cuda(monkeypatch)
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid()) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "no kernel table is a candidate" in reason, reason
    assert "Triton is not importable" in reason, reason
    assert "the hand-CUDA table is blocked" in reason, reason


def test_a_triton_less_cupy_host_still_reaches_the_hand_cuda_table(monkeypatch):
    """The half a whole-plan refusal at rung 4 would have made unreachable."""
    block_triton(monkeypatch)
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid()) is None
    record = fastpath.last_dispatch_report()
    assert record["tables"]["triton"]["candidate"] is False
    assert "Triton is not importable" in record["tables"]["triton"]["refused_because"]
    assert record["tables"]["cuda"]["candidate"] is True
    assert record["arbitration"]["effective_order"] == ["cuda"]
    # And the plan did not die at that rung: it walked on and refused below, which
    # is what a Triton-less CuPy host must do for the other table to be reachable.
    assert "not importable" not in record["refused_because"]


def test_an_uncertified_triton_version_drops_that_table_by_name(monkeypatch):
    """Triton owns the code generation the bit-identity gates are claims about.

    So an unvalidated compiler invalidates the Triton certifications wholesale — and
    says nothing at all about the hand-written CUDA kernels, which is why this is a
    table drop rather than a plan refusal. The version is still named, in the
    table's own row, and the whole plan still refuses when it is the last candidate.
    """
    validated = fastpath.validated_triton_versions()
    assert validated, "fingerprints.json must record which Triton the gates ran on"
    block_cuda(monkeypatch)
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                      {"step_B": "PML"}),
                               version="9.9.9-unreleased")
    assert plan is None
    record = fastpath.last_dispatch_report()
    assert "9.9.9-unreleased" in record["refused_because"]
    assert "9.9.9-unreleased" in record["tables"]["triton"]["refused_because"]
    assert list(validated) == record["environment"]["validated_triton_versions"]


# ---------------------------------------------------------------------------
# The complex-expansion licence, at the rung that consumes it
# ---------------------------------------------------------------------------
#
# The artifact shipped x86 naturally cuts is stamped 'flush' — CuPy appends
# -ftz=true to every NVRTC compile — while CERTIFICATION_SUBNORMAL_POLICY is
# 'keep' and the subnormal gate refuses any process that installed anything else.
# So the natural artifact and the only permitted run policy disagree BY
# CONSTRUCTION, and the licence is consumed by ``plan_step`` FOUR RUNGS BEFORE
# that gate runs. Until 2026-08-15 nothing compared them.

def _probe_artifact(tmp_path, resolved, name="probe.json"):
    """A well-formed, fully-discriminating probe record cut under ``resolved``."""
    patterns = ("c8_mul_c8", "c8_mul_f4_field_left",
                "f4_mul_c8_coefficient_left", "python_float_left")
    record = {
        "backend": "cupy",
        "patterns": {p: "FMA_V1" for p in patterns},
        "vectors": {p: 2792 for p in patterns},
        "detail": {p: {"licensable_arms_disagreement_words": 128,
                       "discriminates": True,
                       "matches": {"FMA_V1": True, "NAIVE": False,
                                   "PLANEWISE_diagnostic": False}}
                   for p in patterns},
        "subnormal_policy": {"policy": f"stamp_{resolved}", "resolved": resolved},
        "candidates": {"policy": resolved},
        "environment": {"backend": "cupy", "machine": "x86_64",
                        "cupy_version": "13.5.1"},
    }
    path = tmp_path / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, record


def test_a_probe_cut_under_the_wrong_policy_is_dropped_before_the_composer(
        monkeypatch, tmp_path):
    """THE FIX. A flush-cut licence must not reach a run that will step under
    keep — the policy under which the three zero-imaginary patterns licensing
    every real-coefficient multiply in both complex kernels are NOT bit-identical
    between the arms. The record is DROPPED rather than the plan refused: a
    refused probe makes the complex predicates refuse by name, which is the array
    path for the complex arms and no change for a run with no complex storage.

    WHAT THIS TEST USED TO ASSERT, AND WHY THAT WAS THE HOLE. It asserted the
    composer was handed ``None``. It was — and the composer then re-read the same
    environment variable, got the same artifact back, and handed it to all eight
    complex arms, because ``None`` is ALSO how "no artifact was offered" is
    spelled and that case may legitimately read the environment. A test that
    accepts ``None`` here cannot tell a refusal from an absence, which is exactly
    the confusion the code had. It now asserts the refusal is a value of its own.
    ``test_dispatch_expansion_refusal.py`` carries what that value does at the
    rungs below."""
    from meep_gpu.expansion_refusal import RefusedExpansionProbe

    assert fastpath.CERTIFICATION_SUBNORMAL_POLICY == "keep"
    path, record = _probe_artifact(tmp_path, "flush")
    monkeypatch.setenv(fastpath.COMPLEX_PROBE_ENV, str(path))
    calls = install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                    {"step_B": "PML"}))
    stub_triton(monkeypatch, "3.1.0")
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())

    assert calls, "the composer was never called"
    offered = calls[0]["probe"]
    assert offered != record, (
        "the composer was handed a licence cut under the wrong policy")
    assert isinstance(offered, RefusedExpansionProbe), (
        "the refusal was passed as None, which every rung below reads as "
        f"'nothing was offered' and answers by reading {fastpath.COMPLEX_PROBE_ENV} "
        f"for itself; the composer got {offered!r}")
    assert offered.key == fastpath.COMPLEX_PROBE_ENV
    assert any("does not transfer" in reason for reason in offered.reasons)
    environment = fastpath.last_dispatch_report()["environment"]
    dropped = environment["complex_expansion_probe_dropped"]
    assert dropped and "does not transfer" in dropped[0]
    assert "flush" in dropped[0] and "keep" in dropped[0]
    assert environment["complex_expansion_probe_resolved"] is False


def test_a_probe_cut_under_the_required_policy_reaches_the_composer(
        monkeypatch, tmp_path):
    """The other direction, or the check would be a blanket refusal."""
    path, record = _probe_artifact(tmp_path, "keep")
    monkeypatch.setenv(fastpath.COMPLEX_PROBE_ENV, str(path))
    calls = install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                    {"step_B": "PML"}))
    stub_triton(monkeypatch, "3.1.0")
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())

    assert calls and calls[0]["probe"] == record
    environment = fastpath.last_dispatch_report()["environment"]
    assert "complex_expansion_probe_dropped" not in environment
    assert environment["complex_expansion_probe_resolved"] is True


def test_the_dispatch_record_says_which_licence_it_consumed(monkeypatch, tmp_path):
    """AUDITABILITY. The record used to carry only 'was an env var set' and 'did
    a record resolve' — not the arm, not the basis, not the policy, and not the
    environment the artifact CLAIMS, which is what the environment-default table
    keys on. A probe.json copied from another machine licensed dispatch here and
    left no trace of having done so."""
    path, _ = _probe_artifact(tmp_path, "keep")
    monkeypatch.setenv(fastpath.COMPLEX_PROBE_ENV, str(path))
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    stub_triton(monkeypatch, "3.1.0")
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())

    licence = fastpath.last_dispatch_report()["environment"]["complex_expansion_licence"]
    assert licence["arm"] == "FMA_V1"
    assert licence["basis"] == "measured"
    assert licence["policy_resolved"] == "keep"
    assert licence["refusals"] == []
    assert licence["claimed_environment"]["machine"] == "x86_64"
    assert licence["claimed_environment"]["cupy_version"] == "13.5.1"


def test_an_artifact_claiming_another_machine_is_named_in_the_record(
        monkeypatch, tmp_path):
    """``environment_default`` matches its table against the record's own
    ``environment`` block and nothing cross-checked that block against the host
    it is being READ on. Reported rather than refused — the policy check is what
    holds the line — but a wrong-host artifact must be visible in the audit."""
    path, _ = _probe_artifact(tmp_path, "keep")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["environment"]["machine"] = "s390x"
    record["environment"]["cupy_version"] = "99.0.0"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv(fastpath.COMPLEX_PROBE_ENV, str(path))
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    stub_triton(monkeypatch, "3.1.0")
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())

    mismatch = fastpath.last_dispatch_report()["environment"][
        "complex_expansion_probe_environment_mismatch"]
    assert any("s390x" in line for line in mismatch)
    assert any("99.0.0" in line for line in mismatch)


def test_a_malformed_probe_artifact_does_not_take_the_planner_down(
        monkeypatch, tmp_path):
    """``plan_fast_path`` never raises. A JSON list where a pattern verdict
    belongs used to raise TypeError out of the licence rule; the seam must reach
    the array path instead."""
    path, _ = _probe_artifact(tmp_path, "keep")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["patterns"]["c8_mul_c8"] = ["FMA_V1"]
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv(fastpath.COMPLEX_PROBE_ENV, str(path))
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    stub_triton(monkeypatch, "3.1.0")
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())  # must not raise
    licence = fastpath.last_dispatch_report()["environment"]["complex_expansion_licence"]
    assert licence["refusals"], "a malformed artifact must refuse by name"


def test_a_deliberately_uncovered_grid_reaches_the_array_path_with_reasons(monkeypatch):
    """Nothing admitted: refused, and every slot carries the refusal text."""
    reasons = {slot: (f"{slot}: the grid carries a feature no kernel implements",)
               for slot in launch_module.STEP_ORDER}
    assert plan_on_a_cupy_host(monkeypatch, step_plan(reasons=reasons)) is None
    record = fastpath.last_dispatch_report()
    assert "no slot is left carrying a kernel" in record["refused_because"]
    for slot in launch_module.STEP_ORDER:
        entry = record["slots"][slot]
        assert entry["state"] == "array"
        assert "no kernel implements" in entry["reason"]


def test_the_real_composer_turns_a_raising_predicate_into_a_named_refusal(monkeypatch):
    """Clause (a), measured on the REAL composer rather than on a stand-in."""
    def exploding(*args, **kwargs):
        raise RuntimeError("the predicate exploded")

    monkeypatch.setattr(launch_module, "pml_curl_coverage", exploding)
    driver = numpy_driver()
    driver.setup_pml(1)
    plan = launch_module.plan_step(driver.fields, driver.pml)
    assert "step_B" not in plan.plans, "a raising predicate may never admit"
    assert any("raised" in reason for reason in plan.reasons["step_B"])
    driver.close()


def test_the_real_composer_turns_a_none_builder_into_a_named_refusal(monkeypatch):
    """Clause (b): coverage admits, the builder refuses, the slot stays empty."""
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda *a, **k: Coverage(True, ()))
    monkeypatch.setattr(launch_module, "plan_constitutive", lambda *a, **k: None)
    driver = numpy_driver()
    driver.setup_pml(1)  # An ACTIVE absorber gates the null arm off, so "ordinary"
    plan = launch_module.plan_step(driver.fields, driver.pml)  # is the sole admitter.
    assert "update_H" not in plan.plans
    assert any("builder refused" in reason for reason in plan.reasons["update_H"])
    driver.close()


def test_the_real_composer_leaves_two_admitters_unselected(monkeypatch):
    """Clause (c): an overlap is a predicate defect and resolves to the array path."""
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda *a, **k: Coverage(True, ()))
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda *a, **k: Coverage(True, ()))
    driver = numpy_driver()
    driver.setup_pml(1)
    plan = launch_module.plan_step(driver.fields, driver.pml)
    assert "update_E" not in plan.plans
    assert plan.reasons["update_E"] == (
        "both update_E predicates (ordinary, dispersive) admitted the same "
        "configuration; refusing an ambiguous numerical product",), plan.reasons["update_E"]
    driver.close()


def test_a_partial_plan_dispatches_its_subset_and_leaves_the_rest_array(monkeypatch):
    """Clause (d): partial coverage of a step is legal and expected."""
    b, h = CountingPlan("step_B"), CountingPlan("update_H")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": b, "update_H": h},
        {"step_B": "PML", "update_H": "ordinary"},
        {"step_D": ("step_D: the conductive product refused",)}))
    assert plan is not None
    assert plan.slots == ("step_B", "update_H")
    record = plan.report()
    assert record["step_path"] == "fused"
    assert record["slots"]["step_B"]["state"] == "dispatched"
    assert record["slots"]["step_D"]["state"] == "array"
    assert "conductive product refused" in record["slots"]["step_D"]["reason"]


def test_plan_fast_path_never_raises_even_when_the_composer_does(monkeypatch):
    """Clause (i). ``plan_step`` never raises; this covers fastpath's own code."""
    assert plan_on_a_cupy_host(monkeypatch, RuntimeError("composer blew up")) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert reason.startswith("internal error")
    assert "composer blew up" in reason


# ---------------------------------------------------------------------------
# Completeness (C3), the null drop (C4) and the fusion refusal (C5)
# ---------------------------------------------------------------------------


def test_a_filled_slot_the_seam_cannot_run_refuses_the_whole_plan(monkeypatch):
    """C3. Running the remainder would be a composition no gate ever certified.

    ``fill_B`` USED TO BE THIS TEST'S SUBJECT and stopped being it on 2026-09-02,
    when the two driver consults put ``fill_B``/``fill_D`` into ``DRIVER_SLOTS``.
    What C3 pins is the RULE, not the slot that happened to trip it, so the case is
    armed with a slot no consult stands in front of — which is what the composer
    growing an eighth sub-step would look like from here. The day that happens the
    step must not run half on kernels; it must refuse whole.
    """
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b"), "zero_metal_B": CountingPlan("wall")},
        {"step_B": "PML", "zero_metal_B": "a wall clear from a later composer"}))
    assert plan is None, "a step must not dispatch a slot the seam cannot complete"
    record = fastpath.last_dispatch_report()
    assert "zero_metal_B" in record["refused_because"]
    assert record["built_not_dispatched"] == {
        "zero_metal_B": "a wall clear from a later composer"}


def test_every_slot_the_composer_can_fill_now_has_a_driver_consult():
    """The 2026-09-02 move, stated as the identity that makes rung 6 unreachable.

    Both halves are asserted because they are different claims: that the driver has
    a consult for every ``STEP_ORDER`` name, and that it claims no name the composer
    cannot write. A ``DRIVER_SLOTS`` carrying a name ``STEP_ORDER`` does not would be
    a consult wired to nothing.
    """
    assert set(fastpath.DRIVER_SLOTS) == set(launch_module.STEP_ORDER)
    assert fastpath.DRIVER_SLOTS == launch_module.STEP_ORDER, (
        "the slots are consulted in this order; a permutation would put a consult "
        "in the wrong slot of the step")
    assert set(fastpath.FAR_FILL_PASSES) == {"fill_B", "fill_D"}
    assert set(fastpath.FAR_FILL_PASSES) < set(fastpath.DRIVER_SLOTS)
    assert "zero_metal_B" not in fastpath.FAR_FILL_PASSES.values(), (
        "the wall clear runs behind no consult; deposit_repair reads that")


def test_a_folded_run_that_fills_both_fill_slots_now_dispatches(monkeypatch):
    """THE MOVE ITSELF. This exact composition refused at rung 6 until 2026-09-02.

    ``folded_2d`` in the driver-route gate selected ``fused pair B (folded)`` at
    ``step_B``/``update_H``, the mirror-fill arm won ``fill_B``/``fill_D``, and the
    completeness rung refused the whole plan because the two fills sat outside
    ``DRIVER_SLOTS``. With the consults they do not, and the plan survives.
    """
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan(
        {"step_B": CountingPlan("b"), "fill_B": CountingPlan("fill_b"),
         "update_H": CountingPlan("h"), "step_D": CountingPlan("d"),
         "fill_D": CountingPlan("fill_d"), "update_E": CountingPlan("e")},
        {"step_B": "folded PML", "fill_B": "mirror fill", "update_H": "folded",
         "step_D": "folded PML", "fill_D": "mirror fill", "update_E": "folded"}))
    plan = fastpath.plan_fast_path(object(), object(), folded_grid())
    assert plan is not None, fastpath.last_dispatch_report().get("refused_because")
    assert plan.slots == ("step_B", "fill_B", "update_H",
                          "step_D", "fill_D", "update_E"), plan.slots
    assert fastpath.last_dispatch_report()["decision"] == "dispatched"


# ---------------------------------------------------------------------------
# The vocabulary: MEASURED against the composer, not shared with it
# ---------------------------------------------------------------------------
#
# The kernel package must not know a dispatcher exists — every file in it is
# sha256-welded to the device gate that certified it — so these names live in
# ``fastpath`` and their agreement with what the composer WRITES is a
# measurement rather than an import.


def test_the_null_arm_label_is_what_the_composer_actually_writes():
    """Read off a real composition: a no-absorber run gives both constitutive
    slots to the null family, whose product launches nothing."""
    driver = numpy_driver()  # No absorber installed, so the null arm is gated on.
    plan = launch_module.plan_step(driver.fields, driver.pml)
    assert plan.selected.get("update_H") == fastpath.NULL_ARM_LABEL
    assert plan.selected.get("update_E") == fastpath.NULL_ARM_LABEL
    driver.close()


def test_every_arm_the_composer_can_select_is_certified_or_refused_as_fused():
    """A family wired into ``plan_step`` without a certification entry fails HERE.

    Otherwise it would reach a user's artifact as ``"family": "unmapped"`` — a
    dispatched kernel whose provenance the record could not name, which is the
    one thing the artifact exists to prevent.
    """
    import pathlib
    import re

    from meep_gpu.triton_kernels.coverage import FUSED_PAIRS

    source = pathlib.Path(launch_module.__file__).read_text(encoding="utf-8")
    # The three labels the composer writes OUTSIDE the arm table; asserted to be
    # present so this set cannot go stale against a rename.
    for literal in ('"ADE update_P"', '"fused ADE state"', '"dispersive fused pair"',
                    'f"fused pair {pair_name}"'):
        assert literal in source, f"the composer no longer writes {literal}"
    labels = set(re.findall(r'_Arm\("([^"]+)"', source))
    assert len(labels) > 25, f"only {len(labels)} arms found; the regex went stale"
    labels |= {"ADE update_P", "fused ADE state", "dispersive fused pair"}
    labels |= {f"fused pair {name}" for name in FUSED_PAIRS}
    unmapped = sorted(label for label in labels
                      if label not in fastpath.ARM_CERTIFICATION
                      and label not in fastpath.PENDING_DEVICE_GATE_ARMS
                      and not fastpath.arm_is_fused(label))
    assert not unmapped, f"arms with no recorded certification: {unmapped}"


def test_a_pending_device_gate_refuses_the_whole_step_before_warming(monkeypatch):
    """A future planner-visible kernel cannot launch on borrowed provenance."""
    monkeypatch.setattr(
        fastpath, "PENDING_DEVICE_GATE_ARMS",
        {"synthetic pending arm": "its dedicated byte gate has not run"})
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("conductive")},
        {"step_B": "synthetic pending arm"}))
    assert plan is None
    report = fastpath.last_dispatch_report()
    assert "dedicated byte gate has not run" in report["refused_because"]
    assert report["built_not_dispatched"] == {"step_B": "synthetic pending arm"}


def _recut_tool():
    """``parity/meep_gpu/recut_driver_dispatch_record.py``: the licence's constants.

    IMPORTED, NEVER RESTATED. The dispatched floor and the environment rule are
    decided by the tool that transcribes the licence, by the go/no-go driver and by
    the weld below, and three sites that each typed their own floor disagreed —
    one dispatched case transcribed and read GO while this test asked for seven.
    The harness tree is not part of the standalone package, so its absence is a
    declared resource rather than a failure.
    """
    import importlib

    parity = pathlib.Path(fastpath.__file__).resolve().parent.parent / "parity" / "meep_gpu"
    if not (parity / "recut_driver_dispatch_record.py").is_file():
        requires_resource_skip(
            "parity_meep_gpu_harness",
            "parity/meep_gpu/recut_driver_dispatch_record.py holds the licence's "
            "constants and is not shipped with the standalone package")
    if str(parity) not in sys.path:
        sys.path.insert(0, str(parity))
    return importlib.import_module("recut_driver_dispatch_record")


def _validated_capabilities_by_table():
    """The compute capabilities each NVIDIA table's ledger certifies, read live."""
    from meep_gpu import fastpath_cuda

    # ``fastpath.validated_compute_capabilities`` reads the Triton ledger; the
    # hand-CUDA table derives its own list from its certification blocks.
    readers = {fastpath.CUDA_TABLE: fastpath_cuda.validated_compute_capabilities}
    return {table: set(readers.get(table, fastpath.validated_compute_capabilities)())
            for table in fastpath.NVIDIA_TABLE_PRECEDENCE}


def _licensed_capabilities():
    """The compute capabilities whose dispatch-by-default licence the Triton record holds.

    A licence lives on the PRIMARY table for its capability
    (:func:`fastpath.primary_table`): the first table in the shipped precedence that
    admits it, whose plan composes first and is therefore what the ship leg measured.
    That is where ``recut_driver_dispatch_record.py`` writes it, so it is where a
    reader looks -- one licence per architecture, never one for the record as a whole.
    """
    admitted = set()
    for table in fastpath.NVIDIA_TABLE_PRECEDENCE:
        admitted |= set(fastpath.capability_admission(table)["admitted"] or ())
    return sorted(wanted for wanted in admitted
                  if fastpath.primary_table(wanted) == "triton")


def _licence_at(record, capability):
    """``runs[<capability>].dispatch_by_default_licence``, or ``None`` while unwritten."""
    run = (record.get(fastpath.RUNS) or {}).get(capability) or {}
    return run.get("dispatch_by_default_licence")


def _assert_the_licence_holds(record, package) -> None:
    """Every licence the Triton record has to carry, each checked field by field."""
    licensed = _licensed_capabilities()
    assert licensed, (
        "no compute capability has the Triton table as its primary table, so there is "
        "no record a dispatch-by-default licence could be transcribed into")
    for wanted in licensed:
        _assert_one_licence_holds(record, package, wanted)


def _assert_one_licence_holds(record, package, wanted) -> None:
    """``driver_dispatch.runs[<capability>].dispatch_by_default_licence``, field by field.

    What the transcription enforces when it writes the block, checked again here
    against the block as it stands, so a hand-typed licence carrying only the fields
    an older weld read cannot pass: every case released and none divergent; at
    least :data:`recut_driver_dispatch_record.LICENCE_MIN_DISPATCHED` dispatched;
    nothing installed by the harness; the enable REMOVED (``enable_from_default``)
    and read at its default on every case, in an environment carrying no
    dispatch-shaping variable; a null control beside it that is clean over the same
    case list; a device whose compute capability every table that served certifies;
    and a slot that still binds the record's bytes, whose ``fastpath.py`` is the tree's.
    """
    import hashlib

    recut = _recut_tool()
    licence = _licence_at(record, wanted)
    assert licence, (
        f"dispatch_by_default is True but driver_dispatch carries no "
        f"dispatch_by_default_licence for compute capability {wanted} under "
        f"runs[{wanted!r}], where recut_driver_dispatch_record.py writes it: run "
        "gate_dispatch_end_to_end's ship leg on the shipping bytes (harness installing "
        "no policy, MEEP_GPU_DISPATCH unset) and transcribe it with "
        "recut_driver_dispatch_record.py --end-to-end")
    assert licence["gate"] == recut.LICENCE_GATE, licence.get("gate")
    assert licence["run"] and licence["recorded_utc"] and licence["host"], licence
    verdicts = licence["verdicts"]
    assert len(verdicts) == 9, sorted(verdicts)
    assert set(verdicts.values()) <= set(recut.LICENCE_VERDICTS), verdicts
    dispatched = sorted(case for case, verdict in verdicts.items()
                        if verdict == "PASS-DISPATCHED")
    assert len(dispatched) >= recut.LICENCE_MIN_DISPATCHED, (
        f"{len(dispatched)} of {len(verdicts)} dispatched ({dispatched}); the "
        f"licence needs at least {recut.LICENCE_MIN_DISPATCHED}")
    assert sorted(licence["dispatched_cases"]) == dispatched, licence["dispatched_cases"]
    assert sorted(licence["fell_back_cases"]) == sorted(
        set(verdicts) - set(dispatched)), licence["fell_back_cases"]
    divergent = licence["first_divergent_checkpoint"]
    assert set(divergent) == set(verdicts), sorted(divergent)
    assert all(checkpoint is None for checkpoint in divergent.values()), divergent
    assert licence["install_policy_requested"] is None, (
        "the licence is the leg where the harness installs NOTHING")
    assert licence["null_control"] is False, "a null control licenses nothing"
    assert licence["enable_from_default"] is True, (
        "the licence leg must REMOVE the enable (--enable-from-default); a leg that "
        "set it measured a run that asked for dispatch, not the default")
    environment = licence["environment_at_start"]
    assert isinstance(environment, dict), (
        f"environment_at_start is {environment!r}; a plain environment cannot be "
        "shown without it")
    shaping = sorted(
        name for name in environment
        if name not in recut.LICENCE_ENVIRONMENT_ALLOWED
        and (name.startswith(recut.LICENCE_ENVIRONMENT_REFUSED_PREFIXES)
             or name in recut.LICENCE_ENVIRONMENT_REFUSED_NAMES))
    assert not shaping, (
        f"the licence leg inherited {shaping}; a plain install sets none of them")
    enable = licence["enable"]
    assert set(enable) == set(verdicts), sorted(enable)
    for case, block in enable.items():
        assert block["variable"] == fastpath.DISPATCH_ENABLE, (case, block)
        assert block["value"] is None, (
            f"{case}: the licence leg set the enable to {block['value']!r}; it has to "
            "measure the unset default a user gets")
        assert block["effective"] is True, (case, block)
    null_run = licence["null_control_run"]
    assert isinstance(null_run, dict), "no null control beside the licence leg"
    assert null_run["clean"] is True, (
        f"the null control is not clean: {null_run.get('verdicts')}, "
        f"{null_run.get('first_divergent_checkpoint')}")
    assert set(null_run["verdicts"]) == set(verdicts), (
        f"the null control covers {sorted(null_run['verdicts'])}; the licence leg "
        f"covers {sorted(verdicts)}")
    capability = (licence.get("device") or {}).get("compute_capability")
    assert capability, (
        f"the licence recorded no readable compute capability: {licence.get('device')}")
    assert fastpath._normalized_capability(capability) == wanted, (
        f"the licence filed under runs[{wanted!r}] ran on compute capability "
        f"{capability}: a licence licenses the architecture it measured, and one filed "
        f"under another's key would license a card its run never touched")
    served = {table for names in (licence.get("tables_dispatched") or {}).values()
              for table in (names or ())}
    assert served, "no case of the licence leg recorded a table that served it"
    certified = _validated_capabilities_by_table()
    for table in sorted(served):
        assert table in certified, f"{table} is not an NVIDIA table this weld knows"
        assert capability in certified[table], (
            f"the licence ran on compute capability {capability}, which the {table} "
            f"table's ledger does not certify ({sorted(certified[table])})")
    # THE BYTES ARE PINNED BY THE SLOT, NOT BY THE LICENCE. The transcription proves
    # the ship leg ran every file the record binds and then writes NO digest map into
    # the block (``recut_driver_dispatch_record.transcribe_end_to_end``): the licence
    # sits in ``runs[<capability>]``, whose ``bound_sha256`` is the digest of the
    # record's own ``source_sha256``, so the moment those bytes move the whole slot,
    # licence included, reads stale. A map inside the block would be a second pin the
    # weld walk raw-checks forever, which a superseded licence can never satisfy. So
    # "cut on the shipping bytes" is three facts here: the slot is live, the record
    # binds the tree's ``fastpath.py``, and the block carries no map of its own. What
    # the run itself digested is compared against the record in
    # ``test_the_dispatch_by_default_licence_is_the_artifact_it_names``, from the
    # artifact ``artifact_sha256`` pins.
    assert "source_sha256" not in licence, (
        f"the licence under runs[{wanted!r}] carries its own source_sha256 map; the "
        "slot's bound_sha256 is the pin, and the run's digests stay in its artifact")
    assert wanted in fastpath.live_capabilities(record), (
        f"the licence sits in runs[{wanted!r}], whose bound_sha256 no longer matches "
        f"the record's source_sha256 (live: "
        f"{list(fastpath.live_capabilities(record))}): the bytes it was cut on have "
        "moved, so it licenses nothing that ships")
    bound = record["source_sha256"]["fastpath.py"]
    tree = hashlib.sha256((package / "fastpath.py").read_bytes()).hexdigest()
    assert bound == tree, (
        f"the record binds fastpath.py {bound[:12]} and the tree ships {tree[:12]}: "
        "the licence has to be cut on the shipping bytes")


def _assert_each_admitted_capability_names_its_route_campaign(record, table, stamp):
    """``runs[<capability>].released_fused_arms`` names THAT capability's route campaign.

    WHICH CAMPAIGN DROVE THE ARMS IS A FACT ABOUT ONE RUN. The recut moves ``gate``,
    ``artifact`` and ``what_was_measured``
    (:data:`recut_driver_dispatch_record.RELEASED_RUN_SUBKEYS`) out of
    ``exclusions.released_fused_arms`` into the route run's slot, and refuses a
    campaign directory other than the one :func:`fastpath.route_campaign` spells for
    the capability its legs stamped. So for every capability the table ADMITS: the
    slot is live (its ``bound_sha256`` still binds the record's bytes), its ``gate``
    and ``records`` name ``route_campaign(stamp, capability)``, and its ``artifact``
    names that directory. Asked of every admitted capability, because an admitted
    architecture with no live route run dispatches arms no route campaign drove on
    it; and the entry level must have shed the three subkeys, because a ``gate``
    left there is a second answer to which campaign drove the release.

    AND EVERY LIVE ROUTE RUN READS PASS, in the words the three weld contracts use
    (``weld_record_walk.route_run_problems``): a live run that did not release
    licenses nothing it filed. AND NO ROUTE-RUN FIELD SITS BESIDE THE DIGESTS, read
    against ``fastpath.DISPATCH_RUN_FIELDS``, which names ``status``, as the Metal
    record is held to in
    ``test_the_metal_record_is_welded_to_the_live_sources_when_it_exists``. Those
    contracts also hold the record to the tree; this test compares the tree its own
    way.
    """
    recut = _recut_tool()
    admitted = fastpath.capability_admission(table)["admitted"] or ()
    assert admitted, (
        f"the {table} table admits no compute capability, so no route run can be "
        "checked against one")
    entry_level = sorted(set(recut.RELEASED_RUN_SUBKEYS)
                         & set(record["exclusions"]["released_fused_arms"]))
    assert not entry_level, (
        f"exclusions.released_fused_arms still carries {entry_level} beside the "
        f"per-capability route runs; they describe one run and live in "
        f"{fastpath.RUNS}[<capability>].released_fused_arms")
    live = fastpath.live_capabilities(record)
    for capability in admitted:
        assert capability in live, (
            f"the {table} table admits compute capability {capability} and its "
            f"driver_dispatch record has no live route run for it (live: {list(live)}); "
            "run the route campaign on these bytes and recut")
        run = record[fastpath.RUNS][capability]
        moved = run["released_fused_arms"]
        expected = fastpath.route_campaign(stamp, capability)
        assert moved["gate"] == expected, (
            f"runs[{capability!r}] names route campaign {moved['gate']!r}; the release "
            f"constant spells {expected!r} for this capability")
        assert run["records"] == f"apps/api/parity/meep_gpu/results/{expected}", (
            capability, run["records"])
        assert f"results/{expected}/" in moved["artifact"], (capability, moved["artifact"])
        assert moved["what_was_measured"], capability
    problems = walk.route_run_problems(record, live)
    assert not problems, f"the {table} driver_dispatch record: {problems}"
    shape = fastpath.retired_shape_reasons(record, run_fields=fastpath.DISPATCH_RUN_FIELDS)
    assert not shape, f"the {table} driver_dispatch record: {shape}"


def test_the_recorded_dispatch_shape_matches_the_wiring_it_describes():
    """The same weld ``host_sha256`` puts on the kernel package, on the seam.

    ``fingerprints.json`` records what the driver seam CAN run, what it refuses,
    and the sha256 of the two files that implement it. Editing either without
    re-cutting the record fails here — which is the point, because the record is
    what the driver-route gate will be bound to when it runs.
    """
    import hashlib
    import json
    import pathlib

    package = pathlib.Path(fastpath.__file__).parent
    record = json.loads((package / "triton_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))["driver_dispatch"]
    assert record["dispatch_by_default"] is fastpath.DISPATCH_BY_DEFAULT
    assert tuple(record["runnable_slots"]) == fastpath.DRIVER_SLOTS
    assert set(record["unrunnable_slots"]) - {"consequence"} == (
        set(launch_module.STEP_ORDER) - set(fastpath.DRIVER_SLOTS))
    # THE TRIPWIRE FIRED, and this is what replaced it. It used to assert the
    # driver-route gate had status "NOT RUN"; the gate ran on 2026-08-15, and what
    # has to hold now is that its RESULT and the shipped default agree with each
    # other. It passed under a uniform subnormal policy and failed as shipped, so
    # the default was to stay False until a round closed the executor policy split
    # AND re-ran the ship leg on the shipping bytes (not yet run); the licence block
    # below is where that run is transcribed, and the two move in one change or fail.
    gate = record["driver_route_gate_2026-08-15"]
    assert "RUN 2026-08-15" in record["what_would_certify_this"]["status"]
    assert gate["recorded_utc"] and gate["host"] and gate["records"]
    dispatching = [name for name, entry in gate["under_a_uniform_keep_policy"]
                   ["per_case"].items() if entry["verdict"] == "PASS-DISPATCHED"]
    assert len(dispatching) >= 8, dispatching
    for name in dispatching:
        entry = gate["under_a_uniform_keep_policy"]["per_case"][name]
        assert entry["state_identical"] and entry["observables_identical"], name
        assert entry["first_divergent_checkpoint"] is None, name
        assert entry["triton_launches"] > 0, f"{name} passed without a launch"
        assert entry["killswitch_leg_clean"], name
        assert entry["comparator_control_fired"], name
        assert all(entry["slot_dispatches"].values()), name
        assert all(entry["programs_per_dispatch"].values()), name
    # THE DEFAULT IS BOUND TO ITS LICENCE, in one change. Until 2026-09-27 this
    # asserted ``dispatch_by_default is False``, with the measurement that would
    # license the flip in its message: gate_dispatch_end_to_end's ship leg on the
    # shipping bytes, the harness installing NOTHING and the enable left unset,
    # released. What has to hold with the default on is that the record carries
    # that measurement and that it ran THESE bytes — see
    # :func:`_assert_the_licence_holds`. A flip without the licence, or a licence
    # cut on other bytes, fails here.
    assert record["dispatch_by_default"] is True
    _assert_the_licence_holds(record, package)
    # THE FUSED RELEASE, welded to the record the same way the default is. The
    # record's ``exclusions.fused_arms`` used to say the composer is called with
    # ``fuse=False`` unconditionally; that stopped being true on 2026-08-29 and a
    # wiring record that still said it would be describing a seam that no longer
    # exists. Both directions: the released set the record names is the set the
    # code ships, AND every label in it is one ``arm_is_fused`` recognises — a
    # release the refusal clause could not see would never be consulted.
    released = record["exclusions"]["released_fused_arms"]
    assert set(released["arms"]) == set(fastpath.RELEASED_FUSED_ARMS), released
    _assert_each_admitted_capability_names_its_route_campaign(
        record, "triton", fastpath.DRIVER_ROUTE_FUSED_GATE)
    for arm, cases in fastpath.RELEASED_FUSED_ARMS.items():
        assert fastpath.arm_is_fused(arm), arm
        assert cases, f"{arm} is released without naming a case it was driven on"
        assert set(cases) <= set(released["driven_on_cases"]), arm
    # And the envelope: the record has to name the axes, so a reader who finds a
    # dispatched fused arm in an artifact can check it against the wiring without
    # reading Python.
    assert (set(released["envelope"])
            == {axis for axis, _, _ in fastpath.FUSED_RELEASE_ENVELOPE}), released
    for name, digest in record["source_sha256"].items():
        live = hashlib.sha256((package / name).read_bytes()).hexdigest()
        assert digest == live, (
            f"{name} changed since the dispatch shape was recorded. Re-cut "
            "fingerprints.json['driver_dispatch']['source_sha256'].")


def _licence_artifact(package, spelled: str) -> pathlib.Path:
    """A licence artifact path as the ledger spells it (``...`` or absolute)."""
    prefix = "apps/api/"
    if spelled.startswith(prefix):
        return package.parent / spelled[len(prefix):]
    return pathlib.Path(spelled)


def _results_tree_or_skip(package, what: str) -> pathlib.Path:
    """``parity/meep_gpu/results``, or a declared skip on the TREE's absence.

    Never on an artifact's absence: a checkout that holds results but not the run a
    record names is the state these checks exist to catch, so callers assert the
    artifact is there once the tree is.
    """
    results = package.parent / "parity" / "meep_gpu" / "results"
    if not results.is_dir():
        requires_resource_skip(
            "parity_meep_gpu_results",
            f"parity/meep_gpu/results is gitignored and absent here, so the record "
            f"cannot be read against its artifacts ({what})")
    return results


def _run_artifact(package, run) -> pathlib.Path:
    """The gate artifact a weld's run record names in the first word of ``records``.

    The rebind tools spell it ``apps/api/parity/meep_gpu/results/<root>/<gate>/`` (a
    directory holding ``gate.json``) or the artifact file itself; both are resolved
    and the file must exist once the results tree does.
    """
    path = _licence_artifact(package, run["records"].split()[0].rstrip("/"))
    if path.is_dir():
        path = path / "gate.json"
    assert path.is_file(), (
        f"the run record names {run['records'].split()[0]}, which is not here while "
        "the results tree is: the record cites a run this checkout does not hold")
    return path


def test_the_dispatch_by_default_licence_is_the_artifact_it_names():
    """``artifact_sha256`` is the digest of the summary the licence was cut from.

    The weld above checks the licence's FIELDS; this checks that they are the
    artifact's. A block edited after transcription — a verdict softened, a case
    dropped — keeps every field well-formed and changes nothing on disk, so only the
    digest can see it. The null control's artifact is held to the same rule.

    THE RESULTS TREE IS GITIGNORED, so a clone has no artifacts and this skips on
    the TREE's absence, never on the artifact's: a checkout that has results but not
    the run the licence names must fail.
    """
    import hashlib

    package = pathlib.Path(fastpath.__file__).resolve().parent
    record = json.loads((package / "triton_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))["driver_dispatch"]
    if not fastpath.DISPATCH_BY_DEFAULT:
        return  # no default to license; the weld above pins the record to False
    licences = {wanted: _licence_at(record, wanted)
                for wanted in _licensed_capabilities()}
    assert licences, (
        "no compute capability has the Triton table as its primary table, so there is "
        "no record a dispatch-by-default licence could be transcribed into")
    missing = sorted(wanted for wanted, licence in licences.items() if not licence)
    assert not missing, (
        f"dispatch_by_default is True and driver_dispatch carries no "
        f"dispatch_by_default_licence for compute capability {missing} under "
        f"runs[<capability>]; transcribe the ship leg with "
        "recut_driver_dispatch_record.py --end-to-end")
    results = package.parent / "parity" / "meep_gpu" / "results"
    if not results.is_dir():
        requires_resource_skip(
            "parity_meep_gpu_results",
            "parity/meep_gpu/results is gitignored and absent here, so the licence's "
            "artifacts cannot be re-hashed")
    pairs = []
    for wanted, licence in sorted(licences.items()):
        pairs.append((f"{wanted} ship leg", licence.get("artifact"),
                      licence.get("artifact_sha256")))
        null_run = licence.get("null_control_run") or {}
        pairs.append((f"{wanted} null control", null_run.get("artifact"),
                      null_run.get("artifact_sha256")))
    for what, spelled, recorded in pairs:
        assert spelled and recorded, f"the licence names no {what} artifact and digest"
        path = _licence_artifact(package, spelled)
        assert path.is_file(), (
            f"the licence's {what} artifact {spelled} is not here, while the results "
            "tree is: the licence cites a run this checkout does not hold")
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        assert live == recorded, (
            f"the {what} artifact {spelled} hashes to {live[:12]} and the licence "
            f"recorded {recorded[:12]}: the block was not cut from these bytes")
    # AND THE BYTES THE RUN EXECUTED ARE THE BYTES THE RECORD BINDS. The licence
    # carries no digest map (see :func:`_assert_one_licence_holds`), so the third leg
    # of "artifact, record, tree" is read here, from the artifact the digest above
    # just proved is the one transcribed: every file the record binds, as the ship
    # leg's provenance spells it, at the record's digest; and the null control on
    # the same fastpath.py. Harness files (``parity/``) are skipped exactly as the
    # transcription skips them -- they move no shipped byte.
    recut = _recut_tool()
    for wanted, licence in sorted(licences.items()):
        ship = json.loads(_licence_artifact(package, licence["artifact"])
                          .read_text(encoding="utf-8"))
        ran = (ship.get("provenance") or {}).get("source_sha256") or {}
        checked = 0
        for name, digest in sorted(record["source_sha256"].items()):
            if name.startswith("parity/"):
                continue
            key = recut._leg_key("triton", name)
            assert ran.get(key) == digest, (
                f"the {wanted} ship leg ran {key} at {str(ran.get(key))[:12]} and the "
                f"record binds {digest[:12]}: the licence was cut on other bytes")
            checked += 1
        assert checked and "fastpath.py" in ran, (
            f"the {wanted} ship leg's provenance names none of the bound files")
        null = json.loads(_licence_artifact(package, licence["null_control_run"]["artifact"])
                          .read_text(encoding="utf-8"))
        assert ((null.get("provenance") or {}).get("source_sha256") or {}).get(
            "fastpath.py") == ran["fastpath.py"], (
            f"the {wanted} null control ran another fastpath.py than its ship leg")


def test_the_cuda_record_states_the_default_the_code_ships():
    """The CUDA ``driver_dispatch`` record's ``dispatch_by_default`` IS the constant.

    ``recut_driver_dispatch_record.py`` writes the field from
    ``fastpath.DISPATCH_BY_DEFAULT`` on both NVIDIA records — the CUDA branch used to
    type a literal ``False`` — but nothing bound the CUDA record's copy to the code,
    so a flip that re-cut only the Triton record left the other stating the old
    default with every test green. Skips only while the record does not exist.
    """
    package = pathlib.Path(fastpath.__file__).parent
    ledger = json.loads((package / "cuda_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))
    record = ledger.get("driver_dispatch")
    if record is None:
        requires_resource_skip(
            "cuda_driver_dispatch_record",
            "the CUDA driver_dispatch block has not been cut")
    assert record.get("dispatch_by_default") is fastpath.DISPATCH_BY_DEFAULT, (
        f"the CUDA record reads dispatch_by_default "
        f"{record.get('dispatch_by_default')!r} and fastpath.DISPATCH_BY_DEFAULT is "
        f"{fastpath.DISPATCH_BY_DEFAULT}; re-cut it with "
        "recut_driver_dispatch_record.py --backend cuda")


def test_the_cited_driver_route_gate_ran_the_bytes_the_record_binds():
    """THE THREE DIGESTS HAVE TO BE ONE DIGEST: artifact, record, tree.

    THE DEFECT THIS EXISTS FOR, measured 2026-08-29 and not hypothetical. Three
    different sha256s for ``meep_gpu/fastpath.py`` were live at once: 87f740ee
    shipped, ``fingerprints.json['driver_dispatch']['source_sha256']`` recorded
    06cdbf37, and all four legs of the gate this file cites as its release
    authority had actually executed 29ac1216. The test above compares the RECORD
    against the TREE, which is one of the three pairs; nothing compared either
    against the ARTIFACT, so the release could go on citing a run that had never
    executed the code citing it. That is the whole of the defect — the drift was
    executable, not prose.

    So this reads the artifact the release names, leg by leg, and requires every
    leg to have run the bytes the record binds and the tree ships. A stale
    citation is fixed by RE-RUNNING the gate against the current bytes; there is
    no digest here to edit, because every digest is read rather than written.

    THE RESULTS TREE IS GITIGNORED, so a clone has no artifacts and this skips.
    It skips on the TREE's absence, never on the artifact's: a checkout that has
    results but not the run the release names is exactly the state the defect
    left behind, and it must fail rather than skip.
    """
    import hashlib
    import json
    import pathlib

    package = pathlib.Path(fastpath.__file__).resolve().parent
    results = package.parent / "parity" / "meep_gpu" / "results"
    if not results.is_dir():
        pytest.skip("parity/meep_gpu/results is gitignored and absent here")

    # ONE ROUTE CAMPAIGN PER ADMITTED CAPABILITY. The release constant is a STAMP; the
    # campaign a capability's record cites is ``fastpath.route_campaign(stamp, cc)``
    # (``<stamp>_cc86``), the directory the recut refuses to file under any other
    # name, and the one the record's own slot names. Every capability the table
    # admits is read, so an architecture certified without a route run fails here.
    record = json.loads((package / "triton_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))["driver_dispatch"]
    bound = record["source_sha256"]
    admitted = fastpath.capability_admission("triton")["admitted"] or ()
    assert admitted, "the Triton table admits no compute capability to read a run for"
    for capability in admitted:
        campaign = fastpath.route_campaign(fastpath.DRIVER_ROUTE_FUSED_GATE, capability)
        slot = (record.get(fastpath.RUNS) or {}).get(capability) or {}
        assert (slot.get("released_fused_arms") or {}).get("gate") == campaign, (
            f"runs[{capability!r}] cites "
            f"{(slot.get('released_fused_arms') or {}).get('gate')!r}; the release "
            f"constant spells {campaign!r} for this capability")
        run = results / campaign
        assert run.is_dir(), (
            f"DRIVER_ROUTE_FUSED_GATE spells {campaign} for compute capability "
            f"{capability}, which is not a directory under {results}. The release "
            f"cites an artifact that is not here.")
        legs = sorted(p for p in run.glob("*/gate.json"))
        assert len(legs) >= 4, (
            f"{campaign} carries {len(legs)} legs; the gate is four (shipped, "
            f"harness_keep, flush, shipped_expansion_probe)")
        for leg in legs:
            artifact = json.loads(leg.read_text(encoding="utf-8"))
            assert artifact["release"]["released"] is True, (
                f"{leg.parent.name}: released={artifact['release']['released']} "
                f"{artifact['release']['reasons']}")
            ran = artifact["provenance"]["source_sha256"]
            for name, digest in bound.items():
                live = hashlib.sha256((package / name).read_bytes()).hexdigest()
                assert ran.get(name) == digest == live, (
                    f"{leg.parent.name} ran {name} at {ran.get(name)}, the record binds "
                    f"{digest}, the tree ships {live}. Re-run the gate against the "
                    f"bytes that ship; do not type a digest in.")
        # AND THE CASE LIST IS THE RUN'S OWN. RELEASED_FUSED_ARMS names the cases each
        # arm was driven on; a case no leg drove would be provenance for a measurement
        # nobody took. The record's driven_on_cases is checked against the constant
        # above; this checks the ARTIFACT, which is the thing that did the driving.
        # A CASE WHOSE LICENCE LIVES ON ANOTHER LEG IS NOT THIS LEG'S EVIDENCE (phase B,
        # 2026-09-13): the complex, beta-complex, folded-complex and cylindrical-complex
        # arms take their expansion licence from the unified probe record, which ONE
        # leg exports, so every other leg records PASS-NOT-THIS-LEG for their cases --
        # the gate's own spelling, read off the artifact rather than off a table here.
        # What is still required of EVERY (arm, case) is that some leg drove it.
        driven_somewhere: set = set()
        for leg in legs:
            artifact = json.loads(leg.read_text(encoding="utf-8"))
            if artifact.get("uncertified_policy_installed"):
                continue          # this leg's licence is rung 8b refusing, not a dispatch
            driven = {case: set((arms or {}).values())
                      for case, arms in (artifact.get("arms_driven") or {}).items()}
            verdicts = artifact.get("verdicts") or {}
            for arm, cases in fastpath.RELEASED_FUSED_ARMS.items():
                for case in cases:
                    if verdicts.get(case) == "PASS-NOT-THIS-LEG":
                        continue
                    assert arm in driven.get(case, set()), (
                        f"{leg.parent.name}: RELEASED_FUSED_ARMS says {arm!r} was "
                        f"driven on {case!r}, and this leg's artifact does not show it")
                    driven_somewhere.add((arm, case))
        for arm, cases in fastpath.RELEASED_FUSED_ARMS.items():
            for case in cases:
                assert (arm, case) in driven_somewhere, (
                    f"RELEASED_FUSED_ARMS says {arm!r} was driven on {case!r}, and no "
                    f"leg of {campaign} shows it")


def test_every_recorded_certification_points_at_a_readable_record():
    """EVERY arm, with no exemption — the exemption was the defect.

    This used to ``continue`` past the nine families certified 2026-08-14 because
    their gate was a path into a gitignored results directory, which meant those
    nine reached a user's artifact with no ``recorded_utc``, no host and no run id
    while the eight legacy families carried all three. The record is transcribed
    into ``fingerprints.json`` now, so the exemption is gone and the assertion is
    what stops it coming back.

    ASKED ON ONE ARCHITECTURE, because that is where a run record now lives: the
    three facts are the certifying RUN's, and the run is the one keyed by
    :data:`CERTIFIED_CAPABILITY`. ``capabilities_live`` is asserted beside them so a
    weld whose record stopped binding the shipped bytes cannot answer here by
    carrying a run for an architecture the entry no longer certifies.
    """
    for arm, (family, gate) in fastpath.ARM_CERTIFICATION.items():
        assert family and gate, arm
        entry = fastpath._certification_for(arm, capability=CERTIFIED_CAPABILITY)
        assert CERTIFIED_CAPABILITY in entry["capabilities_live"], (
            f"{arm} names {gate}, whose records bind bytes that have since moved: "
            f"live capabilities {entry['capabilities_live']}")
        assert "recorded_utc" in entry, f"{arm} names {gate}, which fingerprints.json lacks"
        assert entry["host"], arm
        assert entry["step_budget"], arm


def test_the_no_absorber_arms_are_device_certified_and_no_longer_pending():
    """The three no-absorber arms are certified by live runs of their own gates.

    THE BUDGET IS READ FROM THE RUN'S ARTIFACT, because a re-bind does not carry it.
    The ``step_budget`` sentence ("8 launches per row; ...") was hand-curated from the
    2026-08-19 run, and ``rebind_triton_welds.py`` carries a curated run field only
    while the record it replaces still binds the bytes: a run on moved bytes drops it
    rather than restating a narrative about another tree. What the sentence stated is
    in every one of these gates' artifacts as numbers -- ``cycles`` at the top level
    and on each case -- so it is asserted there, from the artifact the run record
    names and ``artifact_sha256`` pins. A record that still carries the sentence must
    still say it; the artifact half skips only on the results tree's absence.
    """
    import hashlib

    expected = {
        "complex no-PML curl": "triton_complex_no_pml_curl_device_gate",
        "conductive no-PML": "triton_no_pml_conductive_device_gate",
        "no-PML stored E": "triton_no_pml_stored_e_device_gate",
    }
    runs = {}
    for arm, gate in expected.items():
        assert arm not in fastpath.PENDING_DEVICE_GATE_ARMS
        entry = fastpath._certification_for(arm, capability=CERTIFIED_CAPABILITY)
        assert entry["gate"] == gate
        assert entry["capability"] == CERTIFIED_CAPABILITY
        assert "A6000" in entry["host"]
        # NOT a fixed date. The intent is "certified no EARLIER than the
        # no-absorber closure", i.e. a weld may never silently revert to an
        # older run — but a legitimate RE-certification must not fail here.
        # Measured 2026-08-19: pinning the exact day made re-running the three
        # gates (to record their own provenance) look like a regression, which
        # pressures the next person to edit the record instead of re-running.
        assert entry["recorded_utc"] >= "2026-08-17T", entry["recorded_utc"]
        assert entry["recorded_utc"].endswith("Z"), "ISO-8601 UTC expected"
        run = fastpath._fingerprints("triton")[gate][fastpath.RUNS][CERTIFIED_CAPABILITY]
        if "step_budget" in run:
            assert "8 launches per" in run["step_budget"], (arm, run["step_budget"])
        runs[arm] = run
    package = pathlib.Path(fastpath.__file__).resolve().parent
    _results_tree_or_skip(package, "the gates' artifacts state the launches per row")
    for arm, run in sorted(runs.items()):
        artifact = _run_artifact(package, run)
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == run["artifact_sha256"], (
            f"{arm}: {artifact} is not the artifact the run record pins")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        cases = payload.get("cases") or []
        assert payload.get("cycles") == 8 and cases, (arm, payload.get("cycles"), len(cases))
        assert all(case.get("cycles") == 8 for case in cases), (
            f"{arm}: per-case launches {sorted({case.get('cycles') for case in cases})}")
        # The requested count above, and the launches each row actually recorded: one
        # ``per_cycle`` measurement per launch, so a row that stopped early (or ran
        # extra) reads differently from what it asked for.
        assert all(len(case.get("per_cycle") or ()) == 8 for case in cases), (
            f"{arm}: per-case recorded launches "
            f"{sorted({len(case.get('per_cycle') or ()) for case in cases})}")


def test_the_residual_closure_arms_are_explicitly_fail_closed_until_recertified():
    """THE SUB-STEP ARMS half of the pending map, kept exact.

    The map gained a SECOND population on 2026-09-02 — the installer wave's fused
    LABELS, which are pending for a different reason (their gate released, and no
    ledger entry has been cut from it) and are pinned by the test below. The two
    are told apart by ``arm_is_fused``, so neither can grow into the other's
    allowlist unnoticed.
    """
    # THREE LEFT THIS SET ON 2026-09-14, BY A RELEASE DECISION RATHER THAN A NEW RUN.
    # `complex ADE update_P`, `complex conductive no-PML curl` and
    # `complex no-PML stored E` were held here on a reason that had gone stale: all
    # three family gates DID run and release on 2026-09-13 and are in
    # triton_kernels/fingerprints.json with status PASS and no source drift
    # (triton_complex_ade_device_gate, triton_complex_no_pml_conductive_device_gate,
    # triton_complex_no_pml_stored_e_device_gate, recorded 2026-09-13T07:53:3xZ on
    # the GPU host). What the table's own rule additionally asks for -- "the central
    # planner and driver seam must also be recertified against the bytes that
    # contain the arm" -- was ruled satisfied on the grounds that the
    # driver half is re-cut from the route run, and that the only planner record
    # (planner_integration_recert_2026-08-14) is hand-authored, predates these
    # gates, and carries a `dispatch` field that is itself stale. THE ARMS THEREFORE
    # MOVE TO ARM_CERTIFICATION, and the disjointness assertion below now covers the
    # three that remain -- which is what keeps this test honest about the three it
    # no longer names.
    #
    # A FOURTH LEFT ON 2026-09-15, ON THE SAME RULING EXTENDED BY NAME:
    # `complex folded off-diagonal`, the update_E arm (not `folded complex
    # off-diagonal`, the update_H admission, which was never here). Its family gate
    # is triton_complex_offdiag_device_gate (batch_D, 2026-09-13: 5 of 5 products
    # IDENTICAL, 5 of 5 mutations DIVERGENT), and the lift was conditioned on that
    # gate being re-run under a recorded keep policy and the key rebound in the same
    # batch -- which the test below holds the ledger to. Two remain here.
    expected = {
        "complex no-PML off-diagonal",
        "folded off-diagonal dispersive",
    }
    arms = {arm for arm in fastpath.PENDING_DEVICE_GATE_ARMS
            if not fastpath.arm_is_fused(arm)}
    assert arms == expected
    assert expected.isdisjoint(fastpath.ARM_CERTIFICATION)
    assert all(fastpath.PENDING_DEVICE_GATE_ARMS[arm] for arm in expected)


def test_the_complex_folded_offdiag_arm_is_certified_on_a_rebound_weld():
    """The 2026-09-15 lift, held to the two conditions it was ruled on.

    ``complex folded off-diagonal`` (update_E) left PENDING_DEVICE_GATE_ARMS on an
    release decision whose licence was conditional: its ledger entry's launch.py pins had
    drifted and its ``subnormal_policy`` read "see artifact". The lift therefore
    counts only once ``gate_triton_complex_offdiag.py`` has been re-run with its
    policy stamp and ``rebind_triton_welds.py`` has rebound the key from that run.
    ``recorded_utc`` is when the rebind cut the entry, so a floor at the ruling date
    says the rebind happened; a policy that names ``keep`` says the stamp reached the
    ledger. Both are red between the source edit and the rebind, by design.

    THE NEAR-HOMONYM IS PINNED BESIDE IT. ``folded complex off-diagonal`` is the
    update_H admission mapped to the complex family since 2026-08-16; a lift that
    edited that label instead would leave this arm pending and pass review by eye.

    BOTH CONDITIONS ARE THE RUN'S, so both are read out of the run record for
    :data:`CERTIFIED_CAPABILITY` rather than off the entry: the rebind writes one
    record per architecture under ``fastpath.RUNS``, and it is that record's date and
    policy stamp that say the re-run happened on the card this arm dispatches on.
    """
    import json
    import pathlib

    arm = "complex folded off-diagonal"
    assert arm not in fastpath.PENDING_DEVICE_GATE_ARMS
    assert fastpath.ARM_CERTIFICATION[arm] == (
        "complex_offdiag", "triton_complex_offdiag_device_gate")
    assert fastpath.ARM_CERTIFICATION["folded complex off-diagonal"] == (
        "complex", fastpath.FAMILY_RECERT_GATE)
    assert "folded complex off-diagonal" not in fastpath.PENDING_DEVICE_GATE_ARMS

    ledger = json.loads((pathlib.Path(fastpath.__file__).parent / "triton_kernels"
                         / "fingerprints.json").read_text(encoding="utf-8"))
    entry = ledger["triton_complex_offdiag_device_gate"]
    assert entry["status"] == "PASS", entry["status"]
    assert CERTIFIED_CAPABILITY in fastpath.live_capabilities(entry), (
        f"triton_complex_offdiag_device_gate records no live run on "
        f"{CERTIFIED_CAPABILITY}: {fastpath.live_capabilities(entry)}")
    run = entry[fastpath.RUNS][CERTIFIED_CAPABILITY]
    assert run["recorded_utc"] >= "2026-09-15T", (
        f"triton_complex_offdiag_device_gate was recorded {run['recorded_utc']}, "
        "before the ruling that lifted its arm; re-run gate_triton_complex_offdiag.py "
        "and rebind with parity/meep_gpu/rebind_triton_welds.py")
    assert re.search(r"\bkeep\b", str(run["subnormal_policy"])), (
        f"triton_complex_offdiag_device_gate names no policy "
        f"({run['subnormal_policy']!r}); the lifted arm dispatches on this weld")


def test_every_pending_fused_label_is_one_the_composer_installs():
    """THE FUSED half, and it is a partition rather than a list.

    A fused label is pending here for ONE reason again, which is where this
    docstring started and where the re-attribution round returns it: the composer
    can select the label and no ``fingerprints.json`` entry has been cut from the
    gate that released it, so a dispatched run could not name the record that
    certified its bytes.

    THE SECOND REASON LASTED FROM 2026-09-11 TO THE RE-ATTRIBUTION ROUND and is
    worth keeping in the record. ``fused pair D (folded dispersive)`` was pending on
    a MEASUREMENT rather than on a missing entry: its case drove, dispatching the
    arm at 7 of 7 slots with every ``step()`` checkpoint reading fused==array to
    1600 steps, and the route gate's SECOND consult site
    (``synchronize_magnetic_fields``) was read as the kernel and array paths
    disagreeing at that half-step with fusion OFF as well as on. Re-driven three
    times in fresh processes, that site reads PASS with both comparisons byte-clean,
    and the signature was attributed to the lazily installed subnormal policy — the
    ARRAY leg flushing subnormals the dispatch legs keep — which no arm can cause
    and no ledger entry can excuse. The label is released now; the RULE that a
    measurement can hold a label whose ledger entry exists is unchanged and this
    test still expresses it, because the partition below is read off the composer's
    table rather than typed.

    Read in BOTH directions — a pending label no composer installs would be a
    refusal about nothing, and an installed label in neither map could dispatch
    while naming no gate at all.
    """
    products = launch_module.CERTIFIED_FUSED_PRODUCTS
    installed = {spec["label"] for spec in products.values()}
    pending = {arm for arm in fastpath.PENDING_DEVICE_GATE_ARMS
               if fastpath.arm_is_fused(arm)}
    assert pending <= installed, sorted(pending - installed)
    assert pending == installed - set(fastpath.ARM_CERTIFICATION)
    # 20 / 23 -> 22 / 25 ON 2026-09-07: the two cylindrical H->D products routed
    # through the composer's table, both declaring INSTALLABLE = False (the flag is
    # named in each pending reason beside the missing ledger entry).
    # 22 / 25 -> 23 / 26 ON 2026-09-08: the Cartesian complex H->D product
    # (`complex_fused_hd_pair`, label `fused pair H->D (complex)`) routed through the
    # same table on the same terms — INSTALLABLE = False, no ledger entry yet.
    # 23 / 26 -> 21 / 26 ON 2026-09-11: nothing new was wired; two products LEFT
    # this map for ARM_CERTIFICATION when `dispatch_fused_route_2026-09-11_realarms`
    # released them and seed_triton_welds.py cut their ledger entries from that
    # fleet. The installed count is unchanged because the composer's table did not
    # move — releasing is not wiring, in the direction nobody had exercised yet.
    # It is 21 rather than 20 because `fused pair D (folded dispersive)` STAYED:
    # the same campaign's preflight measured its route-gate divergence at
    # `synchronize_magnetic_fields`, so it is held on this rung for a reason no
    # ledger entry answers.
    # 21 / 26 -> 6 / 26 ON 2026-09-13 (phase B): fifteen labels LEFT this map for
    # ARM_CERTIFICATION when the 2026-09-12 identity fleet's artifacts -- the probes
    # recording the device identity and the provenance stamp the weld contract
    # requires -- were seeded by seed_triton_welds.py and the route cases each
    # needed were driven on the GPU host. The six that stayed: the held folded
    # dispersive D pair, the two no-absorber D pairs, and the three uninstallable
    # H->D products.
    # 6 / 26 -> 5 / 26 ON THE RE-ATTRIBUTION ROUND: `fused pair D (folded
    # dispersive)` left for ARM_CERTIFICATION, and it is the one row that ever left
    # this map without a ledger entry being cut for it -- its entry
    # (`triton_folded_dispersive_fused_pair_device_gate`, PASS) had been cut on
    # 2026-09-13 and the hold was the divergence, which the re-drive attributed to
    # the subnormal-policy install ordering instead. The four that stay are the
    # complex no-absorber D pair, whose own pair weld has not been cut, and the
    # three uninstallable H->D products; `fused pair D (no-PML stored E)` left in
    # the target round, on the ledger entry its pending row said was missing.
    # 4 / 26 -> 3 / 26 ON 2026-09-15: the complex no-absorber D pair left for
    # ARM_CERTIFICATION with its release on `complex_no_pml_3d`; its pair weld is
    # cut by seed_triton_welds.py from the pair gate re-run after that edit. The
    # three that stay are the uninstallable H->D products.
    # 3 / 26 -> 3 / 28 ON 2026-09-17: the two scratch-output off-diagonal welds were
    # routed (their shared plan base gained a ``warm`` that never rotates) and their
    # labels went straight to ARM_CERTIFICATION, so the installed side grows by two
    # and the pending side does not move.
    assert len(pending) == 3 and len(installed) == 28
    # AND THE RELEASED HALF, WHICH USED TO BE "NONE OF THEM".
    #
    # This line read `assert not (installed & set(RELEASED_FUSED_ARMS))` from the
    # 2026-09-02 wiring wave until 2026-09-11, and what it was defending is worth
    # keeping rather than deleting: a wiring round must not be able to release a
    # product by adding a row to the composer's table. The intent is UNCHANGED and
    # the assertion is now the enumeration that says it — releasing three products
    # is a campaign, and a fourth appearing here without one fails on this line.
    #
    # THREE AND NOT FOUR ON 2026-09-11. The campaign drove a fourth product,
    # `fused pair D (folded dispersive)`, and its preflight refused it: the route
    # gate's second consult site, `synchronize_magnetic_fields`, read
    # fused_vs_array False AND unfused_vs_array False on a leg installing no fused
    # product, which was read at the time as the seam's divergence rather than the
    # kernel's. It stayed in PENDING_DEVICE_GATE_ARMS and out of this set.
    #
    # AND IT IS THE NINETEENTH NOW. Re-driven three times in fresh processes over
    # the full 1600-step ladder, that site reads PASS with both comparisons
    # byte-clean and the armed nulls firing; the signature belonged to the lazy
    # subnormal-policy install (the ARRAY leg flushing subnormals the dispatch legs
    # keep), which is why BOTH legs appeared to diverge by the same amount and why
    # no mechanism spanning step_D/update_E could explain a magnetic half-step. The
    # label is released on `folded_dispersive_2d` and joins this enumeration.
    #
    # The second half is the invariant that used to be implied by the first: a
    # label may not be released and pending at once, which would be a plan claiming
    # a certification the pending rung says it lacks.
    # EIGHTEEN SINCE PHASE B (2026-09-13): the three above plus the fifteen labels
    # `dispatch_fused_route_2026-09-13_phaseB` drove on their seeded welds.
    # TWENTY-TWO AFTER THE TARGET AND RE-ATTRIBUTION ROUNDS: `fused pair D (no-PML
    # stored E)`, released by the target round on `absorber_1d` and
    # `material_dispersion_0d` against a ledger entry that already existed, and
    # `fused pair D (folded dispersive)`, released here. Both rows say in
    # RELEASED_FUSED_ARMS which half of their evidence the succeeding route
    # campaign still owes; this enumeration is about the released SET and not about
    # what each row banked.
    # TWENTY-THREE ON 2026-09-15: `fused pair D (complex conductive no-PML)`, released
    # on `complex_no_pml_3d` in the same batch as its ARM_CERTIFICATION row.
    # TWENTY-FIVE ON 2026-09-17: `fused pair D (off-diagonal)` and `fused pair D
    # (folded off-diagonal)`, the two scratch-output welds, routed and released in
    # one batch on `offdiag_magnetic_2d` / `folded_offdiag_magnetic_2d`. Their drive
    # with the weld installed is what `DRIVER_ROUTE_FUSED_GATE`'s campaign owes.
    assert (installed & set(fastpath.RELEASED_FUSED_ARMS)) == {
        "fused pair D (off-diagonal)", "fused pair D (folded off-diagonal)",
        "fused pair B (cylindrical)", "fused pair D (cylindrical)",
        "fused pair D (conductive)", "fused pair D (folded dispersive)",
        "fused pair D (no-PML stored E)", "fused pair D (complex conductive no-PML)",
        "fused pair B (real beta)", "fused pair D (real beta)",
        "fused pair B (folded real beta)", "fused pair D (folded real beta)",
        "fused pair B (complex beta)", "fused pair D (complex beta)",
        "fused pair B (folded complex beta)", "fused pair D (folded complex beta)",
        "fused pair B (BFAST)", "fused pair D (BFAST)",
        "fused pair B (nonlinear)",
        "fused pair B (complex)", "fused pair D (complex)",
        "fused pair B (folded complex)", "fused pair D (folded complex)",
        "fused pair B (cylindrical complex)", "fused pair D (cylindrical complex)"}
    assert not (pending & set(fastpath.RELEASED_FUSED_ARMS))


def test_the_no_absorber_gate_records_are_welded_to_the_live_sources():
    """Every entry that PINS bytes, not every entry that also says ``status``.

    ENUMERATED, NOT LISTED. This test named three gates by hand, and the six
    welds added on 2026-08-19 were therefore ORPHANS: they recorded a
    source_sha256 that nothing recomputed, so editing a welded kernel left the
    record asserting a device result about bytes that no longer shipped — the
    exact failure a weld exists to prevent, silently, on two thirds of them. A
    hand-maintained list is a second place to forget a weld; the record IS the
    list.

    ENUMERATING IT WAS NOT ENOUGH, and that is the 2026-08-30 correction. The
    enumeration still carried a filter — ``status == "PASS"`` — and the record
    holds ten entries with a complete device-gate record and no ``status`` key at
    all. They were exactly as unchecked as the six orphans had been, on 88 more
    pinned pairs. A filter on a field a gate record need not carry is the same
    defect as a hand-maintained list, wearing an enumeration's clothes. The
    denominator is now every pinned pair, and it is asserted, so shrinking it
    fails here.

    The one-home rule is why this imports ``_resolve``/``_source_pins`` from the
    weld contract instead of re-deriving them: the pre-2026-08-19 entries pin a
    BARE BASENAME, and two spellings of "where does this pin point" is two places
    to disagree about which file a digest describes.

    IT WAS RED FROM 2026-08-30 TO 2026-08-31, ON ONE ENTRY —
    ``dispersive_composition_gate``'s ``__init__.py`` — AND THAT IT WAS RED HERE
    WAS INFORMATIVE. This test carries the ``weld_survives_edit`` allowance and the
    weld contract deliberately does not, so the two can disagree: a comment-only
    edit is drift there and harmless here. That one failed in BOTH, which is the
    allowance saying, on its own terms, that the change reached executable code.
    ``triton_kernels/__init__.py`` had moved from ``33d9ae4ecaca`` to
    ``017f710e0002`` since that gate's 2026-08-11 run.

    IT WAS NOT CLEARABLE BY A REBIND, and what actually cleared it is written
    where the drift is measured: ``test_triton_weld_contract.py::
    test_the_triton_welds_are_bound_to_the_live_sources``. In short — the gate DID
    re-run on 2026-08-30, and ``probe_triton_engine_route.py`` wrote a payload with
    no ``canonical_verdict``, no ``summary.status`` and no
    ``imported_source_sha256``, so the run neither claimed a release nor recorded
    which bytes it read. Re-pinning from it would have been reading a verdict into
    a payload that declines to state one. On 2026-08-31 the probe gained a stated
    release condition and a ``gate_provenance`` stamp, its
    ``refusals_returned_none`` 3/4 was adjudicated against the engine rather than
    papered over (``2d_cond_pml``'s aggregate verdict and its ``step_B`` verdict
    disagree BY DESIGN since the conductive PML family landed), it re-ran on
    the GPU host and released, and ``recut_composition_records.py`` bound the entry
    with both moved counters declared in ``CLAIM_RECUTS``.

    THIS TEST IS THEREFORE GREEN AND THE WELD CONTRACT IS NOT: that one still
    reports 2 of 314 raw-tier digests in drift, both on ``bit_identity_gate``,
    whose two pins are HARNESS files and which no binder in the tree can reach.
    The two tests disagreeing here is not a bug in either — this one walks the
    no-absorber gate records and ``bit_identity_gate`` is not among them.
    """
    import hashlib

    from .test_triton_weld_contract import (
        SOURCE_PIN_FLOOR, UNBOUND_PLACEHOLDER, _resolve, _source_pins)

    records = fastpath._fingerprints()
    pins = _source_pins(records)
    owners = {owner for owner, _, _ in pins}
    assert len(pins) >= SOURCE_PIN_FLOOR, (
        f"only {len(pins)} (entry, path) pairs discovered across {len(owners)} "
        f"entries, was {SOURCE_PIN_FLOOR} — coverage SHRANK, and an entry does "
        f"not leave this check by dropping a key")

    for gate, name, digest in pins:
        entry = records[gate]
        assert digest != UNBOUND_PLACEHOLDER, (
            f"{gate}: {name} is pinned but unbound — run "
            f"parity/meep_gpu/rebind_triton_welds.py")
        path = _resolve(name)
        assert path.is_file(), (
            f"{gate}: welded path {name} no longer exists in the tree")
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if live == digest:
            continue
        if weld_survives_edit(path, entry, name):
            continue
        assert live == digest, (
            f"{gate}: {name} drifted from the bytes the gate executed and the change "
            f"reaches executable code")


def test_a_future_pending_arm_still_refuses_the_whole_plan(monkeypatch):
    monkeypatch.setattr(
        fastpath, "PENDING_DEVICE_GATE_ARMS",
        {"synthetic pending arm": "its dedicated byte gate has not run"})
    plan = plan_on_a_cupy_host(
        monkeypatch,
        step_plan({"step_B": CountingPlan("b")},
                  {"step_B": "synthetic pending arm"}))
    assert plan is None
    assert "dedicated byte gate has not run" in \
        fastpath.last_dispatch_report()["refused_because"]


def test_a_null_arm_is_dropped_before_completeness_is_judged(monkeypatch):
    """C4: both sides of a null slot are empty, so dropping it changes nothing."""
    b = CountingPlan("step_B")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": b, "update_H": CountingPlan("null")},
        {"step_B": "PML", "update_H": fastpath.NULL_ARM_LABEL}))
    assert plan is not None
    assert plan.slots == ("step_B",)
    assert plan.dropped_null == {"update_H": fastpath.NULL_ARM_LABEL}
    assert plan.report()["slots"]["update_H"]["dropped_as_null"] == fastpath.NULL_ARM_LABEL


def test_a_plan_of_nothing_but_nulls_is_not_a_fused_step(monkeypatch):
    """The finding this closes: a NumPy laptop reading "fused" off the null arm."""
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"update_H": CountingPlan("nh"), "update_E": CountingPlan("ne")},
        {"update_H": fastpath.NULL_ARM_LABEL,
         "update_E": fastpath.NULL_ARM_LABEL}))
    assert plan is None
    assert "no slot is left carrying a kernel" in fastpath.last_dispatch_report()["refused_because"]


@pytest.mark.parametrize("label", ["fused pair B", "fused pair D",
                                   "fused pair B (folded)", "fused pair D (folded)",
                                   "dispersive fused pair", "fused ADE state"])
def test_a_fused_arm_refuses_the_whole_plan(monkeypatch, label):
    """C5. A fused product refuses unless a gate RELEASED it here, or this
    process opted into THAT LABEL.

    BOTH DIRECTIONS, since 2026-08-29. The one-directional form of this test
    ("a fused arm always refuses") was true of a ladder whose composer call
    passed ``fuse=False`` as a literal, and it would have stayed green against a
    build where the opt-in admitted everything — it never asked what happens when
    an arm IS opted into. It now measures the refusal AND the admission, so a
    widening of clause (8) fails here rather than showing up on a device.

    THE GRID IS THE THIRD DIRECTION and it is why this leg still measures a
    refusal for every one of the six labels. ``cupy_like_grid`` carries no shape
    at all, so it is outside :data:`fastpath.FUSED_RELEASE_ENVELOPE` and the
    release does not reach it — including for the three arms that ARE released.
    ``test_a_released_fused_arm_dispatches_only_inside_the_envelope`` runs the
    same six labels on the configuration the gate drove; between them the two
    tests pin the released population and its complement, on both sides of the
    envelope.
    """
    assert fastpath.arm_is_fused(label)
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": label}))
    assert plan is None
    refusal = fastpath.last_dispatch_report()["refused_because"]
    assert "fused" in refusal and label in refusal

    # The other direction: opted into by name, the same plan dispatches.
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, label)
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": label}))
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = fastpath.last_dispatch_report()
    assert record["fusion"]["opted_in"] == [label]
    assert record["fusion"]["driven"] == {"step_B": label}

    # And the opt-in is PER LABEL: a different name admits nothing.
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "some other fused pair")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": label}))
    assert plan is None
    assert label in fastpath.last_dispatch_report()["refused_because"]


def test_a_second_fused_arm_riding_along_refuses_the_whole_step(monkeypatch):
    """Opting into one label does not admit a different one in the same step.

    "Each half was measured" is not "the step was measured", and the cheap bug
    here is an opt-in read as a boolean: one name in the variable turning every
    fused product in the plan on.
    """
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "fused pair B")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b"), "step_D": CountingPlan("d")},
        {"step_B": "fused pair B", "step_D": "fused pair D"}))
    assert plan is None
    refusal = fastpath.last_dispatch_report()["refused_because"]
    assert "Not admitted: fused pair D" in refusal, refusal
    # AND THE REASON IS NAMED, not merely the label. ``fused pair D`` IS a
    # released arm, so a reader who sees it refused has to be told that what is
    # missing is the CONFIGURATION and not the arm — otherwise the refusal reads
    # as a contradiction of the release list two lines above it in the record.
    assert fastpath.DRIVER_ROUTE_FUSED_GATE in refusal, refusal
    # WHERE THE AXIS IS NAMED MOVED IN THE 2026-09-14 TARGET ROUND, so the assertion
    # moves with it rather than being dropped. The refusal STRING quotes the SHARED
    # half of the release only (fastpath's "... is released only for the
    # configuration that gate drove, and this one is outside it: ..." clause), and
    # that half is now EMPTY: ``pml_active`` was its last row and the round put it on
    # all twenty-seven arms individually, because ``fused pair D (no-PML stored E)``
    # is released on two absorber-free cells and one shared row cannot say True and
    # False at once. So on a shape that READS there is nothing left for the string to
    # quote, and the axis that refuses a released arm is named in the record's
    # per-arm half — which fastpath reports precisely WHEN the shared half is silent.
    # The fact being pinned is unchanged: this stub's configuration refuses ``fused
    # pair D`` on the very axis it was refused on before the move.
    why = fastpath.last_dispatch_report()["fusion"]["outside_this_arms_own_cases"]
    assert any("pml_active=False" in reason for reason in why["fused pair D"]), why

    # AND THE STRING'S OWN CLAUSE IS STILL MEASURED, on the one configuration that
    # still fires the shared half: a run shape that will not read. That branch is
    # what an empty ``FUSED_RELEASE_ENVELOPE`` leaves it — a refusal about the
    # RECORD rather than about the configuration — and it is the half a reader of
    # ``refused_because`` alone depends on, so it is driven rather than assumed.
    unreadable = certified_envelope_grid()
    unreadable.shape = object()          # ``tuple(...)`` in _run_shape raises on it
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "fused pair B")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b"), "step_D": CountingPlan("d")},
        {"step_B": "fused pair B", "step_D": "fused pair D"}),
        fields=certified_envelope_fields(), grid=unreadable,
        pml=certified_envelope_pml())
    assert plan is None
    refusal = fastpath.last_dispatch_report()["refused_because"]
    assert "Not admitted: fused pair D" in refusal, refusal
    assert "outside it" in refusal and "the run shape did not read" in refusal, refusal

    # THE OTHER DIRECTION, on the configuration the gate drove: an unreleased
    # label riding along still refuses the whole step even though both released
    # arms in the same plan would have dispatched on their own.
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    plan = plan_inside_the_released_envelope(monkeypatch, step_plan(
        {"step_B": CountingPlan("b"), "step_D": CountingPlan("d")},
        {"step_B": "fused pair B", "step_D": "fused pair B (folded)"}))
    assert plan is None
    refusal = fastpath.last_dispatch_report()["refused_because"]
    not_admitted = refusal.split("Not admitted: ")[1].split(" (released by")[0]
    assert not_admitted == "fused pair B (folded)", refusal


def test_the_composer_is_asked_for_an_unfused_plan(monkeypatch):
    """C5's other half: fusion is refused by NOT ASKING as well as by checking.

    BOTH DIRECTIONS. Unset, the composer is asked for exactly the composition the
    old ``fuse=False`` literal asked for; opted in, it is asked for the fused one
    — otherwise the switch would be a refusal the composer never heard, and the
    driver-route gate would measure a fallback while reading as a fused run.
    """
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    calls = install_composer(monkeypatch, step_plan())
    stub_triton(monkeypatch)
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())
    assert calls and calls[0]["fuse"] is False and calls[0]["fuse_ade"] is False

    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "fused pair B")
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())
    assert calls[-1]["fuse"] is True and calls[-1]["fuse_ade"] is False

    # ``fused ADE state`` rides the OTHER composer argument. A switch that drove
    # only ``fuse`` would answer this one silently and dispatch nothing.
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, fastpath.FUSED_ADE_STATE_LABEL)
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())
    assert calls[-1]["fuse"] is False and calls[-1]["fuse_ade"] is True

    # THE RELEASE ASKS TOO, and this is the half that would silently do nothing if
    # the release were consulted only when judging the answer: on the
    # configuration the gate drove, an unset switch asks for the fused
    # composition. ``fuse_ade`` stays False because ``fused ADE state`` is not
    # released — it rides the other composer argument and no gate has driven it.
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    stub_triton(monkeypatch)
    fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                            certified_envelope_grid())
    assert calls[-1]["fuse"] is True and calls[-1]["fuse_ade"] is False


def test_the_real_composer_accepts_every_argument_the_dispatcher_passes(monkeypatch):
    """The one call in this seam that every other test in this file STUBS OUT.

    ``install_composer`` replaces ``triton_kernels.plan_step`` with a recorder, so
    every assertion about what the dispatcher asks for is made against a function
    that accepts anything. The real composer is TWO functions — a package wrapper
    that forwards POSITIONALLY into ``launch.plan_step`` — and a keyword the wrapper
    does not name is a ``TypeError`` that ``plan_fast_path`` catches and reports as
    a one-line "internal error" inside a refusal.

    MEASURED, 2026-09-02, which is why this test exists rather than a comment: a
    driver-route campaign on the GPU host reported ``DID-NOT-FUSE`` on all four of its
    cases and ``ENVELOPE-DID-NOT-HOLD``, because ``fuse_labels`` had been added to
    ``launch.plan_step`` and to the dispatcher and not to the wrapper between them.
    Every local test passed. The signature is bound rather than called: binding is
    what fails, and it needs no Triton, no CuPy and no grid.
    """
    import inspect

    import meep_gpu.triton_kernels as package
    from meep_gpu.triton_kernels import launch as launch_real

    # Exactly the keywords ``_decide`` names at its one composer call site, read off
    # the source so a new argument there fails here instead of on a device.
    source = pathlib.Path(fastpath.__file__).read_text(encoding="utf-8")
    call = source.split("triton_kernels.plan_step(", 1)[1].split(")\n", 1)[0]
    passed = set(re.findall(r"(\w+)=(?!=)", call))
    assert {"fuse", "fuse_ade", "fuse_labels", "sources", "probe"} <= passed, passed

    for function in (package.plan_step, launch_real.plan_step):
        parameters = inspect.signature(function).parameters
        missing = sorted(name for name in passed if name not in parameters)
        assert not missing, (
            f"{function.__module__}.plan_step does not accept {missing}, which "
            "plan_fast_path passes; the TypeError would surface as an 'internal "
            "error' refusal and every case would read DID-NOT-FUSE")

    # AND THE WRAPPER MUST FORWARD IT, not merely accept it. A positional forward
    # that dropped the new argument would pass the signature check above and
    # silently install every fused product the table can build.
    seen = {}
    monkeypatch.setattr(launch_real, "plan_step",
                        lambda *args, **kwargs: seen.update(
                            {"args": args, "kwargs": kwargs}) or step_plan())
    package.plan_step(object(), object(), fuse=True,
                      fuse_labels=("fused pair B",))
    assert ("fused pair B",) in seen["args"] or \
           seen["kwargs"].get("fuse_labels") == ("fused pair B",), seen


def test_the_composer_is_not_asked_to_fuse_outside_the_released_envelope(monkeypatch):
    """THE ORDER IS THE CONTRACT: the envelope gates the REQUEST, not the answer.

    Asking the composer to fuse on a configuration the release does not cover and
    then refusing the result at clause (8) would turn a run that dispatches its
    separate arms today into a whole-plan refusal — the fused arm wins the slot,
    and clause (8) refuses the PLAN, not just the fusion. So an unreleased
    configuration must be asked for exactly the composition it is asked for today.

    Measured on every axis of the envelope, one at a time, so a single axis
    dropped from the table fails here.
    """
    calls = install_composer(monkeypatch, step_plan())
    stub_triton(monkeypatch)
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)

    # THE SHARED TABLE IS EMPTY SINCE THE 2026-09-14 TARGET ROUND, and this block is
    # what that move rewrote. It used to open with the shared half: one axis here
    # refused EVERY released arm, so the composer was asked for nothing fused at all.
    #
    # ``cylindrical`` LEFT THIS DICT ON 2026-09-11 and is measured per-arm below.
    # It could not stay: a shared row that refuses every arm is exactly what the
    # two Dcyl products require the OPPOSITE of, and leaving it here would have
    # asserted that a cylindrical grid empties the request — which is now the
    # measurement of one shape where two arms dispatch.
    #
    # ``pml_active`` WAS THE LAST ROW IN IT AND LEFT ON THE SAME RULE, which empties
    # :data:`fastpath.FUSED_RELEASE_ENVELOPE` outright: ``fused pair D (no-PML stored
    # E)`` is released on absorber_1d and material_dispersion_0d — cells with no
    # split-field PML at all — so the shared row would have had to say True and
    # False at once. SO NO AXIS EMPTIES THE FUSION REQUEST BY ITSELF ANY MORE, and
    # asserting that one did would now be asserting something false. The two halves
    # of what this block used to say are both still measured, one axis over and one
    # shape over: ``pml_active`` joins the moved axes below, where an axis is
    # measured for what it LEAVES ADMITTED; and the request emptying — the half the
    # docstring is about — is measured at the bottom of this test on the five-pole
    # fold, the one shape in this file that admits nothing at all.
    assert fastpath.FUSED_RELEASE_ENVELOPE == (), (
        "a shared row is back; it needs its own emptied-request measurement here")
    # THE SEVEN AXES THAT LEFT THE SHARED TABLE (six on 2026-09-13, ``pml_active``
    # in the target round) are per-arm, so each is measured for what it leaves
    # admitted -- asked of the shipped predicate, never assumed -- and every released
    # arm it refuses must NAME the axis. Three of them (beta, complex storage and
    # now pml_active) carry a released product on this stub's 2-D grid, so the
    # composer IS asked there; the other four leave nothing admitted on it (BFAST is
    # released at 3-D and the nonlinear pair at 1-D, off-diagonal on nothing).
    # EVERY EXPECTATION HERE IS WRITTEN OUT BY HAND, and that is the whole value of
    # this loop rather than an inconvenience. An earlier repair of this test read
    # ``expected`` off ``arms_released_on`` — the function under test — and then
    # asserted the composer agreed with it. That version stays green under the exact
    # regression the loop exists to catch: widen an arm until it admits the BFAST
    # shape and the derived expectation widens with it, so nothing reddens. The
    # values below were MEASURED on 2026-09-14 and are pinned as constants, so a row
    # that quietly widens lands here as a mismatch instead of being ratified.
    #
    # Four axes still leave NOTHING admitted, which is the emptied-request contract
    # this test was written for, now carried per-arm instead of by a shared row.
    # Three leave exactly the arms whose own cases drove that value of the axis.
    moved = {
        "bloch": ("grid", "has_bloch", True, []),
        "beta": ("grid", "beta", 0.4,
                 ["fused pair B (real beta)", "fused pair D (real beta)"]),
        "bfast": ("grid", "bfast_active", True, []),
        "complex_storage": ("fields", "force_complex_fields", True,
                            ["fused pair B (complex beta)", "fused pair B (complex)",
                             "fused pair D (complex beta)", "fused pair D (complex)"]),
        "nonlinearity": ("fields", "has_nonlinearity", True, []),
        # OFF-DIAGONAL ADMITS THE ORDINARY MAGNETIC PAIR SINCE THE TARGET ROUND: the B
        # seam never receives ``chi1inv`` (its constitutive side is H), so an
        # off-diagonal epsilon cannot reach its arithmetic, while the ordinary
        # electric pair that does read the tensor is still refused here by name.
        # SINCE 2026-09-15 IT ALSO ADMITS THE SCRATCH-OUTPUT WELD on the D seam, the
        # product whose constitutive half IS the off-diagonal arm; its folded twin
        # stays out on this unfolded stub, refused on the fold axis.
        "off_diagonal_epsilon": ("fields", "has_offdiagonal_epsilon", True,
                                 ["fused pair B", "fused pair D (off-diagonal)"]),
        # THE TARGET ROUND'S OWN MOVE, and the first axis here whose value lives on
        # the PML rather than on the grid or the fields — which is why the loop
        # builds all three objects and dispatches on the row's ``home``.
        "pml_active": ("pml", "is_active", False,
                       ["fused pair D (no-PML stored E)"]),
    }
    for axis, (home, attribute, value, expected) in moved.items():
        grid, fields = certified_envelope_grid(), certified_envelope_fields()
        pml = certified_envelope_pml()
        setattr({"grid": grid, "fields": fields, "pml": pml}[home], attribute, value)
        assert arms_released_on(fields=fields, grid=grid, pml=pml) == expected, (
            f"{axis}: the shipped predicate no longer admits what this loop pins. "
            f"If that is intended, change the constant and say what drove it.")
        fastpath.plan_fast_path(fields, pml, grid)
        assert calls[-1]["fuse"] is bool(expected), (axis, expected)
        fusion = fastpath.last_dispatch_report()["fusion"]
        assert fusion["outside_the_released_envelope"] == [], axis
        assert fusion["released_here"] == expected, axis
        assert not (set(fusion["outside_this_arms_own_cases"]) & set(expected)), axis
        assert any(any(axis in reason for reason in why)
                   for why in fusion["outside_this_arms_own_cases"].values()), axis
    assert arms_released_on(grid=dict_grid(certified_envelope_grid(), beta=0.4)) == [
        "fused pair B (real beta)", "fused pair D (real beta)"]
    # AND THE ABSENT ABSORBER LEAVES EXACTLY THE ONE ARM RELEASED ON IT, named here
    # the way the beta and complex lines name theirs. This is the assertion the
    # emptied shared row used to make, inverted: the axis that used to refuse all
    # twenty-seven now admits precisely the arm whose own cases drove it, and the
    # offer the composer is handed carries that label and no other.
    #
    # THE CALL IS RE-ISSUED RATHER THAN READ OFF THE LOOP'S LAST ITERATION, so that
    # an eighth axis appended to ``moved`` cannot silently retarget the three
    # assertions below at a different configuration while they go on passing.
    assert arms_released_on(pml=inactive_envelope_pml()) == [
        "fused pair D (no-PML stored E)"]
    fastpath.plan_fast_path(certified_envelope_fields(), inactive_envelope_pml(),
                            certified_envelope_grid())
    assert calls[-1]["fuse_labels"] == ("fused pair D (no-PML stored E)",), calls[-1]
    # Every OTHER released arm refuses, which is the per-arm half carrying what the
    # shared row said before the move. SINCE 2026-09-15 ONE OF THEM REFUSES ON A
    # DIFFERENT AXIS: ``fused pair D (complex conductive no-PML)`` also pins
    # ``pml_active`` False, so this absorber-free shape agrees with it there and it is
    # refused on the axes this real 2-D stub does not carry -- asserted by name for
    # ``complex_storage`` below. Every remaining arm names ``pml_active``.
    absent_absorber = fastpath.last_dispatch_report()["fusion"][
        "outside_this_arms_own_cases"]
    assert set(absent_absorber) == (set(fastpath.RELEASED_FUSED_ARMS)
                                    - {"fused pair D (no-PML stored E)"})
    second_no_absorber_arm = "fused pair D (complex conductive no-PML)"
    assert not any("pml_active" in reason
                   for reason in absent_absorber[second_no_absorber_arm]), absent_absorber
    assert any("complex_storage=False" in reason
               for reason in absent_absorber[second_no_absorber_arm]), absent_absorber
    assert all(any("pml_active=False" in reason for reason in why)
               for arm, why in absent_absorber.items()
               if arm != second_no_absorber_arm), absent_absorber
    # The complex BETA pairs are offered on a complex grid too: their rows carry
    # no beta pin (the arm IS the beta product), exactly as the real beta pairs'.
    assert arms_released_on(fields=dict_fields(certified_envelope_fields(),
                                               force_complex_fields=True)) == [
        "fused pair B (complex beta)", "fused pair B (complex)",
        "fused pair D (complex beta)", "fused pair D (complex)"]
    # THE PER-ARM TABLE, 2026-09-02. ``dimensions=3`` and the FOLD no longer refuse
    # every arm from one shared row — the folded pairs REQUIRE a fold and the
    # ordinary pairs refuse one — so each is measured for what it leaves admitted
    # rather than for emptying the request. A fold refuses the unfolded arms and
    # admits the folded ones, which is the release the 09-11 round added and is why
    # the composer is still asked.
    #
    # 3-D STOPPED BEING AN EMPTY SHAPE IN THE 2026-09-14 TARGET ROUND, so the
    # assertion that it refuses every arm had to go: ``fused pair B`` and ``fused
    # pair D`` each widened their ``dimensions`` row to {1, 2, 3} when
    # pml_3d_diagonal lifted off-device to a (40, 40, 40) diagonal PML cell. The
    # measurement that replaces it is stronger than the one it replaces, because
    # ONE axis now gets THREE different answers out of the same table and each is
    # pinned by hand rather than derived from the table being read:
    #
    #   * the two widened pairs are ADMITTED, and the composer is offered exactly
    #     them — the contract this whole test is about, on a shape that used to be
    #     the file's example of asking for nothing;
    #   * the arms still pinned at a single dimensionality NAME the axis;
    #   * the BFAST pair, which REQUIRES 3-D (refl-angular's z-only cell declared
    #     3-D), does not name it at all and is refused on its own axis instead.
    grid = certified_envelope_grid()
    grid.dimensions = 3
    expected = arms_released_on(grid=grid)
    assert expected == ["fused pair B", "fused pair D"], expected
    fastpath.plan_fast_path(certified_envelope_fields(),
                            certified_envelope_pml(), grid)
    assert calls[-1]["fuse"] is True, "dimensions"
    assert calls[-1]["fuse_labels"] == tuple(expected), calls[-1]
    fusion = fastpath.last_dispatch_report()["fusion"]
    assert fusion["released_here"] == expected
    per_arm = fusion["outside_this_arms_own_cases"]
    assert set(per_arm) == set(fastpath.RELEASED_FUSED_ARMS) - set(expected)
    # WHICH ARMS STOP NAMING THE AXIS IS THE WIDENING ITSELF, written out rather
    # than derived: an arm is absent from this set exactly when its own row pins
    # ``dimensions`` at a value 3 is not. A row silently widened to admit 3-D would
    # land here as an arm that stopped naming it, which is what this set is for.
    # ``fused pair D (complex conductive no-PML)`` JOINED IT ON 2026-09-15: like the
    # BFAST pair it REQUIRES 3-D (complex_no_pml_3d is its one case), so this stub
    # agrees with it on the axis and it is refused on its own axes instead, asserted
    # by name below.
    assert {arm for arm, why in per_arm.items()
            if not any("dimensions" in reason for reason in why)} == {
        "fused pair B (BFAST)", "fused pair D (BFAST)",
        "fused pair B (complex)", "fused pair D (complex)",
        "fused pair B (folded)", "fused pair D (folded)",
        "fused pair B (folded complex)", "fused pair D (folded complex)",
        "fused pair D (complex conductive no-PML)"}, per_arm
    assert all(any("bfast" in reason for reason in why)
               for arm, why in per_arm.items() if "BFAST" in arm), per_arm
    assert any("complex_storage=False" in reason for reason
               in per_arm["fused pair D (complex conductive no-PML)"]), per_arm
    assert any("pml_active=True" in reason for reason
               in per_arm["fused pair D (complex conductive no-PML)"]), per_arm
    # And the membership rows say which VALUES they were driven on rather than one:
    # the no-PML arm drove 1-D and 2-D and names 3 as outside that set.
    assert any("dimensions=3, not one of [1, 2]" in reason for reason
               in per_arm["fused pair D (no-PML stored E)"]), per_arm

    fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                            folded_envelope_grid())
    assert calls[-1]["fuse"] is True, "a fold is what the folded pairs are released on"
    fusion = fastpath.last_dispatch_report()["fusion"]
    # TWO LABELS AND NOT THREE ON THIS GRID, and since the re-attribution round the
    # third is refused on an AXIS rather than absent from the table. ``fused pair D
    # (folded dispersive)`` is released again — its hold was a divergence
    # re-attributed to the subnormal-policy install ordering — and its row pins
    # ``susceptibilities`` at 1, so on this zero-pole fold it is refused per-arm
    # like any other released label whose shape does not match. The folded
    # dispersive grid further down is where it is admitted.
    # SINCE PHASE B the folded REAL BETA pairs are offered here too: like every
    # beta product's row (and the CUDA special_kz rows before them) theirs carries
    # no ``beta`` pin -- the arm IS the beta product and its predicate requires
    # beta != 0 -- so on a plain fold the release offers them and the composer's
    # predicate is what declines. The two folded pairs are still what dispatches.
    assert fusion["released_here"] == arms_released_on(grid=folded_envelope_grid())
    assert {"fused pair B (folded)", "fused pair D (folded)"} <= set(
        fusion["released_here"])
    assert set(fusion["outside_this_arms_own_cases"]) == (
        set(fastpath.RELEASED_FUSED_ARMS) - set(fusion["released_here"]))
    assert all(any("folded" in reason for reason in why)
               for arm, why in fusion["outside_this_arms_own_cases"].items()
               if "folded" not in arm), fusion["outside_this_arms_own_cases"]

    # AND THE AXES ``fused pair B``'s OWN CASES DROVE BOTH VALUES OF STAY UNPINNED
    # FOR IT: conductive_2d carried a conductivity and dispersive_2d a
    # susceptibility. Pinning either would refuse two of the five cases that drove
    # it. Its SIBLING ``fused pair D`` is pinned on both — its own cases drove
    # neither — which is the narrowing this round made, so the same grid admits one
    # arm and not the other and the request is still made.
    fields = certified_envelope_fields()
    fields.has_conductivity = True
    fields.polarizations = (object(),)
    fastpath.plan_fast_path(fields, certified_envelope_pml(),
                            certified_envelope_grid())
    assert calls[-1]["fuse"] is True
    fusion = fastpath.last_dispatch_report()["fusion"]
    assert fusion["released_here"] == ["fused pair B"]
    assert "fused pair D" in fusion["outside_this_arms_own_cases"]

    # THE OFFER IS WHAT THE COMPOSER IS HANDED, not just the boolean. A product the
    # ladder cannot admit occupies BOTH slots of its seam, so a composer that
    # installed it would force clause (8) to reject the arms underneath it too.
    assert calls[-1]["fuse_labels"] == ("fused pair B",), calls[-1]

    # THE 2026-09-11 SHAPES, measured the way the fold block above is: what the
    # shape ADMITS, what it leaves refused, and the offer the composer is actually
    # handed. Each of the three is a shape on which the shared table used to answer
    # for every arm at once and now cannot. All three carry released arms since the
    # re-attribution round: the folded dispersive grid carried none while the
    # product that takes its D seam was held at the route gate's second consult
    # site, and now carries two.

    # A Dcyl m = 0 grid. The shared row this replaces refused all five Cartesian
    # arms AND both Dcyl products, so the request emptied; it now carries exactly
    # the two cylindrical labels, which is what `launch.py`'s has_cylindrical branch
    # leaves selectable there.
    fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                            cylindrical_envelope_grid())
    assert calls[-1]["fuse"] is True, "the Dcyl pairs are released on a Dcyl grid"
    fusion = fastpath.last_dispatch_report()["fusion"]
    assert fusion["outside_the_released_envelope"] == [], fusion
    assert fusion["released_here"] == ["fused pair B (cylindrical)",
                                       "fused pair D (cylindrical)"]
    assert set(fusion["outside_this_arms_own_cases"]) == (
        set(fastpath.RELEASED_FUSED_ARMS) - {"fused pair B (cylindrical)",
                                             "fused pair D (cylindrical)"})
    assert all(any("cylindrical" in reason for reason in why)
               for why in fusion["outside_this_arms_own_cases"].values())
    assert calls[-1]["fuse_labels"] == ("fused pair B (cylindrical)",
                                        "fused pair D (cylindrical)"), calls[-1]

    # A conductive grid. ``fused pair B`` was already admitted here — its own case
    # list drove both values of the axis — and the D seam, which was refused
    # outright before this round, now carries the conductive product.
    fastpath.plan_fast_path(conductive_envelope_fields(), certified_envelope_pml(),
                            certified_envelope_grid())
    assert calls[-1]["fuse"] is True
    fusion = fastpath.last_dispatch_report()["fusion"]
    assert fusion["released_here"] == ["fused pair B", "fused pair D (conductive)"]
    assert "fused pair D" in fusion["outside_this_arms_own_cases"]
    assert calls[-1]["fuse_labels"] == ("fused pair B",
                                        "fused pair D (conductive)"), calls[-1]

    # A folded dispersive grid, AND BOTH SEAMS ARE SERVED HERE. The pole is what
    # separates the two folded D-seam products — ``fused pair D (folded)`` pins
    # ``susceptibilities`` at 0 precisely because this case gives its D seam to the
    # dispersive one — and since the re-attribution round the dispersive one is
    # released to take it: the hold at ``synchronize_magnetic_fields`` was a
    # divergence of the ARRAY comparison leg under a lazily installed subnormal
    # policy, not of this seam, and three fresh-process re-drives read the site
    # PASS. So the pin is measured for what it always meant (the ordinary folded
    # pair is refused here) beside the product that does run, and the offer the
    # composer is handed carries both labels rather than one.
    #
    # THE ONE POLE IS LOAD-BEARING IN BOTH DIRECTIONS: the dispersive product's own
    # row pins ``susceptibilities`` at 1, so this fixture is inside it and the
    # five-pole fold the route gate uses as its envelope witness is not.
    fastpath.plan_fast_path(folded_dispersive_envelope_fields(),
                            certified_envelope_pml(), folded_envelope_grid())
    assert calls[-1]["fuse"] is True
    fusion = fastpath.last_dispatch_report()["fusion"]
    assert fusion["released_here"] == ["fused pair B (folded)",
                                       "fused pair D (folded dispersive)"]
    why = fusion["outside_this_arms_own_cases"]["fused pair D (folded)"]
    assert any("susceptibilities" in reason for reason in why), why
    assert "fused pair D (folded dispersive)" in fusion["released_here"]
    assert "fused pair D (folded dispersive)" not in fusion[
        "outside_this_arms_own_cases"], (
        "an arm the shape admits is not also refused per-arm")
    assert calls[-1]["fuse_labels"] == ("fused pair B (folded)",
                                        "fused pair D (folded dispersive)"), calls[-1]

    # AND THE SAME FOLD AT FIVE POLES ADMITS NOTHING, which is the guard that keeps
    # the route gate's Triton ENVELOPE WITNESS (``folded_dispersive5_2d``) a witness.
    # All three folded rows have to refuse it: the two folded pairs on 0/{0,1} and
    # the dispersive product on 1. If any of them is widened to five, the release
    # admits the shape the envelope leg exists to be shown declining, and this
    # assertion is what says so before a campaign spends a device hour finding out.
    five_pole = dict_fields(folded_dispersive_envelope_fields(),
                            polarizations=tuple(object() for _ in range(5)))
    fastpath.plan_fast_path(five_pole, certified_envelope_pml(),
                            folded_envelope_grid())
    fusion = fastpath.last_dispatch_report()["fusion"]
    assert fusion["released_here"] == [], fusion["released_here"]
    for arm in ("fused pair B (folded)", "fused pair D (folded)",
                "fused pair D (folded dispersive)"):
        why = fusion["outside_this_arms_own_cases"][arm]
        assert any("susceptibilities=5" in reason for reason in why), (arm, why)
    # AND THIS IS WHERE THE REQUEST EMPTIES, which is the half of this test the
    # shared table used to carry and the 2026-09-14 target round took off it. Not
    # one axis but every arm's own row refusing at once is what leaves the composer
    # asked for exactly the composition an un-released configuration is asked for
    # today — no ``fuse``, and no label offered for it to install. It is asserted on
    # THIS shape because it is the only one in this file where nothing is admitted:
    # the axes that emptied the request before the move each leave something
    # admitted now, and every one of them is measured above for what it leaves.
    assert calls[-1]["fuse"] is False and calls[-1]["fuse_ade"] is False, calls[-1]
    assert calls[-1]["fuse_labels"] == (), calls[-1]
    assert fusion["outside_the_released_envelope"] == [], (
        "the shape READ; an empty shared table has nothing to say about it")
    assert set(fusion["outside_this_arms_own_cases"]) == set(
        fastpath.RELEASED_FUSED_ARMS)


#: THE SHAPE EACH LABEL IS MEASURED ON, as a pair of factories so every test gets
#: its own objects. This map is the 2026-09-11 replacement for the ``"(folded)" in
#: label`` sniff below it: with eight released arms across five distinct shapes,
#: deriving the shape from the label's SPELLING stopped being possible — the
#: conductive product is released on an unfolded Cartesian grid that differs from
#: the certified triple only in a field, and the two Dcyl products on a grid that
#: differs only in a flag. A label whose shape is wrong here does not fail
#: quietly: the test's own "the shape has to be the arm's own" assertion below
#: reports it as a case list and a shape that drifted apart.
SHAPE_FOR_LABEL = {
    # A VALUE MAY CARRY A THIRD BUILDER, the PML, and only the arms released on an
    # absorber-free grid use it. Two elements means "the certified active PML",
    # which is what every arm released before the 2026-09-14 target round carries.
    "fused pair B": (certified_envelope_grid, certified_envelope_fields),
    "fused pair D": (certified_envelope_grid, certified_envelope_fields),
    "dispersive fused pair": (certified_envelope_grid, certified_envelope_fields),
    "fused pair B (folded)": (folded_envelope_grid, certified_envelope_fields),
    "fused pair D (folded)": (folded_envelope_grid, certified_envelope_fields),
    # RELEASED AGAIN SINCE THE RE-ATTRIBUTION ROUND, on the same shape it always
    # carried: the fold and the ONE-pole fields its case drove. It was the one row
    # here a campaign put back — held while the route gate's second consult site was
    # read as diverging from the array path on this very shape with fusion off as
    # well as on — and the re-drive attributed that to the lazy subnormal-policy
    # install rather than to the seam. The fields matter as much as the grid: this
    # arm's row pins ``susceptibilities`` at 1, so a zero-pole fold would measure a
    # refusal where the release is what this parametrisation is asking about.
    "fused pair D (folded dispersive)": (folded_envelope_grid,
                                         folded_dispersive_envelope_fields),
    # THE FIRST ARM RELEASED ON AN ABSORBER-FREE GRID (2026-09-14 target round,
    # when ``pml_active`` moved off the shared envelope onto per-arm rows). Its
    # row pins ``pml_active`` False, so it is measured against the inactive PML;
    # against the certified one it would read as outside its own envelope.
    "fused pair D (no-PML stored E)": (certified_envelope_grid,
                                       certified_envelope_fields,
                                       inactive_envelope_pml),
    # THE SECOND ARM RELEASED ON AN ABSORBER-FREE GRID (2026-09-15), and the first
    # on a complex one: its row pins ``pml_active`` False beside ``complex_storage``
    # True, ``conductivity`` True, ``susceptibilities`` 1 and ``dimensions`` 3, so it
    # is measured on its own case's grid and fields against the inactive PML.
    "fused pair D (complex conductive no-PML)": (
        complex_no_pml_3d_envelope_grid, complex_conductive_envelope_fields,
        inactive_envelope_pml),
    "fused pair D (conductive)": (certified_envelope_grid,
                                  conductive_envelope_fields),
    "fused pair B (cylindrical)": (cylindrical_envelope_grid,
                                   certified_envelope_fields),
    "fused pair D (cylindrical)": (cylindrical_envelope_grid,
                                   certified_envelope_fields),
    # THE SCRATCH-OUTPUT OFF-DIAGONAL WELDS (2026-09-15), on the shapes
    # offdiag_magnetic_2d and folded_offdiag_magnetic_2d lifted.
    "fused pair D (off-diagonal)": (certified_envelope_grid, offdiag_envelope_fields),
    "fused pair D (folded off-diagonal)": (folded_envelope_grid,
                                           offdiag_envelope_fields),
    # THE PHASE B ARMS (2026-09-13), each on the shape its own route case lifted.
    "fused pair B (real beta)": (beta_envelope_grid, certified_envelope_fields),
    "fused pair D (real beta)": (beta_envelope_grid, certified_envelope_fields),
    "fused pair B (folded real beta)": (folded_beta_envelope_grid,
                                        certified_envelope_fields),
    "fused pair D (folded real beta)": (folded_beta_envelope_grid,
                                        certified_envelope_fields),
    "fused pair B (complex beta)": (beta_envelope_grid, complex_envelope_fields),
    "fused pair D (complex beta)": (beta_envelope_grid, complex_envelope_fields),
    "fused pair B (folded complex beta)": (folded_beta_envelope_grid,
                                           complex_envelope_fields),
    "fused pair D (folded complex beta)": (folded_beta_envelope_grid,
                                           complex_envelope_fields),
    "fused pair B (BFAST)": (bfast_envelope_grid, certified_envelope_fields),
    "fused pair D (BFAST)": (bfast_envelope_grid, certified_envelope_fields),
    "fused pair B (nonlinear)": (nonlinear_envelope_grid, nonlinear_envelope_fields),
    "fused pair B (complex)": (bloch_envelope_grid, complex_envelope_fields),
    "fused pair D (complex)": (bloch_envelope_grid, complex_envelope_fields),
    "fused pair B (folded complex)": (folded_bloch_envelope_grid,
                                      complex_envelope_fields),
    "fused pair D (folded complex)": (folded_bloch_envelope_grid,
                                      complex_envelope_fields),
    "fused pair B (cylindrical complex)": (cylindrical_envelope_grid,
                                           complex_envelope_fields),
    "fused pair D (cylindrical complex)": (cylindrical_envelope_grid,
                                           complex_envelope_fields),
    # NOT RELEASED, and measured on the shape that admits its siblings so the
    # refusal it produces is about the label rather than about the configuration.
    "fused ADE state": (certified_envelope_grid, certified_envelope_fields),
}


@pytest.mark.parametrize("label", sorted(SHAPE_FOR_LABEL))
def test_a_released_fused_arm_dispatches_only_inside_the_envelope(monkeypatch, label):
    """THE RELEASE, both directions, over the whole fused label space.

    Every released label in this map was driven through the driver's own consults
    by ``fastpath.DRIVER_ROUTE_FUSED_GATE`` and dispatches here with nothing set;
    the unreleased ones do not and still refuse by name on the very configuration
    that admits their siblings. The refusal for an unreleased label must not read as
    a configuration problem, and the refusal for a released label outside the
    envelope must.

    ``fused pair D (folded dispersive)`` MOVED SIDES ON THE RE-ATTRIBUTION ROUND,
    and it is worth saying which measurement moved it. It was unreleased on a
    MEASUREMENT rather than for want of a run: the gate DID drive it — 7 of 7 slots
    dispatched, every ``step()`` checkpoint fused==array — and its second consult
    site, ``synchronize_magnetic_fields``, was read as fused_vs_array False and
    unfused_vs_array False on a leg installing no fused product. Three fresh-process
    re-drives of the same case over the full ladder read that site PASS with both
    comparisons byte-clean, and the signature was attributed to the subnormal policy
    installing after CuPy's first compile, which makes the ARRAY leg flush what the
    dispatch legs keep. So the label is on the dispatching side now and ``fused ADE
    state`` is alone on the refusing one.

    EACH ARM IS DRIVEN ON THE SHAPE ITS OWN CASES RAN, which is the 2026-09-02
    change: a folded label on the unfolded grid is refused by
    ``FUSED_RELEASE_ARM_AXES`` and would have measured that refusal rather than the
    release. ``released_here`` is asked of the shipped predicate for the shape in
    hand rather than compared against the whole table, so a narrowing shows up here
    as a moved list rather than as a test nobody re-read.

    THE SHAPE NOW COMES FROM ``SHAPE_FOR_LABEL`` rather than from the label's text
    (2026-09-11): four of these ten labels are separated from a sibling by a FIELD
    or a grid flag, which no spelling of the label can carry.
    """
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    # THE MAP IS A DENOMINATOR, not a convenience: a released arm with no shape
    # here would simply not be measured, which is the one way this file can be
    # green about a release nobody drove.
    assert set(fastpath.RELEASED_FUSED_ARMS) <= set(SHAPE_FOR_LABEL), sorted(
        set(fastpath.RELEASED_FUSED_ARMS) - set(SHAPE_FOR_LABEL))
    shape_builders = SHAPE_FOR_LABEL[label]
    build_grid, build_fields = shape_builders[0], shape_builders[1]
    build_pml = shape_builders[2] if len(shape_builders) > 2 else None
    grid, fields = build_grid(), build_fields()
    pml = build_pml() if build_pml is not None else None
    folded = any(grid.is_mirrored(axis) for axis in range(3))
    plans = {"step_B": CountingPlan("b")}
    selected = {"step_B": label}
    if folded:
        # RUNG (6b) IS NOT BEING WORKED AROUND, it is being SATISFIED: a folded
        # composition that leaves the ghost fills on the array path is exactly what
        # that rung refuses, and the gate's folded cases carry the mirror-fill
        # kernel in both slots. A stub without them would measure 6b, not the
        # release.
        plans["fill_B"] = CountingPlan("fill_b")
        plans["fill_D"] = CountingPlan("fill_d")
        selected["fill_B"] = selected["fill_D"] = "mirror fill"
    plan = plan_inside_the_released_envelope(
        monkeypatch, step_plan(plans, selected), grid=grid, fields=fields, pml=pml)
    record = fastpath.last_dispatch_report()
    admitted_here = arms_released_on(grid=grid, fields=fields, pml=pml)
    assert record["fusion"]["released_here"] == admitted_here
    assert record["fusion"]["outside_the_released_envelope"] == []
    # THE SHAPE HAS TO BE THE ARM'S OWN, or the branch below measures the wrong
    # refusal: an arm absent from what this shape admits is not evidence about the
    # release, it is evidence the case list and the shape drifted apart.
    assert (label in admitted_here) == (label in fastpath.RELEASED_FUSED_ARMS), (
        f"{label} is released={label in fastpath.RELEASED_FUSED_ARMS} but this "
        f"shape admits {admitted_here}")

    if label in fastpath.RELEASED_FUSED_ARMS:
        assert plan is not None, record["refused_because"]
        assert record["fusion"]["driven"] == {"step_B": label}
        assert record["fusion"]["opted_in"] == []
        assert record["fusion"]["released_on_cases"][label] == list(
            fastpath.RELEASED_FUSED_ARMS[label])
        assert "driven_outside_the_released_envelope" not in record["fusion"]
    else:
        assert plan is None
        refusal = record["refused_because"]
        assert label in refusal and "Not admitted" in refusal
        assert "outside it" not in refusal, (
            "an unreleased label is refused because no gate drove it, not because "
            f"of the configuration: {refusal}")


@pytest.mark.parametrize("label,needle", [
    ("fused pair B (folded)", "folded is absent"),
    ("fused pair D (folded)", "folded is absent"),
    # THE 2026-09-11 ARMS, on the same Cartesian unfolded grid. The two Dcyl
    # products are refused on the axis that LEFT the shared envelope this round,
    # which is the assertion that would silently pass if `cylindrical` had been
    # left there.
    #
    # `fused pair D (folded dispersive)` IS A ROW HERE SINCE THE RE-ATTRIBUTION
    # ROUND, and until then it could not be: the campaign's preflight had held it
    # back at the route gate's second consult site (`synchronize_magnetic_fields`
    # read fused_vs_array False and unfused_vs_array False on a leg installing no
    # fused product), so it was refused by NAME at clause (8) and the per-arm table
    # this test reads had no entry for it at all. The re-drive attributed that
    # divergence to the lazy subnormal-policy install, the arm is released, and the
    # refusal it owes on a plain grid is now a per-arm one — on `folded`, like the
    # two folded pairs above, since this product exists only on a fold.
    ("fused pair D (folded dispersive)", "folded is absent"),
    ("fused pair B (cylindrical)", "cylindrical=False, not True"),
    ("fused pair D (cylindrical)", "cylindrical=False, not True"),
    # THE PHASE B ARMS (2026-09-13), refused on the plain grid by the axis that left
    # the shared envelope for them: storage, the BFAST flag, the nonlinearity flag.
    ("fused pair B (complex)", "complex_storage=False, not True"),
    ("fused pair D (complex)", "complex_storage=False, not True"),
    ("fused pair B (BFAST)", "bfast=False, not True"),
    ("fused pair B (nonlinear)", "nonlinearity=False, not True"),
    ("fused pair B (cylindrical complex)", "cylindrical=False, not True"),
])
def test_an_arm_is_refused_on_the_plain_grid_its_cases_never_ran(
        monkeypatch, label, needle):
    """The other direction of the per-arm table, and the one a shared row could not say.

    ``FUSED_RELEASE_ENVELOPE`` used to carry ONE ``folded`` row for every arm, so
    releasing the folded pairs meant either refusing them everywhere or admitting
    the ORDINARY pairs on a folded grid — a configuration no gate has ever driven.
    The per-arm table is what holds both halves, and this is the half that would
    silently pass if the folded arms were released on the shared table instead.

    ``cylindrical`` MADE THE SAME MOVE ON 2026-09-11 and is measured here on the
    same terms, which is why this test is no longer named for the fold: the shared
    row said "no arm runs on a Dcyl grid" and two arms now do, so the row had to
    become per-arm and the ordinary pairs' refusal on that grid — and these two
    products' refusal on the Cartesian one — had to become an assertion rather
    than a consequence of the table's shape.
    """
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    plan = plan_inside_the_released_envelope(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": label}))
    record = fastpath.last_dispatch_report()
    assert plan is None, f"{label} won a slot on a shape its cases never ran"
    assert label not in record["fusion"]["released_here"]
    why = record["fusion"]["outside_this_arms_own_cases"][label]
    assert any(needle in reason for reason in why), why


def test_the_opt_in_still_reaches_past_the_release_and_says_so(monkeypatch):
    """The switch's remaining job: measuring what the release does not cover.

    A gate that has to drive an unmeasured shape needs a way in, and the record
    has to mark that run so its artifact cannot be read as the released
    composition. Both halves measured: it dispatches, and it is stamped.
    """
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "fused pair B")
    # THE SHAPE MOVED IN THE 2026-09-14 TARGET ROUND AND THE ASSERTIONS DID NOT.
    # This test drove a 3-D grid as its "past the release" shape; the round widened
    # ``fused pair B``'s ``dimensions`` row to {1, 2, 3} when pml_3d_diagonal lifted
    # off-device, so 3-D is INSIDE that arm's envelope now and an opt-in there would
    # reach past nothing — the stamp this test exists for would correctly be absent
    # and the test would be measuring its own fixture. A NONLINEAR 2-D grid is
    # outside every released arm: the one nonlinear product is released at 1-D, and
    # every other arm's row pins ``nonlinearity`` False because no case that drove
    # it is nonlinear. The precondition is asserted rather than assumed, because a
    # shape that quietly became released is exactly how this test would go hollow.
    fields = dict_fields(certified_envelope_fields(), has_nonlinearity=True)
    grid = certified_envelope_grid()
    assert arms_released_on(fields=fields, grid=grid) == [], (
        "an opt-in reaches past nothing on a shape the release covers")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "fused pair B"}),
        fields=fields, grid=grid, pml=certified_envelope_pml())
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    fusion = fastpath.last_dispatch_report()["fusion"]
    assert fusion["released_here"] == []
    assert fusion["driven"] == {"step_B": "fused pair B"}
    stamp = fusion["driven_outside_the_released_envelope"]
    assert stamp["arms"] == ["fused pair B"]
    assert stamp["reached_by"] == fastpath.FUSE_ARMS_SWITCH
    assert any("nonlinearity=True" in reason for reason in stamp["reasons"]), stamp
    # AND THE STAMP NAMES THE ARM'S OWN ROW, not the shared table, which is the half
    # that would have silently stopped working when the target round emptied
    # ``FUSED_RELEASE_ENVELOPE``: the stamp reads BOTH halves of the release, and
    # with the shared half empty the per-arm half is the whole of what marks this
    # artifact as something other than the released composition.
    assert all(reason.startswith("fused pair B: ") for reason in stamp["reasons"]), stamp


def test_the_fusion_veto_takes_the_release_back_and_leaves_dispatch_running(monkeypatch):
    """``MEEP_GPU_FUSE_ARMS=0``, both directions, on the configuration that fuses.

    WHY THE VETO IS PRODUCT CODE AND NOT A GATE FLAG. Before the release, "dispatch
    the separate arms, do not fuse" was what an unset switch meant; after it, that
    same unset switch is the fused composition, so the configuration that isolates
    a fusion defect from a sub-step defect stopped being reachable. It is also the
    substitution proof's baseline: measured on the GPU host GPU 6 against the released
    tree, the driver-route gate's unfused leg fused too — pml_2d at 2.0 launches
    per step on both sides, ``launch_drop_per_step 0.0`` — and the gate refused to
    release until this switch gave the baseline back.

    BOTH DIRECTIONS, because a veto that vetoes everything is as broken as one that
    vetoes nothing: with the token the composer is asked ``fuse=False`` and the
    separate arm still dispatches; without it, on the same grid, ``fuse=True``.

    AND THE RECORD MUST KEEP THE TWO FACTS APART. ``released_here`` is a statement
    about the CONFIGURATION and does not move; ``admitted`` is what this run will
    take and empties. An artifact that collapsed them would read "the release does
    not cover this shape", which is a different and false thing.
    """
    calls = install_composer(monkeypatch, step_plan())
    stub_triton(monkeypatch)

    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, fastpath.FUSE_ARMS_VETO)
    fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                            certified_envelope_grid())
    assert calls[-1]["fuse"] is False and calls[-1]["fuse_ade"] is False
    block = fastpath.last_dispatch_report()["fusion"]
    assert block["vetoed"] is True
    assert block["admitted"] == []
    assert block["released_here"] == arms_released_on(), (
        "the veto is the run's choice, not a fact about the configuration")
    assert block["outside_the_released_envelope"] == []
    assert block["opted_in"] == [], "the token is a value, not an arm label"

    # The other direction, same grid, same process: unset and it fuses.
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                            certified_envelope_grid())
    assert calls[-1]["fuse"] is True
    assert fastpath.last_dispatch_report()["fusion"]["vetoed"] is False

    # A VETO A LABEL CAN OVERRIDE IS NOT A VETO. The ambiguous value answers to the
    # safe branch, and the label it names is still reported as asked for.
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH,
                       f"{fastpath.FUSE_ARMS_VETO},fused pair B")
    fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                            certified_envelope_grid())
    assert calls[-1]["fuse"] is False
    block = fastpath.last_dispatch_report()["fusion"]
    assert block["vetoed"] is True and block["opted_in"] == ["fused pair B"]
    assert block["admitted"] == []


def test_a_vetoed_run_still_dispatches_its_separate_arms(monkeypatch):
    """The veto subtracts the fusion and NOTHING ELSE — the point of having it.

    A switch that turned dispatch off along with the fusion would be the kill
    switch with a second name, and would make the substitution proof compare
    against the array path, which launches nothing and would make any number look
    like a saving.
    """
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, fastpath.FUSE_ARMS_VETO)
    plan = plan_inside_the_released_envelope(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = fastpath.last_dispatch_report()
    assert record["decision"] == "dispatched"
    assert record["step_path"] == "fused"      # the artifact's word for "a kernel ran"
    assert record["fusion"]["driven"] == {}
    assert record["fusion"]["vetoed"] is True


def test_the_veto_still_refuses_a_fused_label_by_name(monkeypatch):
    """If a fused arm reached clause (8) anyway, the veto must refuse it AND say so.

    Unreachable today — the veto empties ``admitted``, so ``fuse`` is False and no
    fused label is selected — which is exactly why it is worth pinning: the clause
    is the backstop for a composer that fuses without being asked, and a backstop
    nothing measures is a comment. The refusal must not tell a vetoing caller to
    set the switch they already set.
    """
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, fastpath.FUSE_ARMS_VETO)
    plan = plan_inside_the_released_envelope(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "fused pair B"}))
    assert plan is None
    refusal = fastpath.last_dispatch_report()["refused_because"]
    assert "fused pair B" in refusal and "Not admitted" in refusal
    assert f"{fastpath.FUSE_ARMS_SWITCH}={fastpath.FUSE_ARMS_VETO} vetoed" in refusal
    assert "set MEEP_GPU_FUSE_ARMS to drive one anyway" not in refusal

    # And with the veto lifted, the SAME arm on the SAME grid dispatches — so the
    # refusal above is attributable to the veto and not to the arm or the shape.
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    assert plan_inside_the_released_envelope(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "fused pair B"})) is not None


def test_the_fusion_opt_in_is_recorded_on_a_run_that_never_asked(monkeypatch):
    """A reader must be able to tell "asked for nothing" from "asked and refused".

    THREE STATES NOW, not two, and the third is the one 2026-08-29 added: a run
    that asked for nothing and was GIVEN the release, versus one that asked for
    nothing and is outside it. An absent key would collapse them.
    """
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    assert plan is not None
    block = fastpath.last_dispatch_report()["fusion"]
    assert block["variable"] == "MEEP_GPU_FUSE_ARMS"
    assert block["value"] is None and block["opted_in"] == []
    assert block["driven"] == {}
    assert block["driver_route_gate"] == fastpath.DRIVER_ROUTE_FUSED_GATE
    assert block["released_by_the_gate"] == sorted(fastpath.RELEASED_FUSED_ARMS)
    # Outside the envelope: released_here is empty AND says which axes said so.
    assert block["released_here"] == [] and block["admitted"] == []
    # WHICH KEY CARRIES "WHICH AXES" MOVED IN THE 2026-09-14 TARGET ROUND, and the
    # three states this test is about did not. ``pml_active`` was the last row in
    # :data:`fastpath.FUSED_RELEASE_ENVELOPE` and is now per-arm, so the shared half
    # is EMPTY on every shape that reads and ``outside_the_released_envelope`` fires
    # only on the unreadable-shape branch — a refusal about the RECORD rather than
    # about the configuration. Asserting it truthy here would now be asserting that
    # this stub's shape did not read, which is a different and false fact.
    #
    # SO THE STATE IS READ OFF TWO KEYS INSTEAD OF ONE, and the distinction the
    # docstring is about is pinned harder than it was: the key is PRESENT and an
    # empty LIST — "measured, and the shared half had nothing to say" — which is a
    # different value from the ``"the run shape was not reached"`` string that
    # ``test_a_refused_record_says_the_run_shape_was_never_reached`` pins for the
    # state above it, and ``outside_this_arms_own_cases`` names an axis for every
    # one of the released arms this configuration refuses, which is all of them.
    assert block["outside_the_released_envelope"] == [], block
    why = block["outside_this_arms_own_cases"]
    assert set(why) == set(fastpath.RELEASED_FUSED_ARMS), why
    # BOTH AXES, EACH OVER THE DENOMINATOR IT ACTUALLY COVERS, rather than one
    # either/or that would be carried by the first and say nothing about the
    # second: this stub names neither ``dimensions`` nor an absorber, so every arm
    # reports the unread dimensionality, and all but the arms released WITHOUT an
    # absorber report the axis the target round moved here. TWO SUCH ARMS SINCE
    # 2026-09-15: ``fused pair D (complex conductive no-PML)`` pins ``pml_active``
    # False beside ``fused pair D (no-PML stored E)``, and both are written out.
    assert all(any("dimensions did not read" in reason for reason in reasons)
               for reasons in why.values()), why
    assert {arm for arm, reasons in why.items()
            if any("pml_active=False" in reason for reason in reasons)} == (
        set(fastpath.RELEASED_FUSED_ARMS) - {"fused pair D (no-PML stored E)",
                                             "fused pair D (complex conductive no-PML)"}
    ), why

    # Inside it, with the same empty switch, the same block reads the other way.
    # ``released_here`` is the arms THIS SHAPE admits, which since the per-arm table
    # is a strict subset of the whole release: the folded pairs are released and
    # this grid is not folded.
    plan = plan_inside_the_released_envelope(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    assert plan is not None
    block = fastpath.last_dispatch_report()["fusion"]
    assert block["released_here"] == arms_released_on()
    assert block["admitted"] == arms_released_on()
    assert set(block["released_here"]) < set(fastpath.RELEASED_FUSED_ARMS), (
        "the whole table would admit the folded pairs on an unfolded grid")
    assert block["outside_the_released_envelope"] == []
    assert block["driven"] == {}, "no fused arm won a slot, so none was driven"


def test_a_refused_record_says_the_run_shape_was_never_reached(monkeypatch):
    """The run-dependent half of the block, on a run that never got a shape.

    ``"the run shape was not reached"`` rather than ``[]``: an empty list means
    "measured, and nothing applied", and a rung above the composer measured
    nothing at all. Reading the two as one is how a kill-switched run comes to
    look like an unreleased configuration.
    """
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    assert plan_on_a_cupy_host(monkeypatch, step_plan()) is None
    block = fastpath.last_dispatch_report()["fusion"]
    assert block["released_here"] == "the run shape was not reached"
    assert block["admitted"] == "the run shape was not reached"
    assert block["outside_the_released_envelope"] == "the run shape was not reached"


def test_every_released_arm_carries_a_certification_and_a_case(monkeypatch):
    """A released arm whose artifact names no gate is the over-claim to prevent.

    ``ARM_CERTIFICATION`` maps each fused label to the family gate that cut its
    BYTES; ``RELEASED_FUSED_ARMS`` maps it to the seam gate's cases that drove it
    through the driver. Both are required, and neither substitutes for the other.

    The gate must resolve ON THE ARCHITECTURE THE ARM DISPATCHES ON, which is what
    ``capability=`` asks: a released arm whose weld has a run for some other card is
    the same over-claim one step in.
    """
    for arm, cases in fastpath.RELEASED_FUSED_ARMS.items():
        family, gate = fastpath.ARM_CERTIFICATION[arm]
        assert family != "unmapped" and gate != "unmapped", arm
        assert "recorded_utc" in fastpath._certification_for(
            arm, capability=CERTIFIED_CAPABILITY), arm
        assert cases and all(isinstance(case, str) and case for case in cases), arm
    # And nothing pending may be released: the two sets must not intersect.
    assert not (set(fastpath.RELEASED_FUSED_ARMS)
                & set(fastpath.PENDING_DEVICE_GATE_ARMS))


# ---------------------------------------------------------------------------
# The warm pass (S4): compilation as a plan-time event
# ---------------------------------------------------------------------------


def test_every_dispatching_slot_is_warmed_at_plan_time(monkeypatch):
    b, h = CountingPlan("step_B"), CountingPlan("update_H")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": b, "update_H": h}, {"step_B": "PML", "update_H": "ordinary"}))
    assert plan is not None
    assert (b.warms, h.warms) == (1, 1)
    assert (b.runs, h.runs) == (0, 0), "warming must not launch the sub-step"


def test_a_slot_whose_kernel_will_not_compile_unfills_and_the_rest_dispatches(monkeypatch):
    b = CountingPlan("step_B")
    broken = CountingPlan("update_H", warm_raises=RuntimeError("PTX assembly failed"))
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": b, "update_H": broken}, {"step_B": "PML", "update_H": "ordinary"}))
    assert plan is not None
    assert plan.slots == ("step_B",)
    entry = plan.report()["slots"]["update_H"]
    assert entry["state"] == "array"
    assert "PTX assembly failed" in entry["reason"]


@pytest.mark.parametrize("broken", ["step_B", "update_H"])
def test_half_a_fused_pair_surviving_the_warm_pass_refuses_the_whole_plan(monkeypatch, broken):
    """A PAIR UNFILLS AS A UNIT. The test above is right for independent slots and wrong
    for a pair, in one of two ways depending on which half is left standing:

    * the LEADING half dropped — the driver runs the array curl, then the absorbed slot
      dispatches its sentinel, which does nothing, so the constitutive sub-step happens
      on neither path;
    * the ABSORBED half dropped — the fused launch performs both halves and the driver
      then runs the array constitutive call ON TOP, which accumulates.

    Neither is guardable at the seam, which reasons about a pair through the slots that
    dispatch and would only ever see half of one. So it is refused at plan time and named,
    and the whole step goes to the array path, which is correct everywhere. The warm pass
    is the reachable route because it unfills PER SLOT.
    """
    pair = CountingPlan("step_B")
    absorbed = CountingPlan("update_H")
    absorbed.absorbed_by = pair            # what NoopPlan/TrailingRepairPlan carry
    plans = {"step_B": pair, "update_H": absorbed}
    plans[broken]._warm_raises = RuntimeError("PTX assembly failed")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        plans, {"step_B": "PML", "update_H": "ordinary"}))
    assert plan is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "step_B, update_H" in reason, reason
    assert f"{broken} did not survive to the dispatch set" in reason, reason


def test_a_pair_that_loses_BOTH_halves_is_the_ordinary_unfill_not_the_split(monkeypatch):
    """THE CONTROL for the test above, and the difference is what makes it a rule
    rather than a blanket. A pair with NEITHER half left is not half a pair — it is
    both halves on the array path, which is what every unfilled slot gets. Refusing
    the plan over it would take an unrelated working dispatch set down with one
    product's compile failure, and that is a regression against
    ``test_a_slot_whose_kernel_will_not_compile_unfills_and_the_rest_dispatches``.
    """
    pair = CountingPlan("step_B", warm_raises=RuntimeError("PTX assembly failed"))
    absorbed = CountingPlan("update_H", warm_raises=RuntimeError("PTX assembly failed"))
    absorbed.absorbed_by = pair
    survivor = CountingPlan("step_D")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": pair, "update_H": absorbed, "step_D": survivor},
        {"step_B": "PML", "update_H": "ordinary", "step_D": "PML"}))
    assert plan is not None, fastpath.last_dispatch_report().get("refused_because")
    assert plan.slots == ("step_D",), "the unbroken slot must still dispatch"


def _one_hop_identity(entry):
    """What ``_pair_identity`` answered before it followed the chain. The reference.

    Kept here rather than imported because the point of the test below is that the
    shipped function no longer looks like this and still answers the same thing on
    every shape that shipped: a reference read out of the subject would move with it.
    """
    absorbed = getattr(entry, "absorbed_by", None)
    return id(entry if absorbed is None else absorbed)


def test_the_transitive_absorb_walk_groups_every_shipped_triton_shape_AS_ONE_HOP_DID():
    """``_pair_identity`` follows ``absorbed_by`` to the end now. Nothing moved here.

    THE WALK WAS MADE TRANSITIVE FOR A SHAPE NO TRITON INSTALLER WRITES: a hand-CUDA
    three-slot weld puts a half-plan in ``update_P`` that names the weld's LEADING
    half, which names the weld — two hops, and a single-hop answer would have put one
    product in two identities, which ``_split_pairs`` reads as half a pair and the
    merge reads as two units to adopt separately.

    THE COST OF THAT CHANGE IS THE THING TO PIN, because it is paid on the Triton
    table whether or not a second table is present, and it is paid silently: a moved
    grouping does not raise, it re-partitions the plan and the first visible symptom
    is a released pair refused as split (or, worse, adopted in halves). So every
    absorbed shape any installer in this tree writes is built here against the REAL
    classes and asserted to answer BOTH the terminal product's identity AND exactly
    what the one-hop rule answered — the four Triton-side shapes are one hop deep by
    construction, which is why the change is invisible to them.
    """
    from meep_gpu.deposit_repair import LeadingRepairPlan, TrailingRepairPlan
    from meep_gpu.withdraw_hoist import LeadingWithdrawPlan

    pair = CountingPlan("step_B")
    fields = types.SimpleNamespace()
    pml = types.SimpleNamespace()
    leading_repair = LeadingRepairPlan(pair, fields, pml, (), "step_D")
    shapes = {
        "the fused pair itself": pair,
        "NoopPlan in the absorbed slot": launch_module.NoopPlan("update_H", pair),
        "LeadingRepairPlan wrapping the pair": leading_repair,
        "TrailingRepairPlan in the absorbed slot": TrailingRepairPlan(
            "update_E", leading_repair, fields, pml),
        "LeadingWithdrawPlan wrapping the pair": LeadingWithdrawPlan(
            pair, fields, ()),
    }
    for name, entry in shapes.items():
        assert fastpath._pair_identity(entry) == id(pair), name  # noqa: SLF001
        assert fastpath._pair_identity(entry) == _one_hop_identity(entry), (  # noqa: SLF001
            f"{name} groups differently under the transitive walk than it did under "
            "the one-hop rule; a Triton shape moved")


def test_the_absorb_walk_terminates_on_a_malformed_chain_rather_than_hanging():
    """The bound is a guard against a plan nobody writes, and it runs at freeze time.

    A cycle or a self-reference here would hang the configuration freeze — not raise,
    not refuse — which is the one failure mode a ladder whose contract is "every rung
    answers with a named reason" cannot report at all. The limit is why the walk can
    be transitive without that risk, and it is asserted rather than argued.
    """
    self_referential = CountingPlan("step_B")
    self_referential.absorbed_by = self_referential
    assert fastpath._pair_identity(self_referential) == id(self_referential)  # noqa: SLF001

    first, second = CountingPlan("step_B"), CountingPlan("step_D")
    first.absorbed_by = second
    second.absorbed_by = first
    # The cycle stops at the first repeat, so the answer is the far end of the one
    # hop that was real. What matters is that it ANSWERS.
    assert fastpath._pair_identity(first) == id(second)  # noqa: SLF001

    # A chain longer than the limit stops at the limit; the deepest shape that ships
    # is two hops, so this is headroom being measured, not a budget being spent.
    chain = [CountingPlan(f"slot_{index}") for index in range(20)]
    for earlier, later in zip(chain, chain[1:]):
        earlier.absorbed_by = later
    answered = fastpath._pair_identity(chain[0])  # noqa: SLF001
    assert answered == id(chain[fastpath._ABSORB_CHAIN_LIMIT])  # noqa: SLF001


@pytest.mark.parametrize("broken", ["fill_B", "fill_D"])
def test_a_fill_slot_lost_at_the_warm_pass_takes_the_whole_folded_plan_down(
        monkeypatch, broken):
    """C3b IS JUDGED TWICE, and this is the second time.

    Rung 6b runs before anything compiles; the warm pass then unfills per slot. A
    folded run whose fill slot lost its kernel THERE would dispatch its curls with
    the ghost fills back on the array path — the same composition 6b refuses,
    enabled by a compile failure instead of by a sibling product's refusal. No
    shipped fill plan can reach it (neither fill plan exposes a grid attribute the
    warm pass can empty, so both are recorded unwarmed rather than unfilled), which
    is a fact about two classes and not an invariant — so the rung is re-applied and
    this drives it.
    """
    stub_triton(monkeypatch)
    plans = {"step_B": CountingPlan("b"), "update_H": CountingPlan("h"),
             "step_D": CountingPlan("d"), "update_E": CountingPlan("e"),
             "fill_B": CountingPlan("fill_b"), "fill_D": CountingPlan("fill_d")}
    plans[broken] = CountingPlan(broken, warm_raises=RuntimeError("nvrtc failed"))
    install_composer(monkeypatch, step_plan(
        plans, {"step_B": "folded PML", "update_H": "folded", "step_D": "folded PML",
                "update_E": "folded", "fill_B": "mirror fill",
                "fill_D": "mirror fill"}))
    assert fastpath.plan_fast_path(object(), object(), folded_grid()) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "mirror-folded" in reason and "warm pass unfilled" in reason, reason
    assert broken in reason, reason


def test_a_whole_plan_that_will_not_compile_refuses_rather_than_dying_in_step_one(monkeypatch):
    broken = CountingPlan("step_B", warm_raises=RuntimeError("nvrtc: no such device"))
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": broken}, {"step_B": "PML"}))
    assert plan is None
    assert "failed to compile" in fastpath.last_dispatch_report()["refused_because"]


def test_the_warm_pass_can_be_switched_off_and_says_so(monkeypatch):
    monkeypatch.setenv(fastpath.WARM_SWITCH, "0")
    b = CountingPlan("step_B")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": b}, {"step_B": "PML"}))
    assert plan is not None and b.warms == 0
    assert fastpath.WARM_SWITCH in plan.report()["slots"]["step_B"]["unwarmed"]


def test_update_P_is_warmed_too_so_a_compile_failure_cannot_land_mid_step(monkeypatch):
    """The slot the warm pass used to skip is the LAST one in the step.

    Skipping it meant a kernel that would not compile there raised after step_B,
    update_H, step_D and update_E had already advanced the fields — a fatal
    half-applied step where every other slot gets a free plan-time refusal.
    """
    entry = CountingPlan("polarization 0")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"update_P": [entry]}, {"update_P": "ADE update_P"}))
    assert plan is not None
    assert entry.warms == 1, "the polarization plans must be compiled at plan time"
    assert "update_P" not in plan.report()["slots"]["update_P"].get("unwarmed", "")


def test_an_update_P_kernel_that_will_not_compile_unfills_at_plan_time(monkeypatch):
    """And the whole plan refuses when nothing else is left, rather than dying in step 1."""
    broken = CountingPlan("p0", warm_raises=RuntimeError("PTX assembly failed for ade_update_p"))
    b = CountingPlan("step_B")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": b, "update_P": [broken]},
        {"step_B": "PML", "update_P": "ADE update_P"}))
    assert plan is not None and plan.slots == ("step_B",)
    entry = plan.report()["slots"]["update_P"]
    assert entry["state"] == "array"
    assert "PTX assembly failed" in entry["reason"]


def test_warming_a_polarization_plan_restores_the_buffer_rotation_it_performs():
    """The obstacle that kept update_P unwarmed, handled rather than avoided.

    ``AdeUpdatePPlan.run`` rotates three buffer REFERENCES per component. The warm
    launch enqueues zero programs, so no array content moves and the rotation is a
    pure permutation — snapshot the three names, launch, put them back.
    """
    class State:
        def __init__(self):
            self.P = {"Ez": "P0"}
            self.P_prev = {"Ez": "Pprev0"}
            self._scratch = "scratch0"

    class Entry:
        def __init__(self, state):
            self.state = state
            self.components = ("Ez",)
            self._grid = (21,)
            self.launched = []

        def run(self, drive, guard=None):
            self.launched.append((self._grid, drive))
            scratch = self.state._scratch
            previous = self.state.P["Ez"]
            self.state.P["Ez"] = scratch
            self.state._scratch = self.state.P_prev["Ez"]
            self.state.P_prev["Ez"] = previous

    state = State()
    entry = Entry(state)
    assert fastpath.warm_plan([entry], drive="drive-field") is None
    assert entry.launched == [((0,), "drive-field")], "zero programs, with the drive reader"
    assert entry._grid == (21,)
    assert (state.P, state.P_prev, state._scratch) == (
        {"Ez": "P0"}, {"Ez": "Pprev0"}, "scratch0"), "the rotation must be undone"


def test_warm_plan_refuses_a_plan_it_has_no_mechanism_for():
    class Opaque:
        def run(self, guard=None):
            raise AssertionError("a plan with no stored grid must not be launched")

    reason = fastpath.warm_plan(Opaque())
    assert reason is not None and "empties its launch grid" in reason


def test_warm_plan_empties_a_read_only_derived_launch_grid_through_its_backing_count():
    """The shape of ``FoldedOffdiagConstitutivePlan``, which the old mechanism could not warm.

    It declares ``__slots__`` WITHOUT ``_grid`` and exposes ``launch_grid`` as a
    read-only property, so ``setattr(plan, 'launch_grid', (0,))`` raised
    AttributeError inside the try — and the ``finally`` raised a SECOND one that
    masked it. The certified ``folded off-diagonal`` family could therefore never
    be warmed: every configuration where it won a slot unfilled at plan time and
    fell to the array path, reported as a compile failure it never had.
    """
    class DerivedGrid:
        __slots__ = ("n_elem", "block", "seen")

        def __init__(self):
            self.n_elem = 4096
            self.block = 256
            self.seen = []

        @property
        def launch_grid(self):
            return ((self.n_elem + self.block - 1) // self.block,)

        def run(self, guard=None):
            self.seen.append(self.launch_grid)

    plan = DerivedGrid()
    assert plan.launch_grid == (16,)
    assert fastpath.warm_plan(plan) is None
    assert plan.seen == [(0,)], "the warm launch must enqueue zero programs"
    assert plan.n_elem == 4096 and plan.launch_grid == (16,)


def test_a_failed_restore_after_a_warm_raises_rather_than_leaving_an_empty_grid():
    """A plan left with an emptied grid would dispatch and compute NOTHING."""
    class Sticky:
        def __init__(self):
            self._value = (9,)
            self.runs = 0

        @property
        def _grid(self):
            return self._value

        @_grid.setter
        def _grid(self, value):
            if value == (9,):
                raise AttributeError("cannot restore")
            self._value = value

        def run(self, guard=None):
            self.runs += 1

    plan = Sticky()
    with pytest.raises(RuntimeError, match="could not put it back"):
        fastpath.warm_plan(plan)


def test_warm_plan_empties_the_stored_grid_and_puts_it_back():
    seen = []

    class Griddy:
        def __init__(self):
            self._grid = (17,)

        def run(self, guard=None):
            seen.append(self._grid)

    plan = Griddy()
    assert fastpath.warm_plan(plan) is None
    assert seen == [(0,)], "the warm launch must enqueue zero programs"
    assert plan._grid == (17,), "the real launch grid must be restored"


def test_warm_plan_restores_the_grid_even_when_the_launch_raises():
    class Exploding:
        def __init__(self):
            self._grid = (17,)

        def run(self, guard=None):
            raise RuntimeError("compile failed")

    plan = Exploding()
    with pytest.raises(RuntimeError):
        fastpath.warm_plan(plan)
    assert plan._grid == (17,)


# ---------------------------------------------------------------------------
# Dispatch itself
# ---------------------------------------------------------------------------


def test_dispatch_runs_the_slot_and_answers_true(monkeypatch):
    b = CountingPlan("step_B")
    fields = object()
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": b}, {"step_B": "PML"}),
                               fields=fields)
    assert plan.dispatch("step_B", fields) is True and b.runs == 1
    assert plan.dispatch("update_E", fields) is False, "an unfilled slot is the array path"


def test_dispatch_refuses_a_fields_object_the_plan_was_not_built_from(monkeypatch):
    """The identity guard: a plan holds cached device views of ONE allocation."""
    b = CountingPlan("step_B")
    fields = object()
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": b}, {"step_B": "PML"}),
                               fields=fields)
    assert plan.dispatch("step_B", object()) is False
    assert b.runs == 0


def test_update_P_dispatch_passes_drive_field_to_every_entry(monkeypatch):
    """The one asymmetric slot: a LIST, and each entry takes ``fields.drive_field``."""
    entries = [CountingPlan("p0"), CountingPlan("p1")]
    fields = types.SimpleNamespace(drive_field=lambda component: component)
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"update_P": entries}, {"update_P": "ADE update_P"}), fields=fields)
    assert plan.dispatch("update_P", fields) is True
    assert [entry.runs for entry in entries] == [1, 1]
    assert all(entry.drives == [fields.drive_field] for entry in entries)


def test_an_exception_out_of_a_dispatched_run_propagates(monkeypatch):
    """C2. A launch that raised mid-write left the targets PARTLY updated; running
    the array call on top would double-apply the sub-step, which is the silent
    wrong answer this whole design exists to prevent."""

    class Exploding(CountingPlan):
        def run(self, drive=None, guard=None):
            raise RuntimeError("launch failed after writing Bx")

    fields = object()
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": Exploding("b")}, {"step_B": "PML"}), fields=fields)
    with pytest.raises(RuntimeError, match="after writing Bx"):
        plan.dispatch("step_B", fields)


# ---------------------------------------------------------------------------
# The artifact (T1/T2/T3)
# ---------------------------------------------------------------------------


def test_the_artifact_answers_all_four_questions(monkeypatch):
    """Question 3 IS A PER-ARCHITECTURE QUESTION, so this drives a grid whose device
    reads: a weld keeps one run record per compute capability, and a plan on a host
    whose card cannot be read quotes no run at all rather than guessing one. The
    unreadable-device case is its own test (``test_an_unreadable_device_is_recorded_
    as_unknown_and_not_refused``); here the artifact has to carry the evidence.
    """
    b = CountingPlan("step_B")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": b}, {"step_B": "PML"},
        {"update_E": ("update_E: the off-diagonal rows are live",)}),
        grid=cupy_like_grid_with_device())
    record = plan.report()
    # 1. Did I get kernels, and where?
    assert record["step_path"] == "fused" and record["decision"] == "dispatched"
    assert record["slots"]["step_B"]["state"] == "dispatched"
    # 2. Why not, for the ones I did not get?
    assert "off-diagonal rows are live" in record["slots"]["update_E"]["reason"]
    # 3. On what certification did the ones I got ride?
    certification = record["families"]["PML"]
    assert certification["gate"] == "bit_identity_gate"
    assert certification["capability"] == CERTIFIED_CAPABILITY, (
        certification.get("run_record"))
    assert certification["recorded_utc"] and certification["host"]
    assert certification["certification_policy"] == "keep"
    assert "24000/24000" in record["certification_budget"]
    # 4. Under what policy did this run resolve?
    assert record["subnormal"]["stamp"]["resolved"] in ("keep", "flush")
    assert "transitively" in record["subnormal"]["note"]


def test_every_array_slot_carries_a_named_reason(monkeypatch):
    """R4's whole mitigation: a missing speedup must never read as a bug."""
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    for slot, entry in plan.report()["slots"].items():
        if entry["state"] == "array":
            assert entry["reason"], f"{slot} went to the array path with nothing to say"


def test_the_artifact_records_the_expansion_probe_state(monkeypatch):
    """R5: four certified families refuse without the complex-expansion licence,
    and that must be visible rather than show up as families quietly missing."""
    monkeypatch.delenv(fastpath.COMPLEX_PROBE_ENV, raising=False)
    plan_on_a_cupy_host(monkeypatch, step_plan())
    environment = fastpath.last_dispatch_report()["environment"]
    assert environment["complex_expansion_probe_env_set"] is False
    assert environment["complex_expansion_probe_resolved"] in (True, False)


def test_one_json_object_is_appended_per_configuration_freeze(tmp_path, monkeypatch):
    """T1: readable with ``tail`` on the machine that owns the job (the progress-reporting rule)."""
    log = tmp_path / "dispatch.jsonl"
    monkeypatch.setenv(fastpath.DISPATCH_LOG, str(log))
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    for _ in range(3):
        fastpath.plan_fast_path(object(), object(), cupy_like_grid())
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["decision"] == "dispatched" for line in lines)


def test_an_unwritable_log_never_refuses_a_step(monkeypatch, tmp_path):
    monkeypatch.setenv(fastpath.DISPATCH_LOG, str(tmp_path / "no" / "such" / "dir.jsonl"))
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                      {"step_B": "PML"}))
    assert plan is not None


def test_one_stderr_line_at_the_first_dispatching_freeze(monkeypatch, capsys):
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    for _ in range(3):
        fastpath.plan_fast_path(object(), object(), cupy_like_grid())
    announced = [line for line in capsys.readouterr().err.splitlines()
                 if line.startswith("meep_gpu: step path")]
    assert len(announced) == 1, "one line per process, not one per freeze"
    assert "fused" in announced[0] and "step_B" in announced[0] and "PML" in announced[0]


def test_a_refusal_that_could_have_dispatched_says_so_once(monkeypatch, capsys):
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan())
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())
    err = capsys.readouterr().err
    assert "step path array" in err and "dispatch refused" in err


def test_an_opted_out_run_says_nothing(monkeypatch, capsys):
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "0")
    fastpath.plan_fast_path(None, None, cupy_like_grid())
    assert "meep_gpu:" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The driver seam: the seven consults
# ---------------------------------------------------------------------------


def counting_sub_steps(monkeypatch):
    """Replace the driver's array sub-step calls with counters, keeping behavior."""
    counts = {}
    for name in ("step_B", "update_H", "step_D", "update_E", "update_P"):
        real = getattr(driver_module, name)
        counts[name] = 0

        def counted(fields, pml, _name=name, _real=real):
            counts[_name] += 1
            return _real(fields, pml)

        monkeypatch.setattr(driver_module, name, counted)
    return counts


def install_plan(monkeypatch, driver, plans, selected):
    """Give ``driver`` a FastPathPlan over counting plans at its next freeze."""
    plan = fastpath.FastPathPlan(
        fields=driver.fields,
        step_plan=step_plan(plans, selected),
        slots=tuple(name for name in launch_module.STEP_ORDER if name in plans),
        arms=dict(selected),
        dropped_null={},
        unwarmed={},
        record={"step_path": "fused", "decision": "dispatched", "slots": {}},
    )
    # A GPU driver's freeze consults the planner; a reference driver's never does.
    driver.gpu = "metal"
    monkeypatch.setattr(driver_module, "plan_fast_path", lambda *args: plan)
    return plan


def test_a_dispatched_slot_replaces_its_array_call_and_only_that_one(monkeypatch):
    driver = numpy_driver()
    counts = counting_sub_steps(monkeypatch)
    b = CountingPlan("step_B")
    install_plan(monkeypatch, driver, {"step_B": b}, {"step_B": "PML"})
    driver.run(num_steps=3)
    assert b.runs == 3, "the dispatched sub-step must run once per step"
    assert counts["step_B"] == 0, "and its array call must not run at all"
    assert counts["update_H"] == counts["step_D"] == 3, "the rest stays on the array path"
    driver.close()


def test_every_one_of_the_driver_slots_can_dispatch(monkeypatch):
    """All seven, ``fill_B``/``fill_D`` included — parametrised off ``DRIVER_SLOTS``
    so a slot added there without a consult fails here rather than being trusted."""
    driver = numpy_driver()
    counts = counting_sub_steps(monkeypatch)
    trace = trace_fill_passes(monkeypatch)
    plans = {name: CountingPlan(name) for name in fastpath.DRIVER_SLOTS
             if name != "update_P"}
    plans["update_P"] = [CountingPlan("p0")]
    install_plan(monkeypatch, driver, plans,
                 {name: "PML" for name in fastpath.DRIVER_SLOTS})
    driver.step()
    assert all(count == 0 for count in counts.values()), counts
    assert all(plans[name].runs == 1 for name in fastpath.DRIVER_SLOTS
               if name != "update_P"), {n: plans[n].runs for n in plans
                                        if n != "update_P"}
    assert plans["update_P"][0].runs == 1
    assert trace == ["zero_metal_B", "zero_metal_D"], (
        "the four array fill passes are replaced and the two wall clears are not")
    driver.close()


# ---------------------------------------------------------------------------
# The fill seam: two consults and one unconsulted pass between them
# ---------------------------------------------------------------------------
#
# ``launch.STEP_ORDER``'s ``fill_B``/``fill_D`` stand for the near-symmetry AND
# folded-far passes together, and the driver runs ``zero_metal_*`` BETWEEN them
# (MEEP's ``step_boundaries``). So the seam has two consult sites per family, and
# the two shipped fill plans want different things from them:
# ``symmetry.MirrorGhostFillPlan`` fuses near and far into one launch per axis,
# while ``folded_complex.FoldedMirrorGhostFillComplexPlan`` splits them and says in
# its own docstring that a composed plan MUST NOT fuse across the wall clear.
# Both shapes are driven here, through the shipped ``step()``.


def trace_fill_passes(monkeypatch, trace=None):
    """Record the driver's array fill passes and wall clears IN ORDER.

    Order is the measurement, not a nicety: the defect a count alone cannot see is a
    far-ghost pass that moved to the wrong side of the wall clear.
    """
    trace = [] if trace is None else trace
    for name in ("fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
                 "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D"):
        real = getattr(driver_module, name)

        def traced(fields, _name=name, _real=real):
            trace.append(_name)
            return _real(fields)

        monkeypatch.setattr(driver_module, name, traced)
    return trace


class FusedFillPlan:
    """``MirrorGhostFillPlan``'s SHAPE: one launch carries near AND far."""

    def __init__(self, trace=None):
        self.runs = 0
        self._trace = trace

    def run(self, guard=None):
        self.runs += 1
        if self._trace is not None:
            self._trace.append("kernel_fill_near_and_far")


class SplitFillPlan:
    """``FoldedMirrorGhostFillComplexPlan``'s SHAPE: two launches, one per pass."""

    def __init__(self, trace=None):
        self.near = 0
        self.far = 0
        self._trace = trace

    def run_near(self, guard=None):
        self.near += 1
        if self._trace is not None:
            self._trace.append("kernel_fill_near")

    def run_far(self, guard=None):
        self.far += 1
        if self._trace is not None:
            self._trace.append("kernel_fill_far")

    def run(self, guard=None):  # pragma: no cover - the assertion IS the contract
        raise AssertionError(
            "a split fill plan was asked to fuse across the wall clear; "
            "FoldedMirrorGhostFillComplexPlan forbids exactly this")


def test_a_fused_fill_plan_replaces_both_of_the_drivers_fill_passes(monkeypatch):
    """DIRECTION ONE: the fill slot dispatches, and the driver runs neither pass.

    ``MirrorGhostFillPlan`` writes the near planes and the folded-far planes in one
    launch per axis, so answering True at the near site has to suppress the far site
    too — otherwise the array pass runs on top of a kernel that already did it.
    The wall clear is NOT suppressed, and the other family is untouched: a consult
    that swallowed either would be replacing a sub-step nobody planned.
    """
    driver = numpy_driver()
    trace = []
    trace_fill_passes(monkeypatch, trace)
    fill = FusedFillPlan(trace)
    install_plan(monkeypatch, driver, {"fill_B": fill}, {"fill_B": "mirror fill"})
    driver.run(num_steps=2)
    assert fill.runs == 2, "one launch per step"
    assert trace == ["kernel_fill_near_and_far", "zero_metal_B",
                     "fill_symmetry_bc_D", "zero_metal_D",
                     "fill_folded_far_ghosts_D"] * 2, trace
    driver.close()


def test_a_split_fill_plan_runs_its_far_half_in_the_drivers_own_far_slot(monkeypatch):
    """DIRECTION TWO: a plan that must not fuse gets the wall clear between its halves.

    The branch is on the PLAN (``run_near``), not on ``selected[slot]``, so this is
    driven with the arm label of the FUSED shape deliberately: a seam that read the
    label would call ``run`` here and trip the plan's own assertion.
    """
    driver = numpy_driver()
    trace = []
    trace_fill_passes(monkeypatch, trace)
    fill = SplitFillPlan(trace)
    install_plan(monkeypatch, driver, {"fill_D": fill}, {"fill_D": "mirror fill"})
    driver.step()
    assert (fill.near, fill.far) == (1, 1)
    assert trace == ["fill_symmetry_bc_B", "zero_metal_B",
                     "fill_folded_far_ghosts_B",
                     "kernel_fill_near", "zero_metal_D", "kernel_fill_far"], trace
    driver.close()


def test_a_fused_product_that_declares_the_fills_still_gets_the_drivers_passes(
        monkeypatch):
    """DIRECTION THREE, and the one that keeps ``deposit_repair`` honest.

    ``folded_fused_magnetic_pair.REPLACES`` names ``fill_B``, ``zero_metal_B`` and
    ``fill_folded_far_ghosts_B`` — the launch really does perform them. It performs
    them against the PRE-INJECTION field, though: ``_install_fused_pair`` writes only
    the curl and constitutive slots and the driver injects between those two
    consults, so the post-injection array passes are what image a source deposit into
    the ghost planes, and ``deposit_repair.repair_cells`` recomputes the constitutive
    at exactly those images by READING the field after they have run.

    So a fused product's ``REPLACES`` must NOT suppress them. The fill slot is served
    from the fill slot's own plan or from the array path, never from a sibling's
    declaration — which is what this drives: the composer's real fused product is
    installed over ``step_B``/``update_H`` with ``fill_B`` deliberately EMPTY, and
    both array fill passes still run.
    """
    from meep_gpu.triton_kernels import folded_fused_magnetic_pair as family

    assert "fill_B" in family.REPLACES, (
        "the premise of this test is the declaration; if it moved, so did the risk")
    driver = numpy_driver()
    trace = []
    trace_fill_passes(monkeypatch, trace)
    counts = counting_sub_steps(monkeypatch)
    pair = CountingPlan("fused pair B (folded)")
    sentinel = launch_module.NoopPlan("update_H", pair)
    install_plan(monkeypatch, driver, {"step_B": pair, "update_H": sentinel},
                 {"step_B": "fused pair B (folded)",
                  "update_H": "fused pair B (folded)"})
    driver.step()
    assert pair.runs == 1 and counts["step_B"] == 0
    assert trace[:3] == ["fill_symmetry_bc_B", "zero_metal_B",
                         "fill_folded_far_ghosts_B"], trace
    driver.close()


def test_synchronize_magnetic_fields_dispatches_the_fill_seam_too(monkeypatch):
    """The half-step is ``step``'s B half VERBATIM, consults included.

    A synchronization that filled on the array path while the step filled on kernels
    would be the mixed composition ``synchronize_magnetic_fields``' own comment
    refuses — and it would be invisible, because both answers are numerically the
    same until a kernel and an array pass disagree by a bit.
    """
    driver = numpy_driver()
    trace = []
    trace_fill_passes(monkeypatch, trace)
    fill = FusedFillPlan(trace)
    install_plan(monkeypatch, driver, {"fill_B": fill}, {"fill_B": "mirror fill"})
    driver.step()
    trace.clear()
    driver.synchronize_magnetic_fields()
    assert fill.runs == 2, "the half-step consulted the same slot the step did"
    assert trace == ["kernel_fill_near_and_far", "zero_metal_B"], trace
    driver.restore_magnetic_fields()
    driver.close()


def test_a_fill_consult_answers_false_for_a_fields_the_plan_was_not_built_from(
        monkeypatch):
    """The identity guard covers BOTH fill sites, or the far one writes a stale view.

    ``dispatch``'s guard was never the whole seam for a fill: the far half has its
    own entry point, and a plan holding device views of a rebound ``Fields`` would
    keep writing the previous allocation through it.
    """
    driver = numpy_driver()
    trace = []
    trace_fill_passes(monkeypatch, trace)
    fill = FusedFillPlan(trace)
    plan = install_plan(monkeypatch, driver, {"fill_B": fill},
                        {"fill_B": "mirror fill"})
    driver.step()
    assert fill.runs == 1
    object.__setattr__(plan, "fields", object())  # A Fields the plan no longer holds.
    trace.clear()
    driver.step()
    assert fill.runs == 1, "the identity guard refused the near half"
    assert trace[:3] == ["fill_symmetry_bc_B", "zero_metal_B",
                         "fill_folded_far_ghosts_B"], (
        "and the far half fell back with it, rather than answering True off a plan "
        "whose near launch never ran")
    driver.close()


def test_the_far_ghost_pass_and_the_wall_clear_commute():
    """THE ONE REORDERING A FUSED FILL PLAN FORCES, measured on the array path.

    A fused fill plan writes the far ghost planes inside the near launch, so on that
    shape the far pass moves from AFTER ``zero_metal_*`` to BEFORE it. The argument
    that this is safe is structural — ``_zero_metal`` writes stored plane 0 of an
    axis that is metallic AND NOT mirrored, ``_fill_folded_far_ghosts`` writes the
    last stored plane of a folded PERIODIC axis and reads another plane of that same
    axis, so the two only meet inside the wall plane, where the clear is the last
    writer in one order and the fill's own source row is already zero in the other.
    An argument is not a measurement, so this is the measurement, on a grid built to
    make the two passes overlap: a metallic X wall and a folded periodic Y axis.

    THE CONTROL IS THE SHAPE, not a second assertion: if the case were built without
    a wall or without a fold one of the passes would write nothing and the two orders
    would agree vacuously, so both passes are checked to have MOVED bytes first.
    """
    from meep_gpu import stepping
    from meep_gpu.grid import Mirror

    def build():
        driver = FdtdDriver(cell_size=(2.0, 2.0, 0.0), resolution=10,
                            dimensions=2, force_complex_fields=False,
                            boundaries=("metallic", "periodic", "periodic"),
                            symmetry=(Mirror("Y", 1),))
        driver.set_epsilon(np.full(driver.shape, 1.7, dtype=np.float32))
        rng = np.random.default_rng(20260902)
        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
            array = getattr(driver.fields, name, None)
            if array is not None:
                array[...] = rng.standard_normal(array.shape).astype(np.float32)
        return driver

    def bytes_of(driver):
        return {name: getattr(driver.fields, name).copy()
                for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
                if getattr(driver.fields, name, None) is not None}

    for family, near, wall, far in (
            ("B", stepping.fill_symmetry_bc_B, stepping.zero_metal_B,
             stepping.fill_folded_far_ghosts_B),
            ("D", stepping.fill_symmetry_bc_D, stepping.zero_metal_D,
             stepping.fill_folded_far_ghosts_D)):
        drivers_order = build()
        before = bytes_of(drivers_order)
        near(drivers_order.fields)
        wall(drivers_order.fields)
        after_wall = bytes_of(drivers_order)
        far(drivers_order.fields)
        driver_order = bytes_of(drivers_order)

        armed = build()
        near(armed.fields)
        after_near = bytes_of(armed)
        far(armed.fields)
        after_far = bytes_of(armed)
        wall(armed.fields)
        kernel_order = bytes_of(armed)

        moved_wall = any(not np.array_equal(after_wall[n], before[n])
                         for n in before)
        moved_far = any(not np.array_equal(after_far[n], after_near[n])
                        for n in after_near)
        assert moved_wall, f"{family}: the wall clear wrote nothing; the case is vacuous"
        assert moved_far, f"{family}: the far fill wrote nothing; the case is vacuous"
        for name in driver_order:
            assert np.array_equal(driver_order[name], kernel_order[name]), (
                f"{family}: {name} differs between the driver's own order "
                f"(near, wall, far) and a fused fill plan's (near+far, wall)")
        drivers_order.close()
        armed.close()


def test_launch_counters_count_the_sub_steps_a_kernel_ACTUALLY_ran(monkeypatch):
    """The plan's one measurement, against its many statements about the plan.

    ``step_path``, ``slots`` and ``families`` all describe a DECISION. This counts
    what the decision did, and the end-to-end dispatch proof reads it: a leg whose
    record says "fused" and whose counters read zero fell back and matched for that
    reason, which is the vacuous pass and not a pass.
    """
    driver = numpy_driver()
    counting_sub_steps(monkeypatch)
    plans = {"step_B": CountingPlan("step_B"), "update_P": [CountingPlan("p0")]}
    plan = install_plan(monkeypatch, driver, plans,
                        {"step_B": "PML", "update_P": "ADE"})
    driver.run(num_steps=3)
    counters = plan.launch_counters
    assert counters["step_B"]["dispatches"] == 3
    assert counters["update_P"]["dispatches"] == 3
    assert counters["step_B"]["arm"] == "PML"
    assert driver.fast_path_report()["launch_counters"] == counters, (
        "the artifact a user reads must carry the live counts, not the freeze's zeros")
    driver.close()


def test_a_dispatched_record_with_a_zero_counter_is_the_vacuous_pass(monkeypatch):
    """The identity guard's False: the record says dispatched, nothing ran.

    Reachable without anything unusual — a ``Fields`` rebound outside
    ``invalidate_fast_path`` is exactly the case :meth:`FastPathPlan.dispatch`
    guards — and it produces a run that is on the array path while every planning
    field of the artifact reads "fused". The counter is what tells them apart.
    """
    driver = numpy_driver()
    counts = counting_sub_steps(monkeypatch)
    plan = install_plan(monkeypatch, driver, {"step_B": CountingPlan("step_B")},
                        {"step_B": "PML"})
    driver.step()
    object.__setattr__(plan, "fields", object())  # A Fields the plan no longer holds.
    driver.step()
    assert plan.launch_counters["step_B"]["dispatches"] == 1, (
        "one step dispatched and one fell through the identity guard")
    assert counts["step_B"] == 1, "and the array call served the step that fell through"
    assert driver.fast_path_report()["step_path"] == "fused", (
        "while the planning half of the artifact still reads fused — the point")
    driver.close()


class LapsingPolicy:
    """The subnormal-policy module a plan holds its licence against, with a switch.

    ``uninstall_subnormal_policy`` is exported from ``meep_gpu/__init__.py``, so a
    caller reaching this state mid-run is a supported gesture, not a contrivance.
    """

    def __init__(self):
        self.installed = True

    def policy_epoch(self):
        return 7

    def policy_is_installed(self):
        return self.installed

    def get_subnormal_policy(self):
        return "keep" if self.installed else None


def test_a_step_that_aborts_between_a_pairs_consults_cannot_skip_the_next_ones(monkeypatch):
    """THROUGH THE SHIPPED ``step()``: a leaked commitment SKIPS a sub-step entirely.

    A fused pair's leading consult performs both halves and hands the absorbed consult a
    permission to dispatch regardless of the policy licence — refusing it after the
    launch is what would double-apply the constitutive half. That permission used to be
    released only by the consult that consumed it, so a step which reached
    ``dispatch('step_B')`` and never ``dispatch('update_H')`` left it standing.

    Everything between the two sites in ``FdtdDriver.step`` (driver.py:3286-3296) can
    raise — a source ``inject``, the symmetry fill, the metallic-wall pass — and one of
    them is stood in for here. The step AFTER that is the defect: the licence has
    genuinely lapsed, the leading slot correctly falls back to the array curl, and the
    absorbed slot then answers True off the previous step's permission and runs only the
    sentinel. ``update_H`` happens on neither path, silently, for every remaining step.
    """
    driver = numpy_driver()
    counts = counting_sub_steps(monkeypatch)
    pair = CountingPlan("fused pair B")
    sentinel = launch_module.NoopPlan("update_H", pair)
    policy = LapsingPolicy()
    plan = install_plan(monkeypatch, driver, {"step_B": pair, "update_H": sentinel},
                        {"step_B": "fused pair B", "update_H": "fused pair B"})
    object.__setattr__(plan, "policy_licence", (policy, "keep", 7))

    armed = {"raise": True}
    real_fill = driver_module.fill_symmetry_bc_B

    def aborting_fill(fields):
        if armed["raise"]:
            armed["raise"] = False
            raise RuntimeError("something between the pair's two consults raised")
        return real_fill(fields)

    monkeypatch.setattr(driver_module, "fill_symmetry_bc_B", aborting_fill)

    with pytest.raises(RuntimeError):
        driver.step()
    assert pair.runs == 1, "the leading consult dispatched before the step aborted"
    assert plan.launches == {"step_B": 1}, "the absorbed consult was never reached"

    policy.installed = False          # the licence lapses between the two steps
    driver.step()
    assert counts["step_B"] == 1, "the lapsed licence puts the leading slot on the array"
    assert counts["update_H"] == 1, (
        "and update_H must follow it there. Answering True off the aborted step's "
        "commitment runs only the sentinel, so the sub-step happens on neither path")
    assert pair.runs == 1, "no second launch"
    assert plan.launches == {"step_B": 1}, (
        "the sentinel dispatched on the aborted step's commitment — the counter that "
        "tells a dispatched slot from a fallen-back one says it ran")
    assert plan.licence_lapses == {"step_B": 1, "update_H": 1}
    assert sentinel.absorbed_by is pair          # the shape under test, not a stand-in
    driver.close()


def test_an_emptied_launch_grid_reports_zero_programs(monkeypatch):
    """A launch over no data is a launch. ``programs_per_dispatch`` separates them.

    Not hypothetical: :func:`fastpath._warm_with_empty_grid` empties exactly this
    attribute on purpose so a plan-time warm compiles and enqueues nothing. A slot
    that reached a user with the grid still empty would count dispatches and do no
    arithmetic.
    """
    driver = numpy_driver()
    counting_sub_steps(monkeypatch)
    empty, live = CountingPlan("empty"), CountingPlan("live")
    empty._grid = (0,)
    live._grid = (4096,)
    plan = install_plan(monkeypatch, driver, {"step_B": empty, "step_D": live},
                        {"step_B": "PML", "step_D": "PML"})
    driver.step()
    counters = plan.launch_counters
    # THE WHOLE ENTRY, not three keys of it, so a counter added later has to be
    # SEEN here rather than sliding past a subset comparison. ``plan_launches`` is
    # the second table's witness (a Metal KernelPlan spells no launch grid, so its
    # ``programs_per_dispatch`` is None by construction) and it is None on a Triton
    # plan, which is UNKNOWN and not zero. ``backend`` is the THIRD table's: a merged
    # NVIDIA step launches both composers' kernels over the same arrays, so a count
    # that did not say whose could not be checked at all; a single-table plan
    # answers with its own table, which is what this one is.
    assert counters["step_B"] == {"arm": "PML", "backend": "triton",
                                  "dispatches": 1,
                                  "programs_per_dispatch": 0,
                                  "plan_launches": None}
    assert counters["step_D"]["programs_per_dispatch"] == 4096
    driver.close()


def test_a_plan_that_spells_no_launch_grid_reports_unknown_not_zero(monkeypatch):
    """``None`` is the third answer, and collapsing it into 0 would be a false alarm."""
    driver = numpy_driver()
    counting_sub_steps(monkeypatch)
    plan = install_plan(monkeypatch, driver, {"step_B": CountingPlan("step_B")},
                        {"step_B": "PML"})
    driver.step()
    assert plan.launch_counters["step_B"]["programs_per_dispatch"] is None
    driver.close()


def test_synchronize_magnetic_fields_dispatches_the_same_two_slots(monkeypatch):
    """The B/H half of ``step``, verbatim — so it takes the same dispatch."""
    driver = numpy_driver()
    counts = counting_sub_steps(monkeypatch)
    b, h = CountingPlan("step_B"), CountingPlan("update_H")
    install_plan(monkeypatch, driver, {"step_B": b, "update_H": h},
                 {"step_B": "PML", "update_H": "ordinary"})
    driver.step()
    driver.synchronize_magnetic_fields()
    driver.restore_magnetic_fields()
    assert (b.runs, h.runs) == (2, 2), "the synchronization half-step dispatches too"
    assert counts["step_B"] == counts["update_H"] == 0
    driver.close()


def test_a_stale_plan_is_no_plan_inside_synchronize(monkeypatch):
    driver = numpy_driver()
    counts = counting_sub_steps(monkeypatch)
    b, h = CountingPlan("step_B"), CountingPlan("update_H")
    install_plan(monkeypatch, driver, {"step_B": b, "update_H": h},
                 {"step_B": "PML", "update_H": "ordinary"})
    driver.step()
    driver._fast_path_stale = True  # As after any material mutation.
    driver.synchronize_magnetic_fields()
    driver.restore_magnetic_fields()
    assert (b.runs, h.runs) == (1, 1), "a stale plan must not be launched"
    assert counts["step_B"] == counts["update_H"] == 1
    driver.close()


def test_active_step_path_reads_fused_only_when_a_kernel_will_launch(monkeypatch):
    driver = numpy_driver()
    assert driver.active_step_path == "array"
    install_plan(monkeypatch, driver, {"step_B": CountingPlan("b")}, {"step_B": "PML"})
    driver.step()
    assert driver.active_step_path == "fused"
    driver.invalidate_fast_path()
    monkeypatch.setattr(driver_module, "plan_fast_path", lambda *args: None)
    driver.step()
    assert driver.active_step_path == "array"
    driver.close()


def test_the_driver_serves_the_dispatch_report(monkeypatch):
    driver = numpy_driver()
    plan = install_plan(monkeypatch, driver, {"step_B": CountingPlan("b")},
                        {"step_B": "PML"})
    driver.step()
    # EQUAL, not identical: report() now merges the plan's live launch counters
    # over the frozen record, so each call builds a fresh dict. The counters have
    # to be read at the moment the question is asked -- the record itself was
    # emitted at the freeze, when every count is zero by construction.
    assert driver.fast_path_report() == plan.report()
    assert driver.fast_path_report()["launch_counters"]["step_B"]["dispatches"] == 1
    driver.close()


#: The two drivers that hold no plan after a freeze under ``MEEP_GPU_DISPATCH=0``,
#: and the reason each one's record carries. ``None`` is a ``prefer_gpu=False``
#: driver, the NumPy reference: it never plans, whatever the enable says, and its
#: freeze publishes the reference record. ``"metal"`` is a GPU driver's engine: its
#: freeze goes through the planner, which refuses at the enable rung.
UNPLANNED_DRIVERS = [
    pytest.param(None, True, fastpath.REFERENCE_REASON, id="reference"),
    pytest.param("metal", False, f"dispatch disabled by {fastpath.DISPATCH_ENABLE}=0",
                 id="gpu_driver_refused_at_the_enable"),
]


@pytest.mark.parametrize("gpu,reference,reason", UNPLANNED_DRIVERS)
def test_a_refused_run_still_serves_a_report_naming_the_reason(monkeypatch, gpu,
                                                               reference, reason):
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "0")
    driver = numpy_driver()
    driver.gpu = gpu
    driver.step()
    report = driver.fast_path_report()
    assert report is not None and report["decision"] == "refused"
    assert report["reference_driver"] is reference
    assert report["refused_because"] == reason
    driver.close()


def test_the_kill_switch_disables_dispatch_rather_than_hiding_it(monkeypatch):
    """From the env var, on a driver that would otherwise have dispatched.

    IT HAS TO BE THE OPERATIVE RUNG. This test used to drive a NumPy driver, which
    the ladder refuses one rung LOWER ("backend is not CuPy"), so all three of its
    assertions held whether or not the kill switch existed — mutating
    ``fused_dispatch_enabled`` to ``lambda: True`` failed four other tests and left
    this one green. The grid's backend is stubbed cupy-like here so the kill switch
    is the only thing that can refuse, and the paired case below proves the same
    configuration DOES dispatch without it.
    """
    driver = numpy_driver()
    counts = counting_sub_steps(monkeypatch)
    b = CountingPlan("step_B")
    monkeypatch.setattr(driver.grid, "xp", numpy_backed_cupy_alias())
    driver.gpu = "cuda"  # the engine is stubbed cupy-like: a CUDA driver plans
    stub_triton(monkeypatch)  # The real planner, not an installed plan.
    install_composer(monkeypatch, step_plan({"step_B": b}, {"step_B": "PML"}))

    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    driver.run(num_steps=2)
    assert driver.active_step_path == "array"
    assert b.runs == 0 and counts["step_B"] == 2
    assert "kill switch" in driver.fast_path_report()["refused_because"]

    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH)
    driver.invalidate_fast_path()
    driver.run(num_steps=2)
    assert driver.active_step_path == "fused", (
        "without the kill switch this same configuration must dispatch, or the "
        "assertions above measure the ladder's other rungs instead")
    assert b.runs == 2 and counts["step_B"] == 2
    driver.close()


# ---------------------------------------------------------------------------
# The fold (C3b): the completeness rule's blind spot
# ---------------------------------------------------------------------------
#
# Rule C3 fires on a slot the composer FILLED. A mirror run whose two fill arms
# both REFUSE fills neither, passes C3, and dispatched four folded kernel slots
# with the ghost fills on the array path — dispatch ENABLED BY a sibling
# product's refusal, and the composition ``fields.py`` twice states this seam
# keeps off the kernels. The fold is now read from the grid.
#
# NARROWED 2026-09-02 and NOT retired. The blanket "every fold refuses" clause was
# there because the seam could not run a fill slot at all; with the two consults it
# can, so what survives is exactly the hole above — a folded run whose fill slots
# are NOT covered while other slots are. The cases below are unchanged because they
# are that hole; the case that changed is the one where the fills ARE covered, and
# it lives with the completeness tests as
# ``test_a_folded_run_that_fills_both_fill_slots_now_dispatches``.


def folded_grid(axes=(0,)):
    grid = cupy_like_grid()
    grid.is_mirrored = lambda axis: axis in axes
    grid.has_symmetry = True
    return grid


def test_a_mirror_folded_run_refuses_even_when_both_fill_arms_declined(monkeypatch):
    """The measured hole: complex storage refuses both fill arms, so C3 saw nothing."""
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan(
        {"step_B": CountingPlan("b"), "update_H": CountingPlan("h"),
         "step_D": CountingPlan("d"), "update_E": CountingPlan("e")},
        {"step_B": "folded complex PML", "update_H": "folded complex",
         "step_D": "folded complex PML", "update_E": "folded complex"},
        {"fill_B": ("mirror fill: force_complex_fields=True (complex64 storage is "
                    "not carried)",),
         "fill_D": ("folded complex fill: the expansion probe artifact lacks the "
                    "pattern this tranche adds",)}))
    plan = fastpath.plan_fast_path(object(), object(), folded_grid())
    assert plan is None, "a folded run must not dispatch while the seam cannot fill"
    record = fastpath.last_dispatch_report()
    assert "mirror-folded" in record["refused_because"]
    assert "X" in record["refused_because"]
    assert record["step_path"] == "array"


def test_the_fold_refusal_does_not_depend_on_which_arms_won(monkeypatch):
    """Read from the GRID, not sniffed from arm labels.

    A label rule decides the fold from the same product selection whose gaps made
    the hole, and answers "not folded" for exactly the case that matters.
    """
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan(
        {"update_P": [CountingPlan("p0")]}, {"update_P": "ADE update_P"}))
    assert fastpath.plan_fast_path(object(), object(), folded_grid(axes=(2,))) is None
    assert "Z" in fastpath.last_dispatch_report()["refused_because"]


@pytest.mark.parametrize("covered,uncovered", [("fill_B", "fill_D"),
                                               ("fill_D", "fill_B")])
def test_a_fold_with_only_one_fill_covered_still_refuses_the_whole_plan(
        monkeypatch, covered, uncovered):
    """HALF A FILL SEAM IS THE HOLE, and both halves are driven.

    The narrowing is "BOTH fill slots carry a kernel", not "a fill slot does". A
    rule written as ``any`` rather than ``all`` passes this file's other fold cases
    unchanged and admits exactly the mixed composition C3b exists to refuse, on
    whichever family the composer happened to cover — so the case is parametrised
    over which one that is.
    """
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan(
        {"step_B": CountingPlan("b"), "update_H": CountingPlan("h"),
         "step_D": CountingPlan("d"), "update_E": CountingPlan("e"),
         covered: CountingPlan(covered)},
        {"step_B": "folded PML", "update_H": "folded", "step_D": "folded PML",
         "update_E": "folded", covered: "mirror fill"},
        {uncovered: (f"mirror fill: {uncovered} was refused",)}))
    assert fastpath.plan_fast_path(object(), object(), folded_grid()) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "mirror-folded" in reason and uncovered in reason, reason
    assert covered not in reason, (
        f"{covered} IS covered; naming it would tell a reader to fix the wrong half")
    assert "warm pass" not in reason, (
        "the rung UNDER TEST is 6b, judged before anything compiles. The post-warm "
        "copy of the same rule catches this composition too, so an assertion that "
        "only read 'is None' would pass with 6b narrowed to 'BOTH fills uncovered' "
        "— measured, with that exact mutation")


def test_an_unfolded_grid_of_the_same_shape_still_dispatches(monkeypatch):
    """The control: the refusal above must be the FOLD and not the stub."""
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    assert plan is not None and plan.slots == ("step_B",)


def test_the_run_shape_records_the_fold_the_composer_was_asked_about(monkeypatch):
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "folded PML"}))
    fastpath.plan_fast_path(object(), object(), folded_grid(axes=(0, 1)))
    shape = fastpath.last_dispatch_report()["run_shape"]
    assert shape["folded"] == "mirror plane on X, Y"


# ---------------------------------------------------------------------------
# The device: the other half of the codegen input
# ---------------------------------------------------------------------------


class StubDeviceBackend(types.SimpleNamespace):
    pass


def cupy_like_grid_with_device(capability=(8, 6), name=b"NVIDIA RTX A6000"):
    grid = cupy_like_grid()
    runtime = types.SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {
            "name": name, "major": capability[0], "minor": capability[1]},
        runtimeGetVersion=lambda: 11080,
        driverGetVersion=lambda: 12030,
    )
    grid.xp.cuda = types.SimpleNamespace(runtime=runtime)
    return grid


#: Compute capabilities that stand in for a SUPPORTED device no cited gate ran on, in
#: order of preference. A test takes the first ones that NEITHER NVIDIA table
#: certifies and both support (:func:`supported_uncertified_capabilities`), so a round
#: that certifies another architecture (9.0 is the next) leaves the stand-ins
#: uncertified instead of turning them into certified devices under the tests that
#: read them.
SUPPORTED_UNCERTIFIED_CANDIDATES = ((8, 9), (7, 5), (8, 0), (7, 0))

#: Compute capabilities outside the range either NVIDIA table supports, in order of
#: preference: one below the floor and two above the ceiling.
UNSUPPORTED_CANDIDATES = ((6, 1), (10, 0), (12, 0))


def _certified_anywhere():
    from meep_gpu import fastpath_cuda

    return (set(fastpath.validated_compute_capabilities())
            | set(fastpath_cuda.validated_compute_capabilities()))


def supported_uncertified_capabilities(count=1):
    """The first ``count`` candidates both tables support and neither certifies.

    Fails rather than returning fewer: an empty parametrization is a skip, and a test
    that silently ran on nothing would read as a pass. Supported is read from the
    package (:func:`fastpath.capability_supported`), never typed here.
    """
    certified = _certified_anywhere()
    chosen = [capability for capability in SUPPORTED_UNCERTIFIED_CANDIDATES
              if f"{capability[0]}.{capability[1]}" not in certified
              and all(fastpath.capability_supported(capability, table) is True
                      for table in ("triton", fastpath.CUDA_TABLE))]
    assert len(chosen) >= count, (
        f"only {len(chosen)} of {SUPPORTED_UNCERTIFIED_CANDIDATES} are supported and "
        f"uncertified (certified: {sorted(certified)}); add a candidate")
    return chosen[:count]


def unsupported_capabilities(count=1):
    """The first ``count`` candidates NEITHER table supports (so neither certifies)."""
    chosen = [capability for capability in UNSUPPORTED_CANDIDATES
              if all(fastpath.capability_supported(capability, table) is False
                     for table in ("triton", fastpath.CUDA_TABLE))]
    assert len(chosen) >= count, (
        f"only {len(chosen)} of {UNSUPPORTED_CANDIDATES} are unsupported; add a "
        "candidate")
    return chosen[:count]


def test_the_recorded_gates_name_the_architecture_they_ran_on():
    """DERIVED FROM THE LEDGER ON DISK, not compared against a typed tuple.

    The list used to be one hand-written key, and asserting ``("8.6",)`` here was the
    same defect one layer up: a round that certified a second architecture would have
    to come and edit this line, so the line said what it had always said. The
    expectation is re-derived instead — the intersection, over the gates the arms
    cite, of the capabilities whose run record still binds the entry's bytes — read
    out of ``fingerprints.json`` directly rather than through the package's cache.
    ``CERTIFIED_CAPABILITY`` is asserted to be IN it rather than to BE it, because
    this file's run-fact lookups read that architecture's record and would otherwise
    go quiet if it ever left the ledger.
    """
    import json
    import pathlib

    ledger = json.loads(
        (pathlib.Path(fastpath.__file__).parent / "triton_kernels"
         / "fingerprints.json").read_text(encoding="utf-8"))
    derived = None
    for _family, gate in fastpath.ARM_CERTIFICATION.values():
        live = set(fastpath.live_capabilities(ledger.get(gate)))
        derived = live if derived is None else (derived & live)
    assert derived, (
        "fingerprints.json must record which GPU architecture the gates ran on, and "
        "every cited gate must have a run that still binds the shipped bytes")
    assert fastpath.validated_compute_capabilities() == tuple(sorted(derived))
    assert CERTIFIED_CAPABILITY in derived, (
        f"this file reads run facts on {CERTIFIED_CAPABILITY} and the ledger now "
        f"derives {sorted(derived)}")


@pytest.mark.parametrize("value,expected", [((8, 6), "8.6"), ("86", "8.6"),
                                            ("8.6", "8.6"), ("90", "9.0")])
def test_the_capability_spellings_normalize_to_one(value, expected):
    """CuPy hands the same number out three ways; a comparison needs one spelling."""
    assert fastpath._normalized_capability(value) == expected


def test_a_device_the_gates_never_ran_on_is_refused_by_name_when_it_is_unsupported(
        monkeypatch):
    """Triton generates PTX for an ARCH, so a bit-identity claim is arch-scoped; a
    device outside the supported range is refused by name, with the way past it."""
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    (capability,) = unsupported_capabilities(1)
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    record = fastpath.last_dispatch_report()
    spelled = f"{capability[0]}.{capability[1]}"
    assert spelled in record["refused_because"]
    assert "outside the compute capabilities that table supports" in \
        record["refused_because"]
    assert record["refused_because"].endswith(fastpath.UNSUPPORTED_HINT)
    assert record["environment"]["device"]["name"] == "another device"
    assert record["environment"]["device_supported"] is False


def test_a_supported_device_the_gates_never_ran_on_dispatches_uncertified(monkeypatch):
    """A device in the supported range that no gate ran on runs by default, and the
    record says it is not certified; ``MEEP_GPU_ALLOW_UNCERTIFIED=0`` refuses it."""
    stub_triton(monkeypatch)
    block_cuda(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    (capability,) = supported_uncertified_capabilities(1)
    grid = cupy_like_grid_with_device(capability=capability, name=b"another device")
    plan = fastpath.plan_fast_path(object(), object(), grid)
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = plan.report()
    assert record["certified"] is False
    assert record["environment"]["device_certified"] is False
    assert record["environment"]["device_supported"] is True
    assert [entry["table"] for entry in record["uncertified"]["served"]] == ["triton"]
    monkeypatch.setenv(fastpath.UNCERTIFIED_SWITCH, "0")
    assert fastpath.plan_fast_path(object(), object(), grid) is None
    assert fastpath.last_dispatch_report()["refused_because"].endswith(
        fastpath.CERTIFIED_ONLY_TAIL)


def test_the_certified_device_dispatches_and_is_recorded(monkeypatch):
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    plan = fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device())
    assert plan is not None
    device = plan.report()["environment"]["device"]
    assert device["compute_capability"] == "8.6"
    assert device["name"] == "NVIDIA RTX A6000"
    assert device["cuda_runtime"] == 11080 and device["cuda_driver"] == 12030
    assert plan.report()["environment"]["device_certified"] is True


def test_an_unreadable_device_is_recorded_as_unknown_and_not_refused(monkeypatch):
    """Three-valued: refusing on a fact that was not read would be asserting it."""
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                      {"step_B": "PML"}))
    assert plan is not None
    environment = plan.report()["environment"]
    assert environment["device_certified"] is None
    assert "unreadable" in environment["device"]


# ---------------------------------------------------------------------------
# Provenance: what the artifact says the dispatched kernels ride on
# ---------------------------------------------------------------------------


#: How a family block of ``family_recert_2026-08-14`` names its own artifact: the path
#: ``recut_composition_records.py --family-recert`` writes from the campaign root it
#: read, ``parity/meep_gpu/results/<root>/<family>/gate.json``.
_FAMILY_ARTIFACT = re.compile(
    r"^parity/meep_gpu/results/(?P<root>[^/]+)/(?P<family>[^/]+)/gate\.json$")

#: The one timestamp spelling the family re-cut and the fleet campaign rows write.
_UTC = "%Y-%m-%dT%H:%M:%SZ"


def _family_recert_run(capability=CERTIFIED_CAPABILITY):
    """``family_recert_2026-08-14``'s run record on ``capability``, which must be live."""
    entry = fastpath._fingerprints("triton")[fastpath.FAMILY_RECERT_GATE]
    assert capability in fastpath.live_capabilities(entry), (
        f"{fastpath.FAMILY_RECERT_GATE} has no live run on {capability}: live "
        f"{list(fastpath.live_capabilities(entry))}")
    return entry[fastpath.RUNS][capability]


def _family_recut_tool():
    """``parity/meep_gpu/recut_composition_records.py``: the family re-cut's own rules.

    IMPORTED, NEVER RESTATED, for the reason :func:`_recut_tool` gives: where a family
    gate's artifact states its step budget, the sentence a block carries when it states
    none, where the campaign row lives and how a gate's release is read are decided by
    the tool that writes the family blocks, and a second spelling here would be a second
    place for them to disagree. The harness tree is not part of the standalone package,
    so its absence is a declared resource.
    """
    import importlib

    parity = pathlib.Path(fastpath.__file__).resolve().parent.parent / "parity" / "meep_gpu"
    if not (parity / "recut_composition_records.py").is_file():
        requires_resource_skip(
            "parity_meep_gpu_harness",
            "parity/meep_gpu/recut_composition_records.py holds the family re-cut's "
            "rules and is not shipped with the standalone package")
    if str(parity) not in sys.path:
        sys.path.insert(0, str(parity))
    return importlib.import_module("recut_composition_records")


def assert_the_family_block_names_its_own_run(family, block, run):
    """RECORD LEVEL, no artifact needed: the block is the record of ONE run of ONE gate.

    The run id is the campaign root the block's own ``records`` path names, the log
    sits beside that artifact, the run-level ``records`` line names the root, the exit
    code is 0, both pins are digests, and the run-level ``recorded_utc`` -- stamped by
    the re-cut when it wrote the record, not read from the artifact -- is a UTC
    timestamp no earlier than the campaign row's ``started_utc`` for this family.
    """
    import datetime

    match = _FAMILY_ARTIFACT.match(str(block.get("records")))
    assert match and match["family"] == family, (
        f"{family}: records {block.get('records')!r} does not name "
        "parity/meep_gpu/results/<root>/<family>/gate.json")
    root = match["root"]
    assert block["run_id"] == root, (
        f"{family}: run_id {block['run_id']!r} is not the campaign root its own "
        f"artifact path names ({root!r})")
    assert block["log"] == f"parity/meep_gpu/results/{root}/{family}/gate.log", (
        family, block["log"])
    assert f"parity/meep_gpu/results/{root}" in run["records"], (family, run["records"])
    assert block["rc"] == 0, (family, block["rc"])
    for field in ("artifact_sha256", "log_sha256"):
        assert re.fullmatch(r"[0-9a-f]{64}", str(block.get(field))), (family, field)
    started = datetime.datetime.strptime(block["started_utc"], _UTC)
    recorded = datetime.datetime.strptime(run["recorded_utc"], _UTC)
    assert recorded >= started, (
        f"{family}: the record was cut at {run['recorded_utc']}, before the run it "
        f"describes started ({block['started_utc']})")


def assert_the_family_budget_is_its_artifacts(family, block, payload, recut):
    """The block's ``step_budget`` is what THIS family's artifact states, and nothing else.

    A stated budget is a non-empty mapping equal to the artifact's, read where the
    re-cut reads it: measured 2026-10-04 over results/triton_fleet_2026-10-03_cc86, the
    six family gates that state a budget state it as a block, and offdiag, nonlinear
    and folded_offdiag state none. The tool's "not stated" sentence is accepted only
    where the artifact states none at any of those places.
    """
    stated, read_from = recut._step_budget(payload)
    assert block["step_budget"] == stated, (
        f"{family}: the block records step_budget {block['step_budget']!r} and its own "
        f"artifact states {stated!r} (read from {read_from})")
    if read_from is None:
        assert stated == recut.BUDGET_NOT_STATED, family
    else:
        assert isinstance(stated, dict) and stated, (family, read_from, stated)
        assert all(isinstance(key, str) for key in stated), (family, sorted(stated))


def assert_the_family_block_is_its_artifacts(family, block, package):
    """ARTIFACT LEVEL: every fact of the block read back from the run it names.

    The artifact and its log hash to the block's pins; the campaign row for this gate
    in the root's ``campaign.json`` exited with the block's ``rc``, released, names the
    same artifact digest, started at the block's ``started_utc`` and ran on the block's
    GPU index; the artifact itself reads as released; the block's ``verdict`` is the
    artifact's ``canonical_verdict``; and its budget is the artifact's
    (:func:`assert_the_family_budget_is_its_artifacts`). Skips only on the results
    tree's absence or the harness's.
    """
    import hashlib

    assert _FAMILY_ARTIFACT.match(str(block.get("records"))), (
        f"{family}: records {block.get('records')!r} names no gate artifact of a "
        "campaign root, so there is no run to read the block back from")
    recut = _family_recut_tool()
    results = _results_tree_or_skip(package, f"the {family} family gate")
    artifact = package.parent / block["records"]
    assert artifact.is_file(), (
        f"{family}: {block['records']} is not here while the results tree is: the "
        "record cites a run this checkout does not hold")
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == block["artifact_sha256"], (
        f"{family}: {block['records']} is not the artifact the block pins")
    log = package.parent / block["log"]
    assert log.is_file() and hashlib.sha256(log.read_bytes()).hexdigest() == \
        block["log_sha256"], f"{family}: {block['log']} is not the log the block pins"
    row, why_not = recut.campaign_row(results / block["run_id"], family)
    assert row is not None, f"{family}: {why_not}"
    assert row.get("exit_code") == block["rc"], (family, row.get("exit_code"))
    assert row.get("released") is True, (family, row.get("released"))
    assert row.get("artifact_sha256") == block["artifact_sha256"], family
    assert row.get("started_utc") == block["started_utc"], (
        family, row.get("started_utc"), block["started_utc"])
    assert row.get("gpu") == block["pinned_gpu_index"], (family, row.get("gpu"))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    verdict, read_from = recut.released(payload)
    assert verdict is True, f"{family}: the artifact reads released={verdict!r} ({read_from})"
    assert block["verdict"] == payload.get("canonical_verdict"), family
    assert_the_family_budget_is_its_artifacts(family, block, payload, recut)


def test_the_nine_families_certified_this_round_carry_real_provenance(monkeypatch):
    """Their gate was a path into a GITIGNORED directory, so it resolved to nothing.

    The nine families' own rows live inside the RUN record now (``families`` is a run
    field), so the per-family ``run_id`` and ``rc`` are reached only by naming the
    architecture — which is the point: a run id is a statement about one run.

    RE-CUT FROM THE ROUND'S OWN ARTIFACTS, NOT TRANSCRIBED. Until the 2026-10-03 round
    the blocks were the 2026-08-14 launcher's transcription and this test pinned its
    literals (that date, ``recert_<family>_*`` run ids). ``recut_composition_records.py
    --family-recert`` rebuilds every block from a fleet campaign, so what is pinned is
    the relation each block must satisfy with the run it names — for all nine
    families, not one: see :func:`assert_the_family_block_names_its_own_run` and
    :func:`assert_the_family_block_is_its_artifacts`.
    """
    package = pathlib.Path(fastpath.__file__).resolve().parent
    run = _family_recert_run()
    entry = fastpath._certification_for("nonlinear", capability=CERTIFIED_CAPABILITY)
    assert entry["gate"] == fastpath.FAMILY_RECERT_GATE
    assert entry["capability"] == CERTIFIED_CAPABILITY
    assert entry["recorded_utc"] == run["recorded_utc"]
    assert entry["host"] == run["host"] and "A6000" in entry["host"]
    assert entry["run_id"] == run["families"]["nonlinear"]["run_id"]
    assert entry["rc"] == 0
    families = run["families"]
    assert sorted(families) == sorted(
        {family for family, gate in fastpath.ARM_CERTIFICATION.values()
         if gate == fastpath.FAMILY_RECERT_GATE}), sorted(families)
    for family, block in sorted(families.items()):
        assert_the_family_block_names_its_own_run(family, block, run)
    for family, block in sorted(families.items()):
        assert_the_family_block_is_its_artifacts(family, block, package)


def test_each_family_is_credited_with_its_own_step_budget_not_the_campaigns(monkeypatch):
    """The constant said 24000 steps on two cases most families never ran.

    A budget is the measurement of ONE run, so every lookup here names the
    architecture; asked without one, each arm gets the "not stated per family"
    sentence instead and the set below would collapse to a single element — the very
    shape this test refuses, arrived at from the other direction.

    WHAT A BUDGET IS NOW. The family re-cut copies the budget each gate's artifact
    STATES, as the artifact states it (a block such as ``{"engine": 4, "reference":
    4}``), and writes its "not stated" sentence where the artifact states none. So the
    credit is checked against each family's OWN artifact rather than against
    "80/80"-style literals of the retired transcription: every arm quotes its own
    family's block, each block equals what that family's artifact states, and the three
    families those literals named (BFAST, special-kz, complex) state a budget.
    """
    families = _family_recert_run()["families"]

    def budget(arm):
        return fastpath._certification_for(
            arm, capability=CERTIFIED_CAPABILITY)["step_budget"]

    for arm, (family, gate) in sorted(fastpath.ARM_CERTIFICATION.items()):
        if gate == fastpath.FAMILY_RECERT_GATE:
            assert budget(arm) == families[family]["step_budget"], (arm, family)
    for arm in ("BFAST run", "complex beta run", "complex"):
        stated = budget(arm)
        assert isinstance(stated, dict) and stated, (arm, stated)
        assert all(isinstance(key, str) for key in stated), (arm, sorted(stated))
    budgets = {json.dumps(budget(arm), sort_keys=True) for arm in fastpath.ARM_CERTIFICATION}
    assert len(budgets) > 1, "one budget for every family is the defect, not the fix"
    package = pathlib.Path(fastpath.__file__).resolve().parent
    recut = _family_recut_tool()
    _results_tree_or_skip(package, "the nine family gates state their budgets")
    for family, block in sorted(families.items()):
        artifact = package.parent / block["records"]
        assert artifact.is_file(), (family, block["records"])
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert_the_family_budget_is_its_artifacts(family, block, payload, recut)


def test_a_gate_whose_re_run_was_blocked_says_so_in_the_artifact():
    """Two dispatchable arms cite gates whose re-run exited rc=1 and are BLOCKED."""
    for arm in ("conductive PML", "no-PML"):
        entry = fastpath._certification_for(arm)
        assert entry["last_re_run"]["rc"] == 1, arm
        assert entry["last_re_run"]["blocked_and_why"], arm
        assert "did NOT re-run clean" in entry["last_re_run"]["read_this_as"]


def test_the_gate_the_sweep_did_not_re_run_carries_that_fact_and_its_digests():
    """The two most-dispatched arms ride on it, and its probe/adapter drift is live."""
    for arm in ("PML", "ordinary"):
        entry = fastpath._certification_for(arm)
        assert entry["gate"] == "bit_identity_gate"
        assert "probe/adapter drift" in entry["not_re_run"], arm
        digests = entry["recorded_digests"]
        assert digests["probe_sha256"] and digests["adapter_sha256"], arm


def test_the_family_modules_drift_declaration_matches_the_shipped_bytes():
    """The weld the nine families never had.

    ``host_sha256`` covers ten files and none of the nine families' modules, so a
    change to one of them was invisible to every weld in the tree — two had
    already drifted from the bytes their certification was cut against. The
    declaration is recomputed here in both directions: an undeclared drift fails,
    and a declaration that outlives its drift fails too. The FILE SET is what is
    asserted; the digest beside each name is the value at declaration time, not a
    pin, because a declared file may keep moving while its recut is pending.

    THE TWO HALVES SIT AT DIFFERENT LEVELS and that is what the comparison is
    between: the nine families' ``source_sha256_at_recert`` maps belong to the RUN
    that measured them (``families`` is a run field, under ``fastpath.RUNS``), while
    the drift DECLARATION is a statement about the tree and stays on the entry. So
    the declaration is compared against the drift of one architecture's run, which
    is the only level where "what was certified" has a single answer.
    """
    import hashlib
    import json
    import pathlib

    package = pathlib.Path(fastpath.__file__).parent / "triton_kernels"
    record = json.loads((package / "fingerprints.json").read_text(encoding="utf-8"))
    entry = record[fastpath.FAMILY_RECERT_GATE]
    declared = entry["source_drift_since_recert"]

    drifted = {}
    for family in entry[fastpath.RUNS][CERTIFIED_CAPABILITY]["families"].values():
        for name, digest in family["source_sha256_at_recert"].items():
            path = package / name
            if not path.exists():
                continue
            live = hashlib.sha256(path.read_bytes()).hexdigest()
            if live != digest:
                drifted.setdefault(name, set()).add(live)
    assert set(declared) == set(drifted), (
        f"declared={sorted(declared)} drifted={sorted(drifted)}; a family module "
        "changed without its drift being declared, or a declaration went stale")
    for name in drifted:
        assert declared[name]["certified"], name
        assert declared[name]["live_at_declaration"], name


# ---------------------------------------------------------------------------
# Composition: a mixed step is normal, and the mix itself is not certified
# ---------------------------------------------------------------------------


def test_a_partial_family_dispatch_is_disclosed_rather_than_refused(monkeypatch):
    """The measured case: special_kz's constitutive arms dispatch, its curl refuses.

    Each dispatched sub-step is bit-identity gated against the array sub-step it
    replaces, and the array sub-steps still run and are the oracle — so this is
    disclosure, not a refusal that would cost coverage no measurement asks for.
    """
    plan = plan_on_a_cupy_host(monkeypatch, step_plan(
        {"update_H": CountingPlan("h"), "update_E": CountingPlan("e")},
        {"update_H": "complex beta run", "update_E": "complex beta run"},
        {"step_B": ("complex beta PML: the expansion probe artifact lacks "
                    "'c8_mul_c8_imaginary_coefficient_left'",)}))
    assert plan is not None
    composition = plan.report()["composition"]
    assert composition["mixed"] is True
    assert composition["dispatched_slots"] == ["update_H", "update_E"]
    assert "step_B" in composition["array_slots"]
    assert composition["families_dispatched"] == ["special_kz"]
    assert "The MIX itself was not composed by any gate" in composition["read_this_as"]
    assert plan.report()["families"]["complex beta run"]["dispatched_slots"] == [
        "update_H", "update_E"]


def test_update_P_alone_records_the_shape_its_gate_never_composed(monkeypatch):
    """``ade_update_p_coverage`` carries no geometry clause, so it dispatches alone
    on a cylindrical grid the dispersive composition gate never ran.

    Reading the arithmetic says there is nothing for the geometry to change —
    ``stepping.update_P`` is a loop over polarizations with no coordinate branch —
    so the answer is a shape a reader can compare against the gate, not a refusal.
    """
    grid = cupy_like_grid()
    grid.cylindrical = True
    grid.m = 0
    grid.dimensions = 2
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan(
        {"update_P": [CountingPlan("p0")]}, {"update_P": "ADE update_P"}))
    plan = fastpath.plan_fast_path(object(), object(), grid)
    assert plan is not None and plan.slots == ("update_P",)
    record = plan.report()
    assert record["run_shape"]["cylindrical"] is True
    assert record["run_shape"]["m"] == 0
    # THE TWO FILL SLOTS ARE ARRAY SLOTS HERE, and that is the true answer rather
    # than a widened expectation: this run dispatches ``update_P`` alone, so the
    # driver's ``fill_symmetry_bc_*``/``fill_folded_far_ghosts_*`` passes really do
    # run on the array path. They joined the list when they joined ``DRIVER_SLOTS``
    # on 2026-09-02; before that the composition block could not report them at all.
    assert record["composition"]["array_slots"] == [
        "step_B", "fill_B", "update_H", "step_D", "fill_D", "update_E"]
    assert record["families"]["ADE update_P"]["gate"] == "dispersive_composition_gate"


# ---------------------------------------------------------------------------
# The artifact belongs to the driver that made it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gpu,reference,reason", UNPLANNED_DRIVERS)
def test_a_refused_driver_reports_its_own_refusal_not_a_later_drivers_dispatch(
        monkeypatch, gpu, reference, reason):
    """The measured inversion: refused, and reporting 'dispatched'/'fused'.

    ``fast_path_report`` fell through to the module-level last record whenever the
    driver held no plan, and that record is one process-wide slot every freeze
    overwrites. Held for both drivers that freeze without a plan: the NumPy reference
    and a GPU driver the planner refused.
    """
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "0")
    refused = numpy_driver()
    refused.gpu = gpu
    refused.step()
    assert refused.fast_path_report()["decision"] == "refused"
    assert refused.fast_path_report()["refused_because"] == reason

    dispatching = numpy_driver()
    install_plan(monkeypatch, dispatching, {"step_B": CountingPlan("b")},
                 {"step_B": "PML"})
    dispatching.step()
    assert dispatching.active_step_path == "fused"

    report = refused.fast_path_report()
    assert report["decision"] == "refused", (
        "a refused driver must not serve another driver's dispatch record")
    assert report["reference_driver"] is reference
    assert report["refused_because"] == reason
    assert report["step_path"] == refused.active_step_path == "array"
    refused.close()
    dispatching.close()


def test_a_driver_that_has_not_frozen_yet_reports_nothing(monkeypatch):
    """No configuration has frozen, so there is no verdict — and none is borrowed."""
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    stub_triton(monkeypatch)
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())  # Fills the module slot.
    driver = numpy_driver()
    assert driver.fast_path_report() is None
    assert driver.active_step_path == "array"
    driver.close()


def test_active_step_path_answers_about_the_frozen_plan_not_the_future(monkeypatch):
    """Its contract, stated: nothing has decided before the first step()."""
    driver = numpy_driver()
    install_plan(monkeypatch, driver, {"step_B": CountingPlan("b")}, {"step_B": "PML"})
    assert driver.active_step_path == "array", "no configuration has frozen yet"
    driver.step()
    assert driver.active_step_path == "fused"
    driver.close()


# ---------------------------------------------------------------------------
# The one-line announcement, and the warm switch
# ---------------------------------------------------------------------------


def test_the_stderr_line_says_whether_a_policy_was_actually_installed(monkeypatch, capsys):
    """It read 'policy keep' on a process whose stamp says policy='none_installed'.

    The stamp's own note calls that a FALLBACK, not a measurement of MEEP, and a
    line that prints only the resolution states the fallback as a fact.
    """
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    monkeypatch.setattr("meep_gpu.subnormal_policy.policy_stamp",
                        lambda: {"policy": "none_installed", "installed": False,
                                 "resolved": "keep", "requested": "match_meep"})
    fastpath.plan_fast_path(object(), object(), cupy_like_grid())
    line = [text for text in capsys.readouterr().err.splitlines()
            if text.startswith("meep_gpu: step path")][0]
    assert "policy keep NOT INSTALLED (requested match_meep)" in line


def test_the_warm_switch_records_that_it_widens_the_dispatch_set(monkeypatch):
    """It is not a timing switch: with it off, nothing unfills."""
    monkeypatch.setenv(fastpath.WARM_SWITCH, "0")
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                      {"step_B": "PML"}))
    assert plan.report()["warm_pass"]["changes_the_dispatch_set_when_disabled"] is True
    monkeypatch.delenv(fastpath.WARM_SWITCH)
    plan = plan_on_a_cupy_host(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                      {"step_B": "PML"}))
    assert plan.report()["warm_pass"]["changes_the_dispatch_set_when_disabled"] is False


# ---------------------------------------------------------------------------
# The SECOND kernel table: the candidate-table rung and the Metal record welds
# ---------------------------------------------------------------------------
#
# WHY THESE LIVE HERE RATHER THAN IN test_metal_dispatch. That file holds the Metal
# ladder's own rungs; this one holds the CONTRACT the two tables share — which
# backend reaches which table, and whether a record that says a table dispatches is
# welded to the bytes that dispatched. A reader asking "what does the dispatch seam
# promise" should find both tables' answer in one place, and a reader asking "what
# does the Metal ladder do on a fold" should find that beside the Metal composer.


def _metal_table_is_wired() -> bool:
    return hasattr(fastpath, "candidate_tables")


def _requires_the_metal_table() -> None:
    if not _metal_table_is_wired():
        requires_resource_skip(
            "fastpath_metal_dispatch_batch",
            "fastpath.candidate_tables has not landed; the rung-3 branch that "
            "routes a NumPy engine with an MPS device to the Metal table is a "
            "separate edit batch on this file")


def test_a_numpy_engine_with_an_mps_device_takes_the_metal_table(monkeypatch):
    """HARDWARE PICKS THE CANDIDATE SET; a preference selects WITHIN it.

    The rung is not "which table do we like" — it is "which tables could possibly
    run here", answered from the engine's array module and the device that is
    present. A CuPy engine gets the NVIDIA precedence; a NumPy engine with an MPS
    device gets ``("metal",)``; a NumPy engine with neither gets nothing and is
    refused with the compound reason the two tests above read.
    """
    _requires_the_metal_table()
    monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: True)
    grid = types.SimpleNamespace(xp=np)
    assert fastpath.candidate_tables(grid) == (fastpath.METAL_TABLE,)


def test_a_numpy_engine_with_no_device_has_no_candidate_table(monkeypatch):
    """The control: without the device the same engine yields an EMPTY set.

    A rung that answered ``("metal",)`` unconditionally would pass the test above
    and be wrong everywhere else, so both directions are driven.
    """
    _requires_the_metal_table()
    monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: False)
    assert fastpath.candidate_tables(types.SimpleNamespace(xp=np)) == ()


def test_the_precedence_constant_is_triton_first_on_cupy():
    """A CuPy engine's candidate set is an ORDER, and the order is the ruling.

    Triton is the release-gated incumbent and hand-CUDA fills its refusals in a
    later phase; the constant is what records which one a host that could run both
    takes, so flipping it after a timing round is a one-line edit rather than a
    search.
    """
    _requires_the_metal_table()
    assert fastpath.NVIDIA_TABLE_PRECEDENCE[0] == "triton"


def test_a_preference_outside_the_candidate_set_refuses_by_name(monkeypatch):
    """Asking for a table this hardware cannot run is a refusal, never a silent fallback.

    The preference SELECTS WITHIN the candidate set; it does not widen it. A run
    that asked for Metal on an NVIDIA box and silently got Triton would be a run
    whose artifact describes a table the user did not ask for.
    """
    _requires_the_metal_table()
    monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: False)
    monkeypatch.setenv(fastpath.KERNEL_TABLE_SWITCH, "metal")
    assert fastpath.plan_fast_path(None, None, types.SimpleNamespace(xp=np)) is None
    reason = fastpath.last_dispatch_report()["refused_because"]
    assert "metal" in reason, reason


def test_the_metal_arm_certification_rows_all_resolve(monkeypatch):
    """The Metal twin of ``test_every_recorded_certification_points_at_a_readable_record``.

    THE ANTI-FORGERY CHECK, and it does not wait for the dispatch batch because the
    table it checks already ships: ``metal_dispatch.ARM_CERTIFICATION`` names a
    ledger key per arm, and a key that resolves to nothing is a record forgery that
    is invisible in an artifact — the arm names a gate, the gate names nothing.
    """
    from meep_gpu import metal_dispatch

    assert metal_dispatch.unresolved_certification_rows() == ()
    assert metal_dispatch.release_rows_without_a_weld() == ()


def test_the_metal_record_is_welded_to_the_live_sources_when_it_exists():
    """The Metal ``driver_dispatch`` block, held to the same rule as the Triton one.

    THE RULE IS "IF IT EXISTS IT IS WELDED", not "it exists". The block is cut by
    ``parity/meep_gpu/recut_driver_dispatch_record.py --backend metal`` from a
    released campaign, and before that campaign has run there is nothing to weld —
    so its ABSENCE is the honest pre-campaign state and is asserted as such, while
    its presence brings the full comparison: every digest it binds must equal the
    live tree's, and every GPU architecture the Metal table admits must have a live
    route run in ``runs[<architecture>]`` naming the campaign the shipped constant
    spells for it (``metal_runs.route_campaign``).

    That second clause is the ordering trap in one assertion. The route-gate
    constant must be edited BEFORE the campaign, because editing it is itself a
    source edit; a record whose ``gate`` and whose module's constant disagree is a
    record cut after the fact.

    AND EVERY LIVE ROUTE RUN READS PASS, in the words the three weld contracts use
    (``weld_record_walk.route_run_problems``), as the NVIDIA records are held to in
    ``_assert_each_admitted_capability_names_its_route_campaign``.
    """
    import hashlib
    import json
    import pathlib

    from meep_gpu import metal_dispatch, metal_runs

    package = pathlib.Path(fastpath.__file__).parent
    ledger = json.loads((package / "metal_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))
    record = ledger.get("driver_dispatch")
    stamp = metal_dispatch.METAL_DRIVER_ROUTE_GATE
    if record is None:
        # THE PRE-CAMPAIGN BRANCH, AND IT NOW ASSERTS SOMETHING. It read
        # ``assert <expr> or True`` until 2026-09-11 -- a tautology, in a branch whose
        # docstring says the absence "is asserted as such". What is actually worth
        # refusing is not the absence but the PAIR: a campaign that RELEASED and a
        # ledger with no block is a record that is owed, which is the state a cut
        # forgotten at the end of a long round leaves behind and the one this test is
        # positioned to see.
        results = package.parent / "parity" / "meep_gpu" / "results"
        for summary in sorted(results.glob(f"{stamp}_applegpu_*/campaign.txt")):
            text = summary.read_text(encoding="utf-8")
            assert "fail=0" not in text, (
                f"{summary.parent.name} released every leg (campaign.txt says "
                f"fail=0) and metal_kernels/fingerprints.json carries no "
                f"driver_dispatch block: the record is owed. Cut it with "
                f"parity/meep_gpu/recut_driver_dispatch_record.py --backend metal "
                f"--run {summary.parent.name}")
        return
    assert record.get("table") == "metal", record.get("table")
    released = record["exclusions"]["released_fused_arms"]
    assert set(released["arms"]) == set(metal_dispatch.METAL_RELEASED_FUSED_ARMS)
    for name, digest in record["source_sha256"].items():
        path = (package.parent / name if name.startswith("parity/")
                else package.parent / name if name.startswith("meep_gpu/")
                else package / name)
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == live, (
            f"{name} changed since the Metal dispatch record was cut. Re-run the "
            f"campaign and re-cut with recut_driver_dispatch_record.py "
            f"--backend metal; do not type a digest in.")
    # ONE ROUTE RUN PER GPU ARCHITECTURE, as the NVIDIA records keep one per compute
    # capability (_assert_each_admitted_capability_names_its_route_campaign): the
    # run's subkeys have left the entry level, and every architecture the cited welds
    # certify has a live route run naming that architecture's campaign.
    assert metal_runs.shape_reasons(record, run_fields=metal_runs.DISPATCH_RUN_FIELDS) \
        == [], metal_runs.shape_reasons(record, run_fields=metal_runs.DISPATCH_RUN_FIELDS)
    entry_level = sorted({"gate", "artifact", "what_was_measured"} & set(released))
    assert not entry_level, (
        f"exclusions.released_fused_arms still carries {entry_level} beside the "
        f"per-architecture route runs; they describe one run and live in "
        f"{metal_runs.RUNS}[<architecture>].released_fused_arms")
    admitted = metal_runs.admission_report(ledger, metal_runs.cited_keys())["admitted"]
    assert admitted, "the Metal table admits no GPU architecture, so no route run can be checked"
    live = metal_runs.live_architectures(record)
    for architecture in admitted:
        assert architecture in live, (
            f"the Metal table admits {architecture} and its driver_dispatch record has "
            f"no live route run for it (live: {list(live)}); run the route campaign "
            f"{metal_runs.route_campaign(stamp, architecture)} on these bytes and recut")
        run = record[metal_runs.RUNS][architecture]
        expected = metal_runs.route_campaign(stamp, architecture)
        assert run["released_fused_arms"]["gate"] == expected, (
            architecture, run["released_fused_arms"]["gate"], expected)
        assert run["records"] == f"apps/api/parity/meep_gpu/results/{expected}", (
            architecture, run["records"])
        assert f"results/{expected}/" in run["released_fused_arms"]["artifact"], architecture
        assert run["residency"]["mode"], architecture
    problems = walk.route_run_problems(record, live)
    assert not problems, f"the metal driver_dispatch record: {problems}"


# ---------------------------------------------------------------------------
# THE SECOND NVIDIA TABLE: candidacy, precedence, arbitration, and the merge
# ---------------------------------------------------------------------------
#
# The ladder composes the hand-CUDA table after Triton on a CuPy engine and merges
# whole units of it into one plan. What is measured here is the LADDER — which table
# is a candidate, which composes first, who gets a slot both admit, and how each
# refusal is named — with both composers stubbed at the module attribute the ladder
# reads. The COMPOSER's own half (the offer, the labels, the release tables) is
# ``test_cuda_certified_fused_products``; the arithmetic is each family's device gate.


class LaunchableCudaPair:
    """A ``CudaFusedPairPlan``-shaped stub: declares a launch and counts it.

    ``launchable`` is the declaration the merge reads through ``absorbed_by``, and a
    stub without it would be adopted for the wrong reason — an absent declaration
    means "this composer does not declare launchability" and is answered from
    ``run``, which is the Triton and Metal composers' case, not this one's.
    """

    def __init__(self, label):
        self.label = label
        self.launchable = True
        self.launches = 0
        self.last_launch = None

    def run(self, *args, **kwargs):
        self.launches += 1
        self.last_launch = {"launched": True, "blocks": 512}
        return self.last_launch

    @property
    def launch_grid(self):
        blocks = (self.last_launch or {}).get("blocks")
        return (int(blocks),) if blocks is not None else None

    def warm(self):
        return None


def cuda_step_plan(pairs=(("step_D", "update_E", "fused electric pair"),),
                   reasons=None):
    """A ``CudaStepPlan`` holding the named fused pairs and nothing else."""
    from meep_gpu.cuda_kernels import arms as cuda_arms
    from meep_gpu.triton_kernels.launch import NoopPlan

    plans, selected = {}, {}
    for curl, update, label in pairs:
        pair = LaunchableCudaPair(label)
        plans[curl] = pair
        plans[update] = NoopPlan(update, pair)
        selected[curl] = selected[update] = label
    return cuda_arms.CudaStepPlan(plans, dict(reasons or {}), selected)


def install_cuda_composer(monkeypatch, plan):
    """Route the ladder's one ``cuda_kernels.arms.plan_step`` call at a prepared plan.

    Patched on the MODULE ATTRIBUTE rather than on a local name, because the ladder
    imports the module function-locally inside ``_decide`` — the import that
    ``test_triton_kernels`` pins the position of — so a patched attribute is what it
    will find.
    """
    from meep_gpu.cuda_kernels import arms as cuda_arms

    calls = []

    def fake_plan_step(fields, pml, grid=None, **kwargs):
        calls.append(kwargs)
        if isinstance(plan, Exception):
            raise plan
        return plan

    monkeypatch.setattr(cuda_arms, "plan_step", fake_plan_step)
    return calls


def plan_on_a_two_table_host(monkeypatch, triton_plan, cuda_plan, fields=None,
                             grid=None, pml=None, preference=None):
    stub_triton(monkeypatch)
    install_composer(monkeypatch, triton_plan)
    cuda_calls = install_cuda_composer(monkeypatch, cuda_plan)
    if preference is None:
        monkeypatch.delenv(fastpath.BACKEND_PREFERENCE_SWITCH, raising=False)
    else:
        monkeypatch.setenv(fastpath.BACKEND_PREFERENCE_SWITCH, preference)
    plan = fastpath.plan_fast_path(fields or object(), pml or object(),
                                   grid or cupy_like_grid())
    return plan, cuda_calls


def test_both_nvidia_tables_are_candidates_on_a_cupy_engine(monkeypatch):
    """Rung 3 answers with a LIST, and rung 4d decides which of them goes first."""
    assert fastpath.candidate_tables(cupy_like_grid()) == ("triton", "cuda")
    assert fastpath.BACKEND_PRECEDENCE == ("triton", "cuda")
    plan, _calls = plan_on_a_two_table_host(
        monkeypatch, step_plan({"step_B": CountingPlan("b")}, {"step_B": "PML"}),
        cuda_step_plan(pairs=()))
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = fastpath.last_dispatch_report()
    assert record["tables"]["triton"]["candidate"] is True
    assert record["tables"]["cuda"]["candidate"] is True
    assert record["arbitration"]["effective_order"] == ["triton", "cuda"]
    assert record["table"] == "triton"


def test_the_default_precedence_leaves_a_contested_seam_with_the_incumbent(monkeypatch):
    """ARBITRATION-DEFAULT-HELD, and the refusal is named rather than silent.

    Both tables admit the D seam. Triton is the release-gated incumbent, so it keeps
    both slots — and the record says WHICH slots the CUDA unit lost and to WHAT, so a
    reader of a default-precedence artifact can see the arbitration instead of
    inferring it from an absence.
    """
    plan, _calls = plan_on_a_two_table_host(
        monkeypatch,
        step_plan({"step_D": CountingPlan("d")}, {"step_D": "PML"}),
        cuda_step_plan())
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = fastpath.last_dispatch_report()
    assert plan.arms["step_D"] == "PML"
    assert plan.backends.get("step_D") == "triton"
    refusal = record["arbitration"]["refused"]["cuda"]["cuda:fused electric pair"]
    assert "step_D held by triton:PML" in refusal, refusal
    assert record["composition"]["tables_dispatched"] == ["triton"]


def test_the_preference_flips_the_order_and_the_cuda_unit_takes_the_seam(monkeypatch):
    """INCUMBENT-YIELDED: the same two composers, the switch the campaign drives.

    ON THE CONFIGURATION THE CUDA RELEASE ADMITS, not on the bare stub grid: the
    release is what decides whether the composer is OFFERED the label at all, so a
    shapeless grid would measure the envelope rather than the arbitration.
    """
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    plan, calls = plan_on_a_two_table_host(
        monkeypatch,
        step_plan({"step_D": CountingPlan("d")}, {"step_D": "PML"}),
        cuda_step_plan(), preference="cuda",
        grid=certified_envelope_grid(), fields=certified_envelope_fields(),
        pml=certified_envelope_pml())
    assert plan is not None, fastpath.last_dispatch_report()["refused_because"]
    record = fastpath.last_dispatch_report()
    assert record["arbitration"]["effective_order"] == ["cuda", "triton"]
    assert plan.arms["step_D"] == "cuda:fused electric pair"
    assert plan.arms["update_E"] == "cuda:fused electric pair"
    assert plan.backends["step_D"] == "cuda"
    assert record["composition"]["tables_dispatched"] == ["cuda"]
    # AND THE FAMILY CITES THE CUDA LEDGER, not the Triton one. A CUDA label looked
    # up in the Triton ledger resolves to nothing and would reach a user's artifact
    # as a family name beside a dead key.
    entry = record["families"]["cuda:fused electric pair"]
    assert entry["table"] == "cuda"
    assert entry["gate"].startswith("cuda_")
    # The offer reached the composer BARE: the namespace is this file's, not the
    # composer's, and handing it a prefixed label would offer a product it cannot
    # name.
    assert calls and all(not label.startswith("cuda:")
                         for label in calls[-1]["fuse_labels"])


def test_a_preference_naming_a_non_candidate_refuses_BY_NAME(monkeypatch):
    """Not "internal error": ``plan_fast_path``'s wrapper spells a raise that way,
    and a deliberate refusal must not be readable as a defect."""
    plan, _calls = plan_on_a_two_table_host(
        monkeypatch, step_plan({"step_B": CountingPlan("b")}, {"step_B": "PML"}),
        cuda_step_plan(pairs=()), preference="metal")
    assert plan is None
    record = fastpath.last_dispatch_report()
    reason = record["refused_because"]
    assert fastpath.BACKEND_PREFERENCE_SWITCH in reason
    assert "names no candidate table" in reason
    assert "internal error" not in reason
    assert record["arbitration"]["refused_because"] == reason


def test_an_unlaunchable_cuda_single_arm_is_never_adopted(monkeypatch):
    """A CUDA single with no established launch is never adopted, and the reason is
    recorded; the two certified real-PML seams are the only singles that are."""
    from meep_gpu.cuda_kernels import arms as cuda_arms

    single = types.SimpleNamespace(launchable=False)
    cuda_plan = cuda_arms.CudaStepPlan({"step_B": single}, {}, {"step_B": "PML"})
    plan, _calls = plan_on_a_two_table_host(
        monkeypatch, step_plan({"step_D": CountingPlan("d")}, {"step_D": "PML"}),
        cuda_plan, preference="cuda",
        grid=certified_envelope_grid(), fields=certified_envelope_fields(),
        pml=certified_envelope_pml())
    record = fastpath.last_dispatch_report()
    assert plan is not None, record["refused_because"]
    assert "step_B" not in plan.slots
    assert "step_B" in record["arbitration"]["refused"]["cuda"]


def test_the_null_drop_reads_the_bare_label_on_either_nvidia_table():
    """Both NVIDIA composers spell the null constitutive arm the same way.

    Measured on ``no_pml_2d``: each selects ``"no-PML null"`` at ``update_H``, and
    the CUDA one would arrive at rung 7 as ``"cuda:no-PML null"``. A namespaced null
    that survived the drop would launch nothing while the record called the step
    fused, which is the vacuity this rung exists to remove. The rung cannot be
    reached from the composer — the null arm is a SINGLE arm with no resolver, and
    the merge adopts fused products and the two certified launchable seams only —
    so the comparison itself is what is pinned, rather than a path that would have
    to be fabricated to exercise it.
    """
    from meep_gpu import fastpath_cuda

    assert fastpath_cuda.NULL_ARM_LABEL == fastpath.NULL_ARM_LABEL
    assert fastpath._bare("cuda:no-PML null") == fastpath.NULL_ARM_LABEL
    assert fastpath._bare("no-PML null") == fastpath.NULL_ARM_LABEL
    assert fastpath._table_of("cuda:no-PML null") == "cuda"
    assert fastpath._table_of("no-PML null") is None


def test_the_update_P_slot_holding_ONE_cuda_plan_dispatches(monkeypatch):
    """The three-slot shape puts a SINGLE object in ``update_P``, not a list.

    The Triton table puts a LIST there and the dispatch branch iterates it; a single
    object raised ``TypeError`` mid-step before the branch read the SHAPE instead —
    on the dispatch path, where this module's own docstring says exceptions
    PROPAGATE. Driven here through the real ``CudaFusedTriplePlan`` so the
    transitive ``absorbed_by`` identity is exercised at the same time: a single hop
    would put the weld's three slots in TWO units and ``_split_pairs`` would refuse
    the plan as half a pair.
    """
    from meep_gpu.cuda_kernels import arms as cuda_arms
    from meep_gpu.cuda_kernels import fused_pairs as cuda_fused
    from meep_gpu.triton_kernels.launch import NoopPlan

    launched = []
    triple = cuda_fused.CudaFusedTriplePlan(
        family="cuda_three_slot_dispersive_weld",
        label="three-slot dispersive weld", kernel_label="k",
        replaces=("step_D", "update_E", "update_P"),
        slots=("step_D", "update_E", "update_P"),
        context=types.SimpleNamespace(fields=object()),
        resolve_launch_args=lambda context: {},
        launch_leading=lambda fields, arguments: launched.append("leading"),
        launch_trailing=lambda fields, arguments: launched.append("trailing"))
    label = "three-slot dispersive weld"
    cuda_plan = cuda_arms.CudaStepPlan(
        {"step_D": triple.leading,
         "update_E": NoopPlan("update_E", triple.leading),
         "update_P": triple.trailing},
        {}, {"step_D": label, "update_E": label, "update_P": label})
    monkeypatch.delenv(fastpath.FUSE_ARMS_SWITCH, raising=False)
    fields = certified_envelope_fields()
    fields.drive_field = lambda *args, **kwargs: None
    plan, _calls = plan_on_a_two_table_host(
        monkeypatch, step_plan({"step_B": CountingPlan("b")}, {"step_B": "PML"}),
        cuda_plan, preference="cuda", grid=certified_envelope_grid(),
        fields=fields, pml=certified_envelope_pml())
    record = fastpath.last_dispatch_report()
    assert plan is not None, record["refused_because"]
    assert plan.arms["update_P"] == "cuda:three-slot dispersive weld"
    # The three slots are ONE unit: a split pair would have refused the plan.
    assert {"step_D", "update_E", "update_P"} <= set(plan.slots)
    assert plan.dispatch("step_D", fields) is True
    assert plan.dispatch("update_P", fields) is True
    assert launched == ["leading", "trailing"]


def test_the_composer_signature_matches_the_call_the_ladder_makes():
    """The ladder's CUDA call, read off the source, against the real signature.

    The composer is stubbed in every test above, so a keyword the shipped function
    does not take would never be exercised here — the same hole the Triton twin of
    this test was written for.
    """
    import inspect

    from meep_gpu.cuda_kernels import arms as cuda_arms

    source = pathlib.Path(fastpath.__file__).read_text(encoding="utf-8")
    call = re.search(r"_cuda_arms\.plan_step\((.*?)\)\)\)", source, re.S)
    assert call, "the ladder no longer calls _cuda_arms.plan_step"
    passed = set(re.findall(r"(\w+)=", call.group(1)))
    accepted = set(inspect.signature(cuda_arms.plan_step).parameters)
    assert passed <= accepted, sorted(passed - accepted)
    assert {"fuse", "fuse_labels", "sources", "licenses",
            "subnormal_policy"} <= passed


def test_the_record_carries_both_tables_on_a_refused_early_run(monkeypatch):
    """``tables`` and ``arbitration`` are on EVERY record, for the reason ``table``
    is: a reader comparing two refusals must be able to tell "refused before any
    table was consulted" from "both were asked and both declined"."""
    monkeypatch.setenv(fastpath.FUSED_KILL_SWITCH, "0")
    assert fastpath.plan_fast_path(None, None, cupy_like_grid()) is None
    record = fastpath.last_dispatch_report()
    assert "not_read" in record["tables"]
    assert "not_reached" in record["arbitration"]


def test_the_recorded_cuda_dispatch_shape_matches_the_wiring_it_describes():
    """The CUDA ``driver_dispatch`` weld against the tables that ship.

    Skips while the record does not exist — the gate is NAMED before it runs, so the
    interval between the edit and the campaign is expected and is not a failure.
    What is refused is a record that exists and disagrees.
    """
    import json
    import pathlib as _pathlib

    from meep_gpu import fastpath_cuda

    package = _pathlib.Path(fastpath.__file__).parent
    ledger = json.loads((package / "cuda_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))
    record = ledger.get("driver_dispatch")
    if record is None:
        requires_resource_skip(
            "cuda_driver_dispatch_record",
            f"{fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE} has not been run and "
            "recut_driver_dispatch_record.py --backend cuda has not written the "
            "block; the gate is NAMED before it runs, so this interval is the "
            "ordering rule rather than a gap")
    assert record.get("table") == "cuda", record.get("table")
    exclusions = record["exclusions"]["released_fused_arms"]
    _assert_each_admitted_capability_names_its_route_campaign(
        record, fastpath.CUDA_TABLE, fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE)
    assert set(exclusions["arms"]) == set(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS)
    for arm, cases in fastpath_cuda.CUDA_RELEASED_FUSED_ARMS.items():
        assert set(exclusions["per_arm_cases"][arm]) == set(cases), arm


def test_the_cited_cuda_route_gate_ran_the_bytes_the_record_binds():
    """Every file the CUDA dispatch record pins, rehashed against the checkout."""
    import hashlib
    import json
    import pathlib as _pathlib

    package = _pathlib.Path(fastpath.__file__).parent
    ledger = json.loads((package / "cuda_kernels" / "fingerprints.json")
                        .read_text(encoding="utf-8"))
    record = ledger.get("driver_dispatch")
    if record is None:
        requires_resource_skip(
            "cuda_driver_dispatch_record",
            "the CUDA driver_dispatch block has not been cut")
    for name, digest in record["source_sha256"].items():
        path = package.parent / name
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == live, (
            f"{name} changed since the CUDA dispatch record was cut. Re-run the "
            f"campaign and re-cut with recut_driver_dispatch_record.py "
            f"--backend cuda; do not type a digest in.")
