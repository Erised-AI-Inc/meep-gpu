"""Device-free tests for the fused-product bench on the METAL table.

Added 2026-09-20. ``bench_fused_products.py`` named a ``metal`` drive table from its
first day and every instrument in it was written for the two NVIDIA tables: the launch
witness read ``delta["triton"]`` and ``delta["cuda"]``, the window-edge synchronise was
keyed on CuPy and so was a no-op on a host lift, "compile" meant a kernel NAME the
launch counters had not seen, and the report pooled every row into one headline. These
tests pin the Metal spelling of each, with the device stubbed the way the sibling
provenance tests stub it: a driver whose plans are plain objects holding a function
table, read by the route gate's OWN ``MetalLaunchCounter`` and proved by the route
gate's OWN ``substitution_proof`` -- the bench imports both and re-implements neither.

The five shapes a decision engine's tests take here (``the development notes``): KNOWN VALUE (two
fused pairs: 2 launches a step against 4, EXACT), DEGENERATE (a fused leg that fused
nothing, a leg whose owner the witness cannot see, a host with no device loaded),
SCALING (launches scale with the window, the rate does not), SHAPE/SERIALISATION (a
whole row through ``json.dumps``) and ONE REALISTIC SYSTEM (``run_case`` end to end over
three stub legs, every floor evaluated).

No test here touches a device. Wherever the code under test would reach for ``torch``
a fake module stands in ``sys.modules``; the route gate and ``meep_gpu.metal_kernels``
are the real modules, imported and never launched.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
from types import SimpleNamespace

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bench_fused_products as bench  # noqa: E402
import build_fused_timing_report as report  # noqa: E402


# ---------------------------------------------------------------------------
# The stubbed device
# ---------------------------------------------------------------------------

class _mps_MetalKernel:  # noqa: N801 - the leaf type NAME the gate's witness requires
    """Stands in for a ``torch.mps.compile_shader`` function: callable, does nothing."""

    def __call__(self, *_args, **_kwargs):
        return None


class _KernelPlan:
    """A plan owner in ``plans.KernelPlan``'s shape: a function table and ``launches``."""

    __slots__ = ("_functions", "launches", "per_run")

    def __init__(self, per_run=1, modes=("main",)):
        self._functions = {mode: _mps_MetalKernel() for mode in modes}
        self.launches = 0
        self.per_run = per_run

    def run(self, mode="main"):
        for _ in range(self.per_run):
            self.launches += 1
            self._functions[mode]()


class _Absorbed:
    """The absorbed slot of a fused pair: names the plan that did the work."""

    def __init__(self, pair):
        self.absorbed_by = pair


class _FastPath:
    """What the driver holds in ``_fast_path``: the frozen step plan and its residency.

    A NAMED CLASS, as the driver's is, so a test reading the objects a timed window
    holds (:func:`test_a_window_holds_the_plan_it_compares_against`) can tell the plan
    from the namespaces around it.
    """

    def __init__(self, plans, residency):
        self.step_plan = SimpleNamespace(plans=plans)
        self.residency = residency


class _Unwitnessable:
    """An owner that books launches and holds no function table at all."""

    def __init__(self):
        self.launches = 0

    def run(self, mode="main"):
        self.launches += 1


RESIDENCY = {"invariant": "host-authoritative between launches",
             "mode": "per-launch sync_in/sync_out installed at plan time",
             "mirrors": 45, "mirror_names": [f"m{i}" for i in range(45)],
             "synced_passes": ["fill_D"]}

KILL_SWITCH = "the kill switch MEEP_GPU_FUSED=0 refuses every plan"


class _Driver:
    """A host-backed driver whose step is: run every slot's plan, move the field."""

    def __init__(self, slots, fused_slots=(), vetoed=False):
        # ``slots``: ``{slot: (arm, plan)}`` in step order; empty for the array leg.
        self.xp = numpy
        # EVERY LEG ON THIS TABLE IS AN APPLE GPU DRIVER (``prefer_gpu=True``), the
        # array oracle included: it is the same driver under ``MEEP_GPU_FUSED=0``.
        self.gpu = "metal"
        self.fields = SimpleNamespace(Ez=numpy.zeros(16, dtype=numpy.float32))
        self.pml = None
        self.step_count = 0
        self._fast_path_stale = False
        self._slots = dict(slots)
        self._fused_slots = tuple(fused_slots)
        self._vetoed = vetoed
        self._dispatches = {slot: 0 for slot in slots}
        self._dft_monitors, self._flux_monitors = ["a monitor"], []
        self.inside_run = None          # a hook a test arms to act inside a window
        self.mode = "main"
        self.residency = SimpleNamespace(syncs_in=0, syncs_out=0,
                                         names=tuple(RESIDENCY["mirror_names"]),
                                         host=lambda _name: self.fields.Ez)
        self._fast_path = None
        if slots:
            self._freeze()

    def _freeze(self):
        self._fast_path = _FastPath({s: p for s, (_a, p) in self._slots.items()},
                                    self.residency)

    @property
    def active_step_path(self):
        return "fused" if self._slots else "array"

    def run(self, num_steps):
        for _ in range(int(num_steps)):
            for slot, (_arm, plan) in self._slots.items():
                self._dispatches[slot] += 1
                if hasattr(plan, "run"):
                    plan.run(self.mode)
                    self.residency.syncs_in += 1
                    self.residency.syncs_out += 1
            self.fields.Ez += numpy.float32(1.0)
            self.step_count += 1
        if self.inside_run is not None:
            self.inside_run(self)

    def fast_path_report(self):
        if not self._slots:
            return {"decision": "array", "slots": {}, "refused_because": KILL_SWITCH,
                    "composition": None, "fusion": {}}
        counters = {slot: {"arm": arm, "backend": "metal",
                           "dispatches": self._dispatches[slot],
                           "plan_launches": getattr(plan, "launches", None)}
                    for slot, (arm, plan) in self._slots.items()}
        return {"table": "metal", "decision": "dispatch", "refused_because": None,
                "slots": {slot: {"state": "dispatched", "arm": arm}
                          for slot, (arm, _plan) in self._slots.items()},
                "fusion": {"driven": {slot: self._slots[slot][0]
                                      for slot in self._fused_slots},
                           "vetoed": self._vetoed, "opted_in": []},
                "launch_counters": counters,
                "composition": {"table": "metal", "tables_dispatched": ["metal"]},
                "residency": dict(RESIDENCY)}

    def close(self):
        pass


B_PAIR, D_PAIR = "fused magnetic B/H pair", "fused electric D/E pair"


def _fused_two_pairs():
    b, d = _KernelPlan(), _KernelPlan()
    return _Driver({"step_B": (B_PAIR, b), "update_H": (B_PAIR, _Absorbed(b)),
                    "step_D": (D_PAIR, d), "update_E": (D_PAIR, _Absorbed(d))},
                   fused_slots=("step_B", "update_H", "step_D", "update_E"))


def _singles(update_e_per_run=1):
    return _Driver({"step_B": ("PML", _KernelPlan()),
                    "update_H": ("ordinary", _KernelPlan()),
                    "step_D": ("PML", _KernelPlan()),
                    "update_E": ("ordinary", _KernelPlan(per_run=update_e_per_run))},
                   vetoed=True)


def _leg(label, driver):
    return {"label": label, "driver": driver, "env": {}, "grid_shape": [16, 1, 1],
            "until": 1.0}


TWO_PAIRS = {"arms": [B_PAIR, D_PAIR], "pairs": 2, "seam": ["B", "D"],
             "expect": "dispatch", "reached_by": "release", "why": "stub"}


@pytest.fixture
def fake_torch(monkeypatch):
    """A ``torch`` that records what the bench asks of it, in ``sys.modules``."""
    calls = {"synchronize": 0, "compile_shader": 0}

    def synchronize():
        calls["synchronize"] += 1

    def compile_shader(_source):
        calls["compile_shader"] += 1
        return "a library"

    fake = types.ModuleType("torch")
    fake.__version__ = "0.0-stub"
    fake.mps = SimpleNamespace(synchronize=synchronize, compile_shader=compile_shader)
    fake.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    fake.calls = calls
    return fake


@pytest.fixture
def metal():
    return bench.table_module("metal")


@pytest.fixture
def witness(metal, fake_torch, monkeypatch):
    built = bench.MetalWitness(metal)
    monkeypatch.setattr(bench, "WITNESSES", built.install())
    yield built
    built.uninstall()


def _primed(witness, fused=None, unfused=None, steps=6):
    legs = {"fused": _leg("fused", fused or _fused_two_pairs()),
            "unfused": _leg("unfused", unfused or _singles()),
            "array": _leg("array", _Driver({}))}
    for label in ("fused", "unfused", "array"):
        bench.prime("stub", legs[label], witness, steps)
    return legs


# ---------------------------------------------------------------------------
# Synchronisation at the window edges
# ---------------------------------------------------------------------------

def test_a_host_lift_with_the_device_loaded_synchronises_the_device(fake_torch):
    driver = SimpleNamespace(xp=numpy)
    bench.sync_for(driver)()
    assert fake_torch.calls["synchronize"] == 1
    assert bench.sync_name(driver) == "torch.mps.synchronize"


def test_a_host_lift_with_no_device_loaded_is_a_named_no_op(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    driver = SimpleNamespace(xp=numpy)
    bench.sync_for(driver)()            # must not import torch to find out
    assert "torch" not in sys.modules
    assert bench.sync_name(driver).startswith("none")


def test_a_loaded_torch_without_the_device_is_a_no_op(fake_torch):
    fake_torch.backends.mps.is_available = lambda: False
    driver = SimpleNamespace(xp=numpy)
    bench.sync_for(driver)()
    assert fake_torch.calls["synchronize"] == 0
    assert bench.sync_name(driver).startswith("none")


def test_a_timed_window_ends_on_the_device_sync(fake_torch):
    """The order that makes a window a measurement of WORK: sync, run, sync, stop."""
    events = []
    fake_torch.mps.synchronize = lambda: events.append("sync")
    driver = SimpleNamespace(xp=numpy)
    bench.timed_window(lambda: events.append("run"), 1, bench.sync_for(driver))
    assert events == ["sync", "run", "sync"]


# ---------------------------------------------------------------------------
# The compile witness
# ---------------------------------------------------------------------------

def test_the_shader_compile_counter_counts_the_platform_entry_point_and_restores_it(
        fake_torch):
    original = fake_torch.mps.compile_shader
    counter = bench.ShaderCompileCounter()
    assert counter.install() is True
    assert fake_torch.mps.compile_shader is not original
    assert fake_torch.mps.compile_shader("kernel void f() {}") == "a library"
    assert counter.calls == 1 and fake_torch.calls["compile_shader"] == 1
    counter.uninstall()
    assert fake_torch.mps.compile_shader is original
    counter.uninstall()                 # idempotent
    assert fake_torch.mps.compile_shader is original


def test_the_shader_compile_counter_says_so_when_there_is_nothing_to_wrap(monkeypatch):
    broken = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", broken)
    counter = bench.ShaderCompileCounter()
    assert counter.install() is False and counter.calls == 0
    assert counter.attached is False
    counter.uninstall()


def test_the_compile_witnesses_carry_a_positive_control(witness, monkeypatch):
    """THE FLOOR IS ZERO, so the instrument has to be seen moving somewhere else.

    ``no_compiles_inside_timed_region`` passes on a row whose compile witnesses never
    attached exactly as it passes on a row that compiled nothing, and the two are
    indistinguishable in the artifact. So every row carries the cumulative counts at
    row time and the freeze pass's own compiles: the plan build is where this table
    compiles, so a first case whose warm pass compiled nothing is a witness that is
    not watching, not a plan that needed no kernels.
    """
    from meep_gpu.metal_kernels import device
    monkeypatch.setattr(device, "_LIBRARY_CACHE", dict(device._LIBRARY_CACHE))
    fired = []

    def compile_while_warming(_driver):
        # ONCE AT THE FREEZE AND ONCE LATER, because the warm pass is two chunks and
        # the control covers both: a compile on a later warm step is a compile too.
        fired.append(len(fired))
        sys.modules["torch"].mps.compile_shader("kernel void warm() {}")
        device._LIBRARY_CACHE[f"a source compiled while warming {fired[-1]}"] = object()

    leg = _leg("fused", _fused_two_pairs())
    leg["driver"].inside_run = compile_while_warming
    bench.prime("stub", leg, witness, 6)
    assert fired == [0, 1]              # the freeze chunk, then the steady-state one
    assert leg["warm_compiles"] == {"compile_shader_calls": 2,
                                    "library_cache_growth": 2}
    stamp = witness.row_stamp()["compile_witness"]
    assert stamp["attached"] is True
    assert stamp["compile_shader_calls"] == 2
    assert stamp["library_cache_entries"] == len(device._LIBRARY_CACHE)
    assert stamp["first_warm_pass"] == {"case": "stub", "leg": "fused",
                                        "compile_shader_calls": 2,
                                        "library_cache_growth": 2}


def test_the_positive_control_says_when_the_compile_counter_never_attached(
        metal, monkeypatch):
    """The reading that makes the floor vacuous, named in the row rather than inferred."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    built = bench.MetalWitness(metal)
    assert built.install() == []
    stamp = built.row_stamp()["compile_witness"]
    assert stamp["attached"] is False and stamp["compile_shader_calls"] == 0
    assert stamp["first_warm_pass"] is None
    built.uninstall()


# ---------------------------------------------------------------------------
# The leg environments
# ---------------------------------------------------------------------------

def test_the_route_a_drive_row_does_not_name_is_the_tables_own_default(metal):
    """``reached_by`` unstated means what the TABLE means by it, not what one table does.

    The two gate modules disagree by design: the NVIDIA ``_fuse_env`` defaults to the
    opt-in, the Metal one to ``release``, because the Metal envelope admits its whole
    released set and the opt-in is the exception there. A default spelled in the bench
    ran every unstated Metal row through the opt-in route -- naming the arms on
    ``MEEP_GPU_FUSE_ARMS`` -- and recorded it as the shipped release path.
    """
    import gate_dispatch_fused_route as nvidia
    assert bench.default_reached_by(metal) == "release"
    assert bench.default_reached_by(nvidia) == "opt-in"
    spec = {k: v for k, v in TWO_PAIRS.items() if k != "reached_by"}
    env = bench.leg_env(metal, spec, "fused")
    assert env == metal._fuse_env(spec["arms"])
    assert env["MEEP_GPU_FUSE_ARMS"] is None        # the release route, not the opt-in
    named = bench.leg_env(metal, dict(spec, reached_by="opt-in"), "fused")
    assert named["MEEP_GPU_FUSE_ARMS"] == ",".join(spec["arms"])


# ---------------------------------------------------------------------------
# The substitution, proved with the route gate's own proof
# ---------------------------------------------------------------------------

def test_known_value_two_fused_pairs_save_two_launches_a_step(witness):
    legs = _primed(witness)
    proof = witness.substitution(TWO_PAIRS, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["proved"] is True, proof["checks"]
    assert proof["launches_per_step"] == {"fused": 2.0, "unfused": 4.0, "array": 0.0}
    assert proof["compiled_calls_per_step"] == {"fused": 2.0, "unfused": 4.0,
                                                "array": 0.0}
    assert proof["gate_proof"]["verdict"] == "EXACT"
    assert proof["gate_proof"]["launch_drop_per_step"] == 2.0
    assert proof["gate_proof"]["levels_agree"] is True
    assert proof["tables_by_leg"] == {"fused": ["metal"], "unfused": ["metal"],
                                      "array": []}
    assert proof["fused_and_singles_used_the_same_table"] is True
    assert proof["launch_witness_available"] is True
    assert proof["instruments"] == ["slot_arm_map", "metal_launch_counter",
                                    "gate_substitution_proof"]
    assert proof["welded_arms"] == {B_PAIR: ["step_B", "update_H"],
                                    D_PAIR: ["step_D", "update_E"]}


def test_a_fused_leg_that_fused_nothing_is_not_proved(witness):
    legs = _primed(witness, fused=_singles())
    proof = witness.substitution(TWO_PAIRS, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["proved"] is False
    assert proof["checks"]["fused_names_an_arm_the_singles_do_not"] is False
    assert proof["checks"]["gate_substitution_proof_is_exact"] is False
    assert proof["checks"]["fused_launches_per_step_below_unfused"] is False


def test_the_baseline_must_show_the_veto_in_its_own_record(witness):
    """A control whose record does not say ``vetoed`` is not a control.

    The bench spells the baseline's environment and the leg's own dispatch record says
    whether the veto reached it. Divorce the two -- the arms were vetoed, the record
    says they were not -- and the gate's proof names it ``BASELINE-FUSED`` rather than
    reading a launch drop off a baseline that may itself have fused.
    """
    unfused = _singles()
    unfused._vetoed = False
    legs = _primed(witness, unfused=unfused)
    proof = witness.substitution(TWO_PAIRS, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["gate_proof"]["verdict"] == "BASELINE-FUSED"
    assert proof["checks"]["gate_substitution_proof_is_exact"] is False
    assert proof["proved"] is False


def test_a_baseline_that_fell_to_the_array_path_is_not_the_same_table(witness):
    """The veto leg dispatched nothing at all: no singles, hence no comparison.

    This table's own shape of ``reference_cuda_no_unfused_baseline``. The ratio would
    be a fused Metal plan against the HOST array path under a fusion heading, which is
    the `vs array` column and not a fusion number; the same-table clause refuses it by
    name before any ratio is computed.
    """
    legs = _primed(witness, unfused=_Driver({}))
    proof = witness.substitution(TWO_PAIRS, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["tables_by_leg"]["unfused"] == []
    assert proof["fused_and_singles_used_the_same_table"] is False
    assert proof["proved"] is False


COLLAPSE = {"arms": ["fused complex D/E pair"], "pairs": 1, "seam": ["D"],
            "expect": "dispatch", "reached_by": "release", "why": "stub",
            "collapsed_single_launches_per_step": {"update_E": 3}}


def _collapse_legs(witness):
    pair = _KernelPlan()
    arm = COLLAPSE["arms"][0]
    fused = _Driver({"step_B": ("curl", _KernelPlan()),
                     "update_H": ("stored H", _KernelPlan()),
                     "step_D": (arm, pair), "update_E": (arm, _Absorbed(pair))},
                    fused_slots=("step_D", "update_E"))
    unfused = _Driver({"step_B": ("curl", _KernelPlan()),
                       "update_H": ("stored H", _KernelPlan()),
                       "step_D": ("curl", _KernelPlan()),
                       "update_E": ("stored E", _KernelPlan(per_run=3))}, vetoed=True)
    return _primed(witness, fused=fused, unfused=unfused)


def test_the_declared_collapse_is_exact_at_a_drop_of_three(witness):
    """``complex_no_pml_3d``'s shape: the pair folds a 3-a-step single into its launch."""
    legs = _collapse_legs(witness)
    proof = witness.substitution(COLLAPSE, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["proved"] is True, proof["gate_proof"]
    assert proof["launches_per_step"] == {"fused": 3.0, "unfused": 6.0, "array": 0.0}
    assert proof["gate_proof"]["expected_drop_per_step"] == 3
    assert proof["gate_proof"]["collapsed_single_launches_per_step"] == {"update_E": 3}


def test_the_same_collapse_undeclared_is_refused_by_the_gates_own_verdict(witness):
    legs = _collapse_legs(witness)
    undeclared = {k: v for k, v in COLLAPSE.items()
                  if k != "collapsed_single_launches_per_step"}
    proof = witness.substitution(undeclared, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["proved"] is False
    assert proof["gate_proof"]["verdict"] == "DROPPED-BUT-NOT-EXACT"
    # the launches DID drop, so only the gate's exactness clause refuses it
    assert proof["checks"]["fused_launches_per_step_below_unfused"] is True
    assert proof["checks"]["gate_substitution_proof_is_exact"] is False


def test_an_owner_the_witness_cannot_see_fails_the_proof_and_is_named(witness):
    blind = _Unwitnessable()
    d = _KernelPlan()
    fused = _Driver({"step_B": (B_PAIR, blind), "update_H": (B_PAIR, _Absorbed(blind)),
                     "step_D": (D_PAIR, d), "update_E": (D_PAIR, _Absorbed(d))},
                    fused_slots=("step_B", "update_H", "step_D", "update_E"))
    legs = _primed(witness, fused=fused)
    proof = witness.substitution(TWO_PAIRS, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["unwitnessed_owners"]["fused"] == ["_Unwitnessable"]
    assert proof["launch_witness_available"] is False
    assert proof["checks"]["fused_leg_is_real"] is False
    assert proof["proved"] is False


def test_an_array_leg_that_launched_is_not_a_reference(witness):
    legs = _primed(witness)
    legs["array"] = _leg("array", _singles())
    bench.prime("stub", legs["array"], witness, 6)
    proof = witness.substitution(TWO_PAIRS, legs["fused"], legs["unfused"],
                                 legs["array"])
    assert proof["checks"]["array_leg_launched_nothing"] is False
    assert proof["proved"] is False


def test_prime_refuses_a_warm_pass_too_short_to_have_a_steady_state(witness):
    with pytest.raises(ValueError, match="steady"):
        bench.prime("stub", _leg("fused", _fused_two_pairs()), witness, 1)


def test_prime_records_the_residency_the_plan_installed(witness):
    legs = _primed(witness)
    facts = legs["fused"]["residency"]
    assert facts["mirrors"] == 45
    assert facts["invariant"] == "host-authoritative between launches"
    assert facts["host_authoritative_between_launches"] is True
    assert facts["mode"].startswith("per-launch")
    assert facts["mirrored_bytes"] == 45 * 16 * 4
    assert legs["array"]["residency"] is None


# ---------------------------------------------------------------------------
# Timed windows
# ---------------------------------------------------------------------------

def test_a_window_counts_launches_and_compiled_calls_and_sees_no_compile(witness):
    legs = _primed(witness)
    frozen = bench.freeze(legs["fused"]["driver"])
    entry = bench.window("stub", legs["fused"], frozen, witness, 5, 0)
    assert entry["launches"] == 10 and entry["launches_per_step"] == 2.0
    assert entry["compiled_calls"] == 10 and entry["compiled_calls_per_step"] == 2.0
    assert entry["residency_syncs"] == {"in": 10, "out": 10}
    assert entry["new_kernels_inside_timed_region"] == 0
    assert entry["compile_shader_calls"] == 0 and entry["library_cache_growth"] == 0
    assert entry["plan_refroze_inside_window"] is False
    assert witness.compile_events(entry) == 0
    # the window started from the frozen state and ended five steps after it
    assert legs["fused"]["driver"].step_count == frozen["step_count"] + 5


def test_launches_scale_with_the_window_and_the_rate_does_not(witness):
    legs = _primed(witness)
    frozen = bench.freeze(legs["unfused"]["driver"])
    short = bench.window("stub", legs["unfused"], frozen, witness, 3, 0)
    long = bench.window("stub", legs["unfused"], frozen, witness, 12, 1)
    assert (short["launches"], long["launches"]) == (12, 48)
    assert short["launches_per_step"] == long["launches_per_step"] == 4.0
    assert short["compiled_calls_per_step"] == long["compiled_calls_per_step"] == 4.0


def _first_launch(driver):
    driver.mode = "late"                # a function warmup never reached


def _shader_compile(_driver):
    sys.modules["torch"].mps.compile_shader("kernel void late() {}")


def _library_growth(_driver):
    from meep_gpu.metal_kernels import device
    device._LIBRARY_CACHE["a source compiled inside the window"] = object()


def _refreeze(driver):
    driver._freeze()


@pytest.mark.parametrize("event, field", [
    (_first_launch, "new_kernels_inside_timed_region"),
    (_shader_compile, "compile_shader_calls"),
    (_library_growth, "library_cache_growth"),
    (_refreeze, "plan_refroze_inside_window"),
])
def test_each_way_a_compile_can_land_inside_a_window_is_an_event(
        witness, monkeypatch, event, field):
    from meep_gpu.metal_kernels import device
    monkeypatch.setattr(device, "_LIBRARY_CACHE", dict(device._LIBRARY_CACHE))
    b, d = _KernelPlan(modes=("main", "late")), _KernelPlan(modes=("main", "late"))
    fused = _Driver({"step_B": (B_PAIR, b), "update_H": (B_PAIR, _Absorbed(b)),
                     "step_D": (D_PAIR, d), "update_E": (D_PAIR, _Absorbed(d))},
                    fused_slots=("step_B", "update_H", "step_D", "update_E"))
    legs = _primed(witness, fused=fused)
    driver = legs["fused"]["driver"]
    frozen = bench.freeze(driver)
    clean = bench.window("stub", legs["fused"], frozen, witness, 2, 0)
    assert witness.compile_events(clean) == 0

    if event is _first_launch:
        event(driver)                   # the window itself makes the first launch
    else:
        driver.inside_run = event       # fires at the end of ``run``, inside the timer
    entry = bench.window("stub", legs["fused"], frozen, witness, 2, 1)
    assert entry[field], entry
    assert witness.compile_events(entry) >= 1
    if event is _first_launch:
        assert len(entry["new_kernel_names"]) == 2      # one per pair owner
        assert all("late" in name for name in entry["new_kernel_names"])


def _window_frame_holds(plan):
    """Is ``plan`` a local of the :func:`bench_fused_products.window` frame timing us?"""
    frame = inspect.currentframe()
    while frame is not None:
        if frame.f_code is bench.window.__code__:
            return any(value is plan for value in frame.f_locals.values())
        frame = frame.f_back
    raise AssertionError("no window frame on the stack")


def test_a_window_holds_the_plan_it_compares_against(witness):
    """The re-freeze witness holds the plan OBJECT for as long as it is timing.

    AN ADDRESS IS NOT AN IDENTITY once the object at it has been freed. The driver
    drops the plan (``driver.py:3112``) before the next step rebuilds it
    (``driver.py:3269``), so the rebuild may be handed the dropped plan's address:
    measured 2026-09-20 on this stub, 2 times in 50. Those windows re-ran the whole
    ladder, rebuilt every kernel and timed all of it, and an identity read from an
    address alone calls them clean. Holding the object makes the reuse impossible,
    which is why this asserts on the reference and not on a sampled address.
    """
    legs = _primed(witness)
    driver = legs["fused"]["driver"]
    frozen = bench.freeze(driver)
    seen = {}

    def drop_and_replan(drv):
        seen["window_holds_the_plan"] = _window_frame_holds(drv._fast_path)
        drv._fast_path = None
        drv._fast_path_stale = True
        drv._freeze()
        drv._fast_path_stale = False

    driver.inside_run = drop_and_replan
    entry = bench.window("stub", legs["fused"], frozen, witness, 2, 0)
    assert seen["window_holds_the_plan"] is True
    assert entry["plan_refroze_inside_window"] is True
    # THE ONLY WITNESS THAT SEES THIS ONE: the slots and their owners are unchanged, so
    # the launch counts, the kernel names and the compile memo all read exactly as they
    # do in a clean window.
    assert entry["launches"] == 4 and entry["new_kernels_inside_timed_region"] == 0
    assert entry["compile_shader_calls"] == 0 and entry["library_cache_growth"] == 0
    assert witness.compile_events(entry) == 1


def test_a_refreeze_between_the_proof_and_the_window_fails_the_rate_floor(witness):
    """New owners after the proof: the witness never attached to them, so both counts
    read zero -- which the compile floor reads as a clean window and only the rate
    floor refuses."""
    legs = _primed(witness)
    driver = legs["fused"]["driver"]
    b, d = _KernelPlan(), _KernelPlan()
    driver._slots = {"step_B": (B_PAIR, b), "update_H": (B_PAIR, _Absorbed(b)),
                     "step_D": (D_PAIR, d), "update_E": (D_PAIR, _Absorbed(d))}
    driver._freeze()
    entry = bench.window("stub", legs["fused"], bench.freeze(driver), witness, 5, 0)
    assert entry["launches"] == 0 and entry["compiled_calls"] == 0
    assert witness.compile_events(entry) == 0
    assert bench.windows_launched_at_the_proved_rate(legs, [entry]) is False


def test_the_proved_rate_floor():
    legs = {"fused": {"warm_launches_per_step": 2.0},
            "array": {"warm_launches_per_step": 0.0}}

    def w(leg, launches, calls, steps=5):
        return {"leg": leg, "steps": steps, "launches": launches,
                "launches_per_step": launches / steps, "compiled_calls": calls}

    good = [w("fused", 10, 10), w("array", 0, 0), w("fused", 10, 10)]
    assert bench.windows_launched_at_the_proved_rate(legs, good) is True
    # a window that launched at another rate timed another composition
    assert bench.windows_launched_at_the_proved_rate(
        legs, good + [w("fused", 20, 20)]) is False
    # the plan's launches and the compiled calls are two counts of ONE event
    assert bench.windows_launched_at_the_proved_rate(
        legs, good + [w("fused", 10, 9)]) is False
    # a re-frozen plan is unattached: both counts read zero on a leg proved at two
    assert bench.windows_launched_at_the_proved_rate(
        legs, [w("fused", 0, 0)]) is False
    # NO DATA IS NOT A PASS
    assert bench.windows_launched_at_the_proved_rate(legs, []) is False


# ---------------------------------------------------------------------------
# The host
# ---------------------------------------------------------------------------

BATTERY = ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=25231459)\t77%; "
           "discharging; 5:19 remaining present: true\n")
MAINS = ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; "
         "0:00 remaining present: true\n")


def test_power_source_is_parsed_from_the_platform_report():
    assert bench.parse_power(BATTERY) == {"power_source": "Battery Power",
                                          "on_ac_power": False, "battery_percent": 77}
    assert bench.parse_power(MAINS) == {"power_source": "AC Power",
                                        "on_ac_power": True, "battery_percent": 100}
    assert bench.parse_power("") == {"power_source": None, "on_ac_power": None,
                                     "battery_percent": None}


def test_power_mode_is_parsed_under_either_spelling():
    assert bench.parse_powermode(" standby 1\n powermode            1\n") == 1
    assert bench.parse_powermode(" lowpowermode         0\n") == 0
    assert bench.parse_powermode(" hibernatemode 3\n") is None


def test_thermal_limits_are_parsed_and_an_unrecorded_level_is_none():
    text = ("Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n")
    assert bench.parse_thermal(text)["cpu_speed_limit"] is None
    limited = "CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 10\nCPU_Speed_Limit \t= 62\n"
    assert bench.parse_thermal(limited)["cpu_speed_limit"] == 62


def test_foreign_device_processes_names_the_neighbours_and_not_itself():
    ps = ("  100     1 /sbin/launchd\n"
          "  150   100 zsh results/round_drivers/metal_timing.sh stamp\n"
          "  160   150 /bin/zsh -c python -u parity/meep_gpu/bench_fused_products.py --out z\n"
          "  200   100 python -u parity/meep_gpu/gate_dispatch_metal_route.py --out x\n"
          "  201   100 zsh parity/meep_gpu/recut_metal_gates.sh root stamp\n"
          "  202   201 python -u parity/meep_gpu/gate_metal_bfast.py --out y\n"
          "  300   160 python -u parity/meep_gpu/bench_fused_products.py --drive-table metal\n"
          "  310   100 python -u parity/meep_gpu/bench_fused_products.py --drive-table metal\n"
          "  301   300 caffeinate -i python bench_fused_products.py\n"
          "  400   100 vim notes_about_gate_metal_.md\n")
    found = bench.foreign_device_processes(ps, own_pid=300)
    # 160 launched this process and carries its command line; 310 is ANOTHER bench
    assert [line.split()[0] for line in found] == ["200", "201", "202", "310"]


def test_host_state_off_this_platform_reads_nothing_and_says_nothing(monkeypatch):
    def refuse(_command):
        raise AssertionError("no platform command may run off darwin")
    monkeypatch.setattr(bench.sys, "platform", "linux")
    monkeypatch.setattr(bench, "_read_command", refuse)
    bench._static_host_facts.cache_clear()
    try:
        state = bench.host_state()
    finally:
        bench._static_host_facts.cache_clear()
    assert state["model"] is None and state["on_ac_power"] is None
    assert state["low_power_mode"] is None and state["foreign_processes"] == []
    assert isinstance(state["loadavg"], (list, type(None)))
    json.dumps(state)


def test_host_state_on_this_platform_is_assembled_from_the_three_reports(monkeypatch):
    answers = {"sysctl -n hw.model": "MacBookPro18,2",
               "sysctl -n machdep.cpu.brand_string": "Apple M1 Max",
               "pmset -g batt": BATTERY, "pmset -g": " powermode            1\n",
               "pmset -g therm": "CPU_Speed_Limit \t= 100\n",
               "ps -axo pid=,ppid=,command=": "  7  1 zsh recut_metal_gates.sh a b\n"}
    monkeypatch.setattr(bench.sys, "platform", "darwin")
    monkeypatch.setattr(bench, "_read_command", lambda c: answers[" ".join(c)])
    bench._static_host_facts.cache_clear()
    try:
        state = bench.host_state()
    finally:
        bench._static_host_facts.cache_clear()
    assert state["model"] == "MacBookPro18,2" and state["cpu"] == "Apple M1 Max"
    assert state["on_ac_power"] is False and state["battery_percent"] == 77
    assert state["low_power_mode"] is True and state["cpu_speed_limit"] == 100
    assert len(state["foreign_processes"]) == 1
    json.dumps(state)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_a_metal_row_pins_the_metal_sources_as_well():
    import hashlib
    stamp = bench.provenance("metal")
    metal_keys = [key for key, _m, _p in bench.TABLE_PINNED_SOURCES["metal"]]
    assert metal_keys == list(report.METAL_PINNED_KEYS)
    assert list(stamp["sha256"]) == list(report.PINNED_KEYS) + metal_keys
    for key in metal_keys:
        path = os.path.join(REPO_API, stamp["paths"][key])
        with open(path, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == stamp["sha256"][key]
        assert key in stamp["git"]["dirty"] or stamp["git"]["error"] is not None
    assert stamp["paths"]["metal_launch"] == "meep_gpu/metal_kernels/launch.py"
    assert stamp["not_imported"] == {}
    # the NVIDIA stamp is the one it always was
    assert list(bench.provenance()["sha256"]) == list(report.PINNED_KEYS)


# ---------------------------------------------------------------------------
# The estimate the driver prints first
# ---------------------------------------------------------------------------

def _drive(n, probe=None):
    return {f"case_{i}": {"expect": "dispatch", "needs_probe": probe} for i in range(n)}


def test_estimate_known_value_scaling_degenerate_and_shape(monkeypatch):
    monkeypatch.delenv("PROBE_X", raising=False)
    one = bench.estimate(["case_0"], _drive(1), repeats=6, target_seconds=2.0,
                         warm_steps=6)
    # three legs x six windows x two seconds is the floor under any case
    assert one["seconds"] >= 36.0
    assert one["per_case_seconds"]["case_0"] == one["seconds"]
    ten = bench.estimate([f"case_{i}" for i in range(10)], _drive(10), repeats=6,
                         target_seconds=2.0, warm_steps=6)
    assert ten["seconds"] == pytest.approx(10 * one["seconds"])
    more = bench.estimate(["case_0"], _drive(1), repeats=12, target_seconds=2.0,
                          warm_steps=6)
    assert more["seconds"] - one["seconds"] == pytest.approx(3 * 6 * 2.0)
    assert bench.estimate([], {}, 6, 2.0, 6)["seconds"] == 0.0
    # a case whose licence this process does not carry is skipped, and costs nothing
    skipped = bench.estimate(["case_0"], _drive(1, probe="PROBE_X"), 6, 2.0, 6)
    assert skipped["seconds"] == 0.0 and skipped["skipped_without_probe"] == ["case_0"]
    json.dumps(ten)


def test_estimate_prefers_a_measured_step_cost_to_the_model():
    measured = {"case_0": {"fused": 0.5, "unfused": 1.0, "array": 0.001}}
    model = bench.estimate(["case_0"], _drive(1), 6, 2.0, 6)
    slow = bench.estimate(["case_0"], _drive(1), 6, 2.0, 6, measured=measured)
    assert slow["seconds"] > model["seconds"]
    assert slow["cost_source"]["case_0"] == "measured"
    assert model["cost_source"]["case_0"] == "model"


# ---------------------------------------------------------------------------
# main(), with the device and the cases stubbed
# ---------------------------------------------------------------------------

def _on_an_apple_gpu(monkeypatch, gpu="metal"):
    """The host fact ``--drive-table metal`` checks, stubbed with the device."""
    from meep_gpu import backends  # noqa: PLC0415

    monkeypatch.setattr(backends, "available_gpu", lambda: gpu)


@pytest.fixture
def stubbed_main(monkeypatch, fake_torch, metal):
    _on_an_apple_gpu(monkeypatch)
    monkeypatch.setattr(bench, "selected_cases", lambda *_a: ["pml_2d"])
    monkeypatch.setattr(bench, "drive_rows", lambda _n: {"pml_2d": dict(TWO_PAIRS)})
    monkeypatch.setattr(bench.e2e, "PREFER_GPU", True)
    monkeypatch.setattr(bench.route, "BACKEND", bench.route.BACKEND)
    monkeypatch.setattr(bench, "WITNESSES", [])
    monkeypatch.setattr(bench, "ONLY_ARMS", [])
    seen = {}

    # ``**_kw`` because ``run_case`` takes keyword arguments now (``monitors_mode``)
    # and this stub is about WITNESS STAMPING, not about the signature. A stub that
    # pins the argument list turns every honest parameter addition into a test
    # failure while checking nothing the test claims to check.
    def fake_run_case(case, _spec, _module, table, _gpu, counter, *_rest, **_kw):
        seen["witness"] = counter
        seen["prefer_gpu_during_the_case"] = bench.e2e.PREFER_GPU
        seen["compile_patched"] = (sys.modules["torch"].mps.compile_shader
                                   is not seen["original"])
        return {"case": case, "drive_table": table, "verdict": "SKIP-NOT-THIS-LEG"}

    seen["original"] = fake_torch.mps.compile_shader
    monkeypatch.setattr(bench, "run_case", fake_run_case)
    return seen


def test_main_on_the_metal_table_stamps_every_row(stubbed_main, tmp_path, fake_torch):
    assert bench.main(["--drive-table", "metal", "--out", str(tmp_path)]) == 1
    (row,) = [json.loads(line) for line in open(tmp_path / "rows.jsonl")]
    assert isinstance(stubbed_main["witness"], bench.MetalWitness)
    # LIFTED AS A GPU DRIVER ON THE APPLE GPU, AS THE ROUTE GATE LIFTS -- not a smoke row
    assert stubbed_main["prefer_gpu_during_the_case"] is True
    assert row["prefer_gpu"] is True and row["smoke"] is False
    assert row["sync_call"] == "torch.mps.synchronize"
    assert "metal_dispatch" in row["provenance"]["sha256"]
    assert stubbed_main["compile_patched"] is True
    assert fake_torch.mps.compile_shader is stubbed_main["original"], \
        "the compile witness must leave the platform entry point as it found it"
    # THE COMPILE FLOOR'S POSITIVE CONTROL, on every row including this skipped one
    assert row["compile_witness"]["attached"] is True
    assert row["compile_witness"]["library_cache_entries"] is not None
    summary = json.load(open(tmp_path / "summary.json"))
    assert summary["witnesses"] == ["metal"] and summary["drive_table"] == "metal"


def test_the_summary_medians_are_taken_over_witnessed_rows_only(
        monkeypatch, fake_torch, tmp_path):
    """A row nothing counted is not a substitution, whatever its floors say.

    ``launch_witness_available`` False means the launch clauses were not evaluated:
    the slot map proved the plans differ and no instrument saw a launch. Such a row
    is kept -- its seconds are real -- but a median over it is a median over a
    substitution nobody witnessed, so the medians name the rows they are over and the
    rest are named too.
    """
    cases = ["pml_2d", "pml_3d"]
    ratios = {"pml_2d": 2.0, "pml_3d": 6.0}
    _on_an_apple_gpu(monkeypatch)
    monkeypatch.setattr(bench, "selected_cases", lambda *_a: cases)
    monkeypatch.setattr(bench, "drive_rows",
                        lambda _n: {case: dict(TWO_PAIRS) for case in cases})
    monkeypatch.setattr(bench, "WITNESSES", [])
    monkeypatch.setattr(bench, "ONLY_ARMS", [])
    monkeypatch.setattr(bench, "run_case", lambda case, *_a, **_k: {
        "case": case, "verdict": "TIMED", "reportable": True, "floors": {},
        "ratios": {"unfused_over_fused": ratios[case],
                   "array_over_fused": ratios[case]},
        "substitution": {"launch_witness_available": case == "pml_2d"}})
    assert bench.main(["--drive-table", "metal", "--out", str(tmp_path)]) == 0
    summary = json.load(open(tmp_path / "summary.json"))
    assert summary["cases_reportable"] == 2
    assert summary["median_fused_vs_singles"] == 2.0
    assert summary["median_fused_vs_array"] == 2.0
    assert summary["medians_over"] == ["pml_2d"]
    assert summary["reportable_without_a_launch_witness"] == ["pml_3d"]
    # every reportable row keeps its own ratio; only the medians are narrowed
    assert set(summary["fused_vs_singles"]) == {"pml_2d", "pml_3d"}


def test_main_refuses_what_has_no_meaning_on_this_table(stubbed_main, tmp_path):
    with pytest.raises(SystemExit, match="only-arm"):
        bench.main(["--drive-table", "metal", "--out", str(tmp_path),
                    "--only-arm", B_PAIR])
    with pytest.raises(SystemExit, match="warm-steps"):
        bench.main(["--drive-table", "metal", "--out", str(tmp_path),
                    "--warm-steps", "1"])
    assert not (tmp_path / "rows.jsonl").exists()


@pytest.mark.parametrize("gpu", [None, "cuda"])
def test_main_refuses_the_metal_table_off_an_apple_gpu(stubbed_main, monkeypatch,
                                                       tmp_path, gpu):
    """``prefer_gpu=True`` is this host's GPU, so the Metal table needs an Apple one.

    On a CUDA host the legs would lift CuPy drivers and time the NVIDIA tables under
    Metal labels; on a host with no GPU the first lift raises. Refused before a row is
    written, while an estimate (which lifts nothing) is still answered.
    """
    _on_an_apple_gpu(monkeypatch, gpu)
    with pytest.raises(SystemExit, match="REFUSING: --drive-table metal"):
        bench.main(["--drive-table", "metal", "--out", str(tmp_path / "run")])
    assert "witness" not in stubbed_main, "a refused campaign runs no case"
    assert not (tmp_path / "run").exists()
    assert bench.main(["--drive-table", "metal", "--out", str(tmp_path / "run"),
                       "--estimate"]) == 0


@pytest.mark.parametrize("table", ["triton", "cuda"])
def test_main_refuses_an_nvidia_table_off_a_cuda_host(stubbed_main, monkeypatch,
                                                      tmp_path, table):
    """The NVIDIA drive tables need a CUDA host outside ``--smoke``.

    ``prefer_gpu=True`` resolves an Apple GPU driver on a Mac, so an unguarded run
    lifts, dispatches the Metal table and writes a timed row stamped with the NVIDIA
    table it was asked for (measured before the guard: 1 of 1 case, ``drive_table``
    triton, ``tables_dispatched`` metal). Refused before a row is written; ``--smoke``
    lifts the NumPy reference and ``--estimate`` lifts nothing, so both still run.
    """
    from meep_gpu import backends  # noqa: PLC0415

    monkeypatch.setattr(backends, "cupy_available", lambda: False)
    monkeypatch.setattr(bench.e2e, "LaunchCounters", lambda: SimpleNamespace(
        install=lambda: None, uninstall=lambda: None, triton=None, cuda=None))
    with pytest.raises(SystemExit, match=f"REFUSING: bench_fused_products.py "
                                         f"--drive-table {table} drives the NVIDIA "
                                         f"kernel tables"):
        bench.main(["--drive-table", table, "--out", str(tmp_path / "run")])
    assert "witness" not in stubbed_main, "a refused campaign runs no case"
    assert not (tmp_path / "run").exists()
    assert bench.main(["--drive-table", table, "--out", str(tmp_path / "run"),
                       "--estimate"]) == 0
    assert not (tmp_path / "run").exists()
    # ON A CUDA HOST the guard is silent, and the campaign reaches its cases.
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    assert bench.e2e.nvidia_host_refusal("bench_fused_products.py") is None


def test_main_estimate_prints_and_lifts_nothing(stubbed_main, tmp_path, capsys):
    assert bench.main(["--drive-table", "metal", "--out", str(tmp_path),
                       "--estimate"]) == 0
    printed = capsys.readouterr().out
    assert "ESTIMATE" in printed and "pml_2d" in printed
    assert "witness" not in stubbed_main, "an estimate runs no case"
    assert not (tmp_path / "rows.jsonl").exists()


# ---------------------------------------------------------------------------
# One realistic system: run_case end to end over three stub legs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gpu", [None, "cuda"])
def test_run_case_refuses_a_leg_that_is_not_an_apple_gpu_driver(
        witness, metal, monkeypatch, gpu):
    """A reference lift (``gpu`` None) never plans, so it cannot be a timed leg.

    The bench lifts through ``e2e.run_leg`` and carries its own post-lift check. The
    first leg is closed before the campaign stops.
    """
    closed = []

    def fake_run_leg(_case, _builder, label, _env, _counter, _gpu, _res):
        driver = _fused_two_pairs()
        driver.gpu = gpu
        driver.close = lambda: closed.append(label)
        return _leg(label, driver)

    monkeypatch.setattr(bench.e2e, "run_leg", fake_run_leg)
    monkeypatch.setattr(bench, "box_state", lambda: {"loadavg": [0.0, 0.0, 0.0]})
    monkeypatch.setattr(bench, "host_state", lambda: {
        "model": "stub", "on_ac_power": True, "low_power_mode": False,
        "loadavg": [0.1, 0.1, 0.1], "foreign_processes": []})
    stub_module = SimpleNamespace(CASES={"pml_2d": None},
                                  _fuse_env=lambda arms, reached_by="release": {},
                                  unfused_env=lambda: {}, ARRAY_ENV={})
    with pytest.raises(SystemExit, match=f"pml_2d/fused: the Metal bench lifted a "
                                         f"{gpu!r} driver"):
        bench.run_case("pml_2d", dict(TWO_PAIRS), stub_module, "metal", 0, witness,
                       None, repeats=2, target_seconds=0.002, step_cap=8,
                       warm_steps=4, memory_budget=0)
    assert closed == ["fused"]


def test_run_case_end_to_end_evaluates_every_floor_on_this_table(
        witness, metal, monkeypatch):
    built = {"fused": _fused_two_pairs, "unfused": _singles, "array": lambda: _Driver({})}

    def fake_run_leg(_case, _builder, label, _env, _counter, _gpu, _res):
        return _leg(label, built[label]())

    monkeypatch.setattr(bench.e2e, "run_leg", fake_run_leg)
    monkeypatch.setattr(bench, "box_state", lambda: {"loadavg": [0.0, 0.0, 0.0]})
    monkeypatch.setattr(bench, "host_state", lambda: {
        "model": "stub", "on_ac_power": True, "low_power_mode": False,
        "loadavg": [0.1, 0.1, 0.1], "foreign_processes": []})
    # the three leg environments are the module's to spell; the stub lift ignores them
    # ``_fuse_env`` carries the table's default route in its signature, which is where
    # the bench reads it from, so the stub spells it the way the Metal gate does.
    stub_module = SimpleNamespace(CASES={"pml_2d": None},
                                  _fuse_env=lambda arms, reached_by="release": {},
                                  unfused_env=lambda: {}, ARRAY_ENV={})
    row = bench.run_case("pml_2d", dict(TWO_PAIRS), stub_module, "metal", 0, witness,
                         None, repeats=2, target_seconds=0.002, step_cap=8,
                         warm_steps=4, memory_budget=0)
    assert row["verdict"].startswith("TIMED"), row.get("why")
    floors = dict(row["floors"])
    floors.pop("spread_within_gate")    # a millisecond stub window has no stable spread
    assert floors == {
        "substitution_proved": True, "bit_identical": True,
        "moved_words_positive": True, "non_finite_zero": True,
        "no_compiles_inside_timed_region": True,
        "singles_are_the_same_table": True,
        "windows_launched_at_the_proved_rate": True}
    assert row["substitution"]["launches_per_step"] == {"fused": 2.0, "unfused": 4.0,
                                                        "array": 0.0}
    assert row["monitors_detached"]["fused"] == {"_dft_monitors": 1,
                                                 "_flux_monitors": 0}
    assert row["residency"]["fused"]["mirrors"] == 45
    assert row["residency"]["array"] is None
    assert row["sync"]["call"] == "torch.mps.synchronize"
    assert row["sync"]["idle_seconds"] >= 0.0
    assert row["reached_by"] == "release"
    assert row["host_before"]["on_ac_power"] is True and "host_after" in row
    per = row["per_leg"]
    assert per["fused"]["launches_per_step"] == 2.0
    assert per["fused"]["compiled_calls_per_step"] == 2.0
    assert per["fused"]["residency_syncs_per_step"] == {"in": 2.0, "out": 2.0}
    assert per["array"]["launches_per_step"] == 0.0
    assert per["array"]["residency_syncs_per_step"] is None
    assert set(row["ratios"]) == {"unfused_over_fused", "array_over_fused"}
    # SHAPE: the row is what ``append_jsonl`` will be handed
    json.loads(json.dumps(row, default=str))
    # and the report renders it without a device either
    row.update(provenance={"sha256": {k: "f" * 64 for k in report.PINNED_KEYS
                                      + report.METAL_PINNED_KEYS}},
               prefer_gpu=True, smoke=False)
    assert "`pml_2d`" in report.render([row], "t")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _nvidia_row(case, vs_singles, table="triton"):
    per = {leg: {"median_seconds_per_step": s}
           for leg, s in (("fused", 0.001), ("unfused", 0.001 * vs_singles),
                          ("array", 0.004))}
    plan = {"arms": {"step_D": "fused pair D", "update_E": "fused pair D"},
            "composition": {"tables_dispatched": [table]}}
    return {"case": case, "drive_table": table, "utc": "2026-09-19T00:00:00Z",
            "grid_shape": [8, 8, 1], "reportable": True, "verdict": "TIMED",
            "prefer_gpu": True, "floors": {"substitution_proved": True},
            "ratios": {"unfused_over_fused": vs_singles, "array_over_fused": 4.0},
            "per_leg": per, "plans": {"fused": plan, "unfused": plan},
            "provenance": {"sha256": {k: "1" * 64 for k in report.PINNED_KEYS},
                           "git": {"head": "a" * 40, "any_dirty": False}},
            "substitution": {"tables_dispatched": [table],
                             "tables_by_leg": {"fused": [table], "unfused": [table]},
                             "fused_and_singles_used_the_same_table": True,
                             "launches_per_step": {"fused": 2, "unfused": 4}}}


def _metal_row(case, fused_ms=30.0, singles_ms=60.0, array_ms=0.6, cells=(200, 120, 1),
               reportable=True, smoke=False, on_ac=False, mirrors=45):
    per = {leg: {"median_seconds_per_step": ms / 1e3, "launches_per_step": lps,
                 "compiled_calls_per_step": lps,
                 "residency_syncs_per_step": ({"in": lps, "out": lps} if lps else None)}
           for leg, ms, lps in (("fused", fused_ms, 2.0), ("unfused", singles_ms, 4.0),
                                ("array", array_ms, 0.0))}
    plan = {"arms": {"step_B": B_PAIR, "update_H": B_PAIR, "step_D": D_PAIR,
                     "update_E": D_PAIR},
            "composition": {"tables_dispatched": ["metal"]}}
    floors = {"substitution_proved": True, "bit_identical": True,
              "moved_words_positive": True, "non_finite_zero": True,
              "no_compiles_inside_timed_region": True,
              "spread_within_gate": reportable, "singles_are_the_same_table": True,
              "windows_launched_at_the_proved_rate": True}
    host = {"model": "MacBookPro18,2", "cpu": "Apple M1 Max", "on_ac_power": on_ac,
            "power_source": "AC Power" if on_ac else "Battery Power",
            "low_power_mode": not on_ac, "loadavg": [2.5, 2.0, 1.5],
            "cpu_speed_limit": None, "foreign_processes": []}
    facts = {"mirrors": mirrors, "mirrored_bytes": mirrors * 96000,
             "invariant": "host-authoritative between launches",
             "host_authoritative_between_launches": True,
             "mode": "per-launch sync_in/sync_out installed at plan time"}
    return {"case": case, "drive_table": "metal", "utc": "2026-09-20T00:00:00Z",
            "grid_shape": list(cells), "reportable": reportable,
            "verdict": "TIMED" if reportable else "TIMED-NOT-REPORTABLE",
            "prefer_gpu": False, "smoke": smoke, "floors": floors,
            "ratios": {"unfused_over_fused": singles_ms / fused_ms,
                       "array_over_fused": array_ms / fused_ms},
            "per_leg": per, "plans": {"fused": plan, "unfused": plan},
            "residency": {"fused": facts, "unfused": facts, "array": None},
            "host_before": host, "host_after": host,
            "sync": {"call": "torch.mps.synchronize", "idle_seconds": 2e-5},
            "provenance": {"sha256": {k: "1" * 64 for k in report.PINNED_KEYS
                                      + report.METAL_PINNED_KEYS},
                           "git": {"head": "a" * 40, "any_dirty": False}},
            "substitution": {"tables_dispatched": ["metal"],
                             "tables_by_leg": {"fused": ["metal"], "unfused": ["metal"],
                                               "array": []},
                             "fused_and_singles_used_the_same_table": True,
                             "instruments": ["slot_arm_map", "metal_launch_counter",
                                             "gate_substitution_proof"],
                             "gate_proof": {"verdict": "EXACT"},
                             "launches_per_step": {"fused": 2.0, "unfused": 4.0,
                                                   "array": 0.0}}}


def _headline(text):
    return [line for line in text.splitlines() if line.startswith("Median **")]


def test_metal_rows_never_enter_the_nvidia_headline():
    nvidia = [_nvidia_row("a", 1.5), _nvidia_row("b", 2.5), _nvidia_row("c", 3.5, "cuda")]
    alone = report.render(nvidia, "t")
    mixed = report.render(nvidia + [_metal_row("pml_2d"), _metal_row("pml_3d", 45, 60)],
                          "t")
    assert _headline(alone) == ["Median **2.50x** over 3 cases whose control ran the "
                                "same table, faster on **3 of 3**; the one-pair / "
                                "two-pair split:"]
    assert _headline(mixed)[0] == _headline(alone)[0]
    assert "## The Metal table" in mixed and "## The Metal table" not in alone
    # and the NVIDIA table holds no Metal row
    nvidia_table = mixed[:mixed.index("## The Metal table")]
    assert "`pml_2d`" not in nvidia_table


def test_the_metal_section_says_what_the_rows_say():
    rows = [_metal_row("pml_2d", 30.0, 60.0, 0.6),
            _metal_row("pml_3d", 45.0, 60.0, 1.5, cells=(48, 48, 48), mirrors=105)]
    text = report.render(rows, "t")
    section = text[text.index("## The Metal table"):]
    # fused against the singles it replaces: its own headline, derived
    assert "Median **1.67x** over 2 cases" in section
    # and against the array path, said plainly and from the ratios
    assert "SLOWER than the array path on **2 of 2**" in section
    assert "the array path is a median **40.0x** faster" in section
    line = next(text for text in section.splitlines()
                if text.startswith("| `pml_2d` |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[:5] == ["`pml_2d`", "200x120x1", "30.000", "60.000", "0.600"]
    assert "**2.00x**" in line and "0.020x" in line
    assert "2 : 4" in line and "15.000 : 15.000" in line     # launches, ms per launch
    assert "| 45 |" in line
    assert "battery" in line.lower()
    # the residency context is read off the rows
    assert "host-authoritative between launches" in section
    assert "45–105 mirrors" in section
    assert "torch.mps.synchronize" in section
    assert "MacBookPro18,2" in section and "2 of 2 rows on battery" in section
    assert "low power mode on 2 of 2" in section.lower()
    # the NVIDIA-only prose stays out of a Metal-only document
    assert "deposit-repair bracket, a fixed host cost" not in text
    assert "MEEP_GPU_BACKEND_PREFERENCE" not in text


def test_a_metal_row_is_not_a_smoke_row_unless_it_says_so():
    text = report.render([_metal_row("pml_2d")], "t")
    assert "host smoke rows" not in text
    assert "Median **2.00x** over 1 case" in text
    smoke = report.render([_metal_row("pml_2d", smoke=True)], "t")
    assert "**1 of 1 rows are host smoke rows**" in smoke
    assert "Median **" not in smoke


def test_a_metal_row_that_failed_a_floor_gets_no_ratio_and_is_named():
    rows = [_metal_row("pml_2d"), _metal_row("noisy_2d", reportable=False)]
    text = report.render(rows, "t")
    assert "- **spread_within_gate** (1): `noisy_2d`" in text
    assert "| `noisy_2d` |" not in text
    assert "Median **2.00x** over 1 case" in text


def _contended(row, when="host_before"):
    return {**row, when: dict(row[when], foreign_processes=[
        "200 100 python gate_dispatch_metal_route.py"])}


def test_a_contended_metal_row_says_who_was_on_the_device():
    text = report.render([_contended(_metal_row("pml_2d"))], "t")
    assert "yes — before: 1 foreign process(es)" in text
    assert "**Contended rows.**" in text
    # AND IT IS NOT A QUIET-BOX MEDIAN. With no quiet row there is no quiet headline.
    assert "over 1 case whose control ran the same table" not in text
    assert "**No row was measured on a quiet device, so there is no quiet-box " \
           "median.**" in text
    assert "the median is **2.00x** against the singles" in text


def test_a_contended_row_is_kept_out_of_the_quiet_headline_and_reported_beside_it():
    """The CPU baseline's pairing: a quiet median, a contended one, and the difference.

    A neighbour on the device is exactly the thing this median must not average in --
    it is the most favourable-looking row here (6.00x against its own singles) and it
    is 6.00x because the box was busy, not because the product is faster there.
    """
    quiet = _metal_row("pml_2d")                       # 60 / 30 = 2.00x
    loud = _contended(_metal_row("pml_3d", 10.0, 60.0, 1.5))    # 6.00x, shared device
    text = report.render([quiet, loud], "t")
    assert "Median **2.00x** over 1 case whose control ran the same table" in text
    assert "**1 of 2 rows were measured while another process held the device" in text
    assert "over those alone the median is **6.00x**" in text
    assert "**Contended rows.**" in text


def test_a_metal_rows_digest_is_not_a_second_program_in_the_nvidia_table():
    """Which PROGRAM the NVIDIA rows were timed on is a question about NVIDIA rows.

    A Metal row pins a different set of sources and is timed on its own machine, so its
    `deposit_repair` digest differs from the NVIDIA campaign's as a matter of course.
    Counted into that decision it split the NVIDIA headline into per-digest buckets and
    added a `src` column to every NVIDIA row -- a document saying two programs were
    timed where one was.
    """
    nvidia = [_nvidia_row("a", 1.5), _nvidia_row("b", 2.5)]
    elsewhere = _metal_row("pml_2d")
    elsewhere["provenance"]["sha256"] = dict(elsewhere["provenance"]["sha256"],
                                             deposit_repair="2" * 64)
    text = report.render(nvidia + [elsewhere], "t")
    assert "### Rows timed on deposit_repair" not in text
    header = next(line for line in text.splitlines()
                  if line.startswith("| case | grid | fused ms/step"))
    assert not header.rstrip().endswith(" src |")
    # two NVIDIA programs in one document still split it, which is what the split is for
    second = _nvidia_row("c", 3.5)
    second["provenance"]["sha256"] = dict(second["provenance"]["sha256"],
                                          deposit_repair="3" * 64)
    split = report.render(nvidia + [second], "t")
    assert "### Rows timed on deposit_repair" in split
    assert next(line for line in split.splitlines()
                if line.startswith("| case | grid | fused ms/step")).rstrip().endswith(
                    " src |")


# ---------------------------------------------------------------------------
# Who else was on the device
# ---------------------------------------------------------------------------

#: One line of ``nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory``, and the
#: ``--query-gpu=index,uuid,…`` listing beside it: the shape every NVIDIA row records.
NEIGHBOUR = "GPU-00000000-0000-0000-0000-000000000003, 999, 5000 MiB"
PINNED_UUID = "GPU-00000000-0000-0000-0000-000000000006"
GPUS = (f"0, {PINNED_UUID}, 3 MiB, 0 %\n"
        "7, GPU-00000000-0000-0000-0000-000000000003, 5000 MiB, 100 %")


def _box(pin, apps=(NEIGHBOUR,)):
    return {"CUDA_VISIBLE_DEVICES": pin, "gpus": GPUS, "pid": 4507,
            "foreign_compute_apps_on_pinned_gpu": list(apps)}


def test_an_nvidia_neighbour_is_read_from_the_key_the_probe_writes():
    """``box_state`` writes ``foreign_compute_apps_on_pinned_gpu``, and nothing else.

    The report read ``foreign_compute_processes``, which no probe has ever written, so
    every NVIDIA row's contention column read *no* whatever the box was doing.
    """
    row = dict(_nvidia_row("a", 2.0), box_before=_box("7"))
    assert report.contended(row) == "before: 1 foreign process(es)"
    assert "yes — before: 1 foreign process(es)" in report.render([row], "t")


def test_a_neighbour_on_another_device_is_not_contention():
    """The field's NAME is only true when the pin is a UUID.

    ``box_state`` filters by UUID and skips the filter when ``CUDA_VISIBLE_DEVICES``
    is an index, so on an eight-GPU host the field lists every compute app on the box.
    Reading it as written marks a row contended because a stranger was busy on device
    7 while this row had device 0 to itself.
    """
    row = dict(_nvidia_row("a", 2.0), box_before=_box("0"))
    assert report.pinned_device(row["box_before"]) == PINNED_UUID
    assert report.contended(row) is None


def test_a_uuid_pin_needs_no_resolving_and_an_unresolvable_one_says_so():
    uuid_pin = dict(_nvidia_row("a", 2.0),
                    box_before=_box("GPU-00000000-0000-0000-0000-000000000003"))
    assert report.contended(uuid_pin) == "before: 1 foreign process(es)"
    # unpinned: the process saw every device, so every app on the box is a neighbour
    unpinned = dict(_nvidia_row("a", 2.0), box_before=_box("<unset>"))
    assert report.pinned_device(unpinned["box_before"]) is None
    assert report.contended(unpinned) == ("before: 1 foreign process(es) on a box "
                                          "whose pinned device the row does not name")


def test_a_row_the_metal_table_served_without_its_witness_enters_no_headline():
    """A row another table's drive served, which this table's witness never counted.

    Rows like it were written by ``--drive-table triton --smoke`` on a laptop before
    2026-09-27, when a ``prefer_gpu=False`` lift still planned the Metal table. A
    smoke lift is the NumPy reference now and the bench refuses an NVIDIA drive table
    off a CUDA host outside ``--smoke``, so the report's clause guards the rows
    already on disk.
    """
    seen = _metal_row("pml_2d")
    blind = _metal_row("pml_3d", 10.0, 60.0)       # would be the best ratio in the table
    blind["drive_table"] = "triton"
    blind["substitution"] = dict(blind["substitution"], instruments=["slot_arm_map"],
                                 launches_per_step={"fused": 0.0, "unfused": 0.0,
                                                    "array": 0.0})
    blind["substitution"].pop("gate_proof")
    text = report.render([seen, blind], "t")
    assert "**1 of 2 timed rows were measured WITHOUT this table's launch witness** " \
           "(`pml_3d`)" in text
    assert "Median **2.00x** over 1 case" in text          # the 6.00x row is not in it
    line = next(t for t in text.splitlines() if t.startswith("| `pml_3d` |"))
    assert "| no witness | — |" in line and "0 : 0" not in line
