"""The harness's host guards: which GPU a gate grades, checked before it lifts.

``prefer_gpu=True`` is THIS HOST'S GPU -- CuPy on a CUDA host, Metal kernels over
NumPy host arrays on an Apple GPU -- and ``prefer_gpu=False`` is the NumPy reference,
which never consults a kernel table. A gate written for one kernel table therefore
cannot take "the lift succeeded" as "the lift is mine":

* the three NVIDIA gates refuse a non-smoke run on a host with no CUDA device, where
  the same lift would produce Apple GPU drivers (or raise);
* the Metal route gate and the Metal bench refuse a host whose GPU is not an Apple
  one, and check ``driver.gpu == "metal"`` after every lift, so a reference lift
  cannot be compared or timed as a fused leg;
* the throughput benchmark refuses ``--backend numpy --step-path fused``, which asks
  the reference to dispatch;
* the NVIDIA gates' array-leg control accepts a reference leg on a ``--smoke`` run
  (the reference reads no kill switch, so its record cannot name one) and on no
  other;
* the corpus sweep's ``--dispatch`` lifts ``prefer_gpu=True`` in its children, is
  refused on a host with no GPU route, and every row records what the lift resolved.

Every test stubs the host fact (``backends.cupy_available`` /
``backends.available_gpu``) and lifts nothing, so the file runs on any host. Each
refusal is checked for landing BEFORE the gate's output directory exists.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
for _path in (HERE, REPO_API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import benchmark_gpu  # noqa: E402
import gate_dispatch_end_to_end as e2e  # noqa: E402
import gate_dispatch_fused_route as fused_route  # noqa: E402
import gate_dispatch_metal_route as metal_route  # noqa: E402
import gate_driver_route_fused as driver_route  # noqa: E402

from meep_gpu import backends, fastpath  # noqa: E402
from parity.meep_gpu import sweep_corpus_lift_parity as sweep  # noqa: E402


def _cuda(monkeypatch, present: bool) -> None:
    monkeypatch.setattr(backends, "cupy_available", lambda: present)


def _gpu(monkeypatch, gpu) -> None:
    monkeypatch.setattr(backends, "available_gpu", lambda: gpu)


def _keep(monkeypatch, module, *names: str) -> None:
    """Restore the run-axis globals a gate's ``main`` sets before it refuses."""
    for name in names:
        monkeypatch.setattr(module, name, getattr(module, name))


# ---------------------------------------------------------------------------
# The NVIDIA gates
# ---------------------------------------------------------------------------

def test_the_nvidia_refusal_names_the_gate_and_the_escape(monkeypatch):
    _cuda(monkeypatch, False)
    assert e2e.nvidia_host_refusal("gate_x.py") == (
        "REFUSING: gate_x.py drives the NVIDIA kernel tables and this host has no "
        "CUDA device (meep_gpu.backends.cupy_available() is False). prefer_gpu=True "
        "resolves this host's GPU, which is not an NVIDIA table here; pass --smoke to "
        "exercise the harness on the NumPy reference.")
    _cuda(monkeypatch, True)
    assert e2e.nvidia_host_refusal("gate_x.py") is None


NVIDIA_GATES = [
    pytest.param(e2e, "gate_dispatch_end_to_end.py", ("PREFER_GPU", "_PROGRESS_PATH"),
                 id="end_to_end"),
    pytest.param(fused_route, "gate_dispatch_fused_route.py",
                 ("BACKEND", "TABLE_PREFERENCE"), id="fused_route"),
    pytest.param(driver_route, "gate_driver_route_fused.py", ("_PROGRESS_PATH",),
                 id="driver_route"),
]


@pytest.mark.parametrize("gate,name,globals_set", NVIDIA_GATES)
def test_an_nvidia_gate_refuses_a_host_with_no_cuda_device(
        monkeypatch, capsys, tmp_path, gate, name, globals_set):
    """A non-smoke run off NVIDIA exits 2, by name, and writes nothing."""
    _cuda(monkeypatch, False)
    _keep(monkeypatch, gate, *globals_set)
    _keep(monkeypatch, e2e, "PREFER_GPU")
    out = tmp_path / "leg"
    monkeypatch.setattr(sys, "argv", [name, "--out", str(out)])
    assert gate.main() == 2
    err = capsys.readouterr().err
    assert f"REFUSING: {name} drives the NVIDIA kernel tables" in err, err
    assert "pass --smoke" in err, err
    assert not out.exists(), "the refusal must land before the leg directory exists"


def test_the_nvidia_guard_is_not_reached_by_a_smoke_run(monkeypatch):
    """``--smoke`` lifts the reference and is never refused for the host's GPU.

    Read off the guard's own condition rather than by running a smoke leg, which
    lifts real simulations: each ``main`` consults the refusal only when it is about
    to lift ``prefer_gpu=True``.
    """
    import inspect  # noqa: PLC0415

    _cuda(monkeypatch, False)
    for gate, spelling in ((e2e, "if PREFER_GPU else None"),
                           (fused_route, "None if arguments.smoke"),
                           (driver_route, "if prefer_gpu else None")):
        source = inspect.getsource(gate.main)
        assert "nvidia_host_refusal(" in source, gate.__name__
        assert spelling in source, (gate.__name__, spelling)


# ---------------------------------------------------------------------------
# The NVIDIA gates' array-leg control
# ---------------------------------------------------------------------------

KILL_SWITCH_PLAN = {"reference_driver": False,
                    "refused_because": "kill switch MEEP_GPU_FUSED=0"}
REFERENCE_PLAN = {"reference_driver": True,
                  "refused_because": fastpath.REFERENCE_REASON}

ARRAY_LEG_CONTROLS = [
    pytest.param(e2e.killswitch_leg_is_clean, id="end_to_end_and_fused_route"),
    pytest.param(driver_route.array_leg_is_clean, id="driver_route"),
]


def _array_leg(plan, launches: int = 0, path: str = "array"):
    """One oracle leg, in both gates' spellings of the launch count."""
    return {"active_step_path": path, "plan": dict(plan),
            "triton_launches": {"total": launches}, "launches": {"total": launches}}


def test_the_plan_summaries_carry_the_reference_flag():
    """The control reads ``reference_driver`` off the leg's plan summary."""
    report = {"reference_driver": True, "refused_because": fastpath.REFERENCE_REASON}
    for summary in (e2e._plan_summary, driver_route._plan_summary):  # noqa: SLF001
        assert summary(report)["reference_driver"] is True
        assert summary({"reference_driver": False})["reference_driver"] is False


@pytest.mark.parametrize("control", ARRAY_LEG_CONTROLS)
def test_a_smoke_leg_is_the_reference_and_the_control_accepts_it(monkeypatch, control):
    """``--smoke`` lifts ``prefer_gpu=False``: array by construction, no kill switch read."""
    monkeypatch.setattr(e2e, "PREFER_GPU", False)
    answer = control(_array_leg(REFERENCE_PLAN))
    assert answer["clean"] is True and answer["failures"] == []
    assert answer["reference_driver"] is True
    assert answer["refused_because"] == fastpath.REFERENCE_REASON


@pytest.mark.parametrize("control", ARRAY_LEG_CONTROLS)
def test_a_reference_leg_on_a_gate_run_fails_the_control(monkeypatch, control):
    """A gate run lifts GPU drivers; a reference leg there is not its oracle."""
    monkeypatch.setattr(e2e, "PREFER_GPU", True)
    answer = control(_array_leg(REFERENCE_PLAN))
    assert answer["clean"] is False
    assert any("only a --smoke run lifts the reference" in failure
               for failure in answer["failures"]), answer


@pytest.mark.parametrize("control", ARRAY_LEG_CONTROLS)
@pytest.mark.parametrize("prefer_gpu", [True, False])
def test_a_gpu_leg_must_still_be_refused_at_the_kill_switch(monkeypatch, control,
                                                            prefer_gpu):
    monkeypatch.setattr(e2e, "PREFER_GPU", prefer_gpu)
    answer = control(_array_leg(KILL_SWITCH_PLAN))
    assert answer["clean"] is True and answer["reference_driver"] is False
    other = control(_array_leg({"reference_driver": False,
                                "refused_because": "backend is not CuPy"}))
    assert other["clean"] is False
    assert any("does not name the kill switch" in failure
               for failure in other["failures"]), other


@pytest.mark.parametrize("control", ARRAY_LEG_CONTROLS)
def test_a_reference_leg_that_launched_or_fused_is_not_clean(monkeypatch, control):
    """The path and the launch witness are asked of a reference leg too."""
    monkeypatch.setattr(e2e, "PREFER_GPU", False)
    assert control(_array_leg(REFERENCE_PLAN, launches=3))["clean"] is False
    assert control(_array_leg(REFERENCE_PLAN, path="fused"))["clean"] is False


# ---------------------------------------------------------------------------
# The corpus sweep's --dispatch lane
# ---------------------------------------------------------------------------

def test_the_sweep_refuses_dispatch_on_a_host_with_no_gpu_route(
        monkeypatch, capsys, tmp_path):
    """``--dispatch`` lifts ``prefer_gpu=True``; with no GPU every child would raise."""
    _gpu(monkeypatch, None)
    monkeypatch.setattr(backends, "missing_dependencies", lambda: ["cupy"])
    out = tmp_path / "sweep"
    monkeypatch.setattr(sys, "argv", ["sweep_corpus_lift_parity.py", "--stage", "lift",
                                      "--dispatch", "--out", str(out)])
    assert sweep.main() == 2
    err = capsys.readouterr().err
    assert "REFUSING: --dispatch lifts prefer_gpu=True" in err, err
    assert "available_gpu() is None; missing: cupy" in err, err
    assert not out.exists(), "the refusal must land before the output directory exists"


class _SweepDriver:
    """What the sweep's child reads off a lifted driver."""

    shape = (4, 4, 1)
    sigma_lift = None

    def __init__(self, prefer_gpu: bool) -> None:
        self.gpu = "metal" if prefer_gpu else None
        self.active_step_path = "fused" if prefer_gpu else "array"

    def fast_path_report(self):
        if self.gpu is None:
            return {"reference_driver": True, "decision": "refused", "table": None,
                    "slots": {}, "refused_because": fastpath.REFERENCE_REASON}
        return {"reference_driver": False, "decision": "dispatched", "table": "metal",
                "slots": {"update_B": {"state": "dispatched"},
                          "update_H": {"state": "array"}},
                "refused_because": None}

    def close(self) -> None:
        pass


@pytest.mark.parametrize("dispatch", [False, True])
def test_the_sweep_child_lifts_what_dispatch_asks_and_records_it(
        monkeypatch, tmp_path, dispatch):
    """The lift follows ``--dispatch`` and the row says what the lift resolved."""
    import json  # noqa: PLC0415
    import meep_gpu  # noqa: PLC0415

    asked = []

    def fake_lift(sim, *, prefer_gpu, progress_cb=None):
        asked.append(prefer_gpu)
        return _SweepDriver(prefer_gpu)

    monkeypatch.setattr(sweep, "capture_simulation",
                        lambda script: ({"script": "stub.py", "outcome": "captured",
                                         "has_simulation": True},
                                        object(), lambda: None))
    monkeypatch.setattr(meep_gpu, "gpu_compatibility",
                        lambda sim: SimpleNamespace(supported=True, reasons=[]))
    monkeypatch.setattr(meep_gpu, "lift_simulation", fake_lift)
    row_path = tmp_path / "row.json"
    sweep.child_lift("stub.py", str(row_path), sweep.ProgressLog(None, "stub.py"),
                     prefer_gpu=dispatch)
    row = json.loads(row_path.read_text(encoding="utf-8"))
    assert asked == [dispatch]
    assert row["prefer_gpu"] is dispatch and row["lifts"] is True
    assert row["driver_gpu"] == ("metal" if dispatch else None)


def test_the_sweep_reads_the_step_path_off_the_driver():
    """A parity row's ``step_path`` and ``dispatch`` block are the driver's own."""
    fused = sweep.dispatch_facts(_SweepDriver(True))
    assert fused["step_path"] == "fused"
    assert fused["dispatch"] == {
        "reference_driver": False, "decision": "dispatched", "table": "metal",
        "dispatched_slots": ["update_B"], "slots": 2, "refused_because": None}
    reference = sweep.dispatch_facts(_SweepDriver(False))
    assert reference["step_path"] == "array"
    assert reference["dispatch"]["reference_driver"] is True
    assert reference["dispatch"]["table"] is None
    assert reference["dispatch"]["refused_because"] == fastpath.REFERENCE_REASON


def test_the_sweep_hands_dispatch_to_its_children():
    """The flag reaches the child's command line and both of its lifts."""
    import inspect  # noqa: PLC0415

    main = inspect.getsource(sweep.main)
    assert 'command += ["--dispatch"]' in main
    assert "child_lift(args.one, args.out_json, progress, prefer_gpu=args.dispatch)" in main
    assert "prefer_gpu=args.dispatch)" in main
    assert "prefer_gpu=False" not in inspect.getsource(sweep.child_lift)
    assert "prefer_gpu=False" not in inspect.getsource(sweep.child_parity)


# ---------------------------------------------------------------------------
# The Metal route gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gpu", [None, "cuda"])
def test_the_metal_gate_refuses_a_host_whose_gpu_is_not_apple(
        monkeypatch, capsys, tmp_path, gpu):
    _gpu(monkeypatch, gpu)
    _keep(monkeypatch, e2e, "PREFER_GPU", "_PROGRESS_PATH")
    out = tmp_path / "leg"
    assert metal_route.main(["--out", str(out), "--cases", "pml_2d"]) == 2
    err = capsys.readouterr().err
    assert "REFUSING: gate_dispatch_metal_route.py drives the Metal kernel table" in err
    assert f"available_gpu() is {gpu!r}" in err, err
    assert not out.exists(), "the refusal must land before the leg directory exists"


def test_the_metal_refusal_is_silent_on_an_apple_gpu(monkeypatch):
    _gpu(monkeypatch, "metal")
    assert metal_route.metal_host_refusal("gate_dispatch_metal_route.py") is None


@pytest.mark.parametrize("gpu", [None, "cuda"])
def test_a_metal_leg_must_be_an_apple_gpu_driver(gpu):
    """A reference lift (``gpu`` None) never plans; a CuPy driver is the other tables'."""
    with pytest.raises(SystemExit, match=f"lifted a {gpu!r} driver for pml_2d/fused"):
        metal_route.require_metal_driver(SimpleNamespace(gpu=gpu), "pml_2d/fused")
    metal_route.require_metal_driver(SimpleNamespace(gpu="metal"), "pml_2d/fused")


def test_a_driver_with_no_gpu_attribute_is_refused_too():
    with pytest.raises(SystemExit, match="lifted a None driver"):
        metal_route.require_metal_driver(object(), "pml_2d/array")


def test_every_metal_gate_lift_is_checked(monkeypatch):
    """Both ``e2e.run_leg`` sites and ``_lift`` hand their drivers to the check."""
    seen = []
    monkeypatch.setattr(metal_route, "require_metal_driver",
                        lambda driver, what: seen.append((driver.gpu, what)))
    monkeypatch.setattr(e2e, "run_leg",
                        lambda case, _builder, label, *_rest: {
                            "label": label, "driver": SimpleNamespace(gpu="metal")})
    counter = object()
    spec = {"arms": ["fused magnetic B/H pair"], "reached_by": "release"}
    fused, array = metal_route._pair_of_lifted(  # noqa: SLF001
        "pml_2d", spec, None, counter, "fused")
    assert (fused["label"], array["label"]) == ("fused", "array")
    assert seen == [("metal", "pml_2d/fused"), ("metal", "pml_2d/array")]
    assert metal_route.e2e is e2e


class _WitnessDriver:
    """A lifted driver as the band witness reads it, stepping the path it is told."""

    gpu = "metal"

    def __init__(self, path: str) -> None:
        self.active_step_path = path
        self.steps = 0
        self.closed = 0

    def run(self, num_steps: int) -> None:
        self.steps += num_steps

    def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("path", ["array", "fused"])
def test_the_band_witness_checks_that_it_dispatched_nothing(monkeypatch, path):
    """The witness lifts a GPU driver, so its array path is asserted, not assumed.

    It censuses the array path with the host keeping; a leg that fused would census a
    trajectory stepped under the table's installed flush. The campaign hands the leg
    ``MEEP_GPU_DISPATCH=0``; a run that did not is refused at its first step.
    """
    driver = _WitnessDriver(path)
    monkeypatch.setattr(metal_route, "_lift", lambda case, res: (driver, None, None))
    monkeypatch.setattr(metal_route, "_census_state", lambda _driver: {"Ez": 3})
    if path == "fused":
        with pytest.raises(RuntimeError, match="the band witness stepped 'fused' at "
                                               "step 1"):
            metal_route.band_witness("pml_2d", None, 4)
        assert (driver.steps, driver.closed) == (1, 1)
        return
    row = metal_route.band_witness("pml_2d", None, 4)
    assert (driver.steps, driver.closed) == (4, 1)
    assert row["verdict"] == "BAND-NON-EMPTY" and row["subnormal_words_total"] == 12
    assert row["step_path"] == "array" and row["driver_gpu"] == "metal"


def test_the_metal_gate_stamps_how_its_legs_were_lifted():
    """The driver_dispatch record reads the lift off the legs' provenance."""
    import inspect  # noqa: PLC0415

    source = inspect.getsource(metal_route._provenance)  # noqa: SLF001
    assert '"lift": {"prefer_gpu": True, "driver_gpu": "metal"' in source
    assert "e2e.PREFER_GPU = True" in inspect.getsource(metal_route.main)


# ---------------------------------------------------------------------------
# The throughput benchmark
# ---------------------------------------------------------------------------

def test_the_benchmark_refuses_a_fused_run_on_the_reference(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark_gpu.py", "--backend", "numpy",
                                      "--step-path", "fused"])
    monkeypatch.delenv("MEEP_GPU_DISPATCH", raising=False)
    with pytest.raises(SystemExit) as raised:
        benchmark_gpu.main()
    assert raised.value.code == 2
    err = capsys.readouterr().err
    assert ("--backend numpy is the NumPy reference and never dispatches; use "
            "--backend gpu") in err, err
    # REFUSED AT THE COMMAND LINE: before the tool pins or reads the enable.
    assert "MEEP_GPU_DISPATCH" not in os.environ
