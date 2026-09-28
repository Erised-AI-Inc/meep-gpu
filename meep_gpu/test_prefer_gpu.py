"""prefer_gpu: True is this host's GPU; False is the NumPy reference.

The reference never consults a kernel table, whatever the environment says. The
reference and stub tests run on every host; the tests that need a GPU driver skip by
declared resource (``mps_device`` / ``gpu_device`` / ``meep``) where it is absent.

THE SUITE'S DISPATCH PIN IS REMOVED HERE. ``conftest.py`` starts every test with
``MEEP_GPU_DISPATCH=0``; the autouse fixture below deletes it (and every other
switch), so a test that sets nothing measures the SHIPPED default and a test that
sets a value measures that value. Every test asserts what actually ran --
``driver.gpu``, ``driver.active_step_path`` and the ``fast_path_report`` record's
``decision`` / ``table`` / ``refused_because`` -- never the request alone.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from conftest import requires_resource_skip

from meep_gpu import backends, fastpath
from meep_gpu import driver as driver_module
from meep_gpu.driver import FdtdDriver

SWITCHES = ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED", "MEEP_GPU_KERNEL_TABLE",
            "MEEP_GPU_FUSE_ARMS", "MEEP_GPU_METAL_RESIDENCY", "MEEP_GPU_DISPATCH_LOG",
            "MEEP_GPU_SUBNORMAL_POLICY", "MEEP_GPU_SUBNORMAL_INSTALL")
STEPS = 5


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    fastpath.reset_dispatch_announcements()
    flushing_at_entry = backends.subnormals_flushed()
    yield
    try:
        from meep_gpu import host_writes, metal_dispatch  # noqa: PLC0415

        for residency in host_writes._live():  # noqa: SLF001
            metal_dispatch.release_residency_hold(residency, None)
    except Exception:  # noqa: BLE001
        pass
    subnormal_policy._reset_for_tests()  # noqa: SLF001
    if backends.subnormals_flushed() != flushing_at_entry:
        subnormal_policy._drive_host_fpu(flushing_at_entry)  # noqa: SLF001


def _require_metal() -> None:
    if backends.available_gpu() != "metal":
        requires_resource_skip("mps_device", "a prefer_gpu=True driver resolves Metal "
                                             "only on an Apple GPU")


def _require_gpu() -> None:
    if backends.available_gpu() is None:
        requires_resource_skip("gpu_device", "a prefer_gpu=True driver needs a CUDA "
                                             "device or an Apple GPU")


def _driver(prefer_gpu: bool) -> FdtdDriver:
    driver = FdtdDriver(cell_size=(6.0, 6.0, 0.0), resolution=10.0, dimensions=2,
                        force_complex_fields=False, prefer_gpu=prefer_gpu)
    driver.setup_pml({"x": 10.0, "y": 10.0})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0),
                       "amplitude": 1.0})
    return driver


def _meep_lines(err: str) -> list:
    return [line for line in err.splitlines() if line.startswith("meep_gpu:")]


#: The one line a reference driver prints, once per process, when the environment
#: asked for dispatch. Spelled in full: it is what a user reads.
REFERENCE_NOTE = ("meep_gpu: step path array (NumPy reference, prefer_gpu=False); "
                  "MEEP_GPU_DISPATCH={value} does not apply to a reference driver; "
                  "kernels are dispatched only by a driver built with "
                  "prefer_gpu=True, which needs a GPU on the host")

#: Appended when the value is one a GPU driver would refuse by name as well, so the
#: advice to build with ``prefer_gpu=True`` does not send the bad value along.
UNACCEPTED_VALUE_NOTE = ("; NOTE MEEP_GPU_DISPATCH={value} is not an accepted value "
                         "either, and a prefer_gpu=True driver refuses it by name "
                         "(unset or 1 dispatches, 0 disables)")


def _reference_note(value: str) -> str:
    """The reference's one line for this enable value, as a user reads it."""
    note = REFERENCE_NOTE.format(value=repr(value))
    if value not in fastpath.DISPATCH_ENABLE_VALUES:
        note += UNACCEPTED_VALUE_NOTE.format(value=repr(value))
    return note


# --- real hardware (this Mac) -----------------------------------------------------------


@pytest.mark.parametrize("enable", [None, "0", "1"])
def test_the_numpy_reference_on_an_mps_host_runs_numpy(monkeypatch, capsys, enable):
    if enable is not None:
        monkeypatch.setenv("MEEP_GPU_DISPATCH", enable)
    flushing_before = backends.subnormals_flushed()
    driver = _driver(False)
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.active_step_path == "array", (report or {}).get("slots")
    assert driver.xp is np and driver.gpu is None
    assert report["reference_driver"] is True and report["table"] is None
    assert report["decision"] == "refused" and report["step_path"] == "array"
    assert report["slots"] == {}, report["slots"]
    assert report["refused_because"] == fastpath.REFERENCE_REASON
    assert report["enable"]["value"] == enable
    driver.close()
    assert backends.subnormals_flushed() == flushing_before
    lines = _meep_lines(capsys.readouterr().err)
    if enable == "1":
        assert lines == [_reference_note("1")], lines
    else:
        assert lines == [], lines


def test_the_reference_note_prints_once_per_process(monkeypatch, capsys):
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "1")
    for _ in range(2):
        driver = _driver(False)
        driver.step()
        driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines


@pytest.mark.parametrize("enable", [None, "1"])
def test_a_gpu_driver_on_an_mps_host_dispatches_metal(monkeypatch, capsys, enable):
    """``prefer_gpu=True`` on an Apple GPU: Metal, by default and under an explicit 1."""
    _require_metal()
    if enable is not None:
        monkeypatch.setenv("MEEP_GPU_DISPATCH", enable)
    driver = _driver(True)
    assert driver.gpu == "metal" and driver.xp is np
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.active_step_path == "fused", report.get("refused_because")
    assert report["decision"] == "dispatched" and report["refused_because"] is None
    assert report["table"] == "metal" and report["reference_driver"] is False
    assert report["enable"]["value"] == enable
    dispatched = [slot for slot, entry in report["slots"].items()
                  if entry["state"] == "dispatched"]
    assert dispatched, report["slots"]
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    assert any(line.startswith("meep_gpu: step path fused;") for line in lines), lines


@pytest.mark.parametrize("veto", [("MEEP_GPU_DISPATCH", "0"), ("MEEP_GPU_FUSED", "0")])
def test_a_vetoed_gpu_driver_steps_the_array_path_quietly(monkeypatch, capsys, veto):
    _require_gpu()
    monkeypatch.setenv(*veto)
    driver = _driver(True)
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.gpu == backends.available_gpu()
    assert driver.active_step_path == "array"
    assert report["reference_driver"] is False
    assert report["decision"] == "refused" and report["table"] is None
    assert veto[0] in report["refused_because"], report["refused_because"]
    assert report["refused_because"] != fastpath.REFERENCE_REASON
    driver.close()
    assert _meep_lines(capsys.readouterr().err) == []


def test_an_uncertified_toolchain_runs_on_the_host_cpu_and_says_so(monkeypatch, capsys):
    """An Apple GPU request can finish on the host CPU: announced, recorded, not raised.

    ``prefer_gpu=True`` resolves from the HARDWARE; whether this torch is one a Metal
    weld ran on is rung 4M's question, asked at the freeze. A torch outside the
    certified set is refused there by name, and on an Apple GPU the array path it
    falls to is the host CPU, which the line a user reads says.
    """
    _require_metal()
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    real = metal_dispatch.metal_toolchain

    def other_torch():
        toolchain = dict(real())
        toolchain["version"] = "0.0.0-not-certified"
        return toolchain

    monkeypatch.setattr(metal_dispatch, "metal_toolchain", other_torch)
    driver = _driver(True)
    assert driver.gpu == "metal" and driver.xp is np
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.active_step_path == "array"
    assert report["decision"] == "refused" and report["reference_driver"] is False
    assert report["refused_because"].startswith(
        "torch 0.0.0-not-certified is not in the toolchains any Metal weld recorded "
        "running on"), report["refused_because"]
    assert report["environment"]["backend"] == "numpy"
    assert report["environment"]["torch_certified"] is False
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert lines[0].startswith(
        "meep_gpu: step path array on the host CPU; dispatch refused: torch "
        "0.0.0-not-certified is not in the toolchains"), lines[0]


def _entry_point_simulation(mp):
    return mp.Simulation(
        cell_size=mp.Vector3(6, 6, 0), resolution=10,
        boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ez,
                           center=mp.Vector3())])


def _import_meep_or_skip():
    try:
        import meep as mp  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("meep", "the entry points lift an mp.Simulation")
    mp.verbosity(0)
    return mp


def test_the_entry_points_on_an_mps_host():
    """lift default, prefer_gpu=True and run_on_gpu = Metal; False = reference; fields agree.

    The bare lift is the subject: nothing is set in the environment and nothing is
    passed, and what stepped is read off the driver.
    """
    _require_metal()
    mp = _import_meep_or_skip()
    import meep_gpu  # noqa: PLC0415

    def sim():
        return _entry_point_simulation(mp)

    assert "MEEP_GPU_DISPATCH" not in os.environ
    reference = meep_gpu.lift_simulation(sim(), prefer_gpu=False)
    default = meep_gpu.lift_simulation(sim())
    gpu = meep_gpu.lift_simulation(sim(), prefer_gpu=True)
    for _ in range(200):
        reference.step()
        default.step()
        gpu.step()
    assert reference.gpu is None and reference.active_step_path == "array"
    assert reference.fast_path_report()["reference_driver"] is True
    for name, lifted in (("default", default), ("prefer_gpu=True", gpu)):
        report = lifted.fast_path_report() or {}
        assert lifted.gpu == "metal" and lifted.xp is np, name
        assert lifted.active_step_path == "fused", (name, report.get("refused_because"))
        assert report["decision"] == "dispatched" and report["table"] == "metal", name
        assert report["reference_driver"] is False, name
    ez_ref = np.asarray(reference.get_field("Ez"))
    for name, lifted in (("default", default), ("prefer_gpu=True", gpu)):
        ez_gpu = np.asarray(lifted.get_field("Ez"))
        differing = int(np.count_nonzero(ez_ref != ez_gpu))
        print(f"MERGED_EZ {name} differing={differing} of {ez_ref.size} "
              f"max_abs={float(np.max(np.abs(ez_ref - ez_gpu))):.3e} "
              f"peak={float(np.max(np.abs(ez_ref))):.3e}", flush=True)
        assert differing == 0, name
    reference.close()
    default.close()
    gpu.close()
    result = meep_gpu.run_on_gpu(sim(), until=1.0)
    assert result.driver.gpu == "metal"
    assert result.driver.active_step_path == "fused"
    result.close()


def test_a_default_lift_on_this_host_is_its_gpu_or_a_named_refusal():
    """A bare ``lift_simulation(sim)`` on THIS host, no stubs: its GPU, or the refusal."""
    mp = _import_meep_or_skip()
    import meep_gpu  # noqa: PLC0415

    assert "MEEP_GPU_DISPATCH" not in os.environ
    gpu = backends.available_gpu()
    sim = _entry_point_simulation(mp)
    if gpu is None:
        with pytest.raises(RuntimeError) as raised:
            meep_gpu.lift_simulation(sim)
        assert "pass prefer_gpu=False to step the NumPy reference" in str(raised.value)
        assert sim.fields is None, "the refusal is raised before sim.init_sim()"
        return
    driver = meep_gpu.lift_simulation(sim)
    assert driver.gpu == gpu
    assert driver.xp.__name__ == {"cuda": "cupy", "metal": "numpy"}[gpu]
    driver.step()
    assert driver.fast_path_report()["reference_driver"] is False
    driver.close()


def test_an_explicit_false_lift_is_the_reference_on_every_host(monkeypatch, capsys):
    """``prefer_gpu=False`` never reads the host and never consults the planner.

    It prints no dispatch line. The one line a lift may print is the MEEP-precision
    notice, which a double-precision MEEP build gets once per process whatever
    ``prefer_gpu`` says. The process's record of that notice is emptied for this test,
    so the line is expected exactly once on a double-precision build and never on a
    single-precision one, whichever test lifted first.
    """
    mp = _import_meep_or_skip()
    import meep_gpu  # noqa: PLC0415
    from meep_gpu import from_meep  # noqa: PLC0415

    def refuse(*_args, **_kwargs):
        raise AssertionError("a reference lift read the host or consulted the planner")

    monkeypatch.setattr(driver_module, "plan_fast_path", refuse)
    monkeypatch.setattr(backends, "available_gpu", refuse)
    monkeypatch.setattr(backends, "_gpu_route", refuse)
    monkeypatch.setattr(from_meep, "_PRECISION_ANNOUNCED", set())
    driver = meep_gpu.lift_simulation(_entry_point_simulation(mp), prefer_gpu=False)
    for _ in range(STEPS):
        driver.step()
    report = driver.fast_path_report()
    assert driver.gpu is None and driver.xp is np
    assert driver.active_step_path == "array"
    assert report["reference_driver"] is True and report["table"] is None
    assert report["refused_because"] == fastpath.REFERENCE_REASON
    precision = "single" if mp.is_single_precision() else "double"
    assert driver.lift_record == {"meep_version": mp.__version__,
                                  "meep_precision": precision,
                                  "engine_precision": "single"}, driver.lift_record
    driver.close()
    expected = [from_meep.DOUBLE_PRECISION_NOTICE] if precision == "double" else []
    assert _meep_lines(capsys.readouterr().err) == expected


def test_a_reference_step_imports_neither_torch_nor_the_metal_ladder():
    """In a fresh process, with the enable exported: the reference pays for nothing."""
    code = (
        "import sys, json, numpy as np\n"
        "from meep_gpu.driver import FdtdDriver\n"
        "from meep_gpu import backends\n"
        "before = backends.subnormals_flushed()\n"
        "d = FdtdDriver(cell_size=(6.0, 6.0, 0.0), resolution=10.0, dimensions=2,"
        " force_complex_fields=False)\n"
        "d.setup_pml({'x': 10.0, 'y': 10.0})\n"
        "for _ in range(3): d.step()\n"
        "d.close()\n"
        "print('RESULT ' + json.dumps({'torch': 'torch' in sys.modules,"
        " 'metal_dispatch': 'meep_gpu.metal_dispatch' in sys.modules,"
        " 'flush_changed': backends.subnormals_flushed() != before,"
        " 'path': d.active_step_path}))\n")
    env = dict(os.environ, MEEP_GPU_DISPATCH="1")
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=300)
    line = [x for x in done.stdout.splitlines() if x.startswith("RESULT ")]
    assert line, done.stderr[-2000:]
    result = json.loads(line[0][len("RESULT "):])
    assert result == {"torch": False, "metal_dispatch": False, "flush_changed": False,
                      "path": "array"}, result


# --- stubs: any host ---------------------------------------------------------------------


@pytest.mark.parametrize("kill", [None, "0", "yes"])
@pytest.mark.parametrize("enable", [None, "0", "1", "true"])
def test_a_reference_driver_never_consults_the_planner(monkeypatch, capsys, enable, kill):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a reference driver consulted the planner")

    monkeypatch.setattr(driver_module, "plan_fast_path", refuse)
    if enable is not None:
        monkeypatch.setenv("MEEP_GPU_DISPATCH", enable)
    if kill is not None:
        monkeypatch.setenv("MEEP_GPU_FUSED", kill)
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    for _ in range(3):
        driver.step()
    report = driver.fast_path_report()
    assert driver.gpu is None and driver.active_step_path == "array"
    assert report["reference_driver"] is True and report["table"] is None
    assert report["decision"] == "refused"
    assert report["refused_because"] == fastpath.REFERENCE_REASON
    assert report["environment"] == {"not_read": "reference driver"}
    assert report["enable"]["value"] == enable
    driver.invalidate_fast_path()
    assert "superseded" in driver.fast_path_report()
    driver.step()
    assert driver.fast_path_report()["reference_driver"] is True
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    expected = [_reference_note(enable)] if enable in ("1", "true") else []
    assert lines == expected, lines
    if enable == "true":
        # The advice names prefer_gpu=True, where this value would be refused by
        # name; the line says so rather than sending the value along.
        assert "is not an accepted value either" in lines[0], lines[0]
    elif enable == "1":
        assert "is not an accepted value" not in lines[0], lines[0]


def test_a_gpu_intent_numpy_refusal_says_host_cpu(monkeypatch, capsys):
    monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: False)
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "1")
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    driver.gpu = "metal"      # an Apple GPU driver's engine, without the hardware
    driver.step()
    report = driver.fast_path_report()
    assert report["reference_driver"] is False
    assert "backend is not CuPy" in report["refused_because"]
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    assert any("step path array on the host CPU; dispatch refused" in x for x in lines), lines


def test_the_reference_reason_gives_no_advice_a_gpu_less_host_cannot_take():
    """The reference reads nothing about the host, so its reason names no host GPU."""
    reason = fastpath.REFERENCE_REASON
    assert "this host's GPU" not in reason
    assert "prefer_gpu=True" in reason and "raises without one" in reason
    assert "this host's GPU" not in _reference_note("1")


#: The announced refusals rungs 1 and 2 answer before the ``environment`` block is
#: read: an unrecognised enable, an unrecognised kill switch, a retired switch name.
EARLY_REFUSALS = [
    pytest.param({"MEEP_GPU_DISPATCH": "true"},
                 "MEEP_GPU_DISPATCH='true' is not an accepted value", id="enable_true"),
    pytest.param({"MEEP_GPU_DISPATCH": ""},
                 "MEEP_GPU_DISPATCH='' is not an accepted value", id="enable_empty"),
    pytest.param({"MEEP_GPU_FUSED": "off"},
                 "MEEP_GPU_FUSED='off'", id="kill_switch_off"),
    pytest.param({"TRIDENT_FDTD_TRITON": "1"},
                 "retired environment switch set: TRIDENT_FDTD_TRITON",
                 id="retired_name"),
]


@pytest.mark.parametrize("environment,named", EARLY_REFUSALS)
def test_an_early_refusal_on_a_numpy_engine_says_host_cpu(monkeypatch, capsys,
                                                          environment, named):
    """Rungs 1 and 2 say where the array path runs, as rung 3 and below do.

    An Apple GPU driver's array path is the host CPU, and that is the whole run; the
    line a user reads says so whichever rung refused.
    """
    def no_hardware_read():
        raise AssertionError("rungs 1 and 2 answer before the hardware is read")

    monkeypatch.setattr(fastpath, "metal_hardware_present", no_hardware_read)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    driver.gpu = "metal"      # an Apple GPU driver's engine, without the hardware
    driver.step()
    report = driver.fast_path_report()
    assert report["reference_driver"] is False and report["decision"] == "refused"
    assert named in report["refused_because"], report["refused_because"]
    assert report["engine_backend"] == "numpy"
    assert "backend" not in report["environment"], report["environment"]
    driver.close()
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1, lines
    assert lines[0].startswith(
        "meep_gpu: step path array on the host CPU; dispatch refused: "), lines[0]
    assert named in lines[0], lines[0]


@pytest.mark.parametrize("environment,named", EARLY_REFUSALS)
def test_an_early_refusal_on_a_cupy_like_engine_does_not(monkeypatch, capsys,
                                                         environment, named):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    xp = types.SimpleNamespace(__name__="cupy", __version__="13.5.1")
    fastpath.plan_fast_path(object(), object(), types.SimpleNamespace(xp=xp))
    assert fastpath.last_dispatch_report()["engine_backend"] == "cupy"
    lines = _meep_lines(capsys.readouterr().err)
    assert len(lines) == 1 and named in lines[0], lines
    assert lines[0].startswith("meep_gpu: step path array; dispatch refused: "), lines[0]


def test_the_record_names_the_engine_on_every_driver(monkeypatch):
    """``engine_backend`` is on the reference's record and on a planned one."""
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "0")
    reference = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    reference.step()
    assert reference.fast_path_report()["engine_backend"] == "numpy"
    assert reference.fast_path_report()["reference_driver"] is True
    gpu = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    gpu.gpu = "metal"
    gpu.step()
    assert gpu.fast_path_report()["engine_backend"] == "numpy"
    assert gpu.fast_path_report()["reference_driver"] is False
    reference.close()
    gpu.close()


def test_a_cupy_like_refusal_does_not_say_host_cpu(monkeypatch, capsys):
    monkeypatch.setenv("MEEP_GPU_KERNEL_TABLE", "metal")
    xp = types.SimpleNamespace(__name__="cupy", __version__="13.5.1")
    fastpath.plan_fast_path(object(), object(), types.SimpleNamespace(xp=xp))
    lines = _meep_lines(capsys.readouterr().err)
    assert lines and not any("host CPU" in x for x in lines), lines


def test_repr_names_the_reference():
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    assert "backend=numpy reference" in repr(driver)
    driver.gpu = "metal"
    assert "backend=numpy, gpu=metal" in repr(driver)
    driver.close()


def test_the_cpu_request_is_numpy_and_no_gpu():
    assert backends.resolve_backend(prefer_gpu=False) == (np, None)


def _host(monkeypatch, cuda: bool, mps: bool):
    monkeypatch.setattr(backends, "cupy_available", lambda: cuda)
    monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: mps)


def test_a_gpu_request_on_an_mps_host_resolves_metal(monkeypatch):
    _host(monkeypatch, cuda=False, mps=True)
    assert backends.resolve_backend(prefer_gpu=True) == (np, "metal")


def test_a_gpu_request_with_neither_route_raises_naming_the_escape(monkeypatch):
    _host(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError) as raised:
        backends.resolve_backend(prefer_gpu=True)
    message = str(raised.value)
    assert message.startswith("GPU backend requested but unavailable: ")
    assert "prefer_gpu=False" in message


def test_metal_refuses_a_nonzero_gpu_id(monkeypatch):
    _host(monkeypatch, cuda=False, mps=True)
    with pytest.raises(ValueError, match="gpu_id must be 0"):
        backends.resolve_backend(prefer_gpu=True, gpu_id=1)


def test_cuda_wins_when_both_are_present(monkeypatch):
    _host(monkeypatch, cuda=True, mps=True)
    assert backends.available_gpu() == "cuda"


@pytest.mark.parametrize("cuda,mps", [(False, True), (False, False)])
def test_is_available_iff_a_gpu_request_resolves(monkeypatch, cuda, mps):
    import meep_gpu  # noqa: PLC0415

    _host(monkeypatch, cuda=cuda, mps=mps)
    try:
        backends.resolve_backend(prefer_gpu=True)
        resolves = True
    except RuntimeError:
        resolves = False
    assert meep_gpu.is_available() is resolves
    assert (meep_gpu.missing_dependencies() == []) is resolves


@pytest.mark.parametrize("gpu_id,mps,error", [(0, False, RuntimeError), (1, True, ValueError)])
def test_lift_refuses_before_init_sim(monkeypatch, gpu_id, mps, error):
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415

    _host(monkeypatch, cuda=False, mps=mps)
    calls = []
    stub = types.SimpleNamespace(init_sim=lambda: calls.append("init_sim"))
    with pytest.raises(error):
        lift_simulation(stub, prefer_gpu=True, gpu_id=gpu_id)
    assert calls == []


@pytest.mark.parametrize("missing", [["cupy"], ["cuda-device"], ["mps-device"], ["torch"]])
def test_a_default_lift_with_no_gpu_route_raises_by_name_before_init_sim(monkeypatch,
                                                                         missing):
    """The BARE call on a host with no route: named, with the escape, nothing built."""
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415

    _host(monkeypatch, cuda=False, mps=False)
    monkeypatch.setattr(backends, "missing_dependencies", lambda: list(missing))
    calls = []
    stub = types.SimpleNamespace(init_sim=lambda: calls.append("init_sim"))
    with pytest.raises(RuntimeError) as raised:
        lift_simulation(stub)
    assert calls == []
    assert str(raised.value) == (
        f"GPU backend requested but unavailable: {missing[0]}. prefer_gpu=True runs on "
        "a CUDA device through CuPy or on an Apple GPU through Metal, and this host "
        "has neither; pass prefer_gpu=False to step the NumPy reference (host CPU, no "
        "kernels). lift_simulation and run_on_gpu run on the GPU by default: "
        "prefer_gpu=True is what a call passing no prefer_gpu asks for.")


def test_a_default_lift_refuses_a_nonzero_gpu_id_on_an_apple_gpu(monkeypatch):
    """``gpu_id`` alone is a GPU request now: refused by name, before ``init_sim``."""
    from meep_gpu.from_meep import lift_simulation  # noqa: PLC0415

    _host(monkeypatch, cuda=False, mps=True)
    calls = []
    stub = types.SimpleNamespace(init_sim=lambda: calls.append("init_sim"))
    with pytest.raises(ValueError, match="gpu_id must be 0"):
        lift_simulation(stub, gpu_id=1)
    assert calls == []


def test_run_on_gpu_with_no_gpu_route_raises_the_lifts_refusal(monkeypatch):
    """The one-call path reaches the same refusal, with the same text."""
    try:
        import meep  # noqa: F401, PLC0415
    except ImportError:
        requires_resource_skip("meep", "run_on_gpu imports MEEP before it lifts")
    from meep_gpu.from_meep import run_on_gpu  # noqa: PLC0415

    _host(monkeypatch, cuda=False, mps=False)
    calls = []
    stub = types.SimpleNamespace(init_sim=lambda: calls.append("init_sim"))
    with pytest.raises(RuntimeError) as raised:
        run_on_gpu(stub, until=1.0)
    assert calls == []
    message = str(raised.value)
    assert message.startswith("GPU backend requested but unavailable: ")
    assert "pass prefer_gpu=False to step the NumPy reference" in message
    assert message.endswith("prefer_gpu=True is what a call passing no prefer_gpu asks for.")


# --- prefer_gpu=True on THIS host, no stubs ---------------------------------------------


def test_a_gpu_request_on_this_host_resolves_its_gpu_or_raises_by_name():
    """What ``prefer_gpu=True`` does here, read off the driver and not the request."""
    gpu = backends.available_gpu()
    if gpu is None:
        with pytest.raises(RuntimeError) as raised:
            FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8, prefer_gpu=True)
        message = str(raised.value)
        assert message.startswith("GPU backend requested but unavailable: ")
        assert "pass prefer_gpu=False to step the NumPy reference" in message
        return
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8, prefer_gpu=True)
    assert driver.gpu == gpu
    assert driver.xp.__name__ == {"cuda": "cupy", "metal": "numpy"}[gpu]
    assert f"gpu={gpu}" in repr(driver)
    assert not hasattr(driver, "is_gpu"), "is_gpu was renamed outright to gpu"
    driver.close()


def test_an_apple_gpu_has_one_device():
    _require_metal()
    with pytest.raises(ValueError) as raised:
        FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8, prefer_gpu=True, gpu_id=1)
    assert str(raised.value) == (
        "gpu_id=1 selects a CUDA device; this host's GPU is an Apple GPU, which has "
        "one device, so gpu_id must be 0")


def test_the_refusal_names_what_is_missing_and_the_escape(monkeypatch):
    """The whole message a GPU-less host prints, not a fragment of it."""
    _host(monkeypatch, cuda=False, mps=False)
    monkeypatch.setattr(backends, "missing_dependencies", lambda: ["cupy"])
    with pytest.raises(RuntimeError) as raised:
        FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8, prefer_gpu=True)
    assert str(raised.value) == (
        "GPU backend requested but unavailable: cupy. prefer_gpu=True runs on a CUDA "
        "device through CuPy or on an Apple GPU through Metal, and this host has "
        "neither; pass prefer_gpu=False to step the NumPy reference (host CPU, no "
        "kernels).")


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_missing_dependencies_is_never_empty_without_a_route(monkeypatch, platform):
    """``[]`` means a route exists; with none, this platform's route names a piece."""
    _host(monkeypatch, cuda=False, mps=False)
    monkeypatch.setattr(backends.sys, "platform", platform)
    missing = backends.missing_dependencies()
    assert missing, "is_available() is False and nothing was named as missing"
    metal_pieces = {"torch", "mps-device", "torch.mps.compile_shader"}
    expected = metal_pieces if platform == "darwin" else {"cupy", "cuda-device"}
    assert set(missing) <= expected, missing


def test_a_cuda_request_is_unchanged_but_for_the_label(monkeypatch):
    """The CuPy branch: device selected, ``(cupy, "cuda")`` — never Metal when both exist."""
    selected = []
    cupy = types.ModuleType("cupy")
    cupy.cuda = types.SimpleNamespace(
        Device=lambda gpu_id: types.SimpleNamespace(use=lambda: selected.append(gpu_id)))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    _host(monkeypatch, cuda=True, mps=True)
    monkeypatch.setattr(backends, "guard_kernel_compilation", lambda _cupy: True)
    assert backends.resolve_backend(prefer_gpu=True, gpu_id=3) == (cupy, "cuda")
    assert selected == [3]


def test_the_meep_entry_points_default_to_the_gpu_and_the_engine_to_the_reference():
    """``lift_simulation`` and ``run_on_gpu`` default True; ``FdtdDriver`` defaults False."""
    import inspect  # noqa: PLC0415

    from meep_gpu.from_meep import lift_simulation, run_on_gpu  # noqa: PLC0415

    assert inspect.signature(FdtdDriver.__init__).parameters["prefer_gpu"].default is False
    assert inspect.signature(lift_simulation).parameters["prefer_gpu"].default is True
    assert inspect.signature(run_on_gpu).parameters["prefer_gpu"].default is True
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    driver.step()
    assert driver.gpu is None and driver.active_step_path == "array"
    assert driver.fast_path_report()["reference_driver"] is True
    driver.close()
