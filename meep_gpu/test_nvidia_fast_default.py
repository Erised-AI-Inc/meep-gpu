"""The NVIDIA kernels run fast by default; certification is a guarantee a run opts into.

Fast by default: a run is never slower because a configuration is uncertified. Each run
states whether it is certified, and ``MEEP_GPU_HARDWARE=certified`` refuses an uncertified
configuration by name. On the NVIDIA tables that means:

* a Triton that imports but cannot compile and launch a kernel on this device is dropped
  by name at rung 4 (the health probe), so the hand-CUDA table composes alone rather than
  every slot Triton would have held going to the array path at the warm pass;
* a device is supported when the table can run on it -- the health probe launched on it
  (Triton table), the installed NVRTC can target it (hand-CUDA table) -- not when it falls
  inside 7.0 to 9.0, and only ``certified`` refuses a supported device no gate ran on;
* every Triton from 3.1 on is supported, and one whose version cannot be placed is
  admitted unchecked;
* a table certified here no longer drops a supported one (there is no rung 4e);
* Triton's subnormal policy is installed only when a Triton plan is in the step;
* a keep-policy CuPy cache directory that cannot serve is replaced by a fresh, marked one;
* a caller-installed uniform flush runs uncertified, with the NOTE, by default;
* ``certified`` also compares the stack the kernels run beside by name: CuPy, NVRTC, the
  CUDA runtime and driver, the host's OS and architecture, NumPy, and the MEEP build,
  against the stack the device's own compute capability was certified on.

Everything here runs on a host with no NVIDIA device: the device, its NVRTC, Triton and its
health probe, and the stack are stand-ins (``test_dispatch_contract``), each through the
seam the ladder reads it by.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import copy
import os
import platform
import subprocess
import sys
import types

import numpy as np
import pytest

from meep_gpu import backends, fastpath, fastpath_cuda
from meep_gpu import subnormal_policy as policy_module
from meep_gpu.test_capability_records import (ledger_also_certified_on,
                                              record_certified_stack)
from meep_gpu.test_dispatch_contract import (
    NVRTC_118_ARCHS,
    NVRTC_128_ARCHS,
    CountingPlan,
    block_cuda,
    certified_envelope_fields,
    certified_envelope_grid,
    certified_envelope_pml,
    certified_stack_identity,
    cuda_step_plan,
    cupy_like_grid,
    cupy_like_grid_with_device,
    failing_triton_probe,
    install_composer,
    install_cuda_composer,
    stand_in_stack,
    step_plan,
    stub_triton,
)

SWITCH = fastpath.HARDWARE_SWITCH
SWITCHES = ("MEEP_GPU_DISPATCH", "MEEP_GPU_FUSED", "MEEP_GPU_KERNEL_TABLE",
            "MEEP_GPU_BACKEND_PREFERENCE", "MEEP_GPU_FUSE_ARMS", "MEEP_GPU_DISPATCH_LOG",
            "MEEP_GPU_SUBNORMAL_POLICY", "MEEP_GPU_SUBNORMAL_INSTALL", "MEEP_GPU_WARM",
            SWITCH, fastpath.RETIRED_HARDWARE_SWITCH)

#: How a Triton that cannot launch fails, by cause: the three known causes.
LAUNCH_FAILURES = {
    "no libcuda.so symlink": subprocess.CalledProcessError(
        1, ["/usr/bin/gcc", "main.c", "-lcuda"], stderr=b"/usr/bin/ld: cannot find -lcuda"),
    "no C compiler": RuntimeError("Failed to find C compiler. Please specify via CC "
                                  "environment variable."),
    "no ptxas target": RuntimeError("ptxas fatal   : Value 'sm_120' is not defined for "
                                    "option 'gpu-name'"),
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    """The switches unset, dispatch enabled, nothing said, a cold policy and a host FPU
    that keeps subnormals (``test_uncertified_opt_in`` gives the reason for the last)."""
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    policy_module._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    fastpath.reset_dispatch_announcements()
    flushing_at_entry = backends.subnormals_flushed()
    if flushing_at_entry:
        policy_module._drive_host_fpu(False)  # noqa: SLF001
    yield
    policy_module._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    fastpath.reset_dispatch_announcements()
    if backends.subnormals_flushed() != flushing_at_entry:
        policy_module._drive_host_fpu(flushing_at_entry)  # noqa: SLF001


def _notes(err):
    return [line for line in err.splitlines()
            if line.startswith("meep_gpu:") and "NOT CERTIFIED" in line]


def _grid(capability=(8, 6), archs=NVRTC_118_ARCHS, nvrtc=(11, 8)):
    """The configuration both releases admit, on a stand-in device and NVRTC."""
    grid = certified_envelope_grid()
    grid.xp.cuda = cupy_like_grid_with_device(capability=capability, nvrtc=nvrtc,
                                              nvrtc_archs=archs).xp.cuda
    return grid


def _two_tables(monkeypatch, *, capability=(8, 6), archs=NVRTC_118_ARCHS,
                version="3.1.0", probe=None, triton_plan=None, nvrtc=(11, 8)):
    """One freeze on a stand-in host with both tables: Triton fills ``step_B``, the
    hand-CUDA table a fused pair over ``step_D``/``update_E``."""
    stub_triton(monkeypatch, version, probe=probe)
    install_composer(monkeypatch, triton_plan or step_plan(
        {"step_B": CountingPlan("b")}, {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    return fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid(capability, archs, nvrtc))


def _refused():
    return fastpath.last_dispatch_report()["refused_because"]


# --- The Triton health probe ------------------------------------------------------------


@pytest.mark.parametrize("cause", sorted(LAUNCH_FAILURES))
def test_a_triton_that_cannot_launch_leaves_the_hand_cuda_table_running_alone(
        monkeypatch, capsys, cause):
    """The 1.0 failure: Triton composed first, held every slot, and the warm pass sent
    those slots to the array path. Now Triton is dropped by name and the hand-CUDA table
    serves its slots, on the certified device, certified."""
    error = LAUNCH_FAILURES[cause]

    def probe(xp):
        raise error

    plan = _two_tables(monkeypatch, probe=probe)
    assert plan is not None, _refused()
    record = plan.report()
    assert record["decision"] == "dispatched"
    assert record["table"] == "cuda"
    assert record["composition"]["tables_dispatched"] == ["cuda"]
    assert sorted(plan.slots) == ["step_D", "update_E"]
    assert record["certified"] is True
    triton = record["tables"]["triton"]
    assert triton["candidate"] is False
    assert triton["refused_because"].startswith(
        "Triton 3.1.0 imports but cannot compile and launch a kernel on this device "
        f"({type(error).__name__}: "), triton
    assert triton["refused_because"].endswith(
        "so the triton table is dropped and the hand-CUDA table composes without it")
    health = record["environment"]["triton_health"]
    assert health["healthy"] is False and health["capability"] == "8.6"
    assert record["environment"]["device_supported_by_table"]["triton"] is False
    assert record["arbitration"]["effective_order"] == ["cuda"]
    assert _notes(capsys.readouterr().err) == []


def test_a_triton_that_cannot_launch_is_dropped_on_every_scope(monkeypatch):
    for value in ("certified", "any"):
        monkeypatch.setenv(SWITCH, value)
        plan = _two_tables(monkeypatch, probe=failing_triton_probe([(8, 6)]))
        assert plan is not None, (value, _refused())
        assert plan.report()["composition"]["tables_dispatched"] == ["cuda"], value
        assert plan.report()["tables"]["triton"]["candidate"] is False, value


def test_a_version_admitted_triton_that_fails_its_probe_names_nothing_that_ran(
        monkeypatch, capsys):
    """Triton 3.1.1 is admitted on its version and then dropped by its probe; with the
    other table blocked the plan is refused, and the version admission moves to
    ``dropped`` with the probe's refusal rather than staying ``admitted``."""
    block_cuda(monkeypatch)
    stub_triton(monkeypatch, "3.1.1", probe=failing_triton_probe([(8, 6)]))
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device()) is None
    record = fastpath.last_dispatch_report()
    assert record["uncertified"]["admitted"] == []
    (dropped,) = record["uncertified"]["dropped"]
    assert (dropped["what"], dropped["read"]) == ("Triton", "3.1.1")
    assert dropped["dropped_because"] == record["tables"]["triton"]["refused_because"]
    assert _notes(capsys.readouterr().err) == []


def test_a_healthy_triton_composes_first_as_before(monkeypatch):
    plan = _two_tables(monkeypatch)
    assert plan is not None, _refused()
    record = plan.report()
    assert record["composition"]["tables_dispatched"] == ["cuda", "triton"]
    assert record["environment"]["triton_health"]["healthy"] is True
    assert record["environment"]["device_supported_by_table"]["triton"] is True
    assert record["certified"] is True


def test_the_probe_runs_once_per_triton_version_and_device(monkeypatch):
    calls = []
    stub_triton(monkeypatch, probe=lambda xp: calls.append(xp))
    xp = cupy_like_grid_with_device().xp
    first = fastpath.triton_health(xp, "3.1.0")
    second = fastpath.triton_health(xp, "3.1.0")
    assert first["healthy"] is True and "cached" not in first
    assert second["healthy"] is True and second["cached"] is True
    assert len(calls) == 1
    fastpath.triton_health(cupy_like_grid_with_device(capability=(8, 9)).xp, "3.1.0")
    fastpath.triton_health(xp, "3.2.0")
    assert len(calls) == 3


def test_a_probe_with_nothing_to_launch_on_drops_nothing(monkeypatch):
    """Unknown, not unhealthy: an unread device, or a triton with no ``jit``."""
    stub_triton(monkeypatch, probe=failing_triton_probe([(8, 6)]))
    unread = fastpath.triton_health(cupy_like_grid().xp, "3.1.0")
    assert unread["healthy"] is None
    assert unread["not_run_because"].startswith("the device could not be read")
    del sys.modules["triton"].jit
    nojit = fastpath.triton_health(cupy_like_grid_with_device().xp, "3.1.0")
    assert nojit["healthy"] is None and "has no jit" in nojit["not_run_because"]
    # ...and the ladder keeps Triton a candidate on an unread device.
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    block_cuda(monkeypatch)
    sys.modules["triton"].jit = lambda function: function
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid()) is not None


def test_the_libcuda_check_leads_and_does_not_decide(monkeypatch):
    from meep_gpu.triton_kernels import launch

    def no_symlink():
        raise RuntimeError("Triton links its cuda_utils extension with -lcuda, which "
                           "needs a libcuda.so development symlink\nmore")

    monkeypatch.setattr(launch, "require_triton", no_symlink)
    stub_triton(monkeypatch)
    xp = cupy_like_grid_with_device().xp
    healthy = fastpath.triton_health(xp, "3.1.0")
    assert healthy["healthy"] is True
    assert healthy["libcuda_check"].startswith("RuntimeError: Triton links its cuda_utils")
    stub_triton(monkeypatch, probe=failing_triton_probe([(8, 6)], "ld: cannot find -lcuda"))
    monkeypatch.setattr(launch, "require_triton", no_symlink)
    failed = fastpath.triton_health(xp, "3.1.0")
    assert failed["healthy"] is False
    assert failed["because"] == (
        "RuntimeError: ld: cannot find -lcuda; triton_kernels.launch's libcuda check "
        "also failed: RuntimeError: Triton links its cuda_utils extension with -lcuda, "
        "which needs a libcuda.so development symlink")


def test_the_probe_launch_writes_and_checks_real_words(monkeypatch):
    """The probe's own launch, on a stand-in Triton whose kernel runs on NumPy: what it
    writes is read back and compared, so a launch that writes the wrong words fails."""
    import meep_gpu.triton_kernels.launch as launch

    class Kernel:
        def __init__(self, function, wrong=False):
            self.function, self.wrong = function, wrong

        def __getitem__(self, grid):
            def run(source, target, count, BLOCK):
                values = source.array * np.float32(2.0) + np.float32(1.0)
                target.array[:] = values + (1 if self.wrong else 0)
            return run

    module = types.ModuleType("triton")
    module.__version__ = "3.1.0"
    module.language = types.ModuleType("triton.language")
    numpy_xp = types.SimpleNamespace(asarray=np.array, asnumpy=np.asarray)
    for wrong in (False, True):
        module.jit = lambda function, wrong=wrong: Kernel(function, wrong)
        monkeypatch.setitem(sys.modules, "triton", module)
        monkeypatch.setitem(sys.modules, "triton.language", module.language)
        monkeypatch.setattr(launch, "CupyPointer", lambda array: types.SimpleNamespace(
            array=array))
        if wrong:
            with pytest.raises(RuntimeError, match="the probe kernel launched and wrote"):
                fastpath._triton_probe_launch(numpy_xp)  # noqa: SLF001
        else:
            fastpath._triton_probe_launch(numpy_xp)  # noqa: SLF001


# --- Support is a self-check, not 7.0 to 9.0 --------------------------------------------


def test_a_blackwell_device_with_a_capable_nvrtc_dispatches_by_default(monkeypatch, capsys):
    """Compute capability 12.0, NVRTC 12.8 (which targets it), Triton 3.1 (whose ptxas
    does not): the hand-CUDA table runs, uncertified, with the NOTE; 1.0 refused both
    tables by the range and ran the whole step on the array path."""
    plan = _two_tables(monkeypatch, capability=(12, 0), archs=NVRTC_128_ARCHS,
                       nvrtc=(12, 8), probe=failing_triton_probe([(12, 0)]))
    assert plan is not None, _refused()
    record = plan.report()
    assert record["composition"]["tables_dispatched"] == ["cuda"]
    assert record["certified"] is False
    (entry,) = record["uncertified"]["served"]
    assert (entry["table"], entry["read"], entry["supported"]) == ("cuda", "12.0", True)
    assert "the NVRTC 12.8 of this CuPy targets sm_120" in entry["because"], entry
    assert record["environment"]["supported_by"]["cuda"]["compiles_for"] == "sm_120"
    (note,) = _notes(capsys.readouterr().err)
    assert "GPU compute capability 12.0 (certified:" in note, note
    # ``certified`` refuses it by name.
    monkeypatch.setenv(SWITCH, "certified")
    assert _two_tables(monkeypatch, capability=(12, 0), archs=NVRTC_128_ARCHS,
                       nvrtc=(12, 8), probe=failing_triton_probe([(12, 0)])) is None
    row = fastpath.last_dispatch_report()["tables"]["cuda"]["refused_because"]
    assert row.startswith("GPU compute capability 12.0 is not in the recorded"), row
    assert row.endswith(fastpath.CERTIFIED_SCOPE_TAIL), row


def test_a_newer_triton_that_launches_on_blackwell_composes_too(monkeypatch):
    plan = _two_tables(monkeypatch, capability=(12, 0), archs=NVRTC_128_ARCHS,
                       nvrtc=(12, 8), version="3.3.1")
    assert plan is not None, _refused()
    record = plan.report()
    assert record["composition"]["tables_dispatched"] == ["cuda", "triton"]
    assert record["certified"] is False
    served = {(entry["table"], entry["what"]) for entry in record["uncertified"]["served"]}
    assert served == {("triton", "Triton"), ("triton", "GPU compute capability"),
                      ("cuda", "GPU compute capability")}, served


@pytest.mark.parametrize("capability", [(6, 1), (7, 0), (10, 0)])
def test_a_device_outside_the_old_range_runs_where_nvrtc_targets_it(monkeypatch,
                                                                      capability):
    archs = NVRTC_128_ARCHS if capability >= (10, 0) else NVRTC_118_ARCHS
    nvrtc = (12, 8) if capability >= (10, 0) else (11, 8)
    plan = _two_tables(monkeypatch, capability=capability, archs=archs, nvrtc=nvrtc)
    assert plan is not None, _refused()
    assert "cuda" in plan.report()["composition"]["tables_dispatched"]


def test_a_device_nvrtc_cannot_target_is_refused_by_name_unless_any(monkeypatch):
    """12.0 under NVRTC 11.8: the hand-CUDA table cannot run there."""
    plan = _two_tables(monkeypatch, capability=(12, 0),
                       probe=failing_triton_probe([(12, 0)]))
    assert plan is None
    row = fastpath.last_dispatch_report()["tables"]["cuda"]["refused_because"]
    assert ("outside what that table can run on: the NVRTC 11.8 of this CuPy targets "
            "sm_35 to sm_90, and none of those loads on sm_120") in row, row
    assert row.endswith(fastpath.UNSUPPORTED_HINT), row


@pytest.mark.parametrize("capability,archs,supported", [
    ((8, 9), (35, 50, 60, 70, 75, 80, 86), True), ((9, 0), (35, 50, 60, 70, 75, 80, 86),
                                                    False),
    ((12, 0), NVRTC_118_ARCHS, False), ((12, 0), NVRTC_128_ARCHS, True)])
def test_the_nvrtc_check_follows_binary_compatibility(capability, archs, supported):
    """A binary for X.y loads on X.z (z >= y): the newest listed target of the device's
    own major."""
    xp = cupy_like_grid_with_device(capability=capability, nvrtc_archs=archs).xp
    assert fastpath_cuda.nvrtc_target(xp, f"{capability[0]}.{capability[1]}")[
        "supported"] is supported


def test_an_nvrtc_that_will_not_answer_leaves_the_device_unchecked(monkeypatch, capsys):
    # 8.7 retires 8.9 as the uncertified example: 8.9 is certified now; no round ran on 8.7.
    plan = _two_tables(monkeypatch, capability=(8, 7), nvrtc=None)
    assert plan is not None, _refused()
    cuda = [entry for entry in plan.report()["uncertified"]["served"]
            if entry["table"] == "cuda"]
    (entry,) = cuda
    assert "supported" not in entry, entry
    assert entry["because"].startswith(
        "GPU compute capability 8.7 could not be checked against the cuda table "
        "(this CuPy's NVRTC would not say what it targets"), entry
    monkeypatch.setenv(SWITCH, "certified")
    assert _two_tables(monkeypatch, capability=(8, 7), nvrtc=None) is None


# --- Triton 3.1 and later ---------------------------------------------------------------


def test_triton_3_2_runs_by_default(monkeypatch, capsys):
    block_cuda(monkeypatch)
    stub_triton(monkeypatch, "3.2.0")
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    plan = fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device())
    assert plan is not None, _refused()
    record = plan.report()
    assert record["composition"]["tables_dispatched"] == ["triton"]
    assert record["certified"] is False
    assert record["environment"]["triton_supported"] is True
    (note,) = _notes(capsys.readouterr().err)
    assert "Triton 3.2.0 (certified: 3.1.0)" in note, note
    monkeypatch.setenv(SWITCH, "certified")
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device()) is None
    assert _refused().startswith("no kernel table is a candidate")


def test_a_triton_version_that_cannot_be_placed_is_admitted_unchecked(monkeypatch):
    block_cuda(monkeypatch)
    stub_triton(monkeypatch, "unknown")
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    plan = fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device())
    assert plan is not None, _refused()
    (entry,) = [entry for entry in plan.report()["uncertified"]["served"]
                if entry["what"] == "Triton"]
    assert "supported" not in entry
    assert "its version has no leading major.minor to place" in entry["because"]
    monkeypatch.setenv(SWITCH, "certified")
    assert fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device()) is None


# --- No rung 4e -------------------------------------------------------------------------


def test_a_certified_triton_table_no_longer_pushes_out_the_cuda_table(monkeypatch, capsys):
    """9.0 certified on the Triton table only (an H100 certification for one table): the
    hand-CUDA table composes beside it rather than being dropped, and the run says it
    is uncertified on the hand-CUDA table."""
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {
        "triton_kernels": ledger_also_certified_on("triton", "9.0"),
        "cuda_kernels": dict(fastpath._fingerprints(fastpath.CUDA_TABLE))})  # noqa: SLF001
    plan = _two_tables(monkeypatch, capability=(9, 0))
    assert plan is not None, _refused()
    record = plan.report()
    assert record["tables"]["cuda"] == {"candidate": True, "refused_because": None}
    assert record["composition"]["tables_dispatched"] == ["cuda", "triton"]
    assert sorted(plan.slots) == ["step_B", "step_D", "update_E"]
    assert record["certified"] is False
    (entry,) = record["uncertified"]["served"]
    assert (entry["table"], entry["read"]) == ("cuda", "9.0")
    (note,) = _notes(capsys.readouterr().err)
    assert "supported but not certified bit-identical on the cuda table" in note, note
    # ``certified`` keeps the certified table alone.
    monkeypatch.setenv(SWITCH, "certified")
    plan = _two_tables(monkeypatch, capability=(9, 0))
    assert plan is not None, _refused()
    assert plan.report()["composition"]["tables_dispatched"] == ["triton"]
    assert plan.report()["certified"] is True


def test_a_triton_build_string_beside_a_certified_cuda_table_composes(monkeypatch):
    """The 1.0 population rung 4e reached: cc 8.6, a Triton spelled ``3.1.0+<hash>``."""
    plan = _two_tables(monkeypatch, version="3.1.0+git45b5bd9")
    assert plan is not None, _refused()
    record = plan.report()
    assert record["composition"]["tables_dispatched"] == ["cuda", "triton"]
    assert record["tables"]["triton"]["candidate"] is True
    assert record["certified"] is False


# --- Triton's policy only where a Triton plan is in the step ----------------------------


def _unreachable_triton_backend(monkeypatch):
    def unreachable():
        raise ImportError("triton.backends.backends['nvidia'] is not importable")

    monkeypatch.setattr(policy_module, "_cuda_backend_class", unreachable)


def test_a_step_with_no_triton_plan_does_not_govern_triton(monkeypatch):
    """An importable Triton, dropped by its probe, whose backend class is out of reach:
    the hand-CUDA plan launches no Triton kernel and is not refused for it (1.0 governed
    Triton on the hand-CUDA table whenever it was importable, and the strict install
    refused the plan, whole, to the array path)."""
    stub_triton(monkeypatch, probe=failing_triton_probe([(8, 6)]))
    _unreachable_triton_backend(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    plan = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid())
    assert plan is not None, _refused()
    gate = plan.report()["subnormal"]["gate"]
    assert gate["step_tables"] == ["cuda"]
    assert gate["governed_executors"] == ["host", "cupy"]
    assert "triton" in gate["not_installed_in_this_process"]
    assert policy_module.executor_report("triton") == {}


def test_a_step_with_a_triton_plan_governs_triton(monkeypatch):
    plan = _two_tables(monkeypatch)
    assert plan is not None, _refused()
    gate = plan.report()["subnormal"]["gate"]
    assert gate["step_tables"] == ["cuda", "triton"]
    assert gate["governed_executors"] == ["host", "cupy", "triton"]
    assert policy_module.executor_report("triton")["attained"] is True


def test_a_later_freeze_with_a_triton_plan_drives_triton_then(monkeypatch):
    """Freeze 1 composes the hand-CUDA table alone and installs keep on the host and
    CuPy; freeze 2 holds a Triton plan, finds keep installed by dispatch and Triton
    never driven, and drives it rather than refusing the plan as a split."""
    monkeypatch.setenv(fastpath.KERNEL_TABLE_SWITCH, "cuda")
    first = _two_tables(monkeypatch)
    assert first is not None, _refused()
    assert first.report()["subnormal"]["gate"]["governed_executors"] == ["host", "cupy"]
    assert policy_module.executor_report("triton") == {}
    epoch = policy_module.policy_epoch()
    monkeypatch.delenv(fastpath.KERNEL_TABLE_SWITCH)
    second = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                     _grid())
    assert second is not None, _refused()
    gate = second.report()["subnormal"]["gate"]
    assert gate["late_driven"] == ["triton"]
    assert gate["installed_by"] == ("dispatch, at an earlier configuration freeze in "
                                    "this process")
    assert policy_module.executor_report("triton")["attained"] is True
    assert policy_module.policy_epoch() == epoch, "the first plan keeps its licence"


def test_a_callers_partial_install_is_not_completed(monkeypatch):
    stub_triton(monkeypatch)
    # Where CuPy is real, a caller's keep install needs the private cache directory
    # first, as dispatch's own install does.
    policy_module.point_cupy_cache_at_keep_policy()
    policy_module.install_subnormal_policy("keep", executors=("host", "cupy"))
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    assert fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid()) is None
    assert "was never driven onto triton" in _refused()


# --- A CuPy cache directory that cannot serve heals ------------------------------------


def _plan_with_cache(monkeypatch, cache_dir):
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cache_dir))
    plan = _two_tables(monkeypatch)
    report = fastpath.last_dispatch_report()
    return plan, (report.get("subnormal") or {}).get("gate", {}).get("cupy_cache_dir")


def test_an_unwritable_cache_directory_heals_to_a_marked_sibling(monkeypatch, tmp_path):
    keep = tmp_path / f"mine-{policy_module.CUPY_CACHE_POLICY_TOKEN}"
    keep.mkdir()
    keep.chmod(0o500)
    try:
        plan, cache = _plan_with_cache(monkeypatch, tmp_path / "mine")
    finally:
        keep.chmod(0o700)
    assert plan is not None, _refused()
    assert cache["healed"]["from"] == str(keep)
    assert "could not be created or written" in cache["healed"]["problem"]
    assert cache["after"] == f"{keep}-2" == os.environ["CUPY_CACHE_DIR"]
    assert cache["before"] == str(tmp_path / "mine")
    assert cache["provenance"]["marked"] is True
    assert os.path.exists(policy_module.keep_policy_cache_marker(cache["after"]))


def test_a_poisoned_sibling_is_skipped_and_a_marked_one_reused(monkeypatch, tmp_path):
    keep = tmp_path / f"c-{policy_module.CUPY_CACHE_POLICY_TOKEN}"
    for directory in (keep, tmp_path / f"{keep.name}-2"):
        directory.mkdir()
        (directory / "foreign.cubin").write_bytes(b"not ours")
    plan, cache = _plan_with_cache(monkeypatch, keep)
    assert plan is not None, _refused()
    assert cache["after"] == f"{keep}-3"
    assert [entry["directory"] for entry in cache["healed"]["tried"]] == [f"{keep}-2"]
    # A later process finds the same failure and reuses the sibling it minted.
    (tmp_path / f"{keep.name}-3" / "ours.cubin").write_bytes(b"ours")
    policy_module._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    plan, again = _plan_with_cache(monkeypatch, keep)
    assert plan is not None, _refused()
    assert again["after"] == f"{keep}-3"


def test_a_cache_parent_that_cannot_be_written_heals_to_a_private_temporary(
        monkeypatch, tmp_path):
    parent = tmp_path / "readonly"
    parent.mkdir()
    keep = parent / f"c-{policy_module.CUPY_CACHE_POLICY_TOKEN}"
    keep.mkdir()
    (keep / "foreign.cubin").write_bytes(b"not ours")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", None)
    parent.chmod(0o500)
    try:
        plan, cache = _plan_with_cache(monkeypatch, keep)
    finally:
        parent.chmod(0o700)
    assert plan is not None, _refused()
    assert cache["healed"]["kind"].startswith("a private temporary directory")
    assert cache["after"].startswith(str(tmp_path / "tmp" / "meep_gpu_cupy_cache-"))
    assert cache["after"].endswith(f"-{policy_module.CUPY_CACHE_POLICY_TOKEN}")
    assert len(cache["healed"]["tried"]) == fastpath.CUPY_CACHE_SIBLINGS


def test_the_plan_is_refused_only_when_no_directory_can_be_made(monkeypatch, tmp_path):
    keep = tmp_path / f"c-{policy_module.CUPY_CACHE_POLICY_TOKEN}"
    keep.mkdir()
    (keep / "foreign.cubin").write_bytes(b"not ours")
    import tempfile

    def no_temporary(**kwargs):
        raise OSError("no temporary directory")

    monkeypatch.setattr(tempfile, "mkdtemp", no_temporary)
    monkeypatch.setattr(fastpath, "CUPY_CACHE_SIBLINGS", 0)
    plan, cache = _plan_with_cache(monkeypatch, keep)
    assert plan is None
    reason = _refused()
    assert "is of unknown provenance" in reason, reason
    assert "no fresh directory could be made either (tempfile.mkdtemp(): " in reason, reason
    assert os.environ["CUPY_CACHE_DIR"] == str(keep), "left as the caller had it"


def test_a_candidate_that_cannot_be_marked_is_not_taken(monkeypatch, tmp_path):
    keep = str(tmp_path / f"c-{policy_module.CUPY_CACHE_POLICY_TOKEN}")
    calls = []

    def unmarked(cupy=None):
        calls.append(os.environ["CUPY_CACHE_DIR"])
        return {"after": os.environ["CUPY_CACHE_DIR"], "writable": True,
                "provenance": {"reasons": [], "marked": False, "marker_error": "EROFS"}}

    stub = types.SimpleNamespace(point_cupy_cache_at_keep_policy=unmarked,
                                 CUPY_CACHE_POLICY_TOKEN=policy_module.CUPY_CACHE_POLICY_TOKEN)
    monkeypatch.setattr(fastpath, "CUPY_CACHE_SIBLINGS", 1)
    import tempfile
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "t"))
    cache = fastpath._heal_cupy_cache_dir(  # noqa: SLF001
        stub, None, {"before": None, "after": keep}, "a stand-in problem")
    assert "to" not in cache["healed"], cache
    assert [entry["directory"] for entry in cache["healed"]["tried"]] == [
        f"{keep}-2", str(tmp_path / "t")]
    assert all("could not be marked" in entry["problem"]
               for entry in cache["healed"]["tried"])


# --- A caller-installed uniform flush runs uncertified by default -----------------------


def test_a_caller_flush_without_evidence_runs_uncertified_by_default(monkeypatch, capsys):
    from meep_gpu.test_dispatch_flush_licence import install_caller_flush

    stub_triton(monkeypatch)
    block_cuda(monkeypatch)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    monkeypatch.setattr(fastpath, "policy_evidence_reasons",
                        lambda arms, table, capability, policy: [
                            "PML: no flush run on 8.6"])
    install_caller_flush()
    plan = fastpath.plan_fast_path(object(), object(), cupy_like_grid_with_device())
    assert plan is not None, _refused()
    record = plan.report()
    gate = record["subnormal"]["gate"]
    assert gate["required_policy"] == "flush"
    assert gate["policy_evidence"]["certified"] is False
    assert record["certified"] is False
    (entry,) = record["uncertified"]["served"]
    assert (entry["what"], entry["read"], entry["certified"]) == (
        "float32 subnormal policy", "flush", ["keep"])
    assert entry["missing_evidence"] == ["PML: no flush run on 8.6"]
    (note,) = _notes(capsys.readouterr().err)
    assert "float32 subnormal policy flush (certified: keep)" in note, note
    assert "which not every dispatched arm was certified under" in note, note


# --- The certified scope compares the stack by name, per compute capability ------------


AXIS_VALUES = [("cupy", "13.6.0"), ("nvrtc", "12.8"), ("cuda_runtime", "12080"),
               ("cuda_driver", "12090"), ("host", "Linux aarch64"), ("numpy", "2.3"),
               ("meep", "1.34.0 single"), ("meep", "1.33.0 double")]

#: The driver a stand-in second capability is certified on (a newer cluster's), and one
#: neither capability is certified on.
DRIVER_X, DRIVER_Y = "12080", "12090"


def _certified_values(capability, axis):
    return ", ".join(fastpath.certified_nvidia_stack(capability)[axis])


def _planned_on(monkeypatch, capability=(8, 6), **stack):
    """``_two_tables`` on a stand-in host whose stack reads the 8.6 values with ``stack``
    read in their place (``_two_tables`` itself installs the 8.6 values)."""
    stub_triton(monkeypatch, stack=False)
    stand_in_stack(monkeypatch, **stack)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    return fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid(capability))


def certify_9_0(monkeypatch, driver=DRIVER_X, stack=True):
    """Both NVIDIA ledgers with 9.0 certified on the same bytes, and (``stack``) its
    certified stack recorded on ``driver``."""
    triton = ledger_also_certified_on("triton", "9.0", stack=False)
    if stack:
        record_certified_stack(triton, "9.0", cuda_driver=driver)
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {
        "triton_kernels": triton,
        "cuda_kernels": ledger_also_certified_on(fastpath.CUDA_TABLE, "9.0")})


@pytest.mark.parametrize("axis,value", AXIS_VALUES)
def test_certified_mode_refuses_each_stack_axis_by_name(monkeypatch, axis, value):
    monkeypatch.setenv(SWITCH, "certified")
    stub_triton(monkeypatch, stack=False)
    stand_in_stack(monkeypatch, **{axis: value})
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    assert fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid()) is None
    reason = _refused()
    what = fastpath.NVIDIA_STACK_NAMES[axis]
    certified = _certified_values("8.6", axis)
    assert reason == ("the stack the NVIDIA kernels would run beside is not the one they "
                      "were certified on for compute capability 8.6: "
                      f"{what} {value} is not the certified {certified}"
                      + fastpath.CERTIFIED_SCOPE_TAIL), reason


@pytest.mark.parametrize("axis,value", AXIS_VALUES)
def test_an_uncertified_stack_runs_by_default_and_says_so(monkeypatch, capsys, axis, value):
    stub_triton(monkeypatch, stack=False)
    stand_in_stack(monkeypatch, **{axis: value})
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    plan = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid())
    assert plan is not None, _refused()
    record = plan.report()
    assert record["certified"] is False
    what = fastpath.NVIDIA_STACK_NAMES[axis]
    served = [entry for entry in record["uncertified"]["served"] if entry["what"] == what]
    assert {entry["table"] for entry in served} == {"triton", "cuda"}
    assert all(entry["because"] == fastpath.NVIDIA_STACK_BECAUSE for entry in served)
    assert record["environment"]["stack"]["axes"][axis]["certified"] is False
    (note,) = _notes(capsys.readouterr().err)
    certified = _certified_values("8.6", axis)
    assert f"{what} {value} (certified: {certified})" in note, note
    assert note.count(fastpath.NVIDIA_STACK_BECAUSE) == 1, note


def test_the_default_note_on_an_8_6_card_off_its_driver_keeps_its_wording(monkeypatch,
                                                                           capsys):
    """Typed out, not read off the constant: recording the stack per compute capability
    changes what ``certified`` compares with, and must not change the line a default run
    on 8.6 beside another driver prints, nor the record's ``because``. The line is the one
    the install guide quotes."""
    because = ("the NVIDIA kernels run by default beside a toolchain or host that is not "
               "the one they were certified on, and MEEP_GPU_HARDWARE=certified would "
               "refuse it")
    plan = _planned_on(monkeypatch, cuda_driver=DRIVER_Y)
    assert plan is not None, _refused()
    record = plan.report()
    assert record["certified"] is False
    assert {entry["because"] for entry in record["uncertified"]["served"]} == {because}
    (note,) = _notes(capsys.readouterr().err)
    assert note == ("meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: CUDA driver "
                    "12090 (certified: 12030, 13040). They were dispatched because " + because
                    + "; compare the results with a prefer_gpu=False run of the same "
                    "simulation before relying on them"), note


def test_an_unreadable_stack_axis_is_unknown_by_default_and_refused_when_certified(
        monkeypatch):
    stub_triton(monkeypatch, stack=False)
    stand_in_stack(monkeypatch, nvrtc=None)
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    plan = fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid())
    assert plan is not None, _refused()
    assert plan.report()["certified"] is None
    assert plan.report()["uncertified"]["served"] == []
    monkeypatch.setenv(SWITCH, "certified")
    assert fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid()) is None
    assert "the NVRTC could not be read, so it cannot be judged" in _refused()


def test_the_certified_stack_reads_certified_everywhere(monkeypatch):
    for value in (None, "certified"):
        if value:
            monkeypatch.setenv(SWITCH, value)
        plan = _two_tables(monkeypatch)
        assert plan is not None, (value, _refused())
        assert plan.report()["certified"] is True, value
        assert plan.report()["environment"]["stack"]["certified"] is True, value
        assert plan.report()["environment"]["stack"]["capability"] == "8.6", value


def test_8_6_keeps_its_own_stack_when_another_capability_is_certified_on_another_driver(
        monkeypatch):
    """8.6 on its own driver is accepted, and 8.6 on the driver 9.0 was certified on is
    refused by name, naming 8.6's own drivers (12030 and 13040, as the round's bind recorded
    them): each device is judged on its own capability's stack, never on the stacks of every
    certified capability together."""
    certify_9_0(monkeypatch, DRIVER_X)
    monkeypatch.setenv(SWITCH, "certified")
    assert fastpath.certified_nvidia_stack("8.6")["cuda_driver"] == ("12030", "13040")
    assert fastpath.certified_nvidia_stack("9.0")["cuda_driver"] == (DRIVER_X,)
    plan = _planned_on(monkeypatch)
    assert plan is not None, _refused()
    assert plan.report()["certified"] is True
    assert _planned_on(monkeypatch, cuda_driver=DRIVER_X) is None
    assert _refused() == ("the stack the NVIDIA kernels would run beside is not the one "
                          "they were certified on for compute capability 8.6: CUDA driver "
                          f"{DRIVER_X} is not the certified 12030, 13040"
                          + fastpath.CERTIFIED_SCOPE_TAIL), _refused()


def test_a_capability_recorded_on_its_own_driver_is_accepted_there_and_refused_elsewhere(
        monkeypatch, capsys):
    """9.0 certified on driver X: accepted on X under ``certified``; refused by name on Y
    and on 8.6's driver; run uncertified, with the NOTE, on Y by default."""
    certify_9_0(monkeypatch, DRIVER_X)
    monkeypatch.setenv(SWITCH, "certified")
    plan = _planned_on(monkeypatch, (9, 0), cuda_driver=DRIVER_X)
    assert plan is not None, _refused()
    record = plan.report()
    assert record["certified"] is True
    stack = record["environment"]["stack"]
    assert stack["capability"] == "9.0" and stack["certified"] is True
    assert stack["axes"]["cuda_driver"]["certified_values"] == [DRIVER_X]
    for other in (DRIVER_Y, "12030"):
        assert _planned_on(monkeypatch, (9, 0), cuda_driver=other) is None, other
        assert _refused() == ("the stack the NVIDIA kernels would run beside is not the one "
                              "they were certified on for compute capability 9.0: CUDA "
                              f"driver {other} is not the certified {DRIVER_X}"
                              + fastpath.CERTIFIED_SCOPE_TAIL), _refused()
    capsys.readouterr()
    monkeypatch.delenv(SWITCH)
    plan = _planned_on(monkeypatch, (9, 0), cuda_driver=DRIVER_Y)
    assert plan is not None, _refused()
    assert plan.report()["certified"] is False
    (note,) = _notes(capsys.readouterr().err)
    assert f"CUDA driver {DRIVER_Y} (certified: {DRIVER_X})" in note, note


def certify_8_6_on_drivers(monkeypatch, drivers):
    """The shipped ledgers with 8.6's certified stack listing ``drivers``, every other axis as
    shipped: the shape the bind writes when lanes on a second driver read the recorded value
    on every other axis (``record_nvidia_stack.py``)."""
    triton = copy.deepcopy(dict(fastpath._fingerprints("triton")))
    record = triton[fastpath.CERTIFIED_STACK_ENTRY][fastpath.STACK_BY_CAPABILITY]["8.6"]
    record["stack"]["cuda_driver"] = list(drivers)
    record["stamps_by_cuda_driver"] = {driver: [f"r_cc86_c{k}"]
                                       for k, driver in enumerate(drivers)}
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {
        "triton_kernels": triton,
        "cuda_kernels": copy.deepcopy(dict(fastpath._fingerprints(fastpath.CUDA_TABLE)))})


def test_a_capability_certified_on_two_drivers_is_certified_on_each_and_refuses_a_third(
        monkeypatch, capsys):
    """8.6 recorded on drivers 12030 and 13040: under ``certified`` a host on either plans
    certified, a host on a driver no certified run used is refused by name with both drivers
    named, and a listed driver beside another value of another axis is refused on that axis;
    by default the third driver runs uncertified, its NOTE naming both. Dispatch reads the
    list as it reads one value: nothing in ``fastpath`` changed for it."""
    certify_8_6_on_drivers(monkeypatch, ("12030", "13040"))
    monkeypatch.setenv(SWITCH, "certified")
    assert fastpath.certified_nvidia_stack("8.6")["cuda_driver"] == ("12030", "13040")
    for driver in ("12030", "13040"):
        plan = _planned_on(monkeypatch, cuda_driver=driver)
        assert plan is not None, (driver, _refused())
        record = plan.report()
        assert record["certified"] is True, driver
        axis = record["environment"]["stack"]["axes"]["cuda_driver"]
        assert (axis["read"], axis["certified"]) == (driver, True)
        assert axis["certified_values"] == ["12030", "13040"]
    assert _planned_on(monkeypatch, cuda_driver=DRIVER_Y) is None
    assert _refused() == ("the stack the NVIDIA kernels would run beside is not the one they "
                          "were certified on for compute capability 8.6: CUDA driver "
                          f"{DRIVER_Y} is not the certified 12030, 13040"
                          + fastpath.CERTIFIED_SCOPE_TAIL), _refused()
    assert _planned_on(monkeypatch, cuda_driver="13040", numpy="2.3") is None
    assert _refused() == ("the stack the NVIDIA kernels would run beside is not the one they "
                          "were certified on for compute capability 8.6: NumPy 2.3 is not "
                          "the certified 2.2" + fastpath.CERTIFIED_SCOPE_TAIL), _refused()
    capsys.readouterr()
    monkeypatch.delenv(SWITCH)
    plan = _planned_on(monkeypatch, cuda_driver=DRIVER_Y)
    assert plan is not None, _refused()
    assert plan.report()["certified"] is False
    (note,) = _notes(capsys.readouterr().err)
    assert f"CUDA driver {DRIVER_Y} (certified: 12030, 13040)" in note, note


def test_a_capability_with_no_recorded_stack_is_refused_by_name_when_certified(monkeypatch):
    """Records that admit 9.0 and no stack recorded for it: ``certified`` refuses by name
    rather than judging 9.0 on another capability's stack; the default leaves it unjudged."""
    certify_9_0(monkeypatch, stack=False)
    assert fastpath.certified_nvidia_stack("9.0") is None
    monkeypatch.setenv(SWITCH, "certified")
    assert _two_tables(monkeypatch, capability=(9, 0)) is None
    assert _refused() == ("the stack the NVIDIA kernels would run beside cannot be judged: "
                          "no certified stack is recorded for compute capability 9.0"
                          + fastpath.CERTIFIED_SCOPE_TAIL), _refused()
    monkeypatch.delenv(SWITCH)
    plan = _two_tables(monkeypatch, capability=(9, 0))
    assert plan is not None, _refused()
    record = plan.report()
    assert record["certified"] is None
    assert record["uncertified"]["served"] == []
    assert record["environment"]["stack"]["no_certified_stack"] == (
        "no certified stack is recorded for compute capability 9.0")


def test_the_uncertified_default_path_is_unaffected(monkeypatch, capsys):
    """A supported device no certification ran on (8.7 on the shipped records) runs by
    default as it did: uncertified on the capability alone, with one NOTE that names the
    capability and no stack axis, its stack unjudged because 8.7 has none."""
    # 8.7 retires 9.0 as the uncertified example, ahead of 9.0's certification: no round has
    # run on 8.7, so the example outlives the next certification.
    plan = _two_tables(monkeypatch, capability=(8, 7))
    assert plan is not None, _refused()
    record = plan.report()
    assert record["certified"] is False
    assert record["composition"]["tables_dispatched"] == ["cuda", "triton"]
    assert {entry["what"] for entry in record["uncertified"]["served"]} == {
        "GPU compute capability"}
    stack = record["environment"]["stack"]
    assert stack["certified"] is None and stack["capability"] == "8.7"
    assert stack["no_certified_stack"] == (
        "no certified stack is recorded for compute capability 8.7")
    (note,) = _notes(capsys.readouterr().err)
    # Retires "(certified: 8.6)": the NOTE lists the admitted capabilities, 8.9 among them.
    assert "GPU compute capability 8.7 (certified: 8.6, 8.9)" in note, note
    assert note.count("(certified: ") == 1, note
    assert fastpath.NVIDIA_STACK_BECAUSE not in note, note


def test_a_malformed_stack_record_certifies_nothing(monkeypatch):
    """A capability record missing an axis is not read in part: 8.6 then has no certified
    stack, and ``certified`` refuses by name."""
    identity = certified_stack_identity()
    triton = copy.deepcopy(dict(fastpath._fingerprints("triton")))  # noqa: SLF001
    del triton[fastpath.CERTIFIED_STACK_ENTRY][fastpath.STACK_BY_CAPABILITY]["8.6"][
        "stack"]["cuda_driver"]
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {
        "triton_kernels": triton,
        "cuda_kernels": dict(fastpath._fingerprints(fastpath.CUDA_TABLE))})  # noqa: SLF001
    assert fastpath.certified_nvidia_stack("8.6") is None
    monkeypatch.setenv(SWITCH, "certified")
    stub_triton(monkeypatch, stack=False)
    monkeypatch.setattr(fastpath, "_nvidia_stack_identity", lambda xp: dict(identity))
    install_composer(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                            {"step_B": "PML"}))
    install_cuda_composer(monkeypatch, cuda_step_plan())
    assert fastpath.plan_fast_path(certified_envelope_fields(), certified_envelope_pml(),
                                   _grid()) is None
    assert _refused().startswith("the stack the NVIDIA kernels would run beside cannot be "
                                 "judged: no certified stack is recorded for compute "
                                 "capability 8.6"), _refused()


def test_an_unreadable_stack_record_is_named(monkeypatch):
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {
        "triton_kernels": {"_unreadable": "a stand-in read error"},
        "cuda_kernels": dict(fastpath._fingerprints(fastpath.CUDA_TABLE))})  # noqa: SLF001
    assert fastpath.certified_nvidia_stacks() is None
    block = fastpath.nvidia_stack_block(cupy_like_grid_with_device().xp, "8.6")
    assert block["certified"] is None
    assert block["no_certified_stack"].startswith("the certified NVIDIA stack record")
    block = fastpath.nvidia_stack_block(cupy_like_grid_with_device().xp, None)
    assert block["no_certified_stack"].startswith("the GPU's compute capability could not")


def test_the_stack_is_read_off_the_process_and_the_device():
    xp = cupy_like_grid_with_device(nvrtc=(12, 4)).xp
    identity = fastpath._nvidia_stack_identity(xp)  # noqa: SLF001
    assert identity["cupy"] == "13.5.1"
    assert identity["nvrtc"] == "12.4"
    assert (identity["cuda_runtime"], identity["cuda_driver"]) == ("11080", "12030")
    assert identity["host"] == f"{platform.system()} {platform.machine()}"
    assert identity["numpy"] == ".".join(np.__version__.split(".")[:2])
    assert fastpath._nvidia_stack_identity(cupy_like_grid().xp)["nvrtc"] is None  # noqa: SLF001


def test_the_meep_build_is_judged_only_when_meep_is_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "meep", raising=False)
    block = fastpath.nvidia_stack_block(cupy_like_grid_with_device().xp, "8.6")
    assert "meep" not in block["axes"]
    assert block["meep"].startswith("not imported in this process")
    meep = types.ModuleType("meep")
    meep.__version__ = "1.34.0"
    meep.is_single_precision = lambda: True
    monkeypatch.setitem(sys.modules, "meep", meep)
    block = fastpath.nvidia_stack_block(cupy_like_grid_with_device().xp, "8.6")
    assert block["axes"]["meep"]["read"] == "1.34.0 single"
    assert block["axes"]["meep"]["certified"] is False


def test_the_recorded_stacks_are_in_shape_and_8_6_keeps_its_stack():
    """Every capability record the ledger holds is read whole (none is dropped as
    malformed), and 8.6's certified stack is the one it was certified on: drivers 12030 (the
    RTX A6000's chunks of the 1.0.1 round) and 13040 (the RTX 3090's)."""
    entry = fastpath._fingerprints("triton")[fastpath.CERTIFIED_STACK_ENTRY]  # noqa: SLF001
    recorded = entry[fastpath.STACK_BY_CAPABILITY]
    stacks = fastpath.certified_nvidia_stacks()
    assert sorted(stacks) == sorted(recorded) and "8.6" in stacks
    for capability, record in recorded.items():
        assert list(record["stack"]) == sorted(fastpath.NVIDIA_STACK_AXES), capability
        assert record["read_from"], capability
    assert fastpath.certified_nvidia_stack("8.6") == {
        "cupy": ("13.5.1",), "nvrtc": ("11.8",), "cuda_runtime": ("11080",),
        "cuda_driver": ("12030", "13040"), "host": ("Linux x86_64",), "numpy": ("2.2",),
        "meep": ("1.35.0 single",)}


def test_the_certified_meep_build_is_declared_once():
    """Every capability's MEEP axis names a certified build on its host, and the bulk read's
    certified list is the same set.

    ``fastpath.CERTIFIED_MEEP_BUILDS`` names the certified reference builds; each recorded
    stack's MEEP axis is a build of that set on that stack's host, and the bulk read
    certifies exactly those builds.
    """
    from meep_gpu import from_meep

    assert from_meep.CERTIFIED_CHI1INV_BUILDS == fastpath.CERTIFIED_MEEP_BUILDS
    for capability, stack in fastpath.certified_nvidia_stacks().items():
        for meep in stack["meep"]:
            for host in stack["host"]:
                assert fastpath.meep_build_is_certified(meep, host), (capability, meep, host)
    assert fastpath.certified_nvidia_stack("8.6")["meep"] == ("1.35.0 single",)
    assert not fastpath.meep_build_is_certified("1.35.0 double", "Linux x86_64")
    assert not fastpath.meep_build_is_certified("1.35.0 single", "Linux aarch64")


def test_the_certified_stack_is_held_to_the_records():
    """Every value of 8.6's certified stack is one a shipped 8.6 record holds."""
    import json
    import pathlib

    here = pathlib.Path(fastpath.__file__).parent
    triton = json.loads((here / "triton_kernels" / "fingerprints.json").read_text())
    cuda = (here / "cuda_kernels" / "certification.json").read_text()
    stack = fastpath.certified_nvidia_stack("8.6")
    text = json.dumps({key: entry for key, entry in triton.items()
                       if key != fastpath.CERTIFIED_STACK_ENTRY})
    live = {run.get("cupy_version") for entry in triton.values() if isinstance(entry, dict)
            for run in (entry.get("runs") or {}).values() if isinstance(run, dict)}
    assert set(stack["cupy"]) <= live - {None}, live
    for axis, key in (("cuda_runtime", "cuda_runtime"), ("cuda_driver", "cuda_driver")):
        for value in stack[axis]:
            assert f'"{key}": {value}' in text, (axis, value)
    for value in stack["numpy"]:
        assert f'"numpy": "{value}.' in text, value
    for value in stack["meep"]:
        assert f'"meep": "{value.split()[0]}"' in text, value
    for value in stack["nvrtc"]:
        assert f'"nvrtc_version": "{value}"' in cuda, value
    assert certified_stack_identity()["host"] in stack["host"]
