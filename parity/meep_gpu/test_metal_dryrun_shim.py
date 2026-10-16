"""The cliff control's composition is the engine's: two clauses of the dry-run shim, off-device.

``gate_dispatch_metal_route.cliff_control`` composes its fused leg through
``probe_metal_dispatch_dryrun.install_shim`` rather than through the ladder, so the shim
has to reproduce what ``fastpath.FastPathPlan`` and ``metal_dispatch`` do for the leg
to measure the composition the engine runs. Two places it did not, both measured on
the 0.10 series (01..106) as disagreements with the array path under an EQUAL host
policy, which no flush-vs-keep reading can explain:

* THE CHUNK DIVISION. The ladder hands the composer the driver's own division
  (``launch.chunk_division(fields, pml, chunk_layout)``); the shim handed none, so
  ``plan_step`` bound the one-rank division. ``folded_complex_kz2d_3d`` differed at
  normal magnitudes on 16 arrays until it was passed.
* THE ABSENT COMPONENTS' CLEAR. ``FastPathPlan`` clears the absent electric
  components after every ``update_E`` consult (``fastpath.AbsentComponentClear``);
  the shim did not, so ``folded_offdiag_magnetic_2d``'s absent ``Ez`` (159 words) and
  ``folded_complex_2d``'s absent ``f_w_Ey`` (160 words) read -0 against +0.

Host-only: the composer, the clears and the residency are stood in for, so these run
with no device and no MEEP.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import probe_metal_dispatch_dryrun as dryrun  # noqa: E402


class _Plan:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def run(self, *_args, **_kwargs):
        self.log.append(("run", self.name))


def _patched_clears(monkeypatch):
    from meep_gpu import fastpath

    log = []
    sentinel = ("the absent clear",)
    monkeypatch.setattr(fastpath, "_absent_component_clears",
                        lambda fields: (sentinel, {"absent": ["Ez"]}))
    monkeypatch.setattr(fastpath, "_clear_absent_components",
                        lambda clears: log.append(("clear", clears)))
    return log, sentinel


def test_the_shim_clears_the_absent_components_after_every_update_e(monkeypatch):
    log, sentinel = _patched_clears(monkeypatch)
    fields = object()
    shim = dryrun.DryRunShim(fields, {"step_D": _Plan(log, "step_D"),
                                      "update_E": _Plan(log, "update_E")}, None)
    # Once before any step, as FastPathPlan applies it at the freeze.
    assert log == [("clear", sentinel)]
    assert shim.dispatch("step_D", fields)
    assert log[-1] == ("run", "step_D")
    assert shim.dispatch("update_E", fields)
    assert log[-2:] == [("run", "update_E"), ("clear", sentinel)]
    # A consult the shim declines runs nothing and clears nothing.
    before = len(log)
    assert not shim.dispatch("update_E", object())
    assert len(log) == before


def test_the_shim_builds_no_clear_without_an_update_e_plan(monkeypatch):
    log, _sentinel = _patched_clears(monkeypatch)
    fields = object()
    shim = dryrun.DryRunShim(fields, {"step_B": _Plan(log, "step_B")}, None)
    assert shim.absent_clears == ()
    assert shim.dispatch("step_B", fields)
    assert log == [("run", "step_B")]


def test_compose_hands_the_drivers_chunk_division_to_both_plans(monkeypatch):
    from meep_gpu.metal_kernels import device, launch

    seen = []

    class _StepPlan:
        plans = {}
        selected = {}
        reasons = {}

    def plan_step(fields, pml, **kwargs):
        seen.append(kwargs.get("chunk_layout", "MISSING"))
        return _StepPlan()

    class _Residency:
        names = ()

    monkeypatch.setattr(launch, "plan_step", plan_step)
    monkeypatch.setattr(launch, "live_sub_steps", lambda *args: ())
    monkeypatch.setattr(launch, "wrap_for_residency", lambda *args: None)
    monkeypatch.setattr(device, "Residency", _Residency)

    class _Driver:
        fields = object()
        pml = object()
        chunk_layout = object()

    driver = _Driver()
    dryrun.compose_metal_step(driver, (), fuse=True, fuse_labels=())
    assert seen == [driver.chunk_layout, driver.chunk_layout]
