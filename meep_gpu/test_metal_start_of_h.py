"""MEEP's start of H on a run's first ``update_H``, carried by the Metal composition.

MEEP allocates no H at first and its first ``update_eh(H_stuff)`` starts H and ``f_w`` as
copies of B on the chunks that carry H's own layer (update_eh.cpp:165-177); the array path
does the same in ``stepping.separate_h_from_b``. No Metal kernel starts H, so the composer
puts :class:`meep_gpu.metal_kernels.launch.StartOfHPlan` around whatever serves the slot.

What this file pins:

1. **The premise the composition used to rest on is false**, on the array path alone: from
   an all +0 state with no source, the first ``step_B`` lands B on -0 wherever ``+0 * kms``
   with a negative ``kms`` forms it, and on the dsigw cells whose own ``kms`` is negative
   the start keeps that -0 in H where the forms alone store +0. Sign of zero only, in H
   only. The two NVIDIA tables never relied on that premise -- the hand-CUDA launchers run
   the start on the array path, the Triton kernels take it as ``start_h`` -- and their own
   host tests pin that they do (``cuda_kernels/test_fused_pair_chunk_forms.py``,
   ``test_triton_chunk_bounds.py``).
2. **The wrapper**, with stub plans and no device: which half of the start runs where, that
   it runs once, and that it is transparent to every reader of the slot (declarations,
   launch counters, pair identity, the residency bracket).
3. **The composer** installs it around every shape of occupant and refuses, by name, the
   one ordering it cannot carry (an integrated magnetic source in the layer).
4. **On the device** (an Apple GPU: ``mps_device`` in ``tools/ci/declared_resources.txt``):
   a complete first step through the composed plans equals the array path word for word,
   for the certified singles and for the fused
   magnetic B/H pair, and the same composition with the start stripped differs on the -0
   words -- the planted revert is caught.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from meep_gpu import fastpath, stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import launch, subnormal
from meep_gpu.pml import PML
from meep_gpu.triton_kernels.launch import NoopPlan

NEGATIVE_ZERO = 0x80000000

_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
          "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
          "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _words(array):
    return np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)


def _first_step_case(dimensions: int = 3):
    """A run frozen BEFORE its first step: all +0, no source, a strong periodic layer.

    Resolution 8 and a 4-cell layer on every periodic axis put a negative ``kms`` on the
    outermost cells of both lattices -- the layer of the corpus row this was found on
    (``ex.differential_cross_section``: 48 cells, 4-cell layers, Bloch k = 0).
    """
    cell = (2.0, 2.0, 2.0) if dimensions == 3 else (2.0, 2.0, 0.0)
    grid = Grid(resolution=8.0, cell_size=cell, dimensions=dimensions, courant=0.5,
                k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    thickness = tuple((4, 4) if grid.shape[axis] > 8 else (0, 0) for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    assert fields.h_separated is False
    return grid, fields, pml


def _copy(fields):
    return {name: np.array(getattr(fields, name), copy=True) for name in _STATE
            if getattr(fields, name, None) is not None}


# ---------------------------------------------------------------------------
# 1. The premise, on the array path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dimensions", [3, 2])
def test_a_zero_state_lands_b_on_negative_zero_and_the_start_keeps_it_in_h(dimensions):
    grid, started, pml = _first_step_case(dimensions)
    _, alone, _ = _first_step_case(dimensions)
    negative = {axis: int(np.count_nonzero(np.asarray(getattr(pml, f"kms_{axis}")) < 0))
                for axis in "xyz"}
    assert sum(negative.values()) > 0, f"VACUOUS: no negative kms in this layer {negative}"

    stepping.step_B(started, pml)
    stepping.step_B(alone, pml)
    negative_b = sum(int(np.count_nonzero(_words(getattr(started, c)) == NEGATIVE_ZERO))
                     for c in ("Bx", "By", "Bz"))
    assert negative_b > 0, "the first step_B left every B word at +0 on this layer"
    for name in ("Bx", "By", "Bz"):
        assert np.array_equal(_words(getattr(started, name)), _words(getattr(alone, name)))
        assert not np.any(np.asarray(getattr(started, name))), "B moved off zero"

    stepping.update_H(started, pml)            # the array path: the start, then the forms
    alone.h_separated = True
    stepping.update_H(alone, pml)              # the forms alone, as the kernels step them

    differing = {}
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        left, right = _words(getattr(started, name)), _words(getattr(alone, name))
        changed = left != right
        differing[name] = int(np.count_nonzero(changed))
        if differing[name]:
            # sign of zero only: the start's word is -0, the forms alone store +0
            assert set(np.unique(left[changed])) == {NEGATIVE_ZERO}, name
            assert set(np.unique(right[changed])) == {0}, name
    assert sum(differing.values()) > 0, f"the start changed no word here: {differing}"
    assert not any(differing[f"f_w_{c}"] for c in ("Hx", "Hy", "Hz")), differing


def _forms_alone(fields, pml):
    """What every constitutive kernel steps when nothing starts H: the array path's forms."""
    started = fields.h_separated
    fields.h_separated = True
    try:
        stepping.update_H(fields, pml)
    finally:
        fields.h_separated = started


def test_the_hand_cuda_launcher_starts_h_before_its_kernel_on_this_state(monkeypatch):
    """The NVIDIA twin, read and run on the host: ``arms._RealPmlConstitutiveLauncher``
    runs ``separate_h_from_b`` on the array path before the kernel on the run's first
    ``update_H``, unconditionally -- it rests on no +0 premise -- so on the state the
    Metal composition got wrong its words are the array path's. The kernel is stood in
    for by the forms alone, which is what the certified constitutive kernel computes."""
    import meep_gpu.cuda_kernels as cuda_package
    from meep_gpu.cuda_kernels import arms as cuda_arms

    _, fields, pml = _first_step_case()
    _, reference, _ = _first_step_case()
    stub = types.SimpleNamespace(
        _CONSTITUTIVE_THREADS=128,
        _update_H_fused_pml_real=lambda fields, **_launch: _forms_alone(fields, pml))
    monkeypatch.setattr(cuda_package, "constitutive_kernels", stub, raising=False)
    monkeypatch.setitem(__import__("sys").modules,
                        "meep_gpu.cuda_kernels.constitutive_kernels", stub)
    for state in (fields, reference):
        stepping.step_B(state, pml)
    _, forms_only, _ = _first_step_case()
    stepping.step_B(forms_only, pml)
    _forms_alone(forms_only, pml)

    report = cuda_arms.launch_real_pml_constitutive(fields, "update_H", {"layer": pml})
    stepping.update_H(reference, pml)
    assert report["h_started"] is True and fields.h_separated is True
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        assert np.array_equal(_words(getattr(fields, name)), _words(getattr(reference, name))), name
    assert any(not np.array_equal(_words(getattr(forms_only, c)), _words(getattr(reference, c)))
               for c in ("Hx", "Hy", "Hz")), "VACUOUS: the start changes nothing here"


def test_the_triton_kernels_carry_the_start_in_their_own_statements():
    """The other NVIDIA twin, read: the Triton H-side constitutive steps MEEP's start
    in-kernel (``prev`` and the field word both ``src`` while ``start_h`` is set), which
    is the array path's start followed by its forms, whatever B holds."""
    import pathlib

    # Read as text: the module imports Triton, which a host without it cannot load.
    source = (pathlib.Path(stepping.__file__).parent / "triton_kernels"
              / "kernels.py").read_text(encoding="utf-8")
    for c in "012":
        assert f"prev{c} = tl.where(start, src{c}, prev{c})" in source, c
        assert f"a{c} = tl.where(start, src{c}, a{c})" in source, c
    # ...and those statements, evaluated on the state the Metal composition got wrong,
    # are the array path's words.
    _, fields, pml = _first_step_case()
    stepping.step_B(fields, pml)
    _, reference, _ = _first_step_case()
    stepping.step_B(reference, pml)
    stepping.update_H(reference, pml)
    plan = fields.region_plan if hasattr(fields, "region_plan") else None
    assert plan is not None
    for component, source_name, axis in stepping.H_CONSTITUTIVE_TERMS:
        kps, kms = stepping._constitutive_coefficients(pml, axis, half_integer=False)
        shape = fields.grid.shape
        index = "xyz".index(axis)
        view = [1, 1, 1]
        view[index] = shape[index]
        kp = np.asarray(kps, dtype=np.float32).reshape(view)
        km = np.asarray(kms, dtype=np.float32).reshape(view)
        dw = np.broadcast_to(plan.mask("constitutive", component, np), shape)
        src = np.asarray(getattr(fields, source_name), dtype=np.float32)
        prev = np.where(dw, np.asarray(getattr(fields, "f_w_" + component)), np.float32(0))
        a = np.where(dw, np.asarray(getattr(fields, component)), np.float32(0))
        prev = np.where(True, src, prev)                    # start: prev := src
        a = np.where(True, src, a)                          # start: f := src
        word = np.where(dw, a + ((kp * src) - (km * prev)), src)
        assert np.array_equal(_words(word), _words(getattr(reference, component))), component


# ---------------------------------------------------------------------------
# 2. The wrapper, with stubs
# ---------------------------------------------------------------------------

class _Launch:
    """A stub launch with the counter a real plan keeps."""

    def __init__(self, log):
        self.log = log
        self.launches = 0

    def run(self, *args, **kwargs):
        self.log.append("launch")
        self.launches += 1


class _Bracket:
    def sync_in(self, names=None):
        pass

    def sync_out(self, names=None):
        pass


def _stub_start(monkeypatch, log):
    def start(fields, pml):
        log.append("start")
        fields.h_separated = True

    def restep(fields, pml):
        log.append("restep")
        fields.h_separated = True

    monkeypatch.setattr(launch, "start_h_from_b", start)
    monkeypatch.setattr(launch, "restep_update_h", restep)


def test_a_leading_slot_starts_h_before_its_launch_once_a_run(monkeypatch):
    log = []
    _stub_start(monkeypatch, log)
    fields = types.SimpleNamespace(h_separated=False)
    occupant = _Launch(log)
    plan = launch.StartOfHPlan(occupant, fields, None, after_launch=False)
    plan.run()
    plan.run()
    assert log == ["start", "launch", "launch"]
    assert plan.starts == 1


def test_an_absorbed_slot_re_steps_update_h_after_its_own_work_once_a_run(monkeypatch):
    log = []
    _stub_start(monkeypatch, log)
    fields = types.SimpleNamespace(h_separated=False)
    pair = _Launch(log)
    plan = launch.StartOfHPlan(NoopPlan("update_H", pair), fields, None, after_launch=True)
    plan.run()
    plan.run()
    assert log == ["restep"]
    assert plan.starts == 1


def test_a_state_started_already_takes_no_start(monkeypatch):
    log = []
    _stub_start(monkeypatch, log)
    fields = types.SimpleNamespace(h_separated=True)
    for after in (False, True):
        launch.StartOfHPlan(_Launch(log), fields, None, after_launch=after).run()
    assert log == ["launch", "launch"]


def test_the_wrapper_is_transparent_to_every_reader_of_the_slot():
    log = []
    fields = types.SimpleNamespace(h_separated=True)
    single = _Launch(log)
    wrapped = launch.StartOfHPlan(single, fields, None, after_launch=False)
    assert launch.declaring_plan(wrapped) is single
    assert fastpath._pair_identity(wrapped) == fastpath._pair_identity(single)
    wrapped.run()
    assert fastpath._plan_launches_of(wrapped) == 1

    # the residency bracket goes INSIDE: around a bare launch...
    bracketed = launch.synced(wrapped, _Bracket())
    assert bracketed is wrapped and isinstance(wrapped.inner, launch.SyncedPlan)
    assert wrapped.inner.inner is single
    # ...around a leading wrapper's own inner, leaving its host work first...
    pair = _Launch(log)
    leading = types.SimpleNamespace(inner=pair, absorbed_by=pair, run=lambda: None)
    around = launch.synced(launch.StartOfHPlan(leading, fields, None, False), _Bracket())
    assert around.inner is leading and isinstance(leading.inner, launch.SyncedPlan)
    assert launch.declaring_plan(around) is pair
    # ...and nowhere for an absorbed slot, whose launch is elsewhere.
    noop = NoopPlan("update_H", pair)
    after = launch.synced(launch.StartOfHPlan(noop, fields, None, True), _Bracket())
    assert after.inner is noop
    assert launch.declaring_plan(after) is pair
    assert fastpath._pair_identity(after) == fastpath._pair_identity(noop)
    assert fastpath._plan_launches_of(after) is None


# ---------------------------------------------------------------------------
# 3. The composer
# ---------------------------------------------------------------------------

def test_the_composer_wraps_every_shape_of_occupant():
    _, fields, pml = _first_step_case()
    log = []
    pair = _Launch(log)
    shapes = {
        "single": (_Launch(log), False),
        "leading": (types.SimpleNamespace(inner=pair, absorbed_by=pair, run=lambda: None),
                    False),
        "absorbed": (NoopPlan("update_H", pair), True),
    }
    for name, (occupant, after) in shapes.items():
        plans, reasons, selected = {"update_H": occupant}, {}, {"update_H": name}
        launch._carry_start_of_h(plans, reasons, selected, fields, pml, ())
        assert isinstance(plans["update_H"], launch.StartOfHPlan), name
        assert plans["update_H"].inner is occupant and plans["update_H"].after_launch is after
        assert not reasons and selected == {"update_H": name}


def test_the_composer_leaves_alone_what_needs_no_start():
    _, fields, pml = _first_step_case()
    cases = {
        "started": (fields, pml, _Launch([])),
        "no layer": (fields, PML(grid=fields.grid, thickness=((0, 0),) * 3), _Launch([])),
        "null": (fields, pml, types.SimpleNamespace(run=lambda: None,
                                                     performs_device_work=False)),
        "starts itself": (fields, pml, types.SimpleNamespace(run=lambda: None,
                                                              starts_h_from_b=True)),
    }
    for name, (state, layer, occupant) in cases.items():
        plans = {"update_H": occupant}
        state.h_separated = name == "started"
        launch._carry_start_of_h(plans, {}, {}, state, layer, ())
        assert plans["update_H"] is occupant, name
    fields.h_separated = False


def test_an_integrated_magnetic_source_in_the_layer_refuses_the_whole_product():
    _, fields, pml = _first_step_case()
    pair = _Launch([])
    plans = {"step_B": pair, "update_H": NoopPlan("update_H", pair),
             "step_D": _Launch([])}
    selected = {"step_B": "fused magnetic B/H pair", "update_H": "fused magnetic B/H pair",
                "step_D": "PML"}
    reasons = {}
    source = types.SimpleNamespace(field_type="B", is_integrated=True, component="Hx")
    launch._carry_start_of_h(plans, reasons, selected, fields, pml, (source,))
    assert set(plans) == {"step_D"} and set(selected) == {"step_D"}
    for slot in ("step_B", "update_H"):
        assert "integrated magnetic source(s) [0]" in reasons[slot][0], reasons
    # a non-integrated one is the deposit bracket's, and the start is carried
    plans = {"step_B": pair, "update_H": NoopPlan("update_H", pair)}
    plain = types.SimpleNamespace(field_type="B", is_integrated=False, component="Hx")
    launch._carry_start_of_h(plans, {}, {}, fields, pml, (plain,))
    assert isinstance(plans["update_H"], launch.StartOfHPlan)


# ---------------------------------------------------------------------------
# 4. On the device
# ---------------------------------------------------------------------------

# The device test is listed under ``mps_device`` in ``tools/ci/declared_resources.txt``:
# the root conftest skips it, by name, on a host whose PyTorch reports no Apple GPU.


def _composed_first_step(fuse_labels, strip_start=False):
    """One complete first step through the composed, bracketed plans; the host words after."""
    _, fields, pml = _first_step_case()
    residency = launch.Residency()
    plan = launch.plan_step(fields, pml, residency=residency, sources=(),
                            fuse=bool(fuse_labels), fuse_labels=fuse_labels or None)
    launch.wrap_for_residency(plan, residency)
    assert plan.replaces[:2] == ("step_B", "update_H"), (plan.replaces, plan.reasons)
    occupant = plan.plans["update_H"]
    assert isinstance(occupant, launch.StartOfHPlan), occupant
    if strip_start:
        plan.plans["update_H"] = occupant.inner
    for name in ("step_B", "update_H", "step_D", "update_E"):
        if name in plan.plans:
            plan.plans[name].run()
        else:
            {"step_D": stepping.step_D, "update_E": stepping.update_E}[name](fields, pml)
    residency.sync_out()
    return _copy(fields), occupant, plan


def _array_first_step():
    _, fields, pml = _first_step_case()
    stepping.step_B(fields, pml)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.update_E(fields, pml)
    return _copy(fields)


@pytest.mark.parametrize("fuse_labels", [(), ("fused magnetic B/H pair",)],
                         ids=["singles", "fused_magnetic_pair"])
def test_the_first_step_on_the_device_equals_the_array_path(monkeypatch, fuse_labels):
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    want = _array_first_step()
    got, occupant, plan = _composed_first_step(fuse_labels)
    if fuse_labels:
        assert plan.selected["update_H"] == "fused magnetic B/H pair", plan.selected
        assert occupant.after_launch is True
    else:
        assert plan.selected["update_H"] == "ordinary", plan.selected
        assert occupant.after_launch is False
    assert occupant.starts == 1
    negative = sum(int(np.count_nonzero(_words(want[c]) == NEGATIVE_ZERO))
                   for c in ("Hx", "Hy", "Hz"))
    assert negative > 0, "VACUOUS: the array path's first H holds no -0"
    differing = {n: int(np.count_nonzero(_words(got[n]) != _words(want[n]))) for n in want}
    assert not {n: c for n, c in differing.items() if c}, differing

    # THE PLANTED REVERT: the same composition with the start stripped differs, on H first
    # (and on what H feeds within the step).
    reverted, _, _ = _composed_first_step(fuse_labels, strip_start=True)
    caught = {n: int(np.count_nonzero(_words(reverted[n]) != _words(want[n]))) for n in want}
    assert sum(caught.values()) > 0, "the revert was not caught"
    assert set(n for n, c in caught.items() if c) <= {"Hx", "Hy", "Hz", "Dx", "Dy", "Dz",
                                                        "Ex", "Ey", "Ez", "f_w_Ex",
                                                        "f_w_Ey", "f_w_Ez"}, caught
    assert any(caught[c] for c in ("Hx", "Hy", "Hz")), caught
