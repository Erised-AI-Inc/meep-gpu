"""Host merge bar for the PyTorch/MPS Lorentz/Drude ``update_P`` family.

The device bytes live in ``parity/meep_gpu/gate_metal_ade_update_p.py``.  These
tests pin the parts a device comparison cannot establish by itself: the exact
left-associated recurrence, scalar/volume sigma specializations, all-or-nothing
slot coverage, live three-buffer rotation, residency declarations, and registry
composition.  No Torch import is required.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import ade_update_p as family
from meep_gpu.metal_kernels import arms, device, launch as metal_launch
from meep_gpu.metal_kernels.complex_dispersive_update_e import MAX_POLES
from meep_gpu.pml import PML


class HostResidency:
    """Torch-free mirror registry for builder and rotation tests."""

    def __init__(self) -> None:
        self._arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self._arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self._arrays[name] = host
        return host

    @property
    def names(self):
        return tuple(sorted(self._arrays))


def _kernel(volume: bool):
    def run(out, now, previous, *args):
        if volume:
            sigma, drive, c_now, c_prev, c_drive, n_elem = args
        else:
            drive, sigma, c_now, c_prev, c_drive, n_elem = args
        assert out.size == n_elem
        np.multiply(now, c_now, out=out)
        out += c_prev * previous
        out += c_drive * (sigma * drive)

    return run


def _functions():
    return {
        ("off", False, "float32"): _kernel(False),
        ("off", True, "float32"): _kernel(True),
        ("off", False, "complex64"): _kernel(False),
        ("off", True, "complex64"): _kernel(True),
    }


def build(*, active_pml: bool = False, complex_storage: bool = False,
          kind: str = LORENTZIAN, volume_sigma: bool = False,
          states: int = 1, counts=(1, 1, 1), seed: int = 20260818):
    grid = Grid(
        resolution=8.0,
        cell_size=(1.0, 0.75, 0.5),
        courant=0.35,
        boundaries="periodic",
        k_point=(0.15, -0.1, 0.05) if complex_storage else (0.0, 0.0, 0.0),
        xp=np,
    )
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)
    dtype = np.complex64 if complex_storage else np.float32
    rng = np.random.default_rng(seed)
    built = []
    for state_index in range(states):
        sigma = {}
        for axis, component in enumerate(("Ex", "Ey", "Ez")):
            if not counts[axis]:
                sigma[component] = 0.0
            elif volume_sigma:
                sigma[component] = np.linspace(
                    0.15 + 0.01 * state_index,
                    0.55 + 0.02 * axis,
                    np.prod(grid.shape),
                    dtype=np.float32,
                ).reshape(grid.shape)
            else:
                sigma[component] = 0.25 + 0.05 * axis + 0.01 * state_index
        state = PolarizationState(
            Susceptibility(1.0 + 0.05 * state_index, 0.1, kind),
            sigma,
            grid,
            dtype,
        )
        fields.polarizations.append(state)
        built.append(state)

    for component in ("Ex", "Ey", "Ez"):
        for prefix in (("f_w_",) if active_pml else ("",)):
            array = getattr(fields, prefix + component)
            array.real[...] = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
            if np.iscomplexobj(array):
                array.imag[...] = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
        # Make the wrong active-PML drive observably distinct.
        if active_pml:
            stored = getattr(fields, component)
            stored.real[...] = rng.uniform(0.6, 0.9, grid.shape).astype(np.float32)
            if np.iscomplexobj(stored):
                stored.imag[...] = rng.uniform(0.6, 0.9, grid.shape).astype(np.float32)

    for state in built:
        for component in state.driven():
            for table in (state.P, state.P_prev):
                array = table[component]
                array.real[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
                if np.iscomplexobj(array):
                    array.imag[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, pml, built


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def test_source_preserves_the_three_pass_grouping_for_both_storage_widths():
    for dtype, value_type in (("float32", "float"), ("complex64", "float2")):
        source = family.ade_source(dtype, sigma_is_volume=True)
        assert f"device {value_type}*       p_out" in source
        assert "if (idx >= n_elem) { return; }" in source
        assert "thread_position_in_grid" in source
    real = family.ade_source("float32", sigma_is_volume=True)
    assert "((p * c_now) + (c_prev * q)) + (c_drive * (s * w))" in real
    complex_source = family.ade_source("complex64", sigma_is_volume=True)
    assert "c_mul_field_left(p, c_now)" in complex_source
    assert "c_mul_coefficient_left(c_prev, q)" in complex_source
    assert "c_mul_coefficient_left(s, w)" in complex_source
    assert "p_out[idx] = (a + b) + d;" in complex_source


def test_scalar_and_volume_sigma_are_distinct_binding_products():
    scalar = family.ade_source("float32", sigma_is_volume=False)
    volume = family.ade_source("float32", sigma_is_volume=True)
    assert "device const float* sigma" not in scalar
    assert "constant float& sigma" in scalar
    assert "device const float* sigma" in volume
    assert "constant float& sigma" not in volume


@pytest.mark.parametrize("active_pml", (False, True))
@pytest.mark.parametrize("complex_storage", (False, True))
@pytest.mark.parametrize("kind", (LORENTZIAN, DRUDE))
@pytest.mark.parametrize("volume_sigma", (False, True))
def test_supported_product_is_admitted(
        backend_ready, active_pml, complex_storage, kind, volume_sigma):
    fields, pml, _states = build(
        active_pml=active_pml,
        complex_storage=complex_storage,
        kind=kind,
        volume_sigma=volume_sigma,
        counts=(1, 0, 1),
    )
    verdict = family.metal_ade_update_p_coverage(
        fields, pml, device.Residency())
    assert verdict.covered, verdict.reasons


def test_active_layer_requires_the_f_w_drive_by_identity(backend_ready):
    fields, pml, _states = build(active_pml=True)
    fields._pml_active = False
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("drive_field('E" in reason and "expected f_w_E" in reason
               for reason in verdict.reasons), verdict.reasons


def test_an_invalid_kind_is_refused_by_name(backend_ready):
    fields, pml, states = build()
    states[0].susceptibility = type("Bad", (), {"kind": "noisy"})()
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("kind 'noisy'" in reason for reason in verdict.reasons)


def test_one_refused_state_refuses_the_whole_update_p_slot(backend_ready):
    fields, pml, states = build(states=2)
    states[1]._coefficients = (1.0, np.nan, 0.25)
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("polarization 1" in reason and "not finite" in reason
               for reason in verdict.reasons)


def test_no_live_pole_is_a_refusal_not_a_vacuous_kernel(backend_ready):
    fields, pml, _states = build(counts=(0, 0, 0))
    verdict = family.metal_ade_update_p_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("no driven component" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("complex_storage", (False, True))
@pytest.mark.parametrize("volume_sigma", (False, True))
def test_plan_rotates_live_buffers_and_matches_the_reference(
        backend_ready, complex_storage, volume_sigma):
    reference, pml, _ = build(
        complex_storage=complex_storage, volume_sigma=volume_sigma,
        states=2, counts=(1, 0, 1))
    actual, actual_pml, states = build(
        complex_storage=complex_storage, volume_sigma=volume_sigma,
        states=2, counts=(1, 0, 1))
    residency = HostResidency()
    plan = family.plan_metal_ade_update_p(
        actual, actual_pml, residency, functions=_functions())
    assert plan is not None
    initial_ids = [
        tuple(id(state.P[c]) for c in state.driven())
        + tuple(id(state.P_prev[c]) for c in state.driven())
        + (id(state._scratch),)
        for state in states
    ]
    for _ in range(4):
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        plan.run()
    assert plan.runs == 4
    assert plan.launches == 4 * sum(len(state.driven()) for state in states)
    for expected_state, actual_state, physical_ids in zip(
            reference.polarizations, states, initial_ids):
        current_ids = (
            tuple(id(actual_state.P[c]) for c in actual_state.driven())
            + tuple(id(actual_state.P_prev[c]) for c in actual_state.driven())
            + (id(actual_state._scratch),)
        )
        assert sorted(current_ids) == sorted(physical_ids)
        for component in actual_state.driven():
            assert np.array_equal(
                expected_state.P[component].view(np.uint32),
                actual_state.P[component].view(np.uint32))
            assert np.array_equal(
                expected_state.P_prev[component].view(np.uint32),
                actual_state.P_prev[component].view(np.uint32))


def test_refused_builder_never_compiles(backend_ready, monkeypatch):
    fields, pml, states = build()
    states[0]._coefficients = (np.inf, 0.0, 1.0)
    monkeypatch.setattr(
        family, "compile_ade",
        lambda *args, **kwargs: pytest.fail("refused builder compiled"))
    assert family.plan_metal_ade_update_p(fields, pml, HostResidency()) is None


def test_family_is_registered_once_on_update_p():
    matches = [spec for spec in arms.registered("update_P")
               if spec.family == family.FAMILY]
    assert len(matches) == 1
    assert matches[0].label == "ADE update_P"
    assert matches[0].wired


def test_composer_selects_update_p_and_requires_the_array_update_e_sync(
        backend_ready, monkeypatch):
    """ADE fills update_P while update_E stays on the array path, so the mirror
    handoff this test exists to exercise is a real one.

    THE PREMISE IS NOW STRUCTURAL, NOT ENVIRONMENTAL. This test used to build a
    complex active-PML fixture and rely on a comment saying "complex PML
    dispersion remains deliberately unwired". That stopped being true when
    complex_dispersive_spine landed: its three arms are wired, and
    complex_dispersive_update_e claims update_E.

    Measured 2026-08-19, the reason it still passed ALONE is worse than the stale
    comment: metal_complex_dispersive_e_coverage refuses that fixture for exactly
    two ENVIRONMENTAL reasons — the resolved subnormal policy being 'keep', and no
    expansion-probe artifact being available. test_metal_planner_composition sets
    MEEP_GPU_SUBNORMAL_POLICY and the MEEP_GPU_METAL_*_EXPANSION_PROBE paths at
    MODULE SCOPE, i.e. during collection, and conftest's policy guard then applies
    that captured set to every test_metal_* module. So whether update_E composed
    depended on WHICH MODULES PYTEST COLLECTED — the test passed in isolation
    because a cold process cannot satisfy clauses it never set.

    A pole count above MAX_POLES refuses update_E in BOTH environments, on the
    kernel's own cap rather than on an unset variable, so the array-path handoff
    is guaranteed by the configuration itself.
    """
    poles = MAX_POLES + 1
    fields, pml, _states = build(active_pml=True, complex_storage=True,
                                 states=poles)
    monkeypatch.setattr(family, "compile_ade", lambda dtype, volume, contract:
                        _functions()[(contract, volume, dtype)])
    residency = HostResidency()
    unsafe = metal_launch.plan_step(fields, pml, residency, sources=())
    assert unsafe.selected["update_P"] == "ADE update_P"
    assert "update_E" not in unsafe.selected, unsafe.selected
    # The refusal must be the POLE CAP, not a missing probe or a policy mismatch —
    # otherwise this test would silently go back to measuring the environment.
    assert any(f"more than MAX_POLES={MAX_POLES}" in reason
               for reason in unsafe.reasons.get("update_E", ())), \
        unsafe.reasons.get("update_E")
    assert not unsafe.residency.covered
    assert any("update_E runs on the array path" in reason
               for reason in unsafe.residency.reasons)

    fields, pml, _states = build(active_pml=True, complex_storage=True,
                                 states=poles)
    residency = HostResidency()
    safe = metal_launch.plan_step(
        fields, pml, residency, sources=(), synced=("step_B", "update_E"))
    assert safe.selected["update_P"] == "ADE update_P"
    assert safe.residency.covered, safe.residency.reasons


def test_gate_exists_and_records_incremental_source_bound_evidence():
    gate = os.path.join(os.path.dirname(__file__), os.pardir, "parity", "meep_gpu",
                        "gate_metal_ade_update_p.py")
    source = open(os.path.abspath(gate), encoding="utf-8").read()
    for needle in ("flush=True", "jsonl", "source_provenance", "mutation"):
        assert needle in source
