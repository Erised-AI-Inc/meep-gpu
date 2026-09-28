"""Host contract for Metal's real no-PML stored-E constitutive family."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu.dispersion import LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import no_pml_stored_e as family
from meep_gpu.metal_kernels import arms
from meep_gpu.metal_kernels import shaders
from meep_gpu.pml import PML
from meep_gpu import stepping


class HostResidency:
    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        current = self.arrays.get(name)
        if current is not None and current is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def build(*, poles: int = 2, stored: bool = True, seed: int = 7):
    grid = Grid(resolution=6.0, cell_size=(1.0, 0.75, 0.5), courant=0.3,
                boundaries="periodic", xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    if stored:
        fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        array = getattr(fields, name)
        if array is not None:
            array[...] = rng.uniform(-0.4, 0.4, grid.shape).astype(np.float32)
    states = []
    for number in range(poles):
        state = PolarizationState(
            Susceptibility(1.0 + number * 0.1, 0.05, LORENTZIAN),
            {"Ex": 0.2 + number * 0.01, "Ey": 0.0, "Ez": 0.3 + number * 0.01},
            grid, np.float32)
        fields.polarizations.append(state)
        states.append(state)
        for component in state.driven():
            state.P[component][...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, PML(grid=grid, thickness=0), states


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _host_function(pole_count):
    def run(out, displacement, inverse, *args):
        *poles, n_elem = args
        assert out.size == n_elem
        source = displacement.copy()
        for pole in poles[:pole_count]:
            source = source - pole
        out[...] = source * inverse
    return run


@pytest.mark.parametrize("pole_count, final_slot", ((0, None), (3, 2), (8, 7)))
def test_source_is_component_serial_to_stay_below_metal_binding_limit(
        pole_count, final_slot):
    source = family.stored_e_source(pole_count)
    assert "[[buffer(11)]]" in source
    assert "[[buffer(12)]]" not in source
    for index in range(family.MAX_POLES):
        subtraction = f"source = source - p{index}[idx];"
        assert (subtraction in source) is (final_slot is not None and index <= final_slot)
    assert "e_out[idx] = source * inv_e[idx];" in source


def test_real_no_pml_polarized_product_is_admitted(backend_ready):
    fields, pml, _ = build(poles=2)
    verdict = family.metal_stored_e_coverage(fields, pml, HostResidency())
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("product, needle", [
    ("no_storage", "recomputed"), ("complex", "complex64"),
    ("active_pml", "active PML"), ("offdiagonal", "off-diagonal"),
])
def test_distinct_products_are_refused_by_name(backend_ready, product, needle):
    fields, pml, _ = build(stored=product != "no_storage")
    if product == "complex":
        fields.force_complex_fields = True
    elif product == "active_pml":
        pml = PML(grid=fields.grid, thickness=1)
    elif product == "offdiagonal":
        fields._chi1inv_offdiagonal = {  # noqa: SLF001 - predicate fixture
            "Ex": {"Ey": np.full(fields.grid.shape, 0.05, np.float32)}}
    verdict = family.metal_stored_e_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_plan_matches_reference_and_resolves_rotating_pointers(backend_ready):
    reference, pml, _ = build(poles=2)
    actual, actual_pml, states = build(poles=2)
    functions = {(shaders.CONTRACT_OFF, count): _host_function(count)
                 for count in (0, 2)}
    plan = family.plan_metal_stored_e(actual, actual_pml, HostResidency(),
                                      functions=functions)
    assert plan is not None
    physical = {id(array) for state in states for component in state.driven()
                for array in (state.P[component], state.P_prev[component])}
    physical.update(id(state._scratch) for state in states)
    for _ in range(3):
        stepping.update_E(reference, pml)
        for state in reference.polarizations:
            state.update(reference.drive_field, reference.grid.dt)
        plan.run()
        for state in states:
            state.update(actual.drive_field, actual.grid.dt)
    for component in ("Ex", "Ey", "Ez"):
        assert np.array_equal(getattr(actual, component).view(np.uint32),
                              getattr(reference, component).view(np.uint32))
    observed = {id(array) for state in states for component in state.driven()
                for array in (state.P[component], state.P_prev[component])}
    observed.update(id(state._scratch) for state in states)
    assert observed == physical
    assert plan.launches == 9


def test_zero_poles_compiles_the_plain_stored_e_degenerate_case(backend_ready):
    fields, pml, _ = build(poles=0)
    plan = family.plan_metal_stored_e(
        fields, pml, HostResidency(),
        functions={(shaders.CONTRACT_OFF, 0): _host_function(0)})
    assert plan is not None
    plan.run()
    for component, displacement in family.E_TERMS:
        expected = getattr(fields, displacement) * fields.inverse_epsilon_for(component)
        assert np.array_equal(getattr(fields, component).view(np.uint32), expected.view(np.uint32))


def test_supplied_host_functions_never_trigger_native_compilation(backend_ready,
                                                                   monkeypatch):
    """Gate tests must not accidentally compile a shader through ``setdefault``."""
    fields, pml, _ = build(poles=2)
    monkeypatch.setattr(family, "compile_stored_e",
                        lambda *args: pytest.fail("unexpected native compilation"))
    plan = family.plan_metal_stored_e(
        fields, pml, HostResidency(),
        functions={(shaders.CONTRACT_OFF, count): _host_function(count)
                   for count in (0, 2)})
    assert plan is not None


def test_more_than_eight_ordered_poles_is_refused_without_truncation(backend_ready):
    fields, pml, _ = build(poles=9)
    verdict = family.metal_stored_e_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any("more than MAX_POLES=8" in reason for reason in verdict.reasons)


def test_arm_is_wired_after_the_device_and_whole_step_gates():
    arm = next(spec for spec in arms.registered("update_E")
               if spec.family == family.FAMILY)
    assert arm.label == "no-PML stored E"
    assert arm.wired
