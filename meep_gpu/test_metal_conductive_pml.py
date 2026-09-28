"""Host contracts for Metal's real conductive split-field PML curl family."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import conductive_pml as family
from meep_gpu.pml import PML


class HostResidency:
    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def build(*, sides=("B", "D"), seed=63):
    grid = Grid(resolution=8.0, cell_size=(1.25, 1.0, 0.75), courant=0.35,
                boundaries=("metallic", "periodic", "metallic"), xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    pml = PML(grid=grid, thickness=1)
    fields.enable_pml_storage()
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx",
                 "fu_Dy", "fu_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
    sigma = np.linspace(np.float32(0.06), np.float32(0.22), grid.shape[0],
                        dtype=np.float32)[:, None, None]
    sigma = np.broadcast_to(sigma, grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity(sigma)
    if "D" in sides:
        fields.set_d_conductivity(sigma)
    for name in ("f_cond_Bx", "f_cond_By", "f_cond_Bz", "f_cond_Dx",
                 "f_cond_Dy", "f_cond_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, pml


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _reference_function(reference, actual, pml, sub_step, target_names):
    def run(*args):
        targets, auxiliary, history = args[:3], args[3:6], args[6:9]
        for index, name in enumerate(target_names):
            assert targets[index] is getattr(actual, name)
            assert auxiliary[index] is getattr(actual, "fu_" + name)
            if actual.condfac_for(name) is None:
                assert history[index] is targets[index]
            else:
                assert history[index] is getattr(actual, "f_cond_" + name)
        getattr(stepping, sub_step)(reference, pml)
        for target, name in zip(targets, target_names):
            target[...] = getattr(reference, name)
        for item, name in zip(auxiliary, target_names):
            item[...] = getattr(reference, "fu_" + name)
        for item, name in zip(history, target_names):
            if reference.condfac_for(name) is not None:
                item[...] = getattr(reference, "f_cond_" + name)
    return run


def test_source_has_all_four_conductive_cases_and_exact_pml_activity_tests():
    source = family.conductive_pml_curl_source(
        (0, 1, 0), False, (True, False, True))
    assert "(km_y != 1.0f) || (si_y != 1.0f)" in source
    assert "if (dsig0 && dsigu0)" in source
    assert "else if (dsigu0)" in source
    assert "else if (dsig0)" in source
    assert "f0[ii] = ((f0_previous * cf0[ii]) - curl0) * ci0[ii];" in source
    assert "f1[ii] = f1[ii] - curl1;" not in source
    assert "[[buffer(28)]]" in source and "[[buffer(29)]]" not in source


def test_each_pml_curl_requires_its_own_conductive_side(backend_ready):
    fields, pml = build(sides=("D",))
    b = family.metal_conductive_pml_curl_coverage(fields, pml, "step_B", HostResidency())
    d = family.metal_conductive_pml_curl_coverage(fields, pml, "step_D", HostResidency())
    assert not b.covered
    assert any("no step_B target" in reason for reason in b.reasons)
    assert d.covered, d.reasons


def test_mixed_pml_targets_bind_only_their_own_histories(backend_ready):
    fields, pml = build(sides=())
    sigma = np.full(fields.grid.shape, np.float32(0.15))
    fields.set_b_conductivity({"Bx": sigma, "By": None, "Bz": None})
    fields.set_d_conductivity({"Dx": None, "Dy": None, "Dz": sigma})
    b = family.plan_metal_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    d = family.plan_metal_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert b is not None and d is not None
    assert b.conductive == (True, False, False)
    assert d.conductive == (False, False, True)


def test_two_active_pml_curl_cycles_match_the_array_path_and_histories(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    residency = HostResidency()
    b_targets, d_targets = ("Bx", "By", "Bz"), ("Dx", "Dy", "Dz")
    b = family.plan_metal_conductive_pml_curl(
        actual, actual_pml, "step_B", residency,
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_B", b_targets)})
    d = family.plan_metal_conductive_pml_curl(
        actual, actual_pml, "step_D", residency,
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_D", d_targets)})
    assert b is not None and d is not None
    for _ in range(2):
        b.run()
        d.run()
    for stem in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        for name in (stem, "fu_" + stem, "f_cond_" + stem):
            assert np.array_equal(getattr(actual, name).view(np.uint32),
                                  getattr(reference, name).view(np.uint32)), name
    assert b.launches == d.launches == 2


@pytest.mark.parametrize("product, needle", [
    ("inactive", "active PML"), ("real_lossless", "no step_B target"),
    ("complex", "complex64"),
])
def test_distinct_products_are_refused_by_name(backend_ready, product, needle):
    fields, pml = build(sides=(() if product == "real_lossless" else ("B", "D")))
    if product == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    elif product == "complex":
        fields.force_complex_fields = True
    verdict = family.metal_conductive_pml_curl_coverage(
        fields, pml, "step_B", HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_arm_is_wired_after_complete_residency_composition():
    """The whole-step gate makes this a dispatchable composition arm.

    The direct family gate owns the stencil and history recurrence.  The
    whole-step gate additionally proves that the B/D mirror can remain resident
    while ordinary H/E execute in the same step, including both directions of
    conductivity.  Keep the assertion here so a future regression cannot make
    the product silently array-only while its direct gate stays green.
    """
    for slot in family.CONDUCTIVE_SUB_STEPS:
        arm = next(spec for spec in arms.registered(slot) if spec.family == family.FAMILY)
        assert arm.label == "conductive PML curl"
        assert arm.wired
