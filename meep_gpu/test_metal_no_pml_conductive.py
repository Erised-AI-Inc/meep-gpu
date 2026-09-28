"""Host contracts for Metal's real no-PML conductive curl family.

The native-MPS byte gate is deliberately separate.  This suite establishes the
arithmetic shape, per-target admission, bindings, pointer sharing, and exact
reference lifecycle without claiming that an unavailable MPS device executed the
shader.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms
from meep_gpu.metal_kernels import no_pml_conductive as family
from meep_gpu.metal_kernels import shaders
from meep_gpu.pml import PML


class HostResidency:
    """Identity mirrors for host-plan contract tests only."""

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        current = self.arrays.get(name)
        if current is not None and current is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def build(*, sides=("B", "D"), stored_e=False, seed=36):
    grid = Grid(resolution=8.0, cell_size=(1.0, 0.875, 0.75), courant=0.3,
                boundaries=("metallic", "periodic", "metallic"), xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    if stored_e:
        fields.enable_field_storage()
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        field = getattr(fields, name, None)
        if field is not None:
            field[...] = rng.uniform(-0.35, 0.35, grid.shape).astype(np.float32)
    sigma = np.linspace(np.float32(0.04), np.float32(0.28), num=grid.shape[0],
                        dtype=np.float32)[:, None, None]
    sigma = np.broadcast_to(sigma, grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity(sigma)
    if "D" in sides:
        fields.set_d_conductivity(sigma)
    return fields, PML(grid=grid, thickness=0)


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _reference_function(reference, actual, pml, sub_step, expected_targets):
    """A host executor that checks plan bindings then runs the actual array path."""
    def run(*args):
        targets = args[:3]
        condfac = args[9:12]
        condinv = args[12:15]
        assert tuple(targets) == tuple(expected_targets)
        for index, target in enumerate(family.SUB_STEPS[sub_step]["targets"]):
            if actual.condfac_for(target) is None:
                assert condfac[index] is targets[index]
                assert condinv[index] is targets[index]
            else:
                assert condfac[index] is actual.condfac_for(target)
                assert condinv[index] is actual.condinv_for(target)
        getattr(stepping, sub_step)(reference, pml)
        for target, name in zip(targets, family.SUB_STEPS[sub_step]["targets"]):
            target[...] = getattr(reference, name)
    return run


def test_shader_preserves_three_ordered_conductive_passes_per_lossy_target():
    source = family.conductive_plain_curl_source(
        (0, 0, 0), backward=False, derive=True, conductive=(True, False, True))
    assert "dtdx * ((c_y - c) + (b - b_z))" in source
    assert "value0 = value0 * cf0[ii];" in source
    assert "value0 = value0 - curl0;" in source
    assert "value0 = value0 * ci0[ii];" in source
    assert "f0[ii] = value0;" in source
    assert "f1[ii] = f1[ii] - curl1;" in source
    assert "value2 = value2 * cf2[ii];" in source
    assert "[[buffer(19)]]" in source
    assert "[[buffer(20)]]" not in source


@pytest.mark.parametrize(("sides", "expected"), [
    (("B",), {"step_B": (True, True, True), "step_D": (False, False, False)}),
    (("D",), {"step_B": (False, False, False), "step_D": (True, True, True)}),
    (("B", "D"), {"step_B": (True, True, True), "step_D": (True, True, True)}),
])
def test_each_substep_gets_its_own_conductivity_flags(backend_ready, sides, expected):
    fields, pml = build(sides=sides)
    residency = HostResidency()
    for slot in family.SUB_STEPS:
        verdict = family.metal_conductive_plain_curl_coverage(
            fields, pml, slot, residency)
        assert verdict.covered, verdict.reasons
        plan = family.plan_metal_conductive_plain_curl(
            fields, pml, slot, residency,
            functions={shaders.CONTRACT_OFF: lambda *args: None})
        assert plan is not None
        assert plan.conductive == expected[slot]


def test_d_only_conductivity_does_not_bind_or_apply_a_b_coefficient(backend_ready):
    fields, pml = build(sides=("D",))
    plan = family.plan_metal_conductive_plain_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert plan is not None
    assert plan.conductive == (False, False, False)
    assert plan._args[9:12] == plan._args[:3]
    assert plan._args[12:15] == plan._args[:3]


def test_mixed_target_flags_do_not_turn_lossless_neighbours_conductive(backend_ready):
    fields, pml = build(sides=())
    sigma = np.full(fields.grid.shape, np.float32(0.2))
    fields.set_b_conductivity({"Bx": sigma, "By": None, "Bz": None})
    fields.set_d_conductivity({"Dx": None, "Dy": None, "Dz": sigma})
    b = family.plan_metal_conductive_plain_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    d = family.plan_metal_conductive_plain_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert b is not None and d is not None
    assert b.conductive == (True, False, False)
    assert d.conductive == (False, False, True)


def test_two_curl_cycles_match_the_array_path_with_shared_live_host_arrays(
        backend_ready):
    reference, reference_pml = build(sides=("B", "D"))
    actual, actual_pml = build(sides=("B", "D"))
    residency = HostResidency()
    functions_b = {shaders.CONTRACT_OFF: _reference_function(
        reference, actual, reference_pml, "step_B",
        [actual.Bx, actual.By, actual.Bz])}
    functions_d = {shaders.CONTRACT_OFF: _reference_function(
        reference, actual, reference_pml, "step_D",
        [actual.Dx, actual.Dy, actual.Dz])}
    plan_b = family.plan_metal_conductive_plain_curl(
        actual, actual_pml, "step_B", residency, functions=functions_b)
    plan_d = family.plan_metal_conductive_plain_curl(
        actual, actual_pml, "step_D", residency, functions=functions_d)
    assert plan_b is not None and plan_d is not None
    for _ in range(2):
        plan_b.run()
        plan_d.run()
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        assert np.array_equal(getattr(actual, name).view(np.uint32),
                              getattr(reference, name).view(np.uint32))
    assert plan_b.launches == plan_d.launches == 2
    assert residency.arrays["Bx"] is actual.Bx
    assert residency.arrays["Bx"] is not actual.Dx


@pytest.mark.parametrize("product, needle", [
    ("lossless", "conductivity"), ("active_pml", "active PML"),
    ("complex", "complex64"),
])
def test_distinct_products_are_refused_by_name(backend_ready, product, needle):
    fields, pml = build(sides=(() if product == "lossless" else ("B", "D")))
    if product == "active_pml":
        pml = PML(grid=fields.grid, thickness=1)
    elif product == "complex":
        fields.force_complex_fields = True
    verdict = family.metal_conductive_plain_curl_coverage(
        fields, pml, "step_B", HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_builder_refuses_before_compilation_when_the_conductivity_pair_is_broken(
        backend_ready, monkeypatch):
    fields, pml = build(sides=("B",))
    monkeypatch.setattr(fields, "condinv_for",
                        lambda component: None if component == "Bx"
                        else fields._condinv.get(component))
    monkeypatch.setattr(family, "compile_source",
                        lambda source: pytest.fail("refused builder compiled"))
    assert family.plan_metal_conductive_plain_curl(
        fields, pml, "step_B", HostResidency()) is None


def test_arm_is_wired_after_complete_no_pml_residency_composition():
    """The no-PML null pair creates no host-write synchronization seam."""
    for slot in family.SUB_STEPS:
        arm = next(spec for spec in arms.registered(slot)
                   if spec.family == family.FAMILY)
        assert arm.label == "conductive no-PML curl"
        assert arm.wired


def test_native_gate_is_incremental_and_arms_conductive_failure_modes():
    from pathlib import Path

    gate = (Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
            / "gate_metal_no_pml_conductive.py")
    source = gate.read_text(encoding="utf-8")
    assert "flush=True" in source and "os.fsync" in source
    assert "condfac_condinv_swapped" in source
    assert "target_store_dropped" in source
    assert "curl_grouping_flattened" in source
