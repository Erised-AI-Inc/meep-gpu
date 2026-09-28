"""Host contracts for Metal's complex/Bloch conductive PML composition.

The source specialization, planner ABI, per-target histories, scoped H/E reuse,
and array-reference lifecycle are checked here.  The native byte and whole-step
gates establish the complementary device and residency evidence required for the
family's experimental-composer arms to be wired.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_conductive_pml as family
from meep_gpu.metal_kernels import complex_fields as shared_predicate
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.pml import PML


class HostResidency:
    """Identity mirrors make the plan's 30-slot ABI observable without Torch."""

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def probe():
    return {"backend": "numpy", "patterns": {
        name: "FMA_V1" for name in (
            "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
            "f4_mul_c8_coefficient_left", "python_float_left")}}


def build(*, sides=("B", "D"), seed=79):
    grid = Grid(resolution=8.0, cell_size=(1.0, 0.875, 0.75), courant=0.3,
                boundaries="periodic", k_point=(0.2, -0.125, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    pml = PML(grid=grid, thickness=1)
    fields.enable_pml_storage()
    sigma = np.linspace(np.float32(0.06), np.float32(0.22), grid.shape[0],
                        dtype=np.float32)[:, None, None]
    sigma = np.broadcast_to(sigma, grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity(sigma)
    if "D" in sides:
        fields.set_d_conductivity(sigma)
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx",
                 "fu_Dy", "fu_Dz", "f_cond_Bx", "f_cond_By", "f_cond_Bz",
                 "f_cond_Dx", "f_cond_Dy", "f_cond_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            real = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
            imag = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
            array[...] = real + 1j * imag
    return fields, pml


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(shared_predicate, "_metal_backend_reasons", lambda grid: [])


def _reference_function(reference, actual, pml, sub_step, target_names):
    def run(*args):
        targets, auxiliary, history = args[:3], args[3:6], args[6:9]
        assert all(target is getattr(actual, name)
                   for target, name in zip(targets, target_names))
        assert all(value is getattr(actual, "fu_" + name)
                   for value, name in zip(auxiliary, target_names))
        for value, target, name in zip(history, targets, target_names):
            expected = getattr(actual, "f_cond_" + name)
            assert value is (expected if expected is not None else target)
        assert args[29].shape == (3, 2)
        getattr(stepping, sub_step)(reference, pml)
        for target, name in zip(targets, target_names):
            target[...] = getattr(reference, name)
        for value, name in zip(auxiliary, target_names):
            value[...] = getattr(reference, "fu_" + name)
        for value, name in zip(history, target_names):
            expected = getattr(reference, "f_cond_" + name)
            if expected is not None:
                value[...] = expected
    return run


def test_source_packs_phase_table_and_preserves_all_four_complex_recurrences():
    source = family.complex_conductive_pml_curl_source(
        (0, 0, 0), False, (1, 0, 0), (True, False, True), "FMA_V1")
    indices = [int(item.split(")")[0]) for item in source.split("[[buffer(")[1:]]
    assert indices == list(range(30))
    assert len(indices) <= MAX_BUFFER_BINDINGS
    assert "device const float2* phases  [[buffer(29)]]" in source
    assert "float2 px = phases[0];" in source
    assert "[[buffer(30)]]" not in source
    assert "if (dsig0 && dsigu0)" in source
    assert "else if (dsigu0)" in source
    assert "else if (dsig0)" in source
    assert "c_mul_field_left(c_mul_field_left(hp0, cf0[ii]) - curl0, ci0[ii])" in source
    assert "f1[ii] = v1;" in source


def test_lossless_target_specialization_has_no_conductivity_or_history_loads():
    source = family.complex_conductive_pml_curl_source(
        (0, 0, 0), True, (1, 0, 1), (False, True, False), "FMA_V1")
    body = source[source.index("kernel void"):]
    assert "h0[ii]" not in body and "cf0[ii]" not in body and "ci0[ii]" not in body
    assert "h2[ii]" not in body and "cf2[ii]" not in body and "ci2[ii]" not in body
    assert "float2 hp1 = h1[ii];" in body
    assert "c_mul_field_left(c_mul_field_left(fp0, km_y) + n0)" not in body
    assert "float2 v0 = c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) - p0, si_z);" in body


def test_global_conductivity_claims_both_slots_but_flags_are_per_substep(backend_ready):
    fields, pml = build(sides=("D",))
    b = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    d = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert b is not None and d is not None
    assert b.conductive == (False, False, False)
    assert d.conductive == (True, True, True)


def test_packed_phase_table_preserves_forward_and_backward_phase_arguments(backend_ready):
    fields, pml = build()
    b = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    d = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert b is not None and d is not None
    phases = shared_predicate.bloch_phase_table(fields.grid, ("periodic",) * 3)
    _flags_b, values_b = shared_predicate.phase_arguments(phases, backward=False)
    _flags_d, values_d = shared_predicate.phase_arguments(phases, backward=True)
    assert np.array_equal(b.phase_table, np.asarray(values_b, dtype=np.float32))
    assert np.array_equal(d.phase_table, np.asarray(values_d, dtype=np.float32))


def test_mixed_complex_targets_bind_only_live_fcond_histories(backend_ready):
    fields, pml = build(sides=())
    sigma = np.full(fields.grid.shape, np.float32(0.17))
    fields.set_b_conductivity({"Bx": sigma, "By": None, "Bz": None})
    fields.set_d_conductivity({"Dx": None, "Dy": None, "Dz": sigma})
    b = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    d = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: lambda *args: None})
    assert b is not None and d is not None
    assert b.conductive == (True, False, False)
    assert d.conductive == (False, False, True)


def test_two_complex_pml_curl_cycles_match_array_fields_and_both_histories(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    residency = HostResidency()
    b = family.plan_metal_complex_conductive_pml_curl(
        actual, actual_pml, "step_B", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_B", ("Bx", "By", "Bz"))})
    d = family.plan_metal_complex_conductive_pml_curl(
        actual, actual_pml, "step_D", residency, probe=probe(),
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_D", ("Dx", "Dy", "Dz"))})
    assert b is not None and d is not None
    for _ in range(2):
        b.run()
        d.run()
    for stem in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        for name in (stem, "fu_" + stem, "f_cond_" + stem):
            assert np.array_equal(getattr(actual, name).view(np.uint32),
                                  getattr(reference, name).view(np.uint32)), name
    assert b.launches == d.launches == 2


def test_complex_constitutive_scope_ignores_only_conductivity(backend_ready):
    """H/E are element-wise here and do not read either conductivity table.

    This must not mask an ADE, nonlinearity, or any other constitutive condition:
    those change H/E arithmetic and stay owned by the common complex predicate.
    """
    fields, pml = build(sides=("D",))
    for side in ("H", "E"):
        verdict = family.metal_complex_conductive_pml_constitutive_coverage(
            fields, pml, side, HostResidency(), probe())
        assert verdict.covered, verdict.reasons

    lossless, lossless_pml = build(sides=())
    verdict = family.metal_complex_conductive_pml_constitutive_coverage(
        lossless, lossless_pml, "E", HostResidency(), probe())
    assert not verdict.covered
    assert any("no curl target carries a conductivity" in reason
               for reason in verdict.reasons), verdict.reasons


def test_complex_constitutive_scope_keeps_a_live_polarization_refused(backend_ready):
    """The scope view must not make an E source of ``D - sum(P)`` look like D."""
    fields, pml = build(sides=("D",))
    fields.polarizations.append(PolarizationState(
        Susceptibility(0.9, 0.06, LORENTZIAN), {"Ex": 0.2}, fields.grid,
        np.complex64))
    verdict = family.metal_complex_conductive_pml_constitutive_coverage(
        fields, pml, "E", HostResidency(), probe())
    assert not verdict.covered
    assert any("susceptibility" in reason for reason in verdict.reasons), verdict.reasons


def test_arm_uses_the_composer_supplied_complex_probe(backend_ready, monkeypatch):
    """An ambient historical probe may not silently license this arm."""
    fields, pml = build(sides=("D",))
    monkeypatch.delenv(shared_predicate.PROBE_PATH_ENVIRONMENT, raising=False)
    context = arms.StepContext(fields, pml, HostResidency(),
                               extra={"complex_probe": probe()})
    verdict = family._arm_coverage(context, "step_B")  # noqa: SLF001
    assert verdict.covered, verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("real", "storage is real"),
    ("lossless", "no curl target"),
    ("inactive", "no active PML"),
    ("missing_probe", "probe artifact"),
])
def test_distinct_products_are_refused_by_name(backend_ready, monkeypatch, mutation, needle):
    fields, pml = build(sides=(() if mutation == "lossless" else ("B", "D")))
    record = probe()
    if mutation == "real":
        fields.force_complex_fields = False
        fields.grid.k_point = (0.0, 0.0, 0.0)
        fields.grid.bloch_phases = (None, None, None)
        fields.grid.has_bloch = False
    elif mutation == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    elif mutation == "missing_probe":
        record = None
        monkeypatch.delenv(shared_predicate.PROBE_PATH_ENVIRONMENT, raising=False)
    verdict = family.metal_complex_conductive_pml_curl_coverage(
        fields, pml, "step_B", HostResidency(), record)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_arm_is_wired_after_complete_complex_residency_composition():
    expected = {
        "step_B": "complex conductive PML curl",
        "step_D": "complex conductive PML curl",
        "update_H": "complex conductive PML constitutive",
        "update_E": "complex conductive PML constitutive",
    }
    for slot, label in expected.items():
        arm = next(spec for spec in arms.registered(slot) if spec.family == family.FAMILY)
        assert arm.label == label
        assert arm.wired


def test_registry_and_lazy_package_surface_include_the_family():
    from meep_gpu.metal_kernels import registry
    import meep_gpu.metal_kernels as package

    assert family.FAMILY in registry.FAMILY_MODULES
    assert family.FAMILY in package.__all__
    assert package.complex_conductive_pml is family
