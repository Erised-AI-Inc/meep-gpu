"""Host contracts for Metal's complex/Bloch conductive PML composition.

The source specialization, planner ABI, per-target histories, scoped H/E reuse,
and array-reference lifecycle are checked here.  The native byte and whole-step
gates establish the complementary device and residency evidence required for the
family's experimental-composer arms to be wired.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record
from meep_gpu.dispersion import LORENTZIAN, PolarizationState, Susceptibility
from meep_gpu.fields import IYEE_SHIFTS, Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_conductive_pml as family
from meep_gpu.metal_kernels import complex_fields as shared_predicate
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.pml import PML


class HostResidency:
    """Identity mirrors make the plan's 31-slot ABI observable on the host.

    ``device`` is the CPU, so the plan-owned chunk record (the 31st argument) is a
    host tensor rather than an MPS upload.
    """

    device = "cpu"

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


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


def record(fields, pml):
    """The one-rank chunk record the array path steps these fields with."""
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


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
        assert tuple(args[30].shape) == (25,)
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
        (0, 0, 0), False, (2, 0, 0), (True, False, True))
    indices = [int(item.split(")")[0]) for item in source.split("[[buffer(")[1:]]
    assert indices == list(range(31))
    assert len(indices) <= MAX_BUFFER_BINDINGS
    assert "device const float2* phases  [[buffer(29)]]" in source
    assert "constant ChunkBounds& cb [[buffer(30)]]" in source
    assert "float2 px = phases[0];" in source
    assert "if (dsig0 && dsigu0)" in source
    assert "else if (dsigu0)" in source
    assert "else if (dsig0)" in source
    assert "c_mul_field_left(c_mul_field_left(hp0, cf0[ii]) - curl0, ci0[ii])" in source
    assert "f1[ii] = v1;" in source


def test_lossless_target_specialization_has_no_conductivity_or_history_loads():
    source = family.complex_conductive_pml_curl_source(
        (0, 0, 0), True, (2, 0, 1), (False, True, False))
    body = source[source.index("kernel void"):]
    assert "h0[ii]" not in body and "cf0[ii]" not in body and "ci0[ii]" not in body
    assert "h2[ii]" not in body and "cf2[ii]" not in body and "ci2[ii]" not in body
    assert "float2 hp1 = dsig1 ? h1[ii] : float2(0.0f, 0.0f);" in body
    assert "float2 up1 = dsigu1 ? u1[ii] : float2(0.0f, 0.0f);" in body
    assert "c_mul_field_left(c_mul_field_left(fp0, km_y) + n0)" not in body
    assert ("float2 v0 = du0 ? c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) "
            "- p0, si_z) : n0;") in body
    assert "if (du0) { u0[ii] = n0; }" in body


@pytest.mark.parametrize("backward", [False, True])
def test_the_four_cases_select_on_chunk_membership_not_coefficient_identity(backward):
    """MEEP selects a conductive cell's case on its chunk's dsig and dsigu layers
    (step_db.cpp:56-59); the array path passes those memberships
    (``stepping._apply_conductive_pml_update``). A coefficient-identity predicate
    takes a different case on a sigma = 0 cell inside a layer chunk."""
    source = family.complex_conductive_pml_curl_source(
        (0, 0, 0), backward, (0, 0, 0), (True, True, True))
    body = source[source.index("kernel void"):]
    assert "!= 1.0f" not in body
    half = shaders.CURL_HALF_INTEGER[backward]
    lattice = "h" if half else "i"
    for t, (dsig_axis, _dsigu) in enumerate(shaders.CURL_CYCLE):
        assert f"const bool dsig{t} = in_{lattice}_{'xyz'[dsig_axis]};" in body
        assert f"const bool dsigu{t} = du{t};" in body


@pytest.mark.parametrize("backward", [False, True])
def test_the_f_cond_increment_is_formed_before_it_is_added(backward):
    """MEEP forms ``(fcnd' - fcnd)`` first and adds it (step_generic.cpp:167, :209), on
    the both-layer and the dsig-only forms; ``((x*kms) + fcnd') - fcnd`` is another
    float32 number. Each part rounds the same way (``stepping._apply_conductive_pml_update``
    on complex storage)."""
    source = family.complex_conductive_pml_curl_source(
        (0, 0, 0), backward, (0, 0, 0), (True, True, True))
    body = source[source.index("kernel void"):]
    for t, (km1, si1) in enumerate((("km_y", "si_y"), ("km_z", "si_z"),
                                    ("km_x", "si_x"))):
        assert body.count(f"float2 hs{t} = hn{t} - hp{t};") == 2
        assert (f"float2 un{t} = c_mul_field_left(c_mul_field_left(up{t}, {km1}) "
                f"+ hs{t}, {si1});") in body
        assert (f"float2 fn{t} = c_mul_field_left(c_mul_field_left(fp{t}, {km1}) "
                f"+ hs{t}, {si1});") in body
        assert f"+ hn{t}) - hp{t}" not in body


def test_each_history_is_loaded_and_stored_only_under_its_forms_guard():
    """The auxiliary contract: ``f_cond`` only on dsig cells, ``fu`` only on dsigu
    cells. The loads sit under the guard (the gate's hoisted load is a declared null,
    so the guard is pinned here) and every store is inside its form's branch."""
    source = family.complex_conductive_pml_curl_source(
        (0, 0, 0), False, (0, 0, 0), (True, True, True))
    body = source[source.index("kernel void"):]
    for t in range(3):
        assert f"float2 hp{t} = dsig{t} ? h{t}[ii] : float2(0.0f, 0.0f);" in body
        assert f"float2 up{t} = dsigu{t} ? u{t}[ii] : float2(0.0f, 0.0f);" in body
        assert body.count(f"h{t}[ii] = hn{t};") == 2
        assert body.count(f"h{t}[ii]") == 3  # the guarded load and the two stores
        assert body.count(f"u{t}[ii] = un{t};") == 2


def test_global_conductivity_claims_both_slots_but_flags_are_per_substep(backend_ready):
    fields, pml = build(sides=("D",))
    b = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
        chunk_bounds=record(fields, pml))
    d = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
        chunk_bounds=record(fields, pml))
    assert b is not None and d is not None
    assert b.conductive == (False, False, False)
    assert d.conductive == (True, True, True)


def test_packed_phase_table_preserves_forward_and_backward_phase_arguments(backend_ready):
    fields, pml = build()
    b = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
        chunk_bounds=record(fields, pml))
    d = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
        chunk_bounds=record(fields, pml))
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
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
        chunk_bounds=record(fields, pml))
    d = family.plan_metal_complex_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
        chunk_bounds=record(fields, pml))
    assert b is not None and d is not None
    assert b.conductive == (True, False, False)
    assert d.conductive == (False, False, True)


def test_two_complex_pml_curl_cycles_match_array_fields_and_both_histories(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    residency = HostResidency()
    b = family.plan_metal_complex_conductive_pml_curl(
        actual, actual_pml, "step_B", residency,
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_B", ("Bx", "By", "Bz"))},
        chunk_bounds=record(actual, actual_pml))
    d = family.plan_metal_complex_conductive_pml_curl(
        actual, actual_pml, "step_D", residency,
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_D", ("Dx", "Dy", "Dz"))},
        chunk_bounds=record(actual, actual_pml))
    assert b is not None and d is not None
    for _ in range(2):
        b.run()
        d.run()
    for stem in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        for name in (stem, "fu_" + stem, "f_cond_" + stem):
            assert np.array_equal(getattr(actual, name).view(np.uint32),
                                  getattr(reference, name).view(np.uint32)), name
    assert b.launches == d.launches == 2


def _partial(side, target):
    """Conductivity on two x rows only: on the one-rank division some chunks keep it."""
    fields, pml = build(sides=())
    sigma = np.zeros(fields.grid.shape, dtype=np.float32)
    sigma[3:5] = np.float32(0.15)
    names = ("Bx", "By", "Bz") if side == "B" else ("Dx", "Dy", "Dz")
    (fields.set_b_conductivity if side == "B" else fields.set_d_conductivity)(
        {name: (sigma if name == target else None) for name in names})
    return fields, pml


@pytest.mark.parametrize("side, slot, target", [("B", "step_B", "Bx"),
                                                ("D", "step_D", "Dz")])
def test_a_chunk_that_keeps_no_conductivity_refuses_by_name(backend_ready, side, slot,
                                                            target):
    """MEEP steps the lossless forms in a chunk that keeps no conductivity
    (structure.cpp:883-900); the record carries no per-chunk presence, so the sub-step
    is refused to the array path, and the family's H/E arms with it."""
    fields, pml = _partial(side, target)
    kept = ChunkLayout.one_rank(fields.grid, pml).conductive_chunks(
        fields.conductivity_for(target), IYEE_SHIFTS[target])
    assert any(kept) and not all(kept)
    needle = f"structure chunks keep the conductivity of {target}"
    verdict = family.metal_complex_conductive_pml_curl_coverage(
        fields, pml, slot, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons
    assert family.plan_metal_complex_conductive_pml_curl(
        fields, pml, slot, HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None},
        chunk_bounds=record(fields, pml)) is None
    other = "step_D" if slot == "step_B" else "step_B"
    assert family.metal_complex_conductive_pml_curl_coverage(
        fields, pml, other, HostResidency()).covered
    for constitutive_side in ("H", "E"):
        companion = family.metal_complex_conductive_pml_constitutive_coverage(
            fields, pml, constitutive_side, HostResidency())
        assert not companion.covered
        assert any(needle in reason for reason in companion.reasons), companion.reasons


def test_the_presence_is_read_on_the_division_it_is_handed(backend_ready):
    """The arms read the composer's ``chunk_layout``, not a default of their own."""
    fields, pml = build(sides=("D",))

    class _HalfTheChunksKeepIt:
        def conductive_chunks(self, sigma, iyee):  # noqa: ARG002
            return (False, True)

    verdict = family.metal_complex_conductive_pml_curl_coverage(
        fields, pml, "step_D", HostResidency(),
        chunk_layout=_HalfTheChunksKeepIt())
    assert not verdict.covered
    assert any("1 of 2 structure chunks" in reason for reason in verdict.reasons)
    companion = family.metal_complex_conductive_pml_constitutive_coverage(
        fields, pml, "E", HostResidency(), chunk_layout=_HalfTheChunksKeepIt())
    assert not companion.covered
    for function in (family._arm_coverage, family._arm_plan,  # noqa: SLF001
                     family._constitutive_arm_coverage,  # noqa: SLF001
                     family._constitutive_arm_plan):  # noqa: SLF001
        assert "chunk_layout=context.chunk_layout" in inspect.getsource(function)


def test_complex_constitutive_scope_ignores_only_conductivity(backend_ready):
    """H/E are element-wise here and do not read either conductivity table.

    This must not mask an ADE, nonlinearity, or any other constitutive condition:
    those change H/E arithmetic and stay owned by the common complex predicate.
    """
    fields, pml = build(sides=("D",))
    for side in ("H", "E"):
        verdict = family.metal_complex_conductive_pml_constitutive_coverage(
            fields, pml, side, HostResidency())
        assert verdict.covered, verdict.reasons

    lossless, lossless_pml = build(sides=())
    verdict = family.metal_complex_conductive_pml_constitutive_coverage(
        lossless, lossless_pml, "E", HostResidency())
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
        fields, pml, "E", HostResidency())
    assert not verdict.covered
    assert any("susceptibility" in reason for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("real", "storage is real"),
    ("lossless", "no curl target"),
    ("inactive", "no active PML"),
])
def test_distinct_products_are_refused_by_name(backend_ready, mutation, needle):
    fields, pml = build(sides=(() if mutation == "lossless" else ("B", "D")))
    if mutation == "real":
        fields.force_complex_fields = False
        fields.grid.k_point = (0.0, 0.0, 0.0)
        fields.grid.bloch_phases = (None, None, None)
        fields.grid.has_bloch = False
    elif mutation == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    verdict = family.metal_complex_conductive_pml_curl_coverage(
        fields, pml, "step_B", HostResidency())
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
