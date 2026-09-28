"""Host contracts for the complex64 no-absorber ADE Triton family."""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML

pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")

MODULE_NAME = "meep_gpu.triton_kernels.complex_ade"
MODULE_PATH = pathlib.Path(__file__).parent / "triton_kernels" / "complex_ade.py"


def _detail(value="FMA_V1"):
    return {
        "matches": {"FMA_V1": value == "FMA_V1", "NAIVE": value == "NAIVE",
                    "PLANEWISE_diagnostic": False},
        "mismatch_words": {"FMA_V1": 0 if value == "FMA_V1" else 7,
                           "NAIVE": 0 if value == "NAIVE" else 7,
                           "PLANEWISE_diagnostic": 31},
        "licensable_arms_disagreement_words": 128,
        "discriminates": True,
        "classified": value,
    }


def probe_record(module, value="FMA_V1"):
    patterns = {name: value for name in module.COMPLEX_ADE_PROBE_PATTERNS}
    return {
        "backend": "cupy",
        "patterns": patterns,
        "vectors": {name: 2792 for name in patterns},
        "detail": {name: _detail(value) for name in patterns},
        "subnormal_policy": {
            "policy": "ieee_keep_ftz_stripped", "resolved": "keep"},
        "candidates": {"policy": "keep"},
    }


@pytest.fixture(scope="module")
def arm():
    return importlib.import_module(MODULE_NAME)


def build(*, kind="lorentzian", counts=(1, 1, 1), volume_sigma=False,
          active_pml=False, complex_storage=True, seed=31):
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.7, 0.6), dimensions=3,
                boundaries="periodic", courant=0.35,
                k_point=(0.2, -0.1, 0.3), xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)
    sigma = {}
    for axis, component in enumerate(("Ex", "Ey", "Ez")):
        if counts[axis] == 0:
            sigma[component] = 0.0
        elif volume_sigma:
            sigma[component] = numpy.linspace(
                0.1 + axis * 0.05, 0.5 + axis * 0.05,
                numpy.prod(grid.shape), dtype=numpy.float32).reshape(grid.shape)
        else:
            sigma[component] = 0.3 + axis * 0.05
    state = PolarizationState(
        Susceptibility(frequency=1.0, gamma=0.1, kind=kind), sigma, grid,
        numpy.complex64 if complex_storage else numpy.float32)
    fields.polarizations.append(state)
    rng = numpy.random.default_rng(seed)
    for component in ("Ex", "Ey", "Ez"):
        array = getattr(fields, component)
        array.real = rng.uniform(-0.4, 0.4, grid.shape).astype(numpy.float32)
        if numpy.iscomplexobj(array):
            array.imag = rng.uniform(-0.4, 0.4, grid.shape).astype(numpy.float32)
    for component in state.driven():
        for slot in (state.P, state.P_prev):
            slot[component].real = rng.uniform(
                -0.2, 0.2, grid.shape).astype(numpy.float32)
            if numpy.iscomplexobj(slot[component]):
                slot[component].imag = rng.uniform(
                    -0.2, 0.2, grid.shape).astype(numpy.float32)
    return fields, pml, state


def clean_reasons(arm, fields, pml, state, component="Ez", probe=None):
    record = probe_record(arm) if probe is None else probe
    return [reason for reason in arm.complex_ade_update_p_coverage(
        fields, pml, state, component, probe=record).reasons
            if "not cupy" not in reason]


def test_module_imports_and_answers_coverage_without_triton(arm, monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    for name in [name for name in sys.modules if name.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    module = importlib.import_module(MODULE_NAME)
    fields, pml, state = build()
    assert isinstance(module.complex_ade_update_p_coverage(
        fields, pml, state, "Ez", probe=probe_record(module)).covered, bool)


def test_package_does_not_eagerly_import_the_new_family():
    sys.modules.pop(MODULE_NAME, None)
    importlib.import_module("meep_gpu.triton_kernels")
    assert MODULE_NAME not in sys.modules


def test_source_uses_word_pairs_and_the_three_normative_orientations(arm):
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "complex_ade_update_p")
    body = ast.unparse(kernel)
    assert "word = 2 * idx" in body
    assert body.count("_mul_field_left") == 1
    assert body.count("_mul_coefficient_left") == 3
    assert body.index("_mul_field_left") < body.index("_mul_coefficient_left")
    assert "out_re = out_re + d_re" in body
    assert arm.COMPLEX_ADE_PROBE_PATTERN == "c8_mul_python_float_field_left"
    assert arm.COMPLEX_ADE_PROBE_PATTERNS[:-1] == arm.PROBE_PATTERNS


@pytest.mark.parametrize("kind", ("lorentzian", "drude"))
@pytest.mark.parametrize("volume_sigma", (False, True))
@pytest.mark.parametrize("counts", ((1, 1, 1), (1, 0, 1)))
def test_reference_recurrence_is_byte_exact_for_complex_storage(
        arm, kind, volume_sigma, counts):
    fields, _pml, state = build(
        kind=kind, counts=counts, volume_sigma=volume_sigma)
    mirror = SimpleNamespace(
        P={name: array.copy() for name, array in state.P.items()},
        P_prev={name: array.copy() for name, array in state.P_prev.items()},
        _scratch=state._scratch.copy(),
    )
    for cycle in range(4):
        state.update(fields.drive_field, fields.grid.dt)
        c_now, c_prev, c_drive = state._coefficients
        for component in state.driven():
            p = mirror.P[component]
            p_prev = mirror.P_prev[component]
            scratch = mirror._scratch
            numpy.multiply(p, c_now, out=scratch)
            scratch += c_prev * p_prev
            scratch += c_drive * (state.sigma[component]
                                  * fields.drive_field(component))
            mirror.P[component] = scratch
            mirror.P_prev[component] = p
            mirror._scratch = p_prev
        for component in state.driven():
            assert mirror.P[component].tobytes() == state.P[component].tobytes(), (
                f"{kind} {component} diverged at cycle {cycle + 1}")
            assert (mirror.P_prev[component].tobytes()
                    == state.P_prev[component].tobytes())


class NumpyKernel:
    """Launch-shaped test double that executes the exact array recurrence."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(out, p, prev, sigma, drive, c_now, c_prev, c_drive, n_elem,
                   **kwargs):
            self.launches.append({"grid": grid, **kwargs})
            shape = out.shape[:-1] if out.shape[-1] == 2 else None
            out_c = out.reshape(-1).view(numpy.complex64)[:n_elem]
            p_c = p.reshape(-1).view(numpy.complex64)[:n_elem]
            q_c = prev.reshape(-1).view(numpy.complex64)[:n_elem]
            w_c = drive.reshape(-1).view(numpy.complex64)[:n_elem]
            numpy.multiply(p_c, c_now, out=out_c)
            out_c += c_prev * q_c
            out_c += c_drive * (sigma.reshape(-1)[:n_elem] * w_c
                                 if kwargs["SIGMA_IS_VOLUME"] else sigma * w_c)
            del shape
        return launch


def test_plan_resolves_rotating_pointers_on_every_launch(arm):
    reference_fields, _pml, reference = build(volume_sigma=True)
    candidate_fields, _, candidate = build(volume_sigma=True)
    kernel = NumpyKernel()
    plan = arm.ComplexAdeUpdatePPlan(
        candidate, candidate.driven(), reference_fields.grid.shape, expansion=1,
        kernel=kernel, pointer=lambda value: value)
    for cycle in range(5):
        reference.update(reference_fields.drive_field, reference_fields.grid.dt)
        plan.run(candidate_fields.drive_field, guard=False)
        for component in reference.driven():
            assert (candidate.P[component].tobytes()
                    == reference.P[component].tobytes()), (cycle, component)
            assert (candidate.P_prev[component].tobytes()
                    == reference.P_prev[component].tobytes()), (cycle, component)
    assert len(kernel.launches) == 5 * len(candidate.driven())
    assert all(row["enable_fp_fusion"] is False for row in kernel.launches)


def test_target_predicate_admits_modulo_backend_and_base_ade_refuses(arm):
    from meep_gpu.triton_kernels.coverage import ade_update_p_coverage

    fields, pml, state = build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.2,
                                         dtype=numpy.float32))
    fields.set_b_conductivity(numpy.full(fields.grid.shape, 0.15,
                                         dtype=numpy.float32))
    for component in state.driven():
        assert clean_reasons(arm, fields, pml, state, component) == []
        incumbent = ade_update_p_coverage(fields, state, component)
        assert any("complex64 storage is not carried" in reason
                   for reason in incumbent.reasons)


def test_the_fifth_probe_pattern_is_mandatory(arm):
    fields, pml, state = build()
    record = probe_record(arm)
    del record["patterns"][arm.COMPLEX_ADE_PROBE_PATTERN]
    reasons = clean_reasons(arm, fields, pml, state, probe=record)
    assert any(arm.COMPLEX_ADE_PROBE_PATTERN in reason for reason in reasons)


def test_active_pml_and_pml_storage_are_refused(arm):
    fields, pml, state = build(active_pml=True)
    reasons = clean_reasons(arm, fields, pml, state)
    assert any("active PML" in reason for reason in reasons)
    assert any("PML storage mode" in reason for reason in reasons)


def test_real_storage_is_refused(arm):
    fields, pml, state = build(complex_storage=False)
    reasons = clean_reasons(arm, fields, pml, state)
    assert any("complex64 storage is required" in reason for reason in reasons)


def test_noncontiguous_complex_history_is_refused(arm):
    fields, pml, state = build()
    state.P["Ez"] = state.P["Ez"][::-1]
    reasons = clean_reasons(arm, fields, pml, state)
    assert any("P[Ez]" in reason and "C-contiguous" in reason
               for reason in reasons)


def test_complex_sigma_volume_is_refused_as_not_float32(arm):
    fields, pml, state = build()
    state.sigma["Ez"] = numpy.ones(fields.grid.shape, dtype=numpy.complex64)
    reasons = clean_reasons(arm, fields, pml, state)
    assert any("sigma[Ez]" in reason and "not float32" in reason
               for reason in reasons)


def test_missing_stored_drive_is_refused(arm):
    fields, pml, state = build()
    fields._stored_E = False
    reasons = clean_reasons(arm, fields, pml, state)
    assert any("recomputed instead of stored" in reason for reason in reasons)


def test_builder_is_all_or_nothing_per_susceptibility(arm, monkeypatch):
    fields, pml, state = build()
    seen = []

    def verdict(_fields, _pml, _state, component, probe=None):
        seen.append(component)
        return arm.Coverage(component != "Ey", ("planted",) if component == "Ey" else ())

    monkeypatch.setattr(arm, "complex_ade_update_p_coverage", verdict)
    assert arm.plan_complex_ade_update_p(
        fields, pml, state, probe=probe_record(arm)) is None
    assert seen == ["Ex", "Ey"]
