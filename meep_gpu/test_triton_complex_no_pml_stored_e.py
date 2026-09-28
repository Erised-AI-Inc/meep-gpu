"""Host contracts for complex, pole-aware, no-PML stored-E ``update_E``."""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import sys

import numpy
import pytest

from meep_gpu import stepping
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML

pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")

MODULE_NAME = "meep_gpu.triton_kernels.complex_no_pml_stored_e"
MODULE_PATH = (
    pathlib.Path(__file__).parent / "triton_kernels" /
    "complex_no_pml_stored_e.py"
)


def _detail(value="FMA_V1"):
    return {
        "matches": {"FMA_V1": value == "FMA_V1",
                    "NAIVE": value == "NAIVE",
                    "PLANEWISE_diagnostic": False},
        "mismatch_words": {"FMA_V1": 0 if value == "FMA_V1" else 7,
                           "NAIVE": 0 if value == "NAIVE" else 7,
                           "PLANEWISE_diagnostic": 31},
        "licensable_arms_disagreement_words": 128,
        "discriminates": True,
        "classified": value,
    }


def probe_record(arm, value="FMA_V1"):
    patterns = {name: value for name in arm.PROBE_PATTERNS}
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


def build(*, poles=1, driven=("Ex", "Ey", "Ez"), kind="lorentzian",
          complex_storage=True, active_pml=False, stored_e=True, seed=73):
    grid = Grid(
        resolution=10.0, cell_size=(0.8, 0.7, 0.6), dimensions=3,
        boundaries="periodic", courant=0.35,
        k_point=(0.2, -0.1, 0.3), xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    states = []
    for index in range(poles):
        sigma = {
            component: (0.25 + 0.04 * index + 0.01 * axis
                        if component in driven else 0.0)
            for axis, component in enumerate(("Ex", "Ey", "Ez"))
        }
        state = PolarizationState(
            Susceptibility(frequency=1.0 + 0.2 * index, gamma=0.1,
                           kind=kind),
            sigma, grid,
            numpy.complex64 if complex_storage else numpy.float32)
        fields.polarizations.append(state)
        states.append(state)
    if active_pml:
        fields.enable_pml_storage()
    elif stored_e:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)

    rng = numpy.random.default_rng(seed)
    for component, displacement in (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz")):
        for name in (component, displacement):
            array = getattr(fields, name)
            if array is None:
                continue
            array.real = rng.uniform(-0.6, 0.6, grid.shape).astype(numpy.float32)
            if numpy.iscomplexobj(array):
                array.imag = rng.uniform(-0.6, 0.6,
                                         grid.shape).astype(numpy.float32)
    for state in states:
        for component in state.driven():
            for array in (state.P[component], state.P_prev[component]):
                array.real = rng.uniform(
                    -0.4, 0.4, grid.shape).astype(numpy.float32)
                if numpy.iscomplexobj(array):
                    array.imag = rng.uniform(
                        -0.4, 0.4, grid.shape).astype(numpy.float32)
    return fields, pml, states


def clean_reasons(arm, fields, pml, probe=None):
    record = probe_record(arm) if probe is None else probe
    return [reason for reason in arm.complex_stored_e_coverage(
        fields, pml, probe=record).reasons if "not cupy" not in reason]


def test_module_imports_and_answers_without_triton(arm, monkeypatch):
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
    fields, pml, _ = build()
    verdict = module.complex_stored_e_coverage(
        fields, pml, probe=probe_record(module))
    assert isinstance(verdict.covered, bool)


def test_package_does_not_eagerly_import_the_new_family():
    sys.modules.pop(MODULE_NAME, None)
    importlib.import_module("meep_gpu.triton_kernels")
    assert MODULE_NAME not in sys.modules


def test_cuda_gate_is_incremental_and_names_exact_group_i_rows(arm):
    parity = pathlib.Path(__file__).parents[1] / "parity" / "meep_gpu"
    sys.path.insert(0, str(parity))
    try:
        gate = importlib.import_module("gate_triton_complex_no_pml_stored_e")
    finally:
        sys.path.remove(str(parity))
    assert gate.GROUP_I_ROWS == (
        "TestLoadDump.test_load_dump_chunk_layout_file_3d",
        "TestLoadDump.test_load_dump_chunk_layout_sim_3d",
        "TestLoadDump.test_load_dump_structure_3d",
        "TestLoadDump.test_load_dump_structure_sharded_3d",
    )
    cases = gate.build_cases(numpy, probe_record(arm))
    assert len([row for row in cases if row["name"].startswith("product_")]) == 7
    assert len([row for row in cases if "MUTATION" in row["name"]]) == 4
    assert len([row for row in cases if "CONTROL" in row["name"]]) == 1
    source = (parity / "gate_triton_complex_no_pml_stored_e.py").read_text(
        encoding="utf-8")
    assert "flush=True" in source
    assert "os.fsync" in source
    assert "os.replace(temporary, path)" in source

    fields, pml, notes = gate.ADE.build_configuration(numpy)
    assert notes["shape"] == [35, 32, 41]
    assert notes["k_point"] == [0.4, -1.3, 0.7]
    assert notes["conductivity"] is True
    assert notes["magnetic_conductivity"] is True
    assert notes["poles"] == 1
    assert clean_reasons(arm, fields, pml) == []


def test_kernel_source_pins_word_pairs_order_and_probe_scope(arm):
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    helper = ast.unparse(functions["_subtract_complex_poles"])
    kernel = ast.unparse(functions["complex_stored_e_step"])
    assert helper.count("if NP >") == arm.MAX_POLES
    assert helper.count("real = real -") == arm.MAX_POLES
    assert helper.count("imag = imag -") == arm.MAX_POLES
    assert "word = 2 * idx" in kernel
    assert kernel.count("_subtract_complex_poles") == 3
    assert kernel.count("_mul_field_left") == 3
    assert "_mul_coefficient_left" not in kernel
    assert tuple(arm.PROBE_PATTERNS) == tuple(
        importlib.import_module(
            "meep_gpu.triton_kernels.complex_fields").PROBE_PATTERNS)


@pytest.mark.parametrize("kind", ("lorentzian", "drude"))
@pytest.mark.parametrize("poles,driven", (
    (1, ("Ex", "Ey", "Ez")),
    (2, ("Ex", "Ez")),
))
def test_reference_transcription_matches_stepping_bytes(kind, poles, driven):
    fields, pml, _ = build(kind=kind, poles=poles, driven=driven)
    expected = {}
    for component, displacement in (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz")):
        source = getattr(fields, displacement)
        states = [state for state in fields.polarizations
                  if state.drives(component)]
        if states:
            source = source.copy()
            for state in states:
                source -= state.P[component]
        expected[component] = source * fields.inverse_epsilon_for(component)
    stepping.update_E(fields, pml)
    for component, value in expected.items():
        assert getattr(fields, component).tobytes() == value.tobytes(), component


def test_order_presum_and_drop_mutations_are_armed():
    rng = numpy.random.default_rng(11)
    d = (rng.normal(size=8192).astype(numpy.float32)
         + 1j * rng.normal(size=8192).astype(numpy.float32)).astype(numpy.complex64)
    p0 = (rng.normal(size=8192).astype(numpy.float32)
          + 1j * rng.normal(size=8192).astype(numpy.float32)).astype(numpy.complex64)
    p1 = (rng.normal(size=8192).astype(numpy.float32)
          + 1j * rng.normal(size=8192).astype(numpy.float32)).astype(numpy.complex64)
    exact = (d - p0) - p1
    reverse = (d - p1) - p0
    presum = d - (p0 + p1)
    dropped = d - p0
    assert exact.tobytes() != reverse.tobytes()
    assert exact.tobytes() != presum.tobytes()
    assert exact.tobytes() != dropped.tobytes()
    # One-pole reversal is the control that must remain identical.
    assert (d - p0).tobytes() == (d - p0).tobytes()


class NumpyKernel:
    """Launch-shaped double executing the exact complex array expression."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append({"grid": grid, **kwargs})
            targets = args[0:3]
            sources = args[3:6]
            inverse = args[6:9]
            slots = args[9:33]
            n_elem = int(args[33])
            counts = (kwargs["NP0"], kwargs["NP1"], kwargs["NP2"])
            for axis in range(3):
                target = targets[axis].reshape(-1).view(numpy.complex64)[:n_elem]
                source = sources[axis].reshape(-1).view(numpy.complex64)[:n_elem]
                value = source.copy()
                for pole in slots[axis * 8:axis * 8 + counts[axis]]:
                    value -= pole.reshape(-1).view(numpy.complex64)[:n_elem]
                target[...] = value * inverse[axis].reshape(-1)[:n_elem]
        return launch


def _direct_plan(arm, fields, kernel, *, live=True):
    order = arm.poles_per_component(fields)
    if live:
        binding = arm.LiveComplexPoleBinding(fields, order)
    else:
        binding = arm.StaticComplexPoleBinding([
            [state.P[component] for state in order[component]]
            for component, _ in arm.E_TERMS
        ])
    return arm.ComplexStoredEPlan(
        fields.grid.shape,
        [getattr(fields, c) for c, _ in arm.E_TERMS],
        [getattr(fields, d) for _, d in arm.E_TERMS],
        [fields.inverse_epsilon_for(c) for c, _ in arm.E_TERMS],
        binding, expansion=1, kernel=kernel, pointer=lambda value: value,
    )


def test_plan_resolves_rotating_pointers_on_every_launch(arm):
    reference, pml, _ = build(poles=2, seed=91)
    candidate, candidate_pml, _ = build(poles=2, seed=91)
    kernel = NumpyKernel()
    plan = _direct_plan(arm, candidate, kernel)
    initial_pointers = [id(state.P["Ex"]) for state in candidate.polarizations]
    for cycle in range(5):
        stepping.update_E(reference, pml)
        plan.run(guard=False)
        for component, _ in arm.E_TERMS:
            assert (getattr(candidate, component).tobytes()
                    == getattr(reference, component).tobytes()), (cycle, component)
        stepping.update_P(reference)
        stepping.update_P(candidate)
    assert [id(state.P["Ex"]) for state in candidate.polarizations] != initial_pointers
    assert len(kernel.launches) == 5
    assert all(row["enable_fp_fusion"] is False for row in kernel.launches)
    assert candidate_pml.is_active is False


def test_static_pointers_diverge_after_ade_rotation(arm):
    reference, pml, _ = build(poles=1, seed=101)
    candidate, _, _ = build(poles=1, seed=101)
    plan = _direct_plan(arm, candidate, NumpyKernel(), live=False)
    for _ in range(3):
        stepping.update_E(reference, pml)
        plan.run(guard=False)
        stepping.update_P(reference)
        stepping.update_P(candidate)
    assert any(getattr(candidate, component).tobytes()
               != getattr(reference, component).tobytes()
               for component, _ in arm.E_TERMS)


def test_target_is_admitted_modulo_backend_and_incumbents_refuse(arm):
    from meep_gpu.triton_kernels.complex_fields import complex_constitutive_coverage
    from meep_gpu.triton_kernels.dispersive_update_e import (
        dispersive_constitutive_coverage,
    )
    from meep_gpu.triton_kernels.no_pml_stored_e import (
        stored_e_constitutive_coverage,
    )

    fields, pml, _ = build()
    fields.set_d_conductivity(numpy.full(
        fields.grid.shape, 0.2, dtype=numpy.float32))
    fields.set_b_conductivity(numpy.full(
        fields.grid.shape, 0.15, dtype=numpy.float32))
    record = probe_record(arm)
    assert clean_reasons(arm, fields, pml, record) == []
    assert any("complex64" in reason
               for reason in stored_e_constitutive_coverage(fields, pml).reasons)
    assert any("no active PML" in reason
               for reason in complex_constitutive_coverage(
                   fields, pml, "E", probe=record).reasons)
    assert any("no active PML" in reason
               for reason in dispersive_constitutive_coverage(fields, pml).reasons)


@pytest.mark.parametrize("mutation,fragment", (
    ("active", "active PML"),
    ("pml_storage", "PML storage mode"),
    ("real", "requires complex64"),
    ("unstored", "not stored"),
    ("no_poles", "at least one live pole"),
    ("offdiag", "off-diagonal"),
    ("nonlinear", "chi2/chi3"),
))
def test_adjacent_products_are_refused(arm, mutation, fragment):
    kwargs = {
        "active_pml": mutation == "active",
        "complex_storage": mutation != "real",
        "stored_e": mutation != "unstored",
        "poles": 0 if mutation == "no_poles" else 1,
    }
    fields, pml, _ = build(**kwargs)
    if mutation == "pml_storage":
        fields._pml_active = True
    elif mutation == "offdiag":
        fields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.ones(
            fields.grid.shape, dtype=numpy.float32)}}
    elif mutation == "nonlinear":
        fields._chi2_components = {"Ex": 0.1}
    reasons = clean_reasons(arm, fields, pml)
    assert any(fragment in reason for reason in reasons), reasons


def test_missing_base_probe_pattern_refuses(arm):
    fields, pml, _ = build()
    record = probe_record(arm)
    missing = arm.PROBE_PATTERNS[-1]
    del record["patterns"][missing]
    reasons = clean_reasons(arm, fields, pml, record)
    assert any(missing in reason for reason in reasons)


def test_malformed_history_and_pole_count_refuse(arm):
    fields, pml, _ = build()
    fields.polarizations[0].P_prev["Ez"] = (
        fields.polarizations[0].P_prev["Ez"][::-1])
    assert any("P_prev[Ez]" in reason and "contiguous" in reason
               for reason in clean_reasons(arm, fields, pml))

    fields, pml, _ = build(poles=arm.MAX_POLES + 1)
    assert any("MAX_POLES" in reason
               for reason in clean_reasons(arm, fields, pml))


def test_live_binding_rejects_polarization_set_mutation(arm):
    fields, _, _ = build()
    binding = arm.LiveComplexPoleBinding(
        fields, arm.poles_per_component(fields))
    fields.polarizations.reverse()
    # One state is order-invariant, so append a second state to make the identity
    # check itself observable instead of merely checking a count.
    extra_fields, _, extra_states = build(seed=92)
    fields.polarizations.append(extra_states[0])
    with pytest.raises(RuntimeError, match="polarization set"):
        binding.arrays()
