"""Laptop contracts for the complex64 no-PML conductive curl family.

The merge-bar host can settle the coverage partition, per-term conductivity
reach, corpus mapping, builder bindings and source transcription.  It cannot
settle Triton lowering or bytes; the sibling CUDA gate owns those claims.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import sys

import numpy
import pytest

from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_complex_fields import _probe_record

pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")

MODULE_NAME = "meep_gpu.triton_kernels.complex_no_pml_conductive"
MODULE_PATH = pathlib.Path(__file__).parent / "triton_kernels" / (
    "complex_no_pml_conductive.py")
GROUP_I_ROWS = (
    "TestLoadDump.test_load_dump_chunk_layout_file_3d",
    "TestLoadDump.test_load_dump_chunk_layout_sim_3d",
    "TestLoadDump.test_load_dump_structure_3d",
    "TestLoadDump.test_load_dump_structure_sharded_3d",
)


class _CupyNamed:
    __name__ = "cupy"

    def __getattr__(self, name):
        return getattr(numpy, name)


@pytest.fixture(scope="module")
def arm():
    return importlib.import_module(MODULE_NAME)


def _build(
    *, d_conductivity=True, b_conductivity=True, poles=1,
    kind="lorentzian", active_pml=False, complex_storage=True,
    k_point=(0.4, -1.3, 0.7), exact_corpus_shape=False,
):
    resolution = 15.0 if exact_corpus_shape else 10.0
    cell_size = (2.3, 2.1, 2.7) if exact_corpus_shape else (0.8, 0.7, 0.6)
    grid = Grid(
        resolution=resolution, cell_size=cell_size, dimensions=3,
        boundaries="periodic", courant=0.5, k_point=k_point, xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    dtype = numpy.complex64 if complex_storage else numpy.float32
    for index in range(poles):
        state = PolarizationState(
            Susceptibility(
                frequency=1.0 + 0.1 * index, gamma=0.1, kind=kind),
            {"Ex": 0.3, "Ey": 0.2, "Ez": 0.1}, grid, dtype)
        fields.polarizations.append(state)
    shape = tuple(grid.shape)
    gradient = numpy.linspace(
        0.2, 0.7, numpy.prod(shape), dtype=numpy.float32).reshape(shape)
    if d_conductivity:
        fields.set_d_conductivity(gradient)
    if b_conductivity:
        fields.set_b_conductivity(gradient * numpy.float32(0.75))
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2 if active_pml else 0)
    grid.xp = _CupyNamed()
    return fields, pml


def _coverage(arm, fields, pml, sub_step, probe=None):
    return arm.complex_conductive_no_pml_curl_coverage(
        fields, pml, sub_step,
        probe=_probe_record() if probe is None else probe)


@pytest.mark.parametrize("row", GROUP_I_ROWS)
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_all_eight_group_i_curl_slots_are_admitted(arm, row, sub_step):
    fields, pml = _build(exact_corpus_shape=True)
    assert tuple(fields.grid.shape) == (35, 32, 41), row
    verdict = _coverage(arm, fields, pml, sub_step)
    assert verdict.covered, (row, sub_step, verdict.reasons)
    assert verdict.reasons == ()


@pytest.mark.parametrize("kind", ("lorentzian", "drude"))
@pytest.mark.parametrize("poles", (1, 2))
@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_supported_poles_are_present_but_do_not_change_curl_coverage(
    arm, kind, poles, sub_step,
):
    fields, pml = _build(kind=kind, poles=poles)
    assert len(fields.polarizations) == poles
    assert _coverage(arm, fields, pml, sub_step).covered


def test_d_sigma_cannot_reach_step_b(arm):
    from meep_gpu.triton_kernels.no_pml_conductive import (
        conductive_no_pml_targets,
    )

    fields, pml = _build(d_conductivity=True, b_conductivity=False)
    assert conductive_no_pml_targets(fields, "step_B") == (False, False, False)
    assert conductive_no_pml_targets(fields, "step_D") == (True, True, True)
    b_plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, "step_B", probe=_probe_record())
    d_plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, "step_D", probe=_probe_record())
    assert b_plan is not None and b_plan.cond == (0, 0, 0)
    assert d_plan is not None and d_plan.cond == (1, 1, 1)


def test_b_sigma_cannot_reach_step_d(arm):
    fields, pml = _build(d_conductivity=False, b_conductivity=True)
    b_plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, "step_B", probe=_probe_record())
    d_plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, "step_D", probe=_probe_record())
    assert b_plan is not None and b_plan.cond == (1, 1, 1)
    assert d_plan is not None and d_plan.cond == (0, 0, 0)


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_predicate_is_disjoint_from_both_incumbent_storage_families(
    arm, sub_step,
):
    from meep_gpu.triton_kernels import complex_no_pml_curl
    from meep_gpu.triton_kernels import no_pml_conductive

    fields, pml = _build()
    target = _coverage(arm, fields, pml, sub_step)
    lossless_complex = complex_no_pml_curl.complex_no_pml_curl_coverage(
        fields, pml, sub_step, probe=_probe_record())
    real_conductive = no_pml_conductive.conductive_plain_curl_coverage(
        fields, pml, sub_step)
    assert target.covered
    assert not lossless_complex.covered
    assert any("conductivity is installed" in reason
               for reason in lossless_complex.reasons)
    assert not real_conductive.covered
    assert any("complex64 storage" in reason for reason in real_conductive.reasons)


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_active_pml_and_absent_conductivity_are_refused_by_name(arm, sub_step):
    fields, pml = _build(active_pml=True)
    active = _coverage(arm, fields, pml, sub_step)
    assert not active.covered
    assert any("active PML layer" in reason for reason in active.reasons)

    fields, pml = _build(d_conductivity=False, b_conductivity=False)
    absent = _coverage(arm, fields, pml, sub_step)
    assert not absent.covered
    assert any("no curl target carries a conductivity" in reason
               for reason in absent.reasons)


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_expansion_probe_is_a_required_platform_license(arm, sub_step):
    fields, pml = _build()
    verdict = arm.complex_conductive_no_pml_curl_coverage(
        fields, pml, sub_step, probe={})
    assert not verdict.covered
    assert any("expansion probe" in reason for reason in verdict.reasons)


def test_a_malformed_conductivity_pair_is_refused(arm, monkeypatch):
    fields, pml = _build()
    real_reader = fields.condinv_for
    monkeypatch.setattr(
        fields, "condinv_for",
        lambda name: None if name == "Dx" else real_reader(name))
    verdict = _coverage(arm, fields, pml, "step_D")
    assert not verdict.covered
    assert any("Dx has only one of condfac/condinv" in reason
               for reason in verdict.reasons)


def test_builder_binds_live_engine_sources_for_both_curls(arm):
    fields, pml = _build()
    b_plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, "step_B", probe=_probe_record())
    d_plan = arm.plan_complex_conductive_no_pml_curl(
        fields, pml, "step_D", probe=_probe_record())
    assert b_plan is not None and d_plan is not None
    assert [pointer.array.base for pointer in b_plan._sources] == [
        fields.Ex, fields.Ey, fields.Ez]
    # With no PML, get_H serves B itself; H storage is intentionally absent.
    assert fields.Hx is None and fields.Hy is None and fields.Hz is None
    assert [pointer.array.base for pointer in d_plan._sources] == [
        fields.Bx, fields.By, fields.Bz]


def test_kernel_source_preserves_the_normative_stencil_and_three_pass_tail():
    raw = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    kernel = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "complex_conductive_no_pml_curl_step")
    source = ast.unparse(kernel)
    assert "t0_re = ((c_y_re - c_re) + (b_re - b_z_re))" in raw
    assert "_mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)" in source
    assert source.count("_rotate_field_left") == 6
    # Two field-left conductive multiplies per target: condfac before subtract,
    # condinv after.  Lossless targets compile both away.
    assert source.count("_mul_field_left") == 6
    for index in range(3):
        first = source.index(f"factor = tl.load(cf{index} + idx")
        subtract = source.index(f"v{index}_re = v{index}_re - curl{index}_re")
        inverse = source.index(f"inverse = tl.load(ci{index} + idx")
        assert first < subtract < inverse


def test_kernel_prefix_is_ast_identical_to_the_certified_complex_curl():
    """Prove the Yee neighbours/phase/mask were reused, not reinterpreted."""
    from meep_gpu.triton_kernels import complex_fields

    candidate_tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    incumbent_tree = ast.parse(
        pathlib.Path(complex_fields.__file__).read_text(encoding="utf-8"))
    candidate = next(
        node for node in candidate_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "complex_conductive_no_pml_curl_step")
    incumbent = next(
        node for node in incumbent_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "bloch_pml_curl_step")

    def assignment_name(statement):
        if not isinstance(statement, ast.Assign):
            return None
        target = statement.targets[0]
        return target.id if isinstance(target, ast.Name) else None

    candidate_prefix = candidate.body[1:next(
        index for index, statement in enumerate(candidate.body)
        if assignment_name(statement) == "v0_re")]
    incumbent_prefix = incumbent.body[1:next(
        index for index, statement in enumerate(incumbent.body)
        if assignment_name(statement) == "km_x")]
    assert [ast.dump(node, include_attributes=False) for node in candidate_prefix] == [
        ast.dump(node, include_attributes=False) for node in incumbent_prefix]


def test_three_pass_complex_array_reference_is_not_the_flattened_formula():
    """A numerical pin for the array path's sequential rounding boundary."""
    rng = numpy.random.default_rng(20260816)
    target = (
        rng.uniform(-4, 4, 4096).astype(numpy.float32)
        + 1j * rng.uniform(-4, 4, 4096).astype(numpy.float32)).astype(numpy.complex64)
    curl = (
        rng.uniform(-4, 4, 4096).astype(numpy.float32)
        + 1j * rng.uniform(-4, 4, 4096).astype(numpy.float32)).astype(numpy.complex64)
    factor = rng.uniform(0.1, 0.9, 4096).astype(numpy.float32)
    inverse = rng.uniform(0.8, 1.3, 4096).astype(numpy.float32)
    sequential = target.copy()
    sequential *= factor
    sequential -= curl
    sequential *= inverse
    # A fused/single-rounded algebraic rewrite is not the array program: the
    # array program commits complex64 bytes after each of its three ufuncs.
    flattened = (
        (target.astype(numpy.complex128) * factor.astype(numpy.float64)
         - curl.astype(numpy.complex128))
        * inverse.astype(numpy.float64)).astype(numpy.complex64)
    assert numpy.count_nonzero(
        sequential.view(numpy.uint32) != flattened.view(numpy.uint32)) > 0


def test_module_remains_importable_when_triton_is_absent(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("blocked by host-contract test")
        return real_import(name, *args, **kwargs)

    for name in tuple(sys.modules):
        if name.startswith("triton"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    module = importlib.import_module(MODULE_NAME)
    assert hasattr(module, "complex_conductive_no_pml_curl_coverage")
