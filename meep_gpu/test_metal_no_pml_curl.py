"""Merge bar for the Metal no-absorber real-field curl family.

The Triton port established the numerical contract: with no active absorber and
no conductivity, both curl sub-steps end in the direct ``target -= curl`` update.
These tests pin the Metal port's stencil (MEEP's left-to-right bracket and its
partner-presence forms, the PML curl's own text), its chunk-record binding,
fail-closed predicate, registry composition and optional-device boundary.  The
exhaustive device comparison lives in ``parity/meep_gpu/gate_metal_no_pml_curl.py``
and reports progress per case; its host half is
``parity/meep_gpu/test_gate_metal_no_pml_curl_host.py``.
"""

from __future__ import annotations

import ast
import inspect
import os
import re

import numpy as np
import pytest

from meep_gpu.chunk_map import (
    RECORD_PRESENCE_WORD,
    ChunkLayout,
    ChunkLayoutRefusal,
    chunk_bounds_record,
)
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, chunk_bounds, coverage as metal_coverage, device
from meep_gpu.metal_kernels import no_pml_curl as family
from meep_gpu.metal_kernels import shaders
from meep_gpu.pml import PML


def _build(*, pml_cells: int = 0, stored_e: bool = False,
           boundaries="periodic", courant: float = 0.35):
    grid = Grid(resolution=8.0, cell_size=(1.25, 1.0, 0.75),
                boundaries=boundaries, courant=courant, xp=np)
    fields = Fields(grid=grid)
    shape = grid.shape
    fields.set_epsilon_volumes(
        {name: np.full(shape, value, np.float32)
         for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))},
        {name: np.full(shape, np.float32(1.0 / value), np.float32)
         for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))},
    )
    if stored_e:
        fields.enable_field_storage()
    rng = np.random.default_rng(20260817)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
    return fields, PML(grid=grid, thickness=pml_cells)


@pytest.fixture
def backend_ready(monkeypatch):
    """Isolate numerical clauses from host availability and policy clauses."""
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _record(shape):
    """The no-absorber record of a grid storing ``shape``: presence where n > 1."""
    record = np.zeros(25, dtype=np.int32)
    record[RECORD_PRESENCE_WORD] = sum(1 << axis for axis in range(3) if shape[axis] > 1)
    return record


def test_source_steps_meeps_stencil_and_the_direct_tail():
    """MEEP's left-to-right bracket with its presence forms; the previous pairwise
    grouping and the flattened one are gone."""
    for backward in (False, True):
        source = family.plain_curl_source((0, 0, 0), backward=backward, derive=True)
        for statement in family.CURL_STATEMENTS:
            assert source.count(statement) == 1, statement
        for previous in ("dtdx * ((c_y - c) + (b - b_z))",
                         "dtdx * ((a_z - a) + (c - c_x))",
                         "dtdx * ((b_x - b) + (a - a_y))",
                         "dtdx * (c_y - c + b - b_z)"):
            assert previous not in source
        assert "f0[ii] = f0[ii] - curl0;" in source
        assert "g0[ii] * e0[ii]" in source
        assert "g0[oy] * e0[oy]" in source


def test_the_stencil_is_the_pml_curls_own_text_character_for_character():
    """One stencil for the package: the no-absorber statements are lines the PML
    curl emits, on both sides and every boundary triple."""
    for codes in ((0, 0, 0), (1, 0, 1), (1, 1, 1)):
        for backward in (False, True):
            pml = shaders.curl_source(codes, backward)
            for statement in family.CURL_STATEMENTS:
                assert statement in pml, statement


def test_the_record_binds_last_and_the_block_reads_presence_only():
    for derive in (False, True):
        source = family.plain_curl_source((1, 0, 1), backward=False, derive=derive)
        slots = sorted(int(n) for n in re.findall(r"\[\[buffer\((\d+)\)\]\]", source))
        assert slots == list(range(family.RECORD_BUFFER + 1))
        assert family.RECORD_BUFFER < device.MAX_BUFFER_BINDINGS
        assert chunk_bounds.binding(family.RECORD_BUFFER) + "," in source
        assert chunk_bounds.struct_source() in source
        block = family.bounds_block()
        assert source.count(chunk_bounds.BLOCK_BEGIN) == 1
        assert block in source
        # No membership is read: no chunk of a run without an absorber has a layer.
        assert not re.search(r"\bcb[ih]_", block) and "in_i_" not in block
        assert "in_h_" not in block
        for axis in "xyz":
            assert f"const bool has_{axis} = " in block


@pytest.mark.parametrize("builder", ("plan_metal_plain_curl",
                                     "plan_metal_plain_curl_from_arrays"))
def test_both_builders_take_the_record_as_a_required_keyword(builder):
    parameter = inspect.signature(getattr(family, builder)).parameters["chunk_bounds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_harness_route_refuses_a_plan_without_the_record_before_compiling(
        sub_step, monkeypatch):
    monkeypatch.setattr(family, "compile_source",
                        lambda source: pytest.fail("a refused plan compiled"))
    shape = (5, 4, 3)
    arrays = {name: np.zeros(shape, np.float32)
              for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")}
    with pytest.raises(ChunkLayoutRefusal, match="no chunk record"):
        family.plan_metal_plain_curl_from_arrays(
            sub_step, arrays, (0, 0, 0), 0.35, device.Residency(),
            chunk_bounds=None)


def test_the_harness_route_refuses_a_presence_word_the_shape_contradicts(
        monkeypatch):
    monkeypatch.setattr(family, "compile_source",
                        lambda source: pytest.fail("a refused plan compiled"))
    shape = (5, 4, 3)
    arrays = {name: np.zeros(shape, np.float32)
              for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez")}
    record = _record(shape)
    record[RECORD_PRESENCE_WORD] &= ~4  # z marked invariant while it stores 3 cells
    with pytest.raises(ChunkLayoutRefusal, match="invariant"):
        family.plan_metal_plain_curl_from_arrays(
            "step_B", arrays, (0, 0, 0), 0.35, device.Residency(),
            chunk_bounds=record)


def test_the_engine_route_refuses_a_plan_without_the_record(backend_ready,
                                                            monkeypatch):
    monkeypatch.setattr(family, "compile_source",
                        lambda source: pytest.fail("a refused plan compiled"))
    fields, pml = _build()
    with pytest.raises(ChunkLayoutRefusal, match="no chunk record"):
        family.plan_metal_plain_curl(fields, pml, "step_B", device.Residency(),
                                     chunk_bounds=None)


@pytest.mark.parametrize("dimensions,cell", ((3, (1.25, 1.0, 0.75)),
                                             (2, (1.25, 1.0, 0.0)),
                                             (1, (0.0, 0.0, 2.0))))
def test_the_no_absorber_record_is_presence_alone(dimensions, cell):
    """What the composer binds on these runs: no run on either lattice, and the
    presence bit exactly where the axis is not invariant."""
    grid = Grid(resolution=8.0, cell_size=cell, courant=0.35, xp=np,
                dimensions=dimensions)
    record = chunk_bounds_record(ChunkLayout.one_rank(grid, PML(grid=grid, thickness=0)))
    assert not record[:RECORD_PRESENCE_WORD].any()
    assert int(record[RECORD_PRESENCE_WORD]) == sum(
        1 << axis for axis in range(3) if not grid.is_invariant(axis))
    words = chunk_bounds.plan_words(record, grid.shape, "test")
    assert words.tolist() == record.tolist()


def test_the_arm_hands_the_composers_record_to_the_builder(monkeypatch):
    seen = {}

    def builder(fields, pml, slot, residency, variants, *, chunk_bounds):
        seen["chunk_bounds"] = chunk_bounds
        return "plan"

    monkeypatch.setattr(family, "plan_metal_plain_curl", builder)

    class _Context:
        fields = pml = residency = None
        contract_variants = (shaders.CONTRACT_OFF,)
        chunk_bounds = object()

    assert family._arm_plan(_Context, "step_D") == "plan"
    assert seen["chunk_bounds"] is _Context.chunk_bounds


def test_stored_e_specialisation_does_not_read_inverse_epsilon():
    source = family.plain_curl_source((0, 0, 0), backward=False, derive=False)
    tree = ast.parse("x = 1")  # keep the source test independent of a compiler
    assert tree is not None
    body = source[source.index("kernel void"):]
    assert "g0[ii] * e0[ii]" not in body
    assert "float a   = g0[ii];" in body


@pytest.mark.parametrize("backward", (False, True))
@pytest.mark.parametrize("codes", ((0, 0, 0), (1, 0, 1), (1, 1, 1)))
def test_every_boundary_specialisation_has_a_guard_and_ownership_rule(codes,
                                                                       backward):
    source = family.plain_curl_source(codes, backward=backward, derive=False)
    assert "if (idx >= n_elem) { return; }" in source
    assert source.count("kernel void no_pml_curl_step") == 1
    expected_masks = sum(
        1 for _target, axis in (
            ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
            if backward else ((0, 0), (1, 1), (2, 2)))
        if codes[axis]
    )
    assert source.count("? 0.0f : curl") == expected_masks


def test_inactive_layer_admits_both_curls(backend_ready):
    fields, pml = _build()
    residency = device.Residency()
    for slot in family.SUB_STEPS:
        verdict = family.metal_plain_curl_coverage(fields, pml, slot, residency)
        assert verdict.covered, verdict.reasons


def test_no_layer_object_admits_both_curls(backend_ready):
    fields, _ = _build()
    residency = device.Residency()
    for slot in family.SUB_STEPS:
        assert family.metal_plain_curl_coverage(
            fields, None, slot, residency).covered


def test_active_layer_is_refused_by_name(backend_ready):
    fields, pml = _build(pml_cells=2)
    verdict = family.metal_plain_curl_coverage(
        fields, pml, "step_B", device.Residency())
    assert not verdict.covered
    assert any("active PML" in reason for reason in verdict.reasons), verdict


def test_residency_is_required_even_when_the_arithmetic_is_covered(backend_ready):
    fields, pml = _build()
    verdict = family.metal_plain_curl_coverage(fields, pml, "step_B", None)
    assert not verdict.covered
    assert any("residency was not declared" in reason for reason in verdict.reasons)


def test_plan_specific_volumes_make_the_no_pml_pair_resident():
    mirrored = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
    planned = ("step_B", "step_D")
    live = ("step_B", "update_H", "step_D", "update_E")
    declared = {
        "step_B": ("Bx", "By", "Bz", "Dx", "Dy", "Dz"),
        "step_D": ("Dx", "Dy", "Dz", "Bx", "By", "Bz"),
    }
    verdict = metal_coverage.residency_coverage(
        mirrored, planned, live, null=("update_H", "update_E"),
        planned_volumes=declared)
    assert verdict.covered, verdict.reasons


def test_legacy_residency_table_still_refuses_the_same_no_pml_mirror_set():
    verdict = metal_coverage.residency_coverage(
        ("Bx", "By", "Bz", "Dx", "Dy", "Dz"),
        ("step_B", "step_D"),
        ("step_B", "update_H", "step_D", "update_E"),
        null=("update_H", "update_E"))
    assert not verdict.covered
    assert any("fu_Bx" in reason or "fu_Dx" in reason
               for reason in verdict.reasons)


def test_complex_storage_is_refused_to_the_complex_family(backend_ready):
    fields, pml = _build(stored_e=True)
    fields.force_complex_fields = True
    verdict = family.metal_plain_curl_coverage(
        fields, pml, "step_B", device.Residency())
    assert not verdict.covered
    assert any("complex64 storage" in reason for reason in verdict.reasons)


def test_conductivity_is_reserved_for_the_distinct_conductive_family(backend_ready):
    fields, pml = _build()
    fields.set_d_conductivity(np.full(fields.grid.shape, 0.2, np.float32))
    b = family.metal_plain_curl_coverage(fields, pml, "step_B", device.Residency())
    d = family.metal_plain_curl_coverage(fields, pml, "step_D", device.Residency())
    assert not b.covered
    assert not d.covered
    assert any("conductivity is installed on D" in reason for reason in b.reasons)
    assert any("conductivity is installed on D" in reason for reason in d.reasons)


def test_builder_returns_none_instead_of_compiling_a_refused_case(backend_ready,
                                                                  monkeypatch):
    fields, pml = _build(pml_cells=2)
    monkeypatch.setattr(family, "compile_source",
                        lambda source: pytest.fail("refused builder compiled"))
    record = chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))
    assert family.plan_metal_plain_curl(
        fields, pml, "step_B", device.Residency(), chunk_bounds=record) is None


def test_family_is_registered_once_on_each_curl_slot():
    for slot in family.SUB_STEPS:
        matches = [spec for spec in arms.registered(slot)
                   if spec.family == family.FAMILY]
        assert len(matches) == 1
        assert matches[0].label == "no-PML curl"


def test_registry_and_lazy_package_surface_include_the_family():
    from meep_gpu.metal_kernels import registry
    import meep_gpu.metal_kernels as package

    assert "no_pml_curl" in registry.FAMILY_MODULES
    assert "no_pml_curl" in package.__all__
    assert package.no_pml_curl is family


def test_gate_exists_and_has_incremental_progress_output():
    gate = os.path.join(os.path.dirname(__file__), os.pardir, "parity", "meep_gpu",
                        "gate_metal_no_pml_curl.py")
    source = open(os.path.abspath(gate), encoding="utf-8").read()
    assert "flush=True" in source
    assert "jsonl" in source.lower()
    assert "product" in source and "mutation" in source
    assert "source_provenance" in source
    assert "specialized_metal_sources" in source
