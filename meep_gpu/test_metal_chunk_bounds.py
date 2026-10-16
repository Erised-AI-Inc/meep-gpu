"""The Metal table's chunk record: the ``ChunkBounds`` ABI, its emitter, and the refusals.

What is pinned, and against what:

* the struct is the record word for word: member ``p`` is record word ``p``
  (``chunk_map.record_index``) for all 25 positions, the numpy dtype puts it at byte
  ``4 * p``, and a ``Params`` that embeds it keeps those bytes;
* the host decode of the packed struct, read by member name, equals
  ``chunk_map.membership_from_record`` and the layout's own lines on the one-rank sweep
  and on recorded divisions;
* the emitter writes the record's formula between its two anchors, and refuses what a
  kernel could not compile;
* ``words`` refuses every malformed record by name;
* the composer and the ladder: a division the record cannot describe refuses every slot
  (``launch.plan_step``) and the whole table (``metal_dispatch.decide`` rung 4bM) by
  name, a division built for another grid is refused, a separable one reaches every arm
  with its validated record, and the driver's division travels from ``FdtdDriver.step``
  through ``plan_fast_path`` to the Metal ladder;
* the owner-decode predicate for the kernels that need the chunk itself
  (``coverage.lih_product_reasons``);
* with an Apple GPU: the emitted block, compiled and dispatched, decodes every cell of
  several divisions to the host's lines, both as its own buffer and embedded in a
  ``Params``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import itertools
import types

import numpy
import pytest

from conftest import requires_resource_skip

from . import fastpath, metal_dispatch
from . import driver as driver_module
from .chunk_map import (
    RECORD_PRESENCE_WORD,
    RECORD_WORDS,
    ChunkLayout,
    ChunkLayoutRefusal,
    chunk_bounds_record,
    grid_frames,
    membership_from_record,
    record_index,
)
from .grid import Grid
from .metal_kernels import arms, chunk_bounds, coverage, launch
from .pml import PML
from .test_chunk_map import _12x12, _case_a_division, _case_a_driver, _case_a_grid, _sweep


def _non_separable(grid=None, layer=None):
    """Case (a) with a local cut inside the low-y effort volume (``test_chunk_map``)."""
    grid = grid or _case_a_grid()
    layer = layer or PML(grid, thickness={"x": 20, "y": 20})
    boxes, flags = _case_a_division(split_y=-160, split_x_range=(-158, 158))
    return ChunkLayout(boxes, flags, grid_frames(grid), faces=layer.thickness_by_face,
                       source="meep")


def _split_y(grid=None, layer=None):
    """Case (a) with a y cut across the whole cell at doubled -160: separable."""
    grid = grid or _case_a_grid()
    layer = layer or PML(grid, thickness={"x": 20, "y": 20})
    boxes, flags = _case_a_division(split_y=-160)
    return ChunkLayout(boxes, flags, grid_frames(grid), faces=layer.thickness_by_face,
                       source="meep")


# --- the ABI ------------------------------------------------------------------------------


def test_member_p_is_record_word_p_for_all_25_positions():
    assert len(chunk_bounds.MEMBERS) == RECORD_WORDS == 25
    for half, axis, word in itertools.product((False, True), range(3), range(4)):
        position = record_index(half, axis, word)
        assert chunk_bounds.MEMBERS[position] == chunk_bounds.member_name(half, axis, word)
    assert chunk_bounds.MEMBERS[RECORD_PRESENCE_WORD] == chunk_bounds.PRESENCE_MEMBER
    assert len(set(chunk_bounds.MEMBERS)) == 25
    assert chunk_bounds.member_name(True, 1, 0) == "cbh_y_a0"
    assert chunk_bounds.member_name(False, 0, 3) == "cbi_x_b1"
    assert chunk_bounds.membership_name(True, 2, "_n") == "in_h_z_n"


def test_the_dtype_places_word_p_at_byte_4p_with_no_padding():
    dtype = chunk_bounds.record_dtype()
    assert dtype.itemsize == 4 * RECORD_WORDS == 100
    for position, name in enumerate(chunk_bounds.MEMBERS):
        field, offset = dtype.fields[name]
        assert offset == 4 * position, name
        assert field == numpy.dtype("<u4" if name == chunk_bounds.PRESENCE_MEMBER else "<i4")


def test_the_struct_source_declares_the_members_in_record_order():
    source = chunk_bounds.struct_source()
    lines = [line.strip() for line in source.splitlines()]
    assert lines[0] == "struct ChunkBounds {" and lines[-1] == "};"
    declared = [line.rstrip(";").split() for line in lines[1:-1]]
    assert [name for _, name in declared] == list(chunk_bounds.MEMBERS)
    assert [kind for kind, _ in declared] == ["int"] * 24 + ["uint"]


def test_packing_is_the_record_bytes_and_a_params_embedding_keeps_them():
    grid, layer = _12x12()
    record = chunk_bounds_record(ChunkLayout.one_rank(grid, layer))
    packed = chunk_bounds.pack(record)
    assert packed.shape == (1,) and packed.dtype == chunk_bounds.record_dtype()
    assert packed.tobytes() == record.astype("<i4").tobytes()
    params = numpy.zeros(1, dtype=numpy.dtype(
        [("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"), ("dtdx", "<f4")]
        + chunk_bounds.record_dtype_fields()))
    params[0]["cb"] = packed[0]
    offset = params.dtype.fields["cb"][1]
    assert offset % 4 == 0 and offset == 20
    assert params.tobytes()[offset:offset + 100] == record.astype("<i4").tobytes()
    assert chunk_bounds.params_member() == "    ChunkBounds cb;"
    assert chunk_bounds.params_alias() == "    constant ChunkBounds& cb = prm.cb;"


def _sweep_layouts():
    """The one-rank divisions of ``test_chunk_map``'s sweep, skipping the tables PML refuses."""
    for grid, spec in _sweep():
        try:
            layer = PML(grid, thickness=spec)
        except ValueError:
            continue
        yield ChunkLayout.one_rank(grid, layer)


def test_the_host_decode_of_the_struct_is_the_record_and_the_layout_on_the_one_rank_sweep():
    checked = tables = 0
    for layout in _sweep_layouts():
        tables += 1
        record = chunk_bounds_record(layout)
        packed = chunk_bounds.pack(record)
        for half, axis in chunk_bounds.ALL_ENTRIES:
            n = layout.frames[axis].stored
            decoded = chunk_bounds.decode(packed, half, axis, n)
            assert numpy.array_equal(decoded, membership_from_record(record, half, axis, n))
            assert numpy.array_equal(decoded, layout.membership(axis, half))
            checked += 1
    assert tables > 500 and checked == 6 * tables


def test_the_host_decode_holds_on_recorded_divisions():
    for layout in (_split_y(), ChunkLayout(*_case_a_division()[:2], grid_frames(_case_a_grid()),
                                           faces=PML(_case_a_grid(), thickness={"x": 20, "y": 20}).thickness_by_face,
                                           source="meep")):
        record = chunk_bounds_record(layout)
        for half, axis in chunk_bounds.ALL_ENTRIES:
            n = layout.frames[axis].stored
            assert numpy.array_equal(chunk_bounds.decode(record, half, axis, n),
                                     layout.membership(axis, half))


# --- the emitter --------------------------------------------------------------------------


def test_the_prologue_is_anchored_and_spells_the_record_formula():
    block = chunk_bounds.prologue(entries=[(True, 1), (False, 0)], presence=(0, 2))
    lines = block.splitlines()
    assert lines[0].strip() == chunk_bounds.BLOCK_BEGIN
    assert lines[-1].strip() == chunk_bounds.BLOCK_END
    assert block.count(chunk_bounds.BLOCK_BEGIN) == block.count(chunk_bounds.BLOCK_END) == 1
    assert "    const int cbh_y_a0 = cb.cbh_y_a0;" in lines
    assert ("    const bool in_h_y = ((j) >= cbh_y_a0 && (j) < cbh_y_a1) || "
            "((j) >= cbh_y_b0 && (j) < cbh_y_b1);") in lines
    assert ("    const bool in_i_x = ((i) >= cbi_x_a0 && (i) < cbi_x_a1) || "
            "((i) >= cbi_x_b0 && (i) < cbi_x_b1);") in lines
    assert "    const bool has_z = (cb.cb_presence & 4u) != 0u;" in lines
    assert "cbi_y" not in block and "in_h_x" not in block
    neighbour = chunk_bounds.membership_lines([(False, 2)], coordinates=("i", "j", "k + 1"),
                                              suffix="_n")
    assert neighbour == ("    const bool in_i_z_n = ((k + 1) >= cbi_z_a0 && (k + 1) < cbi_z_a1)"
                         " || ((k + 1) >= cbi_z_b0 && (k + 1) < cbi_z_b1);")


def test_the_emitter_refuses_what_a_kernel_could_not_compile():
    with pytest.raises(ValueError, match="listed twice"):
        chunk_bounds.prologue(entries=[(True, 1), (True, 1)])
    with pytest.raises(ValueError, match="axis must be 0, 1 or 2"):
        chunk_bounds.bounds_locals(entries=[(False, 3)])
    with pytest.raises(ValueError, match="one index expression per axis"):
        chunk_bounds.membership_lines(coordinates=("i", "j"))
    assert chunk_bounds.binding(30) == "constant ChunkBounds& cb [[buffer(30)]]"
    with pytest.raises(ValueError, match="at most 31 buffers"):
        chunk_bounds.binding(31)


# --- the record's validation --------------------------------------------------------------


def test_words_refuses_every_malformed_record_by_name():
    grid, layer = _12x12()
    good = chunk_bounds_record(ChunkLayout.one_rank(grid, layer))
    copy = chunk_bounds.words(good, grid.shape)
    assert copy.dtype == numpy.int32 and copy.flags["C_CONTIGUOUS"]
    assert numpy.array_equal(copy, good) and copy is not good

    cases = [
        (good[:24], "has 25 words"),
        (good.astype(numpy.float32), "holds integers"),
        (_set(good, 0, 2 ** 31), "int32"),
        (_set(good, record_index(False, 1, 1), -1), "reversed or negative"),
        (_set(_set(good, record_index(False, 0, 2), 30), record_index(False, 0, 1), 42),
         "overlap, touch or are out of order"),
        (_set(_set(good, record_index(False, 0, 0), 0), record_index(False, 0, 1), 0),
         "empty first run"),
        (_set(good, RECORD_PRESENCE_WORD, 0b1000), "outside the three axes"),
    ]
    for record, match in cases:
        with pytest.raises(ChunkLayoutRefusal, match=match):
            chunk_bounds.words(record)
    with pytest.raises(ChunkLayoutRefusal, match="stored rows"):
        chunk_bounds.words(good, (grid.shape[0] - 1, grid.shape[1], grid.shape[2]))


def _set(record, position, value):
    out = numpy.asarray(record).astype(numpy.int64).copy()
    out[position] = value
    return out


# --- the division predicates --------------------------------------------------------------


def test_chunk_division_reasons_admit_separable_divisions_and_name_the_rest():
    grid, layer = _12x12()
    assert coverage.chunk_division_reasons(ChunkLayout.one_rank(grid, layer), grid.shape) == []
    assert coverage.chunk_division_reasons(_split_y(), _case_a_grid().shape) == []
    reasons = coverage.chunk_division_reasons(_non_separable())
    assert len(reasons) == 1
    assert ("not separable along y on the integer lattice, y on the half-integer lattice"
            in reasons[0])
    assert coverage.chunk_division_reasons(None)[0].startswith("no chunk division was given")
    small = (grid.shape[0] - 1, grid.shape[1], grid.shape[2])
    assert "stored rows" in coverage.chunk_division_reasons(ChunkLayout.one_rank(grid, layer),
                                                            small)[0]


def test_lih_product_reasons_name_the_divisions_the_owner_decode_cannot_name():
    grid = _case_a_grid()
    layer = PML(grid, thickness={"x": 20, "y": 20})
    assert coverage.lih_product_reasons(ChunkLayout.one_rank(grid, layer)) == []
    # The y cut at -160 makes two adjacent segments that both carry the layer: one run
    # covers rows of two chunks.
    split = coverage.lih_product_reasons(_split_y(grid, layer), axes=(1,))
    assert any("4 chunk segments along y" in reason for reason in split), split
    # A local cut: not global planes, and not a product.
    local = coverage.lih_product_reasons(_non_separable(grid, layer))
    assert any("not planes across the whole cross-section" in r for r in local), local
    # Two interior chunks along x, no layer on x: class 1 spans two segments.
    bare = Grid(resolution=10.0, cell_size=(4.0, 4.0, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic"})
    frames = grid_frames(bare)
    lo_x, hi_x = frames[0].window
    lo_y, hi_y = frames[1].window
    middle = (lo_x + hi_x) // 2
    halves = ChunkLayout([((lo_x, lo_y, 0), (middle, hi_y, 0)), ((middle, lo_y, 0), (hi_x, hi_y, 0))],
                         [(False, False, False), (False, False, False)], frames, source="meep")
    assert coverage.chunk_division_reasons(halves) == []
    named = coverage.lih_product_reasons(halves, axes=(0,))
    assert any("owner class 1 covers rows of 2 chunk segments" in r for r in named), named
    # A wrapping axis that carries a layer.
    wrapped, wrapped_layer = _12x12()
    reasons = coverage.lih_product_reasons(ChunkLayout.one_rank(wrapped, wrapped_layer))
    assert any("x wraps and carries a layer" in r for r in reasons), reasons


# --- the composer -------------------------------------------------------------------------


def test_chunk_division_resolves_none_to_one_rank_and_validates_the_record():
    driver = _case_a_driver()
    try:
        layout, words, reasons = launch.chunk_division(driver.fields, driver.pml)
        assert reasons == () and layout.source == "one_rank"
        assert numpy.array_equal(words, driver.chunk_bounds_record())
        layout, words, reasons = launch.chunk_division(driver.fields, driver.pml, _split_y())
        assert reasons == () and numpy.array_equal(words, chunk_bounds_record(_split_y()))
        layout, words, reasons = launch.chunk_division(driver.fields, driver.pml,
                                                       _non_separable())
        assert words is None and "not separable along y" in reasons[0]
        other = Grid(resolution=20.0, cell_size=(8.0, 10.0, 0.0), dimensions=2,
                     boundaries={"x": "metallic", "y": "metallic"})
        foreign = ChunkLayout.one_rank(other, PML(other, thickness={"x": 20, "y": 20}))
        _, words, reasons = launch.chunk_division(driver.fields, driver.pml, foreign)
        assert words is None and reasons == (
            "the chunk division was built for another grid: its axis frames differ from "
            "this grid's",)
        _, words, reasons = launch.chunk_division(types.SimpleNamespace(), None)
        assert words is None and "could not be built" in reasons[0]
    finally:
        driver.close()


def test_a_non_separable_division_refuses_every_slot_before_any_arm_is_asked(monkeypatch):
    driver = _case_a_driver()
    asked = []
    monkeypatch.setattr(arms, "arms_for", lambda slot, context: asked.append(slot) or ())
    try:
        plan = launch.plan_step(driver.fields, driver.pml, sources=(), fuse=True,
                                chunk_layout=_non_separable())
        assert plan.plans == {} and asked == []
        assert set(plan.reasons) == set(launch.STEP_ORDER)
        for reasons in plan.reasons.values():
            assert len(reasons) == 1 and "not separable along y" in reasons[0]
    finally:
        driver.close()


def test_a_separable_division_reaches_every_arm_with_its_validated_record(monkeypatch):
    driver = _case_a_driver()
    seen = {}
    original = arms.arms_for

    def spy(slot, context):
        seen[slot] = (context.chunk_layout, context.chunk_bounds)
        return original(slot, context)

    monkeypatch.setattr(arms, "arms_for", spy)
    try:
        layout = _split_y()
        launch.plan_step(driver.fields, driver.pml, sources=(), chunk_layout=layout)
        assert set(seen) == set(launch.STEP_ORDER)
        for given, words in seen.values():
            assert given is layout
            assert numpy.array_equal(words, chunk_bounds_record(layout))
        seen.clear()
        launch.plan_step(driver.fields, driver.pml, sources=())
        for given, words in seen.values():
            assert given.source == "one_rank"
            assert numpy.array_equal(words, driver.chunk_bounds_record())
    finally:
        driver.close()


# --- the ladder ---------------------------------------------------------------------------


def _toolchain():
    """A toolchain that reads as available, so rung 4bM is reached on a host-only run."""
    return {"importable": True, "version": "2.10.0", "built": True, "available": True,
            "compile_shader": True, "metal_frontend": None, "architecture": None,
            "device_name": None, "fast_math": None, "machine": "arm64"}


def _decide(driver, layout, monkeypatch):
    monkeypatch.setattr(metal_dispatch, "metal_toolchain", _toolchain)
    record = fastpath._base_record()  # noqa: SLF001
    tail = []
    result = metal_dispatch.decide(record, driver.fields, driver.pml, driver.grid, (),
                                   types.SimpleNamespace(__name__="numpy"),
                                   finish=lambda **kwargs: tail.append(kwargs) or "TAIL",
                                   chunk_layout=layout)
    return record, result, tail


def test_rung_4bM_refuses_a_division_the_record_cannot_describe(monkeypatch):
    driver = _case_a_driver()
    try:
        record, result, tail = _decide(driver, _non_separable(), monkeypatch)
        assert result is None and tail == []
        assert record["decision"] == "refused" and record["step_path"] == "array"
        assert "not separable along y" in record["refused_because"]
        block = record["chunk_division"]
        assert block["source"] == "meep" and block["chunks"] == 10
        assert block["refused"] == [record["refused_because"]] and "record" not in block
    finally:
        driver.close()


class _Composed(Exception):
    pass


def test_rung_4bM_hands_a_separable_division_to_both_composer_passes(monkeypatch):
    driver = _case_a_driver()
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("chunk_layout"))
        raise _Composed

    monkeypatch.setattr(launch, "plan_step", spy)
    try:
        layout = _split_y()
        record = None
        with pytest.raises(_Composed):
            _decide(driver, layout, monkeypatch)
        assert calls == [layout]
        monkeypatch.setattr(metal_dispatch, "metal_toolchain", _toolchain)
        record = fastpath._base_record()  # noqa: SLF001
        with pytest.raises(_Composed):
            metal_dispatch.decide(record, driver.fields, driver.pml, driver.grid, (),
                                  types.SimpleNamespace(__name__="numpy"),
                                  finish=lambda **kwargs: None, chunk_layout=None)
        assert calls[-1].source == "one_rank"
        block = record["chunk_division"]
        assert block["refused"] == [] and block["source"] == "one_rank" and block["chunks"] == 9
        assert block["record"] == [int(w) for w in driver.chunk_bounds_record()]
        assert block["presence"] == 0b011
        assert block["runs"]["h_y"] == [[0, 21], [179, 200]]
    finally:
        driver.close()


def test_plan_fast_path_hands_the_division_to_the_metal_ladder_untouched(monkeypatch):
    seen = []
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "1")
    monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: True)
    monkeypatch.setattr(metal_dispatch, "decide",
                        lambda *args, **kwargs: seen.append(kwargs.get("chunk_layout")))
    marker = object()
    grid = types.SimpleNamespace(xp=numpy)
    assert fastpath.plan_fast_path(object(), object(), grid, (), chunk_layout=marker) is None
    assert seen == [marker]
    fastpath.plan_fast_path(object(), object(), grid, ())
    assert seen == [marker, None]


def test_the_driver_hands_its_own_division_to_plan_fast_path(monkeypatch):
    driver = _case_a_driver()
    seen = []

    def spy(fields, pml, grid, sources=None, chunk_layout=None):
        seen.append(chunk_layout)

    monkeypatch.setattr(driver_module, "plan_fast_path", spy)
    try:
        driver.gpu = "test-only: route the freeze through plan_fast_path"
        driver.step()
        assert seen == [driver.chunk_layout] and seen[0].source == "one_rank"
    finally:
        driver.gpu = None
        driver.close()
    lifted = _case_a_driver()
    try:
        layout = _split_y(lifted.grid, lifted.pml)
        lifted.set_chunk_layout(layout)
        lifted.gpu = "test-only: route the freeze through plan_fast_path"
        lifted.step()
        assert seen[-1] is layout
    finally:
        lifted.gpu = None
        lifted.close()


def test_chunk_bounds_is_a_fingerprinted_host_module():
    assert "chunk_bounds.py" in launch.HOST_MODULES
    assert "chunk_bounds.py" in launch.compute_fingerprints()["metal_kernels"]["host_sha256"]


# --- on an Apple GPU ----------------------------------------------------------------------


def _probe_source(embedded: bool) -> str:
    from .metal_kernels import shaders  # noqa: PLC0415

    if embedded:
        signature = ("    device int* out [[buffer(0)]],\n"
                     "    constant Params& prm [[buffer(1)]],\n")
        head = ("struct Params { uint nx; uint ny; uint nz; uint n_elem; "
                f"{chunk_bounds.STRUCT} cb; }};")
        unpack = ("    int nyi = int(prm.ny), nzi = int(prm.nz), n = int(prm.n_elem);\n"
                  + chunk_bounds.params_alias() + "\n")
    else:
        signature = ("    device int* out [[buffer(0)]],\n"
                     f"    {chunk_bounds.binding(1)},\n"
                     "    constant uint& ny [[buffer(2)]],\n"
                     "    constant uint& nz [[buffer(3)]],\n"
                     "    constant uint& n_elem [[buffer(4)]],\n")
        head = ""
        unpack = "    int nyi = int(ny), nzi = int(nz), n = int(n_elem);\n"
    bits = " | ".join(
        [f"({chunk_bounds.membership_name(half, axis)} ? {1 << (3 * int(half) + axis)} : 0)"
         for half, axis in chunk_bounds.ALL_ENTRIES]
        + [f"(has_{name} ? {1 << (6 + axis)} : 0)" for axis, name in enumerate("xyz")])
    return "\n".join([
        "#include <metal_stdlib>", "using namespace metal;",
        shaders.contraction_pragma(shaders.CONTRACT_OFF),
        chunk_bounds.struct_source(), head,
        "kernel void chunk_bounds_probe(", signature + "    uint idx [[thread_position_in_grid]])",
        "{", unpack,
        "    int ii = int(idx);", "    if (ii >= n) { return; }",
        "    int k = ii % nzi;", "    int plane = ii / nzi;", "    int j = plane % nyi;",
        "    int i = plane / nyi;",
        chunk_bounds.prologue(presence=(0, 1, 2)),
        f"    out[ii] = {bits};", "}"])


def _expected_bits(layout, record):
    shape = tuple(frame.stored for frame in layout.frames)
    bits = numpy.zeros(shape, dtype=numpy.int32)
    for half, axis in chunk_bounds.ALL_ENTRIES:
        line = membership_from_record(record, half, axis, shape[axis]).astype(numpy.int32)
        view = [1, 1, 1]
        view[axis] = shape[axis]
        bits |= line.reshape(view) << (3 * int(half) + axis)
    presence = int(record[RECORD_PRESENCE_WORD])
    for axis in range(3):
        if presence & (1 << axis):
            bits |= 1 << (6 + axis)
    return bits


@pytest.mark.requires_resource("mps_device")
def test_the_emitted_block_decodes_every_cell_on_the_device():
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        requires_resource_skip("mps_device", "PyTorch is not importable")
    if not getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        requires_resource_skip("mps_device", "this PyTorch reports no usable Apple GPU (MPS)")
    from .metal_kernels import device  # noqa: PLC0415

    layouts = [_split_y(), ChunkLayout.one_rank(*_12x12())]
    layouts.extend(itertools.islice(_sweep_layouts(), 0, None, 97))
    differing = total = 0
    for embedded in (False, True):
        library = device.compile_source(_probe_source(embedded))
        for layout in layouts:
            record = chunk_bounds_record(layout)
            shape = tuple(frame.stored for frame in layout.frames)
            n = int(numpy.prod(shape))
            out = torch.zeros(n, dtype=torch.int32, device="mps")
            if embedded:
                params = numpy.zeros(1, dtype=numpy.dtype(
                    [("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4")]
                    + chunk_bounds.record_dtype_fields()))
                params[0]["nx"], params[0]["ny"], params[0]["nz"] = shape
                params[0]["n_elem"] = n
                params[0]["cb"] = chunk_bounds.pack(record)[0]
                words = torch.from_numpy(numpy.frombuffer(params.tobytes(), numpy.int32).copy())
                library.chunk_bounds_probe(out, words.to("mps"))
            else:
                library.chunk_bounds_probe(out, chunk_bounds.tensor(record), shape[1], shape[2], n)
            torch.mps.synchronize()
            got = out.cpu().numpy().reshape(shape)
            want = _expected_bits(layout, record)
            differing += int(numpy.count_nonzero(got.view(numpy.uint32) != want.view(numpy.uint32)))
            total += n
    assert total > 0
    assert differing == 0, f"{differing} of {total} decoded cells differ from the host lines"
