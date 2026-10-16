"""The chunk record on the Triton table: one record, bound once, by every PML plan.

MEEP selects every PML form per structure chunk, and the kernels key their forms on
the chunk division through one ``int32`` record (``chunk_map.chunk_bounds_record``)
bound as one pointer argument, ``cb``. What these tests pin, all on a host with no
GPU and no Triton:

* :func:`launch.bind_chunk_bounds` accepts exactly the records the record ABI
  describes and refuses every other shape of input by name;
* the three ``launch.py`` plans (curl, constitutive, fused pair) and the two
  composed plans (the dispersive ``update_E`` and the scratch-weld base) refuse to
  exist without a record, and the launch plans pass it as ``cb`` directly after their
  last pointer argument, the same uploaded object on every launch;
* the four ``kernels.py`` entry points those plans launch declare ``cb`` at that
  position and read it through the forms of ``triton_kernels/forms.py``, read from the
  parse tree, at the record word offsets ``chunk_map`` writes;
* the engine route's H-side plans pass MEEP's start of H (``start_h``) on the first
  launch that has programs and on no other, and a warm pass is not that launch;
* the harness route (``*_from_arrays``) has no default record;
* the engine route takes its record from :func:`launch.chunk_bounds_for` alone: the
  division handed over, or the one-rank default;
* ``plan_step`` judges a division handed over before any arm: one that is not
  separable, or not a ``{low, interior, high}`` product where a cylindrical or
  off-diagonal run needs one, leaves every slot on the array path with the reason
  named; an admitted one reaches the root builders;
* every construction of a record-binding plan in the package and the harness passes
  ``chunk_bounds``, read from the parse tree.

The kernels module imports Triton, so the launch tests put a stand-in for it in
``sys.modules`` that records each launch's arguments. No kernel is compiled.
"""

from __future__ import annotations

import ast
import itertools
import pathlib
import sys
import types
from types import SimpleNamespace

import numpy
import pytest

from .chunk_map import (
    RECORD_PRESENCE_WORD,
    RECORD_WORDS,
    ChunkLayout,
    ChunkLayoutRefusal,
    chunk_bounds_record,
    grid_frames,
    record_index,
)
from .driver import FdtdDriver
from .grid import Grid
from .pml import PML
from .triton_kernels import coverage as coverage_module
from .triton_kernels import launch as launch_module

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

#: The plan classes that require a chunk record, by the module that defines them.
RECORD_PLANS = {
    "PmlCurlPlan", "ConstitutivePlan", "FusedPairPlan", "DispersiveConstitutivePlan",
    "ScratchWeldPairPlan", "BetaComplexFusedHdPairPlan", "BetaRealFusedHdPairPlan",
    "ComplexFusedHdPairPlan", "ConductiveBfastFusedHdPairPlan",
    "CylindricalFusedHdPairPlan", "CylindricalRealFusedHdPairPlan",
    "FoldedComplexFusedHdPairPlan", "FoldedFusedHdPairPlan",
    "FoldedOffdiagFusedElectricPairPlan", "OffdiagFusedElectricPairPlan",
    "FusedHdPairPlan", "NonlinearConstitutivePlan",
    "ConductivePmlCurlPlan", "ConductiveFusedElectricPairPlan",
    "PlainCurlPlan", "ConductivePlainCurlPlan", "NoPmlFusedElectricPairPlan",
    "ComplexConductiveNoPmlCurlPlan", "ComplexConductiveFusedPairPlan",
    "DispersiveFusedPairPlan", "FusedDispersiveChainPlan",
    "FoldedDispersiveFusedPairPlan", "FoldedOffdiagDispersiveConstitutivePlan",
}
#: The harness builders that require one.
RECORD_BUILDERS = (
    "plan_from_arrays", "plan_constitutive_from_arrays", "plan_fused_pair_from_arrays",
    "plan_dispersive_constitutive_from_arrays",
    "plan_nonlinear_constitutive_from_arrays",
    "plan_conductive_from_arrays",
    "plan_conductive_fused_electric_pair_from_arrays",
    "plan_plain_curl_from_arrays", "plan_conductive_plain_curl_from_arrays",
    "plan_no_pml_fused_electric_pair_from_arrays",
    "plan_complex_conductive_no_pml_curl_from_arrays",
    "plan_complex_conductive_fused_pair_from_arrays",
    "plan_complex_no_pml_curl_from_arrays",
    "plan_dispersive_fused_pair_from_arrays",
    "plan_folded_dispersive_fused_pair_from_arrays",
    "plan_folded_offdiag_dispersive_constitutive_from_arrays",
)
RECORD_BUILDER_SUFFIXES = ("_fused_hd_pair_from_arrays",
                           "offdiag_fused_electric_pair_from_arrays")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CASE_A_EDGES = ((-200, -158), (-158, 158), (158, 200))


def _case_a_grid():
    return Grid(resolution=20.0, cell_size=(10.0, 10.0, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic"})


def _case_a_division(split_y=None, split_x_range=None):
    """The probe's case (a) boxes (MEEP's one-rank division), optionally cut on low y.

    ``split_y`` cuts the low-y PML row at that doubled y across the whole cell;
    with ``split_x_range`` only the chunk over that x range is cut, which is how a
    forced ``num_chunks`` makes a division that is not separable.
    """
    boxes, flags = [], []
    for (x0, x1), (y0, y1) in itertools.product(_CASE_A_EDGES, _CASE_A_EDGES):
        cut = split_y is not None and (y0, y1) == (-200, -158) and (
            split_x_range is None or (x0, x1) == split_x_range)
        pieces = ([((x0, y0, 0), (x1, split_y, 0)), ((x0, split_y, 0), (x1, y1, 0))]
                  if cut else [((x0, y0, 0), (x1, y1, 0))])
        for lo, hi in pieces:
            boxes.append((lo, hi))
            flags.append(((lo[0], hi[0]) != (-158, 158),
                          (lo[1], hi[1]) in ((-200, -158), (-200, -160), (158, 200)),
                          False))
    return boxes, flags


def _case_a_layout(**cut):
    grid = _case_a_grid()
    layer = PML(grid, thickness={"x": 20, "y": 20})
    boxes, flags = _case_a_division(**cut)
    return grid, layer, ChunkLayout(boxes, flags, grid_frames(grid),
                                    faces=layer.thickness_by_face, source="meep")


@pytest.fixture
def driver():
    built = FdtdDriver(cell_size=(10.0, 10.0, 0.0), resolution=20.0, dimensions=2,
                       boundaries={"x": "metallic", "y": "metallic"})
    built.setup_pml({"x": 20, "y": 20})
    try:
        yield built
    finally:
        built.close()


class _RecordingKernel:
    """A stand-in ``@triton.jit`` object: ``kernel[grid](*args, **kwargs)`` is recorded."""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append(SimpleNamespace(kernel=self.name, grid=grid, args=args,
                                              kwargs=kwargs))
        return launch


@pytest.fixture
def launches(monkeypatch):
    """Put a recording stand-in for ``triton_kernels.kernels`` in ``sys.modules``."""
    calls = []
    module = types.ModuleType("meep_gpu.triton_kernels.kernels")
    module.ENABLE_FP_FUSION = False
    module.DEFAULT_BLOCK = 256
    for name in ("pml_curl_step", "constitutive_step", "fused_curl_constitutive_B",
                 "fused_curl_constitutive_D", "ade_update_p"):
        setattr(module, name, _RecordingKernel(name, calls))
    monkeypatch.setitem(sys.modules, "meep_gpu.triton_kernels.kernels", module)
    return calls


def _volumes(names, shape, seed=7):
    rng = numpy.random.default_rng(seed)
    return {name: rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
            for name in names}


def _flat(stems, shape, seed=11):
    rng = numpy.random.default_rng(seed)
    return {f"{stem}_{axis}": rng.uniform(0.5, 1.0, size=n).astype(numpy.float32)
            for stem in stems for axis, n in zip("xyz", shape)}


def _every_cell(shape):
    record = numpy.zeros(RECORD_WORDS, dtype=numpy.int32)
    for axis, n in enumerate(shape):
        if n > 1:
            record[RECORD_PRESENCE_WORD] |= 1 << axis
            for half in (False, True):
                record[record_index(half, axis, 1)] = n
    return record


# ---------------------------------------------------------------------------
# The record check and its upload
# ---------------------------------------------------------------------------

def test_a_one_rank_record_binds_read_only_and_is_uploaded_once():
    grid, layer, _layout = _case_a_layout()
    record = chunk_bounds_record(ChunkLayout.one_rank(grid, layer))
    host, pointer = launch_module.bind_chunk_bounds(
        list(int(v) for v in record), grid.shape, numpy.zeros(grid.shape, numpy.float32),
        "the test plan")
    assert host.dtype == numpy.int32 and numpy.array_equal(host, record)
    assert not host.flags.writeable
    assert isinstance(pointer, launch_module.CupyPointer)
    # A NumPy host never launches; its device copy is the host record itself.
    assert pointer.array is host


@pytest.mark.parametrize("label, edit, message", [
    ("no record", lambda r: None, "was given no chunk record"),
    ("short", lambda r: r[:-1], "the record is 25 int32 words"),
    ("float", lambda r: r.astype(numpy.float32), "the record is 25 int32 words"),
    ("wide", lambda r: numpy.where(numpy.arange(r.size) == 0, 2 ** 40, r.astype(numpy.int64)),
     "does not fit int32"),
    ("past the extent", lambda r: _set(r, record_index(False, 0, 1), 10_000),
     "does not fit the plan's"),
    ("negative start", lambda r: _set(r, record_index(True, 1, 0), -1), "does not fit"),
    ("inverted run", lambda r: _set(_set(r, record_index(False, 0, 0), 30),
                                    record_index(False, 0, 1), 20), "does not fit"),
    ("second slot only", lambda r: _set(_set(_set(_set(
        r, record_index(False, 0, 0), 0), record_index(False, 0, 1), 0),
        record_index(False, 0, 2), 5), record_index(False, 0, 3), 9),
     "holds its one run in the second slot"),
    ("touching runs", lambda r: _set(r, record_index(False, 0, 2),
                                     int(r[record_index(False, 0, 1)])), "overlap or touch"),
    ("presence bit 3", lambda r: _set(r, RECORD_PRESENCE_WORD, 0b1011), "outside the three"),
    ("presence cleared", lambda r: _set(r, RECORD_PRESENCE_WORD, 0b001),
     "marks y invariant"),
    ("run on an invariant axis", lambda r: _set(r, record_index(False, 2, 1), 1),
     "invariant z axis carries a run"),
])
def test_the_record_check_refuses_by_name(label, edit, message):
    grid, layer, _layout = _case_a_layout()
    record = chunk_bounds_record(ChunkLayout.one_rank(grid, layer))
    with pytest.raises(ChunkLayoutRefusal, match=message):
        launch_module.bind_chunk_bounds(edit(record.copy()), grid.shape,
                                        numpy.zeros(1, numpy.float32), "the test plan")


def _set(record, index, value):
    record = numpy.array(record)
    record[index] = value
    return record


def test_the_record_check_takes_the_plans_extent_not_the_records():
    grid, layer, _layout = _case_a_layout()
    record = chunk_bounds_record(ChunkLayout.one_rank(grid, layer))
    with pytest.raises(ChunkLayoutRefusal, match="does not fit the plan's 100 cells"):
        launch_module.bind_chunk_bounds(record, (100, 200, 1),
                                        numpy.zeros(1, numpy.float32), "the test plan")


# ---------------------------------------------------------------------------
# The engine route's one source
# ---------------------------------------------------------------------------

def test_the_engine_route_binds_the_given_division_or_the_one_rank_default():
    grid, layer, layout = _case_a_layout(split_y=-160)
    fields = SimpleNamespace(grid=grid)
    default = launch_module.chunk_bounds_for(fields, layer, None)
    assert numpy.array_equal(default, chunk_bounds_record(ChunkLayout.one_rank(grid, layer)))
    lifted = launch_module.chunk_bounds_for(fields, layer, layout)
    assert numpy.array_equal(lifted, chunk_bounds_record(layout))
    # The sliver moves exactly the y rows the lifted division moves.
    assert not numpy.array_equal(default, lifted)


def test_the_engine_route_refuses_another_grid_and_a_division_that_is_not_separable():
    grid, layer, _layout = _case_a_layout()
    other = Grid(resolution=20.0, cell_size=(12.0, 12.0, 0.0), dimensions=2)
    with pytest.raises(ChunkLayoutRefusal, match="built for another grid"):
        launch_module.chunk_bounds_for(
            SimpleNamespace(grid=grid), layer,
            ChunkLayout.one_rank(other, PML(other, thickness={"x": 40, "y": 40})))
    _grid, _layer, local = _case_a_layout(split_y=-160, split_x_range=(-158, 158))
    with pytest.raises(ChunkLayoutRefusal, match="not separable along y"):
        launch_module.chunk_bounds_for(SimpleNamespace(grid=grid), layer, local)


def test_the_engine_builders_bind_the_record_of_the_division_they_are_given(
        driver, launches, monkeypatch):
    """Coverage refuses a NumPy host, so it is stood aside; the binding is the subject."""
    admitted = coverage_module.Coverage(True, ())
    for name in ("pml_curl_coverage", "constitutive_coverage", "fused_pair_coverage"):
        monkeypatch.setattr(launch_module, name, lambda *args, **kwargs: admitted)
    fields, pml = driver.fields, driver.pml
    one_rank = driver.chunk_bounds_record()
    _grid, _layer, lifted = _case_a_layout(split_y=-160)
    builders = (
        lambda layout: launch_module.plan_pml_curl(fields, pml, "step_B",
                                                   chunk_layout=layout),
        lambda layout: launch_module.plan_constitutive(fields, pml, "H",
                                                       chunk_layout=layout),
        lambda layout: launch_module.plan_fused_pair(fields, pml, "B", (),
                                                     chunk_layout=layout),
    )
    for build in builders:
        assert numpy.array_equal(build(None).chunk_bounds, one_rank)
        assert numpy.array_equal(build(lifted).chunk_bounds, chunk_bounds_record(lifted))


# ---------------------------------------------------------------------------
# The launch: cb directly after the last pointer argument, one object every launch
# ---------------------------------------------------------------------------

SHAPE = (6, 5, 4)


def _curl_plan(record):
    arrays = _volumes(("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez"),
                      SHAPE)
    return launch_module.plan_from_arrays("step_B", arrays, _flat(("kms", "sinv"), SHAPE),
                                          (0, 1, 0), 0.35, chunk_bounds=record)


def _constitutive_plan(record):
    arrays = _volumes(("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz"),
                      SHAPE)
    return launch_module.plan_constitutive_from_arrays(
        "H", arrays, _flat(("kps", "kms"), SHAPE), chunk_bounds=record)


def _fused_plan(record, pair):
    names = (("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez",
              "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz") if pair == "B" else
             ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz",
              "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
              "inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez"))
    return launch_module.plan_fused_pair_from_arrays(
        pair, _volumes(names, SHAPE), _flat(("kms", "sinv"), SHAPE),
        _flat(("kps", "kms"), SHAPE, seed=13), (0, 1, 0), (False, True, False), 0.35,
        chunk_bounds=record)


#: Each ``launch.py`` plan, the ``kernels.py`` entry point it launches, and the number
#: of pointer arguments before ``cb``: the one position both sides of a launch pin.
LAUNCH_SITES = [
    # targets, auxiliaries, sources, six coefficients
    ("curl", _curl_plan, "pml_curl_step", 15),
    # targets, auxiliaries, sources, three inverse-epsilon placeholders, six coefficients
    ("constitutive H", _constitutive_plan, "constitutive_step", 18),
    # curl half 3 + 3 + 3 + 6, constitutive targets and auxiliaries 3 + 3, coefficients 6
    ("fused pair B", lambda record: _fused_plan(record, "B"), "fused_curl_constitutive_B", 27),
    # ... with the three inverse-epsilon volumes before the constitutive coefficients
    ("fused pair D", lambda record: _fused_plan(record, "D"), "fused_curl_constitutive_D", 30),
]


@pytest.mark.parametrize("label, build, kernel, pointers", LAUNCH_SITES)
def test_each_launch_plan_passes_cb_after_its_last_pointer(label, build, kernel, pointers,
                                                           launches):
    record = _every_cell(SHAPE)
    plan = build(record)
    assert numpy.array_equal(plan.chunk_bounds, record)
    plan.run()
    plan.run()
    first, second = launches
    for call in (first, second):
        assert call.kernel == kernel, label
        assert all(isinstance(arg, launch_module.CupyPointer)
                   for arg in call.args[:pointers + 1]), label
        assert call.args[pointers] is plan._cb  # noqa: SLF001 - the binding is the subject
        assert numpy.array_equal(call.args[pointers].array, record)
        assert call.args[pointers + 1] == SHAPE[0]  # nx: the first scalar follows cb
    # Uploaded once: both launches pass the one bound object.
    assert first.args[pointers] is second.args[pointers]
    assert f"chunk_bounds={record.tolist()}" in repr(plan)


def _kernel_definitions():
    """The ``@triton.jit`` functions of ``kernels.py``, by name, from its parse tree."""
    tree = ast.parse((HERE / "triton_kernels" / "kernels.py").read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


@pytest.mark.parametrize("label, build, kernel, pointers", LAUNCH_SITES)
def test_each_entry_point_declares_cb_where_its_plan_passes_it(label, build, kernel,
                                                               pointers):
    """The kernel side of the launch position above: ``cb``, then ``nx``, and READ.

    Triton binds a launch's arguments by position, so a plan that passes ``cb`` to an
    entry point that does not declare it is refused at binding. Every entry point now
    reads it: the forms of :mod:`~meep_gpu.triton_kernels.forms` are keyed on the
    record's membership lines, and the curl's expression on its presence word, so a
    body with no read of ``cb`` would be a kernel that serves one chunk's form to the
    whole grid.
    """
    function = _kernel_definitions()[kernel]
    names = [arg.arg for arg in function.args.posonlyargs + function.args.args]
    assert names.count("cb") == 1, (label, names)
    assert names.index("cb") == pointers, (label, names.index("cb"), pointers)
    assert names[pointers + 1] == "nx", (label, names[pointers + 1])
    reads = [node.lineno for node in ast.walk(function)
             if isinstance(node, ast.Name) and node.id == "cb"]
    assert len(reads) >= 3, f"{kernel} reads cb at only {len(reads)} site(s): {reads}"
    calls = {node.func.id for node in ast.walk(function)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "record_in" in calls, (label, sorted(calls))
    if kernel != "constitutive_step":
        assert {"curl_stencil", "curl_form"} <= calls, (label, sorted(calls))
    else:
        assert "constitutive_form" in calls, (label, sorted(calls))


def test_the_kernels_record_constants_are_chunk_maps_own_numbers():
    """``RECORD_HALF_INTEGER`` and ``RECORD_PRESENCE`` are restated in the kernels.

    A ``@triton.jit`` body may not read a plain module global, so the two word offsets
    the kernels index the record with are ``tl.constexpr`` constants of their own
    module. Restated means drift is possible, which is what this pins: the word a
    kernel reads has to be the word ``chunk_map`` wrote.
    """
    source = (HERE / "triton_kernels" / "kernels.py").read_text(encoding="utf-8")
    assert f"\nRECORD_HALF_INTEGER = tl.constexpr({record_index(True, 0, 0)})\n" in source
    assert f"\nRECORD_PRESENCE = tl.constexpr({RECORD_PRESENCE_WORD})\n" in source


def _admit_host_arrays(monkeypatch):
    """Let the engine-route builders take NumPy arrays: drop the array-module clause only.

    Coverage refuses a host array module by name, which is right for dispatch and in the
    way here: these tests read what a plan passes to its kernel, and the recording
    stand-in launches nothing. Every other clause of the predicate still applies.
    """
    original = coverage_module._grid_reasons
    monkeypatch.setattr(
        coverage_module, "_grid_reasons",
        lambda fields, pml, grid: [reason for reason in original(fields, pml, grid)
                                   if not reason.startswith("array module is")])


@pytest.fixture
def real_driver():
    """The ``driver`` fixture's 2-D run with float32 storage, which the real kernels take."""
    built = FdtdDriver(cell_size=(10.0, 10.0, 0.0), resolution=20.0, dimensions=2,
                       boundaries={"x": "metallic", "y": "metallic"},
                       force_complex_fields=False)
    built.setup_pml({"x": 20, "y": 20})
    try:
        yield built
    finally:
        built.close()


#: Where ``start_h`` sits in each H-side launch, counted from ``cb``: after the
#: scalars ``nx, ny, nz, n_elem`` (and the curl's ``dtdx`` on the fused pair).
START_H_AFTER_CB = {"constitutive_step": 5, "fused_curl_constitutive_B": 6}


@pytest.mark.parametrize("build, kernel, pointers", [
    (lambda built: launch_module.plan_constitutive(built.fields, built.pml, "H"),
     "constitutive_step", 18),
    # A run with no sources: the pair's predicate refuses an unstated source list.
    (lambda built: launch_module.plan_fused_pair(built.fields, built.pml, "B",
                                                 sources=()),
     "fused_curl_constitutive_B", 27),
])
def test_the_engine_route_starts_h_once_and_a_warm_pass_starts_nothing(
        build, kernel, pointers, real_driver, launches, monkeypatch):
    """MEEP's start of H is ONE launch, and the warm pass's empty launch is not it.

    MEEP starts H as a copy of B on its own layer's cells at the run's first
    ``update_eh(H_stuff)`` (``stepping.separate_h_from_b``). A dispatched ``update_H``
    replaces that call, so the plan passes ``start_h = 1`` while ``Fields.h_separated``
    is false and sets the flag after the launch. The warm pass
    (``fastpath.warm_plan``) runs the same ``run()`` over an EMPTY grid before step 1:
    it must compile the same program -- so it passes the same ``start_h`` -- and must
    not set the flag, or step 1 would skip the start it never performed.
    """
    from . import fastpath  # noqa: PLC0415

    _admit_host_arrays(monkeypatch)
    fields = real_driver.fields
    assert fields.h_separated is False
    plan = build(real_driver)
    assert plan is not None, "the engine-route builder refused the host driver"
    assert fastpath.warm_plan(plan) is None
    assert fields.h_separated is False, "the warm pass started H"
    plan.run()
    assert fields.h_separated is True
    plan.run()
    assert [call.kernel for call in launches] == [kernel] * 3
    assert launches[0].grid == (0,) and launches[1].grid != (0,)
    starts = [call.args[pointers + START_H_AFTER_CB[kernel]] for call in launches]
    assert starts == [1, 1, 0], starts
    for call in launches:
        assert call.args[pointers] is plan._cb  # noqa: SLF001 - the anchor of the count


def test_the_e_side_and_the_harness_route_never_start_h(real_driver, launches,
                                                         monkeypatch):
    """``start_h`` is 0 on the E side, and 0 on every plan built from bare arrays.

    The E side has no H to start; the from-arrays route (gates, batteries) steps a
    state that has already started. Neither may touch ``Fields.h_separated``.
    """
    _admit_host_arrays(monkeypatch)
    fields = real_driver.fields
    plan = launch_module.plan_constitutive(fields, real_driver.pml, "E")
    assert plan is not None, "the engine-route builder refused the host driver"
    plan.run()
    assert fields.h_separated is False
    _constitutive_plan(_every_cell(SHAPE)).run()
    _fused_plan(_every_cell(SHAPE), "B").run()
    first, second, third = launches
    assert first.args[18 + START_H_AFTER_CB["constitutive_step"]] == 0
    assert second.args[18 + START_H_AFTER_CB["constitutive_step"]] == 0
    assert third.args[27 + START_H_AFTER_CB["fused_curl_constitutive_B"]] == 0


def test_the_harness_route_has_no_default_record(launches):
    arrays = _volumes(("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez"),
                      SHAPE)
    flat = _flat(("kms", "sinv"), SHAPE)
    with pytest.raises(TypeError, match="chunk_bounds"):
        launch_module.plan_from_arrays("step_B", arrays, flat, (0, 1, 0), 0.35)
    with pytest.raises(ChunkLayoutRefusal, match="PmlCurlPlan was given no chunk record"):
        launch_module.plan_from_arrays("step_B", arrays, flat, (0, 1, 0), 0.35,
                                       chunk_bounds=None)
    with pytest.raises(TypeError, match="chunk_bounds"):
        launch_module.plan_constitutive_from_arrays(
            "H", _volumes(("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                           "Bx", "By", "Bz"), SHAPE), _flat(("kps", "kms"), SHAPE))
    with pytest.raises(ChunkLayoutRefusal, match="FusedPairPlan was given no chunk record"):
        _fused_plan(None, "B")


def test_the_nonlinear_harness_route_has_no_default_record():
    from .triton_kernels import nonlinear_update_e

    arrays = _volumes(("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez", "Dx", "Dy", "Dz",
                       "inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez"), SHAPE)
    with pytest.raises(TypeError, match="chunk_bounds"):
        nonlinear_update_e.plan_nonlinear_constitutive_from_arrays(
            arrays, _flat(("kps", "kms"), SHAPE), {"Ez": 0.045}, {"Ez": 0.06}, (0, 1, 0))


def test_the_composed_plans_refuse_to_exist_without_a_record():
    from .triton_kernels.dispersive_update_e import (
        DispersiveConstitutivePlan,
        StaticPoleBinding,
    )
    from .triton_kernels.offdiag_scratch_weld import ScratchWeldPairPlan

    names = ("Hx", "Hy", "Hz")
    twins = _volumes(names, SHAPE)
    stand = SimpleNamespace(**_volumes(names, SHAPE, seed=3))
    with pytest.raises(TypeError, match="chunk_bounds"):
        ScratchWeldPairPlan("t", stand, twins, names, SHAPE, 8, ("update_H",))
    weld = ScratchWeldPairPlan("t", stand, twins, names, SHAPE, 8, ("update_H",),
                               chunk_bounds=_every_cell(SHAPE))
    assert numpy.array_equal(weld.chunk_bounds, _every_cell(SHAPE))
    assert weld._cb.array is weld.chunk_bounds  # noqa: SLF001
    with pytest.raises(ChunkLayoutRefusal, match="the t weld plan was given no chunk record"):
        ScratchWeldPairPlan("t", stand, twins, names, SHAPE, 8, ("update_H",),
                            chunk_bounds=None)
    volumes = _volumes(("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez", "Dx", "Dy", "Dz",
                        "ie_x", "ie_y", "ie_z"), SHAPE)
    arguments = (SHAPE, 256, [volumes[n] for n in ("Ex", "Ey", "Ez")],
                 [volumes[n] for n in ("f_w_Ex", "f_w_Ey", "f_w_Ez")],
                 [volumes[n] for n in ("Dx", "Dy", "Dz")],
                 [volumes[n] for n in ("ie_x", "ie_y", "ie_z")],
                 list(_flat(("kps", "kms"), SHAPE).values()),
                 StaticPoleBinding([[], [], []]))
    with pytest.raises(TypeError, match="chunk_bounds"):
        DispersiveConstitutivePlan(*arguments)
    plan = DispersiveConstitutivePlan(*arguments, chunk_bounds=_every_cell(SHAPE))
    assert numpy.array_equal(plan.chunk_bounds, _every_cell(SHAPE))


# ---------------------------------------------------------------------------
# plan_step: the division is judged first, then reaches the root builders
# ---------------------------------------------------------------------------

def test_a_division_that_is_not_separable_leaves_every_slot_on_the_array_path():
    grid, layer, local = _case_a_layout(split_y=-160, split_x_range=(-158, 158))
    plan = launch_module.plan_step(SimpleNamespace(grid=grid), layer, chunk_layout=local)
    assert plan.plans == {} and plan.selected == {}
    assert set(plan.reasons) == set(launch_module.STEP_ORDER)
    for reasons in plan.reasons.values():
        assert len(reasons) == 1
        assert reasons[0].startswith(coverage_module.NON_SEPARABLE_DIVISION)
        assert "along y (integer), y (half-integer)" in reasons[0]


def test_a_division_for_another_grid_is_refused_for_the_whole_plan():
    grid, layer, _layout = _case_a_layout()
    other = Grid(resolution=20.0, cell_size=(12.0, 12.0, 0.0), dimensions=2)
    plan = launch_module.plan_step(
        SimpleNamespace(grid=grid), layer,
        chunk_layout=ChunkLayout.one_rank(other, PML(other, thickness={"x": 40, "y": 40})))
    assert plan.plans == {}
    assert all("built for another grid" in reasons[0] for reasons in plan.reasons.values())


def test_the_low_interior_high_rule_binds_only_cylindrical_and_off_diagonal_runs():
    grid, _layer, product = _case_a_layout()
    _grid, _layer2, split = _case_a_layout(split_y=-160)
    fields = SimpleNamespace(grid=grid)
    for layout in (product, split):
        assert coverage_module.chunk_layout_reasons(
            fields, layout, cylindrical=False, offdiagonal=False) == ()
    assert coverage_module.chunk_layout_reasons(
        fields, product, cylindrical=False, offdiagonal=True) == ()
    # The cut at doubled -160 makes four y segments; r (x) keeps three.
    refused = coverage_module.chunk_layout_reasons(
        fields, split, cylindrical=False, offdiagonal=True)
    assert len(refused) == 1 and refused[0].startswith(coverage_module.NOT_LOW_INTERIOR_HIGH)
    assert "y: 4 chunk segments, more than three" in refused[0]
    assert coverage_module.chunk_layout_reasons(
        fields, split, cylindrical=True, offdiagonal=False) == ()
    # A local cut has no global segments at all.
    _g, _l, local = _case_a_layout(split_y=-160, split_x_range=(-158, 158))
    found = coverage_module.low_interior_high_reasons(local, (0, 1, 2))
    assert found == ["y: a chunk cut does not span the cross-section"]


def test_the_one_rank_division_is_always_a_low_interior_high_product():
    for thickness in ({"x": 20, "y": 20}, {"x": 20}, {"y": 5}, {"x": 100, "y": 100}):
        grid = _case_a_grid()
        layout = ChunkLayout.one_rank(grid, PML(grid, thickness=thickness))
        assert coverage_module.low_interior_high_reasons(layout, (0, 1, 2)) == [], thickness


def test_an_admitted_division_reaches_every_root_builder(driver, monkeypatch):
    _grid, _layer, layout = _case_a_layout(split_y=-160)
    seen = []
    admitted = coverage_module.Coverage(True, ())
    refused = coverage_module.Coverage(False, ("outside this test",))
    for name in dir(launch_module):
        if name.endswith("_coverage") and callable(getattr(launch_module, name)):
            monkeypatch.setattr(launch_module, name, lambda *args, **kwargs: refused)
    for name in ("pml_curl_coverage", "constitutive_coverage"):
        monkeypatch.setattr(launch_module, name, lambda *args, **kwargs: admitted)

    def record(label):
        def builder(*args, chunk_layout="<not passed>", **kwargs):
            seen.append((label, chunk_layout))
            return f"{label}-plan"
        return builder

    monkeypatch.setattr(launch_module, "plan_pml_curl", record("curl"))
    monkeypatch.setattr(launch_module, "plan_constitutive", record("constitutive"))
    plan = launch_module.plan_step(driver.fields, driver.pml, chunk_layout=layout)
    assert sorted(label for label, _ in seen) == ["constitutive", "constitutive",
                                                  "curl", "curl"]
    assert all(given is layout for _, given in seen)
    assert plan.plans["step_B"] == "curl-plan"


# ---------------------------------------------------------------------------
# Every construction passes a record (read from the parse tree)
# ---------------------------------------------------------------------------

def _unbound_constructions():
    found = []
    for path in sorted(list((ROOT / "meep_gpu").rglob("*.py"))
                       + list((ROOT / "parity" / "meep_gpu").rglob("*.py"))):
        relative = path.relative_to(ROOT).as_posix()
        # The Metal table has its own plans of the same names; this rule is Triton's.
        if "metal" in path.name or "metal_kernels" in relative or "/results/" in relative:
            continue
        if path.name == "test_triton_chunk_bounds.py":
            continue  # this file builds the refusals on purpose
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None)
            if name is None:
                continue
            wanted = (name in RECORD_PLANS or name in RECORD_BUILDERS
                      or name.endswith(RECORD_BUILDER_SUFFIXES))
            if not wanted:
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            if "chunk_bounds" in keywords or None in keywords:
                continue
            found.append(f"{relative}:{node.lineno}: {name}")
    return found


def test_every_construction_of_a_record_plan_passes_its_record():
    assert _unbound_constructions() == []
