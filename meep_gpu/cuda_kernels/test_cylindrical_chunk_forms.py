"""The CUDA cylindrical family transcribes the array path's Dcyl forms, read off its text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The column-serial radial scan
(``cylindrical_prefix``, bit_parity_0_10 DESIGN.md D26), the real m = 0 curl pair
(``cylindrical_kernels``) and the complex curl pair at every m class
(``cylindrical_complex_kernels``) are each LAUNCHED THROUGH THEIR OWN LAUNCHERS (loaded with
NumPy wearing CuPy's name), whose compiled kernels are replaced by an interpreter that
EXECUTES the emitted device text in float32 NumPy -- its helpers too, from their own
text. Every array of the ``Fields`` is compared word for word (sign of zero included) with
the array path's own sub-step (``stepping.step_B`` / ``step_D``) on a copy of the same
seeded full-volume state, and the scan alone with ``stepping.cylindrical_rderiv_difference``.

The cases cover the one-rank division of a Dcyl grid under an r layer and z layers (two
chunks along r, so the radial sum restarts at the layer's cut), a lifted division with a
third r chunk, a periodic z axis, a Courant number that is not a float32 word (0.3, where
MEEP's double ``Courant`` products are another word than a float32 factor), the accurate
near-axis branch, and planted signed zeros. Each case asserts that the sub-step moved its
field (a vacuity floor). A battery of planted text defects -- each one rewrite of an
emitted text -- must change a word.

THE INTERPRETER is the complex lane's (``test_complex_chunk_forms.ComplexInterpreter``),
extended here by a ``for`` loop with uniform bounds and the scan's one-lane-per-column
launch. Anything else still raises.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy
import pytest

from .. import part_products, stepping
from ..chunk_map import ChunkLayout, RegionPlan, chunk_bounds_record, grid_frames, region_backend
from ..fields import Fields
from ..grid import Grid
from ..pml import PML
from . import bounds_block, cylindrical_complex_kernels, cylindrical_prefix
from .test_complex_chunk_forms import ComplexInterpreter, InterpretedKernel
from .test_root_chunk_forms import F32, Unmodelled, _NumpyWearingCupysName, _items

HERE = pathlib.Path(__file__).parent
# =============================================================================
# The interpreter: a for loop with uniform bounds, one lane per column
# =============================================================================

def _scalar(value: Any) -> int:
    array = numpy.asarray(value)
    if array.ndim != 0:
        flat = array.reshape(-1)
        if flat.size == 0 or not numpy.all(flat == flat[0]):
            raise Unmodelled("a for bound that is not uniform across the lanes")
        array = flat[0]
    return int(array)


def _block_end(statements: List[str], position: int) -> int:
    depth = 0
    for end in range(position, len(statements)):
        if statements[end].endswith("{"):
            depth += 1
        elif statements[end] == "}":
            depth -= 1
            if depth == 0:
                return end
    raise Unmodelled("an unclosed block")


class LocalFloatArray:
    """A lane-local ``float name[n];``, passed to a helper by name (a pointer parameter)."""

    def __init__(self, size: int):
        self.items = [F32(numpy.nan) for _ in range(size)]


class ScanInterpreter(ComplexInterpreter):
    """The complex interpreter with ``for (int v = a; v < b; ++v) { ... }`` loops whose
    bounds are uniform across the lanes, the scan's column guard, and lane-local
    ``float name[n];`` arrays (the H->D welds' out-parameters)."""

    _FOR = re.compile(r"for \(int (\w+) = (.+); \1 < (.+); \+\+\1\)")

    def statement(self, statement: str, frame: Dict[str, Any]) -> None:
        if statement == "if (idx >= ncol) return;":
            return  # every lane of the launch is a column by construction
        match = re.fullmatch(r"float (\w+)\[(\d+)\];", statement)
        if match:
            frame[match.group(1)] = LocalFloatArray(int(match.group(2)))
            return
        super().statement(statement, frame)

    def _eval(self, node, frame):
        import ast  # noqa: PLC0415

        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            address = name.startswith("_addr_")
            owner = self._owner(name[len("_addr_"):] if address else name, frame)
            target = None if owner is None else owner[name[len("_addr_"):] if address
                                                       else name]
            if isinstance(target, LocalFloatArray):
                slot = int(self._eval(node.slice, frame))
                if address:
                    from .test_complex_chunk_forms import ElementRef  # noqa: PLC0415

                    return ElementRef(target, slot)
                return target.items[slot]
        return super()._eval(node, frame)

    def _store(self, frame, target, index, value):
        owner = self._owner(target, frame)
        if owner is not None and isinstance(owner[target], LocalFloatArray):
            slot = int(self.value(index, frame))
            array = owner[target]
            array.items[slot] = numpy.where(self._active(frame), value, array.items[slot])
            return
        super()._store(frame, target, index, value)

    def call(self, name, args, caller):
        from .test_complex_chunk_forms import CFArray, ElementRef  # noqa: PLC0415
        from .test_root_chunk_forms import _Pointer, _Register  # noqa: PLC0415

        function = self.functions[name]
        if len(args) != len(function["params"]):
            raise Unmodelled(f"{name}: {len(args)} arguments for {len(function['params'])}")
        frame: Dict[str, Any] = {"__pred__": self._active(caller)}
        for (param, pointer), arg in zip(function["params"], args):
            value = self._eval(arg, caller)
            kinds = (_Pointer, _Register, CFArray, ElementRef, LocalFloatArray)
            if pointer != isinstance(value, kinds):
                raise Unmodelled(f"{name}: {param} bound to {type(value).__name__}")
            frame[param] = value
        frame["__ret__"] = None
        frame["__done__"] = numpy.zeros((), dtype=bool)
        self.run(_items(function["body"]), frame)
        return frame["__ret__"]

    def run(self, statements: List[str], frame: Dict[str, Any]) -> None:
        pending: List[str] = []
        position = 0
        while position < len(statements):
            statement = statements[position]
            head = statement[:-1].strip() if statement.endswith("{") else None
            loop = self._FOR.fullmatch(head) if head is not None else None
            if loop is not None:
                if pending:
                    super().run(pending, frame)
                    pending = []
                end = _block_end(statements, position)
                body = statements[position + 1:end]
                name, start, stop = loop.groups()
                low = _scalar(self.value(start, frame))
                high = _scalar(self.value(stop, frame))
                for value in range(low, high):
                    child = {"__parent__": frame, "__pred__": frame["__pred__"],
                             name: numpy.int64(value)}
                    self.run(body, child)
                position = end + 1
                continue
            if head is not None:
                end = _block_end(statements, position)
                after = end + 1
                if after < len(statements) and statements[after] == "else {":
                    after = _block_end(statements, after) + 1
                pending.extend(statements[position:after])
                position = after
                continue
            pending.append(statement)
            position += 1
        if pending:
            super().run(pending, frame)


class InterpretedWeld(InterpretedKernel):
    """``InterpretedKernel`` over the extended grammar (the H->D welds' local arrays)."""

    def __init__(self, text: str):
        super().__init__(text, interpreter=ScanInterpreter)


class InterpretedScan:
    """A compiled-scan stand-in: ``kernel(grid, block, arguments)`` executes the text, one
    lane per column (``ncol``), pointers bound to the flat views the launcher handed over."""

    def __init__(self, text: str):
        self.text = text
        self.interpreter = ScanInterpreter(text)
        self.launches = 0

    def __call__(self, grid_dims, block_dims, arguments) -> None:
        del grid_dims, block_dims
        params = self.interpreter.functions[self.interpreter.entry]["params"]
        if len(params) != len(arguments):
            raise AssertionError(f"{len(arguments)} arguments for {len(params)} parameters")
        bound: Dict[str, Any] = {}
        for (name, pointer), value in zip(params, arguments):
            if pointer:
                array = numpy.asarray(value)
                flat = array.reshape(-1)
                if not numpy.shares_memory(flat, array):
                    raise AssertionError(f"{name}: the launcher's array is not contiguous")
                bound[name] = flat
            else:
                bound[name] = value
        self.interpreter.launch(bound, int(bound["ncol"]))
        self.launches += 1


# =============================================================================
# The modules, loaded against NumPy wearing CuPy's name
# =============================================================================

_MISSING = object()


def _private_load(stem: str, privates: Tuple[str, ...]) -> Any:
    """``stem``.py loaded by path inside the package, with each of ``privates`` loaded the
    same way first and registered under its package name -- in ``sys.modules`` AND as the
    package's attribute, which ``from . import name`` reads first -- with NumPy wearing
    CuPy's name, all for the load only: nothing bound to the stand-in outlives it."""
    import meep_gpu.cuda_kernels as package

    names = ("cupy",) + tuple(f"meep_gpu.cuda_kernels.{p}" for p in privates)
    saved = {name: sys.modules.get(name) for name in names}
    attributes = {p: getattr(package, p, _MISSING) for p in privates}
    sys.modules["cupy"] = _NumpyWearingCupysName()
    try:
        for private in privates:
            spec = importlib.util.spec_from_file_location(
                f"meep_gpu.cuda_kernels._kc8_{stem}_{private}", HERE / f"{private}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[f"meep_gpu.cuda_kernels.{private}"] = module
            setattr(package, private, module)
        spec = importlib.util.spec_from_file_location(
            f"meep_gpu.cuda_kernels._kc8_{stem}", HERE / f"{stem}.py")
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for private, previous in attributes.items():
            if previous is _MISSING:
                if hasattr(package, private):
                    delattr(package, private)
            else:
                setattr(package, private, previous)
    return loaded

def _load_real_pair() -> Any:
    """``cylindrical_kernels`` with ``step_curl_kernels`` loaded privately."""
    return _private_load("cylindrical_kernels", ("step_curl_kernels",))


@pytest.fixture(scope="module")
def real_pair() -> Any:
    module = _load_real_pair()
    texts: Dict[str, str] = {}

    def _get(sub_step):
        text = texts.get(sub_step) or module._kernel_code_map()[sub_step][0]
        return InterpretedKernel(text)

    module._get_cylindrical_kernel = _get
    module._texts = texts
    return module


@pytest.fixture
def interpreted(monkeypatch):
    """CuPy is NumPy for the test, and the scan kernel the interpreter over its text
    (``scan_text`` replaces the text for a defect leg)."""
    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    state = {"scan_text": None}

    def _scan(source=None):
        return InterpretedScan(state["scan_text"] or source
                               or cylindrical_prefix.RADIAL_SCAN_SOURCE)

    monkeypatch.setattr(cylindrical_prefix, "get_radial_scan_kernel", _scan)
    return state


# =============================================================================
# The cases
# =============================================================================

#: name -> (z boundary, layer thickness, extra r cut (doubled) or None, courant)
CASES: Dict[str, Tuple] = {
    "walls": ("metallic", {"x": (0, 5), "z": (5, 5)}, None, 0.5),
    "zperiodic": ("periodic", {"x": (0, 4), "z": (3, 3)}, None, 0.5),
    "split": ("metallic", {"x": (0, 5), "z": (5, 5)}, 12, 0.5),
    "courant03": ("metallic", {"x": (0, 5), "z": (4, 4)}, None, 0.3),
}

REAL_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
              "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
              "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _split_r(grid: Grid, pml: PML, cut: int) -> ChunkLayout:
    """The one-rank boxes with every chunk spanning ``cut`` (doubled r) cut there: a third
    chunk along r whose rows restart the radial sum."""
    one_rank = ChunkLayout.one_rank(grid, pml)
    boxes, flags = [], []
    for (lo, hi), flag in zip(one_rank.boxes, one_rank.pml_flags):
        if lo[0] < cut < hi[0]:
            boxes.append((lo, (cut, hi[1], hi[2])))
            flags.append(flag)
            boxes.append(((cut, lo[1], lo[2]), hi))
            flags.append(flag)
        else:
            boxes.append((lo, hi))
            flags.append(flag)
    return ChunkLayout(boxes, flags, grid_frames(grid), faces=pml.thickness_by_face,
                       source="meep")


def build(case: str, m: int, complex_fields: bool, plant_zeros: bool, seed: int = 7,
          accurate: bool = False):
    """``(fields, grid, pml)``: a seeded full-volume Dcyl state on ``case``'s grid."""
    z_kind, thickness, cut, courant = CASES[case]
    grid = Grid(resolution=10.0, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=m,
                courant=courant, boundaries={"z": z_kind}, xp=numpy,
                accurate_fields_near_cylorigin=accurate)
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=thickness)
    if cut is not None:
        layout = _split_r(grid, pml, cut)
        fields.install_chunk_layout(layout)
        fields.install_region_plan(RegionPlan(layout, grid, pml, backend=region_backend(numpy)))
    rng = numpy.random.default_rng(seed + 31 * (m + 8) + (5 if complex_fields else 0))
    shape = tuple(int(n) for n in grid.shape)
    for name in REAL_STATE:
        target = getattr(fields, name)
        parts = []
        for _ in range(2 if target.dtype.kind == "c" else 1):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        if target.dtype.kind == "c":
            target[...] = part_products.from_parts(parts[0], parts[1], numpy.complex64)
        else:
            target[...] = parts[0]
    fields.h_separated = True
    return fields, grid, pml


def snapshot(fields) -> Dict[str, numpy.ndarray]:
    return {name: numpy.array(getattr(fields, name), copy=True) for name in REAL_STATE}


def restore(fields, state) -> None:
    for name, values in state.items():
        getattr(fields, name)[...] = values


def bits(array) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def differing(want, got) -> Dict[str, int]:
    return {name: int(numpy.count_nonzero(bits(want[name]) != bits(got[name])))
            for name in want}


def record(fields, pml) -> numpy.ndarray:
    return numpy.ascontiguousarray(
        chunk_bounds_record(stepping._region_plan(fields, pml).layout), dtype=numpy.int32)


TARGETS = {"step_B": ("Bx", "By", "Bz"), "step_D": ("Dx", "Dy", "Dz")}


def _assert_identical_and_moved(sub_step, start, want, got, label) -> None:
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (label, {k: v for k, v in diff.items() if v})
    for component in TARGETS[sub_step]:
        moved = int(numpy.count_nonzero(bits(want[component]) != bits(start[component])))
        assert moved > 0, (label, component)


# =============================================================================
# 1. The scan computes the array path's radial difference, word for word
# =============================================================================

def _random_column_field(rng, shape, dtype, wall_row=False):
    values = rng.standard_normal(shape).astype(numpy.float32)
    if dtype is numpy.complex64:
        values = (values + 1j * rng.standard_normal(shape).astype(numpy.float32)
                  ).astype(numpy.complex64)
    words = values.reshape(-1).view(numpy.uint32)
    words[::5] = 0x80000000  # -0 in every fifth word
    words[3::11] = 0x00000001  # and a subnormal
    if wall_row:
        values[-1] = 0
    return values


def _row_starts(chunk_starts, n_target, half_integer):
    line = numpy.zeros((n_target, 1, 1), dtype=numpy.int64)
    for row in range(n_target):
        doubled = 2 * row + (1 if half_integer else 0)
        line[row] = max([s for s in chunk_starts if 2 * s < doubled], default=0)
    return line


def _scan_tables(sub_step, rows_stored, starts):
    weights, rinv = cylindrical_prefix.radial_rows(sub_step, rows_stored)
    n_target = rows_stored
    line = (numpy.zeros(n_target, dtype=numpy.int32) if starts is None
            else numpy.ascontiguousarray(starts.reshape(-1), dtype=numpy.int32))
    return {"w": weights, "rinv": rinv, "starts": line,
            "segs": numpy.ascontiguousarray(numpy.unique(line), dtype=numpy.int32),
            "rows_stored": rows_stored,
            "rows_src": cylindrical_prefix.prefix_rows(sub_step, rows_stored),
            "n_target": n_target, "backward": cylindrical_prefix.RADIAL_BACKWARD[sub_step]}


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.complex64])
@pytest.mark.parametrize("chunk_starts", [(0,), (0, 7), (0, 3, 7)])
@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_scan_is_the_array_paths_radial_difference_word_for_word(sub_step, chunk_starts,
                                                                     dtype):
    """The B side (forward, Ep with its wall row read as +0) and the D side (backward, Hp)
    on one, two and three chunks along r, real and complex, with -0 and subnormal words."""
    rng = numpy.random.default_rng(20261009 + len(chunk_starts) + (sub_step == "step_D"))
    rows, nz = 12, 6
    source = _random_column_field(rng, (rows, 1, nz), dtype)
    backward = cylindrical_prefix.RADIAL_BACKWARD[sub_step]
    ir0 = cylindrical_prefix.PREFIX_IR0[sub_step]
    starts = None if chunk_starts == (0,) else _row_starts(chunk_starts, rows, not backward)
    if backward:
        want = stepping.cylindrical_rderiv_difference(numpy, source, ir0, starts, rows,
                                                      backward=True)
    else:
        extended = numpy.zeros((rows + 1, 1, nz), dtype=dtype)
        extended[:rows] = source
        want = stepping.cylindrical_rderiv_difference(numpy, extended, ir0, starts, rows,
                                                      backward=False)
    got = numpy.full((rows, 1, nz), numpy.nan, dtype=dtype)
    kernel = InterpretedScan(cylindrical_prefix.RADIAL_SCAN_SOURCE)
    cylindrical_prefix.launch_radial_scan(kernel, source, _scan_tables(sub_step, rows, starts),
                                          got, numpy)
    assert kernel.launches == 1
    diff = int(numpy.count_nonzero(bits(got) != bits(want)))
    assert diff == 0, (sub_step, chunk_starts, dtype, diff)
    assert int(numpy.count_nonzero(bits(want))) > 0


def _scan_defects() -> Dict[str, Tuple[str, str]]:
    return {
        # One sum from the axis for every row: the per-chunk restart dropped.
        "one_global_sum": ("            if (starts[row] == s) {", "            if (s == 0) {"),
        # The first row of a chunk summed from its own increment instead of +0.
        "sum_starts_at_the_first_increment": (
            "        float total = 0.0f;",
            "        float total = rinv[s > 0 ? s - 1 : 0] * (w[s] * fp[s * ncol + idx]);"),
        # The two products of the increment's difference swapped in order of subtraction.
        "difference_reversed": ("(rinv[g - 1] * (here - prev))", "(rinv[g - 1] * (prev - here))"),
    }


@pytest.mark.parametrize("defect", sorted(_scan_defects()))
def test_each_planted_scan_defect_changes_a_word(defect):
    old, new = _scan_defects()[defect]
    text = cylindrical_prefix.RADIAL_SCAN_SOURCE
    assert text.count(old) == 1, defect
    rng = numpy.random.default_rng(3)
    rows = 12
    source = _random_column_field(rng, (rows, 1, 6), numpy.float32)
    starts = _row_starts((0, 3, 7), rows, False)
    tables = _scan_tables("step_D", rows, starts)
    want = numpy.empty((rows, 1, 6), dtype=numpy.float32)
    cylindrical_prefix.launch_radial_scan(InterpretedScan(text), source, tables, want, numpy)
    got = numpy.empty_like(want)
    cylindrical_prefix.launch_radial_scan(InterpretedScan(text.replace(old, new, 1)), source,
                                          tables, got, numpy)
    assert int(numpy.count_nonzero(bits(got) != bits(want))) > 0, defect


def test_the_scan_reads_the_array_paths_chunk_starts_on_a_layered_grid():
    """On the one-rank division of an r layer the radial sum restarts at the layer's cut
    (row 14 at resolution 10), and the starts line is ``stepping._cylindrical_scan_starts``'s."""
    fields, _grid, pml = build("walls", 1, True, False)
    for sub_step, target in (("step_B", "Bz"), ("step_D", "Dz")):
        line, segments = cylindrical_prefix.radial_starts(fields, pml, sub_step)
        want = numpy.asarray(stepping._cylindrical_scan_starts(fields, pml, target)).reshape(-1)
        assert line.tolist() == want.tolist()
        assert segments.tolist() == [0, 14]


def test_a_division_whose_r_cuts_differ_along_z_is_refused_by_name(monkeypatch):
    fields, _grid, pml = build("walls", 1, True, False)
    shape = tuple(int(n) for n in fields.Dz.shape)
    monkeypatch.setattr(stepping, "_cylindrical_scan_starts",
                        lambda *_a, **_k: numpy.zeros(shape, dtype=numpy.int64))
    with pytest.raises(cylindrical_prefix.RadialScanRefusal, match="differ along z"):
        cylindrical_prefix.radial_starts(fields, pml, "step_D")


def test_the_scan_text_divides_nothing_and_is_ascii():
    text = cylindrical_prefix.RADIAL_SCAN_SOURCE
    text.encode("ascii")
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    assert "/" not in code


# =============================================================================
# 2. The real m = 0 pair computes the array path, word for word
# =============================================================================

def _launch_real(real_pair, sub_step, fields, grid, pml, chunk_record=None) -> None:
    tables = real_pair.cylindrical_curl_tables(pml, real_pair.HALF_INTEGER[sub_step])
    codes = real_pair.cylindrical_boundary_codes_for(grid)
    scan = cylindrical_prefix.radial_scan_device_tables(
        cylindrical_prefix.radial_scan_tables(fields, pml, sub_step), numpy)
    real_pair.CYLINDRICAL_LAUNCHERS[sub_step](
        fields, tables, codes, float(grid.dt / grid.dx), scratch=None,
        chunk_bounds=record(fields, pml) if chunk_record is None else chunk_record,
        pml=pml, scan=scan)


REAL_PARAMETERS = [(sub_step, case, plant) for sub_step in ("step_B", "step_D")
                   for case in sorted(CASES) for plant in (False, True)]


@pytest.mark.parametrize("sub_step,case,plant", REAL_PARAMETERS,
                         ids=[f"{s}-{c}-{'zeros' if p else 'random'}"
                              for s, c, p in REAL_PARAMETERS])
def test_the_real_m0_text_is_the_array_path_word_for_word(sub_step, case, plant, real_pair,
                                                         interpreted):
    fields, grid, pml = build(case, 0, False, plant)
    start = snapshot(fields)
    getattr(stepping, sub_step)(fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    _launch_real(real_pair, sub_step, fields, grid, pml)
    _assert_identical_and_moved(sub_step, start, want, snapshot(fields), (sub_step, case))


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_each_real_kernel_decodes_the_record_with_the_emitters_block(sub_step, real_pair):
    text = real_pair._kernel_code_map()[sub_step][0]
    components = TARGETS[sub_step]
    block = bounds_block.bounds_block(bounds_block.curl_entries(components), (0, 1, 2))
    assert text.count(bounds_block.DECODE_END + "\n" + block) == 1


def _real_defects() -> Dict[str, Tuple[str, str, str]]:
    return {
        # MEEP's double 4*Courant product put back as a float32 product.
        "axis_rule_in_float32": (
            "step_D",
            "__double2float_rn(axis_courant * (double)(Hy[idx]))",
            "(__double2float_rn(axis_courant) * Hy[idx])"),
        # The 0.9.2 four-operand grouping on a two-partner curl.
        "old_stencil_grouping": (
            "step_B",
            "float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_z, pr_x);",
            "float curl = dtdx * ((sf - f1) + (f2 - ss));"),
        # The chunk record ignored: the recurrence everywhere.
        "recurrence_everywhere": (
            "step_D",
            "kms_z[k], sinv_z[k], kms_x[i], sinv_x[i], in_i_x);",
            "kms_z[k], sinv_z[k], kms_x[i], sinv_x[i], 1);"),
        # The radial difference not scaled by dtdx.
        "radial_unscaled": ("step_B", "float curl = dtdx * rad[idx];", "float curl = rad[idx];"),
        # The m = 0 B tail dropped.
        "no_axis_tail": ("step_B", "    if (i == 0) Bx[idx] = 0.0f;\n", ""),
    }


@pytest.mark.parametrize("defect", sorted(_real_defects()))
def test_each_planted_real_defect_changes_a_word(defect, real_pair, interpreted):
    sub_step, old, new = _real_defects()[defect]
    text = real_pair._kernel_code_map()[sub_step][0]
    assert text.count(old) == 1, defect
    case = "courant03" if defect == "axis_rule_in_float32" else "walls"
    fields, grid, pml = build(case, 0, False, True)
    start = snapshot(fields)
    _launch_real(real_pair, sub_step, fields, grid, pml)
    want = snapshot(fields)
    restore(fields, start)
    real_pair._texts[sub_step] = text.replace(old, new, 1)
    try:
        _launch_real(real_pair, sub_step, fields, grid, pml)
    finally:
        real_pair._texts.clear()
    assert sum(differing(want, snapshot(fields)).values()) > 0, defect


def test_a_scan_handed_one_chunk_on_the_split_division_is_caught(real_pair, interpreted):
    """On the lifted three-chunk division the radial sum restarts at two cuts; a scan handed
    the one-chunk starts line (one sum from the axis) is another word past the first cut."""
    fields, grid, pml = build("split", 0, False, False)
    lifted = record(fields, pml)
    start = snapshot(fields)
    stepping.step_D(fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    scan = cylindrical_prefix.radial_scan_device_tables(
        cylindrical_prefix.radial_scan_tables(fields, pml, "step_D"), numpy)
    scan["starts"] = numpy.zeros_like(scan["starts"])
    scan["segs"] = numpy.zeros(1, dtype=numpy.int32)
    tables = real_pair.cylindrical_curl_tables(pml, False)
    real_pair.CYLINDRICAL_LAUNCHERS["step_D"](
        fields, tables, real_pair.cylindrical_boundary_codes_for(grid),
        float(grid.dt / grid.dx), chunk_bounds=lifted, pml=pml, scan=scan)
    assert sum(differing(want, snapshot(fields)).values()) > 0


# =============================================================================
# 3. The complex pair computes the array path at every m class, word for word
# =============================================================================

def _launch_complex(sub_step, fields, grid, pml, text=None) -> None:
    source = text or cylindrical_complex_kernels.cylindrical_complex_source(sub_step)
    cylindrical_complex_kernels.step_cylindrical_complex(
        sub_step, fields, grid, pml, chunk_bounds=record(fields, pml),
        kernel=InterpretedKernel(source))


COMPLEX_PARAMETERS = [(sub_step, case, m, plant)
                      for sub_step in ("step_B", "step_D")
                      for case in ("walls", "split", "zperiodic", "courant03")
                      for m in (0, 1, -1, 2, 3)
                      for plant in (False, True)
                      if not (plant and case in ("zperiodic", "split") and m in (-1, 3))]


@pytest.mark.parametrize("sub_step,case,m,plant", COMPLEX_PARAMETERS,
                         ids=[f"{s}-{c}-m{m}-{'zeros' if p else 'random'}"
                              for s, c, m, p in COMPLEX_PARAMETERS])
def test_the_complex_text_is_the_array_path_word_for_word(sub_step, case, m, plant,
                                                          interpreted):
    fields, grid, pml = build(case, m, True, plant)
    start = snapshot(fields)
    getattr(stepping, sub_step)(fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    _launch_complex(sub_step, fields, grid, pml)
    _assert_identical_and_moved(sub_step, start, want, snapshot(fields), (sub_step, case, m))


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_accurate_near_axis_branch_is_the_array_path(sub_step, interpreted):
    fields, grid, pml = build("courant03", 2, True, True, accurate=True)
    assert cylindrical_complex_kernels.zero_rows(2, True) == 1
    start = snapshot(fields)
    getattr(stepping, sub_step)(fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    _launch_complex(sub_step, fields, grid, pml)
    _assert_identical_and_moved(sub_step, start, want, snapshot(fields), (sub_step, "accurate"))


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_each_complex_kernel_decodes_the_record_with_the_emitters_block(sub_step):
    text = cylindrical_complex_kernels.cylindrical_complex_source(sub_step)
    block = bounds_block.bounds_block(bounds_block.curl_entries(TARGETS[sub_step]), (0, 1, 2))
    assert text.count(bounds_block.DECODE_END + "\n" + block) == 1


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_source_is_ascii_with_no_fused_multiply_and_no_float_division(sub_step):
    text = cylindrical_complex_kernels.cylindrical_complex_source(sub_step)
    text.encode("ascii")
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    assert "__fmaf_rn" not in code and "fma(" not in code
    assert re.search(r"[^/*]/[^/*]", code.replace("idx / nz", "").replace(
        "idx / (ny * nz)", "")) is None


@pytest.mark.parametrize("target,sign", [("Bx", +1.0), ("Bz", -1.0), ("Dx", -1.0),
                                         ("Dz", +1.0)])
@pytest.mark.parametrize("m", [1, -1, 3])
def test_the_imr_row_is_the_array_paths_quotient_word_for_word(target, sign, m):
    """The bound row is ``stepping._cylindrical_imr_increment``'s rinv: its increment of a
    unit partner (1 + 0j) carries rinv in the imaginary part exactly."""
    grid = Grid(resolution=10.0, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=m,
                courant=0.3, xp=numpy)
    fields = Fields(grid=grid, force_complex_fields=True)
    rows = int(grid.shape[0])
    unit = numpy.ones((rows, 1, 1), dtype=numpy.complex64)
    increment = stepping._cylindrical_imr_increment(fields, target, unit, sign)
    want = numpy.ascontiguousarray(increment.imag.reshape(-1))
    got = cylindrical_complex_kernels.imr_rinv_row(grid, target, sign, rows)
    assert numpy.array_equal(bits(got), bits(want))


def _complex_defects() -> Dict[str, Tuple[str, str, str, str]]:
    return {
        # The i*m/r term folded into the curl before the PML form (the 0.9.2 order).
        "imr_folded_into_the_curl": (
            "step_B", "walls",
            "        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();\n"
            "        // stepping._cylindrical_axis_increment_B",
            "        if (m_class != 0) curl = cf_add(curl, imr_increment(imr0[i], f_1));\n"
            "        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();\n"
            "        // stepping._cylindrical_axis_increment_B"),
        # The unowned cell's increment a NEGATIVE zero: subtracting it is adding a masked
        # +0 to MEEP's raw, which turns a stored -0 into +0 on a cell MEEP never visits
        # (Bz on the metallic z wall).
        "imr_masked_to_negative_zero": (
            "step_B", "walls",
            "            if (bc_z == BC_METALLIC && k == 0) inc = cf_zero();",
            "            if (bc_z == BC_METALLIC && k == 0) inc = cf_neg(cf_zero());"),
        # The |m| = 1 axis product in float32 rather than MEEP's double Courant product.
        "axis_increment_in_float32": (
            "step_D", "courant03",
            "    o.re = __double2float_rn(c * (double)(v.re));",
            "    o.re = __double2float_rn(c) * v.re;"),
        # The i*m/r unowned-cell mask dropped on Dx's metallic z wall, inside the z layer
        # (its fu stage reads the increment there).
        "imr_mask_dropped": (
            "step_D", "walls",
            "            if (bc_z == BC_METALLIC && k == 0) inc = cf_zero();\n"
            "            fold_apply(Dx,",
            "            fold_apply(Dx,"),
        # The near-axis zeroing skipping the auxiliaries.
        "zeroing_skips_fu": ("step_B", "walls",
                             "        cf_store(fu_Bx, idx, cf_zero());\n", ""),
    }


@pytest.mark.parametrize("defect", sorted(_complex_defects()))
def test_each_planted_complex_defect_changes_a_word(defect, interpreted):
    sub_step, case, old, new = _complex_defects()[defect]
    text = cylindrical_complex_kernels.cylindrical_complex_source(sub_step)
    assert text.count(old) == 1, defect
    m = 2 if defect == "zeroing_skips_fu" else 1
    fields, grid, pml = build(case, m, True, True)
    start = snapshot(fields)
    _launch_complex(sub_step, fields, grid, pml)
    want = snapshot(fields)
    restore(fields, start)
    _launch_complex(sub_step, fields, grid, pml, text=text.replace(old, new, 1))
    assert sum(differing(want, snapshot(fields)).values()) > 0, defect


# =============================================================================
# 4. The fused pairs compute the array path's sub-step sequence, word for word
# =============================================================================

SEQUENCES = {
    "B": (("step_B", "zero_metal_B", "update_H"), ("Bx", "By", "Bz")),
    "D": (("step_D", "zero_metal_D", "update_E"), ("Dx", "Dy", "Dz")),
}


def _oracle_sequence(side: str, fields, pml) -> None:
    for name in SEQUENCES[side][0]:
        function = getattr(stepping, name)
        if name.startswith("zero_metal"):
            function(fields)
        else:
            function(fields, pml)


def _with_epsilon(fields, grid, seed: int = 19) -> None:
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))


def _load_real_fused(stem: str) -> Any:
    """A real cylindrical fused module with its three CuPy-backed halves loaded privately."""
    return _private_load(stem, ("step_curl_kernels", "constitutive_kernels",
                                "cylindrical_kernels"))


@pytest.fixture(scope="module")
def real_fused() -> Dict[str, Any]:
    return {"B": _load_real_fused("cylindrical_real_fused_magnetic_pair"),
            "D": _load_real_fused("cylindrical_real_fused_electric_pair")}


def _scan_radial(fields, pml, sub_step):
    return cylindrical_prefix.radial_difference(
        fields, pml, sub_step,
        tables=cylindrical_prefix.radial_scan_device_tables(
            cylindrical_prefix.radial_scan_tables(fields, pml, sub_step), numpy),
        kernel=InterpretedScan(cylindrical_prefix.RADIAL_SCAN_SOURCE))


def _launch_real_fused(side, family, fields, grid, pml, text=None) -> Dict[str, Any]:
    from .coverage import real_pml_boundary_kinds
    from .cylindrical_coverage import cylindrical_boundary_codes

    sub_step = "step_B" if side == "B" else "step_D"
    source = text or (family.cylindrical_real_fused_magnetic_pair_source() if side == "B"
                      else family.cylindrical_real_fused_electric_pair_source())
    tables = (family.cylindrical_real_fused_magnetic_pair_tables(pml) if side == "B"
              else family.cylindrical_real_fused_electric_pair_tables(pml))
    codes = tuple(numpy.int32(code) for code in
                  cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid))))
    from .in_seam_coverage import zero_metal_axes
    radial = _scan_radial(fields, pml, sub_step)
    if side == "B":
        return family.launch_cylindrical_real_fused_magnetic_pair(
            fields, tables, radial, codes, zero_metal_axes(grid), float(grid.dt / grid.dx),
            InterpretedKernel(source), chunk_bounds=record(fields, pml), layer=pml)
    return family.launch_cylindrical_real_fused_electric_pair(
        fields, tables, radial, codes, zero_metal_axes(grid), float(grid.dt / grid.dx),
        InterpretedKernel(source), chunk_bounds=record(fields, pml),
        axis_courant=numpy.float64(4.0 * float(grid.courant)))


REAL_FUSED_PARAMETERS = [(side, case, plant, separated)
                         for side in ("B", "D") for case in ("walls", "zperiodic", "courant03")
                         for plant in (False, True)
                         for separated in ((True, False) if side == "B" else (True,))]


@pytest.mark.parametrize("side,case,plant,separated", REAL_FUSED_PARAMETERS,
                         ids=[f"{s}-{c}-{'zeros' if p else 'random'}-"
                              f"{'midrun' if h else 'first'}"
                              for s, c, p, h in REAL_FUSED_PARAMETERS])
def test_the_real_fused_pair_is_the_array_paths_sequence_word_for_word(
        side, case, plant, separated, real_fused, interpreted):
    fields, grid, pml = build(case, 0, False, plant)
    _with_epsilon(fields, grid)
    fields.h_separated = separated
    start = snapshot(fields)
    _oracle_sequence(side, fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    fields.h_separated = separated
    result = _launch_real_fused(side, real_fused[side], fields, grid, pml)
    assert result["launched"]
    _assert_identical_and_moved("step_B" if side == "B" else "step_D", start, want,
                                snapshot(fields), (side, case, separated))


def _complex_constitutive_tables(pml, half_integer: bool) -> Dict[str, numpy.ndarray]:
    suffix = "_h" if half_integer else ""
    return {f"{stem}_{axis}": numpy.ascontiguousarray(
                getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1))
            for axis in "xyz" for stem in ("kps", "kms")}


@pytest.fixture
def complex_fused(monkeypatch, interpreted):
    from . import cylindrical_fused_electric_pair, cylindrical_fused_magnetic_pair

    monkeypatch.setattr(
        cylindrical_fused_magnetic_pair, "cylindrical_fused_magnetic_pair_tables",
        lambda pml: {"curl": cylindrical_complex_kernels.cylindrical_complex_curl_tables(
                         pml, True),
                     "constitutive": _complex_constitutive_tables(pml, False)})
    monkeypatch.setattr(
        cylindrical_fused_electric_pair, "cylindrical_fused_electric_pair_tables",
        lambda pml: {"curl": cylindrical_complex_kernels.cylindrical_complex_curl_tables(
                         pml, False),
                     "constitutive": _complex_constitutive_tables(pml, True)})
    return {"B": cylindrical_fused_magnetic_pair, "D": cylindrical_fused_electric_pair}


def _launch_complex_fused(side, family, fields, grid, pml, text=None) -> Dict[str, Any]:
    arguments = family.resolve_launch_arguments(fields, grid, pml, record(fields, pml))
    source = text or (family.cylindrical_fused_magnetic_pair_source() if side == "B"
                      else family.cylindrical_fused_electric_pair_source())
    launch = (family.launch_cylindrical_fused_magnetic_pair if side == "B"
              else family.launch_cylindrical_fused_electric_pair)
    return launch(fields, arguments, kernel=InterpretedKernel(source),
                  scan_kernel=InterpretedScan(cylindrical_prefix.RADIAL_SCAN_SOURCE))


COMPLEX_FUSED_PARAMETERS = [(side, case, m, plant)
                            for side in ("B", "D")
                            for case in ("walls", "zperiodic", "courant03")
                            for m in (0, 1, -1, 2, 3)
                            for plant in (False, True)
                            if not (plant and case == "zperiodic")]


@pytest.mark.parametrize("side,case,m,plant", COMPLEX_FUSED_PARAMETERS,
                         ids=[f"{s}-{c}-m{m}-{'zeros' if p else 'random'}"
                              for s, c, m, p in COMPLEX_FUSED_PARAMETERS])
def test_the_complex_fused_pair_is_the_array_paths_sequence_word_for_word(
        side, case, m, plant, complex_fused):
    fields, grid, pml = build(case, m, True, plant)
    _with_epsilon(fields, grid)
    start = snapshot(fields)
    _oracle_sequence(side, fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    result = _launch_complex_fused(side, complex_fused[side], fields, grid, pml)
    assert result["launched"]
    _assert_identical_and_moved("step_B" if side == "B" else "step_D", start, want,
                                snapshot(fields), (side, case, m))


@pytest.mark.parametrize("m", [0, 1])
def test_the_complex_magnetic_pair_starts_h_on_the_runs_first_launch(m, complex_fused):
    fields, grid, pml = build("walls", m, True, True)
    fields.h_separated = False
    start = snapshot(fields)
    _oracle_sequence("B", fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    fields.h_separated = False
    result = _launch_complex_fused("B", complex_fused["B"], fields, grid, pml)
    assert result["h_started"] and fields.h_separated
    _assert_identical_and_moved("step_B", start, want, snapshot(fields), ("first", m))


@pytest.mark.parametrize("side", ["B", "D"])
def test_the_complex_fused_texts_derive_their_helpers(side, complex_fused):
    family = complex_fused[side]
    emit = (family.cylindrical_fused_magnetic_pair_source if side == "B"
            else family.cylindrical_fused_electric_pair_source)
    text = emit()
    for helper in ("cf pml_apply_reg_pre(", "cf fold_apply_reg(", "void constitutive_apply_pre("):
        assert text.count(helper) == 1, helper
    assert not re.search(r"\b(pml_apply|fold_apply|constitutive_apply)\(", text)
    names = cylindrical_complex_kernels.shipped_kernel_names(text)
    assert names == {family.KERNEL_NAME}


# =============================================================================
# 5. The H->D welds: update_H into scratch, the scan over the new Hp, the curl
# =============================================================================

def _load_real_hd() -> Any:
    """``cylindrical_real_fused_hd_pair`` with its halves AND ``fused_hd_pair`` loaded
    privately."""
    return _private_load("cylindrical_real_fused_hd_pair",
                         ("step_curl_kernels", "constitutive_kernels", "cylindrical_kernels",
                          "fused_hd_pair"))


@pytest.fixture(scope="module")
def real_hd() -> Any:
    weld = _load_real_hd()
    curl = weld.cylindrical_kernels
    curl._get_cylindrical_kernel = (
        lambda sub_step: InterpretedKernel(curl._kernel_code_map()[sub_step][0]))
    return weld


HD_PARAMETERS = [(case, plant, separated) for case in ("walls", "zperiodic", "courant03")
                 for plant in (False, True) for separated in (True, False)]


@pytest.mark.parametrize("case,plant,separated", HD_PARAMETERS,
                         ids=[f"{c}-{'zeros' if p else 'random'}-{'midrun' if h else 'first'}"
                              for c, p, h in HD_PARAMETERS])
def test_the_real_hd_weld_is_the_array_paths_sequence_word_for_word(case, plant, separated,
                                                                    real_hd, interpreted):
    fields, grid, pml = build(case, 0, False, plant)
    fields.h_separated = separated
    start = snapshot(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    fields.h_separated = separated
    # The run's three launches through the family's own steps (its predicate refuses a
    # NumPy backend, and its disjointness check reads device base addresses).
    state = real_hd.resolve(fields, grid, pml, chunk_bounds=record(fields, pml))
    first = real_hd.launch_update_H(fields, state,
                                    kernel=InterpretedWeld(real_hd.kernel_source()))
    real_hd.scan_and_rotate(fields, state,
                            scan_kernel=InterpretedScan(cylindrical_prefix.RADIAL_SCAN_SOURCE))
    real_hd.launch_curl(fields, state)
    assert state["launches"] == real_hd.LAUNCHES_PER_RUN
    assert first["h_started"] == (not separated)
    _assert_identical_and_moved("step_D", start, want, snapshot(fields), (case, separated))


HD_COMPLEX_PARAMETERS = [(case, m, plant) for case in ("walls", "courant03")
                         for m in (0, 1, -1, 2, 3) for plant in (False, True)]


@pytest.mark.parametrize("case,m,plant", HD_COMPLEX_PARAMETERS,
                         ids=[f"{c}-m{m}-{'zeros' if p else 'random'}"
                              for c, m, p in HD_COMPLEX_PARAMETERS])
def test_the_complex_hd_weld_is_the_array_paths_sequence_word_for_word(case, m, plant,
                                                                       monkeypatch,
                                                                       interpreted):
    from types import SimpleNamespace

    from . import cylindrical_fused_hd_pair as family

    monkeypatch.setattr(family, "complex_pml_kernels", SimpleNamespace(
        complex_constitutive_tables=_complex_constitutive_tables))
    fields, grid, pml = build(case, m, True, plant)
    start = snapshot(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    want = snapshot(fields)
    restore(fields, start)
    curl = InterpretedKernel(cylindrical_complex_kernels.cylindrical_complex_source(
        "step_D"))
    state = family.resolve(fields, grid, pml, chunk_bounds=record(fields, pml))
    family.launch_update_H(fields, state, kernel=InterpretedWeld(family.kernel_source()))
    family.scan_and_rotate(fields, state,
                           scan_kernel=InterpretedScan(cylindrical_prefix.RADIAL_SCAN_SOURCE))
    family.launch_curl(fields, grid, pml, state, curl_kernel=curl)
    assert state["launches"] == family.LAUNCHES_PER_RUN
    _assert_identical_and_moved("step_D", start, want, snapshot(fields), (case, m))
