"""The CUDA complex singles transcribe the array path's part-wise, per-chunk forms, read off
their text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The four complex PML kernels
(``complex_emitter``: ``step_B`` / ``step_D`` / ``update_H`` / ``update_E`` under complex64
storage) form every product part by part, choose MEEP's PML form per structure chunk from
the run's chunk record ``cb``, and take MEEP's not-owned word on plane 0 of a Bloch-phased
axis (bit_parity_0_10 DESIGN.md D1, D6, D10, D11, D16; ``stepping._apply_curl_regions``,
``_apply_constitutive_regions``, ``_bloch_ghost_operand``; ``part_products``). Each kernel is
LAUNCHED THROUGH ITS OWN LAUNCHER (``complex_pml_kernels.step_fused_pml_complex`` /
``update_fused_pml_complex``, loaded with NumPy wearing CuPy's name), whose compiled kernel is
replaced by an interpreter that EXECUTES the emitted device text in float32 NumPy -- its
helpers too, from their own text -- so the launcher's argument order, its table and record
bindings and its Bloch connection binding are exercised with the text. Every array of the
``Fields`` is compared word for word (sign of zero included) with the array path's own
sub-step on a copy of the same seeded full-volume state.

The cases cover walls, a periodic axis carrying a layer at k = 0, a generic Bloch phase on
two axes (the unfused connection on both lattices), the Brillouin edge (the per-part
negation) and a phase within the tolerance of +1 (a copy) on a two-dimensional grid (the
partner-presence forms), a lifted division whose record differs from the one-rank one, and
planted signed zeros of both signs. Each case asserts a nonzero cell count on and off the
membership run for each component it should, the connection kinds it is built to reach, and
that the sub-step moved its field (a vacuity floor). A battery of planted defects -- each a
rewrite of one emitted text or one host binding -- must change a word.

THE INTERPRETER is the root lane's (``test_root_chunk_forms.Interpreter``, with the
fused-pair lane's stores at an evaluated index), extended here to the complex family's word
pair: a ``cf`` value with ``.re`` / ``.im`` members, declared, assigned member by member,
returned from a helper and selected by the conditional operator. Anything else still
raises, so the grammar cannot grow a construct nothing executes.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import ast
import numpy
import pytest

from .. import part_products, stepping
from ..chunk_map import ChunkLayout, RegionPlan, chunk_bounds_record, grid_frames, region_backend
from ..fields import Fields
from ..grid import Grid
from ..pml import PML
from . import bloch_connection, complex_emitter
from .test_fused_pair_chunk_forms import ScatterInterpreter
from .test_root_chunk_forms import (F32, Unmodelled, _items, _match_close,
                                    _NumpyWearingCupysName, _Pointer, _Register,
                                    _split_top, _truth)

HERE = pathlib.Path(__file__).parent


# =============================================================================
# The interpreter, over the complex word pair
# =============================================================================

class CF:
    """One ``cf`` value per lane: two float32 parts."""

    __slots__ = ("re", "im")

    def __init__(self, re_part: Any, im_part: Any):
        self.re = numpy.asarray(re_part)
        self.im = numpy.asarray(im_part)
        for part in (self.re, self.im):
            if part.dtype != F32:
                raise Unmodelled(f"a cf part of dtype {part.dtype}")


def _nan_cf() -> CF:
    return CF(F32(numpy.nan), F32(numpy.nan))


class Struct:
    """A plain C struct value (the H->D weld's ``WeldArgs``): members set once, read by
    name; the same for every lane."""

    def __init__(self, name: str):
        self.name = name


class CFArray:
    """A local ``cf name[n];`` array, passed to a helper by address."""

    def __init__(self, size: int):
        self.items = [_nan_cf() for _ in range(size)]


class ElementRef:
    """``&array[k]``: one element of a :class:`CFArray`, written through ``*ref = ...``."""

    def __init__(self, array: "CFArray", index: int):
        self.array = array
        self.index = index


def _where(test: Any, body: Any, orelse: Any) -> Any:
    if isinstance(body, CF) or isinstance(orelse, CF):
        if not (isinstance(body, CF) and isinstance(orelse, CF)):
            raise Unmodelled("a conditional selecting a cf against a non-cf")
        return CF(numpy.where(test, body.re, orelse.re), numpy.where(test, body.im, orelse.im))
    return numpy.where(test, body, orelse)


def _parenthesize_arguments(expression: str) -> str:
    """Each comma-separated piece of every bracketed group that carries a conditional
    operator, wrapped in its own parentheses, so the operator stays inside the argument it
    is written in (``f(a ? b : c, d)`` reads as ``f((a ? b : c), d)``, which is C's
    grouping)."""
    out, position = [], 0
    while position < len(expression):
        character = expression[position]
        if character in "([":
            closing = ")" if character == "(" else "]"
            end = _match_close(expression, position, character, closing)
            inner = expression[position + 1:end]
            parts = _split_top(inner)
            if len(parts) > 1:
                inner = ", ".join(f"({_parenthesize_arguments(part)})" if "?" in part
                                  else _parenthesize_arguments(part) for part in parts)
            else:
                inner = _parenthesize_arguments(inner)
            out.append(character + inner + closing)
            position = end + 1
            continue
        out.append(character)
        position += 1
    return "".join(out)


class ComplexInterpreter(ScatterInterpreter):
    """The root interpreter with the complex family's ``cf`` word pair, the H->D weld's
    argument struct and its local ``cf`` arrays."""

    def __init__(self, text: str):  # noqa: F811 - the struct names, then the base init
        self.structs = set(re.findall(r"(?m)^struct (\w+) \{", text))
        lines = [line for line in text.splitlines()
                 if not line.strip().startswith("typedef struct { float re; float im; } cf;")]
        ScatterInterpreter.__init__(self, "\n".join(lines))

    def _eval(self, node: ast.AST, frame: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Attribute):
            value = self._eval(node.value, frame)
            if isinstance(value, CF) and node.attr in ("re", "im"):
                return getattr(value, node.attr)
            if isinstance(value, Struct) and hasattr(value, node.attr):
                return getattr(value, node.attr)
            raise Unmodelled(f"member {node.attr} of a {type(value).__name__}")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            if name.startswith("_addr_"):
                target = self._lookup(name[len("_addr_"):], frame)
                if isinstance(target, CFArray):
                    return ElementRef(target, int(self._eval(node.slice, frame)))
            else:
                target = self._lookup(name, frame)
                if isinstance(target, CFArray):
                    return target.items[int(self._eval(node.slice, frame))]
        if isinstance(node, ast.IfExp):
            test = _truth(self._eval(node.test, frame))
            return _where(test, self._eval(node.body, frame), self._eval(node.orelse, frame))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("double", "float") and len(node.args) == 1):
            # C's casts, written (double)(x) / (float)(x): the widening is exact, the
            # narrowing one IEEE round to nearest even.
            value = numpy.asarray(self._eval(node.args[0], frame))
            if value.dtype not in (F32, numpy.float64):
                raise Unmodelled(f"({node.func.id}) of a {value.dtype} value")
            return value.astype(numpy.float64 if node.func.id == "double" else F32)
        return super()._eval(node, frame)

    def call(self, name: str, args: List[ast.AST], caller: Dict[str, Any]) -> Any:
        # A helper's predicate is the caller's ACTIVE lanes, so a helper called after some
        # lanes of its caller returned (cf_store after an ownership guard) stores nothing
        # for them.
        function = self.functions[name]
        if len(args) != len(function["params"]):
            raise Unmodelled(f"{name}: {len(args)} arguments for {len(function['params'])}")
        frame: Dict[str, Any] = {"__pred__": self._active(caller)}
        for (param, pointer), arg in zip(function["params"], args):
            value = self._eval(arg, caller)
            if pointer != isinstance(value, (_Pointer, _Register, CFArray, ElementRef)):
                raise Unmodelled(f"{name}: {param} bound to {type(value).__name__}")
            frame[param] = value
        frame["__ret__"] = None
        frame["__done__"] = numpy.zeros((), dtype=bool)
        self.run(_items(function["body"]), frame)
        return frame["__ret__"]

    def _set(self, frame: Dict[str, Any], name: str, value: Any) -> None:
        owner = self._owner(name, frame)
        if owner is None:
            raise Unmodelled(f"assignment to undeclared {name}")
        owner[name] = _where(self._active(frame), value, owner[name])

    def _return(self, frame: Dict[str, Any], text: str, test: Any) -> None:
        if "__done__" not in frame:
            raise Unmodelled("a value return in the kernel, or in a nested scope of a helper")
        value = self.value(text, frame)
        effective = numpy.logical_and(test, frame["__pred__"])
        take = numpy.logical_and(effective, numpy.logical_not(frame["__done__"]))
        current = frame["__ret__"]
        if current is None:
            if isinstance(value, CF):
                current = _nan_cf()
            elif numpy.asarray(value).dtype.kind in "iub":
                current = numpy.asarray(value).dtype.type(0)  # a lane no return reaches
            else:
                current = F32(numpy.nan)
        frame["__ret__"] = _where(take, value, current)
        frame["__done__"] = numpy.logical_or(frame["__done__"], effective)

    def _store(self, frame: Dict[str, Any], target: str, index: str, value: Any) -> None:
        pointer = self._lookup(target, frame)
        if isinstance(pointer, CFArray):
            slot = int(self.value(index, frame))
            pointer.items[slot] = _where(self._active(frame), value, pointer.items[slot])
            return
        super()._store(frame, target, index, value)

    def launch(self, arguments: Dict[str, Any], lanes: int) -> None:
        # The off-diagonal weld opens with ONE guarded flat index for both halves,
        # declared const; the index is the same either way.
        function = self.functions[self.entry]
        statements = _items(function["body"])
        opening = "const int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        if statements and statements[0] == opening:
            function = dict(function, body=function["body"].replace(
                opening, opening[len("const "):], 1))
            saved = self.functions[self.entry]
            self.functions[self.entry] = function
            try:
                super().launch(arguments, lanes)
            finally:
                self.functions[self.entry] = saved
            return
        super().launch(arguments, lanes)

    def statement(self, statement: str, frame: Dict[str, Any]) -> None:
        if "?" in statement:
            statement = _parenthesize_arguments(statement)
        if statement == "if (idx >= n_elem) return;":
            return  # every lane of the launch is in range by construction
        if re.fullmatch(r"\(void\) \w+;", statement):
            return
        match = re.fullmatch(r"(\w+) (\w+);", statement)
        if match and match.group(1) in self.structs:
            frame[match.group(2)] = Struct(match.group(1))
            return
        match = re.fullmatch(r"cf (\w+)\[(\d+)\];", statement)
        if match:
            frame[match.group(1)] = CFArray(int(match.group(2)))
            return
        match = re.fullmatch(r"(?:const )?(?:float|int)\* (?:__restrict__ )?(\w+) = (.+);",
                             statement)
        if match:
            frame[match.group(1)] = self.value(match.group(2), frame)
            return
        match = re.fullmatch(r"\*(\w+) = (.+);", statement)
        if match:
            ref = self._lookup(match.group(1), frame)
            if not isinstance(ref, ElementRef):
                raise Unmodelled(f"a store through {match.group(1)}, which is no element")
            value = self.value(match.group(2), frame)
            ref.array.items[ref.index] = _where(self._active(frame), value,
                                                ref.array.items[ref.index])
            return
        match = re.fullmatch(r"(\w+)\.(\w+) = (.+);", statement)
        if match and isinstance(self._lookup(match.group(1), frame), Struct):
            setattr(self._lookup(match.group(1), frame), match.group(2),
                    self.value(match.group(3), frame))
            return
        match = re.fullmatch(r"cf (\w+);", statement)
        if match:
            frame[match.group(1)] = _nan_cf()
            return
        match = re.fullmatch(r"(?:const )?cf (\w+) = (.+);", statement)
        if match:
            value = self.value(match.group(2), frame)
            if not isinstance(value, CF):
                raise Unmodelled(f"cf {match.group(1)} from a {type(value).__name__}")
            frame[match.group(1)] = CF(value.re, value.im)
            return
        match = re.fullmatch(r"(\w+)\.(re|im) = (.+);", statement)
        if match:
            name, part = match.group(1), match.group(2)
            owner = self._owner(name, frame)
            if owner is None or not isinstance(owner[name], CF):
                raise Unmodelled(f"member assignment to {name}")
            value = numpy.asarray(self.value(match.group(3), frame))
            if value.dtype != F32:
                raise Unmodelled(f"{name}.{part} assigned a {value.dtype}")
            old = owner[name]
            new = numpy.where(self._active(frame), value, getattr(old, part))
            owner[name] = CF(new, old.im) if part == "re" else CF(old.re, new)
            return
        super().statement(statement, frame)


class InterpretedKernel:
    """A compiled-kernel stand-in: ``kernel(grid, block, arguments)`` executes ``text``.

    The positional arguments are bound to the entry kernel's parameters in order -- the
    launcher's own argument order is what is exercised -- and every pointer argument is
    the flat VIEW of the array the launcher handed over, so a store lands in the caller's
    volume. One lane per cell of ``nx * ny * nz``.
    """

    def __init__(self, text: str, interpreter=ComplexInterpreter):
        self.text = text
        self.interpreter = interpreter(text)
        self.launches = 0

    def __call__(self, grid_dims, block_dims, arguments) -> None:
        del grid_dims, block_dims
        function = self.interpreter.functions[self.interpreter.entry]
        params = function["params"]
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
        lanes = (int(bound["n_elem"]) if "n_elem" in bound
                 else int(bound["nx"]) * int(bound["ny"]) * int(bound["nz"]))
        self.interpreter.launch(bound, lanes)
        self.launches += 1

    def compile(self) -> None:  # the warm pass's door
        return None


def load_with_numpy_as_cupy(stem: str, private: str) -> Any:
    """``stem``.py loaded by path under ``private``, NumPy wearing CuPy's name for the load.

    The CuPy modules this directory ships fall back to absolute imports when loaded by path;
    ``step_curl_kernels`` is the one of them that imports CuPy too, so it is loaded the same
    way first and handed to the fallback under its package name for the duration of the
    load only (a module bound to the stand-in must not outlive the load in ``sys.modules``).
    """
    saved = {name: sys.modules.get(name)
             for name in ("cupy", "meep_gpu.cuda_kernels.step_curl_kernels")}
    sys.modules["cupy"] = _NumpyWearingCupysName()
    try:
        curl_spec = importlib.util.spec_from_file_location(
            f"{private}__step_curl_kernels", HERE / "step_curl_kernels.py")
        curl = importlib.util.module_from_spec(curl_spec)
        curl_spec.loader.exec_module(curl)
        sys.modules["meep_gpu.cuda_kernels.step_curl_kernels"] = curl
        spec = importlib.util.spec_from_file_location(private, HERE / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


@pytest.fixture(scope="module")
def launcher() -> Any:
    """``complex_pml_kernels`` on NumPy, its compiled kernel replaced by the interpreter."""
    module = load_with_numpy_as_cupy("complex_pml_kernels", "_complex_chunk_forms_pml")
    texts: Dict[str, str] = {}

    def _get_kernel(sub_step):
        text = texts.get(sub_step) or complex_emitter.complex_source(sub_step)
        return InterpretedKernel(text)

    module._get_kernel = _get_kernel
    module._texts = texts  # a defect leg installs a rewritten text here
    return module


# =============================================================================
# The cases
# =============================================================================

#: name -> (cell, dimensions, boundaries, k_point, divide)
CASES: Dict[str, Tuple] = {
    "walls3d": ((9.0, 10.0, 11.0), 3, ("metallic", "metallic", "metallic"), (0, 0, 0), None),
    "periodic3d": ((9.0, 10.0, 11.0), 3, ("periodic", "metallic", "periodic"), (0, 0, 0),
                   None),
    # A generic phase on x and y: the unfused connection on both curls' lattices, with a
    # layer under each wrap.
    "bloch3d": ((9.0, 10.0, 11.0), 3, ("periodic", "periodic", "metallic"),
                (0.13, 0.27, 0.0), None),
    # The Brillouin edge on x (k * n_full == 0.5 * resolution: eikna is exactly -1, the
    # per-part negation) and a phase within the tolerance of +1 on y (k * L == 1: a copy),
    # on a two-dimensional grid (z invariant: the partner-presence forms).
    "edge_copy2d": ((10.0, 10.0, 0.0), 2, ("periodic", "periodic", "periodic"),
                    (0.05, 0.1, 0.0), None),
    # The generic phase on a lifted division whose record differs from the one-rank one.
    "bloch_split3d": ((9.0, 12.0, 8.0), 3, ("periodic", "periodic", "metallic"),
                      (0.21, 0.0, 0.0), "split_y"),
}

#: The connection kinds (B curl, D curl) each case must bind, per axis.
EXPECTED_KINDS = {
    "walls3d": ((0, 0, 0), (0, 0, 0)),
    "periodic3d": ((0, 0, 0), (0, 0, 0)),
    "bloch3d": ((1, 1, 0), (1, 1, 0)),
    "edge_copy2d": ((2, 0, 0), (2, 0, 0)),
    "bloch_split3d": ((1, 0, 0), (1, 0, 0)),
}

#: Every array the four sub-steps read or write, and every other array a step carries.
STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

SUB_STEPS = {
    "step_B": ("curl", ("Bx", "By", "Bz")),
    "step_D": ("curl", ("Dx", "Dy", "Dz")),
    "update_H": ("constitutive", ("Hx", "Hy", "Hz")),
    "update_E": ("constitutive", ("Ex", "Ey", "Ez")),
}


def _split_y(grid: Grid, pml: PML) -> ChunkLayout:
    """The one-rank boxes with the low-y effort volume cut two cells short of its edge;
    the sliver carries no y layer (past the sigma extent), so its rows leave the run."""
    one_rank = ChunkLayout.one_rank(grid, pml)
    window = one_rank.frames[1].window
    low_edge = min(hi[1] for lo, hi in one_rank.boxes if lo[1] == window[0])
    cut = low_edge - 2
    boxes, flags = [], []
    for (lo, hi), flag in zip(one_rank.boxes, one_rank.pml_flags):
        if lo[1] == window[0]:
            boxes.append((lo, (hi[0], cut, hi[2])))
            flags.append(flag)
            boxes.append(((lo[0], cut, lo[2]), hi))
            flags.append((flag[0], False, flag[2]))
        else:
            boxes.append((lo, hi))
            flags.append(flag)
    return ChunkLayout(boxes, flags, grid_frames(grid), faces=pml.thickness_by_face,
                       source="meep")


def build(case: str, plant_zeros: bool, h_separated: bool, seed: int = 11):
    """``(fields, grid, pml)``: a seeded complex full-volume state on ``case``'s grid."""
    cell, dimensions, boundaries, k_point, divide = CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions, boundaries=boundaries,
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    layer = {axis: 2 for axis, n in zip("xyz", shape) if n > 1}
    pml = PML(grid=grid, thickness=layer)
    if divide == "split_y":
        fields.install_region_plan(RegionPlan(_split_y(grid, pml), grid, pml,
                                              backend=region_backend(numpy)))
    for name in STATE:
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        getattr(fields, name)[...] = part_products.from_parts(parts[0], parts[1],
                                                              numpy.complex64)
    fields.h_separated = bool(h_separated)
    return fields, grid, pml


def snapshot(fields) -> Dict[str, numpy.ndarray]:
    return {name: numpy.array(getattr(fields, name), copy=True) for name in STATE}


def restore(fields, state) -> None:
    for name, values in state.items():
        getattr(fields, name)[...] = values


def bits(array) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def differing(want, got) -> Dict[str, int]:
    return {name: int(numpy.count_nonzero(bits(want[name]) != bits(got[name])))
            for name in want}


def oracle(sub_step: str, fields, pml) -> None:
    getattr(stepping, sub_step)(fields, pml)


def device_record(fields) -> numpy.ndarray:
    """The record of the region plan the oracle stepped with (installed on ``fields``)."""
    return numpy.ascontiguousarray(chunk_bounds_record(fields.region_plan.layout),
                                   dtype=numpy.int32)


def launch(launcher, sub_step: str, fields, grid, pml, record, **override) -> None:
    """The family's own launcher, as ``complex_pml_kernels`` exposes it."""
    if sub_step in ("step_B", "step_D"):
        launcher.step_fused_pml_complex(sub_step, fields, grid=grid, pml=pml,
                                        chunk_bounds=record, **override)
    else:
        launcher.update_fused_pml_complex("H" if sub_step == "update_H" else "E", fields,
                                          pml=pml, chunk_bounds=record, **override)


def run(launcher, sub_step: str, case: str, plant: bool, separated: bool,
        record_edit=None, **override):
    """``(start, oracle, kernel, plan)`` for one sub-step on one case."""
    fields, grid, pml = build(case, plant, separated)
    start = snapshot(fields)
    oracle(sub_step, fields, pml)
    want = snapshot(fields)
    plan = fields.region_plan
    record = device_record(fields)
    if record_edit is not None:
        record = record_edit(record.copy())
    restore(fields, start)
    fields.h_separated = bool(separated)
    launch(launcher, sub_step, fields, grid, pml, record, **override)
    return start, want, snapshot(fields), plan, grid


def _selecting_axis(component: str, kind: str) -> str:
    if kind == "curl":
        term = next(t for t in (*stepping.B_CURL_TERMS, *stepping.D_CURL_TERMS)
                    if t.target == component)
        return term.dsigu
    return component[1]


# =============================================================================
# 1. The texts compute the array path, word for word
# =============================================================================

PARAMETERS = [(sub_step, case, plant, separated)
              for sub_step in SUB_STEPS for case in sorted(CASES)
              for plant in (False, True)
              for separated in ((True, False) if sub_step == "update_H" else (True,))]


@pytest.mark.parametrize(
    "sub_step,case,plant,separated", PARAMETERS,
    ids=[f"{s}-{c}-{'zeros' if p else 'random'}-{'midrun' if h else 'first'}"
         for s, c, p, h in PARAMETERS])
def test_the_complex_text_is_the_array_path_word_for_word(sub_step, case, plant, separated,
                                                          launcher):
    start, want, got, plan, grid = run(launcher, sub_step, case, plant, separated)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (sub_step, case, plant, separated,
                                                {k: v for k, v in diff.items() if v})
    kind, components = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in grid.shape)
    for component in components:
        inside = numpy.broadcast_to(plan.mask(kind, component, numpy), shape)
        axis = "xyz".index(_selecting_axis(component, kind))
        if shape[axis] == 1:
            assert int(inside.sum()) == 0, (case, component)
        else:
            assert int(inside.sum()) > 0 and int((~inside).sum()) > 0, (case, component)
        moved = int(numpy.count_nonzero(bits(want[component]) != bits(start[component])))
        assert moved > 0, (sub_step, case, component)


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_case_binds_the_connection_kinds_it_is_built_to_reach(case, launcher):
    _fields, grid, _pml = build(case, False, True)
    for backward, expected in zip((False, True), EXPECTED_KINDS[case]):
        kinds, _values = launcher.bloch_phase_arguments(grid, backward)
        assert tuple(int(kind) for kind in kinds) == expected, (case, backward, kinds)


def test_the_first_update_h_is_not_the_mid_run_case_under_another_name(launcher):
    """The array path's start of H changes the words on these states, so a launcher that
    skipped it would be caught by the first-update cases above."""
    fields, _grid, pml = build("walls3d", False, h_separated=False)
    start = snapshot(fields)
    stepping.update_H(fields, pml)
    first = snapshot(fields)
    restore(fields, start)
    fields.h_separated = True
    stepping.update_H(fields, pml)
    midrun = snapshot(fields)
    assert sum(differing(first, midrun).values()) > 0


# =============================================================================
# 2. The connection the kernel applies is part_products.connect's, word for word
# =============================================================================

PHASES = (complex(-1.0, 0.0), complex(1.0, 0.0), complex(1.0, -2.4e-16),
          complex(-1.0, 3e-15), complex(0.0, 1.0), complex(0.0, -1.0),
          complex(0.6, 0.8), complex(-0.28, 0.96), complex(0.9999999, 0.00044))


@pytest.mark.parametrize("phase", PHASES, ids=[repr(p) for p in PHASES])
def test_the_kernels_connection_is_part_products_connect(phase, launcher):
    rng = numpy.random.default_rng(3)
    lanes = 4096
    parts = [rng.uniform(-1.0, 1.0, size=lanes).astype(F32) for _ in range(2)]
    for part in parts:
        draw = rng.uniform(size=lanes)
        part[draw < 0.1] = F32(-0.0)
        part[(draw >= 0.1) & (draw < 0.2)] = F32(0.0)
    words = part_products.from_parts(parts[0], parts[1], numpy.complex64)
    thephase = (phase.real, phase.imag)
    want = part_products.connect(words, thephase)
    kind, (a, b) = bloch_connection.connection_of(thephase)
    probe = complex_emitter.prelude() + r'''
extern "C" __global__ void connection_probe(
    float* __restrict__ out, const float* __restrict__ g, int nx, int ny, int nz,
    int ph, float pr, float pi
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    cf p; p.re = pr; p.im = pi;
    cf_store(out, idx, cwindow(g, idx, 0, ph, p));
}
'''
    out = numpy.zeros(lanes, dtype=numpy.complex64)
    InterpretedKernel(probe)((1,), (1,), (out.view(F32), words.view(F32), lanes, 1, 1,
                                          int(kind), F32(a), F32(b)))
    assert int(numpy.count_nonzero(bits(out) != bits(want))) == 0, (phase, kind)


# =============================================================================
# 3. The text carries the decode and the flags
# =============================================================================

@pytest.mark.parametrize("sub_step", sorted(complex_emitter.KERNELS))
def test_each_kernel_decodes_the_record_with_the_emitters_block(sub_step):
    from . import bounds_block  # noqa: PLC0415 - NumPy via chunk_map

    source = complex_emitter.complex_source(sub_step)
    if sub_step in complex_emitter.CURL_SUB_STEPS:
        targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
        block = bounds_block.bounds_block(bounds_block.curl_entries(targets), (0, 1, 2))
        flags = complex_emitter.CURL_FLAGS[complex_emitter.KERNELS[sub_step][1]]
        names = [bounds_block.flag_name(*entry) for entry in bounds_block.curl_entries(targets)]
    else:
        side = "H" if sub_step == "update_H" else "E"
        targets = ("Hx", "Hy", "Hz") if side == "H" else ("Ex", "Ey", "Ez")
        block = bounds_block.bounds_block(bounds_block.constitutive_entries(targets))
        flags = complex_emitter.CONSTITUTIVE_FLAGS[side]
        names = [bounds_block.flag_name(*entry)
                 for entry in bounds_block.constitutive_entries(targets)]
    assert source.count(bounds_block.DECODE_END + "\n" + block) == 1, sub_step
    assert tuple(names) == tuple(flags), sub_step


def test_no_product_is_fused_and_no_cross_term_survives():
    for sub_step in complex_emitter.KERNELS:
        source = complex_emitter.complex_source(sub_step)
        assert "__fmaf_rn" not in source and "fma(" not in source, sub_step
        assert "* 0.0f" not in source and "0.0f *" not in source, sub_step


# =============================================================================
# 4. Planted defects change a word
# =============================================================================

def _rewrite(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, old
    return text.replace(old, new)


TEXT_DEFECTS = {
    # The previous package grouping of the stencil (KERNELS.md section 7.1).
    "pairwise_stencil": (("step_B", "step_D"), "walls3d",
                         "cf_sub(cf_add(cf_sub(sf, f1), f2), ss)",
                         "cf_add(cf_sub(sf, f1), cf_sub(f2, ss))"),
    # The select removed: the whole-grid recurrence (section 7.2).
    "select_removed": (("step_B", "step_D"), "walls3d",
                       "int dsigu\n) {\n", "int dsigu_ignored\n) {\n    int dsigu = 1;\n"),
    # The plain constitutive as an accumulation (section 7.2, planted zeros).
    "plain_constitutive_accumulated": (("update_H", "update_E"), "walls3d",
                                       "int dsigw\n) {\n",
                                       "int dsigw_ignored\n) {\n    int dsigw = 1;\n"),
    # The auxiliary stored on a plain cell (section 7.2).
    "aux_stored_off_the_run": (("step_B", "step_D"), "walls3d",
                               "    if (dsigu) cf_store(fu, idx, fu_new);\n",
                               "    cf_store(fu, idx, fu_new);\n"),
    # The previous constitutive grouping (section 7.1 inversion).
    "constitutive_regrouped": (("update_H", "update_E"), "walls3d",
                               "cf_add(cf_load(f, idx), cf_sub(mul_coefficient_left(kps, src), "
                               "mul_coefficient_left(kms, prev)))",
                               "cf_sub(cf_add(cf_load(f, idx), mul_coefficient_left(kps, src)), "
                               "mul_coefficient_left(kms, prev))"),
    # The previous zero-cross-term product (a full complex multiply against (c + 0j)).
    "zero_cross_term_product": (("step_B", "step_D", "update_H", "update_E"), "walls3d",
                                "    o.re = z.re * c;\n    o.im = z.im * c;\n",
                                "    o.re = (z.re * c) - (z.im * 0.0f);\n"
                                "    o.im = (z.re * 0.0f) + (z.im * c);\n"),
    # The connection dropped: the stored word read across the window edge.
    "window_dropped": (("step_B", "step_D"), "bloch3d",
                       "    if (ia != 0 || ph == 0) return w;\n", "    return w;\n"),
    # The connection on the far plane instead of plane 0.
    "window_on_the_far_plane": (("step_B", "step_D"), "bloch3d",
                                "    if (ia != 0 || ph == 0) return w;\n",
                                "    if (ia != 1 || ph == 0) return w;\n"),
    # The absent-partner forms replaced by the full bracket (presence ignored).
    "presence_ignored": (("step_B", "step_D"), "edge_copy2d",
                         "    if (pr1 && pr2) return", "    if (1) return"),
}


@pytest.mark.parametrize("defect", sorted(TEXT_DEFECTS))
def test_each_planted_text_defect_changes_a_word(defect, launcher):
    sub_steps, case, old, new = TEXT_DEFECTS[defect]
    changed = 0
    for sub_step in sub_steps:
        launcher._texts[sub_step] = _rewrite(complex_emitter.complex_source(sub_step),
                                             old, new)
        try:
            _start, want, got, _plan, _grid = run(launcher, sub_step, case, True, True)
        finally:
            launcher._texts.clear()
        changed += sum(differing(want, got).values())
    assert changed > 0, defect


def _offset_record(record: numpy.ndarray) -> numpy.ndarray:
    from ..chunk_map import record_index  # noqa: PLC0415

    record[record_index(True, 0, 1)] += 1   # the half-integer x run's a1, one row late
    record[record_index(False, 0, 1)] += 1
    return record


def _swapped_record(record: numpy.ndarray) -> numpy.ndarray:
    integer, half = record[0:12].copy(), record[12:24].copy()
    record[0:12], record[12:24] = half, integer
    return record


@pytest.mark.parametrize("edit", (_offset_record, _swapped_record),
                         ids=("bound_off_by_one", "lattices_swapped"))
def test_a_wrong_record_changes_a_word(edit, launcher):
    changed = 0
    for sub_step in SUB_STEPS:
        _start, want, got, _plan, _grid = run(launcher, sub_step, "bloch_split3d", False, True,
                                              record_edit=edit)
        changed += sum(differing(want, got).values())
    assert changed > 0


def test_the_b_curl_unconjugated_changes_a_word(launcher):
    """The D curl's factor handed to the B curl: the classic sign error."""
    fields, grid, _pml = build("bloch3d", False, True)
    kinds, values = launcher.bloch_phase_arguments(grid, True)
    _start, want, got, _plan, _grid = run(launcher, "step_B", "bloch3d", False, True,
                                          phase_flags=kinds, phase_values=values)
    assert sum(differing(want, got).values()) > 0


def test_a_negation_bound_as_a_product_is_not_the_connection(launcher):
    """``(-1 + 0j) * word`` is not ``-word``: on a zero real part with a negative imaginary
    part the product's real part is ``(-1 * +0) - (0 * im) = +0`` where the negation gives
    ``-0``. The kind therefore matters; in a stepped state the sign of that zero is absorbed
    unless the whole stencil around it is zero, so it is checked here, at the connection."""
    words = part_products.from_parts(numpy.array([0.0, -0.0, 0.5], dtype=F32),
                                     numpy.array([-0.25, 0.75, -0.0], dtype=F32),
                                     numpy.complex64)
    want = part_products.connect(words, (-1.0, 0.0))
    kind, (a, b) = bloch_connection.connection_of((-1.0, 0.0))
    assert kind == complex_emitter.CONNECT_NEGATE
    probe = complex_emitter.prelude() + r'''
extern "C" __global__ void connection_probe(
    float* __restrict__ out, const float* __restrict__ g, int nx, int ny, int nz,
    int ph, float pr, float pi
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    cf p; p.re = pr; p.im = pi;
    cf_store(out, idx, cwindow(g, idx, 0, ph, p));
}
'''
    outs = {}
    for label, bound in (("negate", kind), ("product", complex_emitter.CONNECT_PHASE)):
        out = numpy.zeros(3, dtype=numpy.complex64)
        InterpretedKernel(probe)((1,), (1,), (out.view(F32), words.view(F32), 3, 1, 1,
                                              int(bound), F32(a), F32(b)))
        outs[label] = out
    assert int(numpy.count_nonzero(bits(outs["negate"]) != bits(want))) == 0
    assert int(numpy.count_nonzero(bits(outs["product"]) != bits(want))) > 0


# =============================================================================
# 5. The folded curl pair: the emitter's text through the fold deltas
# =============================================================================

#: name -> (cell, folded axes, plane phase, boundaries, k_point, dimensions)
FOLD_CASES: Dict[str, Tuple] = {
    "fold_y_periodic": ((8.0, 16.0, 0.0), "Y", 1, ("periodic",) * 3, (0, 0, 0), 2),
    "fold_y_metallic": ((8.0, 16.0, 0.0), "Y", 1, ("periodic", "metallic", "periodic"),
                        (0, 0, 0), 2),
    "fold_y_odd_plane": ((8.0, 16.0, 0.0), "Y", -1, ("periodic",) * 3, (0, 0, 0), 2),
    "fold_y_odd_count": ((8.0, 15.0, 0.0), "Y", 1, ("periodic",) * 3, (0, 0, 0), 2),
    # A phase on the unfolded axis beside the fold: the window on x, the fold on y.
    "fold_y_kx": ((8.0, 16.0, 0.0), "Y", 1, ("periodic",) * 3, (0.31, 0, 0), 2),
    "fold_xy_metallic": ((16.0, 16.0, 0.0), "XY", 1, ("metallic", "metallic", "periodic"),
                         (0, 0, 0), 2),
    "fold_x_3d_phased": ((12.0, 8.0, 10.0), "X", 1, ("periodic",) * 3, (0, 0.19, 0.27), 3),
}


def build_fold(case: str, plant_zeros: bool, seed: int = 5):
    from ..grid import Mirror  # noqa: PLC0415

    cell, axes, phase, boundaries, k_point, dimensions = FOLD_CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis in axes),
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis) else (2, 2)
                      for axis in range(3))
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    for name in STATE:
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        getattr(fields, name)[...] = part_products.from_parts(parts[0], parts[1],
                                                              numpy.complex64)
    fields.h_separated = True
    return fields, grid, pml


@pytest.fixture()
def folded_launcher(launcher, monkeypatch):
    """``complex_folded_kernels.step_folded_complex`` with its CuPy imports answered by the
    NumPy-loaded ``complex_pml_kernels`` and its compiled kernel by the interpreter."""
    from . import complex_folded_kernels  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    monkeypatch.setattr(complex_folded_kernels, "_get_kernel",
                        lambda sub_step: InterpretedKernel(
                            complex_folded_kernels.kernel_source(sub_step)))
    complex_folded_kernels.reset_kernel_sources()
    yield complex_folded_kernels
    complex_folded_kernels.reset_kernel_sources()


FOLD_PARAMETERS = [(sub_step, case, plant) for sub_step in ("step_B", "step_D")
                   for case in sorted(FOLD_CASES) for plant in (False, True)]


@pytest.mark.parametrize("sub_step,case,plant", FOLD_PARAMETERS,
                         ids=[f"{s}-{c}-{'zeros' if p else 'random'}"
                              for s, c, p in FOLD_PARAMETERS])
def test_the_folded_curl_text_is_the_array_path_word_for_word(sub_step, case, plant,
                                                             folded_launcher):
    fields, grid, pml = build_fold(case, plant)
    start = snapshot(fields)
    oracle(sub_step, fields, pml)
    want = snapshot(fields)
    record = device_record(fields)
    restore(fields, start)
    folded_launcher.step_folded_complex(sub_step, fields, grid=grid, pml=pml,
                                        chunk_bounds=record)
    got = snapshot(fields)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (sub_step, case,
                                                {k: v for k, v in diff.items() if v})
    targets = SUB_STEPS[sub_step][1]
    assert all(int(numpy.count_nonzero(bits(want[t]) != bits(start[t]))) > 0 for t in targets)


def test_the_fold_case_with_a_phase_binds_the_product_beside_the_fold(launcher):
    for case, axis in (("fold_y_kx", 0), ("fold_x_3d_phased", 1), ("fold_x_3d_phased", 2)):
        _fields, grid, _pml = build_fold(case, False)
        for backward in (False, True):
            kinds, _values = launcher.bloch_phase_arguments(grid, backward)
            assert int(kinds[axis]) == complex_emitter.CONNECT_PHASE, (case, axis)


# =============================================================================
# 6. The no-absorber family (A forms only): the curl pair, its conductive arm, stored E
# =============================================================================

#: name -> (cell, dimensions, boundaries, k_point, conductive, poles)
NOAB_CASES: Dict[str, Tuple] = {
    "noab_bloch3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.2, 0.31, 0.0), False, 0),
    "noab_edge2d": ((10.0, 9.0, 0.0), 2, ("periodic",) * 3, (0.05, 0.13, 0.0), False, 0),
    "noab_walls3d": ((7.0, 8.0, 9.0), 3, ("periodic", "metallic", "metallic"),
                     (0.17, 0.0, 0.0), False, 0),
    "noab_conductive3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.2, 0.0, 0.29), True, 0),
    "noab_poles3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (0.2, 0.0, 0.0), False, 2),
}


def build_noab(case: str, plant_zeros: bool, seed: int = 13):
    from ..dispersion import PolarizationState, Susceptibility  # noqa: PLC0415

    cell, dimensions, boundaries, k_point, conductive, poles = NOAB_CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions, boundaries=boundaries,
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    if conductive:
        fields.set_d_conductivity(rng.uniform(0.0, 0.4, size=shape).astype(F32))
        fields.set_b_conductivity(rng.uniform(0.0, 0.4, size=shape).astype(F32))
    for index in range(poles):
        state = PolarizationState(Susceptibility(frequency=1.1 + index, gamma=0.05),
                                  numpy.full(shape, F32(0.35)), grid,
                                  fields._field_dtype())
        fields.polarizations.append(state)

    def _random():
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        return part_products.from_parts(parts[0], parts[1], numpy.complex64)

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        getattr(fields, name)[...] = _random()
    for state in fields.polarizations:
        for component in state.P:
            state.P[component][...] = _random() * numpy.complex64(1e-2)
    return fields, grid


NOAB_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")


def _noab_snapshot(fields):
    return {name: numpy.array(getattr(fields, name), copy=True) for name in NOAB_STATE}


@pytest.fixture()
def noab_family(launcher, monkeypatch):
    """``complex_no_pml_kernels``' launchers with NumPy wearing CuPy's name and the
    interpreter for the compiled kernel."""
    from . import complex_no_pml_kernels  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    monkeypatch.setattr(complex_no_pml_kernels, "_get_kernel",
                        lambda key: InterpretedKernel(
                            complex_no_pml_kernels.kernel_source(key)))
    return complex_no_pml_kernels


NOAB_PARAMETERS = [(sub_step, case, plant) for sub_step in ("step_B", "step_D", "update_E")
                   for case in sorted(NOAB_CASES) for plant in (False, True)]


@pytest.mark.parametrize("sub_step,case,plant", NOAB_PARAMETERS,
                         ids=[f"{s}-{c}-{'zeros' if p else 'random'}"
                              for s, c, p in NOAB_PARAMETERS])
def test_the_no_absorber_text_is_the_array_path_word_for_word(sub_step, case, plant,
                                                             noab_family):
    fields, grid = build_noab(case, plant)
    start = _noab_snapshot(fields)
    getattr(stepping, sub_step)(fields, None)
    want = _noab_snapshot(fields)
    for name, values in start.items():
        getattr(fields, name)[...] = values
    if sub_step == "update_E":
        noab_family.update_E_complex_no_pml_stored(fields)
    else:
        record = numpy.ascontiguousarray(
            chunk_bounds_record(ChunkLayout.one_rank(grid, None)), dtype=numpy.int32)
        noab_family.step_complex_no_pml_curl(fields, sub_step, grid,
                                             chunk_bounds=record)
    got = _noab_snapshot(fields)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (sub_step, case,
                                                {k: v for k, v in diff.items() if v})
    targets = {"step_B": ("Bx", "By", "Bz"), "step_D": ("Dx", "Dy", "Dz"),
               "update_E": ("Ex", "Ey", "Ez")}[sub_step]
    assert all(int(numpy.count_nonzero(bits(want[t]) != bits(start[t]))) > 0 for t in targets)


# =============================================================================
# 7. The fused pairs: step_B -> zero_metal_B -> update_H, step_D -> zero_metal_D -> update_E
# =============================================================================

#: The bindings a fused family takes from ``complex_pml_kernels`` (CuPy at scope), answered
#: by the NumPy-loaded launcher.
_LAUNCHER_BINDINGS = ("bloch_phase_arguments", "chunk_record_argument",
                      "complex_boundary_codes", "complex_constitutive_tables",
                      "complex_curl_tables", "word_view")

FUSED_SIDES = {
    "B": ("complex_fused_magnetic_pair", ("step_B", "zero_metal_B", "update_H"), False),
    "D": ("complex_fused_electric_pair", ("step_D", "zero_metal_D", "update_E"), True),
}


def bind_launcher(module, launcher, monkeypatch) -> None:
    for name in _LAUNCHER_BINDINGS:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, getattr(launcher, name))


def fused_oracle(passes, fields, pml) -> None:
    for name in passes:
        function = getattr(stepping, name)
        if name.startswith("zero_metal") or name.startswith("fill_"):
            function(fields)
        else:
            function(fields, pml)


FUSED_PARAMETERS = [(side, case, plant, separated)
                    for side in ("B", "D") for case in sorted(CASES)
                    for plant in (False, True)
                    for separated in ((True, False) if side == "B" else (True,))]


@pytest.mark.parametrize(
    "side,case,plant,separated", FUSED_PARAMETERS,
    ids=[f"{s}-{c}-{'zeros' if p else 'random'}-{'midrun' if h else 'first'}"
         for s, c, p, h in FUSED_PARAMETERS])
def test_the_fused_complex_text_is_the_array_path_word_for_word(side, case, plant, separated,
                                                                launcher, monkeypatch):
    import importlib  # noqa: PLC0415

    stem, passes, backward = FUSED_SIDES[side]
    family = importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")
    bind_launcher(family, launcher, monkeypatch)
    fields, grid, pml = build(case, plant, separated)
    start = snapshot(fields)
    fused_oracle(passes, fields, pml)
    want = snapshot(fields)
    record = device_record(fields)
    restore(fields, start)
    fields.h_separated = bool(separated)
    from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

    tables = getattr(family, f"{stem}_tables")(pml)
    flags, values = launcher.bloch_phase_arguments(grid, backward)
    text = getattr(family, f"{stem}_source")()
    kernel = InterpretedKernel(text)
    extra = {"layer": pml} if side == "B" else {}
    getattr(family, f"launch_{stem}")(
        fields, tables, launcher.complex_boundary_codes(grid), flags, values,
        zero_metal_axes(grid), float(grid.dt / grid.dx), kernel=kernel,
        chunk_bounds=record, **extra)
    assert kernel.launches == 1
    got = snapshot(fields)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (side, case, plant, separated,
                                                {k: v for k, v in diff.items() if v})


def test_the_fused_magnetic_pair_refuses_a_first_launch_with_no_layer(launcher, monkeypatch):
    from . import complex_fused_magnetic_pair as family  # noqa: PLC0415
    from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

    bind_launcher(family, launcher, monkeypatch)
    fields, grid, pml = build("walls3d", False, h_separated=False)
    stepping.step_B(fields, pml)
    record = device_record(fields)
    flags, values = launcher.bloch_phase_arguments(grid, False)
    with pytest.raises(RuntimeError, match="no layer to start it with"):
        family.launch_complex_fused_magnetic_pair(
            fields, family.complex_fused_magnetic_pair_tables(pml),
            launcher.complex_boundary_codes(grid), flags, values, zero_metal_axes(grid),
            float(grid.dt / grid.dx),
            kernel=InterpretedKernel(family.complex_fused_magnetic_pair_source()),
            chunk_bounds=record)


@pytest.mark.parametrize("stem", ("complex_fused_magnetic_pair", "complex_fused_electric_pair"))
def test_each_fused_pair_decodes_the_record_once(stem):
    import importlib  # noqa: PLC0415

    from . import bounds_block  # noqa: PLC0415

    family = importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")
    text = getattr(family, f"{stem}_source")()
    assert text.count("cb[24]") == 1 and text.count(bounds_block.DECODE_END) == 1
    assert text.count(bounds_block.DECODE_END + "\n" + family.union_bounds_block()) == 1


# =============================================================================
# 8. The folded fused pairs: five driver passes in one launch, the two fills carried
# =============================================================================

FOLDED_FUSED_SIDES = {
    "B": ("complex_folded_fused_magnetic_pair",
          ("step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
           "update_H"), False),
    "D": ("complex_folded_fused_electric_pair",
          ("step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
           "update_E"), True),
}

#: name -> (cell, folded axes, plane phase, boundaries, k_point, thickness, lifted)
FOLDED_FUSED_CASES: Dict[str, Tuple] = {
    "ff_y_periodic": ((9.0, 10.0, 11.0), "Y", 1, ("periodic",) * 3, (0, 0, 0),
                      ((2, 2), (0, 2), (2, 2)), False),
    "ff_y_metallic_kx": ((9.0, 10.0, 11.0), "Y", -1, ("periodic", "metallic", "periodic"),
                         (0.23, 0, 0), ((2, 2), (0, 2), (2, 2)), False),
    "ff_y_wall_z": ((9.0, 10.0, 11.0), "Y", 1, ("periodic", "periodic", "metallic"),
                    (0, 0, 0), ((2, 2), (0, 2), (2, 2)), False),
    "ff_xy_mixed": ((9.0, 10.0, 11.0), "XY", -1, ("periodic",) * 3, (0, 0, 0.19),
                    ((0, 2), (0, 2), (2, 2)), False),
    "ff_x_deep": ((9.0, 10.0, 11.0), "X", 1, ("periodic",) * 3, (0, 0, 0),
                  ((0, 5), (2, 2), (2, 2)), False),
    # The same grid on a lifted division: its membership line copies the near ghost's
    # image row onto the ghost row (chunk_map, D5), where the one-rank line does not.
    "ff_x_deep_lifted": ((9.0, 10.0, 11.0), "X", 1, ("periodic",) * 3, (0, 0, 0),
                         ((0, 5), (2, 2), (2, 2)), True),
}


def build_folded_fused(case: str, plant_zeros: bool, h_separated: bool, seed: int = 17):
    from ..grid import Mirror  # noqa: PLC0415

    cell, axes, phase, boundaries, k_point, thickness, lifted = FOLDED_FUSED_CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=3, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis in axes),
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    if lifted:
        one_rank = ChunkLayout.one_rank(grid, pml)
        layout = ChunkLayout(one_rank.boxes, one_rank.pml_flags, grid_frames(grid),
                             faces=pml.thickness_by_face, source="meep")
        fields.install_region_plan(RegionPlan(layout, grid, pml,
                                              backend=region_backend(numpy)))
    for name in STATE:
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        getattr(fields, name)[...] = part_products.from_parts(parts[0], parts[1],
                                                              numpy.complex64)
    fields.h_separated = bool(h_separated)
    return fields, grid, pml


FOLDED_FUSED_PARAMETERS = [(side, case, plant, separated)
                           for side in ("B", "D") for case in sorted(FOLDED_FUSED_CASES)
                           for plant in (False, True)
                           for separated in ((True, False) if side == "B" else (True,))]


@pytest.mark.parametrize(
    "side,case,plant,separated", FOLDED_FUSED_PARAMETERS,
    ids=[f"{s}-{c}-{'zeros' if p else 'random'}-{'midrun' if h else 'first'}"
         for s, c, p, h in FOLDED_FUSED_PARAMETERS])
def test_the_folded_fused_complex_text_is_the_array_path_word_for_word(
        side, case, plant, separated, launcher, monkeypatch):
    import importlib  # noqa: PLC0415

    from . import complex_folded_kernels  # noqa: PLC0415
    from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

    stem, passes, backward = FOLDED_FUSED_SIDES[side]
    family = importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")
    bind_launcher(family, launcher, monkeypatch)
    fields, grid, pml = build_folded_fused(case, plant, separated)
    start = snapshot(fields)
    fused_oracle(passes, fields, pml)
    want = snapshot(fields)
    record = device_record(fields)
    restore(fields, start)
    fields.h_separated = bool(separated)
    codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
    assert refusal is None, refusal
    flags, values = launcher.bloch_phase_arguments(grid, backward)
    kernel = InterpretedKernel(getattr(family, f"{stem}_source")())
    extra = {"layer": pml} if side == "B" else {}
    getattr(family, f"launch_{stem}")(
        fields, getattr(family, f"{stem}_tables")(pml), codes, flags, values,
        zero_metal_axes(grid), getattr(family, f"{stem}_fills")(grid),
        float(grid.dt / grid.dx), kernel=kernel, chunk_bounds=record, **extra)
    assert kernel.launches == 1
    got = snapshot(fields)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (side, case, plant, separated,
                                                {k: v for k, v in diff.items() if v})


def test_the_lifted_division_moves_a_ghost_rows_membership():
    """The lifted case is not the one-rank case under another name: its record puts the
    near ghost row inside the x layer where the one-rank record does not, so a carry keyed
    at the source rather than the destination would be caught there."""
    fields_a, _grid, _pml = build_folded_fused("ff_x_deep", False, True)
    fields_b, grid, pml = build_folded_fused("ff_x_deep_lifted", False, True)
    stepping.update_H(fields_a, _pml)
    stepping.update_H(fields_b, pml)
    assert not numpy.array_equal(device_record(fields_a), device_record(fields_b))


# =============================================================================
# 9. The H->D weld: update_H welded into step_D, scratch output, foreign recompute
# =============================================================================

HD_PARAMETERS = ([("plain", case, plant, separated) for case in sorted(CASES)
                  for plant in (False, True) for separated in (True, False)]
                 + [("folded", case, plant, True) for case in sorted(FOLDED_FUSED_CASES)
                    for plant in (False, True)])


@pytest.mark.parametrize(
    "variant,case,plant,separated", HD_PARAMETERS,
    ids=[f"{v}-{c}-{'zeros' if p else 'random'}-{'midrun' if h else 'first'}"
         for v, c, p, h in HD_PARAMETERS])
def test_the_hd_weld_text_is_the_array_path_word_for_word(variant, case, plant, separated,
                                                          launcher, monkeypatch):
    from . import complex_fused_hd_pair as family  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setattr(family, "complex_pml_kernels", launcher)
    if variant == "plain":
        fields, grid, pml = build(case, plant, separated)
    else:
        fields, grid, pml = build_folded_fused(case, plant, separated)
    start = snapshot(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    want = snapshot(fields)
    record = device_record(fields)
    restore(fields, start)
    fields.h_separated = bool(separated)
    # The tables are handed in: resolve's identity check compares device base addresses,
    # which NumPy arrays do not carry; the two INTEGER groups are the same vectors.
    state = family.resolve(fields, grid, pml, chunk_bounds=record,
                           tables=launcher.complex_curl_tables(pml, False),
                           constitutive=launcher.complex_constitutive_tables(pml, False))
    assert state["variant"] == variant
    kernel = InterpretedKernel(family.kernel_source(variant))
    family.launch_complex_fused_hd_pair(fields, state, kernel=kernel)
    state["scratch"] = family.rotate_into_fields(fields, state["scratch"])
    assert kernel.launches == 1
    got = snapshot(fields)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (variant, case, plant, separated,
                                                {k: v for k, v in diff.items() if v})


# =============================================================================
# 10. The off-diagonal row: MEEP's double row per part, presence, two connections
# =============================================================================

#: name -> (cell, dimensions, boundaries, symmetry, k_point, layer, row mask, block)
OFFDIAG_CASES: Dict[str, Tuple] = {
    "od_walls3d": ((8.0, 9.0, 10.0), 3, ("metallic",) * 3, (), (0, 0, 0), True,
                   (1, 1, 1, 1, 1, 1), False),
    "od_bloch3d": ((8.0, 9.0, 10.0), 3, ("periodic", "periodic", "metallic"), (),
                   (0.13, 0.27, 0.0), True, (1, 1, 1, 1, 1, 1), False),
    # Rows on an interior block: the chunks that carry a layer delete every row's array.
    "od_block3d": ((10.0, 10.0, 10.0), 3, ("periodic", "metallic", "metallic"), (),
                   (0.19, 0.0, 0.0), True, (1, 1, 1, 1, 1, 1), True),
    "od_corpus3d": ((8.0, 9.0, 10.0), 3, ("periodic", "periodic", "metallic"), (),
                    (0.05, 0.11, 0.0), True, (1, 0, 0, 1, 0, 0), False),
    # A walled fold (the one-rank record names its chunks; KC0's product rule refuses a
    # periodic fold with a layer for this family).
    "od_fold_walled": ((9.0, 10.0, 11.0), 3, ("metallic",) * 3, (("Y", -1),), (0, 0, 0),
                       True, (1, 1, 1, 1, 1, 1), False),
    # No absorber: the store arm. The Brillouin edge on x (a negation), 2-D.
    "od_noab_edge2d": ((10.0, 9.0, 0.0), 2, ("periodic",) * 3, (), (0.05, 0.13, 0.0),
                       False, (1, 1, 1, 1, 1, 1), False),
    "od_noab3d": ((7.0, 8.0, 9.0), 3, ("periodic",) * 3, (), (0.23, -0.17, 0.35), False,
                  (1, 0, 0, 1, 0, 0), False),
}


def build_offdiag(case: str, plant_zeros: bool, seed: int = 23):
    from ..grid import Mirror  # noqa: PLC0415
    from .coverage import OFFDIAG_ROW_SLOTS  # noqa: PLC0415

    cell, dimensions, boundaries, symmetry, k_point, layered, mask, block = \
        OFFDIAG_CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry),
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    pml = None
    if layered:
        # Layers on the walled axes only: a periodic axis carrying a layer is not a {low,
        # interior, high} chunk product along it, and this family indexes MEEP's chunks
        # there (chunk_coverage.chunk_division_refusal, CHUNK_AXES_OFFDIAG).
        thickness = tuple((0, 0) if grid.shape[axis] < 6 or not grid.is_metallic(axis)
                          else (0, 2) if grid.is_mirrored(axis) else (2, 2)
                          for axis in range(3))
        pml = PML(grid=grid, thickness=thickness)
        fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=shape).astype(F32)
        epsilon[component] = values
        inverse[component] = (F32(1.0) / values).astype(F32)
    rows: Dict[str, Dict[str, Any]] = {}
    for slot, (row, partner) in enumerate(OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        values = rng.uniform(-0.45, 0.45, size=shape).astype(F32)
        if block:
            inside = tuple(slice(n // 3, n - n // 3) if n > 2 else slice(None)
                           for n in shape)
            kept = numpy.zeros(shape, dtype=F32)
            kept[inside] = values[inside]
            values = kept
        rows.setdefault(row, {})[partner] = values
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    names = ["Dx", "Dy", "Dz", "Ex", "Ey", "Ez"] + (
        ["f_w_Ex", "f_w_Ey", "f_w_Ez"] if layered else [])
    for name in names:
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        getattr(fields, name)[...] = part_products.from_parts(parts[0], parts[1],
                                                              numpy.complex64)
    return fields, grid, pml, tuple(names)


OFFDIAG_PARAMETERS = [(case, plant) for case in sorted(OFFDIAG_CASES)
                      for plant in (False, True)]


@pytest.mark.parametrize("case,plant", OFFDIAG_PARAMETERS,
                         ids=[f"{c}-{'zeros' if p else 'random'}"
                              for c, p in OFFDIAG_PARAMETERS])
def test_the_offdiagonal_complex_text_is_the_array_path_word_for_word(case, plant, launcher,
                                                                      monkeypatch):
    from . import complex_offdiag_update_e as family  # noqa: PLC0415
    from .coverage import offdiag_wall_mask_flags  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    fields, grid, pml, names = build_offdiag(case, plant)
    start = {name: numpy.array(getattr(fields, name), copy=True) for name in names}
    stepping.update_E(fields, pml)
    want = {name: numpy.array(getattr(fields, name), copy=True) for name in names}
    for name, values in start.items():
        getattr(fields, name)[...] = values
    codes = family.complex_offdiag_boundary_codes(grid)
    record, kept = family.chunk_arguments(fields, pml, codes)
    texts: Dict[str, str] = {}
    monkeypatch.setattr(family, "_get_kernel",
                        lambda arm, mask: InterpretedKernel(
                            texts.setdefault(arm, family.kernel_source(arm, mask))))
    overrides = dict(codes=codes, walls=offdiag_wall_mask_flags(grid),
                     weights=family.mirror_ghost_weights(grid),
                     connections=family.bloch_connection_table(grid),
                     chunk_bounds=numpy.asarray(record, dtype=numpy.int32), kept=kept)
    if pml is not None:
        family.update_E_complex_offdiag_fused_pml(
            fields, tables=launcher.complex_constitutive_tables(pml, True), **overrides)
    else:
        family.update_E_complex_no_pml_offdiag(fields, **overrides)
    got = {name: numpy.array(getattr(fields, name), copy=True) for name in names}
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (case, plant,
                                                {k: v for k, v in diff.items() if v})
    assert all(int(numpy.count_nonzero(bits(want[c]) != bits(start[c]))) > 0
               for c in ("Ex", "Ey", "Ez"))


def test_the_block_case_deletes_rows_in_some_chunks(launcher, monkeypatch):
    from . import complex_offdiag_update_e as family  # noqa: PLC0415

    fields, grid, pml, _names = build_offdiag("od_block3d", False)
    _record, kept = family.chunk_arguments(fields, pml,
                                           family.complex_offdiag_boundary_codes(grid))
    from .offdiag_presence import ALL_CHUNKS_KEPT  # noqa: PLC0415

    assert any(word not in (0, ALL_CHUNKS_KEPT) for word in kept)


# =============================================================================
# 11. The folded off-diagonal stencil weld: five driver passes, scratch output, recompute
# =============================================================================

#: name -> (cell, boundaries, symmetry, k_point, thickness, row mask). Layers sit on the
#: metallic axes only (the off-diagonal row indexes MEEP's chunks, and a periodic axis
#: carrying a layer is not a {low, interior, high} product along it).
WELD_CASES: Dict[str, Tuple] = {
    "wf_y_walled": ((9.0, 10.0, 11.0), ("metallic",) * 3, (("Y", -1),), (0, 0, 0),
                    ((2, 2), (0, 2), (2, 2)), (1, 1, 1, 1, 1, 1)),
    # A folded PERIODIC axis (the far fill runs) beside a generic phase on z.
    "wf_y_periodic_kz": ((9.0, 10.0, 11.0), ("metallic", "periodic", "periodic"),
                         (("Y", 1),), (0, 0, 0.19), ((2, 2), (0, 0), (0, 0)),
                         (1, 1, 1, 1, 1, 1)),
    "wf_x_periodic_corpus": ((9.0, 10.0, 11.0), ("periodic", "metallic", "metallic"),
                             (("X", -1),), (0, 0, 0), ((0, 0), (2, 2), (2, 2)),
                             (1, 0, 0, 1, 0, 0)),
    # Two folds (one of each parity) and the Brillouin edge on y (a negation).
    "wf_xz_edge_y": ((9.0, 10.0, 11.0), ("metallic", "periodic", "metallic"),
                     (("X", 1), ("Z", -1)), (0, 0.05, 0), ((0, 2), (0, 0), (0, 2)),
                     (1, 1, 1, 1, 1, 1)),
}

WELD_STATE = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz",
              "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def build_weld(case: str, plant_zeros: bool, seed: int = 29):
    from ..grid import Mirror  # noqa: PLC0415
    from .coverage import OFFDIAG_ROW_SLOTS  # noqa: PLC0415

    cell, boundaries, symmetry, k_point, thickness, mask = WELD_CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=3, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry),
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5)
    pml = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=shape).astype(F32)
        epsilon[component] = values
        inverse[component] = (F32(1.0) / values).astype(F32)
    rows: Dict[str, Dict[str, Any]] = {}
    for slot, (row, partner) in enumerate(OFFDIAG_ROW_SLOTS):
        if mask[slot]:
            rows.setdefault(row, {})[partner] = rng.uniform(
                -0.45, 0.45, size=shape).astype(F32)
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    for name in WELD_STATE:
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        getattr(fields, name)[...] = part_products.from_parts(parts[0], parts[1],
                                                              numpy.complex64)
    return fields, grid, pml


WELD_PASSES = ("step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
               "update_E")

WELD_PARAMETERS = [(case, plant) for case in sorted(WELD_CASES) for plant in (False, True)]


def _weld_launch(family, launcher, fields, grid, pml, record, kernel_text=None):
    """The family's own launcher on the interpreter, with every host fact the family's own
    resolver derives; returns the interpreted kernel."""
    from . import complex_offdiag_stencil_weld as weld  # noqa: PLC0415
    from . import complex_offdiag_update_e as offdiag  # noqa: PLC0415
    from .coverage import (offdiag_row_mask, offdiag_row_volumes,  # noqa: PLC0415
                           offdiag_wall_mask_flags)

    codes = family.folded_complex_offdiag_fused_electric_pair_codes(grid)
    phases = family.folded_complex_offdiag_fused_electric_pair_phases(grid)
    _record, kept = family.folded_complex_offdiag_fused_electric_pair_chunk_arguments(
        fields, pml, codes, record)
    wall_mask = offdiag_wall_mask_flags(grid)
    weights = offdiag.mirror_ghost_weights(grid)
    fills = weld.fill_plan(grid)
    family._validate_launch_arguments(tuple(int(n) for n in fields.Dx.shape), codes,
                                      wall_mask, weights, fills)
    mask = offdiag.normalized_row_mask(offdiag_row_mask(fields))
    rows = [volume for volume in offdiag_row_volumes(fields) if volume is not None]
    tables = {"curl": launcher.complex_curl_tables(pml, False),
              "constitutive": launcher.complex_constitutive_tables(pml, True)}
    nan = numpy.complex64(complex(numpy.nan, numpy.nan))
    scratch = {name: numpy.full_like(getattr(fields, name), nan)
               for name in family.SCRATCH_VOLUMES}
    kernel = InterpretedKernel(kernel_text if kernel_text is not None
                               else family.kernel_source(mask))
    family.launch_folded_complex_offdiag_fused_electric_pair(
        fields, scratch, tables, codes, wall_mask, phases, weights, fills,
        float(grid.dt / grid.dx), mask, rows, kept, record, kernel=kernel)
    family.rotate_into_fields(fields, scratch)
    return kernel


@pytest.mark.parametrize("case,plant", WELD_PARAMETERS,
                         ids=[f"{c}-{'zeros' if p else 'random'}"
                              for c, p in WELD_PARAMETERS])
def test_the_folded_offdiag_weld_text_is_the_array_path_word_for_word(case, plant, launcher,
                                                                      monkeypatch):
    from . import folded_complex_offdiag_fused_electric_pair as family  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    bind_launcher(family, launcher, monkeypatch)
    fields, grid, pml = build_weld(case, plant)
    start = {name: numpy.array(getattr(fields, name), copy=True) for name in WELD_STATE}
    fused_oracle(WELD_PASSES, fields, pml)
    want = {name: numpy.array(getattr(fields, name), copy=True) for name in WELD_STATE}
    record = device_record(fields)
    for name, values in start.items():
        getattr(fields, name)[...] = values
    kernel = _weld_launch(family, launcher, fields, grid, pml, record)
    assert kernel.launches == 1
    got = {name: numpy.array(getattr(fields, name), copy=True) for name in WELD_STATE}
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (case, plant,
                                                {k: v for k, v in diff.items() if v})
    assert all(int(numpy.count_nonzero(bits(want[c]) != bits(start[c]))) > 0
               for c in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz"))


def test_each_weld_case_reaches_what_it_is_built_to_reach(launcher, monkeypatch):
    """The far fill runs on the folded periodic cases, a generic phase and the Brillouin
    edge are bound, and every case's record keeps the off-diagonal row in some chunk."""
    from . import complex_offdiag_stencil_weld as weld  # noqa: PLC0415
    from . import folded_complex_offdiag_fused_electric_pair as family  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    reflect = {}
    kinds = {}
    for case in WELD_CASES:
        _fields, grid, _pml = build_weld(case, False)
        reflect[case] = weld.fill_plan(grid)["reflect"]
        kinds[case] = family.folded_complex_offdiag_fused_electric_pair_phases(grid)
    assert reflect["wf_y_periodic_kz"][1] >= 0 and reflect["wf_x_periodic_corpus"][0] >= 0
    assert all(row < 0 for row in reflect["wf_y_walled"])
    assert kinds["wf_y_periodic_kz"]["pair_kinds"] == (0, 0, complex_emitter.CONNECT_PHASE)
    assert kinds["wf_xz_edge_y"]["pair_kinds"] == (0, complex_emitter.CONNECT_NEGATE, 0)
    assert kinds["wf_y_periodic_kz"]["pair"] != kinds["wf_y_periodic_kz"]["own"]


#: defect -> (case, old text, new text): each a rewrite of the emitted weld text.
WELD_DEFECTS: Dict[str, Tuple[str, str, str]] = {
    "pair_window_unconnected": (
        "wf_y_periodic_kz",
        "    return cconnect(resolve_D_at(comp, index, weld), pa, pph, pphase);",
        "    return resolve_D_at(comp, index, weld);"),
    "home_sample_unconnected": (
        "wf_y_periodic_kz",
        "    cf near_pair = cf_add(cconnect(g_home, pa, pph, pphase),",
        "    cf near_pair = cf_add(g_home,"),
    "split_field_scratch_off_the_recurrence": (
        "wf_y_walled",
        "    *fu_out = dsigu ? fu_new : cf_load(fu, idx);",
        "    *fu_out = fu_new;"),
    "near_parity_dropped": (
        "wf_y_walled",
        "        if (image_y) value = mul_coefficient_left(weld.near_phase_y, value);",
        "        if (image_y) value = value;"),
    "far_parity_dropped": (
        "wf_y_periodic_kz",
        "    if (far) value = mul_coefficient_left(weld.far_phase_y, value);",
        "    if (far) value = value;"),
    # The ghost weight is -phase: -1 on the even fold, where dropping it moves a word.
    "ghost_lane_weight_dropped": (
        "wf_y_periodic_kz",
        "    cf z = resolve_D_at(comp, index, weld);\n"
        "    if (gl && bc == BC_MIRROR) return mul_coefficient_left(w, z);\n",
        "    cf z = resolve_D_at(comp, index, weld);\n"),
}


@pytest.mark.parametrize("defect", sorted(WELD_DEFECTS))
def test_each_planted_weld_defect_changes_a_word(defect, launcher, monkeypatch):
    from . import folded_complex_offdiag_fused_electric_pair as family  # noqa: PLC0415
    from .coverage import offdiag_row_mask  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    bind_launcher(family, launcher, monkeypatch)
    case, old, new = WELD_DEFECTS[defect]
    fields, grid, pml = build_weld(case, True)
    start = {name: numpy.array(getattr(fields, name), copy=True) for name in WELD_STATE}
    fused_oracle(WELD_PASSES, fields, pml)
    want = {name: numpy.array(getattr(fields, name), copy=True) for name in WELD_STATE}
    record = device_record(fields)
    for name, values in start.items():
        getattr(fields, name)[...] = values
    text = family.kernel_source(family._offdiag.normalized_row_mask(
        offdiag_row_mask(fields)))
    assert text.count(old) >= 1, defect
    _weld_launch(family, launcher, fields, grid, pml, record,
                 kernel_text=text.replace(old, new))
    got = {name: numpy.array(getattr(fields, name), copy=True) for name in WELD_STATE}
    assert sum(differing(want, got).values()) > 0, defect
