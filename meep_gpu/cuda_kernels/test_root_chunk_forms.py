"""The CUDA root kernels transcribe the array path's per-chunk PML forms, read off their text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The four root kernels of this table --
``step_B_pml_real`` / ``step_D_pml_real`` (``step_curl_kernels``), ``update_H_pml_real`` /
``update_E_pml_real`` (``constitutive_kernels``) -- and the nonlinear ``update_E``
(``nonlinear_constitutive``) choose MEEP's PML form per structure chunk from the run's
chunk record ``cb`` (bit_parity_0_10 DESIGN.md D1, D6, D10, D11; ``stepping._apply_curl``
and ``_apply_constitutive_regions``). Three things are checked:

1. **The record decode is the emitter's.** Each kernel carries, directly after its index
   decode, exactly the block ``bounds_block.bounds_block`` emits for the entries the
   region plan says it reads, and the emitted flag of every row equals
   ``chunk_map.membership_from_record`` on every record built here.
2. **The text computes the array path, word for word.** Each kernel's device text is
   EXECUTED in float32 NumPy by the small interpreter below -- its helpers too, from
   their own text -- over a seeded full-volume state, and every array of the ``Fields``
   is compared word for word (sign of zero included) with the series array path's own
   sub-step on a copy of the same state: the field, its auxiliary on and off the
   membership run, and every other array (which must not move). The cases cover the
   one-rank default in 3-D, a 2-D grid (an invariant axis: the partner-presence forms
   of the stencil), a periodic axis carrying a layer (the wrapped cell merged into the
   low run), a lifted division whose record differs from the one-rank one (the only
   case that tells "reads the record" from "matches the default"), and planted ``-0``
   words on plain cells. Every case asserts a nonzero cell count for each (component,
   form) class and that the sub-step moved its field (a vacuity floor).
3. **The register-view singles keep the auxiliary contract.** The hoisted
   ``update_H``/``update_E`` load the auxiliary and the old field only under the flag and
   store the auxiliary only under it (``own_cell_hoist``), so an auxiliary word off the
   run is never written: a NaN sentinel planted there survives.

THE INTERPRETER is a targeted one over the statements these texts write: declarations,
assignments, stores at ``idx``, ``if``/``else``, the conditional operator, early
``return`` in a helper, calls of the texts' own ``__device__`` helpers (pointer arguments
by name or by ``&register``), and the ``#define`` constants. Anything else raises, so the
grammar cannot grow a construct nothing executes. Arithmetic is float32, one rounding per
operation, which is the kernels' ``--fmad=false`` contract, except where C's own rules make
it double: a ``double`` declaration widens its float value (exactly), an unsuffixed literal
is a double, and ``__double2float_rn`` rounds a double once to the nearest float. The
nonlinear ``|D|^2`` is the one text that takes that path (MEEP's ``0.0625`` is a double
literal).
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import pathlib
import re
import sys
import types
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy
import pytest

from .. import stepping
from ..chunk_map import (ChunkLayout, chunk_bounds_record, grid_frames,
                         membership_from_record)
from . import bounds_block, own_cell_hoist
from .coverage import real_curl_boundary_codes

if TYPE_CHECKING:  # The driver is imported where a case is built (``_driver``): the ADE gate
    # imports this file for its ``Interpreter`` alone, and its import record must not
    # carry the driver and the dispatch planner it imports.
    from ..driver import FdtdDriver

HERE = pathlib.Path(__file__).parent
F32 = numpy.float32


# =============================================================================
# The modules, loaded against NumPy wearing CuPy's name, and removed afterwards
# =============================================================================

class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in."""

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture(scope="module")
def roots() -> Dict[str, Any]:
    """The three root modules, loaded by path under private names; ``cupy`` stubbed
    only for the load."""
    previous = sys.modules.get("cupy")
    sys.modules["cupy"] = _NumpyWearingCupysName()
    loaded = {}
    try:
        for stem in ("step_curl_kernels", "constitutive_kernels", "nonlinear_constitutive"):
            spec = importlib.util.spec_from_file_location(
                f"_root_chunk_forms_{stem}", HERE / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[stem] = module
    finally:
        if previous is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous
    return loaded


def _texts(roots) -> Dict[str, str]:
    return {
        "step_B_pml_real": roots["step_curl_kernels"]._step_B_pml_real_kernel_code,
        "step_D_pml_real": roots["step_curl_kernels"]._step_D_pml_real_kernel_code,
        "update_H_pml_real": roots["constitutive_kernels"]._update_H_pml_real_kernel_code,
        "update_E_pml_real": roots["constitutive_kernels"]._update_E_pml_real_kernel_code,
        "update_E_pml_real_nonlinear":
            roots["nonlinear_constitutive"]._update_E_pml_real_nonlinear_kernel_code,
    }


#: Each root kernel's record entries and presence axes, by the region plan's rule.
ENTRIES = {
    "step_B_pml_real": (bounds_block.curl_entries(("Bx", "By", "Bz")), (0, 1, 2)),
    "step_D_pml_real": (bounds_block.curl_entries(("Dx", "Dy", "Dz")), (0, 1, 2)),
    "update_H_pml_real": (bounds_block.constitutive_entries(("Hx", "Hy", "Hz")), ()),
    "update_E_pml_real": (bounds_block.constitutive_entries(("Ex", "Ey", "Ez")), ()),
    "update_E_pml_real_nonlinear":
        (bounds_block.constitutive_entries(("Ex", "Ey", "Ez")), ()),
}


# =============================================================================
# The interpreter
# =============================================================================

class Unmodelled(AssertionError):
    """A construct the interpreter does not execute: a failure, never a skip."""


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _match_close(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for position in range(start, len(text)):
        if text[position] == opening:
            depth += 1
        elif text[position] == closing:
            depth -= 1
            if depth == 0:
                return position
    raise Unmodelled("unbalanced brackets")


def _split_top(text: str, separator: str = ",") -> List[str]:
    out, depth, current = [], 0, []
    for character in text:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        if character == separator and depth == 0:
            out.append("".join(current))
            current = []
        else:
            current.append(character)
    out.append("".join(current))
    return [part.strip() for part in out if part.strip()]


def _items(body: str) -> List[str]:
    """Statements at depth 0: ``...;``, ``head {`` and ``}``, whitespace collapsed."""
    out, current, depth = [], [], 0
    for character in body:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        if depth == 0 and character == ";":
            statement = " ".join("".join(current).split())
            if statement:
                out.append(statement + ";")
            current = []
        elif depth == 0 and character == "{":
            out.append(" ".join("".join(current).split()) + " {")
            current = []
        elif depth == 0 and character == "}":
            statement = " ".join("".join(current).split())
            if statement:
                out.append(statement)
            out.append("}")
            current = []
        else:
            current.append(character)
    if " ".join("".join(current).split()):
        raise Unmodelled(f"trailing text {''.join(current)!r}")
    return out


def _functions(text: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for match in re.finditer(r'(?:extern\s+"C"\s+)?(__device__|__global__)\b', text):
        paren = text.index("(", match.end())
        head = text[match.end():paren].split()
        close = _match_close(text, paren, "(", ")")
        body_open = text.index("{", close)
        body_close = _match_close(text, body_open, "{", "}")
        params = []
        for declaration in _split_top(text[paren + 1:close]):
            name = re.findall(r"[A-Za-z_]\w*", declaration)[-1]
            params.append((name, "*" in declaration))
        out[head[-1]] = {"kind": match.group(1), "params": params,
                         "body": text[body_open + 1:body_close]}
    return out


def _find_ternary(expression: str) -> Optional[Tuple[str, str, str]]:
    depth, question = 0, None
    for position, character in enumerate(expression):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == "?" and depth == 0 and question is None:
            question = position
        elif character == ":" and depth == 0 and question is not None:
            return (expression[:question].strip(), expression[question + 1:position].strip(),
                    expression[position + 1:].strip())
    return None


def _python(expression: str) -> str:
    """One C expression of these texts as a Python expression over the same names.

    The conditional operator becomes Python's, at every nesting depth: a top-level one
    splits the expression, and every bracketed group is converted on its own.
    """
    ternary = _find_ternary(expression)
    if ternary is not None:
        test, body, orelse = ternary
        return f"(({_python(body)}) if ({_python(test)}) else ({_python(orelse)}))"
    pieces, position = [], 0
    while position < len(expression):
        character = expression[position]
        if character in "([":
            closing = ")" if character == "(" else "]"
            end = _match_close(expression, position, character, closing)
            pieces.append(character + _python(expression[position + 1:end]) + closing)
            position = end + 1
            continue
        pieces.append(character)
        position += 1
    out = "".join(pieces)
    out = re.sub(r"(\d+\.\d*|\.\d+)f\b", r"\1", out)
    out = out.replace("&&", " and ").replace("||", " or ")
    out = re.sub(r"!(?!=)", " not ", out)
    out = re.sub(r"(^|[(,]\s*)&(\w+)", r"\1_addr_\2", out)
    if "&" in out.replace("& 1", ""):
        raise Unmodelled(f"an address or bitwise and the interpreter does not read: {expression!r}")
    return out.strip()


#: An unsuffixed C float literal, which C types double (``0.0625``, never ``0.0625f``).
_DOUBLE_LITERAL = re.compile(r"(?<![\w.])(\d+\.\d*|\.\d+)(?![\w.])")


def _c_expression(expression: str) -> str:
    """:func:`_python` of a C expression, its unsuffixed float literals marked double."""
    return _python(_DOUBLE_LITERAL.sub(r"__double_literal(\1)", expression))


class _Pointer:
    """A kernel pointer argument: a flat array, read by gather and written at ``idx``."""

    def __init__(self, array: numpy.ndarray):
        self.array = array


class _Register:
    """``&name``: a register of the calling frame, read and written whole (index 0)."""

    def __init__(self, frame: Dict[str, Any], name: str):
        self.frame = frame
        self.name = name


def _truth(value: Any) -> Any:
    return numpy.asarray(value) != 0


class Interpreter:
    """Execute one text's ``__global__`` entry over every lane at once."""

    def __init__(self, text: str):
        stripped = _strip_comments(text)
        self.defines = {name: int(value) for name, value in
                        re.findall(r"(?m)^\s*#define (\w+) (\d+)\s*$", stripped)}
        body_text = "\n".join(line for line in stripped.splitlines()
                              if not line.strip().startswith("#"))
        self.functions = _functions(body_text)
        entries = [n for n, f in self.functions.items() if f["kind"] == "__global__"]
        if len(entries) != 1:
            raise Unmodelled(f"{len(entries)} entry kernels, not one")
        self.entry = entries[0]

    # -- expressions -------------------------------------------------------------------
    def value(self, text: str, frame: Dict[str, Any]) -> Any:
        return self._eval(ast.parse(_c_expression(text), mode="eval").body, frame)

    @staticmethod
    def _owner(name: str, frame: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The frame that holds ``name``: this one or a scope it is nested in."""
        while frame is not None:
            if name in frame:
                return frame
            frame = frame.get("__parent__")
        return None

    def _lookup(self, name: str, frame: Dict[str, Any]) -> Any:
        if name.startswith("_addr_"):
            target = name[len("_addr_"):]
            owner = self._owner(target, frame)
            if owner is None:
                raise Unmodelled(f"&{target} names no local")
            return _Register(owner, target)
        owner = self._owner(name, frame)
        if owner is not None:
            return owner[name]
        if name in self.defines:
            return self.defines[name]
        if name in ("true", "false"):
            return name == "true"
        raise Unmodelled(f"unknown name {name!r}")

    def _eval(self, node: ast.AST, frame: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return F32(node.value)
            if isinstance(node.value, int):
                return node.value
            raise Unmodelled(f"constant {node.value!r}")
        if isinstance(node, ast.Name):
            return self._lookup(node.id, frame)
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name):
                raise Unmodelled("subscript of an expression")
            target = self._lookup(node.value.id, frame)
            index = self._eval(node.slice, frame)
            if isinstance(target, _Register):
                return target.frame[target.name]
            if isinstance(target, _Pointer):
                flat = target.array
                where = numpy.clip(numpy.asarray(index), 0, flat.size - 1)
                return flat[where]
            raise Unmodelled(f"subscript of a non-pointer {node.value.id}")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, frame)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.Not):
                return ~_truth(operand)
            raise Unmodelled(f"unary {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, frame)
            right = self._eval(node.right, frame)
            op = node.op
            integer = all(numpy.asarray(v).dtype.kind in "iub" for v in (left, right))
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                if integer:
                    return left // right
                raise Unmodelled("a float division outside __fdiv_rn")
            if isinstance(op, ast.Mod) and integer:
                return left % right
            if isinstance(op, ast.RShift) and integer:
                return left >> right
            if isinstance(op, ast.BitAnd) and integer:
                return left & right
            raise Unmodelled(f"binary {type(op).__name__}")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise Unmodelled("chained comparison")
            left = self._eval(node.left, frame)
            right = self._eval(node.comparators[0], frame)
            op = node.ops[0]
            table = {ast.Lt: numpy.less, ast.LtE: numpy.less_equal, ast.Gt: numpy.greater,
                     ast.GtE: numpy.greater_equal, ast.Eq: numpy.equal,
                     ast.NotEq: numpy.not_equal}
            return table[type(op)](left, right)
        if isinstance(node, ast.BoolOp):
            values = [_truth(self._eval(v, frame)) for v in node.values]
            combine = numpy.logical_and if isinstance(node.op, ast.And) else numpy.logical_or
            out = values[0]
            for value in values[1:]:
                out = combine(out, value)
            return out
        if isinstance(node, ast.IfExp):
            test = _truth(self._eval(node.test, frame))
            body = self._eval(node.body, frame)
            orelse = self._eval(node.orelse, frame)
            out = numpy.where(test, body, orelse)
            kinds = {numpy.asarray(v).dtype.kind for v in (body, orelse)}
            if kinds == {"f"}:
                assert out.dtype == F32, out.dtype
            return out
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "__fdiv_rn":
                left, right = (self._eval(a, frame) for a in node.args)
                out = numpy.divide(left, right)
                assert numpy.asarray(out).dtype == F32
                return out
            if name == "__double_literal":
                (literal,) = node.args
                if not (isinstance(literal, ast.Constant) and isinstance(literal.value, float)):
                    raise Unmodelled("a marked double literal that is not a float constant")
                return numpy.float64(literal.value)
            if name == "__double2float_rn":
                (argument,) = node.args
                wide = numpy.asarray(self._eval(argument, frame))
                if wide.dtype != numpy.float64:
                    raise Unmodelled(f"__double2float_rn of a {wide.dtype} value")
                return wide.astype(F32)  # IEEE round to nearest even, once
            if name in self.functions:
                return self.call(name, node.args, frame)
            raise Unmodelled(f"call of {name}")
        raise Unmodelled(f"expression node {type(node).__name__}")

    # -- calls -------------------------------------------------------------------------
    def call(self, name: str, args: List[ast.AST], caller: Dict[str, Any]) -> Any:
        function = self.functions[name]
        if len(args) != len(function["params"]):
            raise Unmodelled(f"{name}: {len(args)} arguments for {len(function['params'])}")
        frame: Dict[str, Any] = {"__pred__": caller["__pred__"]}
        for (param, pointer), arg in zip(function["params"], args):
            value = self._eval(arg, caller)
            if pointer != isinstance(value, (_Pointer, _Register)):
                raise Unmodelled(f"{name}: {param} bound to {type(value).__name__}")
            frame[param] = value
        frame["__ret__"] = None
        frame["__done__"] = numpy.zeros((), dtype=bool)
        self.run(_items(function["body"]), frame)
        return frame["__ret__"]

    # -- statements --------------------------------------------------------------------
    def _active(self, frame: Dict[str, Any]) -> Any:
        done = self._lookup("__done__", frame) if self._owner("__done__", frame) else None
        if done is None:
            return frame["__pred__"]
        return numpy.logical_and(frame["__pred__"], numpy.logical_not(done))

    def _set(self, frame: Dict[str, Any], name: str, value: Any) -> None:
        owner = self._owner(name, frame)
        if owner is None:
            raise Unmodelled(f"assignment to undeclared {name}")
        owner[name] = numpy.where(self._active(frame), value, owner[name])

    def _store(self, frame: Dict[str, Any], target: str, index: str, value: Any) -> None:
        pointer = self._lookup(target, frame)
        if index != "idx":
            raise Unmodelled(f"store at {target}[{index}]")
        active = self._active(frame)
        if isinstance(pointer, _Register):
            pointer.frame[pointer.name] = numpy.where(active, value,
                                                      pointer.frame[pointer.name])
            return
        if not isinstance(pointer, _Pointer):
            raise Unmodelled(f"store to a non-pointer {target}")
        stored = numpy.asarray(value)
        assert stored.dtype == F32, (target, stored.dtype)
        pointer.array[...] = numpy.where(active, stored, pointer.array)

    def run(self, statements: List[str], frame: Dict[str, Any]) -> None:
        position = 0
        while position < len(statements):
            statement = statements[position]
            if statement.endswith("{"):
                end, depth = position, 0
                for end in range(position, len(statements)):
                    if statements[end].endswith("{"):
                        depth += 1
                    elif statements[end] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                head = statement[:-1].strip()
                inner = statements[position + 1:end]
                orelse: List[str] = []
                after = end + 1
                if after < len(statements) and statements[after] == "else {":
                    depth, stop = 0, after
                    for stop in range(after, len(statements)):
                        if statements[stop].endswith("{"):
                            depth += 1
                        elif statements[stop] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                    orelse = statements[after + 1:stop]
                    after = stop + 1
                if head == "":
                    self.run(inner, {"__parent__": frame, "__pred__": frame["__pred__"]})
                else:
                    match = re.fullmatch(r"if \((.+)\)", head)
                    if match is None:
                        raise Unmodelled(f"block head {head!r}")
                    test = _truth(self.value(match.group(1), frame))
                    self._predicated(inner, frame, test)
                    if orelse:
                        self._predicated(orelse, frame, ~test)
                position = after
                continue
            self.statement(statement, frame)
            position += 1

    def _predicated(self, statements: List[str], frame: Dict[str, Any], test: Any) -> None:
        held = frame["__pred__"]
        frame["__pred__"] = numpy.logical_and(held, test)
        try:
            self.run(statements, frame)
        finally:
            frame["__pred__"] = held

    def statement(self, statement: str, frame: Dict[str, Any]) -> None:
        if statement.startswith("if ("):
            close = _match_close(statement, 3, "(", ")")
            condition = statement[4:close]
            rest = statement[close + 1:].strip()
            if rest == "return;":
                if condition != "idx >= nx * ny * nz":
                    raise Unmodelled(f"early return in the kernel on {condition!r}")
                return  # every lane of the grid is in range by construction
            test = _truth(self.value(condition, frame))
            match = re.fullmatch(r"return (.+);", rest)
            if match:
                self._return(frame, match.group(1), test)
                return
            held = frame["__pred__"]
            frame["__pred__"] = numpy.logical_and(held, test)
            try:
                self.statement(rest, frame)
            finally:
                frame["__pred__"] = held
            return
        match = re.fullmatch(r"return (.+);", statement)
        if match:
            self._return(frame, match.group(1), numpy.ones((), dtype=bool))
            return
        match = re.fullmatch(r"(?:const )?(int|float|double|bool) (.+);", statement)
        if match:
            for part in _split_top(match.group(2)):
                declared = re.fullmatch(r"(\w+) = (.+)", part)
                if declared is None:
                    raise Unmodelled(f"declaration {part!r}")
                value = self.value(declared.group(2), frame)
                if match.group(1) == "float":
                    value = numpy.asarray(value)
                    if value.dtype != F32:
                        raise Unmodelled(f"float {declared.group(1)} is {value.dtype}")
                elif match.group(1) == "double":
                    value = numpy.asarray(value)
                    if value.dtype not in (F32, numpy.float64):
                        raise Unmodelled(f"double {declared.group(1)} from {value.dtype}")
                    value = value.astype(numpy.float64)  # C's widening, exact
                frame[declared.group(1)] = value
            return
        match = re.fullmatch(r"(\w+)\[(.+)\] = (.+);", statement)
        if match:
            self._store(frame, match.group(1), match.group(2),
                        self.value(match.group(3), frame))
            return
        match = re.fullmatch(r"(\w+) = (.+);", statement)
        if match:
            self._set(frame, match.group(1), self.value(match.group(2), frame))
            return
        match = re.fullmatch(r"(\w+)\((.*)\);", statement)
        if match and match.group(1) in self.functions:
            node = ast.parse(_c_expression(statement[:-1]), mode="eval").body
            self.call(match.group(1), node.args, frame)
            return
        raise Unmodelled(f"statement {statement!r}")

    def _return(self, frame: Dict[str, Any], text: str, test: Any) -> None:
        if "__done__" not in frame:
            raise Unmodelled("a value return in the kernel, or in a nested scope of a helper")
        value = self.value(text, frame)
        take = numpy.logical_and(test, numpy.logical_not(frame["__done__"]))
        current = frame["__ret__"]
        if current is None:
            current = F32(numpy.nan)  # a lane no return reaches reads as NaN
        frame["__ret__"] = numpy.where(take, value, current)
        frame["__done__"] = numpy.logical_or(frame["__done__"], test)

    def launch(self, arguments: Dict[str, Any], lanes: int) -> None:
        function = self.functions[self.entry]
        frame: Dict[str, Any] = {"__pred__": numpy.ones(lanes, dtype=bool)}
        names = [param for param, _pointer in function["params"]]
        if sorted(names) != sorted(arguments):
            raise Unmodelled(f"arguments {sorted(arguments)} for parameters {sorted(names)}")
        for param, pointer in function["params"]:
            value = arguments[param]
            frame[param] = _Pointer(value) if pointer else value
        statements = _items(function["body"])
        first = statements[0]
        if first != "int idx = blockIdx.x * blockDim.x + threadIdx.x;":
            raise Unmodelled(f"the kernel does not open with the flat index: {first!r}")
        frame["idx"] = numpy.arange(lanes, dtype=numpy.int64)
        self.run(statements[1:], frame)


# =============================================================================
# The cases
# =============================================================================

def _split_y_layout(driver: FdtdDriver) -> ChunkLayout:
    """The one-rank boxes with the low-y effort volume cut one cell short of its edge.

    The sliver carries no y layer (it lies past the sigma extent, so every coefficient
    there is exactly 1 and the precondition holds), so its rows leave the membership
    run: the record differs from the one-rank record on the y line of both lattices.
    """
    one_rank = ChunkLayout.one_rank(driver.grid, driver.pml)
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
    return ChunkLayout(boxes, flags, grid_frames(driver.grid),
                       faces=driver.pml.thickness_by_face, source="meep")


#: name -> (cell, resolution, dimensions, boundaries, pml cells, divide)
CASES = {
    "box3d": ((3.0, 3.5, 4.0), 8.0, 3, {"x": "metallic", "y": "metallic", "z": "metallic"},
              {"x": 6, "y": 6, "z": 6}, None),
    "slab2d": ((6.0, 5.0, 0.0), 10.0, 2, {"x": "metallic", "y": "metallic"},
               {"x": 8, "y": 8}, None),
    "periodic3d": ((3.0, 3.0, 3.5), 8.0, 3, {"x": "periodic", "y": "metallic", "z": "metallic"},
                   {"x": 6, "y": 6, "z": 6}, None),
    "split_y2d": ((6.0, 5.0, 0.0), 10.0, 2, {"x": "metallic", "y": "metallic"},
                  {"x": 8, "y": 8}, _split_y_layout),
}


def _driver(case: str) -> FdtdDriver:
    from ..driver import FdtdDriver  # noqa: PLC0415 - see the TYPE_CHECKING import

    cell, resolution, dimensions, boundaries, layer, divide = CASES[case]
    driver = FdtdDriver(cell_size=cell, resolution=resolution, dimensions=dimensions,
                        boundaries=boundaries, force_complex_fields=False, prefer_gpu=False)
    driver.setup_pml(layer)
    if divide is not None:
        driver.set_chunk_layout(divide(driver))
    rng = numpy.random.default_rng(20261007)
    shape = tuple(int(n) for n in driver.grid.shape)
    driver.set_epsilon((1.0 + 11.0 * rng.uniform(size=shape)).astype(F32))
    return driver


_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
          "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
          "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _seed(driver: FdtdDriver, plant_negative_zero: bool) -> None:
    """Every stepped array nonzero everywhere, so every chunk-boundary plane takes its
    form at the first launch; optionally ``-0`` planted on a quarter of the cells."""
    rng = numpy.random.default_rng(7)
    fields = driver.fields
    for name in _STATE:
        array = getattr(fields, name)
        values = rng.uniform(-1.0, 1.0, size=array.shape).astype(F32)
        if plant_negative_zero:
            values[rng.uniform(size=array.shape) < 0.25] = F32(-0.0)
        array[...] = values
    fields.h_separated = True  # a seeded state is a mid-run one


def _bits(array: numpy.ndarray) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def _snapshot(fields) -> Dict[str, numpy.ndarray]:
    return {name: numpy.array(getattr(fields, name), copy=True) for name in _STATE}


def _restore(fields, state: Dict[str, numpy.ndarray]) -> None:
    for name, values in state.items():
        getattr(fields, name)[...] = values


def _flat(fields, name: str) -> numpy.ndarray:
    array = getattr(fields, name)
    assert array.flags.c_contiguous and array.dtype == F32
    return array.reshape(-1)


def _tables(pml, kinds: Tuple[str, str], half_integer: bool) -> Dict[str, numpy.ndarray]:
    suffix = "_h" if half_integer else ""
    return {f"{kind}_{axis}": numpy.ascontiguousarray(
                getattr(pml, f"{kind}_{axis}{suffix}").reshape(-1))
            for axis in "xyz" for kind in kinds}


def _arguments(kernel: str, driver: FdtdDriver, record: numpy.ndarray) -> Dict[str, Any]:
    fields, grid, pml = driver.fields, driver.grid, driver.pml
    nx, ny, nz = (int(n) for n in grid.shape)
    out: Dict[str, Any] = {"nx": nx, "ny": ny, "nz": nz, "cb": record}
    if kernel in ("step_B_pml_real", "step_D_pml_real"):
        side, source = ("B", "E") if kernel == "step_B_pml_real" else ("D", "H")
        for axis in "xyz":
            out[f"{side}{axis}"] = _flat(fields, f"{side}{axis}")
            out[f"fu_{side}{axis}"] = _flat(fields, f"fu_{side}{axis}")
            out[f"{source}{axis}"] = _flat(fields, f"{source}{axis}")
        out["dtdx"] = F32(grid.dt / grid.dx)
        out.update(_tables(pml, ("kms", "sinv"), kernel == "step_B_pml_real"))
        codes, refusal = real_curl_boundary_codes(grid)
        assert refusal is None, refusal
        out.update({"bc_x": codes[0], "bc_y": codes[1], "bc_z": codes[2]})
        return out
    side, source = ("H", "B") if kernel == "update_H_pml_real" else ("E", "D")
    for axis in "xyz":
        out[f"{side}{axis}"] = _flat(fields, f"{side}{axis}")
        out[f"f_w_{side}{axis}"] = _flat(fields, f"f_w_{side}{axis}")
        out[f"{source}{axis}"] = _flat(fields, f"{source}{axis}")
    out.update(_tables(pml, ("kps", "kms"), side == "E"))
    if side == "E":
        for axis in "xyz":
            out[f"inv_eps_E{axis}"] = numpy.ascontiguousarray(
                fields.inverse_epsilon_for(f"E{axis}")).reshape(-1)
    return out


#: The array path's own sub-step for each kernel.
ARRAY_PATH = {
    "step_B_pml_real": stepping.step_B,
    "step_D_pml_real": stepping.step_D,
    "update_H_pml_real": stepping.update_H,
    "update_E_pml_real": stepping.update_E,
}

#: The components each kernel steps, and the region-plan sub-step that names their form.
STEPPED = {
    "step_B_pml_real": (("Bx", "By", "Bz"), "curl", "fu_"),
    "step_D_pml_real": (("Dx", "Dy", "Dz"), "curl", "fu_"),
    "update_H_pml_real": (("Hx", "Hy", "Hz"), "constitutive", "f_w_"),
    "update_E_pml_real": (("Ex", "Ey", "Ez"), "constitutive", "f_w_"),
}


def _split_axis_invariant(kernel: str, component: str, case: str) -> bool:
    """Is the axis that selects this component's form invariant on this case's grid?"""
    _components, sub_step, _aux = STEPPED[kernel]
    if sub_step == "curl":
        term = next(t for t in (*stepping.B_CURL_TERMS, *stepping.D_CURL_TERMS)
                    if t.target == component)
        axis = "xyz".index(term.dsigu)
    else:
        axis = "xyz".index(component[1])
    return CASES[case][2] == 2 and axis == 2


def _form_counts(driver: FdtdDriver, kernel: str) -> Dict[str, Tuple[int, int]]:
    components, sub_step, _aux = STEPPED[kernel]
    plan = driver.fields.region_plan
    shape = tuple(int(n) for n in driver.grid.shape)
    out = {}
    for component in components:
        inside = numpy.broadcast_to(plan.mask(sub_step, component, numpy), shape)
        out[component] = (int(inside.sum()), int((~inside).sum()))
    return out


# =============================================================================
# 1. The record decode
# =============================================================================

@pytest.mark.parametrize("kernel", sorted(ENTRIES))
def test_each_root_kernel_decodes_the_record_with_the_emitters_block(kernel, roots):
    text = _texts(roots)[kernel]
    entries, presence = ENTRIES[kernel]
    marker = f'extern "C" __global__ void {kernel}('
    assert text.count(marker) == 1
    tail = text.split(marker, 1)[1]
    assert tail.count(bounds_block.DECODE_END) == 1
    after = tail.split(bounds_block.DECODE_END, 1)[1]
    assert after.startswith("\n" + bounds_block.bounds_block(entries, presence)), kernel
    assert "const int* __restrict__ cb\n) {" in tail, kernel
    # The block is the only reader of the record.
    assert len(re.findall(r"\bcb\[", tail)) == 4 * len(set(entries)) + (1 if presence else 0)


def test_the_insertion_door_writes_the_block_the_literal_carries(roots):
    """``bounds_block.insert_bounds_block`` on the text with its block removed gives the text."""
    for kernel, text in _texts(roots).items():
        entries, presence = ENTRIES[kernel]
        block = bounds_block.bounds_block(entries, presence)
        stripped = text.replace(bounds_block.DECODE_END + "\n" + block,
                                bounds_block.DECODE_END + "\n", 1)
        assert stripped != text
        assert bounds_block.insert_bounds_block(stripped, kernel, entries, presence) == text
        with pytest.raises(ValueError, match="already reads the chunk record"):
            bounds_block.insert_bounds_block(text, kernel, entries, presence)


def _records() -> List[numpy.ndarray]:
    out = []
    for case in CASES:
        driver = _driver(case)
        try:
            out.append(numpy.asarray(driver.chunk_bounds_record()))
        finally:
            driver.close()
    rng = numpy.random.default_rng(3)
    for _ in range(20):
        record = numpy.zeros(25, dtype=numpy.int32)
        for half, axis in itertools.product((False, True), range(3)):
            cuts = numpy.sort(rng.integers(0, 40, size=4))
            for word, value in enumerate(cuts):
                record[(int(half) * 3 + axis) * 4 + word] = value
        record[24] = int(rng.integers(0, 8))
        out.append(record)
    return out


def test_the_emitted_flags_are_the_record_membership():
    """Every flag line of the emitter, evaluated per row, against the host decoder."""
    records = _records()
    entries = list(itertools.product((False, True), range(3)))
    block = bounds_block.bounds_block(entries, presence=(0, 1, 2))
    statements = [s for s in _items(_strip_comments(block))]
    rows = numpy.arange(48)
    for record in records:
        frame: Dict[str, Any] = {"__pred__": numpy.ones(rows.size, dtype=bool),
                                 "cb": _Pointer(record), "i": rows, "j": rows, "k": rows}
        interpreter = Interpreter('extern "C" __global__ void probe(int idx) {\n}\n')
        interpreter.run(statements, frame)
        for half, axis in entries:
            expected = membership_from_record(record, half, axis, rows.size)
            got = numpy.asarray(frame[bounds_block.flag_name(half, axis)], dtype=bool)
            assert numpy.array_equal(got, expected), (record.tolist(), half, axis)
        for axis in range(3):
            assert int(frame[bounds_block.presence_name(axis)]) == (int(record[24]) >> axis) & 1


def test_the_record_offsets_are_the_record_builders():
    from ..chunk_map import RECORD_PRESENCE_WORD, record_index
    block = bounds_block.bounds_block(itertools.product((False, True), range(3)), (0, 1, 2))
    for half, axis in itertools.product((False, True), range(3)):
        for slot, word in enumerate(bounds_block.WORDS):
            name = bounds_block.bounds_name(half, axis, word)
            assert f"{name} = cb[{record_index(half, axis, slot)}]" in block
    assert f"cb_presence = cb[{RECORD_PRESENCE_WORD}];" in block


@pytest.mark.parametrize("components,expected", [
    (("Bx", "By", "Bz"), [(True, 2), (True, 0), (True, 1)]),
    (("Dx", "Dy", "Dz"), [(False, 2), (False, 0), (False, 1)]),
])
def test_curl_entries_are_the_dsigu_axis_on_the_components_lattice(components, expected):
    assert bounds_block.curl_entries(components) == expected


def test_constitutive_entries_are_the_own_axis_on_the_components_lattice():
    assert bounds_block.constitutive_entries(("Hx", "Hy", "Hz")) == [
        (False, 0), (False, 1), (False, 2)]
    assert bounds_block.constitutive_entries(("Ex", "Ey", "Ez")) == [
        (True, 0), (True, 1), (True, 2)]


# =============================================================================
# 2. The text computes the array path
# =============================================================================

@pytest.mark.parametrize("plant", (False, True), ids=("random", "planted_negative_zero"))
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("kernel", sorted(ARRAY_PATH))
def test_the_root_kernel_text_is_the_array_path_word_for_word(kernel, case, plant, roots):
    driver = _driver(case)
    try:
        _seed(driver, plant)
        fields = driver.fields
        start = _snapshot(fields)
        ARRAY_PATH[kernel](fields, driver.pml)
        oracle = _snapshot(fields)
        _restore(fields, start)
        record = numpy.ascontiguousarray(driver.chunk_bounds_record(), dtype=numpy.int32)
        Interpreter(_texts(roots)[kernel]).launch(_arguments(kernel, driver, record),
                                                  int(numpy.prod(driver.grid.shape)))
        kernel_state = _snapshot(fields)
        counts = _form_counts(driver, kernel)
        components, sub_step, aux = STEPPED[kernel]
        masks = {component: numpy.broadcast_to(
                     numpy.asarray(fields.region_plan.mask(sub_step, component, numpy)),
                     oracle[component].shape)
                 for component in components}
    finally:
        driver.close()
    differing = {name: int(numpy.count_nonzero(_bits(oracle[name]) != _bits(kernel_state[name])))
                 for name in _STATE}
    assert all(v == 0 for v in differing.values()), (kernel, case, plant, differing)
    for component in components:
        inside, outside = counts[component]
        if _split_axis_invariant(kernel, component, case):
            # The axis whose layer selects the form is invariant: no chunk carries it,
            # so every cell takes the form off the run (MEEP installs no face there).
            assert inside == 0 and outside > 0, (kernel, case, component, counts)
        else:
            assert inside > 0 and outside > 0, (kernel, case, component, counts)
        moved = int(numpy.count_nonzero(_bits(oracle[component]) != _bits(start[component])))
        assert moved > 0, (kernel, case, component)
        # The auxiliary contract: no auxiliary word off the membership run moves.
        off = ~masks[component]
        assert numpy.array_equal(_bits(oracle[aux + component])[off],
                                 _bits(start[aux + component])[off]), (kernel, case, component)


def test_the_split_division_moves_the_record_off_the_one_rank_record():
    """The lifted case discriminates: its record is not the default's."""
    driver = _driver("split_y2d")
    try:
        lifted = numpy.asarray(driver.chunk_bounds_record())
        default = chunk_bounds_record(ChunkLayout.one_rank(driver.grid, driver.pml))
    finally:
        driver.close()
    assert not numpy.array_equal(lifted, default)


@pytest.mark.parametrize("kernel", ("step_B_pml_real", "update_E_pml_real"))
def test_a_kernel_handed_the_one_rank_record_on_the_split_division_is_caught(kernel, roots):
    """The equivalence above is not vacuous: the wrong record gives wrong words."""
    driver = _driver("split_y2d")
    try:
        _seed(driver, False)
        fields = driver.fields
        start = _snapshot(fields)
        ARRAY_PATH[kernel](fields, driver.pml)
        oracle = _snapshot(fields)
        _restore(fields, start)
        wrong = numpy.ascontiguousarray(
            chunk_bounds_record(ChunkLayout.one_rank(driver.grid, driver.pml)), dtype=numpy.int32)
        Interpreter(_texts(roots)[kernel]).launch(_arguments(kernel, driver, wrong),
                                                  int(numpy.prod(driver.grid.shape)))
        kernel_state = _snapshot(fields)
    finally:
        driver.close()
    assert any(numpy.count_nonzero(_bits(oracle[n]) != _bits(kernel_state[n])) for n in _STATE)


#: Defects the equivalence must see, each a rewrite of one root text: ``(kernel, find,
#: replace, case)``. Every one must change a word of the comparison above, on random or
#: planted-zero states. The last two are the planted-``-0`` must-catch rows of D13.
DEFECTS = {
    "recurrence_on_every_cell": ("step_B_pml_real", "int dsigu\n) {",
                                 "int dsigu_chunk\n) {\n    const int dsigu = 1;", "box3d"),
    "old_stencil_grouping": ("step_D_pml_real", "dtdx * (((sf - f1) + f2) - ss)",
                             "dtdx * ((sf - f1) + (f2 - ss))", "box3d"),
    "full_bracket_over_an_absent_partner": (
        "step_B_pml_real", "if (pr1) return dtdx * (sf - f1);",
        "if (pr1) return dtdx * (((sf - f1) + f2) - ss);", "slab2d"),
    "bound_off_by_one": ("step_B_pml_real", "(k >= cbh_z_a0 && k < cbh_z_a1)",
                         "(k >= cbh_z_a0 && k < cbh_z_a1 + 1)", "box3d"),
    "integer_record_on_the_B_curl": ("step_B_pml_real", "cbh_z_a0 = cb[20], cbh_z_a1 = cb[21]",
                                     "cbh_z_a0 = cb[8], cbh_z_a1 = cb[9]", "box3d"),
    "flag_on_the_dsig_axis": ("step_D_pml_real", "kms_z[k], sinv_z[k], in_i_z);",
                              "kms_z[k], sinv_z[k], in_i_y);", "box3d"),
    "old_constitutive_grouping": ("update_H_pml_real",
                                  "(f[idx] + ((kps * src) - (kms * prev)))",
                                  "((f[idx] + (kps * src)) - (kms * prev))", "box3d"),
    "accumulation_on_every_cell": ("update_E_pml_real", "int dsigw\n) {",
                                   "int dsigw_chunk\n) {\n    const int dsigw = 1;", "box3d"),
    "auxiliary_stored_on_a_plain_cell": ("update_H_pml_real", "    if (in_i_y) f_w_Hy[idx] = w_y;",
                                         "    f_w_Hy[idx] = w_y;", "box3d"),
    "lifted_record_ignored": ("update_E_pml_real", "(j >= cbh_y_a0 && j < cbh_y_a1)",
                              "(j >= cbh_y_a0 && j < cbh_y_a1 + 1)", "split_y2d"),
    "plain_curl_through_a_negated_curl": (
        "step_B_pml_real", ": (((fcur * kms) - curl) * sinv);",
        ": (((fcur * kms) + (0.0f - curl)) * sinv);", "box3d"),
    "plain_constitutive_as_unit_coefficients": (
        "update_E_pml_real", ": src;\n    f[idx] = value;",
        ": (0.0f + ((1.0f * src) - (0.0f * prev)));\n    f[idx] = value;", "box3d"),
}


#: The two rows a random state cannot catch: ``a - b`` written ``a + (0 - b)`` (the curl,
#: f = -0 against a +0 curl) and the plain constitutive written as the accumulation with
#: ``kps = 1, kms = 0`` and zeroed inputs (src = -0).
PLANTED_ONLY = ("plain_curl_through_a_negated_curl", "plain_constitutive_as_unit_coefficients")


def _run_text(text: str, kernel: str, case: str, plant: bool) -> Tuple[Dict, Dict]:
    driver = _driver(case)
    try:
        _seed(driver, plant)
        fields = driver.fields
        start = _snapshot(fields)
        ARRAY_PATH[kernel](fields, driver.pml)
        oracle = _snapshot(fields)
        _restore(fields, start)
        record = numpy.ascontiguousarray(driver.chunk_bounds_record(), dtype=numpy.int32)
        Interpreter(text).launch(_arguments(kernel, driver, record),
                                 int(numpy.prod(driver.grid.shape)))
        return oracle, _snapshot(fields)
    finally:
        driver.close()


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_planted_defect_changes_a_word(defect, roots):
    kernel, find, replace, case = DEFECTS[defect]
    text = _texts(roots)[kernel]
    assert text.count(find) >= 1, defect
    mutated = text.replace(find, replace)
    counts = {}
    for plant in (False, True):
        oracle, got = _run_text(mutated, kernel, case, plant)
        counts[plant] = sum(int(numpy.count_nonzero(_bits(oracle[n]) != _bits(got[n])))
                            for n in _STATE)
    if defect in PLANTED_ONLY:
        # Equal to the shipped form on every word but a signed zero: only the planted
        # -0 state can see them (KERNELS.md section 7.2), and the random state must not.
        assert counts[False] == 0 and counts[True] > 0, (defect, counts)
    else:
        assert any(counts.values()), (defect, counts)


#: The D magnitude of the planted ``|D|^2`` band: with every D volume drawn in
#: ``(-2^-63, 2^-63)``, ``gs*gs`` reaches ``2^-126`` while ``0.0625*(g1s^2 + g2s^2)``
#: often lies below float32's normal range, the one band where the double and the
#: float32 ``Dsqr`` differ (16-AP6h; phase-0 P0-h). ``chi3`` lifts such a ``Dsqr``
#: into ``c3``: ``Dsqr <= 3 * 2^-126`` and ``chi1inv^3 <= (1/1.2)^3`` bound
#: ``c3 <= 0.18`` at ``chi3 = 9e36``, inside the pole guard's ``1/3``.
DSQR_BAND_SCALE = F32(2.0 ** -63)
DSQR_BAND_CHI3 = 9.0e36


def _run_nonlinear(text: str, case: str, chi_kind: str, module) -> Tuple[Dict, Dict, Dict]:
    """``stepping.update_E`` and the nonlinear text from one seeded state: (start, oracle,
    kernel) snapshots."""
    driver = _driver(case)
    try:
        rng = numpy.random.default_rng(11)
        shape = tuple(int(n) for n in driver.grid.shape)
        if chi_kind == "scalar_chi3":
            driver.set_chi3(0.01)
        elif chi_kind == "volume_chi2_chi3":
            driver.set_chi2(rng.uniform(-0.02, 0.02, size=shape).astype(F32))
            driver.set_chi3(rng.uniform(0.0, 0.02, size=shape).astype(F32))
        elif chi_kind == "near_pole_chi3":
            driver.set_epsilon(rng.uniform(1.2, 3.4, size=shape).astype(F32))
            driver.set_chi3(0.05)
        else:
            assert chi_kind == "dsqr_band", chi_kind
            driver.set_epsilon(rng.uniform(1.2, 3.4, size=shape).astype(F32))
            driver.set_chi3(DSQR_BAND_CHI3)
        _seed(driver, False)
        fields = driver.fields
        if chi_kind == "dsqr_band":
            for name in ("Dx", "Dy", "Dz"):
                getattr(fields, name)[...] *= DSQR_BAND_SCALE
        margin = stepping.nonlinear_margin(fields, driver.pml)
        assert margin is not None and margin.expansion < 1.0 / 3.0, (case, chi_kind, margin)
        start = _snapshot(fields)
        stepping.update_E(fields, driver.pml)
        oracle = _snapshot(fields)
        _restore(fields, start)
        arguments = _arguments("update_E_pml_real", driver,
                               numpy.ascontiguousarray(driver.chunk_bounds_record(),
                                                       dtype=numpy.int32))
        bindings = module.nonlinear_bindings(fields)
        for index, axis in enumerate("xyz"):
            for label in ("chi2", "chi3"):
                arguments[f"{label}_E{axis}"] = numpy.ascontiguousarray(
                    bindings[label][index]).reshape(-1)
                arguments[f"s{label[-1]}_E{axis}"] = F32(bindings[f"{label}_scalar"][index])
            arguments[f"nl_{axis}"] = int(bindings["nonlinear"][index])
            arguments[f"q2v_{axis}"] = int(bindings["chi2_is_volume"][index])
            arguments[f"q3v_{axis}"] = int(bindings["chi3_is_volume"][index])
        codes, refusal = module.boundary_codes(driver.grid)
        assert refusal is None, refusal
        arguments.update({"bc_x": codes[0], "bc_y": codes[1], "bc_z": codes[2]})
        Interpreter(text).launch(arguments, int(numpy.prod(shape)))
        return start, oracle, _snapshot(fields)
    finally:
        driver.close()


@pytest.mark.parametrize("chi_kind", ("scalar_chi3", "volume_chi2_chi3", "dsqr_band"))
@pytest.mark.parametrize("case", ("box3d", "split_y2d"))
def test_the_nonlinear_update_E_text_is_the_array_path_word_for_word(case, chi_kind, roots):
    """The nonlinear ``update_E`` is the array path's, word for word.

    Its constitutive tail takes the per-chunk forms, its transverse sums add the four
    corners left to right in MEEP's order, and its ``|D|^2`` is formed in double and
    rounded once (16-AP6h's forms, after phase-0 P0-h); this checks the whole sub-step
    against ``stepping.update_E`` on a grid carrying chi2/chi3, including the planted
    band where the double ``|D|^2`` is the only form that agrees.
    """
    module = roots["nonlinear_constitutive"]
    start, oracle, got = _run_nonlinear(_texts(roots)["update_E_pml_real_nonlinear"],
                                        case, chi_kind, module)
    differing = {n: int(numpy.count_nonzero(_bits(oracle[n]) != _bits(got[n]))) for n in _STATE}
    assert all(v == 0 for v in differing.values()), (case, chi_kind, differing)
    assert any(numpy.count_nonzero(_bits(oracle[f"E{a}"]) != _bits(start[f"E{a}"])) for a in "xyz")


#: The nonlinear text's two MEEP-arithmetic forms, each put back to the 0.9.2 array
#: path's spelling: ``(find, replace, chi_kind)``. Each must change a word of the
#: comparison above on its case, the float32 ``|D|^2`` only on the planted band.
NONLINEAR_DEFECTS = {
    "four_point_pair_association": (
        "    float own_pair = g[o_c] + (v_u ? g[o_u] : 0.0f);\n"
        "    float with_down = own_pair + (v_d ? g[o_d] : 0.0f);\n"
        "    return with_down + (v_ud ? g[o_ud] : 0.0f);\n",
        "    float near_pair = g[o_c] + (v_d ? g[o_d] : 0.0f);\n"
        "    float far_pair = (v_u ? g[o_u] : 0.0f) + (v_ud ? g[o_ud] : 0.0f);\n"
        "    return near_pair + far_pair;\n",
        "near_pole_chi3"),
    "dsqr_in_float32": (
        "    double own_sq = gs * gs;\n"
        "    double transverse_sq = g1s * g1s + g2s * g2s;\n"
        "    return __double2float_rn(own_sq + 0.0625 * transverse_sq);\n",
        "    float own_sq = gs * gs;\n"
        "    float transverse_sq = g1s * g1s + g2s * g2s;\n"
        "    return own_sq + 0.0625f * transverse_sq;\n",
        "dsqr_band"),
}


@pytest.mark.parametrize("defect", sorted(NONLINEAR_DEFECTS))
def test_each_nonlinear_form_put_back_changes_a_word(defect, roots):
    find, replace, chi_kind = NONLINEAR_DEFECTS[defect]
    text = _texts(roots)["update_E_pml_real_nonlinear"]
    assert text.count(find) == 1, defect
    counts = {}
    for case in ("box3d", "split_y2d"):
        _start, oracle, got = _run_nonlinear(text.replace(find, replace), case, chi_kind,
                                             roots["nonlinear_constitutive"])
        counts[case] = sum(int(numpy.count_nonzero(_bits(oracle[n]) != _bits(got[n])))
                           for n in _STATE)
    assert all(counts.values()), (defect, counts)


def _dsqr_probe(prelude: str) -> str:
    """The shipped prelude with one entry that stores ``nonlinear_dsqr`` per lane."""
    return prelude + (
        'extern "C" __global__ void probe_nonlinear_dsqr(\n'
        "    const float* gs, const float* g1s, const float* g2s, float* out\n"
        ") {\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    out[idx] = nonlinear_dsqr(gs[idx], g1s[idx], g2s[idx]);\n"
        "}\n")


def _dsqr_operands(lanes: int) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Operands around and inside the band, plus every scale from 2^-75 to 2^4."""
    rng = numpy.random.default_rng(20261007)
    out = []
    for _ in range(3):
        scale = numpy.exp2(rng.uniform(-75.0, 4.0, size=lanes))
        band = rng.uniform(size=lanes) < 0.5
        scale[band] = numpy.exp2(rng.uniform(-66.0, -60.0, size=int(band.sum())))
        out.append((rng.uniform(-1.0, 1.0, size=lanes) * scale).astype(F32))
    return out[0], out[1], out[2]


def test_the_dsqr_helper_is_steppings_dsqr_operand_for_operand(roots):
    """``nonlinear_dsqr``'s text, executed, is ``stepping._nonlinear_dsqr`` on every word.

    The band where the double and the float32 forms differ is reached (a vacuity floor),
    and the float32 form put back into the text is caught on exactly the operands where
    the two forms differ, so the comparison measures the form and nothing else.
    """
    lanes = 200_000
    gs, g1s, g2s = _dsqr_operands(lanes)
    oracle = numpy.asarray(stepping._nonlinear_dsqr(gs, (g1s, g2s), stepping._WHOLE))
    assert oracle.dtype == F32
    float32_form = (gs * gs + F32(0.0625) * (g1s * g1s + g2s * g2s)).astype(F32)
    band = int(numpy.count_nonzero(_bits(oracle) != _bits(float32_form)))
    assert band > 0, "no operand reached the band where the two forms differ"

    prelude = roots["nonlinear_constitutive"]._NONLINEAR_PRELUDE
    find, replace, _kind = NONLINEAR_DEFECTS["dsqr_in_float32"]
    for text, expected in ((prelude, 0), (prelude.replace(find, replace), band)):
        out = numpy.full(lanes, numpy.nan, dtype=F32)
        Interpreter(_dsqr_probe(text)).launch(
            {"gs": gs, "g1s": g1s, "g2s": g2s, "out": out}, lanes)
        assert int(numpy.count_nonzero(_bits(out) != _bits(oracle))) == expected, expected


# =============================================================================
# 3. The register view keeps the auxiliary contract
# =============================================================================

@pytest.mark.parametrize("kernel", ("update_H_pml_real", "update_E_pml_real"))
def test_an_auxiliary_sentinel_off_the_run_survives_the_register_view(kernel, roots):
    driver = _driver("box3d")
    try:
        _seed(driver, False)
        fields = driver.fields
        components, sub_step, aux = STEPPED[kernel]
        plan = fields.region_plan
        planted = {}
        for component in components:
            off = ~numpy.broadcast_to(plan.mask(sub_step, component, numpy), fields.grid.shape)
            getattr(fields, aux + component)[off] = F32(numpy.nan)
            planted[component] = off
        record = numpy.ascontiguousarray(driver.chunk_bounds_record(), dtype=numpy.int32)
        Interpreter(_texts(roots)[kernel]).launch(_arguments(kernel, driver, record),
                                                  int(numpy.prod(driver.grid.shape)))
        for component, off in planted.items():
            assert numpy.isnan(getattr(fields, aux + component)[off]).all(), component
            assert not numpy.isnan(getattr(fields, component)).any(), component
    finally:
        driver.close()


@pytest.mark.parametrize("kernel", ("update_H_pml_real", "update_E_pml_real"))
def test_the_hoisted_singles_guard_their_loads_and_auxiliary_stores(kernel, roots):
    text = _texts(roots)[kernel]
    for line in own_cell_hoist.load_block(kernel):
        assert line + "\n" in text
        assert " ? " in line or line.startswith("    float src_")
    for line in own_cell_hoist.writeback_block(kernel):
        assert line + "\n" in text
    statements = own_cell_hoist.unhoisted_kernel_code(text, kernel, kernel)
    assert own_cell_hoist.hoisted_kernel_code(statements, kernel, kernel) == text


def test_a_helper_without_a_membership_flag_is_refused_by_name(roots):
    """A text not stepped to the per-chunk forms is refused, not hoisted on a flag it lacks."""
    text = _texts(roots)["update_E_pml_real"]
    old = text.replace("float kps, float kms, int dsigw\n) {", "float kps, float kms\n) {", 1)
    with pytest.raises(AssertionError, match="takes no membership flag"):
        own_cell_hoist.assert_helper_reads_before_it_stores(old, "constitutive_apply")
