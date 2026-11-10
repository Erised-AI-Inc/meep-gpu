"""Compulsory memory traffic per cell-step of the dispatched pml_3d plans, read off the kernels.

What it counts. For each kernel a plan launches per timestep, the words it must load and the
words it must store, assuming every element is touched once per kernel (a neighbour read of
an array the kernel already reads at its own cell costs nothing more). Summed over the plan's
launches and converted with the bound dtypes, that is the plan's compulsory traffic per
cell-step: a LOWER BOUND on what the kernels move. Caches that miss more, halo re-reads, the
1-D PML coefficient vectors' re-reads and the per-launch chunk record are not compulsory
traffic; the 1-D vectors and the record are still counted once per launch (they are O(n) and
O(1), next to the O(n^3) volumes).

Where the count comes from. Never from a hand-written table of arrays. The kernel TEXT each
table compiles for the plan is obtained from the package's own emitters (the same functions
the launchers call) and parsed:

* CUDA and Metal (C-like): every ``P[...]`` of a pointer parameter is a load, or a store when
  it is the left side of ``=``; a ``__device__``/``inline`` helper that receives a pointer is
  summarised once and its accesses mapped through each call's arguments.
* Triton: every ``tl.load(P + ...)`` / ``tl.store(P + ...)`` of the ``@triton.jit`` function
  read from its source file with :mod:`ast` (no import of Triton), ``tl.constexpr`` branches
  resolved with the plan's specialisation, helpers mapped through their call arguments.

A GUARD is what decides how many cells an access reaches. An access under a PML-form
membership flag -- the chunk record's ``in_{h|i}_{x|y|z}`` (CUDA, Metal) or a
``record_in(cb, line, coord)`` result in a ``tl.load`` mask (Triton) -- reaches only the rows
of that (lattice, axis) line the record marks, times the other two extents; its negation
reaches the complement. Every other condition (neighbour validity, the metallic ghost, a
presence bit) is not a cell-count guard, and such an access counts as reaching every cell. An
array's load set (store set) is the union of its loads (stores); a flag and its negation
together reach every cell.

The element counts of each guard come from the REAL lifted case: the int32[25] chunk record
(``chunk_map.chunk_bounds_record``) of the pml_3d lift at each ladder size, recorded in the
file passed as ``--lift-facts`` (one JSON line per resolution). An index that is a single grid
coordinate (``i``, ``j``, ``k``) marks a 1-D coefficient vector, a constant index the record;
anything else a volume.

MEEP's step loop is not parsed (its loops are C++ templates over chunks): the per-loop access
sets are transcribed from the MEEP 1.33.0 source with their line references
(:data:`MEEP_LOOPS`) and applied mechanically per chunk of MEEP's own division at the rank
count of the row (the file passed as ``--meep-chunks``). Each (chunk, component) loop is a kernel in
the same once-per-kernel sense.

USAGE
    python bytes_per_cell_step.py --lift-facts lift_facts.jsonl [--meep-chunks meep_chunks.jsonl]
        [--json out.json] [--report out.txt]

Runs on any host: the CUDA kernel modules import CuPy at module scope, so they are imported
against a NumPy stand-in that is removed from ``sys.modules`` afterwards (the package tests'
own device-free pattern); nothing is compiled or launched.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

#: Bytes per element of a C parameter's base type.
C_TYPE_BYTES = {"float": 4, "int": 4, "uint": 4, "float2": 8, "double": 8, "half": 2}

#: The record line of (half_integer, axis): chunk_map.record_index(half, axis, 0) // 4.
AXES = "xyz"


def line_of(half: bool, axis: int) -> int:
    return (3 if half else 0) + axis


def line_name(line: int) -> str:
    return f"{'half' if line >= 3 else 'int'}_{AXES[line % 3]}"


# ---------------------------------------------------------------------------------------------
# Accesses
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Access:
    """One load or store of a kernel pointer parameter."""
    array: str               # the kernel's own parameter name
    kind: str                # "load" or "store"
    guard: Optional[Tuple[int, bool]]   # (record line, polarity) or None = every cell
    extent: str              # "volume", "line:x|y|z", "record"
    elem_bytes: int


def _region_of(name: str, aliases: Dict[str, str]) -> Optional[int]:
    """The record line a C membership flag names (chunk_bounds.membership_name), or None."""
    seen = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    m = re.fullmatch(r"in_([hi])_([xyz])", name)
    if not m:
        return None
    return line_of(m.group(1) == "h", AXES.index(m.group(2)))


# ---------------------------------------------------------------------------------------------
# C-like front end (CUDA, Metal)
# ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(\d+\.\d*f?|\d+\.?\d*[eE][-+]?\d+f?|\d+[uUlLf]*)|(\w+)|"
                    r"(==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=|<<|>>|\+\+|--|->|::|\[\[|\]\]|.))", re.S)


def _strip_c(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    src = re.sub(r"^\s*#[^\n]*", " ", src, flags=re.M)
    return src


def _tokens(text: str) -> List[str]:
    out = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            raise ValueError(f"cannot tokenize at {text[pos:pos + 40]!r}")
        tok = m.group(1) or m.group(2) or m.group(3)
        if tok.strip():
            out.append(tok)
        pos = m.end()
    return out


def _match(tokens: List[str], start: int, open_: str, close: str) -> int:
    depth = 0
    for k in range(start, len(tokens)):
        if tokens[k] == open_:
            depth += 1
        elif tokens[k] == close:
            depth -= 1
            if depth == 0:
                return k
    raise ValueError(f"unbalanced {open_}{close}")


@dataclass
class CParam:
    name: str
    pointer: bool
    const: bool
    elem_bytes: int


@dataclass
class CFunction:
    name: str
    params: List[CParam]
    body: List[str]
    is_kernel: bool


_QUALIFIERS = {"__device__", "__global__", "__forceinline__", "kernel", "inline", "static",
               "extern", "__host__"}


def _c_functions(src: str) -> Dict[str, CFunction]:
    toks = _tokens(_strip_c(src))
    funcs: Dict[str, CFunction] = {}
    k = 0
    while k < len(toks):
        if toks[k] in ("__device__", "__global__", "kernel", "inline"):
            # head: qualifiers and a return type, then NAME ( params ) {
            j = k
            while j < len(toks) and toks[j] != "(":
                j += 1
            if j >= len(toks):
                break
            name = toks[j - 1]
            close = _match(toks, j, "(", ")")
            if close + 1 < len(toks) and toks[close + 1] == "{":
                end = _match(toks, close + 1, "{", "}")
                params = _c_params(toks[j + 1:close])
                is_kernel = "__global__" in toks[k:j] or ("kernel" in toks[k:j])
                funcs[name] = CFunction(name, params, toks[close + 2:end], is_kernel)
                k = end + 1
                continue
            k = close + 1
            continue
        k += 1
    return funcs


def _c_params(toks: List[str]) -> List[CParam]:
    params, cur, depth = [], [], 0
    for t in toks + [","]:
        if t in ("(", "[", "[["):
            depth += 1
        elif t in (")", "]", "]]"):
            depth -= 1
        if t == "," and depth == 0:
            if cur:
                params.append(cur)
            cur = []
        else:
            cur.append(t)
    out = []
    for p in params:
        # drop [[...]] attributes
        clean, skip = [], 0
        for t in p:
            if t == "[[":
                skip += 1
                continue
            if t == "]]":
                skip -= 1
                continue
            if not skip:
                clean.append(t)
        idents = [t for t in clean if re.fullmatch(r"\w+", t)]
        if not idents:
            continue
        name = idents[-1]
        base = next((t for t in clean if t in C_TYPE_BYTES), None)
        out.append(CParam(name, "*" in clean, "const" in clean,
                          C_TYPE_BYTES.get(base, 4)))
    return out


def _split_args(toks: List[str]) -> List[List[str]]:
    args, cur, depth = [], [], 0
    for t in toks:
        if t in ("(", "["):
            depth += 1
        elif t in (")", "]"):
            depth -= 1
        if t == "," and depth == 0:
            args.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        args.append(cur)
    return args


def _extent_of(index: List[str], consts: set) -> str:
    names = [t for t in index if re.fullmatch(r"[A-Za-z_]\w*", t)]
    if index and all(re.fullmatch(r"\d+[uU]?", t) or t in consts or t in "+-*" for t in index):
        return "record"
    if len(names) == 1 and names[0] in ("i", "j", "k") and len(index) == 1:
        return "line:" + {"i": "x", "j": "y", "k": "z"}[names[0]]
    return "volume"


@dataclass
class _Raw:
    """An access inside a C function, before the caller maps it: guard conditions as text."""
    array: str
    kind: str
    conds: Tuple[Tuple[str, bool], ...]
    index: Tuple[str, ...]


#: The pml_3d plan's runtime scalar arguments that switch whole code paths off. pml_3d
#: declares no symmetry, so the in-seam fills' plan (in_seam_coverage.mirror_fill_phases,
#: .folded_far_rows) is near_a = 0 and reflect_a = -1 on every axis: those blocks never run.
RUNTIME_SCALARS = {"near_x": 0, "near_y": 0, "near_z": 0,
                   "reflect_x": -1, "reflect_y": -1, "reflect_z": -1}


def _dead(cond: List[str]) -> bool:
    """True when a top-level conjunct of an if-condition is false under RUNTIME_SCALARS."""
    conj, cur, depth = [], [], 0
    for t in cond + ["&&"]:
        if t == "(":
            depth += 1
        elif t == ")":
            depth -= 1
        if t == "&&" and depth == 0:
            conj.append(cur)
            cur = []
        else:
            cur.append(t)
    for c in conj:
        names = [t for t in c if re.fullmatch(r"[A-Za-z_]\w*", t)]
        if not names or any(n not in RUNTIME_SCALARS for n in names):
            continue
        expr = " ".join(str(RUNTIME_SCALARS.get(t, t)) for t in c)
        expr = expr.replace("!", " not ").replace("not =", "!=")
        try:
            if not eval(expr, {"__builtins__": {}}):  # noqa: S307 - integers and comparisons only
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _c_body_accesses(fn: CFunction, funcs: Dict[str, CFunction], memo: Dict[str, List[_Raw]]
                     ) -> List[_Raw]:
    """Every access of a pointer parameter of ``fn`` (directly or through a helper)."""
    if fn.name in memo:
        return memo[fn.name]
    ptrs = {p.name for p in fn.params if p.pointer}
    toks = fn.body
    out: List[_Raw] = []
    if_stack: List[Tuple[Tuple[str, bool], int, str]] = []  # (cond, depth-marker, kind)
    tern: List[List[Any]] = []   # frames [cond, paren_depth, branch]
    brace, paren = 0, 0
    last_if: Optional[str] = None
    expr_start = 0
    k = 0

    def current_conds() -> Tuple[Tuple[str, bool], ...]:
        conds = [c for c, _, _ in if_stack]
        for cond, _d, branch in tern:
            conds.append((cond, branch == "then"))
        return tuple(conds)

    while k < len(toks):
        t = toks[k]
        if t == "if" and k + 1 < len(toks) and toks[k + 1] == "(":
            close = _match(toks, k + 1, "(", ")")
            if _dead(toks[k + 2:close]):
                # a path the plan's runtime scalars switch off: skip its block or statement
                if close + 1 < len(toks) and toks[close + 1] == "{":
                    k = _match(toks, close + 1, "{", "}") + 1
                else:
                    j = close + 1
                    while toks[j] != ";":
                        j += 1
                    k = j + 1
                if k < len(toks) and toks[k] == "else":
                    raise ValueError("an else after a switched-off if: not in these kernels")
                expr_start = k
                continue
            cond = " ".join(toks[k + 2:close])
            # accesses inside the condition itself are unguarded reads
            for raw in _scan_expr(toks[k + 2:close], ptrs, funcs, memo, current_conds()):
                out.append(raw)
            if close + 1 < len(toks) and toks[close + 1] == "{":
                if_stack.append(((cond, True), brace + 1, "block"))
                brace += 1
                k = close + 2
            else:
                if_stack.append(((cond, True), brace, "stmt"))
                k = close + 1
            last_if = cond
            expr_start = k
            continue
        if t == "else":
            cond = last_if or "?"
            if k + 1 < len(toks) and toks[k + 1] == "{":
                if_stack.append(((cond, False), brace + 1, "block"))
                brace += 1
                k += 2
            else:
                if_stack.append(((cond, False), brace, "stmt"))
                k += 1
            expr_start = k
            continue
        if t == "{":
            brace += 1
        elif t == "}":
            while if_stack and if_stack[-1][2] == "block" and if_stack[-1][1] == brace:
                popped = if_stack.pop()
                last_if = popped[0][0]
            brace -= 1
            # a single-statement if whose statement was this block
            expr_start = k + 1
        elif t == "(":
            paren += 1
        elif t == ")":
            while tern and tern[-1][1] >= paren:
                tern.pop()
            paren -= 1
        elif t == "?":
            # the condition: back to the expression start at this paren depth
            j, depth = k - 1, 0
            while j >= expr_start:
                if toks[j] == ")":
                    depth += 1
                elif toks[j] == "(":
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0 and toks[j] in ("=", ",", "return", "?", ":", "&&", "||"):
                    break
                j -= 1
            cond = " ".join(toks[j + 1:k])
            tern.append([cond, paren, "then"])
        elif t == ":" and tern and tern[-1][1] == paren and tern[-1][2] == "then":
            tern[-1][2] = "else"
        elif t in (",", ";"):
            while tern and tern[-1][1] >= paren:
                tern.pop()
            if t == ";":
                while if_stack and if_stack[-1][2] == "stmt" and if_stack[-1][1] == brace:
                    popped = if_stack.pop()
                    last_if = popped[0][0]
                expr_start = k + 1
        elif t in ptrs and k + 1 < len(toks) and toks[k + 1] == "[":
            close = _match(toks, k + 1, "[", "]")
            index = tuple(toks[k + 2:close])
            nxt = toks[close + 1] if close + 1 < len(toks) else ""
            conds = current_conds()
            if nxt == "=":
                out.append(_Raw(t, "store", conds, index))
            elif nxt in ("+=", "-=", "*=", "/="):
                out.append(_Raw(t, "load", conds, index))
                out.append(_Raw(t, "store", conds, index))
            else:
                out.append(_Raw(t, "load", conds, index))
            # the index expression may itself read pointers (not in these kernels)
            k = close + 1
            continue
        elif t in funcs and k + 1 < len(toks) and toks[k + 1] == "(" and t != fn.name:
            close = _match(toks, k + 1, "(", ")")
            args = _split_args(toks[k + 2:close])
            out.extend(_map_call(funcs[t], args, ptrs, funcs, memo, current_conds()))
            # arguments that are expressions may contain direct accesses or calls
            for a in args:
                if not (len(a) == 1):
                    out.extend(_scan_expr(a, ptrs, funcs, memo, current_conds()))
            k = close + 1
            continue
        k += 1
    memo[fn.name] = out
    return out


def _scan_expr(toks: List[str], ptrs: set, funcs: Dict[str, CFunction],
               memo: Dict[str, List[_Raw]], conds) -> List[_Raw]:
    """Accesses in a bare expression (a call argument or an if-condition): loads, and calls."""
    out = []
    k = 0
    while k < len(toks):
        t = toks[k]
        if t in ptrs and k + 1 < len(toks) and toks[k + 1] == "[":
            close = _match(toks, k + 1, "[", "]")
            out.append(_Raw(t, "load", conds, tuple(toks[k + 2:close])))
            k = close + 1
            continue
        if t in funcs and k + 1 < len(toks) and toks[k + 1] == "(":
            close = _match(toks, k + 1, "(", ")")
            args = _split_args(toks[k + 2:close])
            out.extend(_map_call(funcs[t], args, ptrs, funcs, memo, conds))
            k = close + 1
            continue
        k += 1
    return out


def _map_call(callee: CFunction, args: List[List[str]], ptrs: set, funcs, memo, conds
              ) -> List[_Raw]:
    """A helper's accesses, renamed into the caller: pointer params to the caller's pointer
    arguments, guard and index names to the caller's argument text."""
    inner = _c_body_accesses(callee, funcs, memo)
    if len(args) != len(callee.params):
        raise ValueError(f"call of {callee.name} with {len(args)} args, {len(callee.params)} params")
    sub = {p.name: a for p, a in zip(callee.params, args)}
    out = []
    for raw in inner:
        arg = sub.get(raw.array)
        if arg is None:
            raise ValueError(f"{callee.name}: access of {raw.array}, which is not a parameter")
        if not (len(arg) == 1 and arg[0] in ptrs):
            if any(t in ptrs for t in arg):
                raise ValueError(f"{callee.name}: pointer {raw.array} bound to the expression {arg}")
            continue  # a register's address (the own-cell hoist's &w) or a local: no array

        mapped_conds = []
        for cond, pol in raw.conds:
            if cond in sub:
                cond = " ".join(sub[cond])
            mapped_conds.append((cond, pol))
        index: List[str] = []
        for t in raw.index:
            index.extend(sub.get(t, [t]))
        out.append(_Raw(arg[0], raw.kind, tuple(conds) + tuple(mapped_conds), tuple(index)))
    return out


def _c_aliases(fn: CFunction) -> Dict[str, str]:
    """``bool a = b, c = d;``-style local aliases of single identifiers."""
    aliases = {}
    toks = fn.body
    for k in range(1, len(toks) - 2):
        if (re.fullmatch(r"[A-Za-z_]\w*", toks[k]) and toks[k + 1] == "="
                and re.fullmatch(r"[A-Za-z_]\w*", toks[k + 2])
                and toks[k + 3] in (",", ";") and toks[k - 1] in ("bool", "int", ",")):
            aliases[toks[k]] = toks[k + 2]
    return aliases


def c_kernel_accesses(source: str, kernel: Optional[str] = None) -> Tuple[str, List[Access]]:
    funcs = _c_functions(source)
    kernels = [f for f in funcs.values() if f.is_kernel]
    if kernel:
        kernels = [f for f in kernels if f.name == kernel]
    if len(kernels) != 1:
        raise ValueError(f"expected one kernel, found {[f.name for f in kernels]}")
    fn = kernels[0]
    aliases = _c_aliases(fn)
    consts = {t for t in fn.body if re.fullmatch(r"[A-Z][A-Z0-9_]+", t)}
    raws = _c_body_accesses(fn, funcs, {})
    by_name = {p.name: p for p in fn.params}
    out = []
    for raw in raws:
        guard = None
        for cond, pol in raw.conds:
            c = cond.strip()
            neg = False
            while c.startswith("!"):
                neg, c = not neg, c[1:].strip()
            if c.startswith("(") and c.endswith(")"):
                c = c[1:-1].strip()
            line = _region_of(c, aliases)
            if line is None:
                continue
            if guard is not None and guard != (line, pol != neg):
                raise ValueError(f"{raw.array}: two cell-count guards {guard} and {(line, pol)}")
            guard = (line, pol != neg)
        out.append(Access(raw.array, raw.kind, guard, _extent_of(list(raw.index), consts),
                          by_name[raw.array].elem_bytes))
    return fn.name, out


# ---------------------------------------------------------------------------------------------
# Triton front end (ast, no import of Triton)
# ---------------------------------------------------------------------------------------------

def _py_functions(path: str) -> Dict[str, ast.FunctionDef]:
    tree = ast.parse(open(path).read())
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _py_constants(path: str) -> Dict[str, int]:
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if (isinstance(v, ast.Call) and getattr(v.func, "attr", "") == "constexpr" and v.args
                    and isinstance(v.args[0], ast.Constant)):
                out[node.targets[0].id] = v.args[0].value
            elif isinstance(v, ast.Constant) and isinstance(v.value, int):
                out[node.targets[0].id] = v.value
    return out


class _Unknown(Exception):
    pass


def _eval(node: ast.AST, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env and not isinstance(env[node.id], tuple):
            return env[node.id]
        raise _Unknown(node.id)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
               ast.FloorDiv: lambda: a // b}
        return ops[type(node.op)]()
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        a, b = _eval(node.left, env), _eval(node.comparators[0], env)
        return {ast.Eq: a == b, ast.NotEq: a != b}[type(node.ops[0])]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval(node.operand, env)
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    raise _Unknown(ast.dump(node)[:60])


def _base_and_offset(node: ast.AST) -> Tuple[str, Optional[ast.AST]]:
    """``P + a + b`` -> ("P", a + b): the leftmost name of an addition chain."""
    terms = []

    def flatten(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            flatten(n.left)
            flatten(n.right)
        else:
            terms.append(n)
    flatten(node)
    if not isinstance(terms[0], ast.Name):
        raise ValueError(f"pointer expression without a leading name: {ast.dump(node)[:80]}")
    rest = terms[1:]
    return terms[0].id, (rest[0] if len(rest) == 1 else (ast.Tuple(rest) if rest else None))


def _names(node: Optional[ast.AST]) -> List[str]:
    return [] if node is None else [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


#: Scalars that size the grid: an offset made of these and one coordinate is a 1-D index.
SIZE_NAMES = {"nx", "ny", "nz", "nyz", "n_elem"}


def _sizey(name: str, env: Dict[str, Any]) -> bool:
    return (name in SIZE_NAMES or name.isupper() or isinstance(env.get(name), int)
            or env.get(name) == ("size",))


def _py_extent(offset: Optional[ast.AST], env: Dict[str, Any]) -> str:
    names = _names(offset)
    coords = [n for n in names if n in ("i", "j", "k")]
    others = [n for n in names if n not in ("i", "j", "k")]
    consts = all(_sizey(n, env) for n in others)
    if not names or (not coords and consts):
        return "record"
    if len(set(coords)) == 1 and consts and len(coords) == 1:
        return "line:" + {"i": "x", "j": "y", "k": "z"}[coords[0]]
    return "volume"


def _mask_guard(mask: Optional[ast.AST], env: Dict[str, Any]) -> Optional[Tuple[int, bool]]:
    if mask is None:
        return None
    guard = None
    for node in ast.walk(mask):
        if isinstance(node, ast.Name) and isinstance(env.get(node.id), tuple) and env[node.id][0] == "record":
            pol = True
            g = (env[node.id][1], pol)
            if guard is not None and guard != g:
                raise ValueError(f"two cell-count guards in one mask: {guard} {g}")
            guard = g
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            if any(isinstance(env.get(n), tuple) for n in _names(node.operand)):
                raise ValueError("a negated record mask: not in these kernels, refused by name")
    return guard


def triton_kernel_accesses(path: str, kernel: str, constexpr: Dict[str, int],
                           helper_paths: Sequence[str]) -> List[Access]:
    funcs: Dict[str, ast.FunctionDef] = {}
    consts: Dict[str, Any] = {}
    for p in list(helper_paths) + [path]:
        funcs.update(_py_functions(p))
        consts.update(_py_constants(p))
    fn = funcs[kernel]
    ptr_params = [a.arg for a in fn.args.args if a.arg not in constexpr]
    env: Dict[str, Any] = dict(consts)
    env.update(constexpr)
    accesses: List[Access] = []
    _py_walk(fn.body, env, funcs, {p: p for p in ptr_params}, accesses, outer_guard=None)
    return accesses


def _py_walk(stmts, env, funcs, ptrmap: Dict[str, str], out: List[Access], outer_guard) -> None:
    for st in stmts:
        if isinstance(st, ast.If):
            try:
                taken = _eval(st.test, env)
            except _Unknown:
                _py_walk(st.body, env, funcs, ptrmap, out, outer_guard)
                _py_walk(st.orelse, env, funcs, ptrmap, out, outer_guard)
                continue
            _py_walk(st.body if taken else st.orelse, env, funcs, ptrmap, out, outer_guard)
            continue
        if isinstance(st, ast.Assign):
            _py_calls(st.value, env, funcs, ptrmap, out, outer_guard)
            for target in st.targets:
                _bind(target, st.value, env)
            continue
        if isinstance(st, (ast.Expr, ast.Return, ast.AugAssign)):
            _py_calls(st.value, env, funcs, ptrmap, out, outer_guard)
            continue
        if isinstance(st, (ast.ImportFrom, ast.Import, ast.Pass)):
            continue
        raise ValueError(f"statement kind not handled: {type(st).__name__}")


def _bind(target, value, env) -> None:
    if isinstance(target, ast.Tuple):
        if isinstance(value, ast.Tuple) and len(value.elts) == len(target.elts):
            for t, v in zip(target.elts, value.elts):
                _bind(t, v, env)
        else:
            for t in target.elts:
                if isinstance(t, ast.Name):
                    env.pop(t.id, None)
        return
    if not isinstance(target, ast.Name):
        return
    if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "record_in":
        env[target.id] = ("record", _eval(value.args[1], env) // 4)
        return
    try:
        env[target.id] = _eval(value, env)
    except _Unknown:
        names = _names(value)
        if names and all(_sizey(n, env) for n in names):
            env[target.id] = ("size",)
        else:
            env.pop(target.id, None)


def _py_calls(node, env, funcs, ptrmap, out, outer_guard) -> None:
    if node is None:
        return
    for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
        f = call.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tl" \
                and f.attr in ("load", "store"):
            base, offset = _base_and_offset(call.args[0])
            if base not in ptrmap:
                continue
            mask = next((kw.value for kw in call.keywords if kw.arg == "mask"), None)
            guard = _mask_guard(mask, env) or outer_guard
            elem = 4
            out.append(Access(ptrmap[base], f.attr, guard, _py_extent(offset, env), elem))
        elif isinstance(f, ast.Name) and f.id in funcs and f.id != "record_in":
            callee = funcs[f.id]
            params = [a.arg for a in callee.args.args]
            sub_env = {k: v for k, v in env.items() if k.isupper()}
            sub_ptr = {}
            for p, a in zip(params, call.args):
                if isinstance(a, ast.Name) and a.id in ptrmap:
                    sub_ptr[p] = ptrmap[a.id]
                elif isinstance(a, ast.Name) and a.id in env:
                    sub_env[p] = env[a.id]
                else:
                    try:
                        sub_env[p] = _eval(a, env)
                    except _Unknown:
                        pass
            for kw in call.keywords:
                try:
                    sub_env[kw.arg] = _eval(kw.value, env)
                except _Unknown:
                    pass
            for p, a in zip(params, call.args):
                if p not in sub_ptr and p not in sub_env and isinstance(a, ast.Name) and a.id in env:
                    sub_env[p] = env[a.id]
            # a helper's own masks are its parameters; a parameter bound to a record result
            # carries that guard in.
            if sub_ptr:
                _py_walk(callee.body, sub_env, funcs, sub_ptr, out, outer_guard)


# ---------------------------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------------------------

def _cupy_stand_in():
    import numpy  # noqa: PLC0415
    stub = types.ModuleType("cupy")

    class _Raw:  # noqa: D401 - a do-nothing kernel handle; nothing is compiled
        def __init__(self, *a, **k):
            pass
    stub.RawKernel = _Raw
    stub.RawModule = _Raw
    stub.__getattr__ = lambda name: getattr(numpy, name)  # type: ignore[attr-defined]
    return stub


def kernel_sources(row_mask=(1, 1, 1, 1, 1, 1), codes=(1, 1, 1), walls=(1, 1, 1)) -> Dict[str, Any]:
    """Every kernel text the three tables compile for the pml_3d plans, from the emitters.

    ``codes`` (ghost rule per axis) and ``walls`` change index validity only, never which
    arrays or which cells a kernel touches; METALLIC is pml_3d's (MEEP's default walls).
    """
    sys.path.insert(0, REPO)
    had_cupy = "cupy" in sys.modules
    if not had_cupy:
        try:
            import cupy  # noqa: F401,PLC0415
            had_cupy = True
        except Exception:  # noqa: BLE001
            sys.modules["cupy"] = _cupy_stand_in()
    # Without CuPy the hand-CUDA modules are imported afresh under the stand-in and put back
    # after: a copy imported earlier in the process while CuPy was blocked has its certified
    # halves set to None, and this process's own copies must not be replaced.
    stashed = {} if had_cupy else {name: sys.modules.pop(name) for name in list(sys.modules)
                                   if name.startswith("meep_gpu.cuda_kernels")}
    try:
        from meep_gpu.cuda_kernels import (constitutive_kernels, fused_magnetic_pair as cfm,  # noqa: PLC0415
                                           offdiag_emitter, step_curl_kernels)
        cuda = {
            "fused_magnetic_pair": ("c", cfm.fused_magnetic_pair_source(), "fused_magnetic_pair_pml_real"),
            "pml_curl_D": ("c", step_curl_kernels._step_D_pml_real_kernel_code, "step_D_pml_real"),  # noqa: SLF001
            "pml_curl_B": ("c", step_curl_kernels._step_B_pml_real_kernel_code, "step_B_pml_real"),  # noqa: SLF001
            "constitutive_H": ("c", constitutive_kernels._update_H_pml_real_kernel_code  # noqa: SLF001
                               if hasattr(constitutive_kernels, "_update_H_pml_real_kernel_code")
                               else None, "update_H_pml_real"),
            "offdiag_E": ("c", offdiag_emitter.offdiag_launch_source(row_mask), "update_E_pml_real_offdiag"),
        }
    finally:
        if not had_cupy:
            for name in [m for m in sys.modules if m == "cupy" or m.startswith("meep_gpu.cuda_kernels")]:
                del sys.modules[name]
            sys.modules.update(stashed)
            if "meep_gpu.cuda_kernels" in stashed:
                import meep_gpu  # noqa: PLC0415
                meep_gpu.cuda_kernels = stashed["meep_gpu.cuda_kernels"]
    from meep_gpu.metal_kernels import fused_magnetic_pair as mfm, offdiag_update_e, shaders  # noqa: PLC0415
    metal = {
        "fused_magnetic_pair": ("c", mfm.fused_magnetic_pair_source(codes, walls), "fused_magnetic_pair_step"),
        "pml_curl_D": ("c", shaders.curl_source(codes, True), "pml_curl_step"),
        "pml_curl_B": ("c", shaders.curl_source(codes, False), "pml_curl_step"),
        "constitutive_H": ("c", shaders.constitutive_source("H"), "constitutive_step"),
        "offdiag_E": ("c", offdiag_update_e.offdiag_source(row_mask, codes, walls), "offdiag_constitutive_step"),
    }
    tk = os.path.join(REPO, "meep_gpu", "triton_kernels")
    helpers = [os.path.join(tk, "forms.py"), os.path.join(tk, "kernels.py")]
    bc = {"BCX": codes[0], "BCY": codes[1], "BCZ": codes[2]}
    rows = dict(zip(("R01", "R02", "R11", "R12", "R21", "R22"), row_mask))
    pres = dict(zip(("P01", "P02", "P11", "P12", "P21", "P22"), row_mask))
    triton = {
        "fused_magnetic_pair": ("py", os.path.join(tk, "kernels.py"), "fused_curl_constitutive_B",
                                {"BACKWARD": 0, **bc, "ZM_X": walls[0], "ZM_Y": walls[1], "ZM_Z": walls[2],
                                 "BLOCK": 256}, helpers),
        "pml_curl_D": ("py", os.path.join(tk, "kernels.py"), "pml_curl_step", {"BACKWARD": 1, **bc, "BLOCK": 256},
                       helpers),
        "pml_curl_B": ("py", os.path.join(tk, "kernels.py"), "pml_curl_step", {"BACKWARD": 0, **bc, "BLOCK": 256},
                       helpers),
        "constitutive_H": ("py", os.path.join(tk, "kernels.py"), "constitutive_step",
                           {"SCALE": 0, **bc, "BLOCK": 256}, helpers),
        "offdiag_E": ("py", os.path.join(tk, "offdiag_update_e.py"), "offdiag_constitutive_step",
                      {**rows, **pres, "PRESENCE": int(any(row_mask)), **bc,
                       "WM_X": walls[0], "WM_Y": walls[1], "WM_Z": walls[2], "BLOCK": 256}, helpers),
    }
    return {"triton": triton, "cuda": cuda, "metal": metal}


#: The kernel text behind each (backend, family, slot) a row record's launch counters name.
#: A family not listed here is REFUSED by name: its kernels have not been read, so no count of
#: it exists (e.g. a dispersive, a deposit-repair-bracketed or a conductive plan).
FAMILY_KERNEL = {
    ("triton", "fused_electric", "step_B"): "fused_magnetic_pair",   # "fused pair B"
    ("triton", "fused_electric", "update_H"): "fused_magnetic_pair",
    ("triton", "pml_curl", "step_B"): "pml_curl_B",
    ("triton", "pml_curl", "step_D"): "pml_curl_D",
    ("triton", "constitutive", "update_H"): "constitutive_H",
    ("triton", "offdiag", "update_E"): "offdiag_E",
    ("cuda", "cuda_fused_magnetic_pair", "step_B"): "fused_magnetic_pair",
    ("cuda", "cuda_fused_magnetic_pair", "update_H"): "fused_magnetic_pair",
    ("cuda", "cuda_curl", "step_B"): "pml_curl_B",
    ("cuda", "cuda_curl", "step_D"): "pml_curl_D",
    ("cuda", "cuda_constitutive", "update_H"): "constitutive_H",
    ("cuda", "cuda_offdiag", "update_E"): "offdiag_E",
    ("metal", "fused_magnetic_pair", "step_B"): "fused_magnetic_pair",
    ("metal", "fused_magnetic_pair", "update_H"): "fused_magnetic_pair",
    ("metal", "pml_curl", "step_B"): "pml_curl_B",
    ("metal", "pml_curl", "step_D"): "pml_curl_D",
    ("metal", "constitutive", "update_H"): "constitutive_H",
    ("metal", "offdiag_constitutive", "update_E"): "offdiag_E",
}

#: The four slots of a timestep, in MEEP's order.
SLOTS = ("step_B", "update_H", "step_D", "update_E")


class Refused(ValueError):
    """A plan this tool has no kernel count for, named."""


def plan_from_counters(slots: Dict[str, Dict[str, Any]]) -> Tuple[str, Tuple[str, ...]]:
    """(table, launches per step) from one leg's ``launch_counters[...]["slots"]``.

    Every slot must be filled by a mapped family on one backend; a label filling two slots
    (a fused pair) is one launch. Equal dispatch counts across slots are required, so a slot
    launched a different number of times than the others is refused rather than mis-weighted.
    """
    extra = sorted(set(slots) - set(SLOTS))
    if extra:
        named = {s: (slots[s].get("backend"), slots[s].get("family"), slots[s].get("arm")) for s in extra}
        raise Refused(f"slots beyond {SLOTS}: {named}; no kernel text read for them")
    if set(slots) != set(SLOTS):
        raise Refused(f"slots {sorted(slots)}: a plan needs all four of {SLOTS}")
    backends = {slots[s]["backend"] for s in SLOTS}
    if len(backends) != 1:
        raise Refused(f"slots on several tables {sorted(backends)}: not counted")
    counts = {slots[s]["dispatches"] for s in SLOTS}
    if len(counts) != 1:
        raise Refused(f"unequal dispatch counts {counts} across slots")
    backend = backends.pop()
    launches: List[str] = []
    for s in SLOTS:
        key = (backend, slots[s]["family"], s)
        if key not in FAMILY_KERNEL:
            raise Refused(f"no kernel text read for {key} (arm {slots[s].get('arm')!r})")
        label = FAMILY_KERNEL[key]
        if label not in launches:
            launches.append(label)
    return backend, tuple(launches)


#: The launches per timestep of the pml_3d routes, as the 1.0 row records' counters give them
#: (plan_from_counters reproduces both; kept for the report's two named plans).
PLANS = {
    "fused": ("fused_magnetic_pair", "pml_curl_D", "offdiag_E"),
    "unfused": ("pml_curl_B", "constitutive_H", "pml_curl_D", "offdiag_E"),
}


def row_mask_of(facts: Dict[str, Any]) -> Tuple[int, ...]:
    """The offdiag kernel's row mask (coverage.OFFDIAG_ROW_SLOTS order) from the lift."""
    slots = (("Ex", "Ey"), ("Ex", "Ez"), ("Ey", "Ez"), ("Ey", "Ex"), ("Ez", "Ex"), ("Ez", "Ey"))
    return tuple(int(p in facts["offdiag"].get(c, {})) for c, p in slots)


def kernel_accesses(table: str, spec) -> Tuple[str, List[Access]]:
    if spec[0] == "c":
        _, text, name = spec
        if text is None:
            raise ValueError(f"{table}: no text for {name}")
        return c_kernel_accesses(text, name)
    _, path, name, constexpr, helpers = spec
    return name, triton_kernel_accesses(path, name, constexpr, helpers)


# ---------------------------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------------------------

def region_cells(facts: Dict[str, Any]) -> Dict[int, int]:
    """Cells each record line reaches: rows marked on that axis times the other two extents."""
    shape = facts["shape"]
    n = facts["cells"]
    rec = facts["record"]
    out = {}
    for line in range(6):
        axis = line % 3
        a0, a1, b0, b1 = rec[4 * line: 4 * line + 4]
        rows = (a1 - a0) + (b1 - b0)
        out[line] = rows * (n // shape[axis])
    return out


def array_traffic(accesses: List[Access], facts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per array: elements loaded and stored per step (once per kernel), and bytes."""
    n = facts["cells"]
    shape = facts["shape"]
    regions = region_cells(facts)
    per: Dict[str, Dict[str, Any]] = {}
    for acc in accesses:
        e = per.setdefault(acc.array, {"extent": acc.extent, "elem_bytes": acc.elem_bytes,
                                       "load": set(), "store": set()})
        if acc.extent != e["extent"]:
            # a volume indexed once by a coordinate elsewhere is still a volume
            if "volume" in (acc.extent, e["extent"]):
                e["extent"] = "volume"
        e[acc.kind].add(acc.guard)
    out = {}
    for name, e in per.items():
        def count(guards):
            if not guards:
                return 0, "none"
            if None in guards:
                return (n if e["extent"] == "volume" else _small(e["extent"], shape)), "all"
            lines = {g[0] for g in guards}
            if len(lines) > 1:
                raise ValueError(f"{name}: guards on two lines {guards}")
            line = lines.pop()
            pols = {g[1] for g in guards}
            if pols == {True, False}:
                return n, "all"
            cells = regions[line] if True in pols else n - regions[line]
            return cells, ("" if True in pols else "not ") + line_name(line)
        lc, lw = count(e["load"])
        sc, sw = count(e["store"])
        out[name] = {"extent": e["extent"], "elem_bytes": e["elem_bytes"], "load_elems": lc, "load_where": lw,
                     "store_elems": sc, "store_where": sw,
                     "words_per_cell": (lc + sc) / n, "bytes_per_cell": (lc + sc) * e["elem_bytes"] / n}
    return out


def _small(extent: str, shape: Sequence[int]) -> int:
    if extent.startswith("line:"):
        return shape[AXES.index(extent[-1])]
    return 25  # the chunk record, int32[25]


def plan_table(sources, facts) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for table, kernels in sources.items():
        tk: Dict[str, Any] = {}
        for label, spec in kernels.items():
            if spec[0] == "c" and spec[1] is None:
                continue
            name, acc = kernel_accesses(table, spec)
            arrays = array_traffic(acc, facts)
            tk[label] = {"kernel": name, "arrays": arrays,
                         "bytes_per_cell": sum(a["bytes_per_cell"] for a in arrays.values()),
                         "volume_bytes_per_cell": sum(a["bytes_per_cell"] for a in arrays.values()
                                                      if a["extent"] == "volume")}
        plans = {}
        for plan, labels in PLANS.items():
            if all(lb in tk for lb in labels):
                # The distinct elements the plan touches, by parameter name: meaningful where the
                # kernels name the physical arrays (CUDA); Triton and Metal name them by position.
                seen: Dict[str, float] = {}
                for lb in labels:
                    for name, a in tk[lb]["arrays"].items():
                        cells = max(a["load_elems"], a["store_elems"]) * a["elem_bytes"]
                        seen[name] = max(seen.get(name, 0), cells)
                plans[plan] = {"launches": list(labels),
                               "bytes_per_cell": sum(tk[lb]["bytes_per_cell"] for lb in labels),
                               "volume_bytes_per_cell": sum(tk[lb]["volume_bytes_per_cell"] for lb in labels),
                               "working_set_bytes_by_name": sum(seen.values())}
        out[table] = {"kernels": tk, "plans": plans}
    return out


# ---------------------------------------------------------------------------------------------
# MEEP's step loop (MEEP 1.33.0, src/), per chunk of MEEP's own division
# ---------------------------------------------------------------------------------------------

#: The per-loop access sets, transcribed from MEEP 1.33.0 with their sources. ``realnum`` is
#: float32 in the reference builds (--enable-single).
MEEP_LOOPS = {
    "step_curl": "step_db.cpp:47-125 calls STEP_CURL per chunk per B/D component over the chunk; "
                 "step_generic.cpp:65-175 reads g1, g2, reads and writes f, and reads and writes "
                 "fu when the chunk has PML along dsigu = cycle_direction(d_c, 2) (f_u allocated "
                 "there, step_db.cpp:68-72); dsig's sig/kap/siginv are 1-D.",
    "update_EDHB": "update_eh.cpp:143-196: E_c (H_c) is its own array, and its loop runs, only "
                   "where s->chi1inv[c][d_c] exists or the chunk has PML along d_c (dsigw), else it "
                   "aliases D_c (B_c) and nothing runs; step_generic.cpp:566-700 reads g (+ u when "
                   "chi1inv exists, + g1,u1,g2,u2 where the off-diagonal entries exist), and with "
                   "dsigw reads and writes fw and reads and writes f, else writes f.",
    "chi1inv": "anisotropic_averaging.cpp:316-325: an off-diagonal chi1inv entry is deleted where "
               "trivial in the chunk, the diagonal only when the whole row is trivial; mu = 1 "
               "everywhere, so H has no chi1inv (all trivial).",
    "order": "step.cpp:64-120: step_db(B), step_source, step_boundaries, update_eh(H), "
             "step_boundaries, step_db(D), step_source, step_boundaries, update_eh(E), "
             "step_boundaries, fluxes, update_dfts. Boundary exchange (O(chunk surface)), the point "
             "source (O(1)) and the flux plane's DFT (O(plane area)) are not counted.",
    "tiling": "fields.cpp:257-260: loop_tile_base_db 0 (the Simulation default) -> one tile per "
              "chunk: each loop streams its chunk.",
}


def meep_traffic(division: Dict[str, Any], elem_bytes: int = 4) -> Dict[str, Any]:
    """MEEP's compulsory words per cell-step on one chunk division, two ways.

    ``per_loop``: each (chunk, component) loop is a kernel (MEEP's code as written, no tiling):
    a curl partner is read by both components that difference it, a D partner of the
    off-diagonal coupling by each row that couples it. ``per_substep``: each array once per
    MEEP sub-step per chunk, i.e. the three component loops of a chunk sharing their reads --
    the bound a cache that holds a chunk's arrays between consecutive loops reaches. MEEP's own
    DRAM traffic lies between the two (plus write-allocate on its stores).
    ``exchange_estimate``: step_boundaries' copies into each chunk's not-owned layer, estimated
    as one plane per interior high face per component per field type (E, H, B, D), read and
    written once -- an order of magnitude, not a count.
    """
    n = division["cells"]
    loop = {"step_db_B": 0, "update_eh_H": 0, "step_db_D": 0, "update_eh_E": 0}
    sub = dict(loop)
    shape = division["shape"]
    exchange = 0
    for ch in division["chunks"]:
        cells = ch["cells"]
        pml = ch["pml_axes"]
        fu = sum(2 for c in range(3) if pml[(c + 2) % 3])         # fu rw per dsigu chunk
        # STEP_CURL per component: two partners, f rw (+ fu rw)
        loop["step_db_B"] += (3 * (2 + 2) + fu) * cells
        loop["step_db_D"] += (3 * (2 + 2) + fu) * cells
        sub["step_db_B"] += (3 + 3 * 2 + fu) * cells              # each partner array once
        sub["step_db_D"] += (3 + 3 * 2 + fu) * cells
        for c in range(3):
            # update_eh H: mu = 1, so H_c is separate (and stepped) only with dsigw = own axis
            if pml[c]:
                loop["update_eh_H"] += (1 + 2 + 2) * cells       # B_c, f_w rw, H rw
                sub["update_eh_H"] += (1 + 2 + 2) * cells
        d_read = set()
        e_loop = e_sub = 0
        for c in range(3):
            comp = "E" + AXES[c]
            diag = ch["keep_diag"][comp]
            if not (diag or pml[c]):
                continue                                         # E_c aliases D_c: no loop
            partners = [k.split("|")[1] for k, v in ch["keep_offdiag"].items()
                        if k.startswith(comp + "|") and v]
            tail = (2 + 2) if pml[c] else 1                      # f_w rw + E rw, or E written
            e_loop += 1 + (1 if diag else 0) + 2 * len(partners) + tail
            e_sub += (1 if diag else 0) + len(partners) + tail   # the u's and the tail
            d_read |= {"D" + AXES[c]} | {"D" + p[1] for p in partners}
        loop["update_eh_E"] += e_loop * cells
        sub["update_eh_E"] += (e_sub + len(d_read)) * cells
        box = ch["box"]
        for a in range(3):
            if box[a][1] < shape[a]:                             # an interior high face
                face = 1
                for b in range(3):
                    if b != a:
                        face *= box[b][1] - box[b][0]
                exchange += face * 3 * 4 * 2                     # 3 components, E H B D, r+w
    # The distinct elements MEEP's step touches (its working set), per chunk.
    working = 0
    for ch in division["chunks"]:
        pml = ch["pml_axes"]
        arrays = 6                                               # B, D always
        for c in range(3):
            comp = "E" + AXES[c]
            diag = ch["keep_diag"][comp]
            arrays += 2 if pml[(c + 2) % 3] else 0               # fu_B, fu_D
            arrays += 2 if pml[c] else 0                         # H, f_w_H
            if diag or pml[c]:
                arrays += 1                                      # E separate
            arrays += 1 if pml[c] else 0                         # f_w_E
            arrays += 1 if diag else 0                           # chi1inv diagonal
            arrays += sum(1 for k, v in ch["keep_offdiag"].items() if k.startswith(comp + "|") and v)
        working += arrays * ch["cells"]
    per_loop = {k: v / n for k, v in loop.items()}
    per_sub = {k: v / n for k, v in sub.items()}
    return {"words_per_cell": per_loop, "total_words_per_cell": sum(per_loop.values()),
            "bytes_per_cell": sum(per_loop.values()) * elem_bytes,
            "per_substep_words_per_cell": per_sub,
            "per_substep_total_words_per_cell": sum(per_sub.values()),
            "per_substep_bytes_per_cell": sum(per_sub.values()) * elem_bytes,
            "exchange_estimate_bytes_per_cell": exchange * elem_bytes / n,
            "working_set_bytes": working * elem_bytes,
            "chunks": len(division["chunks"])}


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def run(lift_facts: str, meep_chunks: Optional[str]) -> Dict[str, Any]:
    facts_rows = [json.loads(line) for line in open(lift_facts)]
    masks = {row_mask_of(f) for f in facts_rows}
    if len(masks) != 1:
        raise Refused(f"the sizes lift different offdiag row masks {masks}")
    sources = kernel_sources(row_mask=masks.pop())
    out: Dict[str, Any] = {"by_cells": {}, "meep_loops": MEEP_LOOPS}
    for facts in facts_rows:
        out["by_cells"][str(facts["cells"])] = {
            "res": facts["res"], "shape": facts["shape"], "record": facts["record"],
            "region_cells": {line_name(k): v for k, v in region_cells(facts).items()},
            "tables": plan_table(sources, facts)}
    if meep_chunks:
        for line in open(meep_chunks):
            div = json.loads(line)
            key = str(div["cells"])
            out["by_cells"].setdefault(key, {}).setdefault("meep", {})[str(div["num_chunks_requested"])] = \
                meep_traffic(div)
    return out


def text_report(result: Dict[str, Any]) -> str:
    lines = []
    for cells, row in sorted(result["by_cells"].items(), key=lambda kv: int(kv[0])):
        lines.append(f"== {int(cells):,} cells (res {row.get('res')}, shape {row.get('shape')})")
        if "region_cells" in row:
            lines.append("   record line cells: " + ", ".join(f"{k} {v:,}" for k, v in row["region_cells"].items()))
        for table, t in row.get("tables", {}).items():
            for plan, p in t["plans"].items():
                lines.append(f"   {table:7s} {plan:8s} {p['bytes_per_cell']:8.3f} B/cell-step "
                             f"({p['bytes_per_cell'] / 4:.3f} words; volumes {p['volume_bytes_per_cell']:.3f} B) "
                             f"launches {'+'.join(p['launches'])}")
            for label, k in t["kernels"].items():
                lines.append(f"      {label:20s} {k['kernel']:32s} {k['bytes_per_cell']:8.3f} B")
                for name, a in sorted(k["arrays"].items()):
                    lines.append(f"         {name:14s} {a['extent']:8s} load {a['load_where']:12s} "
                                 f"store {a['store_where']:12s} {a['words_per_cell']:.4f} words")
        for n, m in sorted(row.get("meep", {}).items(), key=lambda kv: int(kv[0])):
            lines.append(f"   meep @{n:>3s} ({m['chunks']} chunks) per loop {m['bytes_per_cell']:8.3f} B/cell-step "
                         f"({m['total_words_per_cell']:.3f} words: " +
                         ", ".join(f"{k} {v:.3f}" for k, v in m["words_per_cell"].items()) + ")")
            lines.append(f"   {'':16s} per sub-step {m['per_substep_bytes_per_cell']:8.3f} B/cell-step "
                         f"({m['per_substep_total_words_per_cell']:.3f} words: " +
                         ", ".join(f"{k} {v:.3f}" for k, v in m["per_substep_words_per_cell"].items()) +
                         f"); boundary exchange estimate {m['exchange_estimate_bytes_per_cell']:.2f} B")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lift-facts", required=True)
    ap.add_argument("--meep-chunks")
    ap.add_argument("--json")
    ap.add_argument("--report")
    args = ap.parse_args(argv)
    result = run(args.lift_facts, args.meep_chunks)
    text = text_report(result)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
