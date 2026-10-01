"""The own-cell hoist moves LOADS and nothing else, read off every hoisted kernel's emitted text.

WHAT THE HOIST CLAIMS. Seven device texts on this track issue every own-cell word a thread
reads before the kernel's first store: the fused magnetic pair
(``fused_magnetic_pair_pml_real``, S3d), the two register-view singles
(``update_H_pml_real`` / ``update_E_pml_real``, ``own_cell_hoist.py``) and the Dcyl complex
magnetic and electric pairs on both expansion arms. The claim each module makes is that the
arithmetic, its parenthesisation, the stores and their order are the certified ones and only
the load issue points moved. Until this file that claim rested on design-time instruments
that live under ``results/`` and run by hand: ``singles_design/tools/store_values.py``,
``products_prototype/tools/store_values_sym.py`` and the load-version check of
``emit_census/scripts/verify_arith.py``. This file is the tracked, merge-bar form of the
property those instruments establish.

THE PROPERTY, per kernel, the hoisted text against the certified pre-hoist text:

1. **THE STORE SEQUENCE IS EQUAL.** The entry kernel is executed symbolically and every
   global store is recorded in issue order as (address, predicate, value), the value an
   expression tree over the kernel's loads with one canonical spelling. A dropped, added or
   reordered store, a store moved under a different guard, or a value computed from a
   different word all change this list.
2. **THE LOAD MULTISET IS EQUAL.** Every global load is recorded as (address, version,
   predicate), where the VERSION is the number of this thread's stores to that exact address
   issued before the load. A load moved across a store to its own word changes its version
   (the hazard the hoist must not create); a load moved across anything else does not. An
   extra load, even a dead one whose value no store consumes, changes the multiset.
3. **NO OWN-CELL WORD IS READ AFTER THE FIRST STORE UNLESS THIS LANE WROTE IT.** An own-cell
   load is a load at the thread's own index (``idx``; ``2 * idx`` and ``2 * idx + 1`` for a
   complex word pair) of an array the kernel stores -- the source-level twin of
   ``sass_census/scripts/own_cell_sass.py``'s OWN-CELL LOAD (an LDG, own-cell, of a stored
   array). One issued after the kernel's first store is EXPOSED unless a store by this
   thread to the same address, under a predicate the load's own implies, precedes it: that
   one is a reload of the lane's own write, which the compiler forwards. The hoisted texts
   must carry 0 exposed loads. The certified texts carry 11 (S3d's pair), 5 per single and
   22 per cylindrical text, which is the SASS census's base column for the six Round B texts
   (``cuda_round_b_compose_2026-09-25/compile/own_cell.json``: 5/5/22/22/22/22). The one
   reload is pinned by count rather than hidden: the Dcyl electric pair's m = 0 axis tail
   reads back the ``Dz`` word pair its own lane stored (2 per arm, on both texts).

Clause 3 is what the hoist is FOR and clauses 1-2 are what it must not break.

THE EMITTED SINGLE. The off-diagonal ``update_E`` (``update_E_pml_real_offdiag``) is hoisted
at its launch door: ``offdiag_emitter.offdiag_launch_source`` issues the certified
``offdiag_source`` through ``own_cell_hoist.hoisted_kernel_code``. Its certified text is
emitted per row mask rather than frozen, so the last section of this file asks the three
clauses of every one of its 63 (certified, launch) pairs, requires the inverse to return the
certified text byte for byte, pins the pair the 2026-09-28 kernel screen timed by sha256,
and requires the rule to refuse the folded and dispersive off-diagonal singles by name.
:func:`test_each_clause_catches_its_planted_defect` plants four defects on every hoisted text
and requires each to be caught by the clause named for it -- and the two load defects to be
INVISIBLE to the other load clause, so neither load clause is redundant.

THE REFERENCE IS FROZEN, AND THAT WAS A CHOICE. The certified pre-hoist texts are
``own_cell_hoist_reference.json``, seven line lists cut from the results rounds that emitted
them (each entry names its ``cut_from``), and each is asserted against a sha256 literal
below. The alternative -- re-deriving the pre-hoist text at run time -- is available for the
two singles only: ``own_cell_hoist.unhoisted_kernel_code`` is the inverse every lifter reads,
and it is required here to return the frozen bytes exactly. No function in the tree emits a
pre-hoist PRODUCT text any more (the composers build the hoisted text directly), so a
run-time pre-hoist product would be a second pre-hoist composer written inside this test,
sharing whatever assumption the hoist got wrong and checked against nothing. Reading the
reference from git was ruled out too (once the hoist is committed, HEAD holds the hoisted
text), and so was ``results/``, which is evidence beside the package rather than part of it.
The cost is stated in the JSON's ``_recut``: a certified edit that moves one of these texts
turns this file red until the same edit is applied to the frozen text and the literals are
re-cut -- which coincides with the device re-gate that edit owes.

THE INSTRUMENT is ``store_values_sym.py`` (sha256 ``55dfabd4...``), ported. Changes: its
``Refused`` is an ``AssertionError`` here, so an unmodelled construct is a failure and never
a skip; every load and store is also appended to an ordered trace (address, version,
predicate conjuncts) for clause 3; the SASS census's stored-array rule replaces its
own-word regex count; the CLI and the pairwise report are gone. Its statement forms, helper
inlining, predication, ``cf`` pair handling and load versioning are unchanged, and the
counts it reproduces are pinned in :data:`KNOWN` (the design rounds' figures: stores 78 / 6 /
6 / 44 / 54, loads 120 / 15 / 18 / 62 / 65).

NOTHING HERE COMPILES OR LAUNCHES. The texts are emitted the way the siblings emit them on a
host with no CuPy: the Dcyl pairs directly (``cylindrical_complex_kernels`` imports CuPy only
inside functions), the S3d pair and the singles through the three certified modules loaded
by path under private names with a scoped ``cupy`` stand-in
(``test_fused_magnetic_pair_fill_carry.py``'s discipline).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
import types

import numpy
import pytest

from . import complex_emitter
from . import cylindrical_fused_electric_pair
from . import cylindrical_fused_magnetic_pair
from . import dispersive_offdiag_update_e
from . import folded_offdiag_kernels
from . import fused_magnetic_pair
from . import offdiag_emitter
from . import own_cell_hoist

HERE = pathlib.Path(__file__).parent
REFERENCE = HERE / "own_cell_hoist_reference.json"

#: The certified pre-hoist texts, by sha256. The singles' two are also
#: ``certification.json``'s ``constitutive_2026-08-15`` device pins at the time of the cut
#: (the record is re-cut to the hoisted text when the round that ships the hoist rebinds it,
#: so these are literals rather than reads of the record).
REFERENCE_SHA256 = {
    "fused_magnetic_pair_pml_real":
        "742095b3a1b191f8e196387d8909e1b1702b888dcc7665e9fff2df94cd563400",
    "update_H_pml_real":
        "518b066e839c4419df62bbe61656ac7e25fd3bed843a2604650f3a78ffcfd30b",
    "update_E_pml_real":
        "04ba4648896cb931849acf32ae2f427763f04347ce7edc064854f694c4891b3c",
    "fused_magnetic_pair_pml_cyl_complex[FMA_V1]":
        "032a4828f46f8bb282b96383ee9997a5e004db0ff3222b6b1797cceb10c1f6f0",
    "fused_magnetic_pair_pml_cyl_complex[NAIVE]":
        "7e30dd8aa8b9ee6b998a0da1e0d3728afd6fa6a9e98858fa0b1fb6ee54712e3c",
    "fused_electric_pair_pml_cyl_complex[FMA_V1]":
        "9db74a5e24245d9d72a742791e265b55780813972b350cbcb299e35cc3169102",
    "fused_electric_pair_pml_cyl_complex[NAIVE]":
        "d4a426c205a647fd05b844ef47671a77c1360a5cb7c44d017dc50743e6479830",
}

KEYS = sorted(REFERENCE_SHA256)
SINGLES = ("update_H_pml_real", "update_E_pml_real")

#: KNOWN VALUES, per kernel: global stores, global loads, own-cell loads the hoisted text
#: issues before its first store, exposed own-cell loads after the first store in the
#: CERTIFIED text (clause 3's non-vacuity: the reference really is the unhoisted form), and
#: reloads of the lane's own write after the first store (equal in both texts).
KNOWN = {
    "fused_magnetic_pair_pml_real": dict(stores=78, loads=120, hoisted_before=12,
                                         reference_exposed=11, reloads=0),
    "update_H_pml_real": dict(stores=6, loads=15, hoisted_before=6,
                              reference_exposed=5, reloads=0),
    "update_E_pml_real": dict(stores=6, loads=18, hoisted_before=6,
                              reference_exposed=5, reloads=0),
    "fused_magnetic_pair_pml_cyl_complex[FMA_V1]": dict(stores=44, loads=62, hoisted_before=24,
                                                        reference_exposed=22, reloads=0),
    "fused_magnetic_pair_pml_cyl_complex[NAIVE]": dict(stores=44, loads=62, hoisted_before=24,
                                                       reference_exposed=22, reloads=0),
    "fused_electric_pair_pml_cyl_complex[FMA_V1]": dict(stores=54, loads=65, hoisted_before=24,
                                                        reference_exposed=22, reloads=2),
    "fused_electric_pair_pml_cyl_complex[NAIVE]": dict(stores=54, loads=65, hoisted_before=24,
                                                       reference_exposed=22, reloads=2),
}

#: The product modules that carry the hoist (each defines ``hoisted_loads``), and the
#: register-view marker the hoisted singles' device strings carry.
HOISTED_PRODUCT_MODULES = {"fused_magnetic_pair", "cylindrical_fused_magnetic_pair",
                           "cylindrical_fused_electric_pair"}
REGISTER_VIEW_MARKER = "// THE OWN-CELL HOIST (register view)"


# =============================================================================
# THE INSTRUMENT: one entry kernel, executed symbolically
# (ported from products_prototype/tools/store_values_sym.py, sha256 55dfabd4...)
# =============================================================================

class RefusedText(AssertionError):
    """The executor met a construct it does not model. A failure, never a skip."""


_HEADER = re.compile(r'(?:extern\s+"C"\s+)?(__device__|__global__)\b')
_OWN_INDEX = re.compile(r"(2 \* )?idx( \+ 1)?")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def _match_close(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for position in range(start, len(text)):
        if text[position] == opening:
            depth += 1
        elif text[position] == closing:
            depth -= 1
            if depth == 0:
                return position
    raise RefusedText("unbalanced brackets")


def _split_top(text: str, separator: str = ",") -> list:
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


def _functions(text: str) -> dict:
    out = {}
    for match in _HEADER.finditer(text):
        paren = text.index("(", match.end())
        head = text[match.end():paren].split()
        name = head[-1]
        returns = head[-2] if len(head) >= 2 else "void"
        close = _match_close(text, paren, "(", ")")
        if not text[close + 1:].lstrip().startswith("{"):
            continue
        body_open = text.index("{", close)
        body_close = _match_close(text, body_open, "{", "}")
        params = []
        for declaration in _split_top(text[paren + 1:close]):
            identifier = re.findall(r"[A-Za-z_]\w*", declaration)[-1]
            kind = ("ptr" if "*" in declaration
                    else "cf" if declaration.split()[0] == "cf" else "scalar")
            params.append((identifier, kind))
        out[name] = {"kind": match.group(1), "ret": returns, "params": params,
                     "body": text[body_open + 1:body_close]}
    return out


def _items(body: str) -> list:
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
    trailing = " ".join("".join(current).split())
    if trailing:
        raise RefusedText(f"trailing text without a terminator: {trailing!r}")
    return out


def _find_ternary(expression: str):
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


def _c_to_py(expression: str) -> str:
    expression = re.sub(r"(\d+\.\d*|\.\d+|\d+)f\b", r"\1", expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return expression.strip()


def _unp(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_unp(v)}" for k, v in sorted(value.items())) + "}"
    return ast.unparse(value)


def _same(a, b) -> bool:
    """Structural equality of two expression trees (``ctx`` ignored) -- what comparing their
    ``_unp`` spellings decides on these parse-normal trees, without building either string.
    The instrument compared strings; on a nested predicated tree that is quadratic, and it
    is where the port spent 9.4 of 15.9 profiled seconds on the Dcyl electric text."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, ast.AST):
        return all(field == "ctx" or _same(getattr(a, field, None), getattr(b, field, None))
                   for field in a._fields)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _conjuncts(tree) -> list:
    if isinstance(tree, ast.BoolOp) and isinstance(tree.op, ast.And):
        out = []
        for value in tree.values:
            out += _conjuncts(value)
        return out
    return [tree]


def _negate(tree):
    if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.Not):
        return tree.operand
    return ast.UnaryOp(op=ast.Not(), operand=tree)


def _simplify(value, facts):
    """Select the branch of every IfExp whose test (or its negation) is a known fact."""
    if isinstance(value, dict):
        return {k: _simplify(v, facts) for k, v in value.items()}
    if isinstance(value, ast.IfExp):
        test = _simplify(value.test, facts)
        spelled = _unp(test)
        if spelled in facts["true"]:
            return _simplify(value.body, facts)
        if spelled in facts["false"]:
            return _simplify(value.orelse, facts)
        body, orelse = _simplify(value.body, facts), _simplify(value.orelse, facts)
        if _same(body, orelse):
            return body
        return ast.IfExp(test=test, body=body, orelse=orelse)
    if isinstance(value, ast.AST):
        for field, child in ast.iter_fields(value):
            if isinstance(child, ast.AST):
                setattr(value, field, _simplify(child, facts))
            elif isinstance(child, list):
                setattr(value, field, [_simplify(c, facts) if isinstance(c, ast.AST) else c
                                       for c in child])
    return value


def _copy(value):
    """A private copy of a value, node by node. The instrument round-tripped every copy
    through ``ast.unparse`` and ``ast.parse``, which re-spells the whole tree; the trees
    here are already parse-normal, so a structural copy is the same tree for a fraction
    of the cost."""
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    if isinstance(value, ast.AST):
        node = value.__class__()
        for field in value._fields:
            setattr(node, field, _copy(getattr(value, field, None)))
        return node
    return value


def _merge(condition, new, old):
    if isinstance(new, dict) or isinstance(old, dict):
        if not (isinstance(new, dict) and isinstance(old, dict)):
            raise RefusedText("predicated assignment mixes a cf and a scalar")
        return {k: _merge(condition, new[k], old[k]) for k in new}
    if _same(new, old):
        return new
    return ast.IfExp(test=_copy(condition), body=new, orelse=old)


class Machine:
    """Execute a text's one ``__global__`` entry, statement by statement, over symbols.

    * a local holds an expression tree, or for a ``cf`` a pair ``{re, im}``;
    * ``A[e]`` of a kernel pointer reads ``LOAD_<A>_v<n>[e]``, ``n`` the stores to that exact
      address issued before it; a pointer bound to ``&local`` (index 0) is that local;
    * a helper that stores, or is straight-line, is INLINED; one that stores nothing and
      branches on data (``shift_up``) stays an opaque call, and an array handed to one must
      be an array the kernel never stores;
    * ``if (c) {...}``, ``if (c) stmt;`` and ``c ? a : b`` run PREDICATED: an assignment
      becomes ``x = (v if c else x)``, a store and a load carry the conjunction of the
      enclosing conditions; a helper's ``if (c) return v;`` predicates the rest of it.
    """

    def __init__(self, text: str):
        self.text = _strip_comments(text)
        self.fns = _functions(self.text)
        entries = [name for name, f in self.fns.items() if f["kind"] == "__global__"]
        if len(entries) != 1:
            raise RefusedText(f"{len(entries)} entry kernels, not one")
        self.entry = entries[0]
        self.arrays = {p for p, kind in self.fns[self.entry]["params"] if kind == "ptr"}
        self.stores_in = {name: bool(re.search(r"(?m)^\s*\w+\[[^\]]+\]\s*=(?!=)", f["body"]))
                          for name, f in self.fns.items()}
        changed = True
        while changed:  # a helper that calls a storing helper stores too
            changed = False
            for name, f in self.fns.items():
                if not self.stores_in[name] and any(
                        self.stores_in[other] and re.search(rf"\b{other}\(", f["body"])
                        for other in self.fns if other != name):
                    self.stores_in[name] = changed = True
        self.stored_arrays = set()
        self.frames = [{}]
        self.preds = []
        self.stores = []
        self.store_count = {}
        self.loads = []
        self.trace = []
        self.int_decls = []
        self.guards = []
        self.opaque_reads = []

    # -- predicate -------------------------------------------------------------------
    def facts(self) -> dict:
        true, false = set(), set()
        for predicate in self.preds:
            for conjunct in _conjuncts(predicate):
                true.add(_unp(conjunct))
                false.add(_unp(_negate(conjunct)))
                if isinstance(conjunct, ast.UnaryOp) and isinstance(conjunct.op, ast.Not):
                    false.add(_unp(conjunct.operand))
        return {"true": true, "false": false}

    def pred_text(self) -> str:
        return " and ".join(f"({_unp(p)})" for p in self.preds) or "True"

    def pred_set(self) -> frozenset:
        return frozenset(_unp(c) for p in self.preds for c in _conjuncts(p))

    # -- environment -----------------------------------------------------------------
    def lookup(self, name):
        for frame in reversed(self.frames):
            if name in frame:
                return frame[name]
        return None

    def declare(self, name, value):
        self.frames[-1][name] = value

    def assign(self, name, value):
        for frame in reversed(self.frames):
            if name in frame:
                if self.preds:
                    condition = (ast.BoolOp(op=ast.And(), values=[_copy(p) for p in self.preds])
                                 if len(self.preds) > 1 else self.preds[0])
                    value = _merge(condition, value, frame[name])
                frame[name] = _simplify(value, self.facts())
                return
        raise RefusedText(f"assignment to undeclared {name}")

    # -- expressions -----------------------------------------------------------------
    def expr(self, text: str, binding):
        ternary = _find_ternary(text)
        if ternary:
            condition_text, when_true, when_false = ternary
            condition = self.expr(condition_text, binding)
            self.preds.append(condition)
            true_value = self.expr(when_true, binding)
            self.preds[-1] = _negate(_copy(condition))
            false_value = self.expr(when_false, binding)
            self.preds.pop()
            if isinstance(true_value, dict) != isinstance(false_value, dict):
                raise RefusedText(f"ternary mixes a cf and a scalar: {text!r}")
            if isinstance(true_value, dict):
                return _simplify({k: ast.IfExp(test=_copy(condition), body=true_value[k],
                                               orelse=false_value[k]) for k in ("re", "im")},
                                 self.facts())
            return _simplify(ast.IfExp(test=condition, body=true_value, orelse=false_value),
                             self.facts())
        return self.eval(ast.parse(_c_to_py(text), mode="eval").body, binding)

    def resolve_pointer(self, name, binding):
        if name in binding and binding[name][0] in ("array", "ref"):
            return binding[name]
        if name in self.arrays:
            return ("array", name)
        return None

    def eval(self, node, binding):
        if isinstance(node, ast.Name):
            if node.id in binding and binding[node.id][0] == "value":
                return _copy(binding[node.id][1])
            value = self.lookup(node.id)
            return _copy(value) if value is not None else node
        if isinstance(node, ast.Attribute):
            base = self.eval(node.value, binding)
            if isinstance(base, dict):
                return _copy(base[node.attr])
            return ast.Attribute(value=base, attr=node.attr, ctx=ast.Load())
        if isinstance(node, ast.Subscript):
            name = node.value.id if isinstance(node.value, ast.Name) else None
            index = self.eval(node.slice, binding)
            target = self.resolve_pointer(name, binding)
            if target is None:
                raise RefusedText(f"subscript of a non-pointer {name}")
            if target[0] == "ref":
                if _unp(index) != "0":
                    raise RefusedText("register view indexed at a non-zero index")
                return _copy(self.lookup(target[1]))
            address = f"{target[1]}[{_unp(index)}]"
            version = self.store_count.get(address, 0)
            self.loads.append((address, version, self.pred_text()))
            self.trace.append(("load", target[1], _unp(index), version, self.pred_set()))
            return ast.Subscript(value=ast.Name(id=f"LOAD_{target[1]}_v{version}"),
                                 slice=index, ctx=ast.Load())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in self.fns:
                return self.call(node.func.id, [ast.unparse(a) for a in node.args], binding,
                                 from_expr=True)
            args = [self.eval(a, binding) for a in node.args]
            args = [ast.Tuple(elts=[a["re"], a["im"]], ctx=ast.Load()) if isinstance(a, dict)
                    else a for a in args]
            return ast.Call(func=ast.Name(id=node.func.id), args=args, keywords=[])
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp)):
            for field, child in ast.iter_fields(node):
                if isinstance(child, ast.AST):
                    result = self.eval(child, binding)
                    if isinstance(result, dict):
                        raise RefusedText(f"arithmetic on a cf value in {ast.unparse(node)}")
                    setattr(node, field, result)
                elif isinstance(child, list):
                    new = []
                    for item in child:
                        result = self.eval(item, binding) if isinstance(item, ast.AST) else item
                        if isinstance(result, dict):
                            raise RefusedText(
                                f"arithmetic on a cf value in {ast.unparse(node)}")
                        new.append(result)
                    setattr(node, field, new)
            return node
        return node

    # -- calls -----------------------------------------------------------------------
    def opaque(self, name) -> bool:
        body = self.fns[name]["body"]
        return not self.stores_in[name] and ("if (" in body or "?" in body)

    def call(self, name, args, binding, from_expr=False):
        f = self.fns[name]
        if len(args) != len(f["params"]):
            raise RefusedText(f"{name}: {len(args)} args for {len(f['params'])} params")
        if self.opaque(name):
            values = []
            for (param, kind), arg in zip(f["params"], args):
                if kind == "ptr":
                    target = self.resolve_pointer(arg.lstrip("&").strip(), binding)
                    if target is None or target[0] != "array":
                        raise RefusedText(f"opaque {name} handed a non-array pointer {arg}")
                    self.opaque_reads.append((name, target[1]))
                    if target[1] in self.stored_arrays_all():
                        raise RefusedText(f"opaque {name} reads {target[1]}, which the "
                                          f"kernel stores")
                    values.append(ast.Name(id=target[1]))
                else:
                    value = self.expr(arg, binding)
                    values.append(ast.Tuple(elts=[value["re"], value["im"]], ctx=ast.Load())
                                  if isinstance(value, dict) else value)
            node = ast.Call(func=ast.Name(id=name), args=values, keywords=[])
            if f["ret"] == "cf":
                return {"re": ast.Attribute(value=_copy(node), attr="re", ctx=ast.Load()),
                        "im": ast.Attribute(value=_copy(node), attr="im", ctx=ast.Load())}
            return node
        inner = {}
        for (param, kind), arg in zip(f["params"], args):
            if kind == "ptr":
                if arg.startswith("&"):
                    inner[param] = ("ref", arg[1:].strip())
                else:
                    target = self.resolve_pointer(arg, binding)
                    if target is None:
                        raise RefusedText(f"{name}: pointer argument {arg} is not a kernel "
                                          f"array")
                    inner[param] = target
            else:
                inner[param] = ("value", self.expr(arg, binding))
        self.frames.append({})
        depth = len(self.preds)
        returns = []
        self.run(_items(f["body"]), inner, helper_returns=returns)
        final = self.frames[-1].get("__return__")
        del self.preds[depth:]
        self.frames.pop()
        value = final
        for condition, returned in reversed(returns):  # fold the early returns
            if value is None:
                value = returned
            elif isinstance(returned, dict):
                value = {k: ast.IfExp(test=_copy(condition), body=returned[k], orelse=value[k])
                         for k in returned}
            else:
                value = ast.IfExp(test=_copy(condition), body=returned, orelse=value)
        if from_expr and value is None:
            raise RefusedText(f"void helper {name} used as a value")
        return _simplify(value, self.facts()) if value is not None else None

    def stored_arrays_all(self):
        if not self.stored_arrays:
            stored = set()
            for name, f in self.fns.items():
                for match in re.finditer(r"(?m)^\s*(\w+)\[[^\]]+\]\s*=(?!=)", f["body"]):
                    stored.add(match.group(1))
            for name, f in self.fns.items():
                if not self.stores_in[name]:
                    continue
                for call in re.finditer(rf"\b{name}\(([^;]*?)\)\s*;", self.text):
                    for (param, kind), arg in zip(f["params"], _split_top(call.group(1))):
                        if kind == "ptr":
                            stored.add(arg.lstrip("&").strip())
            self.stored_arrays = stored & self.arrays or {"<none>"}
        return self.stored_arrays

    # -- statements ------------------------------------------------------------------
    def run(self, statements, binding, helper_returns=None):
        position = 0
        while position < len(statements):
            statement = statements[position]
            if statement.endswith("{"):
                depth = 0
                for end in range(position, len(statements)):
                    if statements[end].endswith("{"):
                        depth += 1
                    elif statements[end] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                head = statement[:-1].strip()
                if end + 1 < len(statements) and statements[end + 1].startswith("else"):
                    raise RefusedText("else branches are not modelled")
                self.frames.append({})
                if head:
                    match = re.fullmatch(r"if \((.+)\)", head)
                    if not match:
                        raise RefusedText(f"unsupported block head {head!r}")
                    self.preds.append(_simplify(self.expr(match.group(1), binding),
                                                self.facts()))
                    self.run(statements[position + 1:end], binding, helper_returns)
                    self.preds.pop()
                else:
                    self.run(statements[position + 1:end], binding, helper_returns)
                self.frames.pop()
                position = end + 1
                continue
            self.statement(statement, binding, helper_returns)
            position += 1

    def store(self, array, index_text, value, binding):
        target = self.resolve_pointer(array, binding)
        if target is None:
            raise RefusedText(f"store to a non-pointer {array}")
        index = self.expr(index_text, binding)
        value = _simplify(value, self.facts())
        if target[0] == "ref":
            if _unp(index) != "0":
                raise RefusedText("register view stored at a non-zero index")
            self.assign(target[1], value)
            return
        address = f"{target[1]}[{_unp(index)}]"
        self.stores.append((address, self.pred_text(), _unp(value)))
        self.store_count[address] = self.store_count.get(address, 0) + 1
        self.trace.append(("store", target[1], _unp(index), None, self.pred_set()))

    def statement(self, statement, binding, helper_returns):
        if statement.startswith("if ("):
            close = _match_close(statement, 3, "(", ")")
            condition = _simplify(self.expr(statement[4:close], binding), self.facts())
            rest = statement[close + 1:].strip()
            match = re.fullmatch(r"return( .+)?;", rest)
            if match:
                if helper_returns is None:
                    if match.group(1):
                        raise RefusedText("value return in the kernel")
                    self.guards.append(_unp(condition))
                    return
                value = self.expr(match.group(1).strip(), binding) if match.group(1) else None
                helper_returns.append((condition, value))
                self.preds.append(_negate(condition))
                return
            self.preds.append(condition)
            self.statement(rest, binding, helper_returns)
            self.preds.pop()
            return
        match = re.match(r"^(float|int|const int|cf) (.+);$", statement)
        if match and "(" not in match.group(2).split("=")[0] \
                and len(_split_top(match.group(2))) > 1:
            for part in _split_top(match.group(2)):
                self.statement(f"{match.group(1)} {part};", binding, helper_returns)
            return
        match = re.match(r"^(?:const )?int (\w+) = (.+);$", statement)
        if match:
            self.int_decls.append(statement)
            self.declare(match.group(1), ast.Name(id=match.group(1)))
            return
        match = re.match(r"^(float|cf) (\w+);$", statement)
        if match:
            name = match.group(2)
            self.declare(name, {"re": ast.Name(id=f"UNINIT_{name}_re"),
                                "im": ast.Name(id=f"UNINIT_{name}_im")}
                         if match.group(1) == "cf" else ast.Name(id=f"UNINIT_{name}"))
            return
        match = re.match(r"^(float|cf) (\w+) = (.+);$", statement)
        if match:
            value = self.expr(match.group(3), binding)
            if (match.group(1) == "cf") != isinstance(value, dict):
                raise RefusedText(f"type mismatch in {statement!r}")
            self.declare(match.group(2), _simplify(value, self.facts()))
            return
        match = re.match(r"^(\w+)\.(re|im) = (.+);$", statement)
        if match:
            current = self.lookup(match.group(1))
            if not isinstance(current, dict):
                raise RefusedText(f"field store to a non-cf {match.group(1)}")
            new = dict(current)
            new[match.group(2)] = self.expr(match.group(3), binding)
            self.assign(match.group(1), new)
            return
        match = re.match(r"^(\w+)\[(.+)\] = (.+);$", statement)
        if match:
            self.store(match.group(1), match.group(2), self.expr(match.group(3), binding),
                       binding)
            return
        match = re.match(r"^(\w+)\((.*)\);$", statement)
        if match and match.group(1) in self.fns:
            self.call(match.group(1), _split_top(match.group(2)), binding)
            return
        match = re.match(r"^(\w+) = (.+);$", statement)
        if match:
            self.assign(match.group(1), self.expr(match.group(2), binding))
            return
        match = re.match(r"^return (.+);$", statement)
        if match:
            if helper_returns is None:
                raise RefusedText("value return in the kernel")
            self.frames[-1]["__return__"] = self.expr(match.group(1), binding)
            return
        raise RefusedText(f"unsupported statement {statement!r}")

    # -- the reading -----------------------------------------------------------------
    def execute(self) -> dict:
        self.stored_arrays_all()
        self.run(_items(self.fns[self.entry]["body"]), {})
        stored = {array for kind, array, *_ in self.trace if kind == "store"}
        first = next((n for n, event in enumerate(self.trace) if event[0] == "store"),
                     len(self.trace))
        own_before, exposed, reloads = 0, [], []
        written = {}
        for position, (kind, array, index, _version, predicate) in enumerate(self.trace):
            address = f"{array}[{index}]"
            if kind == "store":
                written.setdefault(address, []).append(predicate)
                continue
            if array not in stored or not _OWN_INDEX.fullmatch(index):
                continue
            if position < first:
                own_before += 1
            elif any(prior <= predicate for prior in written.get(address, ())):
                reloads.append(address)
            else:
                exposed.append(address)
        return {"entry": self.entry, "stores": self.stores, "loads": sorted(self.loads),
                "int_decls": self.int_decls, "guards": self.guards,
                "opaque_reads": sorted(set(self.opaque_reads)),
                "own_before_first_store": own_before,
                "exposed_after_first_store": exposed,
                "reloads_after_first_store": reloads}


def read(text: str) -> dict:
    return Machine(text).execute()


# =============================================================================
# THE TEXTS: the frozen certified references, and the tree's live hoisted emission
# =============================================================================

def _joined(entry: dict) -> str:
    return "\n".join(entry["lines"])


@pytest.fixture(scope="module")
def reference() -> dict:
    document = json.loads(REFERENCE.read_text(encoding="ascii"))
    return {key: _joined(entry) for key, entry in document["texts"].items()}


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in."""

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


def _certified_by_path() -> dict:
    """The three certified modules the S3d pair and the singles need, loaded by path under
    private names with ``cupy`` stubbed only for the duration of the load."""
    previous = sys.modules.get("cupy")
    sys.modules["cupy"] = _NumpyWearingCupysName()
    loaded = {}
    try:
        for stem in ("step_curl_kernels", "constitutive_kernels", "in_seam_passes"):
            spec = importlib.util.spec_from_file_location(
                f"_own_cell_hoist_certified_{stem}", HERE / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[stem] = module
    finally:
        if previous is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous
    return loaded


class _PerKey(dict):
    """A mapping that builds each value on first use. A refusal while building one kernel's
    value fails that kernel's rows and leaves the other six to report for themselves."""

    def __init__(self, make):
        super().__init__()
        self._make = make

    def __missing__(self, key):
        value = self[key] = self._make(key)
        return value


#: Each Dcyl pair's entry name -> its module (``device_sources()`` is keyed by arm).
_CYLINDRICAL = {"fused_magnetic_pair_pml_cyl_complex": cylindrical_fused_magnetic_pair,
                "fused_electric_pair_pml_cyl_complex": cylindrical_fused_electric_pair}


@pytest.fixture(scope="module")
def certified() -> dict:
    return _certified_by_path()


@pytest.fixture(scope="module")
def hoisted(certified) -> dict:
    """Every hoisted text, as the tree's own emitters hand it to the compiler."""

    def emit(key: str) -> str:
        if key in SINGLES:  # the strings constitutive_kernels._get_kernel compiles
            return getattr(certified["constitutive_kernels"], f"_{key}_kernel_code")
        if key == "fused_magnetic_pair_pml_real":
            family = fused_magnetic_pair
            held = {name: getattr(family, name) for name in certified}
            for name, module in certified.items():
                setattr(family, name, module)
            try:
                return family.device_sources()[key]
            finally:
                for name, module in held.items():
                    setattr(family, name, module)
        kernel, arm = key[:-1].split("[")
        return _CYLINDRICAL[kernel].device_sources()[arm]

    return _PerKey(emit)


@pytest.fixture(scope="module")
def readings(reference, hoisted) -> dict:
    return _PerKey(lambda key: (read(reference[key]), read(hoisted[key])))


# =============================================================================
# THE REFERENCE AND THE ENUMERATION
# =============================================================================

def test_the_frozen_references_are_the_certified_pre_hoist_texts(reference):
    document = json.loads(REFERENCE.read_text(encoding="ascii"))
    assert sorted(document["texts"]) == KEYS
    for key in KEYS:
        entry = document["texts"][key]
        digest = hashlib.sha256(reference[key].encode("utf-8")).hexdigest()
        assert digest == REFERENCE_SHA256[key] == entry["sha256"], key
        assert f'extern "C" __global__ void {entry["kernel"]}(' in reference[key], key
        assert entry["cut_from"].startswith("apps/api/parity/meep_gpu/results/"), key


def test_every_hoisted_text_in_the_package_is_in_this_table(certified, hoisted):
    """A hoist added to another kernel without a row here fails by name."""
    products = {path.stem for path in HERE.glob("*.py") if not path.stem.startswith("test_")
                and re.search(r"(?m)^def hoisted_loads\(", path.read_text(encoding="utf-8"))}
    assert products == HOISTED_PRODUCT_MODULES
    singles = set()
    for module in certified.values():
        for name, value in vars(module).items():
            if name.endswith("_kernel_code") and isinstance(value, str) \
                    and REGISTER_VIEW_MARKER in value:
                singles.add(name[1:-len("_kernel_code")])
    assert singles == set(SINGLES)
    for kernel, module in _CYLINDRICAL.items():
        assert sorted(module.device_sources()) == sorted(complex_emitter.EXPANSIONS), kernel
    emitted = sorted(["fused_magnetic_pair_pml_real", *SINGLES]
                     + [f"{kernel}[{arm}]" for kernel in _CYLINDRICAL
                        for arm in complex_emitter.EXPANSIONS])
    assert emitted == KEYS
    for key in KEYS:
        kernel = key.split("[")[0]
        assert hoisted[key].count('extern "C" __global__ void ') == 1, key
        assert f'extern "C" __global__ void {kernel}(' in hoisted[key], key


@pytest.mark.parametrize("kernel", SINGLES)
def test_the_singles_inverse_returns_the_frozen_reference_byte_for_byte(kernel, reference,
                                                                        hoisted):
    """What every lifter reads is the certified text, so the frozen one is not a guess."""
    assert own_cell_hoist.unhoisted_kernel_code(
        hoisted[kernel], kernel, f"_{kernel}_kernel_code") == reference[kernel]


# =============================================================================
# THE THREE CLAUSES
# =============================================================================

@pytest.mark.parametrize("key", KEYS)
def test_the_hoist_keeps_the_store_sequence(key, readings):
    before, after = readings[key]
    assert before["entry"] == after["entry"]
    assert len(after["stores"]) == KNOWN[key]["stores"]
    first = next((n for n, (x, y) in enumerate(zip(before["stores"], after["stores"]))
                  if x != y), None)
    assert after["stores"] == before["stores"], (
        f"{key}: stores {len(before['stores'])} -> {len(after['stores'])}, first difference "
        f"at {first}: {before['stores'][first] if first is not None else None!r} -> "
        f"{after['stores'][first] if first is not None else None!r}")
    assert after["int_decls"] == before["int_decls"]
    assert after["guards"] == before["guards"]
    assert after["opaque_reads"] == before["opaque_reads"]


@pytest.mark.parametrize("key", KEYS)
def test_the_hoist_keeps_the_load_multiset(key, readings):
    before, after = readings[key]
    assert len(after["loads"]) == KNOWN[key]["loads"]
    only_before = [x for x in before["loads"] if x not in after["loads"]]
    only_after = [x for x in after["loads"] if x not in before["loads"]]
    assert after["loads"] == before["loads"], (
        f"{key}: loads {len(before['loads'])} -> {len(after['loads'])}; certified only "
        f"{only_before[:6]!r}; hoisted only {only_after[:6]!r}")


@pytest.mark.parametrize("key", KEYS)
def test_no_own_cell_word_is_read_after_the_first_store_unless_this_lane_wrote_it(
        key, readings):
    before, after = readings[key]
    assert after["exposed_after_first_store"] == [], (
        f"{key}: {len(after['exposed_after_first_store'])} own-cell loads issued after the "
        f"first store read a word this lane has not written: "
        f"{after['exposed_after_first_store']!r}")
    assert after["own_before_first_store"] == KNOWN[key]["hoisted_before"]
    # Non-vacuity: the certified text really is the unhoisted form.
    assert len(before["exposed_after_first_store"]) == KNOWN[key]["reference_exposed"]
    # The one exclusion, pinned by count in both texts rather than hidden.
    assert len(after["reloads_after_first_store"]) == KNOWN[key]["reloads"]
    assert after["reloads_after_first_store"] == before["reloads_after_first_store"]


# =============================================================================
# THE CONTROLS: one planted defect per clause, on every hoisted text
# =============================================================================

def _once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"planting {what}: anchor found {text.count(old)} times, not once")
    return text.replace(old, new, 1)


def _in_helper(text: str, helper: str, edit) -> str:
    """Apply ``edit`` to one ``__device__`` helper's definition, found exactly once."""
    starts = [m.start() for m in re.finditer(
        rf"__device__ __forceinline__ \w+ {helper}\(\n", text)]
    if len(starts) != 1:
        raise AssertionError(f"{helper} is defined {len(starts)} times, not once")
    end = text.index("\n}\n", starts[0]) + len("\n}\n")
    return text[:starts[0]] + edit(text[starts[0]:end]) + text[end:]


def _line_naming(text: str, needle: str, what: str) -> str:
    lines = [line for line in text.splitlines(keepends=True) if needle in line]
    if len(lines) != 1:
        raise AssertionError(f"planting {what}: {len(lines)} lines carry {needle!r}")
    return lines[0]


def _plant_single(text: str, kernel: str, mutation: str) -> str:
    helper, field, aux, freg, areg, _source, indent = own_cell_hoist.KERNELS[kernel]
    back = [line + "\n" for line in own_cell_hoist.writeback_block(kernel)]
    if mutation == "drop_a_store":
        return _once(text, back[0], "", mutation)
    if mutation == "reorder_two_stores":
        return _once(text, back[0] + back[1], back[1] + back[0], mutation)
    if mutation == "add_a_load_after_a_store":
        return _once(text, back[-1], back[-1] + f"    float again = {field}x[idx];\n", mutation)
    # sink_a_hoisted_load: store x's auxiliary right after its call, and issue z's
    # auxiliary load right before its call -- after that store.
    text = _once(text, back[0], "", mutation)
    call_x = _line_naming(text, f"{helper}(&{freg}_x, &{areg}_x, 0, ", mutation)
    text = _once(text, call_x, call_x + back[0], mutation)
    load_z = f"    float {areg}_z = {aux}z[idx];\n"
    text = _once(text, load_z, "", mutation)
    call_z = _line_naming(text, f"{helper}(&{freg}_z, &{areg}_z, 0, ", mutation)
    return _once(text, call_z, load_z + call_z, mutation)


def _plant_product(text: str, family, mutation: str) -> str:
    complex_words = "cf_store(" in text
    aux_store = "    cf_store(fw, idx, src);\n" if complex_words else "    fw[idx] = src;\n"
    field_store = ("    cf_store(f, idx, a);\n" if complex_words
                   else "    f[idx] = a - kms * prev;\n")
    again = ("    cf again = cf_load(fw, idx);\n" if complex_words
             else "    float again = fw[idx];\n")
    if mutation == "drop_a_store":
        return _in_helper(text, "constitutive_apply_pre",
                          lambda h: _once(h, aux_store, "", mutation))
    if mutation == "reorder_two_stores":
        return _in_helper(text, "constitutive_apply_pre", lambda h: _once(
            _once(h, aux_store, "", mutation), field_store, field_store + aux_store, mutation))
    if mutation == "add_a_load_after_a_store":
        return _in_helper(text, "constitutive_apply_pre",
                          lambda h: _once(h, aux_store, aux_store + again, mutation))
    # sink_a_hoisted_load: the hoist's last load, issued right before its consumer, which
    # runs after the first component's stores.
    load = family.hoisted_loads().splitlines()[-1] + "\n"
    register = re.match(r"\s*(?:float|cf) (pre_\w+) =", load).group(1)
    text = _once(text, load, "", mutation)
    consumer = _line_naming(text, f", {register});", mutation)
    indent = consumer[:len(consumer) - len(consumer.lstrip())]
    return _once(text, consumer, indent + load.lstrip() + consumer, mutation)


_FAMILY = {"fused_magnetic_pair_pml_real": fused_magnetic_pair,
           "fused_magnetic_pair_pml_cyl_complex": cylindrical_fused_magnetic_pair,
           "fused_electric_pair_pml_cyl_complex": cylindrical_fused_electric_pair}
MUTATIONS = ("drop_a_store", "reorder_two_stores", "add_a_load_after_a_store",
             "sink_a_hoisted_load")


def plant(key: str, text: str, mutation: str) -> str:
    if key in SINGLES:
        return _plant_single(text, key, mutation)
    return _plant_product(text, _FAMILY[key.split("[")[0]], mutation)


@pytest.mark.parametrize("mutation", MUTATIONS)
@pytest.mark.parametrize("key", KEYS)
def test_each_clause_catches_its_planted_defect(key, mutation, readings, hoisted):
    """Each clause earns its place: every planted defect is caught by the clause named for
    it, and the two load defects are each INVISIBLE to the other load clause."""
    before, _ = readings[key]
    planted = plant(key, hoisted[key], mutation)
    assert planted != hoisted[key]
    after = read(planted)
    stores_equal = after["stores"] == before["stores"]
    loads_equal = after["loads"] == before["loads"]
    exposed = after["exposed_after_first_store"]
    if mutation in ("drop_a_store", "reorder_two_stores"):
        assert not stores_equal, f"{key}: {mutation} left the store sequence equal"
        if mutation == "reorder_two_stores" and key in SINGLES:
            # The lifters' inverse refuses the write-back out of the certified order.
            with pytest.raises(AssertionError, match="write-back"):
                own_cell_hoist.unhoisted_kernel_code(planted, key, key)
    elif mutation == "add_a_load_after_a_store":
        assert not loads_equal, f"{key}: an added load left the load multiset equal"
        assert stores_equal
        assert exposed == [], "a reload of the lane's own write is not an exposed load"
    else:
        assert exposed, f"{key}: a hoisted load sunk past a store was not caught"
        assert stores_equal and loads_equal, (
            f"{key}: the sunk load also moved the stores or the multiset, so clause 3 is "
            f"not what this control isolates")


# =============================================================================
# THE EMITTED SINGLE: the off-diagonal update_E ('cuda:off-diagonal')
# =============================================================================
#
# ``offdiag_source(mask)`` is the certified text (the emitter's corpus digest in
# ``certification.json``'s ``offdiag_2026-08-16`` binds all 63 of them), so it is read here
# rather than frozen, and ``offdiag_launch_source(mask)`` is what NVRTC compiles. Reading
# all 63 pairs with the instrument takes about 1.3 s.

OFFDIAG = offdiag_emitter.KERNEL_NAME

#: The pair the 2026-09-28 kernel screen compiled and timed on ``pml_3d`` (all six rows
#: live): 0 differing words at 512,000, 2,097,152 and 7,077,888 cells. A change to either
#: text is a change to what was measured, and owes the device a new screen and the family
#: gate a new run.
OFFDIAG_SCREENED_MASK = (1, 1, 1, 1, 1, 1)
OFFDIAG_SCREENED_SHA256 = {
    "certified": "3dc04cbe7d958e2adce638a22ab08d547757e2c10a18cc8f3b9ed64b7de3f42c",
    "launch": "45d2bbc49c8c2846c246e934c1ec6831c3515592b5f4f0a860b021a76c4441d8",
}

#: The row masks ``gate_cuda_offdiag.GATE_ROW_MASKS`` sweeps; the planted controls run here.
OFFDIAG_GATE_MASKS = ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0),
                      (0, 0, 1, 1, 0, 0))

#: The table the rule carries, and the one module that issues a hoist through its forward.
RULE_TABLE = {"update_H_pml_real", "update_E_pml_real", "step_B_pml_real",
              "step_D_pml_real", OFFDIAG}
EMITTED_HOIST_MODULES = {"offdiag_emitter"}


def _mask_id(mask) -> str:
    return "".join(str(flag) for flag in mask)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def offdiag_readings() -> dict:
    return _PerKey(lambda mask: (read(offdiag_emitter.offdiag_source(mask)),
                                 read(offdiag_emitter.offdiag_launch_source(mask))))


def test_the_off_diagonal_row_is_the_plain_update_E_row(certified):
    """Same helper byte for byte, same prefixes, registers and call form."""
    assert own_cell_hoist.KERNELS[OFFDIAG] == own_cell_hoist.KERNELS["update_E_pml_real"]
    shipped = certified["constitutive_kernels"]._update_E_pml_real_kernel_code
    emitted = offdiag_emitter.offdiag_source(OFFDIAG_SCREENED_MASK)
    assert (own_cell_hoist._helper_text(emitted, "constitutive_apply")  # noqa: SLF001
            == own_cell_hoist._helper_text(shipped, "constitutive_apply"))  # noqa: SLF001


def test_the_screened_pair_is_the_pair_the_emitter_issues():
    certified = offdiag_emitter.offdiag_source(OFFDIAG_SCREENED_MASK)
    launch = offdiag_emitter.offdiag_launch_source(OFFDIAG_SCREENED_MASK)
    assert _sha(certified) == OFFDIAG_SCREENED_SHA256["certified"]
    assert _sha(launch) == OFFDIAG_SCREENED_SHA256["launch"]


@pytest.mark.parametrize("mask", offdiag_emitter.LIVE_ROW_MASKS, ids=_mask_id)
def test_the_launch_text_is_the_certified_text_through_the_rule(mask):
    """What every lifter reads is the inverse of what compiles, on every row mask."""
    certified = offdiag_emitter.offdiag_source(mask)
    launch = offdiag_emitter.offdiag_launch_source(mask)
    assert launch == own_cell_hoist.hoisted_kernel_code(certified, OFFDIAG, "certified")
    assert own_cell_hoist.unhoisted_kernel_code(launch, OFFDIAG, "launch") == certified
    assert REGISTER_VIEW_MARKER in launch and REGISTER_VIEW_MARKER not in certified
    assert offdiag_emitter.shipped_kernel_names(launch) == {OFFDIAG}
    launch.encode("ascii")  # NVRTC writes the source through the locale encoding


@pytest.mark.parametrize("mask", offdiag_emitter.LIVE_ROW_MASKS, ids=_mask_id)
def test_the_off_diagonal_hoist_keeps_the_three_clauses(mask, offdiag_readings):
    before, after = offdiag_readings[mask]
    assert before["entry"] == after["entry"] == OFFDIAG
    # 1. the store sequence: six stores, auxiliary then field per component
    assert len(after["stores"]) == 6
    assert after["stores"] == before["stores"]
    assert after["int_decls"] == before["int_decls"]
    assert after["guards"] == before["guards"]
    assert after["opaque_reads"] == before["opaque_reads"]
    # 2. the load multiset: per component D and inv_eps, two per live row term, the two
    #    PML coefficients and the two own-cell words -- 18 + 2 per live row
    assert len(after["loads"]) == 18 + 2 * sum(mask)
    assert after["loads"] == before["loads"]
    # 3. no own-cell word read after the first store; the certified text exposes five
    assert after["exposed_after_first_store"] == []
    assert after["own_before_first_store"] == 6
    assert len(before["exposed_after_first_store"]) == 5
    assert after["reloads_after_first_store"] == before["reloads_after_first_store"] == []


@pytest.mark.parametrize("mutation", MUTATIONS)
@pytest.mark.parametrize("mask", OFFDIAG_GATE_MASKS, ids=_mask_id)
def test_each_clause_catches_its_planted_defect_on_the_off_diagonal_single(
        mask, mutation, offdiag_readings):
    before, _ = offdiag_readings[mask]
    launch = offdiag_emitter.offdiag_launch_source(mask)
    planted = _plant_single(launch, OFFDIAG, mutation)
    assert planted != launch
    after = read(planted)
    stores_equal = after["stores"] == before["stores"]
    loads_equal = after["loads"] == before["loads"]
    exposed = after["exposed_after_first_store"]
    if mutation in ("drop_a_store", "reorder_two_stores"):
        assert not stores_equal, f"{mask}: {mutation} left the store sequence equal"
        if mutation == "reorder_two_stores":
            with pytest.raises(AssertionError, match="write-back"):
                own_cell_hoist.unhoisted_kernel_code(planted, OFFDIAG, OFFDIAG)
    elif mutation == "add_a_load_after_a_store":
        assert not loads_equal, f"{mask}: an added load left the load multiset equal"
        assert stores_equal
        assert exposed == [], "a reload of the lane's own write is not an exposed load"
    else:
        assert exposed, f"{mask}: a hoisted load sunk past a store was not caught"
        assert stores_equal and loads_equal


@pytest.mark.parametrize("kernel", SINGLES)
def test_the_forward_reproduces_the_shipped_register_view_singles(kernel, hoisted):
    """The forward spelling is the one Round B spelt by hand, byte for byte."""
    statements = own_cell_hoist.unhoisted_kernel_code(hoisted[kernel], kernel, kernel)
    assert own_cell_hoist.hoisted_kernel_code(statements, kernel, kernel) == hoisted[kernel]


def test_the_forward_refuses_a_text_its_inverse_would_not_return():
    certified = offdiag_emitter.offdiag_source(OFFDIAG_SCREENED_MASK)
    # the helper stores the field before the auxiliary: the hazard half refuses
    swapped = certified.replace(
        "    fw[idx] = src;\n    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;\n",
        "    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;\n    fw[idx] = src;\n",
        1)
    assert swapped != certified
    with pytest.raises(AssertionError, match="write-back order"):
        own_cell_hoist.hoisted_kernel_code(swapped, OFFDIAG, "swapped")
    # a text already in the register view carries no certified call to rewrite
    launch = offdiag_emitter.offdiag_launch_source(OFFDIAG_SCREENED_MASK)
    with pytest.raises(AssertionError, match="not one"):
        own_cell_hoist.hoisted_kernel_code(launch, OFFDIAG, "launch")
    # a second decode line leaves the load block without an anchor
    decode = "    int i = idx / (ny * nz);\n"
    with pytest.raises(AssertionError, match="load block has no anchor"):
        own_cell_hoist.hoisted_kernel_code(certified.replace(decode, decode + decode, 1),
                                           OFFDIAG, "twice")


@pytest.mark.parametrize("kernel,text", [
    (folded_offdiag_kernels.KERNEL_NAME,
     lambda: folded_offdiag_kernels.folded_offdiag_source(OFFDIAG_SCREENED_MASK)),
    (dispersive_offdiag_update_e.KERNEL_NAME,
     lambda: dispersive_offdiag_update_e.dispersive_offdiag_source(OFFDIAG_SCREENED_MASK,
                                                                   (1, 1, 1))),
], ids=["folded", "dispersive"])
def test_the_rule_refuses_the_off_diagonal_singles_it_does_not_carry(kernel, text):
    """Their emitters are their own, so nothing here admits them; the refusal is by name."""
    assert kernel not in own_cell_hoist.KERNELS
    for function in (own_cell_hoist.hoisted_kernel_code, own_cell_hoist.unhoisted_kernel_code):
        with pytest.raises(ValueError, match=f"kernel must be one of .*got '{kernel}'"):
            function(text(), kernel, kernel)


def test_every_emitted_hoist_in_the_package_is_in_this_table():
    """A hoist issued through the rule's forward from another module fails by name."""
    assert set(own_cell_hoist.KERNELS) == RULE_TABLE
    issuers = {path.stem for path in HERE.glob("*.py")
               if not path.stem.startswith("test_") and path.stem != "own_cell_hoist"
               and re.search(r"\bhoisted_kernel_code\(", path.read_text(encoding="utf-8"))}
    assert issuers == EMITTED_HOIST_MODULES
