"""Text witnesses for the bit-parity coverage board: the structural forms, read on text.

The board (``build_parity_board.py``) reads columns (a) and (b) off the device text each
module's builder EMITS (``parity_board_emit.py``), never off a hand-typed flag; that
text is judged here, by :func:`form_counts` (``helpers=False``: every helper's own body
is in the emitted text) and :func:`select_sites_c` / :func:`select_sites_python` (is
every field-word PML form inside a per-cell select?). A module with no emit route is
read off its OWN source text instead -- the v1 witness, below:

* **B select landed** -- the module reads the chunk record
  (``chunk_map.chunk_bounds_record``) in its own device text;
* **MEEP-order A form landed** -- the module's own text spells the curl stencil and the
  PML constitutive accumulation in MEEP's grouping and spells neither in the grouping
  the package shipped through 0.9.2.

THE MODULE'S OWN TEXT, AND NOTHING ELSE. A lifter that derives its device text from a
base module's text at import (an anchored replacement) carries the base's spelling only
after it has run; reading that would credit a lifter with its base's state, which the
kernel design calls false (DESIGN.md D10: same-cell fused pairs do NOT inherit with no
re-cut). So a module is judged only on what its own file spells:

* **CUDA and Metal**: every string constant of the module that is not a docstring (the
  first statement of a module, class or function), with an f-string's interpolated
  fields replaced by an identifier (``_F_``), C comments removed. The text is parsed as
  C expressions (:func:`c_expressions`).
* **Triton**: the body of every function decorated ``@triton.jit`` (or any decorator
  whose name ends in ``jit``), read as Python, plus every non-docstring string constant
  that parses as Python statements (a needle or a generated source).
* **Python calls** of the table's record emitters (``bounds_block`` on CUDA,
  ``chunk_bounds`` on Metal), read from the module's AST.

Comments and docstrings never count: a module that QUOTES a form in prose spells nothing.

THE WITNESSES ARE STRUCTURAL, NOT TEXTUAL. Every expression is parsed into one small
tree shape (``("bin", op, left, right)`` and friends) for both C and Python, the
table's complex helpers are read as the operators they implement (``cf_add`` is ``+``,
``c_mul_field_left`` is ``*``), and the forms are matched on the tree:

* curl, package grouping (0.9.2): ``(a - b) + (c - d)``;
* curl, MEEP grouping (``step_generic.cpp:104``): ``((a - b) + c) - d``, or a call of
  the table's MEEP-order stencil helper (:data:`CURL_HELPERS`);
* constitutive, package grouping: ``f + (kp * s)`` as a sum (the first half of
  ``(f + kps*src) - kms*prev``, whether written in one statement or two);
* constitutive, MEEP grouping: ``f + ((kp * s) - (km * p))``, or a call of the table's
  MEEP-order constitutive helper.

A coefficient is a leaf whose identifier starts ``kp`` / ``km`` (``kps``, ``kp_0``,
``kps_x[i]``). Multiplication operand order is free (KERNELS.md C3), so both orders match.

Stdlib only: the board must run on a host that imports no device library.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

#: Stand-in for an f-string's interpolated field, so ``src{t}`` reads ``src_F_``.
FIELD = "_F_"

Node = Tuple[Any, ...]

# ---------------------------------------------------------------------------
# Per-table spellings (structure of the tables, not a verdict about any module)
# ---------------------------------------------------------------------------

#: Complex helpers read as the operator they implement, per table.
COMPLEX_OPERATORS: Dict[str, Dict[str, str]] = {
    "cuda": {"cf_add": "+", "cf_sub": "-", "mul_coefficient_left": "*",
             "mul_field_left": "*", "cf_mul": "*"},
    "triton": {},
    "metal": {"c_mul": "*", "c_mul_field_left": "*", "c_mul_coefficient_left": "*"},
}

#: The table's MEEP-order curl-stencil helper, whose call is a MEEP-order curl site.
CURL_HELPERS: Dict[str, Tuple[str, ...]] = {
    "cuda": ("curl_stencil",),
    "triton": ("curl_stencil",),
    "metal": (),
}

#: The table's MEEP-order constitutive helper, whose call is a MEEP-order site.
CONSTITUTIVE_HELPERS: Dict[str, Tuple[str, ...]] = {
    "cuda": (),
    "triton": ("constitutive_form",),
    "metal": (),
}

#: Device-text reads of the chunk record, per table (regular expressions over the
#: comment-free device text).
RECORD_READ_TEXT: Dict[str, Tuple[str, ...]] = {
    "cuda": (r"\bcb\s*\[", r"\bin_[ih]_[xyz]\b"),
    "triton": (r"\brecord_in\s*\(\s*cb\b", r"\btl\.load\s*\(\s*cb\b", r"\bchunk_in\s*\("),
    "metal": (r"\bcb\s*\.\s*\w", r"\bprm\s*\.\s*cb\b", r"\bChunkBounds\b",
              r"__RECORD_BINDING__"),
}

#: Python calls that emit the record's read into the module's device text, per table:
#: ``(module the call is made through, function names)`` pairs. The owner matches with
#: its leading underscores removed (``from . import chunk_bounds as _chunk_bounds``), or
#: a bare call of an imported name.
RECORD_EMITTERS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "cuda": (("bounds_block", ("bounds_block", "insert_bounds_block", "membership_test")),),
    "triton": (("forms", ("record_in", "chunk_in")),),
    "metal": (("chunk_bounds", ("prologue", "binding", "struct_source", "membership_name")),
              ("shaders", ("curl_bounds_block", "constitutive_bounds_block",
                           "curl_membership_flags", "constitutive_membership_flags"))),
}

#: Device-text markers of a PML form (the auxiliary, or the PML coefficient tables), per
#: table. A module that spells one but reads no record steps a PML form without the
#: chunk division; a module that spells neither has no PML form of its own.
PML_FORM_TEXT: Dict[str, Tuple[str, ...]] = {
    "cuda": (r"\bpml_apply\w*\s*\(", r"\bconstitutive_apply\w*\s*\(", r"\bsinv\w*\b",
             r"\bkms\w*\b", r"\bkps\w*\b", r"\bfu_\w+", r"\bf_w_\w+"),
    "triton": (r"\bsinv\w*\b", r"\bsi_\w+", r"\bkm\w*\b", r"\bkp\w*\b", r"\bfu\w*\b"),
    "metal": (r"\bsinv\w*\b", r"\bsi_\w+", r"\bkm_\w+", r"\bkp_\w+", r"\bkm\d\b", r"\bkp\d\b"),
}

_C_LINE_COMMENT = re.compile(r"//[^\n]*")
_C_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_DEVICE_MARKERS = (";", "{", "__device__", "__global__", "kernel void", "[idx]", "[ii]",
                   "tl.", "(")


# ---------------------------------------------------------------------------
# The module's own text
# ---------------------------------------------------------------------------


@dataclass
class OwnText:
    """What one module spells in its own file."""

    literals: List[str] = field(default_factory=list)       # comment-free string constants
    jit_functions: List[ast.AST] = field(default_factory=list)
    python_calls: List[Tuple[Optional[str], str]] = field(default_factory=list)
    parse_error: Optional[str] = None

    @property
    def device_text(self) -> str:
        return "\n".join(self.literals)


#: A ``str.format`` / template placeholder: ``{kp}``, ``{prefix}`` -- an identifier alone
#: between braces. It is read as that identifier, so ``{prefix}acc + {kp} * {prefix}src``
#: reads ``prefix_acc + kp * prefix_src`` and the coefficient keeps its name.
_PLACEHOLDER = re.compile(r"\{([A-Za-z_]\w*)\}(?=(\w?))")
#: Prose: four lower-case words in a row ("and must write no"), which no kernel text of
#: these tables spells outside a comment. A message string that QUOTES a form
#: ("accumulates ((f + kps*src) - kms*prev) and ...") is not text a kernel compiles.
_PROSE = re.compile(r"\b[a-z]{2,} [a-z]{2,} [a-z]{2,} [a-z]{2,}\b")


def _placeholder_name(match: "re.Match[str]") -> str:
    return match.group(1) + ("_" if match.group(2) else "")


def _field_name(part: ast.AST) -> str:
    """An f-string field read as an identifier: ``{kp}`` is ``kp``, ``{a[0]}`` is ``a_0_``."""
    value = getattr(part, "value", None)
    try:
        text = ast.unparse(value) if value is not None else FIELD
    except Exception:  # pragma: no cover - unparse handles every expression node
        text = FIELD
    name = re.sub(r"\W", "_", text)
    return name if re.match(r"[A-Za-z_]", name) else FIELD + name


def _is_prose(code: str) -> bool:
    return bool(_PROSE.search(code))


def strip_c_comments(text: str) -> str:
    text = _C_BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    return _C_LINE_COMMENT.sub("", text)


def _docstring_ids(tree: ast.AST) -> Set[int]:
    ids: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _call_target(node: ast.Call) -> Tuple[Optional[str], str]:
    func = node.func
    if isinstance(func, ast.Attribute):
        base = func.value
        owner = base.id if isinstance(base, ast.Name) else (
            base.attr if isinstance(base, ast.Attribute) else None)
        return owner, func.attr
    if isinstance(func, ast.Name):
        return None, func.id
    return None, ""


def own_text(source: str) -> OwnText:
    """Collect a module's own string constants, ``@jit`` function bodies and calls."""
    out = OwnText()
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        out.parse_error = f"the module does not parse: {error}"
        return out
    skip = _docstring_ids(tree)
    inside_fstring: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for part in ast.walk(node):
                if part is not node:
                    inside_fstring.add(id(part))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                _decorator_name(d).endswith("jit") for d in node.decorator_list):
            out.jit_functions.append(node)
        if isinstance(node, ast.Call):
            out.python_calls.append(_call_target(node))
        if id(node) in skip or id(node) in inside_fstring:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raw = node.value
        elif isinstance(node, ast.JoinedStr):
            raw = "".join(part.value if isinstance(part, ast.Constant) else _field_name(part)
                          for part in node.values)
        else:
            continue
        code = _PLACEHOLDER.sub(_placeholder_name, strip_c_comments(raw))
        if any(marker in code for marker in _DEVICE_MARKERS) and not _is_prose(code):
            out.literals.append(code)
    return out


# ---------------------------------------------------------------------------
# One tree shape for C and Python expressions
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"""
    (?P<num>0[xX][0-9a-fA-F]+[uUlL]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuUlL]*)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op>->|\+\+|--|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|
            [-+*/%<>=!&|^~?:.,()\[\]])
  | (?P<space>\s+)
  | (?P<other>.)
""", re.X)

#: C type words: a parenthesised one is a cast, a leading one is a declaration.
_C_TYPES = frozenset({
    "float", "double", "int", "uint", "bool", "half", "float2", "float3", "float4", "cf",
    "unsigned", "long", "short", "char", "size_t", "ptrdiff_t", "const", "constant",
    "device", "thread", "threadgroup", "static", "inline", "auto", "volatile", "signed",
    "__restrict__", "restrict", "__device__", "__forceinline__", "register", "uint2",
    "uint3", "int2", "int3",
})
_C_KEYWORDS = frozenset({"return", "if", "else", "for", "while", "do", "case", "default",
                         "break", "continue", "goto", "switch"})

_BINARY_PRECEDENCE = {
    "||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6, "<": 7, ">": 7, "<=": 7,
    ">=": 7, "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10,
}


class _ParseFailure(Exception):
    pass


def _tokens(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for match in _TOKEN.finditer(text):
        kind = match.lastgroup
        if kind == "space":
            continue
        if kind == "other":
            raise _ParseFailure(f"unexpected character {match.group()!r}")
        out.append((kind, match.group()))
    return out


class _CParser:
    """A precedence-climbing parser for one C expression (no statements)."""

    def __init__(self, tokens: Sequence[Tuple[str, str]]):
        self.tokens = list(tokens)
        self.at = 0

    def peek(self, offset: int = 0) -> Tuple[str, str]:
        index = self.at + offset
        return self.tokens[index] if index < len(self.tokens) else ("end", "")

    def take(self, value: Optional[str] = None) -> Tuple[str, str]:
        token = self.peek()
        if value is not None and token[1] != value:
            raise _ParseFailure(f"expected {value!r}, found {token[1]!r}")
        if token[0] == "end":
            raise _ParseFailure("unexpected end")
        self.at += 1
        return token

    def parse(self) -> Node:
        node = self.ternary()
        if self.peek()[0] != "end":
            raise _ParseFailure(f"trailing {self.peek()[1]!r}")
        return node

    def ternary(self) -> Node:
        condition = self.binary(1)
        if self.peek()[1] == "?":
            self.take("?")
            first = self.ternary()
            self.take(":")
            second = self.ternary()
            return ("tern", condition, first, second)
        return condition

    def binary(self, minimum: int) -> Node:
        left = self.unary()
        while True:
            kind, value = self.peek()
            precedence = _BINARY_PRECEDENCE.get(value) if kind == "op" else None
            if precedence is None or precedence < minimum:
                return left
            self.take()
            right = self.binary(precedence + 1)
            left = ("bin", value, left, right)

    def unary(self) -> Node:
        kind, value = self.peek()
        if kind == "op" and value in ("-", "+", "!", "~", "*", "&", "++", "--"):
            self.take()
            return ("un", value, self.unary())
        if kind == "op" and value == "(" and self.peek(1)[0] == "id" \
                and self.peek(1)[1] in _C_TYPES:
            # a cast: (float) x, (const float*) p
            depth, index = 0, self.at
            while index < len(self.tokens):
                token = self.tokens[index][1]
                if token == "(":
                    depth += 1
                elif token == ")":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            inner = self.tokens[self.at + 1:index]
            if all(t[0] == "id" and t[1] in _C_TYPES or t[1] == "*" for t in inner):
                self.at = index + 1
                return ("cast", self.unary())
        return self.postfix()

    def postfix(self) -> Node:
        node = self.primary()
        while True:
            value = self.peek()[1]
            if value == "(":
                self.take("(")
                arguments: List[Node] = []
                if self.peek()[1] != ")":
                    arguments.append(self.ternary())
                    while self.peek()[1] == ",":
                        self.take(",")
                        arguments.append(self.ternary())
                self.take(")")
                node = ("call", _callee_name(node), tuple(arguments))
            elif value == "[":
                self.take("[")
                index = self.ternary()
                self.take("]")
                node = ("index", node, index)
            elif value in (".", "->"):
                self.take()
                node = ("attr", node, self.take()[1])
            elif value in ("++", "--"):
                self.take()
            else:
                return node

    def primary(self) -> Node:
        kind, value = self.take()
        if kind == "num":
            return ("num", value)
        if kind == "id":
            return ("name", value)
        if value == "(":
            node = self.ternary()
            self.take(")")
            return node
        raise _ParseFailure(f"unexpected {value!r}")


def _callee_name(node: Node) -> str:
    if node[0] == "name":
        return node[1]
    if node[0] == "attr":
        return node[2]
    return ""


def _top_level_split(tokens: Sequence[Tuple[str, str]], separators: Set[str]) -> List[List[Tuple[str, str]]]:
    parts: List[List[Tuple[str, str]]] = [[]]
    depth = 0
    for token in tokens:
        if token[1] in ("(", "["):
            depth += 1
        elif token[1] in (")", "]"):
            depth -= 1
        if depth == 0 and token[0] == "op" and token[1] in separators:
            parts.append([])
            continue
        parts[-1].append(token)
    return parts


_ASSIGNMENTS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}


def _parse_tokens(tokens: List[Tuple[str, str]]) -> Optional[Node]:
    try:
        return _CParser(tokens).parse()
    except (_ParseFailure, IndexError, RecursionError):
        return None


def _groups(tokens: Sequence[Tuple[str, str]]) -> Iterator[List[Tuple[str, str]]]:
    """Every maximal balanced parenthesised group's inner tokens, outermost first."""
    stack: List[int] = []
    for index, token in enumerate(tokens):
        if token[1] == "(":
            stack.append(index)
        elif token[1] == ")" and stack:
            start = stack.pop()
            if not stack:
                yield list(tokens[start + 1:index])


def _statement_trees(tokens: List[Tuple[str, str]]) -> List[Node]:
    """The expression trees of one C statement: each side of an assignment, or a fallback
    to every parenthesised group when the statement does not parse whole."""
    while tokens and tokens[0][0] == "id" and tokens[0][1] in _C_KEYWORDS:
        tokens = tokens[1:]
    # strip a declaration's leading type words and pointer stars: `const float* p = ...`
    index = 0
    while index < len(tokens) and (tokens[index][0] == "id" and tokens[index][1] in _C_TYPES
                                   or tokens[index][1] == "*"):
        index += 1
    if index and index < len(tokens) and tokens[index][0] == "id":
        tokens = tokens[index:]
    trees: List[Node] = []
    sides = _top_level_split(tokens, _ASSIGNMENTS)
    parsed_all = True
    for side in sides:
        if not side:
            continue
        for item in _top_level_split(side, {","}):
            if not item:
                continue
            tree = _parse_tokens(item)
            if tree is None:
                parsed_all = False
            else:
                trees.append(tree)
    if not parsed_all:
        for group in _groups(tokens):
            trees.extend(_statement_trees(group))
    return trees


def c_expressions(text: str) -> List[Node]:
    """Every expression tree in a fragment of C or Metal source."""
    trees: List[Node] = []
    for statement in re.split(r"[;{}]", strip_c_comments(text)):
        if not statement.strip() or statement.lstrip().startswith("#"):
            continue
        try:
            tokens = _tokens(statement)
        except _ParseFailure:
            continue
        if tokens:
            trees.extend(_statement_trees(tokens))
    return trees


_PY_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
              ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<",
              ast.RShift: ">>", ast.FloorDiv: "//", ast.Pow: "**", ast.MatMult: "@"}


def python_tree(node: ast.AST) -> Node:
    """A Python expression in the same tree shape as :func:`c_expressions`."""
    if isinstance(node, ast.BinOp):
        return ("bin", _PY_BINOPS.get(type(node.op), "?"), python_tree(node.left),
                python_tree(node.right))
    if isinstance(node, ast.UnaryOp):
        op = "-" if isinstance(node.op, ast.USub) else ("+" if isinstance(node.op, ast.UAdd) else "!")
        return ("un", op, python_tree(node.operand))
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Constant):
        return ("num", repr(node.value))
    if isinstance(node, ast.Attribute):
        return ("attr", python_tree(node.value), node.attr)
    if isinstance(node, ast.Subscript):
        return ("index", python_tree(node.value), python_tree(node.slice))
    if isinstance(node, ast.Call):
        name = _callee_name(python_tree(node.func))
        return ("call", name, tuple(python_tree(a) for a in node.args))
    if isinstance(node, ast.IfExp):
        return ("tern", python_tree(node.test), python_tree(node.body), python_tree(node.orelse))
    return ("other", type(node).__name__)


def python_expressions(nodes: Iterable[ast.AST]) -> List[Node]:
    """Every maximal expression (statement value or assignment side) under ``nodes``."""
    trees: List[Node] = []
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                value = node.value
                if value is not None:
                    trees.append(python_tree(value))
                if isinstance(node, ast.AugAssign):
                    trees.append(("bin", _PY_BINOPS.get(type(node.op), "?"),
                                  python_tree(node.target), python_tree(node.value)))
            elif isinstance(node, (ast.Return, ast.Expr)) and node.value is not None:
                trees.append(python_tree(node.value))
    return trees


def python_literal_expressions(literals: Iterable[str]) -> List[Node]:
    """String constants that parse as Python statements (a needle, a generated source)."""
    trees: List[Node] = []
    for literal in literals:
        candidates = [literal] + literal.splitlines()
        for candidate in candidates:
            text = candidate.strip()
            if not text:
                continue
            try:
                parsed = ast.parse(text)
            except SyntaxError:
                continue
            trees.extend(python_expressions([parsed]))
            if candidate is literal:
                break
    return trees


# ---------------------------------------------------------------------------
# The forms, matched on the trees
# ---------------------------------------------------------------------------


def normalise(node: Node, operators: Dict[str, str]) -> Node:
    """Read the table's complex helpers as operators, recursively; drop casts."""
    kind = node[0]
    if kind == "cast":
        return normalise(node[1], operators)
    if kind == "bin":
        return ("bin", node[1], normalise(node[2], operators), normalise(node[3], operators))
    if kind == "un":
        return ("un", node[1], normalise(node[2], operators))
    if kind == "tern":
        return ("tern",) + tuple(normalise(part, operators) for part in node[1:])
    if kind == "call":
        arguments = tuple(normalise(a, operators) for a in node[2])
        if node[1] in operators and len(arguments) == 2:
            return ("bin", operators[node[1]], arguments[0], arguments[1])
        return ("call", node[1], arguments)
    if kind == "index":
        return ("index", normalise(node[1], operators), normalise(node[2], operators))
    if kind == "attr":
        return ("attr", normalise(node[1], operators), node[2])
    return node


def walk(node: Node) -> Iterator[Node]:
    yield node
    kind = node[0]
    if kind == "bin":
        yield from walk(node[2])
        yield from walk(node[3])
    elif kind in ("un", "cast"):
        yield from walk(node[-1])
    elif kind == "tern":
        for part in node[1:]:
            yield from walk(part)
    elif kind == "call":
        for argument in node[2]:
            yield from walk(argument)
    elif kind == "index":
        yield from walk(node[1])
        yield from walk(node[2])
    elif kind == "attr":
        yield from walk(node[1])


def _identifier(node: Node) -> Optional[str]:
    """The base identifier of a leaf: ``kps`` of ``kps_x[i]``, ``z`` of ``z.x``."""
    while node[0] in ("index", "attr"):
        node = node[1]
    return node[1] if node[0] == "name" else None


def _is_operand(node: Node) -> bool:
    """A field or coefficient operand: a name, an element, a member, a load or a product
    -- never a bare numeric constant (index arithmetic is not a form)."""
    if node[0] == "num":
        return False
    if node[0] == "un":
        return _is_operand(node[2])
    return node[0] in ("name", "index", "attr", "call", "bin", "tern")


def _is(node: Node, op: str) -> bool:
    return node[0] == "bin" and node[1] == op


def _leaf_operand(node: Node) -> bool:
    return _is_operand(node) and not (node[0] == "bin" and node[1] in "+-")


def _coefficient(node: Node, prefix: str) -> bool:
    name = _identifier(node)
    return name is not None and name.lower().startswith(prefix)


def _product_with(node: Node, prefix: str) -> bool:
    return _is(node, "*") and (_coefficient(node[2], prefix) or _coefficient(node[3], prefix))


def old_curl(node: Node) -> bool:
    """``(a - b) + (c - d)``: the package's curl grouping through 0.9.2."""
    return (_is(node, "+") and _is(node[2], "-") and _is(node[3], "-")
            and all(_leaf_operand(x) for x in (node[2][2], node[2][3], node[3][2], node[3][3])))


def new_curl(node: Node) -> bool:
    """``((a - b) + c) - d``: MEEP's left-to-right bracket (step_generic.cpp:104)."""
    if not (_is(node, "-") and _is(node[2], "+") and _is(node[2][2], "-")):
        return False
    inner = node[2][2]
    return all(_leaf_operand(x) for x in (inner[2], inner[3], node[2][3], node[3]))


def old_constitutive(node: Node) -> bool:
    """``f + (kp * s)``: the first accumulation of ``(f + kps*src) - kms*prev``."""
    return (_is(node, "+") and _leaf_operand(node[2]) and not _product_with(node[2], "kp")
            and _product_with(node[3], "kp")) or (
        _is(node, "+") and _leaf_operand(node[3]) and not _product_with(node[3], "kp")
        and _product_with(node[2], "kp") and not _is(node[3], "*"))


def new_constitutive(node: Node) -> bool:
    """``f + ((kp * s) - (km * p))``: MEEP's grouping (update_eh.cpp)."""
    if not _is(node, "+"):
        return False
    for field_side, sum_side in ((node[2], node[3]), (node[3], node[2])):
        if _leaf_operand(field_side) and _is(sum_side, "-") \
                and _product_with(sum_side[2], "kp") and _product_with(sum_side[3], "km"):
            return True
    return False


@dataclass
class FormCounts:
    """Sites of each form in one module's own text."""

    old_curl: int = 0
    new_curl: int = 0
    old_constitutive: int = 0
    new_constitutive: int = 0
    trees: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {"old_curl": self.old_curl, "new_curl": self.new_curl,
                "old_constitutive": self.old_constitutive,
                "new_constitutive": self.new_constitutive, "expressions": self.trees}


def module_trees(table: str, text: OwnText) -> List[Node]:
    """Every normalised expression tree of the module's own text."""
    operators = COMPLEX_OPERATORS[table]
    if table == "triton":
        trees = python_expressions(text.jit_functions) + python_literal_expressions(text.literals)
    else:
        trees = []
        for literal in text.literals:
            trees.extend(c_expressions(literal))
    return [normalise(tree, operators) for tree in trees]


def form_counts(table: str, text: OwnText, helpers: bool = True) -> FormCounts:
    """Sites of each form. ``helpers=False`` reads no helper call as a site: the emitted
    text (``parity_board_emit.py``) carries every helper's own body, so the grouping is
    matched where it is spelt and a helper's name is never taken for its content."""
    counts = FormCounts()
    curl_helpers = set(CURL_HELPERS[table]) if helpers else set()
    constitutive_helpers = set(CONSTITUTIVE_HELPERS[table]) if helpers else set()
    seen_new_curl: Set[int] = set()
    for tree in module_trees(table, text):
        counts.trees += 1
        for node in walk(tree):
            if old_curl(node):
                counts.old_curl += 1
            if new_curl(node):
                counts.new_curl += 1
            if old_constitutive(node):
                counts.old_constitutive += 1
            if new_constitutive(node):
                counts.new_constitutive += 1
            if node[0] == "call" and node[1] in curl_helpers:
                counts.new_curl += 1
            if node[0] == "call" and node[1] in constitutive_helpers:
                counts.new_constitutive += 1
    del seen_new_curl
    return counts


# ---------------------------------------------------------------------------
# The select: are the PML forms chosen per cell, or spelt everywhere?
# ---------------------------------------------------------------------------
#
# A text that reads the chunk record can still step a PML form at every cell: a
# lifter that splices a base body carrying the record's decode beside a curl of its
# own that runs the recurrence unconditionally reads the record and selects nothing.
# The two forms whose result IS the stored field word are therefore located, and each
# must sit where a per-cell choice puts it -- a branch of a conditional operator, the
# body of an ``if``/``else``, or a value of ``tl.where``:
#
# * the recurrence's second stage, ``(((f * km_u) + n) - p) * si_u`` (any grouping of
#   the inner sum: the select is the question here, not the grouping);
# * the constitutive accumulation, in either grouping (``f + ((kp*s) - (km*p))`` and
#   the package's ``f + (kp * s)``).
#
# The first stage ``((p * km) - curl) * si`` is not located: the dsig-only form has
# the same shape, and a select that computes the auxiliary everywhere and stores it
# under its flag (the base templates' shape) is a legitimate one.


def _starts(node: Node, prefix: str) -> bool:
    name = _identifier(node)
    return name is not None and name.lower().startswith(prefix)


def recurrence_second_stage(node: Node) -> bool:
    """``((f * km_u) + n - p) * si_u``: a product with a ``si`` coefficient whose other
    factor is a difference whose minuend is a sum holding a ``km`` product."""
    if not _is(node, "*"):
        return False
    for coefficient, rest in ((node[3], node[2]), (node[2], node[3])):
        if _starts(coefficient, "si") and _is(rest, "-"):
            inner = rest[2]
            if _is(inner, "+") and (_product_with(inner[2], "km") or _product_with(inner[3], "km")):
                return True
    return False


def accumulation(node: Node) -> bool:
    return new_constitutive(node) or old_constitutive(node)


@dataclass
class SelectSites:
    """Field-word PML form sites: inside a per-cell select, and outside every select."""

    selected: int = 0
    unselected: int = 0
    examples: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"selected": self.selected, "unselected": self.unselected,
                "examples": self.examples[:3]}


def _site_kind(node: Node) -> Optional[str]:
    if recurrence_second_stage(node):
        return "recurrence"
    if accumulation(node):
        return "accumulation"
    return None


def _walk_selected(node: Node, selected: bool, sink: SelectSites, where: Tuple[str, ...]) -> None:
    kind = _site_kind(node)
    if kind:
        if selected:
            sink.selected += 1
        else:
            sink.unselected += 1
            sink.examples.append(kind)
    tag = node[0]
    if tag == "tern":
        _walk_selected(node[1], selected, sink, where)
        _walk_selected(node[2], True, sink, where)
        _walk_selected(node[3], True, sink, where)
        return
    if tag == "call":
        arguments = node[2]
        chooses = node[1] in where
        for index, argument in enumerate(arguments):
            _walk_selected(argument, selected or (chooses and index > 0), sink, where)
        return
    if tag == "bin":
        _walk_selected(node[2], selected, sink, where)
        _walk_selected(node[3], selected, sink, where)
    elif tag in ("un", "cast"):
        _walk_selected(node[-1], selected, sink, where)
    elif tag == "index":
        _walk_selected(node[1], selected, sink, where)
        _walk_selected(node[2], selected, sink, where)
    elif tag == "attr":
        _walk_selected(node[1], selected, sink, where)


_GUARD_HEADER = re.compile(r"(?:\bif\s*\(.*\)|\belse)\s*$", re.S)
_BRACELESS_IF = re.compile(r"^\s*(?:else\s+)?if\s*\(")


def _c_guarded_statements(text: str) -> Iterator[Tuple[str, bool]]:
    """Each statement of a C/MSL text and whether an ``if``/``else`` encloses it."""
    text = strip_c_comments(text)
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    stack: List[bool] = []
    start = 0
    for index, char in enumerate(text):
        if char not in "{};":
            continue
        piece = text[start:index]
        start = index + 1
        guarded = any(stack)
        if char == "{":
            stack.append(guarded or bool(_GUARD_HEADER.search(piece.strip())))
            continue
        if char == "}":
            if piece.strip():
                yield piece, guarded or bool(_BRACELESS_IF.match(piece))
            if stack:
                stack.pop()
            continue
        if piece.strip():
            yield piece, guarded or bool(_BRACELESS_IF.match(piece))


def select_sites_c(table: str, text: str) -> SelectSites:
    """:class:`SelectSites` of a C or MSL text (the operators of the table's complex
    helpers read as operators, as everywhere else)."""
    sink = SelectSites()
    operators = COMPLEX_OPERATORS[table]
    for statement, guarded in _c_guarded_statements(text):
        try:
            tokens = _tokens(statement)
        except _ParseFailure:
            continue
        if not tokens:
            continue
        for tree in _statement_trees(tokens):
            _walk_selected(normalise(tree, operators), guarded, sink, ())
    return sink


def _python_callee(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def select_sites_python(functions: Iterable[ast.AST]) -> SelectSites:
    """:class:`SelectSites` of ``@triton.jit`` bodies: ``tl.where``'s values and a
    conditional expression's branches select; a Python ``if`` does not (in a Triton
    program it is a compile-time or a per-program branch, not a per-cell one).

    A Triton program computes both arms of ``tl.where`` and stores one, so a form
    assigned to a name is selected when every read of that name is a selected value
    (``recurrence = ...`` then ``tl.where(dsigu, recurrence, dsig_only)``)."""
    sink = SelectSites()
    for function in functions:
        loads: Dict[str, List[bool]] = {}
        found: List[Tuple[str, bool, Optional[str]]] = []

        def visit(node: ast.AST, selected: bool, target: Optional[str]) -> None:
            if isinstance(node, ast.Call) and _python_callee(node) == "where":
                visit(node.func, selected, None)
                for index, argument in enumerate(node.args):
                    visit(argument, selected or index > 0, None)
                for keyword in node.keywords:
                    visit(keyword.value, selected, None)
                return
            if isinstance(node, ast.IfExp):
                visit(node.test, selected, None)
                visit(node.body, True, None)
                visit(node.orelse, True, None)
                return
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loads.setdefault(node.id, []).append(selected)
            if isinstance(node, ast.BinOp):
                kind = _site_kind(python_tree(node))
                if kind:
                    found.append((kind, selected, target))
            if isinstance(node, ast.Assign):
                name = (node.targets[0].id if len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) else None)
                visit(node.value, selected, name)
                return
            for child in ast.iter_child_nodes(node):
                visit(child, selected, target if isinstance(node, ast.BinOp) else None)

        visit(function, False, None)
        for kind, selected, target in found:
            if not selected and target is not None:
                reads = loads.get(target, [])
                selected = bool(reads) and all(reads)
            if selected:
                sink.selected += 1
            else:
                sink.unselected += 1
                sink.examples.append(kind)
    return sink


def record_reads(table: str, text: OwnText) -> List[str]:
    """The witnesses that this module's own text reads the chunk record."""
    found: List[str] = []
    device = text.device_text
    for pattern in RECORD_READ_TEXT[table]:
        if re.search(pattern, device):
            found.append(f"device text matches {pattern}")
    if table == "triton":
        for function in text.jit_functions:
            source = ast.unparse(function)
            for pattern in RECORD_READ_TEXT[table]:
                if re.search(pattern, source):
                    found.append(f"@jit {function.name} matches {pattern}")
                    break
    for owner, names in RECORD_EMITTERS[table]:
        for call_owner, name in text.python_calls:
            if name in names and (call_owner is None or call_owner.lstrip("_") == owner):
                found.append(f"calls {call_owner + '.' if call_owner else ''}{name}")
                break
    return found


def pml_form_markers(table: str, text: OwnText) -> List[str]:
    """Device-text markers of a PML form (auxiliary or coefficient tables)."""
    device = text.device_text
    if table == "triton":
        device = device + "\n" + "\n".join(ast.unparse(f) for f in text.jit_functions)
    return [pattern for pattern in PML_FORM_TEXT[table] if re.search(pattern, device)]
