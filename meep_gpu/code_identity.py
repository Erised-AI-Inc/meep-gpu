"""Is this edit capable of changing what executes?

WHY THIS EXISTS. Every device gate welds its PASS verdict to the sha256 of the sources it
imported, and the rule those welds carry is *re-run the gate, do not edit this record*.
That rule is right, and it has a cost that is easy to underestimate: measured 2026-08-22,
renaming five environment switches touched five prose mentions in three modules --
docstrings, not code -- and drifted **76 of 70 welds** across the Triton and Metal tracks,
because ``driver.py`` and ``triton_kernels/launch.py`` are imported by nearly every gate.
Reverting those five prose lines dropped the drift to 7.

A tax that large on a docstring fix does not make the tree safer. It makes doc corrections
expensive, and an expensive correction is one that does not get made -- or worse, one that
gets made by editing the record, which is the failure the welds exist to prevent.

WHAT THIS IS, AND WHAT IT REFUSES TO BE. A comment and a docstring cannot change what a
kernel computes. Nothing else is granted that: a renamed variable, a reordered argument, a
changed string literal all reach execution and must be re-earned on a device. So the
question this module answers is deliberately narrow and mechanically decidable:

    do these two versions of a module have the SAME code, ignoring only
    comments and docstrings?

It is not "is this change trivial" and it is not "did the author intend a behaviour
change". Both of those are judgements, and a weld that can be released by a judgement is
not a weld. This is a parse, a strip, and a comparison.

HOW. :func:`code_digest` parses the source, removes every docstring expression, discards
the comments (which ``ast`` never keeps) and all formatting, and hashes ``ast.dump`` of
the result WITHOUT node attributes, so a docstring that grows by three lines does not
shift every node below it out of identity. Field values -- names, operators, constants,
argument order -- are all still compared.

THE CLAUSE THIS LICENSES, and no other: a weld whose recorded ``source_sha256`` no longer
matches the file MAY still stand if the weld also recorded a ``code_sha256`` and that is
unchanged. Backfilling ``code_sha256`` is only honest for a weld whose ``source_sha256``
still matches, because only then is the file on disk the bytes the gate actually ran.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Union

__all__ = ["canonical_dump", "code_digest", "code_digest_of_path", "strip_docstrings"]

#: Node types that gained a ``type_params`` field in Python 3.12 (PEP 695). Their
#: canonical dump carries ``type_params=[]`` on every interpreter.
_TYPE_PARAM_NODES = tuple(getattr(ast, name) for name in
                          ("FunctionDef", "AsyncFunctionDef", "ClassDef"))


def canonical_dump(node: object) -> str:
    """``ast.dump(node, annotate_fields=True, include_attributes=False)`` as Python 3.12
    prints it, on every interpreter.

    WHY NOT ``ast.dump`` ITSELF. Its text is part of the interpreter, not of the source:
    3.12 added ``type_params=[]`` to every function and class node, 3.10 and 3.11 have no
    such field, and 3.13 omits empty fields by default. Every recorded code and device
    digest was minted under 3.12, so hashing the interpreter's own dump makes a digest a
    fact about the Python that computed it: measured, all 78 Triton device digests fail to
    recompute under 3.10 on an unchanged tree.

    THE FORMAT, as 3.12 defines it: a node is ``Name(field=value, ...)`` over its fields in
    declared order, where a field whose value is ``None`` is left out when the node class
    declares ``None`` as that field's default (an optional field), and kept otherwise
    (``Constant(value=None)``); a list is ``[item, ...]``, ``[]`` when empty; anything else
    is its ``repr``. On an interpreter whose function and class nodes have no
    ``type_params`` field, ``type_params=[]`` is appended, which is what 3.12 prints for
    code that declares no type parameters.

    Verified byte-identical to 3.12's ``ast.dump`` over every module of this package and
    its certification harness, under 3.10 and under 3.12. Other interpreters are not
    verified; the digest tests are what would show a difference. One known exception:
    a string constant is printed by ``repr``, which escapes a character its own Unicode
    database does not class as printable, so a code point whose printability changed
    between that database and 3.12's (Unicode 15.0) prints differently. No module here
    contains one.
    """
    if isinstance(node, ast.AST):
        cls = type(node)
        args = []
        for name in cls._fields:
            try:
                value = getattr(node, name)
            except AttributeError:
                continue
            if value is None and getattr(cls, name, ...) is None:
                continue
            args.append(f"{name}={canonical_dump(value)}")
        if isinstance(node, _TYPE_PARAM_NODES) and "type_params" not in cls._fields:
            args.append("type_params=[]")
        return f"{cls.__name__}({', '.join(args)})"
    if isinstance(node, list):
        return "[" + ", ".join(canonical_dump(item) for item in node) + "]"
    return repr(node)


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove every docstring expression, in place, and return the tree.

    A docstring is the FIRST statement of a module, class or function when it is a bare
    string constant. A bare string anywhere else is not a docstring -- it could be an
    expression statement someone relies on -- so it is left alone.
    """
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            body.pop(0)
    return tree


def code_digest(source: str) -> str:
    """sha256 over the parsed code, with docstrings, comments and layout removed.

    Raises ``SyntaxError`` on unparsable input rather than falling back to hashing the
    text: a file that does not parse has no code identity, and silently degrading to a
    byte hash would make the clause that reads this quietly stricter instead of failing.
    """
    tree = strip_docstrings(ast.parse(source))
    # include_attributes=False IS THE LOAD-BEARING ARGUMENT, and it was wrong first.
    # With attributes ON, the dump carries lineno/col_offset, so ANY edit that changes a
    # line count shifts every node below it -- and a real docstring or comment edit
    # changes line counts almost by definition. Measured: a one-line docstring expanded
    # to four broke the identity, as did adding a two-line comment. The hatch would have
    # admitted only length-preserving prose edits, which is close to none of them.
    # Positions are metadata: they reach tracebacks and co_firstlineno, never arithmetic.
    # Field VALUES -- names, operators, constants, argument order -- are all still in the
    # dump with annotate_fields=True, which is what the refusal battery in
    # test_code_identity.py holds this to. canonical_dump prints that dump the same way
    # on every interpreter.
    return hashlib.sha256(canonical_dump(tree).encode("utf-8")).hexdigest()


def code_digest_of_path(path: Union[str, Path]) -> str:
    return code_digest(Path(path).read_text(encoding="utf-8"))
