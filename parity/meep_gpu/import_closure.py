#!/usr/bin/env python3
"""Which repository files does a gate import? Its import closure, read off the source.

WHY THIS EXISTS. A gate artifact's import record (``gate_provenance.IMPORTED_KEY``, the Metal
runner's ``source_sha256``) is every repository module the gate's process held when it wrote
the artifact, and the rebind tools, the family and composition re-cuts, the bit-identity
re-cut, the Metal runner's per-stamp manifest and the three board builders all check that
WHOLE record against the checkout. That strictness is the point and stays. What it implies
is that the record must be the gate's real dependency set: when the package root imported
the driver, the dispatch planners and the MEEP lift eagerly, every gate's record held them,
and a dispatch-only edit voided the evidence of every kernel. This module measures, without
running anything, what a gate's record can hold, so ``test_gate_import_closure.py`` can keep
a kernel gate's record free of the dispatch layer.

TWO CLOSURES, both over repository files (``meep_gpu/`` and ``parity/meep_gpu/``, keyed
repo-relative, as ``gate_provenance.provenance`` keys them):

* the IMPORT closure: what importing the gate loads, without running its ``main``. Module-level
  statements only (a function body is not executed by an import; an ``if TYPE_CHECKING:``
  body and an ``if __name__ == "__main__":`` body are not executed either), followed
  transitively, with the parent package ``__init__`` of every submodule. This is the closure
  a fresh interpreter measures (``measured_import_closure``); the test holds the measured one
  inside it with the same dispatch-layer hits. Where CuPy and Triton import the two are
  equal; elsewhere the measured one skips what a guarded ``import triton`` stands in front of.
* the RUN closure: an upper bound on what the gate's record can hold after it runs. From the
  gate's module-level code, its ``__main__`` block and its ``main``, every function the code
  reaches contributes the modules it imports and the functions it calls or names (a call
  through a module alias or an imported name is followed into that module; a method call in
  harness code reaches every same-named harness method, since a receiver's type is not read).
  Function bodies a gate never reaches are not counted, which is what separates a kernel gate
  that imports a kit from the kit's driver-building legs it does not run.
  :data:`PACKAGE_LAZY_EDGES` lists the production modules' function-level imports of the
  dispatch layer, for review.

Resolution follows how a gate runs: ``python parity/meep_gpu/<gate>.py`` with the checkout on
``PYTHONPATH``, so a top-level name resolves first beside the gate (a harness module) and then
at the checkout root (the ``meep_gpu`` package); anything else is a third-party library and is
not part of a record. A name the package root binds on first use (its ``_LAZY`` table, read
by its module ``__getattr__``) resolves to the submodule that defines it, the way the
interpreter resolves it: ``from meep_gpu import lift_simulation`` and
``meep_gpu.lift_simulation(...)`` both import ``meep_gpu/from_meep.py``
(:meth:`Resolver.lazy_names`). A module loaded by path
(``importlib.util.spec_from_file_location``) is followed when the path is a literal file name
in the function that loads it, which is how the harness spells every such load.

Stdlib only: it runs on every host the CI view runs on.

CURATED PINS THE OLD EAGER ROOT LEFT BEHIND (:func:`unimported_pins`). A weld's curated pin
set was cut from its gate's import record while the package root loaded its 23 submodules
eagerly, so some welds pin core modules (the driver, the planner, the lift) that their gate
never needed. Their next record cannot hold those files, and the rebind tools keep curated
sets, so such a weld could never bind again. The rule that retires such a pin is computed
here and nowhere else: a pin is retired from a weld only when the file is one of the modules
the eager root loaded (:func:`eager_root_modules`) AND it lies outside the run closure of the
weld's gate. Anything else a gate does not import (a weld's own test file, a kernel module the
gate loads by computed name) is a deliberate pin and stays. Each rebind tool carries the
result as a reviewed literal, ``RETIRED_UNIMPORTED_PINS``, which
``test_rebind_unimported_pins.py`` holds to this function both ways.

usage:
  import_closure.py report [--root R] [--json OUT]       per gate: closures and dispatch-layer hits
  import_closure.py bill FILE [--root R]                 which gates' run closures hold FILE
  import_closure.py unimported-pins TABLE [--root R]     the RETIRED_UNIMPORTED_PINS rows of one
                                                         rebind tool (metal, triton or cuda)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
#: The repository root: this file -> parity/meep_gpu -> parity -> the root.
REPO_ROOT = HERE.parents[1]
HARNESS = "parity/meep_gpu"

#: The dispatch layer: the planners and the driver whose freeze consults them.
DISPATCH_LAYER: Tuple[str, ...] = (
    "meep_gpu/driver.py",
    "meep_gpu/fastpath.py",
    "meep_gpu/fastpath_cuda.py",
    "meep_gpu/metal_dispatch.py",
    "meep_gpu/dispatch_preference.py",
)
#: The MEEP lift. Not dispatch, but no kernel gate needs it either.
LIFT: Tuple[str, ...] = ("meep_gpu/from_meep.py",)
#: What a kernel gate's record must not hold.
NOT_KERNEL: Tuple[str, ...] = DISPATCH_LAYER + LIFT

#: Every function-level import of a :data:`NOT_KERNEL` module inside the package's
#: production modules (its tests excluded): (importing module, function, imported module).
#: The run closure finds these by following the calls a gate makes; this list is the
#: review record, and ``test_gate_import_closure`` holds it to the source both ways, so a
#: new lazy import of the planner is read before any gate inherits it.
PACKAGE_LAZY_EDGES: Tuple[Tuple[str, str, str], ...] = (
    # The array-path oracle builds its cases on reference drivers.
    ("meep_gpu/array_path_oracle.py", "_build", "meep_gpu/driver.py"),
    # ``available_gpu`` reads the MPS hardware through the planner's one definition of it.
    ("meep_gpu/backends.py", "available_gpu", "meep_gpu/fastpath.py"),
)


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _is_main_guard(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare) or len(test.comparators) != 1:
        return False
    sides = [test.left, test.comparators[0]]
    names = [s.id for s in sides if isinstance(s, ast.Name)]
    values = [s.value for s in sides if isinstance(s, ast.Constant)]
    return names == ["__name__"] and values == ["__main__"]


class _Collector(ast.NodeVisitor):
    """Import statements and by-path loads of one file.

    ``module_level`` keeps what an import executes: the module-level statements and, since
    a module-level call runs a function body at import, the bodies of the same-file
    functions those statements call (followed through the calls those bodies make).
    Otherwise every function body and the ``__main__`` block count.
    """

    def __init__(self, module_level: bool, skip: FrozenSet[int] = frozenset()):
        self.module_level = module_level
        #: ``id()`` of lambda nodes not to enter (a run-closure table's deferred lambdas).
        self.skip = skip
        self.imports: List[Tuple[int, str, Optional[List[str]]]] = []  # (level, module, names)
        self.loaded_paths: List[str] = []
        self.called: Set[str] = set()
        self._in_function = 0

    def collect(self, tree: ast.Module) -> "_Collector":
        self.visit(tree)
        if self.module_level:
            functions = {n.name: n for n in tree.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            done: Set[str] = set()
            while True:
                pending = sorted((self.called & set(functions)) - done)
                if not pending:
                    break
                for name in pending:
                    done.add(name)
                    self.loaded_paths.extend(_by_path_literals(functions[name]))
                    self._in_function += 1
                    for statement in functions[name].body:
                        self.visit(statement)
                    self._in_function -= 1
        return self

    # --- scope ---------------------------------------------------------------------
    def visit_FunctionDef(self, node):  # noqa: N802 - ast API
        if self.module_level and not self._in_function:
            return  # a definition runs nothing at import; a CALL of it is followed above
        self.loaded_paths.extend(_by_path_literals(node))
        self._in_function += 1
        self.generic_visit(node)
        self._in_function -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):  # noqa: N802
        if id(node) in self.skip:
            return
        if not self.module_level or self._in_function:
            self.generic_visit(node)

    def visit_If(self, node):  # noqa: N802
        if _is_type_checking(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        if _is_main_guard(node.test) and self.module_level and not self._in_function:
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    # --- what is imported ------------------------------------------------------------
    def visit_Import(self, node):  # noqa: N802
        for alias in node.names:
            self.imports.append((0, alias.name, None))

    def visit_ImportFrom(self, node):  # noqa: N802
        self.imports.append((node.level, node.module or "", [a.name for a in node.names]))

    def visit_Call(self, node):  # noqa: N802
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "")
        if name:
            self.called.add(name)
        if name == "import_module" and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str) and not node.args[0].value.startswith("."):
            self.imports.append((0, node.args[0].value, None))
        if name == "spec_from_file_location":
            self.loaded_paths.extend(_py_literals(node))
        self.generic_visit(node)

    def finish(self) -> List[str]:
        return list(self.loaded_paths)


def _py_literals(node: ast.AST) -> List[str]:
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py")]


def _by_path_literals(function: ast.AST) -> List[str]:
    """The ``.py`` literals of a function that loads a module by path, else nothing."""
    if not any(isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Attribute) and n.func.attr == "spec_from_file_location")
            or (isinstance(n.func, ast.Name) and n.func.id == "spec_from_file_location"))
            for n in ast.walk(function)):
        return []
    return _py_literals(function)


_CONTAINERS = (ast.Tuple, ast.List, ast.Set, ast.Dict)


def _deferred_lambdas(statement: ast.AST) -> Tuple[Tuple[str, ...], List[ast.expr]]:
    """(target names, entries) of a module-level table assignment whose callables do not run.

    ``NAME = (("row", lambda: build(...), builder), ...)`` stores a lambda and a function
    without calling either: they run only when code that reaches ``NAME`` takes them out and
    calls them. An entry is a lambda, a name or a dotted name whose every ancestor up to the
    assigned value is a tuple, list, set or dict display; anything inside another expression
    (a call's argument, a comprehension) may run at import and is left to the statement. Only
    simple-name targets qualify.
    """
    if isinstance(statement, ast.Assign):
        targets, value = statement.targets, statement.value
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets, value = [statement.target], statement.value
    else:
        return (), []
    if not targets or not all(isinstance(t, ast.Name) for t in targets):
        return (), []
    found: List[ast.expr] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, (ast.Lambda, ast.Name, ast.Attribute)):
            found.append(node)
        elif isinstance(node, _CONTAINERS):
            for child in ast.iter_child_nodes(node):
                walk(child)

    walk(value)
    return tuple(t.id for t in targets), found


class Resolver:
    """Repository files for import statements, under one checkout root."""

    def __init__(self, root: Path = REPO_ROOT, overrides: Optional[Mapping[str, str]] = None):
        self.root = root.resolve()
        #: repo-relative path -> source text to read INSTEAD of the file (planted controls).
        self.overrides = dict(overrides or {})
        self._parsed: Dict[str, ast.Module] = {}

    # --- files -------------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def exists(self, rel: str) -> bool:
        return rel in self.overrides or (self.root / rel).is_file()

    def tree(self, rel: str) -> ast.Module:
        if rel not in self._parsed:
            text = self.overrides.get(rel)
            if text is None:
                text = (self.root / rel).read_text(encoding="utf-8")
            self._parsed[rel] = ast.parse(text, filename=rel)
        return self._parsed[rel]

    @lru_cache(maxsize=None)
    def lazy_names(self, rel: str) -> Mapping[str, Tuple[str, str]]:
        """``name -> (defining file, attribute)`` a package ``__init__`` binds on first use.

        Read off the source: a module-level ``_LAZY = {name: (submodule, attribute), ...}``
        dict literal in a package ``__init__`` that also defines a module-level
        ``__getattr__``, which is how the package root spells its lazy names. Empty for any
        other file, so a name the table does not hold resolves as before.
        """
        if not rel.endswith("/__init__.py") or not self.exists(rel):
            return {}
        tree = self.tree(rel)
        if not any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body):
            return {}
        package = rel[: -len("/__init__.py")].replace("/", ".")
        out: Dict[str, Tuple[str, str]] = {}
        for node in tree.body:
            targets = [t.id for t in getattr(node, "targets", ()) if isinstance(t, ast.Name)]
            if not (isinstance(node, ast.Assign) and targets == ["_LAZY"]):
                continue
            for name, (submodule, attribute) in ast.literal_eval(node.value).items():
                target = self.module_file(f"{package}.{submodule}", [""])
                if target is not None:
                    out[name] = (target, attribute)
        return out

    def module_file(self, dotted: str, search: Sequence[str]) -> Optional[str]:
        """The repo-relative file a dotted module name loads, searching ``search`` dirs."""
        return self._module_file(dotted, tuple(search))

    @lru_cache(maxsize=None)
    def _module_file(self, dotted: str, search: Tuple[str, ...]) -> Optional[str]:
        parts = dotted.split(".")
        for base in search:
            stem = "/".join(([base] if base else []) + parts)
            for candidate in (f"{stem}/__init__.py", f"{stem}.py"):
                if self.exists(candidate):
                    return candidate
        return None

    def parents(self, rel: str) -> List[str]:
        """Every package ``__init__`` above ``rel`` that an import of it loads."""
        out: List[str] = []
        if not rel.startswith("meep_gpu/"):
            return out  # a harness module is imported by its top-level name, never as a package member
        parts = rel.split("/")[:-1]
        if rel.endswith("/__init__.py"):
            parts = parts[:-1]
        for i in range(1, len(parts) + 1):
            init = "/".join(parts[:i]) + "/__init__.py"
            if self.exists(init):
                out.append(init)
        return out

    def _package_inits(self, dotted: str, dirs: Sequence[str]) -> List[str]:
        """The ``__init__`` of every package a dotted import passes through (``a.b.c``: a, a.b)."""
        parts = dotted.split(".")
        out: List[str] = []
        for i in range(1, len(parts)):
            found = self.module_file(".".join(parts[:i]), dirs)
            if found is not None and found.endswith("/__init__.py"):
                out.append(found)
        return out

    def _search_and_package(self, rel: str) -> Tuple[List[str], Optional[str]]:
        """(top-level search dirs, dotted package) for imports spelled inside ``rel``."""
        if rel.startswith("meep_gpu/"):
            package = rel[:-3].replace("/", ".")
            if rel.endswith("/__init__.py"):
                package = package[: -len(".__init__")]
            else:
                package = package.rpartition(".")[0]
            return [""], package
        # A harness file: its own directory first (``python <script>``), then the root.
        return [str(Path(rel).parent.as_posix()), ""], None

    @lru_cache(maxsize=None)
    def edges(self, rel: str, module_level: bool) -> Tuple[str, ...]:
        """Repository files ``rel`` imports (module-level only, or anywhere)."""
        collector = _Collector(module_level).collect(self.tree(rel))
        search, package = self._search_and_package(rel)
        out: List[str] = []
        for level, module, names in collector.imports:
            if level:
                if package is None:
                    continue
                base = package
                for _ in range(level - 1):
                    base = base.rpartition(".")[0]
                dotted = ".".join(p for p in (base, module) if p)
                dirs = [""]
            else:
                dotted = module
                dirs = search
            target = self.module_file(dotted, dirs) if dotted else None
            if target is not None:
                out.append(target)
                out.extend(self._package_inits(dotted, dirs))
                for name in names or ():
                    if name == "*":
                        continue
                    sub = self.module_file(f"{dotted}.{name}", dirs)
                    if sub is not None:
                        out.append(sub)
                    elif name in self.lazy_names(target):
                        # ``from pkg import NAME`` runs pkg.__getattr__(NAME): its submodule loads.
                        out.append(self.lazy_names(target)[name][0])
        here = Path(rel).parent.as_posix()
        for literal in collector.finish():
            candidate = f"{here}/{Path(literal).name}" if "/" not in literal else literal
            if self.exists(candidate):
                out.append(candidate)
        return tuple(out)

    # --- symbols, for the run closure ------------------------------------------------
    def _resolve(self, rel: str, level: int, module: str) -> Tuple[Optional[str], str, List[str]]:
        """(file, dotted, search dirs) for one import statement spelled in ``rel``."""
        search, package = self._search_and_package(rel)
        if level:
            if package is None:
                return None, "", []
            base = package
            for _ in range(level - 1):
                base = base.rpartition(".")[0]
            dotted = ".".join(p for p in (base, module) if p)
            dirs = [""]
        else:
            dotted, dirs = module, search
        return (self.module_file(dotted, dirs) if dotted else None), dotted, dirs

    @lru_cache(maxsize=None)
    def symbols(self, rel: str) -> "_Symbols":
        tree = self.tree(rel)
        functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        classes: Dict[str, Dict[str, ast.AST]] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {n.name: n for n in node.body
                                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        #: module-level table name -> the callables its assignment stores without calling.
        #: HARNESS files only: there a method call reaches every same-named harness method,
        #: so a table read inside a method is still reached through the method; a package
        #: method is followed only through a resolved name, so a package table keeps the
        #: old rule (every entry counts as called by the module's import).
        values: Dict[str, List[ast.expr]] = {}
        deferred: Set[int] = set()
        for node in (tree.body if not rel.startswith("meep_gpu/") else ()):
            names, lambdas = _deferred_lambdas(node)
            if lambdas:
                deferred.add(id(node))
                for name in names:
                    values.setdefault(name, []).extend(lambdas)
        aliases: Dict[str, Tuple[str, ...]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target, _dotted, _dirs = self._resolve(rel, 0, alias.name)
                    if target is not None:
                        aliases[alias.asname or alias.name.split(".")[0]] = (
                            ("module", target) if alias.asname else
                            ("module", self._resolve(rel, 0, alias.name.split(".")[0])[0] or target))
            elif isinstance(node, ast.ImportFrom):
                target, dotted, dirs = self._resolve(rel, node.level, node.module or "")
                if target is None:
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    sub = self.module_file(f"{dotted}.{alias.name}", dirs)
                    aliases[alias.asname or alias.name] = (
                        ("module", sub) if sub is not None else ("attr", target, alias.name))
        return _Symbols(functions, classes, aliases, values, frozenset(deferred))

    def _body(self, rel: str, node_name: str) -> List[ast.AST]:
        tree = self.tree(rel)
        if node_name in ("<module>", "<main>"):
            out: List[ast.AST] = []
            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if isinstance(statement, ast.ClassDef):
                    out.extend(s for s in statement.body
                               if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)))
                    continue
                if isinstance(statement, ast.If) and _is_main_guard(statement.test) \
                        and node_name == "<module>":
                    continue
                out.append(statement)
            return out
        symbols = self.symbols(rel)
        if node_name.startswith("<value:"):
            # A table's entries run when called; the reach is a lambda's body or the name.
            return [entry.body if isinstance(entry, ast.Lambda) else entry
                    for entry in symbols.values.get(node_name[len("<value:"):-1], ())]
        if "." in node_name:
            cls, method = node_name.split(".", 1)
            fn = symbols.classes.get(cls, {}).get(method)
        else:
            fn = symbols.functions.get(node_name)
        return list(fn.body) if fn is not None else []

    def _reach(self, rel: str, name: str) -> List[Tuple[str, str]]:
        symbols = self.symbols(rel)
        if name in symbols.functions:
            return [(rel, name)]
        if name in symbols.classes:
            return [(rel, f"{name}.{dunder}")
                    for dunder in ("__init__", "__post_init__", "__call__", "__enter__", "__exit__")
                    if dunder in symbols.classes[name]]
        if name in symbols.values:
            return [(rel, f"<value:{name}>")]
        lazy = self.lazy_names(rel).get(name)
        if lazy is not None:
            # A name the package binds on first use: the defining module loads, and the
            # attribute is reached there (``(file, "<module>")`` puts the file in the closure
            # even when the attribute is a constant with no body to reach).
            target, attribute = lazy
            return [(target, "<module>")] + self._reach(target, attribute)
        return []

    @lru_cache(maxsize=None)
    def _node_effects(self, rel: str, node_name: str
                      ) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...], Tuple[str, ...]]:
        """(files imported, nodes called, method names called) by one function's body."""
        files: List[str] = []
        nodes: List[Tuple[str, str]] = []
        methods: List[str] = []
        harness = not rel.startswith("meep_gpu/")
        symbols = self.symbols(rel)
        for statement in self._body(rel, node_name):
            # A table assignment's entries are not run by the assignment: they are the
            # ``<value:NAME>`` node, reached where code references NAME.
            skip: FrozenSet[int] = frozenset()
            if id(statement) in symbols.deferred:
                skip = frozenset(id(n) for lam in _deferred_lambdas(statement)[1]
                                 for n in ast.walk(lam))
            collector = _Collector(module_level=False, skip=skip)
            collector.visit(statement)
            for level, module, names in collector.imports:
                target, dotted, dirs = self._resolve(rel, level, module)
                if target is None:
                    continue
                files.append(target)
                files.extend(self._package_inits(dotted, dirs))
                for imported in names or ():
                    sub = self.module_file(f"{dotted}.{imported}", dirs)
                    if sub is not None:
                        files.append(sub)
                    else:
                        nodes.extend(self._reach(target, imported))
            here = Path(rel).parent.as_posix()
            for literal in collector.finish():
                candidate = f"{here}/{Path(literal).name}" if "/" not in literal else literal
                if self.exists(candidate):
                    files.append(candidate)
            # A REFERENCE counts as a call: a leg table, a callback or ``map(f, ...)`` runs
            # a function its body never calls by name.
            walked = [n for n in ast.walk(statement) if id(n) not in skip]
            called = {id(n.func) for n in walked if isinstance(n, ast.Call)}
            for node in walked:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    name = node.id
                    if name in symbols.functions or name in symbols.classes \
                            or name in symbols.values:
                        nodes.extend(self._reach(rel, name))
                    elif name in symbols.aliases and symbols.aliases[name][0] == "attr":
                        _kind, target, attr = symbols.aliases[name]
                        files.append(target)
                        nodes.extend(self._reach(target, attr))
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    receiver = node.value
                    if isinstance(receiver, ast.Name) and receiver.id in symbols.aliases \
                            and symbols.aliases[receiver.id][0] == "module":
                        target = symbols.aliases[receiver.id][1]
                        files.append(target)
                        nodes.extend(self._reach(target, node.attr))
                    elif harness and id(node) in called:
                        methods.append(node.attr)
        return tuple(files), tuple(nodes), tuple(sorted(set(methods)))

    def run_closure(self, script: str) -> FrozenSet[str]:
        return self.run_closure_with_witness(script)[0]

    def run_closure_with_witness(self, script: str
                                 ) -> Tuple[FrozenSet[str], Dict[str, Tuple[str, str]]]:
        """An upper bound on the record: the import closure plus what the reached calls import.

        The entry is the gate's module-level code with its ``__main__`` block, and its
        ``main``. A reached
        function contributes the modules it imports and the functions it calls or names: a
        bare name resolves to a same-file function or class (``__init__``), or to what the
        name was imported as; ``alias.f`` resolves through a module alias; a name that is
        only referenced (a leg table, a callback) counts as called; and a method call on any
        other receiver, inside a HARNESS file, reaches every same-named method of a harness
        class in the closure (a receiver's type is not read). A package function is followed
        only through a resolved name, never by method name. In a HARNESS file, a lambda
        that a module-level table defines (a tuple, list, set or dict display assigned to a
        name, the lambda nested only in displays), and a function such a table stores by
        name, are not run by the assignment: they are reached where reached code names the
        table, in its own file, through a module alias or by a from-import
        (:func:`_deferred_lambdas`). A package file's tables keep the rule above.
        """
        files: Set[str] = set()
        why: Dict[str, Tuple[str, str]] = {}
        reached: Set[Tuple[str, str]] = set()
        todo_files: List[Tuple[str, Tuple[str, str]]] = [(script, (script, "<entry>"))]
        # ``main`` is reached even when nothing in the file calls it by name: the Metal gates
        # hand themselves to ``metal_gate_runner``, which loads them by path and calls it.
        todo_nodes: List[Tuple[str, str]] = [(script, "<main>"), (script, "main")]
        method_index: Dict[str, Set[Tuple[str, str]]] = {}
        pending_methods: Set[str] = set()

        def add_file(rel: str, witness: Tuple[str, str]) -> None:
            if rel in files or not self.exists(rel):
                return
            files.add(rel)
            why[rel] = witness
            todo_files.extend((p, (rel, "<module>")) for p in self.parents(rel))
            todo_files.extend((e, (rel, "<module>")) for e in self.edges(rel, module_level=True))
            todo_nodes.append((rel, "<module>"))
            if not rel.startswith("meep_gpu/"):
                for cls, methods in self.symbols(rel).classes.items():
                    for name in methods:
                        method_index.setdefault(name, set()).add((rel, f"{cls}.{name}"))
                        if name in pending_methods:
                            todo_nodes.append((rel, f"{cls}.{name}"))

        while todo_files or todo_nodes:
            while todo_files:
                add_file(*todo_files.pop())
            if not todo_nodes:
                break
            rel, node_name = todo_nodes.pop()
            if (rel, node_name) in reached or not self.exists(rel):
                continue
            reached.add((rel, node_name))
            if rel not in files:
                todo_files.append((rel, (rel, node_name)))
            new_files, new_nodes, methods = self._node_effects(rel, node_name)
            todo_files.extend((f, (rel, node_name)) for f in new_files)
            todo_nodes.extend(new_nodes)
            for method in methods:
                pending_methods.add(method)
                for owner, qualified in method_index.get(method, ()):
                    todo_nodes.append((owner, qualified))
        kept = frozenset(k for k in files if k.startswith("meep_gpu/") or k.startswith(HARNESS + "/"))
        return kept, {k: why[k] for k in kept}

    def import_closure(self, script: str) -> FrozenSet[str]:
        seen: Set[str] = set()
        todo = [script]
        while todo:
            rel = todo.pop()
            if rel in seen or not self.exists(rel):
                continue
            seen.add(rel)
            todo.extend(self.parents(rel))
            todo.extend(self.edges(rel, module_level=True))
        return frozenset(k for k in seen if k.startswith("meep_gpu/") or k.startswith(HARNESS + "/"))


class _Symbols:
    """One file's top-level functions, classes (with their methods), import aliases, and the
    module-level tables whose lambdas run only when called (:func:`_deferred_lambdas`)."""

    def __init__(self, functions, classes, aliases, values=None, deferred=frozenset()):
        self.functions = functions
        self.classes = classes
        self.aliases = aliases
        #: table name -> its deferred entries; reached as the node ``<value:NAME>``.
        self.values = values or {}
        #: ``id()`` of the module-level statements that define them.
        self.deferred = deferred


def package_lazy_edges(resolver: Resolver) -> List[Tuple[str, str, str]]:
    """Every function-level import of a :data:`NOT_KERNEL` module in a production module.

    Read off the source, so ``test_gate_import_closure`` can hold :data:`PACKAGE_LAZY_EDGES`
    to it.
    """
    out = []
    for path in sorted((resolver.root / "meep_gpu").rglob("*.py")):
        rel = _rel(path, resolver.root)
        if rel in NOT_KERNEL or path.name == "conftest.py" or path.name.startswith("test_"):
            continue
        tree = resolver.tree(rel)
        _search, package = resolver._search_and_package(rel)  # noqa: SLF001
        for function in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(function):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                targets: List[str] = []
                if isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                else:
                    if node.level:
                        base = package or ""
                        for _ in range(node.level - 1):
                            base = base.rpartition(".")[0]
                        dotted = ".".join(p for p in (base, node.module or "") if p)
                    else:
                        dotted = node.module or ""
                    targets = [dotted] + [f"{dotted}.{a.name}" for a in node.names]
                for dotted in targets:
                    found = resolver.module_file(dotted, [""])
                    if found in NOT_KERNEL:
                        out.append((rel, function.name, found))
    return sorted(set(out))


def gate_scripts(root: Path = REPO_ROOT) -> List[str]:
    """Every gate and probe script of the harness, repo-relative."""
    base = root / HARNESS
    return sorted(_rel(p, root) for pattern in ("gate_*.py", "probe_*.py")
                  for p in base.glob(pattern))


def measured_import_closure(script: str, root: Path = REPO_ROOT,
                            python: str = sys.executable, timeout: float = 300.0) -> Dict[str, object]:
    """Import ``script`` as a module (never as ``__main__``) in a FRESH interpreter.

    The child's ``sys.path`` and ``sys.argv`` are what ``python <script>`` with the checkout
    on ``PYTHONPATH`` gives, and it reports the repository files left in ``sys.modules`` by
    the rule ``gate_provenance.provenance`` applies. Returns ``{"files": [...], "error":
    str|None}``; an import that raises is reported, not hidden. The child sees the libraries
    this process sees: the same interpreter, this process's ``PYTHONPATH`` after the
    checkout, and ``-S`` / ``-s`` when this process runs with them.
    """
    child = (
        "import importlib.util, json, os, sys\n"
        "root, script = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])\n"
        "sys.path[:0] = [os.path.dirname(script), root]\n"
        "sys.argv = [script]\n"
        "error = None\n"
        "try:\n"
        "    spec = importlib.util.spec_from_file_location('_closure_gate_', script)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules['_closure_gate_'] = module\n"
        "    spec.loader.exec_module(module)\n"
        "except BaseException as exc:\n"
        "    error = f'{type(exc).__name__}: {exc}'[:400]\n"
        "files = set()\n"
        "for m in list(sys.modules.values()):\n"
        "    p = getattr(m, '__file__', None)\n"
        "    if not p or not p.endswith('.py'):\n"
        "        continue\n"
        "    r = os.path.realpath(p)\n"
        "    if r.startswith(root + os.sep):\n"
        "        k = os.path.relpath(r, root).replace(os.sep, '/')\n"
        "        if k.startswith('meep_gpu/') or k.startswith('parity/meep_gpu/'):\n"
        "            files.add(k)\n"
        "print(json.dumps({'files': sorted(files), 'error': error}))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEEP_GPU_")}
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(root), *inherited]))
    flags = ["-B"] + (["-S"] if sys.flags.no_site else []) + (["-s"] if sys.flags.no_user_site else [])
    done = subprocess.run([python, *flags, "-c", child, str(root), str(root / script)],
                          capture_output=True, text=True, timeout=timeout, env=env,
                          cwd=str((root / script).parent))
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {"files": [], "error": f"no report (rc {done.returncode}): {done.stderr[-400:]}"}
    return json.loads(lines[-1])


#: The rebind ledgers whose curated pins :func:`unimported_pins` reads, by table.
REBIND_LEDGERS: Mapping[str, str] = {
    "metal": "meep_gpu/metal_kernels/fingerprints.json",
    "triton": "meep_gpu/triton_kernels/fingerprints.json",
    "cuda": "meep_gpu/cuda_kernels/fingerprints.json",
}

_METAL_WELD = re.compile(r"^metal_(.+)_device_gate$")
_GATE_SCRIPT = re.compile(r"^parity/meep_gpu/(gate|probe)_[^/]+\.py$")


def eager_root_modules(root: Path = REPO_ROOT) -> FrozenSet[str]:
    """The modules ``import meep_gpu`` loaded before the package root bound its names lazily.

    Read off the root's own source: ``_SUBMODULES`` keeps exactly the submodules the eager
    root left bound on the package (they still resolve as attributes), so
    ``meep_gpu/<name>.py`` for each name is that closure, less the root itself, which every
    gate's run closure holds as the parent package.
    """
    tree = ast.parse((root / "meep_gpu" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        targets = [t.id for t in getattr(node, "targets", ()) if isinstance(t, ast.Name)]
        if isinstance(node, ast.Assign) and targets == ["_SUBMODULES"]:
            value = node.value
            if isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            names = ast.literal_eval(value)
            return frozenset(f"meep_gpu/{name}.py" for name in names)
    raise ValueError(f"{root / 'meep_gpu' / '__init__.py'} assigns no _SUBMODULES")


def weld_gate_scripts(table: str, key: str, entry: Mapping[str, object]) -> Tuple[str, ...]:
    """The gate script(s) whose record a weld of ``table`` is rebound from. Empty: not a weld.

    Metal: ``metal_<family>_device_gate`` is gated by ``gate_metal_<family>.py`` (the runner
    writes ``results/metal_<family>_<stamp>/``, which ``rebind_metal_welds.py`` reads); the
    ledger's ``driver_dispatch`` and ``metal_kernels`` rows are not welds. Triton and CUDA: the
    ``gate_*``/``probe_*`` scripts the weld pins (a weld pins the script that gated it; where it
    pins several, the union of their run closures is the bound, which can only keep a pin).
    """
    if key.startswith("_") or not isinstance(entry, Mapping):
        return ()
    pins = entry.get("source_sha256")
    if not isinstance(pins, Mapping) or not pins:
        return ()
    if table == "metal":
        match = _METAL_WELD.match(key)
        return (f"{HARNESS}/gate_metal_{match.group(1)}.py",) if match else ()
    if table in ("triton", "cuda"):
        return tuple(sorted(name for name in pins if _GATE_SCRIPT.match(str(name))))
    raise ValueError(f"unknown rebind table {table!r}; one of {sorted(REBIND_LEDGERS)}")


def gates_run_closure(resolver: "Resolver", gates: Sequence[str]) -> Optional[FrozenSet[str]]:
    """The union of the gates' run closures; None when a gate is not in the checkout."""
    if not gates or not all(resolver.exists(gate) for gate in gates):
        return None
    held: Set[str] = set()
    for gate in gates:
        held |= resolver.run_closure(gate)
    return frozenset(held)


def unimported_pins(table: str, root: Path = REPO_ROOT,
                    ledger: Optional[Mapping[str, object]] = None,
                    resolver: Optional["Resolver"] = None
                    ) -> Dict[Tuple[str, str], Tuple[str, ...]]:
    """``(weld key, pinned path) -> gate scripts`` for every curated pin the rule retires.

    A pin is retired from a weld only when BOTH hold: the file is one of
    :func:`eager_root_modules` (so its place in the pin set came from the import record the
    eager root inflated), and it lies outside the run closure of the weld's gate (an upper
    bound on the gate's next record, :meth:`Resolver.run_closure`). A weld whose gate cannot
    be named, or whose gate is not in the checkout, contributes nothing. ``ledger`` defaults
    to the checkout's ledger of ``table``.
    """
    root = Path(root)
    resolver = resolver or Resolver(root)
    if ledger is None:
        ledger = json.loads((root / REBIND_LEDGERS[table]).read_text(encoding="utf-8"))
    eager = eager_root_modules(root)
    rows: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for key in sorted(ledger):
        entry = ledger[key]
        gates = weld_gate_scripts(table, key, entry)
        held = gates_run_closure(resolver, gates)
        if held is None:
            continue
        for name in sorted(entry["source_sha256"]):
            if name in eager and name not in held:
                rows[(key, name)] = gates
    return rows


def unimported_pins_back(rows: Mapping[Tuple[str, str], Sequence[str]], root: Path = REPO_ROOT,
                         resolver: Optional["Resolver"] = None) -> List[Tuple[str, str, str]]:
    """``(key, path, why)`` for every row the rule no longer licenses on this checkout.

    A row stops holding when its gate's run closure holds the file again, when a gate it names
    is not in the checkout, or when the file is no longer one of the eager root's modules.
    The rebind tools refuse on any such row before they drop a pin.
    """
    resolver = resolver or Resolver(Path(root))
    eager = eager_root_modules(Path(root))
    out: List[Tuple[str, str, str]] = []
    for (key, name), gates in sorted(rows.items()):
        held = gates_run_closure(resolver, tuple(gates))
        if held is None:
            out.append((key, name, f"a gate it names is not in the checkout ({list(gates)})"))
        elif name in held:
            out.append((key, name, f"the run closure of {list(gates)} holds it again"))
        elif name not in eager:
            out.append((key, name, "it is not one of the modules the eager package root loaded"))
    return out


def _hits(closure: Iterable[str]) -> List[str]:
    return sorted(set(closure) & set(NOT_KERNEL))


def report(root: Path = REPO_ROOT) -> Dict[str, Dict[str, object]]:
    resolver = Resolver(root)
    out: Dict[str, Dict[str, object]] = {}
    for script in gate_scripts(root):
        imported = resolver.import_closure(script)
        ran, why = resolver.run_closure_with_witness(script)
        out[script] = {"import_n": len(imported), "run_n": len(ran),
                       "import_hits": _hits(imported), "run_hits": _hits(ran),
                       "run_witness": {hit: list(why[hit]) for hit in _hits(ran)},
                       "import_closure": sorted(imported), "run_closure": sorted(ran)}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--root", default=str(REPO_ROOT))
    rep.add_argument("--json", default="")
    bill = sub.add_parser("bill")
    bill.add_argument("file")
    bill.add_argument("--root", default=str(REPO_ROOT))
    pins = sub.add_parser("unimported-pins")
    pins.add_argument("table", choices=sorted(REBIND_LEDGERS))
    pins.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)
    if args.command == "unimported-pins":
        # The rows as the tool's literal spells them, one per line, for review.
        found = unimported_pins(args.table, Path(args.root))
        for (key, name), gates in found.items():
            print(f"    ({key!r}, {name!r}): {tuple(gates)!r},", flush=True)
        print(f"{len(found)} pin(s) on {len({key for key, _ in found})} weld(s) of "
              f"{REBIND_LEDGERS[args.table]} lie outside their gate's run closure and are "
              f"modules of the eager package root", flush=True)
        return 0
    rows = report(Path(args.root))
    if args.command == "report":
        for script, row in rows.items():
            print(f"{script}: import {row['import_n']} run {row['run_n']} "
                  f"import_hits {row['import_hits']} run_hits {row['run_hits']}", flush=True)
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n",
                                       encoding="utf-8")
        return 0
    held = [s for s, row in rows.items() if args.file in row["run_closure"]]
    for script in held:
        print(script, flush=True)
    print(f"{len(held)} of {len(rows)} gate and probe scripts hold {args.file} in their run closure",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
