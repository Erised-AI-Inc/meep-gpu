#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""The device text each kernel module of the three tables actually compiles, for the
bit-parity coverage board -- emitted through the package's own builders, with no device.

    python parity/meep_gpu/parity_board_emit.py --table cuda [--out FILE] [--dump-dir DIR]

``build_parity_board.py`` reads its columns (a) B select and (b) MEEP-order A form off
this text (:func:`emitted_witnesses`, one child process per table). Most kernel modules do
not spell their device arithmetic in their own file: CUDA lifters derive theirs from the
roots by anchored replacement, Metal families splice the base curl and constitutive bodies
through templates, and Triton kernels inline ``@triton.jit`` helpers of other modules. The
text a module's own file spells is therefore the wrong witness for most rows; the text its
builder hands the compiler is the right one.

WHAT "EMITTED" MEANS, PER TABLE (each UNIT is one compiled text):

* **CUDA** -- the exact string ``cp.RawKernel`` is constructed with. ``cupy`` is replaced,
  in the child process only, by NumPy wearing CuPy's name (the stand-in the table's own
  host tests use) whose ``RawKernel`` RECORDS its source and compiles nothing; NVRTC is
  never reached, and every compile memo is cleared before each call so a kernel two
  modules share is recorded for both. A module is driven through ``device_sources()``
  when it publishes one (the shipped text by kernel name, which is what its record blocks
  pin), else through its own compile seam ``_get_kernel`` over a canonical sweep of its
  parameters, else through the emitters or calls its recipe names. A multi-launch product
  whose later launch is another family's certified kernel names that seam in its recipe
  (``with``), refused unless the module imports its owner.
* **Metal** -- the MSL string the module's emitter returns after template splicing. The
  child process refuses every import of ``torch``, so no MPS device can be reached and no
  shader is compiled. A module is driven through its own ``enumerate_*sources`` when it
  has one, else its own ``specialisations()`` (a spread sample, when its recipe names the
  emitter), else its public ``*_source`` emitters over the canonical sweep. A module that
  plans another family's kernels names that family in its recipe (``delegate``), refused
  unless it imports it. ``refuted_*`` emitters are alternatives a gate refuted and are
  never read.
* **Triton** -- the ``@triton.jit`` program: each kernel with every ``@jit`` function it
  calls, from any module of the table, inlined transitively (Triton inlines them; the
  helper boundary is not a rounding boundary). Read statically from the package's source,
  so it needs no Triton; TTIR and PTX would need the toolchain and are not read. The unit's
  text (and digest) is each member's source as written, which is what Triton reads; its
  verdicts come from the syntax tree. A module that launches another module's kernel and
  has none of its own names it in its recipe (``launches``, ``also_launches``), and the
  recipe is refused unless the module's own text names (or imports) it.

THE CANONICAL SWEEPS are data (``parity_board_emit.json``): per table, a small domain per
emitter parameter -- real PML 3-D (every axis metallic, both half-steps) and the
family's own minimal case (the periodic triple, the first combination its emitter
accepts), wall clears and Bloch phases both ways, both complex-multiply arms -- and per
module the overrides its emitter needs (the folded families' mirror codes, the
off-diagonal wall masks). A parameter with no declared domain is a refusal, never a guess
(the stance of ``metal_kernels/corpus.py``). An emitter's own ``ValueError`` on a
combination is that combination's absence, counted; a sweep larger than ``cap`` is read
at ``cap`` points spread evenly.

REFUSALS, BY NAME. A module whose emitter raises on this tree (an anchor that no longer
matches its re-cut base) is refused with the exception text: no compiled text exists to
read. A module with no emit route at all (a text transform, a host scan, a null arm) is
reported as such, and the board falls back to the v1 source-text witness for it, saying so
in the cell.

Each unit is judged here, in the child, and only digests and verdict inputs cross back to
the board: the chunk-record read, the PML-form markers, the curl and constitutive sites in
each grouping (:mod:`parity_board_text`), the field-word PML forms inside and outside a
per-cell select, and any helper called with an argument count none of its definitions
takes (:func:`helper_arity_problems`: a call site a lift did not re-cut, a text NVRTC or
the Metal compiler would refuse), and the record read's binding
(:func:`record_binding_problems`: a ``cb`` the reading function never binds, or a
``ChunkBounds`` the text never defines; either is a text the compiler would refuse).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.abc
import inspect
import itertools
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent

SCHEMA = "meep_gpu.parity_board_emitted"
RECIPES_SCHEMA = "meep_gpu.parity_board_emit_recipes"
VERSION = 1
RECIPES_FILE = "parity/meep_gpu/parity_board_emit.json"
ROWS_FILE = "parity/meep_gpu/parity_board_rows.json"
TABLES = ("cuda", "triton", "metal")
PACKAGE = {"cuda": "meep_gpu.cuda_kernels", "triton": "meep_gpu.triton_kernels",
           "metal": "meep_gpu.metal_kernels"}
#: Row kinds with a device form (the others are never emitted).
EMITTED_KINDS = ("kernel", "a_only")

#: How each table's compiled text reads the chunk record (comment-free emitted text).
#: Stricter than the source-text witnesses: an emitted text no longer carries the
#: Python calls that emit the read, only the read itself.
EMITTED_RECORD_READ: Dict[str, Tuple[str, ...]] = {
    "cuda": (r"\bcb\s*\[",),
    "triton": (r"\btl\.load\s*\(\s*cb\b",),
    "metal": (r"\bcb\s*\.\s*cb\w+",),
}

#: A Triton unit STORES when its program writes device memory: a kernel, not a helper.
_TRITON_STORE = re.compile(r"\btl\.store\s*\(")


class EmitRefusal(Exception):
    """A named reason a module's device text cannot be emitted on this host or tree."""


class NoEmitRoute(Exception):
    """The module has no emit route: the board falls back to its own source text."""


# ===========================================================================
# Data
# ===========================================================================


def load_recipes(root: Path) -> Dict[str, Any]:
    path = root / RECIPES_FILE
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != RECIPES_SCHEMA or document.get("version") != VERSION:
        raise SystemExit(f"{RECIPES_FILE} is not a version-{VERSION} {RECIPES_SCHEMA} document")
    return document


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


# ===========================================================================
# The child process: no device reachable
# ===========================================================================


class _Blocked(importlib.abc.MetaPathFinder):
    """Refuses every import of a device library in the emitting process."""

    BLOCKED = ("torch", "triton", "cupy_backends", "cupyx", "pycuda", "cuda")

    def find_spec(self, name, path, target=None):  # noqa: D401 - importlib protocol
        top = name.split(".", 1)[0]
        if top in self.BLOCKED:
            raise ImportError(f"{top} is not importable while emitting device text: the "
                              f"board's witnesses run with no device library")
        return None


class _Recorded:
    """What a recorded ``RawKernel`` construction returns: nothing launchable."""

    def __init__(self, code: str, name: str):
        self.code, self.name = code, name

    def __call__(self, *args, **kwargs):  # pragma: no cover - never launched
        raise EmitRefusal("a recorded kernel cannot be launched")


class _Recorder:
    def __init__(self):
        self.captured: List[Tuple[str, str]] = []

    def mark(self, *modules: Any) -> int:
        """Forget every compiled kernel, so the next call constructs (and records) every
        kernel it needs: the shared memo and each module's own memo would otherwise
        answer a second module that compiles the same text without a construction."""
        try:
            from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
            compile_cache.clear_kernel_cache()
        except Exception:  # noqa: BLE001 - a tree without the memo has nothing to clear
            pass
        for module in modules:
            for name in ("_clear_kernel_cache", "clear_kernel_cache"):
                clear = getattr(module, name, None)
                if callable(clear) and _no_required(clear):
                    try:
                        clear()
                    except Exception:  # noqa: BLE001
                        pass
        return len(self.captured)

    def raw_kernel(self, code, name, options=(), **_kwargs):
        self.captured.append((str(code), str(name)))
        return _Recorded(str(code), str(name))

    def raw_module(self, code=None, options=(), name_expressions=None, **_kwargs):
        self.captured.append((str(code), "RawModule"))
        return _Recorded(str(code), "RawModule")


def _install_cupy_recorder(recorder: _Recorder) -> None:
    """``cupy`` as NumPy wearing CuPy's name, whose kernel constructors only record."""
    import numpy  # noqa: PLC0415

    class _Cupy(types.ModuleType):
        def __init__(self):
            super().__init__("cupy")
            self.RawKernel = recorder.raw_kernel
            self.RawModule = recorder.raw_module

        def __getattr__(self, item):
            return getattr(numpy, item)

    sys.modules["cupy"] = _Cupy()


def _install_blocks() -> None:
    sys.meta_path.insert(0, _Blocked())
    for name in list(sys.modules):
        if name.split(".", 1)[0] in _Blocked.BLOCKED:
            del sys.modules[name]


# ===========================================================================
# Parameter sweeps
# ===========================================================================


def _resolve_value(spec: Any, module: Any, constants: Mapping[str, Any]) -> Any:
    """One declared value: ``{"const": NAME}`` reads the table's constants, anything else
    is a literal (lists read as tuples)."""
    if isinstance(spec, dict) and set(spec) == {"const"}:
        if spec["const"] not in constants:
            raise EmitRefusal(f"the recipes name an unknown constant {spec['const']!r}")
        return constants[spec["const"]]
    if isinstance(spec, dict) and set(spec) == {"dict"}:  # a mapping with non-string keys
        return {_resolve_value(k, module, constants): _resolve_value(v, module, constants)
                for k, v in spec["dict"]}
    if isinstance(spec, list):
        return tuple(_resolve_value(v, module, constants) for v in spec)
    return spec


def _domain(name: str, module: Any, overrides: Mapping[str, Any],
            canonical: Mapping[str, Any], constants: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    """The declared values of one emitter parameter, or None when none is declared.

    A domain is a list of values, or ``{"attr": [NAMES], "keys": bool, "else": [...]}``:
    the first of the module's own attributes that exists (its keys when ``keys``), which
    is how a module's ``CERTIFIED_KERNELS`` or ``KERNEL_FOR_SUB_STEP`` names its kernels.
    """
    spec = overrides.get(name, canonical.get(name))
    if spec is None:
        return None
    if isinstance(spec, dict) and "attr" in spec:
        values: List[Any] = []
        for attribute in spec["attr"]:
            found = getattr(module, attribute, None)
            if found is None:
                continue
            items = list(found.keys()) if isinstance(found, Mapping) else list(found)
            values.extend(v for v in items if v not in values)
            if not spec.get("union"):
                break
        if not values:
            fallback = spec.get("else")
            if fallback is None:
                return None
            values = [_resolve_value(v, module, constants) for v in fallback]
        return tuple(values)
    return tuple(_resolve_value(v, module, constants) for v in spec)


def _sweep(function: Callable[..., Any], module: Any, overrides: Mapping[str, Any],
           canonical: Mapping[str, Any], constants: Mapping[str, Any],
           cap: int) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[str]]:
    """The canonical combinations of ``function``'s parameters, as (label, kwargs).

    A parameter with a default and no declared domain keeps its default. A parameter with
    neither is a refusal: guessing would narrow the sweep silently.
    """
    signature = inspect.signature(function)
    names: List[str] = []
    domains: List[Tuple[Any, ...]] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        values = _domain(name, module, overrides, canonical, constants)
        if values is None:
            if parameter.default is parameter.empty:
                raise EmitRefusal(f"{function.__name__} takes {name!r}, which has no "
                                  f"declared canonical domain")
            continue
        names.append(name)
        domains.append(values)
    combos = list(itertools.product(*domains)) if names else [()]
    notes = []
    if len(combos) > cap:
        notes.append(f"{len(combos)} combinations, read {cap} spread evenly")
        step = (len(combos) - 1) / (cap - 1)
        combos = [combos[round(i * step)] for i in range(cap)]
    out = []
    for combo in combos:
        kwargs = dict(zip(names, combo))
        label = ",".join(f"{k}={_short(v)}" for k, v in kwargs.items()) or "()"
        out.append((label, kwargs))
    return out, notes


def _text_units(value: Any, label: str) -> List[Tuple[str, str]]:
    """The device texts an emitter call returned: a string, or the strings of a mapping or
    a sequence."""
    if isinstance(value, str):
        return [(label, value)]
    if isinstance(value, Mapping):
        out = []
        for key, item in value.items():
            out.extend(_text_units(item, f"{label}:{key}"))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for index, item in enumerate(value):
            out.extend(_text_units(item, f"{label}[{index}]"))
        return out
    return []


def _defined_here(module: Any, name: str) -> bool:
    value = getattr(module, name, None)
    return callable(value) and getattr(value, "__module__", None) == module.__name__


def _no_required(function: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in signature.parameters.values())


#: Exceptions an emitter raises to say "this combination is not one I emit".
_ABSENT = (ValueError, KeyError)


def _call_recipe(module: Any, call: Sequence[Any], constants: Mapping[str, Any]) -> Any:
    name, *arguments = call
    function = getattr(module, name, None)
    if not callable(function):
        raise EmitRefusal(f"the recipe calls {name}, which the module does not define")
    resolved = []
    for argument in arguments:
        if isinstance(argument, dict) and set(argument) == {"module_attr"}:
            owner, attribute = argument["module_attr"].rsplit(".", 1)
            resolved.append(getattr(importlib.import_module(owner), attribute))
        elif isinstance(argument, dict) and set(argument) == {"module_call"}:
            dotted, *inner = argument["module_call"]
            owner, attribute = dotted.rsplit(".", 1)
            resolved.append(_call_recipe(importlib.import_module(owner),
                                         [attribute, *inner], constants))
        else:
            resolved.append(_resolve_value(argument, module, constants))
    return function(*resolved)


# ===========================================================================
# CUDA
# ===========================================================================


def emit_cuda(module_name: str, spec: Mapping[str, Any], canonical: Mapping[str, Any],
              recorder: _Recorder, cap: int) -> Tuple[str, List[Tuple[str, str]], List[str]]:
    """:func:`_emit_cuda_own`, plus the launches the module makes through ANOTHER module's
    compile seam (the recipe's ``with``: a multi-launch product whose second launch is a
    certified kernel of another family). Each is refused unless the module imports the
    owner of the seam it names."""
    route, units, notes = _emit_cuda_own(module_name, spec, canonical, recorder, cap)
    extra = spec.get("with") or []
    if extra:
        own = (_HERE.parent.parent / "meep_gpu" / "cuda_kernels" / f"{module_name}.py")
        source = own.read_text(encoding="utf-8") if own.is_file() else ""
        for call in extra:
            dotted, *arguments = call
            owner_path, function_name = dotted.rsplit(".", 1)
            owner = owner_path.rsplit(".", 1)[1]
            if not re.search(r"\bimport [^\n]*\b" + owner + r"\b|\bfrom \." + owner + r" import\b",
                             source):
                raise EmitRefusal(f"the recipe says it also launches {dotted}, and the module "
                                  f"does not import {owner}")
            module = importlib.import_module(owner_path)
            before = recorder.mark(module)
            label = f"with {owner}.{function_name}{tuple(arguments)}"
            try:
                value = _call_recipe(module, [function_name, *arguments], {})
            except Exception as error:  # noqa: BLE001
                raise EmitRefusal(f"{label} raises on this tree: {type(error).__name__}: "
                                  f"{str(error)[:300]}") from error
            units.extend(_text_units(value, label))
            units.extend((f"{label}:RawKernel:{name}", code)
                         for code, name in recorder.captured[before:])
        route += " + " + ", ".join(c[0].rsplit(".", 2)[1] + "." + c[0].rsplit(".", 1)[1]
                                   for c in extra)
    return route, units, notes


def _emit_cuda_own(module_name: str, spec: Mapping[str, Any], canonical: Mapping[str, Any],
                   recorder: _Recorder, cap: int) -> Tuple[str, List[Tuple[str, str]], List[str]]:
    """(route, [(label, text)], notes) for one CUDA module's own compile path.

    Routes, in order: the recipe's ``none`` (no emit route, named); ``device_sources()``
    when the module defines it with no required argument (or the recipe allows an alias
    of another module's, for a module that IS that module's kernel); the recipe's
    ``sweep`` of named emitters; the recipe's explicit ``calls``; the module's compile
    seam (``_get_kernel``, or the recipe's ``seam``) over the canonical sweep. Every
    ``cp.RawKernel`` construction during a call is recorded as a unit, beside any text
    the call returns.
    """
    if spec.get("route") == "none":
        raise NoEmitRoute(spec.get("reason", "the recipe declares no emit route"))
    try:
        module = importlib.import_module(f"{PACKAGE['cuda']}.{module_name}")
    except Exception as error:  # noqa: BLE001 - the refusal names it
        raise EmitRefusal(f"the module does not import with no device: "
                          f"{type(error).__name__}: {error}") from error
    route = spec.get("route")
    overrides = spec.get("domains", {})
    units: List[Tuple[str, str]] = []
    notes: List[str] = []

    def guarded(label: str, thunk: Callable[[], Any]) -> Any:
        try:
            return thunk()
        except _ABSENT:
            raise
        except EmitRefusal:
            raise
        except Exception as error:  # noqa: BLE001
            raise EmitRefusal(f"{label} raises on this tree: {type(error).__name__}: "
                              f"{str(error)[:300]}") from error

    def captured_since(before: int, label: str) -> List[Tuple[str, str]]:
        return [(f"{label}:RawKernel:{name}", code) for code, name in recorder.captured[before:]]

    sources = getattr(module, "device_sources", None)
    if route in (None, "device_sources") and callable(sources) and _no_required(sources) \
            and (_defined_here(module, "device_sources") or spec.get("allow_alias")):
        result = guarded("device_sources()", sources)
        units = _text_units(result, "device_sources()")
        if units:
            alias = "" if _defined_here(module, "device_sources") else \
                f" (an alias of {getattr(sources, '__module__', '?')})"
            return "device_sources()" + alias, units, notes
    if route == "sweep":
        for name in spec["functions"]:
            function = getattr(module, name, None)
            if not callable(function):
                raise EmitRefusal(f"the recipe sweeps {name}, which the module does not define")
            combos, sweep_notes = _sweep(function, module, overrides, canonical, {}, cap)
            notes.extend(sweep_notes)
            absent = 0
            for label, kwargs in combos:
                before = recorder.mark(module)
                try:
                    value = guarded(f"{name}({label})", lambda f=function, k=kwargs: f(**k))
                except _ABSENT:
                    absent += 1
                    continue
                units.extend(_text_units(value, f"{name}({label})"))
                units.extend(captured_since(before, f"{name}({label})"))
            if absent:
                notes.append(f"{name}: {absent} of {len(combos)} combinations refused by the emitter")
            if absent == len(combos):
                raise EmitRefusal(f"{name} refuses every canonical combination")
        return "sweep: " + ", ".join(spec["functions"]), units, notes
    if route == "calls":
        for index, config in enumerate(spec["calls"]):
            label = config.get("label", f"call{index}")
            before = recorder.mark(module)
            value = guarded(f"{config['call'][0]}()",
                            lambda c=config: _call_recipe(module, c["call"], {}))
            units.extend(_text_units(value, label))
            units.extend(captured_since(before, label))
        return "calls: " + ", ".join(sorted({c["call"][0] for c in spec["calls"]})), units, notes
    seam = spec.get("seam", "_get_kernel")
    if route in (None, "get_kernel") and _defined_here(module, seam):
        function = getattr(module, seam)
        combos, notes = _sweep(function, module, overrides, canonical, {}, cap)
        absent = 0
        for label, kwargs in combos:
            before = recorder.mark(module)
            try:
                guarded(f"{seam}({label})", lambda k=kwargs: function(**k))
            except _ABSENT:
                absent += 1
                continue
            units.extend(captured_since(before, f"{seam}({label})"))
        if absent:
            notes.append(f"{absent} of {len(combos)} combinations refused by the emitter")
        if not units and absent == len(combos):
            raise EmitRefusal(f"{seam} refuses every canonical combination "
                              f"({absent} of {len(combos)})")
        return seam, units, notes
    raise NoEmitRoute("no device_sources(), no compile seam and no recipe")


# ===========================================================================
# Metal
# ===========================================================================


def _metal_constants() -> Dict[str, Any]:
    from meep_gpu.metal_kernels import templates  # noqa: PLC0415

    constants = {name: getattr(templates, name) for name in ("PERIODIC", "METALLIC",
                                                             "CONTRACT_OFF")}
    from meep_gpu.metal_kernels import symmetry  # noqa: PLC0415

    constants["MIRROR_METALLIC"] = symmetry.CODE_MIRROR_METALLIC
    constants["MIRROR_PERIODIC"] = symmetry.CODE_MIRROR_PERIODIC
    try:
        from meep_gpu.metal_kernels import cylindrical_complex  # noqa: PLC0415
        constants["M_ONE"] = cylindrical_complex.M_ONE
        constants["M_MANY"] = cylindrical_complex.M_MANY
    except Exception:  # noqa: BLE001 - absent on a tree without the family
        pass
    return constants


_METAL_SKIP = re.compile(r"^(refuted_|_)")


def emit_metal(module_name: str, spec: Mapping[str, Any], canonical: Mapping[str, Any],
               constants: Mapping[str, Any], cap: int) -> Tuple[str, List[Tuple[str, str]], List[str]]:
    if spec.get("route") == "none":
        raise NoEmitRoute(spec.get("reason", "the recipe declares no emit route"))
    try:
        module = importlib.import_module(f"{PACKAGE['metal']}.{module_name}")
    except Exception as error:  # noqa: BLE001
        raise EmitRefusal(f"the module does not import with no device: "
                          f"{type(error).__name__}: {error}") from error
    overrides = spec.get("domains", {})
    units: List[Tuple[str, str]] = []
    notes: List[str] = []

    def call(label: str, function: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
        try:
            return function(**kwargs)
        except _ABSENT:
            raise
        except Exception as error:  # noqa: BLE001
            raise EmitRefusal(f"{label} raises on this tree: {type(error).__name__}: "
                              f"{str(error)[:300]}") from error

    def run(function: Callable[..., Any], route: str) -> None:
        combos, sweep_notes = _sweep(function, module, overrides, canonical, constants, cap)
        notes.extend(sweep_notes)
        absent = 0
        for label, kwargs in combos:
            try:
                value = call(f"{function.__name__}({label})", function, kwargs)
            except _ABSENT:
                absent += 1
                continue
            units.extend(_text_units(value, f"{function.__name__}({label})"))
        if absent:
            notes.append(f"{function.__name__}: {absent} of {len(combos)} combinations "
                         f"refused by the emitter")
        if absent and absent == len(combos):
            raise EmitRefusal(f"{function.__name__} refuses every canonical combination "
                              f"({absent} of {len(combos)})")

    if spec.get("delegate"):
        delegate = spec["delegate"]
        own = inspect.getsource(module)
        if not re.search(r"\bfrom \. import [^\n]*\b" + delegate + r"\b|\bfrom \." + delegate
                         + r" import\b", own):
            raise EmitRefusal(f"the recipe says it delegates to {delegate}, which the module "
                              f"does not import")
        target = importlib.import_module(f"{PACKAGE['metal']}.{delegate}")
        for name in spec["emitters"]:
            function = getattr(target, name, None)
            if not callable(function):
                raise EmitRefusal(f"the recipe names {delegate}.{name}, which does not exist")
            module_for_domains = target
            combos, sweep_notes = _sweep(function, module_for_domains, overrides, canonical,
                                         constants, cap)
            notes.extend(sweep_notes)
            absent = 0
            for label, kwargs in combos:
                try:
                    value = call(f"{delegate}.{name}({label})", function, kwargs)
                except _ABSENT:
                    absent += 1
                    continue
                units.extend(_text_units(value, f"{delegate}.{name}({label})"))
            if absent == len(combos):
                raise EmitRefusal(f"{delegate}.{name} refuses every canonical combination")
        return f"delegates to {delegate}: " + ", ".join(spec["emitters"]), units, notes
    names = spec.get("emitters")
    if names is None:
        enumerators = [n for n in vars(module) if re.match(r"^enumerate_\w*sources$", n)
                       and _defined_here(module, n)]
        if enumerators:
            for name in sorted(enumerators):
                run(getattr(module, name), "enumerate")
            return "enumerate: " + ", ".join(sorted(enumerators)), units, notes
        if _defined_here(module, "specialisations") and spec.get("specialised"):
            emitter = getattr(module, spec["specialised"])
            items = list(module.specialisations())
            take = items if len(items) <= cap else [
                items[round(i * (len(items) - 1) / (cap - 1))] for i in range(cap)]
            if len(items) > cap:
                notes.append(f"specialisations(): {len(items)}, read {cap} spread evenly")
            for item in take:
                label = f"{spec['specialised']}{_short(item)}"
                units.extend(_text_units(call(label, emitter, dict(zip(
                    list(inspect.signature(emitter).parameters)[:len(item)], item))), label))
            return f"specialisations() through {spec['specialised']}", units, notes
        names = sorted(n for n in vars(module) if n.endswith("_source")
                       and not _METAL_SKIP.match(n) and _defined_here(module, n))
    if not names:
        raise NoEmitRoute("no enumerate_*sources, no public *_source emitter, no recipe")
    for name in names:
        run(getattr(module, name), "emitter")
    return "emitters: " + ", ".join(names), units, notes


# ===========================================================================
# Triton (static: the @jit programs, helpers inlined)
# ===========================================================================


def _is_jit(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", None) or ""
        if name.endswith("jit"):
            return True
    return False


class TritonPackage:
    """Every module of the Triton table, parsed once: its @jit functions and imports."""

    def __init__(self, root: Path):
        self.directory = root / PACKAGE["triton"].replace(".", "/")
        self.trees: Dict[str, ast.Module] = {}
        self.sources: Dict[str, str] = {}
        self.jit: Dict[str, Dict[str, List[ast.FunctionDef]]] = {}
        self.imports: Dict[str, Dict[str, Tuple[str, Optional[str]]]] = {}
        for path in sorted(self.directory.glob("*.py")):
            if path.name.startswith("test_"):
                continue
            stem = path.stem
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            self.trees[stem] = tree
            self.sources[stem] = path.read_text(encoding="utf-8")
            jit: Dict[str, List[ast.FunctionDef]] = {}
            imports: Dict[str, Tuple[str, Optional[str]]] = {}
            for node in ast.walk(tree):
                if _is_jit(node):
                    jit.setdefault(node.name, []).append(node)
                if isinstance(node, ast.ImportFrom):
                    target = self._sibling(node)
                    if target is None:
                        continue
                    kind, owner = target
                    for alias in node.names:
                        bound = alias.asname or alias.name
                        if kind == "package":   # from . import x [as y]
                            imports[bound] = (alias.name, None)
                        else:                   # from .x import f [as g]
                            imports[bound] = (owner, alias.name)
            self.jit[stem] = jit
            self.imports[stem] = imports

    @staticmethod
    def _sibling(node: ast.ImportFrom) -> Optional[Tuple[str, str]]:
        package = PACKAGE["triton"]
        if node.level == 1:
            return ("package", "") if not node.module else ("module", node.module.split(".")[0])
        if node.level == 0 and node.module:
            if node.module == package:
                return ("package", "")
            if node.module.startswith(package + "."):
                return ("module", node.module[len(package) + 1:].split(".")[0])
        return None

    def resolve(self, stem: str, call: ast.Call) -> List[Tuple[str, str]]:
        """The @jit functions a call in module ``stem`` reaches: ``(module, name)``."""
        func = call.func
        if isinstance(func, ast.Name):
            if func.id in self.jit.get(stem, {}):
                return [(stem, func.id)]
            target = self.imports.get(stem, {}).get(func.id)
            if target and target[1] and target[1] in self.jit.get(target[0], {}):
                return [(target[0], target[1])]
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            target = self.imports.get(stem, {}).get(func.value.id)
            if target and target[1] is None and func.attr in self.jit.get(target[0], {}):
                return [(target[0], func.attr)]
        return []

    def closure(self, roots: Iterable[Tuple[str, ast.FunctionDef]]) -> List[Tuple[str, ast.FunctionDef]]:
        """The functions a set of @jit functions compiles: themselves and every @jit
        function they call, transitively."""
        seen: Dict[int, Tuple[str, ast.FunctionDef]] = {}
        stack = list(roots)
        while stack:
            stem, function = stack.pop()
            if id(function) in seen:
                continue
            seen[id(function)] = (stem, function)
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    for owner, name in self.resolve(stem, node):
                        for callee in self.jit[owner][name]:
                            stack.append((owner, callee))
        return sorted(seen.values(), key=lambda item: (item[0], item[1].lineno, item[1].col_offset))

    def program_text(self, members: Sequence[Tuple[str, ast.FunctionDef]]) -> str:
        """The source the Triton JIT compiles for one program: each member's own text, as
        written (Triton reads a function's source, not its syntax tree), in a fixed order.
        Python-version independent, unlike an unparse of the tree."""
        parts = []
        for owner, function in members:
            segment = ast.get_source_segment(self.sources[owner], function) or ""
            parts.append(f"# {owner}.{function.name}\n{segment}\n")
        return "\n".join(parts)

    def callees_in_module(self, stem: str) -> set:
        called = set()
        for functions in self.jit.get(stem, {}).values():
            for function in functions:
                for node in ast.walk(function):
                    if isinstance(node, ast.Call):
                        for owner, name in self.resolve(stem, node):
                            if owner == stem and name != function.name:
                                called.add(name)
        return called


def _strip_docstring(function: ast.FunctionDef) -> ast.FunctionDef:
    clone = ast.parse(ast.unparse(function)).body[0]
    body = getattr(clone, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        clone.body = body[1:] or [ast.Pass()]
    return clone


def emit_triton(package: TritonPackage, module_name: str,
                spec: Mapping[str, Any]) -> Tuple[str, List[Tuple[str, List[ast.FunctionDef], bool, str]], List[str]]:
    """(route, [(label, functions of the unit, is_kernel, program text)], notes)."""
    if spec.get("route") == "none":
        raise NoEmitRoute(spec.get("reason", "the recipe declares no emit route"))
    if module_name not in package.trees:
        raise EmitRefusal(f"{module_name}.py does not parse")
    notes: List[str] = []
    own = package.jit.get(module_name, {})
    if spec.get("launches"):
        source = (package.directory / f"{module_name}.py").read_text(encoding="utf-8")
        roots: List[Tuple[str, ast.FunctionDef]] = []
        for target in spec["launches"]:
            owner, name = target.rsplit(".", 1)
            if name not in package.jit.get(owner, {}):
                raise EmitRefusal(f"the recipe says it launches {target}, which is not a "
                                  f"@jit function of the table")
            if target not in source:
                raise EmitRefusal(f"the recipe says it launches {target}, which its own text "
                                  f"never names")
            roots.extend((owner, f) for f in package.jit[owner][name])
        units = []
        for owner, function in roots:
            members = package.closure([(owner, function)])
            units.append((f"launches {owner}.{function.name}", [_strip_docstring(f) for _o, f in members],
                          any(_TRITON_STORE.search(ast.unparse(f)) for _o, f in members),
                          package.program_text(members)))
        return "launches: " + ", ".join(spec["launches"]), units, notes
    if not own:
        raise NoEmitRoute("no @jit function of its own and no declared launch")
    extra_roots: List[Tuple[str, ast.FunctionDef]] = []
    for target in spec.get("also_launches") or []:
        owner, name = target.rsplit(".", 1)
        if name not in package.jit.get(owner, {}):
            raise EmitRefusal(f"the recipe says it also launches {target}, which is not a "
                              f"@jit function of the table")
        if owner not in {m for m, _n in package.imports.get(module_name, {}).values()}:
            raise EmitRefusal(f"the recipe says it also launches {target}, and the module "
                              f"does not import {owner}")
        extra_roots.extend((owner, f) for f in package.jit[owner][name])
    called = package.callees_in_module(module_name)
    entries = [(name, f) for name, functions in sorted(own.items()) for f in functions
               if name not in called]
    if not entries:  # every function calls another: read them all as one unit
        entries = [(name, f) for name, functions in sorted(own.items()) for f in functions]
    units = []
    for name, function in entries:
        members = package.closure([(module_name, function)])
        inlined = sorted({f"{o}.{f.name}" for o, f in members if (o, f.name) != (module_name, name)})
        label = name + (f" (+ {', '.join(inlined)})" if inlined else "")
        units.append((label, [_strip_docstring(f) for _o, f in members],
                      any(_TRITON_STORE.search(ast.unparse(f)) for _o, f in members),
                      package.program_text(members)))
    for owner, function in extra_roots:
        members = package.closure([(owner, function)])
        units.append((f"also launches {owner}.{function.name}",
                      [_strip_docstring(f) for _o, f in members],
                      any(_TRITON_STORE.search(ast.unparse(f)) for _o, f in members),
                      package.program_text(members)))
    route = "@jit programs, helpers inlined"
    if extra_roots:
        route += " + also launches " + ", ".join(spec["also_launches"])
    return route, units, notes


# ===========================================================================
# Judging one unit
# ===========================================================================


def _text_module():
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import parity_board_text  # noqa: PLC0415

    return parity_board_text


_DEFINITION = re.compile(
    r"(?:__device__|__forceinline__|inline|static)[^;{}()]*?\b([A-Za-z_]\w*)\s*\(")
_PREPROCESSOR = re.compile(r"^[ \t]*#[^\n]*$", re.M)


def _balanced(text: str, start: int) -> Optional[int]:
    """The index just past the parenthesis that closes the one at ``start``."""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _top_level_count(arguments: str) -> int:
    if not arguments.strip() or arguments.strip() == "void":
        return 0
    depth = 0
    count = 1
    for char in arguments:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            count += 1
    return count


def helper_arity_problems(text: str) -> List[str]:
    """Calls of a helper the text defines whose argument count fits none of its
    definitions: a call site that predates its helper's re-cut, so the text does not
    compile. Overloads and C++ default arguments are honoured."""
    text = _PREPROCESSOR.sub("", text)
    arities: Dict[str, List[Tuple[int, int]]] = {}
    definitions: set = set()
    for match in _DEFINITION.finditer(text):
        name = match.group(1)
        open_at = match.end() - 1
        close = _balanced(text, open_at)
        if close is None:
            continue
        rest = text[close:close + 40].lstrip()
        if not rest.startswith("{") and not rest.startswith("const"):
            continue
        parameters = text[open_at + 1:close - 1]
        total = _top_level_count(parameters)
        optional = sum(1 for part in re.split(r",(?![^()]*\))", parameters) if "=" in part)
        arities.setdefault(name, []).append((total - optional, total))
        definitions.add(open_at)
    problems: List[str] = []
    for name, ranges in sorted(arities.items()):
        for match in re.finditer(r"\b" + re.escape(name) + r"\s*\(", text):
            open_at = match.end() - 1
            if open_at in definitions:
                continue
            close = _balanced(text, open_at)
            if close is None:
                continue
            count = _top_level_count(text[open_at + 1:close - 1])
            if not any(low <= count <= high for low, high in ranges):
                expected = " or ".join(sorted({str(h) if l == h else f"{l}-{h}" for l, h in ranges}))
                problems.append(f"calls {name} with {count} arguments; the text defines "
                                f"it with {expected}")
                break
    return problems


#: How a C-family unit reads the chunk record (the read EMITTED_RECORD_READ matches): the
#: name the read goes through must be bound in the function that reads it.
_RECORD_USE: Dict[str, "re.Pattern[str]"] = {
    "cuda": re.compile(r"\bcb\s*\["),
    "metal": re.compile(r"\bcb\s*\.\s*cb\w+"),
}
#: A parameter named ``cb`` (a CUDA pointer, a Metal reference with its attribute).
_RECORD_PARAMETER = re.compile(r"(?:^|[\s&*])cb\s*(?:\[\[[^\]]*\]\])?\s*$")
#: A local (or file-scope) declaration of ``cb``: a type token right before the name, then
#: an initialiser or the end of the declaration.
_RECORD_LOCAL = re.compile(r"(?:\b[A-Za-z_]\w*\s*[&*]?|[&*])\s*\bcb\s*(?:=|;)")


def _top_level_blocks(text: str, start: int = 0, end: Optional[int] = None):
    """(head, body) of every brace block at the outermost depth of ``text[start:end]``,
    descending through a block whose head is no function (``extern "C" {``, a namespace)."""
    end = len(text) if end is None else end
    blocks, depth, last, open_at = [], 0, start, None
    for index in range(start, end):
        char = text[index]
        if char == "{":
            if depth == 0:
                open_at = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and open_at is not None:
                head = text[last:open_at]
                if head.rstrip().endswith(")") or re.search(r"\)\s*(?:const\s*)?$", head):
                    blocks.append((head, text[open_at:index + 1]))
                elif not re.match(r"\s*(?:typedef\s+)?struct\b", head.split(";")[-1]):
                    blocks.extend(_top_level_blocks(text, open_at + 1, index))
                last = index + 1
        elif char == ";" and depth == 0:
            last = index + 1
    return blocks


def record_binding_problems(table: str, text: str) -> List[str]:
    """Names the chunk-record read uses that the text does not bind, so the text does not
    compile: a function that reads ``cb`` with no parameter, local or file-scope ``cb`` (a
    record carried as a member of another struct, read as if it were in scope), and a text
    that names the ``ChunkBounds`` type without defining it. The record-read witness alone
    is a token match; this is what makes it a read of a bound record."""
    if table not in _RECORD_USE:
        return []
    text = _PREPROCESSOR.sub("", text)
    blocks = _top_level_blocks(text)
    file_scope = text
    for _head, body in blocks:
        file_scope = file_scope.replace(body, ";")
    file_scope = re.sub(r"\bstruct\s+\w+\s*\{[^{}]*\}", ";", file_scope)
    global_binding = bool(_RECORD_LOCAL.search(file_scope))
    problems: List[str] = []
    for head, body in blocks:
        if not _RECORD_USE[table].search(body):
            continue
        close = head.rstrip().rfind(")")
        depth, open_at = 0, None
        for index in range(close, -1, -1):
            if head[index] == ")":
                depth += 1
            elif head[index] == "(":
                depth -= 1
                if depth == 0:
                    open_at = index
                    break
        if open_at is None:
            continue
        parameters = head[open_at + 1:close]
        name = re.search(r"([A-Za-z_]\w*)\s*$", head[:open_at])
        bound = (global_binding
                 or any(_RECORD_PARAMETER.search(part)
                        for part in re.split(r",(?![^()]*\))", parameters))
                 or bool(_RECORD_LOCAL.search(body)))
        if not bound:
            problems.append(f"reads the chunk record through cb in "
                            f"{name.group(1) if name else 'a function'}, which binds no cb "
                            f"(no parameter, local or file-scope cb)")
            break
    if re.search(r"\bChunkBounds\b", text) and not re.search(r"\bstruct\s+ChunkBounds\b", text):
        problems.append("names the type ChunkBounds, which the text never defines")
    return problems


def judge_c(table: str, text: str) -> Dict[str, Any]:
    tw = _text_module()
    device = tw.strip_c_comments(text)
    own = tw.OwnText(literals=[device])
    counts = tw.form_counts(table, own, helpers=False).as_dict()
    reads = [p for p in EMITTED_RECORD_READ[table] if re.search(p, device)]
    markers = [p for p in tw.PML_FORM_TEXT[table] if re.search(p, device)]
    return {"reads": reads, "markers": markers, "forms": counts, "kernel": True,
            "arity": helper_arity_problems(device) + record_binding_problems(table, device),
            "select": tw.select_sites_c(table, device).as_dict()}


def judge_python(table: str, functions: Sequence[ast.FunctionDef], kernel: bool) -> Dict[str, Any]:
    tw = _text_module()
    own = tw.OwnText(jit_functions=list(functions))
    counts = tw.form_counts(table, own, helpers=False).as_dict()
    device = "\n".join(ast.unparse(f) for f in functions)
    reads = [p for p in EMITTED_RECORD_READ[table] if re.search(p, device)]
    markers = [p for p in tw.PML_FORM_TEXT[table] if re.search(p, device)]
    return {"reads": reads, "markers": markers, "forms": counts, "kernel": kernel,
            "arity": [], "select": tw.select_sites_python(functions).as_dict(),
            "_text": device}


# ===========================================================================
# One table
# ===========================================================================


def emitted_digest(units: Sequence[Mapping[str, Any]]) -> str:
    """One digest of a module's emitted texts: the sorted set of their sha256s. Labels
    (the configurations) are left out, so it moves exactly when a compiled text does."""
    return _sha("".join(f"{digest}\n" for digest in sorted({u["sha256"] for u in units})))


def _dump(dump_dir: Optional[Path], table: str, module: str, index: int, label: str,
          text: str) -> None:
    if dump_dir is None:
        return
    directory = dump_dir / table / module
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.=,-]+", "_", label)[:120]
    (directory / f"{index:03d}_{safe}.txt").write_text(text, encoding="utf-8")


def emit_table(root: Path, table: str, dump_dir: Optional[Path] = None,
               modules: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Every emitted row of one table: per module, its route, units and verdict inputs."""
    rows = json.loads((root / ROWS_FILE).read_text(encoding="utf-8"))["tables"][table]["modules"]
    recipes = load_recipes(root)
    canonical = recipes.get("canonical", {}).get(table, {})
    module_specs = recipes.get("modules", {}).get(table, {})
    cap = int(recipes.get("cap", 16))
    out: Dict[str, Any] = {}
    recorder: Optional[_Recorder] = None
    constants: Dict[str, Any] = {}
    package: Optional[TritonPackage] = None
    if table == "cuda":
        recorder = _Recorder()
        _install_cupy_recorder(recorder)
    if table == "metal":
        constants = _metal_constants()
    if table == "triton":
        package = TritonPackage(root)
    for row in rows:
        name = row["module"]
        if row["kind"] not in EMITTED_KINDS or (modules and name not in modules):
            continue
        if not (root / row["path"]).is_file():
            continue
        spec = module_specs.get(name, {})
        record: Dict[str, Any] = {"recipe": bool(spec)}
        try:
            if table == "triton":
                route, triton_units, notes = emit_triton(package, name, spec)
                judged = []
                for index, (label, functions, kernel, text) in enumerate(triton_units):
                    verdict = judge_python(table, functions, kernel)
                    verdict.pop("_text")
                    _dump(dump_dir, table, name, index, label, text)
                    judged.append({"label": label, "sha256": _sha(text), "chars": len(text),
                                   **verdict})
            else:
                if table == "cuda":
                    route, units, notes = emit_cuda(name, spec, canonical, recorder, cap)
                else:
                    route, units, notes = emit_metal(name, spec, canonical, constants, cap)
                judged = []
                seen = set()
                for index, (label, text) in enumerate(units):
                    digest = _sha(text)
                    if digest in seen:
                        continue
                    seen.add(digest)
                    _dump(dump_dir, table, name, index, label, text)
                    judged.append({"label": label, "sha256": digest, "chars": len(text),
                                   **judge_c(table, text)})
            if not judged:
                raise EmitRefusal(f"{route} emitted no device text")
            record.update({"witness": "emitted", "route": route, "notes": notes,
                           "units": judged,
                           "emitted_sha256": emitted_digest(judged)})
        except NoEmitRoute as reason:
            record.update({"witness": "source", "why": str(reason)})
        except EmitRefusal as refusal:
            record.update({"witness": "refused", "why": str(refusal)})
        out[name] = record
        print(f"  emit {table} {name}: {record['witness']} "
              f"{record.get('route', record.get('why', ''))[:90]} "
              f"({len(record.get('units', []))} units)", file=sys.stderr, flush=True)
    return out


# ===========================================================================
# The parent: one child per table
# ===========================================================================


def emitted_witnesses(root: Path, tables: Sequence[str] = TABLES,
                      dump_dir: Optional[Path] = None,
                      python: Optional[str] = None,
                      modules: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Run one child process per table and read back its JSON record (``modules``: only
    these rows)."""
    import tempfile  # noqa: PLC0415

    out: Dict[str, Dict[str, Any]] = {}
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "MEEP_GPU_DISPATCH": "0"})
    env.pop("PYTHONPATH", None)
    for table in tables:
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / f"{table}.json"
            command = [python or sys.executable, "-B", str(Path(__file__).resolve()),
                       "--root", str(root), "--table", table, "--out", str(target)]
            if dump_dir is not None:
                command += ["--dump-dir", str(dump_dir)]
            for module in modules or ():
                command += ["--module", module]
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            if completed.returncode != 0 or not target.is_file():
                raise RuntimeError(f"emitting the {table} table failed (exit "
                                   f"{completed.returncode}): {completed.stderr[-2000:]}")
            document = json.loads(target.read_text(encoding="utf-8"))
            out[table] = document["modules"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--table", choices=TABLES, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--module", action="append", help="emit only these modules")
    arguments = parser.parse_args(argv)
    root = (arguments.root or _HERE.parent.parent).resolve()
    _install_blocks()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = emit_table(root, arguments.table, arguments.dump_dir, arguments.module)
    document = {"schema": SCHEMA, "version": VERSION, "table": arguments.table,
                "modules": modules}
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    if arguments.out:
        arguments.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
