"""Runtime check that the array path forms no complex product (``DESIGN.md`` D16).

MEEP multiplies a complex field part by part: a real coefficient times each
``realnum`` array (the ``DOCMP`` loops), and a Bloch connection as libc++'s
``complex<float>`` product with every operation rounded on its own (step.cpp:171-221,
boundaries.cpp:495). NumPy's and CuPy's complex multiply is not that arithmetic: it
contracts to ``fma(a, c, -(b*d))`` on hosts with fused multiply-add (NumPy 2.x on arm64
and on x86-64 with FMA3, CuPy under NVRTC's default ``--fmad=true``), and a real
coefficient times a complex array is a full complex multiply against ``(k + 0j)``,
whose zero cross terms can flip the sign of a zero part. So the array path must hold
no ufunc whose loop is complex for an operation that rounds more than once.

This module enforces that at RUN time, not by reading source: :func:`intercept` views
every array a driver steps with as :class:`ProbeArray`, an ``ndarray`` subclass whose
``__array_ufunc__`` and ``__array_function__`` see every ufunc call and every array
function an operand of it reaches. A call is a VIOLATION when it is a multiplicative
operation (:data:`PRODUCT_UFUNCS`, :data:`PRODUCT_FUNCTIONS`) and any input, or the
requested loop ``dtype``, is complex: NumPy has no float-times-complex loop, so a real
array times a complex one runs the complex loop too, and checking only that BOTH inputs
are complex would miss it. Additive ufuncs on complex operands (add, subtract,
negative, the comparisons) are component-wise single operations and are counted as
TRAFFIC, which is what shows the probe actually saw the complex arithmetic.

What the views reach: every ``ndarray`` attribute of the driver's ``Fields``, of its
``PML`` table and of each polarization record (one level into dicts, lists and
tuples), and every buffer the step's ``StepScratch`` pool hands out. A result of a
ufunc on a probe array is a probe array, and so is a slice, a view, a copy and an
``empty_like``; ``numpy.asarray`` strips the subclass, so an array the code converts
that way and then multiplies with ANOTHER stripped array is outside the probe's
reach. That limit is why the probe is paired with a must-catch control
(:func:`planted_complex_product`) and why its report carries the traffic counts.

Scope: the stepping array path on the NumPy reference. DFT and flux accumulation
multiply fields by ``exp(i*omega*t)`` and are outside the identity claim (DESIGN §9);
a run with monitors reports their products as violations, so the identity cases carry
none. Host-only; importing this module imports nothing from the package.
"""

from __future__ import annotations

import contextlib
import os
import traceback
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy

#: Ufuncs that round more than once on a complex loop, or that are a product.
PRODUCT_UFUNCS = frozenset({
    numpy.multiply, numpy.divide, numpy.true_divide, numpy.floor_divide, numpy.power,
    numpy.float_power, numpy.matmul, numpy.square, numpy.reciprocal, numpy.exp,
    numpy.log, numpy.sqrt,
})

#: Array functions that multiply: a complex input to any of these is a violation.
PRODUCT_FUNCTIONS = frozenset({
    numpy.dot, numpy.vdot, numpy.inner, numpy.outer, numpy.tensordot, numpy.einsum,
    numpy.prod, numpy.cumprod, numpy.kron, numpy.convolve, numpy.correlate,
})

#: How many violation tracebacks a report keeps (every violation is counted).
KEEP_SITES = 64

_THIS_FILE = os.path.abspath(__file__)


def _is_complex(value: Any) -> bool:
    if isinstance(value, complex):
        return True
    dtype = getattr(value, "dtype", None)
    return dtype is not None and numpy.dtype(dtype).kind == "c"


class _Recorder:
    """Counts and call sites, shared by every probe array of one :func:`intercept`."""

    def __init__(self) -> None:
        self.active = False
        self.violations = 0
        self.traffic: Dict[str, int] = {}
        self.products: Dict[str, int] = {}
        self.sites: Dict[str, int] = {}
        self.examples: List[Dict[str, Any]] = []

    def call(self, name: str, inputs: Sequence[Any], dtype: Any, product: bool) -> None:
        if not self.active:
            return
        complex_loop = any(_is_complex(x) for x in inputs) or (
            dtype is not None and numpy.dtype(dtype).kind == "c")
        if not complex_loop:
            return
        if not product:
            self.traffic[name] = self.traffic.get(name, 0) + 1
            return
        self.violations += 1
        self.products[name] = self.products.get(name, 0) + 1
        frames = [f for f in traceback.extract_stack()
                  if os.path.abspath(f.filename) != _THIS_FILE]
        caller = frames[-1] if frames else None
        site = f"{caller.filename.rsplit('/', 1)[-1]}:{caller.lineno} {caller.name}" if caller else "?"
        self.sites[site] = self.sites.get(site, 0) + 1
        if len(self.examples) < KEEP_SITES:
            self.examples.append({
                "operation": name,
                "site": site,
                "inputs": [str(getattr(x, "dtype", type(x).__name__)) for x in inputs],
                "stack": [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}" for f in frames[-6:]],
            })

    def report(self) -> Dict[str, Any]:
        return {
            "violations": self.violations,
            "products": dict(sorted(self.products.items())),
            "sites": dict(sorted(self.sites.items(), key=lambda kv: -kv[1])),
            "complex_traffic": dict(sorted(self.traffic.items())),
            "complex_traffic_total": sum(self.traffic.values()),
            "examples": list(self.examples),
        }


class ProbeArray(numpy.ndarray):
    """An ``ndarray`` that reports complex products to its recorder (see the module docstring)."""

    _recorder: Optional[_Recorder] = None

    def __array_finalize__(self, obj: Any) -> None:
        self._recorder = getattr(obj, "_recorder", None)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        recorder = self._recorder
        plain = tuple(x.view(numpy.ndarray) if isinstance(x, ProbeArray) else x for x in inputs)
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(o.view(numpy.ndarray) if isinstance(o, ProbeArray) else o
                                  for o in out)
        if recorder is not None:
            product = ufunc in PRODUCT_UFUNCS
            name = ufunc.__name__ if method == "__call__" else f"{ufunc.__name__}.{method}"
            recorder.call(name, plain, kwargs.get("dtype"), product)
        result = getattr(ufunc, method)(*plain, **kwargs)
        if out is not None:
            return out[0] if len(out) == 1 else out
        return _rewrap(result, recorder)

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        recorder = self._recorder
        if recorder is not None and func in PRODUCT_FUNCTIONS:
            flat = [a for a in args if hasattr(a, "dtype") or isinstance(a, complex)]
            recorder.call(func.__name__, flat, kwargs.get("dtype"), True)
        return super().__array_function__(func, types, args, kwargs)


def _rewrap(result: Any, recorder: Optional[_Recorder]) -> Any:
    if isinstance(result, tuple):
        return tuple(_rewrap(r, recorder) for r in result)
    if isinstance(result, numpy.ndarray) and not isinstance(result, ProbeArray):
        view = result.view(ProbeArray)
        view._recorder = recorder
        return view
    return result


def _probe(array: Any, recorder: _Recorder) -> Any:
    if type(array) is numpy.ndarray:
        view = array.view(ProbeArray)
        view._recorder = recorder
        return view
    if isinstance(array, ProbeArray):
        array._recorder = recorder
    return array


def _wrap_attributes(owner: Any, recorder: _Recorder, saved: List[Tuple[Any, str, Any]]) -> None:
    """Replace every ndarray attribute of ``owner`` (and one container level down) by a probe view."""
    if owner is None:
        return
    names = list(vars(owner)) if hasattr(owner, "__dict__") else []
    for name in names:
        value = getattr(owner, name, None)
        if isinstance(value, numpy.ndarray):
            saved.append((owner, name, value))
            setattr(owner, name, _probe(value, recorder))
        elif isinstance(value, dict):
            if any(isinstance(v, numpy.ndarray) for v in value.values()):
                saved.append((owner, name, value))
                setattr(owner, name, {k: _probe(v, recorder) if isinstance(v, numpy.ndarray) else v
                                      for k, v in value.items()})
        elif isinstance(value, (list, tuple)):
            if any(isinstance(v, numpy.ndarray) for v in value):
                saved.append((owner, name, value))
                wrapped = [_probe(v, recorder) if isinstance(v, numpy.ndarray) else v for v in value]
                setattr(owner, name, type(value)(wrapped) if isinstance(value, tuple) else wrapped)


class _ScratchProbe:
    """Wraps one ``StepScratch`` so every buffer it hands out is a probe view."""

    def __init__(self, scratch: Any, recorder: _Recorder) -> None:
        self.scratch = scratch
        self.take = scratch.take
        self.constant = scratch.constant
        recorder_ref = recorder

        def take(tag: str, shape: Tuple[int, ...], dtype: Any) -> Any:
            return _probe(self.take(tag, shape, dtype), recorder_ref)

        def constant(key: Tuple[Any, ...], build: Any) -> Any:
            value = self.constant(key, build)
            return _probe(value, recorder_ref) if isinstance(value, numpy.ndarray) else value

        scratch.take = take
        scratch.constant = constant

    def restore(self) -> None:
        for name in ("take", "constant"):
            try:
                delattr(self.scratch, name)
            except AttributeError:
                pass


class Interception:
    """The handle :func:`intercept` yields: switch it on around the work to judge, read its report."""

    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    @contextlib.contextmanager
    def judging(self) -> Iterator[None]:
        """Count complex products only inside this block (a step, a stored-word read)."""
        previous = self._recorder.active
        self._recorder.active = True
        try:
            yield
        finally:
            self._recorder.active = previous

    @property
    def violations(self) -> int:
        return self._recorder.violations

    def report(self) -> Dict[str, Any]:
        return self._recorder.report()


@contextlib.contextmanager
def intercept(driver: Any) -> Iterator[Interception]:
    """View a NumPy-reference driver's arrays as probes for the duration of the block.

    Refuses a driver on a GPU backend: the probe is an ``ndarray`` subclass. Every
    attribute it replaced is restored on exit, so the driver steps on unchanged
    afterwards. Judging is off until :meth:`Interception.judging` switches it on.
    """
    fields = driver.fields
    xp = getattr(fields.grid, "xp", numpy)
    if xp is not numpy:
        raise ValueError("the complex-product probe runs on the NumPy reference only "
                         f"(this driver's array module is {getattr(xp, '__name__', xp)!r})")
    recorder = _Recorder()
    saved: List[Tuple[Any, str, Any]] = []
    owners = [fields, getattr(driver, "pml", None)]
    for polarization in getattr(fields, "polarizations", None) or ():
        owners.append(polarization)
    for owner in owners:
        _wrap_attributes(owner, recorder, saved)
    scratch = getattr(fields, "scratch", None)
    scratch_probe = _ScratchProbe(scratch, recorder) if scratch is not None else None
    try:
        yield Interception(recorder)
    finally:
        if scratch_probe is not None:
            scratch_probe.restore()
        for owner, name, value in reversed(saved):
            setattr(owner, name, value)


def planted_complex_product(values: Any, coefficient: float = 0.5) -> Any:
    """The must-catch control: a real coefficient times a complex array as one NumPy multiply.

    ``coefficient * values`` on complex64 runs the complex loop against
    ``(coefficient + 0j)``, which is exactly what the array path must not do; a probe
    that does not count this call is not looking.
    """
    return coefficient * values
