"""Backend resolution, and the subnormal guard the CuPy backend installs.

The guard exists because MEEP switches the CPU into flush-to-zero/denormals-are-zero
mode when its extension module initializes (src/mympi.cpp, meep#1708), and NVRTC's
front end then cannot parse the subnormal literals in CCCL's ``<cuda/std/limits>``
that CuPy's CUB-backed reductions include. See the note at the top of ``backends.py``.

These tests run on any host, with or without CuPy: the floating-point half is
measured directly, and the CuPy half runs against a stub module, so the wrapping
contract is pinned on the NumPy-only reference machine rather than only on a GPU.
"""

from __future__ import annotations

import ctypes
import sys
import types

import pytest

from meep_gpu import backends
from meep_gpu.backends import (
    guard_kernel_compilation,
    keep_subnormals,
    resolve_backend,
    subnormals_flushed,
)


def _read_environment():  # Raw fenv_t bytes, or None where the host cannot be asked.
    library, _ = backends._fenv()
    if library is None:
        return None
    buffer = ctypes.create_string_buffer(backends._FENV_BYTES)
    if library.fegetenv(ctypes.addressof(buffer)) != 0:
        return None
    return buffer.raw


def _rounding(library):  # (get, set) for fegetround/fesetround, typed.
    library.fegetround.argtypes = []
    library.fegetround.restype = ctypes.c_int
    library.fesetround.argtypes = [ctypes.c_int]
    library.fesetround.restype = ctypes.c_int
    return library.fegetround, library.fesetround


def _alternate_rounding(library):
    """A rounding mode this host supports that is NOT the one it is in.

    Found by measurement rather than from a per-ABI constant table (x86 spells
    FE_UPWARD 0x800, aarch64 0x400000). It exists so a test can put the caller into
    an environment that is demonstrably not the default one, which is the only way
    "the guard restored the caller's environment" can be asserted on a host that is
    not flushing subnormals and would otherwise take the guard's no-op path.
    """
    get_round, set_round = _rounding(library)
    original = get_round()
    for candidate in (0x800, 0x400, 0xC00, 0x400000, 0x800000, 0xC00000, 1, 2, 3):
        if candidate == original:
            continue
        if set_round(candidate) == 0 and get_round() == candidate:
            set_round(original)
            return original, candidate
    return original, None


def _stub_cupy(*entry_points: str):
    """A CuPy-shaped object exposing the named compiler front ends, and their call log."""
    calls: list[tuple] = []
    compiler = types.ModuleType("cupy.cuda.compiler")

    def make(name):
        def front_end(*args, **kwargs):
            calls.append((name, args, kwargs, subnormals_flushed()))
            return f"{name}-compiled"

        front_end.__name__ = name
        return front_end

    for name in entry_points:
        setattr(compiler, name, make(name))
    cuda = types.ModuleType("cupy.cuda")
    cuda.compiler = compiler
    cuda.Device = lambda gpu_id=0: types.SimpleNamespace(use=lambda: None)
    cupy = types.ModuleType("cupy")
    cupy.cuda = cuda
    return cupy, compiler, calls


# --- the measurement -------------------------------------------------------


def test_subnormals_flushed_is_a_measurement_of_this_thread():
    assert isinstance(subnormals_flushed(), bool)


def test_no_module_in_this_package_writes_a_subnormal_float_literal():
    """A subnormal spelled as a literal in this package is a literal that may be ZERO.

    Same trap as the NVRTC one, one level down: CPython parses float literals with
    dtoa's ``strtod``, which finishes in floating-point arithmetic, so a module
    compiled while MEEP's FTZ bit is set gets ``0.0`` baked into ``co_consts`` where
    its source says ``5e-324``. This package is imported alongside MEEP by design, so
    every module here is exposed. It cost a debugging cycle: the first
    ``subnormals_flushed`` compared against two subnormal literals and therefore
    reported "flushing" forever — including right after the guard had suspended it,
    which made the guard refuse itself as ineffective.

    Build subnormals from their bits (:func:`backends._smallest_subnormal`) instead.
    """
    import ast
    from pathlib import Path

    smallest_normal = sys.float_info.min
    offenders: list[str] = []
    for path in sorted(Path(backends.__file__).resolve().parent.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                if 0.0 < abs(node.value) < smallest_normal:
                    offenders.append(f"  {path.name}:{node.lineno}: {node.value!r}")
    assert not offenders, (
        "subnormal float literals become 0.0 when the module is compiled under MEEP's "
        "flush-to-zero mode:\n" + "\n".join(offenders)
    )


def test_the_smallest_subnormal_is_built_from_bits_and_survives_flushing():
    """Its BITS are asserted, never its value — arithmetic cannot check this.

    Measured on a flushing host: ``_smallest_subnormal() != 0.0`` evaluates to
    **False** there even though the value is 5e-324, because the comparison is itself
    an SSE operation and DAZ hands the instruction a zero. Every value-level check of
    a subnormal has that hole, which is the whole reason the constant is built from
    its bit pattern rather than computed or written down.
    """
    import numpy

    smallest = backends._smallest_subnormal()
    assert numpy.float64(smallest).dtype == numpy.dtype(numpy.float64)
    assert numpy.float64(smallest).tobytes() == (1).to_bytes(8, sys.byteorder)


def test_keep_subnormals_is_a_no_op_when_nothing_is_flushing():
    """The minimal-intervention half: a healthy host's environment is not touched.

    On a process that HAS loaded MEEP this is the other branch — flushing on, guard
    in force, subnormals available inside — and both are asserted here so the test
    means something either way rather than being skipped on one kind of host.
    """
    before_flag = subnormals_flushed()
    before_bytes = _read_environment()
    with keep_subnormals() as guarded:
        inside_flag = subnormals_flushed()
    assert subnormals_flushed() == before_flag
    assert _read_environment() == before_bytes
    if guarded:
        assert before_flag is True and inside_flag is False
    else:
        assert before_flag is False and inside_flag is False


@pytest.fixture()
def flushing_caller(monkeypatch):
    """Force the guarded path, from a caller environment that is NOT the default one.

    Without this the guard short-circuits on any machine that is not flushing
    subnormals, and "it restores what it saved" is never executed — the assertion
    would compare the default environment against itself and pass no matter what
    the guard did. The rounding mode is the lever because it is the one piece of the
    floating-point environment C exposes portably.
    """
    library, default = backends._fenv()
    if library is None or default is None:
        pytest.fail("this host's default floating-point environment is not addressable")
    original, alternate = _alternate_rounding(library)
    if alternate is None:
        pytest.fail("no alternate rounding mode: the caller environment cannot be made distinct")
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: True)
    _, set_round = _rounding(library)
    set_round(alternate)
    try:
        yield
    finally:
        set_round(original)


def test_keep_subnormals_installs_the_default_environment_and_puts_the_callers_back(
    flushing_caller,
):
    before = _read_environment()
    with keep_subnormals() as guarded:
        inside = _read_environment()
    after = _read_environment()
    assert guarded is True
    assert inside != before, "the guard did not install the default environment"
    assert after == before, "the guard did not restore the caller's environment"


def test_keep_subnormals_restores_even_when_the_body_raises(flushing_caller):
    before = _read_environment()
    with pytest.raises(ZeroDivisionError):
        with keep_subnormals():
            raise ZeroDivisionError("body")
    assert _read_environment() == before


def test_the_hosts_default_environment_is_addressable():
    """Linux, macOS and Windows all have a spelling of FE_DFL_ENV that this resolves.

    A host where this fails is one where a MEEP process cannot compile a CuPy kernel
    at all, and ``resolve_backend`` refuses rather than letting NVRTC fail inside a
    header — so this is the test that says which of the two paths this machine is on.
    """
    library, default = backends._fenv()
    assert library is not None, "no C library exposing fegetenv/fesetenv was found"
    assert default is not None, "this host's FE_DFL_ENV could not be addressed"


# --- the CuPy wrap ---------------------------------------------------------


def test_guard_wraps_every_compiler_front_end_and_calls_through():
    cupy, compiler, calls = _stub_cupy("compile_using_nvrtc", "compile_using_nvcc")
    assert guard_kernel_compilation(cupy) is (backends._fenv()[1] is not None)
    for name in ("compile_using_nvrtc", "compile_using_nvcc"):
        front_end = getattr(compiler, name)
        assert getattr(front_end, backends._GUARD_MARK, False)
        assert front_end.__name__ == name  # functools.wraps: CuPy still sees its own function.
        assert front_end("source", options=("-std=c++17",)) == f"{name}-compiled"
    assert [entry[0] for entry in calls] == ["compile_using_nvrtc", "compile_using_nvcc"]
    assert calls[0][1] == ("source",) and calls[0][2] == {"options": ("-std=c++17",)}


def test_guard_is_idempotent():
    cupy, compiler, _ = _stub_cupy("compile_using_nvrtc")
    guard_kernel_compilation(cupy)
    once = compiler.compile_using_nvrtc
    guard_kernel_compilation(cupy)
    assert compiler.compile_using_nvrtc is once


def test_guard_leaves_front_ends_this_cupy_does_not_ship_alone():
    cupy, compiler, _ = _stub_cupy("compile_using_hipcc")
    guard_kernel_compilation(cupy)
    assert getattr(compiler.compile_using_hipcc, backends._GUARD_MARK, False)
    assert not hasattr(compiler, "compile_using_nvrtc")


def test_guard_reports_failure_when_no_front_end_is_found():
    cupy, _, _ = _stub_cupy()
    assert guard_kernel_compilation(cupy) is False


def test_the_compile_runs_inside_the_subnormal_guard(monkeypatch):
    """Not "a context manager was entered" — the compile must be INSIDE this one."""
    entered: list[str] = []

    import contextlib

    @contextlib.contextmanager
    def recording():
        entered.append("enter")
        yield True
        entered.append("exit")

    monkeypatch.setattr(backends, "keep_subnormals", recording)
    cupy, compiler, calls = _stub_cupy("compile_using_nvrtc")
    guard_kernel_compilation(cupy)
    compiler.compile_using_nvrtc("source")
    assert entered == ["enter", "exit"]
    assert len(calls) == 1


# --- resolve_backend -------------------------------------------------------


def test_numpy_backend_is_returned_untouched():
    import numpy

    xp, gpu = resolve_backend(prefer_gpu=False)
    assert xp is numpy and gpu is None


def test_gpu_resolution_installs_the_guard(monkeypatch):
    cupy, compiler, _ = _stub_cupy("compile_using_nvrtc")
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    xp, gpu = resolve_backend(prefer_gpu=True, gpu_id=0)
    assert xp is cupy and gpu == "cuda"
    assert getattr(compiler.compile_using_nvrtc, backends._GUARD_MARK, False)


def test_gpu_resolution_refuses_a_flushing_host_it_cannot_protect(monkeypatch):
    cupy, _, _ = _stub_cupy("compile_using_nvrtc")
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    monkeypatch.setattr(backends, "guard_kernel_compilation", lambda _cupy: False)
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: True)
    with pytest.raises(RuntimeError, match="floating constant is out of range"):
        resolve_backend(prefer_gpu=True)


def test_gpu_resolution_is_silent_on_a_host_that_is_not_flushing(monkeypatch):
    cupy, _, _ = _stub_cupy()  # No front end to wrap: guard_kernel_compilation is False.
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: False)
    xp, gpu = resolve_backend(prefer_gpu=True)
    assert xp is cupy and gpu == "cuda"
