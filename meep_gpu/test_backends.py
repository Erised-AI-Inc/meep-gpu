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
import importlib
import importlib.util
import sys
import types
import uuid
from typing import Any

import pytest

from meep_gpu import backends
from meep_gpu.backends import (
    compiler_front_ends,
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


# --- CuPy stand-ins: the compile path, in each shape CuPy has shipped ----------
#
# Reduced copies of ``cupy/cuda/compiler.py`` at v13.5.1 and v14.2.0, executed INTO
# a module so that each function looks its callees up in that module, exactly as
# CuPy's do: a hook set on the module is then called if and only if the compile
# path really calls the function it was set on. Kept to what the hooks read: which
# front end the compile path calls, where it caches, and what it keys on. The
# shared helpers (``front_end``, ``cache_name``, ``disk_load``, ``disk_save``) are
# injected by :func:`cupy_compiler_module`; ``front_end(name, source, options)`` is
# the test's compiler.

CUPY_13_COMPILER = """
import os

_default_cache_dir = DEFAULT_CACHE_DIR


def get_cache_dir():
    return os.environ.get("CUPY_CACHE_DIR", _default_cache_dir)


def compile_using_nvrtc(source, options=(), arch=None, filename="kern.cu", *rest):
    return front_end("compile_using_nvrtc", source, options)


def compile_using_nvcc(source, options=(), arch=None, filename="kern.cu", *rest):
    return front_end("compile_using_nvcc", source, options)


def compile_using_hipcc(source, options, arch, log_stream=None):
    return front_end("compile_using_hipcc", source, options)


def _compile_with_cache_cuda(source, options, arch, cache_dir, extra_source=None,
                             backend="nvrtc", *rest):
    if cache_dir is None:
        cache_dir = get_cache_dir()
    options += ("-ftz=true",)
    if backend not in ("nvrtc", "nvcc"):
        raise ValueError("Invalid backend %s" % backend)
    name = cache_name(source, options, extra_source)
    binary = disk_load(cache_dir, name)
    if binary is None:
        if backend == "nvrtc":
            binary = compile_using_nvrtc(source, options, arch)
        else:
            binary = compile_using_nvcc(source, options, arch)
        disk_save(cache_dir, name, binary)
    return binary


def _compile_with_cache_hip(source, options, arch, cache_dir, extra_source=None,
                            backend="hiprtc", *rest):
    if backend == "hiprtc":
        return compile_using_nvrtc(source, options, arch)
    return compile_using_hipcc(source, options, arch)


def _compile_module_with_cache(source, options=(), arch=None, cache_dir=None,
                               extra_source=None, backend="nvrtc", **kwargs):
    return _compile_with_cache_cuda(source, options, arch, cache_dir, extra_source, backend)
"""

CUPY_14_COMPILER = """
import os


class _DiskKernelCacheBackend:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.environ.get("CUPY_CACHE_DIR", DEFAULT_CACHE_DIR)
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load(self, name):
        return disk_load(self._cache_dir, name)

    def save(self, name, cubin, source):
        disk_save(self._cache_dir, name, cubin)


_kernel_cache_backend = _DiskKernelCacheBackend()


def _compile_using_nvrtc_no_warning(source, options=(), arch=None, filename="kern.cu",
                                    *rest):
    return front_end("_compile_using_nvrtc_no_warning", source, options)


def compile_using_nvrtc(source, options=(), arch=None, filename="kern.cu", *rest):
    return _compile_using_nvrtc_no_warning(source, options, arch, filename, *rest)


def compile_using_nvcc(source, options=(), arch=None, filename="kern.cu", *rest):
    return front_end("compile_using_nvcc", source, options)


def compile_using_hipcc(source, options, arch, log_stream=None):
    return front_end("compile_using_hipcc", source, options)


def _compile_with_cache_cuda(source, options, arch, extra_source=None, backend="nvrtc",
                             *rest):
    options += ("-ftz=true",)
    if backend not in ("nvrtc", "nvcc"):
        raise ValueError("Invalid backend %s" % backend)
    name = cache_name(source, options, extra_source)
    binary = _kernel_cache_backend.load(name)
    if binary is None:
        if backend == "nvrtc":
            binary = _compile_using_nvrtc_no_warning(source, options, arch)
        else:
            binary = compile_using_nvcc(source, options, arch)
        _kernel_cache_backend.save(name, binary, source)
    return binary


def _compile_with_cache_hip(source, options, arch, extra_source=None, backend="hiprtc",
                            *rest):
    if backend == "hiprtc":
        return compile_using_nvrtc(source, options, arch)
    return compile_using_hipcc(source, options, arch)


def _compile_module_with_cache(source, options=(), *, arch=None, extra_source=None,
                               backend="nvrtc", **kwargs):
    return _compile_with_cache_cuda(source, options, arch, extra_source, backend)
"""

CUPY_COMPILERS = {"13": CUPY_13_COMPILER, "14": CUPY_14_COMPILER}

#: The NVRTC front end each shape's compile path calls.
NVRTC_FRONT_END = {"13": "compile_using_nvrtc", "14": "_compile_using_nvrtc_no_warning"}


def cache_name(source, options, extra_source) -> str:  # The stand-in's disk-cache key.
    import hashlib  # noqa: PLC0415

    key = repr((source, tuple(options), extra_source)).encode("utf-8")
    return hashlib.sha256(key).hexdigest() + ".cubin"


def _disk_load(directory, name):
    import os  # noqa: PLC0415

    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _disk_save(directory, name, binary):
    import os  # noqa: PLC0415

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        handle.write(str(binary))


def cupy_compiler_module(shape: str, front_end, default_cache_dir, source: str = "") -> Any:
    """``cupy.cuda.compiler`` in CuPy ``shape`` ("13" or "14"), with ``front_end`` as its compiler.

    ``source`` replaces the shape's text (for a CuPy whose seam has moved).
    ``default_cache_dir`` is CuPy's default directory; the 14 shape opens its disk
    backend when built, reading ``CUPY_CACHE_DIR`` then, as CuPy 14 does at import.
    """
    module = types.ModuleType("cupy.cuda.compiler")
    module.__dict__.update(front_end=front_end, cache_name=cache_name, disk_load=_disk_load,
                           disk_save=_disk_save, DEFAULT_CACHE_DIR=str(default_cache_dir))
    exec(compile(source or CUPY_COMPILERS[shape], f"<cupy {shape} compiler stand-in>", "exec"),
         module.__dict__)
    return module


def _stub_cupy(shape: str, cache_dir, source: str = ""):
    """A CuPy-shaped object around a compiler stand-in, and the compile log.

    Each log entry is ``(front end, source, options, flushing during the compile)``.
    """
    calls: list[tuple] = []

    def front_end(name, source, options):
        calls.append((name, source, tuple(options), subnormals_flushed()))
        return f"{name}-compiled"

    compiler = cupy_compiler_module(shape, front_end, cache_dir, source)
    cuda = types.ModuleType("cupy.cuda")
    cuda.compiler = compiler
    cuda.Device = lambda gpu_id=0: types.SimpleNamespace(use=lambda: None)
    cupy = types.ModuleType("cupy")
    cupy.cuda = cuda
    return cupy, compiler, calls


@pytest.fixture(autouse=True)
def private_cupy_cache_dir(monkeypatch, tmp_path):
    """Each test's own ``CUPY_CACHE_DIR``: the stand-ins cache as CuPy does, where it says.

    Without it a stand-in compile on a host that sets the variable for the whole run
    is served an entry an earlier test wrote, and its front end is never called.
    """
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy_cache"))


#: A CuPy whose compile path calls an NVRTC front end under a name nothing here knows.
MOVED_SEAM_COMPILER = CUPY_14_COMPILER.replace(
    "_compile_using_nvrtc_no_warning", "_nvrtc_build").replace(
    "compile_using_nvcc", "_nvcc_build").replace("compile_using_hipcc", "_hipcc_build")


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

SHAPES = pytest.mark.parametrize("shape", ["13", "14"])

#: What each shape's compile paths call, as the guard must find it.
EXPECTED_FRONT_ENDS = {
    "13": ("compile_using_nvrtc", "compile_using_nvcc", "compile_using_hipcc"),
    "14": ("_compile_using_nvrtc_no_warning", "compile_using_nvcc", "compile_using_hipcc"),
}


@pytest.fixture
def recording_guard(monkeypatch):
    """Replace the guard's body with a recorder: one "enter"/"exit" pair per guarded compile."""
    import contextlib

    entered: list[str] = []

    @contextlib.contextmanager
    def recording():
        entered.append("enter")
        yield True
        entered.append("exit")

    monkeypatch.setattr(backends, "keep_subnormals", recording)
    return entered


@SHAPES
def test_the_front_ends_are_read_off_the_compile_path(shape, tmp_path):
    _cupy, compiler, _calls = _stub_cupy(shape, tmp_path)
    assert compiler_front_ends(compiler) == (EXPECTED_FRONT_ENDS[shape], None)


@SHAPES
def test_guard_wraps_the_front_ends_the_compile_path_calls(shape, tmp_path):
    cupy, compiler, calls = _stub_cupy(shape, tmp_path)
    assert guard_kernel_compilation(cupy) is (backends._fenv()[1] is not None)
    for name in EXPECTED_FRONT_ENDS[shape]:
        front_end = getattr(compiler, name)
        assert getattr(front_end, backends._GUARD_MARK, False), name
        assert front_end.__name__ == name  # functools.wraps: CuPy still sees its own function.
    # CuPy 14's public compile_using_nvrtc hands over to the private one, which is guarded.
    assert getattr(compiler.compile_using_nvrtc, backends._GUARD_MARK, False) is (shape == "13")
    assert compiler._compile_module_with_cache("source", ("-std=c++17",)) == (
        f"{NVRTC_FRONT_END[shape]}-compiled")
    assert calls == [(NVRTC_FRONT_END[shape], "source", ("-std=c++17", "-ftz=true"), calls[0][3])]


@SHAPES
@pytest.mark.parametrize("backend", ["nvrtc", "nvcc"])
def test_a_compile_through_the_compile_path_runs_inside_the_guard(
        shape, backend, tmp_path, recording_guard):
    """The hook must be CALLED by the compile path, not merely installed.

    A guard on ``compile_using_nvrtc`` under CuPy 14 installs, reports itself guarded,
    and is never entered, because the compile path calls
    ``_compile_using_nvrtc_no_warning``.
    """
    cupy, compiler, calls = _stub_cupy(shape, tmp_path)
    guard_kernel_compilation(cupy)
    compiler._compile_module_with_cache("source", (), backend=backend)
    assert recording_guard == ["enter", "exit"]
    expected = NVRTC_FRONT_END[shape] if backend == "nvrtc" else "compile_using_nvcc"
    assert [entry[0] for entry in calls] == [expected]


def test_a_direct_call_of_cupy_14s_public_front_end_is_guarded_once(tmp_path, recording_guard):
    cupy, compiler, calls = _stub_cupy("14", tmp_path)
    guard_kernel_compilation(cupy)
    compiler.compile_using_nvrtc("source", ("-std=c++17",))
    assert recording_guard == ["enter", "exit"]
    assert [entry[0] for entry in calls] == ["_compile_using_nvrtc_no_warning"]


@SHAPES
def test_guard_is_idempotent(shape, tmp_path):
    cupy, compiler, _ = _stub_cupy(shape, tmp_path)
    guard_kernel_compilation(cupy)
    once = getattr(compiler, NVRTC_FRONT_END[shape])
    guard_kernel_compilation(cupy)
    assert getattr(compiler, NVRTC_FRONT_END[shape]) is once
    assert compiler_front_ends(compiler) == (EXPECTED_FRONT_ENDS[shape], None)


def test_guard_leaves_a_front_end_the_compile_path_does_not_call_alone(tmp_path):
    source = CUPY_14_COMPILER + (
        "\n\ndef compile_using_nothing(source, options=()):\n"
        "    return front_end('compile_using_nothing', source, options)\n")
    cupy, compiler, _ = _stub_cupy("14", tmp_path, source)
    guard_kernel_compilation(cupy)
    assert not getattr(compiler.compile_using_nothing, backends._GUARD_MARK, False)


def test_a_compile_path_that_calls_no_known_front_end_is_refused_by_name(tmp_path):
    cupy, compiler, _ = _stub_cupy("14", tmp_path, MOVED_SEAM_COMPILER)
    front_ends, problem = compiler_front_ends(compiler)
    assert front_ends == ()
    assert "_compile_with_cache_cuda" in problem and "compile_using" in problem
    assert "the seam has moved" in problem
    assert guard_kernel_compilation(cupy) is False
    for name in ("_nvrtc_build", "_nvcc_build", "compile_using_nvrtc"):
        assert not getattr(getattr(compiler, name), backends._GUARD_MARK, False), name


def test_a_compile_path_that_cannot_be_read_is_refused_by_name(tmp_path):
    def compile_using_nvrtc(source, options=()):
        return "compiled"

    def _compile_with_cache_cuda(source, options, arch, extra_source=None):
        return compile_using_nvrtc(source, options)

    elsewhere = types.SimpleNamespace(compile_using_nvrtc=compile_using_nvrtc,
                                      _compile_with_cache_cuda=_compile_with_cache_cuda)
    assert compiler_front_ends(elsewhere) == ((), (
        "cupy.cuda.compiler._compile_with_cache_cuda does not look its callees up in "
        "cupy.cuda.compiler, so a hook set there would never be called"))
    absent = types.SimpleNamespace(compile_using_nvrtc=compile_using_nvrtc)
    assert compiler_front_ends(absent) == ((), "cupy.cuda.compiler has no _compile_with_cache_cuda")
    compiled = types.SimpleNamespace(compile_using_nvrtc=compile_using_nvrtc,
                                     _compile_with_cache_cuda=len)
    assert "is not a Python function" in compiler_front_ends(compiled)[1]
    for cupy_compiler in (elsewhere, absent, compiled):
        cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(compiler=cupy_compiler))
        assert guard_kernel_compilation(cupy) is False
        assert not getattr(cupy_compiler.compile_using_nvrtc, backends._GUARD_MARK, False)


def test_the_compile_runs_inside_the_subnormal_guard(tmp_path, recording_guard):
    """Not "a context manager was entered" — the compile must be INSIDE this one."""
    cupy, compiler, calls = _stub_cupy("14", tmp_path)
    guard_kernel_compilation(cupy)

    def inside(name, source, options):
        calls.append(list(recording_guard))
        return "compiled"

    compiler.front_end = inside
    compiler._compile_module_with_cache("source")
    assert calls == [["enter"]]
    assert recording_guard == ["enter", "exit"]


@pytest.mark.requires_resource("cupy")
def test_the_real_cupy_compile_path_calls_a_guarded_front_end(tmp_path, recording_guard):
    """On a host with CuPy and a device: a cold compile enters the guard.

    The source carries a fresh token, so neither CuPy's in-process memo nor its disk
    cache can serve it, whichever CuPy version and cache directory are in force.
    """
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CuPy is not installed")
    cupy = importlib.import_module("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception as exc:  # noqa: BLE001 - no driver is no device
        pytest.skip(f"no CUDA device ({exc})")
    front_ends, problem = compiler_front_ends(cupy.cuda.compiler)
    assert problem is None and front_ends
    assert guard_kernel_compilation(cupy) is (backends._fenv()[1] is not None)
    token = f"k{uuid.uuid4().hex}"
    kernel = cupy.RawKernel(
        "extern \"C\" __global__ void " + token + "(float* x) { x[0] = 1.0f; }", token)
    kernel.compile()
    assert recording_guard[:2] == ["enter", "exit"], (front_ends, recording_guard)


# --- resolve_backend -------------------------------------------------------


def test_numpy_backend_is_returned_untouched():
    import numpy

    xp, gpu = resolve_backend(prefer_gpu=False)
    assert xp is numpy and gpu is None


@SHAPES
def test_gpu_resolution_installs_the_guard(shape, monkeypatch, tmp_path):
    cupy, compiler, _ = _stub_cupy(shape, tmp_path)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    xp, gpu = resolve_backend(prefer_gpu=True, gpu_id=0)
    assert xp is cupy and gpu == "cuda"
    assert getattr(getattr(compiler, NVRTC_FRONT_END[shape]), backends._GUARD_MARK, False)


def test_gpu_resolution_refuses_a_flushing_host_it_cannot_protect(monkeypatch, tmp_path):
    cupy, _, _ = _stub_cupy("13", tmp_path)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    monkeypatch.setattr(backends, "guard_kernel_compilation", lambda _cupy: False)
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: True)
    with pytest.raises(RuntimeError, match="floating constant is out of range"):
        resolve_backend(prefer_gpu=True)


def test_gpu_resolution_names_the_moved_seam_on_a_flushing_host(monkeypatch, tmp_path):
    cupy, compiler, _ = _stub_cupy("14", tmp_path, MOVED_SEAM_COMPILER)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: True)
    with pytest.raises(RuntimeError) as excinfo:
        resolve_backend(prefer_gpu=True)
    assert compiler_front_ends(compiler)[1] in str(excinfo.value)
    assert "floating constant is out of range" in str(excinfo.value)


def test_gpu_resolution_is_silent_on_a_host_that_is_not_flushing(monkeypatch, tmp_path):
    # No front end to wrap: guard_kernel_compilation is False.
    cupy, _, _ = _stub_cupy("14", tmp_path, MOVED_SEAM_COMPILER)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backends, "cupy_available", lambda: True)
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: False)
    xp, gpu = resolve_backend(prefer_gpu=True)
    assert xp is cupy and gpu == "cuda"
