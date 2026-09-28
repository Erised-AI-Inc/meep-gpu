"""
Array-backend resolution for the MEEP-compatible FDTD engine.

Every numeric module in this package is written against the NumPy array API and
receives its array module (``xp``) explicitly — NumPy always works (the testable
reference path on any machine), CuPy is selected when a CUDA device is present.
This keeps the engine importable and verifiable without CuPy installed.

``prefer_gpu`` names a GPU, not an array module: ``True`` resolves this host's
GPU — CuPy on a CUDA host, or NumPy host arrays with Metal kernels beside them on
an Apple GPU — and ``False`` is the NumPy reference, which never consults a
kernel table (:func:`resolve_backend`, :func:`available_gpu`).

No module in this package may import ``cupy`` at module level, and none may
import ``meep`` at all; host integrations own CPU-solver dispatch.

Resolving the CuPy backend also installs the subnormal guard below, because the
process this package runs in is a process that has imported MEEP.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import functools
import importlib
import importlib.util
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Subnormal flushing, and why it breaks GPU kernel compilation
#
# MEEP turns on the x86 FTZ and DAZ bits when its extension module initializes
# (src/mympi.cpp `_set_zero_subnormals`: `mxcsr |= 0x8040`, per thread via an
# OpenMP loop; meep#1708). Subnormal values are then flushed to zero, which is a
# real speedup deep in the tails of an exponentially decaying source and costs a
# sliver of range. That is MEEP's call to make, and this package does not undo it.
#
# It does, however, apply to every later call on that thread — including CuPy's
# just-in-time kernel compiler. NVRTC's front end parses floating literals with
# the HOST C library, and CCCL's `<cuda/std/limits>`, which CuPy's CUB-backed
# reductions include, spells `numeric_limits<float>::denorm_min()` as
# `__FLT_DENORM_MIN__` and the double case as `__DBL_DENORM_MIN__`. Both are
# subnormal, so with FTZ set `strtod`/`strtof` underflow them to zero and report
# ERANGE, and the compile dies on the header rather than on anything the caller
# wrote::
#
#     cupy/_core/include/cupy/_cccl/libcudacxx/cuda/std/limits(429):
#         error: floating constant is out of range        (__FLT_DENORM_MIN__)
#     cupy/_core/include/cupy/_cccl/libcudacxx/cuda/std/limits(524):
#         error: floating constant is out of range        (__DBL_DENORM_MIN__)
#
# Measured on the GPU host, CuPy 13.5.1 / MEEP 1.33.0: a fresh NVRTC compile of a
# kernel including that header succeeds before `import meep` and fails after it,
# and `strtod("4.9406564584124654e-324")` returns 5e-324 before and 0.0 after.
# It surfaces only on a COLD kernel cache — a cached kernel is never compiled, so
# the same script passes or fails depending on what else the host has run, which
# is why it showed up on one corpus script out of 57 rather than on all of them.
#
# The fix is scoped to the compiler and nothing else: save the thread's
# floating-point environment, install the default one for the duration of the
# compile so the literals parse, and put MEEP's back. The engine's own arithmetic
# — and MEEP's — runs in exactly the mode MEEP asked for, before and after.
#
# FTZ and DAZ sit OUTSIDE C99's floating-point environment, so no standard says
# `fesetenv(FE_DFL_ENV)` must clear them, and a reading of the header cannot settle
# it. Measured on the GPU host's glibc 2.35: it does — MXCSR goes 0x9fe2 -> 0x1f80 and
# back on restore. `_default_environment` therefore accepts a candidate only after
# watching the FPU stop flushing under it, and a host where that does not happen is
# refused at `resolve_backend` rather than left to fail later inside a CCCL header.
# ---------------------------------------------------------------------------

# `fenv_t` is 32 bytes on x86-64 glibc and 8 on aarch64; over-allocate rather than
# encode a per-ABI layout, since only fegetenv/fesetenv ever read it.
_FENV_BYTES = 256

# Set by wrapped compiler entry points so the wrap is applied at most once.
_GUARD_MARK = "_meep_gpu_keeps_subnormals"

# CuPy's compiler front ends, in every backend it ships. Wrapping the leaves
# rather than the caching layer keeps the guard on the one operation that parses
# source; a cache hit never compiles and never needs it.
_COMPILER_ENTRY_POINTS = ("compile_using_nvrtc", "compile_using_nvcc", "compile_using_hipcc")

_fenv_resolved: tuple[Any, Any] | None = None  # (library, FE_DFL_ENV address) once probed.


def _smallest_subnormal():
    """The smallest positive double, built from its BITS rather than written down.

    A subnormal spelled as a literal is not safe in this package. CPython parses
    float literals with dtoa's ``strtod``, which finishes in floating-point
    arithmetic, so a module compiled after MEEP has set FTZ gets ``0.0`` where its
    source says ``5e-324`` — permanently, in ``co_consts``. The first version of
    :func:`subnormals_flushed` was written that way and therefore answered "yes,
    flushing" forever, including immediately after the guard had suspended it.
    ``test_backends.py`` refuses any subnormal literal in the package for this reason.
    """
    import numpy

    return numpy.frombuffer((1).to_bytes(8, sys.byteorder), dtype=numpy.float64)[0]


def subnormals_flushed() -> bool:
    """Is this thread's FPU flushing subnormal values to zero right now?

    Both bits are measured, not read: FTZ by a product of two NORMAL operands
    (``1e-300`` and ``1e-16``; the smallest normal double is ~2.2e-308) whose exact
    result is subnormal, and DAZ by a subnormal OPERAND that a plain multiply by one
    should return unchanged. Either one set is enough to break the literals in CCCL's
    ``<cuda/std/limits>``.
    """
    import numpy

    flush_to_zero = float(numpy.float64(1e-300) * numpy.float64(1e-16)) == 0.0
    denormals_are_zero = float(_smallest_subnormal() * numpy.float64(1.0)) == 0.0
    return flush_to_zero or denormals_are_zero


def _math_library():  # The C library exposing fegetenv/fesetenv, or None.
    names: list[str] = []
    if sys.platform == "win32":
        names += ["ucrtbase.dll", "msvcrt.dll"]
    else:
        found = ctypes.util.find_library("m")
        if found:
            names.append(found)
        names += ["libm.so.6", "libSystem.B.dylib", "libc.so.6"]
    for name in names:
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        if hasattr(library, "fegetenv") and hasattr(library, "fesetenv"):
            library.fegetenv.argtypes = [ctypes.c_void_p]
            library.fegetenv.restype = ctypes.c_int
            library.fesetenv.argtypes = [ctypes.c_void_p]
            library.fesetenv.restype = ctypes.c_int
            return library
    return None


def _default_environment(library) -> Any:
    """This host's ``FE_DFL_ENV``, chosen by MEASUREMENT rather than by platform table.

    The C standard names the default floating-point environment but not how to
    address it: glibc and musl define ``FE_DFL_ENV`` as the sentinel pointer
    ``(const fenv_t *) -1``, while macOS and the Windows UCRT export a real object
    (``_FE_DFL_ENV``, ``_Fenv1``). Exported symbols are tried first and the sentinel
    only where it is the documented spelling, because installing ``-1`` on a host
    that means it as an address would dereference it.

    A candidate is accepted only if the FPU stops flushing subnormals under it. The
    caller's environment is restored whether or not one is found.
    """
    saved = ctypes.create_string_buffer(_FENV_BYTES)
    if library.fegetenv(ctypes.addressof(saved)) != 0:
        return None
    candidates: list[int] = []
    for symbol in ("_FE_DFL_ENV", "_Fenv1"):
        try:
            candidates.append(ctypes.addressof(ctypes.c_char.in_dll(library, symbol)))
        except (ValueError, AttributeError):
            continue
    if not candidates and sys.platform.startswith("linux"):
        candidates.append(ctypes.c_void_p(-1).value)
    for candidate in candidates:
        if library.fesetenv(candidate) != 0:
            continue
        accepted = not subnormals_flushed()
        library.fesetenv(ctypes.addressof(saved))
        if accepted:
            return candidate
    library.fesetenv(ctypes.addressof(saved))
    return None


def _fenv():  # (library, FE_DFL_ENV) for this host, probed once.
    global _fenv_resolved
    if _fenv_resolved is None:
        library = _math_library()
        _fenv_resolved = (library, None if library is None else _default_environment(library))
    return _fenv_resolved


@contextlib.contextmanager
def keep_subnormals():
    """Run the body with subnormal values intact on this thread, then restore.

    Yields True when the guard is actually in force. A body that runs under False
    is running in whatever mode the caller was already in — which is correct when
    nothing was flushing, and is the honest report when this host's default
    environment could not be addressed.
    """
    library, default = _fenv()
    if library is None or default is None or not subnormals_flushed():
        yield False
        return
    saved = ctypes.create_string_buffer(_FENV_BYTES)
    if library.fegetenv(ctypes.addressof(saved)) != 0 or library.fesetenv(default) != 0:
        yield False
        return
    try:
        yield True
    finally:
        library.fesetenv(ctypes.addressof(saved))


def _with_subnormals(function):  # One compiler entry point, guarded and marked.
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with keep_subnormals():
            return function(*args, **kwargs)

    setattr(wrapper, _GUARD_MARK, True)
    return wrapper


def guard_kernel_compilation(cupy) -> bool:
    """Wrap CuPy's compiler front ends so kernel source parses with subnormals intact.

    Idempotent: a second call finds the wrappers already marked and leaves them
    alone. Returns whether every front end this CuPy ships is now guarded AND this
    host's default floating-point environment is addressable — i.e. whether a
    compile started under MEEP's flush-to-zero mode will actually be protected.
    """
    compiler = getattr(getattr(cupy, "cuda", None), "compiler", None)
    if compiler is None:
        try:
            compiler = importlib.import_module("cupy.cuda.compiler")
        except Exception:
            return False
    found = False
    for name in _COMPILER_ENTRY_POINTS:
        original = getattr(compiler, name, None)
        if original is None:
            continue
        found = True
        if not getattr(original, _GUARD_MARK, False):
            setattr(compiler, name, _with_subnormals(original))
    return found and _fenv()[1] is not None


def cupy_available() -> bool:  # Live probe: cupy importable AND a CUDA device present.
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        cupy = importlib.import_module("cupy")
        return int(cupy.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def available_gpu() -> str | None:
    """The GPU ``prefer_gpu=True`` resolves to here: ``"cuda"``, ``"metal"`` or ``None``.

    CUDA (CuPy importable and a device present) is probed first, then an Apple GPU
    (the three MPS reads :func:`.fastpath.metal_hardware_present` makes, so the
    backend and the planner's candidate rung share one definition). Never raises.
    """
    if cupy_available():
        return "cuda"
    from .fastpath import metal_hardware_present  # noqa: PLC0415 - one definition of the MPS reads

    return "metal" if metal_hardware_present() else None


def _missing_cuda_dependencies() -> list[str]:
    if importlib.util.find_spec("cupy") is None:
        return ["cupy"]
    return [] if cupy_available() else ["cuda-device"]


def _missing_metal_dependencies() -> list[str]:
    if importlib.util.find_spec("torch") is None:
        return ["torch"]
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - a torch that cannot import is a missing torch
        return ["torch"]
    if not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        return ["mps-device"]
    if not hasattr(torch.mps, "compile_shader"):
        return ["torch.mps.compile_shader"]
    return []


def missing_dependencies() -> list[str]:  # What this platform's GPU route lacks; [] when a route exists.
    if available_gpu() is not None:
        return []
    # NEVER EMPTY HERE: this answers why available_gpu() is None, so a read that finds
    # nothing missing still names the piece the probe could not confirm.
    if sys.platform == "darwin":
        return _missing_metal_dependencies() or ["mps-device"]
    return _missing_cuda_dependencies() or ["cuda-device"]


def _gpu_route(gpu_id: int = 0) -> str:
    """The GPU a ``prefer_gpu=True`` request resolves to here, or a named refusal."""
    gpu = available_gpu()
    if gpu is None:
        raise RuntimeError(
            "GPU backend requested but unavailable: " + ", ".join(missing_dependencies())
            + ". prefer_gpu=True runs on a CUDA device through CuPy or on an Apple GPU "
            "through Metal, and this host has neither; pass prefer_gpu=False to step "
            "the NumPy reference (host CPU, no kernels)."
        )
    if gpu == "metal" and int(gpu_id) != 0:
        raise ValueError(
            f"gpu_id={gpu_id} selects a CUDA device; this host's GPU is an Apple GPU, "
            "which has one device, so gpu_id must be 0"
        )
    return gpu


def resolve_backend(prefer_gpu: bool = False, gpu_id: int = 0) -> tuple[Any, str | None]:
    """Return ``(xp, gpu)``: the array module and the GPU the request resolved to.

    ``prefer_gpu=False`` is the NumPy reference, ``(numpy, None)``. ``True`` is this
    host's GPU: ``(cupy, "cuda")`` with the device selected on a CUDA host, or
    ``(numpy, "metal")`` on an Apple GPU, whose Metal kernels run over NumPy-owned
    fields. A request on a host with neither raises, so a caller never degrades
    silently; ``gpu_id`` other than 0 on an Apple GPU raises too.

    Selecting CuPy also guards its kernel compiler against the subnormal flushing
    MEEP switches on process-wide (see the note at the top of this module). A host
    that is flushing and whose default floating-point environment cannot be
    addressed is refused rather than left to fail later inside a CCCL header.
    """
    if prefer_gpu:
        if _gpu_route(gpu_id) == "metal":
            import numpy

            return numpy, "metal"
        cupy = importlib.import_module("cupy")
        cupy.cuda.Device(gpu_id).use()
        if not guard_kernel_compilation(cupy) and subnormals_flushed():
            raise RuntimeError(
                "This thread is flushing subnormal floating-point values to zero (MEEP sets the "
                "FTZ and DAZ bits when it initializes), and that could not be suspended for the "
                "duration of a CuPy kernel compile — either this CuPy exposes none of the "
                f"compiler entry points this guards ({', '.join(_COMPILER_ENTRY_POINTS)}), or this "
                "host's default floating-point environment (FE_DFL_ENV) could not be addressed. "
                "Any kernel whose source includes CCCL's <cuda/std/limits> — CuPy's CUB-backed "
                "reductions do — then fails to build with 'floating constant is out of range' on "
                "__FLT_DENORM_MIN__. Run the GPU backend with subnormals kept instead: call "
                "meep.set_zero_subnormals(False) before the first GPU kernel is compiled."
            )
        return cupy, "cuda"
    import numpy

    return numpy, None


def to_numpy(arr: Any) -> Any:  # Host copy for result assembly; passes NumPy arrays through.
    if hasattr(arr, "get"):
        return arr.get()
    import numpy

    return numpy.asarray(arr)
