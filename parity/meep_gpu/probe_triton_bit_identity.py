"""RECON B: can a Triton kernel be pinned to bit-identity with the array path?

The hand-CUDA track already established what bit-identity costs on this hardware
(``probe_fused_kernel_bit_identity.py`` / ``probe_fused_kernel_grouping.py``):

* NVRTC's ``--fmad=false`` is NECESSARY — with the stencil parenthesized to
  ``stepping.py``'s tree, fused output is bit-identical 16/16 with the flag and
  8/16 without.
* The gate MUST sweep a NON-POWER-OF-TWO Courant number; at ``dtdx = 0.5`` the
  discrepancy vanishes and a gate testing only 0.5 certifies broken kernels.
* Compare BYTES, never magnitudes.

This probe asks the same three questions of Triton, which compiles through
MLIR -> LLVM -> PTX -> ptxas rather than through NVRTC, and therefore has its own
contraction policy and its own knob (or no knob) for it.

Four sections, each writing incrementally into the JSON:

1. ``environment``  — Triton / torch / CuPy / driver / arch / ptxas versions.
2. ``ptx``          — the same instruction census the hand track ran: are the f32
                      ops ``.rn`` (non-contractible per the PTX ISA) or not, and
                      how many ``fma.rn.f32`` did the compiler synthesize?
3. ``bit_identity`` — bytewise comparison against the CuPy array path for the
                      periodic real-field triple curl AND for the split-field PML
                      recurrence (``stepping.py:1975-1982``), which is the
                      multiply-add-shaped expression most exposed to contraction.
4. ``autotune`` / ``interop`` / ``ergonomics`` — measured, not guessed.

Usage::

    CUDA_VISIBLE_DEVICES=<free gpu> python -u probe_triton_bit_identity.py \
        --out results/triton_feasibility.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

SEED = 20260809

# ---------------------------------------------------------------------------
# The CuPy array-path reference (transcribed from meep_gpu/stepping.py)
# ---------------------------------------------------------------------------

B_TERMS = (
    # target, first, first_axis, second, second_axis   (stepping.py B_CURL_TERMS)
    ("Bx", "Ez", 1, "Ey", 2),
    ("By", "Ex", 2, "Ez", 0),
    ("Bz", "Ey", 0, "Ex", 1),
)


def log(msg: str) -> None:
    print(msg, flush=True)


def save(results: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Bit comparison — identical definitions to probe_fused_kernel_grouping.py
# ---------------------------------------------------------------------------

def _ordered_key(raw: np.ndarray) -> np.ndarray:
    signed = raw.view(np.int32).astype(np.int64)
    return np.where(signed < 0, np.int64(-2147483648) - signed, signed)


def bit_compare(a_dev: Any, b_dev: Any) -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415

    a = np.ascontiguousarray(cp.asnumpy(a_dev))
    b = np.ascontiguousarray(cp.asnumpy(b_dev))
    af = a.view(np.float32).ravel() if a.dtype == np.complex64 else a.ravel()
    bf = b.view(np.float32).ravel() if b.dtype == np.complex64 else b.ravel()
    ua, ub = af.view(np.uint32), bf.view(np.uint32)
    identical = bool(np.array_equal(ua, ub))
    out: Dict[str, Any] = {
        "bit_identical": identical,
        "differing_floats": int(np.count_nonzero(ua != ub)),
        "total_floats": int(ua.size),
        # The trap this project already fell into once: a magnitude comparison
        # calls a signed-zero difference identical. Recorded beside the byte
        # verdict so the two can never be confused in the record.
        "allclose_would_say": bool(np.allclose(af, bf, rtol=0, atol=0)),
    }
    if not identical:
        ulp = np.abs(_ordered_key(af) - _ordered_key(bf))
        out["max_ulp"] = int(ulp.max())
        out["max_abs_diff"] = float(
            np.max(np.abs(af.astype(np.float64) - bf.astype(np.float64))))
    return out


def combine(parts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    identical = all(p["bit_identical"] for p in parts.values())
    out: Dict[str, Any] = {
        "bit_identical": identical,
        "differing_floats": sum(p["differing_floats"] for p in parts.values()),
        "total_floats": sum(p["total_floats"] for p in parts.values()),
        "per_component": parts,
    }
    if not identical:
        out["max_ulp"] = max(p.get("max_ulp", 0) for p in parts.values())
        out["max_abs_diff"] = max(p.get("max_abs_diff", 0.0) for p in parts.values())
    return out


# ---------------------------------------------------------------------------
# Section 1: environment
# ---------------------------------------------------------------------------

def collect_environment() -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415
    import triton  # noqa: PLC0415

    env: Dict[str, Any] = {}
    env["triton_version"] = triton.__version__
    env["triton_path"] = os.path.dirname(triton.__file__)
    try:
        import torch  # noqa: PLC0415
        env["torch_version"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        env["torch_version"] = f"ABSENT: {type(exc).__name__}: {exc}"
    env["cupy_version"] = cp.__version__
    env["cuda_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
    env["cuda_driver_version"] = cp.cuda.runtime.driverGetVersion()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    env["gpu_name"] = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    env["compute_capability"] = f"{props['major']}.{props['minor']}"
    env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    # Triton builds a small ``cuda_utils`` C extension at first launch and links it
    # with ``-lcuda``; that needs a ``libcuda.so`` DEV SYMLINK, which a driver-only
    # install does not ship (only ``libcuda.so.1``). Recorded because it is a
    # deployment precondition, not an accident of this box.
    env["triton_libcuda_path"] = os.environ.get("TRITON_LIBCUDA_PATH", "<unset>")
    env["python"] = sys.version.split()[0]
    # Triton ships its own ptxas; record which one it will use.
    for label, path in (
        ("triton_ptxas", os.path.join(os.path.dirname(triton.__file__),
                                      "backends", "nvidia", "bin", "ptxas")),
    ):
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True,
                                 timeout=30)
            env[label] = out.stdout.strip().splitlines()[-1] if out.stdout else out.stderr[:200]
            env[label + "_path"] = path
        except Exception as exc:  # noqa: BLE001
            env[label] = f"{type(exc).__name__}: {exc}"
    # What compile options does the nvidia backend accept? This is the answer to
    # "is there an --fmad=false equivalent", read off the backend itself rather
    # than guessed from documentation.
    try:
        from triton.backends.nvidia.compiler import CUDAOptions  # noqa: PLC0415
        import dataclasses  # noqa: PLC0415
        env["cuda_options_fields"] = {
            f.name: repr(f.default) for f in dataclasses.fields(CUDAOptions)
        }
    except Exception as exc:  # noqa: BLE001
        env["cuda_options_fields"] = f"{type(exc).__name__}: {exc}"
    return env


# ---------------------------------------------------------------------------
# The kernels under test
# ---------------------------------------------------------------------------
# Written with tl.* only; no torch tensors anywhere in the path. The stencil is
# parenthesized to stepping.py's tree -- ``dtdx * ((sf - f) + (s - ss))`` -- which
# the hand track measured to be the OTHER necessary condition beside the fmad flag.

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def triton_step_B(Bx, By, Bz, Ex, Ey, Ez,
                  nx, ny, nz, n_elem, dtdx,
                  BLOCK: tl.constexpr):
    """B -= dtdx * ((g1[i+1] - g1[i]) + (g2[i] - g2[i+1])), periodic, real fp32."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem

    k = idx % nz
    j = (idx // nz) % ny
    i = idx // (ny * nz)

    ip = (i + 1) % nx
    jp = (j + 1) % ny
    kp = (k + 1) % nz

    idx_ip = ip * ny * nz + j * nz + k
    idx_jp = i * ny * nz + jp * nz + k
    idx_kp = i * ny * nz + j * nz + kp

    ex = tl.load(Ex + idx, mask=live, other=0.0)
    ey = tl.load(Ey + idx, mask=live, other=0.0)
    ez = tl.load(Ez + idx, mask=live, other=0.0)
    ex_jp = tl.load(Ex + idx_jp, mask=live, other=0.0)
    ex_kp = tl.load(Ex + idx_kp, mask=live, other=0.0)
    ey_ip = tl.load(Ey + idx_ip, mask=live, other=0.0)
    ey_kp = tl.load(Ey + idx_kp, mask=live, other=0.0)
    ez_ip = tl.load(Ez + idx_ip, mask=live, other=0.0)
    ez_jp = tl.load(Ez + idx_jp, mask=live, other=0.0)

    bx = tl.load(Bx + idx, mask=live, other=0.0)
    by = tl.load(By + idx, mask=live, other=0.0)
    bz = tl.load(Bz + idx, mask=live, other=0.0)

    curl_x = dtdx * ((ez_jp - ez) + (ey - ey_kp))
    curl_y = dtdx * ((ex_kp - ex) + (ez - ez_ip))
    curl_z = dtdx * ((ey_ip - ey) + (ex - ex_jp))

    tl.store(Bx + idx, bx - curl_x, mask=live)
    tl.store(By + idx, by - curl_y, mask=live)
    tl.store(Bz + idx, bz - curl_z, mask=live)


@triton.jit
def triton_pml_update(field, curl, fu, kms, sinv, kms_u, sinv_u,
                      ny, nz, n_elem, BLOCK: tl.constexpr):
    """The split-field PML curl recurrence, stepping.py:1975-1982.

        fu_prev = fu
        fu = ((kms * fu) - curl) * sinv
        field = ((kms_u * field) + fu - fu_prev) * sinv_u

    ``kms``/``sinv``/``kms_u``/``sinv_u`` are the broadcast-shaped (nx, 1, 1)
    coefficient columns the array path multiplies by, so the kernel indexes them
    at ``i = idx // (ny * nz)``.

    This is the expression that decides the question: ``a * b - c`` is exactly the
    shape a compiler contracts into an FMA.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    i = idx // (ny * nz)

    a_kms = tl.load(kms + i, mask=live, other=0.0)
    a_sinv = tl.load(sinv + i, mask=live, other=0.0)
    a_kms_u = tl.load(kms_u + i, mask=live, other=0.0)
    a_sinv_u = tl.load(sinv_u + i, mask=live, other=0.0)

    f = tl.load(field + idx, mask=live, other=0.0)
    c = tl.load(curl + idx, mask=live, other=0.0)
    u_prev = tl.load(fu + idx, mask=live, other=0.0)

    # In-place order of the array path: fu *= kms; fu -= curl; fu *= sinv
    u = u_prev * a_kms
    u = u - c
    u = u * a_sinv

    # field *= kms_u; field += fu; field -= fu_previous; field *= sinv_u
    f = f * a_kms_u
    f = f + u
    f = f - u_prev
    f = f * a_sinv_u

    tl.store(fu + idx, u, mask=live)
    tl.store(field + idx, f, mask=live)


# ---------------------------------------------------------------------------
# CuPy <-> Triton interop
# ---------------------------------------------------------------------------

class CupyPointer:
    """Duck-typed pointer argument: what Triton needs from a tensor, from CuPy.

    Triton's launcher resolves a pointer argument by calling ``arg.data_ptr()``
    and types it from ``arg.dtype``. A CuPy ndarray carries the address at
    ``.data.ptr`` under a different name and has no ``data_ptr`` method, so this
    adapter is the whole of the interop layer: no torch tensor, no device copy,
    no autograd, and the wrapped array is the same allocation the rest of the
    engine already holds.

    ``clone``/``copy_`` are the torch-tensor names ``triton.autotune``'s
    ``restore_value=`` hook calls on in-place arguments. They are not needed to
    LAUNCH a kernel — only to autotune one that writes its inputs, which every
    kernel on this step path does.
    """

    __slots__ = ("_array",)

    def __init__(self, array: Any) -> None:
        self._array = array

    def data_ptr(self) -> int:
        return int(self._array.data.ptr)

    @property
    def dtype(self):
        return self._array.dtype

    def clone(self) -> "CupyPointer":
        return CupyPointer(self._array.copy())

    def copy_(self, other: "CupyPointer") -> "CupyPointer":
        self._array[...] = other._array
        return self


def wrap(array: Any) -> Any:
    return CupyPointer(array)


# ---------------------------------------------------------------------------
# Section 2: PTX census
# ---------------------------------------------------------------------------

F32_OPS = ("add", "sub", "mul", "fma", "div")


def census(ptx: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for op in F32_OPS:
        counts[f"{op}.rn.f32"] = ptx.count(f"{op}.rn.f32")
        # Plain (non-.rn) spellings are the contractible ones ptxas may fuse.
        plain = 0
        for line in ptx.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{op}.f32"):
                plain += 1
        counts[f"{op}.f32_plain"] = plain
    counts["ptx_chars"] = len(ptx)
    return counts


def compiled_of(jit_fn) -> Any:
    """The single CompiledKernel a warmed JITFunction is holding."""
    caches = getattr(jit_fn, "cache", {})
    for per_device in caches.values():
        for compiled in per_device.values():
            return compiled
    return None


# ---------------------------------------------------------------------------
# Case runners
# ---------------------------------------------------------------------------

def make_arrays(shape: Tuple[int, int, int], rng) -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415

    names = ("Bx", "By", "Bz", "Ex", "Ey", "Ez")
    return {
        n: cp.asarray(np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)))
        for n in names
    }


def reference_step_B(arrays: Dict[str, Any], dtdx: float) -> Dict[str, Any]:
    """stepping.py's periodic Cartesian B curl + ``target -= curl``, verbatim."""
    import cupy as cp  # noqa: PLC0415

    out = {}
    for target, first, first_axis, second, second_axis in B_TERMS:
        f = arrays[first]
        s = arrays[second]
        sf = cp.roll(f, -1, axis=first_axis)
        ss = cp.roll(s, -1, axis=second_axis)
        curl = dtdx * ((sf - f) + (s - ss))
        updated = arrays[target].copy()
        updated -= curl
        out[target] = updated
    return out


def reference_pml(field, curl, fu, kms, sinv, kms_u, sinv_u):
    """stepping.py:1975-1982 on copies."""
    field = field.copy()
    fu = fu.copy()
    fu_previous = fu.copy()
    fu *= kms
    fu -= curl
    fu *= sinv
    field *= kms_u
    field += fu
    field -= fu_previous
    field *= sinv_u
    return field, fu


def run_curl_case(shape, dtdx, fp_fusion: bool, block: int) -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415

    rng = np.random.default_rng(SEED)
    arrays = make_arrays(shape, rng)
    expected = reference_step_B(arrays, dtdx)

    got = {n: arrays[n].copy() for n in ("Bx", "By", "Bz")}
    nx, ny, nz = shape
    n_elem = nx * ny * nz
    grid = ((n_elem + block - 1) // block,)

    kwargs: Dict[str, Any] = {"BLOCK": block, "num_warps": 4}
    if not fp_fusion:
        kwargs["enable_fp_fusion"] = False
    triton_step_B[grid](
        wrap(got["Bx"]), wrap(got["By"]), wrap(got["Bz"]),
        wrap(arrays["Ex"]), wrap(arrays["Ey"]), wrap(arrays["Ez"]),
        nx, ny, nz, n_elem, float(dtdx), **kwargs)
    cp.cuda.runtime.deviceSynchronize()

    parts = {n: bit_compare(got[n], expected[n]) for n in ("Bx", "By", "Bz")}
    case = combine(parts)
    case.update({"shape": list(shape), "dtdx": dtdx, "enable_fp_fusion": fp_fusion,
                 "block": block, "kernel": "triton_step_B"})
    compiled = compiled_of(triton_step_B)
    if compiled is not None:
        case["ptx"] = census(compiled.asm["ptx"])
        case["n_regs"] = getattr(compiled, "n_regs", None)
    return case


def run_pml_case(shape, dtdx, fp_fusion: bool, block: int) -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415

    rng = np.random.default_rng(SEED + 1)
    nx, ny, nz = shape
    n_elem = nx * ny * nz

    def vol():
        return cp.asarray(np.ascontiguousarray(
            rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)))

    def column():
        # Coefficients in the range MEEP's PML actually produces; deliberately
        # NOT round numbers, for the same reason the Courant must not be 0.5.
        return cp.asarray(np.ascontiguousarray(
            rng.uniform(0.31, 0.97, size=(nx, 1, 1)).astype(np.float32)))

    field = vol()
    curl = dtdx * vol()
    fu = vol()
    kms, sinv, kms_u, sinv_u = column(), column(), column(), column()

    exp_field, exp_fu = reference_pml(field, curl, fu, kms, sinv, kms_u, sinv_u)

    got_field = field.copy()
    got_fu = fu.copy()
    grid = ((n_elem + block - 1) // block,)
    kwargs: Dict[str, Any] = {"BLOCK": block, "num_warps": 4}
    if not fp_fusion:
        kwargs["enable_fp_fusion"] = False
    triton_pml_update[grid](
        wrap(got_field), wrap(curl), wrap(got_fu),
        wrap(kms), wrap(sinv), wrap(kms_u), wrap(sinv_u),
        ny, nz, n_elem, **kwargs)
    cp.cuda.runtime.deviceSynchronize()

    parts = {"field": bit_compare(got_field, exp_field),
             "fu": bit_compare(got_fu, exp_fu)}
    case = combine(parts)
    case.update({"shape": list(shape), "dtdx": dtdx, "enable_fp_fusion": fp_fusion,
                 "block": block, "kernel": "triton_pml_update"})
    compiled = compiled_of(triton_pml_update)
    if compiled is not None:
        case["ptx"] = census(compiled.asm["ptx"])
        case["n_regs"] = getattr(compiled, "n_regs", None)
    return case


# ---------------------------------------------------------------------------
# Section 4: autotune + interop evidence
# ---------------------------------------------------------------------------

def run_autotune(shape, dtdx) -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415

    configs = [triton.Config({"BLOCK": b}, num_warps=w, num_stages=s)
               for b in (128, 256, 512, 1024)
               for w in (1, 2, 4, 8)
               for s in (1, 2)]

    tuned = triton.autotune(configs=configs, key=["n_elem"])(
        triton.jit(triton_step_B.fn))

    rng = np.random.default_rng(SEED)
    arrays = make_arrays(shape, rng)
    expected = reference_step_B(arrays, dtdx)
    got = {n: arrays[n].copy() for n in ("Bx", "By", "Bz")}
    nx, ny, nz = shape
    n_elem = nx * ny * nz

    started = time.time()
    tuned[lambda meta: ((n_elem + meta["BLOCK"] - 1) // meta["BLOCK"],)](
        wrap(got["Bx"]), wrap(got["By"]), wrap(got["Bz"]),
        wrap(arrays["Ex"]), wrap(arrays["Ey"]), wrap(arrays["Ez"]),
        nx, ny, nz, n_elem, float(dtdx), enable_fp_fusion=False)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.time() - started

    parts = {n: bit_compare(got[n], expected[n]) for n in ("Bx", "By", "Bz")}
    out = combine(parts)
    best = getattr(tuned, "best_config", None)
    out.update({
        "shape": list(shape),
        "dtdx": dtdx,
        "n_configs": len(configs),
        "tuning_seconds": round(elapsed, 2),
        "best_config": str(best),
    })
    # Timings for the top handful, if triton kept them.
    cache = getattr(tuned, "cache", {})
    out["cache_keys"] = [str(k) for k in list(cache)[:4]]
    return out


def run_interop_evidence(shape) -> Dict[str, Any]:
    """Prove the CuPy path needs no torch tensor, and record what it DOES need."""
    import cupy as cp  # noqa: PLC0415

    out: Dict[str, Any] = {}
    out["torch_imported_before_launch"] = "torch" in sys.modules

    # 1. Bare CuPy ndarray, no adapter.
    rng = np.random.default_rng(SEED)
    arrays = make_arrays(shape, rng)
    nx, ny, nz = shape
    n_elem = nx * ny * nz
    grid = ((n_elem + 256 - 1) // 256,)
    try:
        triton_step_B[grid](
            arrays["Bx"], arrays["By"], arrays["Bz"],
            arrays["Ex"], arrays["Ey"], arrays["Ez"],
            nx, ny, nz, n_elem, 0.35, BLOCK=256, enable_fp_fusion=False)
        cp.cuda.runtime.deviceSynchronize()
        out["bare_cupy_ndarray"] = "ACCEPTED"
    except Exception as exc:  # noqa: BLE001
        out["bare_cupy_ndarray"] = f"REJECTED: {type(exc).__name__}: {exc}"[:400]

    # 2. Does Triton reach for torch to find the stream?
    try:
        from triton.runtime.driver import driver  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        # driver.active is a LazyProxy until something forces it; a bound method
        # lookup on the proxy resolves it.
        bound = driver.active.get_current_stream
        src = inspect.getsource(bound)
        out["get_current_stream_source"] = src.strip()
        out["stream_lookup_uses_torch"] = "torch" in src
        out["driver_class"] = type(driver.active).__name__
    except Exception as exc:  # noqa: BLE001
        out["get_current_stream_source"] = f"{type(exc).__name__}: {exc}"

    out["torch_imported_after_launch"] = "torch" in sys.modules

    # 3. WHICH stream does Triton pick when CuPy has a non-default one current?
    # If Triton reads torch's stream while the engine's work is queued on CuPy's,
    # the two are unordered and every launch needs an explicit sync.
    try:
        from triton.runtime.driver import driver  # noqa: PLC0415
        probe_stream = cp.cuda.Stream(non_blocking=True)
        with probe_stream:
            out["cupy_current_stream_ptr"] = int(cp.cuda.get_current_stream().ptr)
            out["triton_current_stream_ptr"] = int(driver.active.get_current_stream(0))
        out["streams_agree"] = (out["cupy_current_stream_ptr"]
                                == out["triton_current_stream_ptr"])
    except Exception as exc:  # noqa: BLE001
        out["streams_agree"] = f"{type(exc).__name__}: {exc}"[:300]

    # 4. Does the kernel still produce a correct result launched inside one?
    try:
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            triton_step_B[grid](
                wrap(arrays["Bx"]), wrap(arrays["By"]), wrap(arrays["Bz"]),
                wrap(arrays["Ex"]), wrap(arrays["Ey"]), wrap(arrays["Ez"]),
                nx, ny, nz, n_elem, 0.35, BLOCK=256, enable_fp_fusion=False)
        stream.synchronize()
        out["launch_inside_cupy_stream_context"] = "OK (see stream_lookup note)"
    except Exception as exc:  # noqa: BLE001
        out["launch_inside_cupy_stream_context"] = f"{type(exc).__name__}: {exc}"[:400]

    return out


def run_ergonomics() -> Dict[str, Any]:
    """Line counts, measured off the two source files rather than estimated."""
    import inspect  # noqa: PLC0415

    def body_lines(fn) -> int:
        src = inspect.getsource(fn.fn if hasattr(fn, "fn") else fn)
        return sum(1 for line in src.splitlines()
                   if line.strip() and not line.strip().startswith("#")
                   and not line.strip().startswith('"'))

    out = {
        "triton_step_B_lines": body_lines(triton_step_B),
        "triton_pml_update_lines": body_lines(triton_pml_update),
    }
    hand = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "probe_fused_kernel_grouping.py")
    try:
        text = open(hand, encoding="utf-8").read()
        start = text.index("FIXED_B_REAL = r'''")
        end = text.index("'''", start + 20)
        block = text[start:end]
        out["hand_cuda_FIXED_B_REAL_lines"] = sum(
            1 for line in block.splitlines()
            if line.strip() and not line.strip().startswith("//"))
    except Exception as exc:  # noqa: BLE001
        out["hand_cuda_FIXED_B_REAL_lines"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/triton_feasibility.json")
    parser.add_argument("--skip-autotune", action="store_true")
    args = parser.parse_args(argv)

    results: Dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    log("[env] collecting")
    results["environment"] = collect_environment()
    save(results, args.out)
    for key in ("triton_version", "torch_version", "cupy_version", "gpu_name",
                "compute_capability", "cuda_driver_version", "triton_ptxas",
                "cuda_visible_devices"):
        log(f"[env] {key}: {results['environment'].get(key)}")
    log(f"[env] CUDAOptions fields: {results['environment'].get('cuda_options_fields')}")

    # The sweep. dtdx 0.5 is the POWER-OF-TWO CONTROL that must NOT be trusted
    # alone; 0.35 and 0.3 are the discriminating values.
    shapes = [(13, 11, 9), (32, 32, 32)]
    dtdx_values = [0.5, 0.35, 0.3]
    fusion_modes = [True, False]

    cases: List[Dict[str, Any]] = results.setdefault("bit_identity", [])
    total = len(shapes) * len(dtdx_values) * len(fusion_modes) * 2
    index = 0
    for fp_fusion in fusion_modes:
        for shape in shapes:
            for dtdx in dtdx_values:
                for runner, label in ((run_curl_case, "curl"), (run_pml_case, "pml")):
                    index += 1
                    started = time.time()
                    try:
                        case = runner(shape, dtdx, fp_fusion, 256)
                    except Exception as exc:  # noqa: BLE001
                        case = {"kernel": label, "shape": list(shape), "dtdx": dtdx,
                                "enable_fp_fusion": fp_fusion,
                                "error": f"{type(exc).__name__}: {exc}"[:800],
                                "bit_identical": False}
                    case["seconds"] = round(time.time() - started, 3)
                    cases.append(case)
                    save(results, args.out)
                    ptx = case.get("ptx", {})
                    log(f"[bit] case {index}/{total} {label} fp_fusion={fp_fusion} "
                        f"shape={shape} dtdx={dtdx!r}: "
                        f"identical={case.get('bit_identical')} "
                        f"diff={case.get('differing_floats')}/{case.get('total_floats')} "
                        f"maxulp={case.get('max_ulp', 0)} "
                        f"fma.rn={ptx.get('fma.rn.f32')} "
                        f"add.rn={ptx.get('add.rn.f32')} sub.rn={ptx.get('sub.rn.f32')} "
                        f"mul.rn={ptx.get('mul.rn.f32')} "
                        f"plain={ptx.get('add.f32_plain')}/{ptx.get('sub.f32_plain')}/"
                        f"{ptx.get('mul.f32_plain')}/{ptx.get('fma.f32_plain')} "
                        f"err={case.get('error', '')} ({case['seconds']} s)")
                    # A fresh JIT cache per case, so each case's PTX is its own.
                    triton_step_B.cache.clear()
                    triton_pml_update.cache.clear()

    log("[interop] measuring")
    try:
        results["interop"] = run_interop_evidence((16, 16, 16))
    except Exception as exc:  # noqa: BLE001
        results["interop"] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
    save(results, args.out)
    log(f"[interop] {json.dumps(results['interop'])[:900]}")

    log("[ergonomics] measuring")
    results["ergonomics"] = run_ergonomics()
    save(results, args.out)
    log(f"[ergonomics] {results['ergonomics']}")

    if not args.skip_autotune:
        log("[autotune] running (32 configs)")
        started = time.time()
        try:
            results["autotune"] = run_autotune((64, 64, 64), 0.35)
        except Exception as exc:  # noqa: BLE001
            results["autotune"] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
        save(results, args.out)
        log(f"[autotune] ({round(time.time()-started,1)} s) "
            f"{json.dumps(results['autotune'])[:600]}")

    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save(results, args.out)
    log(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
