"""What the Triton track's generated code and platform facts mean for the HAND-CUDA path.

This probe exists because the two tracks were certified under different tooling and
the CUDA side's record answers questions the Triton side answered explicitly. It
measures rather than reasons by analogy. Six independent sections, each written to
the output JSON as it finishes so an interrupted run keeps everything before the
failure:

1. ``shipped_options`` — the option tuple CuPy ACTUALLY hands NVRTC for a
   ``cp.RawKernel`` built with ``step_curl_kernels._COMPILE_OPTIONS``. Captured by
   wrapping ``cupy.cuda.compiler.compile_using_nvrtc`` (the seam
   ``subnormal_policy`` strips at) and calling through. Also reads each compiled
   kernel's ``num_regs`` / ``local_size_bytes`` / ``shared_size_bytes``, which is
   the direct counterpart of Triton's ``n_regs`` / ``n_spills`` / ``shared``.
2. ``cuda_ptx`` — real PTX for all fourteen registered kernels under four option
   sets, via NVRTC's own ``getPTX`` (CuPy's normal return value is a CUBIN, not
   PTX). The four sets are the shipped tuple as captured, the shipped tuple with
   ``-ftz=true`` removed (what the ``keep`` policy produces), the bare
   ``('--fmad=false',)`` the 2026-08-09 record used, and no options at all.
3. ``micro_ops`` — one-line kernels that settle the two lowering questions the
   Triton track raised: what ``-x`` lowers to, and what ``/`` lowers to. Compiled
   AND run, so the answer is bytes and not only opcodes.
4. ``triton_ptx`` — the shipped Triton kernels' PTX, register counts and launch
   metadata at several ``num_warps``, censused with the SAME function as the CUDA
   side so the two columns are comparable.
5. ``cache_staleness`` — whether the CuPy/RawKernel path has the Triton cache's
   stale-binary failure mode, at both levels: the kernel module's own memo (in
   ``cuda_kernels/compile_cache.py`` since 2026-08-15; its key is READ from the
   live module rather than asserted here) and CuPy's source-keyed disk cache.
6. ``occupancy`` — the launch geometry each track picks, side by side.

NOTHING HERE MUTATES THE TREE. It imports ``meep_gpu`` read-only.

Usage (compile-only sections need a VISIBLE device to read the architecture off but
no FREE one; sections 1, 3-run and 4 need it free)::

    CUDA_VISIBLE_DEVICES=<one verified-free gpu> python -u probe_cuda_ptx_transfer.py \
        --out results/<dir>/cuda_ptx_transfer.json --ptx-dir results/<dir>/ptx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SEED = 20260814

#: The fourteen entries in ``step_curl_kernels._get_kernel``'s code map, as
#: (cache key name, is_complex, module attribute holding the source, entry point).
CUDA_KERNELS: Tuple[Tuple[str, bool, str, str], ...] = (
    ("step_B", False, "_triple_curl_B_kernel_code", "step_B"),
    ("step_B", True, "_triple_curl_B_kernel_complex_code", "step_B_complex"),
    ("step_D", False, "_triple_curl_D_kernel_code", "step_D"),
    ("step_D", True, "_triple_curl_D_kernel_complex_code", "step_D_complex"),
    ("step_B_sym", True, "_triple_curl_B_sym_kernel_complex_code", "step_B_sym_complex"),
    ("step_D_sym", True, "_triple_curl_D_sym_kernel_complex_code", "step_D_sym_complex"),
    ("step_B_pml_real", False, "_step_B_pml_real_kernel_code", "step_B_pml_real"),
    ("step_D_pml_real", False, "_step_D_pml_real_kernel_code", "step_D_pml_real"),
    ("step_B_pml", True, "_step_B_pml_kernel_complex_code", "step_B_pml_complex"),
    ("step_D_pml", True, "_step_D_pml_kernel_complex_code", "step_D_pml_complex"),
    ("step_B_pml_sym", True, "_step_B_pml_sym_kernel_complex_code", "step_B_pml_sym_complex"),
    ("step_D_pml_sym", True, "_step_D_pml_sym_kernel_complex_code", "step_D_pml_sym_complex"),
    ("update_H_pml", True, "_update_H_pml_kernel_complex_code", "update_H_pml_complex"),
    ("update_E_pml", True, "_update_E_pml_kernel_complex_code", "update_E_pml_complex"),
)

#: Kernels whose full PTX text is written out, not merely censused.
PTX_DUMP = frozenset({
    "step_B_pml_real", "step_D_pml_real", "step_B",
    "step_B_pml_complex", "update_E_pml_complex",
})


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# census — one function, both tracks
# ---------------------------------------------------------------------------

#: f32 opcodes whose rounding or flush suffix decides arithmetic. Counted with and
#: without ``.ftz`` separately, because "which policy did this binary compile
#: under" is exactly the question the CUDA record does not answer.
_ARITH = ("add", "sub", "mul", "fma", "div", "neg", "rcp", "sqrt", "rsqrt", "mad")

_INSTRUCTION = re.compile(r"^\s*(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)")


def census(ptx: str) -> Dict[str, Any]:
    """Opcode counts that decide rounding, flushing, memory width and size."""
    counts: Dict[str, Any] = {}
    for op in _ARITH:
        # Every spelling actually present, so an unexpected variant shows up by
        # name instead of falling into a bucket nobody counted.
        found: Dict[str, int] = {}
        for match in re.finditer(rf"\b{op}((?:\.[a-z0-9]+)*)\b", ptx):
            spelling = op + match.group(1)
            if not spelling.endswith(("f32", "f64", "f16", "s32", "u32", "s64", "u64", "b32")):
                continue
            found[spelling] = found.get(spelling, 0) + 1
        if found:
            counts[op] = dict(sorted(found.items()))
    counts["_f32_with_ftz"] = len(re.findall(r"\b[a-z]+(?:\.[a-z0-9]+)*\.ftz\.f32\b", ptx))
    counts["_any_f64"] = len(re.findall(r"\.f64\b", ptx))
    counts["_cvt_f64_f32"] = ptx.count("cvt.f64.f32")
    loads: Dict[str, int] = {}
    for match in re.finditer(r"\bld\.global((?:\.[a-z0-9]+)*)\b", ptx):
        loads["ld.global" + match.group(1)] = loads.get("ld.global" + match.group(1), 0) + 1
    stores: Dict[str, int] = {}
    for match in re.finditer(r"\bst\.global((?:\.[a-z0-9]+)*)\b", ptx):
        stores["st.global" + match.group(1)] = stores.get("st.global" + match.group(1), 0) + 1
    counts["ld_global"] = dict(sorted(loads.items()))
    counts["st_global"] = dict(sorted(stores.items()))
    counts["ld_global_total"] = sum(loads.values())
    counts["st_global_total"] = sum(stores.values())
    counts["shared_decls"] = len(re.findall(r"\.shared\b", ptx))
    counts["bar_sync"] = len(re.findall(r"\bbar\.sync\b", ptx))
    counts["reg_decls"] = dict(sorted(
        (m.group(1), int(m.group(2)))
        for m in re.finditer(r"\.reg\s+\.([a-z0-9]+)\s+%\w+<(\d+)>", ptx)))
    counts["branches"] = len(re.findall(r"^\s*@?!?%?p?\d*\s*bra\b", ptx, re.M))
    counts["ptx_lines"] = ptx.count("\n")
    body = [line for line in ptx.splitlines()
            if line.strip() and not line.strip().startswith(("//", ".", "{", "}", "$"))]
    counts["instructions"] = len(body)
    return counts


# ---------------------------------------------------------------------------
# NVRTC, straight
# ---------------------------------------------------------------------------

def ptx_arch() -> str:
    """``compute_NN`` for the device THIS PROCESS has open, read when asked.

    WHY NOT A CONSTANT (changed 2026-09-30). This was ``ARCH_PTX = "compute_86"``,
    bound as ``compile_to_ptx``'s default argument — so it was fixed when the module
    was IMPORTED, and every census and every dumped ``.ptx`` file recorded a typed
    architecture rather than a measured one. On any other card this probe would
    compare sm_86 PTX against a Triton kernel compiled for the live device and
    report the difference as a property of the two tracks. PTX is per architecture;
    that is the whole reason this probe counts opcodes.

    ``Device().compute_capability`` is CuPy's own undotted string (``"86"``,
    ``"90"``, ``"100"`` for cc 10.0), already the spelling
    ``--gpu-architecture=compute_NN`` wants, so it is concatenated rather than
    re-derived — the same read ``gate_cuda_complex.py`` makes for its NVRTC guard.
    RAISES on a host with no readable device, which ``main`` records as that
    section's error.
    """
    import cupy as cp  # noqa: PLC0415 - a device read, never at import time

    return "compute_" + str(cp.cuda.Device().compute_capability)


def compile_to_ptx(code: str, options: Sequence[str], arch: str) -> str:
    """Real PTX text for one kernel, from NVRTC's own ``getPTX``.

    ``compile_using_nvrtc`` returns a CUBIN on CuPy 13.5.1, so counting opcodes in
    its result measures nothing. Needs no CUDA context — only the NVRTC library and
    a visible device for :func:`ptx_arch` to read.

    ``arch`` IS REQUIRED and has no default: a default would be evaluated once at
    import and no caller could tell a measured architecture from a stale one. Each
    section reads it once and records the value it compiled under.
    """
    from cupy.cuda import nvrtc  # noqa: PLC0415

    program = nvrtc.createProgram(code, "kernel.cu", [], [])
    try:
        nvrtc.compileProgram(program, tuple(options) + (f"--gpu-architecture={arch}",))
        text = nvrtc.getPTX(program).decode()
    finally:
        nvrtc.destroyProgram(program)
    if ".visible .entry" not in text:
        raise RuntimeError("NVRTC did not return PTX (no '.visible .entry')")
    return text


def _cupy_include() -> str:
    import cupy as cp  # noqa: PLC0415

    return "-I" + os.path.join(os.path.dirname(cp.__file__), "_core", "include")


# ---------------------------------------------------------------------------
# section 1 — what CuPy actually passes, and what the compiled kernel costs
# ---------------------------------------------------------------------------

def section_shipped_options(results: Dict[str, Any]) -> None:
    import cupy as cp  # noqa: PLC0415
    from cupy.cuda import compiler as cupy_compiler  # noqa: PLC0415

    from meep_gpu.cuda_kernels import step_curl_kernels as sck  # noqa: PLC0415

    captured: List[Dict[str, Any]] = []
    original = cupy_compiler.compile_using_nvrtc

    def spy(source, options=(), *args, **kwargs):
        captured.append({
            "options": [str(o) for o in tuple(options)],
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        })
        return original(source, options, *args, **kwargs)

    out: Dict[str, Any] = {
        "module_compile_options": list(sck._COMPILE_OPTIONS),
        "cupy_version": cp.__version__,
        "seam": "cupy.cuda.compiler.compile_using_nvrtc",
        "kernels": {},
    }
    cupy_compiler.compile_using_nvrtc = spy
    try:
        for name, is_complex, _attr, entry in CUDA_KERNELS:
            before = len(captured)
            record: Dict[str, Any] = {"entry": entry, "is_complex": is_complex}
            started = time.time()
            try:
                kernel = sck._get_kernel(name, is_complex)
                # Touching an attribute forces the lazy NVRTC compile + module load.
                record.update(
                    num_regs=int(kernel.num_regs),
                    local_size_bytes=int(kernel.local_size_bytes),
                    shared_size_bytes=int(kernel.shared_size_bytes),
                    const_size_bytes=int(kernel.const_size_bytes),
                    max_threads_per_block=int(kernel.max_threads_per_block),
                )
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"[:400]
            record["nvrtc_calls"] = captured[before:]
            record["seconds"] = round(time.time() - started, 3)
            out["kernels"][entry] = record
            log(f"[opts] {entry}: regs={record.get('num_regs')} "
                f"spill={record.get('local_size_bytes')} "
                f"smem={record.get('shared_size_bytes')} "
                f"nvrtc_calls={len(record['nvrtc_calls'])}")
    finally:
        cupy_compiler.compile_using_nvrtc = original

    tuples = [tuple(c["options"]) for c in captured]
    out["distinct_option_tuples"] = [list(t) for t in sorted(set(tuples))]
    out["ftz_appended_by_cupy"] = any("-ftz=true" in t or "--ftz=true" in t for t in tuples)
    out["compiles_observed"] = len(captured)
    results["shipped_options"] = out


# ---------------------------------------------------------------------------
# section 2 — PTX for all fourteen, four option sets
# ---------------------------------------------------------------------------

def option_sets(shipped: Optional[Sequence[str]]) -> List[Tuple[str, Tuple[str, ...]]]:
    include = _cupy_include()
    sets: List[Tuple[str, Tuple[str, ...]]] = []
    if shipped:
        flush = tuple(shipped)
        keep = tuple(o for o in flush if o not in ("-ftz=true", "--ftz=true"))
        sets.append(("shipped_flush", flush + (include,)))
        sets.append(("shipped_keep_ftz_stripped", keep + (include,)))
    sets.append(("record_2026_08_09_fmad_false", ("--fmad=false", include)))
    sets.append(("nvrtc_default", (include,)))
    return sets


def section_cuda_ptx(results: Dict[str, Any], ptx_dir: Optional[str]) -> None:
    from meep_gpu.cuda_kernels import step_curl_kernels as sck  # noqa: PLC0415

    shipped = None
    tuples = results.get("shipped_options", {}).get("distinct_option_tuples") or []
    if tuples:
        # The tuple that carries our own guard is the one the fast path compiles under.
        for candidate in tuples:
            if "--fmad=false" in candidate:
                shipped = candidate
                break
        if shipped is None:
            shipped = tuples[0]

    # READ FROM THE DEVICE, ONCE, AND RECORDED AS WHAT WAS COMPILED. An unreadable
    # device raises out of this section and ``main`` records it under
    # ``section_errors``: a census of an architecture this run never opened is a
    # false result, not a partial one.
    arch = ptx_arch()
    out: Dict[str, Any] = {"arch": arch, "shipped_tuple": shipped, "census": {}}
    if ptx_dir:
        os.makedirs(ptx_dir, exist_ok=True)
    for label, options in option_sets(shipped):
        for _name, _is_complex, attr, entry in CUDA_KERNELS:
            key = f"{label}:{entry}"
            code = getattr(sck, attr)
            try:
                text = compile_to_ptx(code, options, arch)
                out["census"][key] = census(text)
                if ptx_dir and entry in PTX_DUMP:
                    with open(os.path.join(ptx_dir, f"cuda_{entry}__{label}.ptx"), "w") as fh:
                        fh.write(text)
                log(f"[ptx] {key}: ftz_f32={out['census'][key]['_f32_with_ftz']} "
                    f"lines={out['census'][key]['ptx_lines']} "
                    f"instr={out['census'][key]['instructions']}")
            except Exception as exc:  # noqa: BLE001
                out["census"][key] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
                log(f"[ptx] {key}: FAILED {type(exc).__name__}")
    out["option_sets"] = {label: list(options) for label, options in option_sets(shipped)}
    results["cuda_ptx"] = out


# ---------------------------------------------------------------------------
# section 3 — the two lowering questions, in PTX and in bytes
# ---------------------------------------------------------------------------

_MICRO_SOURCE = r'''
extern "C" __global__ void micro_neg_product(
    float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = -(a[i] * b[i]);
}
extern "C" __global__ void micro_neg_plain(
    float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = -a[i];
}
extern "C" __global__ void micro_sub_from_zero(
    float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 0.0f - a[i];
}
extern "C" __global__ void micro_div(
    float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] / b[i];
}
extern "C" __global__ void micro_div_literal(
    float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] / 3.0f;
}
extern "C" __global__ void micro_double_scalar(
    float* out, const float* a, double s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * (float)s;
}
'''

_MICRO_ENTRIES = ("micro_neg_product", "micro_neg_plain", "micro_sub_from_zero",
                  "micro_div", "micro_div_literal", "micro_double_scalar")


def _bits(value: np.ndarray) -> List[str]:
    return ["0x%08x" % int(w) for w in value.view(np.uint32)]


def section_micro_ops(results: Dict[str, Any], ptx_dir: Optional[str]) -> None:
    import cupy as cp  # noqa: PLC0415

    shipped = results.get("cuda_ptx", {}).get("shipped_tuple")
    # THIS SECTION'S OWN READ, not ``cuda_ptx``'s. It runs independently
    # (``--sections micro_ops``), and a section that recorded the architecture of a
    # section that did not run would be recording a guess.
    arch = ptx_arch()
    out: Dict[str, Any] = {"arch": arch, "ptx": {}, "device": {}}

    for label, options in option_sets(shipped):
        try:
            text = compile_to_ptx(_MICRO_SOURCE, options, arch)
        except Exception as exc:  # noqa: BLE001
            out["ptx"][label] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        if ptx_dir:
            os.makedirs(ptx_dir, exist_ok=True)
            with open(os.path.join(ptx_dir, f"cuda_micro__{label}.ptx"), "w") as fh:
                fh.write(text)
        per_entry: Dict[str, Any] = {}
        blocks = re.split(r"\.visible \.entry ", text)
        for block in blocks[1:]:
            name = block.split("(")[0].strip()
            per_entry[name] = census(block)
        out["ptx"][label] = per_entry
        log(f"[micro] {label}: entries={sorted(per_entry)}")

    # The bytes. Needles chosen so a flush, a sign canonicalisation and an
    # approximate divide each show up as a different word.
    needles = np.array([0.0, -0.0, 1.0, -1.0, 3.0,
                        float.fromhex("0x1p-135"),   # f32 subnormal
                        float.fromhex("-0x1p-135"),
                        float.fromhex("0x1p-140"),
                        1e-30, 7.0], dtype=np.float32)
    ones = np.ones_like(needles)
    zeros = np.zeros_like(needles)
    # The DEVICE leg must go through CuPy's own compile, which appends ``-ftz=true``
    # itself — handing it the captured tuple would pass ``-ftz`` twice and NVRTC
    # hard-errors on that. So the two legs are: the module's own options (CuPy adds
    # the flush flag = today's shipped behaviour) and the same with the strip
    # installed at the seam ``subnormal_policy`` uses (= the ``keep`` policy).
    from cupy.cuda import compiler as cupy_compiler  # noqa: PLC0415

    original_compile = cupy_compiler.compile_using_nvrtc

    def stripped(source, options=(), *args, **kwargs):
        filtered = tuple(o for o in tuple(options) if o not in ("-ftz=true", "--ftz=true"))
        return original_compile(source, filtered, *args, **kwargs)

    # THE THIRD LEG IS THE POINT. CuPy's cache key is computed ABOVE the seam the
    # strip installs at, so the identical-source leg is served leg 1's flushed
    # cubin even with the strip active and the in-process memo cleared — the
    # ``keep`` policy's mandatory policy-suffixed CUPY_CACHE_DIR exists for exactly
    # this. ``cache_busted_keep`` differs from it only by a comment, which changes
    # the key and nothing else, and is therefore the VALID keep measurement.
    device_legs = (
        ("cupy_native_flush", ("--fmad=false",), False, ""),
        ("cupy_strip_keep_same_key", ("--fmad=false",), True, ""),
        ("cupy_strip_keep_cache_busted", ("--fmad=false",), True,
         "\n// cache-key buster: this leg must not be served the flush leg's cubin\n"),
    )
    try:
        for label, options, strip, suffix in device_legs:
            cupy_compiler.compile_using_nvrtc = stripped if strip else original_compile
            cp.clear_memo()
            guard = tuple(options) + (("+strip -ftz=true",) if strip else ())
            module = cp.RawModule(code=_MICRO_SOURCE + suffix, options=tuple(options))
            per: Dict[str, Any] = {"options": list(guard),
                                   "source_key_distinct": bool(suffix)}
            n = np.int32(needles.size)
            grid, block = (1,), (int(needles.size),)
            d_a = cp.asarray(needles)
            d_one = cp.asarray(ones)
            d_zero = cp.asarray(zeros)
            for entry, args in (
                ("micro_neg_product", lambda o: (o, d_a, d_one, n)),
                ("micro_neg_plain", lambda o: (o, d_a, n)),
                ("micro_sub_from_zero", lambda o: (o, d_a, n)),
                ("micro_div", lambda o: (o, d_a, d_one, n)),
                ("micro_div_literal", lambda o: (o, d_a, n)),
            ):
                d_out = cp.zeros_like(d_a)
                module.get_function(entry)(grid, block, args(d_out))
                cp.cuda.runtime.deviceSynchronize()
                per[entry] = _bits(cp.asnumpy(d_out))
            # -0.0 * +1.0 is -0.0; negating it is the discriminating case.
            d_out = cp.zeros_like(d_a)
            module.get_function("micro_neg_product")(
                grid, block, (d_out, cp.asarray(zeros * -1.0 - 0.0), d_one, n))
            cp.cuda.runtime.deviceSynchronize()
            per["neg_product_on_zeros"] = _bits(cp.asnumpy(d_out))
            per["needles"] = _bits(needles)
            out["device"][label] = per
            log(f"[micro-run] {label}: neg_plain={per['micro_neg_plain'][:4]}")
    except Exception as exc:  # noqa: BLE001
        out["device"]["error"] = f"{type(exc).__name__}: {exc}"[:400]
        log(f"[micro-run] FAILED {type(exc).__name__}: {exc}")
    finally:
        cupy_compiler.compile_using_nvrtc = original_compile
        cp.clear_memo()

    # The array path's own negation, for the same needles: this is what a hand-CUDA
    # kernel has to agree with, and it is a different question from what Triton does.
    try:
        d = cp.asarray(needles)
        out["cupy_array_path"] = {
            "negative": _bits(cp.asnumpy(-d)),
            "zero_minus": _bits(cp.asnumpy(cp.float32(0.0) - d)),
            "needles": _bits(needles),
        }
    except Exception as exc:  # noqa: BLE001
        out["cupy_array_path"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    results["micro_ops"] = out


# ---------------------------------------------------------------------------
# section 4 — the Triton side, censused identically
# ---------------------------------------------------------------------------

def _compiled_of(jit_fn) -> Any:
    for per_device in getattr(jit_fn, "cache", {}).values():
        for compiled in per_device.values():
            return compiled
    return None


def _clear(jit_fn) -> None:
    for per_device in getattr(jit_fn, "cache", {}).values():
        per_device.clear()


def section_triton_ptx(results: Dict[str, Any], ptx_dir: Optional[str]) -> None:
    import cupy as cp  # noqa: PLC0415

    from meep_gpu.triton_kernels import kernels as km  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import (  # noqa: PLC0415
        plan_constitutive_from_arrays,
        plan_from_arrays,
        plan_fused_pair_from_arrays,
    )

    rng = np.random.default_rng(SEED)
    shape = (8, 6, 4)
    names = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez",
             "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    arrays = {n: cp.asarray(np.ascontiguousarray(
        rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))) for n in names}
    curl_flat = {}
    h_flat = {}
    for axis, axis_name in enumerate("xyz"):
        for stem in ("kms", "sinv"):
            curl_flat[f"{stem}_{axis_name}"] = cp.asarray(np.ascontiguousarray(
                rng.uniform(0.5, 1.0, size=shape[axis]).astype(np.float32)))
        for stem in ("kps", "kms"):
            h_flat[f"{stem}_{axis_name}"] = cp.asarray(np.ascontiguousarray(
                rng.uniform(0.5, 1.0, size=shape[axis]).astype(np.float32)))
    codes = [1, 1, 0]
    zero_metal = (True, True, False)
    dtdx = 0.35

    out: Dict[str, Any] = {"block": 256, "shape": list(shape), "kernels": {}}
    if ptx_dir:
        os.makedirs(ptx_dir, exist_ok=True)

    plans = (
        ("pml_curl_step", km.pml_curl_step,
         lambda w: plan_from_arrays("step_B", arrays, curl_flat, codes, dtdx,
                                    block=256, num_warps=w)),
        ("constitutive_step", km.constitutive_step,
         lambda w: plan_constitutive_from_arrays("H", arrays, h_flat, block=256,
                                                 num_warps=w)),
        ("fused_curl_constitutive_B", km.fused_curl_constitutive_B,
         lambda w: plan_fused_pair_from_arrays("B", arrays, curl_flat, h_flat, codes,
                                               zero_metal, dtdx, block=256, num_warps=w)),
    )
    for warps in (1, 2, 4, None):
        for label, jit_fn, build in plans:
            key = f"{label}:num_warps={warps}"
            try:
                _clear(jit_fn)
                build(warps).run()
                cp.cuda.runtime.deviceSynchronize()
                compiled = _compiled_of(jit_fn)
                if compiled is None:
                    out["kernels"][key] = {"error": "no compiled kernel cached"}
                    continue
                ptx = compiled.asm["ptx"]
                meta = getattr(compiled, "metadata", None)
                record = {
                    "n_regs": int(getattr(compiled, "n_regs", -1)),
                    "n_spills": int(getattr(compiled, "n_spills", -1)),
                    "shared": int(getattr(meta, "shared", -1)) if meta is not None else -1,
                    "num_warps": int(getattr(meta, "num_warps", -1)) if meta is not None else -1,
                    "num_stages": int(getattr(meta, "num_stages", -1)) if meta is not None else -1,
                    "census": census(ptx),
                }
                out["kernels"][key] = record
                if ptx_dir and warps in (1, 4):
                    stem = f"triton_{label}__warps{warps}"
                    with open(os.path.join(ptx_dir, stem + ".ptx"), "w") as fh:
                        fh.write(ptx)
                    for stage in ("ttir", "ttgir", "llir"):
                        text = compiled.asm.get(stage)
                        if text:
                            with open(os.path.join(ptx_dir, f"{stem}.{stage}"), "w") as fh:
                                fh.write(text)
                log(f"[triton] {key}: regs={record['n_regs']} spills={record['n_spills']} "
                    f"shared={record['shared']} instr={record['census']['instructions']} "
                    f"ld={record['census']['ld_global_total']} "
                    f"st={record['census']['st_global_total']}")
            except Exception as exc:  # noqa: BLE001
                out["kernels"][key] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
                log(f"[triton] {key}: FAILED {type(exc).__name__}: {exc}")

    # The same two lowering questions, on the Triton side, in one artifact so the
    # CUDA and Triton answers are read off the same page rather than two campaigns.
    try:
        import triton  # noqa: PLC0415
        import triton.language as tl  # noqa: PLC0415

        from meep_gpu.triton_kernels.kernels import ENABLE_FP_FUSION  # noqa: PLC0415
        from meep_gpu.triton_kernels.launch import CupyPointer  # noqa: PLC0415

        # ``triton.jit`` resolves names against ``fn.__globals__``, so a kernel
        # defined inside a function cannot see a function-local ``tl``.
        globals()["tl"] = tl
        globals()["triton"] = triton

        @triton.jit
        def _micro(o_neg, o_negp, o_div, a, b, n, BLOCK: tl.constexpr):
            idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            live = idx < n
            x = tl.load(a + idx, mask=live, other=0.0)
            y = tl.load(b + idx, mask=live, other=0.0)
            tl.store(o_neg + idx, -x, mask=live)
            tl.store(o_negp + idx, -(x * y), mask=live)
            tl.store(o_div + idx, x / y, mask=live)

        needles = np.array([0.0, -0.0, 1.0, -1.0, 3.0,
                            float.fromhex("0x1p-135"), float.fromhex("-0x1p-135"),
                            float.fromhex("0x1p-140"), 1e-30, 7.0], dtype=np.float32)
        d_a = cp.asarray(needles)
        d_b = cp.asarray(np.ones_like(needles))
        outs = [cp.zeros_like(d_a) for _ in range(3)]
        # A plain Python int, exactly as the shipped plans pass nx/ny/nz/n_elem:
        # Triton types a numpy scalar as a pointer and the comparison then fails.
        _micro[(1,)](*[CupyPointer(o) for o in outs], CupyPointer(d_a),
                     CupyPointer(d_b), int(needles.size), BLOCK=16,
                     num_warps=1, enable_fp_fusion=ENABLE_FP_FUSION)
        cp.cuda.runtime.deviceSynchronize()
        compiled = _compiled_of(_micro)
        ptx = compiled.asm["ptx"] if compiled is not None else ""
        out["micro"] = {
            "census": census(ptx),
            "needles": _bits(needles),
            "neg": _bits(cp.asnumpy(outs[0])),
            "neg_product": _bits(cp.asnumpy(outs[1])),
            "div_by_one": _bits(cp.asnumpy(outs[2])),
        }
        if ptx_dir and ptx:
            with open(os.path.join(ptx_dir, "triton_micro.ptx"), "w") as fh:
                fh.write(ptx)
        log(f"[triton-micro] neg={out['micro']['neg'][:4]} "
            f"ops={ {k: v for k, v in out['micro']['census'].items() if k in ('neg', 'sub', 'div', 'mul')} }")
    except Exception as exc:  # noqa: BLE001
        out["micro"] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        log(f"[triton-micro] FAILED {type(exc).__name__}: {exc}")
    results["triton_ptx"] = out


# ---------------------------------------------------------------------------
# section 5 — the cache question, at both levels
# ---------------------------------------------------------------------------

_STALE_A = r'''
extern "C" __global__ void stale_probe(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f;
}
'''
_STALE_B = r'''
extern "C" __global__ void stale_probe(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0f;
}
'''
# Same BODY, different entry-point NAME: the Triton failure mode was a renamed
# mutant served the original's binary.
_STALE_C = r'''
extern "C" __global__ void stale_probe_renamed(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0f;
}
'''


def section_cache(results: Dict[str, Any]) -> None:
    import cupy as cp  # noqa: PLC0415

    from meep_gpu.cuda_kernels import step_curl_kernels as sck  # noqa: PLC0415

    out: Dict[str, Any] = {}

    def run(code: str, entry: str) -> float:
        module = cp.RawModule(code=code, options=tuple(sck._COMPILE_OPTIONS))
        d = cp.zeros(4, dtype=cp.float32)
        module.get_function(entry)((1,), (4,), (d, np.int32(4)))
        cp.cuda.runtime.deviceSynchronize()
        return float(cp.asnumpy(d)[0])

    try:
        out["cupy_disk_cache_is_source_keyed"] = {
            "same_name_body_1": run(_STALE_A, "stale_probe"),
            "same_name_body_2": run(_STALE_B, "stale_probe"),
            "renamed_same_body": run(_STALE_C, "stale_probe_renamed"),
            "cache_dir": os.environ.get("CUPY_CACHE_DIR", "<unset>"),
        }
        got = out["cupy_disk_cache_is_source_keyed"]
        got["stale_binary_served"] = bool(got["same_name_body_1"] == got["same_name_body_2"])
    except Exception as exc:  # noqa: BLE001
        out["cupy_disk_cache_is_source_keyed"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    # The kernel module's own memo: does a source edit WITHOUT a clear get served
    # the old binary? The key is read out of the live module rather than asserted
    # here — it was ``(name, is_complex)`` until 2026-08-15 and is now
    # ``(name, is_complex, options, policy, source)`` (disposition §1.2), and a
    # hardcoded label would have this artifact carry the pre-fix key beside a
    # post-fix measurement. Under the new key ``served_stale`` is False by
    # construction; under the old one it was True. Both are recorded the same way.
    try:
        sck._clear_kernel_cache()
        cache = getattr(sck, "compile_cache", None)
        first = sck._get_kernel("step_B_pml_real", False)
        keys_before = list(cache.cached_keys()) if cache is not None else []
        saved = sck._step_B_pml_real_kernel_code
        try:
            sck._step_B_pml_real_kernel_code = saved.replace(
                "float curl = dtdx *", "float curl = 0.0f * dtdx *")
            second = sck._get_kernel("step_B_pml_real", False)
            served_stale = second is first
            digests = (list(cache.compiled_source_digests())
                       if cache is not None else [])
        finally:
            sck._step_B_pml_real_kernel_code = saved
            sck._clear_kernel_cache()
        if cache is not None:
            key_shape = ("(name, is_complex, options, policy_token, source) — "
                         "%d fields" % len(keys_before[0])) if keys_before else \
                "(compile_cache.cached_keys() was empty)"
        else:
            key_shape = "(name, is_complex) only — no source digest (pre-2026-08-15)"
        out["module_memo"] = {
            "key": key_shape,
            "key_read_from": ("compile_cache.cached_keys()" if cache is not None
                              else "step_curl_kernels._compiled_kernels"),
            "source_edit_without_clear_serves_stale": bool(served_stale),
            "distinct_source_digests_after_edit": len(digests),
        }
    except Exception as exc:  # noqa: BLE001
        out["module_memo"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    results["cache_staleness"] = out
    log(f"[cache] {json.dumps(out)[:400]}")


# ---------------------------------------------------------------------------
# section 6 — launch geometry, side by side
# ---------------------------------------------------------------------------

def section_occupancy(results: Dict[str, Any]) -> None:
    from meep_gpu.cuda_kernels import step_curl_kernels as sck  # noqa: PLC0415
    from meep_gpu.triton_kernels import kernels as km  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import FUSED_DEFAULT_NUM_WARPS  # noqa: PLC0415

    results["occupancy"] = {
        "cuda_threads_per_block": int(sck._REAL_PML_THREADS),
        "cuda_threads_per_block_source": "step_curl_kernels._REAL_PML_THREADS",
        "cuda_elements_per_thread": 1,
        "cuda_warps_per_block": int(sck._REAL_PML_THREADS) // 32,
        "triton_block": int(km.DEFAULT_BLOCK),
        "triton_fused_default_num_warps": FUSED_DEFAULT_NUM_WARPS,
        "triton_elements_per_thread_at_1_warp":
            int(km.DEFAULT_BLOCK) // 32,
        "triton_elements_per_thread_at_4_warps":
            int(km.DEFAULT_BLOCK) // 128,
    }
    log(f"[occupancy] {json.dumps(results['occupancy'])}")


# ---------------------------------------------------------------------------

def environment() -> Dict[str, Any]:
    import cupy as cp  # noqa: PLC0415

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR", "<unset>"),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", "<unset>"),
        "MEEP_GPU_SUBNORMAL_POLICY": os.environ.get("MEEP_GPU_SUBNORMAL_POLICY", "<unset>"),
        "nvrtc": ".".join(str(v) for v in cp.cuda.nvrtc.getVersion()),
    }
    try:
        import triton  # noqa: PLC0415

        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["triton"] = f"unavailable: {exc}"
    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = props["name"]
        info.update(
            device=name.decode() if isinstance(name, bytes) else str(name),
            compute_capability=f"{props['major']}.{props['minor']}",
            cuda_runtime=int(cp.cuda.runtime.runtimeGetVersion()),
            driver=int(cp.cuda.runtime.driverGetVersion()),
        )
    except Exception as exc:  # noqa: BLE001
        info["device"] = f"unavailable: {exc}"
    return info


def save(results: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=False)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--ptx-dir", default=None)
    parser.add_argument("--sections", default="all")
    args = parser.parse_args()

    wanted = ("shipped_options,cuda_ptx,micro_ops,triton_ptx,cache,occupancy"
              if args.sections == "all" else args.sections)
    selected = [s.strip() for s in wanted.split(",") if s.strip()]

    results: Dict[str, Any] = {
        "probe": "cuda_ptx_transfer",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "sections_requested": selected,
        "environment": environment(),
    }
    save(results, args.out)
    log(f"[env] {json.dumps(results['environment'])}")

    steps: List[Tuple[str, Callable[[], None]]] = [
        ("shipped_options", lambda: section_shipped_options(results)),
        ("cuda_ptx", lambda: section_cuda_ptx(results, args.ptx_dir)),
        ("micro_ops", lambda: section_micro_ops(results, args.ptx_dir)),
        ("triton_ptx", lambda: section_triton_ptx(results, args.ptx_dir)),
        ("cache", lambda: section_cache(results)),
        ("occupancy", lambda: section_occupancy(results)),
    ]
    for index, (name, fn) in enumerate(steps, start=1):
        if name not in selected:
            continue
        started = time.time()
        log(f"[section {index}/{len(steps)}] {name} starting")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            results.setdefault("section_errors", {})[name] = (
                f"{type(exc).__name__}: {exc}"[:600])
            log(f"[section {index}/{len(steps)}] {name} FAILED "
                f"{type(exc).__name__}: {exc}")
        log(f"[section {index}/{len(steps)}] {name} done in "
            f"{time.time() - started:.1f} s")
        save(results, args.out)

    results["elapsed_seconds"] = round(time.time() - time.mktime(
        time.strptime(results["started_utc"], "%Y-%m-%dT%H:%M:%SZ")), 1)
    save(results, args.out)
    log("[done] " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
