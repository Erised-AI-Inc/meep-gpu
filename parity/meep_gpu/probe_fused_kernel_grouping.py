"""Addendum to ``probe_fused_kernel_bit_identity.py``: PTX evidence + the fix test.

The first probe measured that, on an RTX A6000, the shipped periodic curl kernels
reproduce their own C left-to-right grouping bit-for-bit under ``--fmad=false``
but never reproduce ``stepping.py``'s grouping. This addendum closes the two
loose ends that measurement leaves open:

1. **PTX evidence for the FMA finding.** The first probe's PTX counter compiled
   to CUBIN (CuPy asks NVRTC for ``sm_86`` by default), so its instruction counts
   were meaningless and are reported as invalid. Here NVRTC is driven directly at
   ``compute_86`` so real PTX comes back, and ``fma.rn.f32`` can be counted under
   each option spelling.

2. **Is grouping the only thing left?** Standalone *variant* kernel sources —
   written here, not in the repo — reproduce the shipped kernels exactly except
   that the stencil is parenthesized to ``stepping.py``'s tree
   (``(a - b) + (c - d)``) and the complex variants scale real and imaginary
   parts by the float ``dtdx`` directly instead of multiplying by
   ``complex<float>(dtdx, 0.0f)``. If those reach bit-identity against the array
   path, then the remaining Phase-1 kernel work is exactly those two edits.

Usage::

    CUDA_VISIBLE_DEVICES=7 python -u probe_fused_kernel_grouping.py \
        --out results/fused_kernel_grouping.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import cupy as cp

SEED = 20260804

OPTION_SETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("default_no_options", ()),
    ("fmad_false", ("--fmad=false",)),
    ("fmad_true", ("--fmad=true",)),
)

B_TERMS = (
    ("Bx", "Ez", 1, "Ey", 2),
    ("By", "Ex", 2, "Ez", 0),
    ("Bz", "Ey", 0, "Ex", 1),
)
D_TERMS = (
    ("Dx", "Hz", 1, "Hy", 2),
    ("Dy", "Hx", 2, "Hz", 0),
    ("Dz", "Hy", 0, "Hx", 1),
)

# ---------------------------------------------------------------------------
# Variant kernel sources: shipped indexing, stepping.py's arithmetic tree.
# ---------------------------------------------------------------------------

_INDEX_PROLOGUE = r'''
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
'''

FIXED_B_REAL = r'''
extern "C" __global__ void fixed_step_B(
    float* Bx, float* By, float* Bz,
    const float* Ex, const float* Ey, const float* Ez,
    int nx, int ny, int nz, float dtdx
) {''' + _INDEX_PROLOGUE + r'''
    int ip = (i + 1) % nx;
    int jp = (j + 1) % ny;
    int kp = (k + 1) % nz;

    int idx_jp = i * ny * nz + jp * nz + k;
    int idx_kp = i * ny * nz + j * nz + kp;
    int idx_ip = ip * ny * nz + j * nz + k;

    // stepping.py:1152 -- dtdx * ((shifted_first - first) + (second - shifted_second))
    Bx[idx] -= dtdx * ((Ez[idx_jp] - Ez[idx]) + (Ey[idx] - Ey[idx_kp]));
    By[idx] -= dtdx * ((Ex[idx_kp] - Ex[idx]) + (Ez[idx] - Ez[idx_ip]));
    Bz[idx] -= dtdx * ((Ey[idx_ip] - Ey[idx]) + (Ex[idx] - Ex[idx_jp]));
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 1152->1181

FIXED_B_COMPLEX = r'''
#include <cupy/complex.cuh>

extern "C" __global__ void fixed_step_B_complex(
    complex<float>* Bx, complex<float>* By, complex<float>* Bz,
    const complex<float>* Ex, const complex<float>* Ey, const complex<float>* Ez,
    int nx, int ny, int nz, float dtdx
) {''' + _INDEX_PROLOGUE + r'''
    int ip = (i + 1) % nx;
    int jp = (j + 1) % ny;
    int kp = (k + 1) % nz;

    int idx_jp = i * ny * nz + jp * nz + k;
    int idx_kp = i * ny * nz + j * nz + kp;
    int idx_ip = ip * ny * nz + j * nz + k;

    // Real-by-complex scaling, not complex<float>(dtdx, 0.0f) * z.
    complex<float> tx = (Ez[idx_jp] - Ez[idx]) + (Ey[idx] - Ey[idx_kp]);
    complex<float> ty = (Ex[idx_kp] - Ex[idx]) + (Ez[idx] - Ez[idx_ip]);
    complex<float> tz = (Ey[idx_ip] - Ey[idx]) + (Ex[idx] - Ex[idx_jp]);

    Bx[idx] -= complex<float>(dtdx * tx.real(), dtdx * tx.imag());
    By[idx] -= complex<float>(dtdx * ty.real(), dtdx * ty.imag());
    Bz[idx] -= complex<float>(dtdx * tz.real(), dtdx * tz.imag());
}
'''

FIXED_D_REAL = r'''
extern "C" __global__ void fixed_step_D(
    float* Dx, float* Dy, float* Dz,
    const float* Hx, const float* Hy, const float* Hz,
    int nx, int ny, int nz, float dtdx
) {''' + _INDEX_PROLOGUE + r'''
    int im = (i - 1 + nx) % nx;
    int jm = (j - 1 + ny) % ny;
    int km = (k - 1 + nz) % nz;

    int idx_jm = i * ny * nz + jm * nz + k;
    int idx_km = i * ny * nz + j * nz + km;
    int idx_im = im * ny * nz + j * nz + k;

    Dx[idx] -= dtdx * ((Hz[idx_jm] - Hz[idx]) + (Hy[idx] - Hy[idx_km]));
    Dy[idx] -= dtdx * ((Hx[idx_km] - Hx[idx]) + (Hz[idx] - Hz[idx_im]));
    Dz[idx] -= dtdx * ((Hy[idx_im] - Hy[idx]) + (Hx[idx] - Hx[idx_jm]));
}
'''

FIXED_D_COMPLEX = r'''
#include <cupy/complex.cuh>

extern "C" __global__ void fixed_step_D_complex(
    complex<float>* Dx, complex<float>* Dy, complex<float>* Dz,
    const complex<float>* Hx, const complex<float>* Hy, const complex<float>* Hz,
    int nx, int ny, int nz, float dtdx
) {''' + _INDEX_PROLOGUE + r'''
    int im = (i - 1 + nx) % nx;
    int jm = (j - 1 + ny) % ny;
    int km = (k - 1 + nz) % nz;

    int idx_jm = i * ny * nz + jm * nz + k;
    int idx_km = i * ny * nz + j * nz + km;
    int idx_im = im * ny * nz + j * nz + k;

    complex<float> tx = (Hz[idx_jm] - Hz[idx]) + (Hy[idx] - Hy[idx_km]);
    complex<float> ty = (Hx[idx_km] - Hx[idx]) + (Hz[idx] - Hz[idx_im]);
    complex<float> tz = (Hy[idx_im] - Hy[idx]) + (Hx[idx] - Hx[idx_jm]);

    Dx[idx] -= complex<float>(dtdx * tx.real(), dtdx * tx.imag());
    Dy[idx] -= complex<float>(dtdx * ty.real(), dtdx * ty.imag());
    Dz[idx] -= complex<float>(dtdx * tz.real(), dtdx * tz.imag());
}
'''

VARIANTS = {
    ("step_B", False): (FIXED_B_REAL, "fixed_step_B"),
    ("step_B", True): (FIXED_B_COMPLEX, "fixed_step_B_complex"),
    ("step_D", False): (FIXED_D_REAL, "fixed_step_D"),
    ("step_D", True): (FIXED_D_COMPLEX, "fixed_step_D_complex"),
}


def log(message: str) -> None:
    print(message, flush=True)


def cupy_include_dir() -> str:
    return os.path.join(os.path.dirname(cp.__file__), "_core", "include")


# ---------------------------------------------------------------------------
# PTX
# ---------------------------------------------------------------------------

def emit_ptx(code: str, options: Sequence[str]) -> str:
    """Compile with NVRTC targeting a virtual arch so real PTX comes back."""
    from cupy.cuda import nvrtc  # noqa: PLC0415

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    arch = f"compute_{props['major']}{props['minor']}"
    opts = [f"--gpu-architecture={arch}", "-I" + cupy_include_dir(), *options]
    program = nvrtc.createProgram(code, "probe.cu", [], [])
    try:
        nvrtc.compileProgram(program, opts)
        ptx = nvrtc.getPTX(program)
    finally:
        nvrtc.destroyProgram(program)
    return ptx.decode() if isinstance(ptx, bytes) else str(ptx)


def run_ptx(module_codes: Dict[str, str], results: Dict[str, Any],
            out_path: str) -> None:
    counts: Dict[str, Any] = {}
    for label, options in OPTION_SETS:
        for name, code in module_codes.items():
            key = f"{label}:{name}"
            try:
                ptx = emit_ptx(code, options)
                counts[key] = {
                    "fma_rn_f32": ptx.count("fma.rn.f32"),
                    "mul_rn_f32": ptx.count("mul.rn.f32"),
                    "add_rn_f32": ptx.count("add.rn.f32"),
                    "sub_rn_f32": ptx.count("sub.rn.f32"),
                    "ptx_chars": len(ptx),
                }
            except Exception as exc:  # noqa: BLE001
                counts[key] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
            log(f"[ptx] {key}: {counts[key]}")
    results["ptx"] = counts
    save(results, out_path)


# ---------------------------------------------------------------------------
# Bit comparison (same definitions as the first probe)
# ---------------------------------------------------------------------------

def _ordered_key(raw: np.ndarray) -> np.ndarray:
    signed = raw.view(np.int32).astype(np.int64)
    return np.where(signed < 0, np.int64(-2147483648) - signed, signed)


def bit_compare(a_dev: Any, b_dev: Any) -> Dict[str, Any]:
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


def reference_step(sources, targets, terms, dtdx, backward):
    """stepping.py's Cartesian periodic curl + ``target -= curl``."""
    shift = 1 if backward else -1
    out = {}
    for target, first, first_axis, second, second_axis in terms:
        f = sources[first]
        s = sources[second]
        sf = cp.roll(f, shift, axis=first_axis)
        ss = cp.roll(s, shift, axis=second_axis)
        curl = dtdx * ((sf - f) + (s - ss))
        updated = targets[target].copy()
        updated -= curl
        out[target] = updated
    return out


def make_fields(shape, dtype, rng):
    names = ("Bx", "By", "Bz", "Ex", "Ey", "Ez",
             "Dx", "Dy", "Dz", "Hx", "Hy", "Hz")
    arrays = {}
    for name in names:
        real = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        if dtype == np.complex64:
            imag = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            host = (real + 1j * imag).astype(np.complex64)
        else:
            host = real
        arrays[name] = cp.asarray(np.ascontiguousarray(host))
    return arrays


def run_variants(results: Dict[str, Any], out_path: str,
                 shapes: Sequence[Tuple[int, int, int]],
                 dtdx_values: Sequence[float]) -> None:
    cases: List[Dict[str, Any]] = results.setdefault("variant_bit_identity", [])
    dtypes = (("complex64", np.complex64), ("float32", np.float32))
    total = len(OPTION_SETS) * len(shapes) * len(dtypes) * len(dtdx_values) * 2
    index = 0
    for label, options in OPTION_SETS:
        for shape in shapes:
            for dtype_name, dtype in dtypes:
                for dtdx in dtdx_values:
                    for sub_step in ("step_B", "step_D"):
                        index += 1
                        started = time.time()
                        case = one_variant_case(options, label, shape, dtype_name,
                                                dtype, dtdx, sub_step)
                        case["seconds"] = round(time.time() - started, 3)
                        cases.append(case)
                        cmp_ = case["vs_array_order"]
                        log(f"[fix] case {index}/{total} options={label} {sub_step} "
                            f"{dtype_name} shape={shape} dtdx={dtdx!r}: "
                            f"bit_identical={cmp_['bit_identical']} "
                            f"diff={cmp_['differing_floats']}/{cmp_['total_floats']} "
                            f"maxulp={cmp_.get('max_ulp', 0)} "
                            f"maxabs={cmp_.get('max_abs_diff', 0.0):.3e} "
                            f"({case['seconds']} s)")
                        save(results, out_path)


def one_variant_case(options, label, shape, dtype_name, dtype, dtdx, sub_step):
    rng = np.random.default_rng(SEED)
    arrays = make_fields(shape, dtype, rng)
    is_complex = dtype == np.complex64
    if sub_step == "step_B":
        target_names, source_names, terms, backward = (
            ("Bx", "By", "Bz"), ("Ex", "Ey", "Ez"), B_TERMS, False)
    else:
        target_names, source_names, terms, backward = (
            ("Dx", "Dy", "Dz"), ("Hx", "Hy", "Hz"), D_TERMS, True)

    sources = {n: arrays[n] for n in source_names}
    targets = {n: arrays[n] for n in target_names}
    reference = reference_step(sources, targets, terms, dtdx, backward)

    code, func = VARIANTS[(sub_step, is_complex)]
    kernel = (cp.RawKernel(code, func, options=tuple(options)) if options
              else cp.RawKernel(code, func))

    device_targets = {n: arrays[n].copy() for n in target_names}
    nx, ny, nz = shape
    threads = 256
    blocks = (nx * ny * nz + threads - 1) // threads
    kernel((blocks,), (threads,), (
        device_targets[target_names[0]], device_targets[target_names[1]],
        device_targets[target_names[2]],
        sources[source_names[0]], sources[source_names[1]], sources[source_names[2]],
        np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx)))
    cp.cuda.runtime.deviceSynchronize()

    return {
        "option_label": label,
        "options": list(options),
        "sub_step": sub_step,
        "dtype": dtype_name,
        "shape": list(shape),
        "dtdx": repr(dtdx),
        "vs_array_order": combine({
            n: bit_compare(device_targets[n], reference[n]) for n in target_names}),
    }


def save(results: Dict[str, Any], out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", required=True,
                        help="path to cuda_kernels/step_curl_kernels.py (for PTX)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shapes", default="13,11,9;32,32,32")
    args = parser.parse_args(argv)

    shapes = tuple(tuple(int(v) for v in g.split(","))
                   for g in args.shapes.split(";") if g.strip())
    dtdx_values = (0.5, 0.35)

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.kernels)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("probe_kernels", args.kernels)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["probe_kernels"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    results: Dict[str, Any] = {
        "probe": "fused_kernel_grouping",
        "seed": SEED,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "device_name": name.decode() if isinstance(name, bytes) else str(name),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "cupy_version": cp.__version__,
            "numpy_version": np.__version__,
            "nvrtc_version": ".".join(str(v) for v in cp.cuda.nvrtc.getVersion()),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        },
    }
    log(f"[env] {json.dumps(results['environment'])}")
    save(results, args.out)

    shipped = {
        "shipped_step_B": module._triple_curl_B_kernel_code,
        "shipped_step_B_complex": module._triple_curl_B_kernel_complex_code,
        "shipped_step_D": module._triple_curl_D_kernel_code,
        "shipped_step_D_complex": module._triple_curl_D_kernel_complex_code,
        "variant_fixed_step_B": FIXED_B_REAL,
        "variant_fixed_step_B_complex": FIXED_B_COMPLEX,
        "variant_fixed_step_D": FIXED_D_REAL,
        "variant_fixed_step_D_complex": FIXED_D_COMPLEX,
    }
    run_ptx(shipped, results, args.out)
    run_variants(results, args.out, shapes, dtdx_values)
    save(results, args.out)
    log(f"[done] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
