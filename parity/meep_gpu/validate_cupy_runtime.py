#!/usr/bin/env python3
"""Validate that an installed CuPy build can execute real CUDA work.

This is an explicit, package-dependent host check rather than a pytest test:
it fails when CuPy or a CUDA device is absent.  Run it inside the scheduler's
one-GPU allocation so the reported device count also verifies resource
isolation.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON result. Parent directories must already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        import cupy as cp
    except ImportError as exc:
        raise SystemExit(
            "CuPy is not installed. Install the CUDA-version-matched CuPy wheel "
            "before running this host validation."
        ) from exc

    visible_devices = int(cp.cuda.runtime.getDeviceCount())
    if visible_devices != 1:
        raise RuntimeError(
            "Expected exactly one scheduler-isolated CUDA device, "
            f"but CuPy sees {visible_devices}. Run inside a one-GPU allocation."
        )

    device = cp.cuda.Device(0)
    device.use()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8")

    element_count = 1 << 20
    x = cp.arange(element_count, dtype=cp.float32) / cp.float32(element_count)
    y = cp.full(element_count, cp.float32(0.25), dtype=cp.float32)
    kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void saxpy(const float *x, float *y, const float alpha, const int n) {
            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i < n) {
                y[i] = alpha * x[i] + y[i];
            }
        }
        """,
        "saxpy",
    )
    threads = 256
    blocks = (element_count + threads - 1) // threads
    kernel((blocks,), (threads,), (x, y, cp.float32(2.0), element_count))
    device.synchronize()
    raw_kernel_max_error = float(cp.max(cp.abs(y - (cp.float32(2.0) * x + 0.25))).get())
    if raw_kernel_max_error > 2e-7:
        raise RuntimeError(f"RawKernel SAXPY validation failed: max error {raw_kernel_max_error:.3e}")

    matrix_size = 256
    matrix = cp.arange(matrix_size * matrix_size, dtype=cp.float32).reshape(
        matrix_size, matrix_size
    )
    product = matrix @ cp.eye(matrix_size, dtype=cp.float32)
    device.synchronize()
    matmul_max_error = float(cp.max(cp.abs(product - matrix)).get())
    if matmul_max_error != 0.0:
        raise RuntimeError(f"CUDA matrix multiplication validation failed: max error {matmul_max_error}")

    host_tail = cp.asnumpy(y[-4:]).tolist()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    result = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cupy_version": cp.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": visible_devices,
        "device_name": device_name,
        "compute_capability": device.compute_capability,
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "raw_kernel_max_error": raw_kernel_max_error,
        "matmul_max_error": matmul_max_error,
        "host_transfer_tail": host_tail,
        "free_device_bytes_after_validation": int(free_bytes),
        "total_device_bytes": int(total_bytes),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"MEEP_GPU_CUPY_JSON={json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
