"""Does ``PYTORCH_MPS_FAST_MATH`` change what ``torch.mps.compile_shader`` produces?

Every Metal kernel in this package is compiled through ``torch.mps.compile_shader``.
This probe compiles one float32 divide and one square root, runs them on the same
2**20 inputs in three processes -- the variable unset, ``1`` and ``0`` -- and counts
the output words that differ from the unset run. A nonzero count means the variable
reaches the kernels this package certifies, so ``metal_dispatch`` treats it as part
of the Metal environment.

Measured 2026-10-02 on an Apple M1 Max (macOS 26.2, torch 2.10.0): ``1`` changed
274,523 divide words and 327,338 square-root words of 1,048,576; ``0`` changed none.

    python parity/meep_gpu/probe_metal_fast_math.py [--out results/probe_metal_fast_math.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

VARIABLE = "PYTORCH_MPS_FAST_MATH"
COUNT = 1 << 20

CHILD = r'''
import sys
import numpy as np  # NumPy before torch: one OpenMP runtime (INSTALL.md)
import torch
source = """
#include <metal_stdlib>
using namespace metal;
kernel void divide(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                   device float* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = a[i] / b[i];
}
kernel void root(device const float* a [[buffer(0)]], device float* out [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    out[i] = sqrt(a[i]);
}
"""
library = torch.mps.compile_shader(source)
rng = np.random.default_rng(1234)
count = int(sys.argv[2])
scale = np.float32(10) ** rng.integers(-30, 30, (2, count)).astype(np.float32)
a, b = (rng.standard_normal((2, count)).astype(np.float32) * scale)
left, right = torch.from_numpy(a).to("mps"), torch.from_numpy(b).to("mps")
quotient, root = torch.empty_like(left), torch.empty_like(left)
library.divide(left, right, quotient)
library.root(left.abs(), root)
torch.mps.synchronize()
np.save(sys.argv[1], np.stack([quotient.cpu().numpy(), root.cpu().numpy()]))
print(torch.__version__)
'''


def run(value, directory: Path):
    environment = dict(os.environ)
    environment.pop(VARIABLE, None)
    if value is not None:
        environment[VARIABLE] = value
    path = directory / f"{value}.npy"
    done = subprocess.run([sys.executable, "-c", CHILD, str(path), str(COUNT)],
                          env=environment, capture_output=True, text=True, check=True)
    return path, done.stdout.strip().splitlines()[-1]


def main(argv=None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        reference, torch_version = run(None, directory)
        baseline = np.load(reference).view(np.uint32)
        result = {"torch": torch_version, "words": COUNT, "differing_from_unset": {}}
        for value in ("1", "0"):
            path, _ = run(value, directory)
            differing = (np.load(path).view(np.uint32) != baseline).sum(axis=1)
            result["differing_from_unset"][value] = {
                "divide": int(differing[0]), "sqrt": int(differing[1])}
            print(f"{VARIABLE}={value}: divide {int(differing[0])}, sqrt "
                  f"{int(differing[1])} of {COUNT} words differ from unset", flush=True)
    result["reaches_compile_shader"] = any(
        sum(row.values()) for row in result["differing_from_unset"].values())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
