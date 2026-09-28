"""Does ``install_subnormal_policy("keep")`` reach the HAND-CUDA kernels?

The nine Triton families are certified under ``keep``. The hand-CUDA real-field
PML curl was certified 120/120 in 2026-08-09, before this module existed, and its
record names no policy. Two questions follow, and neither is answerable by
reading: CuPy appends ``-ftz=true`` below our own option tuple, the strip
installs BELOW CuPy's cache key, and ``step_curl_kernels`` holds a second
compiled-kernel memo of its own that no policy install knows about.

THE DETECTOR IS THE SHIPPED KERNEL ITSELF, not a micro-kernel. Set every source
field to zero so the curl is exactly 0, set ``kms = sinv = 1``, and plant a float32
SUBNORMAL in the auxiliary. ``stepping._apply_pml_update`` then computes
``fu_new = ((fu * 1) - 0) * 1``, which is the identity — under ``keep``. Under
``-ftz=true`` the two ``mul.rn.ftz.f32`` flush it and the auxiliary comes back
exactly zero. One bit pattern, one kernel, no reference implementation needed.

Run it TWICE, in two processes, with two different cache directories::

    CUDA_VISIBLE_DEVICES=<free gpu> CUPY_CACHE_DIR=<dir>/late_ftz_stripped \
        python -u probe_cuda_policy_reach.py --order policy_last  --out <...>.json
    CUDA_VISIBLE_DEVICES=<free gpu> CUPY_CACHE_DIR=<dir>/early_ftz_stripped \
        python -u probe_cuda_policy_reach.py --order policy_first --out <...>.json

``policy_last`` is the realistic failure order (something compiled a kernel before
the policy was chosen); ``policy_first`` is the order the policy documents.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SHAPE = (4, 4, 4)
#: 2^-135, a float32 subnormal. Built from bits: a literal would be parsed by a
#: host strtod that may itself be flushing (backends._smallest_subnormal's reason).
NEEDLE = np.frombuffer(np.uint32(0x00004000).tobytes(), dtype=np.float32)[0]


def log(message: str) -> None:
    print(message, flush=True)


def memo_size(sck) -> int:
    """How many compiled kernels the module is holding.

    Read through the module's own cache module, NOT through a
    ``sck._compiled_kernels`` attribute: that dict moved to
    ``cuda_kernels/compile_cache.py`` on 2026-08-15 when its key grew the compile
    options, the subnormal policy and the source (disposition §1.2), and
    ``test_compile_cache.py`` pins that the old attribute stays gone. This probe
    is the source of certification.json's §1.1 evidence — "install-then-compile
    kept a planted 0x00004000 in fu_Bx; compile-then-install flushed it" — so it
    has to keep running on a CUDA host after that move, and an ``AttributeError``
    on the first leg would mean that measurement could never be re-taken.

    Falls back to the pre-move attribute so the probe can also be pointed at an
    older checkout, and reports ``-1`` rather than raising if neither is there:
    the memo size is context beside the bit pattern, and the bit pattern is the
    measurement.
    """
    cache = getattr(sck, "compile_cache", None)
    if cache is not None and hasattr(cache, "cache_size"):
        return int(cache.cache_size())
    legacy = getattr(sck, "_compiled_kernels", None)
    return len(legacy) if legacy is not None else -1


def run_kernel_once(label: str) -> Dict[str, Any]:
    """One launch of the SHIPPED ``step_B_pml_real``, reading fu_Bx back.

    Returns the auxiliary's bit pattern. ``0x00004000`` means the binary kept the
    subnormal; ``0x00000000`` means it flushed.
    """
    import cupy as cp  # noqa: PLC0415

    from meep_gpu.cuda_kernels import step_curl_kernels as sck  # noqa: PLC0415

    zeros = cp.zeros(SHAPE, dtype=cp.float32)
    aux = [cp.full(SHAPE, NEEDLE, dtype=cp.float32) for _ in range(3)]
    targets = [cp.zeros(SHAPE, dtype=cp.float32) for _ in range(3)]
    sources = [zeros, zeros.copy(), zeros.copy()]
    ones = cp.ones(SHAPE[0], dtype=cp.float32)
    tables = {f"{stem}_{axis}": ones for axis in "xyz" for stem in ("kms", "sinv")}
    codes = (np.int32(0), np.int32(0), np.int32(0))

    sck._step_fused_pml_real(
        sck._get_kernel("step_B_pml_real", False),
        targets, aux, sources, tables, codes, 0.35)
    cp.cuda.runtime.deviceSynchronize()
    word = int(cp.asnumpy(aux[0]).view(np.uint32).ravel()[0])
    kernel = sck._get_kernel("step_B_pml_real", False)
    return {
        "leg": label,
        "fu_word": "0x%08x" % word,
        "kept_subnormal": word == 0x00004000,
        "kernel_object_id": id(kernel),
        "memo_size": memo_size(sck),
    }


def array_path_once(label: str) -> Dict[str, Any]:
    """The same identity through CuPy elementwise kernels — the correctness oracle.

    If the two disagree the process is internally split, which is the state the
    policy module exists to prevent.
    """
    import cupy as cp  # noqa: PLC0415

    aux = cp.full(SHAPE, NEEDLE, dtype=cp.float32)
    one = cp.ones(SHAPE, dtype=cp.float32)
    out = ((aux * one) - cp.zeros(SHAPE, dtype=cp.float32)) * one
    cp.cuda.runtime.deviceSynchronize()
    word = int(cp.asnumpy(out).view(np.uint32).ravel()[0])
    return {"leg": label, "fu_word": "0x%08x" % word,
            "kept_subnormal": word == 0x00004000}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--order", choices=("policy_last", "policy_first"),
                        required=True)
    args = parser.parse_args()

    import cupy as cp  # noqa: PLC0415

    from meep_gpu import subnormal_policy  # noqa: PLC0415
    from meep_gpu.cuda_kernels import step_curl_kernels as sck  # noqa: PLC0415

    results: Dict[str, Any] = {
        "probe": "cuda_policy_reach",
        "order": args.order,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "needle": "0x00004000 (2^-135, float32 subnormal)",
        "environment": {
            "cupy": cp.__version__,
            "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR", "<unset>"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        },
        "legs": [],
    }

    def record(entry: Dict[str, Any]) -> None:
        results["legs"].append(entry)
        log("[leg] " + json.dumps(entry))
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)

    if args.order == "policy_last":
        record(run_kernel_once("before_install__cuda_kernel"))
        record(array_path_once("before_install__array_path"))

    started = time.time()
    try:
        report = subnormal_policy.install_subnormal_policy(
            subnormal_policy.KEEP, strict=False)
        results["install_report"] = {
            k: v for k, v in report.items()
            if k in ("policy", "attained", "executors", "reasons")}
    except Exception as exc:  # noqa: BLE001
        results["install_report"] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    log(f"[install] keep requested, {time.time() - started:.1f} s, "
        f"{json.dumps(results['install_report'])[:600]}")

    record(run_kernel_once("after_install__cuda_kernel__module_memo_warm"))
    sck._clear_kernel_cache()
    record(run_kernel_once("after_install__cuda_kernel__module_memo_cleared"))
    record(array_path_once("after_install__array_path"))

    verdicts: List[str] = []
    by_leg = {e["leg"]: e for e in results["legs"]}
    warm = by_leg.get("after_install__cuda_kernel__module_memo_warm")
    cleared = by_leg.get("after_install__cuda_kernel__module_memo_cleared")
    array = by_leg.get("after_install__array_path")
    if warm and cleared and array:
        verdicts.append(
            "cuda kernel obeys keep with the module memo warm: %s" % warm["kept_subnormal"])
        verdicts.append(
            "cuda kernel obeys keep after _clear_kernel_cache(): %s" % cleared["kept_subnormal"])
        verdicts.append("array path obeys keep: %s" % array["kept_subnormal"])
        verdicts.append(
            "process internally consistent after install: %s"
            % (cleared["kept_subnormal"] == array["kept_subnormal"]))
    results["verdicts"] = verdicts
    for line in verdicts:
        log("[verdict] " + line)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    log("[done] " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
