"""Is ``--fmad=false`` load-bearing for the complex family? Measured, per arm.

WHY THIS EXISTS. Every certified hand-CUDA slice carries ``--fmad=false`` as
CORRECTNESS rather than tuning, and every one of their gates shows it by running a
GUARD CONTROL: the same cases compiled with no options, which must DIVERGE. On the
complex family that control came back 36/36 IDENTICAL on the device
(``results/cuda_complex_2026-08-19/smoke/``), and an identical control is either a
gate that cannot see or a source the flag has nothing to do to.

THE DISCRIMINATING EXPERIMENT, and it is a byte question rather than an output
question: compile each of the four kernels under each arm, with the flag and
without it, and compare the IMAGE NVRTC RETURNS. Two images that are bit-identical
cannot produce different words, so the flag is provably inert for that source; two
that differ and still produce the same words would be a much weaker claim, and one
this probe would show instead of hiding.

WHAT THE IMAGE IS, sniffed rather than assumed: CuPy asks NVRTC for a CUBIN when
the device arch is compilable and for PTX otherwise
(``_get_arch_for_options_for_nvrtc``), so this records which it got. A CUBIN is the
stronger comparison -- it is what actually runs -- and the instruction counts below
are only meaningful on the PTX form, which is why they are reported as zero rather
than as evidence when the image is binary.

THE HYPOTHESIS IT TESTS, stated so a refutation is possible. Under ``FMA_V1``
every product in the emitted source is either an explicit ``__fmaf_rn`` -- already
one ``fma.rn.f32``, which no contraction flag reaches -- or a multiply by a
literal (``0.0f``, ``-1.0f``) whose result feeds another multiply or an fma
ADDEND. Contraction needs a ``mul`` feeding an ``add``, and the transcription
leaves none. Under ``NAIVE`` it does: ``(z.re * c) - (z.im * 0.0f)`` is exactly
that shape, so the flag should be load-bearing there and the two arms should come
out differently. If FMA_V1 also differs, the hypothesis is wrong and the flag is
doing something this probe will have named.

Usage (from ``parity/meep_gpu``, on a host with CuPy and a device)::

    PYTHONPATH=../.. python -u probe_cuda_complex_fmad.py --out <dir>/fmad.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cupy as cp  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu.cuda_kernels import complex_emitter  # noqa: E402

#: PTX instruction families the contraction question turns on. ``mul.f32`` feeding
#: ``add.f32``/``sub.f32`` is the only shape ``--fmad`` acts on; ``fma.rn.f32``
#: from ``__fmaf_rn`` is already one instruction and no flag reaches it.
_PTX_COUNTS = ("fma.rn.f32", "mul.f32", "add.f32", "sub.f32", "mul.rn.f32",
               "add.rn.f32", "sub.rn.f32", "neg.f32", "div.rn.f32")

GUARDS = (("fmad_false", ("--fmad=false",)), ("default_no_options", ()))


def image_kind(raw: bytes) -> str:
    """CUBIN or PTX, sniffed from the bytes rather than assumed from the flags."""
    if raw[:4] == b"\x7fELF":
        return "cubin"
    head = raw[:64].decode("ascii", "replace")
    return "ptx" if "//" in head or ".version" in head else "unknown"


def image_for(source: str, options) -> bytes:
    """The image NVRTC returns for this source under these options.

    THE CALL GOES THROUGH ``compile_using_nvrtc`` AND NOT THROUGH ``RawKernel``,
    deliberately: that is the function the float32 subnormal policy's strip seam
    wraps (subnormal_policy.py:866), so the options this probe hands in are the
    options the policy has already filtered -- the same tuple the gate's kernels
    were compiled with, rather than a parallel path that would answer about a
    different compile.
    """
    from cupy.cuda import compiler  # noqa: PLC0415

    device = cp.cuda.Device()
    major, minor = device.compute_capability[0], device.compute_capability[1:]
    out = compiler.compile_using_nvrtc(
        source, options=tuple(options), arch=f"{major}{minor}",
        filename="probe.cu")
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, str):
        out = out.encode("utf-8")
    return bytes(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("keep", "flush", "match_meep", "ieee"))
    args = parser.parse_args(argv)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "probe": "cuda_complex_fmad",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "environment": probe.device_info(),
        "emitter_corpus_digest": complex_emitter.corpus_digest(),
        "rows": [],
    }
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    for sub_step in sorted(complex_emitter.KERNELS):
        name = complex_emitter.KERNELS[sub_step][0]
        for arm in ("FMA_V1", "NAIVE"):
            source = complex_emitter.complex_source(sub_step, arm)
            row: Dict[str, Any] = {"sub_step": sub_step, "kernel": name,
                                   "arm": arm,
                                   "source_sha256": hashlib.sha256(
                                       source.encode("utf-8")).hexdigest()}
            for label, options in GUARDS:
                raw = image_for(source, options)
                kind = image_kind(raw)
                text = raw.decode("utf-8", "replace") if kind == "ptx" else ""
                counts = {token: len(re.findall(re.escape(token), text))
                          for token in _PTX_COUNTS}
                row[label] = {
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                    "image_bytes": len(raw),
                    "image_kind": kind,
                    "instruction_counts": counts}
                print(f"[{sub_step}/{arm}/{label}] {kind} "
                      f"{len(raw)}B sha={hashlib.sha256(raw).hexdigest()[:12]} "
                      + " ".join(f"{k}={v}" for k, v in counts.items() if v),
                      flush=True)
            row["image_identical_across_guards"] = (
                row["fmad_false"]["image_sha256"]
                == row["default_no_options"]["image_sha256"])
            row["fma_delta"] = (
                row["default_no_options"]["instruction_counts"]["fma.rn.f32"]
                - row["fmad_false"]["instruction_counts"]["fma.rn.f32"])
            print(f"[{sub_step}/{arm}] image_identical="
                  f"{row['image_identical_across_guards']} "
                  f"extra_fma_without_the_flag={row['fma_delta']}", flush=True)
            results["rows"].append(row)
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, sort_keys=True, default=str)

    by_arm = {}
    for arm in ("FMA_V1", "NAIVE"):
        rows = [r for r in results["rows"] if r["arm"] == arm]
        by_arm[arm] = {
            "kernels": len(rows),
            "image_identical_across_guards": sum(
                1 for r in rows if r["image_identical_across_guards"]),
            "kernels_the_flag_changes": sum(
                1 for r in rows if not r["image_identical_across_guards"]),
            "max_extra_fma_without_the_flag": max(
                (r["fma_delta"] for r in rows), default=0),
        }
    results["by_arm"] = by_arm
    results["verdict"] = {
        # THE CLAIM, as a measured pair. The flag is inert for the arm this
        # platform licensed and load-bearing for the other one; a source with no
        # mul-feeding-an-add has nothing for a contraction flag to do.
        "flag_is_inert_for_FMA_V1":
            by_arm["FMA_V1"]["kernels_the_flag_changes"] == 0,
        "flag_is_load_bearing_for_NAIVE":
            by_arm["NAIVE"]["kernels_the_flag_changes"] > 0,
    }
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    print(f"[done] {json.dumps(results['verdict'])}", flush=True)
    print(f"[done] by_arm {json.dumps(by_arm)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
