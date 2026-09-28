"""Do the two subnormal policies compile DISTINCT binaries for the OFF-DIAGONAL family?

The gate is cut twice, once under ``keep`` and once under ``flush``. Two runs that
agree prove nothing unless the two runs were actually DIFFERENT runs. This is the
same pair of measurements the curl and the constitutive re-cuts settled, applied
to ``update_E_pml_real_offdiag`` — and applied PER ROW MASK, because this
family's device source is emitted, so "the kernel" is a set of sources sharing one
name and a policy that changed one of them says nothing about the others.

1. THE CUBIN DIGEST, at the seam the policy acts on.
   ``install_nvrtc_binary_observer`` wraps ``cupy.cuda.compiler.compile_using_nvrtc``
   BELOW the strip, so it records the option tuple NVRTC really received and the
   sha256 of the bytes it really returned. Each leg runs in its own process with a
   private, COLD, policy-token-carrying ``CUPY_CACHE_DIR``, so CuPy's disk cache —
   keyed ABOVE that seam — cannot serve one policy's binary under the other's name.

   ATTRIBUTION IS BRACKETED, NOT POSITIONAL. The observer sees every NVRTC call in
   the process and CuPy makes several of its own; a verdict taken over all of them
   measures the wrong thing and measured it wrongly on the constitutive round (two
   of CuPy's arithmetic-free compiles were legitimately identical across the
   policies and scored the whole run ``distinct=False``). The observation index is
   bracketed around each ``_get_kernel`` call, and THE ATTRIBUTE READ IS INSIDE THE
   BRACKET because ``cp.RawKernel`` construction compiles nothing —
   ``RawKernel.kernel`` is lazy and the NVRTC call happens on first access.

2. THE NEEDLE, which is the stronger claim: that the policy reaches THIS FAMILY'S
   OWN ARITHMETIC — the coupling — rather than merely changing its bytes.

   Every D volume is filled with one float32 subnormal ``n = 2**-135``, every
   inverse permittivity and every live row coefficient with exactly 1.0, ``kps``
   with 1 and ``kms`` with 0, on an all-periodic grid so no ghost enters. The
   kernel then computes, for the coupled component::

       near_pair = n + n = 2n ;  far_pair = 2n
       term      = 0.25 * ((2n * 1) + (2n * 1)) = n
       src       = (n * 1) + n = 2n
       f         = 0 + 1*2n - 0*0 = 2n

   Every intermediate is a power of two inside the subnormal band, so under
   ``keep`` the answer is EXACTLY ``2n`` — checked as a bit pattern, not as
   "nonzero" — and under ``-ftz=true`` every one of those multiplies flushes and
   the answer is exactly zero.

   AN UNCOUPLED COMPONENT IS THE DISCRIMINATOR. ``Ey`` and ``Ez`` take the plain
   arm on the single-slot mask, so they compute ``n * 1`` and land on ``n``, not
   ``2n``. A leg where the coupled and the uncoupled components agree is reporting
   a fixture in which the coupling never fired, and the coupling is the only thing
   this family adds.

   THE FIXTURE'S OWN NON-VACUITY is read back off the device after the launch: the
   D volumes are read-only to this sub-step, so whatever they hold afterwards is
   what the kernel was given. Built with ``cupy.full(..., needle)`` the needle does
   NOT survive the trip under a flush policy — measured on the constitutive round,
   where the source array read back as exactly zero BEFORE the kernel ran and "the
   kernel flushed it" was a statement about the fixture. The pattern is delivered
   as ``uint32`` and reinterpreted on the device, so no float touches it in flight.

Run (the GPU host, one GPU) — the parent spawns both legs::

    CUDA_VISIBLE_DEVICES=0 python -u probe_cuda_offdiag_policy_binaries.py \\
        --out results/<dir>/offdiag_policy_binaries.json \\
        --cache-root results/<dir>/policy_binary_caches
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_API = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_API not in sys.path:
    sys.path.insert(0, REPO_API)

SHAPE = (4, 4, 4)

#: 2^-135, a float32 subnormal. Built from BITS: a literal would be parsed by a
#: host strtod that may itself be flushing.
NEEDLE_BITS = 0x00004000
NEEDLE = np.frombuffer(np.uint32(NEEDLE_BITS).tobytes(), dtype=np.float32)[0]

KERNEL = "update_E_pml_real_offdiag"

#: The masks compiled and needled. The two the 186-row corpus drives, plus the
#: single-slot mask that leaves two components on the PLAIN arm — which is what
#: makes the coupled/uncoupled discriminator below possible at all.
MASKS = ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0))
NEEDLE_MASK = (1, 0, 0, 0, 0, 0)


def log(message: str) -> None:
    print(message, flush=True)


def load_probe():
    path = os.path.join(HERE, "probe_fused_kernel_bit_identity.py")
    spec = importlib.util.spec_from_file_location("probe_bit_identity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Arrays:
    """A stand-in for ``Fields`` holding only what the launcher reads."""

    def __init__(self, mapping: Dict[str, Any], inverse_epsilon, rows, grid):
        self.__dict__.update(mapping)
        self._inverse_epsilon = inverse_epsilon
        self._rows = rows
        self.grid = grid

    def inverse_epsilon_for(self, component: str):
        return self._inverse_epsilon[component]

    def chi1inv_offdiagonal_for(self, row: str):
        return self._rows.get(row, {})


def needle_array(cupy, bits: int):
    """A device float32 array holding an EXACT bit pattern, delivered as uint32."""
    return cupy.asarray(np.full(SHAPE, bits, dtype=np.uint32)).view(np.float32)


def needle_leg(kernels, cupy) -> Dict[str, Any]:
    """One launch of the SHIPPED kernel with a planted subnormal; returns the bits."""
    nx, ny, nz = SHAPE
    zeros = lambda: cupy.zeros(SHAPE, dtype=cupy.float32)  # noqa: E731
    ones = lambda: cupy.ones(SHAPE, dtype=cupy.float32)    # noqa: E731
    needles = needle_array(cupy, NEEDLE_BITS)

    targets = ("Ex", "Ey", "Ez")
    auxiliaries = ("f_w_Ex", "f_w_Ey", "f_w_Ez")
    sources = ("Dx", "Dy", "Dz")
    mapping = {name: zeros() for name in targets + auxiliaries}
    mapping.update({name: needles.copy() for name in sources})
    inverse = {name: ones() for name in targets}
    rows: Dict[str, Dict[str, Any]] = {}
    from meep_gpu.cuda_kernels import coverage  # noqa: PLC0415
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if NEEDLE_MASK[slot]:
            rows.setdefault(row, {})[partner] = ones()
    fields = _Arrays(mapping, inverse, rows, grid=None)

    tables = {}
    for axis, extent in zip("xyz", (nx, ny, nz)):
        tables[f"kps_{axis}"] = cupy.ones(extent, dtype=cupy.float32)
        tables[f"kms_{axis}"] = cupy.zeros(extent, dtype=cupy.float32)

    # ALL PERIODIC and no wall mask: the needle's fate is decided by the
    # arithmetic, never by a ghost. codes/walls come through the gate door, so no
    # Grid object is needed and none is built.
    kernels.update_E_offdiag_fused_pml_real(
        fields, tables=tables, codes=(0, 0, 0), walls=(0, 0, 0))
    cupy.cuda.runtime.deviceSynchronize()

    def first_bits(name: str) -> int:
        return int(cupy.asnumpy(getattr(fields, name)).ravel()[0].view(np.uint32))

    coupled_bits = first_bits("Ex")
    uncoupled_bits = first_bits("Ey")
    source_bits = first_bits("Dx")
    aux_bits = first_bits("f_w_Ex")
    return {
        "row_mask": list(NEEDLE_MASK),
        "needle_bits": NEEDLE_BITS,
        "expected_coupled_bits_under_keep": 2 * NEEDLE_BITS,
        "expected_uncoupled_bits_under_keep": NEEDLE_BITS,
        "source_bits_on_device": source_bits,
        "fixture_delivered_the_needle": source_bits == NEEDLE_BITS,
        "coupled_bits_after": coupled_bits,
        "uncoupled_bits_after": uncoupled_bits,
        "auxiliary_bits_after": aux_bits,
        # KEEP: the coupled component lands on exactly 2n and the uncoupled one on
        # exactly n, so the coupling is proven to have fired and to have fired with
        # the transcribed weight.
        "coupled_kept_the_doubled_needle": coupled_bits == 2 * NEEDLE_BITS,
        "uncoupled_kept_the_needle": uncoupled_bits == NEEDLE_BITS,
        "coupling_fired": coupled_bits != uncoupled_bits,
        # FLUSH: every multiply on this path flushes, so both go to exact zero.
        "coupled_flushed_to_zero": coupled_bits == 0,
        "uncoupled_flushed_to_zero": uncoupled_bits == 0,
        "auxiliary_followed_the_arithmetic": aux_bits == coupled_bits,
    }


def run_leg(policy: str, out_path: str) -> int:
    """One process: install the policy, compile each mask, plant the needle."""
    probe = load_probe()
    binary_observer = probe.install_nvrtc_binary_observer()
    meep_import = (probe.import_meep_for_host_policy() if policy == "flush"
                   else {"requested": False})
    policy_install = probe.install_subnormal_policy_for_run(policy, REPO_API)

    import cupy
    from meep_gpu.cuda_kernels import offdiag_constitutive_kernels as kernels

    kernels._clear_kernel_cache()
    attributes: Dict[str, Any] = {}
    compile_index: Dict[str, Any] = {}
    for mask in MASKS:
        label = f"{KERNEL}|R{''.join(str(f) for f in mask)}"
        before = len(probe.nvrtc_binary_report().get("observations", []))
        kernel = kernels._get_kernel(mask)
        entry: Dict[str, Any] = {}
        for attribute in ("num_regs", "local_size_bytes", "shared_size_bytes"):
            try:
                entry[attribute] = int(getattr(kernel, attribute))
            except Exception as exc:  # noqa: BLE001 - report, never abort the leg
                entry[attribute] = f"unreadable: {type(exc).__name__}"
        after = len(probe.nvrtc_binary_report().get("observations", []))
        compile_index[label] = {"observation_first": before,
                                "observation_last": after - 1,
                                "compiles": after - before}
        attributes[label] = entry

    needle = needle_leg(kernels, cupy)

    result = {
        "policy_requested": policy,
        "policy_stamp": probe.subnormal_policy_stamp(REPO_API),
        "subnormal_policy_install": policy_install,
        "meep_import_for_host_policy": meep_import,
        "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR", "<unset>"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "nvrtc_binary_observer": binary_observer,
        "kernel_attributes": attributes,
        "kernel_compile_index": compile_index,
        "needle": needle,
        "nvrtc_binaries": probe.nvrtc_binary_report(),
    }
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    log(f"[leg] policy={policy} "
        f"nvrtc_calls={result['nvrtc_binaries']['nvrtc_calls_observed']} "
        f"distinct={result['nvrtc_binaries']['distinct_binaries']} "
        f"coupled={needle['coupled_bits_after']:#010x} "
        f"uncoupled={needle['uncoupled_bits_after']:#010x} "
        f"delivered={needle['fixture_delivered_the_needle']}")
    return 0


def _digest_by_variant(leg: Dict[str, Any]) -> Dict[str, Any]:
    observations = leg.get("nvrtc_binaries", {}).get("observations", [])
    index = leg.get("kernel_compile_index", {})
    per_variant: Dict[str, Any] = {}
    for name, span in index.items():
        first, last = span.get("observation_first"), span.get("observation_last")
        if first is None or last is None or last < first or last >= len(observations):
            per_variant[name] = None
            continue
        per_variant[name] = {
            "binary_sha256": observations[last].get("binary_sha256"),
            "source_sha256": observations[last].get("source_sha256"),
            "binary_bytes": observations[last].get("binary_bytes"),
            "ftz_true_present": observations[last].get("ftz_true_present"),
            "compiles": span.get("compiles"),
        }
    return {"per_variant": per_variant,
            "all_binaries_in_call_order": [o.get("binary_sha256", "")
                                           for o in observations]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-root", required=True,
                        help="parent of the two COLD, per-policy CUPY_CACHE_DIRs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--leg", default=None, choices=("keep", "flush"))
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.leg:
        return run_leg(args.leg, args.out)

    legs: List[Dict[str, Any]] = []
    for policy in ("keep", "flush"):
        # The token is not decoration: subnormal_policy.cupy_cache_reasons REFUSES
        # a keep install into a directory that does not carry 'ftz_stripped'.
        token = "ftz_stripped" if policy == "keep" else "policy_flush"
        cache_dir = os.path.join(os.path.abspath(args.cache_root), token)
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)  # COLD: a hit above the seam reaches no observer
        os.makedirs(cache_dir, exist_ok=True)
        leg_out = args.out + f".{policy}.json"
        environment = dict(os.environ)
        environment["CUPY_CACHE_DIR"] = cache_dir
        if policy == "flush":
            environment["CUPY_ACCELERATORS"] = ""
        started = time.time()
        completed = subprocess.run(
            [args.python, "-u", os.path.abspath(__file__), "--leg", policy,
             "--out", leg_out, "--cache-root", args.cache_root],
            env=environment, capture_output=True, text=True)
        elapsed = round(time.time() - started, 2)
        record: Dict[str, Any] = {"leg": f"cold_{policy}", "policy": policy,
                                  "cache_dir": cache_dir, "cold_cache": True,
                                  "returncode": completed.returncode,
                                  "seconds": elapsed,
                                  "stdout": completed.stdout[-4000:],
                                  "stderr": completed.stderr[-4000:]}
        if os.path.exists(leg_out):
            with open(leg_out) as handle:
                record["result"] = json.load(handle)
        legs.append(record)
        log(f"[{policy}] rc={completed.returncode} ({elapsed} s)")
        for line in completed.stdout.strip().splitlines()[-4:]:
            log(f"[{policy}]   {line}")

    results: Dict[str, Any] = {
        "probe": "cuda_offdiag_policy_binaries",
        "kernel": KERNEL,
        "row_masks_compiled": [list(m) for m in MASKS],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "legs": legs,
    }

    keep = next((l for l in legs if l["policy"] == "keep"), {})
    flush = next((l for l in legs if l["policy"] == "flush"), {})
    keep_result, flush_result = keep.get("result", {}), flush.get("result", {})
    keep_digests = _digest_by_variant(keep_result) if keep_result else {}
    flush_digests = _digest_by_variant(flush_result) if flush_result else {}
    keep_variants = keep_digests.get("per_variant", {})
    flush_variants = flush_digests.get("per_variant", {})

    per_variant_distinct: Dict[str, Any] = {}
    for mask in MASKS:
        name = f"{KERNEL}|R{''.join(str(f) for f in mask)}"
        keep_entry, flush_entry = keep_variants.get(name), flush_variants.get(name)
        per_variant_distinct[name] = {
            "keep_binary": (keep_entry or {}).get("binary_sha256"),
            "flush_binary": (flush_entry or {}).get("binary_sha256"),
            "keep_ftz_true_present": (keep_entry or {}).get("ftz_true_present"),
            "flush_ftz_true_present": (flush_entry or {}).get("ftz_true_present"),
            "distinct": bool(
                keep_entry and flush_entry
                and keep_entry.get("binary_sha256")
                and flush_entry.get("binary_sha256")
                and keep_entry["binary_sha256"] != flush_entry["binary_sha256"]),
        }
    # THE SOURCES MUST DIFFER FROM EACH OTHER TOO. If two row masks compiled the
    # same source the sweep is covering one variant and calling it three.
    keep_sources = [(keep_variants.get(f"{KERNEL}|R{''.join(str(f) for f in m)}")
                     or {}).get("source_sha256") for m in MASKS]
    keep_needle = keep_result.get("needle", {})
    flush_needle = flush_result.get("needle", {})

    adjudication: Dict[str, Any] = {
        "per_variant_binaries": per_variant_distinct,
        "binaries_are_distinct": all(entry["distinct"]
                                     for entry in per_variant_distinct.values()),
        "row_masks_compiled_distinct_sources":
            len({s for s in keep_sources if s}) == len(MASKS),
        "all_keep_binaries": keep_digests.get("all_binaries_in_call_order", []),
        "all_flush_binaries": flush_digests.get("all_binaries_in_call_order", []),
        # THE NON-VACUITY FLOOR. Under flush, a zero answer proves the kernel
        # flushed only if the kernel was HANDED the needle.
        "fixture_delivered_the_needle": {
            "keep": bool(keep_needle.get("fixture_delivered_the_needle")),
            "flush": bool(flush_needle.get("fixture_delivered_the_needle"))},
        # KEEP: exact bit patterns, and the coupled/uncoupled split that proves the
        # coupling fired rather than merely that something survived.
        "keep_coupled_is_exactly_two_needles":
            bool(keep_needle.get("coupled_kept_the_doubled_needle")),
        "keep_uncoupled_is_exactly_one_needle":
            bool(keep_needle.get("uncoupled_kept_the_needle")),
        "keep_coupling_fired": bool(keep_needle.get("coupling_fired")),
        "flush_coupled_is_zero": bool(flush_needle.get("coupled_flushed_to_zero")),
        "flush_uncoupled_is_zero": bool(flush_needle.get("uncoupled_flushed_to_zero")),
        "keep_attributes": keep_result.get("kernel_attributes"),
        "flush_attributes": flush_result.get("kernel_attributes"),
    }
    adjudication["pass"] = bool(
        adjudication["binaries_are_distinct"]
        and adjudication["row_masks_compiled_distinct_sources"]
        and adjudication["fixture_delivered_the_needle"]["keep"]
        and adjudication["fixture_delivered_the_needle"]["flush"]
        and adjudication["keep_coupled_is_exactly_two_needles"]
        and adjudication["keep_uncoupled_is_exactly_one_needle"]
        and adjudication["keep_coupling_fired"]
        and adjudication["flush_coupled_is_zero"]
        and adjudication["flush_uncoupled_is_zero"])
    results["adjudication"] = adjudication

    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    for name, entry in per_variant_distinct.items():
        log(f"[binary] {name}: keep={str(entry['keep_binary'])[:12]} "
            f"flush={str(entry['flush_binary'])[:12]} distinct={entry['distinct']}")
    log(f"[done] binaries_distinct={adjudication['binaries_are_distinct']} "
        f"sources_distinct={adjudication['row_masks_compiled_distinct_sources']} "
        f"delivered={adjudication['fixture_delivered_the_needle']} "
        f"keep_2n={adjudication['keep_coupled_is_exactly_two_needles']} "
        f"keep_1n={adjudication['keep_uncoupled_is_exactly_one_needle']} "
        f"flush_zero={adjudication['flush_coupled_is_zero']} "
        f"pass={adjudication['pass']}")
    return 0 if adjudication["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
