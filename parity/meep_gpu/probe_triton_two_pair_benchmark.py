"""Benchmark the two-pair Triton PML core against four separate kernels.

The covered nondispersive core can now run as either::

    pml_curl_step(B) -> constitutive_step(H)
    pml_curl_step(D) -> constitutive_step(E)       # four launches

or::

    fused_curl_constitutive_B
    fused_curl_constitutive_D                      # two launches

This probe measures that exact comparison on bare CuPy-resident state through the
shipped plan objects. It is deliberately not a whole-driver speedup claim: sources,
monitors, DFT accumulation and Python orchestration are outside the timed interval.

Measurement discipline:

* independent fused and unfused states start from identical float32 bytes;
* every timed window advances both states by the same number of logical steps;
* window order alternates AB/BA so one leg is not always measured with the same
  cache/order advantage;
* CUDA events measure device work, with synchronization outside the interval;
* all 27 live/input volumes are compared as uint32 bits after correctness warmup,
  after every timed window, and at the end;
* every case is appended atomically to JSON before the next begins;
* the primary result uses the shipped BLOCK and Triton-default warp counts;
* the fair warp experiment first tunes all four separate launches independently,
  proves every explicit warp count reproduces default-warp bits, and only then
  compares fused B/D warp pairs against that tuned control in both 2-D and 3-D.
* the policy experiment carries the winning tuples across every primary size and
  boundary set before either tuple is proposed as a shipped default.

Usage::

    CUDA_VISIBLE_DEVICES=<one empty Slurm-assigned device> python -u \
      parity/meep_gpu/probe_triton_two_pair_benchmark.py \
      --out results/two_pair_benchmark.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_API_ROOT = _HERE.parents[1]
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

SEED = 20260811

# The last 2-D shape is the established tier-4 PML footprint. The 3-D shapes keep
# the same dependency graph while exposing a different memory/coalescing regime.
SIZE_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (64, 64, 1),
    (512, 512, 1),
    (2048, 2048, 1),
    (5120, 5120, 1),
    (128, 96, 64),
    (256, 192, 128),
)
BOUNDARY_SETS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),                 # periodic
    (1, 1, 0),                 # metallic x/y, periodic z
)
BLOCKS: Tuple[int, ...] = (128, 256, 512)
WARPS: Tuple[int, ...] = (1, 2, 4, 8)
TUNING_SHAPE = (2048, 2048, 1)
FAIR_TUNING_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (2048, 2048, 1),
    (256, 192, 128),
)
UNFUSED_STAGES: Tuple[str, ...] = ("step_B", "update_H", "step_D", "update_E")
FUSED_POLICY_WARPS: Tuple[int, int] = (1, 1)
UNFUSED_POLICY_WARPS: Tuple[int, int, int, int] = (8, 1, 2, 1)

FIELD_NAMES: Tuple[str, ...] = (
    "Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
    "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)
INVERSE_EPSILON_NAMES: Tuple[str, ...] = (
    "inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez",
)
ALL_STATE_NAMES = FIELD_NAMES + INVERSE_EPSILON_NAMES


def log(message: str) -> None:
    print(message, flush=True)


def atomic_save(payload: Mapping[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_state(shape: Tuple[int, int, int], seed: int = SEED) -> Dict[str, np.ndarray]:
    """Return the complete two-pair state with non-degenerate float32 values."""
    rng = np.random.default_rng(seed)
    state = {
        name: np.ascontiguousarray(
            rng.uniform(-0.5, 0.5, size=shape).astype(np.float32))
        for name in FIELD_NAMES
    }
    for name in INVERSE_EPSILON_NAMES:
        state[name] = np.ascontiguousarray(
            rng.uniform(0.2, 1.25, size=shape).astype(np.float32))
    return state


def host_coefficients(shape: Tuple[int, int, int], seed: int = SEED + 1
                      ) -> Dict[str, Dict[str, np.ndarray]]:
    """Four independent PML table families on their required Yee sublattices."""
    rng = np.random.default_rng(seed)

    def family(stems: Iterable[str]) -> Dict[str, np.ndarray]:
        return {
            f"{stem}_{axis_name}": np.ascontiguousarray(
                rng.uniform(0.55, 0.999, size=shape[axis]).astype(np.float32))
            for axis, axis_name in enumerate("xyz")
            for stem in stems
        }

    return {
        "B_curl_half": family(("kms", "sinv")),
        "H_const_integer": family(("kps", "kms")),
        "D_curl_integer": family(("kms", "sinv")),
        "E_const_half": family(("kps", "kms")),
    }


def to_device(cp: Any, arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return {name: cp.asarray(value) for name, value in arrays.items()}


def restore_state(state: MutableMapping[str, Any], host: Mapping[str, np.ndarray]) -> None:
    for name in ALL_STATE_NAMES:
        # CuPy 13 rejects ``device_array[...] = whole_numpy_array`` through its
        # scalar-fill branch. ``set`` is the explicit contiguous host-to-device
        # copy and avoids any temporary device allocation during restoration.
        state[name].set(host[name])


def zero_metal_arrays(state: Mapping[str, Any], boundaries: Tuple[int, int, int],
                      pair: str) -> int:
    """Apply the driver's real CuPy wall writes between the separate kernels.

    Returns the number of slice-assignment launches. The fused kernels carry these
    writes inline; omitting them from the separate control makes a metallic case
    physically different, while calling this function honestly exposes the extra
    launch-overhead opportunity that fusion has on a walled grid.
    """
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    if pair == "B":
        names = ("Bx", "By", "Bz")
    elif pair == "D":
        names = ("Dx", "Dy", "Dz")
    else:
        raise ValueError(f"pair must be B or D, got {pair!r}")
    launches = 0
    for name in names:
        for axis, metallic in enumerate(boundaries):
            if not metallic or IYEE_SHIFTS[name][axis] != 0:
                continue
            face = [slice(None)] * 3
            face[axis] = 0
            state[name][tuple(face)] = 0
            launches += 1
    return launches


def unfused_launch_count(boundaries: Tuple[int, int, int]) -> int:
    """Four Triton launches plus the array-path wall slice assignments."""
    # Small NumPy arrays are sufficient because only the Yee shifts and boundary
    # booleans decide the count.
    dummy = {name: np.ones((2, 2, 2), dtype=np.float32)
             for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")}
    return (4 + zero_metal_arrays(dummy, boundaries, "B")
            + zero_metal_arrays(dummy, boundaries, "D"))


def build_unfused_stage_plans(
    state: Mapping[str, Any],
    coefficients: Mapping[str, Mapping[str, Any]],
    boundaries: Tuple[int, int, int],
    block: int,
    unfused_warps: Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]
    = None,
) -> Dict[str, Any]:
    """Build the four independently tunable control plans."""
    from meep_gpu.triton_kernels.launch import (  # noqa: PLC0415
        plan_constitutive_from_arrays,
        plan_from_arrays,
    )

    dtdx = 0.35
    b_curl_warps, h_warps, d_curl_warps, e_warps = (
        unfused_warps or (None, None, None, None))
    return {
        "step_B": plan_from_arrays(
            "step_B", state, coefficients["B_curl_half"], boundaries, dtdx,
            block=block, num_warps=b_curl_warps),
        "update_H": plan_constitutive_from_arrays(
            "H", state, coefficients["H_const_integer"], block=block,
            num_warps=h_warps),
        "step_D": plan_from_arrays(
            "step_D", state, coefficients["D_curl_integer"], boundaries, dtdx,
            block=block, num_warps=d_curl_warps),
        "update_E": plan_constitutive_from_arrays(
            "E", state, coefficients["E_const_half"], block=block,
            num_warps=e_warps),
    }


def build_plans(
    state: Mapping[str, Any],
    coefficients: Mapping[str, Mapping[str, Any]],
    shape: Tuple[int, int, int],
    boundaries: Tuple[int, int, int],
    block: int,
    fused_warps: Optional[Tuple[int, int]] = None,
    unfused_warps: Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]]
    = None,
) -> Tuple[Any, Any]:
    """Return callables for four separate launches and the two fused launches."""
    from meep_gpu.triton_kernels.launch import plan_fused_pair_from_arrays  # noqa: PLC0415

    dtdx = 0.35
    zero_metal = tuple(bool(code) for code in boundaries)
    b_warps, d_warps = fused_warps or (None, None)
    stages = build_unfused_stage_plans(
        state, coefficients, boundaries, block, unfused_warps)
    pair_b = plan_fused_pair_from_arrays(
        "B", state, coefficients["B_curl_half"], coefficients["H_const_integer"],
        boundaries, zero_metal, dtdx, block=block, num_warps=b_warps)
    pair_d = plan_fused_pair_from_arrays(
        "D", state, coefficients["D_curl_integer"], coefficients["E_const_half"],
        boundaries, zero_metal, dtdx, block=block, num_warps=d_warps)

    def unfused() -> None:
        stages["step_B"].run()
        zero_metal_arrays(state, boundaries, "B")
        stages["update_H"].run()
        stages["step_D"].run()
        zero_metal_arrays(state, boundaries, "D")
        stages["update_E"].run()

    def fused() -> None:
        pair_b.run()
        pair_d.run()

    return unfused, fused


def bit_differences(cp: Any, left: Mapping[str, Any], right: Mapping[str, Any]
                    ) -> Dict[str, Any]:
    """Compare raw float32 bits on device; signed zero is therefore visible."""
    differing: Dict[str, int] = {}
    total = 0
    for name in ALL_STATE_NAMES:
        count = int(cp.count_nonzero(
            left[name].view(cp.uint32) != right[name].view(cp.uint32)).item())
        if count:
            differing[name] = count
        total += count
    return {
        "bit_identical": not differing,
        "differing_floats": total,
        "differing_arrays": differing,
    }


def event_seconds(cp: Any, call: Any, repeats: int) -> float:
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        call()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop) / 1e3)


def choose_repeats(pilot_seconds: float, target_window_seconds: float,
                   maximum: int) -> int:
    """Choose enough logical steps for a stable window without creating long cases."""
    if not math.isfinite(pilot_seconds) or pilot_seconds <= 0:
        return 3
    return max(3, min(maximum, int(math.ceil(target_window_seconds / pilot_seconds))))


def benchmark_case(cp: Any, shape: Tuple[int, int, int],
                   boundaries: Tuple[int, int, int], block: int,
                   fused_warps: Optional[Tuple[int, int]], windows: int,
                   warmup: int, target_window_seconds: float,
                   max_repeats: int, label: str,
                   unfused_warps: Optional[
                       Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
                   ] = None) -> Dict[str, Any]:
    """One correctness-welded AB/BA timing case."""
    started = time.perf_counter()
    host = host_state(shape)
    host_coeff = host_coefficients(shape)
    coeff = {name: to_device(cp, arrays) for name, arrays in host_coeff.items()}
    unfused_state = to_device(cp, host)
    fused_state = to_device(cp, host)
    unfused, _ = build_plans(
        unfused_state, coeff, shape, boundaries, block, fused_warps, unfused_warps)
    _, fused = build_plans(
        fused_state, coeff, shape, boundaries, block, fused_warps, unfused_warps)

    # Compilation and one positive correctness discriminator happen before timing.
    unfused()
    fused()
    cp.cuda.runtime.deviceSynchronize()
    compile_check = bit_differences(cp, unfused_state, fused_state)
    if not compile_check["bit_identical"]:
        raise RuntimeError(f"{label}: fused and unfused differ after one step: {compile_check}")

    restore_state(unfused_state, host)
    restore_state(fused_state, host)
    for _ in range(warmup):
        unfused()
        fused()
    cp.cuda.runtime.deviceSynchronize()
    warmup_check = bit_differences(cp, unfused_state, fused_state)
    if not warmup_check["bit_identical"]:
        raise RuntimeError(f"{label}: fused and unfused differ after warmup: {warmup_check}")

    # One equal pilot step per state determines the logical-step count per window.
    pilot_unfused = event_seconds(cp, unfused, 1)
    pilot_fused = event_seconds(cp, fused, 1)
    repeats = choose_repeats(
        max(pilot_unfused, pilot_fused), target_window_seconds, max_repeats)

    unfused_windows: List[float] = []
    fused_windows: List[float] = []
    checks: List[Dict[str, Any]] = []
    for window in range(windows):
        if window % 2 == 0:
            unfused_s = event_seconds(cp, unfused, repeats)
            fused_s = event_seconds(cp, fused, repeats)
            order = "unfused-first"
        else:
            fused_s = event_seconds(cp, fused, repeats)
            unfused_s = event_seconds(cp, unfused, repeats)
            order = "fused-first"
        check = bit_differences(cp, unfused_state, fused_state)
        checks.append(check)
        unfused_windows.append(unfused_s / repeats)
        fused_windows.append(fused_s / repeats)
        ratio = unfused_windows[-1] / fused_windows[-1]
        log(f"[{label}] window {window + 1}/{windows} order={order} repeats={repeats} "
            f"unfused={unfused_windows[-1] * 1e6:.1f}us "
            f"fused={fused_windows[-1] * 1e6:.1f}us ratio={ratio:.4f} "
            f"identical={check['bit_identical']}")
        if not check["bit_identical"]:
            raise RuntimeError(f"{label}: state diverged after timing window {window + 1}")

    unfused_median = statistics.median(unfused_windows)
    fused_median = statistics.median(fused_windows)
    cells = int(np.prod(shape))
    return {
        "label": label,
        "shape": list(shape),
        "cells": cells,
        "boundaries": list(boundaries),
        "block": block,
        "fused_warps": list(fused_warps) if fused_warps else None,
        "unfused_warps": list(unfused_warps) if unfused_warps else None,
        "launches": {"unfused": unfused_launch_count(boundaries), "fused": 2},
        "warmup_steps": warmup,
        "repeats_per_window": repeats,
        "windows": windows,
        "pilot_seconds": {"unfused": pilot_unfused, "fused": pilot_fused},
        "unfused_seconds_per_step": unfused_windows,
        "fused_seconds_per_step": fused_windows,
        "unfused_median_seconds": unfused_median,
        "fused_median_seconds": fused_median,
        "fused_over_unfused": unfused_median / fused_median,
        "unfused_mcells_per_second": cells / unfused_median / 1e6,
        "fused_mcells_per_second": cells / fused_median / 1e6,
        "compile_check": compile_check,
        "warmup_check": warmup_check,
        "window_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def tune_unfused_stage_warps(
    cp: Any,
    shape: Tuple[int, int, int],
    boundaries: Tuple[int, int, int],
    block: int,
    windows: int,
    warmup: int,
    target_window_seconds: float,
    max_repeats: int,
    record: Callable[[Dict[str, Any]], None],
) -> Dict[str, int]:
    """Tune the four-kernel control one separable sub-step at a time.

    Each candidate first has to reproduce the default-warp output bit for bit.
    Only then is it timed. The independent minima provide a disciplined control
    candidate without a 4**4 full-sequence search; the subsequent full-sequence
    benchmark measures that selected tuple together, including cache interactions.
    """
    host = host_state(shape)
    host_coeff = host_coefficients(shape)
    coefficients = {
        name: to_device(cp, arrays) for name, arrays in host_coeff.items()}
    selected: Dict[str, int] = {}

    for stage_index, stage in enumerate(UNFUSED_STAGES):
        reference_state = to_device(cp, host)
        reference_plan = build_unfused_stage_plans(
            reference_state, coefficients, boundaries, block)[stage]
        reference_plan.run()
        cp.cuda.runtime.deviceSynchronize()
        rows: List[Dict[str, Any]] = []

        for candidate_index, num_warps in enumerate(WARPS):
            started = time.perf_counter()
            candidate_state = to_device(cp, host)
            candidate_warps: List[Optional[int]] = [None] * len(UNFUSED_STAGES)
            candidate_warps[stage_index] = num_warps
            candidate_plan = build_unfused_stage_plans(
                candidate_state, coefficients, boundaries, block,
                tuple(candidate_warps))[stage]

            candidate_plan.run()
            cp.cuda.runtime.deviceSynchronize()
            check = bit_differences(cp, candidate_state, reference_state)
            if not check["bit_identical"]:
                raise RuntimeError(
                    f"{shape} {stage} num_warps={num_warps} changed output bits: {check}")

            restore_state(candidate_state, host)
            for _ in range(warmup):
                candidate_plan.run()
            cp.cuda.runtime.deviceSynchronize()
            pilot = event_seconds(cp, candidate_plan.run, 1)
            repeats = choose_repeats(pilot, target_window_seconds, max_repeats)
            samples: List[float] = []
            for window in range(windows):
                sample = event_seconds(cp, candidate_plan.run, repeats) / repeats
                samples.append(sample)
                log(
                    f"[unfused-tune {shape} {stage} warps={num_warps}] "
                    f"window {window + 1}/{windows} repeats={repeats} "
                    f"time={sample * 1e6:.1f}us identical={check['bit_identical']}")

            median = statistics.median(samples)
            row = {
                "label": f"unfused:{shape}:{stage}:block{block}:warps{num_warps}",
                "shape": list(shape),
                "cells": int(np.prod(shape)),
                "boundaries": list(boundaries),
                "stage": stage,
                "block": block,
                "num_warps": num_warps,
                "warmup_steps": warmup,
                "repeats_per_window": repeats,
                "windows": windows,
                "pilot_seconds": pilot,
                "seconds_per_launch": samples,
                "median_seconds": median,
                "mcells_per_second": int(np.prod(shape)) / median / 1e6,
                "vs_default_bits": check,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            record(row)
            del candidate_plan, candidate_state
            cp.get_default_memory_pool().free_all_blocks()

        best = min(rows, key=lambda row: row["median_seconds"])
        selected[stage] = int(best["num_warps"])
        log(f"[unfused-tune {shape} {stage}] selected num_warps={selected[stage]} "
            f"time={best['median_seconds'] * 1e6:.1f}us")
        del reference_plan, reference_state
        cp.get_default_memory_pool().free_all_blocks()

    return selected


def environment(cp: Any, triton: Any) -> Dict[str, Any]:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "triton": triton.__version__,
        "device": name.decode() if isinstance(name, bytes) else str(name),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "l2_bytes": int(props.get("l2CacheSize", 0)),
        "driver": int(cp.cuda.runtime.driverGetVersion()),
        "runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }


def source_hashes() -> Dict[str, str]:
    paths = (
        _API_ROOT / "meep_gpu/triton_kernels/kernels.py",
        _API_ROOT / "meep_gpu/triton_kernels/launch.py",
        _API_ROOT / "meep_gpu/triton_kernels/coverage.py",
        pathlib.Path(__file__).resolve(),
    )
    return {str(path.relative_to(_API_ROOT)): sha256(path) for path in paths}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--windows", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--target-window-seconds", type=float, default=0.12)
    parser.add_argument("--max-repeats", type=int, default=500)
    parser.add_argument(
        "--experiments", default="size,block,warps",
        help="comma-separated subset of size, block, warps, fair-warps, policy")
    args = parser.parse_args(argv)
    experiments = {item.strip() for item in args.experiments.split(",") if item.strip()}
    unknown = experiments - {"size", "block", "warps", "fair-warps", "policy"}
    if unknown:
        parser.error(f"unknown experiments: {sorted(unknown)}")
    if args.windows < 3:
        parser.error("--windows must be at least 3")

    import cupy as cp  # noqa: PLC0415
    import triton  # noqa: PLC0415

    out_path = pathlib.Path(args.out).resolve()
    results: Dict[str, Any] = {
        "probe": "triton_two_pair_benchmark",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(cp, triton),
        "sources": source_hashes(),
        "parameters": {
            "windows": args.windows,
            "warmup": args.warmup,
            "target_window_seconds": args.target_window_seconds,
            "max_repeats": args.max_repeats,
            "experiments": sorted(experiments),
            "fused_policy_warps": list(FUSED_POLICY_WARPS),
            "unfused_policy_warps": list(UNFUSED_POLICY_WARPS),
        },
        "size_sweep": [],
        "block_sweep": [],
        "warp_sweep": [],
        "unfused_stage_sweep": [],
        "tuning_choices": [],
        "fair_warp_sweep": [],
        "policy_sweep": [],
    }
    atomic_save(results, out_path)

    case_index = 0
    total = ((len(SIZE_SHAPES) * len(BOUNDARY_SETS) if "size" in experiments else 0)
             + (len(BLOCKS) if "block" in experiments else 0)
             + (len(WARPS) ** 2 if "warps" in experiments else 0)
             + (len(FAIR_TUNING_SHAPES)
                * (len(UNFUSED_STAGES) * len(WARPS) + len(WARPS) ** 2)
                if "fair-warps" in experiments else 0)
             + (len(SIZE_SHAPES) * len(BOUNDARY_SETS)
                if "policy" in experiments else 0))

    def run_and_record(section: str, **kwargs: Any) -> None:
        nonlocal case_index
        case_index += 1
        label = kwargs.pop("label")
        log(f"[case] {case_index}/{total} {label}")
        row = benchmark_case(
            cp, windows=args.windows, warmup=args.warmup,
            target_window_seconds=args.target_window_seconds,
            max_repeats=args.max_repeats, label=label, **kwargs)
        results[section].append(row)
        atomic_save(results, out_path)
        cp.get_default_memory_pool().free_all_blocks()
        log(f"[case] {case_index}/{total} done ratio={row['fused_over_unfused']:.4f} "
            f"elapsed={row['elapsed_seconds']:.2f}s")

    def record_unfused_stage(row: Dict[str, Any]) -> None:
        nonlocal case_index
        case_index += 1
        results["unfused_stage_sweep"].append(row)
        atomic_save(results, out_path)
        log(f"[case] {case_index}/{total} done {row['label']} "
            f"time={row['median_seconds'] * 1e6:.1f}us "
            f"elapsed={row['elapsed_seconds']:.2f}s")

    if "size" in experiments:
        for shape in SIZE_SHAPES:
            for boundaries in BOUNDARY_SETS:
                run_and_record(
                    "size_sweep", shape=shape, boundaries=boundaries, block=256,
                    fused_warps=None,
                    label=f"size:{shape}:{boundaries}:block256:default-warps")

    if "block" in experiments:
        for block in BLOCKS:
            run_and_record(
                "block_sweep", shape=TUNING_SHAPE, boundaries=BOUNDARY_SETS[0],
                block=block, fused_warps=None,
                label=f"block:{TUNING_SHAPE}:periodic:block{block}:default-warps")

    if "warps" in experiments:
        for b_warps in WARPS:
            for d_warps in WARPS:
                run_and_record(
                    "warp_sweep", shape=TUNING_SHAPE, boundaries=BOUNDARY_SETS[0],
                    block=256, fused_warps=(b_warps, d_warps),
                    label=(f"warps:{TUNING_SHAPE}:periodic:block256:"
                           f"B{b_warps}:D{d_warps}"))

    if "fair-warps" in experiments:
        for shape in FAIR_TUNING_SHAPES:
            selected = tune_unfused_stage_warps(
                cp, shape, BOUNDARY_SETS[0], 256,
                windows=args.windows, warmup=args.warmup,
                target_window_seconds=args.target_window_seconds,
                max_repeats=args.max_repeats,
                record=record_unfused_stage,
            )
            control_warps = tuple(selected[stage] for stage in UNFUSED_STAGES)
            results["tuning_choices"].append({
                "shape": list(shape),
                "boundaries": list(BOUNDARY_SETS[0]),
                "block": 256,
                "unfused_warps": dict(selected),
            })
            atomic_save(results, out_path)
            log(f"[unfused-tune {shape}] selected control {control_warps}")
            for b_warps in WARPS:
                for d_warps in WARPS:
                    run_and_record(
                        "fair_warp_sweep", shape=shape,
                        boundaries=BOUNDARY_SETS[0], block=256,
                        fused_warps=(b_warps, d_warps),
                        unfused_warps=control_warps,
                        label=(f"fair-warps:{shape}:periodic:block256:"
                               f"B{b_warps}:D{d_warps}:"
                               f"unfused{control_warps}"))

    if "policy" in experiments:
        for shape in SIZE_SHAPES:
            for boundaries in BOUNDARY_SETS:
                run_and_record(
                    "policy_sweep", shape=shape, boundaries=boundaries, block=256,
                    fused_warps=FUSED_POLICY_WARPS,
                    unfused_warps=UNFUSED_POLICY_WARPS,
                    label=(f"policy:{shape}:{boundaries}:block256:"
                           f"fused{FUSED_POLICY_WARPS}:"
                           f"unfused{UNFUSED_POLICY_WARPS}"))

    primary = results["size_sweep"]
    tuned = results["block_sweep"] + results["warp_sweep"]
    fair = results["fair_warp_sweep"]
    policy = results["policy_sweep"]
    stage_rows = results["unfused_stage_sweep"]
    best_fair_by_shape = []
    for shape in FAIR_TUNING_SHAPES:
        rows = [row for row in fair if tuple(row["shape"]) == shape]
        if rows:
            best = max(rows, key=lambda row: row["fused_over_unfused"])
            best_fair_by_shape.append({
                "shape": list(shape),
                "label": best["label"],
                "ratio": best["fused_over_unfused"],
                "fused_warps": best["fused_warps"],
                "unfused_warps": best["unfused_warps"],
            })
    results["verdict"] = {
        "all_states_bit_identical": all(
            all(check["bit_identical"] for check in row["window_checks"])
            for row in primary + tuned + fair + policy),
        "all_unfused_warp_outputs_bit_identical": all(
            row["vs_default_bits"]["bit_identical"] for row in stage_rows),
        "primary_cases": len(primary),
        "primary_wins": sum(row["fused_over_unfused"] > 1.0 for row in primary),
        "primary_median_ratio": (
            statistics.median(row["fused_over_unfused"] for row in primary)
            if primary else None),
        "best_primary_ratio": max(
            (row["fused_over_unfused"] for row in primary), default=None),
        "worst_primary_ratio": min(
            (row["fused_over_unfused"] for row in primary), default=None),
        "best_tuned": max(
            ({"label": row["label"], "ratio": row["fused_over_unfused"]}
             for row in tuned), key=lambda item: item["ratio"], default=None),
        "best_fair_by_shape": best_fair_by_shape,
        "policy_cases": len(policy),
        "policy_wins": sum(row["fused_over_unfused"] > 1.0 for row in policy),
        "policy_median_ratio": (
            statistics.median(row["fused_over_unfused"] for row in policy)
            if policy else None),
        "policy_worst_ratio": min(
            (row["fused_over_unfused"] for row in policy), default=None),
        "decision_rule": (
            "The primary claim is default-warp, block-256 two-pair versus four-kernel. "
            "The legacy warp sweep compares against the default four-kernel control. "
            "The fair warp sweep first tunes every separate kernel independently, "
            "requires every warp output to match default bits, and then compares "
            "each fused B/D warp pair against that selected four-kernel tuple. A "
            "fair tuned win is still exploratory until its fused setting is covered "
            "by the full mutation/correctness gates. The policy sweep carries the "
            "selected tuples over the complete primary shape/boundary product."),
    }
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_save(results, out_path)
    log(f"[done] primary={len(primary)} wins={results['verdict']['primary_wins']} "
        f"median={results['verdict']['primary_median_ratio']} "
        f"best_tuned={results['verdict']['best_tuned']} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
