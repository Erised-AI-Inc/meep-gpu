#!/usr/bin/env python3
"""Benchmark stock CPU MEEP on workloads shared with the GPU FDTD core.

The harness measures three increasingly complete 3-D paths on the same grid:
complex-field core stepping, all-face PML, and PML plus a one-frequency DFT
plane.  Warmup, timed stepping, result extraction, and end-to-end time are
reported separately.  Run one profile per scheduler allocation; under MPI use
the cluster's native launcher rather than starting ``mpirun`` from Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import meep as mp
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "This benchmark requires an installed CPU MEEP package and NumPy. "
        "Run it from the FDTD development environment."
    ) from exc


PROFILES = ("core", "pml", "pml_dft")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--cell-size", type=float, default=6.0)
    parser.add_argument("--pml-thickness", type=float, default=0.75)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--expected-ranks",
        type=int,
        help="Fail if MEEP's MPI world does not have this size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the aggregate JSON result; rank zero writes it.",
    )
    args = parser.parse_args()

    for name in ("resolution", "warmup_steps", "steps", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cell_size <= 0:
        parser.error("--cell-size must be positive")
    if args.pml_thickness <= 0 or 2 * args.pml_thickness >= args.cell_size:
        parser.error("--pml-thickness must be positive and leave a nonempty interior")
    return args


def _grid_count(length: float, resolution: int) -> int:
    """MEEP's positive-coordinate nearest-integer cell-count convention."""

    return int(math.floor(length * resolution + 0.5))


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _build_simulation(args: argparse.Namespace) -> tuple[Any, Any]:
    cell = mp.Vector3(args.cell_size, args.cell_size, args.cell_size)
    source_z = -0.25 * args.cell_size
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=1.0),
            component=mp.Ez,
            center=mp.Vector3(0, 0, source_z),
        )
    ]
    boundary_layers = (
        [mp.PML(args.pml_thickness)] if args.profile in {"pml", "pml_dft"} else []
    )
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=boundary_layers,
        sources=sources,
        resolution=args.resolution,
        force_complex_fields=True,
    )

    monitor = None
    if args.profile == "pml_dft":
        interior_width = args.cell_size - 2 * args.pml_thickness
        monitor = sim.add_dft_fields(
            [mp.Ez],
            1.0,
            0.0,
            1,
            center=mp.Vector3(0, 0, 0.25 * args.cell_size),
            size=mp.Vector3(interior_width, interior_width, 0),
        )
    return sim, monitor


def _run_once(args: argparse.Namespace, repeat_index: int) -> dict[str, Any]:
    end_to_end_started = time.perf_counter()
    construction_started = time.perf_counter()
    sim, monitor = _build_simulation(args)
    construction_s = time.perf_counter() - construction_started

    init_started = time.perf_counter()
    sim.init_sim()
    mp.all_wait()
    init_s = time.perf_counter() - init_started
    dt = float(sim.fields.dt)

    warmup_start_step = int(sim.timestep())
    warmup_started = time.perf_counter()
    sim.run(until=args.warmup_steps * dt)
    mp.all_wait()
    warmup_s = time.perf_counter() - warmup_started
    warmup_actual_steps = int(sim.timestep()) - warmup_start_step

    measured_start_step = int(sim.timestep())
    stepping_started = time.perf_counter()
    sim.run(until=args.steps * dt)
    mp.all_wait()
    stepping_s = time.perf_counter() - stepping_started
    measured_actual_steps = int(sim.timestep()) - measured_start_step
    if measured_actual_steps != args.steps:
        raise RuntimeError(
            f"Requested {args.steps} measured timesteps but MEEP advanced "
            f"{measured_actual_steps}; refusing a mislabeled throughput result."
        )

    extraction_started = time.perf_counter()
    if monitor is None:
        extracted = sim.get_array(
            center=mp.Vector3(),
            size=mp.Vector3(args.cell_size, args.cell_size, 0),
            component=mp.Ez,
        )
    else:
        extracted = sim.get_dft_array(monitor, mp.Ez, 0)
    mp.all_wait()
    extraction_s = time.perf_counter() - extraction_started

    array = np.asarray(extracted)
    checksum = {
        "shape": list(array.shape),
        "max_abs": float(np.max(np.abs(array))),
        "l2": float(np.linalg.norm(array.ravel())),
        "sum_real": float(np.sum(array.real, dtype=np.float64)),
        "sum_imag": float(np.sum(array.imag, dtype=np.float64)),
    }
    end_to_end_s = time.perf_counter() - end_to_end_started
    sim.reset_meep()

    grid_edge = _grid_count(args.cell_size, args.resolution)
    grid_cells = grid_edge**3
    return {
        "repeat": repeat_index,
        "dt": dt,
        "grid_shape": [grid_edge, grid_edge, grid_edge],
        "grid_cells": grid_cells,
        "warmup_requested_steps": args.warmup_steps,
        "warmup_actual_steps": warmup_actual_steps,
        "measured_requested_steps": args.steps,
        "measured_actual_steps": measured_actual_steps,
        "construction_s": construction_s,
        "init_s": init_s,
        "warmup_s": warmup_s,
        "stepping_s": stepping_s,
        "extraction_s": extraction_s,
        "end_to_end_s": end_to_end_s,
        "steps_per_s": measured_actual_steps / stepping_s,
        "mcells_per_s": grid_cells * measured_actual_steps / stepping_s / 1e6,
        "checksum": checksum,
    }


def _median(runs: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(run[key] for run in runs))


def main() -> int:
    args = _parse_args()
    rank_count = int(mp.count_processors())
    if args.expected_ranks is not None and rank_count != args.expected_ranks:
        raise RuntimeError(
            f"Expected {args.expected_ranks} MPI ranks but MEEP reports {rank_count}. "
            "Use the scheduler's native PMIx launcher."
        )

    runs = [_run_once(args, repeat_index) for repeat_index in range(1, args.repeats + 1)]
    result = {
        "schema_version": 1,
        "benchmark": "meep_gpu.fdtd.cpu-meep",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_features": {
            "complex_fields": True,
            "all_face_pml": args.profile in {"pml", "pml_dft"},
            "single_frequency_dft_plane": args.profile == "pml_dft",
            "continuous_point_source": True,
        },
        "parameters": {
            "resolution": args.resolution,
            "cell_size": [args.cell_size] * 3,
            "pml_thickness": args.pml_thickness if args.profile != "core" else 0.0,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.steps,
            "repeats": args.repeats,
        },
        "runtime": {
            "meep_version": getattr(mp, "__version__", "unknown"),
            "single_precision": bool(mp.is_single_precision()),
            "with_mpi": bool(mp.with_mpi()),
            "mpi_ranks": rank_count,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "host": {
            "hostname": socket.gethostname(),
            "cpu_model": _cpu_model(),
            "logical_cpus_visible_to_rank": (
                len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
            "slurm_ntasks": os.environ.get("SLURM_NTASKS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "runs": runs,
        "median": {
            key: _median(runs, key)
            for key in (
                "construction_s",
                "init_s",
                "warmup_s",
                "stepping_s",
                "extraction_s",
                "end_to_end_s",
                "steps_per_s",
                "mcells_per_s",
            )
        },
    }

    if mp.my_rank() == 0:
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"MEEP_GPU_MEEP_CPU_JSON={json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
