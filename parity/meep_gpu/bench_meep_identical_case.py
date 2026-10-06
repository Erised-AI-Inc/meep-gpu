#!/usr/bin/env python
"""Stock MEEP stepping the IDENTICAL simulation the GPU harness lifts, in Mcell-steps/s.

WHY THIS EXISTS. The stock-MEEP rows behind the 2026-09-22 3-D table timed a different
problem from the GPU rows (a 1 um epsilon-12 block, PML 1.0, no monitor, against the
GPU case's epsilon-9 sphere, PML 0.8, one flux monitor). This bench removes the
difference at its root: the geometry is NOT re-typed here. The ``mp.Simulation`` is
built by the builder ``timing_cases.resolve`` returns -- for a built-in case the route
gate's own (``gate_dispatch_fused_route.CASES[case]``, which for ``pml_3d`` is
``gate_dispatch_end_to_end.case_pml_3d``), for a new case the one in ``timing_cases.py``
that ``bench_timing_case.py`` injects into the GPU bench -- flux monitor included. A
geometry digest of the initialised simulation is written into every row, and
``digest_harness_lift.py`` computes the same digest on the object the harness lift
itself initialised.

WHAT IS MEASURED. The stepping rate of the whole grid: cells x steps / wall seconds,
over windows of a fixed number of steps, after ``init_sim`` and a warm-up. The build and
``init_sim`` are timed separately and are in no rate. Steps are COUNTED from MEEP's own
timestep counter (``fields.t``), never assumed. Each window is bracketed by an MPI
barrier, and its seconds are the MAXIMUM over ranks, so the rate is the rate at which
the whole grid advanced.

THE STEPPER. ``fields_step`` (default) calls ``sim.fields.step()`` the counted number of
times: MEEP's own C++ step, which is what ``sim.run`` calls once per step, with the DFT
(flux) accumulation inside it and no Python step function beside it. ``run`` calls
``sim.run(until=steps * dt)``, the call a user types; ``until`` is RELATIVE and the step
count it yields is read back from the counter.

THE CONFIGURATION a row ran is recorded on every row (``configuration``): the MEEP
build's label (``--build-label``; the reference build unless a control names another),
OpenMP threads per rank (``--threads-per-rank``), the binding label the driver passed,
``split_chunks_evenly`` (``--split-chunks-evenly``; MEEP's default, ``True``, when not
given) and the stepper. A comparison reads them to refuse a median that blends two
configurations.

THREADS PER RANK. MEEP's ``setup()`` forces one OpenMP thread when ``OMP_NUM_THREADS``
is unset, initialises MPI with ``MPI_THREAD_FUNNELED`` when built with OpenMP, and
threads its step loops inside each chunk (``omp parallel for collapse(3)``). With
``--threads-per-rank T`` (T > 1) the bench refuses unless ``OMP_NUM_THREADS`` is T and
the ``--build-record`` says MEEP's OWN libraries link an OpenMP runtime (a runtime that
OpenBLAS maps into the process does not count), and records the threads the process
gained after MPI initialisation and each rank's CPU seconds per wall second over the
windows, from which ``timing_records.py record-meep`` strikes a row that did not run T.

WHAT EVERY ROW RECORDS. MEEP version, the single-precision assertion, the MPI rank
count (MEEP's and the communicator's), each rank's CPU affinity, thread environment,
nice level, the launcher line and binding label the driver passed in, the MEEP
configure line where a config.log is named, sha256 of this file and of the modules the
builder came from, the geometry digest, the chunk layout (chunks, cells per rank), build
and init seconds, the warm-up actually taken and the calibration.

ROWS. Appended to ``<out>/rows_r<ranks>.jsonl`` as they land, master rank only:
``environment`` (one per process), ``geometry`` (one per process), ``window`` (one per
timed window) and ``measurement`` (one per process: the median over its windows, the
spread, and everything above). ``build_3d_comparison.py`` reads the ``measurement``
rows and takes the median over passes; a pass is one fresh process.

Runs under ``mpirun -np N`` or ``srun``. Reads no GPU: ``CUDA_VISIBLE_DEVICES`` is
emptied before MEEP is imported, so a MEEP rank can never count as a GPU process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_cases  # noqa: E402

#: MEEP's own timers, read back after the windows where the build exposes them.
MEEP_TIMERS = ("Stepping", "Connecting", "Boundaries", "MpiAllTime", "MpiOneTime",
               "FourierTransforming", "FieldUpdateB", "FieldUpdateH", "FieldUpdateD",
               "FieldUpdateE", "BoundarySteppingB", "BoundarySteppingWH",
               "BoundarySteppingPH", "BoundarySteppingH", "BoundarySteppingD",
               "BoundarySteppingWE", "BoundarySteppingPE", "BoundarySteppingE", "Other")

#: Decimals the sampled epsilon is rounded to for the decomposition-independent hash.
EPS_ROUND_DECIMALS = 5

#: OpenMP runtimes whose presence in the process's mappings shows OpenMP is linked.
OPENMP_RUNTIMES = ("libgomp", "libomp", "libiomp5")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_split(value: str) -> bool:
    """``true``/``false`` -> bool; anything else is refused by name."""
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError(f"--split-chunks-evenly takes true or false, "
                                     f"not {value!r}")


# ---------------------------------------------------------------------------
# The geometry digest
# ---------------------------------------------------------------------------

def _num(value: Any, digits: int = 12) -> Any:
    if isinstance(value, complex):
        return [round(value.real, digits), round(value.imag, digits)]
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return repr(value)


def _vec(value: Any) -> Any:
    if value is None or value is False:
        return None
    try:
        return [_num(value.x), _num(value.y), _num(value.z)]
    except AttributeError:
        return repr(value)


def _material(material: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"class": type(material).__name__}
    for name in ("epsilon_diag", "epsilon_offdiag", "mu_diag", "mu_offdiag",
                 "D_conductivity_diag", "B_conductivity_diag",
                 "E_chi2_diag", "E_chi3_diag"):
        if hasattr(material, name):
            out[name] = _vec(getattr(material, name))
    for name in ("E_susceptibilities", "H_susceptibilities"):
        entries = getattr(material, name, None) or []
        out[name] = [{"class": type(s).__name__,
                      "frequency": _num(getattr(s, "frequency", None)),
                      "gamma": _num(getattr(s, "gamma", None)),
                      "sigma_diag": _vec(getattr(s, "sigma_diag", None))}
                     for s in entries]
    return out


def _geometry(item: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"class": type(item).__name__,
                           "center": _vec(getattr(item, "center", None))}
    for name in ("radius", "height"):
        if hasattr(item, name):
            out[name] = _num(getattr(item, name))
    for name in ("size", "axis", "e1", "e2", "e3"):
        if hasattr(item, name):
            out[name] = _vec(getattr(item, name))
    if hasattr(item, "material"):
        out["material"] = _material(item.material)
    return out


def _source(mp: Any, source: Any) -> Dict[str, Any]:
    src = source.src
    width = getattr(src, "width", None)
    try:
        component = mp.component_name(source.component)
    except Exception:  # noqa: BLE001
        component = repr(source.component)
    return {
        "component": component,
        "center": _vec(source.center), "size": _vec(source.size),
        "amplitude": _num(getattr(source, "amplitude", 1.0)),
        "time_profile": type(src).__name__,
        "frequency": _num(getattr(src, "frequency", None)),
        "width": _num(width),
        "fwidth": _num(1.0 / width) if width else None,
        "start_time": _num(getattr(src, "start_time", None)),
        "cutoff": _num(getattr(src, "cutoff", None)),
        "is_integrated": bool(getattr(src, "is_integrated", False)),
    }


def _monitor(item: Any) -> Dict[str, Any]:
    regions = []
    for region in getattr(item, "regions", None) or []:
        regions.append({"center": _vec(getattr(region, "center", None)),
                        "size": _vec(getattr(region, "size", None)),
                        "direction": getattr(region, "direction", None),
                        "weight": _num(getattr(region, "weight", 1.0))})
    return {"class": type(item).__name__,
            "frequencies": [_num(f) for f in (getattr(item, "freq", None) or [])],
            "regions": regions,
            "decimation_factor": getattr(item, "decimation_factor", None)}


def nominal_counts(sim: Any) -> List[int]:
    """Cells per axis as the comparison counts them: round(cell x resolution)."""
    counts = []
    for extent in (sim.cell_size.x, sim.cell_size.y, sim.cell_size.z):
        counts.append(max(1, int(round(float(extent) * float(sim.resolution)))))
    return counts


def geometry_digest(mp: Any, sim: Any, sample_epsilon: bool = True) -> Dict[str, Any]:
    """What this simulation IS, read off the initialised object. COLLECTIVE under MPI.

    ``digest`` is the sha256 of the canonical facts: cell, resolution, grid counts, dt,
    PML layers, geometry, sources, monitors, and epsilon sampled on the whole grid
    (its shape, range, the count of cells above 1, and the hash of the values rounded
    to ``EPS_ROUND_DECIMALS`` decimals). The exact epsilon hash is recorded beside it
    and is outside the canonical digest, so a last-bit difference between domain
    decompositions would be visible rather than fatal. Unchanged from the 2026-09-28
    bench; ``test_timing_cases.py`` pins its value on ``pml_3d`` at resolution 12.
    """
    import numpy  # noqa: PLC0415

    gv = sim.fields.gv
    grid = {"nx": int(gv.nx()), "ny": int(gv.ny()), "nz": int(gv.nz())}
    nominal = nominal_counts(sim)
    cells = 1
    for axis, count in zip(("nx", "ny", "nz"), nominal):
        cells *= max(1, grid[axis]) if float(getattr(sim.cell_size, axis[1])) else 1
    thickness = [float(layer.thickness) for layer in sim.boundary_layers
                 if int(getattr(layer, "direction", -1)) == -1
                 and int(getattr(layer, "side", -1)) == -1]
    interior = 1.0
    if thickness:
        for extent in (sim.cell_size.x, sim.cell_size.y, sim.cell_size.z):
            if float(extent):
                interior *= max(0.0, float(extent) - 2.0 * thickness[0]) / float(extent)
    facts: Dict[str, Any] = {
        "cell_size": _vec(sim.cell_size),
        "resolution": _num(sim.resolution),
        "dimensions": int(sim.dimensions),
        "grid": grid,
        "cells": int(cells),
        "cells_nominal": int(nominal[0] * nominal[1] * nominal[2]),
        "dt": _num(sim.fields.dt, 15),
        "courant": _num(sim.Courant),
        "k_point": _vec(sim.k_point),
        "force_complex_fields": bool(sim.force_complex_fields),
        "eps_averaging": bool(sim.eps_averaging),
        "subpixel_tol": _num(sim.subpixel_tol),
        "subpixel_maxeval": int(sim.subpixel_maxeval),
        "default_material": _material(sim.default_material)
        if hasattr(sim.default_material, "epsilon_diag") else repr(sim.default_material),
        "symmetries": [type(s).__name__ for s in (sim.symmetries or [])],
        "boundary_layers": [{"class": type(layer).__name__,
                             "thickness": _num(layer.thickness),
                             "direction": int(getattr(layer, "direction", -1)),
                             "side": int(getattr(layer, "side", -1)),
                             "R_asymptotic": _num(getattr(layer, "R_asymptotic", None)),
                             "mean_stretch": _num(getattr(layer, "mean_stretch", None))}
                            for layer in sim.boundary_layers],
        "pml_cell_fraction_nominal": _num(1.0 - interior, 6) if thickness else 0.0,
        "geometry": [_geometry(item) for item in sim.geometry],
        "sources": [_source(mp, source) for source in sim.sources],
        "monitors": [_monitor(item) for item in sim.dft_objects],
    }
    exact = None
    if sample_epsilon:
        started = time.perf_counter()
        epsilon = numpy.ascontiguousarray(sim.get_epsilon())
        rounded = numpy.round(epsilon.astype(numpy.float64), EPS_ROUND_DECIMALS)
        rounded = rounded + 0.0  # -0.0 and 0.0 hash alike
        exact = hashlib.sha256(epsilon.tobytes()).hexdigest()
        facts["epsilon"] = {
            "shape": [int(v) for v in epsilon.shape],
            "dtype": str(epsilon.dtype),
            "min": _num(epsilon.min(), 6), "max": _num(epsilon.max(), 6),
            "cells_above_1": int((epsilon > 1.0 + 1e-6).sum()),
            "sha256_rounded": hashlib.sha256(
                numpy.ascontiguousarray(rounded).tobytes()).hexdigest(),
            "rounded_decimals": EPS_ROUND_DECIMALS,
        }
        sampled = {"seconds": round(time.perf_counter() - started, 3),
                   "sum_float64": _num(epsilon.astype(numpy.float64).sum(), 6)}
    else:
        facts["epsilon"] = None
        sampled = None
    canonical = json.dumps(facts, sort_keys=True, separators=(",", ":"))
    return {"digest": sha256_text(canonical), "facts": facts,
            "epsilon_sha256_exact": exact, "epsilon_sampling": sampled}


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                    "KMP_DUPLICATE_LIB_OK", "CUDA_VISIBLE_DEVICES")

CONDA_PACKAGES = ("pymeep", "meep", "libmeep", "openmpi", "mpich", "mpi", "mpi4py",
                  "libctl", "harminv", "mpb", "fftw", "hdf5", "libblas", "libopenblas",
                  "openblas", "numpy", "python", "llvm-openmp", "libgomp")


def conda_builds() -> Dict[str, str]:
    """name -> "version build" for the packages that decide how MEEP runs."""
    found: Dict[str, str] = {}
    meta = os.path.join(sys.prefix, "conda-meta")
    try:
        names = sorted(os.listdir(meta))
    except OSError:
        return found
    for entry in names:
        if not entry.endswith(".json"):
            continue
        stem = entry[:-5]
        parts = stem.rsplit("-", 2)
        if len(parts) == 3 and parts[0] in CONDA_PACKAGES:
            found[parts[0]] = f"{parts[1]} {parts[2]}"
    return found


def configure_line(config_log: Optional[str]) -> Optional[str]:
    if not config_log:
        return None
    try:
        with open(config_log, "r", encoding="utf-8", errors="replace") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                if line.lstrip().startswith("$ "):
                    return line.strip()[2:]
    except OSError:
        return None
    return None


def box_reading() -> Dict[str, Any]:
    """Load and memory, read from the kernel; no subprocess inside an MPI rank.

    Linux: ``/proc/meminfo``. macOS: ``hw.memsize`` and ``kern.memorystatus_level`` (the
    kernel's percentage of memory available) through ``sysctlbyname``.
    """
    reading: Dict[str, Any] = {"utc": utc()}
    try:
        reading["loadavg"] = [round(v, 2) for v in os.getloadavg()]
    except OSError:
        reading["loadavg"] = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("MemAvailable", "MemFree", "MemTotal"):
                    reading[key + "_gb"] = round(int(rest.split()[0]) / 1e6, 1)
    except OSError:
        if sys.platform == "darwin":
            import timing_host  # noqa: PLC0415 - ctypes only, no subprocess
            total = timing_host.sysctl_value("hw.memsize")
            level = timing_host.sysctl_value("kern.memorystatus_level")
            if total:
                reading["MemTotal_gb"] = round(int(total) / 1e9, 1)
            if total and level and level.isdigit():
                reading["memorystatus_level_percent"] = int(level)
                reading["MemAvailable_gb_memorystatus"] = round(
                    int(total) * int(level) / 100.0 / 1e9, 1)
    return reading


def rank_facts(mp: Any) -> Dict[str, Any]:
    try:
        affinity: Optional[List[int]] = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    return {"rank": int(mp.my_rank()), "pid": os.getpid(), "host": platform.node(),
            "affinity": affinity, "nice": os.nice(0)}


def gather_ranks(mp: Any) -> Dict[str, Any]:
    """Every rank's affinity on the master, through the communicator MEEP initialised."""
    mine = rank_facts(mp)
    try:
        import mpi4py  # noqa: PLC0415
        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        from mpi4py import MPI  # noqa: PLC0415
        if not MPI.Is_initialized():
            raise RuntimeError("MPI is not initialised beside MEEP")
        comm = MPI.COMM_WORLD
        everyone = comm.gather(mine, root=0)
        return {"gathered_by": "mpi4py", "communicator_size": int(comm.Get_size()),
                "mpi_library": MPI.Get_library_version().replace("\x00", "")
                .strip().splitlines()[0],
                "mpi4py": mpi4py.__version__,
                "ranks": everyone if everyone is not None else None}
    except Exception as error:  # noqa: BLE001 - recorded, never fatal
        return {"gathered_by": None, "communicator_size": None,
                "why_not": f"{type(error).__name__}: {error}", "ranks": [mine]}


def core_of_cpu(cpus: Sequence[int], sys_root: str = "/sys") -> Optional[Dict[int, str]]:
    """``"<package>:<core>"`` per CPU from sysfs; ``None`` where sysfs has no topology."""
    out: Dict[int, str] = {}
    for cpu in cpus:
        base = os.path.join(sys_root, "devices", "system", "cpu", f"cpu{int(cpu)}",
                            "topology")
        try:
            with open(os.path.join(base, "physical_package_id"), "r") as handle:
                package = handle.read().strip()
            with open(os.path.join(base, "core_id"), "r") as handle:
                core = handle.read().strip()
        except OSError:
            return None
        out[int(cpu)] = f"{package}:{core}"
    return out


def binding_summary(gathered: Dict[str, Any], scope_cpus: Optional[int] = None,
                    cores: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """The ranks' placement. ``bound`` compares each rank's CPU set with the CPUs the
    launching ladder could use (``scope_cpus``; ``os.cpu_count()`` when not given, the
    2026-09-28 rule): inside a restricted allocation an unbound rank holds the whole
    allocation, which is fewer CPUs than the host has, and must not read as bound.
    ``physical_cores_used`` counts the distinct (package, core) pairs the ranks' sets
    touch, when the core map is readable."""
    ranks = gathered.get("ranks") or []
    sets = [tuple(r["affinity"]) for r in ranks if r.get("affinity") is not None]
    if not sets:
        return {"affinity_available": False}
    widths = sorted({len(s) for s in sets})
    union = sorted({cpu for s in sets for cpu in s})
    scope = int(scope_cpus) if scope_cpus else (os.cpu_count() or 0)
    summary: Dict[str, Any] = {
        "affinity_available": True, "ranks_reporting": len(sets),
        "cpus_per_rank": widths, "distinct_cpu_sets": len(set(sets)),
        "cpus_in_union": len(union), "scope_cpus": scope,
        "scope_from": "the ladder's affinity" if scope_cpus else "os.cpu_count()",
        "bound": bool(widths and widths[-1] < scope)}
    if cores is None:
        cores = core_of_cpu(union)
    if cores is not None and all(cpu in cores for cpu in union):
        summary["physical_cores_used"] = len({cores[cpu] for cpu in union})
        summary["physical_cores_per_rank"] = sorted({len({cores[c] for c in s})
                                                     for s in sets})
    else:
        summary["physical_cores_used"] = None
    return summary


def mapped_openmp() -> Optional[List[str]]:
    """OpenMP runtimes mapped into this process, from ``/proc/self/maps``; ``None``
    where the platform has no such file."""
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None
    return sorted({name for name in OPENMP_RUNTIMES if name in text})


def thread_count() -> Optional[int]:
    """Threads of this process: ``/proc/self/task`` on Linux, ``ps -M`` elsewhere."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        pass
    try:
        text = subprocess.run(["ps", "-M", "-p", str(os.getpid())], capture_output=True,
                              text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    return max(0, len(lines) - 1) or None


def chunk_layout(mp: Any, sim: Any, ranks: int) -> Dict[str, Any]:
    """How MEEP split the grid: chunks, and owned cells per rank (the load balance)."""
    out: Dict[str, Any] = {"num_chunks": None, "cells_per_rank": None}
    try:
        out["num_chunks"] = int(sim.fields.num_chunks)
    except Exception:  # noqa: BLE001
        pass
    try:
        volumes = sim.structure.get_chunk_volumes()
        owners = [int(v) for v in sim.structure.get_chunk_owners()]
        per_rank = [0] * max(ranks, (max(owners) + 1) if owners else 0)
        for volume, owner in zip(volumes, owners):
            per_rank[owner] += int(volume.nowned_min())
        out["cells_per_rank"] = per_rank
        out["chunks_listed"] = len(volumes)
        if per_rank and min(per_rank) > 0:
            out["max_over_min_cells"] = round(max(per_rank) / float(min(per_rank)), 4)
    except Exception as error:  # noqa: BLE001 - recorded, never fatal
        out["why_not"] = f"{type(error).__name__}: {error}"
    return out


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------

def advance(sim: Any, steps: int, stepper: str) -> int:
    """Advance ``steps`` steps and return the number MEEP COUNTED."""
    fields = sim.fields
    before = int(fields.t)
    if stepper == "fields_step":
        step = fields.step
        for _ in range(steps):
            step()
    else:
        sim.run(until=steps * float(fields.dt))
    return int(fields.t) - before


def timed(mp: Any, sim: Any, steps: int, stepper: str) -> Dict[str, Any]:
    """One bracketed advance. Seconds are the maximum over ranks."""
    first = int(sim.fields.t)
    mp.all_wait()
    started = time.perf_counter()
    counted = advance(sim, steps, stepper)
    mp.all_wait()
    local = time.perf_counter() - started
    seconds = float(mp.max_to_all(local))
    return {"steps_requested": int(steps), "steps": int(counted),
            "seconds": seconds, "seconds_this_rank": local,
            "step_first": first, "step_last": first + counted}


def spread(values: Sequence[float]) -> float:
    """(max - min) / median, the GPU harness's definition."""
    if not values:
        return float("nan")
    median = statistics.median(values)
    if median == 0:
        return float("inf")
    return (max(values) - min(values)) / median


def append(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness-root", default=None, dest="harness_root",
                        help=f"directory holding {timing_cases.HARNESS_MARKER}; found by "
                             "walking up from this file when omitted")
    parser.add_argument("--case", default="pml_3d")
    parser.add_argument("--table", default=None, choices=("triton", "cuda", "metal"),
                        help="the kernel table whose GPU rows this case must match "
                             "(timing_cases.resolve); needed only for a name the Metal "
                             "route gate builds with a builder of its own")
    parser.add_argument("--res", type=int, required=True)
    parser.add_argument("--out", required=True,
                        help="directory for rows_r<ranks>.jsonl")
    parser.add_argument("--pass-index", type=int, default=1, dest="pass_index")
    parser.add_argument("--passes-planned", type=int, default=1, dest="passes_planned")
    parser.add_argument("--tag", default="record",
                        help="what the row is for: record, preflight, bindprobe, laptop")
    parser.add_argument("--row-id", default=None, dest="row_id")
    parser.add_argument("--expect-ranks", type=int, default=None, dest="expect_ranks")
    parser.add_argument("--warm-steps", type=int, default=400, dest="warm_steps",
                        help="warm-up steps asked for")
    parser.add_argument("--warm-seconds-cap", type=float, default=20.0,
                        dest="warm_seconds_cap",
                        help="the warm-up stops short of --warm-steps at this wall time")
    parser.add_argument("--warm-min-steps", type=int, default=16, dest="warm_min_steps")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--target-seconds", type=float, default=2.0,
                        dest="target_seconds",
                        help="window length the step count is sized for by calibration")
    parser.add_argument("--window-steps", type=int, default=None, dest="window_steps",
                        help="steps per window, stated; overrides the calibration")
    parser.add_argument("--min-window-steps", type=int, default=8,
                        dest="min_window_steps")
    parser.add_argument("--step-cap", type=int, default=4000, dest="step_cap")
    parser.add_argument("--stepper", default="fields_step",
                        choices=("fields_step", "run"))
    parser.add_argument("--no-epsilon", action="store_true", dest="no_epsilon",
                        help="leave epsilon out of the digest (it is sampled on the "
                             "whole grid otherwise)")
    parser.add_argument("--digest-only", action="store_true", dest="digest_only",
                        help="build, initialise, write the geometry row and exit")
    parser.add_argument("--launcher-line", default=None, dest="launcher_line")
    parser.add_argument("--binding-label", default=None, dest="binding_label")
    parser.add_argument("--bindings-report", default=None, dest="bindings_report",
                        help="path of the launcher's binding report output")
    parser.add_argument("--meep-config-log", default=None, dest="meep_config_log")
    parser.add_argument("--box-before", default=None, dest="box_before",
                        help="the driver's own box reading before this row")
    parser.add_argument("--build-label", default=None, dest="build_label",
                        help="the MEEP build this interpreter carries; 'reference' "
                             "when not given")
    parser.add_argument("--build-record", default=None, dest="build_record",
                        help="that build's build_record.json (flags, libraries, OpenMP)")
    parser.add_argument("--threads-per-rank", type=int, default=None,
                        dest="threads_per_rank",
                        help="OpenMP threads per MPI rank; requires OMP_NUM_THREADS to "
                             "match and an OpenMP runtime in the process")
    parser.add_argument("--split-chunks-evenly", type=parse_split, default=None,
                        dest="split_chunks_evenly",
                        help="true or false, set on the simulation before init_sim; "
                             "MEEP's default (true) when not given")
    parser.add_argument("--scope-cpus", type=int, default=None, dest="scope_cpus",
                        help="the CPUs the launching ladder may use (its affinity); a "
                             "rank is 'bound' when its set is narrower; os.cpu_count() "
                             "when not given")
    return parser


def read_build_record(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise SystemExit(f"REFUSING: the build record {path} does not read: {error}")


def check_threads(threads: Optional[int],
                  build_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Refuse a threaded row whose environment cannot run the threads it names.

    WHETHER MEEP WAS BUILT WITH OPENMP is read from the build record's ``openmp``:
    ``timing_records.py meep-build-record`` decides it from the libraries MEEP's own
    images link DIRECTLY. A runtime merely MAPPED into the process proves nothing --
    an OpenBLAS built with OpenMP maps ``libgomp``/``libomp`` into a MEEP that has no
    threaded loop -- so it is recorded, and its absence still refuses, but its presence
    never admits a row. The ladder passes the record it probed itself.
    """
    if threads is None or threads == 1:
        return {"asked": threads or 1, "omp_runtime_mapped": mapped_openmp()}
    if threads < 1:
        raise SystemExit(f"REFUSING: --threads-per-rank {threads}")
    variable = os.environ.get("OMP_NUM_THREADS")
    if variable != str(threads):
        raise SystemExit(f"REFUSING: --threads-per-rank {threads} and OMP_NUM_THREADS is "
                         f"{variable!r}; MEEP's setup() takes its thread count from that "
                         "variable")
    mapped = mapped_openmp()
    linked = (build_record or {}).get("openmp")
    if mapped is not None and not mapped:
        raise SystemExit(f"REFUSING: --threads-per-rank {threads} and no OpenMP runtime "
                         f"({', '.join(OPENMP_RUNTIMES)}) is mapped into this process; "
                         "this MEEP was not built with OpenMP")
    if build_record is None:
        raise SystemExit(f"REFUSING: --threads-per-rank {threads} needs --build-record: "
                         "whether MEEP's own libraries link OpenMP is read from the record "
                         "(a runtime another library maps into the process does not count)")
    if linked is not True:
        raise SystemExit(f"REFUSING: --threads-per-rank {threads} and the build record "
                         f"says MEEP's own libraries link no OpenMP runtime (openmp="
                         f"{linked!r}); a threaded row would run one thread per rank")
    return {"asked": threads, "omp_runtime_mapped": mapped, "build_record_openmp": linked}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.windows < 3 and not args.digest_only:
        raise SystemExit(f"--windows must be at least 3; got {args.windows}")

    # NO GPU, BEFORE ANYTHING IS IMPORTED: a MEEP rank must never be a GPU process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    box_at_start = box_reading()
    build_record = read_build_record(args.build_record)
    thread_check = check_threads(args.threads_per_rank, build_record)

    import meep as mp  # noqa: PLC0415

    try:
        mp.verbosity(0)
    except Exception:  # noqa: BLE001
        pass
    master = bool(mp.am_master())
    ranks = int(mp.count_processors())

    def say(message: str) -> None:
        if master:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    single = bool(mp.is_single_precision())
    if not single:
        raise SystemExit("REFUSING: this MEEP build is double precision "
                         "(is_single_precision() is False); the GPU engine steps "
                         "float32 and the comparison is single against single")
    if args.expect_ranks is not None and ranks != args.expect_ranks:
        raise SystemExit(f"REFUSING: asked for {args.expect_ranks} ranks and MEEP "
                         f"counts {ranks}")

    harness_root = args.harness_root or timing_cases.find_harness_root(HERE)
    builder, builder_facts = timing_cases.resolve(args.case, harness_root, args.table)
    builder_facts["meep_gpu_imported"] = "meep_gpu" in sys.modules
    builder_facts["device_libraries_imported"] = sorted(
        name for name in ("cupy", "triton", "torch") if name in sys.modules)

    rows_path = os.path.join(args.out, f"rows_r{ranks}.jsonl")
    if master:
        os.makedirs(args.out, exist_ok=True)
    row_id = args.row_id or (f"{args.case}_res{args.res}_r{ranks}_p{args.pass_index}"
                             f"_{int(time.time())}")
    gathered = gather_ranks(mp)
    # THE THREAD BASELINE, taken after MPI initialisation (its helper threads are in
    # it) and before any OpenMP region has run: what OpenMP adds is read against it.
    threads_at_mpi_init = thread_count() if (args.threads_per_rank or 1) > 1 or \
        os.path.isdir("/proc/self/task") else None
    if gathered.get("communicator_size") not in (None, ranks):
        raise SystemExit(f"REFUSING: MEEP counts {ranks} ranks and the communicator "
                         f"has {gathered['communicator_size']}")

    configuration = {
        "build": args.build_label or "reference",
        "threads_per_rank": int(args.threads_per_rank or 1),
        "split_chunks_evenly": ("default" if args.split_chunks_evenly is None
                                else bool(args.split_chunks_evenly)),
        "stepper": args.stepper, "binding": args.binding_label,
    }
    common = {"row_id": row_id, "tag": args.tag, "case": args.case,
              "resolution": int(args.res), "ranks": ranks, "pass": int(args.pass_index),
              "passes_planned": int(args.passes_planned), "configuration": configuration}
    bindings_text = None
    if args.bindings_report and master:
        try:
            with open(args.bindings_report, "r", encoding="utf-8",
                      errors="replace") as handle:
                bindings_text = [line.rstrip("\n") for line in handle
                                 if "bound" in line or "cpu-bind" in line
                                 or "cpu_bind" in line][:ranks + 4]
        except OSError:
            bindings_text = None
    environment = {
        "meep": mp.__version__, "meep_file": getattr(mp, "__file__", None),
        "meep_extension": getattr(getattr(mp, "_meep", None), "__file__", None),
        "single_precision": single, "with_mpi": bool(mp.with_mpi()),
        "ranks": ranks, "mpi": {k: v for k, v in gathered.items() if k != "ranks"},
        "rank_bindings": gathered.get("ranks"),
        "binding": binding_summary(gathered, args.scope_cpus),
        "binding_label": args.binding_label,
        "launcher_line": args.launcher_line,
        "launcher_bindings_report": bindings_text,
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "thread_check": thread_check,
        "nice": os.nice(0),
        "host": platform.node(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0], "executable": sys.executable,
        "conda_builds": conda_builds(),
        "meep_configure_line": configure_line(args.meep_config_log),
        "meep_config_log": args.meep_config_log,
        "build_record": ({"path": os.path.abspath(args.build_record),
                          "sha256": sha256_file(args.build_record),
                          "openmp": (build_record or {}).get("openmp"),
                          "makefile_flags": (build_record or {}).get("makefile_flags")}
                         if args.build_record else None),
        "bench": {"path": os.path.abspath(__file__),
                  "sha256": sha256_file(os.path.abspath(__file__))},
        "builder": builder_facts,
        "argv": list(sys.argv[1:]) if argv is None else list(argv),
        "box_at_start": box_at_start, "box_before_from_driver": args.box_before,
    }
    try:
        import numpy  # noqa: PLC0415
        environment["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        environment["numpy"] = None
    if master:
        append(rows_path, {"row": "environment", "utc": utc(), **common, **environment})
    say(f"stock MEEP {mp.__version__} single_precision={single} ranks={ranks} "
        f"binding={args.binding_label} case={args.case} res={args.res} "
        f"pass {args.pass_index}/{args.passes_planned} tag={args.tag} "
        f"build={configuration['build']} threads={configuration['threads_per_rank']} "
        f"split={configuration['split_chunks_evenly']}")

    # BUILD AND INITIALISE, each timed, neither in any rate.
    mp.all_wait()
    started = time.perf_counter()
    sim, monitors, until = builder(mp, args.res)
    build_seconds = float(mp.max_to_all(time.perf_counter() - started))
    if args.split_chunks_evenly is not None:
        # READ AT STRUCTURE CREATION (meep/simulation.py passes the attribute into the
        # structure it builds in init_sim), so it is set between the builder and init.
        sim.split_chunks_evenly = bool(args.split_chunks_evenly)
    mp.all_wait()
    started = time.perf_counter()
    sim.init_sim()
    mp.all_wait()
    init_seconds = float(mp.max_to_all(time.perf_counter() - started))
    say(f"built in {build_seconds:.2f} s, init_sim in {init_seconds:.2f} s")

    digest = geometry_digest(mp, sim, sample_epsilon=not args.no_epsilon)
    facts = digest["facts"]
    cells = int(facts["cells"])
    if cells != int(facts["cells_nominal"]):
        raise SystemExit(f"REFUSING: MEEP's grid holds {cells} cells and "
                         f"round(cell x resolution) gives {facts['cells_nominal']}; "
                         "the rows would not line up with the GPU rows")
    layout = chunk_layout(mp, sim, ranks)
    layout["split_chunks_evenly_read_back"] = bool(getattr(sim, "split_chunks_evenly",
                                                           True))
    common["cells"] = cells
    common["dimensions"] = int(facts["dimensions"])
    common["geometry_digest"] = digest["digest"]
    setup = {"build_seconds": build_seconds, "init_seconds": init_seconds,
             "monitors_attached": {"dft_objects": len(sim.dft_objects),
                                   "returned_by_builder": len(monitors or []),
                                   "classes": [type(m).__name__
                                               for m in sim.dft_objects]},
             "until_of_case": until, "chunks": layout, "dt": float(sim.fields.dt)}
    if master:
        append(rows_path, {"row": "geometry", "utc": utc(), **common, **setup,
                           "geometry": digest})
    say(f"geometry digest {digest['digest'][:16]} cells={cells:,} "
        f"grid={facts['grid']} monitors={len(sim.dft_objects)} "
        f"chunks={layout.get('num_chunks')} "
        f"epsilon_exact={(digest['epsilon_sha256_exact'] or 'not sampled')[:16]}")
    if args.digest_only:
        say("digest only: no stepping")
        return 0
    if not sim.dft_objects:
        raise SystemExit("REFUSING: the case carries no DFT monitor after init_sim; "
                         "the GPU rows are timed with the flux monitor attached")

    # WARM-UP: --warm-steps, or as many as fit the wall cap, never under the floor.
    first = timed(mp, sim, max(1, args.warm_min_steps), args.stepper)
    per_step = first["seconds"] / float(max(1, first["steps"]))
    left = max(0, args.warm_steps - first["steps"])
    affordable = int(max(0.0, args.warm_seconds_cap - first["seconds"]) / per_step) \
        if per_step > 0 else left
    rest_steps = min(left, affordable)
    rest = timed(mp, sim, rest_steps, args.stepper) if rest_steps > 0 else None
    warm = {"steps_asked": int(args.warm_steps),
            "seconds_cap": float(args.warm_seconds_cap),
            "min_steps": int(args.warm_min_steps),
            "steps": first["steps"] + (rest["steps"] if rest else 0),
            "seconds": first["seconds"] + (rest["seconds"] if rest else 0.0),
            "stopped_by": "steps" if rest_steps == left else "seconds_cap"}
    say(f"warm-up {warm['steps']} steps in {warm['seconds']:.2f} s "
        f"(asked {args.warm_steps}, cap {args.warm_seconds_cap} s, "
        f"stopped by {warm['stopped_by']})")

    # CALIBRATION: a chunk long enough to price a step, doubled until it is.
    calibration: List[Dict[str, Any]] = []
    chunk = max(1, args.min_window_steps)
    while True:
        sample = timed(mp, sim, chunk, args.stepper)
        calibration.append({"steps": sample["steps"], "seconds": sample["seconds"]})
        if sample["seconds"] >= 0.25 or chunk >= 512 or len(calibration) >= 8:
            break
        chunk *= 2
    last = calibration[-1]
    per_step = last["seconds"] / float(max(1, last["steps"]))
    if args.window_steps:
        steps_per_window = int(args.window_steps)
        sized_by = "stated"
    else:
        steps_per_window = int(round(args.target_seconds / per_step)) \
            if per_step > 0 else args.step_cap
        steps_per_window = max(args.min_window_steps,
                               min(args.step_cap, steps_per_window))
        sized_by = "calibration"
    say(f"window sized at {steps_per_window} steps by {sized_by} "
        f"(~{1e3 * per_step:.3f} ms/step, target {args.target_seconds} s)")

    windows: List[Dict[str, Any]] = []
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    for index in range(args.windows):
        entry = timed(mp, sim, steps_per_window, args.stepper)
        if entry["steps"] <= 0 or entry["seconds"] <= 0:
            raise SystemExit(f"window {index} advanced {entry['steps']} steps in "
                             f"{entry['seconds']} s; a rate cannot be formed")
        entry["window"] = index
        entry["seconds_per_step"] = entry["seconds"] / float(entry["steps"])
        entry["mcell_steps_per_s"] = cells * entry["steps"] / entry["seconds"] / 1e6
        windows.append(entry)
        if master:
            append(rows_path, {"row": "window", "utc": utc(), **common,
                               "stepper": args.stepper, **entry})
        say(f"window {index + 1}/{args.windows}: {cells:,} cells x {entry['steps']} "
            f"steps in {entry['seconds']:.3f} s = "
            f"{entry['mcell_steps_per_s']:.2f} Mcell-steps/s "
            f"({1e3 * entry['seconds_per_step']:.3f} ms/step)")
    wall = time.perf_counter() - wall_started
    over_wall = (time.process_time() - cpu_started) / wall if wall > 0 else 0.0
    threads_now = thread_count() if (args.threads_per_rank or 1) > 1 or \
        os.path.isdir("/proc/self/task") else None
    added = (threads_now - threads_at_mpi_init
             if threads_now is not None and threads_at_mpi_init is not None else None)
    observed = {
        "threads_this_rank": threads_now,
        "threads_at_mpi_init_this_rank": threads_at_mpi_init,
        "threads_min": (-int(mp.max_to_all(-float(threads_now)))
                        if threads_now is not None else None),
        "omp_threads_added_min": (-int(mp.max_to_all(-float(added)))
                                  if added is not None else None),
        "cpu_seconds_over_wall_this_rank": round(over_wall, 4),
        "cpu_seconds_over_wall_min": round(-float(mp.max_to_all(-over_wall)), 4),
        "cpu_seconds_over_wall_max": round(float(mp.max_to_all(over_wall)), 4),
    }

    timers: Dict[str, Any] = {}
    for name in MEEP_TIMERS:
        what = getattr(mp, name, None)
        if what is None:
            continue
        try:
            spent = [float(v) for v in sim.time_spent_on(what)]
        except Exception:  # noqa: BLE001
            continue
        if spent and max(spent) > 0:
            timers[name] = {"mean_seconds": sum(spent) / len(spent),
                            "max_seconds": max(spent)}

    # THE MONITOR WAS STEPPED, SHOWN NOT ASSUMED: MEEP's Fourier-transform timer having
    # run, or the flux the monitor holds being nonzero. COLLECTIVE: every rank calls.
    fluxes: List[Any] = []
    for item in sim.dft_objects:
        try:
            fluxes.append([float(v) for v in mp.get_fluxes(item)])
        except Exception as error:  # noqa: BLE001 - not every DFT object is a flux
            fluxes.append(f"unread: {type(error).__name__}")
    flux_nonzero = any(isinstance(values, list) and any(v != 0.0 for v in values)
                       for values in fluxes)
    dft_seconds = (timers.get("FourierTransforming") or {}).get("max_seconds")
    monitor = {"fluxes": fluxes, "flux_nonzero": flux_nonzero,
               "meep_fourier_transform_seconds": dft_seconds,
               "stepped": bool(flux_nonzero or (dft_seconds or 0.0) > 0.0)}

    rates = [w["mcell_steps_per_s"] for w in windows]
    per_steps = [w["seconds_per_step"] for w in windows]
    median_rate = statistics.median(rates)
    measurement = {
        "row": "measurement", "utc": utc(), **common, **setup,
        "stepper": args.stepper,
        "windows": len(windows), "steps_per_window": steps_per_window,
        "window_sized_by": sized_by, "target_seconds": float(args.target_seconds),
        "steps_measured": sum(w["steps"] for w in windows),
        "seconds": sum(w["seconds"] for w in windows),
        "median_seconds_per_step": statistics.median(per_steps),
        "ms_per_step": statistics.median(per_steps) * 1e3,
        "mcell_steps_per_s": median_rate,
        "mcell_steps_per_s_min": min(rates), "mcell_steps_per_s_max": max(rates),
        "mcell_steps_per_s_windows": rates,
        "spread": spread(per_steps),
        "spread_within_0.05": bool(spread(per_steps) <= 0.05),
        "drift_last_over_first": rates[-1] / rates[0],
        "steps_first_window_starts_at": windows[0]["step_first"],
        "steps_last_window_ends_at": windows[-1]["step_last"],
        "warm_up": warm, "calibration": calibration,
        "threads_observed": observed,
        "monitor": monitor,
        "meep_timers_since_init": timers,
        "geometry": {"digest": digest["digest"],
                     "epsilon_sha256_exact": digest["epsilon_sha256_exact"]},
        "environment": environment,
        "box_at_end": box_reading(),
    }
    if not monitor["stepped"]:
        measurement["struck"] = ("the flux monitor shows no sign of having been "
                                 "stepped: MEEP's Fourier-transform timer reads zero "
                                 "and the monitor holds zero flux")
    if master:
        append(rows_path, measurement)
    if not monitor["stepped"]:
        say(f"REFUSING the row: {measurement['struck']}")
        return 3
    say(f"MEASURED {args.case} res {args.res} {cells:,} cells ranks {ranks} "
        f"pass {args.pass_index}/{args.passes_planned}: "
        f"{median_rate:.2f} Mcell-steps/s median of {len(windows)} windows "
        f"(min {min(rates):.2f} max {max(rates):.2f} spread "
        f"{measurement['spread']:.3f}) init {init_seconds:.1f} s -> {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
