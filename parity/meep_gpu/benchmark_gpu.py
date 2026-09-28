"""Throughput benchmark for the meep_gpu engine, mirroring benchmark_meep_cpu.py.

Emits the same phase breakdown and derived rates as the stock-MEEP CPU harness so
the two are directly comparable: a speedup is only meaningful when profile, grid,
precision, source, warmup, measured-step count, PML and monitor all match. Run one
profile per allocation and compare median ``stepping_s`` for the kernel path and
median ``end_to_end_s`` for application-level speedup; report ``extraction_s``
separately rather than folding it into an unspecified timing.

Sweeps grid size because the GPU does NOT win at every size: kernel-launch overhead
dominates on small grids (measured 30x30x60: NumPy 0.46 s vs CuPy 2.80 s), so the
crossover point is itself a result worth publishing.

``--backend gpu`` builds ``prefer_gpu=True`` drivers, which run on this host's GPU:
CUDA through the package's optional CuPy runtime, or Metal kernels over NumPy host
arrays on an Apple GPU (a host with neither raises at the first build). The resolved
GPU is recorded in the output, with the array module the engine runs on and where
the timed steps ran: on an Apple GPU the ARRAY path is NumPy on the host CPU, so
``--backend gpu --step-path array`` there times the same code as ``--backend numpy``
and is not a GPU number, while on a CUDA host the same flags time CuPy on the device.
``--backend numpy`` builds ``prefer_gpu=False``
drivers: the NumPy reference, which runs anywhere, never dispatches a kernel
whatever ``MEEP_GPU_DISPATCH`` says, and is what the GPU path is bit-identical to.
``--backend numpy --step-path fused`` is therefore refused at the command line.

``--step-path {array,fused}`` names which stepping implementation the numbers
belong to, asserted against ``driver.active_step_path`` after warmup: a run that
did not take the requested path aborts rather than reporting mislabeled numbers.
``--step-path array`` (the default) sets ``MEEP_GPU_DISPATCH=0`` in this process
before any driver is built: dispatch is on by default, so on a certified host an
unpinned "array" baseline would plan kernels and abort at the step-path check. The
value is recorded in the output.

``--expansion-probe NAME=PATH`` (repeatable) installs an expansion-probe licence in
the environment BEFORE the driver is built. A COMPLEX-storage fused run needs one:
the complex families take their expansion licence from a probe record, and without
it the composer refuses each of them by name -- "no complex-multiply expansion probe
artifact is available ... which arm the numpy reference takes is a measured platform
fact and may not be guessed". Every name and the sha256 of every artifact is written
into the output, because a fused complex number means nothing without saying which
licence admitted the arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

from meep_gpu.driver import FdtdDriver

PROFILES = ("core", "pml", "pml_dft")


def build(profile, cells, resolution, prefer_gpu, pml_cells, storage="complex"):
    """One configured, un-run driver.

    ``storage`` DECIDES WHICH FUSED PRODUCTS CAN BE REACHED, which is why it is an
    argument rather than the constant it was. Every baseline recorded before
    2026-09-11 was taken with ``force_complex_fields=True``, so that stays the default
    and those numbers remain comparable.

    THIS DOCSTRING SAID "REAL ONLY" UNTIL 2026-09-17, AND THAT HAS STOPPED BEING TRUE.
    It read that the released fused arms on all three tables are the REAL-storage ones
    "and the complex families are released on no timing path", so a complex fused run
    was refused whole -- measured on the GPU host, "no slot is left carrying a kernel:
    every product was refused, dropped as a no-op, or failed to compile". The complex
    families are released now: eleven of the twenty-eight released arms store complex
    fields, and the 2026-09-17 round adds more. What a complex fused run still needs is
    the EXPANSION-PROBE LICENCE those families take their admission from, which is what
    ``--expansion-probe`` installs; the refusal above is what an unlicensed complex run
    still reads, and it is the licence that was missing rather than the release.

    The two storages are never mixed in one comparison -- that much is unchanged, and
    a real-storage fused number and a complex-storage one answer different questions.
    """
    cell_size = tuple(c / resolution for c in cells)
    driver = FdtdDriver(cell_size=cell_size, resolution=resolution,
                        force_complex_fields=(storage == "complex"),
                        prefer_gpu=prefer_gpu)
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    if profile in ("pml", "pml_dft"):
        driver.setup_pml(pml_cells)
    z_low = -0.25 * cell_size[2]
    driver.add_source({"component": "Ez", "frequency": 1.0,
                       "center": (0.0, 0.0, z_low), "size": (0.0, 0.0, 0.0)})
    monitor = None
    if profile == "pml_dft":
        monitor = driver.add_dft_monitor(
            frequencies=1.0, components=("Ex", "Ey", "Ez"),
            center=(0.0, 0.0, 0.25 * cell_size[2]),
            size=(0.5 * cell_size[0], 0.5 * cell_size[1], 0.0),
        )
    return driver, monitor


def synchronize(driver):  # Block until queued device work has actually finished.
    module = getattr(driver.xp, "cuda", None)
    if module is not None:
        module.runtime.deviceSynchronize()


def require_step_path(driver, requested):
    """Refuse to time a run whose stepping path is not the one requested.

    ``driver.active_step_path`` reads which implementation actually dispatched
    ("array" or "fused") once the first step froze the configuration. A fused
    number from a run that silently fell back to the array path — or an "array
    baseline" that quietly dispatched kernels — is not a result, so the mismatch
    aborts before a single measured step rather than mislabeling the sweep.
    """
    actual = driver.active_step_path
    if actual != requested:
        raise SystemExit(
            f"--step-path {requested} was requested but the driver stepped the {actual!r} path. "
            f"For 'fused': this configuration is not covered by any gated kernel slice, or "
            f"MEEP_GPU_FUSED=0 or MEEP_GPU_DISPATCH=0 is set. For 'array': this tool sets "
            f"MEEP_GPU_DISPATCH=0 itself, so a dispatched step here means something "
            f"re-enabled it in-process. No numbers are reported for a path that did not run."
        )


def one_repeat(profile, cells, resolution, prefer_gpu, pml_cells, warmup, steps,
               step_path, storage="complex"):
    timings = {}
    start = time.perf_counter()
    driver, monitor = build(profile, cells, resolution, prefer_gpu, pml_cells, storage)
    synchronize(driver)
    timings["construction_s"] = time.perf_counter() - start

    mark = time.perf_counter()
    driver.run(num_steps=warmup)          # Warmup absorbs kernel compilation and pool growth.
    synchronize(driver)
    timings["warmup_s"] = time.perf_counter() - mark
    require_step_path(driver, step_path)  # After the freeze, before anything is measured.

    mark = time.perf_counter()
    driver.run(num_steps=steps)           # The timestep-only measurement.
    synchronize(driver)
    timings["stepping_s"] = time.perf_counter() - mark

    mark = time.perf_counter()
    field = driver.get_field("Ez")        # Device-to-host extraction, timed separately.
    if monitor is not None:
        monitor.get_dft("Ez")
    synchronize(driver)
    timings["extraction_s"] = time.perf_counter() - mark
    timings["end_to_end_s"] = time.perf_counter() - start

    grid_cells = int(np.prod(driver.shape))
    timings["steps_per_s"] = steps / timings["stepping_s"]
    timings["mcells_per_s"] = grid_cells * steps / timings["stepping_s"] / 1e6
    checksum = {"shape": list(driver.shape),
                "max_abs": float(np.abs(field).max()),
                "l2": float(np.linalg.norm(field))}
    driver.close()
    return timings, grid_cells, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="pml_dft")
    parser.add_argument("--backend", choices=("gpu", "numpy"), default="gpu")
    parser.add_argument("--step-path", choices=("array", "fused"), default="array",
                        help="Which stepping path the timed run MUST take, asserted against "
                             "driver.active_step_path after warmup; a mismatch aborts the sweep "
                             "instead of reporting numbers under the wrong label. Every recorded "
                             "baseline to date is the array path.")
    parser.add_argument("--cells", type=int, nargs=3, action="append",
                        help="Grid in cells, repeatable to sweep (default: a 4-point sweep).")
    parser.add_argument("--resolution", type=float, default=16.0)
    parser.add_argument("--pml-cells", type=int, default=12)
    parser.add_argument("--storage", choices=("complex", "real"), default="complex",
                        help="field storage. 'complex' is what every baseline "
                             "before 2026-09-11 used and stays the default so those "
                             "numbers remain comparable. A REAL-storage fused run "
                             "needs nothing else; a COMPLEX-storage fused run needs "
                             "the expansion-probe licence the complex families take "
                             "their admission from, which --expansion-probe installs. "
                             "Never compare a run in one storage against a run in the "
                             "other.")
    parser.add_argument("--expansion-probe", action="append", default=[],
                        metavar="NAME=PATH",
                        help="install an expansion-probe licence in the environment "
                             "before the driver is built, e.g. "
                             "MEEP_GPU_COMPLEX_EXPANSION_PROBE=<artifact>.json. "
                             "Repeatable. The artifact must exist and its digest is "
                             "recorded in the output: which licence admitted an arm is "
                             "part of what a fused complex number means.")
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.backend == "numpy" and args.step_path == "fused":
        parser.error("--backend numpy is the NumPy reference and never dispatches; "
                     "use --backend gpu for --step-path fused (on an Apple GPU that "
                     "is the Metal table)")

    # THE LICENCES FIRST, BEFORE ANYTHING IMPORTS A COMPOSER OR BUILDS A DRIVER. Each
    # is recorded by NAME and by DIGEST rather than by "a probe was set": the arm a
    # licence admits is a measured platform fact, so two runs under different probe
    # artifacts are not the same measurement even where every other field matches.
    probes = {}
    for item in args.expansion_probe:
        name, _, path = item.partition("=")
        if not name or not path:
            parser.error(f"--expansion-probe wants NAME=PATH, got {item!r}")
        if not os.path.isfile(path):
            parser.error(f"--expansion-probe {name}: no such artifact: {path}")
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        os.environ[name] = path
        probes[name] = {"path": path, "sha256": digest}
        print(f"  licence {name} <- {path} ({digest[:12]})", flush=True)
    # THE ARRAY BASELINE IS PINNED BY THE TOOL, NOT BY WHOEVER CALLS IT. Dispatch is on
    # by default (fastpath.DISPATCH_BY_DEFAULT), so an "array" run left to the caller's
    # environment plans kernels on a certified host and require_step_path aborts it --
    # every timing driver would have had to learn a new export. Process-scoped is the
    # right scope for a CLI: nothing else runs in this process. A fused run keeps the
    # caller's enable (unset is the default, which dispatches), and both are recorded.
    dispatch_enable = "MEEP_GPU_DISPATCH"
    if args.step_path == "array":
        if os.environ.get(dispatch_enable) not in (None, "0"):
            print(f"  NOTE: {dispatch_enable}={os.environ[dispatch_enable]!r} in the "
                  f"calling environment is replaced by 0 for --step-path array", flush=True)
        os.environ[dispatch_enable] = "0"
    if args.step_path == "fused" and args.storage == "complex" and not probes:
        print("  NOTE: a complex-storage fused run with no --expansion-probe will be "
              "refused arm by arm; require_step_path aborts rather than mislabelling",
              flush=True)

    sweep = args.cells or [[64, 64, 64], [96, 96, 96], [128, 128, 128], [160, 160, 160]]
    prefer_gpu = args.backend == "gpu"
    # WHICH GPU "gpu" MEANT HERE: the same probe FdtdDriver resolves through, so the
    # output names CUDA or Metal rather than leaving it to the hostname.
    from meep_gpu import backends  # noqa: PLC0415

    resolved_gpu = backends.available_gpu() if prefer_gpu else None
    # WHERE THE TIMED STEPS RUN. "--backend gpu" names the request, and only on a
    # CUDA host does the engine itself live on the device: an Apple GPU driver keeps
    # NumPy host arrays, so its array path is the host CPU and only its dispatched
    # kernels are the GPU's.
    array_module = "cupy" if resolved_gpu == "cuda" else "numpy"
    if resolved_gpu == "cuda":
        timed_on = "cuda device"
    elif resolved_gpu == "metal" and args.step_path == "fused":
        timed_on = "apple gpu (dispatched slots) and host cpu (array slots)"
    else:
        timed_on = "host cpu"
    if prefer_gpu and timed_on == "host cpu":
        print(f"NOTE: --backend gpu resolved {resolved_gpu!r} and --step-path array on "
              "that host is NumPy on the host CPU; these are not GPU numbers",
              flush=True)
    results = []
    for cells in sweep:
        repeats = []
        for index in range(args.repeats):
            timings, grid_cells, checksum = one_repeat(
                args.profile, tuple(cells), args.resolution, prefer_gpu,
                args.pml_cells, args.warmup_steps, args.steps, args.step_path,
                args.storage)
            repeats.append(timings)
            print(f"  {tuple(cells)} rep{index}: stepping {timings['stepping_s']:.3f} s  "
                  f"{timings['mcells_per_s']:.1f} Mcells/s", flush=True)
        median = {key: statistics.median(r[key] for r in repeats) for key in repeats[0]}
        results.append({"cells": cells, "grid_cells": grid_cells,
                        "median": median, "repeats": repeats, "checksum": checksum})
        print(f"  {tuple(cells)} MEDIAN: {median['mcells_per_s']:.1f} Mcells/s  "
              f"stepping {median['stepping_s']:.3f} s", flush=True)

    payload = {
        "benchmark": "meep_gpu.fdtd.gpu",
        "backend": args.backend,
        "gpu": resolved_gpu,  # "cuda", "metal", or None for the NumPy reference.
        "array_module": array_module,  # What the engine's fields live in.
        "timed_on": timed_on,  # Never inferred from "backend": see above.
        "step_path": args.step_path,  # Asserted per repeat via driver.active_step_path.
        # The enable the drivers were built under: "0" is this tool's own pin for an
        # array baseline; None is the shipped default a fused run measures.
        "dispatch_enable": {dispatch_enable: os.environ.get(dispatch_enable)},
        "profile": args.profile,
        "storage": args.storage,
        "expansion_probes": probes,   # name -> {path, sha256}; empty is a claim too
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "python": sys.version.split()[0]},
        "parameters": {"resolution": args.resolution, "pml_cells": args.pml_cells,
                       "warmup_steps": args.warmup_steps, "measured_steps": args.steps,
                       "repeats": args.repeats},
        "sweep": results,
    }
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
