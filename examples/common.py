"""Shared by the example scripts: choose a leg, print what it produced, save it.

Every example builds one MEEP simulation and can run it four ways, called legs:

    gpu         run_on_gpu(sim, until=...)                     this host's GPU
    array       the same call with MEEP_GPU_DISPATCH=0         the array path of the
                                                               same GPU route
    reference   run_on_gpu(sim, until=..., prefer_gpu=False)   the NumPy reference
    meep        sim.run(until=...)                             MEEP itself

One leg runs per process. A GPU run that dispatches compiled kernels installs its
kernel table's float32 subnormal policy for the whole process, so a run made
afterwards in the same process would not be the run a fresh process makes.
``run_examples.py`` starts one process per leg and compares what they saved.

The subnormal policy is part of every comparison. Kernel results are
bit-identical to the array path under the policy the kernel table is certified
for; ``--policy`` installs a policy by name before the array or reference leg
is built, and ``run_examples.py`` passes the policy the GPU leg reported.

Output lines carry one of three prefixes:

    [result]   what the run computed; compared against examples/expected/
    [host]     what this host is and which path served; differs between hosts
    [timing]   wall-clock figures; differ between runs
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time

import numpy as np

LEGS = ("gpu", "array", "reference", "meep")
POLICIES = ("default", "flush", "keep")
DISPATCH_ENABLE = "MEEP_GPU_DISPATCH"


def parse_arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--leg", choices=LEGS, default="gpu",
                        help="gpu: this host's GPU (default). array: the array path "
                             "of the same GPU route, kernel dispatch off. reference: "
                             "the NumPy reference. meep: MEEP itself.")
    parser.add_argument("--policy", choices=POLICIES, default="default",
                        help="float32 subnormal policy installed before an array or "
                             "reference leg is built; default leaves the host's own")
    parser.add_argument("--resolution", type=int, default=None,
                        help="pixels per unit length; default is the example's own")
    parser.add_argument("--save", default=None, metavar="FILE.npz",
                        help="save the field and the spectra for a comparison")
    return parser.parse_args()


def describe_host(leg: str) -> None:
    """The [host] lines: what ran this leg. Never compared against expected output."""
    import meep as mp
    import meep_gpu

    print(f"[host] python {platform.python_version()} on {platform.system()} "
          f"{platform.machine()}")
    print(f"[host] meep {mp.__version__}, single precision: "
          f"{bool(mp.is_single_precision())}")
    print(f"[host] numpy {np.__version__}")
    if leg in ("gpu", "array"):
        print(f"[host] gpu: {meep_gpu.available_gpu()}  "
              f"missing: {list(meep_gpu.missing_dependencies())}")
    sys.stdout.flush()


def describe_path(driver) -> None:
    """Which path served the step, and under which subnormal policy.

    The step path is "fused" when compiled kernels served the step and "array"
    when the array path did. Printed, never asserted. An NVIDIA host outside the
    certified identities is refused by name and steps on the array path; that is
    a correct run, not a failure. On a Mac outside the certified environment the
    Metal kernels serve the step, and the run's NOTE line says they are not
    certified there.
    """
    import meep_gpu

    print(f"[host] driver.gpu: {driver.gpu}")
    print(f"[host] step path: {driver.active_step_path}")
    print(f"[host] subnormal policy: {meep_gpu.get_subnormal_policy()}")
    sys.stdout.flush()


def report(name: str, leg: str, *, cells: int, steps, meep_time: float, field_name: str,
           field, spectra, wall_time_s=None, save=None) -> None:
    """Print the [result] and [timing] lines for one leg, and save it if asked."""
    field = np.ascontiguousarray(np.asarray(field))
    spectra = [np.asarray(values, dtype=np.float64) for values in spectra]
    print(f"[result] example: {name}")
    print(f"[result] leg: {leg}")
    print(f"[result] cells: {cells}")
    if steps is not None:
        print(f"[result] steps: {steps}")
    print(f"[result] meep time: {meep_time:.6f}")
    print(f"[result] {field_name} shape: {tuple(field.shape)}")
    print(f"[result] {field_name} max abs: {float(np.abs(field).max()):.6e}")
    print(f"[result] {field_name} l2 norm: {float(np.linalg.norm(field.ravel())):.6e}")
    for index, values in enumerate(spectra):
        listed = " ".join(f"{value:.6e}" for value in values)
        print(f"[result] flux {index}: {listed}")
    print(f"[host] {field_name} dtype: {field.dtype}")
    print(f"[host] {field_name} sha256: "
          f"{hashlib.sha256(field.tobytes()).hexdigest()[:16]}")
    if wall_time_s is not None and steps:
        rate = cells * steps / wall_time_s / 1.0e6
        print(f"[timing] step loop: {wall_time_s:.3f} s, "
              f"{rate:.1f} million cell updates per second")
    sys.stdout.flush()
    if save:
        arrays = {"field": field}
        for index, values in enumerate(spectra):
            arrays[f"flux_{index}"] = values
        np.savez(save, **arrays)


def run_leg(name: str, build, *, field_name: str = "Ez") -> None:
    """Run the leg named on the command line on the simulation ``build`` returns.

    ``build(resolution)`` returns ``(sim, flux_monitors, until)``: an ordinary
    ``mp.Simulation``, the objects ``sim.add_flux`` returned, and the MEEP time
    to run to.
    """
    arguments = parse_arguments(f"{name}: one MEEP simulation, stepped four ways")
    if arguments.leg == "array":
        # The array path of the GPU route: the same request as the gpu leg, with
        # kernel dispatch switched off for this process before anything is planned.
        os.environ[DISPATCH_ENABLE] = "0"

    import meep as mp
    import meep_gpu
    from meep_gpu import gpu_compatibility, run_on_gpu

    describe_host(arguments.leg)
    if arguments.policy != "default" and arguments.leg in ("array", "reference"):
        if arguments.policy == "keep" and meep_gpu.available_gpu() == "cuda":
            # Under keep, CuPy needs a private kernel cache; dispatch points
            # CUPY_CACHE_DIR at one itself, an install by hand must do the same.
            from meep_gpu.subnormal_policy import point_cupy_cache_at_keep_policy
            point_cupy_cache_at_keep_policy()
        meep_gpu.set_subnormal_policy(arguments.policy)
    sim, monitors, until = build(arguments.resolution)

    if arguments.leg == "meep":
        started = time.perf_counter()
        sim.run(until=until)                        # MEEP itself
        wall = time.perf_counter() - started
        field = sim.get_array(component=getattr(mp, field_name))
        report(name, "meep", cells=int(np.asarray(field).size), steps=None,
               meep_time=sim.meep_time(), field_name=field_name, field=field,
               spectra=[mp.get_fluxes(monitor) for monitor in monitors],
               save=arguments.save)
        print(f"[timing] sim.run: {wall:.3f} s including initialization")
        return

    verdict = gpu_compatibility(sim)                # reads declarations only
    if not verdict.supported:
        print("[result] not supported by the lift:")
        for reason in verdict.reasons:
            print(f"[result]   - {reason}")
        raise SystemExit(2)

    prefer_gpu = arguments.leg in ("gpu", "array")
    result = run_on_gpu(sim, until=until, prefer_gpu=prefer_gpu)
    try:
        describe_path(result.driver)
        field = result.get_array(field_name)
        report(name, arguments.leg, cells=int(np.asarray(field).size),
               steps=result.steps, meep_time=result.meep_time, field_name=field_name,
               field=field,
               spectra=[result.get_flux_spectrum(monitor) for monitor in monitors],
               wall_time_s=result.wall_time_s, save=arguments.save)
    finally:
        result.close()                              # releases host and device arrays
