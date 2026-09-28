# Time your own simulation

The only comparison with MEEP that has been made is one case on one machine
([Will it help?](will-it-help.md)). The kernel mix, and so the speed, depends on
the configuration, so time your own problem before relying on a speed-up. This
page gives one script that times the same simulation three ways: on this host's
GPU, on the NumPy reference, and on MEEP itself, on one process or at the rank
count you normally use.

## What to measure

- **Steady-state stepping rate**, in million cell updates per second: cells ×
  steps / seconds, over a window that starts after a warm-up. The first steps
  include one-time work (kernel compilation on the GPU, MEEP's own first-step
  setup), so they are not timed.
- **The lift**, timed separately: it runs once, before the first step, and
  includes MEEP's own initialization.
- **What ran**: `driver.gpu` and `driver.active_step_path`. A rate from a run
  that took the array path is not a rate of the compiled kernels
  ([Reading what ran](../getting-started/reading-what-ran.md)).

Use the same simulation, the same cell count and the same monitors on every leg.

## The script

Save this as `time_case.py` and replace the body of `build()` with your own
simulation. As written it is the README's first example at resolution 16
(262,144 cells). The GPU and reference legs read the `Ez` field after each
window, which waits for the device to finish; if your simulation does not carry
`Ez`, read a component it does.

```python
"""Time one MEEP simulation on this host's GPU, on the NumPy reference, or on MEEP.

    python time_case.py gpu
    python time_case.py reference
    python time_case.py meep                  # MEEP on one process
    mpirun -np 8 python time_case.py meep     # MEEP on 8 ranks: an MPI build of MEEP only
"""

import sys
import time

import meep as mp

WARMUP_STEPS = 50
TIMED_STEPS = 200


def build():
    """Your simulation, built as your own script builds it, monitors included."""
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 4, 4),
        resolution=16,
        boundary_layers=[mp.PML(0.8)],
        geometry=[mp.Sphere(radius=0.8, material=mp.Medium(epsilon=9))],
        sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4),
                           component=mp.Ez, center=mp.Vector3(-1.2, 0, 0))],
    )
    sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(center=mp.Vector3(0.9, 0, 0),
                                            size=mp.Vector3(0, 2, 2)))
    return sim


def cell_count(sim):
    count = 1
    for extent in (sim.cell_size.x, sim.cell_size.y, sim.cell_size.z):
        count *= max(1, round(extent * sim.resolution))
    return count


def time_lifted(prefer_gpu):
    from meep_gpu import gpu_compatibility, lift_simulation

    sim = build()
    verdict = gpu_compatibility(sim)
    if not verdict.supported:
        sys.exit("not liftable: " + "; ".join(verdict.reasons))
    cells = cell_count(sim)
    start = time.perf_counter()
    driver = lift_simulation(sim, prefer_gpu=prefer_gpu)
    lift_s = time.perf_counter() - start
    try:
        driver.run(num_steps=WARMUP_STEPS)
        driver.get_field("Ez")                     # waits for the device
        start = time.perf_counter()
        driver.run(num_steps=TIMED_STEPS)
        driver.get_field("Ez")
        seconds = time.perf_counter() - start
        print(f"gpu={driver.gpu} step_path={driver.active_step_path} cells={cells} "
              f"lift={lift_s:.1f} s rate={cells * TIMED_STEPS / seconds / 1e6:.1f} "
              "Mcell-steps/s", flush=True)
    finally:
        driver.close()


def time_meep():
    sim = build()
    cells = cell_count(sim)
    start = time.perf_counter()
    sim.init_sim()
    init_s = time.perf_counter() - start
    for _ in range(WARMUP_STEPS):
        sim.fields.step()
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        sim.fields.step()
    seconds = time.perf_counter() - start
    if mp.am_master():
        print(f"meep ranks={mp.count_processors()} cells={cells} init={init_s:.1f} s "
              f"rate={cells * TIMED_STEPS / seconds / 1e6:.1f} Mcell-steps/s", flush=True)


if __name__ == "__main__":
    leg = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    if leg == "meep":
        time_meep()
    else:
        time_lifted(prefer_gpu=(leg == "gpu"))
```

Run each leg in its own process, the GPU and reference legs without `mpirun`:

```bash
python time_case.py gpu
python time_case.py reference
python time_case.py meep
```

The MEEP leg above runs MEEP on one process, and prints `ranks=1`. The three
environment files of INSTALL.md install a serial MEEP (the conda-forge `nompi`
build), which has no `mpirun` and runs on one process only; under an `mpirun`
from elsewhere, a serial MEEP starts independent copies of the whole simulation,
each printing its own line, and times nothing useful. To time MEEP at the rank
count you normally use, run the MEEP leg in the environment where you normally
run MEEP under MPI, with a MEEP built with MPI
([MPI](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#mpi)), and
check that it prints the rank count you asked for:

```bash
mpirun -np 8 python time_case.py meep      # an MPI build of MEEP; prints ranks=8
```

Each leg prints one line. On the certified Apple host, as written, the GPU leg reads
`gpu=metal step_path=fused cells=262144`, and the reference leg `gpu=None
step_path=array`. The rates depend on the host and on what else it is running;
repeat each leg two or three times and compare the medians.

## Reading the result

- **The GPU pays off** when its rate is above MEEP's at your rank count and the
  run is long enough to repay the lift. The break-even step count is
  `lift / (cells × (1 / MEEP rate − 1 / GPU rate))`, with rates in cell updates
  per second. The lift includes MEEP's own initialization, which the MEEP leg
  pays too (`init` in its line), so subtract that for a closer figure.
- **`step_path=array` on the GPU leg** means no compiled kernel served your
  configuration, or the host is outside the certified set; the line on standard
  error says which. That rate is the array path's.
- **The reference leg** is for checking results, not for speed. On a small grid
  it can be faster than the GPU leg
  ([Will it help?](will-it-help.md#small-grids-use-the-reference-or-meep-itself)).
- **Compare results as well as rates.** A default GPU run and a `prefer_gpu=False`
  run differ by about one part in 10^5 of the final field where values pass
  through the float32 subnormal range (8.4e-6 to 1.4e-5 on the examples;
  [the subnormal policy](../design/floating-point.md#the-subnormal-policy)), and a
  comparison with a double-precision MEEP compares single-precision stepping
  with double-precision stepping
  ([the floating-point contract](../design/floating-point.md)).

## The certification harness's timing driver

For a timing that is reported with the package's own checks (repeated windows, a
spread limit, bit identity against the array path, and which kernel table
served), the harness ships `parity/meep_gpu/bench_fused_products.py`. It times the
harness's own cases, not your simulation. On an Apple host pass
`--drive-table metal`; the default, `triton`, refuses a host with no CUDA device.
Expect it to take several minutes per case, longer than its `--estimate`, and
on a loaded machine it may report a row as not reportable. Follow
[Benchmark fairly](validation-and-performance.md#benchmark-fairly) when you
report a number.
