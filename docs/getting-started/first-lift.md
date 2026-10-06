# First lifted simulation

The first run is the README's: `examples/quickstart.py`. It builds an ordinary
MEEP simulation, asks whether it can be lifted, and steps it on this host's GPU.
Install first ([Installation](installation.md)), then, from the root of the
checkout:

```bash
python examples/quickstart.py
```

This is the script:

```python
import meep as mp
from meep_gpu import gpu_compatibility, run_on_gpu

sim = mp.Simulation(                              # your existing script
    cell_size=mp.Vector3(4, 4, 4),
    resolution=12,
    boundary_layers=[mp.PML(0.8)],
    geometry=[mp.Sphere(radius=0.8, material=mp.Medium(epsilon=9))],
    sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4),
                       component=mp.Ez, center=mp.Vector3(-1.2, 0, 0))],
)
flux = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(center=mp.Vector3(0.9, 0, 0),
                                               size=mp.Vector3(0, 2, 2)))

verdict = gpu_compatibility(sim)                  # supported, or the reasons why not
print("supported:", verdict.supported, list(verdict.reasons))

result = run_on_gpu(sim, until=35)                # instead of sim.run(until=35)
try:
    spectrum = result.get_flux_spectrum(flux)
    ez = result.get_array("Ez")
    print("steps:", result.steps, " MEEP time:", result.meep_time)
    print("step path:", result.driver.active_step_path)
    print("flux:", " ".join(f"{value:.6e}" for value in spectrum))
    print("max |Ez|:", f"{abs(ez).max():.6e}")
finally:
    result.close()                                # releases host and device arrays
```

Between MEEP's own setup lines it prints, on the certified Apple host in the
environment of Route 2 (MEEP 1.33.0 from conda-forge, a double-precision build):

```text
supported: True []
steps: 840  MEEP time: 35.0
step path: fused
flux: 9.709557e-06 7.775078e-02 -3.753959e-06
max |Ez|: 1.436076e-01
```

`step path: fused` means compiled kernels served the run; `step path: array`
means the array path did, which is a correct run. The last digits of `flux` and
`max |Ez|` depend on the MEEP build the lift reads from: the recorded output,
`examples/expected/quickstart.txt`, was made against a single-precision MEEP
built from source and reads `9.709987e-06 7.775076e-02 -3.754030e-06` and
`1.436078e-01`.

The package also prints to standard error, at most once per process for each
distinct message: with a double-precision MEEP a note on precision first, then
the line that says what served the run.
[Reading what ran](reading-what-ran.md) explains both.
`python examples/run_examples.py --check` repeats this run and three more
examples on every path your host has, and compares them with the recorded output
within a tolerance, not digit for digit.

Three things to take from the script:

- **The check runs before MEEP initializes the grid.**
  `gpu_compatibility(sim)` reads the declared simulation, so it is inexpensive
  enough to put in a batch preflight. A false verdict carries every reason.
- **`sim.run(...)` becomes `run_on_gpu(sim, ...)`**, and fields and monitor
  results are read from the result: `get_array` returns MEEP's field-array
  layout, and each MEEP monitor attached before the run has a reader named after
  MEEP's own ([Monitors and results](../guides/monitors-and-results.md)).
- **Always call `result.close()`** after copying out the data needed downstream.
  The result owns the lifted driver's arrays; on a CUDA device this releases
  them to the device allocator.

## The two run styles

The fragments below use a small one-dimensional simulation. Save it as
`first_lift.py` to run it; it steps 80 cells and shows the calls, not a speed-up:

```python
import meep as mp
from meep_gpu import gpu_compatibility, run_on_gpu


def build():
    return mp.Simulation(
        cell_size=mp.Vector3(0, 0, 4),
        dimensions=1,
        resolution=20,
        sources=[mp.Source(mp.GaussianSource(frequency=0.15, fwidth=0.1),
                           component=mp.Ex, center=mp.Vector3())],
    )


sim = build()
verdict = gpu_compatibility(sim)
if not verdict.supported:
    raise RuntimeError("\n".join(verdict.reasons))

result = run_on_gpu(
    sim,
    until=40,
    progress_cb=lambda completed, total: print(completed, "/", total, flush=True),
)
try:
    ex = result.get_array("Ex")
    print("steps:", result.steps)
    print("MEEP time:", result.meep_time)
    print("step-loop seconds:", result.wall_time_s)
finally:
    result.close()
```

The cell is one-dimensional along `z`, where MEEP carries the transverse
components `Ex` and `Hy`; asking a 1-D grid for `Ez` is refused by MEEP itself
when the grid is initialized.

<code>run_on_gpu</code> is the one-call path. Give exactly one stopping
condition, and give each call a newly constructed simulation (a lift refuses one
that MEEP has already initialized):

```python
result = run_on_gpu(build(), until=200)                 # MEEP's sim.run(until=200)
result.close()
result = run_on_gpu(build(), until_after_sources=50)    # 50 after the last source ends
result.close()
```

The first has the same meaning as MEEP's <code>sim.run(until=...)</code>. The
second ends after the last source time MEEP reports plus the requested interval.
For a continuous source with no end time, use <code>until</code> or set an end
time explicitly.

Use <code>lift_simulation</code> instead when the caller needs to add driver
monitors, run custom step functions, or control execution in segments:

```python
from meep_gpu import lift_simulation

driver = lift_simulation(build())
try:
    driver.run(until=200)
    ex = driver.get_field("Ex")
finally:
    driver.close()
```

## What <code>prefer_gpu</code> means

Both entry points run on this host's GPU unless told otherwise:
<code>lift_simulation</code> and <code>run_on_gpu</code> default to
<code>prefer_gpu=True</code>. On a host with a CUDA device that is CUDA through
CuPy. On a host with an Apple GPU it is Metal kernels over NumPy host arrays. A
host with neither raises, before MEEP initializes anything, and the message
names <code>prefer_gpu=False</code> as the way out.

<code>prefer_gpu=False</code> is the NumPy reference: host arrays and the array
path. It never runs a compiled kernel, whatever the environment says, so it is
the same run on every host and a CPU-only machine remains a valid reference host:

```python
driver = lift_simulation(build(), prefer_gpu=False)
driver.close()
```

The driver records what the request resolved to in <code>driver.gpu</code>:
<code>"cuda"</code>, <code>"metal"</code>, or <code>None</code> for the
reference. The choice is about time, not about the model: both step the same
equations. They are not always bit-identical: a run that dispatches compiled
kernels installs its kernel table's float32 subnormal policy for the whole
process, and the reference takes the host as it finds it, so the two can differ
in the last bits once values pass through the subnormal range
([the floating-point contract](../design/floating-point.md#the-subnormal-policy)).
On the Apple GPU measured, at 3,600 cells, a default lift and the reference
produced the same field in 11 of 11 cases compared, none of which reaches that
range.

## Small cells are faster on the reference

A GPU run is not automatically the faster one. The 1-D example on this page is
an 80-cell grid, far below every size at which a GPU run has been measured
ahead; it is here to show the calls, not to be fast.

On an Apple GPU a default lift costs one to two seconds before the second step
(importing the device library, then planning and compiling the kernels at the
first step), and each step then costs about 2 ms on a small cell whatever its
size. Measured on one M1 Max on 2026-09-27, on a loaded host and therefore not a
timing of record, in vacuum with an absorber on every side and one point source:

| Cells | Default lift against <code>prefer_gpu=False</code> |
|---|---|
| 3,600 (2-D) | 7.8 to 10.3 times slower |
| 40,000 (2-D) | 2.6 to 3.0 times slower |
| 102,400 (2-D) | 1.16 to 1.45 times slower |
| 230,400 (2-D) | 1.41 to 1.66 times faster |
| 409,600 (2-D) | 2.05 to 2.66 times faster |
| 125,000 (3-D) | level |
| 216,000 (3-D) | 1.61 times faster |

The crossover on that host is between 102,400 and 230,400 cells in 2-D and near
125,000 cells in 3-D. Below it, pass <code>prefer_gpu=False</code>, or set
<code>MEEP_GPU_DISPATCH=0</code> to reach the same array path without editing
the script. On a CUDA device the array path itself runs on the GPU. When to
expect MEEP itself to be the faster choice is in
[Will it help?](../guides/will-it-help.md).

## Compiled kernels run by default

Compiled-kernel dispatch is on by default for a driver built with
<code>prefer_gpu=True</code>, which a default lift is. On a supported NVIDIA
GPU (compute capability 7.0 to 9.0), and on any Apple GPU (every one is
supported), the example above runs compiled kernels on every sub-step a
certified kernel covers, and the array path on the rest; outside the certified
set an NVIDIA or Apple GPU runs the kernels uncertified, and says so
([Certification on an NVIDIA GPU](../guides/kernel-dispatch.md#certification-on-an-nvidia-gpu),
[Certification on an Apple GPU](../guides/kernel-dispatch.md#certification-on-an-apple-gpu)).
To force the array path for the whole run, set one environment variable before
it starts:

```bash
MEEP_GPU_DISPATCH=0 python examples/quickstart.py
```

The switch accepts only <code>0</code> and <code>1</code>. Any other value,
<code>false</code> included, is refused by name and the run takes the array path.
The environment can only turn kernels off. No value turns them on for a
<code>prefer_gpu=False</code> driver; setting <code>MEEP_GPU_DISPATCH=1</code>
there prints one line saying it does not apply.

On an Apple GPU the array path runs on the host CPU, so a configuration no
certified kernel covers is a CPU run, and the line on stderr says so.

Read what a run actually did rather than assuming — after a step,
<code>driver.active_step_path</code> is <code>"array"</code> or
<code>"fused"</code>, and <code>driver.fast_path_report()</code> gives the
outcome of each sub-step with its reason ([Reading what ran](reading-what-ran.md)).
Which kernel table can serve you is decided by your hardware, and the coverage
and timing figures carry their own limits; both are in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

## Preserve MEEP setup

During a lift, MEEP initializes the simulation, builds its grid, rasterizes
geometry, and applies native subpixel smoothing. The lift then reads the
component-resolved constitutive state needed by the stepper. It does not advance
MEEP fields, so the same simulation object can still be run on CPU MEEP
afterward for a comparison. A lifted run that dispatched kernels has installed
its kernel table's subnormal policy for the whole process (<code>flush</code> on
an Apple GPU, <code>keep</code> on a CUDA device). On the Apple GPU measured it
was still installed after <code>close()</code>, so a CPU MEEP run made afterward
in the same process runs under it. A <code>prefer_gpu=False</code> lift installs
nothing. See
[Subnormal policy](../guides/kernel-dispatch.md#subnormal-policy).
