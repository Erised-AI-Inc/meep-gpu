# MEEP user entry points

This package is for a MEEP script that already constructs an
<code>mp.Simulation</code>. Keep constructing that simulation in MEEP. The
accelerator does not ask for a second geometry description: it checks whether
the constructed simulation can be reproduced, lets MEEP initialize and
rasterize it, and then steps the compatible realized state.

For ordinary MEEP construction—geometry, media, source definitions, boundary
semantics, and analysis outside this time-step loop—use the
[MEEP documentation](https://meep.readthedocs.io/en/latest/). This page covers
the three package entry points for the acceleration boundary, and the run-loop
controls <code>run_on_gpu</code> accepts around them.

## Choose the entry point

| Need | Use | What it does | What it does not do |
|---|---|---|---|
| Decide whether a submitted or queued MEEP job is liftable | <code>gpu_compatibility(sim)</code> | Reads declarations without initializing or mutating MEEP | Build a grid, allocate a GPU array, or run a time step |
| Run a supported job in one call | <code>run_on_gpu(sim, ...)</code> | Lifts, steps, and returns fields and migrated-monitor results | Run on a host with no GPU unless <code>prefer_gpu=False</code> is passed: the default raises. On an Apple GPU a configuration no certified kernel covers still steps on the host CPU, announced on stderr and recorded, not raised |
| Add driver monitors, choose the backend deliberately, or run in segments | <code>lift_simulation(sim, ...)</code> | Returns an initialized <code>FdtdDriver</code> before its first step, on this host's GPU by default | Transfer arbitrary MEEP callbacks or unsupported state. Run on a host with no GPU unless <code>prefer_gpu=False</code> is passed: the default raises |

All three are imported from the package root:

    from meep_gpu import gpu_compatibility, lift_simulation, run_on_gpu

The compatibility result is the runtime authority. It is intentionally more
specific than a static feature list: a combination can be supported in one
configuration and refused in another, with every known blocker reported in
<code>reasons</code>.

## 1. Preflight before scheduling

Use <code>gpu_compatibility</code> in a batch dispatcher, a notebook, or a
service before committing hardware time. It is safe on a newly constructed
simulation: it does not call <code>sim.init_sim()</code>, allocate fields, or
change the simulation object.

One thing it does call. A <code>material_function</code>,
<code>epsilon_func</code>, or other callable material is **sampled** here, at a
bounded lattice of cell-interior points, to check that what it returns is a
permittivity and nothing else. That is your own function running, so it is not
free and it is not guaranteed side-effect-free. It is strictly fewer evaluations
than MEEP's own structure build will make of it, and there is no other way to
answer the question without building the structure.

    verdict = gpu_compatibility(sim)
    if not verdict.supported:
        print("This job remains a CPU-MEEP job:")
        print("\n".join(f"- {reason}" for reason in verdict.reasons))
    else:
        queue_for_gpu(sim)

Treat a refusal as physical-model information, not as a generic hardware
failure. The reported reason identifies the feature whose exact lifted
representation is not yet verified. Choose CPU MEEP explicitly if it is the
appropriate fallback; do not change the simulation into a nearby model merely
to force a GPU run.

## 2. One-call accelerated execution

A GPU run is not automatically faster, and both entry points run on the GPU by
default. For two-dimensional problems and small grids, expect MEEP itself or the
NumPy reference to be as fast or faster; what has been measured, per host, and
what has not, is in [Will it help?](../guides/will-it-help.md).
On the one Apple GPU measured, a default lift is slower than
<code>prefer_gpu=False</code> below a crossover between 102,400 and 230,400
cells in 2-D and near 125,000 cells in 3-D (2026-09-27, on a loaded host; not
a timing of record); see
[Small cells are faster on the reference](first-lift.md#small-cells-are-faster-on-the-reference).

<code>run_on_gpu</code> is the closest counterpart to a simple
<code>sim.run(...)</code>. It accepts exactly one MEEP-style stopping condition:
either an absolute simulation time through <code>until</code>, or an interval
after MEEP's own reported last-source time through
<code>until_after_sources</code>.

    result = run_on_gpu(
        sim,
        until_after_sources=50,
        progress_cb=lambda done, total: print(f"{done}/{total}", flush=True),
    )
    try:
        ez = result.get_array("Ez")
        print(result.steps, result.meep_time, result.wall_time_s)
    finally:
        result.close()

The default is <code>prefer_gpu=True</code>, which means this host's GPU: CUDA
through CuPy, or Metal on an Apple GPU. On a host with neither, the call raises
before MEEP initializes anything rather than falling back unnoticed, and the
message names <code>prefer_gpu=False</code>.

<code>prefer_gpu=False</code> is the NumPy reference on every host. It never
runs a compiled kernel, whatever <code>MEEP_GPU_DISPATCH</code> says. See
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

<code>result.get_array("Ez")</code> produces a host copy in MEEP's field-array
layout, suitable for a direct comparison with
<code>sim.get_array(component=mp.Ez)</code>. Always call
<code>result.close()</code>; it releases the driver-owned arrays and, on a CUDA
device, returns device memory to the allocator.

## 3. Controlled lift and driver execution

Use <code>lift_simulation</code> when the step loop needs more explicit control:
for example to attach driver-native monitors, run in time segments, choose a
nondefault CUDA device, or intentionally use the NumPy reference path.

    driver = lift_simulation(sim)
    try:
        driver.run(until=100)
        checkpoint_ez = driver.get_field("Ez")
        driver.run(until=200)
        final_ez = driver.get_field("Ez")
    finally:
        driver.close()

Like <code>run_on_gpu</code>, <code>lift_simulation</code> defaults to
<code>prefer_gpu=True</code>: this host's GPU (CUDA, or Metal on an Apple GPU).
It raises before <code>sim.init_sim()</code> if the host has neither, and the
message names <code>prefer_gpu=False</code>. <code>gpu_id</code> selects a CUDA
device; an Apple GPU has one device, so <code>gpu_id</code> must be
<code>0</code> there, and a nonzero value is refused by name.

    reference = lift_simulation(sim, prefer_gpu=False)

<code>prefer_gpu=False</code> is the NumPy reference. It is the same driver on
every host, it never dispatches a kernel, and it ignores <code>gpu_id</code>.
That keeps a CPU-only host useful for reference work without pretending that it
is a GPU execution. <code>driver.gpu</code> reads <code>"cuda"</code>,
<code>"metal"</code>, or <code>None</code> for the reference.

On a host with no GPU the host's refusal comes first: a simulation the lift
would also refuse for its own features reports those reasons only once
<code>prefer_gpu=False</code> is passed. <code>gpu_compatibility(sim)</code>
answers for the simulation on every host, and
<code>meep_gpu.is_available()</code> answers for the host.

All of <code>prefer_gpu</code>, <code>gpu_id</code> and <code>progress_cb</code>
are keyword-only on both <code>lift_simulation</code> and <code>run_on_gpu</code>;
only <code>sim</code>, <code>until</code> and <code>until_after_sources</code> may
be passed positionally.

The lift calls <code>sim.init_sim()</code>. MEEP therefore remains responsible
for grid construction, geometry rasterization, and native subpixel smoothing.
No MEEP field time step is taken, so the same initialized simulation can still
be run on CPU MEEP later as a comparison.

## Sources, monitors, and readback

Use normal MEEP source and monitor construction before the lift. For the
covered forms, the lift translates their specifications into the accelerated
time-step loop. Supported MEEP DFT monitor kinds are <code>DftFlux</code>,
<code>DftFields</code>, <code>DftNear2Far</code>, <code>DftForce</code>, and
<code>DftEnergy</code>; their frequency lists, regions, and decimation settings
are carried from the attached MEEP monitor. A force region is restricted to
MEEP's diagonal stress-tensor branch — see
[Capability coverage checklist](../guides/capability-coverage.md).

    flux = sim.add_flux(frequency, 0, 1, flux_region)
    result = run_on_gpu(sim, until=200)
    try:
        transmitted = result.get_flux_spectrum(flux)
    finally:
        result.close()

For near-to-far, call <code>result.load_near2far(sim, monitor)</code>, then use
MEEP's unchanged far-field evaluator. For DFT fields, use
<code>result.get_dft_region(monitor, component)</code>; it deliberately returns
the accumulated driver region rather than claiming MEEP's read-time shape
collapse. See [Monitors and results](../guides/monitors-and-results.md) for
the result contracts and [Capability coverage checklist](../guides/capability-coverage.md)
for boundaries on these forms.

## Controls around the run loop

<code>run_on_gpu</code> takes more than a stopping condition, and each control is
keyword-only:

- <code>prepare(driver)</code> runs after a successful lift and before the first
  step, for driver-native setup;
- <code>step_functions</code> accepts MEEP's step-function list in MEEP-style
  one- or two-argument callback form, so <code>mp.at_every</code>,
  <code>mp.at_beginning</code>, <code>mp.at_end</code>,
  <code>mp.after_time</code> and <code>mp.after_sources(mp.Harminv(...))</code>
  run unchanged. Anything the facade does not carry — the HDF5 writers,
  <code>set_materials</code>, <code>output_volume</code> — raises
  <code>StepFunctionNotHosted</code> rather than silently reading the original
  simulation, whose fields are initialized and never stepped. Catch it;
- <code>flux_data</code> and <code>minus_flux_data</code> load MEEP flux
  transforms before the first step, for two-run normalization;
- <code>progress_cb</code> with <code>progress_interval</code>, and
  <code>lift_progress_cb</code> for the separate structured-material lift, whose
  total is not a timestep count; and
- <code>cancel_check</code>, polled with progress, raising
  <code>FdtdCancelled</code> when it asks to stop.

## Direct driver is a separate API

<code>FdtdDriver</code> can be created directly without importing MEEP. That is
useful for a caller that owns a complete Yee-grid specification, but it is not
a MEEP-user entry point and does not by itself establish MEEP equivalence. If
MEEP geometry realization or native subpixel smoothing belongs to the problem
definition, use a lift.
