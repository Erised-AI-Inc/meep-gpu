# Public Python API

Import the documented surface from the package root:

```python
import meep_gpu
```

For a workflow-oriented explanation of the three MEEP lift calls, see
[MEEP user entry points](../getting-started/meep-entry-points.md).

## Capability

### is_available

Returns true exactly when a <code>prefer_gpu=True</code> request resolves on
this host: CuPy can be imported and a CUDA device is visible, or the host has an
Apple GPU (a Torch build with MPS, an MPS device and
<code>torch.mps.compile_shader</code>). It performs a live check rather than
caching the result, and it equals <code>available_gpu() is not None</code>.

It is a hardware probe. It says nothing about whether compiled-kernel dispatch
is enabled, whether the toolchain is a certified one, or which arms would serve.
Use <code>driver.fast_path_report()</code> for that.

### available_gpu

Returns the GPU a <code>prefer_gpu=True</code> request resolves to on this host:
<code>"cuda"</code>, <code>"metal"</code>, or <code>None</code>. CUDA is probed
first, so a host with both resolves <code>"cuda"</code>. It never raises.

### cupy_available

Returns true only when CuPy can be imported and a CUDA device is visible. It
answers the CUDA question specifically, for a caller that routes CUDA only.

### missing_dependencies

Returns an empty list exactly when <code>is_available()</code> is true.
Otherwise it names what this platform's GPU route lacks: on macOS the first
missing of <code>torch</code>, <code>mps-device</code> and
<code>torch.mps.compile_shader</code>; elsewhere <code>cupy</code> or
<code>cuda-device</code>. Use it to explain an unavailable accelerator to an
operator.

### resolve_backend(prefer_gpu=False, gpu_id=0)

Returns the pair <code>(xp, gpu)</code>: the array module and the GPU the
request resolved to.

| Request | Host | Returns |
|---|---|---|
| <code>prefer_gpu=False</code> | any | <code>(numpy, None)</code>, the NumPy reference |
| <code>prefer_gpu=True</code> | CUDA device | <code>(cupy, "cuda")</code>, with device <code>gpu_id</code> selected |
| <code>prefer_gpu=True</code> | Apple GPU | <code>(numpy, "metal")</code>; Metal kernels run over NumPy-owned fields |
| <code>prefer_gpu=True</code> | neither | raises <code>RuntimeError</code>; the message names <code>prefer_gpu=False</code> |

<code>gpu_id</code> selects a CUDA device. An Apple GPU has one device, so a
<code>gpu_id</code> other than <code>0</code> raises <code>ValueError</code>
there.

### to_numpy

Returns a host NumPy copy of a device array and passes a NumPy array through
unchanged. Use it when a caller holds a driver-owned array and needs host data
without assuming which array module owns it.

## MEEP lift

### gpu_compatibility(sim)

Returns <code>GpuCompatibility</code> without initializing or mutating the MEEP
simulation. Inspect <code>supported</code> and <code>reasons</code>; the object
is also truthy exactly when it is supported.

<code>GpuCompatibility</code> is a report rather than an exception. It is the
right object to preserve in a job record or a dispatcher decision: a false
verdict carries all known blockers, not only the first one encountered.

The report is about the simulation and reads the same on every host. It does not
say whether this host has a GPU, so <code>supported</code> does not by itself
mean a default lift succeeds here: pair it with <code>is_available()</code>.

### lift_simulation(sim, \*, prefer_gpu=True, gpu_id=0, progress_cb=None)

Initializes MEEP, reads the compatible realized state, and returns a ready
<code>FdtdDriver</code> without taking a timestep. It defaults to
<code>prefer_gpu=True</code>, as <code>run_on_gpu</code> does: the lift is onto
this host's GPU, CUDA through CuPy or Metal on an Apple GPU, where any
configuration no certified kernel covers steps the host CPU and says so.
<code>prefer_gpu=False</code> is the NumPy reference, the same driver on every
host, which never dispatches a compiled kernel whatever
<code>MEEP_GPU_DISPATCH</code> says. The returned driver's <code>gpu</code>
attribute reads <code>"cuda"</code>, <code>"metal"</code>, or <code>None</code>
for the reference. The optional progress callback receives completed and total
planes while structured material data is sampled.

Raises <code>MeepSimulationNotLiftable</code> with every known blocking reason.
With <code>prefer_gpu=True</code>, the default, it raises
<code>RuntimeError</code> on a host with no GPU, and <code>ValueError</code> for
a nonzero <code>gpu_id</code> on an Apple GPU, both before
<code>sim.init_sim()</code> is called. The <code>RuntimeError</code> names
<code>prefer_gpu=False</code> and says the GPU is the default of both entry
points. It is raised before the simulation's own features are checked, so on a
host with no GPU a simulation that is also not liftable reports its blocking
reasons only once <code>prefer_gpu=False</code> is passed.

It also raises <code>ImportError</code> when MEEP cannot be imported, and
<code>meep_gpu.dispersion.DispersionInstability</code> (a <code>ValueError</code>)
when a Lorentz or Drude pole is unstable at the simulation's time step. The
preflight does not test the poles, so <code>gpu_compatibility(sim)</code> can
report <code>supported</code> for a simulation whose lift then raises this; see
[Refusals a caller catches](#refusals-a-caller-catches).

All three optional parameters are keyword-only; only <code>sim</code> may be
passed positionally.

### run_on_gpu(sim, until=None, until_after_sources=None, \*, ...)

```text
run_on_gpu(sim, until=None, until_after_sources=None, *, prefer_gpu=True, gpu_id=0,
           prepare=None, flux_data=(), minus_flux_data=(), step_functions=(),
           progress_cb=None, lift_progress_cb=None, cancel_check=None,
           progress_interval=100) -> GpuRunResult
```

Lifts and steps the simulation. Give exactly one stopping condition. It defaults
to <code>prefer_gpu=True</code>, this host's GPU, as
<code>lift_simulation</code> does, and returns <code>GpuRunResult</code>. It
raises what <code>lift_simulation</code> raises, and, during the run, the
refusals listed under [Refusals a caller catches](#refusals-a-caller-catches).

Only <code>sim</code>, <code>until</code> and <code>until_after_sources</code>
may be passed positionally. Every control in the table below is keyword-only.

The result records:

| Attribute or method | Meaning |
|---|---|
| <code>steps</code> | Number of timesteps taken by this call |
| <code>meep_time</code> | Reached MEEP simulation time |
| <code>wall_time_s</code> | Wall time inside the driver run loop |
| <code>get_array(component)</code> | Host array in MEEP field-array layout |
| <code>get_field(component, cell_centered=True)</code> | Host array in the driver's cell-centred layout; the argument selects that centring and is not the only behaviour |
| <code>driver</code> | The stepped <code>FdtdDriver</code> this result owns — the route to <code>active_step_path</code> and <code>fast_path_report()</code> |
| <code>get_flux_spectrum(meep_flux)</code> | Spectrum for a migrated MEEP flux monitor |
| <code>get_flux_data(meep_flux)</code> | Accumulated transform for a migrated MEEP flux monitor, MEEP's <code>sim.get_flux_data</code> — hand it back through <code>run_on_gpu(minus_flux_data=...)</code> |
| <code>get_dft_region(meep_fields, component, freq_index=0)</code> | Driver-accumulated DFT region. The first argument is the MEEP DFT-fields object, not a driver monitor |
| <code>get_dft_array(meep_fields, component, freq_index=0)</code> | That region after MEEP's read-time collapse — what <code>sim.get_dft_array</code> returns, shape included |
| <code>get_forces(meep_force)</code> | Force spectrum for a migrated MEEP force monitor, MEEP's <code>mp.get_forces</code> |
| <code>get_force_data(meep_force)</code>, <code>load_force_data(meep_force, data)</code> | MEEP's force-transform save and load |
| <code>get_electric_energy</code>, <code>get_magnetic_energy</code>, <code>get_total_energy</code> | Energy readbacks for a migrated MEEP energy monitor |
| <code>monitor_for(meep_monitor)</code> | The driver-side monitor object; raises for one that was not migrated |
| <code>sigma_lift</code> | The <code>StructuredSigmaLift</code> recording which per-point susceptibility route a structured dispersive lift used |
| <code>pack_flux_data(mp, meep_flux)</code> | This run's flux transform as MEEP's <code>FluxData</code>, for <code>sim.load_flux_data</code> and MEEP's own <code>get_eigenmode_coefficients</code> |
| <code>load_near2far(sim, monitor)</code> | Load migrated near-surface data into MEEP for MEEP far-field evaluation |
| <code>close()</code> | Release driver arrays; required after a GPU run. On a CUDA device it also returns device memory to the allocator |

The optional controls are intentionally explicit:

| Argument | Contract |
|---|---|
| <code>prefer_gpu</code>, <code>gpu_id</code> | <code>prefer_gpu=True</code> (the default) requests this host's GPU: CUDA through CuPy, or Metal on an Apple GPU, where any configuration no certified kernel covers steps the host CPU and says so. It raises on a host with neither. <code>gpu_id</code> selects a CUDA device and must be <code>0</code> on an Apple GPU. <code>prefer_gpu=False</code> is the NumPy reference, which never dispatches. |
| <code>prepare(driver)</code> | Runs after a successful lift and before the first step. Use for driver-native setup in addition to any compatible MEEP monitors already attached to the simulation. |
| <code>flux_data</code>, <code>minus_flux_data</code> | <code>[(meep_flux, data), ...]</code> pairs, applied after the lift and before the first step. These are MEEP's <code>sim.load_flux_data</code> and <code>sim.load_minus_flux_data</code>; the minus form loads and negates, so the run ends at <code>E2 - E1</code> and <code>H2 - H1</code>. A sequence of pairs rather than a dict, because MEEP's monitor objects are not hashable in every release. |
| <code>step_functions</code> | MEEP's step-function list, in MEEP-style one- or two-argument callback form. The object passed is a facade over the stepped driver that answers the <code>mp.Simulation</code> surface MEEP's own wrappers use — <code>round_time()</code>, <code>meep_time()</code>, <code>fields.dt</code>, <code>fields.t</code>, <code>fields.last_source_time()</code>, <code>sources</code>, <code>run_index</code>, <code>get_field_point()</code> — so <code>mp.after_sources(mp.Harminv(...))</code>, <code>mp.at_every</code>, <code>mp.at_beginning</code>, <code>mp.at_end</code> and <code>mp.after_time</code> run unchanged. Every other attribute falls through to the driver, which is what keeps a driver-native step function working. Anything neither carries — the HDF5 writers, <code>set_materials</code>, <code>output_volume</code> — raises <code>StepFunctionNotHosted</code> rather than reading the original <code>mp.Simulation</code>, whose fields are initialized and never stepped. |
| <code>progress_cb</code>, <code>progress_interval</code> | Receives completed and total steps every <code>progress_interval</code> steps (default 100). The package does not print progress on its own. |
| <code>lift_progress_cb</code> | Receives completed and total sampled planes while a structured material is lifted; it is separate because its total is not a timestep count. |
| <code>cancel_check</code> | Is polled with progress and raises <code>FdtdCancelled</code> when it asks to stop. |

<code>GpuRunResult.monitors</code> maps the identity of each migrated MEEP
monitor to its driver counterpart. It is a <code>MigratedMonitors</code>, a
read-only mapping that also OWNS the MEEP monitor objects: identity is what the
lookup compares, and holding the originals is what keeps a later run's monitor
from being allocated at a collected earlier one's address and resolving there.
Use <code>monitor_for(meep_monitor)</code>
when that object is required, and <code>sigma_lift</code> to record which
per-point susceptibility route was used for a structured dispersive lift.
Neither property should be inferred from a field shape or a numerical result.

<code>prepare</code> receives the lifted driver before the first step, so a
driver-native monitor can be added beside the migrated MEEP ones. The driver's
monitor methods are listed under [Driver methods](#driver-methods). With a
lifted MEEP simulation <code>sim</code>:

```python
monitors = {}

def prepare(driver):
    monitors["flux"] = driver.add_flux_monitor(
        fcen=0.4, df=0.4, nfreq=3, center=(0.9, 0.0, 0.0), size=(0.0, 2.0, 2.0))

result = run_on_gpu(sim, until=35, prepare=prepare)
try:
    spectrum = monitors["flux"].get_flux_spectrum()
finally:
    result.close()
```

## Refusals a caller catches

| Exception | Raised when | What to do |
|---|---|---|
| <code>MeepSimulationNotLiftable</code> (a <code>ValueError</code>) | The declared-feature check or the realized-grid check finds a blocker; it carries every known reason, not only the first | Each reason names the object and the way out, where one exists ([Compatibility and refusals](../guides/compatibility.md)). Run a refused simulation on MEEP |
| <code>meep_gpu.dispersion.DispersionInstability</code> (a <code>ValueError</code>; not in <code>__all__</code>) | At the lift: a Lorentz or Drude pole is unstable at this time step (MEEP's rule is <code>f_n < 1/(pi*dt)</code>). The preflight does not run this test, so <code>gpu_compatibility</code> can report <code>supported</code> first | Raise the resolution, lower the Courant factor, or refit the material with a lower-frequency model |
| <code>RuntimeError</code>, message beginning <code>GPU backend requested but unavailable:</code> | A <code>prefer_gpu=True</code> call on a host with no usable GPU; raised before MEEP initializes anything | Pass <code>prefer_gpu=False</code>, or install the missing piece the message names ([Troubleshooting](../guides/troubleshooting.md)) |
| <code>ValueError</code> for <code>gpu_id</code> | A <code>gpu_id</code> other than 0 on an Apple GPU | Pass <code>gpu_id=0</code> |
| <code>ValueError</code> for the stopping condition | <code>run_on_gpu</code> was given neither or both of <code>until</code> and <code>until_after_sources</code> | Give exactly one |
| <code>ImportError</code> | MEEP cannot be imported in this environment | Install MEEP as [Installation](../getting-started/installation.md) describes |
| <code>RuntimeError</code>, message beginning <code>This thread is flushing subnormal floating-point values to zero</code> | On a CUDA host, CuPy's kernel compiler could not be guarded against the subnormal flushing MEEP switches on | Call <code>meep.set_zero_subnormals(False)</code> before the first GPU kernel compiles |
| <code>StepFunctionNotHosted</code> | A MEEP step function reaches an attribute the run-loop facade does not carry — the HDF5 writers, <code>set_materials</code>, <code>output_volume</code>. It refuses rather than silently reading the original simulation, whose fields are initialized and never stepped | Drop that step function from the GPU run, or run the simulation on MEEP |
| <code>FdtdCancelled</code> | <code>cancel_check</code> asked the run to stop | Nothing: the caller asked for it |
| <code>FdtdDivergence</code> (a <code>RuntimeError</code>) | A dispersive run's field energy grew past every physical explanation. Raised mid-run, for the instabilities the setup-time closed-form pole test cannot see — a susceptibility interacting with an absorber, a conductivity, or a boundary | Raise the resolution, lower the Courant factor, or refit the material |
| <code>FdtdNonlinearityOutOfRange</code> (a <code>RuntimeError</code>) | A chi2/chi3 run's **field** left the range MEEP's Padé approximant describes. Raised mid-run, because the quantity that leaves the range is the field and not the material. Nothing diverges: past the pole the recovered E is finite, smooth, plausible and of the wrong sign | Lower the source amplitude or chi2/chi3, or raise the permittivity |
| <code>SubnormalPolicyLocked</code> | A different subnormal policy was requested after one was installed in this process | Set the policy before the first kernel compiles, or use a fresh process |
| <code>SubnormalPolicyUnattainable</code> | The requested policy cannot be installed on one of the executors | <code>keep</code> is not attainable on an Apple GPU; <code>flush</code> with CuPy needs <code>CUPY_ACCELERATORS=''</code> set before CuPy is imported ([the floating-point contract](../design/floating-point.md#the-subnormal-policy)) |

Not every refusal is an exception. A kernel table that cannot serve a
configuration, a device or toolchain outside the certified list, and an
environment switch with a value other than <code>0</code> or <code>1</code>
leave the run on the array path, print one line on standard error, and record
the reason in <code>driver.fast_path_report()</code>; see
[Messages on standard error](#messages-on-standard-error).

## Subnormal policy

Float32 subnormal handling is part of the numerical contract, not a tuning knob.
It affects reproducibility against CPU MEEP, and each compiled-kernel table
declares the policy its certification was cut under: the two NVIDIA tables
(Triton and hand-written CUDA) under <code>keep</code>, the Metal table under
<code>flush</code>. When dispatch is
enabled it installs that policy on all three executors or refuses by name.
Dispatch is on by default. So on a certified host a run that dispatches gets that
policy for the whole process, including its array-path sub-steps, unless it sets
<code>MEEP_GPU_DISPATCH=0</code>. The policy per executor, and what it does to
a comparison, is in
[the floating-point contract](../design/floating-point.md).

| Name | Meaning |
|---|---|
| <code>FLUSH</code>, <code>KEEP</code>, <code>MATCH_MEEP</code> | The requestable policies. <code>MATCH_MEEP</code> is the default and is answered by measuring the host, not by a hardcoded choice |
| <code>default_policy()</code>, <code>resolve_match_meep()</code> | The default request, and the measurement that resolves it on this host |
| <code>install_subnormal_policy(policy=None, *, cupy=None, strict=True, executors=("host", "cupy", "triton"))</code> | Drives the named executors to a policy and returns its stamp; with <code>strict</code>, an executor that cannot attain it raises <code>SubnormalPolicyUnattainable</code> |
| <code>get_subnormal_policy()</code> | The resolved policy, <code>"flush"</code> or <code>"keep"</code> (never <code>"match_meep"</code>): the installed one once a policy is installed, otherwise what the request resolves to on this host |
| <code>set_subnormal_policy(policy=None, *, strict=True)</code> | Installs the policy on the executors and returns the stamp. Call it before the first kernel compiles: a different policy after an install raises <code>SubnormalPolicyLocked</code> |
| <code>policy_stamp()</code> | What is actually installed, for a run record. It is a certification, not a preference, and reads as uninstalled until something drives an executor |

Two operational notes. <code>MEEP_GPU_SUBNORMAL_INSTALL=0</code> moves the
install to the caller but does not relax the rule — dispatch still refuses unless
the certified policy is already installed. And under <code>keep</code>, the
<code>CUPY_CACHE_DIR</code> in force must carry <code>ftz_stripped</code> in its
path, or the run silently executes the other policy's arithmetic.

## Reading what a run dispatched

| Member | Meaning |
|---|---|
| <code>driver.active_step_path</code> | <code>"array"</code> or <code>"fused"</code>, where <code>"fused"</code> means at least one sub-step of the frozen configuration will launch a kernel. A property, and it must be read **after** a step: the plan is built at the configuration freeze inside <code>step()</code> |
| <code>driver.fast_path_report()</code> | This driver's own freeze — the path, the per-slot outcome with its reason, and the certification each dispatched family rides on. Present whether the plan dispatched or was refused; a refusal carries the single named reason. <code>None</code> before the first <code>step()</code> |

<code>MEEP_GPU_DISPATCH_LOG</code> appends the same record, one JSON object per
configuration freeze, so a long run's dispatch state is readable with
<code>tail</code>. See
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

The record is a dictionary of more than twenty keys. The ones to read first are
<code>step_path</code>, <code>decision</code>, <code>refused_because</code>,
<code>table</code>, <code>composition</code>, <code>slots</code>,
<code>certified</code> (three-valued: <code>True</code>, <code>False</code>
under <code>MEEP_GPU_ALLOW_UNCERTIFIED=1</code>, <code>None</code> when nothing
dispatched or an identity could not be read) and <code>reference_driver</code>;
[Reading what ran](../getting-started/reading-what-ran.md) describes each, and
why <code>tables</code> and <code>arbitration</code> can read as not reached on a
dispatched run.

### Messages on standard error

A run prints at most one line per process for each distinct message. None of
them is an exception; each reason is also in the dispatch record.

| Line begins | Meaning |
|---|---|
| <code>meep_gpu: step path fused;</code> | Compiled kernels served part of the step: which sub-steps, which kernels, the table, the subnormal policy installed, and the certified identity |
| <code>meep_gpu: step path array on the host CPU; dispatch refused:</code> | On an Apple GPU no kernel could serve; the whole run stepped on the host CPU |
| <code>meep_gpu: step path array; dispatch refused:</code> | On an NVIDIA GPU no kernel could serve; the run stepped on the array path on the device |
| <code>meep_gpu: NOTE the kernels are NOT CERTIFIED on this host:</code> | <code>MEEP_GPU_ALLOW_UNCERTIFIED=1</code> admitted an identity outside the certified list |
| <code>meep_gpu: step path array (NumPy reference, prefer_gpu=False);</code> | <code>MEEP_GPU_DISPATCH</code> is set on a reference driver, where it does not apply |
| <code>meep_gpu: this MEEP build is double precision</code> | The lift read a double-precision MEEP; the engine steps single precision |

The environment switches that decide what dispatches are listed in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md#environment-switches).

## Direct driver

### FdtdDriver

The direct solver owns a Yee grid, D/B-primary fields, optional PML state,
sources, monitors, and the MEEP-ordered timestep loop. Build it directly only
when the caller owns the numerical setup instead of lifting a MEEP simulation.

Its direct configuration surface includes:

- material setters, including scalar and component-resolved permittivity;
- source, DFT, flux, and near-to-far accumulation methods;
- <code>step()</code> and <code>run()</code>;
- field and permittivity readback; and
- <code>close()</code>.

Direct construction does not imply MEEP equivalence. Use the lift path when
MEEP geometry realization and smoothing are part of the problem definition.

```text
FdtdDriver(cell_size, resolution, courant=0.5, force_complex_fields=True, symmetry=(),
           k_point=(0.0, 0.0, 0.0), boundaries=None, dimensions=3, cylindrical=False,
           m=0, accurate_fields_near_cylorigin=False, beta=0.0,
           bfast_scaled_k=(0.0, 0.0, 0.0), prefer_gpu=False, gpu_id=0)
```

| Parameter | Meaning |
|---|---|
| <code>cell_size</code>, <code>resolution</code> | The cell (Lx, Ly, Lz) in MEEP length units, and grid points per unit length |
| <code>courant</code> | The Courant factor; <code>dt = courant / resolution</code> |
| <code>force_complex_fields</code> | <code>True</code> (the default) stores complex64 fields, <code>False</code> float32. **The opposite of MEEP's default**; complex storage doubles the field memory |
| <code>symmetry</code> | Mirror planes: <code>Mirror</code> objects, or a bare <code>'X'</code>, <code>'Y'</code> or <code>'Z'</code> for an even plane |
| <code>k_point</code> | Bloch wavevector in MEEP's units; all zeros is plain periodicity |
| <code>boundaries</code> | <code>"periodic"</code> or <code>"metallic"</code>, for all axes or per axis. <code>None</code>, the default, is periodic everywhere: **the opposite of MEEP**, whose faces are metallic unless a <code>k_point</code> makes them periodic |
| <code>dimensions</code> | 3 (the default), 2 or 1, as MEEP's <code>dimensions</code>: the missing axes are translationally invariant |
| <code>cylindrical</code>, <code>m</code>, <code>accurate_fields_near_cylorigin</code> | A cylindrical cell, its azimuthal mode number, and MEEP's near-axis option |
| <code>beta</code> | MEEP's out-of-plane wavevector for a 2-D cell (<code>kz_2d</code>) |
| <code>bfast_scaled_k</code> | MEEP's broadband fixed-angle technique (<code>bfast_scaled_k</code>) |
| <code>prefer_gpu</code>, <code>gpu_id</code> | As for the lift, with the opposite default (below) |

The engine class takes the same <code>prefer_gpu</code> and
<code>gpu_id</code> arguments as the lift, with the opposite default. The engine class defaults to
<code>prefer_gpu=False</code>, the NumPy reference, which never dispatches a
compiled kernel whatever the environment says; the two MEEP entry points default
to <code>prefer_gpu=True</code>. <code>prefer_gpu=True</code> runs on this
host's GPU and raises on a host with none. Two attributes record the outcome:

| Attribute | Meaning |
|---|---|
| <code>xp</code> | The array module the engine's fields live in: NumPy or CuPy |
| <code>gpu</code> | What <code>prefer_gpu</code> resolved to: <code>"cuda"</code> (CuPy arrays on a CUDA device), <code>"metal"</code> (Metal kernels over NumPy host arrays on an Apple GPU), or <code>None</code> for the NumPy reference |

The arrays on <code>driver.fields</code> are internal storage, not part of this
surface: read fields with <code>get_field</code> and write a primary field with
<code>set_field</code>. On a driver built with <code>prefer_gpu=True</code> on an
Apple host, the Metal table holds those arrays on the device by default once a
configuration dispatches, so a direct in-place write raises and a reference kept
across a step reads stale values. A <code>prefer_gpu=False</code> driver holds
nothing on any device. See
[Compiled-kernel dispatch](../guides/kernel-dispatch.md#the-metal-table-and-held-residency).

### Driver methods

The methods a direct caller, or a <code>prepare(driver)</code> callback, uses
most. Each has a docstring (<code>help(meep_gpu.FdtdDriver.run)</code>).

| Method | Does |
|---|---|
| <code>run(num_steps=None, until=None, ..., *, step_functions=(), until_after_sources=None, ...)</code> | Steps to one stopping condition and returns the elapsed wall time. <code>progress_interval</code> defaults to 100 |
| <code>step()</code> | One time step |
| <code>get_field(component, cell_centered=True)</code>, <code>get_field_point(component, point)</code> | Read a field component, as an array or at one point |
| <code>set_field(component, array)</code> | Write a primary field component |
| <code>add_source(source_data)</code> | Add a source described by a mapping (<code>component</code>, <code>center</code>, <code>size</code>, <code>amplitude</code>, <code>source_type</code> and the waveform's keys; see its docstring) |
| <code>set_epsilon(eps)</code>, <code>set_epsilon_components(...)</code>, <code>set_conductivity(...)</code>, <code>add_susceptibility(...)</code> | Set materials |
| <code>setup_pml(thickness_cells, order=2)</code> | A PML of the given thickness in cells, uniform or per axis and side, graded as <code>u**order</code> |
| <code>set_absorber(...)</code> | <code>mp.Absorber</code> faces, from <code>AbsorberLayer</code> objects |
| <code>add_flux_monitor(frequencies=None, center=None, size=None, direction=None, *, fcen=None, df=None, nfreq=None, ...)</code> | A flux plane; read it with the returned monitor's <code>get_flux_spectrum()</code> |
| <code>add_dft_monitor(...)</code>, <code>add_energy_monitor(...)</code>, <code>add_force_monitor(...)</code> | DFT field, energy and force monitors; frequencies are spelled as for the flux monitor |
| <code>get_epsilon(frequency=None, component="Ez")</code> | Read the permittivity |
| <code>reset()</code>, <code>close()</code> | Clear the fields, the monitors, the sources' state and the clock; release the arrays |

Properties: <code>shape</code> (the stored grid, halved on a mirror axis),
<code>dt</code>, <code>dx</code>, <code>meep_time</code>,
<code>active_step_path</code>.

### Mirror

```text
Mirror(axis, phase=1)
```

<code>Mirror</code> represents a MEEP-style mirror symmetry plane and declared
phase for direct-driver construction. <code>axis</code> is the string
<code>'X'</code>, <code>'Y'</code> or <code>'Z'</code>, where
<code>mp.Mirror</code> takes <code>mp.X</code>; <code>phase</code> is +1 or −1.
Any other value raises <code>ValueError</code>. Symmetry changes the stored domain and
source/monitor legality; do not add it as a memory optimization without a
parity case for the physical configuration.

### AbsorberLayer

```text
AbsorberLayer(thickness, axis, side, R_asymptotic=1e-15)
```

<code>axis</code> is the integer 0, 1 or 2 (x, y, z) and <code>side</code> is
<code>"low"</code> or <code>"high"</code>; any other value raises
<code>ValueError</code>. <code>AbsorberLayer</code> is one <code>mp.Absorber</code> face — a thickness in
length units on one side of one axis — for <code>FdtdDriver.set_absorber()</code>.
An absorber is a graded scalar conductivity on <b>both</b> the electric and the
magnetic update, not a perfectly matched layer, and it installs no boundary
condition: the face keeps whatever wall it had. A lifted <code>mp.Simulation</code>
resolves its own faces, so this type is for direct-driver construction only.

## Other public utilities

<code>meep_gpu.__version__</code> is the version of the installed
distribution, read from its package metadata on first use. A checkout
that was not installed that way (never installed, or installed in editable
mode) reads the version in <code>pyproject.toml</code>.

<code>driver.lift_record</code>, and <code>GpuRunResult.lift_record</code>
after <code>run_on_gpu</code>, holds what a lift read from MEEP: its version,
its precision (<code>"single"</code> or <code>"double"</code>) and the
precision the engine steps in. A lift from a double-precision MEEP prints one
line per process saying that the engine steps single precision.

<code>Harminv</code>, <code>Mode</code>, and <code>do_harminv</code> expose the
package's harmonic-inversion (filter-diagonalization) interface, named exactly as
MEEP names it so a MEEP script reads the same here:

```text
Harminv(c=None, pt=None, fcen=None, df=None, mxbands=None)
do_harminv(signal, dt, fmin, fmax, mxbands=100, spectral_density=1.1, Q_thresh=50.0,
           rel_err_thresh=inf, err_thresh=0.01, rel_amp_thresh=-1.0, amp_thresh=-1.0,
           nf=None, rank_tol=1e-09) -> list of (freq, amp, err)
Mode(freq, decay, Q, amp, err)
```

<code>Harminv</code> takes MEEP's arguments: the component, the point, the
centre frequency, the width and the maximum number of modes.
<code>do_harminv</code> returns a list of <code>(freq, amp, err)</code> tuples
with complex <code>freq</code> and <code>amp</code>, and raises
<code>ValueError</code> for an argument outside its range. <code>Mode</code> is
the named tuple of MEEP's <code>mp.Harminv</code> result. <code>Harminv</code> is the
step-function object — hand it to a run loop and read <code>h.modes</code> at the
end, with sources off while it collects. It touches no field array and no stepper
state, so it is identical on CPU and GPU by construction. The lowercase
<code>harminv</code> function is deliberately not re-exported from the package
root, because it would shadow the <code>meep_gpu.harminv</code> module.

<code>StructuredSigmaLift</code> reports the route used when a structured
dispersive lift needs per-point susceptibility data. Its fields are
<code>route</code> (<code>"reader"</code>, <code>"declared-geometry lookup"</code>
or <code>"chi1inv inversion"</code>), <code>term_count</code>,
<code>verified_points</code>, <code>verification_frequencies</code>,
<code>worst_residual</code>, <code>tolerance</code> and
<code>fallback_reason</code>. <code>MigratedMonitors</code>
is the read-only mapping <code>GpuRunResult.monitors</code> returns. Use these
types as diagnostics; they do not expand the set of simulations a lift can
accept.
