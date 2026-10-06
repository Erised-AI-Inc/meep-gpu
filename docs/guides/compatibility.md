# Compatibility and refusals

This page says what lifts and what is refused. A refusal is always by name: the
package reports the feature it cannot reproduce and steps nothing, rather than
running a nearby model.

## Ask before building

<code>gpu_compatibility(sim)</code> inspects a constructed MEEP simulation
without initializing MEEP, allocating a grid, or mutating the simulation. It does
make one call into your own code: a <code>material_function</code>,
<code>epsilon_func</code>, or other callable material is sampled at a bounded
lattice of cell-interior points to check that it returns a permittivity and
nothing else. That is your function running, so it is not free and not guaranteed
side-effect-free — it is still strictly fewer evaluations than MEEP's own
structure build makes.

Its result has two fields:

| Field | Meaning |
|---|---|
| <code>supported</code> | True only when the declared simulation has no known lift blocker |
| <code>reasons</code> | Every named blocker found during the preflight |

    verdict = gpu_compatibility(sim)
    if verdict:
        submit(sim)
    else:
        print("\n".join(verdict.reasons))

<code>lift_simulation</code> runs the same declared-feature check and then
checks the realized MEEP grid before any time step. It raises
<code>MeepSimulationNotLiftable</code> if either check finds a mismatch.

One check runs only at the lift. A Lorentz or Drude pole that is unstable at the
simulation's time step raises <code>meep_gpu.dispersion.DispersionInstability</code>
(a <code>ValueError</code>, not a <code>MeepSimulationNotLiftable</code>) when the
lift installs the material, although the preflight reported the simulation as
supported. The message states MEEP's rule, <code>f_n < 1/(pi*dt)</code>: raise the
resolution, lower the Courant factor, or refit the material
([Troubleshooting](troubleshooting.md#when-the-lift-starts-an-unstable-dispersive-pole)).

**The verdict is about the simulation, not the host.** It reads the same on
every machine, with or without a GPU. A default lift also needs a GPU, which is
a separate question with its own answer:

    import meep_gpu

    verdict = gpu_compatibility(sim)
    if verdict and meep_gpu.is_available():
        driver = lift_simulation(sim)                      # this host's GPU
    elif verdict:
        driver = lift_simulation(sim, prefer_gpu=False)    # the NumPy reference

On a host with no GPU a default lift raises the host's refusal first, before
the simulation's features are checked, so an unsupported simulation reports its
own reasons there only through <code>gpu_compatibility</code> or a
<code>prefer_gpu=False</code> lift.

## What lifts

The preflight result is the authority for a particular simulation. This table
is the summary; the
[capability coverage checklist](capability-coverage.md) has the
boundary of each area.

| Area | Lifted |
|---|---|
| Coordinates | Cartesian 1-D, 2-D and 3-D cells; cylindrical cells with an integer <code>m</code> |
| Media | Isotropic and diagonal permittivity, uniform or structured, with MEEP's subpixel smoothing including its off-diagonal terms; electric and magnetic conductivity; Lorentz and Drude susceptibilities; instantaneous χ2 and χ3 on the electric side |
| Material routes | Geometry objects, <code>mp.MaterialGrid</code>, <code>material_function</code>, <code>epsilon_func</code>, an ndarray, and <code>epsilon_input_file</code> |
| Boundaries | <code>mp.PML</code> with MEEP's default profile, <code>mp.Absorber</code> on Cartesian cells, metallic walls, periodic and Bloch-periodic boundaries, mirror symmetry planes |
| Special stepping | <code>bfast_scaled_k</code> on Cartesian cells; a 2-D cell with an out-of-plane wavevector (MEEP's <code>kz_2d</code> settings <code>"complex"</code>, <code>"real/imag"</code> and <code>"3d"</code>) |
| Sources | <code>mp.Source</code> with continuous, Gaussian and custom waveforms; amplitude functions and amplitude arrays; <code>EigenModeSource</code> and Gaussian-beam sources on Cartesian cells in 2-D and 3-D |
| Monitors | <code>DftFlux</code>, <code>DftFields</code>, <code>DftNear2Far</code>, <code>DftForce</code> and <code>DftEnergy</code>; mode monitors through MEEP's own eigenmode decomposition after the run |
| Run loop | <code>until</code> and <code>until_after_sources</code>; MEEP step functions through the run-loop facade, including <code>mp.Harminv</code> |

Each entry is conditional on the combination. A feature that lifts alone can be
refused beside another one, and the refusal names the pair.

### Acceptance on MEEP's own scripts

The lift was run on the simulations that MEEP's own example and test scripts
construct (MEEP 1.33.0, single precision). The harness captures the
<code>mp.Simulation</code> each script builds; the script's own
<code>sim.run</code> call is not what is accepted.

| Corpus | Accepted | Refused | Status |
|---|---|---|---|
| Example scripts that build an <code>mp.Simulation</code> | 60 of 65 | 5 of 65, under 3 named reasons: no source declared at lift time, a multilevel-atom medium, a gyrotropic medium | measured |
| Test methods that captured an <code>mp.Simulation</code> | 134 of 233 | 99 of 233, of which 93 declare no source and are never stepped by MEEP either | measured |

The 194 accepted simulations (60 and 134) were lifted again on 2026-09-25, 194
of 194.

These counts are over MEEP's tutorial and test scripts. They are not a fraction
of MEEP's features, and they say nothing about how often a feature occurs in
research use. How closely the stepped fields agree with MEEP is a separate
measurement, in
[the floating-point contract](../design/floating-point.md#agreement-with-meep-is-measured-separately).

## What is refused, by name

Every refusal below is a reason string in <code>verdict.reasons</code> or in the
<code>MeepSimulationNotLiftable</code> a lift raises. The wording here is a
summary; the reason itself names the offending object and, where one exists,
the way out. [Troubleshooting](troubleshooting.md) lists the messages a run can
raise by their opening words.

### The process and the state of the simulation

| Refused | Why |
|---|---|
| A process that is one of several MPI ranks | The stepper is single-process and would step the whole cell on every rank. Run the lifting process without <code>mpirun</code> |
| A simulation that has already been stepped | The lift starts from zero fields and would discard that history. Lift a freshly constructed simulation, or call <code>sim.reset_meep()</code> first |
| A simulation that has already been initialized | What a caller can do to an initialized simulation is not visible to the lift |
| A simulation that declares no sources | It would step zero fields and return an all-zero volume that reads as a completed run. Sources declared after initialization, fields seeded by hand, and sources changed inside a step function are the common causes |

### Grid and coordinates

| Refused | Why |
|---|---|
| <code>dimensions</code> other than MEEP's Cartesian 1, 2 or 3, or cylindrical | No reduction exists for it |
| A <code>geometry_center</code> that is not the origin | The grid is centred on the origin. Shift the geometry and the source centres instead |
| A resolution that is not one finite positive number | The grid has one resolution for all axes |
| A Courant factor outside the stability limit of the run's dimension | The limit is enforced |
| A non-finite <code>k_point</code> component | — |
| A cylindrical <code>m</code> on a Cartesian grid | It has no meaning there |

### Cylindrical cells

| Refused | Why |
|---|---|
| A non-integer <code>m</code> | The azimuthal wrap phase is single-valued only for integer <code>m</code> |
| Mirror symmetry | Not implemented on cylindrical cells |
| A <code>k_point</code> with a radial or azimuthal component | Only a z Bloch phase is meaningful |
| <code>accurate_fields_near_cylorigin=True</code> with a Courant factor above 1 / (&#124;m&#124; + 0.5) | The near-axis update is unstable above that bound |
| <code>bfast_scaled_k</code> | Not representable on a cylindrical grid |
| A PML on the low side of r | The axis is not a boundary |
| <code>mp.Absorber</code> | Not measured on a cylindrical cell. Use <code>mp.PML</code> |
| <code>EigenModeSource</code>, Gaussian-beam sources | Not reproduced on cylindrical cells |
| A conductivity that varies from point to point | The per-point read is not validated there |
| A <code>DftForce</code> monitor | Not reproduced on cylindrical cells |

### Boundary layers

| Refused | Why |
|---|---|
| A PML with a non-default <code>R_asymptotic</code>, a <code>mean_stretch</code> other than 1, or a custom <code>pml_profile</code> | Only MEEP's default quadratic profile is lifted |
| A layer with a thickness that is not finite and positive, or two layers claiming one face with different thicknesses | One thickness per face |
| An <code>mp.Absorber</code> with a custom profile, or too thin to tabulate at the run's resolution | The absorber grading uses the default profile |
| An absorber on the low side of a mirror-folded axis | That cell straddles the mirror plane |

### Symmetry

| Refused | Why |
|---|---|
| Rotational symmetries | Only mirror planes are folded. Drop the symmetry: the run is still correct, only larger |
| A mirror whose phase is not +1 or −1, or that is not normal to a coordinate axis | — |
| A mirror on an axis the run makes translationally invariant | An invariant axis has no halves |
| A nonzero Bloch <code>k</code> on a folded axis | A mirror plane forces the field to be even or odd about itself |
| Sources that all sit in the half the mirror discards, or whose components break the mirror's parity | The stored half would hold no source |

### Sources

| Refused | Why |
|---|---|
| A source class other than those listed under [What lifts](#what-lifts), for example an indexed source | Only the listed classes are translated |
| A component the coordinate system does not inject | — |
| <code>amp_func_file</code> | The package reads no HDF5 file. Load the data yourself and pass <code>amp_data</code> or <code>amp_func</code> |
| <code>amp_data</code> on a cylindrical cell, with a shape MEEP's interpolator cannot index, or holding a non-finite value | — |
| A spatial profile on a point source | A profile needs an extent |
| A waveform that is not continuous, Gaussian or custom; a zero carrier frequency; a non-callable custom waveform | — |
| <code>EigenModeSource</code> in 1-D, with a volume that is not a line or a plane, with a diffracted-planewave band, with <code>eig_match_freq=False</code> and a custom waveform, with a negative frequency, or with <code>amp_data</code> | The mode solve that the lift re-runs does not cover these |
| A Gaussian beam in 1-D, with a volume that is not a line or a plane, with a zero or negative frequency, with a zero polarization vector, or in 2-D with both in-plane and out-of-plane polarization | MEEP itself aborts or steps no source in several of these |
| <code>GaussianBeam2DSource</code> outside a 2-D Cartesian run, without a line extent, with no in-plane direction, with a non-positive waist, without SciPy installed, or beside a mirror symmetry | — |

### Media

| Refused | Why |
|---|---|
| Magnetic permeability other than 1, magnetic susceptibilities, magnetic nonlinearity | The engine steps with μ = 1 and its dispersion is on the electric side only |
| Noisy, gyrotropic and multilevel-atom susceptibilities | Each adds state and a different update to the time step |
| A susceptibility with off-diagonal <code>sigma</code>, a non-positive frequency, a negative <code>gamma</code>, or a negative or non-finite <code>sigma</code> | A negative <code>gamma</code> is gain, which is unstable in the time domain |
| A rotated (off-diagonal) conductivity tensor | MEEP does not read it either |
| A negative conductivity | It is gain rather than loss |
| A permittivity that is not finite and positive | It has no stable leapfrog |
| A structured cell whose media differ in more than permittivity, electric conductivity and electric susceptibilities | The per-point read recovers only those |
| A <code>MaterialGrid</code> with a negative damping, with susceptibilities on an endpoint medium, or with endpoint fields MEEP's own grid evaluation never reads | — |
| A callable material that raises, returns something that is not a medium or a number, or returns a medium carrying more than a permittivity | A per-point route carries permittivity alone |
| A material of a type that names no medium the lift can read | — |

### Monitors and the run loop

| Refused | Why |
|---|---|
| DFT monitor kinds other than the five listed under [What lifts](#what-lifts) | A monitor that cannot be rebuilt would finish empty: a spectrum of zeros rather than an error |
| A <code>DftForce</code> monitor on a run that declares a symmetry, or whose declared direction differs from its surface normal | Only MEEP's diagonal stress-tensor branch is reproduced |
| <code>add_near2far</code> with <code>nperiods</code> other than 1, or a near-to-far region outside the stored grid | — |
| A <code>DftFields</code> monitor that requests a derived or material component | Only the twelve stored field components are accumulated |
| A mode monitor on the Yee grid (<code>yee_grid=True</code>) | — |
| A non-finite monitor frequency | — |
| A step function that reaches an attribute the run-loop facade does not carry: the HDF5 writers, <code>set_materials</code>, <code>output_volume</code>, LDOS | Raises <code>StepFunctionNotHosted</code> mid-run rather than reading the original simulation, whose fields are initialized and never stepped |

## Why the error is a feature

An FDTD simulation with the wrong boundary, material registration, source, or
monitor convention often produces finite, smooth fields. That makes an
unannounced approximation particularly dangerous. meep_gpu therefore has two
separate behaviours:

- A call asking for GPU execution (<code>prefer_gpu=True</code>, which is the
  default of <code>lift_simulation</code> and <code>run_on_gpu</code>) raises if
  the host has no GPU: neither a CUDA device through CuPy nor an Apple GPU. The
  message names <code>prefer_gpu=False</code>, the NumPy reference.
- A lifted MEEP simulation raises if its semantics are not represented by the
  package's contract.

Neither case is converted to a simplified physical model, and a host with no GPU
is never given a CPU run in place of the GPU one it asked for. The caller may
choose a CPU-MEEP fallback at a higher workflow layer, but that choice must be
visible in its own result metadata and logs.

Two cases are announced rather than raised. On an Apple GPU the device runs
compiled kernels only, and the array path runs on NumPy host arrays. A
<code>prefer_gpu=True</code> run there whose configuration no certified Metal kernel
covers, or, under <code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code>, whose environment
(GPU architecture, PyTorch, Metal frontend, <code>PYTORCH_MPS_FAST_MATH</code>) is
not certified or cannot be judged, steps the same equations on the host CPU. The
driver still reports <code>gpu == "metal"</code>; the run prints one line
beginning <code>meep_gpu: step path array on the host CPU; dispatch refused:</code>
and records the reason in <code>driver.fast_path_report()</code>. A run that
turned kernels off itself (<code>MEEP_GPU_DISPATCH=0</code> or
<code>MEEP_GPU_FUSED=0</code>) is on the host CPU there too, and prints nothing.
On a CUDA host the array path stays on the device: a run on an NVIDIA GPU or
Triton version outside the supported range, or, under
<code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code>, outside the certified list, steps
there with the same one-line announcement.

The second case is a run outside the certified identity on a supported GPU, with
the switch unset or <code>1</code>. On an Apple GPU, every one of which is
supported, it steps on the Metal kernels; on an NVIDIA GPU of compute capability
7.0 to 9.0 (with Triton 3.1 on the Triton table) it steps on the NVIDIA kernels,
except that a kernel table certified for that device and toolchain runs alone in
place of one that is only supported. The run is uncertified, prints one line
beginning <code>meep_gpu: NOTE the kernels are NOT CERTIFIED on this host:</code>,
and <code>driver.fast_path_report()</code> carries <code>certified: False</code>
([Certification on an NVIDIA GPU](kernel-dispatch.md#certification-on-an-nvidia-gpu),
[Certification on an Apple GPU](kernel-dispatch.md#certification-on-an-apple-gpu)).

## What MEEP continues to own

MEEP constructs the computational grid, resolves media, rasterizes geometry,
and applies native subpixel smoothing before a lift. It also remains the
appropriate engine for setup and analysis operations that do not occur inside
the grid-wide timestep loop.

The lift reads component-resolved inverse-permittivity data. A scalar
permittivity diagnostic is not enough at a smoothed or anisotropic interface:
it can blend differently registered field components and produce a plausible
but wrong result. The compatibility and realized-state checks protect against
that failure mode.

## Lifting and dispatch are separate questions

The preflight decides whether a simulation can be **lifted**. It is a separate
question from which implementation then steps it: the array path handles every
lifted configuration, and compiled kernels replace it slot by slot where a
released arm covers that slot on a host its kernel table admits. On a driver
built with <code>prefer_gpu=True</code> dispatch is on by default, and
<code>MEEP_GPU_DISPATCH=0</code> turns it off; a <code>prefer_gpu=False</code>
driver is the NumPy reference and never dispatches. Neither decision changes the
other's answer, and a kernel that does not cover a slot is never a reason a
supported run is refused. See
[Compiled-kernel dispatch](kernel-dispatch.md).

## Keep an evidence trail

For a run whose result will be relied on, keep:

1. the MEEP version and precision, and the simulation source,
2. the preflight verdict and reasons,
3. the selected backend and device,
4. **the dispatch state** — the enable's value (unset, <code>1</code> or
   <code>0</code>), which kernel table served, which arms, the subnormal
   policy installed, and whether the kernels are certified on this host
   (<code>certified</code>), all of which
   <code>driver.fast_path_report()</code> reports and
   <code>MEEP_GPU_DISPATCH_LOG</code> persists per configuration freeze,
5. the returned step count, simulation time, and timing, and
6. the numerical comparison appropriate to the physical observable.

Item 4 matters because two runs of the same script on the same host can execute
different kernels.

Read [Validation and performance](validation-and-performance.md)
before treating a successful run as a performance result.
