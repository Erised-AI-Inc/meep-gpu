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

For 1.0, the lift was run on the simulations that MEEP's own example and test scripts
construct (MEEP 1.33.0, single precision). The harness captures the
<code>mp.Simulation</code> each script builds; the script's own
<code>sim.run</code> call is not what is accepted.

| Corpus | Accepted | Refused | Status |
|---|---|---|---|
| Example scripts that build an <code>mp.Simulation</code> | 60 of 65 | 5 of 65, under 3 named reasons: no source declared at lift time, a multilevel-atom medium, a gyrotropic medium | measured |
| Test methods that captured an <code>mp.Simulation</code> | 134 of 233 | 99 of 233, of which 93 declare no source and are never stepped by MEEP either | measured |

The 194 accepted simulations (60 and 134) were lifted again on 2026-09-25, 194
of 194. Over MEEP 1.35.0's scripts the corpus of 1.0.1 is 223 simulations, 60 examples
and 163 test methods ([MEEP identity on the corpus](../development/meep-identity-corpus.md)).
Accepted of those scripts, measured on 2026-11-02 on the NumPy reference with dispatch off
(one M1 Max, MEEP 1.35.0 single precision): 60 of 65 example scripts that build an
<code>mp.Simulation</code>, the same 65 as 1.33.0's with no verdict moved, and 164 of 274
test methods that build one; of the 229 test methods shared with 1.33.0 no verdict moved,
4 MPI-only methods now skip, and 45 methods are new (34 accepted). Accepted is the preflight's
verdict, and the corpus counts the methods whose simulation lifts, so the two differ: of the 164,
<code>test_plot1D</code>'s lift fails (a sampled permittivity outside its declared range) and
<code>test_ldos_3D</code>'s was not attempted, while <code>test_amp_file_func</code>, refused,
lifts with its source substituted, which gives the corpus's 163. The refusals name their
reason: 4 of the 5 refused examples and 103 of the 110 refused test methods declare no
source; the others are a multilevel-atom or gyrotropic medium, a <code>geometry_center</code>
off the origin, a non-integer <code>m</code>, an off-diagonal <code>ForceRegion</code>, a
near-to-far transform with <code>nperiods=10</code>, and an <code>amp_func_file</code>.

These counts are over MEEP's tutorial and test scripts. They are not a fraction
of MEEP's features, and they say nothing about how often a feature occurs in
research use. How closely the stepped fields agree with MEEP is a separate
measurement, in
[the floating-point contract](../design/floating-point.md#agreement-with-meep-is-measured-separately).

### An `EigenModeSource` can differ from MEEP

An <code>EigenModeSource</code> lifts, but its words are not always MEEP's, and a run with
one is not reproducible word for word from one process to the next, on the GPU or in MEEP
itself. The lift does not read the mode amplitudes that MEEP's own
<code>add_eigenmode_source</code> solve stored: it solves the mode again with its own
<code>get_eigenmode</code> call (MPB). MPB's eigensolver starts with exact line
minimization and switches to an approximate one when a step improved little and its own
<code>clock()</code> timings say the exact one costs more than twice as much. On the two-dimensional corpus solves
measured, each timed operation lasts a few microseconds, so the switch follows timing noise, and any solve, MEEP's
own included, can converge to a slightly different eigenvector in any process. The
difference is inside the eigensolver's tolerance: sampled mode amplitudes differ by 5.2e-15
to 1.1e-7, the frequency is equal, and the group velocity differs in its last digits. The
source then deposits other amplitudes, and every word the source reaches differs from the
first step on: over whole runs of three corpus rows, 1,366,368 of 8,370,000, 1,379,932 of
8,370,000 and 2,842,365 of 8,794,080 words, 16 to 32 %.

So a run with an <code>EigenModeSource</code> is not bit-identical to MEEP in every run. The
GPU steps, word for word, the mode the lift solved; that solve, like MEEP's own, is not
reproducible between processes, so against MEEP such a run can differ, in the same process
or in another, at the rates below, and MEEP's own run is not word-reproducible between
processes either. Runs without an
<code>EigenModeSource</code> are not affected, and neither is the kernels' identity with the
array path on the same inputs, which the certification's gates measure inside one lift.

How often, measured with this release, MEEP 1.35.0 single precision and MPB 1.12.0, on one
x86-64 Linux host and its RTX 3090:

| Measured | Result |
|---|---|
| The lift's mode against MEEP's own solve in the same process (NumPy reference; <code>test_adjoint_solver</code>'s <code>test_DFT_fields</code> and <code>test_eigenmode</code>, <code>test_mode_coeffs</code>'s <code>test_modes</code>) | differs in 4 of 300 processes on an otherwise idle host, 47 of 936 beside other work, and 3 of 108 whole runs |
| MEEP's own words against its own reference run (the same three rows) | differ in 0 of 300 processes on an idle host, 19 of 936 beside other work, and 2 of 108 whole runs |
| By row, idle (the lift's mode against MEEP's own) | <code>test_DFT_fields</code> 1 of 100 processes; <code>test_eigenmode</code> 0 of 100; <code>test_modes</code> 3 of 100 |
| Dispatched runs on the RTX 3090 against the reference words (the 32 corpus rows with an <code>EigenModeSource</code> that the certification's corpus check runs) | 8 of 132 runs differed, on 8 different rows; each of the 8 rows held in its other runs. Their signature is the solve's (first difference at step 1, on the source's planes); the source tables those runs deposited were not recorded, so that they came from the solve is inferred |

On the Apple M1 Max the variation has been seen too, on the NumPy reference: under MEEP
1.33.0, <code>test_mode_decomposition</code>'s <code>test_linear_taper_2d</code> differed in
1 of 16 runs (861,700 of 5,386,560 stored words, from the first step, with MEEP's own words
unchanged), as it did in 1 of 41 runs on the x86-64 Linux reference host of the 1.0
certification round; and under MEEP 1.35.0 the lift's mode differed from MEEP's own solve in
29 of 48 processes of two small two-dimensional waveguide cases (one source and two), on a
host loaded by other work, at nice 19; no idle rate was measured there. Its rate on the corpus
under MEEP 1.35.0 has not been measured. The mechanism and the rates above were measured
only with MEEP 1.35.0 and MPB 1.12.0: other MEEP versions, other MPB builds and
double-precision builds have not been measured for them.

Such a difference has a signature: it starts at the source's first deposit step, on the
source's planes and their neighbours, and the same script run again in a fresh process gives
MEEP's words in most runs (at the rates above). MEEP 1.35.0 offers no switch for it: MEEP
passes fixed flags to MPB, no parameter of <code>EigenModeSource</code> or
<code>Simulation.get_eigenmode</code> carries them, MPB reads no environment variable for
it, and this package adds none. A MEEP built so that its <code>get_eigenmode</code> call
adds <code>EIGS_FORCE_EXACT_LINMIN</code> to those flags would hold every solve to exact line
minimization; that build has not been made or run. Holding <code>clock()</code> constant for
MPB pins the same path: on the measured host it made the solves identical in every process,
and MEEP's own words then equalled that host's reference words on all 34 corpus rows with an
<code>EigenModeSource</code>, under <code>keep</code> and <code>flush</code>. It was used to
measure the effect and is a diagnostic, not a supported configuration.

The corpus check of the certification round declares these rows run-to-run variable: such a
row that differs runs again in a fresh process, up to 4 runs, and holds only when every
earlier run deposited other source tables than the run that held
([Certification](../development/certification.md#the-direct-corpus-check-against-meep-l3)).

Measured on the corpus row <code>ex.coupler</code> (<code>coupler.py</code>, an
<code>EigenModeSource</code> in a GDSII geometry) on the x86-64 Linux references of the
1.0 certification round, under <code>keep</code> and <code>flush</code>: 12 of 20 runs held (2 corpus
legs, 6 re-runs and 12 lift probes); a run that did not hold differed in 1,683,570 to
5,051,638 of 44,047,776 stored words. Across fresh processes the only lifted inputs that
varied were the two sources' point amplitudes from the lift's re-solve, and under
<code>flush</code> MEEP's own eigenmode solve varied too (3 distinct MEEP digests in 6
probes); with identical lifted inputs and identical MEEP words the array path's words were
identical, in 9 of 9 processes. On the Apple M1 Max the row held in 4 of 4 runs. Table 2
excludes <code>ex.coupler</code> by name from its Linux sets
([MEEP identity on the corpus](../development/meep-identity-corpus.md)). 1.0.1 keeps the
re-solve.

One row is a separate exception, deterministic and not caused by MPB's timings. Under MEEP
1.35.0 the corpus row <code>test_mode_coeffs</code>'s
<code>test_eigenmode_source_with_cylinders</code> differs from step 2 on every reference
(3,288,122 to 3,300,604 of 22,348,458 stored words, relative L2 about 1.5e-6), by the same
count in every process: MEEP's own words for it are reproducible between processes, the
re-solved mode differs from MEEP's in its last bits whatever the timings, and holding the
solve to one path does not change it. Table 2 excludes it by name from every set.

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
| A <code>BinaryPartition</code> <code>chunk_layout</code> together with mirror symmetry, given before MEEP builds the structure | MEEP then does not halve its grid: it steps the whole cell with the mirrors unused, while its own <code>get_array</code> and flux monitors still apply them, so no lift reproduces both. Drop the symmetries or the <code>chunk_layout</code>. A <code>chunk_layout</code> given as a Simulation or a file lifts, and so does a simulation whose structure MEEP has already built and halved (MEEP then sets <code>chunk_layout</code> to that structure's partition itself) |
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
covers, or, under <code>MEEP_GPU_HARDWARE=certified</code>, whose environment
(GPU architecture, PyTorch, Metal frontend, <code>PYTORCH_MPS_FAST_MATH</code>) is
not certified or cannot be judged, steps the same equations on the host CPU. The
driver still reports <code>gpu == "metal"</code>; the run prints one line
beginning <code>meep_gpu: step path array on the host CPU; dispatch refused:</code>
and records the reason in <code>driver.fast_path_report()</code>. A run that
turned kernels off itself (<code>MEEP_GPU_DISPATCH=0</code> or
<code>MEEP_GPU_FUSED=0</code>) is on the host CPU there too, and prints nothing.
On a CUDA host the array path stays on the device: a run on an NVIDIA GPU that no
kernel table supports, unless <code>MEEP_GPU_HARDWARE=any</code>, or, under
<code>MEEP_GPU_HARDWARE=certified</code>, one outside every table's certified identity,
steps there with the same one-line announcement, without <code>on the host CPU</code>.

The second case is a run outside the certified identity on a supported GPU, with
<code>MEEP_GPU_HARDWARE</code> unset (<code>supported</code>) or <code>any</code>. On an Apple GPU, every one of which is
supported, it steps on the Metal kernels; on a supported NVIDIA GPU (one the
installed CuPy or Triton can run on) it steps on the NVIDIA kernels, a table that
is only supported composing beside one certified for that device and toolchain. The run is uncertified, prints one line
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
