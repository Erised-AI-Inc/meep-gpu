<div class="manual-hero" markdown>
<p class="manual-kicker">Single-GPU time stepping for MEEP simulations</p>

# meep-gpu manual

Build a simulation with your own MEEP installation, then step it on one NVIDIA
or Apple GPU. MEEP keeps the setup and the post-processing.

</div>

meep-gpu (import name `meep_gpu`) is an add-on for MEEP users, distributed under
GPL-2.0-or-later. It takes the simulation a user's own MEEP installation has
built and time-steps it on one GPU: Triton or hand-written CUDA kernels on
NVIDIA hardware through CuPy, Metal kernels on Apple hardware through PyTorch.
MEEP remains the authority for simulation construction: geometry, material
definitions, grid construction, native subpixel smoothing, mode solving, and
analysis that does not execute in the timestep loop.

It is aimed at large three-dimensional problems run without a cluster. One
simulation runs on one GPU; on a GPU cluster, run one simulation per GPU. There
is no multi-GPU domain decomposition, and a lift under MPI is refused by name.
Expect no gain over MEEP on two-dimensional problems or on small grids, which
have not been timed against MEEP on a matched problem;
[Will it help?](guides/will-it-help.md) states what has been measured and what
each expectation rests on.

<div class="meep-reference" markdown>

**Use the [MEEP documentation](https://meep.readthedocs.io/en/latest/) for all ordinary MEEP work.**
That manual owns simulation construction, geometry, media, sources, boundaries,
monitors, analysis, and the features outside this package's accelerator
contract. This manual documents only the lift, accelerated stepping, its
compatibility gates, and its measurements.

</div>

## Start here

| Step | Page |
|---|---|
| 1. Does your problem fit? | [Will it help?](guides/will-it-help.md): where the GPU has been measured faster than MEEP, and where to expect no gain |
| 2. Install MEEP and the package, and run the check | [Installation](getting-started/installation.md), which sends you to the route for your machine in INSTALL.md |
| 3. Run a first simulation | [First lifted simulation](getting-started/first-lift.md): check a MEEP simulation, run it, read a field, release the arrays |
| 4. Read what ran | [Reading what ran](getting-started/reading-what-ran.md): the line a run prints, and the dispatch record |
| 5. Move your own script over | [MEEP user entry points](getting-started/meep-entry-points.md) and [Compatibility and refusals](guides/compatibility.md) |
| 6. Measure your own case | [Time your own simulation](guides/time-your-simulation.md) |

When something fails, [Troubleshooting](guides/troubleshooting.md) is organized
by the message you see. The [Glossary](reference/glossary.md) defines the terms
the manual uses.

| Also | Page |
|---|---|
| Monitors and readback | [Monitors and results](guides/monitors-and-results.md) |
| Compiled kernels: on by default, how to turn them off, which table your hardware uses | [Compiled-kernel dispatch](guides/kernel-dispatch.md) |
| Single precision, the subnormal policy, and what "bit-identical" is identity with | [Floating-point contract](design/floating-point.md) |
| The callable surface | [Public Python API](reference/public-api.md) |
| The reference path, the test suite, and the rules for a fair benchmark | [Validation and performance](guides/validation-and-performance.md) |
| Contributing, the tests, and the certification harness | [Development](development/index.md) |
| Citing the package and MEEP | [How to cite](citing.md) |

## The execution boundary

| MEEP retains | meep-gpu provides |
|---|---|
| Geometry objects, rasterization, and native subpixel smoothing | Yee-grid curl and constitutive updates |
| Medium definitions and material library | PML, metallic and Bloch boundaries, and mirror folding |
| Mode solving and adjoint optimization | Current-source injection, DFT, and flux accumulation |
| Far-field evaluation from migrated near-surface data | Accelerated stepping for the covered configuration, and a harmonic-inversion (filter-diagonalization) interface named as MEEP names it |

This boundary is intentional. Setup operations occur once, while FDTD updates
run over every grid cell at every time step. Retaining MEEP's setup prevents a
second, reduced geometry interface from becoming a source of silent physical
differences.

> **Warning: Not a whole-MEEP replacement**
>
> This package accelerates a defined subset of MEEP simulations. It does not
> claim that all MEEP features run on a GPU. A simulation outside the lift
> contract raises a named compatibility error rather than being approximated
> or silently changed.

## Choose an entry point

- Start with the [MEEP user entry points](getting-started/meep-entry-points.md)
  when you already have an <code>mp.Simulation</code>.
- Start with the [direct driver API](reference/public-api.md#direct-driver) when
  you need to create a Yee-grid run programmatically without importing MEEP.
- Read [compatibility and refusals](guides/compatibility.md) before putting a
  new simulation into a batch workflow.
- Read [monitors and results](guides/monitors-and-results.md) before assuming a
  MEEP readback API has an identical counterpart here.

## How a run is stepped

The NumPy **array path** — the package's own implementation of the time step as
whole-array operations — is always available and is the executable numerical
reference: a driver built with <code>prefer_gpu=False</code> runs it on every
host and never runs a compiled kernel, whatever the environment says.

A GPU request, <code>prefer_gpu=True</code>, runs on this host's GPU, and it is
what <code>lift_simulation(sim)</code> and <code>run_on_gpu(sim, ...)</code>
request when nothing else is passed. On a host with a live CUDA device that is
CuPy. On a host with an Apple GPU it is Metal kernels over NumPy host arrays. On
a host with neither it raises, naming <code>prefer_gpu=False</code>; it never
quietly becomes the reference. A small cell steps faster on the reference; see
[Small cells are faster on the reference](getting-started/first-lift.md#small-cells-are-faster-on-the-reference).

Each time step is seven sub-steps. **Compiled kernels** serve the sub-steps they
cover, and the array path serves the rest. They come in three **kernel tables**:
Triton and hand-written CUDA on NVIDIA hardware, Metal on Apple hardware. They
are **on by default** for a <code>prefer_gpu=True</code> driver, and
<code>MEEP_GPU_DISPATCH=0</code> keeps such a run on the array path. The
hardware, not a preference, decides which table can serve a run:

- a driver built with <code>prefer_gpu=True</code> on a CUDA device gets the two
  NVIDIA tables, or the hand-written CUDA table alone when no validated Triton
  is installed;
- a driver built with <code>prefer_gpu=True</code> on an Apple GPU gets the
  Metal table; the sub-steps no kernel covers run on the host CPU, and the run
  says so on stderr and in <code>driver.fast_path_report()</code>;
- a driver built with <code>prefer_gpu=False</code> never consults a table.

A device or toolchain the release does not certify is refused by name, and its
run takes the array path. [Reading what ran](getting-started/reading-what-ran.md)
shows how to tell which happened and why.

Kernel results are bit-identical to the package's own array path on the same
host, under the subnormal policy the kernel table is certified for. That is not
identity with MEEP: agreement with MEEP is measured separately, as a tolerance.
See the [floating-point contract](design/floating-point.md).

None of this is a general Torch array backend. The Metal table holds the field
volumes on the device between launches by default, which wins on larger grids
but not on small 2-D ones. Read
[compiled-kernel dispatch](guides/kernel-dispatch.md) before running on an Apple
host.

## Relationship to MEEP

meep-gpu is derived from MEEP and carries MEEP's licence, GPL-2.0-or-later. It
is an independent project, not affiliated with or endorsed by the MEEP
developers. See [License and attribution](license.md).
