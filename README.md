# meep-gpu: GPU time stepping for MEEP simulations

`meep-gpu` takes a simulation that your own
[MEEP](https://github.com/NanoComp/meep) installation has built and time-steps
it on one NVIDIA or Apple GPU. MEEP still does the setup and the
post-processing; the fields and the monitor results come back through a result
object.

**Independent add-on.** This project is not affiliated with or endorsed by the
MEEP developers. It installs beside an unmodified MEEP and does not replace or
modify the `meep` module.

Version 0.9.0 is a preview release; see
[Certification status](#certification-status-of-this-preview). The import name
is `meep_gpu`. Licence: GPL-2.0-or-later.

**Start here.**

1. [Will it help?](#will-it-help): where the GPU has been measured faster than
   MEEP, and where to expect no gain.
2. [INSTALL.md](INSTALL.md): pick the route for your machine, install MEEP and
   this package, and run the installation check.
3. [First run](#first-run): step a MEEP simulation on the GPU, and
   [what the run prints](#what-the-first-run-prints).
4. [Reading what ran](docs/getting-started/reading-what-ran.md): which path
   served the run, and why.

The manual's [Getting started](docs/getting-started/index.md) follows the same
order and continues with porting your own script and timing it.

Coding agents helping someone use or change this package: read
[AGENTS.md](AGENTS.md).

## Will it help?

It is aimed at large three-dimensional problems on one GPU, for users without a
cluster. Expect no gain over MEEP on two-dimensional problems, on small
three-dimensional grids, on short runs, or on a GPU outside the certified set;
most of these cases have not been timed against MEEP (the list below says what
each rests on).

One comparison with MEEP has been made on the identical problem, on
2026-09-28. On one 3-D case, from 512,000 to 7,077,888 cells, the GPU stepped
1.80 to 3.67 times as fast as MEEP on the Triton route (what a default run
takes on a certified NVIDIA host with Triton installed), and 1.68 to 3.24 times
as fast on the CUDA-only route. Read those numbers with their conditions:

- one NVIDIA RTX A6000 against one host with 48 physical cores, in the same
  machine;
- single precision on both sides (MEEP 1.33.0 built from source with
  `--enable-single`);
- this one 3-D case (a dielectric sphere under PML) with one flux monitor, the
  same simulation on both sides;
- steady-state stepping, not counting the one-off lift before the first step,
  which took about 19 s at 512,000 cells and 216 s at 7,077,888 cells: on the
  Triton route the GPU run finishes first only after about 19,000 to 35,000
  steps;
- MEEP at the fastest of the seven rank counts tried (1, 8, 16, 24, 32, 48 and
  64);
- the CUDA-only route is the hand-written CUDA kernel table serving alone, the
  table a host without Triton uses (timed on a host that also had Triton); it
  is being improved for the next release, and these figures will be updated
  then.

[Will it help?](docs/guides/will-it-help.md) has the table, every condition
and the break-even step count at each size. The figures were measured on the
development tree this release was prepared from; a timing campaign from the
release tag is still to be run.

Where it does not help, or has not been measured:

- **Two-dimensional problems and small three-dimensional problems are slower
  than on MEEP.** None has been timed against MEEP on a matched problem: the
  ratio falls as the grid gets smaller (1.80 times at 512,000 cells), and the
  dispatched step has a fixed cost per step. Measure your own case.
- **Apple hardware.** No comparison with MEEP on the identical problem has been
  made on a Mac. On the one Mac measured, a default run is slower than the
  package's own NumPy reference below about 125,000 cells in 3-D.
- **Short runs.** Below the break-even step count, MEEP finishes first. Run to
  its own end time (1,400 to 3,360 steps), the timed case finishes first on
  MEEP at every size.
- **A GPU or library version outside the certified set** takes the array path.
  On an NVIDIA card that is CuPy on the GPU: on the one card measured, slower
  than MEEP's fastest rank count at each of the three sizes (0.22 to 0.73
  times); no other card has been measured. On Apple hardware it is NumPy on the
  host CPU, not timed against MEEP. Expect no gain over MEEP in either case.
- **No GPU.** `prefer_gpu=False`, the NumPy reference, is for checking results,
  not for speed.
- **MPI, or several GPUs for one simulation**: not supported, and refused by
  name. Run one simulation per GPU.

[Will it help?](docs/guides/will-it-help.md) has the detail, and
[Time your own simulation](docs/guides/time-your-simulation.md) measures your
own case.

## Install

This package runs on a MEEP installation and does not install MEEP through pip.
Each environment file below installs MEEP from conda-forge (the package
`pymeep`) together with the device libraries; to use a MEEP you already have,
conda-forge's or one built from source, read [INSTALL.md](INSTALL.md), the one
place that says exactly how, route by route. In short, from a checkout of this
repository:

    conda env create -n meep-gpu -f environments/apple-silicon.yml   # or nvidia-linux.yml, or reference-cpu.yml
    conda activate meep-gpu
    python -m pip install .

The environment name is yours to choose (INSTALL.md names the Route 3
environment `meep-gpu-reference`). On Linux with an NVIDIA GPU, then do step 3 of
[Route 1](INSTALL.md#route-1-one-nvidia-gpu-on-linux): Triton needs the driver
library under the name `libcuda.so`. Then check the installation; the check
steps one small simulation on this host's GPU, on the NumPy reference and on
MEEP, and compares them:

    python tools/check_install.py --require-gpu    # without --require-gpu on a host with no GPU

It must end with `OK: MEEP and meep-gpu work together on this host's GPU.`
(`... on the NumPy reference.` on a host with no GPU); any other `OK:` line
says where the run stepped instead, for example on the array path of a device
outside the certified list. A line starting `FAILED:` names the section of
[INSTALL.md](INSTALL.md) to read.
[INSTALL.md](INSTALL.md) also covers installing into an environment that already
holds MEEP, what the host must provide, and troubleshooting.

## First run

This is [`examples/quickstart.py`](examples/quickstart.py); its recorded output is
in [`examples/expected/quickstart.txt`](examples/expected/quickstart.txt).

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

`sim.run(...)` becomes `run_on_gpu(sim, ...)`, and results are read from the
result object. Call `close()` when you have read them.

### What the first run prints

On the certified Apple host, in the environment of Route 2 (MEEP 1.33.0 from
conda-forge, a double-precision build), the script prints, between MEEP's own
setup lines:

```text
supported: True []
steps: 840  MEEP time: 35.0
step path: fused
flux: 9.709557e-06 7.775078e-02 -3.753959e-06
max |Ez|: 1.436076e-01
```

The last digits of `flux` and `max |Ez|` depend on the MEEP build the lift
reads from.
[`examples/expected/quickstart.txt`](examples/expected/quickstart.txt) was
recorded against a single-precision MEEP built from source (its header says
so); with the double-precision MEEP of the environment files the printed
values agree with it to about 1e-4 (relative), and
`python examples/run_examples.py --check` compares within a tolerance, not
digit for digit.

The package also prints to standard error, at most once per process for each
distinct message. With a double-precision MEEP the first line is a note on
precision, and the second says what served the run:

```text
meep_gpu: this MEEP build is double precision and the engine steps single precision (float32 fields, complex64 for a complex run). The lift proceeds. Expect agreement with a MEEP run of the same simulation at the level of single-precision rounding, not of double precision: final fields differed by 1.5e-6 to 5.1e-4 (relative L2) on the 4 simulations compared
meep_gpu: step path fused; 4/7 slots (step_B,update_H,step_D,update_E) via PML,fused magnetic B/H pair,offdiag; table metal; policy flush (requested flush), installed by dispatch OVERRIDING this run's own keep resolution; torch 2.10.0 certified, metal frontend metalfe-32023.850.10 certified; device unknown uncertified-unknown
```

The second line, part by part:

| Part of the line | What it says |
|---|---|
| `step path fused` | compiled kernels served at least one sub-step; `step path array` would mean none did |
| `4/7 slots (...) via ...` | four of the seven sub-steps of one time step ran as compiled kernels, and which kernels; the other three ran on the array path |
| `table metal` | the kernel table that served: `triton` or `cuda` on NVIDIA hardware, `metal` on Apple hardware |
| `policy flush ..., installed by dispatch OVERRIDING ...` | the float32 subnormal policy the kernel table is certified under, installed for the whole process; this is why a default run and a `prefer_gpu=False` run can differ in the last bits |
| `torch 2.10.0 certified, metal frontend ... certified` | the toolchain was read and is on the certified list |
| `device unknown uncertified-unknown` | expected on a Mac: no device name is read there, and the certified identity is the toolchain pair |

`MEEP_GPU_DISPATCH=0` and `prefer_gpu=False` print no step-path line; the
precision note still appears with a double-precision MEEP.
[Reading what ran](docs/getting-started/reading-what-ran.md) explains the line,
the refusal lines and `result.driver.fast_path_report()`.

- **Defaults.** `run_on_gpu` and `lift_simulation` request this host's GPU
  (`prefer_gpu=True`). `prefer_gpu=False` selects the NumPy reference, which
  runs on any host. On a host with no GPU the default raises before MEEP
  initializes anything, and the message names `prefer_gpu=False`; nothing falls
  back silently.
- **Kernel dispatch is on.** `MEEP_GPU_DISPATCH=0` keeps a run on the array
  path.
- **One OpenMP runtime on an Apple host.** In the environment of
  `environments/apple-silicon.yml` a process holds one OpenMP runtime, and the
  order of the imports does not matter. A PyTorch wheel from PyPI beside
  conda-forge's default OpenBLAS loads two, and the process stops with
  `OMP: Error #15`; `conda install -c conda-forge --override-channels
  "libopenblas=*=*pthreads*"` leaves one and keeps the PyTorch you have. Do not
  set `KMP_DUPLICATE_LIB_OK`
  ([INSTALL.md](INSTALL.md#one-openmp-runtime-on-an-apple-silicon-mac)).

## Supported and certified hardware

By default, compiled kernels run on a certified device identity and on nothing
else. One NVIDIA compute capability and one Apple toolchain are certified.

| Route | Certified identity | Certified on |
|---|---|---|
| NVIDIA, Triton kernels | compute capability 8.6 with Triton 3.1.0 | one RTX A6000, CuPy 13.5.1 |
| NVIDIA, hand-written CUDA kernels | compute capability 8.6 | one RTX A6000, CuPy 13.5.1 |
| Apple, Metal kernels | PyTorch 2.10.0 with Metal frontend 32023.850.10 | one M1 Max |

Two machines have been measured. Other devices that report compute capability
8.6, and other Apple silicon hosts with the certified PyTorch and Metal
frontend, are admitted by the identity rule and have not been measured.

The NVIDIA rows are not a hard-coded list: each is derived from the per-architecture
records the kernel ledgers carry, so another compute capability becomes certified by
running the gates on a card of that architecture and writing its records, with no change
to the package ([Certifying another compute capability](docs/development/certification.md#certifying-another-compute-capability)).

**What any other device gets by default.** A device or a library version
outside the certified identities is refused by name and the run takes the array
path: CuPy on the GPU on NVIDIA hardware, NumPy on the host CPU on Apple
hardware. That run is correct and slow ([Will it help?](#will-it-help)). NVIDIA devices of any
other compute capability, any other Triton or PyTorch version, and an Apple
host whose operating system reports another Metal frontend are all in this
group; the Metal frontend belongs to the operating system, so a system update
can move a host out of the certified pair. A device whose identity cannot be
read is not refused. `driver.fast_path_report()` states which path served a run
and why.

**The newest PyTorch on a Mac.** `python -m pip install --upgrade torch` in the
environment of `environments/apple-silicon.yml` installs PyTorch 2.14.0 (the
newest release on 2026-09-28). It is not certified: by default every run takes
the array path, NumPy on the host CPU, and says so by name. With the opt-in
below the Metal kernels run on it; on one M1 Max their results on the 3 examples
were identical, byte for byte, to the array path
([INSTALL.md](INSTALL.md#the-newest-pytorch-on-a-mac)).

**Opting in on a device that is not certified.** Set the environment variable
`MEEP_GPU_ALLOW_UNCERTIFIED` to `1`:

    MEEP_GPU_ALLOW_UNCERTIFIED=1 python my_simulation.py

The switch is off by default and it is strict. `1` admits; `0`, or leaving it
unset, keeps the refusal described above; any other value (`true`, `yes`, an
empty string) is refused by name and the run takes the array path, on a
certified host as well. With `1`, a device or a library version whose identity
was read and is not certified dispatches the compiled kernels: an NVIDIA device
of another compute capability, another Triton version, another PyTorch version,
another Metal frontend. It admits the identity and nothing else; every other
rule of dispatch applies as before. The run prints one line per process, on
standard error, saying that the kernels are not certified on this host and
naming what is certified, and `driver.fast_path_report()` carries
`certified: False` with the identity that was read. The refusal a run gets
without the switch names it.

A run opted in this way carries no certification: compare its results with a
`prefer_gpu=False` run of the same simulation before relying on them. The
switch has been exercised with substituted identities (compute capability 8.9
and 9.0, another Triton version, another PyTorch version, another Metal
frontend) and on one real identity outside the list, PyTorch 2.14.0 on one M1
Max; no NVIDIA device outside the certified identities has been run.

## Precision

The engine steps single precision, whatever precision your MEEP was built
with: fields come back as `float32`, or `complex64` for complex runs.

The MEEP most users install is double precision. The packaging recipe of the
conda-forge builds has no single-precision option (0 of the 412 builds
published for Linux x86_64 and Apple silicon can be single precision), and
single precision needs a source build with `--enable-single`. Every measurement behind this package, the
timings and the agreement with MEEP alike, was made against MEEP 1.33.0 built
from source in single precision.

What to expect with a double-precision MEEP:

- The lift proceeds and says so. A lift from a double-precision MEEP prints one
  line per process, on standard error, stating that the engine steps single
  precision and the agreement to expect. The precision and the version of the
  MEEP that was read are recorded on the driver as `driver.lift_record`
  (`result.lift_record` after `run_on_gpu`). No MEEP is refused on its version
  or its precision, and no message is printed about the version.
- A comparison with your own MEEP run compares single-precision stepping with
  double-precision stepping. A quantity that needs more than about seven
  significant digits should stay on MEEP.
- What has been run: on one Apple host, with MEEP 1.33.0 from conda-forge
  (double precision, serial), 4 of 4 examples of this repository ran and the
  NumPy reference differed from that MEEP by 5.2×10^-6, 7.6×10^-6 and 5.1×10^-4
  of the final field on the 3 examples that compare with MEEP (2026-09-28). The
  installation check and the examples were also run with MEEP 1.34.0 from
  conda-forge on the same host, and with MEEP 1.33.0 and 1.34.0 from conda-forge
  on one NVIDIA host; `INSTALL.md`, "Tested MEEP versions and builds", lists
  every run. No release other than 1.33.0 and 1.34.0 has been run.
- No speed has been measured against a double-precision MEEP.

Read what you have with
`python -c "import meep as mp; print(mp.__version__, mp.is_single_precision())"`.

## Bit identity, and agreement with MEEP

The kernels are bit-identical to the package's own array path on the same
device under the same subnormal policy. The array path is CuPy on the device on
an NVIDIA host, and NumPy on the host CPU on an Apple host.

A default GPU run and a `prefer_gpu=False` reference run made in another
process are **not** bit-identical once values pass through the float32
subnormal range, because kernel dispatch installs its kernel table's subnormal
policy (flush, on Apple hardware) for the whole process and the reference run
takes the host as it finds it. Measured: 24,550 of the 25,600 cells of one case
differed. On the 24,000-cell two-dimensional example of this repository, on one
M1 Max, 23,968 of 24,000 cells differ between the two runs, by 1.0×10^-5 of the
final field, and 0 of 24,000 differ between the GPU run and the array path run
under the policy the GPU run reported. `MEEP_GPU_DISPATCH=0` alone does not
reproduce a dispatched run byte for byte: it leaves the host's own policy in
place. [Examples](examples/README.md) shows how the three runs are made.

Agreement with MEEP is a separate measurement, and it is a tolerance, not an
identity. Against MEEP 1.33.0 in single precision, the simulations built by 58
of 60 of MEEP's example scripts agree within 2.4×10^-7 to 1.6×10^-5 (relative
L2 over the whole field); the largest grid compared had 338,688 cells. That was
measured on the NumPy reference on 2026-08-09, not through the kernels and not
at the timed sizes. No identity between NVIDIA and Apple results is claimed.
[Floating-point contract](docs/design/floating-point.md) has the detail.

## Certification status of this preview

Stated plainly:

- The kernels were certified on the source as it stood before the last changes
  made for this release: the GPU default of the two entry points, and the
  edits that prepared the files for publication.
- The certification ledgers shipped here pin files by digest. Some pinned files
  changed after the ledgers were cut, so the recorded digests do not match the
  published files.
- **The ledger-contract tests are expected to fail** until the certification
  round has been re-run on the published files and its ledgers are committed.
  That round is still to be run. They form the certification suite (pytest marker
  `certification`), which a run without `-m` does not select and
  `python -m pytest meep_gpu parity/meep_gpu -m certification` runs; there
  they are reported as failed, and none is skipped or removed.
  `tools/ci/pending_certification.txt` lists the failures it may show, and
  [Running the tests](docs/development/testing.md) explains them.
- Kernel dispatch is on by default. Its go/no-go from the default and the route
  campaigns of the three kernel tables ran on the code of this release on
  2026-09-29 and released; they are recorded in the ledgers in that round
  ([Certification status](docs/development/certification.md#certification-status-of-this-release)).

[Running the tests](docs/development/testing.md#what-a-run-of-the-suites-shows-in-this-preview) records what a run of the suites shows
today, in three environments.

## Limits

- One simulation runs on one GPU, in one process. There is no MPI and no
  multi-GPU domain decomposition; a lift in a process that is one of several
  MPI ranks is refused by name.
- Single precision only.
- It covers a measured subset of MEEP. The lift accepts the simulations built
  by 60 of 65 example scripts and 134 of 233 test methods of MEEP 1.33.0's own
  corpus; what it cannot serve it refuses with the reason
  ([Compatibility and refusals](docs/guides/compatibility.md)).
- Monitors are accumulated, and sources deposited, on the array path. Monitors
  are not held on the device.
- Runs that store complex fields (Bloch-periodic, cylindrical with m ≠ 0) take
  the array path for the affected sub-steps.
- The lift precedes the first step and runs in one process, MEEP's own
  initialization included.
- One three-dimensional case has been timed against MEEP on the identical
  problem, at three sizes up to 7,077,888 cells, on one NVIDIA host. None has
  been timed against MEEP on Apple hardware.
- A run that dispatches compiled kernels changes the float32 subnormal policy
  of the whole process, and on the Apple host measured the policy was still
  installed after `close()`.

[CHANGELOG](CHANGELOG.md), "Known limits", has the full list.

## Relation to MEEP's planned GPU backend and to other projects

- **MEEP's planned backend.** No MEEP release has GPU support; the latest is
  v1.34.0 of 2026-07-09. A MEEP collaborator has published a design, "Optional
  Accelerator Backends and NVIDIA GPU Support for Meep"
  ([design document](https://hackmd.io/@alechammond/H1ep-MEIGx), August 2026),
  and wrote on 2026-09-06 that its implementation has begun
  ([discussion 2121](https://github.com/NanoComp/meep/discussions/2121)). As
  designed, it is written in C++ inside MEEP, targets NVIDIA hardware, and is
  selected through an opt-in backend argument. It plans single and double
  precision, several GPUs through MPI, monitors held on the device, and
  material initialization on the device. It validates by tolerance and names
  bitwise identity as a non-goal. No timeline is given.
- **[gpmeep](https://github.com/Semiconductor-Nanophotonics-Laboratory/gpmeep-releases)**
  is a GPL distribution of MEEP with CUDA execution on one or several GPUs. It
  provides its own `meep` module.
- **This package** runs beside an unmodified MEEP installation, on NVIDIA and
  on Apple hardware, on one GPU, in single precision, and its kernels are
  bit-identical to its own reference path. It covers a measured subset of MEEP
  and does not do MPI, several GPUs, double precision or device-held monitors.

The projects are complementary. A user who needs several GPUs can use gpmeep
today; a backend maintained inside MEEP, with double precision, is what MEEP's
planned backend is designed to be, and it has not been released. This package
serves a user with one GPU, NVIDIA or Apple, and the MEEP they already have.

## Documentation

- [INSTALL.md](INSTALL.md): the three install routes, an existing MEEP
  environment, and troubleshooting keyed by the message you see.
- The manual, under [`docs/`](docs/index.md):
  [Getting started](docs/getting-started/index.md),
  [Reading what ran](docs/getting-started/reading-what-ran.md),
  [Will it help?](docs/guides/will-it-help.md),
  [Time your own simulation](docs/guides/time-your-simulation.md),
  [Compatibility and refusals](docs/guides/compatibility.md),
  [Monitors and results](docs/guides/monitors-and-results.md),
  [Compiled-kernel dispatch](docs/guides/kernel-dispatch.md),
  [Troubleshooting](docs/guides/troubleshooting.md),
  [Public Python API](docs/reference/public-api.md) and the
  [Glossary](docs/reference/glossary.md).
- [Examples](examples/README.md): MEEP scripts stepped on the GPU, each with its
  reference runs beside it and its recorded output.
- [CONTRIBUTING.md](CONTRIBUTING.md), [Running the tests](docs/development/testing.md)
  and [the certification harness](docs/development/certification.md). Using the
  package needs nothing under `parity/`.
- [AGENTS.md](AGENTS.md): commands and rules for coding agents, whether they
  help someone run a simulation or change the package.

## How to cite

Cite this software through [`CITATION.cff`](CITATION.cff) (Ivan Biggs, Alicia
Zeng and Yanlin Dou, meep-gpu, version 0.9.0, 2026), and cite MEEP, which
builds every simulation this package steps:

> A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos and
> S. G. Johnson, "MEEP: A flexible free-software package for electromagnetic
> simulations by the FDTD method", Computer Physics Communications 181, 687–702
> (2010), doi:10.1016/j.cpc.2009.11.008.

In BibTeX:

```bibtex
@software{meep_gpu_2026,
  author  = {Biggs, Ivan and Zeng, Alicia and Dou, Yanlin},
  title   = {{meep-gpu: single-GPU time stepping for MEEP simulations on NVIDIA and Apple hardware}},
  version = {0.9.0},
  year    = {2026},
  url     = {https://github.com/Erised-AI-Inc/meep-gpu},
  license = {GPL-2.0-or-later}
}

@article{oskooi_2010,
  author  = {Oskooi, A. F. and Roundy, D. and Ibanescu, M. and Bermel, P. and
             Joannopoulos, J. D. and Johnson, S. G.},
  title   = {{MEEP}: A flexible free-software package for electromagnetic
             simulations by the {FDTD} method},
  journal = {Computer Physics Communications},
  volume  = {181},
  number  = {3},
  pages   = {687--702},
  year    = {2010},
  doi     = {10.1016/j.cpc.2009.11.008}
}
```

## Licence

GPL-2.0-or-later, the licence of MEEP; see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). The package is derived from MEEP and keeps MEEP's copyright
notice.
