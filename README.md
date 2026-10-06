# meep-gpu: GPU time stepping for MEEP simulations

`meep-gpu` takes a simulation that your own
[MEEP](https://github.com/NanoComp/meep) installation has built and time-steps
it on one NVIDIA or Apple GPU. MEEP still does the setup and the
post-processing; the fields and the monitor results come back through a result
object.

**Independent add-on.** This project is not affiliated with or endorsed by the
MEEP developers. It installs beside an unmodified MEEP and does not replace or
modify the `meep` module.

**Status: alpha (0.9.2).** This is an early release. Significant changes are
planned before version 1.0, the version the accompanying paper will cite, and
interfaces and record formats may change until then. See
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
three-dimensional grids, on short runs, or on an NVIDIA GPU outside the
supported range (compute capability 7.0 to 9.0); most of these cases have not
been timed against MEEP (the list below says what each rests on).

One comparison with MEEP has been made on the identical problem, on
2026-09-28. On one 3-D case, from 512,000 to 7,077,888 cells, the GPU stepped
1.80 to 3.67 times as fast as MEEP on the Triton route (what a default run
takes on a certified NVIDIA host with Triton installed), and 1.68 to 3.24 times
as fast on the CUDA-only route. Read those numbers with their conditions:

- one NVIDIA RTX A6000 against one host with 48 physical cores, in the same
  machine;
- single precision on both sides (MEEP 1.33.0 built from source with
  `--enable-single`, `-O2`, as a portable binary, one thread per rank with
  `OMP_NUM_THREADS=1`; no `-march=native` build or MPI×OpenMP run was tried as
  a control);
- this one 3-D case (a dielectric sphere under PML) with one flux monitor, the
  same simulation on both sides; 78 % of its cells are PML and its source sits
  off the fused seams, so it pays no deposit-repair cost, and a case with a thin
  PML or a source on a fused seam may gain less;
- steady-state stepping, not counting the one-off lift before the first step,
  which took about 19 s at 512,000 cells and 216 s at 7,077,888 cells: on the
  Triton route the GPU run finishes first only after about 19,000 to 35,000
  steps;
- MEEP at the fastest of the seven rank counts tried (1, 8, 16, 24, 32, 48 and
  64);
- the CUDA-only route is the hand-written CUDA kernel table serving alone, the
  table a host without Triton uses (timed on a host that also had Triton); its
  off-diagonal electric kernel changed in 0.9.1, and these figures have not
  been re-timed since that change.

[Will it help?](docs/guides/will-it-help.md) has the table, every condition
and the break-even step count at each size. The figures were measured on the
development tree this release was prepared from; a timing campaign from the
release tag is still to be run. The run records behind them are not in this
repository.

Where it does not help, or has not been measured:

- **Two-dimensional problems and small three-dimensional problems are expected
  to be slower than on MEEP** (inferred; not timed). None has been timed
  against MEEP on a matched problem: the ratio falls as the grid gets smaller
  (1.80 times at 512,000 cells), and the dispatched step has a fixed cost per
  step. Measure your own case.
- **Apple hardware.** No comparison with MEEP on the identical problem has been
  made on a Mac. On the one Mac measured (one pass, on a loaded host), a
  default run is slower than the package's own NumPy reference below about
  125,000 cells in 3-D.
- **Short runs.** Below the break-even step count, MEEP finishes first. Run to
  its own end time (1,400 to 3,360 steps), the timed case finishes first on
  MEEP at every size.
- **An NVIDIA GPU or library version outside the supported range** takes the
  array path, CuPy on the GPU: on the one card measured, slower than MEEP's
  fastest rank count at each of the three sizes (0.22 to 0.73 times); no other
  card has been measured. Expect no gain over MEEP. A supported NVIDIA GPU
  outside the certified set (compute capability 7.0 to 9.0) runs the compiled
  kernels by default, uncertified, and says so
  ([Supported and certified hardware](#supported-and-certified-hardware)); no
  such card has been timed.
- **An Apple GPU, PyTorch or macOS build outside the certified environment**
  runs the Metal kernels by default: every Apple GPU is supported. The run is
  uncertified, and says so
  ([Supported and certified hardware](#supported-and-certified-hardware)). No
  Apple GPU but one M1 Max has been timed.
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
says where the run stepped instead, for example on the array path of an NVIDIA
device outside the supported range. A line starting `FAILED:` names the section
of [INSTALL.md](INSTALL.md) to read.
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
meep_gpu: step path fused; 4/7 slots (step_B,update_H,step_D,update_E) via PML,fused magnetic B/H pair,offdiag; table metal; policy flush (requested flush), installed by dispatch OVERRIDING this run's own keep resolution; torch 2.10.0 certified, metal frontend metalfe-32023.850.10 certified; Apple M1 Max (applegpu_g13s) certified
```

The second line, part by part:

| Part of the line | What it says |
|---|---|
| `step path fused` | compiled kernels served at least one sub-step; `step path array` would mean none did |
| `4/7 slots (...) via ...` | four of the seven sub-steps of one time step ran as compiled kernels, and which kernels; the other three ran on the array path |
| `table metal` | the kernel table that served: `triton` or `cuda` on NVIDIA hardware, `metal` on Apple hardware |
| `policy flush ..., installed by dispatch OVERRIDING ...` | the float32 subnormal policy the kernel table is certified under, installed for the whole process; this is why a default run and a `prefer_gpu=False` run can differ in the last bits |
| `torch 2.10.0 certified, metal frontend ... certified` | the PyTorch version and the Metal frontend were read, and every certification record the Metal table cites names them |
| `Apple M1 Max (applegpu_g13s) certified` | the GPU, and its architecture as Metal names it, which every certification record the Metal table cites names |

On a Mac each mark is `certified`, `UNCERTIFIED` (the records name another
value), or `uncertified-unknown` (the fact could not be read, or the records do
not name it; the GPU architecture is read on macOS 14 or later). On an Apple GPU
that is not certified, the GPU's mark starts with `supported,`: every Apple GPU
is supported, and this one is not certified, as in
`Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED`. A Mac whose line carries
`UNCERTIFIED` still steps on the kernels, and the line is followed by one
beginning `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host`
([Supported and certified hardware](#supported-and-certified-hardware)).

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

Supported and certified are two statements. Hardware is *supported* when the
kernels are meant to run on it, and they run there by default. An identity is
*certified* when the certification gates ran the kernels on it and the kernel
table's certification records say so; how each table reads its records is
below. Supported is not certified: an Apple GPU no certification ran on is
supported and uncertified.

- **Apple.** Every Apple GPU is supported, M1 through M5 and later: the Metal
  kernels run on any Apple GPU and say whether its environment is certified. A
  GPU is Apple's when Metal names its architecture `applegpu_*`, or, where the
  architecture cannot be read (before macOS 14), when its name starts `Apple `.
- **NVIDIA.** Every NVIDIA GPU of compute capability 7.0 to 9.0 is supported by
  both kernel tables, with Triton 3.1 on the Triton table: the kernels run on
  such a device by default and say whether it is certified. Outside that range
  the kernel tables are refused by name unless `MEEP_GPU_ALLOW_UNCERTIFIED=1`.

One NVIDIA compute capability and one Apple environment are certified.

| Route | Certified identity | Certified on |
|---|---|---|
| NVIDIA, Triton kernels | compute capability 8.6 with Triton 3.1.0 | one RTX A6000, CuPy 13.5.1 |
| NVIDIA, hand-written CUDA kernels | compute capability 8.6 | one RTX A6000, CuPy 13.5.1 |
| Apple, Metal kernels | GPU architecture `applegpu_g13s`, PyTorch 2.10.0, Metal frontend `metalfe-32023.850.10`, `PYTORCH_MPS_FAST_MATH` unset or `0` | one M1 Max |

A Metal environment is those four facts. The GPU architecture is the one Metal
compiles for (`MTLDevice.architecture`, read on macOS 14 or later); it tells
apart GPU generations that share a Metal GPU family, such as M3 and M4. The
Metal frontend belongs to the operating system: every Mac on one macOS build
reports the same one, so a system update can move a Mac out of the certified
environment. `PYTORCH_MPS_FAST_MATH=1` compiles the kernels in fast-math mode;
on one M1 Max it changed 274,523 of 1,048,576 float32 divides and 327,338 of
1,048,576 square roots (2026-10-02), and no certification ran with it.

Two machines have been measured. Other devices that report compute capability
8.6, and other Macs whose GPU reports `applegpu_g13s` with the certified
PyTorch and Metal frontend and fast math off, are certified by the identity
rule and have not been measured.

The NVIDIA rows are not a hard-coded list: each is derived from the per-architecture
records the kernel ledgers carry, so another compute capability becomes certified by
running the gates on a card of that architecture and writing its records, with no change
to the package ([Certifying another compute capability](docs/development/certification.md#certifying-another-compute-capability)).
The Apple row is read the same way, from the 45 certification records the Metal
table cites: a fact is certified when all 45 name it. Those records hold one run
each, so certifying another Apple GPU architecture replaces the M1 Max's
certification rather than adding to it.

**What any other NVIDIA device gets by default.** On a supported device or
Triton version that is not certified (a compute capability from 7.0 to 9.0 other
than 8.6, or a Triton 3.1 release other than 3.1.0) the compiled kernels run and
the run is labelled uncertified. Its status line marks the device
`supported, UNCERTIFIED`, and the run prints one line per process, on standard
error, naming what was read and what is certified:

    meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU compute capability 8.9 (certified: 8.6). They were dispatched because this NVIDIA GPU and toolchain are supported but not certified bit-identical (the supported range is compute capability 7.0 to 9.0, with Triton >=3.1,<3.2 or with the NVRTC of either CuPy build, CUDA 11.8 for cupy-cuda11x or the host's CUDA 12 for cupy-cuda12x), and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the kernels to certified ones; the gates with no run on this GPU are counted in the dispatch record under uncertified.served[], which names the first five in welds_without_a_live_run_here; compare the results with a prefer_gpu=False run of the same simulation before relying on them

`driver.fast_path_report()` carries `certified: False` with what was read under
`uncertified.served`. Where one NVIDIA kernel table is certified for the device
and toolchain and the other is only supported, the certified table runs alone
and the run is certified; the other is refused by name. A device or library
version outside the supported range (compute capability below 7.0 or above 9.0,
another Triton release line) is refused by name and the run takes the array
path, CuPy on the GPU. That run is correct and slow
([Will it help?](#will-it-help)). A device whose identity cannot be read is not
refused. `driver.fast_path_report()` states which path served a run and why.

**What any other Mac gets by default.** The Metal kernels run: its Apple GPU is
supported. Another GPU architecture (that of an M2, M3, M4 or M5, for example),
another PyTorch version, another Metal frontend, or `PYTORCH_MPS_FAST_MATH` set
to any value but `0` is not certified; the run steps on the kernels all the
same. Its status line marks an Apple GPU that is not the certified one
`supported, UNCERTIFIED` (`Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED`),
and the certified GPU `certified` whatever else is uncertified. The run prints
one line per process, on standard error, naming what was read and what is
certified:

    meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU architecture applegpu_g15p (certified: applegpu_g13s). They were dispatched because this Apple GPU is supported, and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the Metal kernels to certified environments; compare the results with a prefer_gpu=False run of the same simulation before relying on them

`driver.fast_path_report()` carries `certified: False`, with each fact that is
not certified under `uncertified.served`, and `environment.device_supported`:
`True` on an Apple GPU, `False` on a GPU that is not Apple's, `None` when
neither its architecture nor its name was read. A GPU that is not Apple's is not
supported; the Metal kernels run on it too unless
`MEEP_GPU_ALLOW_UNCERTIFIED=0`, its status line carries no `supported`, and the
note gives as the reason
`the Metal table runs on environments outside the certified set unless MEEP_GPU_ALLOW_UNCERTIFIED=0`.
A fact that could not be read, or that the records do not name, runs too; when
no other fact is uncertified the report carries `certified: None` and no note is
printed.
[Certification on an Apple GPU](docs/guides/kernel-dispatch.md#certification-on-an-apple-gpu)
has the detail.

**The newest PyTorch on a Mac.** `python -m pip install --upgrade torch` in the
environment of `environments/apple-silicon.yml` installs PyTorch 2.14.0 (the
newest release on 2026-09-28). It is not certified: the Metal kernels run on
it, and every run prints a note of that kind, naming the PyTorch version. On
one M1 Max (2026-09-28) the kernels ran on it uncertified, and their results on
the 3 examples were identical, byte for byte, to the array path
([INSTALL.md](INSTALL.md#the-newest-pytorch-on-a-mac)).

**`MEEP_GPU_ALLOW_UNCERTIFIED`.** The switch is strict: it takes `1` or `0`,
and any other value (`true`, `yes`, an empty string) is refused by name and the
run takes the array path, on every kernel table and on a certified host as
well.

    MEEP_GPU_ALLOW_UNCERTIFIED=1 python my_simulation.py   # NVIDIA: the kernels on an unsupported device too
    MEEP_GPU_ALLOW_UNCERTIFIED=0 python my_simulation.py   # every table: the kernels on certified identities only

What the two values do depends on the table:

- **NVIDIA tables.** Unset runs a supported identity uncertified and refuses an
  unsupported one, as above. `0` restricts the compiled kernels to certified
  devices and toolchains: anything else, a device whose compute capability could
  not be read included, is refused by name and the run takes the array path,
  CuPy on the GPU. `1` also dispatches on a device or library version outside
  the supported range, recorded `supported: False`, and composes a supported,
  uncertified table beside a certified one. It admits identities and nothing
  else; every other rule of dispatch applies as before. The refusal of an
  unsupported identity names the switch.
- **Metal table.** Unset or `1` runs an uncertified environment, as above. `0`
  restricts the Metal kernels to the certified environment: a fact that is not
  certified is refused by name, and so is one that could not be judged, and the
  run takes the array path, NumPy on the host CPU.

A run on an uncertified identity prints one line per process, on standard
error, saying that the kernels are not certified on this host and naming what
is certified, and `driver.fast_path_report()` carries `certified: False` with
the identity that was read. Such a run carries no certification: compare its
results with a `prefer_gpu=False` run of the same simulation before relying on
them. Uncertified runs have been exercised with substituted identities
(supported compute capabilities 7.5, 8.9 and 9.0, unsupported 6.1 and 10.0,
another Triton version inside and outside 3.1, another GPU architecture,
another PyTorch version, another Metal frontend, `PYTORCH_MPS_FAST_MATH=1`) and
on one real identity outside the list, PyTorch 2.14.0 on one M1 Max; no NVIDIA
device outside the certified identities has been run.

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
an NVIDIA host, and NumPy on the host CPU on an Apple host. That is what the
gates certify, on the certified identities; on a supported Apple GPU outside
the certified environment, where the Metal kernels run by default, no gate has
measured it.

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

- The certification round ran on the files this release ships: the Triton and
  hand-written CUDA tables on compute capability 8.6 (one RTX A6000), and the
  Metal table on the Apple M1 Max (`applegpu_g13s`). Every certification entry
  in the shipped ledgers is bound to the published files except the two
  fused-product timing records.
- Kernel dispatch is on by default. Its go/no-go from the default and the route
  campaigns of the three kernel tables ran on these files, in the rounds
  stamped 2026-10-05, and released, and they are recorded in the ledgers
  ([Certification status](docs/development/certification.md#certification-status-of-this-release)).
- **Two tests are pending**:
  `test_dispatch_preference::test_at_the_top_of_the_sweep_the_two_pair_plans_are_faster`,
  for `[cuda]` and `[triton]`. The timing records they read were cut from rows
  timed on an earlier deposit-repair route, and no timing campaign has run on
  these files yet. Dispatch does not read those records, and correctness does
  not rest on them ([CHANGELOG](CHANGELOG.md), "Pending: fused-product
  timing").
- The two tests belong to the certification suite (pytest marker
  `certification`), which a run without `-m` does not select and
  `python -m pytest meep_gpu parity/meep_gpu -m certification` runs; there
  they are reported as failed, not skipped or removed.
  `tools/ci/pending_certification.txt` lists them, and
  [Running the tests](docs/development/testing.md) explains them.

[Running the tests](docs/development/testing.md#what-a-run-of-the-suites-shows-in-this-preview) records what a run of the suites shows
on these files.

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

[CHANGELOG](CHANGELOG.md), "Known limits", has a longer list as of 0.9.0;
where it and this section differ, this section is current.

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
Zeng and Yanlin Dou, meep-gpu, version 0.9.2, 2026), and cite MEEP, which
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
  version = {0.9.2},
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
