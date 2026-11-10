# meep-gpu: GPU time stepping for MEEP simulations

`meep-gpu` takes a simulation that your own
[MEEP](https://github.com/NanoComp/meep) installation has built and time-steps
it on one NVIDIA or Apple GPU. MEEP still does the setup and the
post-processing; the fields and the monitor results come back through a result
object.

**Independent add-on.** This project is not affiliated with or endorsed by the
MEEP developers. It installs beside an unmodified MEEP and does not replace or
modify the `meep` module.

**Status: beta (1.0.1).** The accompanying paper cites this release.
Interfaces and record formats may still change in later releases. See
[Certification status](#certification-status-of-this-release). The import name
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
three-dimensional grids, or on short runs. An NVIDIA GPU that no kernel table
supports takes the array path unless `MEEP_GPU_HARDWARE=any`. Most of these
cases have not been timed against MEEP (the list below says what each rests on).

The 1.0 timing compared the package with MEEP on the identical problem, on the
bytes of 1.0, on 2026-10-16. On one 3-D case, per step, against MEEP at
its fastest rank count on the same machine:

- **One NVIDIA RTX A6000**, against the machine's 48 CPU cores: from 512,000 to
  7,077,888 cells the GPU stepped 1.65 to 5.04 times as fast as MEEP on the
  Triton route (what a default 1.0 run took on a certified NVIDIA host with
  Triton 3.1.0 installed), and 1.45 to 5.81 times as fast on the CUDA-only route.
- **One Apple M1 Max**, against MEEP at 8 ranks on its own CPU: 0.82 times at
  110,592 cells, rising to 1.69 times at 512,000 and 2.72 times at 7,077,888
  cells. These medians enter under a timing check that was adopted after the runs
  had been seen; under the earlier check only the largest size entered
  ([the change](docs/guides/will-it-help.md#the-timing-check-and-the-change-made-after-these-runs)).

Read those numbers with their conditions:

- one GPU against the CPU of the same machine, one machine per platform;
- single precision on both sides, against MEEP 1.33.0 as the 1.0 release's reference
  environment builds it (single precision, no floating-point contraction, a
  portable binary), not a native-tuned MEEP;
- this one 3-D case (a dielectric sphere under PML) with one flux monitor, the
  same simulation on both sides;
- per step, not counting the one-off lift before the first step, which took
  about 7 s at 512,000 cells and 49 s at 7,077,888 cells on the NVIDIA host:
  there, on the Triton route and derived from the lift and the step rates, the
  GPU run is ahead after about 5,900 steps at 512,000 cells and 460 at 7,077,888,
  and at 2,097,152 cells the lift is shorter than MEEP's own start-up, not
  counting the GPU's own cost of its first steps;
- MEEP at the fastest rank count tried (1 to 64 ranks on the NVIDIA host; MPI
  ranks with OpenMP threads each read up to 2.7 % faster there, but those runs
  are struck by a defect of the timing harness);
- the CUDA-only route is the hand-written CUDA kernel table serving alone, the
  table a host without Triton uses (timed on a host that also had Triton).

The 1.0.1 code was timed on the same M1 Max from 2026-10-31 to 2026-11-01, against MEEP 1.35.0 at 8
ranks, per step, from 512,000 to 7,077,888 cells: 2.24 to 2.99 times as fast on the same case, 1.63
to 2.86 times with eight DFT objects, and 2.04 to 2.64 times on a slab with one Lorentz pole. A
fourth case, an Hz source, was timed and its medians are struck ([the
table](docs/guides/will-it-help.md#the-101-timing-on-the-same-laptop)). The 1.0.1 code was timed on
one RTX 3090 from 2026-10-29 to 2026-10-30, against MEEP 1.35.0 on the same desktop's CPU at each
row's fastest rank count (4 or 8), per step, from 512,000 to 7,077,888 cells: 4.86 to 21.88 times as
fast on the same case, 3.89 to 23.46 times with eight DFT objects, 5.78 to 16.71 times on the
Lorentz slab and 3.02 to 22.25 times with an Hz source; 23 of 24 medians enter ([the
table](docs/guides/will-it-help.md#the-101-timing-on-one-rtx-3090)). These ratios rest on MEEP's
rate on that desktop, which barely scales from 4 to 8 ranks at the two larger sizes and varied
between sessions: at the highest median seen, the same case at 7,077,888 cells reads 15.05 and 17.55
times against the table's 18.76 and 21.88. They license no claim across hosts or vendors.

[Will it help?](docs/guides/will-it-help.md) has the tables and every condition, and,
for the RTX A6000 and the M1 Max, the lift and the break-even step count at each size.
1.0.0 shipped the 2026-09-21 dispatch-preference records. 1.0.1 ships compute capability
8.6's records cut on one RTX 3090 and the Metal table's record cut on one M1 Max; the
records the 1.0 timing cut on the RTX A6000 are retired.

Where it does not help, or has not been measured:

- **Two-dimensional problems and small three-dimensional problems are slower
  than on MEEP.** No two-dimensional problem has been timed against MEEP on a
  matched problem. In 3-D the ratio falls as the grid gets smaller (on the
  Apple laptop 0.82 times MEEP at 110,592 cells; on the NVIDIA host of the 1.0
  timing nothing below 512,000 cells was timed), and the dispatched step has a
  fixed cost per step. Measure your own case.
- **Small grids on a Mac.** On the one Mac measured, on pre-1.0 code, the NumPy reference beat a default
  run below a 2-D crossover of 102,400 to 230,400 cells; on 1.0 code, not at 110,592 3-D cells, none smaller timed.
- **Short runs.** Below the break-even step count, MEEP finishes first. Run to
  its own end time (1,400 to 3,360 steps), the timed case finishes first on
  MEEP at 512,000 cells and on the GPU from 2,097,152 cells up, on the NVIDIA
  host of the 1.0 timing (derived from the lift and the step rates, not timed end
  to end).
- **An NVIDIA GPU that no kernel table supports** takes the array path, CuPy on the
  GPU: on the RTX A6000 of the 1.0 timing, slower than MEEP's fastest rank count at each of the
  three sizes (0.15 to 0.67 times); on one RTX 3090, on the 1.0.1 code, against MEEP on that
  desktop, 0.38 to 0.39 times at 512,000 cells on three cases, 1.00 on the Lorentz slab, and 2.37 to
  5.30 times from 2,097,152 cells up. No run separates the card from the host's MEEP: measure your
  case. A supported NVIDIA GPU outside the certified set (one the installed CuPy or Triton can run
  on) runs the compiled kernels by default, uncertified, and says so ([Supported and certified
  hardware](#supported-and-certified-hardware)); no such card has been timed.
- **An Apple GPU, PyTorch or macOS build outside the certified environments**
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
device that no kernel table supports. A line starting `FAILED:` names the section
of [INSTALL.md](INSTALL.md) to read.
[INSTALL.md](INSTALL.md) also covers installing into an environment that already
holds MEEP, what the host must provide, and troubleshooting.

The device libraries come in two recipes, both declared as extras of
`pyproject.toml`. The certified recipe (`nvidia-certified`, `apple-certified`)
pins the exact versions the certification was cut on: bit-identical to the
reference MEEP on certified hardware, except a run with an `EigenModeSource`,
whose mode the lift solves again through MPB, so that it can differ from MEEP in
the same process or in another (see [Limits](#limits)).
The supported recipe (`nvidia`, `nvidia-cuda11`, `apple`) takes newer releases:
it runs the fast path, and a run on a newer version is recorded as not
certified. The environment files install the certified versions of the device
libraries, beside a conda-forge MEEP that leaves an NVIDIA run uncertified. The
table of both is in [INSTALL.md](INSTALL.md#certified-and-supported-versions).

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

On the certified M1 Max, in the environment of Route 2 with the file's
earlier pin (MEEP 1.33.0 from conda-forge, a double-precision build), the script
prints, between MEEP's own setup lines:

```text
supported: True []
steps: 840  MEEP time: 35.0
step path: fused
flux: 9.709557e-06 7.775078e-02 -3.753959e-06
max |Ez|: 1.436076e-01
```

The last digits of `flux` and `max |Ez|` depend on the MEEP build the lift
reads from: with the environment file as it is now (MEEP 1.35.0), the same Mac
printed `flux: 9.709682e-06 7.775076e-02 -3.753865e-06` and the other lines
unchanged (2026-10-27).
[`examples/expected/quickstart.txt`](examples/expected/quickstart.txt) was
recorded against a single-precision MEEP built from source (its header says
so); with the double-precision MEEP of the environment files the printed
values agree with it to about 1e-4 (relative), and
`python examples/run_examples.py --check` compares within a tolerance, not
digit for digit.

The package also prints to standard error, at most once per process for each
distinct message. With a double-precision MEEP the first line is a note on
precision, and the second says what served the run (its environment marks shown
as this release prints them in the certified environment):

```text
meep_gpu: this MEEP build is double precision and the engine steps single precision (float32 fields, complex64 for a complex run). The lift proceeds. Expect agreement with a MEEP run of the same simulation at the level of single-precision rounding, not of double precision: final fields differed by 1.5e-6 to 5.1e-4 (relative L2) on the 4 simulations compared
meep_gpu: step path fused; 4/7 slots (step_B,update_H,step_D,update_E) via PML,fused magnetic B/H pair,offdiag; table metal; policy flush (requested flush), installed by dispatch OVERRIDING this run's own keep resolution; torch 2.10.0 certified, metal frontend metalfe-32023.921.5 certified; Apple M1 Max (applegpu_g13s) certified
```

The slot text is what the installation check, on this release's kernels and
ledgers, printed on the certified M1 Max on 2026-11-02, under macOS 27.2, where
the frontend part of the
line reads `metal frontend metalfe-32023.922.4 UNCERTIFIED` and one
`meep_gpu: NOTE` line follows it.

The second line, part by part:

| Part of the line | What it says |
|---|---|
| `step path fused` | compiled kernels served at least one sub-step; `step path array` would mean none did |
| `4/7 slots (...) via ...` | four of the seven sub-steps of one time step ran as compiled kernels, and which kernels; the other three ran on the array path |
| `table metal` | the kernel table that served: `triton` or `cuda` on NVIDIA hardware, `metal` on Apple hardware |
| `policy flush ..., installed by dispatch OVERRIDING ...` | the float32 subnormal policy the kernel table is certified under, installed for the whole process; this is why a default run and a `prefer_gpu=False` run can differ in the last bits |
| `torch 2.10.0 certified, metal frontend ... certified` | the PyTorch version and the Metal frontend were read, and every certification record the Metal table cites names them |
| `Apple M1 Max (applegpu_g13s) certified` | the GPU, and its architecture as Metal names it, which every certification record the Metal table cites names (`applegpu_g13s`, or `applegpu_g16g` on an M4) |

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
- **NVIDIA.** An NVIDIA GPU is supported by a kernel table when the installed
  toolchain can run that table on it: the hand-written CUDA table when the
  installed CuPy's NVRTC can compile for the device, the Triton table when Triton
  3.1 or later compiles and launches a probe kernel on it. The kernels run on such
  a device by default and say whether it is certified. A device neither table can run on is
  refused by name unless `MEEP_GPU_HARDWARE=any`; a Triton that failed the probe stays dropped.

Two NVIDIA compute capabilities and two Apple environments are certified.

| Route | Certified identity | Certified on |
|---|---|---|
| NVIDIA, Triton kernels | compute capability 8.6, 8.9 with Triton 3.1.0, each beside its certified stack | 8.6: one RTX 3090 (CUDA driver 13.4), with an RTX A6000's earlier runs (driver 12.3); 8.9: one RTX 4080 SUPER (CUDA driver 13.5); each on CuPy 13.5.1, NVRTC and CUDA runtime 11.8, Linux x86_64, NumPy 2.2 and MEEP 1.35.0 in single precision |
| NVIDIA, hand-written CUDA kernels | compute capability 8.6, 8.9, each beside the same stack | the same machines and stack |
| Apple, Metal kernels | GPU architecture `applegpu_g13s`, `applegpu_g16g` with PyTorch 2.10.0 and `PYTORCH_MPS_FAST_MATH` unset or `0`, each beside its certified Metal frontend | `applegpu_g13s`: one M1 Max, Metal frontend `metalfe-32023.921.5` (macOS 27.0.1); `applegpu_g16g`: one M4, Metal frontend `metalfe-32023.886.1` (macOS 26.6.2) |

A Metal environment is those four facts; PyTorch and the frontend are certified
per architecture. The GPU architecture is the one Metal compiles for
(`MTLDevice.architecture`, read on macOS 14 or later); it tells apart GPU
generations that share a Metal GPU family, such as M3 and M4. The Metal frontend
belongs to macOS: every Mac on one macOS build reports the same one, so a system
update can move a Mac out of its certified environment. The certified M1 Max has
moved: since its update to macOS 27.2 (Metal frontend `metalfe-32023.922.4`) it
runs the kernels supported and uncertified, until a later release certifies that frontend.
`PYTORCH_MPS_FAST_MATH=1` compiles the kernels in fast-math mode;
on one M1 Max it changed 274,523 of 1,048,576 float32 divides and 327,338 of
1,048,576 square roots (2026-10-02), and no certification ran with it.

Five machines have been measured for these records: the RTX 3090, the RTX 4080 SUPER, the M1 Max, the M4,
and the RTX A6000 that 1.0.0 was certified on, whose runs stay in the stack
record, in CUDA certification blocks and in 6 Triton entries
([Certification status](docs/development/certification.md#certification-status-of-this-release)).
Other devices that report compute capability 8.6 or 8.9 beside that capability's certified stack,
and other Macs whose GPU reports `applegpu_g13s` or `applegpu_g16g` with that
architecture's certified PyTorch and Metal frontend and fast math off, are certified by the identity
rule and have not been measured.

The NVIDIA rows are not a hard-coded list: each is derived from the per-architecture
records the kernel ledgers carry, so another compute capability becomes certified by
running the gates on a card of that architecture and writing its records, with no change
to the package ([Certifying another compute capability](docs/development/certification.md#certifying-another-compute-capability)).
The Apple row is read the same way, from the 45 certification records the Metal
table cites: a fact is certified when all 45 name it (PyTorch and the frontend in the
runs of the GPU's own architecture). Those records hold one run per GPU architecture,
so `applegpu_g16g`'s certification was added beside `applegpu_g13s`'s rather than replacing it.

**What any other NVIDIA device gets by default.** On a supported device or
Triton version that is not certified (a compute capability other than 8.6 and 8.9, or a
Triton release other than 3.1.0) the compiled kernels run and
the run is labelled uncertified. Its status line marks the device
`supported, UNCERTIFIED`, and the run prints one line per process, on standard
error, naming what was read and what is certified:

    meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU compute capability 8.0 (certified: 8.6, 8.9). They were dispatched because this NVIDIA GPU and toolchain are supported but not certified bit-identical (supported means for the Triton table, Triton >=3.1 that compiled and launched the health probe on this device, and for the hand-CUDA table, a CuPy whose NVRTC can target this device's architecture; the warm pass then compiles every slot's kernel before the first step), and MEEP_GPU_HARDWARE=certified would restrict the kernels to certified ones; the gates with no run on this GPU are counted in the dispatch record under uncertified.served[], which names the first five in welds_without_a_live_run_here; compare the results with a prefer_gpu=False run of the same simulation before relying on them

`driver.fast_path_report()` carries `certified: False` with what was read under
`uncertified.served`. Where one NVIDIA kernel table is certified for the device
and toolchain and the other is only supported, both run and the run is
uncertified on the supported one; `MEEP_GPU_HARDWARE=certified` runs the certified
table alone. A device or library version neither table supports (no NVRTC target
for the device, and no Triton 3.1 or later that can launch the probe kernel)
is refused by name and the run takes the array path, CuPy on the GPU. That run is correct and slow
([Will it help?](#will-it-help)). A device whose identity cannot be read is not
refused. `driver.fast_path_report()` states which path served a run and why.

**What any other Mac gets by default.** The Metal kernels run: its Apple GPU is
supported. Another GPU architecture (that of an M2, M3 or M5, for example),
another PyTorch version, another Metal frontend, or `PYTORCH_MPS_FAST_MATH` set
to any value but `0` is not certified; the run steps on the kernels all the
same. Its status line marks an Apple GPU that is not a certified one
`supported, UNCERTIFIED` (`Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED`),
and a certified GPU `certified` whatever else is uncertified. The run prints
one line per process, on standard error, naming what was read and what is
certified:

    meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU architecture applegpu_g15p (certified: applegpu_g13s, applegpu_g16g). They were dispatched because this Apple GPU is supported, and MEEP_GPU_HARDWARE=certified would restrict the Metal kernels to certified environments; compare the results with a prefer_gpu=False run of the same simulation before relying on them

`driver.fast_path_report()` carries `certified: False`, with each fact that is
not certified under `uncertified.served`, and `environment.device_supported`:
`True` on an Apple GPU, `False` on a GPU that is not Apple's, `None` when
neither its architecture nor its name was read. A GPU that is not Apple's is not
supported; the Metal kernels run on it too unless
`MEEP_GPU_HARDWARE=certified`, its status line carries no `supported`, and the
note gives as the reason
`the Metal table runs on environments outside the certified set unless MEEP_GPU_HARDWARE=certified`.
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

**`MEEP_GPU_HARDWARE`.** The switch names which hardware the compiled kernels
may run on, by value: `certified`, `supported` (the default, when unset) or
`any`. It is strict: any other value (`1`, `0`, `true`, `Any`, an empty string)
is refused by name and the run takes the array path, on every kernel table and
on a certified host as well.

    MEEP_GPU_HARDWARE=certified python my_simulation.py   # every table: the kernels on certified identities only
    MEEP_GPU_HARDWARE=supported python my_simulation.py   # the default: certified and supported identities
    MEEP_GPU_HARDWARE=any python my_simulation.py         # NVIDIA: the kernels on an unsupported device too

What the three values do depends on the table:

- **NVIDIA tables.** `supported` runs a supported identity uncertified and
  refuses an unsupported one, as above. `certified` restricts the compiled
  kernels to certified devices and toolchains: anything else, a device whose
  compute capability could not be read included, is refused by name on its own
  table (a stack fact on both), and with no table left the run takes the array
  path, CuPy on the GPU. `any` also dispatches on a device or library version
  outside what is supported, recorded `supported: False`; a Triton that failed the
  probe stays dropped. It admits identities and nothing else; every other rule of
  dispatch applies as before. The refusal of an unsupported identity names the switch.
- **Metal table.** `supported` and `any` run an uncertified environment, as
  above. `certified` restricts the Metal kernels to the certified environment: a
  fact that is not certified is refused by name, and so is one that could not be
  judged, and the run takes the array path, NumPy on the host CPU.

`MEEP_GPU_HARDWARE` replaced `MEEP_GPU_ALLOW_UNCERTIFIED`, whose `0` and `1`
did not say what they did. The old name is refused by name, set to anything,
and the refusal names the value that does what it did:
`MEEP_GPU_ALLOW_UNCERTIFIED=0` is `MEEP_GPU_HARDWARE=certified`,
`MEEP_GPU_ALLOW_UNCERTIFIED=1` is `MEEP_GPU_HARDWARE=any`, and unset is
`supported`.

A run on an uncertified identity prints one line per process, on standard
error, saying that the kernels are not certified on this host and naming what
is certified, and `driver.fast_path_report()` carries `certified: False` with
the identity that was read. Such a run carries no certification: compare its
results with a `prefer_gpu=False` run of the same simulation before relying on
them (with an `EigenModeSource`, the two lifts can also differ by the mode each
solves: [Limits](#limits)). Uncertified runs have been exercised with substituted identities
(compute capabilities 6.1, 7.5, 8.9 and 9.0, which a CUDA 11.8 NVRTC targets, and
10.0 and 12.0, which it cannot, a supported Triton release other than 3.1.0 and
one older than 3.1, another GPU architecture, another PyTorch version,
another Metal frontend, `PYTORCH_MPS_FAST_MATH=1`) and
on real identities outside the list: PyTorch 2.14.0 on one M1 Max and Triton 3.8.0
on one RTX A6000 (2026-10-18); compute capability 8.9, run on one RTX 4080 SUPER
before its records landed (123 of 123 NVIDIA units), is now on the list, certified
by those records.

## Precision

The engine steps single precision, whatever precision your MEEP was built
with: fields come back as `float32`, or `complex64` for complex runs.

The MEEP most users install is double precision. The packaging recipe of the
conda-forge builds has no single-precision option (0 of the 428 builds
published for Linux x86_64 and Apple silicon as of 2026-10-27 can be single
precision), and
single precision needs a source build with `--enable-single`. Every certification and timing
measurement behind this package, and every agreement figure that names no other build, was made
against MEEP built from source in single precision: 1.33.0 for 1.0, 1.35.0 for 1.0.1.

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
  on one NVIDIA host. With MEEP 1.35.0 from conda-forge, the release the
  environment files install, they ran on one Apple host and one NVIDIA host
  (2026-10-27), and the reference differed from that MEEP by 2.2×10^-6,
  2.7×10^-6 and 3.0×10^-4 of the final field on the same 3 examples, on both.
  `INSTALL.md`, "Tested MEEP versions and builds", lists every run. No release
  other than 1.33.0, 1.34.0 and 1.35.0 has been run.
- No speed has been measured against a double-precision MEEP.

Read what you have with
`python -c "import meep as mp; print(mp.__version__, mp.is_single_precision())"`.

## Bit identity, and agreement with MEEP

The kernels are bit-identical to the package's own array path on the same
device under the same subnormal policy. The array path is CuPy on the device on
an NVIDIA host, and NumPy on the host CPU on an Apple host. That is what the
gates certify, on the certified identities; on a supported identity outside
them, where the kernels also run by default, the certification records hold no
run that measured it.

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
place. [Examples](examples/README.md) shows how the three runs are made. A run
with an `EigenModeSource` can also differ between the two runs for another
reason: each lift solves the source's mode again through MPB, two solves need
not agree, and every word the source reaches then differs
([Compatibility](docs/guides/compatibility.md#an-eigenmodesource-can-differ-from-meep)).

Agreement with MEEP is a separate measurement, and it is a tolerance, not an
identity. Against MEEP 1.33.0 in single precision, the simulations built by 58
of 60 of MEEP's example scripts agree within 2.4×10^-7 to 1.6×10^-5 (relative
L2 over the whole field); the largest grid compared had 338,688 cells. That was
measured on the NumPy reference on 2026-08-09, not through the kernels and not
at the timed sizes. No identity between NVIDIA and Apple results is claimed.
[Floating-point contract](docs/design/floating-point.md) has the detail.

## Certification status of this release

Stated plainly:

- **Two compute capabilities and two Apple environments are certified, each measured
  on one card**: compute capability 8.6 on one NVIDIA RTX 3090 and 8.9 on one RTX 4080 SUPER, for both NVIDIA
  kernel tables, under `keep` and under `flush`, and `applegpu_g13s` on one Apple M1
  Max and `applegpu_g16g` on one Apple M4 (macOS 26.6.2), for the Metal table, under `flush`. Other cards of that identity read as
  certified without having been measured; a supported GPU of any other identity
  runs the kernels by default, recorded as uncertified, as the certified M1 Max does
  since its update to macOS 27.2 (Metal frontend `metalfe-32023.922.4`; the
  records name 32023.921.5, macOS 27.0.1).
- **The 1.0.1 certification round ran against MEEP 1.35.0**, in one reference environment per
  platform: every device gate unit holds (123 of 123 NVIDIA units on the RTX 3090, 62 of 62 Metal
  units on the M1 Max), the route campaigns of the three kernel tables released (Triton 3 of 3 legs,
  hand-written CUDA 4 of 4, Metal 4 of 4 with residency held), the go/no-go for dispatch on by
  default read GO (9 of 9 whole simulations dispatched), and the direct corpus check against MEEP
  holds (Metal 220 of 220 planned rows; NVIDIA 220 of 220 on each of 4 legs). The ledgers shipped
  here were bound from those runs, and beside them those of compute capability 8.9 (one RTX 4080 SUPER) and GPU architecture `applegpu_g16g` (one M4: 62 of 62 Metal units, the Metal route campaign 4 of 4 legs, residency held). On commit `404d432d0b34`, on the RTX 3090: the direct corpus
  check holds 220 of 220 rows on each of 4 legs, against the second Linux build; the dispatch checks
  collected 12 of 12 ladder keys and 124 of 124 fused keys, and no key owes anything further. On
  commit `c3e11358efbc`, the tip before the 1.0.1 documentation and the version bump, re-read with
  no GPU: the bound ledgers hold at the tree's bytes (every hand-written CUDA and Metal entry, 62 of
  68 Triton entries with the other 6 uncited, every run digest live), the three dispatch records
  hold, the direct corpus check carries 766 of 880 NVIDIA row-legs (the rows that held against the
  primary Linux references; the other 114 held against the second build only, as expected, and are
  not carried) and the dispatch checks read valid at 12 of 12 ladder keys and 124 of 124 fused keys;
  the NVIDIA route legs' own carry was not re-read there, their artifacts being on the GPU host. Read later from the compute capability 8.6 legs themselves, that route carries on the bytes shipped here: 8 of 8 legs (Triton 3 of 3, hand-written CUDA 5 of 5).
- **The 1.0 certification round ran on the bytes of 1.0**, against MEEP 1.33.0, in
  one reference environment per platform built by the repository's own script, on
  2026-10-14 and 2026-10-15, on one RTX A6000 and the M1 Max: every device gate
  unit held (123 of 123 NVIDIA, 62 of 62 Metal), the route campaigns of the three
  kernel tables released, residency held on the Metal table, and the go/no-go for
  dispatch on by default read GO.
- **The ledger-contract tests** form the certification suite (pytest marker
  `certification`), which a run without `-m` does not select and
  `python -m pytest meep_gpu parity/meep_gpu -m certification` runs. The tests it
  still fails on are listed in `tools/ci/pending_certification.txt`, each under
  the step that removes it, and [Running the tests](docs/development/testing.md)
  explains them; none is deselected or removed. 35 are listed. When the suites last
  ran, on commit `c3e11358efbc` in the Apple reference environment and in the
  continuous-integration job's view, 33 of the 35 failed, and in the Apple run the
  other 2 skipped because their evidence is not in the repository.
- **Timing is not part of certification.** The 1.0 timing ran on the bytes of 1.0 on 2026-10-16,
  after that round, and the 1.0.1 timing on one M1 Max from 2026-10-31 to 2026-11-01; the speeds
  against MEEP above are their figures, each labelled. The 1.0.1 timing on one RTX 3090 ran from
  2026-10-29 to 2026-10-30, and its dispatch checks on `404d432d0b34` from 2026-11-01 to 2026-11-02.
  1.0.0 shipped the 2026-09-21 dispatch-preference records. 1.0.1 ships compute capability 8.6's
  records cut on one RTX 3090 and the Metal table's record cut from one M1 Max's 72 fused-product
  rows (72 of 72 admitted); the records the 1.0 timing cut on the RTX A6000 are retired. Compute capability 8.9 has no record of its own: its default reads 8.6's bracketed-veto prior, the nearest timed capability's, and never the record itself. GPU architecture `applegpu_g16g` has no record of its own: its default reads `applegpu_g13s`'s, the nearest timed Apple architecture, whose prior gives way nowhere (that record measures no bracketed veto below its keeps): on `applegpu_g16g` the default dispatches every fused pair the shape rule composes.

[Certification status](docs/development/certification.md#certification-status-of-this-release)
has every figure of the 1.0.1 certification round and points to the 1.0 round's, and
[Running the tests](docs/development/testing.md#what-a-run-of-the-suites-showed-in-the-090-preview)
what a run of the suites showed in three environments at the 0.9.0 preview.

## Limits

- One simulation runs on one GPU, in one process. There is no MPI and no
  multi-GPU domain decomposition; a lift in a process that is one of several
  MPI ranks is refused by name.
- Single precision only.
- It covers a measured subset of MEEP. For 1.0, the lift accepted the simulations
  built by 60 of 65 example scripts and 134 of 233 test methods of MEEP 1.33.0's own
  corpus; over MEEP 1.35.0's scripts it accepts 60 of 65 example scripts and 164 of
  274 test methods (2026-11-02, on the NumPy reference: 0 of the 229 verdicts shared
  with 1.33.0 moved, and the test corpus grew). What it cannot serve it refuses with the reason
  ([Compatibility and refusals](docs/guides/compatibility.md)).
- Monitors are accumulated, and sources deposited, on the array path. Monitors
  are not held on the device.
- Runs that store complex fields (Bloch-periodic, cylindrical with m ≠ 0) dispatch
  where a released arm covers the sub-step; none has been timed against MEEP.
- The lift precedes the first step and runs in one process, MEEP's own
  initialization included.
- What has been timed against MEEP on the identical problem: on one RTX A6000, one three-dimensional
  case at three sizes up to 7,077,888 cells (the 1.0 code), and on one RTX 3090, 4 cases at three
  sizes up to the same size (the 1.0.1 code; 23 of 24 medians enter, 1 of 24 is struck); on one M1
  Max, that case at five sizes up to the same size (the 1.0 code), and 4 cases at three sizes up to
  the same size (the 1.0.1 code; 9 of 12 medians enter, 3 of 12 are struck).
- A run that dispatches compiled kernels changes the float32 subnormal policy
  of the whole process, and on the Apple host measured the policy was still
  installed after `close()`.
- A run with an `EigenModeSource` is not bit-identical to MEEP in every run. The
  lift solves the source's mode again through MPB, whose eigensolver chooses
  exact or approximate line minimization from `clock()` timings, so the solved
  mode, MEEP's own included, can differ in its last bits from one solve to the
  next, and every word the source reaches then differs. The GPU steps the lift's
  own solve word for word; against MEEP the run can differ in the same process or
  in another.
  Measured with MEEP 1.35.0 and MPB 1.12.0 on one x86-64 Linux host, on three
  corpus rows: the lift's mode differed from MEEP's in the same process in 4 of
  300 processes on an idle host and 47 of 936 beside other work, and MEEP's own
  words differed from its reference run in 0 of 300 and 19 of 936. On that host's
  RTX 3090, 8 of 132 dispatched runs of the corpus's eigenmode rows differed from
  the reference. It has been seen on the Apple M1 Max too. One corpus row,
  `test_mode_coeffs`'s
  `test_eigenmode_source_with_cylinders`, differs from MEEP in every process: a
  separate, deterministic exception
  ([Compatibility](docs/guides/compatibility.md#an-eigenmodesource-can-differ-from-meep)).

[CHANGELOG](CHANGELOG.md), "Known limits", has 0.9.0's longer list; this section is current.

## Relation to MEEP's planned GPU backend and to other projects

- **MEEP's planned backend.** No MEEP release has GPU support; the latest is
  v1.35.0 of 2026-10-09. A MEEP collaborator has published a design, "Optional
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
Zeng and Yanlin Dou, meep-gpu, version 1.0.1, 2026), and cite MEEP, which
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
  version = {1.0.1},
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
