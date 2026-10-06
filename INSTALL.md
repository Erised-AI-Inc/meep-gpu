# Installing MEEP and meep-gpu

`meep-gpu` does not contain MEEP and does not replace it. It imports the MEEP
that is installed in the same Python environment, reads the simulation MEEP has
built, and time-steps it on one GPU. This file says exactly how to install a
MEEP that works with it, how to install the package beside that MEEP, and how
to check the result.

Choose the route for your machine, run its commands in order, and stop at the
first one that fails: [Troubleshooting](#troubleshooting) is organized by the
message you see.

| Your machine | Route |
|---|---|
| Linux x86_64 with one NVIDIA GPU | [Route 1](#route-1-one-nvidia-gpu-on-linux) |
| Apple silicon Mac | [Route 2](#route-2-an-apple-silicon-mac) |
| No GPU, or you want the reference only | [Route 3](#route-3-no-gpu-the-numpy-reference-only) |
| You already have a MEEP environment and want to keep it | [Installing into an existing MEEP environment](#installing-into-an-existing-meep-environment) |

Version 0.9.2 is a preview. It is installed from a checkout of this repository;
it is not on a package index. What has and has not been run is stated in
[Tested MEEP versions and builds](#tested-meep-versions-and-builds), row by row.

## What you need

**On every route**

- `conda`, from [Miniforge](https://conda-forge.org/download/). The environment
  files take every conda package from the `conda-forge` channel, which is where
  MEEP's Python module is published under the name `pymeep`. MEEP is not on the
  Python package index. The environments below were created with conda 25.11.0;
  every conda command then printed that a newer version (26.7.2) exists. A
  Miniforge installed today brings conda 26, with which these steps have not
  been run.
- `git`, and a checkout of this repository. Every command below is run from
  the root of the checkout.

      git clone https://github.com/Erised-AI-Inc/meep-gpu
      cd meep-gpu

- Linux x86_64 or an Apple silicon Mac. Nothing has been run on any other
  platform. MEEP's conda packages do not run on native Windows.

**What each environment file installs**

| | Route 1, `nvidia-linux.yml` | Route 2, `apple-silicon.yml` | Route 3, `reference-cpu.yml` |
|---|---|---|---|
| MEEP | 1.33.0, conda-forge `pymeep`, serial, double precision | the same | the same |
| Python | 3.10 | 3.12 | 3.12 |
| NumPy | 2.2 | 2.4 | 2.2 |
| GPU libraries | CuPy 13.5.1 with the CUDA 11.8 runtime (conda-forge); PyTorch 2.5.1 and Triton 3.1.0 (PyPI) | PyTorch 2.10.0 (PyPI), beside OpenBLAS 0.3.34 in its pthreads build (conda-forge), so that a process holds one OpenMP runtime | none |
| Size on disk, measured | 7.7 GB | 1.0 GB | 0.5 GB |

MEEP 1.33.0 is pinned because it is the release the kernels were certified
against. MEEP 1.34.0, the newest release, has also been run with this package,
on an Apple silicon Mac and on an NVIDIA host
([Tested MEEP versions and builds](#tested-meep-versions-and-builds)). The
packaged MEEP is a double-precision build and the engine steps single precision;
[Precision](#precision) says what that means for your results.

**Route 1 also needs, on the machine itself**

- An NVIDIA GPU and its driver: `nvidia-smi` must list the device. The driver
  in use on the certified host is 545.23.08.
- A C compiler on `PATH`: `gcc --version` must work. Triton compiles a small
  extension the first time a kernel is launched.
- The driver library under the name `libcuda.so`. Many driver installations
  provide only `libcuda.so.1`; Route 1, step 3, supplies the name without
  administrator rights.
- No CUDA toolkit, according to CuPy's installation notes: the environment file
  installs the CUDA runtime libraries from conda-forge. On the one host this
  environment has been created on, Route 1 was run with the machine's CUDA
  toolkit hidden from the process (its directory empty in the process's view, no
  `nvcc` on `PATH`), and passed. A machine with no toolkit installed at all has
  not been run.

Compiled kernels are certified on devices of compute capability 8.6 only (for
example the RTX A6000, on which they were certified). Every NVIDIA GPU of compute
capability 7.0 to 9.0 is supported: on one that is not certified they run by
default and the run is labelled uncertified, because supported is not certified.
`MEEP_GPU_ALLOW_UNCERTIFIED=0` restricts them to the certified devices. On a GPU
outside 7.0 to 9.0 the package runs on its array path unless
`MEEP_GPU_ALLOW_UNCERTIFIED=1`:
[GPUs that are not on the certified list](#gpus-that-are-not-on-the-certified-list).

**Route 2 also needs**

- An Apple silicon Mac. Every Apple GPU is supported, M1 through M5 and later:
  the Metal kernels are meant to run on it, and run there by default. They are
  certified in one environment: the GPU architecture `applegpu_g13s` (an Apple
  M1 Max), PyTorch 2.10.0, the Metal frontend 32023.850.10, and
  `PYTORCH_MPS_FAST_MATH` unset or `0`. The frontend is part of macOS and cannot
  be installed; the certified Mac runs macOS 26.2. In any other environment
  (another Apple GPU, another PyTorch, another macOS build) the Metal kernels
  run by default and the run is labelled uncertified: supported is not
  certified. `MEEP_GPU_ALLOW_UNCERTIFIED=0` restricts them to the certified
  environment, and the package then runs on its array path, NumPy on the host
  CPU:
  [GPUs that are not on the certified list](#gpus-that-are-not-on-the-certified-list).
- Nothing else was needed on the certified Mac. In particular the Metal
  Toolchain component of Xcode is not required: that Mac does not have it.

## Route 1: one NVIDIA GPU on Linux

From the root of the checkout ([What you need](#what-you-need)):

1. Create the environment from the file and activate it.

       conda env create -n meep-gpu -f environments/nvidia-linux.yml
       conda activate meep-gpu

2. Install this package into it. Use no extras here: the environment file has
   already installed the GPU libraries, and an extra would add a second CuPy.

       python -m pip install .

3. Give Triton the driver library under the name it links against. Triton
   links with `-lcuda`, which needs a file named `libcuda.so`; many driver
   installations provide only `libcuda.so.1`. These lines put both names into a
   directory you own. They need no administrator rights and do no harm on a
   system that already has both.

       LIBCUDA="$(/sbin/ldconfig -p | grep 'libcuda.so.1 ' | grep 'x86-64' | head -1 | sed 's/.* => //')"
       echo "$LIBCUDA"
       mkdir -p "$HOME/triton_libcuda"
       ln -sf "$LIBCUDA" "$HOME/triton_libcuda/libcuda.so"
       ln -sf "$LIBCUDA" "$HOME/triton_libcuda/libcuda.so.1"
       export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda"
       export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"

   The `echo` must print the path of the driver library, for example
   `/lib/x86_64-linux-gnu/libcuda.so.1`. If it prints an empty line, the NVIDIA
   driver is not installed. The two `export` lines are needed in every shell
   that runs a simulation: put them in your shell start-up file or at the top of
   your job script.

4. Run the check. On a machine with several GPUs, choose an idle one with
   `--gpu-id N`.

       python tools/check_install.py --require-gpu

   The first GPU run compiles kernels and can take a few minutes.
   [The installation check](#the-installation-check) says how to read the
   output. It must end with `OK: MEEP and meep-gpu work together on this host's
   GPU.`

5. Run the first example. It uses the first GPU; to use another one, put
   `CUDA_VISIBLE_DEVICES=N` in front of the command.

       python examples/quickstart.py

   It prints `step path: fused` when compiled kernels served the run and
   `step path: array` when the array path did.

Status of this route: run as written on 1 host (Ubuntu 20.04, one RTX A6000,
driver 545.23.08) on 2026-09-28, in a new environment made from the file, with a
new user's home directory and the machine's CUDA toolkit hidden from the
process. Steps 1 to 5 passed: the check passed with compiled kernels serving its
GPU run, and the first example printed `step path: fused`.
`examples/run_examples.py --check` passed on its 3 examples that compare with
MEEP, with the correction to its array leg that this release carries (without
it, that leg stopped on the CuPy route before comparing anything). The
environment differs from the one the kernels were certified in; the table in
[Tested MEEP versions and builds](#tested-meep-versions-and-builds) lists the
differences, and step 7 of the check names them package by package
([The certification reference environment](#the-certification-reference-environment)).

## Route 2: an Apple silicon Mac

From the root of the checkout ([What you need](#what-you-need)):

1. Create the environment from the file and activate it.

       conda env create -n meep-gpu -f environments/apple-silicon.yml
       conda activate meep-gpu

2. Install this package into it, without extras.

       python -m pip install .

3. Make sure `KMP_DUPLICATE_LIB_OK` is not set in your shell
   ([One OpenMP runtime on an Apple silicon Mac](#one-openmp-runtime-on-an-apple-silicon-mac)
   says why).

       unset KMP_DUPLICATE_LIB_OK

4. Run the check. It must end with `OK: MEEP and meep-gpu work together on this
   host's GPU.`

       python tools/check_install.py --require-gpu

5. Run the first example.

       python examples/quickstart.py

In this environment a process holds one OpenMP runtime, so the order of the
imports in your scripts does not matter
([One OpenMP runtime on an Apple silicon Mac](#one-openmp-runtime-on-an-apple-silicon-mac)).

Status of this route: run as written on 1 Mac (Apple M1 Max, macOS 26.2) on
2026-09-28, in a new environment made from the file as it is now (OpenBLAS in
its pthreads build). The check passed, compiled kernels served its GPU run, and
4 of 4 examples passed. The 6 import orders of
[One OpenMP runtime on an Apple silicon Mac](#one-openmp-runtime-on-an-apple-silicon-mac)
ran in 18 of 18 processes, `torch` first included.

This environment is not the one the Metal kernels were certified in (MEEP is
conda-forge's double-precision `pymeep`, and OpenBLAS is the pthreads build); step 7 of
the check names the differences package by package
([The certification reference environment](#the-certification-reference-environment)).
That is expected for using the package; certifying it takes the reference build.

### The newest PyTorch on a Mac

In the environment of Route 2,

    python -m pip install --upgrade torch

replaces PyTorch 2.10.0 with the newest release on PyPI (2.14.0 on 2026-09-28).
A process still holds one OpenMP runtime, so nothing aborts. That release is
not certified, and this is what it gets:

- **By default, the Metal kernels, uncertified.** The kernels are dispatched on
  PyTorch 2.14.0. On the certified M1 Max the run's status line reads
  `torch 2.14.0 UNCERTIFIED` and still marks the GPU
  `Apple M1 Max (applegpu_g13s) certified`. After it, each process prints one
  line naming what is not certified:
  `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: torch 2.14.0 (certified: 2.10.0). They were dispatched because this Apple GPU is supported, and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the Metal kernels to certified environments; compare the results with a prefer_gpu=False run of the same simulation before relying on them`.
  `result.driver.fast_path_report()` carries `certified: False`. The check
  prints `PyTorch 2.14.0: NOT certified: the records name another` and
  `compiled kernels served the GPU run: yes (kernel table: metal)`, a `NOTE:`
  line saying that this Mac's environment is supported but not the certified
  one, so the Metal kernels ran uncertified, and ends with
  `OK: MEEP and meep-gpu work together on this host's GPU.`
- **With `MEEP_GPU_ALLOW_UNCERTIFIED=0`, the array path.** The Metal kernels are
  refused by name and every run steps on the array path, NumPy on the host CPU;
  the GPU is not used. Each run says so once:
  `meep_gpu: step path array on the host CPU; dispatch refused: torch 2.14.0 is not the one every Metal weld this table cites ran on (certified: ['2.10.0']); MEEP_GPU_ALLOW_UNCERTIFIED=0 restricts the Metal kernels to certified environments`.
  A *Metal weld* is the certification record of one Metal kernel; the list is
  the PyTorch version those records name. The check prints the line twice,
  prints `compiled kernels served the GPU run: NO`, ends with
  `OK: MEEP and meep-gpu work together. The default run took the array path, NumPy on the host CPU; the Apple GPU was not used.`,
  and exits 0 when the runs agree. Whether such a run is slower than the
  compiled kernels depends on its size: on the one Mac measured, the NumPy path
  was faster below about 100,000 cells in 2-D (125,000 in 3-D) and slower above
  about 230,000 cells in 2-D
  ([docs/guides/validation-and-performance.md](docs/guides/validation-and-performance.md)).
- **To return to the certified version:** `python -m pip install "torch==2.10.0"`.

Measured on 1 Mac (Apple M1 Max, macOS 26.2, Metal frontend 32023.850.10) on
2026-09-28, with MEEP 1.33.0, Python 3.12 and NumPy 2.4 as in the environment
file and PyTorch 2.14.0, when an uncertified PyTorch took the array path unless
`MEEP_GPU_ALLOW_UNCERTIFIED=1` admitted it: with `MEEP_GPU_ALLOW_UNCERTIFIED=1`,
4 of 4 examples ran on compiled kernels (step path `fused`, table `metal`), the
final field and every flux spectrum of 3 of 3 examples were identical, byte for
byte, to the array path run in its own process with `MEEP_GPU_DISPATCH=0`, and
`examples/run_examples.py --check` passed. Without the variable, 4 of 4 examples
ran on the array path and `examples/run_examples.py --check` passed.

## Route 3: no GPU (the NumPy reference only)

The reference is the package's own NumPy implementation of the same time
stepping. It runs on any host and it is what the GPU runs are compared with. It
is not built for speed: it is for checking a lift and for comparing results,
not for production runs.

From the root of the checkout ([What you need](#what-you-need)):

1. Create the environment from the file and activate it.

       conda env create -n meep-gpu-reference -f environments/reference-cpu.yml
       conda activate meep-gpu-reference

2. Install this package into it.

       python -m pip install .

3. Run the check, without `--require-gpu`. It must end with `OK: MEEP and
   meep-gpu work together on the NumPy reference.`

       python tools/check_install.py

4. Run the first example on the reference.

       python examples/sphere_flux_3d.py --leg reference

In this environment every call needs `prefer_gpu=False`. The default of
`run_on_gpu` and `lift_simulation` is this host's GPU, and on a host without
one the default raises `RuntimeError: GPU backend requested but unavailable`
instead of falling back. `examples/quickstart.py` makes a default call, so it
raises here; that is the intended behavior.

Status of this route: run as written on 1 Mac (Apple M1 Max) on 2026-09-28; the
check passed and the example ran. On Linux x86_64 the environment file has been
solved, not installed: not tested.

## The installation check

`tools/check_install.py` uses only the package's public entry points. Its own
process imports neither MEEP nor `meep_gpu`; every step runs in a child process
that imports NumPy, then MEEP, then `meep_gpu`. On a Mac with PyTorch installed,
one more child process, run first, imports NumPy and MEEP only, so that the
OpenMP runtimes can be read before anything imports PyTorch. Run it without
`mpirun`.

| Part of the output | What it states |
|---|---|
| 1. Python and NumPy | the versions of the interpreter that ran the check |
| 2. MEEP | version, serial or MPI build and the number of processes, single or double precision; how many of the 148 MEEP names the lift reads are present, and how many of the 10 MEEP constants it relies on have the expected value |
| 3. meep-gpu | the installed version and where it was imported from |
| 4. GPU route of this host | `NVIDIA GPU, through CuPy`, `Apple GPU, through PyTorch`, or none and what is missing; the GPU libraries that are installed; on a Mac with PyTorch, the OpenMP runtimes a process will hold |
| 5. One small simulation | a dielectric sphere in a 24 × 24 × 24 cell with absorbing boundaries and one flux monitor, stepped with the package's default (`prefer_gpu=True`, this host's GPU route), on the NumPy reference and by MEEP itself, each in its own process |
| 6. Result | on an NVIDIA host, for each kernel table, whether the device is supported and on the certified list, and the same for the Triton version; on a Mac, whether the GPU (name and architecture), PyTorch, the Metal frontend and, when it is set, `PYTORCH_MPS_FAST_MATH` are those of the certified environment, and, for a GPU that is not certified, whether it is supported (every Apple GPU is); whether compiled kernels served the default run, and where it stepped when they did not; the relative difference between the runs, for the final field and for the flux spectrum |
| 7. Certification reference environment | whether MEEP is the build the certification reference names (1.33.0, single precision, MPI build); then, for every numerics-relevant package of this platform's certification reference, whether this environment holds the reference's build, another one (both shown), none, or one it cannot read; then one verdict line ([The certification reference environment](#the-certification-reference-environment)). Informational: it fails the check only with `--require-reference` |

The exit status is 0 when everything that ran is in order. Otherwise the last
line starts with `FAILED:`, gives one reason, and names the section of this file
to read; the exit status is 1. Lines that start with `NOTE:` are not failures.

The last line of a passing run says where the default run stepped. Only
`OK: MEEP and meep-gpu work together on this host's GPU.` means that compiled
kernels served it. When they did not (an NVIDIA device or toolchain outside the
supported range, or, with `MEEP_GPU_ALLOW_UNCERTIFIED=0`, outside the certified
list; on a Mac with `MEEP_GPU_ALLOW_UNCERTIFIED=0`, a fact of the environment
that is not certified or cannot be judged; or `MEEP_GPU_DISPATCH=0`), the line
says so: on an NVIDIA host the run stepped on the array path on the GPU (CuPy),
and on a Mac on the array path on the CPU (NumPy), where the line states that
the Apple GPU was not used. On a supported NVIDIA GPU that is not certified, and
on a Mac outside the certified environment, the compiled kernels serve the run
by default: the last line is the `OK:` line above, and a `NOTE:` line before it
says that the GPU and toolchain (on a Mac, the environment) are supported but
not certified, so they ran uncertified.

The check fails when

- MEEP or `meep_gpu` cannot be imported, or this MEEP lacks a name or a
  constant value the lift relies on;
- the process is one of several MPI processes;
- on a Mac, PyTorch loads its own OpenMP runtime whatever the import order and
  NumPy and MEEP load another one: every process that imports PyTorch would stop
  with `OMP: Error #15`, so the check stops before starting one;
- two runs differ by more than 1 part in 10,000;
- compiled kernels were expected to serve the GPU run and it was nevertheless
  served by the array path. They are expected on an NVIDIA host where one
  kernel table can run on what was read: a certified device and toolchain, a
  supported one unless `MEEP_GPU_ALLOW_UNCERTIFIED=0`, any one under
  `MEEP_GPU_ALLOW_UNCERTIFIED=1`, or a device that could not be read unless
  `MEEP_GPU_ALLOW_UNCERTIFIED=0`; and on any Mac unless
  `MEEP_GPU_ALLOW_UNCERTIFIED=0` is set and a fact of the environment is not
  certified or cannot be judged; `MEEP_GPU_DISPATCH=0` and `MEEP_GPU_FUSED=0`
  expect none. When the reason is that Triton could not link against the NVIDIA
  driver library, the line says so and names step 3 of Route 1
  ([Troubleshooting](#troubleshooting));
- `--require-gpu` was given and the host has no GPU route;
- `--require-reference` was given and the environment does not match the
  certification reference of its platform: step 7's verdict is anything but
  "matches", or MEEP is not the reference's build (1.33.0, single precision, MPI
  build).

The end of a passing run, from Route 2. The figures are those of the run of
2026-09-28; the lines that judge the environment are shown as this release
prints them on the certified Mac:

    6. Result
       GPU route        Apple GPU, through PyTorch
       GPU              Apple M1 Max, architecture applegpu_g13s: certified
       PyTorch          2.10.0: certified
       Metal frontend   metalfe-32023.850.10: certified
       compiled kernels served the GPU run: yes (kernel table: metal)
          meep_gpu: this MEEP build is double precision and the engine steps single precision (float32 fields, complex64 for a complex run). The lift proceeds. Expect agreement with a MEEP run of the same simulation at the level of single-precision rounding, not of double precision: final fields differed by 1.5e-6 to 5.1e-4 (relative L2) on the 4 simulations compared
          meep_gpu: step path fused; 4/7 slots (step_B,update_H,step_D,update_E) via PML,fused magnetic B/H pair,offdiag; table metal; policy flush (requested flush), installed by dispatch OVERRIDING this run's own keep resolution; torch 2.10.0 certified, metal frontend metalfe-32023.850.10 certified; Apple M1 Max (applegpu_g13s) certified
       the default run (prefer_gpu=True) against the NumPy reference:
          final field differs by 0.000e+00, flux spectrum by 0.000e+00 (relative); accepted up to 1e-04: agree
       the NumPy reference against MEEP itself:
          final field differs by 1.543e-06, flux spectrum by 9.032e-08 (relative); accepted up to 1e-04: agree

    NOTE: This MEEP is a double-precision build. The engine steps single precision, so results come back as single-precision arrays (INSTALL.md, "Precision").
    OK: MEEP and meep-gpu work together on this host's GPU.

Step 7, the comparison with the certification reference, prints between the
agreement lines and the `NOTE:` lines; this release added it after that run, and
[The certification reference environment](#the-certification-reference-environment)
shows its form. The two `meep_gpu:` lines are the package's own; each is printed once per
process, so every run of your own prints them too. The first appears with every
double-precision MEEP ([Precision](#precision)). The second is the status line
of a run the compiled kernels served, and in the certified environment none of
it is a warning:

- `step path fused; 4/7 slots (...) via ...`: compiled kernels served 4 of the
  driver's 7 sub-step slots (named in parentheses), one pair of them as a fused
  product; the other 3 ran on the array path or had nothing to do in this run.
- `policy flush (requested flush), installed by dispatch OVERRIDING this run's
  own keep resolution`: the Metal kernels are certified with float32 subnormal
  values flushed to zero, so dispatch sets that for the whole process in place
  of the host's own setting ([Precision](#precision)).
- `torch 2.10.0 certified, metal frontend ... certified`: PyTorch and the Metal
  frontend are those of the certified environment.
- `Apple M1 Max (applegpu_g13s) certified`: the GPU's name and its architecture,
  as Metal reports them, and the architecture is the certified one. A fact
  outside the certified environment reads `UNCERTIFIED`, and one that could not
  be judged reads `uncertified-unknown`. On an Apple GPU that is not the
  certified one, the GPU's mark starts with `supported,`, as in
  `Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED`: every Apple GPU is
  supported, and supported is not certified
  ([GPUs that are not on the certified list](#gpus-that-are-not-on-the-certified-list),
  [Troubleshooting](#troubleshooting)).

The manual's [Reading what ran](docs/getting-started/reading-what-ran.md)
explains every `meep_gpu:` line, including the lines of a run that took the
array path, and what `driver.fast_path_report()` records.

`python examples/run_examples.py --check` is the longer check: it runs every
example on every path the host has and compares them
([examples/README.md](examples/README.md)).

## Tested MEEP versions and builds

Every row is one MEEP build on one host. "Check" is `tools/check_install.py`;
"examples" is `examples/run_examples.py --check`, which runs 4 examples, 3 of
which compare with MEEP. A row marked **not tested** has not been run; nothing
is claimed for it.

| MEEP | Build | Platform and libraries | Serial or MPI | Precision | Date tested | Result |
|---|---|---|---|---|---|---|
| 1.33.0 | source build, `--enable-single` ([Precision](#precision)) | Linux x86_64 (Ubuntu 20.04), one NVIDIA RTX A6000 (compute capability 8.6); Python 3.10, NumPy 2.2.6, CuPy 13.5.1 (`cupy-cuda11x` wheel beside a CUDA toolkit installed on the machine, runtime 11.8), Triton 3.1.0, PyTorch 2.5.1 (the build for CUDA 12.1) | MPI build (Open MPI 5.0.10), started as one process | single | 2026-09-28 (names and constants only) | The host the Triton and the hand-written CUDA kernels were certified on. 148 of 148 names and 10 of 10 constants present. Check and examples in this environment: **not tested** |
| 1.33.0 | source build, `--enable-single` | macOS 26.2, Apple M1 Max; Python 3.12.13, NumPy 2.4.3, PyTorch 2.10.0, Metal frontend 32023.850.10 | MPI build (Open MPI 5.0.10), started as one process | single | 2026-09-28 | The host the Metal kernels were certified on. Check passed: 148 of 148 names, 10 of 10 constants, compiled kernels served the run, GPU against reference 0 (final field and flux), reference against MEEP 2.5e-6 (final field) and 9.2e-8 (flux). Examples: 4 of 4 passed, the comparison with MEEP judged and within its band on 3 of 3 |
| 1.33.0 | source build by `parity/meep_gpu/build_meep_133_macos.sh` from the lock `environments/locks/meep-gpu-ref-osx-arm64.*`, in a new environment | macOS 26.2, Apple M1 Max; the lock's 138 conda and 12 PyPI packages (Python 3.12.13, NumPy 2.4.3, PyTorch 2.10.0, the OpenMP build of OpenBLAS), Metal frontend 32023.850.10 | MPI build (Open MPI 5.0.10), started as one process | single | 2026-10-04 | Built in 3 min 10 s: 138 of 138 conda packages the lock's, MEEP linked against OpenBLAS as the certified MEEP is, `pip check` clean, every assertion passed, the comparison with the reference 29 of 29 same. Check passed with `--require-reference`: 148 of 148 names, 10 of 10 constants, compiled kernels served the run, GPU against reference 0 (final field and flux), reference against MEEP 2.5e-6 (final field) and 9.2e-8 (flux), as on the certified host |
| 1.33.0 | source build by `parity/meep_gpu/build_meep_133_macos.sh`, in a new environment (the script's former solve, which this release replaced with the lock) | macOS 26.2, Apple M1 Max; Python 3.12.13, NumPy 2.4.3, PyTorch 2.10.0, Metal frontend 32023.850.10, the OpenMP build of OpenBLAS | MPI build (Open MPI 5.0.10), started as one process | single | 2026-10-02 | Check passed: 148 of 148 names, 10 of 10 constants, compiled kernels served the run, GPU against reference 0 (final field and flux), reference against MEEP 2.5e-6 (final field) and 9.2e-8 (flux), as on the certified host |
| 1.33.0 | conda-forge `pymeep`, build `nompi_py312`, from `environments/apple-silicon.yml` (Route 2) | macOS 26.2, Apple M1 Max; Python 3.12.14, NumPy 2.4.6, PyTorch 2.10.0, OpenBLAS in its pthreads build, Metal frontend 32023.850.10 | serial | double | 2026-09-28 | Check passed: 148 of 148 names, 10 of 10 constants, compiled kernels served the run, GPU against reference 0, reference against MEEP 1.5e-6 (final field) and 9.0e-8 (flux). Examples: 4 of 4 passed; the comparison with a double-precision MEEP is reported, not judged ([Precision](#precision)) |
| 1.33.0 | conda-forge `pymeep`, build `nompi_py312`, from `environments/reference-cpu.yml` (Route 3) | macOS 26.2, Apple M1 Max, no GPU library installed; Python 3.12.14, NumPy 2.2.6 | serial | double | 2026-09-28 | Check passed on the reference: 148 of 148 names, 10 of 10 constants, reference against MEEP 1.5e-6 (final field) and 9.0e-8 (flux). `examples/sphere_flux_3d.py --leg reference` ran |
| 1.33.0 | conda-forge `pymeep`, build `nompi_py310`, from `environments/nvidia-linux.yml` (Route 1) | Linux x86_64 (Ubuntu 20.04), one NVIDIA RTX A6000, driver 545.23.08, the machine's CUDA toolkit hidden from the process; Python 3.10.21, NumPy 2.2.6, CuPy 13.5.1 with the CUDA 11.8 runtime (conda-forge), Triton 3.1.0 and PyTorch 2.5.1 (PyPI) | serial | double | 2026-09-28 | Check passed: 148 of 148 names, 10 of 10 constants, compiled kernels served the run (Triton table), GPU against reference 0 (final field and flux), reference against MEEP 2.1e-6 (final field) and 8.8e-8 (flux). First example: `step path: fused`. Examples: the 3 that compare with MEEP passed, the GPU run identical, byte for byte, to the array path in 3 of 3, 0 mismatches against the expected output, with this release's correction to the examples' array leg |
| 1.33.0 | as the row above, without PyTorch and Triton (CuPy only: the hand-written CUDA kernels) | the same host | serial | double | 2026-09-28 | Check passed: compiled kernels served the run (CUDA table), GPU against reference 0, reference against MEEP 2.1e-6 (final field) and 8.8e-8 (flux). Examples: 1 run (`sphere_flux_3d`), with `CUPY_CACHE_DIR` set by hand to the package's keep-policy cache (what this release's correction to the examples does); passed |
| 1.33.0 | conda-forge `pymeep`, build `nompi_py312`, from `environments/reference-cpu.yml` (Route 3) | Linux x86_64 | serial | double | 2026-09-28 (solved only) | **not tested** |
| 1.33.0 | conda-forge `pymeep` in an existing environment ([Installing into an existing MEEP environment](#installing-into-an-existing-meep-environment)) | macOS 26.2, Apple M1 Max; Python 3.12, NumPy 2.5.3, PyTorch 2.14.0 (PyPI) installed before this package | serial | double | 2026-09-28 | Step 3's OpenBLAS switch kept PyTorch 2.14.0 and left one OpenMP runtime; the check, run with the package installed without an extra, passed on the array path, which an uncertified PyTorch took on that date (PyTorch 2.14.0 is not certified: [The newest PyTorch on a Mac](#the-newest-pytorch-on-a-mac)): reference against MEEP 1.5e-6 (final field) and 9.0e-8 (flux). `python -m pip install ".[apple]"` then replaced PyTorch 2.14.0 with 2.10.0; the check was not run again |
| 1.33.0 | conda-forge `pymeep` in an existing environment, device libraries from an extra | Linux x86_64 with an NVIDIA GPU | serial | double | — | **not tested** |
| 1.33.0 | conda-forge `pymeep`, build `mpi_mpich` | any | MPI build (MPICH), started as one process | double | — | **not tested** |
| 1.34.0 | conda-forge `pymeep`, build `nompi_py313`, in an existing environment made with `conda create -c conda-forge --override-channels pymeep`, then `python -m pip install ".[apple]"` | macOS 26.2, Apple M1 Max; Python 3.13.15, NumPy 2.5.3, PyTorch 2.10.0, Metal frontend 32023.850.10 | serial | double | 2026-09-28 | Check passed: 148 of 148 names, 10 of 10 constants, compiled kernels served the run, GPU against reference 0, reference against MEEP 1.5e-6 (final field) and 9.0e-8 (flux). First example: `step path: fused`. Examples: 4 of 4 passed. 4 test files of the package (the lift, the precision notice, the backends and the default route; 341 tests): 340 passed, 1 skipped by a declared resource, 0 failed |
| 1.34.0 | conda-forge `pymeep`, build `nompi_py310`, from `environments/nvidia-linux.yml` with the MEEP pin changed to 1.34.0 | Linux x86_64 (Ubuntu 20.04), one NVIDIA RTX A6000, the machine's CUDA toolkit hidden from the process; Python 3.10.21, NumPy 2.2.6, CuPy 13.5.1, Triton 3.1.0, PyTorch 2.5.1 | serial | double | 2026-09-28 | Check passed: 148 of 148 names, 10 of 10 constants, compiled kernels served the run (Triton table), GPU against reference 0, reference against MEEP 2.1e-6 (final field) and 8.8e-8 (flux). First example: `step path: fused`. Examples: 1 run (`sphere_flux_3d`), before this release's correction to the examples' array leg; it stopped on the CuPy route in that leg. Not run again with the correction |
| older than 1.33.0 | any | any | any | any | — | not tested with the current code, and not recommended. Two known differences: for cylindrical cells with \|m\| ≥ 2 the engine follows the behavior MEEP adopted in 1.33.0, and MEEP 1.33.0 fixed a defect of Bloch-periodic boundaries in single precision (MEEP release notes) |

One difference between the two releases concerns the package: MEEP 1.34.0
assigns `B_conductivity_offdiag` of `mp.Medium` from its own keyword, where 1.33.0
assigns it from `D_conductivity_offdiag`. The package refuses an off-diagonal
conductivity under either release, and its refusal message states both
behaviors.

How Route 1's environment differs from the one the NVIDIA kernels were
certified in, measured on 1 host in the rows above: MEEP is the packaged
double-precision build instead of a single-precision source build; CuPy and the
CUDA 11.8 runtime come from conda-forge instead of a wheel beside a system
toolkit; PyTorch 2.5.1 comes from PyPI, which brings the CUDA 12.4 runtime
wheels, instead of the build for CUDA 12.1. The versions of CuPy, Triton,
PyTorch and Python are the certified ones.

The first two rows are the MEEP builds on the 2 hosts the package was developed
and certified on: 2 of 2 are MEEP 1.33.0, built from source, single precision,
with MPI. Every agreement figure in the README and the manual was measured
against a single-precision MEEP 1.33.0, except where it names a double-precision
MEEP.

## Installing into an existing MEEP environment

Use this when you already have MEEP installed and want to keep that
environment. On an Apple silicon Mac this section has been run in 2 new
environments that stood in for an existing one (MEEP 1.33.0 and 1.34.0, table
above); on an NVIDIA host it has not been run. Run the check at the end and
believe the check.

1. Activate your environment and read what it holds.

       conda activate <your MEEP environment>
       python -c "import sys, numpy, meep as mp; print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| meep', mp.__version__, '|', 'single' if mp.is_single_precision() else 'double', 'precision |', 'MPI build' if mp.with_mpi() else 'serial build')"

   It prints one line, for example
   `python 3.12.14 | numpy 2.4.6 | meep 1.33.0 | double precision | serial build`,
   followed by MEEP's own `Elapsed run time` line at exit. The first import of
   MEEP in a new environment may also print `Matplotlib is building the font
   cache`. MEEP prints its own lines (structure set-up, `Elapsed run time`) in
   every example as well.

2. Compare it with what is known.

   | The line shows | Consequence |
   |---|---|
   | MEEP 1.34.0 | Run on both kinds of host ([Tested MEEP versions and builds](#tested-meep-versions-and-builds)) |
   | MEEP other than 1.33.0 and 1.34.0 | Not tested. The package does not check the MEEP version; the check script does, and tests the names and constants the lift relies on |
   | Python 3.13, on the NVIDIA route | Triton 3.1.0 has no wheel for Python 3.13, so the certified Triton cannot be installed. Create a new environment with Python 3.10, 3.11 or 3.12 (Route 1). `conda create -c conda-forge pymeep` without a Python version gives Python 3.13 today |
   | Python older than 3.10 | Not supported by this package |
   | NumPy | 2.2 satisfies every pin. CuPy 13.5.1 from conda-forge requires NumPy below 2.3; the CuPy 13.5.1 wheels require NumPy below 2.6 |
   | double precision | Expected for every packaged MEEP: [Precision](#precision) |
   | MPI build | Fine, started as one process: [MPI](#mpi) |

3. Install this package with the device libraries for your host. On Apple
   silicon, first make OpenBLAS the build that loads no OpenMP runtime, so that
   a process holds one
   ([One OpenMP runtime on an Apple silicon Mac](#one-openmp-runtime-on-an-apple-silicon-mac)):

       conda install -c conda-forge --override-channels "libopenblas=*=*pthreads*"

   Then run one of these lines, from the root of the checkout.

       python -m pip install ".[apple]"            # Apple silicon: PyTorch 2.10.0, replacing any other PyTorch
       python -m pip install ".[nvidia-cuda11]"    # NVIDIA, CUDA toolkit 11.2 to 11.8 installed on the machine
       python -m pip install ".[nvidia]"           # NVIDIA, CUDA toolkit 12.x installed on the machine
       python -m pip install .                     # no device library: keeps the PyTorch already here, if any

   The last line installs no device library. With no PyTorch in the
   environment that is the NumPy reference only. On a Mac that already has a
   PyTorch you want to keep, newer than 2.10.0, use it instead of the `apple`
   extra: the check then reports `Apple GPU, through PyTorch`, and runs dispatch
   the Metal kernels, labelled uncertified
   ([The newest PyTorch on a Mac](#the-newest-pytorch-on-a-mac)).

   The two NVIDIA extras install a CuPy wheel, PyTorch 2.5.1 and Triton from
   the 3.1 series. A CuPy wheel contains no CUDA libraries: it loads them from
   the CUDA toolkit on the machine, which it looks for in `CUDA_PATH`, then
   beside `nvcc` on `PATH`, then in `/usr/local/cuda`.

   On an NVIDIA machine with no CUDA toolkit, take CuPy and the CUDA runtime
   from conda-forge instead, and the rest from PyPI:

       conda install -c conda-forge --override-channels cupy=13.5.1 cuda-version=11.8
       python -m pip install . "torch==2.5.1" "triton==3.1.0"

4. On the NVIDIA route, do step 3 of Route 1 (`libcuda.so`).

5. Run the check, then the first example.

       python tools/check_install.py --require-gpu     # without --require-gpu on a host with no GPU
       python examples/quickstart.py                   # examples/sphere_flux_3d.py --leg reference on a host with no GPU

Rules that keep an existing environment working:

- Install conda packages with `-c conda-forge --override-channels`. MEEP's
  documentation warns that a NumPy linked against MKL, which the `defaults`
  channel supplies, can crash when MEEP calls MPB.
- Install exactly one CuPy package. `cupy` (conda-forge), `cupy-cuda11x` and
  `cupy-cuda12x` exclude each other. `python -m pip freeze | grep -i cupy` must
  print at most one line.
- Do not upgrade PyTorch on the NVIDIA route. Every PyTorch release requires
  one exact Triton version; PyTorch 2.5.1 is the release that requires Triton
  3.1.0, and installing another PyTorch replaces Triton with a version that is
  not certified.
- PyTorch 2.10.0 is the Apple version only. On Linux it requires Triton 3.6.0.
- On a Mac, keep one OpenMP runtime per process. Switching OpenBLAS to its
  pthreads build (step 3) makes any PyPI PyTorch work beside MEEP, and it leaves
  the installed PyTorch in place; the `apple` extra installed after it then pins
  the certified PyTorch 2.10.0 and replaces any other version. To keep a newer
  PyTorch, install the package without the extra (`python -m pip install .`):
  its runs dispatch the Metal kernels, labelled uncertified
  ([The newest PyTorch on a Mac](#the-newest-pytorch-on-a-mac)). The check reads
  both runtimes before anything imports PyTorch and names the fix when there are
  two.
- Leave `PYTHONPATH` unset.

## MPI

One process, one GPU. The package does not decompose a cell over MPI processes
and does not use more than one GPU for one simulation.

- **A MEEP built with MPI is fine.** Start the script as one process,
  `python script.py`, not under `mpirun`. Both builds the package was developed
  against are MPI builds (Open MPI 5.0.10) run as one process: 2 of 2.
- **Under `mpirun` with more than one process the lift is refused**, with this
  message: "this MEEP process is one of N MPI ranks; the GPU stepper is
  single-process and would step the whole cell on every rank. Run the lifting
  process without mpirun." Measured with 2 processes on 1 host (2026-09-28);
  `mpirun -np 1` is one process and is accepted. The check script refuses in
  the same way and names this section.
- **A serial MEEP** (conda-forge build `nompi`, what the environment files
  install) is one process by construction. Tested: the conda-forge rows of the
  table in [Tested MEEP versions and builds](#tested-meep-versions-and-builds).
- **The MPI variant on conda-forge is MPICH** (`pymeep=1.33.0=mpi_mpich_*`).
  It has not been run with this package: not tested.
- **Several GPUs in one machine.** Run one process per GPU, each with its own
  simulation, and give each its device: `run_on_gpu(sim, ..., gpu_id=N)`, or
  `CUDA_VISIBLE_DEVICES=N` in the environment of the process.
- **The same environment can do both.** The lift calls `sim.init_sim()` and
  nothing else, so a script can still run MEEP on the CPU, and the same
  environment can run other scripts under `mpirun` with MEEP alone.

## Precision

**The packaged MEEP is double precision; the engine is single precision.**

- Every MEEP build on conda-forge is a double-precision build: the packaging
  recipe has no single-precision option, so 0 of the 412 builds published for
  Linux x86_64 (360) and Apple silicon (52) can be single precision. Measured on
  the 1 build installed here (`pymeep` 1.33.0, `nompi_py312`, Apple silicon):
  `mp.is_single_precision()` is `False`.
- The engine steps single precision whatever the MEEP build is. Fields come
  back as `float32`, or `complex64` for a run with complex fields.
- A lift reads the precision of your MEEP and records it with the version on
  the driver (`driver.lift_record`). From a double-precision build it prints one
  line per process, on standard error, saying that the engine steps single
  precision and the agreement to expect. It refuses nothing.
  `tools/check_install.py` prints the precision as well.

Read what you have:

    python -c "import meep as mp; print(mp.__version__, mp.is_single_precision(), mp.with_mpi())"

### What agreement to expect

Agreement with MEEP is a measurement with a tolerance. It is not bit-identity.
Every figure is the L2 norm of the difference divided by the L2 norm of MEEP's
result.

| Compared with | What was compared | Agreement |
|---|---|---|
| MEEP 1.33.0, single precision | MEEP's own example scripts, fields over the whole cell | 58 of 60 accepted scripts within 2.4e-7 to 1.6e-5; the largest grid compared has 338,688 cells |
| MEEP 1.33.0, single precision | the check's simulation (1 host, 2026-09-28) | 2.5e-6 final field, 9.2e-8 flux |
| MEEP 1.33.0, double precision (conda-forge) | the check's simulation (1 host, 2026-09-28) | 1.5e-6 final field, 9.0e-8 flux |
| MEEP 1.33.0, double precision (conda-forge) | the 3 examples that compare with MEEP (1 host, 2026-09-28) | final field 5.2e-6, 7.6e-6 and 5.1e-4; flux 5.6e-7, 3.3e-7 and 3.0e-5 |

The third example ends after the pulse has left the cell, so the same absolute
difference is a larger fraction of the field that remains
([examples/README.md](examples/README.md)).

What follows for a user of a double-precision MEEP:

- A comparison of a run of this package with your own MEEP run compares
  single-precision stepping with double-precision stepping. On the cases above
  the difference is of the order of single-precision rounding, 1 part in 10^7
  to 1 part in 10^5 for runs that end while the field is still large. That is 4
  simulations on 1 host, not a guarantee.
- A quantity that needs more than about seven significant digits should stay
  on MEEP.
- No speed has been measured against a double-precision MEEP.

### What is bit-identical, exactly

The compiled kernels are bit-identical to the package's own array path **on the
same device under the same subnormal policy**. That is the statement the
certification tests, and `examples/run_examples.py --check` repeats it on your
host (its `gpu` leg against its `array` leg).

A default GPU run and a `prefer_gpu=False` reference run in another process are
**not** bit-identical once values pass through the range of float32 subnormal
numbers (below about 1.2e-38), because a run that dispatches compiled kernels
installs the flush-to-zero policy its kernel table was certified under, and the
reference follows the host's own default. Measured on one 25,600-cell case:
24,550 of 25,600 cells differ. In the 3 examples of this repository the two runs
differ by 8.4e-6 to 1.4e-5 of the final field. The check's simulation is smaller
and shorter; in the 2 environments with a GPU route it has been run in, the two
runs were identical. A small difference there is not a failure: the check
accepts up to 1e-4.

### Building MEEP in single precision

A single-precision MEEP is the configuration every figure in the README and the
manual was measured against. It is not needed to use the package. It exists only
as a source build.

The configure flags, as recorded:

| Host | Recorded in | Flags |
|---|---|---|
| Linux x86_64 | the build script in this repository, from the Linux host's build log | `--enable-shared --enable-single --enable-portable-binary --with-mpi --with-openmp --without-scheme` |
| macOS, Apple silicon | the build script in this repository | `--enable-shared --enable-single --enable-portable-binary --with-mpi --without-scheme --with-libctl=<environment>/share/libctl` |

Both were built from the release tarball `meep-1.33.0.tar.gz` (SHA-256
`bdabc0a112f669f2657fbdca22e31a2aeed515372e8ed56dc55197a4cd7ff0ab`), inside the
source tree, with compilers and libraries from a conda-forge environment, and
installed into that environment.

Three properties of the 1.33.0 tarball that a source builder meets:

1. **Build inside the source tree.** A build in a separate build directory with
   `--enable-single` compiles against the tarball's pre-generated
   double-precision header, installs without an error, and is a
   double-precision library. The configure output is not evidence; the value of
   `mp.is_single_precision()` after installing is.
2. **`configure.ac` needs a one-line change before the build system is
   regenerated:** it calls `AX_CXX_MAXOPT` and the bundled macros define
   `AX_CC_MAXOPT`.
3. **With MPI, use the MPI builds of HDF5 and h5py.** With the `nompi` builds
   MEEP configures without parallel HDF5, and output from several MPI processes
   blocks. One process, which is all this package uses, is not affected.

A Fortran compiler is required although MEEP contains no Fortran: `configure`
uses it to find out how BLAS names its functions.

**On an Apple silicon Mac**, `parity/meep_gpu/build_meep_133_macos.sh` builds the
environment the Metal kernels were certified in: it installs a new conda environment
from the lock `environments/locks/meep-gpu-ref-osx-arm64.conda.txt` without solving
(Python 3.12.13, NumPy 2.4.3, Open MPI 5.0.10, the OpenMP build of OpenBLAS, and every
other package at the build the certified environment holds, except the MPI builds of
HDF5 and h5py), stops unless the new environment holds exactly the lock's packages,
downloads the tarball and verifies its checksum, applies the one-line change,
configures with the flags above, builds, installs, adds the certified PyTorch 2.10.0
and its dependencies, and autograd, which MEEP's adjoint module imports, from
`meep-gpu-ref-osx-arm64.pip.txt` (each by version and wheel hash), and stops with an
error unless MEEP reports 1.33.0, single precision and an MPI build, NumPy, PyTorch and
Metal are what it claims, and
[the comparison with the reference](#the-certification-reference-environment) matches.
It refuses to write into an environment that already exists, and expects conda under
`$HOME/miniforge3` (set `CONDA_ROOT` otherwise). Before this release the script solved
its environment instead; that build differed from the certified environment in 54 of the
139 packages they share (libctl, FFTW, harminv, SciPy and the compiler runtimes among
them), which is why it now installs the lock. The version that installs the lock has been
run end to end on the Mac the Metal kernels were certified on (2026-10-04, 3 min 10 s):
the new environment held 138 of 138 of the lock's conda packages, its MEEP linked against
OpenBLAS as the certified MEEP does, every check passed, the comparison with the
reference read 29 of 29 same, and `tools/check_install.py --require-gpu
--require-reference` in it served the Metal kernels with the GPU run 0 from the reference
and the reference 2.5e-6 (final field) and 9.2e-8 (flux) from MEEP, the figures of the
certified host's own row in the table above
([environments/locks/README.md](environments/locks/README.md)).

    bash parity/meep_gpu/build_meep_133_macos.sh 2>&1 | tee ~/build_meep_133.log
    conda activate meep-gpu-ref

For a certification round, stop there: the gates import this package from the
clone. For everyday use of a single-precision MEEP, switch OpenBLAS to the build
that loads no OpenMP runtime, so a process holds one
([One OpenMP runtime on an Apple silicon Mac](#one-openmp-runtime-on-an-apple-silicon-mac)),
then install the package:

    conda install -c conda-forge --override-channels "libopenblas=*=*pthreads*"
    unset KMP_DUPLICATE_LIB_OK
    python -m pip install ".[apple]"
    python tools/check_install.py --require-gpu

**On Linux x86_64**, `parity/meep_gpu/build_meep_133_linux.sh` performs the
whole build the same way, and also installs the certified CuPy, PyTorch and Triton
into the environment it creates, which makes it the environment a certification
round runs in. It installs the locks `environments/locks/meep-gpu-ref-linux-64.conda.txt`
and `meep-gpu-ref-linux-64.pip.txt` without solving: the environment the 2026-10-03
RTX A6000 certification round ran in, package for package, with autograd added on
2026-10-05 (MEEP's adjoint module imports it; it was installed into that environment
from the lock too). That environment differs
from the one the NVIDIA kernels were first certified in at three points. Open MPI and
h5py: the older versions need the CUDA 12 runtime on conda-forge, which the certified
CuPy cannot sit beside, so the lock has Open MPI 5.0.8 and h5py 3.15.1; both serve only
MPI and parallel HDF5 output, which one process does not use. And the CUDA compiler:
NVRTC 11.8.89, which is what `environments/nvidia-linux.yml` installs, where the records
cut before 2026-10-02 compiled with NVRTC 11.6.55 from the certifying host's own CUDA
11.6. The lock's compilers target glibc 2.28, and the script refuses, before it installs
anything, a host whose glibc is older or cannot be read. It
stops with an error unless the new environment holds exactly the lock's conda packages,
MEEP reports 1.33.0, single precision and an MPI build, the toolchain reports the
certified versions, and
[the comparison with the reference](#the-certification-reference-environment) matches.
Before this release the script solved its environment, and the solve depends on the
host: solved for a glibc 2.35 host (with conda's glibc override, on 2026-10-03), it took
`sysroot_linux-64` 2.34 and kernel headers 5.14 in place of 2.28 and 4.18.

    bash parity/meep_gpu/build_meep_133_linux.sh 2>&1 | tee ~/build_meep_133.log
    conda activate meep-gpu-ref

The solving version ran end to end on the RTX A6000 host on 2026-10-02 in about five
minutes, with conda's package cache warm, and made the environment the lock records; the
version that installs the lock has not yet been run end to end. The lock was checked
against that environment, not built: its 143 package lines are those of the
environment's own export, every URL answers with conda-forge's MD5 and SHA-256 (143 of
143), and every wheel hash resolves to one PyPI wheel (21 of 21). Do not install this package into that environment:
a certification round imports it from the clone, and an installed copy would be
invisible to the record of what ran.

### The certification reference environment

The certification reference of each platform is a pair of lock files under
`environments/locks/`: an explicit conda lock, every package by URL and MD5, and a pip
requirements file, every PyPI package by version and wheel hash.
[environments/locks/README.md](environments/locks/README.md) says what each reproduces,
how it was made, what it cannot capture (MEEP itself, the host, the GPU) and when it is
re-cut. The build scripts above install those files and nothing else; an environment
made any other way, including from the files under `environments/*.yml`, is not the
reference, which is fine for using the package and not for certifying it.

To see how an environment compares, run, with its Python:

    python tools/compare_reference_environment.py
    python tools/compare_reference_environment.py --require-reference   # exit 1 unless it matches

`tools/check_install.py` prints the same comparison as its step 7, after a line that
judges MEEP itself (below). For every numerics-relevant package of the platform (29 on
Apple silicon, 36 on Linux, listed in `environments/locks/numerics-relevant.json`) it
prints the reference's build, this environment's, and one of `same`, `different`,
`missing`, `differs by design` or `not read`, then a verdict. On the Mac the Metal kernels were certified on, the
environment they were certified in reads (rows shortened):

       package        reference                      this environment                    verdict
       libctl         4.5.1=h5505292_1               4.5.1=h5505292_1                    same
       fftw           3.3.10=nompi_haf1500d_112      3.3.10=nompi_haf1500d_112           same
       ...
       hdf5           1.14.6=mpi_openmpi_h8451b09_6  1.14.6=nompi_had3affe_106           differs by design
       ...
       torch          2.10.0, build 2 (PyPI wheel)   2.10.0, build 2 (installed by pip)  same
       of 29: 28 same, 0 different, 0 missing, 1 differ by design, 0 not read
       Verdict: this environment matches the certification reference: 28 of 29 numerics-relevant packages are the reference's builds and 1 differs from it by design (hdf5).

and an environment the build script solved on 2026-10-03, before it installed the lock:

       Verdict: this environment differs from the certification reference in 17 of 29 numerics-relevant packages: libctl, fftw, gsl, harminv, hdf5, libgfortran5, libgcc, libcxx, _openmp_mutex, libfabric, libfabric1, libevent, mpi4py, libgfortran, python, python_abi, scipy.

Conda packages are judged by version and build, PyPI packages by version and origin (a
conda build of PyTorch is not the PyPI wheel the reference holds) and, for PyTorch on a
Mac, by the wheel's build tag. The one documented exception, `differs by design`, is the
nompi HDF5 of the certified Mac environment, that build and no other. In an environment
that is not a conda environment (a pip-only install), the compiled libraries MEEP links
cannot be read: the report marks them `not read`, reports the PyPI builds of NumPy,
SciPy and mpi4py as `different` from the reference's conda builds, and its verdict is
"differs from the certification reference in N of D numerics-relevant packages: ...;
K more could not be read". The host's macOS or glibc version is checked against the
lock's floor (macOS 11.0, glibc 2.28), a populated user site directory (`~/.local`) is
named, since it comes before the environment on the import path, and so is a prefix a
build script solved with `MEEP_REFERENCE_SOLVE=1`.

The comparison judges packages only. MEEP is built from source on top of the lock and is
not one of its packages, so `tools/check_install.py --require-reference` also requires
that the MEEP it imports is the build the reference names, 1.33.0 in single precision
with MPI, and fails otherwise, whatever the packages:

       MEEP       1.33.0, single precision, MPI build: the reference's build

That is the check a host runs before it joins a certification round. Nor does the
comparison judge the compilers, which decide how MEEP is compiled and are not loaded when
it runs: the build scripts hold a build to every package of the lock, compilers included,
before anything is compiled.

## One OpenMP runtime on an Apple silicon Mac

**A process may hold one OpenMP runtime. In the environment of Route 2 it holds
one, whatever the order of the imports and whichever PyTorch release from PyPI
is installed. Do not set `KMP_DUPLICATE_LIB_OK`.**

When a process holds two, the second one stops it:

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.

**Where the two come from.** Measured on 1 Mac (Apple M1 Max, macOS 26.2) on
2026-09-28, from the dynamic loader's own list of the files it loaded:

- The PyTorch wheels on PyPI carry an OpenMP runtime, `torch/lib/libomp.dylib`.
- A conda-forge environment holds another, `lib/libomp.dylib` (package
  `llvm-openmp`). conda-forge's OpenBLAS, in its OpenMP build, loads it, and
  NumPy and MEEP (through LAPACK) load OpenBLAS. conda-forge selects the OpenMP
  build unless told otherwise. MEEP's other libraries load no OpenMP runtime.
- PyTorch 2.10.0 names its runtime `@rpath/libomp.dylib`, and a copy with that
  name already in the process is used instead of its own. So `import meep` or
  `import numpy` before PyTorch left one runtime, and `import torch` first
  loaded two.
- PyTorch 2.14.0, the newest release on PyPI on 2026-09-28, names it
  `@loader_path/libomp.dylib` and loads its own copy whatever the process
  already holds. Beside the OpenMP build of OpenBLAS every import order loads
  two, and changing the order does not help. The releases between 2.10.0 and
  2.14.0 were not examined.

**The fix: an OpenBLAS that loads no OpenMP runtime.** conda-forge also
publishes OpenBLAS in a pthreads build. With it, NumPy and MEEP load no OpenMP
runtime and PyTorch's own is the only one. `environments/apple-silicon.yml`
installs that build. In an environment that already exists, the same change is
one command, and it keeps the PyTorch that is installed:

    conda install -c conda-forge --override-channels "libopenblas=*=*pthreads*"

**Or: PyTorch from conda-forge.** conda-forge's own PyTorch (package `pytorch`)
is built against the environment's `llvm-openmp`, so one runtime serves PyTorch,
NumPy and MEEP. It is the source PyTorch names for conda users since it stopped
publishing its own conda packages
([pytorch/pytorch#138506](https://github.com/pytorch/pytorch/issues/138506),
October 2024). Its builds for Apple silicon have MPS and
`torch.mps.compile_shader`; the newest on 2026-09-28 is 2.13.0. It is a
different build from the certified PyPI wheel:
[GPUs that are not on the certified list](#gpus-that-are-not-on-the-certified-list).

Measured on the same Mac. Each process imports the modules in one order,
computes on the GPU with PyTorch, makes one linear-algebra call with NumPy and,
when it imported MEEP, runs a MEEP simulation for 20 steps. The orders that
import PyTorch were `torch`; `torch`, `meep`; `meep`, `torch`; and `numpy`,
`torch`, `meep`, 2 runs each, unless a row says otherwise. MEEP is conda-forge
`pymeep`.

| Environment | PyTorch | OpenMP runtimes loaded | Processes that ran |
|---|---|---|---|
| MEEP 1.34.0, OpenBLAS in its OpenMP build (what `conda create -c conda-forge pymeep` gives today) | 2.14.0, PyPI | 2 | 0 of 6 (orders `torch`; `meep`, `torch`; `torch`, `meep`) |
| the same | 2.10.0, PyPI | 1 when `meep` or `numpy` came first, 2 when `torch` did | 4 of 8 (`meep`, `torch` and `numpy`, `torch` ran; `torch` and `torch`, `meep` stopped) |
| the same, `llvm-openmp` pinned to 19.1.7, the series that built the runtime inside the 2.14.0 wheel | 2.14.0, PyPI | 2 | 0 of 8 |
| MEEP 1.34.0, OpenBLAS in its pthreads build | 2.14.0, PyPI | 1, PyTorch's | 8 of 8 |
| the environment of the first row after `conda install ... "libopenblas=*=*pthreads*"` | 2.14.0, PyPI | 1, PyTorch's | before: 1 of 1 stopped; after: 8 of 8 |
| MEEP 1.34.0, BLAS from Apple's Accelerate (`libblas=*=*accelerate`, created that way) | 2.14.0, PyPI | 1, PyTorch's | 8 of 8 |
| MEEP 1.34.0, PyTorch from conda-forge | 2.13.0, conda-forge | 1, the environment's | 8 of 8 |
| MEEP 1.33.0, Python 3.12, NumPy 2.4, PyTorch from conda-forge | 2.10.0 and 2.13.0, conda-forge | 1, the environment's | 8 of 8, each |
| NumPy from PyPI, no MEEP | 2.14.0, PyPI | 1, PyTorch's | 6 of 6 (orders `torch`; `numpy`, `torch`; `torch`, `numpy`) |
| **Route 2**: `environments/apple-silicon.yml` (MEEP 1.33.0, OpenBLAS in its pthreads build) | 2.10.0, PyPI | 1, PyTorch's | 18 of 18 (orders `torch`; `torch`, `meep`; `meep`, `torch`; `numpy`, `torch`, `meep`; `meep_gpu`, `meep`, `torch`; and `torch`, `meep` with `KMP_DUPLICATE_LIB_OK=TRUE`; 3 runs each) |
| Route 2 after `python -m pip install --upgrade torch` | 2.14.0, PyPI | 1, PyTorch's | the check and 4 of 4 examples ran ([The newest PyTorch on a Mac](#the-newest-pytorch-on-a-mac)) |

Pinning `llvm-openmp` to the version PyTorch carries does not help: two copies of
one version are still two runtimes.

**`KMP_DUPLICATE_LIB_OK`.** Setting it to `TRUE` removes the abort and leaves
two runtimes in the process. With PyTorch 2.10.0 and the OpenMP build of
OpenBLAS, the order `torch`, `meep` then ended in a segmentation fault, 3 of 3
runs, and with PyTorch 2.14.0 in the same, 3 of 3 runs. Do not set it.

**Import order.** In an environment with one runtime the order does not matter.
In an environment that keeps the OpenMP build of OpenBLAS and a PyPI PyTorch
2.10.0, import `meep` or `numpy` before anything that imports PyTorch; with
PyTorch 2.14.0 there, no order works. `import meep_gpu` alone imports neither
MEEP nor PyTorch, and the package imports NumPy before PyTorch.

**What the check does.** Before any of its processes imports PyTorch,
`tools/check_install.py` reads which runtime PyTorch will load (from the files
of the installed PyTorch) and which one NumPy and MEEP load (from a process that
imports only those two). When PyTorch loads its own whatever the order and NumPy
and MEEP load another, the check stops with `FAILED:` and names the command
above; when the order decides, it prints a `NOTE:`. The runtimes it found are
printed in part 4 of its output. The order has not been examined on Linux.

## GPUs that are not on the certified list

Supported and certified are two statements. A GPU is *supported* when the
kernels are meant to run on it, and they run there by default. An identity is
*certified* when the certification gates ran the kernels on it and the kernel
table's certification records say so; how the Metal table reads its records is
below. Supported is not certified: an Apple GPU no Metal weld ran on is
supported and uncertified.

On an NVIDIA GPU, a device of compute capability 7.0 to 9.0 is supported by
both kernel tables, and the Triton table supports Triton 3.1 (`>=3.1,<3.2`).
The floor is the one Triton 3.1 documents, and the ceiling is the newest
architecture the compilers target: the ptxas bundled with Triton 3.1, and the
NVRTC of the CUDA 11 CuPy build (CUDA 11.8), which compiles the hand-written
CUDA kernels on the certified stack; the CUDA 12 CuPy build compiles them with
the host's CUDA 12, for which 9.0 is a conservative ceiling. These ranges are
read from the compilers' documentation and code; no device outside the certified
list has been run. Compiled kernels run by default on a supported device and
toolchain that is not certified, and the run is labelled uncertified. Every
Apple GPU is supported, M1 through M5 and later: the Metal kernels run on it by
default in any environment, and a run outside the certified environment is
labelled uncertified.

| Kernel table | Certified identity | Supported identity |
|---|---|---|
| Triton (NVIDIA) | compute capability 8.6 with Triton 3.1.0 | compute capability 7.0 to 9.0 with Triton 3.1 |
| hand-written CUDA (NVIDIA) | compute capability 8.6 | compute capability 7.0 to 9.0 |
| Metal (Apple) | the certified environment: GPU architecture `applegpu_g13s` (Apple M1 Max), PyTorch 2.10.0, Metal frontend 32023.850.10, `PYTORCH_MPS_FAST_MATH` unset or `0` | every Apple GPU, in any environment |

1 compute capability and 1 Apple environment are certified, on 2 machines: one
RTX A6000 and one M1 Max. NVIDIA lists the compute capability of every GPU at
<https://developer.nvidia.com/cuda-gpus>; the check prints the one your device
reports.

**The Metal environment** is four facts, each of which changes the code that
runs:

- the GPU architecture, the name Metal compiles GPU code for, read from
  `MTLDevice.architecture` on macOS 14 and later: `applegpu_g13s` on an M1 Max.
  It separates GPU generations that share a Metal GPU family (M3 and M4 are both
  Apple9);
- the PyTorch version, which compiles the Metal sources
  (`torch.mps.compile_shader`);
- the Metal frontend version, which belongs to macOS: there is one per macOS
  build, the same on every Mac on that build, so an update of macOS can move a
  Mac out of the certified environment;
- `PYTORCH_MPS_FAST_MATH`. With it set to `1`, PyTorch compiles every Metal
  source in fast-math mode, `torch.mps.compile_shader` included. Measured
  2026-10-02 on an Apple M1 Max
  (macOS 26.2, PyTorch 2.10.0) with `parity/meep_gpu/probe_metal_fast_math.py`:
  against a run with the variable unset, `1` changed 274,523 of 1,048,576
  float32 divide results and 327,338 of 1,048,576 square-root results; `0`
  changed none.

A fact is *certified* when every certification record the Metal table cites (45
Metal welds) names this value, *not certified* when one names another, and *not
judged* when it could not be read on this host or the records do not name it.
`PYTORCH_MPS_FAST_MATH` is certified unset or `0` and in no other value.

A GPU is *supported* when it is Apple's: when Metal names its architecture
`applegpu_*`, or, where the architecture cannot be read (before macOS 14),
when its name starts `Apple `. An architecture that was read and is not
`applegpu_*` is not Apple's, whatever the name.

`result.driver.fast_path_report()` records each fact read and its verdict under
`environment` (`device`, `device_supported`, `device_certified`,
`torch_certified`, `frontend_certified`, `fast_math_certified`), and the
environments the records name under `environment.recorded_environments`.
`device_supported` is `True` on an Apple GPU, `False` on a GPU that is not
Apple's, and `None` when neither the architecture nor the name was read.

**What happens on anything else.**

- On a supported NVIDIA GPU or Triton version that is not certified, the
  compiled kernels run. The status line marks such a device
  `supported, UNCERTIFIED` (and such a Triton version
  `triton 3.1.1 supported, UNCERTIFIED`), and one line per process names what
  is not certified, for example
  `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU compute capability 8.9 (certified: 8.6). They were dispatched because this NVIDIA GPU and toolchain are supported but not certified bit-identical (the supported range is compute capability 7.0 to 9.0, with Triton >=3.1,<3.2 or with the NVRTC of either CuPy build, CUDA 11.8 for cupy-cuda11x or the host's CUDA 12 for cupy-cuda12x), and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the kernels to certified ones; the gates with no run on this GPU are counted in the dispatch record under uncertified.served[], which names the first five in welds_without_a_live_run_here; compare the results with a prefer_gpu=False run of the same simulation before relying on them`.
  `result.driver.fast_path_report()` carries `certified: False` with what was
  read under `uncertified.served`, and, for a compute capability, the
  certification gates that have no run on it. Compare the run with a
  `prefer_gpu=False` run of the same simulation before relying on it.
- On an NVIDIA host where one kernel table is certified for the device and
  toolchain and the other is only supported, the certified table runs alone and
  the run is certified; the other is refused by name, for example
  `the triton table is supported but not certified here (GPU compute capability 9.0), and the cuda table is certified for this device and toolchain; set MEEP_GPU_ALLOW_UNCERTIFIED=1 to compose it too`.
  `MEEP_GPU_BACKEND_PREFERENCE` naming the refused table is refused by name with
  that reason, unless `MEEP_GPU_ALLOW_UNCERTIFIED=1`.
- On an NVIDIA GPU or Triton version outside the supported range, the kernel
  table is refused by name and the run takes the array path, CuPy on the GPU.
  The refusal ends
  `set MEEP_GPU_ALLOW_UNCERTIFIED=1 to dispatch the kernels on it anyway, without certification or support`;
  with `1` the kernels run there, labelled uncertified and recorded
  `supported: False`.
- On an NVIDIA GPU with `MEEP_GPU_ALLOW_UNCERTIFIED=0`, the compiled kernels
  run on a certified device and toolchain only. Anything else, a device whose
  compute capability could not be read included, is refused by name and the run
  takes the array path, CuPy on the GPU; the refusal ends
  `MEEP_GPU_ALLOW_UNCERTIFIED=0 restricts the NVIDIA kernels to certified devices and toolchains`.
- On an Apple GPU, which is supported, the Metal kernels run. The status line
  marks an Apple GPU that is not the certified one `supported, UNCERTIFIED`, for
  example `Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED`, and the
  certified GPU `certified` whatever else is not. A fact that is not certified
  is named in one line per process, printed after the status line, for example
  `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU architecture applegpu_g15p (certified: applegpu_g13s). They were dispatched because this Apple GPU is supported, and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the Metal kernels to certified environments; compare the results with a prefer_gpu=False run of the same simulation before relying on them`,
  and `result.driver.fast_path_report()` carries `certified: False` with what
  was read under `uncertified.served`. A fact that is not judged runs as
  well, with `certified: None` and no such line. Compare either run with a
  `prefer_gpu=False` run of the same simulation before relying on it.
- On a GPU that is not Apple's, the Metal kernels run as well, unsupported and
  uncertified, unless `MEEP_GPU_ALLOW_UNCERTIFIED=0` refuses its architecture
  by name. Its status line carries no `supported`, and the line after it
  gives as the reason
  `the Metal table runs on environments outside the certified set unless MEEP_GPU_ALLOW_UNCERTIFIED=0`.
  A run whose GPU could not be read at all gives the same reason when another
  fact is not certified.
- On an Apple GPU with `MEEP_GPU_ALLOW_UNCERTIFIED=0`, the Metal kernels run in
  the certified environment only. A fact that is not certified, or not judged,
  is refused by name and the run takes the array path, NumPy on the host CPU.
  The refusal names the fact, for example
  `GPU architecture applegpu_g15p is not the one every Metal weld this table cites ran on (certified: ['applegpu_g13s']); MEEP_GPU_ALLOW_UNCERTIFIED=0 restricts the Metal kernels to certified environments`.

Nothing is wrong with a run on the array path. Whether it is slower than the
compiled kernels depends on the host and the size of the simulation: on the one
Mac measured, the NumPy path was faster than the default GPU route below about
100,000 cells in 2-D (125,000 in 3-D) and slower above about 230,000 cells in
2-D
([docs/guides/validation-and-performance.md](docs/guides/validation-and-performance.md));
no NVIDIA host has been measured this way.

| Host | Array path |
|---|---|
| NVIDIA GPU outside the supported range (compute capability below 7.0 or above 9.0, or a Triton other than 3.1), or outside the certified list with `MEEP_GPU_ALLOW_UNCERTIFIED=0` | CuPy, on the GPU |
| Apple silicon outside the certified environment, with `MEEP_GPU_ALLOW_UNCERTIFIED=0` | NumPy, on the host CPU. The GPU is not used |

- On an NVIDIA host the check prints, for each kernel table's device and for
  the Triton version, `on the certified list`,
  `supported, NOT on the certified list` or
  `NOT supported and NOT on the certified list`. When the kernels ran
  uncertified it adds a `NOTE:` line naming each identity it ran on and whether
  it is supported but not certified or (under `MEEP_GPU_ALLOW_UNCERTIFIED=1`)
  outside the supported range; when they did not serve the run, it prints
  `compiled kernels served the GPU run: NO`, the reason the package gives, and
  an `OK:` line that says where the run stepped instead. On
  a Mac it prints each fact of the environment with `certified`,
  `NOT certified: the records name another`, or
  `not judged: it could not be read, or the records do not name it`
  (`PYTORCH_MPS_FAST_MATH`, when it is set, with `off` for `0` and
  `fast-math kernels, which no certification ran` otherwise). The GPU's mark
  starts with `supported (every Apple GPU is); ` on an Apple GPU that is not
  certified, as in
  `GPU              Apple M3 Pro, architecture applegpu_g15p: supported (every Apple GPU is); NOT certified: the records name another`,
  and with `not an Apple GPU, so not supported; ` on a GPU that is not Apple's.
  Outside the certified environment the check adds a `NOTE:` line saying that
  this Mac's environment is supported but not the certified one (on a GPU that
  is not Apple's or could not be read, that it is not the certified one), so
  the Metal kernels ran uncertified, or, with `MEEP_GPU_ALLOW_UNCERTIFIED=0`,
  prints `compiled kernels served the GPU run: NO` and the reason. Without the
  switch, when no fact is `NOT certified` and one is `not judged`, the kernels
  serve the run and the `NOTE:` says instead that the run is neither certified
  nor uncertified. It still exits 0 when the runs agree.
- The rule reads the PyTorch version, not where PyTorch came from. conda-forge's
  own build of PyTorch 2.10.0 reports 2.10.0, so it reads as certified, although
  the kernels were certified with the PyPI wheel of 2.10.0. conda-forge's newer
  builds (2.13.0 on that date) report their own version and run uncertified.
- On an NVIDIA GPU, an identity that cannot be read is recorded as unknown and
  is not refused, unless `MEEP_GPU_ALLOW_UNCERTIFIED=0`, which refuses it by
  name.
- `MEEP_GPU_ALLOW_UNCERTIFIED` accepts `1` and `0`; any other value is refused
  by name and the run takes the array path, on every kernel table. On every
  table, leaving the variable unset runs a supported identity outside the
  certified one, and `0` restricts the compiled kernels to certified identities
  and refuses one it cannot judge. On the NVIDIA tables, `1` also runs a compute
  capability or a Triton version outside the supported range, recorded
  `supported: False`, and composes a supported, uncertified table beside a
  certified one. A run outside the certified identity carries no certification:
  it prints one line per process naming what is certified, and
  `result.driver.fast_path_report()` carries `certified: False` with the
  identity read. On the NVIDIA tables these rules have been exercised with
  substituted identities only; no NVIDIA device outside the certified list has
  been run. The Metal kernels have run outside the certified environment in
  substituted environments, and in one real one: PyTorch 2.14.0 on one M1 Max,
  admitted by `MEEP_GPU_ALLOW_UNCERTIFIED=1` on 2026-09-28
  ([The newest PyTorch on a Mac](#the-newest-pytorch-on-a-mac)).
- In your own runs, `result.driver.active_step_path` is `"fused"` when compiled
  kernels served the run and `"array"` otherwise, and
  `result.driver.fast_path_report()` states why.
- `MEEP_GPU_DISPATCH=0` keeps a run on the array path on any host.

## Troubleshooting

**`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already
initialized`, on a Mac.** The process loaded two OpenMP runtimes: the one inside
a PyTorch wheel from PyPI and the conda environment's, which conda-forge's
OpenBLAS loads in its OpenMP build. With PyTorch 2.14.0 this happens whatever
the import order. Switch OpenBLAS to its pthreads build, which keeps the PyTorch
you have:

    conda install -c conda-forge --override-channels "libopenblas=*=*pthreads*"

or take PyTorch from conda-forge instead of PyPI. Do not set
`KMP_DUPLICATE_LIB_OK`.
[One OpenMP runtime on an Apple silicon Mac](#one-openmp-runtime-on-an-apple-silicon-mac).

**`FAILED: PyTorch ... and this environment load two different OpenMP
runtimes`, from the check.** The same cause, found before anything imported
PyTorch. Run the command above and run the check again.

**Segmentation fault on a Mac, with `KMP_DUPLICATE_LIB_OK=TRUE` set.** Unset the
variable and apply the fix above. Same section.

**`RuntimeError: GPU backend requested but unavailable: <name>`.** A default
call asks for this host's GPU, and the host has no usable one. The name says
what is missing.

| Name | Remedy |
|---|---|
| `cupy` | CuPy is not installed: Route 1, or step 3 of "Installing into an existing MEEP environment" |
| `cuda-device` | CuPy is installed and finds no device: check `nvidia-smi` and `CUDA_VISIBLE_DEVICES` |
| `torch` | PyTorch is not installed: Route 2, or `python -m pip install ".[apple]"` |
| `mps-device`, `torch.mps.compile_shader` | this PyTorch reports no usable Apple GPU: use the PyTorch wheel from PyPI, `torch==2.10.0`, in a native arm64 Python |

To run without a GPU, pass `prefer_gpu=False`.

**`ImportError: meep_gpu.from_meep needs CPU MEEP importable`.** MEEP is not
installed in the environment this Python belongs to. The message suggests
`conda install -c conda-forge pymeep`, which installs the newest MEEP, 1.34.0
today; it has been run with this package
([Tested MEEP versions and builds](#tested-meep-versions-and-builds)). The
release the environment files pin is
`conda install -c conda-forge --override-channels pymeep=1.33.0`.

**`FAILED: compiled kernels did not serve the run: Triton could not link against
the NVIDIA driver library (/usr/bin/ld: cannot find -lcuda)`, from the check.**
Triton compiles a small launcher the first time a kernel runs and links it with
`-lcuda`, which needs a file named `libcuda.so` on the linker's path. Do step 3
of Route 1 in the shell that runs the check, and put its two `export` lines in
your shell start-up file or job script. On the certified host (2026-09-28), a
run in a new user's environment without step 3 printed
`/usr/bin/ld: cannot find -lcuda` and was served by the array path; with step 3,
compiled kernels served it.

**The check says `compiled kernels served the GPU run: NO` on a certified or
supported NVIDIA device.** No kernel table could launch. On this route the usual cause is
one of the things Triton needs at its first launch. The line under the verdict
gives the reason.

| Reason or message | Remedy |
|---|---|
| `libcuda.so` is named, or `/usr/bin/ld: cannot find -lcuda` | Route 1, step 3 |
| `Failed to find C compiler. Please specify via CC environment variable.` | install `gcc`, or set `CC` to a C compiler |
| `No module named 'torch'` | Triton selects its NVIDIA driver by importing PyTorch: `python -m pip install "torch==2.5.1"` |
| `0 active drivers ([]). There should only be one.` | the installed PyTorch has no CUDA support: `python -m pip install "torch==2.5.1"` from PyPI |
| Triton is reported at a version other than 3.1.0 | another PyTorch was installed and replaced it: `python -m pip install "torch==2.5.1" "triton==3.1.0"` |
| `MEEP_GPU_DISPATCH=0` or `MEEP_GPU_FUSED=0` is set | unset it |

The third and fourth rows are read from Triton's source; they have not been
reproduced.

**`pip install triton==3.1.0` finds no matching distribution.** Triton 3.1.0
has wheels for Linux x86_64 and Python 3.8 to 3.12 only. The environment has
Python 3.13, or the machine is not Linux x86_64.

**CuPy raises `NVRTC_ERROR_COMPILATION` and names `vector_types.h`.** A CuPy
wheel with CUDA 12.2 or later needs the CUDA runtime headers:
`python -m pip install "nvidia-cuda-runtime-cu12==12.X.*"` with your CUDA
version, or use CuPy from conda-forge, where the problem does not occur (CuPy
13.5.1 installation notes).

**CuPy cannot load a CUDA library.** A CuPy wheel loads the libraries of the
CUDA toolkit on the machine. Set `CUDA_PATH` to the toolkit and add its `lib64`
directory to `LD_LIBRARY_PATH`, or use CuPy from conda-forge, which installs the
runtime libraries.

**More than one CuPy is installed** (the check says so). Remove all of them and
install one: `python -m pip uninstall cupy-cuda11x cupy-cuda12x`, then the one
for your route.

**`MeepSimulationNotLiftable: this MEEP process is one of N MPI ranks`.** Start
the script without `mpirun`: [MPI](#mpi).

**`MeepSimulationNotLiftable: this simulation has already been initialized`.**
The lift reads the arguments a simulation was declared with, so it needs one
that MEEP has not initialized or stepped. Lift a newly constructed
`mp.Simulation`, or call `sim.reset_meep()` first.

**A refusal names the "sigma reader", `MEEP_SIGMA_PATCH=1` and
`build_meep_133_macos.sh`.** You do not need a patched MEEP to use this package.
The sigma reader is an optional source patch to MEEP that no MEEP release
contains; stock MEEP lifts uniform dispersive media and dispersive objects with
axis-aligned faces without it. 3 kinds of cell need it: a dispersive object with
a curved surface under `eps_averaging=True`, a rotated dispersive block under
`eps_averaging=True`, and a conductivity shared by media that differ in their
susceptibilities. For the first two, `eps_averaging=False` lifts on stock MEEP.
The patch is `parity/meep_gpu/meep-sigma-reader.patch`; the build script named
in the message applies it on macOS when `MEEP_SIGMA_PATCH=1` is set.

**`mp.is_single_precision()` is `False` after a source build with
`--enable-single`.** The build was made in a separate build directory. Build
inside the source tree: [Building MEEP in single precision](#building-meep-in-single-precision).

**A crash when MEEP calls MPB, after installing other conda packages.** conda
may have replaced packages with ones from the `defaults` channel. Install with
`-c conda-forge --override-channels` (MEEP installation notes).

**The wrong `meep` or `meep_gpu` is imported.** The check prints the location
of both. Unset `PYTHONPATH`.

**The package's own status line ends in `uncertified-unknown` on a Mac.** The
GPU architecture was not judged: it could not be read, or the certification
records do not name one. Metal reports it on macOS 14 and later. On an Apple GPU
the mark reads `supported, uncertified-unknown`: the GPU is supported, and
whether it is certified is not known. When the architecture could not be read,
the line names the GPU without it, and a GPU whose name starts `Apple ` is
supported (`Apple M1 Max supported, uncertified-unknown`); when the name could
not be read either, the line prints `device` and the reason, with no
`supported`. `result.driver.fast_path_report()` records why the architecture
could not be read under `environment.device.unreadable`. The Metal kernels
still run, with `certified: None` when no other fact is uncertified; with
`MEEP_GPU_ALLOW_UNCERTIFIED=0` they are refused by name and the run takes the
array path, NumPy on the host CPU ([GPUs that are not on the certified list](#gpus-that-are-not-on-the-certified-list),
[Reading what ran](docs/getting-started/reading-what-ran.md)).

**`meep_gpu: NOTE the kernels are NOT CERTIFIED on this host`, on a Mac.** The
GPU architecture, PyTorch, the Metal frontend or `PYTORCH_MPS_FAST_MATH` is not
that of the certified environment, and the Metal kernels ran anyway, which is
the default; the status line before it marks the first three `certified` or
`UNCERTIFIED`, and an Apple GPU that is not the certified one
`supported, UNCERTIFIED`. The line names what was read and what is certified,
and on an Apple GPU says the kernels were dispatched because this Apple GPU is
supported: supported is not certified. Compare the
results with a `prefer_gpu=False` run of the same simulation, or set
`MEEP_GPU_ALLOW_UNCERTIFIED=0` to keep the Metal kernels to the certified
environment ([GPUs that are not on the certified list](#gpus-that-are-not-on-the-certified-list)).

**The check passed and my own simulation runs on the array path.** Compiled
kernels cover a set of configurations, not all of them; a sub-step no certified
kernel covers is served by the array path. `result.driver.fast_path_report()`
says which, and the manual's page on compiled-kernel dispatch
([docs/guides/kernel-dispatch.md](docs/guides/kernel-dispatch.md)) lists what is
covered.

**My simulation is refused.** `gpu_compatibility(sim)` returns every reason
without building anything.
[docs/guides/compatibility.md](docs/guides/compatibility.md) lists what is
accepted and what is refused.

**`RuntimeError: This thread is flushing subnormal floating-point values to
zero`** (an NVIDIA host). MEEP sets the flush bits when it initializes, and
this CuPy's kernel compiler could not be guarded against them. Call
`meep.set_zero_subnormals(False)` before the first GPU kernel is compiled.

**A message that appears while a simulation runs** (`DispersionInstability`,
`FdtdDivergence`, `StepFunctionNotHosted`, `SubnormalPolicyLocked`, a
`meep_gpu: step path array ...` line): the manual's
[Troubleshooting](docs/guides/troubleshooting.md) page lists each by its
opening words.

## Reporting a problem

Open an issue at <https://github.com/Erised-AI-Inc/meep-gpu/issues> and include

1. the complete output of

       python tools/check_install.py --verbose

2. the platform (Linux distribution or macOS version) and the GPU model;
3. how MEEP was installed (the environment file, another conda command, or a
   source build with its configure flags);
4. for a simulation that is refused or gives an unexpected result, the smallest
   script that shows it, the output of `gpu_compatibility(sim)`, and for a run
   that completed, `result.driver.fast_path_report()`.

The check's output contains the paths of your Python environment. Read it
before posting and remove anything you do not want to publish.

This package is an independent add-on. It is not affiliated with or endorsed by
the MEEP developers; please do not report problems with this package to the
MEEP project.
