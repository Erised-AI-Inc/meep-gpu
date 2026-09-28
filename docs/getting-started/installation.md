# Installation

The distribution is named `meep-gpu`. The import name is `meep_gpu`, and it does
not change. The package installs beside your own MEEP installation; it does not
replace or modify the `meep` module.

**How to install MEEP and this package is written in one place:
[INSTALL.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md), at the
root of the repository.** Each of its routes has been run as written, and its
troubleshooting is organized by the message you see. This page is a summary that
sends you there; where the two differ, INSTALL.md is right.

## Choose a route

| Your machine | Environment file | Read |
|---|---|---|
| Linux x86_64 with one NVIDIA GPU | `environments/nvidia-linux.yml` | [Route 1](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#route-1-one-nvidia-gpu-on-linux) |
| Apple silicon Mac | `environments/apple-silicon.yml` | [Route 2](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#route-2-an-apple-silicon-mac) |
| No GPU, or you want the reference only | `environments/reference-cpu.yml` | [Route 3](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#route-3-no-gpu-the-numpy-reference-only) |
| You already have a MEEP environment and want to keep it | none | [Installing into an existing MEEP environment](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#installing-into-an-existing-meep-environment) |

Every environment file installs MEEP from conda-forge, where its Python module is
published as `pymeep`, pinned to 1.33.0, the release the kernels were certified
against. MEEP is not on the Python package index, and the package is installed
from a checkout of the repository.

## The commands, in short

From the root of a checkout, create the environment of your route:

```bash
conda env create -n meep-gpu -f environments/nvidia-linux.yml    # Route 1
conda env create -n meep-gpu -f environments/apple-silicon.yml   # Route 2
conda env create -n meep-gpu -f environments/reference-cpu.yml   # Route 3
```

The environment name is yours to choose.

Activate it, and install the package into it **without extras**: the
environment file has already installed the device libraries, and an extra would
add a second copy.

```bash
conda activate meep-gpu
python -m pip install .
```

On Route 1, do step 3 of the route next: Triton links against the NVIDIA driver
library under the name `libcuda.so`, which many driver installations do not
provide. Then run the installation check:

```bash
python tools/check_install.py --require-gpu    # without --require-gpu on Route 3
```

It steps one small simulation on this host's GPU, on the NumPy reference and on
MEEP, and compares them. It must end with
`OK: MEEP and meep-gpu work together on this host's GPU.` (on Route 3,
`OK: MEEP and meep-gpu work together on the NumPy reference.` followed by a
sentence). A line starting `FAILED:` gives one reason and names the section of
INSTALL.md to read.
[The installation check](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#the-installation-check)
says how to read the rest of its output. Then run the
[first lifted simulation](first-lift.md).

## Facts that decide which route fits

- **Precision.** Every MEEP build on conda-forge is double precision, and the
  engine steps single precision whatever MEEP was built with. A lift prints one
  line saying so. What agreement to expect is in
  [Precision](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#precision).
  Check what you have with
  `python -c "import meep as mp; print(mp.__version__, mp.is_single_precision())"`.
- **MEEP versions.** Which MEEP releases and builds have been run with this
  package, row by row, is in
  [Tested MEEP versions and builds](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#tested-meep-versions-and-builds).
- **MPI.** Start the lifting process without `mpirun`: the stepper is
  single-process, and a lift in a process that is one of several MPI ranks is
  refused by name. A MEEP built with MPI is fine as one process
  ([MPI](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#mpi)).
- **One OpenMP runtime on a Mac.** A PyTorch wheel beside conda-forge's default
  OpenBLAS loads two OpenMP runtimes and the process stops with
  `OMP: Error #15`. The Route 2 environment avoids it. Do not set
  `KMP_DUPLICATE_LIB_OK`
  ([One OpenMP runtime on an Apple silicon Mac](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#one-openmp-runtime-on-an-apple-silicon-mac)).
- **Certified identities.** Compiled kernels run on NVIDIA compute capability
  8.6 (Triton 3.1.0, or the hand-written CUDA table) and on Apple PyTorch 2.10.0
  with Metal frontend 32023.850.10. Anything else runs the array path
  ([GPUs that are not on the certified list](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#gpus-that-are-not-on-the-certified-list)).
- **The harness is not needed.** Using the package needs nothing under
  `parity/`: that directory is the certification and benchmark harness, and it is
  not part of the installed distribution.

## Verify the runtime instead of inferring it

Run this in the environment that will execute the simulation:

```python
import meep_gpu
print("GPU available:", meep_gpu.is_available())
print("Which GPU:", meep_gpu.available_gpu())
print("Missing GPU pieces:", meep_gpu.missing_dependencies())
```

<code>is_available()</code> is a live probe, and it answers exactly one
question: does <code>prefer_gpu=True</code> resolve on this host. It is true
with an importable CuPy and a visible CUDA device, and true with a Torch build
that has MPS, an MPS device and <code>torch.mps.compile_shader</code>.
<code>available_gpu()</code> names which: <code>"cuda"</code>,
<code>"metal"</code> or <code>None</code>, with CUDA first when a host has both.
<code>missing_dependencies()</code> is empty exactly when a route exists;
otherwise it names what this platform's route lacks. If
<code>is_available()</code> returns false, a <code>prefer_gpu=False</code>
reference run is still valid; a GPU request raises rather than silently falling
back, and a bare <code>lift_simulation(sim)</code> or
<code>run_on_gpu(sim, ...)</code> is a GPU request. <code>cupy_available()</code> still answers the CUDA
question specifically.

**It is a hardware probe, not a certification.** It says nothing about whether
dispatch will run or which kernels would serve: a Torch or Metal frontend outside
the certified pair resolves <code>"metal"</code> here and is then refused by
name at the first step, leaving the run on the host CPU. For that, run a step
and read what ran — see [Reading what ran](reading-what-ran.md).

## Build this manual

The manual is MkDocs source under <code>docs/</code>, with its configuration in
<code>mkdocs.yml</code> at the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict --site-dir /tmp/manual-site
```

Give the build a site directory outside the repository, as above, so that the
generated site is never mistaken for source.
