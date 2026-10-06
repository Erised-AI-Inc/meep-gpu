# Troubleshooting

Find the message you see by its opening words. Installation problems are
answered in one place,
[INSTALL.md, Troubleshooting](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#troubleshooting);
this page lists them with a pointer, and answers the messages a run can raise
after installation.

## During installation or the installation check

| Message begins | Where it is answered |
|---|---|
| `FAILED:` (from `tools/check_install.py`) | The line names the section of INSTALL.md to read |
| `OMP: Error #15: Initializing libomp.dylib` | [One OpenMP runtime on an Apple silicon Mac](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#one-openmp-runtime-on-an-apple-silicon-mac). Do not set `KMP_DUPLICATE_LIB_OK` |
| `RuntimeError: GPU backend requested but unavailable:` | A default call asks for this host's GPU, and the host has no usable one; the name at the end says what is missing ([Troubleshooting](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#troubleshooting)). To run without a GPU, pass `prefer_gpu=False` |
| `ImportError: meep_gpu.from_meep needs CPU MEEP importable` | MEEP is not installed in this Python environment. Install it as your route in [INSTALL.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md) does: `pymeep` from conda-forge, never `pip install meep` |
| `FAILED: compiled kernels did not serve the run: Triton could not link against the NVIDIA driver library` | Step 3 of [Route 1](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#route-1-one-nvidia-gpu-on-linux) (`libcuda.so`) |
| `compiled kernels served the GPU run: NO` on a certified or supported NVIDIA device, or on a Mac | [Troubleshooting](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#troubleshooting): the line under the verdict gives the reason |
| `NOT on the certified list` (NVIDIA) | [GPUs that are not on the certified list](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#gpus-that-are-not-on-the-certified-list). `supported, NOT on the certified list`: the compiled kernels still run, uncertified, and the check prints a `NOTE:` line saying so; compare with the NumPy reference. `NOT supported and NOT on the certified list`: the run is correct and takes the array path, unless `MEEP_GPU_ALLOW_UNCERTIFIED=1`. Under `MEEP_GPU_ALLOW_UNCERTIFIED=0` either takes the array path |
| `NOT certified: the records name another` or `not judged:` (Mac) | This Mac's GPU, PyTorch or Metal frontend is outside the [certified environment](kernel-dispatch.md#certification-on-an-apple-gpu), or could not be judged against it. On an Apple GPU that is not the certified one the GPU line reads `supported (every Apple GPU is);` before its mark: the Metal kernels still run, uncertified, and the check prints a `NOTE:` line saying the environment is supported but not the certified one. When no fact is `NOT certified` and one is `not judged:`, the kernels still run, with `certified: None`, and the `NOTE:` says instead that the run is neither certified nor uncertified. Under `MEEP_GPU_ALLOW_UNCERTIFIED=0` they are refused and its `NOTE:` says the run took the array path. Compare with the NumPy reference ([GPUs that are not on the certified list](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#gpus-that-are-not-on-the-certified-list)) |
| `not an Apple GPU, so not supported;` (Mac) | The Metal device is not Apple's. The Metal kernels still run on it by default, uncertified, and the check's `NOTE:` says the environment is not the certified one; compare with the NumPy reference as above |

## Before a run: the simulation is refused

**`MeepSimulationNotLiftable`**, or `gpu_compatibility(sim).supported` is
`False`. The simulation uses something the package does not reproduce exactly.
Every reason is listed, each naming the object and, where one exists, the way
out; [Compatibility and refusals](compatibility.md) lists what is accepted and
what is refused. Common ones:

| Reason says | What to do |
|---|---|
| `this MEEP process is one of N MPI ranks` | Start the script without `mpirun` |
| `this simulation has already been initialized` | Lift a newly constructed `mp.Simulation`, or call `sim.reset_meep()` first |
| a custom `pml_profile`, `R_asymptotic` or `mean_stretch` | Only MEEP's default PML is lifted. Use the default profile, or build an `FdtdDriver` directly and call `driver.setup_pml(thickness_cells, order=3)` ([Direct driver](../reference/public-api.md#direct-driver)) |
| the "sigma reader", `MEEP_SIGMA_PATCH=1` | Not needed to use the package; `eps_averaging=False` lifts two of the three cases on stock MEEP ([INSTALL.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#troubleshooting)) |

Do not change the physics of the simulation to get a lift through. Run a
refused simulation on MEEP.

## When the lift starts: an unstable dispersive pole

**`meep_gpu.dispersion.DispersionInstability`** (a `ValueError`), with a message
beginning `lorentzian susceptibility f0=... at dt=...: ... (UNSTABLE)`. A Lorentz
or Drude pole of a material is unstable at this simulation's time step, so the
run would grow without bound while looking plausible. The preflight does not
test for it: `gpu_compatibility(sim)` can say `supported: True` and the lift
then raises this. The message states MEEP's rule, `f_n < 1/(pi*dt)`. Raise the
resolution (both `dt` and `dx` shrink), lower the Courant factor (`dt` alone
shrinks), or refit the material with a lower-frequency model.

## During a run

| Raised | What happened | What to do |
|---|---|---|
| `FdtdDivergence` | A dispersive run's field energy grew past every physical explanation: an instability the setup-time pole test cannot see, from a susceptibility interacting with an absorber, a conductivity or a boundary | Raise the resolution, lower the Courant factor, or refit the material |
| `FdtdNonlinearityOutOfRange` | A χ2/χ3 run's field left the range MEEP's Padé approximant describes; past it the field stays finite and has the wrong sign | Lower the source amplitude or χ2/χ3, or raise the permittivity |
| `StepFunctionNotHosted` | A MEEP step function reached something the run-loop facade does not carry: the HDF5 writers, `set_materials`, `output_volume`, LDOS | Drop that step function from the GPU run, or run the simulation on MEEP |
| `FdtdCancelled` | `cancel_check` asked the run to stop | Nothing: it is your own request |
| `ValueError: assignment destination is read-only` | On an Apple GPU, a script wrote into a `driver.fields` array while the device held it | Write with `driver.set_field`, read with `get_field`, or run with `prefer_gpu=False` ([Compiled-kernel dispatch](kernel-dispatch.md#the-metal-table-and-held-residency)) |

## Messages that are not errors

| Line begins | Meaning |
|---|---|
| `meep_gpu: step path fused;` | Compiled kernels served the run ([Reading what ran](../getting-started/reading-what-ran.md)) |
| `meep_gpu: step path array on the host CPU; dispatch refused:` | On an Apple GPU, no compiled kernel could serve; the run is correct and ran on the CPU. The reason follows. One ending `restricts the Metal kernels to certified environments` means `MEEP_GPU_ALLOW_UNCERTIFIED=0` is set and this Mac is outside the certified environment |
| `meep_gpu: step path array; dispatch refused:` | On an NVIDIA GPU, no compiled kernel could serve; the run is correct and ran on the array path on the device |
| `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host:` | The kernels ran outside the certified list: by default on a supported identity (a Mac outside the certified environment, because every Apple GPU is supported; an NVIDIA GPU of compute capability 7.0 to 9.0 or a Triton 3.1 release outside the certified identity); on an NVIDIA identity outside the supported range, because `MEEP_GPU_ALLOW_UNCERTIFIED=1` is set. Compare with a `prefer_gpu=False` run before relying on the result. `MEEP_GPU_ALLOW_UNCERTIFIED=0` refuses such a run to the array path instead (on a Mac, the host CPU) |
| `supported, UNCERTIFIED` at the end of the step-path line, on a Mac | This Apple GPU is supported, so the Metal kernels ran on it, and it is not the certified one; a NOTE line follows, naming each fact that is not certified ([Certification on an Apple GPU](kernel-dispatch.md#certification-on-an-apple-gpu)) |
| `meep_gpu: this MEEP build is double precision` | Expected with conda-forge MEEP ([Precision](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#precision)) |
| `uncertified-unknown` on a Mac | A fact was not judged: it could not be read, or the certification records do not name it. The GPU architecture needs macOS 14 or later; a GPU named with no architecture after it, marked `supported, uncertified-unknown`, is an Apple GPU whose architecture was not read, and `device` followed by a reason means no GPU name was read either. The kernels still run, with `certified: None` in the record; under `MEEP_GPU_ALLOW_UNCERTIFIED=0` the run is refused to the host CPU |

## Subnormal policy and CuPy

| Raised | What to do |
|---|---|
| `SubnormalPolicyLocked` | A different subnormal policy was requested after one was installed in this process. Set the policy before the first kernel compiles, or use a fresh process |
| `SubnormalPolicyUnattainable` | The requested policy cannot be installed on one of the executors: `keep` is not attainable on an Apple GPU, and `flush` with CuPy needs `CUPY_ACCELERATORS=''` set before CuPy is imported ([the floating-point contract](../design/floating-point.md#the-subnormal-policy)) |
| `RuntimeError: This thread is flushing subnormal floating-point values to zero` (on a CUDA host) | CuPy's kernel compiler could not be guarded against the flushing MEEP switches on. Call `meep.set_zero_subnormals(False)` before the first GPU kernel is compiled |

## Still stuck

Open an issue at <https://github.com/Erised-AI-Inc/meep-gpu/issues> with the output
of `python tools/check_install.py --verbose`, the MEEP version and precision,
`gpu_compatibility(sim).reasons`, `result.driver.fast_path_report()`, and a
script that builds the simulation. A problem that reproduces with `sim.run(...)`
alone, without this package, belongs to MEEP.
