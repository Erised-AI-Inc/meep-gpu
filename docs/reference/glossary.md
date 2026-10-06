# Glossary

Terms the manual and the package's messages use, in the sense they carry here.

| Term | Meaning |
|---|---|
| **Lift** | Reading a MEEP simulation into this package: MEEP initializes the simulation and builds its grid, and the package reads the realized materials, sources and monitors into its own driver. `lift_simulation` and `run_on_gpu` lift; `gpu_compatibility` checks beforehand without lifting |
| **Driver** | The package's time stepper (`FdtdDriver`) holding one lifted simulation. `lift_simulation` returns it; a `run_on_gpu` result holds it as `result.driver` |
| **Array path** | The package's own implementation of the time step as whole-array operations: NumPy on the host CPU, or CuPy on an NVIDIA GPU. It handles every configuration the lift accepts, and it is what compiled kernels are compared with bit for bit |
| **NumPy reference** | A driver built with `prefer_gpu=False`: the array path on NumPy, on any host. It never runs a compiled kernel |
| **Compiled kernel** | A GPU program that performs one or two sub-steps in place of the array path. Written for Triton, for CUDA by hand, or for Metal |
| **Kernel table** | One of the three sets of compiled kernels: `triton` and `cuda` on NVIDIA hardware, `metal` on Apple hardware. The hardware decides which tables are candidates |
| **Sub-step (slot)** | One of the seven parts of a time step: `step_B`, `fill_B`, `update_H`, `step_D`, `fill_D`, `update_E` and `update_P`. The status line counts them as slots, for example `4/7 slots` |
| **Family, arm** | A family is a kind of kernel (for example the PML curl); an arm is a certified kernel of a family, for the configurations its certification covers. The status line lists the arms that served |
| **Fused kernel** | A compiled kernel that performs two consecutive sub-steps in one launch, for example a *fused magnetic B/H pair* |
| **Dispatch** | Choosing, when a run's configuration freezes at its first step, which sub-steps compiled kernels serve. On by default for a `prefer_gpu=True` driver; `MEEP_GPU_DISPATCH=0` turns it off |
| **Step path** | `fused` when at least one sub-step runs a compiled kernel, `array` when none does (`driver.active_step_path`) |
| **Dispatch record** | The dictionary `driver.fast_path_report()` returns: what ran, and why ([Reading what ran](../getting-started/reading-what-ran.md)) |
| **Refused by name** | Declined with a stated reason instead of approximating: a simulation the lift cannot reproduce raises, and a kernel that cannot serve leaves its sub-step on the array path, saying why |
| **Supported GPU** | A GPU the kernels are meant to run on, and run on by default. Every Apple GPU is supported by the Metal table, M1 through M5 and later: Metal names its architecture `applegpu_*`, or, where the architecture cannot be read (before macOS 14), its device name begins with the word `Apple`. On NVIDIA hardware a device of compute capability 7.0 to 9.0 is supported by both kernel tables, with Triton 3.1 on the Triton table. Supported is not certified: an Apple GPU no certification ran on is supported and uncertified, and the status line says both (`supported, UNCERTIFIED`). Not the preflight's `supported`, which is about a simulation |
| **Certified identity** | The device and toolchain a kernel table's certification was run on: NVIDIA compute capability 8.6 (with Triton 3.1.0 for the Triton table), or on Apple hardware the certified environment. The kernels of every table run on a supported identity, labelled uncertified outside the certified one (on NVIDIA hardware, a table certified for the device and toolchain runs alone in place of one that is only supported), unless `MEEP_GPU_ALLOW_UNCERTIFIED=0`; an NVIDIA identity outside the supported range runs only under `MEEP_GPU_ALLOW_UNCERTIFIED=1` |
| **Certified environment** | The four facts the Metal certification ran on: GPU architecture `applegpu_g13s` (Apple M1 Max), PyTorch 2.10.0, Metal frontend `metalfe-32023.850.10`, and `PYTORCH_MPS_FAST_MATH` unset or `0`. A fact is *certified* when every certification record the Metal table cites recorded it (fast math: only unset or `0`), *not certified* when one recorded another value, and *not judged* when it could not be read or the records do not name it ([Certification on an Apple GPU](../guides/kernel-dispatch.md#certification-on-an-apple-gpu)) |
| **Certification ledger** | A file under `meep_gpu/*_kernels/` that binds each certified kernel to the digests of the source files it was certified with. Dispatch reads it at run time |
| **Bit-identical** | Equal in every bit to the array path on the same host, under the same subnormal policy. Not identity with MEEP, which is measured as a tolerance ([the floating-point contract](../design/floating-point.md)) |
| **Subnormal policy** | Whether float32 values below about 1.2 × 10^-38 are kept or flushed to zero. A run that dispatches installs its kernel table's certified policy for the whole process |
| **Held residency** | On an Apple GPU, the device keeps the field arrays between kernel launches (the default). A script then reads fields with `get_field` and writes them with `set_field` |
| **Preflight** | `gpu_compatibility(sim)`: whether a simulation can be lifted, answered before MEEP initializes anything |
| **Mcell-steps/s** | Million cell updates per second: cells in the grid times steps taken, divided by the wall time of the step loop |
