# Capability coverage checklist

This is the package's maintained contract snapshot. It records what the
MEEP-lift interface has a verified implementation for, what is deliberately
left to MEEP, and what the compatibility gate refuses. It is not a substitute
for <code>gpu_compatibility(sim)</code>: that preflight is the authority for a
specific simulation and reports all known blockers before initialization.

For general MEEP use, consult the
[MEEP documentation](https://meep.readthedocs.io/en/latest/). This checklist
describes only the accelerator boundary.

## Status key

| Status | Meaning |
|---|---|
| **Verified** | Implemented and covered by package-local numerical tests for the stated scope. |
| **Conditional** | Implemented for compatible combinations; preflight and the realized-state check decide the particular job. |
| **Delegated** | MEEP remains the implementation of this feature; the lift consumes its result or returns control to it. |
| **Refused / deferred** | The accelerator does not claim an equivalent representation yet, so it reports a named incompatibility rather than simplifying the job. |
| **Dispatch (on by default)** | Released and certified, and it executes inside <code>driver.step()</code> by default wherever the host is one its kernel table admits (a supported NVIDIA host, compute capability 7.0 to 9.0 with Triton 3.1, or any Apple GPU, every one of which is supported; outside the certified identity, labelled uncertified) and the configuration falls inside a released arm's envelope. <code>MEEP_GPU_DISPATCH=0</code> turns it off. Everything else stays on the array path. See [Compiled-kernel dispatch](kernel-dispatch.md). |

## MEEP lift surface

| Area | Status | Current coverage and boundary |
|---|---|---|
| Constructing an <code>mp.Simulation</code> | **Delegated** | MEEP remains the only setup interface. The package consumes a constructed simulation; it does not define a second geometry language. |
| Preflight | **Verified** | <code>gpu_compatibility(sim)</code> is non-mutating and reports every detected blocker. The lift repeats its checks after MEEP realizes the grid. |
| Grid construction, rasterization, and native subpixel smoothing | **Delegated** | <code>lift_simulation</code> calls <code>sim.init_sim()</code> and reads the component-resolved realized constitutive state. It does not recreate MEEP geometry or substitute a scalar epsilon image. |
| Cartesian reduced-dimensional runs | **Conditional** | The checked 1-D, 2-D, and 3-D forms use the compatible cell and component conventions. Unsupported dimension/axis combinations are named by preflight. |
| Cylindrical runs | **Conditional** | Cylindrical lifting and cylindrical near-to-far accumulation are covered for their stated coordinate, symmetry, and mode restrictions. Ask preflight for the exact admissible case. |
| Scalar and diagonal permittivity | **Conditional** | Uniform and structured media are lifted at the component registrations MEEP realizes. Positive, finite, representable media are required. |
| Structured dispersive media | **Conditional** | Lorentz/Drude susceptibility volumes use the available exact reader, declared-geometry lookup with per-point validation, or guarded recovery. The selected route is exposed as <code>result.sigma_lift</code>. |
| Conductivity and instantaneous nonlinearity | **Conditional** | These are covered where the lifted material representation can recover them without ambiguity. The real active-PML nonlinear route has separate B/D/H admissions plus its dedicated Padé E update; complex, folded, and unsupported nonlinear intersections are refused by preflight. |
| MaterialGrid, material functions, epsilon functions, and epsilon-input files | **Conditional** | All four are read rather than refused. An <code>mp.MaterialGrid</code> is expanded into its two endpoint media, which bracket every permittivity the grid can produce, because MEEP itself evaluates a grid into an ordinary medium at every point. A callable material, an ndarray, and an <code>epsilon_input_file</code> are carried as per-point epsilon routes and validated against what MEEP actually built. A material naming no medium this module can read is still refused by name. Ask preflight for the specific job. |
| Off-diagonal inverse permittivity, from subpixel smoothing at a curved or tilted interface or from a declared <code>epsilon_offdiag</code> | **Conditional** | The lift reads every off-diagonal inverse-permittivity entry MEEP realizes and installs it beside the diagonal; nothing is discarded or replaced by a diagonal approximation. 43 of the 194 accepted rows of the MEEP corpus carry such terms. A combination the tensor cannot serve is refused by name at preflight. |
| PML, absorber, metallic, periodic, Bloch, and mirror boundary semantics | **Conditional** | Boundary and symmetry translation is tested for supported combinations, including `mp.Absorber` — a graded electric AND magnetic conductivity rather than a matched layer, installing no boundary condition — and cells mixing it with `mp.PML` per face. Incompatible PML/absorber profiles, phases, folds, or monitor/source placements are rejected by preflight; a cylindrical absorber is rejected by name. <code>bfast_scaled_k</code> — MEEP's broadband fixed-angle technique, which shears time so one run covers a whole band at a single incidence angle — is stepped on Cartesian cells and is refused by name on a cylindrical one. |
| Plain sources and supported synthesized sources | **Conditional** | Supported source waveforms, components, placement, and realizable eigenmode/Gaussian-beam forms are translated. Unsupported source subclasses or variants are named by preflight. |
| DFT flux, DFT fields, near-to-far, force, and energy monitors | **Conditional** | Compatible <code>DftFlux</code>, <code>DftFields</code>, <code>DftNear2Far</code>, <code>DftForce</code>, and <code>DftEnergy</code> specifications migrate. Near-surface accumulation runs here; far-field evaluation stays in MEEP after <code>load_near2far</code>. A force region is limited to the diagonal stress-tensor branch, where the region's declared force direction is its surface normal, and is refused under a declared symmetry or on a cylindrical cell; ask preflight for the exact admissible case. |
| Other MEEP monitor types | **Refused / deferred** | A monitor that cannot be rebuilt is rejected so the package cannot return a valid-looking zero spectrum. |
| MEEP far-field evaluation, mode solving, adjoint optimization, and general analysis | **Delegated** | These are outside the grid-wide time-step loop. Continue to use MEEP's APIs and documentation. |
| Field readback | **Verified** | <code>GpuRunResult.get_array</code> returns host data in MEEP field-array layout; <code>get_field</code> returns the driver's own layout. Their distinction is intentional. |
| NumPy reference backend | **Verified** | <code>prefer_gpu=False</code>. The same array implementation remains executable without a GPU for numerical reference and CPU-host validation. A reference driver never dispatches a compiled kernel on any host, whatever the environment says. |
| CuPy GPU array backend | **Conditional** | A live CUDA/CuPy probe selects it for a GPU request (<code>prefer_gpu=True</code>, the default of <code>lift_simulation</code> and <code>run_on_gpu</code>) on a host with a CUDA device. On a host with an Apple GPU the same request resolves Metal kernels over NumPy arrays; on a host with neither it raises. |
| Triton compiled-kernel track | **Dispatch (on by default)** | The first table on a CuPy engine. It composes per-sub-step plans and cross-sub-step fused products — curl-to-constitutive pairs, a dispersive D/E product, and an ADE product that advances one susceptibility's driven components in one launch — and its released arm table holds 30 arms over 64 arm-case rows spanning 1-D/2-D/3-D, folded, complex and Bloch fields, cylindrical coordinates at m=0 and at nonzero m including complex storage, BFAST, nonlinear, off-diagonal, conductive, and no-PML shapes. Its coverage board reads 334 of 597 seam-instances reachable in dispatch with the complex-storage evidence record, which the package does not ship, and 244 of 597 without it; both are upper bounds. Every other label or shape refuses by name and falls back to the array path. |
| Hand-written CUDA kernel table | **Dispatch (on by default)** | The second table on a CuPy engine, released against its own certification ledger: 23 arms over 68 arm-case rows, board 352 of 597 with the complex-storage evidence record, which the package does not ship, and 258 of 597 without it; both are upper bounds. With a validated Triton installed it fills the first table's refusals: Triton composes first and holds every slot both tables admit, so this table serves nothing else unless a run sets <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code> — its own board records <code>by_default_precedence.served_in_dispatch: 0</code>, and the 352 is the by-preference figure. Without a validated Triton installed it composes alone. That includes its mirror-fill kernels, adopted as a pair, and its off-diagonal electric update beside its own curl. That composition has been run on one device for the installation check's simulation and one example (compiled kernels served, identical to the reference), and by one route leg with Triton withheld from the process (19 of 39 cases byte-identical, CUDA kernels only, 2026-09-29); it has not been timed, and no board publishes its figure. |
| Metal kernel table | **Dispatch (on by default)** | The table a <code>prefer_gpu=True</code> driver takes on a host with an Apple GPU (an MPS device): hand-written Metal shaders launched through <code>torch.mps.compile_shader</code> over persistent device mirrors of the NumPy-owned fields. A configuration no released arm covers steps the array path on the host CPU, and says so. 29 arms over 72 arm-case rows, board 327 of 597 with the complex-storage evidence record, which the package does not ship, and 245 of 597 without it; both are upper bounds. It is not a general PyTorch array backend, and its board is a correctness and reachability number that supports no timing claim. By default the device holds the field volumes between launches (<code>held</code> residency, the default, certified on the files of 0.9.2 on the Apple M1 Max); that wins on larger grids but not on small ones (the measured crossover is between 102,400 and 230,400 cells in 2-D and near 125,000 in 3-D), and it makes direct writes into <code>driver.fields</code> arrays unsupported. When to build with <code>prefer_gpu=False</code> or set <code>MEEP_GPU_DISPATCH=0</code> is in [Compiled-kernel dispatch](kernel-dispatch.md#the-metal-table-and-held-residency). |
| Fused products against the single arms they replace | **Dispatch (on by default)** | Fusing is measured, and it is not a win at small sizes on the NVIDIA tables: against the certified single arms it replaces, the median is 0.52x over 36 route-witness cases on one host (2026-09-21), ahead on 3 of 36, with two-pair plans ahead on 0 of 28. Every row sits at or below 110,592 cells. This is not a complete-job speedup — monitors are detached and the window is the step loop. The sizes and the crossover band are in [Compiled-kernel dispatch](kernel-dispatch.md); they were measured on an earlier deposit-repair route and have not been re-measured on 0.9.2 (CHANGELOG, 0.9.2, "Pending: fused-product timing"). |

Turning dispatch on does not change a result against the array path run under
the same subnormal policy ([Floating point](../design/floating-point.md)). The route gates drive each fused
product through the driver's own consults and byte-compare whole runs against the
array path, with per-step launch counts proving the substitution; the per-family
device gates certify each family's arithmetic under both subnormal policies. None
of that evidence is a throughput claim. How the gates are run is in
[Running the certification harness](../development/certification.md).

## How to update this checklist

Update this page in the same change as any compatibility expansion or
restriction. Each row that changes must have all of the following:

1. a compatibility-gate decision or an explicit statement that no gate is
   needed;
2. package-local tests with a MEEP oracle, a closed-form oracle, or an
   array-reference comparison appropriate to the feature;
3. the tested scope and measured floor, stated in the test module's
   docstring; and
4. a matching update to the other pages of this manual when the release
   claim or dispatch bar changes.

Do not promote a status from **Conditional** to **Verified** merely because a
single example runs. The meaningful unit is the stated feature combination,
with a failure mode that would expose a nearby but physically wrong
implementation.
