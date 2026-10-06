# Execution backends and kernel tables

This page describes the layers a lifted run can execute on, and the axis that
separates them: **on by default, turned off by name, released per table against
its own envelope, and measured.** For ordinary MEEP construction and analysis,
continue to use the [MEEP documentation](https://meep.readthedocs.io/en/latest/).
This page is only about the accelerated time-step implementation after a
supported simulation has been lifted.

## Current execution model

| Layer | Status | What it is | How it is selected |
|---|---|---|---|
| NumPy array path | **Active** | Backend-neutral FDTD implementation on CPU arrays; the executable numerical reference. | <code>prefer_gpu=False</code>: the reference, which never dispatches a kernel. Default for <code>FdtdDriver</code> built directly; <code>lift_simulation</code> and <code>run_on_gpu</code> take it by name. Also where an Apple GPU driver steps whatever its kernel table does not serve. |
| CuPy array path | **Active when available** | The same array implementation on CUDA-resident arrays. CuPy translates its array operations into GPU work. | <code>prefer_gpu=True</code>, the default of <code>lift_simulation</code> and <code>run_on_gpu</code>, with a visible CUDA device. |
| Triton kernel table | **Dispatch (on by default)** | Compiled GPU sub-steps and fused products operating on the existing CuPy allocations. 30 released arms over 64 arm-case rows. | A CuPy engine with the certified Triton (3.1.0) installed. Composes first. |
| Hand-written CUDA kernel table | **Dispatch (on by default)** | <code>cp.RawKernel</code> sources behind the same fail-closed seam, with their own certification ledger. 23 released arms over 68 arm-case rows, plus certified single kernels adopted as seam pairs and the symmetry-fill pair. | A CuPy engine. Second behind a validated Triton; first under <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>; **alone** where no validated Triton is installed, a composition run on one device for the installation check's simulation, one example and one route leg with Triton withheld from the process (2026-09-29), and not timed. |
| Metal kernel table | **Dispatch (on by default)** | Hand-written Metal sub-steps launched through <code>torch.mps.compile_shader</code> over persistent device mirrors of NumPy-owned fields. Not a general Torch array backend. 29 released arms over 72 arm-case rows. | <code>prefer_gpu=True</code> on an Apple GPU, every one of which is supported: a **NumPy** engine with an MPS device and <code>torch.mps.compile_shader</code>, in any environment; outside the certified one (GPU architecture <code>applegpu_g13s</code>, torch 2.10.0, Metal frontend <code>metalfe-32023.850.10</code>, fast math off) the run is labelled uncertified, or refused under <code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code> ([Certification on an Apple GPU](../guides/kernel-dispatch.md#certification-on-an-apple-gpu)). |
| Kernel-table selector | — | Two switches at two rungs. <code>MEEP_GPU_KERNEL_TABLE</code> names a table within the candidate set the hardware allows; <code>MEEP_GPU_BACKEND_PREFERENCE</code> orders the two tables a CuPy engine has. | A value naming a table this host cannot run is refused **by name**, never answered with a different table. |

Whether a driver plans at all is decided by how it was built. A driver built
with <code>prefer_gpu=False</code> is the NumPy reference: its configuration
freeze never calls the planner, imports no device library and installs no
subnormal policy, and it publishes a record saying so
(<code>fastpath.reference_record</code>). Only a driver built with
<code>prefer_gpu=True</code> reaches the ladder. The two MEEP entry points build
that driver by default; the engine class built directly defaults to the
reference.

For a driver that plans, the candidate set is answered from two facts the caller
cannot argue with — the engine's array module and the device actually present —
and only then does a switch choose among what came back. A CuPy engine gets the
two NVIDIA tables; a NumPy engine with an MPS device gets the Metal table;
anything else gets no table and stays on the array path. Splitting the rung that
way is what makes a preference naming an absent table a named refusal instead of
a silent fallback.

**A user's GPU is not always CuPy.** <code>prefer_gpu=True</code>, the entry
points' default, means this host's GPU:

    result = run_on_gpu(sim, until=200)

On a CUDA host that call uses the CuPy array path and the NVIDIA kernel tables
(<code>driver.gpu == "cuda"</code>). On an Apple host it uses NumPy host arrays
and the Metal kernel table (<code>driver.gpu == "metal"</code>), and any
sub-step the table does not serve runs on the host CPU. On a host with neither
it raises before MEEP initializes anything. <code>backends.resolve_backend</code>
returns the pair <code>(xp, gpu)</code>, <code>available_gpu()</code> answers
which GPU a request would resolve to, and <code>is_available()</code> is true
exactly when that answer is not <code>None</code>.

Dispatch is on by default, and <code>MEEP_GPU_DISPATCH=0</code> turns it off. The
switch accepts only <code>0</code> and <code>1</code>; any other value is refused by
name and the run takes the array path. A dispatching run still reaches kernels
only where a released family covers the configuration on a host its kernel
table admits (a supported NVIDIA host, compute capability 7.0 to 9.0 with
Triton 3.1, or any Apple GPU, every one of which is supported; outside the
certified identity, labelled uncertified);
everything else stays on the array path, which handles every configuration by
construction. Whichever happened is
recorded: <code>driver.fast_path_report()</code> names the path, the per-slot
outcome with its reason, and the certification each dispatched family rides on,
and <code>MEEP_GPU_DISPATCH_LOG</code> appends the same record per configuration
freeze. The environment can only veto: no value of either switch makes a
<code>prefer_gpu=False</code> driver dispatch. <code>MEEP_GPU_FUSED=0</code> forces the array path regardless of the
enable, and it is read when the plan is built — set it before the run starts.
Setting it part-way through a run changes nothing until the configuration
re-freezes, because the plan is decided once and kept.

## Coverage, and the number that is not a speed claim

Each table has its own coverage board, over a denominator of **597 seam-instances** on a 194-row basis: Triton 334,
hand-written CUDA 352, Metal 327. Each board states two qualifiers about itself
in its <code>dispatch_agreement</code> block (the boards are not published
with this package):
the figure <code>is_an_upper_bound</code>, and it is a **different number** from
the board's own count of seam-instances a fused product exists for at all (530,
532 and 531 of the same 597). Publishing them separately is the point — a product
existing and a product being reachable in dispatch are different claims. The
three dispatch figures assume the complex-storage evidence record, which the
package does not ship; without it they read 244, 258 and 245 of 597.

Each table's <code>fingerprints.json</code> carries a <code>driver_dispatch</code>
record. That record, not this page's prose, is what a reader checks.

## Why these are different layers

The numerical modules use an array-module abstraction, conventionally named
<code>xp</code>. NumPy and CuPy implement the same solver logic against different
array locations: CPU memory for NumPy and CUDA device memory for CuPy. Moving
from NumPy to CuPy is therefore primarily an array-residency and API-discipline
port, not a rewrite of the FDTD equations.

The kernel tables are different. They replace a tightly defined group of array
operations with a purpose-written GPU kernel. That can remove generic launches
and temporary-array traffic, but it also creates another implementation of the
same numerical sub-step. The burden is therefore much higher: its exact coverage,
update ordering, live state, arithmetic grouping, compiler version, and run
length must all be validated against the array path on that table's own engine.

## How the two NVIDIA tables merge

On a CuPy engine both tables are composed, in precedence order, and merged. The
merge boundary is strict: **whole fused units only, never half of one**, and the
secondary table's labels are namespaced <code>cuda:</code>. Under the shipped
precedence the Triton table composes first and holds every slot
both tables admit, so on measured route cases the hand-written units are refused
at the merge with a reason naming the Triton label that holds the slot. That is
why the hand-written table's own board records
<code>by_default_precedence.served_in_dispatch: 0</code>, and why a change to it
is only observable under <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>.

The hand-written table also adopts two certified real-PML single arms, but only
as a partnered seam where both partners are launchable and carry a certification
row. Every other single arm is refused as unlaunchable.

> **Note: Selection is not cost-based**
>
> An arbitration consult that would choose between admitted products by
> measured cost is built and pinned by its own test, but it is **not wired**.
> Today a longer span wins a slot wherever it is admitted, with no size gate.
> Nothing in this package selects a kernel by timing, and a reader should not
> infer that it does.

## Triton table

Triton is not a PyTorch backend in this package. The Triton launcher wraps the
existing CuPy allocation with a small pointer adapter; it does not create a Torch
tensor, copy memory, or move fields through the host. CuPy stays the owner of
fields, material volumes, PML tables, and the CUDA stream context.

The developer module is <code>meep_gpu.triton_kernels</code>. It is deliberately
not re-exported from the package root and it stays importable when Triton is
absent. Its source contains per-sub-step plans for:

- real-field PML curl updates (<code>step_B</code> and <code>step_D</code>);
- PML constitutive updates (<code>update_H</code> and <code>update_E</code>);
- no-PML curls, including lossless variants and conductive variants; real and
  complex-Bloch conductive PML curls are also covered. The complex conductive-PML
  B/D family packs its three phase pairs into one immutable buffer so its
  30-buffer signature stays under Metal's 31-buffer ceiling;
- complex/Bloch no-PML stored-E <code>update_E</code>, with both its zero-pole
  specialization and its live ADE handoff;
- pole-aware real and complex electric constitutive updates, including the
  fold-specific real admission that reuses the pointwise body over the stored
  mirror extent;
- complex/Bloch tensor rows for both no-PML and folded active-PML operation, and
  a real folded tensor-dispersive <code>update_E</code> plan that packs up to
  eight live poles per E component into three persistent buffers to fit Metal's
  31-binding limit;
- one Lorentzian or Drude polarization recurrence (<code>update_P</code>);
- a composed folded-grid slice: folded PML curls, post-source mirror ghost fills,
  and ordinary nondispersive constitutive updates;
- a composed real <code>m=0</code> Dcyl slice; and
- cross-sub-step fused pairs — <code>step_B → zero_metal_B → update_H</code> and
  <code>step_D → zero_metal_D → update_E</code> — plus a pole-aware D/E variant
  and an ADE product.

The 30 released arms span 1-D, 2-D and 3-D, folded and unfolded grids, complex
and Bloch fields, cylindrical coordinates at <code>m=0</code> and at nonzero m
including complex storage, BFAST, nonlinear, off-diagonal, conductive, and
no-PML stored-E shapes. Coverage is still assessed per sub-step rather than asserted for an
entire simulation, and a plan reports its reasons; it never infers coverage from
a missing error. Intersections outside one or more predicates remain refused by
name and fall back to the array path.

### What the certification record fixes

The shipped fingerprint record certifies **Triton 3.1.0** on the CUDA validation
host — an NVIDIA RTX A6000, compute capability 8.6. A different installed version
is a correctness event rather than an automatic upgrade, and the package test
deliberately fails outside the certified set until the gate is repeated. A
supported identity outside the certified one (compute capability 7.0 to 9.0, a
Triton 3.1 release) runs the kernels by default, uncertified, and says so; a
compute capability or Triton version outside the supported range is refused
**by name** when it can be read
([Certification on an NVIDIA GPU](../guides/kernel-dispatch.md#certification-on-an-nvidia-gpu));
a capability that cannot be read is recorded as unknown and not refused,
because refusing on a fact that was not read would be asserting it.

### The subnormal question is a policy contract, not a blocker

Subnormal arithmetic is handled as a contract. Each table declares the policy its
certification was cut under — the two NVIDIA tables under <code>keep</code>, the
Metal table under <code>flush</code> — and dispatch installs that policy on all
three executors or refuses by name. <code>MEEP_GPU_SUBNORMAL_INSTALL=0</code>
does not relax the rule; it only moves the install to the caller, who must have
installed the certified policy already.

Under <code>keep</code> the <code>CUPY_CACHE_DIR</code> in force must carry the
token <code>ftz_stripped</code> in its path. CuPy computes its cache key above the
seam that policy installs at, so a shared directory either gets poisoned with
stripped binaries or silently serves the other policy's arithmetic. That failure
is silent, which is why the rule is load-bearing rather than hygiene.

The boards publish two numbers for one reason: a measured plan is useful for development decisions, but it is not a
speedup available to a user until it is both numerically admitted and actually
selected by the driver.

## Hand-written CUDA table

<code>cuda_kernels/</code> is the hand-written alternative and follows the same
rule as Triton: it can only replace a frozen, explicitly covered sub-step after
validation against the array path, and any other feature stays on the array path.
It is now a released, dispatching table with its own certification ledger and
board. On a CuPy engine it composes second behind a validated Triton, and alone
where none is installed. Its products include welds wider than a
pair — three-slot dispersive, off-diagonal stencil, folded complex off-diagonal
stencil, and a no-absorber three-slot dispersive weld.

See [The three kernel tables](kernel-tables.md) for the release chain
and the switch semantics.

## Metal table

PyTorch is used by the Metal kernel table. The engine itself stays on NumPy
arrays: a residency layer owns persistent Torch MPS mirrors, and purpose-written
Metal source is compiled and launched with
<code>torch.mps.compile_shader</code>. This is architecturally closer to the
Triton table than to the NumPy/CuPy array abstraction — each plan replaces a
strictly covered sub-step, while an uncovered configuration stays on the
reference array path. It is neither a Torch translation of the whole solver nor
output from <code>torch.compile</code>. Every Metal family has its own
fail-closed predicate, buffer binding, source specializations and device
evidence.

Twenty-nine arms over 72 arm-case rows are released, and the table's pending
device-gate list is empty. The released arms are fused B/H and D/E pairs across
ordinary PML, conductive PML and conductive no-PML, complex and complex-beta
fields, real beta and BFAST, dispersive and folded-dispersive electric pairs,
off-diagonal and folded off-diagonal electric pairs, folded real, folded complex
and folded beta grids, cylindrical products at m=0 and at nonzero m including complex storage, a
complex no-PML off-diagonal electric pair, and a complex conductive no-PML curl
welded to complex stored E.

The device-gate evidence behind them is wider than that list — it also covers
families that have no released fused arm, such as the nonlinear spine with its
dedicated Padé <code>update_E</code>. Each family was byte-gated against the
NumPy oracle with armed mutations that had to diverge. Those per-family rows,
their mutation controls, word counts and subnormal censuses live beside the
source; they are correctness evidence and establish no throughput result.

### Residency is the seam

The residency bracket is installed **at plan time**, not inside the plan's
dispatch: <code>sync_in</code> before a launch, <code>sync_out</code> after it.
What the bracket moves depends on the residency mode
(<code>MEEP_GPU_METAL_RESIDENCY</code>):

- **<code>held</code>, the default.** The device owns each engine
  volume between launches, and its host array is sealed read-only while it does.
  A read barrier on the field and polarization attributes brings a volume back to
  the host when host code reads it. The array-path passes that write a held
  volume — source injection, the metallic wall clears, the symmetry and
  far-ghost fills, the trailing repair — go through sparse write paths
  (<code>meep_gpu.host_writes</code>) that move only the cells they touch.
  Constant coefficient volumes are hoisted to the device once.
- **<code>shipped</code>**, kept for comparison. It copies the **whole** mirror
  registry in both directions around every launch, so the host stays
  authoritative between launches and every interleaved array pass is carried
  with no driver change. It is slower than the array path at every size
  measured.

Any other spelling of the variable is refused by name, and the run takes the
array path.

Three mechanisms keep a supported run whole under a hold. Every Metal freeze
first releases any earlier hold on the same fields — the flush, the seals, and
the guards on the polarization states — so the driver's mutators work mid-run.
<code>reset()</code> releases the hold before it zeroes the fields. And an
admission check, run before anything is installed, refuses a configuration a
hold cannot serve, by name and to the array path, never to
<code>shipped</code>. The user-facing consequences — when the array path is the
faster choice, and why direct writes into <code>driver.fields</code> are
unsupported under a hold — are in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md#the-metal-table-and-held-residency).

The held default is certified on the files of 0.9.2 on the Apple M1 Max: the
Metal device gates, the Metal route campaign with residency held, the re-cut of
the Metal dispatch record and the coverage board ran on them. The
table's route gate supports no timing claim, and the Metal
board's coverage figure is a correctness and reachability number only.

### Gate evidence and source welds

Every native Metal gate emits one machine-readable release record after it has
run. The common direct-entry wrapper records
<code>release: {released, reasons}</code> and hashes the Python sources actually
imported by that process from each module's <code>__file__</code>; it does not
trust the current directory or a manually copied digest. A zero process exit is
released only after those runtime hashes are immediately re-checked. The output
directory also carries <code>source_sha256.txt</code>; a later gate in the same
campaign refuses release if a shared source has changed, preventing mixed-tree
evidence.

To verify a saved Metal artifact against the live source tree without re-running
the shader, from the repository root:

```bash
python parity/meep_gpu/metal_gate_runner.py --verify \
  parity/meep_gpu/results/<campaign>/<artifact>.json
```

This weld proves artifact provenance, not numerical coverage. The gate's own
product rows, mutation results, non-vacuity counters, and whole-step evidence are
still the only basis for an arithmetic claim.

Before starting a corpus re-cut, run its source-access preflight. It reads and
hashes every selected MEEP example and test module plus the explicit probe
artifacts, but does not import or step MEEP. This catches a macOS privacy denial
of an external MEEP checkout before a campaign can leave a misleading partial
record:

```bash
MEEP_GPU_SUBNORMAL_POLICY=flush python -m parity.meep_gpu.recut_metal_corpus \
  --preflight --out /tmp/metal-corpus-preflight \
  --probe complex=/path/to/complex_expansion_probe.json \
  --probe special_kz=/path/to/special_kz.json \
  --probe folded_complex=/path/to/folded_complex.json \
  --probe cylindrical_complex=/path/to/cylindrical_complex.json
```

### What this table still does not have

1. Throughput evidence beyond one case. The only Metal timing of record is one
   3-D PML case on one laptop under held residency, and the route gates support
   no timing claim; the figures and their limits are in
   [Compiled-kernel dispatch](../guides/kernel-dispatch.md#the-metal-table-and-held-residency).
2. A corpus re-cut and complete-job validation on the current basis.
3. A safe process model for MEEP and Torch. On the current development install
   their OpenMP runtimes cannot safely share one test process, so CPU-MEEP
   oracles and MPS device gates run in separate processes rather than using an
   unsafe duplicate-runtime override.

A device capability probe and a table-selection policy are **not** on that list:
the probe is the package's own MPS hardware check, and the policy is
the candidate-table rung described at the top of this page.

The validated Triton specification remains the porting reference — coverage
clauses, arithmetic grouping, update ordering, live-pointer behavior, mutation
classes, and composition checks — but Triton source cannot be reused verbatim.
Upstream Triton has no Metal backend, and <code>torch.compile</code> on MPS
follows a different compiler path from these hand-written shaders.

## Development decision rule

Choose the narrowest layer that serves the goal:

- numerical feature or MEEP-parity work: implement and validate it first on the
  NumPy/CuPy array path;
- CUDA kernel-performance investigation: use the Triton or hand-written CUDA
  table only behind a fail-closed coverage predicate, and remember that a change
  to the hand-written table is only observable under
  <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code> or on a host with no validated
  Triton;
- Apple Silicon kernel work: extend the fail-closed Metal plans and their device
  gates; treat a future general Torch array backend as a separate architectural
  decision rather than routing CuPy or Triton through a compatibility shim.

No layer may change a supported simulation's physical model or turn a missing
optimization into a runtime refusal. The array path remains the correct fallback
until a specialized implementation is explicitly admitted. An admitted
implementation runs by default, and <code>MEEP_GPU_DISPATCH=0</code> takes a run
back to the array path.
