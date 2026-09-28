# The three kernel tables

## Current status

The array path is the numerical oracle and remains the fallback for every
configuration. Compiled kernels replace it slot by slot, inside
<code>driver.step()</code>, by default wherever a released arm covers the
configuration on a certified host; <code>MEEP_GPU_DISPATCH=0</code> turns
dispatch off. Three tables dispatch through the driver's own consults:

| Table | What exists | Runtime status |
|---|---|---|
| Triton | CuPy-pointer interop, per-sub-step coverage predicates, compiled kernels, fingerprinted gate records with a <code>driver_dispatch</code> block, and cross-sub-step fused products | Released and dispatching: 30 arms over 64 arm-case rows; board 334 of 597 seam-instances reachable in dispatch. The first table on a CuPy engine. |
| Hand-written CUDA | <code>cp.RawKernel</code> sources behind the same fail-closed seam, its own certification ledger, and its own board | Released and dispatching: 23 arms over 68 arm-case rows; board 352 of 597, by preference. With a validated Triton it composes second, so it serves nothing there unless <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code> puts it first. Without one it composes alone, a composition run on one device for the installation check's simulation, one example and one route leg with Triton withheld from the process (19 of 39 cases byte-identical, 2026-09-29), and not timed. |
| Metal | Hand-written shaders compiled through <code>torch.mps.compile_shader</code> over persistent MPS mirrors of NumPy-owned fields, with a plan-time residency bracket, held on the device by default | Released and dispatching for a <code>prefer_gpu=True</code> driver on an Apple GPU (a NumPy engine with an MPS device): 29 arms over 72 arm-case rows; board 327 of 597, a correctness and reachability number. For the residency modes see [Execution backends](execution-backends.md#residency-is-the-seam). |

The array path is still the reference the arithmetic is proved against, and a
compiled kernel is a performance specialization rather than a second source of
physics. For the architecture and each table's own boundary, read
[Execution backends and kernel tables](execution-backends.md). For the
user-facing switches, boards and measurements, read
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

## What releases an arm

A benchmark is not in this chain. A fused arm reaches a user run only after all
four of the following. Dispatch is on by default, so once all four hold, a run
that does not set <code>MEEP_GPU_DISPATCH=0</code> gets the arm:

1. **A per-family device gate**, paired with a null control that must be able to
   fail, establishing the family's arithmetic on real hardware under the
   certified subnormal policy.
2. **A ledger weld** binding that certification to the bytes that executed —
   <code>ARM_CERTIFICATION</code> for the Triton table,
   <code>CUDA_ARM_CERTIFICATION</code> for the hand-written table, and the
   Metal table's own registry.
3. **An entry in that table's released-arms map**, restricted by both the shared
   release envelope and that arm's own axes row. Both halves must admit the
   configuration; either one refuses it alone.
4. **A route gate** driving the arm through the driver's own consults,
   byte-identical against the array path over whole runs, with a launch-count
   substitution proof showing the kernel actually replaced the sub-steps.

The predicate is fail-closed throughout. A new feature begins on the array path,
and a missing kernel can never become a reason a supported run is refused.

The admitted set is handed down to the planner explicitly rather than inferred
from a label's spelling. An un-admitted product is therefore never installed, and
its seam keeps the plans it had.

### Single kernels on the hand-written table

The hand-written table also serves a few **single** sub-step kernels, but never
one alone. The merge adopts them only as complete units, and each needs two
things: a launch it can resolve, and a certification row pointing at the byte
gate that certified its entry point. The units are:

- a seam pair, curl plus constitutive: the real-PML curl with the ordinary update,
  or the real-PML curl with one of the three off-diagonal electric updates;
- the symmetry-fill pair. It is adopted on both fill slots or
  on neither, and refused by name when another table holds the other fill slot.
  Its near and far passes run at the driver's two fill consults, with the wall
  clear between them left on the array path.

These singles let the hand-written table serve a mirror-folded run and an
off-diagonal D/E seam without a second table, which matters on a host with no
validated Triton. That composition has been run on one device for the installation check's simulation and one example, and by the
route campaign's leg with Triton withheld from the process (2026-09-29): 19 of 39 cases byte-identical to the array path with only
CUDA kernels launched, 18 complex cases skipped and 2 with no fused arm. It has not been timed.

## The fused unit

The dependency-local pair is the smallest fused unit:

1. <code>step_B → zero_metal_B → update_H</code>; then
2. <code>step_D → zero_metal_D → update_E</code>.

Released products already exceed the pair. The hand-written table ships
three-slot dispersive welds, off-diagonal stencil welds, a folded complex
off-diagonal stencil weld, and a no-absorber three-slot dispersive weld.

**Sources are carried across the seam, not refused.** The deposit repair saves
the stateful arrays at the deposit points before the first slot's launch; the
driver injects, fills and clears as usual; the second slot then recomputes the
constitutive result there in the same operand order. Source-bearing seams are
route witnesses that dispatch today. Dispersion is served by a dedicated
dispersive fused pair rather than by weakening the ordinary pair's predicate.

That bracket is where the two-pair plan's fixed per-step host cost lives — a
seam-swap ablation showed the cost follows the source's seam and that neither
kernel is slow. The bracket is restricted to the component the source's
injection actually writes, bit-identically; the measurement is in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

Combining both pairs into one ordinary Triton launch is still not a simple
extension: D's curl needs the newly computed H field, so every program needs a
grid-wide barrier between the halves, which independent Triton programs cannot
provide inside a normal launch. A one-launch whole step would need a separate
persistent or cooperative design; concatenating the two kernels would be
incorrect.

## Two switches that look alike and are not

<code>MEEP_GPU_FUSED=0</code> is the highest-precedence kill switch: it refuses
the **whole** plan and sends every slot to the array path. It is read when the
plan is built, so set it before the run starts; changing it part-way through
does nothing until the configuration re-freezes.

<code>MEEP_GPU_FUSE_ARMS=0</code> is the arm-level veto: it admits no fused arm
and **leaves the rest of dispatch running**. That is the configuration that
isolates a fusion difference from a sub-step difference, and the one that lets a
launch count show what fusing saves. The route gate depends on this veto:
without it the unfused-dispatch baseline is unreachable and the substitution
proof would measure a drop of zero.

The full switch list, including the subnormal policy and the
<code>CUPY_CACHE_DIR</code> rule that goes with it, is in
[Compiled-kernel dispatch](../guides/kernel-dispatch.md).

## Reading a dispatch decision

<code>driver.active_step_path</code> reports <code>"array"</code> or
<code>"fused"</code>, where <code>"fused"</code> means at least one sub-step will
launch a kernel; read it after a step, because the plan is built at the
configuration freeze inside <code>step()</code>.
<code>driver.fast_path_report()</code> is this driver's own freeze: the path, the
per-slot outcome with its reason, and the certification each dispatched family
rides on. <code>MEEP_GPU_DISPATCH_LOG</code> appends the same record per freeze.

A benchmark script must refuse to report a compiled result when the driver did
not execute one.

## What must be measured

Benchmark a slice against the array path and CPU MEEP on the same physics, both
before it is released and again once it is. Include periodic-core and realistic
PML/monitor workloads, then add the material and boundary features the slice
claims to cover. Report one-rank and well-utilized CPU comparisons separately,
and state which kernel table served.

Do not publish projected kernel speedups. Performance claims follow only from a
measurement and are bounded by the limits stated with it; the measurements
that support a user-facing comparison are in
[Will it help?](../guides/will-it-help.md).
