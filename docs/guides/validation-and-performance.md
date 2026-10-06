# Validation and performance

## Numerical reference hierarchy

The package deliberately separates three questions:

| Question | Reference |
|---|---|
| Does the backend-neutral implementation obey its contract? | NumPy array path |
| Does the package reproduce supported MEEP semantics? | CPU MEEP comparison and targeted closed-form cases |
| Does a compiled kernel preserve the array implementation? | Exact state comparison against the array path **on that table's own engine**, at the claimed run length |

The third row's engine matters. The two NVIDIA kernel tables plan against a CuPy
engine, so their oracle is the CuPy array path; the Metal table plans against a
NumPy engine, so its oracle is the NumPy array path. In both cases the oracle is
the same <code>prefer_gpu=True</code> driver with its kernels vetoed
(<code>MEEP_GPU_FUSED=0</code>). On an Apple GPU that is the same code on the
same arrays as a <code>prefer_gpu=False</code> reference driver, which never
dispatches.

A CuPy run is not a substitute for a CPU-MEEP parity test. Likewise, a passing
field tolerance — or byte equality over only a few steps — is not sufficient to
release a compiled path whose contract is whole-run identity.

The second and third rows are different kinds of statement. Identity with the
array path is exact. Agreement with MEEP is a tolerance, measured separately;
both are in
[the floating-point contract](../design/floating-point.md).

## Test suite

Run the package suite from the repository root, after its dependencies are
installed:

    python -m pytest meep_gpu/ -q

Device-dependent cases declare themselves skipped rather than passing quietly,
because a silent skip and a pass are different claims. A test that needs a
measurement artifact the repository does not carry is skipped by a declared
resource, and the summary lists each resource with its count. See
[Running the tests](../development/testing.md).

There are two device lanes, not one. CUDA conformance and benchmarks run on a
CUDA host; the Metal table has its own Apple-GPU gate lane. A CPU-only
developer should still run the NumPy and CPU-MEEP checks.

> **Note: Expected failures in this release**
>
> The ledger-contract tests compare the source digests recorded in the
> kernel ledgers with the files in the tree. The certification round ran on
> the files of 0.9.2 and the ledgers are bound to them; two tests stay
> pending, those of the fused-product timing records, until a timing campaign
> on these files re-cuts them. They belong to the certification suite (pytest
> marker <code>certification</code>), which a run without <code>-m</code> does
> not select and <code>-m certification</code> runs; there they are failures,
> not skips. See
> [Running the certification harness](../development/certification.md#certification-status-of-this-release).

## What dispatch changes, and what it does not

Compiled-kernel dispatch is on by default on a driver built with
<code>prefer_gpu=True</code> and disabled with <code>MEEP_GPU_DISPATCH=0</code>; a
<code>prefer_gpu=False</code> driver is the NumPy reference and never dispatches.
<code>lift_simulation</code> and <code>run_on_gpu</code> both default to
<code>prefer_gpu=True</code>, so a script that passes nothing is a GPU run and a
reference run is one that says <code>prefer_gpu=False</code>.
When dispatch is on, correctness is held by two kinds of gate, and neither is a
throughput result:

- **Route gates** drive each fused product through the driver's own consults and
  byte-compare whole runs against the array path, with per-step launch counts
  proving the substitution actually occurred.
- **Per-family device gates** certify each family's arithmetic under both
  subnormal policies, each paired with a null control that must be able to fail.

Each gate record states its own limit: it covers the arms it drove, on the
configurations it ran, under the policy it recorded — and nothing about an arm it
did not drive or a shape it did not run. The counts are in
[Compiled-kernel dispatch](kernel-dispatch.md#correctness-under-dispatch).

Because two runs of the same script on the same host can execute different
kernels, record the dispatch state in a run's evidence trail. See
[Compiled-kernel dispatch](kernel-dispatch.md) for the enable, the
table-selection rule, the coverage boards and their denominators, and how to read
<code>driver.fast_path_report()</code>.

## A default lift on a small cell

The default is chosen for the jobs a GPU is for. On a small cell it is the
slower choice, so pass <code>prefer_gpu=False</code> there. Both step the same
equations: at 3,600 cells on the Apple GPU measured, Ez after 200 steps was
bit-identical between the default lift and the reference in 11 of 11 real-field
cases compared. The two run under different subnormal policies, so that
comparison says nothing about values in the subnormal range, which its case does
not reach.

On one Apple M1 Max (2026-09-27; <code>lift_simulation(sim)</code> against
<code>lift_simulation(sim, prefer_gpu=False)</code>; vacuum, an absorber on
every side, one point source, real fields; five 200-step windows per run, median
of three runs):

| Cells | Default lift, ms per step | Reference, ms per step | Default against reference |
|---|---|---|---|
| 3,600 (2-D) | 2.09 to 2.31 | 0.22 to 0.27 | 7.8 to 10.3 times slower |
| 14,400 (2-D) | 2.27 to 2.47 | 0.46 to 0.53 | 4.3 to 5.4 times slower |
| 40,000 (2-D) | 2.30 to 2.39 | 0.79 to 0.88 | 2.6 to 3.0 times slower |
| 102,400 (2-D) | 2.44 to 2.65 | 1.83 to 2.11 | 1.16 to 1.45 times slower |
| 230,400 (2-D) | 2.81 to 2.88 | 3.96 to 4.78 | 1.41 to 1.66 times faster |
| 409,600 (2-D) | 3.32 to 3.62 | 7.43 to 8.83 | 2.05 to 2.66 times faster |
| 64,000 (3-D) | 2.27 | 1.40 | 1.62 times slower |
| 125,000 (3-D) | 2.40 | 2.47 | level |
| 216,000 (3-D) | 2.51 | 4.06 | 1.61 times faster |

The crossover is between 102,400 and 230,400 cells in 2-D and near 125,000
cells in 3-D. A default lift also pays about one to two seconds once, in the
lift and the first step.

These are not timings of record and support no claim beyond their own rows: one
host, loaded by other work while they were taken, one case family, two passes in
2-D and one in 3-D. Another 3-D measurement on the same laptop, of a different
case with a different instrument, read a lower cost per step at a similar size;
the two have not been reconciled. No CUDA host has been measured this way.

## Benchmark fairly

Every performance table must identify:

- MEEP, CuPy, CUDA, compiler, CPU, GPU, and operating-system versions;
- the grid, resolution, boundary conditions, materials, sources, monitors,
  precision, and stopping condition;
- CPU rank count and binding, including both one-rank and well-utilized CPU
  results where relevant;
- warm-up, synchronization, initialization, transfer, I/O, repetition, and
  aggregation policy;
- **the dispatch state**: the enable's value (unset, <code>1</code> or
  <code>0</code>), which kernel table served, and which arms. A benchmark script
  must refuse to report a compiled result when the driver did not execute one.
  An "array path" baseline needs <code>MEEP_GPU_DISPATCH=0</code>, because an
  unset switch dispatches, and a NumPy reference baseline needs
  <code>prefer_gpu=False</code>, because a bare lift is a GPU driver; and
- the numerical error attached to the exact workload that was timed.

State every ratio with its denominator in the same sentence: the MEEP version,
its precision, the rank count and the host. Compare against MEEP at its best
rank count on the host, not against one rank. Do not compare a bare periodic
curl microbenchmark to a CPU run that includes PML, monitors, or a different
precision. Time MEEP on the identical problem, monitors included.

The harness's [timing ladder](../development/timing.md) does all of this in one
run: it interleaves the GPU routes and stock MEEP per size and pass behind a
quiet gate, times MEEP at every rank count of a fixed list and, as controls, a
native-tuned build, MPI ranks with OpenMP threads, ranks bound to hardware
threads and cost-based chunk splitting, and records the CPU state around every row
and the GPU's state (on an NVIDIA GPU its clocks, temperature and power; on a Mac
only the Apple GPU's utilisation, since the rest needs root). A MEEP row whose
geometry digest is not its size's digest of record is struck. The digest of record
is the one the package's own lift computed, when the ladder took one; a size
without it keeps the first MEEP row's digest and its ratios are marked provisional.
A ratio is never formed where the lift's digest differs from the MEEP rows', where
the two sides carried different monitors, or where the rows ran different code.

## Measurements that support a user-facing statement

The measurements that support a statement to a user are collected, each with its
status, in [Will it help?](will-it-help.md). Fused products against
the certified single arms they replace are measured separately; the figures and
the size dependence are in
[Compiled-kernel dispatch](kernel-dispatch.md#what-fusing-buys-and-at-which-size).
The short version: on the NVIDIA tables a two-pair plan loses to its own
certified single arms below a crossover between 1.0 and 2.4 million cells, and
the Metal table's coverage figure supports no timing claim at all. The crossover
was measured on an earlier deposit-repair route and has not been re-measured on
0.9.2 (CHANGELOG, 0.9.2, "Pending: fused-product timing").

The timing harness that produced those rows ships in the repository under
<code>parity/meep_gpu/</code>: <code>bench_fused_products.py</code> times a case
on three legs (the dispatched route, the single kernels, the array path) and
refuses to report a row that fails one of its checks;
<code>bench_meep_identical_case.py</code> times stock MEEP on the same simulation,
built by the same case builder; <code>timing_ladder.py</code> runs both
([Running the timing ladder](../development/timing.md)); and
<code>build_3d_comparison.py</code> builds a comparison table from their rows,
refusing by name any pair whose case label or geometry digest differs.

Do not publish projected speedups. Publish a measurement with its conditions.

See [The three kernel tables](../development/kernel-tables.md) for
the release chain a kernel passes through and
[the backend guide](../development/execution-backends.md) for the
architecture.
