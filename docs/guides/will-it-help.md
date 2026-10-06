# Will it help?

meep-gpu is aimed at large three-dimensional problems on one NVIDIA or Apple
GPU, for users without a cluster. Expect no gain over MEEP on two-dimensional
problems, on small grids, on short runs of large grids, or on an NVIDIA GPU
outside the supported range (compute capability 7.0 to 9.0); most of these cases
have not been timed against MEEP, and the table below says which rest on a
measurement. This page states each case with what it rests on, and says what has
not been measured. To measure your own problem, see
[Time your own simulation](time-your-simulation.md).

## How to read the numbers

| Status | Meaning |
|---|---|
| **of record** | Both runs behind the number, the GPU run and the MEEP run, passed every timing check (below). |
| **not a timing of record** | Measured on a host that was loaded by other work, against the package's own reference, not against MEEP. Read the direction and the rough size. |
| **not measured** | No measurement exists. |

Rates are in million cell updates per second (Mcell-steps/s): cells in the grid
times steps taken, divided by the wall time of the step loop. All stepping is
single precision.

The run records behind these figures are not in this repository.

## The comparison with MEEP

One comparison with MEEP has been made on the identical problem, on one machine,
on 2026-09-28, on the development tree this release was prepared from. A timing
campaign from the release tag, with the released timing ladder
([Running the timing ladder](../development/timing.md)), is still to be run.

| Cells | Triton route | CUDA-only route | MEEP, fastest of the rank counts tried |
|---:|---:|---:|---:|
| 512,000 | 1,045.6 (1.80×) | 975.0 (1.68×) | 581.9 (32 ranks) |
| 2,097,152 | 2,025.2 (3.13×) | 1,780.5 (2.75×) | 647.0 (48 ranks) |
| 7,077,888 | 2,043.7 (3.67×) | 1,806.3 (3.24×) | 557.1 (48 ranks) |

Each rate is the median of three passes, each pass in a fresh process. The ratio
in brackets is the GPU rate over MEEP's rate in the same row. All six ratios are
of record.

The Triton route is what a default run takes on a certified NVIDIA host with
Triton 3.1.0 installed. On the CUDA-only route the hand-written CUDA kernel
table, the table a host without Triton uses, serves every dispatched sub-step
alone. It was timed on this host, which also had Triton installed; the
composition a host without Triton builds has not been timed
([What has not been measured](#what-has-not-been-measured)). **The CUDA-only
figures have not been re-timed since the 0.9.1 change to the hand-written CUDA
off-diagonal electric kernel.**

**Read every number with these conditions:**

- **One GPU against one CPU host.** One NVIDIA RTX A6000 (compute capability
  8.6) against the 48 physical cores (2 × 24, Intel Xeon Gold 5220R) of the
  same machine. No other GPU and no other CPU has been compared with MEEP.
- **Single precision on both sides.** Stock MEEP 1.33.0 built from source with
  `--enable-single` (`-O2`, a portable binary, Open MPI 5.0.10), one thread per
  rank with `OMP_NUM_THREADS=1`; the GPU side steps float32 (CuPy 13.5.1,
  Triton 3.1.0). The MEEP builds on conda-forge are
  double precision, and no speed comparison with one has been made.
- **This one 3-D case, with one flux monitor.** The same simulation on both
  sides: a 4 × 4 × 4 cell with a 0.8 PML on every side, a sphere of
  permittivity 9 and radius 0.8, one source and one flux monitor, at
  resolutions 20, 32 and 48 (80³, 128³ and 192³ cells). At each of the three
  sizes the geometry digest of the package's lift equals MEEP's (6 of 6 MEEP
  runs compared at each size), with the same cell count, time step, PML
  thickness, and source and monitor counts. One flux monitor is the lightest
  real monitor load; expect less with more monitors. 78 % of the cells are PML,
  and the source sits off the fused seams, so the case pays no deposit-repair
  cost; a case with a thin PML or a source on a fused seam may gain less.
- **Steady-state stepping, not the one-off lift.** The rates time the step loop
  after a warm-up. The lift that precedes the first step took about 19 s,
  68 s and 216 s at the three sizes, and on the Triton route the GPU run
  finishes first only after about 35,000, 22,000 and 19,000 steps
  ([Short runs](#short-runs-the-lift-comes-first)).
- **MEEP at the fastest of seven rank counts tried.** 1, 8, 16, 24, 32, 48 and
  64 ranks were tried at every size; up to 48 ranks each rank was bound to one
  core, and 64 ranks ran unbound on hardware threads. Each ratio is against the
  fastest of the seven. An unbound run at 32 and 48 ranks, checked once at each
  size, was slower than the bound one in 6 of 6 cases, so the ratios do not rest
  on a binding that handicaps MEEP. MPI ranks with OpenMP threads each were not
  tried in this comparison, nor a MEEP build other than the portable `-O2` one; the
  released ladder times both as controls.
- **Two-dimensional problems and small three-dimensional problems are expected
  to be slower than on MEEP** (inferred; not timed). That rests on the trend
  and on how the package steps, not on a matched timing: the ratio falls as the
  grid gets smaller (1.80× at 512,000 cells), the dispatched step has a fixed
  cost that dominates a small grid, and no grid below 512,000 cells and no
  two-dimensional problem has been timed against MEEP on the identical problem
  ([below](#where-it-does-not-help-or-has-not-been-measured)).
- **Which numbers are reported.** A run passes the timing check when the timing
  windows of each of its three passes spread by at most 5 % and its three pass
  medians spread by at most 5 % (spread: largest minus smallest, over the
  median). A ratio is reported when both of its runs pass. Every GPU run passed,
  and MEEP's fastest run passed at every size. 4 of the 30 runs failed, all of
  them MEEP runs that are not the fastest at their size (16 and 48 ranks at
  512,000 cells; 1 and 64 ranks at 2,097,152 cells), and no ratio on this page
  uses one.
- **The quiet check in front of the MEEP runs could not fail.** The ladder of
  2026-09-28 read the host's busy CPUs as a busy fraction times a CPU count that
  its own thread setting reduced to one, so the reading could never reach its
  threshold of 10. The GPU runs were held to a load-average check that worked; the
  MEEP runs carry no evidence of a quiet CPU. The released ladder sums the busy CPUs
  per CPU and refuses a reading it cannot take; the figures above are unchanged
  until the release-tag campaign re-measures them.
- **Warm-up.** MEEP's timing windows still rose slightly at the two smaller
  sizes. Against MEEP's fastest single window of any pass, the Triton ratios
  read 1.77×, 3.09× and 3.65× instead of 1.80×, 3.13× and 3.67×.

Nothing here is claimed for another problem, for dispersive, cylindrical,
Bloch-periodic or monitor-heavy runs, for sizes not listed, for another MEEP
build, for a double-precision MEEP, or against one MEEP rank.

MEEP's own rate at each rank count, for a reader whose machine has fewer cores
(Mcell-steps/s, median of three passes; † marks a run that failed the timing
check and enters no ratio):

| Cells | 1 | 8 | 16 | 24 | 32 | 48 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512,000 | 50.3 | 303.2 | 464.6 † | 534.0 | 581.9 | 554.2 † | 383.9 |
| 2,097,152 | 47.4 † | 323.4 | 489.3 | 572.2 | 624.9 | 647.0 | 520.2 † |
| 7,077,888 | 51.9 | 335.9 | 485.7 | 532.6 | 552.7 | 557.1 | 513.8 |

MEEP's rate on this machine rises to 32 or 48 ranks and falls at 64. Against
MEEP at 8 ranks, the Triton route reads 3.45×, 6.26× and 6.08× and the CUDA-only
route 3.22×, 5.50× and 5.38× at 512,000, 2,097,152 and 7,077,888 cells; both runs
behind each of these ratios passed the timing check.

## Where it does not help, or has not been measured

| Your problem | Guidance | What it rests on | Status |
|---|---|---|---|
| 3-D, 2×10^6 cells or more, long run, certified NVIDIA device | Use the GPU | The table above: 3.13× and 3.67× on the Triton route | of record at 2,097,152 and 7,077,888 cells |
| 3-D, around 5×10^5 cells, long run, certified NVIDIA device | A smaller gain | The table above: 1.80× on the Triton route | of record at 512,000 cells |
| 3-D below 512,000 cells | Expect a smaller gain, and a loss on small grids; measure your case | Nothing smaller was timed against MEEP on the identical problem, and the ratio falls as the grid gets smaller | not measured |
| 3-D on an Apple device | Measure your case | No comparison with MEEP on the identical problem has been made on Apple hardware. Against the package's own NumPy reference, a default run is slower below about 125,000 cells in 3-D ([below](#small-grids-use-the-reference-or-meep-itself)) | not measured against MEEP |
| 2-D, any size | Keep using MEEP | Not a target: see [below](#two-dimensional-problems) | not measured against MEEP |
| A short run on a large grid | Keep using MEEP below the break-even step count | [Short runs](#short-runs-the-lift-comes-first) | derived |
| A supported NVIDIA GPU or Triton 3.1 release outside the certified set (compute capability 7.0 to 9.0) | Measure your case | The compiled kernels run on it by default, uncertified ([below](#a-gpu-outside-the-certified-set)); no such card has been timed | not measured |
| An NVIDIA GPU or library version outside the supported range | Expect a run slower than MEEP at your rank count | It takes the array path. On the one NVIDIA card measured, with dispatch off, the array path read 0.22×, 0.58× and 0.73× MEEP's fastest at the three sizes above ([below](#a-gpu-outside-the-certified-set)); no other card has been measured | measured on one card; not measured elsewhere |
| An Apple GPU, PyTorch version or macOS build outside the certified environment | Measure your case | Every Apple GPU is supported: the Metal kernels run on it by default, uncertified ([below](#a-gpu-outside-the-certified-set)). No Apple GPU but one M1 Max has been timed, and none against MEEP on the identical problem | not measured |
| Bloch-periodic, cylindrical with m ≠ 0, or complex β | Expect the array path for the affected sub-steps | These runs store complex fields, and their kernels need an evidence record the package does not ship; 50 of the 194 accepted simulations of the MEEP corpus use complex storage | coverage, not speed |
| Many DFT monitors | Expect less gain | One flux monitor and zero DFT monitors were timed. Monitor accumulation runs on the array path on every kernel table | not measured |
| Dispersive, conductive or nonlinear media in 3-D | Unknown | The kernels exist; the only 3-D case timed is the one above | not measured |
| MPI, or several GPUs for one simulation | Not supported | One simulation runs on one GPU. A lift in a process that is one of several MPI ranks is refused by name | code fact |

### Two-dimensional problems

No two-dimensional problem has been timed against MEEP on a matched case, and
none is a target. Two properties of the package make small two-dimensional grids
slow on the GPU:

- **A flat cost per step.** On one 2-D case on the NVIDIA host the dispatched
  step costs 0.94 to 0.97 ms from 24,000 to 2,400,000 cells, whatever the size
  (measured 2026-09-21, on the package's own kernels).
- **The dispatcher does not choose by cost.** It dispatches a fused kernel
  wherever one is released, including where the package's own single kernels
  are faster: against those single kernels the fused route reads a median of
  0.52 times over 36 cases, ahead on 3 of 36 (measured 2026-09-21; 40 of the 44
  distinct pairs of case and grid are at or below 25,000 cells). Setting
  `MEEP_GPU_FUSE_ARMS=0` runs the single kernels instead.

Expect MEEP to be faster at the sizes MEEP users run in two dimensions.

### Short runs: the lift comes first

Before the first step, the lift initializes MEEP in one process, reads the
realized state and uploads it. In the campaign above one lift took about 19 s at
512,000 cells, 68 s at 2,097,152 cells and 216 s at 7,077,888 cells (median of
three passes on the Triton route; the CUDA-only route's are within 2 s of
these). That includes MEEP's own initialization in one process. MEEP run on its
own pays a start-up too: at its fastest rank count, its initialization and first
steps took about 5 s, 20 s and 43 s more than the same steps at its steady rate.

The step count at which the stepping saving pays for the lift is

    steps = (lift seconds − MEEP start-up seconds) / (cells × (1 / MEEP rate − 1 / GPU rate))

with the rates in cell updates per second. From the table:

| Cells | Lift | MEEP start-up | Break-even, Triton route | Break-even, CUDA-only route |
|---:|---:|---:|---:|---:|
| 512,000 | 19 s | 5 s | about 35,000 steps | about 39,000 steps |
| 2,097,152 | 68 s | 20 s | about 22,000 steps | about 23,000 steps |
| 7,077,888 | 216 s | 43 s | about 19,000 steps | about 20,000 steps |

These are derived from the medians, not timed end to end. Below them, expect
MEEP to finish first. This case run to its own end time (1,400, 2,240 and 3,360
steps at the three sizes) finishes first on MEEP at every size: at 7,077,888
cells, about 227 s on the GPU against about 86 s on MEEP at 48 ranks.

### A GPU outside the certified set

On NVIDIA hardware, kernels are certified on one device identity per kernel
table: compute capability 8.6 with Triton 3.1.0, or compute capability 8.6 for
the hand-written CUDA table. They are supported on compute capability 7.0 to 9.0
with Triton 3.1, where they run by default, uncertified, and print a note saying
so; their speed there has not been measured
([Certification on an NVIDIA GPU](kernel-dispatch.md#certification-on-an-nvidia-gpu)).
A device or version outside the supported range is refused by name and the run
takes the array path, as is any uncertified one under
`MEEP_GPU_ALLOW_UNCERTIFIED=0`. On the NVIDIA card measured, with dispatch off,
the array path read 128.9, 373.2 and 404.5 Mcell-steps/s at 512,000, 2,097,152
and 7,077,888 cells: 0.22×, 0.58× and 0.73× MEEP's fastest, slower than MEEP at
each size (of record, under the same conditions as the table above). A device
whose identity cannot be read is not refused.

On Apple hardware every GPU is supported: the Metal kernels are meant to run
on each, and run there by default. They are certified on one environment: GPU
architecture `applegpu_g13s` (an M1 Max), PyTorch 2.10.0, Metal frontend
`metalfe-32023.850.10`, and `PYTORCH_MPS_FAST_MATH` unset or `0`. Another
Apple GPU architecture, PyTorch version or macOS build runs the same kernels
uncertified, and prints a note saying so; their speed there has not been
measured. With `MEEP_GPU_ALLOW_UNCERTIFIED=0`, an environment that is not
certified, or that cannot be judged, is refused by name and the run takes the
array path, NumPy on the host CPU
([Certification on an Apple GPU](kernel-dispatch.md#certification-on-an-apple-gpu)).

## Small grids: use the reference, or MEEP itself

A default lift pays a fixed cost per step and a start-up cost, so below a
crossover size the package's own NumPy reference is faster than its GPU route.
Pass `prefer_gpu=False` for the reference, or set `MEEP_GPU_DISPATCH=0` to reach
the array path without editing the script. And where MEEP itself is faster than
both, use MEEP: the package does not make a simulation MEEP already runs quickly
any quicker.

On the Apple host (2026-09-27; vacuum, an absorber on every side, one point
source, real fields; a default lift against `prefer_gpu=False`):

| Cells | Default lift against the NumPy reference |
|---|---|
| 3,600 (2-D) | 7.8 to 10.3 times slower |
| 14,400 (2-D) | 4.3 to 5.4 times slower |
| 40,000 (2-D) | 2.6 to 3.0 times slower |
| 102,400 (2-D) | 1.16 to 1.45 times slower |
| 230,400 (2-D) | 1.41 to 1.66 times faster |
| 409,600 (2-D) | 2.05 to 2.66 times faster |
| 64,000 (3-D) | 1.62 times slower |
| 125,000 (3-D) | level |
| 216,000 (3-D) | 1.61 times faster |

Status: **not a timing of record**. The host was loaded by other work during
the measurement; the two-dimensional bracket held on 2 of 2 passes and the
three-dimensional rows are 1 pass. The crossover on that host is between
102,400 and 230,400 cells in 2-D and near 125,000 cells in 3-D. A default lift
also pays about one to two seconds once, in the lift and the first step. This
is a comparison with the package's own reference, not with MEEP.

No NVIDIA host has been measured this way. There the array path itself runs on
the GPU, at the rates given [above](#a-gpu-outside-the-certified-set).

## What has not been measured

- Any three-dimensional case but the one above, and that case against MEEP at
  any size but the three above.
- Any comparison with MEEP on Apple hardware on the identical problem.
- A run with DFT monitors, or with more than one flux monitor.
- A two-dimensional case against MEEP on a matched problem.
- The end-to-end time from constructing the simulation to reading a result.
- Any GPU but one RTX A6000 and one M1 Max.
- The timing of the hand-written CUDA table composing alone on a host with no
  validated Triton. (That composition has run the installation check, one
  example and one route leg with Triton withheld from the process, on one
  device, untimed; the CUDA-only column above ran the CUDA table alone on
  a host that also had Triton.)
- The time of a lift from a double-precision MEEP build, or from MEEP 1.34.0.
  Both have been run, not timed.
- A speed comparison with a double-precision MEEP. Correctness runs against
  double-precision MEEP 1.33.0 and 1.34.0 are listed in
  [INSTALL.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#tested-meep-versions-and-builds).

Do not infer a speed from this page for a different problem:
[Time your own simulation](time-your-simulation.md).
