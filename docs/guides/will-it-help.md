# Will it help?

meep-gpu is aimed at large three-dimensional problems on one NVIDIA or Apple
GPU, for users without a cluster. Expect no gain over MEEP on two-dimensional
problems, on small grids, on short runs of large grids, or on an NVIDIA GPU
outside the certified set; most of these cases have not been timed against
MEEP, and the table below says which rest on a measurement. This page states
each case with what it rests on, and says what has not been measured. To
measure your own problem, see [Time your own simulation](time-your-simulation.md).

## How to read the numbers

| Status | Meaning |
|---|---|
| **of record** | The GPU run behind the number passed the timing check of record ([below](#the-timing-check-and-the-change-made-after-these-runs)) in a session whose precondition passed, and it is compared with MEEP's fastest rank count, timed in the same session on the same machine. |
| **struck** | Measured, and failed the timing check. Its median is printed for the shape only, and no ratio uses it. |
| **derived** | Computed from figures of record by the formula stated beside it, not timed as such. |
| **not a timing of record** | Measured on a host that was loaded by other work, against the package's own reference, not against MEEP. Read the direction and the rough size. |
| **not measured** | No measurement exists. |

Rates are in million cell updates per second (Mcell-steps/s): cells in the grid
times steps taken, divided by the wall time of the step loop. All stepping is
single precision. **Per step and end to end are kept apart.** A rate, and a ratio
of two rates, is per step: it times the step loop after a warm-up and leaves out
the lift before the first step. What a whole run costs, lift included, is in
[Short runs](#short-runs-the-lift-comes-first).

## The comparison with MEEP

The 1.0 timing (CERT-8) compared the package with MEEP on the identical problem,
on the bytes of this release, on one NVIDIA host and on one Apple laptop, on
2026-10-16. The timed runs recorded the files they imported, and every package
file among them is byte-identical to this release's. On each machine the released
timing ladder
([Running the timing ladder](../development/timing.md)) interleaved the GPU runs
and MEEP's runs per size and pass, in one session, on an otherwise idle host, and
the tables below were cut by tool from the stored runs. The run records behind
these figures are not in this repository.

The comparison of 2026-09-28 that earlier versions of this page stated was made on
a development tree that predates 1.0, and the 1.0 round's first timing
(2026-10-13) on an earlier tree of the 1.0 series; neither describes this release
([Against the round's first timing](#against-the-rounds-first-timing-2026-10-13)).

### On one NVIDIA RTX A6000

| Cells | Triton route | CUDA-only route | MEEP, fastest rank count | Array path, dispatch off |
|---:|---:|---:|---:|---:|
| 512,000 | 958.2 (1.65×; passed, 1.75 %) | 841.5 (1.45×; passed, 0.60 %) | 581.9 (32 ranks; 0.84 %) | 84.4 (0.15×; passed, 2.08 %) |
| 2,097,152 | 2,589.8 (3.98×; passed, 0.61 %) | 2,985.9 (4.59×; passed, 0.52 %) | 651.1 (48 ranks; 0.64 %) | 349.9 (0.54×; passed, 1.11 %) |
| 7,077,888 | 2,794.3 (5.04×; passed, 0.38 %) | 3,221.0 (5.81×; passed, 0.03 %) | 554.8 (48 ranks; 0.39 %) | 374.0 (0.67×; passed, 0.05 %) |

Each rate is the median of three passes, each pass in a fresh process. In brackets:
the rate over MEEP's in the same row, per step; the run's status under the timing
check of record; and the spread of its three pass medians (MEEP's is recorded and
not gated). **All six GPU ratios are of record**, and every GPU run and every
array-path run passed the timing check. Every GPU run launched the kernels its route expects (18 of 18; on this
case three fused launches per step) and was bit-identical to the array path
(18 of 18).

The Triton route is what a default run takes on a certified NVIDIA host with
Triton 3.1.0 installed. On the CUDA-only route the hand-written CUDA kernel table,
the table a host without Triton uses, serves every dispatched sub-step alone
(<code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>); on this case it is the faster
route from 2,097,152 cells up. It was timed on this host, which also had Triton
installed; the composition a host without Triton builds has not been timed
([What has not been measured](#what-has-not-been-measured)).

**Read every number with these conditions:**

- **One GPU against one CPU host.** One NVIDIA RTX A6000 (compute capability
  8.6; CuPy 13.5.1, Triton 3.1.0) against the 48 physical cores (2 × 24, Intel
  Xeon Gold 5220R, 96 hardware threads) of the same machine. No other NVIDIA GPU
  has been timed against MEEP.
- **The reference MEEP, single precision on both sides.** MEEP 1.33.0 as this
  release's reference environment builds it (owner ruling R27): the release
  tarball pinned by SHA-256, single precision, MPI (Open MPI 5.0.8) with OpenMP
  linked, no floating-point contraction (`-O2 -ffp-contract=off -fno-fast-math`)
  and a portable binary (`-march=x86-64 -mtune=generic`). It is neither a
  stock-flags nor a native-tuned build; no native-tuned build was timed. The MEEP
  build the round's first timing used (`-g -O2`, a portable binary, built
  without `-ffp-contract=off`) was timed again on the same host right after this
  session, MEEP alone: at MEEP's fastest rank count it agrees with the reference
  build within 0.3 % at 2,097,152 and 7,077,888 cells; at 512,000 cells the
  reference build reads 3.8 % faster, about as much as that MEEP run's own spread
  between its passes (3.8 %), so the bridge does not separate the two builds
  there. The GPU
  side steps float32. The MEEP builds on conda-forge are double precision, and no speed comparison
  with one has been made.
- **The subnormal policy.** The GPU runs step under `keep`, beside MEEP under the
  flush that importing MEEP sets on x86 ([The subnormal policy of each
  row](../development/timing.md#the-subnormal-policy-of-each-row)). The
  dispatched route under `flush` has not been timed.
- **This one 3-D case, with one flux monitor.** The same simulation on both
  sides: a 4 × 4 × 4 cell with a 0.8 PML on every side (78 % of the cells are
  PML), a sphere of permittivity 9 and radius 0.8, one point source and one flux
  monitor, at resolutions 20, 32 and 48 (80³, 128³ and 192³ cells). At each size
  the geometry digest of record is the one the package's own lift computed, and
  every MEEP run at that size carries it (117 of 117); the lift's grid shape,
  cell count, time step, PML thickness and source and monitor counts equal
  MEEP's at all three sizes. The source sits off the fused seams, so
  the case pays no deposit-repair cost. One flux monitor is the lightest real
  monitor load; expect less with more monitors, a thinner PML or a source on a
  fused seam.
- **Per step, not the one-off lift.** The rates time the step loop after a
  warm-up; the lift and the end-to-end figures are in
  [Short runs](#short-runs-the-lift-comes-first).
- **MEEP at the fastest of seven rank counts.** 1, 8, 16, 24, 32, 48 and 64
  ranks were timed at every size; up to 48 ranks each rank was bound to one core,
  and 64 ranks ran unbound on hardware threads. Each ratio is against the fastest
  of the seven. Controls ran in the same session: 64 and 96 ranks bound to
  hardware threads, and MEEP's cost-based chunk splitting
  (<code>split_chunks_evenly=False</code>) at 32 and 48 ranks, each slower than
  MEEP's fastest at every size.
- **MPI ranks with OpenMP threads were measured, and struck.** 24 ranks of 2
  threads read 668.6 and 564.2 Mcell-steps/s at 2,097,152 and 7,077,888 cells,
  2.7 % and 1.7 % faster than MEEP's fastest pure-MPI count (at 512,000 cells
  it reads 522.8, slower). Every such run is struck by a defect of the timing
  harness: its check that OpenMP added threads reads its baseline after the
  thread pool exists, so it saw none added, although each rank used 2.0 CPU
  seconds per wall second with 2 threads and 4.0 with 4. The ratios of record
  stay against pure MPI. Against
  24 × 2 they would read 3.87× and 4.95× on the Triton route and 4.47× and 5.71×
  on the CUDA-only route at those two sizes.
- **A quiet host, checked.** Each timed unit started after two clear readings
  of the host 120 s apart, and a watcher read the host every 15 s through every
  unit (3,829 readings, no intrusion). Another account's process sat idle on a
  different GPU of the host throughout (0 % utilisation on every reading); no
  other process used the timed GPU. Before the timed sizes one point was repeated
  five times: the Triton route at 512,000 cells spread 1.56 % across the repeats
  (the precondition, at most 5 %), and MEEP on 32 ranks 2.81 % (recorded). The
  GPU's clocks were recorded once a second on every GPU run: 1,800 to 1,980 MHz,
  33 to 57 °C, the software power cap engaged on 231 of 576 busy samples. A
  first attempt at the ladder was set aside whole when other work of the same
  account appeared on the host; nothing of it is of record.
- **Warm-up.** Every MEEP run (117 of 117) ended its warm-up at its time limit,
  short of the steps asked; the effect of a longer warm-up on these ratios was
  not measured.

Nothing here is claimed for another problem, for dispersive, cylindrical,
Bloch-periodic or monitor-heavy runs, for sizes not listed, for another GPU, CPU
or MEEP build, for a double-precision MEEP, for MPI ranks with OpenMP threads as
MEEP's best, for the `flush` policy on the GPU, or against one MEEP rank.

MEEP's own rate at each rank count (the reference build above), for a reader whose
machine has fewer cores (Mcell-steps/s, median of three passes; MEEP's spread is
recorded and not gated):

| Cells | 1 | 8 | 16 | 24 | 32 | 48 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512,000 | 50.0 | 303.8 | 455.0 | 534.8 | 581.9 | 549.7 | 380.4 |
| 2,097,152 | 48.0 | 322.1 | 499.9 | 567.0 | 617.3 | 651.1 | 533.0 |
| 7,077,888 | 51.7 | 335.0 | 486.2 | 530.8 | 551.0 | 554.8 | 523.3 |

MEEP's rate on this machine rises to 32 or 48 ranks and falls at 64. The
controls of the same session, on the same build:

| Cells | 64 ranks, bound to hardware threads | 96 ranks, bound to hardware threads | 32 ranks, cost-based split | 48 ranks, cost-based split | 24 ranks × 2 threads (struck) | 12 ranks × 4 threads (struck) |
|---:|---:|---:|---:|---:|---:|---:|
| 512,000 | 387.2 | 200.2 | 410.9 | 343.5 | 522.8 | 461.8 |
| 2,097,152 | 550.6 | 337.0 | 506.7 | 526.7 | 668.6 | 599.5 |
| 7,077,888 | 530.4 | 405.2 | 487.1 | 503.6 | 564.2 | 528.7 |

### On one Apple M1 Max

| Cells | Metal route | MEEP, 8 ranks (fastest) | Ratio | Array path, NumPy on the host |
|---:|---:|---:|---:|---:|
| 110,592 | 261.1 (passed, 0.61 %) | 318.0 (0.79 %) | 0.82× | 21.5 (struck, 10.1 %) |
| 262,144 | 508.2 (passed, 1.06 %) | 418.7 (3.39 %) | 1.21× | 21.7 (struck, 23.1 %) |
| 512,000 | 765.5 (passed, 1.67 %) | 452.3 (3.94 %) | 1.69× | 19.4 (struck, 16.3 %) |
| 2,097,152 | 1,080.2 (passed, 2.44 %) | 449.5 (1.79 %) | 2.40× | 20.0 (struck, 7.3 %) |
| 7,077,888 | 1,363.2 (passed, 0.30 %) | 500.3 (1.08 %) | 2.72× | 20.7 (passed, 0.44 %) |

Each rate is the median of three passes, each pass in a fresh process, with its
status under the timing check of record and the spread of its three pass medians
(MEEP's is recorded and not gated); the ratio is the Metal route's rate over MEEP's,
per step. **All five Metal medians passed the timing check of record and all five
ratios are of record**, under the check ruled after these runs were seen; under the
check in force when they ran, only the 7,077,888-cell row passed
([below](#the-timing-check-and-the-change-made-after-these-runs)). The array path,
the reference these kernels are proved bit-identical to, failed the check at four
of the five sizes and passed at 7,077,888 cells. Every GPU run launched the Metal table (55 of 55; three
fused launches per step), and every GPU run that carries a bit-identity verdict is
bit-identical to the array path (41 of 41).

MEEP at each rank count (the reference build below; Mcell-steps/s, median of three
passes):

| Cells | 1 | 4 | 8 | 10 |
|---:|---:|---:|---:|---:|
| 110,592 | 73.5 | 200.4 | 318.0 | 96.1 |
| 262,144 | 85.2 | 276.3 | 418.7 | 129.1 |
| 512,000 | 95.5 | 318.2 | 452.3 | 164.9 |
| 2,097,152 | 99.8 | 331.6 | 449.5 | 214.3 |
| 7,077,888 | 105.4 | 361.9 | 500.3 | 278.7 |

A second run at the three larger sizes lifted the Metal route alone (one lift, no
single-kernel or array-path leg beside it), in a session of its own: 780.9, 1,102.8
and 1,388.6 Mcell-steps/s against the reference MEEP at 8 ranks 444.7, 451.7 and
498.6 (1.76×,
2.44× and 2.78×; all three passed the timing check of record, their pass medians
spread by 0.16 to 1.09 %), about 2 % above the rows of the table. It supplements the release's timing plan and is not one of its rows.
It carries no bit-identity verdict of its own: each size cites a three-leg run of
the same session at that size, which is bit-identical (3 of 3).

**Read every number with these conditions:**

- **One laptop.** One Apple M1 Max (8 performance and 2 efficiency cores, 32 GPU
  cores, 32 GiB; GPU architecture `applegpu_g13s`, PyTorch 2.10.0) against MEEP on
  its own CPU. No other Apple GPU has been timed.
- **The reference MEEP, single precision on both sides.** MEEP 1.33.0 as this
  release's reference environment builds it for macOS: single precision, Open MPI
  5.0.10, no floating-point contraction (`-O2 -ffp-contract=off -fno-fast-math`),
  a portable binary. 1, 4, 8 and 10 ranks were timed (10 is every core,
  efficiency cores included), unbound, because Open MPI does not bind ranks on
  macOS; 8 ranks is the fastest at every size. The build the round's first Apple
  timing used (built without `-ffp-contract=off`) no longer exists, so no bridge
  between the two builds was run on this machine.
- **The subnormal policy.** The Apple GPU flushes subnormals and has no switch;
  MEEP on Apple silicon keeps them.
- **The same case as above**, at resolutions 12, 16, 20, 32 and 48 (48³ to 192³
  cells), with one flux monitor and held residency (the default).
- **The GPU's timing windows.** Six 2 s windows per leg, interleaved AB/BA, each
  sized from a timed probe, after one full window per leg that enters no
  statistic.
- **A quiet host, checked.** Each timed unit started after two clear readings
  120 s apart, and a watcher read the host every 15 s through every unit (1,984
  readings, no intrusion). The ladder held each run until the macOS background
  services (indexing, media analysis, malware scans) were quiet, once for more
  than two hours. Before each session one point was repeated five times; the
  first attempt of the first session spread 8.52 % on the Metal route, was not
  used, and was repeated 600 s later at 2.35 %; the other two sessions read
  0.49 % and 1.10 %.

Nothing here is claimed for another Apple GPU, another case or size, a MEEP build
other than the reference, or against one MEEP rank.

### The timing check, and the change made after these runs

A GPU run's median enters when the spread of its three pass medians
((largest − smallest) / median) is at most 5 % and no pass of it failed another
check (what it launched, bit identity, its deposit-repair route, the host
conditions). The spread of each pass's timing windows is recorded with the run,
and no table gates on it. MEEP's spread is recorded and never gated. A ratio also needs
its session's precondition (five repeats of one point, within 5 %) to have
passed.

This check (owner ruling R33, 2026-10-16) replaced the one in force when these
runs were made, and it was ruled after the 2026-10-16 runs had been seen. Under the
earlier check a pass in which any leg's timed windows spread above 5 % failed, and
so did the run. It was replaced because a published figure is a median of pass
medians, so what has to hold is that the medians reproduce. On the Apple laptop the
earlier check struck 6 of 8 Metal medians (17 of 24 timed passes): in 11 of the 17
passes the Metal route's own 2 s windows spread 5.2 to 10.3 %, and 2.1 to 5.1 % with
the single farthest window of the pass left out; in the other 6 they spread at most
5 %, and only another leg's did (the single kernels' or the array path's). The same
Metal medians reproduced across passes to 0.16 to 2.44 %. The tables are cut again
by tool from the stored runs; nothing was timed again.
[The gate of record](../development/timing.md#the-gate-of-record-r33) states the
rule in full.

What each check admits:

| Runs | Under the earlier check | Under the check of record |
|---|---|---|
| NVIDIA, 2026-10-16 | 6 of 6 GPU medians and 6 of 6 ratios | the same |
| Apple, 2026-10-16 (the table and the Metal-only runs) | 2 of 8 Metal medians and 2 of 8 ratios, both at 7,077,888 cells | 8 of 8 and 8 of 8 |
| Apple array path, 2026-10-16 | 1 of 5 | 1 of 5 |
| NVIDIA, 2026-10-13, cut again | 4 of 6 ratios | 5 of 6: the Triton run at 512,000 cells enters; the CUDA-only run at 2,097,152 cells stays struck (9.2 % between its passes) |
| Apple, 2026-10-13, cut again | 0 of 5 Metal medians, 0 of 5 ratios | 5 of 5 medians, and still 0 of 5 ratios: that session's precondition failed, its five repeats spreading 5.75 % against the 5 % allowed |

One limit applies to every run the earlier check struck: the ladder stopped
judging a pass at its first failed check, so for a pass struck on its windows
alone it never judged the checks after it, and the stored run keeps none of them
to re-read. Read from the ladders' own logs, none of the passes the check of record
admits, on either date, fails any of those checks.

### Against the round's first timing, 2026-10-13

The 1.0 round first timed its code on 2026-10-13, on an earlier tree of the 1.0
series, on the same two machines; no release published those figures. On the
NVIDIA host MEEP's fastest rate, each date's own MEEP build, moved by at most 0.4 %
at every size, so the moves are the GPU's. From 2,097,152 cells up the GPU routes read within 1.1 % of
2026-10-13, apart from the CUDA-only route at 2,097,152 cells, which was struck then
and reads 5.0 % above its median now, a move these runs cannot attribute. At
512,000 cells both routes are faster (Triton 958.2 against 745.2, CUDA-only 841.5
against 687.8): a fixed cost of about 0.14 to 0.15 ms per step is gone. That is
attributed, by inference only, to this release's skip of a silent source's deposit
after its pulse has ended (no A/B run isolated it). On the Apple laptop the GPU
code, the MEEP build and the timing windows all changed at once, so no difference
is attributed. [The 1.0 timing against 2026-10-13](../development/timing.md#the-10-timing-against-2026-10-13)
compares them row by row.

## Where it does not help, or has not been measured

| Your problem | Guidance | What it rests on | Status |
|---|---|---|---|
| 3-D, 2×10^6 cells or more, long run, certified NVIDIA device | Use the GPU | The table above, per step: 3.98× and 5.04× on the Triton route, 4.59× and 5.81× on the CUDA-only route. End to end, derived: no break-even at 2,097,152 cells (the lift is shorter than MEEP's start-up) and about 460 steps at 7,077,888 ([Short runs](#short-runs-the-lift-comes-first)) | per step of record at 2,097,152 and 7,077,888 cells; end to end derived |
| 3-D, around 5×10^5 cells, long run, certified NVIDIA device | A smaller gain | The table above, per step: 1.65× on the Triton route, 1.45× on the CUDA-only route; end to end, derived, only past about 5,900 to 7,200 steps | per step of record at 512,000 cells; end to end derived |
| 3-D below 512,000 cells | Expect a smaller gain, and a loss on small grids; measure your case | On the NVIDIA host nothing smaller was timed against MEEP. On the Apple laptop the Metal route read 1.21× MEEP at 262,144 cells and 0.82× at 110,592 | of record on the Apple laptop; not measured on NVIDIA |
| 3-D on an Apple device | A gain on large grids; measure your case | On one M1 Max, per step against MEEP at 8 ranks: 1.69× at 512,000 cells, 2.40× at 2,097,152 and 2.72× at 7,077,888 ([above](#on-one-apple-m1-max)). Against the package's own NumPy reference, a default run is slower below about 125,000 cells in 3-D ([below](#small-grids-use-the-reference-or-meep-itself)) | of record on one M1 Max, under the check ruled after the runs |
| 2-D, any size | Keep using MEEP | Not a target: see [below](#two-dimensional-problems) | not measured against MEEP |
| A short run on a large grid | Keep using MEEP below the break-even step count | [Short runs](#short-runs-the-lift-comes-first) | derived |
| A supported NVIDIA GPU or Triton 3.1 release outside the certified set (compute capability 7.0 to 9.0) | Measure your case | The compiled kernels run on it by default, uncertified ([below](#a-gpu-outside-the-certified-set)); no such card has been timed | not measured |
| An NVIDIA GPU or library version outside the supported range | Expect a run slower than MEEP at your rank count | It takes the array path. On the one NVIDIA card measured, with dispatch off, the array path read 0.15×, 0.54× and 0.67× MEEP's fastest at the three sizes above ([below](#a-gpu-outside-the-certified-set)); no other card has been measured | of record on one card; not measured elsewhere |
| An Apple GPU, PyTorch version or macOS build outside the certified environment | Measure your case | Every Apple GPU is supported: the Metal kernels run on it by default, uncertified ([below](#a-gpu-outside-the-certified-set)). No Apple GPU but one M1 Max has been timed | not measured |
| Bloch-periodic, cylindrical with m ≠ 0, or complex β | Measure your case | These runs store complex fields. Their compiled kernels dispatch by default where an arm covers them, since the complex-multiply expansion licence was deleted in the 1.0 series ([Compiled-kernel dispatch](kernel-dispatch.md#correctness-under-dispatch)); none was timed against MEEP. 50 of the 194 accepted simulations of the MEEP corpus use complex storage | coverage, not speed |
| Many DFT monitors | Expect less gain | One flux monitor and zero DFT monitors were timed. Monitor accumulation runs on the array path on every kernel table | not measured |
| Dispersive, conductive or nonlinear media in 3-D | Unknown | The kernels exist; the only 3-D case timed is the one above | not measured |
| MPI, or several GPUs for one simulation | Not supported | One simulation runs on one GPU. A lift in a process that is one of several MPI ranks is refused by name | code fact |

### Two-dimensional problems

No two-dimensional problem has been timed against MEEP on a matched case, and
none is a target. Two properties of the package make small two-dimensional grids
slow on the GPU:

- **A flat cost per step.** On two 2-D cases on the NVIDIA host a dispatched step
  that holds two fused pairs costs 0.74 to 0.88 ms from 24,000 to 2,400,000 cells,
  whatever the size, on both NVIDIA kernel tables (measured 2026-10-16, on this
  release's bytes).
- **The dispatcher does not choose by cost.** It dispatches a fused kernel
  wherever one is released, including where the package's own single kernels
  are faster: against those single kernels the fused route reads a median of
  0.62 times over 46 cases, ahead on 6 of 46
  (measured 2026-10-16; 40 of the 44 distinct pairs of case and grid are at or
  below 25,000 cells, and none is above 110,592), and a plan holding
  two fused pairs first beats its single kernels between 2,400,000 and 5,400,000
  cells ([What fusing buys](kernel-dispatch.md#what-fusing-buys-and-at-which-size)).
  Setting `MEEP_GPU_FUSE_ARMS=0` runs the single kernels instead.

Expect MEEP to be faster at the sizes MEEP users run in two dimensions.

### Short runs: the lift comes first

Before the first step, the lift initializes MEEP in one process, reads the
realized state and uploads it. MEEP run on its own pays a start-up too. The step
count at which the stepping saving pays for the lift is

    steps = (lift seconds − MEEP start-up seconds) / (cells × (1 / MEEP rate − 1 / GPU rate))

with the rates in cell updates per second. Every figure in this section is
**derived** from the medians above, not timed end to end, and none counts the
GPU's own cost of its first steps (compiling and priming its kernels); below the
break-even step count, expect MEEP to finish first.

On the NVIDIA host the lift is the median over three passes of the GPU process's
MEEP set-up and lift, and MEEP's start-up its initialization plus the time its
warm-up took beyond its steady rate, at its fastest rank count:

| Cells | Lift, Triton route | Lift, CUDA-only route | MEEP start-up | Break-even, Triton route | Break-even, CUDA-only route |
|---:|---:|---:|---:|---:|---:|
| 512,000 | 7.4 s | 7.3 s | 5.4 s | about 5,900 steps | about 7,200 steps |
| 2,097,152 | 19.5 s | 19.5 s | 20.7 s | none: the lift is shorter than MEEP's start-up | none: the lift is shorter than MEEP's start-up |
| 7,077,888 | 48.5 s | 48.6 s | 43.8 s | about 460 steps | about 460 steps |

This case run to its own end time (`until=35`: 1,400, 2,240 and 3,360 steps at
the three sizes) takes, lift plus steps on the GPU against start-up plus steps on
MEEP at its fastest rank count:

| Cells | Steps | Triton route | CUDA-only route | MEEP | MEEP over the GPU (Triton, CUDA-only) |
|---:|---:|---:|---:|---:|---:|
| 512,000 | 1,400 | 8.1 s | 8.2 s | 6.6 s | 0.81×, 0.81× |
| 2,097,152 | 2,240 | 21.3 s | 21.1 s | 27.9 s | 1.31×, 1.32× |
| 7,077,888 | 3,360 | 57.0 s | 56.0 s | 86.7 s | 1.52×, 1.55× |

So at 512,000 cells MEEP finishes this case first, and from 2,097,152 cells up the
GPU does.

On the Apple laptop the lift is the median over three passes of the Metal route's
MEEP set-up and lift, and MEEP's start-up its build and initialization at 8 ranks:

| Cells | Lift | MEEP start-up, 8 ranks | Break-even |
|---:|---:|---:|---:|
| 110,592 | 1.6 s | 0.23 s | none: the GPU is slower per step |
| 262,144 | 2.8 s | 0.39 s | about 21,800 steps |
| 512,000 | 4.4 s | 0.59 s | about 8,200 steps |
| 2,097,152 | 11.5 s | 1.45 s | about 3,700 steps |
| 7,077,888 | 27.9 s | 3.30 s | about 2,700 steps |

This case's own 1,400 and 2,240 steps at 512,000 and 2,097,152 cells are short of
the break-even there; its 3,360 steps at 7,077,888 cells are past it.

Both lifts read MEEP's stored permittivity in bulk, as this release does; the
earlier figures of this page (about 19 s to 216 s on the NVIDIA host) were taken
before that change.

### A GPU outside the certified set

On NVIDIA hardware, kernels are certified on one device identity per kernel
table: compute capability 8.6 with Triton 3.1.0, or compute capability 8.6 for
the hand-written CUDA table. They are supported on compute capability 7.0 to 9.0
with Triton 3.1, where they run by default, uncertified, and print a note saying
so; their speed there has not been measured
([Certification on an NVIDIA GPU](kernel-dispatch.md#certification-on-an-nvidia-gpu)).
A device or version outside the supported range is refused by name and the run
takes the array path, as is any uncertified one under
`MEEP_GPU_HARDWARE=certified`. On the NVIDIA card measured, with dispatch off,
the array path read 84.4, 349.9 and 374.0 Mcell-steps/s at 512,000, 2,097,152
and 7,077,888 cells: 0.15×, 0.54× and 0.67× MEEP's fastest, slower than MEEP at
each size (of record, 2026-10-16, under the same conditions as
[the table above](#on-one-nvidia-rtx-a6000)). A device whose identity cannot be
read is not refused.

On Apple hardware every GPU is supported: the Metal kernels are meant to run
on each, and run there by default. They are certified on one environment: GPU
architecture `applegpu_g13s` (an M1 Max), PyTorch 2.10.0, Metal frontend
`metalfe-32023.850.10`, and `PYTORCH_MPS_FAST_MATH` unset or `0`. Another
Apple GPU architecture, PyTorch version or macOS build runs the same kernels
uncertified, and prints a note saying so; their speed there has not been
measured. With `MEEP_GPU_HARDWARE=certified`, an environment that is not
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
  any size but the three timed on the NVIDIA host and the five timed on the Apple
  laptop.
- A run with DFT monitors, or with more than one flux monitor.
- A two-dimensional case against MEEP on a matched problem.
- The end-to-end time from constructing the simulation to reading a result,
  timed as such: the end-to-end figures above are derived from the lift and the
  step rates, and leave out the GPU's own cost of its first steps.
- Any GPU but one RTX A6000 and one M1 Max.
- Power or energy: no figure per watt has been measured.
- MEEP with MPI ranks and OpenMP threads each, as MEEP's best: measured on the
  NVIDIA host, and struck by a defect of the timing harness
  ([above](#on-one-nvidia-rtx-a6000)).
- A native-tuned MEEP build, and the dispatched NVIDIA route under `flush`.
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
