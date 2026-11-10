# Will it help?

meep-gpu is aimed at large three-dimensional problems on one NVIDIA or Apple
GPU, for users without a cluster. Expect no gain over MEEP on two-dimensional
problems, on small grids, or on short runs of large grids; most of these cases
have not been timed against MEEP, and the table below says which rest on a
measurement. This page states each case with what it rests on, and says what
has not been measured. To measure your own problem, see
[Time your own simulation](time-your-simulation.md).

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

The 1.0 timing compared the package with MEEP on the identical problem,
on the bytes of 1.0, on one NVIDIA host and on one Apple laptop, on
2026-10-16. The timed runs recorded the files they imported, and every package
file among them is byte-identical to 1.0's. On each machine the released
timing ladder
([Running the timing ladder](../development/timing.md)) interleaved the GPU runs
and MEEP's runs per size and pass, in one session, on an otherwise idle host, and
the tables of that timing were cut by tool from the stored runs. The run records behind
these figures are not in this repository. The 1.0.1 timings, [on the Apple
laptop](#the-101-timing-on-the-same-laptop) and [on one RTX 3090](#the-101-timing-on-one-rtx-3090),
were made later and state their own conditions.

The comparison of 2026-09-28 that earlier versions of this page stated was made on
a development tree that predates 1.0, and the first 1.0 timing
(2026-10-13) on an earlier tree of the 1.0 series; neither describes 1.0
([Against the first 1.0 timing](#against-the-first-10-timing-2026-10-13)).

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

The Triton route is what a default 1.0 run took on a certified NVIDIA host with
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
  had been timed against MEEP on that date.
- **The reference MEEP, single precision on both sides.** MEEP 1.33.0 as the
  1.0 release's reference environment builds it: the release
  tarball pinned by SHA-256, single precision, MPI (Open MPI 5.0.8) with OpenMP
  linked, no floating-point contraction (`-O2 -ffp-contract=off -fno-fast-math`)
  and a portable binary (`-march=x86-64 -mtune=generic`). It is neither a
  stock-flags nor a native-tuned build; no native-tuned build was timed. The MEEP
  build the first 1.0 timing used (`-g -O2`, a portable binary, built
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
ratios are of record**, under the check adopted after these runs were seen; under the
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
- **The reference MEEP, single precision on both sides.** MEEP 1.33.0 as the
  1.0 release's reference environment builds it for macOS: single precision, Open MPI
  5.0.10, no floating-point contraction (`-O2 -ffp-contract=off -fno-fast-math`),
  a portable binary. 1, 4, 8 and 10 ranks were timed (10 is every core,
  efficiency cores included), unbound, because Open MPI does not bind ranks on
  macOS; 8 ranks is the fastest at every size. The build the first 1.0 Apple
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

This check, set on 2026-10-16, replaced the one in force when these
runs were made, and it was adopted after the 2026-10-16 runs had been seen. Under the
earlier check a pass in which any leg's timed windows spread above 5 % failed, and
so did the run. It was replaced because a published figure is a median of pass
medians, so what has to hold is that the medians reproduce. On the Apple laptop the
earlier check struck 6 of 8 Metal medians (17 of 24 timed passes): in 11 of the 17
passes the Metal route's own 2 s windows spread 5.2 to 10.3 %, and 2.1 to 5.1 % with
the single farthest window of the pass left out; in the other 6 they spread at most
5 %, and only another leg's did (the single kernels' or the array path's). The same
Metal medians reproduced across passes to 0.16 to 2.44 %. The tables are cut again
by tool from the stored runs; nothing was timed again.
[The gate of record](../development/timing.md#the-gate-of-record) states the
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

### Against the first 1.0 timing, 2026-10-13 {#against-the-first-10-timing-2026-10-13}

The 1.0 code was first timed on 2026-10-13, on an earlier tree of the 1.0
series, on the same two machines; no release published those figures. On the
NVIDIA host MEEP's fastest rate, each date's own MEEP build, moved by at most 0.4 %
at every size, so the moves are the GPU's. From 2,097,152 cells up the GPU routes read within 1.1 % of
2026-10-13, apart from the CUDA-only route at 2,097,152 cells, which was struck then
and reads 5.0 % above its median now, a move these runs cannot attribute. At
512,000 cells both routes are faster (Triton 958.2 against 745.2, CUDA-only 841.5
against 687.8): a fixed cost of about 0.14 to 0.15 ms per step is gone. That is
attributed, by inference only, to 1.0's skip of a silent source's deposit
after its pulse has ended (no A/B run isolated it). On the Apple laptop the GPU
code, the MEEP build and the timing windows all changed at once, so no difference
is attributed. [The 1.0 timing against 2026-10-13](../development/timing.md#the-10-timing-against-2026-10-13)
compares them row by row.

### The 1.0.1 timing on the same laptop

The 1.0.1 code was timed on the same M1 Max against MEEP 1.35.0, on four cases at
three sizes, from 2026-10-31 to 2026-11-01. [The 1.0 table](#on-one-apple-m1-max) stays
as the record of the 1.0 code against MEEP 1.33.0. In Mcell-steps/s:

| Case | Cells | Metal route | MEEP, 8 ranks (fastest) | Ratio | Array path, NumPy on the host |
|---|---:|---:|---:|---:|---:|
| Same case as above | 512,000 | 1,024.1 (passed, 2.18 %) | 457.0 (3.49 %) | 2.24× | 20.1 (struck, 11.64 %) |
| Same case as above | 2,097,152 | 1,343.2 (passed, 0.70 %) | 450.3 (0.72 %) | 2.98× | 23.4 (struck, 5.58 %) |
| Same case as above | 7,077,888 | 1,501.2 (passed, 0.27 %) | 502.0 (2.87 %) | 2.99× | 21.6 (passed, 3.34 %) |
| Eight DFT objects | 512,000 | 750.4 (passed, 0.87 %) | 459.7 (0.84 %) | 1.63× | 21.9 (struck, 10.38 %) |
| Eight DFT objects | 2,097,152 | 1,228.5 (passed, 0.38 %) | 446.8 (5.46 %) | 2.75× | 22.7 (struck, 8.70 %) |
| Eight DFT objects | 7,077,888 | 1,446.3 (passed, 0.50 %) | 505.8 (0.33 %) | 2.86× | 21.6 (passed, 2.32 %) |
| Lorentz slab | 512,000 | 643.5 (passed, 0.51 %) | 315.5 (19.13 %) | 2.04× | 32.4 (passed, 2.52 %) |
| Lorentz slab | 2,097,152 | 881.0 (passed, 0.88 %) | 333.9 (1.31 %) | 2.64× | 38.8 (passed, 4.95 %) |
| Lorentz slab | 7,077,888 | 805.2 (passed, 1.54 %) | 372.9 (2.24 %) | 2.16× | 40.5 (passed, 0.33 %) |
| Hz source | 512,000 | 375.2 (struck, 0.27 %) | 465.2 (4.33 %) | none | 19.3 (struck, 17.03 %) |
| Hz source | 2,097,152 | 846.2 (struck, 0.17 %) | 451.6 (2.45 %) | none | 21.4 (struck, 13.67 %) |
| Hz source | 7,077,888 | 1,246.5 (struck, 1.56 %) | 502.8 (1.33 %) | none | 21.6 (passed, 3.03 %) |

Each rate is the median of three passes, each pass in a fresh process, with its
status under [the timing check of record](#the-timing-check-and-the-change-made-after-these-runs)
and the spread of its three pass medians (MEEP's is recorded and not gated). **9 of the
12 Metal medians passed and their 9 ratios are of record.** The 3 Hz-source medians are
struck, so no ratio uses them: their pass medians reproduce, but the single-kernel leg
beside them launched no kernel, so those runs cannot prove what the fused pair
replaced. Every Metal run launched the Metal table and was bit-identical to the array
path (36 of 36). The array path failed the check at 6 of the 12 rows.

MEEP at each rank count (the reference build; Mcell-steps/s, median of three passes):

| Case | Cells | 1 | 3 | 4 | 5 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Same case as above | 512,000 | 95.1 | 235.4 | 323.5 | 298.3 | 457.0 | 171.2 |
| Same case as above | 2,097,152 | 99.5 | 250.4 | 336.7 | 321.6 | 450.3 | 226.3 |
| Same case as above | 7,077,888 | 106.9 | 273.5 | 362.5 | 354.2 | 502.0 | 295.5 |
| Eight DFT objects | 512,000 | 95.9 | 227.3 | 312.8 | 297.3 | 459.7 | 175.9 |
| Eight DFT objects | 2,097,152 | 100.7 | 249.5 | 330.9 | 320.0 | 446.8 | 228.8 |
| Eight DFT objects | 7,077,888 | 105.0 | 271.4 | 356.6 | 350.9 | 505.8 | 293.0 |
| Lorentz slab | 512,000 | 78.0 | 178.6 | 224.8 | 214.7 | 315.5 | 151.6 |
| Lorentz slab | 2,097,152 | 86.3 | 205.2 | 264.8 | 240.0 | 333.9 | 193.9 |
| Lorentz slab | 7,077,888 | 92.4 | 224.2 | 291.8 | 273.0 | 372.9 | 242.1 |
| Hz source | 512,000 | 95.1 | 230.7 | 294.4 | 299.9 | 465.2 | 180.5 |
| Hz source | 2,097,152 | 100.5 | 259.4 | 335.8 | 320.4 | 451.6 | 228.2 |
| Hz source | 7,077,888 | 106.8 | 273.0 | 363.7 | 354.1 | 502.8 | 294.2 |

**Read every number with these conditions:**

- **The cases.** The first is the case of the 1.0 table: a dielectric sphere under
  PML with one flux monitor. "Eight DFT objects" adds six flux planes closing a box
  around the sphere, at 20 frequencies each, and one DFT field plane for Ex, Ey and Ez
  at 5 frequencies. "Lorentz slab" replaces the sphere with an axis-aligned slab of a
  medium with one Lorentz pole, through the absorber in x and z, with an Ez source.
  "Hz source" is the first case with its source changed to Hz at the same point. No
  medium with more than one pole was timed against MEEP.
- **MEEP's spread.** At 8 ranks two rows spread above 5 %. On the Lorentz slab at 512,000 cells
  MEEP's three passes at 8 ranks read 317.0, 315.5 and 256.6 (19.13 %), so that row's
  2.04× is against a MEEP run whose third pass fell. On eight DFT objects at 2,097,152
  cells they read 446.8, 451.5 and 427.1 (5.46 %). The other 18 of the 72 MEEP medians above 5 % are at rank counts slower than 8.
- **Which code.** The Lorentz rows were timed with the fix to dispersive media. The
  other nine were timed first, without it. The fix changed no kernel those rows launch,
  and a check with the fix in place launched the same kernels on the same slots at 9 of
  9 rows. A check is one pass of short windows and is not a timing.
- **The reference MEEP, single precision on both sides.** MEEP 1.35.0 as this release's
  reference environment builds it for macOS: single precision, no floating-point
  contraction, a portable binary. 1, 3, 4, 5, 8 and 10 ranks were timed, unbound; 8 ranks
  is the fastest at all 12 rows.
- **The GPU's timing windows.** Six 2 s windows per leg, interleaved, after one full
  window per leg that enters no statistic, with held residency (the default).
- **A quiet host, checked, on AC power.** Each timed unit started after two clear
  readings 120 s apart, and a watcher read the host every 15 s through every unit (2,404
  readings, no intrusion). Before each session one point was repeated five times; the
  first attempt of the first session spread 6.16 % on the Metal route, was not used, and
  was repeated at 0.87 %; the other two sessions read 0.45 % and 0.88 %.

Nothing here is claimed for another Apple GPU, another case or size, a MEEP build other
than the reference, or against one MEEP rank. 110,592 and 262,144 cells were not timed
on the 1.0.1 code.

### The 1.0.1 timing on one RTX 3090

The 1.0.1 code was timed on one RTX 3090 against MEEP 1.35.0, on four cases at three sizes,
from 2026-10-29 to 2026-10-30. [The 1.0 table](#on-one-nvidia-rtx-a6000) stays as the record of
the 1.0 code against MEEP 1.33.0, on the RTX A6000. In Mcell-steps/s:

| Case | Cells | Triton route | CUDA route | MEEP, fastest rank count | Array path, CuPy on the GPU |
|---|---:|---:|---:|---:|---:|
| Same case as above | 512,000 | 1,402.7 (5.34×; passed, 1.80 %) | 1,275.1 (4.86×; passed, 3.54 %) | 262.6 (8 ranks; 1.17 %) | 100.6 (0.38×; passed, 3.16 %) |
| Same case as above, first attempt | 512,000 | 1,386.2 (struck, 8.85 %) | 1,247.2 (struck, 6.75 %) | 263.6 (8 ranks; 1.44 %) | 100.0 (passed, 4.54 %) |
| Same case as above | 2,097,152 | 2,871.4 (17.54×; passed, 0.66 %) | 3,535.3 (21.59×; passed, 0.48 %) | 163.7 (4 ranks; 1.82 %) | 387.5 (2.37×; passed, 0.90 %) |
| Same case as above | 7,077,888 | 3,221.7 (18.76×; passed, 0.08 %) | 3,757.0 (21.88×; passed, 0.26 %) | 171.7 (4 ranks; 1.56 %) | 438.3 (2.55×; passed, 0.11 %) |
| Eight DFT objects | 512,000 | 1,058.7 (4.13×; passed, 4.14 %) | 998.2 (3.89×; passed, 2.14 %) | 256.5 (8 ranks; 2.99 %) | 100.1 (0.39×; passed, 0.98 %) |
| Eight DFT objects | 2,097,152 | 2,931.4 (17.83×; passed, 2.14 %) | 3,571.2 (21.72×; passed, 0.98 %) | 164.4 (4 ranks; 1.78 %) | 406.0 (2.47×; passed, 1.35 %) |
| Eight DFT objects | 7,077,888 | 3,382.0 (20.14×; passed, 0.91 %) | 3,939.1 (23.46×; passed, 0.90 %) | 167.9 (4 ranks; 5.29 %) | 466.1 (2.78×; passed, 1.85 %) |
| Lorentz slab | 512,000 | 705.2 (5.78×; passed, 3.28 %) | 686.0 (struck, 5.32 %) | 122.1 (8 ranks; 0.90 %) | 122.3 (1.00×; passed, 3.92 %) |
| Lorentz slab | 2,097,152 | 1,562.2 (14.67×; passed, 2.02 %) | 1,631.0 (15.32×; passed, 3.34 %) | 106.5 (4 ranks; 1.16 %) | 457.4 (4.30×; passed, 4.85 %) |
| Lorentz slab | 7,077,888 | 1,707.8 (15.58×; passed, 3.55 %) | 1,831.1 (16.71×; passed, 3.28 %) | 109.6 (4 ranks; 2.54 %) | 580.4 (5.30×; passed, 2.43 %) |
| Hz source | 512,000 | 822.7 (3.18×; passed, 3.91 %) | 782.4 (3.02×; passed, 2.35 %) | 258.9 (8 ranks; 3.73 %) | 97.7 (0.38×; passed, 4.93 %) |
| Hz source | 2,097,152 | 2,806.9 (17.02×; passed, 1.04 %) | 3,028.7 (18.36×; passed, 2.80 %) | 164.9 (4 ranks; 0.52 %) | 390.9 (2.37×; passed, 1.96 %) |
| Hz source | 7,077,888 | 3,238.1 (19.13×; passed, 1.46 %) | 3,766.3 (22.25×; passed, 1.21 %) | 169.3 (4 ranks; 0.73 %) | 445.1 (2.63×; passed, 1.09 %) |

Each rate is the median of three passes, each pass in a fresh process. In brackets: the rate
over MEEP's in the same row, per step; the run's status under
[the timing check of record](#the-timing-check-and-the-change-made-after-these-runs); and the
spread of its three pass medians (MEEP's is recorded and not gated). **23 of the 24 GPU medians
passed and their 23 ratios are of record.** The CUDA route's median on the Lorentz slab at 512,000
cells is struck at 5.32 %, so no ratio uses it. Every ratio is against the fastest MEEP rank count
in its row. No check struck any of the 72 passes behind the 24 medians. The array path passed at
12 of the 12 rows. The first-attempt row is not one of the 12: see the retake below.

MEEP at each rank count (the reference build; Mcell-steps/s, median of three passes). The last two
columns are controls of the same sessions, with MEEP's cost-based chunk splitting
(<code>split_chunks_evenly=False</code>). No ratio is against them, and at 6 of the 12 rows one
reads above the fastest plain rank count, by 0.15 % to 1.21 %:

| Case | Cells | 1 | 4 | 8 | 5, cost-based split | 8, cost-based split |
|---|---:|---:|---:|---:|---:|---:|
| Same case as above | 512,000 | 93.6 | 236.0 | 262.6 | 245.0 | 264.1 |
| Same case as above | 2,097,152 | 90.1 | 163.7 | 163.3 | 161.4 | 164.1 |
| Same case as above | 7,077,888 | 101.3 | 171.7 | 165.5 | 165.3 | 165.6 |
| Eight DFT objects | 512,000 | 97.1 | 225.0 | 256.5 | 236.8 | 256.8 |
| Eight DFT objects | 2,097,152 | 97.8 | 164.4 | 161.0 | 165.5 | 161.0 |
| Eight DFT objects | 7,077,888 | 107.3 | 167.9 | 160.4 | 162.4 | 162.9 |
| Lorentz slab | 512,000 | 61.9 | 115.9 | 122.1 | 120.7 | 122.5 |
| Lorentz slab | 2,097,152 | 65.4 | 106.5 | 105.2 | 106.1 | 105.5 |
| Lorentz slab | 7,077,888 | 71.4 | 109.6 | 105.7 | 107.5 | 105.3 |
| Hz source | 512,000 | 93.9 | 228.9 | 258.9 | 238.0 | 262.0 |
| Hz source | 2,097,152 | 93.6 | 164.9 | 162.7 | 160.3 | 162.2 |
| Hz source | 7,077,888 | 103.9 | 169.3 | 162.3 | 162.0 | 162.3 |

**Read every number with these conditions:**

- **One GPU against one CPU host.** One RTX 3090 at compute capability 8.6 (CuPy 13.5.1, Triton
  3.1.0), under WSL2 on Windows, against MEEP on the i9-12900K of the same machine. The GPU's clocks
  were locked at 1695 MHz graphics and 9501 MHz memory, and read those values at every reading.
- **A split ladder.** The GPU legs and the MEEP legs of each row were timed as two child runs of one
  session: the GPU half under VM affinity <code>0x999988</code>, the MEEP half with the mask cleared
  (Windows' 24 logical CPUs). The GPU half ran first at 12 of 12 rows. A drift between the two
  halves enters that row's ratios in full.
- **The routes.** The Triton route sets no table preference. The CUDA route sets
  <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code> on a host that also has Triton. On three cases it
  launched the CUDA table on 4 of 4 slots. On the Lorentz slab it launched the CUDA table on 2 of 5
  slots and Triton on the other 3, so that case's column is not the CUDA table alone.
  <code>MEEP_GPU_PREFER</code> was <code>span</code> on every timed GPU run, so whether a default
  run takes the timed plan is not established.
- **The cases.** As in [the Apple 1.0.1 timing](#the-101-timing-on-the-same-laptop). There the
  Hz-source medians are struck; here the single-kernel legs launched and those medians enter.
- **The reference MEEP, single precision on both sides.** MEEP 1.35.0 as this release's reference
  environment builds it: single precision, Open MPI 5.0.8, no floating-point contraction
  (`-O2 -ffp-contract=off -fno-fast-math`), a portable binary (`-march=x86-64 -mtune=generic`). 1, 4
  and 8 ranks were timed. 8 ranks is the fastest at 512,000 cells (4 of 4 rows) and 4 ranks at
  2,097,152 and 7,077,888 cells (8 of 8 rows).
- **MEEP barely scales from 4 to 8 ranks on this desktop** at 2,097,152 and 7,077,888 cells: 8 ranks
  reads 0.96 to 1.00 times 4 ranks. These ratios rest on that rate as much as on the GPU's.
- **MEEP's spread.** 3 of the 60 MEEP medians spread above 5 %. One is a row's fastest: eight DFT
  objects at 7,077,888 cells, 4 ranks, 5.29 %. The other two are controls.
- **MEEP's rate varied between sessions.** The same command on the reference build, the first case
  at 7,077,888 cells on 8 ranks, read 158.9 to 222.4 Mcell-steps/s across hours of 2026-11-02. Its
  median in the table's session was 165.5. Nice level, library path and the session's host readings
  were each tried and do not explain it; the cause is not established. The 5 % check does not see
  this, because it reads one row's passes. The effect: at the highest median seen (214.1), that row
  reads 15.05× on the Triton route and 17.55× on the CUDA route, against the table's 18.76× and
  21.88×.
- **A native MEEP control.** A build with `-O3 -march=native` against the reference build, 8 ranks,
  same ladder, native first: 283.8 and 288.1 at 512,000 cells, 195.2 and 204.7 at 2,097,152, 211.3
  and 214.1 at 7,077,888, and 172.0 and 168.6 in a repeat of that size. No gain above the variation.
  No record takes it.
- **The subnormal policy.** The GPU runs step under `keep`, beside MEEP under the flush that
  importing MEEP sets on x86.
- **Which code.** The rows were timed before the round's last commits. 141 of the 141 source files
  their import records name are byte-equal at commit <code>404d432d0b34</code>. 2 of the 3 ledgers
  they recorded are not (the two <code>fingerprints.json</code>), and a moved ledger [owes a row a
  dispatch check](../development/timing.md#valid-stale-or-owed-a-dispatch-check). On that commit, on
  this card, the check launched the same arms on the same slots at 12 of 12 keys. A check is one
  pass of short windows and is not a timing. Check rates over row rates read a median of 0.92 (CUDA
  route) and 0.96 (Triton route), lowest at 512,000 cells (0.64 to 0.85). A session on that commit
  under the same conditions, with full windows and 3 passes, which no record takes, read the first
  case at 512,000 cells at 1,364.9 (Triton route) and 1,221.9 (CUDA route): 0.973 and 0.958 of the
  row, where the short check read 0.641 and 0.708.
- **The retake.** The first case at 512,000 cells is the one retake (1 of 12 rows), declared after
  the first attempt was seen. That attempt read 1,386.2 on the Triton route, struck at 8.85 %, and
  1,247.2 on the CUDA route, struck at 6.75 %. It is in the table beside the retake, and no ratio
  uses it.
- **The GPU's timing windows.** Each pass records the spread of its windows, and no table gates on
  it. On the dispatched leg it reaches 12.49 %, in the retake's first pass.
- **A quiet host, checked.** Each timed unit started after two clear readings, and a watcher read
  the host every 15 s through every unit (2,300 readings, no intrusion). Before each of the 5
  sessions one point on the Triton route was repeated five times: spreads 1.33 % to 3.74 %,
  precondition GO at 12 of 12 rows.

Nothing here is claimed across hosts or vendors, for another NVIDIA GPU, another case or size, a
MEEP build other than the reference, against one MEEP rank, or per watt. Do not compare these ratios
with the RTX A6000's or the M1 Max's: each table's MEEP ran on that machine's own CPU, and the code
and the MEEP version differ from the RTX A6000's. 110,592 and 262,144 cells were not timed. [Short
runs](#short-runs-the-lift-comes-first) covers the RTX A6000 and the M1 Max only: no lift or
break-even figure is stated for the RTX 3090.

## Where it does not help, or has not been measured

| Your problem | Guidance | What it rests on | Status |
|---|---|---|---|
| 3-D, 2×10^6 cells or more, long run, certified NVIDIA device | Use the GPU | On one RTX A6000, on the 1.0 code ([above](#on-one-nvidia-rtx-a6000)), per step: 3.98× and 5.04× on the Triton route, 4.59× and 5.81× on the CUDA-only route. End to end, derived: no break-even at 2,097,152 cells (the lift is shorter than MEEP's start-up) and about 460 steps at 7,077,888 ([Short runs](#short-runs-the-lift-comes-first)). On one RTX 3090, on the 1.0.1 code, per step against MEEP on that desktop: 17.54× and 18.76× on the Triton route, 21.59× and 21.88× on the CUDA route ([above](#the-101-timing-on-one-rtx-3090)); MEEP's rate there varied between sessions, and at the highest median seen 7,077,888 cells reads 15.05× and 17.55×. No end-to-end figure is stated for that card. | per step of record at 2,097,152 and 7,077,888 cells; end to end derived on the RTX A6000 only |
| 3-D, around 5×10^5 cells, long run, certified NVIDIA device | A smaller gain | On one RTX A6000, on the 1.0 code ([above](#on-one-nvidia-rtx-a6000)), per step: 1.65× on the Triton route, 1.45× on the CUDA-only route; end to end, derived, only past about 5,900 to 7,200 steps. On one RTX 3090, on the 1.0.1 code, per step: 5.34× on the Triton route, 4.86× on the CUDA route, on the declared retake ([above](#the-101-timing-on-one-rtx-3090)). No end-to-end figure is stated for that card. | per step of record at 512,000 cells; end to end derived on the RTX A6000 only |
| 3-D below 512,000 cells | Expect a smaller gain, and a loss on small grids; measure your case | On the NVIDIA host nothing smaller was timed against MEEP. On the Apple laptop the Metal route read 1.21× MEEP at 262,144 cells and 0.82× at 110,592 | of record on the Apple laptop; not measured on NVIDIA |
| 3-D on an Apple device | A gain on large grids; measure your case | On one M1 Max, per step against MEEP at 8 ranks: 1.69× at 512,000 cells, 2.40× at 2,097,152 and 2.72× at 7,077,888 on the 1.0 code ([above](#on-one-apple-m1-max)), and on the 1.0.1 code 2.24×, 2.98× and 2.99× on the same case, 1.63× to 2.86× with eight DFT objects and 2.04× to 2.64× on a one-pole Lorentz slab ([above](#the-101-timing-on-the-same-laptop)). Against the package's own NumPy reference, on the 1.0 code the Metal route read 261.1 Mcell-steps/s at 110,592 cells and the reference 21.5 (struck), so the 3-D crossover lies below that size ([below](#small-grids-use-the-reference-or-meep-itself)) | of record on one M1 Max: the 1.0 figures under the check adopted after those runs, the 1.0.1 figures under that check, set before them |
| 2-D, any size | Keep using MEEP | Not a target: see [below](#two-dimensional-problems) | not measured against MEEP |
| A short run on a large grid | Keep using MEEP below the break-even step count | [Short runs](#short-runs-the-lift-comes-first) | derived |
| A supported NVIDIA GPU or Triton release outside the certified set (one the installed CuPy or Triton can run on) | Measure your case | The compiled kernels run on it by default, uncertified ([below](#a-gpu-outside-the-certified-set)); no such card has been timed | not measured |
| An NVIDIA GPU outside what is supported (one no kernel table can compile for) | Measure your case | It takes the array path unless `MEEP_GPU_HARDWARE=any`. On one RTX A6000, on the 1.0 code, with dispatch off, the array path read 0.15×, 0.54× and 0.67× MEEP's fastest at the three sizes above ([below](#a-gpu-outside-the-certified-set)). On one RTX 3090, on the 1.0.1 code, against MEEP on that desktop, it read 0.38× to 0.39× at 512,000 cells on three cases and 1.00× on the Lorentz slab, and 2.37× to 5.30× from 2,097,152 cells up ([above](#the-101-timing-on-one-rtx-3090)). No run separates the card from the host's MEEP | of record on one RTX A6000 and one RTX 3090 |
| An Apple GPU, PyTorch version or macOS build outside the certified environment | Measure your case | Every Apple GPU is supported: the Metal kernels run on it by default, uncertified ([below](#a-gpu-outside-the-certified-set)). No Apple GPU but one M1 Max has been timed | not measured |
| Bloch-periodic, cylindrical with m ≠ 0, or complex β | Measure your case | These runs store complex fields. Their compiled kernels dispatch by default where an arm covers them, since the complex-multiply expansion licence was deleted in the 1.0 series ([Compiled-kernel dispatch](kernel-dispatch.md#correctness-under-dispatch)); none was timed against MEEP. 50 of the 194 accepted simulations of the 1.0 MEEP corpus use complex storage | coverage, not speed |
| Many DFT monitors | Measure your case | On the RTX A6000, one flux monitor and zero DFT monitors were timed. On one RTX 3090, eight DFT objects read 4.13× and 3.89× at 512,000 cells (Triton, CUDA route), below the first case there, and 17.83× to 23.46× from 2,097,152 cells up, above it, in a different session ([above](#the-101-timing-on-one-rtx-3090)). On the Apple laptop, eight DFT objects read 1.63× to 2.86× MEEP ([above](#the-101-timing-on-the-same-laptop)). Monitor accumulation runs on the array path on every kernel table | not measured on the RTX A6000; of record on one M1 Max and one RTX 3090 |
| Dispersive, conductive or nonlinear media in 3-D | Unknown, except one case | The kernels exist. On the Apple laptop a one-pole Lorentz slab read 2.04× to 2.64× MEEP ([above](#the-101-timing-on-the-same-laptop)). On one RTX 3090 that slab read 5.78× at 512,000 cells on the Triton route and 14.67× to 16.71× from 2,097,152 cells up, with the CUDA route's 512,000-cell median struck ([above](#the-101-timing-on-one-rtx-3090)). No other such case has been timed. | of record on one M1 Max and one RTX 3090 for that case (on the RTX 3090, 5 of 6 medians); otherwise not measured |
| MPI, or several GPUs for one simulation | Not supported | One simulation runs on one GPU. A lift in a process that is one of several MPI ranks is refused by name | code fact |

### Two-dimensional problems

No two-dimensional problem has been timed against MEEP on a matched case, and
none is a target. Two properties of the package make small two-dimensional grids
slow on the GPU:

- **A flat cost per step.** On two 2-D cases on the NVIDIA host a dispatched step
  that holds two fused pairs costs 0.74 to 0.88 ms from 24,000 to 2,400,000 cells,
  whatever the size, on both NVIDIA kernel tables (measured 2026-10-16, on the
  bytes of 1.0).
- **Fusing does not pay at these sizes.** Against the package's own single kernels the fused route
  read a median of 0.62 times over 46 cases, ahead on 6 of 46 (measured 2026-10-16; 40 of the 44
  distinct pairs of case and grid are at or below 25,000 cells, and none is above 110,592), and a
  plan holding two fused pairs first beat its single kernels between 2,400,000 and 5,400,000 cells.
  In 1.0.1 a fused product this GPU's timing record measures slower beyond its measured spread gives
  way to its single kernels ([What fusing
  buys](kernel-dispatch.md#what-fusing-buys-and-at-which-size)). Setting `MEEP_GPU_FUSE_ARMS=0` runs
  the single kernels instead.

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

On the RTX A6000, on the 1.0 code, the lift is the median over three passes of the GPU process's
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

On the Apple laptop, on the 1.0 code, the lift is the median over three passes of the Metal route's
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

On NVIDIA hardware, kernels are certified on two identities per table: compute capability 8.6 or 8.9 with
Triton 3.1.0, or 8.6 or 8.9 for the hand-written CUDA table, each beside its certified stack. They are supported
on any device the installed CuPy or Triton (3.1 or later) can run on, where outside that identity
they run by default, uncertified, and print a note saying so; their speed there has not been measured
([Certification on an NVIDIA GPU](kernel-dispatch.md#certification-on-an-nvidia-gpu)). A device that
is not supported is refused by name and the run takes the array path unless `MEEP_GPU_HARDWARE=any`, as is any uncertified device or
stack under `MEEP_GPU_HARDWARE=certified`. On one RTX A6000, on the 1.0 code, with
dispatch off, the array path read 84.4, 349.9 and 374.0 Mcell-steps/s at 512,000, 2,097,152 and
7,077,888 cells: 0.15×, 0.54× and 0.67× MEEP's fastest on that host's 48 cores (of record,
2026-10-16, under the same conditions as [the table above](#on-one-nvidia-rtx-a6000)). On one RTX
3090, on the 1.0.1 code, the same case read 100.6, 387.5 and 438.3 Mcell-steps/s: 0.38×, 2.37× and
2.55× MEEP's fastest on that desktop, at 4 or 8 ranks ([above](#the-101-timing-on-one-rtx-3090)).
The two hosts differ in GPU, CPU, MEEP version and code, and no run separates them, so measure your
own case. A device whose identity cannot be read is not refused.

On Apple hardware every GPU is supported: the Metal kernels are meant to run
on each, and run there by default. They are certified on two environments: GPU
architecture `applegpu_g13s` (an M1 Max) with Metal frontend `metalfe-32023.921.5`, and
`applegpu_g16g` (an M4) with `metalfe-32023.886.1`, each with PyTorch 2.10.0 and `PYTORCH_MPS_FAST_MATH` unset or `0`. Another
Apple GPU architecture, PyTorch version or macOS build (macOS 27.2 with Metal
frontend `metalfe-32023.922.4`, to which the certified M1 Max moved after these
runs, included) runs the same kernels uncertified, and prints a note saying so;
their speed there has not been measured. With `MEEP_GPU_HARDWARE=certified`, an
environment that is not
certified, or that cannot be judged, is refused by name and the run takes the
array path, NumPy on the host CPU
([Certification on an Apple GPU](kernel-dispatch.md#certification-on-an-apple-gpu)).

## Small grids: use the reference, or MEEP itself

A default lift pays a fixed cost per step and a start-up cost, so below a
crossover size the package's own NumPy reference is faster than its GPU route.
Pass `prefer_gpu=False` for the reference, or set `MEEP_GPU_DISPATCH=0` to reach
the array path without editing the script; that size was measured on pre-1.0
code (below). And where MEEP itself is faster than both, use MEEP: the package
does not make a simulation MEEP already runs quickly any quicker.

On the Apple host, on pre-1.0 code (2026-09-27; vacuum, an absorber on every
side, one point source, real fields; a default lift against `prefer_gpu=False`):

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
three-dimensional rows are 1 pass. On that code the 2-D crossover was between
102,400 and 230,400 cells; on the 1.0 code the 3-D one is below 110,592 cells. A
default lift also paid about one to two seconds once, in the lift and the first
step. This is a comparison with the package's own reference, not with MEEP.

No NVIDIA host has been measured this way. There the array path itself runs on
the GPU, at the rates given [above](#a-gpu-outside-the-certified-set).

## What has not been measured

- Any three-dimensional case but the one above, and that case against MEEP at any size but the three
  timed on the NVIDIA host and the five timed on the Apple laptop. The Apple laptop's 1.0.1 timing
  adds three cases at three sizes ([above](#the-101-timing-on-the-same-laptop)), one of them struck.
  The RTX 3090 timing adds the same three cases at 512,000, 2,097,152 and 7,077,888 cells
  ([above](#the-101-timing-on-one-rtx-3090)), 1 of its 24 medians struck.
- A run with DFT monitors, or with more than one flux monitor, on any host but the Apple
  laptop and the RTX 3090 host.
- A two-dimensional case against MEEP on a matched problem.
- The end-to-end time from constructing the simulation to reading a result,
  timed as such: the end-to-end figures above are derived from the lift and the
  step rates, and leave out the GPU's own cost of its first steps.
- Any GPU but one RTX A6000, one M1 Max and one RTX 3090.
- Power or energy: no figure per watt has been measured.
- MEEP with MPI ranks and OpenMP threads each, as MEEP's best: measured on the
  NVIDIA host, and struck by a defect of the timing harness
  ([above](#on-one-nvidia-rtx-a6000)).
- A native-tuned MEEP build as a timing of record: one control on the RTX 3090 host showed no gain
  above the variation there ([above](#the-101-timing-on-one-rtx-3090)).
- The dispatched NVIDIA route under `flush`.
- The timing of the hand-written CUDA table composing alone on a host with no
  usable Triton. (That composition has run the installation check, one
  example and one route leg with Triton withheld from the process, on one
  device, untimed; the CUDA-only column above ran the CUDA table alone on
  a host that also had Triton.)
- The time of a lift from a double-precision MEEP build, or from MEEP 1.34.0.
  Both have been run, not timed.
- A speed comparison with a double-precision MEEP. Correctness runs against
  double-precision MEEP 1.33.0, 1.34.0 and 1.35.0 are listed in
  [INSTALL.md](https://github.com/Erised-AI-Inc/meep-gpu/blob/main/INSTALL.md#tested-meep-versions-and-builds).

Do not infer a speed from this page for a different problem:
[Time your own simulation](time-your-simulation.md).
