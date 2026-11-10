# Running the timing ladder

The timing ladder times the package's GPU routes and stock MEEP on the **same
simulation**, on one host, in one quiet window, and records every row with its
verdict. It is the instrument behind every GPU-against-MEEP ratio this project
states. It lives in the harness, under <code>parity/meep_gpu/</code>, and is not
installed with the package.

| File | What it does |
|---|---|
| <code>timing_ladder.py</code> | The orchestrator: plan, gates, rows, snapshot, guard, resume; <code>--dry-run</code> and <code>--selftest</code> |
| <code>timing_host.py</code> | Host readers: busy CPUs, memory, topology, launchers and binding flags, account processes, CPU and GPU state |
| <code>timing_records.py</code> | The recorders that write the ledger, and <code>meep-build-record</code> |
| <code>timing_cases.py</code> | The timing cases: one builder per case, for both benches |
| <code>bench_meep_identical_case.py</code> | Stock MEEP stepping a timing case under <code>mpirun</code> or <code>srun</code> |
| <code>bench_timing_case.py</code> | <code>bench_fused_products.py</code> on a case of <code>timing_cases.py</code> |
| <code>digest_harness_lift.py</code> | The geometry digest of the simulation the package's own lift initialised |
| <code>gpu_array_control.py</code> | The array-path control as a process of its own (optional) |
| <code>build_identical_case_table.py</code> | The table of a ladder, from its ledger |
| <code>build_3d_comparison.py</code> | A comparison across hosts, refusing pairs that are not the same simulation |
| <code>audit_timing_validity.py</code> | A read-only audit of a ladder's windows, passes and host state |
| <code>build_meep_reference_native.sh</code> | A native-tuned MEEP build, as a timing control |
| <code>timing_rows.py</code> | The row store: a row's record, its validity on a release tree, and the cuts (the identical-case table, the dispatch-preference records, the fused-product report) |
| <code>rounds/1.0/cert8.py</code> | The release round: times any subset of rows, one row at a time ([below](#the-release-round-one-row-at-a-time)) |

## What a row measures

- **A GPU row** is one run of <code>bench_fused_products.py</code> on one case and
  size: three legs (the dispatched route, the certified single kernels, the array
  path), lifted in one process, timed in AB/BA-interleaved windows with monitors
  attached. Its verdict is the harness's own (<code>TIMED</code> or not). The spread
  of each leg's windows is recorded with the row (each leg's <code>spread</code>), and
  the bench still counts it among its floors (<code>spread_within_gate</code>); the
  cutters read a pass struck on that floor alone as unstruck, and a leg enters on the
  spread of its pass rates ([The gate of record](#the-gate-of-record)). The
  ladder adds what launched, bit identity, and the deposit-repair route the case
  declares. Each window's step count is sized from a timed probe that runs after a
  restore and a few untimed steps, so the restore's own cost does not shorten the
  window; each leg runs one full-length window first that enters no statistic
  (<code>windows_discarded</code>), because a leg's first long run read 5-7 % slow.
  With <code>--gpu-legs gpu-only</code> the timed GPU rows lift the dispatched route
  alone (one lift, no singles, no array leg) and carry no bit-identity verdict of
  their own: each cites the three-leg row the <code>identity</code> tier ran at its
  route, case and size, on the same pinned sources, and is struck by name without one.
- **A MEEP row** is one fresh process of stock MEEP at N ranks, timing
  <code>fields.step()</code> over windows sized by calibration, after
  <code>init_sim</code> and a warm-up, with the flux monitor stepped. Its rate is
  cells × steps / wall seconds, the seconds being the maximum over ranks.
- **A digest row** lifts the case through the package's NumPy reference and
  computes the same geometry digest the MEEP bench writes. Its digest is the
  **digest of record** for that case and size: every MEEP row at that size must
  carry it, or it is struck. It is written only when the lifted driver agrees with
  the MEEP object on every fact checked (grid, cell count, `dt`, PML thickness,
  sources, monitors); a lift that disagrees, or that finds a digest of record
  already there that is not its own, stops the ladder.

## The subnormal policy of each row

| Row | Policy | Set by |
|---|---|---|
| NVIDIA GPU row | <code>keep</code> | <code>--subnormal-policy</code>, <code>keep</code> by default on Linux: exported as <code>MEEP_GPU_SUBNORMAL_POLICY</code>, with a CuPy cache the package mints for keep before the first row |
| Apple GPU row | flush | the package's default (<code>--subnormal-policy</code> unset on macOS): the Apple GPU flushes and has no lever ([Floating point](../design/floating-point.md)) |
| MEEP row, Linux | flush | MEEP itself: importing MEEP on x86 sets the FTZ and DAZ bits (<code>set_zero_subnormals</code>); the bench changes nothing |
| MEEP row, macOS | keep | MEEP's switch has no effect on Apple silicon |

Every NVIDIA timing of record is of <code>keep</code>, beside MEEP under its own flush on
the same host. No NVIDIA ladder has run under <code>--subnormal-policy flush</code>, so
the dispatched route's time under flush is unmeasured and no NVIDIA figure states it.
A GPU row's environment delta in the ledger names the policy it exported, and the bench's
own record of the row carries the policy in force (<code>subnormal_policy</code>).

**The NVIDIA dispatch-preference records are keep-only.**
<code>meep_gpu/triton_kernels/timing_cc86.json</code> and
<code>meep_gpu/cuda_kernels/timing_cc86.json</code> are cut from rows timed under
<code>keep</code> (each record's <code>policy</code> is <code>["keep"]</code>;
<code>cut_timing_record.py</code> refuses rows of mixed policy). A record key carries no
policy axis and <code>dispatch_preference.consult</code> takes no policy, so the consult
answers a flush run from keep rows. <code>meep_gpu/metal_kernels/timing_applegpu_g13s.json</code>
is cut from rows timed under <code>flush</code>, the only policy that GPU has. Since 1.0.1 the dispatch path consults the records
(<code>fastpath._apply_preference</code>, switch <code>MEEP_GPU_PREFER</code>; see the
kernel-dispatch guide): a fused product a record measures slower than its certified singles
gives way to them, and an architecture with no record of its own, or a product its
record does not price, is arbitrated by the portable prior of its vendor's nearest timed
architecture (the nearest compute capability, or the closest Apple chip generation), from
that architecture's records that hold on the tree: NVIDIA records for the Triton and
hand-written CUDA tables, Apple records for the Metal table, never one vendor's for the
other's. A vendor with no such record has no prior, and its tables keep the shape rule. The shipped NVIDIA records are compute capability 8.6's representative,
cut under the between-pass gate from one RTX 3090's rows (CUDA driver 13040, under
WSL2): its 74 DRIVE rows and 50 size-sweep rows, 24,000 to 5,400,000 cells. The shipped
Apple record is the <code>applegpu_g13s</code> representative, cut under the same gate from
one Apple M1 Max's rows: its 42 DRIVE rows and 30 size-sweep rows, 72 of 72 admitted, up to
5,400,000 cells. It measures 7 keys and leaves 26 unmeasured, each with one row at its size,
below the two a key needs. The records
cut from the 2026-10-16 rows of one RTX A6000 are retired (1.0.0 shipped the 2026-09-21
records, which nothing consulted). A record is consulted only while the
deposit-repair module and the fused-pair emitters it names are at the bytes its rows timed;
a change to either answers "no record" until the record is cut again.

## The order of work

Tiers run in order, outermost; inside a tier, passes, then sizes, then cases:

| Tier | Rows |
|---|---|
| <code>preflight</code> | Every row kind once at the preflight size with short windows. A GPU row that does not launch what its route expects, bit-identically, with its case's deposit-repair bracket stops the ladder. Never figures of record |
| <code>digests</code> | The harness lift's digest per planned (case, size) |
| <code>identity</code> | With <code>--gpu-legs gpu-only</code> only: one three-leg GPU row per route, case and timed size, with the preflight's short windows; a row that is not bit-identical stops the ladder. Never figures of record |
| <code>bindprobe</code> | MEEP at the probe rank counts, bound against unbound, two passes; a rule fixed in advance chooses the binding (Linux only) |
| <code>priority</code>, <code>rest</code>, <code>largest</code> | The sizes of record, every pass |
| <code>bindcheck</code> | The binding not chosen, at the priority sizes, one pass |

Per (tier, size, pass, case) the sequence is: GPU route 1, MEEP block A, GPU
route 2, MEEP block B, the array-path control, MEEP block C, then the MEEP
controls. Interleaving puts any drift on both sides alike.

## The gate

Every row that starts a process waits for a quiet host, read twice, a hold
apart:

- **CPU.** GPU rows on a Linux host: <code>load1</code> below the threshold; on
  macOS, and with <code>--gate-scope allocation</code>, the busy CPUs instead (a
  host-wide load average sees every other job on a shared node, and on a Mac it runs
  2.5–2.8 times the busy CPUs). <code>load1</code> with an allocation scope is
  refused. MEEP rows: the busy logical CPUs over a 3 s sample, below the threshold:

  busy\_cpus = Σ over CPUs *i* in scope of (1 − Δidle<sub>*i*</sub> / Δtotal<sub>*i*</sub>)

  from <code>/proc/stat</code> on Linux, where total sums user, nice, system, idle,
  iowait, irq, softirq and steal (guest time is already inside user), and idle is
  idle plus iowait. The scope is every CPU line in both samples (<code>--gate-scope
  host</code>) or this process's affinity (<code>allocation</code>, a Slurm job's
  cpuset). On macOS it is the logical CPU count times the busy fraction of the Mach
  host's CPU ticks, host scope only. Nothing reads <code>nproc</code> or the thread
  environment. The default threshold is 10 % of the logical CPUs in scope, at least
  one.
- **Memory.** Available memory above <code>--gate-mem-min-gb</code> (default 25 %
  of total). On macOS, <code>vm_stat</code>'s free, inactive, speculative and
  purgeable pages, an approximation of Linux's <code>MemAvailable</code>.
- **The account's GPU processes**, matched by uid, at most
  <code>--account-max</code>, and below <code>--cap</code> when a cap is set.
- **The device free**: no compute process on the pinned NVIDIA GPU; on macOS no
  other process driving the Apple GPU through the harness, another ladder or its
  drivers included (<code>bench_timing_case.py</code>,
  <code>gpu_array_control.py</code>, <code>timing_ladder.py</code>).
- **macOS only.** No CPU speed limit below 100, and no Spotlight, backup,
  software-update, photo or media analysis or XProtect process above 10 % CPU
  (<code>--gate-darwin-daemons 0</code> waives it). The power source gates nothing
  by default: with <code>--require-ac-power 1</code> a row also waits for AC power
  and no low power mode, and a row taken on battery or in low power mode is struck.
  Either way the power state is read and recorded before and after every row. With
  <code>--gate-apple-gpu-util-max N</code>, a GPU row also waits for the Apple GPU's
  Device Utilization (<code>ioreg</code>) to be at most N %; unset, the utilisation
  is recorded and does not gate, because no idle level has been calibrated.

Under WSL2 every reader above sees the Linux VM alone: its load average and
<code>/proc/stat</code>, and an <code>nvidia-smi</code> that lists no process. The ladder's
own gate reads nothing of Windows; a session of the timing driver
(<code>cert8.py run</code>) adds the Windows side to its quiet window and its watcher
([A desktop under WSL2](#a-desktop-under-wsl2)).

A value that cannot be read is **not quiet**. A 2026-09-28 ladder's MEEP gate
could never fail: it multiplied a busy fraction by <code>nproc</code>, which
honours <code>OMP_NUM_THREADS=1</code>. Its MEEP rows carry no evidence of a quiet
CPU.

## Parameters

Every parameter is a flag; a <code>KEY=VALUE</code> line of <code>--params
FILE</code>; an environment variable <code>MGPU_TIMING_&lt;KEY&gt;</code>; for the
interpreter and the run root, the site variables <code>MGPU_SITE_PYTHON</code>
and <code>MGPU_SITE_STAGING_ROOT</code>; or a detected or built-in default, in
that order. <code>environment.json</code> and <code>params.env</code> in the run
directory record each value and where it came from. <code>--dry-run</code> prints
them.

| Parameter | Default | Notes |
|---|---|---|
| <code>--stamp</code> | required | The run is <code>&lt;run root&gt;/ladder_&lt;stamp&gt;</code> |
| <code>--python</code> | <code>MGPU_SITE_PYTHON</code>, required | Runs the GPU rows, the digests and the reference MEEP; must import single-precision MEEP |
| <code>--gpu</code> | required with an NVIDIA route | A device index, a UUID, or <code>auto</code> when exactly one GPU is allocated. Every name (this one, an inherited <code>CUDA_VISIBLE_DEVICES</code>, <code>--refuse-gpus</code>) is mapped to its UUID through <code>nvidia-smi</code> before it is compared. A live run writes the UUID <code>auto</code> resolved to into its configuration and gate before any row runs |
| <code>--meep-config-log</code> / <code>--meep-build-record</code> | one required | The reference build's record, or <code>none</code> (the column is then labelled "build flags unrecorded") |
| <code>--meep-launcher</code> | <code>mpirun</code> beside <code>--python</code> | Or <code>srun</code>, which needs <code>--srun-mpi</code> |
| <code>--meep-build NAME=PYTHON[,LAUNCHER[,RECORD]]</code> | none | Another MEEP build, repeatable; refused if double precision. Its launcher defaults to the <code>mpirun</code> beside its interpreter, or to the reference launcher under <code>srun</code>; every build's launcher is checked before the run, and each build's MPI implementation is read on its own |
| <code>--run-root</code> | <code>$MGPU_SITE_STAGING_ROOT/timing</code> | Refused inside any tree the ladder reads |
| <code>--tree-src</code>, <code>--tree-mode</code> | this repository, <code>snapshot</code> | A snapshot is copied once into the run and checked by manifest |
| <code>--snapshot-exclude</code> | <code>__pycache__ *.pyc .git .DS_Store parity/meep_gpu/results</code> | Names (<code>fnmatch</code>) and relative paths left out of both the copy and the manifests it is checked by |
| <code>--drivers-mode</code> | <code>tree</code> for this repository, else <code>copy</code> | <code>copy</code> puts these drivers into <code>&lt;run&gt;/drivers</code> with their sha256 |
| <code>--routes</code> | <code>default,cuda</code> on Linux, <code>metal</code> on macOS | <code>default</code> asks the Triton table; <code>cuda</code> exports <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>; <code>none</code> times MEEP only: its rows, the preflight digest and the harness-lift digests |
| <code>--expect-launched ROUTE:CASE=TABLES</code> | none | Absent, the preflight row pins what every later row of that route and case must launch |
| <code>--allow-uncertified</code> | 0 | 1 exports <code>MEEP_GPU_HARDWARE=any</code> to the GPU rows; such rows are uncertified and say so. 0 exports <code>MEEP_GPU_HARDWARE=certified</code> to the NVIDIA rows, so a supported device that is not certified is refused rather than timed uncertified, and unsets it for every other row |
| <code>--cases</code> | <code>pml_3d</code> | Cases of <code>timing_cases.py</code> |
| <code>--priority-sizes</code>, <code>--rest-sizes</code>, <code>--largest-sizes</code>, <code>--probe-size</code>, <code>--preflight-size</code> | the 2026-09-28 cell counts | Cells; or <code>--priority-res</code> etc. as resolutions of the first case. A size a case cannot hit exactly is refused |
| <code>--ranks "A\|B\|C"</code> | from the physical cores P | A = P/3, P/2, 2P/3, P; B = 8 (when below P/3) and 1; C = 4P/3 on an SMT host, or every logical CPU when the host has more of them than P without SMT. P = 48 with 96 logical gives 16, 24, 32, 48 \| 8, 1 \| 64; an M1 Max (P = 8 performance cores, 10 logical) gives 3, 4, 5, 8 \| 1 \| 10. <code>none</code> times the GPU routes only, with the array control read off their rows: no MEEP row, no MEEP control, no preflight MEEP row or digest, no harness-lift digest, no binding probe; refused with <code>--routes none</code> |
| <code>--binding</code> | <code>probe</code> | Or <code>bound</code>, <code>unbound</code>. macOS: <code>unbound</code> only |
| <code>--physical-cores</code>, <code>--logical-cpus</code> | detected | Linux: unique (package, core) over the online CPUs in this process's affinity; macOS: <code>hw.perflevel0.physicalcpu</code> and <code>hw.logicalcpu</code> |
| <code>--sockets</code>, <code>--threads-per-core</code> | detected | <code>srun</code>'s bound modes balance ranks across the sockets and are refused when the count is unknown |
| <code>--controls</code> | <code>native,threads2,threads4,hwthread,split</code> | <code>none</code> for a 2026-09-28 reproduction (below) |
| <code>--meep-rank-ceiling N</code> | none | The most MEEP ranks any row may take ([The MEEP controls](#the-meep-controls)). No parameter of a site file: <code>cert8.py</code> hands it to a split session's MEEP half |
| <code>--gate-cpu-max</code>, <code>--gate-mem-min-gb</code> | 10 % of logical CPUs on Linux, <code>auto</code> on macOS; 25 % of memory | <code>auto</code> takes 18 busy readings 10 s apart before the first row: the 10 % default stands when their 80th percentile is below it, otherwise the gate opens below 25 % of the logical CPUs (2.5 of 10 on an M1 Max, whose owner's session idles at 1.2-2.0). Written to <code>gate_calibration.json</code> and read back on <code>--resume</code> |
| <code>--gate-gpu-metric</code> | <code>load1</code> on Linux; <code>busy</code> on macOS and with <code>--gate-scope allocation</code> | |
| <code>--gate-darwin-daemons</code>, <code>--gate-apple-gpu-util-max</code> | 1, unset | macOS gate conditions, above |
| <code>--gpu-flags</code> | the 2026-09-28 flags; on macOS with <code>--memory-budget-bytes</code> 0.6 × the host's memory | Must include <code>--monitors attached</code>, or the plan is refused: every MEEP row accumulates its flux monitor |
| <code>--gate-hold-gpu-s</code>, <code>--gate-hold-meep-s</code> | 120, 30 | |
| <code>--account</code>, <code>--account-max</code>, <code>--cap</code> | this user, 0, off | |
| <code>--passes</code>, <code>--tiers</code>, <code>--array-control</code> | 3, every tier, <code>extract</code> | <code>--array-control row</code> drives the NVIDIA tables only and is refused with the metal route; <code>none</code> plans no array control |
| <code>--gpu-legs</code> | <code>all</code> | <code>gpu-only</code>: the timed GPU rows lift the dispatched route alone (<code>bench_fused_products.py --legs gpu-only</code>), behind the <code>identity</code> tier; refused with <code>--array-control extract</code> (no array leg to read) or without the identity tier |
| <code>--allow-nice</code> | 0 | A ladder running at a nice level other than 0 is refused |
| <code>--require-ac-power</code> | 0 | macOS: 1 refuses to start on battery or in low power mode, waits those out between rows and strikes a row taken in either. Off by default: no run is gated on the power source |

Binding flags are chosen by the MPI implementation of each build's launcher, read
from the files beside it (<code>ompi_info</code>, <code>mpichversion</code>, ...) and,
in a live run, checked against the launcher's own <code>--version</code>: a
disagreement is refused by name, and a launcher the files could not name takes the
implementation its <code>--version</code> reports. With N ranks, S sockets and H
hardware threads per core:

| Mode | Open MPI | <code>srun</code> |
|---|---|---|
| bound (<code>bound-core-by-package</code>) | <code>--map-by package --bind-to core</code> | <code>--cpu-bind=cores --ntasks-per-core=1 --ntasks-per-socket=⌈N/S⌉ --distribution=block:block</code> |
| unbound | <code>--bind-to none</code> | <code>--cpu-bind=none</code> |
| above the physical cores (<code>unbound-hwthreads</code>) | <code>--use-hwthread-cpus --bind-to none</code> | <code>--cpu-bind=none --ntasks-per-core=H</code> |
| <code>bound-hwthread</code> | <code>--use-hwthread-cpus --map-by pe-list=L:ORDERED --bind-to hwthread</code>, L every core's first hardware thread (packages alternating) before any second, as hwloc logical indices | <code>--cpu-bind=threads --ntasks-per-core=H --ntasks-per-socket=⌈N/S⌉ --distribution=block:block:cyclic</code> |
| T threads per rank | <code>--map-by package:PE=T --bind-to core</code> | <code>--cpus-per-task=T·H --cpu-bind=cores --ntasks-per-core=1 --ntasks-per-socket=⌈N/S⌉ --distribution=block:block</code> |

Open MPI rows add <code>--report-bindings</code>; <code>srun</code> rows add
<code>verbose</code> to <code>--cpu-bind</code>. Any other implementation must be
given its flags (<code>--flags-bound</code> and the rest). The <code>srun</code>
templates follow the Slurm <code>srun</code> manual: <code>--cpu-bind</code> on the
command line overrides <code>--hint</code>, so none uses it;
<code>block:block</code> alone fills one socket before the next, so the bound modes
cap the ranks per socket, which puts them contiguously and evenly on the packages,
as the Open MPI rows were placed. **They have not run on a cluster yet**:
<code>--selftest --launcher-probe</code> checks every mode the plan uses before
anything is timed (below).

## The MEEP controls

A ratio against MEEP is only as strong as the MEEP it divides by. Per size and pass,
after block C, the ladder times:

| Control | Rows | What it rules out |
|---|---|---|
| <code>native</code> | the <code>--meep-build native=...</code> build at 2P/3 and P ranks | a portable <code>-O2</code> reference compiled below what the CPU runs |
| <code>threads2</code>, <code>threads4</code> | P/2 ranks × 2 and P/4 ranks × 4 OpenMP threads | that hybrid MPI × OpenMP beats pure MPI |
| <code>hwthread</code> | 4P/3 and 2P ranks bound to hardware threads (SMT hosts) | that the ranks above P lost only for want of binding |
| <code>split</code> | 2P/3 and P ranks with <code>split_chunks_evenly=False</code> | that MEEP's cost-based chunk splitting balances PML better |

Each control writes to its own directory (<code>meep_native</code>,
<code>meep_t2</code>, ...) and every row records its configuration (build, threads
per rank, binding, chunk splitting, stepper). A control the host cannot run is
listed as omitted, with the reason, by the dry run.

**The host's MEEP rank ceiling.** <code>--meep-rank-ceiling N</code> is the most MEEP
ranks any row of the ladder may take, whatever <code>--controls</code> asks, a catalog's
flag included. A rank block, a binding-probe rank or the preflight MEEP row (which runs at
the physical cores) above it is refused by name; a control none of whose rows fits is
omitted by name (its rank counts and the ceiling under <code>controls_omitted</code>); a
control with a row inside the ceiling and one above it keeps the one inside, and the one
above is named under <code>rows_above_rank_ceiling</code>. It is no parameter of a site
file: <code>cert8.py</code> hands it to the MEEP half of a split session
([The split ladder](#the-split-ladder)), from the rank blocks that session's site file
declares. On an 8-performance-core desktop (<code>PHYSICAL_CORES=8</code>,
<code>RANKS=8|1|</code>, ceiling 8) the catalog's <code>hwthread</code> rows (11 and 16
ranks) are refused by name and every other control runs within 8 ranks:
<code>threads2</code> and <code>threads4</code> at 4 × 2 and 2 × 4, <code>split</code> at 5
and 8. A declared rank block is no ceiling of its own (a GPU host's site file names the
reference blocks it times, 32 and 48, and its <code>hwthread</code> control still runs at
64), and a ladder handed no ceiling plans as it did.

**Threads.** Whether a MEEP build can run threads is read from its build record:
<code>openmp</code> is true only when one of the MEEP libraries the interpreter
actually loads links an OpenMP runtime directly (<code>otool -L</code> on macOS,
the ELF dynamic section on Linux). A runtime another library maps into the
process, such as an OpenMP build of OpenBLAS, does not count; on the reference
Mac builds it is the only runtime there. The ladder probes every build's record
itself at start, writes it to <code>meep_builds/&lt;build&gt;.json</code>, passes
it to every threaded row, and refuses to start when a threaded control is planned
for a build that links no OpenMP. The bench also requires an OpenMP runtime mapped
into the row's process, read once MEEP is imported (before the import MEEP's own
library is not loaded yet, which refused every hybrid row of the 1.0 release's timing). A
threaded row is struck when, on the slowest
rank, the process gained fewer than T − 1 threads between MPI initialisation and
the end of its windows (MPI's helper threads are in the baseline), or when its CPU
seconds per wall second fall below 0.75 T.

**Hardware threads.** A bound row whose ranks used fewer distinct physical cores
than min(ranks, P) is struck. Every Open MPI 5 mapping policy fills a mapped
object's hardware threads in order, so <code>--map-by package --bind-to
hwthread</code> put 64 ranks on 32 of the GPU host's 48 cores; the template names
each rank's hardware thread instead (the table above), which puts 64 ranks on all
48 cores, 32 per package, and needs the socket count: without it the control is
omitted by name. Give other flags with <code>--flags-bound-hwthread</code>.

**The native build.** <code>bash parity/meep_gpu/build_meep_reference_native.sh</code>
builds the reference's MEEP release (the platform script's <code>MEEP_VERSION</code>)
through the platform's reference script with
<code>-O3 -march=native</code> (Linux x86_64) or <code>-O3 -mcpu=native</code>
(macOS arm64) and writes <code>&lt;prefix&gt;/share/meep-gpu-build/build_record.json</code>:
the configure line, the Makefile flags, the compiler, what the native target
resolved to, the libraries' sha256 and the OpenMP runtime they link. It is a
timing control, never a certification reference, so no reference environment carries
one. Without <code>--meep-build native=...</code> the control is omitted by name ("no
--meep-build native=PYTHON[,LAUNCHER[,RECORD]] was given"), as it was in every ladder of
the 1.0 release's timing on both hosts: the MEEP column of such a ladder is the portable reference
build alone, and a table cut from it says so.
<code>timing_records.py meep-build-record --python &lt;interpreter&gt;</code> writes
the same record for a build that already exists.

## The cases

<code>timing_cases.py</code> defines each case once. A name the route gate already
has (<code>pml_3d</code>, <code>pml_3d_diagonal</code>) resolves to the route
gate's own builder, and its GPU row is <code>bench_fused_products.py</code>
unchanged. The cases defined there are injected into the bench's tables at run
time by <code>bench_timing_case.py</code>, which puts them back when the bench
returns:

| Case | Interior | Source | GPU spec copied from | Deposit repair |
|---|---|---|---|---|
| <code>pml_3d</code> | 4 µm cell, 0.8 µm PML (78.4 % PML), ε-9 sphere | Ez | (the route gate's own row) | none |
| <code>thin_pml_3d</code> | 8 µm cell, 0.5 µm PML (33.0 % PML): substrate ε 2.1 through the PML, ε-9 sphere, ε-6 cylinder, a rotated ε-4 block | Ez | <code>pml_3d</code> | none |
| <code>thin_pml_3d_hz</code> | as <code>thin_pml_3d</code> | Hz | <code>pml_3d</code> | B seam |
| <code>thin_pml_3d_diagonal</code> | axis-aligned blocks only (diagonal ε) | Ez | <code>pml_3d_diagonal</code> | D seam |
| <code>pml_3d_lorentz</code> | 4 µm cell, 0.8 µm PML (78.4 % PML): <code>pml_3d_diagonal</code>'s 1 µm slab as a Lorentz medium (ε∞ 2.25, one pole at 0.4, γ 0.02, σ 1.1) through the PML (diagonal ε) | Ez | <code>pml_3d</code> | none |
| <code>pml_3d_monitors</code> | as <code>pml_3d</code>, plus six flux planes closing a 2 µm box around the sphere at 20 frequencies each and one DFT field plane (Ex, Ey, Ez at z = 0, 2.4 × 2.4 µm) at 5 frequencies: 8 DFT objects against <code>pml_3d</code>'s 1 | Ez | <code>pml_3d</code> | none |
| <code>pml_3d_hz</code> | as <code>pml_3d</code> | Hz | <code>pml_3d</code> | B seam |

Because 8 r′ = 4 r, every <code>pml_3d</code> resolution maps to half the
resolution on the thin cases, so sizes are named by cell count across cases.
Every GPU row is checked against the bracket its case declares. To add a case,
add its builder and a <code>TimingCase</code> to <code>NEW_CASES</code>; if a
preflight measures that a table's gate reads a field of the spec that differs from
the template's, add it to <code>TABLE_OVERRIDES</code>, naming the row that
measured it.

## Running it

**Always dry-run first.** <code>--dry-run</code> starts no process, reads no
device and writes nothing; it prints every parameter with its source, the budget
per size and tier, and every row with its command, working directory and
environment. <code>--dry-run --json</code> prints the same plan as JSON.

**Then the self-test** on the host that will run it: <code>--selftest</code> reads
every host reader once, prints the binding flags it would use, the GPU route the
package resolves under the GPU rows' own environment, every refusal the live run
would raise, and both gates' verdicts now. Add <code>--launcher-probe</code> on any
host that binds ranks, and always inside a Slurm allocation: it starts the
launcher in every binding mode the plan uses, at that mode's largest rank count,
reads each rank's CPU set and checks it against the topology — bound: one physical
core per rank, none shared, ranks balanced across packages; hardware threads: at
least min(ranks, P) distinct cores; T threads: T whole cores per rank; unbound and
above the cores: every rank holds the whole allocation. Each failure is printed by
name and the self-test exits non-zero.

**Before the run starts**, the ladder also refuses, by name: an account that names
no user (the account gate and the cap guard would count nothing); a build whose
launcher or interpreter does not exist; a GPU the inherited allocation does not
hold; a package that resolves another GPU route than the plan needs under the GPU
rows' environment; and, inside a Slurm job, a shell that holds the allocation but
is not on its node (an <code>salloc</code> shell on the login node: no
<code>SLURMD_NODENAME</code>), a ladder started under <code>srun</code> (a job
step), CPUs in scope under 90 % of
<code>SLURM_CPUS_ON_NODE</code>, or fewer task slots on the node
(<code>SLURM_TASKS_PER_NODE</code>) than the largest rank count under
<code>mpirun</code>. The first start writes <code>host_identity.json</code> (host
name and GPU UUID); <code>--resume</code> on another host or device is refused, and
every ledger line carries <code>host</code>.

**Every row's environment.** The rows never inherit a package switch: the named
ones (<code>MEEP_GPU_DISPATCH</code>, <code>MEEP_GPU_FUSED</code>, ...) and every
other <code>MEEP_GPU_*</code> variable of the launching shell that the ladder does
not set itself (<code>MEEP_GPU_METAL_RESIDENCY</code>,
<code>MEEP_GPU_KERNEL_TABLE</code>, <code>MEEP_GPU_WARM</code>, the probe switches)
are unset, and Metal rows unset <code>PYTORCH_ENABLE_MPS_FALLBACK</code> and
<code>PYTORCH_MPS_*</code>. Each row's environment delta is in the ledger. Everything
else a row holds comes from the launching shell, the user site directory included:
launch the ladder with <code>PYTHONNOUSERSITE=1</code> exported, as below, so no
distribution under <code>~/.local</code> sits ahead of the reference environment's own in
a GPU row or a MEEP rank (<code>environments/locks/README.md</code>). The 1.0 certification round's child
environments, the timing legs among them, are tabulated in
<code>parity/meep_gpu/rounds/1.0/RUNBOOK.md</code>, section 3.

**Linux, directly.** Launch it un-niced and detached, with its output to a log,
from bash (zsh starts a background job at nice 5, which the ladder refuses, unless
<code>setopt no_bg_nice</code>). Export the environment's own C compiler as
<code>CC</code>: Triton compiles the launcher of every timed kernel with
<code>$CC</code>, else the host's <code>gcc</code>, which is outside the environment
(<code>cert8.py run</code> exports it for every unit itself). The libcuda directory is the
link the environment's build made:

```bash
PYTHONNOUSERSITE=1 CC=<env>/bin/x86_64-conda-linux-gnu-cc \
  setsid nohup python -u parity/meep_gpu/timing_ladder.py --stamp <stamp> \
  --python <env>/bin/python --gpu 0 --meep-config-log <meep source>/config.log \
  --libcuda-stub <env>/lib/meep-gpu-libcuda > ladder_<stamp>.launch.log 2>&1 < /dev/null &
tail -f ~/meep_gpu_validation/timing/ladder_<stamp>/progress.log
```

**Slurm.** Take the node whole (<code>--exclusive --ntasks-per-node=&lt;logical
CPUs&gt;</code>, one GPU), and run the ladder from the batch script or a shell on
the allocated node (an <code>salloc</code> shell is on the login node unless the
site starts an interactive step), never under <code>srun</code>. Then either let it start
MEEP with <code>mpirun</code> inside the allocation, or pass <code>--meep-launcher
srun --srun-mpi pmix</code> (or <code>pmi2</code>); <code>--gpu auto</code> takes
the allocation's one GPU and pins it by UUID (the box string still keys it by its
index, <code>gpu&lt;N&gt;=</code>). Run <code>--selftest --launcher-probe</code> first in the
same allocation. <code>--gate-scope allocation</code> reads the busy CPUs of the
allocation's cpuset only, and gates the GPU rows on them too. A MEEP row inside a
cpuset narrower than the host is told the ladder's CPU count, so an unbound rank
holding the whole allocation is not recorded as bound.

**macOS (Apple silicon).** Ranks are not bound (Open MPI cannot bind there), the
rank basis is the performance cores, and rows above them are labelled
<code>unbound-ecores</code>. The ladder refuses to start at a nice level other
than 0 or at a background darwin priority (which would confine every rank to the
efficiency cores), and its gate waits out busy background daemons between rows; on
battery or in low power mode it refuses and waits only under
<code>--require-ac-power 1</code>. It holds the Mac awake with
<code>caffeinate -i -s -w &lt;its pid&gt;</code> for its own life and records it.
From zsh:

```zsh
setopt no_bg_nice
PYTHONNOUSERSITE=1 nohup python -u parity/meep_gpu/timing_ladder.py --stamp <stamp> \
  --python <env>/bin/python --meep-config-log <prefix>/share/meep-gpu-build/config.log \
  > ladder_<stamp>.launch.log 2>&1 < /dev/null &
```

The default CPU threshold is 10 % of the logical CPUs, which on a 10-CPU Mac is 1:
both kinds of row wait for less than one busy CPU (the Mach tick reading). Read
<code>--selftest</code> first; a threshold set higher with
<code>--gate-cpu-max</code> is recorded in <code>environment.json</code> and belongs
in the table's conditions. The GPU rows' memory budget is 0.6 of the host's memory
(the 30 GB of the NVIDIA instrument would be 94 % of a 32 GB Mac), and the bench
refuses a case whose frozen legs exceed it. The time budgets the dry run prints
are derived from the NVIDIA host's measured row times and are estimates only on a
Mac.

**<code>KMP_DUPLICATE_LIB_OK</code>.** On macOS the GPU rows set it to
<code>TRUE</code>, in their recorded environment delta, and the MEEP rows unset
it. Every Metal row imports PyTorch, whose wheel bundles its own
<code>libomp</code> beside the one the environment's OpenBLAS links: a process that
imports <code>torch</code> before NumPy aborts with "OMP: Error #15" when the
variable is unset (measured on the reference Mac environment; NumPy first does
not abort). The reference Metal runs set it the same way. A MEEP process loads one
runtime at most and needs nothing. On Linux the ladder neither sets nor unsets it:
a value inherited from the launching shell is recorded in
<code>environment.json</code> and in every MEEP row.

## What it writes

Everything goes under <code>&lt;run root&gt;/ladder_&lt;stamp&gt;/</code>, outside
every tree it reads and outside every root <code>cut_timing_record.py</code> reads:

| Path | What |
|---|---|
| <code>progress.log</code> | One flushed line per row start and end |
| <code>ladder_rows.jsonl</code> | The ledger: one JSON line per row, appended as it lands |
| <code>environment.json</code>, <code>params.env</code>, <code>plan.json</code> | The host, every build, every resolved parameter, the plan |
| <code>box_trace.log</code>, <code>capguard.log</code> | Host state every <code>--monitor-every-s</code>; the cap guard's readings |
| <code>cpu_trace/</code>, <code>gpu_trace/</code> | CPU frequency, governor and turbo state during each row (every second on a GPU row, at the box monitor's interval on a MEEP row, whose unbound ranks a per-second reader would preempt); NVIDIA: the pinned GPU's clocks, temperature, power and throttle reasons every second, and before and after each row in the ledger; macOS: the Apple GPU's utilisation and memory from <code>ioreg</code> every 15 s (its clocks, temperature and power need root and are not recorded). That reader starts one short <code>ioreg</code> process every 15 s inside a Metal row's timed windows, which the Mac drivers of record did not |
| <code>meep_builds/</code> | The build record the ladder probed from each MEEP build's interpreter |
| <code>host_identity.json</code> | The host and GPU UUID of the first start; a resume elsewhere is refused |
| <code>gpu_&lt;route&gt;/pass&lt;p&gt;/&lt;case&gt;_res&lt;r&gt;/</code> | GPU rows |
| <code>meep/</code>, <code>meep_&lt;control&gt;/</code> | MEEP rows, one directory per configuration |
| <code>digests/</code> | The digest of record per (case, resolution), with its source: <code>harness-lift:…</code> when the <code>digests</code> tier wrote it, <code>first-meep-row:…</code> when a MEEP row at that size ran first (the preflight size, for one). A lift that finds a file already there compares against it and leaves its source as it was; the table then reads "equal" for that size rather than "harness-lift digest of record" |
| <code>digests_lift/</code> | The harness-lift digest files, for <code>build_3d_comparison.py --digests</code> |
| <code>done/</code> | One marker per finished row; <code>--resume</code> skips them |

## Building the tables

The tables of record of a release are cut from its row store
([The release round, one row at a time](#the-release-round-one-row-at-a-time));
the builders below read one ladder's ledger by their own rules, as the 1.0
release's timing tables were built.

```bash
python parity/meep_gpu/build_identical_case_table.py --pull <ladder dir> \
  --out-md TABLE.md --out-json table.json [--manifest <sha256 listing>]
python parity/meep_gpu/build_3d_comparison.py --host <label> \
  --routes triton=<ladder>/gpu_default/pass1 --meep <ladder>/meep \
  --digests <ladder>/digests_lift --out comparison.md
```

The table builder reads every fact from the ledger and gates every leg; a ladder
written before the ledger carried <code>run_dir</code> is read with
<code>--remote-root &lt;its run directory on the host&gt;</code>, and harness-lift
digests taken outside the ladder are given with <code>--lift-digests</code>. A
size without a harness-lift digest is marked "harness-lift digest pending; ratios
provisional": its ratios are formed, and say so. Where a lift digest exists every
MEEP leg is compared against it. Every ratio at a size is struck, under both gate
rules and by name, when the lift's digest differs from the digest of record, when
the size's GPU, MEEP and lift rows ran different code digests or builder sources,
or when the GPU rows' monitor count differs from the MEEP rows' DFT objects. A
Metal row's neighbours are read from the bench's own <code>host_*</code> record,
and a row with no device reading is counted as not read, never as clean.

The comparison builder forms a ratio only when the two sides are one simulation and
refuses every other pair by name: a case label or geometry digest that differs, a
MEEP row with no digest, digests that disagree across passes, mixed MEEP
configurations, a lift whose driver disagreed with the MEEP object
(<code>lift-driver-disagrees</code>), a GPU row timed without its monitors or with
another monitor count (<code>gpu-monitors-detached</code>), a case of
<code>timing_cases.py</code> built by another builder source on one side
(<code>builder-mismatch</code>; a built-in case's GPU rows carry no builder hash and
are tied to the MEEP rows by the ladder's code digest instead), and a MEEP best rank
count that fails the 5 % gate (<code>meep-best-fails-gate</code>). MEEP rows the
bench struck, or whose monitor was not stepped, enter no median.

These two builders keep their own gate, on every leg and MEEP's included: each pass's
within-window spread and the spread of the pass medians, both within 5 %. It is not the
rule the tables of record are cut by ([The gate of record](#the-gate-of-record)).

**A split ladder unit** ([The split ladder](#the-split-ladder)) is two ladders, its GPU half
and its MEEP half, timed under two VM affinity masks that only the session's row records
hold. Both builders read it from its two halves and the store's row record of the unit
(any of its keys):

```bash
python parity/meep_gpu/build_identical_case_table.py --pull <the GPU half's ladder> \
  --meep-pull <the MEEP half's ladder> --split-row <store>/ladder/<key>/<record>.json \
  [--manifest <the GPU half's listing> --meep-manifest <the MEEP half's listing>] \
  --out-md TABLE.md --out-json table.json
python parity/meep_gpu/build_3d_comparison.py --host <label> \
  --routes triton=<GPU half>/gpu_default/pass1 --meep <MEEP half>/meep \
  --digests <MEEP half>/digests_lift --split-row <record> --out comparison.md
```

The record is refused by name whenever a cut of the row store would refuse it
(<code>timing_rows.refusals</code>): unless both halves ran, exited 0 and held the session's
masks (<code>timing_rows.split_refusals</code>), and unless it opened on a quiet window,
was watched without an intrusion and holds no ledger entry of another session. So is a
directory that is not under the half's ladder the record names, and a half whose ledger
holds the other half's rows. Each half is checked against its own sha256 listing
(<code>--manifest</code>, <code>--meep-manifest</code>), and the table names each half's
count. The ratio is the one a whole ladder would give, the GPU half's
legs over the MEEP half's, and every ratio is labelled <code>(split)</code>, with the two
masks, the order and the record beside the table. The comparison builder refuses by name a
route of a ladder that planned no MEEP row over the MEEP rows of one that planned no GPU row
without <code>--split-row</code>: their ratio would read as one ladder's.

**What a table names.** A table leaves the machine: the identical-case builder names the
pull, the manifest and a harness-lift file by their own names, never by an absolute path,
and writes every string it copies from its inputs (the ladder's environment record, a
struck row's reason) as a record writes it: a path under its root's label, the account as
<code>&lt;account&gt;</code>. A pull whose ladder wrote its records before they were written
so names the timing host's paths, and the table does not pass them on: a home of two levels
(<code>/home/&lt;group&gt;/&lt;account&gt;</code>) cannot be told from a directory by its
shape on another machine, so the account the pull's own ladder records (its
<code>ACCOUNT</code> parameter) is written as <code>&lt;account&gt;</code> wherever it stands
as a word.

## The release round, one row at a time

Timing follows the certification's modular pattern. The unit is a **row**; any subset of
rows can be timed, or timed again, at any time; each row's record is self-contained; and
every table, ladder and dispatch-preference record is **cut by tool from the newest valid
row of each key**, never assembled by hand. <code>parity/meep_gpu/rounds/1.0/cert8.py</code>
runs the rows and <code>parity/meep_gpu/timing_rows.py</code> keeps, validates and cuts
them. Neither times anything itself: every unit is a fresh process of the ladder or the
fused-product bench described above.

### Rows and keys

| Kind | Key | What one record holds |
|---|---|---|
| <code>ladder</code> | <code>ladder\|&lt;architecture&gt;\|&lt;case&gt;\|&lt;cells&gt;\|&lt;tier&gt;</code> | One case and size of the identical-case ladder with every leg it is compared against: each GPU route, the array-path control and every MEEP leg, all from one ladder session, MEEP re-timed beside the GPU legs rather than reused from another session |
| <code>fused</code> | <code>fused\|&lt;architecture&gt;\|&lt;table&gt;\|&lt;case&gt;\|&lt;drive or res&lt;r&gt;&gt;\|p&lt;pass&gt;</code> | One <code>bench_fused_products.py</code> unit, its row's line kept verbatim for the cutter. The pass is in the key because a band edge of a preference record needs two rows |
| <code>check</code> | <code>check\|&lt;key&gt;</code> | A dispatch check of another key: its GPU legs at its case and size on short windows, kept for the launch counters |

The architecture is an NVIDIA compute capability (<code>8.6</code>) or an Apple GPU family
(<code>applegpu_g13s</code>), as the rows stamp it. A record carries its session (id,
start, end), the unit's command, the in-harness import record of every process (below),
the launch counters of every GPU leg (per slot: arm, kernel table, dispatches, and the
family and gate the plan cites), the reference environment's build record and the
MEEP build records, the host gates it passed (the quiet window read twice, the variance
precondition, the watcher's readings and intrusions, power recorded and never gated), the
command, environment and exit status this driver gave the unit, its legs' medians with
their gate status, so a published median always carries whether the gate admitted it,
and a digest of its own body. The store keeps one file per key
and attempt (the session's id and end), written once. It lives under
<code>parity/meep_gpu/results/&lt;name&gt;/</code>, which the ladder's snapshot leaves out,
so a record cut from it names its rows <code>results/...</code> as the shipped records do;
the units' run root (<code>--run-root</code>, by default
<code>$MGPU_SITE_STAGING_ROOT/cert8</code>) stays outside every tree the ladder reads.

**A record names no machine.** A record leaves the machine that timed it, so every path it
holds is written under its root's label (<code>&lt;home&gt;</code>,
<code>&lt;staging-root&gt;</code>, <code>&lt;tmp&gt;</code>; the account as
<code>&lt;account&gt;</code>): the unit's command and environment, the gates, and the
reference environment's build record, whose facts (flags, versions, OpenMP, the FMA census,
its images' digests) are kept with every path in them so spelled. A path of the build record
under none of those roots (an HPC site's scratch or project filesystem, another drive) is
written as <code>&lt;path&gt;/</code> and its last component; a build record whose images
would then share one key, losing a digest, is refused by name before the session starts.
Its <code>sha256</code> is over the file as written, so it compares the same. A system
directory (<code>/usr</code>, <code>/proc</code>, ...) names no machine and stays.
<code>python tools/ci/check_hygiene.py --legs records --records &lt;store&gt;</code> reads
every row record of a store before it is sent, by the hygiene guard's rules and by the
check the records' writers are held to (<code>timing_rows.local_paths</code>: an absolute
path outside the system directories, a Windows profile, a path under <code>~/</code>, a
card's UUID or this machine's host name, a hit of class <code>local_path</code>), so a
command or an environment run from a directory under no labelled root is found there; the
bench line a fused row keeps verbatim beside its record is the bench's own output and is
not read. Records written before this rule hold the build record as the file spelled it,
home paths included; they are kept as they are.

**The import record is the benches' own.** <code>bench_fused_products.py</code> stamps
every row with <code>imported_source_sha256</code> (the sha256 of every repository module
its process holds, keyed as every gate artifact keys it) and
<code>bench_meep_identical_case.py</code> stamps every MEEP measurement row the same way,
which shows the package was not in that process. Beside it a fused-product row records
what the hook of the 1.0 release's timing and its joiner kept with the record: the sha256 of the ledgers
its kernel tables read to admit arms (<code>data_sha256</code>), the versions of the
libraries it loaded (<code>imported_libraries</code>), and the package, device and
compiler switches of its environment (<code>process_environment</code>; a path is
recorded by its last component). No exit hook is put on any path. The ladder's array
control is read off the GPU rows' own array legs, whose processes these records cover;
a standalone control (<code>--array-control row</code>) is a process of its own whose
driver, <code>gpu_array_control.py</code>, writes no import record, so a row timed with
one is refused by name.

### Running rows

The catalog names an architecture's rows: <code>rounds/1.0/cert8_catalog_nvidia.json</code>
and <code>cert8_catalog_apple.json</code> are the 1.0 release's timing catalogs. Each also
names the fused-product rows its dispatch-preference records are cut from: the NVIDIA
catalog the Triton and CUDA tables', the Apple catalog the Metal table's (every DRIVE case
the release dispatches on it, and a size sweep of <code>pml_2d</code>,
<code>magnetic_seam_2d</code> and <code>offdiag_2d</code> at five sizes, two passes). A
fused unit of either runs with <code>MEEP_GPU_HARDWARE=certified</code>, so an identity
the bound ledger does not certify is refused by name and timed by no fused row; the
ladder's Metal rows keep the Metal default. Site values (the MEEP build
record, a library stub, the corpus, the account) stay out of the tree, in a
<code>timing_ladder.py --params</code> file given with <code>--ladder-params</code> and in
<code>--bench-env KEY=VALUE</code>.

```bash
python parity/meep_gpu/rounds/1.0/cert8.py plan --catalog <catalog> \
  --store parity/meep_gpu/results/timing_rows --run-root <runs> \
  --rows 'ladder|8.6|pml_3d|2097152|priority'          # or --owed, or nothing: every row
python -u parity/meep_gpu/rounds/1.0/cert8.py run --catalog <catalog> \
  --store parity/meep_gpu/results/timing_rows --run-root <runs> --session <id> \
  --python <env>/bin/python --gpu 0 \
  --environment-record <the reference environment's build record> \
  --ladder-params <site params> --rows ...
```

On Linux, <code>run</code> first exports the Triton launcher's toolchain to every unit:
<code>CC</code> and <code>TRITON_LIBCUDA_PATH</code> as the launching shell set them, else
the compiler (<code>bin/x86_64-conda-linux-gnu-cc</code>) and the libcuda link
(<code>lib/meep-gpu-libcuda</code>) of the reference environment <code>--python</code> runs
in; a reference environment without either is refused by name before the session starts.
A site's <code>LIBCUDA_STUB</code> and <code>--bench-env</code> still override the
directory in their units.

A session runs, in order, writing one flushed line per step to
<code>&lt;runs&gt;/&lt;session&gt;/progress.log</code> and one marker per finished unit.
The same command resumes it at its first unfinished unit; a unit is never resumed inside
(an earlier output of it is set aside and the unit runs whole), so every leg of a row is
timed in one process under one watcher, in one session.

1. **The quiet window**, before every unit, read twice two minutes apart and recorded in
   the unit's rows: no foreign process on the device, no other GPU process of the
   account, no other work of the account (a harness, test, round or MPI process outside
   the session, a busy Python process, a busy build, compiler or bulk tool, and on Linux
   any busy process of the account; a process's share is the CPU time it used between
   two readings, not the lifetime average <code>ps</code> prints on Linux), no
   neighbour busy; on Linux the load average
   below 10 % of the logical CPUs and available memory above 25 % of the total, the
   ladder's own GPU-row thresholds; on macOS available memory and the background
   daemons, the ladder gating each of its rows on its calibrated CPU threshold. An idle
   resident process of another account on another GPU is recorded and not waited on; one
   that becomes busy during a run invalidates that run. On a WSL2 host the Windows side
   joins both readings ([A desktop under WSL2](#a-desktop-under-wsl2)).
2. **The variance precondition**, once per session that times a ladder row: the catalog's
   point repeated five times on one GPU route and at one MEEP rank count. GO when the GPU
   route's repeat spread is within 5 %. MEEP's repeat spread is recorded and never gated.
   A session run with <code>--no-variance</code> records why, and none of its ratios
   enters a table. A session of fused rows alone runs it only when asked, with
   <code>run --variance</code>, and then refuses on NO-GO before its first unit (exit 5),
   as the timing of a dispatch-preference record runs it. Under a declared VM affinity mask, with <code>--split-ladder</code> or
   without, it times the GPU route alone, as a GPU half under the mask
   ([A desktop under WSL2](#a-desktop-under-wsl2)).
3. **The units**: the ladder rows of one case and tier share one ladder (its passes
   interleave its sizes; a row alone is one size), then the fused DRIVE units, the sweep
   (odd passes ascending, even passes descending), then the dispatch checks. Every ladder
   unit is given <code>--require-ac-power 0</code>: no stage is gated on power. With
   <code>--split-ladder</code> each ladder unit runs as a GPU half and a MEEP half
   ([The split ladder](#the-split-ladder)).
4. **The watcher**, every 15 s while a unit runs: two readings in a row that see a
   foreign process on the device, another GPU process of the account, other work of the
   account, or a neighbour turned busy stop the unit, and so does one reading on which a
   WSL2 host's Windows side fails its rule; its output is set aside (renamed, never
   deleted, never collected) and it runs again once the window is quiet.
5. **Collect**: each unit's rows become records in the store, with the unit's exit
   status. A ladder unit that exited non-zero stopped before its plan finished: its
   records are kept and never cut.

On macOS the session holds the Mac awake for its life (<code>caffeinate -i -s -w</code>
on its own process, the ladder's method) and says so in its log; on a Mac without
<code>caffeinate</code> it says that and runs on. Power is recorded beside every row and
never gated.

<code>cert8.py collect ladder|fused</code> records a unit that was run by hand; such a
record carries no quiet window and no watcher, so it is kept for
<code>status</code> and <code>bill</code> and never cut.
<code>cert8.py selftest</code> exercises the watcher's rule on stand-ins.

### A desktop under WSL2 {#a-desktop-under-wsl2}

Inside WSL2 the session's readers see the VM alone: Windows' own load, the desktop's use of
the card and the card's power state and clocks are invisible to them. On an RTX 3090 that
drives the Windows desktop, timed this way with the card's clocks free, each of three
variance runs had one pass 13 to 25 % slow, and two of the three coincided with the card
dropping from P2 to P3 (half the memory clock), which no reading of the VM showed. So on a
host whose kernel is a Microsoft build for WSL (<code>/proc/sys/kernel/osrelease</code> or
<code>/proc/version</code>), every reading of the quiet window and of the watcher also reads
Windows, through WSL interop:

- the card, from Windows' <code>nvidia-smi.exe --query-gpu=clocks.gr,clocks.mem,pstate,clocks_event_reasons.active,power.draw,temperature.gpu</code>
  (the card the session pins, matched by its UUID, which is the same on both sides and is
  never recorded);
- Windows' CPU load, from one <code>powershell.exe</code> counter sample of two readings a
  second apart: the whole machine (<code>Processor(_Total)</code>) and the WSL VM's own
  process (<code>vmmem</code>), the last reading read. The VM's load is the session's own
  and the VM's readers gate it, so what is gated is Windows' busy logical CPUs **outside
  the VM**. The sample prints numbers only: a counter path carries the Windows computer
  name.

**Clock locks are expected on a desktop timing host.** A desktop card moves its clocks and
P-state with load, temperature and the desktop, and that moves a timing. Lock both clocks
on Windows, in PowerShell as Administrator (<code>nvidia-smi -lgc G,G</code> and
<code>nvidia-smi -lmc M,M</code>, for example 1695 and 9501 MHz on an RTX 3090, whose memory
runs at 9501 MHz in its compute state P2), and declare the lock to the session with
<code>--gpu-clock-lock G,M</code> or <code>MGPU_TIMING_GPU_CLOCK_LOCK=G,M</code>. Read it back
first: at rest a locked card reads the locked clocks, P2 and the <code>gpu_idle</code>
reason.

**A desktop timing host is qualified by the variance check**, with the settings it will be
timed under: the precondition's GPU repeat spread within 5 %, on more than one run. On the
RTX 3090 desktop the clock locks were not enough: a pass was slow when Windows ran the VM's
CPU that launches the GPU work on a logical CPU that also carried the WSL2 GPU path's
interrupt and DPC traffic. That machine passed 3 of 3 variance runs only with the clocks
locked and the VM's process kept off those logical CPUs by a CPU affinity mask
(<code>0x999988</code> there), the first configuration that passed on it.

**The VM affinity mask**, <code>--vm-affinity MASK</code> or
<code>MGPU_TIMING_VM_AFFINITY=MASK</code> (hexadecimal or decimal). The session sets it on
the WSL VM's process (<code>vmmemWSL</code>) through interop before its first unit, reads it
back, and refuses by name unless the read-back is the declared mask. The VM's process
belongs to the VM's virtual account, so only an elevated context can set it: from an
unelevated one the read gives 0 and the set is denied, and the session is refused with both
named. Start WSL, and the session, from an elevated context. Every reading of the quiet
window and of the watcher then requires the mask, and a unit during which it changed is
struck. At its end the session gives the VM's process the mask it held before; a mask also
dies with the VM. **MEEP is never timed under the mask**: MEEP at 8 ranks fell to 146-168
Mcell-steps/s, with between-pass spreads of 0.13-0.29, under it. A ladder unit and a ladder
row's dispatch check time MEEP beside the GPU legs in one ladder process, where no mask can
be switched between them, so a session with a declared mask that holds one is refused by
name before the mask is set, unless it splits each ladder unit into a GPU half under the
mask and a MEEP half without it ([The split ladder](#the-split-ladder)). The variance
precondition is admitted, since it qualifies the host under the mask. It gates the GPU
route's repeat spread alone, so under the mask it times the GPU route alone, as a GPU half,
in a split session and in one that qualifies the host for fused rows alike, and records no
MEEP side (<code>variance.json</code>, <code>meep_side</code>: "not run"); one that would
time MEEP under the mask is refused by name. <code>cert8.py plan</code> takes no mask, so
it shows a masked session's precondition as a GPU half only with
<code>--split-ladder</code>.

**A mask is specific to a machine's interrupt layout**: it names logical CPUs. On the RTX
3090 desktop it leaves out the card's interrupt target set as Windows assigned it, and the
logical CPUs that carried the heaviest DPC load. After a driver update, a hardware change or
a different boot topology, find them again before reusing a mask, read-only: read the card's
interrupt target set (its device's <code>Interrupt Management\Affinity Policy -
Temporal</code> key, <code>TargetSet</code>, in the registry), sample each logical CPU's DPC
and interrupt rates (`Get-Counter '\Processor(*)\DPCs Queued/sec'` and
`'\Processor(*)\% Interrupt Time'`) while GPU rows run, keep the VM off both, and
qualify the host again with the variance check.

The rule, at every reading of the quiet window and of the watcher:

- **The card at the declared lock**, both clocks. With no lock declared, the clocks may not
  move between the quiet window's readings; while a unit runs they are recorded and not
  gated, so a host without a lock relies on the reasons and the variance precondition
  alone.
- **No clock-event reason but <code>gpu_idle</code>.** A card reports
  <code>gpu_idle</code> whenever nothing runs on it, between a unit's legs too (the RTX 3090
  at its lock reported it on all 439 readings of 74 minutes of variance runs, the 49 taken
  while the card was busy included), so it is admitted at every reading; with the clocks at
  the lock it slows nothing. Any other reason (a power cap, a thermal or hardware
  slowdown, a bit NVML does not name) is not quiet.
- **Windows' busy logical CPUs outside the VM below the threshold**:
  <code>--windows-cpu-max</code> or <code>MGPU_TIMING_WINDOWS_CPU_MAX</code>, by default 10 %
  of Windows' logical CPUs (2 of 24). **Declare it per host, from a calibration.** Starting a
  CUDA process under WSL2 shows on the Windows side as a short burst outside the VM, and on
  the RTX 3090 desktop the burst sits above the default: the gate read 2.0, 2.21 and 6.07
  busy CPUs in the three refused attempts of one variance run, and a one-second recorder of
  the gate's own counters read 7 of 96 seconds of GPU rows at 2.0 or more (99th percentile
  and maximum 4.45), against 0 of 260 idle seconds. Declared at 8 (a third of its 24 logical
  CPUs), 74 of 74 fused units opened and ran with no intrusion, the gate's own reading at
  most 4.71 in any unit; a one-unit split session read at most 1.77. Calibrate on the host:
  take the gate's own Windows reading (<code>timing_host.windows_reading</code>, its
  <code>cpu.outside_vm_busy_cpus</code>) again and again through an idle stretch and through a
  few GPU rows, and declare a value above the GPU rows' bursts. The threshold is a trade: at 8 the gate does
  not see a Windows background burst of about 1.7 CPUs over 30 seconds (on that desktop two
  of the 74 unit windows overlapped one, which only a sampler of Windows' load kept beside
  the session showed). Every record states the threshold it ran under
  (<code>gates.windows.cpu_max_outside_vm</code>: the number declared, or the default's
  rule). The default stays at 10 %: it is the host-independent rule, and a value large
  enough for one host's process starts would hide such bursts on every other.
- **With a VM affinity mask declared, the VM's process holding it.** With none declared, its
  mask is recorded and not gated.
- **A Windows value that cannot be read is not quiet.**

While a unit runs, ONE watcher reading that fails this rule strikes it: a card that left its
lock, a busy Windows, or a VM moved off its mask changed the timing whatever the next
reading says. The unit is
stopped, set aside and run again on a quiet window, as any intrusion. A session whose
Windows side cannot be read at its start (interop turned off, <code>nvidia-smi.exe</code> or
<code>powershell.exe</code> not reachable, a counter sample that does not answer, as on a
Windows whose counter names are not English) is refused by name before any unit runs. The
programs are found on the VM's <code>PATH</code>, else in Windows' system directory under
<code>/mnt/c</code>, so a driver started under <code>env -i</code> still reaches them.

What is recorded: the rule, applied or skipped and why, in every row's gates
(<code>gates.windows</code>); the Windows reading inside each reading of the quiet window and
the watcher's first and last (<code>recorded.windows</code>: clocks, P-state, reasons, power,
temperature, Windows' load whole and outside the VM, the VM's affinity mask); the declared
mask and how its set went (<code>gates.windows.vm_affinity</code> and
<code>vm_affinity_set</code>: the mask before, the set, the read-back); a summary of every
reading of the unit in the watcher's block (<code>watcher.windows</code>: the clocks,
P-states, reasons and masks met, the most Windows was busy outside the VM); and the same for
the variance precondition's unit in its record (<code>variance.json</code>,
<code>windows</code>). On native Linux and macOS nothing is read, nothing is set and nothing
changes: the rule is skipped, by name, in <code>gates.windows</code>, and a declared lock or
mask is recorded there as not checked.

**The A/B base** is the previous release, staged by <code>git archive &lt;tag&gt;</code>
and run with <code>--tree &lt;staged base&gt; --role ab</code>: its rows are kept beside
the release's and never enter a cut of the release.

### The split ladder {#the-split-ladder}

On a host whose GPU rows need the VM affinity mask and whose MEEP rows must not run under it
(the RTX 3090 desktop: under the mask MEEP at 8 ranks fell to 146-168 Mcell-steps/s, with
between-pass spreads of 0.13-0.29), <code>cert8.py run --split-ladder</code> (or
<code>MGPU_TIMING_SPLIT_LADDER=1</code>) times each ladder unit as two child runs of the
ladder in the one session, so MEEP is still re-timed in the same session as the GPU legs:

- **the GPU half**, the unit's ladder with <code>--ranks none</code>: its GPU routes, with
  the array control read off their rows, under the declared mask;
- **the MEEP half**, the unit's ladder with <code>--routes none</code>: every MEEP leg, the
  preflight's MEEP rows and digest and the harness-lift digests, with the mask cleared: every
  one of Windows' logical CPUs as the session's first Windows reading counts them
  (<code>0xffffff</code> on 24), or <code>--meep-vm-affinity MASK</code>
  (<code>MGPU_TIMING_MEEP_VM_AFFINITY</code>), which may not be the declared mask. It is
  held to the host's MEEP limit: handed <code>--meep-rank-ceiling</code> at the largest
  rank count of the rank blocks the session's site file declares (<code>RANKS</code> in
  <code>--ladder-params</code>), so no MEEP row and no control runs above it
  ([The MEEP controls](#the-meep-controls)). A split session whose site file declares no
  rank blocks is refused by name, in <code>plan</code> and <code>run</code> alike: a ladder
  would derive them from the cores it detects (12 under WSL2 on the RTX 3090 desktop).

Each half is a unit of its own for the rules above. Its mask is set and read back before
its quiet window, and the session is refused by name unless it reads back; its watcher then
requires that mask at every reading, and a half during which it changed is struck, set
aside and run again, alone. The halves run one after the other, so a drift between them
enters that unit's ratios in full: unlike the interleave inside one ladder, the split does
not cancel it within a row. The first half alternates across the session's ladder units
(the GPU half first on the first ladder unit, the MEEP half first on the second, and so
on), so such a drift does not push every row's ratios the same way; a session of one ladder
unit runs its GPU half first, and so does every one-unit session, a re-time of one key
included: the order is drawn from the unit's place in the session's plan, so the same
command always plans the same units. A half that exits non-zero
ends the unit: the other half is not run, any earlier output of it is set aside, and the
same command runs the unit again, whole. Every other unit (the variance precondition and a
ladder row's dispatch check, each a GPU half alone, and the fused units) runs under the
declared mask, set and read back before it. At its end the session gives the VM's process
the mask it held before the session's first set.

The two halves are composed into one row per key (<code>timing_rows.split_ladder_records</code>)
of the shape a whole ladder's row has: the GPU and array-control legs and the launch
counters from the GPU half, the MEEP legs from the MEEP half, each process's artifact
under its half's ladder directory, the ladder's drivers once. The row adds a
<code>split</code> block: the order the halves ran in, and per half its unit and ladder
directory, its attempt, start, end and elapsed time, its exit status, its quiet window's
times, its watcher's summary, and its VM affinity mask as declared, as it was before the set,
as set, as read back and as every reading of the half saw it. The unit block carries each
half's command, environment and exit status (<code>unit.halves</code>), the gates carry the
two masks (<code>gates.split_ladder</code>), both halves' quiet windows, and one watcher
block whose Windows summary covers both. MEEP's pass spread is recorded and never gated, as
in a whole ladder. A split row is refused by name unless both halves ran, exited 0 and wrote
rows of its key, each held its own mask (set, read back, and the only mask its readings
saw), the MEEP half's mask is not the GPU half's, each half's mask is the one the session
held for that side (<code>gates.split_ladder</code>), the GPU half's is the declared mask
and the MEEP half's is not, and both ran the same ladder drivers.

The tables of record are cut from the row store (<code>timing_rows.py table</code>): each
ratio of a split row is labelled split, with the two masks and the order, the table states
the rule, and the host is described as it is (the RTX 3090 desktop: an 8-performance-core
desktop, MEEP at 1 to 8 ranks). The ledger builders of
[Building the tables](#building-the-tables) read a split unit from its two halves and its
row record, refuse it by name on the same rule, and label every ratio split.

**A split session on a host like the RTX 3090 desktop**, with the shipped NVIDIA catalog: its
site file (<code>--ladder-params</code>) declares the host's limits,
<code>PHYSICAL_CORES=8</code> and <code>RANKS=8|1|</code> (or <code>4,8|1|</code>), and the
MEEP halves are handed the ceiling 8, so no MEEP row and no control row runs above 8 ranks
(the catalog's <code>hwthread</code> rows are refused by name:
[The MEEP controls](#the-meep-controls)); a <code>PHYSICAL_CORES</code> left to detection
(12 under WSL2) is refused by name, since the preflight MEEP row runs at it. The session
declares the clock lock, the mask, the split and the host's own Windows threshold
([A desktop under WSL2](#a-desktop-under-wsl2)):

```bash
python -u parity/meep_gpu/rounds/1.0/cert8.py run --catalog \
  parity/meep_gpu/rounds/1.0/cert8_catalog_nvidia.json --store <store> --run-root <runs> \
  --session <id> --python <env>/bin/python --gpu 0 --environment-record <build record> \
  --ladder-params <site params> --gpu-clock-lock 1695,9501 --vm-affinity 0x999988 \
  --split-ladder --windows-cpu-max 8 --rows ...
```

<code>cert8.py plan --split-ladder</code> with the same catalog, store, run root, session,
interpreter, GPU, site file and rows (the clock lock, the masks and the Windows threshold are
<code>run</code>'s alone) prints each half's ladder command; that command with
<code>--dry-run</code> appended prints what the half plans, the controls refused by name under
<code>controls_omitted</code> (each naming the ceiling) included; with
<code>--dry-run --json</code>, the MEEP half's ceiling is
<code>resolved.meep_rank_ceiling</code>.

Without <code>--split-ladder</code> nothing changes: a ladder unit is one process with its
GPU and MEEP rows interleaved, and its plan is the plan drawn before the ladder could time
one side alone, byte for byte (<code>test_timing_ladder_plan.py</code> pins the A6000
profile's). <code>--split-ladder</code> with no mask to split around (none declared, or a
host that is not WSL2) is refused by name: run without it there.

### Valid, stale, or owed a dispatch check

<code>timing_rows.py status --store S --release-root R</code> reads every key's newest
row against the release tree. The files a row's import records name that the tree holds
at other bytes have moved (the rule of <code>artifact_drift.py</code>). No moved file:
the row is valid. Otherwise each moved file is one of four kinds:

1. **Step-path files re-time every row that loaded them**: every file outside a kernel
   table directory (the driver, the planners, the deposit repair, the host-write doors,
   the lift, the benches), the fused-pair emitters and the shared fusion block
   (<code>dispatch_preference.SUBJECT_FILES</code>), each table's package init and arm
   table, <code>triton_kernels/launch.py</code>, and every file a table's
   <code>driver_dispatch</code> record pins that no arm's entry pins. The 150 NVIDIA import records of the 1.0 release's
   timing held the planners, <code>deposit_repair.py</code>,
   <code>host_writes.py</code>, <code>fused_pairs.py</code> and
   <code>fused_polarization_pair.py</code> in every row of both tables: there the re-time
   stays per table, because pooling rows timed on both sides of such a change would
   compare two programs.
2. **An arm's kernel file re-times exactly the keys whose launch counters show that arm
   dispatched.** The arm's ledger entry (the gate its plan cites) pins its own modules,
   and every kernel file they can import belongs to it too. An arm no ledger entry
   attributes (the 1.0 Triton single arms cite a recertification sweep) takes its
   table's whole kernel slice (27 of the 83 arms the 1.0 rows launched). On the Metal
   table every arm reaches the 63 Triton kernel files its emitters import, so a Triton
   kernel change re-times every Metal row that launched an arm. A Metal kernel change
   re-times only the rows whose arms reach it. The walk stops at each table's
   <code>registry.py</code>, which imports every family so that a plan reads a complete
   table: through it every arm would own every kernel file. A Metal entry may pin only
   <code>launch.py</code>, so the kernel files its gate script names belong to the arm too.
3. **A kernel file no launched arm carries owes the row a dispatch check**, because a
   predicate in it may admit another arm. The row is valid again once a
   <code>check</code> record of its key, cut wholly on the release tree, shows the same
   arms launched on the same slots.
4. **A ledger owes the row a dispatch check**: the four ledgers are what the dispatch
   path reads to admit arms, no arm's kernel text, so a change to one (a run recorded, an
   arm admitted) is checked as rule 3 is, never left unseen. A row is compared only on
   the ledgers its architecture reads (<code>parity/meep_gpu/ledger_scope.py</code>): the
   Triton and hand-CUDA ledgers on an NVIDIA row, the Metal ledger on an Apple row, and
   the ledgers of any table the row launched. The bench also records the other platform's
   digests when its process imported a module of that table; importing one opens no
   ledger, and those digests are kept and not compared. <code>certification.json</code> is
   opened by no dispatch path and owes a row whose device can be placed nothing. Within a
   ledger it reads, a row reads a slice for its architecture
   (<code>docs/development/certification.md</code>, "Carrying a route campaign or an L3 leg
   to a new tree", defines it): a move outside it, another device's bind or a route re-cut,
   is set aside and named under <code>set_aside</code> in the assessment, so a row, or a
   check cut before that bind, stands. A digest that cannot be set aside is compared whole and
   the reason is under <code>compared_whole</code>; <code>status</code> prints how many moved
   digests no archived state answers for. A process that ran package code and records no
   ledger of its own platform that a process reads is refused by name.

   One incidental catch is narrower for it: no row records a timing record's digest, so this
   rule never compares one, and a second card's record that moves a default plan owes a row
   nothing. A ledger move used to force a check that would also have seen that.

A row that records no launched slot (an <code>ERROR</code> row) is never narrowed: every
moved file it read, under rule 4's scope, re-times it. A row is refused by name, whatever the tree, when a process that
ran package code recorded no import record, no package module in it or no ledger
digests; when its unit was not opened on a quiet window of two clear readings or not
watched (a row collected by hand); when the watcher saw an intrusion; when a ladder
entry of it was written outside the session that records it; when its ladder unit
exited non-zero; when it names no reference environment build record; when its rows
stamp another architecture than its key, or the record itself does; or when its body does not
match its digest.

**The environment is no file of the import record.** A rebuilt reference
environment (another compiler, Triton, CuPy or MEEP build) runs another program behind
the same import records, so every command that selects rows takes
<code>--environment-record &lt;the round's build record&gt;</code>: a row that ran in
another environment, or recorded none, is then stale and owed a re-time, and a dispatch
check witnesses a row only from the same environment. A table, record or report whose
selected rows ran in two environments (possible only without the flag), or whose
processes loaded two versions of one library, is refused by name rather than pooled.

Before a change lands, <code>timing_rows.py bill --store S FILE ...</code> lists the keys
it will re-time, the keys it will owe a dispatch check and the keys it leaves alone; after
it lands, <code>cert8.py run --owed</code> (given the session's
<code>--environment-record</code>, which <code>plan --owed</code> reads the same way)
times exactly those. On the 113 fused rows of the 1.0 release's
timing, a change to the CUDA complex fused electric pair's module re-times 5 and owes
5 a check (103 untouched); one to <code>fused_pairs.py</code> or <code>driver.py</code>
re-times all 113.

### The cuts

```bash
python parity/meep_gpu/timing_rows.py table --store S --release-root R --catalog C \
  --environment-record E --architecture 8.6 --out-md TABLE.md --out-json table.json
python parity/meep_gpu/timing_rows.py records --store S --release-root R --catalog C \
  --environment-record E --out-root meep_gpu
python parity/meep_gpu/timing_rows.py report --store S --release-root R --catalog C \
  --environment-record E --architecture 8.6 --out-dir <dir>
```

**The fused rows are timed under the shape rule.** A record key prices the shape rule's
fused product against its certified singles, and the package default arbitrates those
products by the records cut here, so a fused leg timed under the default may already
have composed the singles a record chose. <code>bench_fused_products.py</code> therefore
exports <code>MEEP_GPU_PREFER=span</code> before it lifts a leg, unless the caller's
environment sets the switch (a caller that wants the default timed exports
<code>measured</code>), records the value in each row's
<code>process_environment</code> and in <code>fused_leg_preference</code>, and copies
each leg's preference block from its dispatch record into the row
(<code>plans.&lt;leg&gt;.preference</code>). <code>cut_timing_record.py</code> admits a
fused row only when its <code>process_environment</code> says <code>span</code>, leaves
the others out by name and owes their cases under <code>missing</code>; a record so cut
stamps <code>cut.fused_leg_preference</code>, and <code>--check</code> recomputes a
record without the stamp by the rule it was cut by.

- **The identical-case table** takes each key's newest valid row that carries a
  measurement (a later attempt that failed does not shadow an earlier timing). A GPU
  route or the array control carries the 5 % gate of record: no pass struck by the
  bench's floors (its window floor aside) or the ladder's checks, and the spread of its
  pass rates within 5 %;
  each pass's window spreads are printed beside it and gate nothing. Every leg is
  re-derived from the passes its record stores, not read off the status it was
  collected with. A MEEP leg carries its between-pass spread,
  recorded and never gated, and is struck only for what is not spread (its geometry
  digest, its precision, its threads, its monitor). A ratio is the route's median over
  the best unstruck MEEP reference median of the same row and says whether it enters:
  it enters when the route passed the gate and the session's variance precondition read
  GO. Every median is printed with its gate status. The table is accounted against the
  catalog: every catalog key without a valid measured row, one never timed included, is
  listed with what it is owed.
- **The dispatch-preference records** are cut by <code>cut_timing_record.py</code> from
  the selected fused rows' own lines, one record per table **per architecture and
  card** ([Records per card](#records-per-card)). The first card cut for an
  architecture lands at the name <code>dispatch_preference.record_path</code> gives it,
  <code>&lt;table&gt;_kernels/timing_cc86.json</code> (for the Metal table
  <code>metal_kernels/timing_applegpu_g13s.json</code>), and represents that
  architecture; another card's record lands beside it under its own card name. A new
  architecture's record is added beside the others; a record of an architecture whose
  rows are absent is never touched; a cut never replaces another card's record, and
  there is no override. The release is
  accounted for: a record cut from part of a campaign names every released (product,
  case) it lacks under <code>missing</code>, with the rows' own reason (the eleven
  <code>ERROR</code> rows of the first 1.0 timing, 2026-10-13, before a harness
  fix), instead of failing. Every catalog fused key of the
  table and architecture without a valid measured row (a sweep point never timed
  included) is named in the cut's output and in the report beside the record, and a
  table the catalog names that no valid row reaches is named as not cut (a table whose
  DRIVE cases are the release's and that has no sweep included). A row is admitted on
  its floors without the window spread, and the rows of one case and grid shape timed
  by one instrument on one deposit-repair program (a sweep point's passes; a DRIVE unit
  at the same grid is a row of its own) enter only when their fused and their baseline
  rates agree within 5 %; a record so cut says so in <code>cut.gate</code>
  (<code>between_pass</code>). Each row's window spreads stay in the record and in the
  veto margin, which <code>dispatch_preference.py</code> defines and which is no
  admission gate. An NVIDIA row's neighbours are read from its device probe, joined by
  device UUID; a Metal row's from the bench's own host record (<code>host_before</code>,
  <code>host_after</code>: the other processes driving the one Metal device), because the
  device probe identifies no device on a Mac. A host record that did not read them (no
  list, the read named under <code>unreadable</code>, or a record taken off macOS) is not
  read, never clean, and a Metal row without <code>host_after</code> is admitted and
  counted as one without an after-probe, as an NVIDIA row without <code>box_after</code>
  is. A Metal record states the Metal rule set (<code>RULES_METAL</code>); the NVIDIA
  records' rule sets do not move.
- **Records priced by size (<code>/3</code>)**: <code>timing_rows.py records --costs</code>
  (or <code>cut_timing_record.py --costs</code>) cuts the same keys and adds, per product
  class with a measured key, one cost point per size at least <code>min_rows</code> rows
  measured: the median fused and baseline ms/step, each leg's largest window spread and the
  least margin of that size's rows (a key's veto rule).
  The record is stamped <code>meep_gpu.dispatch_preference.timing/3</code> and
  <code>cut.costs</code>, the reader recomputes the points from the record's own
  measurements, and <code>--check</code> recomputes either kind by the stamp it carries.
  The dispatch preference reads a <code>/3</code> record at a run's own cell count: at a
  measured size the point's own margin answers, strictly between two measured sizes the
  straight line between those two points' margins, and never outside them, so a campaign
  that times a product at more sizes moves the choice at the sizes between them by
  writing the record alone.
- **The fused-product report** is <code>build_fused_timing_report.py</code> on the selected
  fused rows, staged as that builder reads them, with the catalog's keys it lacks, and
  its tables no valid row reaches, in <code>missing_keys.txt</code> beside it. The
  release's DRIVE cases are known only by running the bench's estimate, so a table that
  takes them is named there with a pointer to its record's <code>missing</code> block,
  which names every released (product, case) without a row. It lists a row as refused
  only under a floor that gates; the window spread is not one.

<code>cut_timing_record.py --check</code> recomputes a record from the row files it names,
by the rule it was cut by: a record with no <code>cut.gate</code> was cut by the window
floor (every record cut before the rule changed on 2026-10-16, the 2026-09-21 records
among them) and recomputes by
it. The store is written once per key and attempt, so every file a record names stays
there.

### Records per card {#records-per-card}

A timing record is the evidence of one card: rows of two devices never pool
(<code>mixed_device</code>). The card is the exact device name the rows stamp: the
record's <code>host[].device</code>, which a run's dispatch record carries as
<code>environment.device.name</code>. In a file name it is spelled by its **slug**
(<code>dispatch_preference.card_slug</code>, the reader's, which the cutter imports): the
name in lower case, every run of
characters outside <code>a-z</code> and <code>0-9</code> one <code>_</code>, none at
either end; a name that leaves nothing is refused (<code>unnamed_card</code>). Cards
match by slug, never by prefix or family: an RTX 3090 Ti is not an RTX 3090. Two exact
names with one slug are two cards, and an architecture's records keep one exact name per
slug (the cutter refuses the second), so a slug never stands for two cards.

| Device name | Slug | Its record beside another card's representative |
|---|---|---|
| NVIDIA RTX A6000 | <code>nvidia_rtx_a6000</code> | <code>&lt;table&gt;_kernels/timing_cc86-nvidia_rtx_a6000.json</code> |
| NVIDIA GeForce RTX 3090 | <code>nvidia_geforce_rtx_3090</code> | <code>&lt;table&gt;_kernels/timing_cc86-nvidia_geforce_rtx_3090.json</code> |
| NVIDIA GeForce RTX 4080 | <code>nvidia_geforce_rtx_4080</code> | <code>&lt;table&gt;_kernels/timing_cc89-nvidia_geforce_rtx_4080.json</code> |
| NVIDIA H200 | <code>nvidia_h200</code> | <code>&lt;table&gt;_kernels/timing_cc90-nvidia_h200.json</code> |
| Apple M1 Max | <code>apple_m1_max</code> | <code>metal_kernels/timing_applegpu_g13s-apple_m1_max.json</code> |

**The representative.** <code>&lt;table&gt;_kernels/timing_&lt;arch&gt;.json</code>, the
name <code>dispatch_preference.record_path</code> spells, is the architecture's
representative: the record the dispatch path reads for every card of that architecture
that has no record of its own. It is designated by where it sits, not by code, and the card it was cut
on is data: the record's <code>host[]</code>, the <code>card:</code> line of the report
beside it, and <code>--list</code>. Any other card's record is
<code>timing_&lt;arch&gt;-&lt;slug&gt;.json</code> beside it
(<code>dispatch_preference.record_path(table, architecture, card=...)</code>), with its
report at <code>timing_&lt;arch&gt;-&lt;slug&gt;.report.txt</code>, and the dispatch path
reads it for that card alone. Neither a slug nor an architecture contains
<code>-</code>, so a name reads back one way, and every <code>timing_*.json</code>
listing finds it. The reader refuses a card's own record whose rows were timed on
another card, or on more than one (<code>record_named_for_another_card</code>).

**Where a cut lands.** <code>--out</code> must be exactly the destination, which the
cutter reads off the rows and the directory <code>--out</code> names:

1. the bare name, when it already holds this card's record (a re-cut);
2. else this card's own name, when that file exists (a re-cut of a card record, wherever
   the representative stands);
3. else the bare name, when nothing is there: the first card cut for an architecture
   represents it;
4. else this card's own name, beside the representative.

Anything else is refused by name and writes nothing:
<code>out_names_another_capability</code> (a name of another architecture, or of no
record), <code>existing_record_names_another_device</code> (a file holding another card's
record; another exact device's record under this card's slug, at either name; a file
whose rows name more than one device; there is no override),
<code>existing_record_unreadable</code> (a file at <code>--out</code>, at the bare name or
at the card's own name that does not read as a record, so it cannot say whose it is),
<code>out_names_another_card</code> (any other name; the refusal names the destination and
states the four rules above), <code>card_has_two_records</code> (the card already at both names) and
<code>unnamed_card</code>. A file at either name that cannot say it is one card's record
refuses the cut whatever <code>--out</code> names, because it could be this card's own.
A re-cut rewrites only its own card's file. A cut never retires, renames, moves or
designates: **changing an architecture's representative** is a deliberate change to the
tracked records. Retire the representative's record and report, and the chosen card's
own record and report when it has them (while that file exists, rule 2 sends the card's
cut back to it); then cut the chosen card's rows, which land at the empty bare name by
rule 3. After writing a card record the cutter says which card reads it and which
record every other card of the architecture reads, and says too when the directory holds
no representative.

**What a run reads.** A card reads its own record when the package ships one, and every
other card of the architecture reads the representative; a card's own record that the
reader refuses or that answers a barrier (its deposit-repair route or emitters moved) is
that card's answer, never replaced by the representative. An architecture with neither
reads the portable prior of its vendor's **nearest timed architecture**: one with a
record that holds on the tree and measures a key. For NVIDIA, distance is the compute
capability read as <code>major * 10 + minor</code> (8.6 is 86, 9.0 is 90), and the lower
capability wins a tie: a cc 8.9 card reads 9.0's prior when both 8.6 and 9.0 are timed,
and a cc 8.7 card 8.6's. For Apple, distance is the chip generation the GPU family names
(<code>applegpu_g13s</code> is generation 13): the family itself first, then the smallest
difference in generation, then the lower generation, then the first spelling in order.
Only the bracketed-veto prior crosses architectures, never a whole record and never a
vendor; a nearest architecture whose records measured no bracketed veto below their
keeps gives no prior, and a farther one is not asked instead. A live architecture or a
timed one the rule cannot read a distance from (a device whose architecture was not
read, an Apple family spelled otherwise) is refused by name and its table keeps the
shape rule. <code>dispatch_preference.nearest_architecture</code> is the rule's one home.

**What ships.** The package may ship any number of architectures per vendor and any
number of cards per architecture, each record at the name the reader opens for it:
the representative's bare name, or its own card's name beside it, agreeing with the card
its rows ran on, one file per card per table and architecture, and the NVIDIA tables'
representatives of one architecture cut on one card. In
<code>meep_gpu/test_dispatch_preference.py</code>,
<code>test_every_shipped_record_is_at_a_name_the_reader_opens_and_agrees_with_it</code>
refuses any other layout,
<code>test_every_shipped_records_architecture_is_ranked_by_the_nearest_rule</code> refuses
a record whose architecture the nearest rule cannot rank (it would refuse the prior to
every device of its vendor), and <code>test_every_shipped_record_holds_on_this_tree</code>
refuses a shipped record whose deposit-repair route or compared emitters have moved
since it was cut, because every consult of such a record answers "no record". A
release ships at least one timed NVIDIA record and one timed Apple record, each pricing
at least one key (<code>test_the_package_ships_an_nvidia_and_an_apple_timing_record</code>).
So a second architecture's record, or a second card's, ships as data alone: cut it on
the release bytes, place it where the cutter puts it, and the reader reads it with no
code change. This tree ships three records: compute capability 8.6's for the Triton and
the hand-written CUDA tables, both cut on one RTX 3090, and the Metal table's for
<code>applegpu_g13s</code>, cut on one Apple M1 Max; no other architecture has a record,
and all three are timed, so the release test passes. How a run's
record says which record it read, and whether that is its own card's, is in
[Which record a run read](../guides/kernel-dispatch.md#which-record-a-run-read).

**One store, one card.** <code>timing_rows.py records</code> places each record by the
same rule and returns where it went (<code>path</code>) and its card
(<code>card</code>); the report beside the record names the card. A row's key names no
card, so a store holds one card's rows: two cards' rows of one key shadow each other
(the newest is selected), and selected rows of one table and architecture that name two
cards are refused by name (<code>mixed_device</code>). Time another card into a store of
its own.

**Listing.** <code>cut_timing_record.py --list [--root DIR]</code> (default: the
package) prints, per record file, its table and architecture, its role (representative,
card record, or named for neither), its card and slug, <code>record_sha256</code>,
<code>recorded_utc</code>, schema, and whether its deposit-repair route and compared
emitters hold on this tree. It writes nothing.

<code>plan_changes.py report</code> lays each candidate record over the shipped ones as
its architecture's representative, which is what landing it does, and refuses two
candidates of one table and architecture (<code>two_candidates_for_one_architecture</code>).

### The gate of record {#the-gate-of-record}

A GPU leg enters when the spread of its pass rates, (max − min) / median over the row's
passes, is at most 5 %, and no pass of it is struck for anything else. The rule is the
same on both platforms, for the ladder's rows (each GPU route and the array control) and
for the fused-product rows (a sweep point's two passes; a DRIVE unit is timed once). The within-pass window spread is
still computed and recorded with every pass and every row, and no cut gates on it. MEEP's
pass spread is recorded and never gated, as before. Unchanged: the quiet window, the
variance precondition, the watcher, the identity and dispatch checks, and the bench
itself, floors included: the change is in the cutters, which read past its window floor.

It replaced the window-spread gate on 2026-10-16, and the change was decided after
the 2026-10-16 runs had been seen. The bench marks a pass in which any leg's timed
windows spread above 5 % <code>TIMED-NOT-REPORTABLE</code>
(<code>spread_within_gate</code>), and until then that struck the leg. A published figure is a median
of pass medians, so what has to hold for it is that the medians reproduce. On the Mac's
2026-10-16 ladder the window floor struck 17 of 24 timed GPU passes and 6 of 8 GPU
medians. The jitter was one-window blips: where the fused leg's 2 s windows spread above
5 % in a pass (11 of the 17), leaving out its single farthest window brought the spread
to 2.1-5.1 %. The same medians reproduced across passes to 0.16-2.44 %.

Every row is cut by this rule, whenever it was written. A pass whose ladder struck it
on the window floor alone (<code>TIMED-NOT-REPORTABLE: floors failed
['spread_within_gate']</code>) is read unstruck; a strike naming another floor as well
keeps that floor; every other strike stands. One limit is stated in every table: the
ladder's recorder stops at a row's first strike, so on such a pass it never evaluated
its later checks (what the route launched, a GPU-only row's identity row, the
deposit-repair route, the host conditions), and the record keeps none of them to
re-read. The limit holds for every pass the bench strikes on its window floor, new rows
included, for as long as the bench counts it among its floors. A fused DRIVE row is timed
once, so it has no between-pass spread and is not
struck by one.

## What a row licenses

One host, one GPU, the precision both sides ran, the case and size the row names,
the monitors it carried, the MEEP build and configuration it records, and the
gate rule stated beside it. A row licenses no statement about another case,
size, host, GPU, MEEP build or rank count. Rows taken with
<code>--allow-uncertified 1</code> are not certified and enter no certified
figure.

## The 1.0 timing against 2026-10-13

The 1.0 code was timed twice on the same two machines, with MEEP re-timed in each
session: on 2026-10-13, on an earlier development tree of 1.0, and on 2026-10-16, on 1.0's
released bytes. The 2026-10-16 tables are 1.0's tables of record
([Will it help?](../guides/will-it-help.md#the-comparison-with-meep)), cut from the row store
under [the gate of record](#the-gate-of-record), with the release tree as
<code>--release-root</code>. A row is valid on a tree whose files match the ones its
processes recorded importing: the package and the timing harness, including
<code>bench_fused_products.py</code>, which every GPU row imported. The cutters
(<code>timing_rows.py</code>, <code>cut_timing_record.py</code>,
<code>build_fused_timing_report.py</code>) are imported by none of the timed processes, so a
change to them owes no row a re-time; a change to a file a row imported, the bench
included, owes a re-time to every row that imported it. The 2026-10-13 ladders were recorded
before the row store
existed: they are ladder ledgers, without the quiet-window, watcher and environment records a
row carries, so the row tool selects none of them for a release. For this comparison their
ledgers were cut by the same table cutter, under both gates.

**NVIDIA, one RTX A6000**, per step, Mcell-steps/s (medians of three passes; "new / old" is
2026-10-16 over 2026-10-13; MEEP is each date's own build, the build of the first 1.0 timing on
2026-10-13 and the reference build on 2026-10-16):

| Cells | Leg | 2026-10-13 | 2026-10-16 | New / old | Ratio against MEEP's fastest |
|---:|---|---:|---:|---:|---|
| 512,000 | Triton route | 745.2, struck by the window gate, enters under the gate of record | 958.2 | 1.286 | 1.28× → 1.65× |
| | CUDA-only route | 687.8 | 841.5 | 1.223 | 1.18× → 1.45× |
| | array path | 84.2 | 84.4 | 1.002 | |
| | MEEP, 32 ranks (fastest) | 583.1 | 581.9 | 0.998 | |
| 2,097,152 | Triton route | 2,560.5 | 2,589.8 | 1.011 | 3.94× → 3.98× |
| | CUDA-only route | 2,844.0, struck (9.2 % between passes) | 2,985.9 | 1.050 | 4.37×, struck → 4.59× |
| | array path | 316.5, struck (8.8 % between passes) | 349.9 | 1.105 | |
| | MEEP, 48 ranks (fastest) | 650.1 | 651.1 | 1.001 | |
| 7,077,888 | Triton route | 2,797.1 | 2,794.3 | 0.999 | 5.02× → 5.04× |
| | CUDA-only route | 3,218.5 | 3,221.0 | 1.001 | 5.78× → 5.81× |
| | array path | 374.1 | 374.0 | 1.000 | |
| | MEEP, 48 ranks (fastest) | 557.1 | 554.8 | 0.996 | |

- **The denominator held.** MEEP's fastest moved by 0.996 to 1.001. The 2026-10-13 MEEP build
  (`-g -O2`, portable, built without `-ffp-contract=off`), timed again on the same host after
  the 2026-10-16 session, MEEP alone, reads within 0.3 % of the reference build at MEEP's
  fastest rank count at 2,097,152 and 7,077,888 cells; at 512,000 cells the reference build
  reads 3.8 % faster, about as much as that MEEP run's own spread between its passes (3.8 %),
  so the bridge does not separate the two builds there. Every move above is the GPU's.
- **What moved is a fixed cost per step**, and it vanishes with size: the Triton step went
  from 0.687 to 0.534 ms at 512,000 cells (−0.153 ms), 0.819 to 0.810 ms at 2,097,152 and
  2.530 to 2.533 ms at 7,077,888; the CUDA-only step from 0.744 to 0.608 ms (−0.136 ms),
  0.737 to 0.702 and 2.199 to 2.197. It is **attributed by inference only**, with no A/B run
  to isolate it, to 1.0's skip of a silent source table's deposit (CHANGELOG, "A
  silent source table skips its deposit where MEEP's subtraction leaves every word"): the timed
  windows run after the case's pulse has ended. The fused-product rows agree: at the sweep's
  24,000 cells the CUDA single arms of `pml_2d` fell from 0.502 to 0.366 ms per step
  (−0.136 ms) and its two-pair fused step from 0.825 to 0.768, while from 216,000 cells up
  every sweep point reads within 2 % of 2026-10-13.
- **Not attributable from these rows**: the CUDA-only and array-path moves at 2,097,152 cells
  (+5.0 % and +10.5 % against medians struck on 2026-10-13); the array path reads level at
  the other two sizes.
- **The lift** moved with the code (the bulk permittivity read, CHANGELOG "The lift reads
  MEEP's stored permittivity in bulk, on the builds it was compared on"): 19.3, 69.7 and
  218.4 s on 2026-10-13 against 7.4, 19.5 and 48.5 s on the Triton route; the derived
  break-even went from about 73,000 (struck), 20,700 and 17,200 steps to about 5,900, none and
  460.
- **Fused products**: over the 63 cases timed both times on the same table, the fused step
  fell on 58 and the single arms on 63, the single arms by more, so the fused-against-singles
  ratio reads lower on 58 of 63 (the CUDA `pml_2d` row 0.610x → 0.464x). The eleven
  2026-10-13 cases that stopped on a harness defect all time on 2026-10-16. The two-pair
  crossover stays between 2,400,000 and 5,400,000 cells.

**Apple, one M1 Max**, per step, Mcell-steps/s (MEEP is each date's own build, as on the
NVIDIA host):

| Cells | Metal route, 10-13 → 10-16 | MEEP, 8 ranks | MEEP, 1 rank | Ratio against MEEP at 8 ranks |
|---:|---:|---:|---:|---:|
| 110,592 | 170.9 → 261.1 (+52.7 %) | 312.2 → 318.0 (+1.8 %) | 74.4 → 73.5 (−1.2 %) | 0.55× → 0.82× |
| 262,144 | 367.4 → 508.2 (+38.3 %) | 414.4 → 418.7 (+1.0 %) | 87.3 → 85.2 (−2.3 %) | 0.89× → 1.21× |
| 512,000 | 600.8 → 765.5 (+27.4 %) | 398.6 → 452.3 (+13.5 %) | 91.3 → 95.5 (+4.6 %) | 1.51× → 1.69× |
| 2,097,152 | 1,085.2 → 1,080.2 (−0.5 %) | 441.2 → 449.5 (+1.9 %) | 98.9 → 99.8 (+0.9 %) | 2.46× → 2.40× |
| 7,077,888 | 1,365.3 → 1,363.2 (−0.2 %) | 492.8 → 500.3 (+1.5 %) | 106.8 → 105.4 (−1.3 %) | 2.77× → 2.72× |

No 2026-10-13 ratio enters under either gate, and every 2026-10-16 ratio enters under the
gate of record. **Nothing here is attributed**: three things changed at once, the GPU code
(an earlier development tree of 1.0, then 1.0 as released), the MEEP build (built without
`-ffp-contract=off`, then the reference build) and the GPU's windows (10 windows of 12 s
asked, which came out at about 2.6 to 3.2 s, then six 2 s windows sized from a probe), and the
host's background differed. What the numbers show, and no more: at 2,097,152 and 7,077,888
cells the Metal route and MEEP at 8 ranks both read within 2 % of 2026-10-13, so the
contraction-off build did not slow MEEP at 8 ranks measurably there; at 512,000 cells and
below the Metal route reads 27 to 53 % above 2026-10-13, the sizes where that run's windows
were shortest. No bridge between the two MEEP builds ran on this machine: the earlier
build's environment no longer exists.

**The 2026-10-13 rows under each gate.** On the NVIDIA host the window gate admitted 4 of 6
GPU ratios and the gate of record admits 5 of 6 (the Triton route at 512,000 cells, whose
three pass medians agree within 0.86 %); the CUDA-only route at 2,097,152 cells stays struck.
On the Apple laptop the window gate admitted 0 of 5 Metal medians and the gate of record 5 of
5, yet 0 of 5 ratios enter under either: that session's precondition spread 5.75 % on the
Metal route, above the 5 % of the release's rule (the session itself had restated a 10 %
bound and read GO). Of the passes the gate of record newly admits on 2026-10-13, read against
the ladders' own logs, none fails a check the window gate's recorder skipped (3 of 18 timed
NVIDIA GPU passes, 11 of 15 Apple). The fused rows of 2026-10-13, cut by the record cutter
under each gate for this comparison only: Triton admits 37 of 67 rows under the window floor
and 62 of 67 under the gate of record; CUDA 29 of 57 under both.

## Reproducing the 2026-09-28 ladder

<code>parity/meep_gpu/fixtures/timing/a6000_2026-09-28_b.params</code> is that
ladder as parameters, and
<code>fixtures/timing/ladder_2026-09-28_b_commands.json</code> the commands it ran.
<code>test_timing_ladder_plan.py</code> checks that the dry run plans the same 111
rows, in the same order, under the same keys, with the same commands, working
directories and environments, with paths, stamp and times compared through
placeholders and the box string by its key set (<code>busy_cores</code> renamed
<code>busy_cpus</code>). The differences from that ladder, all deliberate:

1. The busy-CPU reading, in the gate, in the box string and in
   <code>box_trace.log</code>.
2. The gate refuses an unreadable sample.
3. A process group and a timer replace <code>setsid</code> and
   <code>timeout</code> (TERM, then KILL 60 s later; a timed-out row reads rc 124).
4. Ledger fields are added; none is removed.
5. The ported bench, digest and array-control drivers replace that ladder's; at
   default options they behave as before, and the digest is pinned by a test.
6. The repository root is found by name.
7. Neither the ladder nor the bench sets <code>KMP_DUPLICATE_LIB_OK</code>: the
   reproduction's launching shell exports it.
8. Every row unsets <code>MEEP_GPU_HARDWARE</code> unless
   <code>--allow-uncertified 1</code>, which exports <code>any</code>, except the
   NVIDIA rows, which export <code>certified</code>; and every row unsets the
   retired <code>MEEP_GPU_ALLOW_UNCERTIFIED</code>, which the package refuses by
   name. That ladder inherited <code>MEEP_GPU_ALLOW_UNCERTIFIED</code> unrecorded,
   and on NVIDIA unset then meant what <code>certified</code> means now.
9. A MEEP row no longer inherits the GPU rows' cache directories, and an empty
   library path no longer leaves a trailing <code>:</code> in a GPU row's
   <code>LD_LIBRARY_PATH</code>.
10. A preflight MEEP row that is struck (its digest, precision or threads) stops
    the ladder, as a preflight GPU row that fails its checks always did.
11. A GPU row timed without its monitors attached is struck, and a plan whose GPU
    flags lack <code>--monitors attached</code> is refused.
12. Every ledger line carries <code>host</code>, <code>platform</code> and the
    physical core count; the keep-cache mint runs from the tree that runs.
