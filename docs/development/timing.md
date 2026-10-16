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
| <code>build_meep_133_native.sh</code> | A native-tuned MEEP build, as a timing control |
| <code>timing_rows.py</code> | The row store: a row's record, its validity on a release tree, and the cuts (the identical-case table, the dispatch-preference records, the fused-product report) |
| <code>rounds/1.0/cert8.py</code> | The release round: times any subset of rows, one row at a time ([below](#the-release-round-one-row-at-a-time)) |

## What a row measures

- **A GPU row** is one run of <code>bench_fused_products.py</code> on one case and
  size: three legs (the dispatched route, the certified single kernels, the array
  path), lifted in one process, timed in AB/BA-interleaved windows with monitors
  attached. Its verdict is the harness's own (<code>TIMED</code> or not). The
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

**The dispatch-preference records are keep-only.**
<code>meep_gpu/triton_kernels/timing_cc86.json</code> and
<code>meep_gpu/cuda_kernels/timing_cc86.json</code> are cut from rows timed under
<code>keep</code> (each record's <code>policy</code> is <code>["keep"]</code>;
<code>cut_timing_record.py</code> refuses rows of mixed policy). A record key carries no
policy axis and <code>dispatch_preference.consult</code> takes no policy, so once the
consult is wired into dispatch it answers a flush run from keep rows. Nothing in the 1.0
dispatch path consults the records yet (<code>meep_gpu/dispatch_preference.py</code>,
"UNWIRED"): they move no slot under either policy.

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
| <code>--routes</code> | <code>default,cuda</code> on Linux, <code>metal</code> on macOS | <code>default</code> asks the Triton table; <code>cuda</code> exports <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>; <code>none</code> times MEEP only |
| <code>--expect-launched ROUTE:CASE=TABLES</code> | none | Absent, the preflight row pins what every later row of that route and case must launch |
| <code>--allow-uncertified</code> | 0 | 1 exports <code>MEEP_GPU_HARDWARE=any</code> to the GPU rows; such rows are uncertified and say so. 0 exports <code>MEEP_GPU_HARDWARE=certified</code> to the NVIDIA rows, so a supported device that is not certified is refused rather than timed uncertified, and unsets it for every other row |
| <code>--cases</code> | <code>pml_3d</code> | Cases of <code>timing_cases.py</code> |
| <code>--priority-sizes</code>, <code>--rest-sizes</code>, <code>--largest-sizes</code>, <code>--probe-size</code>, <code>--preflight-size</code> | the 2026-09-28 cell counts | Cells; or <code>--priority-res</code> etc. as resolutions of the first case. A size a case cannot hit exactly is refused |
| <code>--ranks "A\|B\|C"</code> | from the physical cores P | A = P/3, P/2, 2P/3, P; B = 8 (when below P/3) and 1; C = 4P/3 on an SMT host, or every logical CPU when the host has more of them than P without SMT. P = 48 with 96 logical gives 16, 24, 32, 48 \| 8, 1 \| 64; an M1 Max (P = 8 performance cores, 10 logical) gives 3, 4, 5, 8 \| 1 \| 10 |
| <code>--binding</code> | <code>probe</code> | Or <code>bound</code>, <code>unbound</code>. macOS: <code>unbound</code> only |
| <code>--physical-cores</code>, <code>--logical-cpus</code> | detected | Linux: unique (package, core) over the online CPUs in this process's affinity; macOS: <code>hw.perflevel0.physicalcpu</code> and <code>hw.logicalcpu</code> |
| <code>--sockets</code>, <code>--threads-per-core</code> | detected | <code>srun</code>'s bound modes balance ranks across the sockets and are refused when the count is unknown |
| <code>--controls</code> | <code>native,threads2,threads4,hwthread,split</code> | <code>none</code> for a 2026-09-28 reproduction (below) |
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
library is not loaded yet, which refused every hybrid row of the 1.0 round). A
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

**The native build.** <code>bash parity/meep_gpu/build_meep_133_native.sh</code>
builds MEEP 1.33.0 through the platform's reference script with
<code>-O3 -march=native</code> (Linux x86_64) or <code>-O3 -mcpu=native</code>
(macOS arm64) and writes <code>&lt;prefix&gt;/share/meep-gpu-build/build_record.json</code>:
the configure line, the Makefile flags, the compiler, what the native target
resolved to, the libraries' sha256 and the OpenMP runtime they link. It is a
timing control, never a certification reference, so no reference environment carries
one. Without <code>--meep-build native=...</code> the control is omitted by name ("no
--meep-build native=PYTHON[,LAUNCHER[,RECORD]] was given"), as it was in every ladder of
the 1.0 round on both hosts: the MEEP column of such a ladder is the portable reference
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
a GPU row or a MEEP rank (<code>environments/locks/README.md</code>). The round's child
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
round's tables were built.

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
| <code>ladder</code> | <code>ladder\|&lt;architecture&gt;\|&lt;case&gt;\|&lt;cells&gt;\|&lt;tier&gt;</code> | One case and size of the identical-case ladder with every leg it is compared against: each GPU route, the array-path control and every MEEP leg, all from one ladder session (R5) |
| <code>fused</code> | <code>fused\|&lt;architecture&gt;\|&lt;table&gt;\|&lt;case&gt;\|&lt;drive or res&lt;r&gt;&gt;\|p&lt;pass&gt;</code> | One <code>bench_fused_products.py</code> unit, its row's line kept verbatim for the cutter. The pass is in the key because a band edge of a preference record needs two rows |
| <code>check</code> | <code>check\|&lt;key&gt;</code> | A dispatch check of another key: its GPU legs at its case and size on short windows, kept for the launch counters |

The architecture is an NVIDIA compute capability (<code>8.6</code>) or an Apple GPU family
(<code>applegpu_g13s</code>), as the rows stamp it. A record carries its session (id,
start, end), the unit's command, the in-harness import record of every process (below),
the launch counters of every GPU leg (per slot: arm, kernel table, dispatches, and the
family and gate the plan cites), the reference environment's build record (R27) and the
MEEP build records, the host gates it passed (the quiet window read twice, the variance
precondition, the watcher's readings and intrusions, power recorded and never gated), the
command, environment and exit status this driver gave the unit, its legs' medians with
their gate status (R20), and a digest of its own body. The store keeps one file per key
and attempt (the session's id and end), written once. It lives under
<code>parity/meep_gpu/results/&lt;name&gt;/</code>, which the ladder's snapshot leaves out,
so a record cut from it names its rows <code>results/...</code> as the shipped records do;
the units' run root (<code>--run-root</code>, by default
<code>$MGPU_SITE_STAGING_ROOT/cert8</code>) stays outside every tree the ladder reads.

**The import record is the benches' own.** <code>bench_fused_products.py</code> stamps
every row with <code>imported_source_sha256</code> (the sha256 of every repository module
its process holds, keyed as every gate artifact keys it) and
<code>bench_meep_identical_case.py</code> stamps every MEEP measurement row the same way,
which shows the package was not in that process. Beside it a fused-product row records
what the 1.0 round's hook and its joiner kept with the record: the sha256 of the ledgers
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
and <code>cert8_catalog_apple.json</code> are the 1.0 round's. Site values (the MEEP build
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
timed in one process under one watcher (R5).

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
   resident process of another account on another GPU is recorded and not waited on
   (R18).
2. **The variance precondition**, once per session that times a ladder row: the catalog's
   point repeated five times on one GPU route and at one MEEP rank count. GO when the GPU
   route's repeat spread is within 5 %. MEEP's repeat spread is recorded and never gated.
   A session run with <code>--no-variance</code> records why, and none of its ratios
   enters a table.
3. **The units**: the ladder rows of one case and tier share one ladder (its passes
   interleave its sizes; a row alone is one size), then the fused DRIVE units, the sweep
   (odd passes ascending, even passes descending), then the dispatch checks. Every ladder
   unit is given <code>--require-ac-power 0</code>: no stage is gated on power.
4. **The watcher**, every 15 s while a unit runs: two readings in a row that see a
   foreign process on the device, another GPU process of the account, other work of the
   account, or a neighbour turned busy stop the unit; its output is set aside (renamed,
   never deleted, never collected) and it runs again once the window is quiet.
5. **Collect**: each unit's rows become records in the store, with the unit's exit
   status. A ladder unit that exited non-zero stopped before its plan finished: its
   records are kept and never cut.

<code>cert8.py collect ladder|fused</code> records a unit that was run by hand; such a
record carries no quiet window and no watcher, so it is kept for
<code>status</code> and <code>bill</code> and never cut.
<code>cert8.py selftest</code> exercises the watcher's rule on stand-ins.

**The A/B base** is the previous release, staged by <code>git archive &lt;tag&gt;</code>
and run with <code>--tree &lt;staged base&gt; --role ab</code>: its rows are kept beside
the release's and never enter a cut of the release.

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
   <code>driver_dispatch</code> record pins that no arm's entry pins. The 1.0 round's
   150 NVIDIA import records held the planners, <code>deposit_repair.py</code>,
   <code>host_writes.py</code>, <code>fused_pairs.py</code> and
   <code>fused_polarization_pair.py</code> in every row of both tables: there the re-time
   stays per table, because pooling rows timed on both sides of such a change would
   compare two programs.
2. **An arm's kernel file re-times exactly the keys whose launch counters show that arm
   dispatched.** The arm's ledger entry (the gate its plan cites) pins its own modules,
   and every kernel file they can import belongs to it too. An arm no ledger entry
   attributes (the 1.0 Triton single arms cite a recertification sweep) takes its
   table's whole kernel slice (27 of the 83 arms the 1.0 rows launched). On the Metal
   table the re-time stays per table: through what its pinned modules import, each of
   the 61 Metal arm entries that pins a kernel file reaches 90 of the 92 Metal kernel
   modules (the family modules register their arms at import), so a Metal kernel change
   re-times every Metal row that loaded it.
3. **A kernel file no launched arm carries owes the row a dispatch check**, because a
   predicate in it may admit another arm. The row is valid again once a
   <code>check</code> record of its key, cut wholly on the release tree, shows the same
   arms launched on the same slots.
4. **A ledger owes the row a dispatch check**: the four ledgers are what the dispatch
   path reads to admit arms, no arm's kernel text, so a change to one (a run recorded, an
   arm admitted) is checked as rule 3 is, never left unseen.

A row that records no launched slot (an <code>ERROR</code> row) is never narrowed: every
moved file re-times it. A row is refused by name, whatever the tree, when a process that
ran package code recorded no import record, no package module in it or no ledger
digests; when its unit was not opened on a quiet window of two clear readings or not
watched (a row collected by hand); when the watcher saw an intrusion; when a ladder
entry of it was written outside the session that records it (R5); when its ladder unit
exited non-zero; when it names no reference environment build record; when its rows
stamp another architecture than its key; or when its body does not match its digest.

**The environment is no file of the import record** (R27). A rebuilt reference
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
times exactly those. On the 1.0 round's 113
fused rows, a change to the CUDA complex fused electric pair's module re-times 5 and owes
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

- **The identical-case table** takes each key's newest valid row that carries a
  measurement (a later attempt that failed does not shadow an earlier timing). A GPU
  route or the array control carries the 5 % gate (every pass timed and unstruck by the bench's own
  floors, the between-pass spread within 5 %). A MEEP leg carries its between-pass spread,
  recorded and never gated, and is struck only for what is not spread (its geometry
  digest, its precision, its threads, its monitor). A ratio is the route's median over
  the best unstruck MEEP reference median of the same row and says whether it enters:
  it enters when the route passed the gate and the session's variance precondition read
  GO. Every median is printed with its gate status. The table is accounted against the
  catalog: every catalog key without a valid measured row, one never timed included, is
  listed with what it is owed.
- **The dispatch-preference records** are cut by <code>cut_timing_record.py</code> from
  the selected fused rows' own lines, one record per table **per architecture**, at the
  name <code>dispatch_preference.record_path</code> gives it:
  <code>&lt;table&gt;_kernels/timing_cc86.json</code>, and for the Metal table
  <code>metal_kernels/timing_applegpu_g13s.json</code>. A new architecture's record is
  added beside the others; a record of an architecture whose rows are absent is never
  touched; a record cut on another device is refused, with no override. The release is
  accounted for: a record cut from part of a campaign names every released (product,
  case) it lacks under <code>missing</code>, with the rows' own reason (the 1.0 round's
  eleven <code>ERROR</code> rows), instead of failing. Every catalog fused key of the
  table and architecture without a valid measured row (a sweep point never timed
  included) is named in the cut's output and in the report beside the record, and a
  table the catalog names that no valid row reaches is named as not cut (a table whose
  DRIVE cases are the release's and that has no sweep included).
- **The fused-product report** is <code>build_fused_timing_report.py</code> on the selected
  fused rows, staged as that builder reads them, with the catalog's keys it lacks, and
  its tables no valid row reaches, in <code>missing_keys.txt</code> beside it. The
  release's DRIVE cases are known only by running the bench's estimate, so a table that
  takes them is named there with a pointer to its record's <code>missing</code> block,
  which names every released (product, case) without a row.

<code>cut_timing_record.py --check</code> recomputes a record from the row files it names;
the store is written once per key and attempt, so every file a record names stays there.

## What a row licenses

One host, one GPU, the precision both sides ran, the case and size the row names,
the monitors it carried, the MEEP build and configuration it records, and the
gate rule stated beside it. A row licenses no statement about another case,
size, host, GPU, MEEP build or rank count. Rows taken with
<code>--allow-uncertified 1</code> are not certified and enter no certified
figure.

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
