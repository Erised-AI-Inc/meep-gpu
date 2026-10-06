# Compiled-kernel dispatch

The array path — the same FDTD implementation running through NumPy or CuPy — is
the numerical reference and the fallback for every configuration. Compiled
kernels are released per family and dispatch inside <code>driver.step()</code>, and
**dispatch is on by default**: a driver built with <code>prefer_gpu=True</code>
that sets nothing else gets every certified kernel its host can run, and every
other slot stays on the array path. That is the driver
<code>lift_simulation(sim)</code> and <code>run_on_gpu(sim, ...)</code> build
when nothing else is passed. A driver built with
<code>prefer_gpu=False</code> is the NumPy reference and never dispatches. This
page is the whole user-facing contract: who dispatches, how to turn dispatch off,
which kernels can serve you, how to read what you actually got, and what the
measurements do and do not support.

## <code>prefer_gpu</code> decides who dispatches

| Driver built with | Host | Engine arrays | <code>driver.gpu</code> | Kernels |
|---|---|---|---|---|
| <code>prefer_gpu=False</code> | any | NumPy | <code>None</code> | never: the NumPy reference, on the host CPU, whatever the environment says |
| <code>prefer_gpu=True</code> | CUDA device | CuPy | <code>"cuda"</code> | Triton, then hand-written CUDA, where released; the CuPy array path on the device elsewhere |
| <code>prefer_gpu=True</code> | Apple GPU | NumPy | <code>"metal"</code> | Metal where released; the array path **on the host CPU** elsewhere |
| <code>prefer_gpu=True</code> | neither | — | — | raises, naming <code>prefer_gpu=False</code> |

The two MEEP entry points, <code>lift_simulation</code> and
<code>run_on_gpu</code>, default to <code>prefer_gpu=True</code>. The engine
class <code>FdtdDriver</code>, built directly, defaults to <code>False</code>.
A host with both a CUDA device and an Apple GPU resolves CUDA.

**The environment can only turn kernels off.** <code>MEEP_GPU_DISPATCH=0</code>
and <code>MEEP_GPU_FUSED=0</code> put a <code>prefer_gpu=True</code> driver on
the array path. No value puts a <code>prefer_gpu=False</code> driver on kernels:
it does not read the kernel tables, imports no device library and installs no
subnormal policy. If <code>MEEP_GPU_DISPATCH</code> is set to anything but
<code>0</code> when a reference driver first steps, one line on stderr, once per
process, says that the variable does not apply and names
<code>prefer_gpu=True</code>.

**On an Apple GPU the array path is the host CPU.** A CUDA driver that falls
back stays on the device, through CuPy. An Apple GPU driver that falls back,
because no released arm covers its configuration or because
<code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code> restricts the Metal kernels to the
certified environment and this host is outside it, runs the whole step on the
CPU. That is announced on stderr
(<code>step path array on the host CPU; dispatch refused: ...</code>) and
recorded, not raised.

## Dispatch is on by default

For a driver built with <code>prefer_gpu=True</code>, nothing else needs to be
set. To force the array path instead, set one environment variable before the
run starts:

    MEEP_GPU_DISPATCH=0 python my_simulation.py

The switch accepts exactly two values: <code>1</code> enables dispatch,
<code>0</code> disables it, and unset takes the package default, which is enabled
(<code>fastpath.DISPATCH_BY_DEFAULT</code> is <code>True</code>). **Every other
value is refused by name**, and the run takes the array path, with the refusal
in <code>driver.fast_path_report()</code> and on stderr. That includes an empty
value, whitespace, <code>true</code>, <code>false</code>, <code>off</code>, and a
value with a stray space. The switch is strict because <code>false</code> is
exactly what someone types to turn a default off. Reading it as "on" would change
what a run measures without saying so.

Dispatch never widens what a run can do. A run reaches kernels only where a
released, certified family covers that exact configuration, on a supported
NVIDIA host (compute capability 7.0 to 9.0, Triton 3.1) or on any Apple GPU,
every one of which is supported; outside the certified identity the run is
labelled uncertified
([Certification on an NVIDIA GPU](#certification-on-an-nvidia-gpu),
[Certification on an Apple GPU](#certification-on-an-apple-gpu)). Every other
slot stays on the array path, which handles every configuration by construction.

## What the default covers, and what it costs

The default is on for the configurations the release evidence covers, and it is
worth knowing where that evidence stops.

- **Every NVIDIA GPU of compute capability 7.0 to 9.0 supported, one identity
  certified.** The Triton table is certified on Triton 3.1.0, and both NVIDIA
  tables on compute capability 8.6; the certification host is an NVIDIA RTX
  A6000. On another supported device or Triton 3.1 release the kernels run
  uncertified and say so on stderr and in the record, except that a table
  certified for the device and toolchain runs alone in place of one that is only
  supported. A device or toolchain outside the supported range is refused by name
  and the run takes the array path, unless
  <code>MEEP_GPU_ALLOW_UNCERTIFIED=1</code>. A device whose compute capability
  **cannot be read** is recorded as unknown and not refused, so such a host
  dispatches on an architecture nothing verified;
  <code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code> refuses it, and restricts the kernels
  to certified devices and toolchains everywhere
  ([Certification on an NVIDIA GPU](#certification-on-an-nvidia-gpu)).
- **Every Apple GPU supported, one environment certified.** The Metal kernels
  are meant to run on every Apple GPU, M1 through M5 and later, and run on each
  by default. They are certified on one environment: GPU architecture
  <code>applegpu_g13s</code> (Apple M1 Max), PyTorch 2.10.0, Metal frontend
  <code>metalfe-32023.850.10</code>, fast math off. On every other Apple GPU,
  PyTorch version and macOS build they run uncertified, and say so on stderr
  and in the record. <code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code>
  restricts them to the certified environment
  ([Certification on an Apple GPU](#certification-on-an-apple-gpu)).
- **Complex storage needs an evidence record the package does not ship.** The
  complex families refuse by name without
  <code>MEEP_GPU_COMPLEX_EXPANSION_PROBE</code>, so a default complex-field run
  takes the array path for those slots. Runs with a Bloch wavevector, a
  cylindrical <code>m</code> other than 0 or a complex β store complex fields.
- **A dispatched plan is not always the fastest one available.** On the NVIDIA
  tables fused dispatch beats the array path at the measured 2-D sizes, but it
  loses to the certified single-arm kernels below a crossover between 1.0 and
  2.4 million cells. Nothing yet arbitrates between the two
  ([What fusing buys](#what-fusing-buys-and-at-which-size)).
- **On an Apple host the default wins only on larger grids.** Small grids run
  faster built with <code>prefer_gpu=False</code>, or with
  <code>MEEP_GPU_DISPATCH=0</code>: a default lift measured slower than the
  reference below a crossover between 102,400 and 230,400 cells in 2-D and near
  125,000 cells in 3-D (2026-09-27, on a loaded host; not a timing of record).
  Under the default residency mode a script that writes into
  <code>driver.fields</code> arrays directly raises. See
  [the Metal section](#the-metal-table-and-held-residency) below.

A run that dispatches also pays for it:

- **Compilation at plan time.** Every dispatching kernel compiles at the first
  step after each configuration freeze. On the Apple GPU measured, a default
  lift spent 0.73 to 1.17 s in the lift, most of it importing the device
  library, and 0.49 to 0.66 s in the first step, against 0.003 to 0.27 s and
  under 0.02 s for a <code>prefer_gpu=False</code> lift.
- **One line on stderr.** The first dispatched step prints the step path, the
  slots served, the table and the policy. The reference prints nothing.
- **A process-wide subnormal policy.** The serving table's certified subnormal
  policy is installed for the whole process, including the sub-steps that stay
  on the array path. On an x86 CUDA host that means <code>keep</code> where the
  host would otherwise flush ([Subnormal policy](#subnormal-policy)).
- **No mid-run fallback.** A kernel that fails at plan time is refused by name
  and its slot stays on the array path. An error from a dispatched kernel after
  the configuration has frozen stops the run, rather than silently switching
  paths part-way through.

> **Note: Certification status of this release**
>
> Each kernel family ships with a ledger entry that binds its certification
> to the digests of the source files that were certified. The certification
> round ran on the files of 0.9.2, every entry is bound to them, and the record
> that backs the on-by-default setting was cut from a go/no-go on them. Two
> tests are pending: those of the fused-product timing records, which no timing
> campaign on these files has re-cut. The counts on this page describe the dated
> rounds they name. See
> [Running the certification harness](../development/certification.md).

## Which kernel table can serve you

There are three tables of compiled kernels: Triton, hand-written CUDA, and
Metal. You do not choose between them by preference — the hardware decides which
are candidates at all (<code>fastpath.candidate_tables</code>), and only for a
driver built with <code>prefer_gpu=True</code>:

| <code>prefer_gpu</code> | Engine the driver is on | Device present | Candidate tables, in order |
|---|---|---|---|
| <code>True</code> | CuPy | CUDA | Triton, then hand-written CUDA |
| <code>True</code> | NumPy | MPS (Apple GPU) | Metal |
| <code>False</code> | NumPy | any | none consulted; the NumPy reference |

**<code>is_available()</code> answers "does <code>prefer_gpu=True</code> resolve
here", not "will this run dispatch".** It is true on a host with CuPy and a
visible CUDA device and on a host with an MPS device, and
<code>available_gpu()</code> names which. Whether a given configuration
dispatches is decided at the first step, by the released arms and, on an NVIDIA
host, the certified identity. Read <code>driver.fast_path_report()</code> after
a step for the dispatch answer.

**On a CUDA host, which NVIDIA table serves depends on whether a validated
Triton is installed.** With Triton 3.1.0 installed, Triton composes first,
holding every slot both tables admit. The hand-written CUDA table fills Triton's
refusals and serves nothing else; its own board records
<code>by_default_precedence.served_in_dispatch: 0</code> for exactly this reason.
To put it first:

    MEEP_GPU_BACKEND_PREFERENCE=cuda python my_simulation.py

**Without a usable Triton, the hand-written CUDA table composes alone.** That
covers four kinds of host:

- an environment with CuPy and no Triton;
- any environment whose Triton is outside 3.1 (<code>&gt;=3.1,&lt;3.2</code>), which
  is refused by name unless <code>MEEP_GPU_ALLOW_UNCERTIFIED=1</code>;
- a device on which the hand-written CUDA table is certified and the Triton
  table, or its Triton version, is only supported: the certified table runs
  alone ([Certification on an NVIDIA GPU](#certification-on-an-nvidia-gpu));
- any host outside Linux x86_64, where Triton 3.1.0 does not install.

The Triton table is dropped by name, and the CUDA table plans by itself with the
array path filling its refusals. Composing alone, it serves mirror-folded runs
through its symmetry-fill kernels, which launch as a pair, and the off-diagonal
electric update beside its own curl.

**That composition has not been timed.** On one device it has run the
installation check and one example, with compiled kernels serving and results
identical to the array path. The route campaign's leg with Triton withheld from
the process ran on 2026-09-29: 19 of 39 cases byte-identical to the array path
with only CUDA kernels launched (18 complex cases skipped, 2 with no fused arm).
Every CUDA timing published so far was measured on a host with Triton
present, including the CUDA column of [Will it help?](will-it-help.md), where the
hand-written CUDA table served every dispatched sub-step of the timed case on the
CUDA-preferred route. Do not read a CUDA figure as the Triton-less one.

## Environment switches

These are the names that decide what a run dispatches and how it is recorded.

| Variable | Effect |
|---|---|
| <code>MEEP_GPU_DISPATCH</code> | The enable, read by drivers built with <code>prefer_gpu=True</code>. Unset takes the package default, which is enabled. <code>1</code> enables and <code>0</code> disables. Any other value, including an empty one, <code>true</code>, <code>false</code> or <code>off</code>, is refused by name and the run takes the array path. A <code>prefer_gpu=False</code> driver ignores it: no value makes the reference dispatch, and a value other than <code>0</code> prints one line per process saying so. |
| <code>MEEP_GPU_KERNEL_TABLE</code> | Names one table outright. It selects *within* the candidate set the hardware allows; a table this host cannot run is refused by name, never silently swapped for another. |
| <code>MEEP_GPU_BACKEND_PREFERENCE</code> | Orders the two tables a CuPy engine has. <code>cuda</code> puts the hand-written table first. A value naming a non-candidate is refused by name. |
| <code>MEEP_GPU_ALLOW_UNCERTIFIED</code> | Unset runs the kernels on a supported identity outside the certified one, labelled uncertified: on the NVIDIA tables a compute capability from 7.0 to 9.0 and a Triton 3.1 release, except that a table certified for the device and toolchain runs alone in place of one that is only supported; on the Metal table any environment. <code>0</code> restricts every table to certified identities: anything else, an identity that cannot be judged included, is refused by name and the run takes the array path (on an Apple GPU, the host CPU). <code>1</code> also runs an NVIDIA identity outside the supported range, recorded <code>supported: False</code>, and composes a supported, uncertified NVIDIA table beside a certified one; on the Metal table it behaves as unset ([Certification on an NVIDIA GPU](#certification-on-an-nvidia-gpu), [Certification on an Apple GPU](#certification-on-an-apple-gpu)). Any other value is refused by name and the run takes the array path, on every table. A run on an uncertified identity carries no certification: it prints one line per process naming what is certified, and its record carries <code>certified: False</code> with the identity read. |
| <code>MEEP_GPU_FUSED</code> | <code>0</code> forces the array path everywhere, regardless of everything else. <code>1</code> or unset leaves the decision to <code>MEEP_GPU_DISPATCH</code>. A veto only: no value of it can enable dispatch. Any other value, including an empty one, <code>false</code> or <code>off</code>, is refused by name and the run takes the array path. On a <code>prefer_gpu=True</code> driver it has the same effect as <code>MEEP_GPU_DISPATCH=0</code>; on a <code>prefer_gpu=False</code> driver it decides nothing, and the driver's record reports the value it was set to beside the reference reason. |
| <code>MEEP_GPU_FUSE_ARMS</code> | <code>0</code> admits no fused arm and leaves the rest of dispatch running — the way to run the separate sub-step kernels without the fused pairs, for bisecting a fusion difference or counting the launches fusion saves. It also accepts a comma-separated list of arm labels; the <code>0</code> veto wins wherever it appears. |
| <code>MEEP_GPU_DISPATCH_LOG</code> | A path. One JSON object is appended per configuration freeze, so a long run's dispatch state is readable with <code>tail</code> on the machine that owns the job. Unset keeps only the last record in memory, served by <code>driver.fast_path_report()</code>. |
| <code>MEEP_GPU_WARM</code> | <code>0</code> skips the plan-time warm pass. It changes *which* slots dispatch, not only when they compile: with the pass off, a slot whose kernel will not compile is no longer unfilled in advance, so the dispatch set is wider and a compile failure lands mid-step. A run made with it off is not the same run made with it on. |
| <code>MEEP_GPU_SUBNORMAL_POLICY</code> | Requests <code>flush</code>, <code>keep</code>, or <code>match_meep</code>. See below. |
| <code>MEEP_GPU_SUBNORMAL_INSTALL</code> | <code>0</code> stops dispatch *installing* the policy so a caller can own that decision. It does not relax the rule: with the install off, dispatch refuses unless the certified policy is already installed on all three executors. |
| <code>PYTORCH_MPS_FAST_MATH</code> | PyTorch's switch, not this package's; the package reads it and never sets it. <code>1</code> compiles the Metal kernels in fast-math mode, which no Metal certification ran. Unset or <code>0</code> is certified, and any other value is outside the certified environment: by default the run carries <code>certified: False</code> and the NOTE line, and under <code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code> it is refused by name. |
| <code>MEEP_GPU_METAL_RESIDENCY</code> | The Metal table's residency mode. Unset is <code>held</code> (also accepted by name): the device keeps the field volumes between launches. <code>shipped</code> selects the bracket that copies every device mirror in both directions around each launch; it is slower than the array path at every size measured and is kept for comparison only. Any other spelling is refused by name and the run takes the array path. See [the Metal section](#the-metal-table-and-held-residency) below. |
| <code>MEEP_GPU_COMPLEX_EXPANSION_PROBE</code> | A path to the complex families' evidence record. Four of the nine certified families refuse without it; the switch is named in the dispatch record so that refusal is visible rather than showing up as families silently missing from a coverage count. |
| <code>CUPY_CACHE_DIR</code> | Not a dispatch switch, but load-bearing under the <code>keep</code> policy: the directory must carry the token <code>ftz_stripped</code> in its path. CuPy computes its cache key above the seam that policy installs at, so a shared directory either gets poisoned with stripped binaries or silently serves flushed ones. Dispatch points the variable at such a directory itself and records the move; a caller that installs the policy by hand supplies one. |

## Reading what you actually got

Never infer the path from a timing. The driver reports it:

    driver = lift_simulation(sim)
    driver.run(until=200)

    print(driver.gpu)                   # "cuda", "metal", or None for the reference
    print(driver.active_step_path)      # "array" or "fused"
    report = driver.fast_path_report()  # this driver's own freeze

<code>active_step_path</code> is two-valued. <code>"fused"</code> means at least
one sub-step of the frozen configuration will launch a kernel. **Read it after a
step**: the plan is built at the configuration freeze inside <code>step()</code>,
so it reads <code>"array"</code> before the first step of a run that will
dispatch, and again between an invalidation and the next step.

<code>fast_path_report()</code> is the record to keep instead of reading the
source: it names the path, the per-slot outcome with its reason, and the
certification each dispatched family rides on. A refusal carries the single named
reason it was refused for. It is <code>None</code> before the first
<code>step()</code>, because no configuration has frozen yet, and it is always
this driver's own freeze rather than a process-wide last record.

A <code>prefer_gpu=False</code> driver always reads <code>"array"</code>, and
its report says why: <code>reference_driver</code> is <code>True</code>,
<code>table</code> is <code>None</code>, and <code>refused_because</code> is the
reference reason. <code>decision</code> reads <code>"refused"</code> there too,
so use <code>reference_driver</code> to tell the reference from a GPU driver
that was refused.

## Subnormal policy

Each kernel table declares the subnormal-arithmetic policy its certification was
cut under: the two CUDA tables are certified under <code>keep</code>, the Metal
table under <code>flush</code>. Dispatch installs that policy on all three
executors or refuses by name. This is user-visible because it affects
reproducibility against CPU MEEP, and the package exports the surface —
<code>FLUSH</code>, <code>KEEP</code>, <code>MATCH_MEEP</code>,
<code>install_subnormal_policy</code>, <code>get_subnormal_policy</code>,
<code>set_subnormal_policy</code>, <code>default_policy</code>,
<code>policy_stamp</code>, <code>resolve_match_meep</code>, and the refusals
<code>SubnormalPolicyLocked</code> and <code>SubnormalPolicyUnattainable</code>.

The default request is <code>match_meep</code>, which is answered by measuring
the host rather than by a hardcoded choice. What each policy does to each
executor is in
[the floating-point contract](../design/floating-point.md).

## Coverage: what fraction of the fusion surface dispatches

The unit is a **seam-instance**: one simulation of the corpus at one of the four
boundaries between consecutive sub-steps in the driver's timestep. The corpus is
the 194 simulations accepted from MEEP's own example and test scripts. Not every
simulation reaches every boundary, so the denominator is not 194 x 4: it is 194
at each of B->H, D->E and H->D, plus the 15 that carry a susceptibility at the
E->P chain seam, giving **597** seam-instances. Boards cut 2026-09-25:

| Kernel table | Reachable in dispatch, with the complex-storage evidence record | Reachable in dispatch without it (what a default install has) | Denominator |
|---|---:|---:|---:|
| Triton | 334 | 244 | 597 |
| Hand-written CUDA | 352 | 258 | 597 |
| Metal | 327 | 245 | 597 |

Read the qualifiers the boards state about themselves, in their own
<code>dispatch_agreement</code> block (the boards are not published with this
package):

- <code>is_an_upper_bound: true</code>. The number is what dispatch can reach,
  derived from the release tables and the parse tree, not a count of runs served.
- It is a **different number** from the board's own count of seam-instances a
  fused product exists for at all — 530 (Triton), 532 (CUDA), 531 (Metal) of the
  same 597. The gap between the two is the point of publishing them separately.
- The CUDA figure is conditioned on <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>
  with Triton present to fill that table's refusals. Under the shipped precedence
  that board's <code>by_default_precedence.served_in_dispatch</code> reads 0. It is
  **not** the figure for a host without Triton, where the table composes alone;
  no board publishes that figure yet.
- All three carry <code>agree: true</code>, meaning the board's two independent
  derivations of the number match.
- No table dispatches a product at the H->D seam (0 of 194), so 194 of the 597
  are out of reach on every table.

These are counts over MEEP's tutorial and test scripts. They are not a fraction
of MEEP's features, and they are not a speed. The Metal and hand-written CUDA
boards were re-cut on the files of 0.9.2 and read the same figures; the Triton
board is the cut of 2026-09-25.

The released arm tables behind those figures hold 30 arms over 64 arm-case rows
(Triton), 23 over 68 (hand-written CUDA), and 29 over 72 (Metal).

## Correctness under dispatch

Turning dispatch on does not change a result: kernel results are bit-identical
to the package's own array path on the same host, under the subnormal policy
the kernel table is certified for. That is identity with the array path, not
with MEEP. Two independent kinds of gate establish it, and neither is a
throughput claim:

- **Route gates** drive each fused product through the driver's own consults and
  byte-compare whole runs against the array path, with per-step launch counts
  proving the substitution actually happened.
- **Per-family device gates** certify the arithmetic of each family under both
  subnormal policies.

| Route campaign | Cases byte-identical, without the complex-storage evidence record (what ships) | With it | Date |
|---|---|---|---|
| Triton | 20 of 37; the other 17 are complex cases, skipped and not compared | 37 of 37 | 2026-09-29 |
| Hand-written CUDA | 19 of 39; 18 are complex cases, skipped and not compared, and 2 have no fused arm | 37 of 39; 2 have no fused arm | 2026-09-29 |
| Metal | 24 of 42; the other 18 are complex cases, skipped and not compared | 42 of 42 | 2026-09-29 |

All three ran on the code of 0.9.0, and these counts are as of 0.9.0
([Dispatch evidence of 0.9.0](../development/certification.md#dispatch-evidence-on-the-published-code-2026-09-29)).
On the files of 0.9.2 the route campaigns ran again, in the rounds stamped
2026-10-05, and released
(Triton 4 of 4 legs, hand-written CUDA its 5 required legs and the leg with Triton
withheld, Metal 5 of 5 legs)
([Certification status](../development/certification.md#certification-status-of-this-release)).

The largest route case is 110,592 cells. The timing ladders add a check at
size: 16 of 16 rows on each of the two NVIDIA ladders (2026-09-22 and
2026-09-26; two kernel tables at 8 sizes) and 7 of 7 on the Apple ladder were
bit-identical to the array path, up to 11,239,424 cells, over 58 to 4,000 steps
per row; so were 18 of 18 GPU rows of the comparison with MEEP in
[Will it help?](will-it-help.md) (two routes at three sizes, three passes).

Each gate's own record states what it covers: "the arms listed under
<code>arms_driven</code>, on the exact configurations listed under
<code>verdicts</code>, with the policy recorded in <code>provenance</code>," and
nothing about an arm it did not drive or a shape it did not run.

## What fusing buys, and at which size

Fusing replaces two or four certified single-arm kernels with one or two fused
products. **It is not a win at small grid sizes on the NVIDIA tables**, and the
numbers below say so. The speed of a dispatched run comes from the kernels;
fusing is reported as coverage.

The measurement was made on 2026-09-21 on one NVIDIA host: 109 route cases
attempted, 61 timed cleanly, 36 of those carrying a fusion number whose control
ran the same kernel table. Monitors are detached and the window is the step
loop.

**Against the certified single arms the fused products replace** — the
comparison that measures fusing — the median is **0.52x over 36 cases**, ahead on
**3 of 36**. Split by plan: a plan holding two fused pairs is ahead on **0 of
28**, spanning 0.40x to 0.72x; a plan holding one fused pair is ahead on **3 of
8**, spanning 0.48x to 1.09x.

**Against the array path** the fused step is faster on 61 of 61 cleanly timed
rows, from 1.53x to 8.41x — but this column is *not* the fusion's speedup: the
array path is the reference the arithmetic is proved against, not the baseline
the product replaces.

**Size is the reason.** 40 of the 44 distinct pairs of case and grid sit at or
below 25,000 cells and the largest is 110,592, the regime a fixed per-step host
cost dominates. A separate size sweep, moving only the grid, shows the two-pair
fused step flat at 0.80 to 0.97 ms per step from 24,000 to 1,014,000 cells and
crossing its single arms between **1,014,000 and 2,400,000 cells** on 4 of 4
sweeps, two on each NVIDIA table. At 2,400,000 cells it reads 1.027x on both
hand-written CUDA sweeps and 1.067x and 1.069x on the Triton sweeps. A one-pair
plan on a clean seam alternates around its single arms, ahead at 3 of 4 swept
sizes (1.055x, 1.048x, 0.975x and 1.001x at 24,000, 1,014,000, 2,400,000 and
5,400,000 cells). These figures were measured on an earlier deposit-repair route
and have not been re-measured on 0.9.2 (CHANGELOG, 0.9.2, "Pending:
fused-product timing").

Where the fixed cost comes from: a fused pair that owns the seam a source is
injected on is bracketed by a repair that saves the affected cells before the
launch and recomputes them after the injection. The cost follows the source's
seam, and neither kernel is slow.

None of these rows supports a corpus-wide claim, an end-to-end or application
speedup, or a GPU-against-CPU claim. For the figures that do support a
user-facing comparison, see [Will it help?](will-it-help.md).

## Certification on an NVIDIA GPU

**Supported is not certified.** Both NVIDIA kernel tables support compute
capability 7.0 to 9.0, and the Triton table supports Triton 3.1
(<code>&gt;=3.1,&lt;3.2</code>). The floor is the one Triton 3.1 documents; the ceiling
is the newest architecture the compilers target: the ptxas Triton 3.1 bundles,
and the NVRTC that compiles the hand-written CUDA kernels, which is CUDA 11.8 for
the cupy-cuda11x build (the certified stack) and the host's CUDA 12 for the
cupy-cuda12x build, for which 9.0 is a conservative ceiling. The ranges are read
from the compilers' documentation and code (<code>fastpath.SUPPORTED_COMPUTE_CAPABILITIES</code>,
<code>fastpath_cuda.SUPPORTED_COMPUTE_CAPABILITIES</code>,
<code>fastpath.TRITON_SUPPORTED_VERSIONS</code>); no NVIDIA device outside the
certified list has been run. A device or Triton version is **certified** when the
table's certification records hold a live run on it: today compute capability
8.6 and Triton 3.1.0 (one RTX A6000).

What each value of <code>MEEP_GPU_ALLOW_UNCERTIFIED</code> does with each kind of
identity:

| Value | Certified | Supported, not certified | Outside the supported range (read) | Could not be read |
|---|---|---|---|---|
| unset | runs, <code>certified: True</code> | runs, <code>certified: False</code>, one NOTE line; refused by name when another candidate table is certified for this device and toolchain | refused by name; the refusal names <code>1</code> | runs, <code>certified: None</code> |
| <code>1</code> | runs | runs, beside a certified table too | runs, recorded <code>supported: False</code>, one NOTE line | runs, <code>certified: None</code> |
| <code>0</code> | runs | refused by name | refused by name | refused by name |

**A certified table outranks a supported one.** Where one NVIDIA table is
certified for the device and toolchain and the other is only supported, the
default runs the certified table alone, and the run is certified; the other is
refused by name, for example

    the triton table is supported but not certified here (GPU compute capability 9.0), and the cuda table is certified for this device and toolchain; set MEEP_GPU_ALLOW_UNCERTIFIED=1 to compose it too

so the default never makes such a host less certified than the certified table
alone. Such a host may serve fewer slots than the two tables merged would.
<code>MEEP_GPU_BACKEND_PREFERENCE</code> naming the refused table is refused by
name, with that reason appended, unless <code>MEEP_GPU_ALLOW_UNCERTIFIED=1</code>.
<code>MEEP_GPU_KERNEL_TABLE</code> naming one table leaves the other unconsulted,
so the named table runs, uncertified if it is not certified.

**What an uncertified run says.** The step-path line marks the device
<code>supported, UNCERTIFIED</code> (and a Triton 3.1 release other than 3.1.0
<code>triton 3.1.1 supported, UNCERTIFIED</code>); the device mark is that of the
tables that served. One line on stderr, once per process, names what is not
certified:

    meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU compute capability 8.9 (certified: 8.6). They were dispatched because this NVIDIA GPU and toolchain are supported but not certified bit-identical (the supported range is compute capability 7.0 to 9.0, with Triton >=3.1,<3.2 or with the NVRTC of either CuPy build, CUDA 11.8 for cupy-cuda11x or the host's CUDA 12 for cupy-cuda12x), and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the kernels to certified ones; the gates with no run on this GPU are counted in the dispatch record under uncertified.served[], which names the first five in welds_without_a_live_run_here; compare the results with a prefer_gpu=False run of the same simulation before relying on them

The reason names what was judged: <code>this NVIDIA GPU and toolchain are</code>
where the GPU and the Triton version beside it are both supported,
<code>this NVIDIA GPU is</code> where no supported toolchain was judged with it (the
hand-CUDA table reads no toolchain version), and <code>this Triton version is</code>
where only the Triton version is uncertified. Under <code>1</code>, an identity
outside the supported range gives as the reason
<code>MEEP_GPU_ALLOW_UNCERTIFIED=1, which also runs this NVIDIA GPU outside the
supported range</code> (or <code>this Triton version</code>). A refusal under <code>0</code> ends
<code>MEEP_GPU_ALLOW_UNCERTIFIED=0 restricts the NVIDIA kernels to certified
devices and toolchains</code>.

**The record carries the verdicts.** The NVIDIA half of
<code>fast_path_report()["environment"]</code> holds
<code>device_certified_by_table</code> and <code>device_supported_by_table</code>
(per table, <code>True</code>, <code>False</code> or <code>None</code> when the
compute capability was not read), <code>device_certified</code> and
<code>device_supported</code> (the Triton table's answers),
<code>triton_certified</code> and <code>triton_supported</code>,
<code>supported_ranges</code> and <code>certified_only</code>. The
<code>uncertified</code> block holds <code>allowed</code> (a supported, uncertified
identity may run: unset and <code>1</code>), <code>unsupported_allowed</code>
(<code>1</code>), <code>admitted</code> and <code>served</code> (each identity
admitted uncertified: table, what was read, what is certified,
<code>supported</code>, <code>because</code> and, for a compute capability,
<code>welds_without_a_live_run_here</code>), and <code>dropped</code> (an identity
admitted and then not composed, with <code>dropped_because</code>). A table
refused because a certified one outranks it carries
<code>dropped_for_a_certified_table</code> under <code>tables</code>.

## Certification on an Apple GPU

**Supported is not certified.** A GPU is **supported** when the kernels are
meant to run on it and run there by default: every Apple GPU is supported by the
Metal table, M1 through M5 and later. The package reads it off the GPU
architecture Metal names (<code>applegpu_*</code>) or, where the architecture
cannot be read (before macOS 14), off a device name that begins with the word
<code>Apple</code>. A GPU is **certified** when the gate fleet measured the
kernels on it: one Metal environment is certified, and an Apple GPU no
certification ran on is supported and uncertified. A GPU that is not Apple's is
not supported; the Metal table still runs on it by default, uncertified. On
NVIDIA hardware the supported range is compute capability 7.0 to 9.0
([Certification on an NVIDIA GPU](#certification-on-an-nvidia-gpu)).

The Metal table's certification names an **environment** of four facts, each of
which changes the code that runs:

| Fact | Certified value | What it is |
|---|---|---|
| GPU architecture | <code>applegpu_g13s</code> (Apple M1 Max) | <code>MTLDevice.architecture.name</code>, read through the Objective-C runtime; it needs macOS 14 or later. It separates GPU generations that share a Metal GPU family: M3 and M4 are both Apple9 |
| PyTorch | 2.10.0 | Supplies <code>torch.mps.compile_shader</code>, which compiles every Metal kernel |
| Metal frontend | <code>metalfe-32023.850.10</code> | The compiler that turns the Metal source into GPU code. One per macOS build, the same on every Mac on that build whatever its GPU |
| <code>PYTORCH_MPS_FAST_MATH</code> | unset or <code>0</code> | PyTorch's fast-math switch for every Metal source it compiles |

Each fact reads one of three ways. It is **certified** when every Metal
certification record the table cites (45 of them) recorded that value; **not
certified** when a cited record recorded another; and **not judged** when the
fact could not be read on this host or the cited records do not name it. Fast
math is certified only when unset or <code>0</code>, because no certification
ran with it: measured 2026-10-02 on an Apple M1 Max (macOS 26.2, PyTorch 2.10.0),
<code>PYTORCH_MPS_FAST_MATH=1</code> reaches <code>torch.mps.compile_shader</code>
and changed 274,523 of 1,048,576 float32 divisions and 327,338 of 1,048,576
square roots against an unset run; <code>0</code> changed none. Each
certification record holds one run, so one GPU architecture is certified at a
time.

**By default the Metal kernels run on every supported GPU, in any environment,
certified or not.** Another Apple GPU (M2, M3, M4, M5, or another variant),
another PyTorch version or another macOS build runs the same kernels, labelled
uncertified. The dispatch record reads <code>certified: False</code> and lists
each fact that is not certified under <code>uncertified["served"]</code>, with
the value read, the certified value and <code>because</code>, why it ran. One
line on stderr, once per process, names them:

    meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU architecture applegpu_g15p (certified: applegpu_g13s). They were dispatched because this Apple GPU is supported, and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the Metal kernels to certified environments; compare the results with a prefer_gpu=False run of the same simulation before relying on them

The reason is about the GPU, whichever fact is not certified: an uncertified
PyTorch on the certified M1 Max gives the same <code>this Apple GPU is
supported</code>. On a GPU that is not Apple's, and on one whose architecture and
name could not be read, the reason is instead <code>the Metal table runs on
environments outside the certified set unless MEEP_GPU_ALLOW_UNCERTIFIED=0</code>.

A fact that is not judged runs too: when no fact is uncertified, the record
reads <code>certified: None</code> and no NOTE line is printed. Compare an
uncertified run with a <code>prefer_gpu=False</code> run of the same simulation
before relying on it.

**<code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code> restricts the Metal kernels to
certified environments.** Every fact must then be certified. One that is not
certified, or that is not judged, is refused by name, and the whole run takes the
array path on the host CPU. The refusal names the fact, for example:

    GPU architecture applegpu_g15p is not the one every Metal weld this table cites ran on (certified: ['applegpu_g13s']); MEEP_GPU_ALLOW_UNCERTIFIED=0 restricts the Metal kernels to certified environments
    PYTORCH_MPS_FAST_MATH=1 compiles the Metal kernels in fast-math mode, which no Metal weld ran; MEEP_GPU_ALLOW_UNCERTIFIED=0 restricts the Metal kernels to certified environments

A refusal of a fact that cannot be judged ends "..., and an environment that
cannot be judged is not one". <code>1</code> behaves as unset on the Metal
table.

**The step-path line names the GPU.** On a Mac it ends with PyTorch, the Metal
frontend and the GPU, each marked <code>certified</code>,
<code>UNCERTIFIED</code>, or <code>uncertified-unknown</code> for a fact not
judged:

    ...; torch 2.10.0 certified, metal frontend metalfe-32023.850.10 certified; Apple M1 Max (applegpu_g13s) certified

A supported GPU that is not certified carries <code>supported</code> before its
mark:

    ...; Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED

The certified GPU reads <code>certified</code> alone, even when PyTorch or the
frontend beside it is marked <code>UNCERTIFIED</code>. An Apple GPU whose
architecture cannot be read is named without one and marked
<code>supported, uncertified-unknown</code>; with no name read either, the last
part reads <code>device</code>, the reason, and <code>uncertified-unknown</code>.
A GPU that is not Apple's is never marked <code>supported</code>. Fast math is not
on the line; when it is set to a value that is not certified, the NOTE line
names it.

**The record carries the environment.** The Metal half of
<code>fast_path_report()["environment"]</code> holds <code>device</code> (its
<code>name</code> and <code>architecture</code>, and the architecture again as
<code>compute_capability</code>, the key each dispatched arm's certification quote
reads to name the run its weld recorded on this GPU), <code>device_supported</code>
(<code>True</code> for an Apple GPU, <code>False</code> for a GPU that is not
Apple's, <code>None</code> when neither the architecture nor the name was read),
<code>device_certified</code>
(the architecture's verdict, in the key the NVIDIA tables use for the compute
capability), <code>torch_certified</code>, <code>frontend_certified</code>,
<code>fast_math</code>, <code>fast_math_certified</code>,
<code>recorded_environments</code> (one row per distinct GPU architecture,
PyTorch and Metal frontend the cited welds' live run records name, with how many
welds have such a run) and
<code>certified_only</code> (<code>True</code> under
<code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code>). The three
<code>*_certified</code> verdicts for the GPU, PyTorch and the frontend read
<code>True</code>, <code>False</code>, or <code>None</code> for not judged.

## The Metal table and held residency

Because dispatch is on by default, a <code>prefer_gpu=True</code> run on an Apple
GPU takes the Metal table unless it says otherwise: every Apple GPU is
supported, in any environment
([Certification on an Apple GPU](#certification-on-an-apple-gpu)). That
table's default residency mode is <code>held</code>: the device keeps the
field volumes between launches, and the host reaches them only through the
package's own paths — a read barrier on the field attributes, and sparse
transport for the array-path passes that touch a held volume (source injection,
the metallic wall clears, the symmetry and far-ghost fills, the trailing repair).
This default is certified on the files of 0.9.2 on the Apple M1 Max: the Metal
device gates, the Metal route campaign with residency held, the re-cut of the
Metal dispatch record and the coverage board ran on them.

Which path is faster depends on the grid, so choose deliberately:

- **Large 3-D grids: keep the default.** On one Apple M1 Max, the 3-D PML case
  at 7,077,888 cells with one flux monitor attached reads 1,141.9 Mcell-steps/s
  under held dispatch against 34.3 on the array path, bit-identical (measured
  2026-09-25 with <code>held</code> set by name, before it became the default).
  No comparison with MEEP on the identical problem has been made on Apple
  hardware ([Will it help?](will-it-help.md)).
- **Small grids: lift with <code>prefer_gpu=False</code>, or set
  <code>MEEP_GPU_DISPATCH=0</code>.** Both step the same array path on the
  host CPU. At 3,600 cells, Ez after 200 steps was bit-identical between the
  default lift, the reference and both vetoes in 11 of 11 real-field cases. A
  default lift, timed through <code>lift_simulation</code> on the same laptop
  (2026-09-27; vacuum, an absorber on every side, one point source, real
  fields), steps at about 2.1 to 2.9 ms per step from 3,600 to 230,400 cells
  whatever the size. About three quarters of that (73 % in the one case
  profiled) is copies between the host and the device, which continue after
  the source has ended. Against the reference that is 7.8 to 10.3 times
  slower at 3,600 cells, 2.6 to 3.0 times slower at 40,000, and 1.16 to 1.45
  times slower at 102,400; it is 1.41 to 1.66 times faster at 230,400 cells and
  2.05 to 2.66 times faster at 409,600. In 3-D it is 1.62 times slower at
  64,000 cells, level at 125,000 and 1.61 times faster at 216,000. **The
  crossover is between 102,400 and 230,400 cells in 2-D and near 125,000 cells
  in 3-D.** The host was loaded during that measurement, so these are not
  timings of record; the 2-D bracket held on both of two passes.

<code>MEEP_GPU_METAL_RESIDENCY=shipped</code> selects the per-launch bracket,
which copies every device mirror in both directions around every launch so that
the host stays authoritative between launches. It is kept for comparison only.
It is slower than the array path at every size measured. To opt out of held
dispatch, build the driver with <code>prefer_gpu=False</code> or set
<code>MEEP_GPU_DISPATCH=0</code>, not <code>shipped</code>.

**A configuration a hold cannot serve takes the array path, by name.** Before
anything is installed, the table checks three things: that every array-path pass
it leaves beside its kernels reaches a held volume through one of the sparse
write paths, that every engine-owned array can be sealed, and that no
polarization state still carries a guard from an earlier hold. A configuration
that fails any of them is refused to the array path, never to
<code>shipped</code>, and <code>fast_path_report()</code> names the reason.
Replayed over the records of the Metal route campaign of 2026-09-25, 0 of its 42
cases would be refused this way.

**The driver's own mutators work mid-run.** <code>set_field</code>,
<code>set_epsilon</code> and its component-resolved and smoothed forms,
<code>add_source</code>, the monitor constructors, <code>reset()</code> and
<code>invalidate_fast_path()</code> were each measured identical to the array
path when called under a hold, and the run is held again afterwards: a call that
changes the configuration releases the hold and hands the host its arrays back
before the next step freezes a new plan. Mutators the driver refuses after the
first step are refused the same way on both paths.

> **Warning: Direct access to driver.fields under held residency**
>
> The arrays on <code>driver.fields</code> are the engine's own storage, not
> part of MEEP's interface (MEEP's <code>sim.fields</code> exposes no such
> arrays). Under a hold they are sealed while the device owns them, so two
> uses are unsupported:
>
> - **An in-place write**, such as <code>driver.fields.Dz[...] += x</code> in
>   a step function or between runs, raises
>   <code>ValueError: assignment destination is read-only</code> at that
>   statement.
> - **A reference kept across a step**, such as an array taken from
>   <code>driver.fields</code> before a run and read after it, reads stale
>   values, and nothing raises.
>
> Read fields through <code>get_field</code> or <code>get_field_point</code>,
> which return current values. Write a primary field with
> <code>set_field</code>. Code that must write an engine array in place calls
> <code>meep_gpu.host_writes.acquire(array)</code> immediately before the
> write, in the same step; an array acquired at an earlier step is sealed
> again by the next launch. Code that cannot follow these rules runs with
> <code>prefer_gpu=False</code> or <code>MEEP_GPU_DISPATCH=0</code>.

One gap remains open. If the enable, the kill switch or the table preference is
changed part-way through a process and the driver is then re-frozen, the package
refuses the new plan before it reaches the Metal table, so the previous hold is
never released and the first array-path write to one of its arrays raises. Keep
those switches fixed for the life of a process.

The Metal board's coverage figure (327 of 597) is a correctness and reachability
number, and the table's route gate supports no timing claim.

## Reproducing a run

Two runs of the same script on the same host can execute different kernels, so
the dispatch state belongs in a run's evidence trail alongside the MEEP version
and the preflight verdict: what <code>prefer_gpu</code> resolved to
(<code>driver.gpu</code>), the enable's value (unset, <code>1</code> or
<code>0</code>) and what it resolved to, which table served, which arms, and the
subnormal policy that was installed. The last four are in
<code>fast_path_report()</code>, and <code>MEEP_GPU_DISPATCH_LOG</code> persists
them per configuration freeze. On an Apple host, record the residency mode as
well: the <code>residency.mode</code> field of <code>fast_path_report()</code>,
since an unset <code>MEEP_GPU_METAL_RESIDENCY</code> means <code>held</code>.
Keep the record's <code>certified</code> key and <code>environment</code> block
with it: the same script runs uncertified on a Mac outside the certified
environment.
