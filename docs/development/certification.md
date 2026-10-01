# Running the certification harness

A compiled kernel reaches a user's run only after it has been shown, on a
device, to return the same bits as the package's array path. The harness under
<code>parity/meep_gpu/</code> is what shows it. This page says how the harness is
run and what it leaves behind. It needs the hardware of the kernel table being
certified: a CUDA device for the two NVIDIA tables, an Apple GPU for the Metal
table.

Using the package needs none of this. A user who only runs simulations does not
run the harness.

## Certification status of this release

| Item | State |
|---|---|
| The kernel ledgers shipped in the package | Cut on the development tree, on the dates each entry records. Their source digests do not match the published files, because the published files were edited after the ledgers were cut |
| The go/no-go for dispatch on by default, and the route campaigns of the three tables | Run on the published code on 2026-09-29, and released. See [Dispatch evidence on the published code](#dispatch-evidence-on-the-published-code-2026-09-29) |
| The record behind the on-by-default dispatch setting | Not yet cut. The go/no-go it is cut from has run |
| The record behind the Metal table's default residency mode | Not yet cut |
| The coverage boards | Cut 2026-09-25; a re-cut on the published files is pending |
| The certification round on the published files | Pending. It re-runs the device gates, the route campaigns and the boards on the published files, and re-cuts the ledgers from what it measured |
| The ledger-contract tests | The certification suite (pytest marker <code>certification</code>), not run by default and run with <code>-m certification</code>. It fails until that round has run. See [The certification suite is not run by default](testing.md#the-certification-suite-is-not-run-by-default) |

Until the round has run, every certification count in this manual describes the
dated round it names, on the files of that date.

### Dispatch evidence on the published code (2026-09-29)

The default-dispatch go/no-go (<code>gate_dispatch_end_to_end.py</code>) and the
route campaigns of the three kernel tables ran on the code of this release.
Every leg recorded the sha256 of the source files it binds (12 to 39 per leg),
and every route leg also that of each package and harness module it imported (74
to 155 per leg). All of them equal the published files except one: the three
Metal legs that imported <code>parity/meep_gpu/probe_metal_dispatch_dryrun.py</code>
recorded it before one of its progress messages was rewritten to parse under
Python 3.10, with the same output text. <code>meep_gpu/fastpath.py</code> reads <code>1b22b59c…</code>,
<code>meep_gpu/driver.py</code> <code>35a05893…</code> and
<code>meep_gpu/metal_dispatch.py</code> <code>1d136843…</code>. The tree that
ran differs from the published one in its Markdown documentation, that harness
probe, the two digest helpers the certification tests use
(<code>meep_gpu/code_identity.py</code> and <code>meep_gpu/device_identity.py</code>,
which no leg imported) and test modules.

| Run | Host | Result |
|---|---|---|
| Go/no-go: the ship leg and its null control | One RTX A6000, compute capability 8.6, Triton 3.1.0, CuPy 13.5.1 | **GO.** The dispatch switch unset, no other <code>MEEP_GPU_*</code> variable than the dispatch log's path, no subnormal policy installed by the harness. 8 of 9 whole simulations stepped on compiled kernels, all on the Triton table; 1 of 9 (<code>bloch_2d</code>) took the array path with its reason recorded. 62 of 62 checkpoints byte-identical to the same simulation on the array path. Null control, the kill switch on both legs: 9 of 9 took the array path and launched no kernel |
| Triton route campaign | The same | 4 of 4 legs released. As shipped: 20 of 37 cases byte-identical through fused kernels, the other 17 complex. With the complex-storage evidence record: 37 of 37 |
| Hand-written CUDA route campaign | The same | 6 of 6 legs released. As shipped, CUDA first: 19 of 39 (18 complex, 2 with no fused arm); with the record: 37 of 39. Under the default precedence the Triton table served 20 of 20 dispatching cases and the CUDA table none, as designed. With Triton withheld from the process: 19 of 39, CUDA kernels only, the first route run of that composition |
| Metal route campaign | One M1 Max, PyTorch 2.10.0, Metal frontend 32023.850.10 | 5 of 5 legs released, residency held (the default). As shipped: 24 of 42, the other 18 complex. With the complex-storage probes: 42 of 42. 0 hold refusals |

No leg had a divergent checkpoint, and every control held. The legs that use the
complex-storage evidence read records measured on earlier bytes: the NVIDIA
expansion records of 2026-09-23 and 2026-09-25, and the Metal probes of
2026-09-25.

Still owed, in the next certification round (0.9.1):

- cutting the record behind the on-by-default setting, and the tables' dispatch
  records, from runs on the published files;
- re-running the per-family device gates and re-cutting the ledgers and the
  boards, which turns the ledger-contract tests green;
- cutting the record behind the Metal table's default residency mode;
- re-measuring the complex-storage evidence records on the published files;
- timing the hand-written CUDA table composing alone, and the timing campaign
  from the release tag.

## What is certified, and against what

| Question | Instrument | Compared with |
|---|---|---|
| Does one kernel family reproduce one sub-step? | A per-family device gate | The array path's own sub-step, on real grid, field and absorber objects installed through the public installer |
| Does a fused product reproduce whole runs through the driver? | A route gate | The array path over the whole run: checkpoints, final state and monitor results, with launch counts proving the kernels ran |
| Do certified neighbours still reproduce beside a new family? | A composition probe | The array path over whole steps |
| Does every executor obey the subnormal policy? | The subnormal conformance gate | The requested policy, operation by operation |

Every gate carries its own null controls: a planted defect that the gate must
catch, and mutations that must go uncaught because they change nothing. A gate
that has never been shown to fail is not evidence.

A kernel table is certified on a named toolchain and device identity, and under
a named subnormal policy. The identity is recorded with the verdict, and
dispatch refuses by name on any other.

## The harness, by script family

The prefix of a script says what kind of question it answers and what kind of
record it leaves.

| Prefix | Question | Leaves |
|---|---|---|
| <code>gate_*</code> | Does this kernel reproduce the array path bit for bit, on this device, for this sub-step or this route? | A JSON artifact with a machine-readable verdict |
| <code>probe_*</code> | A question a single-sub-step gate cannot answer: composition over a whole step, a reconnaissance measurement, or an input another gate consumes | A JSON artifact; most probes release nothing |
| <code>replay_*</code> | Arithmetic over an existing record, such as how many corpus simulations a predicate admits | A projection; never a device measurement |
| <code>sweep_*</code>, <code>survey_*</code> | The MEEP corpus, one script at a time: accepted, lifted, and compared with MEEP | One result per case, written as it lands |
| <code>bench_*</code> | Throughput | Timing rows with a verdict per row |
| <code>build_*_fusion_matrix.py</code> | Which seam-instances of the corpus a table's products serve and dispatch | A coverage board |
| <code>rebind_*</code>, <code>record_*</code>, <code>recut_*</code> | Bind a ledger to the artifacts a campaign left | A ledger entry, derived from the artifacts |
| <code>recut_metal_gates.sh</code>, <code>run_metal_dispatch_campaign.sh</code> | Drive every Metal gate, or the Metal route campaign | The artifacts of the scripts they drive |

Three distinctions the corpus scripts keep apart: a simulation is **accepted**
when the preflight returns supported; it **lifts** when a driver was actually
built, which is stricter; and it is **compared** when both MEEP and the package
stepped it and the fields were compared. They are three numbers.

## Where records go

The harness writes under <code>parity/meep_gpu/results/</code>. That directory
is not part of the repository: records are measurements of one host, and they
are archived separately from the code.

- **Name a record** <code>&lt;subject&gt;_&lt;YYYY-MM-DD&gt;[_&lt;qualifier&gt;]</code>:
  subject first, ISO date second, and a qualifier that names what was varied.
- **A re-run goes into a fresh directory.** Each artifact directory carries a
  manifest of source digests, and a gate refuses to release into a directory
  whose manifest disagrees with the files it imported. A second attempt is
  spelled in the name, never by overwriting.
- **A record is read, never edited.** A number belongs in its record; a document
  points at the record.
- **Record the host.** A campaign directory states what ran, when, on which host
  and device, and on which source digests.

**Your own records do not make this directory the evidence archive.** The tests
listed under <code>evidence_archive</code> in
<code>tools/ci/declared_resources.txt</code> run only when
<code>parity/meep_gpu/results/</code> carries the archive's marker file,
<code>EVIDENCE_ARCHIVE.json</code>
([The evidence archive](testing.md#the-evidence-archive)). No gate, probe or
benchmark writes the marker, so after a gate run in a checkout those tests still
skip, and the skip reason says that the directory carries no marker. A few other
tests read a record under a resource of their own, not
<code>evidence_archive</code>; they do not consult the marker, and they run
wherever their record is present. Do not write the marker into a directory that
holds only your own records: the marker states that the archived records are
there, and the listed tests that read them would run and fail.

## The ledgers

Seven JSON ledgers ship inside the package and are read at run time:

| Ledger | Holds |
|---|---|
| <code>meep_gpu/triton_kernels/fingerprints.json</code> | The Triton table's certification entries and its dispatch record |
| <code>meep_gpu/cuda_kernels/fingerprints.json</code> | The hand-written CUDA table's certification entries |
| <code>meep_gpu/cuda_kernels/certification.json</code> | The hand-written CUDA table's campaign records, including the device source strings |
| <code>meep_gpu/cuda_kernels/own_cell_hoist_reference.json</code> | The reference record of one hand-written CUDA kernel rewrite |
| <code>meep_gpu/metal_kernels/fingerprints.json</code> | The Metal table's certification entries and its dispatch record |
| <code>meep_gpu/triton_kernels/timing_cc86.json</code>, <code>meep_gpu/cuda_kernels/timing_cc86.json</code> | Timing records cut from timing rows, one per compute capability |

Each certification entry binds a gate's verdict to the sha256 digests of the
source files that produced it, and keeps **one record per GPU architecture** it has
been run on, under its <code>runs</code> key:

```json
"triton_complex_no_pml_curl_device_gate": {
  "source_sha256": {"...": "the bytes this gate certified"},
  "status": "PASS",
  "runs": {
    "8.6": {
      "host": "... NVIDIA RTX A6000, cc 8.6; CuPy 13.5.1; Triton 3.1.0 ...",
      "records": "apps/api/parity/meep_gpu/results/triton_fleet_.../complex_no_pml_curl/",
      "recorded_utc": "2026-09-25T16:09:24Z",
      "bound_sha256": "the digest of the digests above, at the time of that run"
    }
  }
}
```

The <code>records</code> paths in the shipped ledgers begin <code>apps/api/</code>: that
is the layout of the tree the records were cut in, and it is quoted as it was
recorded rather than rewritten, since every run record carries it. In this repository
the same directory is <code>parity/meep_gpu/results/</code>, which is not committed, so
a clone holds the path but not the run behind it.

<code>bound_sha256</code> is what makes an architecture's record expire on its own: a
record is LIVE only while it equals the digest of the entry's own pinned bytes, so an
edit to a certified file retires every architecture's evidence for that entry at once,
and no reader has to be told to re-check. A table's
<code>validated_compute_capabilities</code> is DERIVED from these records -- the
intersection, over the entries that table's arms cite, of each entry's live
architectures -- and is not written anywhere by hand.

Two rules follow:

1. **A verdict that has drifted is re-earned by re-running the gate, not by
   editing the ledger.** Editing a certified file puts every entry that pins it
   in drift, and the price of the edit is a device run for each of those gates.
   A file that many gates import, such as a shared launcher, is pinned by most
   entries of its table.
2. **A ledger is written by a tool, from artifacts.** The tools refuse to write
   an entry the artifacts do not support: a campaign directory the record does
   not name, a gate that refused on the device, or a curated measurement that no
   longer describes the tree.

The ledger-contract tests enforce both. They recompute every pinned digest
against the tree, wherever in the ledger it lives, and they assert their own
denominators, so coverage that shrinks fails too. They run on a host with no
GPU.

A green ledger does not by itself mean a coverage board can be cut. A ledger
checks the files each entry names; a board checks every file the gate process
imported, which is the larger set.

## Certifying another compute capability

Everything the package admits is read from the records above, so a second GPU
architecture is certified by writing records, not by editing code. Three properties make
that safe, and each is a test on a host with no GPU
(<code>meep_gpu/test_capability_records.py</code>):

* a run on an architecture **joins** the ones already recorded, as long as the bytes it
  certified have not moved;
* a run on bytes that **have** moved refuses by name rather than silently leaving the
  other architectures' records pointing at code that no longer ships. Re-run those
  architectures, or supersede them explicitly with <code>--supersede</code>;
* a table claims an architecture only when **every** entry its arms cite has a live
  record for it. A partial round admits nothing, and the refusal names the entries that
  are short.

The steps below are in order because the route and licence legs dispatch, and so need
the architecture already admitted by the records steps 1–2 write. The family gates in
step 1 do not: see "Which architecture binds first" for what is actually ordered and
what only looks ordered.

| # | Step | Where | Writes |
|---|---|---|---|
| 0 | Commit the tree and record the commit. Every architecture in the round runs on these bytes. Then ask whether the tree still supports the records, with the weld contracts: <code>python -m pytest meep_gpu parity/meep_gpu -m certification</code>. Any weld they report as drifted has to be re-run for the architecture already certified FIRST, and bound first, before a second architecture can join it — see "Which architecture binds first" below | laptop | nothing |
| 1 | The family gates and composition probes, with dispatch pinned off. A Triton campaign root carries a <code>device.json</code> naming the card; a CUDA artifact names its own (see "What names the card") | GPU host | run artifacts |
| 2 | The rebinds: <code>rebind_triton_welds.py</code>, its bit-identity mode, <code>recut_composition_records.py</code> (including <code>--family-recert</code>) and <code>rebind_cuda_welds.py</code>. Each reads the architecture off the run and adds that record | laptop | the two <code>fingerprints.json</code> |
| 3 | Check admission before spending another GPU hour: <code>python -c "from meep_gpu import fastpath; print(fastpath.capability_admission('triton'))"</code>. Both tables must list the new architecture, and any entry still short is named | laptop | nothing |
| 4 | The route campaigns, from the default, one per table, in the directory <code>&lt;stamp&gt;_cc&lt;capability&gt;</code>; then <code>recut_driver_dispatch_record.py</code> | GPU host, then laptop | the dispatch records |
| 5 | The end-to-end go/no-go, from the default; its licence is transcribed onto the table that composes first | GPU host, then laptop | the licence |
| 6 | Optional: the timing record, <code>timing_cc&lt;capability&gt;.json</code> | quiet GPU host | the timing record |

### Which architecture binds first

**Only the REBIND order is constrained. The gate runs are not.** Step 1 does not go
through dispatch admission, so its gates run on any card, in any order, on any night,
and nothing has to be bound before they do. What a gate produces is an artifact whose
digests are the bytes it ran; step 2 is a separate, laptop-side act that reads those
artifacts and writes records.

So two architectures can be gated in either order, or at the same time on two hosts,
provided both run the SAME COMMIT. They can be bound in either order too: on one commit
the two orders leave the same ledger, byte for byte
(<code>parity/meep_gpu/test_rebind_binding_order.py</code>). The orders differ only in
what the first bind has to say:

* **The architecture already recorded, first.** Its rebind moves the digests onto the
  commit, and the new architecture's rebind then joins it. No flag.
* **The new architecture, first.** Its rebind moves the digests, which would strand the
  recorded architecture's record, so it refuses and names it. Pass
  <code>--supersede &lt;that capability&gt;</code>: the working ledger then admits only the
  new architecture until the recorded one is re-bound on the same commit, which restores
  it. Do not commit or push the ledger between the two binds.

Either way, a rebind carries a per-run fact the fresh artifact does not state (a step
budget, a curated narrative) from that architecture's previous record only while that
record is live on the bytes being bound, so neither order can copy a measurement of
other code into a new record.

A gate artifact does not expire. It stays bindable for as long as the commit it ran is
the one being bound, which is the whole reason the two steps are separable: GPU hours
are the scarce thing, and they do not have to be spent in the binding order. What does
invalidate it is the tree moving — a later commit that touches a pinned file retires
every architecture's record for that weld at once, and the gates have to run again on
the new bytes.

**The first round on a tree has no shortcut.** When the weld contracts in step 0 report
drift across the cited surface, the architecture already named in the ledger is not
actually certified for the bytes being shipped, and the round is a full re-run for THAT
architecture before it is an expansion to any other. <code>--supersede</code> refreshes
nothing on its own: it drops the named architecture's certification, and only that
architecture's own re-bind on the same commit brings it back. Use it for the binding
order above, never as a way to certify one card without re-running the other.

### What names the card

The Triton rebinds read the architecture from a <code>device.json</code> in the
campaign's output root, which <code>drive_triton_weld_gates.py</code> writes when it
drives the fleet. A gate run by hand, outside that driver, has no such file, so write
one into the same directory before leaving the GPU host:

    python parity/meep_gpu/triton_device_identity.py --write <campaign-root>

It takes that one flag and nothing else, and refuses anything else by name. Run it in
the environment the gates ran in, with the same <code>CUDA_VISIBLE_DEVICES</code>, since
it records the card the LIVE host reports rather than the card the gate used.

<code>device.json</code> is a Triton-side file only. The CUDA rebind never reads one:
each CUDA artifact records the device it ran on itself, in one of three places
<code>rebind_cuda_welds.CAPABILITY_PATHS</code> reads —
<code>environment.compute_capability</code>,
<code>provenance.device.compute_capability</code>, or the block
<code>gate_provenance.stamp()</code> writes — and every place a leg fills has to agree.
A leg that fills none is refused by name, and the remedy is to re-run that gate, not to
write a file beside it.

### The toolchain is pinned, and only some tools check it

Install the pinned versions on the GPU host — the ones
<code>environments/nvidia-linux.yml</code> carries — not the current ones:

| | Pinned to |
|---|---|
| Triton | <code>3.1.0</code> (<code>fastpath.validated_triton_versions()</code>) |
| CuPy | <code>13.5.1</code> (derived from the live records; nothing declares it) |

What each step-2 tool does with a run on another version:

| Tool | Triton | CuPy |
|---|---|---|
| <code>rebind_triton_welds.py</code> | refused | refused |
| <code>recut_composition_records.py</code> | refused | not checked |
| <code>rebind_cuda_welds.py</code> | not checked | not checked |

A refusal happens on the laptop, after the GPU hours are spent. Where nothing checks,
a run on another version binds without complaint, so the pin is the operator's to
keep: confirm both versions on the GPU host before the first gate, as the round's
first step. No flag widens either set today; adopting another Triton or CuPy is a
change to the code, made deliberately, never a rebind option.

The coverage census and the dispatch boards carry no device dimension, so a new
architecture needs neither re-cut.

### What a round needs beside the checkout

**A clone of this repository is not enough to run a round.** Many gates read measured
inputs that are not committed: the predicate-coverage censuses
(<code>predicate_coverage_*/</code>, whose <code>examples.jsonl</code> and
<code>tests*.jsonl</code> the corpus-admission legs count), the fusion boards
(<code>fusion_matrix_triton_*/fusion_matrix.json</code>), the expansion and
subnormal-policy records (<code>expansion_probe_*</code>,
<code>complex_expansion_*</code>, <code>device_subnormal_policy_*</code>), and earlier
gate artifacts that some probes compare against. They live in the evidence archive under
<code>parity/meep_gpu/results/</code>, which is not part of this repository, and the
gates open them at those paths.

A gate whose input is absent **refuses** rather than passing on an empty input ("the
census is absent: …"), so a round on a bare clone releases nothing for those gates.
Before step 1, on every GPU host in the round:

1. stage the evidence inputs into <code>parity/meep_gpu/results/</code> of the clone,
   from the same copy on every host — two architectures that read different censuses
   or boards are not measured against the same question;
2. provide a MEEP source tree for the corpus legs, at the MEEP release the censuses were
   cut from: <code>~/meep</code> by default, or set <code>MGPU_SITE_MEEP_SOURCE</code>
   to a MEEP checkout or <code>MEEP_GPU_CORPUS_ROOT</code> to a directory holding
   <code>examples/</code> and <code>tests/</code>.

<code>parity/meep_gpu/cases.py</code>, the benchmark case builders that three
composition gates import, is committed: earlier rounds staged it from the archive as a
file the tree did not track, and the records still name it as such.

**Expansion records belong to the architecture that measured them.** A complex gate
licenses its kernels from an expansion record, and the record describes how one
architecture's compiler expanded complex multiplies, so a record measured on one card
licenses nothing on another. The Triton fleet measures its own on the card it runs on
(<code>unified_expansion</code> and <code>complex</code> run first for that reason), and
the CUDA gates that take <code>--expansion-probe</code> are handed that record.
<code>gate_cuda_complex_offdiag_stencil_welds.py</code> is the exception: it reads its
records from fixed archive paths and checks only their subnormal policy, not the
architecture that measured them. On any architecture other than the one those records
came from, do not run it until it checks the architecture as well; until then the CUDA
table cannot admit that architecture, and the CUDA artifacts the round does produce
stay bindable for when it can.

What a round does NOT need: any edit to the admission code, any new ledger key, or a
switch to let an uncertified card run its own certification. The family gates do not go
through dispatch admission, so they run on an unadmitted card as they are; only the
route and licence legs need step 3 to have passed first.

## Running a gate

### Metal

Run the gate script directly, on an Apple GPU:

    python parity/meep_gpu/gate_metal_pml.py --out parity/meep_gpu/results/<subject>_<date>/gate.json

Every Metal gate calls the gate runner from its own entry point. The runner
reads, after the gate has run, which source files the process actually imported,
hashes them, and writes the release verdict with those digests. A gate's exit
code alone releases nothing. Exit code 75 means the gate could not certify on
this host, which is distinct from a failure.

A Metal device gate does not share a process with a CPU-MEEP oracle; the two run
as separate processes.

Verify a saved artifact against the tree without re-running it:

    python parity/meep_gpu/metal_gate_runner.py --verify \
        parity/meep_gpu/results/<subject>_<date>/gate.json

### Hand-written CUDA

Run the <code>gate_cuda_*.py</code> scripts directly, on a CUDA device. Most
take <code>--subnormal-policy</code> and <code>--out</code>.

- Give each subnormal policy its own <code>CUPY_CACHE_DIR</code>. Under
  <code>keep</code> the path must carry the token <code>ftz_stripped</code>, and
  under <code>flush</code> it must not. CuPy computes its cache key above the
  point where the policy changes the compile, so a directory shared between
  policies serves one policy's binaries to the other, silently.
- A <code>flush</code> leg also needs
  <code>--import-meep-for-host-policy</code>, because MEEP's own switch is the
  route to the host processor's flush setting.

### Triton

Triton gates need a CUDA device with Triton 3.1.0.
<code>probe_fused_kernel_bit_identity.py</code> is the shared gate;
<code>--module</code> is required and names the kernel module under test. Give
each subnormal policy its own <code>TRITON_CACHE_DIR</code> as well.

### On a shared host

Export <code>TRITON_LIBCUDA_PATH</code> and <code>LD_LIBRARY_PATH</code> in the shell
that drives the round, as INSTALL.md step 3 sets them up. Triton links against
<code>-lcuda</code> and most driver installations provide only
<code>libcuda.so.1</code>, so without that directory **every device leg fails at link
time and the gate reports a clean SKIP rather than an error** — a round that looks like
it ran and certifies nothing. The fleet driver falls back to a different directory name
than INSTALL.md creates (<code>$HOME/triton_libcuda_stub</code> against
<code>$HOME/triton_libcuda</code>), so set the variable rather than relying on either
default, and check the first gate's artifact names a device leg before letting the rest
of the fleet run.

Pin one idle device with <code>CUDA_VISIBLE_DEVICES</code> and check immediately
before the run that no other process holds it. The fleet driver ships
<code>DEFAULT_FORBIDDEN_GPUS = {4, 5}</code>, which are set aside on the host it was
written for; on any other host pass your own set, or
<code>--forbidden-gpus ''</code> to declare that nothing is set aside, otherwise the
driver refuses the two cards by index on a machine where they are free. The per-family launchers the
NVIDIA campaigns were run with (batch-scheduler jobs and single-host shell
drivers) are not part of this repository; the certification records still name
them as provenance. <code>parity/meep_gpu/README.md</code> gives the recipe they
followed: one device, one CuPy cache per subnormal policy, a fresh Triton cache,
and unbuffered output to a log.

## Running a campaign

### All Metal gates, one tree, one pass

    zsh parity/meep_gpu/recut_metal_gates.sh parity/meep_gpu/results/<subject>_<date> <stamp>

The script finds the gates by listing the directory, so a gate added later is
included by the same command. It runs the three expansion probes first and hands
their artifacts to every later gate; if a probe fails it runs no gate. It writes
one flushed line per gate to a summary file, and it never rewrites a ledger:
binding the ledger is a separate, deliberate step.

The second argument, a stamp such as <code>2026-10-01_round1</code>, sets the
layout the ledger step reads: with it, each gate's artifact lands in
<code>parity/meep_gpu/results/metal_&lt;family&gt;_&lt;stamp&gt;/gate.json</code>
and the first directory keeps only the logs and the summary. Without a stamp the
artifacts land under the first directory, where
<code>rebind_metal_welds.py</code> does not look, and it reports every entry as
without an artifact.

### The Metal route campaign

    zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --dry-run --lanes 3 <stamp> <probe-root> <complex-probe>
    zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --lanes 3 <stamp> <probe-root> <complex-probe>

The first command prints the plan and launches nothing: the order of the legs,
each leg's arguments, its environment and its command. The campaign has five
legs.

- **Environments are per process.** Each leg is launched with exactly the
  variables it needs, and the script refuses to start if a leg would inherit a
  dispatch or policy variable from the calling shell.
- **Directories are exclusive, and results are never retried.** A leg that
  already holds a finalized artifact is skipped. A leg directory without one is
  refused by name. A leg that lands unreleased is a result, and the re-run is a
  new stamp.
- **One driver per campaign.** A second invocation on the same stamp refuses
  while the first is alive.
- **Exit status.** 0: every leg released. 1: a leg failed, was refused, or is
  unreleased. 2: usage. 3: another live driver holds the campaign. 4: the
  per-leg environments are not isolated.

Three lanes is the measured recommendation on one Apple M1 Max (2026-09-20, a
170 s workload over 4 of the campaign's 42 cases): 14 of 14 concurrent gate
processes released, and three lanes gave 2.9 times the throughput of one. The
measurement says nothing about a campaign of several hours. The script's
default is one lane.

### NVIDIA tables

The NVIDIA device legs run through the per-family drivers above, one family at
a time, each under both subnormal policies. The route gates are
<code>gate_dispatch_fused_route.py</code> for the fused products and
<code>gate_dispatch_end_to_end.py</code> for whole lifted simulations.

## Binding the ledgers and cutting the boards

After a campaign has released, in this order. Each rebind and recut tool reports
by default and writes only with <code>--write</code>; read its report first.

| Step | Tool | What it does |
|---|---|---|
| 1 | <code>record_cuda_regate.py</code> | Transcribes a hand-written CUDA campaign into <code>certification.json</code>, every fact derived from the artifacts |
| 2 | <code>rebind_triton_welds.py</code>, <code>rebind_cuda_welds.py</code>, <code>rebind_metal_welds.py --stamp &lt;stamp&gt;</code> | Rebind each table's certification entries to the campaign's artifacts and digests. The Metal tool reads <code>parity/meep_gpu/results/metal_&lt;family&gt;_&lt;stamp&gt;/</code>, the layout <code>recut_metal_gates.sh</code> writes when given the stamp |
| 2a | <code>python -c "from meep_gpu.metal_kernels import launch; launch.write_fingerprints()"</code> | Rewrites the Metal table's own source and toolchain digests (the <code>metal_kernels</code> block of its ledger) from the tree. Run it deliberately, once, after the last Metal gate of the round has released: a fingerprint written before the gates that check it certifies nothing, and <code>recut_metal_gates.sh</code> never calls it |
| 3 | <code>recut_composition_records.py</code> | Re-cuts the composition records from a composition re-run, and refuses one whose curated measurement no longer describes the tree |
| 4 | <code>recut_driver_dispatch_record.py</code> | Re-cuts a table's dispatch record from a route-gate run |
| 5 | <code>cut_timing_record.py</code> | Cuts a table's timing record from timing rows; <code>--check</code> recomputes a record from the rows it names |
| 6 | <code>build_triton_fusion_matrix.py</code>, <code>build_cuda_fusion_matrix.py</code>, <code>build_fusion_matrix.py</code> | Cut the coverage boards for the Triton, hand-written CUDA and Metal tables |
| 7 | The two test suites | The ledger-contract tests pass when every entry is bound to the files in the tree. Four of them read the round's own records and are listed under <code>evidence_archive</code>: they run only in a checkout where the evidence archive is restored with its marker and the round's new record directories have been added beside the archived ones. In a checkout that holds only the round's records they skip, and the skip reason names the missing marker |

Hash the staged tree before a round starts, and check the digests again when it
ends. A gate refuses release when a source file changes during its run, because
measurements taken on either side of the change describe different programs.

## Timing is a separate activity

A benchmark is not part of the chain that releases a kernel. Timing rows come
from <code>bench_fused_products.py</code>, on an otherwise idle host, and each
row carries its own verdict. See
[Benchmark fairly](../guides/validation-and-performance.md#benchmark-fairly).
