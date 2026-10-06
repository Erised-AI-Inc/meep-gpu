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
| The certification round on the published files | Run on the files of 0.9.2: the Triton and hand-written CUDA tables on compute capability 8.6 (one RTX A6000), the Metal table on the Apple M1 Max (<code>applegpu_g13s</code>). The device gates whose pinned files had moved were re-run, and every certification entry in the shipped ledgers is bound to the published files |
| The go/no-go for dispatch on by default, and the route campaigns of the three tables | Run on the files of 0.9.2, in the rounds stamped 2026-10-05, and released: the go/no-go read GO (8 of 9 cases dispatched, the null control clean); the Triton campaign released 4 of 4 legs, the hand-written CUDA campaign its 5 required legs and the leg with Triton withheld from the process, and the Metal campaign 5 of 5 legs, residency held |
| The record behind the on-by-default dispatch setting | Cut from that go/no-go, as the licence in the Triton table's dispatch record (compute capability 8.6) |
| The record behind the Metal table's default residency mode | Cut from the Metal route campaign, in the Metal dispatch record's run for <code>applegpu_g13s</code> (mode <code>held</code>) |
| The coverage boards | Metal and hand-written CUDA re-cut on the files of 0.9.2, with the same figures (327 and 352 of 597 reachable in dispatch); Triton cut 2026-09-25 (334 of 597) and not re-cut |
| The fused-product timing records | Pending. They were cut from rows timed on an earlier deposit-repair route, and no timing campaign has run on these files |
| The ledger-contract tests | The certification suite (pytest marker <code>certification</code>), not run by default and run with <code>-m certification</code>. It shows 2 failures, both listed in <code>tools/ci/pending_certification.txt</code>: the two tests of the fused-product timing records. See [The certification suite is not run by default](testing.md#the-certification-suite-is-not-run-by-default) |

Other certification counts in this manual describe the dated round they name, on
the files of that date.

### Dispatch evidence on the published code (2026-09-29)

This section is the record of 0.9.0, kept as dated history; the table above is the
state of 0.9.2. The digests it quotes are those of the 0.9.0 files.

The default-dispatch go/no-go (<code>gate_dispatch_end_to_end.py</code>) and the
route campaigns of the three kernel tables ran on the code of 0.9.0.
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

Of what 0.9.0 left owed, the rounds of 0.9.2 cut the record behind the
on-by-default setting and the tables' dispatch records, re-ran the device gates
whose pinned files had moved and bound the ledgers, cut the record behind the
Metal table's default residency mode, and re-cut the Metal and hand-written CUDA
boards. Still owed after 0.9.2:

- a timing campaign on the published files, which re-cuts the two fused-product
  timing records and clears the two pending tests;
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
a named subnormal policy. The identity is recorded with the verdict. The two
NVIDIA tables support compute capability 7.0 to 9.0 (the Triton table with
Triton 3.1): they run on a supported identity outside the certified one by
default and record it as uncertified, except that a table certified for the
device and toolchain runs alone in place of one that is only supported, and they
refuse an identity outside the supported range by name unless
<code>MEEP_GPU_ALLOW_UNCERTIFIED=1</code>. The Metal table supports every Apple
GPU: it runs in any Apple GPU environment and records whether it is the
certified one. <code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code> restricts every table to
certified identities
([Certification on an NVIDIA GPU](../guides/kernel-dispatch.md#certification-on-an-nvidia-gpu),
[Certifying another Apple GPU](#certifying-another-apple-gpu)).
Supported is not certified: certification is what a round measures.

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
source files that produced it. An entry of the two NVIDIA ledgers keeps **one record
per GPU architecture** it has been run on, under its <code>runs</code> key:

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
a clone holds the path but not the run behind it. The run records are not in this
repository.

Commit hashes in the records name the pre-publication development history, which is
not part of this repository; records bind files by sha256.

<code>bound_sha256</code> is what makes an architecture's record expire on its own: a
record is LIVE only while it equals the digest of the entry's own pinned bytes, so an
edit to a certified file retires every architecture's evidence for that entry at once,
and no reader has to be told to re-check. A table's
<code>validated_compute_capabilities</code> is DERIVED from these records -- the
intersection, over the entries that table's arms cite, of each entry's live
architectures -- and is not written anywhere by hand.

An entry of the Metal ledger keeps one record per **Apple GPU architecture**, under
the same <code>runs</code> key, keyed by the architecture Metal names
(<code>applegpu_g13s</code>). Each run also records the PyTorch and Metal frontend it
ran with, and the Metal table's certified environment is derived from the live runs
([Certifying another Apple GPU](#certifying-another-apple-gpu)).

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

The first such round, an H200 beside the RTX A6000, is prepared in
<code>parity/meep_gpu/rounds/h200_2026-10-03/</code>: the round driver
<code>run_round.sh</code> that runs step 1's gates on one card, and the manifest of
the evidence inputs those gates read. The runbooks it follows on a Slurm cluster, for
the round and for the timing that follows the bind, are kept in the development
repository. It is prepared; no 9.0 record ships in 0.9.2, so compute
capability 9.0 is supported, not certified.

The steps below are in order because the route and licence legs dispatch, and so need
the architecture already admitted by the records steps 1–2 write. The family gates in
step 1 do not: see "Which architecture binds first" for what is actually ordered and
what only looks ordered. On a card of compute capability 7.0 to 9.0 that is not yet
certified, the package dispatches by default, uncertified. So the route gate
(<code>gate_dispatch_fused_route.py</code>) refuses to start a leg, before its
directory exists, unless the table under test certifies the card: its compute
capability in <code>capability_admission</code>, and on the Triton table the installed
Triton version too (<code>--smoke</code> is not asked). It also refuses a leg started
with <code>MEEP_GPU_ALLOW_UNCERTIFIED</code> set to any value: <code>=1</code> would let a
table that only supports the card compose beside the certified one, and the legs
measure the default. The licence gate of step 5 is not guarded this way, because it
refuses every <code>MEEP_GPU_*</code> variable and reads no admission: a licence leg started before step 3 passes runs uncertified rows,
and <code>recut_driver_dispatch_record.py</code> refuses every one of them after the
run, so check step 3 first. The family gates of step 1 pin dispatch off and
are not affected; the Triton fleet driver (<code>drive_triton_weld_gates.py</code>)
also runs its children with <code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code>.

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
Before step 1, on every GPU host in the round, stage the evidence inputs into
<code>parity/meep_gpu/results/</code> of the clone, from the same copy on every host:
two architectures that read different censuses or boards are not measured against the
same question. Stage nothing else there, since some gates read an archived record in
preference to the one the round measures.

### The MEEP a round is measured against

**Every certification is measured against MEEP 1.33.0 built from source in single
precision, with MPI and OpenMP**, and every host in a round uses that build. It is the
MEEP behind every certified kernel and every agreement figure in the manual. conda-forge's
<code>pymeep</code> is the documented way to *use* the package, but it is a
double-precision serial build with no single-precision variant, so an architecture
gated against it would differ from the reference in more than its GPU.

| Host | Builds it |
|---|---|
| Linux x86_64 (NVIDIA) | <code>bash parity/meep_gpu/build_meep_133_linux.sh</code>, from the repository root, on top of <code>environments/locks/meep-gpu-ref-linux-64.conda.txt</code> and <code>meep-gpu-ref-linux-64.pip.txt</code>: the environment of the 2026-10-03 RTX A6000 round, with the certified CuPy, PyTorch and Triton |
| macOS (Apple silicon) | <code>bash parity/meep_gpu/build_meep_133_macos.sh</code>, on top of <code>environments/locks/meep-gpu-ref-osx-arm64.conda.txt</code> and <code>meep-gpu-ref-osx-arm64.pip.txt</code>: the environment the Metal kernels were certified in on the Apple M1 Max, with the certified PyTorch |

Both build from the release tarball, pinned by SHA-256, with
<code>--enable-shared --enable-single --enable-portable-binary --with-mpi
--without-scheme</code> (and <code>--with-openmp</code> on Linux, as the NVIDIA reference
was built). <code>build_meep_133_native.sh</code> runs the same scripts with
<code>-O3 -march=native</code> (or <code>-mcpu=native</code> on Apple silicon) into an
environment of its own: that build is a timing control for the comparison with MEEP
([Running the timing ladder](timing.md#the-meep-controls)), never the reference a
round is measured against. Both work around three defects in that tarball, and both end by asserting
what they built, because two of those defects fail silently: an out-of-tree
<code>--enable-single</code> build installs cleanly and is double precision. Each script
refuses to finish unless MEEP reports 1.33.0, single precision and an MPI build, and the
GPU libraries are the certified versions. Neither solves its environment: each installs
its platform's lock files (an explicit conda lock, every package by URL and MD5, and a
pip requirements file, every PyPI package by version and wheel hash), refuses to build
unless the new prefix holds exactly the lock's conda packages, and ends with
<code>tools/compare_reference_environment.py --require-reference</code>, the comparison
of every numerics-relevant package with the lock. <code>environments/locks/README.md</code>
says what each lock reproduces and when it is re-cut: whenever the environment a round
certifies in changes, in the same change that records the round.
<code>MEEP_REFERENCE_SOLVE=1</code> solves as the scripts did before this change and says
that the result is not the reference. Both create <code>meep-gpu-ref</code> by default.
A host joining a round checks its environment with
<code>python tools/check_install.py --require-reference</code> before its first gate: it
fails unless the imported MEEP is 1.33.0, single precision and an MPI build and every
numerics-relevant package is the lock's.
<code>tools/compare_reference_environment.py</code> alone judges the packages, not MEEP.

**The flush policy needs MEEP to flush.** A flush leg reaches the host's float32
subnormal handling through <code>import meep</code>, which switches the processor to
flushing only when MEEP was compiled with that support. On x86_64 it is: after
<code>import meep</code>, <code>meep_gpu.backends.subnormals_flushed()</code> reads
<code>True</code>, measured on both the source build and conda-forge's. On Apple silicon
it reads <code>False</code> by design, and the Metal table takes its flushing from the
GPU. The Linux build script checks it; a host where it reads <code>False</code> cannot run
a flush leg, so every CUDA weld and one Triton weld would refuse there.

<code>parity/meep_gpu/cases.py</code>, the benchmark case builders that three
composition gates import, is committed: earlier rounds staged it from the archive as a
file the tree did not track, and the records still name it as such.

**Expansion records belong to the architecture that measured them.** A complex gate
licenses its kernels from an expansion record, and the record describes how one
architecture's compiler expanded complex multiplies, so a record measured on one card
licenses nothing on another. The Triton fleet measures its own on the card it runs on
(<code>unified_expansion</code> and <code>complex</code> run first for that reason), and
the CUDA gates that take <code>--expansion-probe</code> are handed that record.
<code>gate_cuda_complex_offdiag_stencil_welds.py</code> reads three records, one per
half, and checks each one's stated <code>environment.compute_capability</code> against
the card it is running on: a record measured on another architecture, or one that names
none, refuses the run by name before a kernel is compiled. Its defaults are the archive
records measured on compute capability 8.6, so on any other card pass this card's own
unified record with <code>--expansion-probe</code>, which stands for all three halves.
It also refuses a licence whose basis is not <code>measured</code>: the arbiter's
fallback table names no architecture. <code>gate_cuda_complex_offdiag_update_e.py</code>
still reads a fixed archive record without these checks; it backs no cited weld and
its runs do not bind, so leave it out of a round on another architecture.

What a round does NOT need: any edit to the admission code, any new ledger key, or a
switch to let an uncertified card run its own certification. A certified identity
dispatches whether or not it is in the supported range, so a card outside it needs no
switch once steps 1–2 have certified it. The family gates do not go
through dispatch admission, so they run on an unadmitted card as they are; only the
route and licence legs need step 3 to have passed first.

## Certifying another Apple GPU

A Metal weld is certified for one **environment**: four facts, each of which changes
the code that runs.

| Fact | How it is read | Certified |
|---|---|---|
| GPU architecture | <code>MTLDevice.architecture.name</code>, through the Objective-C runtime with <code>ctypes</code> (no PyTorch, no PyObjC; macOS 14 or later). It is the unit Metal compiles for, and it separates GPU generations that share a Metal GPU family: M3 and M4 are both Apple9 | <code>applegpu_g13s</code>, an Apple M1 Max |
| PyTorch | its version; PyTorch supplies <code>torch.mps.compile_shader</code> | 2.10.0 |
| Metal frontend | the version of the macOS build's shader compiler: one per macOS build, the same on every Mac on that build | <code>metalfe-32023.850.10</code> |
| <code>PYTORCH_MPS_FAST_MATH</code> | the process environment | unset or <code>0</code> |

The first three are typed nowhere. Each weld the Metal table cites
(<code>metal_dispatch.ARM_CERTIFICATION</code>, 45 welds) keeps one run record per GPU
architecture it was certified on, <code>runs[&lt;architecture&gt;]</code>, whose
<code>host</code> line a round writes as

    this machine: Apple MPS device applegpu_g13s, torch 2.10.0, metalfe-32023.850.10

and whose <code>torch</code> and <code>metal_frontend</code> fields are read off that
line when it is written. A run is **live** while its <code>bound_sha256</code> equals
the digest of the entry's pinned bytes ([The ledgers](#the-ledgers)).

This host's GPU architecture is **certified** when every cited weld has a live run for
it, and **not certified** when a cited weld has none: that weld never ran on this GPU
on these bytes, or its run was superseded. PyTorch and the Metal frontend are judged
against the runs recorded for this host's architecture: **certified** when every such
run recorded this host's value, **not certified** when one recorded another. A fact is
**not judged** when it could not be read here, when a cited weld's entry holds no
per-architecture run record, or, for PyTorch and the frontend, when this architecture
has no run to read them from. The admitted set is therefore an intersection, as on the
NVIDIA tables: a second GPU is certified only once every cited weld has a live run for
it. Fast math is a fixed rule rather than a record: no weld ran with it, and with
<code>1</code> PyTorch compiles every Metal source in fast-math mode
([Floating-point contract](../design/floating-point.md#fast-math-on-the-metal-table)).

Dispatch does not wait for a round, because every Apple GPU is already
**supported**: the kernels are meant to run there and run by default. Supported is
read off this host, not off the welds: <code>environment.device_supported</code> is
<code>True</code> when Metal names the architecture <code>applegpu_*</code> or, where
the architecture cannot be read, when the device name begins with the word
<code>Apple</code>; <code>False</code> for a GPU that is not Apple's; <code>None</code>
when neither was read. By default the Metal kernels run in any environment: a fact
that is not certified is recorded under <code>uncertified.served</code>, the run's
record says
<code>certified: False</code>, and the process prints one <code>meep_gpu: NOTE</code>
line, whose reason on an Apple GPU is that the GPU is supported; a fact that is not
judged runs with <code>certified: None</code> and no note. With
<code>MEEP_GPU_ALLOW_UNCERTIFIED=0</code> a fact that is not certified, or not judged,
is refused by name and the run takes the array path on the host CPU. A round on
another Mac is what makes a supported GPU certified, not what lets the kernels run
there.

A round, in order:

| # | Step | Writes |
|---|---|---|
| 0 | Commit the tree and run the weld contracts, as in step 0 of an NVIDIA round | nothing |
| 1 | On the Mac, with <code>PYTORCH_MPS_FAST_MATH</code> unset: <code>recut_metal_gates.sh</code> with a stamp ([All Metal gates, one tree, one pass](#all-metal-gates-one-tree-one-pass)). It records the environment before the first gate and after the last | the gate artifacts, and <code>results/metal_environment_&lt;stamp&gt;/start.json</code> and <code>end.json</code> |
| 2 | <code>rebind_metal_welds.py --stamp &lt;stamp&gt;</code>, or <code>mint_metal_weld.py</code> for a family's first weld; then <code>launch.write_fingerprints()</code> | this architecture's run, <code>runs[&lt;architecture&gt;]</code>, in the Metal <code>fingerprints.json</code> |
| 3 | The Metal route campaign under <code>metal_runs.route_campaign(METAL_DRIVER_ROUTE_GATE, &lt;architecture&gt;)</code>, then <code>recut_driver_dispatch_record.py --backend metal</code> | this architecture's route run in the dispatch record |

A Mac that does not hold the evidence archive runs step 1 from
<code>parity/meep_gpu/rounds/metal_2026-10-04/</code>: <code>run_round.sh</code> checks
the commit, the reference environment and the Metal environment, unpacks the inputs
bundle and verifies every file against the committed manifest before the first gate,
runs the fleet, reports how many of the 45 cited welds released, and packs the outputs
for the host that runs steps 2 and 3. The bundle carries the seven archive files the
Metal gates read: four files of the predicate census (the row set the
<code>*_fused_hd_pair</code> gates re-lift, and the one the cylindrical scan reads), the
H-to-D seam record, the spine baseline, and an NVIDIA signed-zero probe record that
every lift child opens before its battery although no Metal battery reads it. None is
measured per architecture. The four expansion probes are cut by the fleet on the Mac it
runs on and are never shipped. Its README says what else the lift legs need on another
Mac. Every stage of <code>run_round.sh</code> refuses to start before the one it
follows has finished, and the fleet and pack stages check the commit and the clean
checkout again, since <code>all</code> resumes at the first unfinished stage.

What the tools check:

* **The rebind and the mint write each run's <code>host</code> line from the
  campaign's two environment records** (<code>parity/meep_gpu/metal_environment.py</code>),
  not from the gate artifacts, and key the run by the architecture that line names.
  They refuse the whole campaign when a record is missing, when either could not read
  a fact, when a fact differs between the start and the end, or when either records
  <code>PYTORCH_MPS_FAST_MATH</code> set to anything but <code>0</code>. A gate
  artifact that records its own GPU (<code>environment.apple_gpu.architecture</code>,
  14 of the 62 on the Apple M1 Max fleet) must name the campaign's: the rebind skips
  that weld by name and the mint refuses.
* **A run's subnormal-policy line is carried, not retyped:** from this architecture's
  previous run of the gate, else from the one line every other live run of the entry
  carries (the policy is the table's), else from the artifact, by its policy value
  (<code>flush</code>) and never as the repr of a policy report.
* **One run per GPU architecture, so a second Mac joins the first.** On unchanged
  bytes the rebind adds the campaign's architecture beside the runs already recorded
  and leaves them byte for byte; a rebind of the same architecture replaces only that
  architecture's run. On bytes that moved, every other architecture's run would be left
  certifying code that no longer ships, so the rebind skips that weld and names the
  architectures; <code>--supersede &lt;architecture&gt;[,...]</code> rebinds it and
  leaves the named runs in the record, stale. The mint keeps the other architectures'
  runs under <code>--replace</code> by the same rule. Both refuse an entry still in the
  one-run shape: <code>migrate_metal_runs.py</code> moves a ledger once
  ([Records per Apple GPU architecture](#records-per-apple-gpu-architecture)).
* **The dispatch record needs the rebind first.** The Metal recut refuses a route row
  whose plan did not read the GPU architecture, PyTorch and the Metal frontend all
  certified; a fact not judged refuses like one not certified. The NVIDIA recut
  refuses a row on an uncertified device the same way. The route campaign therefore
  runs after the welds it cites are bound, as step 4 of an NVIDIA round follows
  step 2. It runs under <code>&lt;stamp&gt;_&lt;architecture&gt;</code>, where the
  stamp is <code>metal_dispatch.METAL_DRIVER_ROUTE_GATE</code>; the recut reads the
  architecture off every leg's <code>provenance.apple_gpu</code> and refuses any other
  directory name.
* **Nothing compares the versions with the certified ones.** A full round on another
  PyTorch or another macOS build binds, and the certified environment becomes the
  one it ran in; a partial one leaves cited welds that disagree, which certify
  neither value. <code>start.json</code> names what every weld will record: read it
  before letting the gates run on.

### Records per Apple GPU architecture

**Shape.** Each Metal weld entry keeps its digests, <code>status</code> and
<code>purpose</code>, and holds its runs under <code>runs[&lt;architecture&gt;]</code>.
A run holds the fields that describe one run of the gate
(<code>metal_runs.RUN_FIELDS</code>: <code>artifact_sha256</code>,
<code>environment_read_from</code>, <code>host</code>, <code>recorded_utc</code>,
<code>records</code>, <code>subnormal_policy</code>, <code>verdict_read_from</code>,
<code>torch</code>, <code>metal_frontend</code>, and
<code>_subnormal_policy_read_from</code>, which says where a rebind took the policy
line), plus <code>bound_sha256</code>, the digest <code>fastpath.bound_digest</code>
takes over <code>fastpath.BOUND_FIELDS</code>. A run is live while that digest equals
the entry's; <code>fastpath.live_capabilities</code> answers that for the admission and
for every tool.

**Where the rules live.** <code>meep_gpu/metal_runs.py</code> holds the write side: the
run fields, the <code>applegpu_*</code> key check, the shape check
(<code>shape_reasons</code>, which also requires a run's <code>host</code> line to name
the architecture it is keyed by and the PyTorch and frontend it records), the
staleness rule (<code>bind_architecture</code>: a write that moves the bound bytes names
every other architecture's run it would strand, and needs each superseded by name),
the intersection report and the route-campaign name. It is the Metal counterpart of
<code>fastpath.bind_capability</code>, <code>retired_shape_reasons</code> and
<code>route_campaign</code>, which accept only a <code>major.minor</code> key. The
dispatch ladder does not import it: the admission reads the runs inside
<code>metal_dispatch.py</code>, the file the Metal dispatch record binds.

**The certification quote.** <code>metal_dispatch.environment_block</code> writes the
architecture as <code>device.compute_capability</code> as well as
<code>device.architecture</code>, so <code>fastpath._certification_for</code> quotes
each dispatched arm's run for this GPU (its <code>host</code>,
<code>recorded_utc</code> and <code>records</code>) and no other architecture's.
Measured on one Apple M1 Max before it was built, comparing runs with and without it on
the same driver: the decision, the slots, the arms, <code>certified</code>, the status
line and the field were byte-equal; the NVIDIA paths in <code>fastpath.py</code> and
<code>dispatch_preference.py</code> that also read the key are never reached by a Metal
plan. On a GPU no weld ran on, the quote's text reads "no live run on compute
capability applegpu_...", NVIDIA wording; the record's
<code>uncertified.served</code> block carries the reason a Metal run took, and the
status line reads PyTorch and the frontend as <code>uncertified-unknown</code>, since
that architecture has no run to judge them by. Measured end to end on an Apple M1 Max
with a migrated ledger: the plan dispatched certified, both arms quoted the
<code>applegpu_g13s</code> run's <code>host</code>, <code>recorded_utc</code> and
<code>records</code>; with the architecture planted as <code>applegpu_g15s</code> the
same kernels ran (the field byte-equal), certified <code>False</code>, the NOTE line
named only the GPU architecture, and no arm quoted a run.

**The route record.** The Metal <code>driver_dispatch</code> record keeps one route run
per architecture: <code>status</code>, <code>legs</code>, <code>records</code>,
<code>recorded_utc</code>, <code>verdict_read_from</code>, the run's
<code>gate</code>, <code>artifact</code> and <code>what_was_measured</code> of
<code>released_fused_arms</code>, and the <code>residency</code> and <code>lift</code>
blocks read off the legs (<code>metal_runs.DISPATCH_RUN_FIELDS</code>). The release
predicate, the slots, <code>subnormal</code> and <code>environments</code> stay at entry
level. The residency block is the large one: about 217 KB of compact JSON and about
470 KB as the ledger file indents it, once per architecture.

**The migration, once.** <code>parity/meep_gpu/migrate_metal_runs.py</code> moves a
one-run ledger: it reads each weld's architecture, PyTorch and frontend off its own
<code>host</code> line, moves the run fields under that architecture, and dates the
dispatch record's route run as history with <code>runs: {}</code>, because the
admission edit moves <code>metal_dispatch.py</code>, which that record binds. It reports
by default, writes only with <code>--write</code>, refuses a ledger already migrated or
one it would reformat, and checks before writing, entry by entry: the move is
reversible (the route record's too), every digest the ledger held is still held and the only ones added are one
<code>bound_sha256</code> per run, each run is live, each <code>host</code> line is
carried byte for byte, and every cited weld's live runs record the environment its
<code>host</code> line recorded. Measured on the ledger of this tree in report mode: 62
of 62 weld entries move, the 1,020 digests are kept and 62 are added (1,082, every one
classified by the record walker), the cited welds certify
<code>applegpu_g13s</code>, PyTorch 2.10.0 and frontend 32023.850.10 before and after,
and the file grows from 710,634 to 758,334 bytes.

**What it costs.** The migration moves no bound bytes of any weld, so every run stays
live and no fleet is owed. The admission edit moves <code>metal_dispatch.py</code>,
which only the Metal dispatch record binds: the record's digests drift, so its contract
test and the three Metal weld-contract tests that walk those digests are listed in
<code>tools/ci/pending_certification.txt</code> until the round below re-cuts it. The
migrated ledger, the admission edit and the writers land together: a migrated ledger
read by the old admission certifies nothing, and the new admission reading an
unmigrated ledger judges nothing.

**The round that follows, on the Apple M1 Max.** One held Metal route campaign,
<code>zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --lanes 3
2026-10-05_perarch_applegpu_g13s &lt;probe-root&gt; &lt;complex-probe&gt;</code>, then
<code>recut_driver_dispatch_record.py --backend metal --run
dispatch_metal_route_2026-10-05_perarch_applegpu_g13s --write</code>, and the pending
lines above are removed.

**A second Mac**, on the same commit:

1. Run its fleet (step 1 above, or <code>run_round.sh</code> from
   <code>parity/meep_gpu/rounds/</code>) and hand its outputs to the host that holds
   the evidence archive.
2. <code>rebind_metal_welds.py --stamp &lt;its stamp&gt;</code>, read the report, then
   <code>--write</code>. **Do not commit the ledger yet.** On the same bytes it adds
   <code>runs[&lt;its architecture&gt;]</code> beside the Apple M1 Max's; a weld whose
   gate did not release there gets no run, and the intersection leaves that
   architecture not certified until it does. The ledger moves no byte the route record
   binds, so the Apple M1 Max's route run stays live. But once every cited weld has a
   run for the new architecture, that GPU is admitted before it has a route run, and
   <code>test_the_metal_record_is_welded_to_the_live_sources_when_it_exists</code>
   fails until it has one: committed alone, this ledger turns the merge bar red.
3. Copy the rebound ledger, uncommitted, into that Mac's checkout of the same commit
   (the only file that differs from it), and run the route campaign there under
   <code>2026-10-05_perarch_&lt;its architecture&gt;</code>. Bring its legs back,
   then, on the rebound ledger:
   <code>recut_driver_dispatch_record.py --backend metal --run
   dispatch_metal_route_2026-10-05_perarch_&lt;its architecture&gt;</code>, which adds
   its route run beside the first.
4. Commit the weld runs and the route run together.

The order is forced: every route row reads the architecture's verdict off the ledger
in the tree the campaign runs in, and the recut refuses a row that read it not
certified. Running the campaign on an uncommitted ledger is safe because nothing in it
reads git state (neither <code>run_metal_dispatch_campaign.sh</code> nor
<code>gate_dispatch_metal_route.py</code> does), and the route record does not bind
the ledger: each leg records the ledger's digest, but the recut compares only the nine
files the record binds. If the ledger must be committed before the route run exists,
that commit lists the test above in <code>tools/ci/pending_certification.txt</code>
and the recut's commit removes it.

If a round on one Mac moves bound bytes, the other architectures' runs are named by
the rebind and the recut, and the round either re-runs them on the new bytes or
supersedes them by name.

**Separate: the lift legs' interpreters.** The ten <code>*_fused_hd_pair</code> gates
re-lift corpus rows in the interpreter each census row records, an absolute path on the
host that cut the census. A Mac whose paths differ runs its round with
<code>ALLOW_NO_LIFT=1</code>, and those gates do not release there; none is cited.
Mapping a census interpreter onto the running host's moves those ten gates and
<code>beta_hd_gate_common.py</code>, each pinned by an uncited weld.

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
their artifacts to every later gate; if a probe fails it runs no gate. It then runs
<code>gate_metal_complex</code>, whose expansion leg writes the fourth probe, and only
then every other gate, so no gate runs before a probe it binds exists; a probe variable
inherited from the calling shell is unset, and a missing complex probe is named in the
summary and fails the run. It writes
one flushed line per gate to a summary file, and it never rewrites a ledger:
binding the ledger is a separate, deliberate step.

The second argument, a stamp such as <code>2026-10-01_round1</code>, sets the
layout the ledger step reads: with it, each gate's artifact lands in
<code>parity/meep_gpu/results/metal_&lt;family&gt;_&lt;stamp&gt;/gate.json</code>
and the first directory keeps only the logs and the summary. The stamp also has
the script record the Metal environment before the first gate and after the last,
into <code>parity/meep_gpu/results/metal_environment_&lt;stamp&gt;/start.json</code>
and <code>end.json</code>, and check that the two agree; a record it cannot write
stops the run before any gate. Without a stamp the artifacts land under the first
directory, where <code>rebind_metal_welds.py</code> does not look, and no environment
is recorded, so the rebind refuses the campaign.

### The Metal route campaign

    zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --dry-run --lanes 3 <stamp> <probe-root> <complex-probe>
    zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --lanes 3 <stamp> <probe-root> <complex-probe>

The first command prints the plan and launches nothing: the order of the legs,
each leg's arguments, its environment and its command. The campaign has five
legs. The stamp is the one <code>metal_dispatch.METAL_DRIVER_ROUTE_GATE</code> names,
without its <code>dispatch_metal_route_</code> prefix, followed by the GPU architecture
the Mac's fleet recorded: <code>2026-10-05_perarch_applegpu_g13s</code> on an Apple M1
Max.

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
| 2 | <code>rebind_triton_welds.py</code>, <code>rebind_cuda_welds.py</code>, <code>rebind_metal_welds.py --stamp &lt;stamp&gt;</code> | Rebind each table's certification entries to the campaign's artifacts and digests. The Metal tool reads <code>parity/meep_gpu/results/metal_&lt;family&gt;_&lt;stamp&gt;/</code> and <code>metal_environment_&lt;stamp&gt;/</code>, the layout <code>recut_metal_gates.sh</code> writes when given the stamp, and writes each entry's run under <code>runs[&lt;architecture&gt;]</code>, its <code>host</code> line from the environment records: beside other architectures' runs on unchanged bytes, and on moved bytes only for architectures named with <code>--supersede</code> ([Certifying another Apple GPU](#certifying-another-apple-gpu)) |
| 2a | <code>python -c "from meep_gpu.metal_kernels import launch; launch.write_fingerprints()"</code> | Rewrites the Metal table's own source and toolchain digests (the <code>metal_kernels</code> block of its ledger) from the tree. Run it deliberately, once, after the last Metal gate of the round has released: a fingerprint written before the gates that check it certifies nothing, and <code>recut_metal_gates.sh</code> never calls it |
| 3 | <code>recut_composition_records.py</code> | Re-cuts the composition records from a composition re-run, and refuses one whose curated measurement no longer describes the tree |
| 3a | <code>record_numpy_reference_weld.py</code>, <code>record_specialized_kernel_sources.py --module &lt;module&gt;</code> | Re-record two Triton digests that no gate writes, with no GPU. The first runs the NumPy validator that the D/E record names and records its digest only when the run states the record's own case counts. The second records a specialised module's digest in <code>specialized_kernel_sources</code> only when the module's standalone device weld and its composition weld both have a live record and pin the module's live bytes, so it runs after steps 2 and 3 |
| 4 | <code>recut_driver_dispatch_record.py</code> | Re-cuts a table's dispatch record from a route-gate run, and refuses a row that dispatched on an identity the table's records do not certify: an NVIDIA device, or a Metal GPU architecture, PyTorch or Metal frontend that is not certified or not judged |
| 5 | <code>cut_timing_record.py</code> | Cuts a table's timing record from timing rows; <code>--check</code> recomputes a record from the rows it names |
| 6 | <code>build_triton_fusion_matrix.py</code>, <code>build_cuda_fusion_matrix.py</code>, <code>build_fusion_matrix.py</code> | Cut the coverage boards for the Triton, hand-written CUDA and Metal tables |
| 7 | The two test suites | The ledger-contract tests pass when every entry is bound to the files in the tree. Four of them read the round's own records and are listed under <code>evidence_archive</code>: they run only in a checkout where the evidence archive is restored with its marker and the round's new record directories have been added beside the archived ones. In a checkout that holds only the round's records they skip, and the skip reason names the missing marker |

Hash the staged tree before a round starts, and check the digests again when it
ends. A gate refuses release when a source file changes during its run, because
measurements taken on either side of the change describe different programs.

## Timing is a separate activity

A benchmark is not part of the chain that releases a kernel. Timing rows come
from <code>bench_fused_products.py</code>, on an otherwise idle host, and each
row carries its own verdict. A comparison with stock MEEP is run by
<code>timing_ladder.py</code> ([Running the timing ladder](timing.md)), which
writes its rows, caches and logs under its own run directory, outside every tree
it reads and outside every root <code>cut_timing_record.py</code> reads, so a
ladder's rows never enter a timing record by being found there. See
[Benchmark fairly](../guides/validation-and-performance.md#benchmark-fairly).
