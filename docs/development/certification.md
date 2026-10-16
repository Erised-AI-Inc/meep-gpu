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

The 1.0 certification round ran on this release's bytes in the two reference
environments ([The MEEP a round is measured against](#the-meep-a-round-is-measured-against)),
under the round stamp <code>2026-10-15_10</code>, in chunks. The gates, the binds, the
route campaigns and the direct corpus check ran on the round's frozen commit. Every file
the release changes after it, the four ledgers the round's binds and records wrote
included, is in the run closure of none of the 250 gate and probe scripts, so every
artifact stands on the released files (an artifact stands where every file its import
record names is byte-equal).

| Item | State |
|---|---|
| The device gates (CERT-3) | NVIDIA, one RTX A6000 (compute capability 8.6): 123 of 123 units hold, under <code>keep</code> and <code>flush</code>. Metal, one M1 Max (<code>applegpu_g13s</code>): 62 of 62 units hold, under <code>flush</code> |
| The kernel ledgers shipped in the package | Bound from those runs: the two NVIDIA ledgers and the CUDA certification record, and the Metal ledger (62 of 62 welds rebound); <code>flush</code> evidence admitted for 75 of 75 Triton arms and 34 of 34 hand-written CUDA arms; the hygiene guard reads the four ledgers CLEAN |
| The route campaigns and the go/no-go for dispatch on by default (CERT-4) | Released on every table: Triton 3 of 3 legs, hand-written CUDA 5 of 5, Metal 4 of 4 with residency held. The go/no-go: GO, the ship leg 9 of 9 whole simulations dispatched, its null control 9 of 9 on the array path ([Correctness under dispatch](../guides/kernel-dispatch.md#correctness-under-dispatch)) |
| The records behind dispatch on by default and the Metal residency default | The three tables' <code>driver_dispatch</code> records, re-cut from those campaigns: the Triton record carries the go/no-go, the Metal record the held residency every leg-case ran under |
| The coverage boards | Cut by the round: 334, 352 and 327 of 597 seam-instances reachable in dispatch (Triton, CUDA, Metal) ([Coverage](../guides/kernel-dispatch.md#coverage-what-fraction-of-the-fusion-surface-dispatches)) |
| MEEP identity of the array path (CERT-1, Table 2) | 193, 193, 192 and 192 of 194 corpus simulations on the four references ([Table 2](meep-identity-corpus.md)) |
| The direct corpus check (CERT-5) | NVIDIA: 192 of 192 rows on each of four legs. Metal: 160 of 160 against the Linux flush reference outside 32 pre-registered rows, 192 of 192 against the Mac flush reference ([L3](#the-direct-corpus-check-against-meep-l3)) |
| The ledger-contract tests | The certification suite (pytest marker <code>certification</code>), not run by default and run with <code>-m certification</code>. The tests it still fails on are each declared in <code>tools/ci/pending_certification.txt</code>, under a block that names the step that removes them; on the records' tree the contract files and the listed tests read 0 undeclared failures. See [The certification suite is not run by default](testing.md#the-certification-suite-is-not-run-by-default) |
| Timing | Not part of certification. The 1.0 timing (CERT-8) ran on this release's bytes on 2026-10-16, after the certification round; its figures and their conditions are in [Will it help?](../guides/will-it-help.md#the-comparison-with-meep). 1.0.0 ships the 2026-09-21 dispatch-preference records, and the records the 1.0 timing cut land in a later 1.0.x release ([Timing is a separate activity](#timing-is-a-separate-activity)) |

Every GPU but these two runs the kernels by default, uncertified, and a GPU is added by a
round of its own on these bytes that writes records and no code
([Certifying a new GPU](certify-a-new-gpu.md)). The round's own record, with each number
and the record it comes from, is section 12 of
<code>parity/meep_gpu/rounds/1.0/PRE_CERTIFICATION_STATE.md</code>; the release notes in
<code>CHANGELOG.md</code> summarise it.

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

A defect several gates share is spelled once, in the probe's shared default, and
each gate that arms it does so through its own needle table
(`parity/meep_gpu/gate_needles.py`): the gate borrows the shared spelling until its
own kernels need another, which is then that gate's own entry and moves no other
gate's bytes. The curl and constitutive defects are spelled on the real-PML pair's
text, which every gate that arms them rewrites; a grouping defect plants the grouping
the kernels carried before they took MEEP's, and must be caught. Because a mutation
leg's verdict is a function of the bytes it compiles,
those bytes are derived on the host for every table and pinned in
`parity/meep_gpu/gate_needle_anchors.json`; `test_mutation_anchors.py` checks them on
every test run, and a difference names the table, the needle and the kernel that
moved.

A kernel table is certified on a named toolchain and device identity, and under
a named subnormal policy. The identity is recorded with the verdict. The two
NVIDIA tables support compute capability 7.0 to 9.0 (the Triton table with
Triton 3.1): they run on a supported identity outside the certified one by
default and record it as uncertified, except that a table certified for the
device and toolchain runs alone in place of one that is only supported, and they
refuse an identity outside the supported range by name unless
<code>MEEP_GPU_HARDWARE=any</code>. The Metal table supports every Apple
GPU: it runs in any Apple GPU environment and records whether it is the
certified one. <code>MEEP_GPU_HARDWARE=certified</code> restricts every table to
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
| <code>validate_*_reference_vs_stepping.py</code> | Is a gate's hand reference, the transcription its kernels are compared with, the array path itself? Word for word on NumPy, with no device, on the division the array path stepped with | A JSON artifact; the exit status is the verdict |
| <code>replay_*</code> | Arithmetic over an existing record, such as how many corpus simulations a predicate admits | A projection; never a device measurement |
| <code>sweep_*</code>, <code>survey_*</code> | The MEEP corpus, one script at a time: accepted, lifted, and compared with MEEP | One result per case, written as it lands |
| <code>bench_*</code> | Throughput | Timing rows with a verdict per row |
| <code>build_*_fusion_matrix.py</code> | Which seam-instances of the corpus a table's products serve and dispatch | A coverage board |
| <code>build_parity_board.py</code> | Per kernel table, which modules carry the chunk-record select and MEEP's operation order, and are word-identical to the array path on the host, on the device, and certified on the live array path; per array-path family, which are word-identical to MEEP under each subnormal baseline | The bit-parity board, <code>parity/meep_gpu/parity_board/</code> |
| <code>record_dev_evidence.py</code> | Development evidence, not certification: an integrator's re-run of a lane's device legs, bound to the module's bytes, its emitted device text and the live array path | <code>parity/meep_gpu/parity_board/dev_evidence.json</code>, read by the board's column (d-dev) only |
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

### What a gate's import record holds

Every gate artifact records the repository modules its process imported
(<code>gate_provenance.stamp()</code>'s <code>imported_source_sha256</code>, and the
Metal runner's <code>source_sha256</code>), and the rebind tools, the composition and
bit-identity re-cuts, the Metal runner's per-stamp manifest and the board builders check
that whole record against the checkout. An edit to any file in a gate's record therefore
voids that gate's evidence, so the record has to be the gate's own dependency set: its
kernel module, the array path it compares against, the kernel utilities it uses and its
harness.

The package root binds its public names on first use, so importing a kernel module does
not import the driver, the dispatch planners (<code>fastpath.py</code>,
<code>fastpath_cuda.py</code>, <code>metal_dispatch.py</code>,
<code>dispatch_preference.py</code>) or the MEEP lift. A gate that does need them (a
route, dispatch or composition gate, or one whose legs build an
<code>FdtdDriver</code>) is declared in
<code>parity/meep_gpu/test_gate_import_closure.py</code>; any other gate whose import
closure reaches them fails that test. To see which gates an edit voids before making it:

```
python parity/meep_gpu/import_closure.py bill meep_gpu/fastpath.py
```

It lists the gate and probe scripts whose run can import the file (an upper bound read off
the source: the gate's <code>main</code> and every function it reaches).

A weld's curated pins can name a file its gate no longer imports: some curated sets were cut
from import records the eager package root had filled with the driver, the planners and the
lift. The rebind tools keep curated sets, so such a pin leaves only by a row of the tool's
<code>RETIRED_UNIMPORTED_PINS</code>, which this command generates:

```
python parity/meep_gpu/import_closure.py unimported-pins metal
```

(<code>triton</code> and <code>cuda</code> likewise). A row needs both halves: the file is a
module the eager root loaded, and it lies outside the run closure of the weld's gate. A
weld's own test file, or a kernel module its gate loads by computed name, stays pinned.
<code>parity/meep_gpu/test_rebind_unimported_pins.py</code> holds each tool's table to the
command both ways; a rebind drops a row's pin only when it rebinds the weld, and refuses
where the rule stops holding (the file back in the gate's run closure or in its record, or
a run that is not the checkout's bytes).

## The ledgers

Seven JSON ledgers ship inside the package and are read at run time:

| Ledger | Holds |
|---|---|
| <code>meep_gpu/triton_kernels/fingerprints.json</code> | The Triton table's certification entries and its dispatch record |
| <code>meep_gpu/cuda_kernels/fingerprints.json</code> | The hand-written CUDA table's certification entries |
| <code>meep_gpu/cuda_kernels/certification.json</code> | The hand-written CUDA table's campaign records, including the device source strings |
| <code>meep_gpu/cuda_kernels/own_cell_hoist_reference.json</code> | The reference record of one hand-written CUDA kernel rewrite |
| <code>meep_gpu/metal_kernels/fingerprints.json</code> | The Metal table's certification entries and its dispatch record |
| <code>meep_gpu/triton_kernels/timing_cc86.json</code>, <code>meep_gpu/cuda_kernels/timing_cc86.json</code> | Timing records cut from timing rows, one per compute capability. 1.0.0 ships the 2026-09-21 cut; the 1.0 timing's (2026-10-16) lands in a later 1.0.x release |

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

Commit hashes in the records name commits of the development history, which the
public release repository does not carry; records bind files by sha256.

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

### The array path the ledgers were certified against

Every gate compares a kernel with the array path, but an entry pins the kernel's
files and only a few entries pin an array-path module. An arithmetic change to the
array path would leave those pins green while every gate run against the new array
path failed. So each ledger also records the array path's behaviour, under the key
<code>array_path_oracle</code>:

- every entry that pins source bytes carries <code>fixture_set</code> and
  <code>fingerprint_sha256</code>;
- the <code>driver_dispatch</code> record carries the same two values, one digest per
  fixture case (<code>case_sha256</code>), the cell count of every form class the
  fixture reaches (<code>form_classes</code>), the classes the fixture set defers and
  why, and the code digests of the array-path modules
  (<code>provenance_code_sha256</code>, reported and not judged).

The fingerprint is the SHA-256 of the float32 words that the fixture set of
<code>meep_gpu/array_path_oracle.py</code> produces when the NumPy array path steps it:
seeded real-storage cases (<code>CASES</code>: PML, conductivity, off-diagonal and
dispersive materials, nonlinearity, cylindrical m = 0, folds, metallic walls,
out-of-plane and BFAST wavevectors, stored-current sources and the deposit repair),
a few complete steps each, in well under a second. The fixture calls no libm function while it steps, uses
no source envelope, forms no complex product whose rounding depends on the host, and
runs with subnormals kept, so the fingerprint is a property of the array path and not
of the host. Complex storage, Bloch phases and cylindrical m &ne; 0 are deferred to a
later fixture set by name.

Each weld contract (<code>test_&lt;table&gt;_weld_contract.py</code>) recomputes the
fingerprint and fails by name, listing the fixture cases whose words moved, when the
ledger records another one. A comment or docstring edit to the array path leaves it
unchanged. The leaves are written only by <code>record_array_path_oracle.py</code>
(below); <code>--verify</code> recomputes them and writes nothing, which is how the
same tree is checked on another host.

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

The first such round, an H200 beside the RTX A6000, was prepared on 2026-10-03 for the
0.9.x code in <code>parity/meep_gpu/rounds/h200_2026-10-03/</code>: the round driver
<code>run_round.sh</code> that runs step 1's gates on one card, and the manifest of the
evidence inputs those gates read. The runbooks it follows on a Slurm cluster, for the
round and for the timing that follows the bind, are kept in the development repository.
Its bind was held, and no compute capability 9.0 record ships in 1.0: the H200 is
supported and not certified, and its certification on the 1.0 code is planned for a
1.0.x release. That directory is a dated record of the 0.9.x procedure: its scripts
and runbooks name the retired <code>MEEP_GPU_ALLOW_UNCERTIFIED</code>, which 1.0 refuses
by name, and not <code>MEEP_GPU_HARDWARE</code>, and a 1.0.x round runs from
<code>parity/meep_gpu/rounds/1.0/</code> instead, chunked, with the new card named as such:
<code>chunks.py partition --machine nvidia --capability 9.0 --new-device</code> cuts the
card's own chunk index, whose stamps carry the card (<code>&lt;round&gt;_cc90_c&lt;k&gt;</code>,
so no root of it equals or extends a root of the certified card's chunks), and
<code>round.py launch gates --chunk-index &lt;it&gt; --chunk c&lt;k&gt; --capability 9.0
--new-capability</code> runs step 1 (<code>rounds/1.0/RUNBOOK.md</code>, CERT-3 on a new
device). The flag is the round tooling's, not the package's: <code>legs.py check</code>
refuses a gate stage on a capability no table admits without it, refuses it on a capability
both tables admit, and refuses it on the route stage, so a card is never taken for a new
one, or for a certified one, silently. An Apple GPU of another architecture takes
<code>--machine metal --architecture &lt;its architecture&gt; --new-device</code> the same
way. [Certifying a new GPU](certify-a-new-gpu.md) gives that round step by step for a
contributor who does not hold the evidence archive: the reference environment, the round's
inputs bundle, the chunks, the bind that adds the new card's records (run on the
contributor's own host, and run again by the maintainers from the returned artifacts), the
route campaigns, sharing a card with another user's idle process, and what to send back.

The steps below are in order because the route and licence legs dispatch, and so need
the architecture already admitted by the records steps 1–2 write. The family gates in
step 1 do not: see "Which architecture binds first" for what is actually ordered and
what only looks ordered. On a card of compute capability 7.0 to 9.0 that is not yet
certified, the package dispatches by default, uncertified. So the route gate
(<code>gate_dispatch_fused_route.py</code>) refuses to start a leg, before its
directory exists, unless the table under test certifies the card: its compute
capability in <code>capability_admission</code>, and on the Triton table the installed
Triton version too (<code>--smoke</code> is not asked). It also refuses a leg started
with <code>MEEP_GPU_HARDWARE</code> set to any value: <code>=any</code> would let a
table that only supports the card compose beside the certified one, and the legs
measure the default; and one started with the retired
<code>MEEP_GPU_ALLOW_UNCERTIFIED</code> set, which the package refuses by name. The licence gate of step 5 is not guarded this way, because it
refuses every <code>MEEP_GPU_*</code> variable and reads no admission: a licence leg started before step 3 passes runs uncertified rows,
and <code>recut_driver_dispatch_record.py</code> refuses every one of them after the
run, so check step 3 first. The family gates of step 1 pin dispatch off and
are not affected; the Triton fleet driver (<code>drive_triton_weld_gates.py</code>)
also runs its children with <code>MEEP_GPU_HARDWARE=certified</code>.

| # | Step | Where | Writes |
|---|---|---|---|
| 0 | Commit the tree and record the commit. Every architecture in the round runs on these bytes. Then ask whether the tree still supports the records, with the weld contracts: <code>python -m pytest meep_gpu parity/meep_gpu -m certification</code>. Any weld they report as drifted has to be re-run for the architecture already certified FIRST, and bound first, before a second architecture can join it — see "Which architecture binds first" below | laptop | nothing |
| 1 | The family gates and composition probes, with dispatch pinned off: the chunks of the card's own index (<code>round.py launch gates --capability 9.0 --new-capability</code>, above). A Triton campaign root carries a <code>device.json</code> naming the card; a CUDA artifact names its own (see "What names the card") | GPU host | run artifacts |
| 2 | <code>chunks.py verify --selection</code>, which refuses a run whose artifacts stamp another card or whose card cannot be read, then the rebinds <code>chunks.py bind</code> prints (none while a run a tool would bind against the selection stands): <code>record_cuda_regate.py</code> (a block for the card's campaign, beside the others), <code>rebind_triton_welds.py</code>, its bit-identity mode, <code>recut_composition_records.py</code> (including <code>--family-recert</code>) and <code>rebind_cuda_welds.py</code>. Each reads the architecture off the run and adds that record; the bind of a new card re-points no board | laptop | the two <code>fingerprints.json</code>, <code>certification.json</code> |
| 3 | Check admission before spending another GPU hour: <code>python -c "from meep_gpu import fastpath; print(fastpath.capability_admission('triton'))"</code>. Both tables must list the new architecture, and any entry still short is named | laptop | nothing |
| 4 | The route campaigns, from the default, one per table, in the directory <code>&lt;stamp&gt;_cc&lt;capability&gt;</code>, where the stamp is the one <code>meep_gpu/route_stamps.py</code> names for the table (<code>route_stamps.route_gate(&lt;table&gt;)</code>; set once, at the freeze, before the campaigns run); then <code>recut_driver_dispatch_record.py</code>, which refuses any other directory name, an unset stamp and a campaign that has already run | GPU host, then laptop | the dispatch records |
| 5 | The end-to-end go/no-go, from the default, in <code>route_stamps.licence_campaign("triton")</code> followed by the card's device (<code>dispatch_end_to_end_&lt;stamp&gt;_cc90</code>, one per capability, as <code>round.py launch route</code> names it); its licence is transcribed onto the table that composes first, and the re-cut refuses a licence from any other directory | GPU host, then laptop | the licence |
| 6 | Optional: the timing record, <code>timing_cc&lt;capability&gt;.json</code> | quiet GPU host | the timing record |

### Which architecture binds first

**Only the REBIND order is constrained. The gate runs are not.** Step 1 does not go
through dispatch admission, so its gates run on any card, in any order, on any night,
and nothing has to be bound before they do (the round tooling asks only that a card the
ledgers do not admit be named as new: <code>--new-capability</code>). What a gate produces is an artifact whose
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
(<code>fusion_matrix_triton_*/fusion_matrix.json</code>), the subnormal-policy
records (<code>device_subnormal_policy_*</code>), and earlier
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
| Linux x86_64 (NVIDIA) | <code>bash parity/meep_gpu/build_meep_133_linux.sh</code>, from the repository root (it clears the environment itself), on top of <code>environments/locks/meep-gpu-ref-linux-64.conda.txt</code> and <code>meep-gpu-ref-linux-64.pip.txt</code>: the environment of the 2026-10-03 RTX A6000 round, with the certified CuPy, PyTorch and Triton, and what only the MEEP identity corpus needs (<code>environments/locks/README.md</code>, "Corpus-only packages") |
| macOS (Apple silicon) | <code>bash parity/meep_gpu/build_meep_133_macos.sh</code>, from the repository root (it clears the environment itself), on top of <code>environments/locks/meep-gpu-ref-osx-arm64.conda.txt</code> and <code>meep-gpu-ref-osx-arm64.pip.txt</code>: the environment the Metal kernels were certified in on the Apple M1 Max, with the certified PyTorch, and what only the MEEP identity corpus needs (<code>environments/locks/README.md</code>, "Corpus-only packages") |

**The reference is an identity reference.** Its field updates round every operation on
their own, so the array path can be held to its stored words exactly. That is judged on
the machine code, not on the flags: each script refuses to finish unless the fused
multiply-add census of the <code>libmeep</code> and <code>_meep</code> it built
(<code>tools/fma_census.py</code>) finds none outside the compiler-rt division builtins,
its build record (<code>share/meep-gpu-build/build_record.json</code>) holds that census,
the contraction the flags asked for and the host's math library, and MEEP's own words on
four cases hash to the recorded digests (<code>parity/meep_gpu/meep_identity_smoke.py</code>).
Both scripts compile with <code>-O2 -ffp-contract=off -fno-fast-math</code>, the Linux one
with <code>-march=x86-64 -mtune=generic</code> beside them (baseline x86-64 has no fused
multiply-add, and the script refuses flags under which <code>-march</code> is not
<code>x86-64</code>). On a Mac, clang contracts by default: the MEEP the 0.9.x Metal kernels
were certified against carries 1,253 fused operations in <code>libmeep</code>, so the Mac's
certification MEEP is the contraction-off build from the round that certifies 1.0 on. The
locks are the reference and change only through
<code>environments/locks/recut_locks.py</code>. A host joining a round checks its MEEP with
<code>python tools/check_install.py --require-identity-reference keep</code> (INSTALL.md,
"The MEEP identity reference"); an existing build whose census is clean is recorded in
place, not rebuilt.

Both build from the release tarball, pinned by SHA-256, with
<code>--enable-shared --enable-single --enable-portable-binary --with-mpi
--without-scheme</code> (and <code>--with-openmp</code> on Linux, as the NVIDIA reference
was built). <code>build_meep_133_native.sh</code> runs the same scripts with
<code>-O3 -march=native</code> (or <code>-mcpu=native</code> on Apple silicon) into an
environment of its own: that build is a timing control for the comparison with MEEP
([Running the timing ladder](timing.md#the-meep-controls)), never the reference a
round is measured against, and it fuses, so the scripts record its census without
judging it (<code>--timing-control "&lt;flags&gt;"</code>). Both work around three defects in that tarball, and both end by asserting
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
Each build also keeps the corpus (the tarball's <code>python/examples</code> and
<code>python/tests</code>, with every file's SHA-256, and beside them the corpus siblings,
the files outside <code>python/</code> that corpus rows read, each named with its SHA-256 in
the record; <code>--refresh-corpus</code> brings an existing prefix's corpus up to date and
rewrites only the record's <code>corpus</code> entry), links <code>ffmpeg</code> from the
imageio-ffmpeg pin and, on Linux, the driver's <code>libcuda</code> for Triton's launcher,
takes the identity verdicts in keep and flush, and writes
<code>share/meep-gpu-build/reference_build.json</code>, which names all of them. The scripts
refuse their retired knobs by name (<code>MEEP_REFERENCE_SOLVE</code>,
<code>MEEP_SIGMA_PATCH</code>, <code>MEEP_BUILD_ROLE</code>, <code>MEEP_FFMPEG</code>);
<code>build_meep_133_identity.sh</code> is gone, its cleared environment and flags being the
scripts' own. Both create <code>meep-gpu-ref</code> by default.
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
a flush leg, so every CUDA weld and one Triton weld would refuse there. The
package's own x86-64 glibc lever also reaches flush without MEEP (the MEEP-identity
replay uses it), but the gates take the host's flush from MEEP, the mechanism MEEP's
own runs use.

<code>parity/meep_gpu/cases.py</code>, the benchmark case builders that three
composition gates import, is committed: earlier rounds staged it from the archive as a
file the tree did not track, and the records still name it as such.

What a round does NOT need: any edit to the admission code, any new ledger key, or a
switch in the package to let an uncertified card run its own certification. A certified
identity dispatches whether or not it is in the supported range, so a card outside it
needs no switch once steps 1–2 have certified it. The family gates do not go through
dispatch admission, so they run on an unadmitted card as they are; only the route and
licence legs need step 3 to have passed first. The round tooling's
<code>--new-capability</code> changes no gate and no admission: it is the operator saying
the card is new, which <code>legs.py check</code> holds to the ledgers in both directions.

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
<code>MEEP_GPU_HARDWARE=certified</code> a fact that is not certified, or not judged,
is refused by name and the run takes the array path on the host CPU. A round on
another Mac is what makes a supported GPU certified, not what lets the kernels run
there.

A round, in order:

| # | Step | Writes |
|---|---|---|
| 0 | Commit the tree and run the weld contracts, as in step 0 of an NVIDIA round | nothing |
| 1 | On the Mac, with <code>PYTORCH_MPS_FAST_MATH</code> unset: <code>recut_metal_gates.sh</code> with a stamp ([All Metal gates, one tree, one pass](#all-metal-gates-one-tree-one-pass)). It records the environment before the first gate and after the last | the gate artifacts, and <code>results/metal_environment_&lt;stamp&gt;/start.json</code> and <code>end.json</code> |
| 2 | <code>rebind_metal_welds.py --stamp &lt;stamp&gt;</code>, or <code>mint_metal_weld.py</code> for a family's first weld; then <code>launch.write_fingerprints()</code> | this architecture's run, <code>runs[&lt;architecture&gt;]</code>, in the Metal <code>fingerprints.json</code> |
| 3 | The Metal route campaign under <code>metal_runs.route_campaign(route_stamps.route_gate("metal"), &lt;architecture&gt;)</code>, then <code>recut_driver_dispatch_record.py --backend metal</code> | this architecture's route run in the dispatch record |

A Mac that does not hold the evidence archive runs step 1 with
<code>parity/meep_gpu/rounds/1.0/metal_round.py</code> (its RUNBOOK, CERT-3), after
staging the archive files the Metal gates read and the release tree does not carry
(the round's inputs bundle, unpacked as the RUNBOOK's preflight says).
Among them are the four files every lift leg reads (below); the round's preflight names
any that is absent, gate by gate, and stops before the first gate. None is measured per
architecture. [Certifying a new GPU](certify-a-new-gpu.md#part-b-a-new-apple-gpu-architecture)
gives that round step by step, chunked, for a Mac of an architecture the ledger does not
admit, with the bind run on that Mac. The 0.9.2 driver in <code>parity/meep_gpu/rounds/metal_2026-10-04/</code>
is retired: its lift children ran under the interpreters the census recorded, and 1 of
the 7 digests in its input manifest no longer matches the evidence archive.

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
  stamp is <code>route_stamps.route_gate("metal")</code>
  (<code>meep_gpu/route_stamps.py</code>); the recut reads the architecture off every
  leg's <code>provenance.apple_gpu</code> and refuses any other directory name.
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
test and the three Metal weld-contract tests that walk those digests were listed in
<code>tools/ci/pending_certification.txt</code> until the 1.0 round re-cut it. The
migrated ledger, the admission edit and the writers land together: a migrated ledger
read by the old admission certifies nothing, and the new admission reading an
unmigrated ledger judges nothing.

**The 1.0 round, on the Apple M1 Max**, ran one held Metal route campaign,
<code>zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --lanes 3
2026-10-15_10_applegpu_g13s</code> (4 of 4 legs released), then
<code>recut_driver_dispatch_record.py --backend metal --run
dispatch_metal_route_2026-10-15_10_applegpu_g13s --write</code>, and its records entry
removed the pending lines above.

**A second Mac**, on the same commit:

1. Run its fleet (step 1 above, with <code>rounds/1.0/metal_round.py</code>: in a 1.0
   round, the chunks of an index of its own architecture, <code>chunks.py partition
   --machine metal --architecture &lt;its architecture&gt; --new-device</code>, whose stamps
   carry the architecture; <code>rounds/1.0/RUNBOOK.md</code>, CERT-3 on a new device) and
   hand its outputs to the host that holds the evidence archive.
2. <code>rebind_metal_welds.py --stamp &lt;its stamp&gt;</code> (in a chunked round, the
   commands <code>chunks.py bind --selection</code> prints, one per selected stamp), read
   the report, then <code>--write</code>. **Do not commit the ledger yet.** On the same bytes it adds
   <code>runs[&lt;its architecture&gt;]</code> beside the Apple M1 Max's; a weld whose
   gate did not release there gets no run, and the intersection leaves that
   architecture not certified until it does. The ledger moves no byte the route record
   binds, so the Apple M1 Max's route run stays live. But once every cited weld has a
   run for the new architecture, that GPU is admitted before it has a route run, and
   <code>test_the_metal_record_is_welded_to_the_live_sources_when_it_exists</code>
   fails until it has one: committed alone, this ledger turns the merge bar red.
3. Copy the rebound ledger, uncommitted, into that Mac's checkout of the same commit
   (the only file that differs from it), and run the route campaign there under
   <code>2026-10-15_10_&lt;its architecture&gt;</code>. Bring its legs back,
   then, on the rebound ledger:
   <code>recut_driver_dispatch_record.py --backend metal --run
   dispatch_metal_route_2026-10-15_10_&lt;its architecture&gt;</code>, which adds
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

**Separate: the lift legs.** Nine <code>*_fused_hd_pair</code> gates re-lift corpus
rows, one child process per row: <code>gate_metal_fused_hd_pair.py</code>, the five
siblings that keep their own <code>lift_basis</code> (bfast, conductive,
cylindrical_real, folded, nonlinear), and the three built on
<code>hd_tail_gate_common.py</code> (beta_real, beta_complex, folded_complex). The
complex gate's lift leg runs in process. Rules (<code>parity/meep_gpu/lift_host.py</code>):

* **Every child runs under the gate's own interpreter**, the round's certification
  environment (<code>metal_round.py --python</code>). The interpreter path a census row
  records is kept on the row as <code>recorded_interpreter</code>, provenance only; no
  lift child launches it. (<code>probe_h_to_d_withdraw_order.py</code>, which no round
  runs, still does.)
* **Four input files, each required by name.** The census's
  <code>examples.jsonl</code>, <code>tests.jsonl</code> and
  <code>tests_param_matched.jsonl</code> under
  <code>results/metal_coverage_2026-09-04_m0complex/</code>, and
  <code>results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl</code>. None ships with the
  release tree. Without <code>tests_param_matched.jsonl</code> the basis used to
  shrink from 145 rows to 130 with no error; now a missing file stops the basis, and
  <code>metal_round.py preflight</code> stops the round, naming the gate and the file.
* **Rows that need more than MEEP and the package.** Five of the 145 rows declare it
  through their census lift record (<code>lift_host.LIFT_RECORD_NEEDS</code>), which
  names what its rows need between them and holds for each: coupler.py and ring_gds.py
  (a MEEP built against libGDSII; ring_gds.py also writes its layout with gdspy) declare
  both; mie_scattering.py, differential_cross_section.py (PyMieScatt) and
  waveguide_crossing.py (nlopt, autograd) declare all three. A host that lacks one need
  of a record counts out every row of it. The reference environments carry them
  (<code>environments/locks/README.md</code>), and there all 145 rows lift. On a host
  that lacks a need, the gate probes it under its own interpreter and records each
  affected row under <code>unlifted_on_this_host</code>, with the probe's reason. That
  row is counted out of the leg's denominator, as the CUDA gates count out a row their
  host cannot lift; a leg that asks about no row at all fails. The preflight lists
  those rows by gate and does not stop for them. A probe whose answer cannot be read
  fails the leg by name.
* **The complex gate** (<code>gate_metal_complex_fused_hd_pair.py</code>) reads the same
  four files for its cell, whose recorded grids it drives in process, and requires them
  by name too: without <code>tests_param_matched.jsonl</code> its cell used to shrink
  from 17 rows to 12 with no error. The preflight checks its four files as well, and
  computes no basis for it.
* **The corpus** the children walk is <code>MEEP_GPU_CORPUS_ROOT</code>.
  <code>metal_round.py</code> sets it to the corpus the reference environment's build
  keeps in its prefix, named with the sha256 list of its files by the
  <code>corpus</code> entry of
  <code>&lt;prefix&gt;/share/meep-gpu-build/reference_build.json</code>, or to
  <code>--corpus-root</code>, after checking every file against that list and every
  corpus sibling the record names beside it (a record naming none is refused); an
  unlisted file that Python could import (source, byte code outside
  <code>__pycache__</code>, or an extension module) is refused. A gate run by hand, or
  by <code>recut_metal_gates.sh</code>, needs it exported.

None of the nine gates, nor the complex gate, is cited.

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
- Set <code>CUPY_ACCELERATORS=</code> (empty) before CuPy is imported, as every lane
  child of the round does (<code>parity/meep_gpu/rounds/1.0/RUNBOOK.md</code>, "The
  child environments"), so no CuPy accelerator backend stands in for the kernels the gate
  launches.

### Triton

Triton gates need a CUDA device with Triton 3.1.0.
<code>probe_fused_kernel_bit_identity.py</code> is the shared gate;
<code>--module</code> is required and names the kernel module under test. Give
each subnormal policy its own <code>TRITON_CACHE_DIR</code> as well.

The five composition gates (<code>probe_triton_cylindrical_composition.py</code>,
<code>probe_triton_symmetry_composition.py</code>,
<code>probe_triton_conductivity_composition.py</code>,
<code>probe_triton_engine_route.py</code> and <code>gate_triton_no_pml.py</code>) take
no policy flag: they install the policy <code>MEEP_GPU_SUBNORMAL_POLICY</code> names
(<code>keep</code> when it is unset) before they build anything, refuse by name when it
is not attained, and stamp it as the artifact's <code>subnormal_policy</code>. A
<code>keep</code> run needs a <code>CUPY_CACHE_DIR</code> carrying
<code>ftz_stripped</code>, as every keep gate does; the fleet driver sets one per gate. Under
<code>flush</code> on x86 the cylindrical and symmetry probes need MEEP imported first,
which the round's flush fleet does for every gate
(<code>parity/meep_gpu/rounds/1.0/fleet_launch.py --host-import-wrapper</code>); the
other three import MEEP themselves. That wrapper imports MEEP and nothing else before it
hands over, because three ADE-chain gates set <code>CUPY_ACCELERATORS</code> themselves
and refuse a process in which CuPy is already imported.

<code>gate_triton_complex_ade.py</code>, <code>gate_triton_complex_no_pml_stored_e.py</code>
and <code>probe_triton_cylindrical_fused_magnetic_pair.py</code> read the policy the same
way (<code>MEEP_GPU_SUBNORMAL_POLICY</code>, <code>keep</code> when unset) and install it
through the CuPy cache-token guard. The cylindrical probe cuts one policy per process:
run it once per campaign, keep and flush.

A gate floor that demands subnormal operands is declared not applicable on a
<code>flush</code> run only when three measurements agree
(<code>parity/meep_gpu/flush_vacuity.py</code>): the installed policy is flush, the host
processor flushes when the class is seeded, and the class's own census counted no
subnormal word. Under flush the class cannot occur: every value it seeds with host
arithmetic flushes to zero. Every other check on the row is unchanged, and signed-zero
floors stand under both policies.

### On a shared host

Triton links its launchers against <code>-lcuda</code> and most driver installations
provide only <code>libcuda.so.1</code>, so without a directory holding
<code>libcuda.so</code> **every device leg fails at link time and the gate reports a clean
SKIP rather than an error** — a round that looks like it ran and certifies nothing. It
compiles those launchers with <code>$CC</code>, else the first <code>gcc</code> on
<code>PATH</code>, which is the host's, outside the environment. The round's drivers
(<code>run_lane.py</code>, <code>fleet_launch.py</code>, <code>l3_corpus.py</code>,
<code>cert8.py</code>) give every child both from the reference environment, unless the
shell sets them: <code>CC=&lt;prefix&gt;/bin/x86_64-conda-linux-gnu-cc</code> and
<code>TRITON_LIBCUDA_PATH=&lt;prefix&gt;/lib/meep-gpu-libcuda</code>, the link its build
made (<code>share/meep-gpu-build/libcuda.json</code>), also first on
<code>LD_LIBRARY_PATH</code>; a reference environment without either is refused by name
before a GPU is taken. A gate run by hand gets neither: export both in its shell. The
fleet driver run alone falls back to <code>$HOME/triton_libcuda_stub</code>, which no
environment holds, so set the variable rather than relying on that default, and check the
first gate's artifact names a device leg before letting the rest of the fleet run.

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
included by the same command. It writes one flushed line per gate to a summary file, and it never rewrites a ledger:
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

    zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --dry-run --lanes 3 <stamp>
    zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --lanes 3 <stamp>

The first command prints the plan and launches nothing: the order of the legs,
each leg's arguments, its environment and its command. The campaign has four
legs, in launch order: <code>band_witness</code>, <code>harness_keep</code>,
<code>shipped</code> and <code>harness_flush</code>. The stamp is the one
<code>route_stamps.route_gate("metal")</code> names, without its
<code>dispatch_metal_route_</code> prefix, followed by the GPU architecture the Mac's
fleet recorded: <code>2026-10-15_10_applegpu_g13s</code> on an Apple M1 Max for the
1.0 round.

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
- **One residency control cannot bite under held residency** (owner ruling R29).
  The route gate's residency controls are <code>residency-sync_in-dropped</code>,
  <code>residency-inner-bracket-dropped</code> and
  <code>residency-verify-after-every-sync_out</code>. The first leaves unclaimed the
  mirror of the first host write that declares an ownership move through
  <code>Residency.acquire_write</code>, and under a hold no host write does: the
  array-path passes beside the kernels write the held volumes through the sparse
  write paths, which move no ownership. So
  <code>residency-sync_in-dropped</code> reads <code>NOT-APPLICABLE</code> on every
  held case, with that reason. In the 1.0 round's campaign
  (<code>dispatch_metal_route_2026-10-15_10_applegpu_g13s</code>)
  <code>acquire_write</code> was entered 0 times on 42 of 42 dispatched cases of each
  dispatch leg, while the sparse write paths counted writes on all 42. On the same cases
  <code>residency-verify-after-every-sync_out</code> read <code>RESIDENCY-CLEAN</code>, and
  the other must-diverge control, <code>residency-inner-bracket-dropped</code>, read
  <code>NOT-APPLICABLE</code> for a reason of the compositions rather than of the hold: 35
  carry only bracketed wrappers whose pre-launch host work reads the fields, and 7 carry
  none. The held seam therefore has no must-diverge null of its own in 1.0:
  its evidence is <code>residency-verify-after-every-sync_out</code> and the byte
  identity of every case. A null that drops one sparse-door device write and must
  diverge is a 1.0.1 item; the control and its text are unchanged in 1.0.

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
| 2 | <code>rebind_triton_welds.py</code>, <code>rebind_cuda_welds.py</code>, <code>rebind_metal_welds.py --stamp &lt;stamp&gt;</code> | Rebind each table's certification entries to the campaign's artifacts and digests. The Metal tool reads <code>parity/meep_gpu/results/metal_&lt;family&gt;_&lt;stamp&gt;/</code> and <code>metal_environment_&lt;stamp&gt;/</code>, the layout <code>recut_metal_gates.sh</code> writes when given the stamp, and writes each entry's run under <code>runs[&lt;architecture&gt;]</code>, its <code>host</code> line from the environment records: beside other architectures' runs on unchanged bytes, and on moved bytes only for architectures named with <code>--supersede</code> ([Certifying another Apple GPU](#certifying-another-apple-gpu)). It re-cuts each entry's <code>device_sha256</code> as <code>mint_metal_weld.py</code> cuts it, only where the run imported every pinned module at the checkout's bytes, and otherwise skips the entry by name |
| 2a | <code>python -c "from meep_gpu.metal_kernels import launch; launch.write_fingerprints()"</code> | Rewrites the Metal table's own source and toolchain digests (the <code>metal_kernels</code> block of its ledger) from the tree. Run it deliberately, once, after the last Metal gate of the round has released: a fingerprint written before the gates that check it certifies nothing, and <code>recut_metal_gates.sh</code> never calls it |
| 3 | <code>recut_composition_records.py</code> | Re-cuts the composition records from a composition re-run, and refuses one whose curated measurement no longer describes the tree |
| 3a | <code>record_numpy_reference_weld.py</code>, <code>record_specialized_kernel_sources.py --module &lt;module&gt;</code> | Re-record two Triton digests that no gate writes, with no GPU. The first runs the NumPy validator that the D/E record names and records its digest only when the run states the record's own case counts. The second records a specialised module's digest in <code>specialized_kernel_sources</code> only when the module's standalone device weld and its composition weld both have a live record and pin the module's live bytes, so it runs after steps 2 and 3 |
| 3b | <code>record_array_path_oracle.py</code> | Writes the <code>array_path_oracle</code> leaves of the three ledgers from a run of the fixture set, and refuses a run that leaves a declared form class without a cell, a module that moved during the run, a ledger it would reformat, or a write that would move anything outside the leaves. Run it when the array path is frozen for a round, before the gates: from then on an array-path edit fails the three weld contracts until the tables are certified again. <code>--verify</code> recomputes and compares on any host and writes nothing |
| 4 | <code>recut_driver_dispatch_record.py</code> | Re-cuts a table's dispatch record from a route-gate run, and refuses a row that dispatched on an identity the table's records do not certify: an NVIDIA device, or a Metal GPU architecture, PyTorch or Metal frontend that is not certified or not judged |
| 5 | <code>cut_timing_record.py</code> | Cuts a table's timing record from timing rows; <code>--check</code> recomputes a record from the rows it names |
| 5a | <code>repoint_board_bindings.py triton --stamp &lt;stamp&gt;</code>, <code>repoint_board_bindings.py metal --stamp &lt;stamp&gt;</code> | Re-points the Triton board's <code>GATE_BOUND</code> and the Metal board's gate bindings to the round's runs, from a measurement: each binding moves to the one run of its gate in the round that the board's release-binding floor admits with zero drift, and a binding with no such run refuses the write (on the Triton board, <code>--withhold-unrun</code> withholds it by name instead). A run of another device than the one the binding names is refused (<code>--device</code> moves a board to another device on purpose), and so is every run while either device cannot be read, so a second device's round adds ledger records and moves no board. The CUDA board binds through the ledgers and needs none |
| 6 | <code>build_triton_fusion_matrix.py</code>, <code>build_cuda_fusion_matrix.py</code>, <code>build_fusion_matrix.py</code> | Cut the coverage boards for the Triton, hand-written CUDA and Metal tables |
| 6a | <code>build_parity_board.py manifest</code>, then the module test files (every marker), then <code>build_parity_board.py outcomes --junit ... --manifest-before &lt;manifest&gt;</code> and <code>build</code> | Cut the bit-parity board. <code>outcomes</code> refuses a run during which the tree moved, <code>build</code> refuses an outcomes record of another tree ("the outcomes subject and the working tree disagree"), and <code>check</code> refuses a board of another tree. Its CERTIFIED column counts a module only from a ledger entry that records the live array-path oracle and a released run on named hardware, so it reads 0 until step 3b has run on the certified tree. Its development-evidence column (d-dev) is never read by CERTIFIED |
| 7 | The two test suites | The ledger-contract tests pass when every entry is bound to the files in the tree. Four of them read the round's own records and are listed under <code>evidence_archive</code>: they run only in a checkout where the evidence archive is restored with its marker and the round's new record directories have been added beside the archived ones. In a checkout that holds only the round's records they skip, and the skip reason names the missing marker |

Hash the staged tree before a round starts, and check the digests again when it
ends. A gate refuses release when a source file changes during its run, because
measurements taken on either side of the change describe different programs.

## The direct corpus check against MEEP (L3)

The gates compare each kernel table with the package's own array path, and Table 2
([MEEP identity on the corpus](meep-identity-corpus.md)) compares the array path with
MEEP. L3 checks the whole chain at once: each corpus row is lifted and stepped with
compiled kernels dispatched, and every stored word and every <code>get_array</code> word
at every compared step is judged against the stored digests of the reference MEEP's
words, the Tier B record ([MEEP identity](testing.md#meep-identity)). MEEP is imported
for the lift and is never stepped, so a GPU host that runs no MEEP reference can run
it. It needs the reference MEEP installed, and a row refuses unless the MEEP images it
loaded are the ones the record names for its set.

    python -u parity/meep_gpu/rounds/1.0/l3_corpus.py --table triton|cuda|metal --baseline keep|flush \
        --record <Tier B record>/meep_identity_tier_b.json --out <leg directory> \
        [--gpu N] [--parallel K] [--rows LABEL,... | --labels FILE]

| <code>--table</code> | What steps the rows | Judged against, by default |
|---|---|---|
| <code>triton</code> | An NVIDIA GPU under the default precedence: the Triton table composes first | <code>linux_keep</code> or <code>linux_flush</code>, by <code>--baseline</code> |
| <code>cuda</code> | An NVIDIA GPU with <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code>: the hand-written CUDA table composes first | The same |
| <code>metal</code> | The Apple GPU, <code>--baseline flush</code> only | <code>linux_flush</code> (the claim), <code>mac_flush</code> and <code>linux_keep</code>, all from the same run |

- **One leg per invocation, one row per process.** <code>l3_row.py</code> builds the row
  with the corpus module, lifts it with <code>prefer_gpu=True</code> and dispatch on,
  requires the step count, the compared steps, the compared cells and the storage type
  to equal the record's, and then steps the driver alone. At each set's planted-flip
  step it flips one word by one ulp in a copy: the digest must change, and exactly the
  x-plane holding that word, or the row does not hold. Its verdict uses
  <code>meep_identity.py</code> and the Tier C harness unmodified.
- **The tool builds every row's environment.** Each <code>MEEP_GPU_*</code> variable is
  removed (dispatch on, the hardware scope at its default), with
  <code>MEEP_GPU_BACKEND_PREFERENCE=cuda</code> set again for <code>--table cuda</code>;
  threads are 1 and the checkout is alone on <code>PYTHONPATH</code>. A leg under the subnormal baseline it
  names installs that policy before the lift. On NVIDIA it also gives each leg its own
  CuPy and Triton caches, with <code>CUPY_ACCELERATORS</code> empty, and mints the
  <code>keep</code> leg's CuPy cache through the package. <code>--shipped-default</code>
  instead runs a row as a user's run would, installing nothing. It is informational and
  not a certification leg.
- **Verdict and exit status.** <code>VERDICT HOLDS</code> (exit 0) when every row the
  primary set holds was run and holds against it, none errored and none was refused for
  its files; <code>DIFFERS</code> or <code>INCOMPLETE</code> otherwise (exit 1); exit 3
  when the record cannot support the leg, before any row runs. A row that holds on the
  array path (<code>holds_array</code>, its plan refused by name) still holds: it is
  bit-identical to MEEP without kernels. Rows where the two hosts' reference MEEPs
  disagree cannot equal both: <code>--expect-differ references --expect-differ-holds
  mac_flush</code> derives them from the record and counts each as holding when it
  differs against the primary set and holds against the named one.
- **Written as it lands.** One flushed line per row in <code>progress.log</code>, one
  JSON line per row in <code>rows.jsonl</code>, <code>summary.json</code> rewritten after
  every row. A rerun skips the rows already recorded.
- **Rows the record excludes are not run.** <code>ex.stochastic_emitter</code> is excluded
  from every set (its source draws unseeded random numbers). <code>ex.coupler</code> is
  excluded from the two Linux sets of the 1.0 record as an expected difference (owner
  ruling R24), so no leg runs it, every leg's primary set being a Linux one. Its
  <code>EigenModeSource</code> is lifted by re-solving the mode with the lift's own
  <code>get_eigenmode</code> call, and on the x86-64 Linux references that solve, and
  under <code>flush</code> MEEP's own, is not bitwise reproducible across processes:
  12 of 20 runs of the row held there (2 corpus legs, 6 re-runs and 12 lift probes, under
  <code>keep</code> and <code>flush</code>), a run that did not hold differed in 1,683,570
  to 5,051,638 of 44,047,776 stored words, the only lifted inputs that varied between
  processes were the two sources' point amplitudes, and identical lifted inputs with
  identical MEEP words gave identical array-path words in 9 of 9 processes. On the Apple
  M1 Max it held in 4 of 4 runs. A 1.0.1 release reads MEEP's stored eigenmode amplitudes
  instead of re-solving
  ([Compatibility](../guides/compatibility.md#an-eigenmodesource-can-differ-from-meep)).

The 1.0 round's legs, its go/no-go and its environment rules are in
<code>parity/meep_gpu/rounds/1.0/RUNBOOK.md</code> (CERT-5).
<code>test_l3_corpus.py</code>, beside the tool, tests its arguments, its per-row judge
(on synthetic words and on a Tier A fixture, each with a planted one-ulp flip it must
catch) and its output, on any host, with no GPU and no MEEP.

## Timing is a separate activity

A benchmark is not part of the chain that releases a kernel. Timing rows come
from <code>bench_fused_products.py</code>, on an otherwise idle host, and each
row carries its own verdict. A comparison with stock MEEP is run by
<code>timing_ladder.py</code> ([Running the timing ladder](timing.md)), which
writes its rows, caches and logs under its own run directory, outside every tree
it reads and outside every root <code>cut_timing_record.py</code> reads, so a
ladder's rows never enter a timing record by being found there. See
[Benchmark fairly](../guides/validation-and-performance.md#benchmark-fairly).
