# Running the tests

## The two suites

Tests sit beside the code they test. Run each suite from the repository root,
one at a time:

    python -m pytest meep_gpu/ -q
    python -m pytest parity/meep_gpu/ -q

| Suite | What it tests | Size at the 0.9.0 preview, 2026-09-28 |
|---|---|---|
| <code>meep_gpu/</code> | The package: the array path, the lift, the planner, and every kernel table's predicates and emitters | 13,816 tests collected, of which 362 belong to the certification suite; 27 minutes on one Apple laptop with MEEP and PyTorch, 6 minutes with neither |
| <code>parity/meep_gpu/</code> | The harness: the gates' helpers, the record writers, the timing and census tools | 695 tests collected, of which 1 belongs to the certification suite; about 1 minute on the same laptop |

On that laptop, with MEEP and PyTorch installed, the default run of the package
suite read 12,921 passed, 0 failed and 532 skipped by a declared resource, and the
harness suite 682 passed, 0 failed and 12 skipped, at the 0.9.0 preview. [What a run of the
suites showed in the 0.9.0 preview](#what-a-run-of-the-suites-showed-in-the-090-preview)
gives the same runs with a double-precision MEEP and with neither MEEP nor a GPU library.

The suites run on a host with no GPU. A test that needs a device, a device
library or MEEP skips when it is absent, and says so.

In the environment of <code>environments/apple-silicon.yml</code> a process holds
one OpenMP runtime, and nothing has to be set. In an environment where MEEP and
PyTorch each load one, fix the environment (INSTALL.md, "One OpenMP runtime on
an Apple silicon Mac") rather than setting <code>KMP_DUPLICATE_LIB_OK</code>: with
it set, a process that holds two runtimes can crash later. Device gates run a
Metal device gate and a CPU-MEEP oracle in separate processes.

## Running them as CI does

The continuous-integration workflow, <code>.github/workflows/tests.yml</code>,
writes a JUnit report per suite and then accepts only the declared failures:

    python -m pytest meep_gpu -q -p no:cacheprovider -rfEs --junitxml=reports/meep_gpu.xml
    python -m pytest parity/meep_gpu -q -p no:cacheprovider -rfEs --junitxml=reports/harness.xml
    python -m pytest meep_gpu parity/meep_gpu -m certification -q -p no:cacheprovider -rfEs --junitxml=reports/certification.xml
    python tools/ci/check_failures.py reports/meep_gpu.xml reports/harness.xml reports/certification.xml

<code>check_failures.py</code> takes report paths only; it passes when every
failure in the reports is listed in <code>tools/ci/pending_certification.txt</code>.

Before the suites, the workflow runs the hygiene guard, which fails the job on a
host name, account name, local path or personal e-mail address in a tracked file
or in the coverage board's subject:

    python tools/ci/check_hygiene.py --legs tracked,subject

Its third leg, <code>--legs ledgers</code>, reads the four certification ledgers,
which the bind and record tools write; <code>parity/meep_gpu/test_check_hygiene.py</code>
runs all three legs as tests.

## On an NVIDIA host

CI runs the suites with no GPU library. On a host with CuPy, Triton and an NVIDIA
device (the certification round's CERT-7), run them in that platform's reference
environment (<code>environments/locks/</code>, linux-64), from a clean clone, with:

- **the user site directory off**: <code>PYTHONNOUSERSITE=1</code>, or
  <code>python -s</code>, so no distribution under <code>~/.local</code> sits ahead of
  the environment's own;
- **<code>HOME</code> an empty directory**, and no <code>CUPY_CACHE_DIR</code> or
  <code>TRITON_CACHE_DIR</code>: nothing comes from an earlier run's caches, and the
  package mints its keep cache there itself. A directory named for the keep policy that
  CuPy filled before the package marked it is refused as of unknown provenance;
- **<code>OMP_NUM_THREADS=1</code>**, as the CI workflow sets it: a MEEP oracle child
  whose material is a Python function aborts at two threads ("meep: Error in typemaps",
  a segmentation fault), because MEEP's OpenMP threads call the function concurrently;
- **<code>CUPY_ACCELERATORS</code> set and empty**: the flush policy is refused while
  CuPy's accelerated reductions may run;
- **no <code>MEEP_GPU_*</code> variable** in the shell (each suite's
  <code>conftest.py</code> pins <code>MEEP_GPU_DISPATCH=0</code> itself);
- **one card**, <code>CUDA_VISIBLE_DEVICES=&lt;card&gt;</code>, held for the run's
  duration on a shared host as the round's lanes hold theirs;
- **the environment's C compiler for Triton's launchers**:
  <code>CC=&lt;prefix&gt;/bin/x86_64-conda-linux-gnu-cc</code>. Triton compiles a launcher
  for each kernel with <code>$CC</code>, else the first <code>gcc</code> on
  <code>PATH</code>: without it the host's own compiler builds them, outside the
  environment, and a host without one fails at the first launch ("Failed to find C
  compiler"). The round's drivers set it themselves;
- **the libcuda link** the driver does not install (it ships only
  <code>libcuda.so.1</code>, and Triton links <code>-lcuda</code>): the environment's
  build makes it on a host with a driver, in <code>&lt;prefix&gt;/lib/meep-gpu-libcuda</code>
  (named in <code>&lt;prefix&gt;/share/meep-gpu-build/libcuda.json</code>); that directory
  on <code>TRITON_LIBCUDA_PATH</code> and first on <code>LD_LIBRARY_PATH</code>.

Then the three pytest runs and <code>check_failures.py</code>, as
[CI runs them](#running-them-as-ci-does). The 1.0 round measured this on 2026-10-13
(one RTX A6000, compute capability 8.6, the linux-64 reference environment, no
evidence archive): 468 failed of 17,965 run, 39 of them listed in
<code>tools/ci/pending_certification.txt</code> and 429 not. Each of the 429 is a
test's assumption about its host, not a wrong answer from the package or a kernel: 351
Metal tests that run wherever PyTorch is installed (the NVIDIA environment carries it for
Triton) and need an Apple GPU; 29 tests that skip themselves without a declared resource
where CuPy or Triton is present; 18 fixtures that hand host arrays to a CuPy launch path;
9 that open a file without an encoding after a CUDA context reset the locale; 9 that
install the keep policy directly without a cache directory the package minted; 7 that
depend on session order; 4 that assume Triton or a device is absent; 1 that finds the
host's own <code>ffmpeg</code>; and 1 at a relative difference of 1.22e-7 against its
1e-7 single-precision bound on that host's MEEP. Until those tests are fixed or
declared, <code>check_failures.py</code> does not pass on an NVIDIA host, and its
failures are read cause by cause. A run at two OpenMP threads read 14 more failures,
every one a MEEP oracle child that aborted.

## Dispatch is pinned off in the suites

Each suite's <code>conftest.py</code> sets <code>MEEP_GPU_DISPATCH=0</code> for
every test, whatever the calling shell exported. Many tests step a driver as the
array-path oracle; left to the shipped default, an oracle built with
<code>prefer_gpu=True</code> would plan through whatever kernel table the host
offers, and a byte comparison against it would compare kernels with kernels.

- A test that exercises dispatch sets the variable to <code>1</code> itself.
- A test that means "the shipped default" deletes the variable itself.
- Neither may rely on the environment it inherited.

## A skip must declare its resource

A skip and a pass are different claims, so a test may skip for one reason only:
a named external resource is absent. The resource is a tool or library (a CUDA
device, CuPy, Triton, PyTorch with MPS, MEEP) or a measurement artifact that the
repository does not carry. A test declares it in one of two ways:

    @pytest.mark.requires_resource("cupy")
    def test_something_on_a_device(): ...

    from conftest import requires_resource_skip
    requires_resource_skip("metal_fusion_board", "the board artifact is not in this checkout")

Any other skip is converted to a failure, and a <code>test_*.py</code> module
that collects zero tests fails. The session summary lists each declared resource
with the number of skips it caused.

The measurement records the harness writes are not part of the repository. A
test that reads one skips by the resource that names it. It is not removed, and
it does not pass without its artifact. A test that declares its own resource
runs again wherever its artifact is present, for example after a certification
round on the host that ran it. A test listed under <code>evidence_archive</code>
runs again where the evidence archive is restored with its marker
([The evidence archive](#the-evidence-archive)).

## The certification suite is not run by default

The ledger, weld and record contracts compare the certification records with
the files of this tree, and a few tests elsewhere wait on the same records.
Together they are the certification suite: they carry the pytest marker
<code>certification</code>, which the root <code>conftest.py</code> applies to
every test of the contract files it names (<code>CERTIFICATION_FILES</code>) and
to every test listed in <code>tools/ci/pending_certification.txt</code>. A run
without an <code>-m</code> expression deselects them and prints how many; run
them with

    python -m pytest meep_gpu parity/meep_gpu -m certification

Naming a test file on the command line runs its certification-marked tests
too, so a run of one module while editing can show a failure listed in
<code>tools/ci/pending_certification.txt</code>. That failure is expected, and
it is not fixed by regenerating a ledger.

The 1.0 certification round bound the records of this release on its bytes
([Certification status](certification.md#certification-status-of-this-release)),
and its records entry removed from <code>tools/ci/pending_certification.txt</code>
the 21 entries its records turned green, of 56. The 35 that stay are records the
round's tools did not re-cut, or contracts that wait on a later step; each is
declared under a block that names the step that removes it. The suite's failures
are reported, not skipped and not marked as expected failures, and
<code>tools/ci/check_failures.py</code> accepts exactly those listed there. A
failure of the default run is never on that list: it is a defect, or a resource
the test does not declare.

## What a run of the suites showed in the 0.9.0 preview

The 1.0 suites run per environment on the release tip (CERT-7) after this page was
written; their counts are owed to a documentation update. What follows is the 0.9.0
preview's measurement, kept for its method.

Measured on 2026-09-28 on one Apple silicon host, one suite at a time, on a
build of this release made before its last edits, whose code differs from the
published files only in comments, docstrings, the text of messages and record
notes, the names of two tests, one provenance string and one synthetic test
value. The fixes that make the suites pass on Linux and under Python 3.10 came
after it: they change the two digest helpers, one harness probe and six test
modules, and add one package test, so each package-suite count below is one
test short:

| Environment | Package suite (`meep_gpu`) | Harness suite (`parity/meep_gpu`) | Certification suite (`-m certification`) |
|---|---|---|---|
| MEEP 1.33.0 single precision (source build), PyTorch 2.10.0 with MPS | 12,921 passed, 0 failed, 532 skipped by a declared resource, of 13,453 run | 682 passed, 0 failed, 12 skipped, of 694 run | 44 failed of 363, all 44 listed in `tools/ci/pending_certification.txt` |
| conda-forge MEEP 1.33.0 (double precision), no GPU library | 12,405 passed, 0 failed, 936 skipped, of 13,341 run | 679 passed, 0 failed, 15 skipped, of 694 run | the same 44 |
| neither MEEP nor a GPU library | 11,985 passed, 0 failed, 1,356 skipped, of 13,341 run | 662 passed, 0 failed, 32 skipped, of 694 run | the same 44 |

The package suite collects 13,815 tests and the harness suite 695; the 362 and 1
not run by default are the certification suite. Without PyTorch, three package
modules (115 tests) skip whole at collection and report one result each, so the
package suite reports 13,341 results there. Every skip names its resource; 444
of them in the package suite are tests that read the evidence archive of the
certification campaigns, which this repository does not carry. The 44 failures of
the certification suite are record contracts: 40 state that a record no longer
matches the published files (a digest, a weld or a fingerprint that moved, or a
dispatch record cut before dispatch became the default), and 4 are contracts of the
two timing records that disagree with those records in the source the release
was made from as well. The examples and the installation check pass on the same
files.

<code>tools/ci/pending_certification.txt</code> is the authority for which
failures are expected; <code>tools/ci/check_failures.py</code> reads it.

## Tests that need a resource this host may not have

Besides the resources a test declares itself, <code>tools/ci/declared_resources.txt</code>
lists tests by the resource they need, and the root <code>conftest.py</code>
gives each listed test whose resource is absent the marker and a skip that says
why:

| Resource | Present when |
|---|---|
| <code>evidence_archive</code> | <code>parity/meep_gpu/results/</code> holds the evidence archive of the certification campaigns and carries its marker file, <code>EVIDENCE_ARCHIVE.json</code> ([The evidence archive](#the-evidence-archive)). The archive is not part of this repository, and a directory without the marker, such as one a gate run created, is not the archive |
| <code>torch</code> | PyTorch is installed |
| <code>meep</code> | MEEP is installed |
| <code>mps_device</code> | PyTorch is installed and reports a usable Apple GPU (MPS). A test that asserts the Metal kernels dispatch needs nothing more: every Apple GPU is supported, and by default they dispatch on it in any environment, certified or not. Run the suite without <code>MEEP_GPU_HARDWARE=certified</code> in the shell: no conftest pins it, and it refuses an environment that is not certified or cannot be judged |
| <code>single_precision_meep</code> | MEEP is a single-precision build: the test compares with MEEP at single-precision bounds, or asserts that MEEP returns float32 arrays |
| <code>meep_identity_fixtures</code> | The Tier A fixtures of MEEP identity are present and their ledger reads: <code>parity/meep_gpu/fixtures/meep_identity/meep_identity.json</code> ([MEEP identity](#meep-identity)). A checkout carries them; an installed wheel does not. <code>MEEP_IDENTITY_FIXTURES</code> names another directory |
| <code>meep_identity_reference</code> | The imported MEEP is an identity reference in keep mode, measured by <code>tools/check_install.py</code>'s identity check: MEEP 1.33.0 single precision, a build record describing the loaded images, a clean fused multiply-add census of them, re-run, subnormals kept after MEEP's switch, and the math library the record pins. <code>MEEP_IDENTITY_BUILD_RECORD</code> names the record when it is not in the MEEP prefix |
| <code>cupy_absent</code> | CuPy is not installed. The test exercises the arm a CuPy-free host takes, such as a refusal by name where the device library cannot be imported, which a host with CuPy cannot reach |
| <code>triton_absent</code> | Triton is not installed. The test exercises the arm a Triton-free host takes, such as a kernel accessor or a composer that refuses by name without Triton |

A listed test whose resource is present runs as before; the list relaxes no
test. The session summary counts the skips by resource, and names any entry of
the list that matches no collected test.

The timing ladder's tests (<code>parity/meep_gpu/test_timing_*.py</code>,
<code>test_bench_timing_case.py</code>, <code>test_bench_meep_identical_case.py</code>,
<code>test_build_3d_comparison_pairing.py</code>,
<code>test_identical_case_table.py</code>) need no GPU and run in seconds. The ones
that build a simulation declare <code>meep</code> or
<code>single_precision_meep</code> on the test itself and run MEEP in a child
process: the single-precision ones pin the geometry digest of <code>pml_3d</code>
at resolution 12 to the 2026-09-28 digest of record, and time one rank of MEEP at
resolution 6 against the package's own lift of the same simulation.

### The evidence archive

The evidence archive is the record tree of the certification and benchmark
campaigns, restored as <code>parity/meep_gpu/results/</code>. It is not part of
this repository. The harness also writes its own records into that directory
([Where records go](certification.md#where-records-go)), so the directory alone
does not show that the archive is there. The root <code>conftest.py</code>
counts the archive as present only when the directory carries the marker file
<code>parity/meep_gpu/results/EVIDENCE_ARCHIVE.json</code>:

    {"kind": "meep-gpu-evidence-archive", "archive_version": "<version>"}

| Key | Required | Value |
|---|---|---|
| <code>kind</code> | yes | <code>"meep-gpu-evidence-archive"</code> |
| <code>archive_version</code> | no | A non-empty string naming the version of the archive |
| <code>manifest_sha256</code> | no | The sha256 digest of the archive's manifest, as 64 lowercase hexadecimal digits |

No other key is accepted. Whoever restores the archive writes the marker, after
the records are in place; an archive packed for restoring carries it at its root,
so unpacking it restores the marker with the records. No gate, probe or benchmark
writes it.

- **Without the marker**, the directory is a working output directory: the listed
  tests skip, and the reason names the missing marker.
- **With a marker that is not valid** (not a file, not JSON, not an object, another
  <code>kind</code>, an undefined key, or a malformed optional value), they skip,
  and the reason says what is wrong with it.
- **With a valid marker**, they run. The probe reads the marker only: it does not
  hash the records or check them against a manifest. A record missing from a
  marked directory is not skipped by this resource: the test that reads it fails,
  unless it declares a resource of its own for that record.

## Lifted-state capsules

A capsule (<code>meep_gpu/capsule.py</code>) saves a lifted driver so that a host
with no MEEP can load it back and step it. It is the test-support object a test
against stored MEEP words is built on: the lift is made once where MEEP is
installed, and the capsule is stepped anywhere.

    from meep_gpu import capsule, lift_simulation

    with capsule.recording_construction() as recorder:
        driver = lift_simulation(sim, prefer_gpu=False)
    capsule.write_capsule("case.capsule", driver, recorder, sim=sim)
    ...
    loaded = capsule.load_capsule("case.capsule")    # no MEEP needed

What it holds, and how it loads:

- **The construction log.** <code>recording_construction</code> records the
  <code>FdtdDriver</code> constructor call and every configuration call made on
  the driver (<code>capsule.CONSTRUCTION_CALLS</code>), with their arguments as
  they were passed. The loader replays them through the driver's own constructor
  and mutators.
- **The state.** Every array reachable from the driver (fields, PML auxiliaries
  and coefficient tables, material arrays, source deposition tables, grid
  coordinates) as bytes, and every scalar as its IEEE-754 bit pattern, so no
  decimal is parsed. An array reached under two paths is stored once and stays one
  array after loading. After the replay the loader checks that the rebuilt object
  graph has the stored one's classes, attribute names, shapes, dtypes and
  aliasing, refuses by name where it does not, and writes the stored bytes and
  scalars over the rebuilt ones. <code>Capsule.load_report</code> counts what the
  replay rebuilt differently before that paste; the count is reported, not judged.
- **Callables.** A function that crosses into the driver (an
  <code>amp_func</code>, a custom <code>src_func</code>) is recorded as the table
  of calls it answered, and the loaded driver answers only those.
- **Injected currents.** <code>recording_currents(driver)</code> records every
  value each source's time function returns while the driver steps, and
  <code>load_capsule(path, currents=table)</code> (or
  <code>install_currents</code>) makes the loaded sources return exactly those
  values and refuse any other query. A custom waveform cannot be loaded without a
  table.
- **The chunk division.** The lift installs MEEP's division with
  <code>set_chunk_layout</code>, a recorded construction call, so the loaded driver
  steps the lifted simulation's own <code>ChunkLayout</code>. With <code>sim=</code>,
  MEEP's <code>structure.get_chunk_volumes()</code> boxes, the chunk owners and the
  user volume are also recorded as provenance (<code>Capsule.chunk_division</code>).

Refused by name: a driver not built under <code>recording_construction</code>, a
driver with a monitor, a callable that was not recorded, a class from outside the
package, another schema or version (only <code>capsule.SCHEMA_VERSION</code> is
read), and a member that fails its recorded sha256.

<code>meep_gpu/test_capsule.py</code> pins the round trip word for word on
hand-built drivers of each family, with NumPy alone. The identity of a fresh lift
with its capsule is checked by <code>parity/meep_gpu/capsule_identity.py</code>,
in three processes: <code>lift</code> where MEEP is installed, <code>load</code>
in a process where <code>meep</code> cannot be imported, and <code>compare</code>,
which compares the digests of every pinned array at every scheduled step and the
words at step 0 and at the last step, with a planted one-bit flip. Its
<code>--perturb-current</code> load is the control that must differ.

## MEEP identity

MEEP identity is the property that every stored float32 word of the twelve field
components (real and imaginary parts, the sign of zero included) on the cells MEEP owns
or stores as a wall equals the word of MEEP 1.33.0 single precision built without
contraction, at every compared step, and that every word <code>get_array</code>
returns equals MEEP's. It is tested in three tiers, with one schema
(<code>meep_gpu/meep_identity.py</code>):

| Tier | Where it lives | What it does | Runs where | Without its resource |
|---|---|---|---|---|
| **A**, replay | <code>meep_gpu/test_meep_identity.py</code>, fixtures in <code>parity/meep_gpu/fixtures/meep_identity/</code> (outside the wheel) | Loads each minted case's capsule with its recorded source currents injected, steps it with no MEEP, and compares the digest of every component at every compared step with MEEP's | The default package suite, every CI job | Skips by <code>meep_identity_fixtures</code> (an installed wheel) |
| **B**, the corpus record | The evidence archive (<code>parity/meep_gpu/results/meep_identity_tier_b/</code>), written by <code>parity/meep_gpu/record_meep_identity_archive.py</code> | Per reference (host and subnormal mode), every corpus row's Tier C verdict, MEEP's digests and plane digests, and the capsule and currents where the schema carries the row; <code>test_meep_identity_tier_b.py</code> reads every file back, requires every row that is not a named exclusion to hold or be refused by name, and replays the smallest rows with no MEEP | Archive hosts | Skips by <code>evidence_archive</code> |
| **C**, the MEEP lane | <code>parity/meep_gpu/meep_identity_tier_c.py</code>, <code>record_meep_identity.py</code> | Lifts a case, steps the lifted driver and MEEP in pair order, compares every word in process, and mints Tier A | A host with an identity-reference MEEP | <code>test_meep_identity_tier_c.py</code>'s live test skips by <code>meep_identity_reference</code> |

**The digests.** One digest per (case, compared step, component): the SHA-256 of the
component's words as little-endian 32-bit words, in the stored array's C order,
restricted to the compared cells. Mirror ghosts are not compared cells: MEEP stores none,
and the package's ghost storage is never read back, so their words (the known
<code>+0</code> where MEEP's reader returns <code>-0</code>) are a reported column in
Tier C, never hashed. A complex component's real and imaginary parts are interleaved.
Beside it, the digest of the whole-cell <code>get_array</code> words of each component,
and for every compared step up to 20 one 8-byte digest per x-plane of every component,
so a Tier A failure names the first step, component and plane. The compared steps are
every step 0 to 20, Table 2's ten checkpoints and its N.

**Running Tier C and minting.** With an identity-reference MEEP's interpreter, from the
repository root:

    python parity/meep_gpu/meep_identity_tier_c.py compare --out DIR [--cases a,b] [--mode keep|flush]
    python parity/meep_gpu/meep_identity_tier_c.py replay --source DIR [--watch]
    python parity/meep_gpu/record_meep_identity.py --build-record PATH --work DIR [--mode keep|flush]
        [--cases-file parity/meep_gpu/meep_identity_tier_a.py]

<code>compare</code> runs each case in its own process: the lift and its capsule, the
pair-order stepping (the driver to the step, recording every source current, then MEEP
to the same step with <code>sim.run(until=(m - 1/2) dt)</code>), the word compare of
MEEP's stored words on the package's lattice, read exactly (below), the
<code>get_array</code> compare, the FPU mode at every compared step, a planted one-ulp
flip that the word compare, the digest and its plane must each catch, and the four probe
cases' smoke record (<code>meep_identity_smoke.json</code>) reproduced by MEEP's own
words. <code>replay</code> is Tier A on such a directory, in a process that cannot import
MEEP; <code>--watch</code> adds the complex-multiply count. <code>record_meep_identity.py</code>
is the only writer of the fixtures and of <code>meep_identity.json</code>: it judges the
MEEP with the identity check, runs both, writes nothing unless every claim holds (0
value- and 0 word-differing compared words, 0 differing reader words, every flip caught
and placed, every exact MEEP word reproducing MEEP's point read, no compared cell
without a MEEP word, every digest reproduced by the replay, its inputs unchanged, the
fixtures within the 5 MB budget), and then replaces only its own mode's set.

**MEEP's stored words, read exactly.** Each word is copied, with no arithmetic, from the
array MEEP steps: <code>fields_chunk::f[c][0]</code> (and <code>f[c][1]</code> on complex
storage) of the chunk that owns the point, at that chunk's
<code>grid_volume::index(c, ivec)</code>; the owner is the chunk
<code>fields::get_field</code> would read (the first that owns the point, identity
symmetry, no wrap). A cell no chunk owns but a chunk's array holds (the low plane of a
metallic wall, which MEEP never writes after allocating it as zero, or the not-owned low
plane of a Bloch axis, which the chunks' connections fill) is read from the first chunk holding
it, and every chunk holding it must store the same word. One held word is not B's: MEEP
keeps B and H in one array and gives H its own only where it needs one (mu, a
polarization, a PML conductivity along the component), so at a not-owned point of a chunk
whose B array is its H array the word is the one H's connection wrote last. Where that word
is not B's point read, the reader reads B's point read (MEEP's B: the owned image through
its Bloch phase), the shared word must be H's point read, and on complex storage the point
is refused. Nor is a held B word of a chunk that took an H array of its own in the step
just run: MEEP builds that step's connections while the chunk's H is still its B, so no B
reaches the point in that step and the word is the one it held before. Where it is not its
lattice image's word, it must be the word read one step earlier, and the reader reads the
image's exact word (the point one lattice vector up on each axis where it sits on the low
plane, with a zero Bloch wavevector there, on a Cartesian grid with no symmetry). MEEP's point read
(<code>fields::get_field</code>) cannot be the reader: it returns
<code>(1, +0) * word</code> in <code>complex&lt;double&gt;</code>, which drops an
imaginary <code>-0</code> beside a real part <code>&gt;= +0</code> (and a real
<code>-0</code> beside an imaginary part <code>&lt;= -0</code>). It is still read at every
point as the exact read's check: applied to the exact words it must reproduce
<code>get_field</code> word for word at every owned cell. Ghost cells and the smoke record
keep MEEP's point read.

**The corpus.** <code>parity/meep_gpu/meep_identity_corpus.py</code> is a
<code>--cases-file</code> holding Table 2's measured rows: <code>manifest</code> derives
its manifest from the seven 2026-08-09 legs' <code>parity.jsonl</code> files, and each row
is rebuilt the way it was measured (captured by the corpus harnesses' own code from
<code>MGPU_SITE_MEEP_SOURCE</code>, the row's capped resolution, its step count, its grid
checked after the lift). Rows whose capsule the schema cannot carry (DFT monitors) are
still word-compared and record the refusal; <code>MEEP_IDENTITY_CASE_TIMEOUT</code> raises
the per-case ceiling for the largest rows. Two counts are reported and never judged:
<code>stored_read_equivalent</code>, the compare a point-read Tier C made (MEEP's point
read applied to the package's words, against MEEP's point read), and
<code>meep_point_read_loss</code>, the compared words MEEP stores that its point read does
not return (signed zeros of complex words).

**Tier A's cases.** The minting tool replaces a mode's whole set in one run and reads its
cases from one <code>--cases-file</code>, so Tier A is minted through
<code>parity/meep_gpu/meep_identity_tier_a.py</code>: the four probe cases (built by
<code>meep_identity_smoke</code>, stepped to the runner's stock step rule) together with
one small corpus row per family the array path transcribes (<code>FAMILY_ROWS</code>,
built by the corpus module, which needs <code>MEEP_IDENTITY_CORPUS_MANIFEST</code> and
<code>MGPU_SITE_MEEP_SOURCE</code> at mint time; the replay needs neither). The keep set is
minted on Apple silicon against the contract-off reference. The flush set is minted on
x86-64 Linux against the Linux reference (MEEP 1.33.0 single precision, MEEP's own flush
on import, <code>OMP_NUM_THREADS=1</code>; the Linux MEEP passes the identity check in
flush with a build record written by
<code>timing_records.py meep-build-record --require-identity</code>). The minting tool's
replay runs in a process that cannot import MEEP; it reaches flush through the package's
x86-64 glibc lever (MXCSR FTZ and DAZ set on the replaying thread, the two bits MEEP's
switch sets), so the tool judges the replay exactly as it judges a keep one. Both sets
hold the same eight cases: 39,991,116 stored and 65,610,420 reader words equal to MEEP's
in each, 2,318,064 bytes (keep) and 2,318,104 bytes (flush), 4,636,168 of the 5 MB budget.
Both sets were re-minted on the frozen 1.0 bytes (2026-10-11), again at the freeze after
per-kernel certification (2026-10-12), whose lift moved, again at the R22 freeze
(2026-10-14), whose lift, capsule module and deposit moved, and again at the R27 freeze
(2026-10-15), in the new reference environments. Each replays
bit-identically, with MEEP not importable, on x86-64 Linux (flush under the lever) and
on Apple silicon (flush under FZ): 16 of 16 rows on each host.

**Table 2 and Tier B.** Table 2 is the corpus judged by Tier C on four references (Apple
silicon keep and flush against the contract-off build, Linux keep and flush against the
Linux build, all at <code>OMP_NUM_THREADS=1</code>), one
<code>compare --cases-file parity/meep_gpu/meep_identity_corpus.py --cases LABEL</code>
process per row. Then, from the repository root:

    python parity/meep_gpu/record_meep_identity_archive.py --archive parity/meep_gpu/results/meep_identity_tier_b \
        --set mac_keep --mode keep --source DIR [--exclude LABEL=REASON] [--replay] [--link]
    python parity/meep_gpu/meep_identity_context_l2.py run --labels FILE --out DIR
    python parity/meep_gpu/build_meep_identity_table.py --archive parity/meep_gpu/results/meep_identity_tier_b \
        --manifest FILE --context-l2 DIR/rows.jsonl --out-md docs/development/meep-identity-corpus.md

<code>record_meep_identity_archive.py</code> mints Tier B: it judges each row by the Tier C
half of the Tier A minting criteria (<code>holds</code>, <code>differs</code>,
<code>error</code>, or <code>refused</code> with the reason), keeps a named exclusion's
measured status beside its reason, archives the row's files in the runner's layout with
their sha256, and with <code>--replay</code> replays the holding capsule rows with no MEEP.
MEEP's words are not archived: the runner never writes them. <code>meep_identity_context_l2.py</code>
measures Table 2's one remaining number, now context: the whole-field relative L2 of the
array path against the certification MEEP of the Mac lock, which contracts multiplies and
adds, so where a row holds it is MEEP's own build-to-build spread.
<code>build_meep_identity_table.py</code> writes
[Table 2: MEEP identity on the corpus](meep-identity-corpus.md) from those records alone:
per row and reference "0 of W / 0 of V" (or the differing counts), the ghost words
reported, the planted flip, the context L2 beside the L2 of record, the named exclusions
and refusals, and L0, the two reference MEEPs' digests against each other per mode.

**L3.** The Tier B record is also what a run with compiled kernels dispatched is judged
against, word for word, with MEEP imported for the lift and never stepped:
<code>parity/meep_gpu/rounds/1.0/l3_corpus.py</code>
([The direct corpus check against MEEP](certification.md#the-direct-corpus-check-against-meep-l3)).

**The corpus environment.** Each reference runs in its host's certification environment
(<code>environments/locks/</code>) as it is, with nothing beside it: the environment
carries what some rows import or call and no field update reads, the corpus-only packages
of <code>environments/locks/numerics-relevant.json</code> (<code>corpus_only</code>).
shapely and nlopt are pinned by SHA-256 in each platform's pip lock; the PyMieScatt
stand-in (PyMieScatt 1.8.1.1 with <code>trapz</code> taken from NumPy) is built from its
pinned sdist and two-line patch by
<code>parity/meep_gpu/rounds/1.0/build_pymiescatt_standin.py</code>; the build scripts
install both (step 7) and link the imageio-ffmpeg pin's executable (in each platform's pip
lock) into <code>&lt;prefix&gt;/bin/ffmpeg</code> (step 7b), for
<code>test_animation_output</code>.
<code>tools/compare_reference_environment.py</code> lists them apart from the
numerics-relevant packages and never counts them in its verdict, so an environment without
them is still judged the reference on its numerics; <code>--require-pip-lock</code> requires
them. The 1.0 runbook (<code>parity/meep_gpu/rounds/1.0/RUNBOOK.md</code>, "CERT-1: the
corpus environment") says how an environment built before they joined the lock takes them.
A row whose script lacks one is refused by name, never measured.

**Subnormal modes.** A set is minted in <code>keep</code>, the reference of both hosts,
or in <code>flush</code>, which exists only on x86 at <code>OMP_NUM_THREADS=1</code>
(MEEP's <code>set_zero_subnormals(True)</code>). <code>compare</code> also runs
<code>flush</code> on Apple silicon: MEEP and the package step under FZ, installed on the
stepping thread by the package's lever (<code>subnormal_policy</code>), which stores the
x86 flush reference's words (R13); the stored-word reads leave the mode, so a subnormal
word is never read as zero. A set is judged only in its mode: the
replay drives the host to it, through MEEP's switch when MEEP is imported and, with no
MEEP imported, the package's own lever (Darwin's flush environment on Apple silicon, FZ;
MXCSR FTZ and DAZ through glibc's <code>fenv_t</code> on x86-64 Linux), measures it at
every compared step, and refuses by name otherwise, naming the system, on any other
platform. A flush set replayed on x86-64 Linux is the array path under FTZ and DAZ, the
reference's own mode, judged for identity; replayed on Apple silicon it is the Mac array
path under FZ judged against the x86 flush reference, row by row (owner ruling R13).

**What goes red, and when.** Tier A goes red when the array path moves a word on a
minted case. <code>test_the_ledger_records_the_live_array_path_oracle</code> goes red
when the array path's behavioural fingerprint (<code>array_path_oracle</code>) moves,
whether or not a minted case reaches the change: an array-path change re-proves identity
with Tier C and re-mints in the same change, or, while a change is in progress, carries
the test in <code>tools/ci/pending_certification.txt</code> until it does.

**Two more checks on Tier A.** The replay runs once more under a complex-multiply
interceptor (<code>meep_identity.watching_complex_multiplies</code>): the driver's
arrays are viewed as an ndarray subclass that records every multiply of two complex
operands. It observes operations on arrays that descend from the driver's state, not
arrays a step allocates and fills by assignment, so its count is a lower bound and the
test also requires that it observed multiplies at all. The CI job without MEEP reruns
Tier A with <code>NPY_DISABLE_CPU_FEATURES</code> set to NumPy's AVX-512, AVX2 and FMA3
loops: the digests must not move with the SIMD loops NumPy dispatches.

## Long runs report progress

A test, gate or benchmark that can run for more than about a minute prints a
flushed line per unit of work: which case of how many, the elapsed time, and the
number the case produced. Run long pytest cases with <code>-s</code>. A run
whose only signal is its exit code cannot be told from a hang. The timing ladder
is the long-run example to copy: one flushed line per row start and end in
<code>progress.log</code>, the ledger appended as each row lands, a host trace
every few seconds, and <code>--resume</code> to continue an interrupted run
([Running the timing ladder](timing.md)).

## Conventions

- **Write the gate before the kernel.** A kernel with no gate is a claim with no
  measurement.
- **A gate must be able to fail.** Every gate pairs its verdict with a null
  control: a planted defect it must catch, or an unguarded leg that must
  diverge.
- **Say what a measurement does not support.** Each benchmark and each
  certification record states its own limits.
- **Rename outright.** A renamed function leaves no alias behind.
