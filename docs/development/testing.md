# Running the tests

## The two suites

Tests sit beside the code they test. Run each suite from the repository root,
one at a time:

    python -m pytest meep_gpu/ -q
    python -m pytest parity/meep_gpu/ -q

| Suite | What it tests | Size in this release, 2026-09-28 |
|---|---|---|
| <code>meep_gpu/</code> | The package: the array path, the lift, the planner, and every kernel table's predicates and emitters | 13,816 tests collected, of which 362 belong to the certification suite; 27 minutes on one Apple laptop with MEEP and PyTorch, 6 minutes with neither |
| <code>parity/meep_gpu/</code> | The harness: the gates' helpers, the record writers, the timing and census tools | 695 tests collected, of which 1 belongs to the certification suite; about 1 minute on the same laptop |

On that laptop, with MEEP and PyTorch installed, the default run of the package
suite read 12,921 passed, 0 failed and 532 skipped by a declared resource, and the
harness suite 682 passed, 0 failed and 12 skipped. [What a run of the suites shows in this
preview](#what-a-run-of-the-suites-shows-in-this-preview) gives the same
runs with a double-precision MEEP and with neither MEEP nor a GPU library.

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

The records in this release were cut before the published files were finalized:
every record that pins a file the release edited no longer matches it. So the
certification suite fails until the
[certification round](certification.md) has run on the published
files. Its failures are reported, not skipped and not marked as expected
failures, and <code>tools/ci/check_failures.py</code> accepts exactly those
listed in <code>tools/ci/pending_certification.txt</code>. A failure of the
default run is never on that list: it is a defect, or a resource the test does
not declare.

## What a run of the suites shows in this preview

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
| <code>mps_device</code> | PyTorch is installed and reports a usable Apple GPU (MPS) |
| <code>single_precision_meep</code> | MEEP is a single-precision build: the test compares with MEEP at single-precision bounds, or asserts that MEEP returns float32 arrays |
| <code>certified_metal_toolchain</code> | PyTorch and the Metal frontend are a certified pair: the test asserts that the Metal kernels dispatch, which an uncertified pair refuses by design |

A listed test whose resource is present runs as before; the list relaxes no
test. The session summary counts the skips by resource, and names any entry of
the list that matches no collected test.

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

## Long runs report progress

A test, gate or benchmark that can run for more than about a minute prints a
flushed line per unit of work: which case of how many, the elapsed time, and the
number the case produced. Run long pytest cases with <code>-s</code>. A run
whose only signal is its exit code cannot be told from a hang.

## Conventions

- **Write the gate before the kernel.** A kernel with no gate is a claim with no
  measurement.
- **A gate must be able to fail.** Every gate pairs its verdict with a null
  control: a planted defect it must catch, or an unguarded leg that must
  diverge.
- **Say what a measurement does not support.** Each benchmark and each
  certification record states its own limits.
- **Rename outright.** A renamed function leaves no alias behind.
