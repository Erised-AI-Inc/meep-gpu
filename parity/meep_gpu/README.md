# Certification and benchmark harness

This directory is not installed with the package, and using the package needs
nothing from it. It holds the scripts that certify each kernel family against the
package's own array path, the comparisons against MEEP, and the benchmark drivers.
The manual describes the certification model
(`docs/development/certification.md`); this page says how the directory is laid
out and how to point its drivers at a machine.

## Layout

| Files | What they are |
|---|---|
| `gate_*.py` | Device gates. Each runs a kernel family on a GPU beside the array path and writes `gate.json`. |
| `probe_*.py`, `measure_*.py`, `replay_*.py`, `stress_*.py` | Measurements and diagnostics behind a gate or a coverage figure. |
| `recut_metal_gates.sh`, `run_metal_dispatch_campaign.sh` | Drivers of the Metal gate fleet and of the Metal route campaign (`docs/development/certification.md`). |
| `rounds/`, `build_metal_round_inputs.py` | One directory per certification round on another machine: its driver, and the manifest of the evidence-archive inputs its gates read (the bundles themselves are not committed). The Metal one is packed, verified and unpacked by `build_metal_round_inputs.py`. |
| `build_*_fusion_matrix.py`, `build_3d_comparison.py` | Builders of the coverage boards and the throughput table from recorded rows. |
| `build_parity_board.py`, `parity_board_text.py`, `parity_board_emit.py`, `parity_board_emit.json`, `parity_board_rows.json`, `record_dev_evidence.py`, `parity_board/` | The bit-parity coverage board: per kernel table, how many modules have the chunk-record select, MEEP's A forms, host and device word identity with the array path, development evidence on the device and a certification on the live array-path oracle, and per array-path family, identity with MEEP under each subnormal baseline. The rows are the kernel design's step lists, held as data; every column is read off the tree (the device text each module's builder emits with no device, a recorded run of the module tests, the pending list, the ledgers, the integrators' development-evidence record, the Tier A/C record), and a row or column it cannot measure is refused by name. `outcomes` records a test run against the tree's subject manifest and `build` refuses a record whose subject is not the tree; `record_dev_evidence.py` refuses a device leg whose module subject moved during the run. |
| `rebind_*.py`, `recut_*.py`, `mint_*.py`, `seed_*.py`, `record_*.py` | Writers of the ledgers under `meep_gpu/*_kernels/`. `record_array_path_oracle.py` writes the array-path fingerprint into all three, and with `--verify` recomputes it and writes nothing. |
| `neutral_labels.py`, `neutralize_ledger_labels.py` | The neutral labels those writers record for the machine a gate ran on (`the GPU host`, `<staging-root>/...`), never its host name or home paths, which stay in the run's artifact; and the rewrite of the label leaves an earlier bind left, the bind's last command. `tools/ci/check_hygiene.py --legs ledgers` checks the result. |
| `benchmark_*.py`, `bench_*.py` | Throughput drivers. |
| `timing_ladder.py`, `timing_host.py`, `timing_records.py` | The identical-case timing ladder: the GPU routes and stock MEEP interleaved on one host behind a quiet gate, its host readers and its recorders (`docs/development/timing.md`). |
| `timing_cases.py` | The timing cases, each defined once and built by both benches. |
| `bench_meep_identical_case.py`, `bench_timing_case.py`, `digest_harness_lift.py`, `gpu_array_control.py` | Stock MEEP on a timing case; `bench_fused_products.py` on a case of `timing_cases.py`; the geometry digest of the package's own lift; the array-path control as its own process. |
| `build_identical_case_table.py`, `audit_timing_validity.py` | The table of a ladder, from its ledger; a read-only audit of its windows, passes and host state. |
| `gate_needles.py`, `record_gate_needle_anchors.py`, `gate_needle_anchors.json` | The per-gate mutation needle tables, and the record of the device text each needle hands the compiler, derived host-side and checked by `test_mutation_anchors.py`. Re-mint the record with `record_gate_needle_anchors.py --write` in the change that moves it. |
| `capsule_identity.py` | The capsule's own identity check: a fresh lift against `load(capsule)`, every pinned array's digest at every scheduled step and the words at step 0 and N, in a lift process, a load process that cannot import MEEP, and a comparer (`docs/development/testing.md`, "Lifted-state capsules"). |
| `complex_product_interceptor.py` | The runtime check that the array path forms no complex product: a NumPy-reference driver's arrays viewed as an `ndarray` subclass whose `__array_ufunc__` counts every multiplicative ufunc with a complex operand (a violation), the additive complex traffic beside it, and a planted product that must be caught. Checked by `test_complex_product_interceptor.py`; the complex-family identity legs run under it. |
| `b_battery_states.py` | Constructors of the B battery's gate states: full-volume random states, the planted −0 curl and constitutive cases, the ±0 special-word lattice, and the form-coverage check. |
| `gate_chunk_records.py` | The chunk records a gate hands a Triton plan it builds from bare arrays, where the harness route has no default: `layer_record`, the one-rank division of an engine layer, for a reference that is the array path on it, and `every_cell_record`, every cell in a layer chunk, for a hand reference that steps the split-field recurrence on every cell. |
| `test_*.py` | Tests of the harness itself. |
| `build_meep_133_linux.sh`, `build_meep_133_macos.sh` | The reference environment of each platform, built end to end: the locks, MEEP 1.33.0 (single precision) from the release tarball, the identity corpus and the build's record, see below. |
| `build_meep_133_record.py` | The build scripts' Python steps: fetch and corpus (with its siblings, and the refresh of an existing prefix's corpus), load commands, the ffmpeg and libcuda links, the programs, the identity verdicts and `reference_build.json`. |
| `build_meep_133_native.sh` | A native-tuned MEEP build with its build record, a timing control only. |
| `meep_identity_smoke.py`, `meep_identity_smoke.json` | MEEP-only digest check of an identity reference: four cases, every stored word hashed, against the recorded digests (written only by its `--mint`). |
| `meep_identity_tier_c.py` | Tier C of MEEP identity: `compare` lifts a case, steps the lifted driver and MEEP in pair order and compares every stored and `get_array` word in process, with a planted 1-ulp flip; `replay` is Tier A on its output in a process that cannot import MEEP (`docs/development/testing.md`, "MEEP identity"). |
| `meep_identity_corpus.py` | Table 2's corpus as a Tier C `--cases-file`: `manifest` derives one entry per measured row from the seven legs' `parity.jsonl`, and `build` rebuilds a row as it was measured (capture, capped resolution, step count, grid checked after the lift). |
| `record_meep_identity.py`, `fixtures/meep_identity/` | The only writer of the Tier A fixtures and their ledger `meep_identity.json`: judges the MEEP as an identity reference, runs Tier C and the replay, and writes one mode's set only when every claim holds and the fixtures fit the 5 MB budget. |
| `meep_identity_tier_a.py` | Tier A's cases as one `--cases-file`: the four probe cases of `meep_identity_smoke.py` and one small corpus row per family (`FAMILY_ROWS`), minted together because a mint replaces a mode's whole set. |
| `record_meep_identity_archive.py`, `test_meep_identity_tier_b.py` | Tier B: judges each corpus row's Tier C output by the Tier A minting criteria (holds, differs, error, refused by name; named exclusions kept apart), archives MEEP's digests, the plane digests, the result and the capsule and currents in the evidence archive with their sha256, and replays the holding capsule rows with no MEEP; the test reads that record back (`evidence_archive`). |
| `meep_identity_context_l2.py` | Table 2's context column: the whole-field relative L2 (`sweep_corpus_lift_parity.compare_fields`) of the array path against a MEEP that contracts, per corpus row at its N. |
| `build_meep_identity_table.py` | Table 2 as MEEP identity (`docs/development/meep-identity-corpus.md`), from the Tier B record, the corpus manifest and the context L2 rows: per row and reference "0 of W / 0 of V", the named exclusions and refusals, and L0 between the two reference MEEPs. |

Runs write to `results/` in this directory. That tree is not tracked and is not
part of this repository. Ledgers and messages cite artifacts by their path under
`results/`; those paths name where a record was written, and the artifacts
themselves are archived separately. By the suites' convention a test that needs
one is skipped by a declared resource when it is absent. The records a run writes
do not count as the archive: the archive is present only when `results/` carries
its marker file, `EVIDENCE_ARCHIVE.json` (`docs/development/testing.md`, "The
evidence archive").

Some ledger records cite a path that begins with two directories above
`parity/`. Those two directories name the root of the tree the record was written
in, and correspond to the root of this repository.

Commit hashes in the records name commits of the development history, which the
public release repository does not carry; records bind files by sha256.

## Pointing the drivers at a machine

The drivers read their site settings from the environment. Every setting has a
neutral default.

| Variable | Meaning | Default |
|---|---|---|
| `MGPU_SITE_PYTHON` | Interpreter the drivers launch. It must import `meep`, and `cupy` or `torch` for a device leg. | `python` |
| `MGPU_SITE_STAGING_ROOT` | Directory under which a round is staged; each driver uses its own subdirectory. | `$HOME/meep_gpu_validation` |
| `MGPU_SITE_SOURCE_ROOT` | Root of the staged copy of this repository that the round runs. | `<run root>/source`; the two Metal drivers default to the checkout they are in |
| `MGPU_SITE_MEEP_SOURCE` | A MEEP source checkout. Its `python/examples` and `python/tests` are the corpus the surveys and sweeps read. | `~/meep` |

The names do not begin with `MEEP_GPU_` on purpose. That prefix belongs to the
package's own switches, and the end-to-end dispatch record refuses a run whose
environment sets a variable with that prefix
(`recut_driver_dispatch_record.LICENCE_ENVIRONMENT_REFUSED_PREFIXES`).

Drivers that accepted `RUN_ROOT`, `ROOT`, `PYTHON` or `PY` still do; the variables
above supply the defaults.

The timing ladder reads its parameters from `MGPU_TIMING_<KEY>` variables as well
as from flags and a `--params` file (`MGPU_TIMING_GPU`, `MGPU_TIMING_RUN_ROOT`, ...;
`timing_ladder.py --dry-run` lists every key with the source of its value). Its
interpreter falls back to `MGPU_SITE_PYTHON` and its run root to
`$MGPU_SITE_STAGING_ROOT/timing`.

**`KMP_DUPLICATE_LIB_OK`.** Leave it unset in your shell, as INSTALL.md says.
The harness was developed in an environment that loaded two OpenMP runtimes, and
some of it still sets the variable for the processes it starts: the two Metal
campaign drivers (`recut_metal_gates.sh`, `run_metal_dispatch_campaign.sh`), the
gate kits and several test and gate modules, and the usage lines of 16 scripts
whose bytes the certification ledgers pin. In an environment installed per
INSTALL.md a process holds one OpenMP runtime and does not need it; with it set,
a process that imports PyTorch before MEEP crashed in 3 of 3 runs on a Mac.
The pinned usage lines keep it until those files are next certified.

### Running a gate on a CUDA host

The per-family launchers the NVIDIA campaigns were run with (batch-scheduler jobs
and single-host shell drivers) are not part of this repository: they encode one
site's scheduler, devices and staging directories. The certification records
still name them as provenance. What they did, and what a gate run needs on any
host, is:

- **One device, idle.** Pin it with `CUDA_VISIBLE_DEVICES=<index>` and check
  immediately before the run that no other process holds it; a gate that
  shares a device measures the other job as well.
- **One CuPy kernel cache per subnormal policy.** CuPy's cache key does not see
  the flush-to-zero setting, so the two policies must not share a directory. The
  cache of the `keep` policy must carry `ftz_stripped` in its path, and the
  `flush` cache must not (`meep_gpu.subnormal_policy.cupy_cache_reasons` states
  the rule, and a gate refuses at start-up when it is broken):

  ```sh
  export CUPY_CACHE_DIR="$HOME/meep_gpu_validation/<round>/cupy_cache_ftz_stripped_keep"   # keep
  export CUPY_CACHE_DIR="$HOME/meep_gpu_validation/<round>/cupy_cache_flush"               # flush
  ```

- **A fresh Triton cache per run and policy**: `TRITON_CACHE_DIR` pointing at an
  empty directory.
- **The repository root on `PYTHONPATH`**, and unbuffered output to a log
  (`python -u gate_<family>.py ... > <round>/gate.log 2>&1`), so that the log is
  the status of a run in progress.

## Building the reference MEEP

`build_meep_133_linux.sh` and `build_meep_133_macos.sh` build the one environment
the tests, a certification round and the timing run in on each platform, end to end:
the conda packages installed without a solve from the platform's lock under
`environments/locks/` (refused unless the new prefix holds exactly the lock's
artifacts), MEEP 1.33.0 from the release tarball in single precision, with MPI and
libGDSII, the PyPI packages from the pip lock, and ffmpeg linked from its pinned wheel,
into a new Conda environment (`MEEP_ENV_NAME`, default `meep-gpu-ref`, under
`CONDA_ROOT`). Each script re-runs itself under `env -i`, so nothing of the calling
shell reaches the build, and refuses by name the knobs it no longer has
(`MEEP_SIGMA_PATCH`, `MEEP_REFERENCE_SOLVE`, `MEEP_BUILD_ROLE`, `MEEP_FFMPEG`).

Each builds an **identity reference**, the MEEP the array path is held to word for
word, and refuses to finish unless it is one. Both compile with
`-O2 -ffp-contract=off -fno-fast-math`, the Linux script with
`-march=x86-64 -mtune=generic` beside them; step 8 asserts what the environment claims
to be (MEEP 1.33.0, single precision, MPI, a GDSII file read, the drift report of
`tools/compare_reference_environment.py --require-reference`, and the fused
multiply-add census of the installed `libmeep` and `_meep`), step 10 runs
`meep_identity_smoke.py`, which steps MEEP alone on four cases and compares the SHA-256
of every stored word with `meep_identity_smoke.json`, and step 11 takes the identity
verdicts in keep and flush. Before building, each keeps the tarball's `python/examples`
and `python/tests` in `<prefix>/share/meep-gpu-build/meep-1.33.0/python` as the MEEP
identity corpus, with the SHA-256 of every file, and beside it the corpus siblings
(`build_meep_133_record.CORPUS_SIBLINGS`: the files outside `python/` that corpus rows
read, found by a grep of the corpus and a file-access trace of every row), and at the end
writes `<prefix>/share/meep-gpu-build/reference_build.json`, the build's record: the
tarball's digest, the flags, the census, the corpus list and siblings, the locks' digests,
the drift verdict and the identity verdicts. `--refresh-corpus` brings an existing
prefix's corpus to what the build keeps now and rewrites only the record's `corpus` entry. `build_meep_133_native.sh` passes `--timing-control`, under
which the census is recorded and not judged and the digest check is skipped. The digest
record is written only by `meep_identity_smoke.py --mint`, which steps each case twice
in two processes and writes nothing unless the first run equals the minting reference's
stored words word for word, the second reproduces every digest, and a planted 1-ulp
flip is caught.
`environments/locks/README.md` says what each lock holds and how it is re-cut
(`environments/locks/recut_locks.py`). docs/development/certification.md says when a
round uses the environments, and INSTALL.md ("Building MEEP in single precision") how
they are built.

## References in comments

Comments and docstrings in the package and in this directory cite "the design
notes", "the development notes", "the test coverage notes", "the corpus notes"
and "the parity matrix", sometimes with a document name in parentheses. These are
development documents that are not part of this repository. The public record is
the manual under `docs/` and the ledgers beside the kernels
(`meep_gpu/*_kernels/fingerprints.json`, `meep_gpu/cuda_kernels/certification.json`).
