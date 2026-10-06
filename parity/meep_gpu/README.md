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
| `rebind_*.py`, `recut_*.py`, `mint_*.py`, `seed_*.py`, `record_*.py` | Writers of the ledgers under `meep_gpu/*_kernels/`. |
| `benchmark_*.py`, `bench_*.py` | Throughput drivers. |
| `timing_ladder.py`, `timing_host.py`, `timing_records.py` | The identical-case timing ladder: the GPU routes and stock MEEP interleaved on one host behind a quiet gate, its host readers and its recorders (`docs/development/timing.md`). |
| `timing_cases.py` | The timing cases, each defined once and built by both benches. |
| `bench_meep_identical_case.py`, `bench_timing_case.py`, `digest_harness_lift.py`, `gpu_array_control.py` | Stock MEEP on a timing case; `bench_fused_products.py` on a case of `timing_cases.py`; the geometry digest of the package's own lift; the array-path control as its own process. |
| `build_identical_case_table.py`, `audit_timing_validity.py` | The table of a ladder, from its ledger; a read-only audit of its windows, passes and host state. |
| `test_*.py` | Tests of the harness itself. |
| `build_meep_133_linux.sh`, `build_meep_133_macos.sh`, `meep-sigma-reader.patch` | Source build of the reference MEEP 1.33.0 (single precision), see below. |
| `build_meep_133_native.sh` | A native-tuned MEEP build with its build record, a timing control only. |

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

Commit hashes in the records name the pre-publication development history, which is
not part of this repository; records bind files by sha256.

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

`build_meep_133_linux.sh` and `build_meep_133_macos.sh` build the MEEP every
certification is measured against: 1.33.0 from the release tarball, in single
precision and with MPI, into a new Conda environment (`MEEP_ENV_NAME`, default
`meep-gpu-ref`, under `CONDA_ROOT`), installed without a solve from the platform's lock
files under `environments/locks/` (the environment the kernels were certified in on
that platform, with the certified GPU libraries). Each refuses to build unless the new
prefix holds exactly the lock's conda packages, and refuses to finish unless what it
built is what it claims to be, the last check being
`tools/compare_reference_environment.py --require-reference`. `MEEP_REFERENCE_SOLVE=1`
solves instead and labels the result as not the reference.
`environments/locks/README.md` says what each lock reproduces and when it is re-cut.
docs/development/certification.md says when a round uses them, and INSTALL.md
("Building MEEP in single precision") how they are run.

The macOS script, alone, takes an opt-in patch:
with `MEEP_SIGMA_PATCH=1` it applies `meep-sigma-reader.patch` before building.
The patch adds `fields.get_susceptibility_sigma` and three related readers to
MEEP. The package uses them, when the MEEP it runs beside has them, to read the
per-point susceptibility strength of structured dispersive media directly; three
refusal messages of `meep_gpu/from_meep.py` name this build as one way to lift a
case they refuse. Stock MEEP lifts uniform dispersive media and axis-aligned
structured dispersive cells without the patch. A patched build changes MEEP's
source, so it is not the certification reference: build that without the patch.

## References in comments

Comments and docstrings in the package and in this directory cite "the design
notes", "the development notes", "the test coverage notes", "the corpus notes"
and "the parity matrix", sometimes with a document name in parentheses. These are
development documents that are not part of this repository. The public record is
the manual under `docs/` and the ledgers beside the kernels
(`meep_gpu/*_kernels/fingerprints.json`, `meep_gpu/cuda_kernels/certification.json`).
