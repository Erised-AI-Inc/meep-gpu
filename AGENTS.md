# Instructions for coding agents

Commands and rules for coding agents and other automated contributors. The
user's own instructions take precedence; people start at [README.md](README.md).
`meep-gpu` (import `meep_gpu`) time-steps a simulation the user's own MEEP has
built, on one GPU: NVIDIA through CuPy (Triton or hand-written CUDA kernels),
Apple through PyTorch MPS (Metal kernels), or a NumPy reference on any host.
MEEP keeps setup and post-processing; the `meep` module is not replaced.

## Helping someone run a simulation

**Install.** MEEP is `pymeep` on conda-forge, never `pip install meep` (an
unrelated project). This package is not on a package index; it installs from a
checkout. Follow one route of [INSTALL.md](INSTALL.md) in order, and stop at the
first command that fails:

| Machine | Route |
|---|---|
| Linux x86_64, one NVIDIA GPU | `environments/nvidia-linux.yml`, then step 3 of [Route 1](INSTALL.md#route-1-one-nvidia-gpu-on-linux) (`libcuda.so`) |
| Apple silicon Mac | `environments/apple-silicon.yml` ([Route 2](INSTALL.md#route-2-an-apple-silicon-mac)) |
| No GPU | `environments/reference-cpu.yml` ([Route 3](INSTALL.md#route-3-no-gpu-the-numpy-reference-only)) |
| An environment that already holds MEEP | [that section](INSTALL.md#installing-into-an-existing-meep-environment) |

In an environment made from a file, install with `python -m pip install .` and
no extras. Then run `python tools/check_install.py --require-gpu` (without the
flag on a host with no GPU). A last line starting `OK:` passes; only
`OK: MEEP and meep-gpu work together on this host's GPU.` means compiled kernels
served the run. A line starting `FAILED:` names the section of INSTALL.md that
fixes it; read that section instead of improvising.

**Port the script.** `sim.run(until=T)` becomes:

```python
from meep_gpu import gpu_compatibility, run_on_gpu

verdict = gpu_compatibility(sim)     # before any run; does not initialize MEEP
print(verdict.supported, list(verdict.reasons))
result = run_on_gpu(sim, until=T)    # instead of sim.run(until=T)
try:
    ez = result.get_array("Ez")      # MEEP's array layout
    spectrum = result.get_flux_spectrum(flux)   # flux = sim.add_flux(...)
finally:
    result.close()
```

`run_on_gpu` and `lift_simulation` default to `prefer_gpu=True`, this host's
GPU, and raise on a host with none. `prefer_gpu=False` is the NumPy reference:
for checking results, not for speed. `lift_simulation(sim)` returns the driver,
for runs in segments. Copy from `examples/`.

**Read what ran; never infer it from timing.** After at least one step:
`result.driver.gpu` (`"cuda"`, `"metal"` or `None`),
`result.driver.active_step_path` (`"fused"` or `"array"`), and
`result.driver.fast_path_report()`. Quote its `step_path`, `decision`,
`refused_because`, `table`, `reference_driver` and `certified`. `certified` has
three values: `True` when every identity that served is certified, `False` when
`MEEP_GPU_ALLOW_UNCERTIFIED=1` admitted one that is not, and `None` when nothing
dispatched or an identity could not be read. A run prints at most one
`meep_gpu:` line on stderr per process for each distinct message;
[Reading what ran](docs/getting-started/reading-what-ran.md) explains them. `MEEP_GPU_DISPATCH_LOG=<path>` appends one JSON record per
configuration freeze, for long runs.

**Refusals are answers.** `MeepSimulationNotLiftable` carries every reason; a
`RuntimeError` on a host with no GPU names `prefer_gpu=False`; a lifted run that
took the array path names its reason in `refused_because` and, per sub-step, in
`fast_path_report()["slots"][<name>]["reason"]` (`composition` exists only on a
dispatched run);
`DispersionInstability` (a `ValueError` from `meep_gpu.dispersion` that the
preflight does not predict) names an unstable Lorentz or Drude pole. Report the
reason; do not change the physics of the user's script to get a lift through
without saying so. [Troubleshooting](docs/guides/troubleshooting.md) keys each
message.

**State these facts correctly.**

- The engine steps float32 (`complex64` for complex fields) whatever MEEP was
  built with. conda-forge MEEP is double precision: expect a difference of the
  order of single-precision rounding, 1 part in 10^7 to 1 part in 10^5 for
  runs that end while the field is still large. Final fields differed by
  1.5e-6 to 5.1e-4 (relative L2) over the 4 simulations compared on 1 host;
  that is not a guarantee ([Precision](INSTALL.md#precision)). Keep a quantity
  that needs more than about seven significant digits on MEEP.
- Certified identities: NVIDIA compute capability 8.6 (Triton 3.1.0, or the
  hand-written CUDA table); Apple PyTorch 2.10.0 with Metal frontend
  32023.850.10. Any other identity is refused by name to the array path, which
  is correct and slower.
- One simulation, one GPU, one process: no MPI (refused by name), no multi-GPU.
- Speed: point to [Will it help?](docs/guides/will-it-help.md) and repeat no
  ratio without the conditions stated there. The target is large 3-D problems
  on one certified GPU. 2-D problems, small 3-D grids and short runs are slower
  than on MEEP; the first two have not been timed against MEEP on a matched
  problem, so give no ratio for them and say what the statement rests on (the
  ratio falls as the grid gets smaller, and the step has a fixed cost).

**Do not:**

- set `KMP_DUPLICATE_LIB_OK` in a user's shell, script or environment; fix the
  environment instead
  ([One OpenMP runtime](INSTALL.md#one-openmp-runtime-on-an-apple-silicon-mac));
- write `MEEP_GPU_DISPATCH`, `MEEP_GPU_FUSED` or `MEEP_GPU_ALLOW_UNCERTIFIED` as
  `true`, `false` or `off`: only `0` and `1` are read, and anything else is
  refused by name and sends the run to the array path;
- set `MEEP_GPU_ALLOW_UNCERTIFIED=1` without comparing that run with a
  `prefer_gpu=False` run of the same simulation, and without telling the user
  that it carries no certification;
- edit, regenerate or delete a ledger (`meep_gpu/*_kernels/*.json`) to unlock a
  device or to turn a red test green: dispatch reads them at run time. That
  includes calling `launch.write_fingerprints()` and running a `rebind_*` or
  `recut_*` tool with `--write`;
- run under `mpirun`, or set `MEEP_GPU_WARM=0` or a subnormal-policy variable to
  chase speed; install a device extra on top of an environment file;
- report here a problem that reproduces with `sim.run(...)` alone, or report a
  problem of this package to the MEEP developers.

A bug report carries `python tools/check_install.py --verbose`, the MEEP version
and precision, `gpu_compatibility(sim).reasons`, `fast_path_report()`, and a
script that builds the simulation ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Changing the package

| Path | What it holds |
|---|---|
| `meep_gpu/` | The package, tests beside the code: `driver.py` (the engine), `stepping.py` (the array path), `from_meep.py` (the lift), `fastpath.py` (the dispatch planner) |
| `meep_gpu/triton_kernels/`, `cuda_kernels/`, `metal_kernels/` | The three kernel tables, each with its certification ledger (`*.json`) |
| `parity/meep_gpu/` | The certification and benchmark harness; not installed; [its README](parity/meep_gpu/README.md) |
| `tools/` | `check_install.py`, `rename_distribution.py`, and `ci/`: `check_failures.py` (failure arbiter), `check_wheel.py` (what the wheel carries), `pending_certification.txt`, `declared_resources.txt` |
| `examples/`, `environments/`, `docs/` | Example scripts with recorded output; the three conda environments; the MkDocs manual |
| `conftest.py` | The skip policy and the `certification` marker |

Set up: MEEP first ([INSTALL.md](INSTALL.md)), then
`python -m pip install -e ".[test]"`. `import meep_gpu` must keep working with
NumPy alone; MEEP, CuPy, Triton and PyTorch are imported lazily.

Commands, fastest first. Sizes and durations are in
[Running the tests](docs/development/testing.md).

| Command | Use |
|---|---|
| `python -m pytest meep_gpu/test_<module>.py -q` | The loop while editing. Metal and Triton tests are `meep_gpu/test_metal_*.py` and `meep_gpu/test_triton_*.py`; the CUDA table keeps its tests beside the code, `meep_gpu/cuda_kernels/test_*.py` |
| `python -m pytest parity/meep_gpu -q` | The harness suite |
| `python -m pytest meep_gpu -q` | The package suite; long: run it in the background, one suite at a time |
| `python -m pytest meep_gpu parity/meep_gpu -m certification` | Fails until the certification round has run; do not "fix" it |
| The three suites with `--junitxml`, then `python tools/ci/check_failures.py <reports>` | As CI runs them ([Running them as CI does](docs/development/testing.md#running-them-as-ci-does)); passes when every failure is listed in `tools/ci/pending_certification.txt` |
| `python examples/run_examples.py --check` | Must pass on the host |
| `python tools/check_install.py` | Must end `OK:` |
| `python -m mkdocs build --strict --site-dir <a directory outside the repository>` | The manual, after `python -m pip install -r docs/requirements.txt` |

A test skips only by a declared resource (`@pytest.mark.requires_resource`,
module-level `requires_resource_skip`, or `tools/ci/declared_resources.txt`);
any other skip fails, as does a `test_*.py` that collects nothing. Both suites
pin `MEEP_GPU_DISPATCH=0`: a dispatch test sets `1` itself, a test of the
shipped default deletes the variable. A module named on the command line runs
its certification-marked tests too, so a loop run can show a failure listed in
`tools/ci/pending_certification.txt`: that one is expected. Device
certification is the `gate_*.py` scripts on the hardware they name, not pytest
([the certification harness](docs/development/certification.md)). A gate run
writes `parity/meep_gpu/results/`; the tests listed under `evidence_archive` in
`tools/ci/declared_resources.txt` still skip there, because they run only when
that directory carries the evidence archive's marker, `EVIDENCE_ARCHIVE.json`,
which no gate writes. A test that reads a record under a resource of its own
runs wherever that record is present. Never write the marker into a directory of
your own records
([The evidence archive](docs/development/testing.md#the-evidence-archive)).

## Certification

Compiled kernels are held to bit identity, not to a tolerance. Each kernel table
must return the same bits as the package's own array path on the same device,
under the subnormal policy that table declares, and a kernel dispatches only on
a certified device and toolchain identity. Agreement with MEEP is a separate,
measured tolerance. The harness under `parity/meep_gpu` produces the evidence:
device gates per kernel family, route gates over whole runs, and a null control
for each that must be seen to fail.

The ledgers (`meep_gpu/*_kernels/*.json`) bind each verdict to the sha256
digests of the source files that produced it; dispatch reads them at run time.
Editing a pinned file, comments included, puts every entry that pins it in drift
and the ledger-contract tests fail, by design. The remedy is to re-run the
affected gates on the named hardware and rebind the ledger with the harness's
tools. In a pull request, that last step is a maintainer's (below).

## Rules

1. Never edit a certification ledger or record by hand.
2. Before editing a file under `meep_gpu/` or `parity/meep_gpu/`, find out
   whether a ledger pins it. Ledgers name files by path and some by file name
   alone, so search for the file name (here `driver.py`), then measure: run the
   three weld contracts (about 30 s) before and after the edit and compare the
   "N of D" drift counts in their failure messages. Each drifted pin is named
   with its ledger entry. A hit, or a count that rises, makes the edit a
   certification event.

   ```bash
   grep -lE '"([^"]*/)?driver\.py"' meep_gpu/*_kernels/*.json
   python -m pytest meep_gpu/test_metal_weld_contract.py meep_gpu/test_triton_weld_contract.py meep_gpu/test_cuda_weld_contract.py -q
   ```

3. Never weaken a test: no loosened tolerance, no deleted test, no skip without
   a declared resource.
4. Never set `KMP_DUPLICATE_LIB_OK` in package code, CI or instructions to
   users. The harness and some package tests set it for their own child
   processes; running them is permitted, and changing that is not part of an
   unrelated pull request.
5. Every number carries its denominator (N of D) and its conditions: hardware,
   precision, MEEP version and, for MEEP, rank count. A figure pending a
   re-measurement is written as provisional.
6. Rename outright, with no alias. `meep_gpu` and every `MEEP_GPU_*` name are
   fixed; the distribution name changes only through
   `python tools/rename_distribution.py`.
7. Anything that can run longer than about a minute prints a flushed line of
   progress per unit of work (which case of how many, elapsed time, the number
   it produced), and writes partial results as they land.
8. Code, comments and documentation are neutral and technical, and name no
   authoring tool.
9. Never commit anything under `parity/meep_gpu/results/`. Records are archived
   separately; a document points at a record instead of copying its numbers.

## Style and pull requests

Python 3.10 or later; `snake_case` functions, `PascalCase` classes, names that
state intent and units; comments explain assumptions and units. Numerical code
cites the MEEP source it follows, and a numerical change comes with its test
written first ([CONTRIBUTING.md](CONTRIBUTING.md), "Numerical changes"). New
behavior refuses by name instead of approximating.

A pull request states what changed and why; the commands run, on which host, with
results as N of D and every failure either listed in
`tools/ci/pending_certification.txt` or explained; the rule 2 result for each
edited file; and the documentation updated in the same change. It carries no
ledger edit and no `results/` record. When an edit is a certification event, the
pull request names the drifted ledger entries and says either that
re-certification is needed and was not run, or which gates were run, where, and
with what verdict, with the artifact directories attached to the pull request
discussion as an archive rather than committed. A maintainer re-runs or checks
the gates on the named hardware and rebinds the ledger.
