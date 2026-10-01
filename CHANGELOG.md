# Changelog

## 0.9.1 — one certification record per GPU architecture (2026-10-01)

Still a preview, and for the same reason as 0.9.0: the certification round on the
released bytes has not run, so the certification test selection fails until it
does (`tools/ci/pending_certification.txt`). What changes here is that the round,
when it runs, can certify more than one GPU architecture — and that a later round
on another card adds to the first instead of replacing it.

### Certification records are now per compute capability

- **The defect.** Each weld in the two NVIDIA ledgers held one run's facts beside
  the digests of the bytes that run certified. That shape holds exactly one GPU
  architecture, so a round on a second card overwrote the first card's evidence
  and left its label behind; the list of validated compute capabilities was a
  hand-typed key that no tool wrote.
- **The records.** Each weld keeps one record per compute capability under a
  `runs` key, each carrying `bound_sha256`, the digest of the digests it
  certified. A record is live only while that equals the entry's current digests,
  so editing a certified file retires every architecture's evidence for that weld
  at once rather than leaving some of it describing code that no longer ships.
- **The list is derived.** A table's validated compute capabilities are the
  intersection, over the welds its arms cite, of each weld's live architectures.
  The declaration can only say what some run measured, and a round widens it by
  writing records rather than by editing code. A partial round admits nothing and
  names the welds that are short.
- **Certifying another architecture** is documented end to end in
  `docs/development/certification.md`: the round procedure, which architecture
  binds first and why only the rebind order is constrained, what names the card,
  and the toolchain versions to pin, with which step-2 tools check them.

Compute capability 8.6 remains the one architecture the ledgers certify. Every
other NVIDIA card still takes the array path, and a card can be opted in for a
run of its own certification with `MEEP_GPU_ALLOW_UNCERTIFIED=1`.

### Also in this release

- The CUDA off-diagonal `update_E` kernel hoists its own-cell loads and memoizes
  its launch text on the certified digest, instead of regenerating that text on
  every launch.
- Two DFT-region tests and the near2far control no longer sit on a grid tie.
  MEEP's arm64 build fuses `pt*a - .5` where its x86-64 build rounds twice, so a
  coordinate exactly on a grid site resolves differently per platform; beneath
  that tie, `get_dft_array` returns zero for a single-site region on both
  platforms. The engine was correct on both hosts; the fixes are to the tests.
- The Gaussian-beam oracle resolves under the libstdc++ symbol spelling as well.
- The bytecode-cache test reads only this interpreter's cache tag, and the skip
  tally prefers the resource named in a skip's reason.
- `parity/meep_gpu/cases.py`, the benchmark case builders three composition gates
  import, is now in the repository. Earlier certification rounds staged it beside
  the checkout as an untracked file, and a fresh clone could not run those gates.
- Two architectures gated on one commit can be bound in either order and leave the
  same ledger. A rebind carries a per-run fact from an architecture's previous record
  only while that record is live on the bytes being bound; it used to carry whenever
  the bytes had not moved in that pass, so binding the new architecture first copied
  a measurement of other code into the old one's record. The composition re-cut no
  longer carries a run's description of the host it ran on into a later run at all.

### Shipped files that moved

- `meep_gpu/triton_kernels/timing.json` and `meep_gpu/cuda_kernels/timing.json`,
  with their reports, are now `timing_cc86.json` and `timing_cc86.report.txt` —
  one timing record per compute capability. The bytes are unchanged.

## 0.9.0 — preview release (2026-09-28)

The first public release. It is a preview: the package runs and its kernels were
certified on earlier bytes of this source. On the released code, the go/no-go
for dispatch on by default and the route campaigns of the three kernel tables ran
on 2026-09-29 and released; the per-family certification round, the transcription
of those runs into the ledgers, and the timing campaign from the release tag are
still to be run. What that leaves open is listed under "Known limits".

The package is an independent add-on to MEEP. It is not affiliated with or
endorsed by the MEEP developers.

### What works

- **Three entry points for a MEEP user.** `gpu_compatibility(sim)` reads a
  constructed `mp.Simulation` and answers supported, or the reasons why not,
  without initializing it. `run_on_gpu(sim, until=...)` lifts, steps and returns
  fields and monitor results. `lift_simulation(sim)` returns the initialized
  driver for runs in segments. MEEP builds the grid, rasterizes the geometry and
  applies its subpixel smoothing; the package steps the realized state.
- **Three kernel tables and an array path.** Triton kernels and hand-written CUDA
  kernels on NVIDIA hardware through CuPy; Metal kernels on Apple hardware through
  PyTorch MPS. The array path steps the same equations on CuPy or NumPy and is
  what serves any sub-step no certified kernel covers.
- **Defaults.** `run_on_gpu` and `lift_simulation` request this host's GPU
  (`prefer_gpu=True`); `prefer_gpu=False` is the NumPy reference. Kernel dispatch
  is on; `MEEP_GPU_DISPATCH=0` selects the array path. On Apple hardware the
  device holds the field volumes between launches.
- **Physics stepped.** One, two and three dimensions and cylindrical coordinates;
  PML and absorber layers; Bloch-periodic boundaries; mirror symmetry; anisotropic
  permittivity from subpixel smoothing; conductivity; Lorentz and Drude
  dispersion; chi2 and chi3 nonlinearity; real and complex fields.
- **Monitors and results.** MEEP's flux, DFT-field, near-to-far, force and energy
  monitors are carried over from the simulation and read back through the result
  object; MEEP step functions, `mp.Harminv` among them, are hosted around the
  run loop.
- **Refusals are by name.** A simulation or a host the package cannot serve is
  refused with the reason, and a GPU request on a host with no GPU raises and
  names `prefer_gpu=False`; nothing falls back silently.
- **MEEP's own scripts.** The lift accepts the simulations built by 60 of 65
  example scripts and 134 of 233 test methods of the MEEP 1.33.0 corpus that
  construct one.
- **`meep_gpu.__version__`**, read from the installed package metadata, or from
  `pyproject.toml` in a checkout that was not installed.
- **An installation check and a longer check.** `tools/check_install.py` steps one
  simulation with the default, on the NumPy reference and with MEEP, and its last
  line says where the default run stepped; `examples/run_examples.py --check`
  compares every path the host has. Both were run as the install guide writes
  them on one Apple silicon Mac and one NVIDIA host (INSTALL.md, "Tested MEEP
  versions and builds").

### What is certified

- **Device identities, one per kernel table:** NVIDIA compute capability 8.6 with
  Triton 3.1.0 (Triton kernels); NVIDIA compute capability 8.6 (CUDA kernels);
  PyTorch 2.10.0 with Metal frontend 32023.850.10 (Metal kernels). Measured on one
  RTX A6000 with CuPy 13.5.1 and on one M1 Max.
- **Bit identity with the array path.** Each kernel family was certified
  bit-identical to the package's own array path on the same device under the same
  subnormal policy: `keep` and `flush` on NVIDIA, `flush` on Apple. The array path
  is CuPy on the device on an NVIDIA host and NumPy on the host CPU on an Apple
  host. A default GPU run and a `prefer_gpu=False` reference run made in another
  process are not bit-identical once values pass through the float32 subnormal
  range (24,550 of the 25,600 cells of one case differed), because dispatch
  installs the kernel table's policy for the whole process. No identity between
  NVIDIA and Apple results is claimed, and none with MEEP.
- **Agreement with MEEP** is a tolerance, measured separately on the NumPy
  reference: the simulations built by 58 of 60 of MEEP's example scripts agree
  with MEEP 1.33.0 in single precision within 2.4×10^-7 to 1.6×10^-5 (relative
  L2), on grids up to 338,688 cells (2026-08-09).
- **The status of that certification in this release.** The ledgers shipped here
  were cut on earlier bytes of this source. Files that the ledgers pin by digest
  were edited for the release after the ledgers were cut, so the recorded digests
  do not match the released files. The certification round is to be run on
  the released bytes; until its ledgers are committed, the ledger-contract tests
  of the suites fail, and they are reported as failed. They form the
  certification suite (pytest marker `certification`), which a run without `-m`
  does not select and `-m certification` runs; the failures it may show are
  listed in `tools/ci/pending_certification.txt`.
- **Dispatch on the released code (2026-09-29).** The go/no-go for dispatch on
  by default and the route campaigns of the three kernel tables ran on the code
  of this release. Every source digest the runs recorded equals the published
  file (`meep_gpu/fastpath.py` `1b22b59c…` among them) except one: the three
  Metal legs recorded `parity/meep_gpu/probe_metal_dispatch_dryrun.py` before
  one of its progress messages was rewritten to parse under Python 3.10, with
  the same output text. The tree that ran differs from the release in its
  Markdown documentation, that harness probe, the two digest helpers the
  certification tests use (`meep_gpu/code_identity.py` and
  `meep_gpu/device_identity.py`, which no run imported) and test modules.
  - Go/no-go, on one RTX A6000 (compute capability 8.6, Triton 3.1.0, CuPy
    13.5.1), from the default: the dispatch switch unset, no other `MEEP_GPU_*`
    variable in the environment than the dispatch log's path, and no subnormal
    policy installed by the harness.
    Verdict GO: 8 of 9 whole simulations stepped on compiled kernels, all on the
    Triton table, and 1 of 9 (`bloch_2d`, complex fields) took the array path
    with its reason recorded; 62 of 62 checkpoints were byte-identical to the
    same simulation on the array path. With the kill switch on both legs, 9 of 9
    launched no kernel.
  - Triton route campaign: 4 of 4 legs released. As shipped, 20 of 37 cases
    were byte-identical through fused kernels; the other 17 are complex cases,
    and with the complex-storage evidence record 37 of 37 were.
  - Hand-written CUDA route campaign: 6 of 6 legs released. As shipped, with the
    CUDA table put first, 19 of 39 (18 complex cases, 2 with no fused arm); with
    the record, 37 of 39. Under the default precedence the Triton table served
    every dispatching case, as designed. Its leg
    with Triton withheld from the process ran for the first time: 19 of 39 cases
    byte-identical with only CUDA kernels launched.
  - Metal route campaign, on one M1 Max (PyTorch 2.10.0, residency held, the
    default): 5 of 5 legs released. As shipped, 24 of 42 (18 complex cases); with
    the complex-storage probes, 42 of 42; 0 hold refusals.
  - No leg had a divergent checkpoint. These runs are not yet transcribed into
    the ledgers or the boards, so the ledger-contract tests still fail as
    described above. The legs that use the complex-storage evidence read
    records measured on earlier bytes.
- **Examples.** On one Apple M1 Max, 3 of 3 examples step on compiled kernels and
  reproduce the array path byte for byte (`examples/expected/`, recorded
  2026-09-28).
- **Documentation for coding agents.** `AGENTS.md` gives the commands and rules
  for coding agents helping a user run a simulation or changing the package.

### Known limits

- One simulation runs on one GPU. There is no multi-GPU domain decomposition, and
  a lift in a process that is one of several MPI ranks is refused.
- Only the certified identities run compiled kernels. Any other device or library
  version is refused by name and takes the array path: on the one card measured,
  slower than MEEP's fastest rank count at each of the three sizes timed on the
  identical case (`docs/guides/will-it-help.md`). A device whose identity
  cannot be read is not refused. `MEEP_GPU_ALLOW_UNCERTIFIED=1` dispatches the
  kernels on an identity that was read and is not certified; such a run carries
  no certification, says so once per process and records `certified: False`. The
  switch accepts `1` and `0` only. It has been exercised with substituted
  identities and on one real identity outside the certified ones: PyTorch 2.14.0,
  the newest release on 2026-09-28, on one M1 Max, where 4 of 4 examples stepped
  on the Metal kernels and 3 of 3 reproduced the array path byte for byte.
- On an Apple silicon Mac a process may hold one OpenMP runtime. A PyTorch wheel
  from PyPI carries its own, and conda-forge's OpenBLAS in its default OpenMP
  build loads the environment's; with PyTorch 2.14.0 the two together stop every
  process with `OMP: Error #15`, whatever the import order.
  `environments/apple-silicon.yml` installs OpenBLAS in its pthreads build, which
  loads none, and `tools/check_install.py` reads both runtimes before anything
  imports PyTorch and names the fix (INSTALL.md, "One OpenMP runtime on an Apple
  silicon Mac").
- Two-dimensional problems and small three-dimensional problems are not a
  target, and they are slower than on MEEP: none has been timed against MEEP on
  a matched problem, the ratio falls as the grid gets smaller, and the
  dispatched step has a fixed cost per step. The dispatcher does not yet choose
  by measured cost, so it takes a fused kernel where the package's own single
  kernels are faster.
- The lift runs in one process and precedes the first step: about 19 s, 68 s
  and 216 s at 512,000, 2,097,152 and 7,077,888 cells on the NVIDIA host, MEEP's
  own initialization included. On a short run the lift outweighs the stepping
  saving: on the Triton route the GPU run finishes first only after about
  35,000, 22,000 and 19,000 steps at those sizes (derived from the medians,
  with MEEP's own start-up subtracted; `docs/guides/will-it-help.md`).
- The engine steps single precision. Every timing and every agreement figure
  was measured against MEEP 1.33.0 built from source in single precision. The
  MEEP builds on conda-forge are double precision. Against them (MEEP 1.33.0 and
  1.34.0, serial) the installation check and the examples ran on one Apple host
  and on one NVIDIA host on 2026-09-28, and the package's tests ran on the Apple
  host (INSTALL.md, "Tested MEEP versions and builds"). The tests whose premise is
  a single-precision MEEP skip on a double-precision one by a declared resource,
  and the force and energy monitor tests take the bound of the MEEP they find.
- No MEEP is refused on its version or its precision. A lift reads both and
  records them (`driver.lift_record`), and a lift from a double-precision MEEP
  prints one line per process saying that the engine steps single precision and
  the agreement to expect. A lift from a MEEP release other than 1.33.0 proceeds
  without a message.
- One comparison with MEEP on the identical problem has been made
  (`docs/guides/will-it-help.md`): from 512,000 to 7,077,888 cells, 1.80 to 3.67
  times MEEP on the Triton route and 1.68 to 3.24 times on the CUDA-only route.
  That is steady-state stepping, the one-off lift not counted, on one RTX A6000
  against MEEP 1.33.0 at the fastest of seven rank counts tried on the same
  machine's 48 physical cores, single precision on both sides, one 3-D case
  (`pml_3d`) with one flux monitor. A ratio is reported when both of its runs
  passed the timing check; all six against MEEP's fastest did. The CUDA-only
  route is being improved for the next release, and these figures will be
  updated then. A timing campaign from the release tag is still to be run.
- One three-dimensional case has been timed against MEEP on the identical
  problem, at three sizes up to 7,077,888 cells, on one NVIDIA host; none on
  Apple hardware. No dispersive, Bloch-periodic, cylindrical or DFT-heavy
  three-dimensional timing exists.
- Kernel dispatch is on by default. Its go/no-go from the default and the route
  campaigns ran on the released code (above), and the record behind the
  on-by-default setting is not yet cut from them: that is the next
  certification round. The hand-written CUDA table composing alone, on a host
  with no Triton, has been run on one device for the installation check's
  simulation and one example (compiled kernels served, identical to the array
  path), and in one route leg with Triton withheld from the process; it has not
  been timed.
- Some module docstrings and comments in files the certification records pin
  describe earlier states of the code (for example, that dispatch is not wired
  to a kernel package, or that a kernel has no byte-identity claim yet).
  Changing those files changes what the records pin, so they are corrected in
  the next certification round; where they disagree with the manual, the manual
  describes the released code.
- Runs that store complex fields (Bloch-periodic, cylindrical with m ≠ 0, complex
  β) take the array path for the affected sub-steps: their kernels need an
  evidence record this release does not ship. 50 of the 194 accepted simulations
  of the MEEP corpus are of this kind.
- Monitor accumulation and source deposition run on the array path on every
  kernel table.
- Agreement with MEEP was measured on the array path, on grids up to 338,688
  cells, on 2026-08-09. It has not been measured through the kernels or at the
  timed sizes.
- Refused by name, among others: magnetic materials; gyrotropic, multilevel and
  noisy media; rotational symmetries; a simulation that was already initialized
  or stepped; a simulation that declares no sources. LDOS and the HDF5 writers are
  not hosted in step functions. `docs/guides/compatibility.md` has the list.
- A run that dispatches compiled kernels installs its kernel table's float32
  subnormal policy for the whole process. On the Apple host measured it was still
  installed after `close()`.
- The examples' output was recorded on Apple hardware only. On the NVIDIA host
  the 3 examples that compare with MEEP matched the recorded output with 0
  mismatches; their comparison with MEEP was reported, not judged.
- The test suites of this repository are the package's tests and the tests of
  the certification harness. The site launchers the NVIDIA campaigns were run
  with (batch-scheduler jobs and single-host shell drivers), the scripts that
  produced internal documents, and one-off diagnostics with no reader are not
  part of it; `parity/meep_gpu/README.md` gives the recipe the launchers
  followed. Tests that read the evidence archive of the certification campaigns,
  which this repository does not carry, skip by the declared resource
  `evidence_archive` (`tools/ci/declared_resources.txt`). The archive counts as
  present only when `parity/meep_gpu/results/` carries its marker file,
  `EVIDENCE_ARCHIVE.json`, so the records a gate run writes there do not make
  those tests run.
