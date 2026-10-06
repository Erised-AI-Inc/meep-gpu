# Changelog

## 0.9.2 — supported GPUs run the kernels by default, and certification is on the release bytes (2026-10-06)

Still a preview. The certification round ran on the bytes this release ships: the
Triton and CUDA tables on compute capability 8.6 (RTX A6000) and the Metal table on
the Apple M1 Max (`applegpu_g13s`). Every certification entry is bound except the two
fused-product timing tests.

### Pending: fused-product timing

The timing records `meep_gpu/cuda_kernels/timing_cc86.json` and
`meep_gpu/triton_kernels/timing_cc86.json` do not describe this release. They were cut
from rows timed on an earlier deposit-repair route. So it is not yet measured on these
bytes whether a two-pair fused plan beats the single kernels it replaces at the top of
the sweep, at 2.4M and 5.4M cells.

- **Dispatch is unaffected.** The measured-cost preference that would read these
  records (`meep_gpu/dispatch_preference.py`) is not wired into dispatch. A fused
  product takes its slot on span, as in 0.9.1.
- **Correctness is unaffected.** The certification gates bind every dispatched fused
  product bit-identical to the array path.
- **Two entries stay on `tools/ci/pending_certification.txt`:**
  `test_dispatch_preference::test_at_the_top_of_the_sweep_the_two_pair_plans_are_faster`
  for `[cuda]` and `[triton]`.
- **How they clear:** a timing campaign on these bytes, on an otherwise idle RTX A6000,
  then re-cut records. The campaign of 2026-10-06 found no idle window in 8 hours and
  ran no leg.

### Supported NVIDIA GPUs run the kernels by default, and say whether they are certified

The NVIDIA tables now follow the rule the Metal table follows: a supported device or
toolchain that is not certified runs the compiled kernels by default and says so, and
`MEEP_GPU_ALLOW_UNCERTIFIED=0` restricts the kernels to certified identities.

#### Supported and certified on NVIDIA hardware

- **Supported**: compute capability 7.0 to 9.0 on both NVIDIA tables
  (`fastpath.SUPPORTED_COMPUTE_CAPABILITIES`,
  `fastpath_cuda.SUPPORTED_COMPUTE_CAPABILITIES`), and Triton `>=3.1,<3.2` on the
  Triton table (`fastpath.TRITON_SUPPORTED_VERSIONS`). The floor is the one Triton
  3.1 documents; the ceiling is the newest architecture its bundled ptxas and the
  NVRTC of the CUDA 11 CuPy build (CUDA 11.8) target, and is conservative for the
  CUDA 12 CuPy build. Read from the compilers' documentation and code; no NVIDIA
  device outside the certified list has been run.
- **Certified**: unchanged. Compute capability 8.6, with Triton 3.1.0 on the Triton
  table, derived from the ledgers' live records.

#### What each value of `MEEP_GPU_ALLOW_UNCERTIFIED` does on NVIDIA hardware

- **Unset**: a certified identity runs, `certified: True`. A supported, uncertified
  one runs, `certified: False`, with one NOTE line, unless another candidate table
  is certified for this device and toolchain: then the supported one is refused by
  name (rung 4e), the certified table runs alone and the run is certified, so the
  default never makes a certified run uncertified. A table whose compute capability
  or ledger could not be read is not certified, so it outranks nothing: beside it a
  supported table composes, and a run that read `certified: None` before can read
  `certified: False`. An identity read outside
  the supported range is refused by name, and the refusal names `1`. A compute
  capability that could not be read runs as before, `certified: None`.
- **`1`**: every identity runs; one outside the supported range is recorded
  `supported: False`, and a supported table composes beside a certified one.
- **`0`**: only a certified identity runs. A supported, an unsupported and an
  unreadable identity are refused by name, as on the Metal table; before this change
  `0` behaved as unset on the NVIDIA tables.
- **`MEEP_GPU_BACKEND_PREFERENCE`** naming a table rung 4e refused is refused by name
  with that reason appended, unless `1`. `MEEP_GPU_KERNEL_TABLE` naming one table
  leaves the other unconsulted, so the named table runs, uncertified if it is not
  certified.
- **On a host where every candidate table is certified, the decision does not
  change**: the decision, the tables, the slots and `certified` are those of the
  previous release under every value of the switch, on 400 of 400 synthetic hosts
  where every table considered is certified, in a 4,992-case decision matrix
  (compute capability 6.1 to 12.0 and unreadable; Triton 3.1.0, 3.1.1, 3.2.0 and
  absent; every switch value; both table selectors; disjoint and overlapping
  composers; 9.0 ledgers certifying neither, one or both tables). Every run that
  was certified before (282 of 282) has the same decision and slots. The status
  line is unchanged on 364 of the 400. On the other 36 the device's mark is now
  that of the tables that served: a hand-CUDA run on a device only the hand-CUDA
  ledger certifies, whose record said `certified: True`, printed `UNCERTIFIED` and
  now prints `certified`. No shipped ledger is in that state; a 9.0 bind that
  certifies the hand-CUDA table before the Triton table would be.

#### What a run says

- **The NOTE line**: `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU
  compute capability 8.9 (certified: 8.6). They were dispatched because this NVIDIA
  GPU and toolchain are supported but not certified bit-identical (the supported
  range is ...), and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the kernels to
  certified ones; the gates with no run on this GPU are counted in the dispatch
  record under uncertified.served[], which names the first five in
  welds_without_a_live_run_here; compare the results with a prefer_gpu=False run of
  the same simulation before relying on them`. The reason names what was judged:
  `this NVIDIA GPU and toolchain are` where the GPU and the Triton version beside it
  are both supported, `this NVIDIA GPU is` where no supported toolchain was judged
  with it (a hand-CUDA run reads none), `this Triton version is` where only the
  Triton version is uncertified; under `1`, `MEEP_GPU_ALLOW_UNCERTIFIED=1, which
  also runs this NVIDIA GPU` (or `this Triton version`) `outside the supported
  range`. Each admitted entry's `because` names that identity and its table's
  range.
- **The status line** marks a supported, uncertified device `supported,
  UNCERTIFIED`, and a Triton 3.1 release other than 3.1.0 `triton 3.1.1 supported,
  UNCERTIFIED`.
- **The record**: the NVIDIA half of `environment` gains `triton_supported`,
  `device_supported`, `device_supported_by_table`, `supported_ranges` and
  `certified_only`; `uncertified` gains `unsupported_allowed` and `dropped`, and
  `allowed` now means "a supported, uncertified identity may run" (unset and `1`);
  each admitted entry carries `supported`, `because` and, for a compute capability,
  `welds_without_a_live_run_here`; a table refused at rung 4e carries
  `dropped_for_a_certified_table`.
- **Renamed**: `fastpath.uncertified_allowed()` is replaced by
  `fastpath.certified_only()` and `fastpath.unsupported_allowed()`, and
  `UNCERTIFIED_HINT` by `UNSUPPORTED_HINT` (with `CERTIFIED_ONLY_TAIL` for the `0`
  refusals).

#### The harness keeps uncertified identities out of evidence

- `drive_triton_weld_gates.py` runs every gate child with
  `MEEP_GPU_ALLOW_UNCERTIFIED=0` (it removed the variable before).
- `timing_ladder.py` exports `MEEP_GPU_ALLOW_UNCERTIFIED=0` to its NVIDIA rows unless
  `--allow-uncertified 1`; Metal rows keep the Metal default.
- `recut_driver_dispatch_record.py` refuses a route row the Triton table served on a
  Triton version that is not certified, as it already refused one on an uncertified
  device.
- `gate_dispatch_fused_route.py` refuses to start a route leg, before its directory
  exists, unless `fastpath.capability_admission` certifies the card for the table
  under test and, on the Triton table, the installed Triton version is certified. It
  also refuses a leg started with `MEEP_GPU_ALLOW_UNCERTIFIED` set to any value, since
  `=1` would let a table that only supports the card compose beside the certified
  one; `--smoke` is not asked. The end-to-end licence gate is unchanged: it refuses
  every `MEEP_GPU_*` variable.
- `tools/check_install.py` prints, per NVIDIA table and for the Triton version,
  whether each is supported and certified, expects the kernels wherever one table may
  run, and notes an uncertified run.

#### Certification owed

`meep_gpu/fastpath.py` and `meep_gpu/fastpath_cuda.py` changed, so the records that
pin them are stale until the next round re-runs them on these bytes: the Triton
`bit_identity_gate`, the three `driver_dispatch` records and the three Metal device
welds that pin `fastpath.py`. The tests that read them are listed in
`tools/ci/pending_certification.txt`: four under "Added 2026-10-05 for the
uncertified NVIDIA default", and the rest in entries already listed, the four Metal
ones among them in the block added with the per-architecture Metal run records
below. The NVIDIA route-campaign stamps moved before that round, so its campaigns
run as `dispatch_fused_route_2026-10-05_092_cc86` and
`dispatch_fused_route_cuda_2026-10-05_092_cc86` (`DRIVER_ROUTE_FUSED_GATE`,
`CUDA_DRIVER_ROUTE_FUSED_GATE`; one stamp per release, `_cc90` beside `_cc86`), and
two of those four entries are the record tests that read the 0.9.1 campaign names
until they do. The Metal stamp moved with the per-architecture run records
(`dispatch_metal_route_2026-10-05_perarch`).

### Every Apple GPU is supported, and the Metal kernels say whether it is certified

On Apple hardware every GPU is now supported: the
Metal kernels run on any Apple GPU, PyTorch and macOS build, and every run states
whether its environment is certified, the one the certification records name.
Supported is not certified.

#### Certified on these bytes

- **Apple M1 Max, re-certified.** The Metal round `2026-10-02_arch3` ran on these
  bytes: 66 of 66 fleet gates and probes released, the rebind left all 45 cited
  Metal welds recording `applegpu_g13s`, PyTorch 2.10.0 and `metalfe-32023.850.10`,
  and the held route campaign released 5 of 5 legs; with the per-architecture records
  below, that route run is kept as dated history in the Metal dispatch record, and its
  re-cut on these bytes (`dispatch_metal_route_2026-10-05_perarch_applegpu_g13s`) is
  owed, the record's contract tests pending until then. The predicate census was re-cut
  over this tree's 94 Metal subject files, and the Metal fusion board reads 327 of
  597 seam-instances served in dispatch (531 by a fused product), as before; it is
  the first Metal board cut with dispatch on by default.
- **NVIDIA.** No device weld in the CUDA or Triton ledger pins a file this change
  edits. The two `driver_dispatch` records pin `meep_gpu/fastpath.py`, which this
  change edits (the uncertified-admission seam), so they were owed at the next
  route round, and were bound on these bytes in the rounds stamped 2026-10-05
  (see the header). The compute capability 8.6 round of this release (RTX A6000) ran on
  these bytes. An H200 round (compute capability 9.0) is prepared; no 9.0 record
  ships in 0.9.2.

#### Supported and certified

- **Supported**: the kernels are meant to run on that hardware, and run there by
  default. Every Apple GPU is supported, M1 through M5 and later: a GPU whose
  architecture Metal names `applegpu_*`, or, where the architecture cannot be read
  (before macOS 14), whose name starts `Apple `
  (`metal_dispatch.device_supported`).
- **Certified**: unchanged. The certification gates measured the kernels in that
  environment, and every Metal weld the table cites records it (see Verdicts
  below). An Apple GPU no weld ran on is supported and uncertified.
- **On NVIDIA hardware** the same rule now holds for compute capability 7.0 to 9.0
  (the section above).

#### A Metal environment is four facts

- **The GPU architecture**, `MTLDevice.architecture.name`, read with `ctypes`
  through the Objective-C runtime (no PyTorch, no PyObjC; macOS 14 or later):
  `applegpu_g13s` on an Apple M1 Max. It separates GPU generations that share a
  Metal GPU family (M3 and M4 are both Apple9), which the Metal frontend cannot:
  every Mac on one macOS build reports the same frontend.
- **The PyTorch version** and **the Metal frontend version**
  (`metalfe-...`, one per macOS build).
- **`PYTORCH_MPS_FAST_MATH`**, certified only unset or `0`. Measured 2026-10-02 on
  one Apple M1 Max (macOS 26.2, PyTorch 2.10.0;
  `parity/meep_gpu/probe_metal_fast_math.py`): `1` reaches
  `torch.mps.compile_shader` and changed 274,523 of 1,048,576 float32 divide words
  and 327,338 of 1,048,576 square-root words; `0` changed none.
- **Verdicts.** A fact is certified when every Metal weld the table cites (45 welds)
  recorded that value on its host line, not certified when a cited weld recorded
  another, and not judged when it could not be read or the cited welds do not record
  it. The certified environment is GPU architecture `applegpu_g13s` (Apple M1 Max),
  PyTorch 2.10.0, Metal frontend `metalfe-32023.850.10`, fast math off. Other Apple
  GPUs, PyTorch versions and macOS builds run, as supported, and are labelled
  uncertified.

#### Dispatch on an Apple GPU

- **By default the Metal kernels run on any environment.** A fact that is not
  certified is recorded under `uncertified.served`, with the reason it ran, the
  record says `certified: False`, and the process prints one line such as
  `meep_gpu: NOTE the kernels are NOT CERTIFIED on this host: GPU architecture
  applegpu_g15p (certified: applegpu_g13s). They were dispatched because this Apple
  GPU is supported, and MEEP_GPU_ALLOW_UNCERTIFIED=0 would restrict the Metal
  kernels to certified environments; compare the results with a prefer_gpu=False
  run of the same simulation before relying on them`. On a GPU that is not Apple's,
  or one that could not be read, the reason given is `the Metal table runs on
  environments outside the certified set unless MEEP_GPU_ALLOW_UNCERTIFIED=0`. A
  fact that is not judged runs too, with `certified: None` and no note.
- **`MEEP_GPU_ALLOW_UNCERTIFIED=0` restricts the Metal kernels to certified
  environments.** A fact that is not certified, or not judged, is refused by name
  and the run takes the array path on the host CPU. Unset and `1` run an
  uncertified environment. On the NVIDIA tables the switch now does the same for a
  supported identity (the section above). On every table any value but `0` and `1`
  is refused by name and the run takes the array path.
- **The status line names the GPU**: `...; torch 2.10.0 certified, metal frontend
  metalfe-32023.850.10 certified; Apple M1 Max (applegpu_g13s) certified`, where it
  ended `device unknown uncertified-unknown`. An Apple GPU that is not certified is
  marked supported as well: `Apple M3 Pro (applegpu_g15p) supported, UNCERTIFIED`.
  The certified GPU stays `certified` when another fact is not (an uncertified
  PyTorch reads `torch ... UNCERTIFIED`), and a GPU that is not Apple's carries no
  `supported`.
- **The dispatch record.** The Metal half of `environment` carries `device`
  (`name`, `architecture`), `device_supported` (`True` on an Apple GPU, `False` on
  a GPU that is not Apple's, `None` when neither the architecture nor the name was
  read), `device_certified` (the architecture's verdict, in the key the NVIDIA
  tables use for the compute capability), `torch_certified`, `frontend_certified`,
  `fast_math`, `fast_math_certified`, `recorded_environments`
  (one row per distinct architecture, PyTorch and Metal frontend the cited welds
  recorded, with its count of welds) and `certified_only`. `validated_toolchains` is
  removed.
- **`tools/check_install.py`** prints the GPU (name and architecture), PyTorch and
  the Metal frontend, each of the three marked `certified`, `NOT certified: the
  records name another` or `not judged: it could not be read, or the records do not
  name it`, and, when it is set, `PYTORCH_MPS_FAST_MATH` with `off` or `fast-math
  kernels, which no certification ran`. The GPU's mark is preceded by `supported
  (every Apple GPU is); ` on an Apple GPU that is not certified, and by `not an
  Apple GPU, so not supported; ` on another GPU. It expects the kernels to serve on
  any Mac unless `MEEP_GPU_ALLOW_UNCERTIFIED=0`, and adds a note when the run was
  not certified, saying on an Apple GPU that the environment is supported but not
  the certified one.

#### Certification records

- **A Metal weld's host line names the architecture**: `this machine: Apple MPS
  device applegpu_g13s, torch 2.10.0, metalfe-32023.850.10`.
- **`recut_metal_gates.sh`, given a stamp, records the environment** before the
  first gate and after the last, into `results/metal_environment_<stamp>/start.json`
  and `end.json` (`parity/meep_gpu/metal_environment.py`).
- **`rebind_metal_welds.py` and `mint_metal_weld.py` write each host line from those
  records**, and refuse a campaign whose records are missing, could not read a fact,
  disagree between start and end, or record `PYTORCH_MPS_FAST_MATH` set to anything
  but `0`.
- **One run record per Apple GPU architecture.** Each Metal weld entry keeps its
  digests, status and purpose, and holds its runs under `runs[<architecture>]`
  (`applegpu_g13s`): the run fields, the PyTorch and Metal frontend read off the run's
  host line, and a `bound_sha256` of the bytes it certified. A run is live while that
  digest matches the entry's pinned bytes, as on the NVIDIA ledgers. The write rules
  (the `applegpu_*` key, the shape, the staleness rule, the route campaign's name) are
  in the new `meep_gpu/metal_runs.py`, which the dispatch ladder does not import.
- **A second Mac joins the first.** On unchanged bytes, `rebind_metal_welds.py` and
  `mint_metal_weld.py --replace` add the campaign's architecture beside the runs
  already recorded and leave them byte for byte. On bytes that moved they name every
  other architecture's run they would strand, and `--supersede <architecture>` drops
  it by name; `--replace-architecture` is removed. A rebind carries a run's
  `subnormal_policy` line from the same architecture's previous run, else from the one
  line the entry's other live runs carry, else from the artifact by its policy value
  (never the repr of a policy report; the mint likewise), and records which in
  `_subnormal_policy_read_from`. Both refuse a gate artifact whose own
  `environment.apple_gpu.architecture` is not the campaign's. A second Mac's rebound
  ledger is committed together with its route run, not before it
  (`docs/development/certification.md`).
- **The admission reads the runs.** The GPU architecture is certified when every
  cited weld has a live run for it, PyTorch and the frontend when every such run
  recorded this host's (`metal_dispatch.environment_verdict`); a weld whose entry holds
  no per-architecture run is not judged. `recorded_environments` counts welds per
  live-run environment. The values a NOTE line or a refusal calls certified obey the
  same intersection (`metal_dispatch.certified_values`), so a GPU one cited weld short
  of certified is not listed as certified beside its own NOT CERTIFIED.
- **A dispatched Metal arm quotes the run its weld recorded on this GPU.** The Metal
  environment block writes the architecture as `device.compute_capability` too, the
  key the certification quote reads; on a GPU no weld ran on it quotes no run.
- **The Metal route record keeps one route run per architecture**: its status, legs,
  records, timestamp, the run's `released_fused_arms` subkeys, and the residency and
  lift blocks read off the legs (the residency block is about 470 KB of the ledger
  file per architecture). The route campaign
  runs under `<stamp>_<architecture>`, the stamp being
  `metal_dispatch.METAL_DRIVER_ROUTE_GATE` (now `dispatch_metal_route_2026-10-05_perarch`),
  and `recut_driver_dispatch_record.py --backend metal` reads the architecture off every
  leg's `provenance.apple_gpu`, refuses legs that name two, and takes `--supersede`.
- **`parity/meep_gpu/migrate_metal_runs.py` moves a one-run Metal ledger, once.** It
  reads each run's architecture off the entry's own host line, dates the dispatch
  record's route run as history with `runs: {}`, reports by default, and before
  `--write` checks entry by entry that the move is reversible (the route record's
  included), every digest is kept, each run is live, each
  host line is carried byte for byte and the cited welds certify the environment they
  certified before. The admission edit moves `metal_dispatch.py`, which the Metal
  dispatch record binds: that record's contract test and the three Metal weld-contract
  tests that walk its digests are pending until the Apple M1 Max route campaign
  `dispatch_metal_route_2026-10-05_perarch_applegpu_g13s` and its recut.
- **`build_fusion_matrix.py` reads a weld's artifact and timestamp from its live
  runs**, and refuses a cell whose weld has no live run recording an artifact: the
  top-level comparison it replaces would read `None` after the move and pass without
  comparing.
- **`recut_driver_dispatch_record.py --backend metal` refuses a route row** whose
  plan did not read the GPU architecture, PyTorch and the Metal frontend all
  certified, as the NVIDIA recut already refused a row on an uncertified device.
- **The pytest resource `certified_metal_toolchain` is removed.** The tests it
  gated assert that the Metal kernels dispatch, which they now do on any Mac.
- **The Metal fleet cuts every expansion probe before the first gate that binds it.**
  `recut_metal_gates.sh` runs `gate_metal_complex` (whose expansion leg writes the
  complex-multiply probe) before every other gate, unsets probe variables inherited
  from the calling shell, and names a missing complex probe in its summary. Run in
  directory order, `gate_metal_beta_complex_fused_hd_pair` ran before that probe
  existed: on a Mac without the evidence archive its refusal leg failed, and on the
  Apple M1 Max a dated archive probe stood in.
- **A gate records the probe its loaders read.**
  `metal_composition_matrix.prepare_environment` returns the probe in force, the one
  the environment names, rather than the dated archive candidate it found, so
  `gate_metal_residue_fused_pairs` and `gate_metal_tranche7_fused_pairs` no longer
  refuse on a Mac without the archive. `gate_metal_complex_conductive_fused_pair` and
  `gate_metal_complex_fused_ade_chain` read the probe the environment names instead
  of a dated archive file, refuse by name without one, and record its SHA-256.
- **Owed on the Apple M1 Max.** These edits move bytes three Metal welds pin:
  `metal_composition_matrix.py` (`metal_cylindrical_real_fused_magnetic_pair`, cited),
  `gate_metal_complex_conductive_fused_pair.py` (cited) and
  `gate_metal_complex_fused_ade_chain.py` (not cited). `metal_composition_matrix.py`
  is also recorded by the whole-step artifact `test_metal_artifact_provenance` reads
  (`metal_whole_step_2026-10-02_arch3`). Until a Metal fleet re-runs on the M1 Max on
  these bytes, the rebind binds it and `CURRENT_EVIDENCE` names the new whole-step
  artifact, the two cited welds are not bound to the live sources (the other 43 of 45
  are), and the four tests that say so are listed in
  `tools/ci/pending_certification.txt` with that removal trigger.

### The identical-case timing ladder ships in the harness

- **`parity/meep_gpu/timing_ladder.py`** runs the GPU routes and stock MEEP on the
  same simulation, interleaved per size and pass on one host, with a quiet gate in
  front of every row, one ledger line per row, `--resume`, `--dry-run` (every command,
  no process) and `--selftest`. It runs on a Linux host directly or inside a Slurm
  allocation (`mpirun` or `srun`), and on an Apple silicon Mac. Every host-specific
  value of the 2026-09-28 ladder it replaces is a parameter, detected or given
  ([Running the timing ladder](docs/development/timing.md)).
- **The MEEP quiet gate can fail.** The 2026-09-28 ladder read busy CPUs as a busy
  fraction times `nproc`, which its own `OMP_NUM_THREADS=1` reduced to one, so its
  MEEP rows' gate could never fail. The busy count is now summed per CPU, over the
  host or the allocation, and an unreadable reading is not quiet.
- **Stronger MEEP baselines.** Per size the ladder can also time a native-tuned MEEP
  build (`build_meep_133_native.sh`, with a build record), MPI ranks with OpenMP
  threads, ranks bound to hardware threads, and cost-based chunk splitting, and it
  records the CPU governor, frequency and turbo state around every row and, on an
  NVIDIA GPU, its clocks, temperature and power (on a Mac, only the Apple GPU's
  utilisation: the rest needs root). Whether a MEEP build can run threads is read
  from the libraries MEEP itself links, not from a runtime OpenBLAS brings in.
- **More than one case.** `timing_cases.py` defines cases once for both benches:
  besides `pml_3d` (78 % PML, source off the fused seams), three cases with a thin
  PML and a structured interior, one with its source on the fused B seam and one
  with both pairs and the D-seam deposit repair. Every GPU row is checked against
  the deposit-repair bracket its case declares.
- **`build_3d_comparison.py` refuses a pair that is not the same simulation**, by
  name: a case label or geometry digest that differs, a MEEP row with no digest
  (every 2026-09-22 MEEP row), digests that disagree across passes, MEEP rows of
  mixed configurations, a harness lift whose driver disagreed with the MEEP object,
  a GPU row timed without the MEEP rows' monitors, a case built by another builder
  source, or a MEEP best that fails the 5 % gate. MEEP rows that failed a floor
  enter no median. It used to pair rows by cell count alone. The ladder's table
  builder strikes every ratio at a size whose harness-lift digest differs from the
  digest of record, or whose rows ran different code.
- **The ladder refuses what would make a row mean something else**: a GPU route the
  package does not resolve under the rows' own environment, an inherited
  `MEEP_GPU_*` switch (every one the ladder does not set is unset), a Slurm step or
  a partial allocation, a resume on another host or GPU, and on a Mac battery power,
  low power mode, a thermal limit or busy background daemons between rows.
- No published figure changes: the 2026-09-28 comparison stays as stated in
  [Will it help?](docs/guides/will-it-help.md), with its gate defect named there,
  until the release-tag campaign re-measures it.

### The certification reference environments are locked, and a user can compare against them

- **`environments/locks/`** holds each platform's certification reference as an
  explicit conda lock (every package by URL and MD5) and a pip requirements file (every
  PyPI package by version and wheel SHA-256): `meep-gpu-ref-osx-arm64.*`, the
  environment the Metal kernels were certified in on the Apple M1 Max restricted to the
  reference's packages (127 of its 138 conda artifacts identical to that environment's;
  hdf5 and h5py are the MPI builds by design, and 9 build-only packages are added), and
  `meep-gpu-ref-linux-64.*`, the environment of the 2026-10-03 RTX A6000 round, verbatim
  (143 conda and 21 PyPI packages). `numerics-relevant.json` names the packages whose
  builds can change what MEEP or the GPU route computes (29 on Apple silicon, measured
  from the certified MEEP's linkage; 36 on Linux, checked against the round MEEP's linkage). The directory's README
  says what each lock reproduces, how it was made and when it is re-cut.
- **The reference build scripts install the locks instead of solving.**
  `build_meep_133_macos.sh` and `build_meep_133_linux.sh` create the environment with
  `conda create --no-default-packages --file <lock>`, refuse to build MEEP unless the
  prefix holds exactly the lock's conda artifacts, install the PyPI part with
  `pip install --no-deps -r <lock>` and `pip check`, keep the lock files in the prefix,
  and end with the comparison below. Every stage, patch and assertion they had stays.
  Measured on 2026-10-03, the Mac script's former solve differed from the certified
  environment in 54 of the 139 packages they share (libctl 4.7.1 against 4.5.1, FFTW,
  harminv, SciPy, the compiler runtimes); solved for a glibc 2.35 host (with conda's
  glibc override), the Linux script's former solve took `sysroot_linux-64` 2.34 and
  kernel headers 5.14 in place of 2.28 and 4.18. `MEEP_REFERENCE_SOLVE=1` solves as
  before and labels the result as not the reference. The Mac script now refuses any
  platform but Apple silicon and keeps the macOS SDK the new `libmeep` records; the
  Linux script refuses, before installing anything, a glibc older than 2.28 or one it
  cannot read, and its final assertions add NVRTC 11.8.
- **Run and checked on 2026-10-04.** The Mac script installed the lock and built MEEP
  end to end on the Apple M1 Max in 3 min 10 s: 138 of 138 conda packages the lock's,
  MEEP linked against OpenBLAS (`BLAS_LIBS = -lopenblas`) as the certified MEEP is, every assertion
  passed, the comparison 29 of 29 same, and `tools/check_install.py --require-gpu
  --require-reference` gave the certified host's figures (GPU against reference 0,
  reference against MEEP 2.5e-6 and 9.2e-8). The 16 field-update kernels of that MEEP
  have the instruction sequences of the certified MEEP's (16 of 16). The Linux lock was
  checked against the round environment, not built: 143 of 143 lines identical to its
  export, 143 of 143 URLs answer with conda-forge's MD5 and SHA-256, 21 of 21 wheel
  hashes resolve.
- **`tools/compare_reference_environment.py`** compares the running environment (or a
  conda prefix) with its platform's lock: `same`, `different` with both builds,
  `missing`, `differs by design` or `not read` for every numerics-relevant package,
  then one verdict, "matches the certification reference" or "differs from the
  certification reference in N of D numerics-relevant packages: ...". A pip-only
  install is reported as differing (its NumPy, SciPy and mpi4py are PyPI builds), with
  the compiled libraries it cannot read counted as not read. The documented hdf5
  exception admits the certified environment's own build and MD5 and nothing else, and
  PyTorch on a Mac is judged by its wheel's build tag as well. It reads `conda-meta`
  and the installed distributions with the standard library only, checks the host's
  macOS or glibc floor, names a populated user site directory and a prefix a build
  script solved, and never raises. `--require-reference` makes anything but a match
  exit 1. It judges packages only.
- **`tools/check_install.py` prints it as step 7**, after a line that judges MEEP itself
  against the build the reference names (1.33.0, single precision, MPI build);
  informational by default, and with `--require-reference` the check fails unless both
  match. Steps 1 to 6, their output and their exit codes are unchanged.
- **`build_meep_133_native.sh`** finds the conda clang++ driver on a Mac
  (`*-clang++`; there is no plain `clang++` in the environment), so its record names
  the compiler and what `-mcpu=native` resolved to. **`.gitattributes`** keeps the lock
  files at Unix line endings, which step 1b's line comparison needs.
- **Both locks carry autograd** (2026-10-05), which MEEP's adjoint module and MEEP's
  adjoint tests import: 1.8.0 on Apple silicon, the version the certified environment
  holds, and 1.9.1 on Linux, the version the environment of the earlier A6000 rounds
  holds, each pinned to its PyPI wheel by SHA-256. Without it, the first corpus lift in
  the Linux round environment lost 13 adjoint rows across four families and 8 of its 60
  legs did not release. That environment had it installed from the lock line. autograd
  takes no part in the field updates, so `numerics-relevant.json` does not name it.
- No kernel, ledger or certified figure changes.

### The three weld contracts check the dispatch record through its route runs

- A table's `driver_dispatch` record is not a weld: its verdict is the route
  campaign's, which `recut_driver_dispatch_record.py` files under `runs[<key>]` (a
  compute capability on the NVIDIA tables, a GPU architecture on Metal). The CUDA,
  Triton and Metal weld contracts now hold it to one rule, in the same test,
  `test_the_dispatch_record_stands_on_a_live_passing_route_run`: at least one run is
  live, every live run reads `status` PASS, every file the record pins matches the
  tree, and no route-run field, `status` among them, sits beside the digests. An empty
  or fully stale `runs` fails rather than passing a loop over nothing. A second test in
  each suite works on a copy of the shipped record re-pinned to the tree: it requires
  one live PASS run to be accepted, and an entry-level status, an empty `runs`, a
  `runs` whose only run binds other bytes, and a live FAIL run each to be refused.
- **`meep_gpu/weld_record_walk.py`** gains `route_record_problems`, the rule, and
  `route_run_problems`, its run half. Both decide only from what the caller passes in
  (liveness from `fastpath.live_capabilities`, the contract's drift and shape
  measurements), so the module still reads no file and imports nothing from the
  package. `test_dispatch_contract.py` applies the run half to the NVIDIA and Metal
  records, so a live route run that did not release fails there too, and it now
  refuses a route-run field beside the NVIDIA records' digests, as it already did
  beside the Metal record's.
- The CUDA contract's `test_the_status_is_the_one_the_tree_supports` asks the weld
  status rule of weld entries only, as its sibling tests do. It used to ask
  `driver_dispatch` for an entry-level status, which since 0.9.1 no cut writes and the
  Triton and Metal contracts refuse; that request is gone rather than met. A test over
  a synthetic ledger keeps the rule armed: a weld with no status, a PASS over a moved
  pin and a DRIFTED over the tree are refused.
- `recut_driver_dispatch_record.py` is unchanged: every backend's cut files `status`
  under `runs[<key>]` only and refuses a record that carries one beside its digests. A
  new harness test, `parity/meep_gpu/test_recut_route_status.py`, drives a CUDA cut,
  checks the record it writes against the CUDA contract's rule, and checks the refusal
  on all three backends; `test_recut_metal_runs.py` checks the Metal cut's record
  against the Metal contract's rule.
- No ledger is rewritten. The three records fail the new test on bytes the round has
  not yet re-run (the CUDA and Triton records' route runs are live and read PASS, with
  3 and 1 pinned files moved; the Metal record has no live run and 2 moved), and the
  three entries are added to `tools/ci/pending_certification.txt` with the route
  campaign and recut that remove each.

### Round records

- **`parity/meep_gpu/rounds/metal_2026-10-04/`** keeps a Metal round for a Mac that
  does not hold the evidence archive: the manifest of the seven archive files the
  Metal gates read (14,040,445 bytes), the digest of the bundle cut for it, and
  `run_round.sh`, which checks the commit and the environment, unpacks and verifies
  the bundle before the first gate, runs the fleet, reports what released and packs
  the outputs for the rebind. `parity/meep_gpu/build_metal_round_inputs.py` packs,
  verifies and unpacks the bundle. The bundle itself is not in the repository.
- **`parity/meep_gpu/rounds/h200_2026-10-03/`** keeps the prepared H200
  certification round: its driver (`run_round.sh`) and CUDA-leg script and the
  manifest of the 992 evidence inputs its gates read. Its two runbooks, for the
  round and for the timing that follows the bind, and the script with which the
  RTX A6000 ran the same round on a pre-release commit (every gate released) are
  kept in the development repository. The round is prepared; no 9.0 record ships
  in 0.9.2. The inputs bundle itself is not in the repository; it travels in the
  hand-off file with the two scripts.
- The timing fixtures name the reference host by its card
  (`fixtures/timing/a6000_2026-09-28_b.params`) and carry placeholder paths.

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
- The certification reference is now built the same way on Linux as on a Mac:
  `parity/meep_gpu/build_meep_133_linux.sh` builds MEEP 1.33.0 from source in
  single precision with MPI and OpenMP, at the package versions of the environment
  the NVIDIA kernels were certified in, installs the certified CuPy, PyTorch and
  Triton beside it, and refuses to finish unless each is what it claims. A
  certification round is measured against this build on every host, never against
  conda-forge's double-precision MEEP, which remains the way to use the package.
- `parity/meep_gpu/build_meep_133_macos.sh` now builds the Metal certification
  reference the same way: pinned to the environment the Metal kernels were certified
  in, with the certified PyTorch 2.10.0 installed beside it, and refusing to finish
  unless MEEP is 1.33.0, single precision and an MPI build and NumPy, PyTorch and
  Metal are what it claims. Both scripts default to the environment `meep-gpu-ref`.
- The complex off-diagonal stencil gate refuses an expansion record measured on
  another compute capability, or one that names none, and takes `--expansion-probe`
  for the card's own record, as the other complex CUDA gates do. It used to read its
  records from fixed archive paths and check only their subnormal policy, so on a
  second architecture it would have licensed that card from the first card's
  measurement. On the card the archive records came from, nothing changes.
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

As of 0.9.0. Since 0.9.2 a supported NVIDIA GPU outside the certified set runs the
kernels by default, uncertified (see 0.9.2), and the certification round has run on
the release files.

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
