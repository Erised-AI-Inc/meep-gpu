# The 1.0 certification round: runbook

The drivers in this directory run the 1.0 certification round's device stages:
the NVIDIA gates and route campaigns on the GPU host (`round.py`, one
`run_lane.py` per GPU), and the Metal fleet and held route on a Mac
(`metal_round.py`). They run gates only. Every ledger, fingerprint, census and
board is written afterwards, on the host that holds the evidence archive, by the
integrator, from the commands `legs.py bind` prints. The owner commits.

| File | What it does |
|---|---|
| `legs.py` | The leg tables (one home for every gate the round runs), the lane planner, `check` (the no-GPU go/no-go), `flush-coverage`, `bind` (the bind commands, printed, never run) and `estimate` |
| `round.py` | The NVIDIA stages `gates` (CERT-3) and `route` (CERT-4): plan, launch detached, wait on lane PIDs |
| `run_lane.py` | One lane on one GPU: takes its own lock, runs its steps, releases the lock when the last step returns |
| `fleet_launch.py` | Runs the checkout's Triton fleet driver with timeouts that can only rise |
| `round_status.py` | One verdict reader for every artifact shape the round meets |
| `meep_host_import.py` | Runs a gate that has no `--import-meep-for-host-policy` flag with MEEP imported first (the flush host half) |
| `metal_round.py` | The Metal stages: preflight, fleet (CERT-3), route (CERT-4), status |
| `build_pymiescatt_standin.py`, `pymiescatt_standin.patch` | Builds the PyMieScatt stand-in two corpus rows import from its pinned sdist and this two-line patch, and installs it into a reference environment; the build scripts run it (CERT-1: the corpus environment) |
| `l3_corpus.py`, `l3_row.py` | CERT-5, L3: one leg of dispatched corpus runs judged against the stored MEEP digests of the Tier B record (`l3_corpus.py`), one row per process (`l3_row.py`); `test_l3_corpus.py` tests both on the host |

## 1. What changes from the 0.9.2 rounds

- **Both subnormal policies on both NVIDIA tables (R13).** CUDA already gated
  keep and flush. The Triton fleet now runs a second time under `--policy flush`
  into `_flush` roots, beside its keep roots, including the composition shard,
  the stencil welds and the bit identity.
- **The flush dispatch licence.** Until 1.0 every NVIDIA family dispatched under
  `keep` only. With the licence change applied, dispatch admits a
  caller-installed `flush` per arm, only where the arm's gate has a run on this
  compute capability that released under flush
  (`fastpath.TABLE_DISPATCH_POLICIES`, `fastpath.policy_evidence_reasons`). The
  bind tools write that evidence (`certified_policies`) from released legs. The
  route gate's flush legs are then dispatch legs, held to PASS-FUSED, not
  refusal legs. So the bind sits **between** the gates and the route stage.
- **The CUDA lane splits across GPUs** by whole campaign subdirectory: one
  campaign root per GPU, recorded by its own `record_cuda_regate.py --campaign`
  and rebound by its own `rebind_cuda_welds.py --campaign`. Both tools take one
  campaign per call, a certification block claims a whole subdirectory, and the
  rebind requires the re-gate to have cut exactly the legs an entry stands on.
  The 0.9.2 ledger already binds 31 welds from one campaign and 23 from another
  on the same capability.
- **Defects found by the 1.0 pre-certification passes, built in:**
  - the fleet driver's own timeouts govern and can only be raised
    (`fleet_launch.py`); the driver carries 10,800 s for `conductivity` and
    `symmetry` since entry 99, where a scheduler cap of 2,700 s had timed both
    out (at the final integration both released, in 5,148 s and 4,937 s, on a
    card other gates shared), and `fleet_launch.py`'s floors raise those two,
    `folded_complex` and `special_kz` to 3x or more the larger of each gate's
    CERT-3 and pre-certification walls;
  - `fused_offdiag_electric` passes by establishing a refusal, and the five
    composition legs pass on `summary.status`; `round_status.py` reads both;
  - no lock keeper and no bare `wait`: a lane takes its lock synchronously and
    releases it in a `finally` and in its signal handlers;
  - on the Mac, every item takes its slots in one call, all or nothing
    (`--slot-wrapper` with `--classes mac_gpu+mac_cpu`), so the one GPU slot is
    never held while a CPU slot is awaited.

## 2. Order and wall time

Bounds, per step. LOW is the plan `legs.py` prints from its leg tables: every CUDA
gate leg, Triton fleet gate and hand leg at its wall in the 1.0 round's CERT-3 NVIDIA
(stamp 2026-10-11_10; one RTX A6000 per lane, one gate at a time; a Triton gate at the
mean of its keep and flush walls), with 14 prices raised so that their step
watchdogs hold 3x the larger measured wall (`legs.py` says which), and each Triton H->D
gate at the larger of its two measured walls (`TRITON_HD_EXPECT`; CERT-3 has not run
them yet), which it keeps in both columns. HIGH prices every
gate leg at the larger of its CERT-3 wall and its wall in the 1.0 final
pre-certification pass (shared cards: five CUDA campaigns on one, four Triton gates
per card), packed the same way, so it over-estimates. The route and licence legs are
priced at that pass's measured walls in both columns (`legs.py`'s `ROUTE_CUDA`,
`ROUTE_TRITON`, `E2E_EXPECT`; every route lane and the Triton hand lane shared one
card), so CERT-4's row is `legs.py estimate`'s `route_s` per lane. The Metal rows are
one M1 Max.

| Step | What | Where | Wall LOW | Wall HIGH | Idle box |
|---|---|---|---|---|---|
| CERT-1 | Freeze: AP8 exit, L1′, AP9; re-mint `array_path_oracle` and `meep_identity.json`; **set the round's route stamp** (`ROUND` in `meep_gpu/route_stamps.py`, one line; the previous round's campaigns go into its `SPENT`; in a chunked round, the `round` its chunk indices name, never a new date: see CERT-3, chunked); apply the licence change; **the hygiene guard reads CLEAN** on the frozen tree (`python tools/ci/check_hygiene.py --legs tracked,subject`: no host name, account name, local path or personal e-mail address in a tracked file or the board's subject; R22) | Mac CPU | — | — | no |
| CERT-2 | Device smoke: `round.py run gates --only …` with one PML single and one fused pair per table, `metal_round.py fleet --only …` | GPU host, Mac GPU | ~15 min per table | — | no |
| CERT-3 NVIDIA | `round.py launch gates`: 4 lanes (2 CUDA campaigns, Triton keep + flush, the ten Triton H->D gates under keep) | GPU host, 4 GPUs | 23,991 s (6.7 h) | 31,607 s (8.8 h) | no |
| CERT-3 NVIDIA, one CUDA campaign | the same with `--cuda-split 1` | GPU host, 4 GPUs | 31,793 s (8.8 h) | 38,274 s (10.6 h) | no |
| CERT-3 Metal | `metal_round.py all` up to `fleet` | Mac GPU + CPU | 4,237–4,537 s | slot waits on a shared Mac | no |
| Bind | `legs.py bind` (records, rebinds, re-cuts, and last `neutralize_ledger_labels.py`, which gives the label leaves no re-bind replaces their neutral labels), then copy the rebound ledgers into the GPU host's checkout; `legs.py flush-coverage` must read COMPLETE, and `python tools/ci/check_hygiene.py --legs ledgers` CLEAN | evidence host CPU | ~20 min | — | no |
| CERT-4 NVIDIA | `round.py launch route`: CUDA route lane and Triton route lane plus the end-to-end licence, side by side on 2 GPUs | GPU host, 2 GPUs | 12,733 s (3.5 h, CUDA lane) | 12,733 s; Triton lane 9,235 s | no |
| CERT-4 Metal | `metal_round.py route` (3 lanes, held residency) | Mac GPU + CPU | 12,648 s (3.5 h, measured with 3 lanes on a loaded Mac) | 17,920 s (the four legs serially, `--route-lanes 1`) | no |
| CERT-4 records | `recut_driver_dispatch_record.py` per table, three censuses, boards, `board_controls` re-baseline (R8) | evidence host CPU, Mac CPU | ~50 min censuses | — | no |
| CERT-5 | L3 (`l3_corpus.py`): NVIDIA, both table variants against Linux flush and keep (R13); Metal per R1 | GPU host ×2, Mac GPU | NVIDIA 1,035–1,326 s per leg and 2,322–2,822 s per two-leg lane (smoke and per-leg card picks included), 3 rows at a time per card; Metal 4,285 s for 192 rows, one row at a time (R27 round, 2026-10-15) | — | no |
| CERT-6 | A new device (R4, the owner's): an H200 at 9.0 or an Apple M3, the chunked round with `--new-device` / `--new-capability` (section 4; for a contributor, `docs/development/certify-a-new-gpu.md`) | external | 4–5 h + build | — | no |
| CERT-7 | Suites per environment; release staging | Mac, GPU host | ~1 h per env | — | no |
| CERT-8 | **Timing, last**: identical-case ladder, Apple ladder, fused-product and dispatch-preference records; **MEEP rows re-timed in the same session (R5)** | GPU host idle, Mac idle | 16,926 s ladder; Apple 4.4–6 h | — | **yes** |

GPU-hours for CERT-3 on the NVIDIA tables: 26.5 (LOW) to 35.0 (HIGH)
A6000-hours (CERT-3 of stamp 2026-10-11_10 held 22.5 A6000-hours of locks, without the
H->D gates). At LOW the four lanes are 23,790, 23,954, 23,808 and 23,991 s: each Triton
core unit (fleet, shard B, shard C, composition, stencil welds, bit identity) shares its
lane with tier-2 units and H->D gates, and each CUDA campaign with tier-2 units and H->D
gates. The ten Triton H->D gates (`legs.py` `TRITON_HD`, one movable unit each) are
priced at their walls in the 1.0 pre-certification pass, two streams of five on one
card beside that pass's seven other streams: 12,746 s in all, of which seven measured
0.21 to 0.43 of their price when re-run two at a time on the frozen tree of the 1.0
round, so they over-pack their lanes until the round measures them. Without them the
plan was 21,165 s, which re-summed at CERT-3's own walls spans 19,649 to 20,798 s
(CERT-3's plan ran 15,650 to 24,675 s); the planner packs whole units longest first,
and an exhaustive search over the same units found 20,893 s on the prices. With one
CUDA campaign that campaign is the wall. `legs.py estimate` and `legs.py plan lanes --gpus N` print the
plan for other GPU counts.

The steps that need an idle box are CERT-8's: the identical-case ladder and the
R5 same-session MEEP re-time on the GPU host (no other process on the account
or the box, as the timing rules require), the Apple ladder on a quiet Mac, and
the fused-product and dispatch-preference records. Nothing in CERT-2 to CERT-7
times anything: those gates are bit-identity checks, and a neighbour costs wall
time only.

## 3. Preflight

First, on each machine, stage the round's inputs (`INPUT_FILES.txt` in this directory)
from the private bundle, `python parity/meep_gpu/rounds/1.0/round_inputs.py unpack --bundle
<round_inputs_1_0.tgz>` (on the host that holds the evidence archive: `check`), GO only on
`N of N verified`, and derive the corpus manifest CERT-1 and CERT-5 read from the staged
legs: `python parity/meep_gpu/meep_identity_corpus.py manifest --results
parity/meep_gpu/results --out <corpus manifest>`.

On the GPU host, in the reference environment, from a clean clone of the frozen
commit with the round's inputs staged under `parity/meep_gpu/results/`:

```
python parity/meep_gpu/rounds/1.0/legs.py check --repo . --capability 8.6 \
    --git-expect <frozen commit> --fresh-stamp <stamp> \
    --python <reference environment>/bin/python
```

GO only on `CHECK OK`. It refuses on: a missing gate script or tool; a fleet
gate the round does not run or one the fleet driver does not know; the flush
licence absent; with `--route`, a route campaign name `meep_gpu/route_stamps.py`
refuses (the stamp still its placeholder, or a campaign that has already run; without
`--route` the check records it under `route_campaign_refusals` and does not refuse, so gate
chunks may run before the freeze sets the stamp); route legs that differ from what
`recut_driver_dispatch_record.BACKENDS` requires; a capability either table does
not admit; a dirty tree or another HEAD; any results root of the stamp present;
a MEEP corpus root without `examples/` and `tests/` (the four CUDA H->D legs lift
corpus rows from there); a census or seam record the ten Triton H->D legs lift from
(each gate's `CENSUS` and `SEAM_RECORD`) absent from `parity/meep_gpu/results/`, or
lacking a file their lift reads; a hit of the hygiene guard
(`tools/ci/check_hygiene.py`) in a tracked file or the board's subject, and, under
`--route`, in the four ledgers the bind wrote.
Each lane then runs a card and toolchain check as its first step (compute
capability, CuPy 13.5.1, Triton 3.1.0, MEEP 1.33.0 single precision with MPI,
the package imported from the clone) and stops on a refusal.

Environment rules the lanes enforce (every child environment is in the table at the
end of this section): `CUPY_ACCELERATORS` empty in every child but the licence legs,
every `MEEP_GPU_*` and `TRIDENT_FDTD_*` variable removed by prefix and the round's own
set per leg, the user site directory off (`PYTHONNOUSERSITE=1`; a lane whose interpreter
still enables it stops before it takes a GPU), `MEEP_GPU_DISPATCH=0` on
every gate, the Triton launcher's toolchain in every child (`legs.launcher_toolchain`):
`CC` the reference environment's own `<prefix>/bin/x86_64-conda-linux-gnu-cc` and
`TRITON_LIBCUDA_PATH` the libcuda link its build made (`<prefix>/lib/meep-gpu-libcuda`, named
in `<prefix>/share/meep-gpu-build/libcuda.json`), unless the shell sets `CC` or
`--libcuda-stub` (default `$TRITON_LIBCUDA_PATH`) names another directory; a lane stops before
it takes a GPU when the environment lacks either, since Triton would otherwise compile every
launcher with the host's `gcc`, outside the environment; the keep caches minted by
the package under a path carrying `ftz_stripped`, the flush caches plain, and the
MEEP corpus named once by the round (`legs.resolve_corpus`, the rule `metal_round.py`
applies too): `--corpus-root`, else the root the build record of `--python`'s reference
environment names (`<prefix>/share/meep-gpu-build/reference_build.json`, `corpus`: the
release tarball's `python/` tree kept in the prefix by the build, and the list of every
file's sha256 beside it, and the corpus siblings: the files outside `python/` that corpus
rows read, kept beside it by name and sha256); the shell's variables are not read. The tree is checked against
that list (every listed file present at its sha256, every sibling present beside the root at
its sha256, no unlisted file Python could import:
source, byte code outside `__pycache__`, an extension module; a record that names no
siblings is refused, and `build_meep_133_<platform>.sh --refresh-corpus` brings it up to
date) and for `examples/` and `tests/` before a lane with an
H->D lift leg takes a GPU, and exported to every child as `MEEP_GPU_CORPUS_ROOT`; the gates'
own lookup chain is unchanged. MEASURED on the GPU host: none of the gates' defaults exists there, and without
a corpus every CUDA H->D lift refuses. The
account's GPU cap is counted in `gpu_lock_*` directories under `--lock-dir`
(`--max-locks`, default 4); a lane takes a card only after two idle readings
`--poll` seconds apart (no compute process, no utilisation, at most
`--idle-mib` MiB). A card that holds another account's idle compute app is taken only by
opt-in (owner ruling R28): `round.py launch gates ... --share-idle-foreign <uuid>:<pid>[,<pid>]
--share-min-free-mib <the unit's peak + 8,192>` admits that card when, on both readings, every
compute app on it is a named PID, the named PIDs and the card read 0 % utilisation and the
free memory reaches the bound; the lane names the shared card and the PIDs in its
`progress.log` and its lock's label. Yielding the card when its owner needs it again is the
scheduler's: a SIGTERM to the card's `run_lane.py` stops its gate's process group and releases
the lock, and the interrupted unit runs again alone (`chunks.py add --units`). The opt-in is
`run_lane.py`'s card admission only. The Triton fleet driver (`drive_triton_weld_gates.py`)
still refuses a card that carries any compute process, so a unit it drives (`triton_fleet`,
`triton/composition`) needs an unshared card: in the R27 round, on a shared card, it refused
the 10 gates of a flush fleet shard and its environment probe, 11 legs in which no gate ran.
The memory bound is the unit's own measured peak plus what the owner keeps, never a smoke's
reading: a Triton H->D lift there used 12,348 MiB, where the bound had been set from the
270 MiB a smoke read.

On the Mac: `metal_round.py preflight` (commit, clean tree, no shaping switch,
the package from the checkout, `tools/check_install.py --require-reference`,
the Metal environment, the disk floor, no output of the stamp).

The Metal preflight also runs the lift basis of each of the nine gates that launch lift
children, in one child under `--python` with torch unimportable. It stops, naming the
gate and the file, when one of the four files they read is absent from `results/`:
`metal_coverage_2026-09-04_m0complex/{examples,tests,tests_param_matched}.jsonl` and
`h_to_d_seam_2026-09-04/h_to_d_seam.jsonl`. It checks the same four files for
`gate_metal_complex_fused_hd_pair`, which reads them in process. The release tree
carries none of them, so stage them first. It lists by gate each row this host cannot
lift (a need its census lift record declares, `lift_host.LIFT_RECORD_NEEDS`, probed
absent under `--python`) and does not stop for those; on the reference environment
there are none. It checks
the MEEP corpus the lift legs walk, file by file, against the sha256 list the reference
environment's build record names, and its siblings by name (the `corpus` entry of
`<prefix>/share/meep-gpu-build/reference_build.json`; `--corpus-root` names another
copy, whose siblings must sit beside it), and stops on a mismatch.

Before the freeze, on the Mac, the route gate's controls dry run on the candidate,
against the shipped leg of the last released Metal route campaign:

```
python -u parity/meep_gpu/gate_dispatch_metal_route.py --controls-dry-run \
    --out <scratch dir> --dry-run-reference <campaign>/shipped/controls.jsonl
```

GO only on `CONTROLS DRY RUN OK`: every case lifted, every wrapper in its plan named
in `ROUTE_WRAPPER_RULINGS`, no residency control reading NOT-ARMED or
HARNESS-FAILURE from its state, and no case moved from the reference. Every case it
lists in `smoke_cases` (a must-diverge control arms there, or the reference never
graded it) runs through the full gate's shipped leg before the freeze.

### The child environments

Every process the round starts runs in the platform's reference environment
(`environments/locks/`), built by the repository's own build script, and never in a
development environment: that holds for the CERT-4 records, the censuses and the boards
as much as for the gates. `<prefix>` is that environment; `<corpus root>` is the MEEP
corpus its build kept (the MEEP 1.33.0 release tarball's `python/` tree, named with its
file list in `<prefix>/share/meep-gpu-build/reference_build.json`, with the corpus
siblings two `test_simulation.py` rows read from `<corpus root>/../tests`). Start every driver
below from a shell that exports `PYTHONNOUSERSITE=1`: the drivers set it for their own
children, and the commands run by hand inherit it. "Shell" means the starting shell's environment with the
removals named; `env -i` means a fresh environment holding only what the row sets.

| Step | Started by | Interpreter, user site | Starts from | Removed | Threads | Set |
|---|---|---|---|---|---|---|
| CERT-1 corpus, one `meep_identity_tier_c.py compare` process per row | the integrator | `<prefix>/bin/python -s` (the 1.0 round's rows ran without `-s`) | `env -i`: `HOME`, `USER`, `LANG`, `PATH=<prefix>/bin:/usr/bin:/bin` | everything else | `OMP`, `MKL`, `OPENBLAS` `_NUM_THREADS=1` (flush refuses any other OpenMP count), and `VECLIB_MAXIMUM_THREADS=1` on macOS | `PYTHONPATH=<checkout>`, `MEEP_GPU_DISPATCH=0`, `MGPU_SITE_MEEP_SOURCE=<corpus root>/..`, `MEEP_IDENTITY_CORPUS_MANIFEST=<corpus manifest>`, `MEEP_IDENTITY_CASE_TIMEOUT` (the largest rows), `MPLBACKEND=Agg`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR`; Linux: `CUDA_VISIBLE_DEVICES=` (empty); macOS: `KMP_DUPLICATE_LIB_OK=TRUE` |
| CERT-2/3/4 NVIDIA, every lane child but the licence legs: CUDA gate legs, Triton hand legs (H->D, stencil welds, bit identity), route legs, the toolchain check, the keep-cache mint, the device identity, the licence verdict | `run_lane.py`, which `round.py` launches | `--python`, `PYTHONNOUSERSITE=1`; the lane stops before it takes a GPU when the interpreter still enables the user site, and the toolchain step refuses it on the card | shell | `MEEP_GPU_*`, `TRIDENT_FDTD_*` (by prefix), `CUPY_CACHE_DIR`, `TRITON_CACHE_DIR` | `OMP`, `MKL`, `OPENBLAS`, `NUMEXPR` `_NUM_THREADS=1` | every child: `CUDA_VISIBLE_DEVICES=<card>`, `CUPY_ACCELERATORS=` (empty), `PYTHONPATH=<checkout>:<harness>`, `KMP_DUPLICATE_LIB_OK=TRUE`, `MPLBACKEND=Agg`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, the toolchain (`CC`, and the libcuda directory on `TRITON_LIBCUDA_PATH` and first on `LD_LIBRARY_PATH`), `MEEP_GPU_CORPUS_ROOT`. CUDA gate leg: `MEEP_GPU_DISPATCH=0`, `MEEP_GPU_BACKEND_PREFERENCE=cuda`, its own CuPy cache (`cupy_ftz_stripped_<leg>` under keep, `cupy_plain_<leg>` under flush) and Triton cache. H->D hand leg: `MEEP_GPU_DISPATCH=0`, `MEEP_GPU_SUBNORMAL_POLICY`, `MEEP_GPU_HARDWARE=certified`, its caches (keep minted by the package). Stencil and bit identity: `MEEP_GPU_DISPATCH=0`, their caches. Route leg: its caches (keep minted by the package). Flush host half: `meep_host_import.py` first |
| CERT-3 Triton fleet gates | `run_lane.py` → `fleet_launch.py` → `drive_triton_weld_gates.py` | as above; `fleet_launch.py` refuses an interpreter that enables the user site, or a `MEEP_GPU_*`/`TRIDENT_FDTD_*` variable other than `MEEP_GPU_CORPUS_ROOT` | the lane's child environment above | the fleet driver's per gate: `MEEP_GPU_ALLOW_UNCERTIFIED` | `OMP`, `MKL`, `OPENBLAS` `_NUM_THREADS=1` | `fleet_launch.py`: `PYTHONNOUSERSITE=1`, `CUPY_ACCELERATORS=` (empty), the toolchain (the lane's; started alone, its interpreter's environment's, refused by name when that environment lacks its compiler); per gate (`gate_environment`): `MEEP_GPU_SUBNORMAL_POLICY=<policy>`, `MEEP_GPU_DISPATCH=0`, `MEEP_GPU_HARDWARE=certified`, a private CuPy cache (`cupy_cache_ftz_stripped_<gate>` or `cupy_cache_flush_<gate>`) and Triton cache, the stub, the checkout and harness ahead of the inherited `PYTHONPATH`; under flush every gate starts through `meep_host_import.py` |
| CERT-4 NVIDIA licence legs (plan-only preflight, ship, null) | `run_lane.py` | `--python`, `PYTHONNOUSERSITE=1`, checked before a GPU is taken | `env -i`: `HOME`, `USER`, `LOGNAME`, `LANG`, `PATH=<python's directory>:/usr/local/bin:/usr/bin:/bin`, `CUDA_PATH` and `CUDA_HOME` when set | everything else; no `CUPY_ACCELERATORS` and no `MEEP_GPU_*` but the log: the licence verdict refuses both by name | `OMP`, `MKL`, `OPENBLAS`, `NUMEXPR` `_NUM_THREADS=1` | `CUDA_VISIBLE_DEVICES`, `PYTHONPATH=<checkout>`, `KMP_DUPLICATE_LIB_OK=TRUE`, `MPLBACKEND=Agg`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, its CuPy and Triton caches, `MEEP_GPU_DISPATCH_LOG`, the toolchain (`CC`, and the libcuda directory on `TRITON_LIBCUDA_PATH` and `LD_LIBRARY_PATH`) |
| CERT-2/3 Metal fleet, CERT-4 Metal route | `metal_round.py` (through `run_in_slot.sh` with `--slot-wrapper`); the route through `run_metal_dispatch_campaign.sh` | `--python`, `PYTHONNOUSERSITE=1`; the preflight refuses a child interpreter that still enables the user site | shell | the names in `metal_round.SCRUBBED`, each refused at preflight when the shell exports it | inherited | every child: `KMP_DUPLICATE_LIB_OK=TRUE`, `MPLBACKEND=Agg`, `PYTHONUNBUFFERED=1`, `PYTHONPATH=<checkout>`, `PYTHONNOUSERSITE=1`, `MGPU_SITE_PYTHON`, `MGPU_SITE_SOURCE_ROOT`, and `MGPU_SITE_MEEP_SOURCE` when `--meep-source` names one. Fleet gate: `MEEP_GPU_SUBNORMAL_POLICY=flush`, `MEEP_GPU_DISPATCH=0`, `MEEP_GPU_METAL_RESIDENCY=shipped`, `MEEP_GPU_GATE_SOURCE_MANIFEST`, `MEEP_GPU_CORPUS_ROOT=<corpus root>` (`--corpus-root`, else the build record's; verified against the record's file list, the rule `legs.py` holds). Route leg: the campaign driver's per-leg set |
| CERT-4 records, on the evidence host: `recut_driver_dispatch_record.py` per table, `probe_h_to_d_seam.py`, the three censuses (`measure_predicate_coverage.py`, then `match_param_rows.py --census`), `repoint_board_bindings.py`, the three boards, the certification suite behind the pending list | the integrator | `<prefix>/bin/python -s -u` | `env -i`: `HOME`, `USER`, `LANG`, `PATH=<prefix>/bin:/usr/bin:/bin` | everything else | `OMP_NUM_THREADS=1` | every command: `PYTHONPATH=<checkout>`, `PYTHONDONTWRITEBYTECODE=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, `MPLBACKEND=Agg`. Probe and censuses: `MEEP_GPU_CORPUS_ROOT=<corpus root>`; a census row whose lift record names another interpreter runs under this one by `--interpreter <recorded>=<prefix>/bin/python`, never under the recorded one. Boards: `MEEP_GPU_H_TO_D_PROBE=<probe>` and the board's census (`MEEP_GPU_CUDA_CENSUS`; `MEEP_GPU_TRITON_CENSUS` with `--census` and `MEEP_GPU_METAL_LIVENESS_CENSUS`; `MEEP_GPU_METAL_CENSUS`), `MEEP_GPU_DISPATCH=0` on the two NVIDIA boards; on these commands only. The Metal census and board on the Mac, in a `mac_gpu+mac_cpu` slot |
| CERT-5 L3 rows | `l3_corpus.py` (the leg) → `l3_row.py` (one process per row) | `--python`, run as `python -s -u`; `PYTHONNOUSERSITE=1` | shell, or `env -i` with `--clean-env`: `HOME`, `USER`, `LOGNAME`, `LANG=C`, `PATH=<prefix>/bin:/usr/bin:/bin` | `MEEP_GPU_*`, `TRIDENT_FDTD_*` (by prefix), `PYTHONPATH`, `CUPY_CACHE_DIR`, `TRITON_CACHE_DIR` | `OMP`, `MKL`, `OPENBLAS`, `NUMEXPR` `_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` | `PYTHONPATH=[--meep-site:]<checkout>`, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `KMP_DUPLICATE_LIB_OK=TRUE`, `MPLBACKEND=Agg`, `TMPDIR`, `MEEP_GPU_CORPUS_ROOT=<corpus root>` and `MGPU_SITE_MEEP_SOURCE=<corpus root>/..` (`--corpus-root`, else the build record's; verified against the record's file list), `MEEP_IDENTITY_CORPUS_MANIFEST`; per row `MEEP_GPU_DISPATCH_LOG`. NVIDIA: `CUDA_VISIBLE_DEVICES`, `CUPY_ACCELERATORS=` (empty; unset under `--shipped-default`), its caches (keep minted by the package), the toolchain with or without `--clean-env` (`CC`, and `--libcuda-stub` else the libcuda directory on `TRITON_LIBCUDA_PATH` and first on `LD_LIBRARY_PATH`; recorded in `leg.json`, the leg refused by name when the environment lacks its compiler); `--table cuda`: `MEEP_GPU_BACKEND_PREFERENCE=cuda` |
| CERT-7 suites | the integrator | `<prefix>/bin/python`, `PYTHONNOUSERSITE=1` | shell with `HOME` an empty directory | every `MEEP_GPU_*`; `CUPY_CACHE_DIR`, `TRITON_CACHE_DIR` | `OMP_NUM_THREADS=1` | NVIDIA: `CUDA_VISIBLE_DEVICES=<card>`, `CUPY_ACCELERATORS=` (empty), `CC=<prefix>/bin/x86_64-conda-linux-gnu-cc`, `TRITON_LIBCUDA_PATH=<prefix>/lib/meep-gpu-libcuda` and first on `LD_LIBRARY_PATH`; each suite's `conftest.py` pins `MEEP_GPU_DISPATCH=0` (`docs/development/testing.md`, "On an NVIDIA host") |
| CERT-8 GPU rows | `timing_ladder.py` → `bench_timing_case.py` / `bench_fused_products.py`, `digest_harness_lift.py`, `gpu_array_control.py` | `--python`; the user site from the launching shell, so export `PYTHONNOUSERSITE=1` there | shell | `MEEP_GPU_*` (the ladder's own set aside), `MEEP_GPU_ALLOW_UNCERTIFIED`; Metal rows also `PYTORCH_ENABLE_MPS_FALLBACK`, `PYTORCH_MPS_*` | `OMP`, `MKL`, `OPENBLAS`, `NUMEXPR` `_NUM_THREADS=1`, which the ladder sets for itself and every row | the ladder's own: `MPLBACKEND=Agg`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`; `PYTHONPATH=<checkout>:<harness>[:inherited]`; NVIDIA: `MEEP_GPU_HARDWARE=certified`, `CUDA_VISIBLE_DEVICES`, `CUPY_ACCELERATORS=` (empty), a Triton cache, the keep CuPy cache the package minted, the toolchain `cert8.py run` exports to every unit (`CC`, and the libcuda directory on `TRITON_LIBCUDA_PATH`; a site's `LIBCUDA_STUB` overrides the directory), `MEEP_GPU_SUBNORMAL_POLICY=keep` (`--subnormal-policy`); `--routes cuda`: `MEEP_GPU_BACKEND_PREFERENCE=cuda`; macOS: `KMP_DUPLICATE_LIB_OK=TRUE`; `MEEP_GPU_CORPUS_ROOT` from `--corpus-root` (else `$MGPU_SITE_MEEP_SOURCE/python` when it exists) |
| CERT-8 MEEP rows | `timing_ladder.py` → `mpirun` (or `srun`) → `bench_meep_identical_case.py` | `--python` (or a `--meep-build`'s), the user site from the launching shell | shell | the package switches, `MEEP_GPU_SUBNORMAL_POLICY`, `MEEP_GPU_CORPUS_ROOT`, `MEEP_GPU_HARDWARE`, `MEEP_GPU_ALLOW_UNCERTIFIED`, `CUPY_ACCELERATORS`, `TRITON_LIBCUDA_PATH`, both caches, every other `MEEP_GPU_*`; macOS: `KMP_DUPLICATE_LIB_OK` | `OMP_NUM_THREADS=<threads per rank>` (1 but in the threaded controls), `MKL`, `OPENBLAS`, `NUMEXPR` `_NUM_THREADS=1` | `CUDA_VISIBLE_DEVICES=` (empty), `PYTHONPATH` and `LD_LIBRARY_PATH` as inherited; threaded controls: `OMP_PROC_BIND=close`, `OMP_PLACES=cores` (`docs/development/timing.md`) |

A step that starts from the shell inherits what the table does not set, so that shell
exports no CuPy or Triton compile switch (`CUPY_COMPILE_WITH_PTX`, `TRITON_INTERPRET`,
`TRITON_KERNEL_OVERRIDE`, `TRITON_F32_DEFAULT`, ...): no driver removes them. The L3 leg
records the ones it inherits (`leg.json`, `environment`); the licence legs, which start
from `env -i`, record every `CUPY_*`, `TRITON_*` and `CUDA_*` variable they hold
(`environment_at_start`).

## 4. Each step: run, go/no-go, resume

### CERT-1: the corpus environment

The freeze reads Table 2 (`docs/development/meep-identity-corpus.md`): the corpus
judged by Tier C on four references, one
`meep_identity_tier_c.py compare --cases-file parity/meep_gpu/meep_identity_corpus.py`
process per row, in each host's certification environment (`environments/locks/`)
as it is, with nothing beside it. The environment carries what the corpus imports and no
field update reads: the corpus-only packages (`environments/locks/numerics-relevant.json`,
`corpus_only`), which the build scripts install with the rest of the lock (step 7):

- **shapely 2.1.2 and nlopt 2.11.0**, pinned by SHA-256 in each platform's pip lock
  (`meep-gpu-ref-osx-arm64.pip.txt`, `meep-gpu-ref-linux-64.pip.txt`).
  `ex.waveguide_crossing` imports nlopt; the stand-in below imports shapely.
- **The PyMieScatt stand-in**, which `ex.mie_scattering` and
  `ex.differential_cross_section` import: PyMieScatt 1.8.1.1, whose last release imports
  `trapz` from `scipy.integrate` (removed in SciPy 1.14; the references hold 1.17.1 and
  1.15.2), with the two lines of `pymiescatt_standin.patch` taking it from NumPy.
  `build_pymiescatt_standin.py` fetches the sdist (`PyMieScatt-1.8.1.1.tar.gz`, sha256
  `e628096cc0f3560f8170e0280df1d3196731e12bb55be214defc763c46666568`), applies the
  patch, checks the five package files against their pinned SHA-256, builds a wheel with
  the environment's own setuptools (no build isolation, no index) and installs it with
  `--no-deps`. Its source is not in the repository.
- **`ffmpeg`**, for `test_animation_output`, which
  `unittest.skipIf(call(["which", "ffmpeg"]) != 0, ...)` skips (and the corpus then
  refuses) when `which ffmpeg` fails on the reference `PATH`
  (`<prefix>/bin:/usr/bin:/bin`) as the test module is imported. It is the executable of
  the imageio-ffmpeg pin in each platform's pip lock, which the build scripts' step 7b
  links into `<prefix>/bin/ffmpeg` (the wheel's executable loads no library outside the
  operating system, so nothing reaches the libraries MEEP links against). The row's
  capture stops at the test's first `sim.run`, before `ffmpeg` would be called.

An environment built before these packages joined the lock, or before libGDSII and the
other additions of 2026-10-14 (`environments/locks/additions.json`), is rebuilt by the
platform's build script, not patched: its MEEP lacks libGDSII. An environment built before
the build kept the corpus siblings (its record's `corpus` entry has no `siblings`; every
round reader refuses it by name) needs no rebuild: the same script's `--refresh-corpus`
adds the siblings from the pinned tarball and rewrites only that entry. Without them
`test_simulation.py`'s `test_epsilon_input_file` aborts and `test_numpy_epsilon` measures
vacuum. The build's last steps check the result:

```
<prefix>/bin/python -s tools/compare_reference_environment.py --prefix <prefix> \
    --require-pip-lock --require-reference
```

which reads the corpus-only packages apart from the numerics-relevant ones: they never
count in the verdict, and `--require-pip-lock` requires the pip lock's pins and the
stand-in, exactly.

Each row runs with the checkout on `PYTHONPATH`, the `PATH` above, `OMP_NUM_THREADS=1`,
`MEEP_GPU_DISPATCH=0`, `MGPU_SITE_MEEP_SOURCE=<prefix>/share/meep-gpu-build/meep-1.33.0` (the
corpus the build kept) and
`MEEP_IDENTITY_CORPUS_MANIFEST=<corpus manifest>`.

- **GO:** every row holds, or is Table 2's named exclusion or one of its refusals by
  name, and every row's subnormal stamp (`subnormals` and `subnormal_mechanism` in its
  `result.json`) equals the other rows' of its host and mode: the corpus-only packages
  move no FPU state.

### CERT-3, NVIDIA gates

```
python parity/meep_gpu/rounds/1.0/round.py launch gates --repo <clone> \
    --stamp <stamp> --capability 8.6 --home <lanes home> --lock-dir <locks> \
    --expect <frozen commit> --gpus 4 --cuda-split 2 \
    --python <reference environment>/bin/python
```

Follow `<lanes home>/round.log` and `<lanes home>/*/progress.log` (one line per
step with N of D passing, a heartbeat every 120 s from the running gate).

- **GO to bind:** `GATES_DONE` present and `round.py status --home <lanes home>`
  reads every step passing. Every CUDA campaign must be entirely released:
  `record_cuda_regate.py` refuses a campaign with one unreleased canonical leg.
- **NO-GO:** any step not released. Read its row in `steps.jsonl` (state, reasons,
  log). A gate that fails on the frozen bytes needs a fix and a new freeze; a
  gate that died of the environment is re-run (below).
- **Resume** (session limit, connection loss, a killed lane): rerun the same
  `launch` command with `--resume`. Each lane skips a step whose last row and
  artifact pass. A CUDA leg directory holding an unreleased payload is refused
  by name: move it out of the campaign by hand, keep it, and rerun. A fleet step
  runs only its gates without a passing artifact, into a fresh continuation
  root (`<root>_r2`, `_r3`), which the bind commands search first. A lane that
  died holding a lock: its `gpu_lock_<n>` directory names the pid; remove it
  only when that pid is gone. To stop a lane between steps, create
  `<lanes home>/<lane>/STOP`.

### CERT-3, Metal fleet

```
python parity/meep_gpu/rounds/1.0/metal_round.py all --repo <clone> --stamp <stamp> \
    --expect <frozen commit> --meep-source <MEEP 1.33.0 source tree> \
    [--slot-wrapper <run_in_slot.sh> --slot-log <file>]
```

(`all` runs `preflight`, `fleet` and `status`, and exits with `status`'s code: non-zero
when a gate of the fleet did not release. `route` runs on its own after the bind.)

- **GO:** the environment record cut before and after the gates, every gate
  artifact released, `status` reads every cited weld released.
  (`--skip-environment` skips the environment record, for a dry run only.)
- **Lift legs:** every lift child runs under `--python`, never under an interpreter a
  census row records. On the reference environment all 145 lift rows lift. On a host
  that lacks a declared need, each affected row is named under `unlifted_on_this_host`
  in its gate's lift leg and counted out of that leg's denominator, and the preflight
  lists it. The corpus reaches every gate as `MEEP_GPU_CORPUS_ROOT`, from the build
  record or `--corpus-root`; one exported in the calling shell is refused.
- **Resume:** rerun the same command; items with a marker in
  `logs_metal_<stamp>/done/` are skipped. To rerun one gate, delete its marker
  and move its `results/metal_<family>_<stamp>/` out of `results/` (keep it).
- **A gate whose bytes moved after its fleet ran cannot rejoin that stamp.**
  Every gate of a stamp writes to one campaign source manifest
  (`results/metal_campaign_<stamp>/fleet/source_sha256.txt`, `metal_round.py`'s
  fleet environment), and `metal_gate_runner.py` refuses to release a gate whose
  imported file's digest disagrees with the one already recorded there. Rerun such
  gates under a fresh stamp with `metal_round.py fleet --only <gate stems> --stamp
  <fresh>` (it records the environment before and after, which the rebind needs),
  then `rebind_metal_welds.py --stamp <fresh>`: it rebinds the entries that stamp
  holds artifacts for and reports the others "without an artifact", leaving them
  bound to the round's stamp.

### CERT-3, chunked (R22)

A chunked round runs CERT-3 as chunks, each under its own one-shot stamp on its own card or
slot, while series entries are still landing (`chunks.py`; its docstring is the reference).
A chunk is any set of the round's units: under per-kernel certification a unit's evidence is
its own run's, judged by its own import record against the release tree, so any subset can
run at any time on any slot, a unit can run again alone, the plan can be re-partitioned
freely, and nothing about one unit's run can invalidate another's. The route stage (CERT-4)
is not chunked: it keeps one route stamp from the route-stamp registry and runs after the
bind.

```
cd parity/meep_gpu
python rounds/1.0/chunks.py partition --repo <clone> --machine nvidia --capability 8.6 \
    --out <index> [--defer <files still changing> ... --prior-records <records>]
python rounds/1.0/chunks.py partition --repo <clone> --machine metal \
    --architecture applegpu_g13s --out <metal index>
python rounds/1.0/chunks.py check --index <index> --repo <clone> --results results
```

- **The index** names the round (read from the route-stamp registry), the device it
  certifies (`cc86`; a GPU architecture on Metal) and whether that device is `certified`
  or `new`, and per chunk its units and stamp `<round>_<device>_c<k>` (`_m<k>` on Metal).
  A chunk id is allocated once and never reused. Keep the index outside the tracked files,
  one per machine and device. The default partition is four NVIDIA chunks balanced by the
  walls in `legs.py` and three Metal chunks balanced by `metal_round.EXPECT`; a unit keeps
  both policies of its gate together. `--defer` packs the units whose closure holds a
  still-changing file apart from the rest.
- **The round stamp is chosen when the indices are cut.** While `ROUND` reads its
  placeholder, `partition --round <stamp>` names it, and the freeze (CERT-1) then sets
  `ROUND` to that stamp, read from the index (`chunks.py show --index <index>`), not to
  the date the freeze is cut: the chunks and the route campaigns share one round.
  Once `ROUND` is set, `chunks.py check`, `legs.py check --chunk-index`, the launch and
  the Metal preflight refuse an index of another round, so a freeze that picks another
  stamp sends every launched unit back to run again.
- **Ready.** A chunk launches once its closure is final:
  `chunks.py ready --index <index> --repo <pushed tip> --changing-file <open lanes'
  files> --prior-records <records>` prints `READY_CHUNKS`. While the tip's
  `import_closure.py` misses reaches through the package root's `_LAZY` table, the
  previous round's import records are required (`chunks.py records --repo <tip>
  --results <previous round's results> --stamp <previous stamp> --machine <m> --out
  <records>`).
- **Launch.** Any chunk of the index, whenever its closure is final, on the card or slot it
  is given, side by side with other chunks:

  ```
  python parity/meep_gpu/rounds/1.0/round.py launch gates --repo <clone> \
      --chunk-index <index> --chunk c<k> --gpu-ids <card> --capability 8.6 \
      --home <lanes home> --lock-dir <locks> --expect <pushed tip> \
      --python <reference environment>/bin/python
  python parity/meep_gpu/rounds/1.0/metal_round.py all --repo <clone> \
      --chunk-index <metal index> --chunk m<k> --expect <pushed tip> \
      --meep-source <MEEP 1.33.0 source tree> --slot-wrapper <run_in_slot.sh> --slot-log <file>
  ```

  Each chunk's lanes live under `<lanes home>/<chunk stamp>`, and `LAUNCH.lock` there
  (`logs_metal_<chunk stamp>/LAUNCH.lock` on Metal) keeps a second launcher of the chunk
  out. `results/chunk_<chunk stamp>/LAUNCH.json` records the device, the tip, its tree and
  the units that run; a resume (`--resume`, or the Metal markers) runs only on that tip.
  The GO criteria are CERT-3's, per chunk.
- **Running units again.** `chunks.py add --index <index> --repo <clone> --units
  <unit>,... --why <text>` allocates a new chunk holding exactly those units (one unit
  alone is a chunk); `--from-verify <verify --json report>` takes the units `verify` did not
  hold; `relaunch --chunk c<k>` re-runs a chunk's units in a new chunk; `repartition
  --results results --chunks N` retires the chunks that never launched and packs their
  units afresh. Every earlier run stays as evidence, and none of these touches another
  unit's run.
- **Stale.** `chunks.py stale --index <index> --repo <new tip> --results results
  --prior-records <records>` reads each launched chunk's launch tip and names, per unit,
  the ones whose closure a later landing moved (`STALE_UNITS`); run those again with
  `chunks.py add --units ...` and launch the new chunk. It reads checkout files only: a
  rebuilt reference environment moves none of them and reads 0 stale, so it is never the
  bill of an environment change, which re-runs the round under a new round stamp.
- **Before the bind** pull every launched chunk's roots, `chunk_<stamp>` included.

### CERT-3 on a new device: an H200 or an Apple M3

A device the ledgers hold no live record for (an NVIDIA card of another compute capability,
an Apple GPU of another architecture) certifies with the same chunked round, ADDING its
records beside the certified device's and touching none of them. The family gates do not go
through dispatch admission, so they run on the new card as they are; the flag is what lets
the tooling run them there, and it is refused wherever it would be untrue.
`docs/development/certify-a-new-gpu.md` gives this round step by step for a contributor who
does not hold the evidence archive: the reference environment, the inputs bundle, the
index, the chunks (with card sharing under R28), verify, the bind on the contributor's own
host, the route and the package to send back.

```
python rounds/1.0/chunks.py partition --repo <clone> --machine nvidia --capability 9.0 \
    --new-device --out <cc90 index>
python parity/meep_gpu/rounds/1.0/round.py launch gates --repo <clone> \
    --chunk-index <cc90 index> --chunk c<k> --gpu-ids <card> --capability 9.0 \
    --new-capability --home <lanes home> --lock-dir <locks> --expect <frozen commit> \
    --python <reference environment>/bin/python
```

- `legs.py check` and the gate stage admit a capability no table admits only with
  `--new-capability`, refuse the flag on a capability both tables admit, and refuse an
  index whose device is not the card's or whose mode disagrees with the flag. A new
  capability runs chunked: an unchunked gate stage with the flag is refused, because only a
  chunk stamp carries the device. Every root of the cc90 index (`<round>_cc90_c<k>`) is its
  own and extends no root of the cc86 index by `_`.
- **The bind adds.** `chunks.py verify` refuses a run whose artifacts stamp another card;
  the rebinds (`rebind_cuda_welds.py`, `rebind_triton_welds.py`,
  `recut_composition_records.py`) read the capability off the run and add
  `runs["9.0"]` beside `runs["8.6"]` on unchanged bytes; `record_cuda_regate.py` adds a
  block for the new campaign. The bind of a new device re-points no board: the boards keep
  crediting the certified device's runs (`chunks.py bind` prints no re-point for it).
- **Then the route, unchanged.** Once the bind has added the records, `fastpath.
  capability_admission` admits 9.0 on both tables (step 3 of certification.md's protocol)
  and `round.py launch route --capability 9.0` runs without the flag (the route stage
  refuses it), into `dispatch_fused_route_<stamp>_cc90` and
  `dispatch_fused_route_cuda_<stamp>_cc90`, with the licence in
  `dispatch_end_to_end_<stamp>_cc90`; `recut_driver_dispatch_record.py --run ..._cc90` adds
  the record for 9.0.
- **An Apple M3**: `chunks.py partition --machine metal --architecture <its architecture>
  --new-device`, and `metal_round.py all --chunk-index <it> --chunk m<k>
  --new-architecture`, whose preflight and fleet stage each read the Mac's architecture,
  refuse an index of another architecture or mode and hold the flag to the Metal ledger
  both ways (the flag is refused without a chunk and on the route stage; an unchunked round
  runs only on an architecture the ledger admits); `rebind_metal_welds.py --stamp` reads
  the architecture from the chunk's environment record and adds its run; the route
  campaign is
  `dispatch_metal_route_<stamp>_<architecture>`.

### Bind (between CERT-3 and CERT-4)

A chunked round binds each unit from its own newest run that holds. From
`parity/meep_gpu`, first `python rounds/1.0/chunks.py verify --index <index> --repo ../..
--results results --selection <selection> --json <report>`: per unit of the release tree it
reads every launch whose record names the unit and takes the newest run whose import
record holds (no file the release tree holds at other bytes or not at all; with no import
record, no file of its closure moved since its launch tip), whose verdicts pass and whose
device is the index's (read, never assumed: on NVIDIA the compute capability its artifacts
stamp and a Triton campaign's `device.json`, or, for a unit with neither, the one the rest of
its launch's artifacts stamp; on Metal the launch's environment record; a run whose device cannot be read is
refused); it prints `VERIFY OK|REFUSED: N of D units hold` and
writes the choice, with every run and why it was or was not taken. A unit no run holds is
run again (`chunks.py add --from-verify <report>`), never bound. Then
`python rounds/1.0/legs.py bind --chunks <index> --selection <selection>` (NVIDIA) and
`python rounds/1.0/chunks.py bind --index <metal index> --selection <selection>` (Metal)
print the commands that bind exactly that choice: the Triton rebind once per selected
stamp with `--only` the keys selected there that the Triton ledger holds (the tool refuses
a whole `--only` naming a key the ledger lacks, so a selected unit whose `CAMPAIGN_DIRS` key
the ledger lacks is named on a `# REFUSED` line and left out, and a stamp left with no key
prints no Triton line); `rebind_metal_welds.py --stamp`, the CUDA
records and rebinds once per selected stamp in launch order; the composition, bit identity
and ADE re-cut from their selected stamps; the family re-certification over roots ordered
so each family's selected run is the first it reads. A run a tool would bind against the
choice (a later run of the same unit that the tool's own rules accept, or a moved or
unreleased subdirectory of a CUDA campaign that `record_cuda_regate.py` records whole, or a
unit's directory under a selected stamp whose launch record does not list it) is named on a
`# STOP` line with the command that sets it aside. While any STOP line stands the bind
prints no command that writes a ledger, ends with `# BIND WITHHELD` and exits 1: set each
run aside as its line says (it moves under `results/set_aside_<stamp>/`, nothing is
deleted), write the selection again with `verify --selection` and print the bind again.
The bind's first command, `chunks.py verify --against <selection>`, judges again, writes
nothing and exits 1 when the choice or the bytes of a selected run moved since the
selection was written: stop there and write the selection again. The three rebind tools apply the release-tree rule to every round artifact they
bind (`artifact_drift.py`; the CUDA rebind in its `--campaign` mode), and so do the record
writers (`record_cuda_regate.py`, `record_cuda_ade_recut.py`,
`recut_composition_records.py`, and `recut_driver_dispatch_record.py` at CERT-4). The NVIDIA
commands end, as the unchunked bind's do, with `neutralize_ledger_labels.py`; the board
re-point the bind prints as a comment takes the selected stamps, and is not printed for a
new device. An unchunked round binds as below.

Pull every result root as a real directory and compare every file's sha256.
Then, from `parity/meep_gpu` on the evidence host:

```
python rounds/1.0/legs.py bind --results results --stamp <stamp> --cuda-split 2
```

prints the ordered commands in report mode; the integrator reviews each report
and reruns with `--write`. The Triton rebind (`rebind_triton_welds.py --campaign
...`) also retires `triton_unified_expansion_device_gate` by name: its gate
(`gate_triton_unified_expansion.py`) was deleted with the expansion licence, so no
campaign can rebind it, and the tool lists it in `RETIRED_WELDS`. Its report
prints `RETIRED   triton_unified_expansion_device_gate` (never `UNCOVERED`), and
`--write` drops the entry from `meep_gpu/triton_kernels/fingerprints.json`, even
when nothing else rebinds. Never edit a ledger by hand. The tool refuses, editing
nothing, if that gate file is back in the checkout or an arm in
`fastpath.ARM_CERTIFICATION` cites the key. The CUDA rebind
(`rebind_cuda_welds.py --campaign ..._g0`) retires two pins by name the same way:
`cuda_complex_offdiag_stencil_welds_2026-09-17` and
`cuda_cylindrical_fused_hd_pair_2026-09-07` each pin a gate file their own gate
stopped importing (entries 89-X1 and 76-KC8), listed in `RETIRED_PINS`; its report
prints one `RETIRED-PIN` line for each, and the campaign rebinds all 24 welds it
re-gates. The tool skips such a weld, editing nothing, if a leg records the retired
file again. The Triton rebind also drops `driver.py`, by `RETIRED_UNIMPORTED_PINS`, from
the four welds whose gates no longer import it (`triton_cylindrical_fused_magnetic_pair`,
`triton_folded_offdiag_fused_ade_chain` and the two stencil welds): four
`RETIRED-PIN ... (not imported by <gate>)` lines; it skips such a weld if its run records
`driver.py`. A weld's own test file stays pinned and is still taken from the checkout. Copy the rebound
`meep_gpu/{triton,cuda}_kernels/fingerprints.json` and
`meep_gpu/cuda_kernels/certification.json` into the GPU host's clone, uncommitted
(the only files `check --route` allows to differ), then:

```
python parity/meep_gpu/rounds/1.0/legs.py flush-coverage --repo . --capability 8.6
```

- **GO to CERT-4:** the Triton rebind's report named
  `triton_unified_expansion_device_gate` RETIRED and the rebound Triton ledger no
  longer carries it; every arm whose gate released under flush on these bytes is
  listed as admitted, and each table admits at least one. Record the N of D per
  table. A device gate listed as lacking evidence has no released flush run on
  these bytes: rerun it under flush and rebind before the route. The five
  composition gates install and stamp the campaign's policy (R16), so their 11
  arms are admitted once `recut_composition_records.py --flush-run` has bound the
  flush composition root. A gate whose artifact states no attained policy cannot
  carry flush evidence; such a gate's arms are refused by name on the flush legs.
  Since `gate_triton_fused_electric.py` stamps its policy (section 5), every cited
  Triton and CUDA device gate does, so `FLUSH_COVERAGE COMPLETE` is expected once
  its keep and flush runs on these bytes are rebound.

### CERT-4, NVIDIA route and the licence

```
python parity/meep_gpu/rounds/1.0/round.py launch route --repo <clone> \
    --stamp <stamp> --capability 8.6 --home <lanes home> --lock-dir <locks> \
    --expect <frozen commit>
```

- The stage refuses to start unless `check --route` passes and each table admits
  at least one arm under flush.
- **The flush legs are mixed by design.** Dispatch admits flush per arm, so on
  `flush` and `cuda_flush` a case passes either PASS-FUSED (every arm it plans has
  flush evidence) or PASS-POLICY-REFUSED, where rung 8b's refusal names the missing
  per-arm evidence and the run is byte-identical to the array path. Any other
  non-dispatch fails the case, and the leg must dispatch at least one case.
- Each lane's preflight blocks: every case PASS-FUSED and released, or the
  lane stops before its long legs. The CUDA `cuda_alone` preflight is recorded
  and does not block.
- The licence runs only after the four Triton legs released: a plan-only
  preflight (the default enable must dispatch), the ship leg, the null control,
  then `VERDICT.txt` (GO or NO-GO).
- **The watchdog.** The stage gives every route and licence leg
  `max(3600 s, 12 x its measured wall)` (`--route-min-timeout`,
  `--route-timeout-factor`; entry 121), not `run_lane.py`'s default
  `max(1800 s, 4 x expected)`: `legs.py` prices every route and licence leg at
  the wall the pre-certification pass measured on a shared card (Triton
  `shipped` 3,181 s, `flush` 3,066 s, `cuda_flush` 2,792 s), so every watchdog
  is at least 12 x the wall it guards (Triton `shipped` 38,172 s; on the 0.9.2
  prices it was 5,400 s, 1.7 x), and a leg its watchdog kills has used its
  one-shot name. Raise them freely; lower them only with a measured wall on the
  card the round uses.
- **GO to the records:** every required leg released per table and `VERDICT GO`.
- **Resume:** route campaign names are one-shot. A released leg is never rerun;
  a leg that ran and did not release needs a new route stamp: that table's
  `REROUTED` entry in `meep_gpu/route_stamps.py`, a file no gate holds, so no
  other evidence moves (the spent name goes into `SPENT`). The licence is
  `route_stamps.licence_campaign` followed by the card's device
  (`dispatch_end_to_end_<stamp>_cc86`, `legs.licence_directory`: one per capability, so
  two cards' licences of one round never share a directory), and the re-cut accepts a
  licence from no other directory.

### CERT-4, Metal route

`metal_round.py route` runs `run_metal_dispatch_campaign.sh` in one slot call,
into the campaign `recut_driver_dispatch_record.py --backend metal` requires. A
campaign directory that exists without the stage's marker is refused (one-shot).

- **The route stage reads no fleet of its own stamp.** It names the campaign from
  `route_stamps.route_gate("metal")` and this Mac's architecture, refuses a name the
  registry refuses (unset or spent), and lets
  only `meep_gpu/metal_kernels/fingerprints.json` differ from the commit. Its
  marker lives in `logs_metal_<--stamp>/done/route_campaign.rc`, and an item with a
  marker is skipped, so a route under a moved constant runs with a `--stamp` whose
  logs hold no route marker: under the round's own stamp it reads "already done"
  and launches nothing.
- **At the 1.0 re-freeze** (`fastpath.py` moved, `PRE_CERTIFICATION_STATE.md`
  section 8) the Mac re-runs the whole fleet and the route, on a clean checkout of
  the re-frozen commit, with `--stamp 2026-10-12_10`: `metal_round.py all`, then
  `rebind_metal_welds.py --stamp 2026-10-12_10` on the evidence host, then `route`
  into `dispatch_metal_route_2026-10-12_10_applegpu_g13s`, and
  `recut_driver_dispatch_record.py --backend metal --run
  dispatch_metal_route_2026-10-12_10_applegpu_g13s`. By the ledger's pins only the
  three cited welds that pin `fastpath.py` are owed
  (`gate_metal_complex_conductive_fused_pair`, `gate_metal_complex_no_pml_conductive`,
  `gate_metal_cylindrical_real_fused_magnetic_pair`; `fleet --only` those three
  serves the ledger and the route record). But every Metal gate artifact of the round
  records `meep_gpu/fastpath.py` in its `source_sha256`, and the Metal board re-hashes
  the artifacts it credits (`build_fusion_matrix.py`'s release binding), so the board
  can be cut on the re-frozen bytes only from a fleet run on them, with its release
  bindings re-pointed to that stamp.
- **At the freeze after per-kernel certification** (entries 133 to 135,
  `PRE_CERTIFICATION_STATE.md` section 11) every gate's import record changes shape, so
  the Mac re-runs the whole fleet and the route once more, on a clean checkout of the new
  frozen commit, with `--stamp 2026-10-13_10`: `metal_round.py all`, then
  `rebind_metal_welds.py --stamp 2026-10-13_10` on the evidence host, then `route` into
  `dispatch_metal_route_2026-10-13_10_applegpu_g13s`, and
  `recut_driver_dispatch_record.py --backend metal --run
  dispatch_metal_route_2026-10-13_10_applegpu_g13s`. The `2026-10-12_10` runs above are
  the record of the previous pass. That rebind retires, by `RETIRED_UNIMPORTED_PINS`, the
  curated pins three welds' gates no longer import (`metal_complex_no_pml_conductive` 13,
  `metal_cylindrical_real_fused_magnetic_pair` 12, `metal_complex_conductive_fused_pair` 6;
  each also drops `expansion_refusal.py` by `RETIRED_PINS`): expect 34 `RETIRED-PIN` lines
  and all three rebound. It skips such a weld by name, editing nothing, if the run records
  a retired file or disagrees with the checkout, and refuses the whole run if a row's file
  is back in its gate's run closure.

- **The watchdog.** The campaign is one item with one watchdog, and nothing below it
  times a leg (`metal_gate_runner.py` and the gate carry none). It is
  `max(21,600 s, 3 x 17,920 s)` = 53,760 s (`--route-timeout-factor`,
  `--route-min-timeout`), plus `--slot-patience` for the slots: three times the serial
  sum of the four legs' walls measured in the 1.0 pre-certification pass
  (`metal_round.ROUTE_EXPECT`: band_witness 883 s, harness_keep 6,479 s, shipped
  5,283 s, harness_flush 5,275 s, on a loaded Mac), which is 4.25 x that pass's
  3-lane campaign (12,648 s). It was a fixed 6 h, 1.7 x that campaign. A campaign its
  watchdog kills has used its one-shot name. Raise either flag freely; lower one only
  with a measured wall on the Mac the round uses.
- Every child runs with `PYTHONNOUSERSITE=1`, as the reference does, and the
  preflight refuses a child interpreter that still enables the user site directory.

### CERT-4 records

The integrator runs `recut_driver_dispatch_record.py` per table, the three
censuses, the boards and the `board_controls` re-baseline (R8) on the evidence
host. The licence's recorded environment is written under `<staging-root>`: the
recut takes the directory its recorded paths share, or `--staging-root DIR` when
they share none under a home directory, and refuses a path still local after
that. The guard's ledgers leg reads CLEAN on the re-cut records.

- **The Triton and Metal boards' bindings are re-pointed by tool, first.** Both boards
  name the gate artifacts they credit (`build_triton_fusion_matrix.GATE_BOUND`,
  `build_fusion_matrix.RELEASE_BINDING` and its per-cell rows) and re-hash each against
  the tree, so after a round re-runs its gates the previous round's names refuse. From
  `parity/meep_gpu`, with the round's roots under `results/` and the rebound ledgers in
  place:

  ```
  python repoint_board_bindings.py triton --stamp <stamp> --device cc86 [--write]
  python repoint_board_bindings.py metal  --stamp <stamp> --device applegpu_g13s [--write]
  ```

  Report mode first; review, then `--write` (a second `--write` changes nothing). The
  Triton re-point refuses while any entry has no admissible run of the round: the ten
  H->D gates run in CERT-3 (`legs.py` `TRITON_HD`), so every entry should resolve.
  `--withhold-unrun` withholds such entries by name instead, an owner's call, never the
  default. Never edit either table by hand. The CUDA board binds through the ledgers
  and needs no re-point. A board credits one device's runs: a candidate that ran on another
  device than the run its binding names is refused by name (`--device` names the device a
  board is moved to on purpose), so a new device's round never re-points a board. A device
  that cannot be read is refused too, never passed: without `--device`, a bound run that
  records none refuses every candidate, and a candidate that records none is refused.
- **The boards read the H->D probe re-cut on the frozen bytes.** All three boards
  price the update_H -> step_D seam from `h_to_d_seam.PROBE_DIR`, whose default,
  `h_to_d_seam_2026-09-04`, was cut on the driver before entry 25. There the
  integrated electric sources of 7 rows stood a withdraw between the two halves;
  since entry 25 no source carries one and the driver runs none
  (`h_to_d_seam.driver_seam_fact` asserts the seam holds only the magnetic
  `f_minus_p` restore). On a census cut on these bytes the H->D predicates admit
  those rows, so against the default probe `build_cuda_fusion_matrix.py` refuses
  at `h_to_d_seam.price` (7 rows served over a withdraw, no
  `served_over_the_withdraw` licence), and the Triton and Metal boards, which
  derive their H->D verdict from the probe's flag, file the same 7 rows
  `withdraw_seam`. Every freeze therefore re-cuts the probe on the frozen tree, by
  the unchanged `probe_h_to_d_seam.py`, under a name of its own
  (`h_to_d_seam_<date>_<tree prefix>`, so a cut on one tree is never read as
  another's; the cuts on the 1.0 trees measure 0 withdraw rows of the 192 they
  measure), stages it under `results/` with the censuses, and runs every board with
  `MEEP_GPU_H_TO_D_PROBE` naming that cut:

  ```
  python parity/meep_gpu/probe_h_to_d_seam.py --out parity/meep_gpu/results/<name>
  MEEP_GPU_H_TO_D_PROBE=<name> MEEP_GPU_CUDA_CENSUS=<census> \
      python parity/meep_gpu/build_cuda_fusion_matrix.py --out <dir>
  ```

  Set it on the board commands only, never in the shell that runs the suites:
  `test_h_to_d_seam.py` checks `PROBE_DIR` against the standing 2026-09-03
  censuses, two of whose rows the frozen engine refuses by name
  (`MeepSimulationNotLiftable`), and fails on the re-cut wherever the evidence
  archive is present. `metal_round.py` strips the variable from every child it
  starts, and its preflight refuses a shell that exports it. Never overwrite
  `h_to_d_seam_2026-09-04`: the H->D gates read it as `SEAM_RECORD` for their lift
  basis.

### CERT-5: L3, the direct corpus check

L3 runs the corpus with kernels dispatched and judges every stored word and every
`get_array` word against the stored digests of the reference MEEP's words: the Tier
B record that Table 2 cites (`meep_identity_tier_b.json`) and its row files. MEEP
is imported for the lift and never stepped, so the GPU host runs no MEEP reference;
it needs the reference MEEP installed, and every row checks that the MEEP images it
loaded are the record's for its set. One `l3_corpus.py` invocation is one leg (one
table variant, one baseline); every row runs in its own process (`l3_row.py`), and
the row's verdict uses the tree's own Tier C and MEEP-identity functions, unmodified.

Stage the record with its row files (`digests.json`, `planes.npy`, `result.json` per
set and row; the tool verifies each against the record's sha256 before the row runs
and refuses a row that does not verify), the corpus manifest and the MEEP 1.33.0
source tree. Run from the clean clone of the frozen commit with the round's bound
ledgers copied in, uncommitted (as for CERT-4), in the reference environment,
activated. The tool takes no GPU lock: on the GPU host, take `gpu_lock_<card>` under
`--lock-dir` the way `run_lane.py` does (count below `--max-locks`, two idle readings
of the card) and release it when the leg's last row returns.

NVIDIA, four legs (R13), each on one card:

```
python -u parity/meep_gpu/rounds/1.0/l3_corpus.py --table triton --baseline keep \
    --record <Tier B record>/meep_identity_tier_b.json --expect-record-sha256 <its sha256> \
    --out <runs>/triton_keep --gpu <card> --parallel 3 --corpus-manifest <corpus manifest>
```

then `--baseline flush`, and both again with `--table cuda`
(`MEEP_GPU_BACKEND_PREFERENCE=cuda`, the hand-written CUDA table composing first).
In the R27 reference environment the tool takes Triton's libcuda link and the MEEP
corpus from the environment's build record (`share/meep-gpu-build/`), so the R27 legs
passed neither `--libcuda-stub` nor `--corpus-root`. The tool builds every row's environment: each `MEEP_GPU_*` and `TRIDENT_FDTD_*`
variable removed (dispatch on), threads 1, `PYTHONNOUSERSITE=1`, the checkout alone on
`PYTHONPATH`, `CUPY_ACCELERATORS` empty, one CuPy and one Triton cache per leg, and
the keep leg's CuPy cache minted by the package under a path carrying
`ftz_stripped` before the first row.

Metal, one leg (R1, flush only), inside one `mac_gpu+mac_cpu` slot call on a shared
Mac:

```
python -u parity/meep_gpu/rounds/1.0/l3_corpus.py --table metal --baseline flush \
    --record <Tier B record>/meep_identity_tier_b.json --expect-record-sha256 <its sha256> \
    --out <runs>/metal_flush --clean-env --python <reference environment>/bin/python \
    --corpus-manifest <corpus manifest> \
    --expect-differ references --expect-differ-holds mac_flush
```

From one dispatched run it judges `linux_flush` (the claim, and the leg's primary
set), `mac_flush` and `linux_keep`; the lift loads the `mac_flush` MEEP (the
contract-off build). `--clean-env` starts each row from `HOME`, `USER`, `LANG=C` and
the reference `PATH`, `<prefix>/bin:/usr/bin:/bin`, as the reference rows ran; in the
R27 reference environment the lift imports that environment's own MEEP, so no
`--meep-site` is given, and the corpus is the build record's.
Where the two hosts' reference MEEPs disagree (L0: `mac_flush` against `linux_flush`, 32
of the 192 rows `linux_flush` holds in the R27 record, `ex.ring_gds` among them), no run
can equal both;
`--expect-differ references` derives those rows from the record before any row runs,
writes them to `expected_differ.txt`, and counts each one as holding when it differs
against `linux_flush` and holds against `mac_flush`. Every other row must hold against
`linux_flush`. `linux_keep` is read, never required (R1 option (c)).

Follow `<runs>/<leg>/progress.log`: one line per row (i of n, each set's verdict and
digests equal of D, step path, table, dispatched slots, dispatches, wall) and a
heartbeat every 120 s naming the rows in flight; `rows.jsonl` gains one line per row as
it lands, and `summary.json` is rewritten after every row.

- **GO:** every leg's last line reads `VERDICT HOLDS` (exit status 0): every row the
  primary set holds was run and held against it, with every stored and reader digest
  equal at every compared step, the planted one-ulp flip caught and placed in its
  plane, and the FPU in the baseline's mode at every compared step; no row errored or
  was refused for its files. A row that holds on the array path (`holds_array`, its
  reason under `array_fallback_reasons`) holds: under the Triton default precedence a
  plan that composes an arm still pending recertification is refused whole, and its row
  steps the array path, bit-identical to MEEP. Rows the record refuses or excludes are
  named in `summary.json` (`not_run_record`) and never run.
- **NO-GO:** a row that differs (`VERDICT DIFFERS`). Its first differing step,
  components and x-planes are in `rows.jsonl`: a finding against the frozen bytes, read,
  not rerun. A row that errored for its environment is rerun with `--retry-errors`.
- **Resume:** rerun the same command. A row already in `rows.jsonl` is skipped and never
  appended twice; `<runs>/<leg>/STOP` stops the leg before its next row; a signal kills
  the rows in flight, which are not recorded, so the rerun runs them.
- **Record** per leg, from `summary.json`: rows holding of the rows the primary set
  holds, the stored and reader digests equal of those compared, the flips caught and
  placed, and the dispatched rows by table. Which tables' kernels served a row is in its
  `launch_sums` (`launched_slot_backends`), not in the plan's `table` field.

### CERT-6 to CERT-8

The H200 and an Apple M3 certify as a new device (CERT-3 on a new device, above): the
chunked gate stage with the device's own index, the bind that adds its records, then its
route. The suites and the timing ladders have their own drivers; CERT-8 runs last, on
idle boxes, with the MEEP rows re-timed in the same session as the GPU rows (R5). CERT-8's
driver is `cert8.py` beside this file, one row at a time (`docs/development/timing.md`,
"The release round, one row at a time").

## 5. Known blockers to clear before the round

Measured by the 1.0 pre-certification passes, then re-measured at the series'
final integration (entries 94 to 105, `integrate/batch_r13`; its logs name the
tree each leg ran on). An item still open on the frozen commit stops a GO above.

Closed in the series, each re-measured at the final integration:

- **The route gate's armed null control** (absorbed consult withheld) did not
  diverge on cells whose stored E is a pure overwrite and blocked every NVIDIA
  route leg: entry 97 places the control's window where the mechanism exists
  (the component's own-axis layer and the step at which a second application
  first changes a word) and rules NO-OWN-AXIS-LAYER apart. The five folded route
  cases' absent-component words are cleared since entries 90 and 101.
  Re-measured on one RTX A6000: `t_preflight` (6 of 6 PASS-FUSED),
  `cuda_shipped`, `cuda_harness_keep`, `cuda_default_precedence`, `cuda_alone`,
  `shipped` and `harness_keep` released. The two flush legs pass every case
  (`cuda_flush` 39 of 39 after entry 105, `flush` 37 of 37), PASS-POLICY-REFUSED
  until the bind writes per-arm flush evidence, and so cannot release before it:
  their envelope control and the dispatched-case clause need a dispatch (CERT-4
  binds, then the route runs). End to end on the shipped route: `e2e_ship` 9 of
  9 PASS-DISPATCHED, `e2e_null` 9 of 9 PASS-FELL-BACK.
- **`fused_hd_pair` (CUDA, Triton and Metal H->D gates)** stated the removed
  withdraw in their refusal, withdraw and driver-order legs: entries 98 (CUDA
  and Triton) and 87 (Metal) measure the seam the driver runs since entry 25.
  Re-measured on one RTX A6000: the CUDA H->D gates `complex`, `cylindrical` and
  `cylindrical_real` released under keep and flush, `fused` under keep (15 of 15
  legs, the lift on 129 rows), `special_kz`, `complex_beta` and
  `conductive_bfast` under keep (the reduced product); the ten Triton H->D gates
  released under keep with every leg and the lifts on. Metal:
  `gate_metal_folded_fused_hd_pair` released, 20 of 20 legs (Apple M1 Max,
  flush).
- **`route/cuda_alone`, case `cylindrical`**: entry 97 declares the scan beside
  each Dcyl curl as a companion kernel. Re-measured: `cuda_alone` released,
  `cylindrical` among its cases.
- **`conductivity` (Triton)** timed out at 3,600 s: entry 99 gives the fleet
  entry 10,800 s. Re-measured on one RTX A6000 under keep, through the fleet
  driver: `conductivity` released in 5,148 s (24 of 24 mutations caught, 32 of
  32 long-run rows bit-identical), `symmetry` in 4,937 s, on a card other gates
  shared.
- **`dispersive_offdiag_polarization_pair` (CUDA)**, a mutation partial on the
  full product: entry 99 declares the structural null by the array path's region
  plan. **`cuda_complex_regate`'s phased-case floor**: entry 99 grades it by the
  array path's own connection. Re-measured on one RTX A6000: both gates released
  under keep and flush.
- **Metal**: the folded-complex expansion probe is deleted with the licence
  (entry 89); the no-PML conductive kernels step MEEP's stencil (entry 101); the
  symmetry signed-zero floor, the whole-step region plan and the tranche-7
  needle (entry 87); the folded H->D lift rows and the folded_beta floor (entry
  101). Re-measured on the Apple M1 Max: `gate_metal_folded_beta`,
  `gate_metal_whole_step`, `gate_metal_no_pml_conductive`,
  `gate_metal_no_pml_fused_electric_pairs` and `gate_metal_folded_fused_hd_pair`
  released, and the shipped route leg released (9 of 9 cases, 57 of 57
  controls).
- **The Triton folded H->D gate's launch accounting** predated the two-pass
  mirror fill (entry 105). Re-measured on one RTX A6000: the gate released under
  keep with every leg, `launch_structure` among them.

- **The route constants** carried 0.9.2 stamps and `legs.py check` refused on
  them: the freeze (entry 121, CERT-1) moves all three to the round's stamp
  `2026-10-11_10` (`PRE_CERTIFICATION_STATE.md` section 11), and `legs.py check`
  on the frozen bytes reads `CHECK OK`.
- **Rung 8b read a bare label in a CUDA-led plan against the CUDA ledger.**
  `fastpath._table_for_label` answered the composing table for a label without
  the `cuda:` prefix, so under `MEEP_GPU_BACKEND_PREFERENCE=cuda` with `flush`
  installed every Triton arm of the merged plan (`nonlinear`, `dispersive`,
  `ADE update_P`, `conductive no-PML`, `fused pair B`, ...) read "'unmapped' has no
  live run on compute capability 8.6" and the whole plan was refused, although the
  same arms are admitted against the Triton ledger (`legs.py flush-coverage` asks
  each arm against its own table and did not see this). Five bare labels the CUDA
  composer also spells (`PML`, `ordinary`, `off-diagonal`, `folded off-diagonal`,
  `mirror fill`) were answered from their `cuda:` twin's run instead. On
  `cuda_flush` this failed the case `dispersive_2d` (its veto leg plans Triton's
  `ADE update_P` and `dispersive`: `NO-SUBSTITUTION`) and the envelope control
  (`nonlinear_3d`: `ENVELOPE-DID-NOT-HOLD`) in a scratch dry run on the frozen
  commit `2b75bef8`. The re-freeze answers a bare label in a plan composed on the
  CUDA table from the Triton ledger (every CUDA label crosses into `fastpath`
  namespaced) and leaves every other table's bare labels on that table. Measured
  host-side with the scratch-rebound ledgers: the six Triton singles of those
  cases are admitted under flush in a CUDA-led plan (6 of 6; 0 of 6 before), all
  75 bare Triton labels answer under plan table `cuda` exactly as under `triton`
  (75 of 75; 5 of 75 before), and `fused pair B` / `fused pair D` are admitted
  once fix B's flush evidence is bound. Under keep nothing moved: rung 8b asks no
  arm anything, and the record's per-arm certification and every other record
  field read the same (`meep_gpu/test_dispatch_label_table.py`).
- **The Metal route constant moves again at the re-freeze.** The Metal route
  record pins `meep_gpu/fastpath.py`, and `dispatch_metal_route_2026-10-11_10`
  has run (it released every leg on the first frozen bytes), so
  `metal_dispatch.METAL_DRIVER_ROUTE_GATE` names `_2026-10-12_10` and the held
  route re-runs as `dispatch_metal_route_2026-10-12_10_applegpu_g13s`. The NVIDIA
  route constants keep `_2026-10-11_10`, which no NVIDIA route has used. CERT-4,
  Metal route (section 4) gives the Mac's re-run.
- **The route constants move to the per-kernel round's stamp.** Entry 128 changes
  the shape of every gate's import record, so the round re-runs in full on the
  frozen commit after entry 135, and its route campaigns need unused names: entry
  133 moves all three constants to `2026-10-13_10`
  (`dispatch_fused_route_2026-10-13_10_cc86`,
  `dispatch_fused_route_cuda_2026-10-13_10_cc86`,
  `dispatch_metal_route_2026-10-13_10_applegpu_g13s`); `2026-10-11_10` and
  `2026-10-12_10` are spent.
- **Curated pins the round's records cannot hold.** Under per-kernel certification a
  kernel gate's record is its own dependency set, and 3 Metal welds and 4 Triton welds
  pinned core modules (the driver, the planner, the lift) their gates no longer import,
  pins cut from records the eager package root had filled. Each rebind tool now carries
  `RETIRED_UNIMPORTED_PINS`, generated by `import_closure.py unimported-pins <table>` (a
  module of the old eager root outside the gate's run closure) and reviewed: 31 Metal rows,
  4 Triton (`driver.py`), 0 CUDA. Of the 13 Triton entries counted before, the other 9
  need no row: `bit_identity_gate`'s binder writes its pin set from the run's own record,
  and 8 welds pin their own test file on purpose (kept, still taken from the checkout). The
  run closure no longer reaches a harness table's
  stored callables unless reached code names the table, which put `dispersion.py` in the
  bound of `metal_cylindrical_real_fused_magnetic_pair`'s gate. On post-R17 runs of the
  seven gates the Metal rebind binds 3 of 3 and the Triton rebind 4 of 4, dropping exactly
  the rows. Never edit a ledger by hand.

Still open:

- **Flush evidence for the routing arms (owner ruling R16, option (a)).** 11
  Triton arms cite the five composition gates (`cylindrical_composition_gate`,
  `symmetry_composition_gate`, `conductivity_composition_gate`,
  `dispersive_composition_gate`, `no_pml_composition_gate`). Each gate installs
  the campaign's policy from `MEEP_GPU_SUBNORMAL_POLICY` before it builds
  anything, refuses by name when it is not attained, and stamps it as
  `subnormal_policy` (`parity/meep_gpu/composition_policy.py`); the arms are
  admitted under flush once `recut_composition_records.py --flush-run` binds the
  flush composition root at the bind. Under flush the two probes that never
  import MEEP (cylindrical, symmetry) need the flush fleet's MEEP-importing
  wrapper, which `run_lane.py` always uses.
- **`fused pair B` and `fused pair D` carry flush evidence once the round
  rebinds them.** Their gate, `gate_triton_fused_electric.py`, installs the
  campaign's policy before it builds anything, refuses by name when it is not
  attained, stamps it as `subnormal_policy` and confirms it at the end of the run
  (`composition_policy.py`). The same reviewed change retired the
  `_mixed_policy` declaration of `triton_fused_electric_device_gate` and its
  `POLICY_UNNAMED_BUDGET` line. Without the gate's policy, the default-precedence
  read of `cuda_flush`'s arbitration control (Triton composes first) planned these
  two arms and was refused at rung 8b, which no control verdict passes; a scratch
  dry run of `cuda_flush` on the frozen commit `2b75bef8` with the rebound
  ledgers (the keep preflight's 6 cases) read `REFUSED` on `pml_2d`,
  `dispersive_2d` and `pml_3d`. Measured in scratch on one RTX A6000 on these
  bytes: the gate released under keep and under flush (56 of 56 base cases
  bit-identical, 10 of 10 mutations caught, the attained policy stamped on all
  three executors), and `rebind_triton_welds.py --flush-campaign` then wrote
  `certified_policies` `{flush, keep}` for the entry, so `legs.py flush-coverage`
  read Triton 75 of 75 and CUDA 34 of 34 arms. Owed on the round's bytes: the
  `fused_electric` fleet gate under keep and flush (`run_lane.py`, the flush run
  through its MEEP-importing wrapper), then the Triton rebind of
  `triton_fused_electric_device_gate` from those two runs.
- **Two device welds still declare `_mixed_policy`** although their gates
  install a policy: `triton_no_pml_device_gate` (`gate_triton_no_pml.py`, R16)
  and `triton_cylindrical_device_gate` (`gate_triton_cylindrical.py`, since
  entry 68). `rebind_triton_welds.py` refuses a stamped run for an entry that
  declares it, by design, and no tool retires a declaration: retiring it (and
  its `POLICY_UNNAMED_BUDGET` line) is a reviewed change to the tree, as it was
  for `triton_fused_electric_device_gate`. Neither weld is cited by an arm.
- **The Triton flush fleet is unproven.** Of the 62 fleet gates, 39 take the
  policy on a flag and 23 resolve it from the environment; before the round, at
  least one of each must have released under flush on the frozen bytes.
- **The Triton H->D welds' full gates under flush** (every leg, lifts on) have
  not run on these bytes; under keep they ran at the final integration. They are
  not dispatched (INSTALLABLE is False on all ten), so no route leg reads them.
- **The census of record carries the retired `@unified_expansion` keys**, and
  `build_triton_fusion_matrix.py` refuses it by name: CERT-4 cuts the Triton
  census on the frozen bytes before the board.
