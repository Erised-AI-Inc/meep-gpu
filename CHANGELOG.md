# Changelog

## 1.0.0 — MEEP's own words on one GPU, certified per kernel (2026-10-16)

The first certified release, and the version the accompanying paper cites. Its status is
beta (`Development Status :: 4 - Beta`): interfaces and record formats may still change in
later releases. The array path returns the words MEEP 1.33.0 stores and reads, and every
compiled kernel returns the array path's words; the 1.0 certification round measured both,
in the two reference environments below, on the bytes of this release. The entries after
these notes, down to 0.9.2, are the changes that make up 1.0, each with its own measurements.

### What 1.0 certifies

- **Two GPUs, and no other.** Compute capability 8.6 (one RTX A6000; Triton 3.1.0 on the
  Triton table, CuPy 13.5.1) for both NVIDIA tables, under `keep` and under `flush`; and the
  Apple GPU architecture `applegpu_g13s` (one Apple M1 Max; PyTorch 2.10.0, Metal frontend
  `metalfe-32023.850.10`, `PYTORCH_MPS_FAST_MATH` unset or `0`) for the Metal table, under
  `flush`. On a certified GPU, under the same subnormal policy, every dispatched kernel is
  bit-identical to the array path, per family by the device gates and over whole runs by
  the route campaigns.
- **The array path is MEEP's words.** Against MEEP 1.33.0 built in single precision
  without contraction, every stored float32 word of the field components and every word
  `get_array` returns is MEEP's at every compared step, on every corpus simulation Table 2
  (`docs/development/meep-identity-corpus.md`) holds; L3 judged the dispatched runs against
  the same stored words. Agreement with a double-precision or contracting MEEP is a
  tolerance, as before.
- **Supported, not certified:** every NVIDIA GPU of compute capability 7.0 to 9.0 and every
  Apple GPU. The kernels run there by default, the run is recorded `certified: False`, and
  one `meep_gpu: NOTE` line per process says so; `MEEP_GPU_HARDWARE=certified` keeps
  dispatch to the certified identities.
- **Another GPU is certified by writing records, not code.** A GPU outside the certified
  set (an H200 at compute capability 9.0, say, or an Apple M3) is certified by running the
  same round on these bytes, chunked: its bind adds its runs beside the certified ones
  (`docs/development/certify-a-new-gpu.md`), and a 1.0.x release publishes them. No such GPU
  has run it.

### The reference environments

Every certification of 1.0 is measured in one reference environment per platform,
`linux-64` and `osx-arm64` (owner ruling R27), built end to end by the repository's own
script from the locks (`parity/meep_gpu/build_meep_133_linux.sh`,
`build_meep_133_macos.sh`; `environments/locks/`: 147 conda and 26 PyPI packages on Linux,
141 and 17 on Apple silicon, each by hash):

- **MEEP 1.33.0 from the release tarball**, pinned by SHA-256, single precision, MPI (and
  OpenMP on Linux), with libGDSII and no floating-point contraction (`-O2 -ffp-contract=off
  -fno-fast-math`, and `-march=x86-64 -mtune=generic` on Linux), judged on its machine code
  by the fused multiply-add census and on MEEP's own words on four cases;
- **every package, library and tool** the tests, the round and the timing use, the
  corpus-only ones included, and the MEEP corpus itself: the same tarball's `python/` tree,
  with the files outside it that corpus rows read, kept in the environment and named file by
  file in `share/meep-gpu-build/reference_build.json`, which every round driver reads;
- **from the host, three things only**: the GPU driver, the macOS SDK and glibc.

Both scripts ran end to end on 2026-10-14 (Linux 377 s, macOS 296 s); the two
environments were then brought to the corpus siblings by `--refresh-corpus`, and
`tools/check_install.py --require-reference` judges a host's environment against them,
libGDSII included. Because both environments are new, the round ran again in them, CERT-1
to CERT-5, under the round stamp `2026-10-15_10`; CERT-6 is a new device's own round, and
CERT-7, the suites per environment on the release tip, is owed to a documentation update.

### Certification in modular chunks

A gate's record binds the files that gate imports and no others, so certification is per
kernel, and the round runs as chunks (`parity/meep_gpu/rounds/1.0/chunks.py`): any set of
the round's units, under a stamp of its own (`2026-10-15_10_cc86_c<k>`,
`2026-10-15_10_applegpu_g13s_m<k>`), on whatever card or slot is free, side by side. A unit
(a CUDA campaign subdirectory, one Triton or Metal gate, both policies together) binds from
its own newest run whose whole import record holds on the release tree and whose device is
the index's, so a unit can run again alone and the plan can be re-partitioned without
voiding another unit's run. The round's NVIDIA units are 123 and its Metal units 62; a card
holding another user's idle process may be shared, by opt-in, and is yielded the moment its
owner needs it (owner ruling R28). A new GPU runs the same chunks with `--new-device`, and
its bind adds records and re-points no board.

### Timing: the 1.0 timing round (CERT-8), on these bytes

The timing round (CERT-8) is modular in the same way: its unit is a row
(`parity/meep_gpu/rounds/1.0/cert8.py`, `parity/meep_gpu/timing_rows.py`), any row can be
timed again at any time, and every table and record is cut by tool from the newest valid row
of each key, in one reference environment per cut. It is not part of certification: it ran
after the certification round, on the certified bytes of this release, and the documentation
update that states it moves no package file (owner rulings R25 and R32). It ran on
2026-10-16 on one NVIDIA host and one Apple laptop, with MEEP re-timed in the same session as
the GPU on each (owner ruling R5), against each platform's reference MEEP 1.33.0 (single
precision, no floating-point contraction, a portable binary). Per step, on one 3-D case (a
dielectric sphere under PML, one flux monitor, the same simulation on both sides):

- **NVIDIA, one RTX A6000 (compute capability 8.6)**, against MEEP at its fastest rank
  count on the host's 48 cores: the Triton route 1.65×, 3.98× and 5.04× at 512,000,
  2,097,152 and 7,077,888 cells, the CUDA-only route 1.45×, 4.59× and 5.81×; all six of
  record.
- **Apple, one M1 Max (`applegpu_g13s`)**, against MEEP at 8 ranks: 0.82×, 1.21×, 1.69×,
  2.40× and 2.72× from 110,592 to 7,077,888 cells, all five of record under the gate of
  owner ruling R33, which was ruled after the runs had been seen; under the gate in force
  when they ran, only the 7,077,888-cell row entered.
- **End to end**, derived from the lift and the step rates and leaving out the GPU's own
  cost of its first steps: the lift, which reads MEEP's stored permittivity in bulk, took
  7.4, 19.5 and 48.5 s on the NVIDIA Triton route, and the GPU run is ahead of MEEP after
  about 5,900 steps at 512,000 cells and about 460 at 7,077,888 (at 2,097,152 cells the lift
  is shorter than MEEP's own start-up); on the Apple laptop the lift took 1.6 to 27.9 s
  and the break-even falls from about 21,800 steps at 262,144 cells to about 2,700 at
  7,077,888.
- **The gate (owner ruling R33)**: a GPU median enters when its pass medians spread by at
  most 5 %. Each pass's window spread is recorded, and no table gates on it; as a gate it had
  struck 6 of 8 Apple medians whose pass medians agreed within 0.16 to 2.44 %. MEEP's
  spread is recorded and never gated.
- **Fused products against the single arms they replace**, per step on the NVIDIA tables:
  median 0.62× over 46 cases at or below 110,592 cells, and a plan holding two fused pairs
  crosses its single arms between 2.4 and 5.4 million cells.
- **The dispatch-preference records stay at 2026-09-21** in 1.0.0. The records the 1.0
  timing cut land in a later 1.0.x release, as package data the dispatch-preference consult
  reads.
- **What these figures license**: one machine per platform, this one case, single precision
  on both sides, MEEP at the rank counts timed. No figure per watt, and none for another
  GPU: no other GPU has been timed.

`docs/guides/will-it-help.md` has every figure with its conditions and licence,
`docs/guides/kernel-dispatch.md` the fused products, and `docs/development/timing.md` the
gate and the comparison with the round's first timing, of 2026-10-13.

### Results of record: the 1.0 certification round

The gates ran in chunks under the round stamp `2026-10-15_10`, each unit on a tree whose
run closure for it equals the frozen commit's; the binds, the route campaigns and L3 ran on
the frozen commit `2644fc1dd294`; the corpus rows of Table 2 were measured on the series'
entries 01 to 157, whose corpus-runner closure (268 paths) equals the frozen commit's, and
their Tier B record was minted on the frozen commit; the records stage (censuses, boards,
the ledgers' remaining leaves) ran on the development tip after it, whose three later
entries are each in the run closure of 0 of the 250 gate and probe scripts. Per-kernel
certification (owner ruling R17) is what lets one release stand on all of them: an artifact
stands wherever every file its import record names is byte-equal in the release tree.

- **Device gates (CERT-3):** NVIDIA 123 of 123 units hold, from 10 chunks launched (of the
  13 the index allocated, 3 superseded by a re-partition before they launched); Metal 62 of
  62 units hold, from 4 chunks (`m3` is one unit run again alone after the corpus fix).
- **Bind:** `flush` evidence admitted for 75 of 75 Triton arms and 34 of 34 hand-written
  CUDA arms on compute capability 8.6; the Metal ledger rebound 62 of 62 welds. The hygiene
  guard on the four bound ledgers: CLEAN (0 hits).
- **Route campaigns (CERT-4):** Triton 3 of 3 legs released (`shipped`, `harness_keep`,
  `flush`; 37 of 37 cases dispatched on each), hand-written CUDA 5 of 5 (`cuda_shipped`,
  `cuda_harness_keep`, `cuda_flush`, `cuda_default_precedence` and the optional
  `cuda_alone`; 37 of 39 cases dispatched on each of the first four, the other 2 having no
  fused arm), Metal 4 of 4 (`band_witness`, `harness_keep`, `shipped`, `harness_flush`;
  42 of 44 cases dispatched on each dispatch leg, the other 2 engine smoke cases), residency
  held, and 0 failures and 0 control failures on any leg. The go/no-go behind dispatch on by default:
  GO (ship 9 of 9 PASS-DISPATCHED, null 9 of 9 PASS-FELL-BACK).
- **Coverage boards**, seam-instances reachable in dispatch (an upper bound) of 597:
  Triton 334, hand-written CUDA 352, Metal 327; seam-instances a fused product exists for:
  537, 538 and 507 of 597. Both figures equal, instance by instance, those of the boards
  cut for the 2026-10-13 records entry.
- **MEEP identity:** Table 2 holds 193, 193, 192 and 192 of 194 corpus simulations on the
  Mac `keep`, Mac `flush`, Linux `keep` and Linux `flush` references, with 0 differing,
  0 errors and 0 not run on each; `ex.stochastic_emitter` is excluded by name everywhere
  and `ex.coupler` on the two Linux references (Known issues). The two GDSII rows now run:
  `ex.ring_gds` holds on all four references, `ex.coupler` on both Apple ones.
- **L3 (CERT-5):** on the NVIDIA tables, 192 of 192 rows hold on each of the four legs
  (Triton under the default precedence and CUDA preferred, each against the Linux `keep`
  and `flush` references; VERDICT HOLDS). On the Metal table, under `flush`, 192 of 192
  rows run and the leg reads VERDICT HOLDS: 160 of 160 hold against the Linux flush
  reference outside the 32 rows pre-registered as rows where the two reference MEEPs
  differ (`ex.ring_gds` among them), and 192 of 192 against the Mac flush reference.

### Known issues

- **An `EigenModeSource` can differ from MEEP, and from another run of itself** (owner
  ruling R24). The lift solves the source's mode again with its own `get_eigenmode` call
  (MPB) instead of reading the amplitudes MEEP's own `add_eigenmode_source` solve stored,
  and on some hosts that solve, and MEEP's, is not bitwise reproducible across processes.
  Measured on the corpus row `ex.coupler` on the x86-64 Linux references of this round,
  under `keep` and `flush`: 12 of 20 runs held (2 corpus legs, 6 re-runs and 12 lift
  probes); a run that did not hold differed in 1,683,570 to 5,051,638 of 44,047,776 stored
  words; the only lifted inputs that varied between processes were the two sources' point
  amplitudes, under `flush` MEEP's own solve varied too (3 distinct MEEP digests in 6
  probes), and identical lifted inputs with identical MEEP words gave identical array-path
  words in 9 of 9 processes. On the Apple M1 Max the row held 4 of 4. The row is therefore
  excluded by name from the two Linux sets of the Tier B record and not run by L3. 1.0.1
  reads MEEP's stored eigenmode amplitudes instead of re-solving them
  (`docs/guides/compatibility.md`, "An `EigenModeSource` can differ from MEEP").
- **L3 runs only on the hosts that cut its record**: each row requires the MEEP images to
  carry the record's SHA-256, and a build under another prefix never does, so a round on a
  new GPU certifies by its gates and route campaigns.
- **Under held residency one Metal route control cannot bite** (owner ruling R29):
  `residency-sync_in-dropped` reads NOT-APPLICABLE on every held case, because a held
  host write goes through the sparse write paths and never through
  `Residency.acquire_write` (entered 0 times on 42 of 42 dispatched cases of each dispatch
  leg in this round). The held seam's evidence is
  `residency-verify-after-every-sync_out` and byte identity; a null that drops one
  sparse-door device write and must diverge is a 1.0.1 item
  (`docs/development/certification.md`, "The Metal route campaign").
- **Records the round's tools did not re-cut stay as committed**: the Triton family
  re-certification record that 22 of 75 Triton arms cite (its tool requires one host line
  across nine families that ran on three cards), the CUDA ADE re-cut, two Triton entries
  that declare a mixed policy, one CUDA weld and two Triton device-gate keys never seeded.
  35 entries of `tools/ci/pending_certification.txt` stay declared after the records
  entry, each with the step that removes it.

### The bit-parity coverage board is re-cut on series entries 01 to 178: no cell moves

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..178: entry 177's timing cutters and their tests are files of its
outcome subject, so its check refused; entry 178 changes Markdown only. Only three of its
generated files change (`board.json`, `board.md`, `outcomes.json`); the generator is
unchanged. Its test outcomes are the CUDA and Triton test files' cases of the package suite
(CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the device run
(Apple GPU), both run in the Mac reference environment on this tree's outcome subject: 216 test
files, 0 not run, 12,137 cases, per file equal to entry 176's on 216 of 216.
`dev_evidence.json` is carried unchanged: each of its 11 records is still on its live subject.

- **No cell moves** (N / D, refused rows in brackets), 0 transitions: CUDA (a) 55 / 55 (+1),
  (b) 57 / 57 (+8), (c) 18 / 18 (+47), (d) and (e) 53 / 53 (+12) under keep and under flush;
  Triton 55 / 56 (+2), 61 / 61 (+7), 27 / 29 (+39), (d) and (e) 53 / 56 (+12) under keep and
  0 / 56 (+12) under flush; Metal 60 / 60 (+4), 68 / 68 (+9), 29 / 29 (+48), (d) and (e)
  74 / 74 (+3) under flush. The array path against MEEP reads 6 / 6 families identical (+4
  refused) under keep and under flush.
- **The suites**, Apple M1 Max, Mac reference environment, every marker, dispatch off: the
  package suite under the CPU-only shim 33 failed, 15,001 passed, 1,031 skipped, every failure
  declared; the 96 device-visible files on the Apple GPU 1 failed, 4,187 passed, 331 skipped,
  the failure declared. Per id against entry 176's suites, 16,065 of 16,065 package ids read
  the same outcome, and 4,515 of the 4,515 device ids the two runs share; the device run's 1
  gone and 4 new ids are all in `test_cut_timing_record.py` (entry 177's), which the board
  does not carve.
- Of the board's 12,062 leaves 8 move: the build and record times, the two reports' names and
  digests, the outcome subject manifest and the board subject. Outcome subject
  `5baac1faf50bc87b` (1,152 files), board subject `41da12d02fcae9e7` (1,163 files).

### The 1.0 timing (CERT-8) is stated from its records: both platforms' identical-case tables under the gate of owner ruling R33, the lift and the break-even, and the fused products

The pages that state the release's speeds now state the 1.0 timing, which ran on these bytes on
2026-10-16 on one NVIDIA host (one RTX A6000) and one Apple laptop (one M1 Max), with MEEP
re-timed in the same session as the GPU on each. Every figure is read from the round's
records, whose tables were cut by tool from the stored rows under the gate of owner ruling
R33; nothing was timed for this entry. No package file moves.

- **`docs/guides/will-it-help.md`**: "The comparison with MEEP" is the 1.0 timing. On the
  NVIDIA host, per step, the Triton route and the CUDA-only route against MEEP's fastest rank
  count, with the array path beside them (six GPU ratios of record, 1.45× to 5.81×), every
  condition, MEEP's rate at each rank count and the controls of the same session (ranks bound
  to hardware threads, cost-based chunk splitting, and MPI ranks with OpenMP threads, which
  measured up to 2.7 % above MEEP's fastest pure-MPI count and are struck by a defect of the
  timing harness: the ratios of record stay against pure MPI, and the page gives both). On the
  Apple laptop, the Metal route against MEEP at 8 ranks at five sizes (0.82× to 2.72×, all of
  record under the new gate) with the array path and MEEP at each rank count, and the
  Metal-only supplement rows. A section states the gate, that it replaced the window-spread
  gate after the 2026-10-16 runs had been seen, why, and what each gate admits on both dates;
  another compares the round's first timing (2026-10-13) in summary. "Short runs" states the
  lift (now the bulk permittivity read), MEEP's start-up, the break-even step count on both
  machines and the case run to its own end time on the NVIDIA host, all derived and kept apart
  from the per-step figures, without the GPU's own first-step cost. The guidance table, the
  two-dimensional section, the array path off the supported range and "What has not been
  measured" are brought to the same figures.
- **The MEEP build is named on every MEEP figure**: MEEP 1.33.0 as each platform's reference
  environment builds it (owner ruling R27; single precision, `-O2 -ffp-contract=off
  -fno-fast-math`, a portable binary, Open MPI 5.0.8 with OpenMP on Linux and 5.0.10 on
  macOS). On the NVIDIA host the build the round's first timing used was timed again after
  the session, MEEP alone, and agrees within 0.3 % at MEEP's fastest rank count at
  2,097,152 and 7,077,888 cells; no such bridge ran on the Apple laptop.
- **`docs/guides/kernel-dispatch.md`**: "What fusing buys" states the 1.0 fused products
  against the single arms they replace (median 0.62× over 46 cases, two-pair plans ahead on
  0 of 34, and the crossover band between 2.4 and 5.4 million cells, where the 2026-09-21
  campaign had placed it between 1.0 and 2.4 million); the Metal section states the 1.0 Apple
  figures in place of the 2026-09-25 median; bit identity at size counts the 1.0 comparison's
  GPU rows. `docs/guides/capability-coverage.md`, `docs/guides/validation-and-performance.md`,
  `docs/guides/index.md` and `docs/development/execution-backends.md` follow.
- **`docs/development/timing.md`** gains "The 1.0 timing against 2026-10-13": both machines
  row by row, with what can be attributed (on the NVIDIA host the denominator held, and the
  512,000-cell gain, a fixed cost of about 0.14 to 0.15 ms per step, is attributed to the
  skipped deposit of a silent source by inference only) and what cannot (on the Apple laptop
  three things changed at once), the 2026-10-13 rows under each gate, and on which trees the
  2026-10-16 rows stay valid (those whose files match what the rows' processes imported; no
  timed process imported the cutters).
- **The dispatch-preference records stay at 2026-09-21** in 1.0.0, and every page that names
  them says so: the records the 1.0 timing cut land in a later 1.0.x release, because they
  are package data the dispatch-preference consult reads.
- **The release notes' timing section** states the headline figures with their gate status;
  `README.md`, `docs/development/certification.md` and section 12 of
  `parity/meep_gpu/rounds/1.0/PRE_CERTIFICATION_STATE.md` no longer say that the timing
  follows. The release date stays 2026-10-16 in the heading above and in `CITATION.cff`.
- Each of the files this entry changes is in the run closure of 0 of the 250 gate and probe
  scripts (`import_closure.py bill`) and named by no pin of the four ledgers.

### The timing cutters read every row by owner ruling R33; the bench that writes the rows is unchanged

Owner ruling R33 (2026-10-16) moves the 5 % gate of the CERT-8 timing from the spread of a
leg's timed windows inside one pass to the spread of its pass rates across a row's passes. It
was ruled after the 2026-10-16 runs had been seen: on the Apple run the window floor struck 6
of 8 GPU medians whose passes reproduced to 0.16-2.44 %. This entry lands the half of the
change that cuts tables and records (owner ruling R34); the half that writes rows moves to
1.0.1, so no stored row goes stale.

- **The rule, as every cut now applies it.** A GPU leg (each route, and the array control)
  enters when no pass of it is struck by another check and the spread of its pass rates,
  (max − min) / median over the row's passes, is at most 5 %. The within-pass window spread
  is recorded and no cut gates on it. MEEP's pass spread is recorded and never gated, as
  before. The quiet window, the variance precondition, the watcher, and the identity and
  dispatch checks are unchanged.
- **`parity/meep_gpu/timing_rows.py`** re-derives every leg of the identical-case table from
  the passes its record stores, never from the status it was collected with: a pass struck
  on the bench's window floor alone (`spread_within_gate`) reads unstruck, a strike naming
  another floor as well keeps that floor, and every other strike stands. Each table prints a
  leg's between-pass spread, the fused leg's window spread per pass and MEEP's between-pass
  spread, and states the rule and its limit.
- **`parity/meep_gpu/cut_timing_record.py`** neither requires nor gates on the window floor.
  The admitted rows of one case and grid shape, on one deposit-repair program and timed by
  one instrument (window count and target), are that point's passes; two or more enter only
  when their fused and their baseline rates agree within 5 %. A record so cut stamps
  `cut.gate` (`between_pass`); a record with no stamp was cut by the window floor, and
  `--check` recomputes each record by the rule it was cut by.
  **`parity/meep_gpu/build_fused_timing_report.py`** re-derives `reportable` from the floors
  that gate. `docs/development/timing.md` gains "The gate of record (R33)".
- **The bench is unchanged in 1.0.0.** `bench_fused_products.py` and `gpu_array_control.py`
  still count `spread_within_gate` among a row's floors and still mark such a row
  `TIMED-NOT-REPORTABLE`; the cutters read past that floor. Both files are in the import
  records of the rows they wrote (`bench_fused_products.py` in 135 of the 135 rows the
  2026-10-16 round stored, `gpu_array_control.py` in 11), so changing them would make every
  stored row stale and owe it a re-time (`timing_rows.py bill`: 8 of 8 Apple keys, 127 of
  127 NVIDIA keys). Their change, which records the window spread beside the floors rather
  than among them, moves to 1.0.1 with the bench tests that read it. Of
  `test_bench_fused_products_metal.py` only the report-builder tests land here. The
  recorder's limit therefore holds for rows collected on 1.0.0 too: the ladder stops at a
  row's first strike, so for a pass struck on the window floor it never evaluated the later
  checks (what the route launched, a GPU-only row's identity row, the deposit-repair route,
  the host conditions). The 2026-10-16 passes that R33 reads unstruck were audited from the
  ladder ledgers instead: 0 of the 17 on the Apple run fail a skipped check, and the NVIDIA
  run had none (0 of its 18 timed GPU passes carried a window-only strike).
- **Measured on this tree**, nothing re-timed:
  - **The timing tool owes nothing.** `timing_rows.py status`, with this tree as the release
    root, reads every key newest valid and owed nothing: the Apple rows 8 of 8 (5 of the
    release catalog, 3 of the GPU-only supplement) and the NVIDIA rows 127 of 127, with 0
    catalog keys lacking a row. `cert8.py plan --owed` plans 0 keys on both Apple stores. The
    7 files of the cutter half are in 0 of the 135 rows' import records, and
    `timing_rows.py bill` of them re-times 0 of 8 and 0 of 127 keys.
  - **The shipped cutters re-cut the R33 tables byte for byte**, 18 of 18 outputs: the Apple
    identical-case tables of the release catalog and of the GPU-only supplement with their
    status reads, the NVIDIA identical-case table with its status read, the R33
    dispatch-preference cuts (Triton 67 of 67 rows admitted, CUDA 29 of 57) and the
    fused-product report (74 rows, 46 with a fusion number). Those dispatch-preference cuts
    are not shipped: 1.0.0 keeps the 2026-09-21 records.
  - **The shipped records still recompute.** `cut_timing_record.py --check` recomputes the
    2026-09-21 records byte for byte: Triton `965317c136ad` (69 rows from 17 files) and CUDA
    `cff4bdae8f80` (65 rows from 12 files).
  - **Tests** (Apple M1 Max, Mac reference environment, dispatch off): the timing and bench
    test files 255 passed, 2 skipped, 0 failed of 257; the harness suite (`parity/meep_gpu`)
    2,800 passed, 16 skipped (a declared resource), 0 failed of 2,816.
  - **Bill.** `import_closure.py bill`: the 8 files this entry changes (the cutter half and
    this changelog) are in 0 of 250 gate and probe closures. No ledger pins any of them, and
    no package file moves.

### The bit-parity coverage board is re-cut on series entries 01 to 175: the certification columns land

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..175: entries 171 to 175 moved its outcome subject (entry 173's
chunk-tool test and fixtures, the records entry's ledgers and board builders, the
documentation entry's timing-host files and tests) and its board subject (the pending list),
so its check refused. Only three of its generated files change (`board.json`, `board.md`,
`outcomes.json`); the generator is unchanged. Its test outcomes are the CUDA and Triton test
files' cases of the package suite (CPU-only shim, every marker, dispatch off) and the Metal
test files' cases of the device run (Apple GPU), both run in the Mac reference environment on
this tree's outcome subject: 216 test files, 0 not run, 12,137 cases. Per file they equal
entry 170's on 212 of 216 files; the other 4 are the files whose pending tests the records
entry turned green (`test_metal_kernels.py` 1, `test_metal_weld_contract.py` 6,
`test_triton_cylindrical_composition.py` 1, `test_triton_weld_contract.py` 2).
`dev_evidence.json` is carried unchanged: each of its 11 records is still on its live subject.

- **(a) to (c) do not move** (N / D, refused rows in brackets): (a) the chunk-record select,
  CUDA 55 / 55 (+1), Triton 55 / 56 (+2), Metal 60 / 60 (+4); (b) MEEP's A form, 57 / 57
  (+8), 61 / 61 (+7), 68 / 68 (+9); (c) host word identity, 18 / 18 (+47), 27 / 29 (+39),
  29 / 29 (+48). The array path against MEEP reads 6 / 6 families identical (+4 refused)
  under keep and under flush; (d-dev), development evidence, CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, as before.
- **(d) device identity and (e) CERTIFIED land with the R27 records.** Every byte-binding
  ledger entry now records the live array path (CUDA 55 of 55, Triton 66 of 66, Metal 63 of
  63, against 0 of each on entry 170's board). CUDA reads 53 / 53 (+12 refused) under keep
  and under flush, in (d) and in (e); Triton 53 / 56 (+12 refused) under keep, in (d) and in
  (e); Metal 74 / 74 (+3 refused) under flush, in (d) and in (e). 466 cells move, every one
  from gap to landed (CUDA 212, Triton 106, Metal 148), and none the other way.
- **What stays a gap, by the board's own reasons.** Triton keep, 3 of 56 rows (`forms`,
  `no_pml`, `offdiag_update_e`): the bit-identity gate's entry reads no status. Triton flush,
  0 of 56: no module's pinning entry carries a live run under flush on the board's rule (48
  rows read that alone, 7 also no status, 1 also bytes moved since its run).
- **The suites**, Apple M1 Max, Mac reference environment, every marker, dispatch off: the
  package suite under the CPU-only shim 33 failed, 15,001 passed, 1,031 skipped, every
  failure declared (entry 170's read 53, 14,981, 1,031 of the same 16,065: the 20 are the
  records entry's removed pending ids in the package suite); the 96 device-visible files on
  the Apple GPU 1 failed, 4,184 passed, 331 skipped, the failure declared (12 failed at entry
  170). Per id, 16,045 of 16,065 package ids and 4,505 of 4,516 device ids read the same
  outcome as entry 170's suites; the 20 and the 11 that move, failed to passed, are all
  pending ids the records entry removed.
- Of the board's 12,062 leaves 2,875 move, all in the (d) and (e) cells (state, entry, run,
  host label and reason leaves) and their counts, the ledgers' oracle leaves and
  certified-zero reasons, the subject manifests, the two reports' digests and the times.
  Outcome subject `a83679c745e50993` (1,152 files), board subject `d492a18d9930faf8` (1,163
  files).

### The 1.0 documentation: the release notes from the round's records, certifying a new GPU, the `EigenModeSource` caveat, and the R27 text and tool fix-ups

- **`docs/development/certify-a-new-gpu.md`** (Development, after "Running the certification
  harness") takes a contributor who does not hold the evidence archive through the round that
  certifies an NVIDIA compute capability or an Apple GPU architecture the release does not
  (an H200 at 9.0, an Apple M3) on the released bytes: the reference environment from the
  platform's build script (`--refresh-corpus` for an older one), the round's inputs bundle
  (`round_inputs_1_0.tgz`, its SHA-256 the committed `INPUT_BUNDLE.sha256`), the device's own
  chunk index (`chunks.py partition --new-device`), the per-chunk checks and launches with
  `--new-capability` / `--new-architecture`, sharing a card with another user's idle process
  (`--share-idle-foreign`, `--share-min-free-mib`) and yielding it, verify and the units run
  again alone, the bind that adds the device's records (the `# STOP` set-aside loop
  included), the route campaigns, and the package to send back. Its CPU-only commands were
  run on a checkout of this tree with the bundle unpacked (no GPU): the bundle unpacks only
  into an existing `results/` (`1782 of 1782 verified`); the cc 9.0 index cuts four chunks
  of 30 or 31 units (123), each check of a chunk reads `CHECK OK` with `--new-capability`
  and refuses without it (both tables' admission, and the index's new-device mode), and the
  Metal index of a new architecture cuts three chunks (1, 29 and 32 of 62 gates).
  `certification.md`, `docs/development/index.md`, the navigation and the RUNBOOK's CERT-3
  and CERT-6 rows point at it.
- **The `EigenModeSource` caveat of owner ruling R24**, stated with `ex.coupler`'s
  measurement on the x86-64 Linux references (12 of 20 runs held; only the lift's re-solved
  point amplitudes, and under `flush` MEEP's own solve, varied between processes; identical
  inputs gave identical words in 9 of 9 processes; the Apple M1 Max held 4 of 4), in
  `docs/guides/compatibility.md` ("An `EigenModeSource` can differ from MEEP"), INSTALL.md
  ("What is bit-identical, exactly"), `certification.md`'s L3 section and the 1.0 release
  notes.
- **`tools/check_install.py` reads MEEP's libGDSII fact** (`meep.with_libGDSII()`) in both
  of its MEEP fact sets, so `--require-reference` and the identity check judge it, and refuse
  a MEEP built without libGDSII, where they printed "(libGDSII not read)" and passed it.
  Tests in `test_identity_reference.py` drive both children with a stand-in MEEP.
- **The timing harness's two macOS host readers call `/usr/sbin` programs by absolute
  path**: `timing_host.apple_gpu_state` runs `/usr/sbin/ioreg` and
  `bench_fused_products._static_host_facts` `/usr/sbin/sysctl`. The reference `PATH`
  (`<prefix>/bin:/usr/bin:/bin`) holds no `/usr/sbin`, so on it the GPU utilisation read
  unreadable (`--gate-apple-gpu-util-max` could never pass) and the machine model and CPU
  read `None`. `test_timing_host.py` pins the command.
- **`environments/locks/README.md` regains the comparison of the 16 field-update kernels
  across builds**, with this round's measurements: on Apple silicon 16 of 16 kernels
  instruction-identical to the contraction-off build of 2026-10-05 (only absolute addresses
  differ, moved by the libGDSII code), on Linux 145 of 145 kernel bodies and 72,820 of 72,820
  instructions identical to the previous Linux reference environment's, addresses masked.
  `tools/fma_census.py`'s comment on `KERNELS` names that section; the README also says that
  `sysctl` and `ioreg` are called by absolute path, and how `--require-reference` reads
  libGDSII.
- **INSTALL.md** states the two measured end-to-end runs of the build scripts (Linux 377 s,
  macOS 296 s, 2026-10-14, before the corpus siblings, which `--refresh-corpus` then added
  to both environments) where it said they had not been run, and the MEEP line of the
  reference check with libGDSII read.
- **The 1.0 round's inputs**: the counts of "The 1.0 round's inputs are one derived
  manifest" below are the derived manifest's after the Tier B re-mint (1,782 files, 1,728 row
  files, 192 rows per set).
- **The 1.0 release notes** head this file, every figure filled from the round's records:
  the chunked gates, the binds, the route campaigns and the licence, the boards, Table 2 and
  L3, and the known issues (the `EigenModeSource` caveat, L3's host rule, the held-residency
  control of owner ruling R29, and the records the round's tools did not re-cut). The release
  date is 2026-10-16, the same in the heading and in `CITATION.cff`'s `date-released`. The
  `## Unreleased` heading is removed, so every entry from here down to 0.9.2 is part of 1.0.0.
- **Table 2** (`docs/development/meep-identity-corpus.md`) is generated again by
  `build_meep_identity_table.py` from the R27 Tier B record: 193, 193, 192 and 192 of 194
  rows hold, `ex.coupler` excluded on the two Linux references (R24) and run with
  `ex.ring_gds` everywhere else.
- **The 0.9-era status texts are the 1.0 round's**: "Certification status of this release"
  in `docs/development/certification.md` and in `README.md`, the certification suite in
  `docs/development/testing.md` (the 0.9.0 preview's suite counts labelled as such, the 1.0
  counts per environment owed to CERT-7), the expected-failures warning of
  `docs/guides/validation-and-performance.md`, `docs/design/floating-point.md`'s gate
  records, the boards (cut 2026-10-15; 537, 538 and 507 of 597 with a fused product) and the
  route campaigns (the 1.0 round's) in `docs/guides/kernel-dispatch.md`,
  `docs/development/execution-backends.md` and `docs/guides/capability-coverage.md`, and the
  held residency default, which the round certified. The boards' second column of 0.9 (the
  figure without the complex-storage evidence record: 244, 258 and 245) has no 1.0
  counterpart and is removed: since the complex-multiply expansion licence was deleted the
  complex families dispatch by default, and the 1.0 boards read no such record.
- **Owner ruling R29** is stated where the Metal route campaign is described
  (`docs/development/certification.md`): under held residency `residency-sync_in-dropped`
  cannot bite, and the held seam's evidence is `residency-verify-after-every-sync_out` and
  byte identity.
- **Timing (owner ruling R25)**: the pages that state the release's speeds (`README.md`,
  "Will it help?", "Compiled-kernel dispatch", the capability checklist) said the figures
  were measured on earlier bytes and that the 1.0 timing (CERT-8) was still to come (the
  timing entry above now states it), "Will it help?" and "Compiled-kernel dispatch"
  also that 1.0 ships the 2026-09-21 dispatch-preference records; the other speed statements are dated and carry their own
  licence, and per-step figures stay apart from end-to-end ones.
- **`docs/development/certify-a-new-gpu.md` against this round's own commands**: a shared
  card's memory bound from the units' measured peak (a Triton H-to-D lift used 12,348 MiB),
  the Triton fleet units on an unshared card, the lines the round's bind met that a new
  device's will meet too (the two `# REFUSED` lines of `173-BINDFIX`, rebinds exiting 1 on a
  skip after writing, the ADE re-cut's refusal), the family re-certification's nine families
  from one card, the new device's `# no board re-point` line, and the round's route walls.
- **`parity/meep_gpu/rounds/1.0/RUNBOOK.md`**: CERT-5's walls as the R27 round measured them,
  its L3 commands as they ran in the reference environments, the Metal L0 list of 32 of 192,
  and section 3's card sharing (the fleet driver refuses a shared card; the bound is the
  units' peak). **`PRE_CERTIFICATION_STATE.md` gains section 12**, the certification round,
  every number with the record it comes from.
- **The review of these pages against the round's records** corrected: the Triton route's
  legs on the certify page (a blocking preflight and three legs, `legs.ROUTE_TRITON`), the
  measured sizes of the two environments (8.9 GiB and 2.0 GiB) and the Mac build's jobs; the
  steps this round ran (CERT-1 to CERT-5; CERT-7 is owed); the release notes' sentence on a
  new GPU, which now claims nothing about an H200 or an M3; the comparisons with MEEP, stated
  by date (one on 2026-09-28; none on Apple hardware had then produced a figure of record) and the
  one Apple median with its gate status; and "Will it help?"'s row on complex fields, which
  dispatch by default in 1.0. The certify page states its family re-certification recovery as
  read from the tools' code; the held-residency paragraph of `certification.md` states what the
  other two residency controls read on the same 42 cases; Table 2's `ex.coupler` exclusion and
  section 12 of `PRE_CERTIFICATION_STATE.md` name the round's records without their private
  paths.
- Each of the 27 files this entry changes is in the run closure of 0 of the 250 gate and
  probe scripts (`import_closure.py bill`) and named by no pin of the four ledgers, so no
  gate artifact moves. Five are in the bit-parity coverage board's subjects
  (`timing_host.py`, `bench_fused_products.py` and three tests), which this entry moves, and
  `timing_host.py` and `bench_fused_products.py` are timing-harness files a CERT-8 row
  imports. The patched `tools/check_install.py --require-reference` in the Linux reference
  environment reads `1.33.0, single precision, MPI build, libGDSII: the reference's build`
  and exits 0, as does `--require-identity-reference keep`.

### The R27 round's records are committed: the four bound ledgers, the board bindings and the pending list

The R27 certification round of 1.0 ran every gate of the three tables on the frozen commit
`2644fc1d` (stamp `2026-10-15_10`) in the new reference environments, and the NVIDIA and
Metal route campaigns on the same bytes; its records now land in the tree. Nothing else
moves: no package module, kernel, gate, census tool, round driver, needle or mutation anchor.

- **The four ledgers.** `meep_gpu/cuda_kernels/fingerprints.json`,
  `meep_gpu/cuda_kernels/certification.json` and `meep_gpu/triton_kernels/fingerprints.json`
  as the NVIDIA bind wrote them, and `meep_gpu/metal_kernels/fingerprints.json` as the Metal
  bind wrote it with entry 172's device-tier re-cut, each from the round's released gate
  runs. Then `recut_driver_dispatch_record.py` re-cut each table's `driver_dispatch` record
  from the released route campaigns `dispatch_fused_route_2026-10-15_10_cc86` (with the
  licence `dispatch_end_to_end_2026-10-15_10`), `dispatch_fused_route_cuda_2026-10-15_10_cc86`
  and `dispatch_metal_route_2026-10-15_10_applegpu_g13s`; `launch.write_fingerprints()`
  re-cut the Metal ledger's `metal_kernels` block; and `record_array_path_oracle.py --write`
  minted the array-path oracle (fixture set `real-v6`, fingerprint `1ab516279d74002a`) in
  the three kernel ledgers, which `--verify` reads back on each. `certification.json` is the
  bind's byte for byte.
- **The Metal host block re-cut also moves its kernel-source digests.** The one writer of
  the block, `launch.write_fingerprints()`, rewrote `host_sha256` (82 of the 89 recorded host
  modules moved, and `chunk_bounds.py`, `folded_pass.py` and `softfp64.py` are added) and
  `kernel_source_sha256` (36 of 36 entries). The new kernel-source digests are the ones the
  R27 Metal gates ran: 36 of 36 are found in the round's Metal gate runs, against 0 of 36
  for the digests they replace, which were last written at 0.9.0. The validated torch
  version and Metal front end are unchanged.
- **The board bindings.** `repoint_board_bindings.py triton` re-pointed the Triton board's
  `GATE_BOUND` (39 of 39 entries) and `repoint_board_bindings.py metal` the Metal board's
  gate references (45 over 28 directories) to the round's gate roots; the CUDA board binds
  through the ledgers. The Metal re-point ran on a results view trimmed to the bind's
  selection, because the tool refuses between two runs of one gate and the modular re-run of
  one unit left two (one not released, the other the one the ledger welds). On the round's
  three censuses and the H->D probe re-cut on these bytes (`h_to_d_seam_2026-10-15_213713d6`),
  the boards read served in dispatch 352 (CUDA), 334 (Triton) and 327 (Metal) of 597 priced
  seam-instances, unchanged instance by instance, and served 538, 537 and 507 of 597.
- **The pending list.** 21 entries (26 lines) leave `tools/ci/pending_certification.txt`:
  13 turned green with the route re-cuts, 5 with the Metal host block and the oracle, and 3
  were green on the binds' ledgers. Each passes in the evidence view and in the CI view (no
  `results/`), and no entry that stays passes in either. One of them is the hygiene guard's
  ledgers leg: the four ledgers read 0 hits, so `tools/ci/check_hygiene.py` is green on the
  whole tree, all three legs, and no leg of it is declared any more. 35 entries stay: 33 fail
  in both views, each for the reason its block states, one runs only where the evidence
  archive's Metal board is present, and one reads the evidence archive and is not measured
  without its marker. The 8 blocks the removed entries led move to the list's History. 0
  entries are added.

The entry moves no certified byte except the ledgers themselves. No file it changes, other
than the four ledgers, is pinned by any entry of the four ledgers, before or after, and every
digest pin of the new ledgers reads the same hold or drift verdict on the tree before this
entry and on this tree (Metal 649, Triton 901, CUDA 547 and certification 164 digest pins,
each the same on both). None of the 250 gate and probe scripts holds a changed non-ledger file
in its run closure (`import_closure.py bill`), and no import record of the round's artifacts
names one (0 of the 935 payloads that carry one under the round's `2026-10-15` result roots;
the control, `meep_gpu/fastpath.py`, is named by 697). The coverage board is not re-cut here;
`build_parity_board.py check` refuses until the entry that re-cuts it.

### The chunked NVIDIA bind names only Triton keys the ledger holds

- `parity/meep_gpu/rounds/1.0/chunks.py bind` (and `legs.py bind --chunks`) builds each
  stamp's `rebind_triton_welds.py --only` from the selected units' `CAMPAIGN_DIRS` keys that
  the checkout's Triton ledger holds, read by path. The tool refuses a whole `--only` naming
  a key the ledger lacks, and the R27 bind named two (`triton_bfast_device_gate`,
  `triton_special_kz_device_gate`, rows typed ahead of their seed), so two of its seven
  Triton lines bound nothing. A selected unit whose key the ledger lacks is now named on a
  `# REFUSED` line, which withholds nothing; a stamp left with no key prints no Triton line;
  every other line is unchanged (`test_round_chunks.py`, on copies of the R27 index,
  selection and printed bind in `parity/meep_gpu/fixtures/round_chunks_r27/`).

### The Metal rebind re-cuts each weld's device tier from the run it binds

- `parity/meep_gpu/rebind_metal_welds.py` re-cuts every bound weld's `device_sha256` as
  `mint_metal_weld.py` cuts it, only where the run imported each pinned module at the
  checkout's bytes, else skips the entry by name (`test_rebind_metal_device_tier.py`).

### The bit-parity coverage board is re-cut on series entries 01 to 169

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..169: the R27 entries 155 to 169 moved both its subjects (package
modules, the reference environments' build scripts, harness tests and the re-minted Tier A
fixtures in the outcome subject; the pending list in the board's), so its check refused. Only
three of its generated files change (`board.json`, `board.md`, `outcomes.json`); the generator
is unchanged. Its test outcomes are the CUDA and Triton test files' cases of the package suite
(CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the device run
(Apple GPU), both run in the Mac reference environment on this tree's outcome subject: 216
test files, 0 not run, 12,137 cases. Per file they equal entry 154's on 216 of 216 files.
`dev_evidence.json` is carried unchanged: each of its 11 records is still on its live subject.

- **No cell moves** (0 transitions against entry 154's board; N / D, refused rows in
  brackets): (a) the chunk-record select, CUDA 55 / 55 (+1), Triton 55 / 56 (+2), Metal
  60 / 60 (+4); (b) MEEP's A form, CUDA 57 / 57 (+8), Triton 61 / 61 (+7), Metal 68 / 68
  (+9); (c) host word identity, CUDA 18 / 18 (+47), Triton 27 / 29 (+39), Metal 29 / 29
  (+48).
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 56, Metal 0 / 74)
  until the certification round re-gates.
- **The array path against MEEP** reads 6 / 6 families identical (+4 refused) under keep
  and under flush, on the Tier A sets re-minted in the new reference environments.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, as before.
- **The suites**, Apple M1 Max, Mac reference environment, every marker, dispatch off: the
  package suite under the CPU-only shim 53 failed, 14,981 passed, 1,031 skipped, every
  failure declared; the 96 device-visible files on the Apple GPU 12 / 4,173 / 331. Against the
  same suites in the development interpreter every id reads the same outcome (16,065 of 16,065
  package ids on entry 164; 4,514 of 4,514 device ids on entry 153, plus 2 new ids that pass):
  the reference environment moves no test outcome.
- Of the board's 11,412 leaves 14 move, none a cell or a reason text: the two Tier A sets'
  mint times and the flush set's recorded contraction, the build and record times, the two
  reports' names and digests, the subject manifests and the module digests of
  `fastpath_cuda.py` and `metal_dispatch.py` (entries 155 and 157). Outcome subject
  `35292cc599fa5e9c` (1,148 files), board subject `6b627ad9c7e64d75` (1,159 files).

### The R27 1.0 freeze (CERT-1): the identity ledger re-minted in the new reference environments

The series is frozen for the 1.0 certification round that owner ruling R27 (2026-10-14)
orders: one reference environment per platform, built end to end by the repository's own
script from the locks (MEEP 1.33.0 from the pinned tarball, single precision, MPI,
contraction off, with libGDSII; the corpus the same tarball's tree, kept in the
environment), so the round re-runs in full in the new environments, under the stamp entry 165
set (`2026-10-15_10`). Entries 155 to 168 carry that work; this entry re-mints the identity
ledger in the new environments and records the freeze criteria, and the next entry re-cuts the
coverage board, whose subject it moves.

- **`meep_identity.json` and the 16 Tier A capsules are re-minted by
  `record_meep_identity.py` in the new reference environments**: both environments are new
  (both MEEP images and both build records), and 1 of the 20 files the mint hashes moved
  (`meep_gpu/from_meep.py`, entry 155). Keep on the Apple M1 Max against the Mac reference
  environment's MEEP (174 s), flush on x86-64 Linux against the Linux reference environment's
  MEEP (MEEP's flush on import; 335 s, into the keep-minted fixtures), 8 of 8 cases each, one
  invocation per set, each MEEP first judged an identity reference by `tools/check_install.py`
  against its own build record: 39,991,116 stored and 65,610,420 reader words equal to MEEP's
  in each, and the keep files byte-identical after the flush mint (32 of 32). Of the ledger's
  1,899 leaves 80 move and none is a digest, a word count or a verdict: times (36), the 16
  capsules' byte counts and sha256 (30), the moved input's digests in both sets (4), the
  reference blocks (9: each set's build record and its two MEEP images, and the flush set's
  recorded contraction, now `off`) and the fixture bytes (1; 4,636,168 of 5,000,000).
  Every row of both sets replayed with MEEP not importable holds: 16 of 16 on the Apple M1 Max
  (flush rows under FZ) and 16 of 16 on x86-64 Linux (flush rows under the MXCSR lever), 2,916
  of 2,916 stored and 2,916 of 2,916 reader digests per set on each host, the planted flip
  caught and placed in 8 of 8 rows per set on each host. The Tier A test file reads 62 of 62
  on the Mac and, NumPy only, 62 of 62 entering in keep and 62 of 62 entering in flush on
  x86-64 Linux. `docs/development/testing.md` states the new sizes.
- **The array-path oracle does not move**: `record_array_path_oracle.py --report` reads
  `real-v6`, fingerprint `1ab516279d74002a`, 20 of 20 case digests equal on the Apple M1 Max
  (NumPy 2.4.3) and on x86-64 Linux (NumPy 2.2.6), so `FIXTURE_SET` is not bumped; the kernel
  ledgers' `array_path_oracle` leaves stay owed to the round's records stage (CERT-4 records).
- **The freeze criteria**, every leg in the reference environments. L1′ on one RTX A6000 under keep and flush:
  123 case-rows per policy (cylindrical, family, oracle, hand and tiny legs), 0 of
  26,302,894,838 words differing after every sub-step, the planted flip caught 123 of 123, the
  mixed-policy control differing 2,446,768 of 6,272,469 words; the hand references on CuPy 0
  of 197,902,068 words per policy; the native CuPy folded complex 17 of 17 and folded
  off-diagonal 16 of 16; `test_cupy_array_path.py` 20 of 20 on the device; every count equal
  to the R22 freeze's run. AP9 on
  the Apple M1 Max: `validate_pml` 308 of 308 (44 skipped), `validate_no_pml` 112 of 112,
  `validate_fold` 8 of 8, `validate_folded_complex` 17 of 17, `validate_folded_offdiag` 16 of 16
  with refusals 13 of 13; `record_numpy_reference_weld.py` has nothing to write. The Metal
  route gate's controls dry run reads CONTROLS DRY RUN OK against the last released shipped
  leg (42 of 42 cases, 0 moved, nothing to smoke). The per-unit closure digests equal those
  every chunk was cut against (123 of 123 NVIDIA units, 62 of 62 Metal units), and with this
  entry's files counted as changing `chunks.py ready` reads 4 of 4 NVIDIA and 4 of 4 Metal
  chunks ready: the freeze stales no unit. Needle anchors 0 differences; 212 mutation anchors,
  0 stale. The hygiene guard reads CLEAN on the tracked files and the board's subject; its
  ledgers leg stays declared until the round's bind writes neutral labels. The corpus Tier C
  (Table 2) runs beside the round in the reference environments; this entry moves no byte a
  corpus row loads.
- `parity/meep_gpu/rounds/1.0/PRE_CERTIFICATION_STATE.md` section 11 records the freeze.

### A lane may share a card with another account's idle process, by opt-in only

- **`run_lane.py --share-idle-foreign <uuid>:<pid>[,<pid>]` with `--share-min-free-mib`**
  (`round.py` hands both to every lane of the gate stage and refuses them on the route
  stage or without the memory bound): the card named is admitted although compute apps run
  on it, when on both readings every app on it is a named PID, the named PIDs and the card
  read 0 % utilisation, and its free memory reaches the bound. Any other app, a named PID
  computing, or memory short refuses it by name; the lane records the shared card, the PIDs
  and the free memory in its log and its lock's label. Without the flag a card holding any
  compute app is never taken, as before. Owner ruling R28: certification units may share a
  card whose other process is idle, and yield it the moment its owner needs it (RUNBOOK
  section 3). `rounds/1.0/test_run_lane_share.py` pins it.

### The reference environments keep the corpus files that live outside `python/`

- **The corpus siblings.** Two corpus rows read a file outside the release tarball's
  `python/` tree: `test_simulation.py`'s `test_epsilon_input_file` hands
  `../../tests/cyl-ellipsoid-eps-ref.h5` to MEEP as `epsilon_input_file`, and
  `test_numpy_epsilon` reads it with h5py. The reference environments kept only `python/`,
  so in them the first aborted the process and the second silently measured vacuum (the
  capture keeps the simulation built before the read). Both are in the H->D lift basis of
  three gate units (`gate_metal_folded_fused_hd_pair`, `cuda_fused_hd_pair_2026-09-06`,
  `gate_triton_folded_fused_hd_pair`), so their lifts replayed them wrongly too. The files a
  row reads outside `python/` were found by tool over the 194 rows: a grep of the corpus
  for paths that leave `python/`, and every row's capture, structure initialisation and
  first steps under a file-access trace; both find only this file.
  `build_meep_133_record.py`'s `CORPUS_SIBLINGS` names it; `keep-corpus` keeps it beside
  `python/` (`share/meep-gpu-build/meep-1.33.0/tests/`), lists it in
  `corpus_siblings_sha256.txt`, and `verify-corpus` checks it; the record's `corpus` entry
  names each sibling with its SHA-256 (`siblings`).
- **Every reader checks them by name.** `legs.verify_corpus(root, files, siblings)` refuses
  a sibling absent from beside the corpus root or of other bytes, naming it;
  `legs.read_corpus_record` and `l3_corpus.recorded_corpus` read the entry through one
  function, `legs.corpus_siblings`, and refuse a record without it (`legs.SIBLINGS_MISSING`),
  so the gate stage, `metal_round.py` and L3 hold one rule. An explicit `--corpus-root`
  needs the siblings beside it too.
- **Refreshing an existing prefix.** `build_meep_133_{macos,linux}.sh --refresh-corpus`
  brings the corpus of an environment the script built to what it keeps now, and does
  nothing else: it fetches the pinned tarball (or copies `MEEP_TARBALL`, a new pass-through,
  verified by SHA-256 like the download), stages the corpus a build would keep, copies in
  what the prefix lacks, refuses a kept file of other bytes or a corpus file the tarball
  does not hold, rewrites the two lists and only the record's `corpus` entry (a record that
  is not as the build wrote it is refused), verifies, and writes `corpus_refresh.json`
  with the record's SHA-256 before and after. MEEP, the locks and the packages are not
  touched, and nothing is removed.

### Every Linux round child builds Triton's launchers with the reference environment's compiler

- Triton compiles a C launcher for each kernel with `$CC`, else the first `gcc` on `PATH`.
  No round driver set `CC`, so on the GPU host the host's own `gcc` 9.4 built every
  launcher, outside the environment, and a host without one would fail at the first launch
  ("Failed to find C compiler"). `rounds/1.0/legs.py` now holds one rule,
  `launcher_toolchain`: `CC` and `TRITON_LIBCUDA_PATH` as the caller set them (`CC`, and
  `--libcuda-stub` else `TRITON_LIBCUDA_PATH`), else, in a reference environment (one that
  holds `share/meep-gpu-build/reference_build.json`), its own
  `bin/x86_64-conda-linux-gnu-cc` and the libcuda link its build recorded in
  `share/meep-gpu-build/libcuda.json` (`lib/meep-gpu-libcuda`), that directory also first on
  `LD_LIBRARY_PATH`. A reference environment without its compiler, or without the link its
  record names, is refused by name.
- `run_lane.py` gives the toolchain to every child, the licence legs' `env -i` environment
  included, and stops before it takes a GPU on a refusal; `fleet_launch.py` exports it to
  every fleet gate (the fleet driver's own default stub is no longer reached in a round);
  `l3_corpus.py` gives it to every NVIDIA row, with or without `--clean-env`, and records it
  in `leg.json`; `cert8.py run` exports it to every timing unit on Linux. RUNBOOK section 3,
  `docs/development/testing.md` ("On an NVIDIA host"), `docs/development/timing.md` and
  `docs/development/certification.md` ("On a shared host") name `CC` and the libcuda
  directory. The changed drivers are in the run closure of 0 of the 250
  gate and probe scripts.

### The R27 round's stamp is `2026-10-15_10`

- `meep_gpu/route_stamps.py` reads `ROUND = "2026-10-15_10"`, set before the R27 freeze so
  that the round's chunked indices, which name this stamp, are accepted by the chunk tooling
  while the gate chunks run; the freeze entry re-mints the identity records on these bytes.
  The R22 round's three route campaigns, which ran, join `route_stamps.SPENT`:
  `dispatch_fused_route_2026-10-14_10_cc86`, `dispatch_fused_route_cuda_2026-10-14_10_cc86`
  and `dispatch_metal_route_2026-10-14_10_applegpu_g13s`. The pending list's "Removed when"
  clauses and the Metal route examples in `docs/development/certification.md` name this
  round's campaigns. The registry is in the run closure of 0 of the 250 gate and probe
  scripts, so no gate's import record moves.

### The R27 fix-ups: one corpus rule for every driver, and the texts the lanes left stale

- **One corpus rule.** `rounds/1.0/l3_corpus.py` judges the MEEP corpus tree with
  `legs.verify_corpus`, the rule the gate stage and `metal_round.py` already use, instead of
  a rule of its own: an unlisted file with any suffix Python imports
  (`importlib.machinery.all_suffixes()`, byte code outside `__pycache__` included) is
  refused, where L3 refused only `.py` and skipped `._*`, `.DS_Store` and `.pyc` anywhere;
  `._a.py` and a stray `.pyc` are now refused by both, `.DS_Store` is recorded by both. The
  record's file count is required, as `legs.read_corpus_record` requires it. A test holds
  the two drivers to one verdict on the files where they used to differ.
- **The sigma reader's retirement in the user's documents**: the INSTALL.md troubleshooting
  entry and `docs/guides/troubleshooting.md` describe what stock MEEP refuses and what lifts
  it, and `docs/guides/capability-coverage.md`, `docs/design/floating-point.md` and
  `docs/reference/public-api.md` name two sigma routes, not three.
- **The reference environments' build in the remaining documents**:
  `parity/meep_gpu/README.md`'s build rows and "Building the reference MEEP",
  `docs/development/certification.md` ("The MEEP a round is measured against": the platform
  scripts run directly, both with the identity flags, the retired knobs refused, what the
  build keeps), `docs/development/testing.md` (ffmpeg from the imageio-ffmpeg pin),
  `environments/locks/README.md` (pip keeps no wheel cache) and RUNBOOK's CERT-1 section (an
  environment from before the additions is rebuilt, not patched; the kept corpus as
  `MGPU_SITE_MEEP_SOURCE`).
- **The round's runbook and recipes**: RUNBOOK section 3's Metal row names the corpus the
  fleet gates get (`MEEP_GPU_CORPUS_ROOT`, `--corpus-root` else the build record's), its
  CERT-6 to CERT-8 section points at `cert8.py` and timing.md's "The release round, one row
  at a time", and certification.md's hand-run CUDA recipe sets `CUPY_ACCELERATORS=` (empty)
  before CuPy is imported, as every lane child does.

### Any set of gate units certifies as a chunk, and a new device is never admitted silently

Chunked certification (`parity/meep_gpu/rounds/1.0/chunks.py`) judged an artifact by the chunk
it ran in: a launch record naming other units than the index's chunk refused the whole chunk
(nine Metal units moved into a chunk of their own made `verify` refuse 33 of the 62 units that
had not moved), and only whole chunks relaunched. Under per-kernel certification a unit's
evidence is its own run's, so the tooling now says so. No package module, gate, kernel or
ledger moves; the round tooling is in none of the 250 gate and probe run closures
(`import_closure.py bill`).

- **Any subset is a chunk.** `chunks.py add --units` (or `--from-verify`, the units `verify` did
  not hold) runs any units, one alone included, as a new chunk; `relaunch` re-runs a chunk's
  units in a new chunk; `repartition` retires the chunks that never launched and packs their
  units afresh. A unit may be in several chunks; a chunk id is allocated once and never reused.
- **Each unit binds from its own newest run that holds.** `chunks.py verify` reads the units of
  a run from its launch record, never from the index's plan, and takes per unit the newest run
  whose import record holds on the release tree, whose verdicts pass (`round_status.py`) and
  whose device is the index's, read and never assumed (an NVIDIA artifact's
  `stamped_device` with a Triton campaign's `device.json`, or for a unit with neither the rest
  of its launch's; a Metal launch's environment record); a run whose device cannot be read is
  refused. `--selection` writes
  that choice with every run and why it was or was not taken. Re-partitioning the tree's 62 Metal and 123 NVIDIA units into
  arbitrary chunk sets leaves the verdict unchanged, unit by unit
  (`test_round_chunks.py`). `chunks.py bind --selection` and `legs.py bind --chunks
  --selection` print the commands that bind exactly that choice: the Triton rebind once per
  selected stamp with `--only` its keys, the Metal rebind and the CUDA records and rebinds once
  per selected stamp in launch order, the family re-certification over roots ordered so each
  family's selected run is the first it reads. A run a tool would bind against the choice
  (or a unit's directory under a selected stamp whose launch record does not list it) is
  named on a `# STOP` line with the command that sets it aside, and while one stands the
  bind prints no command that writes a ledger and exits 1. Its first command, `verify
  --against <selection>`, writes nothing and stops when the choice or a selected run's bytes
  moved since the selection was written.
- **A device in every stamp, and a new device said out loud.** An index (schema
  `meep_gpu.cert_chunks/2`; the first schema is refused by name) names its device (`cc86`, a
  GPU architecture on Metal) and whether it is new (`partition --new-device`); its stamps are
  `<round>_<device>_c<k>`, so two cards' chunks of one round never share or extend a root.
  `legs.py check` and `round.py` refuse a capability not spelled `major.minor` by name, refuse
  a gate stage on a capability no table admits unless `--new-capability` says so, refuse the
  flag on a certified capability and on the route stage, and refuse an index of another card
  or mode; a new capability runs chunked. Its bind
  adds records beside the certified card's and re-points no board, and
  `repoint_board_bindings.py` refuses a run of another device, or any run while a device
  cannot be read, unless `--device` names it. Every capability's licence lands in
  `dispatch_end_to_end_<stamp>_cc<NN>`.
- **The MEEP corpus from the reference environment.** The gate stage names the corpus once:
  `--corpus-root`, else the root the reference build record of `--python`'s environment
  names (`share/meep-gpu-build/reference_build.json`, `corpus`, the record `metal_round.py`
  reads), checks the tree against that record's sha256 list, and exports it to every lane as
  `MEEP_GPU_CORPUS_ROOT`; the gates' own lookup chain is unchanged.
- **The Metal round** (`rounds/1.0/metal_round.py`) takes `--new-architecture` the same way
  (an index of this Mac's architecture and mode, held to the Metal ledger both ways; refused
  without a chunk and on the route stage), reads this Mac's architecture at the fleet stage
  as well as the preflight, runs an unchunked round only on an architecture the ledger
  admits, reads each gate's status under the newest launch that ran it, and takes the corpus
  rule from `legs.py`, one home for both drivers.
- **Tests.** `parity/meep_gpu/test_round_chunks.py`,
  `parity/meep_gpu/test_round_new_capability.py` and
  `parity/meep_gpu/test_metal_round_new_architecture.py`.

### The Linux reference environment's script, run end to end, installs the pip lock one pin at a time and reads NVRTC's version with every device hidden

`parity/meep_gpu/build_meep_133_linux.sh` was run end to end on the NVIDIA host and stopped
on defects of its own, fixed here; rebuilt from the fixed script, the environment passed every
step and every acceptance check.

- **The pip lock is installed one pin at a time.** pip holds every wheel it downloads until
  the command exits, so installing the whole lock at once held its 26 wheels (3.1 GB, 0.9 GB
  of them PyTorch's) beside the 5.4 GB they unpack to, on a disk with little room above its
  floor. `build_meep_133_record.py pip-lock-parts` writes one requirement file per pin, each
  with the lock's own option lines (`--index-url`, `--only-binary :all:`,
  `--require-hashes`), in the lock's order; each is installed with `--no-deps`, so nothing
  resolves, and `pip check` and `--require-pip-lock` still judge the installed set whole.
- **The same two defects as the macOS script:** the environment's Python ships
  `lib/python3.1` as a link to `lib/python3.10`, so `_meep` and the ffmpeg executable were
  found twice (a linked directory is now skipped; `build_meep_133_record.py`'s
  `site_packages()` is the same code as the macOS entry's); and the GDSII check now prints the
  layers as plain integers.
- **NVRTC's version is read from NVRTC itself.** The build hides every GPU
  (`CUDA_VISIBLE_DEVICES=`), and CuPy's `cupy.cuda.nvrtc.getVersion()` asks the CUDA runtime
  for its version first, which raises `cudaErrorNoDevice` with no device visible. Step 8 now
  calls `nvrtcVersion` on `<prefix>/lib/libnvrtc.so.11.2`, the library CuPy opens on this
  lock, through `ctypes`; a library that does not load or a non-zero status is refused by
  name. CuPy's and Triton's versions are still asserted.
- Tests: 15 new cases in `test_reference_environment.py`: the pip lock's parts (one pin each,
  in order, on both locks; a lock they cannot split exactly is refused; step 7 installs
  every pin once), the Linux layout's linked Python directory, the GDSII layers under
  NumPy 2, and NVRTC's version with every device hidden. Each fix reverted alone fails its
  test.

### Timing is modular: the release round times one row at a time, and every table and record is cut from the newest valid row of each key

CERT-8 follows the certification's modular pattern (owner rulings R25 to R27). Its unit is
a ROW, any subset of rows can be timed again at any time, each row's record is
self-contained, and the identical-case table, the fused-product report and the
dispatch-preference records are cut by tool from the newest valid row of each key. The
procedure the 1.0 round ran from lane scripts (the variance gate, the idle-box watcher, the
fused DRIVE and sweep order, the keep-cache mint, the cut order, the import-record hook)
is in the release: `parity/meep_gpu/rounds/1.0/cert8.py` runs the rows,
`parity/meep_gpu/timing_rows.py` keeps, validates and cuts them, and
`docs/development/timing.md` ("The release round, one row at a time") is the manual. No
timing ran for this entry; every behaviour below is a host test on fixtures.

- **Rows and keys.** `ladder|<architecture>|<case>|<cells>|<tier>` is one case and size of
  the identical-case ladder with every leg it is compared against (GPU routes, array
  control, MEEP legs) from one ladder session (R5); `fused|<architecture>|<table>|<case>|
  <size>|p<pass>` is one fused-product bench unit; `check|<key>` is a dispatch check of
  another key. A record holds its session, its command, every process's import record,
  the launch counters of every GPU leg, the reference environment's build record (R27),
  the host gates it passed (the quiet window read twice, the variance precondition, the
  watcher's first and last readings and its intrusions, power recorded and never gated),
  its legs' medians with their gate status (R20), and a digest of its body. The store is
  written once per key and attempt.
- **Admission is evidence.** A row is cut only when its record shows the unit opened on a
  quiet window of two clear readings and watched while it ran (a row collected by hand
  has neither), with no intrusion (R18: other work of the account includes a busy
  build, compiler or bulk tool, and on Linux any busy process of the account, each
  measured by the CPU time used between readings); when every ladder entry of it was
  written inside its own session, a ladder never being resumed across sessions (R5);
  and when its ladder unit exited zero. A ratio enters a table only from a session whose
  variance precondition read GO.
- **One environment per cut (R27).** The environment is no file of the import record,
  so every selecting command takes `--environment-record`: a row of another reference
  environment is then stale and owed a re-time, and `cert8.py run --owed` re-times it.
  A table, record or report whose selected rows ran in two environments, or loaded two
  versions of one library, is refused by name.
- **The import record is the benches' own.** `bench_fused_products.py` and
  `bench_meep_identical_case.py` stamp every row with `imported_source_sha256`
  (`gate_provenance.provenance()`, the key rule every gate artifact uses); the external
  exit hook behind the 1.0 round's "169 of 169" is no longer needed. Beside it a
  fused-product row records what the hook and its joiner kept: the ledgers its tables
  read, the libraries it loaded and the switches of its environment.
- **What a change re-times.** A moved file outside the kernel table directories, a
  fused-pair emitter or the shared fusion block, a table's package init or arm table, or a
  route-record pin no arm's entry claims re-times every row that loaded it: in the 1.0
  round's 150 NVIDIA import records the planners, `deposit_repair.py`, `host_writes.py`,
  `fused_pairs.py` and `fused_polarization_pair.py` were in every row of both tables (67
  CUDA rows, 83 Triton rows). An arm's kernel file re-times only the keys whose launch
  counters show that arm dispatched; the arm's files are its ledger entry's kernel pins
  and what they import. Of the 83 distinct (table, arm, gate) triples the 1.0 rows
  launched, 56 resolve to a ledger entry that pins a kernel file; the 27 others, all
  Triton single arms citing a recertification sweep, take their table's whole kernel
  slice; and through what its pinned modules import, each of the 61 Metal arm entries
  that pins a kernel file reaches 90 of the 92 Metal kernel modules, so the Metal table
  re-times per table. A kernel file no
  launched arm carries owes the row a dispatch check, cut on the release tree, before the
  row is valid again. `timing_rows.py bill FILE` says it before a change lands: over the
  113 of 124 1.0 fused units that carry their plans (11 are `ERROR` rows), a change to the
  CUDA complex fused electric pair re-times 5 and owes 5 a check (103 untouched), the
  fused magnetic pair 33 and 43 (37), `step_curl_kernels.py` 51 and 61 (1),
  `fused_pairs.py` or `driver.py` all 113. Read against this tree, 113 of 113 of those
  rows are stale (6 step-path files moved: `fastpath.py`, `fastpath_cuda.py`,
  `from_meep.py`, `readback.py`, `sources.py`, the bench), so the whole of CERT-8 is owed
  on this freeze.
- **The records** (`cut_timing_record.py`, `meep_gpu/dispatch_preference.py`, which no
  gate's run closure holds: 0 of 250). A record cut from part of a campaign names every
  released (product, case) it lacks under `missing`, with the rows' own reason, instead
  of failing the release check (CERT-8 H1); the release check accepts a product named
  there. One record per table per architecture, ADDED beside the others and never
  replacing one: `--supersede` is removed, and a record cut on another device is refused
  with no override. The Metal table's record is named for its Apple GPU family
  (`metal_kernels/timing_applegpu_g13s.json`), and each table refuses the other's
  spelling. The shipped 2026-09-21 records are unchanged and recompute as before.
- **Owner rulings.** Every ladder unit, the variance precondition and every dispatch
  check pass `--require-ac-power 0`. MEEP's pass spread, and its repeat spread in the
  variance precondition, are recorded and never gated; a MEEP leg is struck only for
  what is not spread. Every median in a cut prints with its gate status.
- **The catalogs** `rounds/1.0/cert8_catalog_nvidia.json` (compute capability 8.6:
  3 ladder rows, the release's DRIVE cases per table, the 50-unit sweep) and
  `cert8_catalog_apple.json` (`applegpu_g13s`: 5 ladder rows) name the 1.0 round's rows;
  a catalog table no valid row reaches is named as not cut.
- **Tests**: `test_timing_rows.py` (38), `rounds/1.0/test_cert8.py` (20), and new cases in
  `test_cut_timing_record.py` and `test_dispatch_preference.py`: re-time 1 of 3 rows and
  re-cut; a partial set that names its missing key; an arm change re-timing 1 of 2 keys
  and owing the other a dispatch check; a second architecture added beside the first and
  a record of another device refused; every ladder unit accepted by the release ladder's
  `--dry-run`; a row refused for a ladder entry written outside its session, for a
  missing quiet window or watcher, for another environment, and for a standalone array
  control with no import record. On this tree: 564 passed, 2 skipped and 2 failed of the
  568 tests of the timing and record suites, the 2 being the declared-pending
  `test_at_the_top_of_the_sweep_the_two_pair_plans_are_faster[cuda|triton]`.
- **Not changed:** `timing_ladder.py`, and `build_identical_case_table.py`, which still
  reads the 1.0 round's ladders by its own rules.

### The 1.0 round's inputs are one derived manifest, staged by one tool on both machines

`parity/meep_gpu/rounds/1.0/` now names every record the round reads from
`parity/meep_gpu/results/` (`INPUT_FILES.txt`, `INPUT_DIRS.txt`, `INPUTS_SHA256SUMS`,
`INPUT_BUNDLE.sha256`), and `round_inputs.py` derives, checks, packs, verifies and unpacks
them, on the GPU host and on the Mac alike, keeping the guarantees of the Metal round's tool
(a deterministic bundle, digest checks, no write through a link, a present file that differs
refused, every refusal before the first write). The list is derived, not typed: the
module-level names the round's gates and kits bind to a record, with the files each reader
opens there; the seven corpus legs the corpus manifest is derived from (the manifest itself is
re-derived on the staging host, byte for byte); and the Tier B record with the row files the
five L3 legs verify. `derive` first audits the static run closures of the 206 scripts the
round runs: every binding of, or path to, a record in their 238 harness modules is a reader or
a named non-input, and `test_round_inputs.py` holds the manifest to the code both ways.

- **1,782 files in 30 directories** (602,086,047 bytes): 53 code-read files in 29 directories
  and the Tier B record with 1,728 row files (192 rows in each of `linux_keep`, `linux_flush`
  and `mac_flush`; no leg judges `mac_keep`), as the Tier B record re-minted in the R27
  reference environments names them (`ex.ring_gds` joins each set).
- **No longer staged, because no 1.0 stage reads them**: the complex-multiply expansion records
  (three directories of the 0.9.x NVIDIA bundle and five beyond-bundle files; the licence was
  deleted for 1.0), the signed-zero probe the 2026-10-04 Metal bundle carried, and the census
  and seam directories' per-row records and logs. Of the 992 files of the 0.9.x NVIDIA bundle
  27 are read, and of the 259 files staged for the Triton H->D gates 4.
- The 0.9.x round directories and `build_metal_round_inputs.py` are unchanged.

### The round's child environments: the user site off, switches scrubbed by prefix, the L3 corpus named

The 1.0 round's NVIDIA drivers ran their children with the user site directory on, so a
distribution under `~/.local` could sit ahead of the reference environment's own, and
removed package switches by a fixed list, which missed `MEEP_GPU_SUBNORMAL_INSTALL` and
every switch added since. `parity/meep_gpu/rounds/1.0/run_lane.py` now removes every
`MEEP_GPU_*` and `TRIDENT_FDTD_*` variable by prefix, sets `PYTHONNOUSERSITE=1` in every
child and in the licence legs' `env -i` environment, and stops before it takes a GPU when
its interpreter still enables the user site directory under either; its toolchain step
refuses it again on the card. `fleet_launch.py` refuses an interpreter that enables it and
an exported package switch other than the corpus root, and starts every fleet gate with
`PYTHONNOUSERSITE=1` and `CUPY_ACCELERATORS` empty. The licence legs keep running as a
user's process, without `CUPY_ACCELERATORS`, which the licence verdict refuses by name.
`l3_corpus.py` names its corpus: `--corpus-root` (the MEEP 1.33.0 tarball's `python/`
tree) replaces `--corpus-source`, and without it the leg takes the corpus the reference
environment's build record names (`reference_build.json`, `corpus`), verifies the tree
against the record's file list by the round's one rule (`legs.verify_corpus`: every listed
file at its sha256, no unlisted file Python could import),
and exports it as `MEEP_GPU_CORPUS_ROOT` with its parent as
`MGPU_SITE_MEEP_SOURCE`; a leg with neither is refused by name. `timing_ladder.py`'s
`--require-ac-power` defaults to 0: no timing run is gated on the power source, which is
still recorded. `RUNBOOK.md` section 3 tabulates every child environment of the round,
the CERT-4 records, censuses and boards in the reference environment among them;
`docs/development/timing.md` states each row's subnormal policy (NVIDIA times `keep`
only, and the dispatch-preference records are cut from `keep` rows) and that no reference
environment carries the native MEEP control; `docs/development/testing.md` gives the
suites' environment on an NVIDIA host. None of these files is in a gate's run closure
(`import_closure.py bill`: 0 of 250 each) or pinned by a ledger; `timing_ladder.py` is in
the coverage board's subject, so its new default moves that subject.

### One reference environment per platform, built end to end by the repository's script

`parity/meep_gpu/build_meep_133_macos.sh` and `build_meep_133_linux.sh` now build the one
environment the tests, a certification round and the timing run in, from the locks and the
MEEP 1.33.0 tarball, with nothing from the calling shell or the host beyond three pieces:
the GPU driver, the macOS SDK and glibc ([environments/locks/README.md](environments/locks/README.md)).

- **What the locks lacked, they hold** (`environments/locks/additions.json`), found by an
  inventory of every import and every program the package, its tests, the round, L3, the
  timing harness and the 194 corpus rows use: libgdsii 0.21, gdspy 1.6.13 and git (both
  platforms) and binutils 2.46.1 (Linux) in the conda locks, llvmlite 0.47.0 and
  imageio-ffmpeg 0.6.0 in the pip locks. Adding them moved none of the locked packages:
  141 = 138 + 3 (Apple silicon) and 147 = 143 + 4 (Linux) conda packages, 0 moved,
  0 dropped; 17 = 15 + 2 and 26 = 24 + 2 PyPI pins.
- **`environments/locks/recut_locks.py`** writes the locks: it solves every lock line
  pinned exactly with the additions, refuses by name a solve that moves, drops or adds a
  package, and reproduces the committed files byte for byte (`--check`, both platforms).
- **MEEP is built with libGDSII and without contraction on both platforms**
  (`-O2 -ffp-contract=off -fno-fast-math`, and `-march=x86-64 -mtune=generic` on Linux);
  the build stops unless configure found libGDSII and MEEP reads a GDSII file. libgdsii
  joins the judged set of each platform, which becomes 30 (Apple silicon) and 37 (Linux).
- **The corpus is in the environment**: the tarball's `python/examples` and `python/tests`,
  kept before the build in `<prefix>/share/meep-gpu-build/meep-1.33.0/python` with the
  SHA-256 of every file, beside `reference_build.json`, which names the build's record:
  the tarball's digest, the flags, the census, the locks' digests, the drift verdict and
  the identity verdicts in keep and flush.
- **ffmpeg is the imageio-ffmpeg pin's executable**, linked into `<prefix>/bin`; the build
  checks that `libmeep` and `_meep` keep their load commands. The drift tool holds ffmpeg
  only when it is that executable, and refuses the retired `corpus_only.host_tools` entry by
  name.
- **The scripts clear the environment themselves** and refuse the retired knobs by name:
  `MEEP_SIGMA_PATCH` (no reference build carries the sigma-reader patch),
  `MEEP_REFERENCE_SOLVE`, `MEEP_BUILD_ROLE` (a timing control is `--timing-control FLAGS`)
  and `MEEP_FFMPEG`. **`parity/meep_gpu/build_meep_133_identity.sh` is deleted**: its
  cleared environment and flags are the platform scripts' own now.
  `build_meep_133_native.sh` stays, as the timing control.
- The macOS script has been run end to end, and stopped twice on defects of its own, fixed
  here: conda-forge's Python ships `lib/python3.1` as a link to `lib/python3.12`, so a glob
  over `lib/python3.*` found `_meep` twice and the imageio-ffmpeg executable twice (a
  `lib/python3.*` directory that is a link is now skipped, in the script and in
  `build_meep_133_record.py`); and under NumPy 2 `get_GDSII_layers` printed its int32
  layers as `np.int32(...)`, so the GDSII check refused a file MEEP had read (the layers are
  now printed as plain integers). The Linux script's end-to-end run and its fixes are in
  the section on the Linux reference environment's script. The round re-runs in the
  environments these scripts build.

### Stale texts in the gates' closures are reworded for the sigma reader's retirement

Comments and docstrings only, in files that gate and probe closures hold, so that they land
before the round's closures are declared final: `meep_gpu/metal_dispatch.py` (the comment on
`METAL_OFF_DIAGONAL_CORNERS`) and `parity/meep_gpu/gate_dispatch_fused_route.py` (two
docstrings and the comment on `folded_offdiag_dispersive_2d`) no longer send the reader to a
MEEP built with `MEEP_SIGMA_PATCH=1`: stock MEEP 1.33.0 cannot lift those cases, and no
reference environment builds a MEEP that could. `parity/meep_gpu/survey_meep_examples.py` no
longer calls the side environment of the earlier surveys a sigma-reader environment (it
carried packages, not the reader), and `parity/meep_gpu/survey_meep_tests.py` points at the
locks for what the reference environments hold instead of listing it. No executable line
changes.

### The Metal lift children run under the round's interpreter

Nine Metal gates re-lift corpus rows in child processes (`gate_metal_fused_hd_pair.py`,
five siblings with their own `lift_basis`, and the three on `hd_tail_gate_common.py`).
Each child ran under the interpreter the 2026-09-04 census recorded for its row: an absolute
path on the machine that cut the census. There, 145 of 145 children ran outside the round's
environment; on any other Mac the leg died with `FileNotFoundError`. The new
`parity/meep_gpu/lift_host.py` sets the rules:

- **Interpreter.** Every child runs under the gate's own interpreter (`metal_round.py
  --python`). The recorded path is kept on each row as `recorded_interpreter`, provenance
  only.
- **Inputs.** A lift basis reads four files that do not ship: the census's three row files
  and the seam record. Each is now required by name. Without `tests_param_matched.jsonl`
  the basis used to shrink from 145 rows to 130 with no error. The complex gate
  (`gate_metal_complex_fused_hd_pair.py`), which drives its cell's rows in process, reads
  the same four and requires them too; its cell used to shrink from 17 rows to 12.
- **Needs.** 5 of the 145 rows need more than MEEP and the package, declared by their census
  lift record for every row of that record: gdspy and a libGDSII MEEP (2 rows), and
  PyMieScatt, nlopt and autograd (3 rows). A gate probes those needs under its own interpreter. A row whose need is absent
  is named `unlifted_on_this_host` and counted out of the leg's denominator, as the CUDA
  gates do; a leg that asks about no row at all fails, and any other row that fails is
  still a hole.

`rounds/1.0/metal_round.py` changes in three ways:

- Its preflight runs each lifting gate's own basis in one child under `--python` (torch
  unimportable, `--only` respected), and checks the complex gate's four inputs in the same
  child. It stops for an absent input, naming the gate and the file, and lists rows
  unlifted on this host without stopping.
- It exports `MEEP_GPU_CORPUS_ROOT` as the corpus the reference environment's build keeps
  (the `corpus` entry of `<prefix>/share/meep-gpu-build/reference_build.json`), or as
  `--corpus-root`, after checking every file's sha256 against the list that entry pins and
  refusing an unlisted importable file under the corpus root.
- `all`, and `status` on its own, exit with the status stage's code, not 0.

The docs no longer point a Mac at the retired 0.9.2 driver or `ALLOW_NO_LIFT`. The changed
files enter the run closures of 10 of the 250 gate and probe scripts: the nine lifting
gates and the complex gate. Seven uncited Metal welds pin a changed gate script and rebind
from the round's new stamps. No kernel, predicate or cited weld changes.

### The susceptibility-sigma reader patch leaves the package

Owner ruling R27 (2026-10-14): no reference build carries the MEEP source patch that added
`fields.get_susceptibility_sigma`, so the package's branch for it is removed. In
`meep_gpu/from_meep.py`, `_has_sigma_reader`, the reader route
(`_sigma_volumes_through_the_reader`) and `SIGMA_ROUTE_READER` are deleted outright, with no
alias: a structured dispersive lift takes the declared-geometry lookup and falls back to
the chi1inv inversion on every MEEP, and `sigma_lift.route` is `"declared-geometry lookup"`
or `"chi1inv inversion"`. Deleted with it: `parity/meep_gpu/meep-sigma-reader.patch`,
`parity/meep_gpu/validate_sigma_reader.py`, the "optional sigma-reader patch" line of
`tools/check_install.py`, and the four test functions (11 cases) that ran only on a patched
MEEP (on stock MEEP they skipped). Five refusals that sent the user to a patched MEEP (any
nonzero D_conductivity in a cell whose media differ in their E_susceptibilities, a
non-diagonal chi1inv in the lookup and in the inversion, a recovered gain, a singular
recovery basis) now end with running the simulation on CPU MEEP, and the structured-media
gate's refusal no longer names the reader. The two non-diagonal refusals also name what
writes the off-diagonal rows (subpixel averaging at any curved or rotated interface in the
cell, a nonzero `epsilon_offdiag`, a cylindrical cell) and offer `eps_averaging=False` only
for the first. `meep_gpu/test_sigma_lookup.py` gains two tests (6 cases) that pin the
non-diagonal refusal on both routes, the `eps_averaging=False` escape working, and a declared
`epsilon_offdiag` that the escape does not clear; the conductivity and singular-basis tests
pin the new ends of those refusals, and `meep_gpu/test_sigma_recovery.py` gains a test with
no MEEP that pins the recovered-gain clamp and refusal; and the
`sigma_recovery_scale_division_dropped` mutation of `parity/meep_gpu/mutate_from_meep.py`
selects the renamed end-to-end inversion test.

- **Stock MEEP behaves as before.** Without the patch the removed branch is unreachable:
  `fields.get_susceptibility_sigma` exists only on a patched build, and only the macOS build
  script could make one. Measured on Apple silicon with a digest of every array word and
  every scalar of each lifted driver (wall-clock timings excluded), before and after, on the
  coverage census's corpus rows (MEEP 1.33.0, single precision). Under the lock
  environment's own MEEP build, 192 / 192 lifted rows have equal digests (examples
  58 / 58, tests 110 / 110, the `parameterized` modules 24 / 24) and 27 / 27 rows that do not
  lift give the same outcome (2 need a GDSII build, 24 sit in modules that import
  `parameterized`, which the shimmed leg lifts, 1 is refused). Under MEEP built without
  contraction, 191 / 192 in one pass; the one other row (`test_wvg_src.py`, its
  eigenmode-source amplitudes) gave the earlier digest in 3 / 3 repeats on each tree. The two
  GDSII rows match 2 / 2 under a GDSII build. Of the lifted rows, 12 take the lookup and none
  the inversion.
- **Still refused on stock MEEP 1.33.0**, as before this change: a cell whose media differ in
  their `E_susceptibilities` and that holds any curved or rotated interface under
  `eps_averaging=True`, whether or not the shape it bounds is dispersive (the check covers
  every lattice point of the cell), or a medium with a nonzero `epsilon_offdiag`; and such a
  cell with any nonzero `D_conductivity`. The three cells of the 2026-08-04 retirement
  assessment (a dispersive sphere, in 2-D and in 3-D, and a rotated dispersive block under
  `eps_averaging=True`, and a conductivity beside differing susceptibilities: 4 / 4 cells)
  are refused before and after this change under both builds, the three non-diagonal ones
  at the same lattice point and entry, with only the refusal's wording changed.
  `eps_averaging=False` lifts the smoothed kind.
- **Not a capability loss on any reference environment**: the three corpus rows that were
  lifted in the patched environment (`mie_scattering.py`, `differential_cross_section.py`,
  `waveguide_crossing.py`) take no sigma route on any build; they needed its packages.
- Stock MEEP 1.33.0 exposes the arrays the reader returned (`structure_chunk.chiP`,
  `susceptibility.next`, `susceptibility.sigma` through SWIG): reading them from the structure
  chunks reproduced the patched reader's words on 7 / 7 test cells and lifts the refused cells.
  That route is a 1.0.1 item; it is not in this change.
- Two texts in every gate's closure follow: the release reason of
  `cuda:dispersive off-diagonal polarization pair` (`meep_gpu/fastpath_cuda.py`) no longer
  sends the reader to a MEEP built with the patch (stock MEEP 1.33.0 cannot lift the case;
  run the simulation on CPU MEEP), and the comment on the `StructuredSigmaLift` export
  (`meep_gpu/__init__.py`) names the two routes that remain.

### The bit-parity coverage board is re-cut on series entries 01 to 153

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..153; entry 153 (the R22 freeze) moved both its subjects (the
route-stamp registry and the re-minted Tier A fixtures in the outcome subject, the pending list
in the board's), so its check refused. Only three of its generated files change (`board.json`,
`board.md`, `outcomes.json`); the generator is unchanged. Its test outcomes are the CUDA and
Triton test files' cases of the package suite (CPU-only shim, every marker, dispatch off) and
the Metal test files' cases of the device run (Apple GPU), both run on this tree's outcome
subject: 216 test files, 0 of the rows' test files not run, 12,137 cases. Per file they equal
entry 152's on 216 of 216 files. `dev_evidence.json` is carried unchanged: each of its 11
records is still on its live subject (module bytes, emitted text, array path).

- **No cell moves** (0 transitions against entry 152's board; N / D, refused rows in
  brackets): (a) the chunk-record select, CUDA 55 / 55 (+1), Triton 55 / 56 (+2), Metal
  60 / 60 (+4); (b) MEEP's A form, CUDA 57 / 57 (+8), Triton 61 / 61 (+7), Metal 68 / 68
  (+9); (c) host word identity, CUDA 18 / 18 (+47), Triton 27 / 29 (+39), Metal 29 / 29
  (+48).
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 56, Metal 0 / 74)
  until the certification round re-gates.
- **The array path against MEEP** reads 6 / 6 families identical (+4 refused) under keep
  and under flush, on the re-minted Tier A sets.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, as before.
- Of the board's 11,412 leaves 10 move, none a cell or a reason text: the two Tier A sets'
  mint times, the build and record times, the two reports' names and digests, and the two
  subject manifests. Outcome subject `7abf9fe4b3b34512` (1,143 files), board subject
  `908efba2141d580d` (1,154 files).

### The R22 1.0 freeze (CERT-1): the round's stamp is set and the identity ledger re-minted

The series is frozen for the 1.0 certification round that owner ruling R22 (2026-10-13)
orders: the export dry run of 1.0 found host names, account names and local paths in
certified bytes, the source of each is fixed in entries 141 to 152, the 1.0.1 items
are folded in (the bulk permittivity read, the route-stamp registry, the timing harness, the
deposit schedule, the reference environment's test-side fixes), and certification is re-run
for 1.0 on the bytes after this entry. The round runs in chunks (`rounds/1.0/RUNBOOK.md`,
"CERT-3, chunked"); the route stage after the bind is not chunked.

- **The round's stamp is `2026-10-14_10`**: `meep_gpu/route_stamps.py` reads
  `ROUND = "2026-10-14_10"` (one line), the round the chunked indices name
  (`rounds/1.0/chunks.py show`), so the gate chunks (`2026-10-14_10_c0` to `_c3`, `_m0` to
  `_m2`) and the route campaigns share it: `dispatch_fused_route_2026-10-14_10_cc86`,
  `dispatch_fused_route_cuda_2026-10-14_10_cc86`,
  `dispatch_metal_route_2026-10-14_10_applegpu_g13s` and the licence
  `dispatch_end_to_end_2026-10-14_10`. `route_stamps.SPENT` already lists every route
  campaign directory of an earlier round. No gate or probe holds the registry in its run
  closure (0 of 250), and with this entry's files counted as changing, `chunks.py ready`
  reads every chunk ready (4 of 4 NVIDIA, 3 of 3 Metal): the freeze moves no chunk's closure.
  `chunks.py check` reads CHECK OK on both indices. `tools/ci/pending_certification.txt`: the
  3 "Removed when" clauses that named an earlier round's campaigns name these, with one
  comment block; 0 ids added, 0 removed. `docs/development/certification.md`: the 5 Metal
  route examples.
- **`meep_identity.json` is re-minted by `record_meep_identity.py`**, because 3 of the 20
  files the mint hashes moved: `meep_gpu/sources.py` (entries 141 and 151),
  `meep_gpu/from_meep.py` and `meep_gpu/capsule.py` (entry 149). Both sets, 8 of 8 cases
  each, one invocation per set: keep on the Apple M1 Max against the contract-off MEEP 1.33.0
  single-precision reference, flush on x86-64 Linux against MEEP 1.33.0 single precision (its
  flush on import); 39,991,116 stored and 65,610,420 reader words equal to MEEP's in each,
  and the keep files byte-identical after the flush mint (32 of 32). Of the ledger's 1,899
  leaves 88 move and none is a digest, a word count or a verdict: times (36), the 16
  capsules' byte counts and sha256 (31), the three moved inputs' digests (10), the fixture
  bytes (1; 4,636,170 of 5,000,000) and the Tier A load reports' counts of compared scalars
  of the five structured-permittivity cases per set (10; 35 to 37 more each, because the
  capsule now carries the lift's read report, `capsule.LIFT_ATTRIBUTES`). Every row of both
  sets replayed with MEEP not importable holds: 16 of 16 on the Apple M1 Max (flush under FZ)
  and 16 of 16 on x86-64 Linux (flush under the MXCSR lever), 2,916 of 2,916 stored and
  reader digests per set on each host, every planted flip caught and placed. The Tier A test
  file reads 62 of 62 on the Mac and, NumPy only, 62 of 62 entering in keep and in flush on
  Linux. `docs/development/testing.md` states the new sizes.
- **The array-path oracle does not move**: `record_array_path_oracle.py --report` reads
  `real-v6`, fingerprint `1ab516279d74002a`, 20 of 20 case digests equal on the Apple M1 Max
  (NumPy 2.4.3) and on x86-64 Linux (NumPy 2.2.6), so `FIXTURE_SET` is not bumped. The kernel
  ledgers' `array_path_oracle` leaves, the `driver_dispatch` records' module provenance among
  them (`provenance_code_sha256`, which names `from_meep.py`, `sources.py` and the other
  array-path modules), are written by `record_array_path_oracle.py --write` after the round
  on these bytes (CERT-4 records, the pending block of that name), as at the earlier
  freezes: the tool's contract makes that write a certification event.
- **The freeze criteria.** L1′ and AP9 re-ran on these bytes, because 4 of the 29 files that
  decide them moved since entry 131 (`sources.py`, `from_meep.py`, `capsule.py`,
  `readback.py`). L1′ on one RTX A6000 under keep and flush: 123 case-rows per policy, 0 of
  26,302,894,838 words differing after every sub-step, the planted flip caught 123 of 123, the
  mixed-policy control differing 2,446,768 of 6,272,469 words; the hand references on CuPy 0
  of 197,902,068 words per policy; `test_cupy_array_path.py` 20 of 20 on the device. AP9:
  `validate_pml` 308 of 308 (44 skipped), `validate_no_pml` 112 of 112, `validate_fold` 8 of
  8, `validate_folded_complex` 17 of 17, `validate_folded_offdiag` 16 of 16 with refusals 13
  of 13. The Metal route gate's controls dry run reads CONTROLS DRY RUN OK against the last
  released shipped leg (42 of 42 cases, 0 moved, nothing to smoke). The hygiene guard reads
  CLEAN on the tracked files and the board's subject; its ledgers leg stays declared until the
  round's bind writes neutral labels. AP8's exit is not settled at the freeze: entries 149
  and 151 moved the lift and the deposit, so the corpus Tier C re-runs in full on all four
  references beside the round, on the tree after entry 151; this entry moves no byte a corpus
  row loads (no package module imports `route_stamps.py`, and Tier C reads no Tier A fixture).
  A row that does not hold where Table 2 holds stops the round before CERT-4.
- **Measured on these bytes**: the package suite under the CPU-only shim (Apple M1 Max, every
  marker, dispatch off) 54 failed, 14,972 passed, 1,042 skipped, and the 96 device-visible
  files on the Apple GPU 12 / 4,171 / 331. Against entry 151's reports every id reads the
  same outcome (16,068 of 16,068 package ids, 4,514 of 4,514 device ids), counting the one
  package id that differed in this run at its re-run: a negative control that runs a child
  pytest failed only because the run's temporary directory sat under a checkout whose own
  conftest pins the dispatch switch, which contained the leak the control plants; with the
  directory elsewhere its file passes 8 of 8, as on entry 151. Needle anchors 0 differences;
  212 mutation anchors, 0 stale. The coverage board is re-cut in the next entry.
- `PRE_CERTIFICATION_STATE.md` section 11 records the freeze.

### The bit-parity coverage board is re-cut on series entries 01 to 151

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..151; the previous cut was entry 135's, and the board's check had
refused since the entries after it moved its subject. Only three of its generated files change
(`board.json`, `board.md`, `outcomes.json`); the generator is unchanged. Its test outcomes are
the CUDA and Triton test files' cases of the package suite (CPU-only shim, every marker,
dispatch off) and the Metal test files' cases of the device run (Apple GPU), both run on this
tree's outcome subject: 216 test files, 0 of the rows' test files not run, 12,137 cases. Per
file they equal entry 135's on 205 of its 215 files; on 9 Triton test files a weld or record
check that failed at that cut now passes, `test_triton_weld_contract.py` trades one declared
failure for another (a digest-depth control), and the new `test_metal_silent_deposit.py`
passes 5. `dev_evidence.json` is carried unchanged: each of
its 11 records is still on its live subject (module bytes, emitted text, array path).

- **Cells** (N / D, refused rows in brackets): (a) the chunk-record select, CUDA 55 / 55
  (+1), Triton 55 / 56 (+2), Metal 60 / 60 (+4); (b) MEEP's A form, CUDA 57 / 57 (+8),
  Triton 61 / 61 (+7), Metal 68 / 68 (+9); (c) host word identity, CUDA 18 / 18 (+47),
  Triton **27 / 29** (+39; was 26 / 29: `folded_fused_pair`'s host test is no longer
  pending), Metal 29 / 29 (+48).
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 56, Metal 0 / 74)
  until the certification round re-gates. Triton's `forms` moves from refused to a gap in
  (d) and (e) under keep and flush: a ledger entry now pins it, which makes it measurable.
  Every byte-binding entry still records the array path `array_path_oracle/real-v1`; the
  live one is `real-v6`.
- **The array path against MEEP** reads 6 / 6 families identical (+4 refused) under keep
  and under flush.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, as before.
- 292 reason texts change with no state change: 286 drop "the module's bytes moved since
  its run", and 6 Triton texts count one pinning entry fewer. The outcome subject grows
  from 1,127 to 1,143 files and the board subject to 1,154: the phase's new modules and
  tests.

### A silent source table skips its deposit where MEEP's subtraction leaves every word

`sources._deposit` returns before forming anything when the current is bitwise
`(+0, +0)`, the array is the C-contiguous array the door accepts, and the table is
word-neutral (`_silent_deposit_is_identity`): `dt` finite and positive (or none), every
amplitude finite, `condinv` finite with its sign bit clear at the table's cells, and no
point forming a `-0` subtrahend (in the real plane a point whose `amp_re` has its sign bit
set and `amp_im` does not; in the imaginary plane of complex storage one where both have
it). Subtracting `+0` leaves every float32 word, both zeros included, so the skip stores
MEEP's words; every other silent table still subtracts, as MEEP's `step_source` does, and
turns a stored `-0` into `+0` where MEEP does. Under flush the skip also needs no
subnormal at the table's cells, and a flushing run stores none. It is the same rule on
every table: the NumPy and CuPy array paths, and the CUDA, Triton and Metal routes, whose
deposits all go through `_deposit`.

- **Why.** A pulsed source deposits on every step after its pulse ends, as MEEP's loop
  does. On the CuPy route that call is nine small launches and a few hundred
  microseconds of host time, and under a held Metal residency it is an upload and a door
  launch: it was a fixed cost the step carries at small sizes. The skip removes the whole
  call on a silent step of an unflagged table, and nothing on a step whose current is
  nonzero or on a flagged table.
- **The predicate is static per table.** Its verdicts are read once from host copies and
  cached under weak references to the table's amplitude and index arrays and the
  `condinv` volume, so a silent step forms nothing, adds no device read, keeps nothing
  alive, and keeps no state on a source, the driver or the fields (the capsule's object
  graph and the bench's freeze are unchanged). A conductivity replaced after binding is
  read again.
- **Not a threshold and not `last_time()`.** 0.9.2's `abs(current) < 1e-30` return also
  skipped a nonzero tail current, and `last_time()` is not MEEP's zero test; neither is
  used.
- **Tests.** `test_source_deposit.py`: the live deposit against the frozen always-deposit
  form on every sign class of amplitude, current, `dt` and `condinv` on both storages
  under keep and flush, skipping exactly the deposits whose every subtrahend is `+0`; the
  skip set against MEEP's statements in Python doubles; a conductive run past its pulse,
  stepped both ways, word for word at every step, with the deposit cells holding `-0` on
  silent steps and, under flush, no subnormal; every source class at a silent time on
  planted signed zeros; real `Fields` with a conductivity of either sign; an integrated
  source's silent dipole; a conductivity replaced after a skip; the door's refusal of a
  non-contiguous array kept; the same comparison on a CUDA device; and a must-catch
  battery of wrong silent paths (a skip on any silent current, a blanket `-0` to `+0`, the
  magnitude threshold, a sign-blind zero test, the real plane's predicate alone,
  `condinv`'s sign unread, amplitude finiteness unread, a planted wrong sign of zero, a
  `last_time()` predicate). `test_metal_silent_deposit.py`: on a held Metal residency, a
  skipped table launches no door and every device word equals the always-deposit form's,
  under the host's keep and flush, and a flagged table skipped as if unflagged is seen.

### Certification runs in chunks that never collide, and a bind refuses an artifact whose bytes moved

Per-kernel certification makes an artifact valid wherever every file its import record
names, and every curated pin of the entry it binds, is byte-equal in the release tree. The
round can therefore certify a chunk of gates on its own GPU as soon as that chunk's
closure is final, while other entries are still landing. New harness tooling makes that
safe; no package module, kernel, gate, needle or mutation anchor moves.

- **Chunks** (`parity/meep_gpu/rounds/1.0/chunks.py`). An index per machine names the
  round, read from the route-stamp registry, and per chunk its units (a whole CUDA campaign
  subdirectory, one Triton fleet gate, the composition step, a stencil family, the bit
  identity, an H->D gate or a Metal gate, both policies of a gate together) and its stamp
  `<round>_c<k>` (`_m<k>` on Metal, `v<n>` from a relaunch on). The default partition is
  four NVIDIA chunks balanced by the measured walls in `legs.py` and three Metal chunks run
  one after another on the one Mac GPU. Every root a chunk writes is a function of its
  stamp; `chunks.py check` refuses two chunks sharing a unit or a stamp, two stamps whose
  roots are equal or extend one another by `_`, and a foreign root a bind tool would read
  as a chunk's own.
- **Launch.** `round.py launch gates` and `metal_round.py` take `--chunk-index` and
  `--chunk`: the chunk's stamp, its units only, its own lanes home and launch lock, the
  cards it is given (`--gpu-ids`), and a launch record (`results/chunk_<stamp>/LAUNCH.json`:
  tip, tree, units) that a resume must match.
- **Readiness.** `chunks.py ready` prints the chunks whose closure (static run closures,
  argument files, ledger pins and, while the static reader misses reaches through the
  package root's `_LAZY` table, the previous round's import records) holds none of the
  changing files, and `chunks.py stale` the launched chunks whose closure moved since their
  own launch tip.
- **Binds across every chunk stamp.** `legs.py bind --chunks`, `chunks.py bind`,
  `rebind_metal_welds.py --stamps` and `repoint_board_bindings.py --stamps` find each
  artifact under the one current chunk stamp that holds it, and refuse a gate found under
  two. `chunks.py verify`, first in the printed commands, refuses by name every artifact
  whose import record names a file the release tree holds at other bytes or not at all, and
  an artifact with no record whose closure moved since its chunk's launch tip. The Metal and
  Triton rebinds, and the CUDA rebind in its `--campaign` mode, apply the same rule to every
  artifact they bind (`parity/meep_gpu/artifact_drift.py`), and so does every writer of a
  round record: `record_cuda_regate.py` and `record_cuda_ade_recut.py` (a subject `--declare`
  names aside), `recut_composition_records.py` (the bound and flush legs, apart from the files
  the entry pins, which it already compares, and every family of the re-certification), and
  `recut_driver_dispatch_record.py` (every route leg). The chunked NVIDIA bind ends as the
  unchunked one does, with `neutralize_ledger_labels.py`.
- **Tests.** `parity/meep_gpu/test_round_chunks.py`: the default partitions cover every unit
  once and together run exactly the unchunked round's legs; no two chunks collide, at eleven
  chunks and across a relaunch; a planted import makes a chunk not ready; a moved or deleted
  recorded file refuses an artifact in `verify`, in the three rebinds, in the board re-point
  and in each record writer above.
### The lift reads MEEP's stored permittivity in bulk, on the builds it was compared on

Owner rulings R21 and R22 (2026-10-13): the structured lift's permittivity read moves from
one MEEP call per Yee point per tensor entry to a bulk copy of the words MEEP stores, in 1.0.

- **The bulk read.** `_lift_epsilon_structured` copies each structure chunk's stored
  `chi1inv[c][d]` words for every point exactly one chunk owns with no symmetry transform and
  no periodic translation, where `fields::get_chi1inv` returns that chunk's word widened to
  double, or 1.0 / 0.0 for an array MEEP left NULL (monitor.cpp:166-185, 271-272). The chunks
  come from `structure.chunks`, `get_chunk_volumes` and `get_chunk_owners`;
  `offsetof(structure_chunk, chi1inv)` is measured on a throwaway `mp.structure_chunk` and
  each chunk pointer is checked by reading back its Courant number. Every other point (the
  unowned wall planes, the low plane of a periodic axis, a folded image) is read one at a
  time by `fields::get_chi1inv`, as before. `structure::dump` is not used. A sampled
  self-check compares copied words with `fields::get_chi1inv` bit for bit (every copied
  block's two corners and up to 256 seeded points per component); a refusal or a disagreement
  warns (`RuntimeWarning`) and re-reads every point one at a time, which is exact.
- **Only on the builds it was compared on.** `from_meep.BULK_CHI1INV_BUILDS` lists
  `(MEEP version, precision, platform.system(), platform.machine())`: MEEP 1.33.0 single
  precision on Darwin arm64 and on Linux x86_64. Any other build (another release, a
  double-precision build such as conda-forge's `pymeep`, another platform) reads every point
  one at a time and says which build it is. The self-check guards a listed build; it does
  not certify an unlisted one: a one-ulp change to one interior word goes unseen by it.
- **What a lift records and prints.** `driver.chi1inv_read` holds the path, the reason a bulk
  read was not taken, a restart, the build stamp, and per component the points copied, read
  one at a time and self-checked; a component the bulk path can own no point of names its
  shape (a cylindrical cell with no z extent, which MEEP builds one pixel thick, reads Er and
  Ep one point at a time). Unless `mp.verbosity` is 0 a lift prints flushed progress lines to
  stderr, each headed `meep_gpu: lift:`: `init_sim` and its time, each component's read (and
  a line every 10 s of a per-point read), a restart, the sigma and conductivity passes, the
  whole lift. `progress_cb` keeps its contract and never goes back: a restart continues the
  pass, and its total grows once by the planes already reported.
- **The docstrings.** `lift_simulation` states the measured cost in place of "about 0.9 us
  each" per point; `_lift_epsilon_structured` no longer says a mirror-folded grid refuses an
  off-diagonal tensor (`_check_structured_geometry` refuses nothing since the fold was
  measured exact).
- **Tests.** `meep_gpu/test_from_meep_bulk_chi1inv.py`: on six cells (a 3-D PML cell with an
  off-diagonal tensor over four chunks, metallic walls behind a non-vacuum background, a
  mirror fold on a periodic axis, a cylindrical cell, a dispersive background, a cylindrical
  cell with no z extent) the READ words of every component, as uint64, equal
  `fields::get_chi1inv` at every point and the installed arrays equal the per-point path's;
  the bulk path is the one taken, with the per-point-only shape named; a one-ulp word planted
  at a copied block's corner is caught by the self-check, warns, and the restart installs the
  per-point read while the progress sequence never goes back; an unlisted build never
  locates a chunk; a lift prints its progress lines by default and none at
  `mp.verbosity(0)`. The bulk-path tests skip by the declared resource `bulk_chi1inv_build`
  on a build the table does not list.
- **The capsule carries the read report.** `capsule.LIFT_ATTRIBUTES` lists `chi1inv_read`
  beside `sigma_lift`, so a capsule written from a lifted driver loads back with the report
  of how its lift read the permittivity (without it the loader refused the capsule: the
  stored driver carried an attribute the rebuilt one lacked).
- **Mutations.** `parity/meep_gpu/mutate_from_meep.py`: `offdiagonal_chi1inv_sign_flipped_on_ingest`
  is re-anchored on the coupling rows both read paths install (it survived the bulk read,
  which copies most words without reaching the per-point assignment); six bulk-read
  mutations are added (the off-diagonal sign flipped in the copy, the NULL default inverted,
  the off-diagonal read from the diagonal array, the self-check skipped, the bulk read taken
  on an unlisted build, a restart's progress started again), and two on the lift's progress
  lines (never printed; printed at `mp.verbosity(0)`), each caught by the new tests; three readings are declared equivalent beside them with the
  measurement (the slot index floored, the user-volume mask dropped, the cylindrical r = 0
  ownership dropped).
- **The import closure follows the package root's lazy names.** `import_closure.Resolver`
  read `from meep_gpu import lift_simulation` and `meep_gpu.lift_simulation(...)` as reaching
  nothing, because the root binds those names on first use (`_LAZY`), so `import_closure.py
  bill` under-counted what an edit of a lazily bound module voids. It now resolves each
  `_LAZY` name to its defining submodule, at import and in the run closure, with planted
  controls in `test_gate_import_closure.py`. `bill meep_gpu/from_meep.py` reads 38 of 250
  gate and probe scripts, against 7 before. Every module reached through a lazy name gains
  holders with it; on the tree that carries the route-stamp registry `fastpath.py` reads 100
  (98 before), `fastpath_cuda.py` 96 (90), `metal_dispatch.py` 96 (89), `sources.py` 143
  (139), `readback.py` and `stepping.py` 234 (233), and `route_stamps.py` and `capsule.py`
  stay at 0. Two scripts reach the lift and the driver through
  the root and are declared in the `driver` category: `gate_triton_folded_offdiag_fused_ade_chain.py`
  (its optional corpus-lift leg) and `probe_pml_intermediate_flush.py`. The first was the
  gate of a `rebind_triton_welds.RETIRED_UNIMPORTED_PINS` row (`driver.py` on
  `triton_folded_offdiag_fused_ade_chain_device_gate`), which the rule no longer licenses
  (`driver.py` is in that gate's run closure now) and which leaves the table; the weld was
  rebound without the pin in the 1.0 round, so nothing it pins moves. The generator retires
  0 pins on all three tables before and after.

Measured on these bytes (`from_meep.py` sha256 `83599d1f…`, on every leg below).
**The read, word for word.** Each row lifts one MEEP simulation and runs both readers on it in
one process, the per-point reader of the tree before this entry and the bulk read, comparing
every read word (the diagonal and both off-diagonal entries of each component, as uint64), the
lattice index lists and the installed volumes. The rows are the 112 structured rows of the
MEEP-identity corpus manifest (geometry, an epsilon function or a material function) and the
bench's 3-D PML cell at 110,592, 262,144, 512,000, 2,097,152 and 7,077,888 cells, 117 in all,
each build on its default read path (the build table decides; nothing forces the bulk read):

| build (MEEP 1.33.0, single precision) | rows that match | read words that differ | installed words that differ |
|---|---:|---:|---:|
| macOS arm64, the contract-off build (`-ffp-contract=off`) | 114 / 117 | 0 / 121,631,202 | 0 / 103,399,474 |
| macOS arm64, the reference lock build | 114 / 117 | 0 / 121,631,202 | 0 / 103,399,474 |
| Linux x86_64, the NVIDIA tables' reference build | 114 / 117 | 0 / 121,631,202 | 0 / 103,399,474 |
| Linux x86_64, a second build | 113 / 117 | 0 / 121,501,602 | 0 / 103,356,274 |

On every build every matched row's lift took the bulk path (its own record: no refusal, no
restart), and no chunk owned a point the user volume does not. Three matched rows on each build
(`ex.perturbation_theory`, `ex.ring-cyl`, `t.test_ring_cyl`) are cylindrical cells one pixel
thick in z, whose Er and Ep read every point one at a time with the shape named, as the tests
pin. The rows that are not matches compare nothing against the bulk path: `ex.coupler` and
`ex.ring_gds` cannot be built on any of the hosts (MEEP without GDSII, no `gdspy`; Table 2
refuses them by name), `ex.waveguide_crossing` needs `nlopt` on the second Linux build, and
`t.test_material_dispersion…with_user_material` is a one-cell grid the bulk path owns no point
of (its 9 words read one point at a time, and equal). Forced onto the bulk read, a
double-precision build of the same release on Linux x86_64, which the table does not list, also
reads 0 of 117,638,730 words differently over its 93 matching rows (22 rows cannot be built in
that environment and 1 stops in MEEP's GDSII conversion); its lifts still read one point at a
time until a build of it is listed.
**The read's cost** (both readers inside the same rows, other work running, not a timing of
record): the structured read at 7,077,888 cells took 182.8 s one point at a time and 13.3 s
with the bulk read on the Linux reference build (13.7x), and 164.5 s against 7.0 s on the macOS
reference lock build (23.4x); MEEP's own `init_sim` for that cell is 39.5 s and 28.3 s, which
is now most of a lift. The lift-level figures in `lift_simulation`'s docstring (13.76 s to
4.50 s at 512,000 cells, 159.02 s to 28.88 s at 7,077,888, one arm64 host) are the scoping
measurements of the same code; the paper's lift and break-even figures wait on the re-timed
campaign.
**Mutations** (`mutate_from_meep.py --only` on the Linux reference build's CPU, each leg in its
own copy of the tree, every module restored): every mutation anchored in `from_meep.py`, 113 of
the battery's 212 (0 stale anchors over the 212): 110 caught. The 3 that survive survive on the
tree before this entry too, measured in the same environment, and touch nothing this entry
moves: `rotated_block_treated_as_axis_aligned` mutates `_is_axis_aligned_block`, which nothing
calls on either tree, and `gaussian_fwidth_not_inverted` and `continuous_ramp_dropped` mutate
the source time-function fields of `_lift_source`. The other 99 mutate modules this entry does
not touch, whose inputs from the lift are the same words.
**Suites** (dispatch off, one rank, no `results/`), each read test id by test id against the
same run on the tree before this entry, which carries the route-stamp registry. The CI view (the
hosted job's two Pythons on the GPU host's CPU, without MEEP): of the 18,440 ids both trees run,
0 change outcome on Python 3.10 and 0 on 3.12; the entry adds 13 ids, 4 passed and 9 skipped by
the declared resource (the bulk-path tests, which need MEEP); the 77 failures of 16,887 tests
run are the same ids on both trees, so the entry adds no failure to `check_failures.py`'s count.
On the Linux reference build's CPU, with MEEP, 144 of the 397 test files (every package test
that reaches the lift, the capsule or the import closure, the ledger and record contracts, and
every harness test file but `rounds/1.0/test_l3_corpus.py`): 0 of 5,070 shared ids change
outcome, the 13 new ids pass, and the 64 that fail fail on both trees. On the macOS reference
lock build, `test_from_meep_bulk_chi1inv.py`, `test_from_meep.py`, `test_capsule.py`,
`test_meep_precision_notice.py`, `test_lazy_package.py`, `test_gate_import_closure.py`,
`test_rebind_unimported_pins.py`, `test_mutation_anchors.py` and `test_route_stamps.py`: 418
passed, 1 skipped by a declared resource, 0 failed of 419, and the gate-needle record reads 0
differences. On the tree of the release records (entry 140) with and without this entry,
`-m certification` over both test paths in the Linux reference environment fails the same 39
of 383, all 39 declared in that tree's `pending_certification.txt`.

**What it voids.** `from_meep.py` is an input of the certification records: the array-path
oracle's provenance (`driver_dispatch.array_path_oracle` of the three tables, already
listed as pending since the round), the identity ledger's `inputs_sha256` and
`provenance_code_sha256` (re-minted by `record_meep_identity.py`; `capsule.py` is a minted
input too), and the import records of the round's gates that lift (the record-based bill; the
static `bill` above lists 38 of 250 scripts for `from_meep.py` and 0 for `capsule.py`). It is
re-certified with the rest of the 1.0 round under R22.

### Two harness tests no longer depend on their host's git or CPU topology

- **`parity/meep_gpu/test_export_public.py` (22 tests).** Its fixture built the two
  repositories with `git init -b main`, which needs git 2.28 or newer, so on an older git
  every test errored at setup. The fixture now points `HEAD` at `main` by name
  (`git symbolic-ref`), which any git accepts. The export tool is unchanged.
- **`test_timing_ladder_plan.py::test_the_hwthread_control_without_a_topology_is_omitted_by_name`.**
  On a Linux host the ladder reads that host's own topology, which supplied the socket
  count the test means to withhold, so the hardware-thread control was kept and the test
  failed. It now withholds the topology, as `test_srun_launcher_and_its_flags` does.

### The suites pass in the NVIDIA reference environment: 429 test-side failures declared or fixed

The first run of the package and harness suites with CuPy, Triton and a CUDA device present
(the NVIDIA reference environment, one RTX A6000) read 429 failures that no pending entry
declared. None was a wrong answer from the package or a kernel: each was a test's assumption
about its host. This entry changes tests, the two `conftest.py` files,
`tools/ci/declared_resources.txt` and the resource table of `docs/development/testing.md`
only; no package module, kernel, gate, ledger, record, needle or mutation anchor moves.

- **Metal tests where PyTorch has no Apple GPU (344).** The NVIDIA environment installs
  PyTorch for Triton, so the Metal tests declared under `[torch]` ran there and failed for
  want of MPS. Their 228 entries move to `[mps_device]`, which needs PyTorch and a usable
  Apple GPU, with the bench-provenance test whose bracket only the Metal table composes.
  An entry moves only when every id it covers failed there; the 39 `[torch]` entries
  whose tests passed there (37) or already skip themselves by name for want of MPS (2)
  stay.
- **Two new resources, `cupy_absent` and `triton_absent` (31 ids).** Tests that exercise
  the arm a CuPy-free or Triton-free host takes (a refusal by name without the library,
  NumPy wearing CuPy's name through the real CUDA composer, a planner that refuses without
  Triton) skipped themselves with a bare skip where the library is installed, which the
  suite fails. The root `conftest.py` gains a probe for each, present when the library is
  NOT installed, and the tests are listed under them. Hosts without CuPy and Triton run
  them as before.
- **Host FPU state left by an earlier test (15).** MEEP's own initialization flushes
  subnormals on x86-64 and never undoes it, so after any in-process MEEP import a census of
  host subnormals read 0 and the hardware opt-in tests printed two different status lines.
  An opt-in fixture, `host_keeps_subnormals` (`meep_gpu/conftest.py`), and the opt-in
  suite's own fixture drive the host to keep for the test and put it back.
- **Fixtures that handed host arrays to device paths (17).** A kernel module an earlier
  test imported was bound to the real CuPy, so the stand-in's arrays were refused by name:
  the fixture now hides such modules for the test. The scratch-disjointness test builds
  its configuration on the device, and the curl tests fetch device volumes with `get()`.
- **`CUPY_CACHE_DIR` under a direct keep install (9).** A direct `keep` install with CuPy
  among its executors needs a policy-suffixed private cache, which dispatch derives and a
  direct install does not: those tests name one where CuPy is installed, and the policy
  suite starts each test with no cache pointer there.
- **Locale (9).** `open()` and a child's output decoded without an encoding after a CUDA
  context set the C locale: each now names UTF-8.
- **One host tool, one tolerance, one allocator, one stand-in (4).** The
  reference-environment test's corpus count follows the host's `/usr/bin/ffmpeg`, as the
  tool's own lookup does; the diagonal force test's decimation-1 bound is 2e-07 (MEEP's own
  value moves 9.61e-08 between the arm64 and x86-64 builds, this engine's 1.24e-08;
  measured 1.22e-07 at one thread and 1.45e-07 at two on Linux); the address-reuse control
  skips under the sanctioned name `freed_address_reuse` when the allocator declines; a
  Triton stand-in hides the real Triton's submodules and CuPy.

Measured in the NVIDIA reference environment (one RTX A6000; the package suite as the CI
view's ten shards and the harness as its two, every marker, skip enforcement as shipped,
dispatch off): 39 failed of 17,589 run, 39 / 39 declared in the pending list, 0
undeclared, where the release tip read 468 failed of 17,965 run and 429 undeclared. All
18,678 ids are common to the two runs; the 429 moved, 53 to passed and 376 to a skip by a
declared resource, and no other id changed outcome. The CI view, on Python 3.10 and 3.12:
18,247 / 18,247 ids keep their outcome on each, and 164 skips are now named
`[mps_device]` rather than `[torch]`. The Mac reference environment reads as before:
40 failed of 17,981 run, 40 / 40 declared, and all 18,516 ids it judges (11,788
package, 2,227 harness, 4,501 on the Apple GPU) keep their outcome.

No file this entry changes is in the run closure of any of the 250 gate and probe scripts
(`import_closure.py bill`), and no ledger pins one. Five are in the CUDA census's subject
(`meep_gpu/cuda_kernels/*.py`), which an earlier entry of this phase already moves; the
Triton and Metal census subjects do not move.

### The route stamps move to one registry that no gate imports

Each kernel table's driver-route stamp was a constant of its dispatch module
(`fastpath.DRIVER_ROUTE_FUSED_GATE`, `fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE`,
`metal_dispatch.METAL_DRIVER_ROUTE_GATE`), so a re-route, a one-line stamp edit, moved the
bytes of a module in the run closure of 98 (Triton), 90 (CUDA) or 89 (Metal) of the 250 gate
and probe scripts, and voided their evidence. The three stamps now live in
`meep_gpu/route_stamps.py`, which imports nothing of the package and which no package
module, gate or probe imports: `import_closure.py bill meep_gpu/route_stamps.py` reads 0 of
250, and `parity/meep_gpu/test_route_stamps.py` holds it there, with a planted import in the
planner as the control that must fail. A re-route edits only the registry; the route records
are voided by name, because each record's `runs[<device>].released_fused_arms.gate` must be
the campaign the registry spells, and a table re-routed alone (`REROUTED`) moves only its
own name.

- **The dispatch path no longer reads a stamp.** It read one only as a label: the run
  record's `fusion.driver_route_gate` and the "released by" clause of a refusal. Both now
  name the ledger slot that records the campaign, `fastpath.ROUTE_RECORD`,
  `fastpath_cuda.CUDA_ROUTE_RECORD` and `metal_dispatch.METAL_ROUTE_RECORD`
  (`<table>_kernels/fingerprints.json` at
  `driver_dispatch.runs[<device>].released_fused_arms.gate`). No decision changes. The
  stamps' histories move to the registry with them, verbatim.
- **The round's stamp is set once, at the freeze.** `route_stamps.ROUND` reads the
  placeholder `UNSET` until the freeze of the round that runs the campaigns sets it (one
  line, for example `ROUND = "2026-10-14_10"`), and the spent names of the rounds before it
  are in `route_stamps.SPENT`. `python -m meep_gpu.route_stamps` prints the names the stamps
  spell and exits non-zero while any is refused.
- **Every tool that names a campaign keys on the registry, by name.**
  `recut_driver_dispatch_record.py` refuses an unset stamp (`route_stamp_is_unset`), a
  campaign that has already run (`route_campaign_is_spent`), a run directory of another
  stamp (`run_is_not_this_capability_campaign`, `run_is_not_this_architecture_campaign`),
  and a dispatch-by-default licence, transcribed or carried forward, whose directory is not
  `route_stamps.licence_campaign` (`licence_is_for_another_campaign`): with the stamp out of
  the dispatch modules, a licence of another round can run the same bytes, and only its
  name tells them apart. The route stage of `legs.py check` (through
  `legs.route_campaign_names`) and `metal_round.py`'s route stage (through
  `metal_round.ROUTE_NAMES_PROGRAM`) refuse the same names, replacing their hard-coded
  refusal of 0.9.2's stamp; a gate stage records the refusal and goes on, since the stamp
  edit moves a file no gate holds. The NVIDIA route stage writes the licence under the
  registry's name.
- **The pending list** gains 13 entries: the record and weld-contract tests that walk the
  pins of the five ledger entries this change moves (the three `driver_dispatch` records,
  the Triton `bit_identity_gate` and one Metal weld). Measured on the Mac CPU over 30 test
  files on entry 140 and on this change: these 13 pass there and fail here, and no other id
  moves but the 42 new tests, all passing.

### Three round-tool fixes: the held sync_in control's text, the Triton H->D inputs, and a controls dry run

- **The Metal route gate's held `residency-sync_in-dropped` row says what it measured.**
  Under held residency the control patches `Residency.acquire_write` to leave the first
  claimed mirror clean, and reads no `verify()`; its state kept the per-launch path's
  `steps_with_stale_mirrors` at 0, which the gate read as "residency.verify() read EMPTY
  at the instant of every skipped sync_in". On the 1.0 round's held route campaign the
  row read NOT-APPLICABLE with that text on 42 / 42 dispatched cases on both dispatch
  legs, `suppressed` 0 on every row. The held state now carries only what that path
  measures (entries into `acquire_write`, those on a registered mirror, and the sparse
  doors' counts over the armed steps), and `held_sync_in_verdict` grades it on those
  fields: NOT-APPLICABLE when nothing was suppressed and the legs agree (the verdict of
  every row of that campaign is unchanged; its text now states the counts),
  HARNESS-FAILURE when nothing was suppressed and they differ, and NULL-DID-NOT-DIVERGE
  when a mirror was left unclaimed and nothing changed, which the old branch would have
  read as NOT-APPLICABLE. The held null still does not bite on the shipped compositions:
  their host writes go through the sparse doors, which move no ownership; making it
  bite is a new control, not a text fix. `parity/meep_gpu/gate_dispatch_metal_route.py`;
  `test_metal_route_held_sync_in_control.py` (new).
- **`legs.py check` refuses a gate stage whose Triton H->D lift inputs are absent.** The
  ten Triton H->D gates lift from a census and a seam record under
  `parity/meep_gpu/results` (`predicate_coverage_triton_2026-09-04_cylm0`,
  `h_to_d_seam_2026-09-04`), which a checkout does not carry. On the 1.0 round the census
  was not staged on the certification clone and the check passed; the first such leg ran
  every device leg and stopped at its lift 113 s into a leg priced at 3,052 s. The check
  now reads both names from the gates' own source (through their kits) and refuses, by
  name, an input directory that is absent or lacks a file the lift reads.
  `parity/meep_gpu/rounds/1.0/legs.py`, `RUNBOOK.md` section 3;
  `test_round_triton_hd_inputs.py` (new).
- **A pre-freeze dry run of the Metal route gate's residency controls.**
  `gate_dispatch_metal_route.py --controls-dry-run --out DIR [--dry-run-reference
  controls.jsonl]` lifts each drive case once, takes the gate's freeze step and arms each
  residency control in the gate's own schedule, then reports the plan's wrapper chains
  (refusing a wrapper `ROUTE_WRAPPER_RULINGS` does not name), each control's state and
  the verdict class it implies, every move against a previous campaign leg's rows, and
  `smoke_cases`: the cases on which a must-diverge control arms, which need the full gate
  before the freeze. It compares no bytes and writes no `gate.json`. The round's route
  campaign failed 2 of its 4 legs on a wrapper new in that round that the route smoke's 2
  cases did not hold; the dry run reads all 42. `test_metal_route_controls_dry_run.py`
  (new).

### The 0.9.2 release's changes from the development history reach the 1.0 line, and the release workflow publishes only from the public repository

This line branched from the development history at `fa59c2f`, before 0.9.2 was released.
The six commits that followed it there never reached this line: the 0.9.2 binds, the
release notes, a documentation pass, and the status notice with the publish workflow.
This entry carries them over, adapted to 1.0. The 0.9.2 binds stay behind, because the
1.0 round binds every ledger again.

- **0.9.2 is a release in this changelog.** The sections 0.9.2 shipped sat under
  Unreleased here. They now stand under their own `## 0.9.2 — … (2026-10-06)` heading,
  with its lead paragraph and "Pending: fused-product timing". The 0.9.2, 0.9.1 and
  0.9.0 sections now read byte for byte as in the public 0.9.2 release, including the
  NVIDIA note under the Metal section, the prepared H200 round, and the "As of 0.9.0"
  note over the Known limits.
- **Status.** README.md and the manual's landing page open with a status line: beta
  (1.0.0), the version the accompanying paper cites; interfaces and record formats may
  still change in later releases. The package classifier was already
  `Development Status :: 4 - Beta`, and `pyproject.toml` is unchanged.
- **`.github/workflows/publish.yml`**, the trusted-publishing workflow of 0.9.2, with both
  jobs guarded by `if: github.repository == 'Erised-AI-Inc/meep-gpu'`. A `v*` tag on the
  public repository builds the sdist and wheel and publishes them to PyPI. The
  development repository carries the same file and never publishes.
- **Sentences that hold in both repositories.** The H200 round's README, the
  certification page and the 0.9.2 notes say that the hand-off runbooks and the RTX
  A6000 round script are kept in the development repository, so the public export
  leaves those three files out and edits no document. The certification page says that
  the run records are not in this repository. It and the harness README say that commit
  hashes in the records name commits of the development history, which the public
  release repository does not carry, and that records bind files by sha256. The two
  H200 runbooks clone the development repository.
- **Neutral names in the timing fixtures.**
  `parity/meep_gpu/fixtures/timing/a6000_2026-09-28_b.params` and
  `ladder_2026-09-28_b_commands.json` take 0.9.2's one substitution, so their directory
  and environment names carry the `meep_gpu_` prefix.
- **The release-host placeholder in a verdict test.**
  `meep_gpu/cuda_kernels/test_complex_no_pml.py::test_the_recorded_verdict_is_the_artifact_on_disk`
  reads a record that holds `RELEASE_HOST_PLACEHOLDER` by the rule
  `test_certification_metadata.py` states for that placeholder: the artifact's legs must
  stamp exactly one hostname. Any other recorded host must still equal the stamped one.
  With the evidence archive present, the file read 2 failed and 61 passed before this
  change, and 1 failed and 62 passed after it: the verdict test now passes. The other
  failure, `test_the_recorded_kernel_digests_are_the_live_source`, fails before and
  after, because the admission record's kernel digests predate a change on this line to
  `step_B_no_pml_complex`. Without the archive, both tests skip by a declared resource,
  as before.
- **Versions and wording.** 1.0.0 in the citation text of the README and of the citing
  page, "Version 1.0" in INSTALL.md, the certification-suite row of AGENTS.md, and the
  two certification step names of the hosted test workflow.
- **Not carried over.** The 0.9.2 ledger binds and the pending-list removals that went
  with them. The Metal provenance rows and the board and census defaults that pointed at
  0.9.2 runs. The 0.9.2 version and date in `CITATION.cff`, which reads 1.0.0 since
  entry 137. The classifier `3 - Alpha`. The pages that state the 0.9.2 certification
  status: the 1.0 round's status is written by its own documentation entry. The neutral
  names in `parity/meep_gpu/timing_host.py` and its test, which the hygiene entry applies.
- **What it moves.** No ledger changes (0 of 4). None of the 16 changed files is pinned
  by a ledger (0 of 16), and none of the 250 gate and probe scripts holds any of them in
  its run closure (`import_closure.py bill`), so no gate's evidence moves. Four changed
  files are in the coverage board's outcome subject: the complex no-PML test file, the
  harness README and the two fixtures. So `build_parity_board.py check` refuses until
  the board is re-cut. The test file is also a subject of the hand-written CUDA census
  (`meep_gpu/cuda_kernels/*.py`), so the CUDA board refuses its census until that census
  is re-cut.

### The 1.0 round's records are committed: the four bound ledgers, the board bindings and the pending list

The certification round of 1.0 ran every gate of the three tables on the frozen commit
`5b9161c` (stamp `2026-10-13_10`; the two legs of the CUDA off-diagonal polarization-pair
gate re-ran on the tip after entry 138, with that gate's fix), the NVIDIA route campaigns
on that tip and the Metal route campaign on the tip after entry 139, and its records now
land in the tree. Nothing else moves: no package module, kernel, gate, census
tool, round driver, needle or mutation anchor.

- **The four ledgers.** `meep_gpu/cuda_kernels/fingerprints.json`,
  `meep_gpu/cuda_kernels/certification.json` and `meep_gpu/triton_kernels/fingerprints.json`
  as the bind on the GPU host wrote them, and `meep_gpu/metal_kernels/fingerprints.json` as
  the Mac's bind wrote it, each from the round's released gate runs. Then
  `recut_driver_dispatch_record.py` re-cut each table's `driver_dispatch` record once, from
  the released route campaigns `dispatch_fused_route_2026-10-13_10_cc86` (with the licence
  `dispatch_end_to_end_2026-10-13_10`), `dispatch_fused_route_cuda_2026-10-13_10_cc86` and
  `dispatch_metal_route_2026-10-13_20_applegpu_g13s` (the Metal route's own stamp, after the
  route gate's fix). Against the bind's ledgers, the re-cut moves `driver_dispatch` and
  nothing else: 58 leaves on Triton, 125 on CUDA and 12,470 on Metal (its first
  `runs["applegpu_g13s"]`); `certification.json` is the bind's byte for byte.
- **The board bindings.** `repoint_board_bindings.py triton|metal --stamp 2026-10-13_10
  --write` re-pointed the Triton board's `GATE_BOUND` (39 of 39 entries, the ten H->D gates
  included) and the Metal board's gate references (45 over 28 directories) to the round's
  gate roots, each re-hashing with 0 recorded files moved. The CUDA board binds through the
  ledgers. On the round's three censuses and the H->D probe re-cut on these bytes
  (`h_to_d_seam_2026-10-13_73d04e99`), the boards read served in dispatch 352 (CUDA), 334
  (Triton) and 327 (Metal) of 597 priced seam-instances, equal to 0.9.2's arm by arm, and
  served 538, 537 and 507 of 597.
- **The pending list.** 26 entries leave `tools/ci/pending_certification.txt`: 14 that the
  re-cut `driver_dispatch` records turn green and 12 that pass on the bound ledgers alone.
  Each passes on this tree in the evidence view and in the CI view (no `results/`), and no
  entry that stays passes in either. 38 stay: 37 fail on records, pins or text this round
  did not re-cut, and one reads the evidence archive and is not measured without its marker.
  3 entries are added, each red on a constant of its own test that names an earlier round's
  evidence. Two armed controls of the digest walk (the Triton weld contract's and the one
  `test_weld_record_walk.py` shares with the three contracts) plant a digest at depth 8,
  where the bound Triton ledger now holds 18 real ones: each passes on the tree before this
  entry and fails with it. The Metal nonlinear H->D test's `CELLS_BOARD` names the
  2026-10-02 board, whose cell binds a run no ledger since has welded: it runs only where
  the evidence archive's Metal board is present, and fails there before this entry as after.
  The list states what removes each.

The entry moves no certified byte except the ledgers themselves. No file it changes, other
than the four ledgers, is pinned by any entry of the four ledgers, before or after, and
every digest pin of the new ledgers reads the same hold or drift verdict on the tree before
this entry and on this tree. None of the 250 gate and probe scripts holds a changed
non-ledger file in its run closure (`import_closure.py bill`), no import record of the
round's artifacts names a changed file (904 payloads of the `2026-10-13_10` roots and 8 of
the Metal route's), and no freeze-criterion or Table 2 file moves. The coverage board is
not re-cut here; `build_parity_board.py check` refuses until the entry that re-cuts it.

### The Metal route gate's placement control looks through a spent start of H

The held Metal route campaign of the round (`dispatch_metal_route_2026-10-13_10_applegpu_g13s`)
released 2 of its 4 legs: on `shipped` and `harness_flush` the must-diverge control
`residency-inner-bracket-dropped` read NULL-DID-NOT-DIVERGE on `special_kz_2d` and
`bfast_1d`, while the dispatch itself held (44 of 44 cases passed, 0 divergent). The
control moves a wrapper's residency bracket from its inner launch to the whole slot, and
it armed on `launch.StartOfHPlan`, the start of H the composer puts around whatever
serves a run's first `update_H` (entry 134). That wrapper does host work only while
`fields.h_separated` is False, and the control arms after one step, so it was a
pass-through on every armed step and moving the bracket across it could not change a
byte. `parity/meep_gpu/gate_dispatch_metal_route.py` now judges such a slot by the
start's occupant, as the control read those two cases before entry 134
(NOT-APPLICABLE: the only bracketed wrapper left is the read-only deposit repair); a
start still owed when the control arms is armed on as before, and a start that runs
while looked through makes the row a HARNESS-FAILURE. The NULL text of this control
now names its own slots and wrappers, where it read the sync_in control's fields and
printed `None`, and the "upload was a no-op" branch is open only to the control that
measures the upload. `metal_dispatch.METAL_DRIVER_ROUTE_GATE` moves to
`dispatch_metal_route_2026-10-13_20`, so the held route is re-run on these bytes as
`dispatch_metal_route_2026-10-13_20_applegpu_g13s`; the gate roots of the round keep
their stamp `2026-10-13_10`. `parity/meep_gpu/test_metal_route_inner_bracket_control.py`
is new: 7 of its 8 tests fail on the bytes before this change, and the eighth pins the
case that does not change, a start still owed when the control arms.

### The Metal conductive PML presence-words test is declared under `[torch]`

The hosted CI view read one undeclared failure on the round's frozen commit, on all three
jobs: `meep_gpu/test_metal_conductive_pml.py::test_a_chunk_that_keeps_no_conductivity_is_served_through_the_presence_words`
builds a Metal conductive PML plan, whose chunk-bounds record is a `torch.int32` tensor,
and failed on `ModuleNotFoundError` where PyTorch is absent. It is now listed under
`[torch]` in `tools/ci/declared_resources.txt`, beside the three tests of its file that
build the same plan, so the CI view reads 0 undeclared failures on both Pythons; where
PyTorch is installed it runs exactly as before. This is the first entry after the frozen
commit of the round and moves no certified byte: neither changed file is pinned by any of
the four certification ledgers (the curated pins of the three `fingerprints.json` and
`cuda_kernels/certification.json`), and none of the 250 gate and probe scripts holds
either in its run closure (`import_closure.py bill`), so no gate's evidence moves. The
declared-resource list is part of the coverage board's outcome subject, so
`build_parity_board.py check` refuses until the board is next re-cut.

### Curated pins the eager package root left behind leave their welds by a generated row

Per-kernel certification (entry 128) makes a gate's import record its own dependency set,
but some welds' curated pin sets were cut from records the eager package root had filled
with the driver, the dispatch planner, the MEEP lift and their dependencies. The rebind
tools keep curated sets, so before this change the next bind could not hold them:
`rebind_metal_welds.py` skipped such a weld ("artifact does not record") and
`rebind_triton_welds.py` filled the pin from the checkout, which kept the weld coupled to
`driver.py`.

- **Each rebind tool carries `RETIRED_UNIMPORTED_PINS`, `(weld, path) -> gate`, generated by
  `import_closure.py unimported-pins <table>` and reviewed.** A row needs both halves of
  the rule: the file is a module the eager root loaded (`import_closure.eager_root_modules`,
  read off the root's `_SUBMODULES`), and it lies outside the run closure of the weld's gate.
  A weld's own test file and a kernel module its gate loads by computed name are outside
  the closure too and stay pinned. Rows: 31 Metal (`metal_complex_no_pml_conductive` 13 of
  43 pins, `metal_cylindrical_real_fused_magnetic_pair` 12 of 44,
  `metal_complex_conductive_fused_pair` 6 of 46, a declared driver gate that keeps
  `driver.py` and `fastpath.py`), 4 Triton (`driver.py` on
  `triton_cylindrical_fused_magnetic_pair`, `triton_folded_offdiag_fused_ade_chain` and the
  two stencil welds), 0 CUDA. A row acts only inside a rebind, reported
  `RETIRED-PIN (not imported by <gate>)`, every other pin compared as strictly as before; the
  run refuses, writing nothing, if a row's file is back in its gate's run closure, and an
  entry is skipped if its artifact records the file (whether or not the entry still pins
  it) or disagrees with the checkout on any file it recorded. `bit_identity_gate` needs no
  row: its binder writes the pin set from the run's own record. The deleted-file
  `RETIRED_PINS` of the Metal and CUDA tools are unchanged.
  `parity/meep_gpu/test_rebind_unimported_pins.py` holds each literal to the generator both
  ways (a row the generator no longer produces must belong to a weld rebound without the
  file, and still hold), keeps deliberate pins out of the tables, and runs the rule through
  each tool's `main()` on planted roots.
- **The pin history behind the rows**: each Metal weld's curated set equals, file for file,
  the whole import record of its gate's 2026-08-21 re-gate (43 / 44 / 46 files), cut while
  the root was eager; the four Triton welds were seeded by `seed_triton_welds.curated_for`,
  which pins a shared file "when the gate imported" it, and applied to post-R17 records the
  same rule gives each committed set less exactly `driver.py`.
- **The run closure reached a harness table's stored callables whenever the table's module
  was imported.** `metal_composition_matrix.MATRIX` stores a builder per row, one of which
  reaches `dispersion.py` through a function-level import, so the bound held `dispersion.py`
  for 22 gates that never name the table; `gate_metal_cylindrical_real_fused_magnetic_pair`'s
  record does not hold it, and without this its weld could not bind. In a harness file, a
  lambda or a function that a module-level tuple, list, set or dict display stores is now
  reached where reached code names the table (in its file, through a module alias or by a
  from-import); package files keep the old rule. 26 of 250 run closures shrink and none
  grows; the 250 import closures and the `fastpath.py`, `driver.py` and `from_meep.py` bills
  (98, 89 and 7 of 250 scripts) are unchanged, so `test_gate_import_closure.py` needs no
  edit; the fresh-interpreter import closures of e131 lie inside the bound (244 of 244 on the
  Mac, 250 of 250 on the Linux host); the 2026-10-12_10 round's records read 313 of 687
  inside the bound on its tree, as before, with the same five modules loaded by computed
  name.
- **Measured on post-R17 artifacts of the seven welds' gates** (the Apple M1 Max and one
  RTX A6000, every gate released; rule mechanics, not certification): on scratch copies of
  the committed ledgers the Metal tool rebinds 3 of 3 welds and retires 34 pins (the 31 rows
  and the 3 deleted-file pins), the Triton tool rebinds 4 of 4 and retires exactly
  `driver.py` on each; a second `--write` retires nothing and leaves the Metal ledger
  byte-identical. Against the 2026-10-12_10 round's records (cut while the root was eager)
  both tools skip those welds by name, since those records hold the retired files. After
  the rows, no weld outside the declared dispatch, route, composition and driver gates pins
  `fastpath.py` or `driver.py` (2 and 26 welds pin them, every one under a declared gate
  whose run closure holds the file).

It is the last change before the certification round after per-kernel certification, and the
tree after this entry is the round's frozen tree; the round's bind is the first to run the
rows. It changes harness tools, one test file and text only: no package module, kernel, gate,
probe, ledger, needle, mutation anchor, minted input or pending-list id moves, and none of the
250 gate and probe scripts holds a changed file in its run closure, so no gate's import record
moves, the freeze criteria and `meep_identity.json` stand, and the corpus Tier C run the freeze
launched loads none of its files. The coverage board is re-cut, the entry having moved its
outcome subject: every cell and every per-file record as before.

### The Metal composition carries MEEP's start of H on a run's first `update_H`

MEEP starts H and `f_w` as copies of B on the chunks that carry H's own layer at the run's
first `update_eh(H_stuff)` (`update_eh.cpp:165-177`), and the array path does the same in
`stepping.separate_h_from_b`. No Metal kernel starts H. The composition admitted a kernel on
that update where `launch.h_start_reasons` found B, its auxiliary and E at +0 and no magnetic
deposit, on the premise that the first `step_B` then leaves B at +0. The premise is false:
the dsig-only form and the recurrence multiply a +0 word by `kms`, which is negative on the
outer cells of a strong layer, so B lands on -0 there, and on a dsigw cell whose own `kms`
is negative too the start keeps that -0 in H where the forms alone store +0. The corpus run
of 2026-10-12 found it on `ex.differential_cross_section` (48^3, 4-cell layers, Bloch k = 0):
240 H words at step 1, sign of zero only, with the single and the fused magnetic pair alike.
The NVIDIA tables carry the start (the hand-CUDA launchers on the array path, the Triton
kernels as `start_h`) and held 189 of 189 rows there.

`launch.plan_step` now puts `launch.StartOfHPlan` around whatever serves `update_H` on a run
frozen before its first step over an active layer: the start on the array path before a
launch that steps `update_H` at that consult (`start_h_from_b`; the singles), and
`stepping.update_H` re-stepped after the slot's own work where a fused magnetic pair
launched at `step_B` (`restep_update_h`), once a run, keyed on `Fields.h_separated`. The
wrapper is transparent to the planned/null split, the pair identity and the launch
counters, and `synced` puts the residency bracket inside it. An integrated magnetic source
that reaches the cells the start copies refuses the serving product by name: MEEP starts H
before the source's `f_minus_p` words reach B, and a start at the `update_H` consult would
read them (no corpus row has one; `h_start_reasons` already refused it for a single).
`h_start_reasons` is kept as it was, as an admission bound, so the composition admits what
it admitted before; a composition frozen mid-run (`h_separated` True, as every seeded gate
state is) is unchanged. No shader byte changes.

`gate_metal_pml` gains leg `first_update_h` and `gate_metal_fused_magnetic_pair` a
`first_update_h` row (shared harness `parity/meep_gpu/metal_first_update_h.py`): the
composition the dispatch ladder builds, on a run frozen before its first step whose layer
lands B on -0, must equal the array path word for word over two complete steps, and the
same composition with the start stripped from the slot must differ. Every earlier leg seeds
a mid-run state and could not reach the start. `meep_gpu/test_metal_start_of_h.py` pins the
premise on the array path, the wrapper and the composer on any host, and the device step
on an Apple GPU.

It lands after the freeze entry and before the certification round after per-kernel
certification: CERT-5 found the defect on the previous frozen commit (`44232311`). No
array-path module, minted input, ledger, needle or mutation anchor changes, so the freeze
criteria and `meep_identity.json` stand, and the corpus Tier C run the freeze launched
(`prefer_gpu=False`) loads none of its files. The coverage board is re-cut, the entry having
moved its subjects: every cell as before, one new per-file record.

### The 1.0 freeze after per-kernel certification (CERT-1)

The series is frozen again for the one full certification round that per-kernel
certification (entries 127 to 132) owes: entry 128 changes the shape of every gate's import
record, so no record of the round on the previous frozen commit can be re-bound, and the
round runs once in full on these bytes. That round's records (CERT-3 to CERT-5) stay the
record of its pass.

- **The route constants move to the new round's stamp, `2026-10-13_10`**:
  `fastpath.DRIVER_ROUTE_FUSED_GATE` (`dispatch_fused_route_2026-10-13_10`),
  `fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE` (`dispatch_fused_route_cuda_2026-10-13_10`) and
  `metal_dispatch.METAL_DRIVER_ROUTE_GATE` (`dispatch_metal_route_2026-10-13_10`); the
  campaigns run as `dispatch_fused_route_2026-10-13_10_cc86`,
  `dispatch_fused_route_cuda_2026-10-13_10_cc86` and
  `dispatch_metal_route_2026-10-13_10_applegpu_g13s`. `2026-10-11_10` and `2026-10-12_10` are
  spent: their route campaigns released on the frozen commits before this one. Each module
  moves only that string and its comment. `tools/ci/pending_certification.txt`: the 8 "Removed
  when" clauses name the new campaigns, with one comment block; 0 ids added, 0 removed.
  `docs/development/certification.md`: the 5 Metal route examples.
- **`meep_identity.json` is re-minted by `record_meep_identity.py`**, because one of the 20
  files the mint hashes moved: `meep_gpu/from_meep.py` (entry 129). Both sets, 8 of 8 cases
  each, one invocation per set: keep on the Apple M1 Max against the contract-off MEEP 1.33.0
  single-precision reference, flush on x86-64 Linux against MEEP 1.33.0 single precision
  (its flush on import); 39,991,116 stored and 65,610,420 reader words equal to MEEP's in
  each, and the keep files byte-identical after the flush mint (32 of 32). Of the ledger's
  1,841 leaves 70 move and none is a digest, a word count or a verdict: times (35), the 16
  capsules' byte counts and sha256 (30; a capsule records the lift module's digests), the two
  sets' `from_meep.py` digests (4) and the fixture bytes (4,633,685 of 5,000,000). Every row
  of both sets replayed with MEEP not importable holds: 16 of 16 on the Apple M1 Max (flush
  under FZ) and 16 of 16 on x86-64 Linux (flush under the MXCSR lever), 2,916 of 2,916
  stored and reader digests per set on each host, every planted flip caught and placed. The
  Tier A test file reads 62 of 62 on the Mac and, NumPy only, 62 of 62 entering in keep and
  in flush on Linux. `docs/development/testing.md` states the new sizes.
- **The array-path oracle does not move**: `record_array_path_oracle.py --report` reads
  `real-v6`, fingerprint `1ab516279d74002a`, 20 of 20 case digests equal on the Apple M1 Max
  (NumPy 2.4.3) and x86-64 Linux (NumPy 2.2.6), so `FIXTURE_SET` is not bumped.
- **The coverage board is re-cut** by its unchanged generator, the entry having moved both its
  subjects (the three route modules, the fixtures and the pending list): every cell as at
  entry 132 (0 transitions), 214 of 214 per-file outcome records equal, outcome subject
  `169ba1613dc995c3` (1,124 files); `build_parity_board.py check` passes.
- **The freeze criteria.** L1′ and AP9 stand: 0 of the 29 files that decide them moved since
  entry 131, on which both re-ran green, and AP9's NumPy half re-ran green on these bytes. AP8's
  exit is not settled at the freeze: entry 129 moved the lift, so the corpus Tier C re-runs on
  all four references beside the round, and a row that does not hold stops the round before
  CERT-4.
- **Measured on these bytes** (Apple M1 Max, every marker, dispatch off): the package suite
  under the CPU-only shim 226 failed, 14,910 passed, 871 skipped; the harness suite 0 / 2,152 /
  20; the 94 device-visible files on the Apple GPU 13 / 4,151 / 331. Against entry 131's reports
  every id reads the same outcome (16,007 of 16,007 package ids, 2,172 of 2,172 harness ids,
  4,495 of 4,495 device ids); `tools/ci/check_failures.py` 63 failed of 17,896 run, 63 of 63
  declared, 0 undeclared. Needle anchors 0 differences; 204 mutation anchors, 0 stale.
- `PRE_CERTIFICATION_STATE.md` section 11 records the freeze and section 8 the bind's open
  item; `RUNBOOK.md` gives the Mac's re-run under the new stamp and that item.

### The bit-parity coverage board is re-cut on series entries 01 to 131

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..131; the previous cut was entry 125's, and the board's check had
refused since entry 127 moved its subject (entries 128 to 131 moved it further). Only three
of its generated files change (`board.json`, `board.md`, `outcomes.json`); the generator is
unchanged. Its test outcomes are the CUDA and Triton test files' cases of the package suite
(CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the device run
(Apple GPU), both run on this tree's outcome subject: 214 test files, 0 of the rows' test
files not run, 12,119 cases. Per file they equal entry 125's on 212 of its 213 files; the
213th, `test_dispersive_offdiag_polarization_pair.py`, passes 25 cases where it passed 24
(entry 130 renames one test and adds its twin), and the new `test_bloch_connection.py`
passes 7. `dev_evidence.json` is carried unchanged: each of its 11 records is still on its
live subject (module bytes, emitted text, array path).

- **No cell moves** (N / D, refused rows in brackets): (a) the chunk-record select, CUDA
  55 / 55 (+1), Triton 55 / 56 (+2), Metal 60 / 60 (+4); (b) MEEP's A form, CUDA 57 / 57
  (+8), Triton 61 / 61 (+7), Metal 68 / 68 (+9); (c) host word identity, CUDA 18 / 18
  (+47), Triton 26 / 29 (+39), Metal 29 / 29 (+48). Two CUDA refusal texts now name
  `test_bloch_connection.py` among the test files of the two off-diagonal electric products
  whose Bloch table it moved, and `bloch_connection` joins the table's modules no step list
  names.
- **The array path against MEEP** reads 6 / 6 families identical (+4 refused) under keep
  and under flush, as at entry 125.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, as before.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates.
- The subjects grow by 14 files each (outcome subject 1,110 to 1,124 files, board subject
  1,121 to 1,135): the batch's new modules and tests.

### The direct corpus check (L3) has a driver in the tree

`parity/meep_gpu/rounds/1.0/l3_corpus.py` runs CERT-5: the corpus lifted and dispatched on a GPU, every stored and `get_array` word judged against the Tier B record's stored MEEP digests, for the Triton and CUDA precedences under both baselines and for Metal under flush, one row per process (`l3_row.py`) and tested on the host (`test_l3_corpus.py`); no shipped module, gate, ledger, census or board subject changes.

### A composer handed no chunk division reads the run's own; the CUDA Bloch table resolves without CuPy

Three fixes the 1.0 certification records found (CERT-4: the Metal and NVIDIA censuses and
boards), which share the 1.0 round's one re-certification with per-kernel certification.

- **The composers' default division.** `metal_kernels.launch.chunk_division`,
  `cuda_kernels.arms.StepContext` and `triton_kernels.launch.plan_step` resolved
  `chunk_layout=None` to the one-rank division of the grid and layer, while every arm's
  off-diagonal presence clause reads the array path's presence through
  `stepping._chunk_layout`, which is the lifted run's own division. On
  `TestHoleyWvgBands.test_fields_at_kx` (an odd mirror on y with the layer on the folded
  axis's high face) MEEP's layer chunk starts one row lower than the one-rank rule's, so
  the coverage census, which hands no division in, recorded `update_E` refused (Metal:
  "2020 of 2040" cells of one chunk code keep the array; CUDA: "not a {low, interior,
  high} product along y") while the folded complex off-diagonal fused D/E products'
  predicates admitted the row, and the Metal board withheld that credit by name. A
  driver hands its own division in, and the 1.0 round's L3 had dispatched the row on all
  three tables, bit-identical to MEEP. Each composer now resolves `None` to the division
  the fields carry (`Fields.chunk_layout`), else the one-rank division, so a driver built
  without MEEP reads what it read before. Measured on the row with the fused D/E product
  dispatched, 200 steps, every stored word of the twelve components after every step
  against the array path: 0 of 11,712,000 differ on the Apple GPU (flush; MPS refuses keep
  by name) and 0 of 11,712,000 on CUDA (keep).
- **The Bloch table resolves without CuPy.** `bloch_phase_arguments` and `connection_of`
  move, unchanged, from `cuda_kernels/complex_pml_kernels.py`, which imports CuPy, to the
  new CuPy-free `cuda_kernels/bloch_connection.py`; `complex_pml_kernels` imports the
  first for its launchers and `complex_offdiag_update_e.bloch_connection_table` reads it
  there. The Bloch clause of `complex_no_pml_offdiag_fused_electric_pair` and
  `folded_complex_offdiag_fused_electric_pair` raised `ModuleNotFoundError: No module
  named 'cupy'` on a host without CuPy, which the CUDA census scored as a refusal of the
  run on 5 cells; on `TestHoleyWvgBands.test_fields_at_kx` and
  `TestMaterialGrid.test_subpixel_smoothing` the products now read
  `covered_modulo_backend` True, and the other 3 refuse on their electric source.
- **The CUDA E->P weld asks the composer's chunk-division clause.**
  `covers_dispersive_offdiag_fused_polarization_pair` refuses, with the arm table's own
  text, a run whose division the `dispersive off-diagonal` update_E arm refuses. On the
  1.0 CUDA census it admitted `examples/absorbed_power_density.py`, whose division is not
  a {low, interior, high} product along x, while the composer refused the arm (its
  update_E stepped on the array path in the round's L3), so the CUDA board credited a
  weld the composer never installs. The board's served-versus-buildable floor held only
  because `TestMaterialGrid.test_subpixel_smoothing` sat unserved in a stencil cell; the
  Bloch-table fix serves it, and without this clause the board refused to cut.
- New tests: `meep_gpu/test_composer_run_division.py` (the HoleyWvg row's configuration,
  built without MEEP, on all three composers, with the one-rank division handed over
  explicitly as the planted control, and the E->P clause on its weld's fixture) and
  `meep_gpu/cuda_kernels/test_bloch_connection.py` (both products admit in an
  interpreter where CuPy cannot be imported; a planted raise is still refused by name;
  the launcher binds the predicates' function).

### The BinaryPartition refusal reads MEEP's grid, and Table 2 holds 191 of 194 rows on all four references

Owner ruling of 2026-10-12 (R17 amended: the R10 item of section 8.1 ships in 1.0). The
gate refused every simulation that carried a `BinaryPartition` `chunk_layout` beside
mirror symmetry, on the attribute alone. MEEP writes a `BinaryPartition` back onto every
simulation whose structure it built with none (`simulation.py:2117-2121`), after it has
halved the grid. `test_simulation.py`'s `test_infer_dimensions` and
`test_require_dimensions` build the structure with a direct `_init_structure()` before
the capture, so they carried a partition and a HALVED grid: measured, `structure.gv` starts
at doubled (-2, -2) where the user volume starts at (-200, -200), S multiplicity 4, on both
rows. MEEP steps them folded, as it steps `test_in_volume`, the same simulation captured at
its `run` call, and they were refused although no unfolded run was ever involved.

- **`meep_gpu/from_meep.py`**: `_check_symmetries` asks `_partition_with_mirrors_refusal`.
  No structure yet: refused (`PARTITION_WITH_MIRRORS_REASON`), because `init_sim` will
  build the whole cell from the partition without halving it. A structure already built:
  `init_sim` keeps it, so its corners decide. Halved on every declared mirror axis, the
  simulation lifts folded, exactly as MEEP steps it; unhalved on any, it is refused with
  the axes named. `_require_folded_grid_halved`, the lift's check on what `init_sim`
  built, keeps its behaviour; its corner read is now `_unhalved_mirror_axes`, which the
  gate shares.
- **A partition given before the structure is built stays refused, now with the
  measured reason.** Section 8.1 proposed lifting that case whole with the mirrors
  dropped. MEEP's stepping is the unmirrored run's (0 of 27,027,000 stored words, phase
  0's P0-e), but its readers and monitors are not: they loop over the declared images of
  the unhalved chunks (`loop_in_chunks.cpp:377-430`). Measured on the corpus rows'
  simulation built from a one-chunk partition, with against without its mirrors, after
  200 steps: `get_array` of Ez differs in 16,246 of 40,000 words (29,799 of 40,000 with
  the source off the planes), a flux plane clear of the mirror planes reads 1.91 to 1.99
  times the unmirrored flux over its 10 frequencies, and MEEP prints `WARNING vol
  mismatch`; `get_field_point` (4 of 4) and the DFT of the fields (26,244 of 26,244
  words) agree. No lift reproduces both the stepped words and those readings, so the
  refusal stands and its text says why. With the structure built first, the same
  simulation's `get_array` (0 of 240,000 words over the six allocated components), both
  fluxes (0 of 20), point reads (0 of 4) and DFT (0 of 26,244) equal the default run's.
- **The two rows hold on every reference.** Tier C on entries 01 to 126 with this change,
  dispatch off, `OMP_NUM_THREADS=1`: each row 0 of 3,636,060 stored and 0 of 14,400,000
  reader words differing on mac_keep, mac_flush, linux_keep and linux_flush, the planted
  flip caught in each, the FPU checks 30 of 30, the subnormal stamp each reference's own;
  their MEEP digests, plane digests and source currents equal `test_in_volume`'s of the
  same run, on every reference. Context L2 against the Mac lock's certification MEEP:
  1.045e-10 for each, as for `test_in_volume`.
- **Nothing else moves.** Only a simulation with mirrors AND a `BinaryPartition` at gate
  time reaches the new branch, and in the corpus that is these two rows: the only
  `_init_structure` calls in MEEP's examples and tests, and the only rows any reference
  named the clause for (2 of 194). A 20-row sample (1-D, 2-D, 3-D, cylindrical, Bloch and
  special-kz, mirrored, PML and absorber rows) and `test_in_volume` re-run on both hosts
  in both modes hold with 0 differing words, and every row's digests, planes and currents
  are byte-identical to the record's (21 of 21 per reference).
- **Table 2** ([the page](docs/development/meep-identity-corpus.md)) regenerated from a
  Tier B record of the merged cut (the record's rows with the two rows over them): 57 of
  60 examples and 134 of 134 tests hold in each of the four references; refused by name on
  both hosts, in both modes: `ex.coupler` (no libGDSII) and `ex.ring_gds` (no gdspy); one
  exclusion. Words held: 2,736,015,798 stored and 4,026,704,328 reader in each. The two rows
  carry capsules, so each set replays 93 of 93 holding capsule rows with no MEEP (91
  before). L0, the two reference MEEPs' digests on the rows run on both hosts (now 192 of
  194): keep, equal on 163 and different on 29; flush, 160 and 32; the two rows are equal.
- **Tests.** `test_chunk_map`'s MEEP child adds three cases: the written-back partition
  passes the gate and lifts folded on the default lift's grid and division; the same
  partition copied into a new simulation is refused; a structure built from a given
  partition before the lift is refused with "not halved on y, z". `mutate_from_meep.py`
  gains four mutations of these clauses (200 to 204), each caught.
- **Docs.** The compatibility and troubleshooting guides name the case that stays refused
  and the one that lifts. The fused H->D gates already count a lifter refusal by name; the
  two rows now reach their predicates, as `test_in_volume` does, and the certification
  round measures them.

### A gate's import record is its own dependency set (per-kernel certification)

Owner ruling R17 (2026-10-12). Every gate artifact records the repository modules its
process imported, and the rebind tools, the family, composition and bit-identity re-cuts, the
Metal runner's manifest and the three board builders check that whole record against the
checkout. `meep_gpu/__init__.py` imported the driver, the MEEP lift, the subnormal policy and
the backends eagerly, and the driver imports the dispatch planner, so every record held the
package core: 701 of 701 artifacts of the 2026-10-12 round recorded `meep_gpu/fastpath.py`,
and a dispatch-only edit voided every gate on all three tables.

- The package root binds its public names on first use (PEP 562). `import meep_gpu` loads
  `meep_gpu/__init__.py` and nothing else; `meep_gpu.<name>`, `from meep_gpu import <name>`
  and `from meep_gpu import *` give the same objects as before, `dir()` lists the same names,
  and the submodules an eager import left bound on the package (`meep_gpu.stepping`, ...)
  are still attributes. No module reads the environment, installs or reads a subnormal
  policy or probes a backend at import, so binding a name later decides nothing differently
  (`meep_gpu/test_lazy_package.py`).
- `meep_gpu/cuda_kernels/test_root_chunk_forms.py` imports the driver where it builds a case,
  not at import: the CUDA ADE gate imports the file for its interpreter alone.
- `parity/meep_gpu/import_closure.py` reads a gate's import closure and an upper bound on its
  run record off the source; `import_closure.py bill <file>` lists the gates whose records an
  edit of that file can move. `parity/meep_gpu/test_gate_import_closure.py` holds every gate
  and probe script to it: 98 that dispatch, route, compose or build a driver are declared by
  category, and any other whose closure reaches the driver, a dispatch planner or the MEEP
  lift fails, with planted eager imports as controls.

The whole-import checks are unchanged; the records shrink. Measured in a fresh interpreter,
the import closure of the 105 undeclared gate scripts that import the package loses a median
of 14 repository files (0 to 22): the driver, the dispatch planner and the lift leave 103 of
them (`gate_metal_pml.py` 44 to 31 files, `gate_cuda_offdiag.py` 30 to 15), and none of the
112 undeclared `gate_*.py` scripts reaches the dispatch layer, at import or in a run. The change moves every gate's import record, so the next certification round
runs in full once; after it, a dispatch-only edit re-runs the declared route, dispatch,
composition and driver gates and the records that pin the planner, not the kernel gates.
`meep_gpu/__init__.py` is pinned by 7 ledger entries that also pin `meep_gpu/fastpath.py` (the
Triton bit-identity gate and three Metal welds, by code and source digest), which the round
re-binds.

The fixes the owner moved into 1.0 with this ruling share its one certification round. Two
land beside it: "The BinaryPartition refusal reads MEEP's grid" and "A composer handed no
chunk division reads the run's own" (sections above). The third is deferred to 1.0.1: the
H->D products' predicates passing the PML layer to `metal_kernels.launch.h_start_reasons`, as
the base composer does, so a magnetic source refuses only where the start writes its cells.
No verified change was ready for this round, and its measurement changes the premise:
narrowed, the clause still refuses all 31 Metal H->D rows the board files `missing_half`
(their magnetic sources do reach the start cells; serving them needs a cell-level clause in
the base composer, which changes dispatch). What it gains is agreement between the H->D
predicates and the board on rows the board already credits. The H->D products are not
installable, so dispatch is the same either way.

### The round's harness fixes land in the checkout, and the board bindings move by tool

The 1.0 certification round ran six harness fixes from outside its frozen checkout (the
two rebind retirements, the Metal board's credit rules, the Metal route's tree check, the
composition re-cut's key list, and the boards' release bindings re-pointed by hand or by
one-off scripts). They are now in the checkout, so the next round runs only committed tools.
No shipped module, kernel, predicate, gate, ledger, census, needle or mutation anchor
changes, and no ledger pin, census subject or route digest list names a file this changes.

- **`rebind_metal_welds.py` drops a curated pin whose file was deleted, by name.** The
  tool keeps each weld's curated pin set, so the three welds that pinned
  `meep_gpu/expansion_refusal.py` (deleted with the expansion licence) could never
  rebind: every fresh artifact lacks the file, and the entry was skipped as "artifact does
  not record" in every round. A reviewed table, `RETIRED_PINS`, now names that path; when
  an entry rebinds, a retired path leaves both digest maps and is reported as
  `RETIRED-PIN`, and every other pin is compared as strictly as before. A pin is dropped
  only inside a rebind, the run refuses if a retired file is back in the checkout, and an
  artifact that recorded the retired file cannot bind. A rebind that would change nothing
  but the recording time keeps the entry, so a second `--write` leaves the ledger
  byte-identical. On both Metal fleets of the 1.0 round (stamps 2026-10-11_10 and
  2026-10-12_10): 62 of 62 welds rebind (59 before), each of the three with exactly one
  `RETIRED-PIN`, and a second `--write` changes no byte (`test_rebind_metal_pin_retirement.py`
  is new).
- **`rebind_cuda_welds.py` drops a curated pin the weld's gate no longer imports, by
  `(weld, path)`.** `cuda_complex_offdiag_stencil_welds_2026-09-17` (cited by 2 of the 34
  CUDA arms) pins `gate_cuda_complex_no_pml.py`, whose import entry 89-X1 removed with the
  expansion licence; `cuda_cylindrical_fused_hd_pair_2026-09-07` (cited by no arm) pins
  `gate_cuda_cylindrical_real_fused_hd_pair.py`, whose import entry 76-KC8 removed. Both
  files still ship and other welds still pin them, so the reviewed table, `RETIRED_PINS`,
  is keyed by weld. A pin is dropped only inside a rebind, an entry whose legs record a
  retired path again is skipped by name, and a second `--write` leaves the ledger
  byte-identical. On the 1.0 round's CUDA re-gate campaigns: 54 of the 55 claimed welds
  rebind (52 before; the 55th was never in the ledger), the two with exactly one
  `RETIRED-PIN` each, and the CUDA flush licence admits 34 of 34 arms (32 before)
  (`test_rebind_cuda_pin_retirement.py` is new).
- **The Metal board credits the pair that absorbs a slot the base composer pops.** The
  board (`build_fusion_matrix.py`) refused the 1.0 census at its disjointness floor: 354
  seam-instances admitted by a product's clause ladder against 322 credited. The census
  records `launch.plan_step` with fusion off, which since series entry 18 pops `update_H`
  on the run's-first-`update_H` start clause (`launch.h_start_reasons`) after the fusion
  block, "so a pair that absorbs the slot keeps it". Such a seam-instance is now credited
  to the pair that absorbs it, by the composer's own objects (the start-clause wording
  derived from `h_start_reasons`, the arm registry, `FUSED_PAIR_SEAMS`, `FUSED_PAIR_ARMS`
  and its declared extras, `_pair_may_absorb`, `_declared_uninstallable`, and the pair
  plan's `starts_h_from_b`; `metal_board_credit.absorb_credit`, new), and every credit
  names its pair. One D_to_E credit is withheld by name
  (`metal_board_credit.CREDIT_WITHHELD`: `tests:TestHoleyWvgBands.test_fields_at_kx`,
  which `folded_complex_offdiag_fused_electric_pair`'s predicate admits while the composer
  refuses the `update_E` arm it implements on the chunk record's off-diagonal presence),
  re-read off the census on every cut. The disjointness floor is asked per product, as row
  sets (a planted overlap refused and named its 50 rows), and the standing-records
  cross-check names the two rows the frozen lift refuses by name (ruling R10) instead of
  counting them. On the 1.0 census with the bindings re-pointed to the round: served 500 of
  591 priced seam-instances, served in dispatch 323 of 591 (the 0.9.2 board: 531 of 597
  and 327 of 597; `test_metal_board_credit.py` is new).
- **The Metal route's tree check reads `git status --porcelain` line by line.**
  `metal_round.py`'s `tree_is_the_commit` stripped the whole output, so the first line's
  leading space went with it: the one change the route stage allows, the Metal ledger the
  bind writes (unstaged, `" M <path>"`), was read one character short and refused. It now
  splits the output as `legs.py`'s check does (`test_metal_round_tree_check.py`, new,
  drives the four states on a throwaway repository: clean, the Metal ledger unstaged,
  that and another file, the Metal ledger staged; the old check refused the second).
- **The ten Triton H->D gates are in the round's leg table.** The Triton board binds its
  H->D products through `GATE_BOUND`, which re-hashes each gate's artifact against the
  tree, and no step of the round ran those gates, so the 1.0 round's Triton board could
  not credit them on its frozen bytes. `legs.py`'s `TRITON_HD` now runs all ten under
  keep, every leg, lifts on, as hand legs (`run_lane.py` kind `hd`, in the fleet driver's
  per-gate environment), each into `triton_<gate>_<stamp>/keep/gate.json`, the layout
  `GATE_BOUND` names; they are priced at their walls in the 1.0 pre-certification pass
  (12,746 s in all), one movable unit each. The planner's 4-GPU CERT-3 wall moves from
  21,165 s to 23,991 s (HIGH 29,805 s to 31,607 s). They stay out of the fleet driver:
  no ledger welds them, and `INSTALLABLE` is False on all ten.
- **The bind re-cuts the composition records by their keys.** `legs.py bind` printed the
  composition re-cut without `--only`, and `recut_composition_records.py` refuses that
  all or nothing (the composition root holds the legs of five of its nine keys); it now
  prints `--only` with the five keys, which `legs.py check` derives from the tool's
  `LEGS` and refuses on a difference.
- **The boards' gate bindings are re-pointed by a tool, from a measurement.**
  `repoint_board_bindings.py` (new) re-points the Triton board's `GATE_BOUND` and the
  Metal board's `"gate"` bindings to a round's runs, once per round at the records step.
  On the Triton board each entry's candidates are read through the release-binding
  floor's own rules (`build_triton_fusion_matrix.gate_bound_check`, now the one reader
  both use), and exactly one keep run that the floor admits with zero drift is chosen;
  two refuse, and an entry with none refuses the write or, with `--withhold-unrun`, is
  withheld by name in `CREDIT_WITHHELD_STALE_BINDING`. On the Metal board each reference
  moves to the round's directory of the same gate, which must record its release and
  re-hash with zero drift. Report mode writes nothing, a write re-reads the table it wrote,
  and a second write changes no byte. On the 1.0 round's evidence: Metal 45 references
  over 28 directories, and the board reproduces the round's (500 and 323 of 591); Triton
  25 of 39 entries re-pointed and 14 with no run of the round (the H->D entries above),
  so the default refuses, and `--withhold-unrun` reproduces the round's diagnostic
  board (353 and 330 of 591). `test_repoint_board_bindings.py` and
  `test_round_triton_hd.py` are new.
- `RUNBOOK.md`: every board reads the H->D probe each freeze re-cuts
  (`MEEP_GPU_H_TO_D_PROBE`, set on the board commands only), section 2's wall times, and
  the records step's re-point.

### The round planner prices the CERT-3 NVIDIA re-run at CERT-3's measured walls

CERT-3 NVIDIA of stamp 2026-10-11_10 packed its lanes by walls measured elsewhere (the
Triton legs at the final pre-certification pass's, four gates sharing a card; the CUDA legs
at 0.9.2's): its Triton core lanes ran 15,650 s and 17,305 s against 23,864 s planned while
the CUDA lanes ran 23,079 s and 24,675 s, so two GPUs sat idle for about two hours.
`legs.py`'s CUDA tables (118 legs) now carry each leg's CERT-3 wall, and `TRITON_EXPECT` and
`TRITON_HAND_EXPECT` the mean of each gate's CERT-3 keep and flush walls, rounded up. Fourteen
prices are raised so that each step watchdog, `max(1,800 s, 4 x expected)`, is 3x or more the
larger of the leg's CERT-3 and pre-certification walls (on the old prices 15 CUDA legs sat
below that, the lowest at 1.57x), and `fleet_launch.py`'s floors raise four fleet gates' own
timeouts for the same reason (`conductivity` and `symmetry` 10,800 to 15,900 s,
`folded_complex` 10,800 to 13,500 s, `special_kz` 7,200 to 12,600 s). The planner's 4-GPU wall
moves from 23,834 s to 21,165 s (lanes 21,165, 20,136, 20,351 and 21,145 s; 19,649 to 20,798 s
re-summed at CERT-3's walls), and `RUNBOOK.md` section 2 quotes the new plan. Route and
licence prices are unchanged; no shipped module, gate, ledger, needle or mutation anchor
changes. It is the fourth change of the 1.0 re-freeze, after the hardware-scope rename: the
tree after it is the frozen tree of the 1.0 certification round, and it voids nothing, since
no ledger's curated pin names a file it changes.

### `MEEP_GPU_HARDWARE` replaces `MEEP_GPU_ALLOW_UNCERTIFIED`

Owner ruling of 2026-10-11. `MEEP_GPU_ALLOW_UNCERTIFIED` read as a yes/no answer, and
its values did not say what they did: unset already ran a supported device or toolchain
that is not certified (with a NOTE line), `1` also ran an unsupported one, and `0`
restricted the kernels to certified ones. One variable now names the hardware scope by
value:

| `MEEP_GPU_HARDWARE` | What runs | Was |
|---|---|---|
| `certified` | Certified identities only. Everything else, an identity that cannot be read included, is refused by name. | `MEEP_GPU_ALLOW_UNCERTIFIED=0` |
| `supported` (the default, when unset) | Certified identities; a supported, uncertified one with the NOTE line and `certified: False`; an unreadable one with `certified: None`. An unsupported NVIDIA identity is refused by name. | unset |
| `any` | Every identity; an unsupported one is recorded `supported: False`. | `MEEP_GPU_ALLOW_UNCERTIFIED=1` |

Each value admits exactly what its old spelling admitted, on every table; what changes
is the names, and the words of the NOTE line and of the refusals that name them. It is the
third change of the 1.0 re-freeze, after the F2 fix, and on the round's bytes it voids
nothing the F2 fix does not already void: of its files the ledgers pin `fastpath.py` and
`metal_dispatch.py`, which the F2 fix moves too, and `fastpath_cuda.py` (a comment line) and
the route gate, which only the CUDA `driver_dispatch` record pins, a record not yet cut on
the round's bytes.

- **Strict, by name.** Any other value (empty, whitespace, another case such as `Any`,
  padding, the old `0` and `1`) is refused by name at rung 3b, with the three accepted
  values, and the run takes the array path, on every table and on a certified host as
  well. Exact match, as every switch of the package is read.
- **The old name is refused, never read.** `MEEP_GPU_ALLOW_UNCERTIFIED` set to anything
  is refused by name at rung 3b, before any table's identity is read. The refusal names
  `MEEP_GPU_HARDWARE` and the value that does what the old one did:
  `MEEP_GPU_ALLOW_UNCERTIFIED=0` is `MEEP_GPU_HARDWARE=certified`, `=1` is `any`, and
  any other value is told both. There is no alias and no fallback reader. It is not in
  `fastpath.LEGACY_SWITCH_NAMES`, whose renames kept their values; this one changed them.
- **`meep_gpu/fastpath.py`**: `UNCERTIFIED_SWITCH` is `HARDWARE_SWITCH`,
  `UNCERTIFIED_VALUES` is `HARDWARE_VALUES` (`("certified", "supported", "any")`, with
  `HARDWARE_CERTIFIED`, `HARDWARE_SUPPORTED`, `HARDWARE_ANY` and `HARDWARE_DEFAULT`), and
  the retired name is `RETIRED_HARDWARE_SWITCH` with `RETIRED_HARDWARE_VALUES`.
  `certified_only()` and `unsupported_allowed()` are replaced by one reader,
  `hardware_scope()`, which answers the scope, or `None` for a value rung 3b refuses.
  `CERTIFIED_ONLY_TAIL` is `CERTIFIED_SCOPE_TAIL`; `UNSUPPORTED_HINT`,
  `NVIDIA_SUPPORTED_BECAUSE`, `NVIDIA_UNSUPPORTED_BECAUSE` and the rung 4e refusal name
  `MEEP_GPU_HARDWARE=any` or `=certified` where they named `=1` or `=0`.
  `_uncertified_value_refusal` is `_hardware_value_refusal`, and reads both names.
- **`meep_gpu/metal_dispatch.py`**: its own `certified_only()` is gone; the Metal ladder
  asks `fastpath.hardware_scope()`. `SUPPORTED_BECAUSE`, `UNCERTIFIED_BECAUSE` and the
  two refusal tails name `MEEP_GPU_HARDWARE=certified`. On the Metal table `supported`
  and `any` both run every environment, as unset and `1` did.
- **The run record keeps its field names**: `certified`, `supported`,
  `uncertified.served`, `uncertified.admitted`, `uncertified.dropped`,
  `uncertified.allowed`, `uncertified.unsupported_allowed` and `environment.certified_only`
  describe the run, not the switch. `uncertified.variable` now reads
  `MEEP_GPU_HARDWARE` and `uncertified.value` carries its value as set (`None` when
  unset).
- **The NOTE line and the refusals** name the new values, for example
  `... and MEEP_GPU_HARDWARE=certified would restrict the kernels to certified ones`,
  `set MEEP_GPU_HARDWARE=any to dispatch the kernels on it anyway, without
  certification or support`, and `MEEP_GPU_HARDWARE=certified restricts the NVIDIA
  kernels to certified devices and toolchains`.
- **Harness and tools.** The route gate (`gate_dispatch_fused_route.py`,
  `hardware_switch_refusal`, was `uncertified_switch_refusal`) and `rounds/1.0/legs.py
  check` refuse a route leg with either name exported. `drive_triton_weld_gates.py` runs
  every gate child with `MEEP_GPU_HARDWARE=certified` and without the old name.
  `rounds/1.0/run_lane.py` and `metal_round.py` scrub both names (the Metal preflight
  refuses either exported). `timing_ladder.py` exports `certified` to its NVIDIA rows,
  `any` under its own `--allow-uncertified 1`, and unsets the old name on every row; the
  ladder's parameter and its `allow_uncertified` row field keep their names, because
  `build_identical_case_table.py` reads that field back from the dated ladders of
  2026-09-28. `tools/check_install.py` reads `MEEP_GPU_HARDWARE`, and names a refused
  value or the old name instead of expecting the kernels.
- **Docs**: `README.md`, `INSTALL.md`, `AGENTS.md`, `pyproject.toml` and the guides
  describe `MEEP_GPU_HARDWARE`. Dated round records (`parity/meep_gpu/rounds/h200_2026-10-03`,
  `parity/meep_gpu/rounds/metal_2026-10-04`), this changelog's earlier sections and the
  dated prose of the pending list keep the name they were written with.

### Rung 8b answers a bare label in a CUDA-led plan from the Triton ledger (F2)

Defect F2 of the 1.0 certification round. Under `MEEP_GPU_BACKEND_PREFERENCE=cuda` the
merged plan holds Triton arms, under bare labels, in the slots the CUDA products did not
take; every CUDA label carries the `cuda:` prefix. `fastpath._table_for_label` answered
such a bare label with the plan's table, so with a caller-installed `flush` rung 8b's
per-arm evidence (`fastpath.policy_evidence_reasons`) looked every Triton arm up in the CUDA
ledger: it read "'unmapped' has no live run on compute capability 8.6" and refused the whole
plan to the array path. Five bare labels the CUDA composer also spells (`PML`, `ordinary`,
`off-diagonal`, `folded off-diagonal`, `mirror fill`) were answered from their `cuda:`
twin's run instead, the other table's evidence. Keep was untouched: the install default
asks no arm anything. On the route gate's `cuda_flush` leg, a scratch dry run on the frozen
commit `2b75bef8` with the rebound ledgers failed the case `dispersive_2d`
(`NO-SUBSTITUTION`) and the envelope control on `nonlinear_3d` (`ENVELOPE-DID-NOT-HOLD`),
with or without fix B, so that leg could not release.

- **`meep_gpu/fastpath.py`**: `_table_for_label` now does what its docstring said. A
  namespaced label answers for itself; a bare label in a plan composed on the CUDA table is
  a Triton arm and is answered from the Triton ledger; a bare label on any other table
  stays on that table (Triton on a CuPy host, Metal on a NumPy host with an MPS device).
  `_table_of` is unchanged. Its two callers: rung 8b (through `_certification_for`) moves as
  described. The dispatch record's per-arm certification was already keyed on the merged
  plan's `backends`, so it reads the same.
- Measured host-side against the scratch-rebound ledgers: the six Triton singles of those
  cases (`nonlinear`, `nonlinear run PML`, `ADE update_P`, `dispersive`,
  `conductive no-PML`, `complex conductive no-PML curl`) are admitted under flush in a
  CUDA-led plan (6 of 6; 0 of 6 before). All 75 bare Triton labels answer under plan table
  `cuda` exactly as under `triton` (75 of 75; 5 of 75 before), and all 34 `cuda:` labels
  answer as before (34 of 34). `fused pair B` and `fused pair D` are admitted once fix B's
  flush evidence is bound; until then they are refused by name for
  `triton_fused_electric_device_gate`, no longer as unmapped. Under keep every label of
  both maps answers no evidence needed, on both plan tables (109 of 109).
- **`meep_gpu/test_dispatch_label_table.py`** (40 tests): the label table (a CUDA-led plan's
  Triton singles, Triton fused pairs and the five shared spellings, a Triton-led plan, a
  Metal host, namespaced labels) and the lookup it routes. Rung 8b under flush on a
  CUDA-led merged plan: Triton singles admitted, a Triton fused pair refused for its own
  gate and admitted once its flush run is bound, a shared spelling never admitted on its
  CUDA twin. The keep path: no per-arm lookup, and the admission, the plan and every record
  field equal to those of the lookup it replaced, on CUDA-led and Triton-led plans. With
  the fix reverted, 24 of the 40 fail; the 16 that pass are the label-table rows the change
  does not touch, the `_table_of` and Metal-map tests and the keep tests.
- **`metal_dispatch.METAL_DRIVER_ROUTE_GATE` moves to `dispatch_metal_route_2026-10-12_10`.**
  The Metal route record pins `meep_gpu/fastpath.py`, and the `_2026-10-11_10` campaign has
  run (it released every leg on the first frozen bytes, and route campaign names are
  one-shot), so the held route re-runs as
  `dispatch_metal_route_2026-10-12_10_applegpu_g13s`. The NVIDIA route constants keep
  `_2026-10-11_10`, which no NVIDIA route has used. `tools/ci/pending_certification.txt`
  (the two Metal "Removed when" clauses; 0 ids added, 0 removed),
  `docs/development/certification.md` (the Metal route examples) and
  `parity/meep_gpu/rounds/1.0/RUNBOOK.md` name the new campaign.
- **What it voids**, by the ledgers' curated pins: `meep_gpu/fastpath.py` is pinned by 7
  entries (the three tables' `driver_dispatch` records, the Triton `bit_identity_gate`, and
  the Metal welds `metal_complex_conductive_fused_pair`, `metal_complex_no_pml_conductive`
  and `metal_cylindrical_real_fused_magnetic_pair`), `meep_gpu/metal_dispatch.py` by the
  Metal `driver_dispatch` record alone, and no other changed file by any entry. On the
  round's bytes that voids the Metal route record, the three Metal welds' runs and the
  Triton bit identity. The NVIDIA `driver_dispatch` records have not been cut on the frozen
  bytes, so the one-shot NVIDIA route cuts them on the re-frozen ones. Every gate artifact
  of the round also records `meep_gpu/fastpath.py` among its imports (Metal 62 of 62, NVIDIA
  246 of 246), so the tools that bind by a run's whole import record refuse those runs on the
  re-frozen bytes: the Metal board (so the whole Metal fleet re-runs), the family records'
  flush evidence, eight Triton welds that pin their own test file, and the bit identity.
  `PRE_CERTIFICATION_STATE.md` section 8 gives the re-runs.

### The fused D->E gate installs and stamps the campaign's subnormal policy

`triton_fused_electric_device_gate` certifies the arms `fused pair B` and `fused pair D`.
Its gate installed no float32 subnormal policy and stated none, so its ledger entry
declared `_mixed_policy` and no run of it could carry flush evidence: both arms were
refused on every flush dispatch. On the route gate's `cuda_flush` leg that refusal
reaches past the cases. With the backend preference unset, Triton composes first, so the
arbitration control's default-precedence read plans `fused pair B` / `fused pair D` on
every case whose Triton composition uses them. Rung 8b then refuses the whole plan, the
control scores `REFUSED`, which is not a passing control verdict, and the leg cannot
release. A scratch dry run of `cuda_flush` on the frozen commit `2b75bef8` with the
rebound ledgers (the keep preflight's 6 cases) read `REFUSED` on 3 of the 4 cases that
reached arbitration: `pml_2d`, `dispersive_2d` and `pml_3d`. The same run found a second,
independent blocker on that leg, which this change does not touch: rung 8b reads a bare
Triton label in a CUDA-led plan against the CUDA ledger (`RUNBOOK.md` section 5).

- **`parity/meep_gpu/gate_triton_fused_electric.py`** does what the five composition
  gates do (`composition_policy.py`, R16). Before it builds anything it installs the
  policy `MEEP_GPU_SUBNORMAL_POLICY` names (`keep` when unset) on the host, CuPy and
  Triton, and refuses by name when the policy is not attained (`summary.status =
  refused`, nothing measured). It stamps the policy as `subnormal_policy` and confirms at
  the end of the run that it still holds. It lifts no MEEP simulation, so under `flush`
  it runs through the flush fleet's MEEP-importing wrapper, which `run_lane.py` always
  uses. No case, mutation or verdict rule changes.
- **The ledger entry no longer declares `_mixed_policy`**, and
  `test_triton_weld_contract.POLICY_UNNAMED_BUDGET` no longer names it.
  `rebind_triton_welds.py` refuses a stamped run for an entry that declares the
  exception, and no tool retires one, so the declaration goes in the same reviewed change
  as the gate. The entry's run record still reads `NOT INSTALLED - see _mixed_policy` until
  the round's rebind binds the gate's keep and flush runs on these bytes. Until then,
  `test_every_triton_weld_names_the_policy_it_was_cut_under` is declared pending.
- **`test_gate_triton_fused_electric_host.py`** drives the gate's `main` with the device
  work stubbed out and the policy module simulated. Under keep and flush the verdict
  carries the attained stamp and the rebind reads that policy through the shipped entry.
  An unset campaign installs keep. A planted policy that is not attained (raised at the
  install, or one executor short) refuses before anything is measured, a policy name no
  record can claim is refused by name, and a policy that does not hold to the end refuses.
- **Owed by the round on the new freeze**: the `fused_electric` fleet gate under keep and
  flush, then the Triton rebind of `triton_fused_electric_device_gate` from those two runs
  (`RUNBOOK.md` section 5).

### Last pre-freeze fixes: route watchdogs from measured walls, the CUDA board's detector control and Bloch cells, the Metal composition rows

Route campaign names are one-shot: a route or licence leg that its watchdog kills has used
its name, and the certification round would need a new route stamp, which is a new freeze.
These fixes keep a watchdog from firing below a measured wall and clear three findings the
round's own tools would trip on. No shipped module, predicate, kernel, plan or gate decision
changes, and no refusal is relaxed. These were the last changes before the first frozen
commit of the 1.0 certification round (`2b75bef8`); the re-freeze adds the sections above.

- **The route and licence legs are priced at measured walls.** `legs.py`'s `ROUTE_CUDA`,
  `ROUTE_TRITON` and `E2E_EXPECT` carry each leg's wall in the 1.0 final pre-certification
  pass (rounded up; every route lane shared one RTX A6000), where they carried 0.9.2 walls
  (Triton `shipped`, `harness_keep` and `flush` at 450 s against measured 3,181, 1,573 and
  3,066 s). The route stage's watchdog, `max(3600 s, 12 x expected)` (`--route-min-timeout`,
  `--route-timeout-factor`), derives from these: computed from the drivers on `round.py`'s
  own route plans, the smallest is now 12.0 x its leg's measured wall, where Triton `shipped`
  and `flush` sat at 1.70 x and 1.76 x on the 0.9.2 prices. `test_round_route_watchdog.py`
  holds every route and licence watchdog at 3 x or more of the walls it records, through the
  route stage's own plans. `legs.py estimate`'s CERT-4 wall is 12,733 s (the CUDA lane;
  Triton 9,235 s), which `RUNBOOK.md` section 2 now quotes in place of 6,343 s.
- **The Metal route campaign's watchdog comes from its measured walls.** `metal_round.py`
  gave the whole campaign a fixed 6 h, 1.7 x the 12,648 s the pre-certification pass
  measured on a loaded Mac. It is now `ROUTE_TIMEOUT_FACTOR` x the serial sum of the four
  legs' measured walls (`ROUTE_EXPECT`), at least `ROUTE_MIN_TIMEOUT`: 53,760 s, 3.0 x the
  serial wall and 4.25 x the measured campaign, with `--route-timeout-factor` and
  `--route-min-timeout`. Nothing below it times a leg (`run_metal_dispatch_campaign.sh`'s
  measured seconds only order its lanes; the runner and the gate carry no timeout).
- **Every Metal round child runs with `PYTHONNOUSERSITE=1`**, as the certification
  reference runs. The reference interpreter's user site directory holds seven
  distributions (mypy, ruff, openpyxl and their dependencies), none of which the
  environment or the tree imports, so nothing was shadowed; it sits ahead of the
  environment's site-packages on the import path, so the round now keeps it off, and the
  preflight refuses a child interpreter that still enables it.
- **The CUDA fusion board's detector follows an imported prelude.** The control
  `cylindrical_kernels.py` (a curl, a stencil by construction) measured POINTWISE: its
  neighbour reads are `shift_up`/`shift_dn` in the real-PML prelude it imports from
  `step_curl_kernels` and concatenates into its kernels, and the detector read only the
  module's own literals (the body's direct radial read had moved to `cylindrical_prefix`'s
  scan). So `run_detector_controls` raised and the board refused at its controls. The
  detector now resolves a prelude a module imports from a sibling and concatenates into a
  kernel source: of the 46 kernel modules and 7 explicit labels it measures, only
  `cylindrical_kernels.py` moves (POINTWISE to STENCIL), and the four controls reproduce.
  `test_cuda_board_detector.py` (new) pins the controls and that each fails on a planted
  error: the detector made blind again, the cylindrical curl planted to read only its own
  cell, a detector answering STENCIL always, a neighbour read planted in the pointwise
  control, and one planted in each Bloch body.
- **The CUDA board's two Bloch constitutive cells are determined.**
  `fused_update_{H,E}_pml_complex_bloch` were reconstructed from the constitutive template
  with five splice points unfilled (`__BOUNDS__`, `__FLAG0__` to `__FLAG2__`) and read
  UNDETERMINED. They are measured on `complex_emitter.complex_source`, the text the
  shipped launcher compiles, and read POINTWISE: each component reads only its own cell.
- **The Metal composition matrix pins what the planner does on four cylindrical complex
  rows.** At the builder's Courant 0.37, not a float32 word, the cylindrical complex curl
  refuses by name the slots carrying an axis product with Courant (`step_B` and `step_D` on
  `dcyl_m1_complex`, `_negative` and `_zperiodic`; `step_D` alone on `dcyl_m0_complex`);
  the four rows now pin that, and four `_f32_courant` twins at the float32 word pin all
  four slots composed, so the matrix still drives the curl on every specialisation axis.
  The twins need the Apple GPU, so `tools/ci/declared_resources.txt` lists them under
  `torch` beside their siblings. `test_metal_planner_composition.py` and
  `probe_metal_planner_composition.py` pass with the evidence archive (153 of 153 cases;
  114 configurations, VERDICT PASS) and without it.

### The 1.0 freeze (CERT-1)

The freeze of the 1.0 certification round; only the last pre-freeze fixes (the section
above) follow it before the first frozen commit, and the re-freeze's two fixes, the
hardware-scope rename and the CERT-3 re-pricing (the first four sections) before the re-frozen
one. From here on an edit to the array path voids every later device hour.

- **The route campaigns move to the 1.0 stamp.** The three route constants move from 0.9.2's
  stamps to the round's one stamp, `2026-10-11_10`: `fastpath.DRIVER_ROUTE_FUSED_GATE`
  (`dispatch_fused_route_2026-10-11_10`), `fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE`
  (`dispatch_fused_route_cuda_2026-10-11_10`) and `metal_dispatch.METAL_DRIVER_ROUTE_GATE`
  (`dispatch_metal_route_2026-10-11_10`). The campaigns therefore run as
  `dispatch_fused_route_2026-10-11_10_cc86`, `dispatch_fused_route_cuda_2026-10-11_10_cc86`
  and `dispatch_metal_route_2026-10-11_10_applegpu_g13s`. Route campaign names are one-shot:
  a leg that runs and does not release, or is interrupted, needs a new stamp and so a new
  freeze. The 0.9.2 campaigns stay the record of the 0.9.2 route. Each of the three modules
  moves only that string and its comment. `tools/ci/pending_certification.txt`: the "Removed
  when" clauses that named the 0.9.2 campaigns name these, and no id is added or removed;
  `docs/development/certification.md`: the Metal route examples name the new stamp.
- **The route stage's watchdog.** `round.py launch route` gives every route and licence leg
  `max(3600 s, 12 x its expected wall)` (`--route-min-timeout`, `--route-timeout-factor`),
  the values the pre-certification pass ran those legs with, in place of `run_lane.py`'s
  `max(1800 s, 4 x expected)`, which sat below walls that pass measured on shared cards
  (Triton `shipped` 3,180 s, `flush` 3,065 s): a leg its watchdog kills has used its one-shot
  name. The gate stage is unchanged. `parity/meep_gpu/test_round_route_watchdog.py` (new, 3
  tests) pins it.
- **`meep_identity.json` re-minted on the frozen bytes**, both sets by
  `record_meep_identity.py`, 8 of 8 cases each, 39,991,116 stored and 65,610,420 reader words
  equal to MEEP's per set: keep on Apple silicon against the contract-off identity reference,
  flush on x86-64 Linux against the Linux reference, one invocation each. Every digest is
  unchanged. The keep capsules record the array-path modules as they are now (8 capsules
  change, by -2 to +4 bytes each); the flush capsules are byte-identical; the ledger changes
  in the capsule hashes, the fixture byte count (4,633,666 to 4,633,671 of 5,000,000) and
  timings.
  Every row of both sets replays with MEEP not importable, 16 of 16 on each host.
- **The array-path oracle does not move**: `record_array_path_oracle.py --report` reads
  `real-v6`, fingerprint `1ab51627`, the same 20 case digests on Apple silicon and x86-64
  Linux as at entries 61 and 85, so `FIXTURE_SET` is not bumped. The kernel ledgers'
  `array_path_oracle` leaves are written after the certification round, not here.
- **The coverage board is re-cut** on these bytes by its unchanged generator, because this
  change moves both of its subjects (the pending list, the fixtures, three package modules and
  a new harness test): every cell reads as at the previous cut (0 transitions), 213 / 213
  per-file outcome records equal; subject `47170e12428f9ef2` (1,119 files).
- `parity/meep_gpu/rounds/1.0/PRE_CERTIFICATION_STATE.md` gains section 11 (the freeze: the
  stamp, the constants, the re-mint, the spine byte-neutrality re-baseline and the freeze
  criteria); `RUNBOOK.md` records the watchdog and closes the route-constant blocker;
  `docs/development/testing.md` states the re-minted fixture sizes.
- No ledger changes. `meep_gpu/fastpath.py`, `meep_gpu/fastpath_cuda.py` and
  `meep_gpu/metal_dispatch.py` move, and the certification round re-runs every record that
  pins them on these bytes.

### The bit-parity coverage board is re-cut on series entries 01 to 119

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..119; the previous cut was on 01..113, and the board's check had
refused since entry 117 moved its outcome subject. Only three of its generated files change
(`board.json`, `board.md`, `outcomes.json`); the generator is unchanged. Its test outcomes
are the CUDA and Triton test files' cases of the package suite (CPU-only shim, every marker,
dispatch off) and the Metal test files' cases of the device run (Apple GPU), both run on this
tree's outcome subject: 213 test files, 0 of the rows' test files not run, 12,104 cases, the
same per-file outcomes as at entry 113. `dev_evidence.json` is carried unchanged: each of its
11 records is still on its live subject (module bytes, emitted text, array path), since no
entry from 114 to 119 moves a module those records bind.

- **No cell moves** (N / D, refused rows in brackets): (a) the chunk-record select, CUDA
  55 / 55 (+1), Triton 55 / 56 (+2), Metal 60 / 60 (+4); (b) MEEP's A form, CUDA 57 / 57
  (+8), Triton 61 / 61 (+7), Metal 68 / 68 (+9); (c) host word identity, CUDA 18 / 18
  (+47), Triton 26 / 29 (+39), Metal 29 / 29 (+48). No row changes state on any column.
- **The array path against MEEP** reads 6 / 6 families identical (+4 refused) under keep
  and under flush, as at entry 113.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, as before.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates.
- The subjects grow by one file each (outcome subject 1,106 to 1,107 files, board subject
  1,117 to 1,118): entry 119's new harness test, `test_rebind_retirement.py`.

### Pre-freeze fixes: the four items the expansion cleanup left, and measured Triton prices

The owner's ruling of 2026-10-10 deleted the expansion arms and their licence for good. The
residue cleanup ("The expansion licence's last live mentions leave the harness", below) left
four items whose cleanup changes behaviour; each would have tripped one of the certification
round's own tools, so they are fixed before the freeze. No shipped module changes, and no
predicate, kernel, plan or gate decision moves; one package test, `test_triton_weld_contract.py`,
lowers two floors by the retired entry (below).

- **The CUDA fusion board's three emitted-text cells are determined.** The variant tuples of
  `fused_update_P_no_pml_complex[_uniform]`, `update_E_pml_complex_offdiag` and
  `update_E_no_pml_complex_offdiag` passed the deleted arm argument, so every variant raised
  `TypeError` and the three cells read UNDETERMINED. They now call the emitters as they
  ship (two variants each, the duplicates the arm made dropped): the polarization update reads
  POINTWISE, and both off-diagonal electric updates read STENCIL (they read their partner
  components at neighbouring cells).
- **The Metal planner-composition probe runs.** `probe_kwargs()` returned an empty dict and
  the sweep then indexed it, so every run raised `KeyError`. The function and its three
  missing-probe checks are deleted. The `KeyError` had hidden a second staleness: the sweep
  counted every registered arm as a slot's claimant, and a fused product (registered unwired
  on the first slot of the seam it replaces) always co-admits with the arm it welds, so the
  sweep stopped at its first row. Welds are now tallied apart (`ArmSpec.is_weld`, the
  registry's own rule), and the probe sweeps all 110 configurations. Its verdict is FAIL, on
  four cylindrical complex rows whose curl slots refuse by name since entry 36 (Courant 0.37
  is not a float32 word); `test_metal_planner_composition.py` fails on the same four rows
  whenever the evidence archive is present, and skips them without it.
- **The Triton weld rebind retires a deleted gate's entry by name.** `rebind_triton_welds.py`
  drops its mapping for `triton_unified_expansion_device_gate` and lists the key in a new
  `RETIRED_WELDS` table: the campaign pass reports it `RETIRED` and, under `--write`, removes
  the entry from `meep_gpu/triton_kernels/fingerprints.json`, even when nothing else rebinds.
  An unmapped entry used to read `UNCOVERED` ("needs a device re-run") and stay in the ledger.
  It refuses, editing nothing, if the gate file is back in the checkout or a Triton arm still
  cites the key. The certification round's bind performs the retirement; no ledger is edited
  here. `test_rebind_retirement.py` pins it. The two floors of `test_triton_weld_contract.py`
  that count the entry (Triton run records 55 to 54, array-path oracle fingerprints 85 to 84)
  drop by one in the same change: without that, the bound ledger fails nine weld-contract
  tests.
- **The Triton special-kz gate's mutant header** no longer declares the two unused arm
  constants.
- **The round planner's Triton prices are measured walls.** `legs.py` `TRITON_EXPECT` (61
  gates) and `TRITON_HAND_EXPECT` (stencil welds, bit identity) now carry the larger of the
  keep and flush walls of the 1.0 final pre-certification pass. The 0.9.2 prices had
  `special_kz` at 184 s against a measured 2,265 s (keep) and 4,109 s (flush). The planner's
  CERT-3 NVIDIA wall on 4 GPUs moves from 17,195 s to 23,834 s, and `RUNBOOK.md` section 2
  quotes the new plan.

No ledger changes, and none of the 2,270 ledger file pins names a file this change edits.

### The certification reference environments carry the corpus libraries themselves

The corpus overlay (the entry "The corpus environment gains a pinned overlay" below) is
retired. The certification reference
environments now carry what the MEEP identity corpus (Table 2) imports and no field update
reads, so a corpus leg runs in the environment as it is, with nothing beside it on
`PYTHONPATH` or `PATH`. `parity/meep_gpu/rounds/1.0/corpus_overlay-osx-arm64.pip.txt` and
`corpus_overlay-linux-64.pip.txt` are deleted; their pins are now lines of the locks.

- **The locks.** Each platform's pip lock pins shapely 2.1.2 and nlopt 2.11.0, the wheels
  the corpus legs measured with, by SHA-256 (`meep-gpu-ref-osx-arm64.pip.txt` now 15 pins,
  `meep-gpu-ref-linux-64.pip.txt` 24). The PyMieScatt stand-in, which is not on PyPI, is
  built by `parity/meep_gpu/rounds/1.0/build_pymiescatt_standin.py` (new): it fetches the
  PyPI sdist by SHA-256, applies the two lines of `pymiescatt_standin.patch` (new; `trapz`
  from NumPy, since SciPy 1.14 removed `scipy.integrate.trapz`), checks the five package
  files against their pinned SHA-256, builds a wheel with the environment's own setuptools
  (no build isolation, no index) and installs it with `--no-deps`. Its source is not kept
  in the repository. `environments/locks/numerics-relevant.json` gains `corpus_only`, which
  names these packages apart from every numerics set, holds the stand-in's pins, and names
  `ffmpeg` as the one host tool the corpus calls.
- **The build scripts** install them in step 7 (the pip lock, then the stand-in, then
  `pip check`) and then refuse to go on unless the prefix's PyPI packages are exactly the
  pip lock's and the stand-in (`compare_reference_environment.py --require-pip-lock`, new).
  A new step 7b links the host's `ffmpeg` into `<prefix>/bin` when it is not already in
  `/usr/bin` or `/bin` (Homebrew's on a Mac; `MEEP_FFMPEG` names another, and
  `build_meep_133_identity.sh` passes it through); without one the build goes on and names
  the row the corpus will refuse. `ffmpeg` stays a host tool: conda-forge's would bring
  shared libraries into the environment MEEP links against.
- **The drift report judges the numerics exactly as before.**
  `tools/compare_reference_environment.py` lists the corpus-only packages in a section of
  their own (`held`, `absent` or `another build`; the stand-in by its version and the bytes
  of its files, `ffmpeg` by `which` on the reference `PATH`) and never counts them in the
  verdict, so an environment without them, which is all a user who does not re-run the
  corpus needs, reads the same as one with them. On both live reference environments the
  numerics rows were identical before and after the install (Apple 29 of 29 same, Linux 36
  of 36), and the unchanged tool's report differed only in the count of distributions read.
- **Both live reference environments took them from the locks** (2026-10-11; `pip
  install --no-deps --require-hashes` from the lock, a dry run first, then the builder):
  of 43,623 prefix entries on Apple and 46,510 on Linux, 0 were modified (size, mtime or
  SHA-256) and 0 removed; 299 and 297 were added, all in the three packages and their
  metadata and, on Apple, the `ffmpeg` link. Every conda artifact is unchanged (the conda
  list byte for byte, 138 of 138 and 143 of 143 the lock's), every other distribution keeps
  its version, `pip check` is clean, and `--require-pip-lock` reads 16 of 16 and 25 of 25.
- **The corpus rows hold with nothing beside the environment**, re-run on base + series
  01..115 with this change, at `OMP_NUM_THREADS=1`, dispatch off, in keep and flush:
  `ex.waveguide_crossing` 0 / 5,356,800 stored and 0 / 5,356,800 reader words differing on
  both hosts; `test_animation_output` 0 / 4,761,600 and 0 / 4,761,600 on Apple;
  `ex.mie_scattering` 0 / 7,143,840 and 0 / 26,671,680 on both hosts; and
  `ex.differential_cross_section` 0 / 41,140,224 and 0 / 43,765,428 on Linux; control
  `ex.holey-wvg-bands` 0 / 867,600 and 0 / 1,728,000 on both. Every row's digests, plane
  digests and currents equal Table 2's record (16 of 16 row-modes), its planted flip was
  caught and its FPU check held at every compared step, and its subnormal stamp equals the
  single stamp of the other 186 rows of its reference. Importing and calling shapely,
  nlopt and the stand-in, each resolved from the environment's own `site-packages`, moved
  no subnormal flag (4 of 4 host-mode probes). Table 2's page changes only in its
  provenance paragraph (the table's JSON is byte-identical).

### The expansion licence's last live mentions leave the harness

Entry 89 deleted the complex-multiply expansion licence, and the owner's ruling of 2026-10-10
makes that final. What still named it in the harness modules the gates, the fusion boards and
the census read is removed or corrected here. No predicate, kernel, plan or gate decision
moves, and no arm is admitted or refused differently.

- **Dead code removed:** `build_fusion_matrix._EXPANSION`; the unused `PROBE` records in
  `complex_no_pml_gate_common.py` (with the two no-PML gates' aliases) and
  `gate_metal_complex_conductive_pml.py`; `replay_complex_cylindrical_slots.COMPLEX_POLICY`
  and `COMPLEX_LICENCE`; the unreachable missing-probe raise in
  `gate_metal_whole_step.probe_records`; the `"expansion probe artifact"` fragment in
  `gate_metal_cylindrical_real.BACKEND_ONLY_FRAGMENTS`, which no refusal in the tree contains;
  and the Triton board's staleness audit of the deleted `complex_fields._expansion_reasons`.
- **Recorded texts corrected:** the `not_evaluable` clauses of seven Metal board products that
  named a probe artifact or an arm binding. The Metal dry run's intents for `bloch_2d`,
  `folded_complex_2d`, `complex_nobloch_2d` and `cylindrical_m1`, and the Metal route gate's
  `cylindrical_m1` reason, no longer say those cases need a probe: they dispatch on the shipped
  leg. Also corrected: one reachability reason, one does-not-claim line, two gate record texts
  and one mutation description.
- **Comments and docstrings** that stated the licence as current behaviour are corrected.
- **Left, each for a reason:**
  - The CUDA board's `fused_update_P_no_pml_complex[_uniform]`, `update_E_pml_complex_offdiag`
    and `update_E_no_pml_complex_offdiag` variant tuples still pass the deleted arm argument.
    Their emitters raise, so those cells read UNDETERMINED, and fixing them would change a
    verdict.
  - The Metal planner-composition probe's probe checks stay: the probe raises `KeyError` before
    it reaches them.
  - The weld rebind tool keeps its mapping for the Triton ledger entry
    `triton_unified_expansion_device_gate`, which leaves with that entry.
  - The Triton beta gate's mutant header keeps two unused arm constants, because removing them
    moves every mutant's line numbers.
  - Dated records, measurements made while the arms existed, and correct post-deletion
    statements stay as written.

No ledger changes. 30 file pins name a changed file, all of them already drifted on the base;
the certification round's re-gate re-pins them.

### The pre-certification state of the 1.0 series

`parity/meep_gpu/rounds/1.0/PRE_CERTIFICATION_STATE.md` records the final pre-certification
pass on series entries 01 to 115 (tree `87febcc8`): **no class (ii) regression and no class
(iii) harness item on any table**. The pass ran the round's own leg table and step code and
wrote no ledger, fingerprint, census or record.

- **NVIDIA**: CUDA gates 118 / 118, Triton fleet 61 / 61 under keep and 61 / 61 under flush,
  stencil welds and bit identity 6 / 6, route legs 6 / 7 (CUDA) and 3 / 4 (Triton), the
  end-to-end licence 4 / 4. 249 legs ran on entry 111's tree and 2 on entry 114's, each
  reused only where every file it binds is byte-identical on entry 115's, checked per leg;
  the 10 legs whose artifacts record no digests re-ran on entry 115's tree.
- **Metal**: fleet 61 / 62 and held route legs 4 / 4, all run on entry 114's tree, because
  every Metal gate digests `meep_gpu/subnormal_policy.py`, which entry 111 moved; entry 115
  moves no file they bind (6,490 / 6,490 bound-file hashes identical).
- **The non-releases are the certification round's own (class (i))**: the 0.9.2 route stamps
  (CERT-1); the two flush route legs, whose every case passes but which cannot release
  before the bind writes per-arm flush evidence; and the spine byte-neutrality gate's frozen
  baseline (not released; owner ruling pending: re-baseline at CERT-1 or retire).
- The document also carries the board's headline, the pending list by owner (63 entries, all
  the certification round's), Table 2, the owner rulings R1 to R16 and the open items the
  round owns, and states for every leg the tree it ran on. Documentation only.

### The corpus environment gains a pinned overlay, and Table 2 holds 189 of 194 rows on all four references

Three of Table 2's refusals by name were the environment's, not the package's:
`ex.waveguide_crossing` on both hosts (no nlopt), `ex.mie_scattering` and
`ex.differential_cross_section` on Linux (no shapely, which the PyMieScatt stand-in
imports), and `test_animation_output` on Apple (no `ffmpeg` on the leg's `PATH`). These
rows were measured with a corpus overlay beside the reference environment: shapely 2.1.2
and nlopt 2.11.0, one wheel each by SHA-256, installed with `pip install --no-deps
--only-binary=:all: --require-hashes --target <overlay>` and appended to `PYTHONPATH`, the
PyMieScatt stand-in (PyMieScatt 1.8.1.1's source with `trapz` taken from NumPy) on
`PYTHONPATH`, and `ffmpeg` on `PATH` through a directory holding only an `ffmpeg` link. The entry "The
certification reference environments carry the corpus libraries themselves" retires the
overlay: the reference environments now carry these packages themselves, and the rows
reproduce in them with nothing beside the environment.

- **The five row-references hold**, measured on base + series 01..106 in both modes at
  `OMP_NUM_THREADS=1`, dispatch off (stored / reader words differing, every planted flip
  caught): `ex.waveguide_crossing` 0 / 5,356,800 and 0 / 5,356,800 in each of the four
  references; `ex.mie_scattering` 0 / 7,143,840 and 0 / 26,671,680 on Linux keep and
  flush; `ex.differential_cross_section` 0 / 41,140,224 and 0 / 43,765,428 on Linux keep
  and flush; `test_animation_output` 0 / 4,761,600 and 0 / 4,761,600 on Apple keep and
  flush.
- **The overlay moves no FPU state.** Every re-run row's subnormal stamp (the four measured
  flush and denormal flags and the mechanism) equals the stamp of every row measured in its
  reference before (188 of 188 on Apple, 187 of 187 on Linux, per mode), and its FPU check
  held at every compared step. Importing and calling shapely, nlopt and the stand-in moved
  no flag in either mode on either host.
- **Controls.** `ex.holey-wvg-bands` (2,996 subnormal stored words in keep) on both hosts
  and `ex.mie_scattering` on Apple (now importing the overlay's shapely, whose 133 files
  are byte-identical to the copy measured before), re-run with the overlay attached on
  entries 01..106: MEEP's digests and plane digests equal to the 01..86 rows' in 6 of 6
  (host, mode, row) pairs; `ex.holey-wvg-bands`' capsule differs only in the provenance
  hashes of the two modules entries 87..106 edited (62 of 63 members identical, in each of
  the four references).
- **Table 2** ([the page](docs/development/meep-identity-corpus.md)) regenerated from a
  Tier B record of the merged cut (the 01..86 rows with the re-run rows over them): 57 of
  60 examples and 132 of 134 tests hold in each of the four references; refused by name on
  both hosts, in both modes: `ex.coupler` (no libGDSII), `ex.ring_gds` (no gdspy),
  `test_infer_dimensions` and `test_require_dimensions` (a BinaryPartition chunk layout with
  mirror symmetry is not lifted). The 91 holding rows with a capsule replay with no MEEP on
  entries 01..106 in every set (the re-run rows carry monitors, which the capsule schema
  does not hold). L0, the two reference MEEPs' digests on the rows run on both hosts (now
  190 of 194): keep, equal on 161 and different on 29; flush, 158 and 32. Of the four rows
  that now run on both hosts, three are equal; `ex.waveguide_crossing` differs from step 1,
  an eigenmode source (each host's MPB solve), and on every differing row (29 of 29, 32 of
  32) the source currents the two lifts recorded differ too.

### The bit-parity coverage board is re-cut on series entries 01 to 113

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..113; the previous cut was on 01..105. Only three of its
generated files change (`board.json`, `board.md`, `outcomes.json`); the generator is
unchanged. Its test outcomes are the CUDA and Triton test files' cases of the package suite
(CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the device run
(Apple GPU), both run on this tree's outcome subject: 213 test files, 0 of the rows' test
files not run. `dev_evidence.json` is carried unchanged: each of its 11 records is still on
its live subject (module bytes, emitted text, array path), since no entry from 107 to 113
moves a module those records bind.

- **The kernel tables do not move** (N / D, refused rows in brackets): (a) the chunk-record
  select, CUDA 55 / 55 (+1), Triton 55 / 56 (+2), Metal 60 / 60 (+4); (b) MEEP's A form,
  CUDA 57 / 57 (+8), Triton 61 / 61 (+7), Metal 68 / 68 (+9); (c) host word identity,
  CUDA 18 / 18 (+47), Triton 26 / 29 (+39), Metal 29 / 29 (+48). No row changes state on
  (a), (b) or (c). Three Metal test files carry entry 113's cases, every one passing:
  `test_metal_conductive_pml.py` 16 -> 17 passed, `test_metal_conductive_fused_hd_pair.py`
  36 -> 37, `test_metal_kernels.py` 60 -> 62 (its one failure, the checked-in fingerprints,
  is unchanged and declared).
- **The array path against MEEP under flush reads 6 / 6 families identical (+4 refused)**,
  from entry 111's Tier A flush set (8 minted cases on the Linux reference); it read 0 / 0
  (+10 refused), "the record holds no flush set", at entry 106. Under keep it reads 6 / 6
  (+4 refused) as before.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, as at entry 106.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates.

### Metal route coverage: the conductive presence words, the first update_H on a 2-D magnetic run, and the cliff control's composition

- **The Metal conductive PML curl serves a per-chunk conductivity presence.** MEEP
  deletes a chunk's conductivity array where every sample it visits is zero and steps
  the lossless forms there. `metal_kernels/conductive_pml.py` binds one 27-bit word per
  target (`ConductivePresence`, buffer 30, 31 of 31) and selects the lossless forms on a
  lossy target's cell whose chunk keeps none, leaving `f_cond` alone; the conductive
  fused electric and H->D pairs carry the words in `Params` after the chunk record. The
  words are built from `RegionPlan.conductive_mask` through the off-diagonal families'
  chunk-code twin and checked to expand back to it on every stored cell, or the
  sub-step is refused by name. Before, every partial presence was refused, and under
  held residency that refusal left `step_D` on the array path and refused the whole
  composition (`conductive_2d` on the Metal route, 0 launches a step from entry 37).
- **The base `update_H` kernel takes a 2-D run's first `update_H` when the start cannot
  move a word.** The composer's `launch.h_start_reasons` now receives the layer and
  refuses a magnetic source only where its B component's H carries cells in its own
  layer, the only cells `stepping.separate_h_from_b` copies; an `Hz` source on a 2-D xy
  run deposits into a B the start never copies. The refusal is run-long (the plan is
  frozen at the first step), so under held residency it had refused the singles route
  of every magnetic-source case (`magnetic_seam_2d`, `offdiag_magnetic_2d`,
  `folded_offdiag_magnetic_2d`: 0 launches a step on all 47 measured steps). The fused
  and H->D products still ask without the layer.
- **The Metal route gate's subnormal-cliff control composes what the engine runs.**
  `probe_metal_dispatch_dryrun.compose_metal_step` hands the composer the driver's own
  chunk division, and `DryRunShim` applies the absent components' clear after
  `update_E` as `FastPathPlan` does; an array over a band ceiling is licensed when the
  array path stepped on the host with the FPU flushing reproduces it word for word
  (`replay_licensed`), and keeps its ceilings otherwise.

### The hosted test workflow's undeclared failures: resource skips, one test-pollution defect and one interpreter-dependent assertion

The hosted `tests` workflow (`.github/workflows/tests.yml`) failed its "Every failure is a
declared one" step on every push of the series: 45 undeclared failures on Python 3.10 and
44 on 3.12 at entry 106, beside the 62 declared ones. Each is now a skip by a declared
resource, or a repaired test; `tools/ci/pending_certification.txt` is unchanged.

- **A test that left a stand-in `meep` in `sys.modules`.** The `device_free` fixture of
  `parity/meep_gpu/test_composition_policy.py` serves an empty `meep` module through a
  meta-path finder and removed the entry with `monkeypatch.delitem(..., raising=False)`,
  which records nothing when the key is absent. The stand-in stayed for the rest of the
  harness session, so every later `import meep`, and `importlib.util.find_spec("meep")`,
  found a module with no `Vector3`, whether or not MEEP was installed: 7 tests failed on
  `AttributeError` in the hosted jobs without MEEP and 21 in the job with it. The fixture
  now records the entry as it was and restores exactly that.
- **An assertion that depended on the interpreter's spelling.** `meep_gpu/cuda_kernels/
  test_ade_update_p.py` compared `ast.unparse` output with literal statements; Python 3.10
  writes a tuple target as `(a, b) = f()`. The expected statements now go through the same
  parse and unparse.
- **Skips that now name their resource** (test ids of the 45): 24 Metal plan tests build
  a plan whose chunk record is a `torch.int32` tensor and are listed under `[torch]` in
  `tools/ci/declared_resources.txt` (23 entries); 4 route-gate tests that lift through
  MEEP are listed under `[meep]` (3 entries), and the 2 `[meep]` entries of
  `test_h_to_d_seam` renamed with their tests follow the new names. 10 skips that carried
  no resource now carry one (`mps_device` 7, `torch` 2, `darwin_libm` 1), and 3 tests whose
  premise is a host FPU driven to flush skip by `host_fpu_mode` where the drive is measured
  to fail. The 45th, `test_timing_cases`'s NumPy-lift test, skips by `meep` again once
  `find_spec` is no longer answered by the stand-in.

### The host reaches flush and keep without MEEP on x86-64 Linux, and the Tier A flush set is minted

`meep_gpu/subnormal_policy.py` gains a third host lever, `_drive_host_fpu_by_mxcsr`, for a
process that has not imported MEEP on x86-64 Linux with glibc: it reads the thread's
environment with `fegetenv` into a mirror of glibc's x86_64 `fenv_t` (`bits/fenv.h`:
`offsetof(fenv_t, __mxcsr)` 28, `sizeof` 32, read off the header by the compiler on the
Linux reference host, glibc 2.31), sets (flush) or clears (keep) MXCSR FTZ|DAZ (`0x8040`,
the bits MEEP's own `_set_zero_subnormals` sets), installs it with `fesetenv`, and measures
with `backends.subnormals_flushed()`; a measurement that disagrees puts the saved
environment back. It runs only where `glibc_x86_64_abi()` holds (Linux, x86-64, 64-bit
pointers, glibc's `gnu_get_libc_version` exported), writes nothing when the word it reads
has a reserved MXCSR bit set, and reaches the calling thread (a thread created afterwards
inherits it; one already running keeps its own). MEEP's switch stays first whenever MEEP is
imported, and the lever is never tried then; nor is it tried when the thread is already in
the requested state, so a keep install on an x86 host that keeps writes nothing and records
exactly what it recorded before (mechanism `none (MEEP is not imported in this process)`,
`installed` False). On any other system without MEEP, `flush` is refused by name, naming the
system. The mechanism is stamped
`fenv:x86_64-glibc MXCSR |= 0x8040 (FTZ|DAZ)` (or `&= ~0x8040` for keep).

- **Tier A flush set minted** with `parity/meep_gpu/record_meep_identity.py` (unchanged)
  through `meep_identity_tier_a.py` on the GPU host against the Linux reference (MEEP 1.33.0
  single precision, MEEP's flush on import, `OMP_NUM_THREADS=1`): 8 of 8 cases, 39,991,116
  stored and 65,610,420 reader words equal to MEEP's, the replay (MEEP blocked) driven to
  flush by the new lever and reproducing every digest, 2,316,850 bytes; with the keep set
  4,633,666 of the 5,000,000-byte budget (ruling R6). The keep leaf is unchanged.
- **Replays** with no MEEP importable, every stored and reader digest of every row: x86-64
  Linux 8 of 8 flush rows (under the lever) and 8 of 8 keep rows; Apple silicon 8 of 8 flush
  rows (under FZ, judged per row as ruling R13 allows) and 8 of 8 keep rows. The Tier A test
  file passes on x86-64 Linux with no MEEP importable entering in keep and entering in flush
  (the lever clears FTZ|DAZ for the keep rows), with NumPy's AVX-512, AVX2 and FMA3 loops
  disabled, and with MEEP imported (its own switch drives).
- `meep_gpu/meep_identity.py`: the refusal and the module text name the three levers. The
  Tier A test file skips a flush row only where no lever reaches flush in the process (resource
  `host_fpu_mode`), runs the curl-mutation test in keep, and gains a test that flush is reached
  without MEEP where a lever exists. `test_subnormal_policy.py` gains ten tests: the
  layout pin, flush and keep attained and measured on x86-64 glibc, the calling-thread
  scope, the restore on a disagreeing measurement (real FPU and a stand-in C library), the
  reserved-bit guard, the refusal on another system, that the lever declines off its ABI
  without resolving the C library, that an install already in the requested state writes
  nothing, and that the lever is not tried with MEEP imported.
- Docs: the floating-point contract's subnormal section (a table of the host levers),
  the dispatch guide's subnormal-policy section, the MEEP-identity section of the testing
  guide, and the certification guide's flush note.

### The expansion licence's residue in documents, test text and the declared-resource list

The licence was deleted on all three tables, but some text and one configuration list still
described it as present. This entry corrects the ones no gate imports and no ledger pins. No
package module, gate, needle, mutation or ledger changes.

- **`docs/development/certification.md`, the Metal route campaign.** The two commands and
  the round's launch line passed `<probe-root> <complex-probe>`, which
  `run_metal_dispatch_campaign.sh` no longer takes: its usage is `<stamp> [lanes]`, so the
  extra words were read as lane names. The page also said the campaign has five legs. It has
  four, in launch order `band_witness`, `harness_keep`, `shipped` and `harness_flush`.
- **`tools/ci/declared_resources.txt`.** Six entries named tests that were deleted with the
  licence, so they matched no collected test. They are removed. Three older unmatched
  entries for `test_metal_folded_beta` were deleted by another change and are left as they
  are.
- **Test text.**
  - `test_triton_complex_fused_hd_pair.HOST_CLAUSES` no longer subtracts the deleted
    "expansion probe artifact" clause.
  - Two stub refusals in `test_dispatch_contract.py` quote clauses the predicates still
    emit.
  - Docstrings in eight test files no longer describe the probe, the arm or the licence as
    present: three Triton pair tests and one Metal pair test, the CUDA complex H->D and
    released-arms tests, `test_triton_complex_fields.py`, and the no-device legs test of
    the Triton folded complex magnetic pair, which also listed a leg the gate no longer
    has.
  - An unused helper with the deleted arms is removed from
    `test_triton_complex_no_pml_stored_e.py`.
  - The planner test's empty C9 section is marked retired.
  - `meep_gpu/conftest.py`'s comments no longer name the expansion-probe variables.
- **"0.10" reads "1.0"** in Table 2's provenance line
  (`docs/development/meep-identity-corpus.md`). The page was regenerated by
  `build_meep_identity_table.py` from the same three records, and that phrase is its only
  change.

### The Triton cylindrical gate's divide mutant compiles under the flush policy

The 1.0 pre-certification dry run stopped `gate_triton_cylindrical.py` under `flush` at its
fifteenth mutation, `s2_divide_not_multiply`: the policy's PTX audit refused the mutant at
compile (2 of 12 audited f32 instructions, `div.full.f32`, carried no `.ftz`), so the gate
died before its last four mutations and wrote no verdict. The division was in the mutant,
not in a shipped kernel: the shipped radial scan multiplies by `rinv`, as MEEP does, and
every shipped-kernel leg had already passed under `flush`. The mutant spelled its division
as a plain `/`, which Triton emits as inline `div.full.f32` that neither subnormal policy
reaches.

- **The mutant is respelled** `tl.math.div_rn(here - prev, tl.load(w + g))`: the same
  divide-for-multiply defect, round-to-nearest under both policies (`div.rn.f32` under
  `keep`, `div.rn.ftz.f32` under `flush`). It stays must-catch. No shipped kernel, coverage
  predicate or weld record changes.
- **Both cylindrical gates stamp the policy at the end of the run** as
  `subnormal_policy_end`, so the artifact carries the PTX audit's counters over every
  kernel the run compiled; the start-of-run stamp is taken before anything compiles.
  Recorded only: no verdict clause reads it, and no weld reader looks for it.
- **New host test** `parity/meep_gpu/test_gate_triton_cylindrical_host.py`: every mutation
  of both cylindrical gates still matches the shipped kernel text and parses, and neither
  a mutant nor a shipped cylindrical kernel spells a plain float division.

### The pre-certification pass's NVIDIA and round-driver refusals are fixed

The final pre-certification pass on series entries 01 to 106 left 32 of 118 CUDA gate legs and
16 of 61 Triton flush fleet gates unreleased for reasons in the gate and round code. This entry
fixes those reasons.

- **The expansion licence's residue (entry 89).**
  - `gate_cuda_fused_complex_pairs.py` stores its lift blocks under `blocks`, the key entry 89
    renamed. The `wrong_arm` host mutation, its liveness rule and the real families' `licensed`
    and `lift_arms` keys are gone: the mutation had no patch branch left and raised on every
    family once the lift passed.
  - The previous-form mutant in `gate_triton_complex_ade.py` follows the shipped three-argument
    helpers. It installs the same old grouping as before (KERNELS.md section 7.1).
  - `gate_triton_complex_no_pml_conductive.py` now expects no declared nulls. Entry 89 deleted
    its two.
- **The flush host-import wrapper** (`rounds/1.0/meep_host_import.py`) imports only MEEP before
  the gate runs, and refuses to hand over if CuPy is already imported. It used to import the
  bit-identity probe, which imports CuPy, so the three ADE-chain gates refused.
- **The campaign's policy.** `gate_triton_complex_ade.py`, `gate_triton_complex_no_pml_stored_e.py`
  and `probe_triton_cylindrical_fused_magnetic_pair.py` read `MEEP_GPU_SUBNORMAL_POLICY`
  through `composition_policy.requested_policy`, as the composition gates do (`keep` when it
  is unset).
  - The cylindrical probe cuts one policy per process. MEASURED: its earlier two-policy loop
    returned the memoised keep install for the "flush" cut, on all 25 of its artifacts in
    `results/`.
  - The board's verdict-prose reading (`build_triton_fusion_matrix.verdict_prose_policy_reasons`)
    now refuses a stamp that resolved to a policy other than its label.
- **Flush not-applicable, by measurement.** `parity/meep_gpu/flush_vacuity.py` declares a
  subnormal-operand floor not applicable only when three measurements agree: flush is
  installed, the host flushes at seed time, and the class's census is 0. Eight Triton gates and
  the cylindrical probe consult it. Under keep every floor is unchanged.
- **The MEEP corpus for the CUDA H->D lifts.** `run_lane.py`, `round.py` and `legs.py check` take
  `--corpus-root`, resolved the way the lifting gates resolve it. A lane whose plan lifts
  refuses before it takes a GPU when the root has no `examples/` and `tests/`, and exports
  `MEEP_GPU_CORPUS_ROOT` to every child. The RUNBOOK names it in the preflight and the CERT-3
  launch.

### Three Metal complex `update_E` gates arm their mutations again after the expansion licence's deletion

Entry 89 removed the expansion element from the function maps of the Metal complex
dispersive and no-PML stored-E plans (now keyed `(mode, poles, axis)` and
`(mode, poles)`), but `gate_metal_complex_dispersive_update_e.py` and
`gate_metal_complex_no_pml_stored_e.py` still keyed their mutant maps with a trailing
`"FMA_V1"`. The builders compile the shipped body for every key a map does not hold, so no
mutant reached a launch: 7 of 7 and 3 of 3 must-catch source mutations read 0 words and
both gates failed. `gate_metal_complex_no_pml_offdiag_update_e.py` still read the deleted
`plan.expansion` and raised before its first row. The maps are keyed as the plans read
them, the unused `PROBE` constants are deleted, and the offdiag gate reads its case's
specialisation without the expansion. No package code changes.

- **Measured on an Apple M1 Max** (flush, dispatch off, residency `shipped`), each
  gate released: the dispersive gate 3 / 3 products at 0 words, 9 / 9 must-catch mutations
  caught (7 source, 2 record; 99 to 2,170 words) and its declared null
  `auxiliary_load_unconditional` at 0 words; the stored-E gate 5 / 5 products at 0 words
  and 3 / 3 mutations caught (3,780 words each); the offdiag gate 5 / 5 products and
  2 / 2 shipped-source controls at 0 words, 6 / 6 mutations caught (26 to 756 words), and
  its engine row. On base + series 01..106 the same runs read 2 / 9 and 0 / 3 caught, and
  the offdiag gate's `AttributeError`.
- **`parity/meep_gpu/test_gate_metal_complex_e_mutants_host.py`**, torch-free: each gate's
  mutation case is built through the engine builder with the gate's own map and run once
  on host arrays, with the compiler replaced by a recorder; the builder must compile
  nothing of its own and the mutated body must be the one launched (Ex's two-pole body;
  the two-pole body Ex and Ez share; the offdiag case's own specialisation). 3 of 3 pass,
  and 3 of 3 fail on base + series 01..106, each naming the cause.

### The bit-parity coverage board is re-cut on series entries 01 to 105

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..105; the previous cut was on 01..92. Only its four generated
files change. Its test outcomes are the CUDA and Triton test files' cases of the package
suite (CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the
device run (Apple GPU), both run on this tree's outcome subject: 213 test files, 0 of the
rows' test files not run. `dev_evidence.json` holds only the device legs re-run at the
integration of entries 94 to 105.

- **The kernel tables** (N / D, refused rows in brackets; the cut on 01..92, then this
  one): (a) the chunk-record select, CUDA 55 / 55 (+1) -> 55 / 55 (+1), Triton
  45 / 56 (+2) -> 55 / 56 (+2), Metal 60 / 60 (+4) -> 60 / 60 (+4); (b) MEEP's A form,
  CUDA 57 / 57 (+8) -> 57 / 57 (+8), Triton 52 / 61 (+7) -> 61 / 61 (+7), Metal
  66 / 68 (+9) -> 68 / 68 (+9); (c) host word identity, CUDA 18 / 18 (+47) ->
  18 / 18 (+47), Triton 25 / 38 (+30) -> 26 / 29 (+39), Metal 28 / 30 (+47) ->
  29 / 29 (+48).
- **What moved.** Triton (a) and (b): the ten H->D welds land with entry 102, which re-cut
  them onto the forms (on (b) nine, `cylindrical_fused_hd_pair` having landed already).
  Triton (c): `cylindrical_real_fused_hd_pair` lands; the other nine H->D rows move from
  gap to refused, because (c) judges a row once its (a) lands and none of their test files
  compares the arithmetic with `meep_gpu.stepping`, so D falls from 38 to 29. Metal (b):
  `no_pml_conductive` and `no_pml_conductive_fused_electric_pair` land with entry 101
  (MEEP's stencil); on (c) `no_pml_conductive` lands and the pair is refused by the same
  rule. The array path against MEEP under keep reads 6 of 6 families (2 of 2 at entry 93),
  from entry 104's Tier A rows. No other cell changes state.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, re-run on this tree's module bytes, emitted texts and array path.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates.

### The final integration's own changes: the round drivers after the expansion licence, the retired census keys, two H->D harness repairs and one launch the licence deletion left behind

The changes the integration that closes the 1.0 development series makes beside the work
packages it integrates. No needle, mutation anchor or array-path module changes; the
array-path oracle is unchanged (`array_path_oracle/real-v6`, fingerprint `1ab51627`, 20 of 20
case digests).

- **The certification round drivers after the expansion licence.** The drivers under
  `parity/meep_gpu/rounds/1.0/` (moved from `rounds/0.10/`: the release is 1.0) were cut while
  the expansion-licence chain was still in the tree and carried it as optional, read off the
  tree. On the series it is gone, and two of its paths broke: `legs.py check` compared a route
  leg list that still held the two deleted probe legs, and `metal_round.py` cut two Metal
  expansion probes whose scripts are deleted, ran a probe weld through a runner flag that is
  deleted, and handed `run_metal_dispatch_campaign.sh` two positional arguments its usage no
  longer takes. The chain is removed outright: the probe column of the leg tables, the archive
  inputs, the expansion-records stage and `plan probes`, the probe placeholders and
  `estimate --chain`; `round.py` loses its probes stage and `--skip-probes`; `run_lane.py`
  hands no `--probe`; `metal_round.py` cuts no probe and runs no probe weld, runs
  `gate_metal_complex` in the fleet like every other gate, and its dry-run flag is
  `--skip-environment` (it skips only the environment record). The cleanup leaves every plan
  the drivers print unchanged but the two route preflights, which now write
  `<campaign>_preflight/cuda_shipped` and `<campaign>_preflight/shipped`: the route gate reads
  its leg from the directory leaf, and the old leaves named the deleted probe legs.
  `legs.py check` now refuses on the two route stamps alone (CERT-1). The Triton estimate
  prices `conductivity` and `symmetry` at the walls this integration measured (5,148 s and
  4,937 s, both released, on a card other gates shared) instead of the fleet driver's old
  timeouts, which moves only the lane packing (the planned wall 16,479 s -> 17,195 s);
  `fleet_launch.py` states that its default floors raise nothing since the fleet driver gives
  both gates 10,800 s.
- **The retired `@unified_expansion` census keys.** `triton_predicate_battery.py` asks the
  five predicates once; the second key carried the verdict under the unified expansion record
  and was the same call since the licence was deleted. `build_triton_fusion_matrix.py` reads
  the plain keys, drops `UNIFIED_CREDITED_PRODUCTS` and the instance columns of the licence
  swap (`credited_under_a_different_expansion_licence`, `same_product_under_the_census_probe`,
  `complex_chain_under_the_census_probe`), and refuses by name a census that still carries a
  retired key (CERT-4 re-cuts the census on these bytes).
- **`parity/meep_gpu/stress_cuda_breadth.py` is deleted.** Its subject was the expansion arm
  and the licence; its row-mask sweep duplicates `gate_cuda_folded_offdiag_rowmask.py`.
- **The Triton folded H->D gate's launch accounting follows the two-pass mirror fill.**
  `symmetry.MirrorGhostFillPlan.run` is its near pass (one launch per folded axis) and then
  its far pass (one launch per folded axis whose entry declares a far plane);
  `gate_triton_folded_fused_hd_pair.py`'s `launch_structure` leg counted one launch per axis
  and failed on every tree since the two-pass fill. The expected count is read off the plan's
  own axes table.
- **The Triton H->D gates' shims apply the clear the shipped dispatch owes after `update_E`**
  (`fastpath.AbsentComponentClear`), as `gate_metal_folded_fused_hd_pair.Shim` does:
  `gate_triton_fused_hd_pair.Shim` (also the shim of the complex, tail-kit and cylindrical-kit
  gates) and `gate_triton_folded_fused_hd_pair.Shim`.
- **`cuda_kernels/complex_beta_fused_hd_pair.py`'s launch compiles its one body.** The licence
  deletion removed the arm from the weld's state and from `_get_kernel`, and left the launch
  reading `state["arm"]`: `gate_complex_beta_fused_hd_pair.py` stopped in its product leg on
  `KeyError('arm')`.
- **The route gate's admitted flush leg covers its one singles row.** Entry 94's branch for a
  case refused at rung 8b for missing per-arm flush evidence was written for dispatch rows;
  `folded_offdiag_dispersive2_2d`, the no-fused-arm row that names the CUDA singles it exists
  for, is a dispatch case of those singles and read SINGLES-NOT-DISPATCHED on `cuda_flush`
  with its plan refused by name and its run byte-identical. It now passes as
  PASS-POLICY-REFUSED on that leg until its arms carry the evidence, and is judged by its
  singles after.
- **The pending list's emptied groups move to a History section.** The board reads a block of
  `tools/ci/pending_certification.txt` as the comment lines since the previous id and the ids
  after them, and a block whose comment names a kernel lane as ids that wait on that lane.
  Entry 102 removed the last ids of four groups that led their blocks and kept the paragraphs
  as history; with no id between, those paragraphs, which name the lanes that added or emptied
  the groups, joined the next group's block. 6 ids that wait on the certification round then
  read as waiting on kernel lanes, and the board's Triton `conductivity` and
  `folded_fused_magnetic_pair` rows moved from landed to gap on host word identity for that
  reason alone. The four leads and groups, and one paragraph emptied at an earlier integration
  that stood before one of them, move word for word after the list's last id, under History;
  each block that lost its lead keeps a one-line pointer. No id moves (63 before and after),
  and 0 of the 62 ids the list shares with entry 93's change between waiting on the round and
  waiting on a lane (6 did before this change).
- Text that stated the deleted licence as present behaviour:
  `docs/development/kernel-tables.md` (two sentences), `docs/design/floating-point.md` (the
  complex-arithmetic paragraph), `benchmark_gpu.py`'s docstring, and the two Triton
  cylindrical H->D gates' `m_withdraw_after_step_D` texts (the outcome is the seam leg's null
  controls). "0.10" reads "1.0" where the release is meant (this file,
  `docs/development/certification.md`, the round drivers and their runbook).

### Table 2 is re-cut as MEEP identity on the corpus, Tier B is minted, and the Tier A keep set gains four family rows

Table 2's 194 corpus rows (60 examples, 134 tests) were run through Tier C on four
references at `OMP_NUM_THREADS=1`, dispatch off, every step 0 to 20, Table 2's ten
checkpoints and its N compared, on the array path of base + series 01..86: Apple silicon
keep and flush (MEEP 1.33.0 single precision built with `-ffp-contract=off`; flush under
FPCR.FZ) and Linux keep and flush (MEEP 1.33.0 single precision on x86-64, no fused
multiply-add in the build). The new page
[Table 2: MEEP identity on the corpus](docs/development/meep-identity-corpus.md) is
generated from those records alone.

- **L1, the array path against each reference** (rows holding "0 of W / 0 of V" at every
  compared step, of 194): 189 in each of Apple keep, Apple flush, Linux keep and Linux
  flush, five row-references of them measured on entries 01..106 with the corpus overlay
  (the entry above); 0 rows differ in any reference apart from the one named exclusion,
  `ex.stochastic_emitter` (its source draws `np.random.randn()` unseeded on every call, so
  MEEP and the package step different source series: 438,094 / 8,118,000 stored words differ
  in Apple keep). Words held: 2,728,743,678 stored and 3,997,904,328 reader in each of the
  four references.
- **Not run, by name** (the case did not build or lift; the same on both hosts, in both
  modes): `ex.coupler` (no libGDSII), `ex.ring_gds` (no gdspy), `test_infer_dimensions` and
  `test_require_dimensions` (a BinaryPartition chunk layout with mirror symmetry is not
  lifted). The first cut's three environment refusals (`ex.waveguide_crossing`, no nlopt;
  `ex.mie_scattering` and `ex.differential_cross_section` on Linux, no shapely;
  `test_animation_output` on Apple, no ffmpeg) are closed by the corpus-only packages
  (measured with the corpus overlay, which the entry "The certification reference
  environments carry the corpus libraries themselves" retires).
- **L0, the two reference MEEPs against each other** (MEEP's digests, rows run on both
  hosts): keep, every digest equal on 161 of 190 rows; flush, 158 of 190. On every row where
  the two MEEPs differ (29 of 29 in keep, 32 of 32 in flush) the source currents the two
  lifts recorded from their own MEEP differ too: eigenmode sources (each host's MPB solve),
  the random source, and source time functions of cylindrical and Bloch rows evaluated with
  each host's math library. Each row is judged against its own host's MEEP.
- **Context, not a verdict**: the whole-field relative L2 of the array path against the
  certification MEEP of the Mac lock (contracted) at N, now MEEP's own build-to-build
  spread: 189 of the 193 rows that are not excluded measured (the 4 refusals above are not
  run), from 0 to 5.568e-05, median 5.150e-07;
  it is at or below the L2 Table 2 printed for the 0.9 array path on 184 of the 189.
- **Tier B** (`parity/meep_gpu/record_meep_identity_archive.py`, new): per reference, every
  row's Tier C verdict by the Tier A minting criteria, MEEP's digests and plane digests, and
  the capsule and currents where the schema carries the row, archived with their sha256 in
  the evidence archive; `test_meep_identity_tier_b.py` reads it back (`evidence_archive`).
  The 91 holding rows with a capsule replay with no MEEP in every set: Apple keep 91 / 91,
  Apple flush 91 / 91 (under FZ), and the Linux keep and flush capsules replayed on Apple,
  91 / 91 each. MEEP's words are not archived.
- **Tier A keep set** re-minted through the new cases module
  `parity/meep_gpu/meep_identity_tier_a.py`: the four probe cases (every stored, reader and
  plane digest unchanged) and one row per family, `ex.material-dispersion` (dispersive),
  `test_fields_at_kx` (complex storage, Bloch, off-diagonal epsilon), `test_ring_cyl`
  (cylindrical, m = 3) and `ex.absorber-1d` (conductive); 8 of 8 cases hold, 39,991,116
  stored and 65,610,420 reader words, 2,316,816 fixture bytes of the 5 MB budget. **No flush
  set is minted**: the minting tool's replay cannot import MEEP, and on x86 nothing but MEEP's
  switch reaches flush, so it refuses every case ("the host FPU is keep and could not be
  driven to flush"); the eight cases hold in Linux flush Tier C. The keep set is held near
  half the budget so that an equal flush set fits when the replay gains an x86 lever.
- **Context L2 leg** `parity/meep_gpu/meep_identity_context_l2.py` and **table builder**
  `parity/meep_gpu/build_meep_identity_table.py` (new). The curl-mutation Tier A test now
  picks the smallest probe case: the dispersive family row is a 0-D cell, whose curls never
  reach the regrouped arm. `docs/development/testing.md` describes Tier A's cases, Tier B
  and Table 2.

### The shared mutation defaults plant the pre-alignment groupings

- **The probe's shared default** (`probe_fused_kernel_bit_identity.SOURCE_MUTATIONS`)
  is spelled on the real-PML pair's text. Its ten curl and constitutive entries still
  spelled the 0.9.2 text and no gate's table borrowed them; they take the root pair's
  spellings. The two grouping defects find MEEP's grouping and plant the one the kernels
  carried until 1.0 (the pairwise curl `(a-b)+(c-d)`, the two left-to-right
  constitutive accumulations), and stay must-catch (KERNELS.md 7.1). The hand track's
  leg accounts in `run_pml_gate_mutations.py` say so.
- **`ROOT_PAIR_NEEDLES` and `root_pair_own` are removed.** The hand track's table and
  the ten CUDA gates that took the root pair's spellings through them (BFAST,
  conductive, cylindrical constitutive and real, folded constitutive and curl, no-PML
  curl, nonlinear, nonlinear curl null, special kz) borrow the same needles from the
  shared default. Every mutated kernel text and site count in `gate_needle_anchors.json`
  is unchanged; only the borrowed flags move.
- **`no_pml_ref.reference_plain_step` defaults to MEEP's grouping** (`kernel_order`).
  The pre-alignment tree is renamed `pairwise_order`, the probe's name for it, so no
  grouping name means one tree in one module and another in the other; `array_order`
  is refused there as an unknown grouping.
- **Host tests** pin the direction of both grouping defects, that no table keeps its own
  copy of a shared spelling, and the no-PML reference's default.

### The Triton H-to-D welds carry their bases' forms

The ten H-to-D welds of the Triton table (none dispatched; `INSTALLABLE` stays False)
are re-cut onto the forms their base steps took: `fused_hd_pair`,
`folded_fused_hd_pair`, `beta_real_fused_hd_pair`, `conductive_bfast_fused_hd_pair`,
`complex_fused_hd_pair`, `folded_complex_fused_hd_pair`, `beta_complex_fused_hd_pair`,
`nonlinear_fused_hd_pair`, `cylindrical_real_fused_hd_pair` and
`cylindrical_fused_hd_pair`.

- **The constitutive half.** `fused_hd_pair._h_cell` and
  `complex_fused_hd_pair._h_cell_complex` lift the root constitutive to a cell: the
  record's dsigw lines read at the cell handed in (so a recomputed neighbour takes its
  own chunk's form), `forms.constitutive_form`, and MEEP's start of H through a launch
  scalar `start_h` the plans set from `Fields.h_separated`. The complex one forms every
  product part by part.
- **The curl half** is each base curl's current text with the magnetic reads
  redirected: MEEP's left-to-right stencil with partner presence, the chunk-keyed
  recurrence, the beta and BFAST passes after the forms, the Bloch connection on plane 0
  of the unshifted partners.
- **The scratch history** is written at every cell (B where the component's own layer
  accumulates, the pre-launch word elsewhere), because the rotation publishes the twin.
- **The cylindrical welds** step `update_H` into scratch and then run the certified
  `CylindricalCurlPlan` / `CylindricalComplexCurlPlan` (the column-serial radial scan and
  the curl) over the scratch H, built once over each allocation the rotation alternates
  between: three launches against the singles' three. The complex one imports
  `_h_cell_complex`; its two `tl.math.fma` sites and its scaled divide are gone.
- **Gates.** Every weld gate's root-form mutations are inverted to the previous package
  grouping (KERNELS.md 7.1) and carry the 7.2 battery (whole-grid recurrence, coefficient
  predicate, membership a row off, lattices and flags swapped, auxiliary stores on plain
  cells, the scratch history stored on the layer only, the start of H dropped, partner
  presence ignored), with the unconditional auxiliary loads declared nulls; each gate
  scores MEEP's start of H on a first-update fixture. The cylindrical gate kit drops the
  increment-stage and scan-order legs with the stage they measured.

### The Metal no-PML conductive curl and its D->E pair step MEEP's stencil

Precert regression R6: `metal_kernels/no_pml_conductive.py` and
`metal_kernels/no_pml_conductive_fused_electric_pair.py` still summed the curl's two
differences pairwise, `dtdx * ((g1s - g1) + (g2 - g2s))`, after entry 09 moved the array
path to MEEP's left-to-right `dtdx * (((g1s - g1) + g2) - g2s)` and its one-partner
forms, and no entry re-derived them (entry 35 did the plain twins). On a lossy run with
no absorber the Metal conductive single, which is wired, diverged from the array path. On
series 01..93 `gate_metal_whole_step` stopped at `nopml_conductive` after 6 s (step 1:
Bx 769, By 753, Bz 774, Dx 1,353, Dy 1,430, Dz 1,387 words), and its every-case sweep read
58 / 60 (the two no-PML conductive cases); the precert dry run read 327, 286 and 49 words
on three rows of `gate_metal_no_pml_conductive`. The stop was this divergence, not the
gate's run time: released, the whole gate takes 40 s and the sweep 16 s.

- Both modules take the plain curl's statements and bounds block
  (`no_pml_curl.CURL_STATEMENTS`, `bounds_block`) and bind the chunk record last
  (`RECORD_BUFFER` 20 on the single, 21 on the pair, which now binds 22 of the 31-binding
  ceiling); every builder takes `chunk_bounds` with no default and refuses a missing or
  malformed record by name; the composer's arms hand theirs over. The conductive tails
  are unchanged.
- `parity/meep_gpu/gate_metal_no_pml_conductive.py`: plans bound to the engine's one-rank
  record; a 2-D product row (the one-partner forms); the mutation
  `curl_grouping_flattened`, which is no longer a defect (a flattened sum parses as MEEP's
  grouping), is replaced by `pairwise_grouping` (the previous form, must diverge) and
  `z_partner_forced_present` on the 2-D row.
- `parity/meep_gpu/gate_metal_no_pml_fused_electric_pairs.py`: the conductive family is
  planned on its run's one-rank record, gains two 2-D cases and the plain family's
  `curl_previous_grouping` and partner-presence mutations, and its deposit null control
  is built on a scaled source: an integrated source deposits nothing at the curl, so the
  control built integrated read 0 words with the pair on MEEP's stencil (measured) and
  could not see the bracket.
- Measured on the Apple GPU (flush): `gate_metal_no_pml_conductive` 4 / 4 product rows at
  0 words, 4 / 4 mutations caught; `gate_metal_no_pml_fused_electric_pairs` released,
  both families; `gate_metal_whole_step` released, and its every-case sweep 60 / 60.

### On a folded grid the absent-component clear covers the planes the D fills write

A dispatched `update_E` steps an absent electric component (one MEEP never allocated) as
it steps every other, and the array path, since it learnt MEEP's unallocated set, steps
none of them. Entry 90's clear set those words back to `+0` only where the component's
inverse permittivity carries the sign bit. On a folded grid there is a second set: the
array path still steps the absent component's D, and `fill_symmetry_bc_D` and
`fill_folded_far_ghosts_D` write its fill planes as `parity * image`, `-0` with an odd
parity, so the stepped product is `-0` there with no conductor anywhere.

- `meep_gpu/fastpath.py`: `_absent_component_clears` adds every plane the D fills write
  for the component (`_absent_fill_planes`, read from the fills' own rule,
  `deposit_repair._fill_image_rules`) to the sign-bit cells, and records the two counts
  (`sign_bit_cells`, `fill_plane_cells`) in the plan record's `absent_components` block.
  An unfolded run still gets the sign-bit cells only, and none on a positive permittivity.
- Measured on the shipped Metal route against the array path (lifted corpus rows, every
  stored word, 4 steps): `finite_grating.py` 725 / 3,949,800 words differing per step
  before (`f_w_Ey`, the far ghost plane), `TestGetPoint.test_get_point` 124 / 92,256
  (62 `f_w_Ex`, 62 `f_w_Ey`); 0 on both after, and 0 on `absorbed_power_density.py`
  and `mie_scattering.py`.
- `meep_gpu/test_dispatch_absent_components.py`: a folded case (periodic Y mirror, no
  conductor, real and complex storage), 0 words with the clear and `-0` words only on
  the fill planes with it withheld; `parity/meep_gpu/mutate_from_meep.py`: the mutation
  `absent_clear_skips_the_fill_planes`, caught by it. `docs/design/floating-point.md`
  and `Fields.set_absent_components` state the second set.

### The Metal folded H->D gate drives its compositions as the shipped route does

`parity/meep_gpu/gate_metal_folded_fused_hd_pair.py` read 4 of its 53 driven lift rows
as diverging in references 3 and 4 and, on two rows, in the subject. Two causes, both the
harness's, both measured: its own shim never applied the absent-component clear the
shipped dispatch applies after `update_E`, and its composer call omitted the run's chunk
division, which `FdtdDriver.step` hands `plan_fast_path`.

- Its `Shim` now applies `fastpath`'s own clear (`_absent_component_clears`, built once
  per shim where a plan serves `update_E`, applied after the `update_E` consult answers
  True), and `_composed` hands `plan_step` the driver's `chunk_layout`.
- The four rows, every arrangement against the array path (4 steps; 3 on
  `mie_scattering.py`): `finite_grating.py` 725 words at step 1 -> 0,
  `TestGetPoint.test_get_point` 128 -> 0, `absorbed_power_density.py` 5 at step 2 -> 0,
  `mie_scattering.py` 828 at step 1 -> 0. Without the chunk division and with the clear,
  `TestGetPoint` read 4 words from step 1 and `finite_grating` 6 from step 2.

### `gate_metal_folded_beta`'s refusal floor follows the deletion of its probe row

Entry 89 deleted the expansion-probe clause and, with it, the refusal row
`the_complex_arm_without_a_measured_expansion_artifact`, and left the leg's census floor
at 14, so the gate refused itself ("13 (floor 14)"). The floor is 13, one per row the leg
drives, and the stale comment that named the probe row is corrected.

### The Metal driver-route gate judges its unfused leg, and its census no longer repairs the fused leg

`parity/meep_gpu/gate_dispatch_metal_route.py` passed the rows on which the held-residency
defect in the three complex cylindrical plans (the entry below) was live. Three repairs in
the gate and one row family added; no package code changes.

- **The unfused leg is judged.** It dispatches the certified singles, and its comparison
  with the array path was recorded and never read. A divergence on it -- any checkpoint,
  the final state, or the observables -- now fails the row as `UNFUSED-DIVERGENCE` on every
  branch, and the second consult site requires both kernel legs. Audit of the 2026-09-25,
  2026-10-02 and 2026-10-05 campaigns: 6 of 342 compared rows had an unfused divergence,
  every one `cylindrical_m1` or `cylindrical_m0_complex` on the probe leg and scored
  PASS-FUSED; the unfused leg of every other row matched, so no other recorded verdict
  moves.
- **The seam census no longer refreshes the fused leg.** It read the fused driver's state
  before every launch through the residency read barrier, which copies each device-owned
  volume to the host -- the very stale copies the defective plans read. Measured on series
  entries 01..86 without the plan fix, `cylindrical_m1`, released probes, 24 steps: with the
  census the fused leg read 0 differing words while the unfused leg differed from step 2;
  with it removed the fused leg differed too (22,684 of 358,400 words at step 24, as the
  unfused leg). The census now walks without the barrier (`host_writes.raw`), reads values
  with `host_writes.gather`, which moves no ownership, and runs only on the fused leg's own
  launches.
- **Engine-smoke rows.** `engine_cylindrical_complex_m1` and `_m2` drive Dcyl complex runs
  (m = 1 with PML on r and z, m = 2 with PML on r) through the engine's default route, held
  and shipped, with no gate instrument on the legs, comparing the whole state with the array
  leg at steps 1, 4, 16 and 64 and requiring each leg's record to show both cylindrical
  complex fused pairs dispatched on every step under the residency mode it asked for.
- **The armed null places its window** (the fused route gate's rule): where a withheld
  component accumulates on an own-axis layer and twelve steps leave the legs identical, a
  reach leg on the array path finds the first step at which a second application changes a
  word and both legs run twelve steps past it; `NO-OWN-AXIS-LAYER` is NOT-APPLICABLE by name
  under the same guard as the inactive layer; a reach leg that never fires is
  `ACCUMULATION-NOT-REACHED`, a failure.
- **Measured on an Apple M1 Max**, the four released expansion-probe artifacts exported.
  On series entries 01..86 without the plan fix every Dcyl complex row fails:
  `cylindrical_m1` and `cylindrical_m0_complex` read DIVERGENCE (fused and unfused legs
  from step 2; 184,186 and 92,606 of 358,400 words at step 1600), and the engine rows read
  DIVERGENCE on their held legs (m = 1 111,415 of 1,376,256 words, m = 2 118,204 of
  1,146,880, over the four checkpoints, first at step 4; shipped 0). With the census read
  through the barrier again, the two route rows read UNFUSED-DIVERGENCE with the fused leg
  at 0. With the plan fix the probe leg releases: 9 of 9 rows (the four Dcyl route rows,
  both engine rows, `pml_2d`, `pml_1d`, `bloch_2d`) and 57 of 57 controls pass; the armed
  null places its window at step 26 on the four Dcyl rows, 13 on `pml_2d` and 25 on
  `bloch_2d`, and rules `pml_1d` NO-OWN-AXIS-LAYER.
- `parity/meep_gpu/test_metal_route_unfused_leg.py` (41 host-only cases) pins each repair;
  each of 8 single-repair mutations of the gate fails at least one of them.

### Three gate harnesses read by meaning: a structural null, the phase floor, and two fleet timeouts

The pre-certification dry run on series 01..79 left three gate-harness reds whose kernels
were bit-identical to the array path in every case: a must-catch mutation that cannot
differ on one fixture, a phase floor graded by the k-point's label, and two Triton gates
whose 1.0 mutation sweeps outgrew the fleet driver's timeouts. No kernel changes.

- **`gate_cuda_dispersive_offdiag_polarization_pair.py`.** `drive_from_the_stored_E`
  scored PARTIAL: on `one_state_one_component` the weld carries Ez on a 2-D grid, no chunk
  carries a z layer, so `update_E` stores the drive itself and reading the stored Ez back
  is the identity. Such an arm is now a declared structural null (`STRUCTURAL_NULLS`,
  KERNELS.md section 7.1), decided by the array path's region plan
  (`own_layer_absent`), run, and required identical; a declared null that diverges fails
  the scoreboard (`score_source_arms`). The new fixture
  `one_state_one_component_three_d` gives the single-component Ez weld a grid that
  carries its layer, so the mutation is a clean must-catch there. Measured on one RTX
  A6000, both policies: S1 64 / 64, S2 48 / 48, the mutation CAUGHT on 7 / 7 armed arms
  with the declared null confirmed identical; with the declaration removed it scores
  PARTIAL again, and a null planted on every arm scores DECLARED NULL DIVERGED.
- **`gate_cuda_complex.py`.** The phase floor counted every nonzero k-point as phased.
  MEEP's connection rule makes a phase within `CONNECT_TOLERANCE` of +1 a copy, and the
  Brillouin-edge k-point on the 8-cell axis wraps to exactly that, so 96 of 736
  phase-labelled curl cases failed a floor the kernel satisfies. `phase_floor` grades by
  the connection the array path applies, read without the kernel module
  (`array_path_connection`): every curl case encodes each axis as the array path connects
  it, every case the array path connects rotates a plane, every phase-labelled k-point
  connects a plane on some shape, and the Brillouin edge negates on some shape. Measured,
  both policies: single launch 1008 / 1008, multi-step 36 / 36, mutations 26 / 26, floor
  640 / 640 phased cases live, 1152 / 1152 curl cases connecting as the array path.
- **`drive_triton_weld_gates.py`.** `conductivity` 3,600 s -> 10,800 s and `symmetry`
  5,400 s -> 10,800 s, the measured runtime in each entry's note: 3,798 s and 3,706 s
  released on one card shared with each other and two CUDA gates; symmetry took 5,066 s on
  the dry run's card shared by the fleet.
- Host tests `parity/meep_gpu/test_gate_cuda_dispersive_offdiag_structural_null.py` and
  `parity/meep_gpu/test_gate_cuda_complex_phase_floor.py` pin which arms are declared,
  the scoring, the full product's connections (only 8x8x8 at the edge copies) and the
  floor's failures on planted records.

### The CUDA and Triton H->D gates measure the seam the driver runs since entry 25

Entry 25 moved an integrated source's dipole to the constitutive update (`f_minus_p`), so no
engine source stands a withdraw between the `update_H` and `step_D` consults any more; the
one pass there is the magnetic `f_minus_p` restore. The CUDA and Triton H->D gates still
measured the retired withdraw in four legs and failed them. One shared helper,
`parity/meep_gpu/hd_seam_restore.py`, is what those legs read now (the Metal gates took the
same design earlier):

- **driver_order** reads the restore off `driver.py` (`h_to_d_seam.driver_seam_fact`) in place
  of requiring the withdraw loop.
- **refusal** requires an engine integrated electric source to be admitted and a source class
  that brings its own standing withdraw to be refused by name.
- **withdraw** is the seam leg: integrated electric and magnetic sources under an active
  absorber, the weld force-installed and stepped in lockstep with its own array-path
  reference, required identical; the weld with the restore moved before the launch and with
  no magnetic `f_minus_p` must both diverge.
- **lift** derives the refusals a row owes from its own recorded sources (an electric source
  whose withdraw does work), so the seam record's `withdraw_in_seam` rows are admitted and
  driven. A row the engine's lifter refuses by name (`MeepSimulationNotLiftable`) is counted
  out by name instead of scored as unmeasured.
- **`gate_cuda_fused_hd_pair` lift, the certified singles' chunk record.** The singles the weld
  is compared against bound the one-rank record (entry 27) where the weld binds the driver's.
  On three mirror-folded corpus rows (`TestGetPoint.test_get_point`,
  `absorbed_power_density.py`, `finite_grating.py`) the singles left the array path while the
  weld stayed on it for 60 of 60 steps, and the lift scored that as the weld's divergence.
  The lift (and `gate_cuda_conductive_bfast_fused_hd_pair`'s, and
  `probe_cuda_hd_lift_autopsy.py`) now give the singles the driver's record.
- **The Triton H->D lifts, the same defect in every arrangement.** The Triton plans a gate
  builds on a lifted row bound the one-rank record (the weld builders take no division at
  all). On the folded rows above every arrangement left the array path together; with
  the driver's division (`hd_seam_restore.plans_bind_the_driver_chunk_layout`, held
  around each lift child's drive) the certified singles and the seam alone are identical
  to the array path again.
- Touched: `gate_cuda_fused_hd_pair`, `gate_cuda_complex_fused_hd_pair`,
  `cylindrical_hd_gate_common` (both cylindrical gates), `gate_cuda_conductive_bfast_fused_hd_pair`,
  `beta_hd_gate_common` (special_kz, complex_beta), `gate_triton_fused_hd_pair`,
  `gate_triton_folded_fused_hd_pair`, `gate_triton_complex_fused_hd_pair`,
  `triton_cylindrical_hd_gate_kit`, `triton_hd_tail_gate_kit`. New host tests:
  `parity/meep_gpu/test_hd_seam_restore.py`.

### The route gate arms its withheld-consult control where the mechanism exists, counts the Dcyl pairs' scan, and two fleet readers read the gates' own outcomes

`parity/meep_gpu/gate_dispatch_fused_route.py`, `gate_dispatch_end_to_end.py` and
`drive_triton_weld_gates.py`; gate and harness changes only, no kernel or array-path byte
moves.

- **The absorbed-consult-withheld control is placed where it can diverge.** Since the
  kernels' constitutive halves became bit-identical to the array path's (22-KT1 on the
  Triton table), re-running the array `update_H` / `update_E` on top of a fused launch
  parts the two legs only through the dsigw accumulation, which lives on a component's
  OWN-axis layer and needs the field to have reached a graded cell. Measured on the NumPy
  array path of base + 01..86 (a second application of the array arm in its own slot,
  every step): the first changed word is at step 2 (pml_3d, folded_dispersive_2d) to 48
  (folded_complex_2d) on every route case with an own-axis cell on a withheld component
  (pml_2d, offdiag_2d, folded_2d: 13), and never in 400 steps on pml_1d, nonlinear_1d and
  complex_1d, whose Ex and Hy have no own-axis layer. The twelve-step window from the start
  missed the 2-D, cylindrical and complex cases; the precert pass read
  NULL-DID-NOT-DIVERGE there on every route leg of both NVIDIA tables.
  - `withheld_verdict` (pure) decides the row. New: NOT-APPLICABLE `NO-OWN-AXIS-LAYER`
    when the plan the driver installed gives no withheld component an own-axis
    accumulation cell (the per-component form of the inactive-layer ruling), and
    NOT-ARMED `ACCUMULATION-NOT-REACHED` (a failure) when one exists but a second
    application never changes a word within `ARMED_NULL_REACH_CAP` (200) steps.
  - `reach_leg`: where twelve steps leave the legs identical and an own-axis cell exists,
    a fresh array-path lift under `_reapplication_witness` finds the first step at which a
    second application changes a word, and the two compared legs are stepped to
    `CONTROL_STEPS` past it with the consult still withheld. The row records `window`,
    `reach`, `own_axis_structure` and `accumulating_components`.
  - Must-catch kept: a planted defect (every consult inside the control answering False,
    so the control's fused leg is the array path) reads NULL-DID-NOT-DIVERGE on pml_2d
    (window 24, first armed step 13) and pml_3d (window 13), and the leg refuses; the same
    cases unplanted read DIVERGED-AS-REQUIRED (8 and 16 arrays differing) and pml_1d
    NOT-APPLICABLE `NO-OWN-AXIS-LAYER`.
- **The Dcyl pairs' companion scan is counted.** Since 76-KC8 each hand-CUDA cylindrical
  pair launches the column-serial radial scan and then its fused kernel. The CUDA
  cylindrical drive rows declare `companion_kernels` (`cyl_radial_difference` on
  `step_B` and `step_D`, the spelling `gate_cuda_fused_complex_pairs.py` uses), and
  `baseline_launches_per_step` adds a slot's companions: they cancel against a kernel
  baseline (cuda_shipped, Triton singles 6.0 a step against the fused 4.0, drop 2) and are
  spent only where the Triton-less veto leg left the slot on the array path (cuda_alone,
  0.0 against 4.0, drop -4, counted -4 where one launch per slot counted -2).
- **Fleet readers.** `drive_triton_weld_gates.row_verdict` reads
  `gate_triton_fused_offdiag_electric.py`'s own `verdict["REFUSAL ESTABLISHED"]` where
  the canonical reading is UNREADABLE (the widening is in the row reader because
  `gate_provenance.py` is weld-pinned at its live digest). `gate_dispatch_end_to_end.py`
  writes its outcome into `summary.json` as a `release` block (`run_release`, the decision
  its exit code is read off) beside a `canonical_verdict`, so a `--plan-only` run reads
  released with outcome PLAN-ONLY; `recut_driver_dispatch_record.LICENCE_VERDICTS` still
  refuses PLAN-ONLY for the licence.
- **Tests.** New `parity/meep_gpu/test_route_gate_armed_null.py` (the verdict table, the
  structure reader's fail-closed rules, and on NumPy lifts: the witness fires at 13 on
  pml_2d and 2 on pml_3d, never on pml_1d, and leaves the leg it watches bit-identical to an
  unwatched one) and `test_fleet_verdict_readers.py`; five companion cases in
  `test_substitution_expectation.py`.

### The Triton composition gates install and stamp the campaign's subnormal policy

The five composition gates that 11 Triton arms cite (cylindrical, symmetry,
conductivity, the dispersive engine route and no-PML) installed no subnormal policy, so
their CuPy array path flushed while Triton kept, and a flush campaign of them released
with no policy stated: the re-cut could never record `flush` for those arms, and the
flush licence refused them on every flush leg.

- Each gate now installs the policy `MEEP_GPU_SUBNORMAL_POLICY` names (`keep` when it
  is unset) on the host, CuPy and Triton before it builds a driver, refuses by name
  when any executor does not attain it or when it no longer holds at the end of the
  run, and stamps it as the artifact's `subnormal_policy`
  (`parity/meep_gpu/composition_policy.py`). The gates that lift MEEP simulations
  import MEEP before the install, because importing MEEP flushes the host.
- `recut_composition_records.py --flush-run` reads that stamp, so a released flush
  composition run gives the 11 arms flush evidence.

### The NVIDIA tables can dispatch under a caller-installed flush policy, arm by arm

Until now every NVIDIA family's dispatch licence was `keep` only: a process whose
caller had installed `flush` (MEEP's x86 default) was refused at the policy rung, so the
certification round's flush route legs measured the array path against itself.

- `fastpath.TABLE_DISPATCH_POLICIES` names the policies each table may dispatch under:
  `keep` and `flush` on Triton and CUDA, `flush` on Metal. Dispatch still installs the
  table's own policy when nothing is installed (`TABLE_SUBNORMAL_POLICY`, unchanged).
- A caller-installed policy other than that default is kept, and licenses the plan only
  when every arm the plan dispatches has a run on this compute capability whose
  `certified_policies` field names it (`fastpath.policy_evidence_reasons`). One arm
  without that evidence refuses the whole plan by name.
- The field is written only by the rebind tools, from released legs:
  `rebind_cuda_welds.py` from each keep and flush leg's install stamp,
  `rebind_triton_welds.py --flush-campaign` and `--bit-identity-flush` from a flush
  campaign of the same gates on the same bytes and capability, and
  `recut_composition_records.py --flush-run` / `--flush-family-recert` for the
  composition and family records 33 Triton arms cite (the family record only when all
  nine families released under flush).
- The route gate holds a leg that installed an admitted policy to every dispatch
  requirement (`policy_leg_role` "admitted", `admitted_policy_installed` in the
  artifact), and `recut_driver_dispatch_record.py` reads such a leg as a dispatch
  witness. Until the certification round binds flush evidence no arm carries it, so a
  flush leg refuses by name as before.
- The 1.0 certification round's drivers and runbook are in
  `parity/meep_gpu/rounds/1.0/` (`RUNBOOK.md`).

### The bit-parity coverage board is re-cut on series entries 01 to 92

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..92; the previous cut was on 01..85. Its four generated files
change, and one emit recipe follows a signature entry 89 changed (below). Its test
outcomes are the CUDA and Triton test files' cases of the package suite (CPU-only shim,
every marker, dispatch off) and the Metal test files' cases of the device run (Apple GPU),
both run on this tree's outcome subject: 213 test files, 0 of the rows' test files not
run. `dev_evidence.json` holds only the device legs re-run at the integration of entries
87 to 92.

- **The kernel tables** (N / D, refused rows in brackets; the cut on 01..85, then this
  one): (a) the chunk-record select, CUDA 55 / 55 (+1) -> 55 / 55 (+1), Triton
  45 / 56 (+2) -> 45 / 56 (+2), Metal 60 / 60 (+4) -> 60 / 60 (+4); (b) MEEP's A form,
  CUDA 57 / 57 (+8) -> 57 / 57 (+8), Triton 52 / 61 (+7) -> 52 / 61 (+7), Metal
  66 / 68 (+9) -> 66 / 68 (+9); (c) host word identity, CUDA 18 / 18 (+47) ->
  18 / 18 (+47), Triton 25 / 38 (+30) -> 25 / 38 (+30), Metal 26 / 30 (+47) ->
  28 / 30 (+47).
- **What moved.** In the Metal table's (c) column, bfast_curl and folded_fused_hd_pair
  move from gap to landed: entry 91 makes the Metal BFAST predicate's clause 11a admit a
  nonzero k on a declared-invariant axis, so test_metal_bfast's two listed ids pass, and
  entry 87 re-writes the folded H->D pair's refusal test, whose listed id leaves the list
  (scripts/board_transitions.py, 2 transitions). No other row of (a), (b) or (c) changes
  state. The hand-written CUDA cylindrical_fused_hd_pair row's emit recipe in
  parity_board_emit.json is re-pointed to the one-argument
  cylindrical_complex_kernels._get_kernel(sub_step) that entry 89 leaves: it still called
  the deleted arm argument, so the row read "emission refused" on (a) and (b) until the
  recipe followed the signature; with it the row stays landed.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, re-run on this tree's module bytes, emitted texts and array path.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates.

### The subnormal-policy guide states what a `keep` install cannot reach

Owner ruling R14 (2026-10-09): a CuPy `RawKernel` the caller compiled before a `keep`
install is recorded and documented, not refused.

- **Docs**: [kernel dispatch](docs/guides/kernel-dispatch.md#subnormal-policy), section
  Subnormal policy, states that a `RawKernel` the caller compiled before a `keep`
  install keeps the policy it was compiled under, that the install records the compiled
  kernels still alive after its drop as
  `executors["cupy"]["kernel_memo"]["live_compiled_kernels_after_drop"]` in the stamp
  it returns, and the remedy: install the policy before the caller's own CuPy work. No
  code changes.
- **Known issue in 0.9.x**: before this release a `keep` install dropped CuPy's
  in-process kernels with `cupy.clear_memo()` alone, so a float32 arithmetic ufunc or
  elementwise kernel compiled before the install kept CuPy's own flushing. Measured on
  the 0.9.2 release bytes: no shipped configuration is affected. Over the nine
  end-to-end cases, an install after the lift (as dispatch does it, with a plain or a
  pre-set keep cache directory) differs from one before it in 0 of 302 arrays and 0 of
  11 flux spectra, because 0.9.2's stepping reuses at most 3 of the 14 to 30 kernels the
  lift compiles and none of them is float32 arithmetic that can meet a subnormal. A
  0.9.x run is exposed only when all of these hold: `keep` is in force; before the
  install, the same process compiled a float32 arithmetic CuPy kernel with exactly the
  specialization a later array-path operation uses, the caller's own CuPy work being the
  realistic route; and the operands reach the subnormal range. On 0.9.x, installing the
  policy before the first CuPy compile avoids it.

### The Metal BFAST predicate admits a nonzero k on a declared-invariant axis

`metal_kernels/bfast_curl.py`'s clause 11a refused a nonzero `bfast_scaled_k` component on
a declared-invariant axis, as written when the array path zeroed only the absent
partner's coefficient and kept the other product. The array path takes MEEP's one-partner
forms since entry 24 (the absent partner is not read, step_db.cpp:62-63 and
step_generic.cpp:342-346), the Metal BFAST pass selects those forms on the chunk record's
presence word since entry 52, and the Triton predicate admits the class since entry 77.
The Metal clause now reads the Triton one's words: it refuses only a k assignment whose
two readers (`bfast_scaled_k`, `is_invariant`) cannot be read. The three BFAST fused
products consult this predicate and admit the class with it. No kernel text changes.

- **Measured** on the Apple GPU (M1 Max), flush policy, against the array path in the
  same process:
  - `gate_metal_bfast` released: 322 cases compared (entry 52's 310 and 12 one-partner
    rows whose k has a z component), 0 disagreeing; leg 5 admits `k = (0.4, 0, 0.3)` on a
    z-invariant grid, its premise moves 0 words of the array path's step, and the
    kernel's words differ in 0 places on both sub-steps; must-catch mutations 28 / 28.
  - The three fused products, each built by its own plan builder and launched once on a
    z-invariant 2-D run (two boundary sets) and a 1-D run with k on both invariant axes:
    0 / 53,280 words differ over 12 rows. The tree before this change refuses the 6
    invariant-axis rows by clause 11a.
  - An `FdtdDriver` run of 64 steps, its whole stored state compared at steps 1, 4, 16 and
    64: `k = (0.31, -0.17, 0.23)` on a z-invariant 2-D run and on a 1-D run, served by the
    BFAST curl and, with the electric pair opted in, by `bfast_fused_electric_pair`:
    0 / 1,583,712 and 0 / 58,560 words differ over 4 legs each. The tree before this change
    takes the array path on the same runs. On the array path the 2-D run stores the
    `k_z = 0` run's words: 0 / 395,928 differ.
  - The BFAST H->D weld gate, `below_the_cut` (beta, bfast) and `residue`
    (`bfast_fused_electric_pair`) as entry 52 ran them: the last two release; the weld
    gate's product differs in 0 of 81,648,000 words with 37 / 37 mutations caught, and its
    refusal and withdraw legs fail as they do before this change (entry 52's open item 1).
- **The pending list** loses the two `test_metal_bfast` ids declared for this change at
  the batch_r10n integration; both pass under the CPU-only shim and on the Apple GPU.
- **Not re-cut here**: the board, whose Metal `bfast_curl` row reads `gap` on those two
  ids, and the ledgers and fingerprints, whose drift on this module is already declared.

### A dispatched E update leaves MEEP's unallocated components at +0

The array path no longer steps the components MEEP never allocated (below), but no kernel
table reads that set: every table's `update_E` forms `chi1inv * u` on all three
components, and on a perfect-conductor cell (`chi1inv = -1e-20`) of an absent component it
wrote `-0` into E, or into `f_w` inside the absorber, where the array path and MEEP hold
`+0` (bit_parity_0_10 lane XAC).

- **The clear.** A plan that dispatches `update_E` builds, at its freeze, one clear per
  absent electric component whose inverse permittivity carries the sign bit somewhere
  (`fastpath.AbsentComponentClear`: those cells' flat indices, and the component's E and
  `f_w` arrays), and writes `+0` there after every `update_E` consult it answers, whatever
  plan the slot holds (a single, a pair's sentinel or deposit repair, a weld's trailing
  half). The write goes through the new `host_writes.zero_cells`, one device launch per
  component under a held Metal residency. No kernel, slot or arm changes. A run with no
  sign bit in an absent component's inverse permittivity (no perfect conductor under it)
  gets no clear and no per-step cost, and a run that derives E stores no word to clear. A
  clear that cannot be written exactly (an inverse permittivity that does not broadcast, a
  target that is not C-contiguous) refuses the plan by name. The record carries an
  `absent_components` block (the set, the cleared cells and arrays per component) and the
  report adds `clears_run`.
- **The held reader door.** `dft._raw_factors` serves an absent derived E component as
  zeros, as `Fields.component_factors` does, instead of the product `D * inv_eps`.
- **Measured** with the new `parity/meep_gpu/gate_dispatch_absent_components.py` (seven
  2-D cases with perfect-conductor cells under the absent polarization, among them
  metal-cavity-ldos.py's cell folded and unfolded, TE, a conductor across the absorber and
  one beside a Lorentzian slab; the dispatched run against the array path, every stored
  word, along a x4 checkpoint ladder): without the clear every case whose `update_E`
  dispatches differs, and only in those words (Triton, Metal and the default NVIDIA
  composition: 26,250 of 8,344,344 compared words summed over the checkpoints; the
  hand-CUDA table 26,250 of 7,806,744, and 21,570 of 5,445,120 with fused arms off), every
  differing word a `-0` in an absent component's E or `f_w` on a conductor cell; with it,
  0 of the same totals on every table. `metal_bloch` (complex storage) dispatches no
  `update_E` on any table, and the hand-CUDA table serves no `update_E` on `metal_no_pml`
  (nor, fused arms off, on `metal_dispersive`); those cases measure nothing.
- **Served slots do not move**: the fused-route gate's 50 route cases on the Triton,
  hand-CUDA and default NVIDIA compositions and the Metal route gate's 52, each lifted
  with dispatch on and stepped once, with and without this change: the same dispatched
  slots, arms and tables on every case (123, 110, 123 and 130 slots), 27 of 50 (Metal 28
  of 52) with an absent set, none carrying a clear.

Tests: `meep_gpu/test_dispatch_absent_components.py` (new; a plan whose `update_E` steps
every component, at the driver's seam, against the array path, with a control that
withholds the clear and must show the `-0` words). Mutations
(`parity/meep_gpu/mutate_from_meep.py`): `absent_clear_dropped`,
`absent_clear_before_the_launch`, `absent_clear_skips_f_w`,
`absent_clear_takes_the_positive_cells`, `absent_clear_refusal_dropped`,
`raw_factors_absent_read_as_the_product`. `docs/design/floating-point.md`'s "A field
component MEEP never allocated is not stepped" now says how the dispatched path keeps it.

### The complex-multiply expansion licence is deleted, and the complex families dispatch by default

Every complex family forms each product part by part (`part_products`), as MEEP and the
array path do, so the two arms the licence chose between (`FMA_V1` and `NAIVE`) emitted
one body on all three tables and the probe record that licensed one of them decided
nothing about the bytes. It still decided whether the families ran: without a record named
by `MEEP_GPU_COMPLEX_EXPANSION_PROBE` (Triton and CUDA) or the four
`MEEP_GPU_METAL_*_EXPANSION_PROBE` variables, every complex predicate refused and a default
complex-field run took the array path. The chain is deleted outright, with no alias.

- **Package.** `meep_gpu/expansion_refusal.py`, every family's probe loader, licence and
  arm argument, the probe slots of the step context, the complex predicates' policy
  clause, and `complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY` (the Triton table's
  policy is `fastpath.TABLE_SUBNORMAL_POLICY["triton"]`; Metal declares its dispatch
  policy through `metal_kernels.subnormal.declaring_dispatch_policy`).
- **Harness.** The expansion legs of the gates and probes, the expansion-probe legs of
  the route campaigns and of the record recut, the weld driver's probe plumbing and its
  `unified_expansion` gate, the census's `--probe-artifact` and `--run-policy` (the
  battery protocol is `evaluate(driver)`), the Metal corpus re-cut's four `--probe`
  inputs, and `benchmark_gpu.py --expansion-probe`. Deleted files:
  `cut_expansion_probe.py`, `gate_triton_unified_expansion.py`,
  `probe_metal_beta_expansion.py`, `probe_metal_folded_complex_expansion.py`, the two
  fixture records under `fixtures/expansion_records/`, and three test files of the chain.
- **Measured on the route gates, default legs, no record exported** (bit_parity 0.10 X1,
  base + series 01..79 + this change, RTX A6000 and M1 Max): Triton `shipped` 18 / 18
  complex cases PASS-FUSED, CUDA `cuda_shipped` 19 / 19 PASS-FUSED and `cuda_alone`
  19 / 19 at 0 differing arrays (13 fused, the 6 folded cases refused at rung 6b as
  that table alone must), Metal `shipped` 18 / 18 PASS-FUSED; 0 differing words of
  12,383,249, 12,210,609, 12,210,609 and 11,632,749 compared. None of the legs releases:
  the `absorbed-consult-withheld` control fails on every dispatching case, and fails the
  same way on base + 01..79 (`pml_2d` on both default legs; `bloch_2d`, `complex_3d`,
  `cylindrical_m1` on the record-exporting legs), so it is not this change's.
- **Ledgers.** 145 ledger entries pin a file this change edits (144 of them by code
  digest); 37 pins that held on 01..79 move, among them the Triton
  `triton_unified_expansion_device_gate` weld, whose gate file is deleted and which the
  certification round retires rather than re-gates. Every contract test that walks these
  pins already fails on base + 01..79 under an id declared on
  `tools/ci/pending_certification.txt`, so no id is added; the list's X1 block names the
  moved entries.

### Held Metal residency: the cylindrical complex plans read the radial sum's source through the hold

The three Metal plans that form the cylindrical radial difference on the host -- the
cylindrical complex curl and the cylindrical complex fused magnetic and electric pairs --
pulled the sum's source (`Ey` on B, `Hy` on D) with `Residency.sync_out`. Under a hold that
call copies nothing for a held mirror: it marks the mirror device-owned and seals it. The sum
was then formed from the host words of the source's last host read, and `Bz`/`Dz` went wrong.
The pull is now `metal_kernels.cylindrical_complex.pull_radial_source`: `acquire_read` for a
held source, the unchanged `sync_out` otherwise.

- **Measured on the Metal engine route** (Apple M1 Max, the cylindrical complex arms
  admitted by a released expansion-probe artifact, 64 steps, the whole state of every
  checkpoint against the NumPy array path driven to flush). Held residency, before: m = 1
  111,415 / 1,376,256 words differing, m = 2 118,204 / 1,146,880, m = 0 with complex
  storage 55,774 / 1,376,256, the route gate's monitored m = 1 case at resolution 10
  56,090 / 358,400, each first at step 4. After: 0 on all four, held and shipped. Shipped
  read 0 before and after.
- **The driver-route gate** (`gate_dispatch_metal_route.py`, a diagnostic run without
  controls, 1600 steps, 80 x 80): its unfused rows, which step these plans' single curls
  under the same hold, went from 20 / 32 (m = 1) and 10 / 32 (m = 0, complex storage)
  arrays differing at step 1600 to 0 / 32; its fused rows read 0 / 32 before and after.
- **Cost under a hold**: one whole-volume copy-out per curl sub-step whose source the
  device owns (the m = 0 complex-storage and monitored m = 1 runs: 14 copies out per step,
  against 12 before). No throughput is claimed.
- **Unchanged**: the unheld bracket, every kernel source, and the gates. The four
  cylindrical complex gates, the cylindrical complex probe and the whole-step Dcyl rows
  read leaf for leaf equal on this change and without it.
- `meep_gpu/test_metal_cylindrical_radial_source_pull.py` drives each plan's refresh
  through the held-mode doors on every host, and through a real held `Residency` on an
  Apple GPU; both layers fail without the change (8 of the 13 cases on an Apple GPU, the 4
  held stand-in cases on any host).

### The Metal gates and probes the pre-certification dry run stopped on read the 1.0 source path

The Metal round's dry run on series 01..52 stopped on a probe and on gate legs whose
harness still read the 0.9.2 package: the standalone folded-complex expansion probe, the
H->D gates' withdraw legs and lift accounting, a stale citation, a vacuity floor, a
whole-step fixture and one mutation needle. Every change is in a gate, a probe or a test,
except one citation re-point in `meep_gpu/triton_kernels/no_pml_constitutive.py`; no
kernel and no array-path module changes.

- **The folded-complex expansion probe** (`probe_metal_folded_complex_expansion.py`;
  history: the probe is deleted with the expansion licence, section above) read the
  float32 parity word `folded_complex.mirror_parity_coefficients` returns since
  entry 36; it indexed a word pair and raised, and `recut_metal_gates.sh` stops before any
  gate when a probe fails. Each coefficient then also recorded the array path's own
  spelling, `part_products.scale(word, plane)`: it is the sign flip on every word, and
  parts from the complex-by-int product the verdict reads on the 2 signed-zero words the
  probe's sign-flip floor counts.
- **The H->D gates' seam legs** (`gate_metal_fused_hd_pair`, `_folded_`, `_conductive_`,
  `_bfast_` and `_complex_fused_hd_pair`): `driver_order` reads the magnetic `f_minus_p`
  restore as the one statement between the `update_H` and `step_D` consults
  (`h_to_d_seam.EXPECTED_BETWEEN`); `refusal` requires an engine integrated electric
  source to be admitted and to stand no withdraw, while a foreign source class bringing
  a withdraw of its own is still refused by name; `withdraw` drives the seam's one pass,
  an integrated electric and magnetic source under an active absorber, and requires the
  weld identical to the array path and both orderings the commutation does not license
  (the restore before `update_H`, no magnetic `f_minus_p`) to diverge.
- **The lift legs' refusal accounting**
  (`gate_metal_fused_hd_pair.lift_refusal_accounting`, used by the plain, folded,
  conductive and BFAST H->D gates and by `hd_tail_gate_common` for the two beta and the
  folded-complex ones): a lifted row carrying a magnetic source is refused
  by `launch.h_start_reasons`, derived from the row's own source list; every other row is
  admitted, the seam record's `withdraw_in_seam` rows included; a refusal naming a
  standing withdraw, a reason the accounting does not recognise, or a row with no source
  list fails. The BFAST and tail lift legs now also require at least one driven row,
  as the plain and folded ones already did. The conductive cell holds one row
  (`TestAdjointSolver.test_damping`), refused by name on both halves (its magnetic
  source, and the curl half's per-chunk conductivity); that leg passes on the
  accounting, records `no_row_drove`, and licenses no corpus row. The folded gate counts
  the two rows the engine refuses to lift by name (a BinaryPartition `chunk_layout` with
  mirror symmetry, `from_meep.PARTITION_WITH_MIRRORS_REASON`) as that disposition,
  bounded at a tenth of the cell, rather than as unmeasured.
- **Licences**: the plain, conductive and BFAST H->D gates no longer say their withdraw
  leg licenses the hoist wiring, and the folded one no longer says its 3
  `withdraw_in_seam` rows stay refused; no engine source stands a withdraw in this seam.
- **A directly run gate no longer stacks its census.** A gate run as a script executes
  its module twice (as `__main__`, then the runner's copy); the BFAST and conductive
  censuses subclassed the first copy, and the BFAST one restored the first copy's wrapper
  on exit, one wrapper deeper per use, until the `disarm` leg raised `RecursionError`.
  Both extend `gate_metal_fused_hd_pair.BaseIntermediateCensus`, read the original the
  base saved, and refuse by name to wrap a census wrapper.
- **Citation**: `no_pml_constitutive.NULL_SIDES` cites `stepping.py:1229-1230` and
  `:1274-1275`, the no-PML guard and `return` of `update_H` and `update_E` (stale since
  entry 09), with the module's other live references to the same lines (its docstrings,
  its clause records' and refusal reasons' line numbers) and those of
  `gate_metal_no_pml_constitutive`, `test_metal_no_pml_constitutive` and
  `test_triton_no_pml_constitutive` (whose H-side clause test reads the refusal's
  `stepping.py:1973-1974`); `gate_metal_no_pml_constitutive`'s citation check now also requires an ascending range
  that opens on the guard. Two stale citations in the module are left as they are: the
  active-layer refusal's `stepping.py:946-951, :1014-1018`, which
  `gate_triton_no_pml_constitutive`'s mutation `m1` quotes verbatim, and the dated
  `STORED_E_ARM` record's transcription. The module's bytes are pinned by
  `meep_gpu/metal_kernels/fingerprints.json`, `meep_gpu/triton_kernels/fingerprints.json`
  and `parity/meep_gpu/parity_board/board.json`; those pins are left for the
  certification round to re-cut.
- **Harness calibrations**: `gate_metal_symmetry`'s signed-zero curl floor is 400 words
  (the array path's own `step_B` / `step_D` move 455 / 491 of the 2-D case's 1,152 words
  since entry 13, against 750 / 888 before); `gate_metal_whole_step` installs the region
  plan before the first slot, as the driver does; `gate_metal_tranche7_fused_pairs`'s
  `near_ghost_takes_the_far_parity` needle names `c_mul_coefficient_left`.
- **New host test** `parity/meep_gpu/test_gate_metal_hd_lift_host.py` pins the accounting
  on rows built from the product's own refusal text, and that a twice-loaded gate neither
  stacks nor leaks its census.
- **Declared resources**: `tools/ci/declared_resources.txt` lists the folded H->D test
  file's two re-written refusal tests
  (`test_a_foreign_standing_withdraw_is_refused_by_name_while_the_flag_is_false`,
  `test_an_integrated_electric_source_stands_no_withdraw_in_the_seam`) under `[torch]`, in
  place of the retired `test_a_standing_electric_withdraw_is_refused_by_name_while_the_flag_is_false`,
  so a host without PyTorch skips them on the declared resource instead of failing them as
  unsanctioned skips.

### The bit-parity coverage board is re-cut on series entries 01 to 85

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..85; the previous cut was on 01..78. Only its four generated
files change. Its test outcomes are the CUDA and Triton test files' cases of the package
suite (CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the
device run (Apple GPU), both run on this tree's outcome subject: 212 test files, 0 of the
rows' test files not run. `dev_evidence.json` holds only the device legs re-run at the
integration of entries 80 to 85.

- **The kernel tables** (N / D, refused rows in brackets; the cut on 01..78, then this
  one): (a) the chunk-record select, CUDA 55 / 55 (+1) -> 55 / 55 (+1), Triton
  45 / 56 (+2) -> 45 / 56 (+2), Metal 60 / 60 (+4) -> 60 / 60 (+4); (b) MEEP's A form,
  CUDA 57 / 57 (+8) -> 57 / 57 (+8), Triton 52 / 61 (+7) -> 52 / 61 (+7), Metal
  66 / 68 (+9) -> 66 / 68 (+9); (c) host word identity, CUDA 18 / 18 (+47) ->
  18 / 18 (+47), Triton 25 / 38 (+30) -> 25 / 38 (+30), Metal 26 / 30 (+47) ->
  26 / 30 (+47).
- **What moved.** Nothing: no row of (a), (b) or (c) changes state in any table. The
  batch changes the array path's source placement at periodic seams, its absent 2-D
  components and sigma lookup, the Tier C harness and the Tier A fixtures; no row's
  predicate or test file measures any of these differently.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, re-run on this tree's module bytes, emitted texts and array path.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates.

### The MEEP-identity Tier A set is re-minted on this batch's array path; the array-path oracle does not move

The source placement at periodic seams, the absent 2-D components and the two Tier C reader
rules below change modules the Tier A fixtures record (`sources.py`, `driver.py`,
`fields.py`, `from_meep.py`, `stepping.py`, `capsule.py` and the Tier C runner), so the
Tier A keep set is minted again on this tree by `parity/meep_gpu/record_meep_identity.py`.
The array-path oracle's fixture set stays `array_path_oracle/real-v6`.

- **Measured** against MEEP 1.33.0 single precision built without contraction, keeping
  subnormals, on Apple silicon: 4 of 4 cases, 38,040,036 compared stored words and
  61,772,940 reader words equal to MEEP's, every digest reproduced by the replay with MEEP
  never imported; 1,360,006 of the 5,000,000 budget bytes. The four capsules (their
  provenance records the changed modules), the four digests files (only their `words_of`
  line, which now names the exact stored-word reader; every digest is unchanged) and
  `meep_identity.json` (the inputs it records, and the reader's new counters) change.
- **No fixture-set bump.** `record_array_path_oracle.py` in report mode reads fingerprint
  `1ab51627` and the same 20 case digests on this tree as on the series before these
  changes, so nothing the oracle measures moved; the ledger still records `real-v6`. The
  kernel ledgers are not written (a certification event), as before.

### Tier C reads the held B word MEEP's first step leaves unconnected from its image

`parity/meep_gpu/meep_identity_tier_c.py compare` read, at step 1 of a PML run on a Bloch
axis with a zero wavevector, a held B word MEEP never wrote. MEEP builds a step's chunk
connections before `update_eh` gives H an array of its own: `connect_the_chunks` skips B
between two chunks that both keep H in B's array (`B_redundant`, boundaries.cpp:359-365,
402-403, 491-492), and at the first step every chunk still does; `update_eh` allocates H
where a PML conductivity along the component, mu or a polarization needs it
(update_eh.cpp:165-170) and only then marks the connections stale (update_eh.cpp:53-56), so
they are rebuilt at the next `step_boundaries` (step.cpp:81, 226), after that step's B
boundary (step.cpp:72). The not-owned low plane of such a chunk therefore keeps, for one
step, the B word it held before (`step_db` writes owned points only, step_db.cpp:124 and
meep/vec.hpp:1102-1104), where from the second step on B's connection writes the image's
B. The package's stored window takes that plane as the image's word on an axis with no
phase (`Grid.stored_window_low`), and its own readback already takes the zero there after
the first step (`readback.HAliasRule`, `first_step`).

- **The reader rule**: at a held B point of a chunk that kept H in B's array at the
  previous read and has an array of its own now, a word that is not the image's must be the
  word read one step earlier, with the image's owner keeping H in B's array then too, and
  the word read is the image's exact stored word: the point one lattice vector up on each
  axis where it sits on the low plane (`locate_point_in_user_volume`,
  boundaries.cpp:207-230), on a Cartesian grid with no symmetry and a zero Bloch wavevector
  along that axis, where `eikna` is exactly 1 and the connection a copy
  (boundaries.cpp:93-98, 404-405). Anything else counts in `held_unconnected_unexplained`,
  which errors the case and refuses a mint (`record_meep_identity.py`).
  `held_unconnected` and `held_unconnected_substituted` are reported.
- **Row**: `ex.differential_cross_section`, whose last 240 stored words (80 each of Bx, By
  and Bz, all at step 1, all signs of zero) were these words.
- **Tests**: on a 2-D Bloch grid with k = 0, a PML at both ends in y and a field present
  before the first step, the bottom chunk's held By row keeps its `+0` while its image
  carries a nonzero By; the reader reads the image's exact word on real and complex storage,
  refuses a planted word that is neither, and substitutes nothing from step 2 on
  (`test_a_held_b_word_of_a_chunk_that_took_its_own_h_array_this_step_reads_its_image`,
  resource `meep_identity_reference`).

### The array path steps MEEP's unallocated components as MEEP does, and the sigma lookup follows libctl's search

Three places where the array path or the lift did not do what MEEP does, found behind the
residual rows of the corpus Tier C dry run (bit_parity_0_10 lane REF):

- **A component MEEP never allocated is not stepped.** MEEP allocates only the components
  a source requires and those `is_like` them (fields.cpp:471-474, :552-572): a 2-D run with an
  Ez source has no Ex, Ey, Dx, Dy, Hz or Bz, skips them in `update_eh` (update_eh.cpp:150)
  and reads them as 0.0 (monitor.cpp:160-164). The array path stepped them, and on a
  perfect-conductor cell (`chi1inv = -1e-20`) the constitutive product wrote `-0`. The lift
  now reads `fields::have_component` and records the absent ones with the new
  `FdtdDriver.set_absent_components` (a construction call the capsule replays);
  `stepping.update_E` leaves an absent E component untouched, `Fields.get_E` and
  `Fields.component_factors` serve it as zeros, and `add_source` on an absent component
  takes its class back out of the set (MEEP's `require_component`). Refused by name: an
  absent component that carries a source, and a declaration after the first step.
- **The lookup shifts the periodic copies by MEEP's lattice.** libctl replicates each
  object by `geometry_lattice.size`, which MEEP sets to the rounded grid's size
  (`user_volume.n{x,y,z}() / resolution`, meepgeom.cpp:1956-1985), not the declared cell,
  and runs `LOOP_PERIODIC`'s three-dimensional case in every MEEP dimension:
  `make_geom_epsilon` calls `geom_initialize` (geom.c:228-238, `dimensions = 3`), and MEEP's
  `set_dimensions` sets MEEP's own `dim`, not libctl's. So a 1-D cell's copies lie along z
  and a cylindrical cell's over r and z. `from_meep._lattice_shifts` takes
  `_geometry_lattice_size`; it replaces `_lattice_wrap_vectors`.
- **The lookup searches in libctl's order with libctl's bounding-box gate.** Objects in
  reverse declaration order, each as its copies over `LOOP_PERIODIC`'s shifts in that
  loop's order (geom.c:478-519, :1579-1600), each copy gated on its bounding box before
  the object test (`tree_search`, geom.c:1651-1667; `_libctl_bounding_box` transcribes
  `geom_get_bounding_box` for a block, an ellipsoid and a sphere; a cylinder, cone or wedge
  takes a looser box that only spares object tests, and no other kind is gated). The old
  search tried every unshifted object first and took no box gate.

The two lookup changes take two structured dispersive rows off the inexact chi1inv inversion
(stochastic_emitter_reciprocity.py, a cell not a whole number of pixels; test_dump_load.py's
3-D cell, a block across a periodic face); their sigma words are now MEEP's.

Tests: `meep_gpu/test_absent_components.py` (new); in `meep_gpu/test_sigma_lookup.py` three
lookup tests, each with a control that must fail without the change (the rounded lattice,
the bounding-box gate, a 1-D cell's copies along z), the shifts pinned in order for a 1-D,
a 2-D and a cylindrical cell, and the looser cylinder box checked against the object test;
`meep_gpu/test_metal_held_fallback.py`'s table of public driver mutators gains
`set_absent_components` (refused by name mid-run, on the held and the reference path alike).
Mutations (`parity/meep_gpu/mutate_from_meep.py`, 192 in all, 0 stale anchors):
`periodic_copies_shifted_by_the_declared_cell`, `periodic_copy_bounding_box_gate_dropped`,
`lattice_shifts_take_meeps_dimension`, `absent_component_stepped`,
`absent_component_read_as_the_product`, `lift_records_no_absent_component`;
`sigma_lookup_object_precedence_reversed` and `sigma_lookup_periodic_object_images_dropped`
re-anchored on the new search.
`docs/design/floating-point.md` gains "A field component MEEP never allocated is not
stepped" and "The lifted susceptibility sigma is MEEP's float32 word".

### Sources are placed as MEEP's `loop_in_chunks` places them

A source's deposit list is now MEEP's `src_vol` list, piece for piece
(`meep_gpu/sources.py`, `_source_geometry` / `_src_vol_pieces` / `bind_src_vols`):

- **One piece per lattice shift and chunk.** A request that reaches a periodic face is
  placed once per lattice shift, and each (shift, chunk) pair is its own entry; two
  entries meet on one cell exactly at the seam of a source spanning a periodic axis, and
  that cell is deposited twice, rounding twice, as MEEP's `step_source` deposits it
  (loop_in_chunks.cpp:390-531, sources.cpp:243-312, fields.cpp:576-585). The package
  summed the two amplitudes into one deposit: a one-ulp seam row, on the D deposit and on
  an integrated source's `f_minus_p` alike, folded or unfolded, with or without a Bloch
  phase.
- **An axis the run's dimension does not carry is one point** of weight 1, and the
  amplitude function is handed 0 there (`vec2py`). The package placed two z rungs at the
  component's own +-dx/2, so an `amp_func` carrying kz was averaged to `cos(pi*kz*dx)`.
- **Every owned point is listed**, its amplitude zero or not (the package dropped
  `|amp| <= 1e-20`).
- **MEEP's arithmetic**: the image location `(0.5*is + i)*inva + shift*(0.5*inva)`, the
  shift phase `pow(eikna, ishift)` in the host C++ runtime's form, real scalings per part
  and the complex products separately rounded (CPython's complex multiply contracts to a
  fused multiply-add on Apple silicon).
- **The eigenmode sample** is MEEP's: the mode re-centred on the source volume
  (`center -= where.center()`, mpb.cpp:910) and sampled at each site's offset.
- `readback.cpow_int` forms `pow(complex<double>, int)` as the host's C++ runtime does:
  libc++'s `exp(n*log(z))` on Darwin, libstdc++'s repeated squaring and libgcc's
  `__divdc3` elsewhere, so the Bloch duplicate plane of `get_array` is MEEP's on Linux too.

Measured, Tier C at OMP 1 (MEEP 1.33.0 single precision without contraction on Apple
silicon, MEEP 1.33.0 single precision on Linux): of the 26 corpus rows whose residual was
the placement, 19 now store 0 differing words in every mode run (keep and flush on both
hosts; 17 of 24 on Linux, where two rows need PyMieScatt); 6 differ only in signed zeros
that MEEP's own point read drops and an exact stored-word reader reads as equal;
`ex.stochastic_emitter` draws its source from a random stream and is not comparable. No row
that held before differs (173 Cartesian rows, keep, on both hosts).

- **Tests**: `meep_gpu/test_source_deposit.py` (a seam cell's two entries and two roundings;
  the axis the dimension does not carry), `meep_gpu/test_deposit_repair.py` (the fused
  routes' repair bracket carries a cell deposited twice, folded and unfolded, real and
  complex, both seams; the mirror-row precondition reads the rows of nonzero amplitude),
  `meep_gpu/test_sources.py`, `meep_gpu/test_from_meep.py`. `mutate_from_meep.py` gains
  `seam_image_summed_into_one_entry` and re-points three needles at the transcription.
- **Docs**: [floating point](docs/design/floating-point.md#sources-meeps-currents-meeps-deposit-list-meeps-f_minus_p).

### A source's lattice images are MEEP's separate deposit entries at a periodic seam

MEEP calls `src_vol_chunkloop` once per (lattice shift, chunk) (loop_in_chunks.cpp:390-531),
and `fields_chunk::add_source` merges a call into an earlier entry only when the index vectors
are identical (fields.cpp:576-585, sources.cpp:179-181). A request spanning a periodic axis
reaches the seam row through the direct pass's edge weight and through the image of its other
end, so MEEP subtracts at that cell twice and rounds twice (step.cpp:295-317). The array path
summed both into one amplitude and deposited once, 1 ulp from MEEP on the seam row.

- **The entries are the placement's of the section above** (`sources._source_geometry`,
  `_src_vol_pieces`, `bind_src_vols`), which replaces this entry's first form of it: each
  (lattice shift, chunk) piece is its own entry -- a mirror-folded periodic axis's image
  passes and an unfolded periodic axis's wrapped rungs alike -- in `loop_in_chunks`'s order,
  and `_deposit_tables` (replacing `_deposit_table`) returns a bound source's entries, which
  every `inject` and `f_minus_p_begin` deposits in order.
- **Measured** against MEEP 1.33.0 single precision (coff build on Apple silicon; Linux
  reference on the GPU host): the 9 corpus rows of the far-ghost family (ex.metasurface_lens,
  ex.binary_grating_phasemap, ex.diffracted_planewave, ex.chirped_pulse,
  test_diffracted_planewave, test_grating_3d, test_binary_grating oblique_0_0_0, oblique_1_10_7
  and special_kz_2_21_2) store 0 differing words, every step 0 to 20 and the checkpoints to N.
  The array-path oracle does not move (0 of 20 cases). The 9 rows hold on the placement of
  the section above too (Mac keep and Linux keep).
- **Tests**: `test_source_deposit.py` pins the two subtractions on an unfolded and on a folded
  periodic axis at both parities and both stencils, from the source's pieces
  (`_src_vol_pieces`) and its bound entries; `mutate_from_meep.py`'s
  `seam_image_summed_into_one_entry` selects it.
- **Docs**: [floating point](docs/design/floating-point.md#sources-meeps-currents-meeps-deposit-list-meeps-f_minus_p).

### The bit-parity coverage board is re-cut on series entries 01 to 78

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..78; the previous cut was on 01..73. Only its four generated
files change. Its test outcomes are the CUDA and Triton test files' cases of the package
suite (CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the
device run (Apple GPU), both run on this tree's outcome subject: 212 test files, 0 of the
rows' test files not run. `dev_evidence.json` holds only the device legs re-run at the
integration of entries 75 to 78.

- **The kernel tables** (N / D, refused rows in brackets; the cut on 01..73, then this
  one): (a) the chunk-record select, CUDA 35 / 41 (+15) -> 55 / 55 (+1), Triton
  32 / 56 (+2) -> 45 / 56 (+2), Metal 60 / 60 (+4) -> 60 / 60 (+4); (b) MEEP's A form,
  CUDA 37 / 43 (+22) -> 57 / 57 (+8), Triton 39 / 61 (+7) -> 52 / 61 (+7), Metal
  66 / 68 (+9) -> 66 / 68 (+9); (c) host word identity, CUDA 12 / 18 (+47) ->
  18 / 18 (+47), Triton 22 / 48 (+20) -> 25 / 38 (+30), Metal 27 / 30 (+47) ->
  26 / 30 (+47).
- **What moved.** CUDA's cylindrical families (entry 76) and its BFAST and beta families
  (entry 78) land in (a) and (b): 20 rows each, 14 of them refused at emission on 01..73;
  (c) gains 6 (the scan, the real and complex Dcyl curls, the BFAST, special-kz and
  complex beta singles) and refuses 3 H->D welds whose test files measure no arithmetic of
  them. Triton's fold modules and the ten fused pairs that carry a pass (entry 77) land 13
  rows in (a) and (b); 3 land in (c) and 10 fused ones are refused there for the same
  reason. Metal's bfast_curl leaves (c): test_metal_bfast's two comparisons of the Metal
  BFAST predicate with the Triton one fail since entry 77's clause 11a, declared on the
  pending list for the Metal folds lane.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against the
  NumPy array path, re-run on this tree's module bytes, emitted texts and array path.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates.

### The hand-written CUDA BFAST and beta families run MEEP's folded passes after the curl

The CUDA kernels that carry `grid.bfast_active` or `grid.beta` -- the singles
`cuda_kernels/bfast_curl.py`, `special_kz_curl.py` and `complex_beta_kernels.py`, the
fused pairs `bfast_fused_{magnetic,electric}_pair`, `special_kz_fused_{magnetic,electric}_pair`
and `complex_beta_fused_{magnetic,electric}_pair`, the H->D welds
`special_kz_fused_hd_pair`, `complex_beta_fused_hd_pair` and
`conductive_bfast_fused_hd_pair` -- now step what the array path steps
(`stepping._bfast_term`, `_special_kz_beta_term`, `_apply_folded_increment`):

- **The term is a pass of its own.** Each kernel is its root kernel's curl and absorber
  with MEEP's `STEP_BFAST` (step_db.cpp:124-142) or special-kz beta pass (:161-175)
  inserted after the component's own `pml_apply`, instead of a term folded into the
  curl before the masks. The pass forms `-raw` with MEEP's unowned cells at `+0` and adds
  it through its own additive ladder: `f -= sinv*inc` outside the dsigu chunks, and
  `fu -= (df = sinv*inc); f -= sinv_u*df` inside them. The shared helpers live in the
  new stdlib-only `cuda_kernels/folded_pass.py`.
- **BFAST's state step is MEEP's.** Both partners: `F = (k1*(g1s + g1) - k2*(g2s + g2)) -
  F_prev`; one partner: `k*(gs + g) - F_prev`, or `k*(gs + g)` in MEEP's plain chunk; the
  increment is `-(F - F_prev)`, a second rounding, and `F` is stored on MEEP's owned
  cells only.
- **Beta is one real word per call site.** The real family adds `-(c * g)`; the complex
  family forms `-quarter_turn(c, g)` part by part from one host-rounded float32
  coefficient per call site (`complex_beta_kernels.beta_curl_coefficients` now returns
  two floats and the kernels bind `float bp, float bm`), so no complex coefficient is
  formed anywhere.
- **The chunk record.** Every kernel binds the run's record (`cb`) as its root does; the
  launchers require `chunk_bounds`, the magnetic pairs and the welds also take the
  layer the run's first `update_H` starts H with, and the fused-pair plans in
  `fused_pairs.py` bind both. The conductive/BFAST H->D weld binds the conductive-chunk
  masks and its `KEEP<n>` triple, so all of its parameters are bound.
- **Gates.** `gate_cuda_bfast.py`, `gate_cuda_special_kz.py` and
  `gate_cuda_complex_beta.py` hand the kernel the record the array path stepped with and
  arm the record mutations of KERNELS.md §7.2; their NumPy transcriptions and source
  mutations follow the pass form. The fused-pair gate's six fold families, the two beta
  H->D gates and the conductive/BFAST H->D gate bind the record; the constitutive
  regroupings install the grouping used until 1.0 and stay must-catch (KERNELS.md §7.1).
  The harnesses follow the array path's start of H (`stepping.separate_h_from_b`): a
  seeded constitutive state is a mid-run one, and a separate route that launches the
  real `update_H` single runs the start as the shipped launcher does. Four needles were
  re-measured against MEEP's forms, where an unowned cell's word survives a complete
  step only in an auxiliary: `beta_on_an_unowned_cell` moves to target 1 (Dy, live on a
  periodic y fold with an x layer), `codes_all_periodic` on the real beta D seam is live
  where x is walled or y is a periodic fold, and the two wall-clear register needles
  are scored on a mid-run state. The beta H->D kit's movement floor exempts the
  auxiliaries the array path never steps.
- **Tests.** The new `test_fold_chunk_forms.py`, `test_fold_fused_chunk_forms.py` and
  `test_fold_complex_chunk_forms.py` run every emitted text through the C interpreters
  against the array path, word for word, with planted `-0` and subnormals, and plant
  each defect the batteries name. `test_complex_folded.py`'s retired transcription of
  the earlier folded tree is removed with its last user.

### The Triton special-kz and BFAST kernels run MEEP's passes after the curl

The fold kernels of the Triton table -- `meep_gpu/triton_kernels/special_kz.py` (the
beta curls, real and complex), `bfast_curl.py` (the BFAST curl), the two beta curls of
`folded_complex.py` (K3a real, K3b complex) and the ten fused pairs that carry a pass
(`beta_fused_{electric,magnetic}_pair.py`, `bfast_fused_{electric,magnetic}_pair.py`,
`complex_beta_fused_{electric,magnetic}_pair.py`,
`folded_beta_fused_{electric,magnetic}_pair.py`, `folded_beta_complex_fused_pair.py`,
`folded_beta_complex_fused_magnetic_pair.py`) -- now compute what the array path
computes since the passes became MEEP's own.

- **The beta pass after every component's curl.** The increment is the host-rounded word
  times the centre partner, negated by `* -1.0`, an exact `+0` on every cell the curl's
  ownership mask drops (both arms under a fold), and the new `forms.fold_ladder`
  subtracts it: `f - (sinv*q)` outside the component's dsigu chunk, `fu - (sinv*q)` and
  `f - (sinv_u*(sinv*q))` inside it. On complex storage the increment is a quarter turn
  of the partner's parts by one real word, as `part_products.quarter_turn` forms it, so
  `special_kz.beta_curl_coefficients` returns two floats (the D side's negated) and the
  plans pass two words. The two `tl.math.fma` sites of
  `special_kz._mul_imag_coefficient_left` are gone; no beta body calls it, and it keeps
  one body for the H->D weld that still lifts it.
- **The BFAST pass per component.** The new `forms.bfast_advance`: the state takes
  MEEP's two roundings (`F = S - F_prev` stored, then `F - F_prev`), an absent partner is
  not read, MEEP's plain branch (a chunk carrying neither the dsig nor the dsigu layer)
  drops `- F`, and the state keeps its word on the cells the curl's mask drops; the
  increment goes through the ladder. Clause 11a now admits a nonzero k on a
  declared-invariant axis (it multiplies a partner MEEP does not read) and refuses only
  an unreadable k assignment.
- **The chunk record.** Every plan of these modules binds `cb`: the engine builders the
  run's division (`chunk_layout`, forwarded by `plan_step`), the harness builders a
  required keyword `chunk_bounds`; the BFAST run's constitutive plan carries MEEP's start
  of H.
- **Gates and probes.** The new `parity/meep_gpu/fold_pass_mutations.py` holds the
  passes' mutations in one spelling for every carrier (the pass dropped, its partners
  swapped, its sign flipped, scaled by dtdx, inside the curl, the increment unmasked,
  the ladder without dsigu or with its axes swapped, the auxiliary not advanced, the
  quarter turn reversed, the top-plane mask dropped; for BFAST the one-rounding state,
  the plain branch keeping `- F`, the absent partner read, the k pair swapped, the state
  stored on unowned cells or not stored), each with the `needs` its scoring case must
  carry; `fold_pass_reference.py` is the passes' reference on either array module, and
  forms a scalar coefficient as a host float32 word (CuPy's `asarray` of a Python `-0.0`
  is `+0.0`, and a BFAST k gated to zero on the D side is that word).
  `gate_triton_special_kz.py`, `gate_triton_bfast.py` and the beta rows of
  `gate_triton_folded_complex.py` are re-transcribed to the passes and score those rows
  with the root forms' stencil inversion; the ten pair probes are re-cut to them, and
  the complex beta pairs' forced unlicensed `EXPANSION` arm is a declared null (one
  per-part body under both names). In the two folded beta magnetic probes the
  ownership-AND drop (`m_ownership_and_dropped`, `m8_ownership_mask_not_ANDed`) is a
  declared null with its premise measured each run: on the 2-D grids beta admits, the
  stray carry lands on a cell its owning composite rewrites later in the same program,
  and outside the component's own layer. The beta-less twins score the race on their 3-D
  rows. The two gates import `gate_triton_symmetry` (whose
  chunk-record reference they reuse) on first use, and it imports CuPy only if present,
  so the gates import and run their NumPy reference legs without CuPy.
- **Not in this change, and red until their lanes:** the H->D welds that lift the beta
  curl bodies (`beta_real_fused_hd_pair`, `beta_complex_fused_hd_pair`) still carry the
  previous text; their transcription tests are on the pending list for the H->D lane.
  The beta licence still requires the fifth probe pattern, which no pass forms any more;
  retiring it is part of removing the licence chain. Ledgers and fingerprints are
  unchanged, pending the certification round; the folded complex beta magnetic pair's
  weld test is listed until its gate is re-run on these bytes.

### The hand-written CUDA cylindrical families step MEEP's per-chunk radial sum on a column-serial scan

The CUDA Dcyl kernels -- the real m = 0 curl pair `cuda_kernels/cylindrical_kernels.py`,
the complex curl pair `cylindrical_complex_kernels.py` (every m class), the real and
complex fused B->H and D->E pairs and the two H->D welds -- now compute what the array
path computes (`stepping.cylindrical_rderiv_difference`, `_cylindrical_imr_increment`,
`_cylindrical_axis_zero_B` / `_D`, `_apply_curl_regions`, `_apply_constitutive_regions`):

- **The radial sum is MEEP's, per chunk, on a new scan kernel.**
  `cylindrical_prefix.py` no longer calls the array path's scan: its kernel
  `cyl_radial_difference` gives one lane to each column of stored words (each part of a
  complex column) and walks the rows in order, restarting MEEP's `f_rderiv_int` at every
  chunk start of the division the run steps with and writing the difference a target cell
  reads from its own chunk's sum (Bz forward with Ep's zero wall row, Dz backward). The
  weights and `rinv` rows are the array path's own (`stepping._cylindrical_rderiv_rows`),
  so the kernel multiplies by `rinv` and divides by nothing; the chunk starts are
  `stepping._cylindrical_scan_starts`, and a division whose r cuts differ along z is
  refused by name to the array path. A parallel scan associates the sum differently and is
  another word on most rows, so the scan stays column-serial; `--fmad=false` is
  correctness.
- **The curls take the root's forms.** Both curl pairs import the root preludes and decode
  the chunk record once (the split-field recurrence where the component's chunk carries
  its dsigu layer, the dsig-only form elsewhere in a layer, plain `f - curl` outside one;
  MEEP's left-to-right stencil with phi's partner absent); Bz's and Dz's whole curl is
  `dtdx` times the radial difference. On complex storage the i*m/r coupling is a pass of
  its own after the curl, the partner's quarter turn by the array path's realnum
  `rinv = the_m/r`, through the additive ladder keyed on the target's dsig and dsigu
  layers; the |m| = 1 axis-row increments and the m = 0 `Dz` post-add take MEEP's double
  product with the double Courant, rounded once.
- **The fused products derive from those texts.** The real and complex B->H and D->E pairs
  launch the scan before their fused kernel and decode one record over both halves; the
  two H->D welds step `update_H` into scratch, scan the scratch Hp and launch the
  certified curl -- three launches, as many as the D side's own baseline under the
  per-chunk sum, so both stay `INSTALLABLE = False`. `fused_pairs.py`'s cylindrical plans
  bind the composition's record (the magnetic pairs also the layer, for the run's first
  `update_H`), and `own_cell_hoist_reference.json` is re-cut for the cylindrical texts.
- **Gates.** The new `parity/meep_gpu/gate_cuda_cylindrical_scan.py` compares the scan
  with the array path on every division and both storages and plants its defects (a
  start one row late, a dropped segment, the other side's `ir0`, a dropped wall row).
  `gate_cuda_cylindrical_real.py`, `gate_cuda_cylindrical_complex.py` and the two H->D
  gates bind the record; their grouping mutations plant the grouping used until 1.0 and
  stay must-catch (KERNELS.md section 7.1), and each arms the section 7.2 battery. The
  complex expansion-arm swap is a declared null: both names emit one source. The four
  cylindrical families of the shared `gate_cuda_fused_complex_pairs.py` bind the record,
  count the scan as the companion launch it is, seed mid-run states (`h_separated`), arm
  the section 7.2 battery and the two inverted groupings, and read the D seam's
  metallic-code leg as live only where z is walled. Three rows of the real curl gate's
  step_D battery, and the wall-mask row on its periodic-z spec, became unobservable
  under the per-chunk forms (the auxiliary contract on the axis row and the m = 0 axis
  rules) and are declared with their algebra (KERNELS.md section 7.1).
- **Tests.** The new `cuda_kernels/test_cylindrical_chunk_forms.py` runs the scan, every
  curl text and the fused and H->D sequences through the C interpreter against the array
  path, word for word, on layered, three-chunk, periodic-z and accurate-near-axis
  divisions at a Courant that is not a float32 word, with planted signed zeros, and plants
  each defect the battery names.
- **Records** (`certification.json`, `fingerprints.json`) still describe the previous
  texts, pending the certification round.

### A subnormal policy installed after CuPy has compiled now reaches those kernels

`meep_gpu/subnormal_policy.py`: installing a policy dropped CuPy's in-process kernels with
`cupy.clear_memo()` alone, which empties CuPy's memoized functions but not the compiled
kernels a ufunc or an elementwise kernel keeps on itself. A lift compiles
`cupy_multiply__float32_float32_float32` before dispatch installs `keep` at the
configuration freeze, so that kernel kept CuPy's own `-ftz=true` for the life of the
process. Since the PML forms are chosen per structure chunk, stepping reuses that
signature, and the shipped route diverged from the array path in the subnormal range on
11 of 15 landed-family cases of the Triton route campaign.

- **The drop reaches every CuPy kernel object.** The install empties every live ufunc's
  `_kernel_memo` and every elementwise kernel's `_elementwise_kernel_memo`
  (`CUPY_OBJECT_KERNEL_MEMOS`), records the entries it dropped, and counts the compiled
  kernels still alive afterwards. A CuPy whose classes no longer carry those attributes is
  refused by name. An uninstall drops the same caches, so kernels compiled under the strip
  do not outlive it.
- **The keep token is in CuPy's cache key.** The key is computed above the option strip,
  so a dropped kernel rebuilt in a keep directory the lift had already compiled into read
  the pre-install binary back. Under `keep` the install folds `CUPY_KEEP_CACHE_KEY_TOKEN`
  into the key through `extra_source`, which CuPy hashes and passes to no compiler.
- **Measured** on one RTX A6000: a float32 `x * y` compiled before the install returns
  0 before and 9.99995e-41 after, through `cp.multiply`, `*` and an elementwise kernel, in a
  plain and in a pre-set keep cache directory (3 of 3 each; the drop alone gave 0 of 3 in
  the keep directory); a lifted 2-D PML driver holds 48 compiled kernels before the drop and
  0 after. On the route campaign's legs the shipped order now matches the policy installed
  before the lift: on Triton `shipped` and `shipped_expansion_probe`, the 14 cases that
  pass with keep installed before the lift PASS-FUSED with 0 of 12,459,564 final-state
  words differing from the array path, on CUDA `cuda_shipped` 10 of 10 with 0 of
  10,927,936, and every other case's verdict and differing words are identical to the
  keep-before-lift legs: 37 of 37 cases on `shipped`, 39 of 39 on `cuda_shipped`, and 35 of
  37 on `shipped_expansion_probe`, whose two cylindrical cases have no keep-before-lift
  run (neither fused; the dispatched leg differs from the array path in 0 of 716,800
  words). Over the nine
  end-to-end cases run to their end on the CuPy array path, installing after the lift
  differs from installing before it in 0 of 306 arrays and 0 of 11 flux spectra, against
  86 of 306 and 9 of 11 without this change.
- **Not reached:** a `RawKernel` the caller compiled before the install keeps its compiled
  kernel in a private attribute and keeps that kernel's policy. This package's own
  `RawKernel` memo is keyed on the policy, so it is never served across an install.
- **0.9.2:** the release bytes do not show the defect in any configuration measured:
  over the nine end-to-end cases, an install after the lift (as dispatch does it, with a plain or a
  pre-set keep cache directory) differs from one before it in 0 of 302 arrays and 0 of 11
  spectra. Its stepping reuses at most 3 of the 14 to 30 kernels the lift compiles, and
  none is float32 arithmetic that can meet a subnormal: a same-type copy, comparisons of
  PML coefficients with 1.0, a conversion of integer row counts, and integer and boolean
  kernels. The defect was latent there; a 0.9.x run is exposed only when, before the
  policy is installed, the same process compiled a float32 arithmetic CuPy kernel with
  exactly the specialization a later operation uses, such as the caller's own CuPy work.
  Installing the policy before the first CuPy compile avoids it on 0.9.x.

### The bit-parity coverage board is re-cut on series entries 01 to 73

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..73; the previous cut was on 01..70. Only its four generated
files change. Its test outcomes are the CUDA and Triton test files' cases of the package
suite (CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the
device run (Apple GPU), both run on this tree. `dev_evidence.json` keeps entry 71's NVIDIA
records, whose subjects (module bytes, emitted text, array path) this batch does not move,
and replaces its Metal records with the legs re-run at the integration of entries 72 and 73.

- **The kernel tables** (N / D, refused rows in brackets; the cut on 01..70, then this one):
  (a) the chunk-record select, CUDA 35 / 41 (+15) -> 35 / 41 (+15), Triton 32 / 56 (+2) ->
  32 / 56 (+2), Metal 52 / 59 (+5) -> 60 / 60 (+4); (b) MEEP's A form, CUDA 37 / 43 (+22) ->
  37 / 43 (+22), Triton 39 / 61 (+7) -> 39 / 61 (+7), Metal 58 / 67 (+10) -> 66 / 68 (+9);
  (c) host word identity, CUDA 12 / 18 (+47) -> 12 / 18 (+47), Triton 22 / 48 (+20) ->
  22 / 48 (+20), Metal 20 / 32 (+45) -> 27 / 30 (+47).
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4 (+61), Triton keep
  2 / 2 (+66), Metal flush 3 / 3 (+74), each a whole-state compare against the NumPy array
  path of this tree's module bytes, emitted texts and array path.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53 (+12), Triton 0 / 55 (+13), Metal
  0 / 74 (+3)) until the certification round re-gates.

### The Metal folded off-diagonal dispersive and complex no-absorber off-diagonal rows step MEEP's double row in software binary64

Ruling R7's software-binary64 row reaches the last two Metal off-diagonal `update_E`
families, and their arms are wired:

- **`metal_kernels/folded_offdiag_dispersive_update_e.py`** forms the off-diagonal
  single's double row over the ordered `D - P0 - P1 ...` gathers with the single's own
  emitter (`offdiag_update_e._row_assembly`): `gs*us` in float32, widened; each live
  term's float32 pair sum wall-masked, scaled by MEEP's double `0.25` and added left to
  right where the cell's chunk keeps the term; one rounding. The six presence words ride
  in its `Params` before the chunk record (`OffdiagPresence kw`; `params_record_dtype`
  116 -> 140 bytes; `params_words(shape, chunk_bounds, presence)`, both required), the
  predicate's last clause is the presence clause on the folded codes, and its R7 clause is
  gone. `offdiag_update_e._wall_mask`, the total-level wall mask only this module emitted,
  is deleted.
- **`metal_kernels/complex_no_pml_offdiag_update_e.py`** forms the double row per part,
  as `complex_folded_offdiag_update_e.py` does, and connects a Bloch phase on the
  same-index operands' plane 0 as the array path does (`stepping._bloch_ghost_operand`):
  the partner pair's own word takes `eikna`, the formed product `conj(eikna)` on the
  component's own axis, each a copy, a per-part negation or the unfused product. The
  phase on the shifted read, and the emitter's refusal of the negation kind, are gone.
  The chunk record binds at buffer 25 and the presence words last at 26 (27 of 31
  bindings); `plan_metal_complex_no_pml_offdiag` takes a required keyword `chunk_bounds`.
  `R7_REFUSAL` and `r7_reasons` are deleted.
- **The stencil weld `complex_no_pml_offdiag_fused_electric_pair.py`** lifts that row;
  its `Params` carry the presence words before the record (itemsize 192 -> 216) and its
  predicate passes the composer's record to the constitutive half.
- **Gates.** `gate_metal_folded_offdiag_dispersive.py` runs the row on generic states
  (`rows`), the presence leg (rows zero in the PML chunks with `D - P = -0`; presence
  ignored, cleared and forced all-kept caught), the float32 row and the terms summed
  first as must-catch mutations, and an `engine` leg, beside its exact-state `b_select`
  and mutations; its `r7` leg and withheld row leg are gone.
  `gate_metal_complex_no_pml_offdiag_update_e.py` runs its product leg on generic states
  (a negated-connection case added), mutation controls, the connection and float32-row
  mutations and an `engine` leg; `r7` and `row_form` are gone.
  `gate_metal_offdiag_stencil_welds.py` runs every leg on the complex no-PML weld.
- **Composition.** `metal_composition_matrix.py` expects `folded off-diagonal dispersive
  PML E` and `complex no-PML off-diagonal` on `update_E` again, both in
  `EXPECTED_WINNERS`.
- **Certification.** The kernel-source digests of these modules move;
  `fingerprints.json` and every ledger are unchanged, pending the certification round.

### The Metal cylindrical kernels step MEEP's cylindrical forms

The Metal cylindrical families -- `metal_kernels/cylindrical_real.py`,
`cylindrical_complex.py`, `cylindrical_complex_scan.py` and the six products that lift
them (`cylindrical_real_fused_magnetic_pair`, `cylindrical_real_fused_electric_pair`,
`cylindrical_real_fused_hd_pair`, `cylindrical_fused_magnetic_pair`,
`cylindrical_fused_electric_pair`, `cylindrical_complex_fused_hd_pair`) -- now step the
array path's cylindrical forms and bind the chunk record:

- **The radial sum is MEEP's, per chunk.** Bz and Dz differentiate `f_rderiv_int`
  summed in every structure chunk from that chunk's first row, with `rinv` multiplied
  (`stepping.cylindrical_rderiv_difference`). On real storage, and in the complex H->D
  weld's lead launch, a column-serial scan kernel walks each (phi, z) column, restarts
  the sum at each chunk start the array path names (`stepping._cylindrical_scan_starts`,
  bound as an `int32` vector) and stores the difference a row reads from its own chunk's
  sum; the other complex plans call the array path's own difference. A division whose r
  cuts differ along z is refused by name.
- **The i*m/r coupling is a pass of its own.** After the curl's per-chunk forms and
  before the stores, the increment -- a quarter turn per part of the partner by the
  float32 `rinv` word (`stepping._cylindrical_imr_increment`) -- goes through MEEP's
  additive ladder, masked to the owned cells. No complex product is formed.
- **The axis rules.** MEEP forms the on-axis products with Courant in double and rounds
  once; the kernels form them in float32, which is the same word only when Courant is a
  float32 word, so every other Courant is refused by name
  (`cylindrical_real.courant_reasons`) and those slots take the array path.
- **The run's first update_H.** `cylindrical_real_fused_magnetic_pair` now refuses it by
  name while B, its auxiliary or E holds a word other than +0
  (`launch.h_start_reasons`), as every other B->H product does.
- **Gates.** The nine cylindrical gates are re-cut on these forms: the curl's grouping
  mutations plant the package's previous grouping (KERNELS.md §7.1), the §7.2 battery is
  armed on the per-chunk select, the vacuity floors exempt exactly the auxiliaries the
  form contract never touches (those governed by phi), the H->D refusal legs state that an
  integrated source stands no withdraw while a source class that still carried one is
  refused by name, and the rows that only a signed zero or a store the axis rule
  overwrites could carry are declared nulls with their reasons.
  `gate_metal_whole_step.py`'s cylindrical rows take a float32-word Courant.

### The bit-parity coverage board is re-cut on series entries 01 to 70

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..70; the previous cut was on 01..61. Only its four generated
files change. Its test outcomes are the CUDA and Triton test files' cases of the package
suite (CPU-only shim, every marker, dispatch off) and the Metal test files' cases of the
device run (Apple GPU), both run on this tree: 208 test files, 11,860 cases, 0 of the rows'
test files not run. `dev_evidence.json` holds only the device legs re-run at the
integration of entries 66 to 70.

- **The kernel tables** (N / D, refused rows in brackets; the cut on 01..61, then this
  one): (a) the chunk-record select, CUDA 34 / 40 (+16) -> 35 / 41 (+15), Triton 19 / 56
  (+2) -> 32 / 56 (+2), Metal 51 / 59 (+5) -> 52 / 59 (+5); (b) MEEP's A form,
  33 / 39 (+26) -> 37 / 43 (+22), 25 / 61 (+7) -> 39 / 61 (+7), 57 / 67 (+10) ->
  58 / 67 (+10); (c) host word identity, 12 / 18 (+47) -> 12 / 18 (+47), 17 / 57 (+11)
  -> 22 / 48 (+20), 18 / 32 (+45) -> 20 / 32 (+45).
- **What moved.** The four CUDA complex no-absorber products refused on 01..61 emit again
  (entry 66). Thirteen Triton rows land in (a) and fourteen in (b): the six cylindrical
  modules and the dispersive and ADE families (entries 67 and 68); five of them land in
  (c), and nine fused or lifted ones are refused there because their test files measure no
  arithmetic of them (their word evidence is in their device gates and probes). Metal's
  nonlinear update_E and off-diagonal update_E land (entries 63 and 64).
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against
  the NumPy array path, re-run on this tree's module bytes, emitted texts and array path.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates; against MEEP, unchanged: 2 / 2 measured
  families identical under keep, 8 refused, flush refused (no minted set).

### Tier C reads the whole corpus

`parity/meep_gpu/meep_identity_tier_c.py compare` now runs Table 2's rows, through a new
cases file, `parity/meep_gpu/meep_identity_corpus.py` (a manifest derived from the seven
2026-08-09 legs, each row rebuilt as it was measured, its grid checked after the lift).
The runner reads every grid those rows lift onto:

- **Dcyl**: MEEP's stored words on `iveccyl(r, z)` under MEEP's cylindrical component
  names; **Bloch-periodic axes**: the stored window (`Grid.stored_window_low`), so cell 0
  of a Yee-shift-0 component reads MEEP's owned top point; **folded periodic axes**: the
  far ghost slot reads its image about the second mirror, as the lift samples it (a ghost
  cell, reported and never judged); a 0-D cell reads as one word.
- **Flush on Apple silicon**: the thread is driven through the package's lever (FZ), as
  R13 measured; the stored-word reads leave the mode and return to it, so a subnormal is
  never read as zero. Subnormal stored words are counted on both sides.
- **Rows the capsule schema cannot carry** (DFT monitors) are word-compared and record the
  refusal; a case refused before its directory exists still writes its error file;
  `MEEP_IDENTITY_CASE_TIMEOUT` raises the per-case ceiling; the first difference names
  every differing component and the result lists every differing compared step.
- The `parameterized` stand-in in `parity/meep_gpu/test_shims` goes on the path after
  the environment's, as the survey that named the rows placed it.
- **MEEP's point read loses signed zeros of complex words**: `fields::get_field` returns
  `(1, +0) * word` in `complex<double>` (monitor.cpp:152), so an imaginary `-0` beside a
  real part `>= +0`, or a real `-0` beside an imaginary part `<= -0`, reads `+0`. The
  result now also reports `stored_read_equivalent`: the stored compare after that read is
  applied to the package's words (`meep_point_read`). It is reported, never judged: a row
  that differs only there is consistent with MEEP's read, and its stored words are not
  decided by this reader. Real words are read exactly.

### Tier C reads MEEP's stored words exactly

`parity/meep_gpu/meep_identity_tier_c.py compare` no longer reads MEEP's words through its
point read. `fields::get_field` (monitor.cpp:145-157) returns `(1, +0) * word` in
`complex<double>`, which reads an imaginary `-0` beside a real part `>= +0` (and a real
`-0` beside an imaginary part `<= -0`) as `+0`, so on complex storage a Tier C compare
could neither confirm nor refute a signed zero. `MeepReader` now copies each word, with no
arithmetic, from `fields_chunk::f[c][0]` and `f[c][1]` (the class's first member,
meep.hpp:1476) of the chunk `get_field` would read, at that chunk's
`grid_volume::index(c, ivec)`; a cell no chunk owns but a chunk holds (the low plane of a
metallic wall, which MEEP never writes after allocating it as zero, or the not-owned low
plane of a Bloch axis, which the chunks' connections fill) is read from the first chunk
holding it, every holder agreeing.

- **The point read is the exact read's check**: at every owned cell, MEEP's point read
  applied to the exact word must equal `get_field`'s word for word, or the case errors;
  so must every holder of a wall cell agree, and a compared cell no chunk holds is
  refused by name. `record_meep_identity.py` refuses a mint on any of the three.
- **A held B word of a chunk whose B array is its H array has no B of its own**:
  `fields_chunk::alloc_f` sets `f[H] = f[B]` (fields.cpp:480-494) and `update_eh` gives H
  an array of its own only for mu, a polarization or a PML conductivity along the
  component (update_eh.cpp:165-171). At such a chunk's not-owned point B's connection and
  then H's write the same word (step.cpp:72, 90), so the word left there is the owned
  image's H, while MEEP's B at the point (`get_field`, through the Bloch image,
  boundaries.cpp:207-230) is the image's B, a B that MEEP itself never reads
  (boundaries.cpp:369-377). Where that shared word is not B's point read, the reader reads
  B's point read and requires the shared word to be H's point read
  (`held_aliased_unexplained`), and refuses the point on complex storage, where the point
  read cannot carry a signed zero (`held_aliased_complex`); `record_meep_identity.py`
  refuses a mint on either. `held_aliased` and `held_aliased_substituted` are reported.
- **Reported, never judged**: `stored_read_equivalent` keeps the point-read compare;
  `meep_point_read_loss` counts the compared words MEEP stores that its point read drops.
  Ghost cells and the smoke record keep MEEP's point read.
- **Rows**: the 13 rows the 2026-10-08 dry run left on signed zeros alone (4 Cartesian
  in flush, 9 cylindrical) hold: MEEP stores the package's signed zeros, and the
  differences were its point read's. Real-storage words are unchanged, so the Tier A
  fixtures do not move.
- **Tests**: a planted imaginary `-0` in MEEP's own array is read exactly and `+0`
  through `get_field`, on a 2-D Bloch and a Dcyl `m = 1` grid
  (`test_the_exact_reader_reads_the_signed_zero_meeps_point_read_drops`, resource
  `meep_identity_reference`); on a Bloch grid with a PML at one end, a held By of the
  chunk that keeps Hy == By reads MEEP's By where the shared word is Hy's
  (`test_a_held_b_word_of_a_chunk_whose_b_array_is_its_h_array_reads_meeps_b`, the same
  resource).

### Three Triton gate harnesses re-cut onto the 1.0 kernels

The pre-certification dry run found three Triton gates whose harness, not their kernel,
stopped them. Each is re-cut on the shipped 1.0 text, with its MEEP-grouping mutations
inverted (the mutant installs the grouping the kernel carried until 1.0) and kept
must-catch:

- **`gate_triton_fused_electric.py`.** The source-mutation copy of
  `kernels.fused_curl_constitutive_D` now compiles under a header that imports the
  `forms.py` helpers and declares the chunk-record constants the kernel reads (without
  them every copy failed to compile, and a launch error scored as a catch). The
  `drop_fw_store` needle is the masked `f_w` store, and `regroup_constitutive` installs
  the two separate accumulations `(f + kps*src) - kms*prev` in place of
  `forms.constitutive_form`. The needle text is split from the compile
  (`mutated_source`) and checked on the host.
- **`probe_triton_conductive_fused_electric_pair.py`.** `m_wall_clear_after_the_store` is
  scored with the magnetic source (`MUTATION_WITHOUT_ELECTRIC_CARRY`). Under the
  integrated electric source the driver's `f_minus_p` bracket re-runs `zero_metal_D`
  after the fused launch (`sources._boundary_passes`), which clears every wall word the
  rewrite leaves, so the mutation scored there changes no word.
- **`gate_triton_fused_offdiag_electric.py`.** The refutation subject is assembled from
  the chunk-keyed curl half and the per-chunk off-diagonal body: its header imports the
  forms, the record constants, `PRESENCE_LINES` and `_row_add` (the helper that replaced
  `_masked_row_sum`), its signature carries `cb`, `op` and the presence constexprs, and
  both routes bind the every-cell record and `ALL_PRESENT`. `flatten_curl_parens`
  installs the pairwise curl `dtdx * ((c_y - c) + (b - b_z))`.

### The Triton cylindrical kernels take MEEP's Dcyl forms, with a column-serial radial scan

The cylindrical kernels of the Triton table -- `meep_gpu/triton_kernels/cylindrical_triton.py`
(the real m = 0 curl), `cylindrical_complex.py` (the complex curl at every m) and the four
cylindrical fused pairs (`cylindrical_real_fused_magnetic_pair.py`,
`cylindrical_real_fused_electric_pair.py`, `cylindrical_fused_magnetic_pair.py`,
`cylindrical_fused_electric_pair.py`) -- now compute what the array path computes.

- **A column-serial scan kernel for the radial sum.** Bz's and Dz's curl is `dtdx` times
  MEEP's per-chunk radial difference of `f_rderiv_int`
  (`stepping.cylindrical_rderiv_difference`). The new `@triton.jit` kernel
  `cylindrical_triton.cyl_radial_difference` computes it on the device: one lane per
  (phi, z) column, and per stored part on complex storage, walks the r rows in order,
  restarts the sum at `+0` on each chunk's first row, multiplies by `rinv` rather than
  dividing, and stores the difference each target row reads from its own chunk's sum
  (forward on B, with Ep's stored wall row read as `+0`; backward on D). The plan binds
  the two row vectors from `stepping._cylindrical_rderiv_rows` and the chunk start per
  row from `stepping._cylindrical_scan_starts`, and refuses by name a division whose r
  cuts differ along z or that is not the division the chunk record describes. Every
  cylindrical plan launches the scan, then its curl or fused kernel; the host prefix
  and its row vectors are gone.
- **The per-chunk forms, from `forms.py`.** The other curls take MEEP's stencil with the
  record's partner presence (phi is invariant, so Bx/Dx take the one-partner form), the
  split-field recurrence on the cells whose chunk carries the target's dsigu layer and
  the dsig-only form elsewhere, with `fu` read and written there only; the fused pairs'
  constitutive halves take the root accumulation on the component's dsigw cells and
  `f = src` elsewhere, with MEEP's start of H on the magnetic side.
- **The i*m/r pass and the axis rules.** On complex storage the i*m/r coupling is a pass
  of its own after the curl's form, applied in register: the quarter turn of the partner
  by the host-built float32 row `the_m / r` (`imr_rinv_row`), masked to MEEP's owned
  cells and subtracted through the additive ladder. The m = 0 `Dz` add and the |m| = 1
  axis-row increments are `realnum(double(c) * double(v))`, with the double Courant
  words read from a bound `float64` vector (`courant_constants`). The two
  `tl.math.fma` sites of `cylindrical_complex` are gone, and the complex curl forms
  every product per part.
- **Gates.** `gate_triton_cylindrical.py` and `gate_triton_cylindrical_complex.py` compare
  the plans with `stepping` itself, word for word, on random full-volume states, a
  planted signed-zero and subnormal state and a +-0 lattice, with a planted one-word
  flip that must read as 1, over every m class, both z terminations and one and two r
  chunks. Their mutation tables invert the Appendix A grouping (the pairwise curl) and
  arm the section 7.2 battery (whole-grid recurrence, membership a row off, lattices
  swapped, `fu` stored on every cell, the predicate on coefficient identity), the scan
  (one segment, dividing, the first row not summed from `+0`, no wall row, the
  increment distributed over the product), the i*m/r pass, the axis rules, and two declared nulls (an unconditional
  `fu` load, the near ghost put back). The four pair probes are re-cut to the same forms.
- The cylindrical H-to-D welds (`cylindrical_fused_hd_pair.py`,
  `cylindrical_real_fused_hd_pair.py`, not dispatched) still lift the previous text and
  are declared in `tools/ci/pending_certification.txt` for their lane.

### The Triton dispersive and ADE kernels compute MEEP's polarization update

The dispersive and ADE kernels of the Triton table -- `ade_update_p` in
`meep_gpu/triton_kernels/kernels.py`, the pole-aware `update_E`
(`dispersive_update_e.py`) and its folded off-diagonal twin
(`folded_offdiag_dispersive_update_e.py`), the dispersive D/E pair
(`dispersive_fused_pair.py`) and its folded twin (`folded_dispersive_fused_pair.py`), the
one-susceptibility ADE state (`fused_ade_state.py`), the E->P chains
(`fused_ade_chain.py`, `fused_dispersive_chain.py`, `folded_offdiag_fused_ade_chain.py`,
`complex_fused_ade_chain.py`), the complex ADE kernel (`complex_ade.py`) and the
no-absorber ADE family (`no_pml_ade.py`) -- now compute what the array path computes.

- **MEEP's update_P.** Every recurrence is MEEP's isotropic expression,
  `gamma1inv * (((p * two_minus_od) - (gamma1 * pp)) + (omega0dtsqr * (s * w)))`, one
  float32 rounding per operation, on the four `realnum` words
  `PolarizationState.constants` holds (`Susceptibility.realnum_constants`). The folded
  three-coefficient form and the `(c_now, c_prev, c_drive)` triple are gone; every ADE
  kernel takes the four words as float32 scalars, and complex storage steps the same
  expression per part. One predicate clause, `coverage.polarization_constant_reasons`,
  refuses by name a state whose four words are missing, non-finite or not float32 words,
  for every Triton ADE family.
- **The per-chunk forms, from `forms.py`.** Each pole-aware `update_E` and each E half
  selects the constitutive form by the chunk record `cb` on the component's own dsigw
  line: `E + ((kps*src) - (kms*prev))` on the layer's chunks and `E = src` elsewhere, with
  `f_w` and the old E read and written there only. The dispersive pairs' curl halves are
  the root fused D kernel's, statement for statement (MEEP's stencil with the record's
  partner presence, the recurrence on the dsigu cells and `fu` written there only).
- **MEEP's drive.** `update_P`'s drive is MEEP's `w = f_w ? f_w : f` per chunk
  (`Fields.drive_field`): a plan that reads the drive from memory reads that method's
  composed words, and every chain hands the recurrence `src`, which the E half has just
  stored into `f_w` on the layer's chunks and as the field everywhere else.
- **The E->P seam.** The electric `f_minus_p` restore (`sources.f_minus_p_end`) stands
  between `update_E` and `update_P` in the driver's step. It writes D alone, and
  `update_P` never reads D, so the chains form E and P in one launch and the restore runs
  after them; the chains' seam declarations and their tests say so.
- **Gates.** The ADE grouping mutations are inverted (KERNELS.md section 7.1): the
  previous folded three-term form planted back is must-catch on every chain and on
  `ade_update_p`; MEEP's sum regrouped, the drive product re-associated and two words
  crossed are must-catch. The section 7.2 battery on the E tails (`f_w` stored on every
  cell, the integer record line, the select removed) is must-catch, and the unconditional
  `f_w` load is a declared null. The drive re-read from `f_w` on every cell is now
  must-catch, because off the layer's chunks `f_w` keeps its word.

### The hand-written CUDA complex no-absorber lifters step MEEP's stencil and the D curl's connection

The three CUDA products that lift the complex no-absorber curl
(`cuda_kernels/complex_no_pml_kernels.py`) into one launch --
`no_pml_complex_fused_electric_pair.py` (`step_D` and `update_E`),
`complex_no_pml_three_slot_dispersive_weld.py` (`step_D`, `update_E` and `update_P`) and
`complex_no_pml_offdiag_fused_electric_pair.py` (`step_D`, `zero_metal_D` and the
off-diagonal `update_E`) -- are re-derived onto that curl's text:

- **MEEP's stencil with partner presence.** The lifted curl blocks call the certified
  `curl_stencil` helper (`((g1s - g1) + g2) - g2s`, with MEEP's one-partner forms on an
  invariant axis), and each kernel takes the run's chunk record (`cb`, `int32[25]`, last
  parameter) for its partner-presence word; no membership row is read without an
  absorber. Each launcher requires `chunk_bounds` and checks it with
  `complex_pml_kernels.chunk_record_argument`; the fused plans in `fused_pairs.py` bind
  the composition's record.
- **The D curl's connection.** The Bloch table each product binds is the D curl's own
  (`bloch_phase_arguments(grid, True)`: the connection kind per axis and the float32
  parts of `eikna`). The off-diagonal no-absorber weld binds the partner-axis table once
  for both halves and the near product's own-axis table (`conj(eikna)`) beside it, with
  the live rows' per-chunk presence words, exactly as its folded twin does, so the
  stencil weld's no-absorber arm derives again.
- **Both arm names emit one body** on the D->E pair, as on the curl it lifts.

`test_complex_no_pml_chunk_forms.py` runs each product's emitted text through the
complex interpreter against the array path word for word (phased, Brillouin-edge,
two-dimensional, conductive, one- and two-pole, walled and corpus-row cases, with planted
signed zeros) and plants seven text defects that must change a word. The gates re-cut
with them: `gate_cuda_fused_complex_pairs.py` binds the record on the no-absorber family,
adds the inverted stencil grouping and the partner-presence mutation, and scores the arm
swap as a null control there; `gate_cuda_complex_three_slot_weld.py` binds the record,
adds the inverted stencil grouping and names its host phase mutation for what it binds
(`b_curl_connection_table`); `gate_cuda_complex_offdiag_stencil_welds.py` drives both
products and scores the pair-connection and old-grouping mutations on both.

### The Metal folded whole-step fixtures are mid-run states

`meep_gpu/test_metal_folded_offdiag.py` and `parity/meep_gpu/gate_metal_folded_offdiag.py`
build their seeded states with `Fields.h_separated`, as `metal_composition_matrix`'s
builders do. Without it, the whole-step test and the gate's `whole_step` leg compared the
array path's first `update_H`, which starts H and `f_w` from B on the dsigw cells
(`stepping.separate_h_from_b`), with the certified constitutive kernel the folded arm
re-admits, which steps the forms alone (`ConstitutivePlan.starts_h_from_b` is False; the
composer keeps it off a run's real first `update_H` by `launch.h_start_reasons`). No
kernel and no array-path module changes.

- **What diverged.** On `2d_fold_Y_even_periodic` the step-0 divergence (Dz 92, Hx 72,
  Hy 32, fu_Dz 32 words before `update_E`) is the start alone: the array path as built
  against the same state mid-run moves exactly those words. MEASURED on the Apple GPU,
  every pass compared word for word over 6 cases and 6 steps: the first differing word is
  `update_H`'s, on H only, and every one sits on a cell the region plan steps with the
  dsigw form (Hx's x layer columns, Hy's far-face rows including the far ghost slot); B,
  the two folded curls, both mirror fills and every `f_w_H` word are identical. With the
  state mid-run the curls, fills and `update_H` read 0 differing words on every case.
- **When.** Base `fa59c2f` (0.9.2) and base + 01..08 read 0 differing words at every pass:
  the test was certified there, not declared. Entry 09 (AP3) moved the array path's curl
  grouping, the AP3 integration declared these ids on the device, and every later
  signature hid behind that declaration: entry 13 (AP4) added the start, and from entry
  18 (KM1), whose Metal curl and constitutive match the array path, the start is the
  first divergence.
- **Nothing is left.** With the software binary64 off-diagonal row on `update_E`
  (KM7s), the six whole-step ids and the gate's `whole_step` leg read 0 differing words,
  and the six ids leave `tools/ci/pending_certification.txt`. Without the fixture flag the
  same tree still diverges at step 0 in `update_H` (Hx 72, Hy 32 words on
  `2d_fold_Y_even_periodic`).
- **A host test pins the fixture.** `test_the_whole_step_fixture_is_a_mid_run_state`
  (six cases, no device) requires the builder's flag and that the start would move H
  words and nothing else on every case; it fails on the previous builder.

### The Metal nonlinear update adds its transverse sums left to right and forms |D|² in software binary64

The chi2/chi3 `update_E` kernel of the Metal table
(`meep_gpu/metal_kernels/nonlinear_update_e.py`, `nonlinear_constitutive_step`) steps on the
GPU again, under owner ruling R7's choice of software binary64 over refusal, and computes
what the array path computes since "The nonlinear constitutive update is MEEP's arithmetic".
It replaces the refusal "The Metal base kernels read the chunk record and step MEEP's forms"
put in place: `R7_REFUSAL` and `r7_reasons` are removed and the Pade arm is wired.

- **The transverse sums associate left to right**, `((g[i] + g[i+s]) + g[i-s1]) +
  g[i+(s-s1)]` (`stepping._nonlinear_transverse_sums`), where the kernel paired the corners.
- **Dsqr is formed in binary64 and rounded once** (`stepping._nonlinear_dsqr`): `gs*gs` and
  `g1s*g1s + g2s*g2s` stay float32 operations; the helper `nonlinear_dsqr`
  (`DSQR_HELPER`) widens them, scales the transverse term by `0.0625` and adds them in
  software binary64 (`metal_kernels/softfp64.py`: `sd_widen_f`, `sd_scale_pow2`, `sd_add`,
  `sd_narrow_f`, each under the flush policy), and rounds the sum to float32 once.
- **The constitutive tail is the base kernel's, per chunk.** The kernel binds the chunk
  record last (30 of Metal's 31 bindings) and keys each component's form on its own axis's
  half-integer membership: MEEP's accumulation `E + ((kps*src) - (kms*prev))` with
  `f_w = src` on the cells whose chunk carries that layer, `E = src` everywhere else, with
  `f_w` neither read nor written there. Every plan builder takes `chunk_bounds` with no
  default.
- **Composition.** On a chi2/chi3 run with an active absorber the composer now fills all
  four slots, `update_E` with the Pade arm; `metal_composition_matrix`'s nonlinear rows
  expect it and `nonlinear` is an expected winner. The spine arms and the two welds that
  inherit them (`nonlinear_fused_magnetic_pair`, `fused_hd_pair`'s nonlinear widening) form
  no `Dsqr` and are unchanged. Under the held residency default a 2-D TE nonlinear run is
  still refused whole (its `update_H` stays on the array path, outside the door set), and
  steps on the array path.
- **Measured** on one Apple M1 Max, flush policy. `gate_metal_nonlinear.py` releases: 54
  identity cases with 0 differing words, 27 must-catch mutations caught on 79 of 79 armed
  cases with 4 null controls held, and the binary64 helper differing from
  `stepping._nonlinear_dsqr` on 0 of 1,048,576 band operands (the float32 form on all
  1,048,576). `gate_metal_fused_hd_pair_nonlinear.py`, whose composition now dispatches the
  Pade `update_E`, reads 0 of 145,728,000 words differing over 6 cases and 360 driver steps
  (every leg but the corpus lift). Lifted MEEP simulations (2-D TM, 2-D TE, 3-D, 64 steps,
  every stored component at four checkpoints) store 0 differing words against the array
  path, held and shipped residency alike, including a 3-D case at `chi3 = 1000` where
  putting the paired corners back moves 12,435 of 1,296,000 words in the array path and
  the same words on the GPU; at the physical `chi3` of the other cases it moves none,
  because `c3` stays far below float32's resolution of `1 + ...`. `update_E` alone, on
  110,592 to 4,096,000 cells, costs 1.39x to 2.11x its float32-`Dsqr` form and 0.65 % to
  1.45 % of the NumPy array path's time (one loaded host; not a timing of record).
- **Tests.** `test_metal_nonlinear_update_e.py` replaces its R7 tests with the kernel
  against `stepping.update_E` word for word on eight configurations, the binary64 helper
  against `stepping._nonlinear_dsqr` on its band, the band carried through the shipped
  kernel, the left-to-right corners, the per-chunk tail, the required chunk record and the
  wired arm. `parity/meep_gpu/test_gate_metal_nonlinear_host.py` (new) checks on the host
  that every source and record mutation of the gate lands, and that the `dsqr_band` class
  separates the two forms. `gate_metal_nonlinear.py` runs its Pade legs again: the
  package's previous forms are must-catch mutations (`four_corner_pair_association`,
  `previous_two_step_accumulation`, `dsqr_in_float32` on the new `dsqr_band` class), with
  the per-chunk battery and a new `dsqr` leg; its reference steps on a flushing host.
- **Certification.** The kernel's shader digest and host digests move; `fingerprints.json`
  and every ledger are unchanged, pending the certification round.

### The Metal off-diagonal update_E families step MEEP's double row in software binary64

Owner ruling R7 (2026-10-08) chose software binary64 over refusing the off-diagonal row
to the array path. The three off-diagonal `update_E` singles on Metal
(`metal_kernels/offdiag_update_e.py`, `folded_offdiag_update_e.py`,
`complex_folded_offdiag_update_e.py`) and the constitutive halves of the three PML
scratch welds that lift them now form the array path's row, and the three single arms
are wired again:

- **The row** is `stepping._offdiagonal_terms` and `_rounded_row` statement for
  statement, through `metal_kernels/softfp64.py` with the flush policy: `gs*us` in
  float32, widened; each live term's float32 pair sum `(near*u[i]) + (far*u[i+s])`
  wall-masked (each term's sum, before its scale, as the array path masks `summed`),
  scaled by MEEP's double `0.25` and added left to right where the cell's chunk keeps
  the term; one rounding to float32. Complex storage forms one row per part. The
  float32 `0.25f` row and its `total` are gone. A component with no live row keeps the
  float32 `gs*us`.
- **Per-chunk presence** (KERNELS.md §2). One 27-bit word per row slot
  (`struct OffdiagPresence`, members `kept01` .. `kept22`) binds last on the singles
  (buffer 29 on the real singles, 30 on the complex folded one, at Metal's ceiling) and
  rides in each weld's `Params` before the chunk record. The kernel reads bit
  `lx + 3*ly + 9*lz` of a slot's word, each axis labelled off the chunk record on the
  component's own Yee sub-lattice (a metallic axis's integer-lattice wall row 0 read at
  row 1). The host builds the words from the array path's own presence
  (`offdiag_update_e.presence_words` / `presence_for`, through
  `stepping._chunk_layout`), checks they expand back to it on every stored cell, and
  refuses by name (`PresenceRefusal`) a division whose record cannot carry them; each
  predicate carries that clause, and the harness builders take a required keyword
  `presence` (`ALL_KEPT` for rows no chunk deletes).
- **The complex folded row's Bloch connection** is the array path's: the partner
  pair's same-index word takes `eikna` on the partner axis's plane 0 and the formed
  product takes `conj(eikna)` on the component's own axis's plane 0, by MEEP's
  `connect_phase` (a copy, a per-part negation, or the unfused product); the shifted
  reads are plain wraps. The source takes the connection kinds
  `complex_fields.phase_arguments` returns. Bloch-phased complex folded runs, which
  raised at source emission, are admitted.
- **Ruling R7's refusal is gone**: `offdiag_update_e.R7_REFUSAL` and `r7_reasons` are
  deleted, every predicate's R7 clause with them, and the singles register `wired=True`.
  `metal_composition_matrix.py` and `test_metal_planner_composition.py` expect the
  off-diagonal arms on `update_E` again, the planted double-admits naming both
  claimants.
- **Two off-diagonal products keep the refusal, each under its own clause.** The folded
  off-diagonal dispersive product (`folded_offdiag_dispersive_update_e.py`), which
  appended the off-diagonal single's clause, now owns `R7_REFUSAL` and `r7_reasons()`,
  as the complex no-absorber off-diagonal family (`complex_no_pml_offdiag_update_e.py`)
  already did; its test and gate lift and read that clause, and its emitted text does
  not change. Both stay unwired until their rows are formed in software binary64.
  `gate_metal_offdiag_stencil_welds.py` keeps its R7 legs for the complex no-absorber
  weld alone (identity withheld by name, the singles, mutation and rotation legs
  against the certified singles with the family's float32 single on `update_E`, its
  clause lifted in the gate process only); the three PML welds run every leg.
- **Gates.** `gate_metal_offdiag.py`, `gate_metal_folded_offdiag.py`,
  `gate_metal_complex_folded_offdiag.py` and `gate_metal_offdiag_stencil_welds.py` run
  every leg the refusal withheld, on generic states, against the double row; the
  in-gate transcription forms the double row with presence. New legs and mutations:
  `presence` (rows deleted in the PML chunks, `D = -0`; presence ignored, cleared and
  forced all-kept caught), the float32 row (must-catch on two-term masks), the terms
  summed first (must-catch on wide-exponent rows), the quarter in float32 (a declared
  null under the subnormal-free precondition), and on the complex folded gate the
  dropped and misplaced Bloch connections; its mutants are compiled from each case's
  own specialisation with an unmutated control.
- **Certification.** The kernel-source digests of these modules move;
  `fingerprints.json` and every ledger are unchanged, pending the certification round.

### The bit-parity coverage board is re-cut on series entries 01 to 61

`parity/meep_gpu/build_parity_board.py` (the device-text generator of the next section)
cut the board in `parity/meep_gpu/parity_board/` again, on base + series 01..61; the next
section's figures are its cut on 01..49. Only its four generated files change. Its test
outcomes are the CUDA and Triton test files' cases of the package suite (CPU-only shim,
every marker, dispatch off) and the Metal test files' cases of the device run (Apple
GPU), both run on this tree: 207 test files, 11,833 cases, 662 failures, every one
declared in `tools/ci/pending_certification.txt`. `dev_evidence.json` holds only the
device legs re-run at the integration of entries 57 to 61.

- **The kernel tables** (N / D, refused rows in brackets; the cut on 01..49, then this
  one): (a) the chunk-record select, CUDA 15 / 47 (+9) -> 34 / 40 (+16), Triton 13 / 56
  (+2) -> 19 / 56 (+2), Metal 28 / 54 (+10) -> 51 / 59 (+5); (b) MEEP's A form,
  15 / 50 (+15) -> 33 / 39 (+26), 16 / 60 (+8) -> 25 / 61 (+7), 27 / 58 (+19) ->
  57 / 67 (+10); (c) host word identity, 9 / 54 (+11) -> 12 / 18 (+47), 15 / 67 (+1) ->
  17 / 57 (+11), 20 / 62 (+15) -> 18 / 32 (+45).
- **Where the refusals grew.** (c) credits a test file only where it executes the
  module's arithmetic, so 17 rows the earlier cut read landed and 59 it read as gaps are
  refused (CUDA 3 and 33, Triton 3 and 7, Metal 11 and 19). In (a) and (b), 12 CUDA
  lifters of the complex base (the complex beta, cylindrical complex and complex
  no-absorber products and the complex off-diagonal weld) refuse emission by name on
  this tree: each emits on entries 01..56 and raises after entry 57 re-derives the
  complex base, as that entry declares.
- **(d-dev), development evidence, NOT certification**: CUDA keep 4 / 4, Triton keep
  2 / 2, Metal flush 3 / 3, each a whole-state compare of 0 / 20,450,688 words against
  the NumPy array path, re-run on this tree's module bytes, emitted texts and array path.
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74)
  until the certification round re-gates; against MEEP, unchanged: 2 / 2 measured
  families identical under keep, 8 refused, flush refused (no minted set).

### The bit-parity board reads the device text each module compiles, and gains a development-evidence column

`parity/meep_gpu/build_parity_board.py` read columns (a) B select and (b) MEEP-order A form
off each module's own source text, which most kernel modules do not spell their device
arithmetic in: CUDA lifters derive theirs from the roots by anchored replacement, Metal
families splice the base bodies through templates, Triton kernels inline other modules'
`@jit` helpers. Both columns now read the text the module's builder EMITS, with no device.

- **The emitter** (`parity/meep_gpu/parity_board_emit.py`, one child process per table, no
  device library importable): CUDA records the source `cp.RawKernel` is handed, through
  each module's `device_sources()` or its own `_get_kernel` (NVRTC is never reached);
  Metal reads the MSL its emitters return after splicing (`enumerate_*sources`,
  `specialisations()`, or the public `*_source` emitters); Triton reads each `@jit`
  program with its helpers inlined from the source. The canonical sweeps and the
  per-module routes are data (`parity_board_emit.json`): real PML 3-D, each family's
  minimal case, wall clears, Bloch phases and both complex-multiply arms; a parameter with
  no declared domain is refused. A module with no emit route (a text transform, a host
  scan, a null arm) is read off its own source text, and its cells say so.
- **Two more witnesses on that text.** (a) is a gap when a field-word PML form (the
  recurrence's second stage, the constitutive accumulation) sits outside every per-cell
  select, which catches a lifter that splices the base's record decode beside a curl of
  its own that runs the recurrence everywhere; and both columns are a gap when a helper is
  called with an argument count none of its definitions takes (a call site a lift did not
  re-cut: a text the compiler would refuse). An emitter that raises on this tree is
  refused with its exception.
- **The record read must be bound.** A unit that reads the chunk record through `cb` in a
  function that binds no `cb` (no parameter, local or file-scope `cb`: a record carried as
  a member of the parameter block and read as if it were in scope), or that names the
  `ChunkBounds` type without defining it, does not compile, and both columns say so
  (`record_binding_problems`). The review of this board found three Metal rows that read
  (a) landed on base + series 01..49 by the read's token alone, whose texts the Metal
  compiler refuses (0 of 702). On the series this change lands on, those rows bind the
  record and read landed, and the witness changes the reason of six rows that were
  already gaps (CUDA conductive_bfast_fused_hd_pair and five Metal cylindrical pairs).
- **(c) credits a test file only where it executes the module's arithmetic**
  (`executes_arithmetic`): on CUDA it runs the emitted text through the host interpreter
  of `test_root_chunk_forms.py` (it defines an `Interpreter` or imports from that file);
  on Triton and Metal it compares with `meep_gpu.stepping` (it imports it). The review
  found (c) landing on files that test a lift, a predicate, a seam order or a census and
  say so in their own docstrings; on Metal, files whose only device cases check a
  predicate or a compile. A module whose files are all of that kind is refused, and a
  landed cell lists the files not credited. The rule reads each file's own imports: a
  comparison reached only through a helper is not credited, and an import of `stepping`
  for any purpose is (an upper bound). Against X4b's cut, read with this tree's test
  files, it refuses 17 rows that cut counted landed: CUDA fused_hd_pair,
  conductive_fused_electric_pair and folded_offdiag_fused_electric_pair; Triton
  conductive_fused_electric_pair, offdiag_fused_electric_pair and
  nonlinear_fused_magnetic_pair; Metal fused_electric_pair, fused_ade_chain,
  fused_hd_pair, conductive_fused_hd_pair, complex_fields, folded_complex_fused_hd_pair,
  folded_complex_fused_magnetic_pair, folded_fused_pair, folded_fused_magnetic_pair,
  folded_fused_dispersive_pair and no_pml_curl. CUDA ade_kernels stays landed: entry 50's
  `test_dispersive_chunk_forms.py` executes its launch text.
- **Calibration.** On BASE fa59c2f every emitted unit reads 0 chunk records and 0
  MEEP-order sites with 0 call-arity problems and 0 refused modules (CUDA 421 units of 61
  modules, Triton 81 of 66, Metal 2,525 of 75); on this tree every module that reads
  landed was edited by a series entry or emits text that changed (0 exceptions).
- **(d-dev), development evidence, NOT certification.** Integrators record the device legs
  they re-run at integration with `parity/meep_gpu/record_dev_evidence.py` in
  `parity/meep_gpu/parity_board/dev_evidence.json`: words compared and differing, the gate
  or smoke, the date, the host class. A record binds the module's bytes, the digest of its
  emitted text and the array-path oracle; the writer refuses a leg whose subject moved
  during it ("the evidence subject and the working tree disagree") and the board counts a
  record only while its subject is live, so an edit to the module, the base it lifts or the
  array path drops it. (d) and (e) never read it; (e) is unchanged.
- **The board on base + series 01..49**, as committed with this change and re-cut on the
  merged series by the next entry (outcomes re-recorded on that tree: 202 test files,
  779 failures of 10,701 tests, all declared): (a) CUDA 15 / 47, Triton 13 / 56, Metal
  28 / 54, where the source-text generator on this same tree read 8 / 47, 12 / 55, 22 / 48;
  (b) 15 / 50, 16 / 60, 27 / 58, where it read 6 / 15, 15 / 55, 11 / 24 (refused 15, 8, 19
  where it refused 50, 13, 53); (c) 9 / 54, 15 / 67, 20 / 62; (d-dev) CUDA keep 4 / 4,
  Triton keep 2 / 2, Metal flush 3 / 3, each record a whole-state compare of
  0 / 20,450,688 words against the NumPy array path; (d) and (e) 0 everywhere. 93 rows
  change state against the source-text generator's reading of this tree.

### The harness hand references are the 1.0 array path again

The hand references the gates compare their kernels with, and the five
`validate_*_reference_vs_stepping.py` validators that pin those references against
`stepping`, still transcribed the 0.9.2 groupings and stepped the split-field forms on every
cell. On entries 01 to 44 the PML validator read 0 of 308 cases and the folded complex one
0 of 17; both now pass, beside the three that already did.

- **`probe_fused_kernel_bit_identity.py`** (the references the hand track, the Triton PML,
  fused-electric and folded gates and the dynamic-loop experiment share). The curl's
  `array_order` is MEEP's left-to-right stencil, with the one-partner forms where the chunk
  record's presence word clears a partner; the previous grouping is the control
  `pairwise_order` (renamed from `kernel_order`, so the artifact keys are now
  `vs_pairwise_order` and `array_vs_pairwise_order_reference`). Every PML reference takes a
  chunk record and keys each cell's form on it: the split-field recurrence and the dsigw
  accumulation where the record holds the cell, the dsig-only curl and `f = src` elsewhere,
  the auxiliary untouched there. The constitutive accumulation is MEEP's
  `f + ((kps * src) - (kms * prev))`; the 0.9.2 two-accumulation form is the `regrouped`
  control. The gates hand the record their kernels launch with
  (`gate_chunk_records.every_cell_record`).
- **`validate_pml_reference_vs_stepping.py`** hands each reference the record of the region
  plan the array path stepped with (`gate_chunk_records.region_plan_record`), so the
  record's decoding is checked against `stepping` too; marks its seeded state as mid-run
  (`Fields.h_separated`); and checks the auxiliary contract (an auxiliary whose layer has
  no cell in the division must not move). 308 of 308 cases, 44 structurally invalid
  skipped. The Triton fused-electric entry's `numpy_reference_weld` leaf is re-recorded by
  `record_numpy_reference_weld.py --write`: one leaf, every entry's bound digest and live
  capabilities unchanged.
- **The folded complex gate's reference** (`gate_triton_folded_complex.py`) is the one the
  Triton complex kernels' change (below) re-transcribed onto the same forms: real
  coefficients scale each part, a Bloch connection lands on the unshifted operand's plane 0
  as MEEP's connection of a not-owned word, the stencil is MEEP's with presence, each cell
  takes its PML form from the record, beta is a pass of its own after the curl loop through
  the additive ladder, and the ghost fills apply each parity per part. A second
  transcription of it is not added beside that one. Its validator, handed the region plan's
  record: 17 of 17 rows.
- **The dynamic-loop experiment** (`cuda_dynamic_loop_arity.py`): its reference takes the
  per-chunk forms (validated against `stepping.update_E`), its seven kernels call the
  certified flagged `constitutive_apply` with each component's membership from the chunk
  record, each case carries the record its kernels launch with, and the test's host
  emulation follows.
- **On CuPy** every reference the validators call evaluates to the words it does on NumPy:
  0 of 199,642,302 compared words under each float32 subnormal policy on one device, and
  the two validators that step the array path on CuPy themselves read 17 of 17 and 16 of 16.
- **`tools/ci/pending_certification.txt`**: the 7 harness ids of the AP3 block's harness
  group and the dynamic-loop prelude id leave it (the 2 folded complex validator ids this
  change also clears left with the Triton complex kernels' change). No id joins it: the
  folded complex gate's order mutation, which per-part parities make a null, is already
  recorded as a null by that change. The cylindrical hand references wait for the
  cylindrical array path.

### The Triton complex no-absorber kernels step each part with MEEP's forms

The complex members of the Triton no-absorber family now compute what the array path
computes.

- **The conductive curl is the complex curl's body.**
  `meep_gpu/triton_kernels/complex_no_pml_conductive.py`'s kernel is, through the
  ownership mask, `complex_fields.bloch_pml_curl_step` statement for statement: MEEP's
  Bloch connection of the unshifted partner on plane 0 (a copy, a per-part negation or
  the unfused product, `complex_fields.connection_arguments`), and MEEP's left-to-right
  stencil `dtdx * (((g1s - g1) + g2) - g2s)` per stored part, with the one-partner forms
  where the chunk record's presence word marks a partner's derivative axis invariant.
  The conductive tail `((f * condfac) - curl) * condinv` keeps its three per-part passes.
  The pairwise bracket and the rotation of the shifted operand on the last plane are
  gone. The kernel takes the record `cb` after its last pointer; the plan binds it
  (the absorber-free division's on the engine route, a required keyword `chunk_bounds`
  on the harness route) and refuses a connection on an axis that does not wrap.
- **The conductive D->E pair re-lifts it.** `complex_conductive_fused_pair.py`'s curl
  half is the new conductive body's statements; the plan binds the record and the
  connection kinds the same way, and its engine builder takes the run's division.
- **The lossless curl's binding has no residual.** `complex_no_pml_curl.py` binds the
  complex curl under the absorber-free record and unit columns, so the kernel's
  dsig-only form steps every cell, and `((f * 1) - curl) * 1` is MEEP's plain
  `f - curl` on every word. The previous binding lost the `(-0, +0)` pattern; the gate's
  planted leg now requires it identical. The plan no longer zeroes its placeholder
  auxiliary before each launch: the record has no recurrence cell, so the kernel
  neither reads nor writes it.
- **The stored-E constitutive is unchanged**: `(D - P0 - P1 ...) * inv_eps` per part is
  already the array path's word, measured on the host and on a device.
- **Gates.** `gate_triton_complex_no_pml_conductive.py` adds a 2-D product row, a planted
  signed-zero leg and three kernel-text mutations (the previous pairwise grouping, the
  connection on the previous plane, the full bracket over an absent partner), and
  declares the opposite licence arm a null. `gate_triton_complex_no_pml_curl.py`'s
  adversarial leg becomes the planted leg (shipped identical, the previous binding
  caught; the previous leg never planted `step_B`, because its uniform-source fill
  overwrote the B targets), with record and presence mutations and a poisoned-auxiliary
  null. `probe_triton_complex_conductive_fused_pair.py`'s grouping mutation is inverted,
  its connection mutations re-cut to plane 0, and its expansion override declared a
  null.

### The Triton complex kernels step each stored part with MEEP's forms

The complex kernels of the Triton table -- `meep_gpu/triton_kernels/complex_fields.py`
(the complex/Bloch PML curl and constitutive), the two pairs that lift them
(`complex_fused_magnetic_pair.py`, `complex_fused_electric_pair.py`), the complex
off-diagonal `update_E` (`complex_offdiag_update_e.py`), the folded complex curl, fill
and constitutive (`folded_complex.py`) and the two folded complex pairs
(`folded_complex_fused_pair.py`, `folded_complex_fused_magnetic_pair.py`) -- now
compute what the array path computes.

- **Each stored part, with real products.** MEEP steps a complex field as two
  `realnum` arrays with real coefficients, and the array path forms the same products
  (`part_products.scale`): a real coefficient times each part, one rounding each. The
  six `tl.math.fma` sites of `complex_fields` and the zero cross terms are gone, and
  `enable_fp_fusion=False` still governs every launch. `_mul_field_left`,
  `_mul_coefficient_left` and `_rotate_field_left` keep their names for the sibling
  families that call them and now have one per-part body; the `EXPANSION` constexpr
  they take no longer changes a word.
- **MEEP's Bloch connection.** On a phased axis the shifted operand is a plain roll of
  the stored window, and plane 0 of the UNSHIFTED operand takes MEEP's not-owned word
  (`stepping._bloch_ghost_operand`): a copy within `part_products.CONNECT_TOLERANCE` of
  +1, a per-part negation within it of -1, spelled `* -1.0` (Triton lowers `-x` as
  `0.0 - x`, which gives `+0` for `x = +0`), and otherwise the unfused product with the
  phase's float32 parts, `conj(eikna)` on the B side and `eikna` on the D side.
  `complex_fields.connection_arguments` classifies each axis once
  (`CONNECTION_KINDS`), and a plan refuses a connection on an axis that does not wrap.
  The complex off-diagonal row connects its partner pair above the window and its near
  product below it.
- **The per-chunk forms, from `forms.py`.** The curl takes MEEP's left-to-right stencil
  with the record's partner presence, the split-field recurrence on the cells whose
  chunk carries the target's dsigu layer and the dsig-only form elsewhere, with `fu`
  read and written on the recurrence's cells only; the constitutive takes
  `f + ((kps*src) - (kms*prev))` on the component's own dsigw cells and `f = src`
  elsewhere, with `f_w` read and written there only. The complex off-diagonal row is
  the real off-diagonal row per part (`gs*us` in double, each term added in double
  with the wall mask, MEEP's `0.25` and the slot's per-chunk presence, rounded once).
  Every plan binds the chunk record `cb`: the engine builders bind the run's division
  (`chunk_layout`, forwarded by `plan_step` and the package wrappers), and the harness
  builders require a keyword `chunk_bounds` with no default.
- **MEEP's start of H.** The complex `update_H` plan, the complex magnetic pair and the
  folded complex magnetic pair carry `start_h` on the run's first `update_H`, as do the
  folded builders' constitutive plans, which now pass the run's `Fields`.
- **The folded fill is a per-part scale.** The array path writes a mirror ghost as
  `part_products.scale(parity, plane)`, so the fill and the folded pairs' carries
  multiply each part by the host-rounded parity word, which is exact. The near and far
  passes stay separate launches because the driver's wall clear sits between them.
- **Gates.** `gate_triton_complex.py`'s reference is re-transcribed to the forms above;
  its connection mutations are re-cut (the connection dropped, applied on every lane,
  on the previous wrapped plane, with a product sign error, its direction swapped, and
  the negation spelled as unary minus, scored on a sparse state whose stencils are all
  zero), its grouping mutations are inverted (KERNELS.md section 7.1: the pairwise
  stencil, the whole-grid recurrence and the previous constitutive grouping, each
  must-catch), and it arms the section 7.2 battery (membership one row off, lattices
  swapped, `fu` and `f_w` stored on every cell, presence ignored, constitutive select
  removed, `inv_eps` on the word index, and the unconditional `fu` load as a declared
  null). Its mutants are `complex_mutant_*` modules, and the mutation leg refuses a
  mutant whose kernel keys like the shipped kernel or an earlier mutant.
  `gate_triton_folded_complex.py` inverts its headline fill needle (the previous full
  complex product must be caught) and records the fused fill passes and the reversed
  axis order as structural nulls, because the per-part parity products commute; the
  four pair probes and the complex composition probe are re-cut to the forms and the
  connection, the folded pairs' parity-chain re-order and fold are declared nulls, and
  the composition probe's armed negation leg runs on a sparse Brillouin-edge state.
  `gate_triton_complex_offdiag.py` drops the live connections in its phase mutation,
  and records the PML tail substituted on an absorber-free run as a declared null: the
  tail takes the accumulation only where a chunk carries the component's layer, and no
  chunk of such a run carries one (measured identical on the device; its twin, the
  direct tail on a PML run, stays must-catch).
- **Not in this change, and red until their lanes:** `folded_complex.py`'s two beta
  curls (`folded_beta_pml_curl_step`, `folded_beta_bloch_pml_curl_step`) still step the
  previous forms, for the folds work. The sibling families that lift or compare a
  complex body (the complex and folded complex beta pairs, the cylindrical complex
  pairs, the complex no-absorber conductive curl, the complex and cylindrical H->D
  welds, and the CUDA complex test that reads these helpers) quote the previous text;
  their tests are on the pending list, by lane. Ledgers and fingerprints are unchanged,
  pending the certification round.
### The hand-written CUDA complex families step MEEP's per-chunk forms with part-wise products

The complex CUDA kernels -- `cuda_kernels/complex_emitter.py` (the complex/Bloch PML curl
and constitutive) with `complex_pml_kernels.py`, the folded family
`complex_folded_kernels.py` with its two ghost-carry emitters (`complex_fill_carry.py`,
`complex_electric_fill_carry.py`), the no-absorber curls `complex_no_pml_kernels.py`, the
complex and folded-complex fused pairs, the complex H->D weld, and the complex
off-diagonal `update_E` with the two stencil welds built on it -- now compute what the
array path computes (`part_products`, `stepping._apply_curl_regions`,
`_apply_constitutive_regions`, `_bloch_ghost_operand`):

- **Products are MEEP's, part by part.** A real coefficient times a field is one float32
  product per part and every sum is its own rounding. The six `__fmaf_rn` of the `FMA_V1`
  arm and the literal zero cross terms are gone, so both arm names emit one body; the arm
  argument stays, and still refuses an unknown name, until the probe-bound expansion
  licence chain is deleted in its own change. `probe_cuda_complex_fmad.py` now asks
  whether the two names emit one body and whether `--fmad=false` changes the image of
  every kernel.
- **The Bloch connection is the array path's.** Both shift helpers wrap plainly, and the
  operand read at the cell itself is connected on plane 0 of a phased axis: a copy within
  `part_products.CONNECT_TOLERANCE` of +1, a per-part negation within it of -1, otherwise
  MEEP's unfused `complex<realnum>(thephase) * word`, with `conj(eikna)` on the B curl and
  `eikna` on the D curl. `complex_pml_kernels.bloch_phase_arguments` returns the
  connection kind per axis (`CONNECT_NONE` / `CONNECT_NEGATE` / `CONNECT_PHASE`) and the
  rounded phase parts.
- **The per-chunk forms.** Every kernel reads the run's chunk record (`cb`, decoded once
  after the index decode, as the real root kernels do): the curl takes the split-field
  recurrence on a cell whose chunk carries the target's dsigu layer, the dsig-only form
  elsewhere in a layer and the plain `f - curl` outside one, with MEEP's left-to-right
  stencil and partner presence; `fu` is read and written only on the recurrence's cells.
  The constitutive takes `f + ((kps*src) - (kms*prev))` with `f_w = src` on the
  component's own dsigw cells and the output select `f = src` elsewhere.
- **The lifters derive from those texts.** The fused pairs and the folded fused pairs
  lift the flagged helpers with their anchors re-cut; the ghost carries key each carried
  statement's flag on its destination's membership row; the H->D weld passes update_H's
  flag at each recomputed cell and connects its register as the stored word would be;
  the no-absorber curls decode only the partner-presence word. The fused-pair plans in
  `fused_pairs.py` bind the composition's one record (the two magnetic pairs also the
  layer, for the run's first `update_H`).
- **The off-diagonal row.** `complex_offdiag_update_e.py` forms MEEP's row per part: each
  OFFDIAG term's float32 pair sums and products, scaled and summed in double and rounded
  once, a term added only where the cell's chunk keeps the row's `chi1inv` array (the
  27-bit presence words of the real family), with the two connection tables of the curl
  and the row. The PML arm of `complex_offdiag_stencil_weld.py` and
  `folded_complex_offdiag_fused_electric_pair.py` derive from it; the no-absorber arm
  (`complex_no_pml_offdiag_fused_electric_pair.py`) is refused at derivation, by name,
  until its own lane re-derives it.
- **Gates.** `gate_cuda_complex.py`, the folded, no-absorber, fused-pair, H->D, off-diagonal
  and stencil-weld gates and the two complex constitutive gates bind the record and the
  connection tables; their grouping mutations plant the grouping used until 1.0 and stay
  must-catch (KERNELS.md §7.1), and each arms the §7.2 battery where it lifts a body. The
  arm swap is a declared null.
- **Tests.** The new `cuda_kernels/test_complex_chunk_forms.py` runs every emitted text
  through a C interpreter against the array path, word for word, on box, slab, periodic,
  split and folded divisions with planted `-0` and subnormals, and plants each defect the
  battery names.
- **Not in this change, and red until their lanes:** the complex beta families, the
  cylindrical complex kernels and the cylindrical fused pairs lift complex texts the
  forms replaced (`cylindrical_complex_kernels.py` now refuses emission by name), and the
  no-absorber complex off-diagonal weld refuses at derivation; their tests are on the
  pending list, by lane. Records and fingerprints still describe the previous texts,
  pending the certification round.

### The Metal complex no-absorber curls step MEEP's stencil

The complex no-absorber curl body of `metal_kernels/complex_no_pml_conductive.py`, shared by
the lossless `complex_no_pml_curl.py`, forms the curl with MEEP's left-to-right bracket
`dtdx * (((g1s - g1) + g2) - g2s)` and its forms for a partner absent from the step plan,
`dtdx * (g1s - g1)` and `(-dtdx) * (g2s - g2)`, one product per part
(`stepping._curl_from_operands`). On a phased axis the UNSHIFTED partner carries the Bloch
connection on plane 0, as a copy, a negation per part or MEEP's unfused product
(`stepping._bloch_ghost_operand`). The statements are the complex PML curl's own lines
(`complex_no_pml_conductive.CURL_STATEMENTS`, `CONNECTED_REGISTERS`), replacing the pairwise
grouping and the shifted-read phase.

- **The partner presence is read from the chunk record.** The curl binds the record last as
  `constant ChunkBounds& cb` (buffer 20) and decodes its presence word through
  `no_pml_curl.bounds_block()`; no membership entry is read, since no chunk of a run
  without an absorber carries a layer.
- **Every builder takes the record, with no default.**
  `plan_metal_complex_conductive_no_pml_curl`, `plan_metal_complex_no_pml_curl`,
  `plan_metal_complex_conductive_fused_pair` and
  `plan_metal_complex_no_pml_offdiag_fused_electric_pair` require a keyword
  `chunk_bounds`; the arms hand over the composer's, and `chunk_bounds.plan_words` refuses
  a missing record, or one whose presence word disagrees with the shape, before anything
  compiles.
- **The complex conductive D->E pair lifts the certified curl.**
  `complex_conductive_fused_pair.py` cuts its curl half out of the certified emitter's
  output between two anchors instead of re-spelling it; the record rides last in its
  `Params` (11 scalar words, then the 25-word record), so it costs no binding.
- **Ruling R7.** The complex no-absorber off-diagonal row was float32 where the array
  path's is a double row; this change refused it to the array path by name, and the
  software binary64 row that replaces it is the section "The Metal folded off-diagonal
  dispersive and complex no-absorber off-diagonal rows step MEEP's double row in software
  binary64" of this changelog. The stencil weld that lifts it
  (`complex_no_pml_offdiag_fused_electric_pair.py`) has its lifted curl read the record,
  which rides last in its `Params` (itemsize 96 -> 192) and is passed to `step_cell` and
  every `d_final_c`.
- **Unchanged: `complex_no_pml_stored_e.py`.** Its `E = ((D - P0) - P1 ...) * chi1inv` is
  already the array path's word (its gate's zero-pole row: 0 differing words); the rows
  with poles are refused on the ADE half until the polarization update's four constants
  reach the Metal ADE kernel.
- **The gates.** `gate_metal_complex_no_pml_curl.py` and
  `gate_metal_complex_no_pml_conductive.py` share their cases, records, planted states and
  mutations through the new `complex_no_pml_gate_common.py`: 3-D, 2-D and 1-D grids,
  a 3-D grid storing one cell on a live axis, phased, Brillouin-edge and metallic axes, a
  planted-zero leg, the grouping mutation inverted to plant the package's previous pairwise
  bracket, the absent-partner mutations each with a declared-null twin, the plane-0
  connection mutations, and a record that marks every axis present.
  `gate_metal_complex_conductive_fused_pair.py` re-cuts its curl and connection mutations
  to the lifted text. `gate_metal_offdiag_stencil_welds.py` builds the complex no-PML weld
  with the oracle's record. The host half is
  `parity/meep_gpu/test_gate_metal_complex_no_pml_host.py`.
- **`tools/ci/pending_certification.txt`**: 24 ids of these families' tests leave it: 23
  that now pass, and the branch-census test the lifted curl replaced by name.

### The Metal dispersive families step MEEP's polarization arithmetic and select per chunk

The Metal dispersive (Lorentz/Drude) families -- `metal_kernels/ade_update_p.py`,
`dispersive_update_e.py`, `folded_dispersive_update_e.py`,
`complex_dispersive_update_e.py`, `folded_offdiag_dispersive_update_e.py`, and the ADE
halves of `fused_ade_chain.py` and `complex_fused_ade_chain.py` -- now transcribe the
array path's dispersive package:

- **The recurrence is MEEP's.** `update_P` steps
  `gamma1inv * (((two_minus_od * P) - (gamma1 * P_prev)) + (omega0dtsqr * (sigma * W)))`
  on MEEP's four float32 constants (`PolarizationState.constants`,
  susceptibility.cpp:251-256), every operation one float32 operation; complex storage
  takes the real update on each part (MEEP's two real passes) with no complex helper.
  The folded three-coefficient form and `PolarizationState._coefficients` are gone from
  these kernels, and the fused chains' `Params` carry the four constants.
- **The drive is selected per chunk.** Under an absorber `ade_update_p` binds the
  stored E, `f_w` and the chunk record and reads MEEP's `w = f_w ? f_w : f` on the
  component's own half-integer membership line, the line the base constitutive kernel
  selects on; it no longer asks `Fields.drive_field` for a pointer, so a component split
  between the two forms is served on the device instead of refused.
- **The dispersive `update_E` singles step the per-chunk tail.** They bind the record
  last (buffer 18) and, on a cell whose chunk carries the component's own layer, read
  `prev` before writing `f_w` and form `E + ((kps*src) - (kms*prev))`; elsewhere
  `E = src` and `f_w` is untouched.
- **The folded off-diagonal dispersive product gets a `Params`** (its four scalars and the
  chunk record, 28 of 31 bindings) and the per-chunk tail. Its row was float32 and was
  refused under ruling R7; the software binary64 row that replaces it is the section "The
  Metal folded off-diagonal dispersive and complex no-absorber off-diagonal rows step
  MEEP's double row in software binary64" of this changelog.
- **The complex fused chain names no complex arm.** Its ADE half calls no complex
  helper, so the clause that compared the probe's arm with the ADE half's and the
  `ade_expansion_arm` parse are deleted.
- **Every plan builder takes a required keyword `chunk_bounds`**, with no default (the
  ADE plan binds it only under an absorber).
- **Gates.** The ADE, dispersive E (plain, folded, complex), folded off-diagonal
  dispersive and fused-chain gates plant the package's previous forms as must-catch
  mutations (the folded three-term recurrence, the previous accumulation grouping) and
  carry the per-chunk battery and record rows; the ADE gate removes the drive select both
  ways and keys it on the wrong lattice. The folded off-diagonal dispersive gate runs its
  tail on exact-row states.
- **Certification.** The kernel-source digests of these modules move;
  `fingerprints.json` and every ledger are unchanged, pending the certification round.

### The Metal table carries MEEP's double-precision steps in software binary64, and the held source deposit forms its difference on the device

Owner ruling R7 (2026-10-08) chose software binary64 over refusing the Metal off-diagonal
and nonlinear rows to the array path. This change lands the shared library and its first
consumer, the source deposit's door; the off-diagonal and nonlinear rows wire it in their
own changes.

- **`metal_kernels/softfp64.py`**: an MSL fragment (`MSL_HELPERS`, include-guarded) of
  integer-only helpers on binary64 bit patterns -- `sd_widen` / `sd_widen_f`,
  `sd_scale_pow2` (MEEP's `0.25` and `0.0625`), `sd_add`, `sd_sub`, `sd_narrow` /
  `sd_narrow_f`, `sd_subtract_from_word` and `sd_daz` -- each one IEEE operation rounded to
  nearest even, with a `fz` argument for the subnormal policy (`true` on Metal, which
  flushes only: an input subnormal reads as a signed zero and a result below the smallest
  normal is a signed zero, decided before rounding, as the arm64 host does). Subnormals,
  infinities and NaNs follow the host's float64 unit. `library_source` assembles a source
  with the shipped contraction directive. A NumPy transcription of every helper checks the
  arithmetic on a host with no GPU.
- **The held deposit (step M11).** `device.Residency.subtract_double_index`, the door
  `host_writes.subtract_double` already looked for, forms `float(double(f) - real(A))`
  on the device for a held mirror (`door_scatter_sub`, now the software binary64
  statement, complex64 per part), in the host FPU's mode measured at the call; a
  host-current mirror is formed on the host by the array path's statement and written to
  both sides. The held route no longer gathers the deposit cells to the host and scatters
  them back: one device-to-host read per deposit per step is gone, and a new counter,
  `Residency.double_subtracts`, counts the device deposits. The float32 subtract door it
  replaces (`Residency.subtract_index`, unused since the deposit moved to double) is
  removed.
- **Measured** on one Apple M1 Max against NumPy float64 on the same host in both FPU
  modes: 0 differing words over 4,000,000 triples per class and mode (the off-diagonal row
  and the `Dsqr` form on seven classes, the single operations on three); the transcription
  equals the MSL word for word; 16 of 16 planted helper defects are caught. The door
  equals the array path's statement on every word of 1,048,576-cell real and complex64
  held mirrors (three operand classes, both modes), with no gather and no copy out. On the
  held route, the seven source cases of the MEEP-identity work (2-D and 3-D, point, area,
  line, continuous, integrated and PML-crossing sources, 40 steps, every stored component
  at six checkpoints) store 0 differing words against the array path, 9,063,072 words in
  all, and each step's sparse gathers fall by exactly its deposits (one to four a step).
- **Tests**: `meep_gpu/test_metal_softfp64.py` (new); the two held-deposit tests of
  `test_metal_residency_ownership.py` now count a door launch and no gather.
- **Docs**: [floating point](docs/design/floating-point.md#double-precision-on-the-metal-table).

### The bit-parity coverage board is re-cut on series entries 01 to 52

`parity/meep_gpu/build_parity_board.py` cut the board in `parity/meep_gpu/parity_board/`
again, on base + series 01..52; the first board, below, was cut on 01..31. Only its three
generated files change. Its test outcomes are the CUDA and Triton test files' cases of the
package suite (CPU-only shim, every marker, dispatch off) and the Metal test files' cases
of the device run (Apple GPU), both run on this tree: 205 test files, 11,364 cases, 597
failures, every one declared in `tools/ci/pending_certification.txt`.

- **The kernel tables** (N / D, refused rows in brackets; the first board, then this one):
  (a) the chunk-record select, CUDA 6 / 47 -> 11 / 48 (+8), Triton 6 / 55 -> 12 / 55
  (+3), Metal 2 / 43 -> 34 / 45 (+19); (b) MEEP's A form, 5 / 16 -> 7 / 14 (+51),
  6 / 55 -> 15 / 55 (+13), 2 / 25 -> 13 / 22 (+55); (c) host word identity,
  8 / 62 -> 14 / 61 (+4), 10 / 67 -> 16 / 67 (+1), 9 / 64 -> 29 / 55 (+22).
- **(d) and (e) stay 0 under every baseline** (CUDA 0 / 53, Triton 0 / 55, Metal 0 / 74):
  every byte-binding ledger entry (55 / 67 / 63) records the array path `real-v1`, and this
  tree steps `real-v6`, until the certification round re-gates.
- **Against MEEP, unchanged**: 2 / 2 measured families identical under keep (the real PML
  forms and mirror symmetry), the other 8 refused (no minted case exercises them), flush
  refused (no minted set).

### The Metal beta and BFAST kernels run MEEP's separate passes after the curl

The Metal special-kz beta and BFAST kernels -- `metal_kernels/special_kz.py`,
`bfast_curl.py` and `folded_beta.py`, and every product that lifts their bodies
(`beta_fused_magnetic_pair`, `beta_fused_electric_pair`, `beta_complex_fused_magnetic_pair`,
`beta_complex_fused_electric_pair`, `beta_real_fused_hd_pair`, `beta_complex_fused_hd_pair`,
`bfast_fused_magnetic_pair`, `bfast_fused_electric_pair`, `bfast_fused_hd_pair`,
`folded_beta_real_fused_pair`, `folded_beta_real_fused_magnetic_pair`,
`folded_beta_complex_fused_pair` and `folded_beta_complex_fused_magnetic_pair`) -- now
step the array path's order: the curl's per-chunk forms, then MEEP's BFAST pass, then
the beta pass, each through its own additive ladder (`stepping._apply_folded_increment`):

- **One Metal spelling of the ladder** (`metal_kernels/folded_pass.py`). A pass runs in
  register after the certified curl's per-chunk forms and before its stores: the
  increment, negated and masked to `+0` by the same ownership rows the curl carries, is
  scaled by the dsig coefficient, subtracted from the field outside the component's
  dsigu chunks and, inside them, from `fu` and (scaled by the dsigu coefficient) from
  the field. The certified store lines are kept byte for byte.
- **Every source is derived from a certified curl.** The beta and BFAST curls are the
  certified plain, complex, folded and folded-complex curls with the pass inserted
  (`folded_pass.with_folded_pass`); the per-family templates, the in-template beta
  insert and the BFAST tail are gone. The same-cell and H->D products lift those bodies;
  the four folded beta pairs are the certified fold-carry welds' own output with the pass
  carried in (`folded_pass.insert_split_pass`), where the displacement is formed after
  the `fu` store from the curl step's word, so the pass's `fu` word is a register of its
  own.
- **The beta pass.** The increment is `-(w * g)` with the real word
  `float32(sign * 2*pi*beta*dt)` (`special_kz.beta_curl_coefficients`); on complex
  storage it is the negated quarter turn of the partner's parts by that real word, the D
  side's word negated, as `part_products.quarter_turn` forms it. No complex product is
  formed and no complex coefficient pair is bound: the beta words are two floats in
  every signature and `Params` record.
- **The BFAST pass.** The state takes MEEP's two roundings (`F = S - F_prev` stored on
  owned cells, then `raw = F - F_prev`), an absent partner is not read, and in MEEP's
  plain one-partner branch the state is `k1*(g1s + g1)` (`stepping._bfast_term`). Clause
  11a (a k on an invariant axis) stays a declared over-refusal: the Metal and Triton
  clause lists are pinned equal, and its premise is gone (the array path's NULL partner
  moves no word).
- **Every plan binds the chunk record, with no default.** The singles bind it last; the
  pairs embed it in `Params` (the folded beta pairs extend their parents' records with
  the two beta words in front of it); the arms pass the composer's. The products that
  take `update_H` refuse the run's first `update_H` on `launch.h_start_reasons`' rule.
- **Gates.** `gate_metal_special_kz.py`, `gate_metal_bfast.py` and
  `gate_metal_folded_beta.py` compare the kernels with `stepping` as the in-process
  oracle (the in-gate transcriptions are gone) and arm the pass's own mutations (folded
  into the curl, unmasked, the dsigu branch dropped, one-rounding state, sign) with the
  KERNELS.md §7.2 rows. The product gates re-cut their needles onto the pass;
  `gate_metal_bfast_fused_hd_pair.py`'s and
  `gate_metal_folded_beta_complex_fused_magnetic_pair.py`'s grouping mutations are
  inverted (KERNELS.md §7.1: each plants the package's previous grouping and stays
  must-catch), the BFAST H->D gate arms the §7.2 battery, and the residue gate's
  movement floor exempts the auxiliaries the per-chunk forms leave untouched, as the
  tranche-7 gate's does. A beta run is 2-D, so under the per-chunk forms no `fu` word of
  target 0 (dsigu z) is written: the beta products' gates arm their mask, auxiliary and
  grouping rows on targets 1 and 2, the beta H->D gates declare the cell-0 mask a null
  on a 2-D fold, and the folded beta complex B/H gate declares the moved ghost
  coefficient a null under the one-rank record (MEEP's division reaches it; the parent
  weld's gate catches it there).
- **Not in this change:** the nonlinear `|D|^2` row on Metal waits on the owner's ruling
  (R7). Ledgers and fingerprints are unchanged, pending the certification round.

### The Metal off-diagonal families select their PML tail per chunk

The off-diagonal (tensor epsilon) `update_E` families on Metal --
`metal_kernels/offdiag_update_e.py`, `folded_offdiag_update_e.py`,
`complex_folded_offdiag_update_e.py` and the three PML scratch welds that lift them,
`offdiag_fused_electric_pair`, `folded_offdiag_fused_electric_pair` and
`folded_complex_offdiag_fused_electric_pair` (machinery in `offdiag_weld_common`):

- **Ruling R7.** The row these kernels formed was float32 where the array path's is a
  double row; this change refused it to the array path by name, and the software
  binary64 row that replaces it is the section "The Metal off-diagonal update_E
  families step MEEP's double row in software binary64" of this changelog.
- **The PML tail is the per-chunk select.** The singles bind the chunk record last
  (buffer 28 on the real singles, 29 on the complex folded one) and step the base
  constitutive kernel's tail: on a cell whose chunk carries the component's own layer
  `E = E + ((kps*src) - (kms*prev))` with `prev` read before `f_w` is written;
  elsewhere `E = src` and `f_w` is untouched. The previous
  `((E + kps*src) - kms*prev)`, written on every cell, is gone.
- **The welds lift the per-chunk curl.** The record rides in each weld's `Params`
  (`ChunkBounds cb`) and `step_cell` takes it, so a re-derived foreign cell selects
  its form where it is. The curl's guarded auxiliary store moves to the register
  (`n0 = du0 ? n0 : u0[ii]`) because the rotating `fu` scratch must carry the
  pre-launch word off the recurrence's cells. The folded complex weld's parity is the
  per-part product `c_mul_coefficient_left(+/-1, z)`, the array path's
  `part_products.scale`.
- **Every plan builder takes a required keyword `chunk_bounds`**, with no default.
- **Gates.** The per-chunk tail runs against `stepping.py` on exact-row states (dyadic
  words) with the B battery armed and the accumulation-grouping mutation inverted
  (`m10_previous_accumulation_grouping`).
- **Certification.** The kernel-source digests of these modules move;
  `fingerprints.json` and every ledger are unchanged, pending the certification round.

### The hand-written CUDA dispersive kernels step MEEP's recurrence and per-chunk forms

The dispersive families of the hand-written CUDA table now compute what the array path
computes (`dispersion.PolarizationState.update`, `stepping._apply_constitutive_regions`,
`Fields.drive_field`): `update_P_pml_real` and its uniform-sigma twin (`ade_kernels.py`),
`update_E_pml_real_dispersive` (`dispersive_kernels.py`), the E->P pair
(`fused_polarization_pair.py`), the three-slot welds (`three_slot_dispersive_weld.py`,
`no_pml_three_slot_dispersive_weld.py`), the dispersive D->E pairs
(`dispersive_fused_electric_pair.py`, `no_pml_dispersive_fused_electric_pair.py`), the
off-diagonal dispersive single and E->P weld (`dispersive_offdiag_update_e.py`,
`dispersive_offdiag_fused_polarization_pair.py`), and the complex no-absorber
`update_P` with its E->P pair and three-slot weld.

- **The recurrence.** `p_out = gamma1inv * (((p * two_minus_od) - (gamma1 * q)) +
  (omega0dtsqr * (s * w)))` on the four float32 words of
  `Susceptibility.realnum_constants`, passed as scalars, where the kernels took the
  three folded coefficients (`c_now`, `c_prev`, `c_drive`). The complex kernels take it
  one part at a time (`ade_part`), with no zero-imaginary complex product.
- **The drive.** MEEP's `w = f_w ? f_w : f` per chunk: the single binds `f_w_<c>` and the
  stored `E<c>` and selects per cell from the chunk record (`own_axis` names the
  component's half-integer entry); without a layer `ade_kernels.drive_pair` binds the
  stored E for both. The E->P pair keeps its register hand-off, which is the drive on
  both forms, and the three-slot weld selects in the kernel.
- **The constitutive half.** The dispersive `update_E`, both D->E pairs' E halves and
  the off-diagonal dispersive tail call the root kernels' flagged `constitutive_apply`:
  MEEP's accumulation on the cells whose chunk carries the component's own layer,
  `f = src` elsewhere, `f_w` read and written only there. The off-diagonal dispersive
  single takes the off-diagonal family's double row and presence words.
- **The launchers and plans.** Every launcher of these families takes the chunk record
  (`chunk_bounds`); the fused plans bind the composition's one upload. The off-diagonal
  E->P plan now hands its product the layer, which it refused without, so that seam's
  plan no longer skips its two sub-steps. `coverage.covers_real_pml_ade_*` checks the
  drive pair's two arrays and that each constant is a float32 word.
- **Gates.** `gate_cuda_ade.py`, `gate_cuda_dispersive.py` (its constitutive regroup
  mutation now plants the grouping used until 1.0, beside the select removed, a record
  bound off by one, the integer record and the plain form as a unit-coefficient
  accumulation), `gate_cuda_fused_polarization_pair.py`, `gate_cuda_three_slot_weld.py`,
  both dispersive D->E pair gates, both off-diagonal dispersive gates, the three complex
  no-absorber dispersive gates and the two dispersive probes bind the record and take
  the new recurrence. The NumPy legs: `gate_cuda_dispersive.py --product full` 414 of
  414 single and 414 of 414 multi-step cases identical, every mutation as required;
  `gate_cuda_ade.py --product reduced` 32 of 32 and 32 of 32, 25 of 25 mutations.
- **Records.** `certification.json` still describes the previous texts; the dispersive
  record tests stay listed in `tools/ci/pending_certification.txt` until the gates are
  re-run on these bytes.

### The array-path oracle moves to `real-v6` and the MEEP-identity Tier A set is re-minted on it

The cylindrical radial sum, i*m/r pass and axis products below move the array-path oracle
(1 of its 20 fixture cases, `cylindrical_m0`), so its fixture set is now
`array_path_oracle/real-v6` (one bump for the changes below; the CuPy test, the parity board
and the Metal complex conductive curl below do not move it) and the Tier A keep set is
minted again on this tree by `parity/meep_gpu/record_meep_identity.py`.

- **Measured** against MEEP 1.33.0 single precision built without contraction, keeping
  subnormals, on Apple silicon: 4 of 4 cases, 38,040,036 compared stored words and
  61,772,940 reader words equal to MEEP's, every digest reproduced by the replay with MEEP
  never imported; 1,359,862 of the 5,000,000 budget bytes. The four capsules (their
  provenance records the changed `stepping.py` and `driver.py`) and `meep_identity.json`
  change; the digests and currents tables come out byte-identical. The ledger records
  `real-v6`, fingerprint `1ab51627` (it recorded `real-v5`, `307b5575`).
- **The kernel ledgers are not written.** `record_array_path_oracle.py` in report mode
  reaches all 259 declared form classes over 20 cases and would change the CUDA, Triton
  and Metal ledgers' `array_path_oracle` leaves in 452, 476 and 468 leaves and nowhere
  else, with every entry's bound digest and live capabilities unchanged. Writing them is
  a certification event, so their three oracle tests stay listed.
- **`tools/ci/pending_certification.txt`**: the Tier A ledger's oracle test, which the
  cylindrical change below listed, leaves it.

### The Metal complex conductive PML curl steps MEEP's conductive forms per structure chunk

`metal_kernels/complex_conductive_pml.py`, the conductive PML curl on complex storage,
now transcribes the array path's per-chunk conductive forms
(`stepping._apply_conductive_pml_update`) part by part, as the real-field curl of
`metal_kernels/conductive_pml.py` does:

- **MEEP's grouping.** On the both-layer and dsig-only forms the `f_cond` increment
  `(fcnd' - fcnd)` is formed first and then added (step_generic.cpp:167, :209). Before
  this change the kernel added `fcnd'` and subtracted `fcnd` in two operations. The
  dsigu-only and neither forms, the lossless tails and the lifted stencil are unchanged.
- **Per-chunk conductivity presence is refused by name.** A curl sub-step whose lossy
  target is not kept by every chunk (`ChunkLayout.conductive_chunks`) is refused to the
  array path (`conductive_pml.conductive_presence_reasons`, read on the composer's
  `chunk_layout`). The family's H/E companion also carries the D side's presence, so a
  configuration whose `step_D` the family refuses gets none of its H/E arms. The plan
  builders take `chunk_layout`, and the arms hand over the composer's.
- **Gate.** `gate_metal_complex_conductive_pml.py` records a form census per product,
  inverts its grouping mutations (`curl_previous_grouping` and
  `increment_previous_grouping` plant the package's previous groupings), carries the
  per-chunk battery (`b1`..`b8`, `b7` a declared null) and the two record mutations, and
  holds a partial-presence refusal row on each side.
  `parity/meep_gpu/test_gate_metal_complex_conductive_pml_host.py` checks its census,
  needles and refusal fixtures on the host.
- **Certification.** The module's kernel-source digests move; `fingerprints.json` and
  every ledger are unchanged, pending the certification round.

### A bit-parity coverage board

`parity/meep_gpu/build_parity_board.py` cuts the bit-parity counterpart of the fusion
boards: for each kernel table, how many modules of the kernel design's step lists carry
(a) the chunk-record select, (b) MEEP's operation order in the curl and the PML
constitutive accumulation, (c) word identity with the array path on the host, (d) word
identity on the device, and (e) a certification on the live array-path oracle; and for
each array-path family, word identity with MEEP under each subnormal baseline (owner
ruling R13: keep and flush on CUDA, Triton and the array path, flush only on Metal).

- **Every column is read off the tree.** (a) and (b) from each module's own text, parsed
  into expression trees (the 0.9.2 grouping and MEEP's are told apart structurally,
  complex helpers read as their operators); (c) from a recorded run of the module's own
  test files and the pending list; (d) and (e) from the three ledgers and a fresh run of
  the array-path oracle; the MEEP rows from the Tier A/C record, each case's families
  read off its capsule's construction log. The rows are the step lists, held as data in
  `parity_board_rows.json`. A row or column the board cannot measure is refused by name
  and counted beside the denominator, never as a gap or a pass.
- **It refuses a stale subject.** `outcomes` refuses a run during which the tree moved;
  `build` refuses an outcomes record whose subject manifest is not the tree's ("the
  outcomes subject and the working tree disagree"), as the CUDA fusion board refuses a
  census; `check` refuses a board of another tree.
- **The first board** (`parity/meep_gpu/parity_board/`), on base + series 01..31: (a)
  CUDA 6 / 47, Triton 6 / 55, Metal 2 / 43; (b) 5 / 16, 6 / 55, 2 / 25; (c) 8 / 62,
  10 / 67, 9 / 64; (d) and (e) 0 under every baseline (CUDA 0 / 53, Triton 0 / 55,
  Metal 0 / 74): every byte-binding ledger entry (55 / 67 / 63) records the array path
  `real-v1` and this tree steps `real-v4`. Against MEEP: 2 / 2 measured families
  identical under keep (the real PML forms on 4 / 4 cases, 0 / 38,040,036 stored and
  0 / 61,772,940 reader words differing; mirror symmetry on 2 / 2), the other 8 refused
  (no minted case exercises them), and flush refused (no minted set). The run of 201
  test files behind (c) had 1,107 failures of 10,518 tests run, all declared.

### The CuPy array path is measured against the NumPy array path after every sub-step

The NVIDIA kernel tables are certified against the CuPy array path on their own device,
while the array-path oracle and MEEP identity are measured on the NumPy array path. The
link between the two is now measured instead of assumed, and pinned on CUDA hosts by a new
test, `meep_gpu/test_cupy_array_path.py`: two drivers hold one seeded state, one on NumPy
and one on CuPy with dispatch off; each takes the real `FdtdDriver.step`, and after every
sub-step the driver runs, every array the capsule pins is compared as 32-bit words, the
sign of zero included. The test runs 20 cases under the keep policy and skips by the
declared resource `cupy` elsewhere. No array-path code changes.

- **Measured** on one RTX A6000 (x86-64 Linux, CuPy 13.5.1, NumPy 2.2.6), under keep and
  under flush (the host through MEEP's own switch, CuPy native), with dispatch off
  (every CuPy driver's freeze refused, no kernel slot):
  - 59 lifted cases (complex 12, conductive 6, dispersive 3, forced chunk divisions 4,
    two of them not separable, nonlinear 5, off-diagonal 8, sources 17, the four Tier A
    capsules), 100 steps each, 97,603 sub-step snapshots: 0 of 25,639,152,898 words
    differ, under each policy. Under keep, 27 of the 59 runs carry subnormal words, the
    first of them at step 16 to 100;
  - 19 array-path-oracle fixtures and 11 complex-storage hand-built states, 12 steps
    each: 0 of 45,903,888 and 0 of 40,400,760 words, under each policy;
  - 10 states near the bottom of the float32 range, where most operations meet a
    subnormal: 0 of 25,089,876 words under keep and under flush. The control with the
    host keeping and CuPy flushing differs in 2,443,455 of 6,272,469 words, in 10 of 10
    cases (and in 254 of 11,475,972 and 363 of 10,100,190 words on the normal-range
    states, 4 of 19 and 2 of 11 cases);
  - a fresh lift with `prefer_gpu=True` stores the words of a fresh NumPy lift: 15 cases,
    0 of 6,266,843 words, under each policy;
  - the harness hand references that take an array module store, on CuPy, the words they
    store on NumPy: 0 of 29,290,752 (PML curl, fused magnetic pair, fused electric
    pair), 0 of 1,355,376 (no-PML curl), 0 of 1,762,236 (fold), 0 of 3,469,140 (folded
    complex), under each policy. The folded off-diagonal reference takes no array
    module and is NumPy only.
- **The cylindrical rows agree too**, with the cylindrical change below in the tree: its
  per-chunk radial sum is a `cumsum` per chunk segment on NumPy and a row loop on every
  other array module, so CuPy's parallel scan is not used. Four lifted cylindrical
  cases, 100 steps each, and the cylindrical oracle fixture, 12 steps: 0 of 125,291,400
  and 0 of 1,073,340 words, under each policy, and the test's two cylindrical rows pass.
  Before that change the prefix was `xp.cumsum` along r, a sequential accumulation on
  NumPy and a parallel scan on CuPy, and the same cases differed in 44,821,841 and
  227,299 words from the first `step_B` on (0 of 126,364,740 with CuPy's cumsum
  replaced by NumPy's on the host).

### Cylindrical runs step MEEP's per-chunk radial sum, its i*m/r pass and its axis products

A cylindrical (Dcyl) run takes the three forms MEEP writes for it (step_db.cpp:86-434), in
`stepping.step_B` and `stepping.step_D`:

- **The radial derivative of Bz and Dz is the difference of MEEP's `f_rderiv_int`, summed in
  every structure chunk** from that chunk's first row (`F[0] = 0`, then `F[ir] = F[ir-1] +
  rinv*(f_p[ir]*(ir+ir0) - f_p[ir-1]*((ir-1)+ir0))` in float32, left to right). A cell reads
  the sum of the chunk that owns it, so where an r layer's chunk begins the sum restarts at
  `+0` (`stepping.cylindrical_rderiv_difference`; the chunk starts come from the run's
  division, `ChunkLayout.boundaries`, or the owner volume where the r cuts differ along z).
  `rinv = 1.0/((ir+ir0) - 0.5)` is formed in double, stored as float32 and MULTIPLIED; the
  engine divided by `(ir+ir0) - 0.5` and summed once over the whole r extent.
  `cylindrical_rderiv_prefix` and `_cylindrical_rderiv_weights` are replaced by
  `cylindrical_rderiv_difference` and `_cylindrical_rderiv_rows` (weights and `rinv`), and
  the pooled buffers by `cyl_radial` and one `cyl_prefix_<row>` per chunk start. NumPy's
  `cumsum` along r is the sequential row loop (pinned); any other array module takes the row
  loop.
- **The i*m/r coupling is a pass of its own after every component's curl** (step_db.cpp:
  177-294), through the folded increments' ladder (`stepping._apply_folded_increment`), where
  it was folded into the curl. Its row is MEEP's `rinv = the_m / r`: `the_m`, an integer
  times the double `Courant` rounded once to float32, divided in float32 by the target's
  doubled r coordinate (`stepping._cylindrical_imr_increment`, which replaces
  `_cylindrical_imr_term`); each part adds `rinv` times the partner's other part, a quarter
  turn. An R target (dsig phi, never a layer) takes `f += raw` or, in a z-layer chunk,
  `fu += raw; f += siginvu*raw`; a Z target takes `f += raw`, or `f += raw*siginv` in an
  r-layer chunk, and never writes `fu`.
- **The axis rules' products with Courant are double**: m = 0, `Dz += float(Hp*(4*Courant))`;
  |m| = 1, `(sd*Courant)*(bracket)` with the bracket in float32 left to right
  (`stepping._realnum_double_product`). A float32 factor is the same word only where Courant
  is a float32 word: the default 0.5 is, 0.3 is not.
- **Stored words change** in every cylindrical run: m = 0 and |m| = 1 through the prefix, and
  every m != 0 run through the i*m/r pass. MEASURED against MEEP 1.33.0 single precision
  built without contraction, on Apple silicon, keeping subnormals, at every step 0 to 300
  and the checkpoints to 2,000 (311 steps per case): the cylindrical family's four cases of
  phase 0 (m = 0 real and m = 1 complex, each without and with PML(0.5) on the r-high and
  both z faces, MEEP dividing the layered cell into 6 chunks with the r cut at row 14) and
  two at Courant 0.3 (`m0_c3`, `m1_pml_c3`) store MEEP's word in every one of 35,827,200
  compared words (5,971,200 per case), signed zeros included; a planted one-ulp flip is
  caught on every leg. With the former forms the same legs read m0 1,234,454, m1
  5,020,274, m1_pml 5,080,404, m0_pml 1,248,231, m0_c3 1,139,898 and m1_pml_c3 4,959,523
  differing values. Each form is necessary: one sum over the whole r extent leaves m1_pml
  4,788,174 and m0_pml 1,183,127 (first at step 12); dividing by `(ir+ir0) - 0.5` leaves
  1,227,110 to 4,987,751 on all four; the i*m/r term folded into the curl leaves m1
  4,990,324 and m1_pml 5,017,962; a double-rounded `2m*Courant/r` row moves nothing at
  Courant 0.5 (m1_pml: 0) and leaves m1_pml_c3 4,645,111; float32 axis products leave
  m0_c3 1,108,330 and m1_pml_c3 87,112 (at 0.5 they are the same word by exactness, not
  measured). The same six cases on Linux (MEEP 1.33.0 single precision, x86-64,
  keeping subnormals) read the same: 0 of 35,827,200, and the former forms the same counts.
- **The readers over these words**: `GpuRunResult.get_array`, the stored centres and
  `get_field_point` against `sim.get_array` and `sim.get_field_point` on the same six cases
  at ten steps from 1 to 2,000, all twelve components: 0 differing words of 864,000 whole-array,
  576,000 centred and 56,160 point words.
- **No complex product.** The complex-product probe judges every step of the three complex
  cases with 0 violations (its planted product is caught on 3 of 3); it sees the per-chunk
  sums as `add.accumulate` on complex storage.
- **Tests.** `test_cylindrical.py` pins the radial difference against MEEP's per-chunk loop
  word for word (forward and backward, one to three chunks, real and complex, -0 and
  subnormal inputs, pooled and allocating), that multiplying by `rinv` is not dividing,
  that the sum restarts at a cut, NumPy's sequential `cumsum`, the chunk starts of a
  layered grid, the i*m/r row as MEEP's realnum quotient (at Courant 0.3, where a double
  quotient is another word), the pass's two ladders (an R target's `fu` stage and a Z
  target's `siginv`, an unowned `-0` kept), and the axis rules' double products at Courant
  0.3. `test_step_scratch_is_bit_identical` adds a layered m = 1 case. In
  `parity/meep_gpu/mutate_from_meep.py` two anchors move to the new spelling (the
  conjugated i*m/r and the halved axis limit) and two mutations are added (one radial sum
  for the whole extent; division instead of `rinv`); 5 of 5 cylindrical rows in
  `stepping.py` are caught and their selectors pass unmutated.
- **The array-path oracle** moves on its `cylindrical_m0` case (1 of 20 case digests at the
  same fixture-set identity, `array_path_oracle/real-v5`; fingerprint `307b5575` ->
  `0537c1dd`, the same on Linux and on the Mac) and declares the prefix's chunk restart as a
  form class of that case (259 classes, all reached). The fixture set moves once, at the
  batch's Tier A re-mint.
- **Certification.** The three kernel tables still transcribe the former forms; 60 of their
  tests, and the Tier A ledger's oracle test, are listed in
  `tools/ci/pending_certification.txt` for the cylindrical lanes (KC8, KT8, KT11, KM10) and
  the re-mint. With the package defaults a cylindrical lift no longer dispatches the former
  kernels: on an Apple GPU the Metal cylindrical builders raise at plan build and on an
  NVIDIA GPU the Triton pair's warm pass does, each refused by name per slot, so the run
  takes the array path (MEASURED on an M1 Max and an RTX A6000; before this change the
  layered m = 0 case dispatched the Metal and Triton cylindrical kernels).

### The array-path oracle moves to `real-v5` and the MEEP-identity Tier A set is re-minted on it

The dispersive polarization update below moves the array-path oracle (2 of its 20 fixture
cases, `dispersive_3d` and `dispersive_pml_2d`), so its fixture set is now
`array_path_oracle/real-v5` (one bump for the changes below; the CUDA, Triton and Metal
kernel changes below do not move it) and the Tier A keep set is minted again on this tree by
`parity/meep_gpu/record_meep_identity.py`.

- **Measured** against MEEP 1.33.0 single precision built without contraction, keeping
  subnormals, on Apple silicon: 4 of 4 cases, 38,040,036 compared stored words and
  61,772,940 reader words equal to MEEP's, every digest reproduced by the replay with MEEP
  never imported; 1,359,847 of the 5,000,000 budget bytes. The four capsules (their
  provenance records the changed `dispersion.py`) and `meep_identity.json` change; the
  digests and currents tables come out byte-identical. The ledger records `real-v5`,
  fingerprint `307b5575` (it recorded `real-v4`, `f61fe1df`).
- **The kernel ledgers are not written.** `record_array_path_oracle.py` in report mode
  reaches all 257 declared form classes over 20 cases and would change the CUDA, Triton
  and Metal ledgers' `array_path_oracle` leaves in 448, 472 and 464 leaves and nowhere
  else, with every entry's bound digest and live capabilities unchanged. Writing them is
  a certification event, so their three oracle tests stay listed.
- **`tools/ci/pending_certification.txt`**: the Tier A ledger's oracle test, which the
  dispersive change below listed, leaves it.

### The Triton no-absorber curls take MEEP's stencil per partner presence

The three Triton kernels whose curl is the no-absorber one are one transcription:
`no_pml.plain_curl_step`, `no_pml_conductive.conductive_plain_curl_step` and the curl
half of `no_pml_fused_electric_pair.no_pml_fused_curl_constitutive_D`. They formed the
curl in the 0.9.2 grouping, `dtdx * ((g1s - g1) + (g2 - g2s))`, with every partner taken
as present, which the array path no longer does.

- **The curl is `forms.curl_stencil`**, the root kernels' helper: C's left-to-right
  `dtdx * (((g1s - g1) + g2) - g2s)` with both partners, `dtdx * (g1s - g1)` or
  `(-dtdx) * (g2s - g2)` where the other partner's derivative axis is invariant, and an
  exact `+0` with neither (`stepping._curl_from_operands`). The tails are unchanged: the
  plain `f - curl` and the conductive `((f * condfac) - curl) * condinv` already are the
  array path's.
- **Which partners exist is read from the chunk record.** Each kernel takes `cb` directly
  after its last pointer argument and reads its partner-presence word alone (a run with
  no absorber has no membership run). The presence is the run's (`Grid.is_invariant`),
  not the stored extent's: a 3-D run that stores one cell on an axis still takes both
  partners there.
- **The plans bind the record.** `PlainCurlPlan`, `ConductivePlainCurlPlan` and
  `NoPmlFusedElectricPairPlan` require `chunk_bounds` (`launch.bind_chunk_bounds`); the
  `*_from_arrays` builders have no default, and the engine builders take
  `chunk_layout` and bind `launch.chunk_bounds_for`. Every division of a no-absorber run
  binds the same words as its one-rank record, which is what the certified-product
  composer binds for the fused pair.
- **The gates.** `gate_triton_no_pml.py` takes MEEP's grouping with each case's presence
  as its reference (`no_pml_ref.reference_plain_step` gains `presence`), sweeps a 3-D run
  with a one-cell resolved axis beside the invariant-axis shapes, adds a planted
  signed-zero leg, and replaces its flattening mutation with `regroup_stencil` (the
  0.9.2 grouping), `full_bracket_over_an_absent_partner` and
  `presence_from_the_extent`, each scored where it can reach a word. The conductive gate
  adds a 2-D and a one-cell 3-D row and a cleared-presence mutation; the fused pair's
  probe inverts its curl mutation and adds a presence one.
- **Owed to certification.** `no_pml.KERNEL_SOURCE_SHA256`, its `GATE_VERDICT` and the
  `fingerprints.json` records of the three modules pin 0.9.2 bytes; they are re-cut when
  the gates run on these bytes. The tests that read them are listed in
  `tools/ci/pending_certification.txt`.

### The Triton off-diagonal kernels form MEEP's row and step each chunk's own form

The off-diagonal `update_E` kernels of the Triton table
(`meep_gpu/triton_kernels/offdiag_update_e.py`, `folded_offdiag_update_e.py`) and the two
scratch-output welds that carry them (`offdiag_fused_electric_pair.py`,
`folded_offdiag_fused_electric_pair.py`) now compute what the array path computes.

- **MEEP's row, in double, rounded once.** `gs*us` is widened, each surviving term's
  float32 sum `(g[i] + g[i-sx])*u[i] + (g[i+s] + g[(i+s)-sx])*u[i+s]` is wall-masked,
  widened and scaled by MEEP's double `0.25`, the terms are added in double, offset 1
  then offset 2, and the row is rounded to float32 once (`stepping._offdiagonal_terms`,
  `_rounded_row`). The kernels formed the row in float32 before.
- **Per-chunk presence.** A term is added only on the cells whose chunk keeps the row
  slot's `chi1inv` array (`Fields.offdiag_presence`). The plans bind it as a second
  `int32` vector, `op` (`offdiag_update_e.presence_record`: a 27-bit mask per slot over
  the `{low, interior, high}` product and the per-axis segment lines of the division's
  owner volume); a slot no chunk drops compiles to no read. The engine route packs the
  array path's own presence (`presence_for`); the harness builders require `presence`
  (`ALL_PRESENT` states every term present), and a division the kernel cannot index is
  refused by the coverage predicates by name.
- **The constitutive form per chunk.** The kernels read the chunk record `cb`
  (`launch.chunk_bounds_for` on the engine route, required on the harness route) and
  take MEEP's accumulation `E + ((kps*src) - (kms*f_w))` on the cells whose chunk carries
  the component's own layer and `E = src` elsewhere, with `f_w` read and written on the
  accumulation's cells only (`forms.constitutive_form`).
- **The unfolded weld's curl half is the root curl's body again**, lifted at its anchors
  (the store anchor is now `tl.store(u0 + idx, n0, mask=live & du0)`), so it reads the
  record at each recomputed neighbour; its `fu` twins take the pre-launch word outside the
  recurrence's cells, which keeps the auxiliary contract across the rotation. The folded
  weld's curl half is re-lifted from the folded curl single
  (`symmetry.pml_curl_step_folded`, on the chunk forms since the folded real lane) at
  the same anchors, with the same `fu` twin rule.
- **Builders.** `plan_offdiagonal_constitutive`, `plan_folded_offdiagonal_constitutive`
  and the two weld builders take `chunk_layout`, and `plan_step` hands the run's division
  to the two `update_E` arms. A folded plan composed by another family for its bindings
  alone binds neither vector and refuses to launch, by name.
- **Gates.** The references of `gate_triton_offdiag.py`, `gate_triton_folded_offdiag.py`
  and `gate_triton_offdiag_stencil_welds.py` are re-transcribed to the forms above; the
  off-diagonal gate's sweep binds two-run records and hand-built per-chunk presences,
  and gains a chunk-form battery (select removed, presence ignored, float32 row, `f_w`
  stored on plain cells, integer lattice, membership one row off, presence on the wrong
  lattice, and the unconditional auxiliary load as a declared null); the needles that
  quoted the previous text are re-cut, and the weld gate's curl grouping mutation is
  inverted to plant the previous pairwise grouping.
- **The off-diagonal gate's mutants key their own binaries.** Triton's dependency finder
  leaves every callee whose module name starts with `triton` out of a kernel's cache
  key, so under the former `triton_mutated_offdiag_*` names a mutation of a helper alone
  (`_row_add`, `_offdiag_term`) keyed like every other such mutant of the same
  specialization and was served the first one's binary: the float32-row mutant ran the
  distributed-quarter null's binary and read as not caught. The mutants are now
  `offdiag_mutant_*`, and the mutation leg refuses a mutant whose kernel keys like the
  shipped kernel or an earlier mutant.

### The Triton conductive kernels step each chunk's own conductive form

`meep_gpu/triton_kernels/conductivity.py` (`conductive_pml_curl_step`) and the
conductive fused electric pair (`conductive_fused_electric_pair.py`) now carry the array
path's conductive forms under a PML (`stepping._apply_conductive_pml_update`), selected
per structure chunk as MEEP selects them. Before this change they keyed the forms on
coefficient identity (`kms != 1`), added the conductive increment in two operations and
formed the curl pairwise.

- **The forms.** Each component's dsig and dsigu memberships are the chunk record's lines
  (`cb`) on its own sub-lattice. In a chunk that keeps the component's conductivity the
  four conductive forms are selected, with the increment `(f_cond' - f_cond)` formed
  first and then added, MEEP's grouping. In a chunk that keeps none, the lossless forms
  of `forms.curl_form` run. A lossless component (a per-component conductivity) runs
  `forms.curl_form` everywhere, as `kernels.pml_curl_step` does. The curl, its partner
  presence and the ownership mask are `kernels.pml_curl_step`'s.
- **The third test.** Whether a cell's chunk keeps the component's conductivity is a
  per-chunk fact the record cannot carry. A component whose chunks do not all keep it
  binds the division's conductive-chunk volume (`ChunkLayout.conductive_mask`, as
  `uint8`) and its `CMASK` constexpr is 1. A component whose chunks all keep it binds
  none and reads none.
- **The auxiliary contract.** `fu` is loaded and stored only on the component's dsigu
  cells. `f_cond` is loaded and stored only on the dsig cells of conductive chunks.
- **The fused pair's E half** is `kernels.fused_curl_constitutive_D`'s: the dsigw
  accumulation through `forms.constitutive_form`, `f = src` elsewhere, `f_w` only on its
  own layer's cells.
- **Plans.** `ConductivePmlCurlPlan` and `ConductiveFusedElectricPairPlan` bind the record
  through `launch.bind_chunk_bounds`. Their engine builders take `chunk_layout`. The
  harness builders (`plan_conductive_from_arrays`,
  `plan_conductive_fused_electric_pair_from_arrays`) require `chunk_bounds`.
  `launch.plan_step` hands the run's division to the `conductive PML` arm and to every
  certified fused product whose builder takes it.
- **Coverage.** A conductive component whose conductivity volume cannot be read
  (`Fields.conductivity_for`) is refused by name, since which chunks keep it cannot be
  decided.
- **Gate and probe.** `parity/meep_gpu/gate_triton_conductivity.py` re-transcribes its
  reference onto the record and the conductive-chunk volume. Its `validate` leg now also
  steps whole conductive sub-steps against `stepping.step_B` / `step_D`. It adds a
  planted signed-zero leg, inverts the stencil and increment groupings (KERNELS.md 7.1),
  adds the B battery (7.2) and retires the near-one tolerance leg with the predicate.
  Its mutation leg now compiles each mutant under its own kernel and helper names: on
  Triton 3.1.0 a mutant compiled after another one in the same process launched the
  first one's binary (the outer kernel's text was the same, and its cache entry was
  reused), so every row after the first reported the first mutant's verdict.
  `--mutation-sweep smoke` runs the leg on one shape and the synthetic source.
  `probe_triton_conductive_fused_electric_pair.py` re-cuts its transcription anchors and
  mutation needles the same way, adds an interior-only conductivity case, and replaces
  its coefficient-predicate leg with `unit_coefficients`.
- **Measured** on the host interpreter of the kernel text against the array path:
  0 differing words at steps 1, 4, 16 and 64 on 14 of 14 rows (7 configurations x singles
  and fused). The same configurations on the previous kernels differ in 9,433 to 76,844
  words at step 1. The host mutation battery puts 16 planted defects (two of them
  declared nulls) and the unmutated text at their verdict, 17 of 17 runs. The gate's
  mutation leg, run on the same interpreter, puts 24 of 24 at theirs. The device smoke is
  in the lane report.
- **Listed by contract** in `tools/ci/pending_certification.txt`, each with the lane that
  removes it. The conductive BFAST H->D weld (`conductive_bfast_fused_hd_pair.py`, KT11)
  still calls `_conductive_component` at its previous arity. The weld digests of the two
  modules moved. Ten listed ids pass now and left the list: the gate reference's six
  (AP6d's block) and the four transcription checks of KT1's block.


### The hand-written CUDA off-diagonal update_E steps MEEP's double row, per structure chunk

The off-diagonal (tensor epsilon) `update_E` of the hand-written CUDA table, its folded
variant, and the two stencil welds built on them now compute what the array path
computes (`stepping._offdiagonal_terms`, `_rounded_row`, `_apply_constitutive_regions`):
`update_E_pml_real_offdiag` (`cuda_kernels/offdiag_emitter.py`,
`offdiag_constitutive_kernels.py`), `update_E_pml_real_folded_offdiag`
(`folded_offdiag_kernels.py`), `offdiag_fused_electric_pair_pml_real` and
`folded_offdiag_fused_electric_pair_pml_real` (`offdiag_stencil_weld.py` and the two
family modules).

- **The row.** Each OFFDIAG term's pair sums, products and their sum stay float32;
  `offdiag_term` now returns that sum, the wall mask zeroes it per term, and the scale
  `0.25` and the row sum are double: `row = (double)(gs * us)`, each kept term added in
  partner order, `src = (float)row`, rounded once. A component with no surviving row
  keeps the float32 diagonal product.
- **The presence words.** MEEP drops an OFFDIAG term in every chunk that deleted the
  row's `chi1inv` array. Each kernel takes one 27-bit word per live row (`kept_*`) and
  reads, per cell, the chunk its labels on the run's chunk record name
  (`chunk_label`, `chunk_row`, `chunk_keeps`; the metallic wall row reads the chunk of
  the nearest owned row). The new `cuda_kernels/offdiag_presence.py` builds the words
  from the array path's own per-cell presence and refuses by name a record whose
  chunk labels do not reproduce it.
- **The tail.** `constitutive_apply` is the root kernels' flagged helper, character for
  character: MEEP's accumulation on the cells whose chunk carries the component's own
  layer, `f = src` elsewhere, `f_w` read and written only there. The single's launch
  text is hoisted by `own_cell_hoist` again (its row was already in the table).
- **The welds.** The curl half lifts the root `pml_apply` with its membership flag; the
  scratch carries the pre-launch split field on cells outside the recurrence's chunks,
  and `raw_step_D_cell` evaluates the chunk record at the cell it is called for. The
  folded weld's chunk-label lines read the constitutive half's boundary codes.
- **The launchers.** Every launcher takes the chunk record (`chunk_bounds`) and the six
  presence words (`kept`); the dispatch resolvers and the two fused plans bind the
  composition's one record upload and build the words against it once per frozen
  configuration.
- **Gates.** `gate_cuda_offdiag.py`, `gate_cuda_folded_offdiag_kernel.py`,
  `gate_cuda_folded_offdiag.py` and `gate_cuda_offdiag_stencil_welds.py` bind the
  record and the words on every launch; the constitutive-grouping mutation now plants
  the grouping used until 1.0, and the battery adds the float32 row, the select
  removed, the form keyed on coefficient identity, a record bound off by one, the two
  lattices' records swapped, an ignored or cleared presence bit and the plain form as a
  unit-coefficient accumulation (the last three on presence cases with planted `-0`).
  The stencil-weld gate's block-size sweep records a size above the shipped 256 that the
  compiled weld cannot launch as not launchable, by name, as the complex weld gate does:
  the folded weld now compiles to 72 registers a thread on compute capability 8.6, which
  allows at most 896 threads a block.
- **Records.** `certification.json` still describes the previous texts; the off-diagonal
  record tests stay listed in `tools/ci/pending_certification.txt` until the gates are
  re-run on these bytes.

### The hand-written CUDA conductive curl takes MEEP's per-chunk forms

The conductive PML pair of the hand-written CUDA table (`cuda_kernels/conductive_kernels.py`:
`step_B_pml_conductive`, `step_D_pml_conductive`) is derived again from the root curl pair's
text ("The hand-written CUDA PML kernels step MEEP's forms per structure chunk", below) and
steps MEEP's conductive curl as the array path does (`stepping._apply_conductive_pml_update`).

- **The form is chosen per structure chunk, from data.** Each conductive component reads the
  membership flags of its dsig and dsigu layers from the chunk record (the root decode block
  declares them on the component's own sub-lattice), not `kms != 1`, and whether its chunk
  keeps the conductivity: `KEEP<n>` is compiled in where every chunk keeps it, and where some
  chunk keeps none the region plan's `conductive_mask` is bound as `ck<n>` and that chunk
  steps the non-conductive forms (the root `pml_apply`). `conductive_keeps` asks the run's
  region plan, and the predicate refuses by name a run whose plan cannot answer.
- **The arithmetic.** The increment `(c' - c)` is formed before it is added, MEEP's grouping
  (`((u*km1) + c') - c` before). `fu` is read and written only where the cell's chunk carries
  dsigu and `f_cond` only where it carries dsig, and both keep their words elsewhere.
- **The record.** `step_conductive_curl` requires the chunk record on every arm now.
- **The fused D/E pair.** `conductive_fused_electric_pair.py` lifts the re-derived pair with one
  record decode over both halves (the real twin's union block), takes the record and the three
  conductive-chunk masks, and emits one text per conductivity mask and keep mask;
  `fused_pairs.py`'s plan binds the composition's record and the run's keep masks.
- **The gates.** `gate_cuda_conductive.py` hands every case its one-rank record, adds two
  specs with the conductivity on a half-space (so some chunk keeps none), retires the near-one
  table and its tolerance mutation (the coefficient predicate is gone), inverts the increment
  grouping, and carries the B battery: the predicate on coefficient identity, the flags
  swapped, a record bound off by one, the two lattices' records swapped, the conductive chunks
  ignored, and the two unguarded auxiliary loads as declared nulls.
  `gate_cuda_conductive_fused_electric_pair.py` binds the record and the keep masks, adds the
  corpus row's half-space layout, and re-spells its battery the same way.
  `gate_needle_anchors.json` is re-minted; only `gate_cuda_conductive`'s table moved.
- **Still owed.** The admission record and the certification blocks pin the 2026-08-21 and
  2026-09-02 bytes (re-cut at certification), and the conductive BFAST H->D weld derives from
  the re-derived pair but its launcher binds four arguments short (its lane); both are listed in
  `tools/ci/pending_certification.txt`.

### The dispersive polarization update is MEEP's arithmetic

MEEP's `lorentzian_susceptibility::update_P` steps each Lorentzian and Drude term with four
`realnum` constants and the isotropic expression
`gamma1inv * ((p * (2 - omega0dtsqr_denom) - gamma1 * pp) + omega0dtsqr * (s * w))`
(susceptibility.cpp:188-256). The array path now forms the same words.

- **The constants.** `Susceptibility.realnum_constants(dt)` returns them as float32 words
  (`RealnumConstants`: `two_minus_od`, `gamma1`, `omega0dtsqr`, `gamma1inv`), rounded where
  MEEP rounds them: `omega_0`, `gamma` and `dt` as float32, `2 * pi * omega_0` and
  `gamma * 2 * pi` in double and stored as float32, the rest float32 left to right
  (:191-195). It replaces `Susceptibility.coefficients(dt)`, which folded the three-term
  constants `(c_now, c_prev, c_drive)` in double; `PolarizationState.constants` replaces
  `PolarizationState._coefficients`.
- **The update.** `PolarizationState.update` is MEEP's expression with MEEP's grouping,
  every operation float32, a uniform sigma rounded to float32 once as MEEP stores `s[i]`,
  and every product on complex storage formed part by part (`part_products.scale`), MEEP's
  `DOCMP2`. The drive is `Fields.drive_field`, MEEP's per-chunk `w = f_w ? f_w : f`
  (update_pols.cpp:44), unchanged.
- **Stored words change** in every run with a susceptibility. MEASURED against MEEP 1.33.0
  single precision built without contraction, on Apple silicon: phase 0's eight dispersive
  lifts (P0-c: Lorentzian, Drude, both, lossless; TM and TE; 2-D and 3-D; uniform media and
  a block on the declared-geometry sigma route; 9 or 27 chunks) are identical to MEEP at
  every compared step, on every cell and component, sign of zero included: every step 0 to
  20 and the checkpoints to step 400, 0 of 15,453,504 compared words (16 of 16 legs, a
  planted one-ulp flip caught on each); the tree before this change differs in
  3,056,502 values (16 of 16 legs).
- **Tests.** `test_dispersion.py` pins the constants and one update against MEEP's source
  compiled without contraction (five terms, fourteen updates, signed zeros and subnormal
  operands included), the vectorized update against MEEP's scalar expression on every word
  of a random volume (the folded form differs on a large share of them), complex storage
  against two real passes on its parts, and the Drude and angular-frequency rules on the
  float32 words. `test_stepping.py`'s two byte pins with a susceptibility (the complex core
  digest and the periodic `dispersive` case) are re-derived by their own procedures.
  `parity/meep_gpu/mutate_from_meep.py` gains four mutations, each caught: the folded
  three-term form (the inverted mutation), the drive product re-associated, a constant
  formed in double, and the drive taken from `f_w` on every cell under a layer.
- **Not compared with MEEP yet:** complex storage (pinned against two real passes), a
  susceptibility with mirror symmetry, cylindrical coordinates, a nonlinear component, a
  conductivity or an off-diagonal `chi1inv` row. MEEP's anisotropic sigma branches are not
  implemented on this engine.
- **Certification.** The three kernel tables' polarization kernels still transcribe the
  folded form and read the triple from `PolarizationState._coefficients`, which is gone, so
  their coverage checks refuse the `update_P` product by name and a dispatched run steps
  the polarization on the array path, its other slots dispatching as before (MEASURED on an
  RTX A6000 under the Triton and the CUDA preference, no step raised; on an Apple GPU the
  held-residency guard then refuses the whole configuration to the array path, 0 differing
  stored words against the NumPy reference, with and without a layer, real and complex).
  136 tests are listed in `tools/ci/pending_certification.txt`, each with the lane that
  removes it: 21 CUDA, 101 Triton and 13 Metal polarization tests, and the Tier A ledger's
  oracle check. The `array_path_oracle` fingerprint moves (2 of its 20 fixture cases,
  `dispersive_3d` and `dispersive_pml_2d`; this change keeps the fixture set
  `array_path_oracle/real-v4`, which the batch's re-mint moves once), so that check is
  listed until the set is minted again. Measured on Apple
  silicon: apart from these and this change's own tests, no id of the package suite (14,527
  ids, every marker), its 74 device files on the Apple GPU (3,460 ids) or the harness suite
  (1,413 ids) changes outcome.

### The Metal conductive PML curl steps MEEP's conductive forms per structure chunk

The real-field conductive PML curl of `metal_kernels/conductive_pml.py` and the two
fused products that lift it (`conductive_fused_electric_pair`, D -> E, and
`conductive_fused_hd_pair`, H -> D) now transcribe the array path's per-chunk conductive
forms (`stepping._apply_conductive_pml_update`):

- **The four forms are chosen per structure chunk, from the bound chunk record.** The
  kernel binds the record as `constant ChunkBounds& cb` (buffer 29, its last) and reads
  it through the base curl's anchored bounds block, on the target's own sub-lattice
  (half-integer for B, integer for D). A lossy target steps MEEP's four conductive forms
  on its dsig and dsigu memberships (both layers, dsigu only, dsig only, neither); a
  lossless target of the same sub-step steps the base curl's two forms. The forms were
  keyed on coefficient identity (`kms != 1.0f || sinv != 1.0f`) before this change.
- **MEEP's grouping.** The `f_cond` increment `(fcnd' - fcnd)` is formed first and then
  added (step_generic.cpp:167, :209), and the stencil is the base curl's own text,
  lifted from `shaders.curl_source` (MEEP's left-to-right bracket, with an absent
  partner per the record's presence word).
- **The auxiliary contract.** `fu` is read and written only on dsigu cells and `f_cond`
  only on dsig cells.
- **Per-chunk conductivity presence is refused by name.** MEEP keeps a chunk's
  conductivity only where a sample it visits is nonzero, and a chunk without one steps
  the lossless forms (`ChunkLayout.conductive_chunks`). The record carries no such
  presence, so a sub-step whose lossy target is not kept by every chunk is refused to
  the array path (`conductive_pml.conductive_presence_reasons`, read on the composer's
  `chunk_layout`).
- **Every plan binds a record, with no default.** `plan_metal_conductive_pml_curl` and
  the two fused builders take a required keyword `chunk_bounds` (and `chunk_layout` for
  the presence); the arms hand over the composer's. The fused products carry the record
  in `Params` (`ChunkBounds cb`, aliased with `chunk_bounds.params_alias()`), so their
  binding counts do not move (29 and 27).
- **The H -> D weld** takes `fused_hd_pair.h_cell`'s total `f_w` word (`wnext`) and
  refuses a run's first `update_H` on `launch.h_start_reasons`' rule. Its constitutive
  half is `fused_hd_pair`'s lift as the H->D welds' change re-derived it (`h_cell` takes
  the record and returns the total `f_w` word), so the weld emits and compiles on every
  shipped specialisation.
- **Gates.** `gate_metal_conductive_pml.py` binds the one-rank record, records a form
  census per product, inverts its grouping mutations (`curl_previous_grouping`,
  `increment_previous_grouping` plant the package's previous groupings) and carries the
  per-chunk battery (`b1`..`b8`, `b7` a declared null) and the two record mutations;
  `parity/meep_gpu/test_gate_metal_conductive_pml_host.py` checks its census and needles
  on the host. `gate_metal_conductive_fused_hd_pair.py` is re-cut to the same forms and
  to the AP6d helper signature; `gate_metal_residue_fused_pairs.py` binds the record for
  the conductive pair and plants the previous increment grouping.
- **Certification.** The kernel-source digests of the three modules move;
  `fingerprints.json` and every ledger are unchanged, pending the certification round.

### The Metal complex families step the per-chunk forms with part-wise products

The complex Metal kernels -- `metal_kernels/complex_fields.py` (the complex/Bloch PML curl
and constitutive), its lifters `complex_fused_magnetic_pair`, `complex_fused_electric_pair`
and `complex_fused_hd_pair`, the folded family `folded_complex` (curl, fill, constitutive)
with `folded_complex_fused_magnetic_pair`, `folded_complex_fused_pair` and
`folded_complex_fused_hd_pair`, and `complex_conductive_pml` -- now transcribe the
array path's per-chunk forms and its part-wise complex arithmetic (`part_products`):

- **Products are MEEP's, part by part.** `templates.complex_helpers` emits one body for
  every arm name: a real coefficient times a field is one float32 product per part, and
  the Bloch connection is MEEP's unfused `complex<realnum>(phase) * word`. The `FMA_V1`
  arm (NumPy's fused complex multiply) and the zero-cross-term helpers are gone. The arm
  names stay as the vocabulary of the probe-bound expansion licence, whose chain is
  deleted in its own change; the predicates still refuse without a probe record.
- **The Bloch connection is the array path's.** A phased axis wraps plainly and the
  UNSHIFTED operand is connected on plane 0 (`stepping._bloch_ghost_operand`): a copy
  within 1e-13 of +1, a negation within 1e-13 of -1, otherwise the unfused product with
  `conj(eikna)` on the B side and `eikna` on the D side. `complex_fields.phase_arguments`
  returns the per-axis connection kind (`CONNECT_NONE` / `CONNECT_NEGATE` /
  `CONNECT_PHASE`) a curl source is specialised on.
- **The per-chunk forms.** The curl takes the split-field recurrence on a cell whose chunk
  carries the target's dsigu layer and the dsig-only form `((f*kms) - curl)*sinv`
  elsewhere (`fu` touched only on the recurrence's cells), with MEEP's left-to-right
  stencil and partner presence; the constitutive takes `f + ((kps*src) - (kms*prev))`
  with `f_w = src` on the component's own dsigw cells and the output select `f = src`
  elsewhere. The singles bind the chunk record last (the curl at 24 bindings, the
  constitutive at 23); the fused pairs carry it in `Params`, and the H->D welds' `h_cell`
  takes it as a parameter, so a recompute selects its form at the recomputed cell.
- **The folded fill is a per-part scale.** The array path writes a mirror ghost as
  `part_products.scale(parity, plane)`, so the fill multiplies each part by the
  host-rounded parity word, a float (the folded fused pairs' `Params` carry the six words
  as floats). On every normal word and zero that is the real fold's sign flip.
- **`complex_conductive_pml` selects on chunk membership.** A target's four cases are its
  chunk's (dsig and dsigu from the record), `f_cond` touched only on dsig cells; its
  conductive forms are MEEP's (the section on the complex conductive PML curl above).
  The record is the 31st binding.
- **Every plan binds a record, with no default.** Every complex and folded-complex plan
  builder takes a required keyword `chunk_bounds`; the arms pass the composer's, and
  `complex_dispersive_spine` forwards it to the base plans it launches. The four pairs
  that take `update_H` refuse the run's first `update_H` on `launch.h_start_reasons`' rule.
- **Gates.** `gate_metal_complex.py`'s in-gate transcription keys its forms on the record
  and connects as the array path does; its grouping and product mutations are inverted
  (KERNELS.md §7.1: each plants the package's previous spelling and stays must-catch), it
  arms the §7.2 battery (b1-b13 with the two declared nulls) and a `planted_zero` leg, leg
  `zero_cross_terms` is re-pointed as `part_products` (the shipped helpers against
  `part_products` on exhaustive signed-zero tables), leg `call_site_shapes` is deleted, and
  legs `synthetic` and `contraction` assert that the contraction directive moves words.
  The fused-pair, H->D, folded and conductive gates bind the record, invert their
  grouping mutations, arm the battery on both halves where they lift a body, and declare
  the connection's operand order an equivalence; `gate_metal_folded_complex.py`'s leg
  `parity` and `gate_metal_folded_complex_fused_magnetic_pair.py`'s `parity_chain_order`
  are inverted (a product by +/-1 per part is exact).
- **The mirror ghost row under two divisions.** The one-rank record never puts a folded
  axis's ghost row (stored row 0) inside a layer; MEEP's division does where the layer
  reaches stored row 2 (`test_chunk_map`'s third pinned departure), and a lift steps
  MEEP's division. `gate_metal_folded_complex_fused_magnetic_pair.py` therefore steps its
  deep case under both the one-rank record and `meep_boxes` (the same boxes resolved
  through MEEP's ownership rule, installed on both engines): the shipped bytes are
  bit-identical under both, the moved ghost coefficient, the dropped ghost `f_w` store
  and the dropped ownership carve-out are caught under `meep_boxes`, and under one-rank
  the first two are declared nulls and the carve-out is a recorded write-write race. The
  product leg's movement floor exempts the auxiliaries the record leaves untouched.
- **Signed-zero spellings are planted, not predicted.** Armed as a predicted null, the
  previous real-coefficient product diverged in `gate_metal_complex_fused_magnetic_pair.py`
  (one `fu_Bx` word on the uniform class, where an exact-zero operand follows it), so it
  and the plain-constitutive row `b13` are must-catch on the +-0 lattice there. In
  `gate_metal_complex_fused_hd_pair.py` `b13` moved to the lattice too, and the previous
  coefficient-left spelling, record-only before, is must-catch on it. Leg
  `zero_spellings` compiles its table under the shipped contraction directive.
- **Not in this change, and red until their lanes:** the Metal families that lift a
  complex body and are not in this step (`complex_conductive_fused_pair`, the complex
  no-PML kernels, `complex_fused_ade_chain`, the cylindrical complex kernels, the complex
  off-diagonal welds) quote lines the forms replaced; their tests are on the pending list,
  by lane. Ledgers and fingerprints are unchanged, pending the certification round.

### The Metal no-absorber curl steps MEEP's stencil

The real-field no-absorber curl of `metal_kernels/no_pml_curl.py`, and the lossless no-PML
D->E pair that lifts it (`metal_kernels/no_pml_fused_electric_pair.py`), form the curl with
MEEP's left-to-right bracket `dtdx * (((g1s - g1) + g2) - g2s)` and its forms for a partner
absent from the step plan, `dtdx * (g1s - g1)` and `(-dtdx) * (g2s - g2)`
(`stepping._curl_from_operands`). The statements are the PML curl's own text
(`no_pml_curl.CURL_STATEMENTS`), so the package spells one stencil.

- **The partner presence is read from the chunk record.** Both kernels bind the record as
  `constant ChunkBounds& cb`, last (buffer 14 on the curl, 19 on the pair), and decode its
  presence word through the anchored bounds block; no membership entry is read, since no
  chunk of a run without an absorber carries a layer. A one-cell axis of a 3-D run keeps
  both partners; an invariant axis drops one.
- **Every builder takes the record, with no default.** `plan_metal_plain_curl`,
  `plan_metal_plain_curl_from_arrays` and `plan_metal_no_pml_fused_electric_pair` require
  a keyword `chunk_bounds`; the composer hands over its own, and `chunk_bounds.plan_words`
  refuses a missing record, or one whose presence word disagrees with the shape, before
  anything compiles. The pair's per-component signature is 20 bindings.
- **The gates.** `gate_metal_no_pml_curl.py` binds each case's record and reads the
  oracle's presence from the same invariance, adds 2-D and 1-D grids, a 2-D engine leg, a
  planted-zero leg (every field word +0 or -0) and the absent-partner mutations (each with
  a declared-null twin on a grid where the arm never runs, the reversed difference on the
  planted state, and a record that marks every axis present), and its grouping mutation
  now plants the package's previous pairwise bracket. `gate_metal_no_pml_fused_electric_pairs.py`
  does the same for the plain pair, with two 2-D cases. Their host half is
  `parity/meep_gpu/test_gate_metal_no_pml_curl_host.py`.

### The Metal fold-carry pairs step MEEP's per-chunk forms

The three folded fused pairs of `metal_kernels/` — `folded_fused_pair` (the D curl, both
mirror fills, the wall clear and the ordinary `update_E` in one launch),
`folded_fused_magnetic_pair` (its B/H twin) and `folded_fused_dispersive_pair` (the
pole-aware D/E pair, per component) — now take the array path's per-chunk forms and bind
the chunk record:

- **The record rides in `Params`.** Each kernel's packed struct carries the 100-byte
  `ChunkBounds` after its eight scalars (132 bytes, `params_record_dtype()`), aliased as
  `cb` before the lifted text runs, so the binding counts are unchanged (31, 28 and 25).
  Every plan builder takes a required keyword `chunk_bounds` with no default; the engine
  route hands over the composer's record, and an admitted configuration given none is
  refused by name (`chunk_bounds.plan_words`).
- **The curl is lifted, not retyped.** Both three-component pairs lift the certified
  folded curl (`symmetry.folded_curl_source`) down to its per-chunk form block, and its
  form statements line by line; the dispersive pair lifts each target's own curl statement
  and the loads it reads (`certified_target_curl`). The stencil is MEEP's left-to-right
  bracket with the absent-partner forms; the recurrence runs where the target's dsigu
  layer holds and the dsig-only form elsewhere, with `fu` touched only on the
  recurrence's cells. One declared edit pair (`LIFT_EDITS`) moves the dsig-only form's
  read of `f` into the owned block, so a destination thread never loads the word its
  source thread writes; the edited `v` is the `n` statement with `p := f`, the same
  operations in the same order.
- **The constitutive half is the certified body, statement for statement.** The
  accumulation `f + ((kps*src) - (kms*prev))` with `f_w = src` where the component's own
  layer holds, and the output select elsewhere, `f_w` untouched there;
  `constitutive_transcription` compares the re-emitted statements with
  `shaders.constitutive_source` on every build (H on the magnetic pair, E on the D pair).
- **Ghost carries select where the cell is.** A destination thread steps its own curl
  form and `fu`. The constitutive membership is on the component's own axis, so the ghost
  the fill moves along that axis — the near ghost on the B side, the far ghost on the D
  side — evaluates its own flag at its destination, beside the coefficient pair it already
  reloads there; the other ghost shares its owner's. The record's lines are
  ghost-adjusted, so this is the array path's own select.
- **The dispersive pair's pole chain is unchanged** (`D - P0 - P1 ...` in pole order); the
  ADE form itself is outside this change.
- **Gates.** `gate_metal_folded_fused_pair.py` and
  `gate_metal_folded_fused_magnetic_pair.py` invert their grouping mutations (they plant
  the package's previous curl and constitutive groupings) and carry the per-chunk B
  battery: the select removed, keyed on the wrong lattice or axis, an auxiliary stored on
  a plain cell, and the record one row wide or with its lattices exchanged.
  `gate_metal_tranche7_fused_pairs.py` hands a record to every plan builder that takes one,
  and its dispersive row carries the inverted groupings and both select removals. In all
  three gates the movement floor admits an auxiliary volume (`fu_*`, `f_w_*`) that stood
  still only when the array path left it untouched too, which the auxiliary contract
  requires wherever a component's layer is absent.
- **Declared resources.** The folded pairs' Metal routing and wiring tests move from
  `[torch]` to `[mps_device]` in `tools/ci/declared_resources.txt`: they build the Metal
  folded pairs, so a PyTorch with no Apple GPU refuses them by name.
- **Tests.** `test_metal_folded_fused_pair.py` and `test_metal_folded_fused_magnetic_pair.py`
  pin the lift (verbatim head, the declared edits and nothing else), the destination
  threads' loads, the transcription, the ghost membership and the record in `Params`; the
  new `test_metal_folded_fused_dispersive_pair.py` pins the same for the dispersive pair.

### The Metal H->D welds step the per-chunk forms of the base kernels

`metal_kernels/fused_hd_pair.py` (the `update_H` -> `step_D` scratch weld) and
`metal_kernels/folded_fused_hd_pair.py` (its folded twin) lift the base kernels' per-chunk
forms again, so both compile and step the array path's words:

- **The record rides in `Params`.** The signature is 31 of 31 bindings, so `Params`
  embeds `ChunkBounds cb` after the five scalars (120 bytes) and the kernel aliases it
  (`chunk_bounds.params_alias()`) before any lifted text. `h_cell`, the certified
  `update_H` body as a function of a cell, takes the record as its last parameter, so
  every foreign recompute selects its form at the recomputed cell.
- **The constitutive lift is re-cut** (`CONSTITUTIVE_LIFT_EDITS`, still seven entries):
  the guarded `prev` load and the accumulation `f + ((kps*src) - (kms*prev))` read the
  pre-launch buffers, and the guarded `f_w` store becomes the register the caller stores to
  the scratch. The scratch is total, because the plan rotates `f_w_H` against a twin that
  starts at zero: off the dsigw cells it carries the pre-launch word, unchanged.
- **The run's first `update_H`.** Both predicates refuse, with `launch.h_start_reasons`'
  words, a run where `stepping.separate_h_from_b` could move a word; a mid-run state
  (`Fields.h_separated`) is admitted as before.
- **Every plan binds a record, with no default.** `plan_metal_fused_hd_pair` and
  `plan_metal_folded_fused_hd_pair` take a required keyword `chunk_bounds`; the arms pass
  the composer's.
- **Gates.** `gate_metal_fused_hd_pair.py` and `gate_metal_folded_fused_hd_pair.py` invert
  their grouping mutations (`curl_previous_grouping`,
  `constitutive_previous_accumulation_grouping` plant the package's previous groupings),
  arm the per-chunk form battery on both halves of the weld (KERNELS.md §7.2: b1, b2, b5,
  b6, b8-b11 must-catch; b3, b4 as record defects; b7, b12 declared nulls), and record the
  carried select's keying as a predicted null. The plain gate gains a `planted_zero` leg.
  The synthetic fixtures, and the nonlinear gate's, are mid-run states.
- **Not in this change:** the four pairs that lift `fused_hd_pair`'s constitutive tail
  (`beta_real_fused_hd_pair`, `bfast_fused_hd_pair`, `conductive_fused_hd_pair`,
  `cylindrical_real_fused_hd_pair`) bind no record in their own templates yet and stay red
  until their lanes. Ledgers and fingerprints are unchanged, pending the certification
  round.

### The Metal same-cell fused pairs step MEEP's per-chunk forms

The fused magnetic and electric pairs, the nonlinear magnetic weld, the fused dispersive
D/E pair and the E->P chain (`metal_kernels/fused_magnetic_pair.py`,
`fused_electric_pair.py`, `nonlinear_fused_magnetic_pair.py`, `fused_dispersive_pair.py`,
`fused_ade_chain.py`) bind the chunk record and step the array path's per-chunk forms:

- **The lifting pairs bind the record through `Params`.** The magnetic and electric pairs
  (and the nonlinear weld, which compiles the magnetic pair's source) splice the base
  curl and constitutive bodies unchanged; their `Params` embeds `ChunkBounds cb` after
  the five scalars (`chunk_bounds.params_member`) and one line before the lifted text
  aliases it (`chunk_bounds.params_alias`). The curl half reads one Yee sub-lattice and
  the constitutive half the other, so the two anchored bounds blocks declare no local
  twice. The binding counts are unchanged (28 and 31 of 31).
- **The dispersive pair takes its curl lines from the base curl's emitted source**
  (`certified_target_lines`: the presence-aware stencil, the dsig-only form and the
  recurrence, and the guarded `fu` store), binds the record as its last buffer (30 of 31),
  and selects its E half per chunk: `f + ((kps*src) - (kms*prev))` with `f_w = src` where
  the component's own chunk holds, `E = src` elsewhere. `_TARGETS` keeps its bytes for the
  three families that import it.
- **The E->P chain's PML arm** selects its E half the same way and carries the record in
  `Params` after the shared blob (`record_params`; `pack_params` and `params_words` are
  unchanged for `complex_fused_ade_chain`). Its ADE recurrence is unchanged: that form
  moves with the dispersive lane. The drive is the register `src`, which is `f_w` on the
  accumulating cells and the stored E elsewhere, MEEP's per-chunk `w`. The no-PML arm's
  source keeps its bytes.
- **Every builder takes the record as a required keyword** (`chunk_bounds`, no default),
  checked by `chunk_bounds.plan_words`; the composer hands over `StepContext.chunk_bounds`.
- **The run's first `update_H`.** The magnetic pair refuses that update where the start
  could change a word (`launch.h_start_reasons`, asked with no source list: a magnetic
  deposit is its bracket's to start).
- **Gates.** The grouping mutations are inverted (they plant the package's previous
  groupings and stay must-catch), every needle on an auxiliary line is re-cut on its guard,
  and each gate runs the per-chunk form battery (`parity/meep_gpu/metal_fused_battery.py`,
  the base gate's own rows) and a form census. The dispersive pair's and the E->P chain's
  separate controls score the fused product against the array path and record the
  pointwise dispersive single beside it, which steps the pre-alignment forms until its
  lane; the E->P chain's PML rows plan with exactly one ADE-half clause withheld by name
  (the pointwise single's requirement that `drive_field` be the `f_w` array).
- **Certification.** The ledgers and `fingerprints.json` are unchanged, pending the
  certification round.

### The array-path oracle moves to `real-v4` and the MEEP-identity Tier A set is re-minted on it

The Tier A ledger records the array-path oracle its set was minted on. The folded-increment
and sources changes below move that oracle, and the sources change also drops the
per-source offset from its digest, so its fixture set is now `array_path_oracle/real-v4`
(one bump for the changes below) and the keep set is minted again on this tree by
`parity/meep_gpu/record_meep_identity.py`.

- **Measured** against MEEP 1.33.0 single precision built without contraction, keeping
  subnormals, on Apple silicon: 4 of 4 cases, 38,040,036 compared stored words and
  61,772,940 reader words equal to MEEP's, every digest reproduced by the replay with MEEP
  never imported; 1,359,846 of the 5,000,000 budget bytes. The four capsules (those of
  cases c and c' no longer carry the removed `VolumeSource._applied_dipole`) and
  `meep_identity.json` change; the digests and currents tables come out byte-identical.
  The ledger records `real-v4`, fingerprint `f61fe1df` (it recorded `real-v3`,
  `b198a15e`).
- **The kernel ledgers are not written.** `record_array_path_oracle.py` in report mode
  reaches all 257 declared form classes over 20 cases and would change the CUDA, Triton
  and Metal ledgers' `array_path_oracle` leaves in 450, 474 and 466 leaves and nowhere
  else, with every entry's bound digest and live capabilities unchanged. Writing them is
  a certification event, so their three oracle tests stay listed.
- **`tools/ci/pending_certification.txt`**: the 7 Tier A tests that the two changes below
  listed (the ledger's oracle test, and the six tests of cases c and c') leave it, and 4
  tests the sources change turned red without listing them are listed: two Apple-GPU
  tests that expect a Metal H->D predicate to refuse an engine integrated electric
  source by its standing withdraw (none stands one now; for KM3 and KM10), and two that
  require the `update_E` and `update_P` consults to be adjacent, where the electric
  `f_minus_p` restore now stands (for KT6).

### The Triton nonlinear update steps MEEP's arithmetic and each chunk's own form

The chi2/chi3 `update_E` kernel of the Triton table
(`meep_gpu/triton_kernels/nonlinear_update_e.py`, `nonlinear_constitutive_step`) now
computes what the array path computes since "The nonlinear constitutive update is
MEEP's arithmetic", and reads the chunk record the way the root kernels do.

- **The transverse sums associate left to right**, `((g[i] + g[i+s]) + g[i-s1]) +
  g[i+(s-s1)]` (`stepping._nonlinear_transverse_sums`), where the kernel paired the
  corners.
- **Dsqr is formed in double and rounded once** (`stepping._nonlinear_dsqr`): `gs*gs` and
  `g1s*g1s + g2s*g2s` stay float32 operations, are widened, scaled by the double
  `0.0625` and added in `tl.float64`, and the sum is rounded to float32 once. For `sm_86`
  the compiled kernel carries, per element and nonlinear component, two widenings, one
  double multiply, one double add and one rounding back, and no `fma.f32`; the
  all-linear specialization carries no double operation.
- **The constitutive tail is the root kernel's.** The kernel takes the run's chunk record
  as `cb`, after its six coefficient pointers, and keys each component's form on the
  record's half-integer line for its own axis: on the cells whose chunk carries that
  layer, MEEP's accumulation `E + ((kps*src) - (kms*prev))` with `f_w = src`; everywhere
  else `E = src`, with `f_w` and the old field neither read nor written. Both forms are
  `forms.constitutive_form`. `NonlinearConstitutivePlan` requires the record
  (`launch.bind_chunk_bounds`); the engine builder binds the run's division and the
  harness builder has no default.
- **The rest of a nonlinear run binds the run's division.** The spine curl and `update_H`
  arms take `chunk_layout` from `plan_step`, and `update_H` is bound to the run's `Fields`
  so its first launch is MEEP's start of H. The nonlinear fused B/H pair takes the
  division too (`launch._install_certified_fused_products` hands `chunk_layout` to a
  certified product builder whose signature takes it), and its plan's H state is the
  wrapped `Fields`, not the scope view, which cannot record that H has started.
- **The gate and the probe are re-pointed** (`parity/meep_gpu/gate_triton_nonlinear.py`,
  `probe_triton_nonlinear_fused_magnetic_pair.py`). The gate's reference takes the new
  forms on a record with cells inside and outside a layer on every axis; its two grouping
  mutations now plant the former pair association and the former two-step accumulation;
  it gains the B battery (select removed, `f_w` stored on plain cells, the integer
  lattice, an own axis swapped, the plain form folded into the accumulation, and the
  unmasked `f_w` load as a declared null), a `planted` leg (D = -0 on cells of both forms)
  and a `dsqr` leg (the shipped `_dsqr` against the array path on operands inside and
  outside the band where the double form differs). Every mutant compiles under its own
  kernel name: under the shipped name the JIT cache served one mutant another binary on
  the device. The allclose-blindness control binds a record with no layer cells, so E's
  error is the Pade factor's own. The probe's needles follow the root fused pair's text,
  and the composition probe's seeded state records that H has started, as a state
  seeded mid-run does.
- **MEASURED, on Mac CPU, with no device.** The kernel text was interpreted on NumPy
  against the array path through `plan_step`, on base + 01-F4 .. 17-F5 with the CUDA and
  Triton root kernels' changes and this one. Eight lifted chi2/chi3 configurations were
  run (a 3-D metallic box with a chi2 volume, a 2-D slab, a 3-D run with two periodic
  axes and a chi3 volume, chi on Ez only, a 1-D line, a division whose x chunks carry the
  layer three cells deeper than the effort volume, electric sources, and a magnetic
  source), each with the fused pair offered and without: every array on `Fields` and
  `PML` was word-compared at steps 1, 4, 16 and 64, giving **0 differing words of 359,856
  (3-D), 202,536 (periodic), 104,786 (2-D) and 3,852 (1-D)** on 16 of 16 rows, four of
  four sub-steps served, and a planted one-bit flip seen. Three planted rows (D = -0 on
  both forms, one `update_E`) read 0 differing words with 2,856, 1,332 and 2,856 plain
  E words at -0. Without this change the same tree's 3-D box differs in 115,529 words at
  step 1, and every fused row but the magnetic-source one raises on its first launch.
  Fifteen planted host defects reach their verdict (12 caught, 3 declared nulls); the
  gate's 23 mutation entries, its planted and dsqr legs reach theirs on the host
  transcription.
- **MEASURED on one RTX A6000 with Triton 3.1.0**, as a development smoke and not the
  gate, under the keep subnormal policy. Through the driver's dispatch, 24 legs (the eight
  configurations through the Triton table, with fused arms off, and through the default
  composition) equal the CuPy array path word for word at steps 1, 4, 16 and 64, 96 of 96
  checkpoints, with Triton kernels launched at each and no CUDA launch; `update_E` is
  `NonlinearConstitutivePlan` on the driver's record. The nonlinear pair's release covers
  only the 1-D line, where the dispatcher installs it; with its label opted in
  (`MEEP_GPU_FUSE_ARMS`) it serves the seven configurations without a magnetic source and
  is refused by name on the magnetic-source one, 16 of 16 legs at 0 differing words on 64
  of 64 checkpoints. For `sm_86` the update compiles in four constexpr variants with no
  `fma.f32`; per element and nonlinear component it carries two widenings, one double
  multiply, one double add and one rounding back, and the all-linear variant no double
  operation. The probe of the nonlinear fused pair releases (12 of 12 device legs, 13 of
  13 mutations at their declared outcome, 3 of 3 refusals), and the composition probe
  passes, 5 of 5 cases and 110 of 110 complete steps exact. The gate passes every
  default leg in one process: the reference 5 of 5 on NumPy and on CuPy, 10 of 10
  refusals, 119 of 119 guarded synthetic rows identical (the 108 fusion-on rows differ, as
  they have since the gate's first run), subnormal 2 of 2, identity 2 of 2, dsqr 2 of 2
  bands, planted 4 of 4, 16 must-catch mutations caught and 3 declared nulls null, and
  the engine route 5 of 5.
- **Certification.** `nonlinear_update_e.py`, `nonlinear_fused_magnetic_pair.py`,
  `launch.py` and `__init__.py` changed, so the Triton welds and records that pin them are
  stale by construction; the tests that read that drift were already listed in
  `tools/ci/pending_certification.txt`. Five ids leave the list: the four
  `test_triton_nonlinear_update_e` reference tests, which now reproduce stepping, and the
  pair-association test 16-AP6h listed, renamed and inverted here. No id is added.

### The folded Triton kernels step each chunk's own form

The folded real family of the Triton table now reads the chunk record it binds as `cb`
and selects every form through `triton_kernels/forms.py`, as the root kernels do: the
folded curl `symmetry.pml_curl_step_folded`, the two folded fused pairs
`folded_fused_pair.folded_fused_curl_constitutive_D` and
`folded_fused_magnetic_pair.folded_fused_curl_constitutive_B`, and the pairs' ghost
carries `_carry_ghost_E` and `_carry_ghost`. Before this change they carried the 0.9.2
forms on every cell.

- **The curl halves are the root curl with the fold's ghost rule and masks**: MEEP's
  left-to-right stencil with the partner presence (`forms.curl_stencil`), the split-field
  recurrence on the cells whose chunk carries the component's dsigu layer and the
  dsig-only form elsewhere (`forms.curl_form`), and `fu` read and written on the
  recurrence's cells only.
- **The constitutive halves and every carried ghost** take MEEP's accumulation on their
  own layer's cells and `f = src` elsewhere (`forms.constitutive_form`), with `f_w` read
  and written there only. A carried ghost cell keys its form where it is, on the record
  line of the component's own axis read at the destination's row; the record is
  ghost-adjusted, so a ghost row reads its image row's line.
- **MEEP's start of H is carried on the folded runs.** The folded `update_H` plan
  (`symmetry.plan_folded_constitutive`) now binds the run's `Fields`, which it did not,
  so a folded run's first `update_H` on the Triton singles did not start H from B. The
  folded magnetic pair takes a runtime `start_h`, which Triton does not specialize on,
  applies it at its owned cells and at every ghost it carries, and sets
  `Fields.h_separated` after its first launch that has programs.
- **The folded mirror fill launches its near and far passes separately**
  (`symmetry.MirrorGhostFillPlan.run_near` / `run_far`, a `NEAR` constexpr beside `FAR`
  in `mirror_ghost_fill`). The dispatcher runs the near pass at the fill slot's consult
  and the far pass at the driver's folded-far consult, after the wall clear, which is
  MEEP's order. The fused launch it replaces wrote the far planes before the clear, and
  where a far ghost plane meets a wall plane of another axis that stores `+0` where MEEP
  stores `-0`: MEASURED on the array path, 13 words of Bz and 13 of Dx per sub-step on a
  grid folding x over a periodic declaration with a z wall, every one a signed zero, on
  this tree and on the tree before it alike.
- **Plumbing.** The folded builders take the run's division from `plan_step(...,
  chunk_layout=...)` (`launch.py`, `__init__.py`); the harness builders
  (`plan_folded_from_arrays`, `plan_folded_fused_pair_from_arrays`,
  `plan_folded_fused_magnetic_pair_from_arrays`) require `chunk_bounds`.
- **The gates are re-pointed.** `parity/meep_gpu/gate_triton_symmetry.py`'s fold
  reference is re-transcribed (MEEP's grouping with the presence, the forms keyed on a
  record; the pairwise grouping is now the control the kernel must not reproduce) and
  sweeps two divisions, every cell in a layer and a split division with unit
  coefficients outside its runs and a sigma = 0 row inside each, with MEEP's plain-curl
  signed zero planted on the plain cells; its mutations invert to the previous grouping
  and gain the chunk-form battery (the select removed, the coefficient predicate, a bound
  off by one, the sub-lattices swapped, the auxiliary store unmasked, the presence
  ignored, the plain form collapsed, and the unmasked auxiliary load as a declared null).
  `validate_fold_reference_vs_stepping.py` hands the reference the oracle's division.
  The two folded fused-pair probes' transcription lines, carry-table parser and
  mutations are re-cut (the association mutation inverted), each gains the battery, their
  mutants import the forms, their moved-state census holds an auxiliary whose layer the
  division gives no cell to the auxiliary contract (it must not move), and the magnetic
  probe's design sweep runs the array path's `update_H`. Two host tests pin MEEP's start
  of H on the folded `update_H` plan and on the folded magnetic pair, the warm pass's
  empty launch included.
- **MEASURED on Mac CPU, with no device.** The three kernels' text was interpreted on
  NumPy against the array path on ten folded configurations (one, two and three folded
  axes over metallic and periodic declarations, even and odd counts, 2-D and 3-D, a
  division whose x chunks are three cells deeper than the layer, and sources on both
  seams), through the singles and the fused pairs, with the dispatcher's two fill
  consults modelled: every array on `Fields` and `PML` word-compared at steps 1, 4, 16
  and 64 gives **0 differing words** (19,074 to 146,346 words per checkpoint), and 90 of
  90 single-slot legs run from a random full-volume state with MEEP's plain-form signed
  zeros planted are identical. Without this change the same tree's first case differs in
  44,128 of 109,806 words at step 1 (56,570 through the pairs). Thirty-three planted
  defects reach their declared verdicts: 26 caught, 3 caught only by a planted signed
  zero, 4 declared nulls null. The symmetry gate's legs on the same interpreter: 1,024 of
  1,024 sweep cases identical, 96 of 96 multi-step cases, 15 of 15 mutations as declared;
  the re-transcribed fold reference equals the array path on 8 of 8 configurations (0 of
  8 before).
- **MEASURED on one RTX A6000 with Triton 3.1.0**, as a development smoke and not the
  gate. The keep subnormal policy was requested on every leg. For `sm_86` the three
  kernels compile in eight constexpr variants with no `fma.f32`. The ten configurations
  above, with the kernels compiled and launched, equal the CuPy array path word for word
  (20 of 20 whole-run rows, 90 of 90 single-slot legs). Through the driver's own
  dispatcher, four configurations under the default composition, the Triton fused pairs
  and the Triton singles equal the array path at steps 1, 4, 16 and 64 on 12 of 12 legs;
  before the fill was split, the three legs of the configuration that folds a periodic
  axis beside a wall differed by 26 and 52 words, all signed zeros. The symmetry gate:
  1,024 of 1,024 guarded sweep cases identical with the unguarded control identical on
  0 of 1,024, 96 of 96 multi-step cases, 14 of 14 fills, 6 of 6 fill mutations as
  expected, 7 of 7 engine grids, and the first 6 of its 15 mutations, each caught (the
  rest of that leg did not fit the window). The folded electric pair's probe: 13 of 13
  device legs, 26 of 26 mutations as declared, 5 of 5 refusals. The folded magnetic
  pair's probe released: 15 of 15, 32 of 32, 5 of 5. The complete-driver folded
  composition gate: 6 of 6 cases, 52 of 52 steps exact.
- **Certification.** `symmetry.py`, `folded_fused_pair.py`, `folded_fused_magnetic_pair.py`
  and `launch.py` changed, so the folded welds are stale by construction. Eighteen ids
  are listed in `tools/ci/pending_certification.txt` by contract: the folded electric
  pair's weld digest, and 17 transcription checks of modules that lift the folded
  kernels' previous text (the folded beta pairs, the folded dispersive pair, the folded
  off-diagonal pair and the folded H->D weld). Four listed ids now pass and left the
  list.
- **Not carried yet.** The folded beta pairs (`folded_beta_fused_electric_pair`,
  `folded_beta_fused_magnetic_pair`) call the folded pairs' ghost carries at their
  previous arity, so on a device their kernels do not compile until the lane that owns
  them re-aligns them; the dispatcher's warm pass refuses such a slot by name and the
  array path serves it.


### The hand-written CUDA no-absorber curl takes MEEP's stencil

The three no-absorber curl kernels of the hand-written CUDA table
(`cuda_kernels/no_pml_curl.py`: `step_B_no_pml_real`, `step_B_no_pml_real_derived`,
`step_D_no_pml_real`) form the curl as the root curl pair does ("The hand-written CUDA
PML kernels step MEEP's forms per structure chunk", below): with the shared prelude's
`curl_stencil`, MEEP's `step_curl` grouping `dtdx * (((g1s - g1) + g2) - g2s)`, and the
one-partner forms where a partner's derivative axis is invariant
(`stepping._curl_from_operands`).

- **The record.** Each kernel takes the run's chunk record, `cb`, as its last argument and
  decodes its partner-presence word alone, with the block `bounds_block.py` emits: with
  no absorber no chunk carries a layer, so there is no membership to read. A one-cell
  axis of a 3-D grid that is not invariant takes the full bracket, which the extent
  alone would not say. `step_no_pml_curl` takes the record as a required argument and
  refuses a launch without it by name, before anything compiles.
- **The constitutive null** (`no_pml_constitutive.py`) admits what it admitted. Its
  `stepping` citations are re-pointed, and the active-layer row of its evidence table is
  re-measured on the per-chunk forms: 2380 of 12288 words move at `update_H` and 2362 of
  12288 at `update_E` (2734 and 2715 before).
- **What is built from these bytes.** The three no-absorber conductive curls
  (`conductive_kernels.py`) are templates of this text, so they carry the stencil and the
  `cb` argument with it; `step_conductive_curl` requires the record on those arms and
  refuses a launch without it by name. Their conductive tail and its host transcription
  stay with the conductive lane. The no-absorber dispersive fused electric pair, which
  lifts the conductive template and asserts the former stencil line, no longer emits
  until the dispersive lane re-derives it.
- **The gate.** `gate_cuda_no_pml_curl.py` hands every case the record of its own
  one-rank division, and its NumPy backend reads the same presence word. The grouping
  defect is inverted (it plants the pairwise grouping these kernels carried until now)
  and stays must-catch wherever a component has both partners, scored as a null on a
  1-D grid, where none has. A new defect, the full bracket over an absent partner, is
  must-catch wherever an axis is invariant and a null where every axis is resolved.
  `gate_needle_anchors.json` is re-minted for this gate's table.
- **Measured.** On an RTX A6000 (CuPy 13.5.1, NVRTC 11.8), the gate's full product
  under each subnormal policy: 224 of 224 licensed cases store `stepping`'s words at
  one launch and after 60 launches, 0 differing words of 430,656 each, under the
  shipped `--fmad=false` build and the unguarded control alike; the misdeclaration
  controls diverge. Under keep, with the mutation battery: every must-catch source and
  host mutation is caught (the grouping defect on 8 of 8, 8 of 8 and 16 of 16 legs, the
  absent-partner defect on 3 of 3, 3 of 3 and 8 of 8) and every declared null moves no
  word, the four dead-prelude nulls and the two new structural nulls included; the
  verdict flips against all three planted defects. The no-absorber conductive curls,
  launched with the record on the conductive gate's own no-absorber cases: under keep
  640 of 640 licensed cases identical, 0 differing words of 1,419,840 at one launch and
  of 1,419,840 after 60; under flush 16 of 16 (reduced product), 0 of 95,040; with the
  record's presence word cleared, 2,580 of 2,970 words move.
- **Certification.** On host (CPU-only shim, every marker) 30 listed tests pass again
  and left `tools/ci/pending_certification.txt`, 8 new tests pass, and 10 are listed in
  a block of their own: the admission record's device digests (re-cut at
  certification) and 9 tests of the no-absorber dispersive fused pair (removed by the
  dispersive lane).

### The hand-written CUDA real fused products step MEEP's forms per structure chunk

The three real-field fused products of `meep_gpu/cuda_kernels/` -- the B->H pair
(`fused_magnetic_pair`: `step_B`, the two B fills, `zero_metal_B`, `update_H`), the D->E
pair (`fused_electric_pair`) and the H->D weld (`fused_hd_pair`) -- are re-derived onto the
root kernels' per-chunk forms, so each steps the forms the array path steps from the same
chunk division.

- **One record decode per kernel.** Each lifted body carried the chunk-record block its
  own entries need; the two pairs remove both
  (`bounds_block.strip_bounds_block`, which refuses unless each is exactly the emitter's
  block) and emit one block over the union directly after the index decode. The H->D weld
  keeps the curl's block at its head and update_H's inside `raw_update_H_cell`, where it is
  evaluated at the recomputed cell. Every launcher takes the record as a required
  keyword, `chunk_bounds` (`step_curl_kernels.chunk_record_argument` checks it), and the
  composer binds the composition's one upload.
- **The lifted helpers take the membership flags.** `pml_apply_reg` /
  `pml_apply_reg_pre` take dsigu and the ownership flag, with the ownership guard placed
  directly before the one load of the field; `constitutive_apply_pre` takes dsigw. The
  own-cell hoist issues `fu` behind dsigu and `f_w_H`/`H` behind ownership and then
  dsigw, and its statement walker binds the guarded load `float x = flag ? y[idx] :
  0.0f;`, so a hoisted helper derives only under the same guard. The H->D weld's pure
  helper hands back the auxiliary on the dsigw run and the word it already holds off it,
  because the rotation makes the scratch the live `f_w_H`.
- **Each carried ghost is keyed on its destination's membership row**, the row its
  coefficient entry is read at. On the one-rank division the near ghost's row reads its
  own flag (`chunk_map.one_rank_line`), so keying at the source row is a different
  answer there, and the host test catches it; on a lifted division the row copies its
  image and the two agree.
- **MEEP's start of H.** The magnetic pair forms B itself, so on the run's first
  `update_H` its launcher re-steps `update_H` on the array path right after the kernel
  (`fused_magnetic_pair.start_of_h`, given the layer); the H->D weld starts H before its
  launch. Both refuse a first launch handed no layer.
- **Records and gates re-cut.** `LIFT_EDITS` gains the union block and the carried
  membership row on both pairs (14 and 11 entries); the hoist's frozen pre-hoist text in
  `own_cell_hoist_reference.json` is re-cut as the unhoisted composition of the emitter's
  own pieces, the construction that reproduces the previous frozen text byte for byte on
  base + dev/series 01..17. In the three gates (`gate_cuda_fused_{magnetic,electric,hd}_pair`)
  the stencil and accumulation mutations now install the 0.9.2 groupings, every other
  needle matches the new text, and each gains the per-chunk battery (the select removed,
  the predicate on coefficient identity, a bound off by one, the two lattices' records
  swapped, an auxiliary stored on a plain cell, and the unguarded auxiliary load as a
  declared null).
- **MEASURED on Mac CPU, no device.** `test_fused_pair_chunk_forms.py` executes both
  pairs' device texts in float32 and compares every array of the `Fields` word for word
  with the array path on 11 cases (walled, periodic, 2-D, a lifted division, one to three
  folds, a deep fold on the one-rank and on a lifted division) with random and planted
  -0 states, mid-run and at the run's first update: 0 differing words on every case.
- **MEASURED on one RTX A6000 (development smoke, not certification).** With dispatch on
  and the CUDA table, the default composition serves **4 of 4** PML slots
  (`step_B`/`update_H` by the B->H pair, `step_D`/`update_E` by the D->E pair) on 6 of 6
  separable cases -- a 3-D metallic box, a 2-D slab, a 3-D run with a periodic layer
  axis, a lifted division whose record differs from the one-rank one, and one- and
  two-mirror folds -- against **0 of 4** on the same cases before this change, and its
  words equal the array path's at steps 1, 4, 16 and 64 (0 differing of 3,538,944 /
  768,000 / 2,048,000 / 768,000 / 1,916,928 / 865,280 per checkpoint). A non-separable
  division is still refused by name (0 of 4). The B->H and D->E gates release on their
  full product (34 of 34 cases each) with every mutation at its declared verdict; the
  H->D weld is word-identical on 22 of 22 product cases, and its gate's product leg fails
  only on the two arrangements that dispatch the nonlinear `update_E` single, whose
  |D|^2 form is a separate lane's (measured on a tree without it; the nonlinear section
  below lands that form).
- **Three mutation verdicts re-declared, and one row narrowed, from the device run**, each
  with its arithmetic reason in the gate: the B->H pair's
  `near_ghost_reuses_the_source_coefficient_pair` (a null on the one-rank division, whose
  near ghost row is never in a run) and `ownership_guard_dropped_on_the_constitutive` (a
  data race, reported); the D->E pair's `curl_metallic_mask_dropped` (a null: the masked
  wall row is in no run, so `fu_D` is not stored there and the wall clear overwrites D);
  and the ghost-helper regrouping is scored only where a far fill runs. The
  certification records that pin these kernels' bytes are re-cut at certification.

### The CUDA nonlinear update adds its transverse sums left to right and forms |D|² in double

`meep_gpu/cuda_kernels/nonlinear_constitutive.py`'s `update_E_pml_real_nonlinear` now
takes the two forms the array path took when its nonlinear update became MEEP's
arithmetic (`stepping._nonlinear_transverse_sums`, `stepping._nonlinear_dsqr`):

- `four_point_sum` adds the four corners in the order MEEP's C reads them,
  `((g[i] + g[i+s]) + g[i-s1]) + g[i+(s-s1)]`, one float rounding per addition, where it
  paired them.
- The new helper `nonlinear_dsqr` forms `gs*gs + 0.0625*(g1s*g1s + g2s*g2s)` as MEEP's C
  evaluates it: the two float squares widened, scaled by the double `0.0625` and added in
  double, rounded once with `__double2float_rn`.

The host checks follow. `test_root_chunk_forms.py`'s interpreter executes the `double`
declarations, the double literal and the one rounding, and compares the whole sub-step
with `stepping.update_E` word for word on three chi kinds, one of them a planted band
(every D in `(-2^-63, 2^-63)`, `chi3 = 9e36`) where only the double form agrees; it runs
the helper itself against `stepping._nonlinear_dsqr` operand for operand. Its NumPy
reader in `test_nonlinear_constitutive.py` takes both forms, and each form put back to
its former spelling is caught. `parity/meep_gpu/gate_cuda_nonlinear.py` re-transcribes
its NumPy backend, inverts `four_point_left_to_right` into
`four_point_pair_association`, re-spells `dsqr_drop_transverse` and
`commute_dsqr_transverse_sum` on the helper, and adds the `dsqr_in_float32` mutation and
the planted `dsqr_band` class it is scored on (every D in `(-2^-62.75, 2^-62.75)`,
`chi3 = 6e36`: the band where the two forms differ under both subnormal policies; one drawn
below `2^-63` reaches the `keep` difference only); the full product grows from 112 to 140
cases. `gate_needle_anchors.json` records the new bytes of the nonlinear target. The
certification record of the kernel declares the device edit and is owed a re-cut.

### Sources: MEEP's own currents, MEEP's merged deposit list and MEEP's `f_minus_p`

A lifted run's sources now reach the fields as MEEP's do, in four parts:

- **The currents are MEEP's own.** The lift binds each source to the simulation's own
  `src_time` (`src.swigobj`; for an `EigenModeSource` with `eig_match_freq=False`, the
  retuned clone MEEP itself deposits through), and every `dipole(t)` and `current(t, dt)`
  the run forms is `src_time::dipole` / `src_time::current` as MEEP's build computes them
  (`sources.SourceEnvelope` classes and the electric source classes,
  `bind_meep_src_time`). The carrier's words depend on the libm entry and the complex
  division the C++ build used, which differ between Darwin and Linux and which the package
  cannot see; MEEP's own object answers for both. The package's own waveforms remain for a
  driver built without MEEP, with no identity claim, and `current` is MEEP's forward
  difference per part.
- **The deposit list is MEEP's `src_vol` list.** A source is one piece per chunk of the
  run's division; a piece with the same component (D and E spellings fold together), the
  same src_time (MEEP's `src_times_equal`, as data) and the same cells as an earlier entry
  of that chunk merges into it, amplitudes summed in double, and the pair deposits once
  (`sources.bind_src_vols`, rebound by the driver whenever the source list or the division
  changes). A `ContinuousSource` folds its amplitude into the point amplitudes, as MEEP
  does, instead of into its waveform.
- **An integrated source enters through `f_minus_p`.** MEEP never puts the dipole in D or
  B; it subtracts it from a copy of D (B) inside `update_eh` and computes E (H) from the
  copy (update_eh.cpp:66-140). The driver now does the same around each constitutive
  consult (`sources.f_minus_p_begin` / `f_minus_p_end`): the words at the integrated cells
  are replaced by `f_minus_p`'s, imaged into ghost and wall cells by the driver's own
  boundary passes, and restored after the update. Removed: `VolumeSource.withdraw`,
  `VolumeSource.reset` and `_applied_dipole`, the standing-offset stand-in, which rounded D
  through the dipole and back. An integrated electric source makes E a stored array, as
  MEEP allocates E once `f_minus_p` exists. In a run without a PML the package stores no
  H, so B keeps `f_minus_p`'s words through `step_D`, which reads H as B: every D, E and B
  word steps as MEEP's, and only H read back at an integrated magnetic source's own cells
  differs (MEEP holds `B - dipole` there). With a live polarization MEEP subtracts P
  before the dipole and the package after it, so those words are not claimed.
- **The `update_H -> step_D` seam holds no withdraw.** `withdraw_hoist` now states that
  nothing stands between the two consults but the magnetic `f_minus_p` restore, which
  writes B alone and which `step_D` does not read; its predicates answer `()` for every
  engine source and still refuse a source class that brings a `withdraw` of its own.
- **A capsule of a lifted run loads only with its currents.** MEEP's `src_time` cannot
  leave the process; the capsule lists the sources that held one
  (`provenance["source_times_from_meep"]`) and refuses to build them without a currents
  table.
- **Stored words change** in a run with an integrated source, with two sources of one
  src_time on the same cells, or with a `ContinuousSource` whose amplitude is not 1. MEASURED
  against MEEP 1.33.0 single precision built without contraction, on Apple silicon: the
  14 cases of the bit-parity phase-0 sources family (Gaussian, continuous, extended and
  volume sources, merged and overlapping pairs, integrated electric and magnetic sources,
  sources inside and across the PML), every step 0 to 20 with a planted one-ulp flip and
  the checkpoints to step 400, store MEEP's word on every cell, sign of zero included: 0
  differing words of 35,481,600 compared, against 2,355,016 on the tree without this
  change; so do three further cases (integrated sources under a mirror plane, a merge
  across a four-chunk division, and integrated sources with no PML, where the only
  differing words are the H words at the integrated magnetic source's own cells).
- **The array-path oracle** moves: its digest no longer carries the per-source standing
  offset, which no longer exists, and its fused-route fixture writes and restores
  `f_minus_p` around the repair as the driver does. The fixture set moves once for the
  batch that carries this change, at its Tier A re-mint.

### Special-kz beta and BFAST are MEEP's separate passes after the curl

MEEP's `step_db` (step_db.cpp:47-175) runs, per component, `STEP_CURL` and then
`STEP_BFAST` on the same component, and only after every component the special-kz beta
pass over Bx and By (Dx and Dy). Each pass adds its increment through its own additive
ladder (step_generic.cpp:255-533). The array path folded both increments into the curl,
which is the same sum in exact arithmetic and a different float32 word. It now runs them
as MEEP does (`stepping.step_B`, `stepping.step_D`):

- **The order on every cell is curl form, then BFAST ladder, then beta ladder.** Two
  additions into one word do not commute in rounding. In a 2-D run, the only one with
  beta, the two passes add nonzero increments to disjoint components (BFAST's reach the
  z components, a one-partner component's BFAST k being zero; beta's reach x and y), so
  there the order can reach signed zeros only: reversing it moves no word of `bzf`
  (below).
- **One ladder for both passes** (`stepping._apply_folded_increment`): `f += raw*siginv`
  outside the component's dsigu chunks, `fu += (df = raw*siginv); f += siginvu*df`
  inside them, keyed on the curl's own region plan; with a conductivity, `raw*cndinv`
  in place of `raw` and `f_cond += it` on the dsig cells of the chunks that keep the
  component's conductivity, as the curl writes `f_cond`. `fu` and `f_cond` are written
  on their own cells only, and a cell MEEP's loop does not own keeps its word, `-0`
  included (the increment is subtracted negated, with `+0` there).
- **BFAST's state takes MEEP's two roundings** (`stepping._bfast_term`): `F = S - F_prev`
  is stored, then `raw = F - F_prev`; the former `S - 2*F_prev` was one rounding. An
  absent partner is not read (MEEP hands `step_bfast` a NULL pointer and swaps a missing
  `g1` for `g2`), and in MEEP's plain one-partner branch the state is `k1*(g1s + g1)`
  with no `- F` (step_generic.cpp:373-379). The surviving k of a one-partner component
  is zero, so that line reaches signed zeros only. A nonzero k on the invariant axis
  multiplies only absent partners, so it now changes no word, as in MEEP (it added
  `2*k*g` before). `stepping._gathered_partners` is removed: no pass reads an absent
  partner's shifted operand.
- **The beta increment** is unchanged, `float32(2*pi*beta*dt*sign) * g` per part (a
  quarter turn of the parts on complex storage), now a pass of its own.
- **Stored words change** in every run with special-kz beta or BFAST. MEASURED against
  MEEP 1.33.0 single precision built without contraction, on Apple silicon, at every
  dump step (0 to 20 and the checkpoints to 400; 200 for `k3`): the complex family's
  nine cases of phase 0, plus three added cases, special-kz beta and BFAST on one run
  (`bzf`), special-kz beta in real storage (`bzr`) and BFAST with a nonzero k on the
  invariant axis (`bfz`), give 0 value-differing words on all twelve (0 of 6,558,296).
  The only differing words are signed zeros on ghost cells, as before: 98 on `bz` and
  196 to 1,012 on `kedge`, `k1`, `k2` and `k3`. A one-ulp flip planted in a copy of the
  package's words at every step 0 to 20 is caught on every leg (21 of 21 each). With the
  passes folded into the curl, the same legs read bz 147,772 of 251,596, bzp 441,340 of
  991,132, bf 120,035 of 278,318, bzf 289,557 of 556,636, bzr 219,994 of 495,566 and bfz
  168,879 of 278,318. Real storage without beta or BFAST does not move: the four probe
  cases step to MEEP's words at every step 0 to 20, as before.
- **Tests.** `test_stepping.py` pins the BFAST state's two roundings against the
  one-rounding spelling, MEEP's plain one-partner line on a planted state, the beta
  ladder's two regions against a restatement, and an unowned `-0` that survives the
  BFAST pass; the invariant-axis test now pins MEEP's NULL partner (the increment is an
  exact zero).
- **The array-path oracle** moves on its beta and BFAST cases (2 of 19 case digests,
  compared at the same fixture-set identity); the fixture set moves once for the batch
  that carries this change, at its Tier A re-mint, and the oracle declares the folded ladders' form classes (the
  `fu` stage and the plain form, and BFAST's two one-partner lines): 227 classes, all
  reached.
- **Certification.** The three kernel tables still fold the two increments into the
  curl. Their host transcriptions of the beta and BFAST arithmetic were already listed in
  `tools/ci/pending_certification.txt` and stay listed for the fold lanes (KC7, KT7,
  KM9). Two Triton BFAST tests that state the array path's former one-partner forms are
  added to the list, for KT7; the Apple-GPU files changed no outcome. The Tier A identity
  ledger records the array-path oracle it was minted on, so its oracle test is listed
  until the next re-mint. Conductivity in a folded pass is transcribed from MEEP's forms
  and keyed per component and per structure chunk, as the curl is; no case measured it against MEEP.

### The MEEP-identity Tier A set is re-minted on the array path of these changes

The Tier A ledger records the array-path oracle its set was minted on. The conductive
and off-diagonal changes below move that oracle, and its fixture set is now
`array_path_oracle/real-v3` (one new case), so the keep set is minted again on this tree
by `parity/meep_gpu/record_meep_identity.py`.

- **The set.** The four probe cases a, b, c and c_prime, minted from MEEP 1.33.0 single
  precision built without contraction, keep subnormals: 38,040,036 compared stored words
  and 61,772,940 reader words equal to MEEP's, every digest reproduced by the replay with
  MEEP never imported; 1,359,817 of the 5,000,000 budget bytes. The four capsules and
  `meep_identity.json` change; the digests, plane digests and currents tables come out
  byte-identical. The ledger now records the oracle `real-v3`, fingerprint `b198a15e`
  (it recorded `real-v2`, `d87e6615`).
- **The kernel ledgers are not re-minted.** `parity/meep_gpu/record_array_path_oracle.py`
  was run in report mode: every one of the 237 declared form classes is reached, and the
  three ledgers' `array_path_oracle` leaves, which still record `real-v1`, would change in
  417 (CUDA), 441 (Triton) and 433 (Metal) leaves and nowhere else. Writing them is a
  certification event, so the three ledger oracle tests stay listed in
  `tools/ci/pending_certification.txt`.
- **34 tests listed** that the change storing complex fields part by part ("Complex
  storage is stepped part by part, and a Bloch axis stores MEEP's owned window", below)
  turned red without listing them: CUDA, Triton and Metal tests that compare a complex
  kernel or its host transcription with the array path, each with the lane that removes
  it.

### The Triton root kernels step each chunk's own form

The four entry points of `meep_gpu/triton_kernels/kernels.py` (`pml_curl_step`,
`constitutive_step`, `fused_curl_constitutive_B`, `fused_curl_constitutive_D`) now read
the chunk record they bind as `cb` (`chunk_map.chunk_bounds_record`, an `int32[25]`).
Before this change they bound it and served one form to the whole grid, which is not
what MEEP does and not what the array path does.

- **`meep_gpu/triton_kernels/forms.py`** is the one home of each form the kernels select
  between, as `@triton.jit` helpers that inline into the caller: `chunk_in` / `record_in`
  (membership of a stored row in a record line), `axis_present` (the partner-presence
  word), `curl_stencil` (the curl per partner presence), `curl_form` (the split-field
  recurrence against the dsig-only form) and `constitutive_form` (MEEP's accumulation
  against `f = src`). Every select is an output select: the discarded form is computed
  and never stored, because folding a plain form into another with unit or zero
  coefficients moves signed zeros. Every Triton digest list that names `kernels.py`
  (the gates' and probes' source digests, `seed_triton_welds.SHARED`,
  `propose_triton_welds.BINDS`, the route gate's `DIGESTED`) names `forms.py` beside it,
  and the bit-identity record gains a `forms_source_sha256` key, which
  `test_triton_kernels` requires.
- **Three arithmetic forms follow the array path** where the kernels had the shapes of
  0.9.2: the curl associates left to right, `dtdx * (((g1s - g1) + g2) - g2s)`
  (`stepping._curl_from_operands`), with the one-partner forms where the record says the
  other partner's derivative axis is invariant; the constitutive update is one add of
  the difference, `f + ((kps * src) - (kms * prev))`
  (`stepping._apply_constitutive_pml`); and `fu` and `f_w` are read and written only on
  the cells whose chunk carries their layer (the auxiliary contract,
  `stepping._apply_curl_regions` / `_apply_constitutive_regions`).
- **MEEP's start of H is carried.** `constitutive_step` and `fused_curl_constitutive_B`
  take a runtime `start_h`, which Triton does not specialize on; on the launch that
  starts H, the accumulation reads the source word as both the field and the previous
  auxiliary on its own cells, as `stepping.separate_h_from_b` writes them.
  `ConstitutivePlan` and `FusedPairPlan` pass it from `Fields.h_separated` and set that
  flag after the first launch that has programs. The warm pass's launch over an empty
  grid compiles the same program and starts nothing, so it leaves the flag alone. The
  from-arrays route and the gates pass 0.
- **Not carried yet:** the constitutive plans that `bfast_curl.py`, `folded_complex.py`
  (folded beta runs) and `cylindrical_triton.py` build from arrays pass no `Fields`, so
  they pass `start_h = 0`. When one of them serves a run's first `update_H`, H does not
  start from B. The lanes that own those modules carry it.
- **The gate's needle table is re-pointed** (`parity/meep_gpu/track_triton_pml.py`): the
  two grouping mutations now plant the 0.9.2 groupings, the auxiliary-store needles
  match the masked stores, and the module the mutation leg compiles imports the forms
  and declares the record's two word offsets.
- **MEASURED, on Mac CPU, with no device.** The kernels' text was interpreted on NumPy
  against the array path, with both compositions (the four separate kernels and the two
  fused pairs), on base + 01-F4 .. 14-AP5 with this change, and again on base + 01-F4 ..
  17-F5 with the CUDA root kernels' change and this one. Five lifted configurations
  were run: a 3-D metallic box; a 2-D slab, with one invariant axis, so the presence
  select decides the curl; a 3-D run with two periodic axes; a division whose x chunks
  carry the layer three cells deeper than the effort volume, so cells inside a chunk have
  sigma exactly 0 and membership, not a coefficient, decides the form; and a run with
  magnetic and electric sources on both seams. Every array on `Fields` and `PML` was
  word-compared at steps 1, 4, 16 and 64, giving **0 differing words of 359,856 (3-D),
  104,786 (2-D) and 202,536 (periodic)**, four of four sub-steps served, and a planted
  one-bit flip seen. Without this change the same tree's 3-D box differs in 189,005
  words at step 1 and the 2-D slab in 45,094. Nineteen planted defects are caught: both
  regroupings, the whole-grid recurrence, an off-by-one membership, each sub-lattice
  swapped, a coefficient predicate in place of membership, each auxiliary store
  unmasked, each plain form folded into the other, the start of H dropped (separate
  and fused), presence ignored, and the fused D pair's lattice. Two declared nulls
  (commuted products, an auxiliary load outside its region) stay null. The gate's
  thirteen root needles reach the verdicts the probe requires, 13 of 13.
- **MEASURED on one RTX A6000 with Triton 3.1.0**, as a development smoke and not the
  gate, on base + 01-F4 .. 14-AP5 with this change. The keep subnormal policy was
  installed on every leg, both sides of every
  compare. Seventeen engine-route legs, with the kernels launched at every checkpoint,
  equal the CuPy array path of the same tree word for word at steps 1, 4, 16 and 64:
  the five configurations above, each through the fused pairs, through the separate
  kernels, and through the default composition, which puts all four slots on Triton's
  fused pairs, plus two runs with the warm pass off. Each checkpoint compares 1.0M to
  6.8M words over 44 arrays, and a planted one-bit flip is seen on every compared leg.
  The base + 01..14 array path equals this tree's on two configurations, and a division
  that is not separable is refused to the array path. Six planted defects are caught
  on the device: the gate's two re-pointed grouping needles, the plain constitutive
  form folded into the accumulation, the whole-grid recurrence, the start of H dropped
  and presence ignored. Two declared nulls stay null. For `sm_86` the four entry points
  compile in ten constexpr variants with no `fma.f32`. Re-run on base + 01-F4 .. 17-F5
  with the CUDA root kernels' change, the 3-D box and 2-D slab legs (fused pairs,
  separate kernels, default composition, series array path) and the two re-pointed
  grouping needles reach the same verdicts.
- **Certification.** `kernels.py`, `forms.py` and `launch.py` changed, so the Triton
  bit-identity gate's record and the device tier of the Triton welds are stale by
  construction; the tests that pin them were already listed in
  `tools/ci/pending_certification.txt`. Thirty-five more ids are listed there, by
  contract, each group with what removes it. Thirty-three are transcription checks of
  modules that lift the root kernels' 0.9.2 text and still carry it (conductive, folded
  real, off-diagonal, beta and BFAST, cylindrical real, and the H->D welds), and two are
  weld digests of probes whose digest lists now name `forms.py`. On the Triton,
  adjacent, radius and harness host legs no other test changes verdict.

### The hand-written CUDA PML kernels step MEEP's forms per structure chunk

The root kernels of the hand-written CUDA table read the run's chunk record and compute
the array path's PML forms: the curl pair `step_B_pml_real` / `step_D_pml_real`
(`cuda_kernels/step_curl_kernels.py`), the constitutive pair `update_H_pml_real` /
`update_E_pml_real` (`constitutive_kernels.py`) and the nonlinear
`update_E_pml_real_nonlinear` (`nonlinear_constitutive.py`).

- **The record.** Each kernel takes `const int* __restrict__ cb`, the run's
  `chunk_map.chunk_bounds_record`, as its last argument, and decodes the entries it reads
  once per thread, directly after its index decode, with the block the new
  `cuda_kernels/bounds_block.py` emits. Its names carry the sub-lattice (`cbi_*` /
  `cbh_*`, `in_i_*` / `in_h_*`, `pr_*`), so a product that welds two kernels emits one
  block over both kernels' entries without a collision. A curl component reads its
  dsigu axis and a constitutive component its own axis, each on the component's own
  Yee sub-lattice there, as `chunk_map.RegionPlan` does. The launch wrappers take the
  record as a required argument and refuse anything but one contiguous `int32[25]`
  device vector; the dispatch resolvers bind the composition's one upload
  (`StepContext.chunk_bounds_device`).
- **The curl.** The stencil is MEEP's `step_curl` grouping, `dtdx * (((g1s - g1) + g2)
  - g2s)`, with the absent-partner forms from the record's presence word
  (`curl_stencil`). `pml_apply` takes one flag, dsigu: the split-field recurrence on the
  cells whose chunk carries the component's dsigu layer, `((f * kms) - curl) * sinv`
  elsewhere, and `fu` read and written only on the recurrence's cells.
- **The constitutive.** `constitutive_apply` takes the dsigw flag: `f + ((kps * src) -
  (kms * prev))` on the cells whose chunk carries the component's own layer, `f = src`
  elsewhere, where neither `f_w` nor the old field is read. The register-view singles
  load `f_w` and the old field, and store `f_w`, only under the flag.
  `own_cell_hoist` and `own_cell_hoist_reference.json` follow, and the rule refuses by
  name a text whose helper takes no flag.
- **MEEP's start of H.** A dispatched `update_H` on fields that have not started H runs
  `stepping.separate_h_from_b` first, as `stepping.update_H` does.
- **Nonlinear.** The constitutive tail of `update_E_pml_real_nonlinear` takes the same
  per-chunk forms. Its source arithmetic is unchanged: it still pairs the transverse
  sums and forms Dsqr in float32, which the array path no longer does ("The nonlinear
  constitutive update is MEEP's arithmetic", below), so its host transcription tests
  stay listed until the CUDA nonlinear lane transcribes MEEP's forms.
- **The kernels derived from these texts are not re-derived yet.** The conductive PML
  pair holds its derivation by name (`conductive_kernels.PML_PAIR_REFUSAL`) and its
  active-layer rows are refused. The fused products, the off-diagonal `update_E`
  singles and the BFAST, beta and cylindrical curls no longer derive or compile; the
  plan-time warm pass leaves their slots on the array path. On a run with an absorbing
  layer, the CUDA table's default composition (the fused pairs) therefore serves none of
  the four PML slots until those lanes land: the warm pass unfills both pairs and the
  half-pair rule refuses the plan, so the run steps the array path. With
  `MEEP_GPU_FUSE_ARMS=0` the four root singles serve them. With the warm pass off
  (`MEEP_GPU_WARM=0`) the default composition raises by name at its first step, as a
  kernel that does not build does there.
- **The gates.** The hand track's needles, and those of the folded, cylindrical and
  nonlinear gates that rewrite the root kernels' text, are re-spelled on that text
  (`probe_fused_kernel_bit_identity.ROOT_PAIR_NEEDLES`). The two grouping defects are
  inverted: they plant the grouping these kernels carried until now, and stay
  must-catch. The hand track launches with `gate_chunk_records.every_cell_record`, the
  gates whose oracle is `stepping` with the new `region_plan_record`, and the lifted-case
  validators with the driver's record. The NumPy legs of the four root-kernel gates are
  re-transcribed. `record_gate_needle_anchors.py` records a table whose targets do not
  derive by name rather than stopping, and `gate_needle_anchors.json` is re-minted.
- **Measured.** On an RTX A6000 under the keep subnormal policy, the root singles store
  the series array path's word in every array of the fields and the layer, sign of zero
  included, at steps 1, 4, 16 and 64 on four cases: a 3-D box with a layer on every
  face, a 2-D slab, a 3-D grid with a layer on a periodic axis, and a 2-D lifted division
  whose record differs from the one-rank one. Every component takes both forms on the
  two 3-D cases; on the 2-D ones 8 of the 12 do, and the four whose layer axis is z (no
  z layer in 2-D) take one. A non-separable division is refused by name with no record
  uploaded and no launch. The singles store the same words with the warm pass off. The
  CUDA legs of the root-kernel gates release against `stepping` on the same device
  (measured before the nonlinear change below moved `stepping`'s nonlinear update):
  nonlinear 112 of 112 cases and cylindrical constitutive 96 of 96 (full products),
  folded constitutive 12 of 12 and the folded curl's resolved-mirror legs (reduced
  products), with each gate's controls differing. Re-run with the nonlinear change below
  in the tree, the root singles store the series array path's words on the same cases
  and the two folded gates' reduced products release.
- **Certification.** 325 host tests that read a derived kernel or a record of the root
  texts' bytes now differ by contract and are listed in
  `tools/ci/pending_certification.txt`, each group with the lane that removes it; 6
  listed tests of the root pair and the nonlinear tail pass again (one under a new
  name) and left the list. The 12 nonlinear-tail tests that read the former pair
  association and float32 Dsqr are listed in one block, removed by the CUDA nonlinear
  lane.

### The off-diagonal constitutive row is MEEP's double-precision sum, per chunk

Where the inverse permittivity has off-diagonal entries, MEEP's `step_update_EDHB`
forms `gs*us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)` in double, because each
`OFFDIAG` is scaled by the double literal `0.25`, and the store rounds the row once
(step_generic.cpp:580-581, :611, :719). The array path now does the same.

- **The row.** `stepping._offdiagonal_terms` returns the unrounded double row, one
  float64 volume per stored part: `gs*us` as one float32 product, each term's pair sums,
  products and their sum in float32, scaled by `0.25` in double, added in
  `cycle_direction` order. `update_E` rounds it once (`stepping._rounded_row`). Before,
  each term was rounded to float32 and the row summed in float32. The metallic-wall mask
  zeroes each term's float32 sum before the scale. Complex storage forms the row per part
  with the real coefficients (`part_products.parts`, `part_products.from_parts`), never
  through a complex multiply. A nonlinear component multiplies the double row by its Pade factor in
  double and rounds once, MEEP's most general case (:590-601); with no off-diagonal row
  its arithmetic is unchanged.
- **Per-chunk presence.** MEEP deletes an off-diagonal `chi1inv` array in every chunk
  where it is zero over every sample the chunk allocates (anisotropic_averaging.cpp:245-292)
  and passes NULL there (update_eh.cpp:191-196), so the term is absent, which differs from
  adding a zero only when the row is `-0`. `ChunkLayout.chunk_samples` (the rows each chunk
  allocates: its owned rows and one past them), `ChunkLayout.offdiag_kept` (per chunk) and
  `ChunkLayout.offdiag_presence` (per cell, `None` when every chunk keeps the array) read
  it from the run's division; `Fields.offdiag_presence` caches it per row and partner and
  drops every term on the cylindrical `r = 0` row, which MEEP re-steps with the
  off-diagonals NULL (update_eh.cpp:197-208). The driver installs its lifted division on
  the fields with or without an absorbing layer (`Fields.chunk_layout`,
  `Fields.install_chunk_layout`), because a lift without one can still carry several
  chunks; with none installed, a run under a layer reads its region plan's one-rank
  division and an unlayered one-rank grid is one chunk. `membership_mask`, the
  conductive presence and this one read every cell's chunk from one owner lookup
  (`ChunkLayout.owner_volume`).
- **`chi1inv` is the caller's reciprocal, rounded once.** `FdtdDriver.set_epsilon` and
  `set_epsilon_components` store `float32(1 / eps)` formed from the caller's float64
  values, as MEEP stores the inverse it forms in double, instead of
  `float32(1 / float32(eps))`. `_coerce_epsilon_array` returns both arrays from one trim
  and validation, so MEEP's padded shape is trimmed once for both. The lift hands
  `1 / chi1inv` in float64, so the installed word is MEEP's own. A float32 permittivity
  keeps its reciprocal word for word: a float32 quotient rounded through double is the
  correctly rounded one.
- **Stored words change** in every run with an off-diagonal row, and in a run whose
  permittivity is not float32-exact where the two reciprocals differ. MEASURED against
  MEEP 1.33.0 single precision built without contraction, on Apple silicon: the eight
  off-diagonal lifts of phase 0 (a uniform declared tensor with and without PML, an
  anisotropic block that crosses into one PML chunk, a smoothed cylinder and sphere; 2-D
  and 3-D, 1 to 27 chunks) are identical to MEEP at every compared step, on every cell
  and component, sign of zero included: every step 0 to 20, and every checkpoint to
  Table 2's N and to step 1000. 0 of 43,557,888 compared words differ (24 of 24 legs);
  the tree before this change differs in 15,980,934 values (24 of 24 legs). With the old
  reciprocal and the new row, 4 of the 8 lifts differ (12 of 24 legs, 7,951,667 values).
  The four lifts of the bit-identity probe install the same reciprocal words as before
  (0 moved).
- **Tests.** `test_offdiag_row.py` pins the row against a scalar transcription of the C,
  word for word on real and complex storage, with the float32 row as a control that must
  differ; the term order; a `-0` row that a dropped array leaves `-0`; the chunk samples,
  kept verdicts and presence on hand-built divisions; the cylindrical axis row; the
  reciprocal on both installation routes and shapes; and the nonlinear rounding. Each
  pin was run against a planted defect when the change was made (float32 accumulation,
  reversed term order, presence ignored or read from the owned rows only, the wrap image
  row dropped, `-0` counted as nonzero, the axis row keeping its terms, the float32
  reciprocal, the nonlinear row rounded first): 9 of 9 caught, counts in the work
  package's report. The two `stepping.py` mutations of
  `parity/meep_gpu/mutate_from_meep.py` that plant off-diagonal defects
  (`offdiagonal_average_unstabilized`, `offdiagonal_wall_mask_dropped`) are re-pointed at
  the new statements and still caught. `gate_cuda_dispersive_offdiag.coupling_is_live`
  reads the row less its widened `gs*us`, since `_offdiagonal_terms` now returns the whole
  row.
- **Not compared with MEEP yet:** complex storage, a nonlinear component with an
  off-diagonal row, cylindrical coordinates, mirror symmetry, a layer on a wrapping axis
  and an off-diagonal row with a susceptibility. The presence rule changes no stored word
  of the eight lifts (forcing every term present leaves them unchanged); it is pinned by
  the constructed case and by its verdicts, which equal MEEP's rule on MEEP's own words
  and chunk volumes in 48 of 48 (row, partner) cases.
- **Certification.** The three kernel tables still form an off-diagonal term as a float
  (`0.25f * (...)`, `cuda_kernels/offdiag_emitter.py:202`, and its Triton and Metal
  counterparts) and sum the row in float, in the off-diagonal `update_E` kernels and the
  fused and welded products that inherit them, so a dispatched off-diagonal row differs
  from the array path until each table's off-diagonal lane transcribes the double row.
  Their host comparisons against `stepping` are already certification tests, so
  `tools/ci/pending_certification.txt` is unchanged. Measured on Apple silicon: the
  package suite's default view (13,350 ids) changes no outcome except the 11 new ids,
  which pass; the certification selection (1,028 ids) has the same outcome for every id
  as before the change, its 606 failures all declared. The `array_path_oracle`
  fingerprint moves (2 of the 18 fixture cases it had when this was measured,
  `offdiag_3d` and `offdiag_pml_3d`); the three ledger checks that compare it are
  already declared.

### A conductivity under a PML steps MEEP's conductive forms per structure chunk

MEEP keeps a conductivity array per structure chunk and component, and deletes it when
every sample the chunk takes is zero (structure.cpp:883-900); a chunk without one steps
the lossless PML forms even where the component is lossy elsewhere. The array path now
does the same, and forms the conductive arithmetic in MEEP's order.

- **Which chunks keep a conductivity.** `ChunkLayout.conductive_chunks(sigma, iyee)` judges
  each chunk on the samples MEEP's `LOOP_OVER_VOL` takes (`ChunkLayout.loop_rows`,
  vec.hpp:170-172): its owned samples plus one more, below them on the integer lattice
  and above them on the half-integer one. On a wrapping axis that extra sample is read at
  its periodic image; past the top of an axis that does not wrap, at the last stored row
  (an approximation, stated in the docstring). `ChunkLayout.owner_volume(iyee)` gives the
  chunk owning each stored cell, and `membership_mask` now reads its PML flags through
  it. `RegionPlan.conductive_chunks` and `RegionPlan.conductive_mask` cache the
  presence per component against the identity of the conductivity volume.
- **The forms.** `stepping._apply_conductive_pml_update` takes the per-cell
  `conductive` volume (keyword-only; `None` when every chunk keeps the conductivity):
  MEEP's four conductive forms on the dsig and dsigu memberships in the chunks that keep
  it (step_generic.cpp:84-90, :112-119, :160-168, :200-210), and the lossless forms
  (`_apply_pml_update`, `_apply_dsig_only_update`) in the chunks that do not. The
  `f_cond` increment is formed first and added as one term,
  `((kms * x) + (fcnd' - fcnd)) * sinv`, where the earlier code formed
  `((kms * x) + fcnd') - fcnd`.
- **The auxiliary contract** gains the chunk: `f_cond_*` is written only on the dsig
  cells of the chunks that keep the component's conductivity.
- **The coefficients.** `condinv` is formed as structure.cpp:699 forms it,
  `1 / (1 + sigma * dt * 0.5)` in double from the float32 sigma word, rounded once to
  float32; the float32 spelling rounds twice and differs on 5 of 10 sampled sigma
  words (0.1, 0.25, 0.3, 0.5 and 2.0) with dt = 0.05. `condfac` is spelled as
  step_generic.cpp:88 forms it, `1 - dt2 * sigma` in float32 with
  `dt2 = float32(dt) * 0.5` (the same words as before).
- **Stored words change** in runs with a conductivity: everywhere through `condinv`,
  and under a PML through the grouping and the per-chunk forms. MEASURED against MEEP
  1.33.0 single precision built without contraction, on Apple silicon (keep
  subnormals, one rank, the lifted simulation's own division): the conductive family's
  six lifted cases (no layer; a block crossing a layer; D and B conductivity in every
  chunk; two blocks whose chunks are kept by the extra sample alone, and one of them
  alone; a 3-D case with a B conductivity everywhere and a D block crossing a layer)
  store MEEP's word on every owned and wall cell, sign of zero included, at every step
  0 to Table 2's N = 200; so do the two lossless controls at steps 0 to 20 and at the
  checkpoints to their N (200 and 375). The only differing words are signed zeros on
  the ghost cells of the control with mirror symmetry, the reader difference the
  source-deposit entry below describes. Counts are in the work package's report.
- **Tests.** `test_conductive_chunks.py` pins `condinv` and `condfac` word for word,
  the samples that decide a chunk's conductivity (both lattices, the periodic image),
  every cell's form against a scalar transcription of step_generic.cpp's eight
  branches on a division where some chunks keep the conductivity and some do not, the
  two mutants (the flattened grouping; conductive forms in a chunk that keeps none,
  including a planted signed zero), and the auxiliary contract. `test_region_dispatch`
  steps a conductivity confined to two chunks on every route. The array-path oracle
  gains the case `conductive_chunks_pml_2d` and the classes `curl_<B|D>_chunk`, and
  its fixture-set identity is now `array_path_oracle/real-v3`.
- **Certification.** The kernel tables' conductive families still run the earlier
  forms. The Triton conductivity gate's NumPy reference, which its kernel transcribes,
  adds the increment in two operations, so its test against the array path differs by
  contract; its 6 ids are listed in `tools/ci/pending_certification.txt` with what
  removes them.

### The Metal base kernels read the chunk record and step MEEP's forms

The real-field PML curl and constitutive kernels of `metal_kernels/shaders.py`, the folded
curl of `metal_kernels/symmetry.py`, and the three chi2/chi3 spine arms of
`metal_kernels/nonlinear_update_e.py` that launch them now transcribe the array path's
per-chunk forms (`stepping._apply_curl_regions`, `_apply_constitutive_regions`):

- **The chunk record is bound, not baked.** Each kernel takes the record as `constant
  ChunkBounds& cb`, last (buffer 20 on the curl, 22 on the constitutive), and reads it
  through the anchored bounds block of `metal_kernels/chunk_bounds.py`, evaluated at the
  cell's own stored index. The specialisation set and its labels are unchanged (18 per
  contraction mode).
- **Curl:** the split-field recurrence on a cell whose chunk carries the target's dsigu
  layer, and the dsig-only form `((f*kms) - curl)*sinv` everywhere else, which is the
  recurrence's first stage with `fu := f`; `fu` is read and written only on the
  recurrence's cells. The stencil is MEEP's left-to-right `((g1s - g1) + g2) - g2s`, with
  `dtdx * (g1s - g1)` and `(-dtdx) * (g2s - g2)` for an absent partner (the record's
  presence word).
- **Constitutive:** `f + ((kps*src) - (kms*prev))` with `f_w = src` on a cell whose chunk
  carries the component's own layer, and `f = src` (an output select) elsewhere; `f_w` is
  touched only on the accumulation's cells.
- **Every base plan binds a record, with no default.** `launch.plan_pml_curl`,
  `plan_constitutive`, `plan_from_arrays`, `plan_constitutive_from_arrays`, the plan
  classes, the folded and nonlinear-spine builders, and the four families whose
  constitutive half is the base kernel (`bfast_curl`, `cylindrical_real`, `folded_beta`,
  `special_kz`) take a required keyword `chunk_bounds`; the composer hands over its own
  (`StepContext.chunk_bounds`), and `chunk_bounds.plan_words` refuses by name a missing
  record or one whose presence word disagrees with the plan's shape. The harness callers
  in `parity/meep_gpu/` pass the record of the division their reference steps with
  (`gate_chunk_records.layer_record`, or the driver's own `chunk_bounds_record()`).
- **The run's first `update_H`.** The array path starts H and `f_w` from B there
  (`stepping.separate_h_from_b`); a base kernel does not, so the composer leaves that
  slot to the array path unless B is +0 on every cell at that update
  (`launch.h_start_reasons`: H already started, or B, `fu_B` and E +0 at plan time and no
  magnetic source). A pair that absorbs the slot keeps it. `metal_composition_matrix`'s
  seeded states are mid-run ones and carry `Fields.h_separated`.
- **Ruling R7: chi2/chi3 `update_E` steps on the array path.** The array path forms
  `|D|^2` in float64 (`stepping._nonlinear_dsqr`) and the Apple GPU has no float64; no
  software-float64 form is built, so the Pade arm stays registered but unwired, and its
  predicate refuses every configuration by name (`nonlinear_update_e.R7_REFUSAL`). Under
  the held residency default a composition that leaves `update_E` on the array path is
  refused whole by the residency guard, so a nonlinear run steps on the array path; on
  the shipped bracket the spine dispatches. `gate_metal_nonlinear.py` withholds its Pade
  legs and runs the spine; `metal_composition_matrix`'s nonlinear rows no longer expect
  the Pade arm. (Replaced by software binary64: "The Metal nonlinear update adds its
  transverse sums left to right and forms |D|² in software binary64".)
- **Not in this change, and red until their lanes:** every Metal family that lifts a base
  body (the same-cell, straddle, fold-carry and nonlinear fused pairs) now lifts text that
  reads `cb` without binding it. Its source fails to compile, so `plan_step` refuses its
  slot at plan build and the singles serve. Families with their own templates
  (`bfast_curl`, `special_kz`, `cylindrical_real`, `complex_fields` and the rest) still
  step the pre-alignment forms. Their tests and gates are on the pending list, by lane.
- **Tests.** `test_metal_kernels.py` pins the forms, the inverted groupings, the record
  binding, the auxiliary guards, the splice anchors (a lifting family needs only
  `chunk_bounds.params_alias()` before its body), the routes' required record and the H
  start; `test_metal_symmetry.py` pins the folded curl as the base curl plus the top-plane
  block; `test_metal_nonlinear_update_e.py` pins the R7 refusal;
  `parity/meep_gpu/test_gate_metal_pml_host.py` runs the gate's hand reference and form
  census on the host and checks every needle lands. `gate_metal_pml.py`'s grouping
  mutations are inverted (they plant the package's previous groupings) and it carries the
  per-chunk form battery, a form census and a planted -0 leg.
- **Certification.** The Metal kernel-source fingerprints move (36 of 36) and so do the
  host digests of the edited modules; `fingerprints.json` and every ledger are unchanged,
  pending the certification round.

### MEEP identity is tested in three tiers, and Tier A runs in the default suite

Every stored word of the twelve components on owned and wall cells, and every
`get_array` word, is held to MEEP 1.33.0 single precision built without contraction
(`docs/development/testing.md`, "MEEP identity").

- **Tier A** (`meep_gpu/meep_identity.py`, `meep_gpu/test_meep_identity.py`): the four
  probe cases are replayed from capsules with their recorded source currents, with no
  MEEP, and every digest of MEEP's words is judged at every step 0 to 20 and at Table
  2's checkpoints to N. A failure names the first step, component and x-plane. The
  fixtures (`parity/meep_gpu/fixtures/meep_identity/`, ledger `meep_identity.json`) are
  outside the wheel; without them the tests skip by the declared resource
  `meep_identity_fixtures`.
- **Tier C** (`parity/meep_gpu/meep_identity_tier_c.py`): pair-order stepping of the
  lifted driver and MEEP, a word compare of every stored and `get_array` word, a planted
  1-ulp flip, the four cases' smoke record, in `keep` or in `flush` (x86, one OpenMP
  thread). Its live test declares the resource `meep_identity_reference`, measured by
  `tools/check_install.py`'s identity check.
- **Minting** (`parity/meep_gpu/record_meep_identity.py`): the only writer of the
  fixtures; it writes nothing unless Tier C finds 0 differing words, the replay
  reproduces every digest, every planted flip is caught and placed, its inputs did not
  move and the fixtures fit 5 MB. The ledger records the array-path oracle fingerprint
  it was minted on, and `test_the_ledger_records_the_live_array_path_oracle` holds the
  live array path to it.
- **Two checks on the replay**: a complex-multiply interceptor (an ndarray subclass on
  the driver's arrays that records every multiply of two complex operands), and a CI
  step that reruns Tier A with NumPy's AVX-512, AVX2 and FMA3 loops disabled
  (`NPY_DISABLE_CPU_FEATURES`).
- **NumPy 2.0 is the floor** (`numpy>=2.0`, was `>=1.26.0`): identity is judged on NumPy
  2 releases only.
- A capsule records MEEP's module path from its `site-packages` directory on, not as an
  absolute path, so a capsule names no host directory.

### The nonlinear constitutive update is MEEP's arithmetic

MEEP's chi2/chi3 update (`step_update_EDHB`, step_generic.cpp:542-655) is now transcribed
operation for operation by the array path (`stepping.update_E`):

- **The transverse sums associate left to right**, `((g[i] + g[i+s]) + g[i-s1]) +
  g[i+(s-s1)]`, as MEEP's C does (:646-648), each addition rounded once in the field's
  precision (`stepping._nonlinear_transverse_sums`). The array path paired the corners,
  `(g[i] + g[i-s1]) + (g[i+s] + g[i+(s-s1)])`: the same sum in exact arithmetic, and a
  different float32 word.
- **Dsqr is formed in double and rounded once.** `0.0625` is a double literal in MEEP, so
  `gs*gs + 0.0625*(g1s*g1s + g2s*g2s)` takes its two float32 terms, scales and adds in
  double, and `calc_nonlinear_u`'s `realnum` parameter rounds the result
  (`stepping._nonlinear_dsqr`). This moves Dsqr only where the exact `0.0625*t` is below
  float32's normal range, and on the measured cases no stored word.
- **E and its `f_w` at a mirror ghost are the parity image of the owned cell** in a
  nonlinear run (`stepping._image_nonlinear_mirror_ghosts`, after the constitutive
  update). MEEP never evaluates the update at a ghost (update_eh.cpp:191); its boundary
  pass writes each ghost as an image (step.cpp:105-121). The array path evaluates the
  whole volume, which is the image while the update is pointwise; the nonlinear Dsqr reads
  the transverse partners' neighbours, so at a ghost it is not. `f_w` is written only on
  the accumulation's cells (the auxiliary contract). A linear run is unchanged.
- **Stored words change** in a nonlinear run whose components have a transverse partner
  (four of the five measured cases; the fifth, a TM run, has none). MEASURED
  against MEEP 1.33.0 single precision built without contraction, on Apple silicon, on the
  five nonlinear cases of the bit-parity phase-0 leg P0-h (2-D and 3-D, chi2 and chi3, PML
  and metallic walls, 1 to 27 chunks, one case folded on two mirror planes), every step 0
  to 20 (0 to 100 on the folded case) and the checkpoints to Table 2's N and to step 1000:
  0 value-differing words on any cell and 0 differing words on owned and wall cells, of
  37,055,472 compared. The folded case's only differing words are 12,609 signed zeros on
  ghost cells. On the tree without this change the same lockstep legs read 268,298 to
  2,914,477 differing words on four of the five cases, and the reference words equal, step
  for step, the sha256 the phase-0 leg recorded.
- **Tests.** `test_stepping.py` pins the left-to-right association against a restated
  sum on inexact operands (the pairing must differ there), the double Dsqr in and out of
  the subnormal band, and the mirror-ghost image: E ghosts are images, `f_w` is imaged
  only on the accumulation's cells (a sentinel elsewhere survives), a linear folded run
  takes no write, and the update without the write leaves a ghost that is not its image.
  Each of the four planted defects (pair association, float32 Dsqr, no write, a
  whole-face `f_w` write) fails its test.
- **The array-path oracle** gains a folded nonlinear fixture case,
  `nonlinear_folded_3d` (chi3 only: a chi2 medium is refused on a plane that makes its
  component odd), with declared form classes for the E and `f_w` ghost image, so the
  fixture set is now `array_path_oracle/real-v2`. The pair association and a removed
  ghost write each move the fingerprint; the float32 Dsqr does not, because the
  fixture's words never reach the subnormal band, and is pinned by the operand test
  above instead.
- **Certification.** The three kernel tables still transcribe the pair association and
  the float32 Dsqr, so their host transcription tests of the nonlinear update differ by
  contract and are listed in `tools/ci/pending_certification.txt`, removed by the
  nonlinear lanes of the CUDA and Triton tables (`nonlinear_constitutive`, the Triton
  `nonlinear_update_e`). The Metal `nonlinear_update_e` also forms Dsqr in float32; Metal
  has no double type, so that row takes a software double or steps on the array path, a
  decision still open. No device ran for this change.

### Complex storage is stepped part by part, and a Bloch axis stores MEEP's owned window

MEEP stores a complex field as two `realnum` arrays and forms no complex-by-complex
product while it steps: a real coefficient multiplies each part (the `DOCMP` loops,
step_db.cpp:47, update_eh.cpp:149), and a Bloch connection is
`complex<realnum>(thephase) * word` with every product and sum rounded on its own
(step.cpp:183-192; libc++ `operator*`), or a copy or a unary minus within 1e-13 of
+1 or -1 (`phase_isclose`, boundaries.cpp:359-366). The array path now does the same.

- **`meep_gpu/part_products.py`** is the one home of every product on complex storage:
  `scale` (a `realnum` coefficient times each part; a scalar is rounded to the part
  dtype, a wider array or a complex coefficient is refused by name), `divide`,
  `quarter_turn` (`+-i` times a real coefficient, as an exchange of parts: the
  special-kz beta and cylindrical `i*m/r` couplings), `connect` / `connect_parts`
  (MEEP's connection rule) and `from_parts`. Real storage takes the plain ufunc in the
  same operand order, so a real run's words do not move. NumPy's and CuPy's complex
  multiply is not MEEP's arithmetic: it contracts to a fused multiply-add on a host that
  has one, and with no float-times-complex loop a real coefficient times complex64
  storage multiplies against `(k + 0j)`, whose zero cross terms can flip the sign of a
  zero part. Every curl, PML recurrence, constitutive update, conductive form, BFAST and
  beta increment, cylindrical term, dispersive update, mirror fill, deposit repair and
  the magnetic synchronisation average goes through it.
- **A Bloch-periodic axis stores MEEP's owned window** (`Grid.stored_window_low`,
  `Grid.wrap_count`). MEEP owns `(lc, lc + 2N]` of a component whose Yee shift is 0 on
  the axis and fills `lc` from `lc + 2N` with `conj(eikna)` (boundaries.cpp:203-230).
  Stored cell 0 of such a component is now the owned top point `lc + 2N`, so the
  stencil's shifts are plain rolls and the reads that leave the window are the
  same-index reads across the half cell on plane 0
  (`stepping._bloch_ghost_operand`): `conj(eikna)` times stored cell 0 below the
  window (B's forward differences), `eikna` above it (D's backward differences),
  formed as MEEP forms them. Positional readers follow the same window: the readback
  (`readback.stored_window_axes`, `ghost_phase`; `_wrap_image` is removed), DFT and
  flux gathers (`dft._lattice_fold`) and source placement (`sources._build_source_points`).
  At `k = 0`, and on every Yee-shift-1 component, nothing moves.
- **A runtime probe**, `parity/meep_gpu/complex_product_interceptor.py`, views a
  NumPy-reference driver's arrays as an `ndarray` subclass and counts every
  multiplicative ufunc that reaches a complex operand; additive complex arithmetic is
  counted as traffic, and a planted product must be caught.
  `test_complex_product_interceptor.py` steps nine complex configurations under it (Bloch
  on one and on three axes, a PML on a phased axis, a fold beside a phased axis, a
  Lorentzian with PML, a D conductivity with PML, BFAST, special-kz beta, cylindrical
  `m = 1`) with 0 violations. DFT, flux and energy reductions are outside it, as they
  are outside the identity claim.
- **Stored words change** in complex runs with a Bloch phase (by design: stored cell 0
  holds another point) and, in signed zeros only, in complex runs with a mirror plane
  (the ghost is now the exact negation per part). MEASURED against MEEP 1.33.0 single
  precision built without contraction, the nine complex-family cases of phase 0 at
  every dump step (30 or 31): k0, kedge, k1, k2, k3 and f0 store MEEP's value in every
  word on Apple silicon and on x86-64 Linux, with the two hosts equal row for row,
  and every owned word of k3 carries MEEP's sign of zero (0 of 1,382,400 differ); the
  only signed-zero differences are ghost cells (196 to 1,012 words per case). The
  three cases that need special-kz beta or BFAST as MEEP's separate passes (bz, bzp,
  bf) still differ, by exactly the counts phase 0 measured with those passes folded
  into the curl. Real storage does not move: the array-path oracle's real fixture set
  keeps its fingerprint, and the four probe cases step to the same words at every step
  0 to 20. Counts are in the work package's report.
- **Tests.** `test_part_products.py` pins each function word for word, the sign of
  zero, the `realnum` refusals and MEEP's connection rule. `test_stepping` pins the
  window, its plain rolls and its same-index reads (`stored_ladder`); five of its ten
  raw-byte pins are re-derived by the test's own procedure (the three Bloch cases, and
  the two mirror cases in signed zeros only: 600 of 52,800 and 1,001 of 41,472 words, 0
  values); the other five are unchanged. The Bloch-face tests of `test_fields`,
  `test_dft` and `test_driver_vs_meep` read the new window, and the mutations of
  `mutate_from_meep.py` that named the removed spellings are re-pointed (7 of 7 caught).
  The kernel tables still transcribe the previous complex arithmetic; their complex
  families refuse by name without the expansion record, which is not shipped, so a
  default complex run takes the array path.

### The source deposit is MEEP's double-precision subtraction

MEEP's `step_source` forms `A = sv.current(j) * dt` (times `double(cndinv[i])` under a
conductivity) in `complex<double>` and subtracts `real(A)` from the stored float word,
so the word is widened, subtracted once in double and rounded once
(step.cpp:295-317). Every source deposit of the array path now does the same.

- **The door.** `host_writes.subtract_double(array, linear, real64, imag64=None)`
  gathers the deposit cells, widens them to float64, subtracts, rounds to float32 and
  scatters them back, each a single-operation ufunc, so nothing can contract and NumPy
  and CuPy run the same statements. Complex64 storage takes MEEP's second line,
  `f[c][1][i] -= imag(A)`, on the imaginary word. It replaces `host_writes.subtract`.
  Under a held Metal residency the difference is formed on the host from the owner's
  words: one gather and one scatter of the deposit cells per source per step, exact
  moves, so the words are the array path's. A residency that offers
  `subtract_double_index` is handed the deposit instead.
- **The arithmetic** (`sources._deposit`), per point in MEEP's order, as separate
  float64 ufuncs: `(amp_re * c_re) - (amp_im * c_im)` (and `(amp_re * c_im) +
  (amp_im * c_re)` for complex storage), then `* dt`, then `* double(condinv)`. Never a
  complex ufunc multiply, which a contracting host fuses. Amplitudes are packed in
  complex128, as MEEP's `amp[j]` is `complex<double>`. `ContinuousSource` packs its
  weights unscaled (`_point_amps`) and applies `dt` after the product. Removed:
  `sources._step_source_values`, `_inject_points`, `_sparse_subtract`,
  `ContinuousSource._point_dt_weights` and `_deposit_values`, and the
  `DEBUG_VALIDATE_SOURCE_INDICES` flag.
- **No early return.** A silent current is subtracted like any other, as MEEP's loop
  does: `(-0) - (-0) = +0`, so skipping it is not word-neutral. The Gaussian class's
  `abs(current) < 1e-30` and the volume class's `scale == 0` returns are gone.
- **The conductivity scale is inside the deposit.** The driver's float32 replay,
  `FdtdDriver._inject_electric_through_conductivity`, is removed, with its
  whole-volume fallback for a source that publishes no deposit table. Each source reads
  its own component's `condinv` (`sources._condinv_at`): the D one for an electric
  source and, as MEEP does, the B one for a magnetic source. A magnetic deposit in a B
  conductivity used to go unscaled. An integrated source is not scaled.
- **Integrated sources** subtract `real(amp * dipole)` through the same door, with no
  `dt` and no `condinv`. Their withdraw-and-reinject stand-in for MEEP's `f_minus_p` is
  unchanged; identity is not claimed for them.
- **A deposition table that lists a cell twice is refused by name** when it is packed
  (`sources._point_index_arrays`), as well as at its first deposit: MEEP subtracts each
  listed point separately, which a vectorized deposit cannot reproduce.
- **Stored words change** in every run with a source. MEASURED against MEEP 1.33.0
  single precision built without contraction, on Apple silicon: the four lifted
  simulations of the bit-identity probe store MEEP's word on every owned and wall cell,
  sign of zero included, at every step 0 to 20 and at every checkpoint to Table 2's N,
  and so do the six alternative-division lifts of phase 0 (`num_chunks` 4, 11, 14 and
  53 and two `chunk_layout` partitions). The only differing words are on the folded
  grids' ghost cells, where MEEP's reader returns `-0` for a stored `+0`. Counts are in
  the work package's report.
- **Tests.** `test_source_deposit.py` pins the door, every source class's deposit
  against a scalar transcription of step.cpp (the double operands and the stored
  words), the missing early return and the complex128 packing, with a must-catch
  battery of planted defects. Removed with the arithmetic they pinned:
  `test_complex_mode_bytes_are_untouched_by_the_real_field_code_path` and its control,
  which compared complex runs against the float32 deposit. The conductive lanes'
  tripwires now measure the deposit's own cells instead of the replay's partition.
  The ten raw-byte pins of `test_stepping`'s periodic and Bloch cases (each drives a CW
  point source) are re-derived by the test's own procedure. The withdraw-hoist test
  steps 30 steps instead of 6: with the double deposit its `after_step_D` null control
  first diverges at step 12 on that fixture. One id leaves
  `tools/ci/pending_certification.txt`: the real-against-complex signed zero of
  `test_real_mode_is_bit_for_bit_the_real_part_of_the_complex_run[pml_odd]` no longer
  occurs, so the test passes; the sign rule it exposed stays with the per-part complex
  arithmetic.

### The array path steps every PML form per structure chunk, as MEEP does

MEEP picks the form of every PML curl and constitutive update per structure chunk
(`sigsize`, step_db.cpp:56-59 and update_eh.cpp:162-163). The array path now does the
same, from the run's chunk division: `chunk_map.RegionPlan`, built from
`FdtdDriver.chunk_layout` (the lifted simulation's own, or the one-rank default) and the
layer, installed on `Fields` with the absorber (`Fields.region_plan`,
`Fields.install_region_plan`). A `Fields` stepped directly, with no driver, builds the
one-rank plan itself.

- **The curl** runs the split-field recurrence on the cells whose chunk carries the
  component's dsigu layer, and `f = ((kms * f) - curl) * sinv` on the field alone
  everywhere else (`stepping._apply_dsig_only_update`). That is MEEP's dsig-only form
  (step_generic.cpp:183), and its plain `f - curl` (:104) wherever the dsig coefficients
  are exactly `1.0f`. Before, the recurrence ran on the whole grid.
- **The constitutive** accumulates on the cells whose chunk carries the component's own
  layer and stores `f = src` elsewhere: `H = B`, as MEEP's H is B's storage there, and
  `E = (D - P) * chi1inv`.
- **The conductive curl** selects MEEP's four forms on the dsig and dsigu chunk
  memberships, not on `kms != 1`. A chunk that carries a layer owns cells whose own
  sigma is zero, and MEEP steps them with the in-chunk form.
- **The auxiliary contract.** `fu_*` is read and written only on the recurrence's cells,
  `f_w_*` only on the accumulation's, and `f_cond_*` only on the dsig cells. Everywhere
  else an auxiliary keeps its word. `Fields.drive_field` composes MEEP's per-chunk
  `w = f_w ? f_w : f` (update_pols.cpp:44): `f_w` in the layer's chunks, the stored E
  elsewhere. `deposit_repair.apply` takes the plain form, `field = fresh`, at a deposit
  cell outside the layer's chunks and writes no `fw` there.
- **MEEP's start of H.** The run's first PML `update_H` copies B into H and `f_w_H` on
  the dsigw cells, as MEEP's lazy allocation does (update_eh.cpp:165-177;
  `stepping.separate_h_from_b`, recorded in `Fields.h_separated`, carried by a capsule
  and cleared by `FdtdDriver.reset`). It moves words only where a magnetic current was
  deposited inside a layer whose `kms` is not 1 before that first update. Phase 0's
  P0-f measured MEEP's step-1 H word equal to the copy start on 4 of 4 such cells and to
  the zero start on 0 of 4. The deposit repair reads the start when its launch runs that
  first update.
- **Three routes, one word.** A component splits along one axis into views (`split`),
  selects on the membership line (`select`, NumPy's innermost axis), or selects on the
  full membership volume (`mask`, a division that is not separable). The route is a
  cost choice: `RegionPlan(..., route=...)` forces one, and the three store the same word
  in every array.
- **The unit-coefficient precondition** is checked when the plan is built, per line and,
  for a division that is not separable, per cell. A layer and a division that disagree
  are refused by name (`ChunkLayoutRefusal`) before anything steps. This includes the
  one-rank departure the lift section below records: a driver built without MEEP, with a
  folded periodic axis carrying a fractional high-face layer, is now refused at
  `setup_pml` (a lift takes MEEP's own division and is not).
- **Removed:** the f_u deposit mirror (`sources._bind_fu_mirror`,
  `_inject_fu_mirror`, `_unsplit_deposit_flags`, the four `bind_pml` methods and their
  `_fu_*` state, the driver's binding calls) and `PML.sigma_bites`. A deposit in a
  dsig-only strip is now damped by the field's own form, which is what the mirror
  emulated. The `fu_deposit_mirror_*` mutation needles went with the code they guarded,
  and `fu_mirror_chunk_ownership_shrunk_to_sigma_support` is renamed
  `pml_chunk_ownership_shrunk_to_sigma_support`.
- **The lift names R10 on MEEP's own state too:** a mirrored axis whose grid MEEP did
  not halve (`structure.gv` starts where the user volume does while the symmetry's
  multiplicity is above one) is refused with the `BinaryPartition` reason, ahead of the
  registration pin.
- **Stored words change** in every run with an absorbing layer. MEASURED on the four
  lifted MEEP simulations of the bit-identity probe, steps 0 to 20: every field word,
  and every auxiliary word on its own form's cells, equals a reference copy of the array
  path carrying the same per-chunk selection (prototype tokens `A1,A2,A3,B`). The three
  routes agree, word for word, in every array, and a NaN sentinel in every auxiliary
  word outside its form's cells survives. Identity with MEEP is not claimed until the
  double-precision deposit lands (it has: see the deposit section above). The byte
  pins of `test_stepping` with a layer are re-derived.
- **Certification.** The kernel tables still run the whole-grid forms and read the drive
  field as `f_w`, so 160 host tests differ by contract and are listed in
  `tools/ci/pending_certification.txt` in three groups, each with what removes it: the
  ADE drive (139, the ADE lanes), kernel host transcriptions and gate fixtures (20, the
  root and family lanes), and one real-against-complex signed zero (the per-part complex
  arithmetic). Under a PML with a susceptibility the three tables' ADE predicates now
  refuse, so those rows step on the array path. Ten Apple-GPU tests of the ADE drive
  group, measured on the device when the change was integrated, are listed after them.

### The Triton PML kernels declare the chunk record

`kernels.pml_curl_step`, `constitutive_step`, `fused_curl_constitutive_B` and
`fused_curl_constitutive_D` declare `cb` directly after their last pointer argument,
where the `launch.py` plans pass it, and do not read it yet, so those plans launch on
a device again with their outputs unchanged.

### Both NVIDIA tables read the run's chunk division

The dispatch planner now hands the run's chunk division to the Triton and the
hand-written CUDA composers, as it already did to the Metal table. No kernel source
changed and no kernel reads the division yet.

- **One division for every table.** `fastpath.plan_fast_path(..., chunk_layout=)`
  hands its `chunk_layout` to `triton_kernels.plan_step` and to
  `cuda_kernels.arms.plan_step`, the same object for both. `FdtdDriver.step` hands
  its own `FdtdDriver.chunk_layout`: the lifted simulation's division, or the
  one-rank division of a driver built without MEEP. A direct caller that hands no
  division still gets the one-rank default, and the CUDA row of its record names it.
- **The record says which division was read.** On a driver's dispatch,
  `tables.cuda.chunk_division.source` reads `one_rank` or `meep` and its record is
  the driver's `chunk_bounds_record()`.
- **Dispatch narrows on lifted divisions the record cannot describe.** A lifted
  division that is not separable is refused by name on NVIDIA GPUs as on Apple GPUs:
  by the Triton table on every slot, and by every CUDA arm. On a cylindrical run, or
  one with live off-diagonal rows, a lifted division that is not a
  `{low, interior, high}` product is refused by the Triton table for the whole plan,
  and the CUDA families that index MEEP's chunks refuse one whose record does not
  name those chunks one to one. The array path steps those runs.
  A driver built without MEEP binds the same record words as before.

### The readback is MEEP's arithmetic, word for word

Every field reading is now formed as MEEP 1.33.0 forms it, in a new module
`meep_gpu/readback.py`: `GpuRunResult.get_array`, `FdtdDriver.get_field(c,
cell_centered=True)` and `FdtdDriver.get_field_point`.

- **The centred sum is MEEP's** (array_slice.cpp:413-415): `0.25 * (((a + b) + c) + d)`
  in float32 with the `0.25` in double, `((a + b) + a) + b` for one averaged axis, axes
  in MEEP's direction order (z before r in Dcyl). 0.9.2 averaged one axis at a time,
  `0.5 * (a + b)`, a different association.
- **Neighbours are MEEP's chunk words**: a sample past a face is rebuilt from owned
  words by MEEP's connection rules (copy, negation, a `complex<float>` phase product
  formed part by part, or a metal plane's `0.0`). The package's ghost storage is never
  read.
- **B reads the owner's H where H aliases B**: in a chunk where MEEP holds H as B's
  alias (no `sigsize > 1` on the component's own axis and no integrated magnetic source
  point, update_eh.cpp:162-172 and :70-101), a not-owned B word is the owner's H word.
  After the run's first step a chunk with H of its own still holds the zero it was
  allocated with at its not-owned B points: MEEP's first B boundary pass runs on
  connections built while every chunk aliased (boundaries.cpp:350-404), rebuilt only
  after the first `update_eh` (update_eh.cpp:55, step.cpp:226). Measured on MEEP's own
  chunk words: +0 at a cylindrical PML corner where the owner holds -0 at step 1.
  `FdtdDriver.h_alias_rule()` builds the rule from `chunk_layout`, the sources and the
  step count; a B read through `Fields.to_cell_center` or `Fields.to_meep_array` with H
  stored needs it and is refused by name without it. `ChunkLayout.chunk_of` and
  `ChunkLayout.chunk_owns` are MEEP's `grid_volume::owns`.
- **The store is MEEP's**: weight, then `shift_phase * phase_shift` in double, one
  rounding per part. The Bloch duplicate plane carries `pow(eikna, -1)` in libc++'s
  arithmetic, formed from the reader's own double sum, in `Fields`; the per-axis
  `conj(bloch_phase)` complex64 correction, the metallic-plane slice and the tripwire
  in `GpuRunResult.get_array` are removed, and `Fields.to_meep_array` is now the whole
  of MEEP's convention (no duplicate on a metallic axis).
- **`get_array` takes `center=` and `size=`**: a volume as `mp.Volume` selects it,
  with MEEP's interpolation weights and collapse on an empty direction. A volume on a
  cylindrical grid is refused by name.
- **`get_field_point` is `fields::get_field`** (monitor.cpp:127-157): MEEP's
  `grid_volume::interpolate` weights (the total-weight correction, the 1e-13 cut,
  `stupidsort`, the equal-weight reset), `res += w * get_field(c, iloc)` in double in
  that order, `locate_point_in_user_volume`'s lattice translation, the mirror image and,
  on a cylindrical grid, the `r -> -r` image with the `m` parity. **A sample no chunk
  owns reads `0.0`, as in MEEP**: a point past the far face of a folded axis, or outside
  the cell, now reads `0.0` where 0.9.2 refused by name. The 0.9.2 reader differed from
  MEEP in 169,322 of 443,448 point words on MEEP's own injected words (P0-g).
- **`Grid.bloch_phase` is MEEP's `eikna`** in MEEP's arithmetic,
  `exp(I * kk * ((2*pi/a) * n))`, its sine and cosine from one fused call as MEEP's
  compiled `exp` takes them (`readback.sincos`). Against MEEP's own `eikna` read from its
  memory: 408 of 408 axes equal in double, against 247 of 408 for the 0.9.2
  `exp(2j*pi*k*L)`; the complex64 phase the stepping uses is unchanged on 408 of 408.
  The source-image deposit (`sources.py`) and the DFT lattice factors (`dft.py`) take
  the same double value; whether that moves any of their words was not measured.
- **The duplicate plane's `pow(eikna, -1)` is libc++'s** (`exp(-1 * log(eikna))`), the
  form the Apple-silicon reference build compiles. A MEEP built against libstdc++ takes
  its own `pow(complex, int)` overload, so the plane is not claimed against a Linux
  MEEP.
- **The lift's D conductivity read is taken at 1e8 and 1e20** in place of 0.9 and 0.3,
  far above every sigma, where MEEP's tensor inversion leaves no round-off: a cell whose
  sigma is 0 reads 0, not +-5e-17. Word-affecting on conductive lifts only, and only on
  those cells.
- **Removed**: `Fields._wrapped_neighbour`, `Fields._reconstruct_full_domain`,
  `Fields._add_boundary_cells`, `Fields._drop_invariant_axes`,
  `Fields._collapse_empty_direction`, and the driver's `_point_field_reader`,
  `_build_point_field_reader` and `_point_stencil_along`; the readback is their one home.
  A capsule leaves the readback's memos out (`_point_stencils` and `_h_alias_cache`, in
  place of `_point_readers`) and records `readback.py` among the modules that produce
  lifted state, since the Bloch phase is formed there.
- **Measured** on MEEP 1.33.0 single precision built without contraction (Apple
  silicon), all twelve components, against MEEP's own injected words and the phase-0
  prototype's stepped words: 0 differing words of every whole-cell array, stored
  centre, volume and point read on 15 Cartesian and 4 cylindrical cases. Identity under
  a Bloch phase waits on MEEP's owned-window storage on such an axis.

### The array path forms the curl and the PML constitutive update in MEEP's order

- **The curl** is MEEP's C expression left to right, `dtdx * (((g1s - g1) + g2) - g2s)`
  (`step_generic.cpp:104`), where it was `dtdx * ((g1s - g1) + (g2 - g2s))`. A partner
  whose derivative axis is invariant (z at 2-D, x and y at 1-D, phi under cylindrical
  coordinates) is absent from MEEP's step plan (`fields.cpp:428-453`), and the curl takes
  MEEP's form for it: `dtdx * (g1s - g1)`, or `(-dtdx) * (g2s - g2)` with the partners
  swapped (`:72-76`, `:108`). `stepping.curl_presence` decides it and
  `stepping._curl_from_operands` takes it as a required argument; the absent partner's
  shifted operand is no longer formed except under BFAST, which sums both partners, and
  the `curl_partial` scratch buffer is gone.
- **The PML constitutive update** forms MEEP's right-hand side first,
  `f + ((kps * fw) - (kms * fw_prev))` (`:697`), where it was
  `(f + kps * fw) - kms * fw_prev`, in `stepping._apply_constitutive_pml` and in
  `deposit_repair.apply`, which recomputes it at deposit cells.
- **The Gaussian envelope's argument** is MEEP's, `-tt * tt / (2 * width * width)`
  (`sources.cpp:106`), where it was `-0.5 * (tt / width) ** 2`.
- **Stored words change** wherever these round. MEASURED on the four lifted MEEP
  simulations of the bit-identity probe ((a) and (b) 2-D, (c) and (c') 3-D with two
  mirror planes), steps 0 to 20: every array the NumPy driver holds equals, word for
  word, a reference copy of the array path carrying the same three changes (0 of
  26,880,000, 26,880,000, 8,963,136 and 8,963,136 words differ), and differs from the
  previous array path in 38,094, 35,125, 295,620 and 293,622 of them. Identity with MEEP
  is not claimed by these changes alone; the per-chunk region forms and the
  double-precision deposit are still to come. Cylindrical, beta and BFAST runs take the
  same bracket with their folded increments still inside the curl, and no identity is
  claimed for them until those increments are aligned.
- **A mirror fold now reproduces the full domain to rounding, not bit for bit.** MEEP's
  bracket is not reflection-symmetric in rounding: across a plane normal to the second
  partner's axis the image cell forms `(a - g2s) + g2` where the stored cell forms
  `(a + g2) - g2s`, so the full domain's two halves differ by rounding and a folded run
  differs from its quadrant by the same. The fold-equivalence tests are held to
  `test_stepping.FOLD_ROUNDING_BOUND`, 64 units of 2**-24 of their scale, wherever the
  arithmetic rounds; the integer-seeded linear fold stays at zero. MEASURED over the 9
  combination, 5 chi3, 52 driver plumbing, 3 live-far-face, 3 graded-medium, 3 tensor, 3
  DFT, 2 folded-PML and 1 odd-count cases: at most 15.3 units. The plumbing and
  combination cases folded with every plane phase inverted move by at least 5.7e6 units,
  and a flux plane a tenth of a cell off its mirror position by at least 9.5e-03
  (1.6e5 units). Two new tests
  (`test_the_fold_rounding_bound_rejects_a_wrong_mirror_phase`,
  `test_the_driver_fold_bound_rejects_a_wrong_mirror_phase`) and a control inside the
  flux-plane test pin that. Four tests whose names claimed exactness are renamed to say
  "to rounding".
- **Byte pins re-derived** in `meep_gpu/test_stepping.py` by each test's own procedure:
  the complex core digest, the even X/Y fold and readback digests, and the ten periodic
  byte pins.
- **Callers outside the array path**: `parity/meep_gpu/gate_metal_no_pml_curl.py` passes
  the curl presence; the intermediate census of `parity/meep_gpu/gate_metal_fused_hd_pair.py`
  models the new chains (its subnormal keys for the curl and the constitutive tail are
  renamed with them); the array-path oracle tests plant the pairwise grouping as their
  arithmetic change.
- **Certification**: the array-path oracle moves in 18 of 18 fixture cases (fingerprint
  `8f01eeb0…` against the ledgers' `416194ea…`), so the three
  `test_<table>_ledger_records_the_live_array_path_oracle` tests fail by name, as they
  are meant to. They are on the pending-certification list with the kernel host tests
  and harness hand references that compare a transcription of the previous groupings
  with the array path, 415 entries in all: the 3 oracle tests, 319 CUDA, 74 Triton and
  3 Metal host tests, 9 harness tests, and 7 CUDA tests that read `stepping.py` at the
  line numbers a kernel module cites (the statements are unchanged; their lines moved).
  A further 103 Metal tests that skip without an Apple GPU join them, measured on one
  (M1 Max): byte identity of a Metal kernel or a Metal-dispatched run with the array path,
  including 28 held-residency and 1 entry-point comparison against the reference driver.
  They leave it as the kernel tables and the hand references are re-transcribed and the
  ledgers re-minted. Drift on these bytes: Triton 256 -> 292 of 592 raw-tier and
  171 -> 204 of 444 code-tier digests, Metal 95 -> 100 of 441 pinned paths and 92 -> 97
  of 352 code digests, CUDA 69 of 269 unchanged.

### The Metal table reads the chunk division

The chunk division a run steps with now reaches the Metal table's composer, and the
kernels have one way to read it. No kernel source changed and no kernel reads it yet;
the templates take it up when they select PML forms per chunk.

- **`meep_gpu/metal_kernels/chunk_bounds.py`**: the `ChunkBounds` struct, whose 25
  members are the words of `chunk_map.chunk_bounds_record` in record order (24 `int`
  bounds, `cbi_*` on the integer lattice and `cbh_*` on the half-integer one, then the
  `uint` presence word; 100 bytes, no padding), so the flat `int32[25]` record binds as
  `constant ChunkBounds&` or embeds in a `Params` as `ChunkBounds cb`. It also holds the
  record's validation (`words`), its packing and host decode, the plan-owned tensor, and
  one anchored emitter for the bounds block: the locals, `in_i_<axis>` / `in_h_<axis>`
  membership as `(i >= a0 && i < a1) || (i >= b0 && i < b1)`, and the presence bits.
- **The division travels with the freeze**: `FdtdDriver.step` hands `chunk_layout`
  to `plan_fast_path`, which passes it untouched to the table ladder;
  `metal_kernels.launch.plan_step(..., chunk_layout=)` resolves `None` to the one-rank
  division and puts the layout and its validated record on `arms.StepContext`.
- **Refused by name on the Metal table**: a division the per-axis record cannot
  describe (not separable along some axis and lattice, or a line with more than two
  runs), at rung 4bM of `metal_dispatch.decide` for the whole table and in
  `plan_step` on every slot; and a layout built for another grid. The array path
  steps such a division. The dispatch record carries a `chunk_division` block (the
  layout's source, its chunk count, the record and its runs, or the refusal).
- **`coverage.lih_product_reasons`**: the precondition for kernels that need the
  owning chunk and not only its layer flag (off-diagonal presence, cylindrical prefix
  segments). Not yet consulted by any arm.
- **Certification**: kernel-source digests 0 of 36 moved; the host digests of
  `launch.py`, `coverage.py` and `arms.py` moved and `chunk_bounds.py` is new, so
  `test_metal_kernels::test_the_checked_in_fingerprints_match_the_tree` is on the
  pending-certification list until the Metal fingerprints are re-cut and its welds
  re-gated.

### Every Triton PML plan binds the chunk record

The Triton table's host side now carries the chunk division to its kernels as one
`int32` record, `chunk_map.chunk_bounds_record`, bound once per plan as a single
pointer argument, `cb`. No kernel source and no array-path module changes: the record
is bound and not yet read.

- **The plans.** `launch.PmlCurlPlan`, `ConstitutivePlan` and `FusedPairPlan`, the
  dispersive `update_E` plan (`DispersiveConstitutivePlan`) and the scratch-weld plan
  base (`ScratchWeldPairPlan`, and through it its eleven subclasses) take a required
  `chunk_bounds` and keep its read-only host copy as `plan.chunk_bounds`.
  `launch.bind_chunk_bounds` checks it against the plan's extent and refuses by name a
  record that is missing, is not 25 integers that fit `int32`, holds a run outside the
  plan's extent or out of order, or carries a presence word that disagrees with the
  plan's invariant axes. The three `launch.py` plans pass it as `cb` directly after
  their last pointer argument, where the `kernels.py` entry points they launch declare
  it (see "The Triton PML kernels declare the chunk record").
- **The engine route** takes the record from `launch.chunk_bounds_for` alone: the
  division handed over, or `ChunkLayout.one_rank` of the grid and layer. `plan_step`,
  `plan_pml_curl`, `plan_constitutive`, `plan_fused_pair`,
  `plan_dispersive_constitutive` and `plan_folded_dispersive_constitutive` take a
  `chunk_layout` argument (default `None`, the one-rank division). `plan_step` judges a
  division handed over before any arm, with `coverage.chunk_layout_reasons`: a division
  for another grid, one that is not separable, and, on a cylindrical run or one with
  live off-diagonal rows, one that is not a `{low, interior, high}` product along the
  axes those kernels index, leave every slot on the array path with the reason named.
  The dispatcher hands the driver's division over (see "Both NVIDIA tables read the
  run's chunk division").
- **The harness route has no default.** `plan_from_arrays`,
  `plan_constitutive_from_arrays`, `plan_fused_pair_from_arrays`,
  `plan_dispersive_constitutive_from_arrays` and the scratch-weld products'
  `*_from_arrays` builders require `chunk_bounds`. The gates and probes state their
  division through `parity/meep_gpu/gate_chunk_records.py`: `layer_record` where the
  reference is the array path on an engine layer, `every_cell_record` where it is a
  hand reference that steps the split-field recurrence on every cell.

### The hand-written CUDA table reads the chunk division

The hand-written CUDA table now takes the run's chunk division as an input, as one
`int32[25]` record per composition. No kernel reads the record yet.

- **The division.** `cuda_kernels.arms.plan_step(..., chunk_layout=)` takes a
  `ChunkLayout`. With none, it reads the one-rank division of the grid and PML:
  the default for a driver built without MEEP, and the division a default lift
  steps. The dispatch planner passes the driver's own division (see "Both NVIDIA
  tables read the run's chunk division").
- **The binding.** The record is uploaded once per composition
  (`StepContext.chunk_bounds_device`). Every slot plan and fused product exposes
  that same vector as `chunk_bounds`.
- **Refused by name, to the array path.** Every CUDA arm and fused product refuses
  a division that no record describes: one that is not separable, or a line with
  more than two runs. The off-diagonal families (every axis) and the cylindrical
  curls (along r) index MEEP's chunks, so they also refuse a division whose record
  does not name those chunks one to one. A layer on a wrapping axis is one such
  case: its stored cell 0 joins the low run while the high chunk owns it. That case
  changes dispatch today: an off-diagonal `update_E` with a layer on a periodic
  axis is no longer served by the CUDA table.
- **The record.** The dispatch record's CUDA row carries the division read
  (`tables.cuda.chunk_division`). The clause lives in the new import-free module
  `cuda_kernels/chunk_coverage.py`.

### Lifted-state capsules

`meep_gpu.capsule` saves a lifted driver and loads it back with no MEEP: the
constructor and configuration calls the lift made, replayed through the driver's
own code, and every array and scalar of the state written over the rebuilt one as
bytes and IEEE-754 bit patterns. A recorded currents table can be injected into the
loaded sources. Monitors, unrecorded callables, other schema versions and members
that fail their digest are refused by name. `parity/meep_gpu/capsule_identity.py`
checks a fresh lift against its loaded capsule word for word
([Lifted-state capsules](docs/development/testing.md#lifted-state-capsules)).

### The reference MEEP is an identity reference, judged on its machine code

The array path is being aligned with MEEP 1.33.0 word for word, and the MEEP that claim
is made against has to round every operation on its own. The build scripts now build
such a MEEP and refuse to finish unless it is one; a host checks the one it runs.

- **`tools/fma_census.py`**: the fused multiply-add census of MEEP images (`otool` on
  Mach-O, `objdump` on ELF, or a disassembly made on another host): arm64 `fmadd`,
  `fmsub`, `fnmadd`, `fnmsub`, `fmla`, `fmls` (and the widening and SVE forms) and x86-64
  `vfmadd*`, `vfmsub*`, `vfnmadd*`, `vfnmsub*`, per function, with MEEP's 16 field-update
  kernels reported by name; the compiler-rt complex-division builtins are exempt and
  counted. `--require-none` exits 2 on any other fused instruction. MEASURED on an M1 Max:
  the contraction-off build of 2026-10-05 has 0 outside the builtins (18 inside) in
  `libmeep` and `_meep`; the MEEP the 0.9.x Metal kernels were certified against has 1,239
  (1,235 in `libmeep`, 634 of them in the field-update kernels, and 4 in `_meep`).
- **The build scripts**: `build_meep_133_macos.sh` compiles with
  `-O2 -ffp-contract=off -fno-fast-math` unless `CFLAGS` or `CXXFLAGS` is set (clang
  contracts by default); `build_meep_133_linux.sh` sets none and refuses flags under which
  `-march` is not `x86-64`. Both run the census at step 8 (`--require-none`), keep the
  build record at step 9 (`build_meep_133_linux.sh` kept none before) and run the digest
  check at step 10. `MEEP_BUILD_ROLE=timing-control`, which `build_meep_133_native.sh`
  now passes, records the census without judging it and skips the digest check.
- **`parity/meep_gpu/build_meep_133_identity.sh`**: runs the platform script under
  `env -i` (only `HOME`, `USER`, `LOGNAME`, `TMPDIR`, `TERM`, `LANG` and the scripts' own
  knobs pass; `PATH` is the system's) with `PYTHONDONTWRITEBYTECODE=1`. `DRY_RUN=1` prints
  the plan.
- **The build record** (`timing_records.py meep-build-record`) gains `fp_contract`, the
  census of the loaded images (`fma_census`), the host's math library (`libm`: the UUID of
  `libsystem_m` on macOS; on Linux the glibc version, the sha256 of `libm`, the CPU
  features glibc chooses its implementations by and `GLIBC_TUNABLES`) and the subnormal
  handling after `import meep`. `--require-identity` exits 4 unless the census is clean
  and the math library was read; a recorder that stops under `--require-single` or
  `--require-identity` now exits 4 instead of 0.
- **`parity/meep_gpu/meep_identity_smoke.py`**: steps MEEP alone on four cases (the
  2026-10-05 probe's) at steps 0, 1, 2, 5, 10, 20 and N, and compares the SHA-256 of every
  stored word with `meep_identity_smoke.json`, with a planted 1-ulp flip as its control.
  The record was minted from the contraction-off build's own dumps: 8,800,728 of 8,800,728
  words equal, every digest reproduced by a second process. A fresh run on that build reads
  324 of 324 digests equal; on the 0.9.x certified MEEP, 276 of 324.
- **`tools/check_install.py --require-identity-reference keep|flush`** (and
  `--identity-build-record PATH`): judged first, in a process that imports NumPy and MEEP
  only, and the check stops on it: MEEP 1.33.0 single precision with MPI; a build record
  that describes the loaded images and holds a clean census; the census re-run clean; the
  measured subnormal handling (float32 and float64) after MEEP's switch equal to the mode,
  with `OMP_NUM_THREADS=1` for `flush` and no `flush` reference on Apple silicon; the math
  library the record pins. INSTALL.md, "The MEEP identity reference".

### The lift steps MEEP's own chunk division

MEEP chooses the PML form of every curl and constitutive update per structure chunk
(`sigsize`, step_db.cpp:56-59 and update_eh.cpp:162-163), so the form a cell steps
with is a property of the chunk division. The division is now data, read from the
lifted simulation, in a new module `meep_gpu/chunk_map.py`.

- **`ChunkLayout`**: each chunk's box in MEEP's doubled coordinates and its per-axis
  `sigsize > 1` flags. `ChunkLayout.from_meep` reads `structure.get_chunk_volumes()`
  after the lift's `init_sim` and transcribes `structure_chunk::use_pml`
  (structure.cpp:625-652) for the flags; `ChunkLayout.one_rank` is the default for a
  driver built without MEEP. Derived from it: a membership line per (axis, Yee
  sub-lattice), with ghost rows reading the row they image, or `None` where the
  division is not separable; the full mask; the chunk cuts; the kernels' `int32[25]`
  record (`chunk_bounds_record`); and `RegionPlan`, the runs and evaluation mode per
  component, with the unit-coefficient precondition checked. No kernel reads the record
  yet; the array path dispatches on the plan (the per-chunk forms, above).
- **`FdtdDriver.set_chunk_layout`**, **`FdtdDriver.chunk_layout`** and
  **`FdtdDriver.chunk_bounds_record`**: the lift installs MEEP's division before the
  absorber; a layout for another grid or another absorber is refused by name, and so
  is a change after the first step.
- **`PML.in_pml_chunk` is removed.** Its rule moved verbatim to
  `chunk_map.one_rank_line`, the one-rank default's membership. Its one consumer at the
  time, the f_u deposit mirror, read the driver's division; the mirror is removed with
  the per-chunk forms (above).
- **Stepped words are unchanged on default divisions**: on the four probe cases and on
  four more whose f_u mirror is live, 0 words of any collected array differ at steps
  0, 20 and Table 2's last step, and every array's SHA-256 is equal at every other
  step 1 to 19 and every checkpoint; `one_rank` equals the rule it replaced on 6,732
  of 6,732 lines (1,122 grids and face tables). They change where the lift now follows
  MEEP and the one-rank rule did not: on a folded periodic axis with a high-face layer
  the rule counted from the stored end and missed one row per lattice. Measured on
  such a lift (8 x 8 cell, Mirror(Y), `k_point = 0`, PML(1), CW Ez line on that row):
  relative L2 against CPU MEEP 8.91e-01 before, 7.43e-07 now. Drivers built without
  MEEP keep the one-rank rule, departures included.
- **Refused by name**: a `BinaryPartition` `chunk_layout` together with mirror
  symmetry. Given one, MEEP does not halve its grid (structure.cpp:109): it steps the
  whole cell with the mirrors unused, while the lift would fold it. A `chunk_layout`
  given as a Simulation or a file is loaded after the halving and lifts.
- **A division that is not separable** (a forced `num_chunks` cutting a PML chunk over
  part of the cell) is lifted with a one-line notice beginning `meep_gpu: the lifted
  chunk division is not separable along`.

### Every kernel ledger records the array path it was certified against

- `meep_gpu/array_path_oracle.py` steps a fixed fixture set (18 seeded real-storage
  cases, six steps each, NumPy) on the array path and fingerprints the float32
  words. The fixture calls no libm function while it steps, evaluates no source
  envelope (its sources read stored current words), forms no host-dependent complex
  product and keeps subnormals, so the fingerprint describes the array path and not
  the host: the same value on NumPy 2.2.6, 2.4.1, 2.4.3, 2.5.1 and 2.5.3 on one
  Apple M1 Max, measured. Complex storage, Bloch phases and cylindrical m != 0 are
  deferred to a later fixture set, by name.
- The three kernel ledgers carry it under `array_path_oracle`: in every entry that
  pins source bytes (54 CUDA, 66 Triton, 62 Metal) and, with the per-case digests,
  the form-class counts and the array-path module code digests, in each
  `driver_dispatch` record. `parity/meep_gpu/record_array_path_oracle.py` is the only
  writer; it refuses a fixture that leaves a declared form class without a cell, and
  any write that would move more than the leaves. `--verify` recomputes and compares.
- Each weld contract gains `test_the_<table>_ledger_records_the_live_array_path_oracle`:
  an arithmetic change to the array path fails it by name on all three tables; a
  comment edit does not. `meep_gpu/weld_record_walk.py` classifies the new digests
  under two named rules.

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
