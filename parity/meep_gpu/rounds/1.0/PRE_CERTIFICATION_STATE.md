# Pre-certification state of the 1.0 series

The state of the 1.0 series at the end of its development, as the final pre-certification pass
measured it. The certification round (`RUNBOOK.md` in this directory) starts from it. Nothing
here is a certification: the pass ran every leg of the round's leg table (`legs.py`) through the
round's own step code (`run_lane.py`, the Triton fleet driver's `run_one`, `metal_round.py`),
wrote no ledger, fingerprint, census or record, and its verdicts are what the round's gates
print. Every number is N / D with D named.

## 1. Series and trees

- The series: 0.9.2 (`fa59c2f`) plus 115 entries, `01` to `115-LHF`, the series list's sha256
  `5bb2fbbd82924db6a3307cdd7f61c16565639853d3af8279af10f6f497684cf2`; tree
  `87febcc85b4d85706b35baa507035f31b7e0bb7a`. This entry adds this file and one `CHANGELOG.md`
  section and changes no other byte.
- Three trees carry verdicts, and every leg is stated on the tree it ran on (section 9).
  **`87febcc8`**, entries 01 to 115, the tree this entry is cut on: 10 NVIDIA legs ran on it.
  **`15da8cd4`** (`15da8cd4d3fba035cd26660e177db8429df12882`), entries 01 to 114: every Metal
  leg (68) and 2 NVIDIA legs ran on it. **`b676c5f5`**
  (`b676c5f53af71cf6056471f84d17c28d25fae114`), entries 01 to 111: 249 NVIDIA legs ran on it. A
  leg that ran on an earlier tree is reused on `87febcc8` only because every file it binds
  (every source its artifact digests, its gate module and subjects, and the round drivers that
  ran it) is byte-identical on `87febcc8`: 14,400 / 14,400 bound-file hashes over the 251 reused
  NVIDIA legs and 6,490 / 6,490 over the 68 Metal legs.
- Entries 112 to 114 changed 31 files; one is bound by an NVIDIA leg,
  `meep_gpu/cuda_kernels/test_ade_update_p.py`, a subject of the two CUDA ADE re-cut legs, which
  re-ran on `15da8cd4`. Entry 115 changed 6 files, none of them Python (`CHANGELOG.md`, two
  pages under `docs/development/`, `RUNBOOK.md` and the two corpus overlay pins in this
  directory), and no leg binds any of them. The 10 composition-shaped Triton legs record no
  digests by design, so no byte check can license them: they re-ran on `15da8cd4` and again on
  `87febcc8`.
- Every Metal gate artifact of this pass digests `meep_gpu/subnormal_policy.py`, which entry 111
  moved (62 / 62), and 61 / 62 also digest `meep_gpu/metal_kernels/launch.py`, which entry 113
  moved. So every Metal leg's subject moved after entry 106 and the whole Metal table re-ran; a
  stability sample of an unchanged rest would have been empty. The route legs of entry 113's own
  campaign (tree `82524592`) were not reused: of the 142 to 144 sources each digests,
  `meep_gpu/subnormal_policy.py` differs on `15da8cd4`.
- **Landed after this pass: `117-XR`** (work package XR, the deleted expansion licence's residue in
  the harness), text and dead code only. No file under `meep_gpu/` changes and no ledger, needle,
  mutation anchor, pending-list id, predicate, kernel, plan or gate decision moves; 26 harness
  modules under `parity/meep_gpu/` change, 10 in comments and docstrings only and 16 in recorded
  strings and dead definitions. The verdicts in sections 2 to 9 were measured on the bytes before
  it, and the CERT-1 bind must re-pin these: 30 `code_sha256` / `source_sha256` pins in 14 ledger
  entries, every one already drifted before this entry (CUDA and Metal `driver_dispatch`; the Metal
  device gates `metal_complex_conductive_fused_pair`, `metal_complex_conductive_pml`,
  `metal_complex_no_pml_conductive`, `metal_complex_no_pml_curl`,
  `metal_cylindrical_complex_fused_hd_pair`, `metal_cylindrical_real`,
  `metal_cylindrical_real_fused_magnetic_pair` and `metal_whole_step`; the Triton device gates
  `triton_complex`, `triton_complex_fused_ade_chain`, `triton_cylindrical` and
  `triton_unified_expansion`, which the bind retires instead of re-pinning: see the XF item
  below),
  plus every gate artifact's source map that digests one of the 26
  files, and the coverage board, whose outcome subject it moves (`build_parity_board.py check`
  refuses on this entry's tree until the board is re-cut). Measured with and without it on entries
  01 to 114 (Apple M1 Max): package suite 221 F / 14,751 P / 851 S on both; harness suite
  0 / 1,925 / 20 on both, 1,945 of 1,945 ids the same; the 92 device-visible files on the Apple GPU
  13 / 4,016 / 331 on both, 4,360 of 4,360 the same; `check_failures.py` 62 of 62 failures declared,
  0 undeclared; needles 0 differences; 200 mutation anchors, 0 stale; census child 15 of 15 records
  identical.
- **Landed after this pass: `118-ENVL`** (work package ENVL): the certification reference
  environments carry the corpus libraries themselves. shapely 2.1.2 and nlopt 2.11.0 become pins
  of both reference pip locks (osx-arm64 13 to 15 pins, linux-64 22 to 24), the PyMieScatt 1.8.1.1
  stand-in is built from its PyPI sdist (checked by SHA-256) and a two-line patch by this
  directory's `build_pymiescatt_standin.py`, and both build scripts install and check them and
  link a host `ffmpeg` into the prefix (their step 7b). Entry 115's two corpus overlay requirement
  files (sections 1 and 6) are deleted, and `tools/compare_reference_environment.py` reports the
  corpus-only packages apart from its verdict and gains `--require-pip-lock`. Measured by the work
  package on entries 01 to 115 plus it: both live reference environments hold all three as
  additions only (modified 0 / 43,623 prefix files on the Mac and 0 / 46,510 on Linux, removed 0,
  conda lists byte-identical, every other distribution at its version; `--require-pip-lock`
  16 / 16 and 25 / 25), and with no overlay and no extra `PATH` entry the rows that need them plus
  one control (four rows per host) hold with 0 differing words in both subnormal modes on both
  hosts, 16 / 16 row-modes, each row's digests, planes and currents equal to Table 2's Tier B
  record and its subnormal stamp equal to the single stamp of its reference's other 186 rows.
  Table 2's JSON is unchanged; its page changes only in its provenance paragraph. It binds no
  leg: none of its 20 files is among the 338 bound paths of the pass's 283 recorded leg bindings,
  a key of any of the 1,167 digest maps in the pass's artifacts (Metal `15da8cd4`, NVIDIA
  `b676c5f5` and the two NVIDIA re-run sets) or of the 588 in the tree's 50 JSON records (the four
  kernel ledgers among them), or a subject of the three censuses of record, so no verdict in
  sections 2 to 9 moves. Five of its files are top-level `parity/meep_gpu` files, in the coverage
  board's outcome subject, which entry 117 already moved; the board re-cut owed there covers it.
- **Landed after this pass: `119-XF`** (work package XF, the pre-freeze fixes), harness and one
  package test, no shipped module: the four behaviour items entry 117 left, and the planner's
  prices.
  The CUDA board's three emitted-text cells are determined
  (`fused_update_P_no_pml_complex[_uniform]` POINTWISE, both complex off-diagonal electric
  updates STENCIL; each was UNDETERMINED by `TypeError`); `probe_metal_planner_composition.py`
  sweeps 110 / 110 configurations, VERDICT FAIL on the four cylindrical complex rows of section 8;
  `rebind_triton_welds.py` retires `triton_unified_expansion_device_gate` by name at the bind
  (section 8), and `test_triton_weld_contract.py`'s two floors that count it drop by one; the
  special-kz gate's mutant header loses its two unused arm constants; `legs.py` prices the Triton
  fleet and hand legs at this pass's measured walls (CERT-3 NVIDIA on 4 GPUs: 17,195 s to
  23,834 s). For the round: its bind (`RUNBOOK.md`, between CERT-3 and CERT-4) re-pins no XF
  file (none of the four kernel ledgers' 2,270 `code_sha256` / `source_sha256` pins names one,
  and no ledger changes) and retires `triton_unified_expansion_device_gate` by tool, never by
  hand (its GO wants the report's `RETIRED` line and the rebound ledger without the entry),
  which removes 41 lines of `meep_gpu/triton_kernels/fingerprints.json`, a file no pin binds.
  Of the pass's recorded bindings only two XF files are digested: `rounds/1.0/legs.py`, a
  round driver every NVIDIA leg binds (261 / 261 legs of the FPC2N set, 566 digest hits with
  the re-run sets), and
  `gate_triton_special_kz.py` (the two `special_kz` fleet legs, 12 hits). So no NVIDIA leg of
  sections 2 and 9 is byte-reusable on a tree carrying XF; CERT-3 re-runs every one. No Metal leg
  binds an XF file. Six XF files are in the coverage board's outcome subject (and the pending
  list in its board subject); the re-cut owed since entry 117 covers them. Measured on entries 01
  to 118 with XF (Apple M1 Max), against the controls of entries 117 and 118 (work packages A16
  and A17): package suite (CPU-only shim, every marker) 221 F / 14,751 P / 851 S, 15,823 of
  15,823 ids the same outcome; harness suite 0 / 1,978 / 20, 1,989 of 1,989 the same and 9 new,
  all passing; the 92 device-visible files on the Apple GPU 13 / 4,016 / 331, 4,360 of 4,360 the
  same; `check_failures.py` 62 of 17,538 failures, 62 / 62 declared, 0 undeclared; needles 0
  differences; 200 mutation anchors, 0 stale; `legs.py check` exactly the two route-stamp problems
  (i-1) on both trees. The bind's retirement, simulated on a copy of this tree whose Triton ledger
  `rebind_triton_welds.py --write` produced: package 15,823 of 15,823 and harness 1,998 of 1,998
  ids the same outcome as on the tree itself, and `check_failures.py` 0 undeclared.

## 2. Verdict

**The bar is met: 0 class (ii) regressions and 0 class (iii) harness items on all three
tables.** Every non-release is class (i), an item the certification round itself resolves.

| Legs | Released / D | Ran on |
|---|---|---|
| CUDA gates, keep and flush | 118 / 118 | 116 ran on `b676c5f5`, reused; 2 ran on `15da8cd4`, reused |
| CUDA route legs | 6 / 7 | 7 ran on `b676c5f5`, reused |
| Triton fleet, keep | 61 / 61 | 56 ran on `b676c5f5`, reused; 5 ran on `87febcc8` |
| Triton fleet, flush | 61 / 61 | 56 ran on `b676c5f5`, reused; 5 ran on `87febcc8` |
| Triton stencil welds and bit identity, keep and flush | 6 / 6 | 6 ran on `b676c5f5`, reused |
| Triton route legs | 3 / 4 | 4 ran on `b676c5f5`, reused |
| End-to-end licence (preflight, ship, null, verdict) | 4 / 4 | 4 ran on `b676c5f5`, reused |
| Metal fleet, flush | 61 / 62 | 62 ran on `15da8cd4`, reused |
| Metal environment records (start, end) | 2 / 2 rc 0 | 2 ran on `15da8cd4`, reused |
| Metal held route legs | 4 / 4 | 4 ran on `15da8cd4`, reused |

`metal_round.py status` on `15da8cd4`: 61 / 62 Metal gate artifacts released; the welds the
Metal table cites released 45 / 45 (before the bind).

Class (i), each with the step that resolves it:

- **i-1, the route constants carry 0.9.2 stamps** (CERT-1 moves them). `legs.py check` refuses
  on exactly two problems, the NVIDIA constants `fastpath.DRIVER_ROUTE_FUSED_GATE` and
  `fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE`; the Metal held route ran under `metal_round.py
  route --dry-run-allow` because `metal_dispatch.METAL_DRIVER_ROUTE_GATE` still names
  `2026-10-05`. No gate verdict depends on the stamp.
- **i-2, CUDA `cuda_flush`** (every case passes, 39 of 39: 38 PASS-POLICY-REFUSED, 1
  PASS-NO-FUSED-ARM; 0 divergent cases; ran on `b676c5f5`, reused on `87febcc8` by the byte
  check). It cannot release before the bind writes per-arm flush evidence from the flush fleet
  (CERT-3, bind, CERT-4): its envelope control and the dispatched-case clause need a dispatch,
  and no arm dispatches under flush without that evidence.
- **i-3, Triton `flush`** (every case passes, 37 of 37 PASS-POLICY-REFUSED; 0 divergent cases;
  ran on `b676c5f5`, reused on `87febcc8` by the byte check). Same reason.
- **i-4, `gate_metal_spine_byte_neutrality`**: not released: "A CERTIFIED KERNEL SOURCE MOVED"
  against the frozen round input `results/metal_pml_2026-08-14/provenance.json`; the series
  moves the tranche-1 PML kernel sources by design (first at entry 18). **Owner ruling
  pending**: re-baseline its frozen artifact at CERT-1 (recommended) or retire the gate and its
  weld (`metal_spine_byte_neutrality_device_gate`). Until it is ruled it is class (i).
- The other items the certification round resolves produce no non-release in this pass, because
  no gate verdict reads them: the `_mixed_policy` declarations of `triton_no_pml_device_gate`
  and `triton_cylindrical_device_gate`, retired at the bind; the Triton census re-cut without
  the retired `@unified_expansion` keys (CERT-4). The Tier A flush set is entry 111 and is in
  every tree named here, so no class (i) is claimed for it.

Behaviour, not hashes, is the acceptance: a gate, fleet, stencil or bit-identity leg releases
only through its own bit-identity clauses over every product leg (0 differing words by its own
verdict); for the route legs the per-case first divergent checkpoint was read directly.

| Metal route leg | Released | Cases | Verdicts | Divergent cases | Controls (failures) |
|---|---|---|---|---|---|
| `band_witness` | True | - | - | 0 | 42 (0) |
| `harness_keep` | True | 44 | PASS-POLICY-REFUSED 44 | 0 | 85 (0) |
| `shipped` | True | 44 | PASS-ENGINE-SMOKE 2, PASS-FUSED 42 | 0 | 326 (0) |
| `harness_flush` | True | 44 | PASS-ENGINE-SMOKE 2, PASS-FUSED 42 | 0 | 326 (0) |

## 3. The blockers of the previous pass, closed

The pass on entries 01 to 106 (tree `c2d533b0`) did not meet the bar. Each of its class (ii) and
(iii) items is closed by an entry of this series and re-measured above:

| Item | What it was | Closed by | Re-measured here |
|---|---|---|---|
| A | entry 89 left the deleted expansion licence half-removed in gate code (CUDA complex pairs `KeyError('arms')` on 32 legs; Metal complex mutant maps and `plan.expansion`; Triton `complex_ade` needle, `complex_no_pml_conductive` null count) | 107 (Metal), 108 (CUDA, Triton) | CUDA 118 / 118, Triton keep 61 / 61, the three Metal complex gates released |
| B | Metal route `conductive_2d` did not fuse (first bad entry 37) | 113: per-chunk conductivity presence words | Metal `shipped`, `harness_flush` |
| C | Metal route singles refused on three 2-D magnetic cases (first bad entry 18) | 113: the first `update_H` takes the layer | same |
| D | Metal subnormal-cliff control outside the band on four cases | 113: the control composes the driver's chunk division and absent clear; a band overrun is licensed only by a word-for-word host flush replay | same |
| E | the flush host-import wrapper imported CuPy before the gate set `CUPY_ACCELERATORS` (three ADE chains) | 108 | Triton flush 61 / 61 |
| F | Triton cylindrical under flush: two `div.full.f32` not flush-uniform | 109 (they were in a planted mutant, not a shipped kernel) | same |
| G | three Triton gates installed keep under the flush campaign | 108 | same |
| H | the spine byte-neutrality gate's frozen baseline | not closed: class (i), ruling pending (section 2, i-4) | - |
| I | eight Triton flush calibrations vacuous under flush | 108 (declared by `flush_vacuity.py`) | same |

## 4. The board (entry 114)

Built by `parity/meep_gpu/build_parity_board.py` at 2026-10-11T02:38:07Z on subject
`303d0c66046ed4ae` (1117 files). Generated: do not edit; re-run `build`.

| Table | Rows | (a) B select | (b) MEEP-order A | (c) host identical | (d) device keep | (d) device flush | (d-dev) keep, NOT certification | (d-dev) flush, NOT certification | (e) CERTIFIED keep | (e) CERTIFIED flush |
|---|---|---|---|---|---|---|---|---|---|---|
| CUDA | 75 | 55 / 55 (+1 refused) | 57 / 57 (+8 refused) | 18 / 18 (+47 refused) | 0 / 53 (+12 refused) | 0 / 53 (+12 refused) | 4 / 4 (+61 refused) | 0 / 0 (+65 refused) | 0 / 53 (+12 refused) | 0 / 53 (+12 refused) |
| Triton | 73 | 55 / 56 (+2 refused) | 61 / 61 (+7 refused) | 26 / 29 (+39 refused) | 0 / 55 (+13 refused) | 0 / 55 (+13 refused) | 2 / 2 (+66 refused) | 0 / 0 (+68 refused) | 0 / 55 (+13 refused) | 0 / 55 (+13 refused) |
| Metal | 92 | 60 / 60 (+4 refused) | 68 / 68 (+9 refused) | 29 / 29 (+48 refused) | n/a (R13) | 0 / 74 (+3 refused) | n/a (R13) | 3 / 3 (+74 refused) | n/a (R13) | 0 / 74 (+3 refused) |

The array path against MEEP: keep 6 / 6 (+4 refused) families identical; flush 6 / 6 (+4
refused) families identical. (d) and (e) are 0 under every baseline until the certification
round re-gates and the ledgers are re-minted.

## 5. The pending list by owner

`tools/ci/pending_certification.txt` lists 63 tests in 31 blocks. Every one is owned by the
certification round; no development lane owns any. By the step whose output removes it (each
block's own "Removed when" clause):

| Owner (step) | Entries | Test files |
|---|---|---|
| CERT-3 gates and the bind: re-gate on the frozen bytes, rebind the welds, re-cut the records and fingerprints | 39 | `test_ade_update_p` 1, `test_certification_record` 2, `test_complex_offdiag_update_e` 1, `test_conductive_kernels` 1, `test_constitutive_pml_real` 2, `test_cuda_weld_contract` 1, `test_dispatch_contract` 1, `test_dispersive` 1, `test_folded_offdiag` 1, `test_kernel_partition` 1, `test_metal_kernels` 1, `test_metal_weld_contract` 1, `test_no_pml_constitutive` 1, `test_no_pml_curl` 2, `test_offdiag_constitutive_pml_real` 2, `test_triton_complex_conductive_fused_pair` 1, `test_triton_complex_fused_magnetic_pair` 1, `test_triton_conductivity` 1, `test_triton_cylindrical_composition` 1, `test_triton_cylindrical_fused_electric_pair` 1, `test_triton_cylindrical_real_fused_magnetic_pair` 1, `test_triton_dispersive_fused_pair` 1, `test_triton_dispersive_update_e` 1, `test_triton_folded_beta_complex_fused_magnetic_pair` 1, `test_triton_folded_fused_magnetic_pair` 1, `test_triton_folded_fused_pair` 1, `test_triton_fused_ade_state` 1, `test_triton_kernels` 3, `test_triton_no_pml` 2, `test_triton_planner_composition` 1, `test_triton_symmetry_composition` 1, `test_triton_weld_contract` 1 |
| CERT-3 bind then CERT-4: bind the Triton bit-identity gate, re-run and rebind three Triton welds, recut the Triton route record with --end-to-end | 4 | `test_dispatch_contract` 1, `test_triton_weld_contract` 3 |
| CERT-4 route campaigns and recut_driver_dispatch_record.py (CUDA, Triton with the end-to-end licence, Metal) | 13 | `test_cuda_weld_contract` 5, `test_dispatch_contract` 3, `test_metal_weld_contract` 4, `test_triton_weld_contract` 1 |
| CERT-3 and CERT-4 together: the bit-identity gate, both NVIDIA route campaigns with the licence, and the Metal fleet and held route | 2 | `test_dispatch_contract` 2 |
| CERT-4 records: the ledgers re-minted by record_array_path_oracle.py | 3 | `test_cuda_weld_contract` 1, `test_metal_weld_contract` 1, `test_triton_weld_contract` 1 |
| CERT-8: the two timing records re-cut by a timing campaign on the shipped bytes | 2 | `test_dispatch_preference` 2 |

`tools/ci/check_failures.py` on entry 113's tree (the suites of the integration that appended
entries 113 and 114): 62 failed of 17,485 run, 62 / 62 declared, 0 undeclared.

## 6. Table 2: MEEP identity on the corpus

From `docs/development/meep-identity-corpus.md` as entry 115 regenerated it (the array path of
the 1.0 development series, dispatch off, every row in its own process, against four reference
MEEPs; entry 115's pinned corpus overlay brought in five row-references):

| reference | examples holding | tests holding | differ | error | not run | excluded | stored words held | reader words held |
|---|---|---|---|---|---|---|---|---|
| mac_keep | 57 of 60 | 132 of 134 | 0 | 0 | 4 | 1 | 2,728,743,678 | 3,997,904,328 |
| mac_flush | 57 of 60 | 132 of 134 | 0 | 0 | 4 | 1 | 2,728,743,678 | 3,997,904,328 |
| linux_keep | 57 of 60 | 132 of 134 | 0 | 0 | 4 | 1 | 2,728,743,678 | 3,997,904,328 |
| linux_flush | 57 of 60 | 132 of 134 | 0 | 0 | 4 | 1 | 2,728,743,678 | 3,997,904,328 |

One named exclusion (`ex.stochastic_emitter`, an unseeded random source); the rows not run are
named there with their reason (two need a MEEP built with libGDSII or gdspy, two a
BinaryPartition chunk layout with mirror symmetry, which the lift refuses).

- Entry 115 (`115-LHF`, the corpus overlay: `rounds/1.0/corpus_overlay-*.pip.txt`, shapely 2.1.2 and
  nlopt 2.11.0, the PyMieScatt stand-in and `ffmpeg`) closes three environment refusals, and Table 2
  now holds 189 of 194 rows on each of the four references (57 of 60 examples, 132 of 134 tests;
  differ 0, error 0; 4 refused by name: `ex.coupler`, `ex.ring_gds`, `test_infer_dimensions`,
  `test_require_dimensions`; 1 excluded). Re-run on entry 115's tree
  `87febcc85b4d85706b35baa507035f31b7e0bb7a` (01 to 115) at its integration: the five new
  row-references and two controls hold with 0 differing words in both modes on both hosts (Apple M1
  Max 4 of 4 rows per mode, x86-64 Linux 4 of 4 per mode; each row's digests, planes and currents
  equal to the Tier B record the page is cut from), the two Apple context L2 rows reproduce
  (3.760e-07, 4.378e-07), and the Tier B tests, the corpus and environment tests and
  `check_failures.py` read 0 failed of 156, 0 undeclared. It landed before this document, and it
  touched no gate-bound file: its six files (`CHANGELOG.md`,
  `docs/development/meep-identity-corpus.md` and `testing.md`, this directory's `RUNBOOK.md` and its
  two new overlay requirement files) hold no Python, none of the four kernel ledgers' 2,270
  `code_sha256` / `source_sha256` pins names one, and the board's subject does not include them, so
  no verdict in sections 2 to 9 moves.

## 7. Owner rulings

- **R1**, Metal's claim against MEEP: the x86 flush reference, falling back to a per-row claim
  where Linux keep and flush coincide, then none (default; decided at CERT-5).
- **R2**, the Mac's contraction-off MEEP becomes the Mac certification MEEP, replacing the
  osx-arm64 lock environment, whose lock is re-cut (default).
- **R3**, lifted runs form their currents through MEEP's `src_time` (default; design decision
  D24).
- **R4**, RULED 2026-10-09: hold the 0.9.x H200 bind; the H200 is certified on the 1.0 bytes in
  CERT-6 with the 1.0 runbook.
- **R5**, the timing ladder re-times both sides in one session (default; CERT-8).
- **R6**, RULED 2026-10-09: the Tier A fixture budget in the repository is 5 MB, measured at
  mint.
- **R7**, RULED 2026-10-08: Metal off-diagonal and nonlinear rows step in soft-fp64 rather than
  refusing to the array path.
- **R8**, `board_controls` is re-baselined by tool at CERT-4 if refusals move
  `served_in_dispatch` (default).
- **R9**, development in a fresh owner-named clone at the 0.9.2 tag, staging copies per lane;
  the owner commits (default).
- **R10**, `chunk_layout` together with mirror symmetry is refused until the separability probe
  shows 0 (default).
- **R11**, RULED 2026-10-09: the NumPy floor is raised to >= 2.
- **R12**, non-separable chunk divisions are refused on every kernel table in 1.0; the array
  path steps them (default).
- **R13**, RULED 2026-10-08: certify against both subnormal baselines where the hardware
  supports both: keep and flush on the NVIDIA tables and the array path, flush only on Metal
  (the Apple GPU flushes and exposes no lever).
- **R14**, RULED 2026-10-09: a CuPy RawKernel the caller compiled before a keep install keeps
  its policy; the install records it and the guide documents the remedy, and it does not refuse.
- **R15**, RULED 2026-10-09: where MEEP aliases H onto B's array, those B points are compared
  through MEEP's point read under both reader rules, each erroring on anything unexplained.
- **R16**, RULED 2026-10-10: option (a), the composition gates install and stamp the campaign's
  subnormal policy, so their flush runs carry per-arm flush evidence.
- **Pending**: the spine byte-neutrality gate (section 2, i-4): re-baseline its frozen artifact
  at CERT-1 (recommended) or retire the gate and its weld.

## 8. Open items the certification round owns

- **CERT-1**: move `fastpath.DRIVER_ROUTE_FUSED_GATE`,
  `fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE` and `metal_dispatch.METAL_DRIVER_ROUTE_GATE` to
  1.0 stamps (i-1), and act on the spine byte-neutrality ruling (i-4). Done at the freeze
  (section 11): both, the second by the owner's default. The constants moved again at the
  re-freeze (the Metal one) and at the freeze after per-kernel certification (all three, to
  `2026-10-13_10`; section 11's last bullet).
- **Bind: the curated pins the round's records cannot hold leave by generated rows.** Under
  per-kernel certification a kernel gate's record is its own import closure, and 3 Metal welds
  and 4 Triton welds pin core modules their gates no longer import (section 11's last bullet).
  Each rebind tool carries `RETIRED_UNIMPORTED_PINS` (31 Metal rows, 4 Triton, 0 CUDA),
  generated by `import_closure.py unimported-pins <table>` and reviewed; the Metal bind
  prints 34 `RETIRED-PIN` lines (with the three deleted-file pins) and rebinds the three
  welds, the Triton bind prints 4 (`driver.py`). Built before the freeze; nothing to do at the
  bind but read the lines.
- **Bind**: write per-arm flush evidence from the flush runs (the CUDA gates and the Triton
  fleet released under flush on these bytes, 59 / 59 and 61 / 61), with
  `recut_composition_records.py --flush-run` for the five composition gates (R16), before the
  flush route legs run (i-2, i-3).
- **Bind**: the Triton rebind retires `triton_unified_expansion_device_gate` by name. Its gate was
  deleted with the expansion licence, so no campaign can rebind it; `rebind_triton_welds.py` lists
  it in `RETIRED_WELDS`, reports it `RETIRED` and drops it from the Triton ledger under `--write`
  (never by hand). Check the report's `RETIRED` line and that the rebound ledger no longer carries
  the entry.
- **Bind**: retire the `_mixed_policy` declarations (and their `POLICY_UNNAMED_BUDGET` lines) of
  `triton_no_pml_device_gate` and `triton_cylindrical_device_gate`, a reviewed change.
  `triton_fused_electric_device_gate`'s gate now installs and stamps the campaign's policy and
  its declaration is retired (`RUNBOOK.md` section 5), so the arms `fused pair B` and `fused
  pair D` carry flush evidence once its keep and flush runs on the round's bytes are rebound.
- **F2, closed at the re-freeze (owner ruling 2026-10-11)**: rung 8b read a bare label in a
  CUDA-led plan against the CUDA ledger, so under `MEEP_GPU_BACKEND_PREFERENCE=cuda` with
  `flush` every Triton arm of the merged plan read as unmapped and `cuda_flush` could not
  release (`dispersive_2d` `NO-SUBSTITUTION`, the `nonlinear_3d` envelope control
  `ENVELOPE-DID-NOT-HOLD`). `fastpath._table_for_label` now answers a bare label in a plan
  composed on the CUDA table from the Triton ledger (`RUNBOOK.md` section 5; keep unchanged).
  `meep_gpu/fastpath.py` is pinned by 7 ledger entries and `meep_gpu/metal_dispatch.py` by
  1, so on the round's bytes the change voids, and the round re-runs on the re-frozen
  commit:
  - **Metal**, under a fresh Mac stamp `2026-10-12_10`, on a clean checkout: `metal_round.py
    all --stamp 2026-10-12_10` (the fleet manifest under `_2026-10-11_10` records the old
    `fastpath.py` digest, which the gate runner refuses to mix into that stamp),
    `rebind_metal_welds.py --stamp 2026-10-12_10 --write`, `metal_round.py route --stamp
    2026-10-12_10` into `dispatch_metal_route_2026-10-12_10_applegpu_g13s`
    (`metal_dispatch.METAL_DRIVER_ROUTE_GATE` moved; under `--stamp 2026-10-11_10` the route
    item's marker would skip it), then `recut_driver_dispatch_record.py --backend metal --run
    dispatch_metal_route_2026-10-12_10_applegpu_g13s --write`. The ledger's pins owe only the
    three welds that pin `fastpath.py` (`fleet --only` their three gates would serve the
    ledger and the route record; the rebind is then expected to report the other 59 without
    an artifact and leave them bound to `_2026-10-11_10`). The whole fleet is owed by the
    Metal board: all 62 gate artifacts of `_2026-10-11_10` record `meep_gpu/fastpath.py` in
    their `source_sha256`, which `build_fusion_matrix.py` re-hashes for every artifact it
    credits, so its release bindings move to `_2026-10-12_10`.
  - **The NVIDIA bind, carried or re-run (owner ruling).** Every NVIDIA artifact of the round
    with a source record names `meep_gpu/fastpath.py` too (246 of 246), and the bind tools
    that check a run's whole import record refuse those runs on the re-frozen bytes.
    Measured against the round's roots with `legs.py bind`'s own commands (report mode), on
    the FXE tree and on the re-frozen tree: `rebind_triton_welds.py --campaign ...` rebinds
    43 and skips 11 (51 and 3 before): eight device welds that pin their own test file,
    which their gate does not import, cannot take that pin from a tree the run disagrees with
    (`triton_complex_fused_ade_chain`, `triton_complex_fused_electric_pair`,
    `triton_complex_fused_magnetic_pair`, `triton_cylindrical_real_fused_magnetic_pair`,
    `triton_folded_deposit_closure`, `triton_folded_fused_magnetic_pair`,
    `triton_folded_fused_pair`, `triton_fused_ade_chain`); `recut_composition_records.py
    --family-recert` refuses all 9 families of `family_recert_2026-08-14` (flush evidence for
    22 arms, `nonlinear` among them); the bit-identity binder refuses. The CUDA rebinds, the
    ADE re-cut and the composition records read the same on both trees. Either the bind
    written on the frozen bytes is carried to the re-frozen commit (no curated pin of those
    entries names a file F2X moves) or those gates are re-run on it. The Triton board's
    gate-bound rows and the CUDA board's gate-tier columns re-hash their artifacts' import
    records the same way.
  - **Triton**: the bit identity (`bit_identity_gate` pins `fastpath.py`). A hand step whose
    artifact already passes is skipped, so move `results/triton_bit_identity_2026-10-11_10`
    and its `_flush` twin out of `results/` (keep them), rerun `round.py launch gates ...
    --stamp 2026-10-11_10 --resume --only triton/keep/bit_identity,triton/flush/bit_identity`,
    then the printed `rebind_triton_welds.py --bind-bit-identity ... --bit-identity-flush
    ...`.
  - **NVIDIA routes**: neither `driver_dispatch` record was cut on the frozen bytes; the
    one-shot route (stamps `2026-10-11_10`, unused) cuts both on the re-frozen ones.
- **CERT-4, the CUDA board's own controls**: closed before the freeze (work package XG). The
  detector control `cylindrical_kernels.py` read POINTWISE from entry 76 because the curl's
  neighbour reads (`shift_up`/`shift_dn`, `g[idx +/- stride]`) live in the real-PML prelude the
  module imports from `step_curl_kernels` and concatenates into its kernels, and the detector read
  only the module's own literals (the body's direct `pfx[idx + sx]` read had left for
  `cylindrical_prefix`'s scan). The detector now follows a prelude a module imports from a sibling
  and concatenates into a kernel (`_imported_preludes`); of the 46 kernel modules and 7 explicit
  labels it measures, that moves `cylindrical_kernels.py` alone (POINTWISE -> STENCIL, a curl,
  which no constitutive cell resolves to), and the four controls reproduce.
  `fused_update_{H,E}_pml_complex_bloch` are measured on `complex_emitter.complex_source`, the
  text the shipped launcher compiles, and read POINTWISE (were UNDETERMINED on five unfilled
  splice points). `test_cuda_board_detector.py` pins both, and that each control fails on a
  planted error.
- **The Metal composition matrix's four cylindrical complex rows**: closed before the freeze
  (work package XG). At the builder's Courant 0.37, not a float32 word, the cylindrical complex
  curl refuses by name the slots that carry an axis product with Courant: `step_B` and `step_D`
  on `dcyl_m1_complex`, `_negative` and `_zperiodic`, `step_D` alone on `dcyl_m0_complex`. The
  four rows are pinned to that (the refusal is unchanged), and four `_f32_courant` twins at the
  float32 word pin all four slots composed, so the matrix still drives the curl on each
  specialisation axis (listed under `torch` in `declared_resources.txt`, as their siblings). The
  composition test and probe pass with the evidence archive (153 of 153; 114 configurations,
  VERDICT PASS) and without it.
- **CERT-4, the route watchdogs**: closed before the freeze (work package XG). The route and
  licence legs' expected walls in `legs.py` are this pass's measured walls, so the route stage's
  `max(3600 s, 12 x expected)` (entry 121) is at least 12.0 x the wall each leg guards, computed
  from `round.py`'s own route plans (on the 0.9.2 prices it was 1.70 x for Triton `shipped` and
  1.76 x for `flush`); `metal_round.py`'s route campaign watchdog is 3 x the serial sum of its
  four legs' measured walls (53,760 s, 4.25 x the measured campaign; it was a fixed 6 h, 1.71 x),
  with `--route-timeout-factor` and `--route-min-timeout`, and every Metal child runs with
  `PYTHONNOUSERSITE=1`. `test_round_route_watchdog.py` holds every route watchdog at 3 x or
  more of the measured walls it records.
- **CERT-4**: re-cut the Triton census without the retired `@unified_expansion` keys before the
  board (`build_triton_fusion_matrix.py` refuses the census of record by name); re-baseline
  `board_controls` by tool (R8).
- **CERT-4 records**: re-mint the ledgers with `record_array_path_oracle.py` (every byte-binding
  entry records `real-v1`; the live oracle is `real-v6`), so (e) can read anything.
- **The Triton H->D welds' full gates under flush** (every leg, lifts on) are not in the round's
  leg table and have not run on these bytes; they are not dispatched (`INSTALLABLE` is False on
  all ten), so no route leg reads them. Under keep they are in it since the harness-tools entry
  after the 1.0 round (`legs.py` `TRITON_HD`), because the Triton board binds their keep runs.
- **The planner's Triton prices**: closed after this pass. `legs.py` `TRITON_EXPECT` and
  `TRITON_HAND_EXPECT` now carry this pass's measured walls (the larger of keep and flush per
  gate), where they priced `special_kz` at 184 s against a measured 2,265 s (keep) and 4,109 s
  (flush) and `folded_complex`, `bfast` and `complex` 1.9x to 5.2x under. The planner's CERT-3
  NVIDIA wall on 4 GPUs moves from 17,195 s to 23,834 s (`RUNBOOK.md` section 2).
- **Two Triton fleet gates record no subnormal policy** in their artifacts under either policy
  (`fused_electric`, `source_seams`); their logs show the policy reached. A stamp before the
  bind uses flush evidence would make them as strong as their siblings.
- **The composition-shaped gates record no imported digests** (five gates, ten legs); their
  records bind the root's `staged_source_sha256.txt` instead, so the round must re-run them
  rather than reuse them.
- **The Tier A flush record states no contraction setting**: the board renders its reference as
  "contraction None" beside the keep reference's "contraction off".
- **CERT-6** (H200, R4) and **CERT-8** (timing on an idle box; it re-cuts the two timing records
  the pending list holds) run on the frozen bytes.

## 9. Every leg, with the tree it ran on

A reused leg's last column is its byte-identity check on `87febcc8`: bound files identical /
bound files. A leg is reused only when the two are equal.

### 9.1 Metal (Apple M1 Max, flush): every leg ran on `15da8cd4d3fba035cd26660e177db8429df12882`

| Leg | Verdict | Wall (s) | Byte-identical on `87febcc8` |
|---|---|---|---|
| `metal_environment_start` | rc 0 | 0.9 | 2 / 2 |
| `metal_environment_end` | rc 0 | 0.9 | 2 / 2 |
| `gate_metal_ade_update_p` | released | 3.3 | 50 / 50 |
| `gate_metal_below_the_cut_fused_pairs` | released | 18.1 | 58 / 58 |
| `gate_metal_beta_complex_fused_hd_pair` | released | 97.0 | 135 / 135 |
| `gate_metal_beta_real_fused_hd_pair` | released | 87.6 | 135 / 135 |
| `gate_metal_bfast` | released | 30.4 | 135 / 135 |
| `gate_metal_complex` | released | 26.2 | 132 / 132 |
| `gate_metal_complex_conductive_fused_pair` | released | 9.8 | 56 / 56 |
| `gate_metal_complex_conductive_pml` | released | 3.7 | 54 / 54 |
| `gate_metal_complex_dispersive_update_e` | released | 3.0 | 53 / 53 |
| `gate_metal_complex_folded_offdiag` | released | 3.1 | 56 / 56 |
| `gate_metal_complex_fused_ade_chain` | released | 10.0 | 57 / 57 |
| `gate_metal_complex_fused_hd_pair` | released | 23.7 | 135 / 135 |
| `gate_metal_complex_fused_magnetic_pair` | released | 11.3 | 52 / 52 |
| `gate_metal_complex_no_pml_conductive` | released | 3.1 | 54 / 54 |
| `gate_metal_complex_no_pml_curl` | released | 3.0 | 55 / 55 |
| `gate_metal_complex_no_pml_offdiag_update_e` | released | 3.2 | 130 / 130 |
| `gate_metal_complex_no_pml_stored_e` | released | 2.8 | 50 / 50 |
| `gate_metal_conductive_fused_hd_pair` | released | 185.1 | 134 / 134 |
| `gate_metal_conductive_pml` | released | 5.5 | 54 / 54 |
| `gate_metal_cylindrical_complex` | released | 41.7 | 136 / 136 |
| `gate_metal_cylindrical_complex_fused_hd_pair` | released | 8.8 | 60 / 60 |
| `gate_metal_cylindrical_complex_scan` | released | 24.6 | 52 / 52 |
| `gate_metal_cylindrical_fused_electric_pair` | released | 14.9 | 132 / 132 |
| `gate_metal_cylindrical_fused_magnetic_pair` | released | 15.2 | 55 / 55 |
| `gate_metal_cylindrical_real` | released | 52.9 | 135 / 135 |
| `gate_metal_cylindrical_real_fused_electric_pair` | released | 12.7 | 132 / 132 |
| `gate_metal_cylindrical_real_fused_hd_pair` | released | 83.1 | 134 / 134 |
| `gate_metal_cylindrical_real_fused_magnetic_pair` | released | 10.4 | 53 / 53 |
| `gate_metal_dispersive_update_e` | released | 3.0 | 51 / 51 |
| `gate_metal_folded_beta` | released | 31.6 | 135 / 135 |
| `gate_metal_folded_beta_complex_fused_magnetic_pair` | released | 20.7 | 131 / 131 |
| `gate_metal_folded_complex` | released | 7.3 | 54 / 54 |
| `gate_metal_folded_complex_fused_hd_pair` | released | 116.1 | 135 / 135 |
| `gate_metal_folded_complex_fused_magnetic_pair` | released | 24.3 | 55 / 55 |
| `gate_metal_folded_dispersive_update_e` | released | 3.0 | 55 / 55 |
| `gate_metal_folded_fused_magnetic_pair` | released | 34.3 | 54 / 54 |
| `gate_metal_folded_fused_pair` | released | 14.4 | 57 / 57 |
| `gate_metal_folded_offdiag` | released | 15.1 | 55 / 55 |
| `gate_metal_folded_offdiag_dispersive` | released | 3.7 | 130 / 130 |
| `gate_metal_fused_ade_chain` | released | 13.2 | 59 / 59 |
| `gate_metal_fused_dispersive_pair` | released | 18.3 | 55 / 55 |
| `gate_metal_fused_electric_pair` | released | 14.9 | 134 / 134 |
| `gate_metal_fused_hd_pair_nonlinear` | released | 159.3 | 134 / 134 |
| `gate_metal_fused_magnetic_pair` | released | 15.9 | 134 / 134 |
| `gate_metal_no_pml_conductive` | released | 2.8 | 48 / 48 |
| `gate_metal_no_pml_constitutive` | released | 8.4 | 130 / 130 |
| `gate_metal_no_pml_curl` | released | 3.3 | 129 / 129 |
| `gate_metal_no_pml_fused_electric_pairs` | released | 16.4 | 131 / 131 |
| `gate_metal_no_pml_stored_e` | released | 2.8 | 50 / 50 |
| `gate_metal_nonlinear` | released | 13.7 | 51 / 51 |
| `gate_metal_offdiag` | released | 63.2 | 132 / 132 |
| `gate_metal_offdiag_stencil_welds` | released | 31.8 | 131 / 131 |
| `gate_metal_pml` | released | 14.1 | 133 / 133 |
| `gate_metal_residue_fused_pairs` | released | 10.7 | 132 / 132 |
| `gate_metal_special_kz` | released | 37.6 | 135 / 135 |
| `gate_metal_spine_byte_neutrality` | not released (i) | 0.7 | 45 / 45 |
| `gate_metal_symmetry` | released | 18.5 | 133 / 133 |
| `gate_metal_tranche7_fused_pairs` | released | 18.3 | 131 / 131 |
| `gate_metal_whole_step` | released | 38.0 | 131 / 131 |
| `gate_metal_fused_hd_pair` | released | 1327.0 | 133 / 133 |
| `gate_metal_folded_fused_hd_pair` | released | 926.9 | 134 / 134 |
| `gate_metal_bfast_fused_hd_pair` | released | 223.0 | 134 / 134 |
| `route/band_witness` | released | - | 134 / 134 |
| `route/harness_flush` | released | - | 136 / 136 |
| `route/harness_keep` | released | - | 135 / 135 |
| `route/shipped` | released | - | 136 / 136 |

Beside its source files, a Metal gate artifact records digests of emitted kernel text, keyed by
variant labels (524 over the 68 legs); they derive from the bound sources, and no Python file
differs between `15da8cd4` and `87febcc8`.

### 9.2 NVIDIA legs that ran on `87febcc85b4d85706b35baa507035f31b7e0bb7a` (one RTX A6000, compute capability 8.6)

| Leg | Policy | Verdict | Why it re-ran | Earlier runs |
|---|---|---|---|---|
| `triton/keep/fleet/cylindrical_complex_composition` | keep | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/keep/t2_rest/source_seams` | keep | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/keep/composition/conductivity_composition` | keep | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/keep/composition/cylindrical_composition` | keep | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/keep/composition/symmetry_composition` | keep | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/flush/fleet/cylindrical_complex_composition` | flush | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/flush/t2_rest/source_seams` | flush | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/flush/composition/conductivity_composition` | flush | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/flush/composition/cylindrical_composition` | flush | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |
| `triton/flush/composition/symmetry_composition` | flush | released | its artifact records no digests, so no byte check can license a reuse | `b676c5f5` released; `15da8cd4` released |

### 9.3 NVIDIA legs that ran on `15da8cd4d3fba035cd26660e177db8429df12882`

| Leg | Policy | Verdict | Why it re-ran there | Byte-identical on `87febcc8` |
|---|---|---|---|---|
| `ade_recut/keep` | keep | released | bound file moved after entry 111: meep_gpu/cuda_kernels/test_ade_update_p.py | 40 / 40 |
| `ade_recut/flush` | flush | released | bound file moved after entry 111: meep_gpu/cuda_kernels/test_ade_update_p.py | 40 / 40 |

### 9.4 NVIDIA legs that ran on `b676c5f53af71cf6056471f84d17c28d25fae114`

| Leg | Policy | Verdict | Class | Byte-identical on `87febcc8` |
|---|---|---|---|---|
| `residue/cuda_fused_hd_pair_2026-09-06/keep` | keep | released | released | 123 / 123 |
| `residue/cuda_fused_hd_pair_2026-09-06/flush` | flush | released | released | 123 / 123 |
| `cited/cuda_dispersive_offdiag_2026-08-20b/keep/gate.json` | keep | released | released | 47 / 47 |
| `cited/cuda_dispersive_offdiag_2026-08-20b/flush/gate.json` | flush | released | released | 47 / 47 |
| `cited/cuda_fused_polarization_pair_2026-09-01_pml/keep/gate.json` | keep | released | released | 40 / 40 |
| `cited/cuda_fused_polarization_pair_2026-09-01_pml/flush/gate.json` | flush | released | released | 40 / 40 |
| `residue/cuda_complex_no_pml_2026-08-27/keep` | keep | released | released | 39 / 39 |
| `residue/cuda_complex_no_pml_2026-08-27/flush` | flush | released | released | 39 / 39 |
| `cited/cuda_fused_complex_pairs_2026-09-11_regate/complex_electric_keep/gate.json` | keep | released | released | 43 / 43 |
| `cited/cuda_fused_complex_pairs_2026-09-11_regate/complex_keep/gate.json` | keep | released | released | 43 / 43 |
| `cited/cuda_fused_complex_pairs_2026-09-11_regate/no_pml_complex_electric_keep/gate.json` | keep | released | released | 42 / 42 |
| `cited/cuda_fused_complex_pairs_2026-09-11_regate/complex_electric_flush/gate.json` | flush | released | released | 43 / 43 |
| `cited/cuda_fused_complex_pairs_2026-09-11_regate/complex_flush/gate.json` | flush | released | released | 43 / 43 |
| `cited/cuda_fused_complex_pairs_2026-09-11_regate/no_pml_complex_electric_flush/gate.json` | flush | released | released | 42 / 42 |
| `cited/cuda_fused_polarization_pair_2026-09-01_no_pml/keep/gate.json` | keep | released | released | 40 / 40 |
| `cited/cuda_fused_polarization_pair_2026-09-01_no_pml/flush/gate.json` | flush | released | released | 40 / 40 |
| `cited/cuda_in_seam_2026-08-21_final/fill_folded_far/keep/gate.json` | keep | released | released | 35 / 35 |
| `cited/cuda_in_seam_2026-08-21_final/fill_symmetry/keep/gate.json` | keep | released | released | 35 / 35 |
| `cited/cuda_in_seam_2026-08-21_final/zero_metal/keep/gate.json` | keep | released | released | 35 / 35 |
| `cited/cuda_in_seam_2026-08-21_final/fill_folded_far/flush/gate.json` | flush | released | released | 35 / 35 |
| `cited/cuda_in_seam_2026-08-21_final/fill_symmetry/flush/gate.json` | flush | released | released | 35 / 35 |
| `cited/cuda_in_seam_2026-08-21_final/zero_metal/flush/gate.json` | flush | released | released | 35 / 35 |
| `residue/cuda_fused_magnetic_pair_2026-08-28_fillcarry/keep` | keep | released | released | 118 / 118 |
| `residue/cuda_fused_magnetic_pair_2026-08-28_fillcarry/flush` | flush | released | released | 118 / 118 |
| `residue/cuda_fused_magnetic_pair_2026-09-01_nonlinear/keep` | keep | released | released | 118 / 118 |
| `residue/cuda_fused_magnetic_pair_2026-09-01_nonlinear/flush` | flush | released | released | 118 / 118 |
| `residue/cuda_folded_constitutive_2026-08-21_rename/keep` | keep | released | released | 36 / 36 |
| `residue/cuda_folded_constitutive_2026-08-21_rename/flush` | flush | released | released | 36 / 36 |
| `cited/cuda_fused_residual_bfast_2026-09-02/keep/gate.json` | keep | released | released | 49 / 49 |
| `cited/cuda_fused_residual_bfast_2026-09-02/flush/gate.json` | flush | released | released | 49 / 49 |
| `cited/cuda_fused_residual_special_kz_2026-09-02/keep/gate.json` | keep | released | released | 50 / 50 |
| `cited/cuda_fused_residual_special_kz_2026-09-02/flush/gate.json` | flush | released | released | 50 / 50 |
| `cited/cuda_residue_complex_beta_electric_2026-09-02/keep/gate.json` | keep | released | released | 48 / 48 |
| `cited/cuda_residue_complex_beta_electric_2026-09-02/flush/gate.json` | flush | released | released | 48 / 48 |
| `residue/cuda_residue_special_kz_fillcarry_2026-09-02/keep` | keep | released | released | 50 / 50 |
| `residue/cuda_residue_special_kz_fillcarry_2026-09-02/flush` | flush | released | released | 50 / 50 |
| `residue/cuda_conductive_2026-08-21_rename/keep` | keep | released | released | 39 / 39 |
| `residue/cuda_conductive_2026-08-21_rename/flush` | flush | released | released | 39 / 39 |
| `cited/cuda_complex_offdiag_stencil_welds_2026-09-17/keep/gate.json` | keep | released | released | 53 / 53 |
| `cited/cuda_complex_offdiag_stencil_welds_2026-09-17/flush/gate.json` | flush | released | released | 53 / 53 |
| `residue/cuda_ade_2026-08-19/keep` | keep | released | released | 39 / 39 |
| `residue/cuda_ade_2026-08-19/flush` | flush | released | released | 39 / 39 |
| `cited/cuda_three_slot_weld_2026-09-11_regate/keep_no_pml/gate.json` | keep | released | released | 121 / 121 |
| `cited/cuda_three_slot_weld_2026-09-11_regate/keep_pml/gate.json` | keep | released | released | 121 / 121 |
| `cited/cuda_three_slot_weld_2026-09-11_regate/flush_no_pml/gate.json` | flush | released | released | 121 / 121 |
| `cited/cuda_three_slot_weld_2026-09-11_regate/flush_pml/gate.json` | flush | released | released | 121 / 121 |
| `cited/cuda_fused_electric_pair_2026-08-31_fillcarry_r3/keep/gate.json` | keep | released | released | 118 / 118 |
| `cited/cuda_fused_electric_pair_2026-08-31_fillcarry_r3/flush/gate.json` | flush | released | released | 118 / 118 |
| `residue/cuda_complex_folded_2026-09-01/keep` | keep | released | released | 40 / 40 |
| `residue/cuda_complex_folded_2026-09-01/flush` | flush | released | released | 41 / 41 |
| `residue/cuda_complex_beta_2026-09-01/keep` | keep | released | released | 43 / 43 |
| `residue/cuda_complex_beta_2026-09-01/flush` | flush | released | released | 44 / 44 |
| `cited/cuda_dispersive_offdiag_polarization_pair_2026-09-02/keep/gate.json` | keep | released | released | 52 / 52 |
| `cited/cuda_dispersive_offdiag_polarization_pair_2026-09-02/flush/gate.json` | flush | released | released | 52 / 52 |
| `cited/cuda_fused_residual_cylindrical_real_2026-09-02/keep/gate.json` | keep | released | released | 47 / 47 |
| `cited/cuda_fused_residual_cylindrical_real_2026-09-02/flush/gate.json` | flush | released | released | 47 / 47 |
| `cited/cuda_residue_bfast_electric_2026-09-02/keep/gate.json` | keep | released | released | 46 / 46 |
| `cited/cuda_residue_bfast_electric_2026-09-02/flush/gate.json` | flush | released | released | 46 / 46 |
| `cited/cuda_residue_special_kz_electric_2026-09-02/keep/gate.json` | keep | released | released | 49 / 49 |
| `cited/cuda_residue_special_kz_electric_2026-09-02/flush/gate.json` | flush | released | released | 49 / 49 |
| `residue/cuda_cylindrical_complex_2026-08-26/keep` | keep | released | released | 41 / 41 |
| `residue/cuda_cylindrical_complex_2026-08-26/flush` | flush | released | released | 41 / 41 |
| `residue/cuda_complex_regate_2026-08-26/keep` | keep | released | released | 38 / 38 |
| `residue/cuda_complex_regate_2026-08-26/flush` | flush | released | released | 38 / 38 |
| `residue/cuda_cylindrical_fused_hd_pair_2026-09-07/keep` | keep | released | released | 126 / 126 |
| `residue/cuda_cylindrical_fused_hd_pair_2026-09-07/flush` | flush | released | released | 126 / 126 |
| `cited/fused_pml_bit_identity_hand_2026-08-22_rename/keep/probe.json` | keep | released | released | 32 / 32 |
| `cited/fused_pml_bit_identity_hand_2026-08-22_rename/flush/probe.json` | flush | released | released | 32 / 32 |
| `residue/cuda_cylindrical_real_fused_hd_pair_2026-09-07/keep` | keep | released | released | 126 / 126 |
| `residue/cuda_cylindrical_real_fused_hd_pair_2026-09-07/flush` | flush | released | released | 126 / 126 |
| `cited/cuda_constitutive_recut_2026-08-22_rename/keep/cgate.json` | keep | released | released | 33 / 33 |
| `cited/cuda_constitutive_recut_2026-08-22_rename/flush/cgate.json` | flush | released | released | 33 / 33 |
| `residue/cuda_nonlinear_2026-08-21_rename/keep` | keep | released | released | 37 / 37 |
| `residue/cuda_nonlinear_2026-08-21_rename/flush` | flush | released | released | 37 / 37 |
| `cited/cuda_fused_magnetic_pair_2026-08-27/keep/gate.json` | keep | released | released | 118 / 118 |
| `cited/cuda_fused_magnetic_pair_2026-08-27/flush/gate.json` | flush | released | released | 118 / 118 |
| `cited/cuda_complex_three_slot_weld_2026-09-04_gatefix/keep/gate.json` | keep | released | released | 124 / 124 |
| `cited/cuda_complex_three_slot_weld_2026-09-04_gatefix/flush/gate.json` | flush | released | released | 124 / 124 |
| `cited/cuda_conductive_fused_electric_pair_2026-09-02b/keep/gate.json` | keep | released | released | 97 / 97 |
| `cited/cuda_conductive_fused_electric_pair_2026-09-02b/flush/gate.json` | flush | released | released | 97 / 97 |
| `cited/cuda_fused_residual_complex_folded_2026-09-02/keep/gate.json` | keep | released | released | 44 / 44 |
| `cited/cuda_fused_residual_complex_folded_2026-09-02/flush/gate.json` | flush | released | released | 44 / 44 |
| `cited/cuda_offdiag_stencil_welds_2026-09-02/keep/gate.json` | keep | released | released | 45 / 45 |
| `cited/cuda_offdiag_stencil_welds_2026-09-02/flush/gate.json` | flush | released | released | 45 / 45 |
| `cited/cuda_residue_folded_complex_electric_2026-09-02/keep/gate.json` | keep | released | released | 45 / 45 |
| `cited/cuda_residue_folded_complex_electric_2026-09-02/flush/gate.json` | flush | released | released | 45 / 45 |
| `residue/cuda_no_pml_curl_2026-08-21_rename/keep` | keep | released | released | 37 / 37 |
| `residue/cuda_no_pml_curl_2026-08-21_rename/flush` | flush | released | released | 37 / 37 |
| `residue/cuda_complex_fused_hd_pair_2026-09-07/keep` | keep | released | released | 125 / 125 |
| `residue/cuda_complex_fused_hd_pair_2026-09-07/flush` | flush | released | released | 125 / 125 |
| `residue/cuda_dispersive_2026-08-26/keep` | keep | released | released | 38 / 38 |
| `residue/cuda_dispersive_2026-08-26/flush` | flush | released | released | 38 / 38 |
| `cited/cuda_folded_offdiag_kernel_2026-08-21_rename/keep/gate.json` | keep | released | released | 44 / 44 |
| `cited/cuda_folded_offdiag_kernel_2026-08-21_rename/flush/gate.json` | flush | released | released | 44 / 44 |
| `residue/cuda_cylindrical_real_2026-08-26/keep` | keep | released | released | 40 / 40 |
| `residue/cuda_cylindrical_real_2026-08-26/flush` | flush | released | released | 40 / 40 |
| `cited/cuda_fused_complex_pairs_2026-09-13_cyl/cylindrical_electric_keep/gate.json` | keep | released | released | 46 / 46 |
| `cited/cuda_fused_complex_pairs_2026-09-13_cyl/cylindrical_keep/gate.json` | keep | released | released | 45 / 45 |
| `cited/cuda_fused_complex_pairs_2026-09-13_cyl/cylindrical_electric_flush/gate.json` | flush | released | released | 46 / 46 |
| `cited/cuda_fused_complex_pairs_2026-09-13_cyl/cylindrical_flush/gate.json` | flush | released | released | 45 / 45 |
| `residue/cuda_special_kz_2026-08-21_rename_r2/keep` | keep | released | released | 43 / 43 |
| `residue/cuda_special_kz_2026-08-21_rename_r2/flush` | flush | released | released | 43 / 43 |
| `residue/cuda_folded_constitutive_2026-08-19/keep` | keep | released | released | 36 / 36 |
| `residue/cuda_folded_constitutive_2026-08-19/flush` | flush | released | released | 36 / 36 |
| `residue/cuda_bfast_2026-08-21_rename_r2/keep` | keep | released | released | 43 / 43 |
| `residue/cuda_bfast_2026-08-21_rename_r2/flush` | flush | released | released | 43 / 43 |
| `cited/cuda_fused_residual_complex_beta_2026-09-02/keep/gate.json` | keep | released | released | 47 / 47 |
| `cited/cuda_fused_residual_complex_beta_2026-09-02/flush/gate.json` | flush | released | released | 47 / 47 |
| `cited/cuda_offdiag_2026-08-21_rename/keep/gate.json` | keep | released | released | 44 / 44 |
| `cited/cuda_offdiag_2026-08-21_rename/flush/gate.json` | flush | released | released | 44 / 44 |
| `cited/cuda_residue_cylindrical_real_electric_2026-09-02/keep/gate.json` | keep | released | released | 47 / 47 |
| `cited/cuda_residue_cylindrical_real_electric_2026-09-02/flush/gate.json` | flush | released | released | 47 / 47 |
| `cited/cuda_residue_complex_polarization_pair_2026-09-02b/keep/gate.json` | keep | released | released | 40 / 40 |
| `cited/cuda_residue_complex_polarization_pair_2026-09-02b/flush/gate.json` | flush | released | released | 40 / 40 |
| `residue/cuda_residue_complex_polarization_pair_2026-09-02/keep` | keep | released | released | 40 / 40 |
| `residue/cuda_residue_complex_polarization_pair_2026-09-02/flush` | flush | released | released | 40 / 40 |
| `route/cuda/c_preflight` | keep | released | released | 159 / 159 |
| `route/cuda/cuda_shipped` | keep | released | released | 167 / 167 |
| `route/cuda/cuda_harness_keep` | keep | released | released | 167 / 167 |
| `route/cuda/c_preflight_alone` | keep | released | released | 101 / 101 |
| `route/cuda/cuda_default_precedence` | keep | released | released | 167 / 167 |
| `route/cuda/cuda_alone` | keep | released | released | 109 / 109 |
| `route/cuda/cuda_flush` | flush | not_released | i | 167 / 167 |
| `route/triton/t_preflight` | keep | released | released | 158 / 158 |
| `route/triton/shipped` | keep | released | released | 167 / 167 |
| `route/triton/harness_keep` | keep | released | released | 167 / 167 |
| `route/triton+cuda/e2e_preflight` | default | released | released | 41 / 41 |
| `route/triton+cuda/e2e_ship` | default | released | released | 41 / 41 |
| `route/triton+cuda/e2e_null` | default | released | released | 41 / 41 |
| `route/triton+cuda/e2e_verdict` | default | released | released | 41 / 41 |
| `route/triton/flush` | flush | not_released | i | 167 / 167 |
| `triton/keep/stencil_offdiag_fused_electric_pair` | keep | released | released | 45 / 45 |
| `triton/keep/stencil_folded_offdiag_fused_electric_pair` | keep | released | released | 45 / 45 |
| `triton/keep/bit_identity` | keep | released | released | 45 / 45 |
| `triton/flush/stencil_offdiag_fused_electric_pair` | flush | released | released | 46 / 46 |
| `triton/flush/stencil_folded_offdiag_fused_electric_pair` | flush | released | released | 46 / 46 |
| `triton/flush/bit_identity` | flush | released | released | 45 / 45 |
| `triton/keep/fleet/cylindrical_complex` | keep | released | released | 41 / 41 |
| `triton/keep/env_probe/folded_beta_complex_fused_pair` | keep | released | released | 50 / 50 |
| `triton/keep/env_probe/folded_beta_complex_fused_magnetic_pair` | keep | released | released | 52 / 52 |
| `triton/keep/t2_rest/fused_offdiag_electric` | keep | released | released | 41 / 41 |
| `triton/keep/fleet/complex` | keep | released | released | 41 / 41 |
| `triton/keep/fleet/fused_ade_chain` | keep | released | released | 43 / 43 |
| `triton/keep/fleet/folded_beta_fused_magnetic_pair` | keep | released | released | 47 / 47 |
| `triton/keep/t2_rest/folded_offdiag_fused_ade_chain` | keep | released | released | 51 / 51 |
| `triton/keep/fleet/folded_complex` | keep | released | released | 47 / 47 |
| `triton/keep/env_probe/folded_complex_fused_pair` | keep | released | released | 47 / 47 |
| `triton/keep/t2_conductivity/conductivity` | keep | released | released | 43 / 43 |
| `triton/keep/fleet/cylindrical_fused_electric_pair` | keep | released | released | 46 / 46 |
| `triton/keep/t2_symmetry/symmetry` | keep | released | released | 44 / 44 |
| `triton/keep/fleet/nonlinear` | keep | released | released | 44 / 44 |
| `triton/keep/fleet/bfast` | keep | released | released | 50 / 50 |
| `triton/keep/fleet/cylindrical_fused_magnetic_pair` | keep | released | released | 47 / 47 |
| `triton/keep/fleet/folded_dispersive_fused_pair` | keep | released | released | 43 / 43 |
| `triton/keep/fleet/complex_conductive_fused_pair` | keep | released | released | 48 / 48 |
| `triton/keep/fleet/probe_triton_folded_fused_pair` | keep | released | released | 42 / 42 |
| `triton/keep/fleet/folded_offdiag` | keep | released | released | 47 / 47 |
| `triton/keep/env_probe/complex_beta_fused_electric_pair` | keep | released | released | 44 / 44 |
| `triton/keep/fleet/bfast_fused_magnetic_pair` | keep | released | released | 42 / 42 |
| `triton/keep/fleet/probe_triton_folded_deposit_closure` | keep | released | released | 88 / 88 |
| `triton/keep/fleet/bfast_fused_electric_pair` | keep | released | released | 43 / 43 |
| `triton/keep/fleet/folded_beta_fused_electric_pair` | keep | released | released | 47 / 47 |
| `triton/keep/fleet/offdiag` | keep | released | released | 44 / 44 |
| `triton/keep/fleet/probe_triton_folded_fused_magnetic_pair` | keep | released | released | 40 / 40 |
| `triton/keep/fleet/beta_fused_electric_pair` | keep | released | released | 44 / 44 |
| `triton/keep/fleet/no_pml_fused_electric_pair` | keep | released | released | 41 / 41 |
| `triton/keep/fleet/beta_fused_magnetic_pair` | keep | released | released | 43 / 43 |
| `triton/keep/t2_rest/probe_triton_fused_dispersive_chain` | keep | released | released | 91 / 91 |
| `triton/keep/env_probe/probe_triton_folded_complex_fused_magnetic_pair` | keep | released | released | 46 / 46 |
| `triton/keep/env_probe/complex_fused_electric_pair` | keep | released | released | 40 / 40 |
| `triton/keep/env_probe/complex_beta_fused_magnetic_pair` | keep | released | released | 44 / 44 |
| `triton/keep/t2_rest/cylindrical` | keep | released | released | 38 / 38 |
| `triton/keep/fleet/cylindrical_real_fused_electric_pair` | keep | released | released | 41 / 41 |
| `triton/keep/fleet/fused_electric` | keep | released | released | 40 / 40 |
| `triton/keep/fleet/probe_triton_fused_ade_state` | keep | released | released | 89 / 89 |
| `triton/keep/fleet/conductive_fused_electric_pair` | keep | released | released | 41 / 41 |
| `triton/keep/composition/engine_route` | keep | released | released | 48 / 48 |
| `triton/keep/fleet/no_pml_constitutive` | keep | released | released | 57 / 57 |
| `triton/keep/fleet/complex_ade` | keep | released | released | 45 / 45 |
| `triton/keep/env_probe/probe_triton_complex_fused_magnetic_pair` | keep | released | released | 40 / 40 |
| `triton/keep/env_probe/nonlinear_fused_magnetic_pair` | keep | released | released | 40 / 40 |
| `triton/keep/fleet/complex_no_pml_stored_e` | keep | released | released | 47 / 47 |
| `triton/keep/composition/no_pml` | keep | released | released | 49 / 49 |
| `triton/keep/fleet/no_pml_conductive` | keep | released | released | 40 / 40 |
| `triton/keep/fleet/probe_triton_dispersive_fused_pair` | keep | released | released | 88 / 88 |
| `triton/keep/env_probe/complex_fused_ade_chain` | keep | released | released | 42 / 42 |
| `triton/keep/fleet/no_pml_stored_e` | keep | released | released | 40 / 40 |
| `triton/keep/env_probe/complex_no_pml_conductive` | keep | released | released | 43 / 43 |
| `triton/keep/t2_rest/folded_offdiag_dispersive` | keep | released | released | 44 / 44 |
| `triton/keep/fleet/probe_triton_cylindrical_real_fused_magnetic_pair` | keep | released | released | 40 / 40 |
| `triton/keep/offdiag_probe/complex_offdiag` | keep | released | released | 45 / 45 |
| `triton/keep/env_probe/complex_no_pml_curl` | keep | released | released | 40 / 40 |
| `triton/keep/fleet/special_kz` | keep | released | released | 49 / 49 |
| `triton/flush/fleet/cylindrical_complex` | flush | released | released | 43 / 43 |
| `triton/flush/env_probe/folded_beta_complex_fused_pair` | flush | released | released | 52 / 52 |
| `triton/flush/env_probe/folded_beta_complex_fused_magnetic_pair` | flush | released | released | 54 / 54 |
| `triton/flush/t2_rest/fused_offdiag_electric` | flush | released | released | 43 / 43 |
| `triton/flush/fleet/complex` | flush | released | released | 43 / 43 |
| `triton/flush/fleet/fused_ade_chain` | flush | released | released | 45 / 45 |
| `triton/flush/fleet/folded_beta_fused_magnetic_pair` | flush | released | released | 49 / 49 |
| `triton/flush/t2_rest/folded_offdiag_fused_ade_chain` | flush | released | released | 53 / 53 |
| `triton/flush/fleet/folded_complex` | flush | released | released | 49 / 49 |
| `triton/flush/env_probe/folded_complex_fused_pair` | flush | released | released | 49 / 49 |
| `triton/flush/fleet/bfast` | flush | released | released | 52 / 52 |
| `triton/flush/fleet/cylindrical_fused_electric_pair` | flush | released | released | 48 / 48 |
| `triton/flush/t2_conductivity/conductivity` | flush | released | released | 45 / 45 |
| `triton/flush/t2_symmetry/symmetry` | flush | released | released | 46 / 46 |
| `triton/flush/fleet/cylindrical_fused_magnetic_pair` | flush | released | released | 49 / 49 |
| `triton/flush/fleet/probe_triton_folded_fused_pair` | flush | released | released | 44 / 44 |
| `triton/flush/fleet/nonlinear` | flush | released | released | 46 / 46 |
| `triton/flush/fleet/folded_dispersive_fused_pair` | flush | released | released | 45 / 45 |
| `triton/flush/fleet/complex_conductive_fused_pair` | flush | released | released | 50 / 50 |
| `triton/flush/fleet/folded_offdiag` | flush | released | released | 49 / 49 |
| `triton/flush/env_probe/complex_beta_fused_electric_pair` | flush | released | released | 46 / 46 |
| `triton/flush/fleet/bfast_fused_magnetic_pair` | flush | released | released | 44 / 44 |
| `triton/flush/fleet/probe_triton_folded_deposit_closure` | flush | released | released | 90 / 90 |
| `triton/flush/fleet/folded_beta_fused_electric_pair` | flush | released | released | 49 / 49 |
| `triton/flush/fleet/bfast_fused_electric_pair` | flush | released | released | 45 / 45 |
| `triton/flush/fleet/offdiag` | flush | released | released | 46 / 46 |
| `triton/flush/fleet/beta_fused_electric_pair` | flush | released | released | 46 / 46 |
| `triton/flush/fleet/probe_triton_folded_fused_magnetic_pair` | flush | released | released | 42 / 42 |
| `triton/flush/fleet/no_pml_fused_electric_pair` | flush | released | released | 43 / 43 |
| `triton/flush/fleet/beta_fused_magnetic_pair` | flush | released | released | 45 / 45 |
| `triton/flush/t2_rest/probe_triton_fused_dispersive_chain` | flush | released | released | 93 / 93 |
| `triton/flush/env_probe/complex_fused_electric_pair` | flush | released | released | 42 / 42 |
| `triton/flush/env_probe/probe_triton_folded_complex_fused_magnetic_pair` | flush | released | released | 48 / 48 |
| `triton/flush/t2_rest/cylindrical` | flush | released | released | 40 / 40 |
| `triton/flush/env_probe/complex_beta_fused_magnetic_pair` | flush | released | released | 46 / 46 |
| `triton/flush/fleet/cylindrical_real_fused_electric_pair` | flush | released | released | 43 / 43 |
| `triton/flush/fleet/fused_electric` | flush | released | released | 42 / 42 |
| `triton/flush/fleet/conductive_fused_electric_pair` | flush | released | released | 43 / 43 |
| `triton/flush/fleet/probe_triton_fused_ade_state` | flush | released | released | 91 / 91 |
| `triton/flush/fleet/complex_ade` | flush | released | released | 47 / 47 |
| `triton/flush/composition/engine_route` | flush | released | released | 50 / 50 |
| `triton/flush/fleet/no_pml_constitutive` | flush | released | released | 59 / 59 |
| `triton/flush/env_probe/probe_triton_complex_fused_magnetic_pair` | flush | released | released | 42 / 42 |
| `triton/flush/env_probe/nonlinear_fused_magnetic_pair` | flush | released | released | 42 / 42 |
| `triton/flush/fleet/complex_no_pml_stored_e` | flush | released | released | 49 / 49 |
| `triton/flush/composition/no_pml` | flush | released | released | 51 / 51 |
| `triton/flush/fleet/probe_triton_dispersive_fused_pair` | flush | released | released | 90 / 90 |
| `triton/flush/fleet/no_pml_conductive` | flush | released | released | 42 / 42 |
| `triton/flush/env_probe/complex_fused_ade_chain` | flush | released | released | 44 / 44 |
| `triton/flush/env_probe/complex_no_pml_conductive` | flush | released | released | 45 / 45 |
| `triton/flush/fleet/no_pml_stored_e` | flush | released | released | 42 / 42 |
| `triton/flush/t2_rest/folded_offdiag_dispersive` | flush | released | released | 46 / 46 |
| `triton/flush/fleet/probe_triton_cylindrical_real_fused_magnetic_pair` | flush | released | released | 42 / 42 |
| `triton/flush/offdiag_probe/complex_offdiag` | flush | released | released | 47 / 47 |
| `triton/flush/env_probe/complex_no_pml_curl` | flush | released | released | 42 / 42 |
| `triton/flush/fleet/special_kz` | flush | released | released | 51 / 51 |

## 10. How the pass ran

- NVIDIA, on RTX A6000 GPUs of one host (compute capability 8.6), one card per lane group: every
  leg spelled by `legs.py` (CUDA gate lanes and route lanes through `run_lane.py`; the Triton
  fleet through `drive_triton_weld_gates.run_one`, flush gates through `meep_host_import.py`;
  the hand legs and the end-to-end licence through `run_lane.py`), the route legs with
  `--timeout-factor 12 --min-timeout 3600` because they shared cards. Inputs staged from the
  round's input manifests, each sha256-checked; the tree read 0 dirty lines at every card's
  start and end.
- Metal, on one Apple M1 Max: `metal_round.py fleet` (the environment record, every
  `gate_metal_*`, the environment record), `status`, then `route`
  (`run_metal_dispatch_campaign.sh`, held residency, three lanes), each item holding the Mac's
  GPU and one CPU slot in one call; flush only (R13); the 7 Metal round inputs staged and
  sha256-checked; 0 dirty lines.
- Dry-run deviations, none changing a gate's argv or what it measures: the route stamps (i-1);
  route legs run before a bind, so the flush route legs can only be refused (i-2, i-3); the
  end-to-end legs' needs omit `flush` (bind first); lanes shared cards.

## 11. The freeze (CERT-1)

Entry 121 (`121-CERT1`) is the freeze: from here on an array-path edit voids every later
device hour. It is cut on entries 01 to 120 (tree `e60f69b4ede7555e2c9244f4c1bc3b3647eca6e9`).
**The first frozen tree was the tree after entry 122** (`122-XG`, the last pre-freeze fixes;
its bullet below), the frozen commit `2b75bef8`. The re-freeze made the tree after entry 126
the frozen tree (`126-WALLS`, the frozen commit `44232311`; its bullet below). The freeze after
per-kernel certification made the tree after entry 135 the frozen tree (`133-CERT1b`;
`134-MZ`, the Metal fix for what CERT-5 found on `44232311`; then `135-PINR`, the reviewed
retirement of the curated pins the round's records cannot hold; its bullet below). **The R22
freeze made the tree after entry 154 the frozen tree** (`153-FREEZE`, the round's CERT-1,
and `154-BOARD13`, the board's re-cut; its bullet below). **The R27 freeze makes the tree
after entry 170 the frozen tree** (`169-FREEZE`, the round's CERT-1, and `170-BOARD14`, the
board's re-cut; the last bullet of this section). Entry 122 is cut on
entry 121's tree (`051a71a2ff2f8ecf76b432bb9c2059045ba6be02`); it changes no file under
`meep_gpu/` and none of the 29 files that decide the freeze criteria (0 / 29 moved since entry
117). The numbers in the other bullets were measured on entries 01 to 119 with entry 121's
changes (tree `edabb4bffc3d91e0ca1f247fe7275fc02601ff00`), except where a line says these
bytes (entry 121's): entry 120 changes only the coverage board's three generated files and
`CHANGELOG.md`, which no leg below reads. Every number is N / D with D named.

- **The stamp: `2026-10-11_10`**, one string for both hosts and the whole round: `round.py
  launch gates|route --stamp`, `metal_round.py <stage> --stamp`, `legs.py check
  --fresh-stamp`, `legs.py bind --stamp`, and the body of the three route constants. It
  carries no hardware suffix: `fastpath.route_campaign` appends `_cc86` and
  `metal_runs.route_campaign` the GPU architecture. No name containing it existed on either
  host when it was chosen.
- **The route constants (i-1, closed).** `fastpath.DRIVER_ROUTE_FUSED_GATE =
  "dispatch_fused_route_2026-10-11_10"`, `fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE =
  "dispatch_fused_route_cuda_2026-10-11_10"` and `metal_dispatch.METAL_DRIVER_ROUTE_GATE =
  "dispatch_metal_route_2026-10-11_10"`; the campaigns run as
  `dispatch_fused_route_2026-10-11_10_cc86`, `dispatch_fused_route_cuda_2026-10-11_10_cc86`
  and `dispatch_metal_route_2026-10-11_10_applegpu_g13s`. Each module moves only that string
  and its comment. The 8 "Removed when" clauses of the pending list that named the 0.9.2
  campaigns name these; 0 ids added, 0 removed (63). Route campaign names are one-shot on all
  three tables alike: a route or preflight leg that runs and does not release, or is
  interrupted (killed, timed out, its connection lost), needs a new route stamp and so a new
  freeze. The end-to-end licence is named by `round.py --stamp`, not by a constant, and reruns
  under a fresh one.
- **The route stage's watchdog.** `round.py launch route` gives every route and licence leg
  `max(3600 s, 12 x its expected wall)` (`--route-min-timeout`, `--route-timeout-factor`), the
  values this pass ran those legs with (section 10), in place of `run_lane.py`'s default
  `max(1800 s, 4 x expected)`, which sat below walls this pass measured on shared cards
  (Triton `shipped` 3,180 s, `flush` 3,065 s). A leg its watchdog kills has used its one-shot
  name. On the 0.9.2 prices the watchdogs were 3,600 s to 13,044 s per leg; entry 122 prices
  the expected walls at this pass's measured walls, which makes them 3,600 s to 38,172 s, at
  least 12.0 x each leg's measured wall. The gate stage is unchanged.
- **The array-path oracle.** `record_array_path_oracle.py --report` on these bytes: fixture
  set `real-v6`, fingerprint
  `1ab516279d74002ad997a1e37e713b44d526395fa7cf41b1c63169515e925d0a`, 20 / 20 case digests
  equal on the Apple M1 Max (NumPy 2.4.3) and on x86-64 Linux (NumPy 2.2.6), and equal to
  entries 61 and 85. The fingerprint did not move, so `FIXTURE_SET` is not bumped. Writing the
  three kernel ledgers' `array_path_oracle` leaves is a certification event that follows the
  round on these bytes (the tool's contract; the pending block "CERT-4 records", 3 ids), so
  the ledgers still record `real-v1` and are not written here; `meep_identity.json` records
  the live run in both sets.
- **`meep_identity.json`, re-minted by `record_meep_identity.py`, 8 / 8 cases per set, one
  invocation per set.** Keep, on the Apple M1 Max against the contract-off MEEP 1.33.0
  single-precision identity reference: 39,991,116 stored and 65,610,420 reader words equal to
  MEEP's (178 s). Flush, on x86-64 Linux against the Linux MEEP 1.33.0 single-precision
  reference (MEEP's flush on import), into the keep-minted fixtures: 39,991,116 and
  65,610,420 equal (205 s); the keep leaf and the keep files were byte-identical after it.
  Every `digests.json` and currents file is unchanged; the 8 keep capsules change (they record
  the array-path modules as they are now), the 8 flush capsules are byte-identical, and the
  fixtures hold 4,633,671 of the 5,000,000 bytes R6 allows. Every row of both sets replayed
  with MEEP not importable holds: 16 / 16 rows on the Apple M1 Max (flush rows under FZ) and
  16 / 16 on x86-64 Linux (flush rows under the MXCSR lever), 2,916 / 2,916 stored and
  2,916 / 2,916 reader digests per set on each host, each planted flip caught and placed. The
  Tier A test file reads 62 / 62 on the Apple M1 Max after the keep mint and, NumPy only with
  no MEEP, 62 / 62 entering in keep and 62 / 62 entering in flush on x86-64 Linux.
- **The spine byte-neutrality baseline, re-baselined (i-4, owner default).** No ruling
  arrived, so the default applies: re-baseline at CERT-1. The baseline is not in the package:
  it is `results/metal_pml_2026-08-14/provenance.json` in the evidence archive, which every
  Mac checkout's `results/metal_pml_2026-08-14` links to. The gate has no write mode; its
  producer, `gate_metal_pml.provenance()`, run on these bytes (no kernel compiled or
  launched), gave the `kernel_sources` block (the 18 specialised Metal sources, all 18 moved
  since 2026-08-14), and the 2026-08-14 `sources` block was kept: the gate's refactor-present
  clause requires a tranche-1 host module (`shaders.py`, `coverage.py`, `launch.py`,
  `subnormal.py`) to differ from that block, so a whole-file re-baseline would leave it
  VACUOUS for good. The new file (sha256
  `bc1269c8dd6ffff853acbed33345be09faa55da72fe8dbf7f02ec028c2235d45`) names both origins in
  its `rebaseline` block; the 2026-08-14 file (sha256
  `dbe7fdcf1ff56eb7099b96353ec89f7d94e722daf31e684e8c14d9e8fde0695f`) is kept beside it as
  `provenance.json.2026-08-14`. On these bytes, against the archive's file, the gate reads
  passed and released: 18 / 18
  certified kernel sources unchanged, 4 / 4 tranche-1 host modules moved; against the
  2026-08-14 file it reads FAILED (18 / 18 moved), as in section 2. The input manifest of the
  2026-10-04 Metal round (`rounds/metal_2026-10-04/INPUTS_SHA256SUMS`, line 7) stays the
  record of that round and no longer verifies this file, by design: a Mac that stages its
  inputs from a bundle needs the re-baselined file.
- **The coverage board, re-cut** by its unchanged generator (`build_parity_board.py
  outcomes`, `build`) on this entry's stage, whose subjects the entry moves (the pending list,
  the re-minted fixtures, the three route modules and the new top-level harness test): every
  cell as at entry 120 (section 4's table; 0 transitions), 213 / 213 per-file outcome records
  equal, subject `47170e12428f9ef2` (1,119 files); `build_parity_board.py check` passes on
  these bytes. Entry 122 re-cuts it again (its bullet).
- **The suites** (entry 122's are in its bullet), on this entry's stage before its board
  re-cut and this section's final text
  (neither moves a file either suite reads), Apple M1 Max, every marker, dispatch off: the
  package suite under the CPU-only shim 221 F / 14,751 P / 851 S, the harness suite 0 F /
  1,981 P / 20 S, the 92 device-visible files on the Apple GPU 13 F / 4,016 P / 331 S. Against
  entry 119's: 11,732 / 11,732 package ids (less the device files), 1,998 / 1,998 harness ids
  and 4,360 / 4,360 device ids read the same outcome, plus this entry's 3 new tests, which
  pass. `tools/ci/check_failures.py`: 62 failed of 17,541 run, 62 / 62 declared (63 entries),
  0 undeclared. Needle anchors 0 differences; 200 mutation anchors, 0 stale.
- **The freeze criteria**, each on these bytes or by byte check. Against entry 117
  (`ee0ed45d`), on which L1′ and AP9 were last re-run, 0 of the 29 files that decide them
  moved: the 18 array-path files (`array_path_oracle.PROVENANCE_MODULES`, `capsule.py`,
  `array_path_oracle.py`, `subnormal_policy.py`, `backends.py`, `readback.py`) and AP9's 11
  harness files.
  - **AP8's exit**: Table 2 holds 189 / 194 rows on each of the four references (57 / 60
    examples, 132 / 134 tests; differ 0, error 0; 4 refused by name, 1 excluded by name;
    section 6), measured on entry 115's tree. Of the 289 package modules and identity-harness
    files it depends on, 4 moved since: Table 2's page (its provenance paragraph, entry 118)
    and the three route modules (only the constant and its comment; no Tier C row reads a
    route constant). Entry 118 re-measured the rows that need the corpus libraries, with a
    control, in the environments as they are now (16 / 16 row-modes), and the two mints above
    ran Tier C on these bytes on 8 cases per host, 4 of them corpus rows, with 0 differing
    words.
  - **L1′ green** (CuPy equals NumPy after every sub-step, D22/AP7), re-run on `ee0ed45d` on
    one RTX A6000 under keep and flush: 118 cases, 0 / 26,176,530,098 words differing per
    policy, the planted one-bit flip caught 118 / 118, the mixed-policy control differing
    2,446,768 / 6,272,469 words (10 / 10 cases), `test_cupy_array_path.py` 20 / 20 on the
    device, and the hand references on CuPy 0 / 197,902,068 words per policy.
  - **AP9 green** (the hand references re-validated, D23), re-run on these bytes on the Apple
    M1 Max: `validate_pml` 308 / 308 (44 skipped), `validate_no_pml` 112 / 112 (regrouped
    control differs 56 / 56, derivation 28 / 28), `validate_fold` 8 / 8,
    `validate_folded_complex` 17 / 17 (0 NaN words), `validate_folded_offdiag` 16 / 16 with
    refusals 13 / 13; `record_numpy_reference_weld.py` has nothing to write; needle anchors 0
    differences; 200 mutation anchors, 0 stale. Its CuPy half ran with L1′.
- **Entry 122 (`122-XG`), the last pre-freeze fixes** (work package XG): harness, round
  drivers, documents and the pending list's comments only. No shipped module, kernel,
  predicate, plan, gate decision, ledger, needle or mutation anchor changes, no refusal is
  relaxed and no pending-list id moves. Section 8 records the three items it closes.
  - **Route watchdogs from measured walls.** `legs.py`'s `ROUTE_CUDA`, `ROUTE_TRITON` and
    `E2E_EXPECT` carry each route and licence leg's wall in this pass, rounded up, where they
    carried 0.9.2 walls (Triton `shipped` 450 s against 3,181 s measured). With entry 121's
    `max(3600 s, 12 x expected)`, computed from `round.py`'s own route plans, every one of the
    14 route and licence legs with a measured wall of 1 s or more gets at least 12.0 x that
    wall (on the 0.9.2 prices Triton `shipped` got 1.70 x and `flush` 1.76 x).
    `metal_round.py`'s route campaign watchdog is `ROUTE_TIMEOUT_FACTOR` (3) x the serial sum
    of its four legs' measured walls (`ROUTE_EXPECT`), at least `ROUTE_MIN_TIMEOUT`: 53,760 s,
    3.0 x the serial wall and 4.25 x the measured campaign (a fixed 21,600 s before, 1.71 x),
    with `--route-timeout-factor` and `--route-min-timeout`. Every Metal round child runs with
    `PYTHONNOUSERSITE=1`, and the preflight refuses a child interpreter that enables the user
    site. `test_round_route_watchdog.py` (3 to 8 tests) holds every route and licence watchdog
    at 3 x or more of the walls it records. `RUNBOOK.md` section 2 prices CERT-4 at these
    walls: 12,733 s on the CUDA lane (Triton 9,235 s), and for Metal the measured 12,648 s on
    3 lanes (17,920 s serial).
  - **The CUDA board's detector control and Bloch cells.** `build_cuda_fusion_matrix.py`
    follows a prelude a kernel module imports from a sibling and concatenates into a kernel
    source, so the control `cylindrical_kernels.py` reads STENCIL and the four controls
    reproduce; `fused_update_{H,E}_pml_complex_bloch` are measured on
    `complex_emitter.complex_source`, the text the shipped launcher compiles, and read
    POINTWISE. Of the 53 verdicts the board can report, these 3 move.
    `test_cuda_board_detector.py` (new, 10 tests) pins them, each against a planted error.
  - **The Metal composition matrix.** The four cylindrical complex rows pin the curl's by-name
    Courant refusal, and four `_f32_courant` twins at the float32 word pin all four slots
    composed, listed under `torch` in `tools/ci/declared_resources.txt` beside their siblings:
    `test_metal_planner_composition.py` 153 / 153 with the evidence archive and
    `probe_metal_planner_composition.py` VERDICT PASS (114 configurations, 0 disagreeing).
  - **The coverage board**, re-cut by its unchanged generator: every cell as at entry 121
    (0 transitions), subject `d24d19280e6470d9` (1,120 files); `check` passes.
  - **Measured by the work package** on its code-final tree (`5dc97d7b`; the frozen tree
    differs from it only in comments, documents, the pending list's comment block and the
    coverage board's files), Apple M1 Max, every marker, dispatch off: the package suite under
    the CPU-only shim 225 F / 14,751 P / 851 S, the harness suite 0 / 1,996 / 20, the 92
    device-visible files on the Apple GPU 13 / 4,020 / 331. Against entry 121's reports,
    15,823 / 15,823 common package ids, 2,000 / 2,000 harness ids and 4,360 / 4,360 device ids
    read the same outcome. The new ids: the 4 twins in the package suite, which fail where the
    CPU-only shim hides the Apple GPU, as their `torch` siblings do, and pass on the device;
    16 harness tests, all passing (entry 121's watchdog test at 1 x is replaced by the one at
    3 x). `check_failures.py`: 62 failed of 17,560 run, 62 / 62 declared, 0 undeclared.
    Needle anchors 0 differences; 200 mutation anchors, 0 stale.
  - **Open at the freeze, resolved for CERT-4 without a code change** (no dispatch is
    affected: both products are `INSTALLABLE = False`): on a CUDA census cut on these bytes,
    `build_cuda_fusion_matrix.py` passes its four controls and then refuses at
    `h_to_d_seam.price`, because 7 rows are filed served on a seam that holds the withdraw
    with no `served_over_the_withdraw` licence (5 by `cuda_fused_hd_pair`, 2 by
    `cuda_cylindrical_fused_hd_pair`). The withdraw is the PROBE's, not the driver's: the
    boards read `h_to_d_seam_2026-09-04`, cut on the driver before entry 25, where those
    rows' integrated electric sources carried one; on these bytes no source carries one,
    `driver.py` calls none, and the predicates admit the rows for that reason. Re-cut on the
    frozen tree (`h_to_d_seam_2026-10-11_8ac75ea5`, the unchanged `probe_h_to_d_seam.py`),
    the probe measures 0 withdraw rows of 192, and with it the CUDA board on a scratch census
    cut on these bytes completes: served 531 of 591, `served_in_dispatch` 347, the 7 rows
    move from `withdraw_seam` to served and nothing else on the H->D seam moves. The boards
    read it through `MEEP_GPU_H_TO_D_PROBE` (`RUNBOOK.md`, CERT-4 records). The H->D gates
    lifted 5 of the 7 rows on the device, the weld equal word for word to the certified singles
    (keep and flush, on `b676c5f5`); `differential_cross_section.py` and `mie_scattering.py`
    did not lift on that host (`measured=False`).
    Re-cut again on the re-frozen tree (`h_to_d_seam_2026-10-12_729b161c`, 0 withdraw rows of
    192), the probe the 1.0 round's boards read: its CUDA board, on a census cut on those
    bytes, served 531 of 591 with 347 in dispatch.
- **The re-freeze (entries 123 to 126; owner rulings 2026-10-11).** The first frozen commit
  `2b75bef8` (tree `8ac75ea542f580c9fd34fedca9a7bde6903ae363`) carried two defects that keep
  the required CUDA route leg `cuda_flush` from releasing, and a route leg that runs and does
  not release spends its one-shot name. The NVIDIA route lane therefore stopped before
  launching anything, after the CERT-3 gates and the bind on both hosts and the Metal route
  had run: the stamp `2026-10-11_10` is unused on the NVIDIA route, and none of its 5 route and
  licence roots exists on the GPU host. With the bind's evidence, the default-precedence read
  of `cuda_flush`'s arbitration control plans Triton's `fused pair B` / `fused pair D`, whose
  gate certified no subnormal policy, and rung 8b refuses that whole plan; the pre-certification
  pass's plans predicted it on 11 of the 37 cases that dispatch, `pml_2d` among them. A scratch
  dry run of `cuda_flush` on one RTX A6000, on the frozen commit with the rebound ledgers,
  confirmed it (`REFUSED` on 3 of the 4 cases that reached arbitration) and found F2 (section
  8): rung 8b read a bare label in a CUDA-led plan against the CUDA ledger, which failed
  `dispersive_2d` and the `nonlinear_3d` envelope control with or without the first fix. The
  owner ruled both fixed in 1.0. **Entry 123** (`123-FXE`, fix B): `gate_triton_fused_electric.py`
  installs, stamps and confirms the campaign's policy, and the ledger entry
  `triton_fused_electric_device_gate` no longer declares `_mixed_policy`; it is cut on
  `8ac75ea5` and gives `3da5bfe3ac97f259e282c60d8e127c1923674429`. **Entry 124** (`124-F2X`,
  F2): `fastpath._table_for_label` answers a bare label in a plan composed on the CUDA table
  from the Triton ledger, `metal_dispatch.METAL_DRIVER_ROUTE_GATE` moves to
  `dispatch_metal_route_2026-10-12_10`, and this bullet; it is cut on `3da5bfe3`. **Entry 125**
  (`125-HW`, the hardware-scope rename, owner ruling 2026-10-11): `MEEP_GPU_HARDWARE` =
  `certified` | `supported` | `any` replaces `MEEP_GPU_ALLOW_UNCERTIFIED` (`0` | unset | `1`),
  which is now refused by name, set to anything, naming the value that does what it did; each
  value admits exactly what its old spelling admitted, on every table, so no admission moves.
  It is cut on the tree after entry 124 (`02daf8364c1785c588118ca31cee4dc23e235b2a`) and
  amends this bullet. **Entry 126** (`126-WALLS`, cut on `b1917f71`) prices the CERT-3 NVIDIA
  re-run at CERT-3's measured walls, round-driver data and docs only, and voids nothing: no
  ledger's curated pin names a file it changes. **The frozen tree is the tree after entry 126**;
  its id is in the series log and the round's frozen-tree record, since this file is part of
  it. The NVIDIA route constants keep `_2026-10-11_10`.
  - **The freeze criteria, by byte check against the first frozen tree**: 0 of the 18
    array-path files moved, and no Metal kernel source or host module, so the array-path
    oracle, `meep_identity.json`, the spine baseline and L1′ stand. Of the 289 files AP8's
    Table 2 depends on, the same 4 as at the freeze have moved since entry 115: `fastpath.py`
    and `metal_dispatch.py` again (the lookup and the constant), and Table 2's rows lift with
    `prefer_gpu=False`, the NumPy reference, which consults no kernel table. Of AP9's 11
    harness files 1 moved, `gate_triton_fused_electric.py` (fix B: the policy install, stamp
    and confirmation; no case, mutation or verdict rule), and AP9's NumPy half re-ran on these
    bytes: `validate_pml` 308 / 308 (44 skipped), `validate_no_pml` 112 / 112 (regrouped
    control differs 56 / 56, derivation 28 / 28), `validate_fold` 8 / 8,
    `validate_folded_complex` 17 / 17 (0 NaN words), `validate_folded_offdiag` 16 / 16 with
    refusals 13 / 13; `record_numpy_reference_weld.py` has nothing to write; the array-path
    oracle `real-v6` fingerprint `1ab51627` unchanged (20 cases). Its CuPy half for that gate
    is the gate's own keep and flush runs, owed below (in scratch on one RTX A6000 both
    released, 56 / 56 base cases bit-identical to the array path and to two kernels, 10 / 10
    mutations caught).
  - **What is re-run on the re-frozen commit** (section 8 gives the commands):
    - **Metal**, under the Mac stamp `2026-10-12_10` on a clean checkout: the whole fleet
      (`metal_round.py all`), then `rebind_metal_welds.py --stamp 2026-10-12_10 --write`. The
      ledger alone owes the three welds that pin `fastpath.py`; the Metal board owes all 62,
      because every gate artifact of `_2026-10-11_10` records `meep_gpu/fastpath.py` among its
      imports and the board re-hashes the artifacts it credits. Then the route, **the new Metal
      route stamp**: `metal_round.py route --stamp 2026-10-12_10` into
      `dispatch_metal_route_2026-10-12_10_applegpu_g13s` (`_2026-10-11_10` has run; its record
      pins the moved `fastpath.py` and `metal_dispatch.py`), and `recut_driver_dispatch_record.py
      --backend metal` on that campaign.
    - **Triton**: the `fused_electric` fleet gate under keep and under flush, then the Triton
      rebind of `triton_fused_electric_device_gate` from those two runs (fix B; its pending id
      clears there); the bit identity under keep and flush, then `--bind-bit-identity`
      (`bit_identity_gate` pins `fastpath.py`).
    - **The NVIDIA route and the licence**: a scratch dry run of the two flush legs
      (`cuda_flush`, `flush`) on the re-frozen bytes, then the one-shot route under the unused
      stamp `2026-10-11_10`; both NVIDIA `driver_dispatch` records are cut there (neither was
      cut on the first frozen bytes).
    - **The boards**: the coverage board is re-cut in entry 124 and again in entry 125 (below);
      the Metal board after the Metal fleet, its release bindings moved to `_2026-10-12_10`; the
      Triton board's gate-bound rows and the CUDA board's gate-tier columns after the ruling
      below.
  - **What stands, by byte check of the ledgers' curated pins.** Every 64-hex digest in the
    ledgers the bind wrote (the evidence host's checkout, uncommitted) was compared with the 12
    files entries 123 and 124 change, at the first frozen bytes and at these:
    - Triton: 75 of 79 entries pin no changed file. 2 are voided: `bit_identity_gate` (runs
      `triton_bit_identity_2026-10-11_10` and `_flush`) and `triton_fused_electric_device_gate`
      (`triton_fleet_2026-10-11_10`'s `fused_electric`). 2 already disagreed with the first frozen
      bytes, so the re-freeze changes nothing about them: `driver_dispatch`, never cut on them,
      and `fused_electric_gate`'s probe pin, a 2026-10-03 record no arm cites. So the NVIDIA
      CERT-3 runs stand except the bit identity and `fused_electric`: the other 16 Triton roots
      of the stamp (both fleets and their probes and tranches, both composition roots, both
      stencil-weld roots) are cited by entries that stand (`triton_fleet_2026-10-11_10` also
      by the voided `fused_electric` entry).
    - CUDA: 54 of 55 fingerprint entries and 110 of 110 certification entries stand
      (`driver_dispatch` already disagreed), and the CUDA bind's inputs,
      `cuda_regate_2026-10-11_10_g0` and `_g1`, are cited only by entries that stand.
    - Metal: 60 of 64 entries stand; voided are the route record (`driver_dispatch`, which pins
      `fastpath.py` and `metal_dispatch.py`) and the three welds above. So the Metal CERT-3
      runs stand by pins except those three welds.
  - **Entry 125 voids nothing beyond what entry 124 voids.** Of the 42 files the rename changes
    (the entry also changes this section and the coverage board's three files, which no pin
    names), the ledgers' curated pins name 4: `fastpath.py` and `metal_dispatch.py`, which
    entry 124 moves too, `fastpath_cuda.py` (one comment line) and
    `gate_dispatch_fused_route.py` (the route gate refuses a leg with either name exported).
    Every 64-hex digest in the ledgers the bind wrote (the evidence host's checkout,
    uncommitted) and in the committed ledgers was compared with the files this entry changes,
    at entry 124's bytes and at these: no entry is voided. The 7 entries that pin a changed
    file are the Triton `bit_identity_gate` and `driver_dispatch`, the CUDA `driver_dispatch`
    (the only entry that pins `fastpath_cuda.py` and the route gate), and the Metal
    `driver_dispatch` and the three welds above, and none of them pins entry 124's bytes
    either: entry 124 voids the Triton bit identity and the four Metal entries, and the two
    NVIDIA `driver_dispatch` records were never cut on the frozen bytes. Of the bind's ledgers,
    Triton 77 of 79, CUDA 54 of 55 and 110 of 110, and Metal 60 of 64 entries pin no file the
    entry changes (of the committed ones, Triton 78 of 80, CUDA 54 of 55 and 108 of 108, Metal
    60 of 64). By import record it voids none either: every gate artifact with a source record
    already names `fastpath.py`, which entry 124 moves (Metal 62 of 62, NVIDIA 246 of 246; the
    open ruling below), so a tool that compares a run's whole import record with the tree
    refuses the same runs with this entry as without it. Against the first frozen tree, with
    entry 125's files added to the 12: Triton 75 of 79 entries stand, CUDA 54 of 55 and 110 of
    110, Metal 60 of 64, with the same 2 Triton and 4 Metal entries voided: the counts above
    are unchanged. The freeze criteria, by byte check against the tree after entry 124: 0 of
    the 18 array-path files and 0 of AP9's 11 harness files moved (since entry 117, 1 of the
    29, `gate_triton_fused_electric.py`, as at entry 124), so the array-path oracle,
    `meep_identity.json`, the spine baseline and L1′ stand; of the 289 files AP8's Table 2
    depends on, 3 moved (`fastpath.py`, `fastpath_cuda.py` and `metal_dispatch.py`), and since
    entry 115 the same 4 as at the freeze. On these bytes the oracle reads fixture set
    `real-v6`, fingerprint `1ab51627` (20 cases), and AP9's NumPy half passes: `validate_pml`
    308 / 308 (44 skipped), `validate_no_pml` 112 / 112, `validate_fold` 8 / 8,
    `validate_folded_complex` 17 / 17, `validate_folded_offdiag` 16 / 16 with refusals 13 / 13;
    `record_numpy_reference_weld.py` has nothing to write. The round's runs on the re-frozen
    commit export neither name: the route gate and `legs.py check` refuse a route leg with
    either exported, `run_lane.py` and `metal_round.py` scrub both, the Metal preflight refuses
    either, and `drive_triton_weld_gates.py` runs its gate children with
    `MEEP_GPU_HARDWARE=certified`; a shell or environment file that still exports
    `MEEP_GPU_ALLOW_UNCERTIFIED` sees the named refusal on every GPU run.
  - **Open, an owner ruling: the NVIDIA bind on these bytes.** By import record the picture
    differs: every gate artifact with a source record names `meep_gpu/fastpath.py` (Metal 62
    of 62, NVIDIA 246 of 246), and the tools that compare a run's whole import record with the
    tree refuse those runs here. Measured with `legs.py bind`'s own commands in report mode
    against the round's roots: the Triton rebind skips 11 of 54 welds (3 on the tree after
    entry 123), the family re-certification refuses all 9 families of `family_recert_2026-08-14`
    (flush evidence for 22 arms), and the bit-identity binder refuses; the CUDA rebinds, the
    ADE re-cut and the composition records read the same on both trees. Either the bind
    written on the first frozen bytes is carried to the re-frozen commit (no curated pin of
    those entries names a changed file) or those gates are re-run on it. Either way the Triton
    rebind on these bytes runs once, from the committed Triton ledger, never on top of the
    copy the first bind wrote: a second `--write` nests one records line a level deeper.
  - **Measured on the tree after entry 124** (Apple M1 Max, every marker, dispatch off):
    - **Suites**, on the tree of entries 123 and 124 before this section's text and the board's
      re-cut (neither moves a file either suite's outcome reads but the board's own tests): the
      package suite under the CPU-only shim 226 F / 14,790 P / 851 S, the harness suite 0 /
      2,005 / 20, the 92 device-visible files on the Apple GPU 13 / 4,020 / 331. Against the
      work package's own reports every id reads the same outcome: 15,867 / 15,867 package ids,
      2,025 / 2,025 harness ids and 4,364 / 4,364 device ids, 0 new and 0 gone.
      `tools/ci/check_failures.py`: 63 failed of 17,609 run, 63 / 63 declared (64 entries), 0
      undeclared. Needle anchors 0 differences; 200 mutation anchors, 0 stale. Re-run after the
      re-cut on this entry's tree but for this sentence: the harness suite 0 / 2,005 / 20 (the
      board's own tests and `test_reference_environment.py`, which reads the CHANGELOG, among
      them), and the four dispatch and Triton weld test files 223 passed and 7 failed of 230, 7
      / 7 declared (the package suite's 7 `test_triton_weld_contract.py` failures); needle
      anchors 0 differences; 200 mutation anchors, 0 stale.
    - **The coverage board.** Re-cut by its unchanged generator (`build_parity_board.py
      outcomes`, `build`) from those reports, the two entries having moved both its subjects
      (the pending list, two top-level harness files and five files under `meep_gpu/`): every
      cell as on the first frozen tree (0 transitions); 212 / 213 per-file outcome records
      equal, the other being `test_triton_weld_contract.py`, whose one newly failing id is
      entry 123's declared pending id; subject `1471b953384bc25f` (1,121 files); `check`
      passes; a second build equal but for `built_utc`.
    - **The GPU host.** `legs.py check` on the GPU host, on a scratch clone of its
      certification checkout (`2b75bef8`) with entries 123 and 124 applied and committed (the
      clone's tree equal to this tree): `CHECK OK` (0 problems) with `--fresh-stamp
      2026-10-12_10` and the corpus root, and `check --route` `CHECK OK`; fleet 61 / 61, the
      flush licence, cc 8.6 admitted by both tables, 8 corpus-lifting legs,
      `dispatch_by_default` true, route campaigns `dispatch_fused_route_2026-10-11_10_cc86` and
      `dispatch_fused_route_cuda_2026-10-11_10_cc86`; the control with the checkout's own HEAD
      refuses on exactly 1 problem (not the frozen commit). On the committed ledgers
      `flush-coverage` reads Triton 0 / 75 and CUDA 0 / 34 (no bind on these bytes yet); with
      the first bind's three ledgers copied in, `check --route` `CHECK OK` and Triton 73 / 75
      (lacking `triton_fused_electric_device_gate`, owed above), CUDA 34 / 34.
  - **Measured on the tree after entry 125** (Apple M1 Max, every marker, dispatch off):
    - **Suites**, on the tree of entries 01 to 122, FXE.patch, F2X.patch and RNM.patch
      (`4ec404ac`: entry 124's suite tree `e6512c6c` with the rename merged), which differs
      from this tree only in `CHANGELOG.md`, this section and the coverage board's three files
      and carries the same outcome subject (`1c2b9a2b`): the package suite under the CPU-only
      shim 226 F / 14,888 P / 870 S, the harness suite 0 / 2,031 / 20, the 92 device-visible
      files on the Apple GPU 13 / 4,137 / 331. Against entry 124's reports every id common to
      both reads the same outcome: 15,798 / 15,798 package ids, 2,014 / 2,014 harness ids and
      4,295 / 4,295 device ids. The ids that move are the rename's renamed and re-parametrised
      tests, in six files: 69 gone and 186 new in the package suite and on the device
      (`test_uncertified_opt_in`, `test_metal_environment`, `test_prefer_gpu`; 186 / 186 new
      pass on the device, 158 pass and 28 are Metal cases the shim skips in the package suite),
      and 11 gone and 37 new in the harness (`test_gpu_host_guards`,
      `test_uncertified_harness`, `test_timing_ladder_plan`; 37 / 37 pass).
      `tools/ci/check_failures.py`: 63 failed of 17,752 run, 63 / 63 declared (64 entries), 0
      undeclared. Needle anchors 0 differences; 200 mutation anchors, 0 stale. Re-run after the
      cut on this entry's tree but for this sentence: the harness suite 0 / 2,031 / 20, every
      id the same outcome as on the suites' tree (2,051 / 2,051; the board's own tests and
      `test_reference_environment.py`, which reads the CHANGELOG, among them);
      `tools/ci/check_failures.py` over it with the package and device reports 63 failed of
      17,752 run, 63 / 63 declared, 0 undeclared; the oracle, needles, AP9's validators and the
      200 mutation anchors as above; `build_parity_board.py check` passes.
    - **The coverage board.** Re-cut by its unchanged generator (`build_parity_board.py
      outcomes`, `build`) from those package and device reports; before the cut `check` refused
      on this entry's bytes (the board's subject and the working tree disagree). Every cell as
      at entry 124 (0 transitions); 212 / 213 per-file outcome records equal, the other being
      `test_metal_environment.py` (52 passed before, 55 after, 0 failed: the rename's ids);
      subject `3cf55d149a2e8706` (1,121 files); `check` passes; a second build equal but for
      `built_utc`; the development evidence 11 current, 0 stale, before and after.
    - **The GPU host.** `legs.py check` on the GPU host, on a scratch clone of its
      certification checkout (`2b75bef8`) with entries 123, 124 and 125 applied and committed
      (the clone's tree equal to this tree): `CHECK OK` (0 problems) with `--fresh-stamp
      2026-10-12_10` and the corpus root, and `check --route` `CHECK OK`, with the facts of
      entry 124 (fleet 61 / 61, the flush licence, cc 8.6 admitted by both tables, 8
      corpus-lifting legs, `dispatch_by_default` true, route campaigns
      `dispatch_fused_route_2026-10-11_10_cc86` and
      `dispatch_fused_route_cuda_2026-10-11_10_cc86`). `check --route` refuses on exactly 1
      problem with `MEEP_GPU_HARDWARE=certified` exported and on exactly 1 with
      `MEEP_GPU_ALLOW_UNCERTIFIED=0` exported, each naming the variable; the control with the
      checkout's own HEAD refuses on exactly 1 problem (not the frozen commit).
      `flush-coverage` as at entry 124: Triton 0 / 75 and CUDA 0 / 34 on the committed ledgers;
      with the first bind's three ledgers copied in, `check --route` `CHECK OK` and Triton 73 /
      75, CUDA 34 / 34.
- **The freeze after per-kernel certification (entries 127 to 135; owner ruling R17, amended
  2026-10-12: in 1.0, so the paper can cite it).** The round ran on the frozen commit
  `44232311` (tree `729b161c05cdde2d1d58903711ffba02107e0e03`): CERT-3 GO on NVIDIA and Metal,
  CERT-4's NVIDIA route and licence and its Metal route released, CERT-5's NVIDIA L3 189 / 189
  rows on both baselines and both precedences. Its records stand as the record of that pass.
  Entries 127 to 132 (the R17 batch) make a kernel gate's import record its own dependency set
  (`meep_gpu/__init__.py` binds its names on first use; `parity/meep_gpu/
  test_gate_import_closure.py` holds every gate and probe to its declared closure), bring two of
  the fixes that share the round (the lift's partition clause keyed on MEEP's built grid, entry
  129; the composers' default chunk division, the CUDA E->P weld's chunk clause and the
  CuPy-free Bloch table, entry 130), the L3 driver (entry 131) and the coverage board's re-cut
  (entry 132); the H->D start-clause narrowing is deferred to 1.0.1. Entry 128 changes the
  shape of every gate's import record, so no record of the previous pass can be re-bound: the
  round runs once in full on the tree after entry 135. Entry 133 (`133-CERT1b`) is this freeze;
  it is cut on the tree after entry 132 (`deb1354f1b2206a7d55251785e91dde6b86d95fd`). Entry 134
  (`134-MZ`) adds the one Metal fix CERT-5 found, and entry 135 (`135-PINR`), the last change
  before the round, the reviewed retirement of the curated pins the round's records cannot hold
  (their sub-bullets below).
  - **The stamp: `2026-10-13_10`**, one string for both hosts and the whole round (`round.py
    launch gates|route --stamp`, `metal_round.py <stage> --stamp`, `legs.py check
    --fresh-stamp`, `legs.py bind --stamp`, `rebind_metal_welds.py --stamp`,
    `repoint_board_bindings.py --stamp`) and the body of the three route constants. No name
    containing it existed on either host when it was chosen. `2026-10-11_10` and
    `2026-10-12_10` are spent; their roots stay the record of the passes that used them.
  - **The route constants.** `fastpath.DRIVER_ROUTE_FUSED_GATE =
    "dispatch_fused_route_2026-10-13_10"`, `fastpath_cuda.CUDA_DRIVER_ROUTE_FUSED_GATE =
    "dispatch_fused_route_cuda_2026-10-13_10"` and `metal_dispatch.METAL_DRIVER_ROUTE_GATE =
    "dispatch_metal_route_2026-10-13_10"`; the campaigns run as
    `dispatch_fused_route_2026-10-13_10_cc86`, `dispatch_fused_route_cuda_2026-10-13_10_cc86`
    and `dispatch_metal_route_2026-10-13_10_applegpu_g13s`, and the licence as
    `dispatch_end_to_end_2026-10-13_10`. Each module moves only that string and its comment.
    The pending list's 8 "Removed when" clauses name these, with one comment block; 0 ids added,
    0 removed (64). Route campaign names stay one-shot.
  - **`meep_identity.json`, re-minted by `record_meep_identity.py`**, because a minted input
    moved: of the 20 files the tool hashes, `meep_gpu/from_meep.py` (entry 129) differed from
    both sets' record, 19 of 20 equal. Keep, on the Apple M1 Max against the contract-off MEEP
    1.33.0 single-precision identity reference (201 s), and flush, on x86-64 Linux against the
    Linux MEEP 1.33.0 single-precision reference (MEEP's flush on import; 159 s, into the
    keep-minted fixtures), 8 / 8 cases per set, one invocation per set: 39,991,116 stored and
    65,610,420 reader words equal to MEEP's in each; the keep files byte-identical after the
    flush mint (32 / 32). Of the ledger's 1,841 leaves 70 move and none is a digest, a word count
    or a verdict: times (35), the 16 capsules' byte counts and sha256 (30; a capsule records the
    lift module's digests, and those two are the only fields of its document that move), the
    two sets' `from_meep.py` digests (4) and the fixture bytes (4,633,685 of the 5,000,000 R6
    allows). Every row of both sets replayed with MEEP not importable holds: 16 / 16 rows on the
    Apple M1 Max (flush rows under FZ) and 16 / 16 on x86-64 Linux (flush rows under the MXCSR
    lever), 2,916 / 2,916 stored and 2,916 / 2,916 reader digests per set on each host, the
    planted flip caught and placed in 8 / 8 rows per set on each host. The Tier A test file
    reads 62 / 62 on the Apple M1 Max after the keep mint and, NumPy only with no MEEP, 62 / 62
    entering in keep and 62 / 62 entering in flush on x86-64 Linux.
  - **The array-path oracle.** `record_array_path_oracle.py --report` on these bytes: fixture
    set `real-v6`, fingerprint `1ab516279d74002a`, 20 / 20 case digests equal on the Apple M1
    Max (NumPy 2.4.3) and on x86-64 Linux (NumPy 2.2.6). It did not move, so `FIXTURE_SET` is
    not bumped; the kernel ledgers' leaves are written after the round (the pending block
    "CERT-4 records"), as at the first freeze.
  - **The coverage board, re-cut** by its unchanged generator on this entry's bytes, whose
    subjects the entry moves (the three route modules and the fixtures; the pending list):
    every cell as at entry 132 (0 transitions), 214 / 214 per-file outcome records equal,
    outcome subject `169ba1613dc995c3` (1,124 files), board subject 1,135 files; `check`
    passes; a second build equal but for `built_utc`; the development evidence 11 current, 0
    stale, before and after.
  - **The suites**, on this entry's bytes before its text (which no suite's outcome reads but
    the reference-environment test's, re-run after it), Apple M1 Max, every marker, dispatch
    off: the package suite under the CPU-only shim 226 F / 14,910 P / 871 S, the harness suite
    0 / 2,152 / 20, the 94 device-visible files on the Apple GPU 13 / 4,151 / 331. Against
    entry 131's reports every id reads the same outcome: 16,007 / 16,007 package ids (11,781 /
    11,781 less the device files), 2,172 / 2,172 harness ids and 4,495 / 4,495 device ids, 0
    new and 0 gone. `tools/ci/check_failures.py`: 63 failed of 17,896 run, 63 / 63 declared (64
    entries), 0 undeclared. Needle anchors 0 differences; 204 mutation anchors, 0 stale.
  - **The freeze criteria.** Against entry 131 (`dbca770023f978c9ca28b4cb8ebaa776eb2ee034`), on
    which L1′ and AP9 re-ran green, 0 of the 29 files that decide them moved (against the
    previous frozen tree 1 of 29: `from_meep.py`, entry 129).
    - **L1′ green** on entry 131, on one RTX A6000 under keep and flush: 123 case-rows over the
      cylindrical, family (67 capsules lifted on that tree), oracle, hand and tiny legs, 0 /
      26,302,894,838 words differing per policy after every sub-step, the planted flip caught
      123 / 123, the mixed-policy control differing 2,446,768 / 6,272,469 words (10 / 10
      cases); the hand references on CuPy 0 / 197,902,068 words per policy;
      `test_cupy_array_path.py` 20 / 20 on the device.
    - **AP9 green**, re-run on these bytes on the Apple M1 Max: `validate_pml` 308 / 308 (44
      skipped), `validate_no_pml` 112 / 112 (regrouped control differs 56 / 56, derivation 28 /
      28), `validate_fold` 8 / 8, `validate_folded_complex` 17 / 17 (0 NaN words),
      `validate_folded_offdiag` 16 / 16 with refusals 13 / 13; `record_numpy_reference_weld.py`
      has nothing to write; needle anchors 0 differences; 204 mutation anchors, 0 stale. Its
      CuPy half ran with L1′.
    - **AP8's exit is not settled at the freeze.** Of the 290 package modules and
      identity-harness files Table 2 depends on, 14 moved since the previous frozen tree, the
      lift (`from_meep.py`, entry 129) among them. Entry 129 measured its two rows on all four
      references with a 20-row sample (Table 2 191 / 194 on each), not the whole corpus. The
      corpus Tier C therefore re-runs in full on all four references (Apple keep and flush,
      x86-64 Linux keep and flush, 194 rows each), launched when this entry landed and run
      beside the round's device stages, CPU only; a row that does not hold where Table 2 holds
      stops the round before CERT-4.
  - **Entry 134 (`134-MZ`), the Metal start of H, found by CERT-5 on the previous frozen commit.**
    CERT-5's Metal L3 on `44232311` held 188 / 189 corpus rows against this host's flush MEEP; the
    one row that differed against every reference, `ex.differential_cross_section`, stored +0 in
    240 H words at step 1 where MEEP and the array path store -0 (sign of zero only, absent from
    step 2 on), with the fused magnetic pair and with the singles alike; the NVIDIA tables held
    it. MEEP starts H and its `f_w` as copies of B at the run's first `update_eh(H_stuff)` on the
    chunks that carry H's own layer (`update_eh.cpp:165-177`), as the array path does
    (`stepping.separate_h_from_b`). No Metal kernel starts H, and the composition admitted one on
    that update on the premise that the first `step_B` leaves B at +0, which fails where a strong
    layer's outer `kms` is negative (B lands on -0 there). The entry changes the Metal composer
    only: `launch.plan_step` puts `launch.StartOfHPlan` around whatever serves a run's first
    `update_H` (the start on the array path before a single's launch; `stepping.update_H`
    re-stepped after a fused magnetic pair's), as the hand-CUDA launchers carry it and the Triton
    kernels carry it in their own statements. No shader byte, array-path module, minted input,
    ledger, needle or mutation anchor changes, so the freeze criteria, `meep_identity.json`, the
    array-path oracle, L1′ and AP9 stand as above (0 of the 29 files that decide L1′ and AP9 move;
    of the 290 files Table 2 depends on, 1, `metal_kernels/launch.py`, a kernel-table module), and
    no Tier A replay is owed. The corpus Tier C run launched at this freeze lifts with
    `prefer_gpu=False` and loads none of the entry's files (measured on these bytes: a lift and
    four steps of a 2-D and a 3-D PML run load 22 repository files, no `metal_kernels` module
    among them), so it stands for these bytes. `gate_metal_pml` gains leg `first_update_h` and
    `gate_metal_fused_magnetic_pair` a `first_update_h` row; on these bytes both release, the
    composition equals the array path with 0 differing words at steps 1 and 2 and the start run
    once, and with the start stripped it differs on 48 H words at step 1 (16 each in Hx, Hy and
    Hz). L3M's instrument on these bytes: `ex.differential_cross_section` holds against all three
    reference sets, 372 / 372 stored and 372 / 372 reader digests each (it held 369 / 372 stored
    on the previous frozen commit), dispatched and fused; a 12-row sample covering every
    `update_H` shape the corpus dispatches holds 12 / 12 against the two flush sets (4,416 / 4,416
    stored and reader digests each), every verdict as before the fix. No import closure moves (250
    / 250 gate and probe scripts; the new harness module joins the run closures of 10 Metal
    scripts, and no dispatch-layer file enters any). Measured on these bytes before this text
    (Apple M1 Max, every marker, dispatch off): the package suite under the CPU-only shim 226 F /
    14,921 P / 873 S, the harness suite 0 / 2,152 / 20, the 95 device-visible files on the Apple
    GPU 13 / 4,164 / 331; against entry 133's reports every common id reads the same outcome
    (16,007 / 16,007 package ids, 2,172 / 2,172 harness ids, 4,495 / 4,495 device ids), and the
    entry's 13 new tests pass on the device (under the shim 11 pass and 2 skip by their declared
    resource); `tools/ci/check_failures.py` 63 failed of 17,909 run, 63 / 63 declared (64
    entries), 0 undeclared. Needle anchors 0 differences; 204 mutation anchors, 0 stale. The
    coverage board is re-cut, the entry having moved both its subjects: every cell as at entry 133
    (0 transitions), 214 / 214 per-file outcome records equal and one new
    (`test_metal_start_of_h.py`, 13 passed), outcome subject `852c4fdb040756de` (1,126 files),
    board subject 1,137 files; `check` passes. `legs.py check` on the GPU host with the fresh
    stamp and the corpus root `CHECK OK`, `check --route` `CHECK OK`; the Metal preflight rc 0, 0
    FAIL lines, 29 / 29. The round's bill is unchanged: it re-runs every gate on these bytes
    (`metal_kernels/launch.py` is in the import closure of 64 of the 250 scripts).
  - **What the round re-runs on the new frozen commit**, once in full (entry 128 moves every
    record): CERT-2's device smoke per table; CERT-3, every gate on all three tables (the NVIDIA
    gate lanes, both CUDA campaigns, Triton keep and flush with the ten Triton H->D gates under
    keep, and the Metal fleet of 62 gates); the bind; CERT-4, the NVIDIA route and licence and
    the Metal route under this stamp, then the records, the three censuses, the H->D probe and
    the boards; CERT-5, L3 with the committed driver; CERT-7 and CERT-8 as `RUNBOOK.md` orders.
    After this round a dispatch-only edit re-runs the declared route, dispatch, composition and
    driver gates and the records that pin the planner, and a kernel edit the gates
    `parity/meep_gpu/import_closure.py bill <file>` names.
  - **Entry 135 (`135-PINR`), closed before the round: the curated pins the round's records cannot
    hold.** 3 Metal welds (`metal_complex_no_pml_conductive` 13 / 42 pins,
    `metal_cylindrical_real_fused_magnetic_pair` 12 / 43, `metal_complex_conductive_fused_pair` 6
    / 45, less the deleted-file pin) and 4 Triton welds (`driver.py`) pinned core modules their
    gates no longer import; CUDA 0. The rebind tools retire them by `RETIRED_UNIMPORTED_PINS`,
    generated from `import_closure.py` (a module of the old eager root outside the gate's run
    closure) and reviewed; each tool refuses where the rule stops holding. The 12th Metal pin
    (`dispersion.py`) needed a sharper run closure: a harness table's stored callables are reached
    only where reached code names the table. Of the 13 Triton entries counted earlier,
    `bit_identity_gate`'s binder writes its own pin set and 8 welds pin their own test file on
    purpose. On post-R17 runs the Metal rebind binds the three welds and the Triton rebind the
    four; never a hand edit of a ledger. The entry changes harness tools (`import_closure.py` and
    the three rebind tools), one test file (`test_rebind_unimported_pins.py`, 44 tests) and text:
    no package module, kernel, gate, probe, ledger, needle, mutation anchor, minted input or
    pending-list id. None of the 250 gate and probe scripts holds a changed file in its run
    closure, so no gate's import record moves and the round's bill is unchanged; 0 of the 29 files
    that decide the freeze criteria and 0 of the 290 files Table 2 depends on move, so the freeze
    criteria, `meep_identity.json`, L1′, AP9 and the corpus Tier C run stand, and no Tier A replay
    is owed. Measured on these bytes before this text (Apple M1 Max, every marker, dispatch off):
    the package suite under the CPU-only shim 226 F / 14,921 P / 873 S, the harness suite 0 /
    2,196 / 20, the 95 device-visible files on the Apple GPU 13 / 4,164 / 331; against entry 134's
    reports every common id reads the same outcome (16,020 / 16,020 package ids, 2,172 / 2,172
    harness ids, 4,508 / 4,508 device ids), and the entry's 44 new tests pass;
    `tools/ci/check_failures.py` 63 failed of 17,953 run, 63 / 63 declared (64 entries), 0
    undeclared. Needle anchors 0 differences; 204 mutation anchors, 0 stale. The rebind tools in
    report mode on scratch copies of the committed ledgers: on post-R17 runs of the seven welds'
    gates the Metal tool rebinds 3 / 3 welds and retires 34 pins (the 31 rows and the 3
    deleted-file pins) and the Triton tool 4 / 4, retiring `driver.py` on each; a second `--write`
    retires nothing and leaves the Metal ledger byte-identical; against the 2026-10-12_10 round's
    records both skip those welds by name. The coverage board is re-cut, the entry having moved
    its outcome subject: every cell as at entry 134 (0 transitions), 215 / 215 per-file outcome
    records equal and none new, outcome subject `bc1bc9656fd217ea` (1,127 files), board subject
    1,138 files; `check` passes. `legs.py check` on the GPU host with the fresh stamp and the
    corpus root `CHECK OK`, `check --route` `CHECK OK`; the Metal preflight rc 0, 0 FAIL lines, 29
    / 29.
- **The R22 freeze (entries 141 to 153; owner ruling R22, 2026-10-13).** The export dry run of
  1.0 found host names, account names and local paths in certified bytes (the NVIDIA ledgers'
  host fields and dispatch environment, two pinned source comments, three files of the board's
  subject), and the 1.0 line had not taken the 0.9.2 release's development commits. The owner
  ruled that the source of each is fixed, that certification is re-run for 1.0, and that the
  1.0.1 items land with it. Entries 141 to 152 carry that work: the hygiene guard and
  neutral record labels (141), the 0.9.2 commits (142), the export tooling (143), three
  round-tool fixes (144), the route-stamp registry (145), the timing harness (146), the
  reference environment's test-side fixes (147, 148), the bulk permittivity read (149),
  chunked certification (150), the deposit schedule (151) and the coverage board (152).
  Entry 153 (`153-FREEZE`) is this freeze, cut on the tree after entry 152
  (`0e76e9c44d6d4241988a41f1207e27221c042462`); entry 154 re-cuts the coverage board, whose subject it moves. The
  round runs in chunks (`RUNBOOK.md`, "CERT-3, chunked"): a gate chunk whose closure no longer
  changes may certify before the freeze, and an artifact whose import record moves later is
  re-run, by name, at the bind.
  - **The stamp: `2026-10-14_10`**, the round the chunk indices name (`chunks.py show`: NVIDIA
    `2026-10-14_10_c0` to `_c3`, Metal `_m0` to `_m2`), chosen when the indices were cut (entry
    150) and set here as `route_stamps.ROUND`, one line. The campaigns run as
    `dispatch_fused_route_2026-10-14_10_cc86`, `dispatch_fused_route_cuda_2026-10-14_10_cc86`
    and `dispatch_metal_route_2026-10-14_10_applegpu_g13s`, the licence as
    `dispatch_end_to_end_2026-10-14_10`. `route_stamps.SPENT` already holds every route campaign
    directory of an earlier round found on the evidence hosts; no name containing `2026-10-14`
    names a route or licence directory on either host. The registry is in the run closure of 0
    of the 250 gate and probe scripts, and with this entry's files counted as changing
    `chunks.py ready` reads 4 of 4 NVIDIA and 3 of 3 Metal chunks ready, so the freeze moves no
    chunk's closure; `chunks.py check` CHECK OK on both indices. The pending list's 3 "Removed
    when" clauses that named an earlier round's campaigns name these, with one comment block; 0
    ids added, 0 removed.
  - **`meep_identity.json`, re-minted by `record_meep_identity.py`**, because a minted input
    moved: of the 20 files the tool hashes, 3 differed from both sets' record
    (`meep_gpu/sources.py`, entries 141 and 151; `meep_gpu/from_meep.py` and
    `meep_gpu/capsule.py`, entry 149). Keep on the Apple M1 Max against the contract-off MEEP
    1.33.0 single-precision identity reference (175 s), flush on x86-64 Linux against the Linux
    MEEP 1.33.0 single-precision reference (MEEP's flush on import; 164 s, into the keep-minted
    fixtures), 8 / 8 cases per set, one invocation per set: 39,991,116 stored and 65,610,420
    reader words equal to MEEP's in each; the keep files byte-identical after the flush mint (32
    / 32). Of the ledger's 1,899 leaves 88 move and none is a digest, a word count or a verdict:
    times (36), the 16 capsules' byte counts and sha256 (31), the two sets' digests of the three
    moved inputs (10), the fixture bytes (4,636,170 of the 5,000,000 R6 allows) and the Tier A
    load reports' compared-scalar counts of the five structured-permittivity cases per set (10;
    35 to 37 more each: the capsule carries the lift's read report since entry 149). Every row of
    both sets replayed with MEEP not importable holds: 16 / 16 rows on the Apple M1 Max (flush
    rows under FZ) and 16 / 16 on x86-64 Linux (flush rows under the MXCSR lever), 2,916 / 2,916
    stored and 2,916 / 2,916 reader digests per set on each host, the planted flip caught and
    placed in 8 / 8 rows per set on each host. The Tier A test file reads 62 / 62 on the Apple M1
    Max after the keep mint and, NumPy only with no MEEP, 62 / 62 entering in keep and 62 / 62
    entering in flush on x86-64 Linux.
  - **The array-path oracle.** `record_array_path_oracle.py --report` on these bytes: fixture
    set `real-v6`, fingerprint `1ab516279d74002a`, 20 / 20 case digests equal on the Apple M1
    Max (NumPy 2.4.3) and on x86-64 Linux (NumPy 2.2.6). It did not move, so `FIXTURE_SET` is
    not bumped. The kernel ledgers' leaves, and with them each `driver_dispatch` record's
    `provenance_code_sha256` (the code digests of the array-path modules the oracle was minted
    on, which name the moved `from_meep.py` and `sources.py`; reported, never judged), are
    written after the round on these bytes (the pending block "CERT-4 records"), as at the
    earlier freezes.
  - **The freeze criteria.** Against entry 131 (`dbca770023f978c9ca28b4cb8ebaa776eb2ee034`), on
    which L1′ and AP9 last re-ran green, 4 of the 29 files that decide them moved
    (`sources.py`, `from_meep.py`, `capsule.py`, `readback.py`), so both re-ran on these bytes.
    - **L1′ green** on one RTX A6000 under keep and flush: 123 case-rows per policy over the
      cylindrical, family (67 substrates, 59 capsules lifted on these bytes), oracle, hand and
      tiny legs, 0 /
      26,302,894,838 words differing per policy after every sub-step, the planted flip caught
      123 / 123, the mixed-policy control differing 2,446,768 / 6,272,469 words; the hand
      references on CuPy 0 / 197,902,068 words per policy (2,564 calls); the native CuPy folded
      complex 17 / 17 and folded off-diagonal 16 / 16 per policy; `test_cupy_array_path.py` 20
      / 20 on the device. Every per-leg, per-family and per-sub-step count equals entry 131's
      run.
    - **AP9 green** on the Apple M1 Max: `validate_pml` 308 / 308 (44 skipped),
      `validate_no_pml` 112 / 112 (regrouped control differs 56 / 56, derivation 28 / 28),
      `validate_fold` 8 / 8, `validate_folded_complex` 17 / 17 (0 NaN words),
      `validate_folded_offdiag` 16 / 16 with refusals 13 / 13; `record_numpy_reference_weld.py`
      has nothing to write. Its CuPy half ran with L1′.
    - **The hygiene guard reads CLEAN** (`tools/ci/check_hygiene.py --legs tracked,subject`: 0
      hits) on these bytes, and on this entry's patch. The ledgers leg (the previous round's
      machine labels) stays declared until the round's bind writes neutral labels.
    - **The Metal route gate's controls dry run** (`RUNBOOK.md` section 3) against the shipped
      leg of `dispatch_metal_route_2026-10-13_20_applegpu_g13s`: CONTROLS DRY RUN OK, 42 / 42
      cases read, 0 moved, no must-diverge control arming, so nothing to smoke.
    - **AP8's exit is not settled at the freeze.** Of the 291 package modules and
      identity-harness files Table 2 depends on, 10 moved since the tree the corpus Tier C last
      ran on (entry 133), the lift (`from_meep.py`, entry 149) and the deposit (`sources.py`,
      entry 151) among them. The corpus Tier C therefore re-runs in full on all four
      references (Apple keep and flush, x86-64 Linux keep and flush, 194 rows each), launched
      on the tree after entry 151 (`d662668e`) and run beside the round, CPU only, against the
      previous run's rows. This entry changes none of the bytes a corpus row loads: of the 291
      files it moves one, `route_stamps.py`, which no package module imports, and Tier C reads
      no Tier A fixture. A row that does not hold where Table 2 holds stops the round before
      CERT-4.
  - **The suites**, on the bytes of this entry's code and fixtures (its text and the board's
    re-cut change no file their outcomes read), Apple M1 Max, every marker, dispatch off: the
    package suite under the CPU-only shim 54 F / 14,972 P / 1,042 S, the 96 device-visible files
    on the Apple GPU 12 / 4,171 / 331. Against entry 151's reports every id reads the same
    outcome: 16,068 / 16,068 package ids and 4,514 / 4,514 device ids, counting one package id
    at its re-run (`test_conftest_guards.py::test_the_function_pin_is_load_bearing`, a negative
    control that runs a child pytest, failed only because the run's temporary directory sat
    under a checkout whose own conftest pins the dispatch switch; with the directory elsewhere
    the file passes 8 / 8). Needle anchors 0 differences; 212 mutation anchors, 0 stale. The
    coverage board is re-cut by entry 154 on these bytes: 0 transitions, 216 / 216 per-file
    outcome records equal to entry 152's, outcome subject `7abf9fe4b3b34512` (1,143 files),
    board subject 1,154 files; `check` passes.
- **The R27 freeze (entries 155 to 169; owner ruling R27, 2026-10-14).** The owner ruled
  that the reference environments are one clean set of builds per platform, made end to end
  by the repository's own script from the locks: MEEP 1.33.0 from the sha256-pinned tarball,
  single precision, MPI, contraction off, with libGDSII, and every package, library and tool
  the tests, the round and the timing need, the corpus-only ones included; the corpus is the
  same tarball's tree, kept in the environment. Both machines' reference environments
  changed, so the round re-runs in full (CERT-1 to CERT-7) on a new stamp in the new
  environments. Entries 155 to 168 carry that work: the sigma-reader retirement (155), the
  lift fix (156), the closure-held stale texts (157), the reference environments, their
  build scripts and the host documentation (158, 159, 162), the round's inputs bundle (160), the timing module (161), the
  device-stamped chunk tooling and new-capability path (163), the 1.0 fix-ups (164), the round's stamp (165), the Linux
  round children's compiler and libcuda (166), the corpus files the reference environments
  keep beside the corpus tree (167) and the round drivers' sharing of an idle card (168). Entry
  169 (`169-FREEZE`) is this freeze, cut on the tree after entry 168
  (`66e079c85c82a2c811b4d2a60a9eb5b9b29bce3e`); entry 170 re-cuts the coverage board, whose
  subject it moves.
  Every unit's closure has been final since entry 157: an artifact whose closure is unchanged
  certifies on any pushed tip that holds the chunk tooling, and one whose closure moves is
  re-run alone.
  - **The stamp: `2026-10-15_10`**, set by entry 165 as `route_stamps.ROUND`; the three route
    campaigns the R22 round ran under `2026-10-14_10` are in `route_stamps.SPENT`. The chunk
    indices name it on both machines (NVIDIA `2026-10-15_10_cc86_c0` to `_c3`, Metal
    `2026-10-15_10_applegpu_g13s_m0` to `_m3`, `_m3` the re-run of the one unit the corpus gap
    touched).
  - **`meep_identity.json`, re-minted by `record_meep_identity.py` in the new reference
    environments**, because both environments are new and 1 of the 20 files the tool hashes
    moved (`meep_gpu/from_meep.py`, entry 155). Keep on the Apple M1 Max against the Mac
    reference environment's MEEP (174 s, on a tree whose 20 minted inputs are byte-equal to
    this entry's), flush on x86-64 Linux against the Linux reference environment's MEEP
    (MEEP's flush on import; 335 s, into the keep-minted fixtures), 8 / 8 cases per set, one
    invocation per set, each MEEP first judged an identity reference by
    `tools/check_install.py` against its own build record: 39,991,116 stored and 65,610,420
    reader words equal to MEEP's in each; the keep files byte-identical after the flush mint
    (32 / 32). Of the ledger's 1,899 leaves 80 move and none is a digest, a word count or a
    verdict: times (36), the 16 capsules' byte counts and sha256 (30), the moved input's digest
    in both sets (4), the reference blocks (9: each set's build record and its two MEEP images,
    and the flush set's recorded contraction, now `off`) and the fixture bytes (4,636,168 of
    the 5,000,000 R6 allows). Every row of both sets replayed with MEEP not importable holds:
    16 / 16 rows on the Apple M1 Max (flush rows under FZ) and 16 / 16 on x86-64 Linux (flush
    rows under the MXCSR lever), 2,916 / 2,916 stored and 2,916 / 2,916 reader digests per set
    on each host, the planted flip caught and placed in 8 / 8 rows per set on each host. The
    Tier A test file reads 62 / 62 on the Apple M1 Max and, NumPy only with no MEEP, 62 / 62
    entering in keep and 62 / 62 entering in flush on x86-64 Linux.
  - **The array-path oracle.** `record_array_path_oracle.py --report` on these bytes: fixture
    set `real-v6`, fingerprint `1ab516279d74002a`, 20 / 20 case digests equal on the Apple M1
    Max (NumPy 2.4.3) and on x86-64 Linux (NumPy 2.2.6). It did not move, so `FIXTURE_SET` is
    not bumped; the kernel ledgers' leaves are written after the round on these bytes (the
    pending block "CERT-4 records"), as at the earlier freezes.
  - **The freeze criteria**, every leg in the reference environments.
    - **L1′ green** on one RTX A6000: under keep and flush, 123 case-rows per policy over the
      cylindrical, family (67 substrates), oracle, hand and tiny legs, 0 / 26,302,894,838 words
      differing per policy after every sub-step, the planted flip caught 123 / 123, the
      mixed-policy control differing 2,446,768 / 6,272,469 words; the hand references on CuPy
      0 / 197,902,068 words per policy (2,564 calls); the native CuPy folded complex 17 / 17 and
      folded off-diagonal 16 / 16 per policy; `test_cupy_array_path.py` 20 / 20 on the device;
      the toolchain check OK (compute capability 8.6, CuPy 13.5.1, Triton 3.1.0, MEEP 1.33.0
      single precision with MPI). Every per-leg, per-family and per-sub-step count equals the
      R22 freeze's run. 1 of the 29 files that decide L1′ and AP9 moved since that run
      (`meep_gpu/from_meep.py`, entry 155), and the environment is new, so both re-ran here.
    - **AP9 green** on the Apple M1 Max: `validate_pml` 308 / 308 (44 skipped),
      `validate_no_pml` 112 / 112 (regrouped control differs 56 / 56, derivation 28 / 28),
      `validate_fold` 8 / 8, `validate_folded_complex` 17 / 17 (0 NaN words),
      `validate_folded_offdiag` 16 / 16 with refusals 13 / 13; `record_numpy_reference_weld.py`
      has nothing to write.
    - **No unit goes stale.** The per-unit closure digests on these bytes equal those the
      chunks were cut against (123 / 123 NVIDIA units over 442 files, 62 / 62 Metal units over
      337 files), and with this entry's files counted as changing `chunks.py ready` reads 4 / 4
      NVIDIA and 4 / 4 Metal chunks ready; `chunks.py check` CHECK OK on both indices.
    - **The hygiene guard reads CLEAN** (`tools/ci/check_hygiene.py --legs tracked,subject`: 0
      hits) on these bytes and on this entry's patch. The ledgers leg (the previous round's
      machine labels) stays declared until the round's bind writes neutral labels.
    - **The Metal route gate's controls dry run** (`RUNBOOK.md` section 3) against the shipped
      leg of `dispatch_metal_route_2026-10-14_10_applegpu_g13s`: CONTROLS DRY RUN OK, 42 / 42
      cases read, 0 moved, no must-diverge control arming, so nothing to smoke; the route gate's
      run closure (263 files) is byte-equal to the tree it ran on.
    - Needle anchors 0 differences; 212 mutation anchors, 0 stale.
    - **The corpus (Table 2) runs beside the round** in the reference environments
      (the round's corpus record, `CORPUS5`); this entry changes none of the bytes a corpus row
      loads (Tier C reads no Tier A fixture). The Tier B record that L3 judges against is
      re-minted from that run, with the round's inputs manifest re-derived, after the freeze:
      neither is in a gate's closure or the board's subject.
  - **The suites and the coverage board.** The package suite (CPU-only shim) and the
    device-visible files on the Apple GPU run in the Mac reference environment on the bytes of
    this entry's code and fixtures, every marker, dispatch off, and entry 170 re-cuts the board
    from them; its CHANGELOG section states the counts and the transitions.

## 12. The certification round (R27)

The round that certifies 1.0, CERT-1 to CERT-5, under the round stamp `2026-10-15_10`
(`165-ROUND`), on 2026-10-14 and 2026-10-15, in the R27 reference environments. Every
number below is N / D with D named, and each bullet names the round record it comes from by
the record's name (`SCHN2`, `MR27`, ...): private evidence of the development archive, never
shipped, which names hosts by role. The timing (CERT-8) is not part of it (owner ruling
R25); CERT-6 (a new device) and CERT-7 (the suites on the release tip) are their own steps.

- **The trees.** The frozen commit is `2644fc1dd29499cfd5aae8e8372b6c356cadfe5f` (tree
  `0223919995b72a10a7e6fbb5fa34fbfdd33a1f2a`, entries 01 to 170; `FROZEN_R27`,
  `FREEZE27`); the binds, the route campaigns and L3 ran on it. The gate chunks
  launched on the tips of entries 165, 167 and 168 as they landed, and the closure of every
  unit is the frozen commit's: 123 of 123 NVIDIA unit digests over 442 files and 62 of 62
  Metal unit digests over 337 files equal the final closures, read on the frozen commit
  (`SCHN2`, `SCHM2`, `BIND27M` section 2). The corpus rows were
  measured on entries 01 to 157, whose corpus-runner closure (268 paths) equals the frozen
  commit's (`CORP2`, `TIERB27`). The records stage ran on the development
  tip after `173-BINDFIX` (`0e83e1cea08b`, tree `213713d630cc`): the 13 files entries 171 to
  173 change are each in the run closure of 0 of the 250 gate and probe scripts, and 0 of
  the 4 ledgers name any of them (`REC27`, BASE). Per-kernel certification (owner
  ruling R17) is what lets these trees make one release: an artifact stands wherever every
  file its import record names is byte-equal.
- **One script-built reference environment per platform (owner ruling R27).**
  `build_meep_133_linux.sh` and `build_meep_133_macos.sh` build each from the locks: MEEP
  1.33.0 from the SHA-256-pinned tarball, single precision, MPI, libGDSII, no floating-point
  contraction, with every package the tests, the round and the timing use (`158-ENV`,
  `162-LINUX`). The MEEP corpus is that tarball's `python/` tree kept in the environment,
  with the one file outside it two corpus rows read (`tests/cyl-ellipsoid-eps-ref.h5`), named
  file by file in the build record and verified by every round reader: 330 of 330 files and
  1 of 1 sibling on both hosts (`167-CORPFIX`; `CORP2` finding 1,
  `CORPUS_FIX_MAC`, `CORPUS_FIX_LINUX`). The Linux environment's 16 field
  kernels are instruction-identical to the previous Linux reference's, 72,820 of 72,820
  instructions (the Linux environment review, `SCHN2` ENVIRONMENT). Every step of the
  round ran in these two environments with the user site off.
- **The sigma reader retired on measured evidence (`155-SIGMA`, owner ruling R27).** No
  reference build carries the MEEP patch, and the package's branch for it left. Measured
  before and after the removal, a digest of every array word and scalar of each lifted
  driver: under the lock environment's MEEP 192 of 192 lifted rows equal and 27 of 27 rows
  that do not lift with the same outcome; under the contraction-off MEEP 191 of 192 in one
  pass, the other (`test_wvg_src.py`, its eigenmode amplitudes) equal in 3 of 3 repeats on
  each tree; the two GDSII rows 2 of 2 under a GDSII build; the 4 cells of the 2026-08-04
  retirement assessment refused before and after (`CHANGELOG.md`, "The susceptibility-sigma
  reader patch leaves the package").
- **The Metal lift children run under the round's interpreter (`156-LIFT`, owner ruling
  R26).** Before the fix 145 of 145 lift children ran under the interpreter a 2026-09-04
  census recorded, outside the round's environment. In this round's re-cut of the H->D probe,
  60 of its 194 rows differ from the previous cut's only in the interpreter, the reference
  Python now, and the other 134 only in the subject field (`REC27` section 3); every
  Metal L3 row imported
  MEEP from the reference environment's own site-packages, 192 of 192 (`L3M27`).
- **Modular chunks: any subset of the units certifies on its own (`150-CHUNK`, `163-CC90`;
  owner rulings R17, R22, R26).** The index schema is `cert_chunks/2`; a unit binds from its
  own newest run whose import record holds on the release tree and whose device is the
  index's, so a unit runs again alone and the plan re-partitions without voiding another
  unit's run. The chunks of this round (the `CHUNKS27` indices;
  walls from each lane's start to its end, item walls on the Mac):

  | Device | Chunks allocated / launched | Units (selected per launched chunk) | Walls |
  |---|---|---|---|
  | NVIDIA, compute capability 8.6 (RTX A6000) | 13 / 10 (`c1`, `c3`, `c4` superseded by a re-partition before they launched) | 123: `c0` 30, `c2` 13, `c5` 30, `c6` 2, `c7` 15, `c8` 15, `c9` 3, `c10` 1, `c11` 5, `c12` 9 | 1,304 to 12,789 s per chunk; 53,285 s of lane walls in all (14.8 GPU-hours), at most 4 GPUs held by the account |
  | Apple, `applegpu_g13s` (M1 Max) | 4 / 4 | 62: `m0` 1, `m1` 28, `m2` 32, `m3` 1 | 972.1 to 1,710.3 s per chunk; 5,285.3 s in all, one chunk after another on the one GPU |

  Units run again alone: `c6`, the two NVIDIA units whose lift basis reads the corpus
  sibling, after the corpus fix; `c9` to `c12`, the units of two chunks the scheduler
  stopped (`c2` on the shared card, `c10` split over two cards); `m3`, the one Metal unit
  whose lift basis reads the sibling, whose `m1` run failed for it. No closure moved, so no
  unit went stale (`SCHN2`, `SCHM2`).
- **Card sharing that yields at once to other users (owner ruling R28, `168-SHARE`).** One
  card holding another account's idle process (4,970 MiB, 0 % utilisation on 538 of 538
  readings) was shared by opt-in for chunks `c2`, `c6` and `c7`. The yield supervisor read
  every held card every 30 s, 1,743 readings, and yielded 0 times; the shared card's free
  memory never fell below 31,073 MiB. The Triton fleet driver refuses a card with any compute
  process, so on the shared card 11 legs of `c2` ran no gate (an environment refusal; its
  fleet units ran again on unshared cards), and the memory bound must be the units' measured
  peak (a Triton H->D lift used 12,348 MiB, against a bound set from a smoke's 270 MiB)
  (`SCHN2` CARDS and findings 3 and 4).
- **CERT-1: the freeze and the corpus.** The freeze criteria on the frozen tree, in the new
  environments: Tier A re-minted, 8 of 8 cases per set, 16 of 16 replays per host; L1′ 0 of
  26,302,894,838 words differing per policy, 123 of 123 plants caught; AP9 green; the hygiene
  guard CLEAN on the tracked files and the board's subject (`FREEZE27`). Table 2,
  from the Tier B record `126e3255…` re-minted from the corpus run (94 of 94 replays per set;
  `TIERB27`): 193, 193, 192 and 192 of 194 rows hold on `mac_keep`, `mac_flush`,
  `linux_keep` and `linux_flush`, 0 differing, 0 errors, 0 not run (`REC27` section
  7b). `ex.stochastic_emitter` is excluded by name on all four; `ex.coupler` on the two Linux
  sets, the R24 expected difference: 12 of 20 of its GPU-host runs held, 4 of 4 on the Mac,
  and only the lift's re-solved eigenmode amplitudes and, under flush, MEEP's own solve
  varied between processes (`CORP2` finding 2). The two GDSII rows run now:
  `ex.ring_gds` holds on 4 of 4 sets, `ex.coupler` on both Apple sets. The round's inputs
  bundle: 1,782 files, `round_inputs_1_0.tgz` `1cc0c933…` (`TIERB27`).
- **CERT-2: the device smokes**, the first launches in the new environments: NVIDIA 7 of 7
  steps (`SCHN2`), Metal 2 of 2 gates released (`SCHM2`).
- **CERT-3: the device gates.** NVIDIA: `chunks.py verify` reads VERIFY OK, 123 of 123
  units hold, from 10 launches and 440 artifacts; gate failures on the frozen bytes 0; 11
  legs died of their environment, the shared-card refusals above (`SCHN2`).
  Metal: VERIFY OK, 62 of 62 units hold, from 4 launches and 81 artifacts; 62 of 62 gate
  artifacts released and 45 of 45 cited welds; one `m1` run failed by the corpus gap and is
  superseded by `m3` (`SCHM2`).
- **The binds.** NVIDIA (`BIND27N`): 55 CUDA families with 116 canonical legs
  recorded, 54 CUDA welds and 52 Triton welds rebound, the bit identity, 5 of 5 composition
  records, the labels made neutral; `flush` admitted for 75 of 75 Triton arms and 34 of 34
  CUDA arms; `legs.py check --route` CHECK OK. Finding N1 (the chunked bind's `--only` named
  two keys the Triton ledger never held) is fixed by `173-BINDFIX`, which names each such unit
  on a `# REFUSED` line (`BINDFIX`). Metal (`BIND27M`,
  `METALDIG`): 62 of 62 weld keys rebound; `172-METALDIG` re-cuts the welds' device
  tier from the runs they bind, 49 digests moved and 1 added on 28 welds. The hygiene guard
  reads the four bound ledgers CLEAN, 0 hits.
- **CERT-4: the route campaigns and the licence, one shot each.** NVIDIA (`NR27`,
  dry run `DRY27N`): every required leg released, Triton 3 of 3 (37 of 37 cases
  dispatched on each) and CUDA 4 of 4 with the optional `cuda_alone` (37 of 39 dispatched on
  each of the four, the other 2 with no fused arm); both blocking preflights 6 of 6;
  0 failures and 0 control failures on 11 of 11 legs; 0 of 323 cases moved against the R22
  route. The go/no-go: VERDICT GO, ship 9 of 9 PASS-DISPATCHED, null 9 of 9 PASS-FELL-BACK.
  Metal (`MR27`, dry run `DRY27M`): 4 of 4 legs released under held
  residency (held on 84 of 84 leg-cases), 42 of 44 cases dispatched on each dispatch leg,
  the comparison with the R22 route equal on every leg. `residency-sync_in-dropped` reads
  NOT-APPLICABLE on 42 of 42 dispatched cases per dispatch leg (`acquire_write` entered 0
  times): under held residency the seam has no must-diverge null of its own in 1.0, and its
  evidence is `residency-verify-after-every-sync_out` and byte identity (owner ruling R29; a
  door-write null is a 1.0.1 item). The three `driver_dispatch` records are re-cut from these
  campaigns.
- **CERT-4: the records** (`REC27`). The H->D probe and the three censuses are cut
  again on the records' tree (194 of 194 rows on each basis), and the boards from them:
  seam-instances reachable in dispatch 334 (Triton), 352 (CUDA) and 327 (Metal) of 597, and
  seam-instances a fused product exists for 537, 538 and 507 of 597, each equal instance by
  instance to the boards cut for the 2026-10-13 records entry. The Metal ledger's
  `metal_kernels` block is re-cut by `launch.write_fingerprints()` (finding K1: its
  kernel-source digests now name
  the shaders this round certified, 36 of 36), and the array-path oracle is minted into the
  three kernel ledgers (fixture set `real-v6`, fingerprint `1ab516279d74002a`). The pending
  list's removal block is measured in the evidence and the CI views: 21 entries removed,
  56 -> 35, 0 undeclared failures.
- **CERT-5: L3, the direct corpus check.** NVIDIA (`L3N27`): 192 of 192 rows hold
  on each of four legs (Triton under the default precedence and CUDA preferred, against
  `linux_keep` and `linux_flush`), 70,968 of 70,968 stored and reader digests each, 189 of 192
  rows with Triton kernels dispatched (3 refused by name to the array path, still identical)
  and 192 of 192 under the CUDA preference. Metal (`L3M27`, `flush`): 192 of 192
  rows run, VERDICT HOLDS: 160 of 160 hold against `linux_flush` outside the 32 pre-registered
  rows where the two reference MEEPs differ, which hold against `mac_flush`, as all 192 do;
  190 of 192 dispatched on the Metal table under held residency.
- **What this round does not license.**
  - Certification of any GPU but one RTX A6000 (compute capability 8.6) and one M1 Max
    (`applegpu_g13s`), in these two reference environments; another GPU certifies by its own
    round on these bytes (`docs/development/certify-a-new-gpu.md`).
  - Identity with a double-precision or contracting MEEP, or on a row Table 2 excludes;
    `EigenModeSource` runs can differ from MEEP until 1.0.1 reads MEEP's stored amplitudes
    (owner ruling R24).
  - L3 on any host but the two that cut its record (each row checks the MEEP images'
    SHA-256).
  - Any timing or speed. The 1.0 timing (CERT-8) is not part of certification: it ran on
    2026-10-16 on the release's bytes, and its figures, each with its gate status under owner
    ruling R33, are in `docs/guides/will-it-help.md` (owner rulings R20, R25, R32, R33); 1.0.0
    ships the 2026-09-21 dispatch-preference records, and the records the 1.0 timing cut land
    in a later 1.0.x release.
  - A must-diverge null for the held Metal seam (R29), and a board figure for the CUDA table
    composing without Triton.
  - The records this round's tools did not re-cut, carried unchanged (`BIND27N`):
    the Triton family re-certification record that 22 of 75 Triton arms cite (its tool
    requires one host line, and the nine families ran on three cards), the CUDA ADE re-cut
    (refused; admission does not read it), two Triton entries with a declared mixed policy,
    one CUDA weld never seeded and two Triton device-gate keys never seeded. The 35 pending
    ids that stay are declared with the step that removes each.
- **Owed after this round.** CERT-7, the suites per environment on the release tip. CERT-8,
  the timing, ran on 2026-10-16 (`docs/guides/will-it-help.md`).
- **The bit-parity coverage board** is re-cut on entries 01 to 175 (`176-BOARD`; its subjects
  had moved after `170-BOARD14`). (a) to (c) do not move. (d) device identity and (e)
  CERTIFIED land with this round's records: CUDA 53 / 53 under keep and under flush, Triton
  53 / 56 under keep and 0 / 56 under flush (no module's pinning entry carries a live run
  under flush, by the board's rule), Metal 74 / 74 under flush; each was 0 before them.
  It is re-cut again on entries 01 to 178 (`179-BOARD`; entry 177's timing cutters moved
  its outcome subject), and no cell moves.
