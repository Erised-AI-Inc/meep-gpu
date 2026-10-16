# Bit-parity coverage board

Built by `parity/meep_gpu/build_parity_board.py` at 2026-10-16T01:30:01Z on subject `d492a18d9930faf8` (1163 files). Generated: do not edit; re-run `build`.

Live array-path oracle: `array_path_oracle/real-v6` `1ab516279d74002a` (20 fixture cases).
Test outcomes: recorded 2026-10-16T01:29:32Z on Darwin arm64, 216 test files run; carved from one integration's suites: the CUDA and Triton test files' cases of its package suite (CPU-only shim, Apple GPU hidden, SUITE_ENFORCE_SKIPS=0) and the Metal test files' cases of its device run (Apple GPU, no shim), on this tree's outcome subject; every marker; MEEP_GPU_DISPATCH=0.

Each count is N / D: D is the rows the column applies to and could measure; refused rows are counted beside it, never as a gap and never as a pass. Baselines (owner ruling R13): keep and flush on CUDA, Triton and the array path; flush only on Metal.

Columns (a) and (b) read the device text each module's builder EMITS (`parity_board_emit.py`: the CUDA source `cp.RawKernel` is handed, the Metal MSL after template splicing, the Triton `@jit` program with its helpers inlined), with no device. A module with no emit route is read off its own source text instead, and its cell says so.

Column (d-dev) is DEVELOPMENT EVIDENCE, NOT CERTIFICATION: an integrator's re-run of a lane's device legs (`record_dev_evidence.py`), counted only while its subject (the module's bytes, its emitted text, the array path) is the live one. Columns (d) and (e) never read it.

## Summary

| Table | Rows | (a) B select | (b) MEEP-order A | (c) host identical | (d) device keep | (d) device flush | (d-dev) keep, NOT certification | (d-dev) flush, NOT certification | (e) CERTIFIED keep | (e) CERTIFIED flush |
|---|---|---|---|---|---|---|---|---|---|---|
| CUDA | 75 | 55 / 55 (+1 refused) | 57 / 57 (+8 refused) | 18 / 18 (+47 refused) | 53 / 53 (+12 refused) | 53 / 53 (+12 refused) | 4 / 4 (+61 refused) | 0 / 0 (+65 refused) | 53 / 53 (+12 refused) | 53 / 53 (+12 refused) |
| Triton | 73 | 55 / 56 (+2 refused) | 61 / 61 (+7 refused) | 27 / 29 (+39 refused) | 53 / 56 (+12 refused) | 0 / 56 (+12 refused) | 2 / 2 (+66 refused) | 0 / 0 (+68 refused) | 53 / 56 (+12 refused) | 0 / 56 (+12 refused) |
| Metal | 92 | 60 / 60 (+4 refused) | 68 / 68 (+9 refused) | 29 / 29 (+48 refused) | n/a (R13) | 74 / 74 (+3 refused) | n/a (R13) | 3 / 3 (+74 refused) | n/a (R13) | 74 / 74 (+3 refused) |

Witness of (a) and (b), per table (rows with a device form): CUDA 61 emitted, 4 source text; Triton 67 emitted, 1 source text; Metal 75 emitted, 2 source text.

Development evidence: 11 record(s) in `parity/meep_gpu/parity_board/dev_evidence.json`.

Why (e) reads what it reads:

- CUDA: 55 of 55 byte-binding entries record the live array path (array_path_oracle/real-v6 1ab516279d74002a).
- Triton: 66 of 66 byte-binding entries record the live array path (array_path_oracle/real-v6 1ab516279d74002a).
- Metal: 63 of 63 byte-binding entries record the live array path (array_path_oracle/real-v6 1ab516279d74002a).

## Array path against MEEP

- keep: 8 minted cases (a, b, c, c_prime, ex.absorber-1d, ex.material-dispersion, t.test_holey_wvg_bands.TestHoleyWvgBands.test_fields_at_kx, t.test_ring_cyl.TestRingCyl.test_ring_cyl); reference MEEP 1.33.0 on Darwin arm64, contraction off, OMP 1.
- flush: 8 minted cases (a, b, c, c_prime, ex.absorber-1d, ex.material-dispersion, t.test_holey_wvg_bands.TestHoleyWvgBands.test_fields_at_kx, t.test_ring_cyl.TestRingCyl.test_ring_cyl); reference MEEP 1.33.0 on Linux x86_64, contraction off, OMP 1.

| Family | WPs | keep | flush |
|---|---|---|---|
| base_pml (Real PML forms: A1 + A2 + A3, B (per-chunk select), C (double deposit)) | AP3, AP4, AP5 | identical: 4 / 4 cases; 0 / 38,040,036 stored and 0 / 61,772,940 reader words differ | identical: 4 / 4 cases; 0 / 38,040,036 stored and 0 / 61,772,940 reader words differ |
| mirror_symmetry (Mirror symmetry: ghost rows copy their image row) | AP4 (D5 ghost rows) | identical: 3 / 3 cases; 0 / 10,982,436 stored and 0 / 36,601,740 reader words differ | identical: 3 / 3 cases; 0 / 10,982,436 stored and 0 / 36,601,740 reader words differ |
| complex_bloch (Complex storage and Bloch phases (CXP, CXW, CXB, CXD)) | AP6a | identical: 2 / 2 cases; 0 / 1,801,920 stored and 0 / 3,688,320 reader words differ | identical: 2 / 2 cases; 0 / 1,801,920 stored and 0 / 3,688,320 reader words differ |
| folds (Folded increments: beta and BFAST as separate ladder passes (FOLD)) | AP6b | refused: no minted keep case exercises the family | refused: no minted flush case exercises the family |
| cylindrical (Cylindrical: per-chunk r prefix, rinv multiply, i*m/r pass (F_CYL, F_CYL_CPLX)) | AP6c | identical: 1 / 1 cases; 0 / 59,520 stored and 0 / 59,520 reader words differ | identical: 1 / 1 cases; 0 / 59,520 stored and 0 / 59,520 reader words differ |
| conductive (Conductivity under a PML: four forms per chunk (F_COND*)) | AP6d | refused: no minted keep case exercises the family | refused: no minted flush case exercises the family |
| offdiag (Off-diagonal chi1inv: double row sum, per-chunk presence (F_OFFDIAG, F_CHI1INV)) | AP6e | identical: 1 / 1 cases; 0 / 1,742,400 stored and 0 / 3,628,800 reader words differ | identical: 1 / 1 cases; 0 / 1,742,400 stored and 0 / 3,628,800 reader words differ |
| dispersive (Dispersive (ADE): MEEP's float32 constants and grouping (F_DISPERSIVE)) | AP6f | identical: 2 / 2 cases; 0 / 149,160 stored and 0 / 149,160 reader words differ | identical: 2 / 2 cases; 0 / 149,160 stored and 0 / 149,160 reader words differ |
| sources (Sources: continuous and custom forms, src_vol merge, integrated f_minus_p) | AP6g | refused: no minted keep case exercises the family | refused: no minted flush case exercises the family |
| nonlinear (Nonlinear |D|^2 in double) | AP6h | refused: no minted keep case exercises the family | refused: no minted flush case exercises the family |

- keep: 6 / 6 (+4 refused) families identical.
- flush: 6 / 6 (+4 refused) families identical.

## CUDA (75 rows: 9 a_only, 5 host, 56 kernel, 5 unchanged)

Refused cells, by reason:

- b_select: refused, no PML form and no chunk-record read in its own text: 1 (cylindrical_prefix)
- a_form: refused, no A-form witness in its own text: 4 (complex_electric_fill_carry, complex_fill_carry, cylindrical_prefix, no_pml_constitutive)
- a_form: refused, no curl or constitutive A-form site in its emitted text: 4 (ade_kernels, no_pml_three_slot_dispersive_weld, offdiag_stencil_weld, three_slot_dispersive_weld)
- host_identical: refused, no test file of the table names or documents the module: 22 (bfast_fused_electric_pair, bfast_fused_magnetic_pair, complex_beta_fused_electric_pair, complex_beta_fused_magnetic_pair, complex_electric_fill_carry, complex_fill_carry, complex_folded_fused_electric_pair, complex_folded_fused_magnetic_pair, complex_fused_hd_pair, complex_fused_magnetic_pair, complex_no_pml_fused_polarization_pair, complex_offdiag_stencil_weld, ...)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_bloch_connection.py, test_complex_no_pml_chunk_forms.py, test_complex_offdiag_stencil_welds.py measure no arithmetic of it: 1 (complex_no_pml_offdiag_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_bloch_connection.py, test_complex_offdiag_stencil_welds.py measure no arithmetic of it: 1 (folded_complex_offdiag_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_complex_beta_fused_hd_pair.py measure no arithmetic of it: 1 (complex_beta_fused_hd_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_complex_folded.py measure no arithmetic of it: 1 (complex_folded_kernels)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_complex_fused_electric_pairs.py measure no arithmetic of it: 1 (complex_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_complex_no_pml.py measure no arithmetic of it: 1 (complex_no_pml_kernels)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_complex_no_pml_three_slot_dispersive_weld.py, test_complex_no_pml_chunk_forms.py measure no arithmetic of it: 1 (complex_no_pml_three_slot_dispersive_weld)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_complex_offdiag_update_e.py measure no arithmetic of it: 1 (complex_offdiag_update_e)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_conductive_bfast_fused_hd_pair.py measure no arithmetic of it: 1 (conductive_bfast_fused_hd_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_conductive_fused_electric_pair.py measure no arithmetic of it: 1 (conductive_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_conductive_kernels.py measure no arithmetic of it: 1 (conductive_kernels)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_cylindrical_fused_electric_pair_hoist.py, test_complex_fused_electric_pairs.py measure no arithmetic of it: 1 (cylindrical_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_cylindrical_fused_magnetic_pair_hoist.py measure no arithmetic of it: 1 (cylindrical_fused_magnetic_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_dispersive_fused_electric_pair.py measure no arithmetic of it: 1 (dispersive_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_dispersive_offdiag_polarization_pair.py measure no arithmetic of it: 1 (dispersive_offdiag_fused_polarization_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_dispersive_offdiag_update_e.py measure no arithmetic of it: 1 (dispersive_offdiag_update_e)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_folded_offdiag.py measure no arithmetic of it: 1 (folded_offdiag_kernels)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_fused_hd_pair.py measure no arithmetic of it: 1 (fused_hd_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_no_pml_complex_fused_electric_pair.py, test_complex_no_pml_chunk_forms.py measure no arithmetic of it: 1 (no_pml_complex_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_no_pml_constitutive.py measure no arithmetic of it: 1 (no_pml_constitutive)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_no_pml_curl.py measure no arithmetic of it: 1 (no_pml_curl)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_no_pml_dispersive_fused_electric_pair.py measure no arithmetic of it: 1 (no_pml_dispersive_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_offdiag_constitutive_pml_real.py measure no arithmetic of it: 1 (offdiag_constitutive_kernels)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_offdiag_stencil_welds.py measure no arithmetic of it: 1 (folded_offdiag_fused_electric_pair)
- host_identical: refused, none of its test files runs the emitted device text through the host interpreter (test_root_chunk_forms): test_special_kz_fused_hd_pair.py measure no arithmetic of it: 1 (special_kz_fused_hd_pair)
- device_identical:keep: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (complex_beta_fused_hd_pair, complex_electric_fill_carry, complex_emitter, complex_fill_carry, complex_offdiag_update_e, conductive_bfast_fused_hd_pair, cylindrical_prefix, dispersive_fused_electric_pair, no_pml_complex_fused_electric_pair, no_pml_constitutive, no_pml_dispersive_fused_electric_pair, special_kz_fused_hd_pair)
- device_identical:flush: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (complex_beta_fused_hd_pair, complex_electric_fill_carry, complex_emitter, complex_fill_carry, complex_offdiag_update_e, conductive_bfast_fused_hd_pair, cylindrical_prefix, dispersive_fused_electric_pair, no_pml_complex_fused_electric_pair, no_pml_constitutive, no_pml_dispersive_fused_electric_pair, special_kz_fused_hd_pair)
- dev_evidence:keep: refused, no development evidence recorded on this subject: 61 (ade_kernels, bfast_curl, bfast_fused_electric_pair, bfast_fused_magnetic_pair, complex_beta_fused_electric_pair, complex_beta_fused_hd_pair, complex_beta_fused_magnetic_pair, complex_beta_kernels, complex_electric_fill_carry, complex_emitter, complex_fill_carry, complex_folded_fused_electric_pair, ...)
- dev_evidence:flush: refused, no development evidence recorded on this subject: 65 (ade_kernels, bfast_curl, bfast_fused_electric_pair, bfast_fused_magnetic_pair, complex_beta_fused_electric_pair, complex_beta_fused_hd_pair, complex_beta_fused_magnetic_pair, complex_beta_kernels, complex_electric_fill_carry, complex_emitter, complex_fill_carry, complex_folded_fused_electric_pair, ...)
- certified:keep: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (complex_beta_fused_hd_pair, complex_electric_fill_carry, complex_emitter, complex_fill_carry, complex_offdiag_update_e, conductive_bfast_fused_hd_pair, cylindrical_prefix, dispersive_fused_electric_pair, no_pml_complex_fused_electric_pair, no_pml_constitutive, no_pml_dispersive_fused_electric_pair, special_kz_fused_hd_pair)
- certified:flush: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (complex_beta_fused_hd_pair, complex_electric_fill_carry, complex_emitter, complex_fill_carry, complex_offdiag_update_e, conductive_bfast_fused_hd_pair, cylindrical_prefix, dispersive_fused_electric_pair, no_pml_complex_fused_electric_pair, no_pml_constitutive, no_pml_dispersive_fused_electric_pair, special_kz_fused_hd_pair)

Modules of the table directory that no step list names: bloch_connection, bounds_block, chunk_coverage, folded_pass, offdiag_presence.

| Step | Module | Kind | Witness | b select | a form | host identical | device identical:keep | device identical:flush | dev evidence:keep | dev evidence:flush | certified:keep | certified:flush | Why not (first) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | `fused_pairs` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| C0 | `coverage` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| C0 | `registry` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| C0 | `arms` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| C0 | `fastpath_cuda` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| C1 | `step_curl_kernels` | kernel | emitted | yes | yes | yes | yes | yes | yes | refused | yes | yes |  |
| C1 | `constitutive_kernels` | kernel | emitted | yes | yes | yes | yes | yes | yes | refused | yes | yes |  |
| C1 | `nonlinear_constitutive` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C1 | `own_cell_hoist` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C2 | `fused_magnetic_pair` | kernel | emitted | yes | yes | yes | yes | yes | yes | refused | yes | yes |  |
| C2 | `fused_electric_pair` | kernel | emitted | yes | yes | yes | yes | yes | yes | refused | yes | yes |  |
| C2 | `fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C3 | `conductive_kernels` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C3 | `conductive_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C4 | `offdiag_emitter` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C4 | `offdiag_constitutive_kernels` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C4 | `folded_offdiag_kernels` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C4 | `offdiag_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C4 | `offdiag_stencil_weld` | kernel | emitted | yes | refused | refused | yes | yes | refused | refused | yes | yes |  |
| C4 | `folded_offdiag_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C5 | `ade_kernels` | kernel | emitted | yes | refused | yes | yes | yes | refused | refused | yes | yes |  |
| C5 | `dispersive_kernels` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C5 | `dispersive_fused_electric_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| C5 | `fused_polarization_pair` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C5 | `three_slot_dispersive_weld` | kernel | emitted | yes | refused | yes | yes | yes | refused | refused | yes | yes |  |
| C5 | `dispersive_offdiag_update_e` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C5 | `dispersive_offdiag_fused_polarization_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_emitter` | kernel | emitted | yes | yes | yes | refused | refused | refused | refused | refused | refused |  |
| C6 | `complex_pml_kernels` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_fill_carry` | kernel | source text | yes | refused | refused | refused | refused | refused | refused | refused | refused |  |
| C6 | `complex_electric_fill_carry` | kernel | source text | yes | refused | refused | refused | refused | refused | refused | refused | refused |  |
| C6 | `complex_folded_kernels` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_no_pml_kernels` | a_only | emitted | n/a | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_folded_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_folded_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `complex_offdiag_update_e` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| C6 | `complex_offdiag_stencil_weld` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C6 | `folded_complex_offdiag_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C7 | `bfast_curl` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C7 | `special_kz_curl` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C7 | `complex_beta_kernels` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C7 | `bfast_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C7 | `bfast_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C7 | `special_kz_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C7 | `special_kz_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| C7 | `special_kz_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C7 | `complex_beta_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C7 | `complex_beta_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| C7 | `complex_beta_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C7 | `conductive_bfast_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| C8 | `cylindrical_prefix` | kernel | source text | refused | refused | yes | refused | refused | refused | refused | refused | refused |  |
| C8 | `cylindrical_kernels` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C8 | `cylindrical_real_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C8 | `cylindrical_real_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C8 | `cylindrical_real_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C8 | `cylindrical_complex_kernels` | kernel | emitted | yes | yes | yes | yes | yes | refused | refused | yes | yes |  |
| C8 | `cylindrical_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C8 | `cylindrical_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C8 | `cylindrical_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C9 | `complex_no_pml_fused_polarization_pair` | a_only | emitted | n/a | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C9 | `complex_no_pml_offdiag_fused_electric_pair` | a_only | emitted | n/a | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C9 | `complex_no_pml_three_slot_dispersive_weld` | a_only | emitted | n/a | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C9 | `no_pml_complex_fused_electric_pair` | a_only | emitted | n/a | yes | refused | refused | refused | refused | refused | refused | refused |  |
| C9 | `no_pml_constitutive` | a_only | source text | n/a | refused | refused | refused | refused | refused | refused | refused | refused |  |
| C9 | `no_pml_curl` | a_only | emitted | n/a | yes | refused | yes | yes | refused | refused | yes | yes |  |
| C9 | `no_pml_dispersive_fused_electric_pair` | a_only | emitted | n/a | yes | refused | refused | refused | refused | refused | refused | refused |  |
| C9 | `no_pml_three_slot_dispersive_weld` | a_only | emitted | n/a | refused | refused | yes | yes | refused | refused | yes | yes |  |
| UNCHANGED | `in_seam_passes` | unchanged | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| UNCHANGED | `compile_cache` | unchanged | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| UNCHANGED | `__init__` | unchanged | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| UNCHANGED | `cylindrical_coverage` | unchanged | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| UNCHANGED | `in_seam_coverage` | unchanged | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |

## Triton (73 rows: 10 a_only, 5 host, 58 kernel)

Refused cells, by reason:

- b_select: refused, no PML form and no chunk-record read in its emitted text: 2 (complex_ade, complex_fused_ade_chain)
- a_form: refused, no A-form witness in its own text: 1 (no_pml_constitutive)
- a_form: refused, no curl or constitutive A-form site in its emitted text: 6 (complex_ade, complex_fused_ade_chain, complex_no_pml_stored_e, fused_ade_state, no_pml_ade, no_pml_stored_e)
- host_identical: refused, no test file of the table names or documents the module: 1 (forms)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_beta_fused_electric_pair.py measure no arithmetic of it: 1 (beta_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_beta_fused_magnetic_pair.py measure no arithmetic of it: 1 (beta_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_bfast_fused_electric_pair.py measure no arithmetic of it: 1 (bfast_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_bfast_fused_magnetic_pair.py measure no arithmetic of it: 1 (bfast_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_ade.py measure no arithmetic of it: 1 (complex_ade)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_beta_fused_electric_pair.py measure no arithmetic of it: 1 (complex_beta_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_beta_fused_magnetic_pair.py measure no arithmetic of it: 1 (complex_beta_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_conductive_fused_pair.py, test_triton_certified_fused_products.py measure no arithmetic of it: 1 (complex_conductive_fused_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_fields.py measure no arithmetic of it: 1 (complex_fields)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_fused_ade_chain.py measure no arithmetic of it: 1 (complex_fused_ade_chain)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_fused_hd_pair.py measure no arithmetic of it: 1 (complex_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_complex_fused_magnetic_pair.py measure no arithmetic of it: 1 (complex_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_conductive_fused_electric_pair.py measure no arithmetic of it: 1 (conductive_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_cylindrical_fused_electric_pair.py measure no arithmetic of it: 1 (cylindrical_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_cylindrical_fused_hd_pair.py measure no arithmetic of it: 1 (cylindrical_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_cylindrical_fused_magnetic_pair.py measure no arithmetic of it: 1 (cylindrical_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_cylindrical_real_fused_electric_pair.py measure no arithmetic of it: 1 (cylindrical_real_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_cylindrical_real_fused_magnetic_pair.py measure no arithmetic of it: 1 (cylindrical_real_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_dispersive_fused_pair.py measure no arithmetic of it: 1 (dispersive_fused_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_beta_complex_fused_magnetic_pair.py measure no arithmetic of it: 1 (folded_beta_complex_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_beta_complex_fused_pair.py measure no arithmetic of it: 1 (folded_beta_complex_fused_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_beta_fused_electric_pair.py measure no arithmetic of it: 1 (folded_beta_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_beta_fused_magnetic_pair.py measure no arithmetic of it: 1 (folded_beta_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_complex_fused_pair.py measure no arithmetic of it: 1 (folded_complex_fused_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_dispersive_fused_pair.py measure no arithmetic of it: 1 (folded_dispersive_fused_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_fused_hd_pair.py measure no arithmetic of it: 1 (folded_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_folded_offdiag_fused_ade_chain.py measure no arithmetic of it: 1 (folded_offdiag_fused_ade_chain)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_fused_ade_chain.py measure no arithmetic of it: 1 (fused_ade_chain)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_fused_dispersive_chain.py measure no arithmetic of it: 1 (fused_dispersive_chain)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_fused_hd_pair.py measure no arithmetic of it: 1 (fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_hd_tail_products.py measure no arithmetic of it: 5 (beta_complex_fused_hd_pair, beta_real_fused_hd_pair, conductive_bfast_fused_hd_pair, folded_complex_fused_hd_pair, nonlinear_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_no_pml.py measure no arithmetic of it: 1 (no_pml)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_nonlinear_fused_magnetic_pair.py measure no arithmetic of it: 1 (nonlinear_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_triton_offdiag_fused_electric_pair.py measure no arithmetic of it: 1 (offdiag_fused_electric_pair)
- device_identical:keep: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (beta_complex_fused_hd_pair, beta_real_fused_hd_pair, bfast_curl, complex_fused_hd_pair, conductive_bfast_fused_hd_pair, cylindrical_fused_hd_pair, cylindrical_real_fused_hd_pair, folded_complex_fused_hd_pair, folded_fused_hd_pair, fused_hd_pair, nonlinear_fused_hd_pair, nonlinear_update_e)
- device_identical:flush: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (beta_complex_fused_hd_pair, beta_real_fused_hd_pair, bfast_curl, complex_fused_hd_pair, conductive_bfast_fused_hd_pair, cylindrical_fused_hd_pair, cylindrical_real_fused_hd_pair, folded_complex_fused_hd_pair, folded_fused_hd_pair, fused_hd_pair, nonlinear_fused_hd_pair, nonlinear_update_e)
- dev_evidence:keep: refused, no development evidence recorded on this subject: 66 (beta_complex_fused_hd_pair, beta_fused_electric_pair, beta_fused_magnetic_pair, beta_real_fused_hd_pair, bfast_curl, bfast_fused_electric_pair, bfast_fused_magnetic_pair, complex_ade, complex_beta_fused_electric_pair, complex_beta_fused_magnetic_pair, complex_conductive_fused_pair, complex_fields, ...)
- dev_evidence:flush: refused, no development evidence recorded on this subject: 68 (beta_complex_fused_hd_pair, beta_fused_electric_pair, beta_fused_magnetic_pair, beta_real_fused_hd_pair, bfast_curl, bfast_fused_electric_pair, bfast_fused_magnetic_pair, complex_ade, complex_beta_fused_electric_pair, complex_beta_fused_magnetic_pair, complex_conductive_fused_pair, complex_fields, ...)
- certified:keep: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (beta_complex_fused_hd_pair, beta_real_fused_hd_pair, bfast_curl, complex_fused_hd_pair, conductive_bfast_fused_hd_pair, cylindrical_fused_hd_pair, cylindrical_real_fused_hd_pair, folded_complex_fused_hd_pair, folded_fused_hd_pair, fused_hd_pair, nonlinear_fused_hd_pair, nonlinear_update_e)
- certified:flush: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 12 (beta_complex_fused_hd_pair, beta_real_fused_hd_pair, bfast_curl, complex_fused_hd_pair, conductive_bfast_fused_hd_pair, cylindrical_fused_hd_pair, cylindrical_real_fused_hd_pair, folded_complex_fused_hd_pair, folded_fused_hd_pair, fused_hd_pair, nonlinear_fused_hd_pair, nonlinear_update_e)

| Step | Module | Kind | Witness | b select | a form | host identical | device identical:keep | device identical:flush | dev evidence:keep | dev evidence:flush | certified:keep | certified:flush | Why not (first) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | `launch` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| T0 | `coverage` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| T0 | `__init__` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| T0 | `offdiag_scratch_weld` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| T0 | `folded_dispersive_update_e` | host | - | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |  |
| T1 | `kernels` | kernel | emitted | yes | yes | no | yes | no | yes | refused | yes | no | host_i: 1 id(s) listed as pending under kernel lanes (KM1, KM8, KT1, KT10, KT2, KT3, KT4, KT5, KT6, KT7, KT8, KT9; 1 name its own lane KT1) |
| T1 | `forms` | kernel | emitted | yes | yes | refused | no | no | yes | refused | no | no | device: bit_identity_gate: status None |
| T2 | `conductivity` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: 2 pinning entries, none passes: bit_identity_gate: status None; no live run under flush |
| T2 | `conductive_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_conductive_fused_electric_pair_device_gate: no live run under flush |
| T3 | `symmetry` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: 6 pinning entries, none passes: bit_identity_gate: status None; no live run under flush |
| T3 | `folded_fused_pair` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: 2 pinning entries, none passes: triton_folded_deposit_closure_device_gate: no live run under flush |
| T3 | `folded_fused_magnetic_pair` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: 3 pinning entries, none passes: triton_folded_deposit_closure_device_gate: no live run under flush |
| T4 | `offdiag_update_e` | kernel | emitted | yes | yes | yes | no | no | refused | refused | no | no | device: bit_identity_gate: status None |
| T4 | `offdiag_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_offdiag_fused_electric_pair_device_gate: no live run under flush |
| T4 | `folded_offdiag_update_e` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_folded_offdiag_device_gate: no live run under flush |
| T4 | `folded_offdiag_fused_electric_pair` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_folded_offdiag_fused_electric_pair_device_gate: no live run under flush |
| T5 | `complex_fields` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: 6 pinning entries, none passes: triton_complex_device_gate: no live run under flush |
| T5 | `complex_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: 2 pinning entries, none passes: triton_complex_fused_magnetic_pair_device_gate: no live run under flush |
| T5 | `complex_fused_electric_pair` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_complex_fused_electric_pair_device_gate: no live run under flush |
| T5 | `complex_offdiag_update_e` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_complex_offdiag_device_gate: no live run under flush |
| T5 | `folded_complex` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_folded_complex_device_gate: no live run under flush |
| T5 | `folded_complex_fused_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_folded_complex_fused_pair_device_gate: no live run under flush |
| T5 | `folded_complex_fused_magnetic_pair` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_folded_complex_fused_magnetic_pair_device_gate: no live run under flush |
| T6 | `dispersive_update_e` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: 3 pinning entries, none passes: bit_identity_gate: status None; no live run under flush |
| T6 | `dispersive_fused_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: 4 pinning entries, none passes: triton_dispersive_fused_pair_device_gate: no live run under flush |
| T6 | `fused_ade_chain` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_fused_ade_chain_device_gate: no live run under flush |
| T6 | `fused_ade_state` | kernel | emitted | no | refused | no | yes | no | refused | refused | yes | no | b_sele: spells a PML form and reads no chunk record (1 of 1 kernel units, e.g. fused_ade_state) |
| T6 | `fused_dispersive_chain` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_fused_dispersive_chain_device_gate: no live run under flush |
| T6 | `folded_dispersive_fused_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_folded_dispersive_fused_pair_device_gate: no live run under flush |
| T6 | `complex_ade` | kernel | emitted | refused | refused | refused | yes | no | refused | refused | yes | no | device: 2 pinning entries, none passes: triton_complex_ade_device_gate: no live run under flush |
| T6 | `complex_fused_ade_chain` | kernel | emitted | refused | refused | refused | yes | no | refused | refused | yes | no | device: triton_complex_fused_ade_chain_device_gate: no live run under flush |
| T6 | `folded_offdiag_dispersive_update_e` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_folded_offdiag_dispersive_device_gate: no live run under flush |
| T6 | `folded_offdiag_fused_ade_chain` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_folded_offdiag_fused_ade_chain_device_gate: no live run under flush |
| T7 | `special_kz` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_folded_complex_device_gate: no live run under flush |
| T7 | `bfast_curl` | kernel | emitted | yes | yes | yes | refused | refused | refused | refused | refused | refused |  |
| T7 | `beta_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_beta_fused_electric_pair_device_gate: no live run under flush |
| T7 | `beta_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_beta_fused_magnetic_pair_device_gate: no live run under flush |
| T7 | `bfast_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_bfast_fused_electric_pair_device_gate: no live run under flush |
| T7 | `bfast_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_bfast_fused_magnetic_pair_device_gate: no live run under flush |
| T7 | `complex_beta_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_complex_beta_fused_electric_pair_device_gate: no live run under flush |
| T7 | `complex_beta_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_complex_beta_fused_magnetic_pair_device_gate: no live run under flush |
| T7 | `folded_beta_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_folded_beta_fused_electric_pair_device_gate: no live run under flush |
| T7 | `folded_beta_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_folded_beta_fused_magnetic_pair_device_gate: no live run under flush |
| T7 | `folded_beta_complex_fused_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_folded_beta_complex_fused_pair_device_gate: no live run under flush |
| T7 | `folded_beta_complex_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_folded_beta_complex_fused_magnetic_pair_device_gate: no live run under flush |
| T8 | `cylindrical_triton` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: 2 pinning entries, none passes: triton_cylindrical_device_gate: the module's bytes moved since its run; no live run under flush |
| T8 | `cylindrical_complex` | kernel | emitted | yes | yes | yes | yes | no | refused | refused | yes | no | device: triton_cylindrical_complex_device_gate: no live run under flush |
| T8 | `cylindrical_real_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_cylindrical_real_fused_electric_pair_device_gate: no live run under flush |
| T8 | `cylindrical_real_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_cylindrical_real_fused_magnetic_pair_device_gate: no live run under flush |
| T8 | `cylindrical_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_cylindrical_fused_electric_pair_device_gate: no live run under flush |
| T8 | `cylindrical_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_cylindrical_fused_magnetic_pair_device_gate: no live run under flush |
| T9 | `nonlinear_update_e` | kernel | emitted | yes | yes | yes | refused | refused | refused | refused | refused | refused |  |
| T9 | `nonlinear_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | no | refused | refused | yes | no | device: triton_nonlinear_fused_magnetic_pair_device_gate: no live run under flush |
| T10 | `no_pml` | a_only | emitted | n/a | yes | refused | no | no | refused | refused | no | no | device: 2 pinning entries, none passes: bit_identity_gate: status None |
| T10 | `no_pml_ade` | a_only | emitted | n/a | refused | yes | yes | no | refused | refused | yes | no | device: triton_fused_ade_chain_device_gate: no live run under flush |
| T10 | `no_pml_conductive` | a_only | emitted | n/a | yes | yes | yes | no | refused | refused | yes | no | device: triton_no_pml_conductive_device_gate: no live run under flush |
| T10 | `no_pml_constitutive` | a_only | source text | n/a | refused | yes | yes | no | refused | refused | yes | no | device: triton_no_pml_constitutive_device_gate: no live run under flush |
| T10 | `no_pml_fused_electric_pair` | a_only | emitted | n/a | yes | yes | yes | no | refused | refused | yes | no | device: triton_no_pml_fused_electric_pair_device_gate: no live run under flush |
| T10 | `no_pml_stored_e` | a_only | emitted | n/a | refused | yes | yes | no | refused | refused | yes | no | device: 2 pinning entries, none passes: triton_fused_ade_chain_device_gate: no live run under flush |
| T10 | `complex_no_pml_curl` | a_only | emitted | n/a | yes | yes | yes | no | refused | refused | yes | no | device: triton_complex_no_pml_curl_device_gate: no live run under flush |
| T10 | `complex_no_pml_conductive` | a_only | emitted | n/a | yes | yes | yes | no | refused | refused | yes | no | device: triton_complex_no_pml_conductive_device_gate: no live run under flush |
| T10 | `complex_no_pml_stored_e` | a_only | emitted | n/a | refused | yes | yes | no | refused | refused | yes | no | device: 2 pinning entries, none passes: triton_complex_fused_ade_chain_device_gate: no live run under flush |
| T10 | `complex_conductive_fused_pair` | a_only | emitted | n/a | yes | refused | yes | no | refused | refused | yes | no | device: triton_complex_conductive_fused_pair_device_gate: no live run under flush |
| T11 | `fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `beta_real_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `beta_complex_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `complex_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `conductive_bfast_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `cylindrical_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `cylindrical_real_fused_hd_pair` | kernel | emitted | yes | yes | yes | refused | refused | refused | refused | refused | refused |  |
| T11 | `folded_complex_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `folded_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |
| T11 | `nonlinear_fused_hd_pair` | kernel | emitted | yes | yes | refused | refused | refused | refused | refused | refused | refused |  |

## Metal (92 rows: 13 a_only, 15 host, 64 kernel)

Refused cells, by reason:

- b_select: refused, no PML form and no chunk-record read in its emitted text: 3 (ade_update_p, cylindrical_complex_scan, templates)
- b_select: refused, no PML form and no chunk-record read in its own text: 1 (offdiag_weld_common)
- a_form: refused, no A-form witness in its own text: 2 (no_pml_constitutive, offdiag_weld_common)
- a_form: refused, no curl or constitutive A-form site in its emitted text: 7 (ade_update_p, complex_fused_ade_chain, complex_no_pml_offdiag_update_e, complex_no_pml_stored_e, cylindrical_complex_scan, no_pml_stored_e, templates)
- host_identical: refused, no test file of the table names or documents the module: 22 (beta_fused_electric_pair, beta_fused_magnetic_pair, bfast_fused_magnetic_pair, complex_dispersive_spine, complex_fused_electric_pair, complex_no_pml_offdiag_fused_electric_pair, conductive_fused_electric_pair, folded_beta_complex_fused_magnetic_pair, folded_beta_complex_fused_pair, folded_beta_real_fused_magnetic_pair, folded_beta_real_fused_pair, folded_complex_fused_pair, ...)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_beta_complex_fused_hd_pair.py measure no arithmetic of it: 1 (beta_complex_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_beta_real_fused_hd_pair.py measure no arithmetic of it: 1 (beta_real_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_bfast_fused_hd_pair.py measure no arithmetic of it: 1 (bfast_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_complex_conductive_fused_pair.py measure no arithmetic of it: 1 (complex_conductive_fused_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_complex_fields.py measure no arithmetic of it: 1 (complex_fields)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_complex_folded_offdiag_update_e.py measure no arithmetic of it: 1 (complex_folded_offdiag_update_e)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_complex_fused_ade_chain.py measure no arithmetic of it: 1 (complex_fused_ade_chain)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_complex_fused_magnetic_pair.py, test_metal_fused_pair_deposit_wiring.py measure no arithmetic of it: 1 (complex_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_conductive_fused_hd_pair.py measure no arithmetic of it: 1 (conductive_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_cylindrical_fused_electric_pair.py measure no arithmetic of it: 1 (cylindrical_fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_cylindrical_fused_magnetic_pair.py measure no arithmetic of it: 1 (cylindrical_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_cylindrical_real_fused_hd_pair.py measure no arithmetic of it: 1 (cylindrical_real_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_folded_complex_fused_hd_pair.py measure no arithmetic of it: 1 (folded_complex_fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_folded_complex_fused_magnetic_pair.py measure no arithmetic of it: 1 (folded_complex_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_folded_fused_dispersive_pair.py measure no arithmetic of it: 1 (folded_fused_dispersive_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_folded_fused_magnetic_pair.py measure no arithmetic of it: 1 (folded_fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_folded_fused_pair.py measure no arithmetic of it: 1 (folded_fused_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_fused_ade_chain.py measure no arithmetic of it: 1 (fused_ade_chain)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_fused_dispersive_pair.py, test_metal_fused_pair_deposit_wiring.py measure no arithmetic of it: 1 (fused_dispersive_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_fused_electric_pair.py measure no arithmetic of it: 1 (fused_electric_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_fused_hd_pair.py, test_metal_fused_hd_pair_nonlinear.py measure no arithmetic of it: 1 (fused_hd_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_fused_magnetic_pair.py, test_metal_fused_pair_deposit_wiring.py measure no arithmetic of it: 1 (fused_magnetic_pair)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_no_pml_curl.py measure no arithmetic of it: 1 (no_pml_curl)
- host_identical: refused, none of its test files compares the arithmetic with meep_gpu.stepping: test_metal_residue_fused_pairs.py measure no arithmetic of it: 3 (beta_complex_fused_electric_pair, beta_complex_fused_magnetic_pair, bfast_fused_electric_pair)
- device_identical:flush: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 3 (bfast_curl, complex_dispersive_spine, folded_offdiag_dispersive_update_e)
- dev_evidence:flush: refused, no development evidence recorded on this subject: 74 (ade_update_p, beta_complex_fused_electric_pair, beta_complex_fused_hd_pair, beta_complex_fused_magnetic_pair, beta_fused_electric_pair, beta_fused_magnetic_pair, beta_real_fused_hd_pair, bfast_curl, bfast_fused_electric_pair, bfast_fused_hd_pair, bfast_fused_magnetic_pair, complex_conductive_fused_pair, ...)
- certified:flush: refused, no ledger entry pins the module's file (its device text may be pinned by digest elsewhere; the board does not attribute that): 3 (bfast_curl, complex_dispersive_spine, folded_offdiag_dispersive_update_e)

Modules of the table directory that no step list names: folded_pass.

| Step | Module | Kind | Witness | b select | a form | host identical | device identical:flush | dev evidence:flush | certified:flush | Why not (first) |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | `chunk_bounds` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `launch` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `metal_dispatch` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `coverage` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `device` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M11 | `softfp64` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `__init__` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `arms` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `barrier` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `coefficient_pack` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `corpus` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `plans` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `preconditions` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `registry` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M0 | `subnormal` | host | - | n/a | n/a | n/a | n/a | n/a | n/a |  |
| M1 | `shaders` | kernel | emitted | yes | yes | refused | yes | yes | yes |  |
| M1 | `symmetry` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M1 | `nonlinear_update_e` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M2 | `fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | yes | yes |  |
| M2 | `fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | yes | yes |  |
| M2 | `fused_dispersive_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M2 | `nonlinear_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M2 | `fused_ade_chain` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M3 | `fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M3 | `folded_fused_hd_pair` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M4 | `conductive_pml` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M4 | `conductive_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M4 | `conductive_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M5 | `templates` | kernel | emitted | refused | refused | refused | yes | refused | yes |  |
| M5 | `complex_fields` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M5 | `complex_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M5 | `complex_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M5 | `complex_fused_hd_pair` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M5 | `complex_conductive_pml` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M5 | `folded_complex` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M5 | `folded_complex_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M5 | `folded_complex_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M5 | `folded_complex_fused_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M6 | `folded_fused_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M6 | `folded_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M6 | `folded_fused_dispersive_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M7 | `offdiag_update_e` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M7 | `folded_offdiag_update_e` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M7 | `offdiag_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M7 | `folded_offdiag_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M7 | `complex_folded_offdiag_update_e` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M7 | `folded_complex_offdiag_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M7 | `offdiag_weld_common` | kernel | source text | refused | refused | refused | yes | refused | yes |  |
| M8 | `dispersive_update_e` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M8 | `folded_dispersive_update_e` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M8 | `folded_offdiag_dispersive_update_e` | kernel | emitted | yes | yes | yes | refused | refused | refused |  |
| M8 | `complex_dispersive_update_e` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M8 | `complex_dispersive_spine` | kernel | emitted | yes | yes | refused | refused | refused | refused |  |
| M8 | `ade_update_p` | kernel | emitted | refused | refused | yes | yes | refused | yes |  |
| M9 | `bfast_curl` | kernel | emitted | yes | yes | yes | refused | refused | refused |  |
| M9 | `bfast_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `bfast_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `bfast_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `special_kz` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M9 | `folded_beta` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M9 | `beta_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `beta_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `beta_real_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `beta_complex_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `beta_complex_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `beta_complex_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `folded_beta_real_fused_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `folded_beta_real_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `folded_beta_complex_fused_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M9 | `folded_beta_complex_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M10 | `cylindrical_real` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M10 | `cylindrical_complex_scan` | kernel | emitted | refused | refused | yes | yes | refused | yes |  |
| M10 | `cylindrical_complex` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M10 | `cylindrical_real_fused_magnetic_pair` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M10 | `cylindrical_real_fused_electric_pair` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M10 | `cylindrical_real_fused_hd_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M10 | `cylindrical_fused_magnetic_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M10 | `cylindrical_fused_electric_pair` | kernel | emitted | yes | yes | refused | yes | refused | yes |  |
| M10 | `cylindrical_complex_fused_hd_pair` | kernel | emitted | yes | yes | yes | yes | refused | yes |  |
| M12 | `complex_conductive_fused_pair` | a_only | emitted | n/a | yes | refused | yes | refused | yes |  |
| M12 | `complex_fused_ade_chain` | a_only | emitted | n/a | refused | refused | yes | refused | yes |  |
| M12 | `complex_no_pml_conductive` | a_only | emitted | n/a | yes | yes | yes | refused | yes |  |
| M12 | `complex_no_pml_curl` | a_only | emitted | n/a | yes | yes | yes | refused | yes |  |
| M12 | `complex_no_pml_offdiag_fused_electric_pair` | a_only | emitted | n/a | yes | refused | yes | refused | yes |  |
| M12 | `complex_no_pml_offdiag_update_e` | a_only | emitted | n/a | refused | yes | yes | refused | yes |  |
| M12 | `complex_no_pml_stored_e` | a_only | emitted | n/a | refused | yes | yes | refused | yes |  |
| M12 | `no_pml_conductive` | a_only | emitted | n/a | yes | yes | yes | refused | yes |  |
| M12 | `no_pml_conductive_fused_electric_pair` | a_only | emitted | n/a | yes | refused | yes | refused | yes |  |
| M12 | `no_pml_constitutive` | a_only | source text | n/a | refused | yes | yes | refused | yes |  |
| M12 | `no_pml_curl` | a_only | emitted | n/a | yes | refused | yes | refused | yes |  |
| M12 | `no_pml_fused_electric_pair` | a_only | emitted | n/a | yes | refused | yes | refused | yes |  |
| M12 | `no_pml_stored_e` | a_only | emitted | n/a | refused | yes | yes | refused | yes |  |
