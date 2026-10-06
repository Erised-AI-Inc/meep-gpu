# H200 round, 2026-10-03

The prepared certification round that would add an NVIDIA H200 (compute capability
9.0, Hopper) beside the RTX A6000 (8.6), and the timing that would follow it. The
round is prepared; no 9.0 record ships in 0.9.2, so compute capability 9.0 is
supported, not certified. Its scripts target the pre-release commit
`516026f3d3bcc2a4b8cb40ddbdbd3d8b64752a05` (commit hashes here name the
pre-publication development history), against MEEP 1.33.0 built from source in
single precision with MPI (`build_meep_133_linux.sh`). The records of 0.9.2 on
compute capability 8.6 come from the later RTX A6000 rounds stamped 2026-10-05 on the
release files.

| File | What it is |
|---|---|
| `run_round.sh` | Runs the round stage by stage (preflight, inputs, Triton shards A to D, stencil welds, bit identity, the flush expansion record, the 70 CUDA legs, status, pack). Each stage refuses to start if the one before it did not finish. |
| `run_cuda_welds.sh` | The 70 CUDA legs, keep and flush policies, called by `run_round.sh`. |
| `INPUT_DIRS.txt`, `INPUT_FILES.txt`, `INPUTS_SHA256SUMS`, `meep_gpu_round_inputs.tgz.sha256` | The manifest of the inputs bundle the gates read: 992 files from the certification evidence archive, with the digest of each and of the bundle. |

The hand-off runbooks are kept in the development repository, not in this one: the
certification runbook, the steps the round follows on a Slurm cluster (build the
reference MEEP, check the card, run every gate as one batch job, pack the results);
the timing runbook, the identical-case timing ladder on one H200 against stock MEEP
on the same node's CPU cores (`docs/development/timing.md`), run after the H200's
records are bound; and the script with which the RTX A6000 ran the round.

The RTX A6000 ran the same round on the pre-release commit on 2026-10-03, with the
same scripts and inputs bundle: every gate released (Triton 33 of 33, 11 of 11, 1 of
1, the five composition legs, both stencil welds, bit identity, the flush record;
CUDA 70 of 70), stamp `2026-10-03_cc86`.

The inputs bundle itself (`meep_gpu_round_inputs.tgz`, 25 MB) is not in the
repository: it is a slice of the evidence archive, whose records carry the paths and
host names of the machines that produced them. It travels in the hand-off file
`h200_round_handoff.tgz` (sha256
`f6c6355daf3565eccd0d5e1a338354fb62c3bc12b9780a024d29a81f9efb7ad5`), which also
carries the two scripts above, so a round runs from the hand-off without a second
clone. `run_round.sh` verifies every file of the bundle against `INPUTS_SHA256SUMS`
before the first gate.

The hand-off unpacks to `~/h200_round/`; its files serve any compute capability 9.0
card. The A6000 ran the same four files on 2026-10-03, packed then as
`h100_round_handoff.tgz` (sha256 `6d67ec9ce941…`).

The rebinds that turn the round's results into ledger records run on the laptop that
holds the evidence archive (`docs/development/certification.md`, "Certifying another
compute capability").
