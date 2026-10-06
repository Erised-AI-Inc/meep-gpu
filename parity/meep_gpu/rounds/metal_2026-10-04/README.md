# Metal round, 2026-10-04

The inputs and the driver of a Metal certification round on a Mac that does not hold
the evidence archive: every Metal gate on one Apple GPU, written where the rebind
reads it. The manifest describes what the Metal gates of this commit read from
`parity/meep_gpu/results/`; a change to what they read cuts a new round directory
rather than editing this one, so a round can always be checked against the manifest it
was verified with.

| File | What it is |
|---|---|
| `run_round.sh` | Runs the round stage by stage: preflight, inputs, fleet, status, pack. Each stage refuses to start if the one before it did not finish, and the fleet and pack stages check the commit and the clean checkout again, because `all` resumes at the first unfinished stage. |
| `INPUT_FILES.txt` | The 7 files the bundle carries, relative to `results/`. |
| `INPUT_DIRS.txt` | The 4 directories they sit in. Only the listed files ship from each, not the whole directory. |
| `INPUTS_SHA256SUMS` | The digest of each file, in the format `shasum -a 256 -c` reads. |
| `metal_round_inputs.tgz.sha256` | The digest of the bundle cut from the evidence archive for this manifest. |

The bundle (`metal_round_inputs.tgz`, 1,158,473 bytes; 14,040,445 bytes unpacked) is
not in the repository: its census rows record the interpreter paths of the host that
cut them. It travels beside the checkout. `parity/meep_gpu/build_metal_round_inputs.py`
packs it from the archive (deterministically: the same files give the same digest),
verifies it, and unpacks it.

## What ships, and what the round cuts itself

| Files | Read by | Why it ships |
|---|---|---|
| `metal_coverage_2026-09-04_m0complex/examples.jsonl`, `tests.jsonl`, `tests_param_matched.jsonl` | The 10 `*_fused_hd_pair` gates: which corpus rows sit in each product's cells | The standing census row set. A gate re-lifts and re-checks every row on the Mac it runs on; the census only chooses the rows. Without `tests_param_matched.jsonl` the row set shrinks without an error. |
| `metal_coverage_2026-09-04_m0complex/tests_param.jsonl` | `gate_metal_cylindrical_complex_scan`, its `corpus_shapes` leg | Without the census that leg records `checked=0` and the gate still releases. |
| `h_to_d_seam_2026-09-04/h_to_d_seam.jsonl` | The same 10 gates: the rows whose seam withdraws | A record of the H-to-D seam per corpus row. |
| `metal_pml_2026-08-14/provenance.json` | `gate_metal_spine_byte_neutrality` (exit 75 without it) | The frozen baseline: 18 Metal kernel-source digests and 6 host-module digests, with no host or architecture field. The gate compiles nothing; its claim is a property of the source tree, so every architecture reads the same baseline. |
| `complex_signed_zero_2026-08-12/results/run_2345/probe.json` | Every lift child (`measure_predicate_coverage.py` opens its `PROBE` before the battery) | No Metal battery reads its content: it is an NVIDIA (`cupy`) record. It ships only because the child refuses to start without a readable file there. The NVIDIA round refuses this directory in `results/`, so do not unpack this bundle into a tree that will run an NVIDIA round. |

**The four complex expansion probes never ship.** Which complex-multiply arm a Mac's
NumPy reference takes is a measured fact of that Mac, so the round cuts all four on the
Mac it runs on: `recut_metal_gates.sh` runs the three standalone probes, then
`gate_metal_complex` (whose expansion leg writes the fourth), and only then every other
gate, and it unsets any probe variable inherited from the calling shell. The dated
records under `metal_complex_audit_2026-08-16/`, `metal_special_kz_2026-08-16_audit/`,
`metal_folded_complex_2026-08-16/` and `metal_cylindrical_complex_2026-08-16/` were
measured on the Apple M1 Max; a gate that names one of them on another Mac would be
licensing that Mac's arms from the M1 Max. The status stage warns on any artifact that
names such a record.

Nothing in the bundle is measured per architecture.

## What a Mac needs beside the bundle

- The reference MEEP and environment (`bash parity/meep_gpu/build_meep_133_macos.sh`;
  the preflight runs `tools/check_install.py --require-reference`).
- **A MEEP 1.33.0 source tree**, named by `MGPU_SITE_MEEP_SOURCE` (it must hold
  `python/examples` and `python/tests`). The lift legs of 9 of the 10
  `*_fused_hd_pair` gates re-run corpus rows from it.
- **The interpreters the census rows name.** Each lift child runs in the interpreter
  its census row records, an absolute path on the host that cut the census, and five
  rows of the examples leg name two further environments (MEEP built with GDSII
  support, and the sigma-reader patch). The inputs stage checks every one by name: an
  absent interpreter crashes those gates, because their lift legs catch only a
  timeout. Until the lift legs can map a census interpreter onto this host's, a Mac
  whose paths differ runs with `ALLOW_NO_LIFT=1`, and those gates do not release
  there. **None of the 10 is cited by the Metal table** (`metal_dispatch.ARM_CERTIFICATION`),
  so they do not decide whether the Mac's architecture is certified.

## Running it

    conda activate meep-gpu-ref
    export REPO=$HOME/meep-gpu  EXPECT=<full commit hash>  STAMP=<date>_<architecture suffix>
    export INPUTS=$HOME/metal_round_inputs.tgz
    export MGPU_SITE_MEEP_SOURCE=<MEEP 1.33.0 source tree>
    shasum -a 256 $INPUTS              # must equal metal_round_inputs.tgz.sha256
    zsh $REPO/parity/meep_gpu/rounds/metal_2026-10-04/run_round.sh all > $HOME/metal_round_$STAMP.out 2>&1 &

On the host that holds the evidence archive, leave `INPUTS` unset: the inputs stage
then verifies the archive's files against the manifest in place. It never unpacks
through a symbolic link, so a `results/` tree whose entries link into the archive is
checked, not written.

`status` prints how many gate artifacts released, how many of the welds the Metal
table cites released, and any artifact that names a dated archive probe. `pack` writes
`metal_round_$STAMP.tgz` and its `.sha256` (to `$OUT_DIR`, default `$HOME`): every
`results/metal_*_$STAMP` directory and the round's logs, without the lift legs' work
directories.

## After the round

The rebind runs on the host that holds the evidence archive
(`docs/development/certification.md`, "Certifying another Apple GPU"). **The Metal
ledger keeps one run per GPU architecture** (`runs[<architecture>]`), as the NVIDIA
ledgers keep one per compute capability: on the same bytes, `rebind_metal_welds.py
--stamp $STAMP` adds this Mac's runs beside the Apple M1 Max's, and the GPU is
certified once every cited weld has one. A gate artifact that records its own GPU
must name this campaign's, or its weld is skipped by name. On bytes that moved the
rebind names the architectures whose runs it would strand, and `--supersede
<architecture>` drops them by name.

**The rebound ledger is not committed on its own**: it admits this GPU before its
route run exists, and the route-record contract test fails until it does. It is
copied, uncommitted, into this Mac's checkout of the same commit (the only file that
differs), the Metal route campaign runs there under
`<stamp of metal_dispatch.METAL_DRIVER_ROUTE_GATE>_<architecture>`, its legs come back,
`recut_driver_dispatch_record.py --backend metal` adds its route run beside the first,
and the weld runs and the route run are committed together. Every route row reads the
architecture's verdict off the ledger in the tree it runs in, and the recut refuses a
row that read it not certified; nothing in the campaign reads git state, and the route
record does not bind the ledger
(`docs/development/certification.md`, "Records per Apple GPU architecture").
