# Certifying a new GPU

This release certifies two GPUs: the NVIDIA RTX A6000 (compute capability 8.6) and the
Apple M1 Max (Apple GPU architecture `applegpu_g13s`). Every other card in the supported
range already runs the compiled kernels by default, recorded as uncertified
([Certification on an NVIDIA GPU](../guides/kernel-dispatch.md#certification-on-an-nvidia-gpu),
[Certifying another Apple GPU](certification.md#certifying-another-apple-gpu)). This page
takes you through the round that certifies one more, written for two cases:

- **[Part A](#part-a-a-new-nvidia-compute-capability)**: an NVIDIA GPU of a compute
  capability the release does not certify, for example an H200 (compute capability 9.0),
  on a Linux x86_64 host;
- **[Part B](#part-b-a-new-apple-gpu-architecture)**: a Mac whose Apple GPU has an architecture
  the release does not certify, for example an M3.

A round writes records and nothing else. The gates run on your card in chunks and leave
artifacts; the bind then adds one run per weld for your GPU, `runs["9.0"]` or
`runs["<architecture>"]`, beside the runs already recorded, and leaves those byte for byte;
the route campaign then shows that the dispatched route is byte-identical to the array path
on your card. You send the artifacts, the bound ledgers and the logs back, the maintainers
check them, and the records are published in a 1.0.x release. No package file is edited,
and nothing is installed with `pip`: every driver imports the package from your clone.

[Running the certification harness](certification.md) explains what the gates, ledgers
and route campaigns are, and `parity/meep_gpu/rounds/1.0/RUNBOOK.md` is the round's own
runbook. You do not need to read either first; this page links to them where a step
depends on them.

## Before you start

### What you need

| | Part A: NVIDIA | Part B: Apple |
|---|---|---|
| The GPU | one or more cards of the new compute capability (7.0 to 9.0) on Linux x86_64, with an NVIDIA driver of the CUDA 12 series (the 1.0 round's host ran 545.23.08) | an Apple silicon Mac with the new GPU, on macOS 14 or later |
| On the host | a conda installation (Miniforge), `git`, `bash`, glibc 2.28 or later, network access to conda-forge, PyPI and the MEEP release | a conda installation (Miniforge), `git`, `zsh`, the Command Line Tools (or Xcode), the same network access |
| Disk | the reference environment (8.9 GiB measured on the 1.0 round's Linux host) and its package cache, plus the clone's results and the lanes' logs and caches | the reference environment (2.0 GiB measured on the 1.0 round's Mac) and its package cache; the round stops before a gate when less than about 9.5 GiB is free (`--disk-floor-kb`) |
| Wall time | the build: 377 s measured (8 build jobs, an empty package cache). The gates: four chunks of 23,861 to 23,895 s each at RTX A6000 prices (`chunks.py show` prints them), one card per chunk. The route: 12,733 s at RTX A6000 prices (the 1.0 round's took 4,930 s) | the build: 296 s measured on an M1 Max (4 build jobs). The gates: three chunks of 1,685, 1,423 and 1,422 s at M1 Max prices, one after another. The route: 4,752 s with 3 lanes in the 1.0 round on an M1 Max (12,648 s earlier, on a loaded one) |
| From the maintainers | the round's input records (below) | the same archive |

### The rules of a round

- **One commit.** Everything runs on a clean clone of the release tag `v1.0.0`. The
  drivers refuse a checkout whose HEAD is another commit or whose tracked files differ.
- **Nothing is committed.** The only tracked files that change are the ledgers the bind
  writes: three on NVIDIA, one on Apple. Never edit a ledger by hand.
- **Results go under `parity/meep_gpu/results/`**, which git ignores. Every results
  directory name is one-shot: never delete, rename or edit one, and never re-run into one
  by hand. The only move a round makes is a set-aside the bind prints, which moves a
  directory under `results/set_aside_<stamp>/` and deletes nothing.
- **The round stamp is the release's.** The chunk tooling reads it from
  `meep_gpu/route_stamps.py` (`2026-10-15_10` in 1.0), and every chunk's stamp adds your
  device to it (`2026-10-15_10_cc90_c0`, `2026-10-15_10_<architecture>_m0`), so no root of
  your round can equal or extend a root of the certified GPU's. You do not choose a stamp,
  and you do not edit the registry.
- **The route campaign names are one-shot too**, one per GPU. Run the route stage only
  when the steps before it read GO.

### Placeholders

Each command below is written for a shell in which these are set. Set them once:

| Variable | What it is |
|---|---|
| `REPO` | your clone of the release (Part A on `<your-gpu-host>`, Part B on `<your-mac>`) |
| `EXPECT` | the full hash of the commit the tag `v1.0.0` names, `git -C "$REPO" rev-parse HEAD` |
| `BUNDLE` | the input archive `round_inputs_1_0.tgz` the maintainers sent you, outside the clone |
| `ROUND` | the release's round stamp, read once the environment is active: `ROUND=$(cd "$REPO" && python -c "from meep_gpu import route_stamps; print(route_stamps.ROUND)")` |
| `INDEX` | your chunk index, a file outside the clone (`$HOME/meep-gpu-round/INDEX_cc90.json`) |
| `SEL`, `VERIFY` | the selection and the report `chunks.py verify` writes, files outside the clone |
| `CC` | Part A: the compute capability, as `major.minor` (`9.0`) |
| `ARCH` | Part B: the GPU architecture Metal names (B3) |
| `LANES` | Part A: a directory outside the clone for the lanes' logs (`$HOME/meep-gpu-round/lanes`) |
| `LOCKS` | Part A: the directory the lanes take their GPU locks in, one per machine (`$HOME/meep-gpu-locks/<machine>`) |

### The round's input records

Many gates read measured records that the repository does not carry: census rows, coverage
boards, seam records, the Metal spine baseline and earlier artifacts. A gate whose input is
absent refuses, so a round on a bare clone releases nothing. The maintainers send them to
you privately, as one archive for both machines. Do not publish it or commit any part of
it: its records carry the machine stamps of the hosts that cut them.

| File | Size | sha256 |
|---|---|---|
| `round_inputs_1_0.tgz` | 43,969,616 bytes, 1,782 files | `1cc0c93349633540823540645225134cfa98954c702c0a8ef505cb1a4ca2821a` |

The release carries the same digest in `parity/meep_gpu/rounds/1.0/INPUT_BUNDLE.sha256`,
and every file of the archive, with its sha256, in `INPUTS_SHA256SUMS` beside it.
`round_inputs.py` checks both before it writes anything (A2, B2).

## Part A: a new NVIDIA compute capability

### A1. The reference environment

Every certification is measured in one reference environment per platform: MEEP 1.33.0
from the release tarball (pinned by SHA-256), single precision, MPI and OpenMP, with
libGDSII and no floating-point contraction, on top of exactly the packages the Linux locks
pin, with the MEEP corpus the lift legs walk kept inside it
([The MEEP a round is measured against](certification.md#the-meep-a-round-is-measured-against)).
The repository's own script builds it, end to end. On `<your-gpu-host>`:

```bash
git clone https://github.com/Erised-AI-Inc/meep-gpu.git meep-gpu
cd meep-gpu
git checkout --detach v1.0.0
export REPO="$PWD" EXPECT="$(git rev-parse HEAD)"
MEEP_BUILD_JOBS=8 bash parity/meep_gpu/build_meep_133_linux.sh 2>&1 | tee "$HOME/build_meep_133_linux.log"
```

The script clears its own environment, creates the conda environment `meep-gpu-ref` under
`$CONDA_ROOT` (default `$HOME/miniforge3`; `MEEP_ENV_NAME` or `MEEP_ENV_PREFIX` names
another), builds in `$MEEP_BUILD_ROOT` (on a cluster point it at scratch storage), and
refuses to finish unless what it built is the reference: the lock's packages artifact for
artifact, MEEP 1.33.0 in single precision with MPI and libGDSII, no fused multiply-add in
the field updates, MEEP's own words on four cases equal to the recorded digests, and the
identity verdicts in keep and in flush. It also links the driver's `libcuda.so.1` into the
environment for Triton's launcher, and keeps the corpus (the tarball's `python/` tree, with
the file outside it that two corpus rows read) with its sha256 list, named in
`<prefix>/share/meep-gpu-build/reference_build.json`. The only pieces of the host the
environment uses are the GPU driver and glibc.

An environment built by an earlier version of the script whose record names no corpus
siblings is refused by every round reader; bring it up to date in place, without
rebuilding:

```bash
bash parity/meep_gpu/build_meep_133_linux.sh --refresh-corpus
```

Then, in every shell and job of the round:

```bash
conda activate meep-gpu-ref
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
PYTHONPATH="$REPO" python tools/check_install.py --require-reference
python -c "import meep; from meep_gpu import backends; print('MEEP flushes subnormals:', backends.subnormals_flushed())"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv
```

The check must pass (its MEEP line reads `1.33.0, single precision, MPI build, libGDSII:
the reference's build`), the second line must print `True` (the flush legs take the host's
flush from MEEP), and `nvidia-smi` must list your cards with the compute capability you put
in `CC`. The check runs with the clone on `PYTHONPATH` because the reference environment
holds no installed copy of the package; the round's drivers put the clone there themselves,
so do not export it for them. Do not export `CC`, `TRITON_LIBCUDA_PATH` or any `MEEP_GPU_*`
variable either: each lane takes Triton's launcher compiler and the libcuda link from the
reference environment and logs both, and sets each gate's own switches.

### A2. Stage the inputs

```bash
cd "$REPO"
mkdir -p parity/meep_gpu/results
python parity/meep_gpu/rounds/1.0/round_inputs.py unpack --bundle "$BUNDLE"
git status --porcelain
```

`unpack` checks the archive's digest against the release's, extracts only files that are
absent, and ends with `1782 of 1782 verified`. The last line must print nothing:
`results/` is ignored, and nothing tracked has moved.

### A3. Your chunk index

The gates run in chunks, each under a stamp of its own on the card it is given. Cut the
index of your card once, outside the clone:

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/chunks.py partition --repo ../.. --machine nvidia --capability "$CC" \
    --new-device --out "$INDEX"
python rounds/1.0/chunks.py show --index "$INDEX"
python rounds/1.0/chunks.py check --index "$INDEX" --repo ../.. --results results
```

- **`--new-device` says your card is new**: the index records it, and every later step
  holds the index to the ledgers in both directions, so a new card is never admitted
  silently. Without it the index names a certified device, and the gate stage refuses it.
- **The default is four chunks**, balanced by the round's measured walls (123 units in
  all; a unit is the smallest piece of evidence that binds on its own, a CUDA campaign
  subdirectory or one Triton gate under both policies). `--chunks N` cuts N. `show` prints
  each chunk's stamp, units and wall at RTX A6000 prices:

  ```
  nvidia cc90 (new device), round 2026-10-15_10, next_k 4
  c0 2026-10-15_10_cc90_c0: 30 units, 23861 s, 12 roots (partition)
  ...
  ```

- **`check` must read `CHECK OK (0 problem(s))`**: no two chunks, and no chunk and a
  directory already under `results/`, share or extend a root.
- Keep the index for the whole round and never edit it by hand. The tools that change it
  (`add`, below) allocate a new chunk id each time and never reuse one. It is a
  `meep_gpu.cert_chunks/2` document; an index of the first chunk tooling
  (`cert_chunks/1`) is refused by name, so cut a new one rather than reuse an old one.

### A4. The checks before starting

For each chunk you are about to launch:

```bash
cd "$REPO"
python parity/meep_gpu/rounds/1.0/legs.py check --repo . --capability "$CC" --new-capability \
    --git-expect "$EXPECT" --chunk-index "$INDEX" --chunk c0 --python "$CONDA_PREFIX/bin/python"
```

**GO on `CHECK OK (0 problem(s))`.** It checks every gate script and tool, the fleet driver,
the flush licence, the hygiene guard, the clean checkout at `EXPECT`, the chunk's roots, the
corpus the reference environment's record names (file by file), the census and seam records
the Triton H-to-D gates lift from, and that the index's device is your card's. Without
`--new-capability` the same check refuses, by design: each NVIDIA table's
`capability_admission` does not admit your capability yet (that is what the round is about
to change), and the index says the device is new. The flag is refused on a certified
capability and on the route stage. Fix whatever else it names before going on.

### A5. CERT-3: the gate chunks

Launch each chunk on the card or cards you give it, side by side with the others:

```bash
cd "$REPO"
python -u parity/meep_gpu/rounds/1.0/round.py launch gates --repo "$REPO" \
    --chunk-index "$INDEX" --chunk c0 --gpu-ids <card> --capability "$CC" --new-capability \
    --home "$LANES" --lock-dir "$LOCKS" --max-locks <cards> --expect "$EXPECT" \
    --python "$CONDA_PREFIX/bin/python"
```

- **One lane per card named** in `--gpu-ids` (comma-separated). A chunk on one card runs
  in one lane at its full wall; on two cards it splits into two lanes of about half of it
  each (`chunks.py plan --index "$INDEX" --chunk c0 --repo ../.. --results results --out <dir> --lanes 2`, from `parity/meep_gpu`, prints the split). `--max-locks` caps how many cards
  the round's lanes hold at once, across every chunk; the default is 4.
- **Each lane takes its card's lock** under `LOCKS` after two idle readings of the card a
  minute apart, runs a card and toolchain check first (compute capability `CC`, CuPy
  13.5.1, Triton 3.1.0, MEEP 1.33.0 in single precision with MPI, the package from the
  clone) and stops if it fails, and releases its lock when its last step returns.
- **Follow** `$LANES/<chunk stamp>/round.log` and the lanes' `progress.log` beside it: one
  line per step with N of D passing, and a heartbeat every 120 s from the running gate.
  `launch` returns at once; the stage runs in its own session.
- **On a batch scheduler** (Slurm, PBS), submit one job per chunk holding its cards on one
  node, and use `round.py run gates` with the same arguments instead of `launch`: `run`
  stays in the foreground, and a scheduler ends every process a job started when the job
  script exits. Pass the cards the job holds as `--gpu-ids`, the indices `nvidia-smi
  --query-gpu=index --format=csv` lists inside the job, and keep `LOCKS` per machine,
  since lock directories are matched by card index.
- **Resume** after a lost connection or a job's time limit: run the same command again
  with `--resume` added. Each lane skips every step whose last row and artifact pass, and
  a Triton fleet step runs only its gates without a passing artifact, into a fresh
  `<root>_r2` directory. To stop a lane between steps, create `STOP` in its directory. A
  lane that died holding a lock leaves `$LOCKS/gpu_lock_<n>` naming its pid: remove it
  only when that pid is gone.

**GO** for a chunk when `$LANES/<chunk stamp>/GATES_DONE` exists and

```bash
python parity/meep_gpu/rounds/1.0/round.py status --home "$LANES/<chunk stamp>"
```

reads every step passing. A step that did not release has its reasons and its log in its
lane's `steps.jsonl`. If it died of its environment (a node lost, a full disk), resume the
chunk as above, or, where the resume refuses a directory by name, run its units again in a
new chunk (A6). If a gate ran and refused on your card, that is a finding about the card:
stop, and send the package of A9 with what you have.

#### Sharing a card with another user's idle process

On a shared host a card can hold another user's process that keeps memory and computes
nothing. A lane never takes a card that holds any compute process, unless you name that
process:

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
python -u parity/meep_gpu/rounds/1.0/round.py launch gates ... \
    --share-idle-foreign <uuid>:<pid>[,<pid>] --share-min-free-mib <MiB>
```

The card is then admitted when, on both idle readings, every compute process on it is one
you named, those processes and the card read 0 % utilisation, and the card's free memory is
at least `--share-min-free-mib`; the lane records the shared card and the processes in its
`progress.log` and in its lock. The bound is your units' own peak plus what you keep free
for the card's owner, and the peak is one a unit reached, not a smoke's reading: the 1.0
round set 12,288 MiB, 4,096 MiB for its units (from the 270 MiB a smoke read) and 8,192 MiB
for the owner, and a Triton H-to-D lift on the shared card then used 12,348 MiB. The options
belong to the gate stage only; the route stage refuses them.

**A chunk that holds Triton fleet units needs an unshared card.** The options are the lane's
own card admission. The Triton fleet driver (`drive_triton_weld_gates.py`), which runs every
`triton/fleet/...` unit and `triton/composition`, still refuses a card that carries any
compute process: in the 1.0 round, on a shared card, it refused 11 legs (a flush fleet shard
of 10 gates and its environment probe) and no gate ran. To share a card, split a chunk that
has not launched into its fleet units and the rest (`chunks.py add --units <unit>,... --why
"<why>"`, twice, the unit ids `chunks.py show` lists), launch the fleet chunk on an unshared
card and the other on the shared one, and never launch the chunk you split.

**Yield the card the moment its owner needs it**: one of their processes computing, a new
process on the card, or its free memory falling below what you keep for them. Send
`SIGTERM` to the card's lane (the pid in `$LOCKS/gpu_lock_<n>/pid`): it stops its gate's
process group, which frees the device, and releases the lock. Check that the lock directory
is gone and that no process of yours is left on the card. The interrupted unit does not
hold, and neither do the units its lane had not reached; they run again in a new chunk
(A6), on whatever card is free.

### A6. Verify the chunks, and run again what does not hold

When every chunk has finished (or stopped), judge every unit against the release, from
`parity/meep_gpu`:

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/chunks.py verify --index "$INDEX" --repo ../.. --results results \
    --selection "$SEL" --json "$VERIFY"
```

For each unit it reads every launch whose record names it and takes the newest run whose
import record holds on the release tree, whose verdicts pass and whose card is your
index's (read from the artifacts, never assumed). **GO on `VERIFY OK: 123 of 123 units
hold`.** Otherwise:

```bash
python rounds/1.0/chunks.py add --index "$INDEX" --repo ../.. --from-verify "$VERIFY" \
    --why "<what happened>"
python rounds/1.0/chunks.py show --index "$INDEX"
```

allocates a new chunk holding exactly the units no run holds (`--units <unit>,...` names
them instead); check it, launch it as in A4 and A5, and verify again. Every earlier run stays
as evidence, and none of this touches another unit's run.

### A7. The bind: add the records for your capability

On the same host, in the same environment, with no GPU. Print the bind's commands:

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/legs.py bind --chunks "$INDEX" --selection "$SEL" | tee "$LANES/bind_commands.txt"
```

It prints the commands that bind exactly the selection and never runs them. The first is
`chunks.py verify --against "$SEL"`, which judges again, writes nothing and exits 1 when
the choice or the bytes of a selected run moved since the selection was written: then
write the selection again (A6) and print the bind again. The others are the CUDA records
and rebinds (`record_cuda_regate.py`, `rebind_cuda_welds.py`, once per selected stamp in
launch order), the ADE re-cut, the Triton rebinds (`rebind_triton_welds.py --only`, once
per selected stamp), the bit identity, the composition records and the family
re-certification, and last `neutralize_ledger_labels.py`. Run them **in order, one at a
time**: each first as printed (report mode, which writes nothing), read its report, then the
same command with `--write` added (`legs.py bind ... --write` prints them that way), then
the next, since a later tool reads what an earlier one wrote. Keep every report:

```bash
<the command as printed> 2>&1 | tee -a "$LANES/bind_reports.txt"
<the command as printed> --write 2>&1 | tee -a "$LANES/bind_reports.txt"
```

- **`# STOP` lines.** A run a tool would bind against the selection (a later run of a unit,
  an unreleased subdirectory left in a CUDA campaign by a yielded lane) is named on a
  `# STOP` line with the command that sets it aside, and while one stands the bind prints no
  command that writes a ledger, ends with `# BIND WITHHELD` and exits 1. Run each set-aside
  command as printed (it moves the run under `results/set_aside_<stamp>/` and deletes
  nothing), write the selection again (A6) and print the bind again.
- **Stop**, and write nothing more, if a report asks for `--supersede`, names a file whose
  digest differs from the checkout, or says an artifact's import record disagrees with the
  tree: the artifacts then describe other bytes than the release's. Never pass
  `--supersede` in this round: it is for a round that moves bound bytes, and it would drop
  the certified card's records.
- **Lines the 1.0 round's own bind met, which you will meet too.** None of them is a reason
  to stop:
  - two `# REFUSED rebind_triton_welds.py --only` comment lines, for `triton/fleet/bfast`
    and `triton/fleet/special_kz`, whose weld keys the Triton ledger has never held. They
    withhold nothing: the tool would refuse a whole `--only` naming a key the ledger
    lacks, so each key is left out, and the two units' runs still serve the family
    re-certification;
  - a rebind exits 1 when it skips an entry, after it has written the others:
    `rebind_cuda_welds.py` skips the one CUDA weld the ledger never seeded and
    `rebind_triton_welds.py` skips `triton_cylindrical_device_gate`, whose entry declares
    a mixed policy. Judge each line by its summary, and never run the lines under
    `set -e`, or the ones after it stay unbound;
  - `record_cuda_ade_recut.py` refuses and writes nothing (the kernels it re-cuts are
    compiled differently from its record); the ADE weld itself is rebound by
    `rebind_cuda_welds.py`, so admission does not depend on it.
- **The family re-certification needs its nine families from one card.**
  `recut_composition_records.py --family-recert` requires one host line across the nine
  families it re-certifies (`triton/fleet/` `bfast`, `complex`, `cylindrical_complex`,
  `folded_complex`, `folded_offdiag`, `no_pml_constitutive`, `nonlinear`, `offdiag` and
  `special_kz`), and a card's index is part of the line, so when they ran on different cards
  it exits 3 with `the families disagree about the host they ran on` and writes nothing (it
  did in the 1.0 round, whose families ran on three cards). Of the Triton table's 75 arms 22
  cite that record, and a table admits only a capability every record its arms cite has a
  live run for, so without it the Triton table does not admit yours. Run the nine units
  again together in one new chunk on one card (`chunks.py add --units
  triton/fleet/bfast,triton/fleet/complex,...`), verify (A6) and print the bind again: the
  newest run of each unit is the one selected. This recovery is read from the tools' code
  and has not yet been run on a new device: `chunks.py verify` takes each unit's run from
  the newest launch that holds, the bind then prints its `--family-recert` line over the
  fleet roots of the selected stamps only, and one chunk on one card gives one host line.

Then check what the bind did:

```bash
cd "$REPO"
git status --porcelain
python -c "from meep_gpu import fastpath; print({t: fastpath.capability_admission(t)['admitted'] for t in ('triton', 'cuda')})"
python parity/meep_gpu/rounds/1.0/legs.py flush-coverage --repo . --capability "$CC"
python tools/ci/check_hygiene.py --legs ledgers && echo "ledgers clean"
```

- `git status` lists exactly `meep_gpu/cuda_kernels/certification.json`,
  `meep_gpu/cuda_kernels/fingerprints.json` and `meep_gpu/triton_kernels/fingerprints.json`,
  modified. Do not commit them.
- Both tables admit your capability beside 8.6 (`{'triton': ('8.6', '9.0'), 'cuda':
  ('8.6', '9.0')}`; before the bind they read `('8.6',)`). If one does not,
  `fastpath.capability_admission('<table>')['by_key']` names every entry with no live run
  for it.
- `flush-coverage` ends `FLUSH_COVERAGE COMPLETE`, and the hygiene guard exits 0.

The bind of a new card re-points no coverage board: the boards keep crediting the
certified card's runs, and the bind's last line says so (`# no board re-point: cc90 is a new
device, certified by the records its bind adds; ...`).

### A8. CERT-4: the route campaigns and the licence

The route legs dispatch, so they run only after the bind, in the clone that holds the three
bound ledgers. Two lanes, side by side when two cards are free and one after the other when
one is: the hand-written CUDA route (a blocking preflight, then six legs) and the Triton
route (a blocking preflight, three legs, then the end-to-end licence of dispatch by default).

```bash
cd "$REPO"
python parity/meep_gpu/rounds/1.0/legs.py names --repo . --capability "$CC"
python -u parity/meep_gpu/rounds/1.0/round.py launch route --repo "$REPO" --stamp "$ROUND" \
    --capability "$CC" --home "$LANES/route" --lock-dir "$LOCKS" --max-locks 2 \
    --expect "$EXPECT" --python "$CONDA_PREFIX/bin/python"
```

Without `--gpu-ids` each lane takes any idle card; `--gpu-ids <card>,<card>` names the two
(the 1.0 round's route ran its two lanes side by side in 4,930 s on two RTX A6000 cards).

- `names` prints the two campaign directories, `dispatch_fused_route_<round>_cc90` and
  `dispatch_fused_route_cuda_<round>_cc90`; the licence goes to
  `dispatch_end_to_end_<round>_cc90`. The stage refuses to start unless the bound ledgers
  are the only changed files, their hygiene guard is clean, both tables admit `CC`, and each
  admits at least one arm under flush. There is no `--new-capability` here (the route stage
  refuses it): the bind made the admission true. On a batch scheduler use `round.py run
  route`, as in A5.
- **The flush legs are mixed by design**: a case passes either dispatched (every arm it
  plans has flush evidence) or refused by name and byte-identical on the array path.
- **Every route and licence leg runs once.** A leg that ran and did not release cannot run
  again under its name on this release: stop, and send the package of A9. The watchdogs
  are 12 times the walls measured on an RTX A6000, at least an hour; do not lower them.

**GO** when `$LANES/route/ROUTE_DONE` exists, `round.py status --home "$LANES/route"` reads
every step passing, and `parity/meep_gpu/results/dispatch_end_to_end_<round>_cc90/VERDICT.txt`
reads `VERDICT GO`. The maintainers cut the dispatch records from these legs; you do not
run `recut_driver_dispatch_record.py`.

### A9. What you send back

One archive, outside the clone, without compiler caches:

```bash
cd "$REPO/parity/meep_gpu/results"
DEVICE="cc$(echo "$CC" | tr -d .)"
OUT="$HOME/meep-gpu-round-${ROUND}_$DEVICE"
mkdir -p "$OUT"
tar --exclude=caches -czf "$OUT/results.tgz" *_"${ROUND}_${DEVICE}"_c* \
    dispatch_fused_route_"${ROUND}_$DEVICE" dispatch_fused_route_"${ROUND}_$DEVICE"_preflight \
    dispatch_fused_route_cuda_"${ROUND}_$DEVICE" dispatch_fused_route_cuda_"${ROUND}_$DEVICE"_preflight \
    dispatch_end_to_end_"${ROUND}_$DEVICE" $(ls -d set_aside_* 2>/dev/null)
tar --exclude=caches -czf "$OUT/lanes.tgz" -C "$LANES" .
cp "$INDEX" "$SEL" "$VERIFY" "$OUT/"
cd "$REPO"
mkdir -p "$OUT/ledgers" && cp --parents meep_gpu/cuda_kernels/certification.json \
    meep_gpu/cuda_kernels/fingerprints.json meep_gpu/triton_kernels/fingerprints.json "$OUT/ledgers/"
PYTHONPATH="$REPO" python tools/check_install.py --require-reference > "$OUT/check_install.txt" 2>&1
cp "$CONDA_PREFIX/share/meep-gpu-build/reference_build.json" \
    "$CONDA_PREFIX/share/meep-gpu-build/build_record.json" "$OUT/"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv > "$OUT/gpus.csv"
echo "release $EXPECT round $ROUND capability $CC" > "$OUT/ROUND.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs sha256sum > SHA256SUMS)
tar -czf "$OUT.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
sha256sum "$OUT.tgz"
```

The first archive holds every root your chunks wrote (their launch records
`chunk_<chunk stamp>/LAUNCH.json` among them), the route campaigns, the licence and any run
the bind set aside. `lanes.tgz` carries the round's logs (every chunk's `round.log`, every
lane's `progress.log`, `steps.jsonl` and step logs) and the bind's commands and reports.
Send the archive privately, and its sha256 in a separate message. Keep your clone as it is
until the maintainers confirm the records: they may ask for a file.

## Part B: a new Apple GPU architecture

### B1. The reference environment

On `<your-mac>`, the macOS build of the same reference: MEEP 1.33.0 in single precision
with MPI and libGDSII, compiled with `-O2 -ffp-contract=off -fno-fast-math` so that its
field updates round every operation, on top of the macOS locks, with the certified
PyTorch:

```bash
git clone https://github.com/Erised-AI-Inc/meep-gpu.git meep-gpu
cd meep-gpu
git checkout --detach v1.0.0
export REPO="$PWD" EXPECT="$(git rev-parse HEAD)"
MEEP_BUILD_JOBS=4 bash parity/meep_gpu/build_meep_133_macos.sh 2>&1 | tee "$HOME/build_meep_133_macos.log"
conda activate meep-gpu-ref
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$REPO" python tools/check_install.py --require-reference
```

The script does what A1 says of the Linux one, against the host's macOS SDK in place of
glibc, and `--refresh-corpus` updates an environment built before it kept the corpus
siblings. On Apple silicon MEEP's own subnormal switch has no effect and the Metal table
flushes on the GPU, so the build records its flush identity verdict as pending, by name;
that is expected. The check must pass (the clone is on `PYTHONPATH` for it alone, as in A1).
Keep `PYTORCH_MPS_FAST_MATH` unset (no weld ran fast-math shaders), and do not export any
`MEEP_GPU_*` variable: the drivers set what each gate needs and refuse a shell that exports
one.

### B2. Stage the inputs

```bash
cd "$REPO"
mkdir -p parity/meep_gpu/results
python parity/meep_gpu/rounds/1.0/round_inputs.py unpack --bundle "$BUNDLE"
git status --porcelain
```

The same archive as Part A, and the same GO: `1782 of 1782 verified`, and nothing from
`git status`. Stage it with this tool, not with `parity/meep_gpu/build_metal_round_inputs.py`
and the manifest of `parity/meep_gpu/rounds/metal_2026-10-04/`, a dated record of the 0.9.x
round whose spine baseline predates the 1.0 freeze.

### B3. The architecture, and your chunk index

Read the architecture Metal names this Mac's GPU, and cut the index of your architecture:

```bash
cd "$REPO"
ARCH=$(python -c "import sys; sys.path.insert(0, 'parity/meep_gpu'); import metal_environment as m; print(m.read()['architecture'])")
echo "$ARCH"
cd parity/meep_gpu
python rounds/1.0/chunks.py partition --repo ../.. --machine metal --architecture "$ARCH" \
    --new-device --out "$INDEX"
python rounds/1.0/chunks.py show --index "$INDEX"
python rounds/1.0/chunks.py check --index "$INDEX" --repo ../.. --results results
```

If `ARCH` reads `applegpu_g13s`, this Mac's GPU is already certified. Otherwise the index
holds three chunks of the 62 Metal gates (`m0`, `m1`, `m2`, stamps
`<round>_<architecture>_m<k>`), and `check` must read `CHECK OK (0 problem(s))`.

### B4. CERT-3: the Metal chunks

The Mac has one GPU, so run the chunks one after another, each with its preflight:

```bash
cd "$REPO"
for k in 0 1 2; do
  python -u parity/meep_gpu/rounds/1.0/metal_round.py all --repo "$REPO" \
      --chunk-index "$INDEX" --chunk m$k --new-architecture --expect "$EXPECT" \
      --meep-source "$CONDA_PREFIX/share/meep-gpu-build/meep-1.33.0" \
      > "$HOME/metal_round_m$k.out" 2>&1 || break
done
```

`all` runs the preflight, the chunk's gates one at a time (each with a watchdog, keeping
the Mac awake with `caffeinate`), the environment record before the first gate and after
the last, and the chunk's status; follow
`parity/meep_gpu/results/logs_metal_<chunk stamp>/round.log`. Run the loop under `nohup`, or
in a terminal that stays open, and resume by running the same command again: a gate with a
marker in `logs_metal_<chunk stamp>/done/` is skipped.

- **The preflight** checks the commit and a clean checkout, that the calling shell exports
  no shaping switch, the package from the clone, `tools/check_install.py
  --require-reference`, that every Metal environment fact can be read (it prints
  `Metal environment: <architecture> 2.10.0 <device name>`), the disk floor, no output of
  the chunk's stamp, the corpus file by file against the reference environment's record,
  and the inputs of the nine gates that re-lift corpus rows. It reads this Mac's
  architecture and holds the index and `--new-architecture` to the Metal ledger: the flag
  is refused on an architecture the ledger admits, without a chunk and on the route stage.
  It lists, by gate, any corpus row this host cannot lift (a need such as libGDSII or
  PyMieScatt that its census record declares and this environment lacks) and stops for
  none; the reference environment lacks none, so all 145 lift rows lift.
- **GO** when every chunk has run and the status of the whole index,

  ```bash
  python -u parity/meep_gpu/rounds/1.0/metal_round.py status --repo "$REPO" --chunk-index "$INDEX"
  ```

  reads `62 of 62 Metal gate artifacts released` and `cited welds released: 45 of 45` (the
  welds the Metal table cites are the ones that decide certification). A gate that did not
  release is named with its reason. If it died of its environment, verify (B5) and run the
  units no run holds again in a new chunk; if it ran and refused on your GPU, that is a
  finding about the GPU: stop, and send the package of B7.

### B5. Verify, then bind: add the records for your architecture

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/chunks.py verify --index "$INDEX" --repo ../.. --results results \
    --selection "$SEL" --json "$VERIFY"
python rounds/1.0/chunks.py bind --index "$INDEX" --selection "$SEL" | tee "$HOME/metal_bind_commands.txt"
```

**GO on `VERIFY OK: 62 of 62 units hold`**; otherwise `chunks.py add --from-verify` as in A6.
The bind prints `chunks.py verify --against "$SEL"`, then `rebind_metal_welds.py --stamp
<chunk stamp>` once per selected stamp in launch order. Run each in report mode first, read
its report, then with `--write`, keeping every report in `$HOME/metal_bind_reports.txt`; the
`# STOP` lines and the stops of A7 apply here too, and so does the exit status: a
rebind exits 1 when it skips a weld (a run that did not release, whose unit a later chunk
ran again), after writing the others. Each write adds the released weld's run
under `runs["<architecture>"]`, with its `host` line from the chunk's two environment
records, beside the `applegpu_g13s` run, which it leaves byte for byte. **Do not run `launch.write_fingerprints()`**: it
rewrites the table's own toolchain block from this Mac's PyTorch and Metal frontend, and the
bytes it covers have not moved.

```bash
cd "$REPO"
git status --porcelain
python -c "from meep_gpu import metal_runs as r; from meep_gpu.metal_kernels import launch; print(r.admission_report(launch.load_fingerprints(), r.cited_keys())['admitted'])"
python tools/ci/check_hygiene.py --legs ledgers && echo "ledgers clean"
```

`git status` lists only `meep_gpu/metal_kernels/fingerprints.json`, modified; do not commit
it. The admission lists your architecture beside `applegpu_g13s` (before the bind it reads
`('applegpu_g13s',)`); if it does not, `admission_report(...)['by_key']` names every cited
weld with no live run for it.

### B6. CERT-4: the Metal route campaign

```bash
cd "$REPO"
zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --dry-run --lanes 3 "${ROUND}_$ARCH"
nohup python -u parity/meep_gpu/rounds/1.0/metal_round.py route --repo "$REPO" \
    --stamp "${ROUND}_${ARCH}_route" --expect "$EXPECT" \
    --meep-source "$CONDA_PREFIX/share/meep-gpu-build/meep-1.33.0" \
    > "$HOME/metal_route.out" 2>&1 &
```

The first command prints the campaign's plan (its four legs, `band_witness`,
`harness_keep`, `shipped` and `harness_flush`, each leg's arguments and environment) and
launches nothing. The second runs it under held residency into
`results/dispatch_metal_route_<round>_<architecture>/`, the name the route-stamp registry
and this Mac's architecture give; `--stamp` names only the stage's logs
(`logs_metal_<stamp>`). It refuses if any file but the bound Metal ledger differs from the
commit, and takes no `--new-architecture`. Its one watchdog is 53,760 s. The campaign name
is one-shot: a leg that lands unreleased cannot run again under it on this release.

**GO** when the stage's last line reports exit 0: every leg released.

### B7. What you send back

```bash
cd "$REPO/parity/meep_gpu/results"
OUT="$HOME/meep-gpu-round-${ROUND}_$ARCH"
mkdir -p "$OUT"
tar --exclude=workdir -czf "$OUT/results.tgz" *_"${ROUND}_${ARCH}"_m* logs_metal_"${ROUND}_${ARCH}"_route \
    dispatch_metal_route_"${ROUND}_$ARCH" $(ls -d set_aside_* 2>/dev/null)
cp "$INDEX" "$SEL" "$VERIFY" "$OUT/"
cd "$REPO"
mkdir -p "$OUT/ledgers/meep_gpu/metal_kernels" && cp meep_gpu/metal_kernels/fingerprints.json "$OUT/ledgers/meep_gpu/metal_kernels/"
cp "$HOME"/metal_round_m*.out "$HOME/metal_bind_commands.txt" "$HOME/metal_bind_reports.txt" "$HOME/metal_route.out" "$OUT/"
PYTHONPATH="$REPO" python tools/check_install.py --require-reference > "$OUT/check_install.txt" 2>&1
cp "$CONDA_PREFIX/share/meep-gpu-build/reference_build.json" \
    "$CONDA_PREFIX/share/meep-gpu-build/build_record.json" "$OUT/"
echo "release $EXPECT round $ROUND architecture $ARCH" > "$OUT/ROUND.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs shasum -a 256 > SHA256SUMS)
tar -czf "$OUT.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
shasum -a 256 "$OUT.tgz"
```

Send the archive privately, and its sha256 in a separate message.

## The direct corpus check (L3) is not run on a new host

The 1.0 round also ran L3 (`rounds/1.0/l3_corpus.py`,
[The direct corpus check against MEEP](certification.md#the-direct-corpus-check-against-meep-l3)):
the corpus stepped with kernels dispatched and judged against the stored digests of the
reference MEEP's words. Each of its rows refuses unless the MEEP libraries the process
loaded have the sha256 its record names, and a MEEP built in an environment at another path
does not have them: the build links the environment's absolute path into `libmeep` and
`_meep`. On this release L3 can therefore run only on the hosts that cut its record. It is
not part of a round on a new GPU; the gates and the route campaign are.

## Timing is not part of the round

Nothing in a round times anything: the gates and the route legs are bit-identity checks,
so a neighbour on the host costs wall time only. A timing record for your GPU
([Running the timing ladder](timing.md)) is a separate, optional run after the bind, on an
otherwise idle machine, with MEEP re-timed in the same session.

## What happens to what you send

The maintainers check the archive against its sha256; run `chunks.py verify` on your
artifacts, which checks that every file each of them recorded importing is byte-equal to
the release and that each ran on your card; re-run the bind from your artifacts and compare
the ledgers it writes with yours; check that the only change is the added runs for your
GPU; cut the dispatch records from your route legs; and publish the records in a 1.0.x
release. Your GPU is then certified: a run on it records its kernels as certified, and
`MEEP_GPU_HARDWARE=certified` admits them.
