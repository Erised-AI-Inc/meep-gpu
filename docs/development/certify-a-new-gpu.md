# Certifying a new GPU

This release certifies two compute capabilities and two Apple environments: compute capability 8.6 on
one NVIDIA RTX 3090, with an RTX A6000's earlier runs kept beside it, 8.9 on one RTX 4080 SUPER, and Apple GPU architectures
`applegpu_g13s` on one Apple M1 Max and `applegpu_g16g` on one Apple M4. Other cards of that identity read as certified without
having been measured, and a supported card of any other identity already runs the compiled
kernels by default, recorded as uncertified
([Certification on an NVIDIA GPU](../guides/kernel-dispatch.md#certification-on-an-nvidia-gpu),
[Certifying another Apple GPU](certification.md#certifying-another-apple-gpu)). This page
takes you through the certification round that certifies one more NVIDIA card, on a
desktop under WSL2, a Linux workstation or a node of a shared cluster, and ends with the
same round for an Apple GPU.

A certification round writes records and nothing else. The gates run on your card in chunks
and leave artifacts; the bind then adds one run per weld for your GPU, `runs["9.0"]` or
`runs["<architecture>"]`, beside the runs already recorded, and leaves those byte for byte;
the route campaign then shows that the dispatched route is byte-identical to the array path
on your card. You send the artifacts, the bound ledgers and the logs back, the maintainers
check them, and the records are published in a 1.0.x release. No package file is edited,
and nothing is installed with `pip`: every driver imports the package from your clone.

[Running the certification harness](certification.md) explains what the gates, ledgers
and route campaigns are, and `parity/meep_gpu/rounds/1.0/RUNBOOK.md` is the round's own
runbook. You do not need to read either first. The stage names below, CERT-3 (the gates)
and CERT-4 (the route), are the round's
([The stages of a certification round](certification.md#certification-stages)).

## The round in two parts

- **Part 1, the gates (CERT-3).** Get the commit the maintainers name, build the reference
  MEEP 1.35.0, run the preflight, run a small smoke chunk, run the gate chunks, verify
  them, and send the results back.
- **Part 2, the bind and the route (CERT-4), later.** When the maintainers send the
  records commit, check it out in the same clone, bind your card's records, run the route
  campaigns and send the results again.

Part 2 waits for a later commit because the bind adds your card's runs beside the certified
card's records of this release, and those records land after the commit Part 1 runs. On a
tree whose ledgers still hold the previous records the bind adds nothing, by design: tried
with relabelled runs of another capability, 0 of 4 Triton and 0 of 3 CUDA campaign entries
were rebound, each refused as one that "would leave the records for [8.6] certifying bytes
that no longer ship"; on ledgers re-bound for 8.6 the same runs added 4 of 4 and 3 of 3.
Your Part 1 artifacts stay valid on the records commit: a gate's evidence is the import
record of its own files, and the records commit changes no file a gate imports.

## Before you start

### What you need

| | |
|---|---|
| The GPU | one or more NVIDIA cards of one compute capability, whole cards (not MIG slices), on Linux x86_64 (natively, or WSL2 on Windows) |
| On the host | `git`, `bash`, `curl`, `tar`, glibc 2.28 or later, a conda installation (step 2 installs Miniforge if you have none), network access to GitHub, conda-forge and PyPI for the build |
| Disk | about 60 GiB free under the clone before the build: the reference environment is 9.1 GB and its package cache 4.8 GB (measured on the certification round's Linux host), and a trial round used 15 GB up to the end of the smoke chunk; the rest is results, lane logs and compile caches. After the build the preflight asks for 40 GiB free |
| Memory | 24 GiB for one card (one unit took at most 9.9 GiB beyond the idle reading in a trial) and 12 GiB more for each further card running at the same time (an estimate) |
| The driver | an NVIDIA driver. Part 1 runs on any driver, and your card is certified on each driver Part 1's gates run on (one, unless your chunks ran on several): keep it for Part 2, whose route stage runs only beside a driver the bind listed. See [The CUDA driver the route stage needs](#the-cuda-driver-the-route-stage-needs) |
| From the maintainers | the repository, branch and 40-character commit to run; the inputs archive (below); later, the records commit for Part 2 |

How long each stage takes is given where it is run, measured where it was measured and
marked as an estimate where it was not.

### The inputs archive

Many gates read measured records that the repository does not carry: census rows, coverage
boards, seam records, the Metal spine baseline and earlier artifacts. A gate whose input is
absent refuses, so a round on a bare clone releases nothing. The maintainers send these
records as one archive:

| File | Size | sha256 |
|---|---|---|
| `round_inputs_1_0.tgz` | 66,515,954 bytes, 3,386 files | `03d4fd98d2d1cdeb36b38fbeb1b3500842b1003201132bbf616c1619ee0006d1` |

The release names the same digest in `parity/meep_gpu/rounds/1.0/INPUT_BUNDLE.sha256`, and
every file of the archive, with its sha256, in `INPUTS_SHA256SUMS` beside it;
`round_inputs.py` checks both before it writes anything (step 5). Its records name no
machine, account or local path: a path is written under a label (`<home>`, `<tmp>` or
`<staging-root>`). The archive is not in the repository; do not commit any part of it.

### The rules of a round

- **One commit.** Everything in Part 1 runs on a clean clone of the commit the maintainers
  name, and Part 2 on the records commit. The drivers refuse a checkout whose HEAD is
  another commit, or with any changed or untracked path but the ledgers the bind writes.
- **Nothing is committed.** The only tracked files that change are the ledgers the bind
  writes: three on NVIDIA, one on Apple. Never edit a ledger by hand, and never push.
- **Results go under `parity/meep_gpu/results/`**, which git ignores. Every results
  directory name is one-shot: never delete, rename or edit one, and never re-run into one
  by hand. The only move a round makes is a set-aside the bind prints, which moves a
  directory under `results/set_aside_<stamp>/` and deletes nothing.
- **The round stamp is the release's.** The chunk tooling reads it from
  `meep_gpu/route_stamps.py`, and every chunk's stamp adds your device to it
  (`2026-10-18_101_cc90_c0`), so no root of your round can equal or extend a root of the
  certified GPU's. You do not choose a stamp, and you do not edit the registry. In 1.0.1 the
  NVIDIA gate chunks run under `2026-10-18_101`. The commit that records the round's results
  re-stamps the NVIDIA route campaigns `2026-10-28_101`, and from that commit on the chunk
  tooling reads `2026-10-28_101` and refuses to launch a chunk of `2026-10-18_101`: run your
  gate chunks on the commit the maintainers name for them, and your route on the records
  commit. A Mac's table was re-routed alone, under `2026-10-26_101`.
- **The route campaign names are one-shot too**, one per GPU. Run the route stage only
  when the steps before it read GO.
- **Nothing is timed.** The gates and the route legs check bits, so other work on the
  machine costs wall time only (see [Timing is not part of the round](#timing-is-not-part-of-the-round)).

### Placeholders

Each command below is written for a shell in which these are set. Set them once, in a file
every shell and batch job sources (step 1):

| Variable | What it is |
|---|---|
| `REPO` | your clone, on a Linux file system (`$HOME/meep-gpu`) |
| `EXPECT` | the 40-character commit the maintainers name |
| `CAP` | your cards' compute capability, `major.minor` (`9.0`) |
| `ROUND_DIR` | a directory outside the clone for the round's own files (`$HOME/meep-gpu-round`) |
| `BUNDLE` | the inputs archive, `$ROUND_DIR/round_inputs_1_0.tgz` |
| `INDEX` | your chunk index, a file outside the clone (`$ROUND_DIR/INDEX_cc90.json`) |
| `SEL`, `VERIFY` | the selection and the report `chunks.py verify` writes, files outside the clone |
| `LANES` | the lanes' home: their logs and markers (`$ROUND_DIR/lanes`) |
| `LOCKS` | the directory the lanes take their card locks in: one per machine, and on a cluster one per job ([An HPC node](#an-hpc-node)) |
| `GPU_IDS` | the cards to run on, as `nvidia-smi` indexes them (`0`, or `0,1,2,3`); the preflight prints them |

The capability is `CAP`, never `CC`: `CC` names the C compiler, which the reference
environment sets and every lane hands to Triton's launcher.

### Which platform notes apply

The steps below are the same on every Linux host. Each platform has notes of its own,
which the steps point to:

- [A desktop under WSL2](#a-desktop-under-wsl2), including a card that drives the display;
- [A Linux workstation](#a-linux-workstation);
- [An HPC node](#an-hpc-node): a batch scheduler, several cards, a shared machine.

## Part 1: the gates

### 1. The commit

```bash
mkdir -p "$HOME/meep-gpu-round"
cat > "$HOME/meep-gpu-round/env.sh" <<'EOF'
export REPO="$HOME/meep-gpu"
export EXPECT=<the 40-character commit the maintainers name>
export CAP=9.0
export ROUND_DIR="$HOME/meep-gpu-round"
export BUNDLE="$ROUND_DIR/round_inputs_1_0.tgz"
export INDEX="$ROUND_DIR/INDEX_cc90.json"
export SEL="$ROUND_DIR/selection_cc90.json"
export VERIFY="$ROUND_DIR/verify_cc90.json"
export LANES="$ROUND_DIR/lanes"
export LOCKS="$ROUND_DIR/locks"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
EOF
source "$HOME/meep-gpu-round/env.sh"
mkdir -p "$LANES" "$LOCKS"
git clone --branch <the branch> --single-branch <the repository URL> "$REPO"
cd "$REPO"
git -c advice.detachedHead=false checkout --detach "$EXPECT"
git remote set-url --push origin PUSHING-IS-DISABLED
git rev-parse HEAD 'HEAD^{tree}'
git status --porcelain
```

Put the inputs archive at `$BUNDLE`. In every new shell from now on, start with
`source "$HOME/meep-gpu-round/env.sh"`.

**Expect** the commit (`EXPECT`), the tree the maintainers name with it, and nothing from
`git status`. The `set-url --push` line makes any push from this clone fail. Keep the clone
on a Linux file system: under WSL2, under your Linux home directory, never under `/mnt/c`.

Do not export `PYTHONPATH`, `TRITON_LIBCUDA_PATH` or any variable starting `MEEP_GPU_`,
`CUPY_`, `TRITON_` or `TRIDENT_FDTD_`: the round sets what each gate needs and refuses a
shell that sets them. (`conda activate`, step 2, sets `CC` to the environment's own
compiler; that is expected.)

### 2. The reference environment

Every certification is measured in one reference environment per platform: MEEP 1.35.0 from
its release tarball (pinned by SHA-256), single precision, MPI and OpenMP, compiled with
`-O2 -ffp-contract=off -fno-fast-math -march=x86-64` (no fused multiply-add), on top of
exactly the packages the Linux locks pin (CuPy 13.5.1 with NVRTC 11.8, Triton 3.1.0, NumPy
2.2.6 among them), with the MEEP corpus the lift legs walk kept inside it
([The MEEP a round is measured against](certification.md#the-meep-a-round-is-measured-against)).
The repository's own script builds it, end to end, into a conda installation. If you have
none, install Miniforge first; if you have one (Miniforge, Miniconda or Anaconda), set
`CONDA_ROOT` to its root, the directory holding `bin/conda`, for the build, and source its
`etc/profile.d/conda.sh` wherever this page sources `$HOME/miniforge3`'s:

```bash
cd "$HOME"
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
cd "$REPO"
MEEP_BUILD_JOBS=8 bash parity/meep_gpu/build_meep_reference_linux.sh 2>&1 | tee "$ROUND_DIR/build.log"
echo "build exit ${PIPESTATUS[0]}"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate meep-gpu-ref
```

The script clears its own environment, creates the conda environment `meep-gpu-ref` under
`$CONDA_ROOT` (default `$HOME/miniforge3`; `MEEP_ENV_NAME` or `MEEP_ENV_PREFIX` names
another), builds in `$MEEP_BUILD_ROOT` (default under `$TMPDIR`), and refuses to finish
unless what it built is the reference: the lock's packages artifact for artifact, MEEP
1.35.0 in single precision with MPI, no fused multiply-add in the field updates, MEEP's own
words on four cases equal to the recorded digests, and the identity verdicts in keep and in
flush. It also links the driver's `libcuda.so.1` into the environment for Triton's launcher,
**when the build host has an NVIDIA driver** ([An HPC node](#an-hpc-node) says what to do
when it does not). `MEEP_BUILD_JOBS` sets the compile jobs (default: every processor but
two) and `MEEP_GPU_BUILD_THREADS` the OpenMP threads of every MEEP and NumPy process the
build runs (default the smaller of 8 and the processor count); on a shared machine set both
to your share. The build took 381 s on the certification round's Linux host and 448 s on a
desktop PC; the downloads are a few GB.

**Expect** `build exit 0`, and in the log, under step 8, one `ok` line each for `MEEP
version 1.35.0`, `single precision True`, `MPI build True`, `subnormals flushed after import
meep True`, `CuPy 13.5.1`, `NVRTC (cudatoolkit 11.8) 11.8`, `Triton 3.1.0`, `NumPy 2.2.6`,
then `reference environment OK` and `Triton launcher compiles and links with
x86_64-conda-linux-gnu-cc against .../lib/meep-gpu-libcuda`; under step 10 `MEEP identity
smoke: ... REPRODUCED.`; both identity verdicts passing; and last `DONE`.

**If not:** an environment named `meep-gpu-ref` that already exists is never written into:
run again with `MEEP_ENV_NAME=<another name>` and activate that name everywhere below.
`subnormals flushed after import meep False`, `NOT REPRODUCED`, a failing identity verdict
or anything about MPI: stop and send `build.log`.

Then, in the environment:

```bash
cd "$REPO"
PYTHONPATH="$REPO" python tools/check_install.py --require-reference 2>&1 | tee "$ROUND_DIR/check_install.txt"
python -c "import meep; from meep_gpu import backends; print('MEEP flushes subnormals:', backends.subnormals_flushed())"
python -c "import cupy; r = cupy.cuda.runtime; print('driver', r.driverGetVersion(), 'runtime', r.runtimeGetVersion())"
```

The check must pass (its MEEP line ends `the reference's build`; a NOTE that your NVIDIA GPU
is supported but not certified is what this round changes), the second line must print
`True`, and the third prints the CUDA version your driver supplies (`driver 12030` is CUDA
12.3; any number runs Part 1, and your card is certified on the driver Part 1 runs on). The check runs
with the clone on `PYTHONPATH` because the
reference environment holds no installed copy of the package; the round's drivers put the
clone there themselves.

### 3. Preflight

```bash
cd "$REPO"
python parity/meep_gpu/rounds/1.0/preflight.py --capability "$CAP" --expect "$EXPECT" \
    2>&1 | tee "$ROUND_DIR/preflight.txt"
```

Run it where the gates will run (on a cluster, inside the job). It changes nothing. It checks
the platform (Linux x86_64, glibc, WSL2 or not), the clone's file system and free disk (40 GiB
once the environment is built), the memory this process may use, the tools, a shell that
exports none of the round's variables, the commit and a clean checkout, the reference
environment and its link to the driver's `libcuda.so.1` (a WARN where the environment records
none but the driver's own directory holds the `libcuda.so` Triton links against), and each
card: its compute capability, no MIG slices, the CUDA version the
driver supplies, and the round's own card rule (`display_card.py`), read twice a minute apart
on every card at once, as a lane reads it.

**Expect** the last line `PREFLIGHT OK (0 FAIL)` (a little over a minute), a PASS line per
card (`TAKEN: card N is taken under the idle rule`, or `... under the display rule` for a card
that drives a display), and an INFO line naming the cards to run on (`the cards to run on:
0,1,2,3 (...)`): set `GPU_IDS` to them (`--gpu-ids` chooses others; `--cards-only` prints them
alone). For a new capability the driver line passes on any driver and says the bind records
it; a WARN on it means the ledgers certify your capability on other drivers only: the gates
still run, and the bind adds this driver if every other axis of the stack your lanes read
matches the record (see
[The CUDA driver the route stage needs](#the-cuda-driver-the-route-stage-needs)).
Each FAIL line says what to change; a card refused here (a FAIL line `REFUSED: card N is not
taken`, with the `LANE` and `FLEET` lines above it saying why) is refused by the round too.

### 4. Your chunk index

The gates run in chunks, each under a stamp of its own. First ask whether your capability
is new to the ledgers:

```bash
cd "$REPO"
python -c "from meep_gpu import fastpath; print({t: fastpath.capability_admission(t)['admitted'] for t in ('triton', 'cuda')})"
```

On this release it prints `{'triton': ('8.6', '8.9'), 'cuda': ('8.6', '8.9')}`: both NVIDIA tables admit 8.6
and 8.9, so every other capability, an H200's 9.0 among them, is **new**, and is cut with
`--new-device`. Cut the index of your card once, outside the clone, then add the smoke
chunk:

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/chunks.py partition --repo ../.. --machine nvidia --capability "$CAP" \
    --new-device --chunks 8 --out "$INDEX"
python rounds/1.0/chunks.py add --index "$INDEX" --repo ../.. --why "smoke on the new card" \
    --units cuda/fused_pml_bit_identity_hand_2026-08-22_rename,cuda/cuda_fused_magnetic_pair_2026-08-27,cuda/cuda_cylindrical_real_fused_hd_pair_2026-09-07,triton/fleet/fused_electric,triton/fleet/cylindrical_complex,triton/hd/cylindrical_real_fused_hd_pair
python rounds/1.0/chunks.py show --index "$INDEX"
python rounds/1.0/chunks.py check --index "$INDEX" --repo ../.. --results results
```

- **`--new-device` says your card is new**: the index records it (`device_mode` `new`),
  and every later step holds the index to the ledgers in both directions, so a new card is
  never admitted silently. Without it the index is written as a certified device's, and
  the gate stage refuses it. The flag goes on the index only: `run_chunks.py` (step 6) reads
  it there and runs the chunks with `--new-capability`, which `legs.py check` and `round.py`
  require for a capability the ledgers hold no record for (and refuse for one they do).
- **The chunks.** 123 units in all (a unit is the smallest piece of evidence that binds on
  its own: a CUDA campaign subdirectory, or one Triton gate under both policies). The default
  is four chunks of 23,861 to 23,895 s each at RTX A6000 prices; `--chunks 8` cuts eight of
  11,902 to 11,958 s, which spread over more cards and fit shorter batch jobs. `show` prints
  `nvidia cc90 (new device), round 2026-10-18_101, next_k 9`, the eight chunks `c0` to `c7`
  and the smoke chunk `c8` (6 units, 1,458 s); the smoke chunk's id is the one `show` prints
  last.
- **`check` must read `CHECK OK (0 problem(s))`**: no two chunks, and no chunk and a
  directory already under `results/`, share or extend a root.
- Keep the index for the whole round and never edit it by hand. The tools that change it
  (`add`) allocate a new chunk id each time and never reuse one. It is a
  `meep_gpu.cert_chunks/2` document; an index of the first chunk tooling (`cert_chunks/1`)
  is refused by name.

### 5. Stage the inputs

```bash
cd "$REPO"
sha256sum "$BUNDLE"
mkdir -p parity/meep_gpu/results
python parity/meep_gpu/rounds/1.0/round_inputs.py unpack --bundle "$BUNDLE"
git status --porcelain
```

**Expect** the sha256 above; `unpacked 3386 files, 0 already present and identical; 3386 of
3386 verified`; and nothing from `git status` (`results/` is ignored). The tool refuses any
other archive, by design.

### 6. The smoke chunk, then the gate chunks

`run_chunks.py` runs chunks of the index one after another, each in the foreground through
`round.py run gates`. For each chunk it skips one already done (`GATES_DONE` and every step
passing), resumes one that has a launch record (`--resume`: every step that passed is
skipped), and otherwise requires `legs.py check` to read `CHECK OK` first. `--spread` spreads
the chunks over the cards named: each card runs one chunk at a time and the next chunk goes to
the next free card; `--max-locks` (default: the number of cards) caps the cards the lanes
under `LOCKS` hold at once. It writes one line per event; each lane writes its own logs under
`$LANES/<chunk stamp>`.

Start the card log, then run the smoke chunk alone, on one card:

```bash
cd "$REPO"
python parity/meep_gpu/rounds/1.0/gpu_log.py start --dir "$ROUND_DIR" --gpu-ids "$GPU_IDS"
python -u parity/meep_gpu/rounds/1.0/run_chunks.py --index "$INDEX" --lanes "$LANES" \
    --lock-dir "$LOCKS" --expect "$EXPECT" --gpu-ids "${GPU_IDS%%,*}" c8 \
    2>&1 | tee -a "$LANES/run_chunks.log"
```

**Expect** `CHECK OK c8`, `START c8`, then `DONE c8 ...: every step passing`. The six units'
legs took 631 s on an RTX 3090 and the lane adds 2 to 3 minutes of checks at the start. The
lane's `progress.log` starts with `took gpu_lock_<n>` a minute after the start, then
`TOOLCHAIN card ... | cc <CAP> | cupy 13.5.1 | triton 3.1.0 | meep 1.35.0 single MPI | ...`
and `TOOLCHAIN OK`. Do not start the rest until the smoke chunk reads every step passing.

Then the gate chunks, on every card you have:

```bash
python -u parity/meep_gpu/rounds/1.0/run_chunks.py --index "$INDEX" --lanes "$LANES" \
    --lock-dir "$LOCKS" --expect "$EXPECT" --gpu-ids "$GPU_IDS" --spread --continue-on-fail \
    c0 c1 c2 c3 c4 c5 c6 c7 2>&1 | tee -a "$LANES/run_chunks.log"
```

**Expect** a `DONE` line per chunk and last `ALL GIVEN CHUNKS DONE`. With
`--continue-on-fail` a chunk that does not pass is logged as `NOT PASSED` and the cards go
on with the next ones (each unit's evidence is its own); the last line then names them
(`FINISHED WITH CHUNKS NOT PASSED: ...`): step 7. On a desktop, start it detached so a closed
terminal does not stop it (`setsid nohup ... >> "$LANES/run_chunks.log" 2>&1 < /dev/null &`;
under WSL2 keep Linux up as [A desktop under WSL2](#a-desktop-under-wsl2) says); on a
cluster it is the body of a batch job ([An HPC node](#an-hpc-node)).

How long it takes. A chunk of eight is priced at about 11,950 s of RTX A6000 time. The gates
are mostly host work (MEEP references, the NumPy array path, compiling), so the processor
matters about as much as the card: on the 1.0 certification round's host (a shared Xeon
server) seven chunks of eight ran in 6,554 to 8,376 s each (0.55 to 0.70 of the price), and on a
desktop with an RTX 3090 and an i9-12900K 21 of the round's legs ran at 0.33 of their price.
`gpu_log.py peak` shows each card's largest device memory so far beside its lowest reading;
on an RTX A6000 the largest unit (`triton/hd/fused_hd_pair`) reached about 12.6 GB and every
other unit stayed under 3.4 GB. A card with less memory that runs a unit out of memory is a
finding: name the unit when you send the results, and do not retry it.

#### A card that also drives your display

A card that drives a monitor always holds memory for the desktop that no compute process
accounts for, so it never reads idle by the idle rule (no compute process listed, 0 %
utilisation, at most 200 MiB in use). The round takes it anyway, by default, when it reads
idle the way a display card does:

- **It is a display card**: `nvidia-smi` says a display is attached or active on it
  (`display_attached` Yes, or `display_mode` Enabled on a driver that predates that field,
  or `display_active` Enabled), or the machine is a WSL2 host, where Windows owns the display.
- **No compute process of any user is listed on it**, on both readings.
- **Its utilisation is at most 10 %** on both readings: a desktop at rest reads 0 %, with an
  occasional 6 % (measured on an RTX 3090 driving a Windows desktop, 706 of 709 readings at
  0 % over 15 minutes).
- **Its memory stays within 128 MiB of the first reading** (the display resident): that
  desktop's memory moved by 5 MiB at rest, and one bare CUDA context takes 258 MiB, so a
  program that starts computing on the card after the first reading is seen.

The lane names the card, its display state and the resident in its `progress.log` (`DISPLAY
CARD`), in its lock and in every row of its `steps.jsonl`. The one rule places every gate: the
lane asks it again before each step it starts, and the Triton fleet driver, which keeps no
rule of its own, asks it before each fleet gate, holding every gate to the same resident; each
gate's row in the fleet's `campaign.json` records the decision, so results taken under this
rule say that a display was attached. Keep the desktop quiet while the round runs: a program
that takes memory on the card after the lane took it (a browser with hardware acceleration, a
video, a game) reads as another resident, and the next step waits up to five minutes for it to
leave and is then refused (a fleet gate by the driver, any other step by the lane), and the
chunk has to be resumed (step 7). To see what the round would decide for a card:

```bash
python parity/meep_gpu/rounds/1.0/display_card.py --gpu <card>
```

A card that drives no display keeps the idle rule: memory that no listed process accounts
for is another user's (a container hides other accounts' processes), and the card waits.

#### Sharing a card with another user's idle process

On a shared machine a card can hold another user's process that keeps memory and computes
nothing. A lane never takes a card that holds any compute process, unless you name that
process. `run_chunks.py` does not pass that option, so run the chunk with `round.py` itself,
after `legs.py check` reads `CHECK OK` for it (the option belongs to the gate stage only; the
route stage refuses it):

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
cd "$REPO"
python parity/meep_gpu/rounds/1.0/legs.py check --repo . --capability "$CAP" --new-capability \
    --git-expect "$EXPECT" --chunk-index "$INDEX" --chunk <chunk> --python "$CONDA_PREFIX/bin/python"
python -u parity/meep_gpu/rounds/1.0/round.py run gates --repo "$REPO" --chunk-index "$INDEX" \
    --chunk <chunk> --gpu-ids <card> --capability "$CAP" --new-capability --home "$LANES" \
    --lock-dir "$LOCKS" --max-locks 1 --expect "$EXPECT" --python "$CONDA_PREFIX/bin/python" \
    --share-idle-foreign <uuid>:<pid>[,<pid>] --share-min-free-mib <MiB>
```

(drop `--new-capability` from both for a capability the ledgers already admit).

The card is then admitted when, on both idle readings, every compute process on it is one
you named, those processes and the card read 0 % utilisation, and the card's free memory is
at least `--share-min-free-mib` (your units' own peak plus what you keep free for the card's
owner; the certification round set 12,288 MiB). **A chunk that holds Triton fleet units needs
an unshared card**: the fleet driver, which runs every `triton/fleet/...` unit and
`triton/composition`, refuses a card that carries any compute process. Split such a chunk
before it launches (`chunks.py add --units <unit>,... --why "<why>"`, once for its fleet units
and once for the rest), and never launch the chunk you split. **Yield the card the moment its
owner needs it**: send `SIGTERM` to the card's lane (the pid in `$LOCKS/gpu_lock_<n>/pid`); it
stops its gate's process group and releases the lock. The interrupted unit runs again in a new
chunk (step 7). The lane's log names the card by its card id and counts the processes you
named, with their memory; the card's UUID and the process ids stay in the lock directory
(`$LOCKS/gpu_lock_<n>/`), which nothing packs.

### 7. When a chunk stops, and verify

For each chunk `run_chunks.py` names as not passed,

```bash
python parity/meep_gpu/rounds/1.0/round.py status --home "$LANES/<chunk stamp>"
```

prints each lane's `N of D steps passing` and, for each step that did not pass, its state and
reasons; its log is named in the step's row of `$LANES/<chunk stamp>/*/steps.jsonl`.

- **The environment stopped it** (a reboot, a lost session, a batch job's time limit, a full
  disk, a card that was busy): fix the cause and run the same `run_chunks.py` command again;
  an interrupted chunk resumes and a finished one is skipped. If a lock remains in
  `$LOCKS/gpu_lock_<n>`, remove that directory only when the process its `pid` file names is
  gone.
- **The resume refuses a directory by name** (a CUDA leg directory that holds an unreleased
  result): move nothing, and run that chunk's units again in a new chunk (below).
- **A gate ran and refused on your card** (a verdict that is not released, for example a word
  that differs from the reference): that is a finding about the card. Finish the other chunks,
  then send Part 1 with a note naming the step.

When every chunk has run, judge every unit against the commit:

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/chunks.py verify --index "$INDEX" --repo ../.. --results results \
    --selection "$SEL" --json "$VERIFY"
```

For each unit it reads every launch whose record names it and takes the newest run whose
import record holds on the tree, whose verdicts pass and whose card is your index's (read
from the artifacts, never assumed). **GO on `VERIFY OK: 123 of 123 units hold`.** Otherwise:

```bash
python rounds/1.0/chunks.py add --index "$INDEX" --repo ../.. --from-verify "$VERIFY" \
    --why "units that did not hold after the first pass"
python rounds/1.0/chunks.py show --index "$INDEX" | tail -1
```

allocates a new chunk holding exactly the units no run holds; run it as in step 6 (its id is
the one `show` prints last) and verify again. Every earlier run stays as evidence. A unit
that refused on your card twice is a finding: leave it, and say so when you send Part 1.

### 8. Send Part 1

```bash
cd "$REPO"
python parity/meep_gpu/rounds/1.0/gpu_log.py stop --dir "$ROUND_DIR"
python parity/meep_gpu/rounds/1.0/pack_results.py part1 --index "$INDEX" --lanes "$LANES" \
    --round-dir "$ROUND_DIR" --selection "$SEL" --verify "$VERIFY"
```

It packs, without compiler caches, every results root your chunks wrote (their launch
records `chunk_<stamp>/LAUNCH.json` among them) and any run set aside; the lanes' home
(every chunk's `round.log`, every lane's `progress.log`, `steps.jsonl` and step logs); the
index, the selection and the verify report; the `*.txt`, `*.log` and `*.out` files and
`gpu_log.csv` of `ROUND_DIR` (the build log, the preflight, the check, a batch job's output); the clone's
HEAD and `git status`; the reference environment's build records and `check_install.py
--require-reference`; the cards (`nvidia-smi` queried for their index, model, compute
capability, driver and memory, and its version); and a `SHA256SUMS` of all of it.

#### What the results say about your machine

Nothing in them names your machine, your account or where your files are. Every record and
log the round writes (the gates' artifacts, the fleet's `campaign.json` and `gates.jsonl`,
the lanes' `steps.jsonl`, `plan.json` and logs, `round.log`, the preflight's report, the
dispatch logs) spells:

- **the machine** as an anonymous machine id, `m-` and 16 hex digits: a keyed hash
  (HMAC-SHA256) of the machine's own random id (`/etc/machine-id` on Linux and WSL2), the
  construction systemd recommends for an application's machine id. It is the same on every run
  of one machine, so the maintainers can check that the families of a re-certification ran on
  one machine, and it gives back neither the machine's id nor its host name;
- **each card** as a card id, `c-` and 16 hex digits, a salted hash of its UUID: the same card
  always has the same id;
- **paths** under a named root: `<staging-root>` (the directory your clone lies in), `<home>`,
  `<tmp>`, and `<account>` for your account's name as a directory elsewhere or at the start of
  a directory name (`<account>_runs`);
- **another account's processes** as a count and their memory, never their process ids; a
  process of your own account that a timing's quiet window names, as its program and the
  harness script it runs, never its arguments.

What stays is what a certification is about: the GPU model, compute capability, driver and
CUDA versions, the CPU model and the OS release. Lock and pid files stay on your machine:
`pack_results.py` leaves out every `*.lock` directory and every pid file, as it leaves out the
compiler caches, and every symbolic link (a link's target is a path on your machine); the
archive's members carry no owner name. The spellings are those of
`parity/meep_gpu/neutral_ids.py`.

`pack_results.py` also checks this before it packs anything: it reads every file the part
would carry (allow a few minutes for a large part; it prints its progress) and refuses the
part if one names your home directory, another home directory (one under the Linux,
macOS or Windows home root) or your account as a directory. It then prints
`STOP:`, lists each such file with the keys (or lines) that name it, and writes nothing.
Do not edit a listed file to get past it (the verify report re-hashes every artifact): send
the maintainers the list, which names files and keys and never your paths. A temporary
directory (`/tmp/<random>`) is not refused; it names no account.

**Expect** `packed $HOME/meep-gpu-cc90-part1.tgz (...)` and a sha256 line. Send the archive
privately, and the sha256 line in a separate message. Keep the clone, the environment and
`ROUND_DIR` exactly as they are: Part 2 needs them, and the maintainers may ask for a file.

## Part 2: the bind and the route

Start Part 2 only when the maintainers send the records commit. The bind (step 10) runs on
any driver; the route stage (step 11) only beside a stack the bind recorded for your
capability, so on a driver Part 1 ran on.

### 9. The records commit, and verify again

```bash
source "$HOME/meep-gpu-round/env.sh"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate meep-gpu-ref
R=<the records commit, 40 hexadecimal characters>
cd "$REPO"
git fetch origin <the branch>
git -c advice.detachedHead=false checkout --detach "$R"
git status --porcelain
sed -i "s/^export EXPECT=.*/export EXPECT=$R/" "$ROUND_DIR/env.sh"
sed -i "s#^export SEL=.*#export SEL=\"\$ROUND_DIR/selection_cc90_part2.json\"#; s#^export VERIFY=.*#export VERIFY=\"\$ROUND_DIR/verify_cc90_part2.json\"#" "$ROUND_DIR/env.sh"
source "$ROUND_DIR/env.sh"
cd parity/meep_gpu
python rounds/1.0/chunks.py verify --index "$INDEX" --repo ../.. --results results \
    --selection "$SEL" --json "$VERIFY"
```

**Expect** nothing from `git status` and `VERIFY OK: 123 of 123 units hold` (the records
commit changes no file a gate imports). Units that no longer hold run again as in step 7, now
on the records commit; those chunks travel with Part 2 (step 12).

### 10. The bind: add the records for your capability

On the same host, in the same environment, with no GPU work. Print the bind's commands:

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/legs.py bind --chunks "$INDEX" --selection "$SEL" | tee "$ROUND_DIR/bind_commands.txt"
```

It prints the commands that bind exactly the selection and never runs them. The first is
`chunks.py verify --against "$SEL"`, which judges again, writes nothing and exits 1 when
the choice or the bytes of a selected run moved since the selection was written: then
write the selection again (step 9) and print the bind again. The second records the
certified NVIDIA stack of your capability (`record_nvidia_stack.py`, from the stack each gate
lane recorded on its card in `chunk_<stamp>/stack/`); it comes before every line that records
or rebinds, so a stack it refuses stops the bind before anything is bound to those runs. The
others are the CUDA records and rebinds (`record_cuda_regate.py`, `rebind_cuda_welds.py`, once
per selected stamp in launch order), the ADE re-cut, the Triton rebinds
(`rebind_triton_welds.py --only`, once per selected stamp), the bit identity, the composition
records and the family re-certification, `neutralize_ledger_labels.py`, and last the two filing
lines, `ledger_scope.py archive`, which write when run. After the verify, run all but those **in
order, one at a time**: each first as printed (report mode, which writes nothing), read its report,
then the same command with `--write` added, then the next, since a later tool reads what an
earlier one wrote. Keep every report, and never run the lines under `set -e` or a `--write` twice:

```bash
<the command as printed> 2>&1 | tee -a "$ROUND_DIR/bind_reports.txt"
<the command as printed> --write 2>&1 | tee -a "$ROUND_DIR/bind_reports.txt"
```

- **`# STOP` lines.** A run a tool would bind against the selection (a later run of a unit,
  an unreleased subdirectory left in a CUDA campaign by a yielded lane) is named on a
  `# STOP` line with the command that sets it aside, and while one stands the bind prints no
  command that writes a ledger, ends with `# BIND WITHHELD` and exits 1. Run each set-aside
  command as printed (it moves the run under `results/set_aside_<stamp>/` and deletes
  nothing), write the selection again (step 9) and print the bind again. They are expected
  when one chunk holds runs of several units and only some are selected from it (the smoke
  chunk's, when its units ran again later).
- **Stop**, and write nothing more, if a report asks for `--supersede`, names a file whose
  digest differs from the checkout, or says an artifact's import record disagrees with the
  tree: the artifacts then describe other bytes than the commit's. Never pass `--supersede`:
  it is for a round that moves bound bytes, and it would drop the certified card's records.
- **The stack record.** `record_nvidia_stack.py` adds your capability's certified stack (CuPy,
  NVRTC, the CUDA runtime and driver, the host, NumPy, the MEEP build) to
  `meep_gpu/triton_kernels/fingerprints.json`, as the lanes read it. A GPU family can be
  certified on several CUDA drivers: lanes that ran on different drivers (a driver updated
  between chunks, two machines with two drivers) are accepted when they read the same value
  on every other axis, and the record lists each driver with the chunk stamps that ran on it
  (the family re-certification still takes its nine families from one machine: below).
  It refuses, writing nothing, when the lanes disagree on any other axis: then run the
  disagreeing chunks' units again on one stack (`chunks.py add`) and print the bind again. For
  a capability the ledgers already certify it keeps the recorded values and adds a driver
  whose lanes read them on every other axis; anything else (another value on another axis, or
  a recorded driver none of your lanes ran on) it refuses. Never pass `--restate`: it
  re-states the stack of a capability the ledgers already certify (the record becomes exactly
  the stack your lanes read, dropping any driver they did not run on), which is the
  maintainers' decision.
- **Lines that are no reason to stop**: a `# REFUSED rebind_triton_welds.py --only` comment
  for a weld key the ledger has never held (the tool leaves such a key out of `--only`, and
  the unit's run still serves the family re-certification); a rebind that exits 1 after
  writing the others because it skipped one entry (judge each line by its summary);
  `record_cuda_ade_recut.py` refusing and writing nothing (the ADE weld itself is rebound by
  `rebind_cuda_welds.py`).
- **The family re-certification needs its nine families from one machine.**
  `recut_composition_records.py --family-recert` takes the nine Triton fleet families it
  re-certifies (`triton/fleet/` `bfast`, `complex`, `cylindrical_complex`, `folded_complex`,
  `folded_offdiag`, `no_pml_constitutive`, `nonlinear`, `offdiag` and `special_kz`) from the
  selected runs, accepts them from different cards of one machine, and refuses, writing
  nothing, when they ran on more than one machine ("the families stamp 2 machines; one
  record names one GPU host"). Triton arms cite that record, so without it the Triton table
  does not admit your capability. Run the chunks on one machine ([An HPC
  node](#an-hpc-node) says how on a cluster); if the nine did run on several, run them again
  together in one new chunk on one machine (`chunks.py add --units
  triton/fleet/bfast,triton/fleet/complex,...`), verify and print the bind again: the newest
  run of each unit is the one selected.

Then check what the bind did:

```bash
cd "$REPO"
git status --porcelain
python -c "from meep_gpu import fastpath; print({t: fastpath.capability_admission(t)['admitted'] for t in ('triton', 'cuda')})"
python -c "from meep_gpu import fastpath; print(fastpath.certified_nvidia_stack('$CAP'))"
python parity/meep_gpu/rounds/1.0/legs.py flush-coverage --repo . --capability "$CAP"
python tools/ci/check_hygiene.py --legs ledgers,ledger_states && echo "ledgers clean"
```

**Expect** `git status` listing exactly `meep_gpu/cuda_kernels/certification.json`,
`meep_gpu/cuda_kernels/fingerprints.json` and `meep_gpu/triton_kernels/fingerprints.json`,
modified, and nothing else (do not commit them; the route stage refuses any other change, so
the two filing lines wait until it has run); both tables admitting your capability beside 8.6
and 8.9 (`{'triton': ('8.6', '8.9', '9.0'), 'cuda': ('8.6', '8.9', '9.0')}`); your capability's certified stack,
with the CUDA driver your gates ran beside (`'cuda_driver': ('12080',)` for driver 12080; one
value per driver, in version order, if your chunks ran on several);
`FLUSH_COVERAGE COMPLETE`; and `ledgers clean`. If a table does not admit it,
`fastpath.capability_admission('<table>')['by_key']` names every entry with no live run for
it: send the bind reports, and do not start the route. The bind of a new card re-points no
coverage board; the last line before the filing says so.

A bind on bytes the ledger already binds writes only your capability's runs and stack, so it
leaves the other capabilities' route legs, L3 rows, timing rows and dispatch checks standing
(`CARRY-SLICE` and `valid`; `docs/development/certification.md`, "Carrying a route campaign or
an L3 leg to a new tree"). Those tools resolve a recorded digest from the filed states, and the
filing line is `python ledger_scope.py archive --table <table> <file>`, run from
`parity/meep_gpu`; a state may be filed later, from any copy. A bind that had to move a pinned
file refuses until `--supersede` names the other capabilities, and those then re-run. A bind
that would lose a policy your capability's run holds is refused by name
(`policy_dropped`) unless `--allow-policy-drop` says so: pass the flush campaign as well.

### 11. CERT-4: the route campaigns and the licence

The route legs dispatch, so they run only after the bind, in the clone that holds the three
bound ledgers, and only beside a stack your capability was certified on, so on a driver the
gates ran on:

```bash
cd "$REPO"
python -c "import cupy; print('driver', cupy.cuda.runtime.driverGetVersion())"
python parity/meep_gpu/rounds/1.0/legs.py names --repo . --capability "$CAP"
python parity/meep_gpu/rounds/1.0/gpu_log.py start --dir "$ROUND_DIR" --gpu-ids "$GPU_IDS"
python -u parity/meep_gpu/rounds/1.0/round.py run route --repo "$REPO" \
    --stamp "$(python -c 'from meep_gpu import route_stamps; print(route_stamps.stamp("triton"))')" \
    --capability "$CAP" --home "$LANES/route" --lock-dir "$LOCKS" --max-locks 2 \
    --gpu-ids "$GPU_IDS" --expect "$EXPECT" --python "$CONDA_PREFIX/bin/python"
```

**Expect** a driver the bind recorded for your capability (if not, stop here: next
section; the route stage itself refuses to start, naming the axis, before any leg runs);
`names` prints the two campaign
directories, `dispatch_fused_route_<stamp>_cc90` and `dispatch_fused_route_cuda_<stamp>_cc90`,
where `<stamp>` is the NVIDIA tables' stamp (`route_stamps.stamp("triton")`, `2026-10-28_101`
in 1.0.1); the licence goes to `dispatch_end_to_end_<stamp>_cc90`. Two lanes run: side by side with two
cards named and `--max-locks 2`, or one after the other with one card and `--max-locks 1`
(the second lane then waits for the first's lock and takes an idle card; on a machine with
other people's idle cards, name two of yours): the hand-written CUDA route (a blocking
preflight, then six legs) and the Triton route (a blocking preflight, three legs, then the
end-to-end licence of dispatch by default). The route is priced at 12,733 s of RTX A6000
time; the 1.0 certification round's ran its two lanes side by side in 4,930 s on two RTX A6000
cards, and 1.0.1's ran them one after the other in 6,608 s on one RTX 3090. There is no
`--new-capability` here (the route stage refuses it): the bind made the card admitted. The
stage refuses to start unless the bound ledgers are the only changed
files, their hygiene guard is clean, both tables admit `CAP`, and each admits at least one
arm under flush. `round.py run` stays in the foreground; `round.py launch` takes the same
arguments and returns at once, its stage running in its own session.

**Every route and licence leg runs once.** A leg that ran and did not release cannot run
again under its name on this release: if anything in the route goes wrong, do not relaunch
it; send Part 2 as it is. The watchdogs are 12 times the walls measured on an RTX A6000, at
least an hour; do not lower them.

**GO** when `$LANES/route/ROUTE_DONE` exists, `round.py status --home "$LANES/route"` reads
every step passing, and `parity/meep_gpu/results/dispatch_end_to_end_<stamp>_cc90/VERDICT.txt`
reads `VERDICT GO`. The maintainers cut the dispatch records from these legs; you do not run
`recut_driver_dispatch_record.py`.

### 12. Send Part 2

```bash
cd "$REPO"
python parity/meep_gpu/rounds/1.0/gpu_log.py stop --dir "$ROUND_DIR"
python parity/meep_gpu/rounds/1.0/pack_results.py part2 --index "$INDEX" --lanes "$LANES" \
    --round-dir "$ROUND_DIR" --selection "$SEL" --verify "$VERIFY"
```

It packs the route campaigns, their preflights and the licence (`dispatch_*_cc90`,
`dispatch_*_cc90_preflight`), any run set aside, the three bound ledgers, the bind's commands
and reports, and the same logs and records as Part 1; where the route stage did not run (next
section) it packs the bind alone and says so. Send it as Part 1, and keep everything as it is
until the maintainers confirm the records.

If you ran chunks again during Part 2, send them alongside, packed on their own: we already
hold everything you sent with Part 1, so only the new chunks need to travel.

```bash
python parity/meep_gpu/rounds/1.0/pack_results.py part1 --index "$INDEX" --lanes "$LANES" \
    --round-dir "$ROUND_DIR" --selection "$SEL" --verify "$VERIFY" --since <the last chunk of Part 1>
```

It packs only the chunks newer than the one you name (or exactly those you list with
`--chunks c9,c10,c11`), with your whole index, selection and verify report, and its
`CHUNKS.json` says which chunks it carries and which it leaves out. An earlier run that a
newer chunk replaced stays on your machine: the verify report already selects the newer one.

## The CUDA driver the route stage needs

The NVIDIA certification is a statement about the whole step, so it names the stack the
kernels ran beside, axis by axis: CuPy, NVRTC, the CUDA runtime, **the CUDA driver**, the
host's OS and architecture, NumPy and the MEEP build. It is recorded **per compute
capability**, from that capability's own certification: each gate lane records the stack it
ran beside on its card (`chunk_<stamp>/stack/<lane>.json`), and the bind writes it into
`meep_gpu/triton_kernels/fingerprints.json` under `certified_nvidia_stack`
(`record_nvidia_stack.py`, step 10; read it with `fastpath.certified_nvidia_stack('<cap>')`).
Compute capabilities 8.6 and 8.9 are certified on CuPy 13.5.1, NVRTC 11.8, CUDA runtime 11080, Linux
x86_64, NumPy 2.2 and the MEEP 1.35.0 single-precision build, on each CUDA driver its record
lists (8.6: 12030 and 13040; 8.9: 13050). Under the
default hardware scope a run beside another stack than its capability's is recorded as
uncertified, with a NOTE, and refuses nothing. Under `MEEP_GPU_HARDWARE=certified` every plan
whose stack is off its capability's is refused, and so is a capability with no recorded stack.

- **Part 1 runs on any driver.** No gate of CERT-3 enables dispatch, so none reaches that
  check: a trial released every smoke gate on a driver that supplies CUDA 13.4. The drivers
  Part 1's gates run on are the ones the capability is certified on: keep one of them until
  the route has run.
- **A GPU family can be certified on several drivers.** The bind accepts lanes that ran on
  different drivers when they read the same value on every other axis, and the record lists
  every driver its certified runs used, with the chunk stamps that ran on each; the release's
  records are that list (`fastpath.certified_nvidia_stack('<cap>')` prints it). Each listed
  driver is certified by runs on it: a driver no certified run used is never listed, however
  close its version. Lanes that disagree on any other axis are refused.
- **The route stage's admitted-policy legs run under `MEEP_GPU_HARDWARE=certified`** (each
  route gate sets it for its admitted-policy leg, the flush legs, and requires at least one
  case to dispatch there). On a stack that is not the one your capability was certified on
  every such case would be refused at the stack check, so the route stage refuses to start,
  naming the axis, and the leg itself refuses before it runs a case. The legs under the
  default scope only record the NOTE.

So, before the route stage, `cupy.cuda.runtime.driverGetVersion()` must print a driver the
bind recorded for your capability: one Part 1 ran on. If the driver changed between Part 1
and Part 2 (an update, a cluster node on another driver), do not start the route: put a
driver Part 1 ran on back, or tell the maintainers the driver version step 2 printed, and they
decide where and how the route of your capability runs. A capability the ledgers already
certify is judged on the drivers its record lists: on 8.6 the route needs driver 12030 or
13040 (12030 is the 545 series: 545.23.08 on Linux, under WSL2 a Windows driver such as
GeForce 546.33; 13040 under WSL2 is a Windows driver such as 616.56, the RTX 3090's; check
the number, not the name: drivers newer than the 545 series print
`CUDA UMD Version` in the header of `nvidia-smi` and no `CUDA Version` line).

## Platform notes

### A desktop under WSL2

**WSL2 and Ubuntu.** Virtualization must be on in the firmware (Task Manager, Performance,
CPU: `Virtualization: Enabled`). In PowerShell as Administrator run `wsl --install -d
Ubuntu-24.04` and restart; Ubuntu 24.04 is the version this was tried on. At the first start
choose a Linux user name and password. Then, in an ordinary PowerShell, `wsl --update` and
`wsl -l -v`: the line for `Ubuntu-24.04` must end in **2**. Inside Ubuntu, `sudo apt update
&& sudo apt install -y git curl ca-certificates tar gzip bzip2 python3`.

**Memory.** WSL2 takes half of the PC's memory unless told otherwise. Create
`%UserProfile%\.wslconfig` containing

```
[wsl2]
memory=24GB
```

(more if the PC has it), then `wsl --shutdown` in PowerShell and open Ubuntu again; `free -g`
then shows about that much.

**The driver.** Under WSL2 the Windows NVIDIA driver supplies CUDA to Linux. Install nothing
NVIDIA inside WSL2: no Linux driver, no `cuda` packages; the libraries appear under
`/usr/lib/wsl/lib` by themselves. If `nvidia-smi` is there but not on `PATH`, the preflight
prints the `sudo ln -s` that fixes it; if `libcuda.so.1` is not in the linker cache,
`/etc/ld.so.conf.d/ld.wsl.conf` should name `/usr/lib/wsl/lib`, then `sudo ldconfig`. For Part
2 the Windows driver must be one Part 1 ran on: hold Windows driver updates until the route
has run.

**The display.** Under WSL2 every card counts as a display card, because Windows owns the
display: the round takes it by the display rule ([A card that also drives your
display](#a-card-that-also-drives-your-display)). On the RTX 3090 this was tried on, the
desktop held 692 MiB and the smoke chunk took the card at once and passed every step. Close,
before you start, every Windows program that uses the card (games, video players, browsers
with hardware acceleration, overlays and capture tools, RGB and monitoring tools that use the
GPU), and do not open them until the round is done.

**Keep Linux up.** WSL2 stops the Linux side, and every job in it, within a minute after the
last Windows program that uses it closes (measured: stopped 25 s later; the jobs were gone
after 150 s). A job started with `setsid nohup` survives closing its own terminal only while
something else keeps WSL2 up, so open a second, separate PowerShell window, run `wsl -d
Ubuntu-24.04 -- sleep infinity`, and leave it open until the results are packed. Settings,
System, Power, Sleep: Never; pause Windows Update for the length of the round. A reboot loses
no finished work (step 7 resumes) but costs the step that was running.

**The display watchdog.** Windows resets a GPU when one piece of its work runs longer than
`TdrDelay` (2 s when the value is not set). The round's kernels are short: a whole time step
of a 7.1-million-cell 3-D case takes about 2.5 ms on an RTX A6000 across all its launches,
and the gates step smaller cells. The preflight prints `TdrDelay`; after the round,
`Get-WinEvent -FilterHashtable @{LogName='System'; Id=4101} -MaxEvents 5` in PowerShell lists
any "Display driver stopped responding" events. If there are any, say so; do not change
`TdrDelay`.

**Timing on a desktop.** The round times nothing ([Timing is not part of the
round](#timing-is-not-part-of-the-round), which also says where a timing record for your
GPU lands). A timing record taken later on a desktop under WSL2 runs only through the
timing driver, whose gate reads the Windows side there, because the layer and the desktop
disturb a measurement in ways the Linux side cannot see: on the RTX 3090 this was tried
on, with its clocks free, slow passes came with the card dropping from P2 to P3. **Clock
locks are expected on a desktop timing host**: lock both clocks in
PowerShell as Administrator (`nvidia-smi -lgc G,G`, `nvidia-smi -lmc M,M`; 1695 and 9501 MHz
on that RTX 3090) and give the same values to the driver (`--gpu-clock-lock G,M`). At every
reading of its quiet window and its watcher the gate then reads the card through Windows'
`nvidia-smi.exe` and requires both clocks at the lock and no clock-event reason but
`gpu_idle`, and reads Windows' CPU load through `powershell.exe` and requires it, outside
the WSL VM, below a threshold: 10 % of Windows' logical CPUs by default, which starting a CUDA
process under WSL2 can exceed on its own (2.0 to 6.07 busy CPUs on that RTX 3090), so a
desktop declares its own from a calibration (`--windows-cpu-max 8` there, a third of its 24
logical CPUs; every record states the value it ran under). One reading that fails while a unit runs
strikes the unit, which runs again; a Windows side that cannot be read (WSL interop off)
refuses the session by name. **A desktop timing host is qualified by the variance check**,
under the settings it will be timed with: that RTX 3090 passed only with its clocks locked
**and** the WSL VM's process kept off the logical CPUs that carry the GPU path's interrupts
and DPCs, by a CPU affinity mask (`--vm-affinity 0x999988` there). The driver sets the mask
through interop before its first unit, refuses unless it reads back (only an elevated context
can set it, so start WSL and the driver from one), requires it at every reading, strikes a
unit during which it changed, and restores the earlier mask at its end. MEEP is never timed
under it: with `--split-ladder` the driver times each ladder unit as two child runs of one
session, the GPU legs under the mask and the MEEP legs with it cleared, the first of the two
alternating unit by unit, and composes them into one row whose ratios are labelled split
(on that RTX 3090: an 8-performance-core desktop, MEEP at 1 to 8 ranks); without the
split, a session that declares a mask and holds MEEP rows is refused. Under a mask the
variance precondition times the GPU route alone, split or not. The MEEP halves are held to
the host's limits as its site file declares them: the largest of its `RANKS` (`8|1|` there)
is the most MEEP ranks any MEEP row or control the catalog asks for may take, a control
above it (the catalog's `hwthread`, at 11 and 16 ranks) is refused by name, its
`PHYSICAL_CORES` (`8` there; WSL2 reports 12) may not exceed it, and a site file that
declares no `RANKS` is refused by name. A mask names one
machine's logical CPUs: after a driver update or a hardware change, find the card's interrupt
target set and the busiest DPC logical CPUs again, read-only, and qualify again. The rule and
the readings are in [Running the timing ladder](timing.md#a-desktop-under-wsl2), the split
ladder in [The split ladder](timing.md#the-split-ladder).

### A Linux workstation

The steps run as written. Install the NVIDIA driver from the distribution or NVIDIA's
packages, and keep the one Part 1 runs on for Part 2. A card that drives the
workstation's monitor is taken by the display rule; a card that drives none by the idle rule,
so another user's process or memory on it keeps the round off it. To run with no desktop at
all, `sudo systemctl isolate multi-user.target` and work over SSH or a text console. Use
`setsid nohup` for the long steps (step 6) so a closed terminal does not stop them, and keep
the machine from sleeping. If others use the machine, the etiquette of [An HPC
node](#an-hpc-node) applies.

### An HPC node

The round runs on a cluster with a batch scheduler as on any Linux host, with these
differences. The examples are Slurm's; on another scheduler use its equivalents.

**Where to build.** The build needs the network (GitHub, conda-forge, PyPI) and no GPU, but
it makes the link Triton's launcher compiles against only on a host with an NVIDIA driver.
If the GPU nodes reach the network, build in a short job on one of them. If only the login
node does, build there with your share of its processors, then make the link once, on a GPU
node, with the build's own step (read from the build script; it writes the record every lane
reads):

```bash
MEEP_BUILD_JOBS=4 MEEP_GPU_BUILD_THREADS=4 MEEP_BUILD_ROOT=<scratch>/meep_build \
    bash parity/meep_gpu/build_meep_reference_linux.sh 2>&1 | tee "$ROUND_DIR/build.log"
# then, inside a job on a GPU node, with the environment active:
python parity/meep_gpu/build_meep_reference_record.py link-libcuda "$CONDA_PREFIX" \
    "$CONDA_PREFIX/share/meep-gpu-build/libcuda.json"
```

Without the link, and without a `libcuda.so` beside the driver's `libcuda.so.1`, Triton's
launcher fails to link (`cannot find -lcuda`); the preflight refuses that environment, naming
the command. A home directory with a small quota
holds neither the environment (9.1 GB) nor its package cache (4.8 GB): point
`MEEP_ENV_PREFIX` and `CONDA_PKGS_DIRS` at a larger file system the GPU nodes see.

**One job, one node, several cards.** Run every gate chunk in one job on one node: the family
re-certification of the bind takes its nine families from one machine (step 10), and cards
of one node are one machine. Ask for whole cards of one model, and run the preflight and the
chunks inside the job:

```bash
#!/bin/bash
#SBATCH --job-name=meep-gpu-cert
#SBATCH --partition=<the GPU partition>
#SBATCH --gres=gpu:<the card's gres name>:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=<ROUND_DIR>/%x_%j.out
source "$HOME/meep-gpu-round/env.sh"
source "$HOME/miniforge3/etc/profile.d/conda.sh" && conda activate meep-gpu-ref
export LOCKS="$ROUND_DIR/locks/job_$SLURM_JOB_ID"
mkdir -p "$LOCKS"
cd "$REPO"
python parity/meep_gpu/rounds/1.0/preflight.py --capability "$CAP" --expect "$EXPECT" \
    > "$ROUND_DIR/preflight_$SLURM_JOB_ID.txt" 2>&1 || exit 1
GPU_IDS=$(python parity/meep_gpu/rounds/1.0/preflight.py --capability "$CAP" --cards-only) || exit 1
python parity/meep_gpu/rounds/1.0/gpu_log.py start --dir "$ROUND_DIR" --gpu-ids "$GPU_IDS"
python -u parity/meep_gpu/rounds/1.0/run_chunks.py --index "$INDEX" --lanes "$LANES" \
    --lock-dir "$LOCKS" --expect "$EXPECT" --gpu-ids "$GPU_IDS" --spread --continue-on-fail \
    c0 c1 c2 c3 c4 c5 c6 c7 >> "$LANES/run_chunks_$SLURM_JOB_ID.log" 2>&1
python parity/meep_gpu/rounds/1.0/gpu_log.py stop --dir "$ROUND_DIR"
```

Write `ROUND_DIR`'s absolute path in the `--output` line (a scheduler does not expand the
shell's variables there), so that each job's output file lands where the packer takes it.
Run the smoke chunk first the same way, in a short job of one card naming `c8`. Then:

- **The cards** are the ones `nvidia-smi` lists inside the job, as the preflight reads them:
  a scheduler that confines a job to its cards lists only those (indexed from 0, as many as
  `CUDA_VISIBLE_DEVICES` names); one that does not lists every card of the node, and the job's
  are the ones `CUDA_VISIBLE_DEVICES` names. `--cards-only` prints them, and refuses a mask it
  cannot match (devices named by UUID: then pass `--gpu-ids` yourself).
- **The locks are per job** (`$LOCKS` above): lock directories are matched by card index, and
  two jobs on one node may both see their card as index 0, so a lock directory shared between
  jobs would make one wait for the other's card.
- **The time limit.** With eight chunks over four cards each card runs two; at 0.55 to 0.70 of
  the price that is about 3.7 to 4.6 hours, plus a few minutes of lane start a chunk (an
  estimate from the A6000 walls above; nobody has timed the gates on a cluster node). A job
  that ends at its limit leaves its chunks resumable: submit the same script again, with
  `#SBATCH --nodelist=<the node the first job ran on>` (`sacct -j <job id> --format=NodeList`),
  so that every run stays on one machine. If that node is not available, finish on another
  and run the nine families again together there (step 10).
- **The memory and processors.** Each gate process runs with one OpenMP thread; four
  processors a card leave room for the lane and the compilers. The preflight asks for 24 GiB
  for one card and 12 GiB more for each further card, and reads the job's own limit.
- **An idle card on a server** has no display and keeps the idle rule: no compute process,
  0 % utilisation, at most 200 MiB in use, for the lane and for the Triton fleet driver
  before each gate alike (one rule places every gate). A card that holds memory no listed process accounts for (another job's process in
  another container, a monitoring agent) is refused, and the lane waits for it; tell the
  maintainers if a card the scheduler gave you reads that way.
- **The bind and the packer** need no GPU, but run the packer in a short job on a GPU node,
  so that the `nvidia-smi` and `check_install.py` records it takes describe the cards.
- **The route stage** runs in a job of two cards (step 11), and only beside a driver
  Part 1's gates ran on, so on nodes of such a driver: [The CUDA driver the route stage
  needs](#the-cuda-driver-the-route-stage-needs).

**On a shared machine.** Take only what you use, and give it back: ask for the cards the
round runs on and no more, run no allocation that waits idle, and build with your share of a
login node's processors. On a shared machine without a scheduler, cap the cards the round
holds with `--max-locks`, leave at least a quarter of the machine's cards free for others,
and follow [Sharing a card with another user's idle process](#sharing-a-card-with-another-users-idle-process)
to yield a card its owner needs. The lanes never take a card that carries another user's
compute process.

## An Apple GPU

A Mac whose Apple GPU has an architecture the release does not certify, for example an M3,
runs the same round with the Metal tools. The commit, the rules and the inputs archive are
the ones above.

**The reference environment**, the macOS build of the same reference (MEEP 1.35.0 in single
precision with MPI, compiled with `-O2 -ffp-contract=off -fno-fast-math` so that its field
updates round every operation, on top of the macOS locks, with the certified PyTorch):

```bash
cd "$REPO"
MEEP_BUILD_JOBS=4 bash parity/meep_gpu/build_meep_reference_macos.sh 2>&1 | tee "$ROUND_DIR/build.log"
conda activate meep-gpu-ref
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$REPO" python tools/check_install.py --require-reference
```

On Apple silicon MEEP's own subnormal switch has no effect and the Metal table flushes on the
GPU, so the build records its flush identity verdict as pending, by name; that is expected.
Keep `PYTORCH_MPS_FAST_MATH` unset, and do not export any `MEEP_GPU_*` variable. Stage the
inputs as in step 5 (`3386 of 3386 verified`).

**The architecture, and your chunk index:**

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

If `ARCH` reads `applegpu_g13s` or `applegpu_g16g`, this Mac's GPU is already certified. Otherwise the index
holds three chunks of the 62 Metal gates (`m0`, `m1`, `m2`, of 1,685, 1,423 and 1,422 s at
M1 Max prices, stamps `<round>_<architecture>_m<k>`), and `check` must read `CHECK OK (0
problem(s))`.

**CERT-3, the Metal chunks**, one after another (the Mac has one GPU):

```bash
cd "$REPO"
for k in 0 1 2; do
  python -u parity/meep_gpu/rounds/1.0/metal_round.py all --repo "$REPO" \
      --chunk-index "$INDEX" --chunk m$k --new-architecture --expect "$EXPECT" \
      --meep-source "$CONDA_PREFIX/share/meep-gpu-build/meep-1.35.0" \
      > "$ROUND_DIR/metal_round_m$k.out" 2>&1 || break
done
```

`all` runs the preflight, the chunk's gates one at a time (each with a watchdog, keeping the
Mac awake with `caffeinate`), the environment record before the first gate and after the
last, and the chunk's status; follow `parity/meep_gpu/results/logs_metal_<chunk
stamp>/round.log`, and resume by running the same command again. The preflight holds the
index and `--new-architecture` to the Metal ledger: the flag is refused on an architecture
the ledger admits. **GO** when `python -u parity/meep_gpu/rounds/1.0/metal_round.py status
--repo "$REPO" --chunk-index "$INDEX"` reads `62 of 62 Metal gate artifacts released` and
`cited welds released: 45 of 45`.

**Verify, then bind** (Part 2, on the records commit, as for NVIDIA):

```bash
cd "$REPO/parity/meep_gpu"
python rounds/1.0/chunks.py verify --index "$INDEX" --repo ../.. --results results \
    --selection "$SEL" --json "$VERIFY"
python rounds/1.0/chunks.py bind --index "$INDEX" --selection "$SEL" | tee "$ROUND_DIR/metal_bind_commands.txt"
```

**GO on `VERIFY OK: 62 of 62 units hold`.** The bind prints `chunks.py verify --against
"$SEL"`, then `rebind_metal_welds.py --stamp <chunk stamp>` once per selected stamp in launch
order, and last one filing line, `ledger_scope.py archive`; run them as in step 10, the filing
line after the route. Each write adds the released weld's run under `runs["<architecture>"]`
beside the `applegpu_g13s` and `applegpu_g16g` runs. **Do not run `launch.write_fingerprints()`**: it rewrites the
table's own toolchain block from this Mac's PyTorch and Metal frontend. `git status` then lists
only `meep_gpu/metal_kernels/fingerprints.json`, and

```bash
python -c "from meep_gpu import metal_runs as r; from meep_gpu.metal_kernels import launch; print(r.admission_report(launch.load_fingerprints(), r.cited_keys())['admitted'])"
```

lists your architecture beside `applegpu_g13s` and `applegpu_g16g`.

**CERT-4, the Metal route campaign:**

```bash
cd "$REPO"
ROUND=$(python -c "from meep_gpu import route_stamps; print(route_stamps.stamp('metal'))")
zsh parity/meep_gpu/run_metal_dispatch_campaign.sh --dry-run --lanes 3 "${ROUND}_$ARCH"
nohup python -u parity/meep_gpu/rounds/1.0/metal_round.py route --repo "$REPO" \
    --stamp "${ROUND}_${ARCH}_route" --expect "$EXPECT" \
    --meep-source "$CONDA_PREFIX/share/meep-gpu-build/meep-1.35.0" \
    > "$ROUND_DIR/metal_route.out" 2>&1 &
```

The first command prints the campaign's plan and launches nothing; the second runs it under
held residency into `results/dispatch_metal_route_<round>_<architecture>/`. It refuses if any
file but the bound Metal ledger differs from the commit, and its campaign name is one-shot.
**GO** when the stage's last line reports exit 0: every leg released.

**What you send back:** the chunks' and the route's results roots
(`*_<round>_<architecture>_m*`, `logs_metal_<round>_<architecture>_route`,
`dispatch_metal_route_<round>_<architecture>`, any `set_aside_*`), the index, the selection
and the verify report, the bound `meep_gpu/metal_kernels/fingerprints.json`, the `*.out`,
bind and build logs of `ROUND_DIR`, `check_install.py --require-reference`, the build
records under `$CONDA_PREFIX/share/meep-gpu-build/`, and a `SHA256SUMS` of all of it, in one
archive; send it privately, and its sha256 in a separate message.

## The direct corpus check (L3) is not run on a new host

The certification round also runs L3 (`rounds/1.0/l3_corpus.py`,
[The direct corpus check against MEEP](certification.md#the-direct-corpus-check-against-meep-l3)):
the corpus stepped with kernels dispatched and judged against the stored digests of the
reference MEEP's words. Each of its rows refuses unless the MEEP libraries the process
loaded have the sha256 its record names (`rounds/1.0/l3_row.py`), and a MEEP built in an
environment at another path does not have them: the build links the environment's absolute
path into `libmeep` and `_meep`. L3 therefore runs only where the record holds the reference
build the host loads: on the hosts that cut its record, and on a host whose own build of the
reference recipe the maintainers minted into the record as a second reference, from that
build's own corpus run (`--second-reference`, RUNBOOK CERT-5). The 1.0.1 record holds one,
`linux_avx2_keep` and `linux_avx2_flush`, minted on a CPU whose widest vector extension is
AVX2. The two Linux builds' words agree on every row where the Apple and Linux references
agree; on the host-dependent rows, the eigenmode solves, reference words depend on the BLAS
kernel family OpenBLAS picks for the reference CPU's vector extensions, so a second build can
differ there, and L3 then judges those rows against that build's own words. A later release
may pin the recipe's BLAS kernel family. Where L3 runs, the rows with an `EigenModeSource`
are declared run-to-run variable and may run up to 4 times, since MPB's solve of their mode is
not reproducible from one process to the next. L3 is not part of a round on a new GPU; the
gates and the route campaign are.

## Timing is not part of the round

Nothing in a round times anything: the gates and the route legs are bit-identity checks,
so a neighbour on the host costs wall time only. A timing record for your GPU
([Running the timing ladder](timing.md)) is a separate, optional run after the bind, on an
otherwise idle machine, with MEEP re-timed in the same session. On a desktop under WSL2 it
needs the card's clocks locked ([A desktop under WSL2](#a-desktop-under-wsl2)). The record
lands at its architecture's name (<code>timing_cc&lt;capability&gt;.json</code>) when that
architecture has none, and then represents it: every card of the architecture without a
record of its own reads it, and a GPU of an architecture with no record whose nearest
timed compute capability is yours reads its bracketed-veto prior (never the record
itself). It ships as data alone, with no code change, once it is cut on the release's
bytes and holds on the tree. When the
architecture already has a record cut on another card, yours lands beside it under its
card name (<code>timing_cc&lt;capability&gt;-&lt;card&gt;.json</code>), and once it ships
your card reads it, while every other card of the architecture reads the representative
([Records per card](timing.md#records-per-card)). Until a record of your GPU's
architecture ships, your GPU reads the prior of the nearest compute capability that has
one.

## What happens to what you send

The maintainers check each archive against its sha256 and its `SHA256SUMS`; unpack its
results into a clone of the same commit; run `chunks.py verify` with your index on your
artifacts, which checks that every file each of them recorded importing is byte-equal to the
commit and that each ran on your card; after Part 2, re-run the bind from your artifacts on
the records commit and compare the ledgers it writes with yours, check that the only change is the
added runs for your GPU, and cut the dispatch records from your route legs; and publish the records
in a 1.0.x release. Your GPU is then certified beside the stack its round ran on (on NVIDIA, with
Triton 3.1.0): a run beside them is certified, and `MEEP_GPU_HARDWARE=certified` admits it.
