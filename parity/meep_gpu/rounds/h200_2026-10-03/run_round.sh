#!/usr/bin/env bash
# A meep-gpu certification round on ONE NVIDIA GPU: the device gates for both NVIDIA
# tables, written where the laptop-side rebinds read them. It runs gates only; it writes
# no ledger. Run it from anywhere, under nohup, one stage at a time or all of them:
#
#   conda activate meep-gpu-ref     # built by parity/meep_gpu/build_meep_133_linux.sh
#   export REPO=$HOME/meep-gpu  EXPECT=<commit to certify>  STAMP=2026-10-02_cc90  GPU=0
#   export INPUTS=$HOME/meep_gpu_round_inputs.tgz
#   nohup bash run_round.sh all > $HOME/round_$STAMP.out 2>&1 &
#   tail -f $HOME/round_$STAMP.out          # one flushed line per stage and per gate
#
# Stages, in order (each one refuses to start if the one before it did not finish):
#   preflight  toolchain, card, commit, environment            (writes nothing but notes)
#   inputs     unpack and verify the evidence inputs bundle     (needs INPUTS=<.tgz>)
#   triton_a   33 gates; measures this card's expansion record  (~80 min on an A6000)
#   triton_b   11 complex gates on A's expansion record          (~25 min)
#   triton_c   complex_offdiag with A's record as --probe        (~1 min)
#   triton_d   5 composition legs, after the staged manifest     (~5 min)
#   stencil    2 Triton stencil welds, by hand                   (~4 min)
#   bitid      the Triton bit-identity probe, by hand            (~10 min)
#   flushexp   this card's FLUSH expansion record, for CUDA      (~2 min)
#   cuda       70 CUDA legs, keep + flush (run_cuda_welds.sh)    (~100 min)
#   status     what released, per root
#   pack       one tarball of every root and log, without caches
#
# ALLOW_NO_FLUSH=1 lets the round go on when MEEP cannot flush subnormals on this host;
# the flush-dependent work is then skipped by name (see stage_preflight).
#
# Everything lands under $REPO/parity/meep_gpu/results/ (gitignored), which is where the
# rebinds require it. Every root is new: a stage refuses an out-root that already exists,
# because a gate re-run into an old root mixes two runs.
set -u -o pipefail
STAGE=${1:?usage: run_round.sh <stage|all>}
REPO=${REPO:?set REPO to the clone of the commit being certified}
# Resolved once: a HOME with a trailing slash or a symlink would otherwise make the
# clone path differ, as a string, from the path Python reports for the same files.
REPO=$(cd "$REPO" 2>/dev/null && pwd -P) || { echo "REPO is not a directory"; exit 2; }
EXPECT=${EXPECT:?set EXPECT to the full commit hash being certified}
STAMP=${STAMP:?set STAMP, e.g. 2026-10-02_cc90 (must differ from any 8.6 root name)}
GPU=${GPU:-0}
PY=${PY:-$(command -v python)}
HERE=$(cd "$(dirname "$0")" && pwd)
G=$REPO/parity/meep_gpu
R=$G/results
L=$R/logs_$STAMP
HC=${HC:-$HOME/round_caches_$STAMP}       # hand-run gate caches: OUTSIDE results/
DRIVER="$PY -u $G/drive_triton_weld_gates.py --gpu $GPU --forbidden-gpus '' --policy"
mkdir -p "$L" "$HC"
export PYTHONUNBUFFERED=1 CUPY_ACCELERATORS=
unset MEEP_GPU_COMPLEX_EXPANSION_PROBE MEEP_GPU_SUBNORMAL_POLICY MEEP_GPU_BACKEND_PREFERENCE \
      MEEP_GPU_ALLOW_UNCERTIFIED MEEP_GPU_DISPATCH 2>/dev/null || true

say()  { echo "$(date -u +%FT%TZ) [$STAGE_NOW] $*" | tee -a "$L/round.log"; }
die()  { say "STOP: $*"; exit 3; }
done_mark() { touch "$L/.done_$1"; say "finished in $(( $(date +%s) - T_STAGE ))s"; }
need() { [ -f "$L/.done_$1" ] || die "stage '$1' has not finished; run it first"; }
fresh() { [ ! -e "$1" ] || die "$1 already exists; a gate re-run into an old root mixes two runs. Use a new STAMP or move it aside"; }
gpu_idle() {
  local used; used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${used:-999}" -le 64 ] || die "GPU $GPU holds ${used} MiB; the fleet driver needs <= 64 MiB (another process is on the card)"
}
# The gates' own environment for the two hand-run stages, as the fleet driver sets it.
hand_env() {
  env CUDA_VISIBLE_DEVICES="$GPU" MEEP_GPU_DISPATCH=0 \
      PYTHONPATH="$REPO:$G" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      KMP_DUPLICATE_LIB_OK=TRUE "$@"
}

stage_preflight() {
  [ -n "${TRITON_LIBCUDA_PATH:-}" ] || die "TRITON_LIBCUDA_PATH is unset: do INSTALL.md step 3. Without it every Triton device leg SKIPS cleanly and nothing certifies"
  [ -e "$TRITON_LIBCUDA_PATH/libcuda.so" ] || die "$TRITON_LIBCUDA_PATH/libcuda.so is missing (INSTALL.md step 3)"
  case ":${LD_LIBRARY_PATH:-}:" in *":$TRITON_LIBCUDA_PATH:"*) ;; *) die "LD_LIBRARY_PATH does not contain $TRITON_LIBCUDA_PATH (INSTALL.md step 3)";; esac
  # Triton compiles a launcher with $CC, else gcc, else clang (triton/runtime/build.py).
  # The reference build sets CC to its own compiler, so a host without a system
  # compiler still has one.
  local cc=${CC:-$(command -v gcc || command -v clang || true)}
  { [ -n "$cc" ] && command -v "${cc%% *}" >/dev/null; } \
      || die "no C compiler: Triton compiles a launcher with \$CC, else gcc, else clang, and found none"
  say "C compiler for Triton launchers: $cc ($("${cc%% *}" -dumpfullversion 2>/dev/null || "${cc%% *}" -dumpversion 2>/dev/null))"
  local head; head=$(git -C "$REPO" rev-parse HEAD) || die "$REPO is not a git checkout"
  [ "$head" = "$EXPECT" ] || die "HEAD is $head, not $EXPECT: both architectures must run the SAME commit"
  [ -z "$(git -C "$REPO" status --porcelain)" ] || die "the checkout has local changes; the rebinds compare every digest to the commit"
  git -C "$REPO" ls-files --error-unmatch parity/meep_gpu/cases.py >/dev/null 2>&1 \
      || die "parity/meep_gpu/cases.py is not in this commit; three composition gates import it"
  ( cd "$REPO" && PYTHONPATH="$REPO:$G" "$PY" - "$REPO" <<'EOF' ) | tee -a "$L/round.log" || die "the package import check failed (above)"
import os, sys
repo = os.path.realpath(sys.argv[1])
try:
    import pytest
except ImportError:
    print("REFUSED: pytest is not installed (python -m pip install 'pytest>=8.2'); four shard-B gates import a test module"); sys.exit(1)
import meep_gpu
f = os.path.realpath(meep_gpu.__file__)
print("meep_gpu from", f, "| pytest", pytest.__version__)
if not f.startswith(repo + os.sep):
    print(f"REFUSED: meep_gpu imports from {f}, not from the clone {repo}; an installed copy is shadowing it"); sys.exit(1)
EOF
  "$PY" - "$GPU" <<'EOF' | tee -a "$L/round.log" || die "toolchain or card check failed (above)"
import sys
import cupy, triton, meep
p = cupy.cuda.runtime.getDeviceProperties(0)
name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
cc = f"{p['major']}.{p['minor']}"
from cupy_backends.cuda.libs import nvrtc
nvrtc_version = ".".join(str(part) for part in nvrtc.getVersion())
print(f"card {name} | cc {cc} | cupy {cupy.__version__} | nvrtc {nvrtc_version} | triton {triton.__version__} | meep {meep.__version__} "
      f"{'single' if meep.is_single_precision() else 'double'} {'MPI' if meep.with_mpi() else 'serial'}")
bad = []
if triton.__version__ != "3.1.0": bad.append(f"Triton {triton.__version__} (the rebind accepts only 3.1.0)")
if cupy.__version__ != "13.5.1": bad.append(f"CuPy {cupy.__version__} (the rebind accepts only 13.5.1)")
if meep.__version__ != "1.33.0": bad.append(f"MEEP {meep.__version__} (the censuses were cut against 1.33.0)")
if not (meep.is_single_precision() and meep.get_realnum_size() == 4):
    bad.append("MEEP is double precision: a round is measured against the single-precision source build "
               "(bash parity/meep_gpu/build_meep_133_linux.sh, then conda activate meep-gpu-ref)")
if not meep.with_mpi():
    bad.append("MEEP is not an MPI build: the reference is the source build with MPI (build_meep_133_linux.sh)")
if bad: print("REFUSED: " + "; ".join(bad)); sys.exit(1)
EOF
  # FLUSH NEEDS MEEP TO FLUSH. The flush subnormal policy reaches the host through
  # 'import meep', which sets FTZ/DAZ only when MEEP was built with that support. Every
  # certified round ran a source-built MEEP; the conda-forge build is unmeasured here.
  # MEEP prints a preamble and an "Elapsed run time" trailer on stdout, so the answer is
  # tagged and picked out, and stderr is kept: no answer is a different failure from "False".
  local flushed
  ( cd "$REPO" && PYTHONPATH="$REPO" "$PY" -c "import meep; from meep_gpu import backends; print('FLUSHED=' + str(backends.subnormals_flushed()))" ) \
      > "$L/flush_check.out" 2> "$L/flush_check.err"
  flushed=$(grep -o 'FLUSHED=[A-Za-z]*' "$L/flush_check.out" | head -1 | cut -d= -f2)
  [ -n "$flushed" ] || die "could not evaluate the flush check (MEEP or meep_gpu did not answer): $(tail -3 "$L/flush_check.err" | tr '\n' ' ')"
  say "subnormals flushed after 'import meep': $flushed"
  if [ "$flushed" != True ]; then
    if [ "${ALLOW_NO_FLUSH:-0}" = 1 ]; then
      touch "$L/.no_flush"
      say "WARNING: no flush on this host. Going on under ALLOW_NO_FLUSH=1: the flush expansion record and all CUDA legs are skipped, and Triton's cylindrical_fused_magnetic_pair will refuse its flush half. Neither table can admit this card from tonight alone"
    else
      die "this MEEP does not flush subnormals on import, so the 34 CUDA flush legs, the flush expansion record and Triton's cylindrical_fused_magnetic_pair would all refuse and neither table could admit this card. Tell Ivan (a source-built MEEP fixes it), or re-run preflight with ALLOW_NO_FLUSH=1 to produce the keep-side evidence anyway"
    fi
  fi
  gpu_idle
  { date -u +%FT%TZ; hostname; git -C "$REPO" rev-parse HEAD; nvidia-smi; "$PY" -m pip freeze; } > "$L/environment_start.txt" 2>&1
  say "commit $EXPECT, GPU $GPU idle, toolchain pinned"
}

stage_inputs() {
  need preflight
  : "${INPUTS:?set INPUTS to the path of meep_gpu_round_inputs.tgz}"
  [ ! -e "$R/complex_signed_zero_2026-08-12" ] || die "results/complex_signed_zero_2026-08-12 is present: two shard-B gates read that A6000 record IN PREFERENCE to this card's own, without refusing. Move it out of results/; it is not part of the bundle"
  mkdir -p "$R"
  tar -xzf "$INPUTS" -C "$R" || die "could not unpack $INPUTS"
  ( cd "$R" && sha256sum -c --quiet INPUTS_SHA256SUMS ) || die "an input file does not match its checksum; re-copy the bundle"
  cp "$R/INPUTS_SHA256SUMS" "$L/"
  say "inputs: $(wc -l < "$R/INPUTS_SHA256SUMS") files verified"
}

run_driver() {  # <log> <out-root> <policy> <gates> [extra driver args...]
  local log=$1 root=$2 policy=$3 gates=$4; shift 4
  fresh "$root"; gpu_idle
  say "driver -> $root ($(echo "$gates" | tr ',' '\n' | grep -c .) gates)"
  eval "$DRIVER $policy --patience 300 --out-root \"$root\" --gates \"$gates\" $*" >> "$L/$log" 2>&1
  local rc=$?
  grep -E "\[gate\]" "$L/$log" | tail -40 >> "$L/round.log"
  [ -f "$root/campaign.json" ] || die "the driver did not write $root/campaign.json (exit $rc); see $L/$log"
}

A_GATES=unified_expansion,complex,fused_electric,no_pml_constitutive,offdiag,nonlinear,bfast,special_kz,folded_complex,cylindrical_complex,complex_ade,complex_no_pml_stored_e,no_pml_conductive,no_pml_stored_e,folded_offdiag,cylindrical_fused_electric_pair,cylindrical_fused_magnetic_pair,complex_conductive_fused_pair,folded_dispersive_fused_pair,cylindrical_real_fused_electric_pair,beta_fused_electric_pair,folded_beta_fused_electric_pair,folded_beta_fused_magnetic_pair,no_pml_fused_electric_pair,bfast_fused_electric_pair,conductive_fused_electric_pair,probe_triton_cylindrical_real_fused_magnetic_pair,probe_triton_dispersive_fused_pair,probe_triton_folded_fused_magnetic_pair,probe_triton_folded_fused_pair,probe_triton_fused_ade_state,beta_fused_magnetic_pair,bfast_fused_magnetic_pair
B_GATES=complex_beta_fused_electric_pair,complex_beta_fused_magnetic_pair,complex_fused_electric_pair,folded_beta_complex_fused_magnetic_pair,folded_beta_complex_fused_pair,folded_complex_fused_pair,nonlinear_fused_magnetic_pair,probe_triton_complex_fused_magnetic_pair,complex_no_pml_conductive,complex_no_pml_curl,probe_triton_folded_complex_fused_magnetic_pair
D_GATES=cylindrical_composition,symmetry_composition,conductivity_composition,engine_route,no_pml
A_ROOT=$R/triton_fleet_$STAMP
UNIFIED_KEEP=$A_ROOT/unified_expansion/gate.json

stage_triton_a() {
  need inputs
  run_driver triton_a.log "$A_ROOT" keep "$A_GATES"
  # The canary the rest of the round stands on: this card's own expansion record.
  "$PY" - "$UNIFIED_KEEP" "$A_ROOT/device.json" <<'EOF' | tee -a "$L/round.log" || die "shard A did not produce a usable expansion record for this card; every complex gate after it would refuse"
import json, sys
gate, device = (json.load(open(p)) for p in sys.argv[1:3])
cc = device.get("compute_capability")
ok = (gate.get("summary") or {}).get("passed") is True
bases = {k: (v or {}).get("basis") for k, v in (gate.get("unified_licenses") or {}).items()}
measured = bool(bases) and all(b == "measured" for b in bases.values())
print(f"unified_expansion summary.passed={ok}; licence bases={bases}; device.json compute_capability={cc}")
sys.exit(0 if ok and cc and measured else 1)
EOF
  if grep -ls "cannot find -lcuda" "$A_ROOT"/*/gate.log >/dev/null 2>&1; then die "a gate could not link -lcuda: the libcuda stub is not in effect"; fi
  # THE FALLBACK IS SILENT. If 'complex' wrote no probe, cylindrical_fused_electric_pair,
  # cylindrical_fused_magnetic_pair and complex_conductive_fused_pair licensed themselves
  # from the staged A6000 records instead of this card's, and nothing in their verdicts
  # says so. The driver announces the hand-over; its absence taints those three.
  grep -q "\[probe\] later gates will bind the probe this campaign measured" "$L/triton_a.log" \
      || die "the driver never handed this card's measured probe to later gates, so cylindrical_fused_electric_pair, cylindrical_fused_magnetic_pair and complex_conductive_fused_pair licensed from A6000 records. Their results must not be bound. Find why 'complex' wrote no probe, then restart with a new STAMP"
}

stage_triton_b() {
  need triton_a
  MEEP_GPU_COMPLEX_EXPANSION_PROBE=$UNIFIED_KEEP run_driver triton_b.log "${A_ROOT}_env_probe" keep "$B_GATES"
}

stage_triton_c() {
  need triton_a
  run_driver triton_c.log "${A_ROOT}_offdiag_probe" keep complex_offdiag --probe "\"$UNIFIED_KEEP\""
}

stage_triton_d() {
  need triton_a
  local root=$R/triton_composition_$STAMP
  fresh "$root"; mkdir -p "$root"
  # The composition legs record no import set; this manifest is what binds them to bytes.
  ( cd "$REPO" && find meep_gpu parity/meep_gpu -name '*.py' -not -path 'parity/meep_gpu/results/*' \
      -not -path '*/__pycache__/*' -not -name '._*' | LC_ALL=C sort | xargs sha256sum ) > "$root/staged_source_sha256.txt"
  say "staged manifest: $(wc -l < "$root/staged_source_sha256.txt") files"
  local log=triton_d.log; gpu_idle
  say "driver -> $root (5 gates)"
  eval "$DRIVER keep --patience 300 --out-root \"$root\" --gates \"$D_GATES\"" >> "$L/$log" 2>&1
  grep -E "\[gate\]" "$L/$log" | tail -10 >> "$L/round.log"
  [ -f "$root/campaign.json" ] || die "the driver did not write $root/campaign.json; see $L/$log"
}

stage_stencil() {
  need triton_a
  local root=$R/triton_offdiag_stencil_welds_$STAMP fam
  fresh "$root"
  for fam in offdiag_fused_electric_pair folded_offdiag_fused_electric_pair; do
    gpu_idle; say "stencil $fam"
    ( cd "$REPO" && hand_env CUPY_CACHE_DIR="$HC/cupy_cache_ftz_stripped_stencil_$fam" \
        TRITON_CACHE_DIR="$HC/triton_cache_stencil_$fam" \
        "$PY" -u "$G/gate_triton_offdiag_stencil_welds.py" --family "$fam" \
        --subnormal-policy keep --out-root "$root" ) > "$L/stencil_$fam.log" 2>&1
    say "stencil $fam exit=$?"
  done
  ( cd "$REPO" && env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO:$G" \
      "$PY" "$G/triton_device_identity.py" --write "$root" ) >> "$L/stencil_device.log" 2>&1 \
      || die "could not write $root/device.json; the rebind cannot key these runs without it"
}

stage_bitid() {
  need triton_a
  local root=$R/triton_bit_identity_$STAMP
  fresh "$root"; mkdir -p "$root"; gpu_idle
  say "bit identity: 7 experiments"
  ( cd "$REPO" && hand_env CUPY_CACHE_DIR="$HC/cupy_cache_ftz_stripped_bit_identity" \
      TRITON_CACHE_DIR="$HC/triton_cache_bit_identity" \
      "$PY" -u "$G/probe_fused_kernel_bit_identity.py" --module "$G/track_triton_pml.py" \
      --track triton --experiments pml,multistep,constitutive,constitutive_multistep,fused_pair,fused_pair_multistep,wholestep \
      --subnormal-policy keep --out "$root/gate.json" ) > "$L/bit_identity.log" 2>&1
  say "bit identity exit=$?"
  ( cd "$REPO" && env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO:$G" \
      "$PY" "$G/triton_device_identity.py" --write "$root" ) >> "$L/bitid_device.log" 2>&1 \
      || die "could not write $root/device.json; the bit-identity rebind reads it from the artifact's own directory"
}

stage_flushexp() {
  need triton_a
  if [ -f "$L/.no_flush" ]; then say "skipped: this host does not flush subnormals (preflight)"; return; fi
  run_driver flushexp.log "$R/triton_unified_flush_$STAMP" flush unified_expansion
}

stage_cuda() {
  need triton_a; need flushexp
  if [ -f "$L/.no_flush" ]; then say "skipped: every CUDA weld needs its flush legs, and this host does not flush subnormals"; return; fi
  gpu_idle
  REPO=$REPO STAMP=$STAMP GPU=$GPU PY=$PY LANE=$HOME/cuda_lane_$STAMP \
  PROBE_KEEP=$UNIFIED_KEEP PROBE_FLUSH=$R/triton_unified_flush_$STAMP/unified_expansion/gate.json \
    bash "$HERE/run_cuda_welds.sh" >> "$L/cuda.log" 2>&1
  say "cuda exit=$? ($(tail -1 "$HOME/cuda_lane_$STAMP/progress.log" 2>/dev/null))"
  cp "$HOME/cuda_lane_$STAMP/progress.log" "$L/cuda_progress.log" 2>/dev/null
  cp "$HOME/cuda_lane_$STAMP/legs.jsonl" "$L/cuda_legs.jsonl" 2>/dev/null
}

stage_status() {
  "$PY" - "$R" "$STAMP" <<'EOF' | tee "$L/status.txt"
import json, pathlib, sys
R, stamp = pathlib.Path(sys.argv[1]), sys.argv[2]
for root in sorted(R.glob(f"*{stamp}*")):
    camp = root / "campaign.json"
    if camp.is_file():
        rows = json.load(open(camp)).get("rows") or []
        bad = [r.get("gate") or r.get("name") for r in rows if r.get("released") is not True]
        print(f"{root.name}: {len(rows) - len(bad)} of {len(rows)} released" + (f"; NOT: {bad}" if bad else ""))
    elif (root / "gate.json").is_file() or any(root.glob("*/gate.json")):
        n = len(list(root.glob("*/gate.json"))) or 1
        print(f"{root.name}: {n} artifact(s), judge by their canonical_verdict")
legs = R / f"logs_{stamp}" / "cuda_legs.jsonl"
if legs.is_file():
    rows = [json.loads(l) for l in open(legs)]
    ok = sum(1 for r in rows if r["verdict"].startswith("released=True"))
    print(f"cuda_regate_{stamp}: {ok} of {len(rows)} legs released (70 expected)")
    if ok < len(rows):
        print("  NOTE: the CUDA recorder refuses the WHOLE campaign while any leg in it is unreleased. "
              "Move each unreleased leg's subdirectory out of cuda_regate_" + stamp + "/ (keep it, and send it "
              "back) so the rest can be recorded; the welds it belongs to then wait for a re-run.")
EOF
  echo "composition legs cylindrical/symmetry/conductivity always read released=None in campaign.json; judge them by summary.status in gate.json (passed on the A6000 control, 2026-10-02)" | tee -a "$L/status.txt"
}

stage_pack() {
  { date -u +%FT%TZ; git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --porcelain; nvidia-smi; } > "$L/environment_end.txt" 2>&1
  ( cd "$R" && find . -maxdepth 1 -name "*$STAMP*" -print | sed 's|^\./||' | LC_ALL=C sort > "$L/roots.txt" )
  local out=$HOME/meep_gpu_round_$STAMP.tgz
  ( cd "$R" && tar --exclude=caches --exclude=__pycache__ -czf "$out" -T "$L/roots.txt" )
  sha256sum "$out" | tee "$out.sha256"
  say "packed $(wc -l < "$L/roots.txt") roots -> $out ($(du -h "$out" | cut -f1)); send it and its .sha256 back"
}

ORDER="preflight inputs triton_a triton_b triton_c triton_d stencil bitid flushexp cuda status pack"
run_stage() { STAGE_NOW=$1; T_STAGE=$(date +%s); say "start"; "stage_$1"; done_mark "$1"; }
if [ "$STAGE" = all ]; then
  for s in $ORDER; do [ -f "$L/.done_$s" ] && { STAGE_NOW=$s; say "already finished; skipping"; continue; }; run_stage "$s"; done
else
  case " $ORDER " in *" $STAGE "*) run_stage "$STAGE";; *) echo "unknown stage $STAGE; one of: $ORDER all"; exit 2;; esac
fi
