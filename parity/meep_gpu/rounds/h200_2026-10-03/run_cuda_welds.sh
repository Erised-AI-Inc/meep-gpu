#!/usr/bin/env bash
# Every CUDA weld gate the 31 cited CUDA welds need, keep + flush, sequentially on ONE GPU,
# into ONE re-gate campaign laid out the way record_cuda_regate.py / rebind_cuda_welds.py read it:
#   $REPO/parity/meep_gpu/results/cuda_regate_$STAMP/<block artifact basename>/<leg>
# 35 gate pairs = 70 legs (68 for the 31 cited welds + the 2 no_pml_complex_electric legs
# record_cuda_regate.py requires because a third block claims
# cuda_fused_complex_pairs_2026-09-11_regate/). Every complex leg is handed THIS card's
# own expansion record (PROBE_KEEP / PROBE_FLUSH), including the stencil-weld pair,
# whose gate now refuses a record measured on another compute capability.
#
# usage:
#   REPO=/abs/meep-gpu STAMP=2026-10-01_h100 GPU=0 PY=/abs/env/bin/python \
#   PROBE_KEEP=/abs/fleet_keep/unified_expansion/gate.json \
#   PROBE_FLUSH=/abs/fleet_flush/unified_expansion/gate.json \
#   bash run_cuda_welds.sh
#   DRY_RUN=1 ... prints every leg's argv and runs nothing.
#
# Progress: one flushed line per leg in $LANE/progress.log, one JSON row per leg in
# $LANE/legs.jsonl. Logs and caches live in $LANE, never inside the campaign directory.
set -u
REPO=${REPO:?set REPO to the checkout of the pushed commit}
STAMP=${STAMP:?set STAMP, e.g. 2026-10-01_h100}
GPU=${GPU:-0}
PY=${PY:-python}
PROBE_KEEP=${PROBE_KEEP:?absolute path to a keep-cut unified_expansion/gate.json}
PROBE_FLUSH=${PROBE_FLUSH:?absolute path to a flush-cut unified_expansion/gate.json}
DRY_RUN=${DRY_RUN:-0}
G=$REPO/parity/meep_gpu
CAMPAIGN=$G/results/cuda_regate_$STAMP
LANE=${LANE:-$HOME/cuda_lane_$STAMP}
case "$LANE" in *ftz_stripped*) echo "LANE must not contain ftz_stripped (flush caches live under it)"; exit 2;; esac
case "$REPO$LANE" in *" "*) echo "REPO and LANE must not contain spaces"; exit 2;; esac
case "$PROBE_KEEP$PROBE_FLUSH" in /*/*) ;; *) echo "PROBE_KEEP/PROBE_FLUSH must be absolute"; exit 2;; esac

export CUDA_VISIBLE_DEVICES=$GPU
export CUPY_ACCELERATORS=                     # literally empty, before any CuPy import
export MEEP_GPU_DISPATCH=0                    # reference drivers must stay on the array path
export MEEP_GPU_BACKEND_PREFERENCE=cuda       # as the 8.6 CUDA lane ran
unset MEEP_GPU_SUBNORMAL_POLICY MEEP_GPU_COMPLEX_EXPANSION_PROBE MEEP_GPU_ALLOW_UNCERTIFIED || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO:$G
if [ -n "${TRITON_LIBCUDA_PATH:-}" ]; then export LD_LIBRARY_PATH=$TRITON_LIBCUDA_PATH:${LD_LIBRARY_PATH:-}; fi

mkdir -p "$LANE/logs" "$LANE/caches"
LOG=$LANE/progress.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# ---- preflight ----------------------------------------------------------------------------
say "START repo=$REPO head=$(git -C "$REPO" rev-parse HEAD) campaign=$CAMPAIGN gpu=$GPU"
git -C "$REPO" status --porcelain > "$LANE/git_status_porcelain.txt"
say "git status --porcelain: $(grep -c . "$LANE/git_status_porcelain.txt") line(s) (must be 0 on the pushed commit)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.used --format=csv,noheader | tee -a "$LOG"
check_probe() {  # <path> <policy>: the record the 36 expansion-consuming legs will read
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
path, policy = sys.argv[1], sys.argv[2]
d = json.load(open(path))
problems = []
if d.get("backend") != "cupy": problems.append(f"backend={d.get('backend')!r}")
if (d.get("subnormal_policy") or {}).get("resolved") != policy:
    problems.append(f"subnormal_policy.resolved={(d.get('subnormal_policy') or {}).get('resolved')!r}")
if (d.get("summary") or {}).get("passed") is not True:
    problems.append(f"summary.passed={(d.get('summary') or {}).get('passed')!r}")
print(("OK " if not problems else "BAD ") + path + " " + "; ".join(problems))
sys.exit(1 if problems else 0)
EOF
}
if [ "$DRY_RUN" != 1 ]; then
  r=$(check_probe "$PROBE_KEEP" keep); rc=$?; say "$r"
  [ "$rc" = 0 ] || { say "ABORT: keep expansion record unusable"; exit 3; }
  r=$(check_probe "$PROBE_FLUSH" flush); rc=$?; say "$r"
  [ "$rc" = 0 ] || { say "ABORT: flush expansion record unusable"; exit 3; }
fi
# ---- the leg table ----------------------------------------------------------------------
LEGS=()
pair() {  # <subdir> <leg template, @P@ = policy> <file|dir> <0|probe> <script> [args...]
  local sub=$1 tmpl=$2 kind=$3 exp=$4 script=$5 p; shift 5
  for p in keep flush; do LEGS+=("$sub|${tmpl//@P@/$p}|$p|$kind|$exp|$script|$*"); done
}
CURL=$REPO/meep_gpu/cuda_kernels/step_curl_kernels.py
pair fused_pml_bit_identity_hand_2026-08-22_rename @P@/probe.json file 0 probe_fused_kernel_bit_identity.py --module "$CURL" --track hand --experiments pml,multistep --coefficients synthetic,real
pair cuda_constitutive_recut_2026-08-22_rename @P@/cgate.json file 0 probe_fused_kernel_bit_identity.py --module "$CURL" --track hand --experiments constitutive,constitutive_multistep --coefficients synthetic,real
pair cuda_offdiag_2026-08-21_rename @P@/gate.json file 0 gate_cuda_offdiag.py --backend cupy --product full
pair cuda_dispersive_offdiag_2026-08-20b @P@/gate.json file 0 gate_cuda_dispersive_offdiag.py --backend cupy --product full
pair cuda_folded_offdiag_kernel_2026-08-21_rename @P@/gate.json file 0 gate_cuda_folded_offdiag_kernel.py --backend cupy --product full
pair cuda_in_seam_2026-08-21_final fill_folded_far/@P@/gate.json file 0 gate_cuda_in_seam_passes.py --pass fill_folded_far --backend cuda
pair cuda_in_seam_2026-08-21_final fill_symmetry/@P@/gate.json file 0 gate_cuda_in_seam_passes.py --pass fill_symmetry --backend cuda
pair cuda_in_seam_2026-08-21_final zero_metal/@P@/gate.json file 0 gate_cuda_in_seam_passes.py --pass zero_metal --backend cuda
pair cuda_fused_magnetic_pair_2026-08-27 @P@/gate.json file 0 gate_cuda_fused_magnetic_pair.py
pair cuda_fused_electric_pair_2026-08-31_fillcarry_r3 @P@/gate.json file 0 gate_cuda_fused_electric_pair.py
pair cuda_fused_polarization_pair_2026-09-01_pml @P@/gate.json file 0 gate_cuda_fused_polarization_pair.py --arm pml --product full
pair cuda_fused_polarization_pair_2026-09-01_no_pml @P@/gate.json file 0 gate_cuda_fused_polarization_pair.py --arm no_pml --product full
pair cuda_conductive_fused_electric_pair_2026-09-02b @P@/gate.json file 0 gate_cuda_conductive_fused_electric_pair.py
pair cuda_offdiag_stencil_welds_2026-09-02 @P@/gate.json file 0 gate_cuda_offdiag_stencil_welds.py --product full --steps 60 --repeats 4
pair cuda_dispersive_offdiag_polarization_pair_2026-09-02 @P@/gate.json file 0 gate_cuda_dispersive_offdiag_polarization_pair.py --product full --steps 60 --repeats 4
pair cuda_three_slot_weld_2026-09-11_regate @P@_pml/gate.json dir 0 gate_cuda_three_slot_weld.py --arms pml
pair cuda_three_slot_weld_2026-09-11_regate @P@_no_pml/gate.json dir 0 gate_cuda_three_slot_weld.py --arms no_pml
pair cuda_residue_complex_polarization_pair_2026-09-02b @P@/gate.json file probe gate_cuda_complex_polarization_pair.py --product full
pair cuda_complex_three_slot_weld_2026-09-04_gatefix @P@/gate.json file probe gate_cuda_complex_three_slot_weld.py
pair cuda_fused_complex_pairs_2026-09-11_regate complex_@P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family complex
pair cuda_fused_complex_pairs_2026-09-11_regate complex_electric_@P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family complex_electric
pair cuda_fused_complex_pairs_2026-09-11_regate no_pml_complex_electric_@P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family no_pml_complex_electric
pair cuda_fused_complex_pairs_2026-09-13_cyl cylindrical_@P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family cylindrical
pair cuda_fused_complex_pairs_2026-09-13_cyl cylindrical_electric_@P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family cylindrical_electric
for fam in bfast complex_beta complex_folded cylindrical_real special_kz; do
  pair cuda_fused_residual_${fam}_2026-09-02 @P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family $fam --product full
done
for fam in bfast complex_beta cylindrical_real special_kz; do
  pair cuda_residue_${fam}_electric_2026-09-02 @P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family ${fam}_electric --product full
done
pair cuda_residue_folded_complex_electric_2026-09-02 @P@/gate.json file probe gate_cuda_fused_complex_pairs.py --family folded_complex_electric --product full
pair cuda_complex_offdiag_stencil_welds_2026-09-17 @P@/gate.json file probe gate_cuda_complex_offdiag_stencil_welds.py
N=${#LEGS[@]}
say "legs: $N (expect 70); 8.6 gate time for these legs was ~5964 s on an A6000"

verdict() {  # canonical_verdict.released of one payload, as gate_provenance.stamp wrote it
  "$PY" -c 'import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("none(%s)" % type(e).__name__); raise SystemExit
v=d.get("canonical_verdict") or {}
print("released=%s reasons=%d" % (v.get("released"), len(v.get("reasons") or [])))' "$1" 2>/dev/null || echo "unreadable"
}

# ---- run --------------------------------------------------------------------------------
[ "$DRY_RUN" = 1 ] || mkdir -p "$CAMPAIGN"
T0=$(date +%s); i=0; n_rel=0; n_bad=0; n_skip=0
for spec in "${LEGS[@]}"; do
  i=$((i+1))
  IFS='|' read -r sub leg policy kind exp script argstr <<< "$spec"
  read -r -a args <<< "$argstr"
  art=$CAMPAIGN/$sub/$leg; legdir=$(dirname "$art")
  if [ "$kind" = dir ]; then out=$legdir; else out=$art; fi
  pol=(--subnormal-policy "$policy"); [ "$policy" = flush ] && pol+=(--import-meep-for-host-policy)
  probe=()
  if [ "$exp" = probe ]; then
    if [ "$policy" = keep ]; then probe=(--expansion-probe "$PROBE_KEEP"); else probe=(--expansion-probe "$PROBE_FLUSH"); fi
  fi
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY [$i/$N] $sub|$leg|$policy|$script ${args[*]:-} ${pol[*]} ${probe[*]:-} --out $out"; continue
  fi
  if [ -f "$art" ] && verdict "$art" | grep -q '^released=True'; then
    n_skip=$((n_skip+1)); say "[$i/$N] SKIP already released $sub/$leg"; continue
  fi
  if [ -d "$legdir" ] && [ -n "$(find "$legdir" -maxdepth 1 -name '*.json' -print -quit)" ]; then
    n_bad=$((n_bad+1)); say "[$i/$N] REFUSED $sub/$leg: $legdir already holds a payload ($(verdict "$art")); move it OUT of $CAMPAIGN by hand (keep it) and re-run"; continue
  fi
  mkdir -p "$legdir"
  tag=$(echo "${sub}__${leg%/*}" | tr '/' '_')_$(date +%s)
  if [ "$policy" = keep ]; then cache=$LANE/caches/cupy_ftz_stripped_$tag; else cache=$LANE/caches/cupy_plain_$tag; fi
  mkdir -p "$cache" "$LANE/caches/triton_$tag"
  logf=$LANE/logs/$tag.log
  s=$(date +%s)
  ( cd "$REPO" && CUPY_CACHE_DIR=$cache TRITON_CACHE_DIR=$LANE/caches/triton_$tag \
      "$PY" -u "$G/$script" ${args[@]+"${args[@]}"} "${pol[@]}" ${probe[@]+"${probe[@]}"} --out "$out" ) > "$logf" 2>&1
  rc=$?; dt=$(( $(date +%s) - s ))
  cp "$logf" "$legdir/run.log"
  v=$(verdict "$art")
  case "$v" in released=True*) n_rel=$((n_rel+1));; *) n_bad=$((n_bad+1));; esac
  say "[$i/$N] $sub/$leg $policy rc=$rc ${dt}s $v | elapsed $(( ($(date +%s)-T0)/60 )) min"
  printf '{"utc":"%s","idx":%d,"subdir":"%s","leg":"%s","policy":"%s","rc":%d,"seconds":%d,"verdict":"%s","log":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$i" "$sub" "$leg" "$policy" "$rc" "$dt" "$v" "$logf" >> "$LANE/legs.jsonl"
done
say "DONE released=$n_rel not_released_or_refused=$n_bad skipped=$n_skip of $N -> $CAMPAIGN"
exit $(( n_bad > 0 ))
