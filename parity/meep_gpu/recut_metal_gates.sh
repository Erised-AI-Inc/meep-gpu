#!/bin/zsh
# Re-cut EVERY Metal gate on one tree, in one pass, and record what each returned.
#
# WHY A SCRIPT RATHER THAN A LIST IN A REPORT. "Every gate was re-cut" is a claim
# about all discovered processes; running them by hand is how one gets missed and the
# report still says fifteen. This enumerates `gate_metal_*.py` from the DIRECTORY
# rather than from a literal list, so a gate added later is re-cut by the same
# command without anyone remembering to add it — and a gate that vanishes shows up
# as a shorter run rather than as a silently unchecked one.
#
# ORDER MATTERS AT TWO POINTS. Inside this file: every expansion probe a gate binds is
# cut before the first gate that binds it -- the three standalone probes, then
# `gate_metal_complex`, whose expansion leg cuts the fourth, and only then the rest
# (see below). Outside it: `write_fingerprints` must run AFTER the last gate,
# deliberately and separately, because a fingerprint cut before the gates that check
# it certifies nothing. This script never calls it.
#
# progress reporting: one flushed line per gate as it lands, plus a per-gate log, so the state
# of a fifteen-gate run is readable from the filesystem while it is still running.
#
# THE OPTIONAL SECOND ARGUMENT IS A REBIND STAMP, and it exists because the artifact
# LAYOUT is a contract rather than a convenience. `rebind_metal_welds.py` resolves a
# family's fresh artifact as `results/metal_<family>_<stamp>/`, and this script's own
# default layout — `<root>/gate_metal_<family>/` — is not that. A campaign that re-cut
# every gate and then could not rebind the ledger has done half the work, so the
# layout the rebind reads is producible from here rather than from a one-off loop
# somebody writes beside it. With a stamp, each gate lands in
# `results/metal_<family>_<stamp>/gate.json` and `<root>` keeps only the logs.
#
# Usage (from the repository root):
#   zsh parity/meep_gpu/recut_metal_gates.sh <artifact-root> [rebind-stamp]
set -u
API=${MGPU_SITE_SOURCE_ROOT:-${0:A:h:h:h}}
PY=${MGPU_SITE_PYTHON:-python}
ROOT=${1:?usage: recut_metal_gates.sh <artifact-root> [rebind-stamp]}
STAMP=${2:-}
RESULTS="$API/parity/meep_gpu/results"
cd "$API" || exit 1

export KMP_DUPLICATE_LIB_OK=TRUE
export MPLBACKEND=Agg
export PYTHONPATH="$API"
export MEEP_GPU_SUBNORMAL_POLICY=flush
# THE DISPATCH PIN (2026-09-27): fastpath.DISPATCH_BY_DEFAULT is True, so an unset
# MEEP_GPU_DISPATCH now dispatches, and the gates here step NumPy reference drivers
# that must stay on the array path -- on this Mac an unpinned one plans through the
# Metal table. Exported here, beside the policy, so the fleet is pinned whoever calls
# it (metal_sparse_round.sh stage 1 also prefixes it). A gate that measures dispatch
# sets the enable itself, in-process, per leg. Exactly 0 -- any other value is refused
# by name.
export MEEP_GPU_DISPATCH=0
# NO PROBE FROM THE CALLING SHELL. Every expansion probe a gate binds is cut by this run
# and exported by it below; a value inherited from the shell would license this host's
# arms from a record another run, or another machine, measured.
unset MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE MEEP_GPU_METAL_EXPANSION_PROBE \
      MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE \
      MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE

mkdir -p "$ROOT"
export MEEP_GPU_GATE_SOURCE_MANIFEST="$ROOT/source_sha256.txt"
SUMMARY="$ROOT/recut_summary.txt"
: > "$SUMMARY"

fail=0
# THE ENVIRONMENT THE WELDS WILL NAME, recorded before the first gate and after the
# last: the GPU architecture, torch and Metal frontend `metal_dispatch` certifies a
# weld for. `rebind_metal_welds.py` writes each weld's host line from these two
# records and refuses a campaign whose records are missing or disagree.
if [[ -n "$STAMP" ]]; then
  ENVIRONMENT="$RESULTS/metal_environment_$STAMP"
  # Appended rather than piped through tee: zsh reports a pipeline's LAST status,
  # so a refusal piped through tee would read as success.
  if ! $PY "$API/parity/meep_gpu/metal_environment.py" --write "$ENVIRONMENT/start.json" \
      >> "$SUMMARY" 2>&1; then
    print -r -- "VERDICT: COULD NOT RECORD THE METAL ENVIRONMENT" | tee -a "$SUMMARY"
    exit 1
  fi
fi
# Four complex families are licensed by their own MPS expansion measurements.  The
# gate sources used to search an ambient results directory, which made a green gate
# dependent on whichever old artifact happened to be present.  Cut the three
# standalone records first, source-weld them through the same runner, and bind them
# explicitly for every later child.  The base complex gate creates its own probe in
# its expansion leg, is welded immediately after that gate returns, and RUNS BEFORE
# EVERY OTHER GATE: several earlier-sorting gates bind it too (the refusal leg of
# `gate_metal_beta_complex_fused_hd_pair` asks the plain complex family to admit, which
# needs the probe). Run in directory order they ran without it, and on a Mac holding
# the evidence archive `metal_composition_matrix.prepare_environment` silently filled
# the gap with a dated archive probe measured on another run.
PROBE_ROOT="$ROOT/probes"
mkdir -p "$PROBE_ROOT"

run_probe() {
  script="$1"
  artifact="$2"
  label=$(basename "$script" .py)
  log="$ROOT/$label.log"
  started=$(date +%s)
  $PY -u "$API/parity/meep_gpu/metal_gate_runner.py" \
    "$API/parity/meep_gpu/$script" -- --out "$artifact" > "$log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - started ))
  line=$(printf "%-36s exit=%d  %4ds  %s" "$label" "$rc" "$elapsed" "$log")
  print -r -- "$line" | tee -a "$SUMMARY"
  [[ $rc -ne 0 ]] && fail=1
}

run_probe "probe_metal_beta_expansion.py" "$PROBE_ROOT/special_kz.json"
run_probe "probe_metal_folded_complex_expansion.py" "$PROBE_ROOT/folded_complex.json"
run_probe "probe_metal_cylindrical_complex.py" "$PROBE_ROOT/cylindrical_complex.json"

if [[ $fail -ne 0 ]]; then
  print -r -- "---" | tee -a "$SUMMARY"
  print -r -- "VERDICT: EXPANSION PROBE RECUT FAILED; NO GATE WAS RUN" | tee -a "$SUMMARY"
  exit 1
fi

export MEEP_GPU_METAL_EXPANSION_PROBE="$PROBE_ROOT/special_kz.json"
export MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE="$PROBE_ROOT/folded_complex.json"
export MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE="$PROBE_ROOT/cylindrical_complex.json"
# The base complex gate writes its expansion probe BESIDE its own artifact, so
# this path follows the same layout switch the loop below applies.
if [[ -n "$STAMP" ]]; then
  COMPLEX_PROBE="$RESULTS/metal_complex_$STAMP/complex_expansion_probe.json"
else
  COMPLEX_PROBE="$ROOT/gate_metal_complex/complex_expansion_probe.json"
fi

run_gate() {
  gate="$1"
  name=$(basename "$gate" .py)
  log="$ROOT/$name.log"
  started=$(date +%s)
  if [[ -n "$STAMP" ]]; then
    # THE REBIND LAYOUT: results/metal_<family>_<stamp>/, which is what
    # `rebind_metal_welds._artifact_for` looks for. `${name#gate_}` is the family
    # prefixed with `metal_`, which is exactly how the ledger keys are spelled.
    dir="$RESULTS/${name#gate_}_$STAMP"
  else
    dir="$ROOT/$name"
  fi
  mkdir -p "$dir"
  if [[ "$name" == "gate_metal_whole_step" ]]; then
    # The one gate whose --out is a DIRECTORY rather than an artifact path.
    out="$dir"
  else
    out="$dir/gate.json"
  fi
  $PY -u "$gate" --out "$out" > "$log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - started ))
  # `rc`, not `status`: `status` is READ-ONLY in zsh and assigning it aborts
  # the script at the first gate — which this script did, silently, until the
  # empty summary file said so.
  line=$(printf "%-36s exit=%d  %4ds  %s" "$name" "$rc" "$elapsed" "$log")
  print -r -- "$line" | tee -a "$SUMMARY"
  [[ $rc -ne 0 ]] && fail=1
  if [[ "$name" == "gate_metal_complex" && $rc -eq 0 ]]; then
    probe_log="$ROOT/gate_metal_complex_probe.log"
    probe_started=$(date +%s)
    $PY -u "$API/parity/meep_gpu/metal_gate_runner.py" \
      --weld-related "$COMPLEX_PROBE" --parent-artifact "$out" > "$probe_log" 2>&1
    probe_rc=$?
    probe_elapsed=$(( $(date +%s) - probe_started ))
    probe_line=$(printf "%-36s exit=%d  %4ds  %s" "complex expansion probe" \
      "$probe_rc" "$probe_elapsed" "$probe_log")
    print -r -- "$probe_line" | tee -a "$SUMMARY"
    [[ $probe_rc -ne 0 ]] && fail=1
  fi
}

# The base complex gate first: its expansion leg writes the probe every complex-family
# gate binds. Once present it is an explicit input for all of them; no source is allowed
# to fall back to a glob of historic results.
run_gate "$API/parity/meep_gpu/gate_metal_complex.py"
if [[ -f "$COMPLEX_PROBE" ]]; then
  export MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE="$COMPLEX_PROBE"
else
  fail=1
  print -r -- "$(printf "%-36s MISSING  %s" "complex expansion probe" "$COMPLEX_PROBE")" \
    | tee -a "$SUMMARY"
  print -r -- "  every gate that binds the complex probe will refuse by name; none falls back" \
    | tee -a "$SUMMARY"
fi

for gate in "$API"/parity/meep_gpu/gate_metal_*.py; do
  [[ "$(basename "$gate")" == "gate_metal_complex.py" ]] && continue
  run_gate "$gate"
done

if [[ -n "$STAMP" ]]; then
  $PY "$API/parity/meep_gpu/metal_environment.py" --write "$ENVIRONMENT/end.json" \
    >> "$SUMMARY" 2>&1 || fail=1
  $PY "$API/parity/meep_gpu/metal_environment.py" --check "$STAMP" \
    >> "$SUMMARY" 2>&1 || fail=1
  tail -1 "$SUMMARY"
fi
print -r -- "---" | tee -a "$SUMMARY"
print -r -- "VERDICT: $([[ $fail -eq 0 ]] && echo ALL GREEN || echo 'AT LEAST ONE GATE FAILED')" \
  | tee -a "$SUMMARY"
exit $fail
