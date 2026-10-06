#!/bin/zsh
# A meep-gpu Metal certification round on ONE Mac: every Metal gate, written where the
# rebind on the host that holds the evidence archive reads it. It runs gates only; it
# writes no ledger. Run it from anywhere, one stage at a time or all of them:
#
#   conda activate meep-gpu-ref      # built by parity/meep_gpu/build_meep_133_macos.sh
#   export REPO=$HOME/meep-gpu  EXPECT=<full commit hash>  STAMP=2026-10-05_g15s
#   export INPUTS=$HOME/metal_round_inputs.tgz      # leave unset on the archive host
#   export MGPU_SITE_MEEP_SOURCE=$HOME/meep         # MEEP 1.33.0 source: python/examples, python/tests
#   zsh run_round.sh all > $HOME/metal_round_$STAMP.out 2>&1 &
#   tail -f $HOME/metal_round_$STAMP.out           # one flushed line per stage and per gate
#
# Stages, in order (each refuses to start if the one before it did not finish):
#   preflight  commit, clean checkout, reference MEEP, Metal environment, idle GPU
#   inputs     unpack and verify the inputs bundle (INPUTS=<.tgz>), or, with INPUTS
#              unset, verify the evidence archive in place; then check that the lift
#              legs can run here (MEEP corpus, census interpreters)
#   fleet      re-checks the commit, the clean checkout and the idle GPU, then runs
#              recut_metal_gates.sh with the stamp: three probes, gate_metal_complex,
#              then every other gate (about 70 min on an M1 Max)
#   status     what released; how many of the welds the Metal table cites released;
#              any artifact that names a dated archive probe
#   pack       re-checks the commit and the clean checkout, then writes one tarball of
#              the round's outputs and logs, for the rebind host
#
# `all` skips a stage that already finished, so a resumed run starts at the first
# unfinished one; fleet and pack re-check the tree themselves for that reason. The
# manifest is read from $REPO, the checkout being certified, never from wherever this
# script was copied to.
#
# ALLOW_NO_LIFT=1 lets the round go on when the lift legs cannot run here (no MEEP
# corpus, or an interpreter the census rows name is absent). The gates whose lift leg
# re-lifts corpus rows then cannot release on this Mac; none of them is cited by the
# Metal table (README.md).
#
# Everything lands under $REPO/parity/meep_gpu/results/ (gitignored), which is where the
# rebind requires it. Every output is new: a stage refuses a stamp that already has one.
set -u
setopt pipefail
STAGE=${1:?usage: run_round.sh <stage|all>}
REPO=${REPO:?set REPO to the clone of the commit being certified}
REPO=$(cd "$REPO" 2>/dev/null && pwd -P) || { print -r -- "REPO is not a directory"; exit 2; }
EXPECT=${EXPECT:?set EXPECT to the full commit hash being certified}
STAMP=${STAMP:?set STAMP, e.g. 2026-10-05_g15s (it must differ from every earlier Metal stamp)}
PY=${PY:-$(command -v python)}
G=$REPO/parity/meep_gpu
ROUND=$G/rounds/metal_2026-10-04
R=$G/results
L=$R/logs_metal_$STAMP
ROOT=$R/metal_campaign_$STAMP/fleet
# BEFORE THE FIRST WRITE: the log directory lives under results/, so a results/ that is
# a symbolic link would receive it in whatever the link points at.
[[ ! -L "$R" ]] || { print -r -- "STOP: $R is a symbolic link; the round's outputs must not land in whatever it points at"; exit 3; }
[[ -f "$ROUND/INPUT_FILES.txt" ]] || { print -r -- "STOP: $ROUND holds no manifest; REPO is not a checkout of this round's commit"; exit 3; }
mkdir -p "$L"
export PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE MPLBACKEND=Agg
unset MEEP_GPU_DISPATCH

say()  { print -r -- "$(date -u +%FT%TZ) [$STAGE_NOW] $*" | tee -a "$L/round.log"; }
die()  { say "STOP: $*"; exit 3; }
done_mark() { touch "$L/.done_$1"; say "finished in $(( $(date +%s) - T_STAGE ))s"; }
need() { [[ -f "$L/.done_$1" ]] || die "stage '$1' has not finished; run it first"; }
gpu_idle() {
  local busy
  busy=$(ps -axo pid=,command= | grep -E 'python[0-9.]* .*(gate_|bench_|probe_)' | grep -v grep)
  [[ -z "$busy" ]] || die "another gate, probe or benchmark is running on this Mac: ${busy%%$'\n'*}"
}
py() { ( cd "$REPO" && PYTHONPATH="$REPO:$G" "$PY" "$@" ) }
# The commit and the clean checkout, asserted by every stage that runs or packs gates:
# `all` skips a finished preflight, so a resumed run would otherwise reach the fleet
# on whatever the checkout became in between.
tree_is_the_commit() {
  local head
  head=$(git -C "$REPO" rev-parse HEAD) || die "$REPO is not a git checkout"
  [[ "$head" == "$EXPECT" ]] || die "HEAD is $head, not $EXPECT: every Mac in a round runs the SAME commit"
  [[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "the checkout has local changes; the rebind compares every digest to the commit"
}

stage_preflight() {
  tree_is_the_commit
  local v
  for v in MEEP_GPU_METAL_RESIDENCY MEEP_GPU_ALLOW_UNCERTIFIED MEEP_GPU_SUBNORMAL_POLICY \
           MEEP_GPU_KERNEL_TABLE MEEP_GPU_CORPUS_ROOT MEEP_GPU_H_TO_D_PROBE \
           MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE MEEP_GPU_METAL_EXPANSION_PROBE \
           MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE \
           MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE; do
    [[ -n "${(P)v+set}" ]] && die "$v is set in the calling shell; the fleet sets what it needs per process"
  done
  [[ "${PYTORCH_MPS_FAST_MATH:-0}" == 0 ]] || die "PYTORCH_MPS_FAST_MATH=$PYTORCH_MPS_FAST_MATH: no weld ran fast-math sources"
  py - "$REPO" <<'EOF' | tee -a "$L/round.log" || die "the package import check failed (above)"
import os, sys
repo = os.path.realpath(sys.argv[1])
import meep_gpu
where = os.path.realpath(meep_gpu.__file__)
print("meep_gpu from the clone:", where.startswith(repo + os.sep))
if not where.startswith(repo + os.sep):
    print("REFUSED: an installed meep_gpu shadows the clone"); sys.exit(1)
EOF
  # The documented check a host joining a round runs before its first gate: MEEP 1.33.0,
  # single precision, an MPI build, and every numerics-relevant package the lock's.
  py tools/check_install.py --require-reference > "$L/check_install.out" 2>&1 \
      || die "tools/check_install.py --require-reference failed: $(tail -3 "$L/check_install.out" | tr '\n' ' ')"
  say "reference environment: $(tail -1 "$L/check_install.out")"
  local env_now
  env_now=$(py - <<'EOF'
import metal_environment as m
e = m.read()
print(f"{e['architecture']}|{e['torch']}|{e['metal_frontend']}|{e['device_name']}|{e['macos']} {e['macos_build']}")
EOF
) || die "could not read the Metal environment"
  say "Metal environment: $env_now"
  [[ "$env_now" != None\|* && "$env_now" != *\|None\|* ]] \
      || die "a Metal environment fact could not be read here; no weld could name it"
  local taken
  taken=$(cd "$R" && ls -d metal_*_"$STAMP" 2>/dev/null | head -3)
  [[ -z "$taken" ]] || die "outputs of stamp $STAMP already exist ($taken); a gate re-run into an old root mixes two runs. Use a new STAMP"
  gpu_idle
  { date -u +%FT%TZ; git -C "$REPO" rev-parse HEAD; sw_vers; "$PY" -m pip freeze; } > "$L/environment_start.txt" 2>&1
  say "commit $EXPECT, clean, reference MEEP, environment read, no gate running"
}

stage_inputs() {
  need preflight
  if [[ -n "${INPUTS:-}" ]]; then
    py "$G/build_metal_round_inputs.py" unpack --round "$ROUND" --results "$R" --bundle "$INPUTS" \
        2>&1 | tee -a "$L/round.log" || die "the inputs bundle did not unpack and verify (above)"
  else
    py "$G/build_metal_round_inputs.py" check --round "$ROUND" --results "$R" \
        2>&1 | tee -a "$L/round.log" \
        || die "the archive inputs are absent or differ; set INPUTS to metal_round_inputs.tgz"
  fi
  cp "$ROUND/INPUTS_SHA256SUMS" "$L/"
  # THE LIFT LEGS RE-RUN CORPUS ROWS, each in the interpreter its census row names. Both
  # lift variants catch only a timeout, so an absent interpreter crashes the gate rather
  # than failing a leg; it is checked here, by name, before the first gate.
  local lift
  lift=$(py - "$R" "${MGPU_SITE_MEEP_SOURCE:-}" <<'EOF'
import json, os, sys
results, corpus = sys.argv[1], sys.argv[2]
seen = set()
for name in ("metal_coverage_2026-09-04_m0complex/examples.jsonl",
             "metal_coverage_2026-09-04_m0complex/tests.jsonl",
             "metal_coverage_2026-09-04_m0complex/tests_param_matched.jsonl",
             "h_to_d_seam_2026-09-04/h_to_d_seam.jsonl"):
    with open(os.path.join(results, name), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                for key in ("interpreter", "python"):
                    if row.get(key):
                        seen.add(row[key])
missing = sorted(path for path in seen if not os.access(path, os.X_OK))
problems = [f"interpreter named by census rows is absent here: {path}" for path in missing]
if not (corpus and os.path.isdir(os.path.join(corpus, "python", "examples"))
        and os.path.isdir(os.path.join(corpus, "python", "tests"))):
    problems.append("MGPU_SITE_MEEP_SOURCE does not hold python/examples and python/tests")
print(f"{len(seen) - len(missing)} of {len(seen)} census interpreters present")
for problem in problems:
    print("PROBLEM:", problem)
EOF
) || die "could not read the census interpreters"
  print -r -- "$lift" | tee -a "$L/round.log"
  if [[ "$lift" == *PROBLEM:* ]]; then
    if [[ "${ALLOW_NO_LIFT:-0}" == 1 ]]; then
      touch "$L/.no_lift"
      say "WARNING: going on under ALLOW_NO_LIFT=1; the gates whose lift leg re-lifts corpus rows cannot release on this Mac (none is cited by the Metal table)"
    else
      die "the lift legs cannot run here (above). Fix it, or re-run this stage with ALLOW_NO_LIFT=1"
    fi
  fi
}

stage_fleet() {
  need inputs
  tree_is_the_commit
  gpu_idle
  [[ ! -e "$ROOT" ]] || die "$ROOT exists"
  mkdir -p "$ROOT"
  say "fleet -> $ROOT (MEEP_GPU_DISPATCH=0, residency shipped)"
  ( cd "$REPO" && env MEEP_GPU_DISPATCH=0 MEEP_GPU_METAL_RESIDENCY=shipped \
      MGPU_SITE_PYTHON="$PY" MGPU_SITE_SOURCE_ROOT="$REPO" \
      caffeinate -i zsh "$G/recut_metal_gates.sh" "$ROOT" "$STAMP" ) >> "$L/fleet.log" 2>&1
  local rc=$?
  tail -4 "$ROOT/recut_summary.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$L/round.log"
  say "fleet exit=$rc (the status stage says what released)"
  [[ -f "$ROOT/recut_summary.txt" ]] || die "the fleet wrote no summary; see $L/fleet.log"
}

stage_status() {
  need fleet
  py - "$R" "$STAMP" "$L" "$G" <<'EOF' | tee "$L/status.txt"
import json, os, pathlib, re, sys
from meep_gpu import metal_dispatch
R, stamp, logs, harness = (pathlib.Path(sys.argv[1]), sys.argv[2],
                           pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
ARCHIVE_PROBE = re.compile(
    r"metal_(complex_audit|complex|special_kz|folded_complex|cylindrical_complex)_2026-08-1\d")

def verdict(artifact):
    stated = {name: (artifact.get(name) or {}).get("released")
              for name in ("canonical_verdict", "release")}
    stated = {k: v for k, v in stated.items() if v is not None}
    return bool(stated) and all(stated.values())

def strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value)
    elif isinstance(node, str):
        yield node

released, refused, archive_named = [], [], []
# ONE ROW PER GATE SCRIPT, as the fleet enumerates them, so a gate that wrote nothing
# is counted rather than missed.
for script in sorted(harness.glob("gate_metal_*.py")):
    family = script.stem[len("gate_metal_"):]
    directory = R / f"metal_{family}_{stamp}"
    artifact = next((directory / n for n in ("gate.json", f"{family}.json", "whole_step.json")
                     if (directory / n).is_file()), None)
    if artifact is None:
        refused.append((family, "no artifact"))
        continue
    record = json.loads(artifact.read_text(encoding="utf-8"))
    if verdict(record):
        released.append(family)
    else:
        reasons = (record.get("release") or {}).get("reasons") or "no reason given"
        refused.append((family, str(reasons)[:160]))
    if any(ARCHIVE_PROBE.search(text) for text in strings(record)):
        archive_named.append(family)
total = len(released) + len(refused)
print(f"{len(released)} of {total} gate artifacts released")
for family, why in refused:
    print(f"  NOT released: {family}: {why}")
cited = sorted({gate for _family, gate in metal_dispatch.ARM_CERTIFICATION.values()})
cited_families = [gate[len("metal_"):-len("_device_gate")] for gate in cited]
cited_released = [f for f in cited_families if f in released]
print(f"cited welds released: {len(cited_released)} of {len(cited)}")
for family in cited_families:
    if family not in released:
        print(f"  cited, not released: {family}")
if archive_named:
    print(f"WARNING: {len(archive_named)} artifact(s) name a dated archive probe: "
          f"{archive_named}. A round binds the probes it cut; such an artifact records "
          f"a file its loaders did not read, or read a record another run measured")
else:
    print("no artifact names a dated archive probe")
if (logs / ".no_lift").exists():
    print("NOTE: run under ALLOW_NO_LIFT=1; the gates whose lift leg re-lifts corpus "
          "rows were not expected to release here")
EOF
}

stage_pack() {
  need status
  # No idle-GPU check here: packing reads files and runs nothing on the GPU.
  tree_is_the_commit
  { date -u +%FT%TZ; git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --porcelain; } > "$L/environment_end.txt" 2>&1
  ( cd "$R" && ls -d metal_*_"$STAMP" "logs_metal_$STAMP" | LC_ALL=C sort > "$L/roots.txt" )
  local out=${OUT_DIR:-$HOME}/metal_round_$STAMP.tgz
  ( cd "$R" && tar --exclude workdir --exclude __pycache__ --exclude '*.tmp' -czf "$out" -T "$L/roots.txt" ) \
      || die "could not pack $out"
  shasum -a 256 "$out" | tee "$out.sha256"
  say "packed $(wc -l < "$L/roots.txt" | tr -d ' ') roots -> $out ($(du -h "$out" | cut -f1)); send it and its .sha256 to the rebind host"
}

ORDER=(preflight inputs fleet status pack)
run_stage() { STAGE_NOW=$1; T_STAGE=$(date +%s); say "start"; "stage_$1"; done_mark "$1"; }
if [[ "$STAGE" == all ]]; then
  for s in $ORDER; do
    if [[ -f "$L/.done_$s" ]]; then STAGE_NOW=$s; say "already finished; skipping"; continue; fi
    run_stage "$s"
  done
elif (( ${ORDER[(Ie)$STAGE]} )); then
  run_stage "$STAGE"
else
  print -r -- "unknown stage $STAGE; one of: $ORDER all"; exit 2
fi
