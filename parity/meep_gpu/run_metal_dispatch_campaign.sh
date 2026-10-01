#!/bin/zsh
# THE METAL DISPATCH CAMPAIGN: five legs of gate_dispatch_metal_route.py -- a serial
# prefix of two, then up to LANES of the three long legs at once.
#
# WHY FIVE PROCESSES AND NOT ONE. Each leg differs in exactly one thing the process
# cannot change after it has started: the float32 subnormal policy is installed once
# and LOCKED (`subnormal_policy.SubnormalPolicyLocked` refuses a second, different
# one, because device binaries and cache entries already exist under the first), and
# the expansion-probe environment is read when a family binds its multiply arm. So
# "the shipped leg" and "the harness_keep leg" are two processes by construction,
# not by preference.
#
#   shipped                  the harness installs NOTHING. Rung 8bM installs flush on
#                            host + mps itself, which is precisely the evidence
#                            DISPATCH_BY_DEFAULT would eventually wait on.
#   harness_flush            --install-policy flush BEFORE lifting: dispatch finds
#                            the policy installed by something other than its own
#                            freeze, and must reach the same answer.
#   harness_keep             --install-policy keep: every dispatch case must be
#                            PASS-POLICY-REFUSED BY NAME and byte-identical to the
#                            array path. THE POLARITY IS INVERTED against the Triton
#                            campaign — Triton certifies under keep so its harness
#                            leg installs flush; this table certifies under flush, so
#                            the refusal leg is the keeping one.
#   shipped_expansion_probe  the four MEEP_GPU_METAL_*_EXPANSION_PROBE paths, taken
#                            from the fleet recut's probe outputs and handed to THIS
#                            leg's process only. bloch_2d and folded_complex_2d
#                            dispatch ONLY here; the shipped leg must show them
#                            refused by name without a licence.
#   band_witness             the array path alone with the host KEEPING, censusing
#                            every stored array every step. The vacuity floor under
#                            the whole flush-equivalence claim: if no case's own
#                            trajectory enters the subnormal band, the equivalence
#                            this campaign licenses is about nothing. Handed
#                            MEEP_GPU_DISPATCH=0 (see below), so "the array path
#                            alone" is a declaration and not a consequence.
#
# THE ORDER, AND WHAT RUNS BESIDE WHAT
# (the design notes (sharded-route-campaign-design) §3a item 6, §3c "Pool").
#
#   1. band_witness, FIRST AND ALONE. Its gate.json is an INPUT to every later leg
#      (the cliff control reads it through MEEP_GPU_METAL_BAND_WITNESS), so nothing
#      may start before it has landed.
#   2. harness_keep, still in the serial prefix. It costs ~120 s, and that its keep
#      install is process-local is read from the code, not measured beside a
#      flushing neighbour; until a co-run has been diffed it runs with the device to
#      itself.
#   3. shipped_expansion_probe, shipped, harness_flush as LANES -- at most LANES at
#      once, LONGEST FIRST by the measured 12,161 / 7,796 / 7,428 s
#      (dispatch_metal_route_2026-09-19_components/campaign.txt), so the longest leg
#      is never the one left waiting for a slot.
#
# THIS ORDER DIFFERS FROM THE COMMITTED SEQUENTIAL SCRIPT'S AT EVERY LANES, 1
# INCLUDED. That one ran band_witness, shipped, harness_flush, harness_keep,
# shipped_expansion_probe; this one runs band_witness, harness_keep,
# shipped_expansion_probe, shipped, harness_flush. The band witness is the only
# artifact one leg writes for another to read: every leg has its own directory, log
# and source manifest, and the gate has no wall-clock logic. What else the legs share
# on disk is the operating system's compiled-shader cache, keyed by the shader source
# and shared under the old order too. So no leg's input depends on where it sits
# after the witness.
#
# LANES=1 (the default) is the strictly sequential campaign: one process at a time,
# no background job, and every leg is handed exactly the arguments and environment
# it was handed before lanes existed. Leg directory names and leg arguments are the
# same at every LANES, so `recut_driver_dispatch_record.py` reads a laned campaign
# unchanged. THIS GATE MAKES NO TIMING CLAIM, so a neighbour on the device costs wall
# time and nothing else: under a measured K=2 overlap every verdict, launch count,
# checkpoint and float observable was identical on 42 of 42 case rows and 186 of 186
# control rows (design §5). The measured slowdown per process is in
# parity/meep_gpu/README.md ("Re-running a campaign"); choose LANES from it.
#
# EACH LEG'S ENVIRONMENT IS BUILT ON ITS OWN COMMAND AND NOTHING IS EXPORTED. The
# sequential script exported the four probe paths around the probe leg and unset
# them after. With legs side by side an export reaches every sibling started while
# it is set, and `run_case` tests the ENVIRONMENT, not the leg's name: a shipped leg
# started under the probe paths dispatches bloch_2d and reads PASS-FUSED where
# PASS-NOT-THIS-LEG is owed -- silently consuming a licence it was there to show
# refused. So every leg is launched as `env -u <each name it must not see>
# NAME=value ... python ...`, the same array the dry run prints, and BEFORE ANYTHING
# IS LAUNCHED the script reads back the environment each command would really hand
# its child and refuses to start unless:
#   * only shipped_expansion_probe carries any MEEP_GPU_METAL_*_EXPANSION_PROBE;
#   * every leg that consumes the band witness carries MEEP_GPU_METAL_BAND_WITNESS
#     naming THIS run's artifact (band_witness itself produces that file and is
#     handed no such path, as before);
#   * MEEP_GPU_SUBNORMAL_POLICY, MEEP_GPU_SUBNORMAL_INSTALL, MEEP_GPU_DISPATCH,
#     MEEP_GPU_FUSED, MEEP_GPU_FUSE_ARMS, MEEP_GPU_DISPATCH_LOG and
#     MEEP_GPU_GATE_SOURCE_MANIFEST reach no leg from the calling shell, whatever it
#     exported -- and band_witness, alone, is then HANDED MEEP_GPU_DISPATCH=0, the
#     exact value, which the check requires of it.
# THE BAND WITNESS'S PIN, and why it is set rather than scrubbed. That leg is the
# array path by construction: it installs keep and censuses the oracle's own
# trajectory. Before dispatch turned on by default (fastpath.DISPATCH_BY_DEFAULT,
# 2026-09-27) the scrubbed enable was what guaranteed that; with the enable unset it
# now rests only on rung 8bM refusing the installed keep policy, a refusal of a
# different fact. Handing the leg MEEP_GPU_DISPATCH=0 makes "dispatches nothing" the
# leg's declaration again. Every other leg still gets the enable scrubbed and
# measures the default, or sets it per sub-leg in-process.
# The first of those seven is the fleet-recut trap: `recut_metal_gates.sh` exports
# MEEP_GPU_SUBNORMAL_POLICY=flush, and run inside that export the `shipped` leg
# would find the policy already resolved and this campaign would carry two copies
# of `harness_flush` and no evidence that rung 8bM installs anything. The second is
# its neighbour: exported as 0 it stops dispatch installing the policy, which turns
# `shipped` from "rung 8bM installs flush" into a refusal on every dispatching leg
# at once. The last is a shared manifest the runner merges with an unlocked
# read-modify-write; unset, each leg writes its own source_sha256.txt, which is what
# every campaign on file did.
# THREE MORE SWITCHES ARE RECORDED, NOT SCRUBBED: MEEP_GPU_WARM,
# MEEP_GPU_KERNEL_TABLE and MEEP_GPU_BACKEND_PREFERENCE reach every leg alike, as
# they always did, and a value that does not belong refuses by name rather than
# passing silently. campaign.txt and the dry run state what the calling shell holds
# for each, and name every other MEEP_GPU_* it exported with what became of it.
#
# INTEGRITY RULES (design review, §3a item 7 and §3c):
#   * THE MACHINE IS HELD AWAKE, TWICE. `caffeinate -i -w <this pid>` covers the
#     driver's life, the gaps between legs included. EACH LEG ALSO HOLDS ITS OWN
#     ASSERTION, because a lane outlives a killed driver (below) and would otherwise
#     run on with none: the leg's command is prefixed `caffeinate -i`. Measured on
#     this host: that tool execs the command IN PLACE -- same pid, same parent, argv
#     and exit status untouched (7, 137, 139 and 143 read back as themselves) -- and
#     leaves a helper CHILD of the gate holding the assertion, gone within 0.2 s of
#     the gate's end however it ends. Killing the helper does not touch the gate. The
#     one thing it adds to the gate's environment is `__CF_USER_TEXT_ENCODING`,
#     and only where the calling shell lacks it (a login session never does);
#     nothing under meep_gpu/ or the gate reads that name. Where the tool is absent
#     the campaign runs without either assertion and says so. A 35,356 s probe leg
#     against 7,955 s (2026-09-15) is what a sleeping host looks like.
#   * THE BAND WITNESS'S sha256 IS RECORDED in campaign.txt beside its path, and
#     checked again at the end. That artifact decides 16 of 24 shipped cliff
#     verdicts and no leg artifact records its path or its bytes.
#   * A LEG DIRECTORY IS CREATED EXCLUSIVELY. One that holds a gate.json THE RUNNER
#     FINALIZED is skipped. One that exists WITHOUT gate.json is REFUSED BY NAME and
#     never relaunched into: the gate would exit 5 on the earlier rows and the
#     runner would then finalize whatever gate.json it finds. Moving such a
#     directory aside is an owner action.
#   * "FINALIZED" IS THE RUNNER'S `weld` KEY. The gate writes gate.json itself,
#     non-atomically and already carrying `release.released`, BEFORE
#     `metal_gate_runner.finalize_artifact` replaces it with the source-welded one.
#     A process killed inside that window leaves a gate.json that reads
#     released=True and that no runner ever welded. So a gate.json without `weld`
#     (or one that does not parse) is not a result: its directory is REFUSED like
#     any other used one, a leg that lands one reads `unfinalized(...)` and fails,
#     and a band witness in that state is not handed to the later legs.
#   * ONE DRIVER PER CAMPAIGN. If campaign.txt's last `started` line has no
#     `finished` line after it and the driver_pid it names is still this script on
#     this stamp, alive, a second invocation refuses (exit 3) before it writes
#     anything: two drivers would both append to campaign.txt and could hold
#     2 x LANES lanes between them. A driver that was killed leaves a dead pid, and
#     the next invocation resumes normally.
#   * campaign.txt HAS ONE WRITER, this process. A lane writes its result to
#     <run>/<leg>.result (atomically, as its last act); the parent appends landed
#     results in a fixed leg order, so two legs finishing together cannot interleave.
#   * A LEG THAT LANDS released=False IS A RESULT. It is recorded, it sets fail=1,
#     and it is never retried -- a campaign that retried would be retry-until-green.
#   * `fail=0` MEANS EVERY LEG RELEASED, which is how
#     `meep_gpu/test_dispatch_contract.py` reads it: a non-zero exit, an unreleased or
#     unreadable gate.json (from this run or a skipped earlier one), a refused
#     directory, a lane that died without a result and a skipped probe leg each set
#     fail=1. The parent waits for EVERY lane and reports every exit code.
#
# STOPPING A CAMPAIGN, AND WHAT Ctrl-C DOES. There is no INT/TERM trap: zsh defers a
# trap until its foreground child ends, which at LANES=1 would make the driver
# unkillable for a whole leg. So:
#   * In the serial prefix, and everywhere at LANES=1, the gate is the driver's
#     foreground child and Ctrl-C reaches it, as it always did.
#   * AT LANES>1 THE LANES IGNORE SIGINT AND SIGQUIT (a background job of a
#     non-interactive zsh starts with both ignored, and the gate inherits that).
#     Ctrl-C stops the driver and the driver's assertion and NOT the gates: they run
#     on under their own assertions, finish, and leave gate.json and <leg>.result,
#     and a re-invocation SKIPs the finished ones and REFUSES the live ones.
#   * THE HANDLE ON A LANE IS ITS GATE'S PID, which the parent records as a `LANE`
#     line in campaign.txt (and the lane keeps in <run>/<leg>.pid, marked `ended`
#     once the gate has). `kill -TERM <that pid>` stops that gate; its lane then
#     lands `exit=143` like any other result. Do not signal the lane's own shell:
#     that orphans the gate and lands `exit=?` while the gate is still running.
#   * A STOPPED OR UNRELEASED LEG HAS USED ITS DIRECTORY, and a campaign that lands
#     one has used its stamp: nothing is retried, so the re-run is a new stamp, and
#     `metal_dispatch.METAL_DRIVER_ROUTE_GATE` must name the new run before it
#     starts, because every leg records that file's bytes.
#
# EXIT STATUS: 0 every leg released; 1 any leg failed, was refused, skipped without
# a probe root, or is unreleased; 2 usage; 3 another live driver holds this
# campaign; 4 the per-leg environments are not isolated.
#
# progress reporting: one flushed line per leg as it starts and as it lands, with UTC times, a
# per-leg log and the gate's own `progress.log` inside each leg directory, and a
# closing line giving wall clock against the sum of leg seconds -- so the state of a
# multi-hour campaign is readable from the filesystem while it is still running.
#
# Usage (from the repository root):
#   zsh parity/meep_gpu/run_metal_dispatch_campaign.sh [--dry-run] [--lanes N] \
#       <stamp> [probe-root] [complex-probe] [lanes]
#
# <stamp> names the run directory: results/dispatch_metal_route_<stamp>/, which is
# what `metal_dispatch.METAL_DRIVER_ROUTE_GATE` must equal and what
# `recut_driver_dispatch_record.py --backend metal --run` reads.
#
# LANES is `--lanes N`, else the fourth positional argument, else $LANES, else 1.
# `--dry-run` (or DRY_RUN=1) prints the plan -- order, lanes, each leg's state,
# arguments, environment and full command -- runs the environment check, and
# creates, writes and launches NOTHING.
#
# Three hooks exist for the device-free tests and are recorded in campaign.txt when
# set: METAL_CAMPAIGN_RESULTS_ROOT (where run directories live),
# METAL_CAMPAIGN_RUNNER (the script handed to python in place of
# metal_gate_runner.py) and METAL_CAMPAIGN_POLL_SECONDS (how often the parent looks
# for a landed lane; default 5).
set -u
zmodload zsh/datetime
SCRIPT_NAME=${0:t}
API=${MGPU_SITE_SOURCE_ROOT:-${0:A:h:h:h}}
PY=${MGPU_SITE_PYTHON:-python}
USAGE='usage: run_metal_dispatch_campaign.sh [--dry-run] [--lanes N] <stamp> [probe-root] [complex-probe] [lanes]'

DRY_RUN=${DRY_RUN:-0}
lanes_option=
positional=()
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --lanes)   shift; lanes_option=${1:?--lanes needs a count. $USAGE} ;;
    --lanes=*) lanes_option=${1#--lanes=} ;;
    --)        shift; positional+=("$@"); break ;;
    -*)        print -u2 -r -- "unknown option $1"; print -u2 -r -- "$USAGE"; exit 2 ;;
    *)         positional+=("$1") ;;
  esac
  shift
done
STAMP=${positional[1]:?$USAGE}
PROBE_ROOT=${positional[2]:-}
# THE BASE COMPLEX PROBE IS NAMED SEPARATELY because it is written beside the
# complex gate's own artifact and the other three are written into a probes/
# directory by the fleet. Requiring all four in one directory meant copying a
# measured artifact to satisfy a path, which is how a record loses its provenance.
COMPLEX_PROBE=${positional[3]:-}
LANES=${lanes_option:-${positional[4]:-${LANES:-1}}}
if [[ "$LANES" != <1-> ]]; then
  print -u2 -r -- "lanes must be a positive integer, not '$LANES'. $USAGE"
  exit 2
fi
[[ "$DRY_RUN" == (1|true|yes) ]] && DRY_RUN=1 || DRY_RUN=0

RESULTS=${METAL_CAMPAIGN_RESULTS_ROOT:-$API/parity/meep_gpu/results}
RUNNER=${METAL_CAMPAIGN_RUNNER:-$API/parity/meep_gpu/metal_gate_runner.py}
GATE="$API/parity/meep_gpu/gate_dispatch_metal_route.py"
POLL_SECONDS=${METAL_CAMPAIGN_POLL_SECONDS:-5}
RUN="$RESULTS/dispatch_metal_route_$STAMP"
SUMMARY="$RUN/campaign.txt"
WITNESS="$RUN/band_witness/gate.json"
cd "$API" || exit 1

# THE SERIAL PREFIX AND THE LANES, in launch order. The lane order is LONGEST FIRST
# by the measured leg seconds below; the numbers are printed with the plan so a
# reader can see what the order was decided on.
SERIAL_LEGS=(band_witness harness_keep)
LANE_LEGS=(shipped_expansion_probe shipped harness_flush)
ALL_LEGS=($SERIAL_LEGS $LANE_LEGS)
typeset -A MEASURED_SECONDS
MEASURED_SECONDS=(band_witness 27 harness_keep 120
                  shipped_expansion_probe 12161 shipped 7796 harness_flush 7428)

# NAMES NO LEG MAY INHERIT, whatever the calling shell exported (see the header).
SCRUBBED=(MEEP_GPU_SUBNORMAL_POLICY MEEP_GPU_SUBNORMAL_INSTALL MEEP_GPU_DISPATCH
          MEEP_GPU_FUSED MEEP_GPU_FUSE_ARMS MEEP_GPU_DISPATCH_LOG
          MEEP_GPU_GATE_SOURCE_MANIFEST)
# SWITCHES THAT REACH EVERY LEG ALIKE and are therefore put on record, not scrubbed.
RECORDED_SWITCHES=(MEEP_GPU_WARM MEEP_GPU_KERNEL_TABLE MEEP_GPU_BACKEND_PREFERENCE)
PROBE_VARIABLES=(MEEP_GPU_METAL_EXPANSION_PROBE
                 MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE
                 MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE
                 MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE)
WITNESS_VARIABLE=MEEP_GPU_METAL_BAND_WITNESS
# THE ENABLE, pinned OFF on band_witness only (header, "THE BAND WITNESS'S PIN").
DISPATCH_ENABLE=MEEP_GPU_DISPATCH

# THE PER-LEG SLEEP ASSERTION: the tool's own path, resolved once, so the plan shows
# the command that really runs. Empty where the tool is absent.
CAFFEINATE=$(command -v caffeinate 2> /dev/null) || CAFFEINATE=

# THE PROBE LEG'S INPUTS, resolved once. Without a probe root the two complex cases
# cannot dispatch, and a leg that ran them anyway would record a refusal as though
# it were the answer. So the leg is SKIPPED BY NAME rather than run empty.
PROBE_AVAILABLE=0
COMPLEX_PROBE_PATH=
if [[ -n "$PROBE_ROOT" && -f "$PROBE_ROOT/special_kz.json" ]]; then
  PROBE_AVAILABLE=1
  # The base complex probe is written BESIDE the complex gate's own artifact rather
  # than into the probe root, so it is named separately.
  if [[ -n "$COMPLEX_PROBE" && -f "$COMPLEX_PROBE" ]]; then
    COMPLEX_PROBE_PATH="$COMPLEX_PROBE"
  elif [[ -f "$PROBE_ROOT/complex_expansion_probe.json" ]]; then
    COMPLEX_PROBE_PATH="$PROBE_ROOT/complex_expansion_probe.json"
  fi
fi

# ---------------------------------------------------------------------------
# What each leg is handed
# ---------------------------------------------------------------------------

leg_arguments() {  # leg_arguments <leg>  ->  $reply
  case "$1" in
    band_witness)            reply=(--install-policy keep --band-witness) ;;
    harness_keep)            reply=(--install-policy keep) ;;
    harness_flush)           reply=(--install-policy flush) ;;
    shipped)                 reply=() ;;
    shipped_expansion_probe) reply=() ;;
    *) print -u2 -r -- "no such leg: $1"; exit 2 ;;
  esac
}

build_leg() {  # build_leg <leg>  ->  leg_set, leg_unset, leg_args, leg_environment, leg_command
  local leg="$1" name
  leg_set=(KMP_DUPLICATE_LIB_OK=TRUE MPLBACKEND=Agg "PYTHONPATH=$API")
  leg_unset=($SCRUBBED)
  if [[ "$leg" == band_witness ]]; then
    # The producer of the witness is handed no witness path.
    leg_unset+=($WITNESS_VARIABLE)
    # The array path by declaration: the enable is SET to 0 on this leg rather than
    # scrubbed (header, "THE BAND WITNESS'S PIN").
    leg_unset=(${leg_unset:#$DISPATCH_ENABLE})
    leg_set+=("$DISPATCH_ENABLE=0")
  else
    leg_set+=("$WITNESS_VARIABLE=$WITNESS")
  fi
  if [[ "$leg" == shipped_expansion_probe ]]; then
    leg_set+=("MEEP_GPU_METAL_EXPANSION_PROBE=$PROBE_ROOT/special_kz.json"
              "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE=$PROBE_ROOT/folded_complex.json"
              "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE=$PROBE_ROOT/cylindrical_complex.json")
    if [[ -n "$COMPLEX_PROBE_PATH" ]]; then
      leg_set+=("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE=$COMPLEX_PROBE_PATH")
    else
      leg_unset+=(MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE)
    fi
  else
    leg_unset+=($PROBE_VARIABLES)
  fi
  leg_arguments "$leg"; leg_args=("${reply[@]}")
  leg_environment=(/usr/bin/env)
  for name in $leg_unset; do leg_environment+=(-u "$name"); done
  leg_environment+=("${leg_set[@]}")
  # `caffeinate -i <command>` execs the command in place and leaves a helper child
  # holding the assertion for exactly the gate's life (header, INTEGRITY RULES).
  leg_command=()
  [[ -n "$CAFFEINATE" ]] && leg_command=("$CAFFEINATE" -i)
  leg_command+=("${leg_environment[@]}" "$PY" -u "$RUNNER" "$GATE" --
                --out "$RUN/$leg" "${leg_args[@]}")
}

read_released() {  # read_released <gate.json>  ->  stdout, one line
  # True | False from a gate.json THE RUNNER FINALIZED (it carries `weld`);
  # unfinalized(...) where the gate wrote one and no runner welded it;
  # unreadable(...) where it is absent or does not parse.
  "$PY" - "$1" <<'EOF'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
    released = payload["release"]["released"]
    if not isinstance(payload.get("weld"), dict):
        released = f"unfinalized({released})"
    print(released)
except Exception as exc:
    print(f"unreadable({exc!r})")
EOF
}

leg_state() {  # leg_state <leg>  ->  $REPLY in RUN | SKIP | REFUSE | NO-PROBE-ROOT
  # SKIP also sets $LEG_RELEASED; REFUSE also sets $LEG_REFUSAL, the reason by name.
  local leg="$1"
  LEG_RELEASED= LEG_REFUSAL=
  if [[ "$leg" == shipped_expansion_probe && $PROBE_AVAILABLE -eq 0 ]]; then
    REPLY=NO-PROBE-ROOT
  elif [[ -e "$RUN/$leg/gate.json" ]]; then
    LEG_RELEASED=$(read_released "$RUN/$leg/gate.json")
    if [[ "$LEG_RELEASED" == (True|False) ]]; then
      REPLY=SKIP
    else
      REPLY=REFUSE
      LEG_REFUSAL="$RUN/$leg holds a gate.json the runner never finalized ($LEG_RELEASED)"
    fi
  elif [[ -e "$RUN/$leg" || -L "$RUN/$leg" ]]; then
    REPLY=REFUSE
    LEG_REFUSAL="$RUN/$leg exists without gate.json"
  else
    REPLY=RUN
  fi
}

# ANOTHER LIVE DRIVER OF THIS CAMPAIGN: the last `started` line of campaign.txt with
# no `finished` line after it, naming a pid that is alive AND is this script run on
# this stamp. A pid the system has since handed to something else -- another stamp's
# driver included -- is not this campaign's driver. What this does not cover is two
# drivers started in the same instant, before either has written its `started` line;
# the exclusive leg directories still keep those two from sharing a leg.
other_driver() {  # other_driver  ->  $REPLY, that driver's pid, or empty
  local last pid
  REPLY=
  [[ -s "$SUMMARY" ]] || return 0
  last=$(grep -E "^campaign dispatch_metal_route_.*  (started|finished) " "$SUMMARY" | tail -1)
  [[ "$last" == *"  started "* ]] || return 0
  pid=${${last##*driver_pid=}%%[^0-9]*}
  [[ "$pid" == <1-> && "$pid" != "$$" ]] || return 0
  kill -0 "$pid" 2> /dev/null || return 0
  [[ "$(ps -o command= -p "$pid" 2> /dev/null)" == *"$SCRIPT_NAME"*"$STAMP"* ]] || return 0
  REPLY=$pid
}

# WHAT THE CALLING SHELL HOLDS that a leg could see: the three recorded switches by
# value, and every other MEEP_GPU_* name it exported with what becomes of it.
caller_environment_lines() {  # -> $reply, the lines to print
  local name value others=()
  local switches=()
  for name in $RECORDED_SWITCHES; do
    if [[ "${parameters[$name]:-}" == *export* ]]; then
      switches+=("$name=${(P)name}")
    else
      switches+=("$name=(unset)")
    fi
  done
  reply=("SWITCHES recorded, not scrubbed; each reaches every leg as the calling shell holds it: ${(j:  :)switches}")
  for name in ${(ok)parameters[(I)MEEP_GPU_*]}; do
    [[ "${parameters[$name]:-}" == *export* ]] || continue
    (( ${RECORDED_SWITCHES[(Ie)$name]} )) && continue
    if [[ "$name" == "$DISPATCH_ENABLE" ]]; then
      others+=("$name (scrubbed from every leg; band_witness is handed 0)")
    elif (( ${SCRUBBED[(Ie)$name]} )); then
      others+=("$name (scrubbed from every leg)")
    elif (( ${PROBE_VARIABLES[(Ie)$name]} )) || [[ "$name" == "$WITNESS_VARIABLE" ]]; then
      others+=("$name (replaced or unset on each leg)")
    else
      others+=("$name=${(P)name} (reaches every leg)")
    fi
  done
  (( ${#others} )) && reply+=("CALLER exported: ${(j:, :)others}")
  return 0
}

# THE ENVIRONMENT CHECK reads back what each leg's `env` prefix REALLY hands a child
# -- the inherited environment included -- by running that same prefix around
# `env` itself. It launches no gate and touches no device.
ENVIRONMENT_FAULTS=()
check_leg_environment() {  # check_leg_environment <leg>  ->  $REPLY (a one-line report)
  local leg="$1" effective name probes=0 scrubbed=0 must_scrub=0 witness=no dispatch=unset
  build_leg "$leg"
  effective=$("${leg_environment[@]}" /usr/bin/env)
  for name in $PROBE_VARIABLES; do
    if print -r -- "$effective" | grep -q "^$name="; then
      (( probes += 1 ))
      [[ "$leg" == shipped_expansion_probe ]] || \
        ENVIRONMENT_FAULTS+=("$leg carries $name; only shipped_expansion_probe may")
    fi
  done
  if [[ "$leg" == shipped_expansion_probe ]] && ! print -r -- "$effective" | \
       grep -Fqx -- "MEEP_GPU_METAL_EXPANSION_PROBE=$PROBE_ROOT/special_kz.json"; then
    ENVIRONMENT_FAULTS+=("$leg does not carry MEEP_GPU_METAL_EXPANSION_PROBE")
  fi
  for name in $SCRUBBED; do
    if [[ "$leg" == band_witness && "$name" == "$DISPATCH_ENABLE" ]]; then
      # The one name this leg is HANDED, and it must be exactly 0.
      if print -r -- "$effective" | grep -Fqx -- "$DISPATCH_ENABLE=0"; then
        dispatch=0
      else
        dispatch=$(print -r -- "$effective" | grep "^$DISPATCH_ENABLE=" | cut -d= -f2-)
        ENVIRONMENT_FAULTS+=("band_witness does not carry exactly $DISPATCH_ENABLE=0 (it carries '${dispatch:-(unset)}'); the array-path census would be left to the default")
      fi
      continue
    fi
    (( must_scrub += 1 ))
    if print -r -- "$effective" | grep -q "^$name="; then
      ENVIRONMENT_FAULTS+=("$leg carries $name; no leg may")
    else
      (( scrubbed += 1 ))
    fi
  done
  if print -r -- "$effective" | grep -Fqx -- "$WITNESS_VARIABLE=$WITNESS"; then
    witness=yes
    [[ "$leg" == band_witness ]] && \
      ENVIRONMENT_FAULTS+=("band_witness is handed $WITNESS_VARIABLE; it produces that file")
  elif print -r -- "$effective" | grep -q "^$WITNESS_VARIABLE="; then
    witness=OTHER
    ENVIRONMENT_FAULTS+=("$leg carries a $WITNESS_VARIABLE that is not this run's")
  elif [[ "$leg" != band_witness ]]; then
    ENVIRONMENT_FAULTS+=("$leg does not carry $WITNESS_VARIABLE")
  fi
  REPLY=$(printf 'ENVIRONMENT %-26s band_witness=%s  probe_variables=%d/%d  scrubbed_absent=%d/%d  dispatch=%s' \
          "$leg" "$witness" "$probes" "${#PROBE_VARIABLES}" "$scrubbed" "$must_scrub" "$dispatch")
}

print_plan() {
  local leg order=0 phase name
  print -r -- "PLAN dispatch_metal_route_$STAMP  lanes=$LANES  run=$RUN"
  print -r -- "PLAN order: ${SERIAL_LEGS[*]} (serial, one at a time) then ${LANE_LEGS[*]} (at most $LANES at once, longest first)"
  for leg in $ALL_LEGS; do
    (( order += 1 ))
    phase=lane; (( ${SERIAL_LEGS[(Ie)$leg]} )) && phase=serial
    leg_state "$leg"; build_leg "$leg"
    print -r -- "LEG order=$order phase=$phase leg=$leg state=$REPLY measured_seconds=${MEASURED_SECONDS[$leg]}"
    if [[ "$REPLY" == NO-PROBE-ROOT ]]; then
      print -r -- "  not launched: no probe root holding special_kz.json was given, so there is no command to show"
      continue
    fi
    [[ "$REPLY" == SKIP ]] && print -r -- "  held      gate.json, finalized by the runner, released=$LEG_RELEASED"
    [[ "$REPLY" == REFUSE ]] && print -r -- "  refused   $LEG_REFUSAL"
    print -r -- "  out       $RUN/$leg"
    print -r -- "  log       $RUN/$leg.log"
    print -r -- "  arguments ${leg_args[*]:-(none)}"
    for name in "${leg_set[@]}"; do print -r -- "  set       $name"; done
    for name in $leg_unset; do print -r -- "  unset     $name"; done
    print -r -- "  command   ${(j: :)${(q-)leg_command[@]}}"
  done
}

# ---------------------------------------------------------------------------
# Checked before anything is created or launched, in both modes
# ---------------------------------------------------------------------------

environment_reports=()
for leg in $ALL_LEGS; do
  [[ "$leg" == shipped_expansion_probe && $PROBE_AVAILABLE -eq 0 ]] && continue
  check_leg_environment "$leg"
  environment_reports+=("$REPLY")
done
caller_environment_lines; caller_lines=("${reply[@]}")
other_driver; OTHER_DRIVER=$REPLY

if (( DRY_RUN )); then
  print -r -- "DRY RUN: nothing is created, written or launched"
  [[ -n "${METAL_CAMPAIGN_RESULTS_ROOT:-}" ]] && print -r -- "NOTE   results root overridden: $RESULTS"
  [[ -n "${METAL_CAMPAIGN_RUNNER:-}" ]] && print -r -- "NOTE   runner overridden: $RUNNER"
  print_plan
  for line in "${environment_reports[@]}"; do print -r -- "$line"; done
  for line in "${caller_lines[@]}"; do print -r -- "$line"; done
  if [[ -f "$WITNESS" ]]; then
    print -r -- "BAND WITNESS present  sha256=$(shasum -a 256 "$WITNESS" | cut -d' ' -f1)  $WITNESS"
  else
    print -r -- "BAND WITNESS produced by the first leg; its sha256 is recorded in campaign.txt when it lands  $WITNESS"
  fi
  if [[ -n "$CAFFEINATE" ]]; then
    print -r -- "AWAKE  caffeinate -i -w <driver pid> would be held for the campaign's life, and each leg's command holds its own (caffeinate -i)"
  else
    print -r -- "AWAKE  caffeinate is absent; the campaign would run without a sleep assertion and say so"
  fi
  dry_fail=0
  if [[ -n "$OTHER_DRIVER" ]]; then
    print -r -- "WOULD REFUSE the campaign (exit 3): driver pid $OTHER_DRIVER started it and has not finished  $SUMMARY"
    dry_fail=1
  fi
  for leg in $ALL_LEGS; do
    leg_state "$leg"
    case "$REPLY" in
      REFUSE)        print -r -- "WOULD REFUSE $leg: $LEG_REFUSAL"; dry_fail=1 ;;
      NO-PROBE-ROOT) print -r -- "WOULD SKIP $leg: no probe root given; bloch_2d and folded_complex_2d cannot dispatch"; dry_fail=1 ;;
      SKIP)          [[ "$LEG_RELEASED" == True ]] || { print -r -- "WOULD FAIL on $leg: the gate.json it holds is released=$LEG_RELEASED"; dry_fail=1 } ;;
    esac
  done
  if (( ${#ENVIRONMENT_FAULTS} )); then
    for line in "${ENVIRONMENT_FAULTS[@]}"; do print -r -- "ENVIRONMENT FAULT  $line"; done
    print -r -- "DRY RUN verdict: REFUSING, the per-leg environments are not isolated"
    exit 4
  fi
  print -r -- "ENVIRONMENT CHECK ok"
  print -r -- "DRY RUN verdict: $( (( dry_fail )) && print 'the campaign would not exit 0 (see WOULD lines)' || print 'every leg would run or skip on a held, released gate.json')"
  exit $dry_fail
fi

if (( ${#ENVIRONMENT_FAULTS} )); then
  for line in "${ENVIRONMENT_FAULTS[@]}"; do print -u2 -r -- "ENVIRONMENT FAULT  $line"; done
  print -u2 -r -- "REFUSING: the per-leg environments are not isolated; nothing was launched"
  exit 4
fi

if [[ -n "$OTHER_DRIVER" ]]; then
  print -u2 -r -- "REFUSING: driver pid $OTHER_DRIVER started this campaign and has not finished ($SUMMARY); nothing was written or launched"
  exit 3
fi

# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------

mkdir -p "$RUN"
# ONE WRITER, AND IT APPENDS. A resumed campaign keeps the earlier block -- its
# START/exit lines and the band witness digest are the only record of them.
record() { print -r -- "$*" | tee -a "$SUMMARY"; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

campaign_started=$EPOCHSECONDS
[[ -s "$SUMMARY" ]] && record "=== resumed: the lines above are an earlier run of this campaign ==="
record "campaign dispatch_metal_route_$STAMP  started $(utc)  lanes=$LANES  driver_pid=$$"
record "per-leg environments built on each command, nothing exported: ${(j:, :)SCRUBBED} reach no leg (fleet-recut inheritance trap)"
[[ -n "${METAL_CAMPAIGN_RESULTS_ROOT:-}" ]] && record "NOTE   results root overridden: $RESULTS"
[[ -n "${METAL_CAMPAIGN_RUNNER:-}" ]] && record "NOTE   runner overridden: $RUNNER"
for line in "${environment_reports[@]}"; do record "$line"; done
for line in "${caller_lines[@]}"; do record "$line"; done
{ print -r -- "=== $(utc) driver_pid=$$"; print_plan } >> "$RUN/campaign_plan.txt"
record "plan (each leg's arguments, environment and full command): $RUN/campaign_plan.txt"

CAFFEINATE_PID=
if [[ -n "$CAFFEINATE" ]]; then
  "$CAFFEINATE" -i -w $$ &
  CAFFEINATE_PID=$!
  record "AWAKE  caffeinate -i -w $$ held (pid $CAFFEINATE_PID) for the campaign's life; each leg's command also holds its own (caffeinate -i), which outlives a killed driver"
else
  record "AWAKE  NOTE caffeinate is absent: no sleep assertion is held, by the driver or by any leg, and a sleeping host stalls every leg"
fi
(( LANES > 1 )) && record "SIGNALS  lanes ignore SIGINT and SIGQUIT: Ctrl-C stops this driver and not the gates. Each lane's gate pid is on its LANE line; stop one with kill -TERM <gate pid>"

fail=0
leg_seconds_total=0
typeset -A LEG_OUTCOME

write_gate_pid() {  # write_gate_pid <leg> <pid> running|ended  ->  <run>/<leg>.pid, atomically
  print -r -- "driver $$ gate_pid $2 $3" > "$RUN/$1.pid.partial"
  mv -f "$RUN/$1.pid.partial" "$RUN/$1.pid"
}

leg_process() {  # leg_process <leg> [lane]: run it, then write <leg>.result. NEVER writes campaign.txt.
  local leg="$1" mode="${2:-alone}" out="$RUN/$1" log="$RUN/$1.log"
  local started rc elapsed released line gate_pid
  build_leg "$leg"
  started=$EPOCHSECONDS
  if [[ "$mode" == lane ]]; then
    # A LANE IS ALREADY A BACKGROUND SHELL, so a second `&` changes nothing the gate
    # can see -- measured: the same stdin, and SIGINT and SIGQUIT ignored, either way
    # -- and it yields the gate's OWN pid, because the command execs in place all the
    # way to python. The serial form below stays a foreground launch, as it always was.
    "${leg_command[@]}" > "$log" 2>&1 &
    gate_pid=$!
    write_gate_pid "$leg" "$gate_pid" running
    wait $gate_pid
    rc=$?
    write_gate_pid "$leg" "$gate_pid" ended
  else
    "${leg_command[@]}" > "$log" 2>&1
    rc=$?
  fi
  elapsed=$(( EPOCHSECONDS - started ))
  released=$(read_released "$out/gate.json")
  line=$(printf "%-26s exit=%d  released=%s  %5ds  %s  landed %s" \
         "$leg" "$rc" "$released" "$elapsed" "$log" "$(utc)")
  {
    print -r -- "driver $$"
    print -r -- "exit $rc"
    print -r -- "released $released"
    print -r -- "seconds $elapsed"
    print -r -- "line $line"
  } > "$RUN/$leg.result.partial"
  mv -f "$RUN/$leg.result.partial" "$RUN/$leg.result"
}

admit_leg() {  # admit_leg <leg> -> 0 when the leg may be launched; records why not otherwise
  local leg="$1"
  leg_state "$leg"
  case "$REPLY" in
    NO-PROBE-ROOT)
      record "$(printf '%-26s SKIP   no probe root given; bloch_2d and folded_complex_2d cannot dispatch' "$leg")"
      LEG_OUTCOME[$leg]="skipped(no probe root)"; fail=1; return 1 ;;
    SKIP)
      record "$(printf '%-26s SKIP   already holds a finalized gate.json  released=%s' "$leg" "$LEG_RELEASED")"
      LEG_OUTCOME[$leg]="skipped(released=$LEG_RELEASED)"
      [[ "$LEG_RELEASED" == True ]] || fail=1
      return 1 ;;
    REFUSE)
      record "$(printf '%-26s REFUSED  %s; a used leg directory is never relaunched into' "$leg" "$LEG_REFUSAL")"
      LEG_OUTCOME[$leg]=refused; fail=1; return 1 ;;
  esac
  if [[ "$leg" != band_witness ]] && (( ! WITNESS_USABLE )); then
    record "$(printf '%-26s REFUSED  no runner-finalized band witness artifact at %s; every cliff control would fail closed' "$leg" "$WITNESS")"
    LEG_OUTCOME[$leg]=refused; fail=1; return 1
  fi
  # EXCLUSIVE: a plain mkdir fails if anything took the name since the check above.
  if ! mkdir "$RUN/$leg" 2> /dev/null; then
    record "$(printf '%-26s REFUSED  %s could not be created exclusively' "$leg" "$RUN/$leg")"
    LEG_OUTCOME[$leg]=refused; fail=1; return 1
  fi
  if [[ "$leg" == shipped_expansion_probe && -z "$COMPLEX_PROBE_PATH" ]]; then
    record "$(printf '%-26s NOTE   no base complex probe given; bloch_2d skips by name' "$leg")"
  fi
  record "$(printf '%-26s START  %s' "$leg" "$(utc)")"
  return 0
}

land_leg() {  # land_leg <leg> <lane exit status>: the ONLY place a result reaches campaign.txt
  local leg="$1" lane_status="$2" key value driver= rc= released= seconds=0 line=
  if [[ -f "$RUN/$leg.result" ]]; then
    while read -r key value; do
      case "$key" in
        driver) driver=$value ;; exit) rc=$value ;; released) released=$value ;;
        seconds) seconds=$value ;; line) line=$value ;;
      esac
    done < "$RUN/$leg.result"
  fi
  if [[ "$driver" != "$$" || -z "$line" ]]; then
    record "$(printf '%-26s exit=?  released=unreadable  lane ended with status %s and left no result  %s  landed %s' \
             "$leg" "$lane_status" "$RUN/$leg.log" "$(utc)")"
    LEG_OUTCOME[$leg]="exit ? (lane status $lane_status)"; fail=1
    return 0
  fi
  record "$line"
  LEG_OUTCOME[$leg]="exit $rc"
  (( leg_seconds_total += seconds ))
  # A RESULT, NEVER RETRIED: an unreleased leg fails the campaign and stays as it is.
  [[ "$rc" == 0 && "$released" == True ]] || fail=1
  return 0
}

run_leg_alone() {  # the serial form: the leg is the only process this campaign has running
  local leg="$1"
  admit_leg "$leg" || return 0
  leg_process "$leg"
  land_leg "$leg" $?
}

# THE CENSUS FLOOR RUNS FIRST AND ALONE, AND ITS ARTIFACT IS AN INPUT TO THE OTHERS.
# It is the leg that answers "does this case's trajectory enter the subnormal band at
# all", and that is exactly what the cliff control needs to tell "the run never enters
# the band" from "the device did not flush" -- the two hypotheses its NO-CLIFF verdict
# could only list. Measured 2026-09-11: the five cases that read NO-CLIFF are EXACTLY
# the five the witness reports BAND-EMPTY, so the pairing is not a guess. It installs
# keep and dispatches nothing, so it costs under half a minute.
WITNESS_USABLE=0
run_leg_alone band_witness
WITNESS_SHA256=
if [[ ! -f "$WITNESS" ]]; then
  record "band witness  MISSING  $WITNESS  (every later leg is refused by name)"
elif [[ "$(read_released "$WITNESS")" != (True|False) ]]; then
  record "band witness  UNFINALIZED  $WITNESS carries no runner weld  (every later leg is refused by name)"
else
  WITNESS_USABLE=1
  WITNESS_SHA256=$(shasum -a 256 "$WITNESS" | cut -d' ' -f1)
  record "band witness  sha256=$WITNESS_SHA256  $WITNESS_VARIABLE=$WITNESS"
fi

run_leg_alone harness_keep

if (( LANES == 1 )); then
  for leg in $LANE_LEGS; do run_leg_alone "$leg"; done
else
  typeset -A LANE_PID LANE_NOTED
  note_lane() {  # note_lane <leg>: put the gate's pid on record, once. Parent only.
    local leg="$1" word driver key pid state
    (( ${+LANE_NOTED[$leg]} )) && return 0
    [[ -f "$RUN/$leg.pid" ]] || return 1
    read -r word driver key pid state < "$RUN/$leg.pid"
    [[ "$driver" == "$$" && "$pid" == <1-> ]] || return 1
    if [[ "$state" == running ]]; then
      record "$(printf '%-26s LANE   gate_pid=%d  lane_shell=%d  stop it with: kill -TERM %d' \
               "$leg" "$pid" "${LANE_PID[$leg]}" "$pid")"
    else
      record "$(printf '%-26s LANE   gate_pid=%d  lane_shell=%d  (the gate had ended when this was noted)' \
               "$leg" "$pid" "${LANE_PID[$leg]}")"
    fi
    LANE_NOTED[$leg]=1
  }
  queue=($LANE_LEGS)
  running=()
  while (( ${#queue} + ${#running} )); do
    # LAND, in the fixed leg order, every lane whose process has ended.
    for leg in $LANE_LEGS; do
      (( ${running[(Ie)$leg]} )) || continue
      note_lane "$leg"
      kill -0 ${LANE_PID[$leg]} 2> /dev/null && continue
      wait ${LANE_PID[$leg]}
      land_leg "$leg" $?
      running=(${running:#$leg})
    done
    # LAUNCH while a lane is free. A leg that is skipped or refused takes no lane.
    while (( ${#queue} && ${#running} < LANES )); do
      leg=${queue[1]}; queue=(${queue[2,-1]})
      admit_leg "$leg" || continue
      leg_process "$leg" lane &
      # Through a scalar: zsh stores the two characters `$!` when that parameter is
      # assigned straight into an associative-array element.
      lane_pid=$!
      LANE_PID[$leg]=$lane_pid
      running+=("$leg")
      # The lane writes <leg>.pid within milliseconds; give it two seconds, and let
      # the next poll pick it up otherwise.
      for attempt in {1..40}; do note_lane "$leg" && break; sleep 0.05; done
    done
    (( ${#running} )) && sleep "$POLL_SECONDS"
  done
fi

if [[ -n "$WITNESS_SHA256" ]]; then
  if [[ "$(shasum -a 256 "$WITNESS" 2> /dev/null | cut -d' ' -f1)" == "$WITNESS_SHA256" ]]; then
    record "band witness  sha256 unchanged at the end of the campaign"
  else
    record "band witness  CHANGED while the campaign ran: the legs did not all read the bytes recorded above"
    fail=1
  fi
fi

wall=$(( EPOCHSECONDS - campaign_started ))
record "---"
outcomes=()
for leg in $ALL_LEGS; do outcomes+=("$leg=${LEG_OUTCOME[$leg]:-not reached}"); done
record "legs  ${(j:  :)outcomes}"
record "$(printf 'wall clock %ds against %ds of leg seconds run by this driver (%.2fx)  lanes=%d' \
         "$wall" "$leg_seconds_total" "$(( wall > 0 ? 1.0 * leg_seconds_total / wall : 0 ))" "$LANES")"
record "campaign dispatch_metal_route_$STAMP  finished $(utc)  fail=$fail"
record "NEXT: $PY parity/meep_gpu/recut_driver_dispatch_record.py --backend metal --run dispatch_metal_route_$STAMP"
[[ -n "$CAFFEINATE_PID" ]] && kill "$CAFFEINATE_PID" 2> /dev/null
exit $fail
