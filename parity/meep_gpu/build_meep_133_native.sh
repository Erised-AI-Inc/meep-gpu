#!/usr/bin/env bash
# Build a NATIVE-TUNED MEEP 1.33.0 (single precision, MPI) as a TIMING CONTROL, and record
# what was built.
#
# WHAT THIS IS FOR. The certification reference (build_meep_133_linux.sh,
# build_meep_133_macos.sh) is a portable -O2 build. A comparison against MEEP has to show
# that the ratio does not rest on a MEEP compiled below what the host can run, so the
# timing ladder can time a second build beside the reference
# (timing_ladder.py --meep-build native=<this prefix>/bin/python,,<build_record.json>).
# This build is a timing control ONLY: certification is never measured against it
# (docs/development/certification.md, "The MEEP a round is measured against").
#
# HOW. It sets the optimisation flags and delegates the build to the platform's reference
# script, unchanged, so every defect that script works around is worked around here too:
#
#   Linux x86_64 (gcc):   CFLAGS = CXXFLAGS = "-O3 -march=native"
#   macOS arm64 (clang):  CFLAGS = CXXFLAGS = "-O3 -mcpu=native"
#
# configure takes CFLAGS/CXXFLAGS from the environment. --enable-portable-binary, which
# the reference scripts pass, changes only what AX_CC_MAXOPT picks when CFLAGS is UNSET;
# with the flags set here it has no effect, and the build record says so.
#
# WHAT IT WRITES, before the temporary build directory can vanish:
#   <prefix>/share/meep-gpu-build/config.log       configure's log, as run
#   <prefix>/share/meep-gpu-build/Makefile          the top-level Makefile (flags)
#   <prefix>/share/meep-gpu-build/native.json       compiler, -march resolution, CPU
#   <prefix>/share/meep-gpu-build/build_record.json timing_records.py meep-build-record:
#        MEEP version and precision, sha256 of libmeep and _meep*.so, the OpenMP
#        runtime each links, the configure line and the Makefile flags
#
# USAGE (from the repository root; every stage prints a flushed, timestamped marker):
#   bash parity/meep_gpu/build_meep_133_native.sh 2>&1 | tee ~/build_meep_133_native.log
# Environment: MEEP_ENV_NAME (default meep-gpu-ref-native), CONDA_ROOT, MEEP_BUILD_JOBS,
# NATIVE_FLAGS (overrides the flags above), DRY_RUN=1 (print what would run, run nothing).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MEEP_ENV_NAME="${MEEP_ENV_NAME:-meep-gpu-ref-native}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
export CONDA_ROOT
PREFIX="$CONDA_ROOT/envs/$MEEP_ENV_NAME"
OS="$(uname -s)"
ARCH="$(uname -m)"

say() { printf '\n=== native %s === %s\n' "$1" "$(date -u +%FT%TZ)"; }

case "$OS/$ARCH" in
  Linux/x86_64)
    DELEGATE="$HERE/build_meep_133_linux.sh"
    DEFAULT_FLAGS="-O3 -march=native"
    BUILD_ROOT="${MEEP_BUILD_ROOT:-${TMPDIR:-/tmp}/meep_build_133_native_linux}"
    CXX_GLOB="$PREFIX/bin/x86_64-conda-linux-gnu-c++"
    ;;
  Darwin/arm64)
    DELEGATE="$HERE/build_meep_133_macos.sh"
    DEFAULT_FLAGS="-O3 -mcpu=native"
    BUILD_ROOT="${MEEP_BUILD_ROOT:-/tmp/meep_build_133_native_macos}"
    CXX_GLOB="$PREFIX/bin/*-clang++"
    ;;
  *)
    echo "REFUSING: no native build is defined for $OS/$ARCH (Linux x86_64 with gcc, macOS"
    echo "arm64 with clang); build one by hand and record it with"
    echo "  python parity/meep_gpu/timing_records.py meep-build-record --python <its python> ..."
    exit 2
    ;;
esac
FLAGS="${NATIVE_FLAGS:-$DEFAULT_FLAGS}"
export MEEP_BUILD_ROOT="$BUILD_ROOT"
SOURCE_DIR="$BUILD_ROOT/meep-1.33.0"
RECORD_DIR="$PREFIX/share/meep-gpu-build"

say "0. plan"
echo "  delegate     $DELEGATE"
echo "  prefix       $PREFIX"
echo "  build root   $BUILD_ROOT"
echo "  CFLAGS       $FLAGS"
echo "  CXXFLAGS     $FLAGS"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "  DRY RUN: nothing is built"
  exit 0
fi

say "1. build through the platform's reference script, with the native flags"
CFLAGS="$FLAGS" CXXFLAGS="$FLAGS" bash "$DELEGATE"

say "2. keep config.log and the Makefile before the build directory can vanish"
mkdir -p "$RECORD_DIR"
cp "$SOURCE_DIR/config.log" "$RECORD_DIR/config.log"
cp "$SOURCE_DIR/Makefile" "$RECORD_DIR/Makefile"
echo "  kept $RECORD_DIR/config.log and Makefile"

say "3. what -march / -mcpu native resolved to on this CPU"
CXX_BIN="$(ls $CXX_GLOB 2>/dev/null | head -1 || true)"
NATIVE_JSON="$RECORD_DIR/native.json"
"$PREFIX/bin/python" - "$CXX_BIN" "$FLAGS" "$OS" "$NATIVE_JSON" <<'PYEOF'
import json, platform, subprocess, sys
cxx, flags, system, out = sys.argv[1:5]

def run(command):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return (done.stdout + done.stderr).strip()
    except Exception as error:  # recorded, never fatal
        return f"unavailable: {type(error).__name__}: {error}"

record = {"flags": flags, "compiler": cxx or None, "platform": platform.platform(),
          "portable_binary_note": ("--enable-portable-binary changes only what "
                                   "AX_CC_MAXOPT picks when CFLAGS is unset; CFLAGS and "
                                   "CXXFLAGS were set, so it had no effect")}
if cxx:
    record["compiler_version"] = run([cxx, "-v"])[-2000:]
    if system == "Linux":
        record["march_native_resolution"] = run(
            [cxx, "-march=native", "-Q", "--help=target"])[-6000:]
    else:
        record["mcpu_native_resolution"] = run(
            [cxx, "-mcpu=native", "-###", "-x", "c++", "-c", "/dev/null"])[-4000:]
if system == "Linux":
    record["cpu"] = next((line.split(":", 1)[1].strip() for line in
                          run(["lscpu"]).splitlines() if line.startswith("Model name")), None)
else:
    record["cpu"] = run(["sysctl", "-n", "machdep.cpu.brand_string"])
with open(out, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
print(f"  compiler {cxx or 'not found'}; cpu {record.get('cpu')}")
PYEOF

say "4. the build record"
"$PREFIX/bin/python" "$HERE/timing_records.py" meep-build-record \
  --python "$PREFIX/bin/python" --config-log "$RECORD_DIR/config.log" \
  --launcher "$PREFIX/bin/mpirun" --extra-json "$NATIVE_JSON" --require-single \
  --out "$RECORD_DIR/build_record.json"

say "DONE"
echo "Time it beside the reference with:"
echo "  python parity/meep_gpu/timing_ladder.py ... --meep-build native=$PREFIX/bin/python,$PREFIX/bin/mpirun,$RECORD_DIR/build_record.json"
