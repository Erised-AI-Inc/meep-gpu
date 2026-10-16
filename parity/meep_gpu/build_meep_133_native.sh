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
# (docs/development/certification.md, "The MEEP a round is measured against"). Native
# flags let the compiler fuse multiply-adds (-mcpu=native on Apple silicon, an FMA3
# -march on most x86 hosts), so it is never an identity reference either: the platform
# script's --timing-control records the fused multiply-add census of what it built
# without judging it, and skips the identity checks.
#
# HOW. It runs the platform's reference script with --timing-control and these flags, so
# the environment is the same lock, MEEP the same tarball and every defect that script works
# around is worked around here too; only the compile flags and the role differ:
#
#   Linux x86_64 (gcc):   CFLAGS = CXXFLAGS = "-O3 -march=native"
#   macOS arm64 (clang):  CFLAGS = CXXFLAGS = "-O3 -mcpu=native"
#
# --enable-portable-binary, which the reference scripts pass, changes only what
# AX_CC_MAXOPT picks when CFLAGS is UNSET; with the flags set here it has no effect, and the
# build record says so.
#
# WHAT IT WRITES: what the platform script keeps in <prefix>/share/meep-gpu-build/, with
# native.json (the compiler, what -march / -mcpu native resolved to, the CPU) inside
# build_record.json (timing_records.py meep-build-record --extra-json).
#
# USAGE (from the repository root; every stage prints a flushed, timestamped marker):
#   bash parity/meep_gpu/build_meep_133_native.sh 2>&1 | tee build_meep_133_native.log
# Environment: MEEP_ENV_NAME (default meep-gpu-ref-native), MEEP_ENV_PREFIX, CONDA_ROOT,
# CONDA_PKGS_DIRS, MEEP_BUILD_ROOT, MEEP_BUILD_JOBS, NATIVE_FLAGS (overrides the flags
# above), DRY_RUN=1 (print what would run, run nothing).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MEEP_ENV_NAME="${MEEP_ENV_NAME:-meep-gpu-ref-native}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
export CONDA_ROOT
PREFIX="${MEEP_ENV_PREFIX:-$CONDA_ROOT/envs/$MEEP_ENV_NAME}"
OS="$(uname -s)"
ARCH="$(uname -m)"

say() { printf '\n=== native %s === %s\n' "$1" "$(date -u +%FT%TZ)"; }

case "$OS/$ARCH" in
  Linux/x86_64)
    DELEGATE="$HERE/build_meep_133_linux.sh"
    DEFAULT_FLAGS="-O3 -march=native"
    BUILD_ROOT="${MEEP_BUILD_ROOT:-${TMPDIR:-/tmp}/meep_build_133_native_linux}"
    ;;
  Darwin/arm64)
    DELEGATE="$HERE/build_meep_133_macos.sh"
    DEFAULT_FLAGS="-O3 -mcpu=native"
    BUILD_ROOT="${MEEP_BUILD_ROOT:-/tmp/meep_build_133_native_macos}"
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
[ -n "${MEEP_ENV_PREFIX:-}" ] && export MEEP_ENV_PREFIX
RECORD_DIR="$PREFIX/share/meep-gpu-build"

say "0. plan"
echo "  delegate     $DELEGATE"
echo "  prefix       $PREFIX"
echo "  build root   $BUILD_ROOT"
echo "  CFLAGS       $FLAGS"
echo "  CXXFLAGS     $FLAGS"
echo "  role         timing-control (the census is recorded, not judged)"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "  DRY RUN: nothing is built"
  exit 0
fi

say "1. build through the platform's reference script, as a timing control"
bash "$DELEGATE" --timing-control "$FLAGS"

say "DONE"
echo "Time it beside the reference with:"
echo "  python parity/meep_gpu/timing_ladder.py ... --meep-build native=$PREFIX/bin/python,$PREFIX/bin/mpirun,$RECORD_DIR/build_record.json"
