#!/usr/bin/env bash
# Build the reference environment on Linux x86_64, end to end: the conda and PyPI packages
# from the locks (with the certified CuPy, PyTorch and Triton), MEEP 1.33.0 from the
# release tarball with SINGLE-PRECISION fields, MPI, OpenMP and libGDSII and no
# floating-point contraction, the MEEP identity corpus, and the record of what was built.
# It is the one environment the package's tests, a certification round and the timing are
# run in on an NVIDIA host; the Apple silicon counterpart is build_meep_133_macos.sh.
#
#   bash parity/meep_gpu/build_meep_133_linux.sh 2>&1 | tee build_meep_133.log
#   conda activate meep-gpu-ref
#
# SINGLE PRECISION is not optional: the engine stores float32 and every certified figure
# was measured against a --enable-single MEEP; conda-forge's `pymeep` is a
# double-precision serial build, a supported way to USE the package but not a reference.
#
# NO CONTRACTION, and a portable instruction set. It compiles with "-O2 -ffp-contract=off
# -fno-fast-math -march=x86-64 -mtune=generic": the baseline x86-64 instruction set has no
# fused multiply-add, and the contraction is off besides. Step 4 refuses, before anything
# is compiled, flags under which the compilers' -march is not x86-64, and step 8 refuses
# the build unless the fused multiply-add census of the installed libmeep and _meep
# (tools/fma_census.py) finds none outside the compiler-rt division builtins.
#
# A CLEARED ENVIRONMENT. The script re-runs itself under `env -i`, so no compile flag,
# conda activation, PYTHONPATH or pip configuration of the calling shell reaches the
# build. Only HOME, USER, LOGNAME, TMPDIR, TERM, LANG, CUDA_VISIBLE_DEVICES (the build
# steps nothing on a GPU; an empty value hides every device) and the knobs below pass through;
# PATH becomes /usr/bin:/bin:/usr/sbin:/sbin (every tool of the environment is named by
# path or found in <prefix>/bin), Python runs without the user site directory and pip
# reads no configuration file and keeps no wheel cache (the pins are installed once, by
# hash; on a nearly full disk the cache is gigabytes that serve nothing). A knob this
# script no longer has is refused by name: MEEP_SIGMA_PATCH, MEEP_REFERENCE_SOLVE,
# MEEP_BUILD_ROLE, MEEP_FFMPEG (build_meep_133_macos.sh says why).
#
# THE ENVIRONMENT IS INSTALLED FROM THE LOCKS, NOT SOLVED. Step 1 installs
# environments/locks/meep-gpu-ref-linux-64.conda.txt (an explicit conda lock: every
# package by URL and MD5), step 1b refuses to go on unless the prefix holds exactly those
# artifacts, and step 7 installs meep-gpu-ref-linux-64.pip.txt (every PyPI package by
# version and wheel hash) and refuses any other. Its first 143 conda packages are the
# environment of the 2026-10-03 RTX A6000 certification round, verbatim; a solve depends
# on the day and on the host (for a host with glibc 2.35 it took sysroot_linux-64 2.34 and
# kernel-headers 5.14 where the lock has 2.28 and 4.18). environments/locks/README.md says
# what the locks hold, and environments/locks/recut_locks.py is how they are re-cut.
#
# THREE DEFECTS IN THE 1.33.0 RELEASE TARBALL, all handled below (INSTALL.md, "Building
# MEEP in single precision"):
#   1. configure.ac calls AX_CXX_MAXOPT while the bundled m4 defines AX_CC_MAXOPT.
#      Patched, then Autotools regenerated.
#   2. An OUT-OF-TREE --enable-single build installs cleanly and is DOUBLE precision:
#      compilation follows the quoted include to the tarball's pre-generated header,
#      where MEEP_SINGLE=0. This script builds IN-SOURCE and asserts the runtime value
#      after installing; the configure substitution is not evidence.
#   3. nompi HDF5/h5py make MEEP configure without H5Pset_fapl_mpio, which deadlocks
#      multi-rank output. The lock holds the mpi_openmpi builds of both.
#
# WHAT IT KEEPS in <prefix>/share/meep-gpu-build/: what build_meep_133_macos.sh keeps
# there (the locks, the identity corpus, its siblings and their SHA-256 lists, the configure log and
# Makefile, the build record, the census, the load commands, the ffmpeg link, the drift
# report, the digest smoke's rows, the identity verdicts in keep and flush,
# reference_build.json), and libcuda.json: the link to the GPU driver's libcuda.so.1 under
# <prefix>/lib/meep-gpu-libcuda/, the directory Triton's launcher links -lcuda against
# (TRITON_LIBCUDA_PATH), made when the host has an NVIDIA driver.
#
# REFRESHING THE CORPUS OF AN EXISTING PREFIX (--refresh-corpus): as in
# build_meep_133_macos.sh, which says what it does; it touches no MEEP, lock or package.
#
# Environment: CONDA_ROOT (default $HOME/miniforge3), MEEP_ENV_NAME (default meep-gpu-ref,
# or meep-gpu-ref-native for a timing control), MEEP_ENV_PREFIX (an absolute prefix in
# place of $CONDA_ROOT/envs/$MEEP_ENV_NAME), CONDA_PKGS_DIRS, MEEP_BUILD_ROOT,
# MEEP_BUILD_JOBS, MEEP_TARBALL (a local copy of the release tarball, used in place of the
# download and refused unless its SHA-256 is the pinned one), DRY_RUN=1 (print the plan,
# build nothing).
# Arguments: --refresh-corpus refreshes the corpus of the existing prefix (above) and does
# nothing else. --timing-control "FLAGS" builds a TIMING CONTROL with those compile flags
# (build_meep_133_native.sh): its census is recorded, not judged, the -march guard does not
# apply, and it is never an identity reference.
#
# Progress: every stage prints a flushed, timestamped marker.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PASS_THROUGH=(HOME USER LOGNAME TMPDIR TERM LANG CONDA_ROOT CONDA_PKGS_DIRS MEEP_ENV_NAME
              MEEP_ENV_PREFIX MEEP_BUILD_ROOT MEEP_BUILD_JOBS MEEP_TARBALL CUDA_VISIBLE_DEVICES DRY_RUN)
RETIRED=(MEEP_SIGMA_PATCH MEEP_REFERENCE_SOLVE MEEP_BUILD_ROLE MEEP_FFMPEG)

if [ "${MEEP_BUILD_ENVIRONMENT_CLEARED:-0}" != 1 ]; then
  for name in "${RETIRED[@]}"; do
    if [ -n "${!name+x}" ]; then
      echo "REFUSING: $name is retired (read the header of $(basename "$SELF")); unset it"
      exit 2
    fi
  done
  kept=()
  for name in "${PASS_THROUGH[@]}"; do
    if [ -n "${!name+x}" ]; then kept+=("$name=${!name}"); fi
  done
  dropped=()
  for name in CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CC CXX F77 FC MPICXX OMPI_CC OMPI_CXX OMPI_FC \
              CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH PYTHONPATH PYTHONHOME \
              PIP_CONFIG_FILE PIP_INDEX_URL; do
    if [ -n "${!name+x}" ]; then dropped+=("$name"); fi
  done
  if [ "${#dropped[@]}" -gt 0 ]; then
    echo "reference build: clearing ${dropped[*]} from the environment (env -i)"
  fi
  exec env -i "${kept[@]}" MEEP_BUILD_ENVIRONMENT_CLEARED=1 \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin bash "$SELF" "$@"
fi

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_CONFIG_FILE=/dev/null \
  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_NO_CACHE_DIR=1

MEEP_VERSION=1.33.0
MEEP_SHA256=bdabc0a112f669f2657fbdca22e31a2aeed515372e8ed56dc55197a4cd7ff0ab
MEEP_URL="https://github.com/NanoComp/meep/releases/download/v${MEEP_VERSION}/meep-${MEEP_VERSION}.tar.gz"
# The identity reference's compile flags: no contraction, no fast math, the baseline
# instruction set.
IDENTITY_FLAGS="-O2 -ffp-contract=off -fno-fast-math -march=x86-64 -mtune=generic"
ROLE=reference
FLAGS="$IDENTITY_FLAGS"
REFRESH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --timing-control)
      [ $# -ge 2 ] && [ -n "$2" ] || { echo "REFUSING: --timing-control needs the compile flags"; exit 2; }
      ROLE=timing-control; FLAGS="$2"; shift 2 ;;
    --refresh-corpus) REFRESH=1; shift ;;
    *) echo "REFUSING: unknown argument '$1' (the arguments are --timing-control FLAGS and --refresh-corpus)"; exit 2 ;;
  esac
done
DEFAULT_NAME=meep-gpu-ref
[ "$ROLE" = timing-control ] && DEFAULT_NAME=meep-gpu-ref-native
ENV_NAME="${MEEP_ENV_NAME:-$DEFAULT_NAME}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_PREFIX="${MEEP_ENV_PREFIX:-$CONDA_ROOT/envs/$ENV_NAME}"
BUILD_ROOT="${MEEP_BUILD_ROOT:-${TMPDIR:-/tmp}/meep_build_133_linux}"
JOBS="${MEEP_BUILD_JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
SCRIPT_DIR="$(dirname "$SELF")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCK_DIR="$REPO_ROOT/environments/locks"
LOCK_CONDA="$LOCK_DIR/meep-gpu-ref-linux-64.conda.txt"
LOCK_PIP="$LOCK_DIR/meep-gpu-ref-linux-64.pip.txt"
RECORD_DIR="$ENV_PREFIX/share/meep-gpu-build"
CORPUS_ROOT="$RECORD_DIR/meep-${MEEP_VERSION}/python"
CORPUS_LIST="$RECORD_DIR/corpus_sha256.txt"
CORPUS_SIBLINGS="$RECORD_DIR/corpus_siblings_sha256.txt"
HELPER="$SCRIPT_DIR/build_meep_133_record.py"

say() { printf '\n=== %s === %s\n' "$1" "$(date -u +%FT%TZ)"; }
py() { "$ENV_PREFIX/bin/python" -s "$@"; }
helper() { "$ENV_PREFIX/bin/python" -s -I "$HELPER" "$@"; }
# The release tarball, downloaded or copied from MEEP_TARBALL, refused unless its SHA-256 is
# the pinned one.
fetch_tarball() {
  if [ -n "${MEEP_TARBALL:-}" ]; then
    helper fetch "$MEEP_URL" "$MEEP_SHA256" "$1" --from "$MEEP_TARBALL"
  else
    helper fetch "$MEEP_URL" "$MEEP_SHA256" "$1"
  fi
}
# The installed MEEP images (the files, not their symlinks): libmeep and _meep. The
# environment's Python ships lib/python3.1 as a link to lib/python3.10 (for build systems
# that read the version as sys.version[:3]), so the glob reaches _meep under both names:
# a python3.* directory that is a link is skipped.
meep_images() {
  local image
  for image in "$ENV_PREFIX"/lib/libmeep.* "$ENV_PREFIX"/lib/python3.*/site-packages/meep/_meep.*; do
    case "$image" in *.a|*.la) continue ;; esac
    case "$image" in
      "$ENV_PREFIX"/lib/python3.*/*) if [ -L "${image%%/site-packages/*}" ]; then continue; fi ;;
    esac
    if [ -f "$image" ] && [ ! -L "$image" ]; then echo "$image"; fi
  done
}

if [ "$REFRESH" = 1 ]; then
  say "R. refresh the identity corpus of an existing prefix (MEEP, the locks and the packages are not touched)"
  [ "$(uname -s)/$(uname -m)" = "Linux/x86_64" ] || { echo "  this script builds for Linux x86_64; on a Mac use build_meep_133_macos.sh"; exit 1; }
  echo "  prefix       $ENV_PREFIX"
  echo "  record       $RECORD_DIR/reference_build.json"
  echo "  build root   $BUILD_ROOT/corpus-refresh"
  if [ ! -x "$ENV_PREFIX/bin/python" ] || [ ! -f "$RECORD_DIR/reference_build.json" ]; then
    echo "  REFUSING: $ENV_PREFIX is not a reference environment this script built (no"
    echo "  bin/python or no share/meep-gpu-build/reference_build.json); build it instead."
    exit 1
  fi
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "  DRY RUN: nothing is refreshed"
    exit 0
  fi
  REFRESH_ROOT="$BUILD_ROOT/corpus-refresh"
  rm -rf "$REFRESH_ROOT"; mkdir -p "$REFRESH_ROOT"
  fetch_tarball "$REFRESH_ROOT/meep.tar.gz"
  helper extract "$REFRESH_ROOT/meep.tar.gz" "$REFRESH_ROOT/source"
  helper refresh-corpus "$REFRESH_ROOT/source/meep-${MEEP_VERSION}/python" "$RECORD_DIR" \
    "$REFRESH_ROOT/staging" --tarball-sha256 "$MEEP_SHA256" \
    --corpus-root "share/meep-gpu-build/meep-${MEEP_VERSION}/python" \
    --corpus-list "$CORPUS_LIST" --corpus-siblings "$CORPUS_SIBLINGS"
  helper verify-corpus "$CORPUS_ROOT" "$CORPUS_LIST" "$CORPUS_SIBLINGS"
  rm -rf "$REFRESH_ROOT"
  say "DONE"
  echo "The record: $RECORD_DIR/reference_build.json (its corpus entry; what changed:"
  echo "$RECORD_DIR/corpus_refresh.json)"
  exit 0
fi

say "0. plan, and refuse to clobber an existing prefix"
[ "$(uname -s)/$(uname -m)" = "Linux/x86_64" ] || { echo "  this script builds for Linux x86_64; on a Mac use build_meep_133_macos.sh"; exit 1; }
echo "  role         $ROLE"
echo "  prefix       $ENV_PREFIX"
echo "  conda        $CONDA_ROOT/bin/conda"
echo "  build root   $BUILD_ROOT"
echo "  CFLAGS       $FLAGS"
echo "  CXXFLAGS     $FLAGS"
echo "  PATH         $PATH"
echo "  HOME         $HOME"
[ -x "$CONDA_ROOT/bin/conda" ] || { echo "  no conda at $CONDA_ROOT (set CONDA_ROOT)"; exit 1; }
for lock in "$LOCK_CONDA" "$LOCK_PIP" "$HELPER"; do
  [ -f "$lock" ] || { echo "  missing $lock (run this script from a checkout of the repository)"; exit 1; }
done
# The lock's compilers target glibc 2.28, which an explicit install does not check;
# checked here, before anything is installed.
GLIBC="$( (getconf GNU_LIBC_VERSION 2>/dev/null || true) | awk '{print $2}')"
echo "  host glibc   ${GLIBC:-not read} (the lock needs 2.28 or later)"
[ -n "$GLIBC" ] || { echo "  REFUSING: the host's glibc version could not be read (getconf GNU_LIBC_VERSION); the lock needs glibc 2.28 or later"; exit 1; }
if [ "$(printf '%s\n%s\n' 2.28 "$GLIBC" | sort -V | head -1)" != 2.28 ]; then
  echo "  REFUSING: glibc $GLIBC is older than 2.28"; exit 1
fi
if [ -e "$ENV_PREFIX" ]; then
  echo "  $ENV_PREFIX already exists. This script never writes into an existing"
  echo "  prefix; remove it yourself, or set MEEP_ENV_NAME or MEEP_ENV_PREFIX."
  exit 1
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "  DRY RUN: nothing is built"
  exit 0
fi
export CFLAGS="$FLAGS" CXXFLAGS="$FLAGS"

say "1. create the environment from the conda lock"
# The first 143 packages are the conda environment the 2026-10-03 RTX A6000 round ran in,
# exported verbatim: the compilers MEEP is built with (gcc 15.2.0), NumPy 2.2.6 (fixed
# BEFORE the build, because MEEP's SWIG module compiles against NumPy's C interface), every
# library MEEP links, Open MPI 5.0.8 and h5py 3.15.1 (the versions the certified CuPy,
# CUDA 11.8, can sit beside), the CUDA compiler NVRTC 11.8.89 from cudatoolkit 11.8,
# OpenBLAS 0.3.34 (pthreads) and the compiler runtimes at 16.2.0. The rest is what the
# build and the round need beside them: libGDSII (which configure links when it finds it),
# gdspy, git, and binutils' unprefixed names. --no-default-packages: conda refuses an
# explicit file to which create_default_packages would add names.
echo "  installing $(grep -c '^https://' "$LOCK_CONDA") packages from $LOCK_CONDA, without a solve"
"$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" --no-default-packages --file "$LOCK_CONDA"
# The Fortran compiler is not optional although MEEP has no Fortran in it: configure
# runs AC_F77_WRAPPERS to work out the BLAS/LAPACK name-mangling scheme. Nor is
# matplotlib, although nothing here plots: MEEP 1.33.0's simulation module imports it
# when meep is imported.

say "1b. MANDATORY: the environment is the lock, artifact for artifact"
# Every conda package of the new prefix, by URL and MD5, against every line of the lock,
# before anything is built on it: a package missing, added or of another build refuses.
mkdir -p "$RECORD_DIR"
LOCK_DIFF="$(diff <(grep '^https://' "$LOCK_CONDA" | sort) \
  <("$CONDA_ROOT/bin/conda" list --explicit --md5 -p "$ENV_PREFIX" | grep '^https://' | sort) || true)"
if [ -n "$LOCK_DIFF" ]; then
  echo "  the prefix differs from the lock ('<' only in the lock, '>' only in the prefix):"
  echo "$LOCK_DIFF" | sed 's/^/    /'
  echo "  REFUSING: $ENV_PREFIX is not the reference environment; nothing was built on it."
  exit 1
fi
LOCKED=$(grep -c '^https://' "$LOCK_CONDA")
echo "  $LOCKED of $LOCKED conda packages are the lock's artifacts (URL and MD5), and the prefix holds no other"
cp "$LOCK_CONDA" "$LOCK_PIP" "$LOCK_DIR/numerics-relevant.json" "$LOCK_DIR/additions.json" "$RECORD_DIR/"
echo "  kept the lock files in $RECORD_DIR"

say "2. fetch and verify the release tarball; keep its python/ tree and the corpus siblings as the identity corpus"
rm -rf "$BUILD_ROOT"; mkdir -p "$BUILD_ROOT"
fetch_tarball "$BUILD_ROOT/meep.tar.gz"
helper extract "$BUILD_ROOT/meep.tar.gz" "$BUILD_ROOT"
# The corpus is copied BEFORE anything is built in the source tree, so it is the tarball's
# bytes: examples/ and tests/ of python/, which the round's lifts, the census and the
# identity corpus read through MEEP_GPU_CORPUS_ROOT=<prefix>/share/meep-gpu-build/
# meep-1.33.0/python, and the siblings beside python/ that corpus rows read
# (build_meep_133_record.py's CORPUS_SIBLINGS), kept where the rows look for them.
helper keep-corpus "$BUILD_ROOT/meep-${MEEP_VERSION}/python" "$CORPUS_ROOT" "$CORPUS_LIST" \
  "$CORPUS_SIBLINGS"
cd "$BUILD_ROOT/meep-${MEEP_VERSION}"

say "3. patch AX_CXX_MAXOPT -> AX_CC_MAXOPT (defect 1) and regenerate autotools"
if grep -q 'AX_CXX_MAXOPT' configure.ac; then
  sed -i 's/AX_CXX_MAXOPT/AX_CC_MAXOPT/' configure.ac
  echo "  patched configure.ac"
else
  echo "  configure.ac does not call AX_CXX_MAXOPT: upstream may have fixed it"
fi
export PATH="$ENV_PREFIX/bin:$PATH"
autoreconf -fi

say "4. configure IN-SOURCE with --enable-single (defect 2), MPI, OpenMP and libGDSII"
# The environment is never activated: the build runs on explicit paths, so the conda
# compilers are named here and handed to Open MPI's wrappers as well.
first() { ls $1 2>/dev/null | head -1; }
CCX="$(first "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc")"
CXXX="$(first "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++")"
FCX="$(first "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gfortran")"
for tool in "$CCX" "$CXXX" "$FCX"; do
  [ -n "$tool" ] || { echo "  a conda compiler is missing from $ENV_PREFIX/bin (step 1)"; exit 1; }
done
export OMPI_CC="$CCX" OMPI_CXX="$CXXX" OMPI_FC="$FCX"
export CC="$CCX" CXX="$CXXX" MPICXX="$ENV_PREFIX/bin/mpicxx"
export F77="$FCX" FC="$FCX"
export CPPFLAGS="-I$ENV_PREFIX/include"
export LDFLAGS="-L$ENV_PREFIX/lib -Wl,-rpath,$ENV_PREFIX/lib"
export PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig"
export PYTHON="$ENV_PREFIX/bin/python"
# A reference has no fused multiply-add because x86-64 has none: refuse flags under which
# either compiler targets more (a -march of haswell or later has FMA3). The census of step
# 8 judges the result; this stops the build before it is compiled. The reader takes the
# compiler's whole answer: gcc exits 2 when the pipe it writes to closes early, which
# under pipefail would stop the build here without a word (MEASURED 2026-10-14 with the
# lock's gcc 15.2.0 and a reader that stopped at the -march line).
if [ "$ROLE" = reference ]; then
  for pair in "$CCX|${CFLAGS:-}" "$CXXX|${CXXFLAGS:-}"; do
    compiler="${pair%%|*}"; flags="${pair#*|}"
    MARCH="$("$compiler" $flags -Q --help=target 2>/dev/null | awk '$1 == "-march=" && m == "" {m = $2} END {print m}')"
    echo "  $(basename "$compiler") ${flags:-(no flags)}: -march resolves to ${MARCH:-not read}"
    [ "$MARCH" = x86-64 ] || { echo "  REFUSING: a reference is compiled for -march=x86-64 (no FMA instruction); these flags target ${MARCH:-an unread architecture}"; exit 1; }
  done
fi
# NO separate build/ directory: see defect 2.
./configure \
  --prefix="$ENV_PREFIX" \
  --enable-shared --enable-single --enable-portable-binary \
  --with-mpi --with-openmp --without-scheme \
  --with-libctl="$ENV_PREFIX/share/libctl" \
  CC="$CC" CXX="$CXX" MPICXX="$MPICXX" F77="$F77" \
  PYTHON="$ENV_PREFIX/bin/python"
# MANDATORY: configure found libGDSII (AC_CHECK_LIB(GDSII, libGDSIIExists)); without it
# MEEP's GDSII functions abort and two corpus rows cannot run.
grep -q '^ac_cv_lib_GDSII_libGDSIIExists=yes$' config.log || {
  echo "  REFUSING: configure did not find libGDSII (config.log: ac_cv_lib_GDSII_libGDSIIExists)"; exit 1; }
echo "  libGDSII: found by configure"

say "5. build (-j$JOBS)"
make -j"$JOBS"

say "6. install, and record what libmeep and _meep link"
make install
IMAGES=()
while IFS= read -r image; do IMAGES+=("$image"); done < <(meep_images)
[ "${#IMAGES[@]}" = 2 ] || { echo "  REFUSING: expected libmeep and _meep, found ${#IMAGES[@]} image(s)"; exit 1; }
helper load-commands "$RECORD_DIR/load_commands_installed.json" "${IMAGES[@]}"

say "7. the PyPI packages, from the pip lock"
# PyTorch 2.5.1, the release that requires Triton 3.1.0 (the certified Triton; never
# `--upgrade` PyTorch here, it replaces Triton), their dependencies with the CUDA 12.4
# libraries PyTorch loads, autograd (MEEP's adjoint module imports it), the corpus-only
# packages (shapely, nlopt, imageio-ffmpeg) and llvmlite (the test suite's LLVM parser),
# each by version and SHA-256, with no dependency resolution (--no-deps); then the
# PyMieScatt stand-in from its pinned sdist and patch; pip check confirms that nothing is
# missing, and the last check refuses any PyPI package that is not the lock's or the
# stand-in. The lock is installed one pin at a time: pip keeps every wheel it downloads
# until the command exits, so one install of the whole lock would hold all of its wheels
# (3.1 GB, 0.9 GB of them PyTorch's) beside the 5.4 GB they unpack to, the build's peak on
# a nearly full disk. Each part is the lock's option lines and one of its pins, unchanged;
# pip check and the drift tool's --require-pip-lock below judge the installed set whole.
helper pip-lock-parts "$LOCK_PIP" "$BUILD_ROOT/pip-lock-parts"
for part in "$BUILD_ROOT"/pip-lock-parts/*.txt; do
  py -m pip install --no-deps -r "$part"
done
py "$SCRIPT_DIR/rounds/1.0/build_pymiescatt_standin.py" --python "$ENV_PREFIX/bin/python" --install
py -m pip check
py "$REPO_ROOT/tools/compare_reference_environment.py" \
    --prefix "$ENV_PREFIX" --require-pip-lock

say "7b. ffmpeg: the imageio-ffmpeg executable, linked into the prefix"
# test_animation_output is skipped, and its corpus row refused by name, unless `which
# ffmpeg` succeeds on the reference PATH (<prefix>/bin:/usr/bin:/bin). The executable
# comes from the pinned imageio-ffmpeg wheel and is statically linked: it installs no
# shared library, and MEEP's images keep the load commands they were installed with
# (checked here).
helper link-ffmpeg "$ENV_PREFIX" "$RECORD_DIR/ffmpeg.json"
helper load-commands "$RECORD_DIR/load_commands.json" "${IMAGES[@]}"
helper same-load-commands "$RECORD_DIR/load_commands_installed.json" "$RECORD_DIR/load_commands.json"

say "7c. the GPU driver's libcuda.so.1, for Triton's launcher"
# Triton compiles its launcher with the C compiler and links it with -lcuda, and a driver
# installs libcuda.so.1 only. The link lives in its own directory, which a round passes as
# TRITON_LIBCUDA_PATH (and on LD_LIBRARY_PATH); without a driver on this host nothing is
# made and the record says so.
helper link-libcuda "$ENV_PREFIX" "$RECORD_DIR/libcuda.json"

say "8. MANDATORY assertions: the reference this environment claims to be"
PYTHONPATH="$REPO_ROOT" py - <<'PYEOF'
import sys
import meep as mp
from meep_gpu import backends
flushed = backends.subnormals_flushed()     # read before any other library loads
import numpy, cupy, triton
import ctypes, os
# NVRTC's version, by NVRTC's own call on the library CuPy's CUDA 11 build opens
# (<prefix>/lib/libnvrtc.so.11.2). cupy.cuda.nvrtc.getVersion() first asks the CUDA
# runtime for its major version, to choose that library, and with every device hidden
# (CUDA_VISIBLE_DEVICES empty) the runtime answers cudaErrorNoDevice; nvrtcVersion needs
# no device.
def nvrtc_version(library):
    try:
        nvrtc = ctypes.CDLL(library)
    except OSError as error:
        return f"not loaded ({error})"
    major, minor = ctypes.c_int(), ctypes.c_int()
    status = nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    return f"{major.value}.{minor.value}" if status == 0 else f"nvrtcVersion status {status}"
nvrtc = nvrtc_version(os.path.join(sys.prefix, "lib", "libnvrtc.so.11.2"))
checks = [
    ("MEEP version", mp.__version__, "1.33.0"),
    ("single precision", mp.is_single_precision() and mp.get_realnum_size() == 4, True),
    ("MPI build", mp.with_mpi(), True),
    ("subnormals flushed after import meep", flushed, True),
    ("CuPy", cupy.__version__, "13.5.1"),
    ("NVRTC (cudatoolkit 11.8)", nvrtc, "11.8"),
    ("Triton", triton.__version__, "3.1.0"),
    ("NumPy", numpy.__version__, "2.2.6"),
]
failed = []
for name, got, want in checks:
    print(f"  {name:38} {got!s:12} {'ok' if got == want else 'EXPECTED ' + str(want)}", flush=True)
    if got != want:
        failed.append(name)
if failed:
    print(f"\n  FAILED: {failed}. 'single precision' False is defect 2: rebuild in-source.")
    sys.exit(1)
print("\n  reference environment OK")
PYEOF
# MEEP's GDSII support, in a process of its own (MEEP aborts without libGDSII): the layers
# of the corpus's own coupler.gds, as plain integers (MEEP returns a NumPy int32 array,
# whose elements NumPy 2 prints as np.int32(0) inside a list).
GDSII="$(cd "$CORPUS_ROOT/examples" && py -c 'import meep as mp; mp.verbosity(0); print("LAYERS", [int(layer) for layer in mp.get_GDSII_layers("coupler.gds")])' 2>&1 | grep '^LAYERS' || true)"
echo "  GDSII                                  ${GDSII:-not read}"
[ "$GDSII" = "LAYERS [0, 1, 2, 3, 4, 5, 31, 32]" ] || { echo "  REFUSING: MEEP does not read the corpus's coupler.gds (no libGDSII support)"; exit 1; }
# Triton's launcher, compiled the way Triton compiles it, with the environment's C compiler
# (a round passes it as CC), against the libcuda link of step 7c. CPU only: nothing runs.
LIBCUDA_DIR="$ENV_PREFIX/lib/meep-gpu-libcuda"
if [ -d "$LIBCUDA_DIR" ]; then
  TRITON_DIR="$(py -c 'import os, triton; print(os.path.dirname(triton.__file__))')"
  PY_INCLUDE="$(py -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
  "$CCX" "$TRITON_DIR/backends/nvidia/driver.c" -O3 -shared -fPIC -o "$BUILD_ROOT/cuda_utils.so" \
    -lcuda -L"$LIBCUDA_DIR" -I"$TRITON_DIR/backends/nvidia/include" -I"$PY_INCLUDE"
  echo "  Triton launcher                        compiles and links with $(basename "$CCX") against $LIBCUDA_DIR"
else
  echo "  Triton launcher                        not compiled here: no NVIDIA driver on this host (step 7c)"
fi
# The drift report a user runs (tools/compare_reference_environment.py): every
# numerics-relevant package against the lock, kept as JSON beside the build.
py "$REPO_ROOT/tools/compare_reference_environment.py" --require-reference
py "$REPO_ROOT/tools/compare_reference_environment.py" --json > "$RECORD_DIR/reference_environment.json"
# MANDATORY: the fused multiply-add census of the MEEP images a process of this
# environment loads. The census is the criterion, not the flags: a reference has no fused
# multiply-add outside the compiler-rt division builtins. A timing control records it
# without judging it.
if [ "$ROLE" = timing-control ]; then
  py "$REPO_ROOT/tools/fma_census.py" --python "$ENV_PREFIX/bin/python" \
    --out "$RECORD_DIR/fma_census.json"
else
  py "$REPO_ROOT/tools/fma_census.py" --python "$ENV_PREFIX/bin/python" \
    --out "$RECORD_DIR/fma_census.json" --require-none
fi

say "9. keep the build's record in the prefix"
# The build root is under $TMPDIR, which a reboot may empty. config.log carries the
# configure line, the Makefile the compiler flags, and build_record.json what the
# interpreter loads (timing_records.py meep-build-record: version, precision, the sha256 of
# the MEEP images, the contraction the flags ask for, their census and the host's math
# library: glibc, libm and the CPU features glibc picks its implementations by).
# tools/check_install.py --require-identity-reference reads it.
cp "$BUILD_ROOT/meep-${MEEP_VERSION}/config.log" "$RECORD_DIR/config.log"
cp "$BUILD_ROOT/meep-${MEEP_VERSION}/Makefile" "$RECORD_DIR/Makefile"
if [ "$ROLE" = reference ]; then
  RECORD_OPTIONS=(--require-identity)
else
  # a timing control also records what its flags made of this CPU
  helper compiler-facts "$CXXX" "$FLAGS" "$RECORD_DIR/native.json"
  RECORD_OPTIONS=(--extra-json "$RECORD_DIR/native.json")
fi
py "$SCRIPT_DIR/timing_records.py" meep-build-record \
  --python "$ENV_PREFIX/bin/python" --config-log "$RECORD_DIR/config.log" \
  --launcher "$ENV_PREFIX/bin/mpirun" --require-single "${RECORD_OPTIONS[@]}" \
  --out "$RECORD_DIR/build_record.json"

say "10. MANDATORY for a reference: MEEP's own words on the four probe cases"
# parity/meep_gpu/meep_identity_smoke.py steps MEEP alone on four cases in keep mode
# (subnormals kept: MEEP's set_zero_subnormals(False) after import, since MEEP flushes on
# import here) and compares the SHA-256 of every stored word at the compared steps with the
# recorded set; it exits 1 unless every digest reproduces and its planted 1-ulp flip is
# caught.
if [ "$ROLE" = timing-control ]; then
  echo "  skipped: a timing control is not an identity reference"
else
  py "$SCRIPT_DIR/meep_identity_smoke.py" \
    --mode keep --rows "$RECORD_DIR/identity_smoke.jsonl"
fi

say "11. MANDATORY for a reference: the identity verdicts, keep and flush"
# tools/check_install.py's identity check judges the build a host runs, in a process that
# imports NumPy and MEEP and nothing else; the helper takes that verdict alone (the rest
# of check_install steps a simulation on the GPU, which a build does not). Both modes must
# pass: keep (MEEP's set_zero_subnormals(False) after import) and flush (MEEP's own flush
# on import, a reference at OMP_NUM_THREADS=1).
if [ "$ROLE" = timing-control ]; then
  echo "  skipped: a timing control is not an identity reference"
else
  for MODE in keep flush; do
    set +e
    if [ "$MODE" = flush ]; then
      OMP_NUM_THREADS=1 helper identity "$REPO_ROOT/tools/check_install.py" flush \
        "$RECORD_DIR/identity_flush.txt"
    else
      helper identity "$REPO_ROOT/tools/check_install.py" keep "$RECORD_DIR/identity_keep.txt"
    fi
    STATUS=$?
    set -e
    echo "$STATUS" > "$RECORD_DIR/identity_$MODE.status"
    if [ "$STATUS" = 0 ]; then
      echo "pass" > "$RECORD_DIR/identity_$MODE.outcome"
    else
      echo "fail" > "$RECORD_DIR/identity_$MODE.outcome"
      echo "  REFUSING: the identity verdict in $MODE is not a pass (exit $STATUS)"
      exit 1
    fi
  done
fi

say "12. the programs the round runs, and the reference build record"
helper tools "$ENV_PREFIX" linux-64 "$RECORD_DIR/tools.json"
helper verify-corpus "$CORPUS_ROOT" "$CORPUS_LIST" "$CORPUS_SIBLINGS"
CONFIGURE_LINE="$(sed -n 's/^  \$ \(\.\/configure .*\)$/\1/p' "$RECORD_DIR/config.log" | head -1)"
helper manifest "$RECORD_DIR" "$RECORD_DIR/reference_build.json" \
  --platform linux-64 --role "$ROLE" --tarball-url "$MEEP_URL" --tarball-sha256 "$MEEP_SHA256" \
  --cflags "$CFLAGS" --cxxflags "$CXXFLAGS" --configure "$CONFIGURE_LINE" \
  --corpus-root "share/meep-gpu-build/meep-${MEEP_VERSION}/python" --corpus-list "$CORPUS_LIST" \
  --corpus-siblings "$CORPUS_SIBLINGS"

say "DONE"
echo "Activate it with:  conda activate $ENV_PREFIX"
echo "The identity corpus: MEEP_GPU_CORPUS_ROOT=$CORPUS_ROOT"
echo "Triton's launcher:   CC=$CCX TRITON_LIBCUDA_PATH=$LIBCUDA_DIR (when it exists)"
echo "The record: $RECORD_DIR/reference_build.json"
echo "Do not install the package into this environment: a certification round imports it"
echo "from the checkout, and an installed copy would be invisible to the record of what ran."
