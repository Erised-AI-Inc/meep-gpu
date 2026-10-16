#!/usr/bin/env bash
# Build the reference environment on Apple silicon, end to end: the conda and PyPI
# packages from the locks, MEEP 1.33.0 from the release tarball with SINGLE-PRECISION
# fields, MPI and libGDSII and no floating-point contraction, the MEEP identity corpus,
# and the record of what was built. It is the one environment the package's tests, a
# certification round and the timing are run in on a Mac; the Linux x86_64 counterpart
# is build_meep_133_linux.sh.
#
#   bash parity/meep_gpu/build_meep_133_macos.sh 2>&1 | tee build_meep_133.log
#   conda activate meep-gpu-ref
#
# SINGLE PRECISION is not optional: the engine stores float32 and every certified figure
# was measured against a --enable-single MEEP; conda-forge's `pymeep` is a
# double-precision build, a supported way to USE the package but not a reference.
#
# NO CONTRACTION. It compiles with "-O2 -ffp-contract=off -fno-fast-math", so no multiply
# and add is fused into one rounding, and step 8 refuses the build unless the fused
# multiply-add census of the installed libmeep and _meep (tools/fma_census.py) finds none
# outside the compiler-rt division builtins. clang contracts a*b+c by default: the MEEP
# the Metal kernels of 0.9.x were certified against, compiled with configure's "-g -O2",
# carries 1,253 fused operations in libmeep, 634 of them in the 16 field-update kernels
# (MEASURED by the census, 2026-10-06).
#
# A CLEARED ENVIRONMENT. The script re-runs itself under `env -i`, so no compile flag,
# conda activation, PYTHONPATH or pip configuration of the calling shell reaches the
# build. Only HOME, USER, LOGNAME, TMPDIR, TERM, LANG and the knobs below pass through;
# PATH becomes /usr/bin:/bin:/usr/sbin:/sbin (every tool of the environment is named by
# path or found in <prefix>/bin), Python runs without the user site directory and pip
# reads no configuration file and keeps no wheel cache (the pins are installed once, by
# hash; on a nearly full disk the cache is gigabytes that serve nothing). A knob this
# script no longer has is refused by name: MEEP_SIGMA_PATCH (no reference build carries
# the sigma-reader patch), MEEP_REFERENCE_SOLVE (the environment is installed from the
# locks, never solved), MEEP_BUILD_ROLE (a timing control is asked for with
# --timing-control) and MEEP_FFMPEG (ffmpeg comes from the pip lock).
#
# THE ENVIRONMENT IS INSTALLED FROM THE LOCKS, NOT SOLVED. Step 1 installs
# environments/locks/meep-gpu-ref-osx-arm64.conda.txt (an explicit conda lock: every
# package by URL and MD5), step 1b refuses to go on unless the prefix holds exactly those
# artifacts, and step 7 installs meep-gpu-ref-osx-arm64.pip.txt (every PyPI package by
# version and wheel hash) and refuses any other. environments/locks/README.md says what
# the locks hold, and environments/locks/recut_locks.py is how they are re-cut.
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
# WHAT IT KEEPS in <prefix>/share/meep-gpu-build/: the lock files it installed, the MEEP
# identity corpus (the tarball's python/examples and python/tests, under
# meep-1.33.0/python/, with corpus_sha256.txt listing every file's SHA-256, and the corpus
# siblings, the files outside python/ that corpus rows read, under meep-1.33.0/ where the
# rows look for them, listed in corpus_siblings_sha256.txt), config.log,
# the Makefile, build_record.json, fma_census.json, the load commands of libmeep and _meep,
# the ffmpeg link, the drift report, the digest smoke's rows, the identity verdicts in
# keep and flush, and reference_build.json, which names all of them.
#
# REFRESHING THE CORPUS OF AN EXISTING PREFIX (--refresh-corpus). A prefix built before the
# build kept a file it keeps now (the corpus siblings: tests/cyl-ellipsoid-eps-ref.h5, which
# two test_simulation.py rows read from beside python/) is brought to what a build keeps by
# this script itself, never by hand: the pinned tarball is fetched (or MEEP_TARBALL copied)
# and verified by SHA-256, the corpus a build would keep is staged from it, and what the
# prefix lacks is copied in; a kept file with other bytes, or a corpus file the tarball does
# not hold, refuses the refresh before anything is written (rebuild the environment then).
# Then corpus_sha256.txt and corpus_siblings_sha256.txt are rewritten, the "corpus" entry of
# reference_build.json and nothing else in it is replaced, the corpus is verified, and
# corpus_refresh.json records the record's SHA-256 before and after. MEEP, the locks and
# every package of the prefix are not touched, and nothing is removed from it.
#
# Environment: CONDA_ROOT (default $HOME/miniforge3), MEEP_ENV_NAME (default meep-gpu-ref,
# or meep-gpu-ref-native for a timing control), MEEP_ENV_PREFIX (an absolute prefix in
# place of $CONDA_ROOT/envs/$MEEP_ENV_NAME), CONDA_PKGS_DIRS, MEEP_BUILD_ROOT,
# MEEP_BUILD_JOBS, MEEP_TARBALL (a local copy of the release tarball, used in place of the
# download and refused unless its SHA-256 is the pinned one), DRY_RUN=1 (print the plan,
# build nothing).
# Arguments: --refresh-corpus refreshes the corpus of the existing prefix (above) and does
# nothing else. --timing-control "FLAGS" builds a TIMING CONTROL with those compile flags
# (build_meep_133_native.sh): its census is recorded, not judged, and it is never an
# identity reference.
#
# Progress: every stage prints a flushed, timestamped marker.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PASS_THROUGH=(HOME USER LOGNAME TMPDIR TERM LANG CONDA_ROOT CONDA_PKGS_DIRS MEEP_ENV_NAME
              MEEP_ENV_PREFIX MEEP_BUILD_ROOT MEEP_BUILD_JOBS MEEP_TARBALL DRY_RUN)
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
              CONDA_PREFIX CONDA_DEFAULT_ENV SDKROOT MACOSX_DEPLOYMENT_TARGET PYTHONPATH \
              PYTHONHOME PIP_CONFIG_FILE PIP_INDEX_URL; do
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
# The identity reference's compile flags: no contraction, no fast math.
IDENTITY_FLAGS="-O2 -ffp-contract=off -fno-fast-math"
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
BUILD_ROOT="${MEEP_BUILD_ROOT:-${TMPDIR:-/tmp}/meep_build_133_macos}"
JOBS="${MEEP_BUILD_JOBS:-6}"
SCRIPT_DIR="$(dirname "$SELF")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCK_DIR="$REPO_ROOT/environments/locks"
LOCK_CONDA="$LOCK_DIR/meep-gpu-ref-osx-arm64.conda.txt"
LOCK_PIP="$LOCK_DIR/meep-gpu-ref-osx-arm64.pip.txt"
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
# environment's Python ships lib/python3.1 as a link to lib/python3.12 (for build systems
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
  [ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || { echo "  this script builds for Apple silicon (Darwin/arm64); on Linux x86_64 use build_meep_133_linux.sh"; exit 1; }
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
# The lock is for osx-arm64, and an explicit install checks neither the platform nor the
# macOS version (its packages need 11.0 or later).
[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || { echo "  this script builds for Apple silicon (Darwin/arm64); on Linux x86_64 use build_meep_133_linux.sh"; exit 1; }
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
if [ -e "$ENV_PREFIX" ]; then
  echo "  $ENV_PREFIX already exists."
  echo "  This script never writes into an existing prefix; remove it yourself, or set"
  echo "  MEEP_ENV_NAME or MEEP_ENV_PREFIX."
  exit 1
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "  DRY RUN: nothing is built"
  exit 0
fi
export CFLAGS="$FLAGS" CXXFLAGS="$FLAGS"

say "1. create the environment from the conda lock"
# 127 of the lock's first 138 packages are the artifacts installed beside the MEEP the
# Metal kernels were certified against (Python 3.12.13, NumPy 2.4.3, fixed BEFORE the build
# because MEEP's SWIG module compiles against NumPy's C interface, Open MPI 5.0.10, the
# OpenMP build of OpenBLAS 0.3.30 with llvm-openmp 22.1.0, libctl, FFTW, GSL, harminv,
# MPB, swig, clang 22.1.0, gfortran 15.2.0). HDF5 and h5py are the mpi_openmpi builds
# (defect 3). The rest is what the build and the round need beside them: the compiler
# drivers and activation packages, the libc++ headers, make, libGDSII (which configure
# links when it finds it), gdspy and git. --no-default-packages: conda refuses an
# explicit file to which create_default_packages would add names.
echo "  installing $(grep -c '^https://' "$LOCK_CONDA") packages from $LOCK_CONDA, without a solve"
"$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" --no-default-packages --file "$LOCK_CONDA"
"$ENV_PREFIX/bin/clang" --version | head -1
# gfortran is NOT optional even though MEEP has no Fortran in it: configure runs
# AC_F77_WRAPPERS to work out the BLAS/LAPACK name-mangling scheme.

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
  sed -i '' 's/AX_CXX_MAXOPT/AX_CC_MAXOPT/' configure.ac
  echo "  patched configure.ac"
else
  echo "  configure.ac does not call AX_CXX_MAXOPT: upstream may have fixed it"
fi
export PATH="$ENV_PREFIX/bin:$PATH"
autoreconf -fvi >/dev/null 2>&1 || autoreconf -fi

say "4. configure IN-SOURCE with --enable-single (defect 2), MPI and libGDSII"
export CC="$ENV_PREFIX/bin/mpicc"
export CXX="$ENV_PREFIX/bin/mpic++"
# conda-forge ships gfortran only under its target-triplet name; the plain `gfortran`
# alias comes from the environment's activation scripts, which this script never runs.
GFORTRAN="$(ls "$ENV_PREFIX"/bin/*-gfortran 2>/dev/null | head -1)"
[ -n "$GFORTRAN" ] || { echo "no gfortran in $ENV_PREFIX/bin (step 1)"; exit 1; }
export F77="$GFORTRAN" FC="$GFORTRAN"
export CPPFLAGS="-I$ENV_PREFIX/include"
export LDFLAGS="-L$ENV_PREFIX/lib -Wl,-rpath,$ENV_PREFIX/lib"
export PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig"
export PYTHON="$ENV_PREFIX/bin/python"
# NOTE: no separate build/ directory (defect 2).
./configure \
  --prefix="$ENV_PREFIX" \
  --enable-shared --enable-single --enable-portable-binary \
  --with-mpi --without-scheme \
  --with-libctl="$ENV_PREFIX/share/libctl" \
  PYTHON="$ENV_PREFIX/bin/python"
# MANDATORY: configure found libGDSII (AC_CHECK_LIB(GDSII, libGDSIIExists)); without it
# MEEP's GDSII functions abort and two corpus rows cannot run.
grep -q '^ac_cv_lib_GDSII_libGDSIIExists=yes$' config.log || {
  echo "  REFUSING: configure did not find libGDSII (config.log: ac_cv_lib_GDSII_libGDSIIExists)"; exit 1; }
echo "  libGDSII: found by configure"

say "5. build (-j$JOBS, leaving cores free)"
make -j"$JOBS"

say "6. install, and record what libmeep and _meep link"
make install
IMAGES=()
while IFS= read -r image; do IMAGES+=("$image"); done < <(meep_images)
[ "${#IMAGES[@]}" = 2 ] || { echo "  REFUSING: expected libmeep and _meep, found ${#IMAGES[@]} image(s)"; exit 1; }
helper load-commands "$RECORD_DIR/load_commands_installed.json" "${IMAGES[@]}"

say "7. the PyPI packages, from the pip lock"
# PyTorch 2.10.0 (the build-tag-2 wheel the Metal kernels were certified with) and its
# dependencies, pytest, autograd (MEEP's adjoint module imports it), the corpus-only
# packages (shapely, nlopt, imageio-ffmpeg) and llvmlite (the test suite's LLVM parser),
# each by version and SHA-256, with no dependency resolution (--no-deps); then the
# PyMieScatt stand-in from its pinned sdist and patch; pip check confirms that nothing is
# missing, and the last check refuses any PyPI package that is not the lock's or the
# stand-in.
py -m pip install --no-deps -r "$LOCK_PIP"
py "$SCRIPT_DIR/rounds/1.0/build_pymiescatt_standin.py" --python "$ENV_PREFIX/bin/python" --install
py -m pip check
py "$REPO_ROOT/tools/compare_reference_environment.py" \
    --prefix "$ENV_PREFIX" --require-pip-lock

say "7b. ffmpeg: the imageio-ffmpeg executable, linked into the prefix"
# test_animation_output is skipped, and its corpus row refused by name, unless `which
# ffmpeg` succeeds on the reference PATH (<prefix>/bin:/usr/bin:/bin). The executable
# comes from the pinned imageio-ffmpeg wheel, which installs no shared library: it loads
# only the operating system's libraries, and MEEP's images keep the load commands they
# were installed with (checked here).
helper link-ffmpeg "$ENV_PREFIX" "$RECORD_DIR/ffmpeg.json"
helper load-commands "$RECORD_DIR/load_commands.json" "${IMAGES[@]}"
helper same-load-commands "$RECORD_DIR/load_commands_installed.json" "$RECORD_DIR/load_commands.json"

say "8. MANDATORY assertions: the reference this environment claims to be"
# The configure substitution is not evidence; only the runtime values are. The process
# loads two OpenMP runtimes (OpenBLAS's and PyTorch's), which this check allows for its
# own process only.
PYTHONPATH="$REPO_ROOT" KMP_DUPLICATE_LIB_OK=TRUE py - <<'PYEOF'
import sys
import meep as mp
import numpy, torch
checks = [
    ("MEEP version", mp.__version__, "1.33.0"),
    ("single precision", mp.is_single_precision() and mp.get_realnum_size() == 4, True),
    ("MPI build", mp.with_mpi(), True),
    ("NumPy", numpy.__version__, "2.4.3"),
    ("PyTorch", torch.__version__.split("+")[0], "2.10.0"),
    ("Metal (MPS) available", torch.backends.mps.is_available(), True),
]
failed = []
for name, got, want in checks:
    print(f"  {name:24} {got!s:12} {'ok' if got == want else 'EXPECTED ' + str(want)}", flush=True)
    if got != want:
        failed.append(name)
try:
    from meep_gpu.metal_kernels.device import metal_frontend_version
    print(f"  {'Metal frontend':24} {metal_frontend_version()}  (the certified pair is torch "
          f"2.10.0 with metalfe-32023.850.10; another frontend means the Metal table does "
          f"not admit this Mac, not that the build failed)", flush=True)
except Exception as exc:  # informational only
    print(f"  Metal frontend           unread ({type(exc).__name__})", flush=True)
if failed:
    print(f"\n  FAILED: {failed}. 'single precision' False is defect 2: rebuild in-source.")
    sys.exit(1)
print("\n  reference environment OK")
PYEOF
# MEEP's GDSII support, in a process of its own (MEEP aborts without libGDSII): the layers
# of the corpus's own coupler.gds, as plain integers (MEEP returns a NumPy int32 array,
# whose elements NumPy 2 prints as np.int32(0) inside a list).
GDSII="$(cd "$CORPUS_ROOT/examples" && py -c 'import meep as mp; mp.verbosity(0); print("LAYERS", [int(layer) for layer in mp.get_GDSII_layers("coupler.gds")])' 2>&1 | grep '^LAYERS' || true)"
echo "  GDSII                    ${GDSII:-not read}"
[ "$GDSII" = "LAYERS [0, 1, 2, 3, 4, 5, 31, 32]" ] || { echo "  REFUSING: MEEP does not read the corpus's coupler.gds (no libGDSII support)"; exit 1; }
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
# The build root is under $TMPDIR, which a reboot may empty; a timing that divides by this
# MEEP must still be able to say how it was compiled. config.log carries the configure
# line, the Makefile the compiler flags, and build_record.json what the interpreter loads
# (timing_records.py meep-build-record: version, precision, the sha256 of the MEEP images,
# the contraction the flags ask for, their census and the host's math library);
# tools/check_install.py --require-identity-reference reads it.
cp "$BUILD_ROOT/meep-${MEEP_VERSION}/config.log" "$RECORD_DIR/config.log"
cp "$BUILD_ROOT/meep-${MEEP_VERSION}/Makefile" "$RECORD_DIR/Makefile"
if [ "$ROLE" = reference ]; then
  RECORD_OPTIONS=(--require-identity)
else
  # a timing control also records what its flags made of this CPU
  helper compiler-facts "$(ls "$ENV_PREFIX"/bin/*-clang++ | head -1)" "$FLAGS" "$RECORD_DIR/native.json"
  RECORD_OPTIONS=(--extra-json "$RECORD_DIR/native.json")
fi
py "$SCRIPT_DIR/timing_records.py" meep-build-record \
  --python "$ENV_PREFIX/bin/python" --config-log "$RECORD_DIR/config.log" \
  --launcher "$ENV_PREFIX/bin/mpirun" --require-single "${RECORD_OPTIONS[@]}" \
  --out "$RECORD_DIR/build_record.json"
# The macOS SDK MEEP is compiled against is the host's (Command Line Tools or Xcode), not
# a package of the lock: kept as the built libmeep records it (LC_BUILD_VERSION).
SDK_LINE="$(otool -l "$ENV_PREFIX/lib/libmeep.dylib" 2>/dev/null \
  | awk '/LC_BUILD_VERSION/{f=1} f && $1=="minos"{m=$2} f && $1=="sdk"{print "minos " m ", sdk " $2; exit}' || true)"
echo "  libmeep's macOS build version: ${SDK_LINE:-not read}"
echo "${SDK_LINE:-not read}" > "$RECORD_DIR/macos_sdk.txt"

say "10. MANDATORY for a reference: MEEP's own words on the four probe cases"
# parity/meep_gpu/meep_identity_smoke.py steps MEEP alone on four cases and compares the
# SHA-256 of every stored word at the compared steps with the recorded set; it exits 1
# unless every digest reproduces and its planted 1-ulp flip is caught.
if [ "$ROLE" = timing-control ]; then
  echo "  skipped: a timing control is not an identity reference"
else
  py "$SCRIPT_DIR/meep_identity_smoke.py" \
    --mode keep --rows "$RECORD_DIR/identity_smoke.jsonl"
fi

say "11. MANDATORY for a reference: the identity verdicts, keep and flush"
# tools/check_install.py's identity check judges the build a host runs, in a process that
# imports NumPy and MEEP and nothing else; the helper takes that verdict alone (the rest
# of check_install steps a simulation on the GPU, which a build does not). keep must pass.
# flush is reached on Apple silicon through the floating-point control register's FZ bit
# (MEEP's own switch has no effect there) and is judged at OMP_NUM_THREADS=1; a
# check_install that does not drive that bit refuses it by name ("no flush reference"):
# that refusal alone is recorded as pending (exit 3), any other refusal stops the build.
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
    elif [ "$MODE" = flush ] && [ "$STATUS" = 3 ]; then
      echo "pending: tools/check_install.py does not drive the FZ bit" > "$RECORD_DIR/identity_flush.outcome"
      echo "  flush: PENDING (check_install refuses it by name on this host; recorded)"
    else
      echo "fail" > "$RECORD_DIR/identity_$MODE.outcome"
      echo "  REFUSING: the identity verdict in $MODE is not a pass (exit $STATUS)"
      exit 1
    fi
  done
fi

say "12. the programs the round runs, and the reference build record"
helper tools "$ENV_PREFIX" osx-arm64 "$RECORD_DIR/tools.json"
helper verify-corpus "$CORPUS_ROOT" "$CORPUS_LIST" "$CORPUS_SIBLINGS"
CONFIGURE_LINE="$(sed -n 's/^  \$ \(\.\/configure .*\)$/\1/p' "$RECORD_DIR/config.log" | head -1)"
helper manifest "$RECORD_DIR" "$RECORD_DIR/reference_build.json" \
  --platform osx-arm64 --role "$ROLE" --tarball-url "$MEEP_URL" --tarball-sha256 "$MEEP_SHA256" \
  --cflags "$CFLAGS" --cxxflags "$CXXFLAGS" --configure "$CONFIGURE_LINE" \
  --corpus-root "share/meep-gpu-build/meep-${MEEP_VERSION}/python" --corpus-list "$CORPUS_LIST" \
  --corpus-siblings "$CORPUS_SIBLINGS"

say "DONE"
echo "Activate it with:  conda activate $ENV_PREFIX"
echo "The identity corpus: MEEP_GPU_CORPUS_ROOT=$CORPUS_ROOT"
echo "The record: $RECORD_DIR/reference_build.json"
echo "For everyday use instead, INSTALL.md's 'One OpenMP runtime on an Apple silicon Mac'"
echo "switches OpenBLAS to the pthreads build."
