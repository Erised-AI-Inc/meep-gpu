#!/usr/bin/env bash
# Build the certification reference on Apple silicon: MEEP 1.33.0 with SINGLE-PRECISION
# fields and MPI, in its own conda prefix, with the certified PyTorch beside it.
#
# THIS IS THE MEEP THE METAL KERNELS WERE CERTIFIED AGAINST, and the Mac counterpart of
# build_meep_133_linux.sh. Single precision is not optional: the engine stores float32
# and every certified figure was measured against a --enable-single MEEP; conda-forge's
# `pymeep` is a double-precision build, a supported way to USE the package but not a
# reference to certify against.
#
#   bash parity/meep_gpu/build_meep_133_macos.sh 2>&1 | tee ~/build_meep_133.log
#   conda activate meep-gpu-ref
#
# THREE DEFECTS IN THE 1.33.0 RELEASE TARBALL, all handled below
# (INSTALL.md, "Building MEEP in single precision"):
#
#   1. configure.ac calls AX_CXX_MAXOPT while the bundled m4 defines AX_CC_MAXOPT.
#      Patched, then Autotools regenerated.
#   2. An OUT-OF-TREE --enable-single build writes MEEP_SINGLE=1 into
#      build/src/meep/meep-config.h, but compilation follows the quoted include in
#      the source tree to the tarball's pre-generated header where MEEP_SINGLE=0.
#      Such a build installs cleanly while mp.is_single_precision() returns False.
#      This script therefore builds IN-SOURCE, and asserts the runtime value after
#      installing — the configure substitution is NOT evidence.
#   3. nompi HDF5/h5py make MEEP configure without H5Pset_fapl_mpio, which deadlocks
#      multi-rank output. The lock below holds the mpi_openmpi builds of both.
#
# THE ENVIRONMENT IS INSTALLED FROM A LOCK, NOT SOLVED. Step 1 installs
# environments/locks/meep-gpu-ref-osx-arm64.conda.txt (an explicit conda lock: every
# package by URL and MD5), step 1b refuses to go on unless the prefix holds exactly
# those artifacts, and step 7 installs meep-gpu-ref-osx-arm64.pip.txt (every PyPI
# package by version and wheel hash). A solve picks whatever conda-forge serves on the
# day: run on 2026-10-03, this script's former solve differed from the certified
# environment in 54 of the 139 packages they share, libctl, fftw, harminv, scipy and the
# compiler runtimes among them. environments/locks/README.md says what the lock
# reproduces and when it is re-cut. MEEP_REFERENCE_SOLVE=1 solves instead, as before;
# the result is then NOT the certification reference, and the script says so.
#
# progress reporting: every stage prints a flushed marker and the whole run is logged.
set -euo pipefail

MEEP_VERSION=1.33.0
MEEP_SHA256=bdabc0a112f669f2657fbdca22e31a2aeed515372e8ed56dc55197a4cd7ff0ab
ENV_NAME="${MEEP_ENV_NAME:-meep-gpu-ref}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
BUILD_ROOT="${MEEP_BUILD_ROOT:-/tmp/meep_build_133_macos}"
JOBS="${MEEP_BUILD_JOBS:-6}"
# Resolved BEFORE any cd: later stages run inside the build tree, where a
# relative script path no longer exists.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCK_CONDA="$REPO_ROOT/environments/locks/meep-gpu-ref-osx-arm64.conda.txt"
LOCK_PIP="$REPO_ROOT/environments/locks/meep-gpu-ref-osx-arm64.pip.txt"
SOLVE="${MEEP_REFERENCE_SOLVE:-0}"
NOT_REFERENCE="MEEP_REFERENCE_SOLVE=1: this environment is SOLVED, not installed from the lock; it is NOT the certification reference"

say() { printf '\n=== %s === %s\n' "$1" "$(date +%H:%M:%S)"; }

say "0. refusing to clobber an existing prefix"
# The lock is for osx-arm64, and an explicit install checks neither the platform nor
# the macOS version (its packages need 11.0 or later).
[ "$(uname -s)/$(uname -m)" = "Darwin/arm64" ] || { echo "  this script builds for Apple silicon (Darwin/arm64); on Linux x86_64 use build_meep_133_linux.sh"; exit 1; }
if [ "$SOLVE" != 1 ]; then
  for lock in "$LOCK_CONDA" "$LOCK_PIP"; do
    [ -f "$lock" ] || { echo "  missing $lock (run this script from a checkout of the repository)"; exit 1; }
  done
fi
if [ -d "$ENV_PREFIX" ]; then
  echo "  $ENV_PREFIX already exists."
  echo "  This script never writes into an existing prefix; remove it yourself, or set"
  echo "  MEEP_ENV_NAME."
  exit 1
fi

say "1. create the environment: the Metal reference's libraries and compilers"
# THE LOCK IS THE REFERENCE. 127 of its 138 packages are the artifacts installed beside
# the MEEP the Metal kernels were certified against (read from that environment,
# 2026-10-03): Python 3.12.13, NumPy 2.4.3 (fixed BEFORE the build, because MEEP's SWIG
# module compiles against NumPy's C interface), Open MPI 5.0.10, the OpenMP build of
# OpenBLAS 0.3.30 (and the openblas package, whose libopenblas.dylib MEEP links) with
# llvm-openmp 22.1.0, libctl, FFTW, GSL, harminv, MPB, swig, clang
# 22.1.0 and gfortran 15.2.0. HDF5 and h5py differ by design: the reference's HDF5 is
# the nompi build, and the lock holds the mpi_openmpi builds (defect 3); one process,
# which is all this package uses, never reaches parallel HDF5. The other 9 are what the
# build needs and the certified environment lacks (the C++ driver of the same compiler
# build, the compiler activation packages, the libc++ headers matching its libc++
# runtime, make). --no-default-packages: conda refuses an explicit file to which
# create_default_packages would add names.
if [ "$SOLVE" = 1 ]; then
  echo "  $NOT_REFERENCE."
  "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels \
    "python=3.12.13" "numpy=2.4.3" scipy pytest pip matplotlib-base \
    "openmpi=5.0.10" "hdf5=1.14.6=mpi_openmpi*" "h5py=*=mpi_openmpi*" \
    gsl fftw libctl harminv mpb "swig=4.4.1" \
    autoconf automake libtool m4 pkg-config make \
    clang_osx-arm64 clangxx_osx-arm64 "gfortran_osx-arm64=15.2.0" libblas liblapack \
    "libopenblas=0.3.30=*openmp*" "llvm-openmp=22.1.0"
else
  echo "  installing $(grep -c '^https://' "$LOCK_CONDA") packages from $LOCK_CONDA, without a solve"
  "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" --no-default-packages --file "$LOCK_CONDA"
fi
"$ENV_PREFIX/bin/clang" --version | head -1
# gfortran is NOT optional even though MEEP has no Fortran in it: configure runs
# AC_F77_WRAPPERS to work out the BLAS/LAPACK name-mangling scheme, and without a
# Fortran compiler it stops with "cannot compile a simple Fortran program" long
# before anything is built.

say "1b. MANDATORY: the environment is the lock, artifact for artifact"
# Every conda package of the new prefix, by URL and MD5, against every line of the lock,
# before anything is built on it: a package missing, added or of another build refuses.
RECORD_DIR="$ENV_PREFIX/share/meep-gpu-build"
mkdir -p "$RECORD_DIR"
if [ "$SOLVE" = 1 ]; then
  echo "  skipped: $NOT_REFERENCE."
  echo "$NOT_REFERENCE." > "$RECORD_DIR/NOT_THE_CERTIFICATION_REFERENCE.txt"
else
  LOCK_DIFF="$(diff <(grep '^https://' "$LOCK_CONDA" | sort) \
    <("$CONDA_ROOT/bin/conda" list --explicit --md5 -p "$ENV_PREFIX" | grep '^https://' | sort) || true)"
  if [ -n "$LOCK_DIFF" ]; then
    echo "  the prefix differs from the lock ('<' only in the lock, '>' only in the prefix):"
    echo "$LOCK_DIFF" | sed 's/^/    /'
    echo "  REFUSING: $ENV_PREFIX is not the certification reference; nothing was built on it."
    exit 1
  fi
  LOCKED=$(grep -c '^https://' "$LOCK_CONDA")
  echo "  $LOCKED of $LOCKED conda packages are the lock's artifacts (URL and MD5), and the prefix holds no other"
  cp "$LOCK_CONDA" "$LOCK_PIP" "$RECORD_DIR/"
  echo "  kept the lock files in $RECORD_DIR"
fi

say "2. fetch and verify the release tarball"
rm -rf "$BUILD_ROOT"; mkdir -p "$BUILD_ROOT"; cd "$BUILD_ROOT"
curl -fsSL -o meep.tar.gz \
  "https://github.com/NanoComp/meep/releases/download/v${MEEP_VERSION}/meep-${MEEP_VERSION}.tar.gz"
echo "  expected $MEEP_SHA256"
ACTUAL=$(shasum -a 256 meep.tar.gz | awk '{print $1}')
echo "  actual   $ACTUAL"
[ "$ACTUAL" = "$MEEP_SHA256" ] || { echo "  SHA-256 MISMATCH — refusing to build"; exit 1; }
tar xzf meep.tar.gz
cd "meep-${MEEP_VERSION}"

say "3. patch AX_CXX_MAXOPT -> AX_CC_MAXOPT (defect 1) and regenerate autotools"
if grep -q 'AX_CXX_MAXOPT' configure.ac; then
  sed -i '' 's/AX_CXX_MAXOPT/AX_CC_MAXOPT/' configure.ac
  echo "  patched configure.ac"
else
  echo "  configure.ac does not call AX_CXX_MAXOPT — upstream may have fixed it"
fi
export PATH="$ENV_PREFIX/bin:$PATH"
autoreconf -fvi >/dev/null 2>&1 || autoreconf -fi

# Opt-in sigma-reader patch (meep-sigma-reader.patch, parity/meep_gpu/README.md). Applied
# AFTER autoreconf because it touches only src/meep.hpp and src/monitor.cpp —
# no configure inputs — so the regenerated build system is untouched and `make`
# rebuilds the SWIG module because meep.hpp changed. Off by default so the
# pristine certification reference can always be rebuilt from this same script.
if [ "${MEEP_SIGMA_PATCH:-0}" = "1" ]; then
  say "3b. apply the susceptibility-sigma reader patch"
  patch -p1 < "$SCRIPT_DIR/meep-sigma-reader.patch"
  echo "  applied meep-sigma-reader.patch (fields::get_susceptibility_sigma et al.)"
  # The SWIG regeneration rule in python/Makefile.am sits inside MAINTAINER_MODE;
  # without it the tarball's PRE-GENERATED meep-python.cxx is compiled as shipped
  # and the patched declarations never reach Python — the library builds, the
  # module imports, and hasattr(mp.fields, "get_susceptibility_sigma") is False.
  # Measured on the first build of this patch; maintainer mode is the fix.
  EXTRA_CONFIGURE_FLAGS="--enable-maintainer-mode"
  # A maintainer regeneration also needs the SWIG interface sources the release
  # tarball omits (the shipped .cxx already embeds them): numpy.i, vec.i and
  # mpb.i. Fetch each from the SAME release tag, SHA-pinned like the tarball
  # itself (SHAs computed from `git show v1.33.0:python/<f>` on the upstream
  # checkout).
  fetch_swig_src() {
    local name="$1" expected="$2" actual
    curl -fsSL -o "python/$name" \
      "https://raw.githubusercontent.com/NanoComp/meep/v${MEEP_VERSION}/python/$name"
    actual=$(shasum -a 256 "python/$name" | awk '{print $1}')
    [ "$actual" = "$expected" ] || {
      echo "  $name SHA-256 mismatch ($actual) — refusing to build"; exit 1; }
    echo "  fetched python/$name from the v${MEEP_VERSION} tag"
  }
  fetch_swig_src numpy.i bc658c3e9f4ecc5343041339c0697314398fd7cd0e442aa2d424b71086e305ec
  fetch_swig_src vec.i   3f56ebf2730a7ac5d52d5cb4365c62edc111aa7d86e1e5adbf745663118e81a1
  fetch_swig_src mpb.i   1c9cea21b50ddd635d26bc97996ba03bee8265bab8cb20fea744f576393627b6
else
  EXTRA_CONFIGURE_FLAGS=""
fi

say "4. configure IN-SOURCE with --enable-single (defect 2)"
export CC="$ENV_PREFIX/bin/mpicc"
export CXX="$ENV_PREFIX/bin/mpic++"
# conda-forge ships gfortran only under its target-triplet name; the plain
# `gfortran` alias comes from the env's ACTIVATION scripts, which this script
# deliberately never runs (it builds with explicit paths, not an activated
# shell). Without F77, configure falls through to clang for the name-mangling
# probe and stops with "cannot compile a simple Fortran program".
GFORTRAN="$(ls "$ENV_PREFIX"/bin/*-gfortran 2>/dev/null | head -1)"
[ -n "$GFORTRAN" ] || { echo "no gfortran in $ENV_PREFIX/bin — see step 1's package list"; exit 1; }
export F77="$GFORTRAN"
export FC="$GFORTRAN"
export CPPFLAGS="-I$ENV_PREFIX/include"
export LDFLAGS="-L$ENV_PREFIX/lib -Wl,-rpath,$ENV_PREFIX/lib"
export PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig"
export PYTHON="$ENV_PREFIX/bin/python"
# NOTE: no separate build/ directory. See defect 2 — an out-of-tree build silently
# produces a DOUBLE-precision library from a --enable-single configure.
./configure \
  --prefix="$ENV_PREFIX" \
  --enable-shared --enable-single --enable-portable-binary \
  --with-mpi --without-scheme \
  --with-libctl="$ENV_PREFIX/share/libctl" \
  $EXTRA_CONFIGURE_FLAGS \
  PYTHON="$ENV_PREFIX/bin/python"

say "5. build (-j$JOBS, leaving cores free)"
make -j"$JOBS"

say "6. install"
make install

say "7. the certified PyTorch"
# PyTorch 2.10.0 is the version the Metal kernels were certified with. The OpenMP build
# of OpenBLAS stays, as in the reference; a process then loads two OpenMP runtimes,
# which the reference run allowed with KMP_DUPLICATE_LIB_OK (recut_metal_gates.sh sets
# it), so the check below does the same. From the lock: PyTorch's wheel (build tag 2,
# the one the certified environment holds) and every other PyPI package of the
# reference, each by version and SHA-256, with no dependency resolution (--no-deps);
# pip check then confirms that nothing is missing. -s keeps the user site directory out.
if [ "$SOLVE" = 1 ]; then
  echo "  $NOT_REFERENCE."
  "$ENV_PREFIX/bin/python" -m pip install "torch==2.10.0"
else
  "$ENV_PREFIX/bin/python" -s -m pip install --no-deps -r "$LOCK_PIP"
  "$ENV_PREFIX/bin/python" -s -m pip check
fi

say "8. MANDATORY assertions: the reference this environment claims to be"
# The configure substitution is not evidence; only the runtime values are.
PYTHONPATH="$REPO_ROOT" KMP_DUPLICATE_LIB_OK=TRUE "$ENV_PREFIX/bin/python" -s - <<'PYEOF'
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
# Then the drift report a user runs (tools/compare_reference_environment.py), on the
# environment just built: every numerics-relevant package against the lock.
if [ "$SOLVE" = 1 ]; then
  echo "  $NOT_REFERENCE; the comparison with the reference is informational:"
  "$ENV_PREFIX/bin/python" -s "$REPO_ROOT/tools/compare_reference_environment.py" || true
else
  "$ENV_PREFIX/bin/python" -s "$REPO_ROOT/tools/compare_reference_environment.py" --require-reference
fi

say "9. keep the build's record in the prefix"
# The build root is under /tmp, which a reboot empties; a timing that divides by this
# MEEP must still be able to say how it was compiled. config.log carries the configure
# line, the Makefile the compiler flags, and build_record.json what the interpreter
# loads (timing_records.py meep-build-record: version, precision, the MEEP libraries
# the process loaded and their sha256, whether MEEP's own libraries link OpenMP).
mkdir -p "$RECORD_DIR"
cp "$BUILD_ROOT/meep-${MEEP_VERSION}/config.log" "$RECORD_DIR/config.log"
cp "$BUILD_ROOT/meep-${MEEP_VERSION}/Makefile" "$RECORD_DIR/Makefile"
"$ENV_PREFIX/bin/python" "$SCRIPT_DIR/timing_records.py" meep-build-record \
  --python "$ENV_PREFIX/bin/python" --config-log "$RECORD_DIR/config.log" \
  --launcher "$ENV_PREFIX/bin/mpirun" --require-single \
  --out "$RECORD_DIR/build_record.json"
# The macOS SDK MEEP is compiled against is the host's (Command Line Tools or Xcode),
# not a package of the lock, and the certified MEEP records SDK 26.2: kept as the built
# libmeep records it (LC_BUILD_VERSION), since xcrun's answer depends on the host's setup.
SDK_LINE="$(otool -l "$ENV_PREFIX/lib/libmeep.dylib" 2>/dev/null \
  | awk '/LC_BUILD_VERSION/{f=1} f && $1=="minos"{m=$2} f && $1=="sdk"{print "minos " m ", sdk " $2; exit}' || true)"
echo "  libmeep's macOS build version: ${SDK_LINE:-not read} (the certified MEEP: minos 26.0, sdk 26.2)"
echo "${SDK_LINE:-not read}" > "$RECORD_DIR/macos_sdk.txt"
echo "  kept $RECORD_DIR/{config.log,Makefile,build_record.json,macos_sdk.txt}"

say "DONE"
echo "Activate it with:  conda activate $ENV_NAME"
if [ "$SOLVE" = 1 ]; then
  echo "$NOT_REFERENCE."
else
  echo "It reproduces the certification reference, which keeps the OpenMP build of OpenBLAS."
fi
echo "For everyday use instead, INSTALL.md's 'One OpenMP runtime on an Apple silicon Mac'"
echo "switches it to the pthreads build."
