#!/usr/bin/env bash
# Build MEEP 1.33.0 with SINGLE-PRECISION fields into its own macOS/arm64 prefix.
#
# This ADDS a forward-compatibility oracle; it does not replace anything. The MEEP
# version policy in the design notes (meep-gpu-port-reference) is explicit that the
# 1.29.0 environment stays frozen, because it is the exact reference behind every
# measured floor in the parity matrix. Having both installed side by
# side is also the only way to attribute a parity difference to an upstream change
# rather than to this package.
#
# Single precision ("32-bit fields") is not optional here: the engine stores float32
# and every parity number was measured against a --enable-single MEEP. A
# double-precision oracle would move every comparison for a reason that has nothing
# to do with the code under test.
#
# THREE DEFECTS IN THE 1.33.0 RELEASE TARBALL, all documented in the port reference
# and all handled below:
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
#      multi-rank output. The env below pins the mpi_openmpi builds of both.
#
# Rule 7: every stage prints a flushed marker and the whole run is logged.
set -euo pipefail

MEEP_VERSION=1.33.0
MEEP_SHA256=bdabc0a112f669f2657fbdca22e31a2aeed515372e8ed56dc55197a4cd7ff0ab
ENV_NAME="${MEEP_ENV_NAME:-meep133}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
BUILD_ROOT="${MEEP_BUILD_ROOT:-/tmp/meep_build_133_macos}"
JOBS="${MEEP_BUILD_JOBS:-6}"
# Resolved BEFORE any cd: later stages run inside the build tree, where a
# relative script path no longer exists.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n=== %s === %s\n' "$1" "$(date +%H:%M:%S)"; }

say "0. refusing to clobber an existing prefix"
if [ -d "$ENV_PREFIX" ]; then
  echo "  $ENV_PREFIX already exists."
  echo "  This script never writes into an existing prefix — the whole point is that"
  echo "  the 1.29.0 oracle survives. Remove it yourself, or set MEEP_ENV_NAME."
  exit 1
fi

say "1. create the candidate environment"
# hdf5/h5py pinned to their OpenMPI builds (defect 3). mpb and harminv come from
# conda-forge; MEEP links whatever is in this prefix.
"$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" -c conda-forge \
  python=3.12 numpy scipy pytest matplotlib-base \
  openmpi "hdf5=*=mpi_openmpi*" "h5py=*=mpi_openmpi*" \
  gsl fftw libctl harminv mpb swig \
  autoconf automake libtool m4 pkg-config make \
  clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64 libblas liblapack
# gfortran is NOT optional even though MEEP has no Fortran in it: configure runs
# AC_F77_WRAPPERS to work out the BLAS/LAPACK name-mangling scheme, and without a
# Fortran compiler it stops with "cannot compile a simple Fortran program" long
# before anything is built.

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

# Opt-in sigma-reader patch (the design notes (meep-sigma-reader-plan)). Applied
# AFTER autoreconf because it touches only src/meep.hpp and src/monitor.cpp —
# no configure inputs — so the regenerated build system is untouched and `make`
# rebuilds the SWIG module because meep.hpp changed. Off by default so the two
# pristine oracle environments can always be rebuilt from this same script.
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

say "7. MANDATORY post-install assertions"
# The configure substitution is not evidence; only the runtime value is.
"$ENV_PREFIX/bin/python" - <<'PYEOF'
import sys
import meep as mp
single = mp.is_single_precision()
size = mp.get_realnum_size()
print(f"  meep                 {mp.__version__}")
print(f"  is_single_precision  {single}")
print(f"  realnum size (bytes) {size}")
print(f"  with_mpi             {mp.with_mpi()}")
failures = []
if mp.__version__ != "1.33.0":
    failures.append(f"expected 1.33.0, got {mp.__version__}")
if not single or size != 4:
    failures.append("NOT single precision — this is defect 2; rebuild in-source")
if failures:
    print("\n  FAILED:")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
print("\n  candidate oracle OK")
PYEOF

say "DONE — 1.29.0 environments untouched"
echo "Before PyTorch goes into it, make OpenBLAS the pthreads build (INSTALL.md):"
echo "  conda install -p $ENV_PREFIX -c conda-forge --override-channels \"libopenblas=*=*pthreads*\""
echo "Run the suite against it with (KMP_DUPLICATE_LIB_OK unset):"
echo "  cd <repository root> && $ENV_PREFIX/bin/python -m pytest meep_gpu -q"
