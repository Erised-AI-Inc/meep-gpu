#!/usr/bin/env bash
# Build the certification reference on Linux x86_64: MEEP 1.33.0 with SINGLE-PRECISION
# fields, MPI and OpenMP, in its own conda prefix, with the certified GPU libraries
# beside it.
#
# THIS IS THE MEEP EVERY CERTIFICATION ROUND IS MEASURED AGAINST. The NVIDIA kernels
# were certified on a Linux host whose MEEP was built from the 1.33.0 release tarball
# with exactly the configure flags below, and the Metal kernels on a Mac built the same
# way (build_meep_133_macos.sh). conda-forge's `pymeep` is a double-precision serial
# build and has no single-precision variant, so it is a supported way to USE the
# package but not a reference to certify against: a second architecture gated against
# a different MEEP differs from the first in more than its GPU.
#
#   bash parity/meep_gpu/build_meep_133_linux.sh          # from the repository root
#   conda activate meep-gpu-ref
#
# THREE DEFECTS IN THE 1.33.0 RELEASE TARBALL, all handled below
# (INSTALL.md, "Building MEEP in single precision"):
#
#   1. configure.ac calls AX_CXX_MAXOPT while the bundled m4 defines AX_CC_MAXOPT.
#      Patched, then Autotools regenerated.
#   2. An OUT-OF-TREE --enable-single build installs cleanly and is DOUBLE precision:
#      compilation follows the quoted include to the tarball's pre-generated header,
#      where MEEP_SINGLE=0. This script builds IN-SOURCE and asserts the runtime
#      value after installing; the configure substitution is not evidence.
#   3. nompi HDF5/h5py make MEEP configure without H5Pset_fapl_mpio, which deadlocks
#      multi-rank output. The lock holds the mpi_openmpi builds of both.
#
# THE ENVIRONMENT IS INSTALLED FROM A LOCK, NOT SOLVED. Step 1 installs
# environments/locks/meep-gpu-ref-linux-64.conda.txt (an explicit conda lock: every
# package by URL and MD5), step 1b refuses to go on unless the prefix holds exactly
# those artifacts, and step 7 installs meep-gpu-ref-linux-64.pip.txt (every PyPI
# package by version and wheel hash). The lock is the environment of the 2026-10-03
# RTX A6000 certification round. A solve depends on the day and on the host: for a host
# with glibc 2.35, this script's former solve took sysroot_linux-64 2.34 and
# kernel-headers 5.14 where the reference has 2.28 and 4.18 (solved on 2026-10-03 with
# conda's glibc override), so MEEP would be compiled against another sysroot.
# environments/locks/README.md says what the lock reproduces and when it is re-cut.
# MEEP_REFERENCE_SOLVE=1 solves instead, as before; the result is then NOT the
# certification reference, and the script says so.
#
# Every stage prints a flushed, timestamped marker; send the whole output to a log:
#   bash parity/meep_gpu/build_meep_133_linux.sh 2>&1 | tee ~/build_meep_133.log
set -euo pipefail

MEEP_VERSION=1.33.0
MEEP_SHA256=bdabc0a112f669f2657fbdca22e31a2aeed515372e8ed56dc55197a4cd7ff0ab
ENV_NAME="${MEEP_ENV_NAME:-meep-gpu-ref}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
BUILD_ROOT="${MEEP_BUILD_ROOT:-${TMPDIR:-/tmp}/meep_build_133_linux}"
JOBS="${MEEP_BUILD_JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"

# The repository this script ships in: step 8 reads meep_gpu from it, since the
# package is never installed into this environment (a gate imports it from the clone).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCK_CONDA="$REPO_ROOT/environments/locks/meep-gpu-ref-linux-64.conda.txt"
LOCK_PIP="$REPO_ROOT/environments/locks/meep-gpu-ref-linux-64.pip.txt"
SOLVE="${MEEP_REFERENCE_SOLVE:-0}"
NOT_REFERENCE="MEEP_REFERENCE_SOLVE=1: this environment is SOLVED, not installed from the lock; it is NOT the certification reference"

say() { printf '\n=== %s === %s\n' "$1" "$(date -u +%FT%TZ)"; }

say "0. refusing to clobber an existing prefix"
[ "$(uname -s)/$(uname -m)" = "Linux/x86_64" ] || { echo "  this script builds for Linux x86_64; on a Mac use build_meep_133_macos.sh"; exit 1; }
[ -x "$CONDA_ROOT/bin/conda" ] || { echo "  no conda at $CONDA_ROOT (set CONDA_ROOT)"; exit 1; }
if [ "$SOLVE" != 1 ]; then
  for lock in "$LOCK_CONDA" "$LOCK_PIP"; do
    [ -f "$lock" ] || { echo "  missing $lock (run this script from a checkout of the repository)"; exit 1; }
  done
  # The lock's compilers target glibc 2.28, which an explicit install does not check;
  # checked here, before anything is installed.
  GLIBC="$( (getconf GNU_LIBC_VERSION 2>/dev/null || true) | awk '{print $2}')"
  echo "  host glibc ${GLIBC:-not read} (the lock needs 2.28 or later)"
  [ -n "$GLIBC" ] || { echo "  REFUSING: the host's glibc version could not be read (getconf GNU_LIBC_VERSION); the lock needs glibc 2.28 or later"; exit 1; }
  if [ "$(printf '%s\n%s\n' 2.28 "$GLIBC" | sort -V | head -1)" != 2.28 ]; then
    echo "  REFUSING: glibc $GLIBC is older than 2.28"; exit 1
  fi
fi
if [ -d "$ENV_PREFIX" ]; then
  echo "  $ENV_PREFIX already exists. This script never writes into an existing"
  echo "  prefix; remove it yourself, or set MEEP_ENV_NAME."
  exit 1
fi

say "1. create the environment: the reference's libraries and compilers, the certified GPU libraries"
# THE LOCK IS THE REFERENCE: the conda environment the 2026-10-03 RTX A6000 round ran
# in, exported verbatim (143 packages). That environment was made on 2026-10-02 by
# this script's former solve, whose pins follow the environment the NVIDIA kernels
# were first certified in: the compilers MEEP was built with (15.2.0), NumPy 2.2.6
# (fixed BEFORE the build, because MEEP's SWIG module compiles against NumPy's C
# interface), and every library MEEP links. Three differ from that older environment.
# Two only in MPI and parallel-HDF5 plumbing a one-process gate never exercises: its
# Open MPI 5.0.10 and h5py 3.16.0 require the CUDA 12 runtime on conda-forge, which the
# certified CuPy (CUDA 11.8) cannot sit beside, so the lock has 5.0.8 and 3.15.1. The
# third is the CUDA compiler: NVRTC 11.8.89, from cudatoolkit 11.8, which is what
# environments/nvidia-linux.yml gives a user, where the A6000 records cut before
# 2026-10-02 compiled with NVRTC 11.6.55 from that host's own CUDA 11.6. A gate records
# the version it compiled with (environment.nvrtc_version). The solve also chose what
# the pins left open, and the lock keeps those choices: OpenBLAS 0.3.34 (pthreads) and
# the compiler runtimes libgcc, libstdcxx, libgfortran and libgomp at 16.2.0.
# --no-default-packages: conda refuses an explicit file to which
# create_default_packages would add names.
if [ "$SOLVE" = 1 ]; then
  echo "  $NOT_REFERENCE."
  "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" -c conda-forge --override-channels \
    "python=3.10.20" "numpy=2.2.6" scipy pytest pip "matplotlib-base=3.10.8" \
    "openmpi=5.0" "hdf5=1.14.6=mpi_openmpi*" "h5py=*=mpi_openmpi*" \
    "gsl=2.8" "fftw=3.3.10" "libctl=4.5.1" "harminv=1.4.2" "mpb=1.12.0" "swig=4.4.1" \
    autoconf automake libtool m4 pkg-config make \
    "gcc_linux-64=15.2.0" "gxx_linux-64=15.2.0" "gfortran_linux-64=15.2.0" \
    "libblas=3.11.0=*openblas" liblapack \
    "cupy=13.5.1" "cuda-version=11.8"
else
  echo "  installing $(grep -c '^https://' "$LOCK_CONDA") packages from $LOCK_CONDA, without a solve"
  "$CONDA_ROOT/bin/conda" create -y -p "$ENV_PREFIX" --no-default-packages --file "$LOCK_CONDA"
fi
# The Fortran compiler is not optional although MEEP has no Fortran in it: configure
# runs AC_F77_WRAPPERS to work out the BLAS/LAPACK name-mangling scheme.
# Nor is matplotlib, although nothing here plots: MEEP 1.33.0's simulation module
# imports it when meep is imported, so without it `import meep` fails.

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
ACTUAL=$(sha256sum meep.tar.gz | awk '{print $1}')
echo "  expected $MEEP_SHA256"
echo "  actual   $ACTUAL"
[ "$ACTUAL" = "$MEEP_SHA256" ] || { echo "  SHA-256 MISMATCH: refusing to build"; exit 1; }
tar xzf meep.tar.gz
cd "meep-${MEEP_VERSION}"

say "3. patch AX_CXX_MAXOPT -> AX_CC_MAXOPT (defect 1) and regenerate autotools"
if grep -q 'AX_CXX_MAXOPT' configure.ac; then
  sed -i 's/AX_CXX_MAXOPT/AX_CC_MAXOPT/' configure.ac
  echo "  patched configure.ac"
else
  echo "  configure.ac does not call AX_CXX_MAXOPT: upstream may have fixed it"
fi
export PATH="$ENV_PREFIX/bin:$PATH"
autoreconf -fi

say "4. configure IN-SOURCE with --enable-single (defect 2)"
# The environment is never activated: the build runs on explicit paths, so the conda
# compilers are named here and handed to Open MPI's wrappers as well.
first() { ls $1 2>/dev/null | head -1; }
CCX="$(first "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc")"
CXXX="$(first "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++")"
FCX="$(first "$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gfortran")"
for tool in "$CCX" "$CXXX" "$FCX"; do
  [ -n "$tool" ] || { echo "  a conda compiler is missing from $ENV_PREFIX/bin (step 1)"; exit 1; }
done
# As the reference was configured (INSTALL.md, "Building MEEP in single precision"):
# the compilers themselves as CC/CXX/F77, and Open MPI's C++ wrapper as MPICXX, told
# which compiler to wrap since the environment is not activated.
export OMPI_CC="$CCX" OMPI_CXX="$CXXX" OMPI_FC="$FCX"
export CC="$CCX" CXX="$CXXX" MPICXX="$ENV_PREFIX/bin/mpicxx"
export F77="$FCX" FC="$FCX"
export CPPFLAGS="-I$ENV_PREFIX/include"
export LDFLAGS="-L$ENV_PREFIX/lib -Wl,-rpath,$ENV_PREFIX/lib"
export PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig"
export PYTHON="$ENV_PREFIX/bin/python"
# NO separate build/ directory: see defect 2.
./configure \
  --prefix="$ENV_PREFIX" \
  --enable-shared --enable-single --enable-portable-binary \
  --with-mpi --with-openmp --without-scheme \
  --with-libctl="$ENV_PREFIX/share/libctl" \
  CC="$CC" CXX="$CXX" MPICXX="$MPICXX" F77="$F77" \
  PYTHON="$ENV_PREFIX/bin/python"

say "5. build (-j$JOBS)"
make -j"$JOBS"

say "6. install"
make install

say "7. the certified PyTorch and Triton"
# PyTorch 2.5.1 is the release that requires Triton 3.1.0, the certified Triton.
# Never `--upgrade` PyTorch in this environment: it replaces Triton.
# From the lock: the 22 PyPI packages of the reference (PyTorch, Triton and their
# dependencies, the CUDA 12.4 libraries PyTorch loads among them, and autograd, which
# MEEP's adjoint module imports), each by version and SHA-256, with no dependency
# resolution (--no-deps); pip check then confirms that nothing is missing. -s keeps the
# user site directory out of the install and the check.
if [ "$SOLVE" = 1 ]; then
  echo "  $NOT_REFERENCE."
  "$ENV_PREFIX/bin/python" -m pip install "torch==2.5.1" "triton==3.1.0"
else
  "$ENV_PREFIX/bin/python" -s -m pip install --no-deps -r "$LOCK_PIP"
  "$ENV_PREFIX/bin/python" -s -m pip check
fi

say "8. MANDATORY assertions: the reference this environment claims to be"
PYTHONPATH="$REPO_ROOT" "$ENV_PREFIX/bin/python" -s - <<'PYEOF'
import sys
import meep as mp
from meep_gpu import backends
flushed = backends.subnormals_flushed()     # read before any other library loads
import numpy, cupy, triton
nvrtc = ".".join(str(part) for part in cupy.cuda.nvrtc.getVersion())
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
# Then the drift report a user runs (tools/compare_reference_environment.py), on the
# environment just built: every numerics-relevant package against the lock.
if [ "$SOLVE" = 1 ]; then
  echo "  $NOT_REFERENCE; the comparison with the reference is informational:"
  "$ENV_PREFIX/bin/python" -s "$REPO_ROOT/tools/compare_reference_environment.py" || true
else
  "$ENV_PREFIX/bin/python" -s "$REPO_ROOT/tools/compare_reference_environment.py" --require-reference
fi

say "DONE"
echo "Activate it with:  conda activate $ENV_NAME"
if [ "$SOLVE" = 1 ]; then echo "$NOT_REFERENCE."; fi
echo "Then, from the repository root, use it in place of environments/nvidia-linux.yml."
