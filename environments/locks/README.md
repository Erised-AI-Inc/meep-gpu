# Certification reference locks

The environments every certification round is measured against, locked. The three
`*.yml` files one directory up are the routes for **using** the package (conda-forge's
double-precision `pymeep`). The files here are the route for **certifying** it: MEEP
1.33.0 built from source in single precision with MPI, on top of exactly these packages.

| File | What it holds |
|---|---|
| `meep-gpu-ref-osx-arm64.conda.txt` | Apple silicon: an explicit conda lock, 138 packages, each by URL and MD5 |
| `meep-gpu-ref-osx-arm64.pip.txt` | Apple silicon: 13 PyPI packages, each by version and wheel SHA-256 |
| `meep-gpu-ref-linux-64.conda.txt` | Linux x86_64: an explicit conda lock, 143 packages, each by URL and MD5 |
| `meep-gpu-ref-linux-64.pip.txt` | Linux x86_64: 22 PyPI packages, each by version and wheel SHA-256 |
| `numerics-relevant.json` | per platform, the packages whose builds can change what MEEP or the GPU route computes, and the documented exceptions |

`parity/meep_gpu/build_meep_133_macos.sh` and `build_meep_133_linux.sh` install these
files without solving (`conda create --no-default-packages --file <lock>`, then
`pip install --no-deps -r <pip lock>`), refuse to build MEEP unless the new prefix holds
exactly the lock's conda artifacts, and end by running the drift report below with
`--require-reference`.

## What each lock reproduces

**Apple silicon (`osx-arm64`).** The conda environment on the Apple M1 Max (macOS 26.2)
in which the Metal kernels were certified, restricted to the packages the reference
needs; read from that environment on 2026-10-03.

- 127 of the 138 conda artifacts are the ones that environment installed (same URL,
  same MD5), libctl 4.5.1, FFTW 3.3.10, harminv 1.4.2, GSL 2.8, Open MPI 5.0.10,
  OpenBLAS 0.3.30 (OpenMP build, with the `openblas` package that installs
  `libopenblas.dylib`, so MEEP's configure links it as the certified build did), llvm-openmp 22.1.0, the gfortran 15.2.0 runtime,
  libc++ 22.1.1, Python 3.12.13, NumPy 2.4.3 and SciPy 1.17.1 among them.
- **hdf5 and h5py differ by design.** The certified environment holds the nompi build of
  hdf5 1.14.6 (build 106) and the PyPI wheel of h5py 3.14.0; the lock holds the
  mpi_openmpi builds (hdf5 build 6, h5py 3.14.0), because with a nompi HDF5, multi-rank
  MEEP output deadlocks. One process, which is all the package uses, never reaches
  parallel HDF5. The drift report reports the certified environment's own nompi hdf5
  (that build and MD5, named in `numerics-relevant.json`) as "differs by design", not
  as drift; any other hdf5 that is not the lock's is drift.
- 9 packages the build needs and the certified environment lacks: the C++ driver of its
  compiler build (`clangxx_impl_osx-arm64` 22.1.0), the compiler activation packages
  (`clang_osx-arm64`, `clangxx_osx-arm64`, `gcc_osx-arm64`, `gfortran_osx-arm64`,
  `sdkroot_env_osx-arm64`), the libc++ headers that match its libc++ runtime
  (`libcxx-headers`, `libcxx-devel` 22.1.1) and `make`.
- The PyPI part is the certified environment's PyPI packages at their versions: PyTorch
  2.10.0 (the build-tag-2 wheel, `torch-2.10.0-2-cp312-none-macosx_11_0_arm64.whl`, which
  is the one installed there; PyPI's earlier untagged 2.10.0 wheel is not admitted, and
  the drift report reads the installed wheel's build tag) and its dependencies, pytest
  with its dependencies, and autograd 1.8.0 ([autograd](#autograd-added-2026-10-05)).
- **MEEP links the same BLAS.** The lock holds the certified environment's `openblas`
  0.3.30 (`openmp_hea878ba_4`), which installs `lib/libopenblas.dylib`, so MEEP's configure
  sets `BLAS_LIBS = -lopenblas` and `libmeep` and `_meep.so` load OpenBLAS and not Apple's
  Accelerate, as the certified build does (MEASURED 2026-10-04, the kept Makefile and
  `otool -L`). An earlier version of the lock left `openblas` out; its MEEP carried an
  Accelerate load command it never called.

**Linux x86_64 (`linux-64`).** The environment the 2026-10-03 RTX A6000 certification
round ran in, verbatim: all 143 conda packages and all 21 PyPI packages (PyTorch 2.5.1,
Triton 3.1.0, the CUDA 12.4 libraries PyTorch loads, and their dependencies). It was
created on 2026-10-02 by the build script's former solve and was not changed by conda
afterwards. It is not the older environment the NVIDIA kernels were first certified in;
the build script's step 1 lists where the two differ (Open MPI, h5py, NVRTC). One PyPI
package was added to the lock afterwards and installed into that environment from it,
autograd 1.9.1 ([autograd](#autograd-added-2026-10-05)), making 22.

## What a lock cannot capture

- **MEEP itself.** It is built from the release tarball (SHA-256
  `bdabc0a112f669f2657fbdca22e31a2aeed515372e8ed56dc55197a4cd7ff0ab`), in the source
  tree, with the configure flags the build scripts carry. A rebuild embeds its prefix's
  path, so it is never byte-identical to another build. On Apple silicon, the 16
  field-update kernels of the certified MEEP have the same instruction sequences as those
  of three other builds: two by the build script's former solve and one by the build
  script from this lock (immediates and addresses masked; 16 of 16 in each; MEASURED
  2026-10-03). Code outside the kernels differs: the HDF5 output code (MPI HDF5, by
  design), the C++ standard-library templates (the certified MEEP was compiled against
  the macOS SDK's libc++ headers, INFERRED; the lock compiles against conda's
  `libcxx-headers`).
- **The macOS SDK.** MEEP is compiled against the host's SDK (Command Line Tools or
  Xcode), not a package of the lock: the lock's `sdkroot_env_osx-arm64` ships no SDK
  (activated, it points `SDKROOT` at the host's), and the build script does not
  activate the environment.
  The certified `libmeep`, and the one the build script made from this lock on the same
  Mac, record SDK 26.2 and minimum macOS 26.0 (MEASURED 2026-10-03, `otool -l`). The
  build script keeps what the new `libmeep` records in
  `share/meep-gpu-build/macos_sdk.txt`.
- **The host.** An explicit install never checks it. The macOS lock needs macOS 11.0 or
  later; the Linux lock's compilers target glibc 2.28 (`sysroot_linux-64` 2.28), so it
  needs glibc 2.28 or later. The A6000 round ran on NVIDIA driver 545.23.08 (CUDA 12.3)
  and glibc 2.31; CuPy uses the conda CUDA 11.8 runtime, and PyTorch's CUDA 12.4
  libraries ran there under CUDA minor-version compatibility (INFERRED from NVIDIA's
  compatibility policy; the round released). The GPU, and on a Mac the Metal frontend
  (one per macOS build), belong to the host too; the package reports both.
- **The user site directory.** Packages under `~/.local` come before the environment's
  own on its import path. The build scripts run Python with `-s`; run the reference the
  same way, or set `PYTHONNOUSERSITE=1`.

## Is my environment the reference?

    python tools/compare_reference_environment.py                      # informational
    python tools/compare_reference_environment.py --require-reference  # exit 1 unless it matches

It reads the running interpreter's environment (`--prefix PATH` reads another conda
environment) and prints, for every package of `numerics-relevant.json`, `same`,
`different` (with both builds), `missing`, `differs by design` or `not read`, then one
verdict: the environment matches the certification reference, differs from it in N of D
numerics-relevant packages (named), or cannot be confirmed to match it because some
packages could not be read. Conda packages are judged by version and build, PyPI
packages by version, origin (a conda build of PyTorch is not the PyPI wheel) and, for
PyTorch on Apple silicon, the wheel's build tag. Outside a conda environment the
compiled libraries MEEP links cannot be read and are reported as `not read`, while the
PyPI builds of NumPy, SciPy and mpi4py are reported as `different` from the reference's
conda builds, so a pip-only install is reported as differing. A prefix that a build
script solved (`MEEP_REFERENCE_SOLVE=1`) is named in a note.

The report judges packages only. Two things it does not judge, by design:

- **MEEP itself**, which is built from source and is not a package of the lock.
  `tools/check_install.py --require-reference` judges it: its step 7 prints the same
  report, preceded by a line that compares the imported MEEP with the build the
  reference names (1.33.0, single precision, MPI build, from `numerics-relevant.json`),
  and fails unless both match. A host joining a certification round runs that.
- **The compilers and the sysroot**, which decide how MEEP is compiled and are not
  loaded when it runs. The build scripts' step 1b holds a build to every artifact of the
  lock, compilers included; the drift report judges the environment MEEP runs in.

The numerics-relevant set of Apple silicon is MEASURED: the libraries the certified
`libmeep` and its Python module load, the BLAS/LAPACK and OpenMP runtimes NumPy and MEEP
reach, the MPI stack and the Python numerics (29 packages). The Linux set (36 packages) was
carried over from it, with the CUDA libraries of the GPU route added, and then checked
against the round environment's `libmeep.so` and `_meep.so` (`readelf -d`, 2026-10-03):
every library they load — libctlgeom, HDF5, GSL, OpenBLAS, MPB, harminv, FFTW,
libgfortran, libquadmath, Open MPI, libstdc++, libgomp, libgcc_s — belongs to a judged
package.

## How the locks were made

**Linux.** The conda part is the round environment's
`conda list --explicit --md5 -p <prefix>`, its generated header replaced by the one in
the file; every URL returned HTTP 200 and matched conda-forge's MD5 and SHA-256 (143 of
143). The PyPI part lists the packages of that environment's own `site-packages`
(`python -s -m pip list`), each pinned to the PyPI wheel whose tags match the installed
copy; every installed file was compared with that wheel's `RECORD` (21 of 21 identical).
autograd was added later ([below](#autograd-added-2026-10-05)).

**Apple silicon.** Assembled, then proved closed:

1. `conda list --explicit --md5 -p <prefix>` of the certified environment, and its
   PyPI distributions.
2. A spec pinning every package the build script's environment shares with the certified
   one to the certified artifact (`name==version=build`), hdf5 and h5py to their
   mpi_openmpi builds, and the build script's remaining top-level packages (the compiler
   activation packages, `make`) without a version; the certified environment's PyPI
   packages are left to the pip part.
3. A dry-run solve of that spec against conda-forge alone (`conda create --dry-run
   --json`) gave 137 packages. Fed back as 137 exact specs, the solve gave the same 137,
   none added or dropped: the closure an explicit install needs, since it never checks
   dependencies.
4. The lock lines are the certified environment's own export lines where the artifact is
   the same (126), and the solve's URLs and MD5s for the other 11.
5. `openblas`, which the solve does not pull in, was added from the certified
   environment's export (2026-10-04), making 127 of 138. The PyPI hashes come
   from PyPI's JSON API for the certified versions.

## autograd (added 2026-10-05)

MEEP's adjoint module (`meep.adjoint`) imports `autograd`, and so do MEEP's adjoint tests.
Neither lock held it, so in an environment built from a lock those tests fail on import.
The first corpus lift in the Linux round environment, on 2026-10-05, lost 13 of their rows
across four families, and 8 of its 60 legs did not release on those rows (one also on a
timeout; MEASURED). Both locks now pin it, to the PyPI wheel by SHA-256: the digest PyPI's
JSON API gives, and the digest of the wheel as downloaded (MEASURED 2026-10-05).

- **Apple silicon: 1.8.0**, the version the certified environment holds; its installed
  files are the 1.8.0 wheel's (39 of 39, MEASURED).
- **Linux: 1.9.1**, the version the environment of the earlier A6000 rounds holds, whose
  corpus lifts kept those rows; its installed files are the 1.9.1 wheel's (40 of 40,
  MEASURED). The round environment lacked it, so this line was installed into that
  environment from the lock (`pip install --no-deps --require-hashes`), not read from it:
  the one entry of either lock that went from the lock to the environment.

autograd requires only `numpy<3`, which both locks satisfy, and takes no part in the field
updates, so `numerics-relevant.json` does not name it and the drift report does not judge
it; `pip check` in the build scripts' step 7 catches a lock that leaves out a
requirement.

## When to re-cut

Re-cut a platform's lock whenever its certified environment changes: a package added,
removed, upgraded or rebuilt, or a round certified in another environment. Re-cut from
that environment, as above, in the same change that records the round, and edit
`numerics-relevant.json` when the set of judged packages changes. A lock that conda-forge
or PyPI can no longer serve (a URL that stops resolving) also has to be re-cut, from the
environment that is certified at that time; never re-solve a lock and call the result the
reference.
