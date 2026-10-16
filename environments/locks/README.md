# Reference environment locks

The one environment per platform that the package's tests, a certification round and the
timing run in, locked. The three `*.yml` files one directory up are the routes for
**using** the package (conda-forge's double-precision `pymeep`). The files here are the
route for **certifying** it: MEEP 1.33.0 built from the release tarball in single
precision with MPI, libGDSII and no floating-point contraction, on top of exactly these
packages, by the repository's own build script.

| File | What it holds |
|---|---|
| `meep-gpu-ref-osx-arm64.conda.txt` | Apple silicon: an explicit conda lock, 141 packages, each by URL and MD5 |
| `meep-gpu-ref-osx-arm64.pip.txt` | Apple silicon: 17 PyPI packages, each by version and wheel SHA-256 |
| `meep-gpu-ref-linux-64.conda.txt` | Linux x86_64: an explicit conda lock, 147 packages, each by URL and MD5 |
| `meep-gpu-ref-linux-64.pip.txt` | Linux x86_64: 26 PyPI packages, each by version and wheel SHA-256 |
| `numerics-relevant.json` | per platform, the packages whose builds can change what MEEP or the GPU route computes, and the documented exceptions; apart (`corpus_only`), what only the MEEP identity corpus needs, the PyMieScatt stand-in's pins among them |
| `additions.json` | what the locks hold beyond the environments they were read from, each pinned to one artifact, with its role and what needs it |
| `recut_locks.py` | the tool that writes the locks from their own lines and `additions.json` (below) |

`parity/meep_gpu/build_meep_133_macos.sh` and `build_meep_133_linux.sh` build the
environment end to end, in a cleared environment (`env -i`, the user site directory off,
no pip configuration file, and no pip wheel cache: `PIP_NO_CACHE_DIR=1`, since every pin is
installed once, by hash):

1. install the conda lock without solving (`conda create --no-default-packages --file
   <lock>`) and refuse to go on unless the new prefix holds exactly the lock's artifacts;
2. fetch the MEEP 1.33.0 release tarball, refuse it unless its SHA-256 is
   `bdabc0a112f669f2657fbdca22e31a2aeed515372e8ed56dc55197a4cd7ff0ab`, and keep its
   `python/examples` and `python/tests` in the prefix as the MEEP identity corpus, with
   the SHA-256 of every file, and the corpus siblings (files outside `python/` that
   corpus rows read) beside it;
3. build MEEP in the source tree (`--enable-single --with-mpi`, `--with-openmp` on Linux;
   `-O2 -ffp-contract=off -fno-fast-math`, and `-march=x86-64 -mtune=generic` on Linux),
   refusing a configure that did not find libGDSII;
4. install the pip lock (`pip install --no-deps -r <pip lock>`), build and install the
   PyMieScatt stand-in, and refuse to go on unless the prefix's PyPI packages are exactly
   the pip lock's and the stand-in (`--require-pip-lock`, below);
5. link the imageio-ffmpeg pin's executable into `<prefix>/bin/ffmpeg`, refusing if the
   load commands of `libmeep` and `_meep` changed since they were installed;
6. refuse the build unless MEEP is 1.33.0, single precision, MPI and reads a GDSII file,
   the drift report below matches (`--require-reference`), the fused multiply-add census
   of `libmeep` and `_meep` is clean, MEEP's own words on four cases reproduce the
   recorded digests, and the identity verdict of `tools/check_install.py` passes in keep
   and in flush (on Apple silicon, flush is reached through the processor's FZ bit, and a
   check that does not drive that bit refuses it by name; the build records that refusal
   alone as pending);
7. refuse a program the round runs (`git`, `ffmpeg`, `mpirun`, the disassembler and
   demangler of the census, and on Linux the C compiler Triton's launcher is built with)
   that is not found in `<prefix>/bin`, and write the record of the build.

The record lives in `<prefix>/share/meep-gpu-build/`: these lock files, the corpus
(`meep-1.33.0/python/`) and its list (`corpus_sha256.txt`), the corpus siblings beside it
(the files outside `python/` that corpus rows read, under `meep-1.33.0/`) and their list
(`corpus_siblings_sha256.txt`), `config.log`, the
`Makefile`, `build_record.json`, `fma_census.json`, `load_commands.json`, `ffmpeg.json`,
the drift report (`reference_environment.json`), the digest smoke's rows, the identity
verdicts (`identity_{keep,flush}.*`), `tools.json`, on Linux `libcuda.json`, and
`reference_build.json`, which names all of them with the tarball's digest and the flags.
The build script's `--refresh-corpus` brings the corpus of an existing prefix to what the
build keeps now, rewriting only the record's `corpus` entry and recording the change in
`corpus_refresh.json` ([INSTALL.md](../../INSTALL.md)).

## What each lock holds

**Apple silicon (`osx-arm64`).** The conda environment on the Apple M1 Max (macOS 26.2)
in which the Metal kernels were certified, restricted to the packages the reference
needs (read from that environment on 2026-10-03), and the additions below.

- 127 of the first 138 conda artifacts are the ones that environment installed (same
  URL, same MD5), libctl 4.5.1, FFTW 3.3.10, harminv 1.4.2, GSL 2.8, Open MPI 5.0.10,
  OpenBLAS 0.3.30 (OpenMP build, with the `openblas` package that installs
  `libopenblas.dylib`, so MEEP's configure links it as the certified build did),
  llvm-openmp 22.1.0, the gfortran 15.2.0 runtime, libc++ 22.1.1, Python 3.12.13, NumPy
  2.4.3 and SciPy 1.17.1 among them.
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
  with its dependencies, autograd 1.8.0 ([autograd](#autograd-added-2026-10-05)) and the
  corpus-only shapely and nlopt.
- **MEEP links the same BLAS.** The lock holds the certified environment's `openblas`
  0.3.30 (`openmp_hea878ba_4`), which installs `lib/libopenblas.dylib`, so MEEP's configure
  sets `BLAS_LIBS = -lopenblas` and `libmeep` and `_meep.so` load OpenBLAS and not Apple's
  Accelerate, as the certified build does (MEASURED 2026-10-04, the kept Makefile and
  `otool -L`).
- The additions ([below](#what-the-environment-carries-added-2026-10-14)): libgdsii
  0.21, gdspy 1.6.13 and git 2.53.0 (conda), llvmlite 0.47.0 and imageio-ffmpeg 0.6.0
  (PyPI).

**Linux x86_64 (`linux-64`).** The environment the 2026-10-03 RTX A6000 certification
round ran in, verbatim: its 143 conda packages and its 21 PyPI packages (PyTorch 2.5.1,
Triton 3.1.0, the CUDA 12.4 libraries PyTorch loads, and their dependencies). It was
created on 2026-10-02 by the build script's former solve and was not changed by conda
afterwards. It is not the older environment the NVIDIA kernels were first certified in;
the build script's step 1 lists where the two differ (Open MPI, h5py, NVRTC). autograd
1.9.1 joined the pip lock on 2026-10-05 ([autograd](#autograd-added-2026-10-05)), shapely
and nlopt on 2026-10-11 ([corpus-only packages](#corpus-only-packages-added-2026-10-11)),
and the additions on 2026-10-14: libgdsii 0.21, gdspy 1.6.13, git 2.55.0 and binutils
2.46.1 (conda), llvmlite 0.47.0 and imageio-ffmpeg 0.6.0 (PyPI).

## What the environment carries (added 2026-10-14)

The environment is everything the package, its tests (every suite), the round's drivers,
gates, censuses, boards and record tools, the direct corpus check (L3), the MEEP identity
corpus (all 194 rows of Table 2, `ex.coupler` and `ex.ring_gds` included) and the timing
harness import or run. That set was found by an inventory of the tree and of the
tarball's `python/` tree (every import of every file, read without running it, and the
program at the head of every subprocess call, every `shutil.which`, every shell script),
not by memory. What the locks lacked, they now hold, each in `additions.json`:

| Addition | Lock | Role | Needed by |
|---|---|---|---|
| libgdsii 0.21 | conda | MEEP links it (`numerics-relevant.json`, `meep_links`) | MEEP's configure links libGDSII when it finds it; without it `mp.get_GDSII_prisms` and `mp.GDSII_vol` abort, and `ex.coupler` and `ex.ring_gds` cannot run |
| gdspy 1.6.13 | conda | corpus only | `ex.ring_gds` imports it; PyPI has no wheel of it for either platform |
| git | conda | tool | the chunk tooling, the boards, L3, the hygiene guard and their tests |
| binutils 2.46.1 (Linux) | conda | tool | the unprefixed `objdump` and `c++filt` the census runs, and the `ar`, `nm`, `ranlib` and `strip` MEEP's build runs, which otherwise came from the host; the same binutils the lock's compilers use |
| llvmlite 0.47.0 | PyPI | tests | `meep_gpu/subnormal_policy.py` hands the Triton IR it rewrites to LLVM's own parser when llvmlite imports; 5 tests skip by the declared resource `[llvmlite]` without it |
| imageio-ffmpeg 0.6.0 | PyPI | corpus only | the `ffmpeg` executable `test_animation_output` needs on the reference `PATH`; the wheel installs it and no shared library |

Adding them moved none of the packages already locked: `recut_locks.py` solved every lock
line pinned exactly together with the additions and got exactly the lock plus the
additions on both platforms (141 = 138 + 3, 147 = 143 + 4; 0 moved, 0 dropped). Every
added conda artifact answered with conda-forge's MD5 and SHA-256, and every added wheel's
SHA-256 is PyPI's and the downloaded file's (MEASURED 2026-10-14).

**ffmpeg.** conda-forge's ffmpeg is not used: it would bring shared libraries into the
`<prefix>/lib` MEEP links against. The imageio-ffmpeg wheel carries one executable and no
shared library: on Linux it is statically linked (no program interpreter, no dynamic
section), on Apple silicon it loads only `/usr/lib` and `/System/Library` (MEASURED
2026-10-14, from the wheels' headers). It is installed after MEEP is, and the build checks
that `libmeep` and `_meep` keep the load commands they were installed with.

**What the environment does not carry, on purpose:**

- `parameterized`: six corpus test modules import it, and the harness supplies its own
  (`parity/meep_gpu/shim/parameterized.py`), which stamps the generated methods the census
  rows are joined on; the PyPI package would break that join.
- `jax`, `IPython`, `ipywidgets`, `vispy`, `scikit-image`, `imageio`: MEEP imports them
  only inside functions no corpus row, test or gate calls (a 3-D plot, notebook display),
  or in a guarded import (`simulation.py`), and only `test_adjoint_jax`, which is not a
  corpus row, imports jax.
- A plain `gcc` on Linux: conda-forge's `gcc` package brings `conda-gcc-specs`, which
  changes the compiler's defaults. Triton builds its launcher with `$CC`, and a round
  passes the environment's own `x86_64-conda-linux-gnu-cc`; the build script compiles the
  launcher that way once to show it links.

## What the host supplies

Three things, and nothing else:

- **The GPU driver.** On Linux the NVIDIA driver: `libcuda.so.1` (Triton's launcher links
  `-lcuda`, so the Linux build links the driver's `libcuda.so.1` into
  `<prefix>/lib/meep-gpu-libcuda/` as `libcuda.so`, the directory a round passes as
  `TRITON_LIBCUDA_PATH`) and `nvidia-smi`. The A6000 round ran on driver 545.23.08
  (CUDA 12.3); CuPy uses the conda CUDA 11.8 runtime, and PyTorch's CUDA 12.4 libraries
  ran there under CUDA minor-version compatibility (INFERRED from NVIDIA's compatibility
  policy; the round released). On a Mac the Metal stack is macOS's own; the Metal
  frontend is one per macOS build, and the package reports it.
- **The macOS SDK** (Command Line Tools or Xcode). MEEP is compiled against the host's
  SDK: the lock's `sdkroot_env_osx-arm64` ships none. The certified `libmeep`, and the
  one the build script made from this lock on the same Mac, record SDK 26.2 and minimum
  macOS 26.0 (MEASURED 2026-10-03, `otool -l`); the build keeps what the new `libmeep`
  records in `share/meep-gpu-build/macos_sdk.txt`. The lock needs macOS 11.0 or later.
- **glibc.** The Linux lock's compilers target glibc 2.28 (`sysroot_linux-64` 2.28), so
  the host needs glibc 2.28 or later; the build script refuses an older one. The A6000
  round ran on glibc 2.31.

Each comes with the operating system it belongs to, and with it that system's own base
programs, which the scripts and tools call by name and the environment does not replace
(a POSIX shell and its utilities, `ps`, `df`, `hostname`, `uname`, `which`, `tar`; on
Linux `ldconfig`, `getconf` and `lscpu`; on macOS `zsh`, `sysctl`, `sw_vers`, `pmset`,
`ioreg`, `vm_stat`, `caffeinate` and the SDK's `xcrun`). `sysctl` and `ioreg` live in
`/usr/sbin`, which the reference `PATH` (`<prefix>/bin:/usr/bin:/bin`) does not hold, so
the tools that read them on a Mac (`tools/check_install.py`, the timing harness's host
readers) call them by absolute path. The build itself runs from a
conda installation (`CONDA_ROOT`) with network access to conda-forge, PyPI and the MEEP
release; neither is part of the environment it makes. Nothing comes from `~/.local`, a
Homebrew prefix or `/usr/local`: the build clears `PATH` to `/usr/bin:/bin:/usr/sbin:/sbin`
and runs Python without the user site directory. Run the reference the same way
(`python -s`, or `PYTHONNOUSERSITE=1`): packages under `~/.local` come before the
environment's own on its import path.

## Is my environment the reference?

    python tools/compare_reference_environment.py                      # informational
    python tools/compare_reference_environment.py --require-reference  # exit 1 unless it matches

It reads the running interpreter's environment (`--prefix PATH` reads another conda
environment) and prints, for every package of `numerics-relevant.json` (30 on Apple
silicon, 37 on Linux), `same`, `different` (with both builds), `missing`, `differs by
design` or `not read`, then one verdict: the environment matches the certification
reference, differs from it in N of D numerics-relevant packages (named), or cannot be
confirmed to match it because some packages could not be read. Conda packages are judged
by version and build, PyPI packages by version, origin (a conda build of PyTorch is not
the PyPI wheel) and, for PyTorch on Apple silicon, the wheel's build tag. Outside a conda
environment the compiled libraries MEEP links cannot be read and are reported as `not
read`, while the PyPI builds of NumPy, SciPy and mpi4py are reported as `different` from
the reference's conda builds, so a pip-only install is reported as differing. A prefix a
build script once solved (the retired `MEEP_REFERENCE_SOLVE=1`) is named in a note.

Before the verdict it lists the [corpus-only packages](#corpus-only-packages-added-2026-10-11)
apart: shapely, nlopt and imageio-ffmpeg by version and origin, gdspy by version and build,
the PyMieScatt stand-in by version and the SHA-256 of its five files (PyPI's own 1.8.1.1
is not the stand-in), and `ffmpeg` by where `which` finds it on the reference `PATH`
(`<prefix>/bin:/usr/bin:/bin`): held only when it resolves into the imageio-ffmpeg pin's
`imageio_ffmpeg/binaries/` (a host's own ffmpeg is another build), each `held`,
`absent` or `another build`. They never count in the verdict: an environment without them
is judged on its numerics exactly as one with them, and a user who does not re-run the
identity corpus never needs them.

    python tools/compare_reference_environment.py --prefix <prefix> --require-pip-lock

exits 1 unless the PyPI packages installed in the prefix (every distribution an installer
other than conda put there) are exactly the pip lock's pins at their versions and the
stand-in with its files, none missing and none added. The build scripts run it after
installing them, beside step 1b's artifact-for-artifact check of the conda part.

The report judges packages only. Two things it does not judge, by design:

- **MEEP itself**, which is built from source and is not a package of the lock.
  `tools/check_install.py --require-reference` judges it: its step 7 prints the same
  report, preceded by a line that compares the imported MEEP with the build the
  reference names (1.33.0, single precision, MPI build, libGDSII, from
  `numerics-relevant.json`; libGDSII as MEEP's own `with_libGDSII()` reports it), and
  fails unless both match. A host joining a certification round runs that. The build's own record
  (`reference_build.json`) holds the rest: the images' SHA-256, the census, the flags and
  the identity verdicts.
- **The compilers, the tools and the sysroot**, which decide how MEEP is compiled and are
  not loaded when it runs. The build scripts hold a build to every artifact of the lock,
  compilers and tools included; the drift report judges the environment MEEP runs in.

The numerics-relevant set of Apple silicon is MEASURED: the libraries the certified
`libmeep` and its Python module load, the BLAS/LAPACK and OpenMP runtimes NumPy and MEEP
reach, the MPI stack and the Python numerics (29 packages), and libgdsii (30), which the
reference MEEP links since it is built with libGDSII (`otool -L` of a libGDSII build of
MEEP 1.33.0 lists `@rpath/libGDSII.0.dylib`, MEASURED 2026-10-14). The Linux set (37) was
carried over from it, with the CUDA libraries of the GPU route added, and then checked
against the round environment's `libmeep.so` and `_meep.so` (`readelf -d`, 2026-10-03):
every library they load (libctlgeom, HDF5, GSL, OpenBLAS, MPB, harminv, FFTW,
libgfortran, libquadmath, Open MPI, libstdc++, libgomp, libgcc_s) belongs to a judged
package; libGDSII joins them by configure's rule, and each build records its images'
NEEDED lists (`load_commands.json`).

## The field-update kernels across builds

A rebuild links its prefix's path into `libmeep` and `_meep`, so it is never byte-identical
to another build, and what makes a build the reference is judged on what was built: the
census, the images' SHA-256 in the record, and MEEP's own words on four cases. Beside those
checks, a new reference build is compared with the build it replaces where the stored words
are made: the machine code of MEEP's 16 field-update kernels (`tools/fma_census.py`'s
`KERNELS`: `step_curl`, `step_beta`, `step_bfast` and `step_update_EDHB` with their
`_stride1` forms, `fields_chunk::step_db`, `update_eh`, `update_pols` and `step_source`, and
the lorentzian, noisy lorentzian, gyrotropic and multilevel susceptibilities' `update_P`),
disassembled, matched by demangled name, and compared as instruction sequences with
addresses masked. No tool in the repository runs it: it is a review step, and these are its
measurements.

- **Apple silicon, the 1.0 reference environment (2026-10-14).** The new
  `libmeep.37.dylib` against the contraction-off build of 2026-10-05 that the earlier 1.0
  round's Metal L3 leg loaded (SHA-256 `933cd385…`): the same clang 22.1.0, the same
  `CFLAGS` and `CXXFLAGS` (`-O2 -ffp-contract=off -fno-fast-math`) and configure flags,
  `config.h` differing only by `HAVE_LIBGDSII`. Disassembled by the environment's own
  `llvm-objdump --no-show-raw-insn --no-leading-addr`, with immediates, `0x` constants and
  `<symbol>` targets masked: 16 of 16 kernels found in each, their instruction counts 16 of
  16 equal and their masked sequences 16 of 16 equal. Unmasked, 0 of 16 are equal, and
  every differing line is a branch, a call, an `adr` or a PC-relative load whose absolute
  address differs; with only the absolute addresses masked (immediates, registers, call
  targets and in-function offsets kept) 16 of 16 are equal. The kernels sit at other
  addresses, moved by the libGDSII code, and nothing else in them differs.
- **Linux x86_64, the 1.0 reference environment (2026-10-14).** The new
  `libmeep.so.37.0.0` against the previous Linux reference environment's (the 2026-10-03
  RTX A6000 round's), both built by conda-forge's gcc 15.2.0 with `-O2 -march=x86-64
  -mtune=generic -fopenmp` (the previous one also with `-g`, which changes no code
  generation); the new one adds `-ffp-contract=off -fno-fast-math`, which change no code
  where the instruction set has no fused multiply-add (the census reads 0 fused
  instructions in both). Disassembled by GNU objdump 2.46.1: 158 functions in each,
  matched one to one by name (the 145 bodies of the 16 kernels with their OpenMP, `.cold`
  and `.isra` clones, and 13 PLT stubs). With addresses masked (branch and call targets,
  RIP-relative displacements, objdump's comments) and every immediate kept, 145 of 145
  bodies and 72,820 of 72,820 instructions are identical; with immediates masked too, 158 of
  158 (the PLT stubs differ only in their relocation index, since libGDSII adds entries).
  Of the 418 RIP-relative operands in the bodies, 413 point at equal bytes; the other 5
  load switch tables whose entries all differ by one offset, code addresses again.
- **Apple silicon, 2026-10-03.** The 16 kernels of the MEEP the 0.9.x Metal kernels were
  certified against had the same instruction sequences as those of three other builds (two
  by the build script's former solve, one by the build script from this lock), immediates
  and addresses masked, 16 of 16 in each.

## How the locks were made

**Linux.** The first 143 conda lines are the round environment's
`conda list --explicit --md5 -p <prefix>`; every URL returned HTTP 200 and matched
conda-forge's MD5 and SHA-256 (143 of 143). The first 21 PyPI pins list the packages of
that environment's own `site-packages` (`python -s -m pip list`), each pinned to the PyPI
wheel whose tags match the installed copy; every installed file was compared with that
wheel's `RECORD` (21 of 21 identical). autograd was added later
([below](#autograd-added-2026-10-05)).

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

**Both, since 2026-10-14.** `recut_locks.py` wrote the additions and the generated block
of each file (below the marker line); the comment block above the marker is
documentation and kept as it is.

## Re-cutting a lock: `recut_locks.py`

    python environments/locks/recut_locks.py --platform osx-arm64            # what would change
    python environments/locks/recut_locks.py --platform osx-arm64 --write
    python environments/locks/recut_locks.py --platform linux-64 --check     # exit 1 unless reproduced

A lock is never re-solved and called the reference. The tool adds: it pins every line of
the conda lock exactly (`name==version=build`) together with every package
`additions.json` names for the platform, solves that once against conda-forge alone at the
lock's floor (macOS 11.0; glibc 2.28 and CUDA 11.8) with sharded repodata and an empty
package cache of its own (a cached package would come back without its URL and MD5), and
refuses,
naming each, a solve in which a locked package moved to another build or dropped out, an
addition is of another build or missing, or anything else appears. It then writes the lock
lines byte for byte in their order, the additions' lines after them (URL and MD5 as the
solve gives them), and the pip lock's pins byte for byte with each added wheel pinned by
the SHA-256 PyPI's JSON API gives for the file `additions.json` names. Run on locks that
already hold the additions, it reproduces them byte for byte; `--check` says so (both
platforms, 2026-10-14, with a package cache that already held libgdsii). `--save-solve
PATH` keeps the solve as evidence.

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
  MEASURED).

autograd requires only `numpy<3`, which both locks satisfy, and takes no part in the field
updates, so `numerics-relevant.json` does not name it and the drift report does not judge
it; `pip check` in the build scripts' step 7 catches a lock that leaves out a
requirement.

## Corpus-only packages (added 2026-10-11)

The MEEP identity corpus (Table 2, `docs/development/meep-identity-corpus.md`) runs in
each host's reference environment as it is. A row whose script imports a package, or
calls a program, that the environment lacks is refused by name, never measured.
`numerics-relevant.json` names what only the corpus needs under `corpus_only`, apart from
every numerics set, since none takes part in MEEP's field updates or the GPU route;
importing and calling shapely, nlopt and the stand-in moved no floating-point control flag
on either host (MEASURED 2026-10-10 and 2026-10-11, in both subnormal modes).

| Package | Corpus rows | How the locks carry it |
|---|---|---|
| shapely 2.1.2 | `ex.mie_scattering`, `ex.differential_cross_section` (the stand-in's `Inverse` module imports it) | a pin of each pip lock: the cp312 macOS arm64 wheel, the cp310 manylinux2014 wheel |
| nlopt 2.11.0 | `ex.waveguide_crossing` | a pin of each pip lock: the cp312 macOS arm64 wheel, the cp310 manylinux_2_28 wheel (glibc 2.28, the lock's floor); requires `numpy>=2,<3` |
| PyMieScatt 1.8.1.1, the stand-in | `ex.mie_scattering`, `ex.differential_cross_section` | built by `parity/meep_gpu/rounds/1.0/build_pymiescatt_standin.py`: the PyPI sdist by SHA-256, two lines changed by `pymiescatt_standin.patch` so `trapz` comes from NumPy (PyMieScatt's last release imports it from `scipy.integrate`, which SciPy 1.14 removed), the five package files checked against their SHA-256 in `numerics-relevant.json`, a wheel built by the environment's own setuptools, installed with `--no-deps` |
| gdspy 1.6.13 (2026-10-14) | `ex.ring_gds` | a line of each conda lock (conda-forge's build; PyPI has no wheel) |
| imageio-ffmpeg 0.6.0 and its `ffmpeg` (2026-10-14) | `test_animation_output` | a pin of each pip lock; the build scripts' step 7b links its executable into `<prefix>/bin`. Until 2026-10-14 the host's ffmpeg was linked there instead (`corpus_only.host_tools`, now retired and refused by name) |

The pins of shapely and nlopt are the wheels the corpus legs of 2026-10-10 measured with.
The stand-in's wheel is not pinned: its metadata names the setuptools that built it, which
the two locks hold at different versions; what is pinned is everything it is built from and
the bytes of the files it installs. `ex.coupler` and `ex.ring_gds` also need MEEP built
with libGDSII, which the reference build is (`numerics-relevant.json`, `meep_links`).

## When to re-cut

The locks are the reference: the environment a round, the tests and the timing run in is
built from them by the build scripts, so a round re-cuts nothing. A lock changes only by
`recut_locks.py`: an addition (a package the inventory finds missing, named in
`additions.json`), which moves no locked package, or a ruled change of a locked package (a
new certified environment), after which the round that certifies it re-runs in an
environment built from the new lock. A lock that conda-forge or PyPI can no longer serve (a
URL that stops resolving) is re-cut the same way, by ruling; never re-solve a lock and call
the result the reference. Edit `numerics-relevant.json` in the same change when the set of
judged packages changes.
