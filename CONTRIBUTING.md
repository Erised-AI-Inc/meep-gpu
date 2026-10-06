# Contributing

Thank you for looking at the code. This page says what belongs in the package, how
to run its tests, and the three rules that are particular to it. Coding agents and
other automated contributors: [AGENTS.md](AGENTS.md) states the same rules as
commands.

## What belongs here

The package accelerates MEEP; it does not replace it. The test for a proposed
feature is one question: **does it execute inside the per-timestep loop, over the
grid?** If it does, it belongs here, because that loop is what a GPU accelerates.
If it runs once at setup, or afterwards on a handful of numbers, it is delegated
to MEEP: geometry and its rasterization, subpixel smoothing, the material library,
mode solving, harmonic inversion and far-field evaluation all stay there.

The package may import MEEP, lazily, inside the entry points that accept an
`mp.Simulation`. Importing `meep_gpu` must keep working on a host with no MEEP, no
CuPy, no Triton and no PyTorch.

## Set up

Install MEEP into the environment first ([INSTALL.md](INSTALL.md)), then the
package from your checkout:

    python -m pip install -e ".[test]"
    python tools/check_install.py

Add `".[nvidia]"` or `".[apple]"` for the device libraries of your host, unless
the environment was made from a file under `environments/`, which already holds
them.

## Run the tests

From the repository root, one suite at a time:

    python -m pytest meep_gpu -q
    python -m pytest parity/meep_gpu -q

Both suites are written to run on a host with no GPU: a test that needs a
device, a device library, MEEP or an archived record skips by a declared
resource. The certification suite (the ledger, weld and record contracts) is
not run by these commands; run it with

    python -m pytest meep_gpu parity/meep_gpu -m certification

[Running the tests](docs/development/testing.md) describes the suites, what a
run of them shows today, and the two failures that are expected while the
fused-product timing records are pending. The examples have their own check:

    python examples/run_examples.py --check

A change is ready for review when the suites show no failure that the pending
certification does not explain, and when the examples' check passes on your host.
The continuous-integration workflow runs the same commands, and the
certification suite, on a host with no GPU and reports every failure; it accepts
only the failures declared in `tools/ci/pending_certification.txt`, all of which
belong to the certification suite.

## Three rules

**1. Editing a pinned file is a certification event.** The kernel ledgers under
`meep_gpu/*_kernels/` record the digest of every source file a certified kernel
depends on, in the package and in the harness under `parity/meep_gpu/`.
Changing one of those files, comments included, makes the ledger-contract tests
fail, and they are meant to. Do not edit a ledger to make a test pass: a ledger
is rewritten only by the harness's tools, from a run on the hardware it names.

To find out before you edit whether a file is pinned, search the ledgers for its
file name (some name a file by path, some by file name alone), and run the three
weld contracts before and after the edit:

    grep -lE '"([^"]*/)?driver\.py"' meep_gpu/*_kernels/*.json
    python -m pytest meep_gpu/test_metal_weld_contract.py meep_gpu/test_triton_weld_contract.py meep_gpu/test_cuda_weld_contract.py -q

They pass on the files of this release. A failure reports how many pinned files
have drifted, as N of D, and names each drifted file with its ledger entry; one
that appears after your edit means the edit touched a pinned file.

A pull request with such an edit leaves the ledgers as they are. It names the
drifted entries and says either that re-certification is needed and was not
run, or which gates you ran, where, and with what verdict, with their artifact
directories attached to the pull request discussion as an archive rather than
committed. A maintainer re-runs or checks the gates on the named hardware and
rebinds the ledger.
[Running the certification harness](docs/development/certification.md) has the
procedure.

**2. A test is never weakened.** Do not loosen a tolerance, delete a test or skip
one to make a change pass. A test that needs something the host does not have
skips by a declared resource (`requires_resource`); every other skip is turned
into a failure by the root `conftest.py`.

**3. A number carries its denominator.** A count is N of D with D named. A rate or
a ratio names the hardware, the precision, the MEEP version and, for MEEP, the
rank count, in the same sentence. A figure that is pending a re-measurement is
written as provisional.

## Numerical changes

The engine follows MEEP's C++ implementation, and the modules cite the MEEP source
files they follow. A change to the arithmetic cites what it follows, and comes
with a test written first: a known value, a degenerate case, a scaling check and
one realistic simulation compared with MEEP. Kernel results must stay
bit-identical to the array path; agreement with MEEP is measured, and reported as
measured.

## Reporting a problem

Open an issue at <https://github.com/Erised-AI-Inc/meep-gpu/issues>. Include the
output of:

    python tools/check_install.py --verbose
    python -c "import meep as mp; print(mp.__version__, mp.is_single_precision())"

and, for a simulation that is refused or slow, `gpu_compatibility(sim).reasons`
and `driver.fast_path_report()`. A script that builds the simulation is the most
useful thing a report can carry.

This project is independent of MEEP. Please do not report a problem with this
package to the MEEP developers; a problem that reproduces with `sim.run(...)`
alone, without this package, is theirs.

## Style

Code, comments and documentation are neutral and technical. Names are renamed
outright, with no alias left behind. A test, gate or benchmark that can run for
more than about a minute prints a flushed line of progress per unit of work.

## Names

The import name `meep_gpu` is fixed. The distribution and repository name is
changed everywhere by one command, never by hand:

    python tools/rename_distribution.py --name NEW-NAME --apply

## Licence of contributions

The package is licensed GPL-2.0-or-later, as MEEP is. By contributing you agree
that your contribution is licensed under the same terms.
