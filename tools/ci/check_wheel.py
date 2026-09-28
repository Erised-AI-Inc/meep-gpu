#!/usr/bin/env python
"""Build the wheel and check what it carries.

    python tools/ci/check_wheel.py [--no-build-isolation]

The kernel tables open their certification ledgers beside the modules when a run
is planned, so a wheel without them would leave every table unreadable. This
builds the wheel from the checkout and checks:

    * every JSON ledger of the three kernel directories is inside it;
    * the licence text and the notice are inside it;
    * nothing of the certification harness, the examples, the tools or the manual
      is inside it.

Exit status 0 when every check holds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "meep_gpu"
KERNEL_DIRECTORIES = ("cuda_kernels", "triton_kernels", "metal_kernels")
NOT_INSTALLED = ("parity/", "examples/", "tools/", "docs/", "benchmarks/",
                 f"{PACKAGE}/manual/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-build-isolation", action="store_true",
                        help="build with the setuptools already installed")
    arguments = parser.parse_args()

    ledgers = sorted(path.relative_to(ROOT).as_posix()
                     for directory in KERNEL_DIRECTORIES
                     for path in (ROOT / PACKAGE / directory).glob("*.json"))
    if not ledgers:
        print(f"no ledger found under {PACKAGE}/: is this the repository root?")
        return 1

    with tempfile.TemporaryDirectory() as scratch:
        command = [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
                   "--wheel-dir", scratch, "--quiet"]
        if arguments.no_build_isolation:
            command.append("--no-build-isolation")
        subprocess.run(command, check=True)
        wheels = sorted(Path(scratch).glob("*.whl"))
        if len(wheels) != 1:
            print(f"expected 1 wheel, found {len(wheels)}")
            return 1
        with zipfile.ZipFile(wheels[0]) as archive:
            names = archive.namelist()
        print(f"wheel: {wheels[0].name}, {len(names)} files")

    inside = set(names)
    problems = []
    present = [ledger for ledger in ledgers if ledger in inside]
    print(f"ledgers in the wheel: {len(present)} of {len(ledgers)}")
    for ledger in ledgers:
        if ledger not in inside:
            problems.append(f"ledger missing from the wheel: {ledger}")
    licence = [name for name in names
               if name.endswith(("/LICENSE", "/NOTICE")) and ".dist-info/" in name]
    print(f"licence files in the wheel's metadata: {len(licence)} of 2")
    if len(licence) != 2:
        problems.append("LICENSE and NOTICE are not both in the wheel's metadata")
    strays = [name for name in names if name.startswith(NOT_INSTALLED)]
    print(f"files of the harness, examples, tools or manual in the wheel: "
          f"{len(strays)}")
    problems += [f"must not be installed: {name}" for name in strays[:20]]
    top_level = sorted({name.split("/", 1)[0] for name in names})
    print(f"top-level entries: {', '.join(top_level)}")

    for problem in problems:
        print(f"CHECK FAILED: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
