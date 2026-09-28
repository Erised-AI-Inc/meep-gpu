#!/usr/bin/env python
"""Judge a test run: every failure must be one that is declared as pending.

    python tools/ci/check_failures.py REPORT.xml [REPORT.xml ...]

Nothing is deselected and nothing is skipped to make a run pass. The suites run in
full and report every failure. This script reads their JUnit reports and the list
``tools/ci/pending_certification.txt`` and exits 0 only when every failure and
every error is on that list.

The list holds the tests that fail because the certification ledgers were cut
before the released files were finalized. It is emptied when the certification
round has run on the released files. A listed test that now passes is reported, so
the list can only shrink.

A report that is missing, unreadable or empty is a failure: a run that did not
finish proves nothing.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

PENDING = Path(__file__).resolve().parent / "pending_certification.txt"


def identifier(case: ElementTree.Element) -> str:
    return f"{case.get('classname', '')}::{case.get('name', '')}"


def read_pending() -> set[str]:
    if not PENDING.exists():
        return set()
    lines = PENDING.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines
            if line.strip() and not line.lstrip().startswith("#")}


def main(arguments: list[str]) -> int:
    if not arguments:
        print(__doc__)
        return 2
    pending = read_pending()
    ran = failed = skipped = 0
    failures: list[str] = []
    passed: set[str] = set()
    for name in arguments:
        path = Path(name)
        try:
            root = ElementTree.parse(path).getroot()
        except (OSError, ElementTree.ParseError) as problem:
            print(f"UNREADABLE REPORT {path}: {problem}")
            return 1
        cases = list(root.iter("testcase"))
        if not cases:
            print(f"EMPTY REPORT {path}: no test ran")
            return 1
        for case in cases:
            if case.find("skipped") is not None:
                skipped += 1
                continue
            ran += 1
            if case.find("failure") is not None or case.find("error") is not None:
                failed += 1
                failures.append(identifier(case))
            else:
                passed.add(identifier(case))

    unexpected = sorted(set(failures) - pending)
    declared = sorted(set(failures) & pending)
    now_passing = sorted(pending & passed)
    print(f"{failed} failed of {ran} tests run; {skipped} skipped by a declared "
          f"resource")
    print(f"{len(declared)} of the {failed} failures are declared in "
          f"{PENDING.name} ({len(pending)} entries)")
    for entry in declared:
        print(f"  pending certification: {entry}")
    for entry in now_passing:
        print(f"  listed as pending and now passing; remove it from the list: "
              f"{entry}")
    for entry in unexpected:
        print(f"  NOT DECLARED: {entry}")
    if unexpected:
        print(f"FAILED: {len(unexpected)} failures are not explained by the pending "
              f"certification")
        return 1
    print("PASSED: no failure outside the declared list")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
