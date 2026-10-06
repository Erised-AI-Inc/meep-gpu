#!/usr/bin/env python
"""``bench_fused_products.py`` on a timing case that is not one of the route gates' rows.

WHY A WRAPPER. ``bench_fused_products.py`` is the timing harness of record and its bytes
are pinned; it times a case only when the case is in its table module's ``CASES`` and a
``dispatch`` row of the drive table (``selected_cases``). The cases of
``timing_cases.py`` are neither, by design: they are defined once, in a file no ledger
pins, and read by the MEEP bench and this wrapper alike. So this wrapper, for the
length of one ``bench_fused_products.main(argv)`` call and no longer:

* puts the case's builder into ``table_module(<table>).CASES[<case>]`` (the Metal
  table's ``CASES`` is its own dict, so the table's module is asked, not the route
  gate's);
* puts a DEEP COPY of the case's template DRIVE row into ``drive_rows(<table>)[<case>]``
  (``timing_cases.drive_spec``: ``pml_3d`` for the magnetic pair alone, or
  ``pml_3d_diagonal`` for both pairs, with any per-table override);
* stamps every row the bench writes for the case with a ``timing_case`` field (the
  case's declaration, its builder's source digest and this module's digest), by
  wrapping the bench module's ``append_jsonl`` -- the one call every row passes
  through. ``provenance.sha256`` is NOT touched: ``cut_timing_record.py`` copies every
  key of it into the timing records, and the keys a row pins are the bench's;

and removes all three in ``finally``, on a normal return and on an exception alike.

REFUSED BY NAME: a built-in case (the GPU row of a built-in case IS
``bench_fused_products.py``, unwrapped, so its command stays the 2026-09-28 command),
and a name that already exists in the table's ``CASES`` or drive rows.

Everything after the wrapper's own options is passed to ``bench_fused_products.main``
unchanged; ``--case`` must name exactly the injected cases.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import timing_cases  # noqa: E402


def _wrapper_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], add_help=False)
    parser.add_argument("--harness-root", default=None, dest="harness_root")
    parser.add_argument("--drive-table", default="triton", dest="drive_table")
    parser.add_argument("--case", action="append", default=[])
    known, _rest = parser.parse_known_args(list(argv))
    return known


def _passthrough(argv: Sequence[str]) -> List[str]:
    """``argv`` without the wrapper's own ``--harness-root``."""
    out: List[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == "--harness-root":
            skip = True
            continue
        if token.startswith("--harness-root="):
            continue
        out.append(token)
    return out


class Injected:
    """The injection, as a context manager: in on enter, out on exit, whatever happens."""

    def __init__(self, bench: Any, table: str, cases: Sequence[str]) -> None:
        self.bench = bench
        self.table = table
        self.cases = list(cases)
        self.module = bench.table_module(table)
        self.rows = bench.drive_rows(table)
        self.original_append = bench.append_jsonl
        self.identities: Dict[str, Dict[str, Any]] = {}

    def __enter__(self) -> "Injected":
        for name in self.cases:
            if timing_cases.is_builtin(name):
                raise SystemExit(f"REFUSING: {name!r} is a built-in case; time it with "
                                 "bench_fused_products.py directly")
            if name in self.module.CASES or name in self.rows:
                raise SystemExit(f"REFUSING: {name!r} already exists in the {self.table} "
                                 "table's CASES or drive rows; an injected case may not "
                                 "replace one")
        added: List[str] = []
        try:
            for name in self.cases:
                builder, _facts = timing_cases.resolve(name, os.path.dirname(
                    os.path.abspath(self.bench.__file__)))
                spec = timing_cases.drive_spec(self.table, name, self.rows)
                self.module.CASES[name] = builder
                self.rows[name] = spec
                added.append(name)
                self.identities[name] = dict(timing_cases.case_identity(name),
                                             table=self.table,
                                             arms=spec.get("arms"))
        except BaseException:
            self._remove(added)
            raise
        identities = self.identities
        original = self.original_append

        @functools.wraps(original)
        def stamped(path: str, row: Dict[str, Any]) -> Any:
            if isinstance(row, dict) and row.get("case") in identities:
                row["timing_case"] = identities[row["case"]]
            return original(path, row)

        self.bench.append_jsonl = stamped
        return self

    def _remove(self, names: Sequence[str]) -> None:
        for name in names:
            self.module.CASES.pop(name, None)
            self.rows.pop(name, None)

    def __exit__(self, *_exc: Any) -> None:
        self.bench.append_jsonl = self.original_append
        self._remove(self.cases)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known = _wrapper_arguments(argv)
    if not known.case:
        raise SystemExit("REFUSING: name the timing case with --case")
    harness_root = os.path.abspath(known.harness_root
                                   or timing_cases.find_harness_root(HERE) or HERE)
    if not os.path.isfile(os.path.join(harness_root, "bench_fused_products.py")):
        raise SystemExit(f"REFUSING: no bench_fused_products.py under {harness_root}")
    timing_cases.put_on_path(harness_root)
    import bench_fused_products as bench  # noqa: PLC0415

    if known.drive_table not in bench.DRIVE_TABLES:
        raise SystemExit(f"REFUSING: --drive-table {known.drive_table!r}")
    with Injected(bench, known.drive_table, known.case):
        bench.say(f"TIMING CASE {', '.join(known.case)} injected into the "
                  f"{known.drive_table} table (timing_cases.py); spec from "
                  + ", ".join(f"{n}<-{timing_cases.declared(n).template}"
                              for n in known.case))
        return int(bench.main(_passthrough(argv)) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
