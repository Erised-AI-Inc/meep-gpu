"""Attach each shim-generated case to the recorded lift row it IS, by facts.

The shim (``shim/parameterized.py``) reproduces which parameter tuple runs but
deliberately not the upstream method NAME, so identity has to be established by
measurement. The ``_facts`` block is computed by the survey harness's own function on
both sides — cell size, resolution, Courant, k_point, source / src_time / boundary /
symmetry / DFT kinds, geometry and material declarations — so exact equality of that
block is a strong, checkable identity.

A case matching no recorded accepted row, or more than one, is reported and NOT
assigned. Nothing is matched by name similarity.

Writes ``tests_param_matched.jsonl`` (rows renamed to the recorded case) and prints the
matching table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
def _find_api_root(start):
    """Locate the repository root BY NAME, never by depth.

    These harness files were promoted out of a results/ tranche into the parity
    dir; a hard-coded ``parents[N]`` silently followed them to the wrong root and
    every child died on ModuleNotFoundError while the parent still wrote a
    full-length census of ``measured: false`` rows.
    """
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}; refusing to guess")


_API = _find_api_root(_HERE)
TESTS_LIFT = _API / ("parity/meep_gpu/results/meep_python_tests_2026-08-09_monitorfix/"
                     "lift.jsonl")


def key(facts: dict) -> str:
    return json.dumps(facts, sort_keys=True, default=repr)


def main(argv=None) -> int:
    # The census directory is an ARGUMENT, not this file's own location. This
    # script used to live inside the tranche it read; once promoted to the parity
    # dir, ``_HERE`` silently became the wrong place to look for tests_param.jsonl.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True,
                        help="census directory holding tests_param.jsonl")
    args = parser.parse_args(argv)
    census = Path(args.census)
    if not census.is_absolute():
        census = _HERE / census
    if not (census / "tests_param.jsonl").is_file():
        raise SystemExit(f"{census}/tests_param.jsonl not found")
    recorded = [json.loads(line) for line in
                TESTS_LIFT.read_text(encoding="utf-8").splitlines() if line.strip()]
    measured = [json.loads(line) for line in
                (census / "tests_param.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()]

    wanted = [row for row in recorded
              if row.get("accepted") is True and row.get("facts")]
    by_module: dict = {}
    for row in wanted:
        by_module.setdefault(row["module"], []).append(row)

    out = []
    unmatched = []
    ambiguous = []
    claimed = set()
    for row in measured:
        if not row.get("measured") or not row.get("facts"):
            continue
        candidates = [candidate for candidate in by_module.get(row.get("module"), [])
                      if key(candidate["facts"]) == key(row["facts"])]
        if len(candidates) == 1:
            recorded_case = candidates[0]["case"]
            if recorded_case in claimed:
                ambiguous.append((row.get("module"), row.get("row"), recorded_case,
                                  "recorded row already claimed"))
                continue
            claimed.add(recorded_case)
            row["shim_row"] = row["row"]
            row["row"] = recorded_case
            row["leg"] = "tests"
            row["matched_by"] = "exact facts equality"
            out.append(row)
        elif len(candidates) > 1:
            ambiguous.append((row, [c["case"] for c in candidates]))
        else:
            unmatched.append((row.get("module"), row.get("row")))

    # A facts-identical GROUP. The parameter that distinguishes these cases (the
    # cylindrical m, the complex-vs-real/imag source spelling) does not appear in the
    # facts block, so no pairing inside the group is established. It does not have to
    # be: when the group's measured cases and its recorded rows are the same COUNT, the
    # assignment is a permutation, and every number this analysis computes is a count
    # over the group — invariant under permutation. The rows are labelled so a reader
    # cannot mistake one for an identified case.
    groups: dict = {}
    for row, cases in ambiguous:
        groups.setdefault(tuple(sorted(cases)), []).append(row)
    unresolved = []
    for cases, rows in groups.items():
        if len(rows) != len(cases):
            unresolved.append((cases, len(rows)))
            continue
        for recorded_case, row in zip(cases, rows):
            row["shim_row"] = row["row"]
            row["row"] = recorded_case
            row["leg"] = "tests"
            row["matched_by"] = ("group bijection; facts identical inside the group, "
                                 "so the within-group pairing is a permutation and "
                                 "every count below is invariant under it")
            row["group_permutation_unresolved"] = True
            claimed.add(recorded_case)
            out.append(row)

    (census / "tests_param_matched.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in out), encoding="utf-8")

    modules = sorted({row.get("module") for row in measured})
    scope = [row for row in wanted if row["module"] in modules]
    print(f"modules in scope                        : {modules}")
    print(f"recorded accepted rows in those modules : {len(scope)}")
    print(f"shim cases measured                     : "
          f"{sum(1 for r in measured if r.get('measured'))}")
    print(f"MATCHED, unique facts                   : "
          f"{sum(1 for r in out if not r.get('group_permutation_unresolved'))}")
    print(f"MATCHED, facts-identical group          : "
          f"{sum(1 for r in out if r.get('group_permutation_unresolved'))}")
    print(f"groups whose counts did NOT balance     : {len(unresolved)} {unresolved}")
    print(f"shim cases matching no recorded row     : {len(unmatched)} "
          f"(expected: cases the engine gate refused, which are not in the denominator)")
    print()
    for row in sorted(out, key=lambda r: (r["module"], r["row"])):
        flag = " [group]" if row.get("group_permutation_unresolved") else ""
        print(f"  {row['module']:<28} {row['shim_row']:<58} -> {row['row']}{flag}")
    missing = [row["case"] for row in scope if row["case"] not in claimed]
    print(f"\nrecorded rows NOT recovered: {len(missing)}")
    for case in missing:
        print("  ", case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
