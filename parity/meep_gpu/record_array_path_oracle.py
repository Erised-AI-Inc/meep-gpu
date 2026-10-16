"""Mint, or verify, the array-path oracle leaf of the three kernel ledgers.

    python record_array_path_oracle.py                 # report what would change
    python record_array_path_oracle.py --write         # write the leaves
    python record_array_path_oracle.py --verify        # recompute; exit 1 if any ledger differs
    python record_array_path_oracle.py --report <path> # also write the run as JSON

WHAT IT WRITES. The ``array_path_oracle`` leaf (``meep_gpu.array_path_oracle``) of
``meep_gpu/cuda_kernels/fingerprints.json``, ``meep_gpu/triton_kernels/fingerprints.json``
and ``meep_gpu/metal_kernels/fingerprints.json``:

* in every entry that binds source bytes (any of ``fastpath.BOUND_FIELDS``), the fixture
  set and ``fingerprint_sha256``: the array-path behaviour the entry's verdict was
  certified against;
* in each ``driver_dispatch`` record, the same two values plus the per-case digests, the
  form-class counts, the deferred classes and the module code digests (reported, not
  judged).

The weld contracts compare the ledgers to a fresh run of the fixture
(``test_<table>_weld_contract.py::test_the_<table>_ledger_records_the_live_array_path_oracle``).
Writing is a certification event: it says every weld of the three tables was certified
against the array path as it steps now. Mint after a certification round on these bytes,
never to turn that test green.

WHAT IT REQUIRES BEFORE IT WRITES.

* The fixture runs in this process and reaches every declared form class
  (``array_path_oracle.coverage_problems`` is empty);
* the array-path modules and the oracle module hash the same before and after the run,
  so the fingerprint written is the fingerprint of the bytes that ran;
* each ledger round-trips byte for byte through ``json.dumps(indent=2, sort_keys=True)``
  plus a newline, so writing it changes only the leaves;
* after the change, in memory, every changed path lies under an ``array_path_oracle``
  leaf, and every entry's ``fastpath.bound_digest`` and ``fastpath.live_capabilities``
  are unchanged: no per-capability record is retired by this write.

``--verify`` recomputes the run, prints every case digest, and compares each ledger's
leaves with it. It writes nothing and exits 0 only when all three ledgers record the run
this host produces, which is the host-independence check: the same tree on another host
must verify against the leaves minted here. One flushed line is printed per case.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent


def _find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu import array_path_oracle as oracle  # noqa: E402
from meep_gpu import fastpath  # noqa: E402

#: The three ledgers, relative to the repository root, by table.
LEDGERS: Dict[str, Path] = {
    "cuda": Path("meep_gpu") / "cuda_kernels" / "fingerprints.json",
    "triton": Path("meep_gpu") / "triton_kernels" / "fingerprints.json",
    "metal": Path("meep_gpu") / "metal_kernels" / "fingerprints.json",
}


class Refusal(Exception):
    """A named reason not to write. Nothing has been written when it is raised."""


def canonical(ledger: Mapping[str, Any]) -> str:
    """The binders' serialisation of a ledger."""
    return json.dumps(ledger, indent=2, sort_keys=True) + "\n"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leaves(value: Any, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    """Every scalar of a JSON document, keyed by its path."""
    if isinstance(value, Mapping):
        out: Dict[Tuple[str, ...], Any] = {}
        if not value:
            out[prefix] = {}
        for key, item in value.items():
            out.update(leaves(item, prefix + (str(key),)))
        return out
    if isinstance(value, list):
        out = {}
        if not value:
            out[prefix] = []
        for index, item in enumerate(value):
            out.update(leaves(item, prefix + (f"[{index}]",)))
        return out
    return {prefix: value}


def changed_leaves(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[Tuple[str, ...]]:
    old, new = leaves(before), leaves(after)
    return sorted(path for path in set(old) | set(new) if old.get(path, ...) != new.get(path, ...))


def entry_bindings(ledger: Mapping[str, Any]) -> Dict[str, Tuple[Optional[str], Tuple[str, ...]]]:
    """Per top-level entry: its bound digest (None if it binds nothing) and live capabilities."""
    out: Dict[str, Tuple[Optional[str], Tuple[str, ...]]] = {}
    for key, entry in ledger.items():
        if not isinstance(entry, Mapping):
            continue
        try:
            bound: Optional[str] = fastpath.bound_digest(entry)
        except fastpath.CapabilityRecordError:
            bound = None
        out[key] = (bound, fastpath.live_capabilities(entry))
    return out


def load_ledger(path: Path) -> Dict[str, Any]:
    """The ledger document; refused unless its text is the canonical form."""
    text = path.read_text(encoding="utf-8")
    ledger = json.loads(text)
    if canonical(ledger) != text:
        raise Refusal(f"{path} does not round-trip through json.dumps(indent=2, "
                      "sort_keys=True); writing it would reformat more than the leaves")
    return ledger


def stamped(ledger: Mapping[str, Any], result: oracle.OracleRun,
            root: Path) -> Dict[str, Any]:
    """``ledger`` with every leaf set from ``result``; the input is not modified."""
    changed = copy.deepcopy(dict(ledger))
    for key, entry in changed.items():
        if key == oracle.DISPATCH_KEY or not oracle.binds_bytes(entry):
            continue
        entry[oracle.LEAF_KEY] = oracle.entry_leaf(result)
    dispatch = changed.get(oracle.DISPATCH_KEY)
    if not isinstance(dispatch, dict):
        raise Refusal(f"the ledger has no {oracle.DISPATCH_KEY} record to carry the run")
    dispatch[oracle.LEAF_KEY] = oracle.dispatch_leaf(result, root)
    return changed


def watched_files(root: Path) -> Tuple[Path, ...]:
    """The modules the fingerprint describes, and the module that computes it."""
    names = tuple(oracle.PROVENANCE_MODULES) + (oracle.ORACLE_MODULE,)
    return tuple(root / name for name in names if (root / name).is_file())


def ledger_paths(root: Path, ledgers: Optional[Mapping[str, Path]] = None) -> Dict[str, Path]:
    """The three ledgers under ``root``, or the paths ``ledgers`` names instead."""
    if ledgers is not None:
        if set(ledgers) != set(LEDGERS):
            raise Refusal(f"ledgers must name exactly {sorted(LEDGERS)}, got {sorted(ledgers)}")
        return {table: Path(ledgers[table]) for table in LEDGERS}
    return {table: root / relative for table, relative in LEDGERS.items()}


def record(root: Path = _API, *, write: bool = False,
           ledgers: Optional[Mapping[str, Path]] = None,
           progress=None) -> Dict[str, Any]:
    """Run the fixture, check it, and (with ``write``) move the leaves. Returns a report.

    ``root`` is the tree whose modules the provenance reports; ``ledgers`` names other
    copies of the three ledgers to write instead of the tree's own (the tool's tests
    write into a temporary directory).
    """
    paths = ledger_paths(root, ledgers)
    ledgers = {table: load_ledger(path) for table, path in paths.items()}
    watched = watched_files(root)
    before = {path: sha256_of(path) for path in watched}
    result = oracle.run(progress=progress)
    moved = sorted(path.name for path in watched if sha256_of(path) != before[path])
    if moved:
        raise Refusal(f"{moved} changed while the fixture ran; the fingerprint would name "
                      "bytes that did not run")
    coverage = oracle.coverage_problems(result.form_classes)
    if coverage:
        raise Refusal("the fixture does not reach every declared form class: "
                      + "; ".join(coverage))
    report: Dict[str, Any] = {
        "fixture_set": result.fixture_set,
        "fingerprint_sha256": result.fingerprint_sha256,
        "case_sha256": result.case_sha256,
        "form_classes_declared": len(oracle.DECLARED_FORM_CLASSES),
        "form_classes_counted": len(result.form_classes),
        "host": result.host,
        "elapsed_seconds": result.elapsed_seconds,
        "watched": {path.relative_to(root).as_posix(): digest
                    for path, digest in before.items()},
        "tables": {},
        "written": False,
    }
    pending: Dict[str, Dict[str, Any]] = {}
    for table, ledger in ledgers.items():
        changed = stamped(ledger, result, root)
        diff = changed_leaves(ledger, changed)
        foreign = [path for path in diff
                   if len(path) < 2 or path[1] != oracle.LEAF_KEY]
        if foreign:
            raise Refusal(f"{table}: the change would move {foreign[:6]}, which lie outside "
                          f"the {oracle.LEAF_KEY} leaves")
        if entry_bindings(changed) != entry_bindings(ledger):
            raise Refusal(f"{table}: the change would move a bound digest or a live capability")
        binding = sum(1 for key, entry in ledger.items()
                      if key != oracle.DISPATCH_KEY and oracle.binds_bytes(entry))
        report["tables"][table] = {
            "ledger": str(paths[table]),
            "byte_binding_entries": binding,
            "changed_leaves": len(diff),
            "problems_before": oracle.ledger_problems(ledger, result, table),
            "problems_after": oracle.ledger_problems(changed, result, table),
        }
        if report["tables"][table]["problems_after"]:
            raise Refusal(f"{table}: the stamped ledger still fails the contract: "
                          + "; ".join(report["tables"][table]["problems_after"]))
        if diff:
            pending[table] = changed
    report["status"] = "stale" if pending else "current"
    if write and pending:
        for table, changed in pending.items():
            paths[table].write_text(canonical(changed), encoding="utf-8")
            if paths[table].read_text(encoding="utf-8") != canonical(changed):
                raise Refusal(f"{paths[table]} does not read back as written")
        report["written"] = True
    return report


def verify(root: Path = _API, *, ledgers: Optional[Mapping[str, Path]] = None,
           progress=None) -> Dict[str, Any]:
    """Recompute the run and compare every ledger with it. Writes nothing."""
    paths = ledger_paths(root, ledgers)
    result = oracle.run(progress=progress)
    report: Dict[str, Any] = {
        "fixture_set": result.fixture_set,
        "fingerprint_sha256": result.fingerprint_sha256,
        "case_sha256": result.case_sha256,
        "coverage_problems": oracle.coverage_problems(result.form_classes),
        "host": result.host,
        "elapsed_seconds": result.elapsed_seconds,
        "tables": {},
    }
    for table, path in paths.items():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        report["tables"][table] = oracle.ledger_problems(ledger, result, table)
    report["verified"] = (not report["coverage_problems"]
                          and not any(report["tables"].values()))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true",
                      help="write the leaves (default: report only)")
    mode.add_argument("--verify", action="store_true",
                      help="recompute and compare with the ledgers; write nothing")
    parser.add_argument("--report", type=Path,
                        help="also write the run and the verdict as JSON to this path")
    args = parser.parse_args(argv)

    def progress(line: str) -> None:
        print(line, flush=True)

    try:
        report = (verify(progress=progress) if args.verify
                  else record(write=args.write, progress=progress))
    except (Refusal, oracle.OracleRefusal) as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 2
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    print(f"fixture set {report['fixture_set']}: fingerprint {report['fingerprint_sha256']} "
          f"({len(report['case_sha256'])} cases, {report['elapsed_seconds']} s, "
          f"numpy {report['host']['numpy']}, {report['host']['machine']})", flush=True)
    for name, digest in report["case_sha256"].items():
        print(f"  {name}: {digest}", flush=True)
    if args.verify:
        for table, problems in report["tables"].items():
            print(f"{table}: " + ("records this run" if not problems
                                  else "; ".join(problems)), flush=True)
        if report["coverage_problems"]:
            print("coverage: " + "; ".join(report["coverage_problems"]), flush=True)
        print("VERIFIED" if report["verified"] else "NOT VERIFIED", flush=True)
        return 0 if report["verified"] else 1
    for table, facts in report["tables"].items():
        print(f"{table}: {facts['changed_leaves']} leaves change over "
              f"{facts['byte_binding_entries']} byte-binding entries and driver_dispatch; "
              f"before: {len(facts['problems_before'])} problem(s)", flush=True)
    if report["status"] == "current":
        print("every ledger already records this run; nothing to write", flush=True)
        return 0
    print("every changed path lies under an array_path_oracle leaf; every entry's bound "
          "digest and live capabilities are unchanged", flush=True)
    print("written" if report["written"] else "report only; pass --write to write",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
