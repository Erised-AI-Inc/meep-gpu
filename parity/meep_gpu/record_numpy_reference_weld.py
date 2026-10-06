"""Re-record the NumPy reference weld of the Triton D/E product from a fresh run.

    python record_numpy_reference_weld.py
    python record_numpy_reference_weld.py --write
    python record_numpy_reference_weld.py --validator-out <dir>/pml_reference_vs_stepping.json

WHAT IT WRITES. One leaf of ``meep_gpu/triton_kernels/fingerprints.json``:
``fused_electric_gate.numpy_reference_weld.probe_sha256``, the sha256 of the NumPy
validator that the D/E record names under ``numpy_reference_weld.probe``
(``validate_pml_reference_vs_stepping.py``). The validator checks the bit-identity
probe's PML reference against ``stepping.py`` itself, on NumPy, so it needs no GPU.
``test_triton_kernels.py::test_the_fused_electric_record_is_welded_to_both_oracles_and_the_route``
holds the leaf against the file in the tree.

WHY A TOOL. No rebind or recut tool writes this leaf: it is nested inside an
uncited entry, and no device gate produces it. Editing the validator (comments
included) leaves the record naming bytes that no longer ship, and rule 1 of the
repository forbids moving the digest by hand.

WHAT IT REQUIRES BEFORE IT WRITES. It runs the validator itself, in a child
process of this interpreter with the inherited environment, and writes the digest
only when all of these hold:

* the child exits 0 and its artifact states ``check == "pml_reference_vs_stepping"``,
  ``backend == "numpy"`` and ``pass`` true;
* the artifact's counts are the record's own claims: ``valid_cases_bit_identical``
  ``"N/N"`` gives ``identical == ran == N``, and ``structurally_invalid_skipped`` gives
  ``skipped``; every case is accounted for (``ran + skipped`` cases). A run that
  measured a different number of cases is refused, not transcribed: this tool moves
  one digest and never a claim;
* the validator and every script it loads by path from its own directory (today
  ``probe_fused_kernel_bit_identity.py`` and ``gate_triton_fused_electric.py``, read
  off the validator's source) hash the same before and after the run, so the
  digest written is the digest of the bytes that ran;
* the ledger round-trips byte for byte through the binders' serialisation
  (``json.dumps(indent=2, sort_keys=True)`` plus a newline), so writing it changes
  only the leaf;
* after the change, in memory, exactly that one leaf differs, and every entry's
  :func:`fastpath.bound_digest` and :func:`fastpath.live_capabilities` are
  unchanged. The leaf is not one of :data:`fastpath.BOUND_FIELDS` (those are
  top-level fields), so no per-capability record is retired by this write.

It reports by default and writes only with ``--write``. The validator's artifact
goes to ``--validator-out`` when given, and to a temporary directory otherwise.
The validator prints one flushed line per case; they are passed through as they
arrive.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
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

from meep_gpu import fastpath  # noqa: E402

#: The ledger, relative to the repository root.
LEDGER_RELATIVE = Path("meep_gpu") / "triton_kernels" / "fingerprints.json"

#: The entry and the nested block this tool writes into.
GATE_KEY = "fused_electric_gate"
WELD_KEY = "numpy_reference_weld"

#: The block's fields. The tool writes ``probe_sha256`` and reads the other three.
WELD_FIELDS = ("probe", "probe_sha256", "structurally_invalid_skipped",
               "valid_cases_bit_identical")

#: Where the validator must live, relative to the repository root.
HARNESS_DIR = Path("parity") / "meep_gpu"

#: How the validator loads a sibling script by path (``os.path.join(HERE, "x.py")``).
_LOADED_BY_PATH = re.compile(r"""os\.path\.join\(\s*HERE\s*,\s*["']([A-Za-z0-9_]+\.py)["']\s*\)""")

#: The artifact's identity, as the validator spells it.
EXPECTED_CHECK = "pml_reference_vs_stepping"
EXPECTED_BACKEND = "numpy"


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


def load_ledger(ledger_path: Path) -> Tuple[str, Dict[str, Any]]:
    """The ledger's text and document; refused unless the text is the canonical form."""
    text = ledger_path.read_text(encoding="utf-8")
    ledger = json.loads(text)
    if canonical(ledger) != text:
        raise Refusal(f"{ledger_path} does not round-trip through json.dumps(indent=2, "
                      "sort_keys=True); writing it would reformat more than the one leaf")
    return text, ledger


def claimed_counts(weld: Mapping[str, Any]) -> Tuple[int, int]:
    """``(ran, skipped)`` as the record claims them; refused unless every ran case is identical."""
    claim = weld["valid_cases_bit_identical"]
    match = re.fullmatch(r"(\d+)/(\d+)", str(claim))
    if not match:
        raise Refusal(f"valid_cases_bit_identical is {claim!r}, not 'N/N'")
    identical, ran = int(match.group(1)), int(match.group(2))
    if identical != ran or ran == 0:
        raise Refusal(f"the record claims {claim!r}; this tool writes a digest only for a "
                      "record that claims every case it ran identical")
    skipped = weld["structurally_invalid_skipped"]
    if not isinstance(skipped, int) or isinstance(skipped, bool) or skipped < 0:
        raise Refusal(f"structurally_invalid_skipped is {skipped!r}, not a count")
    return ran, skipped


def validator_path(api_root: Path, weld: Mapping[str, Any]) -> Path:
    """The validator the record names, as the record test resolves it; inside the harness."""
    named = weld["probe"]
    path = (api_root / named).resolve()
    harness = (api_root / HARNESS_DIR).resolve()
    if path.parent != harness:
        raise Refusal(f"numpy_reference_weld.probe is {named!r}, which is not a script of "
                      f"{HARNESS_DIR.as_posix()}/")
    if not path.is_file():
        raise Refusal(f"numpy_reference_weld.probe names {named!r}, which is not in the tree")
    return path


def loaded_by_path(validator: Path) -> Tuple[Path, ...]:
    """The sibling scripts the validator loads by path, read off its source."""
    names = sorted(set(_LOADED_BY_PATH.findall(validator.read_text(encoding="utf-8"))))
    paths = tuple(validator.parent / name for name in names)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise Refusal(f"{validator.name} loads {missing} by path, and they are not in the tree")
    return paths


def run_validator(validator: Path, out: Path, python: str) -> Tuple[int, float]:
    """Run the validator to ``out``, passing its lines through as they arrive."""
    started = time.time()
    command = [python, "-u", str(validator), "--out", str(out)]
    print(f"[run] {' '.join(command)}", flush=True)
    with subprocess.Popen(command, cwd=str(validator.parent), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
        assert child.stdout is not None
        for line in child.stdout:
            print(f"  {line.rstrip()}", flush=True)
        returncode = child.wait()
    return returncode, time.time() - started


def check_artifact(artifact: Mapping[str, Any], ran: int, skipped: int) -> Dict[str, Any]:
    """The artifact's summary, refused unless it states exactly the record's claims."""
    reasons = []
    if artifact.get("check") != EXPECTED_CHECK:
        reasons.append(f"check is {artifact.get('check')!r}, not {EXPECTED_CHECK!r}")
    if artifact.get("backend") != EXPECTED_BACKEND:
        reasons.append(f"backend is {artifact.get('backend')!r}, not {EXPECTED_BACKEND!r}")
    summary = artifact.get("summary")
    if not isinstance(summary, Mapping):
        raise Refusal("the validator's artifact has no summary: the run did not finish")
    expected = {"ran": ran, "identical": ran, "skipped": skipped}
    for field, value in expected.items():
        if summary.get(field) != value:
            reasons.append(f"summary.{field} is {summary.get(field)!r}; the record claims {value}")
    if summary.get("pass") is not True:
        reasons.append(f"summary.pass is {summary.get('pass')!r}")
    cases = artifact.get("cases")
    if not isinstance(cases, list) or len(cases) != ran + skipped:
        count = len(cases) if isinstance(cases, list) else None
        reasons.append(f"the artifact holds {count} cases, not ran + skipped = {ran + skipped}")
    if reasons:
        raise Refusal("the validator's run does not state the record's claims: "
                      + "; ".join(reasons))
    return dict(summary)


def record(api_root: Path = _API, *, write: bool = False, validator_out: Optional[Path] = None,
           python: str = sys.executable) -> Dict[str, Any]:
    """Run the validator, check it, and (with ``write``) move the one leaf. Returns a report."""
    ledger_path = api_root / LEDGER_RELATIVE
    text, ledger = load_ledger(ledger_path)
    gate = ledger.get(GATE_KEY)
    if not isinstance(gate, Mapping) or not isinstance(gate.get(WELD_KEY), Mapping):
        raise Refusal(f"the ledger has no {GATE_KEY}.{WELD_KEY} block")
    weld = gate[WELD_KEY]
    absent = [field for field in WELD_FIELDS if field not in weld]
    if absent or set(weld) != set(WELD_FIELDS):
        raise Refusal(f"{GATE_KEY}.{WELD_KEY} has fields {sorted(weld)}, expected "
                      f"{sorted(WELD_FIELDS)}; this tool does not reshape it")
    ran, skipped = claimed_counts(weld)
    validator = validator_path(api_root, weld)
    watched = (validator,) + loaded_by_path(validator)
    before = {path: sha256_of(path) for path in watched}

    with tempfile.TemporaryDirectory(prefix="numpy_reference_weld_") as scratch:
        out = validator_out if validator_out is not None else (
            Path(scratch) / f"{EXPECTED_CHECK}.json")
        returncode, elapsed = run_validator(validator, out, python)
        after = {path: sha256_of(path) for path in watched}
        moved = sorted(path.name for path in watched if before[path] != after[path])
        if moved:
            raise Refusal(f"{moved} changed while the validator ran; the digest would name "
                          "bytes that did not run")
        if returncode != 0:
            raise Refusal(f"the validator exited {returncode}")
        if not out.is_file():
            raise Refusal(f"the validator wrote no artifact at {out}")
        artifact_sha256 = sha256_of(out)
        summary = check_artifact(json.loads(out.read_text(encoding="utf-8")), ran, skipped)

    new_digest = before[validator]
    old_digest = weld["probe_sha256"]
    report: Dict[str, Any] = {
        "ledger": str(ledger_path),
        "leaf": f"{GATE_KEY}.{WELD_KEY}.probe_sha256",
        "validator": weld["probe"],
        "before": old_digest,
        "after": new_digest,
        "watched": {path.name: before[path] for path in watched},
        "summary": summary,
        "artifact_sha256": artifact_sha256,
        "artifact": str(validator_out) if validator_out is not None else None,
        "elapsed_seconds": round(elapsed, 2),
        "written": False,
    }
    if old_digest == new_digest:
        report["status"] = "current"
        return report

    changed = copy.deepcopy(ledger)
    changed[GATE_KEY][WELD_KEY]["probe_sha256"] = new_digest
    diff = changed_leaves(ledger, changed)
    if diff != [(GATE_KEY, WELD_KEY, "probe_sha256")]:
        raise Refusal(f"the change would move {diff}, not the one leaf")
    if entry_bindings(changed) != entry_bindings(ledger):
        raise Refusal("the change would move a bound digest or a live capability")
    report["status"] = "stale"
    if write:
        ledger_path.write_text(canonical(changed), encoding="utf-8")
        if ledger_path.read_text(encoding="utf-8") != canonical(changed):
            raise Refusal(f"{ledger_path} does not read back as written")
        report["written"] = True
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="write the leaf (default: report only)")
    parser.add_argument("--validator-out", type=Path,
                        help="where the validator writes its artifact (default: a "
                             "temporary directory)")
    args = parser.parse_args(argv)
    try:
        report = record(write=args.write, validator_out=args.validator_out)
    except Refusal as exc:
        print(f"REFUSED: {exc}", flush=True)
        return 2
    summary = report["summary"]
    print(f"validator {report['validator']}: {summary['identical']}/{summary['ran']} "
          f"identical, {summary['skipped']} skipped, pass {summary['pass']} "
          f"({report['elapsed_seconds']} s); artifact sha256 {report['artifact_sha256']}"
          + (f" at {report['artifact']}" if report["artifact"] else ""), flush=True)
    for name, digest in report["watched"].items():
        print(f"  unchanged across the run: {name} {digest}", flush=True)
    if report["status"] == "current":
        print(f"{report['leaf']}: {report['before']} is already the validator's digest; "
              "nothing to write", flush=True)
        return 0
    print(f"{report['leaf']}: {report['before']} -> {report['after']}", flush=True)
    print("one leaf changes; every entry's bound digest and live capabilities are unchanged",
          flush=True)
    print(f"written: {report['ledger']}" if report["written"]
          else "report only; pass --write to write", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
