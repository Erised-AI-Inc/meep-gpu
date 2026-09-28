#!/usr/bin/env python3
"""Re-measure the fixed MEEP capability corpus through the live Metal planner.

This is intentionally not another predicate battery.  Each accepted MEEP case is
captured and lifted, then :func:`parity.meep_gpu.metal_corpus.evaluate` asks the
unmodified production composer which plans it actually selects.  A slot is counted
only when that call returned a selection; a plan that falls back to the array path
is visible as an unselected slot, not a pass.

The corpus this campaign prices is fixed by two lift ledgers of the 2026-08-09 corpus
campaign, the stock-interpreter examples leg and the Python-test leg: 52 accepted
Python examples and 134 accepted upstream Python-test cases, 186 rows.  It contains
759 logical slots (186 x four stepping slots, plus 15 live-polarization slots).  The
ledgers are inputs to this campaign, are SHA-256-stamped in its report, and are
checked for those row counts before a child process starts.  Thus a changed discovery
scope cannot silently inherit the 759 denominator.

BASIS, DATED.  On 2026-09-03 the predicate censuses and the fusion boards moved to a
194-row basis: 60 examples (the stock leg's 52 plus 8 scripts accepted only by the
same campaign's ``_blocked``, ``_gdsii`` and ``_sigma`` legs) and the same 134 tests.
This planner census still reads the stock examples ledger alone and prices the
186-row / 759-slot basis by design; the row-count checks above refuse a ledger of any
other size outright rather than inheriting it.  Extending the census to the four
example legs, each row under the interpreter its lift record names as
``measure_predicate_coverage.py`` does since 2026-09-03, is a named follow-up.

Every child process writes a result record as it lands and a flushed progress line
to ``progress.log``.  Test modules are deliberately replayed in one process per
module because several MEEP tests communicate through files or class state.  A
small, harness-only ``parameterized`` shim is used only where an upstream module
imports that uninstalled package; generated cases are assigned to the ledger by
exactly equal measured facts, never a recreated upstream method name.

The corpus requires four explicit expansion-probe artifacts.  A planner result that
consulted an ambient, unrecorded probe is not useful release evidence.  The campaign
stamps each supplied probe, source test/example, and every imported API Python source
into its output; conflicting imported source bytes make its final release false.

Run from ``the repository root`` on the Apple-Silicon development host::

    MEEP_GPU_SUBNORMAL_POLICY=flush python -u \\
      -m parity.meep_gpu.recut_metal_corpus --out /path/to/corpus-recut \\
      --probe complex=/path/to/complex.json \\
      --probe special_kz=/path/to/special-kz.json \\
      --probe folded_complex=/path/to/folded-complex.json \\
      --probe cylindrical_complex=/path/to/cylindrical-complex.json

No CUDA host is contacted and this program does not launch a Metal shader; device
gates establish arithmetic while this separate campaign establishes live-planner
reachability.  The child mode is internal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
DEFAULT_MEEP_ROOT = Path(os.environ.get("MGPU_SITE_MEEP_SOURCE", os.path.join(os.path.expanduser("~"), "meep")))
DEFAULT_EXAMPLES_LEDGER = API_ROOT / (
    "parity/meep_gpu/results/corpus_final_numpy_2026-08-09_monitorfix/lift.jsonl")
DEFAULT_TESTS_LEDGER = API_ROOT / (
    "parity/meep_gpu/results/meep_python_tests_2026-08-09_monitorfix/lift.jsonl")
SHIM_ROOT = HERE / "parameterized_shim"

EXPECTED_EXAMPLE_ROWS = 52
EXPECTED_TEST_ROWS = 134
EXPECTED_ROWS = EXPECTED_EXAMPLE_ROWS + EXPECTED_TEST_ROWS
EXPECTED_SLOTS = 759
CORE_SLOTS = ("step_B", "step_D", "update_H", "update_E")
PROBE_ENVIRONMENTS = {
    "complex": "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
    "special_kz": "MEEP_GPU_METAL_EXPANSION_PROBE",
    "folded_complex": "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
    "cylindrical_complex": "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE",
}


def sha256(path: Path) -> str:
    """Return the digest of an existing input or runtime source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, failing loudly when an input or child record is malformed."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def accepted_rows(path: Path, *, leg: str) -> list[dict[str, Any]]:
    """Return the fixed accepted ledger rows and reject duplicate identities."""
    rows = [row for row in read_jsonl(path) if row.get("accepted") is True]
    if leg == "examples":
        keys = [str(row.get("script", "")) for row in rows]
    else:
        keys = [f"{row.get('module', '')}::{row.get('case', '')}" for row in rows]
    if any(not key or key.endswith("::") for key in keys):
        raise ValueError(f"{path} contains an accepted {leg} row with no identity")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path} contains duplicate accepted {leg} identities")
    return sorted(rows, key=lambda row: str(row.get("script") or
                                             f"{row.get('module')}::{row.get('case')}"))


def source_inventory(*, examples_dir: Path, tests_dir: Path,
                     examples: Sequence[Mapping[str, Any]],
                     tests: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """Read and hash every external source needed by the fixed corpus before launch.

    ``Path.is_dir`` can succeed under macOS privacy controls while a child Python
    process is still refused when it opens a MEEP script.  A full corpus campaign
    must not discover that after it has already produced a partial-looking report,
    so preflight reads every selected example and every selected test *module*.
    """
    paths = {examples_dir / str(row["script"]) for row in examples}
    paths.update(tests_dir / str(row["module"]) for row in tests)
    sources: dict[str, str] = {}
    errors: list[str] = []
    for path in sorted(paths):
        try:
            sources[str(path.resolve())] = sha256(path)
        except OSError as exc:
            errors.append(f"cannot read corpus source {path}: {type(exc).__name__}: {exc}")
    return sources, errors


def facts_key(facts: Mapping[str, Any] | None) -> str | None:
    """Stable key for the survey's measured configuration facts."""
    if not isinstance(facts, Mapping):
        return None
    return json.dumps(facts, sort_keys=True, default=str, separators=(",", ":"))


def uses_parameterized(module_path: Path) -> bool:
    """Whether this source requires the constrained harness-only import path."""
    text = module_path.read_text(encoding="utf-8", errors="replace")
    return "parameterized" in text and "import" in text


def mps_status() -> dict[str, Any]:
    """Record native-MPS availability rather than treating a planner-only host as MPS."""
    try:
        import torch  # noqa: PLC0415

        built = bool(getattr(torch.backends.mps, "is_built", lambda: False)())
        available = bool(getattr(torch.backends.mps, "is_available", lambda: False)())
        return {"torch": str(torch.__version__), "built": built, "available": available}
    except Exception as exc:  # noqa: BLE001 - availability failure is an evidence result.
        return {"torch": None, "built": False, "available": False,
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def _probe_paths(values: Iterable[str]) -> dict[str, Path]:
    """Parse and validate the four explicitly supplied, source-welded probe inputs."""
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--probe must be NAME=/absolute/or/relative/path.json")
        if name not in PROBE_ENVIRONMENTS:
            raise ValueError(f"unknown probe {name!r}; expected {sorted(PROBE_ENVIRONMENTS)}")
        if name in output:
            raise ValueError(f"probe {name!r} was supplied more than once")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"probe {name!r} does not exist: {path}")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"probe {name!r} is not readable JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"probe {name!r} must have a JSON-object root")
        output[name] = path
    missing = sorted(set(PROBE_ENVIRONMENTS) - set(output))
    if missing:
        raise ValueError(f"missing explicit probe(s): {', '.join(missing)}")
    return output


def _set_probe_environment(probes: Mapping[str, Path]) -> None:
    for name, environment in PROBE_ENVIRONMENTS.items():
        os.environ[environment] = str(probes[name])


def validate_probe_records(probes: Mapping[str, Path]) -> dict[str, str]:
    """Require that each supplied record licenses the matching live family.

    A syntactically valid JSON file is not necessarily an expansion experiment for
    this executor or a complete probe for the family that reads it.  Calling each
    family's own parser keeps those rules in one place and turns an old/foreign
    artifact into an early campaign refusal rather than a broad unexplained gap.
    """
    from meep_gpu.metal_kernels import (  # noqa: PLC0415
        complex_fields,
        cylindrical_complex,
        folded_complex,
        special_kz,
    )

    parsers = {
        "complex": complex_fields.expansion_from_probe,
        "special_kz": special_kz.beta_expansion_from_probe,
        "folded_complex": folded_complex.expansion_from_probe,
        "cylindrical_complex": cylindrical_complex.expansion_from_probe,
    }
    arms: dict[str, str] = {}
    for name, parser in parsers.items():
        record = json.loads(probes[name].read_text(encoding="utf-8"))
        arm = parser(record)
        if arm is None:
            raise ValueError(
                f"probe {name!r} does not license an unambiguous current Metal "
                f"expansion arm")
        arms[name] = str(arm)
    return arms


def _probe_record(probes: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sorted(probes.items())}


def _progress(path: Path, identity: str, message: str, started: float) -> None:
    line = (f"[{time.strftime('%H:%M:%S')}] {identity:<58.58} "
            f"+{time.time() - started:7.1f}s  {message}\n")
    print(line, end="", flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _runtime_sources() -> dict[str, str]:
    """Hash actual modules imported by a child after it has asked the live planner."""
    from parity.meep_gpu.metal_gate_runner import runtime_source_sha256  # noqa: PLC0415

    return runtime_source_sha256(gate_path=Path(__file__).resolve())


def _stamp_record(record: dict[str, Any], *, source: Path,
                  probes: Mapping[str, Path]) -> dict[str, Any]:
    record["input_sha256"] = {
        "simulation_source": {"path": str(source.resolve()), "sha256": sha256(source)},
        "expansion_probes": _probe_record(probes),
    }
    record["source_sha256"] = _runtime_sources()
    return record


def _lift_and_evaluate(sim: Any, record: dict[str, Any], *,
                       timeout_hooks: tuple[Any, Any] | None = None) -> dict[str, Any]:
    """Lift exactly once, then query the production Metal planner without mutation."""
    availability = mps_status()
    record["mps"] = availability
    if not availability.get("available"):
        record["measured"] = False
        record["note"] = "native MPS is unavailable; planner reachability is not an MPS result"
        return record
    import meep_gpu  # noqa: PLC0415
    from parity.meep_gpu import metal_corpus  # noqa: PLC0415

    started = time.time()
    alarm, cancel = timeout_hooks or (None, None)
    if alarm is not None:
        alarm()
    driver = None
    try:
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
        record["lift_s"] = round(time.time() - started, 3)
        record["grid_shape"] = [int(value) for value in driver.shape]
        record["grid_cells"] = int(math.prod(int(value) for value in driver.shape))
        record.update(metal_corpus.evaluate(driver))
        record["measured"] = True
    except BaseException as exc:  # noqa: BLE001 - a row-level lift refusal is evidence.
        record["measured"] = False
        record["lift_error"] = f"{type(exc).__name__}: {exc}"[:1200]
        record["lift_s"] = round(time.time() - started, 3)
    finally:
        if cancel is not None:
            cancel()
        if driver is not None:
            try:
                driver.close()
            except BaseException:  # noqa: BLE001 - close must not erase the row outcome.
                pass
    return record


def child_example(*, script: Path, out_json: Path, progress_log: Path,
                  probes: Mapping[str, Path]) -> int:
    """Capture, lift, and census exactly one example script in a fresh process."""
    _set_probe_environment(probes)
    started = time.time()
    identity = script.name
    _progress(progress_log, identity, "capture starting", started)
    record: dict[str, Any] = {"leg": "examples", "row": script.name,
                              "candidate": script.name}
    try:
        from parity.meep_gpu.sweep_corpus_lift_parity import capture_simulation  # noqa: PLC0415

        captured, sim, restore = capture_simulation(str(script))
        record.update(captured)
        _progress(progress_log, identity,
                  f"captured has_sim={record.get('has_simulation')}", started)
        if sim is None:
            record.update({"measured": False, "note": "script built no mp.Simulation"})
        else:
            restore()
            record = _lift_and_evaluate(sim, record)
    except BaseException as exc:  # noqa: BLE001 - preserve a child failure as one durable row.
        record.update({"measured": False,
                       "child_error": f"{type(exc).__name__}: {exc}"[:1200]})
    _stamp_record(record, source=script, probes=probes)
    _atomic_json(out_json, record)
    _progress(progress_log, identity,
              f"measured={record.get('measured')} lift_s={record.get('lift_s')}", started)
    return 0


def _parameterized_metadata(module: Any, class_name: str, method_name: str) -> dict[str, Any] | None:
    method = getattr(getattr(module, class_name, None), method_name, None)
    if getattr(method, "__parameterized_index__", None) is None:
        return None
    return {
        "source": str(method.__parameterized_source__),
        "index": int(method.__parameterized_index__),
        "args": [repr(value)[:240] for value in method.__parameterized_args__],
    }


def child_module(*, module_path: Path, wanted: set[str], output_jsonl: Path,
                 progress_log: Path, probes: Mapping[str, Path], case_timeout: float,
                 take_all: bool) -> int:
    """Replay a test module in order, flushing a raw record after every selected case."""
    _set_probe_environment(probes)
    from parity.meep_gpu import survey_meep_tests as harness  # noqa: PLC0415

    started = time.time()
    identity = module_path.name
    namespace = harness.build_child_namespace()
    try:
        module = namespace["_import_module"](str(module_path))
    except BaseException as exc:  # noqa: BLE001 - parent synthesizes every expected row.
        _progress(progress_log, identity, f"MODULE IMPORT FAILED {type(exc).__name__}", started)
        _append_jsonl(output_jsonl, {
            "leg": "tests", "module": module_path.name, "row": "<module import>",
            "measured": False, "module_error": f"{type(exc).__name__}: {exc}"[:1200],
        })
        return 0

    cases = namespace["_enumerate_cases"](module)
    _progress(progress_log, identity, f"imported {len(cases)} runnable cases", started)
    for index, (class_name, method_name) in enumerate(cases, 1):
        case_id = f"{class_name}.{method_name}"
        record, sim, restore, _case = namespace["_run_case"](
            module, str(module_path), class_name, method_name, case_timeout)
        # _run_case installs MEEP capture stubs.  Restoring before either the next
        # case or the lift is mandatory; otherwise this harness manufactures a
        # refusal for later rows.
        restore()
        if not take_all and case_id not in wanted:
            continue
        record.update({"leg": "tests", "module": module_path.name, "row": case_id,
                       "candidate": case_id})
        metadata = _parameterized_metadata(module, class_name, method_name)
        if metadata is not None:
            record["parameterized"] = metadata
        if sim is None:
            record.update({"measured": False, "note": "test replay built no mp.Simulation"})
        else:
            record = _lift_and_evaluate(
                sim, record,
                timeout_hooks=(
                    lambda: namespace["_alarm"](case_timeout),
                    namespace["_cancel_alarm"],
                ),
            )
        _stamp_record(record, source=module_path, probes=probes)
        _append_jsonl(output_jsonl, record)
        _progress(progress_log, identity,
                  f"case {index}/{len(cases)} {case_id}: measured={record.get('measured')}",
                  started)
    return 0


def match_parameterized_rows(raw: Sequence[dict[str, Any]],
                             expected: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Map shim-generated test rows to ledger identities by exact measured facts.

    A group of fact-identical cases can only be assigned up to a permutation.  It is
    retained when its generated and ledger group sizes match, marked as such in each
    row, and rejected when the sizes differ.  Counts over the group are invariant to
    that permutation; pretending a more specific identity would not be.
    """
    expected_by_facts: dict[str, list[dict[str, Any]]] = {}
    for row in expected:
        key = facts_key(row.get("facts"))
        if key is None:
            return [], [f"ledger row {row.get('case')} has no facts for shim matching"]
        expected_by_facts.setdefault(key, []).append(row)
    raw_by_facts: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for row in raw:
        if row.get("row") == "<module import>":
            errors.append(f"{row.get('module')}: module import failed")
            continue
        key = facts_key(row.get("facts"))
        if key is None:
            errors.append(f"{row.get('module')}::{row.get('row')} has no facts")
            continue
        raw_by_facts.setdefault(key, []).append(row)

    matched: list[dict[str, Any]] = []
    for key, candidates in expected_by_facts.items():
        observed = raw_by_facts.pop(key, [])
        candidates = sorted(candidates, key=lambda row: str(row.get("case")))
        observed = sorted(observed, key=lambda row: str(row.get("row")))
        if len(observed) != len(candidates):
            errors.append(
                f"facts group has {len(observed)} generated rows but {len(candidates)} ledger rows "
                f"({', '.join(str(row.get('case')) for row in candidates)})")
            continue
        for candidate, record in zip(candidates, observed):
            record = dict(record)
            record["shim_row"] = record["row"]
            record["row"] = str(candidate["case"])
            record["matched_by"] = "exact facts group"
            if len(candidates) > 1:
                record["group_permutation_unresolved"] = True
            matched.append(record)
    # ``take_all`` replays every generated case so upstream module ordering and side
    # effects match the original survey.  Some of those cases were not accepted by
    # the fixed corpus ledger and therefore have no place in its denominator.  They
    # are intentionally not errors merely for existing; a mismatch in an *accepted*
    # facts group above remains an error.
    return matched, errors


def _synthetic_missing(*, leg: str, expected: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """Preserve a fixed ledger identity when a child did not return a usable row."""
    row = str(expected.get("script") if leg == "examples" else expected.get("case"))
    output: dict[str, Any] = {"leg": leg, "row": row, "measured": False,
                              "note": reason, "ledger_facts": expected.get("facts")}
    if leg == "tests":
        output["module"] = expected.get("module")
    return output


def canonicalize_rows(*, leg: str, raw: Sequence[dict[str, Any]],
                      expected: Sequence[dict[str, Any]], parameterized: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Return exactly the ledger rows where possible, preserving failures explicitly."""
    errors: list[str] = []
    if parameterized:
        usable, matching_errors = match_parameterized_rows(raw, expected)
        errors.extend(matching_errors)
    else:
        wanted = {str(row.get("script") if leg == "examples" else row.get("case"))
                  for row in expected}
        usable = [dict(row) for row in raw if str(row.get("row")) in wanted]
        extras = [row for row in raw if str(row.get("row")) not in wanted]
        errors.extend(f"unexpected {leg} raw row {row.get('row')}" for row in extras)

    by_row: dict[str, dict[str, Any]] = {}
    for record in usable:
        key = str(record["row"])
        if key in by_row:
            errors.append(f"duplicate canonical {leg} row {key}")
        else:
            by_row[key] = record
    canonical: list[dict[str, Any]] = []
    for ledger in expected:
        key = str(ledger.get("script") if leg == "examples" else ledger.get("case"))
        record = by_row.pop(key, None)
        if record is None:
            canonical.append(_synthetic_missing(
                leg=leg, expected=ledger, reason="child returned no canonical row"))
            errors.append(f"missing canonical {leg} row {key}")
        else:
            canonical.append(record)
    errors.extend(f"extra canonical {leg} row {key}" for key in by_row)
    return canonical, errors


def merge_sources(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """Merge actual-runtime source digests and make mixed-tree evidence a failure."""
    merged: dict[str, str] = {}
    errors: list[str] = []
    for row in rows:
        source_map = row.get("source_sha256")
        if not row.get("measured"):
            continue
        if not isinstance(source_map, Mapping) or not source_map:
            errors.append(f"{row.get('leg')}::{row.get('row')} lacks runtime source digests")
            continue
        for name, digest in source_map.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                errors.append(f"{row.get('leg')}::{row.get('row')} has malformed source digest")
                continue
            previous = merged.setdefault(name, digest)
            if previous != digest:
                errors.append(f"mixed runtime source digest for {name}")
    return dict(sorted(merged.items())), errors


def verify_runtime_sources(sources: Mapping[str, str], *, root: Path = API_ROOT) -> list[str]:
    """Rehash the sources a child reported against the tree at campaign close."""
    errors: list[str] = []
    root = root.resolve()
    for name, digest in sources.items():
        path = Path(name)
        try:
            path.resolve().relative_to(root)
        except ValueError:
            errors.append(f"runtime source lies outside API root: {name}")
            continue
        if not path.is_file():
            errors.append(f"runtime source disappeared before campaign close: {name}")
            continue
        if sha256(path) != digest:
            errors.append(f"runtime source changed during campaign: {name}")
    return errors


def verify_row_inputs(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Rehash source scripts and probe artifacts, including resumed child rows."""
    errors: list[str] = []
    for row in rows:
        if not row.get("measured"):
            continue
        inputs = row.get("input_sha256")
        if not isinstance(inputs, Mapping):
            errors.append(f"{row.get('leg')}::{row.get('row')} lacks input digests")
            continue
        records: list[Mapping[str, Any]] = []
        source = inputs.get("simulation_source")
        if isinstance(source, Mapping):
            records.append(source)
        else:
            errors.append(f"{row.get('leg')}::{row.get('row')} lacks simulation-source digest")
        probes = inputs.get("expansion_probes")
        if isinstance(probes, Mapping):
            records.extend(value for value in probes.values() if isinstance(value, Mapping))
        else:
            errors.append(f"{row.get('leg')}::{row.get('row')} lacks probe digests")
        for record in records:
            name, digest = record.get("path"), record.get("sha256")
            if not isinstance(name, str) or not isinstance(digest, str):
                errors.append(f"{row.get('leg')}::{row.get('row')} has malformed input digest")
                continue
            path = Path(name)
            if not path.is_file():
                errors.append(f"input disappeared before campaign close: {name}")
            elif sha256(path) != digest:
                errors.append(f"input changed during campaign: {name}")
    return errors


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize only actual live planner measurements; never infer missing slots."""
    from parity.meep_gpu import metal_corpus  # noqa: PLC0415

    result = metal_corpus.summary([dict(row) for row in rows])
    result["expected_rows"] = EXPECTED_ROWS
    result["row_denominator_matches"] = len(rows) == EXPECTED_ROWS
    result["selected_by_arm"] = {
        arm: sum(1 for row in rows for _slot, selected in (row.get("selected") or {}).items()
                 if selected == arm)
        for arm in sorted({selected for row in rows
                           for selected in (row.get("selected") or {}).values()})
    }
    return result


def _write_manifest(path: Path, sources: Mapping[str, str]) -> None:
    _atomic_text(path, "".join(f"{digest}\t{name}\n" for name, digest in sources.items()))


def _command_for_child(arguments: argparse.Namespace, *, mode: str,
                       probes: Mapping[str, Path], extra: Sequence[str]) -> list[str]:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", mode,
               "--subnormal-policy", arguments.subnormal_policy]
    for name, path in sorted(probes.items()):
        command.extend(["--probe", f"{name}={path}"])
    command.extend(extra)
    return command


def _child_environment(*, parameterized: bool) -> dict[str, str]:
    environment = dict(os.environ)
    roots = [str(API_ROOT)]
    if parameterized:
        roots.insert(0, str(SHIM_ROOT))
    existing = environment.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    environment["MEEP_GPU_SUBNORMAL_POLICY"] = "flush"
    return environment


def _run_child(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str],
               timeout: float) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, cwd=str(cwd), env=dict(environment),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   timeout=timeout, check=False)
        return completed.returncode, (completed.stderr or b"").decode("utf-8", "replace")[-1600:]
    except subprocess.TimeoutExpired as exc:
        text = (exc.stderr or b"").decode("utf-8", "replace")[-1600:]
        return 124, text


def _load_child_records(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def run_campaign(arguments: argparse.Namespace) -> int:
    """Run the two corpus legs sequentially and release only a fully welded census."""
    if arguments.subnormal_policy != "flush":
        raise SystemExit("the Metal corpus is only valid under --subnormal-policy flush")
    probes = _probe_paths(arguments.probe)
    probe_arms = validate_probe_records(probes)
    meep_root = Path(arguments.meep_root).expanduser().resolve()
    examples_dir, tests_dir = meep_root / "python/examples", meep_root / "python/tests"
    if not examples_dir.is_dir() or not tests_dir.is_dir():
        raise SystemExit(f"MEEP checkout lacks readable python/examples and python/tests: {meep_root}")
    examples_ledger = Path(arguments.examples_ledger).resolve()
    tests_ledger = Path(arguments.tests_ledger).resolve()
    if not examples_ledger.is_file() or not tests_ledger.is_file():
        raise SystemExit("the fixed corpus lift ledgers must both exist")
    examples = accepted_rows(examples_ledger, leg="examples")
    tests = accepted_rows(tests_ledger, leg="tests")
    if len(examples) != EXPECTED_EXAMPLE_ROWS or len(tests) != EXPECTED_TEST_ROWS:
        raise SystemExit(
            f"fixed corpus mismatch: expected {EXPECTED_EXAMPLE_ROWS}+{EXPECTED_TEST_ROWS} "
            f"accepted rows, got {len(examples)}+{len(tests)}")
    corpus_sources, source_errors = source_inventory(
        examples_dir=examples_dir, tests_dir=tests_dir, examples=examples, tests=tests)

    root = Path(arguments.out).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not arguments.resume:
        raise SystemExit(f"output directory is non-empty: {root}; use a new directory or --resume")
    root.mkdir(parents=True, exist_ok=True)
    progress = root / "progress.log"
    if not progress.exists():
        _atomic_text(progress, "")
    manifest = {
        "schema": 1,
        "created_unix_s": time.time(),
        "meep_root": str(meep_root),
        "expected": {"examples": EXPECTED_EXAMPLE_ROWS, "tests": EXPECTED_TEST_ROWS,
                     "rows": EXPECTED_ROWS, "slots": EXPECTED_SLOTS},
        "subnormal_policy": arguments.subnormal_policy,
        "inputs": {
            "examples_ledger": {"path": str(examples_ledger), "sha256": sha256(examples_ledger)},
            "tests_ledger": {"path": str(tests_ledger), "sha256": sha256(tests_ledger)},
            "probes": _probe_record(probes),
            "probe_arms": probe_arms,
        },
    }
    if arguments.preflight:
        report = {
            **manifest,
            "preflight": {
                "mps": mps_status(),
                "corpus_source_sha256": corpus_sources,
                "readable_sources": len(corpus_sources),
                "expected_source_files": len({str(row["script"]) for row in examples})
                                         + len({str(row["module"]) for row in tests}),
            },
            "release": {"released": not source_errors, "reasons": source_errors},
        }
        _atomic_json(root / "preflight.json", report)
        print(f"[corpus preflight] released={not source_errors} "
              f"sources={len(corpus_sources)} out={root}", flush=True)
        return 0 if not source_errors else 75
    if source_errors:
        raise SystemExit("corpus source preflight failed:\n" + "\n".join(source_errors))
    _atomic_json(root / "campaign.json", manifest)
    started = time.time()
    all_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    # Examples are independent and are intentionally isolated one script per child.
    examples_rows_dir = root / "raw_examples"
    examples_rows_dir.mkdir(exist_ok=True)
    work = root / "examples_workdir"
    work.mkdir(exist_ok=True)
    for entry in examples_dir.iterdir():
        if entry.suffix in (".py", ".ipynb"):
            continue
        link = work / entry.name
        if not link.exists():
            try:
                link.symlink_to(entry)
            except OSError:
                # A missing helper asset will be recorded by the child rather than
                # silently changing its working directory semantics.
                pass
    for index, ledger in enumerate(examples, 1):
        name = str(ledger["script"])
        out = examples_rows_dir / f"{Path(name).stem}.json"
        if arguments.resume and out.is_file():
            raw = [json.loads(out.read_text(encoding="utf-8"))]
        else:
            command = _command_for_child(
                arguments, mode="example", probes=probes,
                extra=["--script", str(examples_dir / name), "--out-json", str(out),
                       "--progress-log", str(progress)],
            )
            code, stderr = _run_child(command, cwd=work, environment=_child_environment(parameterized=False),
                                       timeout=arguments.example_timeout)
            raw = [json.loads(out.read_text(encoding="utf-8"))] if out.is_file() else []
            if code != 0:
                errors.append(f"example child {name} exited {code}: {stderr}")
        canonical, row_errors = canonicalize_rows(
            leg="examples", raw=raw, expected=[ledger], parameterized=False)
        all_rows.extend(canonical)
        errors.extend(row_errors)
        _progress(progress, "examples", f"{index}/{len(examples)} {name}", started)

    # Upstream tests are replayed module by module, preserving their observed order.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ledger in tests:
        grouped.setdefault(str(ledger["module"]), []).append(ledger)
    raw_tests_dir = root / "raw_tests"
    raw_tests_dir.mkdir(exist_ok=True)
    for index, (module_name, wanted_rows) in enumerate(sorted(grouped.items()), 1):
        module_path = tests_dir / module_name
        raw_path = raw_tests_dir / f"{Path(module_name).stem}.jsonl"
        parameterized = uses_parameterized(module_path)
        if arguments.resume and raw_path.is_file():
            raw = _load_child_records(raw_path)
        else:
            command = _command_for_child(
                arguments, mode="module", probes=probes,
                extra=["--module-path", str(module_path), "--wanted-json",
                       json.dumps([str(row["case"]) for row in wanted_rows]),
                       "--out-jsonl", str(raw_path), "--progress-log", str(progress),
                       "--case-timeout", str(arguments.case_timeout),
                       "--take-all" if parameterized else "--no-take-all"],
            )
            code, stderr = _run_child(command, cwd=root,
                                       environment=_child_environment(parameterized=parameterized),
                                       timeout=arguments.module_timeout)
            raw = _load_child_records(raw_path)
            if code != 0:
                errors.append(f"test child {module_name} exited {code}: {stderr}")
        canonical, row_errors = canonicalize_rows(
            leg="tests", raw=raw, expected=wanted_rows, parameterized=parameterized)
        all_rows.extend(canonical)
        errors.extend(row_errors)
        _progress(progress, "tests", f"{index}/{len(grouped)} {module_name}", started)

    all_rows.sort(key=lambda row: (str(row.get("leg")), str(row.get("module", "")), str(row.get("row"))))
    _atomic_text(root / "rows.jsonl", "".join(
        json.dumps(row, sort_keys=True, default=str) + "\n" for row in all_rows))
    sources, source_errors = merge_sources(all_rows)
    errors.extend(source_errors)
    errors.extend(verify_runtime_sources(sources))
    errors.extend(verify_row_inputs(all_rows))
    _write_manifest(root / "source_sha256.txt", sources)
    census = summarize_rows(all_rows)
    if census["rows"] != EXPECTED_ROWS:
        errors.append(f"row count is {census['rows']}, not {EXPECTED_ROWS}")
    if census["measured_rows"] != EXPECTED_ROWS:
        errors.append(f"only {census['measured_rows']}/{EXPECTED_ROWS} rows reached the live planner")
    if census["slots"] != EXPECTED_SLOTS:
        errors.append(f"measured slot denominator is {census['slots']}, not {EXPECTED_SLOTS}")
    if not sources:
        errors.append("no measured row supplied a runtime source manifest")
    report = {
        **manifest,
        "completed_unix_s": time.time(),
        "elapsed_s": round(time.time() - started, 3),
        "census": census,
        "source_sha256": sources,
        "release": {"released": not errors, "reasons": errors},
    }
    _atomic_json(root / "report.json", report)
    print(f"[corpus] released={not errors} rows={census['rows']} "
          f"measured={census['measured_rows']} slots={census['slots']}/{EXPECTED_SLOTS} "
          f"out={root}", flush=True)
    return 0 if not errors else 75


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="new directory for campaign artifacts")
    parser.add_argument("--meep-root", default=str(DEFAULT_MEEP_ROOT))
    parser.add_argument("--examples-ledger", default=str(DEFAULT_EXAMPLES_LEDGER))
    parser.add_argument("--tests-ledger", default=str(DEFAULT_TESTS_LEDGER))
    parser.add_argument("--probe", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--subnormal-policy", default="flush")
    parser.add_argument("--example-timeout", type=float, default=900.0)
    parser.add_argument("--case-timeout", type=float, default=300.0)
    parser.add_argument("--module-timeout", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true",
                        help="hash every corpus input without importing or stepping MEEP")

    parser.add_argument("--child", choices=("example", "module"), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--script", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--module-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--wanted-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out-jsonl", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress-log", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--take-all", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--no-take-all", dest="take_all", action="store_false", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        probes = _probe_paths(arguments.probe)
        validate_probe_records(probes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if arguments.subnormal_policy != "flush":
        raise SystemExit("the Metal corpus is only valid under --subnormal-policy flush")
    if arguments.child == "example":
        if not all((arguments.script, arguments.out_json, arguments.progress_log)):
            raise SystemExit("example child requires --script, --out-json, and --progress-log")
        return child_example(script=Path(arguments.script).resolve(), out_json=Path(arguments.out_json),
                             progress_log=Path(arguments.progress_log), probes=probes)
    if arguments.child == "module":
        if not all((arguments.module_path, arguments.wanted_json, arguments.out_jsonl,
                    arguments.progress_log)):
            raise SystemExit("module child requires module, wanted, output, and progress arguments")
        return child_module(
            module_path=Path(arguments.module_path).resolve(),
            wanted=set(json.loads(arguments.wanted_json)), output_jsonl=Path(arguments.out_jsonl),
            progress_log=Path(arguments.progress_log), probes=probes,
            case_timeout=float(arguments.case_timeout), take_all=bool(arguments.take_all),
        )
    if not arguments.out:
        raise SystemExit("parent mode requires --out")
    return run_campaign(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
