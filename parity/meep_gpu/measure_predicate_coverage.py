"""Lift every engine-accepted corpus row and call one backend's live predicates on it.

WHICH backend is ``--battery``'s answer, not this file's: it was the Triton battery
when this driver was written, the Metal battery is the default now, and the header
said "Triton" through both. A driver that names a backend it does not choose is how a
census gets read as the other one's.

The denominators are the lift records of the 2026-08-09 corpus campaign and are
quoted, not recomputed:

* ``parity/meep_gpu/results/corpus_final_numpy_2026-08-09_monitorfix*/lift.jsonl``
  — MEEP's example corpus. The stock leg holds 57 rows that built an
  ``mp.Simulation``, 52 engine-accepted. The campaign's three sibling legs lifted the
  examples the stock interpreter cannot: ``_blocked`` (3 rows whose scripts take
  argv), ``_gdsii`` (2 rows needing ``gdspy``) and ``_sigma`` (3 rows needing the
  patched σ-reader). 60 distinct scripts are accepted across the four; see
  ``EXAMPLES_LIFTS`` for how they are unioned and under which interpreter each runs.
* ``parity/meep_gpu/results/meep_python_tests_2026-08-09_monitorfix/lift.jsonl``
  — MEEP's ``python/tests`` corpus, 324 case rows, 134 engine-accepted (133 of which
  actually lifted; ``TestLDOS.test_ldos_3D`` was above that survey's lift-cell cap).

194 = 60 + 134 is the headline from 2026-09-03; this run re-lifts them and reports
what the predicates say, per row and per sub-step. It was 186 = 52 + 134 before,
because this driver read only the stock leg while the campaign's own sibling legs had
accepted eight more scripts — priced as +16 seam-instances per backend by the
fusion-residue audit (``the design notes (meep-gpu-fusion-residue-audit)`` §8).

Every row is stamped with ``subject_manifest_sha256`` (a digest over the battery's
subject package plus the shared engine files, computed once before any leg runs) and
``battery_sha256``, so a board can refuse a census cut against a different tree
instead of pairing a stale selection with a fresh read of device code.

The capture and lift machinery is the surveys' own, imported rather than re-written, so
the object the predicates see is the object the corpus campaign scored.

Rule 7: one flushed line per row, a JSONL row appended as it lands, and a shared
progress log every child appends to.

Usage (from ``the repository root``)::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/measure_predicate_coverage.py \\
        --battery cuda_predicate_battery --leg examples --out <dir>
    ... --leg tests --out <dir>
    ... --leg tests_param --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
#: The package root that makes ``parity.meep_gpu.*`` and ``meep_gpu.*`` importable.
#: Found BY NAME, never by depth: this file has already moved once (out of a
#: results/ tranche into the parity dir), and a hard-coded ``parents[N]`` followed
#: it silently -- every child then died on ModuleNotFoundError while the parent
#: still wrote a full-length jsonl of ``measured: false`` rows that looks like a
#: census. Refuse to run rather than resolve to the wrong root.
def _find_api_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(
        f"cannot locate the repository root above {start}: no ancestor holds "
        "both parity/meep_gpu and meep_gpu. Refusing to guess -- a wrong root makes "
        "every child die on import while still producing a plausible-looking census.")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

#: WHICH BACKEND'S PREDICATES THIS CENSUS CALLS. The lift, the corpus, the three
#: legs, the param match and the 194 = 60 + 134 denominator are backend-independent
#: -- only the battery differs -- which is exactly what makes a Metal number and a
#: Triton number commensurable rather than merely similar-looking. Selected by
#: MODULE NAME so the driver stays one file: ``predicate_battery`` (Metal, the
#: default, unchanged for every caller that predates this switch) or
#: ``triton_predicate_battery``. Named, never derived from the output directory's
#: name -- a census that silently measured the other backend would still be
#: full-length, right-shaped and exit 0, which is this harness's whole failure mode.
DEFAULT_BATTERY = "predicate_battery"


def _battery_module(module_name: str):
    """Import the named battery from beside this file and return the module."""
    import importlib  # noqa: PLC0415
    module = importlib.import_module(module_name)
    # The battery must have come from THIS directory, not from some other entry on
    # PYTHONPATH that happens to carry the name. A census attributed to the wrong
    # battery is unfalsifiable after the fact, so the file is reported on every run.
    print(f"[battery] {module_name} -> {getattr(module, '__file__', '?')}", flush=True)
    return module


def _load_battery(module_name: str):
    """The named battery's ``evaluate``."""
    return _battery_module(module_name).evaluate

# --- the runtime preflight ---------------------------------------------------------------
#
# WHY. A predicate battery may need a runtime the process does not have -- the Metal
# battery launches through torch. Evaluated anyway, every predicate of every row refuses
# for that one process-level reason, the row is still full-length and right-shaped, and
# the board reads it as a coverage gap: on 2026-09-03 the five gdsii/sigma example rows,
# run under their lift records' own interpreters (which carried no torch), reached the
# Metal board as eleven missing halves. So the battery may declare ``runtime_reasons()``
# and the driver asks it ONCE per interpreter before any row runs there: the census's
# own interpreter must answer ``[]`` or the cut is refused outright; a sibling
# interpreter that does not answer ``[]`` has every one of its rows refused BY NAME,
# the record naming the interpreter, the lift record and the reasons -- never a gap.

RUNTIME_PREFLIGHT_PREFIX = "RUNTIME_REASONS_JSON="


def runtime_reasons_of(module_name: str) -> list:
    """The battery's own statement of why THIS process cannot evaluate it, or ``[]``.

    A battery that declares no ``runtime_reasons`` has no process-level precondition
    (the Triton and CUDA batteries evaluate their predicates in pure Python).
    """
    hook = getattr(_battery_module(module_name), "runtime_reasons", None)
    return [str(reason) for reason in hook()] if hook is not None else []


def preflight_runtime(interpreter: str, battery: str, environment: dict,
                      timeout: float = 300.0) -> list:
    """``runtime_reasons_of`` evaluated UNDER ``interpreter``; fails closed.

    The census's own process asks in-process. Any other interpreter is asked by running
    this file's ``--runtime-preflight`` child and reading the one line it prints; a
    child that dies, times out or prints no list is itself a reason, never an empty
    answer -- an interpreter whose answer cannot be read is not one a row may run under.
    """
    if interpreter == sys.executable:
        return runtime_reasons_of(battery)
    command = [interpreter, "-u", str(Path(__file__).resolve()), "--leg", "examples",
               "--runtime-preflight", "--battery", battery]
    try:
        completed = subprocess.run(command, env=environment, capture_output=True,
                                   timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return [f"runtime preflight under {interpreter} timed out after {timeout:g} s"]
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", "replace")[-600:].strip()
        return [f"runtime preflight under {interpreter} exited {completed.returncode}: "
                f"{tail or 'no stderr'}"]
    for line in completed.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith(RUNTIME_PREFLIGHT_PREFIX):
            value = json.loads(line[len(RUNTIME_PREFLIGHT_PREFIX):])
            if isinstance(value, list):
                return [str(reason) for reason in value]
    return [f"runtime preflight under {interpreter} printed no reason list"]


def runtime_refusal(name: str, interpreter: str, origin: str, reasons: list) -> dict:
    """The row record for a refusal BY NAME: which interpreter, named by which record, why."""
    return {"row": name, "measured": False,
            "note": (f"subject runtime unavailable under {interpreter} (named by "
                     f"{origin}): " + "; ".join(reasons)),
            "runtime_reasons": list(reasons)}


# --- the subject digest ----------------------------------------------------------------
#
# WHAT IT PINS. The predicates and the composer are read from the kernels package the
# battery measures, and the engine facts from a handful of shared engine files. A row
# records one digest over all of them so a board that mixes this census's selection with
# a fresh read of device code can refuse when the two describe different trees
# (``build_cuda_fusion_matrix.subject_pin`` recomputes the same manifest against the
# working tree and compares). The package is the battery's OWN declaration,
# ``SUBJECT_PACKAGE`` — the CUDA battery imports all three kernels packages because it
# compares against the other two, so the imports cannot name the subject; a table here
# could go stale without anything noticing.

#: Engine files every battery's predicates read facts from, beside the kernels package.
SUBJECT_EXTRA = ("meep_gpu/stepping.py", "meep_gpu/fields.py",
                 "meep_gpu/dispersion.py", "meep_gpu/pml.py", "meep_gpu/grid.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def battery_imports(battery: Path, package: str) -> list:
    """The ``meep_gpu.<package>`` modules the battery names, read off its source.

    Derived, not transcribed: a leg added later is picked up without editing a list
    here, and a list here could go stale silently.
    """
    import ast  # noqa: PLC0415

    text = battery.read_text(encoding="utf-8")
    prefix = f"meep_gpu.{package}"
    names = set(re.findall(re.escape(prefix) + r"\.([A-Za-z_][A-Za-z0-9_]*)", text))
    # AST, not only the regex: ``from meep_gpu.<pkg> import (  # noqa\n mod as arm)``
    # is how two legs import, and a line-oriented pattern misses it -- which it did,
    # silently, until this was checked against the module list.
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == prefix:
                names.update(alias.name for alias in node.names)
            elif node.module.startswith(prefix + "."):
                names.add(node.module.split(".")[2])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix + "."):
                    names.add(alias.name.split(".")[2])
    return sorted(names)


def subject_digest(api_root: Path, package: str, battery: Path) -> dict:
    """sha256 of every subject file, plus one digest over the whole manifest.

    The aggregate is a sha256 over ``"<relpath>  <sha256>\\n"`` lines in sorted
    order, so it changes if a file's content changes, if a file is added, or if one
    is removed -- the three ways the subject can move under a census.
    """
    glob = f"meep_gpu/{package}/*.py"
    entries = {}
    for path in sorted(api_root.glob(glob)):
        entries[str(path.relative_to(api_root))] = _sha256(path)
    for name in SUBJECT_EXTRA:
        path = api_root / name
        if path.exists():
            entries[name] = _sha256(path)
    manifest = "".join(f"{name}  {digest}\n" for name, digest in sorted(entries.items()))
    return {"files": entries,
            "file_count": len(entries),
            "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
            "battery": str(battery),
            "battery_sha256": _sha256(battery),
            "battery_imports": battery_imports(battery, package),
            "scope": [glob, *SUBJECT_EXTRA]}


#: THE EXAMPLES DENOMINATOR: the 2026-08-09 corpus campaign's four example legs, stock
#: first. A script accepted in more than one record is taken from the first that
#: accepts it; a record's non-accepted rows are never consulted, so a stock-leg row
#: that failed for want of ``gdspy`` cannot shadow the ``_gdsii`` leg's accepted lift
#: of the same script. The two ``_native_*`` records of the same campaign re-lift two
#: stock-leg scripts under a different capture and add no script, so they are not
#: listed. Each row is run under the interpreter its record names (``python``), which
#: is what made the sibling legs' lifts possible in the first place; a row from the
#: stock record runs under this process's own interpreter, as every cut before
#: 2026-09-03 did.
EXAMPLES_LIFTS = (
    "parity/meep_gpu/results/corpus_final_numpy_2026-08-09_monitorfix/lift.jsonl",
    "parity/meep_gpu/results/corpus_final_numpy_2026-08-09_monitorfix_blocked/lift.jsonl",
    "parity/meep_gpu/results/corpus_final_numpy_2026-08-09_monitorfix_gdsii/lift.jsonl",
    "parity/meep_gpu/results/corpus_final_numpy_2026-08-09_monitorfix_sigma/lift.jsonl",
)
TESTS_LIFT = ("parity/meep_gpu/results/meep_python_tests_2026-08-09_monitorfix/"
              "lift.jsonl")
#: The MEEP corpus this census walks. HARDCODED UNTIL 2026-08-27, which made the census
#: runnable on exactly one machine and one account -- and on macOS the path sits under
#: ~/Documents, so a TCC prompt nobody answered turned the whole chain into
#: "PermissionError: Operation not permitted" on two of its three legs. Overridable now,
#: because the census is host-side (the Metal coverage predicates import and evaluate
#: with no MPS device present, measured on the GPU host) and so may legitimately be cut
#: wherever a corpus lives.
_CORPUS_ROOT = os.environ.get("MEEP_GPU_CORPUS_ROOT")
EXAMPLES_DIR = (os.path.join(_CORPUS_ROOT, "examples") if _CORPUS_ROOT
                else os.path.join(os.environ.get("MGPU_SITE_MEEP_SOURCE", os.path.join(os.path.expanduser("~"), "meep")), "python", "examples"))
TESTS_DIR = (os.path.join(_CORPUS_ROOT, "tests") if _CORPUS_ROOT
             else os.path.join(os.environ.get("MGPU_SITE_MEEP_SOURCE", os.path.join(os.path.expanduser("~"), "meep")), "python", "tests"))

#: The certified expansion probe. A MEASURED device fact from the recut job that
#: certified both the complex tranche (2343/2345) and, through its extended pattern,
#: the special_kz tranche. Passed in so the probe clause is satisfied by the same
#: artifact the gate ran under rather than factored out as unmeasurable.
#:
#: THIS DEFAULT IS THE ARTIFACT THE 2026-08-16 ROUND *REJECTED*, and it is kept as
#: the default only because two censuses have already been cut against it and must
#: stay reproducible. ``score_probe_artifacts.py`` in
#: ``results/predicate_coverage_2026-08-16_wired_convention/`` put five candidates
#: through the shipped ``complex_fields.expansion_license``; this one is refused
#: because its ``subnormal_policy`` block states a policy but no RESOLVED policy,
#: and "an artifact from before artifacts stated their policy certifies nothing".
#: Cutting with it does not fail -- it silently costs 153 of 759 slots, and the run
#: still looks like a full census. MEASURED: the promoted driver at this default
#: reproduces ``predicate_coverage_2026-08-16_wired`` exactly (563/759, 508, 482,
#: 41 rows admitted nowhere) rather than the licensed ``_wired_convention`` cut's
#: 716/759. PASS ``--probe-artifact`` to choose the licensed one.
PROBE = ("parity/meep_gpu/results/complex_signed_zero_2026-08-12/results/run_2345/"
         "probe.json")

#: THE LICENSED CUT NEEDS A DECLARED POLICY AS WELL AS THE RIGHT ARTIFACT, and the
#: two are one decision rather than two. ``complex_fields._policy_in_force`` reads
#: an INSTALLED policy first and a DECLARED one second, and answers
#: ``UNREADABLE_POLICY`` with neither. This harness installs nothing -- it steps no
#: field and compiles no kernel -- so without a declaration every complex-family
#: predicate refuses BY NAME and the run measures the absence of a declaration
#: rather than the tree's coverage. Declared, never installed: a declaration is the
#: honest shape of a coverage number CONDITIONAL on a run under that policy
#: consuming an artifact cut under the same one, which is the configuration every
#: device gate in this campaign ran under. ``None`` declares nothing, which is what
#: every cut before 2026-08-28 did.
DEFAULT_RUN_POLICY = None


def log(path: Path, message: str) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def load_probe(api_root: Path) -> dict:
    return json.loads((api_root / PROBE).read_text(encoding="utf-8"))


def _policy_argv(args) -> tuple:
    """``--run-policy`` for a child, or nothing. Nothing is not the same as ``None``:
    passing the string "None" would DECLARE a policy named None."""
    return ("--run-policy", args.run_policy) if args.run_policy else ()


def _evaluate(battery: str, driver, probe, probe_path: str, run_policy):
    """The battery's ``evaluate``, under a DECLARED run policy when one is asked for.

    The declaration and the artifact are both stamped into the row, so a reader can
    see what the verdicts are conditional on instead of having to find the harness.
    """
    evaluate = _load_battery(battery)
    if not run_policy:
        result = evaluate(driver, probe)
    else:
        from meep_gpu.expansion_refusal import declaring_run_policy  # noqa: PLC0415
        with declaring_run_policy(run_policy):
            result = evaluate(driver, probe)
    result["run_policy_declared"] = run_policy
    result["probe_artifact"] = probe_path
    return result


# --- children ------------------------------------------------------------------------


def child_examples(script: str, out_json: str, probe_path: str, progress: str,
                   battery: str, run_policy) -> None:
    from parity.meep_gpu.sweep_corpus_lift_parity import capture_simulation

    def mark(message: str) -> None:
        with open(progress, "a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%H:%M:%S')}] {Path(script).name:<38} "
                         f"{message}\n")
            handle.flush()

    probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    record, sim, restore = capture_simulation(script)
    record["row"] = Path(script).name
    mark(f"captured has_sim={record.get('has_simulation')}")
    if sim is None:
        record["measured"] = False
        record["note"] = "no mp.Simulation to lift"
        Path(out_json).write_text(json.dumps(record), encoding="utf-8")
        return
    restore()
    import meep_gpu

    started = time.time()
    try:
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
    except BaseException as exc:  # noqa: BLE001
        record["measured"] = False
        record["lift_error"] = f"{type(exc).__name__}: {exc}"[:600]
        Path(out_json).write_text(json.dumps(record), encoding="utf-8")
        mark(f"LIFT FAILED {type(exc).__name__}")
        return
    record["lift_s"] = round(time.time() - started, 2)
    record["grid_shape"] = [int(v) for v in driver.shape]
    record["grid_cells"] = int(math.prod(int(v) for v in driver.shape))
    try:
        record.update(_evaluate(battery, driver, probe, probe_path, run_policy))
        record["measured"] = True
    except BaseException as exc:  # noqa: BLE001
        record["measured"] = False
        record["battery_error"] = f"{type(exc).__name__}: {exc}"[:600]
    finally:
        try:
            driver.close()
        except BaseException:  # noqa: BLE001
            pass
    Path(out_json).write_text(json.dumps(record), encoding="utf-8")
    mark(f"measured={record.get('measured')} ({record.get('lift_s')} s lift)")


def child_tests(module_path: str, wanted: list, out_json: str, probe_path: str,
                progress: str, case_timeout: float, battery: str,
                run_policy) -> None:
    """Run EVERY case in the module in enumeration order; measure the wanted ones.

    Order matters: the survey ran whole modules in one process, and several MEEP test
    modules depend on an earlier case having run (test_dump_load writes then reads).
    Running only the wanted subset would be a different experiment.
    """
    from parity.meep_gpu import survey_meep_tests as harness

    probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    namespace = harness.build_child_namespace()
    module = namespace["_import_module"](module_path)
    cases = namespace["_enumerate_cases"](module)
    rows = []
    wanted_set = set(wanted)

    def mark(message: str) -> None:
        with open(progress, "a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%H:%M:%S')}] "
                         f"{Path(module_path).name:<34} {message}\n")
            handle.flush()

    import meep_gpu

    take_all = wanted == ["*"]
    for class_name, method_name in cases:
        case_id = f"{class_name}.{method_name}"
        record, sim, restore, _case = namespace["_run_case"](
            module, module_path, class_name, method_name, case_timeout)
        restore()
        if not take_all and case_id not in wanted_set:
            continue
        record["row"] = case_id
        bound = getattr(getattr(module, class_name, None), method_name, None)
        if getattr(bound, "__parameterized_index__", None) is not None:
            record["parameterized"] = {
                "source": bound.__parameterized_source__,
                "index": bound.__parameterized_index__,
                "args": [repr(value)[:120] for value in bound.__parameterized_args__]}
        if sim is None:
            record["measured"] = False
            record["note"] = "no mp.Simulation to lift on this replay"
            rows.append(record)
            mark(f"{case_id}: NO SIM on replay")
            continue
        started = time.time()
        namespace["_alarm"](case_timeout)
        try:
            driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
        except BaseException as exc:  # noqa: BLE001
            record["measured"] = False
            record["lift_error"] = f"{type(exc).__name__}: {exc}"[:600]
            rows.append(record)
            mark(f"{case_id}: LIFT FAILED {type(exc).__name__}")
            continue
        finally:
            namespace["_cancel_alarm"]()
        record["lift_s"] = round(time.time() - started, 2)
        record["grid_shape"] = [int(v) for v in driver.shape]
        record["grid_cells"] = int(math.prod(int(v) for v in driver.shape))
        try:
            record.update(_evaluate(battery, driver, probe, probe_path, run_policy))
            record["measured"] = True
        except BaseException as exc:  # noqa: BLE001
            record["measured"] = False
            record["battery_error"] = f"{type(exc).__name__}: {exc}"[:600]
        finally:
            try:
                driver.close()
            except BaseException:  # noqa: BLE001
                pass
        rows.append(record)
        mark(f"{case_id}: measured={record.get('measured')} "
             f"({record.get('lift_s')} s lift)")
    Path(out_json).write_text(json.dumps(rows), encoding="utf-8")


# --- parent --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", choices=("examples", "tests", "tests_param"),
                        required=True)
    parser.add_argument("--out", default=str(_HERE))
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--case-timeout", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child-script", default=None)
    parser.add_argument("--child-module", default=None)
    parser.add_argument("--child-cases", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--probe", default=None)
    parser.add_argument("--battery", default=DEFAULT_BATTERY,
                        help="module name of the predicate battery to call: "
                             "predicate_battery (Metal) or "
                             "triton_predicate_battery")
    parser.add_argument("--probe-artifact", default=PROBE,
                        help="repository-relative path to the expansion-probe "
                             "artifact; see PROBE for why the default is the one "
                             "the 2026-08-16 round rejected")
    parser.add_argument("--run-policy", default=DEFAULT_RUN_POLICY,
                        help="subnormal policy to DECLARE for the run (e.g. keep). "
                             "Required to consume a policy-stamped probe artifact")
    parser.add_argument("--progress-log", default=None)
    parser.add_argument("--interpreter", action="append", default=None,
                        metavar="RECORDED=PATH",
                        help="run rows whose lift record names interpreter RECORDED "
                             "under PATH instead. An explicit remap for another host; "
                             "without it a missing recorded interpreter is refused by "
                             "name, never substituted")
    parser.add_argument("--runtime-preflight", action="store_true",
                        help="child mode: print the battery's runtime_reasons() for "
                             "THIS interpreter as one RUNTIME_REASONS_JSON= line and "
                             "exit; see preflight_runtime")
    args = parser.parse_args()
    if args.runtime_preflight:
        print(RUNTIME_PREFLIGHT_PREFIX + json.dumps(runtime_reasons_of(args.battery)),
              flush=True)
        return 0

    if args.child_script:
        child_examples(args.child_script, args.out_json, args.probe,
                       args.progress_log, args.battery, args.run_policy)
        return 0
    if args.child_module:
        child_tests(args.child_module, json.loads(args.child_cases), args.out_json,
                    args.probe, args.progress_log, args.case_timeout, args.battery,
                    args.run_policy)
        return 0

    api_root = _API
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    per_row = out_root / f"per_row_{args.leg}"
    per_row.mkdir(exist_ok=True)
    jsonl = out_root / f"{args.leg}.jsonl"
    log_path = out_root / f"{args.leg}.log"
    progress_log = out_root / f"{args.leg}.progress.log"
    probe_path = api_root / args.probe_artifact
    if not probe_path.exists():
        print(f"probe artifact missing: {probe_path}", flush=True)
        return 2

    if not args.resume:
        for path in (jsonl, log_path, progress_log):
            path.write_text("", encoding="utf-8")
    # Stamped into the leg log because the record on disk otherwise carries no trace
    # of which backend was measured -- the jsonl schema is the same either way.
    log(log_path, f"battery: {args.battery}   corpus: "
                  f"{_CORPUS_ROOT or 'default (~/Documents/meep/python)'}")
    log(log_path, f"probe  : {args.probe_artifact}")
    log(log_path, f"policy : {args.run_policy or 'NONE DECLARED'}")

    # The subject digest is computed ONCE, before any leg runs, and stamped on every
    # row: rows measured against different trees are not one census. The package it
    # covers is the battery's own declaration.
    battery_module = _battery_module(args.battery)
    package = getattr(battery_module, "SUBJECT_PACKAGE", None)
    if not package:
        raise SystemExit(
            f"{args.battery} declares no SUBJECT_PACKAGE, so the subject digest cannot "
            f"name the kernels package it pins. Refusing to cut an unpinned census.")
    subject_before = subject_digest(
        api_root, package, Path(battery_module.__file__).resolve())
    (out_root / f"subject_digests_{args.leg}.json").write_text(
        json.dumps(subject_before, indent=1, sort_keys=True), encoding="utf-8")
    log(log_path, f"subject: {subject_before['file_count']} files (meep_gpu/{package} + "
                  f"engine), manifest_sha256={subject_before['manifest_sha256']}")
    log(log_path, f"battery: {subject_before['battery_sha256']} importing "
                  f"{len(subject_before['battery_imports'])} {package} modules")

    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = str(api_root)
    # The census's own interpreter must be able to evaluate the battery at all; a leg
    # cut where it cannot would be every row refusing for one process-level reason.
    own_runtime = preflight_runtime(sys.executable, args.battery, environment)
    if own_runtime:
        raise SystemExit(
            f"{args.battery} cannot be evaluated under this interpreter "
            f"({sys.executable}): {'; '.join(own_runtime)}. Every row of this leg "
            f"would carry that refusal and read as coverage; refusing to cut.")
    log(log_path, f"runtime: {args.battery} evaluable under {sys.executable}")

    started = time.time()
    if args.leg == "examples":
        remap = dict(item.split("=", 1) for item in (args.interpreter or []))
        chosen: dict = {}          # script -> (interpreter, lift record it came from)
        stock_python = None
        for position, relative in enumerate(EXAMPLES_LIFTS):
            origin = Path(relative).parent.name
            accepted = [row for row in (
                json.loads(line) for line in (api_root / relative).read_text(
                    encoding="utf-8").splitlines() if line.strip())
                if row.get("accepted") is True]
            if position == 0:
                pythons = sorted({row.get("python") for row in accepted})
                if len(pythons) != 1:
                    raise SystemExit(
                        f"{relative} records {pythons} as its interpreter; the stock "
                        f"leg must name exactly one so its rows can run under this "
                        f"process's own")
                stock_python = pythons[0]
            added = 0
            for row in accepted:
                if row["script"] in chosen:
                    continue
                recorded = row.get("python")
                interpreter = (sys.executable if recorded == stock_python
                               else remap.get(recorded, recorded))
                chosen[row["script"]] = (interpreter, origin)
                added += 1
            log(log_path, f"  {origin:<52} {len(accepted):>3} accepted, "
                          f"{added:>2} new scripts")
        names = sorted(chosen)
        # Each sibling interpreter is asked ONCE whether it can evaluate the battery;
        # its rows are refused by name below if it cannot, and never run.
        preflighted: dict = {}
        for interpreter in sorted({item[0] for item in chosen.values()}):
            if interpreter == sys.executable or not Path(interpreter).exists():
                continue
            preflighted[interpreter] = preflight_runtime(interpreter, args.battery,
                                                         environment)
            log(log_path, f"  runtime under {interpreter}: "
                          + ("evaluable" if not preflighted[interpreter]
                             else "; ".join(preflighted[interpreter])))
        work = out_root / "workdir"
        work.mkdir(exist_ok=True)
        for entry in Path(EXAMPLES_DIR).iterdir():
            if entry.suffix in (".py", ".ipynb"):
                continue
            link = work / entry.name
            if not link.exists():
                try:
                    link.symlink_to(entry)
                except OSError:
                    pass
        log(log_path, f"examples leg: {len(names)} engine-accepted rows across "
                      f"{len(EXAMPLES_LIFTS)} lift records (denominator "
                      f"{Path(EXAMPLES_LIFTS[0]).parent.name}*)")
        for index, name in enumerate(names, start=1):
            interpreter, origin = chosen[name]
            record_path = per_row / f"{Path(name).stem}.json"
            if args.resume and record_path.exists():
                continue
            case_started = time.time()
            if not Path(interpreter).exists():
                # A sibling leg's interpreter is not on this host. Refuse the row by
                # name rather than run it under another interpreter: the record says
                # which one lifted it, and a census that quietly ran it elsewhere
                # would still be full-length and right-shaped.
                record = {"row": name, "measured": False,
                          "note": (f"interpreter missing: {interpreter} (named by "
                                   f"{origin}; pass --interpreter {interpreter}=PATH "
                                   f"to remap)")}
            elif preflighted.get(interpreter):
                # The interpreter exists but cannot evaluate the battery: refused BY
                # NAME, with the reasons, rather than run into a row of unanimous
                # refusals that the board would price as a gap.
                record = runtime_refusal(name, interpreter, origin,
                                         preflighted[interpreter])
            else:
                command = [interpreter, "-u", str(Path(__file__).resolve()),
                           "--leg", "examples",
                           "--child-script", str(Path(EXAMPLES_DIR) / name),
                           "--out-json", str(record_path), "--probe", str(probe_path),
                           "--progress-log", str(progress_log),
                           "--battery", args.battery, *_policy_argv(args)]
                status = "ok"
                try:
                    completed = subprocess.run(command, cwd=str(work), env=environment,
                                               timeout=args.timeout,
                                               stdout=subprocess.DEVNULL,
                                               stderr=subprocess.PIPE, check=False)
                    stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
                except subprocess.TimeoutExpired as expired:
                    status = "timeout"
                    stderr_text = (expired.stderr or b"").decode("utf-8", "replace") \
                        if expired.stderr else ""
                if record_path.exists():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                else:
                    record = {"row": name, "measured": False,
                              "note": ("child died" if status == "ok"
                                       else "child timeout"),
                              "stderr_tail": stderr_text[-1200:]}
            record["leg"] = "examples"
            record["lift_record"] = origin
            record["interpreter"] = interpreter
            record["subject_manifest_sha256"] = subject_before["manifest_sha256"]
            record["battery_sha256"] = subject_before["battery_sha256"]
            with jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            tag = "" if interpreter == sys.executable else f"  [{origin}]"
            log(log_path, f"  {index:>3}/{len(names)} {name:<40} "
                          f"measured={record.get('measured')} "
                          f"({time.time() - case_started:.1f} s){tag}")
    else:
        source = json.loads("[" + ",".join(
            line for line in (api_root / TESTS_LIFT).read_text(
                encoding="utf-8").splitlines() if line.strip()) + "]")
        wanted: dict = {}
        for row in source:
            if row.get("accepted") is True:
                wanted.setdefault(row["module"], []).append(row["case"])
        if args.leg == "tests_param":
            # The six modules whose import needs `parameterized`. Every case in each is
            # run and measured; the analysis matches them to recorded rows by facts.
            shim = _HERE / "shim"
            environment["PYTHONPATH"] = f"{shim}{os.pathsep}{api_root}"
            needs_shim = set()
            for module_name in list(wanted):
                text = (Path(TESTS_DIR) / module_name).read_text(
                    encoding="utf-8", errors="replace")
                if "import parameterized" in text:
                    needs_shim.add(module_name)
            wanted = {name: ["*"] for name in needs_shim}
        modules = sorted(wanted)
        total = sum(len(v) for v in wanted.values())
        log(log_path, f"tests leg: {total} engine-accepted rows across {len(modules)} "
                      f"modules (denominator {TESTS_LIFT})")
        for index, module_name in enumerate(modules, start=1):
            record_path = per_row / f"{Path(module_name).stem}.json"
            if args.resume and record_path.exists():
                continue
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--leg", "tests",
                       "--child-module", str(Path(TESTS_DIR) / module_name),
                       "--child-cases", json.dumps(wanted[module_name]),
                       "--out-json", str(record_path), "--probe", str(probe_path),
                       "--progress-log", str(progress_log),
                       "--case-timeout", str(args.case_timeout),
                       "--battery", args.battery, *_policy_argv(args)]
            case_started = time.time()
            status = "ok"
            try:
                completed = subprocess.run(command, cwd=str(out_root), env=environment,
                                           timeout=args.timeout,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, check=False)
                stderr_text = (completed.stderr or b"").decode("utf-8", "replace")
            except subprocess.TimeoutExpired as expired:
                status = "timeout"
                stderr_text = (expired.stderr or b"").decode("utf-8", "replace") \
                    if expired.stderr else ""
            if record_path.exists():
                rows = json.loads(record_path.read_text(encoding="utf-8"))
            else:
                rows = []
            seen = {row.get("row") for row in rows}
            missing = [case_id for case_id
                       in (wanted[module_name] if wanted[module_name] != ["*"] else [])
                       if case_id not in seen]

            # ONE ABORT MUST NOT COST A WHOLE MODULE. MEEP calls abort() on some
            # errors, which kills the child outright -- so a single bad case took
            # every other case in the same module down with it and they were all
            # recorded as "child died". test_simulation.py lost 21 rows that way,
            # and the fusion board FAILED CLOSED because its cross-check against
            # the standing per-product record could not reconcile 37 against 45.
            # Re-run the survivors ONE CASE PER CHILD, so an abort costs exactly
            # the case that aborted.
            if missing and len(missing) > 1:
                log(log_path, f"      {module_name}: child died with {len(missing)} "
                              f"cases unrecorded - retrying one case per child",
                    )
                recovered = []
                for case_id in list(missing):
                    solo_path = per_row / (f"{Path(module_name).stem}"
                                           f"__{case_id.replace('.', '_')}.json")
                    solo_command = [sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--leg", "tests",
                                    "--child-module", str(Path(TESTS_DIR) / module_name),
                                    "--child-cases", json.dumps([case_id]),
                                    "--out-json", str(solo_path),
                                    "--probe", str(probe_path),
                                    "--progress-log", str(progress_log),
                                    "--case-timeout", str(args.case_timeout),
                                    "--battery", args.battery,
                                    *_policy_argv(args)]
                    try:
                        subprocess.run(solo_command, cwd=str(out_root), env=environment,
                                       timeout=args.timeout, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, check=False)
                    except subprocess.TimeoutExpired:
                        pass
                    if solo_path.exists():
                        solo_rows = json.loads(solo_path.read_text(encoding="utf-8"))
                        rows.extend(solo_rows)
                        recovered.extend(row.get("row") for row in solo_rows)
                missing = [case_id for case_id in missing if case_id not in set(recovered)]
                log(log_path, f"      {module_name}: recovered {len(recovered)}, "
                              f"{len(missing)} still unrecorded")

            for case_id in missing:
                rows.append({"module": module_name, "row": case_id,
                             "measured": False,
                             "note": ("child died" if status == "ok"
                                      else "child timeout"),
                             "stderr_tail": stderr_text[-1200:]})
            with jsonl.open("a", encoding="utf-8") as handle:
                for row in rows:
                    row["leg"] = args.leg
                    row.setdefault("module", module_name)
                    row["subject_manifest_sha256"] = subject_before["manifest_sha256"]
                    row["battery_sha256"] = subject_before["battery_sha256"]
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
            measured = sum(1 for row in rows if row.get("measured"))
            denominator = ("all" if wanted[module_name] == ["*"]
                           else str(len(wanted[module_name])))
            log(log_path, f"  {index:>3}/{len(modules)} {module_name:<34} "
                          f"{measured}/{denominator} measured "
                          f"({time.time() - case_started:.1f} s)")
    log(log_path, f"{args.leg} leg complete in {time.time() - started:.1f} s -> {jsonl}")

    # A leg that measured NOTHING is a broken harness, not an empty corpus, and it
    # must not exit 0: the parent otherwise writes a full-length jsonl of
    # ``measured: false`` rows that has the right row count, the right leg names and
    # a clean rc, and is indistinguishable from a real census until a downstream
    # board divides by its zero. That is precisely how the 2026-08-27 re-cut passed
    # for a full round trip. Report the children's own reason, not a bare count.
    total = 0
    survivors = 0
    reasons = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        if row.get("measured"):
            survivors += 1
        else:
            tail = (row.get("stderr_tail") or "").strip().splitlines()
            reasons[tail[-1][:160] if tail else (row.get("note") or "unmeasured")] = \
                reasons.get(tail[-1][:160] if tail else (row.get("note") or "unmeasured"), 0) + 1
    if total and not survivors:
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1])[:3]:
            log(log_path, f"  {count:>4}x {reason}")
        log(log_path, f"{args.leg} leg measured 0 of {total} rows -- refusing to "
                      "report success for a census that contains no measurements")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
