"""Survey MEEP's own ``python/tests`` suite — gate, lift, parity, and MEEP's assertions.

The examples corpus (``survey_meep_examples.py`` / ``sweep_corpus_lift_parity.py``)
answers "did the converter accept the script, and did the fields match CPU MEEP".
This module answers a strictly stronger question on a different corpus: MEEP's own
regression tests carry **their authors' numerical oracles** — hard-coded expected
values, tolerances, and physics invariants written by the MEEP developers, not by
this package. Reproducing those is a claim about agreeing with MEEP's definition of
correct, not merely with a second run of MEEP.

Four stages, each writing its own JSONL as it goes:

``inventory``
    Static AST pass over every ``test_*.py``: which files construct an
    ``mp.Simulation`` and how many sites, which classes are ``unittest.TestCase``,
    which test methods exist, which third-party imports are needed, which reference
    MPI or a data file, and what assertion calls the file makes. This is the
    **denominator**, measured rather than assumed.

``lift``
    One subprocess per test module. Each test method runs with MEEP's terminal calls
    stubbed, so the ``mp.Simulation`` is captured at the moment the test asks for it
    to run — after every monitor, source change and ``init_sim``. The captured object
    is then handed to ``gpu_compatibility`` (gate verdict, verbatim reasons) and to
    ``lift_simulation`` (lift verdict, verbatim throw).

``parity``
    For each case that lifted: step the lifted driver and CPU MEEP over the SAME
    object for a bounded time and compare complex fields over the whole volume.

``assertions``
    The headline. For cases whose observable this engine can serve, the test's own
    assertion arithmetic is re-run against **this engine's** output: the captured
    ``mp.Simulation`` is stepped on the driver and the simulation's reader methods
    are then bound to the driver's arrays, so ``self.assertAlmostEqual(...)`` inside
    MEEP's test file compares MEEP's published constant against our number. Nothing
    under the MEEP checkout is written to or modified — the interception is entirely
    on the ``mp.Simulation`` instance the test built.

Reused verbatim from the examples harness so the two corpora stay comparable:
``survey_meep_examples._facts`` (the per-simulation fact block),
``survey_meep_examples.tag_reason`` (the refusal histogram key), and
``sweep_corpus_lift_parity``'s ``plan_run`` / ``_timing`` / ``compare_fields`` /
``source_signal_time`` / ``ProgressLog``. There is one definition of "how long is
long enough for the source to radiate" and one definition of "how are two field
volumes compared" in this directory, not several.

Progress reporting: every stage prints one flushed line per case and appends its row to the JSONL
as it lands; children append phase markers to a shared progress log, so a run sitting
inside a slow lift is distinguishable from a hung one by ``tail`` alone.

**Time budgets are recorded, not silent.** A row that no number came out of because a
clock ran out is not the same result as a refusal, and the three clocks are not the
same either: ``cost_class`` on the row says whether the CHILD's per-case alarm fired
(``alarm-limited`` — raise ``--assert-timeout`` and the row is driven), the PARENT
killed the child (``module-ceiling``), or the case was measured to outrun any alarm
this harness would set (``beyond-any-alarm``, from ``COST_LIMITED_CASES``, which
carries the measurement). ``module_ceiling`` derives the parent's ceiling from the
module's case count and the per-case timeout, so raising one raises the other rather
than being cancelled by it.

Usage (from ``the repository root``)::

    python -u -m parity.meep_gpu.survey_meep_tests \\
        --stage inventory --out parity/meep_gpu/results/meep_python_tests_2026-08-06
    ... --stage lift
    ... --stage parity
    ... --stage assertions
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DEFAULT = os.path.join(os.environ.get("MGPU_SITE_MEEP_SOURCE", os.path.join(os.path.expanduser("~"), "meep")), "python", "tests")

# Imports a MEEP test module may need beyond MEEP's required dependencies: third-party
# packages and MEEP's optional modules. A test module that needs one of these is not a
# fact about the engine until it is run in an environment that has it, so the
# inventory names the dependency rather than reporting an import error as a refusal.
# Membership says nothing about the reference environments: both locks hold autograd
# (added on 2026-10-05, so meep.adjoint imports there), pytest, h5py, matplotlib,
# scipy and MPB, and neither holds jax, nlopt, gdspy, gdstk, PyMieScatt or
# parameterized.
OPTIONAL_DEPS = {
    "jax", "autograd", "nlopt", "gdspy", "gdstk", "PyMieScatt", "parameterized",
    "meep.adjoint", "meep.mpb", "mpb", "pytest", "h5py", "matplotlib", "scipy",
}

# Names that mean the module drives MPI rather than a single rank.
MPI_MARKERS = (
    "divide_parallel_processes", "count_processors", "merge_subgroup_data",
    "am_master", "my_rank", "begin_global_communications",
)


# --- stage: inventory (static) -------------------------------------------------------


def _dotted(node: ast.AST) -> str:
    """``mp.Simulation`` / ``meep.adjoint.OptimizationProblem`` as a dotted string."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def inventory_one(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    row: dict = {"module": path.name, "n_lines": source.count("\n") + 1}

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
            if node.module.startswith("meep"):
                imports.add(node.module)
    row["imports"] = sorted(imports)
    row["optional_deps"] = sorted(imports & OPTIONAL_DEPS)

    sim_sites, calls, assertions = 0, [], {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        calls.append(name)
        if name in ("mp.Simulation", "meep.Simulation", "mp.simulation.Simulation"):
            sim_sites += 1
        if name.split(".")[-1].startswith("assert"):
            key = name.split(".")[-1]
            assertions[key] = assertions.get(key, 0) + 1
    row["simulation_sites"] = sim_sites
    row["assertions"] = dict(sorted(assertions.items(), key=lambda kv: -kv[1]))
    row["n_assertion_calls"] = sum(assertions.values())

    classes, methods = [], []
    subclasses_simulation: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [_dotted(base) for base in node.bases]
        if any(base in ("mp.Simulation", "meep.Simulation") for base in bases):
            # test_chunk_balancer's MockSimulation. A file with no literal
            # `mp.Simulation(` still constructs one through a subclass, and counting
            # only the literal sites is how such a file goes missing from the
            # denominator.
            subclasses_simulation.append(node.name)
        is_case = any(
            base.endswith("TestCase") or base.endswith("ApproxComparisonTestCase")
            or base in ("Test", "TestCase")
            for base in bases
        )
        entry = {"name": node.name, "bases": bases, "is_testcase": is_case,
                 "methods": []}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"):
                entry["methods"].append(item.name)
                if is_case:
                    methods.append(f"{node.name}.{item.name}")
        classes.append(entry)
    row["classes"] = classes
    row["testcase_classes"] = [c["name"] for c in classes if c["is_testcase"]]
    row["test_methods"] = methods
    row["n_test_methods"] = len(methods)
    row["simulation_subclasses"] = subclasses_simulation
    row["constructs_simulation"] = bool(sim_sites) or bool(subclasses_simulation)
    # Classes whose base is another class defined in the same file (MEEP's tests use
    # this: e.g. a base class holding the setup, subclasses supplying the parameters).
    local = {c["name"] for c in classes}
    row["subclass_of_local"] = [c["name"] for c in classes
                                if not c["is_testcase"] and any(b in local for b in c["bases"])]

    row["mpi_markers"] = sorted({m for m in MPI_MARKERS if m in source})
    row["uses_data_dir"] = ("data/" in source) or ("data_dir" in source)
    row["uses_mpb"] = "mpb" in imports or "meep.mpb" in row["imports"]
    row["uses_adjoint"] = any(i.startswith("meep.adjoint") for i in row["imports"]) or "mpa" in source
    return row


def stage_inventory(tests: Path, out_root: Path, say) -> None:
    rows = []
    jsonl = out_root / "inventory.jsonl"
    jsonl.write_text("", encoding="utf-8")
    files = sorted(tests.glob("*.py"))
    say(f"inventory: {len(files)} python files under {tests}")
    for index, path in enumerate(files, start=1):
        try:
            row = inventory_one(path)
        except BaseException as exc:  # noqa: BLE001
            row = {"module": path.name, "parse_error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        say(f"  {index:>2}/{len(files)} {path.name:<38} sims={row.get('simulation_sites', '?'):>3} "
            f"methods={row.get('n_test_methods', '?'):>3} asserts={row.get('n_assertion_calls', '?'):>4} "
            f"deps={','.join(row.get('optional_deps') or []) or '-'}")
    with_sim = [r for r in rows if r.get("constructs_simulation")]
    summary = {
        "n_files": len(rows),
        "n_test_modules": sum(1 for r in rows if r["module"].startswith("test_")),
        "n_construct_simulation": len(with_sim),
        "n_test_methods_total": sum(r.get("n_test_methods", 0) for r in rows),
        "n_test_methods_in_sim_modules": sum(r.get("n_test_methods", 0) for r in with_sim),
        "n_simulation_sites_total": sum(r.get("simulation_sites", 0) for r in rows),
        "n_assertion_calls_total": sum(r.get("n_assertion_calls", 0) for r in rows),
        "no_simulation": sorted(r["module"] for r in rows if not r.get("constructs_simulation")),
        "needs_optional_deps": {r["module"]: r["optional_deps"] for r in rows if r.get("optional_deps")},
        "mpi_modules": {r["module"]: r["mpi_markers"] for r in rows if r.get("mpi_markers")},
    }
    (out_root / "inventory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    say("")
    say(f"python files                     : {summary['n_files']}")
    say(f"test_*.py modules                : {summary['n_test_modules']}")
    say(f"construct an mp.Simulation       : {summary['n_construct_simulation']}")
    say(f"mp.Simulation( sites (all files) : {summary['n_simulation_sites_total']}")
    say(f"test methods (all files)         : {summary['n_test_methods_total']}")
    say(f"test methods in sim modules      : {summary['n_test_methods_in_sim_modules']}")
    say(f"assertion calls (all files)      : {summary['n_assertion_calls_total']}")


# --- child: run one test method with the Simulation captured -------------------------


CHILD_SOURCE = r'''
import json, os, signal, sys, time, types, unittest

os.environ.setdefault("MPLBACKEND", "Agg")


class _StopTest(Exception):
    "Raised in place of the test's first terminal MEEP call, to capture the Simulation."


class _CaseTimeout(Exception):
    pass


# Every route into MEEP's own stepper or grid build, exactly the set the examples
# harness stubs. Capturing at the FIRST of them means monitors added between
# construction and run are already attached, which is what the test hands the engine.
_TERMINAL = ("run", "init_sim", "solve_cw", "run_k_points", "_run_until",
             "get_eigenmode_coefficients", "load_minus_flux")


def _patch(mp, state):
    originals = []
    original_init = mp.Simulation.__init__
    originals.append((mp.Simulation, "__init__", original_init))

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        state["constructed"].append(self)

    mp.Simulation.__init__ = patched_init

    def make_stub(name):
        def stub(self, *args, **kwargs):
            state["captured"] = self
            state["capture_point"] = name
            state["capture_args"] = _describe_call(args, kwargs)
            raise _StopTest(name)
        return stub

    for name in _TERMINAL:
        if hasattr(mp.Simulation, name):
            originals.append((mp.Simulation, name, getattr(mp.Simulation, name)))
            setattr(mp.Simulation, name, make_stub(name))

    original_set_boundary = mp.Simulation.set_boundary
    originals.append((mp.Simulation, "set_boundary", original_set_boundary))

    def patched_set_boundary(self, side, direction, condition):
        state["set_boundary"].append((int(side), int(direction), int(condition)))
        state["captured"] = self
        state["capture_point"] = "set_boundary"
        raise _StopTest("set_boundary")

    mp.Simulation.set_boundary = patched_set_boundary
    try:
        from meep import mpb
        originals.append((mpb.ModeSolver, "init_params", mpb.ModeSolver.init_params))
        mpb.ModeSolver.init_params = make_stub("mpb.init_params")
    except Exception:
        pass

    def restore():
        for owner, attribute, value in reversed(originals):
            setattr(owner, attribute, value)

    return restore


def _describe_call(args, kwargs):
    out = {"n_args": len(args), "arg_types": [type(a).__name__ for a in args][:8]}
    for key in ("until", "until_after_sources"):
        if key in kwargs:
            try:
                out[key] = float(kwargs[key])
            except Exception:
                out[key] = repr(kwargs[key])[:80]
    return out


def _alarm(seconds):
    def handler(signum, frame):
        raise _CaseTimeout(f"case exceeded {seconds} s")
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
    except Exception:
        pass


def _cancel_alarm():
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    except Exception:
        pass


def _import_module(module_path):
    """Import the test module by path, with its own directory importable.

    MEEP's tests import siblings (``utils``, ``binary_partition_utils``) and read
    fixtures from ``data/`` beside them, so the directory goes on ``sys.path``.
    """
    import importlib.util
    directory = os.path.dirname(os.path.abspath(module_path))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _enumerate_cases(module):
    """Every runnable test in the module, as unittest itself would collect them.

    Deliberately dynamic rather than read off the source: MEEP decorates ten
    modules with ``parameterized.expand``, which replaces one source-level method
    with N generated ones. A static list of ``def test_*`` names would name methods
    that do not exist at runtime and miss every case that does.
    """
    import unittest
    loader = unittest.TestLoader()
    out = []
    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type) or not issubclass(obj, unittest.TestCase):
            continue
        if obj.__module__ != module.__name__:  # imported mixins (utils.ApproxComparisonTestCase)
            continue
        for method in loader.getTestCaseNames(obj):
            out.append((name, method))
    return out


def _run_case(module, module_path, class_name, method_name, timeout_s):
    """setUpClass + setUp + the test body, with the Simulation captured mid-flight.

    Returns ``(record, sim, restore, case)``. ``restore()`` puts MEEP's stubbed
    methods back and MUST be called by anything that intends to step the captured
    object.
    """
    import meep as mp

    state = {"constructed": [], "captured": None, "capture_point": None,
             "capture_args": None, "set_boundary": []}
    restore = _patch(mp, state)
    record = {
        "module": os.path.basename(module_path),
        "case": f"{class_name}.{method_name}",
        "python": sys.executable,
        "meep_version": str(getattr(mp, "__version__", None)),
        "single_precision": bool(getattr(mp, "is_single_precision", lambda: None)()),
    }
    started = time.time()
    _alarm(timeout_s)
    case = None
    try:
        cls = getattr(module, class_name)
        cls.setUpClass()
        case = cls(method_name)
        case.setUp()
        getattr(case, method_name)()
        record["outcome"] = "ran_to_end"
    except _StopTest as exc:
        record["outcome"] = "captured"
        record["capture_point"] = str(exc)
    except _CaseTimeout as exc:
        record["outcome"] = "case_timeout"
        record["error"] = str(exc)
    except unittest.SkipTest as exc:
        record["outcome"] = "skipped"
        record["error"] = str(exc)[:300]
    except SystemExit as exc:
        record["outcome"] = "sysexit"
        record["error"] = repr(exc)
    except BaseException as exc:
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        _cancel_alarm()
    record["setup_s"] = round(time.time() - started, 2)
    record["n_simulations_constructed"] = len(state["constructed"])
    record["set_boundary_calls"] = state["set_boundary"]
    record["capture_args"] = state["capture_args"]

    sim = state["captured"]
    if sim is None and state["constructed"]:
        sim = state["constructed"][-1]
        record["capture_point"] = "constructed_only"
    record["has_simulation"] = sim is not None
    if sim is None:
        return record, None, restore, case
    if record["outcome"] != "captured":
        record["outcome"] = record["outcome"] + "+simulation"
    record["facts"] = _facts(mp, sim)
    return record, sim, restore, case
'''


def child_module_source() -> str:
    """The child preamble: this module's capture plus the examples harness's ``_facts``.

    ``_facts`` is lifted verbatim out of ``survey_meep_examples.CHILD_PREAMBLE`` rather
    than re-written, so a fact block from a test row and one from an example row are
    the same block and can be read side by side.
    """
    from parity.meep_gpu.survey_meep_examples import CHILD_PREAMBLE

    marker = "def _facts(mp, sim):"
    facts = CHILD_PREAMBLE[CHILD_PREAMBLE.index(marker):]
    return CHILD_SOURCE + "\n\n" + facts


def build_child_namespace() -> dict:
    namespace: dict = {}
    exec(compile(child_module_source(), "<meep-tests-child>", "exec"), namespace)  # noqa: S102
    return namespace


# --- child stages --------------------------------------------------------------------


def child_lift(namespace: dict, module, module_path: str, case: tuple, timeout_s: float,
               lift_cell_cap: int, progress) -> dict:
    record, sim, restore, _case = namespace["_run_case"](
        module, module_path, case[0], case[1], timeout_s)
    # UNCONDITIONALLY, and before anything else touches MEEP. Every case in a module
    # runs in one process, so a path that returns without restoring leaves
    # `Simulation.init_sim` a stub for every LATER case — which then fails its lift
    # with the harness's own _StopTest and is scored "accepted but does not lift".
    # That is the defect class this survey exists to find, so the harness must not
    # be able to manufacture it.
    #
    # THE COUNT, and the convention it counts. `summarize()` below prints
    # "accepted by the gate but does NOT lift" over rows where `accepted is True and
    # lifts is False`; the same predicate written `lifts is not True` admits one more
    # row, TestLDOS.test_ldos_3D, which has no lift result at all (5 359 375 estimated
    # cells, above --lift-cell-cap, so the lift was never attempted). Every number
    # here is the strict `lifts is False` form.
    #
    # Measured, same corpus, same gate, two lift runs recorded side by side under
    # results/meep_python_tests_2026-08-06/:
    #   lift_before_restore_fix.stdout.log:106   40 rows
    #   lift.log:106 (after this restore())      11 rows
    # so 29 of the 40 were this harness manufacturing the verdict, and all 29 carried
    # `_StopTest: init_sim` verbatim. The 11 that remained are real refusals (mirror
    # symmetry with a transverse k_point, odd-source-about-even-mirror, a reversed
    # near-field bound, a non-flat flux plane); engine work since has taken them to 7
    # in results/meep_python_tests_2026-08-07/lift.log:108. Do not re-derive these
    # from a later run — quote the two logs, which differ only by this call.
    restore()
    record["stage"] = "lift"
    progress(f"{record['case']}: captured outcome={record.get('outcome')} "
             f"has_sim={record.get('has_simulation')}")
    if sim is None:
        record["accepted"] = None
        record["lifts"] = None
        record["lift_note"] = "the test method never built an mp.Simulation to lift"
        return record
    import meep_gpu

    try:
        verdict = meep_gpu.gpu_compatibility(sim)
        record["accepted"] = bool(verdict.supported)
        record["accept_reasons"] = list(verdict.reasons)
    except BaseException as exc:  # noqa: BLE001
        record["accepted"] = None
        record["accept_error"] = f"{type(exc).__name__}: {exc}"[:600]
    progress(f"{record['case']}: gate accepted={record.get('accepted')}")

    # THE VERDICT AS WRITTEN IS RECORDED ABOVE AND NEVER OVERWRITTEN. What follows is
    # the one substitution the assertions stage performs — reading an `amp_func_file`
    # source's HDF5 dataset here and handing it over as the `amp_data` array MEEP's own
    # file route builds from it (sources.cpp:421-460). The package refuses the file
    # spelling for a packaging reason (no HDF5 dependency) and lifts the array one, so
    # without this the lift stage would report a case the assertions stage can drive as
    # unliftable, and the row would never reach that stage at all. Both facts stay on
    # the row: `accept_reasons` is the gate's answer to the test as the test is
    # written, `amp_func_file_substituted` says the rest of this row is not.
    if record.get("accepted") is False:
        substituted: list[str] = []
        try:
            import meep as mp

            from parity.meep_gpu.drive_meep_test_assertions import substitute_amp_func_file

            substitute_amp_func_file(mp, sim, substituted)
        except BaseException as exc:  # noqa: BLE001 — the row keeps the original refusal.
            substituted = []
            record["amp_func_file_substitution_error"] = f"{type(exc).__name__}: {exc}"[:400]
        if substituted:
            record["amp_func_file_substituted"] = substituted
            try:
                verdict = meep_gpu.gpu_compatibility(sim)
                record["accepted_after_substitution"] = bool(verdict.supported)
                record["accept_reasons_after_substitution"] = list(verdict.reasons)
            except BaseException as exc:  # noqa: BLE001
                record["accepted_after_substitution"] = None
                record["accept_error"] = f"{type(exc).__name__}: {exc}"[:600]
            progress(f"{record['case']}: amp_func_file read here; gate after substitution "
                     f"accepted={record.get('accepted_after_substitution')}")
            if record.get("accepted_after_substitution") is not True:
                return record

    facts = record.get("facts") or {}
    cells = 1
    for extent in (facts.get("cell_size") or []):
        if extent:
            cells *= max(1, int(round(abs(extent) * float(facts.get("resolution") or 1))))
    record["cells_estimated"] = cells
    if cells > lift_cell_cap:
        record["lifts"] = None
        record["lift_note"] = (f"not attempted: an estimated {cells} cells exceeds the "
                               f"{lift_cell_cap}-cell lift cap for this survey")
        return record

    started = time.time()
    namespace["_alarm"](timeout_s)
    try:
        driver = meep_gpu.lift_simulation(sim, prefer_gpu=False)
        record["lifts"] = True
        record["grid_shape"] = [int(value) for value in driver.shape]
        record["grid_cells"] = int(math.prod(int(value) for value in driver.shape))
        record["sigma_lift"] = repr(getattr(driver, "sigma_lift", None))[:200]
        driver.close()
    except BaseException as exc:  # noqa: BLE001 - a failed lift is the headline result.
        record["lifts"] = False
        record["lift_error_type"] = type(exc).__name__
        record["lift_error"] = str(exc)[:1500]
    finally:
        namespace["_cancel_alarm"]()
    record["lift_s"] = round(time.time() - started, 2)
    progress(f"{record['case']}: lift lifts={record.get('lifts')} ({record['lift_s']} s)")
    return record


def child_parity(namespace: dict, module, module_path: str, case: tuple, timeout_s: float, *,
                 cell_cap: int, steps: int, res_floor: float, max_steps: int,
                 progress, force_resolution: float | None = None) -> dict:
    from parity.meep_gpu.sweep_corpus_lift_parity import compare_fields, plan_run, _timing

    record, sim, restore, _case = namespace["_run_case"](
        module, module_path, case[0], case[1], timeout_s)
    restore()  # Unconditional; see the note in child_lift.
    record["stage"] = "parity"
    if sim is None:
        record["parity"] = "NO-SIMULATION"
        record["parity_rel_l2"] = None
        return record
    import meep as mp
    import numpy
    import meep_gpu

    try:
        mp.verbosity(0)
    except Exception:
        pass

    # THE SAME ONE SUBSTITUTION the lift and assertions stages make, applied here for
    # the same reason: `amp_func_file` names an HDF5 dataset, the package carries no
    # HDF5 dependency and refuses that spelling, and MEEP's own file route reads the
    # dataset and then calls the ARRAY overload (src/sources.cpp:421-460). Reading it
    # here hands both sides the identical array — the driver lifts it through MEEP's
    # transcribed interpolator, and the CPU control below runs the SAME mutated
    # `sim`, so the comparison is array-vs-array rather than array-vs-file.
    #
    # Without this, child_lift and drive_meep_test_assertions substituted and this
    # stage did not, so test_source.py::TestAmpFileFunc.test_amp_file_func lifted,
    # reproduced 2/2 of MEEP's own assertions, and recorded LIFT-OR-STEP-FAILED at
    # parity with the refusal the other two stages had already stepped past — the
    # whole of the 116-lifts / 115-parity-measured gap.
    substituted: list[str] = []
    try:
        from parity.meep_gpu.drive_meep_test_assertions import substitute_amp_func_file

        substitute_amp_func_file(mp, sim, substituted)
    except BaseException as exc:  # noqa: BLE001 — the row then carries the refusal below.
        substituted = []
        record["amp_func_file_substitution_error"] = f"{type(exc).__name__}: {exc}"[:400]
    if substituted:
        record["amp_func_file_substituted"] = substituted
        progress(f"{record['case']}: amp_func_file read here; both sides get the array")

    facts = record.get("facts") or {}
    plan = plan_run(mp, sim, facts, cell_cap, steps, res_floor, force_resolution, max_steps)
    record["capped"] = plan
    if plan["too_expensive"]:
        record["parity"] = "TOO-EXPENSIVE"
        record["parity_rel_l2"] = None
        return record
    already_initialized = getattr(sim, "fields", None) is not None
    plan["already_initialized"] = already_initialized
    if plan.get("resolution_capped") and not already_initialized:
        sim.resolution = plan["resolution_used"]
    elif plan.get("resolution_capped"):
        plan["resolution_used"] = plan["resolution_original"]
        plan["resolution_capped"] = False
        plan["cap_not_applied"] = "the test had already initialized MEEP's grid"
        plan.update(_timing(mp, sim, float(plan["resolution_original"]), steps, max_steps))
    until = float(plan["until"])
    cylindrical = (facts.get("dimensions_attr") == -2) or bool(facts.get("is_cylindrical"))
    record["cylindrical"] = cylindrical
    progress(f"{record['case']}: plan res {plan['resolution_original']} -> {plan['resolution_used']}, "
             f"~{plan['cells_estimated_used']} cells, until={until:.4g}, "
             f"signal_reached={plan['signal_reached']}")

    started = time.time()
    namespace["_alarm"](timeout_s)
    try:
        result = meep_gpu.run_on_gpu(sim, until=until, prefer_gpu=False)
    except BaseException as exc:  # noqa: BLE001
        record["parity"] = "LIFT-OR-STEP-FAILED"
        record["parity_rel_l2"] = None
        record["parity_error_type"] = type(exc).__name__
        record["parity_error"] = str(exc)[:1500]
        record["lifted_step_s"] = round(time.time() - started, 2)
        return record
    finally:
        namespace["_cancel_alarm"]()
    record["lifted_step_s"] = round(time.time() - started, 2)
    record["driver_steps"] = int(result.steps)
    record["grid_shape"] = [int(value) for value in result.driver.shape]
    record["grid_cells"] = int(math.prod(int(value) for value in result.driver.shape))

    started = time.time()
    namespace["_alarm"](timeout_s)
    try:
        sim.run(until=until)
    except BaseException as exc:  # noqa: BLE001
        record["parity"] = "CPU-MEEP-FAILED"
        record["parity_rel_l2"] = None
        record["parity_error_type"] = type(exc).__name__
        record["parity_error"] = str(exc)[:1500]
        result.close()
        return record
    finally:
        namespace["_cancel_alarm"]()
    record["cpu_meep_s"] = round(time.time() - started, 2)
    record["cpu_meep_steps"] = int(sim.fields.t)
    record["time_matched"] = bool(record["cpu_meep_steps"] == record["driver_steps"])
    try:
        record.update(compare_fields(mp, numpy, result, sim, cylindrical))
        record["parity"] = "MEASURED" if record.get("parity_rel_l2") is not None else "NO-COMPARABLE-FIELD"
    except BaseException as exc:  # noqa: BLE001
        record["parity"] = "COMPARE-FAILED"
        record["parity_rel_l2"] = None
        record["parity_error_type"] = type(exc).__name__
        record["parity_error"] = str(exc)[:1500]
    finally:
        result.close()
    progress(f"{record['case']}: parity={record.get('parity')} rel_l2={record.get('parity_rel_l2')}")
    return record


# --- parent --------------------------------------------------------------------------


def load_modules(out_root: Path, only_modules: set[str] | None) -> list[str]:
    """Test modules that construct an ``mp.Simulation``, from the inventory."""
    inventory = out_root / "inventory.jsonl"
    if not inventory.exists():
        raise SystemExit("run --stage inventory first")
    names: list[str] = []
    for line in inventory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("constructs_simulation"):
            continue
        if only_modules and row["module"] not in only_modules:
            continue
        names.append(row["module"])
    return sorted(names)


def inventory_case_methods(out_root: Path) -> dict[str, list[str]]:
    """``module -> [Class.method, ...]`` from the inventory.

    The lift stage lets the child enumerate its own cases, so the parent has no case
    list of its own to budget or filter with; this is that list, statically measured
    by the inventory stage rather than guessed.
    """
    methods: dict[str, list[str]] = {}
    inventory = out_root / "inventory.jsonl"
    if not inventory.exists():
        return methods
    for line in inventory.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            methods[row.get("module", "")] = list(row.get("test_methods") or [])
    return methods


# How many timed passes over ONE case a stage can make, which is what the per-case
# alarm bounds and therefore what the module's wall-clock ceiling has to cover.
# `lift` builds a driver once. `parity` steps the driver, then steps CPU MEEP.
# `assertions` runs driven, then (when a stopping condition blocked the driven pass)
# control, then driven again with the control's stopping time — see
# drive_meep_test_assertions.child_assertions.
PASSES_PER_CASE = {"lift": 1, "parity": 2, "assertions": 3}
# Import MEEP, import the test module, run setUpClass. Measured at 1.7-1.9 s per
# module in results/meep_python_tests_2026-08-07_ceiling_probe/assertions.progress.log
# ("imported, N cases" at +1.8 s); 60 s is that with room for a slow first import.
MODULE_STARTUP_S = 60.0
# Per case, outside the timed passes: capture, gate, plan, JSONL write.
CASE_OVERHEAD_S = 30.0


# Rows whose STOCK-MEEP control pass alone costs more than any alarm this harness
# would set. THE DISTINCTION THIS TABLE EXISTS TO KEEP: a row killed by the per-case
# alarm is alarm-limited — raising --assert-timeout drives it — whereas a row in here
# is not, and a bigger alarm only buys a longer kill. Both were measured; conflating
# them is how a budget question gets mistaken for a capability one.
#
# Alarm-limited, for contrast, and NOT listed here:
#   test_antenna_radiation.py::TestAntennaRadiation.test_pec_ground_plane — killed at
#   the 300 s default (results/meep_python_tests_2026-08-07/assertions.jsonl carries
#   `_CaseTimeout: case exceeded 300.0 s`); at --assert-timeout 1200 it ran to the end
#   in 638.52 s over 145 670 driven steps and reproduced its 1/1
#   (results/meep_python_tests_2026-08-07_ceiling_probe/assertions.jsonl).
COST_LIMITED_CASES: dict[str, dict] = {
    "test_binary_grating.py::TestEigCoeffs.test_binary_grating_oblique_1_10_7": {
        "stage": "assertions",
        "control_s_at_least": 3614.0,
        "measured": (
            "results/meep_python_tests_2026-08-07_ceiling_probe, --assert-timeout 1200 "
            "(module ceiling 3630 s), reference env / MEEP 1.33.0 single: the control "
            "pass started at +15.8 s and the module was still inside it when the parent "
            "killed the child at 3630 s, so stock MEEP alone ran >3614 s without "
            "finishing. The case's own 1200 s alarm never fired — SIGALRM cannot "
            "interrupt the C++ call it was sitting in — which is why the parent ceiling "
            "is what recorded this and why a larger alarm changes nothing."
        ),
        "what_would_reach_it": (
            "a reduced-resolution variant of the case, run as its own measurement and "
            "reported as such; not a bigger time budget"
        ),
    },
}


def module_ceiling(stage: str, n_cases: int, args) -> tuple[float, str]:
    """Wall-clock ceiling for one module's child process, in seconds.

    THE CEILING HAS TO TRACK THE PER-CASE TIMEOUT OR IT CANCELS IT. A module ceiling
    below what its own cases are allowed to spend kills the child mid-batch and loses
    every case after the slow one — measured: test_binary_grating recorded one row and
    then a ``<module tail>`` marker, twice, at both 900 s and 3630 s.

    The previous form, ``max(--module-timeout, 3 * --assert-timeout + 30)``, covered
    ONE case rather than the batch, and at the 300 s default it evaluated to 930 s
    against a 900 s floor — a 30 s change, so raising --assert-timeout to 1200
    genuinely bought time while leaving the default inert. The ceiling is now the
    batch: startup, plus every case's own budget, plus its overhead. It is an upper
    bound on a run that hangs, not a cost — a healthy module exits when its last case
    lands.

    ``--module-timeout`` remains a floor and ``--module-timeout-max`` a hard cap, so a
    long campaign still has a bound it can state in advance.
    """
    per_case = args.assert_timeout if stage == "assertions" else args.case_timeout
    passes = PASSES_PER_CASE.get(stage, 1)
    derived = MODULE_STARTUP_S + max(1, n_cases) * (passes * per_case + CASE_OVERHEAD_S)
    ceiling = max(float(args.module_timeout), derived)
    why = (f"{n_cases} case(s) x {passes} pass(es) x {per_case:.0f} s "
           f"+ {CASE_OVERHEAD_S:.0f} s each + {MODULE_STARTUP_S:.0f} s startup")
    if args.module_timeout_max and ceiling > args.module_timeout_max:
        ceiling = float(args.module_timeout_max)
        why += f", clipped to --module-timeout-max {ceiling:.0f} s"
    return ceiling, why


def cost_limited_row(module: str, case: str, stage: str, entry: dict) -> dict:
    """The record a cost-limited case gets instead of a kill, with its measurement."""
    row = {"module": module, "case": case, "stage": stage,
           "outcome": "not_attempted_cost_limited",
           "cost_class": "beyond-any-alarm", "alarm_limited": False,
           "has_simulation": None, "cost_limited": dict(entry)}
    if stage == "lift":
        row["accepted"] = None
        row["lifts"] = None
    elif stage == "parity":
        row["parity"] = "TOO-EXPENSIVE"
        row["parity_rel_l2"] = None
    else:
        row["assertions_driven"] = False
        row["assertions_total"] = 0
        row["assertions_passed"] = 0
        row["not_driven_reason"] = "cost-limited: stock MEEP's own control pass exceeds any alarm"
    return row


def stamp_cost_class(row: dict, stage: str, args) -> dict:
    """Say on the row whether a time budget ended it, and which budget.

    Three outcomes read alike in a JSONL and mean different things:

    ``alarm-limited``
        the CHILD's per-case alarm fired. ``--assert-timeout`` / ``--case-timeout``
        is the whole of it; raise it and the row is driven.
    ``module-ceiling``
        the PARENT killed the child. The per-case alarm did not fire — either the
        batch outran the ceiling or one case sat in a C++ call SIGALRM cannot reach,
        and the two are told apart by whether earlier rows landed.
    ``beyond-any-alarm``
        measured in ``COST_LIMITED_CASES``: stock MEEP alone costs more than the
        budget, so no alarm reaches it.

    Rows with no time verdict are left unstamped rather than stamped "fine", so the
    field's presence means a budget was involved.
    """
    if "cost_class" in row:
        return row
    budget = args.assert_timeout if stage == "assertions" else args.case_timeout
    if row.get("outcome") == "module_timeout":
        row["cost_class"] = "module-ceiling"
        row["alarm_limited"] = False
        row["note_cost"] = (
            f"the parent killed the child at its module ceiling; the per-case "
            f"{budget:.0f} s alarm did not fire, so this is the batch's budget or a "
            f"call SIGALRM cannot interrupt, not one case's alarm")
        return row
    # A row that produced its result is not budget-ended even if an earlier pass over
    # it was: the assertions stage can lose a driven pass to the alarm, learn the
    # stopping time from a control run, and drive it to the end on the third pass.
    # Stamping that "alarm-limited" would read as a case the budget denied.
    if (row.get("lifts") is True or row.get("parity_rel_l2") is not None
            or row.get("assertions_driven") is True):
        return row
    if row.get("outcome") == "case_timeout" or "_CaseTimeout" in json.dumps(row):
        row["cost_class"] = "alarm-limited"
        row["alarm_limited"] = True
        row["alarm_s"] = float(budget)
        row["note_cost"] = (
            f"a pass over this case hit the {budget:.0f} s per-case alarm; raising it "
            f"drives the row, and the row is not evidence about the engine until then")
    return row


def run_child_module(namespace: dict, module_path: str, stage: str, out_jsonl: Path,
                     args, progress, only_cases: set[str] | None = None) -> None:
    """Child mode: import one test module and run every case in it, row by row."""
    module_name = os.path.basename(module_path)

    def emit(record: dict) -> None:
        with out_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    started = time.time()
    try:
        namespace["_alarm"](args.case_timeout)
        module = namespace["_import_module"](module_path)
        namespace["_cancel_alarm"]()
    except BaseException as exc:  # noqa: BLE001 - an unimportable module is a result.
        namespace["_cancel_alarm"]()
        emit({"module": module_name, "case": "<module import>", "stage": stage,
              "outcome": "module_import_failed", "has_simulation": False,
              "error": f"{type(exc).__name__}: {exc}"[:600],
              "missing_dependency": type(exc).__name__ == "ModuleNotFoundError",
              "wall_s": round(time.time() - started, 2),
              "accepted": None, "lifts": None,
              "parity": None if stage != "parity" else "MODULE-IMPORT-FAILED",
              "parity_rel_l2": None})
        return
    cases = namespace["_enumerate_cases"](module)
    progress(f"{module_name}: imported, {len(cases)} cases")
    for class_name, method_name in cases:
        name = f"{class_name}.{method_name}"
        if only_cases is not None and name not in only_cases:
            continue
        case_started = time.time()
        try:
            if stage == "lift":
                record = child_lift(namespace, module, module_path, (class_name, method_name),
                                    args.case_timeout, args.lift_cell_cap, progress)
            elif stage == "parity":
                record = child_parity(namespace, module, module_path, (class_name, method_name),
                                      args.case_timeout, cell_cap=args.cell_cap,
                                      steps=args.steps, res_floor=args.res_floor,
                                      max_steps=args.max_steps, progress=progress,
                                      force_resolution=args.force_resolution)
            elif stage == "assertions":
                from parity.meep_gpu.drive_meep_test_assertions import child_assertions
                record = child_assertions(namespace, module, module_path,
                                          (class_name, method_name), args, progress)
            else:
                raise SystemExit(f"unknown child stage {stage}")
        except BaseException as exc:  # noqa: BLE001
            record = {"module": module_name, "case": name, "stage": stage,
                      "outcome": "harness_error", "has_simulation": None,
                      "error": f"{type(exc).__name__}: {exc}"[:600]}
            if stage == "lift":
                record["accepted"] = None
                record["lifts"] = None
            elif stage == "parity":
                record["parity"] = "HARNESS-ERROR"
                record["parity_rel_l2"] = None
            else:
                record["assertions_driven"] = None
        record.setdefault("module", module_name)
        record.setdefault("case", name)
        record["wall_s"] = round(time.time() - case_started, 2)
        emit(record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", default=TESTS_DEFAULT)
    parser.add_argument("--out", default="parity/meep_gpu/results/meep_python_tests_2026-08-06")
    parser.add_argument("--stage", choices=("inventory", "lift", "parity", "assertions"),
                        default="inventory")
    parser.add_argument("--only", default=None, help="comma-separated test module names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--module-timeout", type=float, default=900.0,
                        help="parent: FLOOR under the wall-clock ceiling for one module's whole "
                             "batch. The ceiling itself is derived per module from the case "
                             "count and the per-case timeout (see module_ceiling), so a module "
                             "is never killed below what its own cases are allowed to spend.")
    parser.add_argument("--module-timeout-max", type=float, default=7200.0,
                        help="parent: hard cap on the derived module ceiling, so a campaign has "
                             "a bound it can state in advance. 0 disables the cap.")
    parser.add_argument("--include-cost-limited", action="store_true",
                        help="attempt the cases in COST_LIMITED_CASES, whose stock-MEEP control "
                             "pass was MEASURED to outrun any alarm. Off by default: they are "
                             "recorded with that measurement instead of eating the module's "
                             "ceiling and taking every case after them down with it.")
    parser.add_argument("--case-timeout", type=float, default=180.0,
                        help="child: per-case alarm (pure-Python hangs; C++ calls need the parent)")
    parser.add_argument("--lift-cell-cap", type=int, default=3_000_000,
                        help="lift stage: estimated cells above which the lift is not attempted")
    parser.add_argument("--assert-timeout", type=float, default=300.0,
                        help="assertions stage: ceiling on ONE pass of a test method. The test "
                             "runs at its own resolution and its own stopping time — capping "
                             "either would invalidate the oracle it asserts against — so cost "
                             "is bounded by refusing slow cases, not by shrinking them.")
    parser.add_argument("--cell-cap", type=int, default=120_000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--res-floor", type=float, default=4.0)
    parser.add_argument("--max-steps", type=int, default=12_000)
    parser.add_argument("--force-resolution", type=float, default=None,
                        help="parity stage: run at this resolution, ignoring the cap and floor. "
                             "The convergence axis a short-run parity number must be read "
                             "against before it is called a defect.")
    # Child mode.
    parser.add_argument("--child-module", default=None, help="child mode: one test module")
    parser.add_argument("--child-out", default=None)
    parser.add_argument("--child-cases", default=None,
                        help="child mode: comma-separated Class.method names to run")
    parser.add_argument("--progress-log", default=None)
    args = parser.parse_args()

    tests = Path(args.tests)
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.child_module:
        from parity.meep_gpu.sweep_corpus_lift_parity import ProgressLog

        progress = ProgressLog(args.progress_log, Path(args.child_module).name)
        namespace = build_child_namespace()
        only = ({name.strip() for name in args.child_cases.split(",")}
                if args.child_cases else None)
        run_child_module(namespace, str(tests / args.child_module), args.stage,
                         Path(args.child_out), args, progress, only)
        return 0

    log_path = out_root / f"{args.stage}.log"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")

    def say(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
            handle.flush()

    if args.stage == "inventory":
        stage_inventory(tests, out_root, say)
        return 0

    only = {name.strip() for name in args.only.split(",")} if args.only else None
    modules = load_modules(out_root, only)
    jsonl = out_root / f"{args.stage}.jsonl"
    progress_log = out_root / f"{args.stage}.progress.log"

    # Later stages only visit what the previous stage got a driver from; the case
    # list is therefore read off that stage's rows rather than re-derived.
    per_module_cases: dict[str, set[str] | None] = {name: None for name in modules}
    if args.stage in ("parity", "assertions"):
        lift_path = out_root / "lift.jsonl"
        if not lift_path.exists():
            say("this stage needs lift.jsonl; run --stage lift first")
            return 2
        wanted: dict[str, set[str]] = {}
        for line in lift_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lifts") is True:
                wanted.setdefault(row["module"], set()).add(row["case"])
        modules = [name for name in modules if name in wanted]
        per_module_cases = {name: wanted[name] for name in modules}

    # Cases MEASURED to outrun any alarm are held out of the child's batch and given
    # their measurement as their row (see COST_LIMITED_CASES). Held out rather than
    # attempted-and-killed because the kill is not confined to the case: the parent
    # kills the whole child, so every case queued behind it is lost too — which is
    # exactly how test_binary_grating recorded one row and a `<module tail>` marker.
    inventory_methods = inventory_case_methods(out_root)
    held_out: dict[str, list[tuple[str, dict]]] = {}
    if not args.include_cost_limited:
        for key, entry in COST_LIMITED_CASES.items():
            module_name, _, case_name = key.partition("::")
            if entry.get("stage") not in (None, args.stage) or module_name not in modules:
                continue
            available = per_module_cases.get(module_name)
            if available is None:
                available = set(inventory_methods.get(module_name, []))
                per_module_cases[module_name] = available
            if case_name not in available:
                continue
            available.discard(case_name)
            held_out.setdefault(module_name, []).append((case_name, entry))
            say(f"holding out {module_name}::{case_name} — {entry['control_s_at_least']:.0f} s+ "
                f"for stock MEEP's control pass alone; --include-cost-limited to attempt it")

    done_modules: set[str] = set()
    if args.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_modules.add(json.loads(line)["module"])
        modules = [name for name in modules if name not in done_modules]
    else:
        jsonl.write_text("", encoding="utf-8")
        progress_log.write_text("", encoding="utf-8")

    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["MPLBACKEND"] = "Agg"
    # The harness's `parameterized` stand-in comes FIRST only if the real package is
    # absent; `test_shims` is a leaf directory holding nothing else, so a real
    # install elsewhere on the path still wins by ordering below.
    environment["PYTHONPATH"] = os.pathsep.join([
        str(Path(__file__).resolve().parents[2]),
        str(Path(__file__).resolve().parent / "test_shims"),
    ])
    work = out_root / "workdir"
    work.mkdir(exist_ok=True)
    # MEEP's tests read fixtures out of their own ``data/`` directory by a path
    # relative to the CWD, so the child runs with that directory reachable.
    link = work / "data"
    if not link.exists():
        try:
            link.symlink_to(tests / "data")
        except OSError:
            pass

    say(f"stage {args.stage}: {len(modules)} modules (skipping {len(done_modules)} already recorded)")
    run_started = time.time()
    batch_dir = out_root / "batches"
    batch_dir.mkdir(exist_ok=True)
    for index, module in enumerate(modules, start=1):
        cases_here = per_module_cases.get(module)
        # The lift stage lets the child enumerate, so the count comes from the
        # inventory there; later stages carry an explicit case list.
        n_cases = (len(cases_here) if cases_here is not None
                   else len(inventory_methods.get(module, [])) or 1)
        module_timeout, ceiling_why = module_ceiling(args.stage, n_cases, args)

        # Rows the parent owes without running anything: their measurement IS the row.
        pre_rows = [cost_limited_row(module, case, args.stage, entry)
                    for case, entry in held_out.get(module, [])]
        if cases_here is not None and not cases_here:
            with jsonl.open("a", encoding="utf-8") as handle:
                for row in pre_rows:
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
            say(f"  {index:>2}/{len(modules)} {module:<38} "
                f"{len(pre_rows):>3} cases  all held out (cost-limited)  [  0.0 s]")
            continue

        child_out = batch_dir / f"{module}.{args.stage}.jsonl"
        child_out.write_text("", encoding="utf-8")
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--stage", args.stage, "--tests", str(tests), "--out", str(out_root),
            "--child-module", module, "--child-out", str(child_out),
            "--progress-log", str(progress_log),
            "--case-timeout", str(args.case_timeout),
            # The assertions stage's per-case alarm is read in the CHILD
            # (drive_meep_test_assertions.child_assertions). Omitting it here left
            # every child on argparse's 300 s default however the parent was invoked,
            # so the flag was inert and a long case — test_antenna_radiation's
            # pec_ground_plane, 145 670 driven steps and 638.52 s at --assert-timeout
            # 1200 (results/meep_python_tests_2026-08-07_ceiling_probe) — could not be
            # given more time at all.
            "--assert-timeout", str(args.assert_timeout),
            "--lift-cell-cap", str(args.lift_cell_cap), "--cell-cap", str(args.cell_cap),
            "--steps", str(args.steps), "--res-floor", str(args.res_floor),
            "--max-steps", str(args.max_steps),
        ]
        if args.force_resolution is not None:
            command += ["--force-resolution", str(args.force_resolution)]
        # `is not None`, not truthiness: an empty set means "no case survived the
        # hold-out", and a falsy test there would hand the child the whole module.
        if cases_here is not None:
            command += ["--child-cases", ",".join(sorted(cases_here))]
        case_start = time.time()
        status, stderr_text, returncode = "ok", "", None
        try:
            completed = subprocess.run(
                command, cwd=str(work), env=environment, timeout=module_timeout,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
            stderr_text = (completed.stderr or b"").decode("utf8", "replace")
            returncode = completed.returncode
        except subprocess.TimeoutExpired as expired:
            status = "timeout"
            stderr_text = (expired.stderr or b"").decode("utf8", "replace") if expired.stderr else ""
        rows = []
        if child_out.exists():
            rows = [json.loads(line) for line in child_out.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        if not rows or status == "timeout" or (returncode not in (0, None)):
            # The child died (or ran out of time) with cases still unvisited. That is
            # a result about the harness, recorded as such rather than as a refusal.
            # Which cases were lost is unknown from here — the child enumerates them —
            # so one marker row carries the module.
            row = {"module": module, "case": "<module tail>", "stage": args.stage,
                   "outcome": "module_timeout" if status == "timeout" else "child_died",
                   "returncode": returncode, "stderr_tail": stderr_text[-1500:],
                   "has_simulation": None, "rows_before_death": len(rows),
                   "module_ceiling_s": module_timeout, "module_ceiling_why": ceiling_why,
                   "cases_sent": n_cases}
            if args.stage == "lift":
                row["accepted"] = None
                row["lifts"] = None
            elif args.stage == "parity":
                row["parity"] = "TIMEOUT" if status == "timeout" else "CHILD-DIED"
                row["parity_rel_l2"] = None
            else:
                row["assertions_driven"] = None
            rows.append(row)
        rows = pre_rows + [stamp_cost_class(row, args.stage, args) for row in rows]
        if stderr_text.strip():
            (batch_dir / f"{module}.{args.stage}.stderr.txt").write_text(stderr_text, encoding="utf-8")
        with jsonl.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
        elapsed = time.time() - case_start
        if args.stage == "lift":
            lifts = sum(1 for r in rows if r.get("lifts") is True)
            accepted = sum(1 for r in rows if r.get("accepted") is True)
            nosim = sum(1 for r in rows if r.get("has_simulation") is False)
            verdict = f"{len(rows):>3} cases  accepted={accepted} lifts={lifts} no-sim={nosim}"
        elif args.stage == "parity":
            measured = [r for r in rows if r.get("parity_rel_l2") is not None]
            best = min((r["parity_rel_l2"] for r in measured), default=None)
            verdict = (f"{len(rows):>3} cases  measured={len(measured)}"
                       + (f" best={best:.2e}" if best is not None else ""))
        else:
            driven = [r for r in rows if r.get("assertions_driven")]
            passed = sum(r.get("assertions_passed", 0) for r in rows)
            total = sum(r.get("assertions_total", 0) for r in rows)
            verdict = f"{len(rows):>3} cases  driven={len(driven)}  assertions {passed}/{total}"
        say(f"  {index:>2}/{len(modules)} {module:<38} {verdict}  [{elapsed:5.1f} s]"
            + (f"  ({status}, ceiling {module_timeout:.0f} s = {ceiling_why})"
               if status != "ok" else ""))
    say(f"stage {args.stage} complete in {time.time() - run_started:.1f} s -> {jsonl}")
    summarize(jsonl, args.stage, say)
    return 0


def summarize(jsonl: Path, stage: str, say) -> None:
    from collections import Counter

    from parity.meep_gpu.survey_meep_examples import tag_reason

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    say("")
    say(f"rows (test methods): {len(rows)}")
    say(f"modules            : {len({r.get('module') for r in rows})}")

    # Rows a time budget ended, split by WHICH budget — the distinction between "give
    # it more alarm" and "this needs a cheaper variant", kept on the rows themselves.
    by_cost = Counter(r.get("cost_class") for r in rows if r.get("cost_class"))
    if by_cost:
        say("ended by a time budget :")
        for cost_class, count in by_cost.most_common():
            say(f"    {count:>3}  {cost_class}")
        for row in rows:
            if not row.get("cost_class"):
                continue
            extra = ""
            if row.get("cost_class") == "alarm-limited":
                extra = f"alarm {float(row.get('alarm_s') or 0):.0f} s — raising it drives this row"
            elif row.get("cost_class") == "module-ceiling":
                extra = (f"parent ceiling {row.get('module_ceiling_s')} s, "
                         f"{row.get('rows_before_death')} row(s) landed first")
            elif row.get("cost_class") == "beyond-any-alarm":
                entry = row.get("cost_limited") or {}
                extra = (f"stock MEEP's own control pass >{entry.get('control_s_at_least')} s; "
                         f"{entry.get('what_would_reach_it')}")
            say(f"    {row.get('cost_class'):<18} {row['module']}::{row['case']:<44} {extra}")

    if stage == "lift":
        with_sim = [r for r in rows if r.get("has_simulation")]
        accepted = [r for r in with_sim if r.get("accepted") is True]
        refused = [r for r in with_sim if r.get("accepted") is False]
        lifts = [r for r in rows if r.get("lifts") is True]
        lift_failed = [r for r in rows if r.get("lifts") is False]
        say(f"built an mp.Simulation : {len(with_sim)}")
        say(f"gate ACCEPTED          : {len(accepted)}")
        say(f"gate REFUSED           : {len(refused)}")
        say(f"LIFTS                  : {len(lifts)}")
        say(f"LIFT-FAILED            : {len(lift_failed)}")
        tags: Counter = Counter()
        for row in refused:
            tags.update({tag_reason(reason) for reason in row.get("accept_reasons", [])})
        say("refusal reasons ranked :")
        for tag, count in tags.most_common():
            say(f"    {count:>3}  {tag}")
        gap = [r for r in rows if r.get("accepted") is True and r.get("lifts") is False]
        say(f"accepted by the gate but does NOT lift: {len(gap)}")
        for row in gap:
            say(f"    {row['module']}::{row['case']:<40} {row.get('lift_error_type')}: "
                f"{str(row.get('lift_error'))[:120]}")
        outcomes = Counter(r.get("outcome") for r in rows if not r.get("has_simulation"))
        say("no Simulation captured, by outcome:")
        for outcome, count in outcomes.most_common():
            say(f"    {count:>3}  {outcome}")
    elif stage == "assertions":
        driven = [r for r in rows if r.get("assertions_driven")]
        say(f"cases driven through this engine : {len(driven)}")
        total = sum(r.get("assertions_total", 0) for r in driven)
        passed = sum(r.get("assertions_passed", 0) for r in driven)
        say(f"MEEP-authored assertions run     : {total}")
        say(f"MEEP-authored assertions PASSED  : {passed}")
        # A case can be driven end to end and never step a single field: `solve_cw`
        # runs its own solver, and the constructor-bookkeeping tests never call run()
        # at all. Their assertions hold, but they are not evidence about the STEPPER,
        # so the quotable figure excludes them — the same split `analyze_meep_tests
        # _survey.py` reports as "THE QUOTABLE NUMBER" from rollup's engine_stepped_*.
        stepped = [r for r in driven if (r.get("driven") or {}).get("engine_steps")]
        say(f"  of which actually STEPPED this engine : {len(stepped)} cases, "
            f"{sum(r.get('assertions_passed', 0) for r in stepped)}/"
            f"{sum(r.get('assertions_total', 0) for r in stepped)} assertions")
        for row in driven:
            say(f"    {row['module']}::{row['case']:<40} "
                f"{row.get('assertions_passed')}/{row.get('assertions_total')} "
                f"{row.get('assert_verdict', '')}")
        not_driven = Counter(r.get("not_driven_reason") for r in rows if not r.get("assertions_driven"))
        say("not driven, by reason:")
        for reason, count in not_driven.most_common():
            say(f"    {count:>3}  {reason}")
    else:
        measured = [r for r in rows if r.get("parity_rel_l2") is not None]
        say(f"parity measured : {len(measured)}")
        for row in sorted(measured, key=lambda r: r["parity_rel_l2"]):
            flag = "" if row.get("time_matched") else "  [STEP COUNT MISMATCH]"
            if not (row.get("capped") or {}).get("signal_reached", True):
                flag += "  [signal not reached]"
            say(f"    {row['module']}::{row['case']:<38} {row['parity_rel_l2']:.3e}{flag}")
        for label in ("TOO-EXPENSIVE", "TIMEOUT", "LIFT-OR-STEP-FAILED", "CPU-MEEP-FAILED",
                      "COMPARE-FAILED", "NO-COMPARABLE-FIELD", "CHILD-DIED", "HARNESS-ERROR"):
            hit = [r for r in rows if r.get("parity") == label]
            if hit:
                say(f"{label}: {len(hit)}")
                for row in hit:
                    say(f"    {row['module']}::{row['case']:<38} "
                        f"{str(row.get('parity_error') or '')[:110]}")


if __name__ == "__main__":
    raise SystemExit(main())
