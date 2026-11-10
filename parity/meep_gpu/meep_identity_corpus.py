#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Table 2's corpus as a Tier C cases file: every measured row, rebuilt as it was measured.

    python parity/meep_gpu/meep_identity_corpus.py manifest --results DIR --out FILE
    python parity/meep_gpu/meep_identity_corpus.py list --manifest FILE

    MEEP_IDENTITY_CORPUS_MANIFEST=FILE MGPU_SITE_MEEP_SOURCE=DIR \\
        python parity/meep_gpu/meep_identity_tier_c.py compare \\
            --cases-file parity/meep_gpu/meep_identity_corpus.py --cases LABEL --out OUT

What it is. ``meep_identity_tier_c.py compare --cases-file`` takes a module with
``CASES`` and ``build(mp, label)``; this is that module for the manuscript's Table 2:
the rows of the seven 2026-08-09 ``parity.jsonl`` files (the first seven ``LEGS``: MEEP
1.33.0's ``python/examples``, 60 measured rows, and ``python/tests``, 134), as the MEEP
1.35.0 re-surveys (``RESURVEYS``, below) carry, retire and add to them, each rebuilt the
way its row was measured, so Tier C word-compares the run its leg reported an L2 for:

* the simulation is CAPTURED by the corpus harnesses' own code, not re-written: an
  example through ``sweep_corpus_lift_parity.capture_simulation`` (the survey's
  ``_capture``, the script run to its first terminal MEEP call) in a scratch working
  directory carrying the examples' data files (``survey_meep_examples.scratch_workdir``);
  a test through ``survey_meep_tests``'s ``_run_case`` (setUpClass, setUp and the test
  body, stopped at its first terminal call), with the parity stage's one substitution
  (``drive_meep_test_assertions.substitute_amp_func_file``);
* the row's RESOLUTION is applied when the row was capped (``capped.resolution_capped``),
  exactly as the parity stage applied it, and refused if the captured simulation is
  already initialized;
* the row's STEP COUNT is Table 2's ``driver_steps`` (``steps_planned``), so N and the
  compared schedule (``meep_identity.compared_steps``) are the row's;
* after the lift, ``after_lift`` requires the lifted grid to have the row's
  ``grid_cells``: a different grid is a different row, refused rather than reported.

The rows a leg recorded as anything but ``MEASURED`` are not cases: in Table 2's legs
those are two ``examples/main`` rows (``cavity-farfield.py`` at resolution 11 and
``zone_plate.py`` at 18) that the uniform cap could not measure and that the two
``native`` legs measured at their native resolutions instead. ``manifest`` names them.

RE-SURVEYS. A leg named in ``RESURVEYS`` ran the same harnesses, unchanged, over whole
test modules or example scripts of MEEP 1.35.0's corpus, the release the identity corpus
is certified against. It is read after the other legs, in ``RESURVEYS`` order, and its
``lift.jsonl`` is read beside its ``parity.jsonl``: its verdict on a case is its parity
row when the case reached the parity stage, otherwise its lift row (no simulation built,
the test skipped at that release, the gate's refusal or the lift's error, in the stage's
own words). Then:

* a case it measured that no earlier leg measured is a new row;
* a row an earlier leg measured that it measured again stays the row as first measured,
  and must define the same run (``RUN_KEYS``: resolution, step count, grid, the
  ``amp_func_file`` substitution), or the manifest is refused; ``remeasured`` lists each
  such row with both L2s;
* a row an earlier leg measured that it did not measure leaves the rows and is named in
  ``not_cases`` with its verdict and the leg whose row it was (``retires``);
* every other case it did not measure is named in ``not_cases`` with its verdict.

THE MANIFEST (``manifest``) is derived from the legs' JSONL files, never transcribed: one
entry per row with its label, leg, kind, script or module and case, the plan it ran
under, its leg's numbers (``table2``: ``rel_l2``, ``grid_cells``, ``driver_steps``) and its
fact block, plus the sha256 of every input. ``CASES`` is empty until
``MEEP_IDENTITY_CORPUS_MANIFEST`` names one.

THE SOURCE. ``MGPU_SITE_MEEP_SOURCE`` names the MEEP checkout whose ``python/examples``
and ``python/tests`` the rows were captured from (the corpus harnesses' own variable and
default, ``~/meep``). Optional packages a script or test imports (``PyMieScatt``,
``gdspy``) are the environment's business; a missing one is the case's error, by name,
never a result. ``parameterized`` is the exception, as it was for the survey that named
the rows: its stand-in (``parity/meep_gpu/test_shims``, no numerical content) goes on the
path AFTER the environment's, exactly as ``survey_meep_tests`` places it, so a test
module's expanded methods carry the names the rows were recorded under
(``test_reflectance_angular_1_20_6``; the released package 0.9 names that one
``test_reflectance_angular_1``, and a real install ahead of the stand-in is refused by
the missing method, by name).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MANIFEST_ENVIRONMENT = "MEEP_IDENTITY_CORPUS_MANIFEST"
SOURCE_ENVIRONMENT = "MGPU_SITE_MEEP_SOURCE"
MANIFEST_SCHEMA = "meep_gpu.meep_identity_corpus"
MANIFEST_VERSION = 1
#: The corpus legs, relative to results/: Table 2's seven, then the MEEP 1.35.0
#: re-surveys. A leg whose name starts ``tests`` holds test cases, any other example
#: scripts.
LEGS = {
    "examples/main": "corpus_final_numpy_2026-08-09_monitorfix",
    "examples/blocked": "corpus_final_numpy_2026-08-09_monitorfix_blocked",
    "examples/gdsii": "corpus_final_numpy_2026-08-09_monitorfix_gdsii",
    "examples/sigma": "corpus_final_numpy_2026-08-09_monitorfix_sigma",
    "examples/native_cavity_farfield": "corpus_final_numpy_2026-08-09_monitorfix_native_cavity_farfield",
    "examples/native_zone_plate": "corpus_final_numpy_2026-08-09_monitorfix_native_zone_plate",
    "tests": "meep_python_tests_2026-08-09_monitorfix",
    "tests/meep-1.35.0": "meep_python_tests_2026-10-18_meep135",
    "examples/gdsii/meep-1.35.0": "corpus_final_numpy_2026-10-18_meep135_gdsii",
}
#: The legs read as re-surveys, in this order (see RE-SURVEYS above): the test modules
#: holding the MEEP 1.35.0 candidates, the 3-D MaterialGrid row and the sharded dump/load
#: rows, and the two GDSII examples, each on MEEP 1.35.0.
RESURVEYS = ("tests/meep-1.35.0", "examples/gdsii/meep-1.35.0")
#: The file each leg's measured rows are read from, and the one a re-survey adds.
PARITY_FILE = "parity.jsonl"
LIFT_FILE = "lift.jsonl"
#: The run a row defines, which a re-survey that measures the row again must reproduce:
#: ``(part of the row, key)``.
RUN_KEYS = (("plan", "resolution_used"), ("plan", "resolution_capped"),
            ("table2", "driver_steps"), ("table2", "grid_cells"), ("table2", "grid_shape"),
            ("table2", "amp_func_file_substituted"))
#: The per-case ceiling ``_run_case`` arms around a test's own setup and body.
TEST_SETUP_TIMEOUT_S = 600.0
FACT_KEYS = ("cell_size", "dimensions_attr", "is_cylindrical", "effective_dims", "resolution",
             "courant", "k_point", "n_geometry", "geometry_materials", "default_material",
             "source_kinds", "src_time_kinds", "boundary_kinds", "symmetry_kinds",
             "n_dft_objects", "dft_kinds", "has_material_function", "has_epsilon_func")
PLAN_KEYS = ("resolution_original", "resolution_used", "resolution_capped", "resolution_forced",
             "steps_planned", "signal_reached", "until", "already_initialized", "cap_not_applied")

_CURRENT: Dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holds_tests(leg: str) -> bool:
    """Whether a leg's records are test cases (its name starts ``tests``) or examples."""
    return leg.split("/", 1)[0] == "tests"


def label_of(leg: str, record: Dict[str, Any]) -> str:
    """A row's case label: ``ex.<script stem>`` or ``t.<module stem>.<Class.method>``."""
    if holds_tests(leg):
        return f"t.{Path(record['module']).stem}.{record['case']}"
    return f"ex.{Path(record['script']).stem}"


def _records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _neutral(record: Dict[str, Any]) -> Tuple[Any, Any]:
    """``(python, amp_func_file_substituted)`` of a leg record as the manifest writes them:
    the interpreter under its root's label (``neutral_ids.path``) and the substitution notes
    by ``survey_meep_tests.neutral_amp_notes``. Applied on READ, so a leg recorded before
    the neutral spellings (absolute paths) and a re-survey recorded after give the same
    manifest, and the re-survey's ``RUN_KEYS`` comparison sees one spelling of one file."""
    from parity.meep_gpu.survey_meep_tests import neutral_amp_notes  # noqa: PLC0415

    return (_neutral_ids().path(record.get("python")),
            neutral_amp_notes(record.get("amp_func_file_substituted")))


def _neutral_ids():
    """The record spellings of machine stamps: ``neutral_ids.py`` beside this file."""
    try:
        import neutral_ids  # noqa: PLC0415
    except ImportError:  # imported as a package module, not from parity/meep_gpu
        from parity.meep_gpu import neutral_ids  # noqa: PLC0415
    return neutral_ids


def _neutral_text(why: Any) -> str:
    """A not-case's error as the manifest writes it: neutral (``neutral_ids.text``) before
    it is cut to 300 characters, since a lift or parity error can name a path."""
    return _neutral_ids().text(str(why or ""))[:300]


def _row(leg: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """The manifest row of a ``MEASURED`` parity record."""
    plan = record.get("capped") or {}
    facts = record.get("facts") or {}
    python, substituted = _neutral(record)
    return {
        "label": label_of(leg, record), "leg": leg,
        "kind": "test" if holds_tests(leg) else "example",
        "script": record.get("script"), "module": record.get("module"),
        "case": record.get("case"),
        "python": python,
        "plan": {key: plan.get(key) for key in PLAN_KEYS if key in plan},
        "table2": {"rel_l2": record.get("parity_rel_l2"), "grid_cells": record.get("grid_cells"),
                   "grid_shape": record.get("grid_shape"), "driver_steps": record.get("driver_steps"),
                   "cpu_meep_steps": record.get("cpu_meep_steps"),
                   "time_matched": record.get("time_matched"),
                   "cylindrical": record.get("cylindrical"),
                   "amp_func_file_substituted": substituted},
        "facts": {key: facts.get(key) for key in FACT_KEYS if key in facts},
    }


def lift_verdict(record: Dict[str, Any]) -> Tuple[str, str]:
    """``(verdict, why)`` for a case a lift stage did not hand to its parity stage, in that
    stage's own words: no simulation built (the test ran to its end, was skipped or
    errored), the gate's refusal, or the lift's error."""
    if not record.get("has_simulation"):
        return "NO-SIMULATION", (record.get("error") or record.get("lift_note")
                                 or record.get("outcome") or "")
    if record.get("accepted") is False and record.get("accepted_after_substitution") is not True:
        return "REFUSED", "; ".join(record.get("accept_reasons") or []) or record.get("lift_error") or ""
    if record.get("lifts") is False:
        return "LIFT-FAILED", record.get("lift_error") or ""
    if record.get("lifts") is not True:
        return "NOT-LIFTED", record.get("lift_note") or record.get("accept_error") or ""
    return "NOT-MEASURED", "it lifted, and its leg's parity stage recorded no row for it"


def _resurvey(results: Path, leg: str, rows: List[Dict[str, Any]],
              not_cases: List[Dict[str, Any]], inputs: Dict[str, str],
              remeasured: List[Dict[str, Any]]) -> None:
    """Read one re-survey into the rows (see RE-SURVEYS in the module docstring)."""
    directory = LEGS[leg]
    verdicts: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for name in (LIFT_FILE, PARITY_FILE):
        path = results / directory / name
        inputs[f"{directory}/{name}"] = _sha256(path)
        seen = set()
        for record in _records(path):
            label = label_of(leg, record)
            if label in seen:
                raise ValueError(f"{directory}/{name} records {label} twice")
            seen.add(label)
            verdicts[label] = (name, record)  # the parity row, read second, wins
    position = {row["label"]: index for index, row in enumerate(rows)}
    retired = set()
    for label, (name, record) in verdicts.items():
        if name == PARITY_FILE and record.get("parity") == "MEASURED":
            again = _row(leg, record)
            if label not in position:
                position[label] = len(rows)
                rows.append(again)
                continue
            first = rows[position[label]]
            differ = [f"{part}.{key}: {first[part].get(key)!r} -> {again[part].get(key)!r}"
                      for part, key in RUN_KEYS if first[part].get(key) != again[part].get(key)]
            if differ:
                raise ValueError(f"{label}: the re-survey {leg} measured another run than "
                                 f"the {first['leg']} row: {differ}")
            remeasured.append({"label": label, "leg": leg, "row_leg": first["leg"],
                               "rel_l2": again["table2"]["rel_l2"],
                               "row_rel_l2": first["table2"]["rel_l2"]})
            continue
        if name == PARITY_FILE:
            verdict, why = record.get("parity"), record.get("parity_error") or ""
        else:
            verdict, why = lift_verdict(record)
        entry = {"leg": leg, "label": label, "case": record.get("case") or record.get("script"),
                 "parity": verdict, "error": _neutral_text(why)}
        if label in position:
            entry["retires"] = rows[position[label]]["leg"]
            retired.add(label)
        not_cases.append(entry)
    rows[:] = [row for row in rows if row["label"] not in retired]


def manifest_rows(results: Path) -> Dict[str, Any]:
    """The manifest document, derived from the legs' ``parity.jsonl`` files and the
    re-surveys' ``lift.jsonl`` files."""
    unknown = [leg for leg in RESURVEYS if leg not in LEGS]
    if unknown:
        raise ValueError(f"RESURVEYS names legs LEGS does not hold: {unknown}")
    rows: List[Dict[str, Any]] = []
    not_cases: List[Dict[str, Any]] = []
    inputs: Dict[str, str] = {}
    remeasured: List[Dict[str, Any]] = []
    for leg, directory in LEGS.items():
        if leg in RESURVEYS:
            continue
        path = results / directory / PARITY_FILE
        inputs[f"{directory}/{PARITY_FILE}"] = _sha256(path)
        for record in _records(path):
            name = record.get("case") or record.get("script")
            if record.get("parity") != "MEASURED":
                not_cases.append({"leg": leg, "case": name, "parity": record.get("parity"),
                                  "error": _neutral_text(record.get("parity_error"))})
                continue
            rows.append(_row(leg, record))
    labels = [row["label"] for row in rows]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"the corpus labels are not unique: {duplicates}")
    for leg in RESURVEYS:
        _resurvey(results, leg, rows, not_cases, inputs, remeasured)
    superseded = {Path(directory).name for leg, directory in LEGS.items() if "native" in leg}
    return {"schema": MANIFEST_SCHEMA, "version": MANIFEST_VERSION,
            "about": "Table 2's measured rows as Tier C cases (parity/meep_gpu/meep_identity_corpus.py)",
            "legs": dict(LEGS), "resurveys": list(RESURVEYS), "inputs_sha256": inputs,
            "rows": rows, "not_cases": not_cases, "remeasured": remeasured,
            "not_cases_reason": ("recorded as something other than MEASURED by their leg; the "
                                 f"native legs {sorted(superseded)} measured the two "
                                 "examples/main rows of this kind at native resolution; a "
                                 "re-survey's entries carry its verdict, and 'retires' names "
                                 "the leg whose row it took out")}


def load_manifest(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or os.environ.get(MANIFEST_ENVIRONMENT)
    if not path:
        return {"rows": []}
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != MANIFEST_SCHEMA or document.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{path} is not a {MANIFEST_SCHEMA} v{MANIFEST_VERSION} manifest")
    return document


def _rows() -> Dict[str, Dict[str, Any]]:
    return {row["label"]: row for row in load_manifest()["rows"]}


CASES = tuple(_rows())


def _source() -> Path:
    return Path(os.environ.get(SOURCE_ENVIRONMENT, os.path.join(os.path.expanduser("~"), "meep")))


def _apply_plan(sim: Any, row: Dict[str, Any]) -> None:
    """The parity stage's one change to a captured simulation: a capped row's resolution."""
    plan = row["plan"]
    if plan.get("resolution_capped"):
        if getattr(sim, "fields", None) is not None:
            raise RuntimeError(f"{row['label']}: the row ran capped at resolution "
                               f"{plan['resolution_used']}, but the captured simulation is "
                               "already initialized")
        sim.resolution = float(plan["resolution_used"])


def _build_example(mp: Any, row: Dict[str, Any]) -> Any:
    from parity.meep_gpu.survey_meep_examples import scratch_workdir  # noqa: PLC0415
    from parity.meep_gpu.sweep_corpus_lift_parity import capture_simulation  # noqa: PLC0415

    examples = _source() / "python" / "examples"
    script = examples / row["script"]
    if not script.is_file():
        raise FileNotFoundError(f"{script}: no such example (set {SOURCE_ENVIRONMENT})")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(scratch_workdir(examples, Path.cwd()))
    record, sim, restore = capture_simulation(str(script))
    restore()
    if sim is None:
        raise RuntimeError(f"{row['script']}: the script built no mp.Simulation "
                           f"({record.get('outcome')}: {record.get('error')})")
    _apply_plan(sim, row)
    return sim


def _build_test(mp: Any, row: Dict[str, Any]) -> Any:
    from parity.meep_gpu.drive_meep_test_assertions import substitute_amp_func_file  # noqa: PLC0415
    from parity.meep_gpu.survey_meep_examples import scratch_workdir  # noqa: PLC0415
    from parity.meep_gpu.survey_meep_tests import build_child_namespace  # noqa: PLC0415

    tests = _source() / "python" / "tests"
    shims = os.path.join(HERE, "test_shims")
    if shims not in sys.path:
        sys.path.append(shims)
    module_path = tests / row["module"]
    if not module_path.is_file():
        raise FileNotFoundError(f"{module_path}: no such test module (set {SOURCE_ENVIRONMENT})")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(scratch_workdir(tests, Path.cwd()))
    namespace = build_child_namespace()
    module = namespace["_import_module"](str(module_path))
    class_name, method = row["case"].split(".", 1)
    record, sim, restore, _case = namespace["_run_case"](module, str(module_path), class_name,
                                                         method, TEST_SETUP_TIMEOUT_S)
    restore()
    if sim is None:
        raise RuntimeError(f"{row['label']}: the test built no mp.Simulation "
                           f"({record.get('outcome')}: {record.get('error')})")
    substituted: List[str] = []
    substitute_amp_func_file(mp, sim, substituted)
    if bool(substituted) != bool(row["table2"].get("amp_func_file_substituted")):
        raise RuntimeError(f"{row['label']}: amp_func_file substitution {substituted} here, "
                           f"{row['table2'].get('amp_func_file_substituted')} in the row")
    _apply_plan(sim, row)
    return sim


def build(mp: Any, label: str) -> Any:
    """The row's ``mp.Simulation``, captured and planned as Table 2 measured it."""
    rows = _rows()
    if label not in rows:
        raise KeyError(f"{label!r} is not a row of the corpus manifest")
    row = rows[label]
    _CURRENT.clear()
    _CURRENT.update(row)
    return (_build_test if row["kind"] == "test" else _build_example)(mp, row)


def steps_planned(mp: Any, sim: Any) -> int:
    """Table 2's step count for the row ``build`` made."""
    return int(_CURRENT["table2"]["driver_steps"])


def after_lift(mp: Any, sim: Any, driver: Any) -> Dict[str, Any]:
    """The lifted grid must be the row's: same cell count as Table 2 recorded."""
    cells = int(math.prod(int(value) for value in driver.shape))
    facts = {"label": _CURRENT["label"], "grid_cells": cells,
             "table2_grid_cells": _CURRENT["table2"]["grid_cells"],
             "resolution": float(sim.resolution),
             "table2_rel_l2": _CURRENT["table2"]["rel_l2"]}
    if cells != int(_CURRENT["table2"]["grid_cells"]):
        raise RuntimeError(f"{_CURRENT['label']}: the lift built {cells} cells, the row "
                           f"{_CURRENT['table2']['grid_cells']}: not the row Table 2 measured")
    return facts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    make = sub.add_parser("manifest", help="derive the manifest from the legs' parity.jsonl files")
    make.add_argument("--results", required=True, help="the results/ directory holding LEGS")
    make.add_argument("--out", required=True)
    show = sub.add_parser("list", help="one line per row: label, cells, steps, cell-steps")
    show.add_argument("--manifest", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "manifest":
        document = manifest_rows(Path(arguments.results))
        Path(arguments.out).write_text(json.dumps(document, indent=1, sort_keys=True) + "\n",
                                       encoding="utf-8")
        print(f"{len(document['rows'])} rows, {len(document['not_cases'])} not cases -> "
              f"{arguments.out}", flush=True)
        return 0
    if arguments.command == "list":
        for row in load_manifest(arguments.manifest)["rows"]:
            cells, steps = row["table2"]["grid_cells"], row["table2"]["driver_steps"]
            print(f"{row['label']}\t{cells}\t{steps}\t{cells * steps}", flush=True)
        return 0
    parser.print_help()
    return 3


if __name__ == "__main__":
    sys.exit(main())
