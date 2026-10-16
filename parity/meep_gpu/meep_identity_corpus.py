#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""Table 2's corpus as a Tier C cases file: every measured row, rebuilt as it was measured.

    python parity/meep_gpu/meep_identity_corpus.py manifest --results DIR --out FILE
    python parity/meep_gpu/meep_identity_corpus.py list --manifest FILE

    MEEP_IDENTITY_CORPUS_MANIFEST=FILE MGPU_SITE_MEEP_SOURCE=DIR \\
        python parity/meep_gpu/meep_identity_tier_c.py compare \\
            --cases-file parity/meep_gpu/meep_identity_corpus.py --cases LABEL --out OUT

WHAT IT IS. ``meep_identity_tier_c.py compare --cases-file`` takes a module with
``CASES`` and ``build(mp, label)``; this is that module for the manuscript's Table 2:
the rows of the seven 2026-08-09 ``parity.jsonl`` files (``LEGS``), MEEP's
``python/examples`` (60 measured rows) and ``python/tests`` (134), each rebuilt the way
its row was measured, so Tier C word-compares the run Table 2 reported an L2 for:

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

THE MANIFEST (``manifest``) is derived from the legs' JSONL files, never transcribed: one
entry per row with its label, leg, kind, script or module and case, the plan it ran
under, its Table 2 numbers (``rel_l2``, ``grid_cells``, ``driver_steps``) and its fact
block, plus the sha256 of every input. ``CASES`` is empty until
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
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MANIFEST_ENVIRONMENT = "MEEP_IDENTITY_CORPUS_MANIFEST"
SOURCE_ENVIRONMENT = "MGPU_SITE_MEEP_SOURCE"
MANIFEST_SCHEMA = "meep_gpu.meep_identity_corpus"
MANIFEST_VERSION = 1
#: Table 2's seven legs (``bit_identity_probe_2026-10-05/recon_a``), relative to results/.
LEGS = {
    "examples/main": "corpus_final_numpy_2026-08-09_monitorfix",
    "examples/blocked": "corpus_final_numpy_2026-08-09_monitorfix_blocked",
    "examples/gdsii": "corpus_final_numpy_2026-08-09_monitorfix_gdsii",
    "examples/sigma": "corpus_final_numpy_2026-08-09_monitorfix_sigma",
    "examples/native_cavity_farfield": "corpus_final_numpy_2026-08-09_monitorfix_native_cavity_farfield",
    "examples/native_zone_plate": "corpus_final_numpy_2026-08-09_monitorfix_native_zone_plate",
    "tests": "meep_python_tests_2026-08-09_monitorfix",
}
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


def label_of(leg: str, record: Dict[str, Any]) -> str:
    """A row's case label: ``ex.<script stem>`` or ``t.<module stem>.<Class.method>``."""
    if leg == "tests":
        return f"t.{Path(record['module']).stem}.{record['case']}"
    return f"ex.{Path(record['script']).stem}"


def manifest_rows(results: Path) -> Dict[str, Any]:
    """The manifest document, derived from the legs' ``parity.jsonl`` files."""
    rows: List[Dict[str, Any]] = []
    not_cases: List[Dict[str, Any]] = []
    inputs: Dict[str, str] = {}
    for leg, directory in LEGS.items():
        path = results / directory / "parity.jsonl"
        inputs[f"{directory}/parity.jsonl"] = _sha256(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("case") or record.get("script")
            if record.get("parity") != "MEASURED":
                not_cases.append({"leg": leg, "case": name, "parity": record.get("parity"),
                                  "error": str(record.get("parity_error") or "")[:300]})
                continue
            plan = record.get("capped") or {}
            facts = record.get("facts") or {}
            rows.append({
                "label": label_of(leg, record), "leg": leg,
                "kind": "test" if leg == "tests" else "example",
                "script": record.get("script"), "module": record.get("module"),
                "case": record.get("case"),
                "python": record.get("python"),
                "plan": {key: plan.get(key) for key in PLAN_KEYS if key in plan},
                "table2": {"rel_l2": record.get("parity_rel_l2"), "grid_cells": record.get("grid_cells"),
                           "grid_shape": record.get("grid_shape"), "driver_steps": record.get("driver_steps"),
                           "cpu_meep_steps": record.get("cpu_meep_steps"),
                           "time_matched": record.get("time_matched"),
                           "cylindrical": record.get("cylindrical"),
                           "amp_func_file_substituted": record.get("amp_func_file_substituted")},
                "facts": {key: facts.get(key) for key in FACT_KEYS if key in facts},
            })
    labels = [row["label"] for row in rows]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"the corpus labels are not unique: {duplicates}")
    superseded = {Path(directory).name for leg, directory in LEGS.items() if "native" in leg}
    return {"schema": MANIFEST_SCHEMA, "version": MANIFEST_VERSION,
            "about": "Table 2's measured rows as Tier C cases (parity/meep_gpu/meep_identity_corpus.py)",
            "legs": dict(LEGS), "inputs_sha256": inputs, "rows": rows, "not_cases": not_cases,
            "not_cases_reason": ("recorded as something other than MEASURED by their leg; the "
                                 f"native legs {sorted(superseded)} measured the two "
                                 "examples/main rows of this kind at native resolution")}


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
