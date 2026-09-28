#!/usr/bin/env python3
"""THE update_H -> step_D SEAM, MEASURED PER CORPUS ROW — the probe behind the fourth
priced seam (``H_to_D``) on the three fusion boards.

WHY THIS FILE EXISTS. The fusion-residue audit (§2) found that the boundary between
``update_H`` and ``step_D`` was never priced: the three boards price the two
curl->constitutive seams (B->H, D->E) and the E->P chain seam, and the boundary that
closes the loop — the magnetic constitutive into the electric curl — sat outside
every denominator. The driver settles what lies in it (``driver.py``, ``step()``,
between the ``update_H`` consult and the ``step_D`` consult): exactly one pass,

    for source in electric:
        getattr(source, "withdraw", _no_withdraw)(self.fields)

the integrated ELECTRIC sources returning their standing dipole offset to D before
the curl ladder reads it. No injection, no fill, no wall clear. So per row the seam
is ``update_H`` followed by ``step_D`` with the withdraw as its only in-seam pass,
present only where the row carries an integrated electric source with at least one
deposit point — and THAT is a fact about the lifted row, not about a name, which is
what this probe measures.

WHAT IT MEASURES, PER ROW (the ``h_to_d_seam`` block on every census row):

- every source the driver holds: its class, component, field type, envelope, MEEP's
  ``is_integrated`` flag, whether its ``withdraw`` attribute is the driver's no-op
  (``_no_withdraw``) and how many deposit points it owns. A withdraw DOES WORK when
  the source is integrated, carries a ``withdraw`` method and owns at least one
  point — ``VolumeSource.withdraw`` returns before touching the array otherwise.
- whether the array path's ``update_H`` launches anything on this row: the function's
  first statement is ``if not _pml_is_active(pml): return`` (stepping.py), so the fact
  is read off ``driver.pml`` and then MEASURED by poisoning: B is filled with ones,
  ``update_H`` is run once and every stored H array is compared with its copy. A row
  where no H array changed is a row where the seam has one launch and a non-launch.

HOW IT RUNS. It is a BATTERY for ``measure_predicate_coverage.py`` — the census
driver that lifts every row of the 194-row basis under the interpreter its lift
record names, one child per row, across the three legs (examples, tests,
tests_param) with the parameterised leg matched by facts — so the rows here are
the SAME rows the three boards price, lifted the same way, and nothing about the
basis is re-derived here. ``SUBJECT_PACKAGE`` is the engine itself (``meep_gpu/*.py``:
the driver and the sources module are the subject) so the census's subject digest
pins what this probe read. Run without arguments it cuts the three legs, matches the
parameterised rows and then writes the JOIN:

- ``<out>/h_to_d_seam.jsonl`` — one line per row of the basis: the measured block
  above beside the arm the composer selects at ``update_H`` and at ``step_D`` on each
  backend, READ OFF the three standing censuses' ``plan_step`` (no predicate is
  re-run here).
- ``<out>/summary.json`` — the counts, every one with its denominator named.

WHAT IT DOES NOT DO. It runs on the host, on the NumPy array path, and measures
nothing on a device. Whether a fused ``update_H + step_D`` launch can bracket the
withdraw byte-identically is the device probe the audit asked for; the boards file
those rows under ``withdraw_seam`` until that probe exists, and this file is not it.

Progress (the progress-reporting rule): the census driver writes a flushed line per row
to ``<out>/<leg>.log`` and ``<out>/<leg>.progress.log``; this driver writes a flushed
line per leg and per summary step to ``<out>/probe.log``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _find_api_root(start: Path) -> Path:
    """The repository root, BY NAME, never by depth (see measure_predicate_coverage)."""
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}; refusing to guess")


_API = _find_api_root(_HERE)
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

RESULTS = _HERE / "results"

#: THE SUBJECT IS THE ENGINE. The census driver digests ``meep_gpu/<SUBJECT_PACKAGE>/*.py``
#: plus the shared engine files; ``.`` makes that ``meep_gpu/*.py`` — driver.py (the seam)
#: and sources.py (the withdraw) are the files this probe's facts are read from.
SUBJECT_PACKAGE = "."

#: The campaign this probe writes by default, and the three standing censuses whose
#: ``plan_step`` the join reads. Overridable from the command line; dated so a re-cut
#: on a later tree lands beside this one rather than over it.
CAMPAIGN = "h_to_d_seam_2026-09-04"
STANDING_CENSUSES: Dict[str, str] = {
    "metal": "metal_coverage_2026-09-03_complete",
    "triton": "predicate_coverage_triton_2026-09-03_extended",
    "cuda": "cuda_predicate_coverage_2026-09-03_extended",
}

#: The null magnetic constitutive, as each board names it. A board files an instance
#: as ``not_fusion_surface`` where the arm selected at ``update_H`` is this one; the
#: join records the same verdict beside the arm so the summary's tally can be checked
#: against the boards instance by instance. Metal: ``CONSTITUTIVE_BODY[("update_H",
#: "no-PML null")] is None``; Triton: ``NULL_CONSTITUTIVE_ARM``; CUDA:
#: ``NULL_PLAN_LABELS`` on the selected kernel.
NULL_UPDATE_H = {
    "metal": {"arm": "no-PML null"},
    "triton": {"arm": "no-PML null"},
    "cuda": {"kernel": "fused_update_H_no_pml_null"},
}


# --- the battery ---------------------------------------------------------------------


def runtime_reasons() -> List[str]:
    """No process-level precondition: the probe reads the lifted driver in pure Python."""
    return []


def _measure_update_H_work(driver: Any, stepping: Any) -> Dict[str, Any]:
    """Run the array path's ``update_H`` once over poisoned B and report what moved."""
    import numpy  # noqa: PLC0415

    fields = driver.fields
    h_names = [name for name in stepping.H_PML_ARRAYS
               if getattr(fields, name, None) is not None]
    b_names = [name for name in ("Bx", "By", "Bz")
               if getattr(fields, name, None) is not None]
    if not h_names:
        return {"measured": True, "h_storage_present": False, "changed": False,
                "changed_arrays": [],
                "why": ("no stored H array exists on this row: H is served on demand "
                        "from B (Fields.get_H) and update_H has nothing to write")}
    before = {name: numpy.array(getattr(fields, name), copy=True) for name in h_names}
    for name in b_names:
        getattr(fields, name)[...] = 1
    try:
        stepping.update_H(fields, driver.pml)
    except BaseException as exc:  # noqa: BLE001
        return {"measured": False, "h_storage_present": True,
                "error": f"{type(exc).__name__}: {exc}"[:400]}
    changed = [name for name in h_names
               if not numpy.array_equal(before[name], numpy.asarray(getattr(fields, name)))]
    return {"measured": True, "h_storage_present": True, "changed": bool(changed),
            "changed_arrays": changed, "poisoned_b_arrays": b_names}


def evaluate(driver: Any, probe: Any) -> Dict[str, Any]:
    """The census driver's battery hook: the ``h_to_d_seam`` block for one lifted row."""
    from meep_gpu import stepping  # noqa: PLC0415
    from meep_gpu.driver import _no_withdraw  # noqa: PLC0415
    from meep_gpu.sources import FIELD_TYPE_B  # noqa: PLC0415

    per_source: List[Dict[str, Any]] = []
    for source in list(getattr(driver, "_sources", ())):
        field_type = getattr(source, "field_type", None)
        withdraw = getattr(source, "withdraw", _no_withdraw)
        is_integrated = bool(getattr(source, "is_integrated", False))
        n_points = getattr(source, "_n_source_points", None)
        envelope = getattr(source, "envelope", None)
        per_source.append({
            "type": type(source).__name__,
            "component": getattr(source, "component", None),
            "field_type": field_type,
            "electric": field_type != FIELD_TYPE_B,
            "envelope": type(envelope).__name__ if envelope is not None else None,
            "is_integrated": is_integrated,
            "withdraw_is_the_no_op": withdraw is _no_withdraw,
            "n_source_points": int(n_points) if n_points is not None else None,
            # VolumeSource.withdraw returns before touching the array unless the
            # envelope is integrated AND the source owns a point (sources.py).
            "withdraw_does_work": (withdraw is not _no_withdraw and is_integrated
                                   and bool(n_points)),
        })
    electric = [s for s in per_source if s["electric"]]
    magnetic = [s for s in per_source if not s["electric"]]
    pml_active = bool(stepping._pml_is_active(driver.pml))
    poison = _measure_update_H_work(driver, stepping)
    block = {
        "seam": "H_to_D",
        "halves": ["update_H", "step_D"],
        "in_seam_pass": ("the electric integrated-source withdraw: driver.py step(), "
                         "`for source in electric: getattr(source, 'withdraw', "
                         "_no_withdraw)(self.fields)` — the only statement between the "
                         "update_H consult and the step_D consult"),
        "n_sources": len(per_source),
        "n_electric_sources": len(electric),
        "n_magnetic_sources": len(magnetic),
        "n_integrated_electric_sources": sum(1 for s in electric if s["is_integrated"]),
        "n_integrated_magnetic_sources": sum(1 for s in magnetic if s["is_integrated"]),
        "n_electric_withdraws_that_do_work": sum(1 for s in electric
                                                 if s["withdraw_does_work"]),
        "n_electric_withdraws_that_are_the_no_op": sum(1 for s in electric
                                                       if s["withdraw_is_the_no_op"]),
        # THE SEAM FACT: does anything happen between the two halves on this row?
        "withdraw_in_seam": any(s["withdraw_does_work"] for s in electric),
        "update_H_array_path": {
            "pml_is_active": pml_active,
            # stepping.update_H's first statement: `if not _pml_is_active(pml): return`
            "returns_before_its_first_statement": not pml_active,
            "poison_measurement": poison,
        },
        "sources": per_source,
    }
    return {"h_to_d_seam": block}


# --- the join ------------------------------------------------------------------------


def _load(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def rows(census: Path) -> List[dict]:
    """The measured rows of a census, the parameterised leg's matched rows substituted —
    the same rule the three boards read their censuses by."""
    record = _load(census / "examples.jsonl") + _load(census / "tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in _load(census / "tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    # One record per (leg, row): a recovered solo-child record (appended after the
    # module child's `child died` row) supersedes the unmeasured one it recovers.
    by_key: Dict[tuple, dict] = {}
    for r in record:
        key = (r.get("leg"), r.get("row"))
        if key not in by_key or (r.get("measured") and not by_key[key].get("measured")):
            by_key[key] = r
    return [r for r in by_key.values() if r.get("measured")]


def label(row: dict) -> str:
    return f"{row['leg']}:{row['row']}"


def _metal_arms(row: dict) -> Dict[str, Any]:
    selected = (row.get("plan_step") or {}).get("selected") or {}
    arm_h, arm_d = selected.get("update_H"), selected.get("step_D")
    return {"update_H": arm_h, "step_D": arm_d,
            "update_H_is_null": arm_h == NULL_UPDATE_H["metal"]["arm"]}


def _triton_arms(row: dict) -> Dict[str, Any]:
    # build_triton_fusion_matrix.arm: the one-admitter rule off plan_step.slots, because
    # the census host has no Triton and `selected` measures the harness there.
    def arm(slot: str) -> Optional[str]:
        block = (row.get("plan_step") or {}).get("slots", {}).get(slot)
        admitted = tuple(block["admitted"]) if block else ()
        return admitted[0] if len(admitted) == 1 else None
    arm_h, arm_d = arm("update_H"), arm("step_D")
    return {"update_H": arm_h, "step_D": arm_d,
            "update_H_is_null": arm_h == NULL_UPDATE_H["triton"]["arm"]}


def _cuda_arms(row: dict) -> Dict[str, Any]:
    plan = row.get("plan_step") or {}
    selected = plan.get("selected") or {}
    kernels = plan.get("selected_kernel") or {}
    families = plan.get("selected_family") or {}
    arm_h = (f"{families['update_H']}/{selected['update_H']}"
             if selected.get("update_H") else None)
    arm_d = (f"{families['step_D']}/{selected['step_D']}"
             if selected.get("step_D") else None)
    return {"update_H": arm_h, "step_D": arm_d,
            "update_H_kernel": kernels.get("update_H"),
            "step_D_kernel": kernels.get("step_D"),
            "update_H_is_null": kernels.get("update_H") == NULL_UPDATE_H["cuda"]["kernel"]}


_ARMS = {"metal": _metal_arms, "triton": _triton_arms, "cuda": _cuda_arms}


def would_be_bucket(block: dict, arms: dict, backend: str) -> str:
    """The bucket a board files this instance under, from THE ONE RULE the boards use
    (``h_to_d_seam.classify``), so the summary's tally is checkable against the boards
    instance by instance. No product spans update_H+step_D on any backend, so
    ``served`` never occurs here."""
    import h_to_d_seam  # noqa: PLC0415

    return h_to_d_seam.classify("", arms["update_H"], arms["step_D"],
                                bool(arms["update_H_is_null"]), block, backend,
                                f"{backend}_kernels")[0]


def summarize(out: Path, censuses: Dict[str, Path], log) -> int:
    probe_rows = {label(r): r for r in rows(out)}
    log(f"probe rows measured: {len(probe_rows)} "
        f"(examples {sum(1 for r in probe_rows.values() if r['leg'] == 'examples')}, "
        f"tests {sum(1 for r in probe_rows.values() if r['leg'] == 'tests')}, "
        f"tests_param {sum(1 for r in probe_rows.values() if r['leg'] == 'tests_param')})")
    unmeasured_probe = [label(r) for leg in ("examples", "tests")
                        for r in _load(out / f"{leg}.jsonl") if not r.get("measured")]
    log(f"probe rows NOT measured (excluded from the basis, as the boards exclude them): "
        f"{len(unmeasured_probe)}")

    census_rows: Dict[str, Dict[str, dict]] = {}
    for backend, census in censuses.items():
        census_rows[backend] = {label(r): r for r in rows(census)}
        log(f"{backend:6s} census {census.name}: {len(census_rows[backend])} measured rows")

    basis = set()
    for here in census_rows.values():
        basis |= set(here)
    missing_from_probe = sorted(basis - set(probe_rows))
    extra_in_probe = sorted(set(probe_rows) - basis)
    log(f"basis (union of the three censuses' labels): {len(basis)}; "
        f"missing from the probe: {len(missing_from_probe)}; "
        f"probe-only: {len(extra_in_probe)}")
    for name in missing_from_probe[:20]:
        log(f"    MISSING {name}")

    joined: List[dict] = []
    for name in sorted(basis & set(probe_rows)):
        row = probe_rows[name]
        block = row["h_to_d_seam"]
        arms = {}
        for backend, here in census_rows.items():
            if name in here:
                arms[backend] = _ARMS[backend](here[name])
                arms[backend]["would_be_bucket"] = would_be_bucket(block, arms[backend], backend)
            else:
                arms[backend] = None
        joined.append({
            "label": name, "leg": row["leg"], "row": row["row"],
            "lift_record": row.get("lift_record"), "interpreter": row.get("interpreter"),
            "subject_manifest_sha256": row.get("subject_manifest_sha256"),
            "h_to_d_seam": block, "arms": arms,
        })
    with (out / "h_to_d_seam.jsonl").open("w", encoding="utf-8") as handle:
        for entry in joined:
            handle.write(json.dumps(entry) + "\n")
    log(f"wrote {out / 'h_to_d_seam.jsonl'} ({len(joined)} rows)")

    def count(predicate) -> int:
        return sum(1 for e in joined if predicate(e["h_to_d_seam"]))

    n = len(joined)
    withdraw_rows = [e["label"] for e in joined if e["h_to_d_seam"]["withdraw_in_seam"]]
    summary: Dict[str, Any] = {
        "campaign": out.name,
        "date": "2026-09-04",
        "seam": "H_to_D",
        "what_sits_between_the_halves": joined[0]["h_to_d_seam"]["in_seam_pass"] if joined else None,
        "rows_in_the_join": n,
        "basis_labels": len(basis),
        "basis_missing_from_the_probe": missing_from_probe,
        "probe_only_labels": extra_in_probe,
        "probe_rows_not_measured": unmeasured_probe,
        "censuses": {b: str(p) for b, p in censuses.items()},
        "subject_digests": {
            leg: json.loads((out / f"subject_digests_{leg}.json").read_text())
            .get("manifest_sha256")
            for leg in ("examples", "tests", "tests_param")
            if (out / f"subject_digests_{leg}.json").is_file()},
        "sources": {
            "rows_with_any_source": count(lambda b: b["n_sources"] > 0),
            "rows_with_an_electric_source": count(lambda b: b["n_electric_sources"] > 0),
            "rows_with_a_magnetic_source": count(lambda b: b["n_magnetic_sources"] > 0),
            "electric_sources_total": sum(e["h_to_d_seam"]["n_electric_sources"] for e in joined),
            "integrated_electric_sources_total": sum(
                e["h_to_d_seam"]["n_integrated_electric_sources"] for e in joined),
            "rows_with_an_integrated_electric_source": count(
                lambda b: b["n_integrated_electric_sources"] > 0),
            "rows_whose_electric_withdraw_does_work (withdraw_in_seam)": len(withdraw_rows),
            "withdraw_rows": withdraw_rows,
            "rows_with_an_integrated_electric_source_owning_no_point": count(
                lambda b: b["n_integrated_electric_sources"] > 0
                and b["n_electric_withdraws_that_do_work"] == 0),
            "rows_with_an_integrated_magnetic_source": count(
                lambda b: b["n_integrated_magnetic_sources"] > 0),
            "source_types": sorted({s["type"] for e in joined
                                    for s in e["h_to_d_seam"]["sources"]}),
        },
        "update_H_array_path": {
            "rows_where_update_H_returns_before_its_first_statement": count(
                lambda b: b["update_H_array_path"]["returns_before_its_first_statement"]),
            "rows_where_poisoning_moved_no_H_array": count(
                lambda b: b["update_H_array_path"]["poison_measurement"].get("measured")
                and not b["update_H_array_path"]["poison_measurement"]["changed"]),
            "rows_where_poisoning_moved_an_H_array": count(
                lambda b: b["update_H_array_path"]["poison_measurement"].get("measured")
                and b["update_H_array_path"]["poison_measurement"]["changed"]),
            "rows_where_the_poison_measurement_failed": count(
                lambda b: not b["update_H_array_path"]["poison_measurement"].get("measured")),
            "rows_where_the_two_readings_disagree": [
                e["label"] for e in joined
                if e["h_to_d_seam"]["update_H_array_path"]["poison_measurement"].get("measured")
                and (e["h_to_d_seam"]["update_H_array_path"]["returns_before_its_first_statement"]
                     == e["h_to_d_seam"]["update_H_array_path"]["poison_measurement"]["changed"])],
        },
        "per_backend": {},
    }
    for backend in censuses:
        here = [e for e in joined if e["arms"].get(backend)]
        tally: Dict[str, int] = {}
        for e in here:
            bucket = e["arms"][backend]["would_be_bucket"]
            tally[bucket] = tally.get(bucket, 0) + 1
        summary["per_backend"][backend] = {
            "rows": len(here),
            "update_H_is_null": sum(1 for e in here if e["arms"][backend]["update_H_is_null"]),
            "update_H_unselected": sum(1 for e in here if e["arms"][backend]["update_H"] is None),
            "step_D_unselected": sum(1 for e in here if e["arms"][backend]["step_D"] is None),
            "both_halves_selected": sum(
                1 for e in here if e["arms"][backend]["update_H"] is not None
                and e["arms"][backend]["step_D"] is not None),
            "both_halves_selected_and_no_withdraw_in_seam": sum(
                1 for e in here if e["arms"][backend]["update_H"] is not None
                and e["arms"][backend]["step_D"] is not None
                and not e["h_to_d_seam"]["withdraw_in_seam"]),
            "would_be_buckets": tally,
            "would_be_buckets_note": (
                "filed by h_to_d_seam.classify, the rule the three boards apply "
                "(not_fusion_surface > withdraw_seam > missing_half > buildable_not_built; "
                "served is empty by construction — no product spans update_H+step_D)"),
            "cells": _cells(here, backend),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    log(f"wrote {out / 'summary.json'}")
    log(f"  rows {n} / basis {len(basis)}; withdraw rows {len(withdraw_rows)}; "
        f"update_H returns early on "
        f"{summary['update_H_array_path']['rows_where_update_H_returns_before_its_first_statement']} rows")
    for backend, block in summary["per_backend"].items():
        log(f"  {backend:6s} {block['would_be_buckets']}")
    return 2 if missing_from_probe else 0


def _cells(joined: List[dict], backend: str) -> List[dict]:
    cells: Dict[tuple, List[str]] = {}
    for e in joined:
        arms = e["arms"][backend]
        cells.setdefault((arms["update_H"], arms["step_D"]), []).append(e["label"])
    return [{"update_H": k[0], "step_D": k[1], "rows": len(v)}
            for k, v in sorted(cells.items(), key=lambda kv: -len(kv[1]))]


# --- recovery of one aborted case ----------------------------------------------------


def recover_case(out: Path, spec: str, environment: dict, log) -> int:
    """Re-run one tests-leg case alone and append its record, stamped, to tests.jsonl.

    The census driver runs a whole module per child and re-runs the survivors one case
    per child only when an abort cost MORE THAN ONE case; a module that aborts on
    exactly one case leaves that case as ``child died`` with no retry. This is that
    retry, in the driver's own shape (``--child-module`` + ``--child-cases [one]``),
    with the leg's own subject and battery stamps read back from
    ``subject_digests_tests.json`` so the recovered row is welded to the same tree as
    the rest of the leg. The original ``child died`` row is left in place — the join
    substitutes by (leg, row) and prefers the measured record — so the artifact keeps
    the record of the abort beside the recovery.
    """
    import measure_predicate_coverage as census  # noqa: PLC0415

    module_name, case_id = spec.split(":", 1)
    out = out.resolve()  # the child runs with cwd=out; every path handed to it is absolute
    digests = json.loads((out / "subject_digests_tests.json").read_text(encoding="utf-8"))
    solo_path = out / "per_row_tests" / (
        f"{Path(module_name).stem}__{case_id.replace('.', '_')}__recovered.json")
    command = [sys.executable, "-u", str(_HERE / "measure_predicate_coverage.py"),
               "--leg", "tests", "--child-module",
               str(Path(census.TESTS_DIR) / module_name),
               "--child-cases", json.dumps([case_id]),
               "--out-json", str(solo_path),
               "--probe", str(_API / census.PROBE),
               "--progress-log", str(out / "tests.progress.log"),
               "--battery", "probe_h_to_d_seam"]
    log(f"recover {module_name}:{case_id} as a solo child -> {solo_path.name}")
    completed = subprocess.run(command, cwd=str(out), env=environment, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if not solo_path.is_file():
        tail = (completed.stderr or b"").decode("utf-8", "replace")[-600:]
        log(f"  recovery FAILED rc={completed.returncode}: no record written; {tail!r}")
        return 4
    rows_recovered = json.loads(solo_path.read_text(encoding="utf-8"))
    with (out / "tests.jsonl").open("a", encoding="utf-8") as handle:
        for row in rows_recovered:
            row["leg"] = "tests"
            row.setdefault("module", module_name)
            row["subject_manifest_sha256"] = digests["manifest_sha256"]
            row["battery_sha256"] = digests["battery_sha256"]
            row["recovered"] = "solo child after the module child aborted on this case"
            handle.write(json.dumps(row) + "\n")
            log(f"  recovered {row.get('row')}: measured={row.get('measured')} "
                f"({row.get('lift_error') or row.get('note') or 'ok'})")
    return 0


# --- the driver ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS / CAMPAIGN)
    parser.add_argument("--leg", action="append", default=None,
                        choices=("examples", "tests", "tests_param"),
                        help="cut only these legs (default: all three)")
    parser.add_argument("--summarize-only", action="store_true",
                        help="skip the census legs; join an existing cut")
    parser.add_argument("--recover-case", action="append", default=None,
                        metavar="MODULE.py:Class.method",
                        help="re-run ONE tests-leg case as a solo child (the census "
                             "driver's own one-case-per-child recovery shape) and "
                             "append its record to tests.jsonl; for a case whose "
                             "module child aborted on it and which the driver did not "
                             "retry (it retries only when more than one case is lost)")
    for backend, default in STANDING_CENSUSES.items():
        parser.add_argument(f"--census-{backend}", default=default,
                            help=f"{backend} census directory name under results/")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "probe.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONPATH"] = str(_API)
    if not args.summarize_only:
        if not environment.get("MEEP_GPU_CORPUS_ROOT"):
            log("MEEP_GPU_CORPUS_ROOT is not set; the census driver will look for the "
                "corpus at its hard-coded default")
        for leg in (args.leg or ("examples", "tests", "tests_param")):
            started = time.time()
            log(f"leg {leg} start (battery probe_h_to_d_seam, driver "
                f"measure_predicate_coverage.py, out {out})")
            command = [sys.executable, "-u", str(_HERE / "measure_predicate_coverage.py"),
                       "--battery", "probe_h_to_d_seam", "--leg", leg, "--out", str(out)]
            completed = subprocess.run(command, cwd=str(_API), env=environment, check=False)
            log(f"leg {leg} rc={completed.returncode} ({time.time() - started:.1f} s)")
            if completed.returncode != 0:
                log(f"LEG {leg} FAILED rc={completed.returncode}; stopping before the join")
                return completed.returncode
        command = [sys.executable, str(_HERE / "match_param_rows.py"),
                   "--census", str(out.resolve())]
        completed = subprocess.run(command, cwd=str(_API), env=environment, check=False)
        log(f"match_param_rows rc={completed.returncode}")
        if completed.returncode != 0:
            return completed.returncode
    for spec in (args.recover_case or ()):
        rc = recover_case(out, spec, environment, log)
        if rc:
            return rc
    censuses = {backend: RESULTS / getattr(args, f"census_{backend}")
                for backend in STANDING_CENSUSES}
    for backend, census in censuses.items():
        if not (census / "examples.jsonl").is_file():
            raise SystemExit(f"{backend}: {census} carries no examples.jsonl; not a census")
    rc = summarize(out, censuses, log)
    log(f"PROBE COMPLETE rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
