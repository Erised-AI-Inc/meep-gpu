"""What a Metal H->D lift leg needs from the host it runs on: an interpreter, four input
files, and the capabilities some corpus rows' scripts need.

The lift legs in question are ``leg_lift`` in ``gate_metal_fused_hd_pair.py``, in the five
sibling gates that keep their own copy (bfast, conductive, cylindrical_real, folded,
nonlinear), and in ``hd_tail_gate_common.py`` (the beta_real, beta_complex and
folded_complex gates). Each re-lifts the corpus rows the standing census puts in its cell,
one child process per row.

THE INTERPRETER. A lift child runs under the gate's own interpreter, ``sys.executable``:
the round's certification environment (``rounds/1.0/metal_round.py --python``), as the CUDA
gates' children do. Until 1.0 each child ran under the interpreter the census recorded for
its row (``measure_predicate_coverage.py`` stamps ``interpreter``, the corpus survey
``python``): an absolute path on the machine that cut the census. On that machine the
children ran outside the certification environment; on any other the path did not exist
and the leg died with ``FileNotFoundError``. No interpreter is read from evidence now. The
recorded path stays on each row as ``recorded_interpreter``, as provenance only
(:func:`row_fields`).

THE INPUTS. Every ``lift_basis`` reads the census's three row files and the seam record
(:func:`lift_inputs`), and so does ``gate_metal_complex_fused_hd_pair.cell_rows``, whose
legs drive its own cell's recorded grids in process. None of them ships with the release
tree. A missing ``tests_param_matched.jsonl`` used to shrink each row set with no error,
because each leg file was read only if present; all four are now required by name
(:func:`missing_inputs`).

THE CAPABILITIES. A few corpus rows need more than MEEP and this package: a MEEP built
against libGDSII and gdspy, or PyMieScatt, nlopt and autograd. What a row needs is data,
:data:`LIFT_RECORD_NEEDS`, keyed by the row's ``lift_record`` (the corpus campaign leg the
census took the row from, recorded per row in the census and the seam record). How each
need is probed is data too, :data:`NEEDS`. A leg probes the needs of its rows once, in a
child under its own interpreter (:func:`host_check`). A row with an absent need is not
launched: it is recorded under ``unlifted_on_this_host`` with the probe's reason and counted
out of the leg's denominator, the CUDA gate's disposition for a row its host cannot lift
(``gate_cuda_fused_hd_pair.py``, ``unlifted_on_this_host``); a leg that asks the engine
about no row at all fails. Any other row that fails is a hole and fails the leg. A probe
whose answer cannot be read is ignorance, not absence, and fails the leg by name. The
reference environments carry every declared need, so a round there lifts every row; the
round's preflight lists an unlifted one by gate and does not stop for it.

Standard library only. This file is in the closures of the nine lifting gates and of
``gate_metal_complex_fused_hd_pair``; what the ROUND does with it (the per-cell preflight)
lives in ``rounds/1.0/metal_round.py``, which is in none.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: The census row files every ``lift_basis`` reads under ``results/<census>/``.
CENSUS_FILES: Tuple[str, ...] = ("examples.jsonl", "tests.jsonl", "tests_param_matched.jsonl")
#: The seam record's row file, under ``results/<seam record>/``.
SEAM_FILE = "h_to_d_seam.jsonl"

#: What a row needs beyond the certification environment's MEEP and this package, by the
#: row's lift record. A lift record that is not listed needs nothing more. The two listed
#: records are the 2026-08-09 corpus campaign's side legs: ``_gdsii`` lifted coupler.py and
#: ring_gds.py (``mp.get_GDSII_prisms``; ring_gds.py writes its layout with gdspy), and
#: ``_sigma`` lifted mie_scattering.py, differential_cross_section.py (PyMieScatt) and
#: waveguide_crossing.py (nlopt, autograd). A row is unlifted when ANY need of its record is
#: absent, so on a host lacking one package all of that record's rows are counted out, by
#: name.
LIFT_RECORD_NEEDS: Dict[str, Tuple[str, ...]] = {
    "corpus_final_numpy_2026-08-09_monitorfix_gdsii": ("gdspy", "libGDSII"),
    "corpus_final_numpy_2026-08-09_monitorfix_sigma": ("PyMieScatt", "nlopt", "autograd"),
}

#: How each need is probed: ``("module", NAME)`` imports NAME; ``("meep", NAME)`` calls
#: ``meep.NAME()`` and requires a true answer (``meep.with_libGDSII()`` is MEEP's own report
#: of whether it was built against libGDSII; without it the GDSII calls abort).
NEEDS: Dict[str, Tuple[str, str]] = {
    "gdspy": ("module", "gdspy"),
    "libGDSII": ("meep", "with_libGDSII"),
    "PyMieScatt": ("module", "PyMieScatt"),
    "nlopt": ("module", "nlopt"),
    "autograd": ("module", "autograd"),
}

#: The probe's answer line on stdout: this prefix, then ``{need: None | why absent}``.
PROBE_PREFIX = "LIFT_HOST_PROBE="
#: The probe, run as ``<interpreter> -c PROBE_PROGRAM '[[need, kind, name], ...]'``.
PROBE_PROGRAM = (
    "import importlib, json, sys\n"
    "answers = {}\n"
    "for need, kind, name in json.loads(sys.argv[1]):\n"
    "    try:\n"
    "        if kind == 'module':\n"
    "            importlib.import_module(name)\n"
    "            answers[need] = None\n"
    "        else:\n"
    "            meep = importlib.import_module('meep')\n"
    "            answers[need] = (None if getattr(meep, name)() else\n"
    "                             'meep.' + name + '() is False in this MEEP build')\n"
    "    except Exception as exc:\n"
    "        answers[need] = (type(exc).__name__ + ': ' + str(exc))[:300]\n"
    f"print({PROBE_PREFIX!r} + json.dumps(answers), flush=True)\n")


def lift_inputs(census: str, seam_record: str) -> List[str]:
    """The four files a ``lift_basis`` reads, relative to ``results/``."""
    return [f"{census}/{name}" for name in CENSUS_FILES] + [f"{seam_record}/{SEAM_FILE}"]


def missing_inputs(results: Path, census: str, seam_record: str) -> List[str]:
    """The inputs of :func:`lift_inputs` that are not files under ``results``, by name."""
    return [rel for rel in lift_inputs(census, seam_record)
            if not (Path(results) / rel).is_file()]


def missing_reason(results: Path, missing: Sequence[str]) -> str:
    return (f"the lift leg needs {len(missing)} absent input(s) under {results}: "
            f"{', '.join(missing)}. The census and seam records are not shipped; stage "
            f"them under results/ before a lift runs")


def row_fields(record: Mapping[str, Any], seam_row: Mapping[str, Any]) -> Dict[str, Any]:
    """The interpreter and capability fields of one lift row, merged in by ``lift_basis``.

    ``interpreter`` is the one the child runs under (this process's); the census's
    recorded path is ``recorded_interpreter`` and is never launched."""
    lift_record = record.get("lift_record") or seam_row.get("lift_record")
    return {
        "interpreter": sys.executable,
        "recorded_interpreter": (record.get("interpreter") or record.get("python")
                                 or seam_row.get("interpreter")),
        "lift_record": lift_record,
        "lift_needs": list(LIFT_RECORD_NEEDS.get(lift_record or "", ())),
    }


def probe(needs: Iterable[str], environment: Optional[Mapping[str, str]] = None,
          cwd: Optional[Path] = None, timeout: float = 300.0,
          executable: Optional[str] = None) -> Dict[str, Any]:
    """Probe ``needs`` once, in a child under ``executable`` (this interpreter).

    Returns ``{"interpreter", "needs": {need: None if present, else why absent},
    "unreadable": None or why the answer could not be read}``. The child keeps MEEP and
    its MPI out of the calling process."""
    interpreter = executable or sys.executable
    wanted = sorted(set(needs))
    out: Dict[str, Any] = {"interpreter": interpreter, "needs": {}, "unreadable": None}
    if not wanted:
        return out
    undeclared = [need for need in wanted if need not in NEEDS]
    if undeclared:
        out["unreadable"] = (f"needs {undeclared} are declared in no probe "
                             f"(lift_host.NEEDS)")
        return out
    command = [interpreter, "-c", PROBE_PROGRAM,
               json.dumps([[need, *NEEDS[need]] for need in wanted])]
    try:
        done = subprocess.run(command, cwd=None if cwd is None else str(cwd),
                              env=None if environment is None else dict(environment),
                              capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        out["unreadable"] = (f"the capability probe under {interpreter} did not finish: "
                             f"{type(exc).__name__}: {exc}")
        return out
    lines = [line[len(PROBE_PREFIX):] for line in (done.stdout or "").splitlines()
             if line.startswith(PROBE_PREFIX)]
    try:
        answers = json.loads(lines[-1])
    except (IndexError, ValueError):
        answers = None
    if not isinstance(answers, dict) or sorted(answers) != wanted:
        out["unreadable"] = (f"the capability probe under {interpreter} printed no answer "
                             f"for {wanted} (exit {done.returncode}): "
                             f"{(done.stderr or '').strip()[-300:]}")
        return out
    out["needs"] = {need: answers[need] for need in wanted}
    return out


def unlifted(rows: Sequence[Mapping[str, Any]], probed: Mapping[str, Any]) -> Dict[str, str]:
    """``label -> reason`` for every row with a need ``probed`` found absent."""
    answers = probed.get("needs") or {}
    out: Dict[str, str] = {}
    for row in rows:
        needs = list(row.get("lift_needs") or ())
        absent = [f"{need} ({answers[need]})" for need in needs if answers.get(need)]
        if absent:
            out[str(row["label"])] = (f"needs {', '.join(needs)} (lift record "
                                      f"{row.get('lift_record')}); absent under "
                                      f"{probed.get('interpreter')}: {'; '.join(absent)}")
    return out


def host_check(rows: Sequence[Mapping[str, Any]],
               environment: Optional[Mapping[str, str]] = None,
               cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Probe the needs of ``rows`` once and name the rows this host cannot lift.

    ``{"interpreter", "needs", "unreadable", "unlifted_on_this_host": {label: reason}}``;
    nothing is launched when no row needs anything."""
    probed = probe((need for row in rows for need in row.get("lift_needs") or ()),
                   environment=environment, cwd=cwd)
    probed["unlifted_on_this_host"] = ({} if probed["unreadable"]
                                       else unlifted(rows, probed))
    return probed


def unlifted_entry(row: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    """The record a lift leg keeps for a row it did not launch on this host."""
    return {**row, "measured": False, "unlifted_on_this_host": reason}
