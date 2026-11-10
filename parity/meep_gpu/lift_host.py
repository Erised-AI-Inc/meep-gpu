"""What a Metal H->D lift leg needs from the host it runs on: an interpreter, four input
files, and the capabilities some corpus rows' scripts need; and, for the lift legs of every
kernel table, which lift-basis rows the identity corpus has retired.

The lift legs in question are ``leg_lift`` in ``gate_metal_fused_hd_pair.py``, in the five
sibling gates that keep their own copy (bfast, conductive, cylindrical_real, folded,
nonlinear), and in ``hd_tail_gate_common.py`` (the beta_real, beta_complex and
folded_complex gates). Each re-lifts the corpus rows the standing census puts in its cell,
one child process per row.

THE INTERPRETER. A lift child runs under the gate's own interpreter, ``sys.executable``:
the certification environment (``rounds/1.0/metal_round.py --python``), as the CUDA
gates' children do. Until 1.0 each child ran under the interpreter the census recorded for
its row (``measure_predicate_coverage.py`` stamps ``interpreter``, the corpus survey
``python``): an absolute path on the machine that cut the census. On that machine the
children ran outside the certification environment; on any other the path did not exist
and the leg died with ``FileNotFoundError``. No interpreter is read from evidence now. The
recorded path stays on each row as ``recorded_interpreter``, as provenance only
(:func:`row_fields`). What a row and a probe RECORD names an interpreter under its root's
label (``neutral_ids.path``: ``<home>/...``), since a gate artifact leaves the machine;
the probe EXECUTES this process's own ``sys.executable``, which is never read back from a
record.

THE INPUTS. Every ``lift_basis`` reads the census's three row files and the seam record
(:func:`lift_inputs`), and so does ``gate_metal_complex_fused_hd_pair.cell_rows``, whose
legs drive its own cell's recorded grids in process. None of them ships with the release
tree. A missing ``tests_param_matched.jsonl`` used to shrink each row set with no error,
because each leg file was read only if present; all four are now required by name
(:func:`missing_inputs`).

THE CAPABILITIES. A few corpus rows need more than MEEP and this package: gdstk,
PyMieScatt, nlopt or autograd. What a row needs is data: each row
of a listed lift record (the corpus campaign leg the census took the row from, recorded per
row in the census and the seam record as ``lift_record``) needs what
:data:`LIFT_ROW_NEEDS` names for it, within what its record declares
(:data:`LIFT_RECORD_NEEDS`); :func:`row_needs` is the rule. How each need is probed is data
too, :data:`NEEDS`. A leg probes the needs of its rows once, in a child under its own
interpreter (:func:`host_check`). A row with an absent need is not launched: it is
recorded under ``unlifted_on_this_host`` with the probe's reason and counted out of the
leg's denominator, the CUDA gate's disposition for a row its host cannot lift
(``gate_cuda_fused_hd_pair.py``, ``unlifted_on_this_host``); a leg that asks the engine
about no row at all fails, on the Metal and the CUDA gates alike. Any other row that fails
is a hole and fails the leg. A probe whose answer cannot be read is ignorance, not
absence, and fails the leg by name. The reference environments carry every declared need,
so a certification round there lifts every row; its preflight lists an unlifted one by gate and
does not stop for it.

THE ROWS THE IDENTITY CORPUS RETIRES. A lift basis is a census and a seam record, cut once
over the MEEP corpus of their day; the identity corpus follows MEEP. When a re-survey of a
newer MEEP release does not measure a row an earlier corpus leg measured, the corpus manifest
takes the row out and names it in ``not_cases`` with the re-survey's verdict and the leg whose
row it was (``retires``; ``meep_identity_corpus``). The four sharded dump/load tests
(``TestLoadDump.test_load_dump_{fields,structure}_sharded_{2d,3d}``) are such rows: MEEP
1.35.0 runs them only under ``mpirun`` with more than one process and skips them in one, and
a lift inside one of several MPI ranks is refused by design, so their lift children build no
simulation. A basis cut before the retirement still names them. One rule, used by the lift
legs of the Metal, CUDA and Triton gates alike (:func:`corpus_refusals`): a lift-basis row
that reaches no predicate verdict, and no other disposition, and that the manifest retires is
refused by name with the manifest's verdict as its reason and counted out of the rows the
engine was asked about, never scored as unmeasured. A retired row that does reach a verdict
is scored like any other. The manifest is derived where the census is read, from the corpus
legs' row files under ``results/``, by the manifest's own rule
(:func:`corpus_retirements`); with a row file absent nothing is retired, and a manifest
that cannot be derived is named (``unreadable``) and retires nothing, so such a row stays
unmeasured and fails its leg. A refusal never passes a leg by itself: every leg still
requires a row it asked about, as above.

Standard library and ``meep_identity_corpus`` (itself standard library only). This file is
in the closures of the nine Metal lifting gates and of ``gate_metal_complex_fused_hd_pair``,
and of the CUDA and Triton lift legs that read a census; what a certification round does
with it (the per-cell preflight) lives in ``rounds/1.0/metal_round.py``, which is in none.
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

#: What the rows of a lift record need between them, beyond the certification
#: environment's MEEP and this package. A lift record that is not listed needs nothing
#: more. The two listed records are the 2026-08-09 corpus campaign's side legs: ``_gdsii``
#: lifted coupler.py and ring_gds.py, and ``_sigma`` lifted mie_scattering.py,
#: differential_cross_section.py and waveguide_crossing.py.
LIFT_RECORD_NEEDS: Dict[str, Tuple[str, ...]] = {
    "corpus_final_numpy_2026-08-09_monitorfix_gdsii": ("gdstk",),
    "corpus_final_numpy_2026-08-09_monitorfix_sigma": ("PyMieScatt", "nlopt", "autograd"),
}

#: What each row of those records needs itself, by label, read off the MEEP 1.35.0
#: corpus's own text: coupler.py reads its geometry from a GDSII file with gdstk, and
#: ring_gds.py writes its layout with gdstk and reads it back the same way (neither calls
#: MEEP's GDSII functions, which 1.35.0 deprecates, nor imports gdspy); mie_scattering.py and
#: differential_cross_section.py import PyMieScatt; waveguide_crossing.py imports nlopt
#: and autograd. So a host lacking nlopt counts out waveguide_crossing.py alone, not the
#: two PyMieScatt rows beside it.
LIFT_ROW_NEEDS: Dict[str, Tuple[str, ...]] = {
    "examples:coupler.py": ("gdstk",),
    "examples:ring_gds.py": ("gdstk",),
    "examples:mie_scattering.py": ("PyMieScatt",),
    "examples:differential_cross_section.py": ("PyMieScatt",),
    "examples:waveguide_crossing.py": ("nlopt", "autograd"),
}

#: How each need is probed: ``("module", NAME)`` imports NAME; ``("meep", NAME)`` calls
#: ``meep.NAME()`` and requires a true answer, for a MEEP build feature (such as
#: ``meep.with_libGDSII()``, MEEP's own report of whether it was built against libGDSII).
#: No row of the MEEP 1.35.0 corpus needs a build feature.
NEEDS: Dict[str, Tuple[str, str]] = {
    "gdstk": ("module", "gdstk"),
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


def _neutral_ids():
    """The record spellings of machine stamps: ``neutral_ids.py`` beside this file."""
    try:
        import neutral_ids  # noqa: PLC0415
    except ImportError:  # imported as a package module, not from parity/meep_gpu
        from parity.meep_gpu import neutral_ids  # noqa: PLC0415
    return neutral_ids


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


def row_needs(label: str, lift_record: Optional[str]) -> List[str]:
    """What the row ``label`` of ``lift_record`` needs: nothing when its record declares
    nothing; else what :data:`LIFT_ROW_NEEDS` names for it, and a row it does not name
    needs everything its record declares, so it is never launched without one of them."""
    declared = LIFT_RECORD_NEEDS.get(lift_record or "", ())
    if not declared:
        return []
    return list(LIFT_ROW_NEEDS.get(label, declared))


def row_fields(record: Mapping[str, Any], seam_row: Mapping[str, Any]) -> Dict[str, Any]:
    """The interpreter and capability fields of one lift row, merged in by ``lift_basis``.

    ``interpreter`` is the one the child runs under (this process's); the census's
    recorded path is ``recorded_interpreter`` and is never launched. Both are written
    under their roots' labels (``neutral_ids.path``): the rows land in gate artifacts, and
    a census cut before 1.0 recorded the absolute path, which is forwarded neutral."""
    neutral = _neutral_ids()
    lift_record = record.get("lift_record") or seam_row.get("lift_record")
    label = seam_row.get("label") or f"{record.get('leg')}:{record.get('row')}"
    return {
        "interpreter": neutral.path(sys.executable),
        "recorded_interpreter": neutral.path(record.get("interpreter") or record.get("python")
                                             or seam_row.get("interpreter")),
        "lift_record": lift_record,
        "lift_needs": row_needs(label, lift_record),
    }


def probe(needs: Iterable[str], environment: Optional[Mapping[str, str]] = None,
          cwd: Optional[Path] = None, timeout: float = 300.0,
          executable: Optional[str] = None) -> Dict[str, Any]:
    """Probe ``needs`` once, in a child under ``executable`` (this interpreter).

    Returns ``{"interpreter", "needs": {need: None if present, else why absent},
    "unreadable": None or why the answer could not be read}``. The child keeps MEEP and
    its MPI out of the calling process. What is returned is what a gate and the round
    record: the interpreter under its root's label and every reason through
    ``neutral_ids.text`` (an answer can name a path); the child runs ``executable`` as
    given."""
    neutral = _neutral_ids()
    interpreter = executable or sys.executable
    wanted = sorted(set(needs))
    out: Dict[str, Any] = {"interpreter": neutral.path(interpreter), "needs": {},
                           "unreadable": None}
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
        out["unreadable"] = neutral.text(f"the capability probe under {interpreter} did not "
                                         f"finish: {type(exc).__name__}: {exc}")
        return out
    lines = [line[len(PROBE_PREFIX):] for line in (done.stdout or "").splitlines()
             if line.startswith(PROBE_PREFIX)]
    try:
        answers = json.loads(lines[-1])
    except (IndexError, ValueError):
        answers = None
    if not isinstance(answers, dict) or sorted(answers) != wanted:
        # The stderr is neutral before it is cut: a tail cut first can begin in a path.
        out["unreadable"] = neutral.text(
            f"the capability probe under {interpreter} printed no answer for {wanted} "
            f"(exit {done.returncode}): ") + neutral.text((done.stderr or "").strip())[-300:]
        return out
    out["needs"] = {need: neutral.text(answers[need]) for need in wanted}
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


def corpus_label(row: Mapping[str, Any]) -> str:
    """The identity corpus's label for a lift-basis row: ``ex.<script stem>`` for an
    examples row, ``t.<module stem>.<row>`` for a tests row. The row name is the case name
    the corpus records, the expanded one for a parametrised case the census lifted under
    its decorated name."""
    if str(row.get("leg") or "").split("/", 1)[0] == "examples":
        return f"ex.{Path(str(row.get('row'))).stem}"
    return f"t.{Path(str(row.get('module') or '')).stem}.{row.get('row')}"


def corpus_retirements(results: Path) -> Dict[str, Any]:
    """The rows the identity corpus's manifest retires, derived from the corpus legs' row
    files under ``results`` by ``meep_identity_corpus.manifest_rows``.

    ``{"retired": {corpus label: its not_cases entry}, "inputs_sha256": {file: sha256},
    "absent": [row files not under results], "unreadable": None or why}``. Nothing is
    retired when a row file is absent or the manifest cannot be derived."""
    out: Dict[str, Any] = {"retired": {}, "inputs_sha256": {}, "absent": [],
                           "unreadable": None}
    try:
        import meep_identity_corpus as corpus  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 -- named, never raised
        out["unreadable"] = (f"meep_identity_corpus did not import: "
                             f"{type(exc).__name__}: {exc}")[:600]
        return out
    needed = ([f"{directory}/{corpus.PARITY_FILE}" for directory in corpus.LEGS.values()]
              + [f"{corpus.LEGS[leg]}/{corpus.LIFT_FILE}" for leg in corpus.RESURVEYS])
    out["absent"] = [rel for rel in needed if not (Path(results) / rel).is_file()]
    if out["absent"]:
        return out
    try:
        document = corpus.manifest_rows(Path(results))
    except Exception as exc:  # noqa: BLE001 -- named, never raised
        # Named without the host's path to the results tree: the reason lands in a gate artifact.
        out["unreadable"] = (f"the identity corpus's manifest could not be derived from the "
                             f"corpus legs under results/: {type(exc).__name__}: "
                             f"{str(exc).replace(str(results), 'results')}")[:600]
        return out
    out["inputs_sha256"] = dict(document.get("inputs_sha256") or {})
    out["retired"] = {str(entry["label"]): dict(entry)
                      for entry in document.get("not_cases") or ()
                      if entry.get("retires") and entry.get("label")}
    return out


def retirement_reason(entry: Mapping[str, Any]) -> str:
    """A retired row's refusal, in the manifest's words."""
    return (f"retired by the identity corpus's manifest: its re-survey {entry.get('leg')} "
            f"recorded {entry.get('parity')} ({entry.get('error')}) for the row the "
            f"{entry.get('retires')} leg had measured")


def corpus_refusals(results: Path, records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Which of ``records`` the identity corpus's manifest retires, each refused by name.

    ``records`` are the rows of a lift leg that reached no predicate verdict and no other
    disposition, each carrying its basis row's ``label``, ``leg``, ``row`` and ``module``.
    Returns ``{"refused": {label: reason}, "basis": ...}``: ``basis`` is ``None`` when no
    record was given (nothing is read), else what :func:`corpus_retirements` read
    (``inputs_sha256``, ``absent``, ``unreadable``) and the labels it retires."""
    candidates = list(records)
    if not candidates:
        return {"refused": {}, "basis": None}
    retirements = corpus_retirements(results)
    refused: Dict[str, str] = {}
    for record in candidates:
        entry = retirements["retired"].get(corpus_label(record))
        if entry is not None:
            refused[str(record["label"])] = retirement_reason(entry)
    basis = {key: retirements[key] for key in ("inputs_sha256", "absent", "unreadable")}
    basis["retired_labels"] = sorted(retirements["retired"])
    return {"refused": refused, "basis": basis}
