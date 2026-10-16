#!/usr/bin/env python
# SPDX-License-Identifier: GPL-2.0-or-later
"""The bit-parity coverage board: per kernel table, how many modules have direct parity
with the aligned array path at each level, and per array-path family, identity with MEEP.

    python parity/meep_gpu/build_parity_board.py manifest
    python parity/meep_gpu/build_parity_board.py outcomes --junit A.xml [B.xml ...] \\
        --manifest-before SHA [--out parity/meep_gpu/parity_board/outcomes.json]
    python parity/meep_gpu/build_parity_board.py build [--outcomes FILE] [--out-dir DIR]
    python parity/meep_gpu/build_parity_board.py check [--board FILE]

It is the bit-parity counterpart of the fusion boards (``build_fusion_matrix.py``,
``build_cuda_fusion_matrix.py``, ``build_triton_fusion_matrix.py``). It reads, and never
edits, the tree.

ROWS. Every module of the three kernel tables' step lists
(``results/bit_parity_0_10/design/KERNELS.md`` section 8, with ``DESIGN.md`` D27's moves),
held as data in ``parity_board_rows.json``, plus one row per array-path family judged
against MEEP. A row whose file is absent from the tree is refused by name; a module of a
table directory that no step list names is listed by name.

COLUMNS, each derived only from records in the tree, never typed:

(a) **B select landed**: the device text the module's builder EMITS
    (:mod:`parity_board_emit`: the CUDA source ``cp.RawKernel`` is handed, the Metal MSL
    after template splicing, the Triton ``@jit`` program with its helpers inlined; no
    device) reads the chunk record (``chunk_map.chunk_bounds_record``), and every
    field-word PML form in it sits in a per-cell select. Not landed: a unit that does not
    compile (a helper called with an argument count no definition of it takes), a kernel
    unit that spells a PML form and reads no record, or one that steps a field-word form
    outside every select. Refused: neither form nor read, or an emitter that raises on
    this tree (named). Not applicable to host rows, to the unchanged CUDA rows, and to
    the no-absorber rows (no PML, so no select).
(b) **MEEP-order A form landed**: the emitted text spells the curl stencil and the PML
    constitutive accumulation in MEEP's grouping and spells the 0.9.2 grouping nowhere.
    Not landed: the 0.9.2 grouping survives in any unit (a served path still forms it),
    or a unit does not compile. Refused: neither spelling, or no emitted text.
    A module with no emit route (a text transform, a host scan, a null arm; named in the
    recipes, ``parity_board_emit.json``) is read off its OWN source text instead, the
    v1 witness, and its cells say so.
(c) **Host word-identical to the array path**: the outcomes of the module's own test
    files in a recorded run (``outcomes.json``, whose subject manifest must equal the
    tree's), and whether any of their ids is listed in
    ``tools/ci/pending_certification.txt`` in a block that names a kernel lane. A
    module's test files are those of its table (``cuda_kernels/test_*.py``,
    ``test_triton_*.py``, ``test_metal_*.py``) whose name names it, and those that
    reference it in code and name it in their docstring (:func:`test_files_by_module`).
    Of those, a file is CREDITED only where it executes the module's arithmetic
    (:func:`executes_arithmetic`): on CUDA it runs the emitted device text through the host
    interpreter of ``test_root_chunk_forms.py``; on Triton and Metal it compares the
    arithmetic with ``meep_gpu.stepping`` (it imports it). A file that tests a lift, a
    predicate, a compile, a seam order or a census measures no arithmetic, and a module
    with no credited file is refused. The rule reads the file's own imports: a comparison
    reached only through a helper is not credited, and a file that imports ``stepping``
    for any purpose is (an upper bound).
    Identical: (a) and (b) are not gaps (a level above them: a module that still spells
    the 0.9.2 grouping, or steps a PML form without the record, is not word-identical to
    the aligned array path whatever its tests say), at least one test passed, no id is
    listed under a kernel lane, and every failure is listed in a block that names none (a
    record waiting on the certification round). Refused: no test file, no recorded run,
    or no test passed.
(d) **Device word-identical**: a gate record in the tree -- a ledger entry of
    ``meep_gpu/<table>_kernels/fingerprints.json`` that pins the module's file -- with a
    PASS status, a live run (``fastpath.live_capabilities``), the module's pinned bytes
    equal to the tree's, the live array-path oracle in its ``array_path_oracle`` leaf,
    and a run whose recorded subnormal policy covers the baseline. Refused: no entry pins
    the module's file (``driver_dispatch`` is the route record, not a module weld).
(d-dev) **Development evidence, NOT certification**: an integrator's re-run of a
    lane's device legs, recorded in ``parity_board/dev_evidence.json`` by
    ``record_dev_evidence.py``, per subnormal baseline: words compared and words
    differing. Counted only while the record's subject -- the module file's bytes, the
    digest of its emitted device text and the array-path oracle -- is the live one, so
    an edit to the module, to the base it lifts, or to the array path drops it. Columns
    (d) and (e) never read it.
(e) **CERTIFIED**: (d), and every file the entry pins equals the tree's, and the run names
    its hardware (a capability key and a host). The live oracle is computed here
    (``array_path_oracle.run``: NumPy, no MEEP, well under a second).

The subnormal baselines are owner ruling R13: keep and flush on the NVIDIA tables and on
the array path, flush only on Metal.

ARRAY PATH AGAINST MEEP, per family: the Tier A/C records of
``parity/meep_gpu/fixtures/meep_identity/meep_identity.json``, per subnormal set. A case
exercises a family by its capsule's construction log (the predicates are data in the rows
file). Identical: the set was minted on the live array-path oracle, and every exercising
case's Tier C stored and reader words and its Tier A digests are equal, with both planted
flips caught. Refused: no minted set for the baseline, a set minted on another array path,
or no minted case exercising the family.

EVERY COUNT IS N / D. D is the rows a column applies to and that are measured; refused
rows are counted, and named, beside it -- never as a gap and never as a pass.

THE SUBJECT. ``outcomes`` refuses unless the tree it is handed hashes, after the run, to
the manifest taken before it (``--manifest-before``). ``build`` refuses an outcomes record
whose manifest differs from the tree's: "the outcomes subject and the working tree
disagree". The board records the manifest of everything it read, and ``check`` refuses a
board whose manifest differs from the tree's: "the board subject and the working tree
disagree".

Host-only. ``build`` imports NumPy and the package's array path (the oracle run and the
capsule reader) and emits each table's device text in a child process that can import
no device library; ``manifest``, ``outcomes`` and ``check`` import nothing beyond the
standard library.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import re
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

_HERE = Path(__file__).resolve().parent


def _find_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "parity" / "meep_gpu").is_dir() and (candidate / "meep_gpu").is_dir():
            return candidate
    raise SystemExit(f"cannot locate the repository root above {start}")


REPO_ROOT = _find_root(_HERE)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import parity_board_emit as emit_witness  # noqa: E402
import parity_board_text as text_witness  # noqa: E402

SCHEMA = "meep_gpu.parity_board"
OUTCOMES_SCHEMA = "meep_gpu.parity_board_outcomes"
DEV_EVIDENCE_SCHEMA = "meep_gpu.parity_board_dev_evidence"
#: The board's own document version. The outcomes record and the development-evidence
#: record keep their own (1): their shapes did not change.
VERSION = 2
OUTCOMES_VERSION = 1
DEV_EVIDENCE_VERSION = 1
TOOL = "parity/meep_gpu/build_parity_board.py"
ROWS_FILE = "parity/meep_gpu/parity_board_rows.json"
BOARD_DIR = "parity/meep_gpu/parity_board"
OUTCOMES_FILE = f"{BOARD_DIR}/outcomes.json"
#: Development evidence, written by integrators (``record_dev_evidence.py``): NOT
#: certification. Column (d-dev) reads it; columns (d) and (e) never do.
DEV_EVIDENCE_FILE = f"{BOARD_DIR}/dev_evidence.json"
PENDING_FILE = "tools/ci/pending_certification.txt"
MEEP_IDENTITY_FILE = "parity/meep_gpu/fixtures/meep_identity/meep_identity.json"
LEDGERS = {"cuda": "meep_gpu/cuda_kernels/fingerprints.json",
           "triton": "meep_gpu/triton_kernels/fingerprints.json",
           "metal": "meep_gpu/metal_kernels/fingerprints.json"}
TABLES = ("cuda", "triton", "metal")
TABLE_NAMES = {"cuda": "CUDA", "triton": "Triton", "metal": "Metal"}
#: Owner ruling R13: which subnormal baselines each table is certified under.
BASELINES = {"cuda": ("keep", "flush"), "triton": ("keep", "flush"), "metal": ("flush",),
             "array_path": ("keep", "flush")}
#: The test-file prefix of each table's module tests.
TEST_PREFIX = {"cuda": "meep_gpu/cuda_kernels/test_", "triton": "meep_gpu/test_triton_",
               "metal": "meep_gpu/test_metal_"}
#: The route record of each ledger: it pins every dispatched module and is not a weld.
ROUTE_RECORD = "driver_dispatch"
COLUMNS = ("b_select", "a_form", "host_identical", "device_identical", "dev_evidence",
           "certified")
COLUMN_TITLES = {"b_select": "(a) B select landed", "a_form": "(b) MEEP-order A form",
                 "host_identical": "(c) host word-identical",
                 "device_identical": "(d) device word-identical",
                 "dev_evidence": "(d-dev) device development evidence, NOT certification",
                 "certified": "(e) CERTIFIED"}
#: Row kinds a column does not apply to.
NOT_APPLICABLE = {
    "b_select": {"host": "host module: no device form", "unchanged": "no update form",
                 "a_only": "no absorber: no B select"},
    "a_form": {"host": "host module: no device form", "unchanged": "no update form"},
    "host_identical": {"host": "host module: no device form", "unchanged": "no update form"},
    "device_identical": {"host": "host module: no device form", "unchanged": "no update form"},
    "dev_evidence": {"host": "host module: no device form", "unchanged": "no update form"},
    "certified": {"host": "host module: no device form", "unchanged": "no update form"},
}

#: The files the board's own code and outputs live in: never part of the outcomes subject,
#: so writing the board does not move the subject of the run it was built from.
OWN_FILES = ("parity/meep_gpu/build_parity_board.py", "parity/meep_gpu/parity_board_text.py",
             "parity/meep_gpu/test_build_parity_board.py", ROWS_FILE,
             "parity/meep_gpu/parity_board_emit.py", "parity/meep_gpu/parity_board_emit.json",
             "parity/meep_gpu/record_dev_evidence.py",
             "parity/meep_gpu/test_parity_board_emit.py")
OWN_PREFIXES = (BOARD_DIR + "/",)

_LANE_TOKEN = re.compile(r"\bK([CTM])(\d+)[a-z]*\b")
_TABLE_LETTER = {"C": "cuda", "T": "triton", "M": "metal"}
_HOST_STEP = {"cuda": "C0", "triton": "T0", "metal": "M0"}


class BoardRefusal(Exception):
    """A named reason the board, or one of its inputs, cannot be built."""


# ---------------------------------------------------------------------------
# The subject
# ---------------------------------------------------------------------------


def _is_own(rel: str) -> bool:
    return rel in OWN_FILES or any(rel.startswith(prefix) for prefix in OWN_PREFIXES)


def outcome_subject_files(root: Path) -> List[str]:
    """Every file a module test's outcome can depend on: the package, the harness's own
    files and fixtures, and the root test configuration. Bytecode caches, the evidence
    archive (``parity/meep_gpu/results/``) and the board's own files are excluded."""
    files: Set[str] = set()
    for base in ("meep_gpu",):
        for path in (root / base).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path.relative_to(root).as_posix())
    harness = root / "parity" / "meep_gpu"
    for path in harness.iterdir():
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    fixtures = harness / "fixtures"
    if fixtures.is_dir():
        for path in fixtures.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path.relative_to(root).as_posix())
    for name in ("conftest.py", "pyproject.toml", "parity/__init__.py",
                 "tools/ci/declared_resources.txt"):
        if (root / name).is_file():
            files.add(name)
    return sorted(rel for rel in files if not _is_own(rel)
                  and not rel.startswith("parity/meep_gpu/results/"))


def board_subject_files(root: Path) -> List[str]:
    """What ``build`` reads: the outcomes subject, the board's code, rows and emit
    recipes, the pending list, the outcomes record and the development-evidence record."""
    files = set(outcome_subject_files(root))
    for name in (*OWN_FILES, PENDING_FILE, OUTCOMES_FILE, DEV_EVIDENCE_FILE):
        if (root / name).is_file():
            files.add(name)
    return sorted(files)


def manifest(root: Path, files: Sequence[str]) -> Dict[str, Any]:
    """sha256 over sorted ``"<relpath>  <sha256>"`` lines (the census digest's form)."""
    lines = []
    for rel in files:
        digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        lines.append(f"{rel}  {digest}\n")
    text = "".join(sorted(lines))
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "files": len(files)}


def outcome_manifest(root: Path) -> Dict[str, Any]:
    return manifest(root, outcome_subject_files(root))


def board_manifest(root: Path) -> Dict[str, Any]:
    return manifest(root, board_subject_files(root))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def load_rows(root: Path) -> Dict[str, Any]:
    path = root / ROWS_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BoardRefusal(f"the rows file {ROWS_FILE} cannot be read: {error}") from error
    if document.get("schema") != "meep_gpu.parity_board_rows" or document.get("version") != 1:
        raise BoardRefusal(f"{ROWS_FILE} is not a version-1 meep_gpu.parity_board_rows document")
    for table in TABLES:
        modules = document.get("tables", {}).get(table, {}).get("modules")
        if not modules:
            raise BoardRefusal(f"{ROWS_FILE} names no module of the {table} table")
        names = [row["module"] for row in modules]
        if len(set(names)) != len(names):
            raise BoardRefusal(f"{ROWS_FILE}: a {table} module is named twice")
        for row in modules:
            if row.get("kind") not in ("host", "kernel", "a_only", "unchanged"):
                raise BoardRefusal(f"{ROWS_FILE}: {table} {row.get('module')} has kind "
                                   f"{row.get('kind')!r}")
            for verdict in COLUMNS:
                if verdict in row:
                    raise BoardRefusal(f"{ROWS_FILE}: {table} {row['module']} carries a "
                                       f"{verdict!r} value; the rows file holds structure, "
                                       f"never a verdict")
    if not document.get("families"):
        raise BoardRefusal(f"{ROWS_FILE} names no array-path family")
    return document


# ---------------------------------------------------------------------------
# (a) and (b): the device text the module compiles
# ---------------------------------------------------------------------------

#: The witness a cell was read with, named in the cell.
EMITTED = "emitted device text"
SOURCE_TEXT = "the module's own source text"


def source_text_columns(table: str, row: Mapping[str, Any], source: str) -> Dict[str, Any]:
    """(a) and (b) read off the module's OWN source text (the v1 witness). Used only for
    a module the emitter has no route for (a text transform, a host scan, a null arm)."""
    own = text_witness.own_text(source)
    if own.parse_error:
        refusal = own.parse_error
        return {"b_select": {"state": "refused", "why": refusal, "witness": SOURCE_TEXT},
                "a_form": {"state": "refused", "why": refusal, "witness": SOURCE_TEXT}}
    reads = text_witness.record_reads(table, own)
    markers = text_witness.pml_form_markers(table, own)
    forms = text_witness.form_counts(table, own).as_dict()
    if reads:
        b = {"state": "landed", "witness_detail": reads[:3]}
    elif markers:
        b = {"state": "gap", "why": "spells a PML form and reads no chunk record",
             "witness_detail": markers[:3]}
    else:
        b = {"state": "refused", "why": "no PML form and no chunk-record read in its own text"}
    if row["kind"] == "a_only" and reads:
        b["reads_record"] = True  # the presence word, read by an A-only row
    old = forms["old_curl"] + forms["old_constitutive"]
    new = forms["new_curl"] + forms["new_constitutive"]
    if new and not old:
        a = {"state": "landed", "sites": forms}
    elif old and not new:
        a = {"state": "gap", "why": "spells the 0.9.2 grouping and no MEEP grouping",
             "sites": forms}
    elif old and new:
        a = {"state": "refused", "why": "conflict: spells both groupings in its own text",
             "sites": forms}
    else:
        a = {"state": "refused", "why": "no A-form witness in its own text", "sites": forms}
    b["witness"] = a["witness"] = SOURCE_TEXT
    return {"b_select": b, "a_form": a}


def _sum_forms(units: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for unit in units:
        for key, value in unit["forms"].items():
            total[key] = total.get(key, 0) + int(value)
    return total


def emitted_text_columns(table: str, row: Mapping[str, Any],
                         record: Mapping[str, Any]) -> Dict[str, Any]:
    """(a) and (b) read off the device text the module's builder emits
    (``parity_board_emit.py``): every unit is one compiled text.

    (a) GAP when a unit does not compile (a helper called with an argument count none
    of its definitions takes: a call site the lift did not re-cut), when a kernel unit
    spells a PML form and reads no chunk record, or when a kernel unit reads the record
    but steps a field-word PML form outside every per-cell select. LANDED when no unit
    is a gap and a unit reads the record. REFUSED otherwise (no PML form, no read).

    (b) GAP when a unit does not compile or the 0.9.2 grouping survives in any unit
    (whatever else the text spells: a served path still forms it). LANDED when MEEP's
    grouping is spelt and the package's is spelt nowhere. REFUSED otherwise.
    """
    base = {"witness": EMITTED, "route": record.get("route"),
            "units": len(record.get("units", ())), "notes": record.get("notes", [])[:3]}
    if record.get("witness") == "refused":
        refusal = {"state": "refused", "why": f"emission refused: {record['why']}",
                   "witness": EMITTED}
        return {"b_select": dict(refusal), "a_form": dict(refusal)}
    units = record["units"]
    kernels = [u for u in units if u.get("kernel", True)]
    labels = [u["label"] for u in units]
    broken = [(u["label"], u["arity"][0]) for u in units if u.get("arity")]
    unread = [u["label"] for u in kernels if u["markers"] and not u["reads"]]
    unselected = [(u["label"], u["select"]) for u in kernels if u["select"]["unselected"]]
    reading = [u["label"] for u in units if u["reads"]]
    if broken:
        stale = "a call site the lift did not re-cut; " if broken[0][1].startswith("calls ") else ""
        b = {"state": "gap", "why": f"the emitted text does not compile: {broken[0][1]} "
                                    f"({stale}{len(broken)} of {len(units)} units, e.g. "
                                    f"{broken[0][0]})"}
    elif unread:
        b = {"state": "gap", "why": f"spells a PML form and reads no chunk record "
                                    f"({len(unread)} of {len(kernels)} kernel units, e.g. "
                                    f"{unread[0]})"}
    elif unselected:
        kinds = sorted({kind for _label, sites in unselected for kind in sites["examples"]})
        count = sum(sites["unselected"] for _label, sites in unselected)
        lead = "reads the chunk record but steps" if reading else "reads no chunk record and steps"
        b = {"state": "gap", "why": f"{lead} {count} field-word PML form site(s) outside "
                                    f"every per-cell select ({', '.join(kinds)}; "
                                    f"{len(unselected)} of {len(kernels)} kernel units, e.g. "
                                    f"{unselected[0][0]})"}
    elif reading:
        b = {"state": "landed", "reading_units": len(reading), "e.g.": reading[0]}
    else:
        b = {"state": "refused", "why": "no PML form and no chunk-record read in its "
                                        "emitted text"}
    if row["kind"] == "a_only" and reading:
        b["reads_record"] = True
    forms = _sum_forms(units)
    old = forms["old_curl"] + forms["old_constitutive"]
    new = forms["new_curl"] + forms["new_constitutive"]
    if broken:
        a = {"state": "gap", "why": f"the emitted text does not compile: {broken[0][1]} "
                                    f"(e.g. {broken[0][0]})"}
    elif old:
        survivors = [u["label"] for u in units
                     if u["forms"]["old_curl"] + u["forms"]["old_constitutive"]]
        a = {"state": "gap", "why": f"the 0.9.2 grouping survives in the emitted text "
                                    f"({old} site(s) in {len(survivors)} of {len(units)} "
                                    f"units" + (f", beside {new} MEEP-order" if new else "")
                                    + f"; e.g. {survivors[0]})"}
    elif new:
        a = {"state": "landed"}
    else:
        a = {"state": "refused", "why": "no curl or constitutive A-form site in its "
                                        "emitted text"}
    a["sites"] = forms
    b.update(base)
    a.update(base)
    b["configurations"] = labels[:3]
    return {"b_select": b, "a_form": a}


def witness_columns(table: str, row: Mapping[str, Any], source: str,
                    record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The emitted witness, or the source-text witness where the module has no emit
    route (the cell names which, and why)."""
    if record is not None and record.get("witness") in ("emitted", "refused"):
        return emitted_text_columns(table, row, record)
    cells = source_text_columns(table, row, source)
    why = (record or {}).get("why", "the emitter did not run")
    for cell in cells.values():
        cell["witness"] = f"{SOURCE_TEXT} (no emit route: {why})"
    return cells


# ---------------------------------------------------------------------------
# (c): the module's tests, their outcomes and the pending list
# ---------------------------------------------------------------------------


def _referenced_modules(path: Path, package: str) -> Tuple[Set[str], str]:
    """The modules of ``package`` a test file references in code, and its docstring.

    A reference is an import by name (``from meep_gpu.cuda_kernels import x``, ``from .x
    import ...``, ``import meep_gpu.cuda_kernels.x``), a string naming the module's file
    (``HERE / "x.py"``, for a module that imports a device library at import), or a
    string that is exactly a module name (a test that loads its subjects by stem).
    """
    import ast  # noqa: PLC0415

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set(), ""
    docstring = ast.get_docstring(tree) or ""
    skip = {id(tree.body[0].value)} if docstring else set()
    file_package = ("meep_gpu" if path.parent.name == "meep_gpu"
                    else ".".join(path.parent.parts[-2:]))
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = file_package if node.level == 1 else "meep_gpu"
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if module == package:
                found.update(alias.name for alias in node.names)
            elif module.startswith(package + "."):
                found.add(module[len(package) + 1:].split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(package + "."):
                    found.add(alias.name[len(package) + 1:].split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            match = re.fullmatch(r"(?:.*/)?(\w+)(?:\.py)?", node.value)
            if match:
                found.add(match.group(1))
    return found, docstring


def _docstring_names(docstring: str, names: Iterable[str]) -> Set[str]:
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", docstring))
    return {name for name in names if name in words}


#: Suffixes a family-named test file leaves off its module's name
#: (``test_dispersive.py`` is ``dispersive_kernels``'s, ``test_triton_bfast.py``
#: ``bfast_curl``'s), and the case suffix it adds (``test_step_curl_pml_real.py``).
NAME_SUFFIXES = ("_kernels", "_curl", "_triton")
CASE_SUFFIXES = ("_pml_real",)


def named_owner(stem: str, modules: Sequence[str]) -> Optional[str]:
    """The module a test file's stem names, or None.

    ``stem`` names ``m`` when, with a trailing case suffix removed, it is ``m``, starts
    with ``m_``, or is ``m`` less one of :data:`NAME_SUFFIXES`; the longest such ``m``
    wins (``test_triton_no_pml_ade.py`` is ``no_pml_ade``'s, not ``no_pml``'s).
    """
    for suffix in CASE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for name in sorted(modules, key=len, reverse=True):
        if stem == name or stem.startswith(name + "_"):
            return name
        if any(name == stem + suffix for suffix in NAME_SUFFIXES):
            return name
    return None


def test_files_by_module(root: Path, table: str, modules: Sequence[str],
                         applicable: Optional[Set[str]] = None) -> Dict[str, List[str]]:
    """Each module's own test files, by two mechanical rules, in union:

    * NAMED: the test files of the table whose stem names the module
      (:func:`named_owner`). A named file belongs to its module only.
    * DOCUMENTED: a test file of the table that no module owns by name, whose code
      references the module (:func:`_referenced_modules`) and whose module docstring
      names it -- a table that tests several modules in one file
      (``test_root_chunk_forms.py`` names the four CUDA roots).

    ``applicable``: the rows a DOCUMENTED file may be attributed to (the rows with a
    device form); default every module.
    """
    prefix = TEST_PREFIX[table]
    directory = root / Path(prefix).parent
    head = Path(prefix).name
    package = {"cuda": "meep_gpu.cuda_kernels", "triton": "meep_gpu.triton_kernels",
               "metal": "meep_gpu.metal_kernels"}[table]
    targets = set(modules) if applicable is None else set(applicable)
    out: Dict[str, List[str]] = {name: [] for name in modules}
    unnamed: List[Path] = []
    for path in sorted(directory.glob(f"{head}*.py")):
        owner = named_owner(path.stem[len(head):], modules)
        if owner is None:
            unnamed.append(path)
        else:
            out[owner].append(path.relative_to(root).as_posix())
    references = {path: _referenced_modules(path, package) for path in unnamed}
    for name in sorted(targets):
        out[name].extend(path.relative_to(root).as_posix() for path in unnamed
                         if name in references[path][0]
                         and name in _docstring_names(references[path][1], [name]))
    return out


def module_tests(root: Path, table: str, modules: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    """:func:`test_files_by_module` over a table's rows: family files go to device-form rows."""
    names = [row["module"] for row in modules]
    applicable = {row["module"] for row in modules if row["kind"] in ("kernel", "a_only")}
    return test_files_by_module(root, table, names, applicable)


def test_rule(rel_files: Sequence[str], table: str, module: str,
              modules: Sequence[str]) -> Dict[str, List[str]]:
    """Which rule of :func:`test_files_by_module` found each file."""
    head = Path(TEST_PREFIX[table]).name
    out: Dict[str, List[str]] = {"named": [], "documented": []}
    for rel in rel_files:
        rule = "named" if named_owner(Path(rel).stem[len(head):], modules) == module else "documented"
        out[rule].append(Path(rel).name)
    return out


def classname_file(classname: str, known: Mapping[str, str]) -> Optional[str]:
    """The test file of a JUnit classname (``a.b.test_x`` or ``a.b.test_x.TestClass``)."""
    parts = classname.split(".")
    for end in range(len(parts), 0, -1):
        dotted = ".".join(parts[:end])
        if dotted in known:
            return known[dotted]
    return None


def read_junit(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """Every test case of the reports: identifier -> outcome (passed / failed / skipped)."""
    cases: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        try:
            root = ElementTree.parse(path).getroot()
        except (OSError, ElementTree.ParseError) as error:
            raise BoardRefusal(f"unreadable report {path}: {error}") from error
        found = list(root.iter("testcase"))
        if not found:
            raise BoardRefusal(f"empty report {path}: no test ran")
        for case in found:
            identifier = f"{case.get('classname', '')}::{case.get('name', '')}"
            if case.find("skipped") is not None:
                outcome = "skipped"
            elif case.find("failure") is not None or case.find("error") is not None:
                outcome = "failed"
            else:
                outcome = "passed"
            previous = cases.get(identifier)
            if previous and previous["outcome"] != outcome:
                raise BoardRefusal(f"{identifier} is reported twice with different outcomes "
                                   f"({previous['outcome']} in {previous['report']}, "
                                   f"{outcome} in {path.name})")
            cases[identifier] = {"outcome": outcome, "classname": case.get("classname", ""),
                                 "report": path.name}
    return cases


def test_file_of(root: Path, classname: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    """The tree's test file of a JUnit classname (``a.b.test_x`` or ``a.b.test_x.TestCase``)."""
    if classname not in cache:
        parts = classname.split(".")
        cache[classname] = next((rel for rel in ("/".join(parts[:end]) + ".py"
                                                 for end in range(len(parts), 0, -1))
                                 if (root / rel).is_file()), None)
    return cache[classname]


def build_outcomes(root: Path, junit: Sequence[Path], manifest_before: str,
                   note: str = "") -> Dict[str, Any]:
    """The outcomes record of one run on ``root``: per test file the reports name, the
    passed count, the failing ids and the skipped count, stamped with the subject."""
    live = outcome_manifest(root)
    if live["sha256"] != manifest_before:
        raise BoardRefusal(
            f"the outcomes subject and the working tree disagree: the run was started on "
            f"{manifest_before} and the tree now hashes to {live['sha256']} over "
            f"{live['files']} subject files. A file moved during the run, so its outcomes "
            f"describe no tree. Re-run the tests.")
    cases = read_junit(junit)
    cache: Dict[str, Optional[str]] = {}
    per_file: Dict[str, Dict[str, Any]] = {}
    unmapped = 0
    for identifier, case in sorted(cases.items()):
        rel = test_file_of(root, case["classname"], cache)
        if rel is None:
            unmapped += 1
            continue
        entry = per_file.setdefault(rel, {"passed": 0, "failed": [], "skipped": 0})
        if case["outcome"] == "passed":
            entry["passed"] += 1
        elif case["outcome"] == "skipped":
            entry["skipped"] += 1
        else:
            entry["failed"].append(identifier)
    rows = load_rows(root)
    wanted: Set[str] = set()
    for table in TABLES:
        for found in module_tests(root, table, rows["tables"][table]["modules"]).values():
            wanted.update(found)
    return {
        "schema": OUTCOMES_SCHEMA, "version": OUTCOMES_VERSION,
        "about": ("Outcomes of the test files in one recorded run, written only by "
                  "build_parity_board.py outcomes. The board refuses it once the tree no "
                  "longer hashes to its subject."),
        "subject_manifest_sha256": live["sha256"], "subject_files": live["files"],
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": {"python": platform.python_version(), "machine": platform.machine(),
                 "system": platform.system()},
        "note": note,
        "reports": [{"file": p.name, "sha256": _sha256(p)} for p in junit],
        "cases": len(cases), "cases_without_a_test_file": unmapped,
        "files": {rel: per_file[rel] for rel in sorted(per_file)},
        "files_not_run": sorted(wanted - set(per_file)),
    }


def read_outcomes(root: Path, path: Optional[Path]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The outcomes record, or the named reason it cannot be used."""
    path = path or (root / OUTCOMES_FILE)
    if not path.is_file():
        return None, f"no test outcomes supplied ({OUTCOMES_FILE} is absent)"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        return None, f"the outcomes record cannot be read: {error}"
    if record.get("schema") != OUTCOMES_SCHEMA or record.get("version") != OUTCOMES_VERSION:
        return None, "the outcomes record is not this board's schema"
    live = outcome_manifest(root)
    if record.get("subject_manifest_sha256") != live["sha256"]:
        raise BoardRefusal(
            f"the outcomes subject and the working tree disagree. The outcomes record "
            f"was measured on {record.get('subject_manifest_sha256')} over "
            f"{record.get('subject_files')} subject files; this tree hashes to "
            f"{live['sha256']} over {live['files']}. Column (c) would pair another tree's "
            f"test outcomes with this tree's modules. Re-run the module tests and "
            f"re-record the outcomes, or check out the tree they were measured on.")
    return record, None


def pending_blocks(root: Path) -> Dict[str, Dict[str, Any]]:
    """Every listed id of the pending list: the lanes its block names, by table and step.

    A BLOCK is a run of comment lines and the ids that follow it. A block NAMES a kernel
    lane when its comment carries a lane token ``K<table><step>[suffix]`` (``KC2``,
    ``KT10r``, ``KM1``) of a step that is not the table's host step: those ids wait on a
    kernel lane. A block that names none waits on the certification round.
    """
    path = root / PENDING_FILE
    if not path.is_file():
        raise BoardRefusal(f"{PENDING_FILE} is absent")
    listed: Dict[str, Dict[str, Any]] = {}
    comment: List[str] = []
    ids_seen = False
    block = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if ids_seen:
                comment, ids_seen = [], False
                block += 1
            comment.append(stripped)
            continue
        ids_seen = True
        lanes: Set[Tuple[str, str]] = set()
        for letter, number in _LANE_TOKEN.findall(" ".join(comment)):
            table = _TABLE_LETTER[letter]
            step = f"{letter}{int(number)}"
            if step != _HOST_STEP[table]:
                lanes.add((table, step))
        listed[stripped] = {"block": block, "lanes": sorted(lanes),
                            "first_line": comment[0][:160] if comment else ""}
    return listed


#: What a credited test file does, per table (:func:`executes_arithmetic`), as (c) says it.
ARITHMETIC_RULE = {
    "cuda": "runs the emitted device text through the host interpreter (test_root_chunk_forms)",
    "triton": "compares the arithmetic with meep_gpu.stepping",
    "metal": "compares the arithmetic with meep_gpu.stepping",
}


def executes_arithmetic(root: Path, table: str, rel: str) -> bool:
    """Whether a test file executes the arithmetic it is attributed for, read off its code.

    * CUDA: a host executes CUDA text only through the float32 interpreter of
      ``cuda_kernels/test_root_chunk_forms.py``: the file defines an ``Interpreter`` class
      (or one derived from one) or imports from that file.
    * Triton: a host executes no ``@jit`` program, so the measurement there is a NumPy
      transcription compared with the array path: the file imports ``meep_gpu.stepping``.
    * Metal: the same. A Metal file that runs a plan on the Apple GPU and compares its
      words with the array path imports ``stepping`` too; a file that checks a predicate,
      an emitted text or a compile on the device measures no arithmetic.
    The rule reads the file's own imports only (an upper bound: an import for any purpose
    counts). A file that cannot be read or parsed executes nothing.
    """
    import ast  # noqa: PLC0415

    try:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in ast.walk(tree):
        if table == "cuda":
            if isinstance(node, ast.ClassDef) and (
                    node.name == "Interpreter"
                    or any(isinstance(base, ast.Name) and base.id.endswith("Interpreter")
                           for base in node.bases)):
                return True
            if isinstance(node, ast.ImportFrom) and \
                    (node.module or "").split(".")[-1] == "test_root_chunk_forms":
                return True
            continue
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            inside = bool(node.level) or module == "meep_gpu" or module.startswith("meep_gpu.")
            if inside and module.split(".")[-1] == "stepping":
                return True
            if inside and module in ("meep_gpu", "") and \
                    any(alias.name == "stepping" for alias in node.names):
                return True
        if isinstance(node, ast.Import) and \
                any(alias.name == "meep_gpu.stepping" for alias in node.names):
            return True
    return False


def host_column(row: Mapping[str, Any], table: str, files: Sequence[str],
                outcomes: Optional[Mapping[str, Any]], unavailable: Optional[str],
                listed: Mapping[str, Mapping[str, Any]],
                modules: Sequence[str] = (),
                credited: Optional[Mapping[str, bool]] = None) -> Dict[str, Any]:
    """(c): the module's own test files in the recorded run, and the pending list.

    Identical: at least one test passed; no id of the files is listed in a block that
    names a kernel lane (red by contract until a lane re-transcribes); no failure is
    undeclared. A failure listed in a block that names no kernel lane waits on the
    certification round (a ledger or record test) and does not count against host
    identity; it is reported.
    """
    if not files:
        return {"state": "refused", "why": "no test file of the table names or documents "
                                           "the module"}
    not_credited: List[str] = []
    if credited is not None:
        not_credited = [Path(rel).name for rel in files if not credited.get(rel, False)]
        measuring = [rel for rel in files if credited.get(rel, False)]
        if not measuring:
            return {"state": "refused",
                    "why": f"none of its test files {ARITHMETIC_RULE[table]}: "
                           f"{', '.join(not_credited)} measure no arithmetic of it",
                    "test_files": test_rule(files, table, row["module"],
                                            modules or [row["module"]]),
                    "not_credited": not_credited}
        files = measuring
    rules = test_rule(files, table, row["module"], modules or [row["module"]])
    if outcomes is None:
        return {"state": "refused", "why": unavailable, "test_files": rules}
    passed = skipped = 0
    failed: List[str] = []
    ran = [rel for rel in files if rel in outcomes["files"]]
    for rel in ran:
        entry = outcomes["files"][rel]
        passed += entry["passed"]
        skipped += entry["skipped"]
        failed.extend(entry["failed"])
    if not ran:
        return {"state": "refused", "why": "its test files were not in the recorded run",
                "test_files": rules}
    known = {rel[:-3].replace("/", "."): rel for rel in files}
    mine = sorted(i for i in listed if classname_file(i.split("::", 1)[0], known) is not None)
    lane = [i for i in mine if listed[i]["lanes"]]
    certification = [i for i in mine if not listed[i]["lanes"]]
    own_step = row["step"]
    own = [i for i in lane if [table, own_step] in [list(x) for x in listed[i]["lanes"]]]
    undeclared = sorted(set(failed) - set(mine))
    lanes_named = sorted({"K" + step for i in lane for (_t, step) in listed[i]["lanes"]})
    detail = {"test_files": rules, "passed": passed, "failed": len(failed), "skipped": skipped,
              "listed_under_a_kernel_lane": len(lane), "of_which_name_its_own_lane": len(own),
              "lanes_named": lanes_named, "listed_for_certification": len(certification),
              "undeclared_failures": undeclared[:10]}
    if not_credited:
        detail["not_credited"] = not_credited
    if passed == 0:
        return {"state": "refused", "why": "no test of its files passed in the recorded run "
                                           "(every one skipped or failed)", **detail}
    if lane or undeclared:
        reasons = []
        if lane:
            reasons.append(f"{len(lane)} id(s) listed as pending under kernel lanes "
                           f"({', '.join(lanes_named)}; {len(own)} name its own lane K{own_step})")
        if undeclared:
            reasons.append(f"{len(undeclared)} undeclared failure(s)")
        return {"state": "gap", "why": "; ".join(reasons), "examples": (lane or undeclared)[:3],
                **detail}
    return {"state": "landed", **detail}


def levels_gate(cells: Mapping[str, Mapping[str, Any]], host: Dict[str, Any],
                kind: str) -> Dict[str, Any]:
    """(c) is a level above (a) and (b): a module whose own text still spells the 0.9.2
    grouping, or steps a PML form without reading the chunk record, is not word-identical
    to the aligned array path, whatever its tests say (a test that passes there checks a
    refusal, a structure, or a case the grouping does not reach). Such a module is a gap
    in (c), named by the column that holds it back; its test counts are kept."""
    held = [name for name in ("b_select", "a_form")
            if cells[name]["state"] == "gap" and kind not in NOT_APPLICABLE[name]]
    if not held:
        return host
    why = "; ".join(f"{COLUMN_TITLES[name]} is a gap ({cells[name]['why']})" for name in held)
    return {**host, "state": "gap", "why": why,
            **({"tests_state": host["state"]} if host["state"] != "gap" else {})}


# ---------------------------------------------------------------------------
# (d) and (e): the ledgers
# ---------------------------------------------------------------------------

#: The fields whose presence makes an entry bind bytes: ``fastpath.BOUND_FIELDS``,
#: restated so ``check`` and ``manifest`` import nothing; the test asserts the two equal.
BOUND_FIELDS = ("source_sha256", "source_sha256_at_recert", "probe", "probe_sha256",
                "adapter", "adapter_sha256")


def bound_digest(entry: Mapping[str, Any]) -> Optional[str]:
    """``fastpath.bound_digest``, transcribed (the test pins the two equal)."""
    bound = {name: entry[name] for name in BOUND_FIELDS if name in entry}
    if not bound:
        return None
    text = json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def live_runs(entry: Mapping[str, Any]) -> List[str]:
    """``fastpath.live_capabilities``, transcribed: the runs that still bind the bytes."""
    runs = entry.get("runs")
    current = bound_digest(entry)
    if not isinstance(runs, Mapping) or current is None:
        return []
    return sorted(key for key, run in runs.items()
                  if isinstance(run, Mapping) and run.get("bound_sha256") == current)


def policy_baselines(text: Any) -> Optional[Set[str]]:
    """The baselines a run's recorded ``subnormal_policy`` names, or None if it names none.

    A policy string names ``flush`` when it contains the word, and ``keep`` when it
    contains it outside a refusal of it ("keep is refused", "keep is a plan-time
    refusal"). "NOT INSTALLED" names none.
    """
    if not isinstance(text, str) or "NOT INSTALLED" in text:
        return None
    lowered = text.lower()
    found: Set[str] = set()
    if re.search(r"flush", lowered):
        found.add("flush")
    without_refusals = re.sub(r"keep (is|was) (refused|a plan-time refusal)\w*", "", lowered)
    without_refusals = re.sub(r"cannot honou?r .?keep", "", without_refusals)
    if re.search(r"keep", without_refusals):
        found.add("keep")
    return found or None


def entries_pinning(ledger: Mapping[str, Any], path: str) -> List[str]:
    out = []
    for key, entry in ledger.items():
        if key == ROUTE_RECORD or not isinstance(entry, Mapping):
            continue
        pins = entry.get("source_sha256")
        modules = entry.get("kernel_module") or []
        if (isinstance(pins, Mapping) and path in pins) or path in modules:
            out.append(key)
    return sorted(out)


def device_columns(root: Path, table: str, row: Mapping[str, Any], ledger: Mapping[str, Any],
                   oracle: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """(d) and (e) per baseline of the table."""
    path = row["path"]
    keys = entries_pinning(ledger, path)
    out: Dict[str, Dict[str, Any]] = {}
    if not keys:
        refusal = {"state": "refused",
                   "why": "no ledger entry pins the module's file (its device text may be "
                          "pinned by digest elsewhere; the board does not attribute that)"}
        for baseline in BASELINES[table]:
            out[f"device_identical:{baseline}"] = dict(refusal)
            out[f"certified:{baseline}"] = dict(refusal)
        return out
    live_bytes = _sha256(root / path)
    for baseline in BASELINES[table]:
        best_d: Optional[Dict[str, Any]] = None
        best_e: Optional[Dict[str, Any]] = None
        failures_d: List[str] = []
        failures_e: List[str] = []
        for key in keys:
            entry = ledger[key]
            why_d = []
            if entry.get("status") != "PASS":
                why_d.append(f"status {entry.get('status')!r}")
            runs = live_runs(entry)
            if not runs:
                why_d.append("no live run (every run binds other bytes)")
            pins = entry.get("source_sha256") if isinstance(entry.get("source_sha256"), Mapping) else {}
            if pins.get(path) != live_bytes:
                why_d.append("the module's bytes moved since its run")
            leaf = entry.get("array_path_oracle") or {}
            if (leaf.get("fixture_set"), leaf.get("fingerprint_sha256")) != \
                    (oracle["fixture_set"], oracle["fingerprint_sha256"]):
                why_d.append(f"records array path {leaf.get('fixture_set')} "
                             f"{str(leaf.get('fingerprint_sha256'))[:16]}")
            covering = [k for k in runs
                        if baseline in (policy_baselines(entry["runs"][k].get("subnormal_policy")) or set())]
            if runs and not covering:
                why_d.append(f"no live run under {baseline}")
            if why_d:
                failures_d.append(f"{key}: " + "; ".join(why_d))
                failures_e.append(f"{key}: not device-identical")
                continue
            best_d = {"state": "landed", "entry": key, "runs": covering}
            moved = sorted(rel for rel, digest in pins.items()
                           if not (root / rel).is_file() or _sha256(root / rel) != digest)
            named = [k for k in covering if k and entry["runs"][k].get("host")]
            why_e = []
            if moved:
                why_e.append(f"{len(moved)} of {len(pins)} pinned files moved ({moved[:3]})")
            if not named:
                why_e.append("no live run names its hardware")
            if why_e:
                failures_e.append(f"{key}: " + "; ".join(why_e))
                continue
            best_e = {"state": "landed", "entry": key, "runs": named,
                      "host": entry["runs"][named[0]].get("host")}
        out[f"device_identical:{baseline}"] = best_d or {
            "state": "gap", "why": failures_d[0] if len(failures_d) == 1 else
            f"{len(failures_d)} pinning entries, none passes: {failures_d[0]}",
            "entries": keys}
        out[f"certified:{baseline}"] = best_e or {
            "state": "gap", "why": failures_e[0] if len(failures_e) == 1 else
            f"{len(failures_e)} pinning entries, none passes: {failures_e[0]}",
            "entries": keys}
    return out


# ---------------------------------------------------------------------------
# (d-dev): development evidence, NOT certification
# ---------------------------------------------------------------------------

#: The fields of one development-evidence record. ``record_dev_evidence.py`` is the
#: writer; the board refuses a record that lacks one.
DEV_EVIDENCE_FIELDS = ("table", "module", "backend", "subnormal_baseline", "words_compared",
                       "words_differing", "gate", "subject", "date", "host_class")
#: The backends a table's kernels run on.
TABLE_BACKENDS = {"cuda": ("cuda",), "triton": ("triton",), "metal": ("metal",)}


def module_subject(root: Path, row: Mapping[str, Any],
                   record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """What a development-evidence record binds for one module: the module file's bytes
    and the digest of the device text it emits. A lifter's compiled text moves with its
    base's even when its own file does not, so the file alone is not its subject."""
    path = root / row["path"]
    return {"module_sha256": _sha256(path) if path.is_file() else None,
            "emitted_sha256": (record or {}).get("emitted_sha256")}


def read_dev_evidence(root: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """The development-evidence records, or the named reason none can be read."""
    path = root / DEV_EVIDENCE_FILE
    if not path.is_file():
        return [], f"{DEV_EVIDENCE_FILE} is absent: no integrator has recorded device evidence"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise BoardRefusal(f"{DEV_EVIDENCE_FILE} cannot be read: {error}") from error
    if document.get("schema") != DEV_EVIDENCE_SCHEMA or \
            document.get("version") != DEV_EVIDENCE_VERSION:
        raise BoardRefusal(f"{DEV_EVIDENCE_FILE} is not a version-{DEV_EVIDENCE_VERSION} "
                           f"{DEV_EVIDENCE_SCHEMA} document")
    records = document.get("records") or []
    for index, record in enumerate(records):
        missing = [name for name in DEV_EVIDENCE_FIELDS if name not in record]
        if missing:
            raise BoardRefusal(f"{DEV_EVIDENCE_FILE} record {index} lacks {missing}: it was "
                               f"not written by record_dev_evidence.py")
    return records, None


def dev_evidence_columns(table: str, row: Mapping[str, Any], subject: Mapping[str, Any],
                         records: Sequence[Mapping[str, Any]], unavailable: Optional[str],
                         oracle: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """(d-dev) per baseline: an integrator's re-run of the module's device legs.

    COUNTED only when the record's subject is the module's LIVE subject (file bytes and
    emitted-text digest) and its array path is the live oracle, so a later edit to the
    module, to the base it lifts, or to the array path drops it without anyone noticing
    it should. LANDED: words compared and none differs. GAP: a current record with
    differing words. REFUSED: no current record (stale ones are named and counted).
    """
    out: Dict[str, Dict[str, Any]] = {}
    mine = [r for r in records if r["table"] == table and r["module"] == row["module"]]
    for baseline in BASELINES[table]:
        name = f"dev_evidence:{baseline}"
        candidates = [r for r in mine if r["subnormal_baseline"] == baseline]
        current, stale = [], []
        for record in candidates:
            bound = record.get("subject") or {}
            leaf = bound.get("array_path_oracle") or {}
            moved = []
            if bound.get("module_sha256") != subject["module_sha256"]:
                moved.append("the module's bytes")
            if bound.get("emitted_sha256") != subject["emitted_sha256"]:
                moved.append("its emitted device text")
            if (leaf.get("fixture_set"), leaf.get("fingerprint_sha256")) != \
                    (oracle["fixture_set"], oracle["fingerprint_sha256"]):
                moved.append("the array path")
            (stale if moved else current).append((record, moved))
        if not current:
            if stale:
                record, moved = stale[0]
                why = (f"{len(stale)} record(s) on another subject ({record['gate']} "
                       f"{record['date']}: {', '.join(moved)} moved since)")
            else:
                why = unavailable or "no development evidence recorded on this subject"
            out[name] = {"state": "refused", "why": why, "stale_records": len(stale)}
            continue
        differing = [r for r, _m in current if int(r["words_differing"]) != 0
                     or int(r["words_compared"]) <= 0]
        chosen = differing[0] if differing else current[0][0]
        cell = {"gate": chosen["gate"], "backend": chosen["backend"],
                "words_compared": int(chosen["words_compared"]),
                "words_differing": int(chosen["words_differing"]),
                "date": chosen["date"], "host_class": chosen["host_class"],
                "records": len(current), "not": "certification"}
        if differing:
            cell.update({"state": "gap", "why": f"{chosen['gate']}: "
                         f"{int(chosen['words_differing']):,} of "
                         f"{int(chosen['words_compared']):,} words differ"})
        else:
            cell["state"] = "landed"
        out[name] = cell
    return out


def ledger_oracle_census(ledgers: Mapping[str, Mapping[str, Any]],
                         oracle: Mapping[str, str]) -> Dict[str, Any]:
    """What each ledger's ``array_path_oracle`` leaves record, against the live oracle."""
    out: Dict[str, Any] = {}
    for table, ledger in ledgers.items():
        recorded: Dict[str, int] = {}
        binding = 0
        for key, entry in ledger.items():
            if not isinstance(entry, Mapping) or bound_digest(entry) is None:
                continue
            binding += 1
            leaf = entry.get("array_path_oracle") or {}
            label = f"{leaf.get('fixture_set')} {str(leaf.get('fingerprint_sha256'))[:16]}"
            recorded[label] = recorded.get(label, 0) + 1
        live_label = f"{oracle['fixture_set']} {oracle['fingerprint_sha256'][:16]}"
        out[table] = {"byte_binding_entries": binding, "recorded": recorded,
                      "recording_the_live_oracle": recorded.get(live_label, 0)}
    return out


# ---------------------------------------------------------------------------
# The array path against MEEP
# ---------------------------------------------------------------------------


def _decode(node: Any) -> Any:
    """The capsule's construction encoding, read as plain values (floats from their bits)."""
    import struct  # noqa: PLC0415

    if isinstance(node, Mapping):
        if "d" in node and len(node) == 1:
            return {key: _decode(value) for key, value in node["d"]}
        if "t" in node and len(node) == 1:
            return [_decode(value) for value in node["t"]]
        if "f" in node and len(node) == 1:
            return struct.unpack("<d", bytes.fromhex(node["f"]))[0]
        if "c" in node and len(node) == 1:
            real, imaginary = (struct.unpack("<d", bytes.fromhex(part))[0] for part in node["c"])
            return complex(real, imaginary)
        if "a" in node and len(node) == 1:
            return {"array": node["a"]}
        if "o" in node:
            return {"object": node["o"], **({"state": _decode(node["s"])} if "s" in node else {})}
        return {key: _decode(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_decode(value) for value in node]
    return node


def _nonzero(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_nonzero(v) for v in value)
    if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
        return value != 0
    return False


def _test(value: Any, predicate: Mapping[str, Any]) -> bool:
    test = predicate.get("test", "present")
    if test == "present":
        return value is not None
    if test == "true":
        return value is True
    if test == "false":
        return value is False or value is None
    if test == "nonzero":
        return _nonzero(value)
    if test == "nonempty":
        return bool(value)
    if test == "not_null":
        return value is not None
    if test == "not_in":
        return value is not None and value not in predicate.get("values", ())
    raise BoardRefusal(f"unknown family predicate test {test!r}")


def exercises(predicate: Mapping[str, Any], construction: Mapping[str, Any]) -> bool:
    """Does a case's decoded construction log satisfy a family predicate (rows-file data)?"""
    if "any" in predicate:
        return any(exercises(p, construction) for p in predicate["any"])
    if "all" in predicate:
        return all(exercises(p, construction) for p in predicate["all"])
    calls = construction["calls"]
    if "call" in predicate:
        return any(name == predicate["call"] for name, _ in calls)
    if "call_argument" in predicate:
        name, argument = predicate["call_argument"]
        return any(_test((args or {}).get(argument), predicate)
                   for call, args in calls if call == name)
    if "constructor" in predicate:
        return _test(construction["constructor"].get(predicate["constructor"]), predicate)
    if "source" in predicate:
        return any(_test(((args or {}).get("source_data") or {}).get(predicate["source"]), predicate)
                   for call, args in calls if call == "add_source")
    raise BoardRefusal(f"unknown family predicate {predicate}")


def case_constructions(root: Path, mode_set: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Each minted case's decoded construction log, read from its capsule."""
    sys.path.insert(0, str(root))
    from meep_gpu.capsule import read_capsule  # noqa: PLC0415

    fixtures = root / Path(MEEP_IDENTITY_FILE).parent
    out = {}
    for case, entry in sorted(mode_set.get("cases", {}).items()):
        capsule = read_capsule(fixtures / entry["files"]["capsule"]["file"])
        construction = capsule.document["construction"]
        out[case] = {"constructor": _decode(construction["constructor"]),
                     "calls": [(name, _decode(args)) for name, args in construction["calls"]]}
    return out


def case_identity(entry: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Whether a minted case's Tier C and Tier A records state identity, and its counts."""
    tier_c, tier_a = entry.get("tier_c") or {}, entry.get("tier_a") or {}
    stored, reader = tier_c.get("stored") or {}, tier_c.get("reader") or {}
    counts = {
        "tier_c_stored_words": stored.get("words"),
        "tier_c_stored_differing": stored.get("word_differing"),
        "tier_c_reader_words": reader.get("words"),
        "tier_c_reader_differing": reader.get("word_differing"),
        "tier_c_steps": len(tier_c.get("steps") or []), "n_steps": tier_c.get("n_steps"),
        "tier_c_flip_caught": (tier_c.get("planted_flip") or {}).get("caught"),
        "tier_a_digests": tier_a.get("digests"), "tier_a_stored_equal": tier_a.get("stored_equal"),
        "tier_a_reader_equal": tier_a.get("reader_equal"),
        "tier_a_flip_caught": (tier_a.get("planted_flip") or {}).get("caught"),
        "ghost_words_reported": (tier_c.get("ghost") or {}).get("words", 0),
        "ghost_words_differing_reported": (tier_c.get("ghost") or {}).get("word_differing", 0),
    }
    ok = (isinstance(stored.get("words"), int) and stored["words"] > 0
          and stored.get("word_differing") == 0 and stored.get("value_differing") == 0
          and isinstance(reader.get("words"), int) and reader["words"] > 0
          and reader.get("word_differing") == 0 and reader.get("value_differing") == 0
          and counts["tier_c_flip_caught"] is True
          and isinstance(tier_a.get("digests"), int) and tier_a["digests"] > 0
          and tier_a.get("stored_equal") == tier_a.get("digests")
          and tier_a.get("reader_equal") == tier_a.get("digests")
          and counts["tier_a_flip_caught"] is True)
    return ok, counts


def meep_rows(root: Path, families: Sequence[Mapping[str, Any]],
              oracle: Mapping[str, str]) -> Dict[str, Any]:
    path = root / MEEP_IDENTITY_FILE
    if not path.is_file():
        refusal = f"{MEEP_IDENTITY_FILE} is absent: no Tier A/C record in the tree"
        return {"families": [{"family": f["family"], **{b: {"state": "refused", "why": refusal}
                                                        for b in BASELINES["array_path"]}}
                             for f in families], "sets": {}}
    ledger = json.loads(path.read_text(encoding="utf-8"))
    sets = ledger.get("sets") or {}
    per_set: Dict[str, Any] = {}
    constructions: Dict[str, Dict[str, Any]] = {}
    for baseline in BASELINES["array_path"]:
        mode_set = sets.get(baseline)
        if mode_set is None:
            per_set[baseline] = {"refusal": f"the record holds no {baseline} set"}
            continue
        leaf = mode_set.get("array_path_oracle") or {}
        if (leaf.get("fixture_set"), leaf.get("fingerprint_sha256")) != \
                (oracle["fixture_set"], oracle["fingerprint_sha256"]):
            per_set[baseline] = {"refusal": (
                f"the {baseline} set was minted on array path {leaf.get('fixture_set')} "
                f"{str(leaf.get('fingerprint_sha256'))[:16]}; this tree steps "
                f"{oracle['fixture_set']} {oracle['fingerprint_sha256'][:16]}")}
            continue
        constructions[baseline] = case_constructions(root, mode_set)
        reference = mode_set.get("reference") or {}
        per_set[baseline] = {"cases": sorted(mode_set.get("cases", {})),
                             "reference": {"meep": reference.get("meep"),
                                           "system": reference.get("system"),
                                           "omp_num_threads": reference.get("omp_num_threads"),
                                           "fp_contract": (reference.get("build_record") or {}).get("fp_contract")},
                             "minted_utc": (mode_set.get("minted") or {}).get("utc")}
    rows = []
    for family in families:
        row: Dict[str, Any] = {"family": family["family"], "title": family["title"],
                               "work_packages": family["work_packages"]}
        for baseline in BASELINES["array_path"]:
            info = per_set[baseline]
            if "refusal" in info:
                row[baseline] = {"state": "refused", "why": info["refusal"]}
                continue
            cases = [c for c, construction in constructions[baseline].items()
                     if exercises(family["exercised_if"], construction)]
            if not cases:
                row[baseline] = {"state": "refused",
                                 "why": f"no minted {baseline} case exercises the family"}
                continue
            verdicts = {c: case_identity(sets[baseline]["cases"][c]) for c in cases}
            equal = [c for c, (ok, _counts) in verdicts.items() if ok]
            words = sum((counts["tier_c_stored_words"] or 0) for _ok, counts in verdicts.values())
            differing = sum((counts["tier_c_stored_differing"] or 0) for _ok, counts in verdicts.values())
            reader_words = sum((counts["tier_c_reader_words"] or 0) for _ok, counts in verdicts.values())
            reader_differing = sum((counts["tier_c_reader_differing"] or 0) for _ok, counts in verdicts.values())
            row[baseline] = {"state": "landed" if len(equal) == len(cases) else "gap",
                             "cases_identical": len(equal), "cases": len(cases),
                             "case_names": cases,
                             "stored_words_differing": differing, "stored_words": words,
                             "reader_words_differing": reader_differing,
                             "reader_words": reader_words,
                             "per_case": {c: counts for c, (_ok, counts) in verdicts.items()}}
        rows.append(row)
    return {"families": rows, "sets": per_set}


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------


def live_oracle(root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(root))
    from meep_gpu import array_path_oracle  # noqa: PLC0415

    result = array_path_oracle.run(progress=lambda line: print(f"  oracle {line}", flush=True))
    print(f"  oracle {result.fixture_set} {result.fingerprint_sha256[:16]} "
          f"({result.elapsed_seconds:.2f} s)", flush=True)
    # No elapsed time in the board: a re-cut on the same tree is the same board.
    return {"fixture_set": result.fixture_set, "fingerprint_sha256": result.fingerprint_sha256,
            "cases": len(result.case_sha256),
            "coverage_problems": array_path_oracle.coverage_problems(result.form_classes)}


def _tally(cells: Iterable[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    landed = gap = 0
    refused: Dict[str, List[str]] = {}
    not_applicable = 0
    for module, cell in cells:
        state = cell["state"]
        if state == "landed":
            landed += 1
        elif state == "gap":
            gap += 1
        elif state == "refused":
            refused.setdefault(cell.get("why", ""), []).append(module)
        else:
            not_applicable += 1
    measured = landed + gap
    return {"n": landed, "d": measured, "refused": sum(len(v) for v in refused.values()),
            "refused_by_reason": {why: sorted(names) for why, names in sorted(refused.items())},
            "not_applicable": not_applicable}


def build_board(root: Path, outcomes_path: Optional[Path] = None,
                oracle: Optional[Mapping[str, Any]] = None,
                emitted: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
    rows = load_rows(root)
    outcomes, unavailable = read_outcomes(root, outcomes_path)
    listed = pending_blocks(root)
    evidence, evidence_unavailable = read_dev_evidence(root)
    if oracle is None:
        oracle = live_oracle(root)
    if emitted is None:
        emitted = emit_witness.emitted_witnesses(root)
    ledgers = {table: json.loads((root / LEDGERS[table]).read_text(encoding="utf-8"))
               for table in TABLES}
    tables: Dict[str, Any] = {}
    for table in TABLES:
        spec = rows["tables"][table]
        modules = spec["modules"]
        names = [row["module"] for row in modules]
        tests = module_tests(root, table, modules)
        rendered = []
        for row in modules:
            out: Dict[str, Any] = {key: row[key] for key in ("module", "step", "lane", "kind", "path")}
            if "note" in row:
                out["note"] = row["note"]
            file = root / row["path"]
            cells: Dict[str, Dict[str, Any]] = {}
            record = emitted.get(table, {}).get(row["module"])
            if not file.is_file():
                refusal = {"state": "refused", "why": f"{row['path']} is absent from the tree"}
                for column in ("b_select", "a_form", "host_identical"):
                    cells[column] = dict(refusal)
                for baseline in BASELINES[table]:
                    cells[f"device_identical:{baseline}"] = dict(refusal)
                    cells[f"dev_evidence:{baseline}"] = dict(refusal)
                    cells[f"certified:{baseline}"] = dict(refusal)
            else:
                cells.update(witness_columns(table, row, file.read_text(encoding="utf-8"),
                                             record))
                own_tests = tests[row["module"]]
                credited = {rel: executes_arithmetic(root, table, rel) for rel in own_tests}
                cells["host_identical"] = levels_gate(
                    cells, host_column(row, table, own_tests, outcomes, unavailable, listed,
                                       names, credited), row["kind"])
                cells.update(device_columns(root, table, row, ledgers[table], oracle))
                subject = module_subject(root, row, record)
                out["subject"] = subject
                cells.update(dev_evidence_columns(table, row, subject, evidence,
                                                  evidence_unavailable, oracle))
            for name in list(cells):
                column = name.split(":", 1)[0]
                why = NOT_APPLICABLE.get(column, {}).get(row["kind"])
                if why:
                    cells[name] = {"state": "n/a", "why": why,
                                   **({"reads_record": True} if cells[name].get("reads_record") else {})}
            out["cells"] = cells
            rendered.append(out)
        column_names = ["b_select", "a_form", "host_identical"] + [
            f"{column}:{baseline}" for column in ("device_identical", "dev_evidence",
                                                  "certified")
            for baseline in BASELINES[table]]
        summary = {name: _tally((r["module"], r["cells"][name]) for r in rendered)
                   for name in column_names}
        listed_paths = {row["path"] for row in modules}
        unlisted = sorted(p.stem for p in (root / spec["module_dir"]).glob("*.py")
                          if not p.name.startswith("test_")
                          and p.relative_to(root).as_posix() not in listed_paths)
        kinds: Dict[str, int] = {}
        for row in modules:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        witnesses: Dict[str, int] = {}
        for item in rendered:
            if item["kind"] in emit_witness.EMITTED_KINDS:
                kind = item["cells"]["a_form"].get("witness", "")
                label = ("emitted" if kind == EMITTED else
                         "source text" if kind.startswith(SOURCE_TEXT) else "absent")
                witnesses[label] = witnesses.get(label, 0) + 1
        tables[table] = {"rows": len(modules), "kinds": kinds, "columns": column_names,
                         "summary": summary, "modules": rendered, "witnesses": witnesses,
                         "modules_no_step_list_names": unlisted}
    census = ledger_oracle_census(ledgers, oracle)
    meep = meep_rows(root, rows["families"], oracle)
    meep_summary = {baseline: _tally((r["family"], r[baseline]) for r in meep["families"])
                    for baseline in BASELINES["array_path"]}
    board = {
        "schema": SCHEMA, "version": VERSION, "tool": TOOL,
        "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subject": board_manifest(root),
        "outcomes": ({"subject_manifest_sha256": outcomes["subject_manifest_sha256"],
                      "recorded_utc": outcomes["recorded_utc"], "host": outcomes["host"],
                      "reports": outcomes["reports"], "note": outcomes.get("note", ""),
                      "files_run": len(outcomes["files"]),
                      "files_not_run": outcomes.get("files_not_run", [])}
                     if outcomes else {"refusal": unavailable}),
        "array_path_oracle": dict(oracle),
        "ledger_oracle_leaves": census,
        "certified_zero_because": certified_reason(census, oracle),
        "baselines": {k: list(v) for k, v in BASELINES.items()},
        "dev_evidence": {"file": DEV_EVIDENCE_FILE, "records": len(evidence),
                         "unavailable": evidence_unavailable,
                         "about": ("Development evidence: integrators re-run a lane's device "
                                   "legs at integration and record the words compared. It "
                                   "is NOT certification, and columns (d) and (e) never "
                                   "read it.")},
        "tables": tables,
        "array_path_vs_meep": {"summary": meep_summary, **meep},
    }
    return board


def certified_reason(census: Mapping[str, Any], oracle: Mapping[str, Any]) -> Dict[str, str]:
    """Why column (e) reads what it reads, per table, from the ledgers' oracle leaves."""
    out = {}
    live = f"{oracle['fixture_set']} {oracle['fingerprint_sha256'][:16]}"
    for table, info in census.items():
        if info["recording_the_live_oracle"] == 0:
            recorded = ", ".join(f"{label} ({count})" for label, count in sorted(info["recorded"].items()))
            out[table] = (f"0 of {info['byte_binding_entries']} byte-binding ledger entries "
                          f"record the live array path ({live}); they record {recorded}. "
                          f"Every weld was certified against another array path, so no "
                          f"module can be CERTIFIED until the certification round re-gates "
                          f"and the ledgers are re-minted by record_array_path_oracle.py.")
        else:
            out[table] = (f"{info['recording_the_live_oracle']} of {info['byte_binding_entries']} "
                          f"byte-binding entries record the live array path ({live}).")
    return out


# ---------------------------------------------------------------------------
# The markdown
# ---------------------------------------------------------------------------


def _cell(cell: Mapping[str, Any]) -> str:
    state = cell["state"]
    return {"landed": "yes", "gap": "no", "refused": "refused", "n/a": "n/a"}[state]


def _witness_label(cell: Mapping[str, Any]) -> str:
    witness = cell.get("witness", "")
    if witness == EMITTED:
        return "emitted"
    if witness.startswith(SOURCE_TEXT):
        return "source text"
    return "-"


def _fraction(tally: Mapping[str, Any]) -> str:
    text = f"{tally['n']} / {tally['d']}"
    if tally["refused"]:
        text += f" (+{tally['refused']} refused)"
    return text


def render_markdown(board: Mapping[str, Any]) -> str:
    lines = ["# Bit-parity coverage board", ""]
    lines.append(f"Built by `{board['tool']}` at {board['built_utc']} on subject "
                 f"`{board['subject']['sha256'][:16]}` ({board['subject']['files']} files). "
                 f"Generated: do not edit; re-run `build`.")
    lines.append("")
    oracle = board["array_path_oracle"]
    lines.append(f"Live array-path oracle: `{oracle['fixture_set']}` "
                 f"`{oracle['fingerprint_sha256'][:16]}` ({oracle.get('cases')} fixture cases).")
    outcomes = board["outcomes"]
    if "refusal" in outcomes:
        lines.append(f"Test outcomes: refused ({outcomes['refusal']}).")
    else:
        lines.append(f"Test outcomes: recorded {outcomes['recorded_utc']} on "
                     f"{outcomes['host']['system']} {outcomes['host']['machine']}, "
                     f"{outcomes['files_run']} test files run"
                     + (f"; {outcomes['note']}" if outcomes.get("note") else "") + ".")
    lines += ["", "Each count is N / D: D is the rows the column applies to and could "
              "measure; refused rows are counted beside it, never as a gap and never as a "
              "pass. Baselines (owner ruling R13): keep and flush on CUDA, Triton and the "
              "array path; flush only on Metal.", "",
              "Columns (a) and (b) read the device text each module's builder EMITS "
              "(`parity_board_emit.py`: the CUDA source `cp.RawKernel` is handed, the Metal "
              "MSL after template splicing, the Triton `@jit` program with its helpers "
              "inlined), with no device. A module with no emit route is read off its own "
              "source text instead, and its cell says so.", "",
              "Column (d-dev) is DEVELOPMENT EVIDENCE, NOT CERTIFICATION: an integrator's "
              "re-run of a lane's device legs (`record_dev_evidence.py`), counted only while "
              "its subject (the module's bytes, its emitted text, the array path) is the "
              "live one. Columns (d) and (e) never read it.", ""]
    lines += ["## Summary", ""]
    header = "| Table | Rows | " + " | ".join(
        ["(a) B select", "(b) MEEP-order A", "(c) host identical",
         "(d) device keep", "(d) device flush",
         "(d-dev) keep, NOT certification", "(d-dev) flush, NOT certification",
         "(e) CERTIFIED keep", "(e) CERTIFIED flush"]) + " |"
    lines += [header, "|" + "---|" * 11]
    for table in TABLES:
        info = board["tables"][table]
        summary = info["summary"]
        cells = [_fraction(summary["b_select"]), _fraction(summary["a_form"]),
                 _fraction(summary["host_identical"])]
        for column in ("device_identical", "dev_evidence", "certified"):
            for baseline in ("keep", "flush"):
                name = f"{column}:{baseline}"
                cells.append(_fraction(summary[name]) if name in summary else "n/a (R13)")
        lines.append(f"| {TABLE_NAMES[table]} | {info['rows']} | " + " | ".join(cells) + " |")
    lines += ["", "Witness of (a) and (b), per table (rows with a device form): "
              + "; ".join(f"{TABLE_NAMES[t]} " + ", ".join(
                  f"{n} {k}" for k, n in sorted(board['tables'][t].get('witnesses', {}).items()))
                  for t in TABLES) + "."]
    evidence = board.get("dev_evidence") or {}
    lines += ["", f"Development evidence: {evidence.get('records', 0)} record(s) in "
              f"`{evidence.get('file', DEV_EVIDENCE_FILE)}`"
              + (f" ({evidence['unavailable']})" if evidence.get("unavailable") else "") + "."]
    lines += ["", "Why (e) reads what it reads:", ""]
    for table in TABLES:
        lines.append(f"- {TABLE_NAMES[table]}: {board['certified_zero_because'][table]}")
    lines += ["", "## Array path against MEEP", ""]
    meep = board["array_path_vs_meep"]
    for baseline, info in meep["sets"].items():
        if "refusal" in info:
            lines.append(f"- {baseline}: refused ({info['refusal']}).")
        else:
            ref = info["reference"]
            lines.append(f"- {baseline}: {len(info['cases'])} minted cases "
                         f"({', '.join(info['cases'])}); reference MEEP "
                         f"{(ref.get('meep') or {}).get('version')} on {ref.get('system')}, "
                         f"contraction {ref.get('fp_contract')}, OMP {ref.get('omp_num_threads')}.")
    lines += ["", "| Family | WPs | keep | flush |", "|---|---|---|---|"]
    for row in meep["families"]:
        cells = []
        for baseline in ("keep", "flush"):
            cell = row[baseline]
            if cell["state"] in ("landed", "gap"):
                cells.append(f"{'identical' if cell['state'] == 'landed' else 'differs'}: "
                             f"{cell['cases_identical']} / {cell['cases']} cases; "
                             f"{cell['stored_words_differing']:,} / {cell['stored_words']:,} "
                             f"stored and {cell['reader_words_differing']:,} / "
                             f"{cell['reader_words']:,} reader words differ")
            else:
                cells.append(f"refused: {cell['why']}")
        lines.append(f"| {row['family']} ({row['title']}) | {row['work_packages']} | "
                     + " | ".join(cells) + " |")
    lines += [""]
    for baseline in ("keep", "flush"):
        lines.append(f"- {baseline}: {_fraction(meep['summary'][baseline])} families identical.")
    for table in TABLES:
        info = board["tables"][table]
        lines += ["", f"## {TABLE_NAMES[table]} ({info['rows']} rows: "
                  + ", ".join(f"{count} {kind}" for kind, count in sorted(info["kinds"].items()))
                  + ")", ""]
        refusals = []
        for name, tally in info["summary"].items():
            for why, modules in tally["refused_by_reason"].items():
                refusals.append(f"- {name}: refused, {why}: {len(modules)} "
                                f"({', '.join(modules[:12])}{', ...' if len(modules) > 12 else ''})")
        if refusals:
            lines += ["Refused cells, by reason:", ""] + refusals + [""]
        if info["modules_no_step_list_names"]:
            lines += [f"Modules of the table directory that no step list names: "
                      f"{', '.join(info['modules_no_step_list_names'])}.", ""]
        columns = info["columns"]
        lines.append("| Step | Module | Kind | Witness | "
                     + " | ".join(c.replace("_", " ") for c in columns) + " | Why not (first) |")
        lines.append("|" + "---|" * (len(columns) + 5))
        for row in info["modules"]:
            reasons = [f"{name.split(':')[0][:6]}: {row['cells'][name]['why']}" for name in columns
                       if row["cells"][name]["state"] == "gap"]
            lines.append(f"| {row['step']} | `{row['module']}` | {row['kind']} | "
                         f"{_witness_label(row['cells']['a_form'])} | "
                         + " | ".join(_cell(row["cells"][c]) for c in columns)
                         + f" | {reasons[0][:140] if reasons else ''} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def write_board(board: Mapping[str, Any], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "board.json"
    md_path = out_dir / "board.md"
    json_path.write_text(json.dumps(board, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(board), encoding="utf-8")
    return json_path, md_path


def check_board(root: Path, board_path: Path) -> Dict[str, Any]:
    board = json.loads(board_path.read_text(encoding="utf-8"))
    recorded = board.get("subject") or {}
    live = board_manifest(root)
    if recorded.get("sha256") != live["sha256"]:
        raise BoardRefusal(
            f"the board subject and the working tree disagree. The board was built on "
            f"{recorded.get('sha256')} over {recorded.get('files')} files; this tree hashes "
            f"to {live['sha256']} over {live['files']}. Its counts describe another tree. "
            f"Re-record the outcomes and rebuild the board.")
    return live


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help="print the outcomes subject manifest of the tree")
    p_out = sub.add_parser("outcomes", help="record a run's outcomes")
    p_out.add_argument("--junit", type=Path, nargs="+", required=True)
    p_out.add_argument("--manifest-before", required=True)
    p_out.add_argument("--note", default="")
    p_out.add_argument("--out", type=Path)
    p_build = sub.add_parser("build", help="build the board")
    p_build.add_argument("--outcomes", type=Path)
    p_build.add_argument("--out-dir", type=Path)
    p_check = sub.add_parser("check", help="refuse a board whose subject is not this tree")
    p_check.add_argument("--board", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        if arguments.command == "manifest":
            print(outcome_manifest(root)["sha256"])
            return 0
        if arguments.command == "outcomes":
            record = build_outcomes(root, arguments.junit, arguments.manifest_before, arguments.note)
            out = arguments.out or (root / OUTCOMES_FILE)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(f"outcomes: {len(record['files'])} test files run, "
                  f"{len(record['files_not_run'])} not run; subject "
                  f"{record['subject_manifest_sha256'][:16]} -> {out}", flush=True)
            return 0
        if arguments.command == "build":
            board = build_board(root, arguments.outcomes)
            json_path, md_path = write_board(board, arguments.out_dir or (root / BOARD_DIR))
            for table in TABLES:
                summary = board["tables"][table]["summary"]
                print(f"{TABLE_NAMES[table]}: " + "; ".join(
                    f"{name} {_fraction(tally)}" for name, tally in summary.items()), flush=True)
            for baseline, tally in board["array_path_vs_meep"]["summary"].items():
                print(f"array path vs MEEP {baseline}: {_fraction(tally)}", flush=True)
            print(f"wrote {json_path} and {md_path}", flush=True)
            return 0
        if arguments.command == "check":
            live = check_board(root, arguments.board or (root / BOARD_DIR / "board.json"))
            print(f"the board's subject is this tree ({live['sha256'][:16]}, {live['files']} files)")
            return 0
    except BoardRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr, flush=True)
        return 3
    return 2


if __name__ == "__main__":
    sys.exit(main())
