#!/usr/bin/env python3
"""The 1.0 certification round's inputs from the evidence archive: derive, check, pack,
verify and unpack them, on the GPU host and on the Mac alike.

WHY. The round's stages read measured records under ``parity/meep_gpu/results/`` that the
repository does not carry: the censuses and seam records the H->D lifts and the Triton
probes join their rows to, the Metal spine baseline, the Tier B record L3 judges against,
and the corpus legs the corpus manifest is derived from. A fresh checkout on another host
stages them from one private bundle (``unpack``); the host that holds the archive checks
them in place (``check``). Every one is named in this directory's manifest:

``INPUT_FILES.txt``
    one path per line, relative to ``results/``, sorted: exactly the files the bundle
    carries.
``INPUT_DIRS.txt``
    the top-level directories under ``results/`` those files sit in. Only the listed files
    ship from each, never the whole directory.
``INPUTS_SHA256SUMS``
    ``<sha256>  <path>`` per file, the format ``shasum -a 256 -c`` reads.
``INPUT_BUNDLE.sha256``
    ``<sha256>  round_inputs_1_0.tgz``: the bundle cut from the archive for this manifest.

The bundle is never committed or published: its records carry the machine stamps of the
hosts that cut them.

WHAT IS AN INPUT. A file a stage of the round opens and does not write, for the stages a
host other than the archive's runs: the gates (CERT-3, both machines), the bind's record
and rebind tools, the route and the licence (CERT-4), L3 (CERT-5), and the corpus manifest
the corpus (CERT-1) and L3 read. ``derive`` computes the list from three sources and
nothing typed by hand:

* :data:`READERS`: a module-level name in a gate or kit the round runs that binds a
  directory or a file under ``results/``, with the files its reader opens there; the name's
  value is read from the module's source (:func:`bound_path`), never imported. The ten
  Triton H->D gates are read through ``legs.triton_hd_inputs`` and
  ``legs.TRITON_HD_INPUT_FILES``, their one home.
* :data:`CORPUS_MODULE`'s ``LEGS``: the seven ``parity.jsonl`` files
  ``meep_identity_corpus.py manifest`` derives the corpus manifest from. The manifest is
  derived on the staging host from them, byte for byte, and is not shipped.
* The Tier B record at :data:`TIER_B_RECORD` (the home ``docs/development/testing.md``
  names) and, for each of the five L3 legs RUNBOOK CERT-5 runs, the row files
  ``l3_corpus.verify_row`` checks before a row runs (:data:`L3_ROLES`), of every set the
  leg judges, for every row ``l3_corpus.plan_rows`` plans. A set no leg judges ships
  nothing.

GRANULARITY. A reader that names a file ships that file. A census or seam directory ships
the files its reader opens there: a census directory also holds per-row records and logs
that no stage of the round opens.

NOT INPUTS (:data:`NOT_INPUTS`). Every other module-level name that binds a path under
``results/`` in a harness module the round's stages import is listed there with the
reason: an output root, a records-stage reader that runs only on the archive host (the
censuses, the H->D probe re-cut and the boards, RUNBOOK "CERT-4 records"), or a table of
names the round writes. ``test_round_inputs.py`` holds the two lists to the code both
ways, so a gate that starts reading another record fails the test before the round.

THE MEEP CORPUS is not an input: it is the ``python/`` tree of the MEEP release tarball,
which the reference environment keeps (``MEEP_GPU_CORPUS_ROOT``).

    python parity/meep_gpu/rounds/1.0/round_inputs.py derive [--results DIR] [--write] [--code-only]
    python parity/meep_gpu/rounds/1.0/round_inputs.py check  [--results DIR]
    python parity/meep_gpu/rounds/1.0/round_inputs.py pack   --results DIR --out FILE.tgz
    python parity/meep_gpu/rounds/1.0/round_inputs.py verify --bundle FILE.tgz
    python parity/meep_gpu/rounds/1.0/round_inputs.py unpack --bundle FILE.tgz [--results DIR]

``derive`` prints the derived list beside the committed one and the files each reader
contributes; ``--write`` replaces the three manifest files (the digests read from
``--results``); ``--code-only`` derives the code part alone, with no archive. ``check``
verifies the files present under ``results/`` against the manifest, with no bundle: the
staging step on the archive host, where ``results/`` entries may link into the archive.
``verify`` checks a bundle's digest against ``INPUT_BUNDLE.sha256``, its member set and
every member's digest, and writes nothing. ``pack`` writes a deterministic tarball
(members sorted, mtime 0, no owner names, gzip mtime 0) and its ``.sha256`` outside the
repository, then verifies what it wrote; ``--results`` may be a directory of links into
the archive. ``unpack`` checks the bundle's digest against the committed one, extracts
only files that are absent, never writes through a symbolic link, refuses a present file
whose digest differs, makes every refusal before its first write, and ends with
``check``.

Exit status: 0 verified; 1 a file or the bundle is missing, differs or cannot be placed
safely; 2 a usage or manifest error.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
REPO = HARNESS.parents[1]
RESULTS = HARNESS / "results"
for _path in (str(HERE), str(HARNESS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import legs as L  # noqa: E402

FILES = "INPUT_FILES.txt"
DIRS = "INPUT_DIRS.txt"
SUMS = "INPUTS_SHA256SUMS"
BUNDLE = "round_inputs_1_0.tgz"
BUNDLE_SUM = "INPUT_BUNDLE.sha256"
#: Members are copied in chunks of this size, so a large record is never held in memory
#: twice.
CHUNK = 1 << 20

#: The files a census directory's readers open: the legs' rows, and the parameterised leg's
#: matched rows that replace them (``match_param_rows.py``).
CENSUS_LEGS = ("examples.jsonl", "tests.jsonl", "tests_param_matched.jsonl")
#: The H->D seam record's rows, the only file of the seam directory a lift reads.
SEAM_ROWS = ("h_to_d_seam.jsonl",)

#: ``(reader module, module-level name, files opened under the directory it binds)``; an
#: empty tuple: the name binds a file, which is the input. Each row cites its reader.
READERS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    # CERT-3 NVIDIA, Triton fleet: each probe's corpus_admission leg reads the census rows
    # (examples, tests, the matched parameterised rows) and the board's cell beside them.
    ("probe_triton_beta_fused_electric_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_beta_fused_electric_pair", "MATRIX", ()),
    ("probe_triton_beta_fused_magnetic_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_beta_fused_magnetic_pair", "MATRIX", ()),
    ("probe_triton_bfast_fused_electric_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_bfast_fused_electric_pair", "MATRIX", ()),
    ("probe_triton_bfast_fused_magnetic_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_bfast_fused_magnetic_pair", "MATRIX", ()),
    ("probe_triton_conductive_fused_electric_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_conductive_fused_electric_pair", "MATRIX", ()),
    ("probe_triton_cylindrical_real_fused_electric_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_cylindrical_real_fused_electric_pair", "BOARD", ()),
    ("probe_triton_cylindrical_real_fused_magnetic_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_cylindrical_real_fused_magnetic_pair", "METAL_CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_beta_complex_fused_magnetic_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_beta_complex_fused_magnetic_pair", "BOARD", ()),
    ("probe_triton_folded_beta_complex_fused_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_beta_complex_fused_pair", "BOARD", ()),
    ("probe_triton_folded_beta_fused_electric_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_beta_fused_electric_pair", "MATRIX", ()),
    ("probe_triton_folded_beta_fused_magnetic_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_beta_fused_magnetic_pair", "MATRIX", ()),
    ("probe_triton_folded_complex_fused_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_complex_fused_pair", "BOARD", ()),
    ("probe_triton_folded_dispersive_fused_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_dispersive_fused_pair", "BOARD", ()),
    ("probe_triton_folded_fused_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_folded_fused_pair", "METAL_CENSUS", CENSUS_LEGS),
    ("probe_triton_nonlinear_fused_magnetic_pair", "CENSUS", CENSUS_LEGS),
    ("probe_triton_nonlinear_fused_magnetic_pair", "MATRIX", ()),
    # CERT-3 NVIDIA, the four CUDA H->D lift legs (lift_basis): the module a tests row
    # replays is joined from the census, the cell's rows from the seam record or the board.
    ("gate_cuda_fused_hd_pair", "CENSUS", ("tests.jsonl", "tests_param_matched.jsonl")),
    ("gate_cuda_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("cylindrical_hd_gate_common", "BOARD", ("fusion_matrix_cuda.json",)),
    ("cylindrical_hd_gate_common", "CENSUS", ("tests.jsonl", "tests_param_matched.jsonl")),
    ("gate_cuda_complex_fused_hd_pair", "BOARD", ("fusion_matrix_cuda.json",)),
    ("gate_cuda_complex_fused_hd_pair", "CENSUS",
     ("examples.jsonl", "tests.jsonl", "tests_param.jsonl", "tests_param_matched.jsonl")),
    # CERT-3 Metal, the H->D lift legs (lift_basis; the three tail gates through
    # hd_tail_gate_common.lift_basis): census rows joined to the seam record's arms.
    ("gate_metal_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_fused_hd_pair_nonlinear", "CENSUS", CENSUS_LEGS),
    ("gate_metal_fused_hd_pair_nonlinear", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_folded_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_folded_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_bfast_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_bfast_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_conductive_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_conductive_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_complex_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_complex_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_cylindrical_real_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_cylindrical_real_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_beta_real_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_beta_real_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_beta_complex_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_beta_complex_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    ("gate_metal_folded_complex_fused_hd_pair", "CENSUS", CENSUS_LEGS),
    ("gate_metal_folded_complex_fused_hd_pair", "SEAM_RECORD", SEAM_ROWS),
    # CERT-3 Metal: the corpus_shapes leg (without the census it records checked=0 and the
    # gate still releases), and the frozen spine baseline (exit 75 without it).
    ("gate_metal_cylindrical_complex_scan", "CENSUS_RECORD",
     ("examples.jsonl", "tests.jsonl", "tests_param.jsonl")),
    ("gate_metal_spine_byte_neutrality", "CERTIFIED", ()),
)

#: ``(module, name) -> why`` for every other module-level name binding a path under
#: ``results/`` in a harness module the round's stages import: none of them is opened by a
#: stage this manifest serves.
NOT_INPUTS: Dict[Tuple[str, str], str] = {
    ("artifact_drift", "NOT_RELEASE_PREFIXES"):
        "a path prefix the drift check excludes; names no record",
    ("h_to_d_seam", "PROBE_DIR"):
        "read by h_to_d_seam.price and probe_rows, which only the three boards call "
        "(records stage, archive host, with MEEP_GPU_H_TO_D_PROBE naming the freeze's "
        "re-cut); the gates use the module's constants and driver_seam_fact",
    ("lift_host", "LIFT_RECORD_NEEDS"):
        "lift-record labels (the census's per-row lift_record values) keyed to the packages "
        "a row needs; the lifting gates look a row's needs up by label and open nothing there",
    ("measure_predicate_coverage", "EXAMPLES_LIFTS"):
        "read by the census's own main (records stage, archive host); the H->D lift "
        "children run it in child mode, which returns before the lift records are read",
    ("measure_predicate_coverage", "TESTS_LIFT"):
        "read by the census's own main (records stage, archive host); child mode returns "
        "first",
    ("rebind_cuda_welds", "ARTIFACT_PREFIX"):
        "the prefix the rebind writes artifact paths under; names no record",
    ("record_cuda_regate", "ARTIFACT_PREFIX"):
        "the prefix the record writes artifact paths under; names no record",
    ("rounds/1.0/legs", "CUDA_CITED"):
        "certification block names: the subdirectories the round writes in its own "
        "campaign roots",
    ("rounds/1.0/legs", "CUDA_RESIDUE"):
        "certification block names: the subdirectories the round writes in its own "
        "campaign roots",
    ("rebind_cuda_welds", "RETIRED_PINS"):
        "weld keys (certification block names) and pinned file paths; names no record",
    ("gate_metal_offdiag_stencil_welds", "CORPUS_ROWS"):
        "a citation the artifact records (the census its cell counts came from); the gate "
        "opens nothing there",
    ("gate_cuda_complex_no_pml", "results/cuda_complex_no_pml_recon_2026-08-20"):
        "a citation the artifact records ('recon'); the gate opens nothing there",
    ("gate_metal_below_the_cut_fused_pairs", "results/below_the_cut_census_2026-08-20"):
        "a citation the artifact records ('census'); the gate opens nothing there",
    ("gate_metal_fused_magnetic_pair", "results/metal_coverage_recut_2026-08-16"):
        "a citation the artifact records ('census'); the gate opens nothing there",
    ("gate_metal_offdiag", "results/metal_whole_step_2026-08-16"):
        "a citation the artifact records ('evidence'); the gate opens nothing there",
    ("probe_triton_engine_route", "results/fused_pml_throughput_triton_2026-08-09"):
        "a scripts directory put on sys.path only where it exists; NEVER staged: its "
        "modules would shadow the harness's own (cases.py) in the probe's imports",
    ("probe_triton_engine_route", "results/cupy_throughput_2026-08-09"):
        "a scripts directory put on sys.path only where it exists; NEVER staged, for the "
        "same reason",
    ("sweep_corpus_lift_parity", "results/survey_recovery_stock"):
        "a default of the sweep's own main, which the round never runs (the corpus module "
        "and L3 import its capture_simulation and dispatch_facts)",
}

#: The scripts the round's stages run, beside the CERT-3 gate units ``chunks.units_for``
#: lists, on a host other than the archive's: the drivers, the route and the licence
#: (CERT-4), the bind, the corpus (CERT-1) and L3 (CERT-5). Their static run closures are
#: what :func:`audit` holds :data:`READERS` and :data:`NOT_INPUTS` to.
STAGE_SCRIPTS = (
    "drive_triton_weld_gates.py", "triton_device_identity.py", "track_triton_pml.py",
    "metal_environment.py", "rounds/1.0/legs.py", "rounds/1.0/round.py",
    "rounds/1.0/run_lane.py", "rounds/1.0/fleet_launch.py", "rounds/1.0/meep_host_import.py",
    "rounds/1.0/round_status.py", "rounds/1.0/chunks.py", "rounds/1.0/metal_round.py",
    "gate_dispatch_fused_route.py", "gate_dispatch_end_to_end.py",
    "gate_dispatch_metal_route.py",
    "record_cuda_regate.py", "rebind_cuda_welds.py", "record_cuda_ade_recut.py",
    "rebind_triton_welds.py", "recut_composition_records.py", "neutralize_ledger_labels.py",
    "rebind_metal_welds.py",
    "meep_identity_tier_c.py", "meep_identity_corpus.py", "rounds/1.0/l3_corpus.py",
    "rounds/1.0/l3_row.py",
)
#: The corpus module whose ``LEGS`` name the corpus manifest's inputs.
CORPUS_MODULE = "meep_identity_corpus"
CORPUS_FILE = "parity.jsonl"
#: The Tier B record's home under ``results/`` and its file there.
TIER_B = "meep_identity_tier_b"
TIER_B_RECORD = f"{TIER_B}/meep_identity_tier_b.json"
#: The row files ``l3_corpus.verify_row`` checks against the record before a row runs.
L3_ROLES = ("digests", "planes", "result")
#: The L3 legs RUNBOOK CERT-5 runs: NVIDIA, both table variants under both baselines, each
#: judging its own Linux set; Metal, flush, judging its three default sets.
L3_LEGS = (("triton", "keep"), ("triton", "flush"), ("cuda", "keep"), ("cuda", "flush"),
           ("metal", "flush"))
#: A module-level value naming a dated record directory (or a path under one).
DATED = re.compile(r"[A-Za-z0-9_.-]*20\d\d-\d\d-\d\d[A-Za-z0-9_.-]*(/[A-Za-z0-9_./-]*)?")


class ManifestError(Exception):
    """The round directory's manifest is malformed or inconsistent (exit 2)."""


class InputsRefused(Exception):
    """A file or bundle does not match the manifest, or cannot be placed safely (exit 1)."""


def say(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------- derive ------
def _module_path(module: str, harness: Path) -> Path:
    return harness / f"{module}.py"


def _tree(module: str, harness: Path) -> ast.Module:
    path = _module_path(module, harness)
    if not path.is_file():
        raise ManifestError(f"reader module {path} does not exist")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _bindings(tree: ast.Module) -> Dict[str, ast.AST]:
    """Module-level ``name = value`` and ``name: type = value``, the last one winning."""
    out: Dict[str, ast.AST] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            out[node.targets[0].id] = node.value
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
              and node.value is not None):
            out[node.target.id] = node.value
    return out


def _aliases(tree: ast.Module, harness: Path) -> Dict[str, str]:
    """``import x as y`` of a harness module beside the reader: ``y -> x``."""
    out: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if _module_path(item.name, harness).is_file():
                    out[item.asname or item.name] = item.name
    return out


def _parts(node: ast.AST, module: str, harness: Path, seen: Tuple[str, ...] = ()) -> List[str]:
    """The path components a module-level expression spells, roots as ``<root>``.

    Understands what the readers write: a string; ``os.path.join(...)``; ``a / b``;
    a name bound in the same module; ``alias.NAME`` of a harness module beside it. Anything
    else is refused, by name, rather than guessed at.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [part for part in node.value.split("/") if part]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return (_parts(node.left, module, harness, seen)
                + _parts(node.right, module, harness, seen))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join" and not node.keywords):
        out: List[str] = []
        for argument in node.args:
            out += _parts(argument, module, harness, seen)
        return out
    if isinstance(node, ast.Name):
        if node.id in ("HERE", "_HERE", "API_ROOT", "_API", "_API_ROOT", "REPO", "ROOT"):
            return ["<root>"]
        return _parts_of(module, node.id, harness, seen)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        aliases = _aliases(_tree(module, harness), harness)
        if node.value.id in aliases:
            return _parts_of(aliases[node.value.id], node.attr, harness, seen)
    raise ManifestError(f"{module}.py: cannot read a path from "
                        f"{ast.dump(node)[:120]}")


def _parts_of(module: str, name: str, harness: Path, seen: Tuple[str, ...] = ()) -> List[str]:
    key = f"{module}.{name}"
    if key in seen or len(seen) > 6:
        raise ManifestError(f"{key}: the binding refers back to itself ({seen})")
    bindings = _bindings(_tree(module, harness))
    if name not in bindings:
        raise ManifestError(f"{module}.py binds no module-level {name}")
    return _parts(bindings[name], module, harness, seen + (key,))


def bound_path(module: str, name: str, harness: Path = HARNESS) -> str:
    """The path under ``results/`` a reader's module-level ``name`` binds."""
    parts = _parts_of(module, name, harness)
    if "results" in parts:
        parts = parts[len(parts) - parts[::-1].index("results"):]
    elif parts and parts[0] == "<root>":
        raise ManifestError(f"{module}.{name} names a path outside results/: {parts}")
    if not parts or any(part in ("<root>", ".", "..") for part in parts):
        raise ManifestError(f"{module}.{name} does not name a record under results/: "
                            f"{parts}")
    return "/".join(parts)


def _defining_module(module: str, name: str, harness: Path,
                     seen: Tuple[str, ...] = ()) -> Optional[str]:
    """The module that binds ``name``: ``module`` itself, else the first harness module it
    imports that does, the order ``legs._harness_constant`` reads them in."""
    if module in seen or len(seen) > 4 or not _module_path(module, harness).is_file():
        return None
    tree = _tree(module, harness)
    if name in _bindings(tree):
        return module
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if _module_path(item.name, harness).is_file():
                    found = _defining_module(item.name, name, harness, seen + (module,))
                    if found is not None:
                        return found
    return None


def triton_hd_readers(harness: Path = HARNESS) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """The ten Triton H->D gates' inputs as ``legs.py check`` reads them, keyed by the module
    that binds each name (a gate, or the kit it takes its names from)."""
    out = set()
    for gate, names in sorted(L.triton_hd_inputs(harness, L.triton_hd_steps()).items()):
        for kind, value in sorted(names.items()):
            module = _defining_module(f"gate_triton_{gate}", kind, harness)
            if value is None or module is None:
                raise ManifestError(f"gate_triton_{gate}.py: {kind} cannot be read")
            out.add((module, kind, tuple(L.TRITON_HD_INPUT_FILES[kind])))
    return sorted(out)


def corpus_legs(harness: Path = HARNESS) -> List[str]:
    """The corpus legs' ``parity.jsonl`` files, from ``meep_identity_corpus.LEGS``."""
    value = _bindings(_tree(CORPUS_MODULE, harness)).get("LEGS")
    if not isinstance(value, ast.Dict) or not all(
            isinstance(v, ast.Constant) and isinstance(v.value, str) for v in value.values):
        raise ManifestError(f"{CORPUS_MODULE}.py: LEGS is not a dict of directory names")
    return sorted(f"{v.value}/{CORPUS_FILE}" for v in value.values)


def code_inputs(harness: Path = HARNESS) -> Dict[str, List[str]]:
    """``path under results/ -> [the readers that open it]``, from the code alone."""
    out: Dict[str, List[str]] = {}
    for module, name, files in tuple(READERS) + tuple(triton_hd_readers(harness)):
        base = bound_path(module, name, harness)
        for path in ([f"{base}/{f}" for f in files] if files else [base]):
            out.setdefault(_safe_relative(path), []).append(f"{module}.{name}")
    for path in corpus_legs(harness):
        out.setdefault(path, []).append(f"{CORPUS_MODULE}.LEGS (the corpus manifest)")
    return out


def _l3():
    import l3_corpus  # noqa: PLC0415 - beside this file; stdlib only at import

    return l3_corpus


def tier_b_inputs(results: Path) -> Dict[str, List[str]]:
    """The Tier B record and the row files the five L3 legs verify, under ``results``."""
    l3 = _l3()
    record_path = results / TIER_B_RECORD
    if not record_path.is_file():
        raise InputsRefused(f"{record_path} is absent: the Tier B record is staged at "
                            f"{TIER_B_RECORD} under the results tree")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {TIER_B_RECORD: ["l3_corpus --record"]}
    for table, baseline in L3_LEGS:
        args = l3.resolve(l3.parser().parse_args(
            ["--table", table, "--baseline", baseline, "--record", str(record_path),
             "--out", str(results / TIER_B / ".derive")]))
        plan = l3.plan_rows(args, record)
        for label in plan["labels"]:
            for name in args.set_list:
                files = (record["sets"][name]["rows"][label] or {}).get("files") or {}
                for role in L3_ROLES:
                    entry = files.get(role) or {}
                    if not entry.get("file"):
                        raise InputsRefused(f"the record names no {role} file for {label} "
                                            f"in {name}")
                    path = _safe_relative(f"{TIER_B}/{entry['file']}")
                    out.setdefault(path, []).append(f"l3_corpus {table}/{baseline}")
    return out


def derive(results: Optional[Path], harness: Path = HARNESS,
           code_only: bool = False) -> Dict[str, List[str]]:
    """``path -> readers`` for the whole manifest (or its code part alone)."""
    out = code_inputs(harness)
    if not code_only:
        for path, readers in tier_b_inputs(results if results else RESULTS).items():
            out.setdefault(path, []).extend(readers)
    return dict(sorted(out.items()))


def results_bindings(module: str, harness: Path = HARNESS) -> List[str]:
    """The module-level names of a harness module that bind a path under ``results/``:
    a value spelling the component ``results``, a ``parity/meep_gpu/results/`` path, or a
    dated record name. A binding of the ``results/`` root itself names no record."""
    tree = _tree(module, harness)
    names = []
    for name, value in _bindings(tree).items():
        strings = [n.value for n in ast.walk(value)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if not strings:
            continue
        rooted = "results" in strings or any("parity/meep_gpu/results" in s for s in strings)
        dated = any(DATED.fullmatch(s) for s in strings)
        if not (rooted or dated):
            continue
        if strings == ["results"]:
            continue
        names.append(name)
    return sorted(names)


def _docstrings(tree: ast.Module) -> Set[int]:
    out: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            out.add(id(node.value))
    return out


def _operands(node: ast.AST) -> List[Optional[str]]:
    """A path expression's operands in order: a string as itself, anything else ``None``;
    ``a / b`` chains and ``os.path.join(...)`` calls are flattened."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _operands(node.left) + _operands(node.right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"):
        return [part for argument in node.args for part in _operands(argument)]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    return [None]


def _output_defaults(tree: ast.Module) -> Set[int]:
    """The nodes of every ``add_argument("--out...", default=...)`` default: where a tool
    writes, which is never an input."""
    out: Set[int] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("--out")):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    out |= {id(inner) for inner in ast.walk(keyword.value)}
    return out


def _head(text: str) -> str:
    """A path's first component, up to the first separator or blank."""
    return re.split(r"[/\s]", text.strip(), maxsplit=1)[0]


def results_expressions(module: str, harness: Path = HARNESS) -> List[str]:
    """The records a harness module spells a path to, anywhere in its code: the first
    component under ``results/`` of every ``a / "results" / "<name>"`` or
    ``os.path.join(..., "results", "<name>", ...)`` expression, and of every string (not a
    docstring) that starts ``parity/meep_gpu/results/``; a ``--out`` argument's default is
    an output and is not read. A path whose record name is not a string here
    (``results / CENSUS``) is read through :func:`results_bindings`."""
    tree = _tree(module, harness)
    skip = _docstrings(tree) | _output_defaults(tree)
    heads: Set[str] = set()
    prefix = "parity/meep_gpu/results/"
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(prefix):
                head = _head(node.value[len(prefix):])
                if head:
                    heads.add(head)
            continue
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)) and not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"):
            continue
        operands = _operands(node)
        for index, operand in enumerate(operands[:-1]):
            following = operands[index + 1]
            if operand is not None and operand.rstrip("/").endswith("results") and following:
                head = _head(following)
                if head:
                    heads.add(head)
    return sorted(heads)


def stage_programs(repo: Path = REPO) -> Dict[str, List[str]]:
    """``repo-relative script -> why the round runs it``: every CERT-3 gate unit's scripts
    (``chunks.units_for``, both machines) and :data:`STAGE_SCRIPTS`."""
    import chunks as C  # noqa: PLC0415 - beside this file

    out: Dict[str, List[str]] = {}
    for machine in C.MACHINES:
        for unit in C.units_for(repo, machine):
            for script in list(unit["scripts"]) + [s for s in unit.get("extra", ())
                                                    if s.endswith(".py")]:
                out.setdefault(script, []).append(f"CERT-3 {unit['id']}")
    for script in STAGE_SCRIPTS:
        out.setdefault(f"parity/meep_gpu/{script}", []).append("stage script")
    return out


def _spelled_at(module: str, name: str, harness: Path,
                seen: Tuple[str, ...] = ()) -> Tuple[str, str]:
    """The binding that spells ``module.name``'s path: followed through ``NAME = OTHER``
    and ``NAME = alias.OTHER`` (the H->D kits' ``CENSUS = base.CENSUS``)."""
    value = _bindings(_tree(module, harness)).get(name)
    key = f"{module}.{name}"
    if key in seen or len(seen) > 6:
        return module, name
    if isinstance(value, ast.Name):
        if value.id in _bindings(_tree(module, harness)):
            return _spelled_at(module, value.id, harness, seen + (key,))
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        aliases = _aliases(_tree(module, harness), harness)
        if value.value.id in aliases:
            return _spelled_at(aliases[value.value.id], value.attr, harness, seen + (key,))
    return module, name


def audit(repo: Path = REPO) -> Tuple[List[str], Dict[str, Any]]:
    """``(problems, facts)``: :data:`READERS` and :data:`NOT_INPUTS` against the code.

    Over the static run closure (``import_closure.py``) of every script
    :func:`stage_programs` names, in every harness module of the closure: each module-level
    binding of a path under ``results/`` (:func:`results_bindings`) must spell a reader's
    path or be a named non-input ``(module, NAME)``; each record a path expression spells
    (:func:`results_expressions`) must be a directory the readers ship from or a named
    non-input ``(module, "results/<record>")``. Every reader must sit in the closure, and
    every named non-input must still be found. The package's own modules are not read: the
    package never opens ``results/``, which a user's install does not have.
    """
    import chunks as C  # noqa: PLC0415

    harness = repo / "parity" / "meep_gpu"
    resolver = C.load_import_closure(repo).Resolver(repo)
    programs = stage_programs(repo)
    problems: List[str] = []
    closure: Set[str] = set()
    for script in programs:
        if not resolver.exists(script):
            problems.append(f"{script}: a script the round runs, absent from the checkout")
            continue
        closure |= set(resolver.run_closure(script)) | {script}
    prefix = "parity/meep_gpu/"
    modules = sorted(name[len(prefix):-len(".py")] for name in closure
                     if name.startswith(prefix) and name.endswith(".py")
                     and not name.startswith(prefix + "results/"))
    readers = [(m, n) for m, n, _f in READERS] + [(m, n) for m, n, _f in
                                                  triton_hd_readers(harness)]
    readers.append((CORPUS_MODULE, "LEGS"))
    for module, name in readers:
        if module not in modules:
            problems.append(f"{module}.{name}: a reader in no module the round runs")
    spelled = {_spelled_at(m, n, harness) for m, n in readers}
    shipped_from = set(top_dirs(code_inputs(harness)))
    found: Set[Tuple[str, str]] = set()
    for module in modules:
        for name in results_bindings(module, harness):
            found.add((module, name))
        for head in results_expressions(module, harness):
            if head not in shipped_from:
                found.add((module, f"results/{head}"))
    for key in sorted(found - spelled - set(NOT_INPUTS)):
        what = ("names the record" if key[1].startswith("results/")
                else "binds a path under results/ and")
        problems.append(f"{key[0]} {key[1]}: {what} is neither a reader (READERS) nor a "
                        f"named non-input (NOT_INPUTS)")
    for key in sorted(spelled - found):
        if key[0] in modules:
            problems.append(f"{key[0]}.{key[1]}: a reader whose binding names no path under "
                            f"results/ any more")
    for key in sorted(set(NOT_INPUTS) - found):
        problems.append(f"{key[0]} {key[1]}: a named non-input found in no module the round "
                        f"runs")
    return problems, {"programs": len(programs), "closure_files": len(closure),
                      "harness_modules": len(modules), "bindings": len(found)}


# ----------------------------------------------------------------------- manifest ------
def _lines(path: Path) -> List[str]:
    if not path.is_file():
        raise ManifestError(f"{path} is missing")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _safe_relative(text: str) -> str:
    """A manifest path: relative, POSIX, no ``..``, no empty or ``.`` component."""
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts:
        raise ManifestError(f"{text!r} is not a relative path")
    if any(part in ("", ".", "..") for part in path.parts) or "\\" in text:
        raise ManifestError(f"{text!r} has a component that leaves results/")
    if str(path) != text:
        raise ManifestError(f"{text!r} is not in normal form ({path})")
    return str(path)


def top_dirs(files: Iterable[str]) -> List[str]:
    return sorted({PurePosixPath(name).parts[0] for name in files
                   if len(PurePosixPath(name).parts) > 1})


def read_manifest(round_dir: Path = HERE) -> Tuple[List[str], List[str], Dict[str, str]]:
    """``(files, dirs, sums)``, after checking the three files agree with one another."""
    files = [_safe_relative(line) for line in _lines(round_dir / FILES)]
    dirs = [_safe_relative(line) for line in _lines(round_dir / DIRS)]
    if len(set(files)) != len(files):
        raise ManifestError(f"{FILES} lists a file twice")
    if files != sorted(files):
        raise ManifestError(f"{FILES} is not in sorted order")
    if dirs != top_dirs(files):
        raise ManifestError(f"{DIRS} is not the sorted set of top-level directories of "
                            f"{FILES}: {dirs} against {top_dirs(files)}")
    sums: Dict[str, str] = {}
    for line in _lines(round_dir / SUMS):
        digest, _, name = line.partition("  ")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ManifestError(f"{SUMS}: {line!r} is not '<sha256>  <path>'")
        name = _safe_relative(name)
        if name in sums:
            raise ManifestError(f"{SUMS} lists {name} twice")
        sums[name] = digest
    if set(sums) != set(files):
        raise ManifestError(
            f"{SUMS} and {FILES} name different files: "
            f"only in {SUMS} {sorted(set(sums) - set(files))[:5]}, "
            f"only in {FILES} {sorted(set(files) - set(sums))[:5]}")
    return files, dirs, sums


def sums_text(files: Sequence[str], sums: Mapping[str, str]) -> str:
    return "".join(f"{sums[name]}  {name}\n" for name in files)


def committed_bundle_digest(round_dir: Path = HERE) -> str:
    lines = _lines(round_dir / BUNDLE_SUM)
    digest, _, name = lines[0].partition("  ")
    if (len(lines) != 1 or name != BUNDLE or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ManifestError(f"{BUNDLE_SUM} must be one line '<sha256>  {BUNDLE}'")
    return digest


def write_manifest(round_dir: Path, results: Path, files: Sequence[str]) -> Dict[str, str]:
    """Write the three manifest files for ``files``, digests read from ``results``."""
    files = sorted(files)
    missing = [name for name in files if not (results / name).is_file()]
    if missing:
        raise InputsRefused(f"{len(missing)} derived input(s) absent from {results}: "
                            f"{missing[:5]}")
    sums = {}
    for index, name in enumerate(files, 1):
        sums[name] = sha256_file(results / name)
        if index % 200 == 0 or index == len(files):
            say(f"derive: hashed {index} of {len(files)}")
    (round_dir / FILES).write_text("".join(f"{name}\n" for name in files), encoding="utf-8")
    (round_dir / DIRS).write_text("".join(f"{d}\n" for d in top_dirs(files)),
                                  encoding="utf-8")
    (round_dir / SUMS).write_text(sums_text(files, sums), encoding="utf-8")
    return sums


# ------------------------------------------------------------------ check / bundle -----
def check_present(round_dir: Path, results: Path, *, progress: bool = False) -> List[str]:
    """Every manifest file under ``results`` with its digest; return the failures."""
    files, _dirs, sums = read_manifest(round_dir)
    failures = []
    for index, name in enumerate(files, 1):
        path = results / name
        if not path.is_file():
            failures.append(f"missing: {name}")
        else:
            found = sha256_file(path)
            if found != sums[name]:
                failures.append(f"differs: {name} (sha256 {found}, manifest {sums[name]})")
        if progress and (index % 200 == 0 or index == len(files)):
            say(f"check: {index} of {len(files)} read, {len(failures)} failing")
    return failures


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def pack(round_dir: Path, results: Path, out: Path) -> str:
    """Write the bundle; return its sha256. Refuses a source file that disagrees."""
    out = out.resolve()
    if str(out).startswith(str(REPO.resolve()) + os.sep):
        raise InputsRefused(
            f"{out} is inside the repository; the bundle carries host stamps and must "
            f"not be written where it could be committed")
    failures = check_present(round_dir, results, progress=True)
    if failures:
        raise InputsRefused(f"the archive does not match the manifest ({len(failures)} "
                            f"file(s)): " + "; ".join(failures[:10]))
    files, _dirs, sums = read_manifest(round_dir)
    manifest = sums_text(files, sums).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    with open(temporary, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w",
                              format=tarfile.PAX_FORMAT) as archive:
                archive.addfile(_tarinfo(SUMS, len(manifest)), io.BytesIO(manifest))
                for index, name in enumerate(files, 1):
                    source = results / name
                    with open(source, "rb") as handle:
                        archive.addfile(_tarinfo(name, os.fstat(handle.fileno()).st_size),
                                        handle)
                    if index % 200 == 0 or index == len(files):
                        say(f"pack: {index} of {len(files)} members written")
    os.replace(temporary, out)
    digest = sha256_file(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {BUNDLE}\n", encoding="utf-8")
    problems = verify_bundle(round_dir, out)
    if problems:
        raise InputsRefused("the bundle just written does not verify: "
                            + "; ".join(problems[:10]))
    return digest


def _members(archive: tarfile.TarFile) -> Dict[str, tarfile.TarInfo]:
    members: Dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        try:
            name = _safe_relative(member.name)
        except ManifestError as error:
            raise InputsRefused(f"bundle member: {error}") from None
        if not member.isreg():
            raise InputsRefused(f"bundle member {member.name!r} is not a regular file")
        if name in members:
            raise InputsRefused(f"bundle member {name!r} appears twice")
        members[name] = member
    return members


def verify_bundle(round_dir: Path, bundle: Path) -> List[str]:
    """Member set == manifest + ``INPUTS_SHA256SUMS``, and every digest matches."""
    files, _dirs, sums = read_manifest(round_dir)
    problems: List[str] = []
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = _members(archive)
        expected = set(files) | {SUMS}
        if set(members) != expected:
            problems.append(f"members differ from the manifest: extra "
                            f"{sorted(set(members) - expected)[:5]}, missing "
                            f"{sorted(expected - set(members))[:5]}")
        if SUMS in members:
            carried = archive.extractfile(members[SUMS]).read().decode("utf-8")
            if carried != sums_text(files, sums):
                problems.append(f"the bundle's {SUMS} is not the committed manifest")
        for name in files:
            if name not in members:
                continue
            digest = hashlib.sha256()
            handle = archive.extractfile(members[name])
            for block in iter(lambda: handle.read(CHUNK), b""):
                digest.update(block)
            if digest.hexdigest() != sums[name]:
                problems.append(f"differs: {name}")
    return problems


def _symlinked_component(results: Path, name: str) -> str:
    """The first existing component of ``results/name``'s parents that is a link."""
    current = results
    for part in PurePosixPath(name).parts[:-1]:
        current = current / part
        if current.is_symlink():
            return str(current)
        if not current.exists():
            break
    return ""


def unpack(round_dir: Path, bundle: Path, results: Path) -> Tuple[int, int]:
    """Extract absent files; return ``(extracted, already_present)``."""
    if results.is_symlink() or not results.is_dir():
        raise InputsRefused(f"{results} must be a real directory")
    expected = committed_bundle_digest(round_dir)
    found = sha256_file(bundle)
    if found != expected:
        raise InputsRefused(
            f"{bundle} has sha256 {found}; {BUNDLE_SUM} names {expected}. This is "
            f"not the bundle cut for this manifest")
    problems = verify_bundle(round_dir, bundle)
    if problems:
        raise InputsRefused("the bundle does not verify: " + "; ".join(problems[:10]))
    files, _dirs, sums = read_manifest(round_dir)
    # EVERY REFUSAL BEFORE THE FIRST WRITE, so a refused unpack leaves results/ as it
    # found it rather than half-filled.
    absent: List[str] = []
    for name in files:
        target = results / name
        link = _symlinked_component(results, name)
        if link or target.is_symlink():
            raise InputsRefused(
                f"{link or target} is a symbolic link; unpacking would write "
                f"through it, into whatever it points at. Run 'check' instead")
        if target.exists():
            if not target.is_file() or sha256_file(target) != sums[name]:
                raise InputsRefused(
                    f"{target} is present and is not the manifest's file; move "
                    f"it out of results/ rather than overwrite it")
            continue
        absent.append(name)
    extracted, present = 0, len(files) - len(absent)
    with tarfile.open(bundle, mode="r:gz") as archive:
        members = _members(archive)
        for name in absent:
            target = results / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".unpack.tmp")
            source = archive.extractfile(members[name])
            with open(temporary, "wb") as handle:
                for block in iter(lambda: source.read(CHUNK), b""):
                    handle.write(block)
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
            extracted += 1
            if extracted % 200 == 0 or extracted == len(absent):
                say(f"unpack: {extracted} of {len(absent)} absent files extracted")
    failures = check_present(round_dir, results)
    if failures:
        raise InputsRefused("after unpacking: " + "; ".join(failures[:10]))
    return extracted, present


# --------------------------------------------------------------------------- main ------
def _report_derived(derived: Mapping[str, List[str]], committed: Sequence[str],
                    code_only: bool) -> int:
    paths = set(derived)
    if code_only:
        committed = [name for name in committed if not name.startswith(TIER_B + "/")]
    extra = sorted(paths - set(committed))
    gone = sorted(set(committed) - paths)
    by_dir: Dict[str, int] = {}
    for path in derived:
        top = PurePosixPath(path).parts[0]
        by_dir[top] = by_dir.get(top, 0) + 1
    for top, count in sorted(by_dir.items()):
        say(f"{count:5d}  {top}/")
    for path in extra:
        say(f"not in {FILES}: {path} (read by {', '.join(derived[path])})")
    for path in gone:
        say(f"in {FILES}, derived from no reader: {path}")
    say(f"derived {len(paths)} files in {len(by_dir)} directories; committed "
        f"{len(committed)}; {len(extra)} to add, {len(gone)} to drop")
    return 1 if (extra or gone) else 0


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("derive", "check", "pack", "verify", "unpack"))
    parser.add_argument("--round", type=Path, default=HERE,
                        help="the round directory holding the manifest (default: this one)")
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="the results tree (default: the checkout's results/)")
    parser.add_argument("--out", type=Path, help="pack: the bundle to write")
    parser.add_argument("--bundle", type=Path, help="verify, unpack: the bundle")
    parser.add_argument("--write", action="store_true",
                        help="derive: replace the three manifest files")
    parser.add_argument("--code-only", action="store_true",
                        help="derive: the code part alone, with no archive")
    arguments = parser.parse_args(list(argv) or None)
    round_dir = arguments.round.resolve()
    try:
        if arguments.action == "derive":
            problems, facts = audit()
            say(f"audit: {facts['programs']} scripts, {facts['closure_files']} files in their "
                f"run closures, {facts['harness_modules']} harness modules, "
                f"{facts['bindings']} bindings of a path under results/")
            if problems:
                for problem in problems:
                    say(problem)
                raise ManifestError(f"{len(problems)} binding(s) the registry does not "
                                    f"account for; derive stops here")
            derived = derive(arguments.results, code_only=arguments.code_only)
            if arguments.write:
                if arguments.code_only:
                    parser.error("--write needs the whole manifest, not --code-only")
                write_manifest(round_dir, arguments.results, list(derived))
                say(f"wrote {FILES}, {DIRS}, {SUMS}: {len(derived)} files")
                return 0
            committed = read_manifest(round_dir)[0] if (round_dir / FILES).is_file() else []
            return _report_derived(derived, committed, arguments.code_only)
        files, dirs, _sums = read_manifest(round_dir)
        if arguments.action == "check":
            failures = check_present(round_dir, arguments.results, progress=True)
            for failure in failures:
                say(failure)
            say(f"{len(files) - len(failures)} of {len(files)} input files verified "
                f"under {arguments.results}")
            return 1 if failures else 0
        if arguments.action == "pack":
            if arguments.out is None:
                parser.error("pack needs --out")
            digest = pack(round_dir, arguments.results, arguments.out)
            say(f"packed {len(files)} files from {len(dirs)} directories; sha256 {digest}; "
                f"{arguments.out.stat().st_size} bytes")
            if (round_dir / BUNDLE_SUM).is_file() and committed_bundle_digest(round_dir) != digest:
                say(f"NOTE: {BUNDLE_SUM} names another digest; this bundle is not the one "
                    f"the manifest was cut with")
            return 0
        if arguments.bundle is None:
            parser.error(f"{arguments.action} needs --bundle")
        if arguments.action == "verify":
            problems = verify_bundle(round_dir, arguments.bundle)
            # THE SAME DIGEST CHECK ``unpack`` MAKES: a bundle whose members verify but
            # whose digest is not the committed one was not cut for this manifest.
            found = sha256_file(arguments.bundle)
            expected = committed_bundle_digest(round_dir)
            if found != expected:
                problems.append(f"{arguments.bundle} has sha256 {found}; {BUNDLE_SUM} "
                                f"names {expected}")
            for problem in problems:
                say(problem)
            say(f"bundle {'does not verify' if problems else 'verified'}: {len(files)} files")
            return 1 if problems else 0
        extracted, present = unpack(round_dir, arguments.bundle, arguments.results)
        say(f"unpacked {extracted} files, {present} already present and identical; "
            f"{len(files)} of {len(files)} verified")
        return 0
    except ManifestError as error:
        say(f"REFUSED (manifest): {error}")
        return 2
    except InputsRefused as error:
        say(f"REFUSED: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
