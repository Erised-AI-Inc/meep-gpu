"""Device gate for the FOLDED real-beta fused MAGNETIC B/H pair on Triton.

DEVICE STATUS: **UNRUN at authoring time.** This module is written to be run on a
    CUDA host with an idle device; nothing in this tree may cite it as a release
    until an artifact exists with ``device_status: RUN`` and
    ``release.released: true``. The product module's ``DEVICE_STATUS`` is what
    names the released artifact once one exists.

Folded beta ``step_B`` welded into ``update_H`` with the mirror fills carried
inline — the product of
:mod:`meep_gpu.triton_kernels.folded_beta_fused_magnetic_pair`, serving the
``folded real beta PML`` -> ``folded beta run`` B->H cell, and the MAGNETIC TWIN
of ``probe_triton_folded_beta_fused_electric_pair.py``. The leg families, the
oracles, the carry/null-control structure and the release condition are the
electric twin's; what differs is everything the B seam differs by:

* the claim: this kernel is
  ``folded_fused_magnetic_pair.folded_fused_curl_constitutive_B``, character for
  character, PLUS the same three-line insert
  ``folded_complex.folded_beta_pml_curl_step`` adds to
  ``symmetry.pml_curl_step_folded``, taken through ``BACKWARD == 0``;
* the seam: ``step_B -> MAGNETIC SOURCES -> fill_symmetry_bc_B -> zero_metal_B
  -> fill_folded_far_ghosts_B -> update_H`` (driver.py:3281-3289); the corpus
  row's TWO MAGNETIC sources land HERE, so the carry family injects
  magnetically and the quiet family's live source is ELECTRIC;
* the fill geometry: a B component images NEAR on its OWN axis (with the moved
  destination coefficient index) and FAR on the two others — the exact
  complement of the D side — so the parity/carry mutations rewrite different
  text and the composition arithmetic is the B-side one;
* the constitutive half reads no inverse epsilon (``update_H``'s source is B
  itself), so there is no material-component mutation here and the harness
  still installs a component-varying epsilon only for the E half the DRIVER
  runs outside this seam.

BETA IS 2-D ONLY, so every case is 2-D and no case can fold z, exactly as on
the electric side. The corpus context (the census row, the board cell, the
one-row funnel) is re-derived by the ``corpus_admission`` leg at the B->H cell.

Usage::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_beta_fused_magnetic_pair.py \\
        --no-device --out parity/meep_gpu/results/<fresh-dir>

    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_beta_fused_magnetic_pair.py \\
        --out parity/meep_gpu/results/<fresh-dir>
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PACKAGE_DIR = os.path.join(API_ROOT, "meep_gpu", "triton_kernels")
PRODUCT_MODULE = "meep_gpu.triton_kernels.folded_beta_fused_magnetic_pair"
KERNEL_FILE = os.path.join(PACKAGE_DIR, "folded_beta_fused_magnetic_pair.py")
KERNEL_NAME = "folded_beta_fused_curl_constitutive_B"

#: TestSpecialKz.test_eigsrc_kz_1_real_imag's own beta — the single row this
#: cell has.
BETA_CORPUS = 0.2

BETA_INSERT = (
    "if HAS_BETA:",
    "curl0 = curl0 - (beta_plus * b)",
    "curl1 = curl1 - (beta_minus * a)",
)

#: The device cases — the electric twin's table, byte for byte: the same grids
#: compile BOTH kernels' constexpr sets, and a difference between the two
#: gates' case tables would be a difference between the two products' evidence.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any,
                   Tuple[Tuple[str, int], ...], int, Dict[str, Any]], ...] = (
    ("corpus_y_fold_periodic", (1.6, 1.4, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"}, (("Y", 1),), 60,
     {"expect_ghost_destinations": 1}),
    ("odd_y_fold_periodic", (1.6, 1.75, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"}, (("Y", -1),), 60,
     {"expect_ghost_destinations": 1}),
    ("y_fold_metallic_wall_x", (1.6, 1.4, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", 1),), 60, {}),
    # On the B side two folded PERIODIC axes give every component THREE ghost
    # cells too — one near plane on its own axis, one far plane on the other,
    # and their composite — through the complementary fill geometry.
    ("two_folds_periodic_odd", (1.75, 1.75, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", -1)), 60, {"expect_ghost_destinations": 3}),
    # On the B side By composes its near-y image with x's far plane here:
    # (1 + near_1) * 2^(far_x) - 1 = 3.
    ("mixed_terminations", (1.6, 1.6, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"},
     (("X", -1), ("Y", 1)), 60, {"expect_ghost_destinations": 3}),
    ("corpus_single_launch", (1.6, 1.4, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"}, (("Y", 1),), 1,
     {"expect_ghost_destinations": 1}),
)

CASES_BY_NAME: Dict[str, Tuple[Any, ...]] = {case[0]: case for case in CASES}

DEFAULT_MUTATION_CASE = "two_folds_periodic_odd"

MUTATION_CASE: Dict[str, str] = {
    "m_wall_clear_dropped": "y_fold_metallic_wall_x",
    "m_beta_below_the_masks": "y_fold_metallic_wall_x",
    # Bx's near image is on the X axis, so the parity that must be ODD is the
    # X one — mixed_terminations folds x at -1.
    "m_near_parity_dropped": "mixed_terminations",
    "m_far_row_fixed_n_minus_2": "odd_y_fold_periodic",
    "m_far_parity_not_negated": "corpus_y_fold_periodic",
    "m_far_carry_dropped": "corpus_y_fold_periodic",
}

#: The CARRY family: a real MAGNETIC deposit IN THIS SEAM.
CARRY_CASES: Tuple[str, ...] = ("corpus_y_fold_periodic", "y_fold_metallic_wall_x")

CARRY_STEPS = 8
MUTATION_STEPS = 4

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "signed_zero_lattice", "subnormal_band")

SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

MATERIAL = ("eps", "inv_eps")

SEAM_OUTPUTS: Tuple[str, ...] = (
    "Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
    "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)

DEFAULT_OPTIONS: Dict[str, Any] = {"resolution": 12.0, "courant": 0.35, "pml": 4}

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(*parts: str) -> int:
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % (2 ** 31 - 1)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}.")


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    if "subnormal_policy" in payload:
        try:
            from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

            payload["subnormal_policy"] = _policy.policy_stamp()
        except Exception:  # noqa: BLE001
            pass
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        _stamp_provenance(payload)
    except Exception as exc:  # noqa: BLE001
        payload["provenance_stamp_error"] = repr(exc)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    names = (
        "meep_gpu/triton_kernels/folded_beta_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/folded_beta_fused_electric_pair.py",
        "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/folded_complex.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_folded_beta_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# The shipped sources
# ---------------------------------------------------------------------------

def _statements(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def body_statements(path: str, name: str) -> List[str]:
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    if node is None:
        raise AssertionError(f"{name} is not defined in {path}")
    body = node.body
    if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    segments = [ast.get_source_segment(text, statement) for statement in body]
    return _statements("\n".join(segment for segment in segments if segment))


def shipped_source() -> str:
    text = open(KERNEL_FILE, encoding="utf-8").read()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == KERNEL_NAME:
            import textwrap  # noqa: PLC0415

            decorators = [ast.get_source_segment(text, decorator)
                          for decorator in node.decorator_list]
            if any(segment is None for segment in decorators):
                raise AssertionError("the decorator source is unavailable")
            return "\n".join(
                [f"@{segment}" for segment in decorators]
                + [textwrap.dedent(ast.get_source_segment(text, node))])
    raise AssertionError(f"{KERNEL_NAME} is not defined in {KERNEL_FILE}")


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription
# ---------------------------------------------------------------------------

def transcription_leg() -> Dict[str, Any]:
    """kernel - insert == folded_fused_curl_constitutive_B; insert == K3a's
    delta; the insert sits in the array path's position; nothing invented; and
    this is the MAGNETIC weld, not a copy of the electric twin."""
    folded_magnetic_py = os.path.join(PACKAGE_DIR, "folded_fused_magnetic_pair.py")
    folded_complex_py = os.path.join(PACKAGE_DIR, "folded_complex.py")
    symmetry_py = os.path.join(PACKAGE_DIR, "symmetry.py")
    twin_py = os.path.join(PACKAGE_DIR, "folded_beta_fused_electric_pair.py")

    fused = body_statements(KERNEL_FILE, KERNEL_NAME)
    base = body_statements(folded_magnetic_py, "folded_fused_curl_constitutive_B")
    beta_curl = body_statements(folded_complex_py, "folded_beta_pml_curl_step")
    plain_curl = body_statements(symmetry_py, "pml_curl_step_folded")
    twin = body_statements(twin_py, "folded_beta_fused_curl_constitutive_D")

    findings: List[str] = []
    without_beta = [line for line in fused if line not in BETA_INSERT]
    if without_beta != base:
        index = next((i for i, (a, b) in enumerate(zip(without_beta, base))
                      if a != b), min(len(without_beta), len(base)))
        findings.append(
            f"minus the beta insert this is NOT folded_fused_curl_constitutive_B; "
            f"first difference at statement {index}: "
            f"{without_beta[index:index + 1]} vs {base[index:index + 1]}")

    delta = [line for line in beta_curl if line not in plain_curl]
    if delta != list(BETA_INSERT):
        findings.append(f"folded_beta_pml_curl_step's own delta over "
                        f"pml_curl_step_folded is {delta}, not {list(BETA_INSERT)}")

    for label, body in (("fused", fused), ("shipped beta curl", beta_curl)):
        try:
            at = body.index("if HAS_BETA:")
        except ValueError:
            findings.append(f"{label}: the insert is absent")
            continue
        if body[at - 1] != "curl2 = dtdx * ((b_x - b) + (a - a_y))":
            findings.append(
                f"{label}: the insert does not follow the third dtdx curl line "
                f"(found {body[at - 1]!r} before it)")
        if body[at + 3] != "at_x, at_y, at_z = i == 0, j == 0, k == 0":
            findings.append(
                f"{label}: the insert does not precede the cell-0 ownership mask "
                f"(found {body[at + 3]!r} after it)")

    invented = [line for line in fused
                if line not in set(base) | set(beta_curl)]
    if invented:
        findings.append(f"statements that appear in NEITHER shipped body: {invented}")

    if fused == twin:
        findings.append("this kernel is byte-identical to the ELECTRIC twin")
    ours, theirs = set(fused), set(twin)
    # The B side's constitutive source is B itself; the E side multiplies by
    # inverse epsilon. And the B family's near fill uses the MOVED destination
    # coefficient index (kp_d0 at the origin), which the D family does not.
    if "src0 = v0" not in ours:
        findings.append("the magnetic constitutive source `src0 = v0` is missing")
    if any("ie0" in line for line in fused):
        findings.append("the MAGNETIC weld reads an inverse-epsilon volume")
    if not any("kp_d0" in line for line in fused):
        findings.append("the near carry's moved destination coefficient (kp_d0) "
                        "is missing")
    if any("kp_d0" in line for line in twin):
        findings.append("the ELECTRIC twin carries kp_d0; the comparison is not "
                        "what it says")

    return {"leg": "transcription", "device": False,
            "statements_in_the_product": len(fused),
            "statements_in_the_base": len(base),
            "beta_insert_measured": delta,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the weld is never wider than either half
# ---------------------------------------------------------------------------

def _laptop_fixture(**keywords):
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    complex_storage = keywords.pop("complex_storage", False)
    thickness = keywords.pop("thickness", None)
    pml_thickness = keywords.pop("pml_thickness", 2)
    beta = keywords.pop("beta", BETA_CORPUS)
    mirrors = keywords.pop("mirrors", (("Y", 1),))
    grid = Grid(resolution=10.0,
                cell_size=keywords.pop("cell_size", (1.2, 1.0, 0.0)),
                dimensions=2, courant=0.35, beta=beta,
                boundaries=keywords.pop("boundaries", None),
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
                k_point=keywords.pop("k_point", (0.0, 0.0, 0.0)), **keywords)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = np.random.default_rng(17)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(np.float32)
    if thickness is None:
        folded = {("xyz".index(axis.lower())) for axis, _ in mirrors}
        thickness = tuple(
            ((0, pml_thickness) if axis in folded else
             (pml_thickness, pml_thickness)) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


class _Electric:
    field_type = "D"


class _Magnetic:
    """A magnetic source that publishes the index the injection writes."""

    field_type = "B"

    def __init__(self, index=(2, 2, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


class _MagneticWithoutIndex:
    field_type = "B"


EQUIVALENCE_CASES: Tuple[Tuple[str, Dict[str, Any], Any], ...] = (
    ("corpus", {}, (_Magnetic(),)),
    ("electric_source", {}, (_Electric(),)),
    ("undeclared_sources", {}, None),
    ("magnetic_without_an_index", {}, (_MagneticWithoutIndex(),)),
    ("zero_beta", {"beta": 0.0}, (_Magnetic(),)),
    ("complex_storage", {"complex_storage": True}, (_Magnetic(),)),
    ("inactive_absorber", {"pml_thickness": 0, "thickness": ((0, 0),) * 3},
     (_Magnetic(),)),
    ("unfolded", {"mirrors": ()}, (_Magnetic(),)),
    ("bloch", {"k_point": (0.2, 0.0, 0.0)}, (_Magnetic(),)),
)


def equivalence_leg() -> Dict[str, Any]:
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    admitted = 0
    for name, keywords, sources in EQUIVALENCE_CASES:
        fields, pml = _laptop_fixture(**dict(keywords))
        weld = product.folded_beta_fused_magnetic_pair_coverage(
            fields, pml, sources)
        curl = folded_complex.folded_beta_pml_curl_coverage(fields, pml, "step_B")
        magnetic = folded_complex.folded_beta_run_constitutive_coverage(
            fields, pml, "H")
        never_wider = (not weld.covered) or (curl.covered and magnetic.covered)
        covered = not [r for r in weld.reasons if "array module" not in r]
        rows.append({"case": name, "weld_covered_modulo_backend": covered,
                     "weld_never_wider": never_wider,
                     "weld_reasons": list(weld.reasons)[:6]})
        if not never_wider:
            findings.append(f"{name}: the weld admitted where a half refused")
        admitted += int(covered)
    if admitted == 0:
        findings.append("no case was admitted modulo the backend clause")

    fields, pml = _laptop_fixture()
    before = [r for r in product.folded_beta_fused_magnetic_pair_coverage(
        fields, pml, (_Magnetic(),)).reasons if "array module" not in r]
    original = product.CARRIES_DEPOSIT_REPAIR
    try:
        product.CARRIES_DEPOSIT_REPAIR = False
        after = [r for r in product.folded_beta_fused_magnetic_pair_coverage(
            fields, pml, (_Magnetic(),)).reasons if "array module" not in r]
    finally:
        product.CARRIES_DEPOSIT_REPAIR = original
    if before:
        findings.append(f"the corpus family is refused WITH the flag: {before}")
    if not any("is magnetic" in reason for reason in after):
        findings.append(
            "flipping CARRIES_DEPOSIT_REPAIR to False did NOT refuse the "
            "magnetic source: the flag is not what carries this cell")

    return {"leg": "equivalence", "device": False, "cases": rows,
            "admitted_modulo_backend": admitted,
            "reasons_with_the_flag": before,
            "reasons_with_the_flag_off": after,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — the corpus, re-derived from the census
# ---------------------------------------------------------------------------

CENSUS = os.path.join(HERE, "results",
                      "predicate_coverage_triton_2026-08-31b_plainrepair")
MATRIX = os.path.join(HERE, "results",
                      "fusion_matrix_triton_2026-08-31_plainrepair7",
                      "fusion_matrix.json")
CELL = ("B->H", "folded real beta PML", "folded beta run")


def corpus_admission_leg() -> Dict[str, Any]:
    findings: List[str] = []
    record: List[dict] = []
    for name in ("examples.jsonl", "tests.jsonl"):
        path = os.path.join(CENSUS, name)
        if not os.path.exists(path):
            return {"leg": "corpus_admission", "device": False,
                    "findings": [f"the census is absent: {path}"], "passed": False}
        record.extend(json.loads(line) for line in
                      open(path, encoding="utf-8").read().splitlines()
                      if line.strip())
    matched_path = os.path.join(CENSUS, "tests_param_matched.jsonl")
    if os.path.exists(matched_path):
        matched = {(r.get("leg"), r.get("row")): r for r in (
            json.loads(line) for line in
            open(matched_path, encoding="utf-8").read().splitlines()
            if line.strip())}
        record = [matched.get((r.get("leg"), r.get("row")), r) for r in record]
    record = [r for r in record if r.get("measured")]

    in_cell: List[str] = []
    magnetic: List[str] = []
    for row in record:
        configuration = row.get("configuration") or {}
        if float(configuration.get("beta") or 0.0) == 0.0:
            continue
        if not configuration.get("pml_active"):
            continue
        if configuration.get("force_complex_fields"):
            continue
        if not any(configuration.get("mirrored") or ()):
            continue
        if configuration.get("cylindrical") or configuration.get("bfast_active"):
            continue
        if configuration.get("has_bloch") or configuration.get("has_nonlinearity"):
            continue
        if configuration.get("has_offdiagonal_epsilon"):
            continue
        name = f"{row['leg']}:{row['row']}"
        in_cell.append(name)
        if "B" in tuple(configuration.get("source_field_types") or ()):
            magnetic.append(name)

    matrix_rows: List[str] = []
    blocked: List[str] = []
    if os.path.exists(MATRIX):
        matrix = json.load(open(MATRIX, encoding="utf-8"))
        entries = [entry for entry in matrix["seam_instances"]
                   if (entry["seam"], entry["curl_arm"], entry["constitutive_arm"])
                   == CELL]
        matrix_rows = sorted(entry["row"] for entry in entries)
        blocked = sorted(entry["row"] for entry in entries
                         if entry["in_seam_source_blocks"])
        if matrix_rows != sorted(in_cell):
            findings.append(
                f"this leg derived {sorted(in_cell)} from the census while the "
                f"fusion matrix scores {matrix_rows} at the same cell")
        if blocked:
            findings.append(
                f"the board says the deposit repair CANNOT carry {blocked}")
        for entry in entries:
            if entry["product"] not in (None, "folded_beta_fused_magnetic_pair"):
                findings.append(
                    f"the board names {entry['product']!r} in this cell")
    else:
        findings.append(f"the fusion matrix is absent: {MATRIX}")
    if not in_cell:
        findings.append("no row in the cell: this arm would be worth nothing")
    if not magnetic:
        findings.append(
            "no row in the cell declares a MAGNETIC source, so the deposit "
            "carry this gate exists to measure is not what makes the cell "
            "reachable")
    return {"leg": "corpus_admission", "device": False, "rows_scanned": len(record),
            "cell": list(CELL), "rows_in_the_cell": sorted(in_cell),
            "rows_declaring_a_magnetic_source": sorted(magnetic),
            "rows_the_repair_cannot_carry": blocked,
            "matrix_rows_at_the_same_cell": matrix_rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — mutation arming
# ---------------------------------------------------------------------------

def mutation_arming_leg() -> Dict[str, Any]:
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    source = shipped_source()
    for name, why, expectation, rewrite in mutation_table():
        mutated, hits = rewrite(source)
        armed = bool(hits) and mutated != source
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "rewrite_hits": hits, "armed": armed})
        if not armed:
            findings.append(f"{name}: the rewrite matched nothing")
        try:
            case_name, _case = mutation_case_for(name)
            rows[-1]["case"] = case_name
        except AssertionError as exc:
            findings.append(f"{name}: {exc}")
    return {"leg": "mutation_arming", "device": False, "mutations": rows,
            "findings": findings, "passed": not findings}


NO_DEVICE_LEGS = (transcription_leg, equivalence_leg, corpus_admission_leg,
                  mutation_arming_leg)


# ---------------------------------------------------------------------------
# The device harness
# ---------------------------------------------------------------------------

def ghost_destinations(near: Sequence[Any], far: Sequence[Any]) -> Tuple[int, int, int]:
    """Ghost cells ONE source lane owns, per B component.

    Component ``m`` is a NEAR destination on axis ``m`` alone and a FAR
    destination on the two axes that are not ``m`` (``IYEE_SHIFTS``:
    ``Bx (0,1,1)``), and the fills compose:
    ``(1 + near_m) * 2 ** (far axes other than m) - 1``.
    """
    out: List[int] = []
    for component in range(3):
        others = sum(1 for axis in range(3)
                     if axis != component and bool(far[axis]))
        out.append((1 + int(bool(near[component]))) * (2 ** others) - 1)
    return (out[0], out[1], out[2])


def build_grid(cell, boundaries, mirror_specs,
               options: Optional[Dict[str, Any]] = None,
               prefer_gpu: bool = True):
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    settings = dict(DEFAULT_OPTIONS)
    settings.update(options or {})
    driver = FdtdDriver(
        cell_size=cell, resolution=float(settings["resolution"]), dimensions=2,
        force_complex_fields=False, courant=float(settings["courant"]),
        boundaries=boundaries, beta=BETA_CORPUS,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirror_specs),
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    folded = {axis.lower() for axis, _phase in mirror_specs}
    cells = int(settings["pml"])
    driver.setup_pml({
        name: ({"high": cells} if name in folded else cells)
        for name in ("x", "y")
    })
    return driver


def case_constexprs(case) -> Dict[str, Any]:
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_beta_fused_magnetic_pair as product,
    )
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels.symmetry import (  # noqa: PLC0415
        CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, folded_axis_kinds,
    )

    driver = build_grid(case[1], case[2], case[3], case[5], prefer_gpu=False)
    try:
        grid = driver.fields.grid
        codes, reasons = folded_axis_kinds(grid, driver.pml)
        if codes is None:
            raise AssertionError(f"{case[0]}: folded_axis_kinds refused: {reasons}")
        codes = tuple(int(value) for value in codes)
        walls = tuple(bool(value) for value in zero_metal_axes(grid))
        phases = product.mirror_phases(grid)
        near = tuple(code in (CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)
                     for code in codes)
        far = tuple(code == CODE_MIRROR_PERIODIC for code in codes)
        return {
            "BCX": codes[0], "BCY": codes[1], "BCZ": codes[2],
            "NEAR": near, "FAR": far, "ZM": walls, "PH": phases,
            "shape": tuple(int(value) for value in grid.shape),
            "ghost_destinations": ghost_destinations(near, far),
        }
    finally:
        driver.close()


def seed_values(cp, driver, seed: int, value_class: str) -> None:
    rng = np.random.default_rng(seed)
    shape = tuple(driver.shape)

    def draw() -> np.ndarray:
        if value_class == "uniform":
            values = rng.uniform(-0.25, 0.25, size=shape)
        elif value_class == "signed_zero_lattice":
            picks = rng.integers(0, 2, size=shape)
            values = np.where(picks == 0, 0.0, -0.0)
        elif value_class == "subnormal_band":
            values = rng.standard_normal(shape) * 1e-38
        else:  # pragma: no cover
            raise ValueError(f"unknown value class {value_class!r}")
        return np.ascontiguousarray(values.astype(np.float32))

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(draw()))
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())


def _off_plane_center(cell, mirror_specs) -> Tuple[float, float, float]:
    center = [0.0, 0.0, 0.0]
    for axis_name, _phase in mirror_specs:
        axis = "XYZ".index(axis_name.upper())
        center[axis] = 0.25 * (cell[axis] / 2.0)
    return (center[0], center[1], center[2])


def build_driver(cp, cell, boundaries, mirror_specs, seed: int, value_class: str,
                 magnetic: bool, options: Optional[Dict[str, Any]] = None):
    """One 2-D folded REAL-beta PML driver, seeded identically for every route.

    ``magnetic=True`` is the deposit THIS product exists for — injected between
    ``step_B`` and ``update_H`` (driver.py:3283-3284). ``magnetic=False``
    carries an ELECTRIC source instead, which lands in the other seam and keeps
    the run alive without touching this one.
    """
    driver = build_grid(cell, boundaries, mirror_specs, options)
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = {
        name: np.ascontiguousarray(
            (2.25 + 0.30 * np.sin(index * np.float32(0.037 + 0.011 * offset))
             + 0.17 * offset).astype(np.float32))
        for offset, name in enumerate(("Ex", "Ey", "Ez"))}
    driver.set_epsilon_components({name: cp.asarray(values)
                                   for name, values in epsilon.items()})
    center = _off_plane_center(cell, mirror_specs)
    if magnetic:
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": center, "width": 0.4})
    else:
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": center, "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    driver.invalidate_fast_path()
    driver._fast_path = None          # noqa: SLF001 - the harness owns the route
    driver._fast_path_stale = False   # noqa: SLF001
    return driver


def inventory(driver) -> Dict[str, Any]:
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        if name.startswith("_"):
            continue
        if (getattr(value, "shape", None) == shape
                and getattr(value, "dtype", None) is not None):
            found[name] = value
    _assert_material_names_are_real(found)
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not "
            f"complete")
    return found


def words(cp, array) -> np.ndarray:
    return np.ascontiguousarray(cp.asnumpy(array)).view(np.uint32).ravel()


def snapshot(cp, driver) -> Dict[str, np.ndarray]:
    return {name: words(cp, array) for name, array in inventory(driver).items()}


def first_divergence(left: Dict[str, np.ndarray],
                     right: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
    for name in sorted(set(left) & set(right)):
        a, b = left[name], right[name]
        if a.shape != b.shape:
            return {"array": name, "reason": "shape",
                    "left": list(a.shape), "right": list(b.shape)}
        if not np.array_equal(a, b):
            where = int(np.flatnonzero(a != b)[0])
            return {"array": name, "index": where,
                    "left_word": int(a[where]), "right_word": int(b[where]),
                    "differing_words": int(np.count_nonzero(a != b))}
    only = sorted(set(left) ^ set(right))
    return ({"array": "<inventory>", "reason": "asymmetric", "names": only}
            if only else None)


def moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> List[str]:
    return [name for name in sorted(set(before) & set(after))
            if not np.array_equal(before[name], after[name])]


class CountingKernel:
    __slots__ = ("jit", "calls", "grids")

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.calls = 0
        self.grids: List[Any] = []

    def __getitem__(self, grid):
        launcher = self.jit[grid]

        def run(*args, **kwargs):
            self.calls += 1
            self.grids.append(tuple(int(value) for value in grid))
            return launcher(*args, **kwargs)

        return run


def kernel_ptx(jit: Any) -> List[str]:
    out: List[str] = []
    for per_device in (getattr(jit, "cache", None) or {}).values():
        for compiled in per_device.values():
            asm = getattr(compiled, "asm", None)
            if asm and "ptx" in asm:
                out.append(asm["ptx"])
    return out


class Route:
    def __init__(self, plans: Dict[str, Any]) -> None:
        self.plans = plans


class _Absorbed:
    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self) -> None:
        return None


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    originals = {name: getattr(driver_module, name) for name in SEAM_PASSES}

    def bump(role: str, kind: str, name: str) -> None:
        key = f"{role}/{kind}:{name}"
        counter[key] = counter.get(key, 0) + 1

    def replacement(name):
        def wrapper(fields, *args):
            role = roles.get(id(fields), "unknown")
            plan = next((candidate for owner, candidate in routes
                         if fields is owner), None)
            if plan is None or name not in plan.plans:
                bump(role, "array_path", name)
                return originals[name](fields, *args)
            bump(role, "substituted", name)
            plan.plans[name].run()
            return None
        return wrapper

    for name in SEAM_PASSES:
        setattr(driver_module, name, replacement(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


def separate_route(driver):
    """The THREE separately certified Triton products this launch replaces —
    K3a on ``step_B``, the mirror ghost fill on ``fill_B``, and the ``folded
    beta run`` constitutive on ``update_H``; the wall clear and the far fill
    stay on the ARRAY PATH and are counted."""
    from meep_gpu.triton_kernels import folded_complex, symmetry  # noqa: PLC0415

    curl = folded_complex.plan_folded_beta_pml_curl(
        driver.fields, driver.pml, "step_B", num_warps=1)
    fill = symmetry.plan_mirror_ghost_fill(driver.fields, "B")
    constitutive = folded_complex.plan_folded_beta_run_constitutive(
        driver.fields, driver.pml, "H", num_warps=1)
    missing = [name for name, plan in
               (("folded real beta PML curl", curl),
                ("mirror ghost fill", fill),
                ("folded beta run constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case")
    return Route({"step_B": curl, "fill_symmetry_bc_B": fill,
                  "update_H": constitutive})


def fused_route(plan, driver=None, sources=(), bracket: bool = True):
    """The fused plan in its two slots, BRACKETED when the seam carries a
    deposit; ``bracket=False`` is the NULL CONTROL and must diverge."""
    from meep_gpu import deposit_repair  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "B")
    if not seam or not bracket:
        return Route({name: (plan if name == "step_B" else _Absorbed(name, plan))
                      for name in SEAM_PASSES}), None
    leading = deposit_repair.LeadingRepairPlan(plan, driver.fields, driver.pml,
                                               seam, "B")
    trailing = deposit_repair.TrailingRepairPlan("update_H", leading,
                                                 driver.fields, driver.pml)
    plans = {name: _Absorbed(name, plan) for name in SEAM_PASSES}
    plans["step_B"] = leading
    plans["update_H"] = trailing
    return Route(plans), leading


def _swapped_coefficients(driver, plan, group: str):
    """One coefficient group moved to the WRONG lattice. On the B side the curl
    half takes the HALF-INTEGER split-field pair and the constitutive the
    INTEGER kps/kms — the exact mirror of the electric twin."""
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

    for slot in ("_curl_coefficients", "_h_coefficients"):
        if not hasattr(plan, slot):
            raise AssertionError(
                f"the plan has no {slot}; this host mutation would bind nothing")
    pml = driver.pml
    if group == "curl":
        arrays = [getattr(pml, f"{stem}_{axis}")            # INTEGER, wrongly
                  for axis in "xyz" for stem in ("kms", "sinv")]
        plan._curl_coefficients = tuple(  # noqa: SLF001
            CupyPointer(_flat(a)) for a in arrays)
    elif group == "constitutive":
        arrays = [getattr(pml, f"{stem}_{axis}_h")          # HALF-INTEGER, wrongly
                  for axis in "xyz" for stem in ("kps", "kms")]
        plan._h_coefficients = tuple(  # noqa: SLF001
            CupyPointer(_flat(a)) for a in arrays)
    else:  # pragma: no cover
        raise ValueError(group)
    return plan


def _swap_beta_words(plan):
    plan.beta_plus, plan.beta_minus = plan.beta_minus, plan.beta_plus
    return plan


def host_mutation_table() -> Tuple[Tuple[str, str, str,
                                         Callable[[Any, Any], Any]], ...]:
    return (
        ("h_curl_lattice_swapped",
         "the curl half bound the INTEGER split-field pair, which is the "
         "constitutive side's lattice on this seam",
         "caught", lambda driver, plan: _swapped_coefficients(driver, plan, "curl")),
        ("h_constitutive_lattice_swapped",
         "the constitutive half bound the HALF-INTEGER kps/kms pair, which is "
         "the curl side's lattice on this seam",
         "caught",
         lambda driver, plan: _swapped_coefficients(driver, plan, "constitutive")),
        ("h_beta_words_swapped",
         "beta_plus and beta_minus exchanged; they differ in SIGN",
         "caught", lambda driver, plan: _swap_beta_words(plan)),
    )


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, mirrors, steps: int, product,
            value_class: str = "uniform", mutant: Any = None,
            host_mutation: Any = None, install_fused: bool = True,
            freeze_magnetic: bool = False, magnetic: bool = False,
            bracket: bool = True,
            options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    seed = case_seed(name, str(cell), str(boundaries), str(mirrors), value_class)
    made = [build_driver(cp, cell, boundaries, mirrors, seed, value_class,
                         magnetic, options) for _ in range(3)]
    reference, separate, fused = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_beta_fused_curl_constitutive_B_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "mirrors": [list(entry) for entry in mirrors],
        "value_class": value_class, "seed": seed,
        "magnetic_source": bool(magnetic), "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "magnetic_frozen": bool(freeze_magnetic),
        "host_mutation": None, "first_divergence": None,
        "control_divergence": None,
    }
    try:
        plan = product.plan_folded_beta_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.folded_beta_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if host_mutation is not None:
            row["host_mutation"] = host_mutation[0]
            plan = host_mutation[3](fused, plan)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["zero_metal_flags"] = list(plan.zero_metal)
        row["beta_words"] = [plan.beta_plus, plan.beta_minus, plan.has_beta]
        row["near"] = [bool(value) for value in plan.near]
        row["far"] = [bool(value) for value in plan.far]
        row["reflect_rows"] = list(plan.reflect)
        row["ghost_destinations_per_source_lane"] = list(
            ghost_destinations(plan.near, plan.far))
        if install_fused:
            route, leading = fused_route(plan, fused, tuple(fused._sources),
                                         bracket=bracket)
        else:
            route = Route({})
        separate_plans = separate_route(separate)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_magnetic:
            frozen = {pass_name: _Absorbed(pass_name, None)
                      for pass_name in SEAM_PASSES}
            route.plans.update(frozen)
            separate_plans.plans.update(frozen)
            routes.append((reference.fields, Route(dict(frozen))))
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, fused)
        reference_opening = snapshot(cp, reference)
        row["per_step"] = []
        started = time.time()
        for step in range(1, steps + 1):
            before = snapshot(cp, fused)
            reference.step()
            separate.step()
            fused.step()
            cp.cuda.runtime.deviceSynchronize()
            after_reference = snapshot(cp, reference)
            after_separate = snapshot(cp, separate)
            after_fused = snapshot(cp, fused)
            versus_array = first_divergence(after_fused, after_reference)
            versus_separate = first_divergence(after_fused, after_separate)
            control = first_divergence(after_separate, after_reference)
            step_moved = moved(before, after_fused)
            row["per_step"].append({
                "step": step,
                "fused_vs_array": versus_array,
                "fused_vs_separate_certified_products": versus_separate,
                "oracle_control_vs_array": control,
                "arrays_moved": len(step_moved),
                "fused_launches": kernel.calls,
                "deposit_repairs": (leading.repairs if leading is not None
                                    else None),
            })
            divergence = versus_array or versus_separate
            if step == 1 or step % 10 == 0 or divergence is not None:
                log(f"  {name} [{value_class}] step {step}/{steps} "
                    f"identical={divergence is None} control={control is None} "
                    f"moved={len(step_moved)} launches={kernel.calls} "
                    f"repairs={leading.repairs if leading is not None else '-'} "
                    f"({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, fused)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        reference_final = snapshot(cp, reference)
        reference_moved = moved(reference_opening, reference_final)
        row["arrays_total"] = len(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["reference_never_moved"] = sorted(
            set(reference_final) - set(reference_moved) - set(material))
        row["inert_here_but_moving_on_the_array_path"] = sorted(
            (set(final) - set(ever_moved) - set(material)) & set(reference_moved))
        row["seam_outputs_moved"] = sorted(set(ever_moved) & set(SEAM_OUTPUTS))
        row["reference_seam_outputs_moved"] = sorted(
            set(reference_moved) & set(SEAM_OUTPUTS))
        row["material_changed"] = sorted(key for key in material
                                         if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(reference_final))
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["deposit_repairs"] = leading.repairs if leading is not None else None
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        return row
    finally:
        undo()
        for target in made:
            try:
                target.close()
            except Exception:  # noqa: BLE001
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_launches_at_least: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False,
               require_array_reference: bool = True,
               require_ghost_destinations: Optional[int] = None
               ) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if require_ghost_destinations is not None:
        observed = row.get("ghost_destinations_per_source_lane") or []
        if max(observed or [0]) != int(require_ghost_destinations):
            failures.append(
                f"this leg was declared to reach composition depth "
                f"{require_ghost_destinations} and reached "
                f"{max(observed or [0])} ({observed})")
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if not require_identical and row.get("first_divergence") is None:
        failures.append(
            "this leg REQUIRES divergence and found none: the control it "
            "exists to be is inert")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path: "
            f"{row['control_divergence']}")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} "
            f"times, expected {require_launches}")
    if (require_launches_at_least is not None
            and (row.get("fused_kernel_launches") or 0) < require_launches_at_least):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} "
            f"times, fewer than the {require_launches_at_least} this leg needs")
    if require_moved:
        swallowed = row.get("inert_here_but_moving_on_the_array_path") or []
        if swallowed:
            failures.append(
                f"the fused route left {swallowed} INERT while the array path "
                f"moves them")
        reference_movers = set(row.get("reference_seam_outputs_moved") or ())
        if not reference_movers:
            failures.append(
                "VACUOUS: the ARRAY PATH itself moves nothing this seam writes")
        movers = set(row.get("seam_outputs_moved") or ())
        if movers != reference_movers:
            failures.append(
                f"the fused route's seam-output movement {sorted(movers)} is "
                f"not the array path's {sorted(reference_movers)}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    if row.get("inventory_asymmetry_vs_array"):
        failures.append(
            f"the compared inventories differ by NAME: "
            f"{row['inventory_asymmetry_vs_array']}")
    if require_repairs and not row.get("deposit_repairs"):
        failures.append(
            "the carry leg repaired NO deposit cell: the bracket ran but the "
            "seam carried nothing")
    counts = row.get("launches") or {}
    fell_back = {key: value for key, value in counts.items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: "
            f"{fell_back}")
    if require_array_reference:
        steps = len(row.get("per_step") or ())
        for name in SEAM_PASSES:
            seen = counts.get(f"array/array_path:{name}", 0)
            if seen != steps:
                failures.append(
                    f"the reference route reached the array path for {name} "
                    f"{seen} times, expected {steps}: the fast path was not off")
    return (not failures), failures


# ---------------------------------------------------------------------------
# THE REDUCTION LEG — HAS_BETA = 0 must BE the shipped plain folded kernel
# ---------------------------------------------------------------------------

def reduction_leg(cp, product, cell, boundaries, mirrors, steps: int,
                  value_class: str = "uniform") -> Dict[str, Any]:
    """``HAS_BETA = 0`` reproduces ``folded_fused_curl_constitutive_B`` bit for
    bit while DIFFERING from the array path. NO MAGNETIC SOURCE HERE: both
    fused routes are unbracketed, so an in-seam deposit would put both one
    injection behind the array path for the wrong reason."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_fused_magnetic_pair  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels.symmetry import (  # noqa: PLC0415
        _far_reflect_rows, folded_axis_kinds,
    )

    seed = case_seed("reduction", str(cell), str(boundaries), str(mirrors),
                     value_class)
    made = [build_driver(cp, cell, boundaries, mirrors, seed, value_class,
                         magnetic=False) for _ in range(3)]
    reference, shipped, reduced = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    row: Dict[str, Any] = {"leg": "reduction:has_beta_zero", "device": True,
                           "steps_budget": steps, "value_class": value_class,
                           "seed": seed, "shape": list(reference.shape),
                           "boundaries": boundaries,
                           "mirrors": [list(entry) for entry in mirrors]}
    try:
        reduced_plan = product.plan_folded_beta_fused_magnetic_pair(
            reduced.fields, reduced.pml, tuple(reduced._sources), num_warps=1)
        if reduced_plan is None:
            raise AssertionError("the product refused the reduction case")
        reduced_plan.has_beta = 0

        fields, pml = shipped.fields, shipped.pml
        grid = fields.grid
        arrays = {name: getattr(fields, name) for name in
                  ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
                   "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                   "f_w_Hx", "f_w_Hy", "f_w_Hz")}
        curl_flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}_h")
                     for axis in "xyz" for stem in ("kms", "sinv")}
        constitutive_flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                             for axis in "xyz" for stem in ("kps", "kms")}
        codes, reasons = folded_axis_kinds(grid, pml)
        if codes is None:
            raise AssertionError(f"folded_axis_kinds refused: {reasons}")
        shipped_plan = (folded_fused_magnetic_pair
                        .plan_folded_fused_magnetic_pair_from_arrays(
                            arrays, curl_flat, constitutive_flat, codes,
                            zero_metal_axes(grid), product.mirror_phases(grid),
                            grid.dt / grid.dx, num_warps=1,
                            reflect=_far_reflect_rows(grid) or (None, None, None)))

        roles = {id(reference.fields): "array", id(shipped.fields): "shipped",
                 id(reduced.fields): "reduced"}
        routes = [(shipped.fields, fused_route(shipped_plan)[0]),
                  (reduced.fields, fused_route(reduced_plan)[0])]
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, reduced)
        row["per_step"] = []
        for step in range(1, steps + 1):
            reference.step()
            shipped.step()
            reduced.step()
            cp.cuda.runtime.deviceSynchronize()
            after_reference = snapshot(cp, reference)
            after_shipped = snapshot(cp, shipped)
            after_reduced = snapshot(cp, reduced)
            versus_shipped = first_divergence(after_reduced, after_shipped)
            versus_array = first_divergence(after_reduced, after_reference)
            row["per_step"].append({
                "step": step,
                "has_beta_zero_vs_shipped_folded_pair": versus_shipped,
                "has_beta_zero_vs_array_path": versus_array,
            })
            log(f"  reduction [{value_class}] step {step}/{steps} "
                f"reduced==shipped:{versus_shipped is None} "
                f"reduced!=array:{versus_array is not None}")
            row["first_divergence"] = versus_shipped
            row["divergence_from_the_array_path"] = versus_array
            if versus_shipped is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, reduced)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["launches"] = dict(counter)
        failures: List[str] = []
        if row["first_divergence"] is not None:
            failures.append(f"HAS_BETA=0 is NOT the shipped folded magnetic "
                            f"pair: {row['first_divergence']}")
        seam_movers = sorted(set(ever_moved) & set(SEAM_OUTPUTS))
        if not seam_movers:
            failures.append("VACUOUS: nothing this seam writes ever moved")
        row["discriminating"] = row.get("divergence_from_the_array_path") is not None
        if not row["discriminating"]:
            row["not_discriminating_because"] = (
                "HAS_BETA=0 matched the ARRAY PATH too, so this value class "
                "cannot separate the beta kernel from the plain folded one")
        row["passed"], row["failures"] = not failures, failures
        return row
    finally:
        undo()
        for target in made:
            try:
                target.close()
            except Exception:  # noqa: BLE001
                pass
        cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant; ``_carry_ghost`` is imported the way the
    shipped module imports it, so the mutant can call it."""
    header = ("import triton\nimport triton.language as tl\n"
              "from meep_gpu.triton_kernels.symmetry import (\n"
              "    CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC)\n"
              "from meep_gpu.triton_kernels.folded_fused_magnetic_pair import "
              "_carry_ghost\n"
              "PERIODIC = tl.constexpr(CODE_PERIODIC)\n"
              "MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)\n"
              "MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)\n\n")
    text = header + source.replace(KERNEL_NAME, kernel_name)
    ast.parse(text)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_folded_beta_magnetic.py", delete=False,
        encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_beta_magnetic_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    lines = source.splitlines()
    significant = [index for index, line in enumerate(lines)
                   if line.strip() and not line.strip().startswith("#")]
    needle = [fragment.strip() for fragment in before]
    hits = 0
    cursor = 0
    while cursor <= len(significant) - len(needle):
        span = significant[cursor:cursor + len(needle)]
        if [lines[index].strip() for index in span] == needle:
            first, last = span[0], span[-1]
            indent = lines[first][:len(lines[first]) - len(lines[first].lstrip())]
            lines[first:last + 1] = [
                (indent + fragment) if fragment.strip() else fragment
                for fragment in after]
            hits += 1
            significant = [index for index, line in enumerate(lines)
                           if line.strip() and not line.strip().startswith("#")]
            cursor = significant.index(first) + len(after)
        else:
            cursor += 1
    return "\n".join(lines), hits


def _replace_all(source: str, before: str, after: str) -> Tuple[str, int]:
    hits = source.count(before)
    return source.replace(before, after), hits


def mutation_table() -> Tuple[Tuple[str, str, str,
                                    Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite)."""

    def m_beta_term_dropped(source: str) -> Tuple[str, int]:
        return _replace_all(source, "if HAS_BETA:", "if HAS_BETA and False:")

    def m_beta_partners_swapped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["curl0 = curl0 - (beta_plus * b)",
             "curl1 = curl1 - (beta_minus * a)"],
            ["curl0 = curl0 - (beta_plus * a)",
             "curl1 = curl1 - (beta_minus * b)"])

    def m_beta_sign_flipped(source: str) -> Tuple[str, int]:
        return _replace_all(source, "curl0 = curl0 - (beta_plus * b)",
                            "curl0 = curl0 + (beta_plus * b)")

    def m_beta_scaled_by_dtdx(source: str) -> Tuple[str, int]:
        return _replace_all(source, "curl1 = curl1 - (beta_minus * a)",
                            "curl1 = curl1 - (dtdx * (beta_minus * a))")

    def m_beta_below_the_masks(source: str) -> Tuple[str, int]:
        """The insert moved BELOW both ownership masks; the B side's top-plane
        block ends at the BCY -> curl2 pair."""
        without, removed = _rewrite_block(source, list(BETA_INSERT), ["pass"])
        if not removed:
            return source, 0
        return _rewrite_block(
            without,
            ["if BCY == MIRROR_PERIODIC:",
             "curl2 = tl.where(last_y, 0.0, curl2)"],
            ["if BCY == MIRROR_PERIODIC:",
             "    curl2 = tl.where(last_y, 0.0, curl2)",
             "if HAS_BETA:",
             "    curl0 = curl0 - (beta_plus * b)",
             "    curl1 = curl1 - (beta_minus * a)"])

    def m_near_parity_dropped(source: str) -> Tuple[str, int]:
        """``PHX * v0`` -> ``v0`` on Bx's near image."""
        return _replace_all(source, "idx + dn_x, PHX * v0, kp_d0",
                            "idx + dn_x, v0, kp_d0")

    def m_far_parity_not_negated(source: str) -> Tuple[str, int]:
        return _replace_all(
            source,
            "_carry_ghost(f0, w0, h0, idx + df_y, -PHY * v0, kp_0, km_0, own0 & far_j)",
            "_carry_ghost(f0, w0, h0, idx + df_y, PHY * v0, kp_0, km_0, own0 & far_j)")

    def m_far_carry_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["if FAR_Y:",
             "_carry_ghost(f0, w0, h0, idx + df_y, -PHY * v0, kp_0, km_0, own0 & far_j)"],
            ["if FAR_Y:", "    pass"])

    def m_near_carry_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["if NEAR_X:",
             "_carry_ghost(f0, w0, h0, idx + dn_x, PHX * v0, kp_d0, km_d0, own0 & near_i)"],
            ["if NEAR_X:", "    pass"])

    def m_composite_parity_short(source: str) -> Tuple[str, int]:
        """``-PHX * PHY * v0`` -> ``-PHX * v0``: the near/far corner loses one
        plane's parity."""
        return _replace_all(source, "idx + dn_x + df_y, -PHX * PHY * v0",
                            "idx + dn_x + df_y, -PHX * v0")

    def m_far_row_fixed_n_minus_2(source: str) -> Tuple[str, int]:
        return _replace_all(source, "far_j = live & (j == ry)",
                            "far_j = live & (j == ny - 2)")

    def m_ownership_and_dropped(source: str) -> Tuple[str, int]:
        return _replace_all(
            source,
            "_carry_ghost(f0, w0, h0, idx + dn_x, PHX * v0, kp_d0, km_d0, own0 & near_i)",
            "_carry_ghost(f0, w0, h0, idx + dn_x, PHX * v0, kp_d0, km_d0, live & near_i)")

    def m_near_coefficient_not_moved(source: str) -> Tuple[str, int]:
        """The near image billed at the SOURCE row's coefficients instead of
        the destination's origin pair — the moved index is the B family's own
        measured rule.

        AN EXPECTED NULL, and the first device run is what measured it
        (2026-09-01, the GPU host GPU 3): the rewrite exchanges ``kp0[0]`` for
        ``kp0[2]`` on the INTEGER-lattice H coefficients, and a folded axis
        absorbs on its HIGH face only, so rows 0..2 sit outside every
        admissible absorber and the two words are byte-equal on every
        configuration this predicate can admit. The sibling
        ``folded_fused_magnetic_pair`` gate records its M1 the same way. The
        premise is MEASURED per run (``null_premise`` on the row) rather than
        asserted, and ``m_far_coefficient_rebilled_at_origin`` below is the
        CATCHABLE mutation of the same billing seam.
        """
        return _replace_all(
            source,
            "_carry_ghost(f0, w0, h0, idx + dn_x, PHX * v0, kp_d0, km_d0, own0 & near_i)",
            "_carry_ghost(f0, w0, h0, idx + dn_x, PHX * v0, kp_0, km_0, own0 & near_i)")

    def m_far_coefficient_rebilled_at_origin(source: str) -> Tuple[str, int]:
        """By's far-on-x image billed at the ORIGIN pair instead of its own
        row's — the catchable inverse of the null above: the destination of the
        far-on-x image keeps the lane's own j, whose ``kp1[j]`` differs from
        ``kp1[0]`` wherever j is inside the y absorber."""
        return _replace_all(
            source,
            "_carry_ghost(f1, w1, h1, idx + df_x, -PHX * v1, kp_1, km_1, own1 & far_i)",
            "_carry_ghost(f1, w1, h1, idx + df_x, -PHX * v1, kp_d1, km_d1, own1 & far_i)")

    def m_wall_clear_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)"],
            ["if ZM_X:", "    pass"])

    return (
        ("m_beta_term_dropped",
         "no beta term at all — the plain folded magnetic pair on a beta run",
         "caught", m_beta_term_dropped),
        ("m_far_coefficient_rebilled_at_origin",
         "By's far-on-x image billed at the origin coefficient pair instead of "
         "its own row's; kp1[j] differs from kp1[0] wherever j is inside the y "
         "absorber",
         "caught", m_far_coefficient_rebilled_at_origin),
        ("m_beta_partners_swapped",
         "each beta target takes the OTHER source's centre value",
         "caught", m_beta_partners_swapped),
        ("m_beta_sign_flipped",
         "curl + (c * g) instead of curl - (c * g)",
         "caught", m_beta_sign_flipped),
        ("m_beta_scaled_by_dtdx",
         "the beta term multiplied by dtdx; it is an analytic derivative and "
         "is not",
         "caught", m_beta_scaled_by_dtdx),
        ("m_beta_below_the_masks",
         "the insert moved below BOTH ownership masks",
         "caught", m_beta_below_the_masks),
        ("m_near_parity_dropped",
         "the near image loses its declared parity; visible only at an ODD "
         "plane",
         "caught", m_near_parity_dropped),
        ("m_far_parity_not_negated",
         "the far image drops the minus MEEP's periodic fold carries",
         "caught", m_far_parity_not_negated),
        ("m_far_carry_dropped",
         "the far ghost plane is never imaged",
         "caught", m_far_carry_dropped),
        ("m_near_carry_dropped",
         "the near ghost plane is never imaged",
         "caught", m_near_carry_dropped),
        ("m_composite_parity_short",
         "the near/far corner composes ONE parity instead of two",
         "caught", m_composite_parity_short),
        ("m_far_row_fixed_n_minus_2",
         "the reflect row baked as n - 2, a whole cell wrong at an odd count",
         "caught", m_far_row_fixed_n_minus_2),
        ("m_ownership_and_dropped",
         "the near carry images from a lane the ownership rule excludes",
         "caught", m_ownership_and_dropped),
        ("m_near_coefficient_not_moved",
         "the near image billed at the source row's coefficients instead of "
         "the destination's origin pair — an EXPECTED NULL: the integer-lattice "
         "H coefficients are byte-equal at rows 0 and 2 on every admissible "
         "folded configuration (a folded axis absorbs on its HIGH face only), "
         "measured per run as null_premise; the far-side rebilling above is "
         "the catchable mutation of the same seam",
         "null", m_near_coefficient_not_moved),
        ("m_wall_clear_dropped",
         "zero_metal_B's x-wall row is never applied",
         "caught", m_wall_clear_dropped),
    )


def mutation_case_for(name: str) -> Tuple[str, Tuple[Any, ...]]:
    case_name = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    case = CASES_BY_NAME[case_name]
    constexprs = case_constexprs(case)
    if name in ("m_near_parity_dropped",):
        # Bx's near image is on the X axis; the DEFAULT case folds x at +1,
        # so this one is scored on a case whose... the near parity that must
        # be odd is the X one for component 0. odd_y_fold_periodic folds only
        # y, whose near component is By (PHY on v1) — so this mutation targets
        # PHX*v0 and must be scored on a case with an ODD X fold.
        if constexprs["PH"][0] != -1:
            raise AssertionError(
                f"{name} is scored on {case_name}, whose X parity is "
                f"{constexprs['PH'][0]}; a +1 plane makes the multiply a no-op")
    if name in ("m_far_parity_not_negated", "m_far_carry_dropped",
                "m_far_row_fixed_n_minus_2"):
        if not constexprs["FAR"][1]:
            raise AssertionError(
                f"{name} is scored on {case_name}, whose y axis images no far "
                f"ghost; the rewritten branch never executes")
    if name == "m_far_row_fixed_n_minus_2":
        from meep_gpu.triton_kernels.symmetry import _far_reflect_rows  # noqa: PLC0415

        driver = build_grid(case[1], case[2], case[3], case[5], prefer_gpu=False)
        try:
            rows = _far_reflect_rows(driver.fields.grid) or (None, None, None)
            stored = constexprs["shape"][1]
            if rows[1] is None or int(rows[1]) == stored - 2:
                raise AssertionError(
                    f"{name} is scored on {case_name}, whose y reflect row "
                    f"{rows[1]} equals stored - 2 = {stored - 2}; the baked "
                    f"guess is right there and the mutation measures nothing")
        finally:
            driver.close()
    if name in ("m_composite_parity_short",):
        if not (constexprs["NEAR"][0] and constexprs["FAR"][1]):
            raise AssertionError(
                f"{name} is scored on {case_name}, which does not compose "
                f"NEAR_X with FAR_Y; the corner never executes")
        if constexprs["PH"][1] == 1:
            raise AssertionError(
                f"{name} is scored on {case_name}, whose Y parity is +1; the "
                f"shortened product equals the full one there")
    if name in ("m_near_carry_dropped", "m_ownership_and_dropped",
                "m_near_coefficient_not_moved"):
        if not constexprs["NEAR"][0]:
            raise AssertionError(
                f"{name} is scored on {case_name}, which does not fold x; "
                f"Bx's near block never executes")
    if name in ("m_wall_clear_dropped", "m_beta_below_the_masks"):
        if not constexprs["ZM"][0]:
            raise AssertionError(
                f"{name} is scored on {case_name}, whose x axis carries no "
                f"wall; the rewritten rows never execute")
    return case_name, case


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    source = shipped_source()
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "device": True}
        if not hits or mutated == source:
            row.update(caught=False, error="the rewrite matched nothing")
            rows.append(row)
            continue
        try:
            case_name, case = mutation_case_for(name)
            if name == "m_near_coefficient_not_moved":
                # THE NULL'S PREMISE, MEASURED rather than asserted: the words
                # the rewrite exchanges must be byte-equal on the scored grid,
                # or the null declaration is wrong and the row fails.
                premise_driver = build_grid(case[1], case[2], case[3], case[5])
                try:
                    kp = words(cp, premise_driver.pml.kps_x)
                    km = words(cp, premise_driver.pml.kms_x)
                    row["null_premise"] = {
                        "kp_x0_equals_kp_x2": bool(kp[0] == kp[2]),
                        "km_x0_equals_km_x2": bool(km[0] == km[2]),
                    }
                finally:
                    premise_driver.close()
                if not all(row["null_premise"].values()):
                    row.update(caught=False, error=(
                        "the null's premise is violated on the scored case: "
                        f"{row['null_premise']} — the coefficient words the "
                        "rewrite exchanges DIFFER, so this mutation must be "
                        "re-declared caught, not waved through as a null"))
                    rows.append(row)
                    continue
            kernel = compile_mutant(mutated, f"{KERNEL_NAME}_m{index}")
            leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3],
                          MUTATION_STEPS, product, value_class="uniform",
                          mutant=kernel, magnetic=True, options=case[5])
            row["case"] = case_name
            row["caught"] = leg.get("first_divergence") is not None
            row["diverged_at_step"] = leg.get("diverged_at_step")
            row["first_divergence"] = leg.get("first_divergence")
            row["fused_kernel_launches"] = leg.get("fused_kernel_launches")
        except Exception as exc:  # noqa: BLE001
            row.update(caught=False, error=f"{type(exc).__name__}: {exc}"[:400])
        rows.append(row)
        log(f"  mutation {name}: caught={row.get('caught')} "
            f"(expected {expectation}) {row.get('error') or ''}")
    return rows


def run_host_mutations(cp, product) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    case = CASES_BY_NAME[DEFAULT_MUTATION_CASE]
    for entry in host_mutation_table():
        name, why, expectation, _apply = entry
        leg = run_leg(cp, f"host_mutation:{name}", case[1], case[2], case[3],
                      MUTATION_STEPS, product, value_class="uniform",
                      host_mutation=entry, magnetic=True, options=case[5])
        caught = leg.get("first_divergence") is not None
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "caught": caught, "device": True, "leg": leg})
        log(f"  host mutation {name}: caught={caught} (expected {expectation})")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def _deposit(driver, component: str):
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    grid = driver.fields.grid
    cell = tuple(float(v) for v in grid.cell_size)
    center = [0.0, 0.0, 0.0]
    for axis in range(3):
        if bool(grid.is_mirrored(axis)):
            center[axis] = 0.25 * (cell[axis] / 2.0)
    source = VolumeSource(grid=grid, component=component,
                          center=tuple(center), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    if not source._n_source_points:
        raise AssertionError("the fixture source deposits nothing")
    if deposit_repair._deposit_index(source) is None:
        raise AssertionError("the fixture source publishes no deposit index")
    return source


def run_refusals(cp, product) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    corpus = CASES_BY_NAME["corpus_y_fold_periodic"]
    for name, mirrors, sources_of, needle, admit in (
        ("magnetic_source_with_a_deposit_index", corpus[3],
         lambda d: (_deposit(d, "Hz"),), None, True),
        ("electric_source_only", corpus[3],
         lambda d: (_deposit(d, "Ez"),), None, True),
        ("magnetic_source_without_a_deposit_index", corpus[3],
         lambda d: (_MagneticWithoutIndex(),), "does not publish the index",
         False),
        ("undeclared_source_list", corpus[3], lambda d: None,
         "was not declared", False),
        ("unfolded_grid", (), lambda d: (_deposit(d, "Hz"),),
         "mirror", False),
    ):
        driver = build_grid(corpus[1], corpus[2], mirrors, corpus[5])
        try:
            sources = sources_of(driver)
            verdict = product.folded_beta_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources)
            built = product.plan_folded_beta_fused_magnetic_pair(
                driver.fields, driver.pml, sources)
            row = {"refusal": name, "covered": bool(verdict.covered),
                   "reasons": list(verdict.reasons)[:6],
                   "plan_is_None": built is None, "admits": admit}
            if admit:
                row["passed"] = bool(verdict.covered) and built is not None
            else:
                row["passed"] = (not verdict.covered) and built is None and any(
                    needle in reason for reason in verdict.reasons)
            rows.append(row)
            log(f"  refusal {name}: passed={row['passed']}")
        finally:
            driver.close()
    return rows


# ---------------------------------------------------------------------------
# Environment and main
# ---------------------------------------------------------------------------

def environment(cp: Any = None) -> Dict[str, Any]:
    import platform  # noqa: PLC0415

    out: Dict[str, Any] = {
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
    }
    try:
        import socket  # noqa: PLC0415

        out["hostname"] = socket.gethostname()
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton  # noqa: PLC0415

        out["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        out["triton"] = f"unavailable: {exc!r}"
    if cp is not None:
        try:
            out["cupy"] = cp.__version__
            device = cp.cuda.Device()
            out["compute_capability"] = device.compute_capability
            out["device"] = cp.cuda.runtime.getDeviceProperties(
                device.id)["name"].decode()
        except Exception as exc:  # noqa: BLE001
            out["cupy"] = f"unavailable: {exc!r}"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_folded_beta_fused_magnetic_pair"),
        help="a DIRECTORY; gate.json is written inside it")
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--subnormal-policy", default="keep")
    args = parser.parse_args(argv)

    out = args.out
    artifact = out if out.endswith(".json") else os.path.join(out, "gate.json")
    if not out.endswith(".json"):
        os.makedirs(out, exist_ok=True)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_folded_beta_fused_magnetic_pair",
        "product": PRODUCT_MODULE,
        "kernel": KERNEL_NAME,
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": triton_device_identity.record(environment()),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "kernels.DEFAULT_BLOCK",
                   "subnormal_policy": args.subnormal_policy},
        "value_classes": list(VALUE_CLASSES),
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "host_mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in NO_DEVICE_LEGS:
        started = time.time()
        try:
            row = leg()
        except Exception as exc:  # noqa: BLE001
            row = {"leg": getattr(leg, "__name__", str(leg)), "device": False,
                   "error": repr(exc), "passed": False, "findings": [repr(exc)]}
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} "
            f"findings={row.get('findings')} ({row['seconds']} s)")
        save(payload, artifact)

    if args.no_device:
        payload["device_status"] = (
            "UNRUN — invoked with --no-device; no CUDA leg, mutation or refusal "
            "in this artifact")
        payload["passed"] = all(row["passed"] for row in payload["no_device_legs"])
        payload["release"] = {
            "released": False,
            "reasons": ["the no-device legs passed, but no device leg, mutation "
                        "or refusal has run: this artifact releases nothing"],
        }
        save(payload, artifact)
        log(f"\nno-device verdict: {payload['passed']}  ->  {artifact}")
        return 0 if payload["passed"] else 1

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    product = importlib.import_module(PRODUCT_MODULE)
    backends.guard_kernel_compilation(cp)
    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "quiet_cases": {case[0]: case[4] for case in CASES},
        "carry_cases": list(CARRY_CASES),
        "steps_per_carry": CARRY_STEPS,
        "steps_per_mutation": MUTATION_STEPS,
        "value_classes": list(VALUE_CLASSES),
    }
    save(payload, artifact)

    log("\n=== device legs: the QUIET family (an electric source, no seam deposit) ===")
    for name, cell, boundaries, mirrors, steps, options in CASES:
        expect = options.get("expect_ghost_destinations")
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"quiet:{name}", cell, boundaries, mirrors, steps,
                          product, value_class=value_class, options=options)
            passed, failures = verdict_of(row, require_launches=steps,
                                          require_ghost_destinations=expect)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== device legs: the CARRY family (a real magnetic deposit in the seam) ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, mirrors, _steps, options = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{name}", cell, boundaries, mirrors,
                          CARRY_STEPS, product, value_class=value_class,
                          magnetic=True, options=options)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, mirrors, _steps, options = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{name}", cell, boundaries, mirrors,
                      CARRY_STEPS, product, magnetic=True, bracket=False,
                      options=options)
        passed, failures = verdict_of(row, require_identical=False,
                                      require_launches_at_least=1,
                                      require_moved=False,
                                      require_array_reference=False)
        row["armed"] = True
        row["passed"], row["failures"] = passed, failures
        row["why"] = ("the fused launch WITHOUT the shipped deposit repair; it "
                      "consumes a pre-injection B and MUST diverge")
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== the REDUCTION leg: HAS_BETA = 0 is the shipped plain folded kernel ===")
    reductions = []
    for value_class in VALUE_CLASSES:
        name, cell, boundaries, mirrors, _steps, options = CASES_BY_NAME[
            "corpus_y_fold_periodic"]
        row = reduction_leg(cp, product, cell, boundaries, mirrors, 4,
                            value_class)
        reductions.append(row)
        payload["device_legs"].append(row)
        save(payload, artifact)
    if not any(row.get("discriminating") for row in reductions):
        payload["device_legs"].append({
            "leg": "reduction:discrimination_floor", "device": True,
            "passed": False,
            "failures": ["NO value class separated HAS_BETA=0 from the array "
                         "path, so every reduction agreement above is a "
                         "tautology"],
        })
        save(payload, artifact)

    log("\n=== armed harness mutations ===")
    name, cell, boundaries, mirrors, _steps, options = CASES_BY_NAME[
        DEFAULT_MUTATION_CASE]
    row = run_leg(cp, "armed:no_substitution", cell, boundaries, mirrors, 3,
                  product, install_fused=False, magnetic=True, options=options)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree "
                  "and the counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    row = run_leg(cp, "armed:frozen_magnetic_seam", cell, boundaries, mirrors, 3,
                  product, freeze_magnetic=True, magnetic=True, options=options)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's magnetic seam is inert; all three agree "
                  "trivially and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.folded_beta_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, artifact)

    log("\n=== armed host mutations ===")
    payload["host_mutations"] = run_host_mutations(cp, product)
    save(payload, artifact)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, artifact)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    def _mutation_ok(row: Dict[str, Any]) -> bool:
        if row.get("error") is not None:
            return False
        if not row.get("rewrite_hits", 1):
            return False
        return bool(row.get("caught")) is (row["expectation"] == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    host_ok = all(bool(row.get("caught")) is (row["expectation"] == "caught")
                  for row in payload["host_mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and mutation_ok and host_ok
        and all(row["passed"] for row in payload["no_device_legs"]))
    payload["device_status"] = "RUN"
    payload["release"] = {
        "released": payload["passed"],
        "reasons": ([] if payload["passed"] else
                    [f"device legs ok: {device_ok}",
                     "mutations whose measured outcome did not match their "
                     "declared expectation, or that were never armed: "
                     + str(sorted(row["mutation"] for row in payload["mutations"]
                                  if not _mutation_ok(row))),
                     f"refusals ok: {refusal_ok}",
                     f"kernel mutations ok: {mutation_ok}",
                     f"host mutations ok: {host_ok}"]),
        "host": "the measurement machine; every device row above ran there",
    }
    save(payload, artifact)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"\nverdict: passed={payload['passed']} "
        f"released={payload['release']['released']}  ->  {artifact}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
