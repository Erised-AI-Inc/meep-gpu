"""Device gate for the REAL-beta (special_kz) fused magnetic B/H pair on Triton.

Beta ``step_B`` welded into ``update_H`` — the arm
:mod:`meep_gpu.triton_kernels.beta_fused_magnetic_pair` adds, and the Triton twin
of the Metal family ``metal_kernels/beta_fused_magnetic_pair.py`` certified by
``gate_metal_below_the_cut_fused_pairs.py``.

===========================================================================
WHAT THE PRODUCT CLAIMS, AND WHAT EACH LEG MEASURES
===========================================================================

The claim is narrow and checkable: this kernel is
``kernels.fused_curl_constitutive_B``, character for character, PLUS exactly the
three lines ``special_kz.beta_pml_curl_step`` adds to ``kernels.pml_curl_step``.

1. **TRANSCRIPTION (no device).** Two EXACT statement-list equalities against the
   shipped sources, read from the FILES so they bite on the merge bar as well as
   here: the kernel minus the beta insert IS ``fused_curl_constitutive_B``, and
   the kernel minus the weld IS ``beta_pml_curl_step``. Neither is
   ``ast.unparse``d — what is being compared includes the PARENTHESISATION, and
   unparsing re-derives minimal parentheses.
2. **REDUCTION (device).** ``HAS_BETA = 0`` must reproduce the shipped
   ``fused_curl_constitutive_B`` BIT FOR BIT over a complete step, on the same
   volumes. This is the transcription claim measured on silicon instead of in
   source text, and it is the leg that would catch a mis-ordered argument in the
   plan's ``run`` that no amount of source diffing can see.
3. **IDENTITY (device).** One launch of the fused kernel leaves the engine
   bit-identical, as uint32 words, per COMPLETE driver step, to BOTH the CuPy
   array path AND the two separately certified Triton products it replaces
   (``real beta PML`` on ``step_B``, ``real beta run`` on ``update_H``), at one
   launch and at ~60, over every allocated volume, in three value classes.

===========================================================================
THE SEAM
===========================================================================

``driver.step`` runs, in order (driver.py:3282-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

On the configurations this arm admits the two fill passes are DEAD — clause 5 of
``special_kz._beta_real_grid_reasons`` refuses every fold — so the launch spans
``step_B``, ``zero_metal_B`` and ``update_H``. The magnetic source slot is REAL and
is refused by name; an ELECTRIC source is injected in the D/E half and does not
disqualify this pair.

WHAT THE CORPUS SAYS, from ``results/fusion_matrix_triton_2026-08-20_closed/`` at
the B->H cell (``real beta PML``, ``real beta run``): ONE seam-instance,
``examples:refl-angular-kz2d.py``, ``in_seam_source == false``,
``live_in_seam_passes == []``. The ``corpus_admission`` leg re-derives that from
the census rather than citing it.

THAT ROW IS ALL-PERIODIC WITH ``has_metallic == false``, so the inline
``zero_metal_B`` carry compiles to three ``False`` flags ON IT. Every wall-clear
mutation is therefore scored on a WALLED case, where the flags are real —
otherwise the rewrite would touch lines the scored grid never reaches and report
UNCAUGHT while measuring nothing.

===========================================================================
WHAT THIS GATE REFUSES TO INFER
===========================================================================

* **A mutation that does not parse is not a mutation**, and one whose lines the
  scored case never reaches is not a mutation either. Every rewrite is compiled
  with ``ast.parse`` before any device is involved, and every entry declares the
  CASE whose boundary/wall configuration enters the branch it rewrites.
* **Bytes alone cannot prove the fused path ran.** Every launch is counted through
  a proxy that owns the kernel object.
* **A no-op agreeing with a no-op is trivially identical.** Every leg carries a
  NON-VACUITY floor: the compared state must MOVE.
* **The reference must really be the array path.** ``FdtdDriver.step`` consults
  ``fastpath`` BEFORE the module-level sub-step functions this harness
  substitutes, and the ``real beta PML`` / ``real beta run`` arms are WIRED. The
  fast path is disabled explicitly on every driver and the per-route call counter
  must show the reference reaching the array path once per step for all five
  passes.
* **A false ``released``.** Only a device run writes ``release.released``.
* **BETA IS 2-D ONLY, and z is INVARIANT there.** ``Grid`` refuses beta off a 2-D
  Cartesian grid by name (grid.py:690, MEEP fields.cpp:546-547) and refuses a PEC
  on an invariant axis by name (grid.py:842-877). Every case below is 2-D with a
  PERIODIC z for that second reason.

POLICY, stamped: ``num_warps=1``, ``BLOCK=kernels.DEFAULT_BLOCK`` (256),
``enable_fp_fusion=kernels.ENABLE_FP_FUSION`` (False). BOTH float32 subnormal
policies are gated, one process each (``--subnormal-policy keep|flush``), installed
with ``strict=True`` before the first device compile.

Usage::

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_beta_fused_magnetic_pair.py --no-device
"""

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
API_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

PRODUCT_MODULE = "meep_gpu.triton_kernels.beta_fused_magnetic_pair"
KERNEL_NAME = "beta_fused_curl_constitutive_B"
PACKAGE_DIR = os.path.join(API_ROOT, "meep_gpu", "triton_kernels")
KERNEL_FILE = os.path.join(PACKAGE_DIR, "beta_fused_magnetic_pair.py")

#: refl-angular-kz2d.py's own beta.
BETA_CORPUS = 0.3321611318837033

#: The three lines the beta term adds, and the ONLY thing that may separate this
#: kernel from ``kernels.fused_curl_constitutive_B``.
BETA_INSERT: Tuple[str, ...] = (
    "if HAS_BETA:",
    "curl0 = curl0 - (beta_plus * b)",
    "curl1 = curl1 - (beta_minus * a)",
)

#: The device cases. ``(name, cell, boundaries, steps)``.
#:
#: EVERY CELL IS 2-D WITH A ZERO Z EXTENT, because ``Grid`` refuses beta anywhere
#: else by name (grid.py:690; MEEP fields.cpp:546-547 aborts on the same
#: combination). Z THEREFORE STAYS PERIODIC in every case: it is the INVARIANT
#: axis, and ``Grid`` refuses a PEC on an invariant axis by name (grid.py:842-877)
#: — a gate that tripped over that would die in the constructor instead of
#: measuring a wall.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int], ...] = (
    # THE CORPUS GEOMETRY: all-periodic, no wall. ZM_* all compile to False and
    # every ghost arm is the periodic wrap — which is exactly what
    # examples:refl-angular-kz2d.py declares.
    ("corpus_periodic", (1.4, 1.0, 0.0), "periodic", 60),
    # BOTH IN-PLANE WALLS LIVE: ZM_X and ZM_Y fire and both metallic ghost arms
    # are entered. This is the mutation case, for that reason.
    ("wall_xy", (1.4, 1.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, 60),
    # ONE WALLED AXIS ONLY: the case where a ROTATED wall table still clears the
    # same NUMBER of planes and clears the wrong one.
    ("wall_x", (1.4, 1.0, 0.0),
     {"x": "metallic", "y": "periodic", "z": "periodic"}, 60),
    # ONE LAUNCH.
    ("corpus_periodic_single_launch", (1.4, 1.0, 0.0), "periodic", 1),
)

#: The case every kernel mutation is scored on unless it names another.
MUTATION_CASE_INDEX = 1

#: Steps per armed mutation.
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

SEED = 0x8E7A

_TEMPORARY: List[str] = []
def log(message: str) -> None:
    print(message, flush=True)


def case_seed(*parts: str) -> int:
    """A per-case seed from a DIGEST, never ``hash()``.

    ``PYTHONHASHSEED`` salts ``hash()`` of a tuple of strings, so a hash-seeded
    case cannot be replayed from its own record. A blake2b of the joined parts can.
    """
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % (2 ** 31 - 1)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and material_changed would never fire. Fix the names, do not "
            f"loosen the floor.")


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    # THE POLICY STAMP IS RE-READ, NOT CARRIED. Taken once at install time it
    # records every counter at zero, because nothing had compiled yet.
    if "subnormal_policy" in payload:
        try:
            from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

            payload["subnormal_policy"] = _policy.policy_stamp()
        except Exception as exc:  # noqa: BLE001
            payload["subnormal_policy_reread_error"] = repr(exc)
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        _stamp_provenance(payload)
    except Exception as exc:  # noqa: BLE001 - a missing stamper must not lose the run
        payload["provenance_stamp_error"] = repr(exc)
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    """The exact bytes this gate binds itself to."""
    names = (
        "meep_gpu/triton_kernels/beta_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_beta_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}
# ---------------------------------------------------------------------------
# The shipped sources, read from the FILES
# ---------------------------------------------------------------------------

def _statements(text: str) -> List[str]:
    """Executable lines: comments and blanks removed, indentation normalised."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def shipped_body(path: str, name: str, *, with_decorators: bool = False) -> str:
    """One shipped kernel's source, docstring removed. Read from the FILE.

    Read that way on EVERY host — not only the laptop. Two spellings of the same
    text (``inspect.getsource`` on a CUDA box, a file read on the merge bar) would
    let a rewrite match in one place and miss in the other, which is exactly how a
    mutation silently disarms.

    The text is EXACT — never ``ast.unparse``d — because several rewrites are about
    the PARENTHESISATION and unparsing re-derives minimal parentheses. The
    docstring IS stripped: the bodies document the branches they do NOT carry, so
    a text match over the raw source would find those names in prose.

    ``with_decorators`` prepends the decorator list, which
    ``ast.get_source_segment`` on a FunctionDef EXCLUDES. Without it a mutant
    compiled from the segment is a PLAIN PYTHON FUNCTION with no ``[grid]``
    launcher, and the first mutation leg dies with "'function' object is not
    subscriptable" — measured 2026-08-20 on the sibling gate.
    """
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    if node is None:
        raise AssertionError(f"{name} is not defined in {path}")
    segment = ast.get_source_segment(text, node)
    if segment is None:  # pragma: no cover - only on a source-less module
        raise AssertionError(f"{name}'s source segment is unavailable")
    lines = segment.splitlines()
    body = node.body[0]
    if (isinstance(body, ast.Expr) and isinstance(body.value, ast.Constant)
            and isinstance(body.value.value, str)):
        start = body.lineno - node.lineno
        end = (body.end_lineno or body.lineno) - node.lineno
        lines = lines[:start] + lines[end + 1:]
    margin = node.col_offset
    lines = [lines[0].lstrip()] + [
        line[margin:] if line[:margin].strip() == "" else line.lstrip()
        for line in lines[1:]]
    if with_decorators:
        decorators = [ast.get_source_segment(text, d) for d in node.decorator_list]
        if any(segment is None for segment in decorators):  # pragma: no cover
            raise AssertionError(f"{name}'s decorator source is unavailable")
        lines = [f"@{segment}" for segment in decorators] + lines
    return "\n".join(lines)


def body_statements(path: str, name: str) -> List[str]:
    """A shipped kernel's BODY statements — signature and docstring removed."""
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
    if any(segment is None for segment in segments):  # pragma: no cover
        raise AssertionError(f"{name}'s body source is unavailable")
    return _statements("\n".join(segments))


def shipped_source() -> str:
    """The product's kernel text, decorator included — the mutation table's input."""
    return shipped_body(KERNEL_FILE, KERNEL_NAME, with_decorators=True)


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, as two exact statement-list equalities
# ---------------------------------------------------------------------------

def transcription_leg() -> Dict[str, Any]:
    """The claim this whole product rests on, made mechanical.

    Two equalities, both EXACT and both in order:

    * the kernel minus the three beta lines IS ``kernels.fused_curl_constitutive_B``;
    * the kernel minus the WELD (whatever ``fused_curl_constitutive_B`` has that
      ``kernels.pml_curl_step`` does not — DERIVED, not listed) IS
      ``special_kz.beta_pml_curl_step``.

    Together they say: nothing in this body was written here. The third check
    states that positively, over the union of the two shipped bodies.
    """
    kernels_py = os.path.join(PACKAGE_DIR, "kernels.py")
    special_py = os.path.join(PACKAGE_DIR, "special_kz.py")
    fused = body_statements(KERNEL_FILE, KERNEL_NAME)
    ordinary_fused = body_statements(kernels_py, "fused_curl_constitutive_B")
    ordinary_curl = body_statements(kernels_py, "pml_curl_step")
    beta_curl = body_statements(special_py, "beta_pml_curl_step")

    findings: List[str] = []
    without_beta = [line for line in fused if line not in BETA_INSERT]
    if without_beta != ordinary_fused:
        index = next((i for i, (a, b) in enumerate(zip(without_beta, ordinary_fused))
                      if a != b), min(len(without_beta), len(ordinary_fused)))
        findings.append(
            f"minus the beta insert this is NOT fused_curl_constitutive_B; first "
            f"difference at statement {index}: "
            f"{without_beta[index:index + 1]} vs {ordinary_fused[index:index + 1]}")

    weld = [line for line in ordinary_fused if line not in ordinary_curl]
    if not weld:
        findings.append("the weld is empty; the second equality compares nothing")
    without_weld = [line for line in fused if line not in weld]
    if without_weld != beta_curl:
        index = next((i for i, (a, b) in enumerate(zip(without_weld, beta_curl))
                      if a != b), min(len(without_weld), len(beta_curl)))
        findings.append(
            f"minus the weld this is NOT beta_pml_curl_step; first difference at "
            f"statement {index}: "
            f"{without_weld[index:index + 1]} vs {beta_curl[index:index + 1]}")

    delta = [line for line in beta_curl if line not in ordinary_curl]
    if delta != list(BETA_INSERT):
        findings.append(f"beta_pml_curl_step's own delta over pml_curl_step is "
                        f"{delta}, not {list(BETA_INSERT)}")

    invented = [line for line in fused
                if line not in set(beta_curl) | set(ordinary_fused)]
    if invented:
        findings.append(f"statements that appear in NEITHER shipped body: {invented}")

    return {"leg": "transcription", "device": False,
            "statements_in_the_product": len(fused),
            "the_weld_derived_from_the_shipped_pair": weld,
            "beta_insert_measured": delta,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the weld is never wider than either half
# ---------------------------------------------------------------------------

def _laptop_fixture(**keywords):
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    complex_storage = keywords.pop("complex_storage", False)
    pml_thickness = keywords.pop("pml_thickness", 2)
    thickness = keywords.pop("thickness", None)
    cell_size = keywords.pop("cell_size", (1.2, 1.0, 0.0))
    dimensions = keywords.pop("dimensions", 2)
    beta = keywords.pop("beta", BETA_CORPUS)
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                courant=0.35, beta=beta, **keywords)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


class _MagneticSource:
    field_type = "B"


class _ElectricSource:
    field_type = "D"


EQUIVALENCE_CASES: Tuple[Tuple[str, Dict[str, Any], Any], ...] = (
    ("corpus_family", {}, ()),
    ("electric_source", {}, (_ElectricSource(),)),
    ("magnetic_source", {}, (_MagneticSource(),)),
    ("undeclared_sources", {}, None),
    ("zero_beta", {"beta": 0.0}, ()),
    ("complex_storage", {"complex_storage": True}, ()),
    ("no_absorber", {"pml_thickness": 0}, ()),
    ("in_plane_walls", {"boundaries": {"x": "metallic", "y": "metallic",
                                       "z": "periodic"}}, ()),
    ("bloch", {"k_point": (0.2, 0.0, 0.0)}, ()),
)


def equivalence_leg() -> Dict[str, Any]:
    """``weld => spine curl AND spine H``, measured over one-clause perturbations."""
    from meep_gpu.grid import Mirror  # noqa: PLC0415
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)
    cases = list(EQUIVALENCE_CASES) + [
        # A folded axis cannot carry a LOW-face layer: cell 0 there is the mirror
        # plane and pml._resolve_mirror_faces RAISES on a per-side request naming
        # it. This case passes its own table for that reason.
        ("folded", {"symmetry": (Mirror("Y", 1),),
                    "thickness": ((2, 2), (0, 2), (0, 0))}, ()),
    ]
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    admitted = 0
    for name, keywords, sources in cases:
        fields, pml = _laptop_fixture(**dict(keywords))
        weld = product.beta_fused_magnetic_pair_coverage(fields, pml, sources)
        curl = special_kz.beta_pml_curl_coverage(fields, pml, "step_B")
        magnetic = special_kz.beta_run_constitutive_coverage(fields, pml, "H")
        never_wider = (not weld.covered) or (curl.covered and magnetic.covered)
        # The merge-bar host is NumPy, so every predicate carries the backend
        # clause. Discounting it is what lets this leg ask about the others.
        covered = not [r for r in weld.reasons if "array module" not in r]
        rows.append({"case": name, "weld_covered_modulo_backend": covered,
                     "weld_never_wider": never_wider,
                     "weld_reasons": list(weld.reasons)})
        if not never_wider:
            findings.append(f"{name}: the weld admitted where a half refused")
        admitted += int(covered)
    if admitted == 0:
        findings.append("no case was admitted modulo the backend clause: this leg "
                        "measured only refusals and could not see a widening")
    return {"leg": "equivalence", "device": False, "cases": rows,
            "admitted_modulo_backend": admitted, "findings": findings,
            "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — what the corpus says, re-derived from the census
# ---------------------------------------------------------------------------

CENSUS = os.path.join(HERE, "results", "predicate_coverage_2026-08-16_wired_convention")
MATRIX = os.path.join(HERE, "results", "fusion_matrix_triton_2026-08-20_closed",
                      "fusion_matrix.json")
CELL = ("B->H", "real beta PML", "real beta run")


def corpus_admission_leg() -> Dict[str, Any]:
    """The rows this arm can serve, counted rather than cited.

    Read from the census's own configuration block: a row is in this cell when it
    declares a NONZERO beta, an active absorber, real storage, no fold, no
    cylindrical axis, k = 0, no BFAST, no nonlinearity and no off-diagonal
    epsilon; it is REACHABLE when it declares no MAGNETIC source. The matrix's own
    count is loaded beside it and the two must agree.
    """
    findings: List[str] = []
    record: List[dict] = []
    for name in ("examples.jsonl", "tests.jsonl"):
        path = os.path.join(CENSUS, name)
        if not os.path.exists(path):
            return {"leg": "corpus_admission", "device": False,
                    "findings": [f"the census is absent: {path}"], "passed": False}
        record.extend(json.loads(line) for line in
                      open(path, encoding="utf-8").read().splitlines() if line.strip())
    matched = {(r.get("leg"), r.get("row")): r for r in (
        json.loads(line) for line in open(
            os.path.join(CENSUS, "tests_param_matched.jsonl"),
            encoding="utf-8").read().splitlines() if line.strip())}
    record = [matched.get((r.get("leg"), r.get("row")), r) for r in record]
    record = [r for r in record if r.get("measured")]

    in_cell: List[str] = []
    reachable: List[str] = []
    for row in record:
        configuration = row.get("configuration") or {}
        if float(configuration.get("beta") or 0.0) == 0.0:
            continue
        if not configuration.get("pml_active"):
            continue
        if configuration.get("force_complex_fields"):
            continue
        if configuration.get("has_symmetry") or any(configuration.get("mirrored") or ()):
            continue
        if configuration.get("cylindrical") or configuration.get("bfast_active"):
            continue
        if configuration.get("has_bloch") or configuration.get("has_nonlinearity"):
            continue
        if configuration.get("has_offdiagonal_epsilon"):
            continue
        name = f"{row['leg']}:{row['row']}"
        in_cell.append(name)
        if "B" not in tuple(configuration.get("source_field_types") or ()):
            reachable.append(name)

    matrix_rows: List[str] = []
    if os.path.exists(MATRIX):
        matrix = json.load(open(MATRIX, encoding="utf-8"))
        matrix_rows = sorted(
            entry["row"] for entry in matrix["seam_instances"]
            if (entry["seam"], entry["curl_arm"], entry["constitutive_arm"]) == CELL
            and not entry["in_seam_source"])
        if matrix_rows != sorted(reachable):
            findings.append(
                f"this leg derived {sorted(reachable)} from the census while the "
                f"fusion matrix scores {matrix_rows} at the same cell")
    else:
        findings.append(f"the fusion matrix is absent: {MATRIX}")
    if not reachable:
        findings.append("no reachable row: this arm would be worth nothing")
    return {"leg": "corpus_admission", "device": False, "rows_scanned": len(record),
            "cell": list(CELL), "rows_in_the_cell": sorted(in_cell),
            "rows_clearing_the_source_seam": sorted(reachable),
            "matrix_rows_at_the_same_cell": matrix_rows,
            "findings": findings, "passed": not findings}
# ---------------------------------------------------------------------------
# The device harness
# ---------------------------------------------------------------------------

def seed_values(cp, driver, seed: int, value_class: str) -> None:
    """Seed every settable volume in ONE value class, identically on every route."""
    rng = np.random.default_rng(seed)
    shape = tuple(driver.shape)

    def draw() -> np.ndarray:
        if value_class == "uniform":
            values = rng.uniform(-0.25, 0.25, size=shape)
        elif value_class == "signed_zero_lattice":
            # +0.0 and -0.0 on a random lattice. The two words differ (0x00000000
            # vs 0x80000000), so a byte gate SEES the difference — this is what
            # exercises every `tl.where(flag, 0.0, v)` and every subtraction's
            # sign convention, INCLUDING the beta term's `curl - (c * g)`.
            picks = rng.integers(0, 2, size=shape)
            values = np.where(picks == 0, 0.0, -0.0)
        elif value_class == "subnormal_band":
            # Below 2**-126: the words the two float32 subnormal policies disagree
            # about. The policy stamp says which one this process ran.
            values = rng.standard_normal(shape) * 1e-38
        else:  # pragma: no cover - VALUE_CLASSES is the whole domain
            raise ValueError(f"unknown value class {value_class!r}")
        return np.ascontiguousarray(values.astype(np.float32))

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(draw()))
    # E and H are DERIVED and `set_field` refuses them by name (driver.py:3909).
    # Written in place here, which is legal for a harness and is what makes the
    # single-launch leg non-vacuous: without a nonzero E the B curl AND the beta
    # term are both exactly zero on step 1.
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())


def build_driver(cp, cell, boundaries, seed: int, value_class: str,
                 electric: bool, beta: float = BETA_CORPUS):
    """One 2-D REAL-beta PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=2,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        beta=float(beta), prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (2.25 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    # PER AXIS, NOT SCALAR. On a 2-D cell z is translationally invariant and the
    # driver refuses a layer there BY NAME (driver.py:2206): "an invariant axis has
    # no face to absorb at". MEEP drops such a layer silently, so naming X and Y is
    # the same run, said out loud.
    driver.setup_pml({"x": 2, "y": 2})
    if electric:
        # ADMITTED: the driver injects an electric source in the D/E seam, not
        # this one. The corpus row sources electrically.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    # THE FAST PATH IS DISABLED, EXPLICITLY. `FdtdDriver.step` consults `fastpath`
    # BEFORE the module-level sub-step functions this harness substitutes
    # (driver.py:3281-3282), and the `real beta PML` / `real beta run` arms are
    # WIRED — so on this very family a driver left alone would step `step_B` and
    # `update_H` on Triton and the "array path" reference would be no such thing.
    # The per-route call counter re-checks that this held.
    driver.invalidate_fast_path()
    driver._fast_path = None          # noqa: SLF001 - the harness owns the route
    driver._fast_path_stale = False   # noqa: SLF001
    return driver
def inventory(driver) -> Dict[str, Any]:
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        if name.startswith("__"):
            continue
        if (getattr(value, "shape", None) == shape
                and getattr(value, "dtype", None) is not None):
            found[name] = value
    _assert_material_names_are_real(found)
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not complete")
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
    return {"array": "<inventory>", "reason": "asymmetric", "names": only} if only else None


def moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> List[str]:
    return [name for name in sorted(set(before) & set(after))
            if not np.array_equal(before[name], after[name])]


class CountingKernel:
    """Owns the JIT kernel and counts every launch through the plan's ``run``."""

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
    """The sentinel left where a driver pass was absorbed by the fused launch."""

    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self) -> None:
        return None


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the five seam passes for the fields objects named in ``routes``.

    Counting is PER ROUTE. A single global counter would mix the reference driver's
    legitimate array-path calls with a fallback in the fused route, which is
    precisely the event this instrument exists to see.
    """
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
    """The TWO separately certified Triton products this launch replaces.

    They are the arms ``launch.plan_step`` really selects on a beta row today —
    ``real beta PML`` on ``step_B`` and ``real beta run`` on ``update_H`` — so
    this route is not a second model of the seam, it is the shipped one.

    The three in-seam passes stay on the ARRAY PATH here: no Triton product owns
    the wall clear, and both symmetry fills are no-ops without a mirror plane. All
    three are COUNTED, so a difference in which of them ran is a counter event
    rather than an invisible correction.
    """
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    curl = special_kz.plan_beta_pml_curl(driver.fields, driver.pml, "step_B")
    constitutive = special_kz.plan_beta_run_constitutive(driver.fields, driver.pml, "H")
    missing = [name for name, plan in
               (("real beta PML curl", curl),
                ("real beta run constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the certified "
            f"products it replaces")
    return Route({"step_B": curl, "update_H": constitutive})


def fused_route(plan):
    return Route({
        "step_B": plan,
        "fill_symmetry_bc_B": _Absorbed("fill_symmetry_bc_B", plan),
        "zero_metal_B": _Absorbed("zero_metal_B", plan),
        "fill_folded_far_ghosts_B": _Absorbed("fill_folded_far_ghosts_B", plan),
        "update_H": _Absorbed("update_H", plan),
    })
def _swapped_coefficients(driver, plan, group: str):
    """The plan's coefficient bindings with ONE group moved to the WRONG lattice.

    ``SUB_STEPS['step_B']`` carries ``suffix == '_h'`` so the curl half reads the
    HALF-INTEGER split-field pair, while ``CONSTITUTIVE_SIDES['H']['half_integer']
    is False`` so the constitutive half reads the INTEGER one. The kernel takes
    twelve pointers and never asks which lattice they came from, so a swap is a
    half-cell error in the absorber profile: converged, smooth and wrong.
    """
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

    for slot in ("_curl_coefficients", "_h_coefficients"):
        if not hasattr(plan, slot):
            raise AssertionError(
                f"the plan has no {slot}; this host mutation would bind nothing "
                f"and the leg would score a defect it never armed")
    pml = driver.pml
    if group == "curl":
        arrays = [getattr(pml, f"{stem}_{axis}")            # INTEGER, wrongly
                  for axis in "xyz" for stem in ("kms", "sinv")]
        plan._curl_coefficients = tuple(  # noqa: SLF001 - the armed seam
            CupyPointer(_flat(a)) for a in arrays)
    elif group == "constitutive":
        arrays = [getattr(pml, f"{stem}_{axis}_h")          # HALF-INTEGER, wrongly
                  for axis in "xyz" for stem in ("kps", "kms")]
        # ``_h_coefficients`` — BetaFusedMagneticPairPlan's own slot name, not
        # FusedPairPlan's ``_constitutive_coefficients``. Measured 2026-08-21: the
        # helper was carried over from the sibling gate and raised AttributeError
        # here, which aborted the host-mutation leg after h1 had already scored.
        plan._h_coefficients = tuple(  # noqa: SLF001
            CupyPointer(_flat(a)) for a in arrays)
    else:  # pragma: no cover
        raise ValueError(group)
    return plan


def host_mutation_table() -> Tuple[Tuple[str, str, str, Callable[[Any, Any], Any]], ...]:
    """The HOST mutations: plan-level choices, not kernel text."""

    def h1_integer_coefficients_on_the_curl(driver, plan):
        return _swapped_coefficients(driver, plan, "curl")

    def h2_half_integer_coefficients_on_the_constitutive(driver, plan):
        return _swapped_coefficients(driver, plan, "constitutive")

    def h3_zero_metal_all_false(driver, plan):
        """The wall clear switched off at the HOST while the walls are real.

        Scored on ``wall_xy``, where both in-plane flags are really True — on the
        all-periodic corpus case this would change nothing and measure nothing."""
        plan.zero_metal = (0, 0, 0)
        return plan

    def h4_wall_table_is_the_d_side(driver, plan):
        """``zero_metal`` rotated: still the same NUMBER of planes, the WRONG ones.

        ``stepping._zero_metal`` clears component ``m`` on walled axis ``a`` exactly
        when ``IYEE_SHIFTS[m][a] == 0``; for B that table is the DIAGONAL
        (x->Bx, y->By, z->Bz). Scored on ``wall_x``, where exactly ONE axis is
        walled so the rotation moves the clear onto a component the array path
        leaves alone — a defect no counting check can see."""
        flags = tuple(plan.zero_metal)
        plan.zero_metal = (flags[2], flags[0], flags[1])
        return plan

    def h5_beta_coefficients_swapped(driver, plan):
        """``beta_plus`` and ``beta_minus`` exchanged.

        The two are ``+2*pi*beta*dt`` and ``-2*pi*beta*dt``
        (``special_kz.beta_curl_coefficients``), and they are applied to DIFFERENT
        components — target 0 takes the second source at +1, target 1 the first at
        -1 (step_db.cpp:148-176). Exchanging them is a sign error on both
        components at once, which a magnitude check would miss."""
        plan.beta_plus, plan.beta_minus = plan.beta_minus, plan.beta_plus
        return plan

    def h6_beta_switched_off_at_the_host(driver, plan):
        """``HAS_BETA = 0`` on a run whose beta is real.

        The counterpart of the REDUCTION leg, armed: there the same flag must
        reproduce the shipped ordinary kernel; here it must be CAUGHT, because the
        array path really does add the term."""
        plan.has_beta = 0
        return plan

    return (
        ("h1_integer_coefficients_on_the_curl", "the crossed sub-lattices",
         "caught", h1_integer_coefficients_on_the_curl),
        ("h2_half_integer_coefficients_on_the_constitutive",
         "the crossed sub-lattices", "caught",
         h2_half_integer_coefficients_on_the_constitutive),
        ("h3_zero_metal_all_false", "the wall clear, switched off at the host",
         "caught", h3_zero_metal_all_false),
        ("h4_wall_table_is_the_d_side", "the DIAGONAL vs the OFF-DIAGONAL table",
         "caught", h4_wall_table_is_the_d_side),
        ("h5_beta_coefficients_swapped", "the beta sign convention", "caught",
         h5_beta_coefficients_swapped),
        ("h6_beta_switched_off_at_the_host", "the HAS_BETA constexpr", "caught",
         h6_beta_switched_off_at_the_host),
    )


#: Host mutation id -> the CASES index it is scored on. ``h3`` needs a live wall
#: and ``h4`` needs EXACTLY ONE walled axis; the others are wall-independent and
#: run on the default mutation case.
HOST_MUTATION_CASE: Dict[str, int] = {
    "h4_wall_table_is_the_d_side": 2,   # wall_x: exactly ONE walled axis
}


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, steps: int, product,
            value_class: str = "uniform", mutant: Any = None,
            host_mutation: Any = None, install_fused: bool = True,
            freeze_magnetic: bool = False, electric: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    seed = case_seed(name, str(cell), str(boundaries), value_class)
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric)
            for _ in range(3)]
    reference, separate, fused = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.beta_fused_curl_constitutive_B_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "value_class": value_class, "seed": seed,
        "electric_source": bool(electric),
        "fused_substituted": bool(install_fused),
        "magnetic_frozen": bool(freeze_magnetic),
        "host_mutation": None, "first_divergence": None,
        "control_divergence": None,
    }
    try:
        plan = product.plan_beta_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), kernel=kernel)
        if plan is None:
            verdict = product.beta_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if host_mutation is not None:
            row["host_mutation"] = host_mutation[0]
            plan = host_mutation[3](fused, plan)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["zero_metal_flags"] = list(plan.zero_metal)
        row["beta_words"] = [plan.beta_plus, plan.beta_minus, plan.has_beta]
        route = fused_route(plan) if install_fused else Route({})
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
            })
            divergence = versus_array or versus_separate
            if step == 1 or step % 10 == 0 or divergence is not None:
                log(f"  {name} [{value_class}] step {step}/{steps} "
                    f"identical={divergence is None} control={control is None} "
                    f"moved={len(step_moved)} launches={kernel.calls} "
                    f"({time.time() - started:.1f} s)")
            row["first_divergence"] = divergence
            row["control_divergence"] = control
            if divergence is not None:
                row["diverged_at_step"] = step
                break

        final = snapshot(cp, fused)
        ever_moved = moved(opening, final)
        material = [key for key in final if key in MATERIAL]
        row["arrays_total"] = len(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        return row
    finally:
        undo()
        for target in made:
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True,
               require_array_reference: bool = True) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg "
            f"could not have measured the fused launch: {row['control_divergence']}")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"expected {require_launches}: the fused path is not what executed")
    if require_moved and (row.get("arrays_never_moved") or []):
        failures.append(f"VACUOUS: these arrays never moved: {row['arrays_never_moved']}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    counts = row.get("launches") or {}
    fell_back = {key: value for key, value in counts.items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: {fell_back}")
    if require_array_reference:
        # THE REFERENCE MUST REALLY BE THE ARRAY PATH. `fastpath` is consulted
        # before these functions and the beta arms are WIRED, so a missing count
        # here means the reference stepped on Triton and the comparison compared
        # two Triton routes.
        steps = int(row.get("steps_budget") or 0)
        for name in SEAM_PASSES:
            seen = counts.get(f"array/array_path:{name}", 0)
            if seen != steps:
                failures.append(
                    f"the reference route reached the array path for {name} "
                    f"{seen} times, expected {steps}: the fast path was not off")
    return (not failures), failures
# ---------------------------------------------------------------------------
# THE REDUCTION LEG — HAS_BETA = 0 must BE the shipped ordinary kernel
# ---------------------------------------------------------------------------

def reduction_leg(cp, product, cell, boundaries, steps: int,
                  value_class: str = "uniform") -> Dict[str, Any]:
    """``HAS_BETA = 0`` reproduces ``kernels.fused_curl_constitutive_B`` bit for bit.

    The transcription claim, measured on silicon instead of in source text. Two
    drivers, seeded identically, on a REAL beta run:

    * route A launches THIS product's kernel with ``has_beta = 0``;
    * route B launches the SHIPPED ``kernels.fused_curl_constitutive_B`` over the
      same volumes, through ``launch.plan_fused_pair_from_arrays`` — the bare-array
      route, because ``coverage.fused_pair_coverage`` refuses a beta run by name
      (clause 12) and would return ``None``.

    NEITHER IS COMPARED AGAINST THE ARRAY PATH HERE, deliberately: the array path
    really does add the beta term, so both routes must differ from it. What this
    leg measures is that the two KERNELS agree, which is the reduction.

    THE NON-VACUITY FLOOR IS THE OTHER HALF. Two kernels that both did nothing
    would also agree, so the state must MOVE and route A must additionally be shown
    to DIFFER from the array path — otherwise ``has_beta = 0`` would be
    indistinguishable from the shipped beta kernel and this leg would be measuring
    a flag nothing reads.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import (  # noqa: PLC0415
        plan_fused_pair_from_arrays,
    )

    seed = case_seed("reduction", str(cell), str(boundaries), value_class)
    # WITH THE ELECTRIC SOURCE, like every other leg. MEASURED 2026-08-21: without
    # one, the `signed_zero_lattice` class leaves every volume at +-0.0 forever —
    # the curl is a difference of signed zeros and `beta_plus * (+-0.0)` is another
    # signed zero — so HAS_BETA=0 matched the ARRAY PATH as well as the shipped
    # kernel, and five `fu_*` volumes never moved. Both failures were the harness
    # measuring nothing, not the product: the discrimination precondition below is
    # what turns that from a silent pass into a stated one either way.
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric=True)
            for _ in range(3)]
    reference, shipped, reduced = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    row: Dict[str, Any] = {"leg": "reduction:has_beta_zero", "device": True,
                           "steps_budget": steps, "value_class": value_class,
                           "seed": seed, "shape": list(reference.shape),
                           "boundaries": boundaries}
    try:
        from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
        from meep_gpu.stepping import _boundary_kinds as resolve  # noqa: PLC0415

        reduced_plan = product.plan_beta_fused_magnetic_pair(
            reduced.fields, reduced.pml, ())
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
        shipped_plan = plan_fused_pair_from_arrays(
            "B", arrays, curl_flat, constitutive_flat,
            [1 if kind == "metallic" else 0 for kind in resolve(grid, pml)],
            zero_metal_axes(grid), grid.dt / grid.dx, num_warps=1)

        roles = {id(reference.fields): "array", id(shipped.fields): "shipped",
                 id(reduced.fields): "reduced"}
        routes = [(shipped.fields, fused_route(shipped_plan)),
                  (reduced.fields, fused_route(reduced_plan))]
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
                "has_beta_zero_vs_shipped_fused_pair": versus_shipped,
                "has_beta_zero_vs_array_path": versus_array,
            })
            log(f"  reduction step {step}/{steps} "
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
            failures.append(
                f"HAS_BETA=0 is NOT the shipped fused pair: {row['first_divergence']}")
        if row["arrays_never_moved"]:
            failures.append(f"VACUOUS: {row['arrays_never_moved']} never moved")
        # THE DISCRIMINATION PRECONDITION, recorded rather than assumed. If
        # HAS_BETA=0 also matches the ARRAY PATH then this value class cannot tell
        # the two kernels apart at all and its agreement is a tautology. That is a
        # PROPERTY OF THE VALUE CLASS, not a defect, so it is reported as
        # `discriminating: false` with its reason and the gate-level floor demands
        # that at least one class discriminate — a leg that quietly passed on a
        # class measuring nothing is the vacuity trap wearing another coat.
        row["discriminating"] = row.get("divergence_from_the_array_path") is not None
        if not row["discriminating"]:
            row["not_discriminating_because"] = (
                "HAS_BETA=0 matched the ARRAY PATH too, so this value class cannot "
                "separate the beta kernel from the ordinary one")
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
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest."""
    header = ("import triton\nimport triton.language as tl\n"
              "PERIODIC = tl.constexpr(0)\n"
              "METALLIC = tl.constexpr(1)\n\n")
    text = header + source.replace(KERNEL_NAME, kernel_name)
    # A MUTATION THAT DOES NOT PARSE IS NOT A MUTATION. Checked before any device
    # is involved, because a SyntaxError inside the loader aborts the whole
    # mutation leg on the first device run.
    ast.parse(text)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_beta_pair.py", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_beta_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace a contiguous run of statements, preserving the block's indent.

    MATCHED ON COMMENT-STRIPPED TEXT, so a rewrite stays armed across a trailing
    comment or a reflow; a mutation that silently stops matching reports a real
    defect as uncaught. Every replacement line takes the indent of the block's
    FIRST line, so an entry that drops an ``if`` must drop its body with it.
    """
    lines = source.splitlines()
    stripped = [line.split("#", 1)[0].strip() for line in lines]
    needles = [text.split("#", 1)[0].strip() for text in before]
    for index in range(len(lines) - len(needles) + 1):
        if stripped[index:index + len(needles)] != needles:
            continue
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        replaced = lines[:index] + [indent + text for text in after] \
            + lines[index + len(needles):]
        return "\n".join(replaced) + "\n", 1
    return source, 0


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite)."""

    def m1_beta_term_dropped(source: str) -> Tuple[str, int]:
        """The three lines that make this product exist, removed.

        The ``if HAS_BETA:`` goes WITH its body, because ``_rewrite_block``
        re-indents to the block's first line and a compound statement in position 0
        would produce an empty suite — an unarmed mutation, not a caught defect.
        Dropping the guard is NOT the change being measured: every case this gate
        scores has ``has_beta = 1``, so the guarded lines executed on the pristine
        kernel too. The assignments left behind are pure identity, so a signed zero
        survives them."""
        return _rewrite_block(
            source,
            ["if HAS_BETA:",
             "curl0 = curl0 - (beta_plus * b)",
             "curl1 = curl1 - (beta_minus * a)"],
            ["curl0 = curl0", "curl1 = curl1"])

    def m2_beta_partners_exchanged(source: str) -> Tuple[str, int]:
        """The two CENTER partners swapped: target 0 takes ``a``, target 1 ``b``.

        ``step_db.cpp:148-176`` pairs target 0 with the SECOND source's center and
        target 1 with the FIRST. Exchanging them keeps both magnitudes and both
        signs and is invisible to anything but a byte comparison."""
        return _rewrite_block(
            source,
            ["curl0 = curl0 - (beta_plus * b)",
             "curl1 = curl1 - (beta_minus * a)"],
            ["curl0 = curl0 - (beta_plus * a)",
             "curl1 = curl1 - (beta_minus * b)"])

    def m3_beta_scaled_by_dtdx(source: str) -> Tuple[str, int]:
        """``dtdx`` applied to the beta term.

        The term is an ANALYTIC derivative (stepping.py:760-762) and carries no
        grid spacing. Multiplying by ``dtdx`` is the single most natural wrong
        transcription of an insert that sits one line after a ``dtdx`` curl."""
        return _rewrite_block(
            source,
            ["curl0 = curl0 - (beta_plus * b)",
             "curl1 = curl1 - (beta_minus * a)"],
            ["curl0 = curl0 - (dtdx * (beta_plus * b))",
             "curl1 = curl1 - (dtdx * (beta_minus * a))"])

    def m4_beta_added_after_the_ownership_mask(source: str) -> Tuple[str, int]:
        """The insert moved BELOW the ownership mask.

        The array path adds it after the curl and BEFORE the mask (S:356-363 after
        :342, before :369), so a masked-away cell must lose the beta term with the
        curl. Moving it below leaves a nonzero curl on a plane the array path
        zeroed. Guarded on a METALLIC axis being present — scored on ``wall_xy``,
        which has two."""
        source, hits = _rewrite_block(
            source,
            ["if HAS_BETA:",
             "curl0 = curl0 - (beta_plus * b)",
             "curl1 = curl1 - (beta_minus * a)"],
            ["curl0 = curl0", "curl1 = curl1"])
        if not hits:
            return source, 0
        return _rewrite_block(
            source,
            ["km_x = tl.load(kmx + i, mask=live, other=0.0)"],
            ["if HAS_BETA:",
             "    curl0 = curl0 - (beta_plus * b)",
             "    curl1 = curl1 - (beta_minus * a)",
             "km_x = tl.load(kmx + i, mask=live, other=0.0)"])

    def m5_wall_table_is_the_d_side(source: str) -> Tuple[str, int]:
        """The DIAGONAL wall table rotated onto the OFF-DIAGONAL pairing.

        Scored on ``wall_xy``, where ZM_X and ZM_Y are True and both ``at_*``
        planes are real."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)",
             "if ZM_Y:", "v1 = tl.where(at_y, 0.0, v1)",
             "if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"],
            ["if ZM_X:", "    v1 = tl.where(at_x, 0.0, v1)",
             "if ZM_Y:", "    v2 = tl.where(at_y, 0.0, v2)",
             "if ZM_Z:", "    v0 = tl.where(at_z, 0.0, v0)"])

    def m6_zero_metal_only_on_the_store(source: str) -> Tuple[str, int]:
        """Cleared into ``B``, but the constitutive half reads the UNCLEARED value.

        The whole point of applying the clear to the REGISTER: the array path
        leaves ONE value in ``B`` and both consumers must see it. Same guard note
        as ``m5``; scored on ``wall_xy``."""
        source, hits = _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)",
             "if ZM_Y:", "v1 = tl.where(at_y, 0.0, v1)",
             "if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"],
            ["z0 = tl.where(at_x, 0.0, v0)",
             "z1 = tl.where(at_y, 0.0, v1)",
             "z2 = tl.where(at_z, 0.0, v2)"])
        if not hits:
            return source, 0
        for register in ("0", "1", "2"):
            source = source.replace(
                f"tl.store(f{register} + idx, v{register}, mask=live)",
                f"tl.store(f{register} + idx, z{register}, mask=live)")
        return source, hits

    def m7_history_read_after_write(source: str) -> Tuple[str, int]:
        """``f_w`` read AFTER it is written: wrong wherever ``kms != 0``. Unguarded."""
        return _rewrite_block(
            source,
            ["prev0 = tl.load(w0 + idx, mask=live, other=0.0)",
             "src0 = v0",
             "tl.store(w0 + idx, src0, mask=live)"],
            ["src0 = v0",
             "tl.store(w0 + idx, src0, mask=live)",
             "prev0 = tl.load(w0 + idx, mask=live, other=0.0)"])

    def m8_curl_parentheses_flattened(source: str) -> Tuple[str, int]:
        """``stepping._curl_from_operands``' grouping flattened: same algebra,
        different float32 rounding. Unguarded."""
        needle = "curl0 = dtdx * ((c_y - c) + (b - b_z))"
        return source.replace(needle, "curl0 = dtdx * (c_y - c + b - b_z)"), \
            source.count(needle)

    def m9_constitutive_association(source: str) -> Tuple[str, int]:
        """Right-associated accumulation: same algebra, different float32 rounding."""
        return _rewrite_block(
            source,
            ["a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"],
            ["a0 = a0 + (kp_0 * src0 - km_0 * prev0)"])

    def m10_ownership_mask_dropped(source: str) -> Tuple[str, int]:
        """``stepping._mask_non_owned_cells`` undone on the forward arm.

        Guarded on ``BCX/BCY == METALLIC`` and on the ``else`` (forward) arm, both
        of which ``wall_xy`` enters. ``BCZ`` is periodic on every case here, so the
        ``curl2`` line is dead — which is why it is NOT part of the needle: a
        rewrite that includes a line its case never reaches is a rewrite whose
        result cannot be attributed."""
        return _rewrite_block(
            source,
            ["if BCX == METALLIC:", "curl0 = tl.where(at_x, 0.0, curl0)",
             "if BCY == METALLIC:", "curl1 = tl.where(at_y, 0.0, curl1)"],
            ["curl0 = curl0", "curl1 = curl1"])

    def m11_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
        ("m1_beta_term_dropped", "the term this product exists for", "caught",
         m1_beta_term_dropped),
        ("m2_beta_partners_exchanged", "the CENTER partner pairing", "caught",
         m2_beta_partners_exchanged),
        ("m3_beta_scaled_by_dtdx", "the analytic derivative's missing dtdx",
         "caught", m3_beta_scaled_by_dtdx),
        ("m4_beta_added_after_the_ownership_mask", "the insert's position",
         "caught", m4_beta_added_after_the_ownership_mask),
        ("m5_wall_table_is_the_d_side", "the DIAGONAL vs the OFF-DIAGONAL table",
         "caught", m5_wall_table_is_the_d_side),
        ("m6_zero_metal_only_on_the_store", "the register-vs-memory carry",
         "caught", m6_zero_metal_only_on_the_store),
        ("m7_history_read_after_write", "the f_w ordering", "caught",
         m7_history_read_after_write),
        ("m8_curl_parentheses_flattened", "the curl grouping", "caught",
         m8_curl_parentheses_flattened),
        ("m9_constitutive_association", "float32 association", "caught",
         m9_constitutive_association),
        ("m10_ownership_mask_dropped", "the ownership mask", "caught",
         m10_ownership_mask_dropped),
        ("m11_commuted_multiply", "commuted multiply", "null", m11_commuted_multiply),
    )


#: Mutation id -> the CASES index whose grid carries the branch it rewrites.
#: Anything not named here runs on :data:`MUTATION_CASE_INDEX` (``wall_xy``).
MUTATION_CASE: Dict[str, int] = {}


def mutation_arming_leg() -> Dict[str, Any]:
    """Every rewrite ARMS and PARSES, checked on the laptop before any device run."""
    source = shipped_source()
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, why, expectation, rewrite in mutation_table():
        mutated, hits = rewrite(source)
        parses, error = True, None
        try:
            ast.parse("import triton\nimport triton.language as tl\n"
                      "PERIODIC = tl.constexpr(0)\nMETALLIC = tl.constexpr(1)\n\n"
                      + mutated.replace(KERNEL_NAME, "mutant"))
        except SyntaxError as exc:  # noqa: PERF203
            parses, error = False, repr(exc)
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "rewrite_hits": hits, "parses": parses, "error": error,
                     "changed": mutated != source,
                     "case": CASES[MUTATION_CASE.get(name, MUTATION_CASE_INDEX)][0]})
        if not hits or mutated == source:
            findings.append(f"{name}: the rewrite matched nothing")
        if not parses:
            findings.append(f"{name}: the mutant does not parse ({error})")
    return {"leg": "mutation_arming", "device": False, "mutations": rows,
            "findings": findings, "passed": not findings}


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source()
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "kind": "source", "device": True}
        if hits == 0 or mutated == source:
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        mutant = compile_mutant(mutated, f"mutant_{index}_beta_fused_B")
        case_index = MUTATION_CASE.get(name, MUTATION_CASE_INDEX)
        case = CASES[case_index]
        row["case"], row["case_index"] = case[0], case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], MUTATION_STEPS,
                      product, mutant=mutant)
        row["caught"] = leg.get("first_divergence") is not None
        row["first_divergence"] = leg.get("first_divergence")
        row["launches"] = leg.get("fused_kernel_launches")
        row["arrays_never_moved"] = leg.get("arrays_never_moved")
        row["ptx_moved"] = sorted(kernel_ptx(mutant)) != sorted(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} on {row['case']} "
            f"launches={row['launches']} expectation={expectation}")
    return rows


def run_host_mutations(cp, product) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in host_mutation_table():
        name, why, expectation, _apply = entry
        case_index = HOST_MUTATION_CASE.get(name, MUTATION_CASE_INDEX)
        case = CASES[case_index]
        leg = run_leg(cp, f"host_mutation:{name}", case[1], case[2],
                      MUTATION_STEPS, product, host_mutation=entry)
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "kind": "host", "device": True, "case": case[0],
                     "case_index": case_index,
                     "zero_metal_flags": leg.get("zero_metal_flags"),
                     "beta_words": leg.get("beta_words"),
                     "caught": leg.get("first_divergence") is not None,
                     "first_divergence": leg.get("first_divergence"),
                     "launches": leg.get("fused_kernel_launches")})
        log(f"  host mutation {name}: caught={rows[-1]['caught']} on {case[0]} "
            f"expectation={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse, asked of a real CuPy driver."""

    class _Magnetic:
        field_type = "B"

    cell, walls = CASES[0][1], CASES[0][2]
    rows: List[Dict[str, Any]] = []
    for name, beta, sources, needle in (
        ("magnetic_source", BETA_CORPUS, (_Magnetic(),), "is magnetic"),
        ("undeclared_sources", BETA_CORPUS, None, "was not declared"),
        ("zero_beta", 0.0, (), "grid.beta is zero"),
    ):
        driver = build_driver(cp, cell, walls, SEED, "uniform", electric=False,
                              beta=beta)
        try:
            verdict = product.beta_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources)
            plan = product.plan_beta_fused_magnetic_pair(
                driver.fields, driver.pml, sources)
            rows.append({
                "case": name, "device": True, "covered": bool(verdict.covered),
                "reasons": list(verdict.reasons), "plan_is_none": plan is None,
                "passed": (not verdict.covered) and plan is None
                          and any(needle in reason for reason in verdict.reasons),
            })
        finally:
            driver.close()
            cp.get_default_memory_pool().free_all_blocks()
        log(f"  refusal {name}: covered={rows[-1]['covered']} "
            f"passed={rows[-1]['passed']}")
    return rows


def environment(cp: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import triton  # noqa: PLC0415

        payload["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        payload["triton"] = f"absent: {exc!r}"
    if cp is not None:
        payload["cupy"] = cp.__version__
        payload["device"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_beta_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--subnormal-policy", default="keep",
                        choices=("keep", "flush"))
    parser.add_argument(
        "--plant-defect", default=None,
        help="install a named kernel mutation as THE PRODUCT'S kernel for every "
             "device leg. The release verdict must FLIP to False; a gate whose "
             "verdict does not move against a planted defect is not measuring.")
    args = parser.parse_args(argv)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_beta_fused_magnetic_pair",
        "product": PRODUCT_MODULE,
        "kernel": KERNEL_NAME,
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": triton_device_identity.record(environment()),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "kernels.DEFAULT_BLOCK"},
        "planted_defect": args.plant_defect,
        "no_device_legs": [], "device_legs": [], "mutations": [], "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in (transcription_leg, equivalence_leg, corpus_admission_leg,
                mutation_arming_leg):
        started = time.time()
        row = leg()
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} "
            f"findings={row.get('findings')} ({row['seconds']} s)")
        save(payload, args.out)

    if args.no_device:
        payload["device_status"] = (
            "UNRUN — invoked with --no-device; no CUDA leg, mutation or refusal "
            "in this artifact")
        payload["passed"] = all(row["passed"] for row in payload["no_device_legs"])
        # ``release`` IS THE KEY ``gate_provenance.read_verdict`` CONSULTS FIRST.
        payload["release"] = {
            "released": False,
            "reasons": ["the no-device legs passed, but no device leg, mutation or "
                        "refusal has run: this artifact releases nothing"],
        }
        save(payload, args.out)
        log(f"\nno-device verdict: {payload['passed']}  ->  {args.out}")
        return 0 if payload["passed"] else 1

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    if args.subnormal_policy == "flush":
        # MEASURED REFUSAL, not a precaution: `install_subnormal_policy('flush')`
        # raises SubnormalPolicyUnattainable in a process that has not imported
        # MEEP (subnormal_policy.py:1920) because `mp.set_zero_subnormals` is the
        # only exposure of this process's FTZ/DAZ bits the package may use, and it
        # will not import MEEP itself (no-MEEP-import boundary; MPI side effect).
        # A parity probe may do what the package may not. NOT UNCONDITIONAL:
        # importing MEEP moves the host FPU, so a `keep` leg that imported it would
        # differ from one that did not for reasons unrelated to the kernel.
        record: Dict[str, Any] = {"requested": True}
        try:
            import meep  # noqa: PLC0415 - see above; parity probe, not package

            record.update(imported=True,
                          meep_version=getattr(meep, "__version__", "<unknown>"),
                          has_set_zero_subnormals=hasattr(meep,
                                                          "set_zero_subnormals"))
        except Exception as exc:  # noqa: BLE001
            record.update(imported=False, why=f"{type(exc).__name__}: {exc}"[:400])
        payload["meep_import_for_host_policy"] = record
        log(f"[policy] MEEP import for the host half of 'flush': {record}")
        save(payload, args.out)

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
    save(payload, args.out)

    planted = None
    if args.plant_defect:
        table = {name: rewrite for name, _why, _expect, rewrite in mutation_table()}
        if args.plant_defect not in table:
            raise SystemExit(f"unknown mutation {args.plant_defect!r}; "
                             f"choose from {sorted(table)}")
        mutated, hits = table[args.plant_defect](shipped_source())
        if not hits:
            raise SystemExit(f"{args.plant_defect} armed nothing")
        planted = compile_mutant(mutated, "planted_defect_beta_fused_B")
        log(f"PLANTED DEFECT: {args.plant_defect} ({hits} rewrite hits)")

    log("\n=== device legs ===")
    for name, cell, boundaries, steps in CASES:
        for value_class in (VALUE_CLASSES if steps > 1 else ("uniform",)):
            row = run_leg(cp, name, cell, boundaries, steps, product,
                          value_class=value_class, mutant=planted)
            passed, failures = verdict_of(row, require_launches=steps)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            log(f"  {name} [{value_class}]: passed={passed} {failures}")
            save(payload, args.out)

    log("\n=== the reduction leg: HAS_BETA=0 IS the shipped fused pair ===")
    reductions: List[Dict[str, Any]] = []
    for value_class in VALUE_CLASSES:
        row = reduction_leg(cp, product, CASES[0][1], CASES[0][2], 8,
                            value_class=value_class)
        reductions.append(row)
        payload["device_legs"].append(row)
        log(f"  reduction [{value_class}]: passed={row['passed']} "
            f"discriminating={row.get('discriminating')} {row['failures']}")
        save(payload, args.out)
    # THE FLOOR: at least one value class must have been able to tell the two
    # kernels apart, or every reduction row above agreed for a reason that has
    # nothing to do with this product.
    discriminated = [row["value_class"] for row in reductions
                     if row.get("discriminating")]
    payload["reduction_discriminating_value_classes"] = discriminated
    if not discriminated:
        for row in reductions:
            row["passed"] = False
            row.setdefault("failures", []).append(
                "NO value class discriminated HAS_BETA=0 from the array path on "
                "this run, so the reduction leg measured nothing at all")
    save(payload, args.out)

    log("\n=== armed harness mutations ===")
    row = run_leg(cp, "armed:no_substitution", CASES[0][1], CASES[0][2], 3,
                  product, install_fused=False, mutant=planted)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_magnetic_seam", CASES[0][1], CASES[0][2], 3,
                  product, freeze_magnetic=True, mutant=planted)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's magnetic seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.beta_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, args.out)

    log("\n=== armed host mutations ===")
    payload["mutations"].extend(run_host_mutations(cp, product))
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    def _mutation_ok(row: Dict[str, Any]) -> bool:
        # A DECLARED NULL MUST BE CONFIRMED, not merely permitted, and a mutation
        # that was never ARMED measured nothing whatever it then reported.
        if row.get("error") is not None:
            return False
        if row.get("kind") == "source" and not row.get("rewrite_hits"):
            return False
        return bool(row.get("caught")) is (row["expectation"] == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and mutation_ok
        and all(row["passed"] for row in payload["no_device_legs"]))
    payload["device_status"] = "RUN"
    payload["release"] = {
        "released": payload["passed"] and not args.plant_defect,
        "reasons": ([] if (payload["passed"] and not args.plant_defect) else
                    [f"device legs ok: {device_ok}",
                     f"refusals ok: {refusal_ok}",
                     f"mutations ok: {mutation_ok}",
                     "mutations whose measured outcome did not match their "
                     "declared expectation, or that were never armed: "
                     + str(sorted(row["mutation"] for row in payload["mutations"]
                                  if not _mutation_ok(row)))]
                    + ([f"A DEFECT WAS PLANTED ({args.plant_defect}): this run is a "
                        f"control and releases nothing"] if args.plant_defect else [])),
    }
    save(payload, args.out)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"\nverdict: {payload['passed']}  ->  {args.out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
