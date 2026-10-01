"""Device gate for the BFAST fused magnetic B/H pair on Triton.

BFAST ``step_B`` welded into ``update_H`` — the arm
:mod:`meep_gpu.triton_kernels.bfast_fused_magnetic_pair` adds, and the Triton twin
of the Metal family ``metal_kernels/bfast_fused_magnetic_pair.py`` certified by
``gate_metal_below_the_cut_fused_pairs.py``.

===========================================================================
WHAT THE PRODUCT CLAIMS, AND WHAT EACH LEG MEASURES
===========================================================================

The claim is narrow and checkable: this kernel is
``kernels.fused_curl_constitutive_B``, statement for statement, PLUS exactly the
``if HAS_BFAST:`` block ``bfast_curl.bfast_pml_curl_step`` adds to
``kernels.pml_curl_step``.

1. **TRANSCRIPTION (no device).** Two EXACT statement-list equalities against the
   shipped sources, read from the FILES so they bite on the merge bar as well as
   here. THE GRAIN IS A TOP-LEVEL ``ast`` STATEMENT, NOT A LINE, and that is not a
   convenience: the BFAST tail CONTAINS lines that also appear elsewhere in both
   bodies (``if BACKWARD:``, ``if BCX == METALLIC:``), so a flat line subtraction
   would leave those orphans behind and could not express the equality at all. The
   tail is exactly ONE ``If`` node. Nothing is ``ast.unparse``d — what is compared
   includes the PARENTHESISATION.
2. **REDUCTION (device).** ``HAS_BFAST = 0`` must reproduce the shipped
   ``fused_curl_constitutive_B`` BIT FOR BIT over a complete step, on the same
   volumes. The transcription claim measured on silicon instead of in source text,
   and the leg that would catch a mis-ordered argument in the plan's ``run`` that
   no source diff can see.
3. **IDENTITY (device).** One launch leaves the engine bit-identical, as uint32
   words, per COMPLETE driver step, to BOTH the CuPy array path AND the two
   separately certified Triton products it replaces (``BFAST PML`` on ``step_B``,
   ``BFAST run`` on ``update_H``), at one launch and at ~60, over every allocated
   volume, in three value classes.

===========================================================================
THE SEAM, AND THE THREE STATE VOLUMES
===========================================================================

``driver.step`` runs, in order (driver.py:3282-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

On the configurations this arm admits the two fill passes are DEAD —
``bfast_curl._bfast_grid_reasons`` refuses every fold — so the launch spans
``step_B``, ``zero_metal_B`` and ``update_H``.

**THE ``f_bfast_B*`` STATES ARE PART OF THE COMPARED INVENTORY.** They are the
BFAST IIR history, mutated IN PLACE and pointer-identically because the driver's
flux backup/restore depends on it (driver.py:4126-4135). The state scan picks them
up automatically (it enumerates every grid-shaped volume on ``Fields``) and
:data:`REQUIRED` names them, so a rename cannot drop them out of the comparison
silently. A leg where they did not MOVE fails the non-vacuity floor.

WHAT THE CORPUS SAYS, from ``results/fusion_matrix_triton_2026-08-20_closed/`` at
the B->H cell (``BFAST PML``, ``BFAST run``): ONE seam-instance,
``tests:TestReflectanceAngular.test_reflectance_angular_2_35_7``,
``in_seam_source == false``, ``live_in_seam_passes == []``. It is all-periodic with
``has_metallic == false``, so the inline ``zero_metal_B`` carry compiles to three
``False`` flags ON IT — which is why every wall-clear mutation is scored on a
WALLED case, where the flags are real.

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
  ``fastpath`` BEFORE the module-level sub-step functions this harness substitutes,
  and the ``BFAST PML`` / ``BFAST run`` arms are WIRED. The fast path is disabled
  explicitly and the per-route call counter must show the reference reaching the
  array path once per step for all five passes.
* **A false ``released``.** Only a device run writes ``release.released``.

POLICY, stamped: ``num_warps=1``, ``BLOCK=kernels.DEFAULT_BLOCK`` (256),
``enable_fp_fusion=kernels.ENABLE_FP_FUSION`` (False). BOTH float32 subnormal
policies are gated, one process each (``--subnormal-policy keep|flush``), installed
with ``strict=True`` before the first device compile.

Usage::

    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_bfast_fused_magnetic_pair.py --no-device
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

PRODUCT_MODULE = "meep_gpu.triton_kernels.bfast_fused_magnetic_pair"
KERNEL_NAME = "bfast_fused_curl_constitutive_B"
PACKAGE_DIR = os.path.join(API_ROOT, "meep_gpu", "triton_kernels")
KERNEL_FILE = os.path.join(PACKAGE_DIR, "bfast_fused_magnetic_pair.py")

#: A shear with all three components live, so no ``k1``/``k2`` word is zero for a
#: reason unrelated to the invariance gating.
BFAST_K: Tuple[float, float, float] = (0.13, -0.4, 0.07)

#: The device cases. ``(name, cell, boundaries, steps)``. All 3-D: BFAST carries no
#: dimensionality restriction, and a 3-D cell lets every wall be real (declaring an
#: INVARIANT axis metallic is refused by Grid by name, grid.py:842-877).
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int], ...] = (
    # THE CORPUS GEOMETRY: all-periodic, no wall — what
    # tests:TestReflectanceAngular.test_reflectance_angular_2_35_7 declares.
    ("corpus_periodic", (1.0, 1.1, 1.2), "periodic", 60),
    # EVERY WALL LIVE: all three ZM planes, all three metallic ghost arms, and the
    # BFAST advance's OWN ownership mask (a separate block from the curl's). This
    # is the mutation case, for that reason.
    ("wall_xyz", (1.0, 1.1, 1.2), "metallic", 60),
    # ONE WALLED AXIS: where a ROTATED wall table still clears the same NUMBER of
    # planes and clears the wrong one.
    ("wall_z", (1.0, 1.1, 1.2),
     {"x": "periodic", "y": "periodic", "z": "metallic"}, 60),
    # ONE LAUNCH.
    ("corpus_periodic_single_launch", (1.0, 1.1, 1.2), "periodic", 1),
)

MUTATION_CASE_INDEX = 1
MUTATION_STEPS = 4
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "signed_zero_lattice", "subnormal_band")

SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

#: Names that MUST appear in the scanned state inventory. THE THREE ``f_bfast_B*``
#: STATES ARE IN IT: they are the IIR history this kernel advances in place, and a
#: comparison that lost them would certify a kernel that never touched them.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
    "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
)

MATERIAL = ("eps", "inv_eps")

SEED = 0xB7A5

_TEMPORARY: List[str] = []

#: Device buffers a host mutation binds, held so they outlive the call that
#: bound them. See ``h7_states_are_a_copy_not_the_engines_arrays``.
_KEEPALIVE: List[Any] = []
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
        "meep_gpu/triton_kernels/bfast_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/bfast_curl.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_bfast_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}
# ---------------------------------------------------------------------------
# The shipped sources, read from the FILES
# ---------------------------------------------------------------------------

def _clean(lines: Sequence[str]) -> str:
    """Comment-stripped, blank-stripped text. A prose edit is not a transcription."""
    kept = [line.split("#", 1)[0].rstrip() for line in lines]
    return "\n".join(line for line in kept if line.strip())


def shipped_body(path: str, name: str, *, with_decorators: bool = False) -> str:
    """One shipped kernel's source, docstring removed. Read from the FILE.

    Read that way on EVERY host — not only the laptop. Two spellings of the same
    text (``inspect.getsource`` on a CUDA box, a file read on the merge bar) would
    let a rewrite match in one place and miss in the other, which is exactly how a
    mutation silently disarms.

    ``with_decorators`` prepends the decorator list, which
    ``ast.get_source_segment`` on a FunctionDef EXCLUDES. Without it a mutant
    compiled from the segment is a PLAIN PYTHON FUNCTION with no ``[grid]``
    launcher, and the first mutation leg dies with "'function' object is not
    subscriptable" — measured 2026-08-20 on a sibling gate.
    """
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    node = next((found for found in ast.walk(tree)
                 if isinstance(found, ast.FunctionDef) and found.name == name), None)
    if node is None:
        raise AssertionError(f"{name} is not defined in {path}")
    segment = ast.get_source_segment(text, node)
    if segment is None:  # pragma: no cover
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


def body_segments(path: str, name: str) -> List[str]:
    """A shipped kernel's BODY as a list of top-level STATEMENT sources.

    THE GRAIN OF EVERY TRANSCRIPTION COMPARISON IN THIS GATE. A per-LINE
    comparison cannot express this product's claim: the BFAST tail contains lines
    that also appear elsewhere in both bodies, so subtracting them by text would
    leave orphans. The tail is exactly ONE ``ast.If`` node.

    Continuation lines are dedented by the function's own ``col_offset`` so a body
    nested inside ``if triton is not None:`` compares equal to one at module level.
    """
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
    margin = node.col_offset
    out: List[str] = []
    for statement in body:
        segment = ast.get_source_segment(text, statement)
        if segment is None:  # pragma: no cover
            raise AssertionError(f"{name}: unavailable source segment")
        lines = segment.splitlines()
        lines = [lines[0]] + [line[margin:] if line[:margin].strip() == ""
                              else line.lstrip() for line in lines[1:]]
        out.append(_clean(lines))
    return out


def shipped_source() -> str:
    """The product's kernel text, decorator included — the mutation table's input."""
    return shipped_body(KERNEL_FILE, KERNEL_NAME, with_decorators=True)


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, as two exact statement-list equalities
# ---------------------------------------------------------------------------

def transcription_leg() -> Dict[str, Any]:
    """The claim this whole product rests on, made mechanical."""
    kernels_py = os.path.join(PACKAGE_DIR, "kernels.py")
    bfast_py = os.path.join(PACKAGE_DIR, "bfast_curl.py")
    fused = body_segments(KERNEL_FILE, KERNEL_NAME)
    ordinary_fused = body_segments(kernels_py, "fused_curl_constitutive_B")
    ordinary_curl = body_segments(kernels_py, "pml_curl_step")
    bfast_curl_body = body_segments(bfast_py, "bfast_pml_curl_step")

    findings: List[str] = []
    tail = [s for s in bfast_curl_body if s.startswith("if HAS_BFAST:")]
    delta = [s for s in bfast_curl_body if s not in ordinary_curl]
    if len(tail) != 1:
        findings.append(f"the BFAST tail is {len(tail)} statements, not exactly one")
    if delta != tail:
        findings.append(
            f"bfast_pml_curl_step's own delta over pml_curl_step is not the tail: "
            f"{[s[:60] for s in delta]}")

    without_tail = [s for s in fused if not s.startswith("if HAS_BFAST:")]
    if without_tail != ordinary_fused:
        index = next((i for i, (a, b) in enumerate(zip(without_tail, ordinary_fused))
                      if a != b), min(len(without_tail), len(ordinary_fused)))
        findings.append(
            f"minus the tail this is NOT fused_curl_constitutive_B; first "
            f"difference at statement {index}: "
            f"{[s[:80] for s in without_tail[index:index + 1]]} vs "
            f"{[s[:80] for s in ordinary_fused[index:index + 1]]}")

    weld = [s for s in ordinary_fused if s not in ordinary_curl]
    if not weld:
        findings.append("the weld is empty; the second equality compares nothing")
    without_weld = [s for s in fused if s not in weld]
    if without_weld != bfast_curl_body:
        index = next((i for i, (a, b) in enumerate(zip(without_weld, bfast_curl_body))
                      if a != b), min(len(without_weld), len(bfast_curl_body)))
        findings.append(
            f"minus the weld this is NOT bfast_pml_curl_step; first difference at "
            f"statement {index}: {[s[:80] for s in without_weld[index:index + 1]]} "
            f"vs {[s[:80] for s in bfast_curl_body[index:index + 1]]}")

    invented = [s for s in fused
                if s not in set(bfast_curl_body) | set(ordinary_fused)]
    if invented:
        findings.append(f"statements in NEITHER shipped body: "
                        f"{[s[:80] for s in invented]}")

    return {"leg": "transcription", "device": False,
            "statements_in_the_product": len(fused),
            "weld_statements_derived_from_the_shipped_pair": len(weld),
            "the_tail_is_one_statement": len(tail) == 1,
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
    cell_size = keywords.pop("cell_size", (0.8, 0.8, 0.8))
    dimensions = keywords.pop("dimensions", 3)
    shear = keywords.pop("bfast_scaled_k", BFAST_K)
    grid = Grid(resolution=10.0, cell_size=cell_size, dimensions=dimensions,
                courant=0.35, bfast_scaled_k=shear, **keywords)
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
    ("bfast_inactive", {"bfast_scaled_k": (0.0, 0.0, 0.0)}, ()),
    ("complex_storage", {"complex_storage": True}, ()),
    ("no_absorber", {"pml_thickness": 0}, ()),
    ("metallic_walls", {"boundaries": "metallic"}, ()),
    ("bloch", {"k_point": (0.2, 0.0, 0.0)}, ()),
)


def equivalence_leg() -> Dict[str, Any]:
    """``weld => BFAST curl arm AND BFAST H arm``, over one-clause perturbations."""
    from meep_gpu.grid import Mirror  # noqa: PLC0415
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)
    cases = list(EQUIVALENCE_CASES) + [
        # A folded axis cannot carry a LOW-face layer: cell 0 there is the mirror
        # plane and pml._resolve_mirror_faces RAISES on a per-side request naming
        # it. This case passes its own table for that reason.
        ("folded", {"symmetry": (Mirror("Y", 1),),
                    "thickness": ((2, 2), (0, 2), (2, 2))}, ()),
    ]
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    admitted = 0
    for name, keywords, sources in cases:
        fields, pml = _laptop_fixture(**dict(keywords))
        weld = product.bfast_fused_magnetic_pair_coverage(fields, pml, sources)
        curl = bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_B")
        magnetic = bfast_curl.bfast_run_constitutive_coverage(fields, pml, "H")
        never_wider = (not weld.covered) or (curl.covered and magnetic.covered)
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
CELL = ("B->H", "BFAST PML", "BFAST run")


def corpus_admission_leg() -> Dict[str, Any]:
    """The rows this arm can serve, counted rather than cited."""
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
        if not configuration.get("bfast_active"):
            continue
        if not configuration.get("pml_active"):
            continue
        if configuration.get("force_complex_fields"):
            continue
        if configuration.get("has_symmetry") or any(configuration.get("mirrored") or ()):
            continue
        if configuration.get("cylindrical"):
            continue
        if configuration.get("has_bloch") or configuration.get("has_nonlinearity"):
            continue
        if float(configuration.get("beta") or 0.0) != 0.0:
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
            picks = rng.integers(0, 2, size=shape)
            values = np.where(picks == 0, 0.0, -0.0)
        elif value_class == "subnormal_band":
            values = rng.standard_normal(shape) * 1e-38
        else:  # pragma: no cover - VALUE_CLASSES is the whole domain
            raise ValueError(f"unknown value class {value_class!r}")
        return np.ascontiguousarray(values.astype(np.float32))

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(draw()))
    # E, H and the BFAST states are DERIVED or internal and `set_field` refuses
    # them by name (driver.py:3909). Written in place here, which is legal for a
    # harness. SEEDING THE STATES IS LOAD-BEARING, not tidiness: the tail's advance
    # is `total - 2.0*state`, so a zero state hides the `-2*state` term entirely
    # and the mutation that drops it would come back UNCAUGHT.
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                 "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
                 "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())


def build_driver(cp, cell, boundaries, seed: int, value_class: str,
                 electric: bool, shear: Sequence[float] = BFAST_K):
    """One BFAST PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=3,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        bfast_scaled_k=tuple(float(v) for v in shear),
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (2.25 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml(2)
    if electric:
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    # THE FAST PATH IS DISABLED, EXPLICITLY — the `BFAST PML` / `BFAST run` arms
    # are WIRED, so a driver left alone would step this very family on Triton and
    # the "array path" reference would be no such thing (driver.py:3281-3282).
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

    They are the arms ``launch.plan_step`` really selects on a BFAST row today —
    ``BFAST PML`` on ``step_B`` and ``BFAST run`` on ``update_H`` — so this route
    is not a second model of the seam, it is the shipped one.

    The three in-seam passes stay on the ARRAY PATH here: no Triton product owns
    the wall clear, and both symmetry fills are no-ops without a mirror plane. All
    three are COUNTED, so a difference in which of them ran is a counter event
    rather than an invisible correction.
    """
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    curl = bfast_curl.plan_bfast_pml_curl(driver.fields, driver.pml, "step_B")
    constitutive = bfast_curl.plan_bfast_run_constitutive(
        driver.fields, driver.pml, "H")
    missing = [name for name, plan in
               (("BFAST PML curl", curl),
                ("BFAST run constitutive", constitutive)) if plan is None]
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

        Scored on ``wall_xyz``, where all three flags are really True."""
        plan.zero_metal = (0, 0, 0)
        return plan

    def h4_wall_table_is_the_d_side(driver, plan):
        """``zero_metal`` rotated: the same NUMBER of planes, the WRONG ones.

        Scored on ``wall_z``, where exactly ONE axis is walled so the rotation
        moves the clear onto a component the array path leaves alone."""
        flags = tuple(plan.zero_metal)
        plan.zero_metal = (flags[2], flags[0], flags[1])
        return plan

    def h5_k1_k2_exchanged_per_target(driver, plan):
        """``k1`` and ``k2`` exchanged within each target's pair.

        They multiply DIFFERENT operand sums and are read from DIFFERENT axes of
        the shear (``bfast_curl_coefficients``: ``k1 = bfast[own(second)]`` gates on
        ``have_m``, ``k2 = bfast[own(first)]`` on ``have_p`` — CROSS-gated,
        stepping.py:909-914). Exchanging them keeps every magnitude and every sign
        in the six-tuple and is invisible to anything but a byte comparison."""
        k = tuple(plan.ks)
        plan.ks = (k[1], k[0], k[3], k[2], k[5], k[4])
        return plan

    def h6_bfast_switched_off_at_the_host(driver, plan):
        """``HAS_BFAST = 0`` on a run whose shear is real.

        The counterpart of the REDUCTION leg, armed: there the same flag must
        reproduce the shipped ordinary kernel; here it must be CAUGHT, because the
        array path really does fold the tail in."""
        plan.has_bfast = 0
        return plan

    def h7_states_are_a_copy_not_the_engines_arrays(driver, plan):
        """The IIR states bound to COPIES instead of the engine's own arrays.

        The advance is written in place and pointer-identically because the
        driver's flux backup/restore depends on it (driver.py:4126-4135). Bound to
        copies, the kernel still computes the right advance and still stores it —
        into buffers nothing else reads — so ``f_bfast_B*`` freeze while ``B``
        keeps evolving from a state that never advances. NOT a crash; a silently
        different run."""
        import cupy as cp  # noqa: PLC0415
        from meep_gpu.triton_kernels.launch import CupyPointer  # noqa: PLC0415

        fields = driver.fields
        copies = [cp.array(getattr(fields, name), copy=True)
                  for name in ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz")]
        # Held on the plan so the buffers outlive this call; a freed copy would
        # make the leg a use-after-free rather than the defect it is armed for.
        plan._states = tuple(CupyPointer(a) for a in copies)  # noqa: SLF001
        # HELD MODULE-SIDE, not on the plan: BfastFusedMagneticPairPlan declares
        # __slots__, so an attribute the class does not name raises AttributeError
        # (measured 2026-08-21 — it aborted the host-mutation leg after h6 had
        # already scored). The buffers must outlive this call all the same: a freed
        # copy would make the leg a use-after-free rather than the defect it arms.
        _KEEPALIVE.append(copies)
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
        ("h5_k1_k2_exchanged_per_target", "the cross-gated k1/k2 pairing",
         "caught", h5_k1_k2_exchanged_per_target),
        ("h6_bfast_switched_off_at_the_host", "the HAS_BFAST constexpr", "caught",
         h6_bfast_switched_off_at_the_host),
        ("h7_states_are_a_copy_not_the_engines_arrays",
         "the pointer-identical state mutation", "caught",
         h7_states_are_a_copy_not_the_engines_arrays),
    )


#: Host mutation id -> the CASES index it is scored on.
HOST_MUTATION_CASE: Dict[str, int] = {
    "h4_wall_table_is_the_d_side": 2,   # wall_z: exactly ONE walled axis
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
        else product.bfast_fused_curl_constitutive_B_kernel())
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
        plan = product.plan_bfast_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), kernel=kernel)
        if plan is None:
            verdict = product.bfast_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if host_mutation is not None:
            row["host_mutation"] = host_mutation[0]
            plan = host_mutation[3](fused, plan)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["zero_metal_flags"] = list(plan.zero_metal)
        row["bfast_words"] = [list(plan.ks), plan.has_bfast]
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
        steps = int(row.get("steps_budget") or 0)
        for name in SEAM_PASSES:
            seen = counts.get(f"array/array_path:{name}", 0)
            if seen != steps:
                failures.append(
                    f"the reference route reached the array path for {name} "
                    f"{seen} times, expected {steps}: the fast path was not off")
    return (not failures), failures
# ---------------------------------------------------------------------------
# THE REDUCTION LEG — HAS_BFAST = 0 must BE the shipped ordinary kernel
# ---------------------------------------------------------------------------

def reduction_leg(cp, product, cell, boundaries, steps: int,
                  value_class: str = "uniform") -> Dict[str, Any]:
    """``HAS_BFAST = 0`` reproduces ``kernels.fused_curl_constitutive_B`` bit for bit.

    The transcription claim, measured on silicon instead of in source text. Three
    drivers, seeded identically, on a REAL BFAST run:

    * route A launches THIS product's kernel with ``has_bfast = 0``;
    * route B launches the SHIPPED ``kernels.fused_curl_constitutive_B`` over the
      same volumes, through ``launch.plan_fused_pair_from_arrays`` — the bare-array
      route, because ``coverage.fused_pair_coverage`` refuses a BFAST run by name
      and would return ``None``.

    NEITHER IS COMPARED AGAINST THE ARRAY PATH HERE, deliberately: the array path
    really does fold the tail in, so both routes must differ from it. What this leg
    measures is that the two KERNELS agree.

    THE DISCRIMINATION PRECONDITION IS THE OTHER HALF. Two kernels that both did
    nothing would also agree, so route A must additionally DIFFER from the array
    path; otherwise ``has_bfast = 0`` is indistinguishable from the shipped BFAST
    kernel on this value class and the leg measured a tautology. That is a property
    of the VALUE CLASS, so it is reported as ``discriminating`` per class with a
    gate-level floor that at least one class must discriminate.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import (  # noqa: PLC0415
        plan_fused_pair_from_arrays,
    )

    seed = case_seed("reduction", str(cell), str(boundaries), value_class)
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric=True)
            for _ in range(3)]
    reference, shipped, reduced = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    row: Dict[str, Any] = {"leg": "reduction:has_bfast_zero", "device": True,
                           "steps_budget": steps, "value_class": value_class,
                           "seed": seed, "shape": list(reference.shape),
                           "boundaries": boundaries}
    try:
        reduced_plan = product.plan_bfast_fused_magnetic_pair(
            reduced.fields, reduced.pml, ())
        if reduced_plan is None:
            raise AssertionError("the product refused the reduction case")
        reduced_plan.has_bfast = 0

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
            # THE BFAST STATES ARE EXCLUDED FROM THIS ONE COMPARISON, by name.
            # `has_bfast = 0` does not touch them and the shipped ordinary kernel
            # has no pointer to them at all, while the ARRAY-path reference on the
            # same seed advances them — so comparing them here would measure the
            # absence of a tail both routes deliberately skip. Every OTHER volume,
            # including all six B/fu_B and all three H/f_w_H, is compared in full,
            # and the states ARE compared in every identity leg above.
            def without_states(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
                return {k: v for k, v in state.items()
                        if not k.startswith("f_bfast_")}

            versus_shipped = first_divergence(without_states(after_reduced),
                                              without_states(after_shipped))
            versus_array = first_divergence(without_states(after_reduced),
                                            without_states(after_reference))
            row["per_step"].append({
                "step": step,
                "has_bfast_zero_vs_shipped_fused_pair": versus_shipped,
                "has_bfast_zero_vs_array_path": versus_array,
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
            set(final) - set(ever_moved) - set(material)
            - {"f_bfast_Bx", "f_bfast_By", "f_bfast_Bz"})
        row["compared_volumes_exclude"] = ["f_bfast_B*"]
        row["launches"] = dict(counter)
        failures: List[str] = []
        if row["first_divergence"] is not None:
            failures.append(
                f"HAS_BFAST=0 is NOT the shipped fused pair: {row['first_divergence']}")
        if row["arrays_never_moved"]:
            failures.append(f"VACUOUS: {row['arrays_never_moved']} never moved")
        row["discriminating"] = row.get("divergence_from_the_array_path") is not None
        if not row["discriminating"]:
            row["not_discriminating_because"] = (
                "HAS_BFAST=0 matched the ARRAY PATH too, so this value class cannot "
                "separate the BFAST kernel from the ordinary one")
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
    ast.parse(text)   # a mutation that does not parse is not a mutation
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_bfast_pair.py", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_bfast_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace a contiguous run of statements, preserving the block's indent.

    MATCHED ON COMMENT-STRIPPED TEXT, so a rewrite stays armed across a trailing
    comment or a reflow. Every replacement line takes the indent of the block's
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

    def m1_advance_history_term_dropped(source: str) -> Tuple[str, int]:
        """``adv = total - 2.0*state`` becomes ``adv = total``: the IIR history gone.

        Only visible because the harness SEEDS the states — with a zero state this
        rewrite changes nothing, which is why ``seed_values`` writes them and says
        so. Unguarded within the tail, and every case here has ``HAS_BFAST = 1``."""
        return _rewrite_block(
            source,
            ["adv0 = total0 - (2.0 * st0)",
             "adv1 = total1 - (2.0 * st1)",
             "adv2 = total2 - (2.0 * st2)"],
            ["adv0 = total0", "adv1 = total1", "adv2 = total2"])

    def m2_bfast_sums_become_differences(source: str) -> Tuple[str, int]:
        """BFAST SUMS the pairs the curl DIFFERENCES (stepping.py:1594-1598).

        Turning the sums into differences is the single most natural wrong
        transcription of a tail whose operands are the curl's own loads."""
        return _rewrite_block(
            source,
            ["total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b))",
             "total1 = (k1_1 * (a_z + a)) - (k2_1 * (c_x + c))",
             "total2 = (k1_2 * (b_x + b)) - (k2_2 * (a_y + a))"],
            ["total0 = (k1_0 * (c_y - c)) - (k2_0 * (b_z - b))",
             "total1 = (k1_1 * (a_z - a)) - (k2_1 * (c_x - c))",
             "total2 = (k1_2 * (b_x - b)) - (k2_2 * (a_y - a))"])

    def m3_tail_scaled_by_dtdx(source: str) -> Tuple[str, int]:
        """``dtdx`` applied to the tail. There is none in the array path (S:836-837)."""
        return _rewrite_block(
            source,
            ["curl0 = curl0 - adv0",
             "curl1 = curl1 - adv1",
             "curl2 = curl2 - adv2"],
            ["curl0 = curl0 - (dtdx * adv0)",
             "curl1 = curl1 - (dtdx * adv1)",
             "curl2 = curl2 - (dtdx * adv2)"])

    def m4_state_stored_before_the_advance_mask(source: str) -> Tuple[str, int]:
        """The advance masked AFTER the state store instead of before (S:902).

        On a walled axis the array path stores the MASKED advance; storing the raw
        one leaves the history carrying a term the curl never saw. Guarded on a
        METALLIC axis being present — scored on ``wall_xyz``, which has three."""
        source, hits = _rewrite_block(
            source,
            ["tl.store(s0 + idx, st0 + adv0, mask=live)",
             "tl.store(s1 + idx, st1 + adv1, mask=live)",
             "tl.store(s2 + idx, st2 + adv2, mask=live)"],
            ["tl.store(s0 + idx, st0 + raw0, mask=live)",
             "tl.store(s1 + idx, st1 + raw1, mask=live)",
             "tl.store(s2 + idx, st2 + raw2, mask=live)"])
        if not hits:
            return source, 0
        return _rewrite_block(
            source,
            ["adv0 = total0 - (2.0 * st0)",
             "adv1 = total1 - (2.0 * st1)",
             "adv2 = total2 - (2.0 * st2)"],
            ["adv0 = total0 - (2.0 * st0)",
             "adv1 = total1 - (2.0 * st1)",
             "adv2 = total2 - (2.0 * st2)",
             "raw0 = adv0", "raw1 = adv1", "raw2 = adv2"])

    def m5_tail_added_instead_of_subtracted(source: str) -> Tuple[str, int]:
        """The caller-subtracts sign convention (S:904) inverted."""
        return _rewrite_block(
            source,
            ["curl0 = curl0 - adv0",
             "curl1 = curl1 - adv1",
             "curl2 = curl2 - adv2"],
            ["curl0 = curl0 + adv0",
             "curl1 = curl1 + adv1",
             "curl2 = curl2 + adv2"])

    def m6_wall_table_is_the_d_side(source: str) -> Tuple[str, int]:
        """The DIAGONAL wall table rotated onto the OFF-DIAGONAL pairing.

        Scored on ``wall_xyz``, where every ``ZM_*`` is True."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)",
             "if ZM_Y:", "v1 = tl.where(at_y, 0.0, v1)",
             "if ZM_Z:", "v2 = tl.where(at_z, 0.0, v2)"],
            ["if ZM_X:", "    v1 = tl.where(at_x, 0.0, v1)",
             "if ZM_Y:", "    v2 = tl.where(at_y, 0.0, v2)",
             "if ZM_Z:", "    v0 = tl.where(at_z, 0.0, v0)"])

    def m7_zero_metal_only_on_the_store(source: str) -> Tuple[str, int]:
        """Cleared into ``B``, but the constitutive half reads the UNCLEARED value.

        Same guard note as ``m6``; scored on ``wall_xyz``, where all three flags
        were True on the pristine kernel too."""
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

    def m8_history_read_after_write(source: str) -> Tuple[str, int]:
        """``f_w`` read AFTER it is written: wrong wherever ``kms != 0``. Unguarded."""
        return _rewrite_block(
            source,
            ["prev0 = tl.load(w0 + idx, mask=live, other=0.0)",
             "src0 = v0",
             "tl.store(w0 + idx, src0, mask=live)"],
            ["src0 = v0",
             "tl.store(w0 + idx, src0, mask=live)",
             "prev0 = tl.load(w0 + idx, mask=live, other=0.0)"])

    def m9_curl_parentheses_flattened(source: str) -> Tuple[str, int]:
        """``stepping._curl_from_operands``' grouping flattened. Unguarded."""
        needle = "curl0 = dtdx * ((c_y - c) + (b - b_z))"
        return source.replace(needle, "curl0 = dtdx * (c_y - c + b - b_z)"), \
            source.count(needle)

    def m10_constitutive_association(source: str) -> Tuple[str, int]:
        """Right-associated accumulation: same algebra, different float32 rounding."""
        return _rewrite_block(
            source,
            ["a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"],
            ["a0 = a0 + (kp_0 * src0 - km_0 * prev0)"])

    def m11_advance_mask_dropped(source: str) -> Tuple[str, int]:
        """The BFAST advance's OWN ownership mask undone on the forward arm.

        A SEPARATE block from the curl's mask, applied to ``adv`` BEFORE the state
        store (S:902). Guarded on ``BCX/BCY/BCZ == METALLIC`` and on the ``else``
        (forward) arm, all of which ``wall_xyz`` enters."""
        return _rewrite_block(
            source,
            ["if BCX == METALLIC:", "adv0 = tl.where(at_x, 0.0, adv0)",
             "if BCY == METALLIC:", "adv1 = tl.where(at_y, 0.0, adv1)",
             "if BCZ == METALLIC:", "adv2 = tl.where(at_z, 0.0, adv2)"],
            ["adv0 = adv0", "adv1 = adv1", "adv2 = adv2"])

    def m12_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
        ("m1_advance_history_term_dropped", "the IIR history", "caught",
         m1_advance_history_term_dropped),
        ("m2_bfast_sums_become_differences", "sums vs differences", "caught",
         m2_bfast_sums_become_differences),
        ("m3_tail_scaled_by_dtdx", "the tail's missing dtdx", "caught",
         m3_tail_scaled_by_dtdx),
        ("m4_state_stored_before_the_advance_mask", "the advance mask's position",
         "caught", m4_state_stored_before_the_advance_mask),
        ("m5_tail_added_instead_of_subtracted", "the caller-subtracts convention",
         "caught", m5_tail_added_instead_of_subtracted),
        ("m6_wall_table_is_the_d_side", "the DIAGONAL vs the OFF-DIAGONAL table",
         "caught", m6_wall_table_is_the_d_side),
        ("m7_zero_metal_only_on_the_store", "the register-vs-memory carry",
         "caught", m7_zero_metal_only_on_the_store),
        ("m8_history_read_after_write", "the f_w ordering", "caught",
         m8_history_read_after_write),
        ("m9_curl_parentheses_flattened", "the curl grouping", "caught",
         m9_curl_parentheses_flattened),
        ("m10_constitutive_association", "float32 association", "caught",
         m10_constitutive_association),
        ("m11_advance_mask_dropped", "the advance's own ownership mask", "caught",
         m11_advance_mask_dropped),
        ("m12_commuted_multiply", "commuted multiply", "null", m12_commuted_multiply),
    )


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
        mutant = compile_mutant(mutated, f"mutant_{index}_bfast_fused_B")
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
                     "bfast_words": leg.get("bfast_words"),
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
    for name, shear, sources, needle in (
        ("magnetic_source", BFAST_K, (_Magnetic(),), "is magnetic"),
        ("undeclared_sources", BFAST_K, None, "was not declared"),
        ("bfast_inactive", (0.0, 0.0, 0.0), (), "BFAST"),
    ):
        driver = build_driver(cp, cell, walls, SEED, "uniform", electric=False,
                              shear=shear)
        try:
            verdict = product.bfast_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources)
            plan = product.plan_bfast_fused_magnetic_pair(
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
        HERE, "results", "triton_bfast_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--subnormal-policy", default="keep",
                        choices=("keep", "flush"))
    parser.add_argument(
        "--plant-defect", default=None,
        help="install a named kernel mutation as THE PRODUCT'S kernel for every "
             "device leg. The release verdict must FLIP to False.")
    args = parser.parse_args(argv)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_bfast_fused_magnetic_pair",
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
        # MEASURED REFUSAL: `install_subnormal_policy('flush')` raises in a process
        # that has not imported MEEP (subnormal_policy.py:1920) because
        # `mp.set_zero_subnormals` is the only exposure of this process's FTZ/DAZ
        # bits the package may use, and it will not import MEEP itself. A parity
        # probe may do what the package may not. NOT UNCONDITIONAL: importing MEEP
        # moves the host FPU.
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
        planted = compile_mutant(mutated, "planted_defect_bfast_fused_B")
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

    log("\n=== the reduction leg: HAS_BFAST=0 IS the shipped fused pair ===")
    reductions: List[Dict[str, Any]] = []
    for value_class in VALUE_CLASSES:
        row = reduction_leg(cp, product, CASES[0][1], CASES[0][2], 8,
                            value_class=value_class)
        reductions.append(row)
        payload["device_legs"].append(row)
        log(f"  reduction [{value_class}]: passed={row['passed']} "
            f"discriminating={row.get('discriminating')} {row['failures']}")
        save(payload, args.out)
    discriminated = [row["value_class"] for row in reductions
                     if row.get("discriminating")]
    payload["reduction_discriminating_value_classes"] = discriminated
    if not discriminated:
        for row in reductions:
            row["passed"] = False
            row.setdefault("failures", []).append(
                "NO value class discriminated HAS_BFAST=0 from the array path on "
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
    pristine = kernel_ptx(product.bfast_fused_curl_constitutive_B)
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
