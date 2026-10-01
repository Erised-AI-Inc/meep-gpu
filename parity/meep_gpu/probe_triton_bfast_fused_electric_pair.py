"""Device gate for the BFAST fused ELECTRIC D/E pair on Triton.

DEVICE STATUS: **UNRUN.** This module is written to be run on a CUDA host with an
    idle device; nothing in this tree may cite it as a release until an artifact
    exists with ``device_status: RUN`` and ``release.released: true``.

Beta ``step_D`` welded into ``update_E`` — the ELECTRIC twin of
:mod:`meep_gpu.triton_kernels.bfast_fused_magnetic_pair`, and the Triton twin of
``metal_kernels/bfast_fused_electric_pair.py``.

===========================================================================
WHAT THE PRODUCT CLAIMS, AND WHAT EACH LEG MEASURES
===========================================================================

The claim is narrow and checkable: this kernel is
``kernels.fused_curl_constitutive_D``, character for character, PLUS exactly the
ONE ``if HAS_BFAST:`` statement ``bfast_curl.bfast_pml_curl_step`` adds to
``kernels.pml_curl_step``.

1. **TRANSCRIPTION (no device).** Two EXACT statement-list equalities against the
   shipped sources, read from the FILES so they bite on the merge bar as well as
   here: the kernel minus the BFAST insert IS ``fused_curl_constitutive_D``, and
   the kernel minus the weld IS ``bfast_pml_curl_step``. THE INSERT IS DERIVED
   rather than listed — it is whatever ``bfast_pml_curl_step`` has that
   ``kernels.pml_curl_step`` does not — so a change to the shipped tail moves this
   leg with it instead of needing a hand-maintained list. Neither is ``ast.unparse``d —
   what is being compared includes the PARENTHESISATION, and unparsing re-derives
   minimal parentheses.
2. **REDUCTION (device).** ``HAS_BFAST = 0`` must reproduce the shipped
   ``kernels.fused_curl_constitutive_D`` BIT FOR BIT over a complete step, on the
   same volumes. This is the transcription claim measured on silicon instead of in
   source text, and it is the leg that would catch a mis-ordered argument in the
   plan's ``run`` that no amount of source diffing can see.
3. **IDENTITY (device).** One launch of the fused kernel leaves the engine
   bit-identical, as uint32 words, per COMPLETE driver step, to BOTH the CuPy array
   path AND the two separately certified Triton products it replaces (``BFAST
   PML`` on ``step_D``, ``BFAST run`` on ``update_E``), at one launch and at ~40,
   over every allocated volume, in three value classes.
4. **THE CARRY (device), and it is what makes this cell worth anything at all.**
   The single corpus row declares an ELECTRIC source, which the driver injects
   BETWEEN the two halves. Every CARRY case is run TWICE — once with the shipped
   :class:`~meep_gpu.deposit_repair.LeadingRepairPlan` /
   :class:`~meep_gpu.deposit_repair.TrailingRepairPlan` pair and once WITHOUT it —
   and the unbracketed run MUST DIVERGE. A carry family whose null control agreed
   would be measuring a seam that carried no deposit.

===========================================================================
THE SEAM
===========================================================================

``driver.step`` runs, in order (driver.py:3292-3304)::

    step_D -> ELECTRIC SOURCES -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

On the configurations this arm admits the two fill passes are DEAD — the BFAST
family refuses every fold — so the launch spans
``step_D``, ``zero_metal_D`` and ``update_E``, with the electric injection carried
across it by the shipped repair.

WHAT THE CORPUS SAYS, from ``results/fusion_matrix_triton_2026-08-31_folded_disp``
at the D->E cell (``BFAST PML``, ``BFAST run``): ONE seam-instance,
``examples:refl-angular-kz2d.py``, ``in_seam_source == true`` and
``in_seam_source_blocks == false``. The ``corpus_admission`` leg re-derives that
from the census rather than citing it.

THAT ROW IS ALL-PERIODIC WITH ``has_metallic == false``, so the inline
``zero_metal_D`` carry compiles to three ``False`` flags ON IT. Every wall-clear
mutation is therefore scored on a WALLED case, where the flags are real —
otherwise the rewrite would touch lines the scored grid never reaches and report
UNCAUGHT while measuring nothing.

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_bfast_fused_electric_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_bfast_fused_electric_pair.py \\
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
PRODUCT_MODULE = "meep_gpu.triton_kernels.bfast_fused_electric_pair"
KERNEL_FILE = os.path.join(PACKAGE_DIR, "bfast_fused_electric_pair.py")
KERNEL_NAME = "bfast_fused_curl_constitutive_D"

#: A live three-component shear — the class ``test_reflectance_angular_2_35_7``
#: declares, and one whose three components DIFFER so a rotated k table is visible.
BFAST_K: Tuple[float, float, float] = (0.13, -0.21, 0.07)


def bfast_insert() -> List[str]:
    """The block the BFAST tail adds, DERIVED from the two shipped curls.

    Never listed: it is whatever ``bfast_curl.bfast_pml_curl_step`` has that
    ``kernels.pml_curl_step`` does not, so a change to the shipped tail moves every
    leg that reads this rather than leaving a hand-maintained list behind.
    """
    package = os.path.join(API_ROOT, "meep_gpu", "triton_kernels")
    tail = body_statements(os.path.join(package, "bfast_curl.py"),
                           "bfast_pml_curl_step")
    plain = body_statements(os.path.join(package, "kernels.py"), "pml_curl_step")
    return [line for line in tail if line not in plain]

#: The device cases. ``(name, cell, boundaries, steps)``.
#:
#: EVERY CELL IS 3-D. BFAST's ``k1``/``k2`` are CROSS-GATED on ``grid.is_invariant``
#: (bfast_curl.py:346-349 — the grid's own declared dimensionality, never a shape
#: test), so on a 2-D cell the z-invariant axis zeroes two of the six words and the
#: tail's third target degenerates. A 3-D cell is the only place all six are live.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any, int], ...] = (
    # THE CORPUS GEOMETRY: all-periodic, no wall. ZM_* all compile to False and
    # every ghost arm is the periodic wrap.
    ("corpus_periodic", (0.8, 0.8, 0.8), "periodic", 40),
    # ALL THREE WALLS LIVE: ZM_X/ZM_Y/ZM_Z fire, every metallic ghost arm is
    # entered, and the ADVANCE mask — which the tail applies BEFORE its state store
    # — is live on every target. This is the mutation case, for that reason.
    ("wall_xyz", (0.8, 0.8, 0.8),
     {"x": "metallic", "y": "metallic", "z": "metallic"}, 40),
    # ONE WALLED AXIS ONLY: the case where a ROTATED wall table still clears the
    # same NUMBER of planes and clears the wrong ones.
    ("wall_x", (0.8, 0.8, 0.8),
     {"x": "metallic", "y": "periodic", "z": "periodic"}, 40),
    # ONE LAUNCH: a divergence here is attributable to a single launch rather than
    # to an accumulation.
    ("corpus_periodic_single_launch", (0.8, 0.8, 0.8), "periodic", 1),
)

CASES_BY_NAME: Dict[str, Tuple[Any, ...]] = {case[0]: case for case in CASES}

#: The case every kernel mutation is scored on unless it names another.
#: ``wall_xyz``:
#: the wall-clear lines are dead on the corpus geometry.
DEFAULT_MUTATION_CASE = "wall_xyz"

#: Per mutation, the case it is scored on. TRAP: a mutation scored on a grid that
#: never ENTERS the branch it rewrites reports UNCAUGHT while measuring nothing.
MUTATION_CASE: Dict[str, str] = {}

#: Per mutation, axes that must carry a live WALL on the scored case.
MUTATION_REQUIRES_WALL: Dict[str, Tuple[str, ...]] = {
    "m_wall_clear_dropped": ("x",),
    "m_wall_clear_is_the_magnetic_familys_one_row": ("x",),
    "m_wall_clear_after_the_store": ("x",),
    "m_advance_mask_dropped": ("x", "y"),
    "m_advance_mask_after_the_state_store": ("x", "y"),
}

#: The CARRY family: the same grids with a real ELECTRIC deposit IN THIS SEAM.
#: THIS IS THE PRODUCT, not an extra — the cell's only corpus row sources
#: electrically. One all-periodic grid (the corpus geometry) and one walled grid,
#: where the deposit and the wall clear can interact.
CARRY_CASES: Tuple[str, ...] = ("corpus_periodic", "wall_xyz")

#: Steps per carry / null-control leg.
CARRY_STEPS = 8

#: Steps per armed mutation.
MUTATION_STEPS = 4

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "signed_zero_lattice", "subnormal_band")

SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
)

REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

MATERIAL = ("eps", "inv_eps")

#: The volumes THIS SEAM writes. A leg in which none of them moves measured nothing
#: about this launch, whatever else moved.
SEAM_OUTPUTS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
)

#: Private ``Fields`` scratch that is NOT physical state. ``_fmp_scratch`` is the
#: buffer ``Fields.displacement_minus_polarization`` allocates on first use; its
#: presence differs by ROUTE while carrying no state either route reads across a
#: step. THE RULE IS THE LEADING UNDERSCORE, not this list.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(*parts: str) -> int:
    """A per-case seed from a DIGEST, never ``hash()``.

    ``PYTHONHASHSEED`` salts ``hash()`` of a tuple of strings, so a hash-seeded case
    cannot be replayed from its own record. A blake2b of the joined parts can.
    """
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % (2 ** 31 - 1)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input and "
            f"material_changed would never fire.")


def save(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    if "subnormal_policy" in payload:
        try:
            from meep_gpu import subnormal_policy as _policy  # noqa: PLC0415

            payload["subnormal_policy"] = _policy.policy_stamp()
        except Exception:  # noqa: BLE001 - a stamp failure must not lose the payload
            pass
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        _stamp_provenance(payload)
    except Exception as exc:  # noqa: BLE001 - a missing stamper must not lose the run
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
    """The exact bytes this gate binds itself to, KEYED BY REPO-RELATIVE PATH."""
    names = (
        "meep_gpu/triton_kernels/bfast_fused_electric_pair.py",
        "meep_gpu/triton_kernels/bfast_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/bfast_curl.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_bfast_fused_electric_pair.py",
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


def body_statements(path: str, name: str) -> List[str]:
    """One shipped kernel's BODY statements, docstring and signature removed.

    The text is EXACT — never ``ast.unparse``d. What is being compared includes the
    PARENTHESISATION, and unparsing re-derives minimal parentheses, which would
    silently equate ``dtdx * ((c_y - c) + (b - b_z))`` with a different grouping.
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
    segments = [ast.get_source_segment(text, statement) for statement in body]
    return _statements("\n".join(segment for segment in segments if segment))


def _clean(lines: Sequence[str]) -> str:
    """Comment-stripped, blank-stripped text. A prose edit is not a transcription."""
    kept = [line.split("#", 1)[0].rstrip() for line in lines]
    return "\n".join(line for line in kept if line.strip())


def body_segments(path: str, name: str) -> List[str]:
    """A shipped kernel's BODY as a list of top-level STATEMENT sources.

    THE GRAIN OF EVERY TRANSCRIPTION COMPARISON IN THIS GATE, and it is a per-LINE
    comparison that cannot express this product's claim: the BFAST tail contains
    LINES that also appear elsewhere in both bodies — ``if BCY == METALLIC:`` guards
    both the advance mask and the curl mask — so subtracting the tail by text leaves
    orphaned guard headers behind and the equality fails on a body that is correct.
    MEASURED on this gate 2026-08-31 before the grain was changed. The tail is
    exactly ONE ``ast.If`` node, and at THAT grain the subtraction is exact.

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
    """The kernel as source text, decorator included — the mutation harness's input.

    THE DECORATOR IS PREPENDED, and it is not cosmetic. ``ast.get_source_segment``
    on a ``FunctionDef`` EXCLUDES the decorator list, so a mutant compiled from the
    bare segment is a PLAIN PYTHON FUNCTION with no ``[grid]`` launcher and the first
    mutation leg dies with "'function' object is not subscriptable" — measured
    2026-08-20 on a sibling gate, and again 2026-08-31 on this one.
    """
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
    """The claim this whole product rests on, made mechanical.

    Two equalities, both EXACT, both in order and both at STATEMENT grain:

    * the kernel minus the ONE ``if HAS_BFAST:`` statement IS
      ``kernels.fused_curl_constitutive_D``;
    * the kernel minus the WELD (whatever ``fused_curl_constitutive_D`` has that
      ``kernels.pml_curl_step`` does not — DERIVED, not listed) IS
      ``bfast_curl.bfast_pml_curl_step``.

    Together they say: nothing in this body was written here. The rest states that
    positively, checks that the tail really is the shipped curl's own delta, and
    checks that this is the ELECTRIC weld rather than a copy of the magnetic twin.
    """
    kernels_py = os.path.join(PACKAGE_DIR, "kernels.py")
    bfast_py = os.path.join(PACKAGE_DIR, "bfast_curl.py")
    twin_py = os.path.join(PACKAGE_DIR, "bfast_fused_magnetic_pair.py")
    fused = body_segments(KERNEL_FILE, KERNEL_NAME)
    ordinary_fused = body_segments(kernels_py, "fused_curl_constitutive_D")
    ordinary_curl = body_segments(kernels_py, "pml_curl_step")
    bfast_curl_body = body_segments(bfast_py, "bfast_pml_curl_step")
    twin = body_segments(twin_py, "bfast_fused_curl_constitutive_B")

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
            f"minus the tail this is NOT fused_curl_constitutive_D; first "
            f"difference at statement {index}: "
            f"{[s[:80] for s in without_tail[index:index + 1]]} vs "
            f"{[s[:80] for s in ordinary_fused[index:index + 1]]}")

    weld = [s for s in ordinary_fused if s not in ordinary_curl]
    if not weld:
        findings.append("the weld is empty; the second equality compares nothing")
    without_weld = [s for s in fused if s not in weld]
    if without_weld != bfast_curl_body:
        index = next((i for i, (a, b) in enumerate(zip(without_weld,
                                                       bfast_curl_body))
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

    # THIS IS THE ELECTRIC WELD. A body copied from the magnetic twin would pass
    # every equality above against the WRONG pair of shipped sources.
    if fused == twin:
        findings.append("this kernel is byte-identical to the MAGNETIC twin; the D "
                        "seam has a different wall clear and an inv_eps multiply")
    joined = "\n".join(fused)
    joined_twin = "\n".join(twin)
    for component in range(3):
        line = (f"src{component} = v{component} * tl.load("
                f"ie{component} + idx, mask=live, other=0.0)")
        if line not in joined:
            findings.append(f"the E side's inverse-permittivity multiply is missing "
                            f"for component {component}: {line!r}")
        if line in joined_twin:
            findings.append(f"the MAGNETIC twin carries {line!r}; update_H reads no "
                            f"inverse epsilon and the comparison is not what it says")
    if "v2 = tl.where(at_x, 0.0, v2)" not in joined:
        findings.append("zero_metal_D's second tangential row on the x wall is "
                        "missing; the D family clears TWO components per axis")
    if "f_bfast_B" in joined:
        findings.append("this body names a MAGNETIC BFAST state array; the D seam's "
                        "states are f_bfast_D*")

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
    thickness = keywords.pop("thickness", None)
    pml_thickness = keywords.pop("pml_thickness", 2)
    shear = keywords.pop("bfast_scaled_k", BFAST_K)
    grid = Grid(resolution=10.0, cell_size=keywords.pop("cell_size", (0.8, 0.8, 0.8)),
                dimensions=3, courant=0.35, bfast_scaled_k=shear,
                boundaries=keywords.pop("boundaries", None),
                k_point=keywords.pop("k_point", (0.0, 0.0, 0.0)), **keywords)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = np.random.default_rng(17)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez",
                 "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(
                -0.4, 0.4, size=grid.shape).astype(np.float32)
    if thickness is None:
        thickness = tuple(
            (pml_thickness, pml_thickness) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes."""

    field_type = "D"

    def __init__(self, index=(1, 1, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


class _ElectricWithoutIndex:
    field_type = "D"


#: One-clause perturbations off the corpus family. Every one must leave the weld no
#: wider than the narrower of its two halves.
EQUIVALENCE_CASES: Tuple[Tuple[str, Dict[str, Any], Any], ...] = (
    ("corpus", {}, (_Electric(),)),
    ("magnetic_source", {}, (_Magnetic(),)),
    ("undeclared_sources", {}, None),
    ("electric_without_an_index", {}, (_ElectricWithoutIndex(),)),
    ("shear_free", {"bfast_scaled_k": (0.0, 0.0, 0.0)}, (_Electric(),)),
    ("complex_storage", {"complex_storage": True}, (_Electric(),)),
    ("inactive_absorber", {"pml_thickness": 0}, (_Electric(),)),
    ("walled", {"boundaries": {"x": "metallic", "y": "metallic",
                               "z": "metallic"}}, (_Electric(),)),
    ("bloch", {"k_point": (0.2, 0.0, 0.0)}, (_Electric(),)),
)


def equivalence_leg() -> Dict[str, Any]:
    """``weld => spine curl AND spine E``, measured over one-clause perturbations."""
    from meep_gpu.grid import Mirror  # noqa: PLC0415
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)
    cases = list(EQUIVALENCE_CASES) + [
        # A folded axis cannot carry a LOW-face layer: cell 0 there is the mirror
        # plane and pml._resolve_mirror_faces RAISES on a per-side request naming it.
        ("folded", {"symmetry": (Mirror("Y", 1),),
                    "thickness": ((2, 2), (0, 2), (2, 2))}, (_Electric(),)),
    ]
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    admitted = 0
    for name, keywords, sources in cases:
        fields, pml = _laptop_fixture(**dict(keywords))
        weld = product.bfast_fused_electric_pair_coverage(fields, pml, sources)
        curl = bfast_curl.bfast_pml_curl_coverage(fields, pml, "step_D")
        electric = bfast_curl.bfast_run_constitutive_coverage(
            fields, pml, "E")
        never_wider = (not weld.covered) or (curl.covered and electric.covered)
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

    # THE FLAG IS THE PRODUCT, and this leg says so by measurement. With
    # CARRIES_DEPOSIT_REPAIR at False the corpus row's own electric source is
    # refused outright and the cell is worth ZERO.
    fields, pml = _laptop_fixture()
    before = [r for r in product.bfast_fused_electric_pair_coverage(
        fields, pml, (_Electric(),)).reasons if "array module" not in r]
    original = product.CARRIES_DEPOSIT_REPAIR
    try:
        product.CARRIES_DEPOSIT_REPAIR = False
        after = [r for r in product.bfast_fused_electric_pair_coverage(
            fields, pml, (_Electric(),)).reasons if "array module" not in r]
    finally:
        product.CARRIES_DEPOSIT_REPAIR = original
    if before:
        findings.append(f"the corpus family is refused WITH the flag: {before}")
    if not any("is electric" in reason for reason in after):
        findings.append(
            "flipping CARRIES_DEPOSIT_REPAIR to False did NOT refuse the electric "
            "source: the flag is not what carries this cell and the whole product's "
            "premise is wrong")

    return {"leg": "equivalence", "device": False, "cases": rows,
            "admitted_modulo_backend": admitted,
            "reasons_with_the_flag": before,
            "reasons_with_the_flag_off": after,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — what the corpus says, re-derived from the census
# ---------------------------------------------------------------------------

CENSUS = os.path.join(HERE, "results",
                      "predicate_coverage_triton_2026-08-31_folded_disp")
MATRIX = os.path.join(HERE, "results",
                      "fusion_matrix_triton_2026-08-31_folded_disp",
                      "fusion_matrix.json")
CELL = ("D->E", "BFAST PML", "BFAST run")


def corpus_admission_leg() -> Dict[str, Any]:
    """The rows this arm can serve, counted rather than cited.

    Read from the census's own configuration block: a row is in this cell when it
    declares a LIVE SHEAR, an active absorber, real storage, no fold, no
    cylindrical axis, no beta, no nonlinearity and no off-diagonal epsilon (the
    board's own cell is ``(BFAST PML, BFAST run)`` at ``D->E``).
    It is REACHABLE when its in-seam ELECTRIC source is one the repair can carry,
    which is the board's own ``in_seam_source_blocks`` and is read from there.

    AN ABSENT CENSUS IS NOT AN EMPTY ONE: a missing artifact is a REFUSAL here,
    never a funnel of zero reported as a pass.
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
    matched_path = os.path.join(CENSUS, "tests_param_matched.jsonl")
    if os.path.exists(matched_path):
        matched = {(r.get("leg"), r.get("row")): r for r in (
            json.loads(line) for line in
            open(matched_path, encoding="utf-8").read().splitlines() if line.strip())}
        record = [matched.get((r.get("leg"), r.get("row")), r) for r in record]
    record = [r for r in record if r.get("measured")]

    in_cell: List[str] = []
    electric: List[str] = []
    for row in record:
        configuration = row.get("configuration") or {}
        if not configuration.get("bfast_active"):
            continue
        if not configuration.get("pml_active"):
            continue
        if configuration.get("force_complex_fields"):
            continue
        if configuration.get("has_symmetry") or any(
                configuration.get("mirrored") or ()):
            continue
        if configuration.get("cylindrical"):
            continue
        if float(configuration.get("beta") or 0.0) != 0.0:
            continue
        if configuration.get("has_nonlinearity"):
            continue
        if configuration.get("has_offdiagonal_epsilon"):
            continue
        name = f"{row['leg']}:{row['row']}"
        in_cell.append(name)
        if "D" in tuple(configuration.get("source_field_types") or ()):
            electric.append(name)

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
                f"the board says the deposit repair CANNOT carry {blocked}; this "
                f"product could not serve them")
        for entry in entries:
            if entry["product"] is not None:
                findings.append(
                    f"the board already names {entry['product']!r} in this cell")
    else:
        findings.append(f"the fusion matrix is absent: {MATRIX}")
    if not in_cell:
        findings.append("no row in the cell: this arm would be worth nothing")
    if not electric:
        findings.append(
            "no row in the cell declares an ELECTRIC source, so the deposit carry "
            "this gate exists to measure is not what makes the cell reachable")
    return {"leg": "corpus_admission", "device": False, "rows_scanned": len(record),
            "cell": list(CELL), "rows_in_the_cell": sorted(in_cell),
            "rows_declaring_an_electric_source": sorted(electric),
            "rows_the_repair_cannot_carry": blocked,
            "matrix_rows_at_the_same_cell": matrix_rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — every mutation is armed and scored on a live branch
# ---------------------------------------------------------------------------

def mutation_arming_leg() -> Dict[str, Any]:
    """A rewrite that matches nothing measured nothing, whatever it then reported."""
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

def seed_values(cp, driver, seed: int, value_class: str) -> None:
    """Seed every settable volume in ONE value class, identically on every route."""
    rng = np.random.default_rng(seed)
    shape = tuple(driver.shape)

    def draw() -> np.ndarray:
        if value_class == "uniform":
            values = rng.uniform(-0.25, 0.25, size=shape)
        elif value_class == "signed_zero_lattice":
            # +0.0 and -0.0 on a random lattice. The two words differ (0x00000000 vs
            # 0x80000000), so a byte gate SEES the difference — this is what
            # exercises every `tl.where(flag, 0.0, v)` and every subtraction's sign
            # convention, INCLUDING the BFAST tail's `curl - adv` and its
            # `total - (2.0 * state)`.
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
    # single-launch leg non-vacuous: without a nonzero H the D curl AND the BFAST
    # sums are both exactly zero on step 1.
    # THE BFAST IIR STATES ARE SEEDED, and that is not thoroughness: with
    # `state == 0` the tail's zero-k advance `total - 2*state` loses its second term
    # and every mutation of the state store becomes a null. `bfast_pml_curl_step`'s
    # own docstring names the same requirement.
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                 "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
                 "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())


def build_driver(cp, cell, boundaries, seed: int, value_class: str,
                 electric: bool, bfast_scaled_k=BFAST_K):
    """One 3-D BFAST PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=3,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        bfast_scaled_k=tuple(float(v) for v in bfast_scaled_k),
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, deliberately, and a DIFFERENT one per component.
    #
    # BOTH HALVES OF THAT ARE MEASUREMENTS. A uniform material makes a mis-indexed
    # coefficient a scaling many wrong transcriptions reproduce; and a material that
    # varies in SPACE but not in COMPONENT makes `ie0` and `ie1` the same volume, so
    # `src0 = v0 * ie1` is a bitwise no-op — measured 2026-08-31, when
    # `m_inv_eps_component_swapped` reported a real defect as UNCAUGHT against a
    # single `set_epsilon` volume.
    epsilon = {
        name: np.ascontiguousarray(
            (2.25 + 0.30 * np.sin(index * np.float32(0.037 + 0.011 * offset))
             + 0.17 * offset).astype(np.float32))
        for offset, name in enumerate(("Ex", "Ey", "Ez"))}
    driver.set_epsilon_components({name: cp.asarray(values)
                                   for name, values in epsilon.items()})
    # ALL THREE AXES: this cell is 3-D, so none of them is invariant and each one
    # has a face to absorb at.
    driver.setup_pml({"x": 2, "y": 2, "z": 2})
    if electric:
        # THE DEPOSIT THIS PRODUCT EXISTS FOR. Injected BETWEEN the two halves
        # (driver.py:3294-3299), so the fused launch consumes a pre-injection D and
        # the shipped repair is what puts the difference back.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    else:
        # A MAGNETIC source is admitted with no repair at all: the driver injects it
        # in the B/H seam, not this one. Carrying one is what stops the source clause
        # from being tested only in its refusing direction.
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    # THE FAST PATH IS DISABLED, EXPLICITLY. `FdtdDriver.step` consults `fastpath`
    # BEFORE the module-level sub-step functions this harness substitutes, and the
    # `BFAST PML` / `BFAST run` arms are WIRED — so a driver left alone would
    # step `step_D` and `update_E` on Triton and the "array path" reference would be
    # no such thing. The per-route call counter re-checks that this held.
    driver.invalidate_fast_path()
    driver._fast_path = None          # noqa: SLF001 - the harness owns the route
    driver._fast_path_stale = False   # noqa: SLF001
    return driver


def inventory(driver) -> Dict[str, Any]:
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        # THE LEADING UNDERSCORE IS THE RULE — see PRIVATE_SCRATCH.
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
    ``BFAST PML`` on ``step_D`` and ``BFAST run`` on ``update_E`` — so this
    route is not a second model of the seam, it is the shipped one.

    The three in-seam passes stay on the ARRAY PATH here: no Triton product owns the
    wall clear, and both symmetry fills are no-ops without a mirror plane. All three
    are COUNTED. THE INJECTION ALSO STAYS ON THE ARRAY PATH, which is what makes
    this the right oracle for the carry family: the certified kernels with the
    driver's own deposit between them.
    """
    from meep_gpu.triton_kernels import bfast_curl  # noqa: PLC0415

    curl = bfast_curl.plan_bfast_pml_curl(driver.fields, driver.pml, "step_D")
    constitutive = bfast_curl.plan_bfast_run_constitutive(
        driver.fields, driver.pml, "E")
    missing = [name for name, plan in
               (("BFAST PML curl", curl),
                ("BFAST run constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the certified "
            f"products it replaces")
    return Route({"step_D": curl, "update_E": constitutive})


def fused_route(plan, driver=None, sources=(), bracket: bool = True):
    """The fused plan in its two slots, BRACKETED when the seam carries a deposit.

    THE BRACKET IS THE SHIPPED ONE. ``deposit_repair.LeadingRepairPlan`` /
    ``TrailingRepairPlan`` are the exact pair ``launch._install_fused_pair`` puts in
    these two slots; a harness that assembled its own could not license the one that
    ships.

    ``bracket=False`` is the NULL CONTROL and it must diverge: it is the same launch
    with the repair removed, which is the configuration the whole flag exists to
    forbid.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "D")
    if not seam or not bracket:
        return Route({name: (plan if name == "step_D" else _Absorbed(name, plan))
                      for name in SEAM_PASSES}), None
    leading = deposit_repair.LeadingRepairPlan(plan, driver.fields, driver.pml,
                                               seam, "D")
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, driver.fields,
                                                 driver.pml)
    plans = {name: _Absorbed(name, plan) for name in SEAM_PASSES}
    plans["step_D"] = leading
    plans["update_E"] = trailing
    return Route(plans), leading


def _swapped_coefficients(driver, plan, group: str):
    """The plan's coefficient bindings with ONE group moved to the WRONG lattice.

    ``SUB_STEPS['step_D']`` carries no suffix so the curl half reads the INTEGER
    split-field pair, while ``CONSTITUTIVE_SIDES['E']['half_integer'] is True`` so
    the constitutive half reads the HALF-INTEGER one. The kernel takes twelve
    pointers and never asks which lattice they came from, so a swap is a half-cell
    error in the absorber profile: converged, smooth and wrong.
    """
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

    for slot in ("_curl_coefficients", "_e_coefficients"):
        if not hasattr(plan, slot):
            raise AssertionError(
                f"the plan has no {slot}; this host mutation would bind nothing and "
                f"the leg would score a defect it never armed")
    pml = driver.pml
    if group == "curl":
        arrays = [getattr(pml, f"{stem}_{axis}_h")          # HALF-INTEGER, wrongly
                  for axis in "xyz" for stem in ("kms", "sinv")]
        plan._curl_coefficients = tuple(  # noqa: SLF001 - the armed seam
            CupyPointer(_flat(a)) for a in arrays)
    elif group == "constitutive":
        arrays = [getattr(pml, f"{stem}_{axis}")            # INTEGER, wrongly
                  for axis in "xyz" for stem in ("kps", "kms")]
        plan._e_coefficients = tuple(  # noqa: SLF001
            CupyPointer(_flat(a)) for a in arrays)
    else:  # pragma: no cover
        raise ValueError(group)
    return plan


def host_mutation_table() -> Tuple[Tuple[str, str, str,
                                         Callable[[Any, Any], Any]], ...]:
    """The HOST mutations: plan-level choices, not kernel text."""
    return (
        ("h_curl_lattice_swapped",
         "the curl half bound the HALF-INTEGER split-field pair, which is the "
         "constitutive side's lattice",
         "caught", lambda driver, plan: _swapped_coefficients(driver, plan, "curl")),
        ("h_constitutive_lattice_swapped",
         "the constitutive half bound the INTEGER kps/kms pair, which is the curl "
         "side's lattice",
         "caught",
         lambda driver, plan: _swapped_coefficients(driver, plan, "constitutive")),
        ("h_k_words_rotated",
         "the six k1/k2 words rotated by one target; the shear's three components "
         "differ, so every target then reads its neighbour's pair",
         "caught", lambda driver, plan: _rotate_k_words(plan)),
        ("h_k_words_taken_from_the_magnetic_seam",
         "bfast_curl_coefficients called with magnetic=True, which reads "
         "BFAST_TERMS['step_B'] and does NOT negate — the copied-call defect the "
         "beta family is immune to and this one is not",
         "caught", lambda driver, plan: _magnetic_k_words(driver, plan)),
    )


def _rotate_k_words(plan):
    """The six (k1, k2) scalars rotated by one target.

    A rotation rather than a swap: with three DIFFERENT shear components every
    target then reads a neighbour's pair, so no target keeps its own words and the
    defect cannot cancel.
    """
    ks = tuple(plan.ks)
    plan.ks = ks[2:] + ks[:2]
    return plan


def _magnetic_k_words(driver, plan):
    """The words ``bfast_curl_coefficients`` returns for the MAGNETIC seam.

    THE COPIED-CALL DEFECT, planted at the plan layer. The D side reads
    ``BFAST_TERMS['step_D']`` and NEGATES both coefficients in host float64
    (bfast_curl.py:343-351); a builder that passed its twin's ``magnetic=True``
    would hand this product six wrong words, and no source diff of the KERNEL would
    show it. The beta family is immune to the same copy — its flag is inert on real
    storage — which is why this mutation exists here and not there.
    """
    from meep_gpu.triton_kernels.bfast_curl import (  # noqa: PLC0415
        bfast_curl_coefficients,
    )

    grid = driver.fields.grid
    invariant = tuple(grid.is_invariant(axis) for axis in range(3))
    magnetic = bfast_curl_coefficients(grid.bfast_scaled_k, invariant,
                                       magnetic=True)
    if tuple(magnetic) == tuple(plan.ks):
        raise AssertionError(
            "the magnetic and electric k words are EQUAL on this grid, so this "
            "mutation plants nothing; bfast_curl_coefficients negates the D side "
            "(bfast_curl.py:350-351) and a grid where that is invisible cannot "
            "score it")
    plan.ks = tuple(magnetic)
    return plan


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, steps: int, product,
            value_class: str = "uniform", mutant: Any = None,
            host_mutation: Any = None, install_fused: bool = True,
            freeze_electric: bool = False, electric: bool = False,
            bracket: bool = True) -> Dict[str, Any]:
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
        else product.bfast_fused_curl_constitutive_D_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "value_class": value_class, "seed": seed,
        "electric_source": bool(electric), "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "host_mutation": None, "first_divergence": None,
        "control_divergence": None,
    }
    try:
        plan = product.plan_bfast_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), kernel=kernel)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.bfast_fused_electric_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if host_mutation is not None:
            row["host_mutation"] = host_mutation[0]
            plan = host_mutation[3](fused, plan)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["zero_metal_flags"] = list(plan.zero_metal)
        row["k_words"] = [list(plan.ks), plan.has_bfast]
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
        if freeze_electric:
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
        # THE FLOOR IS RELATIVE TO THE ARRAY PATH, and that is a 2026-08-31
        # correction rather than a softening. An absolute "every array moved" floor
        # fails a CORRECT kernel for properties of the CONFIGURATION: measured on
        # this gate's single-launch leg, `fu_By` does not move over ONE complete
        # step, and it is a B-seam auxiliary this product's launch never touches on
        # EITHER route — so its stillness is evidence about the grid, not about the
        # weld. What IS evidence is an array the ARRAY PATH moves and this route does
        # not: that is a pass this weld swallowed. The reference driver's own census
        # is taken and the two are compared, and the POSITIVE floor below keeps the
        # trivial-agreement case refused.
        row["reference_never_moved"] = sorted(
            set(reference_final) - set(reference_moved) - set(material))
        row["inert_here_but_moving_on_the_array_path"] = sorted(
            (set(final) - set(ever_moved) - set(material)) & set(reference_moved))
        row["seam_outputs_moved"] = sorted(set(ever_moved) & set(SEAM_OUTPUTS))
        row["reference_seam_outputs_moved"] = sorted(
            set(reference_moved) & set(SEAM_OUTPUTS))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(reference_final))
        row["private_scratch_on_the_array_route"] = sorted(
            name for name in vars(reference.fields) if name in PRIVATE_SCRATCH)
        row["private_scratch_on_the_fused_route"] = sorted(
            name for name in vars(fused.fields) if name in PRIVATE_SCRATCH)
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
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_launches_at_least: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False,
               require_array_reference: bool = True) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if not require_identical and row.get("first_divergence") is None:
        failures.append(
            "this leg REQUIRES divergence and found none: the control it exists to "
            "be is inert, and the thing it was meant to prove load-bearing is not")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path, so this leg "
            f"could not have measured the fused launch: {row['control_divergence']}")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"expected {require_launches}: the fused path is not what executed")
    if (require_launches_at_least is not None
            and (row.get("fused_kernel_launches") or 0) < require_launches_at_least):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"fewer than the {require_launches_at_least} this leg needs")
    if require_moved:
        swallowed = row.get("inert_here_but_moving_on_the_array_path") or []
        if swallowed:
            failures.append(
                f"the fused route left {swallowed} INERT while the array path moves "
                f"them: a pass this weld was supposed to carry did not run")
        reference_movers = set(row.get("reference_seam_outputs_moved") or ())
        if not reference_movers:
            failures.append(
                "VACUOUS: the ARRAY PATH itself moves nothing this seam writes on "
                "this configuration, so agreement with it is agreement about nothing")
        movers = set(row.get("seam_outputs_moved") or ())
        if movers != reference_movers:
            failures.append(
                f"the fused route's seam-output movement {sorted(movers)} is not the "
                f"array path's {sorted(reference_movers)}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    if row.get("inventory_asymmetry_vs_array"):
        failures.append(
            f"the compared inventories differ by NAME between the fused route and "
            f"the array path: {row['inventory_asymmetry_vs_array']}")
    if require_repairs and not row.get("deposit_repairs"):
        failures.append(
            "the carry leg repaired NO deposit cell: the bracket ran but the seam "
            "carried nothing, so this leg measured the quiet case under a carry name")
    counts = row.get("launches") or {}
    fell_back = {key: value for key, value in counts.items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: "
            f"{fell_back}")
    if require_array_reference:
        # THE REFERENCE MUST REALLY BE THE ARRAY PATH. `fastpath` is consulted before
        # these functions and the BFAST arms are WIRED, so a missing count here means
        # the reference stepped on Triton and the comparison compared two Triton
        # routes.
        steps = len(row.get("per_step") or ())
        for name in SEAM_PASSES:
            seen = counts.get(f"array/array_path:{name}", 0)
            if seen != steps:
                failures.append(
                    f"the reference route reached the array path for {name} {seen} "
                    f"times, expected {steps}: the fast path was not off")
    return (not failures), failures


# ---------------------------------------------------------------------------
# THE REDUCTION LEG — HAS_BFAST = 0 must BE the shipped ordinary kernel
# ---------------------------------------------------------------------------

def reduction_leg(cp, product, cell, boundaries, steps: int,
                  value_class: str = "uniform") -> Dict[str, Any]:
    """``HAS_BFAST = 0`` reproduces ``kernels.fused_curl_constitutive_D`` bit for bit.

    The transcription claim, measured on silicon instead of in source text. Three
    drivers, seeded identically, on a live-shear run:

    * route A launches THIS product's kernel with ``has_bfast = 0``;
    * route B launches the SHIPPED ``kernels.fused_curl_constitutive_D`` over the
      same volumes, through ``launch.plan_fused_pair_from_arrays`` — the bare-array
      route, because ``coverage.fused_pair_coverage`` refuses a BFAST run by name
      (clause 11) and would return ``None``;
    * route C is the array path, and is the DISCRIMINATION precondition.

    NEITHER A NOR B IS COMPARED AGAINST THE ARRAY PATH FOR IDENTITY, deliberately:
    the array path really does add the BFAST advance, so both routes must DIFFER
    from it.
    What this leg measures is that the two KERNELS agree, which is the reduction.

    THE NON-VACUITY FLOOR IS THE OTHER HALF. Two kernels that both did nothing would
    also agree, so the state must MOVE and route A must additionally be shown to
    DIFFER from the array path — otherwise ``has_bfast = 0`` would be
    indistinguishable from the shipped BFAST kernel and this leg would be measuring
    a flag nothing reads.

    THE STATE ARRAYS ARE THE DISCRIMINATION'S OWN LEVER. At ``HAS_BFAST = 0`` the
    tail is gone, so ``f_bfast_D*`` is never written and the array path's is; the
    seeded non-zero state is what makes that difference a WORD rather than a pair of
    zeros.

    NO ELECTRIC SOURCE HERE. Both routes are unbracketed, so a deposit in the seam
    would put both of them one injection behind the array path and the discrimination
    precondition would fire for the wrong reason. The MAGNETIC source keeps the
    fields alive without touching this seam.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import (  # noqa: PLC0415
        plan_fused_pair_from_arrays,
    )

    seed = case_seed("reduction", str(cell), str(boundaries), value_class)
    made = [build_driver(cp, cell, boundaries, seed, value_class, electric=False)
            for _ in range(3)]
    reference, shipped, reduced = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    row: Dict[str, Any] = {"leg": "reduction:has_bfast_zero", "device": True,
                           "steps_budget": steps, "value_class": value_class,
                           "seed": seed, "shape": list(reference.shape),
                           "boundaries": boundaries}
    try:
        from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
        from meep_gpu.stepping import _boundary_kinds as resolve  # noqa: PLC0415

        reduced_plan = product.plan_bfast_fused_electric_pair(
            reduced.fields, reduced.pml, tuple(reduced._sources))
        if reduced_plan is None:
            raise AssertionError("the product refused the reduction case")
        reduced_plan.has_bfast = 0

        fields, pml = shipped.fields, shipped.pml
        grid = fields.grid
        arrays = {name: getattr(fields, name) for name in
                  ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
                   "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
                   "f_w_Ex", "f_w_Ey", "f_w_Ez")}
        arrays.update({f"inv_eps_{name}": fields.inverse_epsilon_for(name)
                       for name in ("Ex", "Ey", "Ez")})
        curl_flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}")
                     for axis in "xyz" for stem in ("kms", "sinv")}
        constitutive_flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}_h")
                             for axis in "xyz" for stem in ("kps", "kms")}
        shipped_plan = plan_fused_pair_from_arrays(
            "D", arrays, curl_flat, constitutive_flat,
            [1 if kind == "metallic" else 0 for kind in resolve(grid, pml)],
            zero_metal_axes(grid), grid.dt / grid.dx, num_warps=1)

        roles = {id(reference.fields): "array", id(shipped.fields): "shipped",
                 id(reduced.fields): "reduced"}
        routes = [(shipped.fields, fused_route(shipped_plan)[0]),
                  (reduced.fields, fused_route(reduced_plan)[0])]
        undo = install(driver_module, routes, counter, roles)

        opening = snapshot(cp, reduced)
        reference_opening = snapshot(cp, reference)
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
        reference_moved = moved(reference_opening, snapshot(cp, reference))
        material = [key for key in final if key in MATERIAL]
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))

        # THE BFAST STATE IS THIS LEG'S LEVER, NOT ITS FLOOR, and that distinction is
        # a 2026-08-31 correction. The docstring above says it outright -- "at
        # HAS_BFAST = 0 the tail is gone, so `f_bfast_D*` is never written" -- so an
        # absolute "every array moved" floor asserts the exact opposite of what the
        # leg exists to measure and cannot be satisfied by a CORRECT kernel. Run c
        # measured that: three legs failed with `VACUOUS: ['f_bfast_Dx',
        # 'f_bfast_Dy', 'f_bfast_Dz'] never moved` while every identity comparison
        # in them passed.
        #
        # The replacement is TWO-SIDED and strictly stronger than the absolute floor
        # was for every other array, because it also pins the direction:
        #
        #   * the state must be INERT on the reduced route     -- that IS the reduction;
        #   * the state must MOVE on the ARRAY PATH            -- without this the two
        #     routes differ by a pair of zeros and `discriminating` is worthless;
        #   * every OTHER array the array path moves must move here too -- a pass this
        #     weld swallowed is still a failure, which is what the old floor was for.
        #
        # The names come from the PRODUCT, so renaming the state without renaming it
        # here turns the assertion red rather than vacuous.
        bfast_state = tuple(product.BFAST_STATE_NAMES["step_D"])
        row["bfast_state_names"] = list(bfast_state)
        row["bfast_state_moved_here"] = sorted(set(ever_moved) & set(bfast_state))
        row["bfast_state_moved_on_the_array_path"] = sorted(
            set(reference_moved) & set(bfast_state))
        row["inert_here_but_moving_on_the_array_path"] = sorted(
            ((set(final) - set(ever_moved) - set(material)) & set(reference_moved))
            - set(bfast_state))
        row["seam_outputs_moved"] = sorted(set(ever_moved) & set(SEAM_OUTPUTS))
        row["reference_seam_outputs_moved"] = sorted(
            set(reference_moved) & set(SEAM_OUTPUTS))
        row["launches"] = dict(counter)
        failures: List[str] = []
        if row["first_divergence"] is not None:
            failures.append(
                f"HAS_BFAST=0 is NOT the shipped fused pair: {row['first_divergence']}")
        if not bfast_state:
            failures.append(
                "the product declares NO BFAST state for `step_D`, so this leg has "
                "no lever and its reduction claim is untestable")
        if row["bfast_state_moved_here"]:
            failures.append(
                f"HAS_BFAST=0 still wrote {row['bfast_state_moved_here']}: the tail "
                f"this flag is supposed to remove ran anyway")
        if set(row["bfast_state_moved_on_the_array_path"]) != set(bfast_state):
            failures.append(
                f"VACUOUS: the ARRAY PATH moved only "
                f"{row['bfast_state_moved_on_the_array_path']} of {list(bfast_state)}, "
                f"so the reduced route's stillness is not a measurable difference")
        if row["inert_here_but_moving_on_the_array_path"]:
            failures.append(
                f"the reduced route left "
                f"{row['inert_here_but_moving_on_the_array_path']} INERT while the "
                f"array path moves them: a pass this weld swallowed")
        if not row["reference_seam_outputs_moved"]:
            failures.append(
                "VACUOUS: the ARRAY PATH itself moves nothing this seam writes on "
                "this configuration")
        if (set(row["seam_outputs_moved"])
                != set(row["reference_seam_outputs_moved"]) - set(bfast_state)):
            failures.append(
                f"the reduced route's seam-output movement {row['seam_outputs_moved']}"
                f" is not the array path's "
                f"{row['reference_seam_outputs_moved']} less the BFAST state")
        # THE DISCRIMINATION PRECONDITION, recorded rather than assumed. If
        # HAS_BFAST=0 also matches the ARRAY PATH then this value class cannot tell
        # the two kernels apart at all and its agreement is a tautology. That is a
        # PROPERTY OF THE VALUE CLASS, not a defect, so it is reported as
        # `discriminating: false` with its reason and the gate-level floor demands
        # that at least one class discriminate.
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
    # A MUTATION THAT DOES NOT PARSE IS NOT A MUTATION. Checked before any device
    # work, so a broken rewrite is reported as unarmed rather than as caught.
    ast.parse(text)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_bfast_electric_pair.py", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_bfast_electric_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace a multi-line fragment, matching on the SIGNIFICANT statements only.

    BLANK LINES AND COMMENTS ARE SKIPPED WHEN MATCHING and dropped from the span
    that is replaced. Without that, a needle spanning two regions of the shipped
    body — the wall clear and the stores after it, say — matches nothing because a
    blank line or an explanatory comment sits between them, and the mutation is
    reported UNARMED while the harness believes it planted a defect. Measured on
    this gate 2026-08-31: two of its fourteen rewrites hit zero lines for exactly
    that reason.
    """
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
            # THE REPLACEMENT KEEPS ITS OWN RELATIVE INDENTATION and is re-prefixed
            # with the matched span's base. Stripping every line to the base instead
            # flattens a nested block, and `if ZM_X:` with its body pulled out to the
            # same column is a mutant that does not PARSE — reported as "armed" by
            # the hit count and then as "did not compile" by the device leg, which is
            # a mutation that measured nothing either way.
            lines[first:last + 1] = [
                (indent + fragment) if fragment.strip() else fragment
                for fragment in after]
            hits += 1
            # The line table moved; re-derive it and restart past the replacement.
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

    def m_tail_dropped(source: str) -> Tuple[str, int]:
        """No BFAST advance at all — the ordinary fused pair on a shear run."""
        return _replace_all(source, "if HAS_BFAST:", "if HAS_BFAST and False:")

    def m_tail_sums_become_differences(source: str) -> Tuple[str, int]:
        """``k1*(g1s - g1c)`` instead of ``k1*(g1s + g1c)``.

        BFAST SUMS the same shifted/center pairs the curl DIFFERENCES
        (stepping.py:1594-1598); a body that reused the curl's operator would be a
        smooth, plausible, wrong field.
        """
        return _rewrite_block(
            source,
            ["total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b))"],
            ["total0 = (k1_0 * (c_y - c)) - (k2_0 * (b_z + b))"])

    def m_tail_k_pair_swapped(source: str) -> Tuple[str, int]:
        """``k1`` and ``k2`` exchanged on target 0.

        ``k1`` multiplies the FIRST partner's sum and ``k2`` the SECOND's
        (stepping.py:911-912, bfast_curl.py:348-349); the two are different entries
        of the shear vector, so the swap is a real defect wherever they differ —
        which the case table's three-component shear guarantees.
        """
        return _rewrite_block(
            source,
            ["total0 = (k1_0 * (c_y + c)) - (k2_0 * (b_z + b))"],
            ["total0 = (k2_0 * (c_y + c)) - (k1_0 * (b_z + b))"])

    def m_advance_state_factor(source: str) -> Tuple[str, int]:
        """``total - state`` instead of ``total - 2.0*state`` (S:900)."""
        return _replace_all(source, "adv0 = total0 - (2.0 * st0)",
                            "adv0 = total0 - st0")

    def m_advance_sign_flipped(source: str) -> Tuple[str, int]:
        """``curl + adv`` instead of ``curl - adv`` — the caller-subtracts sign
        convention (S:904) inverted."""
        return _replace_all(source, "curl0 = curl0 - adv0", "curl0 = curl0 + adv0")

    def m_state_store_dropped(source: str) -> Tuple[str, int]:
        """``state`` never advanced. The IIR history then never moves, which is a
        state defect the NEXT step's advance reads — and one no single-step
        comparison of the field alone would see."""
        return _replace_all(source, "tl.store(s0 + idx, st0 + adv0, mask=live)",
                            "tl.store(s0 + idx, st0, mask=live)")

    def m_advance_mask_dropped(source: str) -> Tuple[str, int]:
        """The advance not masked at all — a non-owned cell keeps its contribution
        in BOTH the state and the curl."""
        return _rewrite_block(
            source,
            ["if BCY == METALLIC:",
             "adv0 = tl.where(at_y, 0.0, adv0)"],
            ["if BCY == METALLIC:",
             "    adv0 = adv0"])

    def m_advance_mask_after_the_state_store(source: str) -> Tuple[str, int]:
        """The advance masked AFTER the state store instead of before it (S:902).

        The curl then sees the same masked value it should, and only the STATE
        differs — which is exactly why the comparison is over EVERY allocated volume
        rather than over the fields.
        """
        return _rewrite_block(
            source,
            ["tl.store(s0 + idx, st0 + adv0, mask=live)",
             "tl.store(s1 + idx, st1 + adv1, mask=live)",
             "tl.store(s2 + idx, st2 + adv2, mask=live)"],
            ["tl.store(s0 + idx, st0 + total0 - (2.0 * st0), mask=live)",
             "tl.store(s1 + idx, st1 + adv1, mask=live)",
             "tl.store(s2 + idx, st2 + adv2, mask=live)"])

    def m_tail_scaled_by_dtdx(source: str) -> Tuple[str, int]:
        """The advance multiplied by ``dtdx``; the tail carries none (S:836-837)."""
        return _replace_all(source, "curl1 = curl1 - adv1",
                            "curl1 = curl1 - (dtdx * adv1)")

    def m_state_component_swapped(source: str) -> Tuple[str, int]:
        """Target 0 advances target 1's state array."""
        return _replace_all(source, "st0 = tl.load(s0 + idx, mask=live, other=0.0)",
                            "st0 = tl.load(s1 + idx, mask=live, other=0.0)")

    def m_wall_clear_dropped(source: str) -> Tuple[str, int]:
        """``zero_metal_D`` not carried on the x wall."""
        return _replace_all(source, "if ZM_X:", "if ZM_X and False:")

    def m_wall_clear_is_the_magnetic_familys_one_row(source: str) -> Tuple[str, int]:
        """The magnetic twin's ONE row on the D family, which needs TWO per axis."""
        return _rewrite_block(
            source,
            ["if ZM_X:",
             "v1 = tl.where(at_x, 0.0, v1)",
             "v2 = tl.where(at_x, 0.0, v2)"],
            ["if ZM_X:",
             "    v0 = tl.where(at_x, 0.0, v0)"])

    def m_wall_clear_after_the_store(source: str) -> Tuple[str, int]:
        """The clear applied after the D store, so the stored D keeps the wall value.

        ``update_E`` would still read the cleared register, so this is a defect in
        the D volume alone — which is exactly why the comparison is over EVERY
        allocated volume rather than over E.
        """
        return _rewrite_block(
            source,
            ["if ZM_X:",
             "v1 = tl.where(at_x, 0.0, v1)",
             "v2 = tl.where(at_x, 0.0, v2)",
             "if ZM_Y:",
             "v0 = tl.where(at_y, 0.0, v0)",
             "v2 = tl.where(at_y, 0.0, v2)",
             "if ZM_Z:",
             "v0 = tl.where(at_z, 0.0, v0)",
             "v1 = tl.where(at_z, 0.0, v1)",
             "tl.store(u0 + idx, n0, mask=live)",
             "tl.store(u1 + idx, n1, mask=live)",
             "tl.store(u2 + idx, n2, mask=live)",
             "tl.store(f0 + idx, v0, mask=live)",
             "tl.store(f1 + idx, v1, mask=live)",
             "tl.store(f2 + idx, v2, mask=live)"],
            ["tl.store(u0 + idx, n0, mask=live)",
             "tl.store(u1 + idx, n1, mask=live)",
             "tl.store(u2 + idx, n2, mask=live)",
             "tl.store(f0 + idx, v0, mask=live)",
             "tl.store(f1 + idx, v1, mask=live)",
             "tl.store(f2 + idx, v2, mask=live)",
             "if ZM_X:",
             "    v1 = tl.where(at_x, 0.0, v1)",
             "    v2 = tl.where(at_x, 0.0, v2)",
             "if ZM_Y:",
             "    v0 = tl.where(at_y, 0.0, v0)",
             "    v2 = tl.where(at_y, 0.0, v2)",
             "if ZM_Z:",
             "    v0 = tl.where(at_z, 0.0, v0)",
             "    v1 = tl.where(at_z, 0.0, v1)"])

    def m_inv_eps_dropped(source: str) -> Tuple[str, int]:
        """``update_E`` without the inverse-permittivity multiply."""
        return _replace_all(
            source, "src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
            "src0 = v0")

    def m_inv_eps_component_swapped(source: str) -> Tuple[str, int]:
        """Component 0 scaled by component 1's inverse permittivity."""
        return _replace_all(
            source, "src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
            "src0 = v0 * tl.load(ie1 + idx, mask=live, other=0.0)")

    def m_prev_read_after_the_store(source: str) -> Tuple[str, int]:
        """``f_w`` read AFTER it is overwritten — the one ordering the constitutive
        cannot survive being wrong about (S:2083-2085)."""
        return _rewrite_block(
            source,
            ["prev0 = tl.load(w0 + idx, mask=live, other=0.0)",
             "src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
             "tl.store(w0 + idx, src0, mask=live)"],
            ["src0 = v0 * tl.load(ie0 + idx, mask=live, other=0.0)",
             "tl.store(w0 + idx, src0, mask=live)",
             "prev0 = tl.load(w0 + idx, mask=live, other=0.0)"])

    def m_constitutive_axis_swapped(source: str) -> Tuple[str, int]:
        """Component 0's constitutive coefficient indexed on axis y.

        ``stepping.E_CONSTITUTIVE_TERMS`` :227 indexes each component on its OWN
        axis.
        """
        return _rewrite_block(
            source,
            ["kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
             "km_0 = tl.load(km0 + i, mask=live, other=0.0)"],
            ["kp_0 = tl.load(kp0 + j, mask=live, other=0.0)",
             "km_0 = tl.load(km0 + j, mask=live, other=0.0)"])

    def m_curl_parens_flattened(source: str) -> Tuple[str, int]:
        """``dtdx * (c_y - c + b - b_z)`` — a different float32 grouping."""
        return _replace_all(source, "curl0 = dtdx * ((c_y - c) + (b - b_z))",
                            "curl0 = dtdx * (c_y - c + b - b_z)")

    def m_d_store_dropped(source: str) -> Tuple[str, int]:
        """The stepped displacement never written back; the next step's curl reads
        it."""
        return _replace_all(source, "tl.store(f0 + idx, v0, mask=live)",
                            "tl.store(f0 + idx, v0, mask=live & (idx < 0))")

    return (
        ("m_tail_dropped", "no BFAST advance at all", "caught", m_tail_dropped),
        ("m_tail_sums_become_differences",
         "the tail differences its operands where the array path sums them",
         "caught", m_tail_sums_become_differences),
        ("m_tail_k_pair_swapped", "k1 and k2 exchanged on target 0", "caught",
         m_tail_k_pair_swapped),
        ("m_advance_state_factor", "total - state instead of total - 2*state",
         "caught", m_advance_state_factor),
        ("m_advance_sign_flipped", "curl + adv instead of curl - adv", "caught",
         m_advance_sign_flipped),
        ("m_state_store_dropped", "the IIR state never advanced", "caught",
         m_state_store_dropped),
        ("m_advance_mask_dropped",
         "the advance not masked by the owned-cell predicate", "caught",
         m_advance_mask_dropped),
        ("m_advance_mask_after_the_state_store",
         "the advance masked after the state store instead of before it (S:902)",
         "caught", m_advance_mask_after_the_state_store),
        ("m_tail_scaled_by_dtdx", "the advance multiplied by dtdx", "caught",
         m_tail_scaled_by_dtdx),
        ("m_state_component_swapped", "target 0 advances target 1's state array",
         "caught", m_state_component_swapped),
        ("m_wall_clear_dropped", "zero_metal_D not carried on the x wall", "caught",
         m_wall_clear_dropped),
        ("m_wall_clear_is_the_magnetic_familys_one_row",
         "the magnetic twin's one-row clear on the D family", "caught",
         m_wall_clear_is_the_magnetic_familys_one_row),
        ("m_wall_clear_after_the_store",
         "the clear applied after the D store, so the stored D keeps the wall value",
         "caught", m_wall_clear_after_the_store),
        ("m_inv_eps_dropped", "update_E without the inverse-permittivity multiply",
         "caught", m_inv_eps_dropped),
        ("m_inv_eps_component_swapped",
         "component 0 scaled by component 1's inverse permittivity", "caught",
         m_inv_eps_component_swapped),
        ("m_prev_read_after_the_store", "f_w read after it is overwritten", "caught",
         m_prev_read_after_the_store),
        ("m_constitutive_axis_swapped",
         "component 0's constitutive coefficient indexed on axis y", "caught",
         m_constitutive_axis_swapped),
        ("m_curl_parens_flattened", "a different float32 grouping in the curl",
         "caught", m_curl_parens_flattened),
        ("m_d_store_dropped", "the stepped displacement never written back",
         "caught", m_d_store_dropped),
    )


def mutation_case_for(name: str) -> Tuple[str, Tuple[Any, ...]]:
    """The case a mutation is scored on, and it must really carry what it needs."""
    case_name = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    case = CASES_BY_NAME[case_name]
    _name, _cell, boundaries, _steps = case

    def declaration(axis_letter: str) -> str:
        if isinstance(boundaries, str):
            return boundaries
        return str(boundaries.get(axis_letter.lower(), "periodic"))

    for axis_letter in MUTATION_REQUIRES_WALL.get(name, ()):
        if declaration(axis_letter) != "metallic":
            raise AssertionError(
                f"mutation {name} rewrites a wall clear on {axis_letter}, and case "
                f"{case_name!r} declares {declaration(axis_letter)!r} there: the "
                f"rewritten lines would be a DEAD BRANCH and the leg would report "
                f"'uncaught' while measuring nothing")
    return case_name, case


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source()
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation,
                               "rewrite_hits": hits, "device": True}
        if hits == 0:
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_bfast_electric_D"
        try:
            mutant = compile_mutant(mutated, kernel_name)
        except Exception as exc:  # noqa: BLE001 - an uncompilable mutant is a failure
            row["error"] = f"the mutant did not import: {exc!r}"
            rows.append(row)
            log(f"  mutation {name}: DID NOT COMPILE")
            continue
        case_name, case = mutation_case_for(name)
        row["case"] = case_name
        # THE CARRY IS ON. A mutation scored without the deposit would leave the
        # repair's interaction with the mutated register unmeasured on every leg but
        # the carry ones.
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], MUTATION_STEPS,
                      product, value_class="uniform", mutant=mutant, electric=True)
        row["leg"] = leg
        row["caught"] = leg.get("first_divergence") is not None
        row["fused_kernel_launches"] = leg.get("fused_kernel_launches")
        row["ptx_specializations"] = leg.get("ptx_specializations")
        row["pristine_ptx_specializations"] = len(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} "
            f"(expected {expectation}) on {case_name}")
    return rows


def run_host_mutations(cp, product) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    case = CASES_BY_NAME[DEFAULT_MUTATION_CASE]
    for entry in host_mutation_table():
        name, why, expectation, _apply = entry
        leg = run_leg(cp, f"host_mutation:{name}", case[1], case[2], MUTATION_STEPS,
                      product, value_class="uniform", host_mutation=entry,
                      electric=True)
        caught = leg.get("first_divergence") is not None
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "caught": caught, "device": True, "leg": leg})
        log(f"  host mutation {name}: caught={caught} (expected {expectation})")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse — and two it must ADMIT."""
    rows: List[Dict[str, Any]] = []
    _name, cell, boundaries, _steps = CASES_BY_NAME["corpus_periodic"]
    for name, shear, sources_of, needle, admit in (
        ("electric_source_with_a_deposit_index", BFAST_K,
         lambda d: (_deposit(d, "Ez"),), None, True),
        ("magnetic_source_only", BFAST_K,
         lambda d: (_deposit(d, "Hz"),), None, True),
        ("electric_source_without_a_deposit_index", BFAST_K,
         lambda d: (_ElectricWithoutIndex(),), "does not publish the index", False),
        ("undeclared_source_list", BFAST_K, lambda d: None,
         "was not declared", False),
        ("shear_free", (0.0, 0.0, 0.0), lambda d: (_deposit(d, "Ez"),),
         "bfast", False),
    ):
        driver = build_driver(cp, cell, boundaries, 11, "uniform", electric=False,
                              bfast_scaled_k=shear)
        try:
            sources = sources_of(driver)
            verdict = product.bfast_fused_electric_pair_coverage(
                driver.fields, driver.pml, sources)
            built = product.plan_bfast_fused_electric_pair(
                driver.fields, driver.pml, sources)
            row = {"refusal": name, "covered": bool(verdict.covered),
                   "reasons": list(verdict.reasons),
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


def _deposit(driver, component: str):
    """A REAL source that publishes the index the injection writes."""
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=driver.fields.grid, component=component,
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    if not source._n_source_points:
        raise AssertionError("the fixture source deposits nothing")
    if deposit_repair._deposit_index(source) is None:
        raise AssertionError("the fixture source publishes no deposit index")
    return source


# ---------------------------------------------------------------------------
# Environment
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
        HERE, "results", "triton_bfast_fused_electric_pair"),
        help="a DIRECTORY; gate.json is written inside it")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile.")
    args = parser.parse_args(argv)

    out = args.out
    artifact = out if out.endswith(".json") else os.path.join(out, "gate.json")
    if not out.endswith(".json"):
        os.makedirs(out, exist_ok=True)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_bfast_fused_electric_pair",
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
        except Exception as exc:  # noqa: BLE001 - a leg that cannot run is a failure
            row = {"leg": getattr(leg, "__name__", str(leg)), "device": False,
                   "error": repr(exc), "passed": False, "findings": [repr(exc)]}
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} "
            f"findings={row.get('findings')} ({row['seconds']} s)")
        save(payload, artifact)

    if args.no_device:
        payload["device_status"] = (
            "UNRUN — invoked with --no-device; no CUDA leg, mutation or refusal in "
            "this artifact")
        payload["passed"] = all(row["passed"] for row in payload["no_device_legs"])
        payload["release"] = {
            "released": False,
            "reasons": ["the no-device legs passed, but no device leg, mutation or "
                        "refusal has run: this artifact releases nothing"],
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
        "quiet_cases": {case[0]: case[3] for case in CASES},
        "carry_cases": list(CARRY_CASES),
        "steps_per_carry": CARRY_STEPS,
        "steps_per_mutation": MUTATION_STEPS,
        "value_classes": list(VALUE_CLASSES),
    }
    save(payload, artifact)

    log("\n=== device legs: the QUIET family (a magnetic source, no seam deposit) ===")
    for name, cell, boundaries, steps in CASES:
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"quiet:{name}", cell, boundaries, steps, product,
                          value_class=value_class)
            passed, failures = verdict_of(row, require_launches=steps)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== device legs: the CARRY family (a real electric deposit in the seam) ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, _steps = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{name}", cell, boundaries, CARRY_STEPS, product,
                          value_class=value_class, electric=True)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, _steps = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{name}", cell, boundaries, CARRY_STEPS,
                      product, electric=True, bracket=False)
        # REQUIRES DIVERGENCE. A bracket that changes nothing is not load-bearing.
        # The launch floor is "at least one" and the vacuity census is off: a leg
        # that must diverge STOPS at the first divergent step, so an exact launch
        # count and a whole-run moved-state census are statements about steps that
        # never ran.
        passed, failures = verdict_of(row, require_identical=False,
                                      require_launches_at_least=1,
                                      require_moved=False,
                                      require_array_reference=False)
        row["armed"] = True
        row["passed"], row["failures"] = passed, failures
        row["why"] = ("the fused launch WITHOUT the shipped deposit repair; it "
                      "consumes a pre-injection D and MUST diverge")
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== the REDUCTION leg: HAS_BFAST = 0 is the shipped ordinary kernel ===")
    reductions = []
    for value_class in VALUE_CLASSES:
        name, cell, boundaries, _steps = CASES_BY_NAME["wall_xyz"]
        row = reduction_leg(cp, product, cell, boundaries, 4, value_class)
        reductions.append(row)
        payload["device_legs"].append(row)
        save(payload, artifact)
    if not any(row.get("discriminating") for row in reductions):
        payload["device_legs"].append({
            "leg": "reduction:discrimination_floor", "device": True,
            "passed": False,
            "failures": ["NO value class separated HAS_BFAST=0 from the array path, "
                         "so every reduction agreement above is a tautology"],
        })
        save(payload, artifact)

    log("\n=== armed harness mutations ===")
    name, cell, boundaries, _steps = CASES_BY_NAME[DEFAULT_MUTATION_CASE]
    row = run_leg(cp, "armed:no_substitution", cell, boundaries, 3, product,
                  install_fused=False, electric=True)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    row = run_leg(cp, "armed:frozen_electric_seam", cell, boundaries, 3, product,
                  freeze_electric=True, electric=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.bfast_fused_curl_constitutive_D)
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
                     "mutations whose measured outcome did not match their declared "
                     "expectation, or that were never armed: "
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
    log(f"\nverdict: {payload['passed']}  ->  {artifact}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
