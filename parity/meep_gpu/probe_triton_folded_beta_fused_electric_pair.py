"""Device gate for the FOLDED real-beta fused ELECTRIC D/E pair on Triton.

DEVICE STATUS: **UNRUN at authoring time.** This module is written to be run on a
    CUDA host with an idle device; nothing in this tree may cite it as a release
    until an artifact exists with ``device_status: RUN`` and
    ``release.released: true``. The product module's ``DEVICE_STATUS`` is what
    names the released artifact once one exists.

Folded beta ``step_D`` welded into ``update_E`` with the mirror fills carried
inline — the product of :mod:`meep_gpu.triton_kernels.folded_beta_fused_electric_pair`,
serving the ``folded real beta PML`` -> ``folded beta run`` D->E cell.

===========================================================================
WHAT THE PRODUCT CLAIMS, AND WHAT EACH LEG MEASURES
===========================================================================

The claim is narrow and checkable: this kernel is
``folded_fused_pair.folded_fused_curl_constitutive_D``, character for character,
PLUS exactly the three lines ``folded_complex.folded_beta_pml_curl_step`` adds to
``symmetry.pml_curl_step_folded``.

1. **TRANSCRIPTION (no device).** Exact statement-list equalities against the
   shipped sources, read from the FILES so they bite on the merge bar as well as
   here: the kernel minus the beta insert IS ``folded_fused_curl_constitutive_D``,
   and the insert IS the delta ``folded_beta_pml_curl_step`` carries over
   ``pml_curl_step_folded``. Neither side is ``ast.unparse``d — what is being
   compared includes the PARENTHESISATION.
2. **REDUCTION (device).** ``HAS_BETA = 0`` must reproduce the shipped
   ``folded_fused_curl_constitutive_D`` BIT FOR BIT over complete steps on the
   same volumes, while DIFFERING from the array path (which really does add the
   beta term on a ``beta = 0.2`` grid) — the transcription claim measured on
   silicon, with the discrimination precondition recorded per value class.
3. **IDENTITY (device).** One launch leaves the engine bit-identical, as uint32
   words, per COMPLETE ``driver.step()``, to BOTH the CuPy array path AND the
   three separately certified Triton products it replaces (the ``folded real
   beta PML`` curl on ``step_D``, the mirror ghost fill on ``fill_D``, and the
   ``folded beta run`` constitutive on ``update_E``), at one launch and at ~60,
   over every allocated volume, in three value classes. ``zero_metal_D`` and
   ``fill_folded_far_ghosts_D`` stay on the ARRAY PATH in the separate oracle —
   no separate Triton product owns either — and both are COUNTED.
4. **THE CARRY (device), and it is what makes this cell worth anything.** The
   single corpus row (``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag``)
   declares TWO ELECTRIC sources, which the driver injects BETWEEN the two
   halves. Every CARRY case runs with the shipped
   :class:`~meep_gpu.deposit_repair.LeadingRepairPlan` /
   :class:`~meep_gpu.deposit_repair.TrailingRepairPlan` pair — whose fold rules
   save the deposit's IMAGES, the same closed form this kernel's forward carry
   implements — and each carry case has a NULL CONTROL with the bracket removed
   that MUST diverge.

BETA IS 2-D ONLY (grid.py:690; MEEP fields.cpp:546-547), so EVERY case here is
2-D and no case can fold z: the kernel's ``NEAR_Z``/``FAR_Z`` blocks are
compile-time absent on every admissible configuration, exactly as the product
docstring states. The deepest composition a 2-D folded beta grid can produce is
ghost-destination depth 3 (two folded PERIODIC axes), and the case table reaches
it.

===========================================================================
THE SEAM
===========================================================================

``driver.step`` runs, in order (driver.py:3292-3304)::

    step_D -> ELECTRIC SOURCES -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

WHAT THE CORPUS SAYS, from ``results/fusion_matrix_triton_2026-08-31_plainrepair7``
at the D->E cell (``folded real beta PML``, ``folded beta run``): ONE
seam-instance, ``tests:TestSpecialKz.test_eigsrc_kz_1_real_imag``,
``in_seam_source == true`` and ``in_seam_source_blocks == false``. The
``corpus_admission`` leg re-derives that from the census rather than citing it.

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_beta_fused_electric_pair.py \\
        --no-device --out parity/meep_gpu/results/<fresh-dir>

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_beta_fused_electric_pair.py \\
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
PRODUCT_MODULE = "meep_gpu.triton_kernels.folded_beta_fused_electric_pair"
KERNEL_FILE = os.path.join(PACKAGE_DIR, "folded_beta_fused_electric_pair.py")
KERNEL_NAME = "folded_beta_fused_curl_constitutive_D"

#: TestSpecialKz.test_eigsrc_kz_1_real_imag's own beta — the single row this cell
#: has (census ``configuration.beta``).
BETA_CORPUS = 0.2

#: The three lines the beta term adds, and the ONLY thing that may separate this
#: kernel from ``folded_fused_pair.folded_fused_curl_constitutive_D``.
BETA_INSERT = (
    "if HAS_BETA:",
    "curl0 = curl0 - (beta_plus * b)",
    "curl1 = curl1 - (beta_minus * a)",
)

#: The device cases. ``(name, cell, boundaries, mirrors, steps, options)``.
#:
#: EVERY CELL IS 2-D WITH A ZERO Z EXTENT: ``Grid`` refuses beta anywhere else by
#: name (grid.py:690). Z THEREFORE STAYS PERIODIC in every case — it is the
#: INVARIANT axis and ``Grid`` refuses a PEC there by name (grid.py:842-877).
#:
#: THE PARITIES AND THE TERMINATIONS ARE BOTH SWEPT. An EVEN plane makes the
#: near-parity mutation invisible (the multiply is by +1), so the odd rows are
#: what give it anywhere to be caught; a folded PERIODIC axis is what makes the
#: far carry live at all; an ODD full count is what separates the runtime reflect
#: row from the fixed ``n - 2`` a wrong transcription would bake.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any,
                   Tuple[Tuple[str, int], ...], int, Dict[str, Any]], ...] = (
    # THE CORPUS GEOMETRY: all-periodic with a Y fold over the PERIODIC
    # declaration, so the FAR fill is LIVE — exactly what
    # tests:TestSpecialKz.test_eigsrc_kz_1_real_imag lifts to (mirrored
    # [false, true, false], has_metallic false, an active PML).
    ("corpus_y_fold_periodic", (1.6, 1.4, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"}, (("Y", 1),), 60,
     {"expect_ghost_destinations": 1}),
    # ODD parity AND an odd full count (1.75 x 12 = 21): the reflect row is
    # ``stored - 3`` here, so a fixed ``n - 2`` transcription is a whole cell
    # wrong; and the near parity multiply is by -1, which is what arms the
    # parity mutation. Both liveness facts are ASSERTED by mutation_case_for
    # rather than trusted to this comment.
    ("odd_y_fold_periodic", (1.6, 1.75, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"}, (("Y", -1),), 60,
     {"expect_ghost_destinations": 1}),
    # A METALLIC fold beside a LIVE WALL on the other axis: the near fill runs,
    # the far fill is inert, and ZM_X fires — the wall-clear mutations are
    # scored here because the corpus geometry has no wall at all.
    ("y_fold_metallic_wall_x", (1.6, 1.4, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", 1),), 60, {}),
    # THE DEEPEST 2-D COMPOSITION: two folded PERIODIC axes at an ODD full
    # count (1.75 x 12 = 21 on both), phases (+1, -1) so the composed parity
    # product PHY * (PHX * v2) differs from PHX * v2 — the pair the
    # composite-parity mutation removes. Every D component owns THREE ghost
    # cells per source lane here.
    ("two_folds_periodic_odd", (1.75, 1.75, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", -1)), 60, {"expect_ghost_destinations": 3}),
    # ONE FOLD OF EACH TERMINATION on the same grid: x images a far ghost and y
    # does not, so a carry keyed off "is folded" rather than "is folded
    # PERIODIC" is wrong here and nowhere else.
    ("mixed_terminations", (1.6, 1.6, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"},
     (("X", -1), ("Y", 1)), 60, {"expect_ghost_destinations": 3}),
    # ONE LAUNCH: a divergence here is attributable to a single launch rather
    # than to an accumulation.
    ("corpus_single_launch", (1.6, 1.4, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"}, (("Y", 1),), 1,
     {"expect_ghost_destinations": 1}),
)

CASES_BY_NAME: Dict[str, Tuple[Any, ...]] = {case[0]: case for case in CASES}

#: The case every kernel mutation is scored on unless it names another.
#: ``two_folds_periodic_odd``: both fills live, both parities real, odd count.
DEFAULT_MUTATION_CASE = "two_folds_periodic_odd"

#: Per mutation, the case it is scored on. TRAP: a mutation scored on a grid that
#: never ENTERS the branch it rewrites reports UNCAUGHT while measuring nothing.
MUTATION_CASE: Dict[str, str] = {
    "m_wall_clear_dropped": "y_fold_metallic_wall_x",
    "m_beta_below_the_masks": "y_fold_metallic_wall_x",
    "m_near_parity_dropped": "odd_y_fold_periodic",
    "m_far_row_fixed_n_minus_2": "odd_y_fold_periodic",
    "m_far_parity_not_negated": "corpus_y_fold_periodic",
    "m_far_carry_dropped": "corpus_y_fold_periodic",
}

#: The CARRY family: a real ELECTRIC deposit IN THIS SEAM, on the corpus
#: geometry (fold over PERIODIC — the repair must image the deposit through the
#: far fill too) and on the walled metallic fold (the deposit beside a live wall
#: clear and the near fill).
CARRY_CASES: Tuple[str, ...] = ("corpus_y_fold_periodic", "y_fold_metallic_wall_x")

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

#: The volumes THIS SEAM writes. A leg in which none of them moves measured
#: nothing about this launch, whatever else moved.
SEAM_OUTPUTS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
)

#: Private ``Fields`` scratch that is NOT physical state. The rule is the
#: leading underscore, not this list.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)

#: Per-case driver defaults; a case that names none takes these.
DEFAULT_OPTIONS: Dict[str, Any] = {"resolution": 12.0, "courant": 0.35, "pml": 4}

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(*parts: str) -> int:
    """A per-case seed from a DIGEST, never ``hash()`` — replayable from the record."""
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % (2 ** 31 - 1)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and material_changed would never fire.")


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
        "meep_gpu/triton_kernels/folded_beta_fused_electric_pair.py",
        "meep_gpu/triton_kernels/folded_beta_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/folded_fused_pair.py",
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
        "meep_gpu/test_triton_folded_beta_fused_electric_pair.py",
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

    The text is EXACT — never ``ast.unparse``d. What is being compared includes
    the PARENTHESISATION, and unparsing re-derives minimal parentheses, which
    would silently equate ``dtdx * ((c_y - c) + (b - b_z))`` with a different
    float32 grouping.
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


def shipped_source() -> str:
    """The kernel as source text, decorator included — the mutation harness input.

    ``ast.get_source_segment`` on a ``FunctionDef`` EXCLUDES the decorator list,
    so a mutant compiled from the bare segment is a plain Python function with no
    ``[grid]`` launcher; the decorator is re-prepended here.
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

    * the kernel minus the three beta lines IS
      ``folded_fused_pair.folded_fused_curl_constitutive_D``, statement for
      statement, in order;
    * the three lines are ``folded_beta_pml_curl_step``'s own measured delta over
      ``symmetry.pml_curl_step_folded``;
    * the insert sits AFTER the third ``dtdx`` curl line and BEFORE the cell-0
      ownership mask — the array path's order and K3a's own placement — checked
      POSITIONALLY, because an order-preserving deletion equality alone cannot
      see where the insert sits;
    * nothing in the body comes from neither shipped source; and this is the
      ELECTRIC weld, not a copy of the magnetic twin.
    """
    folded_pair_py = os.path.join(PACKAGE_DIR, "folded_fused_pair.py")
    folded_complex_py = os.path.join(PACKAGE_DIR, "folded_complex.py")
    symmetry_py = os.path.join(PACKAGE_DIR, "symmetry.py")
    twin_py = os.path.join(PACKAGE_DIR, "folded_beta_fused_magnetic_pair.py")

    fused = body_statements(KERNEL_FILE, KERNEL_NAME)
    base = body_statements(folded_pair_py, "folded_fused_curl_constitutive_D")
    beta_curl = body_statements(folded_complex_py, "folded_beta_pml_curl_step")
    plain_curl = body_statements(symmetry_py, "pml_curl_step_folded")
    twin = body_statements(twin_py, "folded_beta_fused_curl_constitutive_B")

    findings: List[str] = []
    without_beta = [line for line in fused if line not in BETA_INSERT]
    if without_beta != base:
        index = next((i for i, (a, b) in enumerate(zip(without_beta, base))
                      if a != b), min(len(without_beta), len(base)))
        findings.append(
            f"minus the beta insert this is NOT folded_fused_curl_constitutive_D; "
            f"first difference at statement {index}: "
            f"{without_beta[index:index + 1]} vs {base[index:index + 1]}")

    delta = [line for line in beta_curl if line not in plain_curl]
    if delta != list(BETA_INSERT):
        findings.append(f"folded_beta_pml_curl_step's own delta over "
                        f"pml_curl_step_folded is {delta}, not {list(BETA_INSERT)}")

    # THE PLACEMENT, positionally: immediately after the third curl line,
    # immediately before the cell-0 mask assignment — in the fused body AND in
    # the shipped beta curl it is transcribed from.
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
        findings.append("this kernel is byte-identical to the MAGNETIC twin; the "
                        "D seam has a different wall clear, the inv_eps multiply "
                        "and a different fill geometry")
    for component in range(3):
        line = (f"src{component} = v{component} * tl.load("
                f"ie{component} + idx, mask=own{component}, other=0.0)")
        if line not in fused:
            findings.append(f"the E side's inverse-permittivity multiply is "
                            f"missing for component {component}: {line!r}")
        if line in twin:
            findings.append(f"the MAGNETIC twin carries {line!r}; update_H reads "
                            f"no inverse epsilon")

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
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, size=grid.shape).astype(np.float32)
    if thickness is None:
        folded = {("xyz".index(axis.lower())) for axis, _ in mirrors}
        thickness = tuple(
            ((0, pml_thickness) if axis in folded else
             (pml_thickness, pml_thickness)) if grid.shape[axis] >= 6 else (0, 0)
            for axis in range(3))
    return fields, PML(grid=grid, thickness=thickness)


class _Magnetic:
    field_type = "B"


class _Electric:
    """An electric source that publishes the index the injection writes."""

    field_type = "D"

    def __init__(self, index=(2, 2, 0)) -> None:
        self._point_ix, self._point_iy, self._point_iz = index


class _ElectricWithoutIndex:
    field_type = "D"


#: One-clause perturbations off the corpus family. Every one must leave the weld
#: no wider than the narrower of its two halves.
EQUIVALENCE_CASES: Tuple[Tuple[str, Dict[str, Any], Any], ...] = (
    ("corpus", {}, (_Electric(),)),
    ("magnetic_source", {}, (_Magnetic(),)),
    ("undeclared_sources", {}, None),
    ("electric_without_an_index", {}, (_ElectricWithoutIndex(),)),
    ("zero_beta", {"beta": 0.0}, (_Electric(),)),
    ("complex_storage", {"complex_storage": True}, (_Electric(),)),
    ("inactive_absorber", {"pml_thickness": 0, "thickness": ((0, 0),) * 3},
     (_Electric(),)),
    ("unfolded", {"mirrors": ()}, (_Electric(),)),
    ("bloch", {"k_point": (0.2, 0.0, 0.0)}, (_Electric(),)),
)


def equivalence_leg() -> Dict[str, Any]:
    """``weld => K3a curl AND folded beta constitutive``, over one-clause moves."""
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    product = importlib.import_module(PRODUCT_MODULE)
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    admitted = 0
    for name, keywords, sources in EQUIVALENCE_CASES:
        fields, pml = _laptop_fixture(**dict(keywords))
        weld = product.folded_beta_fused_electric_pair_coverage(fields, pml, sources)
        curl = folded_complex.folded_beta_pml_curl_coverage(fields, pml, "step_D")
        electric = folded_complex.folded_beta_run_constitutive_coverage(
            fields, pml, "E")
        never_wider = (not weld.covered) or (curl.covered and electric.covered)
        covered = not [r for r in weld.reasons if "array module" not in r]
        rows.append({"case": name, "weld_covered_modulo_backend": covered,
                     "weld_never_wider": never_wider,
                     "weld_reasons": list(weld.reasons)[:6]})
        if not never_wider:
            findings.append(f"{name}: the weld admitted where a half refused")
        admitted += int(covered)
    if admitted == 0:
        findings.append("no case was admitted modulo the backend clause: this leg "
                        "measured only refusals and could not see a widening")

    # THE FLAG IS THE PRODUCT. With CARRIES_DEPOSIT_REPAIR at False the corpus
    # row's own electric sources are refused outright and the cell is worth ZERO.
    fields, pml = _laptop_fixture()
    before = [r for r in product.folded_beta_fused_electric_pair_coverage(
        fields, pml, (_Electric(),)).reasons if "array module" not in r]
    original = product.CARRIES_DEPOSIT_REPAIR
    try:
        product.CARRIES_DEPOSIT_REPAIR = False
        after = [r for r in product.folded_beta_fused_electric_pair_coverage(
            fields, pml, (_Electric(),)).reasons if "array module" not in r]
    finally:
        product.CARRIES_DEPOSIT_REPAIR = original
    if before:
        findings.append(f"the corpus family is refused WITH the flag: {before}")
    if not any("is electric" in reason for reason in after):
        findings.append(
            "flipping CARRIES_DEPOSIT_REPAIR to False did NOT refuse the electric "
            "source: the flag is not what carries this cell")

    return {"leg": "equivalence", "device": False, "cases": rows,
            "admitted_modulo_backend": admitted,
            "reasons_with_the_flag": before,
            "reasons_with_the_flag_off": after,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — what the corpus says, re-derived from the census
# ---------------------------------------------------------------------------

CENSUS = os.path.join(HERE, "results",
                      "predicate_coverage_triton_2026-08-31b_plainrepair")
MATRIX = os.path.join(HERE, "results",
                      "fusion_matrix_triton_2026-08-31_plainrepair7",
                      "fusion_matrix.json")
CELL = ("D->E", "folded real beta PML", "folded beta run")


def corpus_admission_leg() -> Dict[str, Any]:
    """The rows this arm can serve, counted rather than cited.

    A row is in this cell when it declares a NONZERO beta, an active absorber,
    real storage, a FOLD, no cylindrical axis, k = 0, no BFAST, no nonlinearity
    and no off-diagonal epsilon. AN ABSENT CENSUS IS NOT AN EMPTY ONE: a missing
    artifact is a REFUSAL here, never a funnel of zero reported as a pass.
    """
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
    electric: List[str] = []
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
            # The plainrepair7 board was cut BEFORE this product existed and must
            # score the cell unoccupied; a later board naming THIS product is fine.
            if entry["product"] not in (None, "folded_beta_fused_electric_pair"):
                findings.append(
                    f"the board names {entry['product']!r} in this cell")
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
    """A rewrite that matches nothing measured nothing, whatever it reported."""
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
    """Ghost cells ONE source lane owns, per D component, from the plan's booleans.

    The D family is the B family's complement: component ``m`` is a NEAR
    destination on every folded axis that is NOT ``m`` (``IYEE_SHIFTS``:
    ``Dx (1,0,0)`` — near where the shift is 0, i.e. y and z) and a FAR
    destination on ``m`` alone. The two fills COMPOSE, so the count is
    ``(1 + far_m) * 2 ** (near axes other than m) - 1``: 1 under one fold of
    either kind, 3 under two folds, and the z arms are compile-time absent on
    every 2-D beta grid.
    """
    out: List[int] = []
    for component in range(3):
        others = sum(1 for axis in range(3)
                     if axis != component and bool(near[axis]))
        out.append((1 + int(bool(far[component]))) * (2 ** others) - 1)
    return (out[0], out[1], out[2])


def build_grid(cell, boundaries, mirror_specs,
               options: Optional[Dict[str, Any]] = None,
               prefer_gpu: bool = True):
    """The grid and absorber one CASES row names, with no field seeding.

    A folded axis absorbs on its HIGH face only — the low face is the mirror
    plane — and every case here is 2-D, so only x and y take a layer.
    """
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
    """The constexprs one CASES row compiles the kernel with, read on NumPy."""
    from meep_gpu.triton_kernels import (  # noqa: PLC0415
        folded_beta_fused_electric_pair as product,
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
    """Seed every settable volume in ONE value class, identically per route."""
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
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(draw())


def _off_plane_center(cell, mirror_specs) -> Tuple[float, float, float]:
    """A point OFF the mirror plane on every folded axis (from_meep.py:2686-2698)."""
    center = [0.0, 0.0, 0.0]
    for axis_name, _phase in mirror_specs:
        axis = "XYZ".index(axis_name.upper())
        center[axis] = 0.25 * (cell[axis] / 2.0)
    return (center[0], center[1], center[2])


def build_driver(cp, cell, boundaries, mirror_specs, seed: int, value_class: str,
                 electric: bool, options: Optional[Dict[str, Any]] = None):
    """One 2-D folded REAL-beta PML driver, seeded identically for every route."""
    driver = build_grid(cell, boundaries, mirror_specs, options)
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, and a DIFFERENT one per component: a material that
    # varies in space but not in component makes `src0 = v0 * ie1` a bitwise
    # no-op, and the inv_eps component-swap mutation would report a real defect
    # as UNCAUGHT (measured on the beta electric sibling, 2026-08-31).
    epsilon = {
        name: np.ascontiguousarray(
            (2.25 + 0.30 * np.sin(index * np.float32(0.037 + 0.011 * offset))
             + 0.17 * offset).astype(np.float32))
        for offset, name in enumerate(("Ex", "Ey", "Ez"))}
    driver.set_epsilon_components({name: cp.asarray(values)
                                   for name, values in epsilon.items()})
    center = _off_plane_center(cell, mirror_specs)
    if electric:
        # THE DEPOSIT THIS PRODUCT EXISTS FOR, injected BETWEEN the two halves
        # (driver.py:3294-3299) and OFF the mirror plane on every folded axis.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": center, "width": 0.4})
    else:
        # A MAGNETIC source keeps the run alive without touching this seam.
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": center, "width": 0.4})
    seed_values(cp, driver, seed, value_class)
    # THE FAST PATH IS DISABLED, EXPLICITLY: the `folded real beta PML` and
    # `folded beta run` arms ARE wired, so a driver left alone would step this
    # seam on Triton and the "array path" reference would be no such thing.
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

    Counting is PER ROUTE: a single global counter would mix the reference
    driver's legitimate array-path calls with a fallback in the fused route.
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
    """The THREE separately certified Triton products this launch replaces.

    They are the arms ``launch.plan_step`` really selects on a folded beta row —
    ``folded real beta PML`` on ``step_D`` (K3a through
    ``plan_folded_beta_pml_curl``), the mirror ghost fill on ``fill_D``, and the
    ``folded beta run`` constitutive on ``update_E``. ``zero_metal_D`` and
    ``fill_folded_far_ghosts_D`` stay on the ARRAY PATH here — no separate
    Triton product owns either — and both are COUNTED. THE INJECTION ALSO STAYS
    ON THE ARRAY PATH, which is what makes this the right oracle for the carry
    family: the certified kernels with the driver's own deposit between them.
    """
    from meep_gpu.triton_kernels import folded_complex, symmetry  # noqa: PLC0415

    curl = folded_complex.plan_folded_beta_pml_curl(
        driver.fields, driver.pml, "step_D", num_warps=1)
    fill = symmetry.plan_mirror_ghost_fill(driver.fields, "D")
    constitutive = folded_complex.plan_folded_beta_run_constitutive(
        driver.fields, driver.pml, "E", num_warps=1)
    missing = [name for name, plan in
               (("folded real beta PML curl", curl),
                ("mirror ghost fill", fill),
                ("folded beta run constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the certified "
            f"products it replaces")
    return Route({"step_D": curl, "fill_symmetry_bc_D": fill,
                  "update_E": constitutive})


def fused_route(plan, driver=None, sources=(), bracket: bool = True):
    """The fused plan in its two slots, BRACKETED when the seam carries a deposit.

    THE BRACKET IS THE SHIPPED ONE — ``deposit_repair.LeadingRepairPlan`` /
    ``TrailingRepairPlan``, whose fold rules (``repair_cells``) save the
    deposit's IMAGES through the same fill map this kernel's forward carry
    implements. ``bracket=False`` is the NULL CONTROL and it must diverge.
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
    """One coefficient group moved to the WRONG lattice — a half-cell absorber
    error: converged, smooth and wrong."""
    from meep_gpu.triton_kernels.launch import CupyPointer, _flat  # noqa: PLC0415

    for slot in ("_curl_coefficients", "_e_coefficients"):
        if not hasattr(plan, slot):
            raise AssertionError(
                f"the plan has no {slot}; this host mutation would bind nothing")
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


def _swap_beta_words(plan):
    plan.beta_plus, plan.beta_minus = plan.beta_minus, plan.beta_plus
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
         "the constitutive half bound the INTEGER kps/kms pair, which is the "
         "curl side's lattice",
         "caught",
         lambda driver, plan: _swapped_coefficients(driver, plan, "constitutive")),
        ("h_beta_words_swapped",
         "beta_plus and beta_minus exchanged; they differ in SIGN, so the term "
         "is applied to the wrong partner with the wrong sign",
         "caught", lambda driver, plan: _swap_beta_words(plan)),
    )


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, mirrors, steps: int, product,
            value_class: str = "uniform", mutant: Any = None,
            host_mutation: Any = None, install_fused: bool = True,
            freeze_electric: bool = False, electric: bool = False,
            bracket: bool = True,
            options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    seed = case_seed(name, str(cell), str(boundaries), str(mirrors), value_class)
    made = [build_driver(cp, cell, boundaries, mirrors, seed, value_class,
                         electric, options) for _ in range(3)]
    reference, separate, fused = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_beta_fused_curl_constitutive_D_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "mirrors": [list(entry) for entry in mirrors],
        "value_class": value_class, "seed": seed,
        "electric_source": bool(electric), "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "host_mutation": None, "first_divergence": None,
        "control_divergence": None,
    }
    try:
        plan = product.plan_folded_beta_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.folded_beta_fused_electric_pair_coverage(
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
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
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
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    if require_ghost_destinations is not None:
        observed = row.get("ghost_destinations_per_source_lane") or []
        if max(observed or [0]) != int(require_ghost_destinations):
            failures.append(
                f"this leg was declared to reach composition depth "
                f"{require_ghost_destinations} and reached {max(observed or [0])} "
                f"({observed}); the kernel emits its carry blocks per depth, so a "
                f"leg short of the declared one leaves them unexecuted")
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if not require_identical and row.get("first_divergence") is None:
        failures.append(
            "this leg REQUIRES divergence and found none: the control it exists "
            "to be is inert")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path: "
            f"{row['control_divergence']}")
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
                f"the fused route left {swallowed} INERT while the array path "
                f"moves them: a pass this weld was supposed to carry did not run")
        reference_movers = set(row.get("reference_seam_outputs_moved") or ())
        if not reference_movers:
            failures.append(
                "VACUOUS: the ARRAY PATH itself moves nothing this seam writes "
                "on this configuration")
        movers = set(row.get("seam_outputs_moved") or ())
        if movers != reference_movers:
            failures.append(
                f"the fused route's seam-output movement {sorted(movers)} is not "
                f"the array path's {sorted(reference_movers)}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    if row.get("inventory_asymmetry_vs_array"):
        failures.append(
            f"the compared inventories differ by NAME between routes: "
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
    """``HAS_BETA = 0`` reproduces ``folded_fused_curl_constitutive_D`` bit for
    bit, while DIFFERING from the array path (which really adds the beta term).

    Route B goes through ``folded_fused_pair.plan_folded_fused_pair_from_arrays``
    — the bare-array route, because the plain folded predicate refuses a beta
    run by name and its engine builder would return ``None``. NO ELECTRIC SOURCE
    HERE: both fused routes are unbracketed, so a deposit in the seam would put
    both one injection behind the array path and the discrimination precondition
    would fire for the wrong reason.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_fused_pair  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels.symmetry import (  # noqa: PLC0415
        _far_reflect_rows, folded_axis_kinds,
    )

    seed = case_seed("reduction", str(cell), str(boundaries), str(mirrors),
                     value_class)
    made = [build_driver(cp, cell, boundaries, mirrors, seed, value_class,
                         electric=False) for _ in range(3)]
    reference, shipped, reduced = made
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    row: Dict[str, Any] = {"leg": "reduction:has_beta_zero", "device": True,
                           "steps_budget": steps, "value_class": value_class,
                           "seed": seed, "shape": list(reference.shape),
                           "boundaries": boundaries,
                           "mirrors": [list(entry) for entry in mirrors]}
    try:
        reduced_plan = product.plan_folded_beta_fused_electric_pair(
            reduced.fields, reduced.pml, tuple(reduced._sources), num_warps=1)
        if reduced_plan is None:
            raise AssertionError("the product refused the reduction case")
        reduced_plan.has_beta = 0

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
        codes, reasons = folded_axis_kinds(grid, pml)
        if codes is None:
            raise AssertionError(f"folded_axis_kinds refused: {reasons}")
        shipped_plan = folded_fused_pair.plan_folded_fused_pair_from_arrays(
            arrays, curl_flat, constitutive_flat, codes,
            zero_metal_axes(grid), product.mirror_phases(grid),
            grid.dt / grid.dx, num_warps=1,
            reflect=_far_reflect_rows(grid) or (None, None, None))

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
            failures.append(f"HAS_BETA=0 is NOT the shipped folded fused pair: "
                            f"{row['first_divergence']}")
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
    """Compile a renamed mutant; the rename keeps the JIT cache honest.

    ``_carry_ghost_E`` is IMPORTED by the shipped module, so the mutant module
    imports it the same way — a name the mutant does not define makes it fail to
    IMPORT, which would score as unarmed rather than as caught.
    """
    header = ("import triton\nimport triton.language as tl\n"
              "from meep_gpu.triton_kernels.symmetry import (\n"
              "    CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC)\n"
              "from meep_gpu.triton_kernels.folded_fused_pair import _carry_ghost_E\n"
              "PERIODIC = tl.constexpr(CODE_PERIODIC)\n"
              "MIRROR_METALLIC = tl.constexpr(CODE_MIRROR_METALLIC)\n"
              "MIRROR_PERIODIC = tl.constexpr(CODE_MIRROR_PERIODIC)\n\n")
    text = header + source.replace(KERNEL_NAME, kernel_name)
    ast.parse(text)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_folded_beta_electric.py", delete=False,
        encoding="utf-8")
    handle.write(text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_beta_electric_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace a multi-line fragment, matching on SIGNIFICANT statements only."""
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
        """The insert moved BELOW both ownership masks.

        The fold is what makes this catchable at all: the masks fire on curl0
        and curl1 — the two beta targets — at cell 0 of a walled axis and at the
        top plane of a folded PERIODIC axis, so a beta term added after them
        leaves a live increment on a plane the array path zeroes. Scored on the
        walled fold, where the cell-0 arm is live.
        """
        without, removed = _rewrite_block(
            source, list(BETA_INSERT), ["pass"])
        if not removed:
            return source, 0
        return _rewrite_block(
            without,
            ["if BCZ == MIRROR_PERIODIC:",
             "curl2 = tl.where(last_z, 0.0, curl2)"],
            ["if BCZ == MIRROR_PERIODIC:",
             "    curl2 = tl.where(last_z, 0.0, curl2)",
             "if HAS_BETA:",
             "    curl0 = curl0 - (beta_plus * b)",
             "    curl1 = curl1 - (beta_minus * a)"])

    def m_near_parity_dropped(source: str) -> Tuple[str, int]:
        """``gv_0y = PHY * v0`` -> ``v0``: the near image loses its parity."""
        return _replace_all(source, "gv_0y = PHY * v0", "gv_0y = v0")

    def m_far_parity_not_negated(source: str) -> Tuple[str, int]:
        """The far image drops the minus MEEP's periodic fold carries."""
        return _replace_all(
            source,
            "_carry_ghost_E(f1, w1, e1, ie1, idx + df_y, -PHY * v1,",
            "_carry_ghost_E(f1, w1, e1, ie1, idx + df_y, PHY * v1,")

    def m_far_carry_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["if FAR_Y:",
             "_carry_ghost_E(f1, w1, e1, ie1, idx + df_y, -PHY * v1,",
             "kp_f1, km_f1, own1 & far_j)"],
            ["if FAR_Y:", "    pass"])

    def m_near_carry_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["_carry_ghost_E(f0, w0, e0, ie0, idx + dn_y, gv_0y,",
             "kp_0, km_0, own0 & near_j)"],
            ["pass"])

    def m_composite_parity_short(source: str) -> Tuple[str, int]:
        """``PHY * (PHX * v2)`` -> ``PHX * v2``: the composed corner loses one
        plane's parity. Scored on mixed phases, where the two differ."""
        return _replace_all(source, "gv_2xy = PHY * (PHX * v2)",
                            "gv_2xy = PHX * v2")

    def m_far_row_fixed_n_minus_2(source: str) -> Tuple[str, int]:
        """The runtime reflect row replaced by the even-count guess ``n - 2``."""
        return _replace_all(source, "far_j = live & (j == ry)",
                            "far_j = live & (j == ny - 2)")

    def m_ownership_and_dropped(source: str) -> Tuple[str, int]:
        """The carry mask loses its ownership AND: a non-owned lane images."""
        return _rewrite_block(
            source,
            ["_carry_ghost_E(f0, w0, e0, ie0, idx + dn_y, gv_0y,",
             "kp_0, km_0, own0 & near_j)"],
            ["_carry_ghost_E(f0, w0, e0, ie0, idx + dn_y, gv_0y,",
             "               kp_0, km_0, live & near_j)"])

    def m_wall_clear_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["if ZM_X:",
             "v1 = tl.where(at_x, 0.0, v1)",
             "v2 = tl.where(at_x, 0.0, v2)"],
            ["if ZM_X:", "    pass"])

    def m_inv_eps_component_swapped(source: str) -> Tuple[str, int]:
        return _replace_all(
            source,
            "src0 = v0 * tl.load(ie0 + idx, mask=own0, other=0.0)",
            "src0 = v0 * tl.load(ie1 + idx, mask=own0, other=0.0)")

    return (
        ("m_beta_term_dropped",
         "no beta term at all — the plain folded pair on a beta run",
         "caught", m_beta_term_dropped),
        ("m_beta_partners_swapped",
         "each beta target takes the OTHER source's centre value",
         "caught", m_beta_partners_swapped),
        ("m_beta_sign_flipped",
         "curl + (c * g) instead of curl - (c * g)",
         "caught", m_beta_sign_flipped),
        ("m_beta_scaled_by_dtdx",
         "the beta term multiplied by dtdx; it is an analytic derivative and is "
         "not (S:733-735)",
         "caught", m_beta_scaled_by_dtdx),
        ("m_beta_below_the_masks",
         "the insert moved below BOTH ownership masks; on a folded/walled grid a "
         "masked plane keeps a beta increment the array path zeroes",
         "caught", m_beta_below_the_masks),
        ("m_near_parity_dropped",
         "the near image loses its declared parity; visible only at an ODD plane",
         "caught", m_near_parity_dropped),
        ("m_far_parity_not_negated",
         "the far image drops the minus MEEP's periodic fold carries",
         "caught", m_far_parity_not_negated),
        ("m_far_carry_dropped",
         "the far ghost plane is never imaged",
         "caught", m_far_carry_dropped),
        ("m_near_carry_dropped",
         "one near ghost plane is never imaged",
         "caught", m_near_carry_dropped),
        ("m_composite_parity_short",
         "the two-fold corner composes ONE parity instead of two",
         "caught", m_composite_parity_short),
        ("m_far_row_fixed_n_minus_2",
         "the reflect row baked as n - 2, a whole cell wrong at an odd count",
         "caught", m_far_row_fixed_n_minus_2),
        ("m_ownership_and_dropped",
         "the near carry images from a lane the ownership rule excludes",
         "caught", m_ownership_and_dropped),
        ("m_wall_clear_dropped",
         "zero_metal_D's x-wall rows are never applied",
         "caught", m_wall_clear_dropped),
        ("m_inv_eps_component_swapped",
         "component 0 reads component 1's inverse epsilon; visible because the "
         "harness installs a DIFFERENT material per component",
         "caught", m_inv_eps_component_swapped),
    )


def mutation_case_for(name: str) -> Tuple[str, Tuple[Any, ...]]:
    """The case a mutation is scored on, with the liveness rules asserted."""
    case_name = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    case = CASES_BY_NAME[case_name]
    constexprs = case_constexprs(case)
    if name in ("m_near_parity_dropped",):
        if constexprs["PH"][1] != -1:
            raise AssertionError(
                f"{name} is scored on {case_name}, whose Y parity is "
                f"{constexprs['PH'][1]}; a +1 plane makes the multiply a no-op")
    if name in ("m_far_parity_not_negated", "m_far_carry_dropped",
                "m_far_row_fixed_n_minus_2"):
        if not constexprs["FAR"][1]:
            raise AssertionError(
                f"{name} is scored on {case_name}, whose y axis images no far "
                f"ghost; the rewritten branch never executes")
    if name == "m_far_row_fixed_n_minus_2":
        # ARMED only where the true reflect row differs from the baked n - 2 —
        # read from the array path's own function, on the case's real grid.
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
    if name in ("m_composite_parity_short", "m_ownership_and_dropped"):
        if not (constexprs["NEAR"][0] and constexprs["NEAR"][1]):
            raise AssertionError(
                f"{name} is scored on {case_name}, which does not fold both x "
                f"and y; the composed corner never executes")
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
            kernel = compile_mutant(mutated, f"{KERNEL_NAME}_m{index}")
            leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3],
                          MUTATION_STEPS, product, value_class="uniform",
                          mutant=kernel, electric=True, options=case[5])
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
                      host_mutation=entry, electric=True, options=case[5])
        caught = leg.get("first_divergence") is not None
        rows.append({"mutation": name, "why": why, "expectation": expectation,
                     "caught": caught, "device": True, "leg": leg})
        log(f"  host mutation {name}: caught={caught} (expected {expectation})")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def _deposit(driver, component: str):
    """A REAL source that publishes the index the injection writes, OFF the
    mirror plane on every folded axis."""
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
    """Configurations the product must refuse — and two it must ADMIT."""
    rows: List[Dict[str, Any]] = []
    corpus = CASES_BY_NAME["corpus_y_fold_periodic"]
    for name, mirrors, sources_of, needle, admit in (
        ("electric_source_with_a_deposit_index", corpus[3],
         lambda d: (_deposit(d, "Ez"),), None, True),
        ("magnetic_source_only", corpus[3],
         lambda d: (_deposit(d, "Hz"),), None, True),
        ("electric_source_without_a_deposit_index", corpus[3],
         lambda d: (_ElectricWithoutIndex(),), "does not publish the index",
         False),
        ("undeclared_source_list", corpus[3], lambda d: None,
         "was not declared", False),
        ("unfolded_grid", (), lambda d: (_deposit(d, "Ez"),),
         "mirror", False),
    ):
        driver = build_grid(corpus[1], corpus[2], mirrors, corpus[5])
        try:
            sources = sources_of(driver)
            verdict = product.folded_beta_fused_electric_pair_coverage(
                driver.fields, driver.pml, sources)
            built = product.plan_folded_beta_fused_electric_pair(
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
        HERE, "results", "triton_folded_beta_fused_electric_pair"),
        help="a DIRECTORY; gate.json is written inside it")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before "
             "the first device compile.")
    args = parser.parse_args(argv)

    out = args.out
    artifact = out if out.endswith(".json") else os.path.join(out, "gate.json")
    if not out.endswith(".json"):
        os.makedirs(out, exist_ok=True)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_folded_beta_fused_electric_pair",
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

    log("\n=== device legs: the QUIET family (a magnetic source, no seam deposit) ===")
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

    log("\n=== device legs: the CARRY family (a real electric deposit in the seam) ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, mirrors, _steps, options = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{name}", cell, boundaries, mirrors,
                          CARRY_STEPS, product, value_class=value_class,
                          electric=True, options=options)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, mirrors, _steps, options = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{name}", cell, boundaries, mirrors,
                      CARRY_STEPS, product, electric=True, bracket=False,
                      options=options)
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

    log("\n=== the REDUCTION leg: HAS_BETA = 0 is the shipped plain folded kernel ===")
    reductions = []
    for value_class in VALUE_CLASSES:
        name, cell, boundaries, mirrors, _steps, options = CASES_BY_NAME[
            "corpus_y_fold_periodic"]
        row = reduction_leg(cp, product, cell, boundaries, mirrors, 4, value_class)
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
                  product, install_fused=False, electric=True, options=options)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and "
                  "the counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    row = run_leg(cp, "armed:frozen_electric_seam", cell, boundaries, mirrors, 3,
                  product, freeze_electric=True, electric=True, options=options)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree "
                  "trivially and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.folded_beta_fused_curl_constitutive_D)
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
