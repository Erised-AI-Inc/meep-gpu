#!/usr/bin/env python3
"""THE COMPLEX BETA FUSED ELECTRIC PAIR GATE — complex beta ``step_D`` welded
into complex ``update_E``.

Does ONE launch of ``complex_beta_fused_curl_constitutive_D`` leave the engine
BIT-IDENTICAL, per COMPLETE driver step, to both the CuPy array path and the two
separately certified Triton products it replaces (the K2 beta bloch curl and the
complex constitutive on a beta run) — over every allocated volume, compared as
uint32 words?

THE BETA-LESS ELECTRIC GATE IS THE TEMPLATE AND THE DIFFERENCES ARE THE POINT.
``probe_triton_complex_fused_electric_pair.py`` asks the same question with no
beta term. Three things are genuinely different here:

1. **EVERY CASE CARRIES A NONZERO ``grid.beta``**, on a 2-D cell (the only place
   MEEP allows beta), and the separate oracle is the BETA pair of certified
   singles (``special_kz.plan_beta_bloch_pml_curl`` /
   ``plan_beta_run_complex_constitutive``), not the plain complex ones — whose
   predicates refuse every one of these runs by name.
2. **THE BETA PASS HAS ITS OWN MUTATIONS AND ITS OWN REDUCTION ARM.**
   ``fold_pass_mutations``' shared beta rows (the pass dropped, its partners
   swapped, its sign flipped, scaled by dtdx, run inside the curl, its increment
   unmasked, the ladder without dsigu or with its axes swapped, the auxiliary not
   advanced, the quarter turn reversed) and the harness-armed wrong-side
   coefficient words each must be caught; ``HAS_BETA = 0`` must DIVERGE from the
   beta array path (the reduction arm is discriminating, not decorative).

Correctness only. NOTHING HERE IS TIMED and no throughput claim is admissible.
Dispatch stays disabled: ``launch.plan_step`` does not know this module exists.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fold_pass_mutations as fold_passes  # noqa: E402
import probe_triton_complex_fused_electric_pair as twin_probe  # noqa: E402

SEED = 20260901

_TEMPORARY: List[str] = []

#: (name, dimensions, cell, boundaries, k_point, pml, beta, steps). Every case is
#: 2-D (Grid refuses beta anywhere else, grid.py:668-697) with a NONZERO beta and
#: complex storage. The sweep carries: the corpus family (in-plane Bloch kx on a
#: PML'd axis, the exact TestSpecialKz.test_special_kz shape class), a k = 0
#: complex case (force_complex_fields alone), METALLIC walls (the wall-clear and
#: ownership-mask arms), and both beta signs.
CASES: Tuple[Tuple[str, int, Tuple[float, float, float], Any,
                   Tuple[float, float, float], Dict[str, Any], float, int], ...] = (
    ("corpus_bloch_x", 2, (3.0, 3.0, 0.0), "periodic", (0.23, 0.0, 0.0),
     {"y": 5}, -0.39073112848927377, 10),
    ("bloch_x_positive_beta", 2, (3.0, 3.0, 0.0), "periodic", (0.31, 0.0, 0.0),
     {"y": 5}, 0.2, 10),
    ("k0_complex_storage", 2, (3.0, 3.0, 0.0), "periodic", (0.0, 0.0, 0.0),
     {"x": 5, "y": 5}, 0.2, 10),
    ("metallic_walls_k0", 2, (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
     (0.0, 0.0, 0.0), {"x": 5, "y": 5}, -0.685, 10),
    ("bloch_x_metallic_y", 2, (3.0, 3.0, 0.0),
     ("periodic", "metallic", "periodic"), (0.19, 0.0, 0.0), {"y": 5},
     -0.9120991827764708, 10),
)

#: The CARRY family: the same grids, run with a real electric deposit in the seam
#: and the SHIPPED repair bracket installed. THE PRODUCT IS WORTH NOTHING WITHOUT
#: THIS FAMILY: its one corpus row declares an electric source in this seam.
CARRY_CASES: Tuple[int, ...] = (0, 3, 4)

#: Steps per armed mutation.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3292-3304).
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


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and the material-write check would scan an empty set")


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Write the artifact atomically, STAMPED with the run's own import closure."""
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        _stamp_provenance(payload)
    except Exception as exc:  # noqa: BLE001 - a missing stamper must not lose the run
        payload["provenance_stamp_error"] = repr(exc)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def source_hashes() -> Dict[str, str]:
    """The exact bytes this gate binds itself to."""
    names = (
        "meep_gpu/triton_kernels/complex_beta_fused_electric_pair.py",
        "meep_gpu/triton_kernels/complex_fused_electric_pair.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/complex_fields.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/forms.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_complex_beta_fused_electric_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, read off the shipped source
# ---------------------------------------------------------------------------

def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed."""
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        {"bloch_pml_curl_step": "complex_fields.py",
                         "bloch_constitutive_step": "complex_fields.py",
                         "beta_bloch_pml_curl_step": "special_kz.py",
                         "complex_fused_curl_constitutive_D":
                             "complex_fused_electric_pair.py",
                         "complex_beta_fused_curl_constitutive_D":
                             "complex_beta_fused_electric_pair.py"}[name])
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
    return "\n".join(lines)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace one consecutive run of statements, matched on STRIPPED text."""
    lines = source.splitlines(keepends=True)
    stripped = [line.split("#", 1)[0].strip() for line in lines]
    target = [item.strip() for item in before]
    for start in range(len(lines) - len(target) + 1):
        if stripped[start:start + len(target)] != target:
            continue
        raw = lines[start]
        indent = raw[:len(raw) - len(raw.lstrip())]
        block = "".join(indent + item.strip() + "\n" for item in after)
        return ("".join(lines[:start]) + block
                + "".join(lines[start + len(target):]), 1)
    return source, 0


def _statements(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _shipped_body_statements(name: str) -> List[str]:
    """One shipped kernel's BODY statements — signature and docstring removed.

    The statement-list equalities below are SEQUENCE equalities, so the
    signature must not participate: the kernels legitimately differ there (the
    four beta words and ``HAS_BETA``), and the signature delta is asserted by
    the host suite, not re-litigated here.
    """
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        {"bloch_pml_curl_step": "complex_fields.py",
                         "bloch_constitutive_step": "complex_fields.py",
                         "beta_bloch_pml_curl_step": "special_kz.py",
                         "complex_fused_curl_constitutive_D":
                             "complex_fused_electric_pair.py",
                         "complex_beta_fused_curl_constitutive_D":
                             "complex_beta_fused_electric_pair.py"}[name])
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
        raise AssertionError(f"{name}'s body segments are unavailable")
    return _statements("\n".join(segments))


#: The flag that gates the beta pass. The pass itself is DERIVED from the shipped curl
#: (``special_kz.beta_bloch_pml_curl_step``'s top-level ``if HAS_BETA:`` statement),
#: never listed here: the transcription leg checks that this kernel splices that exact
#: statement after its six part forms (``fold_pass_mutations.body_and_insert``).
BETA_FLAG = "HAS_BETA"

#: Target 2's imaginary-part stencil line, the last of the curl half's stencils: the
#: previous placement of the beta term (inside the curl, before the forms) is planted
#: after it.
CURL_ANCHOR = "curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)"


def transcription_leg() -> Dict[str, Any]:
    """The fused body is the beta-less twin's plus EXACTLY the beta curl's pass.

    Three statement-list equalities, in order, against the shipped sources, and one
    position: the kernel minus its pass IS ``complex_fused_curl_constitutive_D``; its pass IS
    ``special_kz.beta_bloch_pml_curl_step``'s; that curl minus the same pass IS
    ``complex_fields.bloch_pml_curl_step``; and the pass follows the six part forms
    and precedes the wall clear (MEEP's beta pass after every component's curl,
    step_db.cpp:161-175). The kernel forms no complex product: the only complex
    device function it may call is the Bloch connection.
    """
    from meep_gpu.triton_kernels import complex_beta_fused_electric_pair as product  # noqa: PLC0415

    kernel_file = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                               "complex_beta_fused_electric_pair.py")
    special_py = os.path.join(API_ROOT, "meep_gpu", "triton_kernels", "special_kz.py")
    fused = _shipped_body_statements("complex_beta_fused_curl_constitutive_D")
    body, insert = fold_passes.body_and_insert(kernel_file, "complex_beta_fused_curl_constitutive_D", BETA_FLAG)
    twin = _shipped_body_statements("complex_fused_curl_constitutive_D")
    curl_body, curl_insert = fold_passes.body_and_insert(
        special_py, "beta_bloch_pml_curl_step", BETA_FLAG)
    bloch_curl = _shipped_body_statements("bloch_pml_curl_step")

    def first_difference(left: List[str], right: List[str]) -> str:
        index = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                     min(len(left), len(right)))
        return (f"first difference at statement {index}: "
                f"{left[index:index + 1]} vs {right[index:index + 1]}")

    findings: List[str] = []
    if not insert:
        findings.append("the kernel carries no top-level `if HAS_BETA:` pass")
    if body != twin:
        findings.append("the fused body minus its beta pass is not "
                        "complex_fused_curl_constitutive_D; " + first_difference(body, twin))
    if insert != curl_insert:
        findings.append("the beta pass is not beta_bloch_pml_curl_step's; "
                        + first_difference(insert, curl_insert))
    if curl_body != bloch_curl:
        findings.append("beta_bloch_pml_curl_step minus its pass is not "
                        "bloch_pml_curl_step; " + first_difference(curl_body, bloch_curl))
    shipped = set(curl_body) | set(curl_insert) | set(twin)
    invented = [line for line in fused if line not in shipped]
    if invented:
        findings.append(f"statements written here rather than transcribed: "
                        f"{invented}")

    # The licensed complex device functions, all called, and only them.
    helpers = sorted({name for name in product.MULTIPLY_HELPERS
                      if any(name in line for line in fused)})
    stray = sorted({line for line in fused
                    if ("_mul_" in line or "_rotate_" in line or "_connect(" in line)
                    and not any(name in line
                                for name in product.MULTIPLY_HELPERS)})
    if stray:
        findings.append(
            f"the fused body calls a complex device function outside "
            f"{product.MULTIPLY_HELPERS}: {stray}")
    if set(helpers) != set(product.MULTIPLY_HELPERS):
        findings.append(
            f"the fused body calls {helpers}, not all of "
            f"{list(product.MULTIPLY_HELPERS)}")

    # The D-side wall map, re-derived from IYEE_SHIFTS.
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    derived = tuple(
        tuple(index for index, target in enumerate(product.CURL_TARGETS)
              if IYEE_SHIFTS[target][axis] == 0)
        for axis in range(3))
    if derived != product.WALL_CLEARED_COMPONENTS:
        findings.append(
            f"stepping._zero_metal clears {derived} per walled axis, but the "
            f"module bakes {product.WALL_CLEARED_COMPONENTS}")

    if insert:
        before, after = fold_passes.insert_neighbours(kernel_file, "complex_beta_fused_curl_constitutive_D", BETA_FLAG)
        if not before.startswith("v2_im, n2_im = curl_form("):
            findings.append(f"the beta pass does not follow the last part form: it "
                            f"follows {before!r}")
        if after != "if ZM_X:":
            findings.append(f"the beta pass does not precede the wall clear: it "
                            f"precedes {after!r}")

    return {
        "leg": "transcription",
        "device": False,
        "fused_statements": len(fused),
        "beta_insert_statements": len(insert),
        "multiply_helpers_called": helpers,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — the design questions, on the array path's own arithmetic
# ---------------------------------------------------------------------------

def _complex_state(fields) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "fu_Dx", "fu_Dy", "fu_Dz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = np.ascontiguousarray(
                np.asarray(array)).view(np.uint32).ravel().copy()
    return out


def _build_complex(cell, boundaries, k_point, beta, thickness=0.4,
                   dimensions=2, seed=SEED):
    """A complex BETA Grid/Fields/PML triple on NumPy, seeded identically."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries, k_point=k_point, beta=beta)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    index = np.arange(int(np.prod(grid.shape)), dtype=np.float32).reshape(grid.shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    fields.set_epsilon_volumes(
        {component: epsilon for component in ("Ex", "Ey", "Ez")},
        {component: np.ascontiguousarray((1.0 / epsilon).astype(np.float32))
         for component in ("Ex", "Ey", "Ez")})
    pml = PML(grid=grid, thickness=thickness)
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "fu_Dx", "fu_Dy", "fu_Dz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = (rng.uniform(-0.25, 0.25, size=array.shape)
                      + 1j * rng.uniform(-0.25, 0.25, size=array.shape)
                      ).astype(array.dtype)
    return fields, pml


def _array_path_seam(fields, pml, driver_module) -> None:
    """The five driver passes this product replaces, in driver order. ``step_D``
    on a beta grid carries the beta increment inside it (S:438-445)."""
    driver_module.step_D(fields, pml)
    driver_module.fill_symmetry_bc_D(fields)
    driver_module.zero_metal_D(fields)
    driver_module.fill_folded_far_ghosts_D(fields)
    driver_module.update_E(fields, pml)


def _welded_seam(fields, pml, driver_module, *, wall: str = "before",
                 history: str = "read_then_write",
                 scale: str = "before_the_store") -> None:
    """The FUSED semantics, spelled with array ops: the beta-less twin's emulation.

    ``driver_module.step_D`` IS the array path and
    carries the beta pass, so the weld's design knobs are the same questions on a beta
    run, and the twin's per-cell forms (KT5's re-cut of this emulation) are imported
    rather than restated.
    """
    twin_probe._welded_seam(fields, pml, driver_module, wall=wall, history=history,
                            scale=scale)


def design_sweep_leg() -> Dict[str, Any]:
    """The weld's semantics against the array path over the whole beta sweep,
    plus the design knobs each of which must be VISIBLE."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, dimensions, cell, boundaries, k_point, _pml, beta, _steps in CASES:
        left_fields, left_pml = _build_complex(cell, boundaries, k_point, beta,
                                               dimensions=dimensions)
        right_fields, right_pml = _build_complex(cell, boundaries, k_point, beta,
                                                 dimensions=dimensions)
        for _ in range(2):
            _array_path_seam(left_fields, left_pml, driver_module)
            _welded_seam(right_fields, right_pml, driver_module)
        left, right = _complex_state(left_fields), _complex_state(right_fields)
        differing = sorted(item for item in left
                           if not np.array_equal(left[item], right[item]))
        moved_any = any(np.any(left[item] != 0) for item in left)
        rows.append({"case": name, "beta": beta, "arrays_differing": differing,
                     "identical": not differing, "state_is_live": moved_any})
        if differing:
            findings.append(f"{name}: the weld and the array path disagree on "
                            f"{differing}")
        if not moved_any:
            findings.append(f"{name}: the state is all zero, so this row compared "
                            f"nothing")

    knobs: List[Dict[str, Any]] = []
    for label, kwargs in (
        ("wall_clear_after_the_constitutive", {"wall": "after"}),
        ("wall_clear_dropped", {"wall": "dropped"}),
        ("history_written_before_it_is_read", {"history": "write_then_read"}),
        ("inv_eps_applied_after_the_history_store",
         {"scale": "after_the_store"}),
    ):
        boundaries = ("metallic", "metallic", "periodic")
        left_fields, left_pml = _build_complex((3.0, 3.0, 0.0), boundaries,
                                               (0.0, 0.0, 0.0), -0.685)
        right_fields, right_pml = _build_complex((3.0, 3.0, 0.0), boundaries,
                                                 (0.0, 0.0, 0.0), -0.685)
        # LATCHED PER STEP: a flip that differs at one step and is rewritten by the
        # next step's plain form (H or E = src outside the layer) is still visible.
        differing_set: set = set()
        for _ in range(2):
            _array_path_seam(left_fields, left_pml, driver_module)
            _welded_seam(right_fields, right_pml, driver_module, **kwargs)
            left, right = _complex_state(left_fields), _complex_state(right_fields)
            differing_set |= {name for name in left
                              if not np.array_equal(left[name], right[name])}
        differing = sorted(differing_set)
        knobs.append({"flip": label, "arrays_differing": differing,
                      "visible": bool(differing)})
        if not differing:
            findings.append(
                f"design flip {label!r} is INVISIBLE to the array path; the "
                f"kernel's choice there is constrained by nothing this gate "
                f"measures")

    # THE BETA TERM ITSELF MUST BE LIVE in this seam: the same grid stepped with
    # beta and with beta = 0 must differ, or every beta mutation below would be
    # scored on a term the cases never exercise.
    live_fields, live_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                           (0.23, 0.0, 0.0), -0.39073112848927377)
    zero_fields, zero_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                           (0.23, 0.0, 0.0), 0.0)
    _array_path_seam(live_fields, live_pml, driver_module)
    _array_path_seam(zero_fields, zero_pml, driver_module)
    left, right = _complex_state(live_fields), _complex_state(zero_fields)
    beta_moved = sorted(name for name in left
                        if not np.array_equal(left[name], right[name]))
    if not beta_moved:
        findings.append("beta = corpus and beta = 0 are byte-identical on the "
                        "array path here: the beta term is DEAD on these cases "
                        "and no beta mutation can measure anything")

    return {"leg": "design_sweep", "device": False, "rows": rows,
            "design_knobs": knobs, "beta_term_moves": beta_moved[:6],
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the predicate, clause by clause
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every clause of the seam predicate, exercised on real complex beta grids."""
    from meep_gpu.triton_kernels import complex_beta_fused_electric_pair as product  # noqa: PLC0415


    class _Source:
        def __init__(self, field_type: str) -> None:
            self.field_type = field_type

    def _deposit(fields, component):
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        source = VolumeSource(grid=fields.grid, component=component,
                              center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    def _with_flag_false(module, call):
        saved = module.CARRIES_DEPOSIT_REPAIR
        module.CARRIES_DEPOSIT_REPAIR = False
        try:
            return call()
        finally:
            module.CARRIES_DEPOSIT_REPAIR = saved

    def residual(verdict) -> List[str]:
        return [reason for reason in verdict.reasons
                if not ("array module is" in reason and "not cupy" in reason)]

    rows: List[Dict[str, Any]] = []

    def record(label: str, verdict, expect_admitted: bool, needle: str = "") -> None:
        left = residual(verdict)
        ok = (not left) if expect_admitted else bool(
            [reason for reason in left if needle in reason])
        rows.append({"case": label, "expect_admitted": expect_admitted,
                     "residual_reasons": left[:6], "needle": needle, "passed": ok})

    coverage = product.complex_beta_fused_electric_pair_coverage
    bloch, bloch_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                      (0.23, 0.0, 0.0),
                                      -0.39073112848927377)
    record("corpus_no_sources", coverage(bloch, bloch_pml, ()),
           True)
    record("magnetic_source_is_the_other_seams",
           coverage(bloch, bloch_pml, (_Source("B"),)), True)
    record("electric_deposit_is_carried",
           coverage(bloch, bloch_pml, (_deposit(bloch, "Ey"),)),
           True)
    record("electric_source_without_a_deposit_index",
           coverage(bloch, bloch_pml, (_Source("D"),)),
           False, "does not publish the index it writes")
    record("electric_deposit_refused_when_the_flag_is_held_False",
           _with_flag_false(
               product,
               lambda: coverage(bloch, bloch_pml, (_deposit(bloch, "Ey"),))),
           False, "is electric")
    record("undeclared_sources", coverage(bloch, bloch_pml, None),
           False, "was not declared")
    record("no_pml", coverage(bloch, None, ()), False, "PML")

    zero, zero_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                    (0.23, 0.0, 0.0), 0.0)
    record("beta_zero_is_the_plain_complex_pairs",
           coverage(zero, zero_pml, ()), False, "beta is zero")

    walls, walls_pml = _build_complex(
        (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
        (0.0, 0.0, 0.0), -0.685)
    record("metallic_walls_k0", coverage(walls, walls_pml, ()),
           True)

    builder_rows = []
    for label, args, kwargs in (
        ("electric_source_without_a_deposit_index",
         (bloch, bloch_pml, (_Source("D"),)), {}),
        ("undeclared_sources", (bloch, bloch_pml, None), {}),
        ("beta_zero", (zero, zero_pml, ()), {}),
    ):
        plan = product.plan_complex_beta_fused_electric_pair(*args, **kwargs)
        builder_rows.append({"case": label, "plan_is_none": plan is None,
                             "passed": plan is None})

    findings = [row["case"] for row in rows if not row["passed"]]
    findings += [f"builder:{row['case']}" for row in builder_rows
                 if not row["passed"]]
    return {"leg": "predicate", "device": False, "rows": rows,
            "builder_rows": builder_rows, "findings": findings,
            "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — the seam passes are the driver's own
# ---------------------------------------------------------------------------

def seam_binding_leg() -> Dict[str, Any]:
    """``REPLACES`` must name the call sites ``driver.step`` actually runs there.

    The seam's extent is DERIVED from :data:`fastpath.DRIVER_SLOTS`, not counted
    in lines. It was a literal 16-line window until 2026-09-02, when the
    dispatch surface grew the two fill consults — ``fill_B``/``fill_D`` joined
    ``DRIVER_SLOTS`` and the two folded-far passes gained consults of their own —
    which moved ``update_E`` out of a fixed count of lines below ``step_D`` and
    read as "the seam is not where it was". The seam is now walked consult by
    consult: it runs from the ``step_D`` consult to the ``update_E`` consult
    that closes it, and reaching a consult this seam does NOT own before that
    closer is itself the finding. That asserts strictly more than the line count
    did — a foreign slot dispatched inside the seam is now caught, where a window
    could only notice the two endpoints drifting apart.
    """
    from meep_gpu import fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_beta_fused_electric_pair as product  # noqa: PLC0415

    opens, fill, closes = "step_D", "fill_D", "update_E"
    # The consults this seam owns: its curl slot, its fill slot, the far-ghost
    # pass that fill slot's second half stands in for, and the constitutive slot
    # that closes it. Every other name in DRIVER_SLOTS is foreign to this seam.
    owned = {opens, fill, closes, fastpath.FAR_FILL_PASSES[fill]}
    consults = tuple(fastpath.DRIVER_SLOTS) + tuple(fastpath.FAR_FILL_PASSES.values())
    foreign = [name for name in consults if name not in owned]

    def consulted(item: str) -> Optional[str]:
        return next((name for name in consults if f'dispatch("{name}"' in item), None)

    text = open(os.path.join(API_ROOT, "meep_gpu", "driver.py"),
                encoding="utf-8").read().splitlines()
    findings: List[str] = []
    windows: List[Dict[str, Any]] = []
    for index, line in enumerate(text):
        if f'dispatch("{opens}"' not in line:
            continue
        end: Optional[int] = None
        trespass: Optional[Tuple[int, str]] = None
        for offset, item in enumerate(text[index + 1:], start=1):
            name = consulted(item)
            if name is None:
                continue
            if name == closes:
                end = offset
                break
            if name not in owned:
                trespass = (index + 1 + offset, name)
                break
        if end is None:
            if trespass is not None:
                findings.append(
                    f"driver.py:{index + 1} dispatches {opens} and reaches the "
                    f"{trespass[1]} consult at driver.py:{trespass[0]} before any "
                    f"{closes}; the seam this product spans is not where it was")
            else:
                findings.append(
                    f"driver.py:{index + 1} dispatches {opens} with no {closes} "
                    f"consult after it; the seam this product spans is not where "
                    f"it was")
            continue
        body = text[index:index + end + 1]
        calls = [name for name in
                 ("fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D")
                 if any(name + "(" in item for item in body)]
        injects = any("source.inject" in item or "_inject_electric" in item
                      for item in body)
        windows.append({"driver_line": index + 1, "passes_between": calls,
                        "closes_at_line": index + 1 + end,
                        "injects_a_source": injects})
        expected = [name for name in product.REPLACES
                    if name not in ("step_D", "update_E")]
        if calls != expected:
            findings.append(
                f"driver.py:{index + 1} runs {calls} between step_D and update_E, "
                f"but the product declares it replaces {expected}")
        if not injects:
            findings.append(
                f"driver.py:{index + 1} shows no source injection in this seam")
    if not windows:
        findings.append("no step_D/update_E seam was found in driver.py at all")
    return {"leg": "seam_binding", "device": False, "windows": windows,
            "declared_replaces": list(product.REPLACES),
            "seam_owns": sorted(owned), "foreign_consults": foreign,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side — drivers, inventory, launch counting
# ---------------------------------------------------------------------------

def build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec, beta,
                 seed: int, electric: bool):
    """One complex-storage BETA PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35,  # non-power-of-two, deliberate
        boundaries=boundaries, k_point=k_point, beta=beta,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml(dict(pml_spec))
    if electric:
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        host = (rng.uniform(-0.25, 0.25, size=shape)
                + 1j * rng.uniform(-0.25, 0.25, size=shape)).astype(np.complex64)
        driver.set_field(name, cp.asarray(np.ascontiguousarray(host)))
    # f_w_E IS SEEDED so the history-ordering mutation is armed from step 1.
    #
    # H IS SEEDED TOO, AND IT IS NOT DECORATION. H is this seam's beta PARTNER,
    # and the ownership mask fires only on METALLIC axes — where the wall clears
    # pin the wall row's B (and with it an unseeded H and f_w_H) to exact zero
    # from the first step onward. Measured on the first device run of this gate
    # (2026-09-01, metallic_walls_k0): with H unseeded,
    # m16_beta_below_the_ownership_mask multiplied an exact +0 partner at every
    # masked cell and reported UNCAUGHT over all three steps — a statement about
    # the fixture, not the weld. Seeded, the masked rows carry a nonzero partner
    # on step 1 and the moved insert is byte-visible.
    for name, scale in (("f_w_Ex", 0.05), ("f_w_Ey", 0.05), ("f_w_Ez", 0.05),
                        ("Hx", 0.10), ("Hy", 0.10), ("Hz", 0.10)):
        array = getattr(driver.fields, name, None)
        if array is not None:
            host = (rng.uniform(-scale, scale, size=shape)
                    + 1j * rng.uniform(-scale, scale, size=shape)
                    ).astype(np.complex64)
            array[...] = cp.asarray(np.ascontiguousarray(host))
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
    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the five seam passes for the fields objects named in ``routes``."""
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
    """The two separately certified BETA singles this launch replaces."""
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    curl = special_kz.plan_beta_bloch_pml_curl(driver.fields, driver.pml,
                                               "step_D")
    constitutive = special_kz.plan_beta_run_complex_constitutive(
        driver.fields, driver.pml, "E")
    missing = [name for name, plan in
               (("beta bloch curl", curl),
                ("beta complex constitutive", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    return Route({"step_D": curl, "update_E": constitutive})


def fused_route(plan, driver=None, sources=()):
    """The fused plan in its two slots, BRACKETED when the seam carries a deposit."""
    from meep_gpu import deposit_repair  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "D")
    if not seam:
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


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, dimensions, cell, boundaries, k_point, pml_spec,
            beta, steps: int, product, mutant: Any = None,
            install_fused: bool = True, freeze_electric: bool = False,
            electric: bool = False,
            has_beta: Optional[int] = None,
            beta_words_override: Any = None) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                             beta, SEED, electric)
    separate = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                            beta, SEED, electric)
    fused = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                         beta, SEED, electric)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.complex_beta_fused_curl_constitutive_D_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": str(boundaries),
        "k_point": list(k_point), "pml": dict(pml_spec), "beta": beta,
        "electric_source_in_the_seam": bool(electric),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "has_beta_override": has_beta,
        "beta_words_override": (None if beta_words_override is None
                                else list(beta_words_override)),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_complex_beta_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        if plan is None:
            verdict = product.complex_beta_fused_electric_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if has_beta is not None:
            plan.has_beta = int(has_beta)
        if beta_words_override is not None:
            plan.beta_words = tuple(float(word) for word in beta_words_override)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["beta_words"] = list(plan.beta_words)
        if install_fused:
            route, leading = fused_route(plan, fused, tuple(fused._sources))
        else:
            route, leading = Route({}), None
        row["bracketed"] = leading is not None
        row["slot_classes"] = {key: type(value).__name__
                               for key, value in route.plans.items()}
        if electric and install_fused and leading is None:
            raise AssertionError(
                "the case declares an in-seam electric source but the bracket was "
                "not installed; this leg would measure an unbracketed launch and "
                "report it as the shipped composition")
        separate_plans = separate_route(separate)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_electric:
            frozen = {key: _Absorbed(key, None) for key in SEAM_PASSES}
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
                "deposit_repairs": (leading.repairs if leading is not None else None),
            })
            divergence = versus_array or versus_separate
            log(f"  {name} step {step}/{steps} identical={divergence is None} "
                f"control={control is None} moved={len(step_moved)} "
                f"launches={kernel.calls} "
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
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
        row["arrays_ever_moved"] = len(ever_moved)
        # THE AUXILIARY CONTRACT'S EXEMPTION. fu_* and f_w_* move only on their
        # layer's cells (KERNELS.md section 1, C2), so on a 2-D grid the invariant
        # axis's auxiliaries are inert on EVERY route. An auxiliary the array path
        # also leaves inert is exempt; every other unmoved array still refuses.
        reference_moved = moved(reference_opening, snapshot(cp, reference))
        unmoved = set(final) - set(ever_moved) - set(material)
        row["inert_on_every_route"] = sorted(
            name for name in unmoved - set(reference_moved)
            if name.startswith(("fu_", "f_w_")))
        row["arrays_never_moved"] = sorted(unmoved - set(row["inert_on_every_route"]))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["deposit_repairs"] = leading.repairs if leading is not None else None
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        return row
    finally:
        undo()
        for target in (reference, separate, fused):
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False) -> Tuple[bool, List[str]]:
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
    if require_repairs and not row.get("deposit_repairs"):
        failures.append(
            "the carry leg repaired NO deposit cell: the bracket ran but the seam "
            "carried nothing, so this leg measured the quiet case under a carry name")
    fell_back = {key: value for key, value in (row.get("launches") or {}).items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: {fell_back}")
    return (not failures), failures


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def shipped_source(product) -> str:
    return textwrap.dedent(
        inspect.getsource(product.complex_beta_fused_curl_constitutive_D.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest.

    The connection and the shared forms are IMPORTED into the mutant's namespace
    rather than copied (``fold_pass_mutations.MUTANT_IMPORTS``), as the shipped module
    imports them. THE MODULE NAME MUST NOT START WITH ``triton``: Triton's dependency
    finder leaves every callee whose ``__module__`` starts with its package name out of
    the kernel's cache key.
    """
    header = ("import triton\n"
              "import triton.language as tl\n"
              "from meep_gpu.triton_kernels.complex_fields import _connect\n"
              + fold_passes.MUTANT_IMPORTS
              + "PERIODIC = tl.constexpr(0)\n"
              "METALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_complex_beta_electric_pair.py", delete=False,
        encoding="utf-8")
    handle.write(header + source.replace("complex_beta_fused_curl_constitutive_D", kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "complex_beta_electric_pair_mutant_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


#: What each case's grid carries (by ``CASES`` index), in ``fold_pass_mutations.NEEDS``
#: words: a PML layer on x (the dsigu axis of the beta pass's target 1) and a metallic
#: axis (the increment's ownership mask).
CASE_CARRIES: Dict[int, Tuple[str, ...]] = {
    0: (), 1: (), 2: ("layer",), 3: ("wall", "layer"), 4: ("wall",),
}

#: The beta pass's own mutations: ``fold_pass_mutations``' shared spellings, the same
#: text in every carrier of the pass. The root forms' battery and the twin's own rows
#: come from the twin probe's table.
SHARED_MUTATIONS = fold_passes.declare(
    fold_passes.beta_mutations(True, False, CURL_ANCHOR),
    {"b_increment_unmasked": (
        "null",
        "DECLARED NULL on the D seam: every plane the increment's mask drops (Dx at a "
        "y wall, Dy at an x wall) is a plane zero_metal_D clears in the same launch, "
        "and fu_D is not stored there because cell 0 of the integer sub-lattice is in "
        "no chunk on a non-wrapping axis; the unmasked increment reaches no stored "
        "word. The B seam's twin is must-catch.")})
_SHARED_ROWS, MUTATION_NEEDS = fold_passes.table(SHARED_MUTATIONS)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite).

    THE TWIN'S TABLE FIRST: this kernel's body IS ``complex_fused_curl_constitutive_D``'s (the
    transcription leg), so its rewrites -- the wall clear, the Bloch connection, the
    inversions and the chunk-form battery, the f_w order and the weld's registers --
    are the twin probe's own, imported from ``probe_triton_complex_fused_electric_pair``, and score the same
    lines here. Then the beta pass's shared mutations, from
    :data:`SHARED_MUTATIONS`.
    """

    return tuple(twin_probe.mutation_table()) + (
    ) + _SHARED_ROWS


#: Mutation id -> the CASES index whose grid carries the branch it rewrites: the twin
#: probe's own choices for its rows, and for the beta pass's rows the first case that
#: carries what each needs (``fold_pass_mutations.cases_for``).
MUTATION_CASE: Dict[str, int] = {
    **twin_probe.MUTATION_CASE,
    **fold_passes.cases_for(MUTATION_NEEDS, CASE_CARRIES, 4),
}

#: The phased, walled case: nonzero k for the phase legs, a y wall for ZM_Y.
DEFAULT_MUTATION_CASE = 4


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    index = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    fold_passes.check_needs(name, MUTATION_NEEDS.get(name, ()), CASE_CARRIES[index],
                            CASES[index][0])
    return index, CASES[index]


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source(product)
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        mutated, hits = rewrite(source)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "device": True}
        if hits == 0 or mutated == source:
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_complex_beta_fused_D"
        try:
            mutant = compile_mutant(mutated, kernel_name)
        except Exception as exc:  # noqa: BLE001 - an unimportable mutant is a harness fact
            row["error"] = f"the mutant did not import: {exc!r}"
            rows.append(row)
            log(f"  mutation {name}: DID NOT IMPORT ({exc!r})")
            continue
        case_index, case = mutation_case_for(name)
        row["case"] = case[0]
        row["case_index"] = case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3], case[4],
                      case[5], case[6], MUTATION_STEPS, product,
                      mutant=mutant)
        row["caught"] = leg.get("first_divergence") is not None
        row["first_divergence"] = leg.get("first_divergence")
        row["launches"] = leg.get("fused_kernel_launches")
        row["ptx_moved"] = sorted(kernel_ptx(mutant)) != sorted(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} on {row['case']} "
            f"launches={row['launches']} expectation={expectation}")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse, asked of a real CuPy driver."""
    from meep_gpu.sources import FIELD_TYPE_B  # noqa: PLC0415

    class _Magnetic:
        field_type = FIELD_TYPE_B

    class _Electric:
        field_type = "D"

    def _deposit(fields, component):
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        source = VolumeSource(grid=fields.grid, component=component,
                              center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    base = CASES[0]
    rows: List[Dict[str, Any]] = []
    for name, case, beta_override, needle, sources in (
        ("electric_source_without_a_deposit_index", base, None,
         "does not publish the index it writes", (_Electric(),)),
        ("electric_deposit_is_carried", base, None, None,
         lambda fields: (_deposit(fields, "Ey"),)),
        ("magnetic_source_is_the_other_seams", base, None, None,
         (_Magnetic(),)),
        ("undeclared_sources", base, None, "was not declared", None),
        ("beta_zero_is_the_plain_complex_pairs", base, 0.0, "beta is zero",
         ()),
    ):
        beta = case[6] if beta_override is None else beta_override
        driver = build_driver(cp, case[1], case[2], case[3], case[4], case[5],
                              beta, SEED, electric=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.complex_beta_fused_electric_pair_coverage(
                driver.fields, driver.pml, bound)
            plan = product.plan_complex_beta_fused_electric_pair(
                driver.fields, driver.pml, bound)
            if needle is None:
                passed = bool(verdict.covered) and plan is not None
            else:
                passed = ((not verdict.covered) and plan is None
                          and any(needle in reason for reason in verdict.reasons))
            rows.append({
                "case": name, "device": True, "covered": bool(verdict.covered),
                "expected": "admitted" if needle is None else f"refused: {needle}",
                "reasons": list(verdict.reasons)[:6], "plan_is_none": plan is None,
                "passed": passed,
            })
        finally:
            driver.close()
            cp.get_default_memory_pool().free_all_blocks()
        log(f"  refusal {name}: covered={rows[-1]['covered']} "
            f"passed={rows[-1]['passed']}")
    return rows


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def environment(cp: Any = None) -> Dict[str, Any]:
    """What ran this, in the shape the WELD PIPELINE reads."""
    import socket  # noqa: PLC0415

    payload: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }
    try:
        import triton  # noqa: PLC0415
        payload["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        payload["triton"] = None
    if cp is not None:
        try:
            payload["cupy"] = cp.__version__
            properties = cp.cuda.runtime.getDeviceProperties(0)
            payload["device"] = properties["name"].decode()
            payload["compute_capability"] = (
                f"{properties['major']}.{properties['minor']}")
        except Exception:  # noqa: BLE001
            pass
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_complex_beta_fused_electric_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep'.")
    args = parser.parse_args(argv)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_complex_beta_fused_electric_pair",
        "product": "meep_gpu.triton_kernels.complex_beta_fused_electric_pair",
        "kernel": "complex_beta_fused_curl_constitutive_D",
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": triton_device_identity.record(environment()),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "complex_fields.DEFAULT_BLOCK",
                   "subnormal_policy": args.subnormal_policy},
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in (transcription_leg, design_sweep_leg, predicate_leg,
                seam_binding_leg):
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

    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_beta_fused_electric_pair as product  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "quiet_cases": {case[0]: case[7] for case in CASES},
        "carry_cases": [CASES[index][0] for index in CARRY_CASES],
        "steps_per_mutation": MUTATION_STEPS,
    }
    save(payload, args.out)

    log("\n=== device legs: the QUIET family (no in-seam source) ===")
    for name, dimensions, cell, boundaries, k_point, pml_spec, beta, steps in CASES:
        row = run_leg(cp, f"quiet:{name}", dimensions, cell, boundaries, k_point,
                      pml_spec, beta, steps, product, electric=False)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== device legs: the CARRY family (electric deposit, shipped bracket) ===")
    for index in CARRY_CASES:
        name, dimensions, cell, boundaries, k_point, pml_spec, beta, steps = CASES[index]
        row = run_leg(cp, f"carry:{name}", dimensions, cell, boundaries, k_point,
                      pml_spec, beta, steps, product, electric=True)
        passed, failures = verdict_of(row, require_launches=steps,
                                      require_repairs=True)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== null controls: the carry bracket REMOVED must diverge ===")
    for index in CARRY_CASES[:2]:
        name, dimensions, cell, boundaries, k_point, pml_spec, beta, _steps = CASES[index]
        payload["device_legs"].append(_null_unbracketed(
            cp, f"null:unbracketed:{name}", dimensions, cell, boundaries,
            k_point, pml_spec, beta, product))
        save(payload, args.out)

    log("\n=== armed harness legs ===")
    base = CASES[0]
    row = run_leg(cp, "armed:no_substitution", base[1], base[2], base[3], base[4],
                  base[5], base[6], 3, product, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    # THE REDUCTION ARM IS DISCRIMINATING: HAS_BETA = 0 on a beta run must
    # diverge from the beta array path.
    row = run_leg(cp, "armed:has_beta_zero", base[1], base[2], base[3], base[4],
                  base[5], base[6], 3, product, has_beta=0)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = ("HAS_BETA = 0 launches the plain complex weld on a beta run; "
                  "the bytes must diverge, or the beta arm is decorative on "
                  "these cases")
    payload["device_legs"].append(row)
    save(payload, args.out)

    # THE COEFFICIENT WORDS ARE LOAD-BEARING: the MAGNETIC side's +1j pair on
    # this electric weld must diverge (the array path binds -1j here, S:771-772).
    magnetic_words = special_kz.beta_curl_coefficients(
        base[6], 0.35 / 12.0, magnetic=True, complex_storage=True)
    row = run_leg(cp, "armed:magnetic_coefficient_words", base[1], base[2],
                  base[3], base[4], base[5], base[6], 3, product,
                  beta_words_override=magnetic_words)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = ("the +1j (magnetic) coefficient pair on the electric weld must "
                  "diverge; the sign lives in the words and the words are "
                  "load-bearing")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.complex_beta_fused_curl_constitutive_D)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    def _mutation_ok(row: Dict[str, Any]) -> bool:
        if row.get("error") is not None:
            return False
        if not row.get("rewrite_hits"):
            return False
        return bool(row.get("caught")) is (row["expectation"] == "caught")

    mutation_ok = all(_mutation_ok(row) for row in payload["mutations"])
    payload["passed"] = bool(
        device_ok and refusal_ok and mutation_ok
        and all(row["passed"] for row in payload["no_device_legs"]))
    payload["device_status"] = "RUN"
    payload["release"] = {
        "released": payload["passed"],
        "reasons": ([] if payload["passed"] else
                    [f"device legs ok: {device_ok}",
                     f"refusals ok: {refusal_ok}",
                     f"mutations ok: {mutation_ok}"]),
    }
    save(payload, args.out)
    for name in _TEMPORARY:
        try:
            os.unlink(name)
        except OSError:
            pass
    log(f"\nverdict: {payload['passed']}  ->  {args.out}")
    return 0 if payload["passed"] else 1


def _null_unbracketed(cp, name, dimensions, cell, boundaries, k_point, pml_spec,
                      beta, product) -> Dict[str, Any]:
    """The NULL CONTROL: the weld substituted WITHOUT the repair bracket, on a
    case with an in-seam electric deposit. It MUST diverge — a null that agrees
    says the repair repairs nothing on this case and the carry legs above were
    measuring the quiet case under a carry name."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                             beta, SEED, electric=True)
    fused = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                         beta, SEED, electric=True)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(product.complex_beta_fused_curl_constitutive_D_kernel())
    row: Dict[str, Any] = {"leg": name, "device": True, "null_control": True,
                           "beta": beta}
    try:
        plan = product.plan_complex_beta_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        if plan is None:
            raise AssertionError("the product refused the null control's case")
        route = Route({key: (plan if key == "step_D" else _Absorbed(key, plan))
                       for key in SEAM_PASSES})
        roles = {id(reference.fields): "array", id(fused.fields): "fused"}
        undo = install(driver_module, [(fused.fields, route)], counter, roles)
        divergence = None
        for step in range(1, 4):
            reference.step()
            fused.step()
            cp.cuda.runtime.deviceSynchronize()
            divergence = first_divergence(snapshot(cp, fused),
                                          snapshot(cp, reference))
            if divergence is not None:
                row["diverged_at_step"] = step
                break
        row["first_divergence"] = divergence
        row["fused_kernel_launches"] = kernel.calls
        row["passed"] = divergence is not None and kernel.calls > 0
        row["why"] = ("the bracket is REMOVED while the deposit is live; bytes "
                      "must diverge or the repair repairs nothing here")
        log(f"  {name}: diverged={divergence is not None} "
            f"launches={kernel.calls}")
        return row
    finally:
        undo()
        for target in (reference, fused):
            try:
                target.close()
            except Exception:  # noqa: BLE001
                pass
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    raise SystemExit(main())
