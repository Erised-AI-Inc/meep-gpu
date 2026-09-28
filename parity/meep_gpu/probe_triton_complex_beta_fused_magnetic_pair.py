#!/usr/bin/env python3
"""THE COMPLEX BETA FUSED MAGNETIC PAIR GATE — complex beta ``step_B`` welded
into complex ``update_H``.

Does ONE launch of ``complex_beta_fused_curl_constitutive_B`` leave the engine
BIT-IDENTICAL, per COMPLETE driver step, to both the CuPy array path and the two
separately certified Triton products it replaces (the K2 beta bloch curl on
``step_B`` and the complex constitutive on a beta run's H side) — over every
allocated volume, compared as uint32 words?

THE ELECTRIC TWIN'S GATE (``probe_triton_complex_beta_fused_electric_pair.py``)
is the template; the differences are the B side's own:

* the seam is ``step_B -> magnetic sources -> ... -> update_H`` and the CARRY
  family repairs a real MAGNETIC deposit through the shipped bracket;
* the wall clear takes the B-SIDE map — an x wall clears ``Bx`` alone, the exact
  complement of the D side's tangential pair;
* there is no ``SCALE`` arm and no ``inv_eps`` anywhere in the kernel, so the
  D-side scale legs and mutations have no counterpart here;
* the beta coefficient words are the ``+1j`` pair (S:771-772), and the armed
  wrong-words leg forces the ELECTRIC ``-1j`` pair instead.

Correctness only. NOTHING HERE IS TIMED and no throughput claim is admissible.
Dispatch stays disabled: ``launch.plan_step`` does not know this module exists.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import copy
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

SEED = 20260902

_TEMPORARY: List[str] = []

EXTENDED_PROBE = ("parity/meep_gpu/results/"
                  "complex_expansion_beta_extension_2026-09-01/probe_keep/"
                  "probe.json")

BASE_FOUR_PROBE = ("parity/meep_gpu/results/"
                   "complex_expansion_convention_2026-08-16/results/probe_keep/"
                   "probe.json")

#: (name, dimensions, cell, boundaries, k_point, pml, beta, steps) — the same
#: sweep as the electric twin's, on the same beta values.
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

#: The CARRY family: a real MAGNETIC deposit in this seam, shipped bracket on.
#: The corpus row this product serves has a QUIET magnetic seam, so the carry
#: legs are what stand behind CARRIES_DEPOSIT_REPAIR = True rather than behind
#: any corpus row — the flag must be measured, not inherited.
CARRY_CASES: Tuple[int, ...] = (0, 3)

MUTATION_STEPS = 3

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


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}")


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    try:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415

        _stamp_provenance(payload)
    except Exception as exc:  # noqa: BLE001
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
    names = (
        "meep_gpu/triton_kernels/complex_beta_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/complex_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/complex_fields.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_complex_beta_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription
# ---------------------------------------------------------------------------

def _statements(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _shipped_body_statements(name: str) -> List[str]:
    """One shipped kernel's BODY statements — signature and docstring removed."""
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        {"bloch_pml_curl_step": "complex_fields.py",
                         "beta_bloch_pml_curl_step": "special_kz.py",
                         "complex_fused_curl_constitutive_B":
                             "complex_fused_magnetic_pair.py",
                         "complex_beta_fused_curl_constitutive_B":
                             "complex_beta_fused_magnetic_pair.py"}[name])
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


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
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


BETA_INSERT = (
    "if HAS_BETA:",
    "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
    "curl0_re = curl0_re - t_re",
    "curl0_im = curl0_im - t_im",
    "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
    "curl1_re = curl1_re - t_re",
    "curl1_im = curl1_im - t_im",
)


def transcription_leg() -> Dict[str, Any]:
    """The fused body is the beta-less twin's plus EXACTLY K2's delta."""
    from meep_gpu.triton_kernels import complex_beta_fused_magnetic_pair as product  # noqa: PLC0415

    fused = _shipped_body_statements("complex_beta_fused_curl_constitutive_B")
    twin = _shipped_body_statements("complex_fused_curl_constitutive_B")
    beta_curl = _shipped_body_statements("beta_bloch_pml_curl_step")
    bloch_curl = _shipped_body_statements("bloch_pml_curl_step")

    findings: List[str] = []

    without_beta = [line for line in fused if line not in BETA_INSERT]
    if without_beta != twin:
        index = next((i for i, (a, b) in enumerate(zip(without_beta, twin))
                      if a != b), "length")
        findings.append(
            f"the fused body minus the beta insert is not "
            f"complex_fused_curl_constitutive_B; first difference at {index}")

    delta = [line for line in beta_curl if line not in bloch_curl]
    if delta != list(BETA_INSERT):
        findings.append(
            f"the shipped beta curl's delta over bloch_pml_curl_step is {delta}, "
            f"not the insert this gate arms")

    weld = [line for line in twin if line not in bloch_curl]
    if not weld:
        findings.append("the derived weld set is empty; this leg compares nothing")
    without_weld = [line for line in fused if line not in weld]
    if without_weld != beta_curl:
        index = next((i for i, (a, b) in enumerate(zip(without_weld, beta_curl))
                      if a != b), "length")
        findings.append(
            f"the fused body minus the weld is not beta_bloch_pml_curl_step; "
            f"first difference at {index}")

    shipped = set(beta_curl) | set(twin)
    invented = [line for line in fused if line not in shipped]
    if invented:
        findings.append(f"statements written here rather than transcribed: "
                        f"{invented}")

    helpers = sorted({name for name in product.LICENSED_MULTIPLY_HELPERS
                      if any(name in line for line in fused)})
    stray = sorted({line for line in fused
                    if ("_mul_" in line or "_rotate_" in line)
                    and not any(name in line
                                for name in product.LICENSED_MULTIPLY_HELPERS)})
    if stray:
        findings.append(
            f"the fused body calls a complex-multiply helper outside "
            f"{product.LICENSED_MULTIPLY_HELPERS}: {stray}")
    if set(helpers) != set(product.LICENSED_MULTIPLY_HELPERS):
        findings.append(
            f"the fused body calls {helpers}, not all of "
            f"{list(product.LICENSED_MULTIPLY_HELPERS)}")

    # The B-SIDE wall map: axis d clears component d, and only it. Derived from
    # IYEE_SHIFTS over the B targets, then checked against the body's ZM blocks.
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import SUB_STEPS  # noqa: PLC0415

    targets = tuple(SUB_STEPS["step_B"]["targets"])
    derived = tuple(
        tuple(index for index, target in enumerate(targets)
              if IYEE_SHIFTS[target][axis] == 0)
        for axis in range(3))
    if derived != ((0,), (1,), (2,)):
        findings.append(
            f"stepping._zero_metal clears {derived} per walled axis on the B "
            f"side; this gate's map expectation has drifted")
    for axis, coordinate in enumerate(("at_x", "at_y", "at_z")):
        for component in range(3):
            cleared = any(
                f"v{component}_{plane} = tl.where({coordinate}, 0.0, "
                f"v{component}_{plane})" in line
                for plane in ("re", "im") for line in fused)
            expected = component in derived[axis]
            if cleared is not expected:
                findings.append(
                    f"the fused body {'clears' if cleared else 'does not clear'} "
                    f"component {component} at {coordinate}, but _zero_metal's "
                    f"rule says {'it should' if expected else 'it should not'}")

    curl_index = fused.index(
        "curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)")
    insert_index = fused.index("if HAS_BETA:")
    mask_index = fused.index("at_x, at_y, at_z = i == 0, j == 0, k == 0")
    if not (curl_index < insert_index < mask_index):
        findings.append(
            f"the beta insert is not between the dtdx curl ({curl_index}) and "
            f"the ownership mask ({mask_index}): found at {insert_index}")

    return {
        "leg": "transcription",
        "device": False,
        "fused_statements": len(fused),
        "beta_insert_statements": len(BETA_INSERT),
        "multiply_helpers_called": helpers,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the EXTENDED expansion licence
# ---------------------------------------------------------------------------

def expansion_licence_leg() -> Dict[str, Any]:
    from meep_gpu.triton_kernels import complex_fields as complex_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_beta_fused_magnetic_pair as product  # noqa: PLC0415

    patterns = list(product.PRODUCT_PROBE_PATTERNS)
    findings: List[str] = []
    if tuple(patterns) != tuple(special_kz.BETA_PROBE_PATTERNS):
        findings.append(
            f"this product declares pattern set {patterns}, not the beta "
            f"tranche's extended set {list(special_kz.BETA_PROBE_PATTERNS)}")

    extended = json.load(open(os.path.join(API_ROOT, EXTENDED_PROBE),
                              encoding="utf-8"))
    verdict = complex_module.expansion_license(extended, patterns)
    licensed = {
        "artifact": EXTENDED_PROBE,
        "arm": verdict["arm"], "basis": verdict["basis"],
        "expansion": verdict["expansion"],
        "refusals": list(verdict["refusals"]),
        "non_discriminating": list(verdict["non_discriminating"]),
        "policy_resolved": verdict["policy_resolved"],
        "candidate_policy": verdict["candidate_policy"],
    }
    if verdict["arm"] is None:
        findings.append(f"{EXTENDED_PROBE} licenses nothing: {verdict['refusals']}")
    elif verdict["basis"] != "measured":
        findings.append(
            f"{EXTENDED_PROBE} licenses {verdict['arm']} on basis "
            f"{verdict['basis']!r}; this product's arm must be MEASURED")

    base = json.load(open(os.path.join(API_ROOT, BASE_FOUR_PROBE),
                          encoding="utf-8"))
    refused = complex_module.expansion_license(base, patterns)
    base_row = {"artifact": BASE_FOUR_PROBE, "arm": refused["arm"],
                "refusals": list(refused["refusals"])[:3]}
    if refused["arm"] is not None:
        findings.append(
            f"the STANDING base-four artifact licenses {refused['arm']} over the "
            f"extended set; the whole premise of this round is that it cannot")

    falsification: List[Dict[str, Any]] = []
    for name, broken, needle in (
        ("fifth_pattern_removed",
         (lambda r: (r["patterns"].pop(special_kz.BETA_PROBE_PATTERN, None), r)[1])(
             copy.deepcopy(extended)), "not classified"),
        ("candidates_cut_under_another_policy",
         (lambda r: (r.setdefault("candidates", {}).__setitem__("policy", "flush"),
                     r)[1])(copy.deepcopy(extended)), "broken comparison"),
        ("host_backend",
         {**copy.deepcopy(extended), "backend": "numpy"}, "cupy"),
    ):
        v = complex_module.expansion_license(broken, patterns)
        ok = v["arm"] is None and any(needle in reason for reason in v["refusals"])
        falsification.append({"case": name, "arm": v["arm"],
                              "refusals": list(v["refusals"])[:2], "passed": ok})
        if not ok:
            findings.append(f"falsification row {name!r} was NOT refused by name")

    return {
        "leg": "expansion_licence",
        "device": False,
        "probe_patterns": patterns,
        "licensed": licensed,
        "base_four_artifact_refuses": base_row,
        "falsification": falsification,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — the design questions on the B->H seam
# ---------------------------------------------------------------------------

def _complex_state(fields) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in ("Bx", "By", "Bz", "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = np.ascontiguousarray(
                np.asarray(array)).view(np.uint32).ravel().copy()
    return out


def _build_complex(cell, boundaries, k_point, beta, thickness=0.4,
                   dimensions=2, seed=SEED):
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries, k_point=k_point, beta=beta)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=thickness)
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                 "f_w_Hx", "f_w_Hy", "f_w_Hz"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = (rng.uniform(-0.25, 0.25, size=array.shape)
                      + 1j * rng.uniform(-0.25, 0.25, size=array.shape)
                      ).astype(array.dtype)
    return fields, pml


def _array_path_seam(fields, pml, driver_module) -> None:
    driver_module.step_B(fields, pml)
    driver_module.fill_symmetry_bc_B(fields)
    driver_module.zero_metal_B(fields)
    driver_module.fill_folded_far_ghosts_B(fields)
    driver_module.update_H(fields, pml)


def _welded_seam(fields, pml, driver_module, *, wall: str = "before",
                 history: str = "read_then_write") -> None:
    """The FUSED B->H semantics with array ops: step_B (beta included, it IS the
    array path), the wall clear on the chosen side, then the H constitutive with
    the SCALE=0 source (B read directly, no inv_eps)."""
    import meep_gpu.stepping as stepping  # noqa: PLC0415

    driver_module.step_B(fields, pml)
    if wall == "before":
        driver_module.zero_metal_B(fields)

    for component, source_name, axis_name in stepping.H_CONSTITUTIVE_TERMS:
        target = getattr(fields, component)
        source = getattr(fields, source_name)
        history_array = getattr(fields, "f_w_" + component)
        kps, kms = stepping._constitutive_coefficients(pml, axis_name,
                                                       half_integer=False)
        kps, kms = np.asarray(kps), np.asarray(kms)
        stored = source
        if history == "read_then_write":
            previous = history_array.copy()
            history_array[...] = stored
        else:
            history_array[...] = stored
            previous = history_array.copy()
        target += kps * history_array
        target -= kms * previous

    if wall == "after":
        driver_module.zero_metal_B(fields)
    elif wall == "dropped":
        pass


def design_sweep_leg() -> Dict[str, Any]:
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
            findings.append(f"{name}: the state is all zero, so this row "
                            f"compared nothing")

    knobs: List[Dict[str, Any]] = []
    for label, kwargs in (
        ("wall_clear_after_the_constitutive", {"wall": "after"}),
        ("wall_clear_dropped", {"wall": "dropped"}),
        ("history_written_before_it_is_read", {"history": "write_then_read"}),
    ):
        boundaries = ("metallic", "metallic", "periodic")
        left_fields, left_pml = _build_complex((3.0, 3.0, 0.0), boundaries,
                                               (0.0, 0.0, 0.0), -0.685)
        right_fields, right_pml = _build_complex((3.0, 3.0, 0.0), boundaries,
                                                 (0.0, 0.0, 0.0), -0.685)
        for _ in range(2):
            _array_path_seam(left_fields, left_pml, driver_module)
            _welded_seam(right_fields, right_pml, driver_module, **kwargs)
        left, right = _complex_state(left_fields), _complex_state(right_fields)
        differing = sorted(name for name in left
                           if not np.array_equal(left[name], right[name]))
        knobs.append({"flip": label, "arrays_differing": differing,
                      "visible": bool(differing)})
        if not differing:
            findings.append(
                f"design flip {label!r} is INVISIBLE to the array path")

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
                        "array path here: the beta term is DEAD on these cases")

    return {"leg": "design_sweep", "device": False, "rows": rows,
            "design_knobs": knobs, "beta_term_moves": beta_moved[:6],
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the predicate, clause by clause
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    from meep_gpu import expansion_refusal  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_beta_fused_magnetic_pair as product  # noqa: PLC0415

    probe = json.load(open(os.path.join(API_ROOT, EXTENDED_PROBE),
                           encoding="utf-8"))
    base_probe = json.load(open(os.path.join(API_ROOT, BASE_FOUR_PROBE),
                                encoding="utf-8"))

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

    coverage = product.complex_beta_fused_magnetic_pair_coverage
    with expansion_refusal.declaring_run_policy("keep"):
        bloch, bloch_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                          (0.23, 0.0, 0.0),
                                          -0.39073112848927377)
        record("corpus_no_sources", coverage(bloch, bloch_pml, (), probe=probe),
               True)
        record("electric_source_is_the_other_seams",
               coverage(bloch, bloch_pml, (_Source("D"),), probe=probe), True)
        record("magnetic_deposit_is_carried",
               coverage(bloch, bloch_pml, (_deposit(bloch, "Hz"),), probe=probe),
               True)
        record("magnetic_source_without_a_deposit_index",
               coverage(bloch, bloch_pml, (_Source("B"),), probe=probe),
               False, "does not publish the index it writes")
        record("undeclared_sources", coverage(bloch, bloch_pml, None, probe=probe),
               False, "was not declared")
        record("no_probe_artifact", coverage(bloch, bloch_pml, (), probe={}),
               False, "probe record")
        record("base_four_probe_refused_on_the_extended_clause",
               coverage(bloch, bloch_pml, (), probe=base_probe),
               False, "EXTENDED pattern set")
        record("no_pml", coverage(bloch, None, (), probe=probe), False, "PML")

        zero, zero_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                        (0.23, 0.0, 0.0), 0.0)
        record("beta_zero_is_the_plain_complex_pairs",
               coverage(zero, zero_pml, (), probe=probe), False, "beta is zero")

        walls, walls_pml = _build_complex(
            (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
            (0.0, 0.0, 0.0), -0.685)
        record("metallic_walls_k0", coverage(walls, walls_pml, (), probe=probe),
               True)

        builder_rows = []
        for label, args, kwargs in (
            ("magnetic_source_without_a_deposit_index",
             (bloch, bloch_pml, (_Source("B"),)), {"probe": probe}),
            ("undeclared_sources", (bloch, bloch_pml, None), {"probe": probe}),
            ("base_four_probe", (bloch, bloch_pml, ()), {"probe": base_probe}),
            ("beta_zero", (zero, zero_pml, ()), {"probe": probe}),
        ):
            plan = product.plan_complex_beta_fused_magnetic_pair(*args, **kwargs)
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
    which moved ``update_H`` out of a fixed count of lines below ``step_B`` and
    read as "the seam is not where it was". The seam is now walked consult by
    consult: it runs from the ``step_B`` consult to the ``update_H`` consult
    that closes it, and reaching a consult this seam does NOT own before that
    closer is itself the finding. That asserts strictly more than the line count
    did — a foreign slot dispatched inside the seam is now caught, where a window
    could only notice the two endpoints drifting apart.
    """
    from meep_gpu import fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_beta_fused_magnetic_pair as product  # noqa: PLC0415

    opens, fill, closes = "step_B", "fill_B", "update_H"
    # The consults this seam owns: its curl slot, its fill slot, the far-ghost
    # pass that fill slot's second half stands in for, and the constitutive slot
    # that closes it. Every other name in DRIVER_SLOTS is foreign to this seam.
    owned = {opens, fill, closes, fastpath.FAR_FILL_PASSES[fill]}
    consults = tuple(fastpath.DRIVER_SLOTS) + tuple(fastpath.FAR_FILL_PASSES.values())
    foreign = [name for name in consults if name not in owned]

    # A CONSULT MAY BE SPELLED AS A NAME RATHER THAN A LITERAL, and one is.
    # ``synchronize_magnetic_fields`` closes this seam through the by-name channel
    # ``fastpath.SYNC_PASS_OWNERS`` opens — ``dispatch(SYNC_UPDATE_H_PASS, ...)`` —
    # so a scan that matched only ``dispatch("update_H"`` reads the half-step as a
    # ``step_B`` with no closer and reports the seam as moved. It has not moved: that
    # site consults the SAME slot under a name a product spanning into the electric
    # half can decline. The alias is DERIVED from ``fastpath`` (the identifier whose
    # value is the pass, and the slot the channel says owns it), never spelled here,
    # so a renamed constant is a failure rather than a silent miss.
    aliases: Dict[str, str] = {}
    for pass_name, owner in fastpath.SYNC_PASS_OWNERS.items():
        aliases[f'dispatch("{pass_name}"'] = owner
        for attribute in dir(fastpath):
            if getattr(fastpath, attribute, None) == pass_name:
                aliases[f"dispatch({attribute}"] = owner

    def consulted(item: str) -> Optional[str]:
        for spelling, owner in aliases.items():
            if spelling in item:
                return owner
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
                 ("fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B")
                 if any(name + "(" in item for item in body)]
        injects = any("source.inject" in item for item in body)
        windows.append({"driver_line": index + 1, "passes_between": calls,
                        "closes_at_line": index + 1 + end,
                        "injects_a_source": injects})
        expected = [name for name in product.REPLACES
                    if name not in ("step_B", "update_H")]
        if calls != expected:
            findings.append(
                f"driver.py:{index + 1} runs {calls} between step_B and update_H, "
                f"but the product declares it replaces {expected}")
        if not injects:
            findings.append(
                f"driver.py:{index + 1} shows no source injection in this seam")
    if not windows:
        findings.append("no step_B/update_H seam was found in driver.py at all")
    return {"leg": "seam_binding", "device": False, "windows": windows,
            "declared_replaces": list(product.REPLACES),
            "seam_owns": sorted(owned), "foreign_consults": foreign,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side
# ---------------------------------------------------------------------------

def build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec, beta,
                 seed: int, magnetic: bool):
    """One complex-storage BETA PML driver; ``magnetic`` puts a real source IN
    THIS SEAM."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35,
        boundaries=boundaries, k_point=k_point, beta=beta,
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    driver.setup_pml(dict(pml_spec))
    if magnetic:
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": (0.0, 0.0, 0.0), "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        host = (rng.uniform(-0.25, 0.25, size=shape)
                + 1j * rng.uniform(-0.25, 0.25, size=shape)).astype(np.complex64)
        driver.set_field(name, cp.asarray(np.ascontiguousarray(host)))
    # THE DERIVED FIELDS ARE SEEDED DIRECTLY (`set_field` takes primaries only,
    # driver.py:3975). E IS THE BETA PARTNER of this seam's curl, and the
    # ownership mask fires only on METALLIC axes — where the wall clears pin the
    # partner's own row to zero from the first update onward. An unseeded E
    # would make the beta-below-the-mask mutation multiply an exact zero at
    # every masked cell and report a real defect as uncaught. f_w_H is seeded
    # for the same reason the electric twin seeds f_w_E: an all-zero history
    # disarms the ordering mutation on the first step.
    for name, scale in (("Ex", 0.25), ("Ey", 0.25), ("Ez", 0.25),
                        ("Hx", 0.10), ("Hy", 0.10), ("Hz", 0.10),
                        ("f_w_Hx", 0.05), ("f_w_Hy", 0.05), ("f_w_Hz", 0.05)):
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
    return {"array": "<inventory>", "reason": "asymmetric", "names": only} if only else None


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

    def run(self, *_args: Any, **_kwargs: Any) -> None:
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


def separate_route(driver, probe):
    """The two separately certified BETA singles this launch replaces."""
    from meep_gpu.triton_kernels import special_kz  # noqa: PLC0415

    curl = special_kz.plan_beta_bloch_pml_curl(driver.fields, driver.pml,
                                               "step_B", probe=probe)
    constitutive = special_kz.plan_beta_run_complex_constitutive(
        driver.fields, driver.pml, "H", probe=probe)
    missing = [name for name, plan in
               (("beta bloch curl", curl),
                ("beta complex constitutive", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case")
    return Route({"step_B": curl, "update_H": constitutive})


def fused_route(plan, driver=None, sources=()):
    from meep_gpu import deposit_repair  # noqa: PLC0415

    seam = deposit_repair.in_seam_sources(tuple(sources), "B")
    if not seam:
        return Route({name: (plan if name == "step_B" else _Absorbed(name, plan))
                      for name in SEAM_PASSES}), None
    leading = deposit_repair.LeadingRepairPlan(plan, driver.fields, driver.pml,
                                               seam, "B")
    trailing = deposit_repair.TrailingRepairPlan("update_H", leading, driver.fields,
                                                 driver.pml)
    plans = {name: _Absorbed(name, plan) for name in SEAM_PASSES}
    plans["step_B"] = leading
    plans["update_H"] = trailing
    return Route(plans), leading


def run_leg(cp, name: str, dimensions, cell, boundaries, k_point, pml_spec,
            beta, steps: int, product, probe, mutant: Any = None,
            install_fused: bool = True, magnetic: bool = False,
            expansion: Optional[int] = None,
            has_beta: Optional[int] = None,
            beta_words_override: Any = None) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                             beta, SEED, magnetic)
    separate = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                            beta, SEED, magnetic)
    fused = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                         beta, SEED, magnetic)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.complex_beta_fused_curl_constitutive_B_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": str(boundaries),
        "k_point": list(k_point), "pml": dict(pml_spec), "beta": beta,
        "magnetic_source_in_the_seam": bool(magnetic),
        "fused_substituted": bool(install_fused),
        "expansion_override": expansion,
        "has_beta_override": has_beta,
        "beta_words_override": (None if beta_words_override is None
                                else [list(pair) for pair in beta_words_override]),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_complex_beta_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        if plan is None:
            verdict = product.complex_beta_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources), probe=probe)
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if expansion is not None:
            plan.expansion = int(expansion)
        if has_beta is not None:
            plan.has_beta = int(has_beta)
        if beta_words_override is not None:
            plan.beta_words = tuple(tuple(float(word) for word in pair)
                                    for pair in beta_words_override)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["licensed_expansion"] = plan.expansion
        row["beta_words"] = [list(pair) for pair in plan.beta_words]
        if install_fused:
            route, leading = fused_route(plan, fused, tuple(fused._sources))
        else:
            route, leading = Route({}), None
        row["bracketed"] = leading is not None
        row["slot_classes"] = {key: type(value).__name__
                               for key, value in route.plans.items()}
        if magnetic and install_fused and leading is None:
            raise AssertionError(
                "the case declares an in-seam magnetic source but the bracket "
                "was not installed")
        separate_plans = separate_route(separate, probe)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
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
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
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
            except Exception:  # noqa: BLE001
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if require_identical and row.get("first_divergence") is not None:
        failures.append(f"byte divergence: {row['first_divergence']}")
    if row.get("control_divergence") is not None:
        failures.append(
            f"an ORACLE control itself diverged from the array path: "
            f"{row['control_divergence']}")
    if (require_launches is not None
            and row.get("fused_kernel_launches") != require_launches):
        failures.append(
            f"the fused kernel launched {row.get('fused_kernel_launches')} times, "
            f"expected {require_launches}")
    if require_moved and (row.get("arrays_never_moved") or []):
        failures.append(f"VACUOUS: these arrays never moved: "
                        f"{row['arrays_never_moved']}")
    if row.get("material_changed"):
        failures.append(f"a material input changed: {row['material_changed']}")
    if require_repairs and not row.get("deposit_repairs"):
        failures.append(
            "the carry leg repaired NO deposit cell: this leg measured the "
            "quiet case under a carry name")
    fell_back = {key: value for key, value in (row.get("launches") or {}).items()
                 if key.startswith("fused/array_path:")}
    if fell_back:
        failures.append(
            f"the fused route reached the array path for an absorbed pass: "
            f"{fell_back}")
    return (not failures), failures


# ---------------------------------------------------------------------------
# Armed kernel mutations
# ---------------------------------------------------------------------------

def shipped_source(product) -> str:
    return textwrap.dedent(
        inspect.getsource(product.complex_beta_fused_curl_constitutive_B.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    header = ("import triton\n"
              "import triton.language as tl\n"
              "from meep_gpu.triton_kernels.complex_fields import (\n"
              "    _rotate_field_left, _mul_field_left, _mul_coefficient_left)\n"
              "from meep_gpu.triton_kernels.special_kz import (\n"
              "    _mul_imag_coefficient_left)\n"
              "PERIODIC = tl.constexpr(0)\n"
              "METALLIC = tl.constexpr(1)\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_complex_beta_magnetic_pair.py", delete=False,
        encoding="utf-8")
    handle.write(header + source.replace("complex_beta_fused_curl_constitutive_B",
                                         kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_complex_beta_magnetic_pair_" + str(len(_TEMPORARY)),
        handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    def m1_wall_clear_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0_re = tl.where(at_x, 0.0, v0_re)",
             "v0_im = tl.where(at_x, 0.0, v0_im)"],
            ["v0_re = v0_re", "v0_im = v0_im"])

    def m2_wall_clear_only_on_the_real_plane(source: str) -> Tuple[str, int]:
        hits = 0
        for register, coordinate in (("v0_im", "at_x"), ("v1_im", "at_y"),
                                     ("v2_im", "at_z")):
            needle = f"{register} = tl.where({coordinate}, 0.0, {register})"
            hits += source.count(needle)
            source = source.replace(needle, f"{register} = {register}")
        return source, hits

    def m3_wall_clear_takes_the_electric_maps_components(source: str) -> Tuple[str, int]:
        """The D side's tangential map carried onto the B side — the mirror
        image of the electric twin's shipped defect."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0_re = tl.where(at_x, 0.0, v0_re)",
             "v0_im = tl.where(at_x, 0.0, v0_im)"],
            ["v1_re = tl.where(at_x, 0.0, v1_re)",
             "v1_im = tl.where(at_x, 0.0, v1_im)",
             "v2_re = tl.where(at_x, 0.0, v2_re)",
             "v2_im = tl.where(at_x, 0.0, v2_im)"])

    def m4_phase_dropped_on_x(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["rot_re, rot_im = _rotate_field_left(b_x_re, b_x_im, pxr, pxi, EXPANSION)",
             "b_x_re = tl.where(wx, rot_re, b_x_re)",
             "b_x_im = tl.where(wx, rot_im, b_x_im)"],
            ["b_x_re = b_x_re", "b_x_im = b_x_im"])

    def m5_constitutive_association(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
             "a_re = a_re + t_re",
             "a_im = a_im + t_im",
             "t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)",
             "a_re = a_re - t_re",
             "a_im = a_im - t_im"],
            ["kp_re, kp_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
             "km_re, km_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)",
             "a_re = a_re + (kp_re - km_re)",
             "a_im = a_im + (kp_im - km_im)"])

    def m6_history_read_after_write(source: str) -> Tuple[str, int]:
        hits = 0
        for register in ("w0", "w1", "w2"):
            source, hit = _rewrite_block(
                source,
                [f"tl.store({register} + 2 * idx, src_re, mask=live)",
                 f"tl.store({register} + 2 * idx + 1, src_im, mask=live)"],
                [f"tl.store({register} + 2 * idx, src_re, mask=live)",
                 f"tl.store({register} + 2 * idx + 1, src_im, mask=live)",
                 f"prev_re = tl.load({register} + 2 * idx, mask=live, other=0.0)",
                 f"prev_im = tl.load({register} + 2 * idx + 1, mask=live, "
                 f"other=0.0)"])
            hits += hit
        return source, hits

    def m8_imaginary_plane_of_the_weld_dropped(source: str) -> Tuple[str, int]:
        hits = 0
        for component in ("0", "1", "2"):
            needle = f"src_im = v{component}_im"
            hits += source.count(needle)
            source = source.replace(needle, "src_im = 0.0")
        return source, hits

    def m14_beta_insert_dropped(source: str) -> Tuple[str, int]:
        return _rewrite_block(
            source,
            ["if HAS_BETA:",
             "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
             "curl0_re = curl0_re - t_re",
             "curl0_im = curl0_im - t_im",
             "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
             "curl1_re = curl1_re - t_re",
             "curl1_im = curl1_im - t_im"],
            ["curl0_re = curl0_re", "curl0_im = curl0_im",
             "curl1_re = curl1_re", "curl1_im = curl1_im"])

    def m15_beta_partner_rotated(source: str) -> Tuple[str, int]:
        needle = ("t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, "
                  "b_re, b_im, EXPANSION)")
        replacement = ("t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, "
                       "b_x_re, b_x_im, EXPANSION)")
        return source.replace(needle, replacement), source.count(needle)

    def m16_beta_below_the_ownership_mask(source: str) -> Tuple[str, int]:
        without, removed = _rewrite_block(
            source,
            ["if HAS_BETA:",
             "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
             "curl0_re = curl0_re - t_re",
             "curl0_im = curl0_im - t_im",
             "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
             "curl1_re = curl1_re - t_re",
             "curl1_im = curl1_im - t_im"],
            ["curl0_re = curl0_re"])
        if not removed:
            return source, 0
        anchor = "km_x = tl.load(kmx + i, mask=live, other=0.0)"
        insert = (
            "    if HAS_BETA:\n"
            "        t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, "
            "b_re, b_im, EXPANSION)\n"
            "        curl0_re = curl0_re - t_re\n"
            "        curl0_im = curl0_im - t_im\n"
            "        t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, "
            "a_re, a_im, EXPANSION)\n"
            "        curl1_re = curl1_re - t_re\n"
            "        curl1_im = curl1_im - t_im\n")
        if without.count(anchor) != 1:
            return source, 0
        lines = without.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if anchor in line:
                lines.insert(index, insert)
                break
        return "".join(lines), 1

    def m9_commuted_multiply(source: str) -> Tuple[str, int]:
        needle = "r_re = (r_re + n0_re) - p0_re"
        return source.replace(needle, "r_re = (n0_re + r_re) - p0_re"), \
            source.count(needle)

    return (
        ("m1_wall_clear_dropped", "the wall clear's slot", "caught",
         m1_wall_clear_dropped),
        ("m2_wall_clear_only_on_the_real_plane",
         "complex zero is TWO words", "caught",
         m2_wall_clear_only_on_the_real_plane),
        ("m3_wall_clear_takes_the_electric_maps_components",
         "the B-side wall map is the twin's COMPLEMENT", "caught",
         m3_wall_clear_takes_the_electric_maps_components),
        ("m4_phase_dropped_on_x", "the Bloch wrap", "caught", m4_phase_dropped_on_x),
        ("m5_constitutive_association", "float32 association", "caught",
         m5_constitutive_association),
        ("m6_history_read_after_write", "the f_w ordering", "caught",
         m6_history_read_after_write),
        ("m8_imaginary_plane_of_the_weld_dropped", "the weld is two registers",
         "caught", m8_imaginary_plane_of_the_weld_dropped),
        ("m14_beta_insert_dropped", "the beta insert IS the product", "caught",
         m14_beta_insert_dropped),
        ("m15_beta_partner_rotated", "the partner is the CENTER load", "caught",
         m15_beta_partner_rotated),
        ("m16_beta_below_the_ownership_mask",
         "the array path masks the beta increment too", "caught",
         m16_beta_below_the_ownership_mask),
        ("m9_commuted_multiply", "commuted add", "null", m9_commuted_multiply),
    )


MUTATION_CASE: Dict[str, int] = {
    "m1_wall_clear_dropped": 3,
    "m3_wall_clear_takes_the_electric_maps_components": 3,
    "m16_beta_below_the_ownership_mask": 3,
}

DEFAULT_MUTATION_CASE = 4


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    index = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    return index, CASES[index]


def run_mutations(cp, product, probe, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
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
        kernel_name = f"mutant_{index}_complex_beta_fused_B"
        try:
            mutant = compile_mutant(mutated, kernel_name)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"the mutant did not import: {exc!r}"
            rows.append(row)
            log(f"  mutation {name}: DID NOT IMPORT ({exc!r})")
            continue
        case_index, case = mutation_case_for(name)
        row["case"] = case[0]
        row["case_index"] = case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3], case[4],
                      case[5], case[6], MUTATION_STEPS, product, probe,
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

def run_refusals(cp, product, probe) -> List[Dict[str, Any]]:
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

    base_four = json.load(open(os.path.join(API_ROOT, BASE_FOUR_PROBE),
                               encoding="utf-8"))
    base = CASES[0]
    rows: List[Dict[str, Any]] = []
    for name, case, beta_override, needle, sources, use_probe in (
        ("magnetic_source_without_a_deposit_index", base, None,
         "does not publish the index it writes", (_Magnetic(),), probe),
        ("magnetic_deposit_is_carried", base, None, None,
         lambda fields: (_deposit(fields, "Hz"),), probe),
        ("electric_source_is_the_other_seams", base, None, None,
         (_Electric(),), probe),
        ("undeclared_sources", base, None, "was not declared", None, probe),
        ("no_probe_artifact", base, None, "probe record", (), {}),
        ("beta_zero_is_the_plain_complex_pairs", base, 0.0, "beta is zero",
         (), probe),
        ("base_four_probe_refused_on_the_extended_clause", base, None,
         "EXTENDED pattern set", (), base_four),
    ):
        beta = case[6] if beta_override is None else beta_override
        driver = build_driver(cp, case[1], case[2], case[3], case[4], case[5],
                              beta, SEED, magnetic=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.complex_beta_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, bound, probe=use_probe)
            plan = product.plan_complex_beta_fused_magnetic_pair(
                driver.fields, driver.pml, bound, probe=use_probe)
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


def environment(cp: Any = None) -> Dict[str, Any]:
    import socket  # noqa: PLC0415

    payload: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "expansion_probe_environment_variable": os.environ.get(
            "MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
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
        HERE, "results", "triton_complex_beta_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--subnormal-policy", default="keep")
    args = parser.parse_args(argv)

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_complex_beta_fused_magnetic_pair",
        "product": "meep_gpu.triton_kernels.complex_beta_fused_magnetic_pair",
        "kernel": "complex_beta_fused_curl_constitutive_B",
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
    for leg in (transcription_leg, expansion_licence_leg, design_sweep_leg,
                predicate_leg, seam_binding_leg):
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
            "reasons": ["the no-device legs passed, but no device leg, mutation "
                        "or refusal has run: this artifact releases nothing"],
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

    from meep_gpu.triton_kernels import complex_fields, special_kz  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_beta_fused_magnetic_pair as product  # noqa: PLC0415

    probe = complex_fields.load_expansion_probe()
    licence = complex_fields.expansion_license(probe,
                                               product.PRODUCT_PROBE_PATTERNS)
    payload["expansion"] = {
        "probe_path": os.environ.get("MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
        "arm": licence["arm"], "basis": licence["basis"],
        "expansion": licence["expansion"], "refusals": list(licence["refusals"]),
        "policy_resolved": licence["policy_resolved"],
        "candidate_policy": licence["candidate_policy"],
    }
    if licence["expansion"] is None:
        payload["device_status"] = "REFUSED BEFORE THE FIRST LEG"
        payload["passed"] = False
        payload["release"] = {"released": False, "reasons": [
            "no EXPANSION arm is licensed for this run: "
            + "; ".join(licence["refusals"])]}
        save(payload, args.out)
        log(f"\nREFUSED: {licence['refusals']}")
        return 1
    log(f"expansion arm licensed: {licence['arm']} (basis {licence['basis']})")

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
                      pml_spec, beta, steps, product, probe, magnetic=False)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== device legs: the CARRY family (magnetic deposit, shipped bracket) ===")
    for index in CARRY_CASES:
        name, dimensions, cell, boundaries, k_point, pml_spec, beta, steps = CASES[index]
        row = run_leg(cp, f"carry:{name}", dimensions, cell, boundaries, k_point,
                      pml_spec, beta, steps, product, probe, magnetic=True)
        passed, failures = verdict_of(row, require_launches=steps,
                                      require_repairs=True)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== null controls: the carry bracket REMOVED must diverge ===")
    for index in CARRY_CASES:
        name, dimensions, cell, boundaries, k_point, pml_spec, beta, _steps = CASES[index]
        payload["device_legs"].append(_null_unbracketed(
            cp, f"null:unbracketed:{name}", dimensions, cell, boundaries,
            k_point, pml_spec, beta, product, probe))
        save(payload, args.out)

    log("\n=== armed harness legs ===")
    base = CASES[0]
    row = run_leg(cp, "armed:no_substitution", base[1], base[2], base[3], base[4],
                  base[5], base[6], 3, product, probe, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and "
                  "the counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    other = 1 - int(licence["expansion"])
    row = run_leg(cp, "armed:unlicensed_expansion_arm", CASES[4][1], CASES[4][2],
                  CASES[4][3], CASES[4][4], CASES[4][5], CASES[4][6], 3, product,
                  probe, expansion=other)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = (f"the EXPANSION constexpr is forced to arm {other} while the "
                  f"platform measured {licence['expansion']}; the bytes must "
                  f"diverge")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:has_beta_zero", base[1], base[2], base[3], base[4],
                  base[5], base[6], 3, product, probe, has_beta=0)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = ("HAS_BETA = 0 launches the plain complex weld on a beta run; "
                  "the bytes must diverge")
    payload["device_legs"].append(row)
    save(payload, args.out)

    electric_words = special_kz.beta_curl_coefficients(
        base[6], 0.35 / 12.0, magnetic=False, complex_storage=True)
    row = run_leg(cp, "armed:electric_coefficient_words", base[1], base[2],
                  base[3], base[4], base[5], base[6], 3, product, probe,
                  beta_words_override=electric_words)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("first_divergence") is not None
    row["why"] = ("the -1j (electric) coefficient pair on the magnetic weld must "
                  "diverge; the sign lives in the words")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.complex_beta_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, probe, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product, probe)
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
                      beta, product, probe) -> Dict[str, Any]:
    """The NULL CONTROL: the weld substituted WITHOUT the repair bracket, on a
    case with an in-seam magnetic deposit. It MUST diverge."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                             beta, SEED, magnetic=True)
    fused = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                         beta, SEED, magnetic=True)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(product.complex_beta_fused_curl_constitutive_B_kernel())
    row: Dict[str, Any] = {"leg": name, "device": True, "null_control": True,
                           "beta": beta}
    try:
        plan = product.plan_complex_beta_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        if plan is None:
            raise AssertionError("the product refused the null control's case")
        route = Route({key: (plan if key == "step_B" else _Absorbed(key, plan))
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
