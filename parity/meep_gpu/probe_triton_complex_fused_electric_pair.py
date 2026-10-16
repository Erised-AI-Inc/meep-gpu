#!/usr/bin/env python3
"""THE COMPLEX FUSED ELECTRIC PAIR GATE — complex ``step_D`` welded into ``update_E``.

Does ONE launch of ``complex_fused_curl_constitutive_D`` leave the engine
BIT-IDENTICAL, per COMPLETE driver step, to both the CuPy array path and the two
separately certified Triton products it replaces — over every allocated volume,
compared as uint32 words?

THE TWIN'S GATE IS THE TEMPLATE AND THE DIFFERENCES ARE THE POINT.
``probe_triton_complex_fused_magnetic_pair.py`` asks the same question on the B/H
seam. Three things are genuinely different here, and each gets its own machinery
rather than being folded into an existing leg:

1. **THE SOURCE IS IN THIS SEAM.** The driver injects the electric list between
   ``step_D`` and ``update_E`` (driver.py:3296-3299). On the magnetic gate the
   electric source was carried purely as a source the pair may ignore; here it is
   the thing the product must survive. So the device side runs TWO families of
   case: a QUIET family with no in-seam source, which measures the weld alone, and
   a CARRY family whose fused route is bracketed by the SHIPPED
   ``deposit_repair.LeadingRepairPlan`` / ``TrailingRepairPlan``. Fifteen of the
   sixteen complex corpus rows declare an electric source, so a gate that ran only
   the quiet family would license one row and report it as sixteen.
2. **``SCALE`` IS 1.** The constitutive half multiplies its source by ``inv_eps``
   before the ``f_w`` store. Three mutations exist for that arm alone — dropped,
   word-doubled, and applied on the wrong side of the store — because none of them
   is visible to any check the magnetic twin carries.
3. **THE COEFFICIENT LATTICES ARE MIRRORED.** The curl takes the INTEGER lattice
   and the constitutive the HALF-INTEGER one, the opposite pair from the twin. A
   swap is a half-cell error in the absorber profile: converged, smooth and wrong,
   and invisible outside the PML.

Correctness only. NOTHING HERE IS TIMED and no throughput claim is admissible.
Dispatch stays disabled: ``launch.plan_step`` does not know this module exists.

WHAT THIS GATE CANNOT DECIDE, known before the run rather than discovered after:
``m4_phase_dropped_on_x`` is invisible at k = 0 and ``m8_imaginary_plane_...`` is
invisible on a quiet imaginary plane, so both are scored on a case carrying a
nonzero k; ``m1_wall_clear_dropped`` is invisible without a metallic x, so it is
scored on the case that has one. The case each mutation runs on is NAMED in the
artifact, because a mutation reported null on a case that could not see it is a
statement about the case.
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

SEED = 20260830

_TEMPORARY: List[str] = []

#: (name, dimensions, cell, boundaries, k_point, pml, steps). Every case stores
#: complex fields. THE SET IS SWEPT AND NEITHER AXIS IS DECORATION: a k = 0 case is
#: what makes the phase-skip reduction to the plain complex path measurable, a
#: phased case is the only place the rotation mutations have anywhere to be caught,
#: and a case with real METALLIC walls is the only place the wall-clear mutations
#: fire. A run with none of the three would report those mutations as uncaught.
CASES: Tuple[Tuple[str, int, Tuple[float, float, float], Any,
                   Tuple[float, float, float], Dict[str, Any], int], ...] = (
    ("bloch_x_periodic", 2, (3.0, 3.0, 0.0), "periodic", (0.23, 0.0, 0.0),
     {"y": 5}, 10),
    ("bloch_edge_minus_one", 2, (3.0, 3.0, 0.0), "periodic", (0.5, 0.0, 0.0),
     {"y": 5}, 10),
    ("k0_complex_storage", 2, (3.0, 3.0, 0.0), "periodic", (0.0, 0.0, 0.0),
     {"x": 5, "y": 5}, 10),
    ("metallic_walls_k0", 2, (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
     (0.0, 0.0, 0.0), {"x": 5, "y": 5}, 10),
    ("bloch_x_metallic_y", 2, (3.0, 3.0, 0.0),
     ("periodic", "metallic", "periodic"), (0.19, 0.0, 0.0), {"y": 5}, 10),
)

#: The CARRY family: the same grids, run with a real electric deposit in the seam
#: and the SHIPPED repair bracket installed. Three rather than five because the
#: question they answer is about the seam and not about the boundary sweep — one
#: phased, one k = 0 walled, one phased-and-walled. THE PRODUCT IS WORTH ONE
#: CORPUS ROW WITHOUT THIS FAMILY AND SIXTEEN WITH IT.
CARRY_CASES: Tuple[int, ...] = (0, 3, 4)

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3292-3304).
SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
)

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this
#: tuple is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: THE READ-ONLY MATERIAL VOLUMES, under the names ``inventory`` really finds. A
#: leg that changes one of these measured something other than the step. The names
#: are ``vars(fields)``' own; a name that matches nothing is not a weaker check,
#: it is two absent ones (the vacuity floor and ``material_changed``), which is
#: what ``_assert_material_names_are_real`` exists to stop.
#:
#: ON THIS SIDE ``inv_eps`` IS ALSO A KERNEL INPUT, read by the SCALE=1 arm. That
#: makes the read-only assertion stronger here than on the twin, not weaker: a
#: kernel that wrote through its epsilon pointer would be caught by the same check
#: that catches a kernel writing its own material.
MATERIAL = ("eps", "inv_eps")


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    """Every MATERIAL name must appear in the inventory the scan actually built."""
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and the material-write check would scan an empty set")


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Write the artifact atomically, STAMPED with the run's own import closure.

    ``gate_provenance.stamp`` enumerates every repo module this PROCESS imported
    out of ``sys.modules`` and records its digest under ``imported_source_sha256``,
    plus the canonical reading of the verdict. Both are what the weld pipeline
    consumes: ``propose_triton_welds.py`` intersects the family's declared bindings
    with the imported set and REFUSES to propose a path the run did not import, so
    an artifact without this stamp can be released and still weld nothing.

    A stamper failure is recorded and swallowed. Losing a completed device run to a
    provenance exception costs the run; recording the exception costs a key.
    """
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
        "meep_gpu/triton_kernels/complex_fused_electric_pair.py",
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
        "meep_gpu/test_triton_complex_fused_electric_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, read off the shipped source
# ---------------------------------------------------------------------------

def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported: this leg has to run on a host with no
    Triton, where a ``@triton.jit`` object is ``None``, so the transcription check
    bites at the merge bar rather than only on a device run.

    The text is exact — never ``ast.unparse``d — because what is being checked is
    the PARENTHESISATION, and unparsing re-derives minimal parentheses and would
    silently rewrite ``((c_y_re - c_re) + (b_re - b_z_re))`` into a different
    grouping than the one whose float32 bits the gate is about.

    The docstring IS stripped, and that is not cosmetic: these bodies document the
    codes and branches they do NOT carry, so a text search over the raw source
    finds those names in prose and reports them as live code.
    """
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        {"bloch_pml_curl_step": "complex_fields.py",
                         "bloch_constitutive_step": "complex_fields.py",
                         "complex_fused_curl_constitutive_D":
                             "complex_fused_electric_pair.py"}[name])
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
    # Normalise the indentation to what ``textwrap.dedent(inspect.getsource(fn))``
    # produces, which is what :func:`shipped_source` hands the mutation table on a
    # CUDA host. Two spellings of the same text would let a rewrite match here and
    # miss there — the exact way a mutation silently disarms.
    margin = node.col_offset
    lines = [lines[0].lstrip()] + [
        line[margin:] if line[:margin].strip() == "" else line.lstrip()
        for line in lines[1:]]
    return "\n".join(lines)


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace one consecutive run of statements, matched on STRIPPED text.

    Matching on stripped lines rather than on an exact substring is what keeps a
    rewrite armed across a reindentation or a trailing comment. A mutation that
    silently stops matching reports a real defect as uncaught, which this tree has
    already paid for once.
    """
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
    """Executable lines, comments and blanks removed, indentation normalised."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


#: The arithmetic lines the fused body must reproduce VERBATIM from the certified
#: complex curl. Each is a line whose grouping decides float32 bits; a reformat is
#: a different number, not a style change.
CURL_LINES = (
    "r_re, r_im = _connect(c_re, c_im, cxr, cxi, CNX)",
    "c1_re = tl.where(at_x, r_re, c_re)",
    "r_re, r_im = _connect(c_re, c_im, cyr, cyi, CNY)",
    "c0_re = tl.where(at_y, r_re, c_re)",
    "r_re, r_im = _connect(a_re, a_im, czr, czi, CNZ)",
    "a1_im = tl.where(at_z, r_im, a_im)",
    "presence = tl.load(cb + RECORD_PRESENCE)",
    "curl0_re = curl_stencil(c_y_re, c0_re, b0_re, b_z_re, dtdx, py, pz)",
    "curl0_im = curl_stencil(c_y_im, c0_im, b0_im, b_z_im, dtdx, py, pz)",
    "curl1_re = curl_stencil(a_z_re, a1_re, c1_re, c_x_re, dtdx, pz, px)",
    "curl1_im = curl_stencil(a_z_im, a1_im, c1_im, c_x_im, dtdx, pz, px)",
    "curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)",
    "curl2_im = curl_stencil(b_x_im, b2_im, a2_im, a_y_im, dtdx, px, py)",
    "du0 = record_in(cb, lattice + 8, k)",
    "p_re = tl.load(u0 + 2 * idx, mask=live & du0, other=0.0)",
    "tl.store(u0 + 2 * idx, n0_re, mask=live & du0)",
    "tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)",
)

#: The constitutive accumulation, from ``complex_fields.bloch_constitutive_step``.
#: Two separate accumulations, left to right, on BOTH planes; flattening them is a
#: different float32 number.
CONSTITUTIVE_LINES = (
    "prev_re = tl.load(w0 + 2 * idx, mask=live & dw0, other=0.0)",
    "prev_im = tl.load(w0 + 2 * idx + 1, mask=live & dw0, other=0.0)",
    "tl.store(w0 + 2 * idx, src_re, mask=live & dw0)",
    "tl.store(w0 + 2 * idx + 1, src_im, mask=live & dw0)",
    "src_re = src_re * ie",
    "src_im = src_im * ie",
)

#: THE SCALE=1 ARM, which the magnetic twin does not carry at all. The multiply is
#: field-LEFT and the index is UNDOUBLED — ``inv_eps`` is one real coefficient per
#: COMPLEX cell (fields.py:1203-1204). Both facts are float32-bit-deciding and
#: neither is visible to any check the twin's gate runs.
SCALE_LINES = (
    "ie = tl.load(e0 + idx, mask=live, other=0.0)",
    "src_re = src_re * ie",
    "ie = tl.load(e1 + idx, mask=live, other=0.0)",
    "ie = tl.load(e2 + idx, mask=live, other=0.0)",
)

#: The weld itself: the constitutive source is the register the curl produced, on
#: both planes, for all three components.
WELD_LINES = (
    "src_re = v0_re", "src_im = v0_im",
    "src_re = v1_re", "src_im = v1_im",
    "src_re = v2_re", "src_im = v2_im",
)

#: The wall clear, carried inline on BOTH planes, WITH THE D-SIDE COMPONENT MAP.
#: ``stepping._zero_metal`` clears every component whose Yee shift on the walled
#: axis is zero, which on this side is the two TANGENTIAL components — an x wall
#: clears Dy and Dz and leaves Dx alone, the exact complement of the magnetic
#: twin's map. Both planes: ``array[face] = 0`` on a complex64 volume writes complex
#: zero, so a transcription that cleared only the real plane is wrong exactly where
#: the imaginary part is nonzero at a wall.
#:
#: THESE SIX LINES ARE NOT A STYLE CHOICE. The first device run of this gate carried
#: the twin's ``ZM_X -> v0`` map and came back with ``Dx`` differing on both metallic
#: cases in opposite directions — 140 words zeroed that should not have been, 72 left
#: live that should have been cleared — while every periodic case was bit-identical
#: over ten steps. This tuple is what stops that map from coming back.
WALL_LINES = (
    "v1_re = tl.where(at_x, 0.0, v1_re)", "v1_im = tl.where(at_x, 0.0, v1_im)",
    "v2_re = tl.where(at_x, 0.0, v2_re)", "v2_im = tl.where(at_x, 0.0, v2_im)",
    "v0_re = tl.where(at_y, 0.0, v0_re)", "v0_im = tl.where(at_y, 0.0, v0_im)",
    "v2_re = tl.where(at_y, 0.0, v2_re)", "v2_im = tl.where(at_y, 0.0, v2_im)",
    "v0_re = tl.where(at_z, 0.0, v0_re)", "v0_im = tl.where(at_z, 0.0, v0_im)",
    "v1_re = tl.where(at_z, 0.0, v1_re)", "v1_im = tl.where(at_z, 0.0, v1_im)",
)

#: Words per complex cell — the real and imaginary planes at ``2*idx`` / ``2*idx+1``.
WORDS_PER_COMPLEX_CELL = 2


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line of the fused body traced to the source it came from."""
    from meep_gpu.triton_kernels import complex_fused_electric_pair as product  # noqa: PLC0415

    fused = _statements(_shipped_text("complex_fused_curl_constitutive_D"))
    curl = _statements(_shipped_text("bloch_pml_curl_step"))
    constitutive = _statements(_shipped_text("bloch_constitutive_step"))

    findings: List[str] = []
    for line in CURL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the curl line {line!r}")
        if line not in curl:
            findings.append(
                f"complex_fields.bloch_pml_curl_step no longer contains {line!r}; "
                f"the fused body's transcription source has moved")
    for line in CONSTITUTIVE_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain {line!r}")
        if line not in constitutive:
            findings.append(
                f"complex_fields.bloch_constitutive_step no longer contains "
                f"{line!r}; the constitutive grouping's source has moved")
    for line in SCALE_LINES:
        if line not in fused:
            findings.append(
                f"the fused body does not contain the SCALE=1 line {line!r}: this "
                f"is the E side and the inv_eps multiply is not optional here")
        if line not in constitutive:
            findings.append(
                f"complex_fields.bloch_constitutive_step no longer contains "
                f"{line!r}; the inv_eps arm's transcription source has moved")
    for line in WELD_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the weld line {line!r}")
    for line in WALL_LINES:
        if line not in fused:
            findings.append(
                f"the fused body does not contain the wall-clear line {line!r}: "
                f"stepping._zero_metal writes COMPLEX zero and both planes must be "
                f"cleared")

    # THE WELD REPLACED A LOAD, AND THE LOAD MUST BE GONE. If the constitutive half
    # still read D from memory, the "fusion" would be a launch-count optimisation
    # with a live re-read — a different kernel with a different hazard. TWO is the
    # right count and not one: the curl half loads each source component's real and
    # imaginary word at ``2 * idx`` and ``2 * idx + 1``. An unwelded body would
    # carry FOUR, because ``bloch_constitutive_step`` loads the same pair again.
    for needle in ("tl.load(g0 + 2 * idx", "tl.load(g1 + 2 * idx",
                   "tl.load(g2 + 2 * idx"):
        occurrences = sum(1 for line in fused if needle in line)
        if occurrences != WORDS_PER_COMPLEX_CELL:
            findings.append(
                f"{needle!r} appears {occurrences} times, not "
                f"{WORDS_PER_COMPLEX_CELL}; the curl half reads each source "
                f"component's two words once and the constitutive half must read D "
                f"from the register, not from memory")

    # THE inv_eps INDEX IS NEVER DOUBLED. One real coefficient per complex cell; a
    # ``2 * idx`` here reads the imaginary neighbour's coefficient into the real
    # plane, which is smooth, converged and wrong.
    doubled = [line for line in fused
               if any(f"tl.load(e{c} + 2 * idx" in line for c in "012")]
    if doubled:
        findings.append(
            f"the inv_eps load is word-doubled: {doubled}; inv_eps is float32 with "
            f"one coefficient per COMPLEX cell and is indexed at + idx")

    # THE BAKED WALL MAP IS THE ENGINE'S OWN. Recomputed here from IYEE_SHIFTS by
    # `_zero_metal`'s rule, so a body whose `if ZM_*:` blocks drifted back to the
    # magnetic twin's `axis d -> component d` fails at the MERGE BAR rather than on
    # a metallic device case. The shipped predicate carries the same clause; this is
    # the copy that runs with no device.
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    derived = tuple(
        tuple(index for index, target in enumerate(product.CURL_TARGETS)
              if IYEE_SHIFTS[target][axis] == 0)
        for axis in range(3))
    if derived != product.WALL_CLEARED_COMPONENTS:
        findings.append(
            f"stepping._zero_metal clears {derived} per walled axis, but the module "
            f"bakes {product.WALL_CLEARED_COMPONENTS}")
    for axis, coordinate in enumerate(("at_x", "at_y", "at_z")):
        for component in range(3):
            cleared = any(
                f"v{component}_{plane} = tl.where({coordinate}, 0.0, "
                f"v{component}_{plane})" in line
                for plane in ("re", "im") for line in fused)
            expected = component in product.WALL_CLEARED_COMPONENTS[axis]
            if cleared is not expected:
                findings.append(
                    f"the fused body {'clears' if cleared else 'does not clear'} "
                    f"component {component} at {coordinate}, but _zero_metal's rule "
                    f"says {'it should' if expected else 'it should not'}")

    # THE SCALE ARM MUST BE GUARDED, not inlined. The constexpr IS the transcription
    # boundary: an inlined multiply is a hand-specialised body and this leg could no
    # longer compare it to one certified function.
    if not any(line == "if SCALE:" for line in fused):
        findings.append(
            "the fused body has no `if SCALE:` guard; the constitutive half is no "
            "longer a transcription of bloch_constitutive_step's own arm")

    # NO MULTIPLY OUTSIDE THE THREE DECLARED HELPERS. A fourth would be a fourth
    # operand orientation, and this product's whole helper claim is that it adds
    # none.
    helpers = sorted({name for name in product.MULTIPLY_HELPERS
                      if any(name in line for line in fused)})
    stray = sorted({line for line in fused
                    if "_mul_" in line or "_rotate_" in line
                    if not any(name in line for name in product.MULTIPLY_HELPERS)})
    if stray:
        findings.append(
            f"the fused body calls a complex-multiply helper outside "
            f"{product.MULTIPLY_HELPERS}: {stray}")
    if set(helpers) != set(product.MULTIPLY_HELPERS):
        findings.append(
            f"the fused body calls {helpers}, not all of "
            f"{list(product.MULTIPLY_HELPERS)}; the pattern-set claim "
            f"names orientations the kernel no longer launches")
    return {
        "leg": "transcription",
        "device": False,
        "fused_statements": len(fused),
        "curl_lines_checked": len(CURL_LINES),
        "constitutive_lines_checked": len(CONSTITUTIVE_LINES),
        "scale_lines_checked": len(SCALE_LINES),
        "weld_lines_checked": len(WELD_LINES),
        "wall_lines_checked": len(WALL_LINES),
        "multiply_helpers_called": helpers,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — are the two symmetry fills really inert in this seam?
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


def _build_complex(cell, boundaries, k_point, mirrors=(), thickness=0.4,
                   dimensions=2, seed=SEED):
    """A real complex Grid/Fields/PML triple on NumPy, seeded identically."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries, k_point=k_point,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    # A NON-TRIVIAL EPSILON, AND IT IS NOT DECORATION. ``Fields`` starts at
    # eps = 1 everywhere, so ``inverse_epsilon_for`` returns exactly 1.0 and EVERY
    # inv_eps flip becomes a multiply by one — measured: the
    # ``inv_eps_applied_after_the_history_store`` knob below reported INVISIBLE on
    # the vacuum fixture, which is a statement about the fixture and not about the
    # weld. The magnetic twin's gate has no such hazard because its SCALE=0 arm
    # never loads a coefficient.
    index = np.arange(int(np.prod(grid.shape)), dtype=np.float32).reshape(grid.shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    fields.set_epsilon_volumes(
        {component: epsilon for component in ("Ex", "Ey", "Ez")},
        {component: np.ascontiguousarray((1.0 / epsilon).astype(np.float32))
         for component in ("Ex", "Ey", "Ez")})
    pml = PML(grid=grid, thickness=thickness)
    # A state seeded mid-run carries MEEP's H already separated from B
    # (stepping.separate_h_from_b): the seeded H and f_w are the history.
    fields.h_separated = True
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


def _accumulation_cells(plan, component, shape):
    """The cells whose chunk carries ``component``'s own layer (the region plan's)."""
    spec = plan.spec("constitutive", component)
    if spec.mode == "outside_only":
        return np.zeros(shape, dtype=bool)
    if spec.mode == "inside_only":
        return np.ones(shape, dtype=bool)
    return np.broadcast_to(np.asarray(plan.mask("constitutive", component, np)), shape)


def _accumulate(target, history, stored, current, previous, kps, kms, inside):
    """MEEP's two constitutive forms, per part: ``f + ((kps*src) - (kms*prev))`` and
    ``f_w = stored`` on ``inside``, ``f = src`` everywhere else."""
    from meep_gpu import part_products  # noqa: PLC0415

    difference = (part_products.scale(kps, current)
                  - part_products.scale(kms, previous))
    accumulated = target + difference
    np.copyto(history, stored, where=inside)
    target[...] = np.where(inside, accumulated, current)


def _array_path_seam(fields, pml, driver_module) -> None:
    """The five driver passes this product replaces, in driver order."""
    driver_module.step_D(fields, pml)
    driver_module.fill_symmetry_bc_D(fields)
    driver_module.zero_metal_D(fields)
    driver_module.fill_folded_far_ghosts_D(fields)
    driver_module.update_E(fields, pml)


def _welded_seam(fields, pml, driver_module, *, wall: str = "before",
                 history: str = "read_then_write",
                 scale: str = "before_the_store") -> None:
    """The FUSED semantics, spelled with array ops so a laptop can execute them.

    THIS IS NOT THE KERNEL AND DOES NOT PRETEND TO BE. It cannot see a memory
    hazard, a register reuse or a rounding order inside one launch — those are the
    device gate's to measure. What it CAN decide is the DESIGN question the weld
    answers: which value reaches ``update_E``, and in which order relative to the
    wall clear, the ``f_w`` history read and — new on this side — the ``inv_eps``
    multiply.

    ``scale`` IS THE E-SIDE KNOB THE TWIN HAS NO COUNTERPART FOR. The shipped
    ``bloch_constitutive_step`` scales BEFORE the ``f_w`` store, so the history
    holds ``D * inv_eps``; scaling after it stores ``D`` instead, which is wrong
    only where ``kms != 0``, i.e. inside the PML, and only from the SECOND step.

    THE TWO SYMMETRY FILLS ARE SIMPLY ABSENT HERE. That is the module's central
    claim and this leg is where it is measured: on an unfolded complex run the
    five-pass array path and this three-pass emulation must be BIT-IDENTICAL, and
    the folded control must separate.
    """
    import meep_gpu.stepping as stepping  # noqa: PLC0415

    driver_module.step_D(fields, pml)
    if wall == "before":
        driver_module.zero_metal_D(fields)

    # Each component's two forms on its own cells
    # (stepping._apply_constitutive_regions), every product per part.
    from meep_gpu import part_products  # noqa: PLC0415

    plan = stepping._region_plan(fields, pml)
    for component, source_name, axis_name in stepping.E_CONSTITUTIVE_TERMS:
        target = getattr(fields, component)
        source = getattr(fields, source_name)
        history_array = getattr(fields, "f_w_" + component)
        kps, kms = stepping._constitutive_coefficients(pml, axis_name,
                                                       half_integer=True)
        inside = _accumulation_cells(plan, component, target.shape)
        scaled = part_products.scale(fields.inverse_epsilon_for(component), source)
        # The history holds D * inv_eps under the correct order; the scale-after-
        # the-store flip leaves the UNSCALED displacement in it while the kps term
        # still consumes the scaled one. THE kms TERM IS THE WHOLE DEFECT and is
        # not re-scaled here.
        stored = scaled if scale == "before_the_store" else source
        if history == "read_then_write":
            previous = history_array.copy()
        else:
            # The defect: f_w is written before it is read.
            np.copyto(history_array, stored, where=inside)
            previous = history_array.copy()
        _accumulate(target, history_array, stored, scaled, previous, kps, kms, inside)

    if wall == "after":
        # The defect: update_E has already consumed the pre-clear D, so E and
        # f_w_E at the wall carry the stepped displacement rather than zero.
        driver_module.zero_metal_D(fields)
    elif wall == "dropped":
        pass


def symmetry_inertness_leg() -> Dict[str, Any]:
    """Do ``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` write a word here?

    The module refuses a symmetry and then treats both passes as no-ops. That is
    read off ``stepping._fill_symmetry_ghost_cells``'s and
    ``_fill_folded_far_ghosts``' first lines, which is an argument; this is the
    measurement, with the folded control that makes it a statement about the FOLD
    rather than about the probe.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for label, mirrors in (("unfolded_bloch", ()), ("folded_control", (("Y", 1),))):
        boundaries = ("metallic", "metallic", "periodic")
        k_point = (0.0, 0.0, 0.0)
        left_fields, left_pml = _build_complex((3.0, 3.0, 0.0), boundaries,
                                               k_point, mirrors=mirrors)
        right_fields, right_pml = _build_complex((3.0, 3.0, 0.0), boundaries,
                                                 k_point, mirrors=mirrors)
        _array_path_seam(left_fields, left_pml, driver_module)
        _welded_seam(right_fields, right_pml, driver_module)
        left, right = _complex_state(left_fields), _complex_state(right_fields)
        differing = sorted(name for name in left
                           if not np.array_equal(left[name], right[name]))
        rows.append({"case": label, "folded": bool(mirrors),
                     "arrays_differing": differing,
                     "identical": not differing})
        if not mirrors and differing:
            findings.append(
                f"the two symmetry fills are NOT inert on an unfolded complex run: "
                f"{differing} differ between the five-pass array path and the "
                f"three-pass weld")
        if mirrors and not differing:
            findings.append(
                "the FOLDED control did not separate: this leg cannot tell 'the "
                "fills are inert here' from 'the fills never do anything'")

    # THE KNOBS. Each design flip must be visible to the array path, or the
    # corresponding kernel choice is unconstrained by anything.
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
                                               (0.0, 0.0, 0.0))
        right_fields, right_pml = _build_complex((3.0, 3.0, 0.0), boundaries,
                                                 (0.0, 0.0, 0.0))
        # TWO steps: the history defect and the scale defect are both invisible on
        # the first, because f_w starts from the seeded value on both sides.
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
                f"design flip {label!r} is INVISIBLE to the array path; the kernel's "
                f"choice there is constrained by nothing this gate measures")

    return {"leg": "symmetry_inertness", "device": False, "rows": rows,
            "design_knobs": knobs, "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the design sweep
# ---------------------------------------------------------------------------

def design_sweep_leg() -> Dict[str, Any]:
    """The weld's semantics against the array path over the whole boundary sweep.

    Same emulation as leg 3, run over every case in :data:`CASES` rather than one
    grid: a weld that happened to agree on one boundary configuration and not on
    another would otherwise be reported as sound.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for name, dimensions, cell, boundaries, k_point, _pml_spec, _steps in CASES:
        left_fields, left_pml = _build_complex(cell, boundaries, k_point,
                                               dimensions=dimensions)
        right_fields, right_pml = _build_complex(cell, boundaries, k_point,
                                                 dimensions=dimensions)
        for _ in range(2):
            _array_path_seam(left_fields, left_pml, driver_module)
            _welded_seam(right_fields, right_pml, driver_module)
        left, right = _complex_state(left_fields), _complex_state(right_fields)
        differing = sorted(item for item in left
                           if not np.array_equal(left[item], right[item]))
        moved_any = any(np.any(left[item] != 0) for item in left)
        rows.append({"case": name, "arrays_differing": differing,
                     "identical": not differing, "state_is_live": moved_any})
        if differing:
            findings.append(f"{name}: the weld and the array path disagree on "
                            f"{differing}")
        if not moved_any:
            findings.append(f"{name}: the state is all zero, so this row compared "
                            f"nothing")
    return {"leg": "design_sweep", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — the predicate, clause by clause
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every clause of the seam predicate, exercised on real complex grids."""
    from meep_gpu.triton_kernels import complex_fused_electric_pair as product  # noqa: PLC0415

    class _Source:
        def __init__(self, field_type: str) -> None:
            self.field_type = field_type

    def _deposit(fields, component):
        """A REAL source that publishes the index the injection writes.

        ``_Source`` is enough to PLACE a source in the seam and deliberately not
        enough to CARRY one. A row about the carry needs the engine's own source, so
        this builds one and refuses to return a source that deposits nothing --
        otherwise the admitting row would pass by measuring an empty scatter.
        """
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
        """``call()`` with this family's carry declaration held down, then restored."""
        saved = module.CARRIES_DEPOSIT_REPAIR
        module.CARRIES_DEPOSIT_REPAIR = False
        try:
            return call()
        finally:
            module.CARRIES_DEPOSIT_REPAIR = saved

    def residual(verdict) -> List[str]:
        """The reasons that are not the NumPy-host backend clause.

        MATCHED ON THE CLAUSE, NOT ON THE WORD 'cupy'. A filter on the word would
        also swallow any other clause that names a backend, and a refusal that
        vanishes from the residual is a refusal the leg reports as an admission.
        """
        return [reason for reason in verdict.reasons
                if not ("array module is" in reason and "not cupy" in reason)]

    rows: List[Dict[str, Any]] = []

    def record(label: str, verdict, expect_admitted: bool, needle: str = "") -> None:
        left = residual(verdict)
        ok = (not left) if expect_admitted else bool(
            [reason for reason in left if needle in reason])
        rows.append({"case": label, "expect_admitted": expect_admitted,
                     "residual_reasons": left[:6], "needle": needle, "passed": ok})

    coverage = product.complex_fused_electric_pair_coverage
    bloch, bloch_pml = _build_complex((3.0, 3.0, 0.0), "periodic",
                                      (0.23, 0.0, 0.0))
    record("bloch_no_sources", coverage(bloch, bloch_pml, ()), True)
    # THE OTHER SEAM'S SOURCE is not this pair's business at all.
    record("bloch_magnetic_source",
           coverage(bloch, bloch_pml, (_Source("B"),)), True)
    # THE IN-SEAM ELECTRIC DEPOSIT IS CARRIED. This is the row that takes the
    # product from one corpus row to sixteen.
    record("bloch_electric_deposit_is_carried",
           coverage(bloch, bloch_pml, (_deposit(bloch, "Ey"),)),
           True)
    record("bloch_electric_source_without_a_deposit_index",
           coverage(bloch, bloch_pml, (_Source("D"),)),
           False, "does not publish the index it writes")
    record("bloch_electric_deposit_refused_when_the_flag_is_held_False",
           _with_flag_false(
               product,
               lambda: coverage(bloch, bloch_pml, (_deposit(bloch, "Ey"),))),
           False, "is electric")
    record("undeclared_sources", coverage(bloch, bloch_pml, None),
           False, "was not declared")
    record("no_pml", coverage(bloch, None, ()), False, "PML")

    walls, walls_pml = _build_complex(
        (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"), (0.0, 0.0, 0.0))
    record("metallic_walls_k0", coverage(walls, walls_pml, ()),
           True)

    folded, folded_pml = _build_complex(
        (3.0, 3.0, 0.0), ("metallic", "metallic", "periodic"),
        (0.0, 0.0, 0.0), mirrors=(("Y", 1),))
    record("folded_grid", coverage(folded, folded_pml, ()),
           False, "mirror plane")

    builder_rows = []
    for label, args, kwargs in (
        ("electric_source_without_a_deposit_index",
         (bloch, bloch_pml, (_Source("D"),)), {}),
        ("undeclared_sources", (bloch, bloch_pml, None), {}),
        ("folded_grid", (folded, folded_pml, ()), {}),
    ):
        plan = product.plan_complex_fused_electric_pair(*args, **kwargs)
        builder_rows.append({"case": label, "plan_is_none": plan is None,
                             "passed": plan is None})

    findings = [row["case"] for row in rows if not row["passed"]]
    findings += [f"builder:{row['case']}" for row in builder_rows
                 if not row["passed"]]
    return {"leg": "predicate", "device": False, "rows": rows,
            "builder_rows": builder_rows, "findings": findings,
            "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 6 — the seam passes are the driver's own
# ---------------------------------------------------------------------------

def seam_binding_leg() -> Dict[str, Any]:
    """``REPLACES`` must name the call sites ``driver.step`` actually runs there.

    Read off the driver source rather than remembered: the whole product is a
    claim about a specific run of consecutive passes, and a driver that grew a
    sixth pass in this seam would leave this kernel silently swallowing it.

    The seam's extent is DERIVED from :data:`fastpath.DRIVER_SLOTS`, not counted
    in lines. It was a literal 16-line window until 2026-09-02, when the dispatch
    surface grew the two fill consults — ``fill_B``/``fill_D`` joined
    ``DRIVER_SLOTS`` and the two folded-far passes gained consults of their own —
    which pushed ``update_E`` from inside 16 lines of ``step_D`` to exactly 16
    below it (driver.py:3315 -> 3331), so the window read as "the seam is not
    where it was". The seam is now walked consult by consult: it runs from the
    ``step_D`` consult to the ``update_E`` consult that closes it, and reaching a
    consult this seam does NOT own before that closer is itself the finding.
    That asserts strictly more than the line count did — a foreign slot
    dispatched inside the seam is now caught, where a window could only notice
    the two endpoints drifting apart.
    """
    from meep_gpu import fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fused_electric_pair as product  # noqa: PLC0415

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
                f"driver.py:{index + 1} shows no source injection in this seam; the "
                f"whole deposit-repair bracket would be bracketing nothing")
    if not windows:
        findings.append("no step_D/update_E seam was found in driver.py at all")
    return {"leg": "seam_binding", "device": False, "windows": windows,
            "declared_replaces": list(product.REPLACES),
            "seam_owns": sorted(owned), "foreign_consults": foreign,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side — drivers, inventory, launch counting
# ---------------------------------------------------------------------------

def build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec, seed: int,
                 electric: bool):
    """One complex-storage PML driver, seeded identically for every route.

    ``electric`` puts a real source IN THIS SEAM. On the magnetic twin's gate the
    equivalent flag put one in the other seam and was free; here it is the whole
    carry family, and every route gets the same source so the comparison stays a
    statement about the weld.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35,  # non-power-of-two, deliberate
        boundaries=boundaries, k_point=k_point, prefer_gpu=True, gpu_id=0,
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
    # SEEDED ON THE E SIDE, not the H side: an all-zero f_w_E makes the history
    # read indistinguishable from a zero and disarms m6 on the first step.
    for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            host = (rng.uniform(-0.05, 0.05, size=shape)
                    + 1j * rng.uniform(-0.05, 0.05, size=shape)).astype(np.complex64)
            array[...] = cp.asarray(np.ascontiguousarray(host))
    return driver


def inventory(driver) -> Dict[str, Any]:
    """Every device volume the step can touch, found by scanning rather than listing."""
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


#: Each auxiliary's record line: (half-integer lattice?, axis). ``fu`` follows its
#: target's dsigu axis on the curl's lattice, ``f_w`` its component's own axis on the
#: constitutive's.
AUXILIARY_LINES: Dict[str, Tuple[bool, int]] = {
    "fu_Bx": (True, 2), "fu_By": (True, 0), "fu_Bz": (True, 1),
    "fu_Dx": (False, 2), "fu_Dy": (False, 0), "fu_Dz": (False, 1),
    "f_w_Hx": (False, 0), "f_w_Hy": (False, 1), "f_w_Hz": (False, 2),
    "f_w_Ex": (True, 0), "f_w_Ey": (True, 1), "f_w_Ez": (True, 2),
}


def outside_every_layer(driver, names) -> List[str]:
    """The auxiliaries whose record line holds no row of the run's division.

    THE AUXILIARY CONTRACT LEAVES THEM UNTOUCHED: an auxiliary word outside its form's
    region is never read or written, so on a division that gives a component's layer
    no cell at all (a 2-D case layering y only gives fu_Bx, fu_By, f_w_Hx, f_w_Hz none)
    the auxiliary keeps its first word for the whole run, on the array path and in the
    kernel alike. Such an array is not a vacuous leg; it is the contract, and it is
    checked as one: it must NOT move.
    """
    from meep_gpu.chunk_map import membership_from_record  # noqa: PLC0415

    record = driver.chunk_bounds_record()
    shape = tuple(int(n) for n in driver.fields.grid.shape)
    out = []
    for name in names:
        if name not in AUXILIARY_LINES:
            continue
        half, axis = AUXILIARY_LINES[name]
        if not membership_from_record(record, half, axis, shape[axis]).any():
            out.append(name)
    return sorted(out)


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
    """The plan bundle installed for one driver, keyed by driver call site."""

    def __init__(self, plans: Dict[str, Any]) -> None:
        self.plans = plans


class _Absorbed:
    """The sentinel left where a driver pass was absorbed by the fused launch."""

    __slots__ = ("name", "absorbed_by")

    def __init__(self, name: str, absorbed_by: Any) -> None:
        self.name = name
        self.absorbed_by = absorbed_by

    def run(self, *_args: Any, **_kwargs: Any) -> None:
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
    """The two separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import complex_fields  # noqa: PLC0415

    curl = complex_fields.plan_complex_pml_curl(driver.fields, driver.pml, "step_D")
    constitutive = complex_fields.plan_complex_constitutive(
        driver.fields, driver.pml, "E")
    missing = [name for name, plan in
               (("complex curl", curl), ("complex constitutive", constitutive))
               if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    # The three seam passes stay on the ARRAY PATH here: no Triton product owns the
    # wall clear, and both symmetry fills are no-ops on every admitted case. All
    # three are counted, so a difference in which of them ran is a counter event
    # rather than an invisible correction. THE INJECTION ALSO STAYS ON THE ARRAY
    # PATH, which is what makes this route the right oracle for the carry family:
    # it is the two certified kernels with the driver's own deposit between them.
    return Route({"step_D": curl, "update_E": constitutive})


def fused_route(plan, driver=None, sources=()):
    """The fused plan in its two slots, BRACKETED when the seam carries a deposit.

    THE BRACKET IS THE SHIPPED ONE. ``deposit_repair.LeadingRepairPlan`` /
    ``TrailingRepairPlan`` are the exact pair ``launch._install_fused_pair``
    (launch.py:1663-1668) puts in these two slots; a harness that assembled its own
    could not license the one that ships.
    """
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
            steps: int, product, mutant: Any = None,
            install_fused: bool = True, freeze_electric: bool = False,
            electric: bool = False
            ) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                             SEED, electric)
    separate = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                            SEED, electric)
    fused = build_driver(cp, dimensions, cell, boundaries, k_point, pml_spec,
                         SEED, electric)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.complex_fused_curl_constitutive_D_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": str(boundaries),
        "k_point": list(k_point), "pml": dict(pml_spec),
        "electric_source_in_the_seam": bool(electric),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_complex_fused_electric_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        if plan is None:
            verdict = product.complex_fused_electric_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
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
            # ARMED: every route's electric seam is inert. All three then agree
            # trivially and only the moved-state census can refuse it.
            frozen = {key: _Absorbed(key, None) for key in SEAM_PASSES}
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
        outside = outside_every_layer(fused, final)
        row["arrays_outside_every_layer"] = outside
        row["auxiliary_contract_violations"] = sorted(set(outside) & set(ever_moved))
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material) - set(outside))
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
    if row.get("auxiliary_contract_violations"):
        failures.append(
            f"an auxiliary outside every layer of the division moved (the auxiliary "
            f"contract): {row['auxiliary_contract_violations']}")
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
        inspect.getsource(product.complex_fused_curl_constitutive_D.fn))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest.

    The connection helper and the shared forms are IMPORTED into the mutant's
    namespace rather than copied, for the same reason the shipped module imports
    them: a second spelling is a second place to be wrong, and a mutant carrying its
    own copy would no longer be testing the shipped arithmetic. THE MODULE NAME MUST
    NOT START WITH ``triton``: Triton's dependency finder leaves every callee whose
    ``__module__`` starts with its package name out of the kernel's cache key, so a
    helper-only mutant could be served another mutant's binary (measured on the
    off-diagonal gate, 2026-10-08).
    """
    header = ("import triton\n"
              "import triton.language as tl\n"
              "from meep_gpu.triton_kernels.complex_fields import _connect\n"
              "from meep_gpu.triton_kernels.forms import (\n"
              "    axis_present, constitutive_form, curl_form, curl_stencil, record_in)\n"
              "PERIODIC = tl.constexpr(0)\n"
              "METALLIC = tl.constexpr(1)\n"
              "RECORD_HALF_INTEGER = tl.constexpr(12)\n"
              "RECORD_PRESENCE = tl.constexpr(24)\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_complex_electric_pair.py", delete=False,
        encoding="utf-8")
    handle.write(header + source.replace("complex_fused_curl_constitutive_D",
                                         kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "complex_electric_pair_mutant_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was MEASURED."""

    def m1_wall_clear_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's step_boundaries(D_stuff), undone.

        Visible only where the walls are real, which is what the metallic cases in
        CASES exist for. NO ``if`` IN THE REPLACEMENT: ``_rewrite_block`` indents
        every replacement line to the block's first line, so a compound statement
        in position 0 produces an empty suite and the mutant fails to IMPORT —
        recorded as unarmed, not as a caught defect."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v1_re = tl.where(at_x, 0.0, v1_re)",
             "v1_im = tl.where(at_x, 0.0, v1_im)",
             "v2_re = tl.where(at_x, 0.0, v2_re)",
             "v2_im = tl.where(at_x, 0.0, v2_im)"],
            ["v1_re = v1_re", "v1_im = v1_im", "v2_re = v2_re", "v2_im = v2_im"])

    def m2_wall_clear_only_on_the_real_plane(source: str) -> Tuple[str, int]:
        """THE COMPLEX-SPECIFIC HAZARD. ``array[face] = 0`` on a complex64 volume
        writes complex zero; a transcription that cleared only the real plane is
        wrong exactly where the imaginary part is nonzero at a wall, and nowhere
        else — which is invisible to any real-field test."""
        hits = 0
        for register, coordinate in (("v1_im", "at_x"), ("v2_im", "at_x"),
                                     ("v0_im", "at_y"), ("v2_im", "at_y"),
                                     ("v0_im", "at_z"), ("v1_im", "at_z")):
            needle = f"{register} = tl.where({coordinate}, 0.0, {register})"
            hits += source.count(needle)
            source = source.replace(needle, f"{register} = {register}")
        return source, hits

    def m13_wall_clear_takes_the_magnetic_maps_component(source: str) -> Tuple[str, int]:
        """THE DEFECT THIS PRODUCT ACTUALLY SHIPPED TO ITS FIRST DEVICE RUN, armed
        so it can never come back unnoticed.

        The magnetic twin's map is ``axis d -> component d``; the D side's is the
        two TANGENTIAL components. Carrying the twin's map writes zero into a plane
        the array path keeps AND leaves live a plane the array path clears — two
        errors in opposite directions on the same grid, both confined to one cell
        of one plane per axis, both invisible on any periodic case. Measured
        uncaught by every no-device leg in this gate and caught in one step by the
        metallic cases (results/run_elecpair1)."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v1_re = tl.where(at_x, 0.0, v1_re)",
             "v1_im = tl.where(at_x, 0.0, v1_im)",
             "v2_re = tl.where(at_x, 0.0, v2_re)",
             "v2_im = tl.where(at_x, 0.0, v2_im)"],
            ["v0_re = tl.where(at_x, 0.0, v0_re)",
             "v0_im = tl.where(at_x, 0.0, v0_im)"])

    def m3_connection_applied_to_every_lane(source: str) -> Tuple[str, int]:
        """The x connection applied on every lane, not only on plane 0 of the
        unshifted operand: the whole-volume version of the same product."""
        hits = 0
        for register, original in (("c1_re", "c_re"), ("c1_im", "c_im"),
                                   ("b2_re", "b_re"), ("b2_im", "b_im")):
            plane = register[-2:]
            needle = f"{register} = tl.where(at_x, r_{plane}, {original})"
            hits += source.count(needle)
            source = source.replace(needle, f"{register} = r_{plane}")
        return source, hits if hits == 4 else 0

    def m4_connection_dropped_on_x(source: str) -> Tuple[str, int]:
        """The x connection thrown away: the Bloch run becomes a plain periodic one.
        Invisible at k = 0 by construction, which is why the case list carries two
        phased rows."""
        return _rewrite_block(source, ["if CNX:"], ["if 0:"])

    def m14_connection_on_the_old_wrapped_plane(source: str) -> Tuple[str, int]:
        """INVERSION (Appendix A): the x connection on the LAST plane, where the
        rolled operand used to carry the phase, instead of the unshifted operand's
        plane 0."""
        hits = 0
        for register in ("c1_re", "c1_im", "b2_re", "b2_im"):
            needle = f"{register} = tl.where(at_x, "
            hits += source.count(needle)
            source = source.replace(needle, f"{register} = tl.where(i == nx - 1, ")
        return source, hits if hits == 4 else 0

    def a1_pairwise_stencil(source: str) -> Tuple[str, int]:
        """INVERSION (Appendix A): the previous pairwise grouping
        ``(g1s - g1) + (g2 - g2s)`` in place of MEEP's left-to-right stencil."""
        needle = "curl2_re = curl_stencil(b_x_re, b2_re, a2_re, a_y_re, dtdx, px, py)"
        return (source.replace(needle,
                               "curl2_re = dtdx * ((b_x_re - b2_re) + (a2_re - a_y_re))"),
                source.count(needle))

    def a2_whole_grid_recurrence(source: str) -> Tuple[str, int]:
        """INVERSION (Appendix A): the split-field recurrence on every cell, the
        previous whole-grid form, instead of each chunk's own."""
        hits = 0
        for needle in ("du0 = record_in(cb, lattice + 8, k)",
                       "du1 = record_in(cb, lattice + 0, i)",
                       "du2 = record_in(cb, lattice + 4, j)"):
            hits += source.count(needle)
            source = source.replace(needle, needle.split(" = ")[0] + " = live")
        return source, hits if hits == 3 else 0

    def a3_constitutive_old_grouping(source: str) -> Tuple[str, int]:
        """INVERSION (Appendix A): the previous two-accumulation grouping
        ``(f + kps*src) - kms*prev`` in place of MEEP's ``f + ((kps*src) -
        (kms*prev))``, on component 1's real part (component 1's own axis, y, is
        the one the 2-D scoring case layers)."""
        needle = "constitutive_form(a_re, prev_re, src_re, kp_1, km_1, dw1)"
        return (source.replace(
            needle, "tl.where(dw1, (a_re + (kp_1 * src_re)) - (km_1 * prev_re), src_re)"),
            source.count(needle))

    def b1_membership_one_row_off(source: str) -> Tuple[str, int]:
        """The y-dsigu membership read one row over: a plane takes the wrong form
        (y is the axis the scoring case layers)."""
        needle = "du2 = record_in(cb, lattice + 4, j)"
        return (source.replace(needle, "du2 = record_in(cb, lattice + 4, j + 1)"),
                source.count(needle))

    def b2_constitutive_select_removed(source: str) -> Tuple[str, int]:
        """The accumulation on every cell: ``f_w`` and the accumulate outside the
        component's own layer, where MEEP stores ``f = src``."""
        import re  # noqa: PLC0415

        pattern = re.compile(r"dw0 = record_in\(cb, [^\n]*\)")
        return pattern.sub("dw0 = live", source), len(pattern.findall(source))

    def b3_fu_store_unmasked(source: str) -> Tuple[str, int]:
        """``fu`` stored on every cell: the dsig-only cells' auxiliary overwritten."""
        needle = "tl.store(u0 + 2 * idx, n0_re, mask=live & du0)"
        return (source.replace(needle, "tl.store(u0 + 2 * idx, n0_re, mask=live)"),
                source.count(needle))

    def b4_presence_ignored(source: str) -> Tuple[str, int]:
        """Both partners taken whatever the record says: the invariant z partner of
        a 2-D grid adds ``(g + g2) - g2`` where MEEP takes the one-partner form."""
        needle = "presence = tl.load(cb + RECORD_PRESENCE)"
        return source.replace(needle, "presence = 7"), source.count(needle)

    def n1_store_order(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: the two parts of ``fu_0`` stored in the other order --
        independent words, so only the instruction order may move."""
        return _rewrite_block(
            source,
            ["tl.store(u0 + 2 * idx, n0_re, mask=live & du0)",
             "tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)"],
            ["tl.store(u0 + 2 * idx + 1, n0_im, mask=live & du0)",
             "tl.store(u0 + 2 * idx, n0_re, mask=live & du0)"])

    def m6_history_read_after_write(source: str) -> Tuple[str, int]:
        """f_w read AFTER it is written: wrong only where kms != 0, i.e. in the PML.

        THE REWRITE APPENDS A RE-READ RATHER THAN MOVING THE ORIGINAL LOAD, and the
        difference is not cosmetic. The magnetic twin can move its load because its
        six statements are one contiguous run at one indent; on this side the
        ``if SCALE:`` block sits between the load and the store, and
        ``_rewrite_block`` re-indents every replacement line to the block's first
        indent — so a rewrite spanning the compound statement would flatten its
        suite, and a rewrite stopping short of it moves the load two statements
        later WITHOUT crossing the store.

        THAT IS EXACTLY WHAT THE FIRST VERSION OF THIS MUTATION DID. It matched, it
        was reported ARMED, and it was semantically a no-op: measured UNCAUGHT on
        the GPU host 2026-08-30 (run_elecpair2) while every other mutation was caught.
        A rewrite that matches is not a rewrite that means anything, and the
        ``rewrite_hits`` guard cannot tell the two apart — only the device can.
        """
        hits = 0
        for register in ("w0", "w1", "w2"):
            source, hit = _rewrite_block(
                source,
                [f"tl.store({register} + 2 * idx, src_re, mask=live & dw{register[1]})",
                 f"tl.store({register} + 2 * idx + 1, src_im, mask=live & dw{register[1]})"],
                [f"tl.store({register} + 2 * idx, src_re, mask=live & dw{register[1]})",
                 f"tl.store({register} + 2 * idx + 1, src_im, mask=live & dw{register[1]})",
                 f"prev_re = tl.load({register} + 2 * idx, mask=live & dw{register[1]}, "
                 f"other=0.0)",
                 f"prev_im = tl.load({register} + 2 * idx + 1, mask=live & "
                 f"dw{register[1]}, other=0.0)"])
            hits += hit
        return source, hits

    def m7_word_index_not_doubled(source: str) -> Tuple[str, int]:
        """The complex word addressing undone on the E history: ``w1 + idx`` reads
        the wrong plane of the wrong cell. The load and the store move TOGETHER, so
        the mutant is self-consistent and only the array-path comparison refuses
        it."""
        hits = 0
        for needle, replacement in (
                ("tl.load(w1 + 2 * idx, mask=live & dw1, other=0.0)",
                 "tl.load(w1 + idx, mask=live & dw1, other=0.0)"),
                ("tl.store(w1 + 2 * idx, src_re, mask=live & dw1)",
                 "tl.store(w1 + idx, src_re, mask=live & dw1)")):
            hits += source.count(needle)
            source = source.replace(needle, replacement)
        return source, hits

    def m8_imaginary_plane_of_the_weld_dropped(source: str) -> Tuple[str, int]:
        """The weld carried on the real plane only: the imaginary source is zero.

        Zeroing the imaginary source is what the mutation's name says and is
        visible wherever the imaginary plane is live — which the Bloch cases make
        it, and the k = 0 seeded-complex cases are the control for.
        """
        hits = 0
        for component in ("0", "1", "2"):
            needle = f"src_im = v{component}_im"
            hits += source.count(needle)
            source = source.replace(needle, "src_im = 0.0")
        return source, hits

    def m10_inv_eps_multiply_dropped(source: str) -> Tuple[str, int]:
        """THE E-SIDE ARM, REMOVED. ``update_E`` becomes ``E += kps*D - kms*prev``
        with no epsilon at all — the H side's arithmetic on the E side's data. It
        has no counterpart in the magnetic twin's table because that kernel has no
        such arm to remove.

        The load is KEPT and only the multiply is dropped: a mutant that also
        dropped the load would leave ``ie`` unbound and fail to compile, which the
        harness records as unarmed rather than as a caught defect."""
        hits = 0
        for component in ("0", "1", "2"):
            source, hit = _rewrite_block(
                source,
                [f"ie = tl.load(e{component} + idx, mask=live, other=0.0)",
                 "src_re = src_re * ie", "src_im = src_im * ie"],
                [f"ie = tl.load(e{component} + idx, mask=live, other=0.0)",
                 "src_re = src_re", "src_im = src_im"])
            hits += hit
        return source, hits

    def m11_inv_eps_index_word_doubled(source: str) -> Tuple[str, int]:
        """``inv_eps`` read at ``2 * idx``: the imaginary neighbour's coefficient
        applied to the real plane. Smooth, converged and wrong, and invisible to
        every shape and dtype check in the tree."""
        hits = 0
        for component in ("0", "1", "2"):
            needle = f"tl.load(e{component} + idx, mask=live, other=0.0)"
            hits += source.count(needle)
            source = source.replace(
                needle, f"tl.load(e{component} + 2 * idx, mask=live, other=0.0)")
        return source, hits

    def m12_inv_eps_applied_after_the_history_store(source: str) -> Tuple[str, int]:
        """The scale moved to the WRONG SIDE of the ``f_w`` store, so the history
        holds ``D`` rather than ``D * inv_eps``.

        Wrong only where ``kms != 0`` — inside the PML — and only from the SECOND
        step, because the first step's ``prev`` is the seeded value on both sides.
        The array-path emulation in leg 3 carries the same flip as a design knob,
        which is what says the defect is visible at all before a device sees it.
        """
        return _scale_after_store(source)

    def m9_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; the connection's real
        product spelled operand-first. Only the PTX may move. It is the control that
        says a caught mutation means something."""
        return n1_store_order(source)

    return (
        ("m1_wall_clear_dropped", "the wall clear's slot", "caught",
         m1_wall_clear_dropped),
        ("m2_wall_clear_only_on_the_real_plane",
         "complex zero is TWO words", "caught",
         m2_wall_clear_only_on_the_real_plane),
        ("m3_connection_applied_to_every_lane", "the single-plane product",
         "caught", m3_connection_applied_to_every_lane),
        ("m4_connection_dropped_on_x", "the Bloch connection", "caught",
         m4_connection_dropped_on_x),
        ("m14_connection_on_the_old_wrapped_plane", "the stored window",
         "caught", m14_connection_on_the_old_wrapped_plane),
        ("a1_pairwise_stencil", "MEEP's left-to-right curl", "caught",
         a1_pairwise_stencil),
        ("a2_whole_grid_recurrence", "each chunk's own curl form", "caught",
         a2_whole_grid_recurrence),
        ("a3_constitutive_old_grouping", "MEEP's constitutive grouping",
         "caught", a3_constitutive_old_grouping),
        ("b1_membership_one_row_off", "the membership line", "caught",
         b1_membership_one_row_off),
        ("b2_constitutive_select_removed", "the constitutive select", "caught",
         b2_constitutive_select_removed),
        ("b3_fu_store_unmasked", "the auxiliary contract", "caught",
         b3_fu_store_unmasked),
        ("b4_presence_ignored", "the partner presence", "caught",
         b4_presence_ignored),
        ("m6_history_read_after_write", "the f_w ordering", "caught",
         m6_history_read_after_write),
        ("m7_word_index_not_doubled", "complex word addressing", "caught",
         m7_word_index_not_doubled),
        ("m8_imaginary_plane_of_the_weld_dropped", "the weld is two registers",
         "caught", m8_imaginary_plane_of_the_weld_dropped),
        ("m10_inv_eps_multiply_dropped", "the E-side SCALE arm", "caught",
         m10_inv_eps_multiply_dropped),
        ("m11_inv_eps_index_word_doubled", "one real coefficient per complex cell",
         "caught", m11_inv_eps_index_word_doubled),
        ("m12_inv_eps_applied_after_the_history_store",
         "the scale is inside the f_w store", "caught",
         m12_inv_eps_applied_after_the_history_store),
        ("m13_wall_clear_takes_the_magnetic_maps_component",
         "the D-side wall map is the twin's COMPLEMENT", "caught",
         m13_wall_clear_takes_the_magnetic_maps_component),
        ("m9_commuted_multiply", "commuted add", "null", m9_commuted_multiply),
    )


def _scale_after_store(source: str) -> Tuple[str, int]:
    """Move each component's inv_eps multiply to AFTER its ``f_w`` store.

    Written as a helper rather than inline in the table because it rewrites three
    separate blocks and ``_rewrite_block`` replaces one run at a time; a rewrite
    that silently caught only the first component would arm a weaker mutation than
    the one the table names.
    """
    hits = 0
    for component, register in (("0", "w0"), ("1", "w1"), ("2", "w2")):
        before = [
            f"ie = tl.load(e{component} + idx, mask=live, other=0.0)",
            "src_re = src_re * ie",
            "src_im = src_im * ie",
            f"tl.store({register} + 2 * idx, src_re, mask=live & dw{component})",
            f"tl.store({register} + 2 * idx + 1, src_im, mask=live & dw{component})",
        ]
        after = [
            f"tl.store({register} + 2 * idx, src_re, mask=live & dw{component})",
            f"tl.store({register} + 2 * idx + 1, src_im, mask=live & dw{component})",
            f"ie = tl.load(e{component} + idx, mask=live, other=0.0)",
            "src_re = src_re * ie",
            "src_im = src_im * ie",
        ]
        source, hit = _rewrite_block(source, before, after)
        hits += hit
    return source, hits


#: Mutation id -> the CASES index whose grid carries the branch it rewrites.
#:
#: THERE IS NO SINGLE CASE THAT SERVES THEM ALL. A wall-clear mutation needs a
#: METALLIC x, and a phase mutation needs a NONZERO k, and CASES carries no grid
#: with both. Naming the case per mutation is what resolves that; asserting one
#: case covers everything is what hid the twin's uncaught m1 on 2026-08-20.
MUTATION_CASE: Dict[str, int] = {
    # metallic_walls_k0 — x IS metallic there, which is what ZM_X needs. Both of
    # these rewrite the ZM_X block, which on the default case (x PERIODIC) is dead
    # code: scored there they would report a real defect as uncaught.
    "m1_wall_clear_dropped": 3,
    "m13_wall_clear_takes_the_magnetic_maps_component": 3,
}

#: The phased, walled case: the default for every mutation that does not need a
#: metallic x. It carries a nonzero k, so the Bloch-phase legs have somewhere to
#: be caught, and a y wall, so ZM_Y is entered.
DEFAULT_MUTATION_CASE = 4


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    """The (index, case) a mutation is scored on."""
    index = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
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
            # A rewrite that matched nothing is a HARNESS defect, not a null: it
            # would report every mutation as uncaught while launching the shipped
            # kernel.
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_complex_fused_D"
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
                      case[5], MUTATION_STEPS, product, mutant=mutant)
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
    # `sources` may be a CALLABLE taking the built fields, and `needle=None` means
    # "this row must be ADMITTED".
    for name, case, needle, sources in (
        ("electric_source_without_a_deposit_index", base,
         "does not publish the index it writes", (_Electric(),)),
        ("electric_deposit_is_carried", base, None,
         lambda fields: (_deposit(fields, "Ey"),)),
        ("magnetic_source_is_the_other_seams", base, None, (_Magnetic(),)),
        ("undeclared_sources", base, "was not declared", None),
    ):
        driver = build_driver(cp, case[1], case[2], case[3], case[4], case[5],
                              SEED, electric=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.complex_fused_electric_pair_coverage(
                driver.fields, driver.pml, bound)
            plan = product.plan_complex_fused_electric_pair(
                driver.fields, driver.pml, bound)
            if needle is None:
                passed = bool(verdict.covered) and plan is not None
            else:
                passed = ((not verdict.covered) and plan is None
                          and any(needle in reason for reason in verdict.reasons))
            rows.append({
                "case": name, "device": True, "covered": bool(verdict.covered),
                "expected": "admitted" if needle is None else f"refused: {needle}",
                "reasons": list(verdict.reasons), "plan_is_none": plan is None,
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
    """What ran this, in the shape the WELD PIPELINE reads.

    ``hostname``, ``device``, ``compute_capability``, ``triton``, ``cupy`` and
    ``cuda_visible_devices`` are the six ``rebind_triton_welds._host_line`` builds a
    weld's ``host`` string from, and it returns None — SKIPPING the entry rather
    than binding it half-known — if any is missing. The compute capability is not
    decoration: ``test_triton_weld_contract`` binds it and the Triton version to the
    record's ``validated_compute_capabilities`` / ``validated_triton_versions``,
    because Triton generates PTX for an ARCHITECTURE and a weld cut on an
    undeclared one must fail until the declaration is widened deliberately.
    """
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
        HERE, "results", "triton_complex_fused_electric_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep'. A run that installs NOTHING "
             "is a MIXED configuration attributable to no policy at all.")
    args = parser.parse_args(argv)

    payload: Dict[str, Any] = {
        "gate": "triton_complex_fused_electric_pair",
        "product": "meep_gpu.triton_kernels.complex_fused_electric_pair",
        "kernel": "complex_fused_curl_constitutive_D",
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": environment(),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "complex_fields.DEFAULT_BLOCK",
                   "subnormal_policy": args.subnormal_policy},
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in (transcription_leg, symmetry_inertness_leg,
                design_sweep_leg, predicate_leg, seam_binding_leg):
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
        # ``release`` IS THE KEY ``gate_provenance.read_verdict`` CONSULTS FIRST,
        # and it is set explicitly here because ``passed`` alone would stamp
        # ``released: True`` on an artifact that measured no bytes on any device.
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

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep and
    # quietly did not is a process whose bytes mean nothing.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import complex_fused_electric_pair as product  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    payload["environment"] = environment(cp)
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "quiet_cases": {case[0]: case[6] for case in CASES},
        "carry_cases": [CASES[index][0] for index in CARRY_CASES],
        "steps_per_mutation": MUTATION_STEPS,
    }
    save(payload, args.out)

    log("\n=== device legs: the QUIET family (no in-seam source) ===")
    for name, dimensions, cell, boundaries, k_point, pml_spec, steps in CASES:
        row = run_leg(cp, f"quiet:{name}", dimensions, cell, boundaries, k_point,
                      pml_spec, steps, product, electric=False)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== device legs: the CARRY family (electric deposit, shipped bracket) ===")
    for index in CARRY_CASES:
        name, dimensions, cell, boundaries, k_point, pml_spec, steps = CASES[index]
        row = run_leg(cp, f"carry:{name}", dimensions, cell, boundaries, k_point,
                      pml_spec, steps, product, electric=True)
        passed, failures = verdict_of(row, require_launches=steps,
                                      require_repairs=True)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== armed harness mutations ===")
    base = CASES[0]
    row = run_leg(cp, "armed:no_substitution", base[1], base[2], base[3], base[4],
                  base[5], 3, product, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_electric_seam", base[1], base[2], base[3],
                  base[4], base[5], 3, product, freeze_electric=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.complex_fused_curl_constitutive_D)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    # A DECLARED NULL MUST BE CONFIRMED, not merely permitted. Under a weaker rule
    # any expectation other than "caught" passes unconditionally, so moving a
    # stubbornly uncaught mutation to a null spelling would silence it rather than
    # explain it. Every mutation must also have been ARMED: a rewrite that hit
    # nothing measured nothing, whatever it then reported.
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


if __name__ == "__main__":
    raise SystemExit(main())
