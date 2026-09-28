"""Byte gate for the folded DISPERSIVE fused electric pair: ``step_D`` into ``update_E``.

DEVICE STATUS: **UNRUN.** This module is written to be run on a CUDA host with an
    idle device; nothing in this tree may cite it as a release until an artifact
    exists with ``device_status: RUN`` and ``release.released: true``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.folded_dispersive_fused_pair.folded_dispersive_fused_pair_coverage`
admits, ONE launch of ``folded_dispersive_fused_curl_constitutive_D`` — bracketed by
the SHIPPED deposit repair wherever the seam carries an electric deposit — leaves the
engine in a state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_D`` / the electric injection /
    ``fill_symmetry_bc_D`` / ``zero_metal_D`` / ``fill_folded_far_ghosts_D`` /
    ``update_E``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the folded curl
    (``symmetry.plan_folded_pml_curl`` on ``step_D``), the mirror ghost fill
    (``symmetry.plan_mirror_ghost_fill`` on family ``D``) and the FOLDED DISPERSIVE
    constitutive (``folded_dispersive_update_e.plan_folded_dispersive_constitutive``),
    with ``zero_metal_D`` and ``fill_folded_far_ghosts_D`` on the array path in both
    routes because no Triton product owns either,

over every allocated volume: the primaries, the split-field PML auxiliaries, the
constitutive ``f_w`` history AND EVERY POLE VOLUME — ``P``, ``P_prev`` and the shared
scratch of every registered susceptibility. Comparison is on the uint32 view.
``allclose`` appears nowhere.

WHAT IS NEW HERE, AND IT IS THE ONLY THING THAT IS
==================================================

The curl half, the two fills and the wall clear are BYTE-COPIES of
:func:`~meep_gpu.triton_kernels.folded_fused_pair.folded_fused_curl_constitutive_D`,
which released 2026-08-20/21. What this product adds is the pole chain, in two
places, and the second one is the whole hazard:

    owned cell   src = (v  - sum P[idx]) * inv_eps[idx]
    ghost cell   src = (gh - sum P[dst]) * inv_eps[dst]

**THE GHOST READS THE POLES AT ITS OWN INDEX.** The array path's fills write ``D``
and never ``P`` (stepping.py:1497-1498, :1529-1532), and ``update_E`` then runs over
the whole stored extent, so at an imaged cell it reads that cell's own ``P``. A
kernel that imaged the SOURCE lane's poles instead would be a smooth, plausible,
wrong field on every folded plane. :data:`MUTATION_TABLE` arms exactly that
(``mp1``), plus a reversed chain, a pre-summed chain and a dropped chain.

WHAT THE GATE REFUSES TO INFER
==============================

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is counted
  through a proxy that owns the kernel object, every substitution is counted per
  route in the installer, and one ARMED HARNESS MUTATION removes the substitution so
  the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** Every compared array must
  MOVE during the leg; a leg whose electric trio is frozen in all three routes is
  armed and must be caught by the moved-state census.
* **A bracket that changes nothing is not load-bearing.** Every CARRY case is run
  TWICE — once with the shipped :class:`~meep_gpu.deposit_repair.LeadingRepairPlan` /
  :class:`~meep_gpu.deposit_repair.TrailingRepairPlan` pair and once WITHOUT it — and
  the unbracketed run MUST DIVERGE. A carry family whose null control agreed would
  be measuring a seam that carried no deposit.
* **A mutation scored on a grid that never enters its branch measures nothing.**
  Every carry block is guarded on a named axis's fold code, and the pole arms on
  ``NP?``. :data:`MUTATION_CASE` names, per mutation, the case whose fold set and
  pole set actually reach the lines it rewrites, and :func:`mutation_case_for`
  refuses a pairing that does not.

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_dispersive_fused_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_dispersive_fused_pair.py \\
        --out parity/meep_gpu/results/<fresh-dir>
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

SEED = 20260831

#: Temporary mutant modules, deleted at the end of the run.
_TEMPORARY: List[str] = []

#: The susceptibility set every case registers unless it names its own. Two poles on
#: DIFFERENT component sets, so ``NP0``/``NP1``/``NP2`` are NOT all equal — (2, 1, 1)
#: — and the per-component unroll is exercised asymmetrically: a kernel that used
#: component 0's count for all three would pass a symmetric fixture.
#:
#: (frequency, gamma, (sigma_Ex, sigma_Ey, sigma_Ez)). A zero sigma makes the term
#: NOT drive that component (``PolarizationState._driven``, dispersion.py:641-643),
#: which is how the counts are made to differ.
DEFAULT_POLES: Tuple[Tuple[float, float, Tuple[float, float, float]], ...] = (
    (0.58, 0.04, (0.35, 0.00, 0.25)),
    (0.91, 0.08, (0.20, 0.45, 0.00)),
)

#: TWO POLES ON EVERY COMPONENT — (2, 2, 2) — and the ORDER mutations are scored
#: here rather than on :data:`DEFAULT_POLES`. A rewrite that swaps the first two arms
#: is the reversal only where BOTH are live; on a component carrying one pole the
#: same rewrite reads a dead slot instead, which is a different defect and would let
#: the leg report the wrong thing as caught.
TWO_POLES_EVERYWHERE: Tuple[Tuple[float, float, Tuple[float, float, float]], ...] = (
    (0.58, 0.04, (0.35, 0.22, 0.25)),
    (0.91, 0.08, (0.20, 0.45, 0.31)),
)

#: A SINGLE-POLE set, for the case that must show a one-pole unroll is not the
#: two-pole one. A component with ONE pole and a component with NONE both appear.
ONE_POLE: Tuple[Tuple[float, float, Tuple[float, float, float]], ...] = (
    (0.58, 0.04, (0.35, 0.00, 0.25)),
)

#: (name, cell, boundaries, mirrors, poles, steps).
#:
#: THE PARITIES, THE OUTER DECLARATIONS AND THE WALL SET ARE ALL SWEPT, exactly as
#: the plain folded pair's gate sweeps them, and for its reasons: an EVEN plane makes
#: the parity mutations invisible (the multiply is by +1); a folded METALLIC axis
#: makes every ``FAR_a`` block compile-time absent; and a folded PERIODIC axis beside
#: a wall is the only shape where the near ghost's parity-then-clear order and the
#: far ghost's clear-then-parity order can disagree.
#:
#: THE Z AXIS IS PERIODIC IN EVERY 2-D CASE, NOT METALLIC. A 2-D cell is
#: translationally invariant along z; a metallic declaration there is not a wall but
#: a polarization filter, and ``Grid`` refuses it by name (grid.py:842-877).
#:
#: THE 3-D CASES ARE NOT DECORATION. In a 2-D run z is invariant, so ``FAR_Z``,
#: ``NEAR_Z``, every ``gv_2*`` register, the ``kp_f2``/``km_f2`` pair AND component
#: 2's whole pole chain at a ghost are compile-time absent — a third of this kernel
#: would be shipped code no leg executed.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any,
                   Tuple[Tuple[str, int], ...], Any, int], ...] = (
    ("even_y_fold_metallic", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", 1),),
     DEFAULT_POLES, 10),
    ("odd_y_fold_metallic", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", -1),),
     DEFAULT_POLES, 10),
    ("two_folds_metallic", (3.0, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("X", 1), ("Y", -1)),
     DEFAULT_POLES, 10),
    ("x_fold_periodic_even", (3.0, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("X", 1),),
     DEFAULT_POLES, 10),
    ("x_fold_periodic_odd", (37.0 / 12.0, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("X", -1),),
     DEFAULT_POLES, 10),
    ("two_folds_periodic", (3.0, 37.0 / 12.0, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", -1)), DEFAULT_POLES, 10),
    # ONE POLE, and a component driven by NONE. The pole chain's `if NP > k:` arms
    # are constexpr-unrolled per component, so a single-pole grid compiles a
    # DIFFERENT kernel from the two-pole one; a table that only ever ran two poles
    # would leave that specialization unexecuted.
    ("one_pole_y_fold", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", -1),),
     ONE_POLE, 10),
    # TWO POLES ON EVERY COMPONENT. The ORDER mutations are scored here; see
    # TWO_POLES_EVERYWHERE for why a (2, 1, 1) grid would score them wrong.
    ("two_poles_everywhere_y_fold", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", -1),),
     TWO_POLES_EVERYWHERE, 10),
    # THE 3-D CASES.
    ("xyz_all_folded_3d", (2.0, 2.0, 2.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", -1), ("Z", 1)), DEFAULT_POLES, 10),
    # A FAR FOLD, A NEAR FOLD AND A WALL AT ONCE — the only shape in which the
    # CHAINED composite and a raw parity product differ, and (with the ODD y plane)
    # the only one on which `m12` is not a structural null.
    ("x_periodic_y_fold_z_wall_3d", (2.0, 2.0, 1.5),
     {"x": "periodic", "y": "metallic", "z": "metallic"},
     (("X", -1), ("Y", -1)), DEFAULT_POLES, 10),
)

def case_index(name: str) -> int:
    """The index of a case BY NAME. Keyed by name so inserting a case cannot
    silently re-point every mutation at a different grid."""
    for index, case in enumerate(CASES):
        if case[0] == name:
            return index
    raise AssertionError(f"no case named {name!r}; known: "
                         f"{[case[0] for case in CASES]}")


#: The CARRY family: the same grids run with a real ELECTRIC deposit IN THIS SEAM.
#: Named so the two families cannot drift apart.
#:
#: THIS IS THE PRODUCT, not an extra. All four corpus rows of this board cell declare
#: an electric source, so a gate that only ran the quiet family would certify a
#: kernel the corpus never reaches. One folded METALLIC grid, one folded PERIODIC
#: grid (whose far carry images the deposit onto a second plane) and one 3-D grid.
CARRY_CASES: Tuple[str, ...] = ("odd_y_fold_metallic", "x_fold_periodic_odd",
                                "x_periodic_y_fold_z_wall_3d")

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

#: Steps per carry / null-control leg.
CARRY_STEPS = 6

#: The driver call sites this product spans, in driver order (driver.py:3292-3304).
SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
)

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this tuple
#: is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: THE READ-ONLY MATERIAL VOLUMES, under the names ``inventory`` really finds. A set
#: of names that matches nothing disables BOTH the vacuity floor (which would then
#: demand a read-only input move) and ``material_changed`` (which could never fire).
MATERIAL = ("eps", "inv_eps")

#: Pole state that the constitutive READS and ``update_P`` WRITES. Compared like any
#: other volume — this kernel loads ``P`` at eight compiled slots per component and a
#: wrong index there is exactly the defect this product could have.
#:
#: The SIGMA volumes are read-only inputs and are counted as material: a leg in which
#: one moved is a fixture failure, not a kernel result.
POLE_MATERIAL_PREFIX = "sigma"

#: Private ``Fields`` scratch that is NOT physical state, and is out of the
#: comparison for that reason.
#:
#: ``_fmp_scratch`` is the buffer ``Fields.displacement_minus_polarization``
#: (fields.py:1096) allocates on first use to hold ``(D - sum P)``. The ARRAY path
#: calls that function from ``stepping.update_E`` and so allocates it; the fused and
#: separate routes read the pole volumes directly and never do — and on a CARRY leg
#: the fused route allocates it too, through ``deposit_repair.apply``. So its
#: presence differs by ROUTE and by LEG while carrying no state either route reads
#: across a step. MEASURED on the first device run of this gate: it was the ONLY
#: asymmetry, and ``fused_vs_separate_certified_products`` was already ``None``.
#:
#: THE RULE IS THE LEADING UNDERSCORE, not this list: a private attribute is not
#: physical state, and every PUBLIC volume stays in the comparison. The list is the
#: tripwire for that rule, so a private name the scan starts dropping is visible.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)


def log(message: str) -> None:
    print(message, flush=True)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input and "
            f"material_changed would never fire. Fix the names, do not loosen the "
            f"floor.")


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    # THE POLICY STAMP IS RE-READ, NOT CARRIED. Taken once at install time it records
    # every counter at zero, because nothing had compiled yet.
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
    """Every file this run binds itself to, KEYED BY ITS REPO-RELATIVE PATH.

    Not by a nickname: every reader of this map — ``build_triton_fusion_matrix``'s
    ``GATE_BOUND`` branch and the weld-record walk — resolves each key against the
    tree and re-hashes it, so a key that is not a path resolves to nothing and is
    reported as DRIFT on every cut.
    """
    names = (
        "meep_gpu/triton_kernels/folded_dispersive_fused_pair.py",
        "meep_gpu/triton_kernels/folded_fused_pair.py",
        "meep_gpu/triton_kernels/folded_dispersive_update_e.py",
        "meep_gpu/triton_kernels/dispersive_update_e.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/dispersive_fused_pair.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/dispersion.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_folded_dispersive_fused_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Reading the shipped kernel text
# ---------------------------------------------------------------------------

_SOURCE_FILE = {
    "pml_curl_step_folded": "symmetry.py",
    "constitutive_step_dispersive": "dispersive_update_e.py",
    "fused_curl_dispersive_E": "dispersive_fused_pair.py",
    "folded_fused_curl_constitutive_D": "folded_fused_pair.py",
    "_carry_ghost_E": "folded_fused_pair.py",
    "folded_dispersive_fused_curl_constitutive_D": "folded_dispersive_fused_pair.py",
    "_carry_ghost_E_dispersive": "folded_dispersive_fused_pair.py",
    "_subtract_poles_at": "folded_dispersive_fused_pair.py",
}


def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported: this leg has to run on a host with no
    Triton, where importing the kernel module's jit bodies is impossible, so the
    transcription check bites at the merge bar rather than only on a device run.

    The text is exact — never ``ast.unparse``d — because what is being checked is the
    PARENTHESISATION, and unparsing re-derives minimal parentheses and would silently
    rewrite ``dtdx * ((c_y - c) + (b - b_z))`` into a different grouping.
    """
    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels", _SOURCE_FILE[name])
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


def _statements(text: str) -> List[str]:
    """Executable lines, comments and blanks removed, indentation normalised."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.strip():
            out.append(line.strip())
    return out


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace one consecutive run of statements, matched on STRIPPED text.

    Matching on stripped lines rather than on an exact substring is what keeps a
    rewrite armed across a reindentation or a trailing comment. A mutation that
    silently stops matching reports a real defect as uncaught.
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


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, read off the shipped source
# ---------------------------------------------------------------------------

#: The arithmetic lines the fused body must reproduce VERBATIM from the folded curl.
#: Each is a line whose grouping decides float32 bits.
CURL_LINES = (
    "curl0 = dtdx * ((c_y - c) + (b - b_z))",
    "curl1 = dtdx * ((a_z - a) + (c - c_x))",
    "curl2 = dtdx * ((b_x - b) + (a - a_y))",
    "n0 = ((p0 * km_y) - curl0) * si_y",
    "n1 = ((p1 * km_z) - curl1) * si_z",
    "n2 = ((p2 * km_x) - curl2) * si_x",
)

#: The D-family wall clear — the same six rows ``stepping._zero_metal`` writes.
ZERO_METAL_LINES = (
    "v1 = tl.where(at_x, 0.0, v1)",
    "v2 = tl.where(at_x, 0.0, v2)",
    "v0 = tl.where(at_y, 0.0, v0)",
    "v2 = tl.where(at_y, 0.0, v2)",
    "v0 = tl.where(at_z, 0.0, v0)",
    "v1 = tl.where(at_z, 0.0, v1)",
)

#: The constitutive accumulation on the cell the lane owns. Two separate
#: accumulations, left to right; flattening them is a different float32 number and
#: neither multiply may contract into an FMA.
CONSTITUTIVE_LINES = (
    "a0v = a0v + kp_0 * src0",
    "a0v = a0v - km_0 * prev0",
    "a1v = a1v + kp_1 * src1",
    "a1v = a1v - km_1 * prev1",
    "a2v = a2v + kp_2 * src2",
    "a2v = a2v - km_2 * prev2",
)

#: ``symmetry.pml_curl_step_folded``'s TOP-PLANE mask, ``BACKWARD`` arm, verbatim.
TOP_PLANE_LINES = (
    "curl0 = tl.where(last_x, 0.0, curl0)",
    "curl1 = tl.where(last_y, 0.0, curl1)",
    "curl2 = tl.where(last_z, 0.0, curl2)",
)

#: ``_carry_ghost_E_dispersive``'s statements: the certified
#: ``folded_fused_pair._carry_ghost_E`` with its ``src`` line split into the pole
#: chain and the same inverse-epsilon multiply, and NOTHING else moved.
CARRY_GHOST_LINES = (
    "tl.store(f + dst, ghost, mask=mask)",
    "prev = tl.load(w + dst, mask=mask, other=0.0)",
    "src = s * tl.load(ie + dst, mask=mask, other=0.0)",
    "tl.store(w + dst, src, mask=mask)",
    "acc = tl.load(e + dst, mask=mask, other=0.0)",
    "acc = acc + kp_d * src",
    "acc = acc - km_d * prev",
    "tl.store(e + dst, acc, mask=mask)",
)

#: The eight pole arms, as :func:`_subtract_poles_at` spells them. Each must appear
#: in the shared helper AND its shape must be the certified body's.
POLE_ARMS = tuple(
    (f"if NP > {k}:", f"base = base - tl.load(p{k} + where, mask=mask, other=0.0)")
    for k in range(8))

#: The certified arms this helper is a transcription of, in
#: ``dispersive_update_e.constitutive_step_dispersive``'s own register names.
CERTIFIED_POLE_ARMS = tuple(
    (f"if NP0 > {k}:", f"s0 = s0 - tl.load(a{k} + idx, mask=live, other=0.0)")
    for k in range(8))


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line of the fused body traced to the source it came from."""
    fused = _statements(_shipped_text("folded_dispersive_fused_curl_constitutive_D"))
    carry_ghost = _statements(_shipped_text("_carry_ghost_E_dispersive"))
    poles = _statements(_shipped_text("_subtract_poles_at"))
    plain_fused = _statements(_shipped_text("folded_fused_curl_constitutive_D"))
    plain_ghost = _statements(_shipped_text("_carry_ghost_E"))
    folded_curl = _statements(_shipped_text("pml_curl_step_folded"))
    dispersive = _statements(_shipped_text("constitutive_step_dispersive"))
    zm_source = _statements(_shipped_text("fused_curl_dispersive_E"))

    findings: List[str] = []
    for line in CURL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the curl line {line!r}")
        if line not in folded_curl:
            findings.append(
                f"symmetry.pml_curl_step_folded no longer contains {line!r}; the "
                f"fused body's transcription source has moved")
        if line not in plain_fused:
            findings.append(
                f"folded_fused_pair's kernel no longer contains {line!r}; this body "
                f"is a copy of it and the two must not drift")
    for line in ZERO_METAL_LINES:
        if line not in fused:
            findings.append(
                f"the fused body does not contain the wall-clear line {line!r}")
        if line not in zm_source:
            findings.append(
                f"dispersive_fused_pair's ZM block no longer contains {line!r}; the "
                f"wall clear's transcription source has moved")
    for line in CONSTITUTIVE_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain {line!r}")
        if line not in plain_fused:
            findings.append(
                f"folded_fused_pair's kernel no longer contains {line!r}")
    for line in TOP_PLANE_LINES:
        if line not in fused:
            findings.append(
                f"the fused body does not contain the top-plane mask line {line!r}; "
                f"the far carry writes the plane that mask un-owns")
        if line not in folded_curl:
            findings.append(
                f"symmetry.pml_curl_step_folded no longer contains {line!r}")
    masks = [line for line in fused if line.startswith("curl")
             and "tl.where(last_" in line]
    if len(masks) != 3:
        findings.append(
            f"the fused body carries {len(masks)} top-plane mask lines, not 3; the D "
            f"family's Yee shifts put shift 1 on the component's OWN axis only "
            f"(symmetry.py:294-301)")

    # THE ONE SUBSTITUTION, at the owned cell. `(v - sum P) * inv_eps`, with the
    # subtraction through the shared helper at THIS lane's own index and D on the
    # LEFT of the multiply (stepping.py:1011-1013).
    for index, register in ((0, "v0"), (1, "v1"), (2, "v2")):
        want = (f"s{index} = _subtract_poles_at({register}, ")
        if not any(line.startswith(want) for line in fused):
            findings.append(
                f"the fused body does not form s{index} through _subtract_poles_at "
                f"from {register}: the owned cell's constitutive source is "
                f"(D - sum P), not D")
        product = f"src{index} = s{index} * tl.load(ie{index} + idx"
        if not any(line.startswith(product) for line in fused):
            findings.append(
                f"the fused body does not form {product!r}: the E product is "
                f"(D - sum P) * inv_eps with the displacement on the LEFT "
                f"(stepping.py:1011-1013)")
    # ...and the PLAIN product must be GONE. A body that still formed `v0 * inv_eps`
    # anywhere would be the non-dispersive constitutive left standing beside the
    # dispersive one.
    for register in ("src0 = v0 * tl.load(ie0", "src1 = v1 * tl.load(ie1",
                     "src2 = v2 * tl.load(ie2"):
        if any(line.startswith(register) for line in fused):
            findings.append(
                f"the fused body still forms the PLAIN product {register!r}; the "
                f"dispersive source is (D - sum P), not D")

    # THE POLE CHAIN, against the certified body it transcribes. Eight arms, in
    # order, each subtracting ONE pole — never pre-summed, never reordered.
    for guard, arm in POLE_ARMS:
        if guard not in poles or arm not in poles:
            findings.append(
                f"_subtract_poles_at does not carry the arm {guard!r} / {arm!r}")
    for guard, arm in CERTIFIED_POLE_ARMS:
        if guard not in dispersive or arm not in dispersive:
            findings.append(
                f"dispersive_update_e.constitutive_step_dispersive no longer "
                f"contains {guard!r} / {arm!r}; the pole chain's transcription "
                f"source has moved")
    # The helper must be EXACTLY the eight arms and the return — no accumulator, no
    # pre-sum, no reordering. Counted rather than eyeballed.
    subtractions = [line for line in poles if line.startswith("base = base - ")]
    if len(subtractions) != 8:
        findings.append(
            f"_subtract_poles_at carries {len(subtractions)} subtractions, not 8: "
            f"the chain must be eight separate left-to-right subtractions, because "
            f"pre-summing or reordering is a different float32 number at two or "
            f"more poles")
    # ONE load per arm. Two on one line would be a pre-summed pair.
    if any(line.count("tl.load(") != 1 for line in subtractions):
        findings.append(
            f"_subtract_poles_at forms a SUM of poles somewhere ({subtractions}); "
            f"D - (P0 + P1) is a different float32 number from (D - P0) - P1")

    # _carry_ghost_E_dispersive is the certified carry, MOVED and not rewritten...
    for line in CARRY_GHOST_LINES:
        if line not in carry_ghost:
            findings.append(f"_carry_ghost_E_dispersive does not contain {line!r}")
    # ...and every statement it shares with the certified one must be spelled the
    # same way. The ONLY licensed difference is the `src` line, which splits into the
    # pole chain plus the same multiply.
    shared = [line for line in plain_ghost
              if not line.startswith("src = ghost * ")
              and not line.startswith("def ")]
    if len(shared) != 7:
        findings.append(
            f"folded_fused_pair._carry_ghost_E carries {len(shared)} shared "
            f"statements, not the 7 this transcription is against; its body has "
            f"moved and this comparison no longer says what it says")
    for line in shared:
        if line not in carry_ghost:
            findings.append(
                f"_carry_ghost_E_dispersive drops {line!r}, which the certified "
                f"folded_fused_pair._carry_ghost_E carries: the only licensed "
                f"difference between the two is the source line")
    # THE GHOST READS ITS OWN POLES. `where` is bound to `dst`, never to a source
    # index — this is the defect the product could have and the one no per-owned-cell
    # comparison would see.
    # The call spans two physical lines, so the ARGUMENT line is checked beside the
    # opening one rather than inside it.
    opening = [index for index, line in enumerate(carry_ghost)
               if line.startswith("s = _subtract_poles_at(ghost, ")]
    if (len(opening) != 1
            or carry_ghost[opening[0] + 1] != "dst, mask, NP)"):
        findings.append(
            f"_carry_ghost_E_dispersive does not subtract the poles at dst: "
            f"{carry_ghost!r}. The array path's fills write D and never P "
            f"(stepping.py:1497-1498, :1529-1532), so update_E at an imaged cell "
            f"reads THAT cell's poles")

    # The ownership restructure, structurally: the destination lane's D / E / f_w_E
    # traffic must be MASKED OFF, not merely unused.
    for needle in ("mask=own0", "mask=own1", "mask=own2"):
        if sum(1 for line in fused if needle in line) < 6:
            findings.append(
                f"{needle} appears on fewer than six lines; the destination lane's "
                f"D/f_w_E/E traffic is not fully masked off")
    # Every ghost goes through the device function: no carry may open-code it.
    if any(line.startswith(f"tl.store(f{index} + dst") for index in (0, 1, 2)
           for line in fused):
        findings.append(
            "the fused body open-codes a ghost store; every imaged cell must go "
            "through _carry_ghost_E_dispersive")

    # THE COEFFICIENT REGISTERS ARE LOADED TWICE, AND THAT IS THE D GEOMETRY.
    for stem, coordinate, top in (("kp0", "i", "top_x"), ("km0", "i", "top_x"),
                                  ("kp1", "j", "top_y"), ("km1", "j", "top_y"),
                                  ("kp2", "k", "top_z"), ("km2", "k", "top_z")):
        loads = [line for line in fused if f"tl.load({stem} + " in line]
        wanted = {f"tl.load({stem} + {coordinate},", f"tl.load({stem} + {top},"}
        found = {needle for needle in wanted
                 if any(needle in line for line in loads)}
        if len(loads) != 2 or found != wanted:
            findings.append(
                f"{stem} is loaded {len(loads)} times ({loads}); it must be loaded "
                f"exactly twice — once at {coordinate} for the owned cell and every "
                f"near-only ghost, once at {top} for the far half")

    # EVERY CARRY PASSES ITS COMPONENT'S OWN POLE GROUP AND ITS OWN COUNT. A carry
    # that handed component 0's poles to component 2 would be caught on the device
    # only where the two counts differ, which is why it is also read off the text.
    groups = {0: "a0, a1, a2, a3, a4, a5, a6, a7,",
              1: "b0, b1, b2, b3, b4, b5, b6, b7,",
              2: "c0, c1, c2, c3, c4, c5, c6, c7,"}
    for index, group in groups.items():
        calls = sum(1 for line in fused if line.startswith(
            f"_carry_ghost_E_dispersive(f{index}, w{index}, e{index}, ie{index},"))
        # The GROUP appears on: the kernel signature, the owned cell's chain call,
        # and one continuation line per carry. Counted over STATEMENTS so a trailing
        # comment cannot change the answer.
        with_group = sum(1 for line in fused if group in line)
        chains = sum(1 for line in fused if f", NP{index})" in line)
        if calls != 7:
            findings.append(
                f"component {index} makes {calls} carry calls, not the 7 subsets of "
                f"{{near_a, near_b, far_m}} a D component can own")
        if with_group != calls + 2:
            findings.append(
                f"component {index}'s pole group appears on {with_group} lines, not "
                f"the {calls + 2} it must (the signature, the owned cell's chain and "
                f"one per carry); a carry that handed another component's poles "
                f"would be invisible wherever the two counts agree")
        if chains != calls + 1:
            findings.append(
                f"NP{index} appears on {chains} chain calls; it must be on all "
                f"{calls} carries and on the owned cell")
    return {
        "leg": "transcription",
        "device": False,
        "fused_statements": len(fused),
        "carry_ghost_statements": len(carry_ghost),
        "pole_helper_statements": len(poles),
        "curl_lines_checked": len(CURL_LINES),
        "top_plane_lines_checked": len(TOP_PLANE_LINES),
        "zero_metal_lines_checked": len(ZERO_METAL_LINES),
        "constitutive_lines_checked": len(CONSTITUTIVE_LINES),
        "carry_ghost_lines_checked": len(CARRY_GHOST_LINES),
        "pole_arms_checked": len(POLE_ARMS),
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the predicate, on real NumPy drivers
# ---------------------------------------------------------------------------

def _numpy_driver(cell, boundaries, mirrors, poles, electric=False,
                  magnetic=True):
    """A real folded dispersive driver on NumPy — no CUDA, no Triton."""
    from meep_gpu.dispersion import Susceptibility  # noqa: PLC0415
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    dimensions = 3 if float(cell[2]) > 0.0 else 2
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
        prefer_gpu=False)
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    driver.set_epsilon(np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)))
    folded = {axis.lower() for axis, _phase in mirrors}
    layers = {key: ({"high": 5} if key in folded else 5)
              for key in "xyz"[:dimensions]}
    driver.setup_pml(layers)
    for frequency, gamma, sigma_values in poles:
        driver.add_susceptibility(
            Susceptibility(frequency=frequency, gamma=gamma),
            {component: np.full(shape, value, dtype=np.float32)
             for component, value in zip(("Ex", "Ey", "Ez"), sigma_values)})
    center = [0.0, 0.0, 0.0]
    for axis_name, _phase in mirrors:
        axis = "XYZ".index(axis_name.upper())
        center[axis] = 0.25 * (cell[axis] / 2.0)
    if magnetic:
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    if electric:
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    return driver


#: The clause this leg is about: on a NumPy host every Triton predicate carries
#: "array module is 'numpy', not cupy", which is a fact about the harness and not
#: about the configuration. It is the ONLY reason this leg subtracts.
HOST_ARRAY_MODULE_REASON = "not cupy"


def _residual(verdict) -> List[str]:
    return [reason for reason in verdict.reasons
            if HOST_ARRAY_MODULE_REASON not in reason]


def predicate_leg() -> Dict[str, Any]:
    """Every case admitted, and the DISJOINTNESS with ``folded_fused_pair``, measured.

    Two predicates admitting one slot leaves ``_select_slot`` unable to choose and
    the slot UNSELECTED — a silent coverage LOSS, not an error — so the disjointness
    is asserted in BOTH directions on the same two grids rather than argued.
    """
    from meep_gpu.triton_kernels import folded_dispersive_fused_pair as product  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_fused_pair as plain  # noqa: PLC0415

    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    for name, cell, boundaries, mirrors, poles, _steps in CASES:
        for electric in (False, True):
            driver = _numpy_driver(cell, boundaries, mirrors, poles,
                                   electric=electric)
            try:
                verdict = product.folded_dispersive_fused_pair_coverage(
                    driver.fields, driver.pml, tuple(driver._sources))
                residual = _residual(verdict)
                rows.append({"case": name, "electric_source": electric,
                             "residual_reasons": residual,
                             "shape": list(driver.shape)})
                if residual:
                    findings.append(
                        f"case {name!r} (electric={electric}) is refused for "
                        f"{residual}; every case this gate scores must be one the "
                        f"shipped predicate admits")
            finally:
                driver.close()

    # DISJOINTNESS, both directions, on ONE grid each.
    name, cell, boundaries, mirrors, poles, _steps = CASES[0]
    driver = _numpy_driver(cell, boundaries, mirrors, poles)
    try:
        mine = product.folded_dispersive_fused_pair_coverage(
            driver.fields, driver.pml, tuple(driver._sources))
        theirs = plain.folded_fused_pair_coverage(
            driver.fields, driver.pml, tuple(driver._sources))
        poled = {"grid": "a POLED fold", "dispersive": _residual(mine),
                 "plain": _residual(theirs)}
        if _residual(mine):
            findings.append(f"the dispersive pair refuses a poled fold: {poled}")
        if not _residual(theirs):
            findings.append(
                "folded_fused_pair ADMITS a poled fold; both products would claim "
                "the slot and _select_slot would leave it unselected")
    finally:
        driver.close()
    driver = _numpy_driver(cell, boundaries, mirrors, ())
    try:
        mine = product.folded_dispersive_fused_pair_coverage(
            driver.fields, driver.pml, tuple(driver._sources))
        theirs = plain.folded_fused_pair_coverage(
            driver.fields, driver.pml, tuple(driver._sources))
        unpoled = {"grid": "an UNPOLED fold", "dispersive": _residual(mine),
                   "plain": _residual(theirs)}
        if not _residual(mine):
            findings.append(
                "the dispersive pair ADMITS an unpoled fold; that configuration is "
                "folded_fused_pair's and both would claim the slot")
        if _residual(theirs):
            findings.append(
                f"folded_fused_pair refuses an unpoled fold: {unpoled}; the "
                f"disjointness this product rests on is not what it says it is")
    finally:
        driver.close()

    # THE PLAN BUILDS, AND IT BUILDS THE PLAN THE ENGINE WOULD. On NumPy the shipped
    # predicate refuses for the host clause, so the plan builder returns None here by
    # design; what is checked is that it returns None rather than RAISING, which is
    # the contract every builder in this package holds.
    driver = _numpy_driver(cell, boundaries, mirrors, poles)
    try:
        built = product.plan_folded_dispersive_fused_pair(
            driver.fields, driver.pml, tuple(driver._sources))
        if built is not None:
            findings.append(
                "the plan builder returned a plan on a NumPy host; the predicate "
                "refuses that array module and None is the only legal answer")
    except Exception as exc:  # noqa: BLE001
        findings.append(
            f"the plan builder RAISED on a refused configuration ({exc!r}); None is "
            f"the only refusal, because a raise reaches a caller that would "
            f"otherwise have stepped correctly")
    finally:
        driver.close()

    return {"leg": "predicate", "device": False, "cases": rows,
            "poled": poled, "unpoled": unpoled,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — what the corpus says this is worth
# ---------------------------------------------------------------------------

CENSUS = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                      "predicate_coverage_triton_2026-08-31_cells")

#: The board cell this product claims, and the rows the newest cut scored into it.
#: Named so the funnel below is a CROSS-CHECK against an independent derivation
#: rather than a second copy of one.
BOARD = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                     "fusion_matrix_triton_2026-08-31_cells", "fusion_matrix.json")
CELL = ("D->E", "folded PML", "folded dispersive")


def _census_rows(root: str) -> List[dict]:
    def load(name: str) -> List[dict]:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            return []
        return [json.loads(line) for line in open(path, encoding="utf-8")
                if line.strip()]

    record = load("examples.jsonl") + load("tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r
               for r in load("tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return [r for r in record if r.get("measured")]


def corpus_admission_leg() -> Dict[str, Any]:
    """The funnel, and whether it lands on the cell the board scored.

    AN ABSENT CENSUS IS NOT AN EMPTY ONE, and the difference is the whole reason this
    clause exists: a staged tree that did not carry the record would funnel 0 -> 0
    and report "worth zero seam-instances" — a measurement about the staging dressed
    as one about the corpus. Measured on the plain folded pair's first GPU-host run
    2026-08-20: that is exactly what happened.
    """
    rows = _census_rows(CENSUS)
    findings: List[str] = []
    if len(rows) != 186:
        return {"leg": "corpus_admission", "device": False,
                "census": CENSUS, "rows": len(rows),
                "findings": [
                    f"the census at {CENSUS} yielded {len(rows)} measured rows, not "
                    f"the 186 this derivation prices against: the record is absent "
                    f"or truncated on this host and NOTHING below is a measurement "
                    f"about the corpus"],
                "passed": False}

    def covered(row: dict, key: str) -> bool:
        entry = row["predicates"].get(key, {})
        return bool(entry.get("covered_modulo_backend", entry.get("covered")))

    def label(row: dict) -> str:
        return f"{row['leg']}:{row['row']}"

    curl = [r for r in rows if covered(r, "folded_composition_curl@step_D")]
    both = [r for r in curl
            if covered(r, "folded_dispersive_constitutive@update_E")]
    electric = [r for r in both
                if any(str(kind) != "B" for kind in
                       (r["configuration"].get("source_field_types") or []))]
    funnel = {"rows": len(rows), "curl": len(curl), "both": len(both),
              "carrying_an_electric_deposit": len(electric),
              "cell_rows": sorted(label(r) for r in both)}
    if not both:
        findings.append(
            "the two halves admit NO corpus row together: this product is worth "
            "zero seam-instances and should not have been built")

    # THE BOARD'S OWN ANSWER, read rather than re-derived. If the newest cut scored
    # a different row set into this cell, one of the two derivations is wrong and
    # that matters more than the product.
    board_rows: Optional[List[str]] = None
    if os.path.exists(BOARD):
        board = json.loads(open(BOARD, encoding="utf-8").read())
        board_rows = sorted(
            entry["row"] for entry in board["seam_instances"]
            if (entry["seam"], entry["curl_arm"], entry["constitutive_arm"]) == CELL)
        if board_rows != funnel["cell_rows"]:
            findings.append(
                f"this funnel lands on {funnel['cell_rows']} and the board scored "
                f"{board_rows} into cell {CELL}: two derivations over one census "
                f"disagree, and one of them is wrong")
    else:
        findings.append(
            f"the board at {BOARD} is absent, so the cross-check that this funnel "
            f"lands on the cell the matrix scored could not run")
    # EVERY ROW OF THIS CELL CARRIES AN ELECTRIC DEPOSIT, which is why
    # CARRIES_DEPOSIT_REPAIR is the product rather than one clause of it. Asserted,
    # because a cell that stopped being all-electric would make the carry legs below
    # decoration rather than the point.
    if both and len(electric) != len(both):
        findings.append(
            f"only {len(electric)} of {len(both)} rows in this cell carry an "
            f"electric deposit; the module's docstring says all of them do")
    return {"leg": "corpus_admission", "device": False, "census": CENSUS,
            "board_rows": board_rows, "funnel": funnel,
            "seam_instances_gained": len(both),
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side — drivers, inventory, launch counting
# ---------------------------------------------------------------------------

def build_driver(cp, cell, boundaries, mirror_specs, poles, seed: int,
                 electric: bool = False, magnetic: bool = True):
    """One folded dispersive PML driver, seeded identically for every route."""
    from meep_gpu.dispersion import Susceptibility  # noqa: PLC0415
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    dimensions = 3 if float(cell[2]) > 0.0 else 2
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirror_specs),
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, deliberately: the constitutive half multiplies by inverse
    # epsilon at the imaged ghost's OWN index, and a uniform material would make that
    # indistinguishable from reading it at the source's.
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    folded_axes = {axis.lower() for axis, _phase in mirror_specs}
    # A folded axis absorbs on its HIGH face only: the low face is the mirror plane.
    layers = {key: ({"high": 5} if key in folded_axes else 5)
              for key in "xyz"[:dimensions]}
    driver.setup_pml(layers)
    for frequency, gamma, sigma_values in poles:
        driver.add_susceptibility(
            Susceptibility(frequency=frequency, gamma=gamma),
            {component: cp.full(shape, value, dtype=cp.float32)
             for component, value in zip(("Ex", "Ey", "Ez"), sigma_values)})
    # OFF THE MIRROR PLANE ON EVERY FOLDED AXIS. A point source ON the plane is
    # refused with a reason on an odd fold (from_meep.py:2686-2698); a quarter of the
    # owned half-width puts it inside the stored half and clear of the absorber,
    # which a folded axis carries on its HIGH face only.
    center = [0.0, 0.0, 0.0]
    for axis_name, _phase in mirror_specs:
        axis = "XYZ".index(axis_name.upper())
        center[axis] = 0.25 * (cell[axis] / 2.0)
    if magnetic:
        # A MAGNETIC source is admitted with no repair at all: the driver injects it
        # in the B/H seam, not this one. Carrying one is what stops the source clause
        # from being tested only in its refusing direction.
        driver.add_source({"component": "Hz", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    if electric:
        # THE DEPOSIT THIS PRODUCT EXISTS FOR. Injected BETWEEN the two halves
        # (driver.py:3294-3299), so the fused launch consumes a pre-injection D and
        # the shipped repair is what puts the difference back.
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))))
    for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez"):
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(np.ascontiguousarray(
                rng.uniform(-0.05, 0.05, size=shape).astype(np.float32)))
    # THE POLES ARE SEEDED NON-ZERO, and that is not cosmetic. Zero is a fixed point
    # of the subtraction chain: with P == 0 everywhere the dispersive body and the
    # PLAIN folded body are bitwise equal, every pole mutation is a null, and the
    # gate would certify the wrong kernel.
    for state in driver.fields.polarizations:
        for component in state.driven():
            for slot in (state.P, state.P_prev):
                slot[component][...] = cp.asarray(np.ascontiguousarray(
                    rng.uniform(-0.08, 0.08, size=shape).astype(np.float32)))
    return driver


def inventory(driver) -> Dict[str, Any]:
    """Every device volume the step can touch, found by scanning rather than listing.

    THE POLE VOLUMES ARE IN IT. They live on the ``PolarizationState`` objects rather
    than on ``Fields``, so a scan of ``vars(fields)`` alone would leave the arrays
    this kernel loads at eight compiled slots per component OUT of the comparison —
    which is the one place a wrong pole index could hide.
    """
    fields = driver.fields
    shape = tuple(fields.grid.shape)
    found: Dict[str, Any] = {}
    for name, value in vars(fields).items():
        # THE LEADING UNDERSCORE IS THE RULE — see PRIVATE_SCRATCH. A private
        # attribute is not physical state; every PUBLIC volume stays in.
        if name.startswith("_"):
            continue
        if (getattr(value, "shape", None) == shape
                and getattr(value, "dtype", None) is not None):
            found[name] = value
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            found[f"P{index}_{component}"] = state.P[component]
            found[f"Pprev{index}_{component}"] = state.P_prev[component]
            found[f"{POLE_MATERIAL_PREFIX}{index}_{component}"] = state.sigma[component]
        if state._scratch is not None:
            found[f"Pscratch{index}"] = state._scratch
    _assert_material_names_are_real(found)
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise AssertionError(
            f"the state scan lost {missing}; the comparison inventory is not "
            f"complete")
    if not any(name.startswith("P") and "_" in name for name in found):
        raise AssertionError(
            "the state scan found NO pole volume; this product's whole addition is "
            "the pole chain and the arrays it reads are out of the comparison")
    return found


def _material_names(found: Dict[str, Any]) -> List[str]:
    return [name for name in found
            if name in MATERIAL or name.startswith(POLE_MATERIAL_PREFIX)]


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
    """The plan bundle installed for one driver, keyed by driver call site."""

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
    """The three separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import symmetry  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_dispersive_update_e as fd  # noqa: PLC0415

    curl = symmetry.plan_folded_pml_curl(driver.fields, driver.pml, "step_D")
    fill = symmetry.plan_mirror_ghost_fill(driver.fields, "D")
    constitutive = fd.plan_folded_dispersive_constitutive(driver.fields, driver.pml)
    missing = [name for name, plan in
               (("folded curl", curl), ("mirror ghost fill", fill),
                ("folded dispersive constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    # `zero_metal_D` and `fill_folded_far_ghosts_D` stay on the ARRAY PATH here: no
    # Triton product owns either. Both are counted, so a difference in which of them
    # ran is a counter event rather than an invisible correction. THE INJECTION ALSO
    # STAYS ON THE ARRAY PATH, which is what makes this route the right oracle for
    # the carry family: the certified kernels with the driver's own deposit between
    # them.
    return Route({"step_D": curl, "fill_symmetry_bc_D": fill,
                  "update_E": constitutive})


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


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, mirrors, poles, steps: int, product,
            mutant: Any = None, install_fused: bool = True,
            freeze_electric: bool = False, magnetic: bool = True,
            electric: bool = False, bracket: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep, per COMPLETE driver step; stop at the FIRST
    byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    built = [build_driver(cp, cell, boundaries, mirrors, poles, SEED,
                          electric=electric, magnetic=magnetic)
             for _ in range(3)]
    reference, separate, fused = built
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_dispersive_fused_curl_constitutive_D_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "mirrors": [list(entry) for entry in mirrors],
        "poles": [list(entry[:2]) + [list(entry[2])] for entry in poles],
        "magnetic_source": bool(magnetic),
        "electric_source": bool(electric),
        "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        row["pole_counts"] = [
            len([s for s in fused.fields.polarizations if s.drives(component)])
            for component in ("Ex", "Ey", "Ez")]
        plan = product.plan_folded_dispersive_fused_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.folded_dispersive_fused_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
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
            # ARMED: every route's electric seam is inert. All three then agree
            # trivially and only the moved-state census can refuse it.
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
                "deposit_repairs": (leading.repairs if leading is not None
                                    else None),
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
        material = _material_names(final)
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        row["material_changed"] = sorted(key for key in material if key in ever_moved)
        # THE ASYMMETRY IS RECORDED, not compared through. `PRIVATE_SCRATCH`'s note
        # says why the private names are out of the comparison; this row is where a
        # PUBLIC one appearing on one route and not the other becomes visible, and
        # `verdict_of` fails on a non-empty list.
        row["inventory_asymmetry_vs_array"] = sorted(
            set(final).symmetric_difference(snapshot(cp, reference)))
        row["private_scratch_on_the_array_route"] = sorted(
            name for name in vars(reference.fields) if name in PRIVATE_SCRATCH)
        row["private_scratch_on_the_fused_route"] = sorted(
            name for name in vars(fused.fields) if name in PRIVATE_SCRATCH)
        row["launches"] = dict(counter)
        row["fused_kernel_launches"] = kernel.calls
        row["deposit_repairs"] = leading.repairs if leading is not None else None
        row["launch_grids"] = sorted({grid for grid in kernel.grids})
        row["ptx_specializations"] = len(kernel_ptx(kernel.jit))
        row["elapsed_seconds"] = time.time() - started
        return row
    finally:
        undo()
        for target in built:
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()


def verdict_of(row: Dict[str, Any], *, require_identical: bool = True,
               require_launches: Optional[int] = None,
               require_launches_at_least: Optional[int] = None,
               require_moved: bool = True,
               require_repairs: bool = False) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied.

    ``require_launches_at_least`` is for a leg that is EXPECTED to stop early. A leg
    that must diverge breaks at the first divergent step, so an exact launch count
    and the whole-run vacuity census are both statements about a run that did not
    happen; what is still checkable there is that the fused kernel ran at all.
    """
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
            f"fewer than the {require_launches_at_least} this leg needs to have "
            f"measured the fused path at all")
    if require_moved and (row.get("arrays_never_moved") or []):
        failures.append(f"VACUOUS: these arrays never moved: {row['arrays_never_moved']}")
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
        inspect.getsource(product.folded_dispersive_fused_curl_constitutive_D.fn))


def shipped_device_function_sources(product) -> List[Tuple[str, str]]:
    """The two device functions the kernel CALLS, so a mutant module carries them.

    A mutant that carried only the kernel body compiles the call against nothing and
    dies with ``NameError``. Both are renamed with the kernel, because two mutants
    sharing one device-function identity would be two callers of one cached body —
    the hazard the kernel rename exists to close, one level down.

    ``_subtract_poles_at`` comes FIRST: ``_carry_ghost_E_dispersive`` calls it, and a
    mutant module is written in this order.
    """
    return [("_subtract_poles_at",
             textwrap.dedent(inspect.getsource(product._subtract_poles_at.fn))),
            ("_carry_ghost_E_dispersive",
             textwrap.dedent(inspect.getsource(
                 product._carry_ghost_E_dispersive.fn)))]


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was
    MEASURED, never what was expected."""

    # ------------------------------------------------------------------
    # THE POLE MUTATIONS — this product's own arithmetic, and the only lines in
    # the kernel that are not a byte-copy of a released body.
    # ------------------------------------------------------------------

    def mp1_ghost_reads_the_wrong_cells_poles(source: str) -> Tuple[str, int]:
        """THE DEFECT THIS PRODUCT COULD HAVE. The imaged ghost's pole sum taken at a
        cell that is NOT its own.

        The array path's fills write D and never P (stepping.py:1497-1498,
        :1529-1532), so ``update_E`` at an imaged cell reads THAT cell's poles. A
        kernel whose pole index did not follow the fill would be a smooth, plausible,
        wrong field on every folded plane — and no comparison restricted to owned
        cells would ever see it.

        The rewrite binds ``where`` to stored cell 0 instead of ``dst``. Cell 0 is in
        bounds by construction, where a forward shift could address outside the
        allocation and be reported as an inert defect. It is inside the DEVICE
        FUNCTION, where every carry goes through it, so ONE edit arms all seven
        destinations of all three components."""
        return _rewrite_block(
            source,
            ["s = _subtract_poles_at(ghost, p0, p1, p2, p3, p4, p5, p6, p7,",
             "dst, mask, NP)"],
            ["s = _subtract_poles_at(ghost, p0, p1, p2, p3, p4, p5, p6, p7,",
             "dst * 0, mask, NP)"])

    def mp2_first_two_arms_swapped(source: str) -> Tuple[str, int]:
        """The chain applied last-to-first.

        Algebraically identical and a DIFFERENT float32 number at two or more poles,
        which is why the certified body spells eight separate arms rather than a sum.

        SWAPPING THE FIRST TWO ARMS *IS* THE REVERSAL on a two-pole component, and it
        is nothing else on a one-pole one — there the same rewrite would read a DEAD
        slot (the plan binds unused slots to the component's own D volume), which is
        a different defect. :data:`MUTATION_REQUIRES_TWO_POLES_EVERYWHERE` is what
        keeps this scored only where every component carries two."""
        hits = 0
        first = "base = base - tl.load(p0 + where, mask=mask, other=0.0)"
        second = "base = base - tl.load(p1 + where, mask=mask, other=0.0)"
        hits += source.count(first) + source.count(second)
        source = source.replace(first, "base = base - tl.load(SWAP + where, "
                                       "mask=mask, other=0.0)")
        source = source.replace(second, first)
        source = source.replace("tl.load(SWAP + where", "tl.load(p1 + where")
        return source, hits

    def mp3_poles_pre_summed(source: str) -> Tuple[str, int]:
        """``D - (P0 + P1)`` instead of ``(D - P0) - P1``.

        The recon behind ``dispersive_update_e`` caught this 20/20 at two poles, and
        it is the single most plausible "optimisation" a reader could make here.

        NO COMPOUND STATEMENT IS INTRODUCED, and that is deliberate: the two arm
        BODIES are rewritten in place, so the guards, the indentation and the number
        of compiled arms are untouched and the mutant still imports. On a component
        with two poles the pair computes exactly ``base - (P0 + P1)``; on one with
        fewer it would compute something else, which is why this too is scored only
        where every component carries two."""
        hits = 0
        first = "base = base - tl.load(p0 + where, mask=mask, other=0.0)"
        second = "base = base - tl.load(p1 + where, mask=mask, other=0.0)"
        hits += source.count(first) + source.count(second)
        source = source.replace(second, "base = base - (tl.load(p0 + where, "
                                        "mask=mask, other=0.0) + tl.load(p1 + where, "
                                        "mask=mask, other=0.0))")
        source = source.replace(first, "base = base")
        return source, hits

    def mp4_pole_chain_dropped(source: str) -> Tuple[str, int]:
        """No poles subtracted at all — the PLAIN folded pair's constitutive.

        The mutation that says what this whole product bought. It is also the check
        that the fixture's poles are non-zero: with ``P == 0`` everywhere this would
        be a null, and a null here would mean every other pole mutation was
        measuring nothing."""
        hits = 0
        for k in range(8):
            needle = f"base = base - tl.load(p{k} + where, mask=mask, other=0.0)"
            hits += source.count(needle)
            source = source.replace(needle, "base = base")
        return source, hits

    def mp5_component_two_takes_component_zeros_poles(source: str) -> Tuple[str, int]:
        """Component 2's owned cell subtracts component 0's pole group.

        Invisible whenever the two components carry the same poles, which is why
        :data:`MUTATION_REQUIRES_DISTINCT_POLE_GROUPS` checks the case's sigma
        columns rather than assuming they differ. The count constexpr ``NP2`` is
        LEFT ALONE, so what changes is the arrays and not how many are read."""
        needle = "s2 = _subtract_poles_at(v2, c0, c1, c2, c3, c4, c5, c6, c7,"
        return (source.replace(
            needle, "s2 = _subtract_poles_at(v2, a0, a1, a2, a3, a4, a5, a6, a7,"),
            source.count(needle))

    def mp6_owned_cell_reads_the_wrong_cells_poles(source: str) -> Tuple[str, int]:
        """The OWNED cell's pole sum taken at stored cell 0 rather than its own.

        The mirror of ``mp1``: it proves the owned half of the substitution is
        indexed at ``idx``, and it is scored on component 0 alone so the other two
        components still compute correctly and the mutant is not simply "no poles"."""
        return _rewrite_block(
            source,
            ["s0 = _subtract_poles_at(v0, a0, a1, a2, a3, a4, a5, a6, a7,",
             "idx, own0, NP0)"],
            ["s0 = _subtract_poles_at(v0, a0, a1, a2, a3, a4, a5, a6, a7,",
             "idx * 0, own0, NP0)"])

    # ------------------------------------------------------------------
    # THE CARRIED BODY'S MUTATIONS. Every one of these is armed on the copy this
    # product makes of `folded_fused_pair`'s released kernel: a copy that drifted
    # would be a NEW defect in a body nothing else re-certifies.
    # ------------------------------------------------------------------

    def m1_parity_dropped(source: str) -> Tuple[str, int]:
        """The mirror parity thrown away: the ghost becomes a copy, not an image.

        Invisible on an EVEN plane (phase +1) by construction, so it is scored on
        the odd Y fold."""
        hits = 0
        for needle, replacement in (("gv_0y = PHY * v0", "gv_0y = v0"),
                                    ("gv_2y = PHY * v2", "gv_2y = v2")):
            hits += source.count(needle)
            source = source.replace(needle, replacement)
        return source, hits

    def m2_near_fill_dropped(source: str) -> Tuple[str, int]:
        """The NEAR fill not carried: update_E consumes a pre-fill D."""
        hits = 0
        for own, coordinate in (("own0", "j"), ("own2", "j"), ("own0", "k"),
                                ("own1", "k"), ("own1", "i"), ("own2", "i")):
            needle = f"{own} = {own} & ({coordinate} != 0)"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} = {own}")
        for lane, coordinate in (("near_i", "i"), ("near_j", "j"), ("near_k", "k")):
            needle = f"{lane} = live & ({coordinate} == 2)"
            hits += source.count(needle)
            source = source.replace(needle, f"{lane} = live & (nx < 0)")
        return source, hits

    def m3_corner_carry_dropped(source: str) -> Tuple[str, int]:
        """The doubly-unowned NEAR corner left unwritten. Two folds required."""
        hits = 0
        for own, first, second in (("own0", "near_j", "near_k"),
                                   ("own1", "near_i", "near_k"),
                                   ("own2", "near_i", "near_j")):
            needle = f"{own} & {first} & {second}, NP"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} & (nx < 0), NP")
        return source, hits

    def m4_corner_weight_is_one_axis_twice(source: str) -> Tuple[str, int]:
        """The corner imaged with the WRONG parity product. Two DIFFERENT parities
        required, or the two spellings agree."""
        needle = "gv_2xy = PHY * (PHX * v2)"
        return (source.replace(needle, "gv_2xy = PHX * (PHX * v2)"),
                source.count(needle))

    def m5_source_index_off_by_one(source: str) -> Tuple[str, int]:
        """The ghost imaged from stored cell 1 rather than 2 (MEEP's io = -2)."""
        return _rewrite_block(
            source,
            ["near_j = live & (j == 2)", "near_k = live & (k == 2)"],
            ["near_j = live & (j == 1)", "near_k = live & (k == 1)"])

    def m6_constitutive_association(source: str) -> Tuple[str, int]:
        """Right-associated accumulation: same algebra, different float32 rounding."""
        return _rewrite_block(
            source,
            ["a0v = a0v + kp_0 * src0", "a0v = a0v - km_0 * prev0"],
            ["a0v = a0v + (kp_0 * src0 - km_0 * prev0)"])

    def m7_history_read_after_write(source: str) -> Tuple[str, int]:
        """f_w read AFTER it is written: wrong only where kms != 0, i.e. in the PML."""
        return _rewrite_block(
            source,
            ["prev0 = tl.load(w0 + idx, mask=own0, other=0.0)",
             "s0 = _subtract_poles_at(v0, a0, a1, a2, a3, a4, a5, a6, a7,",
             "idx, own0, NP0)",
             "src0 = s0 * tl.load(ie0 + idx, mask=own0, other=0.0)",
             "tl.store(w0 + idx, src0, mask=own0)"],
            ["s0 = _subtract_poles_at(v0, a0, a1, a2, a3, a4, a5, a6, a7,",
             "idx, own0, NP0)",
             "src0 = s0 * tl.load(ie0 + idx, mask=own0, other=0.0)",
             "tl.store(w0 + idx, src0, mask=own0)",
             "prev0 = tl.load(w0 + idx, mask=own0, other=0.0)"])

    def m8_zero_metal_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's step_boundaries(D_stuff), undone."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v1 = tl.where(at_x, 0.0, v1)",
             "v2 = tl.where(at_x, 0.0, v2)"],
            ["v1 = v1", "v2 = v2"])

    def m9_ghost_wall_clear_dropped(source: str) -> Tuple[str, int]:
        """The imaged ghost's OWN wall clear dropped — the parity/clear ORDER."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "gv_2y = tl.where(at_x, 0.0, gv_2y)"],
            ["gv_2y = gv_2y"])

    def m10_inverse_epsilon_at_the_source(source: str) -> Tuple[str, int]:
        """The ghost's material read at the SOURCE cell rather than its own.

        Invisible in a uniform medium, which is why :func:`build_driver` installs a
        varying epsilon. `ie2 - dn_y` makes the load land at `(idx + dn_y) - dn_y`
        = `idx`, the SOURCE lane's own cell — in bounds by construction, where a
        forward shift could address outside the allocation and be reported as an
        inert defect.

        SCORED ON ONE CALL SITE, not on all seven of component 2's: applying the
        offset to a FAR destination would read past the allocation rather than at a
        cell this lane already touches."""
        return _rewrite_block(
            source,
            ["_carry_ghost_E_dispersive(f2, w2, e2, ie2,",
             "c0, c1, c2, c3, c4, c5, c6, c7,",
             "idx + dn_y, gv_2y,",
             "kp_2, km_2, own2 & near_j, NP2)"],
            ["_carry_ghost_E_dispersive(f2, w2, e2, ie2 - dn_y,",
             "c0, c1, c2, c3, c4, c5, c6, c7,",
             "idx + dn_y, gv_2y,",
             "kp_2, km_2, own2 & near_j, NP2)"])

    def m13_far_takes_the_source_coefficient(source: str) -> Tuple[str, int]:
        """The far ghost accumulated with the SOURCE lane's kps/kms pair."""
        needle = "kp_f0, km_f0, own0 & far_i, NP0"
        return (source.replace(needle, "kp_0, km_0, own0 & far_i, NP0"),
                source.count(needle))

    def m14_far_carry_dropped(source: str) -> Tuple[str, int]:
        """The far fill not carried at all — the pre-2026-08-21 kernel's behaviour."""
        hits = 0
        for own, coordinate, extent in (("own0", "i", "nx"), ("own1", "j", "ny"),
                                        ("own2", "k", "nz")):
            needle = f"{own} = {own} & ({coordinate} != {extent} - 1)"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} = {own}")
        for lane, coordinate, reflect in (("far_i", "i", "rx"), ("far_j", "j", "ry"),
                                          ("far_k", "k", "rz")):
            needle = f"{lane} = live & ({coordinate} == {reflect})"
            hits += source.count(needle)
            source = source.replace(needle, f"{lane} = live & (nx < 0)")
        return source, hits

    def m15_far_parity_sign(source: str) -> Tuple[str, int]:
        """The far ghost imaged with ``+phase`` — the NEAR fill's parity."""
        needle = "idx + df_x, -PHX * v0"
        return source.replace(needle, "idx + df_x, PHX * v0"), source.count(needle)

    def m16_top_plane_mask_dropped(source: str) -> Tuple[str, int]:
        """``pml_curl_step_folded``'s top-plane mask, un-carried."""
        hits = 0
        for target, coordinate in (("curl0", "last_x"), ("curl1", "last_y"),
                                   ("curl2", "last_z")):
            needle = f"{target} = tl.where({coordinate}, 0.0, {target})"
            hits += source.count(needle)
            source = source.replace(needle, f"{target} = {target}")
        return source, hits

    def m17_far_reflect_row_is_n_minus_two(source: str) -> Tuple[str, int]:
        """The reflect row baked as ``n - 2`` instead of the runtime one.

        ``stepping._far_reflect_rows`` is ``n_full - stored + 2``, which is
        ``stored - 2`` at an EVEN full count and ``stored - 3`` at an ODD one, so
        this must be scored on an ODD full count."""
        hits = 0
        for lane, coordinate, reflect, extent in (
                ("far_i", "i", "rx", "nx"), ("far_j", "j", "ry", "ny"),
                ("far_k", "k", "rz", "nz")):
            needle = f"{lane} = live & ({coordinate} == {reflect})"
            hits += source.count(needle)
            source = source.replace(
                needle, f"{lane} = live & ({coordinate} == {extent} - 2)")
        for axis, reflect, extent in (("df_x", "rx", "nx"), ("df_y", "ry", "ny"),
                                      ("df_z", "rz", "nz")):
            stride = {"df_x": " * nyz", "df_y": " * nz", "df_z": ""}[axis]
            needle = f"{axis} = ({extent} - 1 - {reflect}){stride}"
            hits += source.count(needle)
            source = source.replace(needle, f"{axis} = 1{stride}")
        return source, hits

    def m11_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
        ("mp1_ghost_reads_the_wrong_cells_poles", "the ghost's OWN poles", "caught",
         mp1_ghost_reads_the_wrong_cells_poles),
        ("mp2_first_two_arms_swapped", "the bit-load-bearing order", "caught",
         mp2_first_two_arms_swapped),
        ("mp3_poles_pre_summed", "D - (P0+P1) is a different float32", "caught",
         mp3_poles_pre_summed),
        ("mp4_pole_chain_dropped", "what this product bought", "caught",
         mp4_pole_chain_dropped),
        ("mp5_component_two_takes_component_zeros_poles", "the per-component group",
         "caught", mp5_component_two_takes_component_zeros_poles),
        ("mp6_owned_cell_reads_the_wrong_cells_poles", "the owned cell's own index",
         "caught", mp6_owned_cell_reads_the_wrong_cells_poles),
        ("m1_parity_dropped", "the mirror parity", "caught", m1_parity_dropped),
        ("m2_near_fill_dropped", "the fill not carried", "caught",
         m2_near_fill_dropped),
        ("m3_corner_carry_dropped", "the doubly-unowned corner", "caught",
         m3_corner_carry_dropped),
        ("m4_corner_weight_is_one_axis_twice", "the parity PRODUCT", "caught",
         m4_corner_weight_is_one_axis_twice),
        ("m5_source_index_off_by_one", "MEEP's io = -2", "caught",
         m5_source_index_off_by_one),
        ("m6_constitutive_association", "float32 association", "caught",
         m6_constitutive_association),
        ("m7_history_read_after_write", "the f_w ordering", "caught",
         m7_history_read_after_write),
        ("m8_zero_metal_dropped", "the wall clear's slot", "caught",
         m8_zero_metal_dropped),
        ("m9_ghost_wall_clear_dropped", "parity-then-clear", "caught",
         m9_ghost_wall_clear_dropped),
        ("m10_inverse_epsilon_at_the_source", "the ghost's own material", "caught",
         m10_inverse_epsilon_at_the_source),
        ("m13_far_takes_the_source_coefficient", "the moved coefficient index",
         "caught", m13_far_takes_the_source_coefficient),
        ("m14_far_carry_dropped", "the far fill not carried", "caught",
         m14_far_carry_dropped),
        ("m15_far_parity_sign", "-phase on a shift-1 component", "caught",
         m15_far_parity_sign),
        ("m16_top_plane_mask_dropped", "the top-plane ownership mask", "caught",
         m16_top_plane_mask_dropped),
        ("m17_far_reflect_row_is_n_minus_two", "the ODD full count", "caught",
         m17_far_reflect_row_is_n_minus_two),
        ("m11_commuted_multiply", "commuted multiply", "null", m11_commuted_multiply),
    )


#: Mutation id -> the CASE NAME whose grid carries the branch it rewrites.
#:
#: THE DEAD-BRANCH TRAP, priced per entry. Every carry block in this kernel sits under
#: a fold constexpr on a named axis, every corner block under two of them, every
#: ``FAR_a`` block under a PERIODIC outer declaration, and every pole arm under
#: ``NP?``. A mutation scored on a grid that does not carry what it rewrites comes
#: back "uncaught" while measuring nothing; :func:`mutation_case_for` REFUSES such a
#: pairing rather than running it.
MUTATION_CASE: Dict[str, str] = {
    # The pole mutations. The ORDER ones need two poles on EVERY component; the rest
    # are scored on the asymmetric (2, 1, 1) grid, which is the harder fixture for a
    # per-component defect.
    "mp1_ghost_reads_the_wrong_cells_poles": "odd_y_fold_metallic",
    "mp2_first_two_arms_swapped": "two_poles_everywhere_y_fold",
    "mp3_poles_pre_summed": "two_poles_everywhere_y_fold",
    "mp4_pole_chain_dropped": "odd_y_fold_metallic",
    "mp5_component_two_takes_component_zeros_poles": "odd_y_fold_metallic",
    "mp6_owned_cell_reads_the_wrong_cells_poles": "odd_y_fold_metallic",
    # The carried body's mutations, on the grids the plain folded pair's released
    # gate scored the same rewrites on.
    "m1_parity_dropped": "odd_y_fold_metallic",
    "m2_near_fill_dropped": "two_folds_metallic",
    "m3_corner_carry_dropped": "two_folds_metallic",
    "m4_corner_weight_is_one_axis_twice": "two_folds_metallic",
    "m5_source_index_off_by_one": "odd_y_fold_metallic",
    "m9_ghost_wall_clear_dropped": "odd_y_fold_metallic",
    "m10_inverse_epsilon_at_the_source": "odd_y_fold_metallic",
    "m13_far_takes_the_source_coefficient": "x_fold_periodic_odd",
    "m14_far_carry_dropped": "two_folds_periodic",
    "m15_far_parity_sign": "x_fold_periodic_odd",
    "m16_top_plane_mask_dropped": "two_folds_periodic",
    "m17_far_reflect_row_is_n_minus_two": "x_fold_periodic_odd",
}

#: The default leg for a mutation that does not name one.
DEFAULT_MUTATION_CASE = "odd_y_fold_metallic"

#: Per mutation: the axes its rewritten lines are guarded on.
MUTATION_REQUIRES_FOLD: Dict[str, Tuple[str, ...]] = {
    "m1_parity_dropped": ("Y",),
    "m2_near_fill_dropped": ("X", "Y"),
    "m3_corner_carry_dropped": ("X", "Y"),
    "m4_corner_weight_is_one_axis_twice": ("X", "Y"),
    "m5_source_index_off_by_one": ("Y",),
    "m9_ghost_wall_clear_dropped": ("Y",),
    "m10_inverse_epsilon_at_the_source": ("Y",),
    "mp1_ghost_reads_the_wrong_cells_poles": ("Y",),
    "m13_far_takes_the_source_coefficient": ("X",),
    "m14_far_carry_dropped": ("X", "Y"),
    "m15_far_parity_sign": ("X",),
    "m16_top_plane_mask_dropped": ("X", "Y"),
    "m17_far_reflect_row_is_n_minus_two": ("X",),
}

#: Per mutation: the axes whose fold must be PERIODIC, not merely folded. Every
#: ``FAR_a`` block sits under a constexpr that is true only on
#: ``CODE_MIRROR_PERIODIC``.
MUTATION_REQUIRES_PERIODIC_FOLD: Dict[str, Tuple[str, ...]] = {
    "m13_far_takes_the_source_coefficient": ("X",),
    "m14_far_carry_dropped": ("X", "Y"),
    "m15_far_parity_sign": ("X",),
    "m16_top_plane_mask_dropped": ("X", "Y"),
    "m17_far_reflect_row_is_n_minus_two": ("X",),
}

#: Per mutation: axes that must carry a WALL.
MUTATION_REQUIRES_WALL: Dict[str, Tuple[str, ...]] = {
    "m9_ghost_wall_clear_dropped": ("X",),
}

#: Per mutation: the full count of a named axis must be ODD.
MUTATION_REQUIRES_ODD_FULL_COUNT: Dict[str, Tuple[str, ...]] = {
    "m17_far_reflect_row_is_n_minus_two": ("X",),
}

#: Per mutation: axes whose mirror plane must be ODD (phase -1). An even plane
#: multiplies by +1, so a dropped or duplicated parity is invisible there by
#: construction.
MUTATION_REQUIRES_ODD_PARITY: Dict[str, Tuple[str, ...]] = {
    "m1_parity_dropped": ("Y",),
    "m9_ghost_wall_clear_dropped": ("Y",),
    "m15_far_parity_sign": ("X",),
}

#: Per mutation: the minimum number of poles on ONE component. A rewrite of the pole
#: chain scored on a grid with no pole at all is a rewrite of a branch the compiler
#: removed.
MUTATION_REQUIRES_POLES: Dict[str, int] = {
    "mp1_ghost_reads_the_wrong_cells_poles": 1,
    "mp4_pole_chain_dropped": 1,
    "mp5_component_two_takes_component_zeros_poles": 1,
    "mp6_owned_cell_reads_the_wrong_cells_poles": 1,
}

#: Per mutation: EVERY component must carry at least two poles.
#:
#: THE SHARPER HALF OF THE POLE TRAP. ``mp2`` swaps the first two arms and ``mp3``
#: pre-sums them; both are the intended defect exactly where both arms are live. On a
#: component carrying ONE pole the same rewrite reads a DEAD slot — the plan binds
#: unused slots to that component's own D volume — which is a different defect, so a
#: (2, 1, 1) grid would have the leg report the wrong thing as caught.
MUTATION_REQUIRES_TWO_POLES_EVERYWHERE: Tuple[str, ...] = (
    "mp2_first_two_arms_swapped", "mp3_poles_pre_summed")

#: Per mutation: the two components whose pole GROUPS must differ, or the swap is a
#: bitwise no-op. Checked against the case's sigma declaration.
MUTATION_REQUIRES_DISTINCT_POLE_GROUPS: Dict[str, Tuple[int, int]] = {
    "mp5_component_two_takes_component_zeros_poles": (0, 2),
}


def _pole_counts(poles) -> Tuple[int, int, int]:
    """Poles per E component, from the case's sigma declaration.

    ``PolarizationState`` drives a component only where its sigma is non-trivial
    (dispersion.py:641-643), so a zero entry means that term does not count there.
    """
    counts = [0, 0, 0]
    for _frequency, _gamma, sigma in poles:
        for index, value in enumerate(sigma):
            if float(value) != 0.0:
                counts[index] += 1
    return (counts[0], counts[1], counts[2])


def _assert_the_far_coefficient_actually_moves(case_name: str, cell, boundaries,
                                               mirrors, axis_letter: str) -> None:
    """The far destination's kps/kms pair must not equal its source's.

    A PROPERTY OF THE ABSORBER'S GRADING, not of the case's fold set, so it is
    MEASURED on the grid the case builds rather than declared. At an even full count
    the reflect row is ``n - 2``, the destination is ``n - 1``, and the coefficient
    array holds the same value at both because the not-owned ghost slot copies its
    neighbour — the plain folded pair's gate measured exactly that on 2026-08-21 and
    reported a real defect as uncaught.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415
    from meep_gpu import stepping  # noqa: PLC0415

    axis = "XYZ".index(axis_letter.upper())
    dimensions = 3 if float(cell[2]) > 0.0 else 2
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        symmetry=tuple(Mirror(name, phase) for name, phase in mirrors),
        prefer_gpu=False)
    folded = {name.lower() for name, _phase in mirrors}
    layers = {key: ({"high": 5} if key in folded else 5)
              for key in ("xyz"[:dimensions])}
    driver.setup_pml(layers)
    grid = driver.fields.grid
    row = stepping._far_reflect_rows(grid)[axis]
    if row is None:
        raise AssertionError(
            f"mutation on case {case_name!r}: axis {axis_letter} has no far reflect "
            f"row, so there is no far destination whose coefficient could move")
    top = int(grid.shape[axis]) - 1
    for stem in ("kps", "kms"):
        values = np.asarray(
            getattr(driver.pml, f"{stem}_{axis_letter.lower()}_h")).reshape(-1)
        if values[int(row)] == values[top]:
            raise AssertionError(
                f"mutation on case {case_name!r}: {stem}_{axis_letter.lower()}_h is "
                f"BITWISE EQUAL at the far source row {int(row)} and the destination "
                f"row {top} ({values[int(row)]!r}); swapping one for the other is a "
                f"no-op and the leg would report a real defect as uncaught")


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    """The (index, case) a mutation is scored on, and it must really carry what it
    needs — the fold set, the outer declaration, the parity, the wall AND the poles."""
    index = case_index(MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE))
    case = CASES[index]
    name_, cell, boundaries, mirrors, poles, _steps = case
    folded = {axis.upper() for axis, _phase in mirrors}
    required = set(MUTATION_REQUIRES_FOLD.get(name, ()))
    if not required <= folded:
        raise AssertionError(
            f"mutation {name} rewrites lines guarded on folds {sorted(required)} but "
            f"is scored on case {name_!r}, which folds {sorted(folded)}: the "
            f"rewritten lines would be a DEAD BRANCH and the leg would report "
            f"'uncaught' while measuring nothing")

    def declaration(axis_letter: str) -> str:
        if isinstance(boundaries, str):
            return boundaries
        return str(boundaries.get(axis_letter.lower(), "periodic"))

    for axis_letter in MUTATION_REQUIRES_PERIODIC_FOLD.get(name, ()):
        if axis_letter.upper() not in folded:
            raise AssertionError(
                f"mutation {name} needs a folded PERIODIC {axis_letter} axis and "
                f"case {name_!r} does not fold {axis_letter} at all")
        if declaration(axis_letter) != "periodic":
            raise AssertionError(
                f"mutation {name} rewrites a FAR_{axis_letter.upper()} block, which "
                f"is compile-time absent unless {axis_letter} is folded over a "
                f"PERIODIC outer declaration; case {name_!r} declares "
                f"{declaration(axis_letter)!r} there")
    for axis_letter in MUTATION_REQUIRES_WALL.get(name, ()):
        if declaration(axis_letter) != "metallic" or axis_letter.upper() in folded:
            raise AssertionError(
                f"mutation {name} rewrites a wall clear on {axis_letter}, and case "
                f"{name_!r} declares {declaration(axis_letter)!r} there "
                f"(folded={axis_letter.upper() in folded}); stepping._zero_metal "
                f"skips a folded axis, so no clear would run")
    phases = {axis.upper(): int(phase) for axis, phase in mirrors}
    for axis_letter in MUTATION_REQUIRES_ODD_PARITY.get(name, ()):
        if phases.get(axis_letter.upper()) != -1:
            raise AssertionError(
                f"mutation {name} is only visible on an ODD {axis_letter} plane; "
                f"case {name_!r} declares phase {phases.get(axis_letter.upper())!r}")
    counts = _pole_counts(poles)
    needed = MUTATION_REQUIRES_POLES.get(name)
    if needed is not None and max(counts) < needed:
        raise AssertionError(
            f"mutation {name} needs at least {needed} poles on one component and "
            f"case {name_!r} carries {counts}: the rewrite would be a null and the "
            f"leg would report a real defect as uncaught")
    if name in MUTATION_REQUIRES_TWO_POLES_EVERYWHERE and min(counts) < 2:
        raise AssertionError(
            f"mutation {name} rewrites the first TWO arms and is the intended defect "
            f"only where both are live; case {name_!r} carries {counts}, so on the "
            f"components with fewer the rewrite would read a DEAD slot instead — a "
            f"different defect, scored under this one's name")
    pair = MUTATION_REQUIRES_DISTINCT_POLE_GROUPS.get(name)
    if pair is not None:
        left = tuple(sigma[pair[0]] for _f, _g, sigma in poles)
        right = tuple(sigma[pair[1]] for _f, _g, sigma in poles)
        if left == right:
            raise AssertionError(
                f"mutation {name} swaps component {pair[0]}'s pole group for "
                f"component {pair[1]}'s, and case {name_!r} declares the same sigma "
                f"set on both ({left}); the swap would be a bitwise no-op")
    if name == "m13_far_takes_the_source_coefficient":
        _assert_the_far_coefficient_actually_moves(name_, cell, boundaries, mirrors,
                                                   "X")
    for axis_letter in MUTATION_REQUIRES_ODD_FULL_COUNT.get(name, ()):
        axis = "XYZ".index(axis_letter.upper())
        full = int(round(12.0 * float(cell[axis])))
        if full % 2 == 0:
            raise AssertionError(
                f"mutation {name} bakes a reflect row that is CORRECT at an even "
                f"full count, and case {name_!r} has full count {full} on "
                f"{axis_letter}")
    return index, case


def run_mutations(cp, product, pristine_ptx: Sequence[str]) -> List[Dict[str, Any]]:
    source = shipped_source(product)
    helpers = "".join(text for _name, text in
                      shipped_device_function_sources(product))
    rows: List[Dict[str, Any]] = []
    for index, (name, why, expectation, rewrite) in enumerate(mutation_table()):
        # A pole mutation rewrites the DEVICE FUNCTION, not the kernel body, so the
        # rewrite is applied to the concatenation and split back out by
        # `compile_mutant`'s own rename. Both texts are offered; the rewrite finds
        # its own lines in whichever carries them.
        mutated_kernel, hits = rewrite(source)
        mutated_helpers, helper_hits = rewrite(helpers)
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation,
                               "rewrite_hits": hits + helper_hits,
                               "rewrote_the_device_function": bool(helper_hits),
                               "device": True}
        if (hits + helper_hits) == 0:
            row["error"] = "the rewrite matched nothing; the mutation was not armed"
            rows.append(row)
            log(f"  mutation {name}: NOT ARMED")
            continue
        kernel_name = f"mutant_{index}_folded_dispersive_D"
        mutant = compile_mutant_with_helpers(
            mutated_kernel, mutated_helpers, kernel_name, product)
        case_index, case = mutation_case_for(name)
        row["case"] = case[0]
        row["case_index"] = case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3], case[4],
                      MUTATION_STEPS, product, mutant=mutant)
        row["caught"] = leg.get("first_divergence") is not None
        row["first_divergence"] = leg.get("first_divergence")
        row["launches"] = leg.get("fused_kernel_launches")
        row["pole_counts"] = leg.get("pole_counts")
        row["ptx_moved"] = sorted(kernel_ptx(mutant)) != sorted(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} on {row['case']} "
            f"launches={row['launches']} expectation={expectation}")
    return rows


def compile_mutant_with_helpers(kernel_source: str, helper_source: str,
                                kernel_name: str, product: Any) -> Any:
    """Compile a renamed mutant from a (possibly mutated) kernel AND helper text."""
    from meep_gpu.triton_kernels import symmetry as _symmetry  # noqa: PLC0415

    constants = "".join(
        f"{name} = tl.constexpr({getattr(_symmetry, 'CODE_' + name)})\n"
        for name in ("PERIODIC", "METALLIC", "MIRROR_METALLIC", "MIRROR_PERIODIC"))
    header = "import triton\nimport triton.language as tl\n" + constants + "\n"
    text = helper_source + "\n\n" + kernel_source
    # THE HELPERS ARE RENAMED WITH THE KERNEL. Two mutants sharing a helper identity
    # would be two callers of one cached device body, which is the hazard the kernel
    # rename exists to close, one level down.
    for helper in ("_carry_ghost_E_dispersive", "_subtract_poles_at"):
        text = text.replace(helper, f"{helper}_{kernel_name}")
    text = text.replace("folded_dispersive_fused_curl_constitutive_D", kernel_name)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_folded_dispersive_pair.py", delete=False,
        encoding="utf-8")
    handle.write(header + text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_dispersive_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product) -> List[Dict[str, Any]]:
    """Configurations the product must refuse — and two it must ADMIT."""

    class _Electric:
        field_type = "D"

    def _deposit(fields, component):
        """A REAL source that publishes the index the injection writes."""
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        source = VolumeSource(grid=fields.grid, component=component,
                              center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    _WALLS = {"x": "metallic", "y": "metallic", "z": "periodic"}
    _CELL = CASES[0][1]
    rows: List[Dict[str, Any]] = []
    for name, cell, boundaries, mirrors, poles, needle, sources in (
        # THE CARRY, on the device's own objects. The refusing row names the
        # fail-closed clause; the ADMITTING row is what stops "refuses electric" from
        # silently becoming "refuses everything electric".
        ("electric_source_without_a_deposit_index", _CELL, _WALLS, (("Y", 1),),
         DEFAULT_POLES, "does not publish the index it writes", (_Electric(),)),
        ("electric_deposit_is_carried", _CELL, _WALLS, (("Y", 1),), DEFAULT_POLES,
         None, lambda fields: (_deposit(fields, "Ez"),)),
        ("undeclared_sources", _CELL, _WALLS, (("Y", 1),), DEFAULT_POLES,
         "was not declared", None),
        # THE DISJOINTNESS CLAUSE, on the device. Without a pole the source is D and
        # the configuration is folded_fused_pair's.
        ("no_susceptibility", _CELL, _WALLS, (("Y", 1),), (),
         "no susceptibility is registered", ()),
        ("unfolded", (3.0, 3.0, 0.0), _WALLS, (), DEFAULT_POLES,
         "no mirror plane", ()),
        # ADMITTED: a folded PERIODIC grid, which the far carry serves.
        ("folded_periodic", _CELL, "periodic", (("Y", 1),), DEFAULT_POLES, None, ()),
    ):
        driver = build_driver(cp, cell, boundaries, mirrors, poles, SEED,
                              magnetic=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.folded_dispersive_fused_pair_coverage(
                driver.fields, driver.pml, bound)
            plan = product.plan_folded_dispersive_fused_pair(
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
    payload: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "argv": list(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
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
        HERE, "results", "triton_folded_dispersive_fused_pair"),
        help="a DIRECTORY; gate.json is written inside it")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep' — the policy every record in "
             "triton_kernels/fingerprints.json is cut under.")
    args = parser.parse_args(argv)

    out = args.out
    if out.endswith(".json"):
        artifact = out
    else:
        os.makedirs(out, exist_ok=True)
        artifact = os.path.join(out, "gate.json")

    payload: Dict[str, Any] = {
        "gate": "triton_folded_dispersive_fused_pair",
        "product": "meep_gpu.triton_kernels.folded_dispersive_fused_pair",
        "kernel": "folded_dispersive_fused_curl_constitutive_D",
        "replaces": list(SEAM_PASSES),
        "source_sha256": source_hashes(),
        "environment": environment(),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "kernels.DEFAULT_BLOCK"},
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
        "refusals": [],
    }

    log("=== no-device legs ===")
    for leg in (transcription_leg, predicate_leg, corpus_admission_leg):
        started = time.time()
        row = leg()
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
        # ``release`` IS THE KEY ``gate_provenance.read_verdict`` CONSULTS FIRST, and
        # it is set explicitly here because ``passed`` alone would stamp
        # ``released: True`` on an artifact that measured no bytes on any device.
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

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep and
    # quietly did not is a process whose bytes mean nothing.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import folded_dispersive_fused_pair as product  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    # THE MACHINE AND THE DEVICE, so this artifact can be SEEDED into a weld. The
    # weld contract binds a host string to the record's validated Triton and
    # capability lists; `environment()` below records the device but no hostname, and
    # a seeding tool that supplied the missing half would be typing a fact instead of
    # reading one. Resolved by name off the parity directory, as gate_provenance is.
    import triton_device_identity  # noqa: PLC0415

    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "quiet_cases": {case[0]: case[5] for case in CASES},
        "carry_cases": list(CARRY_CASES),
        "steps_per_mutation": MUTATION_STEPS,
    }
    save(payload, artifact)

    log("\n=== device legs: the QUIET family (a magnetic source, no seam deposit) ===")
    for name, cell, boundaries, mirrors, poles, steps in CASES:
        row = run_leg(cp, f"quiet:{name}", cell, boundaries, mirrors, poles, steps,
                      product)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== device legs: the CARRY family (a real electric deposit in the seam) ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, mirrors, poles, _steps = CASES[case_index(case_name)]
        row = run_leg(cp, f"carry:{name}", cell, boundaries, mirrors, poles,
                      CARRY_STEPS, product, electric=True, magnetic=False)
        passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                      require_repairs=True)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        name, cell, boundaries, mirrors, poles, _steps = CASES[case_index(case_name)]
        row = run_leg(cp, f"null_control:{name}", cell, boundaries, mirrors, poles,
                      CARRY_STEPS, product, electric=True, magnetic=False,
                      bracket=False)
        # REQUIRES DIVERGENCE. A bracket that changes nothing is not load-bearing,
        # and a carry family whose unbracketed twin agreed would be measuring a seam
        # that carried no deposit.
        #
        # THE LAUNCH FLOOR IS "AT LEAST ONE", NOT THE BUDGET, and the vacuity census
        # is off — both for the same reason, MEASURED on this gate's first device run
        # rather than foreseen: a leg that must diverge STOPS at the first divergent
        # step, so an exact launch count and a whole-run moved-state census are
        # statements about steps that never ran. On these rows the divergence lands
        # on step 1, where the magnetic PML auxiliaries are still exactly zero
        # because E has not been written yet. What remains checkable is that the
        # fused kernel ran at all, and that the ORACLE control did NOT diverge —
        # which `verdict_of` checks unconditionally, and which is what says the
        # divergence is the missing bracket rather than a broken harness.
        passed, failures = verdict_of(row, require_identical=False,
                                      require_launches_at_least=1,
                                      require_moved=False)
        row["armed"] = True
        row["passed"], row["failures"] = passed, failures
        row["why"] = ("the fused launch WITHOUT the shipped deposit repair; it "
                      "consumes a pre-injection D and MUST diverge")
        payload["device_legs"].append(row)
        save(payload, artifact)

    log("\n=== armed harness mutations ===")
    name, cell, boundaries, mirrors, poles, _steps = CASES[
        case_index("odd_y_fold_metallic")]
    row = run_leg(cp, "armed:no_substitution", cell, boundaries, mirrors, poles, 3,
                  product, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    row = run_leg(cp, "armed:frozen_electric_seam", cell, boundaries, mirrors, poles,
                  3, product, freeze_electric=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.folded_dispersive_fused_curl_constitutive_D)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, artifact)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, artifact)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    # A DECLARED NULL MUST BE CONFIRMED, not merely permitted: a null that IS caught
    # means the reasoning behind the null is wrong, and that has to fail too. Every
    # mutation must also have been ARMED: a rewrite that hit nothing measured
    # nothing, whatever it then reported.
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
                     "mutations whose measured outcome did not match their declared "
                     "expectation, or that were never armed: "
                     + str(sorted(row["mutation"] for row in payload["mutations"]
                                  if not _mutation_ok(row))),
                     f"refusals ok: {refusal_ok}",
                     f"mutations ok: {mutation_ok}"]),
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
