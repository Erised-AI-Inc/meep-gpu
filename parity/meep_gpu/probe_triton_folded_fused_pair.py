"""Byte gate for the folded fused ELECTRIC pair: ``step_D`` welded into ``update_E``.

DEVICE STATUS: **RELEASED 2026-08-20** on the GPU host, RTX A6000 GPU 4, Triton 3.1.0 /
    CuPy 13.5.1, ``keep`` subnormal policy —
    ``parity/meep_gpu/results/triton_folded_fused_pair_2026-08-20/``. 5/5 device
    cases bit-identical over 10 complete steps against both oracles, 2/2 armed
    harness mutations refused, 10/11 kernel mutations caught with the eleventh a
    confirmed null, 4/4 refusals.

    THE FIRST RUN FAILED, IN THIS HARNESS. The staged tree did not carry the
    186-row censuses, so :func:`corpus_admission_leg` funnelled 0 -> 0 and reported
    "the Triton predicate admits NO corpus row" — a measurement about the staging
    presented as one about the corpus. That leg now refuses a census that does not
    yield 186 measured rows, naming the path, before it reports any funnel.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.folded_fused_pair.folded_fused_pair_coverage`
admits, ONE launch of ``folded_fused_curl_constitutive_D`` leaves the engine in a
state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_D`` / ``fill_symmetry_bc_D`` /
    ``zero_metal_D`` / ``fill_folded_far_ghosts_D`` / ``update_E``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the folded curl
    (``symmetry.FoldedPmlCurlPlan`` on ``step_D``), the mirror ghost fill
    (``symmetry.MirrorGhostFillPlan`` on family ``D``) and the folded constitutive
    (``symmetry.plan_folded_constitutive`` on side ``E``), with ``zero_metal_D`` on
    the array path in both routes because no Triton product owns the wall clear,

over every allocated volume: the primaries, the split-field PML auxiliaries and the
constitutive ``f_w`` history. Comparison is on the uint32 view. ``allclose`` appears
nowhere.

WHAT THE GATE REFUSES TO INFER.

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is counted
  through a proxy that owns the kernel object, every substitution is counted per
  route in the installer, and one ARMED HARNESS MUTATION removes the substitution
  so the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** Every compared array
  must MOVE during the leg; a leg whose electric trio is frozen in all three routes
  is armed and must be caught by the moved-state census.
* **A mutation scored on a grid that never enters its branch measures nothing.**
  Every carry block in this kernel is guarded by ``BC? == MIRROR_METALLIC`` on a
  SPECIFIC axis, and the corner blocks by two of them. :data:`MUTATION_CASE` names,
  per mutation, the case whose fold set actually reaches the lines it rewrites, and
  :func:`mutation_case_for` refuses a pairing that does not.

===========================================================================
THE ONE STRUCTURAL LEG THAT IS NOT A MIRROR
===========================================================================

This product's new hazard is not arithmetic — the curl, the wall clear and the
constitutive are byte-copies of three certified Triton bodies — it is the CARRY's
geometry: which lane owns which destination cell, at what offset, with which
product of parities. A test that re-implemented that geometry would mirror a defect
instead of executing it, which this project has measured costing 86 of 87 tests
their bite.

So :func:`carry_table_leg` PARSES the carry blocks out of the shipped kernel text —
the ``fill_`` masks, the ``dst_`` offsets, the ``gv_`` parity expressions and the
``ZM_`` guards — and then EXECUTES that parsed table against
``stepping.fill_symmetry_bc_D``'s own output on real grids. Nothing about the
kernel's geometry is assumed; the offsets and weights come out of the source and
the answer comes out of the array path.

Usage::

    # laptop, no CUDA, no Triton — the legs that do not need a device
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_fused_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_fused_pair.py \\
        --out parity/meep_gpu/results/<fresh-dir>/gate.json
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
import itertools
import json
import os
import re
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

SEED = 20260820

#: (name, cell, boundaries, mirrors, steps). Every case folds at least one axis
#: over a METALLIC outer declaration: a folded PERIODIC axis is refused by name.
#:
#: THE Z AXIS IS PERIODIC IN EVERY 2-D CASE, NOT METALLIC. A 2-D cell is
#: translationally invariant along z; a metallic declaration there is not a wall but
#: a polarization filter, and ``Grid`` refuses it by name (grid.py:842-877).
#:
#: THE PARITIES AND THE WALL SET ARE BOTH SWEPT, and neither is decoration. An EVEN
#: plane makes the parity mutations invisible — the multiply is by +1 — so the odd
#: rows are what give them anywhere to be caught. ``two_folds`` is the ONLY case
#: that reaches a corner block, and it carries two DIFFERENT parities so the corner
#: weight is a product that is not its own square.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any,
                   Tuple[Tuple[str, int], ...], int], ...] = (
    ("even_y_fold_metallic", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", 1),), 10),
    ("odd_y_fold_metallic", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", -1),), 10),
    ("even_x_fold_metallic", (3.0, 3.2, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("X", 1),), 10),
    ("two_folds_metallic", (3.0, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("X", 1), ("Y", -1)), 10),
    ("y_fold_periodic_x", (3.2, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("Y", 1),), 10),
    # ------------------------------------------------------------------
    # THE FOLDED PERIODIC CASES (2026-08-21). Added with the far carry, and
    # nothing before them reached a single `FAR_a` block: every case above folds
    # over a METALLIC outer declaration, which is CODE_MIRROR_METALLIC, and the
    # far fill runs only on CODE_MIRROR_PERIODIC (stepping._stored_past_owned).
    # A case table that stopped at the five above would have released a carry no
    # leg executed — which is the defect an adversarial verifier found in the
    # magnetic twin, one rung up, on 2026-08-21.
    #
    # THE FULL COUNT'S PARITY IS SWEPT AND IT IS THE POINT. `_far_reflect_rows` is
    # `n_full - stored + 2`, which is `stored - 2` at an EVEN full count and
    # `stored - 3` at an ODD one. MEASURED on this exact declaration at
    # resolution 12: cell 3.0 -> full 36, stored 20, reflect 18 (= stored - 2);
    # cell 37/12 -> full 37, stored 21, reflect 18 (= stored - 3). A kernel that
    # baked `n - 2` is exactly right on the first and a whole cell wrong on the
    # second, so only the odd row can score `m17`.
    ("x_fold_periodic_even", (3.0, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("X", 1),), 10),
    ("x_fold_periodic_odd", (37.0 / 12.0, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("X", -1),), 10),
    # BOTH axes folded PERIODIC, at DIFFERENT parities: the only case that reaches
    # a far/near composite on two axes and whose composed weight is a product that
    # is not its own square.
    ("two_folds_periodic", (3.0, 37.0 / 12.0, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", -1)), 10),
    # A FOLDED PERIODIC AXIS BESIDE A WALL. This is the only shape in which the
    # near ghost's parity-then-clear order and the far ghost's clear-then-parity
    # order can disagree, and `measure_d_side_composition_law.py` scores exactly
    # that: without a wall, `clear_before_parity` and `far_reclears` match the
    # array path on every declaration.
    ("x_fold_periodic_y_wall", (3.0, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("X", -1),), 10),
    # ------------------------------------------------------------------
    # THE 3-D CASES. Every case above is 2-D, where z is invariant — so `FAR_Z`,
    # `NEAR_Z`, every `gv_2*` register and the `kp_f2`/`km_f2` pair are
    # COMPILE-TIME ABSENT and the whole z half of the carry is shipped code no
    # leg executes. `build_driver` reads the dimensionality off the cell's z
    # extent, so these two rows are the only thing that brings it in.
    #
    # THREE FOLDED PERIODIC AXES is the deepest composition this family can emit:
    # each component owns its far plane, both near planes, and every combination
    # of the three, so one source lane writes SEVEN ghost cells.
    ("xyz_all_folded_3d", (2.0, 2.0, 2.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", -1), ("Z", 1)), 10),
    # A FAR FOLD, A NEAR FOLD AND A WALL AT ONCE. This is the only shape in which
    # the CHAINED composite and a raw parity product differ: `gv_0y` is clamped by
    # the z wall (Dx clears on y and z; y is folded, so only z can), and the
    # far-over-near cell is the far parity times THAT clamped value. In 2-D it
    # cannot exist — the only axis that could clear `gv_0y` is z, and z is
    # invariant there — which is why `m12` reported UNCAUGHT on the 2-D table and
    # was measuring nothing.
    # THE Y PLANE IS ODD, AND THAT IS THE WHOLE OF WHY THIS CASE CATCHES `m12`.
    # The kernel's `v0` is ALREADY wall-clamped when the carry reads it, so the
    # near ghost's re-clamp can only change the SIGN BIT OF A ZERO: at a clamped
    # cell `PHY * v0` is `PHY * (+0.0)`, which is `+0.0` on an even plane and
    # `-0.0` on an odd one, and the re-clamp turns the latter back into `+0.0`.
    # MEASURED: with PHY = +1 the chained composite and the raw product are
    # bitwise equal; with PHY = -1 they differ in exactly that bit. An even plane
    # here would report a real defect as uncaught, which is what the first cut of
    # this case did.
    ("x_periodic_y_fold_z_wall_3d", (2.0, 2.0, 1.5),
     {"x": "periodic", "y": "metallic", "z": "metallic"},
     (("X", -1), ("Y", -1)), 10),
)

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

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

#: THE READ-ONLY MATERIAL VOLUMES, under the names ``inventory`` really finds.
#: A set of names that matches nothing disables BOTH the vacuity floor (which would
#: then demand that a read-only input move) and ``material_changed`` (which could
#: never fire). ``_assert_material_names_are_real`` is what stops a rename doing it.
MATERIAL = ("eps", "inv_eps")

#: Flat-index stride per axis, in the kernel's own spelling.
STRIDE_SPELLING: Tuple[str, str, str] = ("nyz", "nz", "")

#: The kernel's phase constexpr per axis.
PHASE_SPELLING: Tuple[str, str, str] = ("PHX", "PHY", "PHZ")

#: The kernel's wall-clear constexpr and coordinate test per axis.
ZM_SPELLING: Tuple[str, str, str] = ("ZM_X", "ZM_Y", "ZM_Z")
AT_SPELLING: Tuple[str, str, str] = ("at_x", "at_y", "at_z")

#: The kernel's coordinate variable per axis.
COORDINATE: Tuple[str, str, str] = ("i", "j", "k")

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and material_changed would never fire. Fix the names, do not loosen "
            f"the floor.")


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
    names = (
        "meep_gpu/triton_kernels/folded_fused_pair.py",
        "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/dispersive_fused_pair.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_folded_fused_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Reading the shipped kernel text
# ---------------------------------------------------------------------------

_SOURCE_FILE = {
    "pml_curl_step_folded": "symmetry.py",
    "constitutive_step": "kernels.py",
    "fused_curl_dispersive_E": "dispersive_fused_pair.py",
    "folded_fused_curl_constitutive_D": "folded_fused_pair.py",
    # The carry became a device function when the far fill landed (2026-08-21):
    # a D component can now be the source of up to SEVEN ghost cells, and seven
    # inlined copies of the same eight statements is seven places for one to drift.
    "_carry_ghost_E": "folded_fused_pair.py",
}


def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported: this leg has to run on a host with no
    Triton, where importing ``kernels.py`` raises, so the transcription check bites
    at the merge bar rather than only on a device run.

    The text is exact — never ``ast.unparse``d — because what is being checked is
    the PARENTHESISATION, and unparsing re-derives minimal parentheses and would
    silently rewrite ``dtdx * ((c_y - c) + (b - b_z))`` into a different grouping.

    The docstring IS stripped: every one of these bodies documents the codes and
    branches it does NOT carry, so a text search over the raw source finds those
    names in prose and reports them as live code.
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

#: The arithmetic lines the fused body must reproduce VERBATIM from the folded
#: curl. Each is a line whose grouping decides float32 bits; a reformat is a
#: different number, not a style change.
CURL_LINES = (
    "curl0 = dtdx * ((c_y - c) + (b - b_z))",
    "curl1 = dtdx * ((a_z - a) + (c - c_x))",
    "curl2 = dtdx * ((b_x - b) + (a - a_y))",
    "n0 = ((p0 * km_y) - curl0) * si_y",
    "n1 = ((p1 * km_z) - curl1) * si_z",
    "n2 = ((p2 * km_x) - curl2) * si_x",
)

#: The D-family wall clear, byte-copied from ``dispersive_fused_pair``'s ZM block —
#: the same six rows ``stepping._zero_metal`` writes.
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
#: It arrived with the far carry: Dx (1,0,0) Dy (0,1,0) Dz (0,0,1) is shift 1 on the
#: component's OWN axis only, so this arm is three lines against the B arm's six.
#: Carried here because the fused body is a copy of that emitter and a hand-written
#: mask would be a second implementation of the ownership rule.
TOP_PLANE_LINES = (
    "curl0 = tl.where(last_x, 0.0, curl0)",
    "curl1 = tl.where(last_y, 0.0, curl1)",
    "curl2 = tl.where(last_z, 0.0, curl2)",
)

#: ``_carry_ghost_E``'s eight statements — the displacement store followed by
#: ``constitutive_step``'s ``SCALE=1`` arm with ``src`` bound to the carried value.
#: Checked as a body of its own so a drift inside the device function is caught
#: even though every call site below reads identically.
CARRY_GHOST_LINES = (
    "tl.store(f + dst, ghost, mask=mask)",
    "prev = tl.load(w + dst, mask=mask, other=0.0)",
    "src = ghost * tl.load(ie + dst, mask=mask, other=0.0)",
    "tl.store(w + dst, src, mask=mask)",
    "acc = tl.load(e + dst, mask=mask, other=0.0)",
    "acc = acc + kp_d * src",
    "acc = acc - km_d * prev",
    "tl.store(e + dst, acc, mask=mask)",
)


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line of the fused body traced to the source it came from."""
    fused = _statements(_shipped_text("folded_fused_curl_constitutive_D"))
    carry_ghost = _statements(_shipped_text("_carry_ghost_E"))
    folded_curl = _statements(_shipped_text("pml_curl_step_folded"))
    constitutive = _statements(_shipped_text("constitutive_step"))
    dispersive = _statements(_shipped_text("fused_curl_dispersive_E"))

    findings: List[str] = []
    for line in CURL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the curl line {line!r}")
        if line not in folded_curl:
            findings.append(
                f"symmetry.pml_curl_step_folded no longer contains {line!r}; the "
                f"fused body's transcription source has moved")
    for line in ZERO_METAL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the wall-clear line "
                            f"{line!r}")
        if line not in dispersive:
            findings.append(
                f"dispersive_fused_pair's ZM block no longer contains {line!r}; the "
                f"wall clear's transcription source has moved")
    for line in CONSTITUTIVE_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain {line!r}")
    # The constitutive half is `constitutive_step`'s shape with its own register
    # names; what must survive verbatim is the grouping.
    for line in ("a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"):
        if line not in constitutive:
            findings.append(
                f"kernels.constitutive_step no longer contains {line!r}; the "
                f"constitutive grouping's source has moved")
    # D ON THE LEFT of the inverse-epsilon multiply, at the owned cell and at every
    # imaged ghost. stepping.py:1011-1013 writes `source * inverse_epsilon`, and
    # bitwise commutativity is not a licence to transcribe it backwards.
    for register in ("src0 = v0 * tl.load(ie0 + idx",
                     "src1 = v1 * tl.load(ie1 + idx",
                     "src2 = v2 * tl.load(ie2 + idx"):
        if not any(line.startswith(register) for line in fused):
            findings.append(
                f"the fused body does not form {register!r}: the E product is "
                f"D * inv_eps with D on the LEFT (stepping.py:1011-1013)")

    # The ownership restructure, structurally: the destination lane's D / E / f_w_E
    # traffic must be MASKED OFF, not merely unused. A lane that loaded and
    # discarded would still read a word another lane writes.
    for needle in ("mask=own0", "mask=own1", "mask=own2"):
        if sum(1 for line in fused if needle in line) < 6:
            findings.append(
                f"{needle} appears on fewer than six lines; the destination lane's "
                f"D/f_w_E/E traffic is not fully masked off")
    # THE TOP-PLANE MASK MUST NOW BE HERE. It used to be forbidden: while the
    # predicate refused MIRROR_PERIODIC the block was unreachable, and an
    # unreachable branch is a lie about what the kernel does. Carrying
    # fill_folded_far_ghosts_D brings the code in, so the check FLIPS — both that
    # the fused body carries the three lines and that the certified emitter still
    # spells them the same way, so neither side can move alone.
    for line in TOP_PLANE_LINES:
        if line not in fused:
            findings.append(
                f"the fused body does not contain the top-plane mask line {line!r}; "
                f"the far carry writes the plane that mask un-owns")
        if line not in folded_curl:
            findings.append(
                f"symmetry.pml_curl_step_folded no longer contains {line!r}; the "
                f"top-plane mask's transcription source has moved")
    # ...and it must be the BACKWARD arm, three lines and not the B family's six.
    # A body carrying six would be masking the two axes that are NOT the
    # component's own, which is the magnetic geometry on the electric family.
    masks = [line for line in fused if line.startswith("curl")
             and "tl.where(last_" in line]
    if len(masks) != 3:
        findings.append(
            f"the fused body carries {len(masks)} top-plane "
            f"mask lines, not 3; the D family's Yee shifts put shift 1 on the "
            f"component's OWN axis only (symmetry.py:294-301)")

    # _carry_ghost_E is the certified constitutive arm, MOVED and not rewritten.
    for line in CARRY_GHOST_LINES:
        if line not in carry_ghost:
            findings.append(f"_carry_ghost_E does not contain {line!r}")
    # Every ghost goes through it: no carry may open-code the eight statements
    # again, which is the drift the device function exists to prevent.
    if any(line.startswith("tl.store(f0 + dst") or line.startswith("tl.store(f1 + dst")
           or line.startswith("tl.store(f2 + dst") for line in fused):
        findings.append(
            "the fused body open-codes a ghost store; every imaged cell must go "
            "through _carry_ghost_E")

    # THE COEFFICIENT REGISTERS ARE LOADED TWICE, AND THAT IS THE D GEOMETRY.
    # update_E indexes component m on axis m; the NEAR fill images along an axis
    # that is never m, so a near-only destination reuses the source lane's pair,
    # and the FAR fill images along exactly m, so its destination needs the pair at
    # the TOP row. One load would mean a far ghost took its source's coefficient —
    # the error the magnetic twin's gate carries a mutation for, transposed.
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
    return {
        "leg": "transcription",
        "device": False,
        "fused_statements": len(fused),
        "carry_ghost_statements": len(carry_ghost),
        "curl_lines_checked": len(CURL_LINES),
        "top_plane_lines_checked": len(TOP_PLANE_LINES),
        "zero_metal_lines_checked": len(ZERO_METAL_LINES),
        "constitutive_lines_checked": len(CONSTITUTIVE_LINES),
        "carry_ghost_lines_checked": len(CARRY_GHOST_LINES),
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the carry table, PARSED out of the kernel and EXECUTED
# ---------------------------------------------------------------------------

#: The carry's target/workspace/field/material argument quadruple, per component.
#: Read off the call rather than assumed, so a block that imaged ``v0`` into ``f1``
#: is a parse mismatch here instead of a plausible field on a device.
_CARRY_ARGS: Tuple[Tuple[str, str, str, str], ...] = (
    ("f0", "w0", "e0", "ie0"), ("f1", "w1", "e1", "ie1"),
    ("f2", "w2", "e2", "ie2"),
)

#: The source-lane predicate per axis, for each fill. NEAR reads stored cell 2;
#: FAR reads the runtime reflect row.
NEAR_LANE_SPELLING: Tuple[str, str, str] = ("near_i", "near_j", "near_k")
FAR_LANE_SPELLING: Tuple[str, str, str] = ("far_i", "far_j", "far_k")

#: The destination offset per axis, in the kernel's own spelling.
NEAR_OFFSET_SPELLING: Tuple[str, str, str] = ("dn_x", "dn_y", "dn_z")
FAR_OFFSET_SPELLING: Tuple[str, str, str] = ("df_x", "df_y", "df_z")

#: The coefficient pair a destination takes. A NEAR-only ghost does not move the
#: indexed axis and reuses the owned lane's; anything with the FAR half in it sits
#: at the top row of the indexed axis and takes the pair loaded there.
NEAR_COEFFICIENTS: Tuple[Tuple[str, str], ...] = (
    ("kp_0", "km_0"), ("kp_1", "km_1"), ("kp_2", "km_2"))
FAR_COEFFICIENTS: Tuple[Tuple[str, str], ...] = (
    ("kp_f0", "km_f0"), ("kp_f1", "km_f1"), ("kp_f2", "km_f2"))


def _guarded_statements(node: Any) -> List[Tuple[Tuple[str, ...], Any]]:
    """Every statement in ``node``'s body, paired with its ``if`` guard chain.

    The guards are the constexpr tests the kernel branches on, unparsed exactly as
    written (``NEAR_Y``, ``FAR_X and NEAR_Z``, ``ZM_Y``). Reading them off the tree
    rather than matching text is what lets a three-term parenthesised guard be
    understood without a regex that would have to know the parenthesisation rule.
    """
    out: List[Tuple[Tuple[str, ...], Any]] = []

    def walk(body: Sequence[Any], guards: Tuple[str, ...]) -> None:
        for statement in body:
            if isinstance(statement, ast.If):
                inner = guards + (ast.unparse(statement.test),)
                walk(statement.body, inner)
                walk(statement.orelse, guards + ("!" + ast.unparse(statement.test),))
            else:
                out.append((guards, statement))

    walk(node.body, ())
    return out


def parse_carry_table() -> List[Dict[str, Any]]:
    """The kernel's carry blocks, read out of the shipped source text.

    Returns one record per imaged cell: the target component, the NEAR axes and the
    FAR axis it images on, the destination offset in the kernel's own spelling, the
    value expression, the coefficient pair, the guarding constexprs and the source
    lane mask.

    PARSED, NOT MIRRORED. The point of this leg is that a wrong offset, a missing
    corner, a swapped parity or a far ghost that took its source's coefficient
    becomes a wrong answer HERE, on a laptop, rather than a plausible field on a
    device. Nothing below re-derives what the kernel decided; it reads it.

    REWRITTEN 2026-08-21 for the far carry. The pre-carry parser matched three
    named locals per block (``fill_``/``dst_``/``gv_``); the carry moved the eight
    constitutive statements into ``_carry_ghost_E`` and the blocks became calls, so
    the parser follows the kernel rather than the kernel being held to a parser.
    """
    text = _shipped_text("folded_fused_curl_constitutive_D")
    tree = ast.parse(text)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef))

    # Pass 1 — the imaged NEAR values and the wall clears applied to each. A `gv_`
    # name is assigned once with its parity expression and then re-assigned under
    # `if ZM_?:` guards with `tl.where(at_?, 0.0, gv_?)`; both are recorded.
    imaged: Dict[str, str] = {}
    cleared: Dict[str, List[Tuple[str, str]]] = {}
    for guards, statement in _guarded_statements(function):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("gv_"):
            continue
        rhs = ast.unparse(statement.value)
        match = re.fullmatch(r"tl\.where\((at_[xyz]), 0\.0, gv_\w+\)", rhs)
        if match:
            guard = next((g for g in guards if g.startswith("ZM_")), "")
            cleared.setdefault(target.id, []).append((guard, match.group(1)))
        else:
            imaged[target.id] = rhs

    # Pass 1b — THE LANE DEFINITIONS, THE OWNERSHIP MASKS AND THE OFFSETS.
    #
    # READ, NOT RE-DERIVED, and that is a correction rather than a refinement: the
    # first version of this leg recomputed ownership from NEAR_FILL_AXES and
    # hardcoded `== 2` / `== reflect`, which made it a MIRRORED EVALUATOR of the
    # very rule it was meant to check. The planted-defect leg caught it — five
    # mutations were planted and the table still passed, including the far carry
    # being dropped outright. Everything the kernel decides about WHICH lane writes
    # WHERE is now parsed out of the kernel.
    lanes: Dict[str, str] = {}
    ownership: Dict[str, List[Tuple[Tuple[str, ...], str]]] = {}
    offsets: Dict[str, str] = {}
    for guards, statement in _guarded_statements(function):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        rhs = ast.unparse(statement.value)
        if target.id in NEAR_LANE_SPELLING + FAR_LANE_SPELLING:
            lanes[target.id] = rhs
        elif target.id in NEAR_OFFSET_SPELLING + FAR_OFFSET_SPELLING:
            offsets[target.id] = rhs
        elif re.fullmatch(r"own[012]", target.id):
            ownership.setdefault(target.id, []).append((guards, rhs))

    # Pass 2 — the carry calls.
    records: List[Dict[str, Any]] = []
    for guards, statement in _guarded_statements(function):
        if not isinstance(statement, ast.Expr):
            continue
        call = statement.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_carry_ghost_E"):
            continue
        args = [ast.unparse(argument) for argument in call.args]
        if len(args) != 9:
            raise AssertionError(f"_carry_ghost_E takes 9 arguments, got {args}")
        field, workspace, accumulator, material, destination = args[:5]
        value, kp_name, km_name, mask = args[5:]
        try:
            target = _CARRY_ARGS.index((field, workspace, accumulator, material))
        except ValueError:
            raise AssertionError(
                f"a carry writes ({field}, {workspace}, {accumulator}, {material}), "
                f"which is not one component's quadruple") from None
        offset = destination[len("idx"):].strip() if destination.startswith("idx") \
            else destination
        near_axes = tuple(axis for axis in range(3)
                          if NEAR_OFFSET_SPELLING[axis] in offset.split())
        far_axes = tuple(axis for axis in range(3)
                         if FAR_OFFSET_SPELLING[axis] in offset.split())
        base = value
        while base.startswith("-"):
            base = base[1:].strip()
        records.append({
            "target": target,
            "near_axes": near_axes,
            "far_axes": far_axes,
            "offset": offset,
            "value": value,
            "value_base": base.split("*")[-1].strip(),
            "coefficients": (kp_name, km_name),
            "guards": tuple(guards),
            "mask": mask,
            "imaged": imaged,
            "cleared": cleared,
            "lanes": lanes,
            "ownership": ownership,
            "offsets": offsets,
        })
    return records


def _scalar_scope(shape: Sequence[int], reflect: Sequence[int]) -> Dict[str, int]:
    """The scalar names a kernel expression may mention."""
    nx, ny, nz = (int(n) for n in shape)
    rx, ry, rz = (int(r) for r in reflect)
    return {"nx": nx, "ny": ny, "nz": nz, "nyz": ny * nz,
            "rx": rx, "ry": ry, "rz": rz}


def _offset_elements(spelling: str, shape: Sequence[int], reflect: Sequence[int],
                     offsets: Dict[str, str]) -> int:
    """Evaluate one parsed destination offset against a concrete shape.

    The spelling is the kernel's own (``+ df_x + dn_y``) AND SO ARE THE DEFINITIONS
    it resolves through — ``offsets`` is the kernel's own ``dn_x = -2 * nyz`` and
    ``df_x = (nx - 1 - rx) * nyz`` lines, parsed. Nothing about the stride or the
    reflect row is restated here, so a kernel that got either wrong evaluates wrong.
    """
    scope: Dict[str, Any] = dict(_scalar_scope(shape, reflect))
    for name, definition in offsets.items():
        scope[name] = int(eval(definition, {"__builtins__": {}}, scope))  # noqa: S307
    return int(eval(spelling, {"__builtins__": {}}, scope))  # noqa: S307


def _lane_mask(expression: str, coords: Any, shape: Sequence[int],
               reflect: Sequence[int]) -> Any:
    """Evaluate one parsed per-lane boolean over the whole coordinate grid.

    The expression is the kernel's (``live & (i == rx)``, ``own0 & (j != 0)``);
    ``live`` is every lane, because this leg iterates the allocated volume exactly.
    """
    scope: Dict[str, Any] = dict(_scalar_scope(shape, reflect))
    scope.update({"i": coords[0], "j": coords[1], "k": coords[2],
                  "live": np.ones(coords[0].shape, dtype=bool)})
    return np.asarray(eval(expression, {"__builtins__": {}}, scope), dtype=bool)  # noqa: S307


def _guard_holds(guard: str, near: Sequence[bool], far: Sequence[bool],
                 walls: Sequence[bool]) -> bool:
    """Is one parsed constexpr guard true for this configuration?"""
    scope = {
        "NEAR_X": bool(near[0]), "NEAR_Y": bool(near[1]), "NEAR_Z": bool(near[2]),
        "FAR_X": bool(far[0]), "FAR_Y": bool(far[1]), "FAR_Z": bool(far[2]),
        "ZM_X": bool(walls[0]), "ZM_Y": bool(walls[1]), "ZM_Z": bool(walls[2]),
    }
    return bool(eval(guard, {"__builtins__": {}}, scope))  # noqa: S307


def carry_table_leg() -> Dict[str, Any]:
    """The parsed carry table, executed against the driver's own three passes.

    THE ORACLE IS THE ARRAY PATH. A random ``D`` volume is put through
    ``fill_symmetry_bc_D`` -> ``zero_metal_D`` -> ``fill_folded_far_ghosts_D`` on
    one copy; on another, only the parsed table's writes are applied, by the lanes
    the table says own them. Bitwise agreement over the whole volume is what says
    the cover, the offsets, the weights AND THE ORDER are right.

    THIS IS THE LEG THAT WOULD CATCH THE FOUR COMPOSITION ERRORS the measurement in
    ``measure_d_side_composition_law.py`` scored: it executes what the KERNEL says,
    against the array path, on grids that discriminate them. It is NOT a substitute
    for the device gate — it cannot see a memory hazard, a register reuse or a
    rounding order inside one launch, and the collision census below is a statement
    about the parsed table, not about what the hardware does with it.

    THE OWNERSHIP RULE IS THE TABLE'S, NOT THIS FUNCTION'S: a lane carries only if
    it owns its own cell AND sits on every source plane the block names, which is
    read off the parsed ``own<t> & ...`` mask. A kernel that dropped an ownership
    term parses into a table whose writes collide, and the exactly-once census
    fails.
    """
    from meep_gpu.fields import IYEE_SHIFTS, Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu import stepping  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_fused_pair as product  # noqa: PLC0415

    table = parse_carry_table()
    findings: List[str] = []

    # (a) STRUCTURAL: one block per nonempty subset of {near axes} u {far axis}.
    expected = set()
    for target in range(3):
        slots = ([("near", axis) for axis in product.NEAR_FILL_AXES[target]]
                 + [("far", axis) for axis in product.FAR_FILL_AXES[target]])
        for size in range(1, len(slots) + 1):
            for subset in itertools.combinations(slots, size):
                expected.add((target,
                              tuple(sorted(a for kind, a in subset if kind == "near")),
                              tuple(sorted(a for kind, a in subset if kind == "far"))))
    seen = {(record["target"], record["near_axes"], record["far_axes"])
            for record in table}
    if seen != expected:
        findings.append(
            f"the carry blocks do not cover the fill subsets exactly: "
            f"missing {sorted(expected - seen)}, extra {sorted(seen - expected)}")

    for record in table:
        target = record["target"]
        tag = (f"t{target}"
               f"_near{''.join('xyz'[a] for a in record['near_axes']) or '-'}"
               f"_far{''.join('xyz'[a] for a in record['far_axes']) or '-'}")
        # (b) guarded by exactly the NEAR_/FAR_ tests for its axes.
        want_guards = set(f"NEAR_{'XYZ'[axis]}" for axis in record["near_axes"])
        want_guards |= set(f"FAR_{'XYZ'[axis]}" for axis in record["far_axes"])
        have = set()
        for guard in record["guards"]:
            have |= set(re.findall(r"(?:NEAR|FAR)_[XYZ]", guard))
        if have != want_guards:
            findings.append(
                f"{tag}: guarded by {sorted(have)}, expected {sorted(want_guards)}")
        # (c) the source lane mask names this component's ownership AND every
        #     source plane the block images from.
        want_mask = {f"own{target}"}
        want_mask |= {NEAR_LANE_SPELLING[axis] for axis in record["near_axes"]}
        want_mask |= {FAR_LANE_SPELLING[axis] for axis in record["far_axes"]}
        have_mask = set(part.strip() for part in record["mask"].split("&"))
        if have_mask != want_mask:
            findings.append(
                f"{tag}: source lane mask is {sorted(have_mask)}, expected "
                f"{sorted(want_mask)} — every carry must be ANDed with the "
                f"component's own ownership mask (the defect the magnetic twin's "
                f"device gate found on 2026-08-20)")
        # (d) THE COEFFICIENT PAIR. A near-only destination does not move the
        #     indexed axis; anything with the far half in it does.
        want_pair = (FAR_COEFFICIENTS[target] if record["far_axes"]
                     else NEAR_COEFFICIENTS[target])
        if record["coefficients"] != want_pair:
            findings.append(
                f"{tag}: takes coefficients {record['coefficients']}, expected "
                f"{want_pair} — update_E indexes component {target} on axis "
                f"{'xyz'[target]}, which the FAR fill images along and the NEAR "
                f"fill never does")
        # (e) THE VALUE IS CHAINED. A far-over-near cell must be the far parity
        #     applied to the ALREADY-CLAMPED near ghost register, not to `v`.
        if record["far_axes"] and record["near_axes"]:
            axis = record["far_axes"][0]
            near_tag = ("gv_" + str(target)
                        + "".join("xyz"[a] for a in record["near_axes"]))
            want_value = f"-{PHASE_SPELLING[axis]} * {near_tag}"
            if record["value"] != want_value:
                findings.append(
                    f"{tag}: images {record['value']!r}, expected {want_value!r} — "
                    f"the driver clears at :3301 BETWEEN the two fills, so a "
                    f"far-over-near cell is the far parity times the clamped near "
                    f"ghost and not a product of raw parities")

    # (f) EXECUTED: the table's writes against the array path's own three passes.
    #
    # THE CASE TABLE INCLUDES A WALL ON AN UNFOLDED AXIS, and that is the whole
    # point of it: without one, `clear_before_parity` and `far_reclears` are
    # invisible and this leg would agree with a kernel that had either wrong.
    rows: List[Dict[str, Any]] = []
    cases = (
        ("near_only_even_y", (1.6, 3.0, 1.0), (("Y", 1),), "metallic", None),
        ("near_only_odd_y", (1.6, 3.0, 1.0), (("Y", -1),), "metallic", None),
        ("far_even_x", (3.0, 1.6, 1.0), (("X", 1),), "periodic", None),
        ("far_odd_x", (3.2, 1.6, 1.0), (("X", -1),), "periodic", None),
        ("far_two_folds", (3.0, 3.0, 1.0), (("X", 1), ("Y", -1)), "periodic", None),
        ("far_three_folds", (3.0, 3.0, 3.0), (("X", 1), ("Y", -1), ("Z", 1)),
         "periodic", None),
        ("far_x_wall_z", (3.0, 1.6, 1.0), (("X", 1),), "periodic", {"z": "metallic"}),
        ("far_xy_wall_z", (3.0, 3.0, 1.0), (("X", 1), ("Y", -1)), "periodic",
         {"z": "metallic"}),
    )
    names = ("Dx", "Dy", "Dz")
    for case, cell, mirrors, outer, wall in cases:
        boundaries: Any = outer
        if wall is not None:
            boundaries = {"x": outer, "y": outer, "z": outer}
            boundaries.update(wall)
        grid = Grid(resolution=6.0, cell_size=cell, boundaries=boundaries,
                    symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
        fields = Fields(grid=grid, force_complex_fields=False)
        shape = tuple(int(n) for n in grid.shape)
        near = [bool(grid.is_mirrored(axis)) for axis in range(3)]
        far = [bool(stepping._stored_past_owned(grid, axis)) for axis in range(3)]
        walls = [bool(grid.is_metallic(axis) and not grid.is_mirrored(axis))
                 for axis in range(3)]
        rows_reflect = stepping._far_reflect_rows(grid)
        reflect = [(-1 if value is None else int(value)) for value in rows_reflect]
        phases = {axis: int(grid.mirror_phase(axis))
                  for axis in range(3) if near[axis]}

        rng = np.random.default_rng(
            SEED + int.from_bytes(
                hashlib.sha256(f"carry|{case}".encode()).digest()[:4], "big"))
        seeds = {}
        for name in names:
            seeds[name] = np.ascontiguousarray(
                rng.uniform(-0.5, 0.5, size=shape).astype(np.float32))
            getattr(fields, name)[...] = seeds[name]
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        oracle = {name: np.asarray(getattr(fields, name), dtype=np.float32).copy()
                  for name in names}

        # The owned lane's own value, which is `v` after the kernel's ZM block:
        # the seed with zero_metal applied. Every carried ghost is built from it.
        owned = {}
        for name in names:
            volume = seeds[name].copy()
            for axis in range(3):
                if walls[axis] and IYEE_SHIFTS[name][axis] == 0:
                    volume[tuple(0 if a == axis else slice(None)
                                 for a in range(3))] = 0
            owned[name] = volume

        candidate = {name: owned[name].copy() for name in names}
        # Guard against a write landing outside the allocation: a mutated kernel
        # can produce one, and an IndexError is a caught defect rather than a crash.
        n_elem = int(np.prod(shape))
        writes: Dict[Tuple[str, int], int] = {}
        index = np.arange(int(np.prod(shape)), dtype=np.int64)
        coords = np.stack(np.unravel_index(index, shape))
        for record in table:
            if not all(_guard_holds(guard, near, far, walls)
                       for guard in record["guards"]):
                continue
            target = record["target"]
            name = names[target]
            # The value this block images, evaluated from the parsed expression.
            ghost = _evaluate_carry_value(record, owned[name], coords, walls,
                                          phases, names, IYEE_SHIFTS)
            offset = _offset_elements(record["offset"], shape, reflect,
                                      record["offsets"])
            # THE SOURCE LANE IS THE KERNEL'S OWN MASK, evaluated. `own<t>` is
            # rebuilt from the kernel's own accumulation lines under the guards
            # this configuration satisfies; each `near_`/`far_` term is the
            # kernel's own lane definition. Nothing about which lane owns which
            # ghost is restated here — that was the mirrored-evaluator defect the
            # planted-defect leg found in this leg's first version.
            resolved: Dict[str, Any] = {}
            own_name = f"own{target}"
            owns = np.ones(index.shape, dtype=bool)
            for guards, definition in record["ownership"].get(own_name, ()):
                if not all(_guard_holds(guard, near, far, walls)
                           for guard in guards):
                    continue
                scope_expression = definition.replace(own_name, "live", 1)
                owns &= _lane_mask(scope_expression, coords, shape, reflect)
            for axis in record["near_axes"]:
                owns &= _lane_mask(record["lanes"][NEAR_LANE_SPELLING[axis]],
                                   coords, shape, reflect)
            for axis in record["far_axes"]:
                owns &= _lane_mask(record["lanes"][FAR_LANE_SPELLING[axis]],
                                   coords, shape, reflect)
            del resolved
            flat_target = candidate[name].reshape(-1)
            flat_ghost = ghost.reshape(-1)
            for lane in index[owns].tolist():
                destination = lane + offset
                if not 0 <= destination < n_elem:
                    findings.append(
                        f"{case}: a carry writes flat index {destination}, outside "
                        f"[0, {n_elem}) — the destination this block names is not "
                        f"in the allocation")
                    break
                flat_target[destination] = flat_ghost[lane]
                key = (name, destination)
                writes[key] = writes.get(key, 0) + 1
        collisions = sorted(key for key, count in writes.items() if count > 1)
        differing = sorted(name for name in oracle
                           if not np.array_equal(
                               np.ascontiguousarray(oracle[name]).view(np.uint32),
                               np.ascontiguousarray(candidate[name]).view(np.uint32)))
        rows.append({"case": case, "shape": list(shape),
                     "near": near, "far": far, "walls": walls,
                     "reflect": reflect, "phases": phases,
                     "destination_writes": len(writes),
                     "collisions": collisions[:8], "differing_arrays": differing})
        if collisions:
            findings.append(f"{case}: {len(collisions)} destination cells are written "
                            f"by more than one lane, e.g. {collisions[:4]}")
        if differing:
            findings.append(f"{case}: the parsed carry table does not reproduce the "
                            f"driver's three D passes on {differing}")
        if not writes:
            findings.append(f"{case}: the carry table wrote NOTHING — the leg is "
                            f"vacuous on this grid")

    # NON-VACUITY, on the leg as a whole: at least one case must reach a FAR block
    # and at least one must reach a wall, or the leg agrees with a kernel that has
    # the far carry or the clear order wrong.
    if not any(any(row["far"]) for row in rows):
        findings.append("no case in this leg folds a PERIODIC axis, so not one FAR "
                        "block executed and the carry is unmeasured here")
    if not any(any(row["walls"]) and any(row["far"]) for row in rows):
        findings.append("no case carries both a far fill and a wall, so the "
                        "parity/clear ORDER is unmeasured here")
    return {"leg": "carry_table", "device": False,
            "blocks": [{key: value for key, value in record.items()
                        if key not in ("imaged", "cleared")} for record in table],
            "rows": rows, "findings": findings, "passed": not findings}


def _evaluate_carry_value(record: Dict[str, Any], owned: Any, coords: Any,
                          walls: Sequence[bool], phases: Dict[int, int],
                          names: Sequence[str], iyee: Any) -> Any:
    """Evaluate one block's parsed value expression over the whole volume.

    The expression is the KERNEL's (``-PHX * gv_0y``), resolved through the kernel's
    own ``gv_`` definitions and their ``tl.where`` clears. Nothing here decides what
    the value should be; it evaluates what the kernel wrote.
    """
    target = record["target"]
    name = names[target]

    def resolve(expression: str) -> Any:
        expression = expression.strip()
        if expression.startswith("(") and expression.endswith(")"):
            return resolve(expression[1:-1])
        if expression.startswith("-"):
            return -resolve(expression[1:])
        if "*" in expression:
            head, _, tail = expression.partition("*")
            return resolve(head) * resolve(tail)
        if expression == f"v{target}":
            return owned
        if expression in ("PHX", "PHY", "PHZ"):
            axis = ("PHX", "PHY", "PHZ").index(expression)
            return np.float32(phases.get(axis, 0))
        if expression.startswith("gv_"):
            value = resolve(record["imaged"][expression])
            for guard, at in record["cleared"].get(expression, ()):
                axis = ("ZM_X", "ZM_Y", "ZM_Z").index(guard)
                if not walls[axis] or iyee[name][axis] != 0:
                    continue
                value = np.where(coords[axis].reshape(owned.shape) == 0,
                                 np.float32(0.0), value)
            return value
        raise AssertionError(f"the carry value mentions {expression!r}, which this "
                             f"leg does not know how to resolve")

    return np.asarray(resolve(record["value"]), dtype=np.float32)

# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2b — the carry table's verdict, shown to FLIP on a PLANTED DEFECT
# ---------------------------------------------------------------------------

#: Mutations :func:`carry_table_leg` CANNOT see, with the reason each is out of its
#: reach. Declared rather than discovered: a leg that quietly scored nothing on
#: half the table would report a high catch rate and mean nothing by it.
#:
#: Every entry here is a defect in ARITHMETIC ORDER or in MEMORY TRAFFIC, which is
#: the device gate's to measure. The carry table executes the kernel's declared
#: destinations, values and masks with NumPy; it has no rounding order of its own
#: and no notion of a load.
NOT_SCOREABLE_WITHOUT_A_DEVICE: Dict[str, str] = {
    "m6_constitutive_association":
        "float32 association inside update_E; this leg does not perform the "
        "accumulation, it checks the value that reaches it",
    "m7_history_read_after_write":
        "the f_w read/write ORDER inside one launch; this leg has no workspace",
    "m8_zero_metal_dropped":
        "the OWNED lane's wall clear; this leg is handed the owned value and only "
        "checks what is imaged from it",
    "m11_commuted_multiply":
        "a declared null: IEEE multiplication commutes and only the PTX may move",
    "m16_top_plane_mask_dropped":
        "the CURL's top-plane ownership mask; this leg is handed the owned value "
        "and never forms a curl, so the defect is invisible in D and shows up in "
        "fu_D on the NEXT timestep — the device gate's to measure",
}

#: Mutations this leg catches STRUCTURALLY — the parse fails or the table stops
#: covering its subsets — rather than by comparing volumes. Recorded separately
#: because "caught" means something weaker here: it says the kernel stopped being
#: well-formed, not that a wrong number was produced.
CAUGHT_STRUCTURALLY: Dict[str, str] = {
    "m10_inverse_epsilon_at_the_source":
        "the call no longer names one component's (f, w, e, ie) quadruple, so the "
        "parser refuses it before any volume is compared",
}


def planted_defect_leg() -> Dict[str, Any]:
    """Re-run :func:`carry_table_leg` against each MUTATED kernel text.

    A leg that passes proves nothing on its own; what makes it evidence is that it
    FAILS on a kernel that is wrong. Every mutation the table declares is planted
    into the shipped text, the carry table is re-parsed and re-executed from that
    text, and the verdict is recorded.

    THE EXPECTATION IS DECLARED PER MUTATION, not inferred from the result. A
    mutation in :data:`NOT_SCOREABLE_WITHOUT_A_DEVICE` must report NOT CAUGHT with
    its reason; every other one must FLIP the verdict. Either way round, a
    disagreement is a finding — a defect this leg was supposed to see and did not,
    or a leg reacting to something it claims to be blind to.
    """
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    original = _shipped_text
    baseline = carry_table_leg()
    if not baseline["passed"]:
        return {"leg": "planted_defect", "device": False,
                "findings": ["the UNMUTATED carry table already fails; a planted "
                             "defect measures nothing against a broken baseline"],
                "baseline": baseline["findings"], "passed": False}

    for name, why, _expectation, rewrite in mutation_table():
        text = textwrap.dedent(original("folded_fused_curl_constitutive_D"))
        mutated, hits = rewrite(text)
        if hits == 0:
            findings.append(f"{name}: the rewrite matched nothing, so nothing was "
                            f"planted and the result below measures the shipped "
                            f"kernel")
            continue

        def patched(target: str, _mutated: str = mutated) -> str:
            if target == "folded_fused_curl_constitutive_D":
                return _mutated
            return original(target)

        globals()["_shipped_text"] = patched
        try:
            result = carry_table_leg()
        except Exception as error:  # noqa: BLE001 - a mutant that will not parse
            result = {"passed": False, "findings": [f"{type(error).__name__}: {error}"]}
        finally:
            globals()["_shipped_text"] = original

        caught = not result["passed"]
        blind = name in NOT_SCOREABLE_WITHOUT_A_DEVICE
        rows.append({
            "mutation": name, "why": why, "hits": hits,
            "caught_by_this_leg": caught,
            "caught_structurally": name in CAUGHT_STRUCTURALLY,
            "declared_out_of_reach": blind,
            "reason_out_of_reach": NOT_SCOREABLE_WITHOUT_A_DEVICE.get(name),
            "reason_structural": CAUGHT_STRUCTURALLY.get(name),
            "first_finding": (result["findings"][0][:200]
                              if result["findings"] else None),
        })
        if blind and caught:
            findings.append(
                f"{name} is declared out of this leg's reach but the leg CAUGHT it; "
                f"either the declaration is wrong or the leg is reacting to "
                f"something other than what it measures")
        if not blind and not caught:
            findings.append(
                f"{name} ({why}) was planted and the carry table still PASSED; the "
                f"leg does not see the defect it exists to see")

    scoreable = [row for row in rows if not row["declared_out_of_reach"]]
    return {
        "leg": "planted_defect",
        "device": False,
        "mutations_planted": len(rows),
        "scoreable_here": len(scoreable),
        "caught_here": sum(1 for row in scoreable if row["caught_by_this_leg"]),
        "caught_structurally_not_numerically": sorted(CAUGHT_STRUCTURALLY),
        "deferred_to_the_device_gate": sorted(NOT_SCOREABLE_WITHOUT_A_DEVICE),
        "rows": rows,
        "findings": findings,
        "passed": not findings,
    }






# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — the design sweep, on NumPy
# ---------------------------------------------------------------------------

def _array_path_seam(fields, pml, driver_module) -> None:
    """The four driver passes this product replaces, in driver order."""
    driver_module.step_D(fields, pml)
    driver_module.fill_symmetry_bc_D(fields)
    driver_module.zero_metal_D(fields)
    driver_module.fill_folded_far_ghosts_D(fields)
    driver_module.update_E(fields, pml)


def _emulated_seam(fields, pml, driver_module, *, carry_fill: bool = True,
                   order: str = "fill_then_clear",
                   constitutive: str = "after_fill") -> None:
    """The FUSED semantics, spelled with array ops so a laptop can execute them.

    THIS IS NOT THE KERNEL AND DOES NOT PRETEND TO BE. It cannot see a memory
    hazard, a register reuse or a rounding order inside one launch — those are the
    device gate's to measure, and the carry's geometry is
    :func:`carry_table_leg`'s. What it CAN decide is the ORDERING design: which
    value reaches ``update_E`` at the fill's destination, and in which order the
    parity and the wall clear are applied. Each is a knob, and the leg reports
    which flips the array path can see.
    """
    import meep_gpu.stepping as stepping  # noqa: PLC0415

    driver_module.step_D(fields, pml)
    if constitutive == "before_fill":
        # THE DEFECT THIS KNOB IS: update_E consumes a PRE-FILL D, and the mirror
        # image lands afterwards. D ends up right and E does not, which is exactly
        # the failure a fused pair that skipped the fill would produce.
        driver_module.zero_metal_D(fields)
        driver_module.update_E(fields, pml)
        stepping.fill_symmetry_bc_D(fields)
        return
    if carry_fill and order == "fill_then_clear":
        stepping.fill_symmetry_bc_D(fields)
        driver_module.zero_metal_D(fields)
    elif carry_fill and order == "clear_then_fill":
        # The defect: the ghost is imaged from an already-cleared source and the
        # clear is not re-applied to it, so an odd plane leaves -0.0 where the
        # array path leaves +0.0.
        driver_module.zero_metal_D(fields)
        stepping.fill_symmetry_bc_D(fields)
    else:
        driver_module.zero_metal_D(fields)
    driver_module.update_E(fields, pml)


def design_sweep_leg() -> Dict[str, Any]:
    """Each ORDERING choice flipped, on NumPy, against the array-path composition."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(mirrors, boundaries="metallic", cell=(1.6, 3.0, 1.0), tag=""):
        grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                    symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
        fields = Fields(grid=grid, force_complex_fields=False)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness=0.4)
        rng = np.random.default_rng(
            SEED + int.from_bytes(
                hashlib.sha256(f"design|{tag}".encode()).digest()[:4], "big"))
        for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                     "fu_Dx", "fu_Dy", "fu_Dz", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            array = getattr(fields, name, None)
            if array is None:
                continue
            array[...] = rng.uniform(-0.25, 0.25,
                                     size=array.shape).astype(array.dtype)
        return fields, pml

    def words(fields) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "fu_Dx", "fu_Dy", "fu_Dz",
                     "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            array = getattr(fields, name, None)
            if array is not None:
                out[name] = np.ascontiguousarray(
                    np.asarray(array, dtype=np.float32)).view(np.uint32).ravel().copy()
        return out

    knobs = (
        ("faithful", {}, "null"),
        ("drop_the_near_fill", {"carry_fill": False}, "caught"),
        ("clear_before_the_fill", {"order": "clear_then_fill"}, "unknown"),
        ("constitutive_before_the_fill", {"constitutive": "before_fill"}, "caught"),
    )
    cases = (("even_y", (("Y", 1),)), ("odd_y", (("Y", -1),)),
             ("even_x", (("X", 1),)), ("two_folds", (("X", 1), ("Y", -1))))

    rows: List[Dict[str, Any]] = []
    for case_name, mirrors in cases:
        reference_fields, reference_pml = build(mirrors, tag=case_name)
        before = words(reference_fields)
        _array_path_seam(reference_fields, reference_pml, driver_module)
        oracle = words(reference_fields)
        moved_names = sorted(name for name in oracle
                             if not np.array_equal(oracle[name], before[name]))
        for knob_name, kwargs, expectation in knobs:
            fields, pml = build(mirrors, tag=case_name)
            _emulated_seam(fields, pml, driver_module, **kwargs)
            candidate = words(fields)
            differing = sorted(name for name in oracle
                               if not np.array_equal(candidate[name], oracle[name]))
            rows.append({
                "case": case_name, "knob": knob_name, "expectation": expectation,
                "identical": not differing, "differing_arrays": differing,
                "arrays_moved_by_the_array_path": moved_names,
            })
    faithful = [row for row in rows if row["knob"] == "faithful"]
    vacuous = [row for row in rows if not row["arrays_moved_by_the_array_path"]]
    findings: List[str] = []
    for row in faithful:
        if not row["identical"]:
            findings.append(
                f"{row['case']}: the FAITHFUL emulation diverges from the array path "
                f"on {row['differing_arrays']} — the design, not the kernel, is wrong")
    if vacuous:
        findings.append(f"VACUOUS cases (nothing moved): {[r['case'] for r in vacuous]}")
    for row in rows:
        if row["expectation"] == "caught" and row["identical"]:
            findings.append(
                f"{row['case']}/{row['knob']}: expected to be caught and was not")
    return {"leg": "design_sweep", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the predicate battery
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every clause of the seam predicate, exercised on real grids."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_fused_pair as product  # noqa: PLC0415

    class _Source:
        def __init__(self, field_type: str) -> None:
            self.field_type = field_type

    def _deposit(fields):
        """A REAL electric source that publishes the index the injection writes.

        ``_Source`` is enough to PLACE a source in the seam and deliberately not
        enough to CARRY one. A row about the carry needs the engine's own source, so
        this builds one and refuses to return a source that deposits nothing --
        otherwise the admitting row would pass by measuring an empty scatter.
        """
        from meep_gpu import deposit_repair  # noqa: PLC0415
        from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

        source = VolumeSource(grid=fields.grid, component="Ez",
                              center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                              envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                              amplitude=1.0)
        assert source._n_source_points, "the case deposits nothing"
        assert deposit_repair._deposit_index(source) is not None
        return source

    def _with_flag_false(module, call):
        """``call()`` with this family's carry declaration held down, then restored.

        The OTHER DIRECTION of the flip, measured through the shipped predicate
        rather than argued. Restored in a finally so a raising predicate cannot
        leave the module lying to every row after it.
        """
        saved = module.CARRIES_DEPOSIT_REPAIR
        module.CARRIES_DEPOSIT_REPAIR = False
        try:
            return call()
        finally:
            module.CARRIES_DEPOSIT_REPAIR = saved

    def build(cell, boundaries, mirrors):
        grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                    symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
        fields = Fields(grid=grid, force_complex_fields=False)
        fields.enable_pml_storage()
        return fields, PML(grid=grid, thickness=0.2)

    def residual(verdict) -> List[str]:
        return [reason for reason in verdict.reasons if "cupy" not in reason]

    rows: List[Dict[str, Any]] = []

    def record(label: str, verdict, expect_admitted: bool, needle: str = "") -> None:
        left = residual(verdict)
        ok = (not left) if expect_admitted else bool(
            [reason for reason in left if needle in reason])
        rows.append({"case": label, "expect_admitted": expect_admitted,
                     "residual_reasons": left, "needle": needle, "passed": ok})

    fields, pml = build((1.6, 3.0, 1.0), "metallic", (("Y", 1),))
    record("folded_metallic_no_sources",
           product.folded_fused_pair_coverage(fields, pml, ()), True)
    # THE ADMITTING DIRECTION OF THE SOURCE CLAUSE. A MAGNETIC source is injected in
    # the B/H half and must NOT disqualify this pair; a clause tested only in its
    # refusing direction cannot tell "refuses electric" from "refuses everything".
    record("folded_metallic_magnetic_source",
           product.folded_fused_pair_coverage(fields, pml, (_Source("B"),)), True)
    # THE IN-SEAM ELECTRIC DEPOSIT IS CARRIED AS OF 2026-08-30, and the three rows
    # below are the whole of that change's admission story.
    #
    # (1) A source that publishes the index it writes is ADMITTED: the pair is
    #     bracketed by LeadingRepairPlan/TrailingRepairPlan and the deposit, plus
    #     every cell the post-injection fills image it into, is saved and restored.
    # (2) A source that does NOT publish one is still REFUSED BY NAME. This is the
    #     rung that keeps the flip from being a blanket admission: `_Source` above
    #     carries a field_type and nothing else, which is exactly the shape the
    #     repair cannot save.
    # (3) With CARRIES_DEPOSIT_REPAIR held False the ORIGINAL seam refusal comes
    #     straight back, so (1) is the flag's doing and not a clause that went away.
    record("folded_metallic_electric_deposit_is_carried",
           product.folded_fused_pair_coverage(fields, pml, (_deposit(fields),)),
           True)
    record("folded_metallic_electric_source_without_a_deposit_index",
           product.folded_fused_pair_coverage(fields, pml, (_Source("D"),)),
           False, "does not publish the index it writes")
    record("folded_metallic_electric_deposit_refused_when_the_flag_is_held_False",
           _with_flag_false(product, lambda: product.folded_fused_pair_coverage(
               fields, pml, (_deposit(fields),))),
           False, "is electric")
    record("undeclared_sources",
           product.folded_fused_pair_coverage(fields, pml, None),
           False, "was not declared")
    record("no_pml", product.folded_fused_pair_coverage(fields, None, ()),
           False, "no active PML")

    # FOLDED PERIODIC IS ADMITTED as of 2026-08-21. The clause that refused it was
    # retired AFTER `results/triton_folded_far_carry_d_2026-08-21/run_farcarryD5/`
    # came back ALL GREEN, never before, and this row is what would fail if the
    # retirement were ever undone without the carry going with it.
    periodic_fields, periodic_pml = build((1.6, 3.0, 1.0), "periodic", (("Y", 1),))
    record("folded_periodic_axis",
           product.folded_fused_pair_coverage(periodic_fields, periodic_pml, ()),
           True)

    flat_fields, flat_pml = build((1.6, 1.6, 1.0), "metallic", ())
    record("unfolded_grid",
           product.folded_fused_pair_coverage(flat_fields, flat_pml, ()),
           False, "no mirror plane is active")

    findings = [row["case"] for row in rows if not row["passed"]]
    return {"leg": "predicate", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — what the corpus says this is worth
# ---------------------------------------------------------------------------

CENSUS = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                      "predicate_coverage_2026-08-16_wired_convention")
METAL_CENSUS = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                            "metal_coverage_tranche6_2026-08-19")


def _census_rows(root: str) -> List[dict]:
    def load(name: str) -> List[dict]:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            return []
        return [json.loads(line) for line in open(path, encoding="utf-8")
                if line.strip()]

    record = load("examples.jsonl") + load("tests.jsonl")
    matched = {(r.get("leg"), r.get("row")): r for r in load("tests_param_matched.jsonl")}
    record = [matched.pop((r.get("leg"), r.get("row")), r) for r in record]
    return [r for r in record if r.get("measured")]


def corpus_admission_leg() -> Dict[str, Any]:
    """The funnel, on BOTH backends' censuses, and whether the two agree row for row.

    THE CROSS-CHECK IS THE POINT. Two independently written predicate stacks landing
    on the same rows through a clause ladder this long is evidence; a disagreement
    would mean one of them is wrong, and finding which is worth more than the
    product. Reported either way.
    """
    def funnel(root: str, curl_key: str, constitutive_key: str) -> Dict[str, Any]:
        rows = _census_rows(root)

        def covered(row: dict, key: str) -> bool:
            entry = row["predicates"].get(key, {})
            return bool(entry.get("covered_modulo_backend", entry.get("covered")))

        def label(row: dict) -> str:
            return f"{row['leg']}:{row['row']}"

        curl = [r for r in rows if covered(r, curl_key)]
        both = [r for r in curl if covered(r, constitutive_key)]
        no_source = [r for r in both
                     if not any(str(kind) != "B" for kind in
                                (r["configuration"].get("source_field_types") or []))]
        admitted = [
            r for r in no_source
            if all(bool(r["configuration"]["metallic"][axis])
                   for axis in range(3) if r["configuration"]["mirrored"][axis])
            and all(int(r["configuration"]["shape"][axis]) > 2
                    for axis in range(3) if r["configuration"]["mirrored"][axis])
            and not int(r["configuration"].get("n_polarizations") or 0)]
        return {"rows": len(rows), "curl": len(curl), "both": len(both),
                "no_electric_source": len(no_source),
                "no_electric_source_rows": sorted(label(r) for r in no_source),
                "admitted": len(admitted),
                "admitted_rows": sorted(label(r) for r in admitted)}

    triton = funnel(CENSUS, "folded_composition_curl@step_D",
                    "folded_constitutive@update_E")
    metal = funnel(METAL_CENSUS, "folded_pml_curl@step_D",
                   "folded_constitutive@update_E")
    findings: List[str] = []
    # AN ABSENT CENSUS IS NOT AN EMPTY ONE, and the difference is the whole reason
    # this clause exists. A staged tree that did not carry the record would funnel
    # 0 -> 0 -> 0 -> 0 and this leg would report "worth zero seam-instances" — a
    # measurement about the staging, dressed as a measurement about the corpus.
    # MEASURED 2026-08-20: that is exactly what the first GPU-host run did.
    for label, root, block in (("Triton", CENSUS, triton),
                               ("Metal", METAL_CENSUS, metal)):
        if block["rows"] != 186:
            findings.append(
                f"the {label} census at {root} yielded {block['rows']} measured "
                f"rows, not the 186 this derivation prices against: the record is "
                f"absent or truncated on this host and NOTHING below is a "
                f"measurement about the corpus")
    if any("census at" in reason for reason in findings):
        return {"leg": "corpus_admission", "device": False,
                "triton": triton, "metal": metal,
                "seam_instances_gained": None,
                "findings": findings, "passed": False}
    if triton["no_electric_source_rows"] != metal["no_electric_source_rows"]:
        findings.append(
            "the two backends' predicates do not agree on which rows clear the "
            "electric source seam: Triton "
            f"{triton['no_electric_source_rows']} vs Metal "
            f"{metal['no_electric_source_rows']} — one of the two predicate stacks "
            "is wrong and that matters more than this product")
    if not triton["admitted"]:
        findings.append("the Triton predicate admits NO corpus row: this product is "
                        "worth zero seam-instances and should not have been built")
    return {"leg": "corpus_admission", "device": False,
            "triton": triton, "metal": metal,
            "seam_instances_gained": triton["admitted"],
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side — drivers, inventory, launch counting
# ---------------------------------------------------------------------------

def build_driver(cp, cell, boundaries, mirror_specs, seed: int, magnetic: bool):
    """One folded PML driver, seeded identically for every route."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    # THE DIMENSIONALITY IS READ OFF THE CELL, and 3-D cases exist because 2-D
    # ones cannot reach a third of this kernel. In a 2-D run z is invariant, so
    # `FAR_Z`, `NEAR_Z`, every `gv_2*` register and the `kp_f2`/`km_f2` pair are
    # COMPILE-TIME ABSENT — the whole z half of the carry is shipped code no case
    # executes. That is the same defect an adversarial verifier found in the
    # magnetic twin on 2026-08-21, and finding it here cost nothing because the
    # 3-D cases below were added before the release rather than after it.
    dimensions = 3 if float(cell[2]) > 0.0 else 2
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=False, courant=0.35, boundaries=boundaries,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirror_specs),
        prefer_gpu=True, gpu_id=0,
    )
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, deliberately: the constitutive half multiplies by
    # inverse epsilon at the imaged ghost's OWN index, and a uniform material would
    # make that indistinguishable from reading it at the source's.
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    folded_axes = {axis.lower() for axis, _phase in mirror_specs}
    # A folded axis absorbs on its HIGH face only: the low face is the mirror plane.
    layers = {
        "x": {"high": 5} if "x" in folded_axes else 5,
        "y": {"high": 5} if "y" in folded_axes else 5,
    }
    if dimensions == 3:
        layers["z"] = {"high": 5} if "z" in folded_axes else 5
    driver.setup_pml(layers)
    if magnetic:
        # A MAGNETIC source is admitted: the driver injects it in the B/H seam, not
        # this one. Carrying one is what stops the source clause from being tested
        # only in its refusing direction.
        #
        # OFF THE MIRROR PLANE ON EVERY FOLDED AXIS. A point source ON the plane is
        # refused with a reason on an odd fold (from_meep.py:2686-2698); a quarter
        # of the owned half-width puts it inside the stored half and clear of the
        # absorber, which a folded axis carries on its HIGH face only.
        center = [0.0, 0.0, 0.0]
        for axis_name, _phase in mirror_specs:
            axis = "XYZ".index(axis_name.upper())
            center[axis] = 0.25 * (cell[axis] / 2.0)
        driver.add_source({"component": "Hz", "frequency": 0.31,
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

    curl = symmetry.plan_folded_pml_curl(driver.fields, driver.pml, "step_D")
    fill = symmetry.plan_mirror_ghost_fill(driver.fields, "D")
    constitutive = symmetry.plan_folded_constitutive(driver.fields, driver.pml, "E")
    missing = [name for name, plan in
               (("folded curl", curl), ("mirror ghost fill", fill),
                ("folded constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    # `zero_metal_D` and `fill_folded_far_ghosts_D` stay on the ARRAY PATH here: no
    # Triton product owns either. Both are counted, so a difference in which of them
    # ran is a counter event rather than an invisible correction.
    #
    # THE FAR FILL IS NOT A NO-OP ANY MORE, and the comment that said it was is
    # corrected here rather than left standing. It was true while every case folded
    # over a METALLIC outer declaration; the folded PERIODIC cases added on
    # 2026-08-21 make it write a real plane. That is the whole point of this route
    # now: the fused launch ABSORBS the pass (see `fused_route`) and must reproduce
    # what the array path leaves here, so the comparison between the two routes is
    # exactly the far carry's evidence.
    return Route({"step_D": curl, "fill_symmetry_bc_D": fill, "update_E": constitutive})


#: The ONLY predicate reason this leg is allowed to route around, matched on a
#: phrase the clause carries. It is the gate-pending far-carry refusal: the kernel
#: carries `fill_folded_far_ghosts_D`, and the predicate declines to admit a folded
#: PERIODIC grid UNTIL A DEVICE LEG HAS RUN ONE. This gate IS that leg, so the
#: refusal and the run are circular and one of them has to give.
#:
#: WHICH ONE GIVES IS THE WHOLE DESIGN. Retiring the clause first would let the
#: fusion matrix price two more seam-instances against bytes nothing had measured;
#: so instead the clause STANDS and this harness builds its plan through
#: `plan_folded_fused_pair_from_arrays`, the route that documents itself as
#: bypassing the predicate for exactly this purpose. The bytes launched are the
#: shipped kernel's either way — only the admission decision is taken here.
UNGATED_FAR_CARRY_REASON = "no device leg has executed a folded PERIODIC case"


def harness_plan_for(driver, product, kernel):
    """Build the plan for a folded PERIODIC case WITHOUT asking the predicate.

    REFUSES ON ANY OTHER REASON. If the predicate declines for something that is not
    the gate-pending far-carry clause, this raises — a harness route that swallowed
    a second reason would be a way to run a case the product genuinely does not
    cover, and the leg would then report identity for a configuration nothing
    claims.

    Every argument is read from the SAME driver objects `plan_folded_fused_pair`
    reads, through the same `SUB_STEPS` / `CONSTITUTIVE_SIDES` specs, so the plan
    this returns differs from the predicate route's only in how it was admitted.
    """
    from meep_gpu.triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: PLC0415
    from meep_gpu.triton_kernels.launch import SUB_STEPS  # noqa: PLC0415
    from meep_gpu.triton_kernels import symmetry as _symmetry  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    verdict = product.folded_fused_pair_coverage(
        fields, pml, tuple(driver._sources))
    other = [reason for reason in verdict.reasons
             if UNGATED_FAR_CARRY_REASON not in reason]
    if other:
        raise AssertionError(
            f"the harness route is only licensed to bypass the gate-pending "
            f"far-carry clause; this case is also refused for {other}")
    if not any(UNGATED_FAR_CARRY_REASON in reason for reason in verdict.reasons):
        raise AssertionError(
            "the harness route was used on a case the predicate ADMITS; the "
            "predicate route must be used there, or the leg is not testing what "
            "the engine would build")

    curl_spec = SUB_STEPS[product.CURL_SUB_STEP]
    side_spec = CONSTITUTIVE_SIDES[product.CONSTITUTIVE_SIDE]
    codes, _ = _symmetry.folded_axis_kinds(fields.grid, pml)
    arrays = {}
    for name in curl_spec["targets"]:
        arrays[name] = getattr(fields, name)
        arrays["fu_" + name] = getattr(fields, "fu_" + name)
    for name in curl_spec["sources"]:
        arrays[name] = getattr(fields, name)
    for name in side_spec["targets"]:
        arrays[name] = getattr(fields, name)
        arrays["inv_eps_" + name] = fields.inverse_epsilon_for(name)
    for name in side_spec["aux"]:
        arrays[name] = getattr(fields, name)
    curl_flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}")
                 for axis in "xyz" for stem in ("kms", "sinv")}
    suffix = "_h" if side_spec["half_integer"] else ""
    constitutive_flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
                         for axis in "xyz" for stem in ("kps", "kms")}
    return product.plan_folded_fused_pair_from_arrays(
        arrays, curl_flat, constitutive_flat, codes,
        zero_metal_axes(fields.grid), product.mirror_phases(fields.grid),
        fields.grid.dt / fields.grid.dx,
        kernel=kernel, num_warps=1,
        reflect=_symmetry._far_reflect_rows(fields.grid) or (None, None, None))


def fused_route(plan):
    return Route({
        "step_D": plan,
        "fill_symmetry_bc_D": _Absorbed("fill_symmetry_bc_D", plan),
        "zero_metal_D": _Absorbed("zero_metal_D", plan),
        "fill_folded_far_ghosts_D": _Absorbed("fill_folded_far_ghosts_D", plan),
        "update_E": _Absorbed("update_E", plan),
    })


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, mirrors, steps: int, product,
            mutant: Any = None, install_fused: bool = True,
            freeze_electric: bool = False, magnetic: bool = True) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, cell, boundaries, mirrors, SEED, magnetic)
    separate = build_driver(cp, cell, boundaries, mirrors, SEED, magnetic)
    fused = build_driver(cp, cell, boundaries, mirrors, SEED, magnetic)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_fused_curl_constitutive_D_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "mirrors": [list(entry) for entry in mirrors],
        "magnetic_source": bool(magnetic),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_folded_fused_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.folded_fused_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            if not any(UNGATED_FAR_CARRY_REASON in reason
                       for reason in verdict.reasons):
                raise AssertionError(
                    f"the product refused the case: {verdict.reasons}")
            # The gate-pending far-carry clause, and only it. See
            # `harness_plan_for`: the clause stands so nothing can price it, and
            # this run is the measurement that would license retiring it.
            plan = harness_plan_for(fused, product, kernel)
            row["routed_through_the_harness_builder"] = True
            row["predicate_reasons_bypassed"] = list(verdict.reasons)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        route = fused_route(plan) if install_fused else Route({})
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
            })
            divergence = versus_array or versus_separate
            log(f"  {name} step {step}/{steps} identical={divergence is None} "
                f"control={control is None} moved={len(step_moved)} "
                f"launches={kernel.calls} ({time.time() - started:.1f} s)")
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
               require_moved: bool = True) -> Tuple[bool, List[str]]:
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
        inspect.getsource(product.folded_fused_curl_constitutive_D.fn))


def shipped_device_function_source(product) -> str:
    """``_carry_ghost_E``'s source — the kernel CALLS it, so a mutant needs it too.

    It became a device function when the far carry landed (2026-08-21). A mutant
    module that carried only the kernel body compiled the call against nothing and
    died with ``NameError('_carry_ghost_E is not defined')`` — the same class of
    failure as the missing ``MIRROR_PERIODIC`` constant below, and found the same
    way: on the device, in the mutation leg, after every case had already passed.
    """
    return textwrap.dedent(inspect.getsource(product._carry_ghost_E.fn))


def compile_mutant(source: str, kernel_name: str, product: Any = None) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest.

    THE HEADER'S CONSTANTS ARE READ FROM THE SHIPPED MODULE, not spelled here. They
    used to be literals (``PERIODIC = tl.constexpr(0)``), and a constant the kernel
    began branching on was then simply ABSENT from the mutant's namespace: the far
    carry brought ``MIRROR_PERIODIC`` into the body, every device case passed with
    the shipped kernel, and the first mutation died at compile time. Deriving them
    means a constant the kernel starts using cannot go missing.
    """
    from meep_gpu.triton_kernels import symmetry as _symmetry  # noqa: PLC0415

    constants = "".join(
        f"{name} = tl.constexpr({getattr(_symmetry, 'CODE_' + name)})\n"
        for name in ("PERIODIC", "METALLIC", "MIRROR_METALLIC", "MIRROR_PERIODIC"))
    header = "import triton\nimport triton.language as tl\n" + constants + "\n"
    body = source
    if product is not None:
        # THE DEVICE FUNCTION IS RENAMED WITH THE KERNEL. Two mutants that shared a
        # `_carry_ghost_E` identity would be two callers of one cached device body,
        # which is the hazard the kernel rename exists to close, one level down.
        ghost = f"_carry_ghost_E_{kernel_name}"
        body = (shipped_device_function_source(product).replace(
                    "_carry_ghost_E", ghost)
                + "\n\n" + source.replace("_carry_ghost_E", ghost))
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_folded_electric_pair.py", delete=False, encoding="utf-8")
    handle.write(header + body.replace("folded_fused_curl_constitutive_D",
                                       kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_electric_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was MEASURED."""

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
        """The NEAR fill not carried: update_E consumes a pre-fill D.

        Both halves of the carry are undone together — the ownership masks go back
        to ``live`` and every near source lane goes dead — so the mutant is exactly
        ``step_D`` + ``zero_metal_D`` + ``update_E`` with the mirror fill missing,
        rather than a half-carried state no configuration corresponds to. The FAR
        lanes are left alone; ``m14`` is their own mutation."""
        hits = 0
        for own, coordinate in (("own0", "j"), ("own2", "j"), ("own0", "k"),
                                ("own1", "k"), ("own1", "i"), ("own2", "i")):
            needle = f"{own} = {own} & ({coordinate} != 0)"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} = {own}")
        for lane, coordinate in (("near_i", "i"), ("near_j", "j"),
                                 ("near_k", "k")):
            needle = f"{lane} = live & ({coordinate} == 2)"
            hits += source.count(needle)
            source = source.replace(needle, f"{lane} = live & (nx < 0)")
        return source, hits

    def m3_corner_carry_dropped(source: str) -> Tuple[str, int]:
        """The doubly-unowned NEAR corner left unwritten.

        Scored ONLY on a case with two folded axes: on a single fold every corner
        block's guard is compile-time false and the rewrite would be a DEAD BRANCH
        reporting a defect it never executed. The needle is the corner call's mask
        term, which the far/near composites do not share."""
        hits = 0
        for own, first, second in (("own0", "near_j", "near_k"),
                                   ("own1", "near_i", "near_k"),
                                   ("own2", "near_i", "near_j")):
            needle = f"{own} & {first} & {second})"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} & (nx < 0))")
        return source, hits

    def m4_corner_weight_is_one_axis_twice(source: str) -> Tuple[str, int]:
        """The corner imaged with the WRONG parity product.

        ``PHY * (PHX * v2)`` becomes ``PHX * (PHX * v2)`` — the same value whenever
        the two planes share a parity, which is why it is scored on the two-fold
        case whose planes are EVEN and ODD."""
        needle = "gv_2xy = PHY * (PHX * v2)"
        return (source.replace(needle, "gv_2xy = PHX * (PHX * v2)"),
                source.count(needle))

    def m5_source_index_off_by_one(source: str) -> Tuple[str, int]:
        """The ghost imaged from stored cell 1 rather than 2 (MEEP's io = -2).

        THE SOURCE LANE MOVES AND THE DESTINATION OFFSET DOES NOT, which is the
        defect: the lane at stored cell 1 writes the ghost that the lane at cell 2
        owns, so the destination lands at cell -1 of the axis. In FLAT indexing that
        is cell ``n-1`` of the previous plane rather than outside the allocation, so
        the leg measures a WRONG ANSWER over a whole plane instead of faulting —
        which is what makes it scoreable. ``near_i`` is deliberately left alone so
        one axis still images correctly and the mutant is not simply "no fill"."""
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
             "src0 = v0 * tl.load(ie0 + idx, mask=own0, other=0.0)",
             "tl.store(w0 + idx, src0, mask=own0)"],
            ["src0 = v0 * tl.load(ie0 + idx, mask=own0, other=0.0)",
             "tl.store(w0 + idx, src0, mask=own0)",
             "prev0 = tl.load(w0 + idx, mask=own0, other=0.0)"])

    def m8_zero_metal_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's step_boundaries(D_stuff), undone.

        NO ``if`` IN THE REPLACEMENT. ``_rewrite_block`` indents every replacement
        line to the indent of the block's FIRST line, so a compound statement in
        position 0 produces an empty suite and the mutated module fails to IMPORT —
        which the harness records as an unarmed mutation, not as a caught defect.
        Dropping the guard along with the clear is also the truer defect. The
        assignments are pure identity, with no arithmetic, so a signed zero survives
        them."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v1 = tl.where(at_x, 0.0, v1)",
             "v2 = tl.where(at_x, 0.0, v2)"],
            ["v1 = v1", "v2 = v2"])

    def m9_ghost_wall_clear_dropped(source: str) -> Tuple[str, int]:
        """The imaged ghost's OWN wall clear dropped — the parity/clear ORDER.

        The driver runs ``fill_symmetry_bc_D`` and then ``zero_metal_D``, so the
        destination's final value is ``mask(parity * v)``. Without the ghost's clear
        the kernel leaves ``parity * mask(v)``, which on an ODD plane is ``-0.0``
        where the array path leaves ``+0.0`` — a byte difference and nothing else.
        Scored on the odd Y fold, where Dz's carry crosses the X wall."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "gv_2y = tl.where(at_x, 0.0, gv_2y)"],
            ["gv_2y = gv_2y"])

    def m10_inverse_epsilon_at_the_source(source: str) -> Tuple[str, int]:
        """The ghost's material read at the SOURCE cell rather than its own.

        Invisible in a uniform medium, which is why :func:`build_driver` installs a
        varying epsilon rather than a constant one.

        RE-EXPRESSED 2026-08-21, through the POINTER rather than the index. The
        material moved into ``_carry_ghost_E``'s ``ie`` argument when the far carry
        landed, and the first re-expression — swapping in another COMPONENT's
        tensor — came back UNCAUGHT on the device for a measured reason:
        ``FdtdDriver.set_epsilon`` is documented isotropic and
        ``inverse_epsilon_for`` returns the SAME array for Ex, Ey and Ez, so the
        swap was a bitwise no-op. Offsetting the pointer instead reaches the cell
        the index used to name, and the varying epsilon makes it visible."""
        # `ie2 - dn_y` makes the load land at `(idx + dn_y) - dn_y` = `idx`, the
        # SOURCE lane's own cell. In bounds by construction — it is a cell this
        # lane already reads — where a forward shift could address outside the
        # allocation and be reported as an inert defect.
        needle = "_carry_ghost_E(f2, w2, e2, ie2, idx + dn_y, gv_2y,"
        return (source.replace(
            needle, "_carry_ghost_E(f2, w2, e2, ie2 - dn_y, idx + dn_y, gv_2y,"),
            source.count(needle))

    # ------------------------------------------------------------------
    # THE FAR CARRY'S OWN MUTATIONS (2026-08-21). Every one of these targets a
    # branch that arrived with `fill_folded_far_ghosts_D` and that NO DEVICE LEG
    # HAS EVER EXECUTED. They are scored only on a case that folds a PERIODIC
    # axis; on a folded METALLIC grid every `FAR_a` guard is compile-time false
    # and the rewrite is a DEAD BRANCH reporting a defect it never ran.
    # ------------------------------------------------------------------

    def m12_far_composite_is_a_raw_product(source: str) -> Tuple[str, int]:
        """A far-over-near cell imaged from ``v`` instead of the CLAMPED near ghost.

        This is the ``far_flat_product`` rule that
        ``measure_d_side_composition_law.py`` scored: it matches the array path on
        86 of 156 declarations and is caught on 70, so it is wrong exactly where a
        wall meets a fold and invisible everywhere else. Scored on a case that
        carries BOTH a folded PERIODIC axis and a wall."""
        needle = "idx + df_x + dn_y, -PHX * gv_0y"
        return (source.replace(needle, "idx + df_x + dn_y, -PHX * (PHY * v0)"),
                source.count(needle))

    def m13_far_takes_the_source_coefficient(source: str) -> Tuple[str, int]:
        """The far ghost accumulated with the SOURCE lane's kps/kms pair.

        THE D-SPECIFIC DEFECT. ``update_E`` indexes component m on axis m and the
        far fill images along exactly that axis, so the destination at the top row
        takes a DIFFERENT coefficient entry than its source at the reflect row.
        Reusing the source's is the transposed form of the mutation the magnetic
        twin carries for its near half, and it is invisible outside the absorber
        profile's gradient."""
        hits = 0
        for needle, replacement in (("kp_f0, km_f0, own0 & far_i)",
                                     "kp_0, km_0, own0 & far_i)"),):
            hits += source.count(needle)
            source = source.replace(needle, replacement)
        return source, hits

    def m14_far_carry_dropped(source: str) -> Tuple[str, int]:
        """The far fill not carried at all — the pre-2026-08-21 kernel's behaviour.

        Both halves are undone together: the top-plane ownership masks go back to
        ``live`` and every far source lane goes dead, so the mutant is the near-only
        carry rather than a half-carried state no configuration corresponds to. It
        is the mutation that says what the whole round bought."""
        hits = 0
        for own, coordinate, extent in (("own0", "i", "nx"), ("own1", "j", "ny"),
                                        ("own2", "k", "nz")):
            needle = f"{own} = {own} & ({coordinate} != {extent} - 1)"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} = {own}")
        for lane, coordinate, row in (("far_i", "i", "rx"), ("far_j", "j", "ry"),
                                      ("far_k", "k", "rz")):
            needle = f"{lane} = live & ({coordinate} == {row})"
            hits += source.count(needle)
            source = source.replace(needle, f"{lane} = live & (nx < 0)")
        return source, hits

    def m15_far_parity_sign(source: str) -> Tuple[str, int]:
        """The far ghost imaged with ``+phase`` — the NEAR fill's parity.

        ``fields.mirror_parity`` is ``phase * (1 - 2 * iyee)``, so a shift-0
        component takes ``+phase`` and a shift-1 one ``-phase``. The far fill only
        ever touches shift-1 components. Invisible on an EVEN plane by
        construction, so it is scored on an ODD one."""
        needle = "idx + df_x, -PHX * v0"
        return (source.replace(needle, "idx + df_x, PHX * v0"),
                source.count(needle))

    def m16_top_plane_mask_dropped(source: str) -> Tuple[str, int]:
        """``pml_curl_step_folded``'s top-plane mask, un-carried.

        It arrived with the far carry because the slot the fill writes sits past
        MEEP's owned window. Dropping it leaves the split-field auxiliary ``fu_D``
        stepped from an unmasked curl on that plane, which the fill does NOT
        overwrite — so the defect survives into the next timestep even though the
        displacement itself is imaged."""
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
        ``stored - 2`` at an EVEN full count and ``stored - 3`` at an ODD one. Baking
        ``n - 2`` reflects about the window top instead of about the second mirror,
        so it is exactly right on every even-count run and a whole cell wrong on
        every odd one. It must therefore be scored on an ODD full count and
        confirmed a NULL on an even one."""
        hits = 0
        for lane, coordinate, row, extent in (("far_i", "i", "rx", "nx"),
                                              ("far_j", "j", "ry", "ny"),
                                              ("far_k", "k", "rz", "nz")):
            needle = f"{lane} = live & ({coordinate} == {row})"
            hits += source.count(needle)
            source = source.replace(needle, f"{lane} = live & ({coordinate} == {extent} - 2)")
        for axis, row, extent in (("df_x", "rx", "nx"), ("df_y", "ry", "ny"),
                                  ("df_z", "rz", "nz")):
            stride = {"df_x": " * nyz", "df_y": " * nz", "df_z": ""}[axis]
            needle = f"{axis} = ({extent} - 1 - {row}){stride}"
            hits += source.count(needle)
            source = source.replace(needle, f"{axis} = 1{stride}")
        return source, hits

    def m11_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
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
        ("m12_far_composite_is_a_raw_product", "the CHAINED composite", "caught",
         m12_far_composite_is_a_raw_product),
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


#: Mutation id -> the CASES index whose grid carries the branch it rewrites.
#:
#: THE DEAD-BRANCH TRAP, priced per entry. Every carry block in this kernel sits
#: under ``BC? == MIRROR_METALLIC`` on a named axis, and every corner block under
#: two of them. A mutation scored on a grid that does not fold those axes rewrites
#: REAL lines the case never reaches and comes back "uncaught" while measuring
#: nothing. ``mutation_case_for`` REFUSES such a pairing rather than running it.
MUTATION_CASE: Dict[str, int] = {
    # PHY appears in the rewrite: needs a Y fold, and an ODD one to be visible.
    "m1_parity_dropped": 1,
    # rewrites the y and z and x fill masks: any fold reaches some of them, but the
    # two-fold case reaches the most and also exercises the corners.
    "m2_near_fill_dropped": 3,
    # corner blocks exist only where TWO axes are folded.
    "m3_corner_carry_dropped": 3,
    "m4_corner_weight_is_one_axis_twice": 3,
    # rewrites the `fill_2y` block: needs Y folded.
    "m5_source_index_off_by_one": 1,
    # rewrites the ghost's ZM_X clear inside the `fill_2y` block: needs Y folded AND
    # X walled, which `odd_y_fold_metallic` is and `y_fold_periodic_x` is not.
    "m9_ghost_wall_clear_dropped": 1,
    "m10_inverse_epsilon_at_the_source": 1,
    # ---- the far carry's own mutations, on the folded PERIODIC cases ----------
    # m12 needs a far fold AND a wall, or the chained composite and the raw product
    # agree and the rewrite is a NULL that reads as uncaught.
    # RETARGETED 2026-08-21 after the device run reported it UNCAUGHT. It was
    # scored on `x_fold_periodic_y_wall`, which does not fold Y at all, so the
    # `if FAR_X and NEAR_Y:` block it rewrites was compile-time absent — a DEAD
    # BRANCH mutation reporting a defect it never executed. It is now on the 3-D
    # case that folds X PERIODIC, folds Y, and carries a z wall, which is the only
    # shape where the chained composite and a raw product differ at all.
    "m12_far_composite_is_a_raw_product": 10,
    # m13 rewrites the far-only x block's coefficient pair: needs x folded PERIODIC.
    # RETARGETED 2026-08-21 after the device run reported it UNCAUGHT, and the
    # reason was MEASURED rather than guessed: on `x_fold_periodic_even` the
    # reflect row is nx-2 and the destination is nx-1, and the PML coefficient
    # arrays hold the SAME value at both (kps_x_h[18] == kps_x_h[19] exactly, and
    # kms likewise) because the not-owned ghost slot copies its neighbour. Taking
    # the source's pair was therefore a bitwise no-op — a coincidence of the
    # profile, not a property of the carry. At an ODD full count the reflect row is
    # nx-3 and the two differ (1.44425499 vs 1.73438072), so the mutation is live.
    "m13_far_takes_the_source_coefficient": 6,
    # m14 undoes every far lane and top-plane mask: the two-fold periodic case
    # reaches the most of them.
    "m14_far_carry_dropped": 7,
    # m15 drops the far parity's sign: invisible on an EVEN plane by construction,
    # so it is scored on the ODD one.
    "m15_far_parity_sign": 6,
    "m16_top_plane_mask_dropped": 7,
    # m17 bakes `n - 2`, which IS the reflect row at an even full count. Only the
    # ODD-count case can score it; scoring it on an even one would report a real
    # defect as uncaught while measuring a coincidence.
    "m17_far_reflect_row_is_n_minus_two": 6,
}

#: The default leg for a mutation that does not name one.
DEFAULT_MUTATION_CASE = 1

#: Per mutation: the axes its rewritten lines are guarded on. ``mutation_case_for``
#: asserts the scored case folds every one of them, so a rewrite cannot be scored on
#: a grid where its enclosing guard is compile-time false.
MUTATION_REQUIRES_FOLD: Dict[str, Tuple[str, ...]] = {
    "m1_parity_dropped": ("Y",),
    "m2_near_fill_dropped": ("X", "Y"),
    "m3_corner_carry_dropped": ("X", "Y"),
    "m4_corner_weight_is_one_axis_twice": ("X", "Y"),
    "m5_source_index_off_by_one": ("Y",),
    "m9_ghost_wall_clear_dropped": ("Y",),
    "m10_inverse_epsilon_at_the_source": ("Y",),
    "m12_far_composite_is_a_raw_product": ("X", "Y"),
    "m13_far_takes_the_source_coefficient": ("X",),
    "m14_far_carry_dropped": ("X", "Y"),
    "m15_far_parity_sign": ("X",),
    "m16_top_plane_mask_dropped": ("X", "Y"),
    "m17_far_reflect_row_is_n_minus_two": ("X",),
}

#: Per mutation: the axes whose fold must be PERIODIC, not merely folded.
#:
#: THE SECOND HALF OF THE DEAD-BRANCH TRAP, and the half the pre-2026-08-21 guard
#: did not have. Every `FAR_a` block sits under a constexpr that is true only on
#: ``CODE_MIRROR_PERIODIC``; a mutation scored on a folded METALLIC grid rewrites
#: REAL lines the case never reaches and comes back "uncaught" while measuring
#: nothing. Folding the axis is not enough — the OUTER DECLARATION decides.
MUTATION_REQUIRES_PERIODIC_FOLD: Dict[str, Tuple[str, ...]] = {
    "m12_far_composite_is_a_raw_product": ("X",),
    "m13_far_takes_the_source_coefficient": ("X",),
    "m14_far_carry_dropped": ("X", "Y"),
    "m15_far_parity_sign": ("X",),
    "m16_top_plane_mask_dropped": ("X", "Y"),
    "m17_far_reflect_row_is_n_minus_two": ("X",),
}

#: Per mutation: axes that must carry a WALL. ``m9`` and ``m12`` both rewrite a
#: clear, and a case without one scores them as nulls.
MUTATION_REQUIRES_WALL: Dict[str, Tuple[str, ...]] = {
    "m9_ghost_wall_clear_dropped": ("X",),
    # Dx clears on a y wall and a z wall. `m12` rewrites the FAR_X-over-NEAR_Y
    # composite, whose value is `gv_0y`; NEAR_Y needs y FOLDED, and a folded axis is
    # never walled, so the ONLY axis that can clamp `gv_0y` is z. In 2-D z is
    # invariant and cannot carry a wall at all, which is why this mutation is a
    # structural null on every 2-D case and needs the 3-D one.
    "m12_far_composite_is_a_raw_product": ("Z",),
}

#: Per mutation: the full count of a named axis must be ODD. ``m17`` bakes a row
#: that is CORRECT at an even count, so an even case would report it uncaught.
MUTATION_REQUIRES_ODD_FULL_COUNT: Dict[str, Tuple[str, ...]] = {
    "m17_far_reflect_row_is_n_minus_two": ("X",),
}

#: Per mutation: axes whose mirror plane must be ODD (phase -1).
#:
#: An even plane multiplies by +1, so a dropped or duplicated parity is invisible
#: there by construction. For `m12` the requirement is sharper than that and was
#: MEASURED rather than assumed: the kernel's `v` register is already wall-clamped
#: when the carry reads it, so the near ghost's re-clamp can only change the sign
#: bit of a zero — `PHY * (+0.0)` is `+0.0` on an even plane and `-0.0` on an odd
#: one. On an even plane the chained composite and the raw product are BITWISE
#: EQUAL and the mutation measures nothing.
MUTATION_REQUIRES_ODD_PARITY: Dict[str, Tuple[str, ...]] = {
    "m1_parity_dropped": ("Y",),
    "m9_ghost_wall_clear_dropped": ("Y",),
    "m12_far_composite_is_a_raw_product": ("Y",),
    "m15_far_parity_sign": ("X",),
}

#: Per mutation: the axis whose FAR destination coefficient must differ from its
#: source's. Checked by BUILDING the grid, not declared.
MUTATION_REQUIRES_DISTINCT_FAR_COEFFICIENT: Dict[str, str] = {
    "m13_far_takes_the_source_coefficient": "X",
}


def _assert_the_far_coefficient_actually_moves(case_name: str, cell, boundaries,
                                               mirrors, axis_letter: str) -> None:
    """The far destination's kps/kms pair must not equal its source's.

    THE DEVICE RUN OF 2026-08-21 IS WHY THIS EXISTS. `m13` was scored on
    `x_fold_periodic_even` and came back UNCAUGHT; the reason was not a defect in
    the kernel but a coincidence of the absorber profile — at an even full count the
    reflect row is `nx - 2`, the destination is `nx - 1`, and the coefficient array
    holds the same value at both because the not-owned ghost slot copies its
    neighbour. `kps_x_h[18] == kps_x_h[19]` exactly, `kms` likewise. The mutation
    swapped one for the other and changed nothing.

    A DECLARATION WOULD NOT HAVE CAUGHT THAT, because it is a property of the
    absorber's grading and not of the case's fold set. So this builds the grid the
    case builds, reads the coefficients the kernel would read, and refuses.
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
                f"BITWISE EQUAL at the far source row {int(row)} and the "
                f"destination row {top} ({values[int(row)]!r}); swapping one for the "
                f"other is a no-op and the leg would report a real defect as "
                f"uncaught")


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    """The (index, case) a mutation is scored on, and it must really fold what it needs."""
    index = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    case = CASES[index]
    name_, cell, boundaries, mirrors, _steps = case
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

    # THE OUTER DECLARATION, not just the fold: a `FAR_a` block is compile-time
    # absent on a folded METALLIC axis.
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
                f"{declaration(axis_letter)!r} there, so the rewritten lines would "
                f"be a DEAD BRANCH")
    for axis_letter in MUTATION_REQUIRES_WALL.get(name, ()):
        if declaration(axis_letter) != "metallic" or axis_letter.upper() in folded:
            raise AssertionError(
                f"mutation {name} rewrites a wall clear on {axis_letter}, and case "
                f"{name_!r} declares {declaration(axis_letter)!r} there "
                f"(folded={axis_letter.upper() in folded}); stepping._zero_metal "
                f"skips a folded axis, so no clear would run and the rewrite would "
                f"be scored as a null")
    phases = {axis.upper(): int(phase) for axis, phase in mirrors}
    for axis_letter in MUTATION_REQUIRES_ODD_PARITY.get(name, ()):
        if phases.get(axis_letter.upper()) != -1:
            raise AssertionError(
                f"mutation {name} is only visible on an ODD {axis_letter} plane "
                f"(the multiply is by +1 on an even one, and for a re-clamp it is "
                f"the sign bit of a zero); case {name_!r} declares phase "
                f"{phases.get(axis_letter.upper())!r} there")

    # THE FAR COEFFICIENT PRECONDITION, MEASURED ON THE GRID THE CASE BUILDS.
    # `m13` swaps the destination's kps/kms pair for the source's, and on an EVEN
    # full count those two entries are BITWISE EQUAL — the not-owned ghost slot
    # copies its neighbour — so the rewrite is a no-op and the leg reports a real
    # defect as uncaught. This is not declared, it is checked: the grid is built and
    # the two entries compared.
    if name in MUTATION_REQUIRES_DISTINCT_FAR_COEFFICIENT:
        _assert_the_far_coefficient_actually_moves(name_, cell, boundaries, mirrors,
                                                  MUTATION_REQUIRES_DISTINCT_FAR_COEFFICIENT[name])
    for axis_letter in MUTATION_REQUIRES_ODD_FULL_COUNT.get(name, ()):
        axis = "XYZ".index(axis_letter.upper())
        full = int(round(12.0 * float(cell[axis])))
        if full % 2 == 0:
            raise AssertionError(
                f"mutation {name} bakes a reflect row that is CORRECT at an even "
                f"full count, and case {name_!r} has full count {full} on "
                f"{axis_letter}; scoring it there would report a real defect as "
                f"uncaught")
    return index, case


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
        kernel_name = f"mutant_{index}_folded_fused_D"
        mutant = compile_mutant(mutated, kernel_name, product)
        case_index, case = mutation_case_for(name)
        row["case"] = case[0]
        row["case_index"] = case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3],
                      MUTATION_STEPS, product, mutant=mutant)
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
    class _Electric:
        field_type = "D"

    def _deposit(fields, component):
        """A REAL source that publishes the index the injection writes.

        The stub classes in this leg carry a ``field_type`` and nothing else: enough
        to PLACE a source in a seam, and deliberately not enough to CARRY one. The
        carried rows below need a source the repair can actually save, and one that
        deposits nothing would make an admitting row pass by measuring an empty
        scatter, so both are checked here.
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

    # SAME PER-AXIS DECLARATION AS ``CASES``: these cells are 2-D, so a bare
    # "metallic" would put a PEC on the invariant z axis, which Grid refuses by name
    # (grid.py:842-877). This leg is about the product's own refusals; it must not
    # die inside the Grid constructor before reaching one.
    _WALLS = {"x": "metallic", "y": "metallic", "z": "periodic"}
    rows: List[Dict[str, Any]] = []
    # `sources` may be a CALLABLE taking the built fields, for the rows that need a
    # source bound to the driver's own grid. The literal tuples below are unchanged.
    for name, cell, boundaries, mirrors, needle, sources in (
        # THE 2026-08-30 CARRY, on the device's own objects. The refusing row now
        # names the fail-closed clause (a source that cannot publish its deposit
        # index), and the ADMITTING row is what stops "refuses electric" from
        # silently becoming "refuses everything electric" again.
        ("electric_source_without_a_deposit_index", CASES[0][1], _WALLS, (("Y", 1),),
         "does not publish the index it writes", (_Electric(),)),
        ("electric_deposit_is_carried", CASES[0][1], _WALLS, (("Y", 1),),
         None, lambda fields: (_deposit(fields, "Ez"),)),
        ("undeclared_sources", CASES[0][1], _WALLS, (("Y", 1),),
         "was not declared", None),
        # NOT A REFUSAL ANY MORE. `needle=None` means "this row must be ADMITTED",
        # and it is here because the retirement of the far-fill clause is the one
        # thing this round changed about what the engine will build. A row that only
        # ever checked refusals could not have noticed it.
        ("folded_periodic", CASES[0][1], "periodic", (("Y", 1),), None, ()),
        ("unfolded", (3.0, 3.0, 0.0), _WALLS, (), "no mirror plane", ()),
    ):
        driver = build_driver(cp, cell, boundaries, mirrors, SEED, magnetic=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.folded_fused_pair_coverage(
                driver.fields, driver.pml, bound)
            plan = product.plan_folded_fused_pair(driver.fields, driver.pml, bound)
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
        HERE, "results", "triton_folded_fused_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep' — the policy every record in "
             "triton_kernels/fingerprints.json is cut under.")
    args = parser.parse_args(argv)

    payload: Dict[str, Any] = {
        "gate": "triton_folded_fused_pair",
        "product": "meep_gpu.triton_kernels.folded_fused_pair",
        "kernel": "folded_fused_curl_constitutive_D",
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
    for leg in (transcription_leg, carry_table_leg, design_sweep_leg,
                predicate_leg, corpus_admission_leg):
        started = time.time()
        row = leg()
        row["seconds"] = round(time.time() - started, 3)
        payload["no_device_legs"].append(row)
        log(f"  {row['leg']}: passed={row['passed']} "
            f"findings={row.get('findings')} ({row['seconds']} s)")
        save(payload, args.out)

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
        save(payload, args.out)
        log(f"\nno-device verdict: {payload['passed']}  ->  {args.out}")
        return 0 if payload["passed"] else 1

    import cupy as cp  # noqa: PLC0415
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep and
    # quietly did not is a process whose bytes mean nothing, and this refuses at
    # startup rather than writing an artifact whose policy field is a wish.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import folded_fused_pair as product  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    payload["environment"] = environment(cp)
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "byte_cases": {case[0]: case[4] for case in CASES},
        "steps_per_mutation": MUTATION_STEPS,
        "total_steps": (sum(case[4] for case in CASES)
                        + MUTATION_STEPS * (len(mutation_table()) + 2)),
    }
    save(payload, args.out)
    log("\n=== device legs ===")
    for name, cell, boundaries, mirrors, steps in CASES:
        row = run_leg(cp, name, cell, boundaries, mirrors, steps, product)
        passed, failures = verdict_of(row, require_launches=steps)
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    log("\n=== armed harness mutations ===")
    # The substitution removed: the bytes still agree (the array path is what ran)
    # and ONLY the launch counter can refuse it.
    row = run_leg(cp, "armed:no_substitution", CASES[0][1], CASES[0][2], CASES[0][3],
                  3, product, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_electric_seam", CASES[0][1], CASES[0][2],
                  CASES[0][3], 3, product, freeze_electric=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.folded_fused_curl_constitutive_D)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, args.out)

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
