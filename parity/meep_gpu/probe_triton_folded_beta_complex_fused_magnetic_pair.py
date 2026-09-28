"""Byte gate for the FOLDED COMPLEX BETA fused MAGNETIC pair: beta ``step_B`` into ``update_H``.

DEVICE STATUS: **UNRUN.** This module is written to be run on a CUDA host with an
    idle device; nothing in this tree may cite it as a release until an artifact
    exists with ``device_status: RUN`` and ``release.released: true``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.folded_beta_complex_fused_magnetic_pair.folded_beta_complex_fused_magnetic_pair_coverage`
admits, ONE launch of ``folded_beta_complex_fused_curl_constitutive_B`` — bracketed by
the SHIPPED deposit repair wherever the seam carries a MAGNETIC deposit — leaves the
engine in a state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_B`` with its beta term / the magnetic
    injection / ``fill_symmetry_bc_B`` / ``zero_metal_B`` /
    ``fill_folded_far_ghosts_B`` / ``update_H``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the folded complex BETA
    curl K3b (``folded_complex.plan_folded_beta_bloch_pml_curl`` on ``step_B``), the
    complex mirror ghost fill (two passes with the wall clear between them) and the
    folded complex constitutive on side ``H``, with ``zero_metal_B`` on the array
    path in both routes,

over every allocated volume, on the uint32 view. ``allclose`` appears nowhere.

WHAT IS NEW AGAINST THE BETA-LESS TWIN'S RELEASED GATE
======================================================

The weld body is :mod:`.folded_complex_fused_magnetic_pair`'s released kernel plus
EXACTLY K3b's beta insert — the transcription leg proves that as an order-preserving
deletion equality, 432 statements against 432 — so the fold machinery (the parity
chain with the clear inside it, the far carry, the ownership restructure) inherits
that gate's whole fifteen-mutation set unchanged. The magnetic constitutive binds NO
inverse permeability, which is checked here as an ABSENCE in both bodies with the
electric twin's ``ie0`` load asserted present so the discrimination is not vacuous.
What only this gate adds:

1. **BETA CASES ONLY, ALL 2-D.** ``beta`` exists only on the effective-2-D grid
   (grid.py:668-697; MEEP fields.cpp:546-547), so the base gate's three 3-D cases
   cannot be built here and the z-axis fold/wall/phase guards are DELIBERATELY
   UNREACHABLE — declared, not discovered. Every case carries
   ``beta = BETA_CORPUS``.
2. **THE IDENTITY LEG.** ``HAS_BETA = 0`` must reproduce the beta-less shipped
   kernel byte for byte: the same seeded beta driver is stepped through this
   kernel with the insert compiled out and through
   ``plan_folded_beta_complex_fused_magnetic_pair_from_arrays`` (the beta-less product, whose
   engine-route predicate would refuse a beta grid), and the two must agree at
   every step while BOTH diverge from the beta-carrying array path — a vacuity
   guard, or the leg would pass on a case whose beta term never fired.
3. **THREE BETA MUTATIONS.** The insert dropped (must diverge), the two partners
   exchanged (must diverge), and the insert moved after the ownership masks (must
   diverge on a walled case, where the array path's mask zeroes the beta
   contribution the mutant leaves alive).
4. **THE DUAL EXTENDED LICENCE.** This kernel launches six operand orientations,
   so BOTH ``parity_expansion_license`` and ``folded_beta_expansion_license`` must
   answer, agree, and be recorded; a record licensing one and not the other
   refuses the run before the first device leg.

WHAT THE GATE REFUSES TO INFER — unchanged from the base gate: bytes alone cannot
prove the fused path ran (launches are counted); a no-op agreeing with a no-op is
trivially identical (the moved-state census); a bracket that changes nothing is
not load-bearing (the null controls); a mutation scored on a grid that never
enters its branch measures nothing (:func:`mutation_case_for`); a value class the
operands cannot contain is a leg that measured nothing (:data:`CLASS_FLOOR`).

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_beta_complex_fused_magnetic_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_beta_complex_fused_magnetic_pair.py \\
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
import itertools
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

SEED_ROOT = "triton_folded_beta_complex_fused_magnetic_pair/2026-09-02"

#: Temporary mutant modules, deleted at the end of the run.
_TEMPORARY: List[str] = []


def case_seed(name: str, value_class: str = "uniform") -> int:
    """A stable 63-bit seed for one (case, value class).

    A sha256 of the names rather than a process-salted draw: a gate whose seeds moved
    between runs could not be re-run against the same operands after a kernel edit.
    """
    digest = hashlib.sha256(f"{SEED_ROOT}/{name}/{value_class}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


#: The value classes every device case is run under. ``uniform`` provably draws no
#: subnormal and no signed zero, so a gate carrying only it says nothing about the
#: float32 subnormal policy at all; ``subnormal_band`` puts the operands, their
#: products with the absorber coefficients and the whole recurrence in and around the
#: band; ``zero_lattice`` is the +-0 product — every cell one of the four
#: ``(+-0, +-0)`` complex pairs — which is where the parity multiply's zero cross
#: terms decide bytes.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "zero_lattice")

#: Every case's beta — test_eigsrc_kz_0_complex's own value, the corpus
#: family's smallest, and any nonzero value exercises the same insert.
#: The two binary-grating rows carry -0.685 and -0.912; the SIGN rides in
#: the host-rounded coefficient words either way.
BETA_CORPUS = 0.2

#: The values a policy decides the fate of and that ``uniform(-1, 1)`` provably never
#: draws. Transcribed from ``probe_fused_kernel_bit_identity.SUBNORMAL_NEEDLES`` and
#: pinned equal to it by the host suite's sibling on the magnetic twin.
SUBNORMAL_NEEDLES = np.array(
    [0.0, -0.0, 1.1754944e-38, -1.1754944e-38, 1.1754942e-38, 1e-45, -1e-45],
    dtype=np.float32)

#: ``(name, dimensions, cell, boundaries, mirrors, k_point, pml, steps)``. A 2-D case
#: is spelled with ``z = 0.0``.
#:
#: THE Z AXIS IS PERIODIC IN EVERY 2-D CASE, NOT METALLIC. These cells are 2-D, so z
#: is translationally invariant: MEEP does not loop over it and it carries no boundary
#: condition. A metallic declaration there is a polarization filter, not a wall, and
#: ``Grid`` refuses it by name (grid.py:842-877).
#:
#: THE WALL CASES ARE NOT DECORATION ON THIS FAMILY. ``zero_metal_B`` writes SIX rows
#: against the magnetic twin's three, and on the D side the clear's rows are the near
#: fill's axes — so a folded axis beside a wall on a DIFFERENT axis is the only shape
#: where the parity-then-clear order inside a near ghost can be wrong.
CASES: Tuple[Tuple[str, int, Tuple[float, float, float], Any,
                   Tuple[Tuple[str, int], ...], Tuple[float, float, float],
                   Dict[str, Any], int], ...] = (
    # ONE folded axis, MIRROR_METALLIC: the near fill only, no far ghost, no
    # top-plane mask. The simplest admitted shape, and it carries a live x wall.
    ("y_fold_metallic", 2, (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # ONE folded axis, MIRROR_PERIODIC at an EVEN full count: the far carry, the
    # reflect row and the top-plane mask all live. This is the corpus class
    # (``special_kz_2_21_2`` / ``triangular_lattice_oblique``).
    ("y_fold_periodic", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # An ODD full count on the folded axis at the SAME stored extent: the reflect row
    # is ``stored - 3`` here and ``stored - 2`` above, which is what makes
    # ``m_reflect_row_baked`` a real discrimination rather than a shape change.
    ("y_fold_periodic_odd", 2, (3.2, 2.9, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # ODD MIRROR PHASE: the parity words are (-1, +0) near and (+1, +0) far. An EVEN
    # plane multiplies by +1, so a dropped parity is invisible there by construction.
    ("y_fold_periodic_odd_phase", 2, (3.2, 3.0, 0.0), "periodic", (("Y", -1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    # A FOLD ON X WITH AN UNFOLDED PERIODIC Y — the ghost rule's wrap branch on an
    # axis nothing folds, which every Y-folding case leaves uncompiled.
    ("x_fold_periodic_open_y", 2, (3.0, 3.2, 0.0), "periodic", (("X", 1),),
     (0.0, 0.0, 0.0), {"x": {"high": 5}, "y": 5}, 10),
    # TWO folded axes at MIXED phases — the doubly-unowned NEAR corner, which on THIS
    # family is a component's two near axes, and the only shape on which the near
    # chain's order question exists at all.
    ("xy_mixed_phase_periodic", 2, (3.0, 3.0, 0.0), "periodic",
     (("X", 1), ("Y", -1)), (0.0, 0.0, 0.0),
     {"x": {"high": 5}, "y": {"high": 5}}, 10),
    # WALLS, one per axis, each beside a fold on a DIFFERENT axis: `_zero_metal`
    # skips a folded axis, so a wall and a fold never share one. ON THIS FAMILY these
    # are the cases where a NEAR ghost takes a clear line: a component folded on y is
    # still cleared on an x wall.
    ("y_fold_periodic_wall_x", 2, (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "periodic", "z": "periodic"}, (("Y", -1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    ("x_fold_periodic_wall_y", 2, (3.0, 3.2, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("X", -1),),
     (0.0, 0.0, 0.0), {"x": {"high": 5}, "y": 5}, 10),
    # BLOCH PHASES, one per axis, each on an UNFOLDED periodic axis. A folded axis
    # may carry none (`driver._require_bloch_is_representable` refuses that
    # configuration outright), so the phase blocks are reachable only like this.
    ("y_fold_periodic_bloch_x", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.19, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    ("x_fold_periodic_bloch_y", 2, (3.0, 3.2, 0.0), "periodic", (("X", 1),),
     (0.0, 0.23, 0.0), {"x": {"high": 5}, "y": 5}, 10),
    # LONG HORIZON: ``y_fold_periodic``'s shape for SIXTY complete steps. One launch
    # and ten launches are not the same evidence as sixty — a one-ULP drift in the
    # ghost plane compounds, and this is where it shows.
    ("y_fold_periodic_long_horizon", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 60),
    # ONE LAUNCH. The other end of the same axis: exactly one complete step, so a
    # divergence here is attributable to a single launch rather than an accumulation.
    ("y_fold_periodic_single_launch", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 1),
)

#: One row of :data:`CASES` by name — mutations name the SHAPE they need rather than
#: an index, so inserting a case cannot silently re-score every mutation.
CASES_BY_NAME: Dict[str, Tuple[Any, ...]] = {case[0]: case for case in CASES}

#: The CARRY family: the same grids run with a real MAGNETIC deposit IN THIS SEAM.
#:
#: THIS IS THE PRODUCT ON ONE ROW OF THREE, and the honest arithmetic is worth
#: stating rather than inheriting. Only ``tests:TestSpecialKz.test_eigsrc_kz_0_complex``
#: declares in-seam ``'B'`` sources on this cell — the two binary gratings hold
#: none — so CARRIES_DEPOSIT_REPAIR buys one seam-instance here where it buys the
#: electric twin's whole cell. It is still the PRODUCT and not an extra: a gate
#: that ran only the quiet family would certify a kernel one of its three corpus
#: rows never reaches. One folded METALLIC grid (near fill only), one folded
#: PERIODIC grid at an ODD phase (whose far carry images the deposit onto a second
#: plane), one grid with a WALL beside the fold (where the clear sits inside the
#: chain) and the two-folded corner (the deepest composition beta can reach).
CARRY_CASES: Tuple[str, ...] = (
    "y_fold_metallic", "y_fold_periodic_odd_phase", "y_fold_periodic_wall_x",
    # The deepest composition BETA can reach is the two-folded 2-D corner
    # (beta refuses a third dimension outright), so the base gate's 3-D
    # carry case is replaced by the mixed-phase two-fold one.
    "xy_mixed_phase_periodic",
)

#: Steps per carry / null-control leg.
CARRY_STEPS = 6

#: Steps per armed mutation. A mutation needing more than this to become byte-visible
#: is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3292-3304).
SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

#: THERE ARE TWO VOCABULARIES FOR THIS SEAM AND THEY ARE NOT THE SAME LIST. The
#: driver binds ``fill_symmetry_bc_B``; the RESIDENCY model spells that slot
#: ``fill_D``, and ``REPLACES`` is declared in the residency spelling because that is
#: what a composer reads. Mapping them here rather than asserting them equal is the
#: honest form: :func:`seam_binding_leg` checks the map is a bijection and that each
#: driver name is a real ``driver.py`` call.
RESIDENCY_NAME: Dict[str, str] = {
    "step_B": "step_B",
    "fill_symmetry_bc_B": "fill_B",
    "zero_metal_B": "zero_metal_B",
    "fill_folded_far_ghosts_B": "fill_folded_far_ghosts_B",
    "update_H": "update_H",
}

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this tuple
#: is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: The volumes THIS SEAM writes. A leg in which none of them moves measured nothing
#: about this launch, whatever else moved.
#:
#: THEY ARE THE B/H VOLUMES, and getting that wrong would be silent rather than
#: loud. The other half of the same ``driver.step()`` writes ``D``/``E`` on every
#: route, so a census listing the ELECTRIC volumes would see movement on every leg
#: and pass while never once asking whether this launch's own outputs moved.
SEAM_OUTPUTS: Tuple[str, ...] = (
    "Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
    "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: The read-only material volumes, under the names ``inventory`` really finds. A name
#: that matches nothing is not a weaker check, it is two absent ones — the vacuity
#: floor would demand a read-only input move, and ``material_changed`` could never
#: fire. :func:`_assert_material_names_are_real` is what stops a rename doing that.
MATERIAL = ("eps", "inv_eps")

#: Private ``Fields`` scratch that is NOT physical state, and is out of the comparison
#: for that reason. ``_fmp_scratch`` is the buffer
#: ``Fields.displacement_minus_polarization`` allocates on first use; its presence
#: differs by ROUTE while carrying no state either route reads across a step. THE RULE
#: IS THE LEADING UNDERSCORE, not this list; the list is the tripwire for the rule.
PRIVATE_SCRATCH: Tuple[str, ...] = ("_fmp_scratch",)

#: Constexprs no admitted configuration can enter, skipped WHOLE by the branch
#: reachability leg. ``BACKWARD`` is 1 in every builder and its ``if`` arms are the
#: live half.
DELIBERATELY_UNREACHABLE: Tuple[str, ...] = ("BACKWARD",)


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
    """Every file this run binds itself to, KEYED BY ITS REPO-RELATIVE PATH.

    Not by a nickname: every reader of this map — ``build_triton_fusion_matrix``'s
    ``GATE_BOUND`` branch and the weld-record walk — resolves each key against the
    tree and re-hashes it, so a key that is not a path resolves to nothing and is
    reported as DRIFT on every cut.
    """
    names = (
        "meep_gpu/triton_kernels/folded_beta_complex_fused_magnetic_pair.py",
        # THE BASE WELD AND K3b'S HOME STAY PINNED: the transcription
        # leg diffs this kernel against both shipped bodies.
        "meep_gpu/triton_kernels/folded_complex_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/folded_complex.py",
        "meep_gpu/triton_kernels/complex_fields.py",
        "meep_gpu/triton_kernels/complex_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/special_kz.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/deposit_repair.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_folded_beta_complex_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Reading the shipped kernel text
# ---------------------------------------------------------------------------

_MODULE_FILE = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                            "folded_beta_complex_fused_magnetic_pair.py")

KERNEL_NAME = "folded_beta_complex_fused_curl_constitutive_B"
CARRY_HELPER = "_carry_ghost_complex"

#: The seven statements the K3b insert adds — the ONLY thing that may
#: separate this kernel from the beta-less shipped weld.
BETA_INSERT_LINES = (
    "if HAS_BETA:",
    "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
    "curl0_re = curl0_re - t_re",
    "curl0_im = curl0_im - t_im",
    "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
    "curl1_re = curl1_re - t_re",
    "curl1_im = curl1_im - t_im",
)


def _shipped_text(name: str, module_file: Optional[str] = None) -> str:
    """One function's body as SOURCE TEXT, read from the file.

    From the FILE and not through ``inspect`` because the no-device legs must run on
    a host with no Triton at all, where the decorated object is ``None``.
    """
    path = module_file or _MODULE_FILE
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if ast.get_docstring(node) else node.body
            return "\n".join(
                segment for segment in
                (ast.get_source_segment(text, statement) for statement in body)
                if segment)
    raise AssertionError(f"{name} is not defined in {os.path.basename(path)}")


def _statements(text: str) -> List[str]:
    """Non-comment, non-blank source lines, stripped."""
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _rewrite_block(source: str, before: Sequence[str],
                   after: Sequence[str]) -> Tuple[str, int]:
    """Replace a multi-line fragment, matching on the STRIPPED statements.

    Indentation differs between the kernel body and a device function, so a literal
    replace would silently match nothing in one of them and the mutation would be
    reported UNARMED. Matching on the stripped text and re-indenting from the first
    matched line is what makes one rewrite serve both.
    """
    lines = source.splitlines()
    needle = [fragment.strip() for fragment in before]
    hits = 0
    index = 0
    while index <= len(lines) - len(needle):
        window = [line.strip() for line in lines[index:index + len(needle)]]
        if window == needle:
            indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
            # THE REPLACEMENT KEEPS ITS OWN RELATIVE INDENTATION and is re-prefixed
            # with the matched span's base. Stripping every line to the base instead
            # flattens a nested block, and `if ZM_X:` with its body pulled out to
            # the same column is a mutant that does not PARSE — reported as "armed"
            # by the hit count and then as "did not compile" by the device leg,
            # which is a mutation that measured nothing either way. Measured on this
            # gate 2026-08-31, on two of its twenty-two rewrites.
            lines[index:index + len(needle)] = [
                (indent + fragment) if fragment.strip() else fragment
                for fragment in after]
            hits += 1
            index += len(after)
        else:
            index += 1
    return "\n".join(lines), hits


def _replace_all(source: str, before: str, after: str) -> Tuple[str, int]:
    hits = source.count(before)
    return source.replace(before, after), hits


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription
# ---------------------------------------------------------------------------

#: Each region of the kernel, and the certified body it must diff clean against.
CURL_LINES = (
    "t0_re = ((c_y_re - c_re) + (b_re - b_z_re))",
    "t1_re = ((a_z_re - a_re) + (c_re - c_x_re))",
    "t2_re = ((b_x_re - b_re) + (a_re - a_y_re))",
    "curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)",
    "q_re, q_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)",
    "n0_re, n0_im = _mul_field_left(q_re, q_im, si_y, EXPANSION)",
    "v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)",
    "r_re = (r_re + n0_re) - p0_re",
)

CONSTITUTIVE_LINES = (
    "t_re, t_im = _mul_coefficient_left(kp_0, src_re, src_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(km_0, prev_re, prev_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(kp_1, src_re, src_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(km_1, prev_re, prev_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(kp_2, src_re, src_im, EXPANSION)",
    "t_re, t_im = _mul_coefficient_left(km_2, prev_re, prev_im, EXPANSION)",
)

#: THE ACCUMULATOR IS NAMED FOR THE SIDE, and the two spellings are checked
#: against each other rather than either being assumed. The certified UNFOLDED
#: complex magnetic pair writes its constitutive result into ``a_re``; this weld
#: writes it into ``acc_re``, because the folded body already holds the curl
#: register for the ghost carry below it. Same two statements, same order, one
#: destination rename — declared here so a reader is not left to discover that
#: :data:`CONSTITUTIVE_LINES` deliberately stops at the multiplies.
ACCUMULATOR_RENAME: Tuple[Tuple[str, str], ...] = (
    ("a_re = a_re + t_re", "acc_re = acc_re + t_re"),
    ("a_re = a_re - t_re", "acc_re = acc_re - t_re"),
)

#: The B-side lines this product takes from the REAL folded MAGNETIC pair rather
#: than from the complex one: the near/far lane predicates and the destination
#: offsets, which are the B geometry and are spelled identically in both.
#:
#: THE THREE ``top_*`` LINES OF THE ELECTRIC TWIN ARE NOT HERE, and their absence
#: is a fact about the seam rather than an omission. The top-plane mask exists on
#: the D side because ``fill_folded_far_ghosts_D`` writes the LAST stored slot of a
#: folded periodic axis for the components whose Yee shift on it is 1 — the D
#: family's shifts — so the electric weld must name that plane to keep the far
#: carry off it. The B family's shifts put the far plane elsewhere and the magnetic
#: fill needs no such mask; the released REAL folded magnetic pair spells no
#: ``top_x`` either, which is what makes this a transcription difference rather
#: than a dropped check (MEASURED here 2026-09-02: the three lines are in neither
#: the magnetic weld nor the magnetic body it transcribes).
B_GEOMETRY_LINES = (
    "near_i = live & (i == 2)",
    "near_j = live & (j == 2)",
    "near_k = live & (k == 2)",
    "far_i = live & (i == rx)",
    "far_j = live & (j == ry)",
    "far_k = live & (k == rz)",
    "dn_x = -2 * nyz",
    "dn_y = -2 * nz",
    "dn_z = -2",
    "df_x = (nx - 1 - rx) * nyz",
    "df_y = (ny - 1 - ry) * nz",
    "df_z = (nz - 1 - rz)",
)


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic region traced to the shipped body it came from.

    Not a re-derivation: each statement below is read out of the certified emitter in
    the tree and required to appear CHARACTER FOR CHARACTER here.
    """
    package = os.path.join(API_ROOT, "meep_gpu", "triton_kernels")
    findings: List[str] = []
    ours = _shipped_text(KERNEL_NAME)
    helper = _shipped_text(CARRY_HELPER)

    curl = _shipped_text("folded_bloch_pml_curl_step",
                         os.path.join(package, "folded_complex.py"))
    for line in CURL_LINES:
        if line not in curl:
            findings.append(f"the certified folded complex curl no longer spells "
                            f"{line!r}; the transcription cannot be checked")
        if line not in ours:
            findings.append(f"the curl half no longer spells {line!r}")

    constitutive = _shipped_text("complex_fused_curl_constitutive_B",
                                 os.path.join(package,
                                              "complex_fused_magnetic_pair.py"))
    for line in CONSTITUTIVE_LINES:
        if line not in constitutive:
            findings.append(f"the certified complex magnetic pair no longer spells "
                            f"{line!r}")
        if line not in ours:
            findings.append(f"the constitutive half no longer spells {line!r}")
    for certified_spelling, ours_spelling in ACCUMULATOR_RENAME:
        if certified_spelling not in constitutive:
            findings.append(
                f"the certified complex magnetic pair no longer accumulates as "
                f"{certified_spelling!r}, so the rename cannot be checked")
        if ours_spelling not in ours:
            findings.append(f"the constitutive half no longer accumulates as "
                            f"{ours_spelling!r}")

    # THE K3b INSERT, verbatim from the certified folded complex beta curl:
    # the four statements the beta term adds, present in BOTH bodies, and the
    # fused body minus the insert must be the BASE weld exactly (checked as an
    # order-preserving deletion equality below).
    beta_curl = _shipped_text("folded_beta_bloch_pml_curl_step",
                              os.path.join(package, "folded_complex.py"))
    for line in BETA_INSERT_LINES:
        if line not in beta_curl:
            findings.append(f"the certified K3b curl no longer spells {line!r}; "
                            f"the beta transcription cannot be checked")
        if line not in ours:
            findings.append(f"the beta insert no longer spells {line!r}")
    base_body = _statements(_shipped_text(
        "folded_complex_fused_curl_constitutive_B",
        os.path.join(package, "folded_complex_fused_magnetic_pair.py")))
    ours_minus_insert = [line for line in _statements(ours)
                         if line not in set(BETA_INSERT_LINES)]
    if ours_minus_insert != base_body:
        at = next((i for i, (a, b) in enumerate(
            zip(ours_minus_insert, base_body)) if a != b), "length")
        findings.append(
            f"this kernel minus the beta insert is NOT the beta-less shipped "
            f"weld; first difference at statement {at}")
    at = _statements(ours).index("if HAS_BETA:")
    lines = _statements(ours)
    if lines[at - 1] != ("curl2_re, curl2_im = _mul_coefficient_left("
                         "dtdx, t2_re, t2_im, EXPANSION)"):
        findings.append(f"the insert no longer sits after the dtdx curl: "
                        f"{lines[at - 1]!r}")
    if lines[at + 7] != "at_x, at_y, at_z = i == 0, j == 0, k == 0":
        findings.append(f"the insert no longer sits before the ownership "
                        f"masks: {lines[at + 7]!r}")

    real = _shipped_text("folded_fused_curl_constitutive_B",
                         os.path.join(package, "folded_fused_magnetic_pair.py"))
    for line in B_GEOMETRY_LINES:
        if line not in real:
            findings.append(f"the released REAL folded magnetic pair no longer "
                            f"spells {line!r}; the B geometry cannot be checked")
        if line not in ours:
            findings.append(f"the B geometry no longer spells {line!r}")
    # ...and the D-only top-plane mask must be in NEITHER, or the sentence above
    # this tuple has stopped being true and the check has quietly narrowed.
    for line in ("top_x = idx * 0 + (nx - 1)", "top_y = idx * 0 + (ny - 1)",
                 "top_z = idx * 0 + (nz - 1)"):
        if line in real or line in ours:
            findings.append(
                f"{line!r} is present on the MAGNETIC side; the top-plane mask is "
                f"the D family's, and a B weld that carries it is transcribing the "
                f"wrong fill")

    # THE ONE CURL EDIT: a destination lane never READS D.
    for component in range(3):
        certified = f"e_re = tl.load(f{component} + 2 * idx, mask=live, other=0.0)"
        mine = (f"e_re = tl.load(f{component} + 2 * idx, "
                f"mask=own{component}, other=0.0)")
        if certified not in curl:
            findings.append(f"the certified curl no longer spells {certified!r}")
        if mine not in ours:
            findings.append(f"the curl half no longer masks the D load: {mine!r}")
        if certified in ours:
            findings.append(
                f"component {component}'s D load is UNMASKED here; a destination "
                f"lane would read a word another lane writes")

    # THE GHOST CONSTITUTIVE repeats the owned cell's own sequence, in order.
    order = ("prev_re = tl.load(w + 2 * dst",
             "tl.store(w + 2 * dst, ghost_re",
             "acc_re = tl.load(h + 2 * dst",
             "_mul_coefficient_left(kp_d, ghost_re, ghost_im, EXPANSION)",
             "_mul_coefficient_left(km_d, prev_re, prev_im, EXPANSION)",
             "tl.store(h + 2 * dst, acc_re",
             "tl.store(f + 2 * dst, ghost_re")
    positions = []
    for fragment in order:
        if fragment not in helper:
            findings.append(f"the ghost constitutive no longer spells {fragment!r}")
            positions.append(-1)
        else:
            positions.append(helper.index(fragment))
    if positions != sorted(positions):
        findings.append(
            f"the ghost constitutive's statement order moved: {positions}; `prev` "
            f"must be read BEFORE the workspace store, and the flux store must "
            f"come last")

    # NO INVERSE PERMEABILITY, ANYWHERE — the electric twin's inv_eps checks
    # inverted into their magnetic form. ``update_H`` divides by nothing (B and H
    # differ by no volume in any configuration this family admits), so an ``ie``
    # pointer on this side would be an operand the array path never reads and the
    # byte comparison could not attribute. Checked as an ABSENCE in both bodies
    # rather than left unsaid, because "the check found nothing" and "there was
    # nothing to check" are the same green otherwise.
    for where, text in (("the ghost helper", helper), ("the fused body", ours)):
        for spelling in ("ie0", "ie1", "ie2", "ie + dst", "ie + 2 * dst"):
            if spelling in text:
                findings.append(
                    f"{where} names {spelling!r}: update_H binds no inverse "
                    f"permeability, so this weld would be scaling by a volume the "
                    f"array path never applies")
    # ...and the ELECTRIC twin must still carry one, or the discrimination above
    # is vacuous and would pass on a tree that lost inv_eps everywhere.
    in_seam = _shipped_text(
        "folded_beta_complex_fused_curl_constitutive_D",
        os.path.join(package, "folded_beta_complex_fused_pair.py"))
    if "tl.load(ie0 + idx, mask=own0, other=0.0)" not in in_seam:
        findings.append(
            "the ELECTRIC twin no longer reads ie0 at the complex cell index, so "
            "the absence asserted above discriminates nothing")

    # ONLY THE LICENSED MULTIPLY HELPERS ARE CALLED.
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as module  # noqa: PLC0415

    tree = ast.parse(open(_MODULE_FILE, encoding="utf-8").read())
    called = {getattr(node.func, "id", "") for node in ast.walk(tree)
              if isinstance(node, ast.Call)}
    multiplies = {name for name in called
                  if name.startswith("_mul") or name.startswith("_rotate")}
    unlicensed = multiplies - set(module.LICENSED_MULTIPLY_HELPERS)
    if unlicensed:
        findings.append(
            f"the kernel calls unlicensed complex helpers {sorted(unlicensed)}; each "
            f"is an operand orientation with no probe pattern behind it")

    return {"leg": "transcription", "device": False,
            "curl_lines": len(CURL_LINES),
            "constitutive_lines": len(CONSTITUTIVE_LINES),
            "b_geometry_lines": len(B_GEOMETRY_LINES),
            "multiply_helpers": sorted(multiplies),
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — the parity chain, read off the shipped body
# ---------------------------------------------------------------------------

COEFFICIENT_ARGUMENTS = {
    "n0r": (0, "near"), "n1r": (1, "near"), "n2r": (2, "near"),
    "d0r": (0, "far"), "d1r": (1, "far"), "d2r": (2, "far"),
}

DN = {0: "dn_x", 1: "dn_y", 2: "dn_z"}
DF = {0: "df_x", 1: "df_y", 2: "df_z"}
NEAR_LANE = {0: "near_i", 1: "near_j", 2: "near_k"}
FAR_LANE = {0: "far_i", 1: "far_j", 2: "far_k"}


def _kernel_node():
    tree = ast.parse(open(_MODULE_FILE, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == KERNEL_NAME:
            return node
    raise AssertionError(f"{KERNEL_NAME} is not defined")


def carry_blocks() -> List[Dict[str, Any]]:
    """Every emitted carry block, PARSED off the shipped body.

    ``chain`` is the ORDERED ``(axis, pass)`` list taken from the
    ``_mul_imag_coefficient_left`` calls' first argument — the kernel's own
    sequence. Nothing here computes a parity or a ghost value: a leg that
    re-implemented the kernel would mirror its defects instead of executing them.
    """
    out: List[Dict[str, Any]] = []
    for statement in _kernel_node().body:
        if not isinstance(statement, ast.If):
            continue
        calls = [node for node in ast.walk(statement)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", "") == CARRY_HELPER]
        if not calls:
            continue
        if len(calls) != 1:
            raise AssertionError(
                f"the block `if {ast.unparse(statement.test)}:` emits "
                f"{len(calls)} carries; this parser assumes one per block")
        chain: List[Tuple[int, str]] = []
        for inner in statement.body:
            if not isinstance(inner, ast.Assign):
                continue
            value = inner.value
            if (isinstance(value, ast.Call)
                    and getattr(value.func, "id", "") == "_mul_imag_coefficient_left"):
                chain.append(COEFFICIENT_ARGUMENTS[value.args[0].id])
        call = calls[0]
        out.append({
            "guard": ast.unparse(statement.test),
            "target": call.args[0].id,
            "destination": ast.unparse(call.args[3]),
            "coefficient": ast.unparse(call.args[6]),
            "mask": ast.unparse(call.args[8]),
            "chain": tuple(chain),
        })
    return out


def parity_chain_leg() -> Dict[str, Any]:
    """The chain each block applies, against the driver order the product declares.

    THE ORDER IS THE DRIVER'S. ``fill_symmetry_bc_B`` (:3285) runs to completion
    before ``fill_folded_far_ghosts_B`` (:3287), and the far pass applies its axes
    ASCENDING (``stepping._fill_folded_far_ghosts``:1518, :1528). This leg compares
    the SHIPPED sequence against
    :func:`~meep_gpu.triton_kernels.folded_beta_complex_fused_magnetic_pair.parity_chain`,
    which is a statement about which coefficient goes where and in what order — not
    a second copy of the arithmetic.
    """
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as module  # noqa: PLC0415

    findings: List[str] = []
    parameters = [argument.arg for argument in _kernel_node().args.args]
    for name in COEFFICIENT_ARGUMENTS:
        if name not in parameters:
            findings.append(
                f"the kernel takes no argument {name!r}; this leg's coefficient "
                f"table names the signature and would otherwise be vacuous")

    blocks = carry_blocks()
    rows: List[Dict[str, Any]] = []
    matched: set = set()
    for component in range(3):
        near = (component,)
        far = tuple(axis for axis in range(3) if axis != component)
        target = f"f{component}"
        for subset, carries_near in module.carried_destinations(near, far):
            expected = module.parity_chain(near, subset, carries_near)
            hits = [block for block in blocks
                    if block["target"] == target and block["chain"] == expected]
            rows.append({
                "component": component, "far_axes_at_top": list(subset),
                "carries_near": carries_near,
                "expected_chain": [list(entry) for entry in expected],
                "emitted_blocks": len(hits),
                "guard": hits[0]["guard"] if hits else None,
                "coefficient": hits[0]["coefficient"] if hits else None,
                "mask": hits[0]["mask"] if hits else None,
            })
            if len(hits) != 1:
                findings.append(
                    f"component {component}, far {subset}, near {carries_near}: "
                    f"{len(hits)} emitted blocks carry the driver-ordered chain "
                    f"{expected}")
                continue
            matched.add(id(hits[0]))
            block = hits[0]
            # THE COEFFICIENT INDEX MOVES ONLY FOR THE NEAR HALF.
            wanted = f"kp_d{component}" if carries_near else f"kp_{component}"
            if block["coefficient"] != wanted:
                findings.append(
                    f"component {component}, far {subset}, near {carries_near}: "
                    f"the destination takes {block['coefficient']!r}, expected "
                    f"{wanted!r} — update_H indexes component m on axis m and the "
                    f"NEAR fill images along that same axis")
            # EVERY CARRY MASK IS ANDED WITH THE OWNERSHIP MASK.
            if not block["mask"].startswith(f"own{component} &"):
                findings.append(
                    f"component {component}: the carry mask {block['mask']!r} is "
                    f"not ANDed with own{component}; a lane that is the source of "
                    f"one fill and the destination of the other would write a "
                    f"ghost from a `v` its own mask zeroed")
    for block in blocks:
        if id(block) not in matched:
            findings.append(
                f"the kernel emits a carry the enumeration does not name: "
                f"`if {block['guard']}:` -> {block['destination']}")

    # THE FOLDED SPELLING IS ABSENT.
    ours = _shipped_text(KERNEL_NAME)
    for left in COEFFICIENT_ARGUMENTS:
        for right in COEFFICIENT_ARGUMENTS:
            if f"{left} * {right}" in ours:
                findings.append(
                    f"the kernel multiplies two parity coefficients together "
                    f"({left} * {right}); folding a chain into one word moves "
                    f"bytes under complex storage")
    return {
        "leg": "parity_chain",
        "device": False,
        "emitted_blocks": len(blocks),
        "destinations_enumerated": len(rows),
        "rows": rows,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — branch reachability
# ---------------------------------------------------------------------------

def build_grid(case, prefer_gpu: bool = True):
    """The grid and absorber one CASES row names, with no field seeding.

    Split out of :func:`build_driver` so the LAPTOP can build the same geometry the
    device leg compiles against — a second spelling of the geometry would be a
    mirrored evaluator rather than a check.
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    (_name, dimensions, cell, boundaries, mirrors, k_point, pml_spec,
     _steps) = case
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35,  # non-power-of-two, deliberate
        boundaries=boundaries, k_point=k_point, beta=BETA_CORPUS,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    driver.setup_pml(dict(pml_spec))
    return driver


def ghost_destinations(near: Sequence[Any], far: Sequence[Any]) -> Tuple[int, int, int]:
    """Ghost cells ONE source lane owns, per component, for this constexpr set."""
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as module  # noqa: PLC0415

    out: List[int] = []
    for component in range(3):
        near_axes = tuple(axis for axis in module.NEAR_FILL_COMPONENTS[component]
                          if bool(near[axis]))
        far_axes = tuple(axis for axis in module.FAR_FILL_COMPONENTS[component]
                         if bool(far[axis]))
        out.append(len(module.carried_destinations(near_axes, far_axes)))
    return (out[0], out[1], out[2])


def case_constexprs(case) -> Dict[str, Any]:
    """The constexprs one CASES row compiles the kernel with, read on NumPy.

    Every value comes from the function the PLAN calls, and ``near``/``far`` are
    derived from ``bc`` exactly as the plan derives them, so this cannot disagree with
    what a device leg compiles.
    """
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as module  # noqa: PLC0415
    from meep_gpu.triton_kernels.complex_fields import (  # noqa: PLC0415
        _phase_arguments, bloch_phase_table,
    )
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels.folded_complex import (  # noqa: PLC0415
        CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
        folded_axis_kinds,
    )

    driver = build_grid(case, prefer_gpu=False)
    try:
        grid = driver.fields.grid
        codes, reasons = folded_axis_kinds(grid, driver.pml)
        if codes is None:
            raise AssertionError(f"{case[0]}: folded_axis_kinds refused: {reasons}")
        codes = tuple(int(value) for value in codes)
        walls = tuple(bool(value) for value in zero_metal_axes(grid))
        phased, _values = _phase_arguments(
            bloch_phase_table(grid, _boundary_kinds(grid, driver.pml)),
            backward=bool(module.BACKWARD))
        near = tuple(code in module.MIRROR_CODES for code in codes)
        far = tuple(code == CODE_MIRROR_PERIODIC for code in codes)
        return {
            "PERIODIC": int(CODE_PERIODIC),
            "MIRROR_METALLIC": int(CODE_MIRROR_METALLIC),
            "MIRROR_PERIODIC": int(CODE_MIRROR_PERIODIC),
            "BACKWARD": int(module.BACKWARD),
            "BCX": codes[0], "BCY": codes[1], "BCZ": codes[2],
            "NEAR_X": near[0], "NEAR_Y": near[1], "NEAR_Z": near[2],
            "FAR_X": far[0], "FAR_Y": far[1], "FAR_Z": far[2],
            "ZM_X": walls[0], "ZM_Y": walls[1], "ZM_Z": walls[2],
            "PHX": int(phased[0]), "PHY": int(phased[1]), "PHZ": int(phased[2]),
            "shape": tuple(int(value) for value in grid.shape),
            "parity": module.parity_coefficient_words(grid),
            "ghost_destinations": ghost_destinations(near, far),
        }
    finally:
        driver.close()


def _live_constexpr_guards(text: str) -> List[Tuple[int, str]]:
    """Every ``if <constexpr...>:`` in the kernel body, with its line index."""
    out: List[Tuple[int, str]] = []
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("if ") and stripped.endswith(":"):
            out.append((index, stripped[3:-1]))
    return out


def branch_reachability_leg() -> Dict[str, Any]:
    """SHIPPED CODE NO LEG EXECUTES, counted rather than argued.

    THE FINDING THIS LEG EXISTS FOR is the sibling folded gate's: on 2026-08-20 that
    kernel shipped three-axis blocks and NO DEVICE LEG EVER EXECUTED THEM, because
    every ``mirrors`` value in the artifact carried at most two folded axes. Byte
    identity over ten steps on ten cases says nothing about a branch none of the ten
    compiled.
    """
    findings: List[str] = []
    text = _shipped_text(KERNEL_NAME)
    guards = _live_constexpr_guards(text)
    names = sorted({name for name in
                    ("BCX", "BCY", "BCZ", "NEAR_X", "NEAR_Y", "NEAR_Z",
                     "FAR_X", "FAR_Y", "FAR_Z", "ZM_X", "ZM_Y", "ZM_Z",
                     "PHX", "PHY", "PHZ")
                    if any(name in guard for _index, guard in guards)})
    tables = {case[0]: case_constexprs(case) for case in CASES}
    entered: Dict[str, int] = {name: 0 for name in names}
    for table in tables.values():
        for name in names:
            if bool(table.get(name)):
                entered[name] += 1
    never = [name for name, count in entered.items() if count == 0]
    # BETA IS EFFECTIVE-2-D BY CONSTRUCTION (grid.py:668-697), so the z-axis
    # fold, wall and phase guards CANNOT be entered by any admitted case; the
    # blocks behind them are the base weld's, certified by ITS released gate
    # on 3-D grids. Declared here, and the declaration is CHECKED in both
    # directions: an extra never-entered guard is still a finding, and a
    # declared-unreachable guard that some case DOES enter is one too.
    #
    # ``BCZ`` IS IN THE LIST FOR A DIFFERENT REASON, and the difference is worth
    # a sentence because the two look identical in ``never`` and are not the same
    # claim. ``BCZ`` is not a boolean guard at all — it is the z axis's boundary
    # CODE, and ``PERIODIC`` is 0. Every case this gate can build is effective-2-D
    # with an unfolded periodic z, so ``bool(BCZ)`` is False on all of them while
    # the branch behind it (the PERIODIC arm) is the one every case COMPILES AND
    # EXECUTES. Reporting that as "shipped code no leg executes" would be exactly
    # backwards. What actually has to hold for the code-valued constexprs is the
    # check immediately below — every one of the three codes taken by some case's
    # axis — and it is measured there rather than inferred from truthiness here.
    # MEASURED on this gate's own case set, 2026-09-02: ``never`` reads
    # ['BCZ', 'FAR_Z', 'NEAR_Z', 'PHZ', 'ZM_Z'].
    expected_never = sorted(("BCZ", "NEAR_Z", "FAR_Z", "ZM_Z", "PHZ"))
    if sorted(never) != expected_never:
        findings.append(
            f"guards TRUE on no case: {sorted(never)}; this beta gate declares "
            f"exactly {expected_never} unreachable (effective-2-D, plus BCZ whose "
            f"PERIODIC code is 0), so the difference is shipped code no device leg "
            f"executes — or a declaration gone stale")
    # The four boundary codes: each must be taken by some case's BC.
    codes = {table[key] for table in tables.values() for key in ("BCX", "BCY", "BCZ")}
    for required in ("PERIODIC", "MIRROR_METALLIC", "MIRROR_PERIODIC"):
        value = next(iter(tables.values()))[required]
        if value not in codes:
            findings.append(f"no case declares a {required} axis; the curl half's "
                            f"branch for it is never compiled")
    depth = max(max(table["ghost_destinations"]) for table in tables.values())
    if depth != 3:
        findings.append(
            f"the deepest composition any case reaches is {depth} ghost cells per "
            f"source lane; on the effective-2-D grids beta admits, a D component "
            f"with both transverse axes live owns 3 (two nears and a corner, or a "
            f"near, a far and their composite), and the two-fold composite blocks "
            f"are otherwise never compiled")
    return {"leg": "branch_reachability", "device": False,
            "guards_scanned": names,
            "cases_entering_each_guard": entered,
            "deepest_composition": depth,
            "constexprs": {name: {key: (list(value) if isinstance(value, tuple)
                                        else value)
                                  for key, value in table.items()}
                           for name, table in tables.items()},
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — the seam binding
# ---------------------------------------------------------------------------

def seam_binding_leg() -> Dict[str, Any]:
    """``REPLACES`` against the driver's own call sites, in both vocabularies."""
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as module  # noqa: PLC0415

    findings: List[str] = []
    driver_text = open(os.path.join(API_ROOT, "meep_gpu", "driver.py"),
                       encoding="utf-8").read()
    for name in SEAM_PASSES:
        if f"stepping.{name}(" not in driver_text and f" {name}(" not in driver_text:
            findings.append(f"{name} is not a call site in driver.py")
    if sorted(RESIDENCY_NAME) != sorted(SEAM_PASSES):
        findings.append("RESIDENCY_NAME does not cover exactly SEAM_PASSES")
    if len(set(RESIDENCY_NAME.values())) != len(RESIDENCY_NAME):
        findings.append("RESIDENCY_NAME is not injective")
    declared = tuple(RESIDENCY_NAME[name] for name in SEAM_PASSES)
    if declared != tuple(module.REPLACES):
        findings.append(
            f"REPLACES {tuple(module.REPLACES)} is not the residency spelling of the "
            f"driver passes this gate substitutes: {declared}")
    return {"leg": "seam_binding", "device": False,
            "driver_passes": list(SEAM_PASSES),
            "residency_names": declared,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — the predicate, on real NumPy drivers
# ---------------------------------------------------------------------------

HOST_ARRAY_MODULE_REASON = "not cupy"


def _residual(verdict) -> List[str]:
    return [reason for reason in verdict.reasons
            if HOST_ARRAY_MODULE_REASON not in reason]


class _MagneticWithoutIndex:
    """A magnetic source that publishes no deposit index — the refusing direction.

    IT MUST BE MAGNETIC TO REFUSE ANYTHING HERE. On this seam the driver injects
    the ELECTRIC currents in the other half of the step, so an electric source —
    index or not — is quiet at this clause; a fixture that stayed electric would
    have made the refusing direction unmeasurable while still reading green
    against a `not any(...)` assertion. Measured 2026-09-02: with the electric
    fixture the shipped predicate returns NO reasons at all on every one of this
    gate's twelve cases.
    """

    field_type = "B"


class _Electric:
    """An ELECTRIC source, carried because it reaches the OTHER seam.

    Used only to assert the QUIET direction: this product's source clause must
    say nothing about it, at either setting of CARRIES_DEPOSIT_REPAIR.
    """

    field_type = "D"


def _probe_record():
    """A probe artifact carrying this product's EXTENDED pattern set.

    The laptop legs need a licence to ask the predicate anything; the DEVICE legs
    consume the real artifact ``complex_fields.load_expansion_probe`` resolves.
    """
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as module  # noqa: PLC0415

    from meep_gpu.test_triton_complex_fields import (  # noqa: PLC0415
        stamp_probe_record)

    # STAMPED, not bare. A pattern map with no resolved subnormal policy is the
    # pre-2026-08-15 shape every complex family refuses BY NAME: a verdict is only
    # readable against the policy the bytes were cut under. The DEVICE legs consume
    # the real artifact instead; this is the laptop legs' licence.
    return stamp_probe_record(
        {"backend": "cupy",
         "patterns": {name: "FMA_V1"
                      for name in module.PRODUCT_PROBE_PATTERNS}})


def predicate_leg() -> Dict[str, Any]:
    """Every case admitted, and the DISJOINTNESS with the two neighbours, measured.

    Two predicates admitting one slot leaves ``_select_slot`` unable to choose and the
    slot UNSELECTED — a silent coverage LOSS, not an error — so the disjointness is
    asserted in BOTH directions on the same grids rather than argued.
    """
    from meep_gpu.expansion_refusal import declaring_run_policy  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as module  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_beta_fused_magnetic_pair as unfolded  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_beta_fused_magnetic_pair as real  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as beta_less  # noqa: PLC0415

    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    probe = _probe_record()
    folded_complex_grid: Dict[str, Any] = {}
    with declaring_run_policy("keep"):
        for case in CASES:
            driver = build_grid(case, prefer_gpu=False)
            try:
                sources = (_Deposit(driver.fields, "Hz",
                                    deposit_center(case)),)
                verdict = module.folded_beta_complex_fused_magnetic_pair_coverage(
                    driver.fields, driver.pml, sources, probe=probe)
                residual = _residual(verdict)
                rows.append({"case": case[0], "residual_reasons": residual,
                             "shape": list(driver.shape)})
                if residual:
                    findings.append(
                        f"case {case[0]!r} is refused for {residual}; every case this "
                        f"gate scores must be one the shipped predicate admits")
                # THE REFUSING DIRECTION of the source clause, on the same grid: a
                # MAGNETIC source that cannot publish its deposit index. On this
                # seam that is the in-seam field type; see _MagneticWithoutIndex
                # for why an electric fixture would have measured nothing.
                blind = module.folded_beta_complex_fused_magnetic_pair_coverage(
                    driver.fields, driver.pml, (_MagneticWithoutIndex(),),
                    probe=probe)
                if not any("does not publish the index" in reason
                           for reason in _residual(blind)):
                    findings.append(
                        f"case {case[0]!r}: a magnetic source with no deposit index "
                        f"is not refused by name")
                # ...and the QUIET direction, which is what makes the clause a
                # partition rather than a blanket refusal: an ELECTRIC source is
                # injected in the other seam and this product says nothing about it.
                quiet = _residual(
                    module.folded_beta_complex_fused_magnetic_pair_coverage(
                        driver.fields, driver.pml, (_Electric(),), probe=probe))
                if quiet:
                    findings.append(
                        f"case {case[0]!r}: an ELECTRIC source is refused here "
                        f"({quiet}); the driver injects it between step_D and "
                        f"update_E, which is not this seam")
                # AN UNDECLARED SOURCE LIST IS A REFUSAL, never an assumed ().
                undeclared = module.folded_beta_complex_fused_magnetic_pair_coverage(
                    driver.fields, driver.pml, None, probe=probe)
                if not any("was not declared" in reason
                           for reason in _residual(undeclared)):
                    findings.append(
                        f"case {case[0]!r}: an undeclared source list is not refused")
            finally:
                driver.close()

        # DISJOINTNESS, both directions, on ONE folded grid.
        base = CASES_BY_NAME["y_fold_periodic"]
        driver = build_grid(base, prefer_gpu=False)
        try:
            sources = (_Deposit(driver.fields, "Hz", deposit_center(base)),)
            mine = module.folded_beta_complex_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources, probe=probe)
            theirs = unfolded.complex_beta_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources, probe=probe)
            real_verdict = real.folded_beta_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources)
            beta_less_verdict = beta_less.folded_complex_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources, probe=probe)
            folded_complex_grid = {
                "grid": "a FOLDED COMPLEX BETA grid",
                "this": _residual(mine),
                "unfolded_complex_beta": _residual(theirs),
                "real_folded_beta": _residual(real_verdict),
                "beta_less_folded_complex": _residual(beta_less_verdict)}
            if _residual(mine):
                findings.append(f"this product refuses its own grid: "
                                f"{folded_complex_grid}")
            if not _residual(theirs):
                findings.append(
                    "complex_beta_fused_magnetic_pair ADMITS a FOLDED complex beta "
                    "grid; both products would claim the slot and _select_slot "
                    "would leave it unselected")
            if not _residual(real_verdict):
                findings.append(
                    "folded_beta_fused_magnetic_pair ADMITS a COMPLEX-storage grid; "
                    "its parity is a compile-time sign and cannot be that "
                    "product's")
            if not _residual(beta_less_verdict):
                findings.append(
                    "folded_complex_fused_magnetic_pair ADMITS a beta grid; K1's clause set "
                    "requires beta = 0 and two admitters would leave the slot "
                    "unselected")
        finally:
            driver.close()

        # THE PLAN BUILDER RETURNS None ON A REFUSED CONFIGURATION, never raises.
        driver = build_grid(base, prefer_gpu=False)
        try:
            built = module.plan_folded_beta_complex_fused_magnetic_pair(
                driver.fields, driver.pml,
                (_Deposit(driver.fields, "Hz", deposit_center(base)),),
                probe=probe)
            if built is not None:
                findings.append(
                    "the plan builder returned a plan on a NumPy host; the predicate "
                    "refuses that array module and None is the only legal answer")
        except Exception as exc:  # noqa: BLE001
            findings.append(
                f"the plan builder RAISED on a refused configuration ({exc!r}); None "
                f"is the only refusal")
        finally:
            driver.close()

    return {"leg": "predicate", "device": False, "cases": rows,
            "disjointness": folded_complex_grid,
            "findings": findings, "passed": not findings}


def deposit_center(case) -> Tuple[float, float, float]:
    """A point OFF the mirror plane on every folded axis.

    A point ``Ez`` at the origin sits ON the plane, and on an ODD fold that is
    refused with a reason rather than stepped: ``Ez`` has parity -1 there and no
    extent along the folded axis, so the parity condition constrains the profile
    against itself and admits only zero current (from_meep.py:2686-2698). A quarter
    of the owned half-width puts it inside the stored half and clear of the absorber,
    which a folded axis carries on its HIGH face only. THE SAME OFFSET
    :func:`build_driver` uses, so the laptop legs and the device legs deposit at the
    same place.
    """
    center = [0.0, 0.0, 0.0]
    for axis_name, _phase in case[4]:
        axis = "XYZ".index(axis_name.upper())
        center[axis] = 0.25 * (case[2][axis] / 2.0)
    return (center[0], center[1], center[2])


def _Deposit(fields, component: str,  # noqa: N802 - a factory named like a class
             center: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """A REAL source that publishes the index the injection writes."""
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=fields.grid, component=component,
                          center=center, size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    if not source._n_source_points:
        raise AssertionError("the fixture source deposits nothing")
    if deposit_repair._deposit_index(source) is None:
        raise AssertionError("the fixture source publishes no deposit index")
    return source


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 6 — what the corpus says this is worth
# ---------------------------------------------------------------------------

CENSUS = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                      "predicate_coverage_triton_2026-09-02_residue")

#: The frozen board this product was written against: it scored the cell's three
#: rows as no-admitting-arm with ``curl_admitters: []`` — the refusal the probe
#: extension discharged and this product exists to serve.
BOARD = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                     "fusion_matrix_triton_2026-09-01_foldedbeta",
                     "fusion_matrix.json")
CELL = ("B->H", "folded complex beta PML", "folded complex")

#: The three corpus rows of this cell, by census label. Written out so the
#: census derivation below has a fixed point to be checked against.
CELL_ROWS = (
    "tests:TestEigCoeffs.test_binary_grating_special_kz_0_13_2",
    "tests:TestEigCoeffs.test_binary_grating_special_kz_1_17_7",
    "tests:TestSpecialKz.test_eigsrc_kz_0_complex",
)

#: The ONE row of the three whose sources reach THIS seam. The two binary
#: gratings declare electric sources only; ``test_eigsrc_kz_0_complex`` declares
#: two 'D' and two 'B'. Named here so :func:`corpus_admission_leg` has a fixed
#: point to measure the census against rather than a count to trust.
IN_SEAM_MAGNETIC_ROW = "tests:TestSpecialKz.test_eigsrc_kz_0_complex"


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
    """The three rows this cell is worth, derived from the census; the FROZEN
    board must show them as exactly the no-admitting-arm refusal this product
    discharges.

    AN ABSENT CENSUS IS NOT AN EMPTY ONE: a missing artifact is a REFUSAL here,
    never a funnel of zero reported as a pass. The board read here is the one
    this product was WRITTEN against (the 2026-09-01 foldedbeta cut) — the fresh
    board that scores this cell served is cut AFTER this gate releases, and a
    gate that required its own consequence would never run.
    """
    findings: List[str] = []
    if not os.path.isdir(CENSUS):
        findings.append(f"the census {CENSUS} is not in this tree; the funnel cannot "
                        f"be measured and a zero would be a lie")
    if not os.path.exists(BOARD):
        findings.append(f"the frozen board {BOARD} is not in this tree; the refusal "
                        f"this product discharges cannot be cross-checked")
    if findings:
        return {"leg": "corpus_admission", "device": False,
                "findings": findings, "passed": False}

    # THE ROWS, DERIVED from the census configuration blocks: complex storage,
    # a real fold, beta nonzero, an active absorber, no off-diagonal row, no
    # poles — the cell's own partition.
    derived: List[str] = []
    in_seam: List[str] = []
    configuration: Dict[str, Any] = {}
    for row in _census_rows(CENSUS):
        cfg = row.get("configuration") or {}
        label = f"{row['leg']}:{row['row']}"
        if not cfg.get("force_complex_fields"):
            continue
        if float(cfg.get("beta") or 0.0) == 0.0:
            continue
        if not any(cfg.get("mirrored") or ()):
            continue
        if not cfg.get("pml_active"):
            continue
        if cfg.get("has_offdiagonal_epsilon"):
            continue
        if int(cfg.get("n_polarizations") or 0):
            continue
        derived.append(label)
        configuration[label] = {
            "shape": list(cfg["shape"]), "beta": cfg["beta"],
            "reaches_this_seam": "B" in (cfg.get("source_field_types") or ()),
            "mirrored": list(cfg["mirrored"]),
            "source_field_types": list(cfg.get("source_field_types") or ()),
        }
        if "B" in (cfg.get("source_field_types") or ()):
            in_seam.append(label)
    # THE DEPOSIT BRACKET'S REAL PRICE ON THIS CELL, measured from the census
    # rather than inherited from the electric twin. All three rows inject
    # ELECTRICALLY — which is what makes the twin's flag worth its whole cell —
    # and exactly ONE of them declares in-seam 'B' sources, so on THIS seam
    # CARRIES_DEPOSIT_REPAIR buys one row of three and the other two are the
    # quiet family. Asserted as an equality so a corpus that grew a second
    # magnetic row would fail here rather than quietly re-price the flag.
    if sorted(in_seam) != [IN_SEAM_MAGNETIC_ROW]:
        findings.append(
            f"the census says {sorted(in_seam)} declare in-seam magnetic sources "
            f"on this cell, where this gate is written for exactly "
            f"[{IN_SEAM_MAGNETIC_ROW!r}]; the deposit bracket's price has moved")
    if sorted(derived) != sorted(CELL_ROWS):
        findings.append(
            f"the census derives {sorted(derived)} for this cell where this gate "
            f"declares {sorted(CELL_ROWS)}; one of the two is stale")

    # THE FROZEN BOARD: each row's B->H instance must be the no-admitting-arm
    # refusal (curl_admitters empty, the folded complex constitutive admitting,
    # no product named) — the exact shape the probe extension discharged.
    board = json.load(open(BOARD, encoding="utf-8"))
    frozen: Dict[str, Any] = {}
    for entry in board["seam_instances"]:
        if entry["seam"] != "B->H" or entry["row"] not in CELL_ROWS:
            continue
        frozen[entry["row"]] = {
            "curl_admitters": list(entry["curl_admitters"]),
            "constitutive_admitters": list(entry["constitutive_admitters"]),
            "product": entry["product"],
            "in_seam_source_blocks": entry["in_seam_source_blocks"],
        }
        if entry["curl_admitters"]:
            findings.append(
                f"the frozen board shows {entry['row']!r} with curl admitters "
                f"{entry['curl_admitters']}; the refusal this product discharges "
                f"was an EMPTY curl slot, so this gate's premise is stale")
        if entry["constitutive_admitters"] != ["folded complex"]:
            findings.append(
                f"the frozen board shows {entry['row']!r} with constitutive "
                f"admitters {entry['constitutive_admitters']}, not the folded "
                f"complex arm this product conjoins")
        if entry["product"] is not None:
            findings.append(
                f"the frozen board already names {entry['product']!r} on "
                f"{entry['row']!r}; two products in one cell is a coverage "
                f"question, not a gate result")
        if entry["in_seam_source_blocks"]:
            findings.append(
                f"row {entry['row']!r} is BLOCKED by its in-seam source; the "
                f"deposit repair cannot carry it and this product cannot serve it")
    for label in CELL_ROWS:
        if label not in frozen:
            findings.append(f"row {label!r} has no B->H instance on the frozen "
                            f"board")
    return {"leg": "corpus_admission", "device": False,
            "cell": list(CELL), "rows": sorted(derived),
            "configuration": configuration, "frozen_board_instances": frozen,
            "findings": findings, "passed": not findings}


#: The word table the associativity measurement sweeps. Signed zeros, the subnormal
#: needles, the smallest normals and a few ordinary values — the classes a policy or
#: a rounding decides the fate of, and the ones ``uniform(-1, 1)`` provably misses.
PARITY_WORDS: Tuple[float, ...] = (
    0.0, -0.0, 1e-45, -1e-45, 1.1754942e-38, -1.1754942e-38,
    1.1754944e-38, -1.1754944e-38, 5.9604645e-08, -5.9604645e-08,
    1.0, -1.0, 0.1, -0.1, 3.4028235e38, -3.4028235e38,
)

#: What establishes that a mutation declared ``unreached`` is a REAL defect.
#:
#: ``unreached`` is a third expectation beside ``caught`` and ``null``, and it exists
#: because collapsing the two is how a gate lies in both directions. A ``null`` says
#: THE REWRITE CHANGES NOTHING — the defect is not a defect. ``unreached`` says the
#: defect IS real, names what proves it, and records that THIS gate's device legs do
#: not reach it. Calling the second a null would retire a hazard without carrying it;
#: calling it a catch would be false.
#:
#: THE VERDICT RULE IS NOT SOFTER FOR IT. An ``unreached`` mutation must come back
#: UNCAUGHT — a catch means the declaration is stale and the entry must move to
#: ``caught`` — and it must name its evidence here, or the gate fails.
MUTATION_EVIDENCE: Dict[str, str] = {
    "m3_triple_composite_dropped":
        "the defect is REAL and the beta-less magnetic twin's released gate is "
        "what measures it: on its 3-D case xyz_mixed_phase_periodic_3d the "
        "rewrite is CAUGHT. It cannot be reached here because beta exists only "
        "on the effective-2-D grid (grid.py:668-697; MEEP fields.cpp:546-547), so "
        "no case this gate can build folds three axes and the block the rewrite "
        "deletes is never compiled. Measured by this gate's own "
        "branch_reachability_leg on its twelve cases: deepest composition 3 ghost "
        "cells per source lane, against the 7 a three-fold component owns. Held "
        "by construction — parity_chain_leg enumerates every emitted carry block "
        "against carried_destinations x parity_chain — and by the twin's gate, "
        "not by a whole-step device leg here.",
    "m1_parity_chain_reordered":
        "parity_chain_associativity_leg measures the defect REAL at the word "
        "layer: re-ordering a two-parity chain moves 32 of 2048 uint32 words, "
        "identically on BOTH expansion arms, and every moved word sits at a "
        "signed zero (c_re is exactly +-1.0 and c_im is bitwise +0.0, so the "
        "product is a sign flip plus a signed-zero addend and the composition "
        "commutes to the bit on every normal operand). What no whole-step leg "
        "here can do is get such a word to a ghost SOURCE lane at the moment of "
        "the launch. The ORDER is held by construction instead: "
        "parity_chain_leg compares the shipped sequence against "
        "folded_beta_complex_fused_magnetic_pair.parity_chain block by block. "
        "The beta-less magnetic weld's released gate reached the same position "
        "on the same mutation, over 30 complete steps on xy_mixed_phase_periodic "
        "under the +-0 lattice, and so did the Metal board independently "
        "(gate_metal_folded_complex_fused_magnetic_pair's NULL_EDITS). A "
        "synthetic bare-array leg with planted words at the source lane is what "
        "would close it.",
    "m2_parity_chain_folded":
        "the same sweep measures 34 of 2048 words moved by FOLDING the chain "
        "into one coefficient, on both arms, again all at signed zeros; and the "
        "beta-less twin's 30 complete steps did not reach one. Held by "
        "construction in parity_chain_leg, which asserts that no two parity "
        "coefficients are ever multiplied together in the shipped body, rather "
        "than by measurement here.",
    "m_parity_chain_reordered":
        "parity_chain_associativity_leg measures the defect REAL at the word layer: "
        "re-ordering a two-parity chain moves words on this coefficient class, and "
        "every moved word is at a SIGNED ZERO — c_re is exactly +-1.0 and c_im is "
        "bitwise +0.0, so the product is a sign flip plus a signed-zero addend, "
        "exact on every finite operand and order-dependent only where the operand "
        "is itself a zero. WHAT THIS GATE COULD NOT DO is get such a word to a "
        "ghost SOURCE lane at the moment of a launch. Three attempts are recorded: "
        "the default uniform class with the carry (3 steps), and — added 2026-08-31 "
        "specifically for this — the +-0 lattice, QUIET, at ONE step, which is the "
        "only configuration in which every ghost source lane still holds a seeded "
        "signed zero when step_B runs. All three left every compared volume "
        "byte-identical. The ORDER is held by CONSTRUCTION — parity_chain_leg "
        "compares the shipped sequence against folded_complex_fused_magnetic_pair.parity_"
        "chain block by block, including which coefficient argument each call takes "
        "and where the wall clear lands — and NOT by a whole-step device leg. A "
        "synthetic bare-array leg with planted words at the source lane is what "
        "would close it. THE MAGNETIC TWIN REACHED THE SAME POSITION "
        "INDEPENDENTLY on 2026-08-21 (its own DEVICE_STATUS records 30 complete "
        "steps on the same grid class under the +-0 lattice), which is the "
        "cross-check that this is a property of the arithmetic rather than of this "
        "harness.",
    "m_parity_chain_folded":
        "the same sweep measures words moved by FOLDING the chain into one "
        "coefficient, again all at signed zeros, and the same three device "
        "configurations did not reach one. Held by construction in "
        "parity_chain_leg, which asserts that no block applies fewer multiplies "
        "than its destination carries passes, rather than by measurement here.",
}


def _fma32(a: float, b: float, c: float) -> np.float32:
    """``fma(a, b, c)`` on float32 inputs, computed exactly.

    A float32 product is EXACT in float64 (24 + 24 = 48 mantissa bits against 53),
    and the addend is exact there too, so one float64 add followed by one cast to
    float32 rounds exactly once — which is fused-multiply-add's defining property.
    Not an approximation of the hardware instruction: the same value, by
    construction.
    """
    return np.float32(np.float64(np.float32(a)) * np.float64(np.float32(b))
                      + np.float64(np.float32(c)))


def _parity_multiply(c_re, c_im, z_re, z_im, arm: int):
    """``special_kz._mul_imag_coefficient_left``'s two arms, transcribed.

    The lines are the shipped ones (special_kz.py:328-334) and the leg asserts they
    are still there before it trusts this transcription.
    """
    c_re = np.float32(c_re); c_im = np.float32(c_im)
    z_re = np.float32(z_re); z_im = np.float32(z_im)
    if arm == 1:
        out_re = _fma32(c_re, z_re,
                        np.float32(np.float32(c_im * z_im) * np.float32(-1.0)))
        out_im = _fma32(c_re, z_im, np.float32(c_im * z_re))
    else:
        out_re = np.float32(np.float32(c_re * z_re) - np.float32(c_im * z_im))
        out_im = np.float32(np.float32(c_re * z_im) + np.float32(c_im * z_re))
    return out_re, out_im


def _words(pair) -> Tuple[int, int]:
    return (int(np.float32(pair[0]).view(np.uint32)),
            int(np.float32(pair[1]).view(np.uint32)))


def parity_chain_associativity_leg() -> Dict[str, Any]:
    """IS THE ORDERING OBSERVABLE AT ALL? Measured, on the shipped multiply's arms.

    THE MEASUREMENT THIS LEG EXISTS FOR. Two armed device mutations re-order and fold
    this kernel's parity chain, and BOTH come back UNCAUGHT on every device
    configuration this gate can build. A null is only honest if the question behind
    it is settled rather than hidden, so it is settled here, at the word layer, on
    the arithmetic itself rather than on the whole-step state.

    WHAT IT DOES NOT LICENSE. A count here says the ORDER is observable for THIS
    coefficient class on THIS backend; it does not say this gate reached it, and the
    two mutations are declared ``unreached`` rather than ``caught`` for exactly that
    reason. Nor does a ZERO count license re-ordering the kernel: the kernel
    transcribes the driver's order and :func:`parity_chain_leg` pins that by
    construction.

    THE LEG FAILS IF A ``caught`` DECLARATION MEETS A ZERO COUNT — a mutation that
    cannot be seen at the word layer cannot be caught at the whole-step layer either,
    and declaring it caught would be a claim about nothing.
    """
    from meep_gpu.triton_kernels.folded_complex import (  # noqa: PLC0415
        mirror_parity_coefficients,
    )

    findings: List[str] = []
    shipped = _shipped_text(
        "_mul_imag_coefficient_left",
        os.path.join(API_ROOT, "meep_gpu", "triton_kernels", "special_kz.py"))
    for line in ("out_re = tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)",
                 "out_im = tl.math.fma(c_re, z_im, c_im * z_re)",
                 "out_re = (c_re * z_re) - (c_im * z_im)",
                 "out_im = (c_re * z_im) + (c_im * z_re)"):
        if line not in shipped:
            findings.append(
                f"special_kz._mul_imag_coefficient_left no longer carries {line!r}; "
                f"this leg transcribes that body and cannot vouch for a copy of "
                f"something that moved")

    rows: List[Dict[str, Any]] = []
    for arm in (0, 1):
        reorder = fold = total = 0
        for phase_a in (1, -1):
            for phase_b in (1, -1):
                # The two NEAR coefficients of a two-folded-axis grid at those
                # phases — the pair THIS family's near chain composes.
                c_a = mirror_parity_coefficients(phase_a)[0]
                c_b = mirror_parity_coefficients(phase_b)[0]
                folded = _parity_multiply(c_a[0], c_a[1], c_b[0], c_b[1], arm)
                for z_re in PARITY_WORDS:
                    for z_im in PARITY_WORDS:
                        total += 2
                        ascending = _parity_multiply(
                            c_b[0], c_b[1],
                            *_parity_multiply(c_a[0], c_a[1], z_re, z_im, arm),
                            arm)
                        descending = _parity_multiply(
                            c_a[0], c_a[1],
                            *_parity_multiply(c_b[0], c_b[1], z_re, z_im, arm),
                            arm)
                        one_shot = _parity_multiply(
                            folded[0], folded[1], z_re, z_im, arm)
                        reorder += sum(
                            1 for x, y in zip(_words(ascending), _words(descending))
                            if x != y)
                        fold += sum(
                            1 for x, y in zip(_words(ascending), _words(one_shot))
                            if x != y)
        rows.append({"arm": arm, "words_compared": total,
                     "reorder_differing_words": reorder,
                     "fold_differing_words": fold})

    table = {name: expectation for name, _why, expectation, _r in mutation_table()}
    for name, key in (("m_parity_chain_reordered", "reorder_differing_words"),
                      ("m_parity_chain_folded", "fold_differing_words")):
        observed = max(row[key] for row in rows)
        declared = table.get(name)
        if declared == "null" and observed:
            findings.append(
                f"{name} is declared NULL but the arithmetic moves {observed} of "
                f"{rows[0]['words_compared']} words: the declaration and the "
                f"measurement disagree")
        if declared in ("caught", "unreached") and not observed:
            findings.append(
                f"{name} is declared {declared!r} but the arithmetic moves NO word "
                f"on either arm: it cannot be seen at the whole-step layer either, "
                f"and a defect that cannot be seen is not a defect")
    return {"leg": "parity_chain_associativity", "device": False,
            "coefficient_class": "c_re is exactly +-1.0 and c_im is bitwise +0.0 "
                                 "(folded_complex.mirror_parity_coefficients)",
            "value_words": len(PARITY_WORDS), "rows": rows,
            "findings": findings, "passed": not findings}


def mutation_pairing_leg() -> Dict[str, Any]:
    """Every mutation is ARMED, is scored on a LIVE branch, and its tables are sane.

    THE DUPLICATE-KEY CHECK IS NOT DECORATION. ``MUTATION_CASE`` is a dict literal,
    so a name written twice silently takes the LAST value — and on 2026-08-31 that
    is exactly what happened here: ``m_fu_store_masked`` was re-pointed at an
    absorbing destination plane and a stale duplicate further down put it straight
    back on a case where the rewrite is a no-op, where it reported UNCAUGHT. A
    mis-scored mutation is a mutation that measured nothing, so the tables are read
    as TEXT and a repeated key fails here rather than on a device.
    """
    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    source = shipped_source()
    allowed = {"caught", "null", "unreached"}
    for name, why, expectation, rewrite in mutation_table():
        if expectation not in allowed:
            findings.append(f"{name}: expectation {expectation!r} is not one of "
                            f"{sorted(allowed)}")
        if expectation == "unreached" and not MUTATION_EVIDENCE.get(name):
            findings.append(
                f"{name} is declared 'unreached' and MUTATION_EVIDENCE names "
                f"nothing that establishes the defect is real; an unreached "
                f"mutation with no evidence is a null with a longer word")
        if expectation != "unreached" and MUTATION_EVIDENCE.get(name):
            findings.append(
                f"{name} carries MUTATION_EVIDENCE but is not declared 'unreached'; "
                f"the evidence would never be read")
        mutated, hits = rewrite(source)
        armed = bool(hits) and mutated != source
        row: Dict[str, Any] = {"mutation": name, "why": why,
                               "expectation": expectation, "rewrite_hits": hits,
                               "armed": armed}
        if not armed:
            findings.append(f"{name}: the rewrite matched nothing")
        try:
            ast.parse(mutated)
        except SyntaxError as exc:
            findings.append(f"{name}: the mutant does not PARSE ({exc}); it would be "
                            f"reported as 'did not compile' rather than as a defect")
        try:
            row["case"] = mutation_case_for(name)[0]
        except AssertionError as exc:
            findings.append(f"{name}: {exc}")
        rows.append(row)

    text = open(os.path.abspath(__file__), encoding="utf-8").read()
    for table in ("MUTATION_CASE", "MUTATION_VALUE_CLASS", "MUTATION_STEPS_FOR",
                  "MUTATION_ELECTRIC", "MUTATION_REQUIRES_FOLD",
                  "MUTATION_REQUIRES_PERIODIC_FOLD", "MUTATION_REQUIRES_WALL",
                  "MUTATION_REQUIRES_ODD_PARITY", "MUTATION_REQUIRES_ODD_FULL_COUNT",
                  "MUTATION_REQUIRES_MOVING_FAR_COEFFICIENT"):
        start = text.index(f"\n{table}:")
        body = text[start:text.index("\n}\n", start)]
        keys = [line.split('"')[1] for line in body.splitlines()
                if line.strip().startswith('"') and '":' in line]
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        if repeated:
            findings.append(
                f"{table} names {repeated} more than once; a dict literal takes the "
                f"LAST value and the earlier entry is silently discarded")
    declared = {name for name, _w, _e, _r in mutation_table()}
    for table, mapping in (("MUTATION_CASE", MUTATION_CASE),
                           ("MUTATION_VALUE_CLASS", MUTATION_VALUE_CLASS),
                           ("MUTATION_STEPS_FOR", MUTATION_STEPS_FOR),
                           ("MUTATION_ELECTRIC", MUTATION_ELECTRIC),
                           ("MUTATION_REQUIRES_FOLD", MUTATION_REQUIRES_FOLD),
                           ("MUTATION_REQUIRES_PERIODIC_FOLD",
                            MUTATION_REQUIRES_PERIODIC_FOLD),
                           ("MUTATION_REQUIRES_WALL", MUTATION_REQUIRES_WALL),
                           ("MUTATION_REQUIRES_ODD_PARITY",
                            MUTATION_REQUIRES_ODD_PARITY),
                           ("MUTATION_REQUIRES_ODD_FULL_COUNT",
                            MUTATION_REQUIRES_ODD_FULL_COUNT),
                           ("MUTATION_REQUIRES_MOVING_FAR_COEFFICIENT",
                            MUTATION_REQUIRES_MOVING_FAR_COEFFICIENT)):
        unknown = sorted(set(mapping) - declared)
        if unknown:
            findings.append(f"{table} names {unknown}, which the mutation table does "
                            f"not declare; those entries are read by nothing")
    return {"leg": "mutation_pairing", "device": False, "mutations": rows,
            "findings": findings, "passed": not findings}


NO_DEVICE_LEGS = (transcription_leg, parity_chain_leg,
                  parity_chain_associativity_leg, branch_reachability_leg,
                  seam_binding_leg, predicate_leg, corpus_admission_leg,
                  mutation_pairing_leg)


# ---------------------------------------------------------------------------
# Device side — value classes, drivers, inventory, launch counting
# ---------------------------------------------------------------------------

def value_class_hosts(names: Sequence[str], shape, rng,
                      value_class: str) -> Dict[str, np.ndarray]:
    """Host complex64 arrays for one value class."""
    host: Dict[str, np.ndarray] = {}
    if value_class == "uniform":
        for name in names:
            host[name] = (rng.uniform(-0.25, 0.25, size=shape)
                          + 1j * rng.uniform(-0.25, 0.25, size=shape)
                          ).astype(np.complex64)
        return host
    if value_class == "subnormal_band":
        for name in names:
            parts = []
            for _half in range(2):
                exponents = rng.uniform(-45.0, -30.0, size=shape)
                signs = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
                plane = (signs * np.power(10.0, exponents)).astype(np.float32)
                picks = rng.integers(0, 16, size=shape) == 0
                choice = SUBNORMAL_NEEDLES[
                    rng.integers(0, SUBNORMAL_NEEDLES.size, size=shape)]
                parts.append(np.where(picks, choice, plane).astype(np.float32))
            host[name] = (parts[0] + 1j * parts[1]).astype(np.complex64)
        return host
    if value_class == "zero_lattice":
        zeros = np.array([0.0, -0.0], dtype=np.float32)
        for name in names:
            real = zeros[rng.integers(0, 2, size=shape)]
            imaginary = zeros[rng.integers(0, 2, size=shape)]
            host[name] = (real + 1j * imaginary).astype(np.complex64)
        return host
    raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")


def operand_census(arrays: Dict[str, np.ndarray]) -> Dict[str, int]:
    """How many subnormals, signed zeros and zeros a leg's operands ACTUALLY hold.

    Classified off the BITS, not by comparison: ``x < FLT_MIN`` is true for zero too,
    and ``x == -0.0`` is true for ``+0.0``.
    """
    flat = np.concatenate([np.asarray(a, dtype=np.complex64).ravel().view(np.float32)
                           for a in arrays.values()])
    raw = flat.view(np.uint32)
    exponent = (raw >> 23) & 0xFF
    mantissa = raw & 0x7FFFFF
    return {
        "words": int(flat.size),
        "subnormals": int(((exponent == 0) & (mantissa != 0)).sum()),
        "negative_zeros": int((raw == 0x80000000).sum()),
        "zeros": int((raw == 0).sum()),
    }


#: The floor each value class must clear, per leg, or the leg measured nothing of
#: what its name claims.
CLASS_FLOOR: Dict[str, Dict[str, int]] = {
    "uniform": {},
    "subnormal_band": {"subnormals": 1},
    "zero_lattice": {"negative_zeros": 1, "zeros": 1},
}

SEED_FIELDS = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")

#: THE SEAM'S OWN AUXILIARIES ARE SEEDED TOO, and they are the ``H``/``B`` ones,
#: not the electric twin's. This seam writes ``f_w_Hx..z`` and ``fu_Bx..z``
#: (:data:`SEAM_OUTPUTS`); seeding ``f_w_E``/``fu_D`` instead would leave every
#: volume this launch actually touches starting at zero, and a store of an
#: identically-zero value moves no byte — the same vacuity the beta-less twin
#: measured on 2026-08-31, when a real store-masking defect reported UNCAUGHT for
#: exactly that reason. Seeding them makes the destination planes live numbers.
SEED_AUXILIARIES = ("f_w_Hx", "f_w_Hy", "f_w_Hz", "fu_Bx", "fu_By", "fu_Bz")


def build_driver(cp, case, value_class: str, in_seam: bool):
    """One folded complex driver, seeded identically for every route.

    ``in_seam`` IS THE CARRY FLAG AND IT NAMES THIS SEAM'S FIELD TYPE, which on the
    magnetic side is ``B`` — the electric twin's parameter is spelled ``electric``
    and was renamed here rather than carried across, because a flag called
    ``electric`` selecting a magnetic deposit is the kind of name that survives one
    reader and misleads the next. What it selects is a ``Hz`` deposit, injected BETWEEN ``step_B`` and ``update_H``
    (driver.py:3283-3284), which is the deposit this product's
    ``CARRIES_DEPOSIT_REPAIR`` exists for. The FALSE branch carries an ``Ez`` source
    instead — admitted with no repair at all, because the driver injects it in the
    OTHER seam — so the source clause is exercised in both directions and the quiet
    family still has something driving its fields.

    GETTING THIS BACKWARDS IS SILENT. An ``Ez`` deposit on this seam leaves the
    bracket with nothing to repair: every carry leg would report identical with zero
    repairs and certify a kernel the corpus row never reaches.
    """
    driver = build_grid(case)
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, deliberately: not for this launch, which binds no inverse
    # permeability at all, but for the ARRAY PATH's `update_E` in the other half of
    # the same driver step — a uniform material would make the two routes agree for
    # a reason that is not this kernel's.
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    center = [0.0, 0.0, 0.0]
    for axis_name, _phase in case[4]:
        axis = "XYZ".index(axis_name.upper())
        center[axis] = 0.25 * (case[2][axis] / 2.0)
    # OFF THE MIRROR PLANE ON EVERY FOLDED AXIS. A point source at the origin sits
    # ON the plane, and on an ODD fold that is refused with a reason rather than
    # stepped (from_meep.py:2686-2698).
    driver.add_source({"component": "Hz" if in_seam else "Ez",
                       "frequency": 0.31,
                       "center": tuple(center), "width": 0.4})
    rng = np.random.default_rng(case_seed(case[0], value_class))
    hosts = value_class_hosts(SEED_FIELDS, shape, rng, value_class)
    for name, values in hosts.items():
        driver.set_field(name, cp.asarray(np.ascontiguousarray(values)))
    auxiliary = value_class_hosts(SEED_AUXILIARIES, shape, rng, value_class)
    for name, values in auxiliary.items():
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = cp.asarray(np.ascontiguousarray(values))
            hosts[name] = values
    return driver, hosts


def inventory(driver) -> Dict[str, Any]:
    """Every device volume the step can touch, found by scanning rather than listing."""
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


class _TwoPassFill:
    """The certified complex ghost fill, run as the DRIVER runs it.

    ``fill_symmetry_bc_B`` is every folded axis's NEAR plane and
    ``fill_folded_far_ghosts_B`` every folded axis's FAR one, with ``zero_metal_B``
    BETWEEN them. Fusing them per axis is a DIFFERENT ORDER and a measured different
    answer under complex storage, which is exactly why the oracle route must not take
    the shortcut the fused kernel is being measured against.
    """

    __slots__ = ("plan", "phase")

    def __init__(self, plan: Any, phase: str) -> None:
        if phase not in ("near", "far"):
            raise ValueError(f"phase must be 'near' or 'far', got {phase!r}")
        self.plan = plan
        self.phase = phase

    def run(self) -> None:
        if self.phase == "near":
            self.plan.run_near()
        else:
            self.plan.run_far()


def install(driver_module, routes, counter: Dict[str, int], roles: Dict[int, str]):
    """Replace the five seam passes for the fields objects named in ``routes``.

    Counting is PER ROUTE. A single global counter would mix the reference driver's
    legitimate array-path calls with a fallback in the fused route, which is precisely
    the event this instrument exists to see.
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


def separate_route(driver, probe):
    """The three separately certified Triton products this launch replaces."""
    from meep_gpu.triton_kernels import folded_complex  # noqa: PLC0415

    # K3b, the certified folded complex BETA curl — its own released gate
    # carries the beta mutations; the beta-less K1 refuses this grid.
    curl = folded_complex.plan_folded_beta_bloch_pml_curl(
        driver.fields, driver.pml, "step_B", probe=probe)
    constitutive = folded_complex.plan_folded_complex_constitutive(
        driver.fields, driver.pml, "H", probe=probe)
    fill = folded_complex.plan_folded_mirror_ghost_fill_complex(
        driver.fields, "B", probe=probe)
    missing = [name for name, plan in
               (("folded complex beta curl", curl), ("folded complex ghost fill", fill),
                ("folded complex constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so this "
            f"leg could not compare the fused launch against the products it "
            f"replaces")
    # `zero_metal_B` stays on the ARRAY PATH here: no Triton product owns the wall
    # clear. It is counted, so a difference in whether it ran is a counter event
    # rather than an invisible correction. THE INJECTION ALSO STAYS ON THE ARRAY
    # PATH, which is what makes this the right oracle for the carry family: the
    # certified kernels with the driver's own deposit between them.
    return Route({"step_B": curl,
                  "fill_symmetry_bc_B": _TwoPassFill(fill, "near"),
                  "fill_folded_far_ghosts_B": _TwoPassFill(fill, "far"),
                  "update_H": constitutive})


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

    seam = deposit_repair.in_seam_sources(tuple(sources), "B")
    if not seam or not bracket:
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


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, case, steps: int, product, probe,
            value_class: str = "uniform", mutant: Any = None,
            install_fused: bool = True, freeze_seam: bool = False,
            in_seam: bool = False, bracket: bool = True,
            expansion: Optional[int] = None) -> Dict[str, Any]:
    """Three routes in lockstep, per COMPLETE driver step; stop at the FIRST byte
    divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference, seeded = build_driver(cp, case, value_class, in_seam)
    separate, _ = build_driver(cp, case, value_class, in_seam)
    fused, _ = build_driver(cp, case, value_class, in_seam)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_beta_complex_fused_curl_constitutive_B_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "case": case[0], "steps_budget": steps,
        "value_class": value_class, "seed": case_seed(case[0], value_class),
        "shape": list(reference.shape), "boundaries": str(case[3]),
        "mirrors": [list(entry) for entry in case[4]],
        "k_point": list(case[5]), "pml": str(case[6]),
        "electric_source": bool(in_seam),
        "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_seam),
        "expansion_override": expansion,
        "operand_census": operand_census(seeded),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_folded_beta_complex_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.folded_beta_complex_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources), probe=probe)
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        if expansion is not None:
            # THE ARM, OVERRIDDEN. Not a source rewrite: the constexpr IS the licence,
            # so flipping it here measures whether the licence is load-bearing rather
            # than decorative.
            plan.expansion = int(expansion)
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        row["licensed_expansion"] = plan.expansion
        row["near"] = [bool(value) for value in plan.near]
        row["far"] = [bool(value) for value in plan.far]
        row["reflect_rows"] = list(plan.reflect)
        row["parity"] = [list(pair) for pair in plan.parity]
        row["zero_metal"] = [bool(value) for value in plan.zero_metal]
        row["ghost_destinations_per_source_lane"] = list(
            ghost_destinations(plan.near, plan.far))
        if install_fused:
            route, leading = fused_route(plan, fused, tuple(fused._sources),
                                         bracket=bracket)
        else:
            route = Route({})
        separate_plans = separate_route(separate, probe)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_seam:
            # ARMED: every route's B->H seam is inert. All three then agree
            # trivially and only the moved-state census can refuse it.
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
        material = [key for key in final if key in MATERIAL]
        reference_final = snapshot(cp, reference)
        reference_moved = moved(reference_opening, reference_final)
        row["arrays_total"] = len(final)
        row["arrays_compared"] = sorted(final)
        row["arrays_ever_moved"] = len(ever_moved)
        row["arrays_never_moved"] = sorted(
            set(final) - set(ever_moved) - set(material))
        # THE FLOOR IS RELATIVE TO THE ARRAY PATH. An absolute "every array moved"
        # floor fails a CORRECT kernel for properties of the configuration; what IS
        # evidence is an array the ARRAY PATH moves and the fused route does not.
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
        row["elapsed_seconds"] = time.time() - started
        return row
    finally:
        undo()
        for target in (reference, separate, fused):
            try:
                target.close()
            except Exception:  # noqa: BLE001 - a close failure must not hide a result
                pass
        cp.get_default_memory_pool().free_all_blocks()



def identity_has_beta_off_leg(cp, product, probe) -> Dict[str, Any]:
    """``HAS_BETA = 0`` must BE the beta-less shipped weld, byte for byte.

    Three drivers on the same seeded beta grid, quiet family. Route OFF runs
    THIS kernel with the insert compiled out (``plan.has_beta = 0``); route BASE
    runs the beta-less shipped product, whose plan is assembled directly through
    its plan class because its engine-route predicate refuses a beta grid (K1
    requires beta = 0 — the disjointness the predicate leg measures). The two
    must agree at EVERY step, and BOTH must diverge from the beta-carrying array
    path — the vacuity guard: a case whose beta term never fired would pass the
    identity trivially and measure nothing.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.stepping import _boundary_kinds  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_complex_fused_magnetic_pair as base_product  # noqa: PLC0415
    from meep_gpu.triton_kernels.complex_fields import (  # noqa: PLC0415
        _phase_arguments, bloch_phase_table)
    from meep_gpu.triton_kernels.coverage import (  # noqa: PLC0415
        CONSTITUTIVE_SIDES, zero_metal_axes)
    from meep_gpu.triton_kernels.folded_complex import (  # noqa: PLC0415
        _far_reflect_rows, folded_axis_kinds)
    from meep_gpu.triton_kernels.launch import SUB_STEPS  # noqa: PLC0415

    case = CASES_BY_NAME["y_fold_periodic"]
    row: Dict[str, Any] = {"leg": "identity:has_beta_off", "device": True,
                           "case": case[0], "armed": True}
    reference, _ = build_driver(cp, case, "uniform", False)
    off, _ = build_driver(cp, case, "uniform", False)
    base, _ = build_driver(cp, case, "uniform", False)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    try:
        kernel_off = CountingKernel(
            product.folded_beta_complex_fused_curl_constitutive_B_kernel())
        plan_off = product.plan_folded_beta_complex_fused_magnetic_pair(
            off.fields, off.pml, tuple(off._sources), num_warps=1,
            kernel=kernel_off, probe=probe)
        if plan_off is None:
            raise AssertionError("the beta product refused its own identity case")
        plan_off.has_beta = 0
        row["beta_words_bound"] = [list(pair) for pair in plan_off.beta_words]

        kernel_base = CountingKernel(
            base_product.folded_complex_fused_curl_constitutive_B_kernel())
        grid = base.fields.grid
        curl_spec = SUB_STEPS["step_B"]
        side_spec = CONSTITUTIVE_SIDES["H"]
        codes, code_reasons = folded_axis_kinds(grid, base.pml)
        if codes is None:
            raise AssertionError(f"folded_axis_kinds refused: {code_reasons}")
        kinds = _boundary_kinds(grid, base.pml)
        phased, values = _phase_arguments(bloch_phase_table(grid, kinds),
                                          backward=bool(curl_spec["backward"]))
        # THE BASE PLAN TAKES NO INVERSE PERMEABILITY. The electric twin's identity
        # leg passes an `inverse_epsilon_for` list between the constitutive targets
        # and their coefficients; `update_H` divides by nothing, so
        # FoldedComplexFusedMagneticPairPlan's signature simply has no such slot and
        # the argument order below is the base module's own
        # `plan_folded_complex_fused_magnetic_pair_from_arrays` (:1674-1707).
        plan_base = base_product.FoldedComplexFusedMagneticPairPlan(
            grid.shape, grid.dt / grid.dx, codes, zero_metal_axes(grid),
            phased, values, base_product.parity_coefficient_words(grid),
            plan_off.expansion, plan_off.block,
            [getattr(base.fields, name) for name in curl_spec["targets"]],
            [getattr(base.fields, "fu_" + name) for name in curl_spec["targets"]],
            [getattr(base.fields, name) for name in curl_spec["sources"]],
            [getattr(base.pml, f"{stem}_{axis}{curl_spec['suffix']}")
             for axis in "xyz" for stem in ("kms", "sinv")],
            [getattr(base.fields, name) for name in side_spec["targets"]],
            [getattr(base.fields, name) for name in side_spec["aux"]],
            [getattr(base.pml,
                     f"{stem}_{axis}{'_h' if side_spec['half_integer'] else ''}")
             for axis in "xyz" for stem in ("kps", "kms")],
            reflect=_far_reflect_rows(grid) or (None, None, None),
            kernel=kernel_base, num_warps=1)

        route_off, _lead = fused_route(plan_off)
        route_base, _lead = fused_route(plan_base)
        roles = {id(reference.fields): "array", id(off.fields): "has_beta_off",
                 id(base.fields): "beta_less_product"}
        undo = install(driver_module,
                       [(off.fields, route_off), (base.fields, route_base)],
                       counter, roles)
        row["per_step"] = []
        diverged_from_array = False
        for step in range(1, 4):
            reference.step()
            off.step()
            base.step()
            cp.cuda.runtime.deviceSynchronize()
            identical = first_divergence(snapshot(cp, off), snapshot(cp, base))
            vs_array = first_divergence(snapshot(cp, off),
                                        snapshot(cp, reference))
            diverged_from_array = diverged_from_array or vs_array is not None
            row["per_step"].append({"step": step,
                                    "off_vs_beta_less": identical,
                                    "off_vs_beta_array_path": vs_array})
            log(f"  identity step {step}/3 off==beta_less={identical is None} "
                f"off!=array={vs_array is not None}")
        row["launches"] = {"has_beta_off": kernel_off.calls,
                           "beta_less_product": kernel_base.calls}
        failures = []
        if any(entry["off_vs_beta_less"] is not None
               for entry in row["per_step"]):
            failures.append("HAS_BETA=0 is NOT the beta-less shipped weld")
        if not diverged_from_array:
            failures.append(
                "the beta-carrying array path never diverged from the "
                "HAS_BETA=0 route: the beta term did not fire on this case and "
                "the identity was vacuous")
        if kernel_off.calls != 3 or kernel_base.calls != 3:
            failures.append(
                f"launch counts {row['launches']} are not one per step; a "
                f"fallback would make the identity a comparison of the array "
                f"path with itself")
        row["failures"] = failures
        row["passed"] = not failures
        return row
    except Exception as exc:  # noqa: BLE001 - a raised leg is a recorded leg
        row["error"] = repr(exc)
        row["passed"] = False
        return row
    finally:
        undo()
        for target in (reference, off, base):
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
               require_ghost_destinations: Optional[int] = None
               ) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    floor = CLASS_FLOOR.get(str(row.get("value_class")), {})
    census = row.get("operand_census") or {}
    for key, minimum in floor.items():
        if int(census.get(key, 0)) < minimum:
            failures.append(
                f"VACUOUS OPERAND CLASS: value_class={row.get('value_class')!r} "
                f"seeded {census.get(key, 0)} {key}, below the floor {minimum}")
    if require_ghost_destinations is not None:
        observed = row.get("ghost_destinations_per_source_lane") or []
        if max(observed or [0]) != int(require_ghost_destinations):
            failures.append(
                f"this leg was declared to reach composition depth "
                f"{require_ghost_destinations} and reached {max(observed or [0])} "
                f"({observed})")
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

def shipped_source(product: Any = None) -> str:
    """The kernel AND the device function it calls, as ONE mutable text.

    ``_carry_ghost_complex`` is included for two reasons, both failures a mutation
    harness cannot see from its own result. A name the mutant module does not define
    makes it fail to IMPORT, which is recorded as UNARMED rather than as a caught
    defect; and a device function left OUT of the mutated text is a piece of the
    product no mutation can reach at all.

    Read from the FILE, so the mutation table can be applied on a laptop with no
    Triton — which is what the host suite's arming test does.

    THE DECORATOR IS PREPENDED, and it is not cosmetic. ``ast.get_source_segment``
    on a ``FunctionDef`` EXCLUDES the decorator list, so a mutant compiled from the
    bare segment is a PLAIN PYTHON FUNCTION with no ``[grid]`` launcher and the first
    mutation leg dies with "'function' object is not subscriptable" — measured
    2026-08-20 on a sibling gate, and again 2026-08-31 on this one.
    """
    text = open(_MODULE_FILE, encoding="utf-8").read()
    tree = ast.parse(text)
    out = []
    for name in (CARRY_HELPER, KERNEL_NAME):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                decorators = [ast.get_source_segment(text, decorator)
                              for decorator in node.decorator_list]
                if any(segment is None for segment in decorators):
                    raise AssertionError(f"{name}'s decorator source is unavailable")
                out.append("\n".join(
                    [f"@{segment}" for segment in decorators]
                    + [textwrap.dedent(ast.get_source_segment(text, node))]))
                break
        else:  # pragma: no cover - the transcription leg fails first
            raise AssertionError(f"{name} is not defined in the shipped module")
    return "\n\n".join(out)


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant from the (possibly mutated) kernel + helper text."""
    from meep_gpu.triton_kernels import symmetry as _symmetry  # noqa: PLC0415

    constants = "".join(
        f"{name} = tl.constexpr({getattr(_symmetry, 'CODE_' + name)})\n"
        for name in ("PERIODIC", "METALLIC", "MIRROR_METALLIC", "MIRROR_PERIODIC"))
    header = (
        "import triton\nimport triton.language as tl\n"
        "from meep_gpu.triton_kernels.complex_fields import (\n"
        "    _mul_coefficient_left, _mul_field_left, _rotate_field_left)\n"
        "from meep_gpu.triton_kernels.special_kz import _mul_imag_coefficient_left\n"
        + constants + "\n")
    text = source
    # THE HELPER IS RENAMED WITH THE KERNEL. Two mutants sharing a helper identity
    # would be two callers of one cached device body, which is the hazard the kernel
    # rename exists to close, one level down.
    text = text.replace(CARRY_HELPER, f"{CARRY_HELPER}_{kernel_name}")
    text = text.replace(KERNEL_NAME, kernel_name)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_folded_complex_pair.py", delete=False, encoding="utf-8")
    handle.write(header + text)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_complex_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was
    MEASURED, never what was hoped."""

    def m1_parity_chain_reordered(source: str) -> Tuple[str, int]:
        """THE ORDER, reversed on one two-parity chain.

        The far axes ascend in the driver (``_fill_folded_far_ghosts``:1518); this
        applies them descending. THE BLOCK IS COMPONENT 2's ``FAR_X and FAR_Y``, and
        the choice is not free: this mutation is scored on a fold-X + fold-Y grid,
        so a rewrite landing in the ``FAR_X and FAR_Z`` block would be DEAD CODE
        there and would report UNCAUGHT while measuring nothing. (That is exactly
        what the first cut of this table did, and
        ``test_every_mutation_is_scored_on_a_case_that_ENTERS_the_branch_it_rewrites``
        is what caught it.)
        """
        return _rewrite_block(
            source,
            ["gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, v2_re, v2_im,",
             "EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, gc_re, gc_im,",
             "EXPANSION)"],
            ["gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, v2_re, v2_im, EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(d0r, d0i, gc_re, gc_im, EXPANSION)"])

    def m2_parity_chain_folded(source: str) -> Tuple[str, int]:
        """THE GROUPING, folded: two complex multiplies collapsed into one.

        The real-storage twin's spelling — one compile-time sign — applied here.
        Under complex storage the product of the coefficient words is not the
        composition of the two multiplies.
        """
        return _rewrite_block(
            source,
            ["gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v0_re, v0_im,",
             "EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(d1r, d1i, gc_re, gc_im,",
             "EXPANSION)"],
            ["gc_re, gc_im = _mul_imag_coefficient_left(",
             "    n0r * d1r - n0i * d1i, n0r * d1i + n0i * d1r,",
             "    v0_re, v0_im, EXPANSION)"])

    def m3_triple_composite_dropped(source: str) -> Tuple[str, int]:
        """The deepest destination never written. Needs three folded PERIODIC axes."""
        return _rewrite_block(
            source,
            ["_carry_ghost_complex(f0, w0, h0, idx + dn_x + df_y + df_z,",
             "gc_re, gc_im, kp_d0, km_d0,",
             "own0 & near_i & far_j & far_k, EXPANSION)"],
            ["pass"])

    def m4_far_parity_is_the_near_one(source: str) -> Tuple[str, int]:
        """``+phase`` where the far fill takes ``-phase``.

        ``mirror_parity(c, axis, phase) == phase * (1 - 2*iyee)`` (fields.py:117):
        the far fill touches shift-1 components alone and takes the NEGATED phase.
        """
        return _replace_all(
            source,
            "gy_re, gy_im = _mul_imag_coefficient_left(d1r, d1i, v0_re, v0_im,",
            "gy_re, gy_im = _mul_imag_coefficient_left(n1r, n1i, v0_re, v0_im,")

    def m5_reflect_row_baked(source: str) -> Tuple[str, int]:
        """``n - 2`` instead of the runtime row.

        ``_far_reflect_rows`` is ``n_full - stored + 2``, which is ``stored - 2`` at
        an EVEN full count and ``stored - 3`` at an ODD one. Baking reflects about
        the window top instead of about the second mirror. Scored on the ODD case;
        a null on the even one is arithmetic, not a gap.
        """
        return _rewrite_block(
            source, ["far_j = live & (j == ry)"], ["far_j = live & (j == ny - 2)"])

    def m6_top_plane_mask_dropped(source: str) -> Tuple[str, int]:
        """The folded curl's DELTA 3, removed on one axis.

        The mask arrived WITH the far carry: a product that took the fill and left
        the block out would leave that plane unmasked.
        """
        # BODY ONLY. Replacing the `if` header too writes `pass` at the header's
        # indent, which is an IndentationError and an UNARMED mutation — measured
        # 2026-08-21, when this rewrite and m10 both failed to import.
        return _rewrite_block(
            source,
            ["curl0_re = tl.where(last_y, 0.0, curl0_re)",
             "curl0_im = tl.where(last_y, 0.0, curl0_im)"],
            ["pass"])

    def m7_near_source_index_three(source: str) -> Tuple[str, int]:
        """``MIRROR_SOURCE_INDEX`` off by one: the near fill images stored cell 2."""
        return _rewrite_block(
            source, ["near_j = live & (j == 2)"], ["near_j = live & (j == 3)"])

    def m8_ownership_mask_not_ANDed(source: str) -> Tuple[str, int]:
        """The 2026-08-20 device defect, re-planted.

        With two fills a lane can be the SOURCE of one and the DESTINATION of the
        other; dropping ``own?`` from the carry mask lets it write a composite cell
        from a ``v`` its own mask zeroed.
        """
        # THE BLOCK IS COMPONENT 1's FAR-ONLY DESTINATION ON X, and the choice is
        # measured rather than free. `own1` is `live & (j != 0) & (i != nx - 1)`; on
        # the composite destination (`near_j & far_i`, i.e. j == 2) the `j != 0`
        # clause is TRUE anyway, so dropping `own1` there changes no lane and the
        # mutation is a value-level NO-OP — measured UNCAUGHT on 2026-08-21. The
        # lane the real defect was about is the one at `(i == rx, j == 0)`: the far
        # source for By on x, and itself the NEAR fill's destination on y. That lane
        # is excluded by `own1` and by nothing else in this mask.
        return _replace_all(
            source,
            "kp_1, km_1, own1 & far_i, EXPANSION)",
            "kp_1, km_1, far_i, EXPANSION)")

    def m9_destination_coefficient_reused(source: str) -> Tuple[str, int]:
        """The near destination takes the SOURCE lane's coefficient pair.

        ``update_H`` indexes component m on axis m and the near fill images along
        that same axis, so the destination's pair is at index 0 and is NOT the
        source's. On a folded axis whose absorber sits on the HIGH face alone the
        two entries are the same word to the bit, so this is predicted a NULL —
        recorded WITH that reason rather than as a catch.
        """
        return _replace_all(
            source, "kp_d1, km_d1, own1 & near_j, EXPANSION)",
            "kp_1, km_1, own1 & near_j, EXPANSION)")

    def m10_wall_clear_dropped(source: str) -> Tuple[str, int]:
        """``zero_metal_B`` not carried: MEEP's ``step_boundaries(B_stuff)``, undone."""
        # BODY ONLY — see m6.
        return _rewrite_block(
            source,
            ["v0_re = tl.where(at_x, 0.0, v0_re)",
             "v0_im = tl.where(at_x, 0.0, v0_im)"],
            ["pass"])

    def m11_curl_parens_flattened(source: str) -> Tuple[str, int]:
        """``stepping._curl_from_operands``' grouping, re-associated."""
        return _rewrite_block(
            source, ["t0_re = ((c_y_re - c_re) + (b_re - b_z_re))"],
            ["t0_re = (c_y_re - c_re + b_re) - b_z_re"])

    def m12_prev_read_after_the_store(source: str) -> Tuple[str, int]:
        """The one ordering the constitutive cannot survive being wrong about."""
        return _rewrite_block(
            source,
            ["prev_re = tl.load(w + 2 * dst, mask=mask, other=0.0)",
             "prev_im = tl.load(w + 2 * dst + 1, mask=mask, other=0.0)",
             "tl.store(w + 2 * dst, ghost_re, mask=mask)",
             "tl.store(w + 2 * dst + 1, ghost_im, mask=mask)"],
            ["tl.store(w + 2 * dst, ghost_re, mask=mask)",
             "tl.store(w + 2 * dst + 1, ghost_im, mask=mask)",
             "prev_re = tl.load(w + 2 * dst, mask=mask, other=0.0)",
             "prev_im = tl.load(w + 2 * dst + 1, mask=mask, other=0.0)"])

    def m13_far_fill_dropped(source: str) -> Tuple[str, int]:
        """``fill_folded_far_ghosts_B`` not carried on one component."""
        return _rewrite_block(
            source,
            ["gy_re, gy_im = _mul_imag_coefficient_left(d1r, d1i, v0_re, v0_im,",
             "EXPANSION)",
             "_carry_ghost_complex(f0, w0, h0, idx + df_y, gy_re, gy_im,",
             "kp_0, km_0, own0 & far_j, EXPANSION)"],
            ["pass"])

    def m14_near_fill_dropped(source: str) -> Tuple[str, int]:
        """``fill_symmetry_bc_B`` not carried on one component."""
        return _rewrite_block(
            source,
            ["gn_re, gn_im = _mul_imag_coefficient_left(n1r, n1i, v1_re, v1_im,",
             "EXPANSION)",
             "_carry_ghost_complex(f1, w1, h1, idx + dn_y, gn_re, gn_im,",
             "kp_d1, km_d1, own1 & near_j, EXPANSION)"],
            ["pass"])

    def m15_ghost_B_store_dropped(source: str) -> Tuple[str, int]:
        """The ghost's own ``B`` store removed: ``H`` is written and ``B`` is not.

        The fills write the FIELD; ``update_H`` then reads it. A carry that ran the
        constitutive and skipped the field store leaves the next step's curl reading
        a stale plane.
        """
        return _rewrite_block(
            source,
            ["tl.store(f + 2 * dst, ghost_re, mask=mask)",
             "tl.store(f + 2 * dst + 1, ghost_im, mask=mask)"],
            ["pass"])


    # ------------------------------------------------------------------
    # THE THREE THINGS ONLY THE BETA WELD HAS. The fifteen mutations above
    # are the beta-less magnetic weld's, unchanged, because the insert is
    # the WHOLE delta between the two bodies (the transcription leg proves
    # that as an order-preserving deletion equality); these three are what
    # only this gate can arm.
    # ------------------------------------------------------------------

    def m_beta_insert_dropped(source: str) -> Tuple[str, int]:
        """The K3b insert compiled out. The guard is made unsatisfiable
        rather than the statements deleted, so the rewrite is a minimal
        one-token arming; every case carries beta != 0 and seeded center
        partners, so the dropped term must move bytes at step 1.

        THE CONJUNCT IS A CONSTEXPR, and that is the whole care in this
        rewrite. ``EXPANSION`` is a ``tl.constexpr`` arm id (1 or 2, never
        negative), so ``EXPANSION < 0`` is a PYTHON bool at compile time and
        the block is compiled out. An earlier spelling used ``nx < 0``, which
        is a TENSOR comparison: Triton refused the whole kernel with
        ``ValueError('Cannot bitcast data-type of size 32 to data-type of
        size 1')`` — measured on the GPU host GPU 1, 2026-09-02 — so the mutation
        was ARMED by its hit count and measured NOTHING on the device, which
        is the one failure mode `_rewrite_block`'s own comment warns about."""
        return _replace_all(source, "if HAS_BETA:",
                            "if HAS_BETA and (EXPANSION < 0):")

    def m_beta_partners_swapped(source: str) -> Tuple[str, int]:
        """The two CENTER partners exchanged: bp with a, bm with b. The
        array path pairs the +sign coefficient with source 1 (b) on target
        0 and the -sign with source 0 (a) on target 1 (S:770-784); the
        exchange is exact arithmetic on different operands and must
        diverge wherever the two partners differ."""
        source, hits = _replace_all(
            source,
            "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
            "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, a_re, a_im, EXPANSION)")
        source, more = _replace_all(
            source,
            "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
            "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, b_re, b_im, EXPANSION)")
        return source, hits + more

    def m_beta_after_the_masks(source: str) -> Tuple[str, int]:
        """The insert moved AFTER the cell-0 ownership mask. The array path
        adds the beta term BEFORE ``_mask_non_owned_cells`` (S:356-363
        against S:369), so on a walled case the mask zeroes the beta
        contribution at the wall plane; the mutant leaves it alive there.
        Scored on the walled case, whose curl1/curl2 masks fire."""
        moved = [
            "if HAS_BETA:",
            "t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
            "curl0_re = curl0_re - t_re",
            "curl0_im = curl0_im - t_im",
            "t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
            "curl1_re = curl1_re - t_re",
            "curl1_im = curl1_im - t_im",
        ]
        source, removed = _rewrite_block(
            source, moved, ["at_x, at_y, at_z = i == 0, j == 0, k == 0"])
        if not removed:
            return source, 0
        # The line the deletion left behind is the mask preamble; the
        # insert is re-planted after the LAST cell-0 mask arm, i.e. before
        # the top-plane mask section.
        source, planted = _rewrite_block(
            source,
            ["last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1"],
            [line if index else "if HAS_BETA:"
             for index, line in enumerate([
                 "if HAS_BETA:",
                 "    t_re, t_im = _mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im, EXPANSION)",
                 "    curl0_re = curl0_re - t_re",
                 "    curl0_im = curl0_im - t_im",
                 "    t_re, t_im = _mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im, EXPANSION)",
                 "    curl1_re = curl1_re - t_re",
                 "    curl1_im = curl1_im - t_im",
                 "last_x, last_y, last_z = i == nx - 1, j == ny - 1, k == nz - 1"])])
        return source, removed if planted else 0

    return (
        # THE ORDERING IS ARITHMETICALLY REAL, MEASURED, and the operand class
        # these two are scored under is what that measurement decided.
        # `parity_chain_associativity_leg` sweeps `_mul_imag_coefficient_left`'s two
        # arms over signed zeros, the subnormal needles and ordinary values and
        # counts 32 of 2048 words moved by a RE-ORDER and 34 by a FOLD, identically
        # on both arms. It is the SIGNED ZEROS that do it: with `c_im` bitwise +0.0,
        # `(c_im * z_im) * -1.0` is a signed zero whose sign depends on `z_im`, and
        # `fma(c_re, z_re, that)` canonicalizes `-0 + +0` to `+0` — so a flip
        # composed with a flip is NOT the composed flip at a zero word.
        # The 2026-08-21 run scored both under `subnormal_band` and both came back
        # UNCAUGHT, which is a statement about that seeding and not about the
        # kernel: a band-seeded state carries subnormals, and this defect needs
        # SIGNED ZEROS at the ghost source lane. `zero_lattice` is the class that
        # puts them there.
        # UNREACHED, not null. The word-layer sweep says both defects are REAL; 30
        # complete steps under the +-0 lattice say this gate's whole-step legs do not
        # reach them. Both halves of that are recorded, in MUTATION_EVIDENCE.
        ("m1_parity_chain_reordered",
         "the far axes applied DESCENDING; the driver applies them ascending. The "
         "word-layer sweep moves 32 of 2048 words on this rewrite, all of them at "
         "signed zeros; 30 complete steps under the +-0 lattice did not put one at "
         "a ghost source lane",
         "unreached", m1_parity_chain_reordered),
        ("m2_parity_chain_folded",
         "two complex multiplies collapsed into one coefficient product; 34 of 2048 "
         "words at the word layer, same class, same 30-step non-reach",
         "unreached", m2_parity_chain_folded),
        # DECLARED UNREACHED 2026-09-02, on this gate's own branch_reachability
        # measurement rather than on an argument: the triple composite needs THREE
        # folded periodic axes, and beta exists only on the effective-2-D grid
        # (grid.py:668-697), so the deepest composition any case here reaches is 3
        # ghost cells per source lane where the beta-less twin reaches 7. The block
        # this rewrite deletes is never compiled, and the beta-less twin's RELEASED
        # gate is what measures the defect real on a 3-D case.
        ("m3_triple_composite_dropped",
         "the deepest destination never written",
         "unreached", m3_triple_composite_dropped),
        ("m4_far_parity_is_the_near_one",
         "+phase where fields.mirror_parity gives -phase on a shift-1 component",
         "caught", m4_far_parity_is_the_near_one),
        ("m5_reflect_row_baked",
         "n - 2 instead of stepping._far_reflect_rows' answer, at an ODD count",
         "caught", m5_reflect_row_baked),
        ("m6_top_plane_mask_dropped",
         "the folded curl's MIRROR_PERIODIC top-plane mask, removed on one axis",
         "caught", m6_top_plane_mask_dropped),
        ("m7_near_source_index_three",
         "the near fill images stored cell 3 rather than MIRROR_SOURCE_INDEX",
         "caught", m7_near_source_index_three),
        ("m8_ownership_mask_not_ANDed",
         "the ownership defect the 2026-08-20 device run found, re-planted",
         "caught", m8_ownership_mask_not_ANDed),
        ("m9_destination_coefficient_reused",
         "the near destination takes the source lane's kps/kms pair; PREDICTED "
         "NULL because a folded axis absorbs on its HIGH face alone, so the two "
         "entries are the same word to the bit",
         "null", m9_destination_coefficient_reused),
        ("m10_wall_clear_dropped",
         "zero_metal_B not carried, on a case with a live wall",
         "caught", m10_wall_clear_dropped),
        ("m11_curl_parens_flattened",
         "the curl's float32 grouping re-associated",
         "caught", m11_curl_parens_flattened),
        ("m12_prev_read_after_the_store",
         "the ghost constitutive reads its workspace AFTER writing it",
         "caught", m12_prev_read_after_the_store),
        ("m13_far_fill_dropped",
         "fill_folded_far_ghosts_B not carried on component 0",
         "caught", m13_far_fill_dropped),
        ("m14_near_fill_dropped",
         "fill_symmetry_bc_B not carried on component 1",
         "caught", m14_near_fill_dropped),
        ("m15_ghost_B_store_dropped",
         "the ghost's field store removed while its H accumulation stays",
         "caught", m15_ghost_B_store_dropped),
        ("m_beta_insert_dropped",
         "the K3b beta term compiled out; beta != 0 on every case",
         "caught", m_beta_insert_dropped),
        ("m_beta_partners_swapped",
         "the two center partners exchanged between the two beta products",
         "caught", m_beta_partners_swapped),
        ("m_beta_after_the_masks",
         "the beta term applied after the ownership mask; the array path "
         "masks the beta contribution at the wall plane and the mutant "
         "leaves it alive",
         "caught", m_beta_after_the_masks),
    )

MUTATION_CASE: Dict[str, str] = {
    # ---- the fifteen the BETA-LESS MAGNETIC weld's released gate scores, on the
    # cases that gate measured them on. Two of its sixteen cases are 3-D and one
    # is off-diagonal; beta admits neither, so the two entries that named a 3-D
    # case are re-declared below rather than silently re-scored on a 2-D one.
    "m1_parity_chain_reordered": "xy_mixed_phase_periodic",
    "m2_parity_chain_folded": "xy_mixed_phase_periodic",
    # m3_triple_composite_dropped names NO case: the triple composite needs three
    # folded PERIODIC axes and beta is effective-2-D by construction
    # (grid.py:668-697), so branch_reachability_leg's own measurement — deepest
    # composition 3, not 7 — says the block it rewrites is never compiled here.
    # It is declared UNREACHED with that evidence rather than scored on a case
    # that cannot arm it.
    "m4_far_parity_is_the_near_one": "y_fold_periodic",
    "m5_reflect_row_baked": "y_fold_periodic_odd",
    "m6_top_plane_mask_dropped": "y_fold_periodic",
    "m7_near_source_index_three": "y_fold_metallic",
    "m8_ownership_mask_not_ANDed": "xy_mixed_phase_periodic",
    # MEASURED, not chosen: at an EVEN full count the far source row and the
    # destination row hold the BITWISE SAME absorber coefficient (the not-owned
    # ghost slot copies its neighbour), so exchanging one for the other is a
    # no-op and this real defect reads UNCAUGHT. Re-measured on this gate's own
    # grids 2026-09-02: on `y_fold_periodic` kps_y_h is np.float32(1.7343807) at
    # BOTH row 18 and row 19. `y_fold_periodic_odd` is the case where the two
    # rows differ, and `_assert_the_far_coefficient_actually_moves` re-measures
    # that on the grid the device leg will read rather than trusting this note.
    "m9_destination_coefficient_reused": "y_fold_periodic_odd",
    "m10_wall_clear_dropped": "y_fold_periodic_wall_x",
    "m11_curl_parens_flattened": "y_fold_periodic",
    "m12_prev_read_after_the_store": "y_fold_periodic",
    "m13_far_fill_dropped": "y_fold_periodic",
    "m14_near_fill_dropped": "y_fold_metallic",
    "m15_ghost_B_store_dropped": "y_fold_periodic",
    # ---- the three only this gate has. Any case reaches the insert (every case
    # here carries beta != 0); the after-the-masks one needs a live WALL, or the
    # mask it is moved past never fires and the mutant agrees by construction.
    "m_beta_insert_dropped": "y_fold_periodic",
    "m_beta_partners_swapped": "y_fold_periodic",
    "m_beta_after_the_masks": "y_fold_periodic_wall_x",
}

DEFAULT_MUTATION_CASE = "y_fold_periodic"

#: Per mutation, the value class its leg runs under. Default ``uniform``.
#:
#: THE TWO ORDERING MUTATIONS ARE SCORED ON THE SIGNED-ZERO LATTICE, and the class
#: was chosen BY MEASUREMENT: :func:`parity_chain_associativity_leg` sweeps the
#: shipped multiply's two arms and counts 32 of 2048 words moved by a RE-ORDER and
#: 34 by a FOLD, every one of them at a signed zero — on every NORMAL operand the
#: parity multiply is an exact sign flip and the composition commutes to the bit.
#: ``uniform`` provably draws no signed zero, so scoring them there would report a
#: real defect as UNCAUGHT.
MUTATION_VALUE_CLASS: Dict[str, str] = {
    "m1_parity_chain_reordered": "zero_lattice",
    "m2_parity_chain_folded": "zero_lattice",
}

#: Per-mutation step budget. The two ordering mutations run for ONE step, on the
#: lattice: the ghost SOURCE lane must hold a signed zero at the moment of the
#: launch, and after step 1 a driven field has moved every cell away from zero.
MUTATION_STEPS_FOR: Dict[str, int] = {
    "m1_parity_chain_reordered": 1,
    "m2_parity_chain_folded": 1,
}

#: Whether a mutation's leg injects in the seam. The two ordering mutations run
#: QUIET, because a deposit is exactly what drives the lattice away from the zeros
#: they need.
MUTATION_ELECTRIC: Dict[str, bool] = {
    "m1_parity_chain_reordered": False,
    "m2_parity_chain_folded": False,
}

#: Per mutation: axes that must be FOLDED on the scored case.
MUTATION_REQUIRES_FOLD: Dict[str, Tuple[str, ...]] = {
    "m1_parity_chain_reordered": ("X", "Y"),
    "m2_parity_chain_folded": ("X", "Y"),
    "m8_ownership_mask_not_ANDed": ("X", "Y"),
    "m7_near_source_index_three": ("Y",),
    "m10_wall_clear_dropped": ("Y",),
    "m14_near_fill_dropped": ("Y",),
    "m_beta_after_the_masks": ("Y",),
}

#: Per mutation: axes that must be folded over a PERIODIC outer declaration, or the
#: FAR blocks and the top-plane mask are compile-time absent.
MUTATION_REQUIRES_PERIODIC_FOLD: Dict[str, Tuple[str, ...]] = {
    "m4_far_parity_is_the_near_one": ("Y",),
    "m5_reflect_row_baked": ("Y",),
    "m6_top_plane_mask_dropped": ("Y",),
    "m9_destination_coefficient_reused": ("Y",),
    "m13_far_fill_dropped": ("Y",),
    "m8_ownership_mask_not_ANDed": ("X",),
}

#: Per mutation: axes that must carry a live WALL.
MUTATION_REQUIRES_WALL: Dict[str, Tuple[str, ...]] = {
    "m10_wall_clear_dropped": ("X",),
    "m_beta_after_the_masks": ("X",),
}

#: Per mutation: axes whose mirror plane must be ODD (phase -1). An even plane
#: multiplies by +1, so a dropped or duplicated parity is invisible there.
MUTATION_REQUIRES_ODD_PARITY: Dict[str, Tuple[str, ...]] = {}

#: Per mutation: the full count of a named axis must be ODD, so a baked ``n - 2``
#: reflect row is wrong rather than right by coincidence.
MUTATION_REQUIRES_ODD_FULL_COUNT: Dict[str, Tuple[str, ...]] = {
    "m5_reflect_row_baked": ("Y",),
}

#: Per mutation: the FAR destination's constitutive coefficient pair must actually
#: DIFFER from its source lane's, or exchanging the two is a bitwise no-op and the
#: leg reports a real defect as UNCAUGHT. A property of the ABSORBER'S GRADING, so
#: it is MEASURED on the grid the case builds rather than declared.
MUTATION_REQUIRES_MOVING_FAR_COEFFICIENT: Dict[str, Tuple[str, ...]] = {
    "m9_destination_coefficient_reused": ("Y",),
}


def _assert_the_far_coefficient_actually_moves(case, axis_letter: str) -> None:
    """The far destination's kps/kms pair must not equal its source's.

    Built on the CASE'S OWN GRID rather than argued from the shape: the reflect row
    comes from ``stepping._far_reflect_rows`` and the coefficients from the layer the
    case declares, so this is the same pair the device leg will read.
    """
    from meep_gpu import stepping  # noqa: PLC0415

    axis = "XYZ".index(axis_letter.upper())
    driver = build_grid(case, prefer_gpu=False)
    try:
        grid = driver.fields.grid
        rows = stepping._far_reflect_rows(grid)
        row = rows[axis] if rows is not None else None
        if row is None:
            raise AssertionError(
                f"case {case[0]!r}: axis {axis_letter} has no far reflect row, so "
                f"there is no far destination whose coefficient could move")
        top = int(grid.shape[axis]) - 1
        for stem in ("kps", "kms"):
            values = np.asarray(
                getattr(driver.pml, f"{stem}_{'xyz'[axis]}_h")).reshape(-1)
            if (np.float32(values[int(row)]).view(np.uint32)
                    == np.float32(values[top]).view(np.uint32)):
                raise AssertionError(
                    f"case {case[0]!r}: {stem}_{'xyz'[axis]}_h is BITWISE EQUAL at "
                    f"the far source row {int(row)} and the destination row {top} "
                    f"({values[int(row)]!r}); swapping one for the other is a no-op "
                    f"and the leg would report a real defect as uncaught")
    finally:
        driver.close()


def mutation_case_for(name: str) -> Tuple[str, Tuple[Any, ...]]:
    """The case a mutation is scored on, and it must really carry what it needs."""
    case_name = MUTATION_CASE.get(name, DEFAULT_MUTATION_CASE)
    case = CASES_BY_NAME[case_name]
    _name, _dimensions, cell, boundaries, mirrors, _k, _pml, _steps = case
    folded = {axis.upper() for axis, _phase in mirrors}
    required = set(MUTATION_REQUIRES_FOLD.get(name, ()))
    if not required <= folded:
        raise AssertionError(
            f"mutation {name} rewrites lines guarded on folds {sorted(required)} but "
            f"is scored on case {case_name!r}, which folds {sorted(folded)}: the "
            f"rewritten lines would be a DEAD BRANCH and the leg would report "
            f"'uncaught' while measuring nothing")

    def declaration(axis_letter: str) -> str:
        if isinstance(boundaries, str):
            return boundaries
        return str(boundaries.get(axis_letter.lower(), "periodic"))

    for axis_letter in MUTATION_REQUIRES_PERIODIC_FOLD.get(name, ()):
        if axis_letter.upper() not in folded:
            raise AssertionError(
                f"mutation {name} needs a folded PERIODIC {axis_letter} axis and case "
                f"{case_name!r} does not fold {axis_letter} at all")
        if declaration(axis_letter) != "periodic":
            raise AssertionError(
                f"mutation {name} rewrites a FAR_{axis_letter.upper()} block, which "
                f"is compile-time absent unless {axis_letter} is folded over a "
                f"PERIODIC outer declaration; case {case_name!r} declares "
                f"{declaration(axis_letter)!r} there")
    for axis_letter in MUTATION_REQUIRES_WALL.get(name, ()):
        if declaration(axis_letter) != "metallic" or axis_letter.upper() in folded:
            raise AssertionError(
                f"mutation {name} rewrites a wall clear on {axis_letter}, and case "
                f"{case_name!r} declares {declaration(axis_letter)!r} there "
                f"(folded={axis_letter.upper() in folded}); stepping._zero_metal "
                f"skips a folded axis, so no clear would run")
    phases = {axis.upper(): int(phase) for axis, phase in mirrors}
    for axis_letter in MUTATION_REQUIRES_ODD_PARITY.get(name, ()):
        if phases.get(axis_letter.upper()) != -1:
            raise AssertionError(
                f"mutation {name} is only visible on an ODD {axis_letter} plane; case "
                f"{case_name!r} declares phase {phases.get(axis_letter.upper())!r}")
    for axis_letter in MUTATION_REQUIRES_ODD_FULL_COUNT.get(name, ()):
        axis = "XYZ".index(axis_letter.upper())
        full = int(round(12.0 * float(cell[axis])))
        if full % 2 == 0:
            raise AssertionError(
                f"mutation {name} bakes a reflect row that is CORRECT at an even full "
                f"count, and case {case_name!r} has full count {full} on "
                f"{axis_letter}")
    for axis_letter in MUTATION_REQUIRES_MOVING_FAR_COEFFICIENT.get(name, ()):
        _assert_the_far_coefficient_actually_moves(case, axis_letter)
    return case_name, case


def run_mutations(cp, product, probe, pristine_ptx: Sequence[str]
                  ) -> List[Dict[str, Any]]:
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
        kernel_name = f"mutant_{index}_folded_beta_complex_D"
        try:
            mutant = compile_mutant(mutated, kernel_name)
        except Exception as exc:  # noqa: BLE001 - an uncompilable mutant is a failure
            row["error"] = f"the mutant did not import: {exc!r}"
            rows.append(row)
            log(f"  mutation {name}: DID NOT COMPILE")
            continue
        case_name, case = mutation_case_for(name)
        row["case"] = case_name
        # THE CARRY IS ON BY DEFAULT. A mutation scored on the quiet family would
        # leave the ghost-repair interaction unmeasured on every leg but the carry
        # ones. The three tables above are the measured exceptions.
        value_class = MUTATION_VALUE_CLASS.get(name, "uniform")
        steps = MUTATION_STEPS_FOR.get(name, MUTATION_STEPS)
        in_seam = MUTATION_ELECTRIC.get(name, True)
        row["value_class"] = value_class
        row["steps"] = steps
        row["in_seam_source"] = in_seam
        leg = run_leg(cp, f"mutation:{name}", case, steps, product, probe,
                      value_class=value_class, mutant=mutant, in_seam=in_seam)
        row["leg"] = leg
        row["caught"] = leg.get("first_divergence") is not None
        row["fused_kernel_launches"] = leg.get("fused_kernel_launches")
        row["ptx_specializations"] = leg.get("ptx_specializations")
        row["pristine_ptx_specializations"] = len(pristine_ptx)
        rows.append(row)
        log(f"  mutation {name}: caught={row['caught']} "
            f"(expected {expectation}) on {case_name}")
    return rows


# ---------------------------------------------------------------------------
# Refusals — the predicate, on the device's own objects
# ---------------------------------------------------------------------------

def run_refusals(cp, product, probe) -> List[Dict[str, Any]]:
    """Configurations the product must refuse — and two it must ADMIT."""
    rows: List[Dict[str, Any]] = []
    base = CASES_BY_NAME["y_fold_periodic"]
    for name, mutate, needle, admit in (
        # THE REFUSING DIRECTION IS THE MAGNETIC ONE. On this seam the driver
        # injects the electric currents in the OTHER half of the step, so an
        # electric source — index or not — is quiet at this clause; the row below
        # asserts exactly that, and the table's refusing row uses a MAGNETIC
        # source with no deposit index. MEASURED 2026-09-02: with the electric
        # fixture in the refusing slot this row read passed=False, because the
        # predicate returned no reasons at all.
        ("magnetic_source_without_a_deposit_index",
         lambda driver: (_MagneticWithoutIndex(),),
         "does not publish the index", False),
        ("undeclared_source_list", lambda driver: None,
         "was not declared", False),
        ("magnetic_source_with_a_deposit_index",
         lambda driver: (_Deposit(driver.fields, "Hz",
                                  deposit_center(base)),), None, True),
        ("electric_source_only_reaches_the_other_seam",
         lambda driver: (_Electric(),), None, True),
        # NOT SPELLED "STENCIL" ON THIS SIDE, and the difference is where the
        # refusal arrives rather than whether it does. The electric twin takes an
        # off-diagonal row on its CONSTITUTIVE half, where the row product IS the
        # stencil and the reason says so; ``update_H`` binds no chi1inv at all, so
        # here the CURL half refuses first and its reason names the neighbour read.
        # No beta variant of the off-diagonal arms is shipped either way, and the
        # corpus carries no off-diagonal beta row.
        ("offdiagonal_chi1inv_row",
         lambda driver: (_Deposit(driver.fields, "Hz", deposit_center(base)),),
         "off-diagonal chi1inv row is installed", False),
    ):
        driver = build_grid(base)
        try:
            index = np.arange(int(np.prod(driver.shape)),
                              dtype=np.float32).reshape(driver.shape)
            epsilon = np.ascontiguousarray(
                (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
            if name == "offdiagonal_chi1inv_row":
                partner = np.ascontiguousarray(
                    (0.04 * np.cos(index * np.float32(0.019))).astype(np.float32))
                driver.set_epsilon_components(
                    {key: cp.asarray(epsilon) for key in ("Ex", "Ey", "Ez")},
                    chi1inv_offdiagonal={"Ex": {"Ey": cp.asarray(partner)},
                                         "Ey": {"Ex": cp.asarray(partner)}})
            else:
                driver.set_epsilon(cp.asarray(epsilon))
            sources = mutate(driver)
            verdict = product.folded_beta_complex_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, sources, probe=probe)
            built = product.plan_folded_beta_complex_fused_magnetic_pair(
                driver.fields, driver.pml, sources, probe=probe)
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

    # A ZERO-BETA folded complex grid belongs to the beta-less twin; the
    # inverted clause must refuse it by name. Built directly because
    # build_grid bakes BETA_CORPUS into every case.
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    _name, dimensions, cell, boundaries, mirrors, k_point, pml_spec, _s = base
    driver = FdtdDriver(
        cell_size=cell, resolution=12.0, dimensions=dimensions,
        force_complex_fields=True, courant=0.35, boundaries=boundaries,
        k_point=k_point, beta=0.0,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
        prefer_gpu=True, gpu_id=0)
    try:
        driver.setup_pml(dict(pml_spec))
        sources = (_Deposit(driver.fields, "Hz", deposit_center(base)),)
        verdict = product.folded_beta_complex_fused_magnetic_pair_coverage(
            driver.fields, driver.pml, sources, probe=probe)
        built = product.plan_folded_beta_complex_fused_magnetic_pair(
            driver.fields, driver.pml, sources, probe=probe)
        row = {"refusal": "zero_beta_grid", "covered": bool(verdict.covered),
               "reasons": list(verdict.reasons),
               "plan_is_None": built is None, "admits": False,
               "passed": ((not verdict.covered) and built is None and any(
                   "beta" in reason for reason in verdict.reasons))}
        rows.append(row)
        log(f"  refusal zero_beta_grid: passed={row['passed']}")
    finally:
        driver.close()
    return rows


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def environment(cp: Any = None) -> Dict[str, Any]:
    import platform  # noqa: PLC0415

    out: Dict[str, Any] = {
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
            out["device_name"] = cp.cuda.runtime.getDeviceProperties(
                device.id)["name"].decode()
        except Exception as exc:  # noqa: BLE001
            out["cupy"] = f"unavailable: {exc!r}"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        HERE, "results", "triton_folded_beta_complex_fused_magnetic_pair"),
        help="a DIRECTORY; gate.json is written inside it")
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. A run that installs NOTHING is a MIXED "
             "configuration attributable to no policy at all, and a keep-cut licence "
             "read by such a run is a broken comparison.")
    args = parser.parse_args(argv)

    out = args.out
    if out.endswith(".json"):
        artifact = out
    else:
        os.makedirs(out, exist_ok=True)
        artifact = os.path.join(out, "gate.json")

    import triton_device_identity  # noqa: PLC0415
    payload: Dict[str, Any] = {
        "gate": "triton_folded_beta_complex_fused_magnetic_pair",
        "product": "meep_gpu.triton_kernels.folded_beta_complex_fused_magnetic_pair",
        "kernel": KERNEL_NAME,
        "replaces": list(SEAM_PASSES),
        "seed_root": SEED_ROOT,
        "source_sha256": source_hashes(),
        "environment": triton_device_identity.record(environment()),
        "policy": {"num_warps": 1, "enable_fp_fusion": False,
                   "block": "complex_fields.DEFAULT_BLOCK",
                   "subnormal_policy": args.subnormal_policy},
        "value_classes": list(VALUE_CLASSES),
        "no_device_legs": [],
        "device_legs": [],
        "mutations": [],
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

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep and
    # quietly did not is a process whose bytes mean nothing — and on this family it
    # is worse than that, because the EXPANSION licence it consumes is
    # policy-conditional.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import complex_fields, folded_complex  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_beta_complex_fused_magnetic_pair as product  # noqa: PLC0415

    # THE PROBE THE DEVICE LEGS CONSUME, resolved once and RECORDED. The device bytes
    # are a function of the arm, so an artifact that does not name the record that
    # licensed it cannot be read.
    probe = complex_fields.load_expansion_probe()
    licence = folded_complex.parity_expansion_license(probe)
    beta_licence = folded_complex.folded_beta_expansion_license(probe)
    payload["expansion"] = {
        "probe_path": os.environ.get("MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
        "pattern_set": list(product.PRODUCT_PROBE_PATTERNS),
        "arm": licence["arm"], "basis": licence["basis"],
        "expansion": licence["expansion"], "refusals": list(licence["refusals"]),
        "policy_resolved": licence.get("policy_resolved"),
        "candidate_policy": licence.get("candidate_policy"),
        "beta_arm": beta_licence["arm"],
        "beta_expansion": beta_licence["expansion"],
        "beta_refusals": list(beta_licence["refusals"]),
    }
    if licence["expansion"] != beta_licence["expansion"]:
        payload["device_status"] = "REFUSED BEFORE THE FIRST LEG"
        payload["passed"] = False
        payload["release"] = {"released": False, "reasons": [
            "the two extended licences disagree — parity "
            f"{licence['expansion']!r} against beta "
            f"{beta_licence['expansion']!r} — and the kernel binds ONE "
            "EXPANSION constexpr: "
            + "; ".join(list(licence["refusals"])
                        + list(beta_licence["refusals"]))]}
        save(payload, artifact)
        log(f"\nREFUSED: licences disagree")
        return 1
    if licence["expansion"] is None:
        payload["device_status"] = "REFUSED BEFORE THE FIRST LEG"
        payload["passed"] = False
        payload["release"] = {"released": False, "reasons": [
            "no EXPANSION arm is licensed for this run, so no device leg could "
            "measure anything: " + "; ".join(licence["refusals"])]}
        save(payload, artifact)
        log(f"\nREFUSED: {licence['refusals']}")
        return 1
    log(f"expansion arm licensed: {licence['arm']} (basis {licence['basis']})")

    backends.guard_kernel_compilation(cp)
    payload["environment"] = triton_device_identity.record(environment(cp))
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "quiet_cases": {case[0]: case[7] for case in CASES},
        "value_classes": list(VALUE_CLASSES),
        "carry_cases": list(CARRY_CASES),
        "steps_per_carry": CARRY_STEPS,
        "steps_per_mutation": MUTATION_STEPS,
    }
    save(payload, artifact)

    log("\n=== device legs: the QUIET family (a magnetic source, no seam deposit) ===")
    for case in CASES:
        for value_class in VALUE_CLASSES:
            steps = case[7]
            row = run_leg(cp, f"quiet:{case[0]}/{value_class}", case, steps, product,
                          probe, value_class=value_class)
            passed, failures = verdict_of(row, require_launches=steps)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== device legs: the CARRY family (a real MAGNETIC deposit in the seam) ===")
    for case_name in CARRY_CASES:
        case = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{case_name}/{value_class}", case, CARRY_STEPS,
                          product, probe, value_class=value_class, in_seam=True)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        case = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{case_name}", case, CARRY_STEPS, product,
                      probe, in_seam=True, bracket=False)
        # REQUIRES DIVERGENCE. A bracket that changes nothing is not load-bearing, and
        # a carry family whose unbracketed twin agreed would be measuring a seam that
        # carried no deposit.
        #
        # THE LAUNCH FLOOR IS "AT LEAST ONE", NOT THE BUDGET, and the vacuity census
        # is off: a leg that must diverge STOPS at the first divergent step, so an
        # exact launch count and a whole-run moved-state census are statements about
        # steps that never ran. What remains checkable is that the fused kernel ran at
        # all, and that the ORACLE control did NOT diverge — which `verdict_of` checks
        # unconditionally, and which is what says the divergence is the missing
        # bracket rather than a broken harness.
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
    case = CASES_BY_NAME["y_fold_periodic"]
    row = run_leg(cp, "armed:no_substitution", case, 3, product, probe,
                  install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, artifact)

    row = run_leg(cp, "armed:frozen_electric_seam", case, 3, product, probe,
                  freeze_seam=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(
        row.get("inert_here_but_moving_on_the_array_path")
        or not row.get("seam_outputs_moved"))
    row["why"] = ("every route's B->H seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== the IDENTITY leg: HAS_BETA = 0 against the beta-less weld ===")
    payload["device_legs"].append(identity_has_beta_off_leg(cp, product,
                                                            probe))
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.folded_beta_complex_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, probe, pristine)
    save(payload, artifact)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product, probe)
    save(payload, artifact)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])

    # A DECLARED NULL MUST BE CONFIRMED, not merely permitted: a null that IS caught
    # means the reasoning behind the null is wrong, and that has to fail too. Every
    # mutation must also have been ARMED: a rewrite that hit nothing measured nothing,
    # whatever it then reported.
    def _mutation_ok(row: Dict[str, Any]) -> bool:
        if row.get("error") is not None:
            return False
        if not row.get("rewrite_hits"):
            return False
        if row["expectation"] == "unreached":
            # A real defect this gate's device legs do not reach. It must come back
            # UNCAUGHT — a catch means the declaration is stale and the entry must
            # move to "caught" — and it must name the evidence that the defect is
            # real, or this is a null wearing a longer word.
            row["evidence"] = MUTATION_EVIDENCE.get(row["mutation"])
            return (not row.get("caught")) and bool(row["evidence"])
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
