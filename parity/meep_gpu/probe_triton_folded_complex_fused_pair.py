"""Byte gate for the FOLDED COMPLEX fused ELECTRIC pair: ``step_D`` into ``update_E``.

DEVICE STATUS: **UNRUN.** This module is written to be run on a CUDA host with an
    idle device; nothing in this tree may cite it as a release until an artifact
    exists with ``device_status: RUN`` and ``release.released: true``.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT: for every configuration
:func:`~meep_gpu.triton_kernels.folded_complex_fused_pair.folded_complex_fused_pair_coverage`
admits, ONE launch of ``folded_complex_fused_curl_constitutive_D`` — bracketed by the
SHIPPED deposit repair wherever the seam carries an electric deposit — leaves the
engine in a state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_D`` / the electric injection /
    ``fill_symmetry_bc_D`` / ``zero_metal_D`` / ``fill_folded_far_ghosts_D`` /
    ``update_E``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the folded complex curl
    (``folded_complex.plan_folded_complex_pml_curl`` on ``step_D``), the complex
    mirror ghost fill (``folded_complex.plan_folded_mirror_ghost_fill_complex`` on
    family ``D``, run as TWO passes with the wall clear between them) and the folded
    complex constitutive (``folded_complex.plan_folded_complex_constitutive`` on
    side ``E``), with ``zero_metal_D`` on the array path in both routes because no
    Triton product owns the wall clear,

over every allocated volume, on the uint32 view. ``allclose`` appears nowhere.

WHAT IS NEW HERE AGAINST THE TWO PRODUCTS THIS ONE SITS BETWEEN
===============================================================

The MAGNETIC twin (``folded_complex_fused_magnetic_pair``, released 2026-08-21) has
the same curl arithmetic and the same parity chain; the REAL electric twin
(``folded_fused_pair``, released 2026-08-20/21) has the same D-side fill geometry and
the same deposit bracket. What only this product has is their intersection, and it is
three things:

1. **THE DEPOSIT CARRY ON A FOLDED COMPLEX SEAM.** Both corpus rows declare an
   ELECTRIC source, so ``CARRIES_DEPOSIT_REPAIR`` is the difference between two
   seam-instances and zero. Every CARRY case is run TWICE — once with the shipped
   :class:`~meep_gpu.deposit_repair.LeadingRepairPlan` /
   :class:`~meep_gpu.deposit_repair.TrailingRepairPlan` pair and once WITHOUT — and
   the unbracketed run MUST DIVERGE.
2. **THE WALL CLEAR INSIDE THE PARITY CHAIN.** On the D family ``_zero_metal``'s rows
   ARE the near fill's axes, so a near ghost takes the clear AFTER its parity and any
   far parity lands AFTER the clear. ``m_clear_before_parity`` and
   ``m_clear_dropped_on_the_ghost`` arm exactly that.
3. **THE INVERSE-EPSILON MULTIPLY AT THE GHOST'S OWN INDEX.** ``update_E`` scales by
   ``inverse_epsilon_for(component)`` at the cell it writes; a ghost that took the
   SOURCE lane's coefficient would be a smooth, plausible, wrong field on every
   folded plane, and no comparison restricted to owned cells would see it.
   ``m_ghost_inv_eps_at_the_source`` arms it, and every case is built on a VARYING
   epsilon so the rewrite is not a no-op.

WHAT THE GATE REFUSES TO INFER
==============================

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is counted
  through a proxy that owns the kernel object, every substitution is counted per
  route in the installer, and one ARMED HARNESS MUTATION removes the substitution so
  the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** The moved-state census is
  taken on BOTH the fused route and the array path, and a volume the array path moves
  while this route leaves it inert is a pass this weld swallowed.
* **A bracket that changes nothing is not load-bearing.** See the null control above.
* **A mutation scored on a grid that never enters its branch measures nothing.**
  :data:`MUTATION_CASE` names, per mutation, the case whose fold set, parity, wall
  set and outer declaration actually reach the lines it rewrites, and
  :func:`mutation_case_for` refuses a pairing that does not.
* **A value class the operands cannot contain is a leg that measured nothing.** Every
  device leg records an operand census off the BITS and :data:`CLASS_FLOOR` refuses a
  class whose needles never appeared.

Usage::

    # laptop, no CUDA, no Triton — the legs that need neither
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_complex_fused_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_complex_fused_pair.py \\
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

SEED_ROOT = "triton_folded_complex_fused_pair/2026-08-31"

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
#: THE WALL CASES ARE NOT DECORATION ON THIS FAMILY. ``zero_metal_D`` writes SIX rows
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
    # THREE folded PERIODIC axes, mixed phases: SEVEN ghost cells owned by one source
    # lane — the deepest composition this family can emit, and the only shape that
    # compiles the triple-composite blocks.
    ("xyz_mixed_phase_periodic_3d", 3, (2.0, 2.0, 2.0), "periodic",
     (("X", 1), ("Y", -1), ("Z", 1)), (0.0, 0.0, 0.0),
     {"x": {"high": 3}, "y": {"high": 3}, "z": {"high": 3}}, 6),
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
    ("y_fold_periodic_wall_z_3d", 3, (2.0, 2.0, 2.0),
     {"x": "periodic", "y": "periodic", "z": "metallic"}, (("Y", 1),),
     (0.0, 0.0, 0.0), {"x": 3, "y": {"high": 3}, "z": 3}, 6),
    # BLOCH PHASES, one per axis, each on an UNFOLDED periodic axis. A folded axis
    # may carry none (`driver._require_bloch_is_representable` refuses that
    # configuration outright), so the phase blocks are reachable only like this.
    ("y_fold_periodic_bloch_x", 2, (3.2, 3.0, 0.0), "periodic", (("Y", 1),),
     (0.19, 0.0, 0.0), {"x": 5, "y": {"high": 5}}, 10),
    ("x_fold_periodic_bloch_y", 2, (3.0, 3.2, 0.0), "periodic", (("X", 1),),
     (0.0, 0.23, 0.0), {"x": {"high": 5}, "y": 5}, 10),
    ("y_fold_periodic_bloch_z_3d", 3, (2.0, 2.0, 2.0), "periodic", (("Y", 1),),
     (0.0, 0.0, 0.17), {"x": 3, "y": {"high": 3}, "z": 3}, 6),
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

#: The CARRY family: the same grids run with a real ELECTRIC deposit IN THIS SEAM.
#:
#: THIS IS THE PRODUCT, not an extra. BOTH corpus rows of this board cell declare an
#: electric source, so a gate that only ran the quiet family would certify a kernel
#: the corpus never reaches. One folded METALLIC grid (near fill only), one folded
#: PERIODIC grid at an ODD phase (whose far carry images the deposit onto a second
#: plane), one grid with a WALL beside the fold (where the clear sits inside the
#: chain) and one 3-D grid folding three axes (the deepest composition).
CARRY_CASES: Tuple[str, ...] = (
    "y_fold_metallic", "y_fold_periodic_odd_phase", "y_fold_periodic_wall_x",
    "xyz_mixed_phase_periodic_3d",
)

#: Steps per carry / null-control leg.
CARRY_STEPS = 6

#: Steps per armed mutation. A mutation needing more than this to become byte-visible
#: is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3292-3304).
SEAM_PASSES: Tuple[str, ...] = (
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E",
)

#: THERE ARE TWO VOCABULARIES FOR THIS SEAM AND THEY ARE NOT THE SAME LIST. The
#: driver binds ``fill_symmetry_bc_D``; the RESIDENCY model spells that slot
#: ``fill_D``, and ``REPLACES`` is declared in the residency spelling because that is
#: what a composer reads. Mapping them here rather than asserting them equal is the
#: honest form: :func:`seam_binding_leg` checks the map is a bijection and that each
#: driver name is a real ``driver.py`` call.
RESIDENCY_NAME: Dict[str, str] = {
    "step_D": "step_D",
    "fill_symmetry_bc_D": "fill_D",
    "zero_metal_D": "zero_metal_D",
    "fill_folded_far_ghosts_D": "fill_folded_far_ghosts_D",
    "update_E": "update_E",
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
SEAM_OUTPUTS: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
    "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
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
        "meep_gpu/triton_kernels/folded_complex_fused_pair.py",
        "meep_gpu/triton_kernels/folded_complex_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/folded_fused_pair.py",
        "meep_gpu/triton_kernels/folded_complex.py",
        "meep_gpu/triton_kernels/complex_fields.py",
        "meep_gpu/triton_kernels/complex_fused_electric_pair.py",
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
        "meep_gpu/test_triton_folded_complex_fused_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# Reading the shipped kernel text
# ---------------------------------------------------------------------------

_MODULE_FILE = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                            "folded_complex_fused_pair.py")

KERNEL_NAME = "folded_complex_fused_curl_constitutive_D"
CARRY_HELPER = "_carry_ghost_complex_E"


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
    "a_re = a_re + t_re",
    "a_re = a_re - t_re",
)

#: The D-side lines this product takes from the REAL folded electric pair rather than
#: from the complex magnetic one: the near/far lane predicates and the destination
#: offsets, which are the D geometry and are spelled identically in both.
D_GEOMETRY_LINES = (
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
    "top_x = idx * 0 + (nx - 1)",
    "top_y = idx * 0 + (ny - 1)",
    "top_z = idx * 0 + (nz - 1)",
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

    constitutive = _shipped_text("complex_fused_curl_constitutive_D",
                                 os.path.join(package,
                                              "complex_fused_electric_pair.py"))
    for line in CONSTITUTIVE_LINES:
        if line not in constitutive:
            findings.append(f"the certified complex electric pair no longer spells "
                            f"{line!r}")
        if line not in ours:
            findings.append(f"the constitutive half no longer spells {line!r}")

    real = _shipped_text("folded_fused_curl_constitutive_D",
                         os.path.join(package, "folded_fused_pair.py"))
    for line in D_GEOMETRY_LINES:
        if line not in real:
            findings.append(f"the released REAL folded electric pair no longer "
                            f"spells {line!r}; the D geometry cannot be checked")
        if line not in ours:
            findings.append(f"the D geometry no longer spells {line!r}")

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
    order = ("tl.store(f + 2 * dst, ghost_re",
             "prev_re = tl.load(w + 2 * dst",
             "src_re, src_im = _mul_field_left(",
             "tl.store(w + 2 * dst, src_re",
             "acc_re = tl.load(e + 2 * dst",
             "_mul_coefficient_left(kp_d, src_re, src_im, EXPANSION)",
             "_mul_coefficient_left(km_d, prev_re, prev_im, EXPANSION)",
             "tl.store(e + 2 * dst, acc_re")
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
            f"must be read BEFORE the w store and the inv_eps multiply must sit "
            f"between them")

    # THE INV_EPS INDEX. A word index at either site is the next cell's coefficient.
    if "tl.load(ie + dst, mask=mask, other=0.0)" not in helper:
        findings.append("the ghost no longer reads inv_eps at the COMPLEX cell index")
    if "ie + 2 * dst" in helper:
        findings.append("the ghost reads inv_eps at a WORD index")
    for component in range(3):
        if (f"tl.load(ie{component} + idx, mask=own{component}, other=0.0)"
                not in ours):
            findings.append(f"the owned cell no longer reads ie{component} at + idx")
        if f"ie{component} + 2 * idx" in ours:
            findings.append(f"ie{component} is read at a WORD index")

    # ONLY THE LICENSED MULTIPLY HELPERS ARE CALLED.
    from meep_gpu.triton_kernels import folded_complex_fused_pair as module  # noqa: PLC0415

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
            "d_geometry_lines": len(D_GEOMETRY_LINES),
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

    Nothing here computes a parity or a ghost value; a leg that re-implemented the
    kernel would mirror its defects instead of executing them.
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
                f"block {ast.unparse(statement.test)!r} carries {len(calls)} ghosts")
        chain, clears = [], []
        for inner in statement.body:
            if (isinstance(inner, ast.Assign)
                    and isinstance(inner.value, ast.Call)
                    and getattr(inner.value.func, "id", "")
                    == "_mul_imag_coefficient_left"):
                chain.append(COEFFICIENT_ARGUMENTS[inner.value.args[0].id])
            if isinstance(inner, ast.If) and ast.unparse(inner.test).startswith("ZM_"):
                clears.append((len(chain), ast.unparse(inner.test)))
        call = calls[0]
        out.append({
            "guard": ast.unparse(statement.test),
            "target": call.args[0].id,
            "inverse": call.args[3].id,
            "destination": ast.unparse(call.args[4]),
            "coefficient": (f"{ast.unparse(call.args[7])}, "
                            f"{ast.unparse(call.args[8])}"),
            "mask": ast.unparse(call.args[9]),
            "chain": tuple(chain),
            "clears": tuple(clears),
        })
    return out


def parity_chain_leg() -> Dict[str, Any]:
    """The emitted blocks against :func:`carried_destinations` x :func:`parity_chain`.

    THE ORDER IS THE DRIVER'S AND IS TRANSCRIBED, NEVER CHOSEN — near applications
    innermost and ascending, then the wall clear, then the far one. Under complex
    storage re-ordering a two-parity chain moves bytes and folding it into one
    coefficient moves more, both measured; this leg is what says the shipped body
    carries the order it claims.
    """
    from meep_gpu.triton_kernels import folded_complex_fused_pair as module  # noqa: PLC0415

    findings: List[str] = []
    blocks = carry_blocks()
    rows: List[Dict[str, Any]] = []
    if len(blocks) != 21:
        findings.append(
            f"the kernel emits {len(blocks)} carry blocks; three components x seven "
            f"destinations is 21")
    seen = set()
    for component in range(3):
        target = f"f{component}"
        near_axes = module.NEAR_FILL_AXES[component]
        far_axes = module.FAR_FILL_AXES[component]
        for near_subset, far_subset in module.carried_destinations(near_axes,
                                                                   far_axes):
            expected = module.parity_chain(near_subset, far_subset)
            matches = [block for block in blocks
                       if block["target"] == target and block["chain"] == expected]
            if len(matches) != 1:
                findings.append(
                    f"{target} destination near={near_subset} far={far_subset} "
                    f"expects chain {expected} and {len(matches)} blocks carry it")
                continue
            block = matches[0]
            seen.add((block["target"], block["guard"]))
            want_destination = " + ".join(
                ["idx"] + [DF[axis] for axis in far_subset]
                + [DN[axis] for axis in near_subset])
            if block["destination"] != want_destination:
                findings.append(
                    f"{target} {expected}: destination {block['destination']!r} is "
                    f"not {want_destination!r}")
            want_mask = " & ".join(
                [f"own{component}"] + [FAR_LANE[axis] for axis in far_subset]
                + [NEAR_LANE[axis] for axis in near_subset])
            if block["mask"] != want_mask:
                findings.append(
                    f"{target} {expected}: mask {block['mask']!r} is not "
                    f"{want_mask!r}")
            # THE COEFFICIENT ROW MOVES FOR THE FAR HALF ONLY on this family.
            want_coefficient = (f"kp_f{component}, km_f{component}" if far_subset
                                else f"kp_{component}, km_{component}")
            if block["coefficient"] != want_coefficient:
                findings.append(
                    f"{target} {expected}: coefficient {block['coefficient']!r} is "
                    f"not {want_coefficient!r}")
            if block["inverse"] != f"ie{component}":
                findings.append(
                    f"{target} {expected}: binds {block['inverse']!r}, not "
                    f"ie{component}")
            # THE CLEAR LANDS AFTER THE NEAR PARITIES AND BEFORE ANY FAR ONE.
            if near_subset:
                if len(block["clears"]) != 2:
                    findings.append(
                        f"{target} {expected}: {len(block['clears'])} clear guards, "
                        f"expected the component's two")
                for position, flag in block["clears"]:
                    if position != len(near_subset):
                        findings.append(
                            f"{target} {expected}: clear {flag} lands after "
                            f"{position} parities, not {len(near_subset)}")
            elif block["clears"]:
                findings.append(
                    f"{target} {expected}: a FAR-ONLY ghost carries a clear "
                    f"{block['clears']}; the value it images already took one")
            rows.append({"target": target, "guard": block["guard"],
                         "chain": [list(entry) for entry in block["chain"]],
                         "destination": block["destination"],
                         "coefficient": block["coefficient"],
                         "clears": [list(entry) for entry in block["clears"]]})
    unmatched = {(block["target"], block["guard"]) for block in blocks} - seen
    if unmatched:
        findings.append(f"blocks the enumeration does not name: {sorted(unmatched)}")
    # NOT FOLDED, NOT RE-ORDERED.
    for block in blocks:
        if len(block["chain"]) != len(set(block["chain"])):
            findings.append(f"block {block['guard']!r} repeats a coefficient")
        passes = [entry[1] for entry in block["chain"]]
        if passes != sorted(passes, key=lambda name: 0 if name == "near" else 1):
            findings.append(f"block {block['guard']!r} applies a far parity before a "
                            f"near one")
        near = [axis for axis, name in block["chain"] if name == "near"]
        if near != sorted(near):
            findings.append(f"block {block['guard']!r} applies its near axes out of "
                            f"ascending order")
    return {"leg": "parity_chain", "device": False, "blocks": rows,
            "block_count": len(blocks),
            "clear_after_near": bool(
                __import__("meep_gpu.triton_kernels.folded_complex_fused_pair",
                           fromlist=["x"]).CLEAR_AFTER_NEAR),
            "findings": findings, "passed": not findings}


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
        boundaries=boundaries, k_point=k_point,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors),
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    driver.setup_pml(dict(pml_spec))
    return driver


def ghost_destinations(near: Sequence[Any], far: Sequence[Any]) -> Tuple[int, int, int]:
    """Ghost cells ONE source lane owns, per component, for this constexpr set."""
    from meep_gpu.triton_kernels import folded_complex_fused_pair as module  # noqa: PLC0415

    out: List[int] = []
    for component in range(3):
        near_axes = tuple(axis for axis in module.NEAR_FILL_AXES[component]
                          if bool(near[axis]))
        far_axes = tuple(axis for axis in module.FAR_FILL_AXES[component]
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
    from meep_gpu.triton_kernels import folded_complex_fused_pair as module  # noqa: PLC0415
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
    if never:
        findings.append(
            f"these constexpr guards are TRUE on no case, so the blocks behind them "
            f"are shipped code no device leg executes: {never}")
    # The four boundary codes: each must be taken by some case's BC.
    codes = {table[key] for table in tables.values() for key in ("BCX", "BCY", "BCZ")}
    for required in ("PERIODIC", "MIRROR_METALLIC", "MIRROR_PERIODIC"):
        value = next(iter(tables.values()))[required]
        if value not in codes:
            findings.append(f"no case declares a {required} axis; the curl half's "
                            f"branch for it is never compiled")
    depth = max(max(table["ghost_destinations"]) for table in tables.values())
    if depth != 7:
        findings.append(
            f"the deepest composition any case reaches is {depth} ghost cells per "
            f"source lane; a component with all three of its axes folded owns 7, and "
            f"the triple-composite blocks are otherwise never compiled")
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
    from meep_gpu.triton_kernels import folded_complex_fused_pair as module  # noqa: PLC0415

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


class _Electric:
    """An electric source that publishes no deposit index — the refusing direction."""

    field_type = "D"


def _probe_record():
    """A probe artifact carrying this product's EXTENDED pattern set.

    The laptop legs need a licence to ask the predicate anything; the DEVICE legs
    consume the real artifact ``complex_fields.load_expansion_probe`` resolves.
    """
    from meep_gpu.triton_kernels import folded_complex_fused_pair as module  # noqa: PLC0415

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
    from meep_gpu.triton_kernels import folded_complex_fused_pair as module  # noqa: PLC0415
    from meep_gpu.triton_kernels import complex_fused_electric_pair as unfolded  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_fused_pair as real  # noqa: PLC0415

    findings: List[str] = []
    rows: List[Dict[str, Any]] = []
    probe = _probe_record()
    folded_complex_grid: Dict[str, Any] = {}
    with declaring_run_policy("keep"):
        for case in CASES:
            driver = build_grid(case, prefer_gpu=False)
            try:
                sources = (_Deposit(driver.fields, "Ez",
                                    deposit_center(case)),)
                verdict = module.folded_complex_fused_pair_coverage(
                    driver.fields, driver.pml, sources, probe=probe)
                residual = _residual(verdict)
                rows.append({"case": case[0], "residual_reasons": residual,
                             "shape": list(driver.shape)})
                if residual:
                    findings.append(
                        f"case {case[0]!r} is refused for {residual}; every case this "
                        f"gate scores must be one the shipped predicate admits")
                # THE REFUSING DIRECTION of the source clause, on the same grid: an
                # electric source that cannot publish its deposit index.
                blind = module.folded_complex_fused_pair_coverage(
                    driver.fields, driver.pml, (_Electric(),), probe=probe)
                if not any("does not publish the index" in reason
                           for reason in _residual(blind)):
                    findings.append(
                        f"case {case[0]!r}: an electric source with no deposit index "
                        f"is not refused by name")
                # AN UNDECLARED SOURCE LIST IS A REFUSAL, never an assumed ().
                undeclared = module.folded_complex_fused_pair_coverage(
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
            sources = (_Deposit(driver.fields, "Ez", deposit_center(base)),)
            mine = module.folded_complex_fused_pair_coverage(
                driver.fields, driver.pml, sources, probe=probe)
            theirs = unfolded.complex_fused_electric_pair_coverage(
                driver.fields, driver.pml, sources, probe=probe)
            real_verdict = real.folded_fused_pair_coverage(
                driver.fields, driver.pml, sources)
            folded_complex_grid = {
                "grid": "a FOLDED COMPLEX grid",
                "this": _residual(mine),
                "unfolded_complex": _residual(theirs),
                "real_folded": _residual(real_verdict)}
            if _residual(mine):
                findings.append(f"this product refuses its own grid: "
                                f"{folded_complex_grid}")
            if not _residual(theirs):
                findings.append(
                    "complex_fused_electric_pair ADMITS a folded complex grid; both "
                    "products would claim the slot and _select_slot would leave it "
                    "unselected")
            if not _residual(real_verdict):
                findings.append(
                    "folded_fused_pair ADMITS a COMPLEX grid; its parity is a "
                    "compile-time sign and cannot be that product's")
        finally:
            driver.close()

        # THE PLAN BUILDER RETURNS None ON A REFUSED CONFIGURATION, never raises.
        driver = build_grid(base, prefer_gpu=False)
        try:
            built = module.plan_folded_complex_fused_pair(
                driver.fields, driver.pml,
                (_Deposit(driver.fields, "Ez", deposit_center(base)),),
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
                      "predicate_coverage_triton_2026-08-31_folded_disp")

#: The board cell this product claims, and the rows the newest cut scored into it.
BOARD = os.path.join(API_ROOT, "parity", "meep_gpu", "results",
                     "fusion_matrix_triton_2026-08-31_folded_disp",
                     "fusion_matrix.json")
CELL = ("D->E", "folded complex PML", "folded complex")


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
    """What the board says this cell is worth, and that BOTH rows carry a deposit.

    AN ABSENT CENSUS IS NOT AN EMPTY ONE: a missing artifact is a REFUSAL here, never
    a funnel of zero reported as a pass.
    """
    findings: List[str] = []
    if not os.path.isdir(CENSUS):
        findings.append(f"the census {CENSUS} is not in this tree; the funnel cannot "
                        f"be measured and a zero would be a lie")
    if not os.path.exists(BOARD):
        findings.append(f"the board {BOARD} is not in this tree; the cell this "
                        f"product claims cannot be cross-checked")
    if findings:
        return {"leg": "corpus_admission", "device": False,
                "findings": findings, "passed": False}

    board = json.load(open(BOARD, encoding="utf-8"))
    instances = [entry for entry in board["seam_instances"]
                 if (entry["seam"], entry["curl_arm"], entry["constitutive_arm"])
                 == CELL]
    rows = [entry["row"] for entry in instances]
    if not instances:
        findings.append(f"the board scores NO seam-instance into {CELL}; this "
                        f"product would serve nothing")
    for entry in instances:
        if entry["product"] is not None:
            findings.append(
                f"the board already names {entry['product']!r} in this cell; two "
                f"products in one cell is a coverage question, not a gate result")
        if not entry["in_seam_source"]:
            findings.append(
                f"row {entry['row']!r} carries NO in-seam electric source, so the "
                f"deposit carry this gate measures is not what makes the cell "
                f"reachable")
        if entry["in_seam_source_blocks"]:
            findings.append(
                f"row {entry['row']!r} is BLOCKED by its in-seam source; the deposit "
                f"repair cannot carry it and this product cannot serve it")

    # THE CONFIGURATION FACTS the rows declare, read off the census.
    record = {f"{row['leg']}:{row['row']}": row for row in _census_rows(CENSUS)}
    configuration = {}
    for label in rows:
        row = record.get(label)
        if row is None:
            findings.append(f"row {label!r} is on the board and not in the census")
            continue
        cfg = row["configuration"]
        configuration[label] = {
            "shape": list(cfg["shape"]),
            "force_complex_fields": bool(cfg["force_complex_fields"]),
            "pml_active": bool(cfg["pml_active"]),
            "has_symmetry": bool(cfg["has_symmetry"]),
            "mirrored": list(cfg["mirrored"]),
            "source_field_types": list(cfg.get("source_field_types") or ()),
            "has_offdiagonal_epsilon": bool(cfg["has_offdiagonal_epsilon"]),
            "n_polarizations": int(cfg["n_polarizations"]),
        }
        if not cfg["force_complex_fields"]:
            findings.append(f"row {label!r} is not complex-storage")
        if not cfg["pml_active"]:
            findings.append(f"row {label!r} carries no active absorber")
        if "D" not in (cfg.get("source_field_types") or ()):
            findings.append(f"row {label!r} declares no electric source")
    return {"leg": "corpus_admission", "device": False,
            "cell": list(CELL), "rows": rows, "instances": len(instances),
            "configuration": configuration,
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
        "signed zero when step_D runs. All three left every compared volume "
        "byte-identical. The ORDER is held by CONSTRUCTION — parity_chain_leg "
        "compares the shipped sequence against folded_complex_fused_pair.parity_"
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

#: THE CURL AUXILIARIES ARE SEEDED TOO, and that is a measurement rather than
#: thoroughness. ``fu_D`` starts at zero, and on a FAR destination plane the curl is
#: masked to zero by the folded-periodic top-plane mask — so the recurrence there is
#: ``n = p * km * sinv`` with ``p == 0``, which stays zero forever. Every statement
#: about that plane's auxiliary is then a statement about zero: measured 2026-08-31,
#: when ``m_fu_store_masked`` reported a real defect as UNCAUGHT because masking the
#: store of a value that is identically zero moves no byte. Seeding ``fu_D`` makes
#: the plane's auxiliary a live number and the mutation reachable.
SEED_AUXILIARIES = ("f_w_Ex", "f_w_Ey", "f_w_Ez", "fu_Dx", "fu_Dy", "fu_Dz")


def build_driver(cp, case, value_class: str, electric: bool):
    """One folded complex driver, seeded identically for every route."""
    driver = build_grid(case)
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    # A VARYING epsilon, deliberately: the constitutive half multiplies by inverse
    # epsilon at the imaged ghost's OWN index, and a uniform material would make that
    # indistinguishable from reading it at the source lane's.
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    if electric:
        # THE DEPOSIT THIS PRODUCT EXISTS FOR. Injected BETWEEN the two halves
        # (driver.py:3294-3299), so the fused launch consumes a pre-injection D and
        # the shipped repair is what puts the difference back.
        #
        # OFF THE MIRROR PLANE ON EVERY FOLDED AXIS. A point Ez at the origin sits ON
        # the plane, and on an ODD fold that is refused with a reason rather than
        # stepped (from_meep.py:2686-2698).
        center = [0.0, 0.0, 0.0]
        for axis_name, _phase in case[4]:
            axis = "XYZ".index(axis_name.upper())
            center[axis] = 0.25 * (case[2][axis] / 2.0)
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    else:
        # A MAGNETIC source is admitted with no repair at all: the driver injects it
        # in the B/H seam, not this one. Carrying one is what stops the source clause
        # from being tested only in its refusing direction, and it is what drives the
        # quiet family's fields.
        center = [0.0, 0.0, 0.0]
        for axis_name, _phase in case[4]:
            axis = "XYZ".index(axis_name.upper())
            center[axis] = 0.25 * (case[2][axis] / 2.0)
        driver.add_source({"component": "Hz", "frequency": 0.31,
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

    ``fill_symmetry_bc_D`` is every folded axis's NEAR plane and
    ``fill_folded_far_ghosts_D`` every folded axis's FAR one, with ``zero_metal_D``
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

    curl = folded_complex.plan_folded_complex_pml_curl(
        driver.fields, driver.pml, "step_D", probe=probe)
    constitutive = folded_complex.plan_folded_complex_constitutive(
        driver.fields, driver.pml, "E", probe=probe)
    fill = folded_complex.plan_folded_mirror_ghost_fill_complex(
        driver.fields, "D", probe=probe)
    missing = [name for name, plan in
               (("folded complex curl", curl), ("folded complex ghost fill", fill),
                ("folded complex constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so this "
            f"leg could not compare the fused launch against the products it "
            f"replaces")
    # `zero_metal_D` stays on the ARRAY PATH here: no Triton product owns the wall
    # clear. It is counted, so a difference in whether it ran is a counter event
    # rather than an invisible correction. THE INJECTION ALSO STAYS ON THE ARRAY
    # PATH, which is what makes this the right oracle for the carry family: the
    # certified kernels with the driver's own deposit between them.
    return Route({"step_D": curl,
                  "fill_symmetry_bc_D": _TwoPassFill(fill, "near"),
                  "fill_folded_far_ghosts_D": _TwoPassFill(fill, "far"),
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

def run_leg(cp, name: str, case, steps: int, product, probe,
            value_class: str = "uniform", mutant: Any = None,
            install_fused: bool = True, freeze_electric: bool = False,
            electric: bool = False, bracket: bool = True,
            expansion: Optional[int] = None) -> Dict[str, Any]:
    """Three routes in lockstep, per COMPLETE driver step; stop at the FIRST byte
    divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference, seeded = build_driver(cp, case, value_class, electric)
    separate, _ = build_driver(cp, case, value_class, electric)
    fused, _ = build_driver(cp, case, value_class, electric)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_complex_fused_curl_constitutive_D_kernel())
    leading = None
    row: Dict[str, Any] = {
        "leg": name, "device": True, "case": case[0], "steps_budget": steps,
        "value_class": value_class, "seed": case_seed(case[0], value_class),
        "shape": list(reference.shape), "boundaries": str(case[3]),
        "mirrors": [list(entry) for entry in case[4]],
        "k_point": list(case[5]), "pml": str(case[6]),
        "electric_source": bool(electric),
        "bracketed": bool(bracket),
        "fused_substituted": bool(install_fused),
        "electric_frozen": bool(freeze_electric),
        "expansion_override": expansion,
        "operand_census": operand_census(seeded),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_folded_complex_fused_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel, probe=probe)
        row["admitted_by_the_shipped_predicate"] = plan is not None
        if plan is None:
            verdict = product.folded_complex_fused_pair_coverage(
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
        if freeze_electric:
            # ARMED: every route's electric seam is inert. All three then agree
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

    ``_carry_ghost_complex_E`` is included for two reasons, both failures a mutation
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


def mutation_table() -> Tuple[Tuple[str, str, str,
                                    Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was
    MEASURED, never what was expected."""

    # ------------------------------------------------------------------
    # THE THREE THINGS ONLY THIS PRODUCT HAS
    # ------------------------------------------------------------------

    def m_ghost_inv_eps_at_the_source(source: str) -> Tuple[str, int]:
        """THE DEFECT THIS PRODUCT COULD HAVE. The ghost's inverse permittivity read
        at a cell that is NOT its own.

        ``update_E`` scales by ``inverse_epsilon_for(component)`` at the cell it
        writes; the fills write D and never touch the material, so at an imaged cell
        the array path reads THAT cell's coefficient. Binding cell 0 instead is in
        bounds by construction, where a forward shift could address outside the
        allocation and be reported as an inert defect. It is inside the DEVICE
        FUNCTION, so ONE edit arms all seven destinations of all three components."""
        return _replace_all(source,
                            "tl.load(ie + dst, mask=mask, other=0.0)",
                            "tl.load(ie + dst * 0, mask=mask, other=0.0)")

    def m_clear_before_parity(source: str) -> Tuple[str, int]:
        """The wall clear applied BEFORE the near parity instead of after.

        The driver runs ``fill_symmetry_bc_D`` (:3300) then ``zero_metal_D`` (:3301),
        so a near ghost is ``clear(parity (x) v)`` and not ``parity (x) clear(v)``.

        THE DISCRIMINATION IS A SIGNED ZERO, AND THAT IS THE HONEST STATEMENT OF IT.
        The owned register has ALREADY taken the clear before the carry, so on the
        cleared plane the value the parity multiplies is exactly zero and the two
        orders differ only in the sign of the zero the multiply produces:
        ``(-1, +0) (x) (+0, +0)`` is ``(-0, +0)`` and the array path leaves ``+0``
        there, because ``zero_metal_D`` runs LAST of the two. So the shipped order's
        re-clear is what normalises it, and it is load-bearing for exactly that.
        SCORED ON COMPONENT 2's NEAR_Y BLOCK, not component 0's: the clear rows are
        the component's two NON-own axes, so on a grid folding Y with a wall on X it
        is Dz whose near ghost takes a live clear and Dx whose ZM_Y/ZM_Z guards are
        both false."""
        return _rewrite_block(
            source,
            ["gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v2_re, v2_im,",
             "EXPANSION)",
             "if ZM_X:",
             "gc_re = tl.where(at_x, 0.0, gc_re)",
             "gc_im = tl.where(at_x, 0.0, gc_im)",
             "if ZM_Y:",
             "gc_re = tl.where(at_y, 0.0, gc_re)",
             "gc_im = tl.where(at_y, 0.0, gc_im)"],
            ["gc_re, gc_im = v2_re, v2_im",
             "if ZM_X:",
             "    gc_re = tl.where(at_x, 0.0, gc_re)",
             "    gc_im = tl.where(at_x, 0.0, gc_im)",
             "if ZM_Y:",
             "    gc_re = tl.where(at_y, 0.0, gc_re)",
             "    gc_im = tl.where(at_y, 0.0, gc_im)",
             "gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, gc_re, gc_im,",
             "                                          EXPANSION)"])

    def m_clear_dropped_on_the_ghost(source: str) -> Tuple[str, int]:
        """The near ghost takes no clear at all.

        On the D family ``_zero_metal``'s rows ARE the near fill's axes, so a
        component folded on one axis is still cleared on a wall on another. The
        discrimination is the same signed zero ``m_clear_before_parity`` names.

        THE GUARD IS KEPT AND ITS PREDICATE IS MADE UNSATISFIABLE, rather than the
        line being deleted or replaced by ``+ 0.0``: an addition of ``+0.0`` would
        normalise ``-0.0`` to ``+0.0`` and silently reproduce the shipped answer,
        which is a mutation that repairs the defect it is meant to plant."""
        source, hits = _replace_all(
            source, "gc_re = tl.where(at_x, 0.0, gc_re)",
            "gc_re = tl.where(at_x & (idx < 0), 0.0, gc_re)")
        source, more = _replace_all(
            source, "gc_im = tl.where(at_x, 0.0, gc_im)",
            "gc_im = tl.where(at_x & (idx < 0), 0.0, gc_im)")
        return source, hits + more

    def m_far_takes_the_source_coefficient(source: str) -> Tuple[str, int]:
        """The far destination reads the SOURCE lane's kps/kms instead of the top
        row's. ``update_E`` indexes component m on axis m, which is exactly the axis
        the far fill images along, so the two rows are different entries of a graded
        absorber profile."""
        hits = 0
        for component in range(3):
            source, count = _replace_all(
                source, f"kp_f{component}, km_f{component},",
                f"kp_{component}, km_{component},")
            hits += count
        return source, hits

    def m_near_takes_the_moved_coefficient(source: str) -> Tuple[str, int]:
        """The reverse: a near destination reading the TOP row's pair. The near fill
        images along an axis that is NOT the component's own, so its destination sits
        at the same coefficient index as the lane that owns it."""
        hits = 0
        for component in range(3):
            source, count = _replace_all(
                source, f"gc_re, gc_im, kp_{component}, km_{component},",
                f"gc_re, gc_im, kp_f{component}, km_f{component},")
            hits += count
        return source, hits

    # ------------------------------------------------------------------
    # THE PARITY CHAIN — inherited from the magnetic twin as a MEASUREMENT
    # ------------------------------------------------------------------

    def m_parity_chain_reordered(source: str) -> Tuple[str, int]:
        """The two near applications swapped. Complex float multiplication is not
        associative: re-ordering a two-parity chain moved 8 of 128 uint32 words on
        the Metal twin's exhaustive signed-zero table. Scored on a grid folding TWO
        axes at DIFFERING phases, or the composition commutes bit-exactly."""
        return _rewrite_block(
            source,
            ["gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v2_re, v2_im,",
             "EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, gc_re, gc_im,",
             "EXPANSION)"],
            ["gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v2_re, v2_im,",
             "EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, gc_re, gc_im,",
             "EXPANSION)"])

    def m_parity_chain_folded(source: str) -> Tuple[str, int]:
        """The two near applications folded into ONE product of the coefficients.
        Folding moved up to 23 of 128 uint32 words on the same table."""
        return _rewrite_block(
            source,
            ["gc_re, gc_im = _mul_imag_coefficient_left(n0r, n0i, v2_re, v2_im,",
             "EXPANSION)",
             "gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, gc_re, gc_im,",
             "EXPANSION)"],
            ["gc_re, gc_im = _mul_imag_coefficient_left(n0r * n1r - n0i * n1i,",
             "n0r * n1i + n0i * n1r,",
             "v2_re, v2_im, EXPANSION)"])

    def m_parity_dropped(source: str) -> Tuple[str, int]:
        """The near parity not applied at all. Invisible on an EVEN plane by
        construction (the multiply is by +1), which is why this is scored on an ODD
        one."""
        return _replace_all(
            source,
            "gc_re, gc_im = _mul_imag_coefficient_left(n1r, n1i, v0_re, v0_im,",
            "gc_re, gc_im = _mul_imag_coefficient_left(1.0, 0.0, v0_re, v0_im,")

    def m_far_parity_sign(source: str) -> Tuple[str, int]:
        """The far parity taken as the NEAR word. ``mirror_parity`` is
        ``phase * (1 - 2*iyee)``, so the two differ in sign on every axis."""
        return _replace_all(source, "_mul_imag_coefficient_left(d0r, d0i,",
                            "_mul_imag_coefficient_left(n0r, n0i,")

    # ------------------------------------------------------------------
    # THE CARRY GEOMETRY — inherited from the real folded pair
    # ------------------------------------------------------------------

    def m_near_source_index(source: str) -> Tuple[str, int]:
        """The near fill images stored cell 2; imaging cell 3 is one row off."""
        return _replace_all(source, "near_j = live & (j == 2)",
                            "near_j = live & (j == 3)")

    def m_reflect_row_baked(source: str) -> Tuple[str, int]:
        """``n - 2`` instead of the runtime reflect row. Correct at an EVEN full
        count and a whole cell wrong at an ODD one, which is why this is scored on
        the odd case."""
        return _replace_all(source, "far_j = live & (j == ry)",
                            "far_j = live & (j == ny - 2)")

    def m_far_carry_dropped(source: str) -> Tuple[str, int]:
        """The far ghost never written. The top plane is then whatever the masked
        curl left, which is exactly the plane the ownership mask zeroed."""
        return _replace_all(source, "if FAR_Y:", "if FAR_Y and False:")

    def m_near_carry_dropped(source: str) -> Tuple[str, int]:
        return _replace_all(source, "if NEAR_Y:", "if NEAR_Y and False:")

    def m_ownership_mask_dropped(source: str) -> Tuple[str, int]:
        """A destination lane READS its own D — and the result is DISCARDED.

        DECLARED NULL, and the declaration is what the shipped comment actually says
        rather than what a reader would like it to say: the mask on the ``D`` load is
        "a statement about memory traffic and not about dead registers". Every
        consumer of the register it feeds — the ``f``/``h``/``w`` stores and every
        carry block — is masked by the SAME predicate, so removing it lets a
        destination lane read a word another lane writes (a real race, and a real
        traffic change) while moving no byte.

        WHAT IS BYTE-LOAD-BEARING IS THE WRITE SIDE, and it is covered:
        ``m_carry_mask_not_conjoined`` drops the conjunction on a carry that then
        writes a cell it does not own, and ``m_fu_store_masked`` masks a store that
        must not be masked. MEASURED UNCAUGHT on 2026-08-31 before this entry was
        moved to ``null``; a null that IS caught fails this gate, so the declaration
        is checked rather than assumed."""
        return _replace_all(source,
                            "e_re = tl.load(f0 + 2 * idx, mask=own0, other=0.0)",
                            "e_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)")

    def m_carry_mask_not_conjoined(source: str) -> Tuple[str, int]:
        """A carry mask without its component's ownership conjunction — the
        ``run_farcarry2`` defect, transcribed.

        DECLARED NULL, and the declaration is a MEASUREMENT of this body's emission
        order rather than a claim that the conjunction is decorative.

        THE REWRITE IS ARMED WHERE IT BITES. Targeted at the plain ``NEAR_Y`` block,
        whose mask selects ``j == 2`` at every ``i`` while ``own0`` excludes
        ``i == nx - 1`` whenever ``FAR_X`` is live — the ``run_farcarry2`` shape: a
        lane that is the SOURCE of one fill and the DESTINATION of another, writing a
        ghost from a ``v`` its own ownership mask zeroed. (In the COMPOSITE block the
        conjunction is redundant outright: that mask already selects ``i == rx`` and
        ``j == 2``, neither of which ``own0`` excludes — measured 2026-08-31, when
        the rewrite aimed there reported UNCAUGHT.)

        AND IT IS STILL BYTE-NULL, for a reason worth writing down. The cell the
        un-conjoined lane corrupts — ``(nx - 1, 0)`` — is the far-over-near COMPOSITE
        ghost, which the ``FAR_X and NEAR_Y`` block writes CORRECTLY from lane
        ``(rx, 2)``; that block is emitted LATER in the body, and the two lanes are
        216 flat indices apart against a 256-element BLOCK, so they are the same
        program and the later store wins deterministically. On a ONE-fold case there
        is no composite block and ``own0`` has only one exclusion, which the carry
        mask already implies. Either way no byte moves. MEASURED UNCAUGHT twice on
        2026-08-31, at both targets, before this entry was moved to ``null``.

        WHAT THE CONJUNCTION IS FOR IS THEREFORE TRAFFIC AND RACES, exactly as the
        shipped comment beside it says — and it STAYS: the byte-nullity above rests
        on the emission order and on the block size, and a reordering of the carry
        blocks or a different ``BLOCK`` would make it a real defect again. The real
        folded pair's own device gate measured this defect class REAL on ITS body
        (``run_farcarry2``, 2026-08-20, one word of Hx and one of Hy at 1-2 ULP),
        which is why the mask is written and why this entry is a confirmed null
        rather than a deleted mutation."""
        return _replace_all(source, "own0 & near_j, EXPANSION)",
                            "near_j, EXPANSION)")

    def m_top_plane_mask_dropped(source: str) -> Tuple[str, int]:
        """The folded-PERIODIC top-plane mask removed from the curl. The far carry's
        destination lane would then contribute its own curl to a cell the source lane
        also writes."""
        return _replace_all(source, "if BCY == MIRROR_PERIODIC:\n"
                                    "                curl1_re = tl.where(last_y, 0.0,"
                                    " curl1_re)",
                            "if BCY == MIRROR_PERIODIC and False:\n"
                            "                curl1_re = tl.where(last_y, 0.0,"
                            " curl1_re)")

    # ------------------------------------------------------------------
    # THE CONSTITUTIVE — inherited from the complex electric pair
    # ------------------------------------------------------------------

    def m_prev_read_after_the_store(source: str) -> Tuple[str, int]:
        """``f_w`` read AFTER it is overwritten: the one ordering the constitutive
        cannot survive being wrong about (S:2083-2085)."""
        return _rewrite_block(
            source,
            ["prev_re = tl.load(w + 2 * dst, mask=mask, other=0.0)",
             "prev_im = tl.load(w + 2 * dst + 1, mask=mask, other=0.0)",
             "src_re, src_im = _mul_field_left(",
             "ghost_re, ghost_im, tl.load(ie + dst, mask=mask, other=0.0), EXPANSION)",
             "tl.store(w + 2 * dst, src_re, mask=mask)",
             "tl.store(w + 2 * dst + 1, src_im, mask=mask)"],
            ["src_re, src_im = _mul_field_left(",
             "ghost_re, ghost_im, tl.load(ie + dst, mask=mask, other=0.0), EXPANSION)",
             "tl.store(w + 2 * dst, src_re, mask=mask)",
             "tl.store(w + 2 * dst + 1, src_im, mask=mask)",
             "prev_re = tl.load(w + 2 * dst, mask=mask, other=0.0)",
             "prev_im = tl.load(w + 2 * dst + 1, mask=mask, other=0.0)"])

    def m_constitutive_lattice_swapped(source: str) -> Tuple[str, int]:
        """The kps/kms pair taken from the CURL's integer lattice instead of the
        constitutive's half-integer one — a silent half-cell error in the absorber
        profile. Rewritten as an index swap between components, which is the same
        class of defect and one the kernel can express."""
        return _rewrite_block(
            source,
            ["kp_0 = tl.load(kp0 + i, mask=live, other=0.0)",
             "km_0 = tl.load(km0 + i, mask=live, other=0.0)"],
            ["kp_0 = tl.load(kp0 + j, mask=live, other=0.0)",
             "km_0 = tl.load(km0 + j, mask=live, other=0.0)"])

    def m_inv_eps_after_the_fw_store(source: str) -> Tuple[str, int]:
        """``f_w`` holding ``D`` instead of ``D * inv_eps``. The array path scales
        BEFORE the workspace write (complex_fields.py:783-786), so this is a state
        error that only shows on the NEXT step's ``km * prev``."""
        return _rewrite_block(
            source,
            ["src_re, src_im = _mul_field_left(",
             "v0_re, v0_im, tl.load(ie0 + idx, mask=own0, other=0.0), EXPANSION)",
             "tl.store(w0 + 2 * idx, src_re, mask=own0)",
             "tl.store(w0 + 2 * idx + 1, src_im, mask=own0)"],
            ["tl.store(w0 + 2 * idx, v0_re, mask=own0)",
             "tl.store(w0 + 2 * idx + 1, v0_im, mask=own0)",
             "src_re, src_im = _mul_field_left(",
             "v0_re, v0_im, tl.load(ie0 + idx, mask=own0, other=0.0), EXPANSION)"])

    def m_d_store_dropped(source: str) -> Tuple[str, int]:
        """The stepped displacement never written back. The next timestep's curl
        reads D, so a weld that dropped the store would fuse away a value the run
        still needs."""
        return _replace_all(source, "tl.store(f0 + 2 * idx, v0_re, mask=own0)",
                            "tl.store(f0 + 2 * idx, v0_re, mask=own0 & (idx < 0))")

    def m_fu_store_masked(source: str) -> Tuple[str, int]:
        """``fu`` written under the OWNERSHIP mask instead of ``live``. The array
        path's ``step_D`` writes it at every cell and neither fill touches it
        (stepping.py:1497-1498), so a destination lane still owns its ``fu`` word."""
        return _replace_all(source, "tl.store(u0 + 2 * idx, n0_re, mask=live)",
                            "tl.store(u0 + 2 * idx, n0_re, mask=own0)")

    def m_wall_clear_three_rows(source: str) -> Tuple[str, int]:
        """The MAGNETIC twin's three-row wall clear on the D family, which needs
        six. ``_zero_metal`` clears component m on the two axes that are NOT m."""
        return _rewrite_block(
            source,
            ["if ZM_X:",
             "v1_re = tl.where(at_x, 0.0, v1_re)",
             "v1_im = tl.where(at_x, 0.0, v1_im)",
             "v2_re = tl.where(at_x, 0.0, v2_re)",
             "v2_im = tl.where(at_x, 0.0, v2_im)"],
            ["if ZM_X:",
             "    v0_re = tl.where(at_x, 0.0, v0_re)",
             "    v0_im = tl.where(at_x, 0.0, v0_im)"])

    return (
        ("m_ghost_inv_eps_at_the_source",
         "the imaged ghost's inverse permittivity read at cell 0 rather than its own",
         "caught", m_ghost_inv_eps_at_the_source),
        ("m_clear_before_parity",
         "the wall clear applied before the near parity instead of after it",
         "caught", m_clear_before_parity),
        ("m_clear_dropped_on_the_ghost",
         "the near ghost takes no wall clear at all", "caught",
         m_clear_dropped_on_the_ghost),
        ("m_far_takes_the_source_coefficient",
         "the far destination reads the source lane's kps/kms", "caught",
         m_far_takes_the_source_coefficient),
        ("m_near_takes_the_moved_coefficient",
         "a near destination reads the top row's kps/kms", "caught",
         m_near_takes_the_moved_coefficient),
        ("m_parity_chain_reordered",
         "the two near parities applied last-to-first; REAL at the word layer and "
         "unreached at the whole-step layer on every device configuration this "
         "gate can build (see MUTATION_EVIDENCE)",
         "unreached", m_parity_chain_reordered),
        ("m_parity_chain_folded",
         "the two near parities folded into one coefficient; REAL at the word layer "
         "and unreached at the whole-step layer (see MUTATION_EVIDENCE)",
         "unreached", m_parity_chain_folded),
        ("m_parity_dropped", "the near parity not applied", "caught",
         m_parity_dropped),
        ("m_far_parity_sign", "the far parity taken as the near word", "caught",
         m_far_parity_sign),
        ("m_near_source_index", "the near fill images stored cell 3", "caught",
         m_near_source_index),
        ("m_reflect_row_baked", "n - 2 instead of the runtime reflect row", "caught",
         m_reflect_row_baked),
        ("m_far_carry_dropped", "the far ghost never written", "caught",
         m_far_carry_dropped),
        ("m_near_carry_dropped", "the near ghost never written", "caught",
         m_near_carry_dropped),
        ("m_ownership_mask_dropped",
         "a destination lane READS its own D; every consumer of that register is "
         "masked by the same predicate, so the traffic changes and no byte does",
         "null", m_ownership_mask_dropped),
        ("m_carry_mask_not_conjoined",
         "a carry mask without its component's ownership conjunction; the corrupted "
         "cell is rewritten by a LATER composite block in the same program, so the "
         "defect is real for traffic and races and byte-null for this emission "
         "order",
         "null", m_carry_mask_not_conjoined),
        ("m_top_plane_mask_dropped",
         "the folded-periodic top-plane mask removed from the curl", "caught",
         m_top_plane_mask_dropped),
        ("m_prev_read_after_the_store",
         "f_w read after it is overwritten", "caught", m_prev_read_after_the_store),
        ("m_constitutive_lattice_swapped",
         "component 0's constitutive coefficient indexed on axis y", "caught",
         m_constitutive_lattice_swapped),
        ("m_inv_eps_after_the_fw_store",
         "f_w holding D instead of D * inv_eps", "caught",
         m_inv_eps_after_the_fw_store),
        ("m_d_store_dropped", "the stepped displacement never written back",
         "caught", m_d_store_dropped),
        ("m_fu_store_masked", "fu written under the ownership mask", "caught",
         m_fu_store_masked),
        ("m_wall_clear_three_rows", "the magnetic twin's three-row clear on D",
         "caught", m_wall_clear_three_rows),
    )


#: Which case each mutation is scored on — BY NAME, and every entry is justified by
#: the branch the rewrite touches. TRAP: a mutation scored on a grid that never ENTERS
#: the branch it rewrites reports UNCAUGHT while measuring nothing.
MUTATION_CASE: Dict[str, str] = {
    # the ghost material read: any grid with a fold and a varying epsilon
    "m_ghost_inv_eps_at_the_source": "y_fold_periodic_odd_phase",
    # the clear inside the chain needs a WALL beside a FOLD, and an ODD plane or the
    # doubled clear is a no-op
    "m_clear_before_parity": "y_fold_periodic_wall_x",
    "m_clear_dropped_on_the_ghost": "y_fold_periodic_wall_x",
    # the far coefficient pair needs a folded PERIODIC axis AND a reflect row whose
    # kps/kms differ from the top row's. MEASURED 2026-08-31 on this gate's own case
    # table: at an EVEN full count they are BITWISE EQUAL (the not-owned ghost slot
    # copies its neighbour), so `y_fold_periodic` reported this real defect as
    # UNCAUGHT. `y_fold_periodic_odd` is the one case where reflect row 17 and top
    # row 19 hold different words, and `_assert_the_far_coefficient_actually_moves`
    # below re-measures that rather than trusting this comment.
    "m_far_takes_the_source_coefficient": "y_fold_periodic_odd",
    # THE NEAR MUTATION IS NOT IN THE SAME POSITION and needs no such case: it makes
    # a near destination read the TOP row instead of ITS OWN lane's row, and a lane
    # anywhere but the top of a graded absorber holds a different word. Caught on
    # `y_fold_periodic`, measured.
    "m_near_takes_the_moved_coefficient": "y_fold_periodic",
    # the near chain's ORDER exists only with TWO folded axes at DIFFERING phases
    "m_parity_chain_reordered": "xy_mixed_phase_periodic",
    "m_parity_chain_folded": "xy_mixed_phase_periodic",
    # a dropped parity is invisible on an EVEN plane
    "m_parity_dropped": "y_fold_periodic_odd_phase",
    "m_far_parity_sign": "x_fold_periodic_wall_y",
    # the near source lane and the reflect row
    "m_near_source_index": "y_fold_periodic",
    "m_reflect_row_baked": "y_fold_periodic_odd",
    "m_far_carry_dropped": "y_fold_periodic",
    "m_near_carry_dropped": "y_fold_periodic",
    "m_ownership_mask_dropped": "y_fold_periodic",
    "m_carry_mask_not_conjoined": "xy_mixed_phase_periodic",
    "m_top_plane_mask_dropped": "y_fold_periodic",
    # THE FU STORE MASK NEEDS AN ABSORBING DESTINATION PLANE, and that is a
    # measurement rather than a preference. Masking the `fu` store leaves the
    # destination plane's auxiliary unwritten; on a NEAR plane (stored cell 0 of a
    # folded axis) that is a NO-OP, because a folded axis carries its layer on the
    # HIGH face only, so km = sinv = 1 there and the recurrence returns `p0`
    # unchanged — measured UNCAUGHT on `y_fold_periodic` on 2026-08-31, where
    # component 0's only destination plane is the near one. `x_fold_periodic_open_y`
    # folds X PERIODIC, so component 0 has a FAR destination at the TOP of the x
    # absorber, where the coefficients are not unity.
    "m_fu_store_masked": "x_fold_periodic_open_y",
    # the constitutive half needs nothing but a fold
    "m_prev_read_after_the_store": "y_fold_periodic",
    "m_constitutive_lattice_swapped": "y_fold_periodic",
    "m_inv_eps_after_the_fw_store": "y_fold_periodic",
    "m_d_store_dropped": "y_fold_periodic",
    # (``m_fu_store_masked`` is above, on x_fold_periodic_open_y, and was a DUPLICATE
    # KEY here until 2026-08-31 — the later entry silently won and put the mutation
    # back on a case where it is a no-op.)
    # the six-row wall clear needs a live x wall
    "m_wall_clear_three_rows": "y_fold_metallic",
}

DEFAULT_MUTATION_CASE = "y_fold_periodic"

#: Per mutation, the value class and step budget its leg runs under, where the
#: DEFAULT (uniform, MUTATION_STEPS, with the carry) cannot reach the defect.
#:
#: THE PARITY CHAIN IS THE WHOLE REASON THIS TABLE EXISTS, and it is a measurement
#: rather than a preference. Re-ordering or folding a two-parity chain changes bytes
#: ONLY where the multiplied value is a SIGNED ZERO — ``c_re`` is exactly ``±1.0``
#: and ``c_im`` is bitwise ``+0.0``, so the product is a sign flip plus a signed-zero
#: addend, exact on every finite operand (see :func:`parity_chain_associativity_leg`,
#: which counts it). To reach it the ghost SOURCE lane must hold such a word at the
#: moment of the launch, which the uniform class never does and which the ``±0``
#: lattice holds only at STEP 1, before a source has driven anything away from zero.
#: So those two run QUIET, on the ``zero_lattice``, for ONE step.
MUTATION_VALUE_CLASS: Dict[str, str] = {
    "m_parity_chain_reordered": "zero_lattice",
    "m_parity_chain_folded": "zero_lattice",
}
MUTATION_STEPS_FOR: Dict[str, int] = {
    "m_parity_chain_reordered": 1,
    "m_parity_chain_folded": 1,
}
MUTATION_ELECTRIC: Dict[str, bool] = {
    "m_parity_chain_reordered": False,
    "m_parity_chain_folded": False,
}

#: Per mutation: axes that must be FOLDED on the scored case.
MUTATION_REQUIRES_FOLD: Dict[str, Tuple[str, ...]] = {
    "m_parity_chain_reordered": ("X", "Y"),
    "m_parity_chain_folded": ("X", "Y"),
    "m_carry_mask_not_conjoined": ("X", "Y"),
    "m_parity_dropped": ("Y",),
    "m_clear_before_parity": ("Y",),
    "m_clear_dropped_on_the_ghost": ("Y",),
    "m_near_source_index": ("Y",),
    "m_near_carry_dropped": ("Y",),
    "m_wall_clear_three_rows": ("Y",),
    "m_far_parity_sign": ("X",),
}

#: Per mutation: axes that must be folded over a PERIODIC outer declaration, or the
#: FAR blocks and the top-plane mask are compile-time absent.
MUTATION_REQUIRES_PERIODIC_FOLD: Dict[str, Tuple[str, ...]] = {
    "m_fu_store_masked": ("X",),
    "m_carry_mask_not_conjoined": ("X",),
    "m_far_takes_the_source_coefficient": ("Y",),
    "m_near_takes_the_moved_coefficient": ("Y",),
    "m_reflect_row_baked": ("Y",),
    "m_far_carry_dropped": ("Y",),
    "m_top_plane_mask_dropped": ("Y",),
    "m_far_parity_sign": ("X",),
}

#: Per mutation: axes that must carry a live WALL.
MUTATION_REQUIRES_WALL: Dict[str, Tuple[str, ...]] = {
    "m_clear_before_parity": ("X",),
    "m_clear_dropped_on_the_ghost": ("X",),
    "m_wall_clear_three_rows": ("X",),
}

#: Per mutation: axes whose mirror plane must be ODD (phase -1). An even plane
#: multiplies by +1, so a dropped or duplicated parity is invisible there.
MUTATION_REQUIRES_ODD_PARITY: Dict[str, Tuple[str, ...]] = {
    "m_parity_dropped": ("Y",),
    "m_clear_before_parity": ("Y",),
    "m_ghost_inv_eps_at_the_source": ("Y",),
    "m_far_parity_sign": ("X",),
}

#: Per mutation: the full count of a named axis must be ODD, so a baked ``n - 2``
#: reflect row is wrong rather than right by coincidence.
MUTATION_REQUIRES_ODD_FULL_COUNT: Dict[str, Tuple[str, ...]] = {
    "m_reflect_row_baked": ("Y",),
}

#: Per mutation: the FAR destination's constitutive coefficient pair must actually
#: DIFFER from its source lane's, or exchanging the two is a bitwise no-op and the
#: leg reports a real defect as UNCAUGHT.
#:
#: A PROPERTY OF THE ABSORBER'S GRADING, not of the case's fold set, so it is
#: MEASURED on the grid the case builds rather than declared. At an EVEN full count
#: the reflect row is ``n - 2``, the destination is ``n - 1``, and the coefficient
#: array holds the SAME WORD at both because the not-owned ghost slot copies its
#: neighbour — measured on this gate's own case table 2026-08-31, on every case but
#: one, and measured by the plain folded pair's gate on 2026-08-21 before that.
MUTATION_REQUIRES_MOVING_FAR_COEFFICIENT: Dict[str, Tuple[str, ...]] = {
    "m_far_takes_the_source_coefficient": ("Y",),
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
        kernel_name = f"mutant_{index}_folded_complex_D"
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
        electric = MUTATION_ELECTRIC.get(name, True)
        row["value_class"] = value_class
        row["steps"] = steps
        row["electric_source"] = electric
        leg = run_leg(cp, f"mutation:{name}", case, steps, product, probe,
                      value_class=value_class, mutant=mutant, electric=electric)
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
        ("electric_source_without_a_deposit_index",
         lambda driver: (_Electric(),), "does not publish the index", False),
        ("undeclared_source_list", lambda driver: None,
         "was not declared", False),
        ("electric_source_with_a_deposit_index",
         lambda driver: (_Deposit(driver.fields, "Ez",
                                  deposit_center(base)),), None, True),
        ("magnetic_source_only",
         lambda driver: (_Deposit(driver.fields, "Hz",
                                  deposit_center(base)),), None, True),
        ("offdiagonal_chi1inv_row",
         lambda driver: (_Deposit(driver.fields, "Ez", deposit_center(base)),),
         "STENCIL", False),
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
            verdict = product.folded_complex_fused_pair_coverage(
                driver.fields, driver.pml, sources, probe=probe)
            built = product.plan_folded_complex_fused_pair(
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
        HERE, "results", "triton_folded_complex_fused_pair"),
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
        "gate": "triton_folded_complex_fused_pair",
        "product": "meep_gpu.triton_kernels.folded_complex_fused_pair",
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
    from meep_gpu.triton_kernels import folded_complex_fused_pair as product  # noqa: PLC0415

    # THE PROBE THE DEVICE LEGS CONSUME, resolved once and RECORDED. The device bytes
    # are a function of the arm, so an artifact that does not name the record that
    # licensed it cannot be read.
    probe = complex_fields.load_expansion_probe()
    licence = folded_complex.parity_expansion_license(probe)
    payload["expansion"] = {
        "probe_path": os.environ.get("MEEP_GPU_COMPLEX_EXPANSION_PROBE"),
        "pattern_set": list(product.PRODUCT_PROBE_PATTERNS),
        "arm": licence["arm"], "basis": licence["basis"],
        "expansion": licence["expansion"], "refusals": list(licence["refusals"]),
        "policy_resolved": licence.get("policy_resolved"),
        "candidate_policy": licence.get("candidate_policy"),
    }
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

    log("\n=== device legs: the CARRY family (a real electric deposit in the seam) ===")
    for case_name in CARRY_CASES:
        case = CASES_BY_NAME[case_name]
        for value_class in VALUE_CLASSES:
            row = run_leg(cp, f"carry:{case_name}/{value_class}", case, CARRY_STEPS,
                          product, probe, value_class=value_class, electric=True)
            passed, failures = verdict_of(row, require_launches=CARRY_STEPS,
                                          require_repairs=True)
            row["passed"], row["failures"] = passed, failures
            payload["device_legs"].append(row)
            save(payload, artifact)

    log("\n=== the NULL CONTROL: the same carry cases with the bracket REMOVED ===")
    for case_name in CARRY_CASES:
        case = CASES_BY_NAME[case_name]
        row = run_leg(cp, f"null_control:{case_name}", case, CARRY_STEPS, product,
                      probe, electric=True, bracket=False)
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
                  freeze_electric=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(
        row.get("inert_here_but_moving_on_the_array_path")
        or not row.get("seam_outputs_moved"))
    row["why"] = ("every route's electric seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, artifact)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.folded_complex_fused_curl_constitutive_D)
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
