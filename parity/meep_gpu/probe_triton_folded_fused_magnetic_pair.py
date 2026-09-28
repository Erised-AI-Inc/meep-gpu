"""Byte gate for the folded fused magnetic pair: ``step_B`` welded into ``update_H``.

DEVICE STATUS: **RELEASED 2026-08-20; RE-CUT 2026-08-21 for the branches no leg
    executed.** An adversarial verifier refuted the 2026-08-20 release on a gap
    this harness had, not the kernel: the kernel shipped the three-axis
    folded-PERIODIC blocks — ``if NEAR_a and (FAR_b and FAR_c):``, three of them —
    and EVERY ``mirrors`` value in the released artifact carried at most TWO folded
    axes, so no leg ever compiled them. Byte identity over ten steps on ten cases
    is silent about a branch none of the ten contains, and that is the builder's
    own 2026-08-20 argument — "a gate whose case table stopped at ONE folded axis
    would have released it" — one rung higher.

    COUNTING MADE IT WORSE THAN THE INSTANCE. The new ``branch_reachability`` leg
    executes every live constexpr guard in the kernel against the constexprs each
    case really compiles: **20 of 45 were unreachable** on the 2026-08-20 table,
    not 3 — the whole z half of the carry, the z wall clear, and the ghost-rule
    wrap on an unfolded periodic Y. Three cases close all twenty:
    ``xyz_periodic_odd_3d`` (three folded periodic axes, mixed phases, odd full
    count — SEVEN ghost cells owned by one source lane, the deepest composition
    this family can emit), ``grating_3d_two_folds`` (shape [16, 12, 213], the
    corpus row ``tests:TestModeDecomposition.test_grating_3d`` this product's
    census claims, at its own resolution and courant), and
    ``x_fold_periodic_wall_z_3d``. Four mutations arrive with them: the triple
    composite dropped, its parity product short one plane, its destination short
    one hop, and — the one that was missing altogether — the ownership AND the
    2026-08-20 device run itself found, re-planted.

    Its first device run found FOUR defects and every one was in THIS HARNESS rather
    than in the kernel: 2-D cases declaring a PEC on the invariant z axis, which
    ``Grid`` refuses by name; a mutation whose replacement opened a suite on the
    block's first line, so the mutant failed to IMPORT and scored as unarmed;
    ``MATERIAL`` naming files the inventory scan does not produce, which matched
    nothing and thereby disabled both the vacuity floor and ``material_changed``;
    and mutations scored on a grid that never enters the branch they rewrite. The
    last is now a laptop guard in this product's test file.

    STALE TEXT REMOVED 2026-08-21. Three sentences stood here from before the first
    device run — "the CUDA host was unreachable ... no leg below marked ``device``
    has ever executed and this gate has produced no artifact ... nothing in this
    repository claims byte identity for this product" — under a header that already
    said RELEASED. Two device runs and a weld had happened since. A header that
    contradicts itself is read for whichever half the reader wanted.

    The ``--no-device`` legs still run on a laptop and are labelled as such in the
    payload; ``release.released`` is written only by a run that reached a device.

THE CLAIM THIS GATE IS ALLOWED TO SUPPORT, once it runs: for every configuration
:func:`~meep_gpu.triton_kernels.folded_fused_magnetic_pair.folded_fused_magnetic_pair_coverage`
admits, ONE launch of ``folded_fused_curl_constitutive_B`` leaves the engine in a
state that is BIT-IDENTICAL, PER COMPLETE ``driver.step()``, to

  * the CuPy array path (``stepping.step_B`` / ``fill_symmetry_bc_B`` /
    ``zero_metal_B`` / ``fill_folded_far_ghosts_B`` / ``update_H``), and
  * the SEPARATELY CERTIFIED Triton products it replaces — the folded curl
    (``symmetry.FoldedPmlCurlPlan`` on ``step_B``), the mirror ghost fill
    (``symmetry.MirrorGhostFillPlan`` on family ``B``) and the folded constitutive
    (``symmetry.plan_folded_constitutive`` on side ``H``), with ``zero_metal_B``
    on the array path in both routes because no Triton product owns it,

over every allocated volume: the primaries, the split-field PML auxiliaries and
the constitutive ``f_w`` history. Comparison is on the uint32 view. ``allclose``
appears nowhere.

WHAT THE GATE REFUSES TO INFER.

* **Bytes alone cannot prove the fused path ran.** A silent fallback to the array
  path is byte-identical to the array path by construction. Every launch is
  counted through a proxy that owns the kernel object, every substitution is
  counted per route in the installer, and one ARMED HARNESS MUTATION removes the
  substitution so the counters — not the bytes — are what catches it.
* **A no-op agreeing with a no-op is trivially identical.** Every compared array
  must MOVE during the leg; a leg whose magnetic trio is frozen in all three
  routes is armed and must be caught by the moved-state census.
* **A mutation that cannot be seen is not a passing mutation.** The
  destination-coefficient mutation (M1) is measured on the laptop FIRST, and the
  measurement says it is invisible in every real folded configuration probed —
  see ``NO-DEVICE LEG 2`` — so the gate carries a SYNTHETIC bare-array leg with a
  hand-built coefficient gradient across stored cells 0..2, and reports M1's
  device verdict on the real cases as an expected null WITH that reason attached.

===========================================================================
THE SEAM, AND WHY IT IS THE MAGNETIC ONE
===========================================================================

The driver runs five passes (driver.py:3281-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

The magnetic sources are refused by the predicate; ``fill_folded_far_ghosts_B`` is
refused with the folded PERIODIC axis it needs; the other two are carried inline.
From the 186-row Triton census
(``results/predicate_coverage_2026-08-16_wired``) this admits **45 rows** against
2 for the same construction on the electric half — the funnel is reproduced by
``results/triton_folded_fused_magnetic_pair_census_2026-08-19/count_corpus_admission.py``.

POLICY, stamped and unchanged: ``num_warps=1`` (the settled cross-sub-step
policy), ``BLOCK=kernels.DEFAULT_BLOCK``,
``enable_fp_fusion=kernels.ENABLE_FP_FUSION`` (False). This gate does not tune and
does not time.

Usage::

    # laptop, no CUDA, no Triton — the legs that do not need a device
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -u \\
        parity/meep_gpu/probe_triton_folded_fused_magnetic_pair.py --no-device \\
        --out parity/meep_gpu/results/<fresh-dir>/no_device.json

    # CUDA host, verified-empty device — the full gate
    CUDA_VISIBLE_DEVICES=<verified-empty device> python -u \\
        parity/meep_gpu/probe_triton_folded_fused_magnetic_pair.py \\
        --out parity/meep_gpu/results/<fresh-dir>/gate.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
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

SEED = 20260819

#: Per-case driver overrides, and what each key is for. A case that names none
#: takes these.
#:
#:   ``resolution`` / ``courant``   the corpus row a case stands for is built at
#:       ITS numbers rather than at this table's, so "the same shape" can mean the
#:       same shape and not merely a similar one;
#:   ``pml``   absorber cells on each absorbing face (a folded axis absorbs on its
#:       HIGH face alone — the low face is the mirror plane);
#:   ``expect_ghost_destinations``   the number of ghost cells ONE source lane owns
#:       for its most-composed component, ASSERTED per leg by :func:`verdict_of`.
#:       It is the non-vacuity floor for composition DEPTH, and it exists because
#:       depth is exactly what this table used to be short of: a table topping out
#:       at two folded axes reaches 3 and the kernel's triple-composite blocks
#:       never execute at all.
DEFAULT_OPTIONS: Dict[str, Any] = {"resolution": 12.0, "courant": 0.35, "pml": 5}

#: (name, cell, boundaries, mirrors, steps, options). A 2-D case is spelled with
#: ``z = 0.0``, which is what :func:`build_driver` reads to choose ``dimensions``.
#:
#: THE PARITIES AND THE WALL SET ARE BOTH SWEPT, and neither is decoration. An
#: EVEN plane makes the parity mutation (m2) invisible — the multiply is by +1 —
#: so the odd row is what gives that mutation anywhere to be caught. A PERIODIC
#: unfolded axis turns its ``ZM`` flag off and its ghost rule from mask-to-zero to
#: wrap, which is the only case where the wall-clear mutation (m8) has an axis it
#: does NOT fire on.
CASES: Tuple[Tuple[str, Tuple[float, float, float], Any,
                   Tuple[Tuple[str, int], ...], int, Dict[str, Any]], ...] = (
    # THE Z AXIS IS PERIODIC IN EVERY CASE, NOT METALLIC. These cells are 2-D
    # (z = 0.0), so z is translationally invariant: MEEP does not loop over it and
    # it carries no boundary condition at all. A metallic declaration there is not
    # a wall but a polarization filter — it zeroes every component whose z Yee
    # shift is 0, which in a 2-D run is Ex, Ey and Hz, the entire TE polarization —
    # and Grid refuses it by name (grid.py:842-877). Periodic is what a one-pixel
    # invariant direction is in MEEP too (fields.cpp nosize_direction). The x and y
    # walls, which are what these cases fold against, are unchanged.
    ("even_y_fold_metallic", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", 1),), 10, {}),
    ("odd_y_fold_metallic", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("Y", -1),), 10, {}),
    ("even_x_fold_metallic", (3.0, 3.2, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("X", 1),), 10, {}),
    ("two_folds_metallic", (3.0, 3.0, 0.0),
     {"x": "metallic", "y": "metallic", "z": "periodic"}, (("X", 1), ("Y", -1)),
     10, {}),
    ("y_fold_periodic_x", (3.2, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"}, (("Y", 1),), 10, {}),
    # ---- THE FOLDED PERIODIC ROWS, added 2026-08-20 with the far carry --------
    # `fill_folded_far_ghosts_B` is LIVE on every one of these (driver.py:3287)
    # and the kernel now images it. They are additive: the MIRROR_METALLIC rows
    # above stay, because the two terminations emit DIFFERENT text — the top-plane
    # mask and the whole far carry exist only here, and the metallic rows are the
    # only ones where the top plane is owned and stepped.
    #
    # THE CELL EXTENT IS THE FULL-COUNT PARITY, and it is a measured lever, not a
    # size: `_far_reflect_rows` is `n_full - stored + 2`, which is `stored - 2` at
    # an even full count and `stored - 3` at an odd one. At resolution 12 a 3.0
    # cell folds to stored 20 with reflect 18 (= stored - 2) and a 3.1 cell to
    # stored 21 with reflect 18 (= stored - 3). A sweep carrying only the even
    # extent would arm the fixed-`n - 2` mutation and watch it do nothing.
    ("even_y_fold_periodic", (3.2, 3.0, 0.0),
     {"x": "metallic", "y": "periodic", "z": "periodic"}, (("Y", 1),), 10, {}),
    ("odd_y_fold_periodic", (3.2, 3.1, 0.0),
     {"x": "metallic", "y": "periodic", "z": "periodic"}, (("Y", -1),), 10, {}),
    ("two_folds_periodic_even", (3.0, 3.0, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", -1)), 10, {}),
    # THE MUTATION LEG'S GRID. Two folded PERIODIC axes at an ODD full count and
    # MIXED parities: every component carries a near ghost and a far one, one
    # carries the near/far composite, one carries the far/far corner with the
    # PRODUCT of two parities, and the reflect row is `stored - 3` so the fixed
    # `n - 2` is a whole cell wrong.
    ("two_folds_periodic_odd", (3.1, 3.1, 0.0),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", -1), ("Y", 1)), 10, {}),
    # ONE FOLD OF EACH TERMINATION on the same grid: x images a far ghost and y
    # does not, so the two code paths run against each other in one launch and a
    # carry that keyed off "is folded" rather than "is folded PERIODIC" is wrong
    # here and nowhere else.
    ("mixed_terminations", (3.0, 3.0, 0.0),
     {"x": "periodic", "y": "metallic", "z": "periodic"},
     (("X", 1), ("Y", -1)), 10, {}),
    # ---- THE THREE-AXIS ROWS, added 2026-08-21 -------------------------------
    # WHY THIS TABLE NEEDED THEM, in the words of the refutation that found the
    # gap: the previous cut correctly argued that a gate whose case table stopped
    # at ONE folded axis would have released the ownership defect — and then
    # shipped a THREE-axis path its own case table stopped short of. Every
    # `mirrors` value in the 2026-08-20 artifact is one of [[X,-1],[Y,1]],
    # [[X,1],[Y,-1]], [[X,1]], [[Y,-1]], [[Y,1]], so the kernel's triple-composite
    # blocks (folded_fused_magnetic_pair.py, the three
    # `if NEAR_a and (FAR_b and FAR_c):` calls) were SHIPPED CODE NO DEVICE LEG
    # EVER EXECUTED. They were byte-verified on the Metal twin's
    # `xyz_periodic_odd_3d` row and on no Triton leg at all.
    #
    # THREE FOLDED PERIODIC AXES. Metal's row is the worked precedent and this is
    # its Triton counterpart: every axis folded, every axis periodic, so NEAR_a
    # and FAR_a are true on all three and ONE source lane owns SEVEN ghost cells
    # per component — a near plane, two far planes, three pairs and the triple.
    # Seven is the largest number the ownership rule can produce on this family
    # and `expect_ghost_destinations` is what turns it from a claim into a floor.
    # MIXED PHASES (-1, +1, -1), Metal's own, so the triple weight PHX*PHY*PHZ is
    # not a product of like signs.
    # THE ODD FULL COUNT rides along: 2.1 at resolution 12 stores 15 with reflect
    # 12, which is `stored - 3`, so the fixed `n - 2` is a whole cell wrong here
    # too.
    ("xyz_periodic_odd_3d", (2.1, 2.1, 2.1),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", -1), ("Y", 1), ("Z", -1)), 10,
     {"expect_ghost_destinations": 7}),
    # THE SAME SHAPE AS A ROW THIS PRODUCT CLAIMS. tests:
    # TestModeDecomposition.test_grating_3d is cell (1.1, 0.8, 8.5) at resolution
    # 25, courant 0.5, two even mirror planes on x and y, and it lifts to grid
    # shape [16, 12, 213] (`results/predicate_coverage_2026-08-16_wired/
    # tests_param.jsonl`). The Triton census counts it in this family's 45 admitted
    # rows and, until now, no Triton device leg ran anything like it — the whole
    # table was 2-D. Metal covers 3-D two-folds (`xy_periodic_*_wall_z_3d`); this
    # is that coverage on this board, at the claimed row's own numbers rather than
    # at a similar-looking substitute.
    #
    # z IS PLAIN PERIODIC, which is what MEEP's k_point (0, 0, 0) with a PML in z
    # is: the absorber terminates the wave, the lattice still wraps. So this row
    # carries no wall at all, and that is deliberate — it is the only 3-D leg
    # whose ZM flags are all off.
    ("grating_3d_two_folds", (1.1, 0.8, 8.5),
     {"x": "periodic", "y": "periodic", "z": "periodic"},
     (("X", 1), ("Y", 1)), 10,
     {"resolution": 25.0, "courant": 0.5, "expect_ghost_destinations": 3}),
    # THE LAST TWO GUARDS NOTHING ENTERED. Counting the kernel's live constexpr
    # branches against this table (``test_every_constexpr_branch_the_kernel_SHIPS_
    # is_entered_by_some_case``) put the triple composite in company: 20 of the 45
    # were unreachable on the 2026-08-20 table, and the two rows above close 18.
    # The two left over are ``if BCY == PERIODIC:`` — Y UNFOLDED and periodic, the
    # ghost-rule wrap, which every previous row had folded or walled — and
    # ``if ZM_Z:``, the wall clear on z, which no 2-D row can carry at all: a 2-D
    # cell's z is translationally invariant and ``Grid`` refuses a PEC there by
    # name (grid.py:842-877). One 3-D row with a folded PERIODIC x, a plain
    # periodic y and a METALLIC z closes both, and gives this table its only 3-D
    # leg with a live wall.
    ("x_fold_periodic_wall_z_3d", (2.4, 2.0, 1.6),
     {"x": "periodic", "y": "periodic", "z": "metallic"},
     (("X", -1),), 10, {"expect_ghost_destinations": 1}),
)

#: Steps per armed mutation. A mutation needing more than this to become
#: byte-visible is reported as a null WITH its launch evidence.
MUTATION_STEPS = 3

#: The driver call sites this product spans, in driver order (driver.py:3281-3289).
SEAM_PASSES: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
)

#: The names that MUST appear in the dynamic state inventory. Scanned rather than
#: listed so a renamed volume cannot silently drop out of the comparison; this
#: tuple is the tripwire for the scan itself shrinking.
REQUIRED = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)

#: Read-only material inputs: a leg that changes one of these measured something
#: other than the step.
#: THE READ-ONLY MATERIAL VOLUMES, under the names ``inventory`` really finds.
#:
#: MEASURED 2026-08-20: this tuple read ``("inverse_epsilon", "epsilon")`` while
#: the scan takes its keys from ``vars(fields)``, which names them ``eps`` and
#: ``inv_eps``. Nothing matched, and that silently broke BOTH directions of the
#: check it feeds: the vacuity floor demanded that two read-only INPUTS move,
#: which no correct kernel does, so every device leg failed as VACUOUS; and
#: ``material_changed`` -- a kernel writing its own material -- could never fire,
#: because the set it scans was always empty. A name that matches nothing is not a
#: weaker check, it is two absent ones. ``_assert_material_names_are_real`` below
#: is what stops a rename from doing this again.
MATERIAL = ("eps", "inv_eps")


def _assert_material_names_are_real(found: Dict[str, Any]) -> None:
    """Every MATERIAL name must appear in the inventory the scan actually built."""
    missing = [name for name in MATERIAL if name not in found]
    if missing:
        raise RuntimeError(
            f"MATERIAL names {missing} are not in the scanned inventory "
            f"{sorted(found)}: the vacuity floor would fire on a read-only input "
            f"and material_changed would never fire. Fix the names, do not "
            f"loosen the floor.")

_TEMPORARY: List[str] = []


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Serialise the payload, provenance-stamped, atomically."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    # THE POLICY STAMP IS RE-READ, NOT CARRIED. Taken once at install time it
    # records every counter at zero, because nothing had compiled yet — a record
    # that cannot tell an installed policy apart from an inert one.
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
        "meep_gpu/triton_kernels/folded_fused_magnetic_pair.py",
        "meep_gpu/triton_kernels/symmetry.py",
        "meep_gpu/triton_kernels/kernels.py",
        "meep_gpu/triton_kernels/coverage.py",
        "meep_gpu/triton_kernels/launch.py",
        "meep_gpu/stepping.py",
        "meep_gpu/driver.py",
        "meep_gpu/fields.py",
        "meep_gpu/subnormal_policy.py",
        "meep_gpu/test_triton_folded_fused_magnetic_pair.py",
        os.path.relpath(os.path.abspath(__file__), API_ROOT),
    )
    return {name: sha256(os.path.join(API_ROOT, name)) for name in names}


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 1 — the transcription, read off the shipped source
# ---------------------------------------------------------------------------

def _body(function: Any) -> str:
    return textwrap.dedent(inspect.getsource(function))


def _shipped_text(name: str) -> str:
    """One shipped function's EXACT source text, with its docstring removed.

    Read from the FILE rather than imported: this leg has to run on a host with no
    Triton, where importing ``kernels.py`` raises, so the transcription check bites
    at the merge bar rather than only on a device run.

    The text is exact — never ``ast.unparse``d — because what is being checked is
    the PARENTHESISATION, and unparsing re-derives minimal parentheses and would
    silently rewrite ``dtdx * ((c_y - c) + (b - b_z))`` into a different grouping
    than the one whose float32 bits the gate is about.

    The docstring IS stripped, and that is not cosmetic: every one of these bodies
    documents the codes and branches it does NOT carry, so a text search over the
    raw source finds those names in prose and reports them as live code.
    """
    import ast  # noqa: PLC0415

    path = os.path.join(API_ROOT, "meep_gpu", "triton_kernels",
                        {"pml_curl_step_folded": "symmetry.py",
                         "constitutive_step": "kernels.py",
                         "fused_curl_constitutive_B": "kernels.py",
                         "folded_fused_curl_constitutive_B":
                             "folded_fused_magnetic_pair.py",
                         "_carry_ghost": "folded_fused_magnetic_pair.py"}[name])
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

#: The constitutive accumulation, from ``kernels.constitutive_step``'s SCALE=0
#: arm. Two separate accumulations, left to right; flattening them is a different
#: float32 number and neither multiply may contract into an FMA.
CONSTITUTIVE_LINES = (
    "a0v = a0v + kp_0 * src0",
    "a0v = a0v - km_0 * prev0",
    "a1v = a1v + kp_1 * src1",
    "a1v = a1v - km_1 * prev1",
    "a2v = a2v + kp_2 * src2",
    "a2v = a2v - km_2 * prev2",
)

#: The carry's constitutive accumulation at a DESTINATION, in the same shape and
#: with whichever coefficient pair that destination takes. It lives in the
#: ``_carry_ghost`` device function as of 2026-08-20: a B component can now be the
#: source of up to SEVEN ghost cells (a near plane, two far planes and every
#: composition), and seven inlined copies of the same seven statements would be
#: seven places for one of them to drift.
CARRY_LINES = (
    "acc = acc + kp_d * ghost",
    "acc = acc - km_d * prev",
)

#: The top-plane ownership mask the far carry brings with it. Every line must be
#: BYTE-PRESENT in ``symmetry.pml_curl_step_folded``'s own body, which is where it
#: is transcribed from; a line here that is not there is a hand-written mask.
TOP_PLANE_LINES = (
    "curl0 = tl.where(last_y, 0.0, curl0)",
    "curl0 = tl.where(last_z, 0.0, curl0)",
    "curl1 = tl.where(last_x, 0.0, curl1)",
    "curl1 = tl.where(last_z, 0.0, curl1)",
    "curl2 = tl.where(last_x, 0.0, curl2)",
    "curl2 = tl.where(last_y, 0.0, curl2)",
)

#: Every ghost cell one source lane owns, as the call the fused body must make for
#: it. Written out rather than generated: this is the list the ownership rule
#: DERIVES (a near plane, each far plane, and every composition of them, for each
#: of the three components), and generating it here from the same rule the kernel
#: uses would be a mirrored evaluator rather than a check.
CARRY_CALLS = (
    "_carry_ghost(f0, w0, h0, idx + df_y,",
    "_carry_ghost(f0, w0, h0, idx + df_z,",
    "_carry_ghost(f0, w0, h0, idx + df_y + df_z,",
    "_carry_ghost(f0, w0, h0, idx + dn_x,",
    "_carry_ghost(f0, w0, h0, idx + dn_x + df_y,",
    "_carry_ghost(f0, w0, h0, idx + dn_x + df_z,",
    "_carry_ghost(f0, w0, h0, idx + dn_x + df_y + df_z,",
    "_carry_ghost(f1, w1, h1, idx + df_x,",
    "_carry_ghost(f1, w1, h1, idx + df_z,",
    "_carry_ghost(f1, w1, h1, idx + df_x + df_z,",
    "_carry_ghost(f1, w1, h1, idx + dn_y,",
    "_carry_ghost(f1, w1, h1, idx + dn_y + df_x,",
    "_carry_ghost(f1, w1, h1, idx + dn_y + df_z,",
    "_carry_ghost(f1, w1, h1, idx + dn_y + df_x + df_z,",
    "_carry_ghost(f2, w2, h2, idx + df_x,",
    "_carry_ghost(f2, w2, h2, idx + df_y,",
    "_carry_ghost(f2, w2, h2, idx + df_x + df_y,",
    "_carry_ghost(f2, w2, h2, idx + dn_z,",
    "_carry_ghost(f2, w2, h2, idx + dn_z + df_x,",
    "_carry_ghost(f2, w2, h2, idx + dn_z + df_y,",
    "_carry_ghost(f2, w2, h2, idx + dn_z + df_x + df_y,",
)

#: The TRIPLE-COMPOSITE weights, which are the deepest composition the ownership
#: rule derives and the reason this list is separate from :data:`CARRY_CALLS`.
#: The cell at stored 0 on the near axis AND at ``last`` on BOTH far axes carries
#: the product of all three planes' parities — ``mirror_parity`` composed three
#: times — and a carry that stopped at two would leave it holding the doubly-imaged
#: value of an unfilled row. The call itself is in ``CARRY_CALLS``; this is the
#: WEIGHT, checked separately because the two can drift apart in either direction.
TRIPLE_WEIGHT_LINES = (
    "PHX * PHY * PHZ * v0,",
    "PHY * PHX * PHZ * v1,",
    "PHZ * PHX * PHY * v2,",
)


def transcription_leg() -> Dict[str, Any]:
    """Every arithmetic line of the fused body traced to the source it came from."""
    fused = _statements(_shipped_text("folded_fused_curl_constitutive_B"))
    folded_curl = _statements(_shipped_text("pml_curl_step_folded"))
    constitutive = _statements(_shipped_text("constitutive_step"))
    plain_pair = _statements(_shipped_text("fused_curl_constitutive_B"))

    findings: List[str] = []
    for line in CURL_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain the curl line {line!r}")
        if line not in folded_curl:
            findings.append(
                f"symmetry.pml_curl_step_folded no longer contains {line!r}; the "
                f"fused body's transcription source has moved")
    for line in CONSTITUTIVE_LINES:
        if line not in fused:
            findings.append(f"the fused body does not contain {line!r}")
    # The constitutive half is `constitutive_step`'s shape with its own register
    # names; what must survive verbatim is the grouping, which is what the plain
    # fused pair also carries. Compare against THAT, since the substitution
    # (`src = v`) is the same one.
    for line in ("a0 = a0 + kp_0 * src0", "a0 = a0 - km_0 * prev0"):
        if line not in constitutive and line not in plain_pair:
            findings.append(
                f"neither constitutive_step nor fused_curl_constitutive_B contains "
                f"{line!r}; the constitutive grouping's source has moved")
    carry = _statements(_shipped_text("_carry_ghost"))
    for line in CARRY_LINES:
        if line not in carry:
            findings.append(
                f"_carry_ghost does not contain the carry line {line!r}")
    # THE SEVEN STATEMENTS AND THEIR ORDER. `prev` is read BEFORE the workspace is
    # written; a reordering here is the one thing the constitutive cannot survive.
    # `_statements` keeps the `def` line, so the body starts at index 1.
    if carry[1:3] != ["prev = tl.load(w + dst, mask=mask, other=0.0)",
                      "tl.store(w + dst, ghost, mask=mask)"]:
        findings.append(
            f"_carry_ghost no longer reads f_w BEFORE writing it: {carry[1:3]}")
    joined = "\n".join(fused)
    missing_calls = [call for call in CARRY_CALLS if call not in joined]
    if missing_calls:
        findings.append(
            f"the fused body does not carry every ghost one source lane owns; "
            f"missing {missing_calls}")
    missing_triples = [line for line in TRIPLE_WEIGHT_LINES if line not in joined]
    if missing_triples:
        findings.append(
            f"the fused body does not weight the triple composite by all THREE "
            f"planes' parities; missing {missing_triples}")
    # THE TOP-PLANE MASK ARRIVED WITH THE FAR CARRY and must be the certified
    # emitter's own text, not a hand-written twin.
    for line in TOP_PLANE_LINES:
        if line not in fused:
            findings.append(
                f"the fused body does not contain the top-plane mask line {line!r}")
        if line not in folded_curl:
            findings.append(
                f"symmetry.pml_curl_step_folded no longer contains {line!r}; the "
                f"top-plane mask's transcription source has moved")

    # The ownership restructure, structurally: the destination's B / H / f_w_H
    # traffic must be MASKED OFF, not merely unused. A lane that loaded and
    # discarded would still read a word another lane writes.
    for needle in ("mask=own0", "mask=own1", "mask=own2"):
        if sum(1 for line in fused if needle in line) < 4:
            findings.append(
                f"{needle} appears on fewer than four lines; the destination lane's "
                f"B/f_w/H traffic is not fully masked off")
    # THE FLIP OF 2026-08-20. This check used to require the OPPOSITE — that
    # MIRROR_PERIODIC appear nowhere, because the predicate refused it and a branch
    # on it would have been unreachable. The kernel carries
    # fill_folded_far_ghosts_B now, so the code must be branched on: the top-plane
    # mask keys off it, and a kernel that admitted the axis without the branch
    # would step a plane the fill then images over.
    if not any("MIRROR_PERIODIC" in line for line in fused):
        findings.append(
            "the fused body never mentions MIRROR_PERIODIC; the top-plane mask "
            "fires only on that code and the far carry needs it")
    # The destination coefficient must be read at index 0, not at the source's.
    for needle in ("kp0 + origin", "km0 + origin", "kp1 + origin", "km1 + origin",
                   "kp2 + origin", "km2 + origin"):
        if not any(needle in line for line in fused):
            findings.append(
                f"the carry does not read {needle!r}: the destination's constitutive "
                f"coefficient is on the component's OWN axis and the fill images "
                f"ALONG that axis, so it is not the source's")
    return {
        "leg": "transcription",
        "device": False,
        "fused_statements": len(fused),
        "curl_lines_checked": len(CURL_LINES),
        "constitutive_lines_checked": len(CONSTITUTIVE_LINES),
        "carry_lines_checked": len(CARRY_LINES),
        "carry_calls_checked": len(CARRY_CALLS),
        "triple_weight_lines_checked": len(TRIPLE_WEIGHT_LINES),
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 2 — is the moved coefficient index observable at all?
# ---------------------------------------------------------------------------

def coefficient_reach_leg() -> Dict[str, Any]:
    """Does any REAL folded configuration separate ``kps[0]`` from ``kps[2]``?

    The B half's new hazard is that the near fill images ALONG the component's own
    axis, which is the axis its constitutive coefficient is indexed on — so the
    fill's source and destination take DIFFERENT coefficient entries, unlike the
    electric half. That is true by construction. Whether it is VISIBLE is a
    separate question and this leg answers it by measurement rather than by
    argument, because the answer decides how the device gate reports M1.

    The mirror plane is the folded axis's LOW face and MEEP puts no absorber
    there (``driver.setup_pml`` takes ``{"high": n}`` on a folded axis), so
    ``kps``/``kms`` at stored 0 and stored 2 may well be the same word.
    """
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    rows: List[Dict[str, Any]] = []
    separated = 0
    for boundaries in ("metallic", "periodic"):
        for thickness in (0.2, 0.4, 0.8, 1.2):
            for resolution in (10.0, 20.0):
                grid = Grid(resolution=resolution, cell_size=(1.6, 3.0, 1.0),
                            boundaries=boundaries, symmetry=("Y",))
                pml = PML(grid=grid, thickness=thickness)
                for stem in ("kps_y", "kms_y", "kps_y_h", "kms_y_h"):
                    table = np.ravel(np.asarray(getattr(pml, stem)))
                    differs = bool(float(table[0]) != float(table[2]))
                    separated += int(differs)
                    rows.append({
                        "boundaries": boundaries, "thickness": thickness,
                        "resolution": resolution, "table": stem,
                        "at_0": float(table[0]), "at_2": float(table[2]),
                        "separated": differs,
                    })
    # The unfolded control: the same question on an axis with a real low face.
    grid = Grid(resolution=20.0, cell_size=(1.6, 3.0, 1.0),
                boundaries="metallic", symmetry=("Y",))
    pml = PML(grid=grid, thickness=0.8)
    control = np.ravel(np.asarray(pml.kps_x))
    return {
        "leg": "coefficient_reach",
        "device": False,
        "cases": len(rows),
        "separated": separated,
        "rows": rows,
        "unfolded_control_axis": "x",
        "unfolded_control_at_0": float(control[0]),
        "unfolded_control_at_2": float(control[2]),
        "unfolded_control_separated": bool(float(control[0]) != float(control[2])),
        "verdict": (
            "the destination coefficient differs from the source's in "
            f"{separated}/{len(rows)} real folded configurations; the unfolded "
            "control separates, which is what makes this a measurement about the "
            "FOLD rather than about the probe"),
        # Non-vacuity: if the control did not separate, the whole leg would be
        # measuring a broken reader rather than the absorber's reach.
        "passed": bool(float(control[0]) != float(control[2])),
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 3 — the design sweep, on NumPy
# ---------------------------------------------------------------------------

def _array_path_seam(fields, pml, driver_module) -> None:
    """The four driver passes this product replaces, in driver order."""
    driver_module.step_B(fields, pml)
    driver_module.fill_symmetry_bc_B(fields)
    driver_module.zero_metal_B(fields)
    driver_module.fill_folded_far_ghosts_B(fields)
    driver_module.update_H(fields, pml)


def _emulated_seam(fields, pml, driver_module, *, carry_fill: bool = True,
                   destination_coefficient: str = "own",
                   order: str = "fill_then_constitutive") -> None:
    """The FUSED semantics, spelled with array ops so a laptop can execute them.

    THIS IS NOT THE KERNEL AND DOES NOT PRETEND TO BE. It cannot see a memory
    hazard, a register reuse or a rounding order inside one launch — those are the
    device gate's to measure. What it CAN decide is the DESIGN question the carry
    answers: which value reaches ``update_H`` at the fill's destination, at which
    coefficient index, and in which order. Each of the three is a knob, and the
    leg reports which flips the array path can see.

    ``destination_coefficient='source'`` is expressed by editing the coefficient
    TABLE rather than by patching the result: setting entry 0 to entry 2 on the
    folded component's own axis is exactly "the destination lane reused the source
    lane's pair", because the fill's destination plane IS index 0 on that axis.
    """
    import meep_gpu.stepping as stepping  # noqa: PLC0415
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    grid = fields.grid
    folded = [axis for axis in range(3) if grid.is_mirrored(axis)]

    # --- the curl half, at every cell (the kernel's destination lane discards
    # its own B, and the fill overwrites it either way).
    driver_module.step_B(fields, pml)

    if carry_fill and order == "fill_then_constitutive":
        stepping.fill_symmetry_bc_B(fields)
    driver_module.zero_metal_B(fields)

    # --- the constitutive half (stepping.update_H), with the coefficient knob.
    for index, (component, source_name, axis_name) in enumerate(
            stepping.H_CONSTITUTIVE_TERMS):
        target = getattr(fields, component)
        source = getattr(fields, source_name)
        history = getattr(fields, "f_w_" + component)
        kps, kms = stepping._constitutive_coefficients(pml, axis_name,
                                                       half_integer=False)
        kps, kms = np.asarray(kps), np.asarray(kms)
        if (destination_coefficient == "source" and index in folded
                and IYEE_SHIFTS[source_name][index] == 0 and kps.size > 2):
            kps, kms = kps.copy(), kms.copy()
            kps.reshape(-1)[0] = kps.reshape(-1)[2]
            kms.reshape(-1)[0] = kms.reshape(-1)[2]
        previous = history.copy()
        history[...] = source
        target += kps * history
        target -= kms * previous

    if carry_fill and order == "constitutive_then_fill":
        # The defect: B is imaged only AFTER update_H has already consumed the
        # unfilled value, so H and f_w_H at the destination carry the stepped
        # displacement rather than the mirror image of stored cell 2.
        stepping.fill_symmetry_bc_B(fields)


def design_sweep_leg() -> Dict[str, Any]:
    """Each design choice flipped, on NumPy, against the array-path composition.

    A choice whose flip is byte-identical is reported as a NULL WITH ITS REASON —
    that is a measurement about the configuration, not a licence to drop the
    choice from the kernel.
    """
    import meep_gpu.driver as driver_module  # noqa: PLC0415
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    def build(mirrors, boundaries="metallic", cell=(1.6, 3.0, 1.0)):
        grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                    symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
        fields = Fields(grid=grid, force_complex_fields=False)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness=0.4)
        rng = np.random.default_rng(SEED)
        for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                     "fu_Bx", "fu_By", "fu_Bz",
                     "f_w_Hx", "f_w_Hy", "f_w_Hz"):
            array = getattr(fields, name, None)
            if array is None:
                continue
            array[...] = rng.uniform(-0.25, 0.25,
                                     size=array.shape).astype(array.dtype)
        return fields, pml

    def words(fields) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name in ("Bx", "By", "Bz", "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                     "f_w_Hx", "f_w_Hy", "f_w_Hz"):
            array = getattr(fields, name, None)
            if array is not None:
                out[name] = np.ascontiguousarray(
                    np.asarray(array, dtype=np.float32)).view(np.uint32).ravel().copy()
        return out

    knobs = (
        ("faithful", {}, "null"),
        ("drop_the_near_fill", {"carry_fill": False}, "caught"),
        ("destination_coefficient_is_the_sources",
         {"destination_coefficient": "source"}, "unknown"),
        ("constitutive_before_the_fill", {"order": "constitutive_then_fill"},
         "caught"),
    )
    cases = (("even_y", (("Y", 1),)), ("odd_y", (("Y", -1),)),
             ("even_x", (("X", 1),)), ("two_folds", (("X", 1), ("Y", -1))))

    rows: List[Dict[str, Any]] = []
    for case_name, mirrors in cases:
        reference_fields, reference_pml = build(mirrors)
        before = words(reference_fields)
        _array_path_seam(reference_fields, reference_pml, driver_module)
        oracle = words(reference_fields)
        moved_names = sorted(name for name in oracle
                             if not np.array_equal(oracle[name], before[name]))
        for knob_name, kwargs, expectation in knobs:
            fields, pml = build(mirrors)
            _emulated_seam(fields, pml, driver_module, **kwargs)
            candidate = words(fields)
            differing = sorted(name for name in oracle
                               if not np.array_equal(candidate[name], oracle[name]))
            rows.append({
                "case": case_name, "knob": knob_name,
                "expectation": expectation,
                "identical": not differing,
                "differing_arrays": differing,
                "arrays_moved_by_the_array_path": moved_names,
            })
    faithful = [row for row in rows if row["knob"] == "faithful"]
    vacuous = [row for row in rows if not row["arrays_moved_by_the_array_path"]]
    findings: List[str] = []
    for row in faithful:
        if not row["identical"]:
            findings.append(
                f"{row['case']}: the FAITHFUL emulation diverges from the array "
                f"path on {row['differing_arrays']} — the design, not the kernel, "
                f"is wrong")
    if vacuous:
        findings.append(f"VACUOUS cases (nothing moved): {[r['case'] for r in vacuous]}")
    for row in rows:
        if row["expectation"] == "caught" and row["identical"]:
            findings.append(
                f"{row['case']}/{row['knob']}: expected to be caught and was not")
    return {
        "leg": "design_sweep", "device": False, "rows": rows,
        "findings": findings, "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 4 — does any case ENTER every branch the kernel ships?
# ---------------------------------------------------------------------------

#: Constexpr guards whose body is deliberately unreachable, with the reason. The
#: ONLY entry, and it is a design decision stated in the kernel's own docstring:
#: ``BACKWARD`` is bound to 0 by every builder, so the ``if BACKWARD:`` arm is dead
#: in every configuration this product admits. It is carried anyway to keep the
#: curl half a VERBATIM copy of ``symmetry.pml_curl_step_folded`` rather than a
#: hand-specialised one — the transcription leg is what that buys, and a
#: hand-trimmed copy would be a different kind of risk, not less of it.
DELIBERATELY_UNREACHABLE: Tuple[str, ...] = ("BACKWARD",)


def _live_constexpr_guards(text: str) -> List[Tuple[int, str]]:
    """Every ``if <constexpr>:`` in the kernel whose body some case could reach.

    The ``if BACKWARD:`` suites are skipped WHOLE — header and body — because
    ``BACKWARD`` is 0 in every admitted configuration and their nested guards are
    dead with them. Their ``else:`` arms are not skipped: they are the live half.
    """
    out: List[Tuple[int, str]] = []
    skip_below: Optional[int] = None
    for index, raw in enumerate(text.splitlines()):
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if skip_below is not None:
            if stripped and indent <= skip_below:
                skip_below = None
            else:
                continue
        if any(stripped.startswith(f"if {name}")
               for name in DELIBERATELY_UNREACHABLE):
            skip_below = indent
            continue
        if stripped.startswith("if ") and stripped.endswith(":"):
            out.append((index + 1, stripped[3:-1].strip()))
    return out


def branch_reachability_leg() -> Dict[str, Any]:
    """SHIPPED CODE NO LEG EXECUTES, counted rather than argued.

    THE FINDING THIS LEG EXISTS FOR, and it was an adversarial verifier's before it
    was this file's: on 2026-08-20 the kernel shipped three-axis folded-PERIODIC
    blocks — the ``if NEAR_a and (FAR_b and FAR_c):`` triples — and NO DEVICE LEG
    EVER EXECUTED THEM. Every ``mirrors`` value in that artifact carried at most
    two folded axes. Byte identity over ten steps on ten cases says nothing about a
    branch none of the ten compiled.

    THE GENERAL FORM IS WORSE THAN THE INSTANCE. Counting every live constexpr
    guard in the kernel against that table gives 20 of 45 unreachable, not 3: the
    whole z half of the carry, the z wall clear, and the ghost-rule wrap on an
    unfolded periodic Y. A case list is not a coverage argument until something
    counts it.

    THE EVALUATION IS THE ENGINE'S, NOT A MODEL OF IT. Each case's constexprs come
    from :func:`case_constexprs`, which reads ``folded_axis_kinds``,
    ``zero_metal_axes`` and ``mirror_phases`` off a real grid and derives
    ``near``/``far`` from ``bc`` exactly as the plan does. The guard text comes from
    the shipped source. Neither side is a re-implementation, so this cannot pass by
    mirroring the kernel's own mistake.
    """
    text = _shipped_text("folded_fused_curl_constitutive_B")
    guards = _live_constexpr_guards(text)
    environments = [(case[0], case_constexprs(case)) for case in CASES]
    rows: List[Dict[str, Any]] = []
    findings: List[str] = []
    for line, condition in guards:
        entered = sorted(
            name for name, environment in environments
            if eval(condition, {"__builtins__": {}}, dict(environment)))  # noqa: S307
        rows.append({"line": line, "condition": condition,
                     "entered_by": entered, "cases": len(entered)})
        if not entered:
            findings.append(
                f"line {line}: `if {condition}:` is shipped and NO case in this "
                f"table compiles it — the leg would launch a kernel that does not "
                f"contain those lines, and every byte row would be silent about "
                f"them")
    return {
        "leg": "branch_reachability",
        "device": False,
        "guards": len(guards),
        "deliberately_unreachable": list(DELIBERATELY_UNREACHABLE),
        "cases": [{"case": name,
                   "shape": list(environment["shape"]),
                   "ghost_destinations": list(environment["ghost_destinations"])}
                  for name, environment in environments],
        "deepest_ghost_destinations": max(
            max(environment["ghost_destinations"])
            for _name, environment in environments),
        "rows": rows,
        "findings": findings,
        "passed": not findings,
    }


# ---------------------------------------------------------------------------
# NO-DEVICE LEG 5 — the predicate battery
# ---------------------------------------------------------------------------

def predicate_leg() -> Dict[str, Any]:
    """Every clause of the seam predicate, exercised on real grids."""
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.triton_kernels import folded_fused_magnetic_pair as product  # noqa: PLC0415

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
        """``call()`` with this family's carry declaration held down, then restored.

        The OTHER DIRECTION of the 2026-08-30 flip, measured through the shipped
        predicate rather than argued. Restored in a finally so a raising predicate
        cannot leave the module lying to every row after it.
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
           product.folded_fused_magnetic_pair_coverage(fields, pml, ()), True)
    record("folded_metallic_electric_source",
           product.folded_fused_magnetic_pair_coverage(fields, pml, (_Source("D"),)),
           True)
    # THE IN-SEAM MAGNETIC DEPOSIT IS CARRIED AS OF 2026-08-30. Three rows, the same
    # three the electric twin records: the deposit is admitted, a source that cannot
    # publish its index is still refused BY NAME, and holding the flag down puts the
    # original seam refusal straight back.
    record("folded_metallic_magnetic_deposit_is_carried",
           product.folded_fused_magnetic_pair_coverage(
               fields, pml, (_deposit(fields, "Hy"),)), True)
    record("folded_metallic_magnetic_source_without_a_deposit_index",
           product.folded_fused_magnetic_pair_coverage(fields, pml, (_Source("B"),)),
           False, "does not publish the index it writes")
    record("folded_metallic_magnetic_deposit_refused_when_the_flag_is_held_False",
           _with_flag_false(
               product,
               lambda: product.folded_fused_magnetic_pair_coverage(
                   fields, pml, (_deposit(fields, "Hy"),))),
           False, "is magnetic")
    record("undeclared_sources",
           product.folded_fused_magnetic_pair_coverage(fields, pml, None),
           False, "was not declared")
    record("no_pml",
           product.folded_fused_magnetic_pair_coverage(fields, None, ()),
           False, "no active PML")

    # ADMITTED AS OF 2026-08-20. This row asked for a REFUSAL until the far carry
    # landed; the same fixture now measures the flip, and the liveness of the pass
    # it carries is measured on a real driver by ``run_refusals``' own
    # ``folded_periodic_is_now_CARRIED`` row.
    periodic_fields, periodic_pml = build((1.6, 3.0, 1.0), "periodic", (("Y", 1),))
    record("folded_periodic_axis_is_admitted",
           product.folded_fused_magnetic_pair_coverage(periodic_fields,
                                                       periodic_pml, ()),
           True)

    flat_fields, flat_pml = build((1.6, 1.6, 1.0), "metallic", ())
    record("unfolded_grid",
           product.folded_fused_magnetic_pair_coverage(flat_fields, flat_pml, ()),
           False, "no mirror plane is active")

    findings = [row["case"] for row in rows if not row["passed"]]
    return {"leg": "predicate", "device": False, "rows": rows,
            "findings": findings, "passed": not findings}


# ---------------------------------------------------------------------------
# Device side — drivers, inventory, launch counting
# ---------------------------------------------------------------------------

def ghost_destinations(near: Sequence[Any], far: Sequence[Any]) -> Tuple[int, int, int]:
    """Ghost cells ONE source lane owns, per B component, from the plan's booleans.

    Component ``m`` is a NEAR destination on axis ``m`` alone and a FAR destination
    on the two axes that are NOT ``m`` (``IYEE_SHIFTS``: ``Bx (0,1,1)``,
    ``By (1,0,1)``, ``Bz (1,1,0)``), and the two fills COMPOSE — every nonempty
    subset of the destination planes is itself a destination. So the count is
    ``(1 + near_m) * 2 ** (far axes other than m) - 1``: 1 under a single metallic
    fold, 3 under one periodic fold or two folds, and 7 under three folded periodic
    axes, which is the largest this family can produce.

    DERIVED FROM ``plan.near``/``plan.far``, which are themselves derived from
    ``bc`` — so a leg cannot report a composition depth its own constexprs forbid.
    This is the arithmetic the kernel's block structure encodes, counted rather
    than re-implemented: it decides how many ``_carry_ghost`` calls a lane makes,
    not what any of them computes.
    """
    out: List[int] = []
    for component in range(3):
        others = sum(1 for axis in range(3)
                     if axis != component and bool(far[axis]))
        out.append((1 + int(bool(near[component]))) * (2 ** others) - 1)
    return (out[0], out[1], out[2])


def build_grid(cell, boundaries, mirror_specs,
               options: Optional[Dict[str, Any]] = None,
               prefer_gpu: bool = True):
    """The grid and absorber one CASES row names, with no field seeding.

    Split out of :func:`build_driver` so the LAPTOP can build the same geometry
    the device leg compiles against — the guard that every constexpr branch the
    kernel ships is entered by some case reads its constexprs from here, and a
    second spelling of the geometry would be a mirrored evaluator rather than a
    check.

    ``dimensions`` IS READ OFF THE CELL rather than passed: a ``z = 0.0`` extent is
    how this table spells a 2-D case and how MEEP spells one. Hard-coding
    ``dimensions=2`` here is what kept every leg of this gate 2-D until 2026-08-21,
    while the kernel shipped a three-axis path and the census claimed a 3-D corpus
    row.

    ``setup_pml`` NOW RUNS BEFORE ``set_epsilon`` rather than after, which is the
    one behavioural change the split makes, and it was MEASURED rather than argued:
    the two orders leave all 26 stored volumes and every PML coefficient table
    bit-identical after six complete driver steps on ``odd_y_fold_metallic``. (It
    could not have produced a false verdict either way — all three routes are built
    by this one function — but it could have changed what the gate measures, which
    is worth a measurement rather than a shrug.)
    """
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415
    from meep_gpu.grid import Mirror  # noqa: PLC0415

    settings = dict(DEFAULT_OPTIONS)
    settings.update(options or {})
    dimensions = 2 if float(cell[2]) == 0.0 else 3
    driver = FdtdDriver(
        cell_size=cell, resolution=float(settings["resolution"]),
        dimensions=dimensions,
        force_complex_fields=False, courant=float(settings["courant"]),
        boundaries=boundaries,
        symmetry=tuple(Mirror(axis, phase) for axis, phase in mirror_specs),
        prefer_gpu=prefer_gpu, gpu_id=0,
    )
    folded = {axis.lower() for axis, _phase in mirror_specs}
    # A folded axis absorbs on its HIGH face only: the low face is the mirror
    # plane. That is also why the destination coefficient equals the source's in
    # every real case (NO-DEVICE LEG 2).
    cells = int(settings["pml"])
    driver.setup_pml({
        name: ({"high": cells} if name in folded else cells)
        for name in (("x", "y", "z") if dimensions == 3 else ("x", "y"))
    })
    return driver


def case_constexprs(case) -> Dict[str, Any]:
    """The constexprs one CASES row compiles the kernel with, read on NumPy.

    Every value comes from the function the PLAN calls — ``folded_axis_kinds``,
    ``zero_metal_axes``, ``mirror_phases``, and the plan's own
    ``near``/``far`` derivation from ``bc`` — so this cannot disagree with what a
    device leg compiles. Needs neither CuPy nor Triton, which is what lets the
    merge bar refuse a kernel branch no case in this table enters.
    """
    from meep_gpu.triton_kernels import folded_fused_magnetic_pair as product  # noqa: PLC0415
    from meep_gpu.triton_kernels.coverage import zero_metal_axes  # noqa: PLC0415
    from meep_gpu.triton_kernels.symmetry import (  # noqa: PLC0415
        CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
        folded_axis_kinds,
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
            "PERIODIC": int(CODE_PERIODIC),
            "MIRROR_METALLIC": int(CODE_MIRROR_METALLIC),
            "MIRROR_PERIODIC": int(CODE_MIRROR_PERIODIC),
            "BACKWARD": int(product.BACKWARD),
            "BCX": codes[0], "BCY": codes[1], "BCZ": codes[2],
            "NEAR_X": near[0], "NEAR_Y": near[1], "NEAR_Z": near[2],
            "FAR_X": far[0], "FAR_Y": far[1], "FAR_Z": far[2],
            "ZM_X": walls[0], "ZM_Y": walls[1], "ZM_Z": walls[2],
            "PHX": phases[0], "PHY": phases[1], "PHZ": phases[2],
            "shape": tuple(int(value) for value in grid.shape),
            "ghost_destinations": ghost_destinations(near, far),
        }
    finally:
        driver.close()


def build_driver(cp, cell, boundaries, mirror_specs, seed: int, electric: bool,
                 options: Optional[Dict[str, Any]] = None):
    """One folded PML driver, seeded identically for every route."""
    driver = build_grid(cell, boundaries, mirror_specs, options)
    shape = driver.shape
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = np.ascontiguousarray(
        (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32))
    driver.set_epsilon(cp.asarray(epsilon))
    if electric:
        # An ELECTRIC source is admitted: the driver injects it in the D/E seam,
        # not this one. Carrying one is what stops the source clause from being
        # tested only in its refusing direction.
        #
        # OFF THE MIRROR PLANE ON EVERY FOLDED AXIS. A point Ez at the origin sits
        # ON the plane, and on an ODD fold that is refused with a reason rather
        # than stepped: Ez has parity -1 there and no extent along the folded axis,
        # so the parity condition constrains the profile against itself and admits
        # only zero (from_meep.py:2686-2698). Off the plane neither parity rule
        # applies. A quarter of the owned half-width puts it well inside the stored
        # half and clear of the absorber, which a folded axis carries on its HIGH
        # face only.
        center = [0.0, 0.0, 0.0]
        for axis_name, _phase in mirror_specs:
            axis = "XYZ".index(axis_name.upper())
            center[axis] = 0.25 * (cell[axis] / 2.0)
        driver.add_source({"component": "Ez", "frequency": 0.31,
                           "center": tuple(center), "width": 0.4})
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, cp.asarray(np.ascontiguousarray(
            rng.uniform(-0.25, 0.25, size=shape).astype(np.float32))))
    for name in ("f_w_Hx", "f_w_Hy", "f_w_Hz"):
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

    Counting is PER ROUTE. A single global counter would mix the reference
    driver's legitimate array-path calls with a fallback in the fused route, which
    is precisely the event this instrument exists to see.
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

    curl = symmetry.plan_folded_pml_curl(driver.fields, driver.pml, "step_B")
    fill = symmetry.plan_mirror_ghost_fill(driver.fields, "B")
    constitutive = symmetry.plan_folded_constitutive(driver.fields, driver.pml, "H")
    missing = [name for name, plan in
               (("folded curl", curl), ("mirror ghost fill", fill),
                ("folded constitutive", constitutive)) if plan is None]
    if missing:
        raise AssertionError(
            f"the separate oracle is incomplete: {missing} refused this case, so "
            f"this leg could not compare the fused launch against the products it "
            f"replaces")
    # `zero_metal_B` and `fill_folded_far_ghosts_B` stay on the ARRAY PATH here:
    # no Triton product owns the wall clear, and the far fill is a no-op on a
    # folded METALLIC axis. Both are counted, so a difference in which of them ran
    # is a counter event rather than an invisible correction.
    return Route({"step_B": curl, "fill_symmetry_bc_B": fill,
                  "update_H": constitutive})


def fused_route(plan):
    return Route({
        "step_B": plan,
        "fill_symmetry_bc_B": _Absorbed("fill_symmetry_bc_B", plan),
        "zero_metal_B": _Absorbed("zero_metal_B", plan),
        "fill_folded_far_ghosts_B": _Absorbed("fill_folded_far_ghosts_B", plan),
        "update_H": _Absorbed("update_H", plan),
    })


# ---------------------------------------------------------------------------
# The three-route leg
# ---------------------------------------------------------------------------

def run_leg(cp, name: str, cell, boundaries, mirrors, steps: int, product,
            mutant: Any = None, install_fused: bool = True,
            freeze_magnetic: bool = False, electric: bool = True,
            options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Three routes in lockstep; stop at the FIRST byte divergence."""
    import meep_gpu.driver as driver_module  # noqa: PLC0415

    reference = build_driver(cp, cell, boundaries, mirrors, SEED, electric, options)
    separate = build_driver(cp, cell, boundaries, mirrors, SEED, electric, options)
    fused = build_driver(cp, cell, boundaries, mirrors, SEED, electric, options)
    undo: Callable[[], Any] = lambda: None
    counter: Dict[str, int] = {}
    kernel = CountingKernel(
        mutant if mutant is not None
        else product.folded_fused_curl_constitutive_B_kernel())
    row: Dict[str, Any] = {
        "leg": name, "device": True, "steps_budget": steps,
        "shape": list(reference.shape), "boundaries": boundaries,
        "mirrors": [list(entry) for entry in mirrors],
        "electric_source": bool(electric),
        "fused_substituted": bool(install_fused),
        "magnetic_frozen": bool(freeze_magnetic),
        "first_divergence": None, "control_divergence": None,
    }
    try:
        plan = product.plan_folded_fused_magnetic_pair(
            fused.fields, fused.pml, tuple(fused._sources), num_warps=1,
            kernel=kernel)
        if plan is None:
            verdict = product.folded_fused_magnetic_pair_coverage(
                fused.fields, fused.pml, tuple(fused._sources))
            raise AssertionError(f"the product refused the case: {verdict.reasons}")
        row["plan"] = repr(plan)
        row["plan_replaces"] = list(plan.replaces)
        # COMPOSITION DEPTH, recorded per leg rather than argued in a comment. The
        # kernel emits a `_carry_ghost` call per destination one source lane owns,
        # and the three triple-composite calls are emitted only where this reads 7.
        row["near"] = [bool(value) for value in plan.near]
        row["far"] = [bool(value) for value in plan.far]
        row["reflect_rows"] = list(plan.reflect)
        row["ghost_destinations_per_source_lane"] = list(
            ghost_destinations(plan.near, plan.far))
        row["triple_composite_executes"] = bool(
            max(ghost_destinations(plan.near, plan.far)) == 7)
        route = fused_route(plan) if install_fused else Route({})
        separate_plans = separate_route(separate)
        row["separate_products"] = {key: type(value).__name__
                                    for key, value in separate_plans.plans.items()}

        roles = {id(reference.fields): "array", id(separate.fields): "separate",
                 id(fused.fields): "fused"}
        routes = [(separate.fields, separate_plans), (fused.fields, route)]
        if freeze_magnetic:
            # ARMED: every route's magnetic seam is inert. All three then agree
            # trivially and only the moved-state census can refuse it.
            frozen = {name: _Absorbed(name, None) for name in SEAM_PASSES}
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
               require_moved: bool = True,
               require_ghost_destinations: Optional[int] = None
               ) -> Tuple[bool, List[str]]:
    """The leg's pass conditions, stated rather than implied."""
    failures: List[str] = []
    if require_ghost_destinations is not None:
        observed = row.get("ghost_destinations_per_source_lane") or []
        if max(observed or [0]) != int(require_ghost_destinations):
            failures.append(
                f"this leg was declared to reach composition depth "
                f"{require_ghost_destinations} — the ghost cells one source lane "
                f"owns for its most-composed component — and reached "
                f"{max(observed or [0])} ({observed}). The kernel emits its "
                f"blocks per depth, so a leg short of the declared one leaves "
                f"those blocks unexecuted while the row reads as a pass")
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
    """The kernel AND the device function it calls, as one mutable text.

    ``_carry_ghost`` is included as of 2026-08-20 for two reasons, both of them
    failures a mutation harness cannot see from its own result. A name the mutant
    module does not define makes it fail to IMPORT, which is recorded as UNARMED
    rather than as a caught defect; and a device function left OUT of the mutated
    text is a piece of the product no mutation can reach at all — the seven
    constitutive statements every ghost cell runs would be unmeasured while the
    table reported a full set of catches.
    """
    # `inspect.getsource` INCLUDES THE DECORATOR LINE. Measured on 2026-08-20 by
    # adding one: `@triton.jit` twice makes the second decorator receive a
    # JITFunction, which `inspect.getsourcelines` refuses with a TypeError at
    # import — recorded as an unarmed mutation rather than as an error in the row.
    return (textwrap.dedent(inspect.getsource(product._carry_ghost.fn))
            + "\n\n"
            + textwrap.dedent(
                inspect.getsource(product.folded_fused_curl_constitutive_B.fn)))


def compile_mutant(source: str, kernel_name: str) -> Any:
    """Compile a renamed mutant. The rename is what keeps the JIT cache honest.

    THE CONSTEXPR CODES COME FROM ``symmetry``'S OWN CONSTANTS, not from literals.
    They were literals until 2026-08-20, when ``MIRROR_PERIODIC`` arrived with the
    far carry and every mutation compiled against a header that did not define it:
    seven rows came back as a ``NameError`` at Triton compile time, which the run
    records as unarmed. A renumbering there would have been the same failure with
    no error at all — the mutant would have had a different boundary vocabulary
    than the kernel it stands for.
    """
    from meep_gpu.triton_kernels.symmetry import (  # noqa: PLC0415
        CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC,
    )

    header = ("import triton\nimport triton.language as tl\n"
              f"PERIODIC = tl.constexpr({CODE_PERIODIC})\n"
              f"MIRROR_METALLIC = tl.constexpr({CODE_MIRROR_METALLIC})\n"
              f"MIRROR_PERIODIC = tl.constexpr({CODE_MIRROR_PERIODIC})\n\n")
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="_mutated_folded_pair.py", delete=False, encoding="utf-8")
    handle.write(header + source.replace("folded_fused_curl_constitutive_B",
                                         kernel_name))
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_folded_pair_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, kernel_name)


def mutation_table() -> Tuple[Tuple[str, str, str, Callable[[str], Tuple[str, int]]], ...]:
    """(id, why it is armed, expectation, rewrite). The artifact records what was MEASURED."""

    def m1_destination_coefficient_is_the_sources(source: str) -> Tuple[str, int]:
        """The B half's own hazard: reuse the SOURCE lane's constitutive pair.

        PREDICTED NULL ON THE REAL CASES and that is measured, not assumed: the
        mirror plane is the folded axis's low face and no absorber reaches it, so
        ``kps[0] == kps[2]`` in 0/64 separated configurations (NO-DEVICE LEG 2).
        The synthetic bare-array leg is where this becomes visible."""
        hits = 0
        for stem, index in (("kp0", "i"), ("km0", "i"), ("kp1", "j"),
                            ("km1", "j"), ("kp2", "k"), ("km2", "k")):
            needle = f"tl.load({stem} + origin"
            hits += source.count(needle)
            source = source.replace(needle, f"tl.load({stem} + {index}")
        return source, hits

    def m2_parity_dropped(source: str) -> Tuple[str, int]:
        """The mirror parity thrown away: the ghost becomes a copy, not an image.

        Invisible on an EVEN plane (phase +1) by construction, which is why the
        case list carries an odd fold."""
        hits = 0
        for phase, register in (("PHX", "v0"), ("PHY", "v1"), ("PHZ", "v2")):
            needle = f"{phase} * {register}"
            hits += source.count(needle)
            source = source.replace(needle, register)
        return source, hits

    def m3_near_fill_dropped(source: str) -> Tuple[str, int]:
        """The fill not carried: update_H consumes a pre-fill B on one component.

        Both halves of the carry are undone together — the ownership masks go back
        to ``live`` and the fill lanes go dead — so the mutant is exactly
        ``step_B`` + ``zero_metal_B`` + ``update_H`` with the mirror fill missing,
        rather than a half-carried state no configuration corresponds to."""
        hits = 0
        for own, coordinate in (("own0", "i"), ("own1", "j"), ("own2", "k")):
            needle = f"{own} = {own} & ({coordinate} != 0)"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} = {own}")
        for lane, coordinate in (("near_i", "i"), ("near_j", "j"),
                                 ("near_k", "k")):
            needle = f"{lane} = live & ({coordinate} == 2)"
            hits += source.count(needle)
            source = source.replace(needle, f"{lane} = live & (nx < 0)")
        return source, hits

    def m4_destination_lane_still_stores(source: str) -> Tuple[str, int]:
        """PREDICTED NULL, and the reason is WEAKER than this docstring used to say.

        The rewrite is the ownership restructure undone on component 0: the lanes
        ``own0`` excludes store ``B`` as well.

        CORRECTED 2026-08-21. The docstring here read "the unowned lane is i = 0
        only (``own0 = live & (i != 0)``), and the ghost fill later in the same
        kernel writes cell 0 from the source lane at stored cell 2", and offered
        that overwrite as the reason for the null. That describes the PRE-FAR-CARRY
        kernel and it does not describe THIS mutation's scored grid. ``own0`` is
        built as ``live``, then narrowed by ``if NEAR_X`` and by ``if FAR_Y`` /
        ``if FAR_Z`` — and this mutation is scored on ``odd_y_fold_metallic``,
        which folds Y ALONE and terminates it METALLIC. There NEAR_X, FAR_Y and
        FAR_Z are all false, so ``own0`` IS ``live``, ``mask=own0`` and
        ``mask=live`` are the same mask, and the rewrite is BYTE-INERT BY
        CONSTRUCTION rather than by any overwrite argument. The 2026-08-20 device
        row — armed, 1 rewrite hit, PTX moved, UNCAUGHT — is what an inert rewrite
        looks like, and calling it a measurement about the ghost fill was reading
        more out of it than it holds.

        WHAT ACTUALLY EXERCISES THE POST-CARRY OWNERSHIP is
        ``m17_carry_masks_not_anded_with_ownership``, which strips ``own?`` from
        the CARRY masks — the half that has a wrong VALUE rather than a duplicate
        write, and the half the 2026-08-20 device run found a real defect in. This
        one is kept, at its own case, for the store-side half.

        WHY IT IS NOT MOVED TO AN X-FOLDED GRID, where ``own0`` really is
        ``live & (i != 0)``: there the duplicate store is a RACE — two program
        instances writing one word with no ordering between them — so its measured
        outcome is a property of the scheduler, not of the kernel. WHAT THE MASK IS
        FOR is stated three lines above it in the module: race-freedom, not the
        final value ("a lane that merely discarded the value would still have READ
        a word another lane writes"). A byte comparison against the array path is
        the wrong instrument for that, and a null with its reason attached is
        honest where a coin-flip verdict would not be.
        """
        needle = "tl.store(f0 + idx, v0, mask=own0)"
        return source.replace(needle, "tl.store(f0 + idx, v0, mask=live)"), \
            source.count(needle)

    def m5_source_index_off_by_one(source: str) -> Tuple[str, int]:
        """The ghost imaged from stored cell 1 rather than 2 (MEEP's io = -2).

        BOTH LINES MOVE TOGETHER, and that is the correction this mutation needed.
        Rewriting only ``fill_x`` moved the SOURCE LANE to i = 1 while leaving
        ``dst_x = idx - 2 * nyz``, which addresses stored cell -1: out of bounds.
        Measured 2026-08-20 as armed and UNCAUGHT, which is what an out-of-bounds
        store looks like when it lands outside every compared array -- a memory
        fault reported as an inert defect. Moving the destination with the source
        keeps the write IN BOUNDS at cell 0 and makes the leg measure a WRONG
        ANSWER, which is the same rule the sibling gates state for their table
        mutations.
        """
        hits = 0
        for needle, replacement in (
                ("near_i = live & (i == 2)", "near_i = live & (i == 1)"),
                ("dn_x = -2 * nyz", "dn_x = -1 * nyz")):
            hits += source.count(needle)
            source = source.replace(needle, replacement)
        return source, hits

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
             "src0 = v0",
             "tl.store(w0 + idx, src0, mask=own0)"],
            ["src0 = v0",
             "tl.store(w0 + idx, src0, mask=own0)",
             "prev0 = tl.load(w0 + idx, mask=own0, other=0.0)"])

    def m8_zero_metal_dropped(source: str) -> Tuple[str, int]:
        """The wall clear not carried: MEEP's step_boundaries(B_stuff), undone.

        Visible only on a run whose walls are real — every case here declares
        metallic boundaries on the unfolded axes, which is where ``ZM`` fires."""
        return _rewrite_block(
            source,
            ["if ZM_X:", "v0 = tl.where(at_x, 0.0, v0)"],
            # NO ``if`` IN THE REPLACEMENT. ``_rewrite_block`` indents every
            # replacement line to the indent of the block's FIRST line, so a
            # compound statement in position 0 produces an empty suite and the
            # mutated module fails to IMPORT -- which this harness records as an
            # unarmed mutation, not as a caught defect. Dropping the guard along
            # with the clear is also the truer defect: what is being armed is "the
            # wall clear was not carried at all". The assignments are pure
            # identity, with no arithmetic, so a signed zero survives them.
            ["v0 = v0"])

    def m10_far_fill_dropped(source: str) -> Tuple[str, int]:
        """The FAR fill not carried — the state this family shipped in until now.

        Both halves are undone together, as m3 does for the near fill: the far
        ownership masks go back to what they were and the far source lanes go
        dead, so the mutant is exactly the pre-2026-08-20 kernel run on a grid it
        used to refuse, rather than a half-carried state no configuration
        corresponds to. ``update_H`` then consumes an unimaged top plane on two of
        the three components of every folded PERIODIC axis."""
        hits = 0
        for own, coordinate, extent in (
                ("own0", "j", "ny"), ("own0", "k", "nz"),
                ("own1", "i", "nx"), ("own1", "k", "nz"),
                ("own2", "i", "nx"), ("own2", "j", "ny")):
            needle = f"{own} = {own} & ({coordinate} != {extent} - 1)"
            hits += source.count(needle)
            source = source.replace(needle, f"{own} = {own}")
        for lane, coordinate, row in (("far_i", "i", "rx"), ("far_j", "j", "ry"),
                                      ("far_k", "k", "rz")):
            needle = f"{lane} = live & ({coordinate} == {row})"
            hits += source.count(needle)
            source = source.replace(needle, f"{lane} = live & (nx < 0)")
        return source, hits

    def m11_far_parity_is_the_near_rule(source: str) -> Tuple[str, int]:
        """``mirror_parity`` is ``-phase`` on a shift-1 component, not ``+phase``.

        The far fill only ever touches shift-1 components and the near fill only
        shift-0 ones (fields.mirror_parity :117 is ``phase * (1 - 2 * iyee)``), so
        applying the near fill's ``+phase`` to a far ghost is the single most
        plausible slip in this carry. Each needle carries its LEADING MINUS, which
        is what keeps it off the composite weights: ``-PHZ * PHY * v2`` contains
        ``PHY * v2`` but not ``-PHY * v2``."""
        hits = 0
        for needle in ("-PHY * v0", "-PHZ * v0", "-PHX * v1", "-PHZ * v1",
                       "-PHX * v2", "-PHY * v2"):
            hits += source.count(needle)
            source = source.replace(needle, needle[1:])
        return source, hits

    def m12_far_corner_takes_one_parity(source: str) -> Tuple[str, int]:
        """The doubly-imaged corner weighted by ONE plane instead of the product.

        A cell at the top of TWO folded periodic planes carries
        ``mirror_parity(m, a) * mirror_parity(m, b)`` — the composition the array
        path produces by applying its axes in turn. This is the corner alone: the
        two single far planes and the triple keep their own weights, so a catch is
        the corner and nothing else."""
        needle = "idx + df_x + df_y, PHX * PHY * v2"
        return source.replace(needle, "idx + df_x + df_y, PHX * v2"), \
            source.count(needle)

    def m13_far_image_row_is_the_window_top(source: str) -> Tuple[str, int]:
        """The trap ``_far_reflect_rows``' own docstring names in capitals.

        ``n_full - stored + 2`` is ``stored - 2`` at an even full count and
        ``stored - 3`` at an odd one, so the fixed ``n - 2`` reflects about the
        window top instead of about the second mirror. This bakes the even answer
        (``top - reflect == 1``). It is a NO-OP on an even-count grid, which is
        why the mutation leg's case is the ODD one and why CASES carries both."""
        hits = 0
        for axis, extent, row in (("x", "nx", "rx"), ("y", "ny", "ry"),
                                  ("z", "nz", "rz")):
            needle = f"df_{axis} = ({extent} - 1 - {row})"
            hits += source.count(needle)
            source = source.replace(needle, f"df_{axis} = (1)")
        return source, hits

    def m14_near_far_composite_dropped(source: str) -> Tuple[str, int]:
        """The cell at stored 0 on the near axis AND the top plane on a far one.

        A carry that wrote each fill's plane independently and stopped would leave
        this cell holding the far image of an UNFILLED row. It exists only because
        the fills compose, which is the whole finding the ownership rule rests on."""
        needle = "kp_d0, km_d0, own0 & near_i & far_j)"
        return source.replace(
            needle, "kp_d0, km_d0, own0 & near_i & far_j & (nx < 0))"), \
            source.count(needle)

    def m15_far_corner_dropped(source: str) -> Tuple[str, int]:
        """The far/far corner not written at all."""
        needle = "kp_2, km_2, own2 & far_i & far_j)"
        return source.replace(
            needle, "kp_2, km_2, own2 & far_i & far_j & (nx < 0))"), \
            source.count(needle)

    def m16_top_plane_mask_dropped(source: str) -> Tuple[str, int]:
        """``_mask_non_owned_cells``' shift-1 arm, which the fold's top plane needs.

        Emitted only on a folded PERIODIC axis — on a metallic fold the top plane
        IS ``big_corner``, owned and stepped — so this block arrived with the far
        carry and is armed with it. Dropping it steps a cell the fill then images
        over, but ``fu`` keeps the unmasked recurrence and diverges."""
        hits = 0
        for target, flag in (("curl0", "last_y"), ("curl0", "last_z"),
                             ("curl1", "last_x"), ("curl1", "last_z"),
                             ("curl2", "last_x"), ("curl2", "last_y")):
            needle = f"{target} = tl.where({flag}, 0.0, {target})"
            hits += source.count(needle)
            source = source.replace(needle, f"{target} = {target}")
        return source, hits

    def m17_carry_masks_not_anded_with_ownership(source: str) -> Tuple[str, int]:
        """THE DEFECT THE 2026-08-20 DEVICE RUN FOUND, re-planted.

        That run's first cut of the far carry did not AND the carry masks with the
        component's ownership mask. With two fills a lane can be the SOURCE of one
        and the DESTINATION of the other — on a grid folding x and y, the lane at
        ``(i = rx, j = 0)`` is the far source for ``By`` on x and is itself the near
        fill's destination on y — and it would then write a ghost from a ``v`` built
        on an ``f1`` load its own mask zeroed, racing the lane that legitimately
        owns the composite cell. It cost one word of ``Hx`` and one of ``Hy``, 1-2
        ULP, on the two-folded-axis rows alone (``run2.log``).

        A DEFECT A GATE ONCE FOUND AND NO MUTATION RE-PLANTS is a fix nothing
        guards: the ``own? &`` could be deleted tomorrow and every row here would
        stay green. This is that guard, and it is what makes the ownership half of
        the carry mutation-scored rather than merely agreed-with.

        THE REWRITE IS THE FIX UNDONE AND NOTHING ELSE. Only the twenty-one CARRY
        masks lose their ``own?`` — the loads, the stores and the ownership
        derivation itself are untouched, so the mutant is exactly the kernel as it
        stood before the fix rather than a differently broken one. Each needle
        carries its CLOSING PAREN, which is what keeps ``own0 & far_j)`` off
        ``own0 & far_j & far_k)``.

        SCORED ON THE THREE-AXIS ROW, not on the two-axis one it was found on. The
        rewrite makes two lanes write one word, and which write lands last is the
        scheduler's business; on the two-fold 2-D grid that is two words in the
        whole run, and on three folded periodic axes it is tens per component. The
        catch is the same defect either way and the larger set is what keeps the
        verdict from being a coin flip.
        """
        hits = 0
        for component, near, (first, second) in (
                (0, "near_i", ("far_j", "far_k")),
                (1, "near_j", ("far_i", "far_k")),
                (2, "near_k", ("far_i", "far_j"))):
            for mask in (first, second, f"{first} & {second}", near,
                         f"{near} & {first}", f"{near} & {second}",
                         f"{near} & {first} & {second}"):
                needle = f"own{component} & {mask})"
                hits += source.count(needle)
                source = source.replace(needle, f"{mask})")
        return source, hits

    def m18_triple_composite_dropped(source: str) -> Tuple[str, int]:
        """The deepest ghost not written at all — the block no leg used to execute.

        A cell at stored 0 on the near axis AND at ``last`` on BOTH far axes is a
        destination of all three fills composed. Its owning lane is the only one
        that can write it (``own?`` excludes it from every other route), so a carry
        that stopped at the pairs leaves it holding the previous step's value
        forever while the array path images it every step.

        This is the ``m15_far_corner_dropped`` construction one composition deeper,
        and it needs three folded PERIODIC axes to exist at all — which is why it
        arrives with ``xyz_periodic_odd_3d`` and could not have been armed before.
        """
        hits = 0
        for mask in ("own0 & near_i & far_j & far_k",
                     "own1 & near_j & far_i & far_k",
                     "own2 & near_k & far_i & far_j"):
            needle = f"{mask})"
            hits += source.count(needle)
            source = source.replace(needle, f"{mask} & (nx < 0))")
        return source, hits

    def m19_triple_parity_takes_two_planes(source: str) -> Tuple[str, int]:
        """The triple weighted by two of its three planes' parities.

        The composed weight is the PRODUCT over every plane the destination sits
        past, and the triple is the only cell where that product has three factors.
        Dropping one is the plausible slip — it is what a carry written by
        extending the pair blocks by hand would do — and it is invisible on any
        grid folding fewer than three axes.

        NOT EVERY COMPONENT FLIPS ON THE SCORED CASE, and that is measured rather
        than glossed. ``xyz_periodic_odd_3d`` declares phases (x, y, z) =
        (-1, +1, -1); this rewrite drops the LAST factor of each product, which is
        ``PHZ`` (-1) for components 0 and 1 and ``PHY`` (+1) for component 2. So
        the Bx and By triples change sign and the Bz triple does not. Any uniform
        rule has that property at these phases — exactly one of the three is +1 —
        and picking a different factor per component to keep all three observable
        would be tailoring the defect to the fixture. Two of three flipping is a
        catch; one of three not flipping is a fact about the phases.
        """
        hits = 0
        for needle, replacement in (
                ("PHX * PHY * PHZ * v0,", "PHX * PHY * v0,"),
                ("PHY * PHX * PHZ * v1,", "PHY * PHX * v1,"),
                ("PHZ * PHX * PHY * v2,", "PHZ * PHX * v2,")):
            hits += source.count(needle)
            source = source.replace(needle, replacement)
        return source, hits

    def m20_triple_destination_drops_the_far_hop(source: str) -> Tuple[str, int]:
        """The triple written to the near/far PAIR's cell instead of its own.

        The address, rather than the weight or the mask: ``idx + dn_x + df_y +
        df_z`` becomes ``idx + dn_x + df_y``. Two things go wrong at once and both
        are byte-visible. The triple cell is never written by anyone — no other
        lane owns it — so it holds the previous step's value; and the pair cell is
        written twice by the SAME lane, so the triple's weight lands there last and
        overwrites the pair's.

        DETERMINISTIC, unlike ``m4``: both writes come from one program instance,
        where Triton orders them, so this measures the kernel and not the
        scheduler.
        """
        hits = 0
        for needle, replacement in (
                ("idx + dn_x + df_y + df_z,", "idx + dn_x + df_y,"),
                ("idx + dn_y + df_x + df_z,", "idx + dn_y + df_x,"),
                ("idx + dn_z + df_x + df_y,", "idx + dn_z + df_x,")):
            hits += source.count(needle)
            source = source.replace(needle, replacement)
        return source, hits

    def m9_commuted_multiply(source: str) -> Tuple[str, int]:
        """PREDICTED NULL: IEEE multiplication commutes; only the PTX may move."""
        needle = "kp_0 * src0"
        return source.replace(needle, "src0 * kp_0"), source.count(needle)

    return (
        ("m1_destination_coefficient_is_the_sources", "the B half's own hazard",
         "null_on_real_cases", m1_destination_coefficient_is_the_sources),
        ("m2_parity_dropped", "the mirror parity", "caught", m2_parity_dropped),
        ("m3_near_fill_dropped", "the fill not carried", "caught", m3_near_fill_dropped),
        # NULL, with its reason in the rewrite's own docstring: the ghost fill
        # later in the same kernel overwrites the only unowned lane, so the mask
        # is about race-freedom rather than about the final bytes.
        ("m4_destination_lane_still_stores", "the ownership restructure undone",
         "null_on_metallic_folds", m4_destination_lane_still_stores),
        # ON AN X-FOLDED CASE, and that is the whole finding for this mutation.
        # It rewrites the `if BCX == MIRROR_METALLIC:` fill, and the default leg
        # (CASES[1], odd_y_fold_metallic) folds Y — X is an ordinary metallic
        # WALL there, so BCX is METALLIC, the branch never executes, and the
        # mutation came back UNCAUGHT on 2026-08-20 while measuring nothing at
        # all. A mutation must be scored on a grid that carries the code its
        # lines test on; MUTATION_CASE below is where that is stated per entry.
        ("m5_source_index_off_by_one", "MEEP's io = -2", "caught",
         m5_source_index_off_by_one),
        ("m6_constitutive_association", "float32 association", "caught",
         m6_constitutive_association),
        ("m7_history_read_after_write", "the f_w ordering", "caught",
         m7_history_read_after_write),
        ("m8_zero_metal_dropped", "the wall clear's slot", "caught",
         m8_zero_metal_dropped),
        ("m9_commuted_multiply", "commuted multiply", "null", m9_commuted_multiply),
        # ---- the far carry, 2026-08-20. Every one is scored on a grid whose
        # ---- constexprs emit the lines it rewrites; MUTATION_CASE says which.
        ("m10_far_fill_dropped", "the far fill not carried", "caught",
         m10_far_fill_dropped),
        ("m11_far_parity_is_the_near_rule", "mirror_parity on a shift-1 component",
         "caught", m11_far_parity_is_the_near_rule),
        ("m12_far_corner_takes_one_parity", "the corner's parity PRODUCT",
         "caught", m12_far_corner_takes_one_parity),
        ("m13_far_image_row_is_the_window_top", "_far_reflect_rows' own trap",
         "caught", m13_far_image_row_is_the_window_top),
        ("m14_near_far_composite_dropped", "the fills compose", "caught",
         m14_near_far_composite_dropped),
        ("m15_far_corner_dropped", "the doubly-imaged corner", "caught",
         m15_far_corner_dropped),
        ("m16_top_plane_mask_dropped", "the shift-1 ownership arm", "caught",
         m16_top_plane_mask_dropped),
        # ---- the ownership fix and the triple composite, 2026-08-21. The first
        # ---- re-plants the only defect this family's device runs ever found in
        # ---- the KERNEL; the other three are the first planted defects the
        # ---- three-axis path is known to catch. All four are scored on
        # ---- CASES[10] (``xyz_periodic_odd_3d``), which is the only row where
        # ---- the blocks they rewrite are emitted.
        ("m17_carry_masks_not_anded_with_ownership",
         "the ownership AND the device run added", "caught",
         m17_carry_masks_not_anded_with_ownership),
        ("m18_triple_composite_dropped", "the deepest ghost not written",
         "caught", m18_triple_composite_dropped),
        ("m19_triple_parity_takes_two_planes", "the triple's parity PRODUCT",
         "caught", m19_triple_parity_takes_two_planes),
        ("m20_triple_destination_drops_the_far_hop", "the triple's address",
         "caught", m20_triple_destination_drops_the_far_hop),
    )


#: Mutation id -> the CASES index whose grid carries the branch it rewrites.
#: Anything not named here runs on the default leg.
#:
#: MEASURED 2026-08-20: every mutation ran on CASES[1] (odd_y_fold_metallic), and
#: m5 rewrites a fill guarded on `BCX == MIRROR_METALLIC` — X FOLDED, not merely X
#: metallic. On a Y-folded grid that branch is never entered, so the leg launched
#: a kernel whose mutated lines were dead and reported "uncaught". That is a
#: statement about the case table, not about the kernel, and it is exactly the
#: shape of evidence this gate exists to refuse.
#: THE FAR-CARRY ROWS ALL NAME CASES[8] (``two_folds_periodic_odd``), and for the
#: same reason m5 names CASES[2]: every line they rewrite is emitted only under
#: ``FAR_a``, and only a grid with TWO folded PERIODIC axes emits the composite and
#: the corner. The odd full count is what makes m13 observable — on an even one the
#: fixed ``n - 2`` IS the reflect row and the mutation is a no-op.
MUTATION_CASE: Dict[str, int] = {
    "m5_source_index_off_by_one": 2,   # even_x_fold_metallic: X is the FOLDED axis
    "m10_far_fill_dropped": 8,
    "m11_far_parity_is_the_near_rule": 8,
    # CASE 7, NOT 8, and the difference is a MEASURED one. On case 8 the phases
    # are (x = -1, y = +1) and the corner weight is PHX * PHY, so dropping PHY
    # multiplies by +1 and the mutation is bit-identical: armed, launched, and
    # UNCAUGHT on 2026-08-20 while measuring nothing. Case 7 declares (+1, -1),
    # where dropping the second factor flips the sign of every corner word.
    "m12_far_corner_takes_one_parity": 7,
    "m13_far_image_row_is_the_window_top": 8,
    "m14_near_far_composite_dropped": 8,
    "m15_far_corner_dropped": 8,
    "m16_top_plane_mask_dropped": 8,
    # CASE 10 (``xyz_periodic_odd_3d``) — THE ONLY ROW WHERE THESE BLOCKS EXIST.
    # The three triple-composite calls are emitted under
    # ``if NEAR_a and (FAR_b and FAR_c):``, which needs all three axes folded and
    # all three PERIODIC; on case 8 the constexprs are false and the lines are not
    # in the compiled kernel at all, so a mutation scored there would report a
    # verdict about the case table. ``m17`` names the same row for a different
    # reason, in its own docstring.
    "m17_carry_masks_not_anded_with_ownership": 10,
    "m18_triple_composite_dropped": 10,
    "m19_triple_parity_takes_two_planes": 10,
    "m20_triple_destination_drops_the_far_hop": 10,
}

#: The default leg for a mutation that does not name one.
DEFAULT_MUTATION_CASE = 1


def mutation_case_for(name: str) -> Tuple[int, Tuple[Any, ...]]:
    """The (index, case) a mutation is scored on, and it must really fold what it needs."""
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
        kernel_name = f"mutant_{index}_folded_fused_B"
        mutant = compile_mutant(mutated, kernel_name)
        case_index, case = mutation_case_for(name)
        row["case"] = case[0]
        row["case_index"] = case_index
        leg = run_leg(cp, f"mutation:{name}", case[1], case[2], case[3],
                      MUTATION_STEPS, product, mutant=mutant, options=case[5])
        row["caught"] = leg.get("first_divergence") is not None
        row["first_divergence"] = leg.get("first_divergence")
        row["launches"] = leg.get("fused_kernel_launches")
        # The depth the mutation was scored at, beside its verdict: a rewrite that
        # only edits the triple-composite blocks measures nothing below 7, and the
        # row has to say which it got rather than leaving it to MUTATION_CASE.
        row["ghost_destinations_per_source_lane"] = leg.get(
            "ghost_destinations_per_source_lane")
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

    # SAME PER-AXIS DECLARATION AS ``CASES``, and for the same reason: these cells
    # are 2-D, so a bare "metallic" would put a PEC on the invariant z axis, which
    # Grid refuses by name (grid.py:842-877). This leg is about the product's own
    # refusals; it must not die inside the Grid constructor before reaching one.
    _WALLS = {"x": "metallic", "y": "metallic", "z": "periodic"}
    rows: List[Dict[str, Any]] = []
    # `sources` may be a CALLABLE taking the built fields, and `needle=None` means
    # "this row must be ADMITTED" — both for the 2026-08-30 carry rows below.
    for name, cell, boundaries, mirrors, needle, sources in (
        ("magnetic_source_without_a_deposit_index", CASES[0][1], _WALLS, (("Y", 1),),
         "does not publish the index it writes", (_Magnetic(),)),
        ("magnetic_deposit_is_carried", CASES[0][1], _WALLS, (("Y", 1),),
         None, lambda fields: (_deposit(fields, "Hy"),)),
        ("undeclared_sources", CASES[0][1], _WALLS, (("Y", 1),),
         "was not declared", None),
        ("unfolded", (3.0, 3.0, 0.0), _WALLS, (), "no mirror plane", ()),
    ):
        driver = build_driver(cp, cell, boundaries, mirrors, SEED, electric=False)
        try:
            bound = sources(driver.fields) if callable(sources) else sources
            verdict = product.folded_fused_magnetic_pair_coverage(
                driver.fields, driver.pml, bound)
            plan = product.plan_folded_fused_magnetic_pair(
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

    # THE FLIP. `folded_periodic` used to be a REFUSAL row here, on the clause
    # "fill_folded_far_ghosts_B runs inside this seam and is not carried". The
    # kernel carries it as of 2026-08-20, so the same fixture now measures the
    # opposite — and the LIVENESS is asserted alongside, because a carry of a pass
    # that does not run would be decoration. The plan's reflect rows must be
    # stepping._far_reflect_rows' own answer and its far axes the Yee table's.
    driver = build_driver(cp, CASES[0][1], "periodic", (("Y", 1),), SEED,
                          electric=False)
    try:
        import meep_gpu.stepping as stepping  # noqa: PLC0415
        from meep_gpu.triton_kernels.symmetry import (  # noqa: PLC0415
            CODE_MIRROR_PERIODIC, folded_axis_kinds,
        )
        verdict = product.folded_fused_magnetic_pair_coverage(
            driver.fields, driver.pml, ())
        plan = product.plan_folded_fused_magnetic_pair(
            driver.fields, driver.pml, ())
        codes, _ = folded_axis_kinds(driver.fields.grid, driver.pml)
        expected_rows = tuple(
            -1 if row is None else int(row)
            for row in stepping._far_reflect_rows(driver.fields.grid))
        expected_far = tuple(int(code) == CODE_MIRROR_PERIODIC for code in codes)
        rows.append({
            "case": "folded_periodic_is_now_CARRIED", "device": True,
            "covered": bool(verdict.covered),
            "reasons": list(verdict.reasons),
            "plan_is_none": plan is None,
            "boundary_codes": [int(code) for code in codes],
            "far_axes": None if plan is None else list(plan.far),
            "far_axes_expected": list(expected_far),
            "near_axes": None if plan is None else list(plan.near),
            "reflect_rows": None if plan is None else list(plan.reflect),
            "reflect_rows_expected": list(expected_rows),
            "pass_is_in_replaces": "fill_folded_far_ghosts_B" in product.REPLACES,
            "replaces": list(product.REPLACES),
            "passed": bool(verdict.covered and plan is not None
                           and any(expected_far)
                           and tuple(plan.far) == expected_far
                           and tuple(plan.reflect) == expected_rows
                           and "fill_folded_far_ghosts_B" in product.REPLACES),
        })
    finally:
        driver.close()
        cp.get_default_memory_pool().free_all_blocks()
    log(f"  refusal folded_periodic_is_now_CARRIED: "
        f"covered={rows[-1]['covered']} passed={rows[-1]['passed']}")
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
        HERE, "results", "triton_folded_fused_magnetic_pair", "gate.json"))
    parser.add_argument("--no-device", action="store_true",
                        help="run only the legs that need neither CUDA nor Triton")
    parser.add_argument(
        "--subnormal-policy", default="keep",
        help="the float32 subnormal policy to DRIVE EVERY EXECUTOR TO before the "
             "first device compile. Default 'keep' — the policy every record in "
             "triton_kernels/fingerprints.json is cut under. A run that installs "
             "NOTHING is a MIXED configuration (CuPy appends -ftz=true "
             "unconditionally; Triton natively keeps) attributable to no policy "
             "at all, which is what the 2026-08-14 chain run measured.")
    args = parser.parse_args(argv)

    payload: Dict[str, Any] = {
        "gate": "triton_folded_fused_magnetic_pair",
        "product": "meep_gpu.triton_kernels.folded_fused_magnetic_pair",
        "kernel": "folded_fused_curl_constitutive_B",
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
    for leg in (transcription_leg, coefficient_reach_leg, design_sweep_leg,
                branch_reachability_leg, predicate_leg):
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
        # ``release`` IS THE KEY ``gate_provenance.read_verdict`` CONSULTS FIRST, and
        # it is set explicitly here because ``passed`` alone would stamp
        # ``released: True`` on an artifact that measured no bytes on any device.
        # A laptop leg is evidence about the design and the predicate; it is not a
        # release, and the record must not read as one.
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

    # BEFORE THE FIRST DEVICE COMPILE. strict=True: a process that asked to keep
    # and quietly did not is a process whose bytes mean nothing, and this refuses
    # at startup rather than writing an artifact whose policy field is a wish. It
    # is also what enforces the CUPY_CACHE_DIR token rule
    # (``subnormal_policy.cupy_cache_reasons``): CuPy's cache key is computed
    # ABOVE the seam the ``-ftz`` strip installs at, so a directory shared with a
    # flush run would serve flushed binaries under this record's name.
    subnormal_policy.install_subnormal_policy(args.subnormal_policy, cupy=cp,
                                              strict=True)
    payload["subnormal_policy"] = subnormal_policy.policy_stamp()
    log(f"subnormal policy installed: "
        f"{payload['subnormal_policy'].get('policy')!r} "
        f"(requested {args.subnormal_policy!r}, "
        f"CUPY_CACHE_DIR={os.environ.get('CUPY_CACHE_DIR')!r})")

    from meep_gpu.triton_kernels import folded_fused_magnetic_pair as product  # noqa: PLC0415

    backends.guard_kernel_compilation(cp)
    payload["environment"] = environment(cp)
    # Stamped BEFORE the first leg so the partial artifact an aborted run leaves
    # behind — which is the artifact a failure is read from — says what it is.
    payload["device_status"] = "IN PROGRESS (this artifact is partial)"
    payload["budgets"] = {
        "byte_cases": {case[0]: case[4] for case in CASES},
        "steps_per_mutation": MUTATION_STEPS,
        "total_steps": (sum(case[4] for case in CASES)
                        + MUTATION_STEPS * (len(mutation_table()) + 2)),
    }
    save(payload, args.out)
    log("\n=== device legs ===")
    for name, cell, boundaries, mirrors, steps, options in CASES:
        row = run_leg(cp, name, cell, boundaries, mirrors, steps, product,
                      options=options)
        row["options"] = dict(options)
        passed, failures = verdict_of(
            row, require_launches=steps,
            require_ghost_destinations=options.get("expect_ghost_destinations"))
        row["passed"], row["failures"] = passed, failures
        payload["device_legs"].append(row)
        save(payload, args.out)

    # THE FLOOR FOR THE PATH THIS ROUND EXISTS FOR. A table that carries the
    # three-axis case but never reaches depth 7 on any leg has measured the same
    # thing the previous cut did; this refuses the whole artifact rather than
    # letting the case list stand in for the coverage.
    deepest = max((max(row.get("ghost_destinations_per_source_lane") or [0])
                   for row in payload["device_legs"]), default=0)
    payload["composition_depth"] = {
        "deepest_ghost_destinations_per_source_lane": deepest,
        "legs_executing_the_triple_composite": sorted(
            row["leg"] for row in payload["device_legs"]
            if row.get("triple_composite_executes")),
        "why": ("the kernel's three `if NEAR_a and (FAR_b and FAR_c):` blocks are "
                "emitted only where one source lane owns seven ghost cells, which "
                "needs three folded PERIODIC axes"),
        "passed": deepest == 7,
    }
    save(payload, args.out)

    log("\n=== armed harness mutations ===")
    # The substitution removed: the bytes still agree (the array path is what ran)
    # and ONLY the launch counter can refuse it.
    row = run_leg(cp, "armed:no_substitution", CASES[0][1], CASES[0][2],
                  CASES[0][3], 3, product, install_fused=False)
    passed, _ = verdict_of(row, require_launches=3)
    row["armed"] = True
    row["passed"] = not passed and row.get("fused_kernel_launches") == 0
    row["why"] = ("the fused plan is built but never installed; bytes agree and the "
                  "counter is what must refuse the leg")
    payload["device_legs"].append(row)
    save(payload, args.out)

    row = run_leg(cp, "armed:frozen_magnetic_seam", CASES[0][1], CASES[0][2],
                  CASES[0][3], 3, product, freeze_magnetic=True)
    passed, _ = verdict_of(row)
    row["armed"] = True
    row["passed"] = not passed and bool(row.get("arrays_never_moved"))
    row["why"] = ("every route's magnetic seam is inert; all three agree trivially "
                  "and only the moved-state census can refuse it")
    payload["device_legs"].append(row)
    save(payload, args.out)

    log("\n=== armed kernel mutations ===")
    pristine = kernel_ptx(product.folded_fused_curl_constitutive_B)
    payload["mutations"] = run_mutations(cp, product, pristine)
    save(payload, args.out)

    log("\n=== refusals ===")
    payload["refusals"] = run_refusals(cp, product)
    save(payload, args.out)

    device_ok = all(row.get("passed") for row in payload["device_legs"])
    refusal_ok = all(row.get("passed") for row in payload["refusals"])
    # A DECLARED NULL MUST BE CONFIRMED, not merely permitted. The old rule was
    # ``caught if expectation == "caught" else True``, under which any expectation
    # other than "caught" passed unconditionally -- so moving a stubbornly
    # uncaught mutation to a null spelling would silence it rather than explain
    # it. A null that IS caught means the reasoning behind the null is wrong, and
    # that has to fail too. Every mutation must also have been ARMED: a rewrite
    # that hit nothing measured nothing, whatever it then reported.
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
                     "mutations whose measured outcome did not match their "
                     "declared expectation, or that were never armed: "
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
